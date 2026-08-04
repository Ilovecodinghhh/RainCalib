from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ai_weather_rainfall.deep.dataset import (
    RainCalibTensorDataset,
    event_targets,
    load_unet_config,
    masked_bce_with_logits,
    masked_smooth_l1,
    rain_intensity_weights,
)
from ai_weather_rainfall.deep.unet import SmallUNet
from ai_weather_rainfall.experimental import (
    IsotonicProbabilityCalibrator,
    deterministic_event_scores,
    extended_continuous_scores,
    fractions_skill_score,
    fractions_skill_score_events,
    load_experiment_config,
    paired_date_bootstrap,
    probability_scores,
)
from ai_weather_rainfall.raincalib import categorical_scores_frame, choose_probability_threshold


def main() -> None:
    args = _parse_args()
    experiment_cfg = load_experiment_config(args.experiment_config)
    unet_cfg = load_unet_config(args.unet_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for the multi-seed experiment matrix")

    metadata = np.load(experiment_cfg.tensor_store, allow_pickle=False)
    observed_dates = pd.to_datetime(metadata["observed_date"].astype(str))
    selected_folds = [fold for fold in experiment_cfg.folds if not args.folds or fold.name in args.folds]
    selected_seeds = tuple(args.seeds or experiment_cfg.seeds)
    output_root = experiment_cfg.output_root / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    metric_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    fss_frames: list[pd.DataFrame] = []

    for fold in selected_folds:
        indexes = {
            "train": np.flatnonzero((observed_dates >= fold.train.start) & (observed_dates <= fold.train.end)),
            "validation": np.flatnonzero(
                (observed_dates >= fold.validation.start) & (observed_dates <= fold.validation.end)
            ),
            "test": np.flatnonzero((observed_dates >= fold.test.start) & (observed_dates <= fold.test.end)),
        }
        if any(len(value) == 0 for value in indexes.values()):
            raise ValueError(f"fold={fold.name} contains an empty tensor split")
        print(
            f"fold={fold.name} train_fields={len(indexes['train'])} "
            f"validation_fields={len(indexes['validation'])} test_fields={len(indexes['test'])}"
        )
        for seed in selected_seeds:
            run_dir = output_root / fold.name / f"seed_{seed}"
            paths = {
                "metrics": run_dir / "metrics.csv",
                "bootstrap": run_dir / "paired_date_bootstrap.csv",
                "fss": run_dir / "fss.csv",
            }
            if not args.force and all(path.exists() for path in paths.values()):
                print(f"resume fold={fold.name} seed={seed}")
                metric_frames.append(pd.read_csv(paths["metrics"]))
                bootstrap_frames.append(pd.read_csv(paths["bootstrap"]))
                fss_frames.append(pd.read_csv(paths["fss"]))
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"train fold={fold.name} seed={seed}")
            results = _run_one(
                experiment_cfg,
                unet_cfg,
                indexes,
                fold.name,
                seed,
                args.epochs or unet_cfg.epochs,
                run_dir,
                device,
            )
            results["metrics"].to_csv(paths["metrics"], index=False)
            results["bootstrap"].to_csv(paths["bootstrap"], index=False)
            results["fss"].to_csv(paths["fss"], index=False)
            metric_frames.append(results["metrics"])
            bootstrap_frames.append(results["bootstrap"])
            fss_frames.append(results["fss"])
            torch.cuda.empty_cache()

    metrics = pd.concat(metric_frames, ignore_index=True)
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    fss = pd.concat(fss_frames, ignore_index=True)
    metrics.to_csv(output_root / "metrics.csv", index=False)
    bootstrap.to_csv(output_root / "paired_date_bootstrap.csv", index=False)
    fss.to_csv(output_root / "fss.csv", index=False)
    summary = (
        metrics.groupby(["fold", "category", "threshold_mm", "metric"], dropna=False)["value"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.to_csv(output_root / "seed_summary.csv", index=False)
    print(f"metrics={output_root / 'metrics.csv'}")
    print(f"summary={output_root / 'seed_summary.csv'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-origin RainCalib U-Net experiments.")
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/raincalib_experiments.yaml"),
    )
    parser.add_argument("--unet-config", type=Path, default=Path("configs/raincalib_unet.yaml"))
    parser.add_argument("--folds", nargs="*")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-subdir", default="unet")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _run_one(experiment_cfg, cfg, indexes, fold_name, seed, epochs, run_dir, device):
    _set_seed(seed)
    datasets = {
        name: RainCalibTensorDataset(
            experiment_cfg.tensor_store,
            indexes=selection,
            rain_scale_mm=cfg.rain_scale_mm,
            gfs_clip_mm=cfg.gfs_clip_mm,
        )
        for name, selection in indexes.items()
    }
    train_loader = _train_loader(cfg, datasets["train"], seed, device)
    validation_loader = _loader(cfg, datasets["validation"], device)
    model = SmallUNet(
        in_channels=8,
        out_channels=1 + len(cfg.event_thresholds_mm),
        base_channels=cfg.base_channels,
        depth=cfg.depth,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.mixed_precision and device.type == "cuda")
    pos_weight = _event_pos_weight(cfg, datasets["train"], device)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, scaler, device, cfg, pos_weight)
        validation_loss = _validation_loss(model, validation_loader, device, cfg, pos_weight)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        print(
            f"fold={fold_name} seed={seed} epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Training produced no model state")
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": best_state,
            "fold": fold_name,
            "seed": seed,
            "epochs": epochs,
            "best_validation_loss": best_loss,
            "base_channels": cfg.base_channels,
            "depth": cfg.depth,
            "dropout": cfg.dropout,
            "event_thresholds_mm": cfg.event_thresholds_mm,
        },
        run_dir / "model.pt",
    )

    validation = _predict(model, datasets["validation"], cfg, device)
    test = _predict(model, datasets["test"], cfg, device)
    train_target = datasets["train"].target[datasets["train"].indexes]
    train_mask = datasets["train"].mask[datasets["train"].indexes] > 0
    metrics, calibrated_probability, threshold_metadata = _evaluate_predictions(
        validation,
        test,
        train_target,
        train_mask,
        cfg.event_thresholds_mm,
        fold_name,
        seed,
    )
    valid = test["mask"].reshape(-1) > 0
    field_dates = datasets["test"].observed_date[datasets["test"].indexes]
    pixel_dates = np.broadcast_to(
        field_dates[:, None, None],
        test["target"].shape,
    ).reshape(-1)
    bootstrap = paired_date_bootstrap(
        test["prediction"].reshape(-1)[valid],
        test["gfs"].reshape(-1)[valid],
        test["target"].reshape(-1)[valid],
        pixel_dates[valid],
        resamples=experiment_cfg.bootstrap_resamples,
        seed=seed,
    )
    bootstrap.insert(0, "seed", seed)
    bootstrap.insert(0, "fold", fold_name)
    fss_rows: list[dict[str, object]] = []
    for threshold in experiment_cfg.event_thresholds_mm:
        channel = list(experiment_cfg.event_thresholds_mm).index(threshold)
        raw_decision_threshold = threshold_metadata[f"{threshold:g}"]["raw_probability_threshold"]
        calibrated_decision_threshold = threshold_metadata[f"{threshold:g}"][
            "calibrated_probability_threshold"
        ]
        for neighborhood in experiment_cfg.fss_neighborhoods:
            fss_rows.append(
                {
                    "fold": fold_name,
                    "seed": seed,
                    "source": "rain_prediction",
                    "threshold_mm": threshold,
                    "neighborhood": neighborhood,
                    "fss": fractions_skill_score(
                        test["prediction"],
                        test["target"],
                        test["mask"] > 0,
                        threshold_mm=threshold,
                        neighborhood=neighborhood,
                    ),
                }
            )
            fss_rows.append(
                {
                    "fold": fold_name,
                    "seed": seed,
                    "source": "event_head_decision",
                    "threshold_mm": threshold,
                    "neighborhood": neighborhood,
                    "fss": fractions_skill_score_events(
                        test["probability"][:, channel] >= raw_decision_threshold,
                        test["target"] >= threshold,
                        test["mask"] > 0,
                        neighborhood=neighborhood,
                    ),
                }
            )
            fss_rows.append(
                {
                    "fold": fold_name,
                    "seed": seed,
                    "source": "calibrated_decision",
                    "threshold_mm": threshold,
                    "neighborhood": neighborhood,
                    "fss": fractions_skill_score_events(
                        calibrated_probability[:, channel] >= calibrated_decision_threshold,
                        test["target"] >= threshold,
                        test["mask"] > 0,
                        neighborhood=neighborhood,
                    ),
                }
            )
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        indexes=test["indexes"],
        prediction_rain=test["prediction"].astype("float32"),
        event_probability=test["probability"].astype("float32"),
        event_probability_calibrated=calibrated_probability.astype("float32"),
    )
    (run_dir / "event_thresholds.json").write_text(
        json.dumps(threshold_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "metrics": pd.DataFrame(metrics),
        "bootstrap": bootstrap,
        "fss": pd.DataFrame(fss_rows),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(cfg, dataset, device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )


def _train_loader(cfg, dataset, seed, device) -> DataLoader:
    sampler = None
    shuffle = True
    if cfg.balanced_sampler_enabled:
        weights = _field_sample_weights(dataset, cfg.balanced_sampler_thresholds_mm, cfg.balanced_sampler_weights)
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=cfg.balanced_sampler_samples_per_epoch or len(dataset),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )


def _field_sample_weights(dataset, thresholds, values) -> np.ndarray:
    target = dataset.target[dataset.indexes]
    mask = dataset.mask[dataset.indexes] > 0
    field_max = np.where(mask, target, -np.inf).max(axis=(1, 2))
    weights = np.ones(len(dataset), dtype="float64")
    for threshold, value in zip(thresholds, values):
        weights = np.where(field_max >= threshold, np.maximum(weights, value), weights)
    return weights


def _event_pos_weight(cfg, dataset, device):
    target = dataset.target[dataset.indexes]
    mask = dataset.mask[dataset.indexes] > 0
    values = []
    for threshold in cfg.event_thresholds_mm:
        positive = np.sum((target >= threshold) & mask)
        negative = np.sum((target < threshold) & mask)
        values.append(float(min(negative / max(positive, 1), cfg.event_pos_weight_cap)))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _multitask_loss(prediction, residual_target, target_rain, mask, cfg, pos_weight):
    weights = rain_intensity_weights(
        target_rain,
        mask,
        cfg.rain_weight_thresholds_mm,
        cfg.rain_weight_values,
    )
    residual_loss = masked_smooth_l1(prediction[:, :1], residual_target, mask, weights)
    binary_targets = event_targets(target_rain, cfg.event_thresholds_mm)
    event_loss = masked_bce_with_logits(prediction[:, 1:], binary_targets, mask, pos_weight)
    return residual_loss + cfg.event_loss_weight * event_loss


def _run_epoch(model, loader, optimizer, scaler, device, cfg, pos_weight) -> float:
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=cfg.mixed_precision):
            loss = _multitask_loss(model(x), y, target, mask, cfg, pos_weight)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def _validation_loss(model, loader, device, cfg, pos_weight) -> float:
    model.eval()
    losses = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=cfg.mixed_precision):
            loss = _multitask_loss(model(x), y, target, mask, cfg, pos_weight)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def _predict(model, dataset, cfg, device) -> dict[str, np.ndarray]:
    model.eval()
    fields = {name: [] for name in ("indexes", "prediction", "probability", "gfs", "target", "mask")}
    for batch in _loader(cfg, dataset, device):
        x = batch["x"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=cfg.mixed_precision):
            output = model(x)
        gfs = batch["gfs"].numpy()[:, 0]
        residual = output[:, 0].float().cpu().numpy() * cfg.rain_scale_mm
        fields["indexes"].append(batch["index"].numpy())
        fields["prediction"].append(np.clip(gfs + residual, 0.0, None))
        fields["probability"].append(torch.sigmoid(output[:, 1:]).float().cpu().numpy())
        fields["gfs"].append(gfs)
        fields["target"].append(batch["target"].numpy()[:, 0])
        fields["mask"].append(batch["mask"].numpy()[:, 0])
    return {name: np.concatenate(values) for name, values in fields.items()}


def _evaluate_predictions(validation, test, train_target, train_mask, thresholds, fold, seed):
    rows: list[dict[str, object]] = []

    def append(category, threshold, scores):
        for metric, value in scores.items():
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "category": category,
                    "threshold_mm": threshold,
                    "metric": metric,
                    "value": value,
                }
            )

    validation_valid = validation["mask"].reshape(-1) > 0
    test_valid = test["mask"].reshape(-1) > 0
    validation_target = validation["target"].reshape(-1)[validation_valid]
    test_target = test["target"].reshape(-1)[test_valid]
    test_prediction = test["prediction"].reshape(-1)[test_valid]
    append("continuous", None, extended_continuous_scores(test_prediction, test_target))
    calibrated_fields = np.zeros_like(test["probability"], dtype="float32")
    threshold_metadata: dict[str, dict[str, float]] = {}
    for channel, threshold in enumerate(thresholds):
        validation_probability = validation["probability"][:, channel].reshape(-1)[validation_valid]
        test_probability = test["probability"][:, channel].reshape(-1)[test_valid]
        validation_event = validation_target >= threshold
        test_event = test_target >= threshold
        reference_rate = float(np.sum((train_target >= threshold) & train_mask) / np.sum(train_mask))
        raw_threshold, _ = choose_probability_threshold(validation_probability, validation_event, metric="ets")
        calibrator = IsotonicProbabilityCalibrator().fit(validation_probability, validation_event)
        calibrated_validation = calibrator.predict(validation_probability)
        calibrated_test = calibrator.predict(test_probability)
        calibrated_threshold, _ = choose_probability_threshold(
            calibrated_validation,
            validation_event,
            metric="ets",
        )
        channel_calibrated = calibrator.predict(test["probability"][:, channel].reshape(-1)).reshape(
            test["probability"].shape[0], *test["probability"].shape[-2:]
        )
        calibrated_fields[:, channel] = channel_calibrated
        threshold_metadata[f"{threshold:g}"] = {
            "raw_probability_threshold": raw_threshold,
            "calibrated_probability_threshold": calibrated_threshold,
        }
        append("rain_threshold", threshold, deterministic_event_scores(test_prediction, test_target, threshold))
        append(
            "event_head_decision",
            threshold,
            categorical_scores_frame(test_probability >= raw_threshold, test_event),
        )
        append(
            "event_head_probability",
            threshold,
            probability_scores(test_probability, test_event, reference_rate=reference_rate),
        )
        append(
            "calibrated_decision",
            threshold,
            categorical_scores_frame(calibrated_test >= calibrated_threshold, test_event),
        )
        append(
            "calibrated_probability",
            threshold,
            probability_scores(calibrated_test, test_event, reference_rate=reference_rate),
        )
    return rows, calibrated_fields, threshold_metadata


if __name__ == "__main__":
    main()
