from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ai_weather_rainfall.deep.dataset import RainCalibTensorDataset, load_unet_config
from ai_weather_rainfall.deep.unet import SmallUNet
from ai_weather_rainfall.raincalib import (
    categorical_scores_frame,
    choose_probability_threshold,
    continuous_scores_frame,
)


@dataclass
class SplitPredictions:
    raw: np.ndarray
    corrected: np.ndarray
    target: np.ndarray
    raw_fields: np.ndarray
    corrected_fields: np.ndarray
    target_fields: np.ndarray
    mask_fields: np.ndarray
    event_probability_values: list[np.ndarray]
    event_probability_fields: np.ndarray | None


def main() -> None:
    args = _parse_args()
    cfg = load_unet_config(args.config)
    checkpoint_path = args.checkpoint or (cfg.models_dir / f"{cfg.run_name}.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    event_thresholds = tuple(float(v) for v in checkpoint.get("event_thresholds_mm", cfg.event_thresholds_mm))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet(
        in_channels=int(checkpoint["in_channels"]),
        out_channels=int(checkpoint.get("out_channels", 1 + len(event_thresholds))),
        base_channels=int(checkpoint["base_channels"]),
        depth=int(checkpoint["depth"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = _predict_split(cfg, model, device, args.split, event_thresholds)
    probability_thresholds = _tune_event_thresholds(cfg, model, device, args.tune_split, event_thresholds)

    cfg.predictions_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = cfg.predictions_dir / f"{cfg.run_name}_{args.split}_predictions.npz"
    _save_predictions(predictions_path, cfg.run_name, predictions, event_thresholds)

    continuous = _continuous_metrics(predictions.raw, predictions.corrected, predictions.target, cfg)
    categorical = _categorical_metrics(predictions, cfg, event_thresholds, probability_thresholds)
    continuous_path = cfg.metrics_dir / f"{cfg.run_name}_{args.split}_continuous_metrics.csv"
    categorical_path = cfg.metrics_dir / f"{cfg.run_name}_{args.split}_categorical_metrics.csv"
    thresholds_path = cfg.metrics_dir / f"{cfg.run_name}_{args.split}_event_probability_thresholds.csv"
    continuous.to_csv(continuous_path, index=False)
    categorical.to_csv(categorical_path, index=False)
    probability_thresholds.to_csv(thresholds_path, index=False)
    report_path = cfg.reports_dir / f"{cfg.run_name}_report.md"
    report_path.write_text(
        _report_text(
            run_name=cfg.run_name,
            split=args.split,
            tune_split=args.tune_split,
            samples=len(predictions.target),
            checkpoint_path=checkpoint_path,
            predictions_path=predictions_path,
            continuous=continuous,
            categorical=categorical,
            probability_thresholds=probability_thresholds,
        ),
        encoding="utf-8",
    )
    print(f"predictions={predictions_path}")
    print(f"continuous_metrics={continuous_path}")
    print(f"categorical_metrics={categorical_path}")
    print(f"event_probability_thresholds={thresholds_path}")
    print(f"report={report_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RainCalib U-Net against Raw GFS and v1 baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/raincalib_unet.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tune-split", default="val", choices=["train", "val", "test"])
    return parser.parse_args()


@torch.no_grad()
def _predict_split(cfg, model, device: torch.device, split: str, event_thresholds: tuple[float, ...]) -> SplitPredictions:
    dataset = RainCalibTensorDataset(
        cfg.tensor_store,
        split,
        rain_scale_mm=cfg.rain_scale_mm,
        gfs_clip_mm=cfg.gfs_clip_mm,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    raw_values: list[np.ndarray] = []
    corrected_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    raw_fields: list[np.ndarray] = []
    corrected_fields: list[np.ndarray] = []
    target_fields: list[np.ndarray] = []
    mask_fields: list[np.ndarray] = []
    event_probability_values: list[list[np.ndarray]] = [[] for _ in event_thresholds]
    event_probability_fields: list[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)
        output = model(x).detach().cpu()
        residual_prediction = output[:, :1].numpy()
        probabilities = torch.sigmoid(output[:, 1:]).numpy() if output.shape[1] > 1 else None
        gfs = batch["gfs"].numpy()
        target = batch["target"].numpy()
        mask = batch["mask"].numpy()
        corrected = np.clip(gfs + residual_prediction * cfg.rain_scale_mm, 0.0, None)
        valid = mask.astype(bool)
        raw_values.append(gfs[valid])
        corrected_values.append(corrected[valid])
        target_values.append(target[valid])
        raw_fields.append(gfs[:, 0])
        corrected_fields.append(corrected[:, 0])
        target_fields.append(target[:, 0])
        mask_fields.append(mask[:, 0])
        if probabilities is not None:
            event_probability_fields.append(probabilities.astype("float32"))
            for channel in range(min(probabilities.shape[1], len(event_thresholds))):
                event_probability_values[channel].append(probabilities[:, channel : channel + 1][valid])

    probability_fields = np.concatenate(event_probability_fields).astype("float32") if event_probability_fields else None
    return SplitPredictions(
        raw=np.concatenate(raw_values),
        corrected=np.concatenate(corrected_values),
        target=np.concatenate(target_values),
        raw_fields=np.concatenate(raw_fields).astype("float32"),
        corrected_fields=np.concatenate(corrected_fields).astype("float32"),
        target_fields=np.concatenate(target_fields).astype("float32"),
        mask_fields=np.concatenate(mask_fields).astype("float32"),
        event_probability_values=[np.concatenate(values) if values else np.array([], dtype="float32") for values in event_probability_values],
        event_probability_fields=probability_fields,
    )


def _save_predictions(
    path: Path,
    run_name: str,
    predictions: SplitPredictions,
    event_thresholds: tuple[float, ...],
) -> None:
    payload = {
        "raw_gfs": predictions.raw_fields,
        "corrected_rain": predictions.corrected_fields,
        run_name: predictions.corrected_fields,
        "target": predictions.target_fields,
        "mask": predictions.mask_fields,
        "event_thresholds_mm": np.array(event_thresholds, dtype="float32"),
    }
    if predictions.event_probability_fields is not None:
        payload["event_probabilities"] = predictions.event_probability_fields
    np.savez_compressed(path, **payload)


def _tune_event_thresholds(
    cfg,
    model,
    device: torch.device,
    split: str,
    event_thresholds: tuple[float, ...],
) -> pd.DataFrame:
    if not event_thresholds:
        return pd.DataFrame(columns=["threshold_mm", "probability_threshold", "metric", "csi", "pod", "far", "ets"])
    predictions = _predict_split(cfg, model, device, split, event_thresholds)
    rows: list[dict[str, float | str]] = []
    for index, threshold in enumerate(event_thresholds):
        probabilities = predictions.event_probability_values[index]
        if probabilities.size == 0:
            continue
        probability_threshold, scores = choose_probability_threshold(
            probabilities,
            predictions.target >= threshold,
            metric=cfg.event_probability_metric,
        )
        rows.append(
            {
                "threshold_mm": threshold,
                "probability_threshold": probability_threshold,
                "metric": cfg.event_probability_metric,
                **scores,
            }
        )
    return pd.DataFrame(rows)


def _continuous_metrics(raw: np.ndarray, corrected: np.ndarray, target: np.ndarray, cfg) -> pd.DataFrame:
    rows = [
        {"model": "raw_gfs", **continuous_scores_frame(raw, target), "samples": len(target)},
        {"model": cfg.run_name, **continuous_scores_frame(corrected, target), "samples": len(target)},
    ]
    if cfg.raincalib_v1_predictions.exists():
        v1 = pd.read_parquet(cfg.raincalib_v1_predictions, columns=["gfs_rain", "raincalib_rain", "target_rain"])
        rows.insert(
            1,
            {
                "model": "raincalib_v1_sklearn",
                **continuous_scores_frame(v1["raincalib_rain"].to_numpy(), v1["target_rain"].to_numpy()),
                "samples": len(v1),
            },
        )
    return pd.DataFrame(rows)


def _categorical_metrics(
    predictions: SplitPredictions,
    cfg,
    event_thresholds: tuple[float, ...],
    probability_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    v1 = (
        pd.read_parquet(cfg.raincalib_v1_predictions, columns=["raincalib_rain", "target_rain"])
        if cfg.raincalib_v1_predictions.exists()
        else None
    )
    tuned_by_threshold = {
        float(row["threshold_mm"]): float(row["probability_threshold"])
        for _, row in probability_thresholds.iterrows()
    }
    for index, threshold in enumerate(event_thresholds or (50.0, 100.0)):
        observed = predictions.target >= threshold
        rows.append(
            {
                "threshold_mm": threshold,
                "model": "raw_gfs",
                "decision": "rain_threshold",
                "probability_threshold": np.nan,
                **categorical_scores_frame(predictions.raw >= threshold, observed),
            }
        )
        if v1 is not None:
            rows.append(
                {
                    "threshold_mm": threshold,
                    "model": "raincalib_v1_sklearn",
                    "decision": "rain_threshold",
                    "probability_threshold": np.nan,
                    **categorical_scores_frame(
                        v1["raincalib_rain"].to_numpy() >= threshold,
                        v1["target_rain"].to_numpy() >= threshold,
                    ),
                }
            )
        rows.append(
            {
                "threshold_mm": threshold,
                "model": cfg.run_name,
                "decision": "rain_threshold",
                "probability_threshold": np.nan,
                **categorical_scores_frame(predictions.corrected >= threshold, observed),
            }
        )
        if index < len(predictions.event_probability_values) and predictions.event_probability_values[index].size:
            probability_threshold = tuned_by_threshold.get(float(threshold), 0.5)
            rows.append(
                {
                    "threshold_mm": threshold,
                    "model": f"{cfg.run_name}_event_head",
                    "decision": "event_probability",
                    "probability_threshold": probability_threshold,
                    **categorical_scores_frame(
                        predictions.event_probability_values[index] >= probability_threshold,
                        observed,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _report_text(
    *,
    run_name: str,
    split: str,
    tune_split: str,
    samples: int,
    checkpoint_path: Path,
    predictions_path: Path,
    continuous: pd.DataFrame,
    categorical: pd.DataFrame,
    probability_thresholds: pd.DataFrame,
) -> str:
    lines = [
        f"# {run_name} Report\n\n",
        "This report evaluates a small U-Net residual model for regional GFS rainfall correction.\n\n",
        "**Important:** This is a research benchmark, not an official warning service.\n\n",
        f"- Evaluation split: {split}\n",
        f"- Event probability tune split: {tune_split}\n",
        f"- Valid grid-cell samples: {samples}\n",
        f"- Checkpoint: `{checkpoint_path}`\n",
        f"- Predictions: `{predictions_path}`\n\n",
        "## Continuous Skill\n\n",
        "| Model | MAE | RMSE | Bias | Pearson r | Samples |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for _, row in continuous.iterrows():
        lines.append(
            f"| {row['model']} | {row['mae']:.3f} | {row['rmse']:.3f} | "
            f"{row['bias']:.3f} | {row['pearson_r']:.3f} | {int(row['samples'])} |\n"
        )
    lines.extend(
        [
            "\n## Event Probability Thresholds\n\n",
            "| Event | Probability threshold | Tune metric | Tune CSI | Tune POD | Tune FAR | Tune ETS |\n",
            "|---:|---:|---|---:|---:|---:|---:|\n",
        ]
    )
    if probability_thresholds.empty:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |\n")
    else:
        for _, row in probability_thresholds.iterrows():
            lines.append(
                f"| {row['threshold_mm']:g} mm | {row['probability_threshold']:.2f} | {row['metric']} | "
                f"{row['csi']:.3f} | {row['pod']:.3f} | {row['far']:.3f} | {row['ets']:.3f} |\n"
            )
    lines.extend(
        [
            "\n## Event Skill\n\n",
            "| Threshold | Model | Decision | Hits | Misses | False alarms | CSI | POD | FAR | ETS |\n",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for _, row in categorical.sort_values(["threshold_mm", "model", "decision"]).iterrows():
        lines.append(
            f"| {row['threshold_mm']:g} mm | {row['model']} | {row['decision']} | {int(row['hits'])} | "
            f"{int(row['misses'])} | {int(row['false_alarms'])} | {row['csi']:.3f} | "
            f"{row['pod']:.3f} | {row['far']:.3f} | {row['ets']:.3f} |\n"
        )
    return "".join(lines)


if __name__ == "__main__":
    main()
