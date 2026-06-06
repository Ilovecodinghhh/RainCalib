from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from ai_weather_rainfall.deep.dataset import (
    RainCalibTensorDataset,
    event_targets,
    load_unet_config,
    masked_bce_with_logits,
    masked_smooth_l1,
    rain_intensity_weights,
)
from ai_weather_rainfall.deep.unet import SmallUNet


def main() -> None:
    args = _parse_args()
    cfg = load_unet_config(args.config)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type != "cuda":
        print("warning: CUDA is not available; training will run on CPU")

    train_dataset = RainCalibTensorDataset(
        cfg.tensor_store,
        "train",
        rain_scale_mm=cfg.rain_scale_mm,
        gfs_clip_mm=cfg.gfs_clip_mm,
    )
    val_dataset = RainCalibTensorDataset(
        cfg.tensor_store,
        "val",
        rain_scale_mm=cfg.rain_scale_mm,
        gfs_clip_mm=cfg.gfs_clip_mm,
    )
    train_loader = _new_train_loader(cfg, train_dataset, device)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    _run_overfit_smoke(cfg, train_dataset, device)
    model = _new_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = _new_scaler(device, cfg.mixed_precision)
    pos_weight = _event_pos_weight(cfg, train_dataset, device)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    history_path = cfg.metrics_dir / f"{cfg.run_name}_training_history.csv"
    checkpoint_path = cfg.models_dir / f"{cfg.run_name}.pt"

    best_val = float("inf")
    history_rows: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, scaler, device, cfg.mixed_precision, cfg, pos_weight)
        val_loss = _evaluate_loss(model, val_loader, device, cfg.mixed_precision, cfg, pos_weight)
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "run_name": cfg.run_name,
                    "model_state_dict": model.state_dict(),
                    "in_channels": 8,
                    "out_channels": 1 + len(cfg.event_thresholds_mm),
                    "base_channels": cfg.base_channels,
                    "depth": cfg.depth,
                    "dropout": cfg.dropout,
                    "rain_scale_mm": cfg.rain_scale_mm,
                    "gfs_clip_mm": cfg.gfs_clip_mm,
                    "event_thresholds_mm": list(cfg.event_thresholds_mm),
                    "event_pos_weight": pos_weight.detach().cpu().tolist() if pos_weight is not None else [],
                    "event_loss_weight": cfg.event_loss_weight,
                    "rain_weight_thresholds_mm": list(cfg.rain_weight_thresholds_mm),
                    "rain_weight_values": list(cfg.rain_weight_values),
                    "balanced_sampler_enabled": cfg.balanced_sampler_enabled,
                    "best_val_loss": best_val,
                },
                checkpoint_path,
            )
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history_rows)
    print(f"checkpoint={checkpoint_path}")
    print(f"history={history_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small RainCalib U-Net residual model.")
    parser.add_argument("--config", type=Path, default=Path("configs/raincalib_unet.yaml"))
    return parser.parse_args()


def _new_model(cfg) -> SmallUNet:
    return SmallUNet(
        in_channels=8,
        out_channels=1 + len(cfg.event_thresholds_mm),
        base_channels=cfg.base_channels,
        depth=cfg.depth,
        dropout=cfg.dropout,
    )


def _new_scaler(device: torch.device, mixed_precision: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device.type, enabled=mixed_precision and device.type == "cuda")


def _new_train_loader(cfg, train_dataset: RainCalibTensorDataset, device: torch.device) -> DataLoader:
    sampler = None
    shuffle = True
    if cfg.balanced_sampler_enabled:
        weights = _field_sample_weights(
            train_dataset,
            cfg.balanced_sampler_thresholds_mm,
            cfg.balanced_sampler_weights,
        )
        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=cfg.balanced_sampler_samples_per_epoch or len(train_dataset),
            replacement=True,
            generator=generator,
        )
        shuffle = False
        print(
            "balanced_sampler="
            f"enabled samples_per_epoch={cfg.balanced_sampler_samples_per_epoch or len(train_dataset)} "
            f"min_weight={weights.min():.3f} max_weight={weights.max():.3f}"
        )
    return DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )


def _field_sample_weights(
    dataset: RainCalibTensorDataset,
    thresholds_mm: tuple[float, ...],
    values: tuple[float, ...],
) -> np.ndarray:
    if len(thresholds_mm) != len(values):
        raise ValueError("balanced sampler thresholds and weights must have the same length")
    field_target = dataset.target[dataset.indexes]
    field_mask = dataset.mask[dataset.indexes] > 0
    masked_target = np.where(field_mask, field_target, -np.inf)
    field_max = masked_target.max(axis=(1, 2))
    weights = np.ones(len(dataset), dtype="float64")
    for threshold, value in zip(thresholds_mm, values):
        weights = np.where(field_max >= float(threshold), np.maximum(weights, float(value)), weights)
    return weights


def _event_pos_weight(cfg, dataset: RainCalibTensorDataset, device: torch.device) -> torch.Tensor | None:
    if not cfg.event_thresholds_mm:
        return None
    target = dataset.target[dataset.indexes]
    mask = dataset.mask[dataset.indexes] > 0
    values: list[float] = []
    for threshold in cfg.event_thresholds_mm:
        positive = int(np.sum((target >= float(threshold)) & mask))
        negative = int(np.sum((target < float(threshold)) & mask))
        ratio = negative / max(positive, 1)
        values.append(float(min(ratio, cfg.event_pos_weight_cap)))
    pos_weight = torch.tensor(values, dtype=torch.float32, device=device)
    print(f"event_pos_weight={','.join(f'{value:.3f}' for value in values)}")
    return pos_weight


def _run_overfit_smoke(cfg, train_dataset: RainCalibTensorDataset, device: torch.device) -> None:
    count = min(len(train_dataset), cfg.overfit_batches * cfg.batch_size)
    subset = Subset(train_dataset, list(range(count)))
    loader = DataLoader(subset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    batches = list(loader)
    model = _new_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = _new_scaler(device, cfg.mixed_precision)
    pos_weight = _event_pos_weight(cfg, train_dataset, device)
    first_loss = None
    last_loss = None
    for step in range(1, cfg.overfit_steps + 1):
        batch = batches[(step - 1) % len(batches)]
        loss = _train_batch(model, batch, optimizer, scaler, device, cfg.mixed_precision, cfg, pos_weight)
        first_loss = loss if first_loss is None else first_loss
        last_loss = loss
    print(f"overfit_smoke first_loss={first_loss:.6f} last_loss={last_loss:.6f}")


def _run_epoch(model, loader, optimizer, scaler, device, mixed_precision: bool, cfg, pos_weight) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        loss = _train_batch(model, batch, optimizer, scaler, device, mixed_precision, cfg, pos_weight)
        total += loss
        count += 1
    return total / max(count, 1)


def _train_batch(model, batch, optimizer, scaler, device, mixed_precision: bool, cfg, pos_weight) -> float:
    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    target_rain = batch["target"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, enabled=mixed_precision and device.type == "cuda"):
        prediction = model(x)
        loss = _multitask_loss(prediction, y, target_rain, mask, cfg, pos_weight)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach().cpu())


@torch.no_grad()
def _evaluate_loss(model, loader, device, mixed_precision: bool, cfg, pos_weight) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        target_rain = batch["target"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=mixed_precision and device.type == "cuda"):
            prediction = model(x)
            loss = _multitask_loss(prediction, y, target_rain, mask, cfg, pos_weight)
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


def _multitask_loss(prediction, residual_target, target_rain, mask, cfg, pos_weight) -> torch.Tensor:
    residual_prediction = prediction[:, :1]
    event_logits = prediction[:, 1:]
    weights = rain_intensity_weights(
        target_rain,
        mask,
        cfg.rain_weight_thresholds_mm,
        cfg.rain_weight_values,
    )
    residual_loss = masked_smooth_l1(residual_prediction, residual_target, mask, weights)
    if event_logits.shape[1] == 0 or cfg.event_loss_weight <= 0:
        return residual_loss
    binary_targets = event_targets(target_rain, cfg.event_thresholds_mm)
    event_loss = masked_bce_with_logits(event_logits, binary_targets, mask, pos_weight)
    return residual_loss + cfg.event_loss_weight * event_loss


if __name__ == "__main__":
    main()
