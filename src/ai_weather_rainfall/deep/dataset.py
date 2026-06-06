from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml


@dataclass(frozen=True)
class RainCalibUNetConfig:
    run_name: str
    feature_store: Path
    tensor_store: Path
    raincalib_v1_predictions: Path
    rain_scale_mm: float
    gfs_clip_mm: float
    base_channels: int
    depth: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    mixed_precision: bool
    overfit_batches: int
    overfit_steps: int
    event_thresholds_mm: tuple[float, ...]
    rain_weight_thresholds_mm: tuple[float, ...]
    rain_weight_values: tuple[float, ...]
    event_loss_weight: float
    event_pos_weight_cap: float
    event_probability_metric: str
    balanced_sampler_enabled: bool
    balanced_sampler_thresholds_mm: tuple[float, ...]
    balanced_sampler_weights: tuple[float, ...]
    balanced_sampler_samples_per_epoch: int | None
    models_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    reports_dir: Path


def load_unet_config(path: Path) -> RainCalibUNetConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    loss = data.get("loss", {})
    targets = data.get("targets", {})
    sampler = data.get("training", {}).get("balanced_sampler", {})
    return RainCalibUNetConfig(
        run_name=str(data["model"].get("run_name", "raincalib_unet")),
        feature_store=Path(data["data"]["feature_store"]),
        tensor_store=Path(data["data"]["tensor_store"]),
        raincalib_v1_predictions=Path(data["data"]["raincalib_v1_predictions"]),
        rain_scale_mm=float(data["data"].get("rain_scale_mm", 50.0)),
        gfs_clip_mm=float(data["data"].get("gfs_clip_mm", 200.0)),
        base_channels=int(data["model"].get("base_channels", 16)),
        depth=int(data["model"].get("depth", 2)),
        dropout=float(data["model"].get("dropout", 0.0)),
        batch_size=int(data["training"].get("batch_size", 32)),
        epochs=int(data["training"].get("epochs", 5)),
        learning_rate=float(data["training"].get("learning_rate", 1e-3)),
        weight_decay=float(data["training"].get("weight_decay", 1e-4)),
        num_workers=int(data["training"].get("num_workers", 0)),
        seed=int(data["training"].get("seed", 42)),
        mixed_precision=bool(data["training"].get("mixed_precision", True)),
        overfit_batches=int(data["training"].get("overfit_batches", 4)),
        overfit_steps=int(data["training"].get("overfit_steps", 80)),
        event_thresholds_mm=tuple(float(v) for v in targets.get("event_thresholds_mm_24h", [])),
        rain_weight_thresholds_mm=tuple(float(v) for v in loss.get("rain_weight_thresholds_mm", [])),
        rain_weight_values=tuple(float(v) for v in loss.get("rain_weight_values", [])),
        event_loss_weight=float(loss.get("event_loss_weight", 0.0)),
        event_pos_weight_cap=float(loss.get("event_pos_weight_cap", 50.0)),
        event_probability_metric=str(loss.get("event_probability_metric", "csi")),
        balanced_sampler_enabled=bool(sampler.get("enabled", False)),
        balanced_sampler_thresholds_mm=tuple(float(v) for v in sampler.get("thresholds_mm", [])),
        balanced_sampler_weights=tuple(float(v) for v in sampler.get("weights", [])),
        balanced_sampler_samples_per_epoch=(
            int(sampler["samples_per_epoch"]) if sampler.get("samples_per_epoch") is not None else None
        ),
        models_dir=Path(data["outputs"]["models"]),
        predictions_dir=Path(data["outputs"]["predictions"]),
        metrics_dir=Path(data["outputs"]["metrics"]),
        reports_dir=Path(data["outputs"]["reports"]),
    )


class RainCalibTensorDataset(Dataset):
    def __init__(self, tensor_store: Path, split: str, *, rain_scale_mm: float, gfs_clip_mm: float) -> None:
        data = np.load(tensor_store, allow_pickle=False)
        split_values = data["split"].astype(str)
        indexes = np.flatnonzero(split_values == split)
        if len(indexes) == 0:
            raise ValueError(f"No tensor samples found for split={split}")
        self.indexes = indexes
        self.gfs = data["gfs"].astype("float32")
        self.target = data["target"].astype("float32")
        self.mask = data["mask"].astype("float32")
        self.lead_hour = data["lead_hour"].astype("float32")
        self.cycle = data["cycle"].astype("float32")
        self.month_sin = data["month_sin"].astype("float32")
        self.month_cos = data["month_cos"].astype("float32")
        self.day_of_year_sin = data["day_of_year_sin"].astype("float32")
        self.day_of_year_cos = data["day_of_year_cos"].astype("float32")
        self.run_date = data["run_date"].astype(str)
        self.observed_date = data["observed_date"].astype(str)
        self.rain_scale_mm = float(rain_scale_mm)
        self.gfs_clip_mm = float(gfs_clip_mm)

    def __len__(self) -> int:
        return int(len(self.indexes))

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indexes[item])
        gfs = self.gfs[index]
        target = self.target[index]
        mask = self.mask[index]
        residual = (target - gfs) / self.rain_scale_mm
        residual = np.where(mask > 0, residual, 0.0).astype("float32")
        h, w = gfs.shape
        channels = [
            np.log1p(np.clip(gfs, 0.0, self.gfs_clip_mm)) / np.log1p(self.gfs_clip_mm),
            np.full((h, w), self.lead_hour[index] / 360.0, dtype="float32"),
            np.full((h, w), self.cycle[index] / 12.0, dtype="float32"),
            np.full((h, w), self.month_sin[index], dtype="float32"),
            np.full((h, w), self.month_cos[index], dtype="float32"),
            np.full((h, w), self.day_of_year_sin[index], dtype="float32"),
            np.full((h, w), self.day_of_year_cos[index], dtype="float32"),
            mask.astype("float32"),
        ]
        x = np.stack(channels).astype("float32")
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(residual[None, :, :]),
            "mask": torch.from_numpy(mask[None, :, :].astype("float32")),
            "gfs": torch.from_numpy(gfs[None, :, :].astype("float32")),
            "target": torch.from_numpy(target[None, :, :].astype("float32")),
            "index": index,
        }


def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    effective_mask = mask if weights is None else mask * weights
    masked = loss * effective_mask
    denominator = effective_mask.sum().clamp_min(1.0)
    return masked.sum() / denominator


def rain_intensity_weights(
    target_rain: torch.Tensor,
    mask: torch.Tensor,
    thresholds_mm: tuple[float, ...],
    values: tuple[float, ...],
) -> torch.Tensor:
    if len(thresholds_mm) != len(values):
        raise ValueError("rain_weight_thresholds_mm and rain_weight_values must have the same length")
    weights = torch.ones_like(target_rain)
    for threshold, value in zip(thresholds_mm, values):
        weights = torch.where(target_rain >= float(threshold), torch.full_like(weights, float(value)), weights)
    return weights * mask


def event_targets(target_rain: torch.Tensor, thresholds_mm: tuple[float, ...]) -> torch.Tensor:
    if not thresholds_mm:
        return target_rain.new_zeros((target_rain.shape[0], 0, *target_rain.shape[-2:]))
    targets = [(target_rain >= float(threshold)).float() for threshold in thresholds_mm]
    return torch.cat(targets, dim=1)


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum()
    pos_weight_arg = None
    if pos_weight is not None:
        pos_weight_arg = pos_weight.view(1, -1, 1, 1).to(logits.device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight_arg,
    )
    expanded_mask = mask.expand_as(loss)
    return (loss * expanded_mask).sum() / expanded_mask.sum().clamp_min(1.0)
