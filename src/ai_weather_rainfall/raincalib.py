from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


BASE_FEATURE_COLUMNS = [
    "gfs_rain",
    "lead_hour",
    "cycle",
    "latitude",
    "longitude",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


@dataclass(frozen=True)
class RainCalibConfig:
    project_name: str
    output_root: Path
    benchmark_config: Path
    feature_store: Path
    feature_stride: int
    train_negative_sample_fraction: float
    seed: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    event_thresholds: list[float]
    max_train_rows: int
    random_state: int
    optimize_event_metric: str
    models_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    reports_dir: Path


def load_raincalib_config(path: Path) -> RainCalibConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RainCalibConfig(
        project_name=str(data["project"]["name"]),
        output_root=Path(data["project"]["output_root"]),
        benchmark_config=Path(data["data"]["benchmark_config"]),
        feature_store=Path(data["data"]["feature_store"]),
        feature_stride=int(data["data"].get("feature_stride", 2)),
        train_negative_sample_fraction=float(data["data"].get("train_negative_sample_fraction", 0.15)),
        seed=int(data["data"].get("seed", 42)),
        train_start=pd.Timestamp(data["splits"]["train_start"]),
        train_end=pd.Timestamp(data["splits"]["train_end"]),
        val_start=pd.Timestamp(data["splits"]["val_start"]),
        val_end=pd.Timestamp(data["splits"]["val_end"]),
        test_start=pd.Timestamp(data["splits"]["test_start"]),
        test_end=pd.Timestamp(data["splits"]["test_end"]),
        event_thresholds=[float(v) for v in data["targets"]["event_thresholds_mm_24h"]],
        max_train_rows=int(data["training"].get("max_train_rows", 600000)),
        random_state=int(data["training"].get("random_state", 42)),
        optimize_event_metric=str(data["training"].get("optimize_event_metric", "ets")),
        models_dir=Path(data["outputs"]["models"]),
        predictions_dir=Path(data["outputs"]["predictions"]),
        metrics_dir=Path(data["outputs"]["metrics"]),
        reports_dir=Path(data["outputs"]["reports"]),
    )


def split_for_date(value: Any, cfg: RainCalibConfig) -> str:
    day = pd.Timestamp(value)
    if cfg.train_start <= day <= cfg.train_end:
        return "train"
    if cfg.val_start <= day <= cfg.val_end:
        return "val"
    if cfg.test_start <= day <= cfg.test_end:
        return "test"
    return "excluded"


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    observed = pd.to_datetime(result["observed_date"])
    day_of_year = observed.dt.dayofyear.astype(float)
    month = observed.dt.month.astype(float)
    result["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    result["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    result["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    result["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)
    return result


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in BASE_FEATURE_COLUMNS if name in frame.columns]


def continuous_scores_frame(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    valid = np.isfinite(predicted) & np.isfinite(observed)
    if not valid.any():
        return {"mae": math.nan, "rmse": math.nan, "bias": math.nan, "pearson_r": math.nan}
    diff = predicted[valid] - observed[valid]
    pearson = math.nan
    if valid.sum() >= 2:
        pearson = float(np.corrcoef(predicted[valid], observed[valid])[0, 1])
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "pearson_r": pearson,
    }


def categorical_scores_frame(pred_event: np.ndarray, obs_event: np.ndarray) -> dict[str, float]:
    pred_event = np.asarray(pred_event, dtype=bool)
    obs_event = np.asarray(obs_event, dtype=bool)
    hits = int(np.sum(pred_event & obs_event))
    misses = int(np.sum(~pred_event & obs_event))
    false_alarms = int(np.sum(pred_event & ~obs_event))
    correct_negatives = int(np.sum(~pred_event & ~obs_event))
    total = hits + misses + false_alarms + correct_negatives
    random_hits = ((hits + misses) * (hits + false_alarms)) / total if total else math.nan
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "pod": _ratio(hits, hits + misses),
        "far": _ratio(false_alarms, hits + false_alarms),
        "csi": _ratio(hits, hits + misses + false_alarms),
        "frequency_bias": _ratio(hits + false_alarms, hits + misses),
        "ets": _ratio(hits - random_hits, hits + misses + false_alarms - random_hits),
    }


def choose_probability_threshold(
    probabilities: np.ndarray,
    observed_event: np.ndarray,
    *,
    metric: str,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_scores = categorical_scores_frame(probabilities >= best_threshold, observed_event)
    best_value = best_scores.get(metric, math.nan)
    for threshold in np.linspace(0.05, 0.95, 19):
        scores = categorical_scores_frame(probabilities >= threshold, observed_event)
        value = scores.get(metric, math.nan)
        if math.isnan(best_value) or (not math.isnan(value) and value > best_value):
            best_threshold = float(threshold)
            best_scores = scores
            best_value = value
    return best_threshold, best_scores


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan
