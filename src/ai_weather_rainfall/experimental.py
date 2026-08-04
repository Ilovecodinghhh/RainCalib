from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml

from ai_weather_rainfall.raincalib import categorical_scores_frame, continuous_scores_frame


@dataclass(frozen=True)
class DateRange:
    start: pd.Timestamp
    end: pd.Timestamp

    def mask(self, values: pd.Series) -> pd.Series:
        dates = pd.to_datetime(values)
        return (dates >= self.start) & (dates <= self.end)


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train: DateRange
    validation: DateRange
    test: DateRange


@dataclass(frozen=True)
class ExperimentConfig:
    feature_store: Path
    tensor_store: Path
    folds: tuple[TemporalFold, ...]
    seeds: tuple[int, ...]
    event_thresholds_mm: tuple[float, ...]
    bootstrap_resamples: int
    fss_neighborhoods: tuple[int, ...]
    output_root: Path


def load_experiment_config(path: Path) -> ExperimentConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    folds = tuple(
        TemporalFold(
            name=str(item["name"]),
            train=DateRange(pd.Timestamp(item["train_start"]), pd.Timestamp(item["train_end"])),
            validation=DateRange(pd.Timestamp(item["val_start"]), pd.Timestamp(item["val_end"])),
            test=DateRange(pd.Timestamp(item["test_start"]), pd.Timestamp(item["test_end"])),
        )
        for item in data["folds"]
    )
    return ExperimentConfig(
        feature_store=Path(data["data"]["feature_store"]),
        tensor_store=Path(data["data"]["tensor_store"]),
        folds=folds,
        seeds=tuple(int(value) for value in data["seeds"]),
        event_thresholds_mm=tuple(float(value) for value in data["event_thresholds_mm"]),
        bootstrap_resamples=int(data["bootstrap_replicates"]),
        fss_neighborhoods=tuple(int(value) for value in data["fss_neighborhoods_grid_cells"]),
        output_root=Path(data["outputs"]["root"]),
    )


def extended_continuous_scores(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    scores = continuous_scores_frame(predicted, observed)
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    valid = np.isfinite(predicted) & np.isfinite(observed)
    scores["spearman_r"] = math.nan
    if valid.sum() >= 2:
        scores["spearman_r"] = float(spearmanr(predicted[valid], observed[valid]).statistic)
    return scores


def probability_scores(
    probability: np.ndarray,
    observed_event: np.ndarray,
    *,
    reference_rate: float,
) -> dict[str, float]:
    probability = np.asarray(probability, dtype=float)
    observed_event = np.asarray(observed_event, dtype=bool)
    valid = np.isfinite(probability)
    probability = np.clip(probability[valid], 0.0, 1.0)
    observed_event = observed_event[valid]
    if probability.size == 0:
        return {"pr_auc": math.nan, "roc_auc": math.nan, "brier": math.nan, "bss": math.nan}
    brier = float(np.mean((probability - observed_event.astype(float)) ** 2))
    reference_brier = float(np.mean((float(reference_rate) - observed_event.astype(float)) ** 2))
    pr_auc = math.nan
    roc_auc = math.nan
    if np.any(observed_event):
        pr_auc = float(average_precision_score(observed_event, probability))
    if np.unique(observed_event).size == 2:
        roc_auc = float(roc_auc_score(observed_event, probability))
    bss = 1.0 - brier / reference_brier if reference_brier > 0 else math.nan
    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "brier": brier, "bss": bss}


class MeanBiasCorrection:
    def fit(self, frame: pd.DataFrame) -> MeanBiasCorrection:
        working = frame.assign(error=frame["target_rain"] - frame["gfs_rain"])
        self.by_lead_month = working.groupby(["lead_hour", "month"], observed=True)["error"].mean()
        self.by_lead = working.groupby("lead_hour", observed=True)["error"].mean()
        self.global_bias = float(working["error"].mean())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_frame(frame[["lead_hour", "month"]])
        correction = self.by_lead_month.reindex(keys).to_numpy(dtype=float)
        missing = ~np.isfinite(correction)
        if missing.any():
            fallback = frame["lead_hour"].map(self.by_lead).fillna(self.global_bias).to_numpy(dtype=float)
            correction[missing] = fallback[missing]
        return np.clip(frame["gfs_rain"].to_numpy(dtype=float) + correction, 0.0, None)


class LeadQuantileMapping:
    def __init__(self, quantile_count: int = 101) -> None:
        self.quantiles = np.linspace(0.0, 1.0, quantile_count)

    def fit(self, frame: pd.DataFrame) -> LeadQuantileMapping:
        self.mappings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for lead, group in frame.groupby("lead_hour", observed=True):
            forecast_quantiles = np.quantile(group["gfs_rain"], self.quantiles)
            target_quantiles = np.quantile(group["target_rain"], self.quantiles)
            mapping = pd.DataFrame({"forecast": forecast_quantiles, "target": target_quantiles})
            mapping = mapping.groupby("forecast", sort=True, as_index=False)["target"].mean()
            self.mappings[int(lead)] = (
                mapping["forecast"].to_numpy(dtype=float),
                mapping["target"].to_numpy(dtype=float),
            )
        target_values = frame["target_rain"].to_numpy(dtype=float)
        forecast_values = frame["gfs_rain"].to_numpy(dtype=float)
        self.fallback = (
            np.quantile(forecast_values, self.quantiles),
            np.quantile(target_values, self.quantiles),
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        result = np.empty(len(frame), dtype=float)
        lead_values = frame["lead_hour"].to_numpy()
        rain = frame["gfs_rain"].to_numpy(dtype=float)
        for lead in np.unique(lead_values):
            indexes = lead_values == lead
            forecast_quantiles, target_quantiles = self.mappings.get(int(lead), self.fallback)
            unique_forecast, unique_indexes = np.unique(forecast_quantiles, return_index=True)
            unique_target = target_quantiles[unique_indexes]
            result[indexes] = np.interp(rain[indexes], unique_forecast, unique_target)
        return np.clip(result, 0.0, None)


class GridMonthClimatology:
    def fit(self, frame: pd.DataFrame) -> GridMonthClimatology:
        keys = ["month", "latitude", "longitude"]
        self.by_grid_month = frame.groupby(keys, observed=True)["target_rain"].mean()
        self.by_month = frame.groupby("month", observed=True)["target_rain"].mean()
        self.global_mean = float(frame["target_rain"].mean())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_frame(frame[["month", "latitude", "longitude"]])
        prediction = self.by_grid_month.reindex(keys).to_numpy(dtype=float)
        missing = ~np.isfinite(prediction)
        if missing.any():
            fallback = frame["month"].map(self.by_month).fillna(self.global_mean).to_numpy(dtype=float)
            prediction[missing] = fallback[missing]
        return np.clip(prediction, 0.0, None)


class IsotonicProbabilityCalibrator:
    def fit(self, score: np.ndarray, observed_event: np.ndarray) -> IsotonicProbabilityCalibrator:
        score = np.asarray(score, dtype=float)
        observed_event = np.asarray(observed_event, dtype=bool)
        valid = np.isfinite(score)
        score = score[valid]
        observed_event = observed_event[valid]
        self.constant: float | None = None
        self.model: IsotonicRegression | None = None
        if score.size == 0 or np.unique(observed_event).size < 2 or np.unique(score).size < 2:
            self.constant = float(np.mean(observed_event)) if observed_event.size else 0.0
        else:
            self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.model.fit(score, observed_event.astype(float))
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        score = np.asarray(score, dtype=float)
        if self.model is None:
            return np.full(score.shape, self.constant, dtype=float)
        return np.asarray(self.model.predict(score), dtype=float)


def fractions_skill_score(
    forecast: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    *,
    threshold_mm: float,
    neighborhood: int,
) -> float:
    if neighborhood < 1 or neighborhood % 2 == 0:
        raise ValueError("neighborhood must be a positive odd integer")
    forecast = np.asarray(forecast, dtype=float)
    observed = np.asarray(observed, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if forecast.shape != observed.shape or forecast.shape != mask.shape:
        raise ValueError("forecast, observed, and mask must have matching shapes")
    if forecast.ndim == 2:
        forecast = forecast[None, ...]
        observed = observed[None, ...]
        mask = mask[None, ...]
    numerator = 0.0
    denominator = 0.0
    for forecast_field, observed_field, valid in zip(forecast, observed, mask):
        valid = valid & np.isfinite(forecast_field) & np.isfinite(observed_field)
        valid_fraction = uniform_filter(valid.astype(float), size=neighborhood, mode="constant", cval=0.0)
        forecast_fraction = uniform_filter(
            ((forecast_field >= threshold_mm) & valid).astype(float),
            size=neighborhood,
            mode="constant",
            cval=0.0,
        )
        observed_fraction = uniform_filter(
            ((observed_field >= threshold_mm) & valid).astype(float),
            size=neighborhood,
            mode="constant",
            cval=0.0,
        )
        usable = valid & (valid_fraction > 0)
        forecast_fraction[usable] /= valid_fraction[usable]
        observed_fraction[usable] /= valid_fraction[usable]
        numerator += float(np.sum((forecast_fraction[usable] - observed_fraction[usable]) ** 2))
        denominator += float(np.sum(forecast_fraction[usable] ** 2 + observed_fraction[usable] ** 2))
    return 1.0 - numerator / denominator if denominator > 0 else math.nan


def fractions_skill_score_events(
    forecast_event: np.ndarray,
    observed_event: np.ndarray,
    mask: np.ndarray,
    *,
    neighborhood: int,
) -> float:
    return fractions_skill_score(
        np.asarray(forecast_event, dtype=float),
        np.asarray(observed_event, dtype=float),
        mask,
        threshold_mm=0.5,
        neighborhood=neighborhood,
    )


def paired_date_bootstrap(
    predicted: np.ndarray,
    baseline: np.ndarray,
    observed: np.ndarray,
    dates: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    predicted = np.asarray(predicted, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    observed = np.asarray(observed, dtype=float)
    dates = np.asarray(dates).astype(str)
    valid = np.isfinite(predicted) & np.isfinite(baseline) & np.isfinite(observed)
    working = pd.DataFrame(
        {
            "date": dates[valid],
            "model_abs": np.abs(predicted[valid] - observed[valid]),
            "base_abs": np.abs(baseline[valid] - observed[valid]),
            "model_sq": (predicted[valid] - observed[valid]) ** 2,
            "base_sq": (baseline[valid] - observed[valid]) ** 2,
            "model_diff": predicted[valid] - observed[valid],
            "base_diff": baseline[valid] - observed[valid],
        }
    )
    daily = working.groupby("date", sort=True).agg(
        count=("model_abs", "size"),
        model_abs=("model_abs", "sum"),
        base_abs=("base_abs", "sum"),
        model_sq=("model_sq", "sum"),
        base_sq=("base_sq", "sum"),
        model_diff=("model_diff", "sum"),
        base_diff=("base_diff", "sum"),
    )
    if daily.empty:
        raise ValueError("No valid samples available for bootstrap")
    values = daily.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(resamples, len(values)))
    totals = values[sampled].sum(axis=1)
    count = totals[:, 0]
    deltas = {
        "mae": totals[:, 1] / count - totals[:, 2] / count,
        "rmse": np.sqrt(totals[:, 3] / count) - np.sqrt(totals[:, 4] / count),
        "abs_bias": np.abs(totals[:, 5] / count) - np.abs(totals[:, 6] / count),
    }
    rows: list[dict[str, float | int | str]] = []
    for metric, values_for_metric in deltas.items():
        rows.append(
            {
                "metric": metric,
                "dates": int(len(daily)),
                "resamples": int(resamples),
                "delta_mean": float(np.mean(values_for_metric)),
                "ci_lower": float(np.quantile(values_for_metric, 0.025)),
                "ci_upper": float(np.quantile(values_for_metric, 0.975)),
                "probability_improved": float(np.mean(values_for_metric < 0.0)),
            }
        )
    return pd.DataFrame(rows)


def deterministic_event_scores(
    predicted_rain: np.ndarray,
    observed_rain: np.ndarray,
    threshold_mm: float,
) -> dict[str, float]:
    scores = categorical_scores_frame(predicted_rain >= threshold_mm, observed_rain >= threshold_mm)
    return {key: float(value) for key, value in scores.items()}
