import math

import numpy as np
import xarray as xr


def categorical_scores(forecast: xr.DataArray, observed: xr.DataArray, threshold_mm: float) -> dict[str, float]:
    f, o = xr.align(forecast, observed, join="inner")
    valid = np.isfinite(f) & np.isfinite(o)
    f_event = (f >= threshold_mm) & valid
    o_event = (o >= threshold_mm) & valid

    hits = int((f_event & o_event).sum())
    misses = int((~f_event & o_event & valid).sum())
    false_alarms = int((f_event & ~o_event & valid).sum())
    correct_negatives = int((~f_event & ~o_event & valid).sum())

    pod = _ratio(hits, hits + misses)
    far = _ratio(false_alarms, hits + false_alarms)
    csi = _ratio(hits, hits + misses + false_alarms)
    frequency_bias = _ratio(hits + false_alarms, hits + misses)
    ets = _ets(hits, misses, false_alarms, correct_negatives)

    return {
        "threshold_mm": float(threshold_mm),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "pod": pod,
        "far": far,
        "csi": csi,
        "frequency_bias": frequency_bias,
        "ets": ets,
    }


def continuous_scores(forecast: xr.DataArray, observed: xr.DataArray) -> dict[str, float]:
    f, o = xr.align(forecast, observed, join="inner")
    valid = np.isfinite(f) & np.isfinite(o)
    diff = (f - o).where(valid)
    return {
        "mae": float(abs(diff).mean(skipna=True)),
        "rmse": float(np.sqrt((diff**2).mean(skipna=True))),
        "bias": float(diff.mean(skipna=True)),
        "pearson_r": _pearson(f.where(valid), o.where(valid)),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _ets(hits: int, misses: int, false_alarms: int, correct_negatives: int) -> float:
    total = hits + misses + false_alarms + correct_negatives
    random_hits = ((hits + misses) * (hits + false_alarms)) / total if total else math.nan
    return _ratio(hits - random_hits, hits + misses + false_alarms - random_hits)


def _pearson(forecast: xr.DataArray, observed: xr.DataArray) -> float:
    f = forecast.values.reshape(-1)
    o = observed.values.reshape(-1)
    mask = np.isfinite(f) & np.isfinite(o)
    if mask.sum() < 2:
        return math.nan
    return float(np.corrcoef(f[mask], o[mask])[0, 1])
