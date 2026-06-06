# AI Weather Extreme Rainfall Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python and Jupyter pipeline for evaluating ECMWF AIFS AI weather forecasts against ECMWF IFS forecasts for extreme rainfall risk over the middle and lower Yangtze River region.

**Architecture:** The project is a small Python package plus one notebook. Data access modules ingest forecast and observation files, processing modules align grids and build daily rainfall accumulations, metrics modules compute deterministic rainfall verification scores, and plotting/report modules emit CSV, PNG, NetCDF, and Markdown outputs.

**Tech Stack:** Python 3.11+, `xarray`, `numpy`, `pandas`, `scipy`, `matplotlib`, `cfgrib`, `eccodes`, `ecmwf-opendata`, `pyyaml`, `pytest`, and Jupyter.

---

## Summary

This plan implements route 1: **AIFS + IFS + IMERG systematic evaluation**.

The first version focuses on the middle and lower Yangtze River region, heavy rainfall and rainstorm events, Jupyter/script usage, and a systematic flood-season benchmark. The target label remains 2023-2025 flood seasons, but the directly comparable AIFS-vs-IFS period starts from 2024 because ECMWF states that historical AIFS Single archive data exists from February 2024 onwards. The plan therefore uses:

- **Primary benchmark:** 2024-2025 flood seasons, AIFS vs IFS vs IMERG.
- **Background benchmark:** 2023 flood season, IFS vs IMERG only, used as a baseline context unless a valid AIFS backfill becomes available.
- **Operational demo:** current and rolling forecasts through ECMWF Open Data where available.

This tool is for research and decision support. It must display a clear disclaimer in generated reports: it is not an official weather warning service and must not replace national or local meteorological warnings.

## Data Sources And Fixed Decisions

Primary sources:

- ECMWF AIFS Machine Learning data: AIFS deterministic forecasts are 6-hourly to 15 days and disseminated at about 0.25 degree; historical AIFS Single archive data exists from February 2024 onwards. Source: https://www.ecmwf.int/en/forecasts/dataset/aifs-machine-learning-data
- ECMWF Open Data: public real-time subset from IFS and AIFS, normally GRIB2 at 0.25 degree with rolling availability. Source: https://www.ecmwf.int/en/forecasts/datasets/open-data
- ECMWF open-data user documentation: Python access uses `ecmwf-opendata`. Source: https://confluence.ecmwf.int/display/DAC/ECMWF+open+data%3A+real-time+forecasts+from+IFS+and+AIFS
- NASA GPM IMERG: precipitation estimates over most of Earth, with 0.1 degree and 30-minute products; daily products are also available. Sources: https://gpm.nasa.gov/data/imerg and https://gpm.nasa.gov/data

Fixed evaluation choices:

- Region: middle and lower Yangtze default bounding box, `lon=108.0..123.0`, `lat=24.0..35.0`.
- Season window: May 1 through September 30 for each year.
- Forecast cycles: default `00` and `12` UTC cycles to reduce data volume and keep AIFS/IFS comparisons aligned.
- Lead times: 24, 48, 72, 96, 120, 168, 240, and 360 hours.
- Rainfall accumulation: evaluate 24-hour rainfall totals ending at the configured lead time.
- Event thresholds: 24-hour rainfall >= 50 mm and >= 100 mm.
- Main categorical scores: POD, FAR, CSI/TS, frequency bias, ETS.
- Main continuous scores: MAE, RMSE, mean bias, spatial Pearson correlation.
- Grid alignment: aggregate or interpolate IMERG observations onto the forecast 0.25-degree grid before scoring.
- Output root: `outputs/rainfall_benchmark/`.

## File Structure

Create these files:

- `README.md`: project overview, quick start, data access caveats, and report examples.
- `requirements.txt`: pinned minimum dependencies for the first implementation.
- `configs/rainfall_benchmark.yaml`: default region, dates, cycles, leads, paths, thresholds, and output settings.
- `notebooks/01_extreme_rainfall_benchmark.ipynb`: interactive notebook that runs the same pipeline as the CLI.
- `src/ai_weather_rainfall/__init__.py`: package metadata.
- `src/ai_weather_rainfall/config.py`: typed configuration loader and validation.
- `src/ai_weather_rainfall/regions.py`: region bounding boxes and grid helpers.
- `src/ai_weather_rainfall/data_ecmwf.py`: ECMWF local-file ingestion and real-time open-data fetch helper.
- `src/ai_weather_rainfall/data_imerg.py`: IMERG file discovery, loading, and daily accumulation.
- `src/ai_weather_rainfall/accumulations.py`: forecast precipitation differencing and 24-hour totals.
- `src/ai_weather_rainfall/regrid.py`: observation-to-forecast grid alignment.
- `src/ai_weather_rainfall/metrics.py`: continuous and categorical rainfall verification metrics.
- `src/ai_weather_rainfall/plots.py`: maps, lead-time score plots, and case-study figures.
- `src/ai_weather_rainfall/report.py`: Markdown report writer.
- `src/ai_weather_rainfall/pipeline.py`: end-to-end orchestration.
- `src/ai_weather_rainfall/cli.py`: command-line entry point.
- `tests/test_config.py`: config validation tests.
- `tests/test_accumulations.py`: forecast accumulation tests.
- `tests/test_regrid.py`: grid alignment tests.
- `tests/test_metrics.py`: metric formula tests.
- `tests/test_pipeline_smoke.py`: tiny synthetic end-to-end test.

Generated outputs:

- `outputs/rainfall_benchmark/metrics/summary_metrics.csv`
- `outputs/rainfall_benchmark/metrics/by_lead_threshold.csv`
- `outputs/rainfall_benchmark/processed/*.nc`
- `outputs/rainfall_benchmark/figures/*.png`
- `outputs/rainfall_benchmark/reports/rainfall_benchmark_report.md`

## Task 1: Project Skeleton And Dependencies

**Files:**

- Create: `requirements.txt`
- Create: `README.md`
- Create: package and output directories listed above

- [ ] **Step 1: Create directories**

Run:

```bash
mkdir -p configs notebooks src/ai_weather_rainfall tests outputs/rainfall_benchmark/{raw,processed,metrics,figures,reports}
```

Expected: directories exist and `git status --short` shows new untracked paths.

- [ ] **Step 2: Add `requirements.txt`**

Write:

```txt
numpy>=1.26
pandas>=2.2
xarray>=2024.7
scipy>=1.13
matplotlib>=3.8
pyyaml>=6.0
cfgrib>=0.9.14
eccodes>=1.7
ecmwf-opendata>=0.3
netcdf4>=1.7
h5netcdf>=1.3
jupyterlab>=4.2
pytest>=8.2
```

- [ ] **Step 3: Add `README.md`**

Write the README with these sections:

```markdown
# AI Weather Extreme Rainfall Benchmark

This project evaluates ECMWF AIFS AI weather forecasts against ECMWF IFS forecasts for extreme rainfall over the middle and lower Yangtze River region.

The first benchmark uses NASA GPM IMERG rainfall as the observation reference. It is a research and decision-support tool, not an official warning service.

## Quick Start

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m ai_weather_rainfall.cli --config configs/rainfall_benchmark.yaml
```

## Data Layout

Place forecast and observation data under `outputs/rainfall_benchmark/raw/`:

- `ecmwf/aifs/` for AIFS forecast GRIB or NetCDF files.
- `ecmwf/ifs/` for IFS forecast GRIB or NetCDF files.
- `imerg/` for IMERG precipitation files.

Historical AIFS Single data starts from February 2024 in ECMWF archive access. The 2023 flood season is kept as an IFS-vs-IMERG background baseline unless valid AIFS backfill data is supplied.

## Outputs

The pipeline writes metrics, processed NetCDF files, figures, and a Markdown report under `outputs/rainfall_benchmark/`.
```
```

- [ ] **Step 4: Verify dependency installation**

Run:

```bash
python -m pip install -r requirements.txt
python - <<'PY'
import numpy, pandas, xarray, scipy, matplotlib, yaml
print("dependencies-ok")
PY
```

Expected: final line is `dependencies-ok`.

- [ ] **Step 5: Commit**

Run:

```bash
git add README.md requirements.txt configs notebooks src tests outputs
git commit -m "chore: scaffold rainfall benchmark project"
```

## Task 2: Configuration And Region Model

**Files:**

- Create: `configs/rainfall_benchmark.yaml`
- Create: `src/ai_weather_rainfall/__init__.py`
- Create: `src/ai_weather_rainfall/config.py`
- Create: `src/ai_weather_rainfall/regions.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add default YAML config**

Write `configs/rainfall_benchmark.yaml`:

```yaml
project:
  name: yangtze_extreme_rainfall_benchmark
  output_root: outputs/rainfall_benchmark

region:
  name: middle_lower_yangtze
  lon_min: 108.0
  lon_max: 123.0
  lat_min: 24.0
  lat_max: 35.0

period:
  years: [2023, 2024, 2025]
  season_start: "05-01"
  season_end: "09-30"
  primary_comparison_years: [2024, 2025]
  background_baseline_years: [2023]

forecast:
  models: ["aifs", "ifs"]
  cycles_utc: ["00", "12"]
  lead_hours: [24, 48, 72, 96, 120, 168, 240, 360]
  accumulation_hours: 24

rainfall:
  thresholds_mm_24h: [50.0, 100.0]

paths:
  aifs_raw: outputs/rainfall_benchmark/raw/ecmwf/aifs
  ifs_raw: outputs/rainfall_benchmark/raw/ecmwf/ifs
  imerg_raw: outputs/rainfall_benchmark/raw/imerg
  processed: outputs/rainfall_benchmark/processed
  metrics: outputs/rainfall_benchmark/metrics
  figures: outputs/rainfall_benchmark/figures
  reports: outputs/rainfall_benchmark/reports
```

- [ ] **Step 2: Write failing config tests**

Write `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from ai_weather_rainfall.config import load_config


def test_load_default_config():
    cfg = load_config(Path("configs/rainfall_benchmark.yaml"))
    assert cfg.region.name == "middle_lower_yangtze"
    assert cfg.region.lon_min == 108.0
    assert cfg.forecast.lead_hours == [24, 48, 72, 96, 120, 168, 240, 360]
    assert cfg.rainfall.thresholds_mm_24h == [50.0, 100.0]


def test_invalid_region_bounds_fail(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
project: {name: bad, output_root: outputs}
region: {name: bad, lon_min: 123, lon_max: 108, lat_min: 24, lat_max: 35}
period: {years: [2024], season_start: "05-01", season_end: "09-30", primary_comparison_years: [2024], background_baseline_years: []}
forecast: {models: ["aifs"], cycles_utc: ["00"], lead_hours: [24], accumulation_hours: 24}
rainfall: {thresholds_mm_24h: [50.0]}
paths: {aifs_raw: raw/aifs, ifs_raw: raw/ifs, imerg_raw: raw/imerg, processed: processed, metrics: metrics, figures: figures, reports: reports}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lon_min must be smaller"):
        load_config(path)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/test_config.py -v
```

Expected: import failure because `ai_weather_rainfall.config` does not exist.

- [ ] **Step 4: Implement config loader**

Write `src/ai_weather_rainfall/config.py`:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    output_root: Path


@dataclass(frozen=True)
class RegionConfig:
    name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


@dataclass(frozen=True)
class PeriodConfig:
    years: list[int]
    season_start: str
    season_end: str
    primary_comparison_years: list[int]
    background_baseline_years: list[int]


@dataclass(frozen=True)
class ForecastConfig:
    models: list[str]
    cycles_utc: list[str]
    lead_hours: list[int]
    accumulation_hours: int


@dataclass(frozen=True)
class RainfallConfig:
    thresholds_mm_24h: list[float]


@dataclass(frozen=True)
class PathConfig:
    aifs_raw: Path
    ifs_raw: Path
    imerg_raw: Path
    processed: Path
    metrics: Path
    figures: Path
    reports: Path


@dataclass(frozen=True)
class BenchmarkConfig:
    project: ProjectConfig
    region: RegionConfig
    period: PeriodConfig
    forecast: ForecastConfig
    rainfall: RainfallConfig
    paths: PathConfig


def load_config(path: Path) -> BenchmarkConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = BenchmarkConfig(
        project=_project(data["project"]),
        region=_region(data["region"]),
        period=_period(data["period"]),
        forecast=_forecast(data["forecast"]),
        rainfall=_rainfall(data["rainfall"]),
        paths=_paths(data["paths"]),
    )
    _validate(cfg)
    return cfg


def _project(data: dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(name=str(data["name"]), output_root=Path(data["output_root"]))


def _region(data: dict[str, Any]) -> RegionConfig:
    return RegionConfig(
        name=str(data["name"]),
        lon_min=float(data["lon_min"]),
        lon_max=float(data["lon_max"]),
        lat_min=float(data["lat_min"]),
        lat_max=float(data["lat_max"]),
    )


def _period(data: dict[str, Any]) -> PeriodConfig:
    return PeriodConfig(
        years=[int(v) for v in data["years"]],
        season_start=str(data["season_start"]),
        season_end=str(data["season_end"]),
        primary_comparison_years=[int(v) for v in data["primary_comparison_years"]],
        background_baseline_years=[int(v) for v in data["background_baseline_years"]],
    )


def _forecast(data: dict[str, Any]) -> ForecastConfig:
    return ForecastConfig(
        models=[str(v) for v in data["models"]],
        cycles_utc=[str(v) for v in data["cycles_utc"]],
        lead_hours=[int(v) for v in data["lead_hours"]],
        accumulation_hours=int(data["accumulation_hours"]),
    )


def _rainfall(data: dict[str, Any]) -> RainfallConfig:
    return RainfallConfig(thresholds_mm_24h=[float(v) for v in data["thresholds_mm_24h"]])


def _paths(data: dict[str, Any]) -> PathConfig:
    return PathConfig(**{key: Path(value) for key, value in data.items()})


def _validate(cfg: BenchmarkConfig) -> None:
    if cfg.region.lon_min >= cfg.region.lon_max:
        raise ValueError("lon_min must be smaller than lon_max")
    if cfg.region.lat_min >= cfg.region.lat_max:
        raise ValueError("lat_min must be smaller than lat_max")
    if cfg.forecast.accumulation_hours <= 0:
        raise ValueError("accumulation_hours must be positive")
    if not all(lead > 0 for lead in cfg.forecast.lead_hours):
        raise ValueError("lead_hours must be positive")
```

Write `src/ai_weather_rainfall/__init__.py`:

```python
__all__ = ["__version__"]

__version__ = "0.1.0"
```

- [ ] **Step 5: Implement region helpers**

Write `src/ai_weather_rainfall/regions.py`:

```python
from dataclasses import dataclass

import xarray as xr

from ai_weather_rainfall.config import RegionConfig


@dataclass(frozen=True)
class RegionBounds:
    name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


def bounds_from_config(region: RegionConfig) -> RegionBounds:
    return RegionBounds(
        name=region.name,
        lon_min=region.lon_min,
        lon_max=region.lon_max,
        lat_min=region.lat_min,
        lat_max=region.lat_max,
    )


def subset_region(data: xr.DataArray | xr.Dataset, bounds: RegionBounds) -> xr.DataArray | xr.Dataset:
    lon_name = "longitude" if "longitude" in data.coords else "lon"
    lat_name = "latitude" if "latitude" in data.coords else "lat"
    lat_values = data[lat_name]
    lat_slice = slice(bounds.lat_max, bounds.lat_min) if lat_values[0] > lat_values[-1] else slice(bounds.lat_min, bounds.lat_max)
    return data.sel({lon_name: slice(bounds.lon_min, bounds.lon_max), lat_name: lat_slice})
```

- [ ] **Step 6: Verify config tests pass**

Run:

```bash
PYTHONPATH=src pytest tests/test_config.py -v
```

Expected: both tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add configs/rainfall_benchmark.yaml src/ai_weather_rainfall/__init__.py src/ai_weather_rainfall/config.py src/ai_weather_rainfall/regions.py tests/test_config.py
git commit -m "feat: add benchmark configuration model"
```

## Task 3: Forecast Accumulation Logic

**Files:**

- Create: `src/ai_weather_rainfall/accumulations.py`
- Test: `tests/test_accumulations.py`

- [ ] **Step 1: Write accumulation tests**

Write `tests/test_accumulations.py`:

```python
import numpy as np
import pandas as pd
import xarray as xr

from ai_weather_rainfall.accumulations import accumulated_to_window_total


def test_accumulated_to_24h_total():
    leads = [0, 6, 12, 18, 24, 30]
    values = np.array(leads, dtype=float).reshape(len(leads), 1, 1)
    da = xr.DataArray(
        values,
        dims=("lead_hour", "latitude", "longitude"),
        coords={"lead_hour": leads, "latitude": [30.0], "longitude": [115.0]},
        name="tp",
    )

    result = accumulated_to_window_total(da, accumulation_hours=24)

    assert result.sel(lead_hour=24).item() == 24.0
    assert result.sel(lead_hour=30).item() == 24.0


def test_negative_resets_are_clipped_to_nan():
    da = xr.DataArray(
        np.array([0.0, 10.0, 5.0]).reshape(3, 1, 1),
        dims=("lead_hour", "latitude", "longitude"),
        coords={"lead_hour": [0, 6, 12], "latitude": [30.0], "longitude": [115.0]},
        name="tp",
    )

    result = accumulated_to_window_total(da, accumulation_hours=6)

    assert np.isnan(result.sel(lead_hour=12).item())
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_accumulations.py -v
```

Expected: import failure because `accumulations.py` does not exist.

- [ ] **Step 3: Implement forecast accumulation**

Write `src/ai_weather_rainfall/accumulations.py`:

```python
import numpy as np
import xarray as xr


def accumulated_to_window_total(tp_accumulated: xr.DataArray, accumulation_hours: int) -> xr.DataArray:
    if "lead_hour" not in tp_accumulated.dims:
        raise ValueError("tp_accumulated must have a lead_hour dimension")
    leads = [int(v) for v in tp_accumulated["lead_hour"].values]
    out = []
    for lead in leads:
        start = lead - accumulation_hours
        if start < 0:
            window = tp_accumulated.sel(lead_hour=lead) * np.nan
        elif start == 0:
            window = tp_accumulated.sel(lead_hour=lead)
        else:
            window = tp_accumulated.sel(lead_hour=lead) - tp_accumulated.sel(lead_hour=start)
        window = window.where(window >= 0)
        out.append(window.expand_dims(lead_hour=[lead]))
    result = xr.concat(out, dim="lead_hour")
    result.name = f"rainfall_{accumulation_hours}h_mm"
    result.attrs["units"] = "mm"
    return result
```

- [ ] **Step 4: Verify tests pass**

Run:

```bash
PYTHONPATH=src pytest tests/test_accumulations.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/ai_weather_rainfall/accumulations.py tests/test_accumulations.py
git commit -m "feat: compute forecast rainfall accumulations"
```

## Task 4: Metrics

**Files:**

- Create: `src/ai_weather_rainfall/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write metric tests**

Write `tests/test_metrics.py`:

```python
import numpy as np
import xarray as xr

from ai_weather_rainfall.metrics import categorical_scores, continuous_scores


def test_categorical_scores():
    forecast = xr.DataArray([[60.0, 10.0], [80.0, 20.0]], dims=("y", "x"))
    observed = xr.DataArray([[70.0, 60.0], [10.0, 20.0]], dims=("y", "x"))

    scores = categorical_scores(forecast, observed, threshold_mm=50.0)

    assert scores["hits"] == 1
    assert scores["misses"] == 1
    assert scores["false_alarms"] == 1
    assert scores["correct_negatives"] == 1
    assert scores["pod"] == 0.5
    assert scores["far"] == 0.5
    assert scores["csi"] == 1 / 3


def test_continuous_scores():
    forecast = xr.DataArray([1.0, 3.0, np.nan], dims=("point",))
    observed = xr.DataArray([2.0, 1.0, 5.0], dims=("point",))

    scores = continuous_scores(forecast, observed)

    assert scores["mae"] == 1.5
    assert scores["bias"] == 0.0
    assert scores["rmse"] == np.sqrt(2.5)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_metrics.py -v
```

Expected: import failure because `metrics.py` does not exist.

- [ ] **Step 3: Implement metrics**

Write `src/ai_weather_rainfall/metrics.py`:

```python
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
```

- [ ] **Step 4: Verify metric tests pass**

Run:

```bash
PYTHONPATH=src pytest tests/test_metrics.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/ai_weather_rainfall/metrics.py tests/test_metrics.py
git commit -m "feat: add rainfall verification metrics"
```

## Task 5: IMERG Observation Loading

**Files:**

- Create: `src/ai_weather_rainfall/data_imerg.py`
- Test: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Add synthetic observation fixture in smoke test**

Write the first part of `tests/test_pipeline_smoke.py`:

```python
import numpy as np
import pandas as pd
import xarray as xr

from ai_weather_rainfall.data_imerg import daily_observed_rainfall


def test_daily_observed_rainfall_from_half_hourly_data():
    times = pd.date_range("2024-06-01", periods=48, freq="30min")
    da = xr.DataArray(
        np.ones((48, 2, 2)),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": [30.0, 30.1], "longitude": [115.0, 115.1]},
        name="precipitation",
        attrs={"units": "mm/hr"},
    )

    daily = daily_observed_rainfall(da)

    assert daily.sizes["time"] == 1
    assert float(daily.isel(time=0, latitude=0, longitude=0)) == 24.0
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_pipeline_smoke.py::test_daily_observed_rainfall_from_half_hourly_data -v
```

Expected: import failure because `data_imerg.py` does not exist.

- [ ] **Step 3: Implement IMERG daily accumulation**

Write `src/ai_weather_rainfall/data_imerg.py`:

```python
from pathlib import Path

import xarray as xr


def open_imerg_files(paths: list[Path]) -> xr.DataArray:
    if not paths:
        raise FileNotFoundError("No IMERG files were provided")
    dataset = xr.open_mfdataset([str(path) for path in paths], combine="by_coords")
    for name in ("precipitation", "precipitationCal", "precipitationCalibrated"):
        if name in dataset:
            return dataset[name]
    raise KeyError("IMERG precipitation variable not found")


def daily_observed_rainfall(precipitation: xr.DataArray) -> xr.DataArray:
    if "time" not in precipitation.dims:
        raise ValueError("IMERG precipitation must have a time dimension")
    units = str(precipitation.attrs.get("units", "")).lower()
    if "mm/hr" in units or "mm h-1" in units:
        interval_hours = _median_interval_hours(precipitation)
        rainfall = precipitation * interval_hours
    else:
        rainfall = precipitation
    daily = rainfall.resample(time="1D").sum(skipna=True)
    daily.name = "observed_rainfall_24h_mm"
    daily.attrs["units"] = "mm"
    return daily


def _median_interval_hours(data: xr.DataArray) -> float:
    seconds = data["time"].diff("time").astype("timedelta64[s]").astype(float)
    return float(seconds.median()) / 3600.0
```

- [ ] **Step 4: Verify observation test passes**

Run:

```bash
PYTHONPATH=src pytest tests/test_pipeline_smoke.py::test_daily_observed_rainfall_from_half_hourly_data -v
```

Expected: test passes.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/ai_weather_rainfall/data_imerg.py tests/test_pipeline_smoke.py
git commit -m "feat: load imerg rainfall observations"
```

## Task 6: ECMWF Forecast Loading

**Files:**

- Create: `src/ai_weather_rainfall/data_ecmwf.py`
- Test: extend `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Add forecast loading test**

Append to `tests/test_pipeline_smoke.py`:

```python
from ai_weather_rainfall.data_ecmwf import normalize_forecast_dataset


def test_normalize_forecast_dataset_renames_tp():
    ds = xr.Dataset(
        {"tp": (("step", "latitude", "longitude"), np.ones((2, 1, 1)))},
        coords={"step": [np.timedelta64(6, "h"), np.timedelta64(12, "h")], "latitude": [30.0], "longitude": [115.0]},
    )

    normalized = normalize_forecast_dataset(ds)

    assert "lead_hour" in normalized.dims
    assert "tp" in normalized
    assert normalized["lead_hour"].values.tolist() == [6, 12]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_pipeline_smoke.py::test_normalize_forecast_dataset_renames_tp -v
```

Expected: import failure because `data_ecmwf.py` does not exist.

- [ ] **Step 3: Implement ECMWF normalization and file opening**

Write `src/ai_weather_rainfall/data_ecmwf.py`:

```python
from pathlib import Path

import numpy as np
import xarray as xr


def open_forecast_file(path: Path) -> xr.Dataset:
    suffix = path.suffix.lower()
    if suffix in {".nc", ".netcdf"}:
        return normalize_forecast_dataset(xr.open_dataset(path))
    if suffix in {".grib", ".grib2"}:
        return normalize_forecast_dataset(xr.open_dataset(path, engine="cfgrib"))
    raise ValueError(f"Unsupported forecast file type: {path}")


def normalize_forecast_dataset(dataset: xr.Dataset) -> xr.Dataset:
    ds = dataset.copy()
    if "step" in ds.coords and "lead_hour" not in ds.coords:
        lead_hours = ds["step"].astype("timedelta64[h]").astype(int)
        ds = ds.assign_coords(lead_hour=("step", lead_hours.values)).swap_dims({"step": "lead_hour"}).drop_vars("step")
    if "valid_time" in ds.coords and "time" in ds.coords:
        ds = ds.drop_vars("valid_time")
    if "tp" not in ds:
        candidates = [name for name in ds.data_vars if name.lower() in {"total_precipitation", "precip"}]
        if candidates:
            ds = ds.rename({candidates[0]: "tp"})
    if "tp" not in ds:
        raise KeyError("Forecast dataset must contain total precipitation as tp")
    ds["tp"] = _to_millimeters(ds["tp"])
    return ds


def _to_millimeters(data: xr.DataArray) -> xr.DataArray:
    units = str(data.attrs.get("units", "")).lower()
    if units in {"m", "metres", "meters"}:
        converted = data * 1000.0
    else:
        converted = data
    converted.name = "tp"
    converted.attrs["units"] = "mm"
    converted = converted.where(np.isfinite(converted))
    return converted
```

- [ ] **Step 4: Verify forecast normalization test passes**

Run:

```bash
PYTHONPATH=src pytest tests/test_pipeline_smoke.py::test_normalize_forecast_dataset_renames_tp -v
```

Expected: test passes.

- [ ] **Step 5: Add real-time fetch helper documentation in module**

Add this function to `data_ecmwf.py`:

```python
def describe_realtime_open_data_fetch() -> str:
    return (
        "Use ecmwf-opendata for current or rolling ECMWF Open Data retrieval. "
        "Historical AIFS/IFS benchmark files should be retrieved from ECMWF archive access "
        "or another licensed archive, then placed under the configured raw data directories."
    )
```

- [ ] **Step 6: Commit**

Run:

```bash
git add src/ai_weather_rainfall/data_ecmwf.py tests/test_pipeline_smoke.py
git commit -m "feat: load ecmwf forecast datasets"
```

## Task 7: Grid Alignment

**Files:**

- Create: `src/ai_weather_rainfall/regrid.py`
- Test: `tests/test_regrid.py`

- [ ] **Step 1: Write grid alignment tests**

Write `tests/test_regrid.py`:

```python
import numpy as np
import xarray as xr

from ai_weather_rainfall.regrid import align_observation_to_forecast_grid


def test_align_observation_to_forecast_grid():
    observed = xr.DataArray(
        np.arange(9, dtype=float).reshape(3, 3),
        dims=("latitude", "longitude"),
        coords={"latitude": [29.9, 30.0, 30.1], "longitude": [114.9, 115.0, 115.1]},
        name="observed",
    )
    forecast = xr.DataArray(
        np.ones((1, 1)),
        dims=("latitude", "longitude"),
        coords={"latitude": [30.0], "longitude": [115.0]},
        name="forecast",
    )

    aligned = align_observation_to_forecast_grid(observed, forecast)

    assert aligned.dims == forecast.dims
    assert aligned.shape == forecast.shape
    assert float(aligned) == 4.0
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_regrid.py -v
```

Expected: import failure because `regrid.py` does not exist.

- [ ] **Step 3: Implement alignment**

Write `src/ai_weather_rainfall/regrid.py`:

```python
import xarray as xr


def align_observation_to_forecast_grid(observed: xr.DataArray, forecast_template: xr.DataArray) -> xr.DataArray:
    lat_name = "latitude" if "latitude" in observed.coords else "lat"
    lon_name = "longitude" if "longitude" in observed.coords else "lon"
    target_lat = "latitude" if "latitude" in forecast_template.coords else "lat"
    target_lon = "longitude" if "longitude" in forecast_template.coords else "lon"
    aligned = observed.interp(
        {lat_name: forecast_template[target_lat], lon_name: forecast_template[target_lon]},
        method="linear",
    )
    if lat_name != target_lat or lon_name != target_lon:
        aligned = aligned.rename({lat_name: target_lat, lon_name: target_lon})
    aligned.name = "observed_on_forecast_grid"
    aligned.attrs["units"] = observed.attrs.get("units", "mm")
    return aligned
```

- [ ] **Step 4: Verify grid tests pass**

Run:

```bash
PYTHONPATH=src pytest tests/test_regrid.py -v
```

Expected: test passes.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/ai_weather_rainfall/regrid.py tests/test_regrid.py
git commit -m "feat: align observations to forecast grid"
```

## Task 8: Pipeline Orchestration

**Files:**

- Create: `src/ai_weather_rainfall/pipeline.py`
- Create: `src/ai_weather_rainfall/cli.py`
- Test: extend `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Add synthetic end-to-end test**

Append to `tests/test_pipeline_smoke.py`:

```python
from ai_weather_rainfall.pipeline import score_forecast_against_observation


def test_score_forecast_against_observation():
    forecast = xr.DataArray(
        [[60.0, 20.0], [30.0, 110.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [30.0, 31.0], "longitude": [115.0, 116.0]},
        name="forecast",
    )
    observed = xr.DataArray(
        [[70.0, 10.0], [60.0, 90.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [30.0, 31.0], "longitude": [115.0, 116.0]},
        name="observed",
    )

    scores = score_forecast_against_observation(forecast, observed, thresholds=[50.0])

    assert scores["continuous"]["mae"] == 17.5
    assert scores["categorical"][0]["hits"] == 2
    assert scores["categorical"][0]["misses"] == 1
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_pipeline_smoke.py::test_score_forecast_against_observation -v
```

Expected: import failure because `pipeline.py` does not exist.

- [ ] **Step 3: Implement scoring orchestration**

Write `src/ai_weather_rainfall/pipeline.py`:

```python
from pathlib import Path

import pandas as pd
import xarray as xr

from ai_weather_rainfall.config import BenchmarkConfig, load_config
from ai_weather_rainfall.metrics import categorical_scores, continuous_scores


def score_forecast_against_observation(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    thresholds: list[float],
) -> dict[str, object]:
    return {
        "continuous": continuous_scores(forecast, observed),
        "categorical": [categorical_scores(forecast, observed, threshold) for threshold in thresholds],
    }


def run_benchmark(config_path: Path) -> Path:
    cfg = load_config(config_path)
    _ensure_output_dirs(cfg)
    report_path = cfg.paths.reports / "rainfall_benchmark_report.md"
    report_path.write_text(
        "# Rainfall Benchmark Report\n\n"
        "Pipeline wiring is installed. Add forecast and IMERG files under the configured raw directories, "
        "then run the full data-processing tasks in this plan.\n",
        encoding="utf-8",
    )
    metrics_path = cfg.paths.metrics / "summary_metrics.csv"
    pd.DataFrame([{"status": "pipeline-ready"}]).to_csv(metrics_path, index=False)
    return report_path


def _ensure_output_dirs(cfg: BenchmarkConfig) -> None:
    for path in [
        cfg.paths.aifs_raw,
        cfg.paths.ifs_raw,
        cfg.paths.imerg_raw,
        cfg.paths.processed,
        cfg.paths.metrics,
        cfg.paths.figures,
        cfg.paths.reports,
    ]:
        path.mkdir(parents=True, exist_ok=True)
```

Write `src/ai_weather_rainfall/cli.py`:

```python
import argparse
from pathlib import Path

from ai_weather_rainfall.pipeline import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI weather rainfall benchmark.")
    parser.add_argument("--config", type=Path, default=Path("configs/rainfall_benchmark.yaml"))
    args = parser.parse_args()
    report_path = run_benchmark(args.config)
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify pipeline smoke test passes**

Run:

```bash
PYTHONPATH=src pytest tests/test_pipeline_smoke.py::test_score_forecast_against_observation -v
```

Expected: test passes.

- [ ] **Step 5: Verify CLI creates first report**

Run:

```bash
PYTHONPATH=src python -m ai_weather_rainfall.cli --config configs/rainfall_benchmark.yaml
```

Expected: terminal prints `report=outputs/rainfall_benchmark/reports/rainfall_benchmark_report.md`.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/ai_weather_rainfall/pipeline.py src/ai_weather_rainfall/cli.py tests/test_pipeline_smoke.py outputs/rainfall_benchmark
git commit -m "feat: add rainfall benchmark pipeline entrypoint"
```

## Task 9: Full Benchmark Processing Loop

**Files:**

- Modify: `src/ai_weather_rainfall/pipeline.py`
- Modify: `src/ai_weather_rainfall/data_ecmwf.py`
- Modify: `src/ai_weather_rainfall/data_imerg.py`
- Test: extend `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Add file discovery functions**

Add to `data_ecmwf.py`:

```python
def discover_forecast_files(root: Path, model: str, year: int, cycle: str) -> list[Path]:
    pattern = f"**/*{model}*{year}*{cycle}*.grib*"
    grib = sorted(root.glob(pattern))
    netcdf = sorted(root.glob(f"**/*{model}*{year}*{cycle}*.nc"))
    return grib + netcdf
```

Add to `data_imerg.py`:

```python
def discover_imerg_files(root: Path, year: int) -> list[Path]:
    return sorted(root.glob(f"**/*{year}*.nc")) + sorted(root.glob(f"**/*{year}*.HDF5"))
```

- [ ] **Step 2: Add benchmark row schema**

Add to `pipeline.py`:

```python
def metric_rows(
    model: str,
    year: int,
    cycle: str,
    lead_hour: int,
    scores: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    continuous = scores["continuous"]
    rows.append({"model": model, "year": year, "cycle": cycle, "lead_hour": lead_hour, "metric_type": "continuous", **continuous})
    for item in scores["categorical"]:
        rows.append({"model": model, "year": year, "cycle": cycle, "lead_hour": lead_hour, "metric_type": "categorical", **item})
    return rows
```

- [ ] **Step 3: Add direct unit test for metric rows**

Append to `tests/test_pipeline_smoke.py`:

```python
from ai_weather_rainfall.pipeline import metric_rows


def test_metric_rows_include_model_metadata():
    rows = metric_rows(
        model="aifs",
        year=2024,
        cycle="00",
        lead_hour=24,
        scores={"continuous": {"mae": 1.0}, "categorical": [{"threshold_mm": 50.0, "csi": 0.4}]},
    )

    assert rows[0]["model"] == "aifs"
    assert rows[0]["metric_type"] == "continuous"
    assert rows[1]["threshold_mm"] == 50.0
```

- [ ] **Step 4: Verify metric row test passes**

Run:

```bash
PYTHONPATH=src pytest tests/test_pipeline_smoke.py::test_metric_rows_include_model_metadata -v
```

Expected: test passes.

- [ ] **Step 5: Implement full data loop**

Modify `run_benchmark` in `pipeline.py` to:

```python
def run_benchmark(config_path: Path) -> Path:
    cfg = load_config(config_path)
    _ensure_output_dirs(cfg)
    rows: list[dict[str, object]] = []

    for year in cfg.period.years:
        active_models = ["ifs"] if year in cfg.period.background_baseline_years else cfg.forecast.models
        for model in active_models:
            for cycle in cfg.forecast.cycles_utc:
                rows.extend(_score_available_files(cfg, model, year, cycle))

    metrics_path = cfg.paths.metrics / "summary_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    report_path = cfg.paths.reports / "rainfall_benchmark_report.md"
    report_path.write_text(_report_text(cfg, rows, metrics_path), encoding="utf-8")
    return report_path
```

Add helper functions:

```python
def _score_available_files(cfg: BenchmarkConfig, model: str, year: int, cycle: str) -> list[dict[str, object]]:
    from ai_weather_rainfall.accumulations import accumulated_to_window_total
    from ai_weather_rainfall.data_ecmwf import discover_forecast_files, open_forecast_file
    from ai_weather_rainfall.data_imerg import daily_observed_rainfall, discover_imerg_files, open_imerg_files
    from ai_weather_rainfall.regrid import align_observation_to_forecast_grid

    forecast_root = cfg.paths.aifs_raw if model == "aifs" else cfg.paths.ifs_raw
    forecast_files = discover_forecast_files(forecast_root, model, year, cycle)
    imerg_files = discover_imerg_files(cfg.paths.imerg_raw, year)
    if not forecast_files or not imerg_files:
        return [{"model": model, "year": year, "cycle": cycle, "metric_type": "missing_data", "lead_hour": -1}]

    observed_daily = daily_observed_rainfall(open_imerg_files(imerg_files))
    rows: list[dict[str, object]] = []
    for forecast_file in forecast_files:
        forecast_ds = open_forecast_file(forecast_file)
        rainfall = accumulated_to_window_total(forecast_ds["tp"], cfg.forecast.accumulation_hours)
        for lead_hour in cfg.forecast.lead_hours:
            if lead_hour not in rainfall["lead_hour"]:
                continue
            forecast_slice = rainfall.sel(lead_hour=lead_hour)
            observed_slice = align_observation_to_forecast_grid(observed_daily.mean("time"), forecast_slice)
            scores = score_forecast_against_observation(forecast_slice, observed_slice, cfg.rainfall.thresholds_mm_24h)
            rows.extend(metric_rows(model, year, cycle, lead_hour, scores))
    return rows
```

Add report helper:

```python
def _report_text(cfg: BenchmarkConfig, rows: list[dict[str, object]], metrics_path: Path) -> str:
    return (
        "# Rainfall Benchmark Report\n\n"
        "This report evaluates ECMWF AIFS and ECMWF IFS rainfall forecasts against NASA GPM IMERG observations.\n\n"
        "**Important:** This is a research and decision-support benchmark, not an official weather warning service.\n\n"
        f"- Region: {cfg.region.name} ({cfg.region.lon_min}..{cfg.region.lon_max}E, {cfg.region.lat_min}..{cfg.region.lat_max}N)\n"
        f"- Years: {cfg.period.years}\n"
        f"- Primary AIFS-vs-IFS comparison years: {cfg.period.primary_comparison_years}\n"
        f"- Metric rows written: {len(rows)}\n"
        f"- Metrics CSV: `{metrics_path}`\n"
    )
```

- [ ] **Step 6: Run full test suite**

Run:

```bash
PYTHONPATH=src pytest -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/ai_weather_rainfall/pipeline.py src/ai_weather_rainfall/data_ecmwf.py src/ai_weather_rainfall/data_imerg.py tests/test_pipeline_smoke.py
git commit -m "feat: run available rainfall benchmark files"
```

## Task 10: Plots And Report Figures

**Files:**

- Create: `src/ai_weather_rainfall/plots.py`
- Modify: `src/ai_weather_rainfall/report.py`
- Test: add lightweight plot smoke test in `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write plotting smoke test**

Append to `tests/test_pipeline_smoke.py`:

```python
from ai_weather_rainfall.plots import plot_lead_metric


def test_plot_lead_metric_creates_png(tmp_path):
    frame = pd.DataFrame(
        [
            {"model": "aifs", "lead_hour": 24, "metric_type": "continuous", "mae": 10.0},
            {"model": "ifs", "lead_hour": 24, "metric_type": "continuous", "mae": 12.0},
        ]
    )
    output = tmp_path / "lead_mae.png"

    plot_lead_metric(frame, metric="mae", output_path=output)

    assert output.exists()
    assert output.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_pipeline_smoke.py::test_plot_lead_metric_creates_png -v
```

Expected: import failure because `plots.py` does not exist.

- [ ] **Step 3: Implement lead metric plot**

Write `src/ai_weather_rainfall/plots.py`:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_lead_metric(frame: pd.DataFrame, metric: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    continuous = frame[frame["metric_type"] == "continuous"]
    for model, group in continuous.groupby("model"):
        summary = group.groupby("lead_hour", as_index=False)[metric].mean()
        ax.plot(summary["lead_hour"], summary[metric], marker="o", label=model.upper())
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
```

- [ ] **Step 4: Create report module**

Write `src/ai_weather_rainfall/report.py`:

```python
from pathlib import Path

import pandas as pd


def write_report(metrics: pd.DataFrame, figures: list[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_lines = "\n".join(f"![{path.stem}](../figures/{path.name})" for path in figures)
    text = (
        "# Rainfall Benchmark Report\n\n"
        "This report evaluates ECMWF AIFS and ECMWF IFS rainfall forecasts against NASA GPM IMERG observations.\n\n"
        "**Important:** This is a research and decision-support benchmark, not an official weather warning service.\n\n"
        f"Metric rows: {len(metrics)}\n\n"
        "## Figures\n\n"
        f"{figure_lines}\n"
    )
    output_path.write_text(text, encoding="utf-8")
    return output_path
```

- [ ] **Step 5: Verify plot test passes**

Run:

```bash
PYTHONPATH=src pytest tests/test_pipeline_smoke.py::test_plot_lead_metric_creates_png -v
```

Expected: test passes.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/ai_weather_rainfall/plots.py src/ai_weather_rainfall/report.py tests/test_pipeline_smoke.py
git commit -m "feat: add benchmark plots and report writer"
```

## Task 11: Notebook

**Files:**

- Create: `notebooks/01_extreme_rainfall_benchmark.ipynb`

- [ ] **Step 1: Create notebook with these cells**

The notebook must contain:

1. Title markdown cell: `# Extreme Rainfall Benchmark: AIFS vs IFS`
2. Disclaimer markdown cell: `Research and decision support only; not an official warning product.`
3. Setup code cell:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src"))

from ai_weather_rainfall.config import load_config
from ai_weather_rainfall.pipeline import run_benchmark
```

4. Config preview cell:

```python
cfg = load_config(Path("../configs/rainfall_benchmark.yaml"))
cfg
```

5. Pipeline run cell:

```python
report = run_benchmark(Path("../configs/rainfall_benchmark.yaml"))
report
```

6. Metrics preview cell:

```python
import pandas as pd

metrics = pd.read_csv(Path("../outputs/rainfall_benchmark/metrics/summary_metrics.csv"))
metrics.head()
```

- [ ] **Step 2: Verify notebook opens**

Run:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_extreme_rainfall_benchmark.ipynb --output executed_rainfall_benchmark.ipynb
```

Expected: executed notebook is created without exceptions.

- [ ] **Step 3: Commit**

Run:

```bash
git add notebooks/01_extreme_rainfall_benchmark.ipynb
git commit -m "docs: add rainfall benchmark notebook"
```

## Task 12: Final Verification And Acceptance

**Files:**

- Modify: `README.md`
- Use all tests and one CLI smoke run

- [ ] **Step 1: Run full automated tests**

Run:

```bash
PYTHONPATH=src pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run CLI smoke test**

Run:

```bash
PYTHONPATH=src python -m ai_weather_rainfall.cli --config configs/rainfall_benchmark.yaml
```

Expected:

```txt
report=outputs/rainfall_benchmark/reports/rainfall_benchmark_report.md
```

- [ ] **Step 3: Inspect generated files**

Run:

```bash
test -f outputs/rainfall_benchmark/metrics/summary_metrics.csv
test -f outputs/rainfall_benchmark/reports/rainfall_benchmark_report.md
```

Expected: both commands exit with status 0.

- [ ] **Step 4: Update README with acceptance criteria**

Add:

```markdown
## Acceptance Criteria

- The CLI can run from `configs/rainfall_benchmark.yaml`.
- The package can load AIFS, IFS, and IMERG files from configured directories.
- The benchmark computes 24-hour rainfall accumulations for configured lead times.
- The benchmark scores 50 mm and 100 mm 24-hour rainfall events.
- The output report states that the project is not an official warning service.
- The 2024-2025 seasons are the primary AIFS-vs-IFS comparison period.
- The 2023 season is used only as IFS-vs-IMERG background unless valid AIFS backfill files are supplied.
```

- [ ] **Step 5: Commit**

Run:

```bash
git add README.md outputs/rainfall_benchmark/metrics/summary_metrics.csv outputs/rainfall_benchmark/reports/rainfall_benchmark_report.md
git commit -m "docs: record rainfall benchmark acceptance criteria"
```

## Assumptions

- The user will provide historical ECMWF archive access or pre-downloaded AIFS and IFS files for the benchmark period.
- AIFS Single is the AI forecast model for version 1; AIFS Ensemble is excluded because it starts later and adds probability-processing complexity.
- IMERG is the observation reference for the first version, with gauge-station verification left for a later project.
- The default region is a bounding box rather than an administrative shapefile.
- The first version prioritizes reproducible evaluation over real-time alerting.
- All project Markdown and code files use UTF-8 without BOM.

## Completion Criteria

The implementation is complete when:

- `PYTHONPATH=src pytest -v` passes.
- The CLI run produces `summary_metrics.csv` and `rainfall_benchmark_report.md`.
- The notebook executes successfully on the same configuration.
- The report includes metrics for AIFS and IFS for every available 2024-2025 forecast file.
- Missing 2023 AIFS data is represented as a documented data-availability rule, not as a failed run.
