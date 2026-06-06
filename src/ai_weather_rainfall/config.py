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
    gfs_raw: Path | None = None


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
    if not cfg.rainfall.thresholds_mm_24h:
        raise ValueError("thresholds_mm_24h must not be empty")
