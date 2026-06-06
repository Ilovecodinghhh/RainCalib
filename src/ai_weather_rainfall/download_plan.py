from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from ai_weather_rainfall.config import BenchmarkConfig


ECMWF_MODEL_NAMES = {
    "aifs": "aifs-single",
    "ifs": "ifs",
}


def configured_dates(cfg: BenchmarkConfig) -> list[date]:
    dates: list[date] = []
    for year in cfg.period.years:
        dates.extend(season_dates(year, cfg.period.season_start, cfg.period.season_end))
    return dates


def season_dates(year: int, season_start: str, season_end: str) -> list[date]:
    start_month, start_day = _month_day(season_start)
    end_month, end_day = _month_day(season_end)
    start = date(year, start_month, start_day)
    end_year = year if (end_month, end_day) >= (start_month, start_day) else year + 1
    end = date(end_year, end_month, end_day)
    return list(_date_range(start, end))


def required_lead_steps(lead_hours: list[int], accumulation_hours: int) -> list[int]:
    steps = set()
    for lead in lead_hours:
        steps.add(int(lead))
        start = int(lead) - int(accumulation_hours)
        if start > 0:
            steps.add(start)
    return sorted(steps)


def active_forecast_models(cfg: BenchmarkConfig, year: int) -> list[str]:
    if year in cfg.period.background_baseline_years:
        return ["ifs"]
    return cfg.forecast.models


def ecmwf_open_data_model(model: str) -> str:
    try:
        return ECMWF_MODEL_NAMES[model]
    except KeyError as exc:
        raise ValueError(f"Unsupported ECMWF model label: {model}") from exc


def forecast_target_path(cfg: BenchmarkConfig, model: str, run_date: date, cycle: str) -> Path:
    if model == "aifs":
        root = cfg.paths.aifs_raw
    elif model == "ifs":
        root = cfg.paths.ifs_raw
    elif model == "gfs":
        if cfg.paths.gfs_raw is None:
            raise ValueError("paths.gfs_raw is required for GFS downloads")
        root = cfg.paths.gfs_raw
        return root / f"gfs_{run_date:%Y%m%d}_{int(cycle):02d}_tp_steps.grib2"
    else:
        raise ValueError(f"Unsupported forecast model: {model}")
    return root / f"{model}_{run_date:%Y%m%d}_{int(cycle):02d}_tp_steps.grib2"


def gfs_aws_grib_url(run_date: date, cycle: str, step: int) -> str:
    yyyymmdd = f"{run_date:%Y%m%d}"
    hh = f"{int(cycle):02d}"
    fff = f"{int(step):03d}"
    return (
        "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
        f"gfs.{yyyymmdd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{fff}"
    )


def gfs_aws_index_url(run_date: date, cycle: str, step: int) -> str:
    return gfs_aws_grib_url(run_date, cycle, step) + ".idx"


def imerg_url(run_date: date) -> str:
    yyyymmdd = f"{run_date:%Y%m%d}"
    return (
        "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGDL.07/"
        f"{run_date:%Y}/{run_date:%m}/"
        f"3B-DAY-L.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.V07B.nc4"
    )


def imerg_target_path(cfg: BenchmarkConfig, run_date: date) -> Path:
    yyyymmdd = f"{run_date:%Y%m%d}"
    return cfg.paths.imerg_raw / f"3B-DAY-L.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.V07B.nc4"


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value.replace("/", "-"))


def filter_dates(
    dates: list[date],
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    exact_dates: list[date] | None = None,
    max_days: int | None = None,
) -> list[date]:
    selected = sorted(set(exact_dates or dates))
    if start_date is not None:
        selected = [item for item in selected if item >= start_date]
    if end_date is not None:
        selected = [item for item in selected if item <= end_date]
    if max_days is not None:
        selected = selected[:max_days]
    return selected


def _month_day(value: str) -> tuple[int, int]:
    month, day = value.split("-", 1)
    return int(month), int(day)


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)
