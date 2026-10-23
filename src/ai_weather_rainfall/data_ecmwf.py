import re
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from ai_weather_rainfall.accumulations import precipitation_to_millimeters


def discover_forecast_files(
    root: Path,
    model: str,
    year: int,
    cycle: str,
    allowed_dates: set[date] | None = None,
) -> list[Path]:
    candidates = sorted(root.glob(f"**/*{model}*{year}*"))
    return [
        path
        for path in candidates
        if _is_supported_forecast_file(path) and _matches_cycle_token(path, cycle)
        if allowed_dates is None or forecast_file_date(path) in allowed_dates
    ]


def forecast_file_date(path: Path) -> date | None:
    match = re.search(r"(20\d{6})", path.stem)
    if not match:
        return None
    return date.fromisoformat(
        f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:8]}"
    )


def open_forecast_file(path: Path) -> xr.Dataset:
    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() in {".nc", ".nc4", ".netcdf"}:
        return normalize_forecast_dataset(xr.open_dataset(path))
    if ".grib" in suffixes:
        return normalize_forecast_dataset(xr.open_dataset(path, engine="cfgrib"))
    raise ValueError(f"Unsupported forecast file type: {path}")


def normalize_forecast_dataset(dataset: xr.Dataset) -> xr.Dataset:
    ds = dataset.copy()
    if "step" in ds.coords and "lead_hour" not in ds.coords:
        lead_hours = (ds["step"].values / np.timedelta64(1, "h")).astype(int)
        ds = ds.assign_coords(lead_hour=("step", lead_hours)).swap_dims({"step": "lead_hour"})
        ds = ds.drop_vars("step")
    if "valid_time" in ds.coords and "time" in ds.coords:
        ds = ds.drop_vars("valid_time")
    if "tp" not in ds:
        candidates = [name for name in ds.data_vars if name.lower() in {"total_precipitation", "precip"}]
        if candidates:
            ds = ds.rename({candidates[0]: "tp"})
    if "tp" not in ds:
        raise KeyError("Forecast dataset must contain total precipitation as tp")
    ds["tp"] = precipitation_to_millimeters(ds["tp"])
    return ds


def describe_realtime_open_data_fetch() -> str:
    return (
        "Use ecmwf-opendata for current or rolling ECMWF Open Data retrieval. "
        "Historical AIFS/IFS benchmark files should be retrieved from ECMWF archive access "
        "or another licensed archive, then placed under the configured raw data directories."
    )


def _is_grib_file(path: Path) -> bool:
    return path.suffix.lower() in {".grib", ".grib1", ".grib2"}


def _is_supported_forecast_file(path: Path) -> bool:
    return path.suffix.lower() in {".nc", ".nc4", ".netcdf"} or _is_grib_file(path)


def _matches_cycle_token(path: Path, cycle: str) -> bool:
    normalized = path.stem.replace("-", "_").split("_")
    return cycle in normalized
