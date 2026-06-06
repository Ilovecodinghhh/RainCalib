from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr


GFS_PRECIP_VARIABLES = {"tp", "apcp", "total_precipitation", "totalprecipitation", "precip"}


def discover_gfs_forecast_files(
    root: Path,
    year: int,
    cycle: str,
    allowed_dates: set[date] | None = None,
) -> list[Path]:
    candidates = sorted(root.glob(f"**/gfs_{year}*_{int(cycle):02d}_tp_steps.grib2"))
    return [
        path
        for path in candidates
        if path.suffix.lower() in {".grib", ".grib1", ".grib2"}
        if allowed_dates is None or gfs_file_date(path) in allowed_dates
    ]


def gfs_file_date(path: Path) -> date | None:
    match = re.search(r"gfs_(20\d{6})_(\d{2})_", path.stem)
    if not match:
        return None
    return date.fromisoformat(
        f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:8]}"
    )


def open_gfs_forecast_file(path: Path) -> xr.Dataset:
    suffixes = "".join(path.suffixes).lower()
    if ".grib" in suffixes:
        return normalize_gfs_forecast_dataset(xr.open_dataset(path, engine="cfgrib"))
    if path.suffix.lower() in {".nc", ".nc4", ".netcdf"}:
        return normalize_gfs_forecast_dataset(xr.open_dataset(path))
    raise ValueError(f"Unsupported GFS forecast file type: {path}")


def normalize_gfs_forecast_dataset(dataset: xr.Dataset) -> xr.Dataset:
    ds = dataset.copy()
    rename_coords = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_coords["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_coords["lon"] = "longitude"
    if rename_coords:
        ds = ds.rename(rename_coords)
    if "step" in ds.coords and "lead_hour" not in ds.coords:
        lead_hours = (ds["step"].values / np.timedelta64(1, "h")).astype(int)
        ds = ds.assign_coords(lead_hour=("step", lead_hours)).swap_dims({"step": "lead_hour"})
        ds = ds.drop_vars("step")
    if "valid_time" in ds.coords and "time" in ds.coords:
        ds = ds.drop_vars("valid_time")
    if "tp" not in ds:
        precip_name = _precipitation_variable_name(ds)
        ds = ds.rename({precip_name: "tp"})
    ds["tp"] = _to_millimeters(ds["tp"])
    return ds


def _precipitation_variable_name(dataset: xr.Dataset) -> str:
    for name, data in dataset.data_vars.items():
        if name.lower() in GFS_PRECIP_VARIABLES:
            return name
        short_name = str(data.attrs.get("GRIB_shortName", "")).lower()
        name_attr = str(data.attrs.get("GRIB_name", "")).lower()
        if short_name == "tp" or short_name == "apcp" or "precipitation" in name_attr:
            return name
    raise KeyError("GFS forecast dataset must contain APCP/total precipitation")


def _to_millimeters(data: xr.DataArray) -> xr.DataArray:
    units = str(data.attrs.get("units", "")).lower()
    if units in {"m", "metre", "metres", "meter", "meters"}:
        converted = data * 1000.0
    else:
        converted = data
    converted.name = "tp"
    converted.attrs["units"] = "mm"
    return converted.where(np.isfinite(converted))
