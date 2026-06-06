import re
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

PRECIPITATION_VARIABLES = ("precipitation", "precipitationCal", "precipitationCalibrated")


def discover_imerg_files(root: Path, year: int, allowed_dates: set[date] | None = None) -> list[Path]:
    patterns = [f"**/*{year}*.nc", f"**/*{year}*.nc4", f"**/*{year}*.HDF5"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    if allowed_dates is None:
        return files
    return [path for path in files if imerg_file_date(path) in allowed_dates]


def imerg_file_date(path: Path) -> date | None:
    match = re.search(r"(20\d{6})", path.stem)
    if not match:
        return None
    return date.fromisoformat(
        f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:8]}"
    )


def open_imerg_files(paths: list[Path]) -> xr.DataArray:
    if not paths:
        raise FileNotFoundError("No IMERG files were provided")
    if len(paths) == 1:
        dataset = xr.open_dataset(paths[0])
    else:
        dataset = xr.combine_by_coords(
            [xr.open_dataset(path) for path in paths],
            combine_attrs="override",
        )
    return precipitation_from_dataset(dataset)


def precipitation_from_dataset(dataset: xr.Dataset) -> xr.DataArray:
    for name in PRECIPITATION_VARIABLES:
        if name in dataset:
            return dataset[name]
    raise KeyError("IMERG precipitation variable not found")


def daily_observed_rainfall(precipitation: xr.DataArray) -> xr.DataArray:
    if "time" not in precipitation.dims:
        raise ValueError("IMERG precipitation must have a time dimension")
    units = str(precipitation.attrs.get("units", "")).lower()
    if "mm/hr" in units or "mm h-1" in units or "mm/hour" in units:
        interval_hours = _median_interval_hours(precipitation)
        rainfall = precipitation * interval_hours
    else:
        rainfall = precipitation
    daily = rainfall.resample(time="1D").sum(skipna=True)
    daily.name = "observed_rainfall_24h_mm"
    daily.attrs["units"] = "mm"
    return daily


def _median_interval_hours(data: xr.DataArray) -> float:
    time_values = data["time"].values
    if len(time_values) < 2:
        return 0.5
    deltas = np.diff(time_values).astype("timedelta64[s]").astype(float)
    return float(np.nanmedian(deltas)) / 3600.0
