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
    lat_slice = (
        slice(bounds.lat_max, bounds.lat_min)
        if float(lat_values[0]) > float(lat_values[-1])
        else slice(bounds.lat_min, bounds.lat_max)
    )
    return data.sel({lon_name: slice(bounds.lon_min, bounds.lon_max), lat_name: lat_slice})
