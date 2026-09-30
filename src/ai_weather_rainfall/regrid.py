import xarray as xr


def align_observation_to_forecast_grid(observed: xr.DataArray, forecast_template: xr.DataArray) -> xr.DataArray:
    lat_name = "latitude" if "latitude" in observed.coords else "lat"
    lon_name = "longitude" if "longitude" in observed.coords else "lon"
    target_lat = "latitude" if "latitude" in forecast_template.coords else "lat"
    target_lon = "longitude" if "longitude" in forecast_template.coords else "lon"
    source = observed
    if lat_name != target_lat or lon_name != target_lon:
        source = source.rename({lat_name: target_lat, lon_name: target_lon})
        lat_name = target_lat
        lon_name = target_lon
    aligned = source.interp(
        {lat_name: forecast_template[target_lat], lon_name: forecast_template[target_lon]},
        method="linear",
    )
    aligned = aligned.transpose(*forecast_template.dims)
    aligned.name = "observed_on_forecast_grid"
    aligned.attrs["units"] = observed.attrs.get("units", "mm")
    return aligned
