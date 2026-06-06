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
    assert aligned.item() == 4.0
