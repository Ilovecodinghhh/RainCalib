import numpy as np
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
