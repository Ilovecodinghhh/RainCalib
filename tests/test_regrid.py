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


def test_non_square_lon_lat_source_matches_forecast_pixel_order():
    observed = xr.DataArray([[11., 21.], [12., 22.], [13., 23.]], dims=('lon','lat'),
                            coords={'lat':[30.,31.], 'lon':[110.,111.,112.]})
    forecast = xr.DataArray(np.zeros((2,3)), dims=('latitude','longitude'),
                            coords={'latitude':[30.,31.], 'longitude':[110.,111.,112.]})
    aligned = align_observation_to_forecast_grid(observed, forecast)
    assert aligned.dims == forecast.dims
    np.testing.assert_array_equal(aligned.values.reshape(-1), [11,12,13,21,22,23])


def test_correlation_uses_named_spatial_dimensions():
    from ai_weather_rainfall.metrics import continuous_scores
    field = xr.DataArray([[1., 2., 4.], [8.,16.,32.]], dims=('latitude','longitude'),
                         coords={'latitude':[30.,31.], 'longitude':[110.,111.,112.]})
    scores = continuous_scores(field, field.transpose('longitude','latitude'))
    assert scores['rmse'] == 0.0
    assert abs(scores['pearson_r'] - 1.0) < 1e-12
