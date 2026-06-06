import numpy as np
import xarray as xr

from ai_weather_rainfall.metrics import categorical_scores, continuous_scores


def test_categorical_scores():
    forecast = xr.DataArray([[60.0, 10.0], [80.0, 20.0]], dims=("y", "x"))
    observed = xr.DataArray([[70.0, 60.0], [10.0, 20.0]], dims=("y", "x"))

    scores = categorical_scores(forecast, observed, threshold_mm=50.0)

    assert scores["hits"] == 1
    assert scores["misses"] == 1
    assert scores["false_alarms"] == 1
    assert scores["correct_negatives"] == 1
    assert scores["pod"] == 0.5
    assert scores["far"] == 0.5
    assert scores["csi"] == 1 / 3


def test_continuous_scores():
    forecast = xr.DataArray([1.0, 3.0, np.nan], dims=("point",))
    observed = xr.DataArray([2.0, 1.0, 5.0], dims=("point",))

    scores = continuous_scores(forecast, observed)

    assert scores["mae"] == 1.5
    assert scores["bias"] == 0.5
    assert scores["rmse"] == np.sqrt(2.5)
