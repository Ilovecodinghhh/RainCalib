import numpy as np
import pandas as pd
import xarray as xr
from datetime import date

from ai_weather_rainfall.data_ecmwf import discover_forecast_files, normalize_forecast_dataset
from ai_weather_rainfall.data_imerg import discover_imerg_files
from ai_weather_rainfall.data_imerg import daily_observed_rainfall
from ai_weather_rainfall.pipeline import metric_rows, score_forecast_against_observation
from ai_weather_rainfall.plots import plot_lead_metric


def test_daily_observed_rainfall_from_half_hourly_data():
    times = pd.date_range("2024-06-01", periods=48, freq="30min")
    da = xr.DataArray(
        np.ones((48, 2, 2)),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": [30.0, 30.1], "longitude": [115.0, 115.1]},
        name="precipitation",
        attrs={"units": "mm/hr"},
    )

    daily = daily_observed_rainfall(da)

    assert daily.sizes["time"] == 1
    assert float(daily.isel(time=0, latitude=0, longitude=0)) == 24.0


def test_normalize_forecast_dataset_renames_tp():
    ds = xr.Dataset(
        {"tp": (("step", "latitude", "longitude"), np.ones((2, 1, 1)), {"units": "mm"})},
        coords={
            "step": [np.timedelta64(6, "h"), np.timedelta64(12, "h")],
            "latitude": [30.0],
            "longitude": [115.0],
        },
    )

    normalized = normalize_forecast_dataset(ds)

    assert "lead_hour" in normalized.dims
    assert "tp" in normalized
    assert normalized["lead_hour"].values.tolist() == [6, 12]


def test_discover_forecast_files_ignores_cfgrib_indexes(tmp_path):
    forecast_file = tmp_path / "ifs_20251001_00_tp_steps.grib2"
    other_cycle = tmp_path / "ifs_20251001_12_tp_steps.grib2"
    out_of_period = tmp_path / "ifs_20251002_00_tp_steps.grib2"
    index_file = tmp_path / "ifs_20251001_00_tp_steps.grib2.5b7b6.idx"
    forecast_file.write_bytes(b"GRIB")
    other_cycle.write_bytes(b"GRIB")
    out_of_period.write_bytes(b"GRIB")
    index_file.write_bytes(b"index")

    files = discover_forecast_files(
        tmp_path,
        model="ifs",
        year=2025,
        cycle="00",
        allowed_dates={date(2025, 10, 1)},
    )

    assert files == [forecast_file]


def test_discover_imerg_files_filters_allowed_dates(tmp_path):
    included = tmp_path / "3B-DAY-L.MS.MRG.3IMERG.20250501-S000000-E235959.V07B.nc4"
    excluded = tmp_path / "3B-DAY-L.MS.MRG.3IMERG.20251001-S000000-E235959.V07B.nc4"
    included.write_bytes(b"NETCDF")
    excluded.write_bytes(b"NETCDF")

    files = discover_imerg_files(tmp_path, year=2025, allowed_dates={date(2025, 5, 1)})

    assert files == [included]


def test_score_forecast_against_observation():
    forecast = xr.DataArray(
        [[60.0, 20.0], [30.0, 110.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [30.0, 31.0], "longitude": [115.0, 116.0]},
        name="forecast",
    )
    observed = xr.DataArray(
        [[70.0, 10.0], [60.0, 90.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [30.0, 31.0], "longitude": [115.0, 116.0]},
        name="observed",
    )

    scores = score_forecast_against_observation(forecast, observed, thresholds=[50.0])

    assert scores["continuous"]["mae"] == 17.5
    assert scores["categorical"][0]["hits"] == 2
    assert scores["categorical"][0]["misses"] == 1


def test_metric_rows_include_model_metadata():
    rows = metric_rows(
        model="aifs",
        year=2024,
        cycle="00",
        lead_hour=24,
        scores={"continuous": {"mae": 1.0}, "categorical": [{"threshold_mm": 50.0, "csi": 0.4}]},
    )

    assert rows[0]["model"] == "aifs"
    assert rows[0]["metric_type"] == "continuous"
    assert rows[1]["threshold_mm"] == 50.0


def test_plot_lead_metric_creates_png(tmp_path):
    frame = pd.DataFrame(
        [
            {"model": "aifs", "lead_hour": 24, "metric_type": "continuous", "mae": 10.0},
            {"model": "ifs", "lead_hour": 24, "metric_type": "continuous", "mae": 12.0},
        ]
    )
    output = tmp_path / "lead_mae.png"

    plot_lead_metric(frame, metric="mae", output_path=output)

    assert output.exists()
    assert output.stat().st_size > 0
