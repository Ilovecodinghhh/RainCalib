from datetime import date

import numpy as np
import xarray as xr

from ai_weather_rainfall.data_gfs import (
    discover_gfs_forecast_files,
    gfs_file_date,
    normalize_gfs_forecast_dataset,
)


def test_discover_gfs_forecast_files_filters_cycle_and_dates(tmp_path):
    included = tmp_path / "gfs_20250501_00_tp_steps.grib2"
    other_cycle = tmp_path / "gfs_20250501_12_tp_steps.grib2"
    out_of_period = tmp_path / "gfs_20250502_00_tp_steps.grib2"
    index_file = tmp_path / "gfs_20250501_00_tp_steps.grib2.abc.idx"
    included.write_bytes(b"GRIB")
    other_cycle.write_bytes(b"GRIB")
    out_of_period.write_bytes(b"GRIB")
    index_file.write_bytes(b"index")

    files = discover_gfs_forecast_files(
        tmp_path,
        year=2025,
        cycle="00",
        allowed_dates={date(2025, 5, 1)},
    )

    assert files == [included]


def test_gfs_file_date():
    assert gfs_file_date(PathLike("gfs_20250901_12_tp_steps.grib2")) == date(2025, 9, 1)


def test_normalize_gfs_forecast_dataset_renames_apcp_and_step():
    ds = xr.Dataset(
        {
            "unknown": (
                ("step", "lat", "lon"),
                np.ones((2, 1, 1)),
                {"GRIB_shortName": "APCP", "units": "kg m**-2"},
            )
        },
        coords={
            "step": [np.timedelta64(24, "h"), np.timedelta64(48, "h")],
            "lat": [30.0],
            "lon": [115.0],
        },
    )

    normalized = normalize_gfs_forecast_dataset(ds)

    assert "lead_hour" in normalized.dims
    assert "tp" in normalized
    assert normalized["lead_hour"].values.tolist() == [24, 48]
    assert normalized["tp"].attrs["units"] == "mm"


class PathLike:
    def __init__(self, name: str):
        self.name = name
        self.stem = name.rsplit(".", 1)[0]
