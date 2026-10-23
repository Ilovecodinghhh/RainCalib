import numpy as np
import pytest
import xarray as xr

from ai_weather_rainfall.data_ecmwf import normalize_forecast_dataset, open_forecast_file
from ai_weather_rainfall.data_gfs import normalize_gfs_forecast_dataset


@pytest.mark.parametrize("normalize", [normalize_forecast_dataset, normalize_gfs_forecast_dataset])
@pytest.mark.parametrize("units, expected", [
    ("m", 1000.0), (" M ", 1000.0), ("metres", 1000.0),
    ("mm", 1.0), ("kg m**-2", 1.0), ("kg m-2", 1.0), ("kg/m^2", 1.0),
])
def test_forecast_units_are_converted_without_changing_source(normalize, units, expected):
    source = xr.Dataset({"tp": ("point", [1.0, np.nan, np.inf], {"units": units})})

    result = normalize(source)

    np.testing.assert_allclose(result.tp.values, [expected, np.nan, np.nan])
    assert result.tp.attrs["units"] == "mm"
    assert source.tp.attrs["units"] == units
    assert source.tp.values[0] == 1.0


@pytest.mark.parametrize("normalize", [normalize_forecast_dataset, normalize_gfs_forecast_dataset])
@pytest.mark.parametrize("units", [None, "", "mm/hr", "kg m-2 s-1", "inches"])
def test_unknown_or_rate_units_are_not_treated_as_accumulated_rain(normalize, units):
    attrs = {} if units is None else {"units": units}
    source = xr.Dataset({"tp": ("point", [1.0], attrs)})

    with pytest.raises(ValueError, match="precipitation units"):
        normalize(source)


def test_ecmwf_nc4_file_can_be_opened(tmp_path):
    path = tmp_path / "ifs_20250501_00_tp_steps.nc4"
    xr.Dataset({"tp": ("point", [0.012], {"units": "m"})}).to_netcdf(path)

    result = open_forecast_file(path)
    try:
        assert result.tp.item() == pytest.approx(12.0)
    finally:
        result.close()
