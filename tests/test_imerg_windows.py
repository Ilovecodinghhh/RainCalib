import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ai_weather_rainfall.imerg_windows import accumulate_window, forecast_window_start, granule_name, required_slots, read_granule


def rates(start):
    return xr.DataArray(np.ones((48, 2)), dims=['time','point'], coords={'time':pd.date_range(start, periods=48, freq='30min')}, attrs={'units':'mm/hr'})


def test_noon_crosses_year_and_excludes_end_boundary():
    start = forecast_window_start('2024-12-31', 12, 24)
    assert start == pd.Timestamp('2024-12-31T12:00')
    data = rates(start)
    data.values[24:] = 3
    assert accumulate_window(data, start).values.tolist() == [48,48]
    assert required_slots([start])[-1] == pd.Timestamp('2025-01-01T11:30')


@pytest.mark.parametrize('kind', ['missing','duplicate','shifted'])
def test_incomplete_windows_fail(kind):
    data = rates('2025-01-01')
    if kind == 'missing': data = data.isel(time=slice(0,47))
    elif kind == 'duplicate': data = data.assign_coords(time=list(data.time.values[:-1])+[data.time.values[-2]])
    else: data = data.assign_coords(time=data.time + np.timedelta64(30,'m'))
    with pytest.raises(ValueError,match='48'): accumulate_window(data,'2025-01-01')


def test_missing_pixel_is_not_treated_as_zero():
    data = rates('2025-01-01');data.values[0,0] = np.nan
    result = accumulate_window(data,'2025-01-01')
    assert np.isnan(result[0]) and result[1] == 24


def test_unknown_units_fail():
    data = rates('2025-01-01');data.attrs['units']='mm'
    with pytest.raises(ValueError,match='mm/hour'): accumulate_window(data,'2025-01-01')


def test_slot_deduplication_and_filename():
    assert len(required_slots(['2025-01-01', '2025-01-01T12:00'])) == 72
    assert 'S233000-E235959.1410.V07B' in granule_name('2025-01-01T23:30')


def test_granule_bounds_are_checked(tmp_path):
    start = pd.Timestamp('2025-01-01')
    ds = xr.Dataset({'precipitation': (('time','lat','lon'), np.ones((1,1,1))),
                     'time_bnds': (('time','nv'), np.array([[start.to_datetime64(), (start+pd.Timedelta(minutes=30)).to_datetime64()]]))},
                    coords={'time':[start], 'lat':[30.0], 'lon':[115.0]})
    ds.precipitation.attrs['units']='mm/hr'
    path=tmp_path/'granule.nc4';ds.to_netcdf(path)
    assert read_granule(path,start).shape == (1,1,1)
    with pytest.raises(ValueError,match='interval'): read_granule(path,start+pd.Timedelta(minutes=30))


def test_purge_noon_window_sharing_observations_with_next_split():
    from ai_weather_rainfall.imerg_windows import crosses_split_boundary
    assert crosses_split_boundary('2025-07-31T12:00', ['2025-07-31'])
    assert not crosses_split_boundary('2025-07-31T00:00', ['2025-07-31'])
    assert not crosses_split_boundary('2025-08-01T00:00', ['2025-07-31'])
