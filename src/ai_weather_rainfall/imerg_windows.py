"""Exact UTC windows for the IMERG Late V07 half-hourly product."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

PRODUCT = 'GPM_3IMERGHHL.07'


def forecast_window_start(run_date, cycle, lead_hour):
    cycle, lead = float(cycle), float(lead_hour)
    if cycle not in (0, 12) or not lead.is_integer() or lead < 24 or lead % 24:
        raise ValueError('Expected 00/12 UTC and a 24-hour window ending at a daily lead')
    return pd.Timestamp(run_date).normalize() + pd.Timedelta(hours=cycle + lead - 24)


def granule_name(start):
    start = pd.Timestamp(start)
    if start.second or start.microsecond or start.minute not in (0, 30):
        raise ValueError('Granule start must be on a half-hour boundary')
    end = start + pd.Timedelta(minutes=30) - pd.Timedelta(seconds=1)
    return (f'3B-HHR-L.MS.MRG.3IMERG.{start:%Y%m%d}-S{start:%H%M%S}'
            f'-E{end:%H%M%S}.{start.hour * 60 + start.minute:04d}.V07B.HDF5')


def granule_url(start):
    start = pd.Timestamp(start)
    return (f'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/hyrax/GPM_L3/{PRODUCT}/'
            f'{start:%Y}/{start.dayofyear:03d}/{granule_name(start)}')


def required_slots(starts):
    return pd.DatetimeIndex(sorted({slot for start in starts for slot in
                                   pd.date_range(start, periods=48, freq='30min')}))


def cache_path(root, start):
    return Path(root) / f'{pd.Timestamp(start):%Y}' / (granule_name(start) + '.nc4')


def read_granule(path, expected_start):
    # Read only required variables; bypass expensive xarray decoding of large global metadata.
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        if 'time_bnds' not in ds.variables or len(ds.dimensions['time']) != 1:
            raise ValueError(f'Missing interval bounds: {path}')
        time = ds.variables['time']
        bounds = ds.variables['time_bnds']
        dates = netCDF4.num2date(bounds[:].reshape(-1),
                                units=getattr(bounds, 'units', time.units),
                                calendar=getattr(bounds, 'calendar', getattr(time, 'calendar', 'standard')))
        expected = pd.Timestamp(expected_start)
        if len(dates) != 2 or pd.Timestamp(str(dates[0])) != expected or pd.Timestamp(str(dates[1])) != expected + pd.Timedelta(minutes=30):
            raise ValueError(f'Incorrect half-hour interval: {path}')
        variable = ds.variables['precipitation']
        units = str(getattr(variable, 'units', '')).lower().replace(' ', '')
        if units not in ('mm/hr', 'mm/hour', 'mmh-1'):
            raise ValueError(f'Expected precipitation rate in mm/hour, got {units}')
        if set(variable.dimensions) != {'time', 'lat', 'lon'}:
            raise ValueError(f'Unexpected precipitation dimensions: {variable.dimensions}')
        values = np.ma.asarray(variable[:], dtype='float32').filled(np.nan)
        values = np.where(np.isfinite(values) & (values >= 0), values, np.nan)
        coords = {name: np.asarray(ds.variables[name][:]) for name in ('lat', 'lon')}
        coords['time'] = [expected.to_datetime64()]
        return xr.DataArray(values, dims=variable.dimensions, coords=coords,
                            attrs={'units': 'mm/hr'}, name='precipitation')


def accumulate_window(rates, start):
    """Require exactly 48 unique ordered slots; missing pixels remain missing."""
    start = pd.Timestamp(start)
    expected = pd.date_range(start, periods=48, freq='30min')
    actual = pd.DatetimeIndex(rates.time.values)
    if not actual.equals(expected):
        raise ValueError('Window requires exactly 48 consecutive half-hour slots in UTC')
    units = str(rates.attrs.get('units', '')).lower().replace(' ', '')
    if units not in ('mm/hr', 'mm/hour', 'mmh-1'):
        raise ValueError('Expected mm/hour precipitation rates')
    result = (rates.where(np.isfinite(rates) & (rates >= 0)) * 0.5).sum('time', skipna=False)
    result.name = 'observed_rainfall_24h_mm'
    result.attrs = {'units': 'mm', 'window_start_utc': start.isoformat(),
                    'window_end_utc': (start + pd.Timedelta(hours=24)).isoformat(),
                    'source_product': PRODUCT, 'interval_count': 48}
    return result


def load_window(root, start):
    slots = pd.date_range(start, periods=48, freq='30min')
    fields = [read_granule(cache_path(root, slot), slot) for slot in slots]
    return accumulate_window(xr.concat(fields, dim='time', join='exact'), start)


def crosses_split_boundary(start, boundary_dates):
    """A boundary date ends at the following midnight; intervals are half-open."""
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(hours=24)
    return any(start < pd.Timestamp(day) + pd.Timedelta(days=1) < end for day in boundary_dates)
