"""Download restartable IMERG Late V07 regional half-hourly subsets."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import subprocess
import time
import pandas as pd
from ai_weather_rainfall.imerg_windows import cache_path, forecast_window_start, granule_url, read_granule, required_slots

ROOT = Path('outputs/rainfall_benchmark/raw/imerg_halfhourly_late_v07')
# 0.1-degree grid, padded by 0.5 degrees around the forecast domain for interpolation.
SUBSET = '?precipitation[0:0][2875:3035][1135:1255],lat[1135:1255],lon[2875:3035],time,time_bnds'


def download(slot, root):
    target = cache_path(root, slot)
    if target.exists():
        return slot, target, 'exists'
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix('.part.nc4')
    command = ['curl', '-sS', '-L', '-g', '--fail', '--connect-timeout', '20', '--max-time', '90',
               '--retry', '5', '--retry-all-errors', '--retry-delay', '10', '--netrc-file', '.earthdata.netrc',
               '-b', '.urs_cookies', '-o', str(part), '-w', '%{http_code}', granule_url(slot) + '.nc4' + SUBSET]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode and result.stdout.strip() in ('500', '502', '503', '504'):
        raw = target.with_suffix('.raw.hdf5')
        command[command.index('-o') + 1] = str(raw)
        command[-1] = granule_url(slot).replace('gpm1.gesdisc.eosdis.nasa.gov/opendap/hyrax', 'data.gesdisc.earthdata.nasa.gov/data')
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip() == '200':
            return slot, raw, 'raw_downloaded'
    if result.returncode or result.stdout.strip() != '200':
        raise RuntimeError(f'{slot}: HTTP {result.stdout.strip()}, curl exit {result.returncode}')
    # Validation is performed serially by the caller (netCDF4 is not thread-safe).
    return slot, part, 'downloaded'


def subset_raw_file(raw, output):
    """Fallback for unavailable OPeNDAP subsets; only the regional subset is retained."""
    import netCDF4
    with netCDF4.Dataset(raw) as source, netCDF4.Dataset(output, 'w') as target:
        grid = source.groups['Grid']
        for name, size in [('time', 1), ('lon', 161), ('lat', 121), ('nv', 2)]:
            target.createDimension(name, size)
        selection = {'precipitation': (slice(None), slice(2875,3036), slice(1135,1256)),
                     'lon': slice(2875,3036), 'lat': slice(1135,1256),
                     'time': slice(None), 'time_bnds': slice(None)}
        for name, index in selection.items():
            variable = grid.variables[name]
            fill = getattr(variable, '_FillValue', None)
            out = target.createVariable(name, variable.dtype, variable.dimensions, fill_value=fill, zlib=True)
            for attribute in ('units', 'calendar'):
                if attribute in variable.ncattrs():
                    out.setncattr(attribute, variable.getncattr(attribute))
            out[:] = variable[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=Path('outputs/rainfall_benchmark/ml/features/raincalib_features.parquet'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-slots', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('workers must be between 1 and 8')
    keys = pd.read_parquet(args.features, columns=['run_date', 'cycle', 'lead_hour']).drop_duplicates()
    starts = [forecast_window_start(r.run_date, r.cycle, r.lead_hour) for r in keys.itertuples(index=False)]
    slots = required_slots(starts)
    args.root.mkdir(parents=True, exist_ok=True)
    plan = dict(product='GPM_3IMERGHHL.07', fields=len(keys), windows=len(set(starts)), slots=len(slots),
                first=str(slots[0]), last=str(slots[-1]), source_features=str(args.features), subset=SUBSET)
    (args.root / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    slots = slots[:args.max_slots] if args.max_slots else slots
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0, len(slots), 48):
            batch = slots[offset:offset + 48]
            for j, (slot, path, status) in enumerate(pool.map(lambda t: download(t, args.root), batch), 1):
                raw_path = None
                if status == 'raw_downloaded':
                    raw_path = path
                    path = cache_path(args.root, slot).with_suffix('.part.nc4')
                    subset_raw_file(raw_path, path)
                read_granule(path, slot)
                if status in ('downloaded', 'raw_downloaded'):
                    path.replace(cache_path(args.root, slot))
                if raw_path is not None:
                    raw_path.unlink()
                i = offset + j
            progress = dict(validated=i, selected=len(slots), elapsed_seconds=round(time.monotonic()-started, 1), last=str(slot))
            (args.root / 'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
            print(json.dumps(progress), flush=True)



if __name__ == '__main__':
    main()
