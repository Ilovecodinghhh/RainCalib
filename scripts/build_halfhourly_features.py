"""Rebuild forecast pixels and exact 00/12 UTC 24h IMERG labels from raw inputs."""
from functools import lru_cache
from pathlib import Path
import argparse
import json
import pandas as pd
import yaml
from ai_weather_rainfall.accumulations import accumulated_to_window_total
from ai_weather_rainfall.config import load_config
from ai_weather_rainfall.data_gfs import open_gfs_forecast_file
from ai_weather_rainfall.imerg_windows import forecast_window_start, load_window, crosses_split_boundary
from ai_weather_rainfall.raincalib import load_raincalib_config, split_for_date, add_time_features
from ai_weather_rainfall.regions import bounds_from_config, subset_region
from ai_weather_rainfall.regrid import align_observation_to_forecast_grid
from build_raincalib_features import _flatten_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/rainfall_benchmark/ml/features/raincalib_features.parquet'))
    parser.add_argument('--root', type=Path, default=Path('outputs/rainfall_benchmark/raw/imerg_halfhourly_late_v07'))
    parser.add_argument('--config', type=Path, default=Path('configs/raincalib_halfhourly.yaml'))
    parser.add_argument('--experiment-config', type=Path, default=Path('configs/raincalib_experiments_halfhourly.yaml'))
    args = parser.parse_args()
    cfg = load_raincalib_config(args.config)
    benchmark = load_config(cfg.benchmark_config)
    if cfg.feature_store.exists():
        raise FileExistsError(cfg.feature_store)
    keys = pd.read_parquet(args.source, columns=['run_date','cycle','lead_hour']).drop_duplicates()
    base = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    experiments = yaml.safe_load(args.experiment_config.read_text(encoding='utf-8'))
    boundaries = {base['splits'][key] for key in ('train_end', 'val_end')}
    for fold in experiments['folds']:
        boundaries.update(fold[key] for key in ('train_end', 'val_end'))
    observed_window = lru_cache(maxsize=64)(lambda start: load_window(args.root, start))
    frames = []
    purged_fields = 0
    for i, ((run, cycle), group) in enumerate(keys.groupby(['run_date','cycle']), 1):
        run_date = pd.Timestamp(run).date()
        path = benchmark.paths.gfs_raw / f'gfs_{run_date:%Y%m%d}_{int(cycle):02d}_tp_steps.grib2'
        with open_gfs_forecast_file(path) as ds:
            rainfall = accumulated_to_window_total(subset_region(ds.tp, bounds_from_config(benchmark.region)), 24).load()
        for lead in sorted(group.lead_hour):
            start = forecast_window_start(run, cycle, lead)
            if crosses_split_boundary(start, boundaries):
                purged_fields += 1
                continue
            forecast = rainfall.sel(lead_hour=lead)
            observed = align_observation_to_forecast_grid(observed_window(start), forecast)
            frame = _flatten_field(forecast, observed, run_date=run_date, observed_date=start.date(),
                                   cycle=int(cycle), lead_hour=int(lead), thresholds=cfg.event_thresholds,
                                   stride=cfg.feature_stride)
            if frame.empty:
                raise ValueError(f'No valid pixels for {run}/{cycle}/{lead}')
            frame['split'] = split_for_date(start.date(), cfg)
            frame['window_start_utc'] = start
            frame['window_end_utc'] = start + pd.Timedelta(hours=24)
            frames.append(frame)
        if i % 24 == 0:
            print(f'forecast_files={i}, fields={len(frames)}', flush=True)
    frame = add_time_features(pd.concat(frames, ignore_index=True))
    if set(frame.cycle.unique()) != {0,12}:
        raise ValueError('Both cycles are required')
    cfg.feature_store.parent.mkdir(parents=True, exist_ok=True)
    part = cfg.feature_store.with_suffix('.part.parquet')
    frame.to_parquet(part,index=False)
    part.replace(cfg.feature_store)
    summary = dict(rows=len(frame),fields=len(frames),purged_fields=purged_fields,
                   purged_boundary_dates=sorted(boundaries),source_field_selection=str(args.source),
                   note='Only field keys reused. Forecast grids rebuilt from GRIB; all labels rebuilt from half-hourly Late V07 with explicit dimension ordering.')
    cfg.feature_store.with_suffix('.parquet.summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
