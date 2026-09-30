from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ai_weather_rainfall.accumulations import accumulated_to_window_total
from ai_weather_rainfall.config import load_config
from ai_weather_rainfall.pipeline import _observation_date_for_window, validate_daily_windows
from ai_weather_rainfall.data_gfs import discover_gfs_forecast_files, gfs_file_date, open_gfs_forecast_file
from ai_weather_rainfall.download_plan import season_dates
from ai_weather_rainfall.raincalib import add_time_features, load_raincalib_config, split_for_date
from ai_weather_rainfall.regrid import align_observation_to_forecast_grid
from ai_weather_rainfall.regions import bounds_from_config, subset_region


def main() -> None:
    args = _parse_args()
    cfg = load_raincalib_config(args.config)
    benchmark_cfg = load_config(cfg.benchmark_config)
    validate_daily_windows(benchmark_cfg.forecast)
    output_path = args.output or cfg.feature_store
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"feature store exists; pass --overwrite to rebuild: {output_path}")
    output_path.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    rng = np.random.default_rng(cfg.seed)
    counts = {"rows": 0, "fields": 0, "files": 0, "train_rows": 0, "val_rows": 0, "test_rows": 0}
    bounds = bounds_from_config(benchmark_cfg.region)

    try:
        for year in benchmark_cfg.period.years:
            observed = _open_observed_region(benchmark_cfg, year)
            observed_dates = {pd.Timestamp(value).date() for value in observed["time"].values}
            allowed_dates = set(season_dates(year, benchmark_cfg.period.season_start, benchmark_cfg.period.season_end))
            for cycle in benchmark_cfg.forecast.cycles_utc:
                files = discover_gfs_forecast_files(
                    benchmark_cfg.paths.gfs_raw,
                    year,
                    cycle,
                    allowed_dates,
                )
                for forecast_file in files:
                    if args.max_files is not None and counts["files"] >= args.max_files:
                        break
                    counts["files"] += 1
                    frame = _features_for_file(
                        forecast_file,
                        observed,
                        observed_dates,
                        bounds,
                        benchmark_cfg,
                        cfg,
                        rng,
                    )
                    if frame.empty:
                        continue
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema)
                    writer.write_table(table)
                    counts["rows"] += len(frame)
                    counts["fields"] += frame[["run_date", "cycle", "lead_hour"]].drop_duplicates().shape[0]
                    for split, split_count in frame["split"].value_counts().items():
                        counts[f"{split}_rows"] = counts.get(f"{split}_rows", 0) + int(split_count)
                    print(
                        f"features file={forecast_file.name} rows={len(frame)} total_rows={counts['rows']}",
                        flush=True,
                    )
                if args.max_files is not None and counts["files"] >= args.max_files:
                    break
            if args.max_files is not None and counts["files"] >= args.max_files:
                break
    finally:
        if writer is not None:
            writer.close()

    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8")
    print(f"feature_store={output_path}")
    print(f"summary={summary_path}")
    print(json.dumps(counts, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RainCalib feature store from GFS forecasts and IMERG observations.")
    parser.add_argument("--config", type=Path, default=Path("configs/raincalib.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, help="Build a small feature store for smoke tests.")
    return parser.parse_args()


def _open_observed_region(benchmark_cfg, year: int):
    import xarray as xr

    path = benchmark_cfg.paths.processed / "observed" / (
        f"imerg_region_daily_{year}_{benchmark_cfg.period.season_start}_{benchmark_cfg.period.season_end}.nc"
    )
    if not path.exists():
        raise FileNotFoundError(f"observed daily region cache not found: {path}")
    return xr.open_dataarray(path).load()


def _features_for_file(
    forecast_file: Path,
    observed,
    observed_dates: set,
    bounds,
    benchmark_cfg,
    cfg,
    rng: np.random.Generator,
) -> pd.DataFrame:
    forecast_ds = open_gfs_forecast_file(forecast_file)
    rainfall = accumulated_to_window_total(forecast_ds["tp"], benchmark_cfg.forecast.accumulation_hours)
    run_date = gfs_file_date(forecast_file)
    if run_date is None:
        return pd.DataFrame()
    cycle = int(forecast_file.stem.split("_")[2])
    frames: list[pd.DataFrame] = []
    for lead_hour in benchmark_cfg.forecast.lead_hours:
        if lead_hour not in rainfall["lead_hour"].values:
            continue
        observed_date = _observation_date_for_window(run_date, lead_hour, benchmark_cfg.forecast.accumulation_hours, cycle)
        if observed_date not in observed_dates:
            continue
        forecast_slice = subset_region(rainfall.sel(lead_hour=lead_hour), bounds)
        observed_region = observed.sel(time=pd.Timestamp(observed_date))
        observed_slice = align_observation_to_forecast_grid(observed_region, forecast_slice)
        frame = _flatten_field(
            forecast_slice,
            observed_slice,
            run_date=run_date,
            observed_date=observed_date,
            cycle=cycle,
            lead_hour=lead_hour,
            thresholds=cfg.event_thresholds,
            stride=cfg.feature_stride,
        )
        frame["split"] = split_for_date(observed_date, cfg)
        frame = _sample_training_negatives(frame, cfg, rng)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return add_time_features(result)


def _flatten_field(
    forecast,
    observed,
    *,
    run_date,
    observed_date,
    cycle: int,
    lead_hour: int,
    thresholds: list[float],
    stride: int,
) -> pd.DataFrame:
    forecast = forecast.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
    observed = observed.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
    lat = forecast["latitude"].values
    lon = forecast["longitude"].values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    forecast = forecast.transpose("latitude", "longitude")
    observed = observed.transpose("latitude", "longitude")
    gfs = forecast.values.reshape(-1)
    target = observed.values.reshape(-1)
    valid = np.isfinite(gfs) & np.isfinite(target)
    frame = pd.DataFrame(
        {
            "run_date": f"{run_date:%Y-%m-%d}",
            "observed_date": f"{observed_date:%Y-%m-%d}",
            "year": int(observed_date.year),
            "month": int(observed_date.month),
            "day_of_year": int(observed_date.timetuple().tm_yday),
            "cycle": int(cycle),
            "lead_hour": int(lead_hour),
            "latitude": lat_grid.reshape(-1)[valid].astype("float32"),
            "longitude": lon_grid.reshape(-1)[valid].astype("float32"),
            "gfs_rain": gfs[valid].astype("float32"),
            "target_rain": target[valid].astype("float32"),
        }
    )
    for threshold in thresholds:
        label = f"event_{threshold:g}".replace(".", "_")
        frame[label] = (frame["target_rain"] >= float(threshold)).astype("int8")
    return frame


def _sample_training_negatives(frame: pd.DataFrame, cfg, rng: np.random.Generator) -> pd.DataFrame:
    if frame.empty or frame["split"].iloc[0] != "train":
        return frame
    primary_threshold = min(cfg.event_thresholds)
    important = (frame["target_rain"] >= primary_threshold) | (frame["gfs_rain"] >= primary_threshold)
    keep_negative = rng.random(len(frame)) < cfg.train_negative_sample_fraction
    return frame[important.to_numpy() | keep_negative].reset_index(drop=True)


if __name__ == "__main__":
    main()
