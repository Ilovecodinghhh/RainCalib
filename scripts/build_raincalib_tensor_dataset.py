from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_weather_rainfall.deep.dataset import load_unet_config


FEATURE_COLUMNS = [
    "run_date",
    "observed_date",
    "cycle",
    "lead_hour",
    "latitude",
    "longitude",
    "gfs_rain",
    "target_rain",
    "split",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


def main() -> None:
    args = _parse_args()
    cfg = load_unet_config(args.config)
    output = args.output or cfg.tensor_store
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"tensor store exists; pass --overwrite to rebuild: {output}")

    frame = pd.read_parquet(cfg.feature_store, columns=FEATURE_COLUMNS)
    if args.max_samples is not None:
        keys = frame[["run_date", "cycle", "lead_hour"]].drop_duplicates().head(args.max_samples)
        frame = frame.merge(keys, on=["run_date", "cycle", "lead_hour"], how="inner")
    latitudes = np.array(sorted(frame["latitude"].unique()), dtype="float32")
    longitudes = np.array(sorted(frame["longitude"].unique()), dtype="float32")
    lat_to_index = {float(value): index for index, value in enumerate(latitudes)}
    lon_to_index = {float(value): index for index, value in enumerate(longitudes)}
    height = len(latitudes)
    width = len(longitudes)

    groups = list(frame.groupby(["run_date", "observed_date", "cycle", "lead_hour"], sort=True))
    sample_count = len(groups)
    gfs = np.zeros((sample_count, height, width), dtype="float32")
    target = np.zeros((sample_count, height, width), dtype="float32")
    mask = np.zeros((sample_count, height, width), dtype="float32")
    run_date = np.empty(sample_count, dtype="U10")
    observed_date = np.empty(sample_count, dtype="U10")
    split = np.empty(sample_count, dtype="U8")
    cycle = np.zeros(sample_count, dtype="float32")
    lead_hour = np.zeros(sample_count, dtype="float32")
    month_sin = np.zeros(sample_count, dtype="float32")
    month_cos = np.zeros(sample_count, dtype="float32")
    day_of_year_sin = np.zeros(sample_count, dtype="float32")
    day_of_year_cos = np.zeros(sample_count, dtype="float32")

    for index, ((run, observed, cyc, lead), group) in enumerate(groups):
        rows = group["latitude"].map(lambda value: lat_to_index[float(value)]).to_numpy()
        cols = group["longitude"].map(lambda value: lon_to_index[float(value)]).to_numpy()
        valid = np.isfinite(group["gfs_rain"].to_numpy()) & np.isfinite(group["target_rain"].to_numpy())
        rows = rows[valid]
        cols = cols[valid]
        gfs[index, rows, cols] = group["gfs_rain"].to_numpy(dtype="float32")[valid]
        target[index, rows, cols] = group["target_rain"].to_numpy(dtype="float32")[valid]
        mask[index, rows, cols] = 1.0
        run_date[index] = str(run)
        observed_date[index] = str(observed)
        split[index] = str(group["split"].iloc[0])
        cycle[index] = float(cyc)
        lead_hour[index] = float(lead)
        month_sin[index] = float(group["month_sin"].iloc[0])
        month_cos[index] = float(group["month_cos"].iloc[0])
        day_of_year_sin[index] = float(group["day_of_year_sin"].iloc[0])
        day_of_year_cos[index] = float(group["day_of_year_cos"].iloc[0])
        if (index + 1) % 500 == 0:
            print(f"tensorized={index + 1}/{sample_count}", flush=True)

    np.savez_compressed(
        output,
        gfs=gfs,
        target=target,
        mask=mask,
        run_date=run_date,
        observed_date=observed_date,
        split=split,
        cycle=cycle,
        lead_hour=lead_hour,
        month_sin=month_sin,
        month_cos=month_cos,
        day_of_year_sin=day_of_year_sin,
        day_of_year_cos=day_of_year_cos,
        latitude=latitudes,
        longitude=longitudes,
    )
    summary = {
        "samples": int(sample_count),
        "height": int(height),
        "width": int(width),
        "valid_pixels": int(mask.sum()),
        "split_counts": {str(k): int(v) for k, v in pd.Series(split).value_counts().items()},
        "path": str(output),
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"tensor_store={output}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gridded RainCalib tensors from the feature store.")
    parser.add_argument("--config", type=Path, default=Path("configs/raincalib_unet.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, help="Build only the first N forecast fields for smoke tests.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
