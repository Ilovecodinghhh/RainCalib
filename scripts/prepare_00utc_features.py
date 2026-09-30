"""Deprecated: legacy labels require a spatial rebuild, not a cycle-only copy."""
from pathlib import Path
import argparse
import pandas as pd
from ai_weather_rainfall.pipeline import _observation_date_for_window


def select_daily_features(frame):
    selected = frame.loc[pd.to_numeric(frame['cycle'], errors='raise') == 0].copy()
    if selected.empty:
        raise ValueError('No 00 UTC samples found')
    keys = selected[['run_date', 'observed_date', 'lead_hour']].drop_duplicates()
    for row in keys.itertuples(index=False):
        expected = _observation_date_for_window(pd.Timestamp(row.run_date), int(row.lead_hour), 24, 0)
        if pd.Timestamp(row.observed_date).date() != expected:
            raise ValueError(f'Observation date mismatch: {row}')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/rainfall_benchmark/ml/features/raincalib_features.parquet'))
    parser.add_argument('--config', type=Path, default=Path('configs/raincalib.yaml'))
    args = parser.parse_args()
    raise SystemExit("Legacy observation labels have a confirmed spatial-axis error. Rebuild from raw data using build_raincalib_features.py; do not copy legacy labels.")


if __name__ == '__main__':
    main()
