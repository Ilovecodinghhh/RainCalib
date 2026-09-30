from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "run_date",
    "observed_date",
    "cycle",
    "lead_hour",
    "gfs_rain",
    "target_rain",
]


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.features, columns=REQUIRED_COLUMNS)
    frame["observed_date"] = pd.to_datetime(frame["observed_date"])
    frame["run_date"] = pd.to_datetime(frame["run_date"])

    daily = daily_audit(frame, args.thresholds)
    yearly = yearly_audit(daily)
    lead = lead_audit(frame, args.thresholds)
    manifest = experiment_manifest(frame, daily, args)

    daily_path = output_dir / "daily_sample_counts.csv"
    yearly_path = output_dir / "yearly_event_counts.csv"
    lead_path = output_dir / "lead_coverage.csv"
    event_path = output_dir / "event_catalog.csv"
    manifest_path = output_dir / "experiment_manifest.json"
    report_path = output_dir / "data_audit_report.md"
    daily.to_csv(daily_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    lead.to_csv(lead_path, index=False)
    daily[daily["event_pixels_50mm"] > 0].to_csv(event_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_report_text(manifest, yearly, lead), encoding="utf-8")

    print(f"daily={daily_path}")
    print(f"yearly={yearly_path}")
    print(f"lead={lead_path}")
    print(f"events={event_path}")
    print(f"manifest={manifest_path}")
    print(f"report={report_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit independent dates, fields, leads, and event counts.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/rainfall_benchmark/ml_00utc/features/raincalib_features.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rainfall_benchmark/experiments_00utc/audit"),
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=[50.0, 100.0])
    return parser.parse_args()


def daily_audit(frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    working = frame.copy()
    working["field_key"] = (
        working["run_date"].astype(str)
        + "_"
        + working["cycle"].astype(str)
        + "_"
        + working["lead_hour"].astype(str)
    )
    grouped = working.groupby("observed_date", sort=True)
    daily = grouped.agg(
        rows=("target_rain", "size"),
        forecast_fields=("field_key", "nunique"),
        cycles=("cycle", "nunique"),
        lead_hours=("lead_hour", "nunique"),
        target_mean=("target_rain", "mean"),
        target_max=("target_rain", "max"),
        gfs_mean=("gfs_rain", "mean"),
        gfs_max=("gfs_rain", "max"),
    ).reset_index()
    for threshold in thresholds:
        label = f"{threshold:g}mm"
        counts = grouped["target_rain"].apply(lambda values: int(np.sum(values.to_numpy() >= threshold)))
        daily[f"event_pixels_{label}"] = daily["observed_date"].map(counts).astype(int)
        daily[f"event_day_{label}"] = daily[f"event_pixels_{label}"] > 0
    daily["year"] = daily["observed_date"].dt.year
    daily["month"] = daily["observed_date"].dt.month
    return daily


def yearly_audit(daily: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "observed_dates": ("observed_date", "nunique"),
        "rows": ("rows", "sum"),
        "forecast_fields": ("forecast_fields", "sum"),
        "target_mean": ("target_mean", "mean"),
        "target_max": ("target_max", "max"),
    }
    for column in daily.columns:
        if column.startswith("event_day_"):
            aggregations[column.replace("event_day_", "event_days_")] = (column, "sum")
        if column.startswith("event_pixels_"):
            aggregations[column] = (column, "sum")
    return daily.groupby("year", sort=True).agg(**aggregations).reset_index()


def lead_audit(frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for lead_hour, group in frame.groupby("lead_hour", sort=True):
        row: dict[str, float | int] = {
            "lead_hour": int(lead_hour),
            "rows": int(len(group)),
            "observed_dates": int(group["observed_date"].nunique()),
            "run_dates": int(group["run_date"].nunique()),
            "cycles": int(group["cycle"].nunique()),
            "target_mean": float(group["target_rain"].mean()),
            "target_max": float(group["target_rain"].max()),
        }
        for threshold in thresholds:
            row[f"event_pixels_{threshold:g}mm"] = int(np.sum(group["target_rain"].to_numpy() >= threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def experiment_manifest(
    frame: pd.DataFrame,
    daily: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "git_commit": _git_commit(),
        "feature_store": str(args.features),
        "rows": int(len(frame)),
        "forecast_fields": int(frame[["run_date", "cycle", "lead_hour"]].drop_duplicates().shape[0]),
        "observed_dates": int(frame["observed_date"].nunique()),
        "date_start": f"{frame['observed_date'].min():%Y-%m-%d}",
        "date_end": f"{frame['observed_date'].max():%Y-%m-%d}",
        "thresholds_mm": [float(value) for value in args.thresholds],
        "event_days": {
            f"{threshold:g}": int(daily[f"event_day_{threshold:g}mm"].sum())
            for threshold in args.thresholds
        },
        "independence_note": "Rows and forecast fields are correlated; observed_date is the bootstrap unit.",
    }


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _report_text(manifest: dict[str, object], yearly: pd.DataFrame, lead: pd.DataFrame) -> str:
    lines = [
        "# RainCalib Data Audit\n\n",
        f"- Git commit: `{manifest['git_commit']}`\n",
        f"- Date range: {manifest['date_start']} to {manifest['date_end']}\n",
        f"- Observed dates: {manifest['observed_dates']}\n",
        f"- Forecast fields: {manifest['forecast_fields']}\n",
        f"- Valid grid-cell rows: {manifest['rows']}\n",
        f"- Event days: {manifest['event_days']}\n\n",
        "Grid-cell rows are not independent samples. All confidence intervals must resample complete observed dates.\n\n",
        "## Yearly Coverage\n\n",
        _markdown_table(yearly),
        "\n\n## Lead Coverage\n\n",
        _markdown_table(lead),
        "\n",
    ]
    return "".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    def format_value(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value)

    headers = [str(column) for column in frame.columns]
    rows = [headers, ["---"] * len(headers)]
    rows.extend([format_value(value) for value in row] for row in frame.itertuples(index=False, name=None))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


if __name__ == "__main__":
    main()
