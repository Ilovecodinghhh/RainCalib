from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ai_weather_rainfall.plots import plot_categorical_metric, plot_lead_metric


def main() -> None:
    args = _parse_args()
    ecmwf = _read_metrics(args.ecmwf_metrics, required=True)
    gfs = _read_metrics(args.gfs_metrics, required=True)
    combined = pd.concat([ecmwf, gfs], ignore_index=True)
    same_window = _filter_run_window(combined, args.start_date, args.end_date)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _write_plots(same_window, args.figures_dir)
    report = _report_text(
        gfs=gfs,
        same_window=same_window,
        start_date=args.start_date,
        end_date=args.end_date,
        artifacts=artifacts,
    )
    args.output.write_text(report, encoding="utf-8")
    print(f"report={args.output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a GFS/AIFS/IFS model comparison report.")
    parser.add_argument(
        "--ecmwf-metrics",
        type=Path,
        default=Path("outputs/rainfall_benchmark/metrics/summary_metrics.csv"),
    )
    parser.add_argument(
        "--gfs-metrics",
        type=Path,
        default=Path("outputs/rainfall_benchmark/metrics/gfs_summary_metrics.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rainfall_benchmark/reports/model_comparison_report.md"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/rainfall_benchmark/figures"),
    )
    parser.add_argument("--start-date", default="2025-05-01")
    parser.add_argument("--end-date", default="2025-09-30")
    return parser.parse_args()


def _read_metrics(path: Path, *, required: bool) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"metrics file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _filter_run_window(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    result = frame.copy()
    result["run_date"] = pd.to_datetime(result["run_date"], errors="coerce")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return result[(result["run_date"] >= start) & (result["run_date"] <= end)].copy()


def _write_plots(frame: pd.DataFrame, figures_dir: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for metric in ["mae", "rmse", "bias", "pearson_r"]:
        artifacts[f"plot_{metric}"] = plot_lead_metric(
            frame,
            metric,
            figures_dir / f"model_comparison_lead_{metric}.png",
        )
    categorical = frame[frame["metric_type"] == "categorical"]
    for threshold in sorted(categorical["threshold_mm"].dropna().astype(float).unique()):
        for metric in ["csi", "pod", "far", "ets"]:
            artifacts[f"plot_{metric}_{threshold:g}"] = plot_categorical_metric(
                frame,
                metric,
                threshold,
                figures_dir / f"model_comparison_lead_{metric}_{threshold:g}mm.png",
            )
    return artifacts


def _report_text(
    *,
    gfs: pd.DataFrame,
    same_window: pd.DataFrame,
    start_date: str,
    end_date: str,
    artifacts: dict[str, Path],
) -> str:
    lines = [
        "# Model Comparison Report\n\n",
        "This report compares GFS, ECMWF AIFS, and ECMWF IFS rainfall forecasts against NASA GPM IMERG observations.\n\n",
        "**Important:** This is a research and decision-support benchmark, not an official weather warning service.\n\n",
        "The full GFS section uses the complete configured 2023-2025 flood-season archive. "
        "The same-window section uses only the free ECMWF Open Data period shared by all three models.\n\n",
        f"- Same-window period: {start_date}..{end_date}\n",
        "- Region: middle_lower_yangtze (108.0..123.0E, 24.0..35.0N)\n",
        "- Observation reference: NASA GPM IMERG Daily V07B\n\n",
        "## Full GFS 2023-2025\n\n",
    ]
    lines.extend(_continuous_table(gfs))
    lines.append("\n### GFS Samples By Lead\n\n")
    lines.extend(_samples_by_lead_table(gfs))
    lines.append("\n### GFS Event Skill\n\n")
    lines.extend(_event_table(gfs))

    lines.append("\n## Same-Window Model Comparison\n\n")
    lines.extend(_continuous_table(same_window))
    lines.append("\n### Samples By Lead\n\n")
    lines.extend(_samples_by_lead_table(same_window))
    lines.append("\n### Event Skill\n\n")
    lines.extend(_event_table(same_window))
    lines.append("\n### Artifacts\n\n")
    for label, path in sorted(artifacts.items()):
        lines.append(f"- {label}: `{path}`\n")
    return "".join(lines)


def _continuous_table(frame: pd.DataFrame) -> list[str]:
    continuous = frame[frame["metric_type"] == "continuous"].copy()
    if continuous.empty:
        return ["No continuous metrics available.\n"]
    summary = continuous.groupby("model")[["mae", "rmse", "bias", "pearson_r"]].mean(numeric_only=True)
    lines = ["| Model | MAE | RMSE | Bias | Pearson r |\n", "|---|---:|---:|---:|---:|\n"]
    for model, row in summary.sort_index().iterrows():
        lines.append(
            f"| {str(model).upper()} | {row['mae']:.3f} | {row['rmse']:.3f} | "
            f"{row['bias']:.3f} | {row['pearson_r']:.3f} |\n"
        )
    return lines


def _samples_by_lead_table(frame: pd.DataFrame) -> list[str]:
    continuous = frame[frame["metric_type"] == "continuous"].copy()
    if continuous.empty:
        return ["No samples available.\n"]
    lead_counts = continuous.groupby(["lead_hour", "model"]).size().unstack(fill_value=0)
    model_names = [str(item) for item in sorted(lead_counts.columns)]
    lines = ["| Lead hour | " + " | ".join(f"{name.upper()} samples" for name in model_names) + " |\n"]
    lines.append("|---:|" + "|".join("---:" for _ in model_names) + "|\n")
    for lead_hour, row in lead_counts.sort_index().iterrows():
        counts = " | ".join(str(int(row.get(name, 0))) for name in model_names)
        lines.append(f"| {int(lead_hour)} | {counts} |\n")
    return lines


def _event_table(frame: pd.DataFrame) -> list[str]:
    categorical = frame[frame["metric_type"] == "categorical"].copy()
    if categorical.empty:
        return ["No categorical metrics available.\n"]
    event_summary = _aggregate_event_summary(categorical)
    lines = ["| Threshold | Model | Hits | Misses | False alarms | CSI | POD | FAR | ETS |\n"]
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, row in event_summary.iterrows():
        lines.append(
            f"| {row['threshold_mm']:g} mm | {str(row['model']).upper()} | {int(row['hits'])} | "
            f"{int(row['misses'])} | {int(row['false_alarms'])} | {row['csi']:.3f} | "
            f"{row['pod']:.3f} | {row['far']:.3f} | {row['ets']:.3f} |\n"
        )
    return lines


def _aggregate_event_summary(categorical: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = categorical.groupby(["threshold_mm", "model"], dropna=True)
    for (threshold, model), group in grouped:
        hits = int(group["hits"].sum())
        misses = int(group["misses"].sum())
        false_alarms = int(group["false_alarms"].sum())
        correct_negatives = int(group["correct_negatives"].sum())
        total = hits + misses + false_alarms + correct_negatives
        random_hits = ((hits + misses) * (hits + false_alarms)) / total if total else float("nan")
        rows.append(
            {
                "threshold_mm": threshold,
                "model": model,
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "correct_negatives": correct_negatives,
                "pod": _ratio(hits, hits + misses),
                "far": _ratio(false_alarms, hits + false_alarms),
                "csi": _ratio(hits, hits + misses + false_alarms),
                "ets": _ratio(hits - random_hits, hits + misses + false_alarms - random_hits),
            }
        )
    return pd.DataFrame(rows).sort_values(["threshold_mm", "model"])


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


if __name__ == "__main__":
    main()
