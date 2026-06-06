from functools import lru_cache
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr

from ai_weather_rainfall.config import BenchmarkConfig, load_config
from ai_weather_rainfall.plots import plot_categorical_metric, plot_lead_metric
from ai_weather_rainfall.metrics import categorical_scores, continuous_scores


def score_forecast_against_observation(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    thresholds: list[float],
) -> dict[str, object]:
    return {
        "continuous": continuous_scores(forecast, observed),
        "categorical": [categorical_scores(forecast, observed, threshold) for threshold in thresholds],
    }


def metric_rows(
    model: str,
    year: int,
    cycle: str,
    lead_hour: int,
    scores: dict[str, object],
    run_date: str | None = None,
    observed_date: str | None = None,
    forecast_file: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    continuous = scores["continuous"]
    rows.append(
        {
            "model": model,
            "year": year,
            "cycle": cycle,
            "run_date": run_date,
            "observed_date": observed_date,
            "forecast_file": forecast_file,
            "lead_hour": lead_hour,
            "metric_type": "continuous",
            **continuous,
        }
    )
    for item in scores["categorical"]:
        rows.append(
            {
                "model": model,
                "year": year,
                "cycle": cycle,
                "run_date": run_date,
                "observed_date": observed_date,
                "forecast_file": forecast_file,
                "lead_hour": lead_hour,
                "metric_type": "categorical",
                **item,
            }
        )
    return rows


def run_benchmark(config_path: Path) -> Path:
    cfg = load_config(config_path)
    _ensure_output_dirs(cfg)
    shard_paths: list[Path] = []
    missing_rows: list[dict[str, object]] = []

    for year in cfg.period.years:
        active_models = ["ifs"] if year in cfg.period.background_baseline_years else cfg.forecast.models
        for model in active_models:
            for cycle in cfg.forecast.cycles_utc:
                result = _score_available_files(cfg, model, year, cycle)
                shard_paths.extend(result["shards"])
                missing_rows.extend(result["missing_rows"])

    rows = _read_metric_rows(shard_paths, missing_rows)
    metrics_path = cfg.paths.metrics / f"{_artifact_prefix(cfg)}summary_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    analysis_paths = _write_analysis_outputs(cfg, rows)
    report_path = cfg.paths.reports / f"{_artifact_prefix(cfg)}rainfall_benchmark_report.md"
    report_path.write_text(_report_text(cfg, rows, metrics_path, analysis_paths), encoding="utf-8")
    return report_path


def _ensure_output_dirs(cfg: BenchmarkConfig) -> None:
    for path in [
        cfg.paths.aifs_raw,
        cfg.paths.ifs_raw,
        cfg.paths.imerg_raw,
        *([cfg.paths.gfs_raw] if cfg.paths.gfs_raw is not None else []),
        cfg.paths.processed,
        cfg.paths.processed / "observed",
        cfg.paths.processed / "score_shards",
        cfg.paths.processed / "score_shards_daily",
        cfg.paths.metrics,
        cfg.paths.figures,
        cfg.paths.reports,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _score_available_files(cfg: BenchmarkConfig, model: str, year: int, cycle: str) -> dict[str, list]:
    from ai_weather_rainfall.accumulations import accumulated_to_window_total
    from ai_weather_rainfall.data_imerg import discover_imerg_files
    from ai_weather_rainfall.download_plan import season_dates
    from ai_weather_rainfall.regrid import align_observation_to_forecast_grid
    from ai_weather_rainfall.regions import bounds_from_config, subset_region

    allowed_dates = set(season_dates(year, cfg.period.season_start, cfg.period.season_end))
    forecast_files = _discover_forecast_files(cfg, model, year, cycle, allowed_dates)
    imerg_files = discover_imerg_files(cfg.paths.imerg_raw, year, allowed_dates)
    if not forecast_files or not imerg_files:
        return {
            "shards": [],
            "missing_rows": [
                {"model": model, "year": year, "cycle": cycle, "metric_type": "missing_data", "lead_hour": -1}
            ],
        }

    bounds = bounds_from_config(cfg.region)
    observed_region_path = _observed_region_daily_path(cfg, year)
    observed_region_daily = _load_or_create_observed_region_daily(tuple(imerg_files), bounds, observed_region_path)
    observed_dates = {
        pd.Timestamp(value).date()
        for value in observed_region_daily["time"].values
    }
    shard_paths: list[Path] = []
    for forecast_file in forecast_files:
        shard_path = _score_shard_path(cfg, model, year, cycle, forecast_file)
        shard_paths.append(shard_path)
        if shard_path.exists() and shard_path.stat().st_size > 0:
            continue
        forecast_ds = _open_forecast_file(model, forecast_file)
        rainfall = accumulated_to_window_total(forecast_ds["tp"], cfg.forecast.accumulation_hours)
        rows: list[dict[str, object]] = []
        run_date = _forecast_file_date(model, forecast_file)
        run_date_text = f"{run_date:%Y-%m-%d}" if run_date else None
        for lead_hour in cfg.forecast.lead_hours:
            if lead_hour not in rainfall["lead_hour"].values:
                continue
            observed_date = _observation_date_for_window(run_date, lead_hour, cfg.forecast.accumulation_hours)
            if observed_date is None or observed_date not in observed_dates:
                continue
            forecast_slice = subset_region(rainfall.sel(lead_hour=lead_hour), bounds)
            observed_region = observed_region_daily.sel(time=pd.Timestamp(observed_date))
            observed_slice = align_observation_to_forecast_grid(observed_region, forecast_slice)
            scores = score_forecast_against_observation(
                forecast_slice,
                observed_slice,
                cfg.rainfall.thresholds_mm_24h,
            )
            rows.extend(
                metric_rows(
                    model,
                    year,
                    cycle,
                    lead_hour,
                    scores,
                    run_date=run_date_text,
                    observed_date=f"{observed_date:%Y-%m-%d}",
                    forecast_file=forecast_file.name,
                )
            )
        pd.DataFrame(rows).to_csv(shard_path, index=False)
    return {"shards": shard_paths, "missing_rows": []}


def _discover_forecast_files(
    cfg: BenchmarkConfig,
    model: str,
    year: int,
    cycle: str,
    allowed_dates: set,
) -> list[Path]:
    if model in {"aifs", "ifs"}:
        from ai_weather_rainfall.data_ecmwf import discover_forecast_files

        forecast_root = cfg.paths.aifs_raw if model == "aifs" else cfg.paths.ifs_raw
        return discover_forecast_files(forecast_root, model, year, cycle, allowed_dates)
    if model == "gfs":
        from ai_weather_rainfall.data_gfs import discover_gfs_forecast_files

        if cfg.paths.gfs_raw is None:
            raise ValueError("paths.gfs_raw is required when forecast.models includes gfs")
        return discover_gfs_forecast_files(cfg.paths.gfs_raw, year, cycle, allowed_dates)
    raise ValueError(f"Unsupported forecast model: {model}")


def _open_forecast_file(model: str, forecast_file: Path) -> xr.Dataset:
    if model in {"aifs", "ifs"}:
        from ai_weather_rainfall.data_ecmwf import open_forecast_file

        return open_forecast_file(forecast_file)
    if model == "gfs":
        from ai_weather_rainfall.data_gfs import open_gfs_forecast_file

        return open_gfs_forecast_file(forecast_file)
    raise ValueError(f"Unsupported forecast model: {model}")


def _forecast_file_date(model: str, forecast_file: Path):
    if model in {"aifs", "ifs"}:
        from ai_weather_rainfall.data_ecmwf import forecast_file_date

        return forecast_file_date(forecast_file)
    if model == "gfs":
        from ai_weather_rainfall.data_gfs import gfs_file_date

        return gfs_file_date(forecast_file)
    raise ValueError(f"Unsupported forecast model: {model}")


@lru_cache(maxsize=8)
def _load_or_create_observed_region_daily(imerg_files: tuple[Path, ...], bounds, output_path: Path) -> xr.DataArray:
    if output_path.exists() and output_path.stat().st_size > 0:
        return xr.open_dataarray(output_path).load()
    observed = _observed_region_daily(imerg_files, bounds)
    observed.to_netcdf(output_path)
    return observed


def _observed_region_daily(imerg_files: tuple[Path, ...], bounds) -> xr.DataArray:
    from ai_weather_rainfall.data_imerg import daily_observed_rainfall, precipitation_from_dataset
    from ai_weather_rainfall.regions import subset_region

    daily_regions: list[xr.DataArray] = []
    for path in imerg_files:
        with xr.open_dataset(path) as dataset:
            precipitation = precipitation_from_dataset(dataset)
            daily = daily_observed_rainfall(precipitation)
            region = subset_region(daily, bounds).load()
        daily_regions.append(region)
    if not daily_regions:
        raise FileNotFoundError("No IMERG files were available for the configured period")
    observed = xr.concat(daily_regions, dim="time").sortby("time")
    observed.name = "observed_rainfall_24h_mm"
    observed.attrs["units"] = "mm"
    return observed


def _observed_region_daily_path(cfg: BenchmarkConfig, year: int) -> Path:
    return cfg.paths.processed / "observed" / (
        f"imerg_region_daily_{year}_{cfg.period.season_start}_{cfg.period.season_end}.nc"
    )


def _score_shard_path(cfg: BenchmarkConfig, model: str, year: int, cycle: str, forecast_file: Path) -> Path:
    stem = forecast_file.stem.replace(".", "_")
    return cfg.paths.processed / "score_shards_daily" / f"{year}_{model}_{int(cycle):02d}_{stem}.csv"


def _observation_date_for_window(run_date, lead_hour: int, accumulation_hours: int):
    if run_date is None:
        return None
    window_start_hours = int(lead_hour) - int(accumulation_hours)
    if window_start_hours < 0:
        return None
    return run_date + timedelta(hours=window_start_hours)


def _read_metric_rows(shard_paths: Iterable[Path], missing_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frames = [pd.read_csv(path) for path in sorted(set(shard_paths)) if path.exists() and path.stat().st_size > 0]
    if frames:
        rows = pd.concat(frames, ignore_index=True).to_dict("records")
    else:
        rows = []
    rows.extend(missing_rows)
    return rows


def _write_analysis_outputs(cfg: BenchmarkConfig, rows: list[dict[str, object]]) -> dict[str, Path]:
    frame = pd.DataFrame(rows)
    paths: dict[str, Path] = {}
    if frame.empty:
        return paths

    continuous = frame[frame["metric_type"] == "continuous"].copy()
    if not continuous.empty:
        continuous_summary = (
            continuous.groupby(["model", "cycle", "lead_hour"], as_index=False)[["mae", "rmse", "bias", "pearson_r"]]
            .mean(numeric_only=True)
            .sort_values(["model", "cycle", "lead_hour"])
        )
        prefix = _artifact_prefix(cfg)
        continuous_path = cfg.paths.metrics / f"{prefix}continuous_lead_summary.csv"
        continuous_summary.to_csv(continuous_path, index=False)
        paths["continuous_summary"] = continuous_path
        for metric in ["mae", "rmse", "bias", "pearson_r"]:
            paths[f"plot_{metric}"] = plot_lead_metric(frame, metric, cfg.paths.figures / f"{prefix}lead_{metric}.png")

    categorical = frame[frame["metric_type"] == "categorical"].copy()
    if not categorical.empty:
        categorical_summary = (
            categorical.groupby(["model", "cycle", "lead_hour", "threshold_mm"], as_index=False)[
                ["hits", "misses", "false_alarms", "correct_negatives", "pod", "far", "csi", "frequency_bias", "ets"]
            ]
            .mean(numeric_only=True)
            .sort_values(["model", "cycle", "threshold_mm", "lead_hour"])
        )
        prefix = _artifact_prefix(cfg)
        categorical_path = cfg.paths.metrics / f"{prefix}categorical_lead_summary.csv"
        categorical_summary.to_csv(categorical_path, index=False)
        paths["categorical_summary"] = categorical_path
        event_summary = _aggregate_event_summary(categorical)
        event_path = cfg.paths.metrics / f"{prefix}event_skill_summary.csv"
        event_summary.to_csv(event_path, index=False)
        paths["event_summary"] = event_path
        for threshold in sorted(categorical["threshold_mm"].dropna().unique()):
            for metric in ["csi", "pod", "far", "ets"]:
                key = f"plot_{metric}_{threshold:g}"
                paths[key] = plot_categorical_metric(
                    frame,
                    metric,
                    float(threshold),
                    cfg.paths.figures / f"{prefix}lead_{metric}_{threshold:g}mm.png",
                )
    return paths


def _artifact_prefix(cfg: BenchmarkConfig) -> str:
    models = set(cfg.forecast.models)
    if models == {"aifs", "ifs"}:
        return ""
    if models == {"gfs"}:
        return "gfs_"
    return "_".join(sorted(models)) + "_"


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
                "frequency_bias": _ratio(hits + false_alarms, hits + misses),
                "ets": _ratio(hits - random_hits, hits + misses + false_alarms - random_hits),
            }
        )
    return pd.DataFrame(rows).sort_values(["threshold_mm", "model"])


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _report_text(
    cfg: BenchmarkConfig,
    rows: list[dict[str, object]],
    metrics_path: Path,
    analysis_paths: dict[str, Path],
) -> str:
    from ai_weather_rainfall.download_plan import configured_dates

    dates = configured_dates(cfg)
    date_range = f"{dates[0]}..{dates[-1]}" if dates else "none"
    baseline_note = (
        "- Background baseline years: "
        f"{cfg.period.background_baseline_years}\n"
        if cfg.period.background_baseline_years
        else "- Background baseline years: none\n"
    )
    analysis_text = _analysis_text(rows, analysis_paths)
    return (
        "# Rainfall Benchmark Report\n\n"
        "This report evaluates configured weather forecast models against NASA GPM IMERG observations.\n\n"
        "**Important:** This is a research and decision-support benchmark, not an official weather warning service.\n\n"
        "Forecast 24-hour windows are matched to the IMERG calendar day at the start of each accumulation window. "
        "For 12 UTC cycles this is an approximation because the available IMERG product is daily rather than 12-hourly.\n\n"
        f"- Region: {cfg.region.name} ({cfg.region.lon_min}..{cfg.region.lon_max}E, "
        f"{cfg.region.lat_min}..{cfg.region.lat_max}N)\n"
        f"- Years: {cfg.period.years}\n"
        f"- Benchmark dates: {date_range}\n"
        f"- Forecast models: {cfg.forecast.models}\n"
        f"- Primary comparison years: {cfg.period.primary_comparison_years}\n"
        f"{baseline_note}"
        f"- Metric rows written: {len(rows)}\n"
        f"- Metrics CSV: `{metrics_path}`\n"
        f"{analysis_text}"
    )


def _analysis_text(rows: list[dict[str, object]], analysis_paths: dict[str, Path]) -> str:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return ""

    lines = ["\n## Results Summary\n\n"]
    continuous = frame[frame["metric_type"] == "continuous"].copy()
    if not continuous.empty:
        by_model = continuous.groupby("model")[["mae", "rmse", "bias", "pearson_r"]].mean(numeric_only=True)
        lines.append("| Model | MAE | RMSE | Bias | Pearson r |\n")
        lines.append("|---|---:|---:|---:|---:|\n")
        for model, row in by_model.sort_index().iterrows():
            lines.append(
                f"| {str(model).upper()} | {row['mae']:.3f} | {row['rmse']:.3f} | "
                f"{row['bias']:.3f} | {row['pearson_r']:.3f} |\n"
            )
        lines.append("\n### Samples By Lead\n\n")
        lead_counts = continuous.groupby(["lead_hour", "model"]).size().unstack(fill_value=0)
        model_names = [str(item) for item in sorted(lead_counts.columns)]
        lines.append("| Lead hour | " + " | ".join(f"{name.upper()} samples" for name in model_names) + " |\n")
        lines.append("|---:|" + "|".join("---:" for _ in model_names) + "|\n")
        for lead_hour, row in lead_counts.sort_index().iterrows():
            counts = " | ".join(str(int(row.get(name, 0))) for name in model_names)
            lines.append(f"| {int(lead_hour)} | {counts} |\n")

    categorical = frame[frame["metric_type"] == "categorical"].copy()
    if not categorical.empty:
        lines.append("\n### Event Skill\n\n")
        event_summary = _aggregate_event_summary(categorical)
        lines.append("| Threshold | Model | Hits | Misses | False alarms | CSI | POD | FAR | ETS |\n")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in event_summary.iterrows():
            lines.append(
                f"| {row['threshold_mm']:g} mm | {str(row['model']).upper()} | {int(row['hits'])} | "
                f"{int(row['misses'])} | {int(row['false_alarms'])} | {row['csi']:.3f} | "
                f"{row['pod']:.3f} | {row['far']:.3f} | {row['ets']:.3f} |\n"
            )

    if analysis_paths:
        lines.append("\n### Artifacts\n\n")
        for label, path in sorted(analysis_paths.items()):
            lines.append(f"- {label}: `{path}`\n")
    return "".join(lines)
