from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import xarray as xr

from ai_weather_rainfall.config import load_config
from ai_weather_rainfall.data_ecmwf import open_forecast_file
from ai_weather_rainfall.download_plan import configured_dates, forecast_target_path, required_lead_steps


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    dates = configured_dates(cfg)
    expected_dates = {f"{item:%Y%m%d}" for item in dates}
    expected_steps = required_lead_steps(cfg.forecast.lead_hours, cfg.forecast.accumulation_hours)

    imerg_files = sorted(cfg.paths.imerg_raw.glob("*.nc4"))
    imerg_by_date = _files_by_date(imerg_files)
    imerg_missing = [item for item in dates if f"{item:%Y%m%d}" not in imerg_by_date]
    imerg_extra = sorted(set(imerg_by_date) - expected_dates)

    forecast_files = sorted([*cfg.paths.aifs_raw.glob("*.grib2"), *cfg.paths.ifs_raw.glob("*.grib2")])
    forecast_by_key = _forecast_files_by_key(forecast_files)
    forecast_missing = []
    by_model_cycle: dict[str, int] = defaultdict(int)
    for run_date in dates:
        for model in cfg.forecast.models:
            for cycle in cfg.forecast.cycles_utc:
                key = (model, f"{run_date:%Y%m%d}", f"{int(cycle):02d}")
                if key in forecast_by_key:
                    by_model_cycle[f"{model}_{int(cycle):02d}"] += 1
                else:
                    forecast_missing.append(key)
    forecast_extra = sorted(key for key in forecast_by_key if key[1] not in expected_dates)

    temp_files = sorted(cfg.project.output_root.glob("raw/**/*.*"))
    temp_files = [path for path in temp_files if path.suffix in {".part", ".download", ".missing", ".repair"}]
    zero_files = [path for path in [*imerg_files, *forecast_files] if path.stat().st_size == 0]

    print(f"configured_dates={dates[0]}..{dates[-1]} days={len(dates)}")
    print(f"imerg_expected={len(dates)} imerg_present={len(dates) - len(imerg_missing)}")
    print(f"imerg_missing={len(imerg_missing)} imerg_extra={len(imerg_extra)}")
    print(
        "ecmwf_expected="
        f"{len(dates) * len(cfg.forecast.models) * len(cfg.forecast.cycles_utc)} "
        f"ecmwf_present={len(dates) * len(cfg.forecast.models) * len(cfg.forecast.cycles_utc) - len(forecast_missing)}"
    )
    print(f"ecmwf_by_model_cycle={dict(sorted(by_model_cycle.items()))}")
    print(f"ecmwf_missing={len(forecast_missing)} ecmwf_extra={len(forecast_extra)}")
    print(f"zero_files={len(zero_files)} temp_files={len(temp_files)}")

    if imerg_missing:
        print("first_missing_imerg=" + ", ".join(f"{item:%Y-%m-%d}" for item in imerg_missing[:10]))
    if forecast_missing:
        print("first_missing_ecmwf=" + ", ".join(" ".join(item) for item in forecast_missing[:10]))
    if zero_files:
        print("zero_file_paths=" + ", ".join(str(path) for path in zero_files[:10]))
    if temp_files:
        print("temp_file_paths=" + ", ".join(str(path) for path in temp_files[:10]))

    _check_imerg_samples(imerg_by_date, dates, args.sample_count)
    _check_forecast_samples(forecast_by_key, cfg.forecast.models, cfg.forecast.cycles_utc, dates, expected_steps, args.sample_count)
    if args.check_all_imerg:
        _check_all_imerg(imerg_by_date, dates)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check configured IMERG and ECMWF download coverage.")
    parser.add_argument("--config", type=Path, default=Path("configs/rainfall_benchmark.yaml"))
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--check-all-imerg", action="store_true", help="Read a small precipitation sample from every configured IMERG file.")
    return parser.parse_args()


def _files_by_date(paths: list[Path]) -> dict[str, Path]:
    by_date: dict[str, Path] = {}
    for path in paths:
        match = re.search(r"(20\d{6})", path.name)
        if match:
            by_date[match.group(1)] = path
    return by_date


def _forecast_files_by_key(paths: list[Path]) -> dict[tuple[str, str, str], Path]:
    by_key: dict[tuple[str, str, str], Path] = {}
    for path in paths:
        match = re.search(r"^(aifs|ifs)_(20\d{6})_(\d{2})_", path.name)
        if match:
            by_key[(match.group(1), match.group(2), match.group(3))] = path
    return by_key


def _sample_dates(dates: list[date], count: int) -> list[date]:
    if len(dates) <= count:
        return dates
    if count <= 1:
        return [dates[0]]
    indexes = sorted({round(index * (len(dates) - 1) / (count - 1)) for index in range(count)})
    return [dates[index] for index in indexes]


def _check_imerg_samples(imerg_by_date: dict[str, Path], dates: list[date], sample_count: int) -> None:
    for run_date in _sample_dates(dates, sample_count):
        path = imerg_by_date.get(f"{run_date:%Y%m%d}")
        if path is None:
            continue
        dataset = xr.open_dataset(path)
        print(
            "imerg_sample="
            f"{run_date} size={path.stat().st_size} dims={dict(dataset.sizes)} "
            f"vars={list(dataset.data_vars)[:4]}"
        )


def _check_all_imerg(imerg_by_date: dict[str, Path], dates: list[date]) -> None:
    failures = []
    for run_date in dates:
        path = imerg_by_date.get(f"{run_date:%Y%m%d}")
        if path is None:
            continue
        try:
            dataset = xr.open_dataset(path)
            dataset["precipitation"].isel(time=0, lat=900, lon=1800).load()
        except Exception as exc:
            failures.append((run_date, path, exc))
            print(f"imerg_read_failure={run_date} path={path} error={type(exc).__name__}: {exc}")
    print(f"imerg_read_checked={len(dates) - len(failures)} imerg_read_failures={len(failures)}")


def _check_forecast_samples(
    forecast_by_key: dict[tuple[str, str, str], Path],
    models: list[str],
    cycles: list[str],
    dates: list[date],
    expected_steps: list[int],
    sample_count: int,
) -> None:
    for run_date in _sample_dates(dates, sample_count):
        for model in models:
            for cycle in cycles:
                key = (model, f"{run_date:%Y%m%d}", f"{int(cycle):02d}")
                path = forecast_by_key.get(key)
                if path is None:
                    continue
                dataset = open_forecast_file(path)
                leads = [int(value) for value in dataset["lead_hour"].values]
                status = "ok" if leads == expected_steps else f"unexpected_steps={leads}"
                print(f"forecast_sample={model} {run_date} {int(cycle):02d} size={path.stat().st_size} {status}")


if __name__ == "__main__":
    main()
