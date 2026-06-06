from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

from ai_weather_rainfall.config import load_config
from ai_weather_rainfall.data_gfs import open_gfs_forecast_file
from ai_weather_rainfall.download_plan import configured_dates, filter_dates, parse_iso_date, required_lead_steps


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if cfg.paths.gfs_raw is None:
        raise SystemExit("paths.gfs_raw is required for GFS coverage checks")
    dates = _select_dates(args, configured_dates(cfg))
    expected_dates = {f"{item:%Y%m%d}" for item in dates}
    expected_steps = required_lead_steps(cfg.forecast.lead_hours, cfg.forecast.accumulation_hours)

    gfs_files = sorted(cfg.paths.gfs_raw.glob("*.grib2"))
    gfs_by_key = _gfs_files_by_key(gfs_files)
    missing = []
    by_cycle: dict[str, int] = defaultdict(int)
    for run_date in dates:
        for cycle in cfg.forecast.cycles_utc:
            key = (f"{run_date:%Y%m%d}", f"{int(cycle):02d}")
            if key in gfs_by_key:
                by_cycle[f"{int(cycle):02d}"] += 1
            else:
                missing.append(key)
    extra = sorted(key for key in gfs_by_key if key[0] not in expected_dates)
    temp_files = sorted(cfg.paths.gfs_raw.glob("*.download"))
    zero_files = [path for path in gfs_files if path.stat().st_size == 0]

    print(f"configured_dates={dates[0]}..{dates[-1]} days={len(dates)}")
    print(f"gfs_expected={len(dates) * len(cfg.forecast.cycles_utc)}")
    print(f"gfs_present={len(dates) * len(cfg.forecast.cycles_utc) - len(missing)}")
    print(f"gfs_by_cycle={dict(sorted(by_cycle.items()))}")
    print(f"gfs_missing={len(missing)} gfs_extra={len(extra)}")
    print(f"zero_files={len(zero_files)} temp_files={len(temp_files)}")
    print(f"expected_steps={expected_steps}")
    if missing:
        print("first_missing_gfs=" + ", ".join(" ".join(item) for item in missing[:10]))
    if extra:
        print("first_extra_gfs=" + ", ".join(" ".join(item) for item in extra[:10]))
    if zero_files:
        print("zero_file_paths=" + ", ".join(str(path) for path in zero_files[:10]))
    if temp_files:
        print("temp_file_paths=" + ", ".join(str(path) for path in temp_files[:10]))

    _check_gfs_samples(gfs_by_key, cfg.forecast.cycles_utc, dates, expected_steps, args.sample_count)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check configured GFS download coverage.")
    parser.add_argument("--config", type=Path, default=Path("configs/gfs_rainfall_benchmark.yaml"))
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--date", action="append", default=[], help="Exact run date to check, YYYY-MM-DD.")
    parser.add_argument("--start-date", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument("--max-days", type=int, help="Limit selected dates after filtering.")
    return parser.parse_args()


def _select_dates(args: argparse.Namespace, dates: list[date]) -> list[date]:
    exact_dates = [parse_iso_date(value) for value in args.date] if args.date else None
    start_date = parse_iso_date(args.start_date) if args.start_date else None
    end_date = parse_iso_date(args.end_date) if args.end_date else None
    return filter_dates(
        dates,
        start_date=start_date,
        end_date=end_date,
        exact_dates=exact_dates,
        max_days=args.max_days,
    )


def _gfs_files_by_key(paths: list[Path]) -> dict[tuple[str, str], Path]:
    by_key: dict[tuple[str, str], Path] = {}
    for path in paths:
        match = re.search(r"^gfs_(20\d{6})_(\d{2})_tp_steps\.grib2$", path.name)
        if match:
            by_key[(match.group(1), match.group(2))] = path
    return by_key


def _sample_dates(dates: list[date], count: int) -> list[date]:
    if len(dates) <= count:
        return dates
    if count <= 1:
        return [dates[0]]
    indexes = sorted({round(index * (len(dates) - 1) / (count - 1)) for index in range(count)})
    return [dates[index] for index in indexes]


def _check_gfs_samples(
    gfs_by_key: dict[tuple[str, str], Path],
    cycles: list[str],
    dates: list[date],
    expected_steps: list[int],
    sample_count: int,
) -> None:
    for run_date in _sample_dates(dates, sample_count):
        for cycle in cycles:
            key = (f"{run_date:%Y%m%d}", f"{int(cycle):02d}")
            path = gfs_by_key.get(key)
            if path is None:
                continue
            dataset = open_gfs_forecast_file(path)
            leads = [int(value) for value in dataset["lead_hour"].values]
            status = "ok" if leads == expected_steps else f"unexpected_steps={leads}"
            print(f"gfs_sample={run_date} {int(cycle):02d} size={path.stat().st_size} {status}")


if __name__ == "__main__":
    main()
