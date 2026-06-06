from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests

from ai_weather_rainfall.config import load_config
from ai_weather_rainfall.data_gfs import open_gfs_forecast_file
from ai_weather_rainfall.download_plan import (
    configured_dates,
    filter_dates,
    forecast_target_path,
    gfs_aws_grib_url,
    gfs_aws_index_url,
    parse_iso_date,
    required_lead_steps,
)


LOG_DIR = Path("outputs/rainfall_benchmark/download_logs")
MANIFEST_PATH = LOG_DIR / "gfs_downloads.csv"


@dataclass(frozen=True)
class GfsRequest:
    run_date: date
    cycle: str
    target: Path


@dataclass(frozen=True)
class IndexRecord:
    start: int
    end: int | None
    description: str


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    steps = required_lead_steps(cfg.forecast.lead_hours, cfg.forecast.accumulation_hours)
    _validate_supported_steps(steps)
    dates = _select_dates(args, configured_dates(cfg))
    requests_to_fetch = _gfs_requests(cfg, dates, args.cycles)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if cfg.paths.gfs_raw is None:
        raise SystemExit("paths.gfs_raw is required for GFS downloads")
    cfg.paths.gfs_raw.mkdir(parents=True, exist_ok=True)

    print(f"GFS dates selected: {len(dates)}")
    if dates:
        print(f"GFS date range: {dates[0]}..{dates[-1]}")
    print(f"GFS requests selected: {len(requests_to_fetch)}")
    print(f"GFS required cumulative APCP steps: {steps}")
    print("GFS source: AWS noaa-gfs-bdp-pds public bucket")
    if args.dry_run:
        for request in requests_to_fetch[: args.preview]:
            first_step = steps[0]
            print(
                "dry-run "
                f"gfs {request.run_date} {request.cycle} "
                f"idx={gfs_aws_index_url(request.run_date, request.cycle, first_step)} "
                f"target={request.target}"
            )
        if len(requests_to_fetch) > args.preview:
            print(f"dry-run ... {len(requests_to_fetch) - args.preview} more requests")
        return

    session = requests.Session()
    failures = 0
    for index, request in enumerate(requests_to_fetch, start=1):
        print(f"[{index}/{len(requests_to_fetch)}] GFS {request.run_date} {request.cycle}")
        try:
            status, size = _download_one(
                request,
                steps,
                session,
                overwrite=args.overwrite,
                retries=args.retries,
            )
            _append_manifest(request, status, size, "")
            print(f"{status} {request.target} {size} bytes")
        except Exception as exc:
            failures += 1
            _append_manifest(request, "failed", 0, str(exc))
            print(f"failed gfs {request.run_date} {request.cycle}: {exc}")
            if args.stop_on_error:
                raise SystemExit(1) from exc
    if failures:
        raise SystemExit(f"GFS batch finished with {failures} failed downloads")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download cumulative APCP records from historical GFS files.")
    parser.add_argument("--config", type=Path, default=Path("configs/gfs_rainfall_benchmark.yaml"))
    parser.add_argument("--date", action="append", default=[], help="Exact run date to download, YYYY-MM-DD.")
    parser.add_argument("--start-date", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument("--max-days", type=int, help="Limit selected dates after filtering.")
    parser.add_argument("--cycles", nargs="+", help="Override cycles, for example: 00 12.")
    parser.add_argument("--preview", type=int, default=8, help="Number of dry-run rows to print.")
    parser.add_argument("--retries", type=int, default=3, help="Download attempts per request.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload files even if complete files exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected requests without downloading.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed request.")
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


def _gfs_requests(cfg, dates: list[date], cycles: list[str] | None) -> list[GfsRequest]:
    selected_cycles = cycles or cfg.forecast.cycles_utc
    return [
        GfsRequest(
            run_date=run_date,
            cycle=f"{int(cycle):02d}",
            target=forecast_target_path(cfg, "gfs", run_date, cycle),
        )
        for run_date in dates
        for cycle in selected_cycles
    ]


def _validate_supported_steps(steps: list[int]) -> None:
    unsupported = [step for step in steps if step <= 0 or step % 24 != 0]
    if unsupported:
        raise ValueError(
            "The AWS GFS subset downloader expects cumulative 0-N day APCP records; "
            f"unsupported non-24h steps: {unsupported}"
        )


def _download_one(
    request: GfsRequest,
    steps: list[int],
    session: requests.Session,
    *,
    overwrite: bool,
    retries: int,
) -> tuple[str, int]:
    request.target.parent.mkdir(parents=True, exist_ok=True)
    if request.target.exists() and not overwrite:
        existing_steps = _read_existing_steps(request.target)
        if existing_steps is not None and set(steps).issubset(existing_steps):
            return "exists", request.target.stat().st_size

    temp_target = request.target.with_suffix(request.target.suffix + ".download")
    temp_target.unlink(missing_ok=True)
    _remove_cfgrib_indexes(temp_target)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temp_target.unlink(missing_ok=True)
        _remove_cfgrib_indexes(temp_target)
        try:
            print(f"download attempt {attempt}/{retries} -> {temp_target}")
            with temp_target.open("ab") as handle:
                for step in steps:
                    record = _fetch_cumulative_apcp_record(request, step, session)
                    chunk = _download_record_bytes(request, step, record, session)
                    handle.write(chunk)
                    print(f"  f{step:03d} {record.start}-{record.end} {len(chunk)} bytes")
            downloaded_steps = _read_existing_steps(temp_target)
            if downloaded_steps is None or not set(steps).issubset(downloaded_steps):
                raise RuntimeError(f"downloaded GFS file is missing required steps: {temp_target}")
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            print(f"attempt {attempt}/{retries} failed: {exc}")
    if last_error is not None:
        temp_target.unlink(missing_ok=True)
        raise last_error

    _remove_cfgrib_indexes(request.target)
    temp_target.replace(request.target)
    return "downloaded", request.target.stat().st_size


def _fetch_cumulative_apcp_record(request: GfsRequest, step: int, session: requests.Session) -> IndexRecord:
    index_url = gfs_aws_index_url(request.run_date, request.cycle, step)
    response = session.get(index_url, timeout=60)
    response.raise_for_status()
    return select_cumulative_apcp_record(response.text, step)


def select_cumulative_apcp_record(index_text: str, step: int) -> IndexRecord:
    if step % 24 != 0:
        raise ValueError(f"GFS cumulative day APCP records require a multiple-of-24 step, got {step}")
    expected = f":APCP:surface:0-{step // 24} day acc fcst:"
    parsed = _parse_index_lines(index_text)
    for position, (line_number, start, description) in enumerate(parsed):
        if expected in description:
            end = parsed[position + 1][1] - 1 if position + 1 < len(parsed) else None
            return IndexRecord(start=start, end=end, description=description)
    raise KeyError(f"GFS APCP cumulative record not found for f{step:03d}: expected token {expected}")


def _parse_index_lines(index_text: str) -> list[tuple[int, int, str]]:
    parsed: list[tuple[int, int, str]] = []
    for line in index_text.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        try:
            line_number = int(parts[0])
            start = int(parts[1])
        except ValueError:
            continue
        parsed.append((line_number, start, line))
    return parsed


def _download_record_bytes(
    request: GfsRequest,
    step: int,
    record: IndexRecord,
    session: requests.Session,
) -> bytes:
    grib_url = gfs_aws_grib_url(request.run_date, request.cycle, step)
    byte_range = f"bytes={record.start}-" if record.end is None else f"bytes={record.start}-{record.end}"
    response = session.get(grib_url, headers={"Range": byte_range}, timeout=120)
    if response.status_code != 206:
        raise RuntimeError(f"expected HTTP 206 for ranged GFS download, got {response.status_code}: {grib_url}")
    response.raise_for_status()
    return response.content


def _read_existing_steps(path: Path) -> set[int] | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        dataset = open_gfs_forecast_file(path)
    except Exception as exc:
        print(f"existing GFS file cannot be opened and will be redownloaded: {path} ({exc})")
        return None
    if "lead_hour" not in dataset.coords:
        return None
    return {int(value) for value in dataset["lead_hour"].values}


def _remove_cfgrib_indexes(target: Path) -> None:
    for index_path in target.parent.glob(target.name + "*.idx"):
        index_path.unlink(missing_ok=True)


def _append_manifest(request: GfsRequest, status: str, size: int, message: str) -> None:
    first_write = not MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date", "cycle", "model", "status", "size", "path", "message"],
        )
        if first_write:
            writer.writeheader()
        writer.writerow(
            {
                "date": f"{request.run_date:%Y-%m-%d}",
                "cycle": request.cycle,
                "model": "gfs",
                "status": status,
                "size": size,
                "path": str(request.target),
                "message": message,
            }
        )


if __name__ == "__main__":
    main()
