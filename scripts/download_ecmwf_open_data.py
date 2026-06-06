from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests
from ecmwf.opendata import Client

from ai_weather_rainfall.config import load_config
from ai_weather_rainfall.data_ecmwf import open_forecast_file
from ai_weather_rainfall.download_plan import (
    active_forecast_models,
    configured_dates,
    ecmwf_open_data_model,
    filter_dates,
    forecast_target_path,
    parse_iso_date,
    required_lead_steps,
)


LOG_DIR = Path("outputs/rainfall_benchmark/download_logs")
MANIFEST_PATH = LOG_DIR / "ecmwf_downloads.csv"


@dataclass(frozen=True)
class ForecastRequest:
    model: str
    run_date: date
    cycle: str
    target: Path


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    steps = required_lead_steps(cfg.forecast.lead_hours, cfg.forecast.accumulation_hours)
    dates = _select_dates(args, configured_dates(cfg))
    requests = _forecast_requests(cfg, dates, args.models, args.cycles)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.paths.aifs_raw.mkdir(parents=True, exist_ok=True)
    cfg.paths.ifs_raw.mkdir(parents=True, exist_ok=True)

    print(f"ECMWF dates selected: {len(dates)}")
    if dates:
        print(f"ECMWF date range: {dates[0]}..{dates[-1]}")
    print(f"ECMWF requests selected: {len(requests)}")
    print(f"ECMWF required steps: {steps}")
    if args.dry_run:
        for request in requests[: args.preview]:
            print(
                "dry-run "
                f"{request.model} {request.run_date} {request.cycle} "
                f"remote={ecmwf_open_data_model(request.model)} target={request.target}"
            )
        if len(requests) > args.preview:
            print(f"dry-run ... {len(requests) - args.preview} more requests")
        return

    clients: dict[str, Client] = {}
    failures = 0
    for index, request in enumerate(requests, start=1):
        print(f"[{index}/{len(requests)}] ECMWF {request.model} {request.run_date} {request.cycle}")
        try:
            status, size = _download_one(
                request,
                steps,
                clients,
                source=args.source,
                overwrite=args.overwrite,
                retries=args.retries,
                skip_unavailable=not args.fail_unavailable,
            )
            _append_manifest(request, status, size, "")
            print(f"{status} {request.target} {size} bytes")
        except Exception as exc:
            failures += 1
            _append_manifest(request, "failed", 0, str(exc))
            print(f"failed {request.model} {request.run_date} {request.cycle}: {exc}")
            if args.stop_on_error:
                raise SystemExit(1) from exc

    if failures:
        raise SystemExit(f"ECMWF batch finished with {failures} failed downloads")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ECMWF Open Data forecast files.")
    parser.add_argument("--config", type=Path, default=Path("configs/rainfall_benchmark.yaml"))
    parser.add_argument("--source", default="aws", help="ecmwf-opendata source, for example aws or ecmwf.")
    parser.add_argument("--date", action="append", default=[], help="Exact run date to download, YYYY-MM-DD.")
    parser.add_argument("--start-date", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument("--max-days", type=int, help="Limit selected dates after filtering.")
    parser.add_argument("--models", nargs="+", help="Override models, for example: aifs ifs.")
    parser.add_argument("--cycles", nargs="+", help="Override cycles, for example: 00 12.")
    parser.add_argument("--preview", type=int, default=8, help="Number of dry-run rows to print.")
    parser.add_argument("--retries", type=int, default=3, help="Download attempts per request.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload files even if complete files exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected requests without downloading.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed request.")
    parser.add_argument(
        "--fail-unavailable",
        action="store_true",
        help="Treat remote 404/unavailable ECMWF files as failures instead of manifest skips.",
    )
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


def _forecast_requests(cfg, dates: list[date], models: list[str] | None, cycles: list[str] | None) -> list[ForecastRequest]:
    requests: list[ForecastRequest] = []
    for run_date in dates:
        active_models = models or active_forecast_models(cfg, run_date.year)
        active_cycles = cycles or cfg.forecast.cycles_utc
        for model in active_models:
            for cycle in active_cycles:
                requests.append(
                    ForecastRequest(
                        model=model,
                        run_date=run_date,
                        cycle=f"{int(cycle):02d}",
                        target=forecast_target_path(cfg, model, run_date, cycle),
                    )
                )
    return requests


def _download_one(
    request: ForecastRequest,
    steps: list[int],
    clients: dict[str, Client],
    *,
    source: str,
    overwrite: bool,
    retries: int,
    skip_unavailable: bool,
) -> tuple[str, int]:
    request.target.parent.mkdir(parents=True, exist_ok=True)
    remote_model = ecmwf_open_data_model(request.model)
    client = clients.get(remote_model)
    if client is None:
        client = Client(source=source, model=remote_model)
        clients[remote_model] = client

    if request.target.exists() and not overwrite:
        existing_steps = _read_existing_steps(request.target)
        if existing_steps is not None:
            missing_steps = sorted(set(steps) - existing_steps)
            if not missing_steps:
                return "exists", request.target.stat().st_size
            if existing_steps:
                unavailable = _missing_remote_indexes(client, request, missing_steps)
                if unavailable:
                    message = _unavailable_message(unavailable)
                    if skip_unavailable:
                        print(message)
                        return "partial_unavailable", request.target.stat().st_size
                    raise FileNotFoundError(message)
                return _repair_missing_steps(request, missing_steps, steps, client, retries)

    temp_target = request.target.with_suffix(request.target.suffix + ".download")
    temp_target.unlink(missing_ok=True)
    _remove_cfgrib_indexes(temp_target)

    unavailable = _missing_remote_indexes(client, request, steps)
    if unavailable:
        message = _unavailable_message(unavailable)
        if skip_unavailable:
            print(message)
            return "unavailable", 0
        raise FileNotFoundError(message)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temp_target.unlink(missing_ok=True)
        _remove_cfgrib_indexes(temp_target)
        try:
            print(f"download attempt {attempt}/{retries} -> {temp_target}")
            client.retrieve(
                date=f"{request.run_date:%Y%m%d}",
                time=int(request.cycle),
                type="fc",
                step=steps,
                param="tp",
                target=str(temp_target),
            )
            if not _existing_has_steps(temp_target, steps):
                raise RuntimeError(f"downloaded file is missing required steps: {temp_target}")
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            print(f"attempt {attempt}/{retries} failed: {exc}")
    if last_error is not None:
        raise last_error
    _remove_cfgrib_indexes(request.target)
    temp_target.replace(request.target)
    return "downloaded", request.target.stat().st_size


def _existing_has_steps(path: Path, required_steps: list[int]) -> bool:
    existing = _read_existing_steps(path)
    if existing is None:
        return False
    missing = sorted(set(required_steps) - existing)
    if missing:
        print(f"existing file is missing steps {missing} and will be redownloaded: {path}")
        return False
    return True


def _read_existing_steps(path: Path) -> set[int] | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        dataset = open_forecast_file(path)
    except Exception as exc:
        print(f"existing file cannot be opened and will be redownloaded: {path} ({exc})")
        return None
    if "lead_hour" not in dataset.coords:
        return None
    return {int(value) for value in dataset["lead_hour"].values}


def _repair_missing_steps(
    request: ForecastRequest,
    missing_steps: list[int],
    required_steps: list[int],
    client: Client,
    retries: int,
) -> tuple[str, int]:
    missing_target = request.target.with_suffix(request.target.suffix + ".missing")
    repair_target = request.target.with_suffix(request.target.suffix + ".repair")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        missing_target.unlink(missing_ok=True)
        repair_target.unlink(missing_ok=True)
        _remove_cfgrib_indexes(missing_target)
        _remove_cfgrib_indexes(repair_target)
        try:
            print(f"repair attempt {attempt}/{retries} missing steps {missing_steps} -> {missing_target}")
            client.retrieve(
                date=f"{request.run_date:%Y%m%d}",
                time=int(request.cycle),
                type="fc",
                step=missing_steps,
                param="tp",
                target=str(missing_target),
            )
            downloaded_steps = _read_existing_steps(missing_target)
            if downloaded_steps is None or not set(missing_steps).issubset(downloaded_steps):
                raise RuntimeError(f"downloaded repair file is missing steps: {missing_target}")
            with repair_target.open("wb") as output:
                with request.target.open("rb") as existing_file:
                    shutil.copyfileobj(existing_file, output)
                with missing_target.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output)
            missing_target.unlink(missing_ok=True)
            _remove_cfgrib_indexes(missing_target)
            if not _existing_has_steps(repair_target, required_steps):
                raise RuntimeError(f"repaired file is missing required steps: {repair_target}")
            _remove_cfgrib_indexes(request.target)
            repair_target.replace(request.target)
            return "repaired", request.target.stat().st_size
        except Exception as exc:
            last_error = exc
            print(f"repair attempt {attempt}/{retries} failed: {exc}")
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"repair failed without an error: {request.target}")


def _remove_cfgrib_indexes(target: Path) -> None:
    for index_path in target.parent.glob(target.name + "*.idx"):
        index_path.unlink(missing_ok=True)


def _missing_remote_indexes(client: Client, request: ForecastRequest, steps: list[int]) -> dict[int, str]:
    urls = _remote_index_urls(client, request, steps)
    missing: dict[int, str] = {}
    for step, url in urls.items():
        try:
            response = client.session.head(url, allow_redirects=True, timeout=30, verify=client.verify)
        except requests.RequestException as exc:
            print(f"availability check warning for step {step}: {exc}")
            continue
        if response.status_code == 404:
            missing[step] = url
        elif response.status_code >= 400:
            print(f"availability check warning for step {step}: HTTP {response.status_code} {url}")
    return missing


def _remote_index_urls(client: Client, request: ForecastRequest, steps: list[int]) -> dict[int, str]:
    result = client._get_urls(
        use_index=False,
        date=f"{request.run_date:%Y%m%d}",
        time=int(request.cycle),
        type="fc",
        step=steps,
        param="tp",
        target=str(request.target),
    )
    index_urls: dict[int, str] = {}
    for step, url in zip(steps, result.urls):
        index_urls[int(step)] = f"{url.rsplit('.', 1)[0]}.index"
    return index_urls


def _unavailable_message(unavailable: dict[int, str]) -> str:
    steps = ", ".join(str(step) for step in sorted(unavailable))
    first_url = unavailable[sorted(unavailable)[0]]
    return f"remote ECMWF indexes unavailable for steps [{steps}]; first missing URL: {first_url}"


def _append_manifest(request: ForecastRequest, status: str, size: int, message: str) -> None:
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
                "model": request.model,
                "status": status,
                "size": size,
                "path": str(request.target),
                "message": message,
            }
        )


if __name__ == "__main__":
    main()
