from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ai_weather_rainfall.config import load_config
from ai_weather_rainfall.download_plan import (
    configured_dates,
    filter_dates,
    imerg_target_path,
    imerg_url,
    parse_iso_date,
)


TOKEN_PATH = Path("token.txt")
NETRC_PATH = Path(".earthdata.netrc")
COOKIE_PATH = Path(".urs_cookies")
LOG_DIR = Path("outputs/rainfall_benchmark/download_logs")
AUTH_HEADERS_PATH = LOG_DIR / "gesdisc_auth_headers.txt"
MANIFEST_PATH = LOG_DIR / "imerg_downloads.csv"


@dataclass(frozen=True)
class AuthConfig:
    mode: str
    username: str | None = None
    password: str | None = None
    bearer_token: str | None = None


class DownloadError(RuntimeError):
    pass


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    dates = _select_dates(args, configured_dates(cfg))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.paths.imerg_raw.mkdir(parents=True, exist_ok=True)

    print(f"IMERG dates selected: {len(dates)}")
    if dates:
        print(f"IMERG date range: {dates[0]}..{dates[-1]}")
    if args.dry_run:
        for run_date in dates[: args.preview]:
            print(f"dry-run {run_date} {imerg_target_path(cfg, run_date)}")
        if len(dates) > args.preview:
            print(f"dry-run ... {len(dates) - args.preview} more dates")
        return

    auth = _detect_auth()
    if auth.username and auth.password:
        _write_netrc(auth.username, auth.password)
    print(f"IMERG auth mode: {auth.mode}")

    failures = 0
    for index, run_date in enumerate(dates, start=1):
        target = imerg_target_path(cfg, run_date)
        url = imerg_url(run_date)
        print(f"[{index}/{len(dates)}] IMERG {run_date} -> {target}")
        try:
            status, size = _download_one(url, target, auth, overwrite=args.overwrite)
            _append_manifest(run_date, target, status, size, "")
            print(f"{status} {target} {size} bytes")
        except DownloadError as exc:
            failures += 1
            _append_manifest(run_date, target, "failed", 0, str(exc))
            print(f"failed {run_date}: {exc}")
            if args.stop_on_error:
                raise SystemExit(1) from exc

    if failures:
        raise SystemExit(f"IMERG batch finished with {failures} failed downloads")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NASA GPM IMERG daily files from GES DISC.")
    parser.add_argument("--config", type=Path, default=Path("configs/rainfall_benchmark.yaml"))
    parser.add_argument("--date", action="append", default=[], help="Exact date to download, YYYY-MM-DD.")
    parser.add_argument("--start-date", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument("--max-days", type=int, help="Limit selected dates after filtering.")
    parser.add_argument("--preview", type=int, default=8, help="Number of dry-run rows to print.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload files that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected dates without downloading.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed date.")
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


def _download_one(url: str, target: Path, auth: AuthConfig, *, overwrite: bool) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    part_target = target.with_suffix(target.suffix + ".part")
    if overwrite:
        target.unlink(missing_ok=True)
        part_target.unlink(missing_ok=True)

    size = _resolve_content_length(url, auth)
    if target.exists() and target.stat().st_size == size:
        return "exists", size
    if target.exists() and target.stat().st_size != size and not part_target.exists():
        target.replace(part_target)

    print(f"downloading with resume to {part_target}")
    result = _run_curl_download(url, part_target, auth)
    if result.returncode != 0:
        raise DownloadError(f"curl download failed with exit code {result.returncode}")
    actual = part_target.stat().st_size
    if actual != size:
        raise DownloadError(f"partial file has {actual} bytes, expected {size}")
    part_target.replace(target)
    return "downloaded", size


def _resolve_content_length(url: str, auth: AuthConfig) -> int:
    result = _run_curl_headers(url, auth)
    if result.returncode != 0:
        resolution_url = _extract_resolution_url()
        if resolution_url:
            raise DownloadError(
                "GES DISC pre-authorization is required. Approve this URL with the "
                f"Earthdata account from token.txt: {resolution_url}"
            )
        raise DownloadError(f"could not resolve GES DISC headers, curl exit code {result.returncode}")
    size = _extract_content_length()
    if not size:
        raise DownloadError("could not resolve content length from GES DISC headers")
    return size


def _run_curl_headers(url: str, auth: AuthConfig) -> subprocess.CompletedProcess[str]:
    command = [
        "curl",
        "-L",
        "-I",
        "--fail",
        "--http1.1",
        "--ssl-no-revoke",
        "--connect-timeout",
        "120",
        "--max-time",
        "240",
        "-D",
        str(AUTH_HEADERS_PATH),
        "-o",
        "NUL",
    ]
    command.extend(_curl_auth_args(auth))
    command.append(url)
    return subprocess.run(command, check=False, text=True)


def _run_curl_download(url: str, target: Path, auth: AuthConfig) -> subprocess.CompletedProcess[str]:
    command = [
        "curl",
        "-L",
        "--fail",
        "--http1.1",
        "--ssl-no-revoke",
        "--retry",
        "20",
        "--retry-delay",
        "10",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--connect-timeout",
        "120",
        "--max-time",
        "3600",
        "--speed-time",
        "300",
        "--speed-limit",
        "1024",
        "-c",
        str(COOKIE_PATH),
        "-b",
        str(COOKIE_PATH),
        "-D",
        str(AUTH_HEADERS_PATH),
        "-o",
        str(target),
    ]
    command.extend(_curl_auth_args(auth))
    command.append(url)
    return subprocess.run(command, check=False, text=True)


def _curl_auth_args(auth: AuthConfig) -> list[str]:
    if auth.username and auth.password:
        return ["--netrc-file", str(NETRC_PATH), "-c", str(COOKIE_PATH), "-b", str(COOKIE_PATH)]
    if auth.bearer_token:
        return ["-H", f"Authorization: Bearer {auth.bearer_token}"]
    return []


def _detect_auth() -> AuthConfig:
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if username and password:
        return AuthConfig(mode="env-basic", username=username, password=password)

    if not TOKEN_PATH.exists():
        return AuthConfig(mode="none")

    secret = TOKEN_PATH.read_text(encoding="utf-8-sig").strip()
    if not secret:
        return AuthConfig(mode="empty-token-file")

    parsed = _parse_key_value_secret(secret)
    username = _first_value(parsed, ["EARTHDATA_USERNAME", "USERNAME", "USER", "LOGIN"])
    password = _first_value(parsed, ["EARTHDATA_PASSWORD", "PASSWORD", "PASS"])
    if username and password:
        return AuthConfig(mode="token-file-key-values", username=username, password=password)

    bearer = _find_bearer_token(secret, parsed)
    if bearer:
        return AuthConfig(mode="token-file-bearer", bearer_token=bearer)

    lines = [line.strip() for line in secret.splitlines() if line.strip()]
    if len(lines) >= 2 and not lines[0].lower().startswith("bearer "):
        return AuthConfig(mode="token-file-two-line-basic", username=lines[0], password=lines[1])
    if len(lines) == 1 and ":" in lines[0] and not lines[0].lower().startswith("bearer "):
        username, password = lines[0].split(":", 1)
        return AuthConfig(mode="token-file-colon-basic", username=username.strip(), password=password.strip())

    token = secret.removeprefix("Bearer ").removeprefix("bearer ").strip()
    return AuthConfig(mode="token-file-bearer", bearer_token=token)


def _write_netrc(username: str, password: str) -> None:
    NETRC_PATH.write_text(
        "\n".join(
            [
                f"machine urs.earthdata.nasa.gov login {username} password {password}",
                f"machine data.gesdisc.earthdata.nasa.gov login {username} password {password}",
                f"machine gpm1.gesdisc.eosdis.nasa.gov login {username} password {password}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _parse_key_value_secret(secret: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in secret.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.lower().startswith("export "):
            key = key[7:].strip()
        values[key.upper()] = value.strip().strip('"').strip("'")
    return values


def _first_value(values: dict[str, str], keys: list[str]) -> str | None:
    for key in keys:
        value = values.get(key)
        if value:
            return value
    return None


def _find_bearer_token(secret: str, parsed: dict[str, str]) -> str | None:
    for key in ["EARTHDATA_TOKEN", "TOKEN", "ACCESS_TOKEN", "BEARER"]:
        value = parsed.get(key)
        if value:
            return value.removeprefix("Bearer ").removeprefix("bearer ").strip()

    match = re.search(r"([A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,})", secret)
    if match:
        return match.group(1)

    lines = [line.strip() for line in secret.splitlines() if line.strip()]
    if len(lines) == 1 and len(lines[0]) > 80:
        return lines[0].removeprefix("Bearer ").removeprefix("bearer ").strip()
    return None


def _extract_resolution_url() -> str | None:
    if not AUTH_HEADERS_PATH.exists():
        return None
    text = AUTH_HEADERS_PATH.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"resolution_url=(https://[^\s&]+)", text)
    if not match:
        return None
    return match.group(1)


def _extract_content_length() -> int | None:
    if not AUTH_HEADERS_PATH.exists():
        return None
    text = AUTH_HEADERS_PATH.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Content-Length:\s*(\d+)", text, flags=re.I)
    return int(matches[-1]) if matches else None


def _append_manifest(run_date: date, target: Path, status: str, size: int, message: str) -> None:
    first_write = not MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "status", "size", "path", "message"])
        if first_write:
            writer.writeheader()
        writer.writerow(
            {
                "date": f"{run_date:%Y-%m-%d}",
                "status": status,
                "size": size,
                "path": str(target),
                "message": message,
            }
        )


if __name__ == "__main__":
    main()
