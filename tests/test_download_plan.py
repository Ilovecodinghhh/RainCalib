from datetime import date

from pathlib import Path

from ai_weather_rainfall.config import load_config
from ai_weather_rainfall.download_plan import (
    filter_dates,
    forecast_target_path,
    gfs_aws_grib_url,
    gfs_aws_index_url,
    required_lead_steps,
    season_dates,
)


def test_required_lead_steps_include_window_starts():
    steps = required_lead_steps([24, 48, 168, 240, 360], accumulation_hours=24)

    assert steps == [24, 48, 144, 168, 216, 240, 336, 360]


def test_season_dates_support_same_year_periods():
    dates = season_dates(2025, "05-01", "05-03")

    assert dates == [date(2025, 5, 1), date(2025, 5, 2), date(2025, 5, 3)]


def test_season_dates_support_cross_year_periods():
    dates = season_dates(2025, "12-30", "01-02")

    assert dates == [
        date(2025, 12, 30),
        date(2025, 12, 31),
        date(2026, 1, 1),
        date(2026, 1, 2),
    ]


def test_filter_dates_can_select_exact_dates_and_limit():
    dates = [date(2025, 5, 1), date(2025, 5, 2), date(2025, 5, 3)]

    selected = filter_dates(dates, exact_dates=[date(2025, 5, 3), date(2025, 5, 1)], max_days=1)

    assert selected == [date(2025, 5, 1)]


def test_gfs_forecast_target_path():
    cfg = load_config(Path("configs/gfs_rainfall_benchmark.yaml"))

    target = forecast_target_path(cfg, "gfs", date(2025, 9, 1), "0")

    assert target == Path("outputs/rainfall_benchmark/raw/gfs/gfs_20250901_00_tp_steps.grib2")


def test_gfs_aws_urls():
    run_date = date(2025, 9, 1)

    assert gfs_aws_grib_url(run_date, "0", 24).endswith(
        "/gfs.20250901/00/atmos/gfs.t00z.pgrb2.0p25.f024"
    )
    assert gfs_aws_index_url(run_date, "12", 360).endswith(
        "/gfs.20250901/12/atmos/gfs.t12z.pgrb2.0p25.f360.idx"
    )
