from pathlib import Path

import pytest

from ai_weather_rainfall.config import load_config


def test_load_default_config():
    cfg = load_config(Path("configs/rainfall_benchmark.yaml"))
    assert cfg.region.name == "middle_lower_yangtze"
    assert cfg.region.lon_min == 108.0
    assert cfg.forecast.lead_hours == [24, 48, 72, 96, 120, 168, 240, 360]
    assert cfg.rainfall.thresholds_mm_24h == [50.0, 100.0]
    assert cfg.period.years == [2025]
    assert cfg.period.primary_comparison_years == [2025]
    assert cfg.period.background_baseline_years == []


def test_load_gfs_config():
    cfg = load_config(Path("configs/gfs_rainfall_benchmark.yaml"))

    assert cfg.forecast.models == ["gfs"]
    assert cfg.period.years == [2023, 2024, 2025]
    assert cfg.paths.gfs_raw == Path("outputs/rainfall_benchmark/raw/gfs")


def test_invalid_region_bounds_fail(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
project: {name: bad, output_root: outputs}
region: {name: bad, lon_min: 123, lon_max: 108, lat_min: 24, lat_max: 35}
period: {years: [2024], season_start: "05-01", season_end: "09-30", primary_comparison_years: [2024], background_baseline_years: []}
forecast: {models: ["aifs"], cycles_utc: ["00"], lead_hours: [24], accumulation_hours: 24}
rainfall: {thresholds_mm_24h: [50.0]}
paths: {aifs_raw: raw/aifs, ifs_raw: raw/ifs, imerg_raw: raw/imerg, processed: processed, metrics: metrics, figures: figures, reports: reports}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lon_min must be smaller"):
        load_config(path)
