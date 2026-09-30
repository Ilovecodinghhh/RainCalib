from datetime import date
import importlib.util
from pathlib import Path
import pandas as pd
import pytest
from ai_weather_rainfall.pipeline import _observation_date_for_window


@pytest.mark.parametrize('lead,expected', [(24, date(2024, 12, 31)), (48, date(2025, 1, 1)), (360, date(2025, 1, 14))])
def test_midnight_window(lead, expected):
    assert _observation_date_for_window(date(2024, 12, 31), lead, 24, '00') == expected


@pytest.mark.parametrize('cycle,lead,hours', [(12, 24, 24), (0, 36, 24), (0, 24, 12), (0, 12, 24)])
def test_reject_mismatched_daily_window(cycle, lead, hours):
    with pytest.raises(ValueError, match='Daily IMERG'):
        _observation_date_for_window(date(2025, 1, 1), lead, hours, cycle)


def test_legacy_feature_selection_and_date_validation():
    spec = importlib.util.spec_from_file_location('prepare', Path('scripts/prepare_00utc_features.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame({'run_date': ['2025-01-01'] * 2, 'observed_date': ['2025-01-02'] * 2,
                          'cycle': [0, 12], 'lead_hour': [48, 48]})
    assert module.select_daily_features(frame)['cycle'].tolist() == [0]
    frame.loc[0, 'observed_date'] = '2025-01-01'
    with pytest.raises(ValueError, match='mismatch'):
        module.select_daily_features(frame)


def test_benchmark_rejects_noon_before_reading_cached_scores(monkeypatch):
    from dataclasses import replace
    from ai_weather_rainfall import pipeline
    from ai_weather_rainfall.config import load_config
    cfg = load_config(Path('configs/gfs_rainfall_benchmark.yaml'))
    cfg = replace(cfg, forecast=replace(cfg.forecast, cycles_utc=['12']))
    monkeypatch.setattr(pipeline, 'load_config', lambda path: cfg)
    monkeypatch.setattr(pipeline, '_ensure_output_dirs', lambda cfg: pytest.fail('Must reject before touching outputs'))
    with pytest.raises(ValueError, match='Daily IMERG'):
        pipeline.run_benchmark(Path('unused.yaml'))
