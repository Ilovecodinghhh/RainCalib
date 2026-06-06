import math
from pathlib import Path

import pandas as pd

from ai_weather_rainfall.raincalib import (
    add_time_features,
    categorical_scores_frame,
    continuous_scores_frame,
    load_raincalib_config,
    split_for_date,
)


def test_load_raincalib_config_and_splits():
    cfg = load_raincalib_config(Path("configs/raincalib.yaml"))

    assert split_for_date("2024-06-01", cfg) == "train"
    assert split_for_date("2025-08-20", cfg) == "val"
    assert split_for_date("2025-09-20", cfg) == "test"
    assert split_for_date("2026-01-01", cfg) == "excluded"


def test_add_time_features():
    frame = pd.DataFrame({"observed_date": ["2025-06-01"]})

    result = add_time_features(frame)

    assert "month_sin" in result
    assert "day_of_year_cos" in result


def test_raincalib_metric_helpers():
    continuous = continuous_scores_frame([1.0, 3.0], [2.0, 1.0])
    categorical = categorical_scores_frame([True, False, True], [True, True, False])

    assert continuous["mae"] == 1.5
    assert continuous["bias"] == 0.5
    assert categorical["hits"] == 1
    assert categorical["misses"] == 1
    assert categorical["false_alarms"] == 1
    assert math.isclose(categorical["csi"], 1 / 3)
