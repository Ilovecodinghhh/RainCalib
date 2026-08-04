from pathlib import Path

import numpy as np
import pandas as pd

from ai_weather_rainfall.experimental import (
    GridMonthClimatology,
    IsotonicProbabilityCalibrator,
    LeadQuantileMapping,
    MeanBiasCorrection,
    fractions_skill_score,
    fractions_skill_score_events,
    load_experiment_config,
    paired_date_bootstrap,
    probability_scores,
)


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lead_hour": [24, 24, 48, 48],
            "month": [5, 5, 5, 5],
            "latitude": [30.0, 31.0, 30.0, 31.0],
            "longitude": [120.0, 120.0, 120.0, 120.0],
            "gfs_rain": [0.0, 10.0, 0.0, 20.0],
            "target_rain": [2.0, 12.0, 4.0, 24.0],
        }
    )


def test_load_experiment_config() -> None:
    cfg = load_experiment_config(Path("configs/raincalib_experiments.yaml"))
    assert len(cfg.folds) == 2
    assert cfg.folds[0].name == "rolling_2024"
    assert cfg.seeds == (42, 123, 2024, 3407, 777)


def test_statistical_baselines_are_fitted_from_training_data() -> None:
    frame = _training_frame()
    bias = MeanBiasCorrection().fit(frame).predict(frame)
    quantile = LeadQuantileMapping(quantile_count=5).fit(frame).predict(frame)
    climatology = GridMonthClimatology().fit(frame).predict(frame)
    assert np.allclose(bias, [2.0, 12.0, 4.0, 24.0])
    assert np.all(np.isfinite(quantile))
    assert np.allclose(climatology, [3.0, 18.0, 3.0, 18.0])


def test_probability_scores_and_calibrator() -> None:
    score = np.array([0.0, 1.0, 2.0, 3.0])
    observed = np.array([False, False, True, True])
    calibrated = IsotonicProbabilityCalibrator().fit(score, observed).predict(score)
    metrics = probability_scores(calibrated, observed, reference_rate=0.5)
    assert np.all(np.diff(calibrated) >= 0)
    assert metrics["pr_auc"] == 1.0
    assert metrics["bss"] > 0.0


def test_fss_is_one_for_identical_fields() -> None:
    field = np.array([[0.0, 60.0], [80.0, 0.0]])
    mask = np.ones_like(field, dtype=bool)
    assert fractions_skill_score(field, field, mask, threshold_mm=50.0, neighborhood=1) == 1.0
    event = field >= 50.0
    assert fractions_skill_score_events(event, event, mask, neighborhood=1) == 1.0


def test_paired_date_bootstrap_detects_uniform_improvement() -> None:
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    baseline = observed + 2.0
    predicted = observed + 0.5
    dates = np.array(["2025-08-01", "2025-08-01", "2025-08-02", "2025-08-02"])
    result = paired_date_bootstrap(predicted, baseline, observed, dates, resamples=100, seed=42)
    assert (result["ci_upper"] < 0.0).all()
    assert (result["probability_improved"] == 1.0).all()
