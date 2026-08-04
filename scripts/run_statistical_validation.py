from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.utils.class_weight import compute_sample_weight

from ai_weather_rainfall.experimental import (
    GridMonthClimatology,
    IsotonicProbabilityCalibrator,
    LeadQuantileMapping,
    MeanBiasCorrection,
    deterministic_event_scores,
    extended_continuous_scores,
    fractions_skill_score,
    load_experiment_config,
    paired_date_bootstrap,
    probability_scores,
)
from ai_weather_rainfall.raincalib import choose_probability_threshold, model_feature_columns


FEATURE_COLUMNS = [
    "run_date",
    "observed_date",
    "month",
    "cycle",
    "lead_hour",
    "latitude",
    "longitude",
    "gfs_rain",
    "target_rain",
    "event_50",
    "event_100",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


def main() -> None:
    args = _parse_args()
    cfg = load_experiment_config(args.config)
    frame = pd.read_parquet(cfg.feature_store, columns=FEATURE_COLUMNS)
    frame["observed_date"] = pd.to_datetime(frame["observed_date"])
    output_dir = cfg.output_root / "statistical"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, object]] = []
    all_bootstrap: list[pd.DataFrame] = []
    all_fss: list[dict[str, object]] = []

    for fold_number, fold in enumerate(cfg.folds):
        train = frame.loc[fold.train.mask(frame["observed_date"])].reset_index(drop=True)
        validation = frame.loc[fold.validation.mask(frame["observed_date"])].reset_index(drop=True)
        test = frame.loc[fold.test.mask(frame["observed_date"])].reset_index(drop=True)
        _require_nonempty(fold.name, train, validation, test)
        print(
            f"fold={fold.name} train_rows={len(train)} validation_rows={len(validation)} "
            f"test_rows={len(test)}"
        )

        validation_predictions, test_predictions, probability_sources = _statistical_predictions(
            train,
            validation,
            test,
            max_train_rows=args.max_train_rows,
            seed=args.seed,
        )
        prediction_frame = test[
            ["run_date", "observed_date", "cycle", "lead_hour", "latitude", "longitude", "target_rain"]
        ].copy()
        threshold_metadata: dict[str, dict[str, float]] = {}
        observed_test = test["target_rain"].to_numpy(dtype=float)

        for model_name, predicted_test in test_predictions.items():
            predicted_validation = validation_predictions[model_name]
            prediction_frame[f"prediction__{model_name}"] = predicted_test.astype("float32")
            scores = extended_continuous_scores(predicted_test, observed_test)
            all_metrics.extend(_metric_rows(fold.name, model_name, "continuous", None, scores))
            threshold_metadata[model_name] = {}

            for threshold in cfg.event_thresholds_mm:
                label = f"event_{threshold:g}".replace(".", "_")
                observed_validation_event = validation[label].to_numpy(dtype=bool)
                observed_test_event = test[label].to_numpy(dtype=bool)
                validation_source, test_source = probability_sources.get(
                    (model_name, threshold),
                    (predicted_validation, predicted_test),
                )
                calibrator = IsotonicProbabilityCalibrator().fit(validation_source, observed_validation_event)
                validation_probability = calibrator.predict(validation_source)
                test_probability = calibrator.predict(test_source)
                probability_threshold, _ = choose_probability_threshold(
                    validation_probability,
                    observed_validation_event,
                    metric="ets",
                )
                threshold_metadata[model_name][f"{threshold:g}"] = probability_threshold
                probability_column = f"probability_{threshold:g}mm__{model_name}"
                prediction_frame[probability_column] = test_probability.astype("float32")

                rain_scores = deterministic_event_scores(predicted_test, observed_test, threshold)
                all_metrics.extend(
                    _metric_rows(fold.name, model_name, "rain_threshold", threshold, rain_scores)
                )
                probability_metric = probability_scores(
                    test_probability,
                    observed_test_event,
                    reference_rate=float(train[label].mean()),
                )
                all_metrics.extend(
                    _metric_rows(fold.name, model_name, "probability", threshold, probability_metric)
                )
                decision_scores = deterministic_event_scores(
                    test_probability,
                    observed_test_event.astype(float),
                    probability_threshold,
                )
                all_metrics.extend(
                    _metric_rows(fold.name, model_name, "probability_decision", threshold, decision_scores)
                )

            if model_name != "raw_gfs":
                bootstrap = paired_date_bootstrap(
                    predicted_test,
                    test_predictions["raw_gfs"],
                    observed_test,
                    test["observed_date"].to_numpy(),
                    resamples=cfg.bootstrap_resamples,
                    seed=args.seed + fold_number,
                )
                bootstrap.insert(0, "model", model_name)
                bootstrap.insert(0, "fold", fold.name)
                all_bootstrap.append(bootstrap)

        fold_dir = output_dir / fold.name
        fold_dir.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_parquet(fold_dir / "test_predictions.parquet", index=False)
        (fold_dir / "event_decision_thresholds.json").write_text(
            json.dumps(threshold_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        fold_fss = _fss_rows(prediction_frame, test_predictions.keys(), cfg.event_thresholds_mm, cfg.fss_neighborhoods)
        for row in fold_fss:
            row["fold"] = fold.name
        all_fss.extend(fold_fss)

    metrics = pd.DataFrame(all_metrics)
    bootstrap = pd.concat(all_bootstrap, ignore_index=True)
    fss = pd.DataFrame(all_fss)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_date_bootstrap.csv", index=False)
    fss.to_csv(output_dir / "fss.csv", index=False)
    print(f"metrics={output_dir / 'metrics.csv'}")
    print(f"bootstrap={output_dir / 'paired_date_bootstrap.csv'}")
    print(f"fss={output_dir / 'fss.csv'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-origin statistical RainCalib validation.")
    parser.add_argument("--config", type=Path, default=Path("configs/raincalib_experiments.yaml"))
    parser.add_argument("--max-train-rows", type=int, default=600_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _require_nonempty(name: str, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    for split_name, split in (("train", train), ("validation", validation), ("test", test)):
        if split.empty:
            raise ValueError(f"fold={name} has no rows for split={split_name}")


def _statistical_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    max_train_rows: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[tuple[str, float], tuple[np.ndarray, np.ndarray]]]:
    validation_predictions: dict[str, np.ndarray] = {
        "raw_gfs": validation["gfs_rain"].to_numpy(dtype=float)
    }
    test_predictions: dict[str, np.ndarray] = {"raw_gfs": test["gfs_rain"].to_numpy(dtype=float)}
    probability_sources: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}

    baselines = {
        "mean_bias": MeanBiasCorrection(),
        "quantile_mapping": LeadQuantileMapping(),
        "grid_month_climatology": GridMonthClimatology(),
    }
    for name, model in baselines.items():
        print(f"fit={name}")
        model.fit(train)
        validation_predictions[name] = model.predict(validation)
        test_predictions[name] = model.predict(test)

    sampled_train = train
    if len(sampled_train) > max_train_rows:
        sampled_train = sampled_train.sample(n=max_train_rows, random_state=seed)
    feature_columns = model_feature_columns(sampled_train)
    print(f"fit=raincalib_v1 train_rows={len(sampled_train)} features={len(feature_columns)}")
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=seed,
    )
    regressor.fit(sampled_train[feature_columns].astype("float32"), sampled_train["target_rain"])
    validation_predictions["raincalib_v1"] = np.clip(
        regressor.predict(validation[feature_columns].astype("float32")), 0.0, None
    )
    test_predictions["raincalib_v1"] = np.clip(
        regressor.predict(test[feature_columns].astype("float32")), 0.0, None
    )
    for threshold in (50.0, 100.0):
        label = f"event_{threshold:g}".replace(".", "_")
        event_target = sampled_train[label].to_numpy(dtype="int8")
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=180,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=seed,
        )
        classifier.fit(
            sampled_train[feature_columns].astype("float32"),
            event_target,
            sample_weight=compute_sample_weight(class_weight="balanced", y=event_target),
        )
        probability_sources[("raincalib_v1", threshold)] = (
            classifier.predict_proba(validation[feature_columns].astype("float32"))[:, 1],
            classifier.predict_proba(test[feature_columns].astype("float32"))[:, 1],
        )
    return validation_predictions, test_predictions, probability_sources


def _metric_rows(
    fold: str,
    model: str,
    category: str,
    threshold_mm: float | None,
    scores: dict[str, float],
) -> list[dict[str, object]]:
    return [
        {
            "fold": fold,
            "model": model,
            "category": category,
            "threshold_mm": threshold_mm,
            "metric": metric,
            "value": value,
        }
        for metric, value in scores.items()
    ]


def _fss_rows(
    frame: pd.DataFrame,
    model_names,
    thresholds: tuple[float, ...],
    neighborhoods: tuple[int, ...],
) -> list[dict[str, object]]:
    field_keys = pd.MultiIndex.from_frame(frame[["run_date", "cycle", "lead_hour"]])
    field_codes, unique_fields = pd.factorize(field_keys, sort=True)
    latitudes = np.sort(frame["latitude"].unique())
    longitudes = np.sort(frame["longitude"].unique())
    latitude_codes = np.searchsorted(latitudes, frame["latitude"].to_numpy())
    longitude_codes = np.searchsorted(longitudes, frame["longitude"].to_numpy())
    shape = (len(unique_fields), len(latitudes), len(longitudes))
    observed = np.full(shape, np.nan, dtype="float32")
    observed[field_codes, latitude_codes, longitude_codes] = frame["target_rain"].to_numpy(dtype="float32")
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        predicted = np.full(shape, np.nan, dtype="float32")
        predicted[field_codes, latitude_codes, longitude_codes] = frame[
            f"prediction__{model_name}"
        ].to_numpy(dtype="float32")
        mask = np.isfinite(predicted) & np.isfinite(observed)
        for threshold in thresholds:
            for neighborhood in neighborhoods:
                rows.append(
                    {
                        "model": model_name,
                        "threshold_mm": threshold,
                        "neighborhood": neighborhood,
                        "fss": fractions_skill_score(
                            predicted,
                            observed,
                            mask,
                            threshold_mm=threshold,
                            neighborhood=neighborhood,
                        ),
                    }
                )
    return rows


if __name__ == "__main__":
    main()
