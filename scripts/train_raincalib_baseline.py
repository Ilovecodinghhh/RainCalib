from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.utils.class_weight import compute_sample_weight

from ai_weather_rainfall.raincalib import (
    add_time_features,
    choose_probability_threshold,
    load_raincalib_config,
    model_feature_columns,
)


def main() -> None:
    args = _parse_args()
    cfg = load_raincalib_config(args.config)
    frame = pd.read_parquet(args.features or cfg.feature_store)
    frame = add_time_features(frame)
    train = frame[frame["split"] == "train"].copy()
    val = frame[frame["split"] == "val"].copy()
    if train.empty or val.empty:
        raise SystemExit("feature store must contain non-empty train and val splits")
    if len(train) > cfg.max_train_rows:
        train = train.sample(n=cfg.max_train_rows, random_state=cfg.random_state)

    feature_columns = model_feature_columns(train)
    x_train = train[feature_columns].astype("float32")
    y_train = train["target_rain"].astype("float32")
    x_val = val[feature_columns].astype("float32")

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    print(f"train_rows={len(train)} val_rows={len(val)} features={feature_columns}")

    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=cfg.random_state,
    )
    regressor.fit(x_train, y_train)
    joblib.dump(regressor, cfg.models_dir / "raincalib_regressor.joblib")

    metadata: dict[str, object] = {
        "feature_columns": feature_columns,
        "event_thresholds": cfg.event_thresholds,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "classifier_probability_thresholds": {},
        "optimize_event_metric": cfg.optimize_event_metric,
    }

    for threshold in cfg.event_thresholds:
        label = f"event_{threshold:g}".replace(".", "_")
        y_event = train[label].astype("int8")
        if y_event.nunique() < 2:
            print(f"skip classifier threshold={threshold:g}: only one class in training data")
            continue
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=180,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=cfg.random_state,
        )
        weights = compute_sample_weight(class_weight="balanced", y=y_event)
        classifier.fit(x_train, y_event, sample_weight=weights)
        model_path = cfg.models_dir / f"raincalib_event_{threshold:g}mm.joblib"
        joblib.dump(classifier, model_path)
        probabilities = classifier.predict_proba(x_val)[:, 1]
        observed_event = val[label].astype(bool).to_numpy()
        probability_threshold, val_scores = choose_probability_threshold(
            probabilities,
            observed_event,
            metric=cfg.optimize_event_metric,
        )
        metadata["classifier_probability_thresholds"][f"{threshold:g}"] = probability_threshold
        print(
            f"classifier threshold={threshold:g} prob_threshold={probability_threshold:.2f} "
            f"val_{cfg.optimize_event_metric}={val_scores[cfg.optimize_event_metric]:.4f}"
        )

    metadata_path = cfg.models_dir / "raincalib_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"models={cfg.models_dir}")
    print(f"metadata={metadata_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RainCalib sklearn baseline models.")
    parser.add_argument("--config", type=Path, default=Path("configs/raincalib.yaml"))
    parser.add_argument("--features", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
