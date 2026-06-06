from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ai_weather_rainfall.raincalib import (
    add_time_features,
    categorical_scores_frame,
    continuous_scores_frame,
    load_raincalib_config,
)


def main() -> None:
    args = _parse_args()
    cfg = load_raincalib_config(args.config)
    metadata_path = cfg.models_dir / "raincalib_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = list(metadata["feature_columns"])

    frame = pd.read_parquet(args.features or cfg.feature_store)
    frame = add_time_features(frame)
    split = args.split
    eval_frame = frame[frame["split"] == split].copy()
    if eval_frame.empty:
        raise SystemExit(f"feature store has no rows for split={split}")

    regressor = joblib.load(cfg.models_dir / "raincalib_regressor.joblib")
    x_eval = eval_frame[feature_columns].astype("float32")
    eval_frame["raincalib_rain"] = np.clip(regressor.predict(x_eval), 0.0, None).astype("float32")

    for threshold in cfg.event_thresholds:
        model_path = cfg.models_dir / f"raincalib_event_{threshold:g}mm.joblib"
        if not model_path.exists():
            continue
        classifier = joblib.load(model_path)
        eval_frame[f"raincalib_event_prob_{threshold:g}mm"] = classifier.predict_proba(x_eval)[:, 1].astype("float32")

    cfg.predictions_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = cfg.predictions_dir / f"raincalib_{split}_predictions.parquet"
    eval_frame.to_parquet(predictions_path, index=False)

    continuous = _continuous_metrics(eval_frame)
    categorical = _categorical_metrics(
        eval_frame,
        cfg.event_thresholds,
        metadata.get("classifier_probability_thresholds", {}),
    )
    continuous_path = cfg.metrics_dir / f"raincalib_{split}_continuous_metrics.csv"
    categorical_path = cfg.metrics_dir / f"raincalib_{split}_categorical_metrics.csv"
    pd.DataFrame(continuous).to_csv(continuous_path, index=False)
    pd.DataFrame(categorical).to_csv(categorical_path, index=False)

    report_path = cfg.reports_dir / "raincalib_report.md"
    report_path.write_text(
        _report_text(
            split=split,
            rows=len(eval_frame),
            predictions_path=predictions_path,
            continuous_path=continuous_path,
            categorical_path=categorical_path,
            continuous=pd.DataFrame(continuous),
            categorical=pd.DataFrame(categorical),
            probability_thresholds=metadata.get("classifier_probability_thresholds", {}),
        ),
        encoding="utf-8",
    )
    print(f"predictions={predictions_path}")
    print(f"continuous_metrics={continuous_path}")
    print(f"categorical_metrics={categorical_path}")
    print(f"report={report_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RainCalib baseline models.")
    parser.add_argument("--config", type=Path, default=Path("configs/raincalib.yaml"))
    parser.add_argument("--features", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def _continuous_metrics(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subset, group in [("all", frame), *[(f"lead_{int(lead)}", g) for lead, g in frame.groupby("lead_hour")]]:
        for model, column in [("raw_gfs", "gfs_rain"), ("raincalib", "raincalib_rain")]:
            scores = continuous_scores_frame(group[column].to_numpy(), group["target_rain"].to_numpy())
            rows.append({"subset": subset, "model": model, **scores, "samples": int(len(group))})
    return rows


def _categorical_metrics(
    frame: pd.DataFrame,
    thresholds: list[float],
    probability_thresholds: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups = [("all", frame), *[(f"lead_{int(lead)}", g) for lead, g in frame.groupby("lead_hour")]]
    for subset, group in groups:
        for threshold in thresholds:
            observed_event = group["target_rain"].to_numpy() >= threshold
            raw_scores = categorical_scores_frame(group["gfs_rain"].to_numpy() >= threshold, observed_event)
            rows.append({"subset": subset, "model": "raw_gfs", "threshold_mm": threshold, **raw_scores})
            corrected_scores = categorical_scores_frame(group["raincalib_rain"].to_numpy() >= threshold, observed_event)
            rows.append({"subset": subset, "model": "raincalib_rain", "threshold_mm": threshold, **corrected_scores})
            prob_col = f"raincalib_event_prob_{threshold:g}mm"
            prob_threshold = float(probability_thresholds.get(f"{threshold:g}", 0.5))
            if prob_col in group:
                classifier_scores = categorical_scores_frame(group[prob_col].to_numpy() >= prob_threshold, observed_event)
                rows.append(
                    {
                        "subset": subset,
                        "model": "raincalib_event_classifier",
                        "threshold_mm": threshold,
                        "probability_threshold": prob_threshold,
                        **classifier_scores,
                    }
                )
    return rows


def _report_text(
    *,
    split: str,
    rows: int,
    predictions_path: Path,
    continuous_path: Path,
    categorical_path: Path,
    continuous: pd.DataFrame,
    categorical: pd.DataFrame,
    probability_thresholds: dict[str, float],
) -> str:
    lines = [
        "# RainCalib Report\n\n",
        "This report evaluates a sklearn RainCalib baseline that post-processes raw GFS rainfall against IMERG observations.\n\n",
        "**Important:** This is a research benchmark, not an official warning service.\n\n",
        f"- Evaluation split: {split}\n",
        f"- Evaluation rows: {rows}\n",
        f"- Predictions: `{predictions_path}`\n",
        f"- Continuous metrics: `{continuous_path}`\n",
        f"- Categorical metrics: `{categorical_path}`\n\n",
        "## Continuous Skill\n\n",
    ]
    overall = continuous[continuous["subset"] == "all"].copy()
    lines.append("| Model | MAE | RMSE | Bias | Pearson r | Samples |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for _, row in overall.sort_values("model").iterrows():
        lines.append(
            f"| {row['model']} | {row['mae']:.3f} | {row['rmse']:.3f} | "
            f"{row['bias']:.3f} | {row['pearson_r']:.3f} | {int(row['samples'])} |\n"
        )
    lines.append("\n## Event Skill\n\n")
    event = categorical[categorical["subset"] == "all"].copy()
    lines.append("| Threshold | Model | Hits | Misses | False alarms | CSI | POD | FAR | ETS |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, row in event.sort_values(["threshold_mm", "model"]).iterrows():
        lines.append(
            f"| {row['threshold_mm']:g} mm | {row['model']} | {int(row['hits'])} | "
            f"{int(row['misses'])} | {int(row['false_alarms'])} | {row['csi']:.3f} | "
            f"{row['pod']:.3f} | {row['far']:.3f} | {row['ets']:.3f} |\n"
        )
    lines.append("\n## Event Classifier Probability Thresholds\n\n")
    if probability_thresholds:
        for threshold, value in sorted(probability_thresholds.items(), key=lambda item: float(item[0])):
            lines.append(f"- {threshold} mm: {float(value):.2f}\n")
    else:
        lines.append("No event classifiers were available.\n")
    return "".join(lines)


if __name__ == "__main__":
    main()
