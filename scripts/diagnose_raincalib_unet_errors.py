from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ai_weather_rainfall.deep.dataset import load_unet_config
from ai_weather_rainfall.raincalib import continuous_scores_frame


INTENSITY_BINS = [-0.001, 1.0, 10.0, 25.0, 50.0, 100.0, np.inf]
INTENSITY_LABELS = ["0-1", "1-10", "10-25", "25-50", "50-100", "100+"]


def main() -> None:
    args = _parse_args()
    cfg = load_unet_config(args.config)
    run_name = args.run_name or cfg.run_name
    predictions_path = args.predictions or (cfg.predictions_dir / f"{run_name}_{args.split}_predictions.npz")
    model_label = args.model_label or run_name
    predictions = np.load(predictions_path, allow_pickle=False)
    raw_fields = predictions["raw_gfs"].astype("float32")
    corrected_fields = _corrected_fields(predictions, run_name)
    target_fields = predictions["target"].astype("float32")
    mask_fields = predictions["mask"].astype(bool)

    tensor = np.load(cfg.tensor_store, allow_pickle=False)
    split_values = tensor["split"].astype(str)
    indexes = np.flatnonzero(split_values == args.split)
    if len(indexes) != raw_fields.shape[0]:
        raise SystemExit(
            f"Prediction field count {raw_fields.shape[0]} does not match tensor split {args.split} count {len(indexes)}"
        )
    metadata = _metadata_frame(tensor, indexes)
    flat = _flat_frame(raw_fields, corrected_fields, target_fields, mask_fields, metadata, model_label)
    spatial = _spatial_frame(
        raw_fields,
        corrected_fields,
        target_fields,
        mask_fields,
        tensor["latitude"].astype("float32"),
        tensor["longitude"].astype("float32"),
        model_label,
    )

    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{run_name}_{args.split}_diagnostics"
    by_lead = _group_scores(flat, "lead_hour", model_label)
    by_cycle = _group_scores(flat, "cycle", model_label)
    by_month = _group_scores(flat, "month", model_label)
    by_intensity = _group_scores(flat, "intensity_bin", model_label)
    paths = {
        "lead": cfg.metrics_dir / f"{prefix}_by_lead.csv",
        "cycle": cfg.metrics_dir / f"{prefix}_by_cycle.csv",
        "month": cfg.metrics_dir / f"{prefix}_by_month.csv",
        "intensity": cfg.metrics_dir / f"{prefix}_by_intensity.csv",
        "spatial": cfg.metrics_dir / f"{prefix}_spatial.csv",
    }
    by_lead.to_csv(paths["lead"], index=False)
    by_cycle.to_csv(paths["cycle"], index=False)
    by_month.to_csv(paths["month"], index=False)
    by_intensity.to_csv(paths["intensity"], index=False)
    spatial.to_csv(paths["spatial"], index=False)
    report_path = cfg.reports_dir / f"{prefix}_report.md"
    report_path.write_text(
        _report_text(
            run_name=run_name,
            split=args.split,
            predictions_path=predictions_path,
            model_label=model_label,
            by_lead=by_lead,
            by_intensity=by_intensity,
            paths=paths,
        ),
        encoding="utf-8",
    )
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"report={report_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose RainCalib U-Net errors by lead, season, intensity, and grid cell.")
    parser.add_argument("--config", type=Path, default=Path("configs/raincalib_unet.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-name")
    parser.add_argument("--model-label")
    parser.add_argument("--predictions", type=Path)
    return parser.parse_args()


def _corrected_fields(predictions, run_name: str) -> np.ndarray:
    for key in ("corrected_rain", run_name, "raincalib_unet"):
        if key in predictions:
            return predictions[key].astype("float32")
    raise KeyError("Predictions file must contain corrected_rain, run_name, or raincalib_unet")


def _metadata_frame(tensor, indexes: np.ndarray) -> pd.DataFrame:
    observed_dates = tensor["observed_date"].astype(str)[indexes]
    observed = pd.to_datetime(observed_dates)
    return pd.DataFrame(
        {
            "field_index": np.arange(len(indexes)),
            "lead_hour": tensor["lead_hour"].astype("float32")[indexes],
            "cycle": tensor["cycle"].astype("float32")[indexes],
            "month": observed.month.astype(int),
            "observed_date": observed_dates,
        }
    )


def _flat_frame(
    raw_fields: np.ndarray,
    corrected_fields: np.ndarray,
    target_fields: np.ndarray,
    mask_fields: np.ndarray,
    metadata: pd.DataFrame,
    model_label: str,
) -> pd.DataFrame:
    valid = mask_fields.astype(bool)
    field_ids = np.broadcast_to(np.arange(raw_fields.shape[0])[:, None, None], raw_fields.shape)[valid]
    flat = pd.DataFrame(
        {
            "field_index": field_ids,
            "raw_gfs": raw_fields[valid],
            model_label: corrected_fields[valid],
            "target": target_fields[valid],
        }
    )
    flat = flat.merge(metadata.drop(columns=["observed_date"]), on="field_index", how="left")
    flat["intensity_bin"] = pd.cut(
        flat["target"],
        bins=INTENSITY_BINS,
        labels=INTENSITY_LABELS,
        include_lowest=True,
        right=True,
    )
    return flat


def _spatial_frame(
    raw_fields: np.ndarray,
    corrected_fields: np.ndarray,
    target_fields: np.ndarray,
    mask_fields: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    model_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for row_index, latitude in enumerate(latitudes):
        for col_index, longitude in enumerate(longitudes):
            valid = mask_fields[:, row_index, col_index]
            if not valid.any():
                continue
            raw = raw_fields[:, row_index, col_index][valid]
            corrected = corrected_fields[:, row_index, col_index][valid]
            target = target_fields[:, row_index, col_index][valid]
            raw_scores = continuous_scores_frame(raw, target)
            corrected_scores = continuous_scores_frame(corrected, target)
            rows.append(
                {
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "samples": int(valid.sum()),
                    "observed_mean": float(np.mean(target)),
                    "raw_mae": raw_scores["mae"],
                    f"{model_label}_mae": corrected_scores["mae"],
                    "raw_bias": raw_scores["bias"],
                    f"{model_label}_bias": corrected_scores["bias"],
                }
            )
    return pd.DataFrame(rows)


def _group_scores(frame: pd.DataFrame, group_column: str, model_label: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for value, group in frame.groupby(group_column, observed=True, sort=True):
        for model_name, column in [("raw_gfs", "raw_gfs"), (model_label, model_label)]:
            scores = continuous_scores_frame(group[column].to_numpy(), group["target"].to_numpy())
            rows.append(
                {
                    group_column: value,
                    "model": model_name,
                    **scores,
                    "samples": int(len(group)),
                    "target_mean": float(group["target"].mean()),
                    "prediction_mean": float(group[column].mean()),
                }
            )
    return pd.DataFrame(rows)


def _report_text(
    *,
    run_name: str,
    split: str,
    predictions_path: Path,
    model_label: str,
    by_lead: pd.DataFrame,
    by_intensity: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    model_lead = by_lead[by_lead["model"] == model_label].sort_values("mae", ascending=False).head(5)
    intensity = by_intensity[by_intensity["model"] == model_label]
    lines = [
        f"# {run_name} Error Diagnostics\n\n",
        "This diagnostic report groups rainfall correction errors by forecast lead, cycle, month, observed intensity, and grid cell.\n\n",
        f"- Split: {split}\n",
        f"- Predictions: `{predictions_path}`\n\n",
        "## Worst Lead Hours By MAE\n\n",
        "| Lead hour | MAE | RMSE | Bias | Pearson r | Samples | Target mean |\n",
        "|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, row in model_lead.iterrows():
        lines.append(
            f"| {row['lead_hour']:g} | {row['mae']:.3f} | {row['rmse']:.3f} | {row['bias']:.3f} | "
            f"{row['pearson_r']:.3f} | {int(row['samples'])} | {row['target_mean']:.3f} |\n"
        )
    lines.extend(
        [
            "\n## Observed Intensity Bins\n\n",
            "| Intensity bin mm | MAE | RMSE | Bias | Pearson r | Samples | Target mean | Prediction mean |\n",
            "|---|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for _, row in intensity.iterrows():
        lines.append(
            f"| {row['intensity_bin']} | {row['mae']:.3f} | {row['rmse']:.3f} | {row['bias']:.3f} | "
            f"{row['pearson_r']:.3f} | {int(row['samples'])} | {row['target_mean']:.3f} | "
            f"{row['prediction_mean']:.3f} |\n"
        )
    lines.extend(
        [
            "\n## Output Files\n\n",
        ]
    )
    for name, path in paths.items():
        lines.append(f"- {name}: `{path}`\n")
    return "".join(lines)


if __name__ == "__main__":
    main()
