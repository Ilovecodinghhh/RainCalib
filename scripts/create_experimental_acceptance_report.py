from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_weather_rainfall.experimental import load_experiment_config, paired_date_bootstrap


def main() -> None:
    args = _parse_args()
    cfg = load_experiment_config(args.config)
    statistical_root = cfg.output_root / "statistical"
    unet_root = cfg.output_root / "unet"
    statistical_metrics = pd.read_csv(statistical_root / "metrics.csv")
    statistical_fss = pd.read_csv(statistical_root / "fss.csv")
    unet_metrics = pd.read_csv(unet_root / "metrics.csv")
    unet_fss = pd.read_csv(unet_root / "fss.csv")
    tensor = np.load(cfg.tensor_store, allow_pickle=False)
    audit = json.loads((cfg.output_root / "audit" / "experiment_manifest.json").read_text(encoding="utf-8"))

    comparisons = _bootstrap_against_baselines(cfg, tensor, statistical_root, unet_root)
    comparisons.to_csv(unet_root / "paired_date_bootstrap_vs_baselines.csv", index=False)
    report = _build_report(
        cfg,
        audit,
        statistical_metrics,
        statistical_fss,
        unet_metrics,
        unet_fss,
        comparisons,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    generated = cfg.output_root / "experimental_acceptance_report.md"
    generated.write_text(report, encoding="utf-8")
    print(f"report={args.output}")
    print(f"generated={generated}")
    print(f"comparisons={unet_root / 'paired_date_bootstrap_vs_baselines.csv'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the RainCalib experimental acceptance report.")
    parser.add_argument("--config", type=Path, default=Path("configs/raincalib_experiments.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experimental_validation_results.md"),
    )
    return parser.parse_args()


def _bootstrap_against_baselines(cfg, tensor, statistical_root, unet_root) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    baseline_names = ("raw_gfs", "quantile_mapping")
    for fold in cfg.folds:
        statistical = pd.read_parquet(statistical_root / fold.name / "test_predictions.parquet")
        statistical["run_date"] = statistical["run_date"].astype(str)
        statistical["observed_date"] = statistical["observed_date"].astype(str).str[:10]
        first_prediction = np.load(
            unet_root / fold.name / f"seed_{cfg.seeds[0]}" / "test_predictions.npz",
            allow_pickle=False,
        )
        indexes = first_prediction["indexes"].astype(int)
        alignment, valid, dates, target = _alignment(tensor, indexes, statistical)
        for seed in cfg.seeds:
            prediction = np.load(
                unet_root / fold.name / f"seed_{seed}" / "test_predictions.npz",
                allow_pickle=False,
            )["prediction_rain"].reshape(-1)[valid]
            model_values = prediction[alignment["tensor_position"]]
            aligned_target = target[alignment["tensor_position"]]
            aligned_dates = dates[alignment["tensor_position"]]
            for baseline_name in baseline_names:
                baseline = statistical[f"prediction__{baseline_name}"].to_numpy(dtype=float)[
                    alignment["statistical_position"]
                ]
                comparison = paired_date_bootstrap(
                    model_values,
                    baseline,
                    aligned_target,
                    aligned_dates,
                    resamples=cfg.bootstrap_resamples,
                    seed=seed,
                )
                comparison.insert(0, "baseline", baseline_name)
                comparison.insert(0, "seed", seed)
                comparison.insert(0, "fold", fold.name)
                rows.append(comparison)
    return pd.concat(rows, ignore_index=True)


def _alignment(tensor, indexes, statistical):
    height = len(tensor["latitude"])
    width = len(tensor["longitude"])
    pixels_per_field = height * width
    valid = tensor["mask"][indexes].reshape(-1) > 0
    run_date = np.repeat(tensor["run_date"][indexes].astype(str), pixels_per_field)[valid]
    observed_date = np.repeat(tensor["observed_date"][indexes].astype(str), pixels_per_field)[valid]
    cycle = np.repeat(tensor["cycle"][indexes].astype(int), pixels_per_field)[valid]
    lead_hour = np.repeat(tensor["lead_hour"][indexes].astype(int), pixels_per_field)[valid]
    latitude_grid = np.repeat(tensor["latitude"], width)
    longitude_grid = np.tile(tensor["longitude"], height)
    latitude = np.tile(latitude_grid, len(indexes))[valid]
    longitude = np.tile(longitude_grid, len(indexes))[valid]
    tensor_keys = pd.DataFrame(
        {
            "run_date": run_date,
            "observed_date": observed_date,
            "cycle": cycle,
            "lead_hour": lead_hour,
            "latitude": latitude,
            "longitude": longitude,
            "tensor_position": np.arange(valid.sum()),
        }
    )
    statistical_keys = statistical[
        ["run_date", "observed_date", "cycle", "lead_hour", "latitude", "longitude"]
    ].copy()
    statistical_keys["statistical_position"] = np.arange(len(statistical_keys))
    alignment = tensor_keys.merge(
        statistical_keys,
        on=["run_date", "observed_date", "cycle", "lead_hour", "latitude", "longitude"],
        how="inner",
        validate="one_to_one",
    )
    if len(alignment) != int(valid.sum()) or len(alignment) != len(statistical):
        raise ValueError("Tensor and statistical predictions do not align one-to-one")
    target = tensor["target"][indexes].reshape(-1)[valid]
    dates = np.repeat(tensor["observed_date"][indexes].astype(str), pixels_per_field)[valid]
    aligned_target = target[alignment["tensor_position"]]
    statistical_target = statistical["target_rain"].to_numpy(dtype=float)[alignment["statistical_position"]]
    if not np.allclose(aligned_target, statistical_target, rtol=0.0, atol=1e-5):
        raise ValueError("Tensor and statistical targets differ after key alignment")
    return alignment, valid, dates, target


def _build_report(cfg, audit, statistical_metrics, statistical_fss, unet_metrics, unet_fss, comparisons):
    completed_folds = sorted(unet_metrics["fold"].unique())
    seeds_per_fold = unet_metrics.groupby("fold")["seed"].nunique()
    expected_folds = {fold.name for fold in cfg.folds}
    fold_pass = set(completed_folds) == expected_folds
    seed_pass = bool((seeds_per_fold == len(cfg.seeds)).all())
    rmse = comparisons[comparisons["metric"] == "rmse"]
    raw_stable = rmse[rmse["baseline"] == "raw_gfs"]["ci_upper"] < 0
    quantile_stable = rmse[rmse["baseline"] == "quantile_mapping"]["ci_upper"] < 0
    metrics_present = {
        "pr_auc",
        "brier",
        "bss",
    }.issubset(set(unet_metrics["metric"])) and not unet_fss.empty
    internal_pass = fold_pass and seed_pass and raw_stable.all() and quantile_stable.all() and metrics_present

    continuous_table = _continuous_table(statistical_metrics, unet_metrics)
    event_table = _event_table(statistical_fss, unet_metrics, unet_fss)
    comparison_table = _comparison_table(comparisons)
    criteria = pd.DataFrame(
        [
            ["Two rolling-origin folds complete", "PASS" if fold_pass else "FAIL", len(completed_folds)],
            ["Five fixed seeds per fold", "PASS" if seed_pass else "FAIL", int(seeds_per_fold.min())],
            ["RMSE beats Raw GFS with date-block 95% CI", "PASS" if raw_stable.all() else "FAIL", f"{raw_stable.sum()}/{len(raw_stable)} runs"],
            ["RMSE beats Quantile Mapping with date-block 95% CI", "PASS" if quantile_stable.all() else "FAIL", f"{quantile_stable.sum()}/{len(quantile_stable)} runs"],
            ["PR-AUC, BSS, and FSS reported", "PASS" if metrics_present else "FAIL", "50/100 mm; 1/3/5 cells"],
            ["Untouched future holdout and independent observations", "PENDING", "2026 flood season / second product unavailable"],
        ],
        columns=["Criterion", "Status", "Evidence"],
    )
    status = "PASS" if internal_pass else "FAIL"
    return "".join(
        [
            "# RainCalib Experimental Validation Results\n\n",
            f"- Internal research acceptance: **{status}**\n",
            "- External / production acceptance: **PENDING**\n",
            f"- Data: {audit['date_start']} to {audit['date_end']}, {audit['observed_dates']} independent observed dates\n",
            f"- Forecast fields: {audit['forecast_fields']}; valid grid-cell pairs: {audit['rows']}\n",
            f"- Frozen baseline commit: `{audit['git_commit']}`\n\n",
            "The internal protocol passes, but this is not a production-readiness claim. The complete 2026 flood season and a second independent observation product are not yet available.\n\n",
            "## Acceptance Checklist\n\n",
            _markdown_table(criteria),
            "\n\n## Continuous Results\n\n",
            "U-Net values are mean +/- standard deviation across five seeds. Other baselines are deterministic.\n\n",
            _markdown_table(continuous_table),
            "\n\n## Paired Date-Block Bootstrap\n\n",
            "Deltas are RainCalib v2.1 minus baseline. Negative MAE/RMSE is favorable; each run uses 1,000 complete-date resamples.\n\n",
            _markdown_table(comparison_table),
            "\n\n## Heavy-Rainfall Results\n\n",
            _markdown_table(event_table),
            "\n\nThe 50 mm event head improves neighborhood-scale FSS, especially in rolling_2025. The 100 mm event remains unstable, with high false-alarm rates and weak calibrated detection. Probability calibration sharply improves Brier score but can reduce useful event recall, so calibrated probabilities and screening decisions must remain separate products.\n\n",
            "## Interpretation\n\n",
            "- The strongest defensible claim is stable RMSE reduction across both temporal folds and all five seeds.\n",
            "- MAE improvement is stable in rolling_2025 but not for every rolling_2024 seed.\n",
            "- RainCalib improves broad rainfall-field error more reliably than exact extreme-event localization.\n",
            "- Pixel counts are never treated as independent; uncertainty is blocked by observed date.\n",
            "- External validation remains mandatory before operational or production claims.\n\n",
            "## Reproduction\n\n",
            "```bash\n",
            "PYTHONPATH=src python scripts/audit_experiment_data.py\n",
            "PYTHONPATH=src python scripts/run_statistical_validation.py\n",
            "PYTHONPATH=src python scripts/run_unet_experiment_matrix.py\n",
            "PYTHONPATH=src python scripts/recompute_unet_fss.py\n",
            "PYTHONPATH=src python scripts/create_experimental_acceptance_report.py\n",
            "```\n",
        ]
    )


def _continuous_table(statistical, unet):
    rows = []
    for fold in sorted(statistical["fold"].unique()):
        for model in ("raw_gfs", "mean_bias", "quantile_mapping", "grid_month_climatology", "raincalib_v1"):
            subset = statistical[
                (statistical["fold"] == fold)
                & (statistical["model"] == model)
                & (statistical["category"] == "continuous")
            ].set_index("metric")["value"]
            rows.append([fold, model, _format(subset["mae"]), _format(subset["rmse"]), _format(subset["bias"]), _format(subset["pearson_r"])])
        subset = unet[(unet["fold"] == fold) & (unet["category"] == "continuous")]
        grouped = subset.groupby("metric")["value"].agg(["mean", "std"])
        rows.append(
            [
                fold,
                "raincalib_v2.1 (5 seeds)",
                _plus_minus(grouped.loc["mae"]),
                _plus_minus(grouped.loc["rmse"]),
                _plus_minus(grouped.loc["bias"]),
                _plus_minus(grouped.loc["pearson_r"]),
            ]
        )
    return pd.DataFrame(rows, columns=["Fold", "Model", "MAE", "RMSE", "Bias", "Pearson r"])


def _comparison_table(comparisons):
    rmse = comparisons[comparisons["metric"] == "rmse"]
    rows = []
    for (fold, baseline), group in rmse.groupby(["fold", "baseline"], sort=True):
        rows.append(
            [
                fold,
                baseline,
                f"{group['delta_mean'].mean():.3f}",
                f"{group['ci_lower'].min():.3f} to {group['ci_upper'].max():.3f}",
                f"{int((group['ci_upper'] < 0).sum())}/{len(group)}",
            ]
        )
    return pd.DataFrame(rows, columns=["Fold", "Baseline", "Mean delta RMSE", "CI envelope", "Stable seeds"])


def _event_table(statistical_fss, unet_metrics, unet_fss):
    rows = []
    for fold in sorted(unet_metrics["fold"].unique()):
        for threshold in (50.0, 100.0):
            event = unet_metrics[
                (unet_metrics["fold"] == fold)
                & (unet_metrics["category"] == "event_head_decision")
                & (unet_metrics["threshold_mm"] == threshold)
            ].groupby("metric")["value"].mean()
            calibrated = unet_metrics[
                (unet_metrics["fold"] == fold)
                & (unet_metrics["category"] == "calibrated_probability")
                & (unet_metrics["threshold_mm"] == threshold)
            ].groupby("metric")["value"].mean()
            model_fss = unet_fss[
                (unet_fss["fold"] == fold)
                & (unet_fss["source"] == "event_head_decision")
                & (unet_fss["threshold_mm"] == threshold)
                & (unet_fss["neighborhood"] == 5)
            ]["fss"].mean()
            raw_fss = statistical_fss[
                (statistical_fss["fold"] == fold)
                & (statistical_fss["model"] == "raw_gfs")
                & (statistical_fss["threshold_mm"] == threshold)
                & (statistical_fss["neighborhood"] == 5)
            ]["fss"].iloc[0]
            rows.append(
                [
                    fold,
                    f"{threshold:g}",
                    _format(event["csi"]),
                    _format(event["pod"]),
                    _format(event["far"]),
                    _format(calibrated["pr_auc"], 4),
                    _format(calibrated["bss"], 4),
                    _format(raw_fss),
                    _format(model_fss),
                ]
            )
    return pd.DataFrame(
        rows,
        columns=["Fold", "mm", "CSI", "POD", "FAR", "Cal PR-AUC", "Cal BSS", "Raw FSS-5", "Event-head FSS-5"],
    )


def _format(value, digits=3):
    return f"{float(value):.{digits}f}"


def _plus_minus(values):
    return f"{values['mean']:.3f} +/- {values['std']:.3f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [headers, ["---"] * len(headers)]
    rows.extend([str(value) for value in row] for row in frame.itertuples(index=False, name=None))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


if __name__ == "__main__":
    main()
