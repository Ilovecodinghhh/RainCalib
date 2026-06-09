from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import torch
from torch.utils.data import DataLoader

from ai_weather_rainfall.deep.dataset import RainCalibTensorDataset, load_unet_config
from ai_weather_rainfall.deep.unet import SmallUNet
from ai_weather_rainfall.raincalib import categorical_scores_frame, choose_probability_threshold


@dataclass
class EventSplit:
    target: np.ndarray
    probabilities: list[np.ndarray]


def main() -> None:
    args = _parse_args()
    cfg = load_unet_config(args.config)
    checkpoint_path = args.checkpoint or (cfg.models_dir / f"{cfg.run_name}.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    event_thresholds = tuple(float(v) for v in checkpoint.get("event_thresholds_mm", cfg.event_thresholds_mm))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint, event_thresholds, device)

    tune = _predict_event_split(cfg, model, device, args.tune_split, event_thresholds)
    test = _predict_event_split(cfg, model, device, args.split, event_thresholds)

    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.figures_dir or (cfg.metrics_dir.parent / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    figure_paths: list[Path] = []

    for index, threshold in enumerate(event_thresholds):
        tune_observed = tune.target >= threshold
        test_observed = test.target >= threshold
        tune_prob = tune.probabilities[index]
        test_prob = test.probabilities[index]
        calibrator = _fit_isotonic(tune_prob, tune_observed)
        tune_calibrated = calibrator.transform(tune_prob)
        test_calibrated = calibrator.transform(test_prob)

        raw_threshold, raw_tune_scores = choose_probability_threshold(
            tune_prob,
            tune_observed,
            metric=cfg.event_probability_metric,
        )
        calibrated_threshold, calibrated_tune_scores = choose_probability_threshold(
            tune_calibrated,
            tune_observed,
            metric=cfg.event_probability_metric,
        )
        far_threshold, far_tune_scores = _choose_threshold_with_far_cap(
            tune_calibrated,
            tune_observed,
            metric=cfg.event_probability_metric,
            far_cap=args.far_cap,
        )
        policies = [
            ("raw_probability", test_prob, raw_threshold, raw_tune_scores),
            ("isotonic_calibrated", test_calibrated, calibrated_threshold, calibrated_tune_scores),
            (f"isotonic_far_cap_{args.far_cap:.2f}", test_calibrated, far_threshold, far_tune_scores),
        ]
        for policy, probabilities, probability_threshold, tune_scores in policies:
            scores = categorical_scores_frame(probabilities >= probability_threshold, test_observed)
            metric_rows.append(
                {
                    "threshold_mm": threshold,
                    "policy": policy,
                    "probability_threshold": probability_threshold,
                    "test_brier": _brier(probabilities, test_observed),
                    "tune_csi": tune_scores["csi"],
                    "tune_pod": tune_scores["pod"],
                    "tune_far": tune_scores["far"],
                    "tune_ets": tune_scores["ets"],
                    **scores,
                }
            )
        threshold_rows.extend(
            [
                {
                    "threshold_mm": threshold,
                    "policy": "raw_probability",
                    "probability_threshold": raw_threshold,
                    **raw_tune_scores,
                },
                {
                    "threshold_mm": threshold,
                    "policy": "isotonic_calibrated",
                    "probability_threshold": calibrated_threshold,
                    **calibrated_tune_scores,
                },
                {
                    "threshold_mm": threshold,
                    "policy": f"isotonic_far_cap_{args.far_cap:.2f}",
                    "probability_threshold": far_threshold,
                    **far_tune_scores,
                },
            ]
        )
        reliability_rows.extend(_reliability_rows(threshold, "raw_probability", test_prob, test_observed, args.bins))
        reliability_rows.extend(
            _reliability_rows(threshold, "isotonic_calibrated", test_calibrated, test_observed, args.bins)
        )
        figure_paths.append(
            _plot_reliability(
                figures_dir / f"{cfg.run_name}_{args.split}_reliability_{threshold:g}mm.png",
                threshold,
                test_prob,
                test_calibrated,
                test_observed,
                args.bins,
            )
        )

    metrics = pd.DataFrame(metric_rows)
    thresholds = pd.DataFrame(threshold_rows)
    reliability = pd.DataFrame(reliability_rows)
    metrics_path = cfg.metrics_dir / f"{cfg.run_name}_{args.split}_event_calibration_metrics.csv"
    thresholds_path = cfg.metrics_dir / f"{cfg.run_name}_{args.split}_event_calibrated_thresholds.csv"
    reliability_path = cfg.metrics_dir / f"{cfg.run_name}_{args.split}_event_reliability.csv"
    report_path = cfg.reports_dir / f"{cfg.run_name}_{args.split}_event_calibration_report.md"
    metrics.to_csv(metrics_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)
    reliability.to_csv(reliability_path, index=False)
    report_path.write_text(
        _report_text(
            run_name=cfg.run_name,
            split=args.split,
            tune_split=args.tune_split,
            checkpoint_path=checkpoint_path,
            metrics=metrics,
            metrics_path=metrics_path,
            thresholds_path=thresholds_path,
            reliability_path=reliability_path,
            figure_paths=figure_paths,
        ),
        encoding="utf-8",
    )
    print(f"metrics={metrics_path}")
    print(f"thresholds={thresholds_path}")
    print(f"reliability={reliability_path}")
    for path in figure_paths:
        print(f"reliability_figure={path}")
    print(f"report={report_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate RainCalib event-head probabilities with isotonic regression.")
    parser.add_argument("--config", type=Path, default=Path("configs/raincalib_unet.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tune-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--far-cap", type=float, default=0.90)
    return parser.parse_args()


def _load_model(checkpoint: dict, event_thresholds: tuple[float, ...], device: torch.device) -> SmallUNet:
    model = SmallUNet(
        in_channels=int(checkpoint["in_channels"]),
        out_channels=int(checkpoint.get("out_channels", 1 + len(event_thresholds))),
        base_channels=int(checkpoint["base_channels"]),
        depth=int(checkpoint["depth"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def _predict_event_split(cfg, model, device: torch.device, split: str, event_thresholds: tuple[float, ...]) -> EventSplit:
    dataset = RainCalibTensorDataset(
        cfg.tensor_store,
        split,
        rain_scale_mm=cfg.rain_scale_mm,
        gfs_clip_mm=cfg.gfs_clip_mm,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    targets: list[np.ndarray] = []
    probabilities: list[list[np.ndarray]] = [[] for _ in event_thresholds]
    for batch in loader:
        output = model(batch["x"].to(device)).detach().cpu()
        batch_prob = torch.sigmoid(output[:, 1:]).numpy()
        target = batch["target"].numpy()
        mask = batch["mask"].numpy().astype(bool)
        targets.append(target[mask])
        for index in range(len(event_thresholds)):
            probabilities[index].append(batch_prob[:, index : index + 1][mask])
    return EventSplit(
        target=np.concatenate(targets),
        probabilities=[np.concatenate(values) for values in probabilities],
    )


def _fit_isotonic(probabilities: np.ndarray, observed: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    if len(np.unique(observed)) < 2:
        calibrator.fit([0.0, 1.0], [float(observed[0]), float(observed[0])])
        return calibrator
    calibrator.fit(probabilities, observed.astype(float))
    return calibrator


def _choose_threshold_with_far_cap(
    probabilities: np.ndarray,
    observed: np.ndarray,
    *,
    metric: str,
    far_cap: float,
) -> tuple[float, dict[str, float]]:
    best_any_threshold, best_any_scores = choose_probability_threshold(probabilities, observed, metric=metric)
    best_threshold = best_any_threshold
    best_scores = best_any_scores
    best_value = float("-inf")
    for threshold in np.linspace(0.01, 0.99, 99):
        scores = categorical_scores_frame(probabilities >= threshold, observed)
        value = scores.get(metric, np.nan)
        if np.isfinite(scores["far"]) and scores["far"] <= far_cap and np.isfinite(value) and value > best_value:
            best_threshold = float(threshold)
            best_scores = scores
            best_value = float(value)
    return best_threshold, best_scores


def _reliability_rows(
    threshold_mm: float,
    policy: str,
    probabilities: np.ndarray,
    observed: np.ndarray,
    bins: int,
) -> list[dict[str, float | int | str]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float | int | str]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            selected = (probabilities >= left) & (probabilities <= right)
        else:
            selected = (probabilities >= left) & (probabilities < right)
        count = int(selected.sum())
        rows.append(
            {
                "threshold_mm": threshold_mm,
                "policy": policy,
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "mean_probability": float(np.mean(probabilities[selected])) if count else np.nan,
                "observed_frequency": float(np.mean(observed[selected])) if count else np.nan,
            }
        )
    return rows


def _plot_reliability(
    path: Path,
    threshold_mm: float,
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    observed: np.ndarray,
    bins: int,
) -> Path:
    fig, axis = plt.subplots(figsize=(5.5, 5), dpi=160)
    for label, probability in [("raw", raw_probability), ("isotonic", calibrated_probability)]:
        rows = pd.DataFrame(_reliability_rows(threshold_mm, label, probability, observed, bins))
        rows = rows[rows["count"] > 0]
        axis.plot(rows["mean_probability"], rows["observed_frequency"], marker="o", label=label)
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="perfect")
    axis.set_title(f"Reliability: {threshold_mm:g} mm event")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed frequency")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _brier(probabilities: np.ndarray, observed: np.ndarray) -> float:
    return float(np.mean((probabilities - observed.astype(float)) ** 2))


def _report_text(
    *,
    run_name: str,
    split: str,
    tune_split: str,
    checkpoint_path: Path,
    metrics: pd.DataFrame,
    metrics_path: Path,
    thresholds_path: Path,
    reliability_path: Path,
    figure_paths: list[Path],
) -> str:
    lines = [
        f"# {run_name} Event Probability Calibration\n\n",
        "This report calibrates the U-Net event-head probabilities using isotonic regression on the tune split.\n\n",
        "**Important:** This is a research benchmark, not an official warning service.\n\n",
        f"- Evaluation split: {split}\n",
        f"- Tune split: {tune_split}\n",
        f"- Checkpoint: `{checkpoint_path}`\n",
        f"- Metrics: `{metrics_path}`\n",
        f"- Thresholds: `{thresholds_path}`\n",
        f"- Reliability bins: `{reliability_path}`\n\n",
        "## Test Event Metrics\n\n",
        "| Event | Policy | Prob threshold | Brier | Hits | Misses | False alarms | CSI | POD | FAR | ETS |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, row in metrics.sort_values(["threshold_mm", "policy"]).iterrows():
        lines.append(
            f"| {row['threshold_mm']:g} mm | {row['policy']} | {row['probability_threshold']:.2f} | "
            f"{row['test_brier']:.4f} | {int(row['hits'])} | {int(row['misses'])} | "
            f"{int(row['false_alarms'])} | {row['csi']:.3f} | {row['pod']:.3f} | "
            f"{row['far']:.3f} | {row['ets']:.3f} |\n"
        )
    lines.append("\n## Reliability Figures\n\n")
    for path in figure_paths:
        lines.append(f"- `{path}`\n")
    return "".join(lines)


if __name__ == "__main__":
    main()
