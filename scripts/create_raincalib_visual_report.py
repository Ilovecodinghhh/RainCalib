from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_weather_rainfall.deep.dataset import load_unet_config


def main() -> None:
    args = _parse_args()
    cfg = load_unet_config(args.config)
    run_name = args.run_name or cfg.run_name
    figures_dir = args.figures_dir or (cfg.metrics_dir.parent / "figures")
    report_path = args.report or (cfg.reports_dir / f"{run_name}_{args.split}_visual_case_study.md")
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_path = cfg.predictions_dir / f"{run_name}_{args.split}_predictions.npz"
    predictions = np.load(predictions_path, allow_pickle=False)
    tensor = np.load(cfg.tensor_store, allow_pickle=False)
    metadata = _metadata_frame(tensor, args.split)
    thresholds = _load_probability_thresholds(cfg.metrics_dir / f"{run_name}_{args.split}_event_probability_thresholds.csv")

    lead_path = _plot_lead_skill(cfg.metrics_dir, run_name, args.split, figures_dir)
    intensity_path = _plot_intensity_diagnostics(cfg.metrics_dir, run_name, args.split, figures_dir)
    event_path = _plot_event_skill(cfg.metrics_dir, run_name, args.split, figures_dir)
    case_table, case_paths = _plot_cases(predictions, tensor, metadata, thresholds, run_name, args.case_count, figures_dir)

    case_table_path = cfg.metrics_dir / f"{run_name}_{args.split}_case_study.csv"
    case_table.to_csv(case_table_path, index=False)
    report_path.write_text(
        _report_text(
            run_name=run_name,
            split=args.split,
            predictions_path=predictions_path,
            lead_path=lead_path,
            intensity_path=intensity_path,
            event_path=event_path,
            case_table_path=case_table_path,
            case_table=case_table,
            case_paths=case_paths,
        ),
        encoding="utf-8",
    )
    print(f"lead_figure={lead_path}")
    print(f"intensity_figure={intensity_path}")
    print(f"event_figure={event_path}")
    print(f"case_table={case_table_path}")
    for path in case_paths:
        print(f"case_figure={path}")
    print(f"report={report_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create RainCalib visual diagnostics and case-study figures.")
    parser.add_argument("--config", type=Path, default=Path("configs/raincalib_unet.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-name")
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--case-count", type=int, default=3)
    return parser.parse_args()


def _metadata_frame(tensor, split: str) -> pd.DataFrame:
    split_values = tensor["split"].astype(str)
    indexes = np.flatnonzero(split_values == split)
    return pd.DataFrame(
        {
            "sample_index": np.arange(len(indexes)),
            "tensor_index": indexes,
            "run_date": tensor["run_date"].astype(str)[indexes],
            "observed_date": tensor["observed_date"].astype(str)[indexes],
            "cycle": tensor["cycle"].astype("float32")[indexes],
            "lead_hour": tensor["lead_hour"].astype("float32")[indexes],
        }
    )


def _load_probability_thresholds(path: Path) -> dict[float, float]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {float(row["threshold_mm"]): float(row["probability_threshold"]) for _, row in frame.iterrows()}


def _plot_lead_skill(metrics_dir: Path, run_name: str, split: str, figures_dir: Path) -> Path:
    path = metrics_dir / f"{run_name}_{split}_diagnostics_by_lead.csv"
    frame = pd.read_csv(path)
    output = figures_dir / f"{run_name}_{split}_lead_skill.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    for model, group in frame.groupby("model"):
        group = group.sort_values("lead_hour")
        axes[0].plot(group["lead_hour"], group["mae"], marker="o", label=model)
        axes[1].plot(group["lead_hour"], group["bias"], marker="o", label=model)
    axes[0].set_title("MAE by lead hour")
    axes[1].set_title("Bias by lead hour")
    for axis in axes:
        axis.set_xlabel("Lead hour")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("mm / 24h")
    axes[1].set_ylabel("mm / 24h")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_intensity_diagnostics(metrics_dir: Path, run_name: str, split: str, figures_dir: Path) -> Path:
    path = metrics_dir / f"{run_name}_{split}_diagnostics_by_intensity.csv"
    frame = pd.read_csv(path)
    output = figures_dir / f"{run_name}_{split}_intensity_diagnostics.png"
    model_frame = frame[frame["model"] == run_name].copy()
    raw_frame = frame[frame["model"] == "raw_gfs"].copy()
    labels = model_frame["intensity_bin"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.28
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)
    axes[0].bar(x - width, raw_frame["prediction_mean"], width, label="Raw GFS")
    axes[0].bar(x, model_frame["prediction_mean"], width, label=run_name)
    axes[0].bar(x + width, model_frame["target_mean"], width, label="IMERG")
    axes[0].set_title("Mean rainfall by observed intensity")
    axes[0].set_ylabel("mm / 24h")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, raw_frame["mae"], width, label="Raw GFS")
    axes[1].bar(x + width / 2, model_frame["mae"], width, label=run_name)
    axes[1].set_title("MAE by observed intensity")
    axes[1].set_ylabel("mm / 24h")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_event_skill(metrics_dir: Path, run_name: str, split: str, figures_dir: Path) -> Path:
    path = metrics_dir / f"{run_name}_{split}_categorical_metrics.csv"
    frame = pd.read_csv(path)
    output = figures_dir / f"{run_name}_{split}_event_skill.png"
    selected = frame[
        frame["model"].isin(["raw_gfs", "raincalib_v1_sklearn", f"{run_name}_event_head"])
    ].copy()
    selected["label"] = selected["model"].replace({f"{run_name}_event_head": "v2.1 event head"})
    metrics = ["csi", "pod", "far", "ets"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160, sharey=True)
    for axis, threshold in zip(axes, sorted(selected["threshold_mm"].unique())):
        group = selected[selected["threshold_mm"] == threshold]
        x = np.arange(len(metrics))
        width = 0.22
        for offset, (_, row) in enumerate(group.iterrows()):
            axis.bar(x + (offset - 1) * width, [row[m] for m in metrics], width, label=row["label"])
        axis.set_title(f"{threshold:g} mm event")
        axis.set_xticks(x)
        axis.set_xticklabels([m.upper() for m in metrics])
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("score")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_cases(
    predictions,
    tensor,
    metadata: pd.DataFrame,
    thresholds: dict[float, float],
    run_name: str,
    case_count: int,
    figures_dir: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    raw = predictions["raw_gfs"].astype("float32")
    corrected = predictions["corrected_rain"].astype("float32")
    target = predictions["target"].astype("float32")
    mask = predictions["mask"].astype(bool)
    probabilities = predictions["event_probabilities"].astype("float32") if "event_probabilities" in predictions else None
    latitudes = tensor["latitude"].astype("float32")
    longitudes = tensor["longitude"].astype("float32")
    target_max = np.where(mask, target, np.nan).max(axis=(1, 2))
    case_indexes = np.argsort(target_max)[::-1][:case_count]
    rows: list[dict[str, float | int | str]] = []
    paths: list[Path] = []
    for rank, sample_index in enumerate(case_indexes, start=1):
        row = metadata.iloc[int(sample_index)]
        prob50 = probabilities[sample_index, 0] if probabilities is not None and probabilities.shape[1] >= 1 else None
        prob100 = probabilities[sample_index, 1] if probabilities is not None and probabilities.shape[1] >= 2 else None
        path = figures_dir / f"{run_name}_case_{rank:02d}_{row['observed_date']}_lead{int(row['lead_hour'])}.png"
        _plot_case_panel(
            path,
            raw[sample_index],
            corrected[sample_index],
            target[sample_index],
            mask[sample_index],
            prob50,
            prob100,
            latitudes,
            longitudes,
            row,
            thresholds,
        )
        paths.append(path)
        rows.append(
            {
                "rank": rank,
                "observed_date": row["observed_date"],
                "run_date": row["run_date"],
                "cycle": int(row["cycle"]),
                "lead_hour": int(row["lead_hour"]),
                "target_max": float(np.nanmax(np.where(mask[sample_index], target[sample_index], np.nan))),
                "raw_max": float(np.nanmax(np.where(mask[sample_index], raw[sample_index], np.nan))),
                "corrected_max": float(np.nanmax(np.where(mask[sample_index], corrected[sample_index], np.nan))),
                "prob50_max": float(np.nanmax(prob50)) if prob50 is not None else np.nan,
                "prob100_max": float(np.nanmax(prob100)) if prob100 is not None else np.nan,
                "obs_pixels_50mm": int(np.sum((target[sample_index] >= 50.0) & mask[sample_index])),
                "obs_pixels_100mm": int(np.sum((target[sample_index] >= 100.0) & mask[sample_index])),
                "figure": str(path),
            }
        )
    return pd.DataFrame(rows), paths


def _plot_case_panel(
    path: Path,
    raw: np.ndarray,
    corrected: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    prob50: np.ndarray | None,
    prob100: np.ndarray | None,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    metadata_row: pd.Series,
    thresholds: dict[float, float],
) -> None:
    extent = [float(longitudes.min()), float(longitudes.max()), float(latitudes.min()), float(latitudes.max())]
    rain_vmax = float(max(50.0, np.nanpercentile(np.where(mask, target, np.nan), 99)))
    fields = [
        ("IMERG target", target, "rain", rain_vmax),
        ("Raw GFS", raw, "rain", rain_vmax),
        ("RainCalib corrected", corrected, "rain", rain_vmax),
        ("Corrected error", corrected - target, "error", rain_vmax),
    ]
    if prob50 is not None:
        fields.append((f"P >= 50 mm, thr={thresholds.get(50.0, 0.5):.2f}", prob50, "prob", 1.0))
    if prob100 is not None:
        fields.append((f"P >= 100 mm, thr={thresholds.get(100.0, 0.5):.2f}", prob100, "prob", 1.0))
    cols = 3
    rows = int(np.ceil(len(fields) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.8 * rows), dpi=160)
    axes_array = np.atleast_1d(axes).ravel()
    title = (
        f"Observed {metadata_row['observed_date']} | run {metadata_row['run_date']} "
        f"{int(metadata_row['cycle']):02d}Z | lead {int(metadata_row['lead_hour'])}h"
    )
    fig.suptitle(title, fontsize=12)
    for axis, (label, values, kind, vmax) in zip(axes_array, fields):
        masked_values = np.where(mask, values, np.nan)
        if kind == "error":
            image = axis.imshow(masked_values, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        elif kind == "prob":
            image = axis.imshow(masked_values, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=1.0)
        else:
            image = axis.imshow(masked_values, origin="lower", extent=extent, cmap="Blues", vmin=0.0, vmax=vmax)
        axis.set_title(label)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
        fig.colorbar(image, ax=axis, shrink=0.8)
    for axis in axes_array[len(fields) :]:
        axis.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _report_text(
    *,
    run_name: str,
    split: str,
    predictions_path: Path,
    lead_path: Path,
    intensity_path: Path,
    event_path: Path,
    case_table_path: Path,
    case_table: pd.DataFrame,
    case_paths: list[Path],
) -> str:
    lines = [
        f"# {run_name} Visual Case Study\n\n",
        "This report summarizes generated visual diagnostics for RainCalib heavy-rainfall correction.\n\n",
        f"- Split: {split}\n",
        f"- Predictions: `{predictions_path}`\n",
        f"- Lead skill figure: `{lead_path}`\n",
        f"- Intensity diagnostics figure: `{intensity_path}`\n",
        f"- Event skill figure: `{event_path}`\n",
        f"- Case table: `{case_table_path}`\n\n",
        "## Top Observed Rainfall Cases\n\n",
        "| Rank | Observed date | Run date | Cycle | Lead hour | Target max | Raw max | Corrected max | P50 max | P100 max |\n",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, row in case_table.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['observed_date']} | {row['run_date']} | {int(row['cycle'])} | "
            f"{int(row['lead_hour'])} | {row['target_max']:.1f} | {row['raw_max']:.1f} | "
            f"{row['corrected_max']:.1f} | {row['prob50_max']:.3f} | {row['prob100_max']:.3f} |\n"
        )
    lines.append("\n## Case Figures\n\n")
    for path in case_paths:
        lines.append(f"- `{path}`\n")
    return "".join(lines)


if __name__ == "__main__":
    main()
