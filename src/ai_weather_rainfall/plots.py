from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_lead_metric(frame: pd.DataFrame, metric: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    continuous = frame[frame["metric_type"] == "continuous"]
    for model, group in continuous.groupby("model"):
        summary = group.groupby("lead_hour", as_index=False)[metric].mean()
        ax.plot(summary["lead_hour"], summary[metric], marker="o", label=model.upper())
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_categorical_metric(frame: pd.DataFrame, metric: str, threshold_mm: float, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    categorical = frame[
        (frame["metric_type"] == "categorical")
        & (frame["threshold_mm"].astype(float) == float(threshold_mm))
    ]
    for model, group in categorical.groupby("model"):
        summary = group.groupby("lead_hour", as_index=False)[metric].mean()
        ax.plot(summary["lead_hour"], summary[metric], marker="o", label=model.upper())
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{threshold_mm:g} mm / 24h")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
