from pathlib import Path

import pandas as pd


def write_report(metrics: pd.DataFrame, figures: list[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_lines = "\n".join(f"![{path.stem}](../figures/{path.name})" for path in figures)
    text = (
        "# Rainfall Benchmark Report\n\n"
        "This report evaluates ECMWF AIFS and ECMWF IFS rainfall forecasts against NASA GPM IMERG observations.\n\n"
        "**Important:** This is a research and decision-support benchmark, not an official weather warning service.\n\n"
        f"Metric rows: {len(metrics)}\n\n"
        "## Figures\n\n"
        f"{figure_lines}\n"
    )
    output_path.write_text(text, encoding="utf-8")
    return output_path
