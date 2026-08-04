from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_weather_rainfall.experimental import (
    fractions_skill_score,
    fractions_skill_score_events,
    load_experiment_config,
)


def main() -> None:
    args = _parse_args()
    cfg = load_experiment_config(args.config)
    tensor = np.load(cfg.tensor_store, allow_pickle=False)
    rows: list[dict[str, object]] = []
    output_root = cfg.output_root / args.output_subdir
    for fold in cfg.folds:
        for seed in cfg.seeds:
            run_dir = output_root / fold.name / f"seed_{seed}"
            prediction_path = run_dir / "test_predictions.npz"
            threshold_path = run_dir / "event_thresholds.json"
            if not prediction_path.exists() or not threshold_path.exists():
                raise FileNotFoundError(f"Missing completed run outputs in {run_dir}")
            prediction = np.load(prediction_path, allow_pickle=False)
            indexes = prediction["indexes"].astype(int)
            target = tensor["target"][indexes]
            mask = tensor["mask"][indexes] > 0
            thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
            run_rows: list[dict[str, object]] = []
            for channel, threshold in enumerate(cfg.event_thresholds_mm):
                raw_cutoff = thresholds[f"{threshold:g}"]["raw_probability_threshold"]
                calibrated_cutoff = thresholds[f"{threshold:g}"]["calibrated_probability_threshold"]
                sources = {
                    "rain_prediction": (
                        prediction["prediction_rain"],
                        target,
                        threshold,
                    ),
                    "event_head_decision": (
                        prediction["event_probability"][:, channel] >= raw_cutoff,
                        target >= threshold,
                        0.5,
                    ),
                    "calibrated_decision": (
                        prediction["event_probability_calibrated"][:, channel] >= calibrated_cutoff,
                        target >= threshold,
                        0.5,
                    ),
                }
                for source, (forecast, observed, score_threshold) in sources.items():
                    for neighborhood in cfg.fss_neighborhoods:
                        if source == "rain_prediction":
                            score = fractions_skill_score(
                                forecast,
                                observed,
                                mask,
                                threshold_mm=score_threshold,
                                neighborhood=neighborhood,
                            )
                        else:
                            score = fractions_skill_score_events(
                                forecast,
                                observed,
                                mask,
                                neighborhood=neighborhood,
                            )
                        run_rows.append(
                            {
                                "fold": fold.name,
                                "seed": seed,
                                "source": source,
                                "threshold_mm": threshold,
                                "neighborhood": neighborhood,
                                "fss": score,
                            }
                        )
            pd.DataFrame(run_rows).to_csv(run_dir / "fss.csv", index=False)
            rows.extend(run_rows)
            print(f"fold={fold.name} seed={seed} rows={len(run_rows)}")
    output = pd.DataFrame(rows)
    output.to_csv(output_root / "fss.csv", index=False)
    summary = (
        output.groupby(["fold", "source", "threshold_mm", "neighborhood"])["fss"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.to_csv(output_root / "fss_seed_summary.csv", index=False)
    print(f"fss={output_root / 'fss.csv'}")
    print(f"summary={output_root / 'fss_seed_summary.csv'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute U-Net FSS from saved field predictions.")
    parser.add_argument("--config", type=Path, default=Path("configs/raincalib_experiments.yaml"))
    parser.add_argument("--output-subdir", default="unet")
    return parser.parse_args()


if __name__ == "__main__":
    main()
