# RainCalib Experimental Validation Protocol

## Temporal correction (2026-09-29)

The active protocol retains only 00 UTC initialization and 24-hour windows ending at leads divisible by 24. Legacy mixed-cycle results are historical because daily IMERG cannot label 12 UTC windows correctly. Fold boundaries, seeds, model settings, and bootstrap settings remain unchanged. Outputs are isolated under experiments_00utc. See [the revalidation record](00utc_revalidation.md).

## Purpose

This protocol evaluates whether RainCalib improvements are stable across time, random seeds, forecast leads, and heavy-rainfall events. It replaces pixel-level significance claims with date-blocked comparisons.

## Frozen Baseline

The frozen development baseline is RainCalib v2.1 at Git commit `423fc6b` with configuration `configs/raincalib_unet.yaml`.

The previously reported 2025-09-11 through 2025-09-30 test interval has been inspected during model development. It is therefore treated as a development test interval, not as a final untouched holdout.

## Evaluation Unit

- Every sample sharing the same `observed_date` belongs to the same split.
- All cycles, lead hours, and grid cells for one observed date are resampled together during bootstrap inference.
- Grid-cell counts are reported as valid verification pairs, not as independent samples.
- A future extension may group consecutive 2-3 day periods into weather-process blocks.

## Rolling-Origin Folds

| Fold | Train | Validation | Test |
|---|---|---|---|
| rolling_2024 | 2023-05-01 to 2023-09-30 | 2024-05-01 to 2024-07-31 | 2024-08-01 to 2024-09-30 |
| rolling_2025 | 2023-05-01 to 2024-09-30 | 2025-05-01 to 2025-07-31 | 2025-08-01 to 2025-09-30 |

The fold definitions live in `configs/raincalib_experiments.yaml`.

## Model Selection Rules

- Hyperparameters, probability thresholds, quantile mappings, and calibrators may use train and validation data only.
- Test data is used once per frozen experiment configuration.
- Results are aggregated across five fixed seeds: 42, 123, 2024, 3407, and 777.
- Failed runs remain in the experiment manifest with their failure reason.

## Required Baselines

- Raw GFS.
- Lead-month mean bias correction.
- Lead-wise empirical quantile mapping.
- Grid-month climatology.
- RainCalib v1 sklearn baseline where computationally practical.
- RainCalib v2.1 U-Net.

## Required Metrics

Continuous metrics:

- MAE, RMSE, Bias, Pearson correlation, and Spearman correlation.

Event and probabilistic metrics:

- CSI, POD, FAR, ETS, frequency bias, PR-AUC, Brier score, and Brier Skill Score against climatology.

Spatial metrics:

- Fraction Skill Score for 50mm and 100mm events at 1x1, 3x3, and 5x5 grid neighborhoods.

## Statistical Inference

- Use at least 1,000 paired bootstrap replicates.
- Resample complete observed dates with replacement.
- Report the model-minus-baseline metric difference, 95% percentile interval, and bootstrap win rate.
- A continuous improvement is considered stable when the paired MAE or RMSE confidence interval excludes zero in the favorable direction.

## Final Acceptance

Internal validation passes when:

- At least two rolling-origin folds complete.
- The U-Net and statistical baselines are evaluated with the same dates and masks.
- At least one continuous metric improvement over Raw GFS and quantile mapping is stable under date-block bootstrap.
- Results are directionally consistent across most random seeds.
- Event claims include PR-AUC, Brier Skill Score, FSS, and explicit false-alarm tradeoffs.

External validation remains pending until an untouched complete period, preferably the 2026 flood season, is available. No production-readiness claim is permitted before that holdout and an independent observation comparison are complete.

