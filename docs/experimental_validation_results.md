> 时间匹配勘误（2026-09-29）：本文旧指标包含12 UTC预报与IMERG自然日标签的不匹配，不再作为已验证结论或验收依据。00 UTC重新训练评估见 [复核记录](00utc_revalidation.md)。

# RainCalib Experimental Validation Results

- Internal research acceptance: **PASS**
- External / production acceptance: **PENDING**
- Data: 2023-05-01 to 2025-09-30, 459 independent observed dates
- Forecast fields: 7110; valid grid-cell pairs: 4329975
- Frozen baseline commit: `423fc6b793aa5b8ea3ded57a1e5b37aaa2a76803`

The internal protocol passes, but this is not a production-readiness claim. The complete 2026 flood season and a second independent observation product are not yet available.

## Acceptance Checklist

| Criterion | Status | Evidence |
| --- | --- | --- |
| Two rolling-origin folds complete | PASS | 2 |
| Five fixed seeds per fold | PASS | 5 |
| RMSE beats Raw GFS with date-block 95% CI | PASS | 10/10 runs |
| RMSE beats Quantile Mapping with date-block 95% CI | PASS | 10/10 runs |
| PR-AUC, BSS, and FSS reported | PASS | 50/100 mm; 1/3/5 cells |
| Untouched future holdout and independent observations | PENDING | 2026 flood season / second product unavailable |

## Continuous Results

U-Net values are mean +/- standard deviation across five seeds. Other baselines are deterministic.

| Fold | Model | MAE | RMSE | Bias | Pearson r |
| --- | --- | --- | --- | --- | --- |
| rolling_2024 | raw_gfs | 7.050 | 16.225 | 0.558 | 0.062 |
| rolling_2024 | mean_bias | 7.193 | 16.210 | 0.880 | 0.065 |
| rolling_2024 | quantile_mapping | 7.516 | 17.014 | 0.922 | 0.062 |
| rolling_2024 | grid_month_climatology | 7.097 | 13.183 | 1.487 | 0.028 |
| rolling_2024 | raincalib_v1 | 7.360 | 14.043 | 1.471 | -0.012 |
| rolling_2024 | raincalib_v2.1 (5 seeds) | 6.883 +/- 0.369 | 13.395 +/- 0.233 | 1.438 +/- 0.600 | 0.127 +/- 0.032 |
| rolling_2025 | raw_gfs | 7.887 | 15.257 | -0.418 | 0.051 |
| rolling_2025 | mean_bias | 7.826 | 15.215 | -0.420 | 0.054 |
| rolling_2025 | quantile_mapping | 8.451 | 16.489 | -0.046 | 0.053 |
| rolling_2025 | grid_month_climatology | 7.015 | 12.464 | -0.640 | 0.018 |
| rolling_2025 | raincalib_v1 | 7.063 | 12.558 | -0.718 | 0.044 |
| rolling_2025 | raincalib_v2.1 (5 seeds) | 7.078 +/- 0.268 | 12.385 +/- 0.155 | 0.136 +/- 0.375 | 0.165 +/- 0.041 |

## Paired Date-Block Bootstrap

Deltas are RainCalib v2.1 minus baseline. Negative MAE/RMSE is favorable; each run uses 1,000 complete-date resamples.

| Fold | Baseline | Mean delta RMSE | CI envelope | Stable seeds |
| --- | --- | --- | --- | --- |
| rolling_2024 | quantile_mapping | -3.640 | -4.530 to -2.741 | 5/5 |
| rolling_2024 | raw_gfs | -2.841 | -3.722 to -2.003 | 5/5 |
| rolling_2025 | quantile_mapping | -4.105 | -4.845 to -3.511 | 5/5 |
| rolling_2025 | raw_gfs | -2.870 | -3.620 to -2.279 | 5/5 |

## Heavy-Rainfall Results

| Fold | mm | CSI | POD | FAR | Cal PR-AUC | Cal BSS | Raw FSS-5 | Event-head FSS-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rolling_2024 | 50 | 0.035 | 0.121 | 0.948 | 0.0304 | -0.0059 | 0.181 | 0.199 |
| rolling_2024 | 100 | 0.013 | 0.157 | 0.980 | 0.0103 | 0.0027 | 0.130 | 0.072 |
| rolling_2025 | 50 | 0.029 | 0.050 | 0.929 | 0.0311 | 0.0108 | 0.100 | 0.210 |
| rolling_2025 | 100 | 0.006 | 0.073 | 0.993 | 0.0033 | 0.0021 | 0.027 | 0.036 |

The 50 mm event head improves neighborhood-scale FSS, especially in rolling_2025. The 100 mm event remains unstable, with high false-alarm rates and weak calibrated detection. Probability calibration sharply improves Brier score but can reduce useful event recall, so calibrated probabilities and screening decisions must remain separate products.

## Interpretation

- The strongest defensible claim is stable RMSE reduction across both temporal folds and all five seeds.
- MAE improvement is stable in rolling_2025 but not for every rolling_2024 seed.
- RainCalib improves broad rainfall-field error more reliably than exact extreme-event localization.
- Pixel counts are never treated as independent; uncertainty is blocked by observed date.
- External validation remains mandatory before operational or production claims.

## Reproduction

```bash
PYTHONPATH=src python scripts/audit_experiment_data.py
PYTHONPATH=src python scripts/run_statistical_validation.py
PYTHONPATH=src python scripts/run_unet_experiment_matrix.py
PYTHONPATH=src python scripts/recompute_unet_fss.py
PYTHONPATH=src python scripts/create_experimental_acceptance_report.py
```
