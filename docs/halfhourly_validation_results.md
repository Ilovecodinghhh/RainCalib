# RainCalib Half-hourly IMERG 00/12 UTC Validation Results

- Internal research acceptance: **PASS**
- External / production acceptance: **PENDING**
- Data: 2023-05-01 to 2025-09-30, 459 independent observed dates
- Forecast fields: 7062; valid grid-cell pairs: 5035190
- Base commit (working-tree temporal corrections applied): `93ecb6a962d2ea593f66b0c74b61ae7a22d0445e`

The checklist below determines internal acceptance. An untouched future holdout and independent observations have not been evaluated in this run.

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
| rolling_2024 | raw_gfs | 5.535 | 13.829 | 0.469 | 0.324 |
| rolling_2024 | mean_bias | 5.746 | 13.824 | 0.876 | 0.326 |
| rolling_2024 | quantile_mapping | 5.834 | 14.413 | 0.786 | 0.342 |
| rolling_2024 | grid_month_climatology | 7.183 | 13.270 | 1.526 | 0.030 |
| rolling_2024 | raincalib_v1 | 6.226 | 12.760 | 1.039 | 0.264 |
| rolling_2024 | raincalib_v2.1 (5 seeds) | 6.117 +/- 0.114 | 12.525 +/- 0.162 | 1.418 +/- 0.239 | 0.354 +/- 0.028 |
| rolling_2025 | raw_gfs | 6.455 | 13.442 | -0.449 | 0.293 |
| rolling_2025 | mean_bias | 6.457 | 13.418 | -0.358 | 0.294 |
| rolling_2025 | quantile_mapping | 6.880 | 14.374 | -0.114 | 0.302 |
| rolling_2025 | grid_month_climatology | 7.086 | 12.542 | -0.607 | 0.025 |
| rolling_2025 | raincalib_v1 | 6.267 | 11.795 | -0.452 | 0.331 |
| rolling_2025 | raincalib_v2.1 (5 seeds) | 6.251 +/- 0.163 | 11.747 +/- 0.123 | 0.008 +/- 0.434 | 0.370 +/- 0.008 |

## Paired Date-Block Bootstrap

Deltas are RainCalib v2.1 minus baseline. Negative MAE/RMSE is favorable; each run uses 1,000 complete-date resamples.

| Fold | Baseline | Mean delta RMSE | CI envelope | Stable seeds |
| --- | --- | --- | --- | --- |
| rolling_2024 | quantile_mapping | -1.906 | -2.462 to -1.176 | 5/5 |
| rolling_2024 | raw_gfs | -1.311 | -1.874 to -0.681 | 5/5 |
| rolling_2025 | quantile_mapping | -2.627 | -3.051 to -2.206 | 5/5 |
| rolling_2025 | raw_gfs | -1.692 | -2.207 to -1.217 | 5/5 |

## Heavy-Rainfall Results

| Fold | mm | CSI | POD | FAR | Cal PR-AUC | Cal BSS | Raw FSS-5 | Event-head FSS-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rolling_2024 | 50 | 0.098 | 0.230 | 0.852 | 0.1022 | 0.0276 | 0.354 | 0.330 |
| rolling_2024 | 100 | 0.060 | 0.124 | 0.870 | 0.0439 | 0.0193 | 0.194 | 0.241 |
| rolling_2025 | 50 | 0.093 | 0.215 | 0.857 | 0.0864 | 0.0400 | 0.275 | 0.303 |
| rolling_2025 | 100 | 0.021 | 0.095 | 0.972 | 0.0107 | 0.0026 | 0.071 | 0.090 |

Event skill must be assessed separately at each threshold and fold using the table above; RMSE improvements alone do not establish heavy-rainfall warning skill.

## Interpretation

- Check both baseline comparisons above for stable RMSE improvement across every seed.
- MAE and bias require separate interpretation from RMSE; consult the paired intervals.
- Both 00/12 UTC windows use 48 consecutive half-hourly Late V07 intervals. Noon windows crossing split boundaries are purged. Adjacent verification windows overlap, so date-block intervals retain temporal-dependence limitations.
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
