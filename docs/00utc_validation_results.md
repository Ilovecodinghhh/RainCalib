> 空间标签勘误（2026-09-29）：方案②的真实数据端到端检查确认，旧特征生成将IMERG的经度×纬度数组直接按纬度×经度展平，导致格点标签错位。本文包括00 UTC复核在内的性能结论撤回，保留数字仅用于追溯。正在从原始GRIB和半小时IMERG重建数据并重训，见 [方案②进展](halfhourly_revalidation.md)。

# RainCalib 00 UTC Validation Results

- Internal research acceptance: **PASS**
- External / production acceptance: **PENDING**
- Data: 2023-05-01 to 2025-09-30, 459 independent observed dates
- Forecast fields: 3555; valid grid-cell pairs: 2164983
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
| rolling_2024 | raw_gfs | 6.919 | 15.760 | 0.441 | 0.068 |
| rolling_2024 | mean_bias | 7.003 | 15.721 | 0.661 | 0.072 |
| rolling_2024 | quantile_mapping | 7.293 | 16.760 | 0.669 | 0.064 |
| rolling_2024 | grid_month_climatology | 7.097 | 13.183 | 1.487 | 0.028 |
| rolling_2024 | raincalib_v1 | 7.313 | 13.992 | 1.421 | -0.011 |
| rolling_2024 | raincalib_v2.1 (5 seeds) | 6.768 +/- 0.287 | 13.280 +/- 0.138 | 1.116 +/- 0.453 | 0.110 +/- 0.019 |
| rolling_2025 | raw_gfs | 7.905 | 15.255 | -0.398 | 0.047 |
| rolling_2025 | mean_bias | 7.845 | 15.206 | -0.390 | 0.050 |
| rolling_2025 | quantile_mapping | 8.471 | 16.634 | -0.054 | 0.048 |
| rolling_2025 | grid_month_climatology | 7.015 | 12.464 | -0.640 | 0.018 |
| rolling_2025 | raincalib_v1 | 7.035 | 12.481 | -0.714 | 0.053 |
| rolling_2025 | raincalib_v2.1 (5 seeds) | 6.981 +/- 0.321 | 12.707 +/- 0.208 | -0.450 +/- 0.480 | 0.102 +/- 0.026 |

## Paired Date-Block Bootstrap

Deltas are RainCalib v2.1 minus baseline. Negative MAE/RMSE is favorable; each run uses 1,000 complete-date resamples.

| Fold | Baseline | Mean delta RMSE | CI envelope | Stable seeds |
| --- | --- | --- | --- | --- |
| rolling_2024 | quantile_mapping | -3.499 | -4.345 to -2.798 | 5/5 |
| rolling_2024 | raw_gfs | -2.487 | -3.226 to -1.894 | 5/5 |
| rolling_2025 | quantile_mapping | -3.932 | -4.490 to -3.261 | 5/5 |
| rolling_2025 | raw_gfs | -2.550 | -3.077 to -1.909 | 5/5 |

## Heavy-Rainfall Results

| Fold | mm | CSI | POD | FAR | Cal PR-AUC | Cal BSS | Raw FSS-5 | Event-head FSS-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rolling_2024 | 50 | 0.025 | 0.236 | 0.970 | 0.0208 | -0.0161 | 0.203 | 0.120 |
| rolling_2024 | 100 | 0.008 | 0.035 | 0.984 | 0.0061 | 0.0006 | 0.139 | 0.068 |
| rolling_2025 | 50 | 0.012 | 0.030 | 0.974 | 0.0196 | 0.0049 | 0.098 | 0.126 |
| rolling_2025 | 100 | 0.001 | 0.058 | 0.998 | 0.0018 | 0.0018 | 0.019 | 0.017 |

Event skill must be assessed separately at each threshold and fold using the table above; RMSE improvements alone do not establish heavy-rainfall warning skill.

## Interpretation

- Check both baseline comparisons above for stable RMSE improvement across every seed.
- MAE and bias require separate interpretation from RMSE; consult the paired intervals.
- These results apply only to 00 UTC; restoring 12 UTC requires subdaily observation windows and a new evaluation.
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
