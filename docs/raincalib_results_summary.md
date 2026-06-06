# RainCalib v1 结果总结

## 已完成内容

RainCalib v1 已完成从 GFS/IMERG 数据到 sklearn 订正模型的完整闭环：

1. 构建 feature store。
2. 训练降水回归订正模型。
3. 训练 50mm/100mm 事件分类器。
4. 在时间独立 test split 上评估 Raw GFS 和 RainCalib。
5. 准备 v2 轻量深度学习路线和 GPU 检查入口。

## 数据与样本

Feature store：

```text
outputs/rainfall_benchmark/ml/features/raincalib_features.parquet
```

样本规模：

| Item | Count |
|---|---:|
| GFS files | 918 |
| Forecast fields | 7110 |
| Total rows | 4329975 |
| Train rows | 3881752 |
| Validation rows | 253344 |
| Test rows | 194879 |

默认特征：

```text
gfs_rain
lead_hour
cycle
latitude
longitude
month_sin
month_cos
day_of_year_sin
day_of_year_cos
```

## Test Split 结果

连续降水订正：

| Model | MAE | RMSE | Bias | Pearson r |
|---|---:|---:|---:|---:|
| Raw GFS | 8.726 | 17.209 | -0.600 | 0.048 |
| RainCalib v1 | 7.785 | 13.363 | -0.335 | 0.132 |

相对改善：

- MAE 改善约 10.8%。
- RMSE 改善约 22.3%。
- Bias 更接近 0。
- Pearson r 从 0.048 提升到 0.132。

事件结果：

| Threshold | Model | CSI | POD | FAR | ETS |
|---:|---|---:|---:|---:|---:|
| 50 mm | Raw GFS | 0.010 | 0.016 | 0.971 | 0.004 |
| 50 mm | RainCalib rain | 0.006 | 0.006 | 0.878 | 0.005 |
| 50 mm | RainCalib classifier | 0.023 | 0.948 | 0.977 | 0.004 |
| 100 mm | Raw GFS | 0.000 | 0.000 | 1.000 | -0.001 |
| 100 mm | RainCalib rain | 0.000 | 0.000 | nan | 0.000 |
| 100 mm | RainCalib classifier | 0.003 | 1.000 | 0.997 | 0.001 |

结论：v1 已经证明连续降水订正有效，但事件分类器仍是探索性结果。强降水事件识别需要加入空间上下文、邻域统计或 v2 深度学习模型。

## 复现命令

```bash
PYTHONPATH=src python scripts/build_raincalib_features.py --config configs/raincalib.yaml --overwrite
PYTHONPATH=src python scripts/train_raincalib_baseline.py --config configs/raincalib.yaml
PYTHONPATH=src python scripts/evaluate_raincalib.py --config configs/raincalib.yaml --split test
```

GPU 检查：

```bash
python scripts/check_raincalib_gpu.py
```

## 后续进展

v2.1 已经完成轻量 U-Net 多任务空间订正模型，结果见：

```text
docs/raincalib_v2_results_summary.md
outputs/rainfall_benchmark/ml/reports/raincalib_unet_v21_report.md
outputs/rainfall_benchmark/ml/reports/raincalib_unet_v21_test_diagnostics_report.md
```

v2.1 连续指标已进一步优于 v1 和旧 v2，并通过事件头明显改善 50mm/100mm 事件识别。下一轮应优先处理事件概率校准、长时效误差和强降水连续雨量幅度不足。
