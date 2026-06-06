# RainCalib v2/v2.1 U-Net 结果总结

## 已完成内容

RainCalib v2 已完成一个轻量 U-Net 残差订正闭环，v2.1 在此基础上加入强降水加权 loss、balanced sampler 和多任务事件头：

1. 将 v1 feature store 重建为按预报场组织的张量数据集。
2. 使用 GFS 降水、lead、cycle 和季节编码作为输入通道。
3. 训练小型 U-Net 输出 `IMERG - GFS` 残差场。
4. 在独立 test split 上对比 Raw GFS、RainCalib v1 和 RainCalib v2。
5. 诊断 v2 强降水尾部低估问题。
6. 训练 v2.1 多任务模型，同时输出连续订正雨量和 50mm/100mm 事件概率。
7. 保存模型、训练历史、预测结果、指标 CSV、诊断 CSV 和 Markdown 报告。

## 数据与模型

张量数据集：

```text
outputs/rainfall_benchmark/ml/tensors/raincalib_tensor_dataset.npz
```

样本规模：

| Item | Count |
|---|---:|
| Forecast fields | 7110 |
| Train fields | 6374 |
| Validation fields | 416 |
| Test fields | 320 |
| Valid grid-cell samples | 4329975 |
| Grid size | 23 x 31 |

模型：

```text
outputs/rainfall_benchmark/ml/models/raincalib_unet_v21.pt
```

训练设置：

| Item | Value |
|---|---:|
| Base channels | 16 |
| Depth | 2 |
| Batch size | 32 |
| Epochs | 5 |
| Learning rate | 0.001 |
| Mixed precision | true |

v2.1 过拟合烟测从 `0.161579` 降到 `0.037428`，说明数据、模型、损失函数和 GPU 训练链路已经连通。

## Test Split 连续指标

| Model | MAE | RMSE | Bias | Pearson r |
|---|---:|---:|---:|---:|
| Raw GFS | 8.726 | 17.209 | -0.600 | 0.048 |
| RainCalib v1 sklearn | 7.785 | 13.363 | -0.335 | 0.132 |
| RainCalib v2 U-Net | 7.188 | 13.333 | -0.855 | 0.195 |
| RainCalib v2.1 U-Net | 7.049 | 13.084 | -0.751 | 0.241 |

相对 Raw GFS：

- MAE 改善约 19.2%。
- RMSE 改善约 24.0%。
- Pearson r 从 0.048 提升到 0.241。

相对 RainCalib v1：

- MAE 继续改善约 9.5%。
- RMSE 改善约 2.1%。
- Pearson r 从 0.132 提升到 0.241。
- Bias 仍为负，说明连续雨量场对强降水仍偏保守。

## Test Split 事件指标

| Threshold | Model | CSI | POD | FAR | ETS |
|---:|---|---:|---:|---:|---:|
| 50 mm | Raw GFS | 0.010 | 0.016 | 0.971 | 0.004 |
| 50 mm | RainCalib v1 sklearn | 0.006 | 0.006 | 0.878 | 0.005 |
| 50 mm | RainCalib v2 U-Net | 0.001 | 0.001 | 0.977 | 0.000 |
| 50 mm | RainCalib v2.1 event head | 0.060 | 0.489 | 0.936 | 0.043 |
| 100 mm | Raw GFS | 0.000 | 0.000 | 1.000 | -0.001 |
| 100 mm | RainCalib v1 sklearn | 0.000 | 0.000 | nan | 0.000 |
| 100 mm | RainCalib v2 U-Net | 0.000 | 0.000 | 1.000 | -0.000 |
| 100 mm | RainCalib v2.1 event head | 0.011 | 0.043 | 0.985 | 0.009 |

结论：v2.1 已经同时改善连续订正和事件识别。需要注意的是，连续雨量阈值本身仍会低估强降水；50mm/100mm 事件应用应优先使用事件头概率，而不是直接用订正雨量阈值。

## 误差诊断

旧 v2 诊断显示，强降水尾部被明显平滑：

| Observed bin | v2 prediction mean | Target mean |
|---|---:|---:|
| 50-100 mm | 9.230 | 65.699 |
| 100+ mm | 8.881 | 125.247 |

v2.1 温和权重版本有所改善，但连续雨量场仍偏保守：

| Observed bin | v2.1 prediction mean | Target mean |
|---|---:|---:|
| 50-100 mm | 11.042 | 65.699 |
| 100+ mm | 10.361 | 125.247 |

因此 v2.1 的主要事件收益来自多任务事件头，而不是连续雨量场直接跨过 50mm/100mm 阈值。

## 复现命令

激活 GPU 环境：

```bash
conda activate raincalib-gpu
```

构建张量数据集：

```bash
PYTHONPATH=src python scripts/build_raincalib_tensor_dataset.py --config configs/raincalib_unet.yaml --overwrite
```

训练 v2.1 U-Net：

```bash
PYTHONPATH=src python scripts/train_raincalib_unet.py --config configs/raincalib_unet.yaml
```

评估 test split：

```bash
PYTHONPATH=src python scripts/evaluate_raincalib_unet.py --config configs/raincalib_unet.yaml --split test --tune-split val
PYTHONPATH=src python scripts/diagnose_raincalib_unet_errors.py --config configs/raincalib_unet.yaml --split test
```

完整报告：

```text
outputs/rainfall_benchmark/ml/reports/raincalib_unet_v21_report.md
outputs/rainfall_benchmark/ml/reports/raincalib_unet_v21_test_diagnostics_report.md
```

## 下一轮建议

下一轮不应先扩大模型，而应先继续改善尾部降水幅度和事件概率校准：

- 对事件头做可靠性曲线和分位数校准，降低 false alarms。
- 尝试 lead-specific 或 month-specific calibration，修正 240h/360h 长时效误差。
- 用邻域最大值、地形和水汽/风场等附加特征提高强降水定位能力。
- 保留当前温和权重配置作为默认基线；激进权重会提高事件命中，但会显著破坏 MAE 和 bias。
