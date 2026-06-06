# RainCalib v2 轻量深度学习路线

## 目标

在 RainCalib v1 sklearn baseline 已经证明连续降水订正有效后，v2 使用 12GB 显存可承受的小型深度学习模型进一步学习空间结构。

## 推荐模型

优先实现 patch-based 小型 U-Net：

- 输入：GFS 24h rainfall、lead hour、cycle、day-of-year 编码。
- 输出：`IMERG - GFS` 残差场。
- 最终订正：`corrected = max(GFS + residual, 0)`。

不建议本阶段训练全球基础模型。当前数据、显存和项目目标更适合区域订正模型。

## 训练策略

- 区域范围保持长江中下游。
- patch size 从 `32x32` 或整区小网格开始。
- batch size 从 `1-4` 开始。
- 使用 mixed precision。
- 先训练单 lead 或少数 lead，再扩展到全部 lead。
- 优先以 MAE/RMSE 作为连续目标，事件 CSI/ETS 作为辅助评估。

## 验收标准

- GPU 环境能通过 `scripts/check_raincalib_gpu.py`。
- 1 epoch 训练不 OOM。
- 小样本 overfit smoke test 能下降 loss。
- test split 上至少一个核心指标优于 Raw GFS；理想情况下也优于 v1 sklearn baseline。

## 当前 v2/v2.1 进展

第一版轻量 U-Net 和 v2.1 多任务增强版已经完成端到端闭环：

- 张量数据集：`outputs/rainfall_benchmark/ml/tensors/raincalib_tensor_dataset.npz`
- v2.1 checkpoint：`outputs/rainfall_benchmark/ml/models/raincalib_unet_v21.pt`
- v2.1 评估报告：`outputs/rainfall_benchmark/ml/reports/raincalib_unet_v21_report.md`
- v2.1 诊断报告：`outputs/rainfall_benchmark/ml/reports/raincalib_unet_v21_test_diagnostics_report.md`
- 结果总结：`docs/raincalib_v2_results_summary.md`

张量数据规模：

| Item | Count |
|---|---:|
| Forecast fields | 7110 |
| Train fields | 6374 |
| Validation fields | 416 |
| Test fields | 320 |
| Valid grid-cell samples | 4329975 |
| Grid size | 23 x 31 |

v2.1 过拟合烟测从 `0.161579` 降到 `0.037428`，GPU 训练链路已经验证。

## 当前 v1 基线

当前 v1 baseline 的 test split 连续指标已经优于 Raw GFS：

| Model | MAE | RMSE | Bias | Pearson r |
|---|---:|---:|---:|---:|
| Raw GFS | 8.726 | 17.209 | -0.600 | 0.048 |
| RainCalib v1 | 7.785 | 13.363 | -0.335 | 0.132 |
| RainCalib v2 U-Net | 7.188 | 13.333 | -0.855 | 0.195 |
| RainCalib v2.1 U-Net | 7.049 | 13.084 | -0.751 | 0.241 |

连续订正已经优于 Raw GFS、v1 baseline 和旧 v2。v2.1 的事件头也显著提高了 50mm/100mm 事件识别：

| Threshold | Decision | CSI | POD | FAR | ETS |
|---:|---|---:|---:|---:|---:|
| 50 mm | v2.1 event head | 0.060 | 0.489 | 0.936 | 0.043 |
| 100 mm | v2.1 event head | 0.011 | 0.043 | 0.985 | 0.009 |

连续雨量场仍低估强降水尾部，因此事件类应用应优先使用事件头概率。

## 当前 GPU 环境

- 硬件：NVIDIA GeForce RTX 5070 Ti Laptop GPU，约 12GB 显存。
- 环境名：`raincalib-gpu`
- Python：3.12.13
- PyTorch：2.11.0+cu128
- torchvision：0.26.0+cu128
- torchaudio：2.11.0+cu128
- CUDA 检查：`torch.cuda.is_available()` 为 true，CUDA tensor 运算通过。

激活环境：

```bash
conda activate raincalib-gpu
```

验证环境：

```bash
python scripts/check_raincalib_gpu.py
PYTHONPATH=src python -m pytest -q
```

如果在非交互脚本里运行，可以用：

```bash
conda run -n raincalib-gpu python scripts/check_raincalib_gpu.py
```

## v2.1 复现命令

```bash
conda activate raincalib-gpu
PYTHONPATH=src python scripts/build_raincalib_tensor_dataset.py --config configs/raincalib_unet.yaml --overwrite
PYTHONPATH=src python scripts/train_raincalib_unet.py --config configs/raincalib_unet.yaml
PYTHONPATH=src python scripts/evaluate_raincalib_unet.py --config configs/raincalib_unet.yaml --split test --tune-split val
PYTHONPATH=src python scripts/diagnose_raincalib_unet_errors.py --config configs/raincalib_unet.yaml --split test
```

## 已完成的 v2.1 增强

1. 误差诊断脚本：按 lead、cycle、month、观测强度和空间格点输出诊断。
2. 加权残差损失：对 25mm/50mm/100mm 以上像元做温和加权。
3. Balanced sampler：训练中轻微提高含 50mm/100mm 样本的出现概率。
4. 多任务事件头：同时输出连续残差和 50mm/100mm 事件概率。
5. 重新评估报告：事件概率阈值在 validation split 上调优，并在 test split 上评估。

## 下一轮优先级

1. 事件概率校准：做 reliability curve、isotonic 或分位数校准，降低 false alarms。
2. Lead-specific calibration：重点修正 240h/360h 长时效误差。
3. 强降水幅度增强：加入邻域最大值、地形、水汽和风场等特征，改善连续雨量场的尾部幅度。
4. 阈值策略评估：针对不同业务目标分别优化 CSI、POD 或 FAR。
