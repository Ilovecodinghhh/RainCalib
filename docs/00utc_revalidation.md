> 空间标签勘误（2026-09-29）：方案②的真实数据端到端检查确认，旧特征生成将IMERG的经度×纬度数组直接按纬度×经度展平，导致格点标签错位。本文包括00 UTC复核在内的性能结论撤回，保留数字仅用于追溯。正在从原始GRIB和半小时IMERG重建数据并重训，见 [方案②进展](halfhourly_revalidation.md)。

# 00 UTC时间匹配复核

2026-09-29：执行方案①，仅保留00 UTC起报，保留原有24、48、72、96、120、168、240、360小时时效及24小时累计定义。观测日 = 起报日 + (时效 − 24)/24 天。12 UTC暂不参与训练、验证、测试。

原始特征7110场、4329975行；筛选后3555场、2164983行。对每个保留场校验观测日期，再重新构建张量，未复用旧模型或旧预测。保留既有空间插值和缺测处理，不能据此宣称所有空间代表性问题已经消除。

新产物位于 outputs/rainfall_benchmark/ml_00utc 和 outputs/rainfall_benchmark/experiments_00utc，旧目录保留作为历史记录。原有两折、五种子、5轮训练和1000次日期块bootstrap设置不变。阈值和概率校准仅使用验证集。

## 复现

```bash
PYTHONPATH=src python scripts/prepare_00utc_features.py
PYTHONPATH=src python scripts/build_raincalib_tensor_dataset.py
PYTHONPATH=src python scripts/run_statistical_validation.py
PYTHONPATH=src python scripts/run_unet_experiment_matrix.py
PYTHONPATH=src python scripts/audit_experiment_data.py
PYTHONPATH=src python scripts/create_experimental_acceptance_report.py
```

首次筛选要求新特征文件不存在，避免覆盖历史或已生成的复核数据。没有旧特征时，先运行基准流程生成观测缓存，再使用 build_raincalib_features.py 从原始00 UTC预报构建特征。

## 状态

已完成两折 × 五种子共10次U-Net重训、统计基线重训及1000次日期块bootstrap。旧混合时次性能不再作为当前结论。详细指标见 [00 UTC完整结果](00utc_validation_results.md)。

| 测试折 | 原始GFS RMSE | U-Net RMSE（五种子均值 ± 标准差） | 相对降低 |
| --- | ---: | ---: | ---: |
| rolling_2024 | 15.760 | 13.280 ± 0.138 | 15.7% |
| rolling_2025 | 15.255 | 12.707 ± 0.208 | 16.7% |

10/10次运行相对原始GFS及分位数映射的RMSE差值95%置信区间均低于零。因此“00 UTC上，相对这两个基线的RMSE改善”保留。

结论边界：

- U-Net平均RMSE未优于格点月气候态（两折分别13.183、12.464）；2025折也未优于v1（12.481）。不能宣称优于所有基线。
- MAE均值从6.919降至6.768、从7.905降至6.981，但2024折不是所有种子都改善。
- 暴雨识别依然很弱：事件头50 mm FAR为0.970/0.974，100 mm FAR为0.984/0.998。50 mm FSS-5在2024折由原始GFS的0.203降至0.120，在2025折由0.098升至0.126；不能沿用“两个折均改善”的表述。
- 既有内部验收规则通过只说明相对GFS和分位数映射的RMSE及指标完整性达标，不等于暴雨预警能力验证。
- 数据筛选改变了样本构成，不能把新旧指标差直接解释为时间修正的净效应。当前结果也不能推广到12 UTC。

验证：原有完整测试48项通过；追加缓存入口回归测试后，时间匹配相关9项全部通过（合计49项已验证）。测试临时目录使用工作区内目录，避免系统临时目录权限问题。

## 后续方案②

在本次复核后，再引入半小时IMERG以恢复12 UTC。必须按真实UTC起止时间收集48个连续半小时时段，检查缺失、重复、单位和边界；若变量为mm/h，需乘0.5小时后求和。该步骤尚未实施，当前代码拒绝用自然日观测评估12 UTC。
