# 方案A详细计划：GFS Historical Archive + IMERG 长江中下游强降水 Benchmark 与订正模型

生成日期：2026-06-05

## 1. 背景与目标

当前项目已经完成了一个基于 ECMWF Open Data 的 AIFS/IFS 强降水评估原型，但 ECMWF MARS/Archive 当前不可获得，导致 2023-2025 完整历史时段无法通过 ECMWF 免费数据补齐。方案A的核心思路是：把主 benchmark 数据源切换到可长期获取的 NOAA/NCAR GFS 历史预报档案，同时保留当前 ECMWF 2025 免费窗口作为高价值对照。

本方案要完成的最终目标：

1. 建立 2023-2025 年汛期长江中下游区域的 GFS-vs-IMERG 强降水评估基准。
2. 保留并复用现有 ECMWF AIFS/IFS 2025-05-01 至 2025-09-30 评估结果。
3. 构建一个区域降水订正机器学习项目 RainCalib，为后续气象大模型算法项目提供训练样本、评估框架和对照基线。
4. 在 12GB 显存约束下优先完成可运行、可复现、可发表为项目成果的区域级算法路线，而不是训练全球尺度基础模型。

## 2. 官方数据依据

本方案采用的数据源和参考入口如下：

- GFS 历史预报：NSF NCAR GDEX `ds084.1` / `d084001`，NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive  
  <https://gdex.ucar.edu/datasets/ds084.1/>
- NOAA NCEI GFS 产品说明：Global Forecast System  
  <https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast>
- NOAA/NCEP GFS 0.25 实时/近实时筛选接口，可用于小样本测试和字段确认  
  <https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl>
- NASA GPM IMERG Late Run GES DISC 1 Day V07  
  <https://gpm.nasa.gov/node/3322>
- WeatherBench2 数据指南，作为后续 AI 模型对照分支的参考，不作为方案A主数据源  
  <https://weatherbench2.readthedocs.io/en/latest/data-guide.html>

GFS 0.25 历史档案提供全球格点预报，预报步长覆盖中短期到 384h。IMERG Daily V07B 提供 0.1 度日尺度降水观测。两者足够支持 24h 累积降水、50mm/100mm 极端降水事件、lead-time skill 曲线和区域机器学习订正任务。

## 3. 当前项目基础

当前工作目录：

```text
D:\weathermodel
```

已有能力：

- 已有 ECMWF AIFS/IFS Open Data 下载脚本。
- 已有 IMERG GES DISC 下载脚本。
- 已有下载覆盖检查脚本。
- 已有 24h 累积计算、IMERG 匹配、区域裁剪、regrid、连续指标、分类指标、图表和 Markdown 报告生成流程。
- 当前默认配置评估时段为 2025-05-01 至 2025-09-30。
- 当前 ECMWF 结果已经生成，可作为后续对照基线。

当前瓶颈：

- ECMWF Archive/MARS 不可用时，无法免费获取 2023-2025 全历史 AIFS/IFS。
- ECMWF Open Data 不是完整历史档案，旧日期和部分长 lead 请求会 404。
- 12GB 显存不足以训练全球级气象基础模型，但足够完成区域级 post-processing、轻量 CNN/U-Net、patch 训练和概率订正模型。

## 4. 方案A范围

### 4.1 纳入范围

- 2023-2025 年 05-01 至 09-30 汛期。
- 长江中下游区域：经度 108E-123E，纬度 24N-35N。
- GFS 0.25 度预报数据。
- IMERG Daily V07B 观测数据。
- 24h 累积降水评估。
- 50mm/24h 和 100mm/24h 强降水事件评估。
- GFS 原始预报、GFS 订正结果、ECMWF 2025 对照结果的统一报告。

### 4.2 暂不纳入范围

- 不依赖 ECMWF MARS/Archive。
- 不训练全球尺度天气基础模型。
- 不把 2023-2025 GFS 完整历史结果和 2025 ECMWF 短窗口结果解释为严格公平的全历史模型比较。
- 不做业务化预警服务，仅作为研究和算法评估项目。

## 5. 数据设计

### 5.1 时段设置

主 benchmark：

```yaml
period:
  years: [2023, 2024, 2025]
  season_start: "05-01"
  season_end: "09-30"
  primary_comparison_years: [2023, 2024, 2025]
  background_baseline_years: []
```

ECMWF 对照 benchmark 保持现状：

```yaml
period:
  years: [2025]
  season_start: "05-01"
  season_end: "09-30"
  primary_comparison_years: [2025]
```

### 5.2 GFS 周期与 lead

建议先使用 00/12 UTC，与当前 ECMWF pipeline 对齐：

```yaml
forecast:
  models: ["gfs"]
  cycles_utc: ["00", "12"]
  lead_hours: [24, 48, 72, 96, 120, 168, 240, 360]
  accumulation_hours: 24
```

为了计算所有 24h 窗口，下载时必须补齐 `lead - 24` step：

```text
评估 lead: 24, 48, 72, 96, 120, 168, 240, 360
所需 step: 24, 48, 72, 96, 120, 144, 168, 216, 240, 336, 360
```

计算规则：

```text
rain_24h(lead) = accumulation(step=lead) - accumulation(step=lead-24)
```

当 lead 为 24h 时，如果产品的 step=24 已经表示从 forecast start 到 +24h 的累计量，则直接使用 step=24。

### 5.3 文件目录

建议新增目录：

```text
outputs/rainfall_benchmark/raw/gfs/
outputs/rainfall_benchmark/raw/gfs/manifests/
outputs/rainfall_benchmark/processed/gfs/
outputs/rainfall_benchmark/ml/features/
outputs/rainfall_benchmark/ml/models/
outputs/rainfall_benchmark/ml/predictions/
outputs/rainfall_benchmark/ml/metrics/
outputs/rainfall_benchmark/ml/reports/
```

建议文件命名：

```text
gfs_YYYYMMDD_CC_fSSS.grib2
```

示例：

```text
gfs_20250501_00_f024.grib2
gfs_20250501_00_f048.grib2
gfs_20250501_12_f024.grib2
```

## 6. 配置文件计划

新增配置文件：

```text
configs/gfs_rainfall_benchmark.yaml
configs/raincalib.yaml
```

`configs/gfs_rainfall_benchmark.yaml` 用于完整 GFS benchmark：

```yaml
project:
  name: yangtze_gfs_extreme_rainfall_benchmark
  output_root: outputs/rainfall_benchmark

region:
  name: middle_lower_yangtze
  lon_min: 108.0
  lon_max: 123.0
  lat_min: 24.0
  lat_max: 35.0

period:
  years: [2023, 2024, 2025]
  season_start: "05-01"
  season_end: "09-30"
  primary_comparison_years: [2023, 2024, 2025]
  background_baseline_years: []

forecast:
  models: ["gfs"]
  cycles_utc: ["00", "12"]
  lead_hours: [24, 48, 72, 96, 120, 168, 240, 360]
  accumulation_hours: 24

rainfall:
  thresholds_mm_24h: [50.0, 100.0]

paths:
  gfs_raw: outputs/rainfall_benchmark/raw/gfs
  imerg_raw: outputs/rainfall_benchmark/raw/imerg
  processed: outputs/rainfall_benchmark/processed
  metrics: outputs/rainfall_benchmark/metrics
  figures: outputs/rainfall_benchmark/figures
  reports: outputs/rainfall_benchmark/reports
```

`configs/raincalib.yaml` 用于订正模型：

```yaml
project:
  name: yangtze_raincalib
  output_root: outputs/rainfall_benchmark/ml

data:
  benchmark_config: configs/gfs_rainfall_benchmark.yaml
  feature_stride: 2
  negative_sample_fraction: 0.15
  seed: 42

splits:
  train_start: "2023-05-01"
  train_end: "2025-08-15"
  val_start: "2025-08-16"
  val_end: "2025-09-10"
  test_start: "2025-09-11"
  test_end: "2025-09-30"

targets:
  rainfall_regression: true
  event_thresholds_mm_24h: [50.0, 100.0]

model:
  baseline_regressor: hist_gradient_boosting
  baseline_classifier: hist_gradient_boosting
```

## 7. 实施阶段

### 阶段1：GFS 下载能力

新增脚本：

```text
scripts/download_gfs_historical.py
```

功能要求：

- 从配置文件读取日期、cycle、lead 和 accumulation。
- 自动生成 required steps。
- 支持 dry-run，只打印计划下载清单和预计文件数量。
- 支持按日期范围试跑。
- 支持断点续跑，已存在且非零大小的文件默认跳过。
- 支持失败重试。
- 对每个请求记录 manifest。
- 对 404、认证失败、超时、文件过小、解码失败分别记录状态。

建议命令：

```bash
PYTHONPATH=src python scripts/download_gfs_historical.py --config configs/gfs_rainfall_benchmark.yaml --dry-run
PYTHONPATH=src python scripts/download_gfs_historical.py --config configs/gfs_rainfall_benchmark.yaml --start-date 2025-09-01 --end-date 2025-09-07
PYTHONPATH=src python scripts/download_gfs_historical.py --config configs/gfs_rainfall_benchmark.yaml
```

验收标准：

- dry-run 能列出 2023-2025 汛期所有日期、cycle、step。
- 小样本能下载并通过 GRIB 打开检查。
- 完整下载可中断后继续。

### 阶段2：GFS 覆盖检查

扩展或新增覆盖检查：

```text
scripts/check_gfs_coverage.py
```

检查内容：

- 期望文件数。
- 实际文件数。
- 缺失日期、cycle、step。
- 零字节文件。
- 临时文件。
- 无法被 cfgrib/xarray 打开的文件。
- 是否包含所有计算 24h 窗口所需的 start/end step。

建议命令：

```bash
PYTHONPATH=src python scripts/check_gfs_coverage.py --config configs/gfs_rainfall_benchmark.yaml --sample-count 3
```

验收标准：

- 完整时段 expected 与 present 一致。
- missing 为 0，或者有明确的 unavailable manifest。
- sample open 检查通过。

### 阶段3：GFS 数据读取和 24h 累积

新增模块：

```text
src/ai_weather_rainfall/data_gfs.py
```

实现要求：

- 精确识别 `gfs_YYYYMMDD_CC_fSSS.grib2`。
- 过滤 `.idx`、临时文件和非目标日期文件。
- 读取降水累积变量。
- 支持经纬度裁剪。
- 支持将 GFS 网格 regrid 到 IMERG 网格，或复用现有 regrid 模块。
- 计算每个 lead 的 24h 累积。

边界条件：

- 如果某个 lead 缺少 `lead - 24` step，则跳过该样本并记录原因。
- 如果 step 差值出现负降水，应记录并裁剪到 0，报告中注明处理规则。
- 如果经度是 0-360，需要转换或裁剪到项目区域。

验收标准：

- 能对 1 天、1 cycle、2 个 step 算出一个 24h 降水场。
- 输出维度与 IMERG 区域场一致或能被统一 regrid。

### 阶段4：GFS benchmark pipeline

扩展现有 CLI：

```bash
PYTHONPATH=src python -m ai_weather_rainfall.cli --config configs/gfs_rainfall_benchmark.yaml
```

输出文件：

```text
outputs/rainfall_benchmark/metrics/gfs_summary_metrics.csv
outputs/rainfall_benchmark/metrics/gfs_continuous_lead_summary.csv
outputs/rainfall_benchmark/metrics/gfs_categorical_lead_summary.csv
outputs/rainfall_benchmark/metrics/gfs_event_skill_summary.csv
outputs/rainfall_benchmark/reports/gfs_rainfall_benchmark_report.md
```

报告内容：

- 数据源。
- 时段。
- 区域。
- 样本量。
- 每个 lead 的 MAE/RMSE/Bias/Pearson r。
- 每个阈值的 CSI/POD/FAR/ETS。
- 与当前 ECMWF 2025 结果的窗口内对照。
- 明确说明该项目不是官方预警服务。

验收标准：

- 完整 GFS pipeline 可跑完。
- 每个 lead 都有样本数统计。
- 报告能解释样本数随 lead 下降的原因。

### 阶段5：多模型对照报告

新增或扩展报告：

```text
outputs/rainfall_benchmark/reports/model_comparison_report.md
```

对照规则：

- GFS 使用 2023-2025 完整汛期结果。
- ECMWF AIFS/IFS 只使用 2025 免费可用窗口。
- 另外生成一个 2025-only 公平窗口对照：
  - GFS 2025。
  - AIFS 2025。
  - IFS 2025。

推荐图表：

- lead vs MAE。
- lead vs RMSE。
- lead vs Bias。
- lead vs Pearson r。
- lead vs CSI。
- lead vs POD。
- lead vs FAR。
- lead vs ETS。

验收标准：

- 报告中不混淆“完整历史能力”和“同窗口公平比较”。
- 所有图表标注数据源和年份范围。

### 阶段6：RainCalib 特征工程

新增脚本：

```text
scripts/build_raincalib_features.py
```

输出：

```text
outputs/rainfall_benchmark/ml/features/raincalib_features.parquet
```

默认特征：

```text
run_date
observed_date
cycle
lead_hour
latitude
longitude
gfs_rain
month
day_of_year
target_rain
event_50
event_100
```

如果存在 ECMWF 2025 对照数据，可额外加入：

```text
aifs_rain
ifs_rain
gfs_minus_aifs
gfs_minus_ifs
aifs_minus_ifs
```

采样策略：

- 默认 `feature_stride=2`，降低样本量和内存压力。
- 对非强降水样本进行下采样，默认 `negative_sample_fraction=0.15`。
- 强降水样本全部保留。

验收标准：

- parquet 能被 pandas/pyarrow 读取。
- 标签列无明显错位。
- train/val/test 日期划分无泄漏。

### 阶段7：RainCalib v1 基线模型

新增脚本：

```text
scripts/train_raincalib_baseline.py
scripts/evaluate_raincalib.py
```

依赖：

```text
scikit-learn
joblib
pyarrow
```

模型：

- 回归：`HistGradientBoostingRegressor`
- 50mm 分类：`HistGradientBoostingClassifier`
- 100mm 分类：`HistGradientBoostingClassifier`

输出：

```text
outputs/rainfall_benchmark/ml/models/raincalib_regressor.joblib
outputs/rainfall_benchmark/ml/models/raincalib_event50.joblib
outputs/rainfall_benchmark/ml/models/raincalib_event100.joblib
outputs/rainfall_benchmark/ml/predictions/raincalib_test_predictions.parquet
outputs/rainfall_benchmark/ml/metrics/raincalib_metrics.csv
outputs/rainfall_benchmark/ml/reports/raincalib_report.md
```

评估指标：

- 回归：MAE、RMSE、Bias、Pearson r。
- 分类：CSI、POD、FAR、ETS、Precision、Recall、F1。
- 分 lead 统计。
- 分阈值统计。

验收标准：

- 订正后 GFS 在 test split 上 MAE 或 RMSE 优于原始 GFS。
- 50mm 事件 CSI 或 ETS 优于原始 GFS。
- 如果 100mm 事件过稀疏，报告中允许标注为探索性指标。

### 阶段8：RainCalib v2 轻量深度学习

该阶段在 v1 成功后执行。

可选模型：

- 小型 U-Net。
- ConvLSTM。
- Patch-based CNN。

显存策略：

- 使用区域裁剪后的网格。
- 使用 patch 训练。
- batch size 从 1-4 起步。
- 使用 mixed precision。
- 优先预测残差：`target_rain - gfs_rain`。
- 输出时将残差加回 GFS，并裁剪到非负。

环境建议：

- 不建议继续在混合依赖的 `pytorch-gpu` 环境中训练。
- 建议新建干净环境 `raincalib-gpu`。
- PyTorch、CUDA、torchvision、torchaudio 版本保持同一安装来源。

验收标准：

- GPU 能正常识别。
- 训练 1 epoch 不 OOM。
- 小样本 overfit 测试能收敛。
- test split 上至少一个核心指标优于原始 GFS 或 v1 baseline。

## 8. 测试计划

### 8.1 单元测试

新增或扩展测试：

```text
tests/test_gfs_download_plan.py
tests/test_data_gfs.py
tests/test_gfs_accumulations.py
tests/test_raincalib_features.py
tests/test_raincalib_splits.py
```

测试点：

- required steps 是否正确补齐。
- GFS 文件名解析是否正确。
- 非目标日期文件是否被过滤。
- `.idx` 文件是否被忽略。
- 24h 累积差分是否正确。
- 负降水差值是否被处理。
- train/val/test 是否无日期泄漏。

### 8.2 集成测试

建议命令：

```bash
PYTHONPATH=src python scripts/download_gfs_historical.py --config configs/gfs_rainfall_benchmark.yaml --start-date 2025-09-01 --end-date 2025-09-02 --dry-run
PYTHONPATH=src python scripts/check_gfs_coverage.py --config configs/gfs_rainfall_benchmark.yaml --sample-count 1
PYTHONPATH=src python -m ai_weather_rainfall.cli --config configs/gfs_rainfall_benchmark.yaml
PYTHONPATH=src python scripts/build_raincalib_features.py --config configs/raincalib.yaml
PYTHONPATH=src python scripts/train_raincalib_baseline.py --config configs/raincalib.yaml
PYTHONPATH=src python scripts/evaluate_raincalib.py --config configs/raincalib.yaml
```

### 8.3 回归测试

现有测试必须继续通过：

```bash
PYTHONPATH=src python -m pytest -q
```

现有 ECMWF benchmark 不应被破坏：

```bash
PYTHONPATH=src python -m ai_weather_rainfall.cli --config configs/rainfall_benchmark.yaml
```

## 9. 成果物清单

方案A完成后，应至少产生以下成果物：

```text
configs/gfs_rainfall_benchmark.yaml
configs/raincalib.yaml
scripts/download_gfs_historical.py
scripts/check_gfs_coverage.py
scripts/build_raincalib_features.py
scripts/train_raincalib_baseline.py
scripts/evaluate_raincalib.py
src/ai_weather_rainfall/data_gfs.py
outputs/rainfall_benchmark/reports/gfs_rainfall_benchmark_report.md
outputs/rainfall_benchmark/reports/model_comparison_report.md
outputs/rainfall_benchmark/ml/reports/raincalib_report.md
```

可选成果物：

```text
notebooks/02_gfs_imerg_benchmark.ipynb
notebooks/03_raincalib_modeling.ipynb
docs/raincalib_method_note.md
```

## 10. 风险与解决思路

### 风险1：GFS 历史数据下载体积较大

解决思路：

- 先只下载区域字段或必要变量。
- 优先使用 NOMADS/GDEX 支持的 subset 能力。
- 分日期、分 cycle、分 step 断点下载。
- manifest 记录状态，避免重复请求。

### 风险2：GFS 累积降水字段解释存在差异

解决思路：

- 先用 1-2 天样本检查 GRIB metadata。
- 明确 total precipitation 的 step range。
- 用 `lead - 24` 差分构造 24h 降水。
- 与 IMERG 区域均值做 sanity check。

### 风险3：强降水样本稀疏

解决思路：

- 保留全部 50mm/100mm 事件样本。
- 对非事件样本下采样。
- 报告中同时提供样本量、hits、misses、false alarms。
- 100mm 指标作为高不确定性探索指标。

### 风险4：12GB 显存限制深度学习模型规模

解决思路：

- v1 使用 CPU 友好的 sklearn baseline。
- v2 使用区域裁剪、patch、mixed precision 和小 batch。
- 训练残差而不是直接训练完整降水场。
- 不在本阶段训练全球基础模型。

### 风险5：多模型比较时段不一致

解决思路：

- 主报告明确区分：
  - GFS 2023-2025 完整历史 benchmark。
  - GFS/AIFS/IFS 2025 同窗口对照。
- 图表标题和表格列必须标明年份范围。

## 11. 推荐执行顺序

```text
1. 新增 GFS 配置和下载计划。
2. 小样本下载并确认 GFS GRIB 字段。
3. 实现 GFS 覆盖检查。
4. 实现 GFS 读取和 24h 累积。
5. 跑通 GFS 小样本 pipeline。
6. 下载完整 2023-2025 汛期。
7. 生成 GFS 完整 benchmark 报告。
8. 生成 GFS/AIFS/IFS 2025 同窗口对照报告。
9. 构建 RainCalib feature store。
10. 训练 RainCalib v1 baseline。
11. 评估订正收益。
12. 再决定是否进入 v2 深度学习模型。
```

## 12. 验收标准

方案A整体完成的验收标准：

- GFS 2023-2025 汛期下载覆盖检查通过。
- GFS benchmark 可一键运行并生成报告。
- 当前 ECMWF 2025 benchmark 仍可运行。
- GFS、AIFS、IFS 的 2025 同窗口对照报告可生成。
- RainCalib v1 生成 feature store、模型、预测和评估报告。
- 至少一个核心指标显示订正模型相对原始 GFS 有改善。
- README 中明确说明：
  - 主数据路线是 GFS Historical Archive + IMERG。
  - ECMWF Open Data 是免费可用窗口对照。
  - 项目不是官方预警服务。

## 13. 最小可交付版本

如果需要尽快形成阶段性成果，最小可交付版本可以只包含：

1. 2025-05-01 至 2025-09-30 的 GFS 下载和评估。
2. GFS/AIFS/IFS 2025 同窗口对照。
3. RainCalib v1 sklearn baseline。
4. 一份 `model_comparison_report.md` 和一份 `raincalib_report.md`。

这个版本不需要等待 2023-2024 全量下载完成，适合先验证方法、图表和论文/项目展示叙事。
