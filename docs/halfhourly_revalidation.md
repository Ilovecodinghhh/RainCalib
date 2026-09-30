# 方案②：半小时IMERG重建与复核

## 当前状态

全量下载及后续训练流水线已启动，尚未完成，不能提供方案②性能数字。需要22,104个半小时区域子集，对应既有7110个预报场、918个不同的24小时窗口，覆盖2023—2025汛期并延伸至每年10月1日中午。出现HTTP 503后已降低为4并发，并增加重试；OPeNDAP持续返回5xx时会尝试原始HDF5下载并在本地裁剪；该备用路径已与真实子集逐值比对一致。每份文件验证通过才纳入缓存，任何阶段失败都会停止后续训练。

机器可读状态：outputs/rainfall_benchmark/experiments_halfhourly/pipeline_status.json。
下载计数：outputs/rainfall_benchmark/raw/imerg_halfhourly_late_v07/progress.json。
每个阶段的日志保存在experiments_halfhourly目录。

## 新发现：旧格点标签的维度错误

真实数据检查发现：IMERG日观测数组为经度×纬度，插值后维度顺序未改变，而旧特征构建将它与纬度×经度的GFS分别直接展平，导致标签错位。2023-05-01 00 UTC实测证实，旧标签与错误顺序展平的观测完全一致。因此，原混合时次实验和刚完成的00 UTC实验性能都撤回，仅保留历史追溯。

已修正插值输出维度顺序、特征展平顺序及相关系数计算的维度对齐。新流程仅从旧表读取预报场清单，GFS格点和观测标签均从原始文件重建，避免继承错误的像元掩码。旧数据和模型文件未覆盖。

## 观测与窗口规则

- 产品与旧日观测保持一致：IMERG Late V07，GPM_3IMERGHHL.07；不混用Early或Final。
- 通过GES DISC OPeNDAP获取区域降水子集，保留周边缓冲区用于插值。
- 窗口起点为起报UTC时间 + 时效 − 24小时；终点不包含在累加内。
- 每个窗口必须有48个唯一、连续的半小时时段，每份文件的time_bnds必须吻合。
- 降水率单位必须为mm/hr，逐份乘0.5小时再求和。任何像元出现缺测或负值，该窗口该像元保持缺测，不能按零补齐。
- 删除跨训练/验证边界的中午窗口，防止相邻数据集共享12小时观测。既有日期块bootstrap保留用于可比性，但相邻日期窗口仍可能相关，不能将日期视为完全独立。

## 已完成验证

- 真实48份半小时数据重建00 UTC自然日，与旧日产品在同一观测网格的平均绝对差约0.000013 mm。
- 真实00/12 UTC两场从原始GRIB重建完成，共1426格点；时间窗口分别为2023-05-01 00:00至05-02 00:00、05-01 12:00至05-02 12:00。
- 完整测试59项通过，随后新增相关系数维度回归检查，相关12项全部通过（共60项已验证）。单元测试覆盖跨年窗口、48份完整性、重复/错位时段、缺测传播、单位、文件时间边界、非方形经纬维度及数据集边界清除。

## 恢复运行

在仓库根目录使用现有环境执行以下命令；单实例锁会拒绝重复启动：

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/run_halfhourly_revalidation.py --gpu-python C:/Users/admin/miniforge3/envs/raincalib-gpu/python.exe --workers 4
```

顺序为下载并校验全部子集、重建特征、生成张量、数据审计、统计基线、两折五种子U-Net、配对bootstrap报告。新配置使用half-hourly专用目录ml_halfhourly和experiments_halfhourly，方案①目录保持不变。完成报告路径为docs/halfhourly_validation_results.md。

## 官方资料

[GES DISC Late V07目录](https://gpm1.gesdisc.eosdis.nasa.gov/opendap/hyrax/GPM_L3/GPM_3IMERGHHL.07/contents.html)；[NASA IMERG技术说明](https://gpm.nasa.gov/sites/default/files/2023-07/IMERG_TechnicalDocumentation_final_230713.pdf)。
