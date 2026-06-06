# AI Weather Extreme Rainfall Benchmark

This project evaluates and post-processes heavy-rainfall forecasts over the middle and lower Yangtze River region.

It contains an end-to-end workflow for downloading public meteorological data, computing 24-hour accumulated rainfall, benchmarking numerical weather prediction models against NASA GPM IMERG observations, and training RainCalib machine-learning models to correct GFS rainfall forecasts. This is a research and decision-support project, not an official warning service.

## Highlights

- Benchmarks GFS, ECMWF AIFS, and ECMWF IFS rainfall forecasts against IMERG observations.
- Range-downloads only the required GFS APCP GRIB messages from public NOAA archives.
- Scores both continuous rainfall skill and 50mm/100mm heavy-rainfall events.
- Trains RainCalib v1 sklearn and RainCalib v2.1 PyTorch U-Net post-processing models.
- Includes diagnostics by lead time, cycle, month, rainfall intensity, and grid cell.

## Repository Contents

```text
configs/   Benchmark and RainCalib configuration files.
docs/      Project plans, experiment summaries, and result writeups.
scripts/   Download, coverage check, feature build, train, evaluate, and diagnose entry points.
src/       Python package for data loading, regridding, metrics, reports, and ML utilities.
tests/     Unit and smoke tests.
```

Raw data, generated metrics, trained model weights, local credentials, and notebook execution outputs are intentionally excluded from Git. See [Data And Artifacts](#data-and-artifacts).

## Quick Start

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m ai_weather_rainfall.cli --config configs/rainfall_benchmark.yaml
```

Run the GFS Plan A benchmark with:

```bash
PYTHONPATH=src python -m ai_weather_rainfall.cli --config configs/gfs_rainfall_benchmark.yaml
```

## Data Layout

Place forecast and observation data under `outputs/rainfall_benchmark/raw/`:

- `ecmwf/aifs/` for AIFS forecast GRIB or NetCDF files.
- `ecmwf/ifs/` for IFS forecast GRIB or NetCDF files.
- `gfs/` for GFS forecast GRIB files containing cumulative APCP records.
- `imerg/` for IMERG precipitation files.

The default configuration uses the freely available ECMWF Open Data window that is usable for both AIFS and IFS in this project: 2025-05-01 through 2025-09-30. For a complete 2023-2025 historical benchmark, use ECMWF Archive/MARS or another licensed archive.

The Plan A GFS configuration uses the NOAA GFS public AWS archive for 2023-2025 flood seasons. The GFS downloader reads each `.idx` file and range-downloads only the cumulative `APCP:surface:0-N day acc fcst` GRIB messages required for 24-hour rainfall windows. This avoids downloading full 500 MB global GRIB files for every step.

## Batch Downloads

The configured benchmark period selects 153 IMERG days and 612 ECMWF forecast requests. Run a dry run first:

```bash
PYTHONPATH=src python scripts/download_imerg_gesdisc.py --dry-run
PYTHONPATH=src python scripts/download_ecmwf_open_data.py --dry-run
PYTHONPATH=src python scripts/download_gfs_historical.py --config configs/gfs_rainfall_benchmark.yaml --dry-run
```

For GES DISC, put Earthdata credentials in `token.txt` or the `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` environment variables, then run:

```bash
PYTHONPATH=src python scripts/download_imerg_gesdisc.py
```

For ECMWF Open Data, the downloader automatically requests every configured lead time plus the prior accumulation-window step needed for 24-hour totals. For the default leads, this is:

```text
24, 48, 72, 96, 120, 144, 168, 216, 240, 336, 360
```

Run the full ECMWF batch with:

```bash
PYTHONPATH=src python scripts/download_ecmwf_open_data.py --source aws --retries 3
```

Run a GFS smoke download first:

```bash
PYTHONPATH=src python scripts/download_gfs_historical.py --config configs/gfs_rainfall_benchmark.yaml --date 2025-09-01 --cycles 00 12 --retries 3
PYTHONPATH=src python scripts/check_gfs_coverage.py --config configs/gfs_rainfall_benchmark.yaml --date 2025-09-01 --sample-count 1
```

Run the full GFS Plan A batch after the smoke check passes:

```bash
PYTHONPATH=src python scripts/download_gfs_historical.py --config configs/gfs_rainfall_benchmark.yaml --retries 3
PYTHONPATH=src python scripts/check_gfs_coverage.py --config configs/gfs_rainfall_benchmark.yaml --sample-count 3
PYTHONPATH=src python -m ai_weather_rainfall.cli --config configs/gfs_rainfall_benchmark.yaml
```

After both the ECMWF 2025 benchmark and the GFS benchmark have been run, generate the same-window comparison report with:

```bash
PYTHONPATH=src python scripts/create_model_comparison_report.py
```

## RainCalib Baseline

RainCalib trains a regional post-processing model that corrects raw GFS rainfall toward IMERG observations.

Build the feature store, train the sklearn baseline, and evaluate the test split:

```bash
PYTHONPATH=src python scripts/build_raincalib_features.py --config configs/raincalib.yaml --overwrite
PYTHONPATH=src python scripts/train_raincalib_baseline.py --config configs/raincalib.yaml
PYTHONPATH=src python scripts/evaluate_raincalib.py --config configs/raincalib.yaml --split test
```

Outputs are written under `outputs/rainfall_benchmark/ml/`. The summary report is:

```text
outputs/rainfall_benchmark/ml/reports/raincalib_report.md
```

Check GPU readiness for the optional v2 deep-learning branch with:

```bash
python scripts/check_raincalib_gpu.py
```

The prepared GPU conda environment is `raincalib-gpu`:

```bash
conda activate raincalib-gpu
python scripts/check_raincalib_gpu.py
```

## RainCalib v2.1 U-Net

RainCalib v2.1 adds a small GPU-trained U-Net that predicts an `IMERG - GFS` residual rainfall field and auxiliary 50mm/100mm event probabilities. Build the tensor dataset, train, evaluate, and diagnose with:

```bash
conda activate raincalib-gpu
PYTHONPATH=src python scripts/build_raincalib_tensor_dataset.py --config configs/raincalib_unet.yaml --overwrite
PYTHONPATH=src python scripts/train_raincalib_unet.py --config configs/raincalib_unet.yaml
PYTHONPATH=src python scripts/evaluate_raincalib_unet.py --config configs/raincalib_unet.yaml --split test --tune-split val
PYTHONPATH=src python scripts/diagnose_raincalib_unet_errors.py --config configs/raincalib_unet.yaml --split test
```

Current test split continuous metrics:

| Model | MAE | RMSE | Bias | Pearson r |
|---|---:|---:|---:|---:|
| Raw GFS | 8.726 | 17.209 | -0.600 | 0.048 |
| RainCalib v1 sklearn | 7.785 | 13.363 | -0.335 | 0.132 |
| RainCalib v2.1 U-Net | 7.049 | 13.084 | -0.751 | 0.241 |

Current event-head metrics:

| Threshold | CSI | POD | FAR | ETS |
|---:|---:|---:|---:|---:|
| 50 mm | 0.060 | 0.489 | 0.936 | 0.043 |
| 100 mm | 0.011 | 0.043 | 0.985 | 0.009 |

The v2.1 event head improves heavy-rain detection, while the continuous residual field remains conservative for 50mm+ rainfall. The tracked summary is `docs/raincalib_v2_results_summary.md`; full generated reports are written under `outputs/rainfall_benchmark/ml/reports/` and are not committed.

ECMWF Open Data mirrors are not a complete historical archive. Older dates or long lead steps can return 404; the ECMWF downloader preflights the remote index files and records those requests as `unavailable` instead of retrying them forever. Use ECMWF Archive/MARS or another licensed archive for a complete 2023-2025 historical benchmark.

Both downloaders are restartable and skip complete files. To download a smaller slice while testing:

```bash
PYTHONPATH=src python scripts/download_imerg_gesdisc.py --start-date 2025-09-01 --end-date 2025-09-07
PYTHONPATH=src python scripts/download_ecmwf_open_data.py --start-date 2025-09-01 --end-date 2025-09-07 --source aws
PYTHONPATH=src python scripts/download_gfs_historical.py --config configs/gfs_rainfall_benchmark.yaml --start-date 2025-09-01 --end-date 2025-09-07
```

## Data And Artifacts

This repository tracks source code, configuration, tests, notebooks, and Markdown summaries only.

Do not commit:

- Raw forecast or observation files.
- Generated NetCDF, GRIB, HDF5, Parquet, NPZ, CSV metric bundles, or plots under `outputs/`.
- Trained model weights such as `.pt` or `.joblib`.
- Credentials such as `token.txt`, `.earthdata.netrc`, or `.urs_cookies`.

The `.gitignore` file is configured to ignore these local data and artifact files. Reproduce them locally with the commands above.

## Generated Outputs

The pipeline writes metrics, processed NetCDF files, figures, model predictions, checkpoints, and Markdown reports under `outputs/rainfall_benchmark/`. These files are local experiment artifacts and are intentionally not pushed to the remote repository.

## Acceptance Criteria

- The CLI can run from `configs/rainfall_benchmark.yaml`.
- The package can load AIFS, IFS, and IMERG files from configured directories.
- The package can load GFS cumulative APCP files from `configs/gfs_rainfall_benchmark.yaml`.
- The benchmark computes 24-hour rainfall accumulations for configured lead times.
- The benchmark scores 50 mm and 100 mm 24-hour rainfall events.
- The output report states that the project is not an official warning service.
- The 2025 flood season is the default AIFS-vs-IFS comparison period for free ECMWF Open Data.
- The 2023-2025 flood seasons are the default GFS-vs-IMERG Plan A comparison period.
- The benchmark date filters prevent out-of-period local files from being scored.
