"""Resume download, rebuild exact-window labels, then run the frozen experiments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('outputs/rainfall_benchmark/experiments_halfhourly')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-python', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock_handle = (ROOT / 'pipeline.lock').open('a+b')
    lock_handle.seek(0)
    try:
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit('A half-hourly pipeline is already running')
    env = dict(os.environ, PYTHONPATH='src', PYTHONUNBUFFERED='1')
    cfg = 'configs/raincalib_experiments_halfhourly.yaml'
    unet = 'configs/raincalib_unet_halfhourly.yaml'
    features = Path('outputs/rainfall_benchmark/ml_halfhourly/features/raincalib_features.parquet')
    tensor = Path('outputs/rainfall_benchmark/ml_halfhourly/tensors/raincalib_tensor_dataset.npz')
    stages = [
        ('download', sys.executable, ['scripts/download_imerg_halfhourly.py', '--workers', str(args.workers)], None),
        ('features', sys.executable, ['scripts/build_halfhourly_features.py'], features),
        ('tensors', str(args.gpu_python), ['scripts/build_raincalib_tensor_dataset.py', '--config', unet], tensor),
        ('audit', sys.executable, ['scripts/audit_experiment_data.py', '--features', str(features), '--output-dir', str(ROOT/'audit')], None),
        ('statistical', sys.executable, ['scripts/run_statistical_validation.py', '--config', cfg], None),
        ('unet', str(args.gpu_python), ['scripts/run_unet_experiment_matrix.py', '--experiment-config', cfg, '--unet-config', unet], None),
        ('report', sys.executable, ['scripts/create_experimental_acceptance_report.py', '--config', cfg, '--output', 'docs/halfhourly_validation_results.md'], None),
    ]
    status = dict(state='running', stage=None, updated_utc=None, pid=os.getpid())
    def save():
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        (ROOT/'pipeline_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    for name, python, command, artifact in stages:
        status.update(state='running', stage=name);save()
        if artifact is not None and artifact.exists():
            continue
        with (ROOT/(name+'.log')).open('a', encoding='utf-8') as log:
            result = subprocess.run([python, '-u', *command], env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state='failed', returncode=result.returncode);save()
            raise SystemExit(result.returncode)
    status.update(state='complete');save()


if __name__ == '__main__':
    main()
