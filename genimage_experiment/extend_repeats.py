"""Extend the fixed pilot protocol with predeclared seeds and constraint ablations."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import yaml

deadline = time.monotonic()+1800
while not Path('EXPERIMENT_COMPLETE.json').exists():
    for status in Path('logs').glob('gpu*_status.json'):
        if json.loads(status.read_text()).get('state') == 'failed':
            raise RuntimeError('Initial pilot failed; inspect logs before extending.')
    if time.monotonic() > deadline:
        raise TimeoutError('Initial pilot did not finish within 30 minutes.')
    time.sleep(5)

cfg = yaml.safe_load(Path('config_server_pilot.yaml').read_text())
cfg['seeds'] = [42, 43, 44]
Path('config_server_repeated.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
subprocess.run([sys.executable, '-u', 'server_runner.py', '--config', 'config_server_repeated.yaml'], check=True)

# Predeclared constraint ablations. Same manifest, backbone, LR, epoch budget,
# seed and augmentation as the main non-augmented Effort arm.
for method in ['effort_no_orth', 'effort_no_energy', 'effort_no_constraints']:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4')
    checkpoint = Path(cfg['run_dir']) / method / 'seed_42/best.pt'
    with Path(f'logs/{method}_42.log').open('a') as log:
        if not (checkpoint.parent / 'run.json').exists():
            subprocess.run([sys.executable, '-u', 'run.py', 'train', '--config', 'config_server_repeated.yaml',
                            '--method', method, '--seed', '42'], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, '-u', 'run.py', 'evaluate', '--config', 'config_server_repeated.yaml',
                        '--checkpoint', str(checkpoint)], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
subprocess.run([sys.executable, 'run.py', 'report', '--config', 'config_server_repeated.yaml'], check=True)
Path('REPEATS_AND_ABLATIONS_COMPLETE.json').write_text(json.dumps(dict(
    methods=cfg['methods'], seeds=cfg['seeds'], ablations=['effort_no_orth','effort_no_energy','effort_no_constraints'],
    ablation_seeds=[42], total_trained_models=24, config='config_server_repeated.yaml'), indent=2))
