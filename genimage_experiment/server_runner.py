"""Run independent experiment groups on two GPUs and persist status/logs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


def worker(config_path, methods, gpu):
    cfg = yaml.safe_load(Path(config_path).read_text())
    logdir = Path('logs'); logdir.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               HF_ENDPOINT='https://hf-mirror.com', TOKENIZERS_PARALLELISM='false')
    status_path = logdir / f'gpu{gpu}_status.json'
    completed = []
    for method in methods:
        for seed in cfg['seeds']:
            status_path.write_text(json.dumps(dict(state='running', method=method, seed=seed, completed=completed), indent=2))
            run = Path(cfg['run_dir']) / method / f'seed_{seed}'
            checkpoint = run / 'best.pt'
            commands = []
            if not (run / 'run.json').exists():
                commands.append([sys.executable, '-u', 'run.py', 'train', '--config', config_path, '--method', method, '--seed', str(seed)])
            commands.append([sys.executable, '-u', 'run.py', 'evaluate', '--config', config_path, '--checkpoint', str(checkpoint)])
            with (logdir / f'{method}_{seed}.log').open('a') as log:
                for command in commands:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        status_path.write_text(json.dumps(dict(state='failed', method=method, seed=seed, returncode=result.returncode, completed=completed), indent=2))
                        return result.returncode
            completed.append(f'{method}/seed_{seed}')
    status_path.write_text(json.dumps(dict(state='complete', completed=completed), indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_server_pilot.yaml')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--methods', nargs='+')
    args = parser.parse_args()
    if args.gpu is not None:
        sys.exit(worker(args.config, args.methods, args.gpu))
    cfg = yaml.safe_load(Path(args.config).read_text())
    groups = [cfg['methods'][:3], cfg['methods'][3:]]
    processes = []
    for gpu, methods in enumerate(groups):
        if methods:
            processes.append(subprocess.Popen([sys.executable, '-u', __file__, '--config', args.config,
                                               '--gpu', str(gpu), '--methods', *methods]))
    codes = [p.wait() for p in processes]
    if any(codes):
        raise RuntimeError(f'One or more experiment groups failed: {codes}; see logs/')
    subprocess.run([sys.executable, 'run.py', 'report', '--config', args.config], check=True)
    Path('EXPERIMENT_COMPLETE.json').write_text(json.dumps(dict(config=args.config, methods=cfg['methods'], seeds=cfg['seeds'],
                                                               completed_unix_time=time.time()), indent=2))


if __name__ == '__main__':
    main()
