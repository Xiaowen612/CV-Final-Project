#!/usr/bin/env bash
set -euo pipefail
cd /home/smbu/lxw/genimage_experiment
export OMP_NUM_THREADS=4
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
.venv/bin/python -m pip freeze > environment_server.txt
.venv/bin/python -m pytest -q > tests.log 2>&1
if [ ! -f manifests_pilot/audit.json ]; then
  .venv/bin/python -u run.py prepare --config config_server_pilot.yaml
fi
.venv/bin/python -u server_runner.py --config config_server_pilot.yaml
