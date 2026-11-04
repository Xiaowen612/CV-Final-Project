#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export OMP_NUM_THREADS=4
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

if [ -f REPEATS_AND_ABLATIONS_COMPLETE.json ] && [ -f results_verification.json ] && [ -f EXPERIMENT_REPORT_ZH.md ]; then
  .venv/bin/python -c 'import json; r=json.load(open("results_verification.json")); assert r["status"]=="passed" and r["trained_models"]==24'
  printf '%s\n' '24 个模型的实验已完成，无需重复训练。' '中文报告：EXPERIMENT_REPORT_ZH.md' '图表与汇总：runs_pilot/report/' '模型权重与逐图预测：runs_pilot/<method>/seed_<N>/'
  exit 0
fi

bash start_server.sh
.venv/bin/python -u extend_repeats.py
.venv/bin/python verify_outputs.py
.venv/bin/python check_adaptation.py
.venv/bin/python write_final_report.py
.venv/bin/python export_results.py
printf '%s\n' '实验完成。请查看 EXPERIMENT_REPORT_ZH.md 和 runs_pilot/report/。'
