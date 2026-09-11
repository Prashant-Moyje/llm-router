#!/usr/bin/env bash
set -euo pipefail
PROVIDER="${1:-mock}"
CONFIG="${2:-configs/default.yaml}"
export PYTHONPATH=src
python scripts/01_build_dataset.py --config "$CONFIG" --sources synthetic --limit 600
python scripts/02_run_offline_eval.py --config "$CONFIG" --provider "$PROVIDER" --workers 8
python scripts/03_train_router.py --config "$CONFIG"
python scripts/04_simulate.py --config "$CONFIG" --quality-floor-drop 0.02
python scripts/05_plot.py --config "$CONFIG"
python scripts/06_bench_latency.py --config "$CONFIG"
