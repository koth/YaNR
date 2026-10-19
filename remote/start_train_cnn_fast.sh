#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/data2/hezm/klss}"
export RUN="${RUN:-$ROOT/runs/cnn_unet_fast_run1}"
export CONFIG="${CONFIG:-$ROOT/code/remote/run_cnn_unet_fast.json}"
exec bash "$ROOT/code/remote/start_train_single.sh" "$@"
