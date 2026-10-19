#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/hezm/klss}"
RUN="${RUN:-$ROOT/runs/cnn_single_56m_run1}"
CONFIG="${CONFIG:-$ROOT/config/run_cnn_single_56m.json}"
export CUDA_VISIBLE_DEVICES=6
export NR_CUDA_ARCH=sm_80
export NVCC=/usr/local/cuda/bin/nvcc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

active_pids="$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
if [ -n "$active_pids" ]; then
  echo "GPU 6 already has compute processes: $active_pids" >&2
  exit 1
fi
if [ -f "$RUN/config.json" ]; then
  echo "run already exists: $RUN; select a new RUN or resume explicitly" >&2
  exit 1
fi
source "$ROOT/env/bin/activate"
cd "$ROOT/code"
python3 -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'
mkdir -p "$RUN" "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/$(basename "$RUN").log") 2>&1
echo "$$" > "$RUN/train.pid"
exec python3 teacher/train_distill_single.py --config "$CONFIG" --out "$RUN" "$@"
