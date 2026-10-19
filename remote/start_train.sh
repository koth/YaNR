#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/hezm/klss
RUN="$ROOT/runs/v46final_a100_50k"

export CUDA_VISIBLE_DEVICES=6
export NR_CUDA_ARCH=sm_80
export NVCC=/usr/local/cuda/bin/nvcc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [ "${CUDA_VISIBLE_DEVICES}" != "6" ]; then
  echo "refusing to start: CUDA_VISIBLE_DEVICES must be exactly 6" >&2
  exit 1
fi

if ! nvidia-smi -i 6 --query-gpu=index,name --format=csv,noheader | grep -q '^6,'; then
  echo "refusing to start: physical GPU 6 is not visible" >&2
  exit 1
fi

active_pids="$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
if [ -n "$active_pids" ]; then
  echo "refusing to start: GPU 6 already has compute processes: $active_pids" >&2
  exit 1
fi

mkdir -p "$RUN" "$ROOT/logs"
cd "$ROOT/code/teacher"

source "$ROOT/env/bin/activate"
exec > >(tee -a "$ROOT/logs/v46final_a100_50k.log") 2>&1

python3 - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is not available'
assert torch.cuda.device_count() == 1, f'expected exactly one visible GPU, got {torch.cuda.device_count()}'
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))
PY

echo "starting training on physical GPU 6"
echo "config: $ROOT/config/run_v46_a100.json"
echo "run:    $RUN"

echo "$$" > "$RUN/train.pid"

exec python3 train_distill.py \
  --config "$ROOT/config/run_v46_a100.json" \
  --data \
    "$ROOT/data/train/proxy_proc" \
    "$ROOT/data/raw/DIV2K_train_HR" \
    "$ROOT/data/raw/DAVIS/JPEGImages/480p" \
  --out "$RUN" \
  "$@"
