#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/hezm/klss
TF32="${1:-0}"
STEPS="${2:-30}"
LOG_EVERY="${3:-10}"
RUN="$ROOT/runs/bench_tf32_${TF32}_${STEPS}"

export CUDA_VISIBLE_DEVICES=6
export NR_CUDA_ARCH=sm_80
export NVCC=/usr/local/cuda/bin/nvcc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

mkdir -p "$RUN" "$ROOT/logs"
cd "$ROOT/code/teacher"
source "$ROOT/env/bin/activate"
exec > >(tee -a "$ROOT/logs/bench_tf32_${TF32}_${STEPS}.log") 2>&1

python3 train_distill.py \
  --config "$ROOT/config/bench_v46.json" \
  --data \
    "$ROOT/data/train/proxy_proc" \
    "$ROOT/data/raw/DIV2K_train_HR" \
    "$ROOT/data/raw/DAVIS/JPEGImages/480p" \
  --out "$RUN" \
  --teacher-workers 1 \
  --teacher-batch-rows 1277952 \
  --student-precision fp32 \
  --student-compile 0 \
  --student-tf32 "$TF32" \
  --steps "$STEPS" \
  --log-every "$LOG_EVERY"
