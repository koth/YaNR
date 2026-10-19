#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/hezm/klss
RUN="$ROOT/runs/bench_compile"

export CUDA_VISIBLE_DEVICES=6
export NR_CUDA_ARCH=sm_80
export NVCC=/usr/local/cuda/bin/nvcc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$ROOT/torchinductor_compile_v3"
export TRITON_CACHE_DIR="$ROOT/triton_compile_v3"

mkdir -p "$RUN" "$ROOT/logs"
cd "$ROOT/code/teacher"
source "$ROOT/env/bin/activate"
exec > >(tee -a "$ROOT/logs/bench_compile.log") 2>&1

python3 train_distill.py \
  --config "$ROOT/config/run_v46_a100.json" \
  --data \
    "$ROOT/data/train/proxy_proc" \
    "$ROOT/data/raw/DIV2K_train_HR" \
    "$ROOT/data/raw/DAVIS/JPEGImages/480p" \
  --out "$RUN" \
  --steps 20 \
  --log-every 10 \
  --ckpt-every 0 \
  --vis-every 0 \
  --student-compile 1 \
  --comet-disabled
