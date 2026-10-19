#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/hezm/klss
PREP="${1:-4}"
TEACHER="${2:-1}"
PAIR="${3:-0.35}"
POST="${4:-4}"
RUN="$ROOT/runs/bench_pipeline_p${PREP}_t${TEACHER}_post${POST}"

export CUDA_VISIBLE_DEVICES=6
export NR_CUDA_ARCH=sm_80
export NVCC=/usr/local/cuda/bin/nvcc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

mkdir -p "$RUN" "$ROOT/logs"
cd "$ROOT/code/teacher"
source "$ROOT/env/bin/activate"
exec > >(tee -a "$ROOT/logs/bench_pipeline_p${PREP}_t${TEACHER}_post${POST}.log") 2>&1

python3 train_distill.py \
  --config "$ROOT/config/run_v46_a100.json" \
  --data \
    "$ROOT/data/full/DIV2K_train_HR" \
    "$ROOT/data/full/Flickr2K" \
    "$ROOT/data/full/Full-Resolution" \
    "$ROOT/data/full/proxy_proc" \
    "$ROOT/data/full/rgb" \
    "$ROOT/data/full/gameplay" \
    "$ROOT/data/full/portraits" \
    "$ROOT/data/full/frames_2k" \
  --out "$RUN" \
  --steps 30 \
  --log-every 10 \
  --ckpt-every 0 \
  --vis-every 0 \
  --comet-disabled \
  --prep-workers "$PREP" \
  --post-workers "$POST" \
  --teacher-workers "$TEACHER" \
  --pair-fraction "$PAIR"
