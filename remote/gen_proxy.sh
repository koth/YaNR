#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/hezm/klss

cd "$ROOT/code/teacher"
source "$ROOT/env/bin/activate"
mkdir -p "$ROOT/data/proc" "$ROOT/data/train/proxy_proc" "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/gen_proxy.log") 2>&1

python3 gen_sequences.py \
  -o "$ROOT/data/proc" \
  --count 200 \
  --frames 8 \
  --size 1920x1080 \
  --seed 20261007

python3 degrade.py \
  "$ROOT/data/proc" \
  -o "$ROOT/data/train/proxy_proc" \
  --seed 4242 \
  --mode random

echo "proxy_proc generation complete"
find "$ROOT/data/train/proxy_proc" -type f | wc -l
