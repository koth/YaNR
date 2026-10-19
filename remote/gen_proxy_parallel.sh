#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/hezm/klss
cd "$ROOT/code/teacher"
source "$ROOT/env/bin/activate"
mkdir -p "$ROOT/data/proc" "$ROOT/data/train/proxy_proc" "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/gen_proxy.log") 2>&1

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  start=$((shard * 25))
  python3 gen_sequences.py \
    -o "$ROOT/data/proc" \
    --count 25 \
    --frames 8 \
    --size 1920x1080 \
    --seed 20261007 \
    --start-index "$start" \
    > "$ROOT/logs/gen_seq_${shard}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

mapfile -t all_seqs < <(find "$ROOT/data/proc" -mindepth 1 -maxdepth 1 -type d -name 'seq-*' | sort)
if [ "${#all_seqs[@]}" -ne 200 ]; then
  echo "expected 200 sequence directories, found ${#all_seqs[@]}" >&2
  exit 1
fi

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  start=$((shard * 25))
  out="$ROOT/data/train/proxy_proc/shard$shard"
  mkdir -p "$out"
  python3 degrade.py \
    "${all_seqs[@]:start:25}" \
    -o "$out" \
    --seed 4242 \
    --mode random \
    --ordinal-offset "$start" \
    > "$ROOT/logs/degrade_${shard}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "proxy_proc generation complete"
find "$ROOT/data/train/proxy_proc" -type f | wc -l
