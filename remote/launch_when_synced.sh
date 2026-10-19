#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/koth/klss
LOG="$ROOT/remote/sync_full_data.log"
STATUS="$ROOT/remote/full_run_status.log"
HOST=hezm@10.2.0.16
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

exec > >(tee -a "$STATUS") 2>&1

until grep -q "all roots synced" "$LOG" 2>/dev/null; do
  if grep -q "FAILED" "$LOG" 2>/dev/null; then
    echo "$(date -Is) data sync failed; not launching"
    exit 1
  fi
  sleep 60
done

until $SSH "$HOST" 'echo ssh_ok'; do
  echo "$(date -Is) waiting for ssh/vpn..."
  sleep 60
done

rsync -az -e "$SSH" "$ROOT/remote/start_train_full.sh" "$HOST:/data2/hezm/klss/scripts/"
rsync -az -e "$SSH" "$ROOT/remote/run_v46_a100.json" "$HOST:/data2/hezm/klss/config/"
rsync -az -e "$SSH" "$ROOT/remote/CONSTRAINTS.md" "$HOST:/data2/hezm/klss/"

$SSH "$HOST" '
  if [ -f /data2/hezm/klss/runs/v46final_a100_50k/train.pid ]; then
    kill "$(cat /data2/hezm/klss/runs/v46final_a100_50k/train.pid)" 2>/dev/null || true
  fi
  tmux kill-session -t klss-train 2>/dev/null || true
  sleep 5
  nvidia-smi -i 6 --query-compute-apps=pid,used_memory --format=csv,noheader
'

$SSH "$HOST" 'tmux new-session -d -s klss-train "bash /data2/hezm/klss/scripts/start_train_full.sh"'

echo "$(date -Is) full-data training launched"

for _ in $(seq 1 40); do
  url="$($SSH "$HOST" "grep -m1 '^comet: https://' /data2/hezm/klss/logs/v47_full_a100_50k.log 2>/dev/null" || true)"
  if [ -n "$url" ]; then
    echo "$url" >> "$STATUS"
    break
  fi
  sleep 30
done

$SSH "$HOST" "grep -m1 '^train:' /data2/hezm/klss/logs/v47_full_a100_50k.log 2>/dev/null" || true
