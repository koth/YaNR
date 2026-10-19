#!/usr/bin/env bash
set -uo pipefail

SRC=/f/yanr/rs
ROOT=/data2/hezm/klss
DEST="$ROOT/data/full"
LOG=/home/koth/klss/remote/sync_full_data.log
HOST=hezm@10.2.0.16

exec > >(tee -a "$LOG") 2>&1

SSH_CMD="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

wait_remote() {
  until ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "mkdir -p '$DEST'"; do
    echo "$(date -Is) waiting for ssh/vpn..."
    sleep 60
  done
}

sync_root() {
  local name=$1
  shift
  local attempt
  for attempt in $(seq 1 30); do
    echo "$(date -Is) sync $name attempt $attempt"
    if rsync -a --partial --append-verify -e "$SSH_CMD" "$@" "$SRC/$name" "$HOST:$DEST/"; then
      echo "$(date -Is) done $name"
      return 0
    fi
    sleep 60
  done
  echo "$(date -Is) FAILED $name"
  return 1
}

wait_remote

# Small/medium roots first, then the 8.4G gameplay corpus.
sync_root DIV2K_train_HR
sync_root Flickr2K
sync_root Full-Resolution
sync_root proxy_proc
sync_root rgb
sync_root portraits
sync_root frames_2k -L
sync_root gameplay

echo "$(date -Is) all roots synced"
ssh -o BatchMode=yes "$HOST" "
  for d in DIV2K_train_HR Flickr2K Full-Resolution proxy_proc rgb portraits frames_2k gameplay; do
    printf '%s ' \"\$d\"
    find '$DEST/'\"\$d\" -type f \\( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \\) | wc -l
  done
"
