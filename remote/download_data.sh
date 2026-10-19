#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/hezm/klss
DEST="$ROOT/data/downloads"

mkdir -p "$DEST" "$ROOT/data/raw" "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/download_data.log") 2>&1
cd "$DEST"

aria2c -c -x 8 -s 8 -k 1M --file-allocation=none \
  -o DIV2K_train_HR.zip \
  https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
aria2c -c -x 8 -s 8 -k 1M --file-allocation=none \
  -o DAVIS-2017-trainval-480p.zip \
  https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip

unzip -q -o DIV2K_train_HR.zip -d "$ROOT/data/raw"
unzip -q -o DAVIS-2017-trainval-480p.zip -d "$ROOT/data/raw"

echo "data download and extraction complete"
find "$ROOT/data/raw/DIV2K_train_HR" -type f | wc -l
find "$ROOT/data/raw/DAVIS/JPEGImages/480p" -type f | wc -l
