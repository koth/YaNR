#!/usr/bin/env bash
# tools/fetch_data.sh — 蒸馏训练数据下载(openspec fast-student-distillation, task 4.1)
#
# 用法:
#   bash tools/fetch_data.sh [目标目录]          # 默认 ~/klss/data/raw
#
# 下载内容:
#   DIV2K_train_HR.zip                     3.5 GB  800 张 2K 干净图(蒸馏主数据)
#   DAVIS-2017-trainval-Full-Resolution.zip 2.9 GB  时序序列原分辨率帧(时序训练)
#   DAVIS-2017-trainval-480p.zip            833 MB  (可选,快速测试用)
#
# 注: DIV2K_valid_HR.zip 官方镜像当前 404,验证集改为从 train 划出 held-out,
#     不阻塞下载(见 manifest.txt 备注)。
#
# 服务器网络走代理:  PROXY=http://192.168.3.127:7897 bash tools/fetch_data.sh
# 断点续传: 重复运行自动 -C - 续传;每项完成后记录 sha256/字节数到 manifest.txt 并打 .done 标记。

set -euo pipefail

DEST=${1:-"$HOME/klss/data/raw"}
PROXY=${PROXY:-}
CURL=(curl -L --retry 5 --retry-delay 5 --connect-timeout 20 -C -)
if [ -n "$PROXY" ]; then CURL+=(-x "$PROXY"); fi

BASE_DIV2K=https://data.vision.ee.ethz.ch/cvl/DIV2K
BASE_DAVIS=https://data.vision.ee.ethz.ch/csergi/share/davis

# name | url | required(yes/no)
ITEMS=(
  "DIV2K_train_HR.zip|$BASE_DIV2K/DIV2K_train_HR.zip|yes"
  "DAVIS-2017-trainval-Full-Resolution.zip|$BASE_DAVIS/DAVIS-2017-trainval-Full-Resolution.zip|yes"
  "DAVIS-2017-trainval-480p.zip|$BASE_DAVIS/DAVIS-2017-trainval-480p.zip|no"
)

mkdir -p "$DEST"
MANIFEST="$DEST/manifest.txt"
[ -f "$MANIFEST" ] || printf '# sha256  bytes  name  url  finished-at\n' > "$MANIFEST"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

for item in "${ITEMS[@]}"; do
  IFS='|' read -r name url required <<< "$item"
  done_mark="$DEST/.done-$name"
  if [ -f "$done_mark" ]; then
    echo "[skip] $name (已完成)"
    continue
  fi
  echo "[get ] $name  <-  $url"
  if ! "${CURL[@]}" -o "$DEST/$name.part" "$url"; then
    if [ "$required" = yes ]; then
      echo "[fail] $name 下载失败(必需项)" >&2
      exit 1
    fi
    echo "[warn] $name 下载失败(可选项,跳过)"
    rm -f "$DEST/$name.part"
    continue
  fi
  mv "$DEST/$name.part" "$DEST/$name"
  bytes=$(wc -c < "$DEST/$name" | tr -d ' ')
  hash=$(sha256_of "$DEST/$name")
  echo "[unzp] $name"
  unzip -q -n "$DEST/$name" -d "$DEST"
  printf '%s  %s  %s  %s  %s\n' "$hash" "$bytes" "$name" "$url" "$(date -u +%FT%TZ)" >> "$MANIFEST"
  touch "$done_mark"
  echo "[done] $name  sha256=$hash"
done

echo "全部完成。manifest: $MANIFEST"
cat "$MANIFEST"
