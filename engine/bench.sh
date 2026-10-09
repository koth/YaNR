#!/usr/bin/env bash
# 一键基准(openspec 8.5):lane 构造 / 前向 / composite 分段计时,30 次取中位。
#
#   ./bench.sh <student.idx> <features.bin> <proxy.f32> <size> [--int8]
#
# 先构建:cmake -B build -S . && cmake --build build -j 8
# 线程口径:OMP_NUM_THREADS=8(i9-10850K 实测最优;20 线程 HT 过订阅慢 ~8x)。
set -euo pipefail
IDX=${1:?usage: bench.sh <student.idx> <features.bin> <proxy.f32> <size> [--int8]}
FEAT=$2
PROXY=$3
SIZE=$4
EXTRA=${5:-}
DIR="$(cd "$(dirname "$0")" && pwd)"
BIN=$DIR/build
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
if [ "$EXTRA" = "--int8" ]; then export NR_INT8=1; fi

echo "== lane 构造 =="
"$BIN/cpu_lanes" --proxy "$PROXY" --out /tmp/bench_lanes.bin --vw "$SIZE" --vh "$SIZE" \
    --seed 7 --bench 30
echo "== 前向 =="
"$BIN/cpu_engine" --idx "$IDX" --features "$FEAT" --bench --repeats 30
echo "== composite + PNG =="
"$BIN/cpu_engine" --idx "$IDX" --features "$FEAT" --out /tmp/bench_head.bin \
    --proxy "$PROXY" --png /tmp/bench.png --bench --repeats 30 | grep composite
