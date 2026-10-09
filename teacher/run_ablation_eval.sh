#!/usr/bin/env bash
# 消融 cell 的纯评估段(shape 感知):训练已完成/崩溃后补评估用。
#   ./run_ablation_eval.sh <cell>
# 与 run_ablation.sh 的 cell 清单一一对应;不动训练,只跑 eval_quality + eval_temporal。
set -euo pipefail
CELL=${1:?usage: run_ablation_eval.sh <cell>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T=$ROOT/teacher
OUT=/tmp/ablation
IMAGES="$ROOT/data/raw/DAVIS/JPEGImages/Full-Resolution/bear/0000*.jpg \
        $ROOT/data/raw/DAVIS/JPEGImages/Full-Resolution/camel/0000*.jpg \
        $ROOT/data/raw/DAVIS/JPEGImages/Full-Resolution/dogs/0000*.jpg"
SEQS=("$ROOT/data/raw/DAVIS/JPEGImages/Full-Resolution/bear/*.jpg" \
      "$ROOT/data/raw/DAVIS/JPEGImages/Full-Resolution/camel/*.jpg")
case $CELL in
  shape_v0|loss_*) SHAPE="$ROOT/shapes/student_v0.json" ;;
  shape_blocks)    SHAPE="$ROOT/shapes/student_slim_blocks.json" ;;
  shape_mixed)     SHAPE="$ROOT/shapes/student_slim_mixed.json" ;;
  shape_deep)      SHAPE="$ROOT/shapes/student_slim_deep.json" ;;
  *) echo "unknown cell: $CELL"; exit 2 ;;
esac
mkdir -p "$OUT/$CELL"
echo "== [$CELL] eval_quality =="
python3 "$T/eval_quality.py" --teacher-weights "$ROOT/models/nr" --shape "$SHAPE" \
    --sizes 320 --limit 8 --checkpoint "$OUT/$CELL/run/ckpt.pt" --images $IMAGES \
    -o "$OUT/$CELL"
echo "== [$CELL] eval_temporal =="
python3 "$T/eval_temporal.py" --teacher-weights "$ROOT/models/nr" --shape "$SHAPE" \
    --seq "${SEQS[@]}" --checkpoint "$OUT/$CELL/run/ckpt.pt" --size 320 --max-frames 4 \
    -o "$OUT/$CELL"
echo "== [$CELL] eval done =="
