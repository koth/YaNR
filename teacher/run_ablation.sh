#!/usr/bin/env bash
# 形状敏感度 + 损失消融矩阵(5.14 + 30ms 形状决策):各 20K 步 @320²,同预算对比。
#
#   ./run_ablation.sh <cell>        跑一个 cell(可开多个实例并行)
#   ./run_ablation.sh summary       汇总所有已完成 cell 成 markdown 表
#
# cell 清单:
#   shape_v0 | shape_blocks | shape_mixed | shape_deep     形状敏感度(质量-MAC 曲线)
#   loss_nofeature | loss_nodetail                         5.14 损失消融
#   loss_temporal                                         5.14 时序损失(实现后才有意义)
# 注:历史来源消融暂缺 —— train 侧历史腿未实现(见 6.3/5.13 记录)。
# 依赖:与 v1 相同的数据/损失配置,仅 --shape 或损失权重不同;评估用固定图集。
set -euo pipefail
CELL=${1:?usage: run_ablation.sh <cell>|summary}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T=$ROOT/teacher
OUT=/tmp/ablation
IMAGES="$ROOT/data/raw/DAVIS/JPEGImages/Full-Resolution/bear/0000*.jpg \
        $ROOT/data/raw/DAVIS/JPEGImages/Full-Resolution/camel/0000*.jpg \
        $ROOT/data/raw/DAVIS/JPEGImages/Full-Resolution/dogs/0000*.jpg"
SEQS=("$ROOT/data/raw/DAVIS/JPEGImages/Full-Resolution/bear/*.jpg" \
      "$ROOT/data/raw/DAVIS/JPEGImages/Full-Resolution/camel/*.jpg")

if [ "$CELL" = summary ]; then
    python3 - "$OUT" <<'EOF'
import json, os, sys
out = sys.argv[1]
rows = []
for cell in sorted(os.listdir(out)):
    p = os.path.join(out, cell, 'eval_quality.json')
    t = os.path.join(out, cell, 'eval_temporal.json')
    if not os.path.isfile(p):
        continue
    q = json.load(open(p))
    key = str(q['config']['sizes'][0])
    agg = q['aggregate'][key]
    tr = json.load(open(t))['aggregate'] if os.path.isfile(t) else {}
    rows.append((cell, agg, tr))
print('| cell | psnr 学生 | psnr 教师 | 退化 dB | vs 教师 dB | SSIM | blend_mae | flicker × | warp × |')
print('|---|---|---|---|---|---|---|---|---|')
for cell, q, tr in rows:
    print(f"| {cell} | {q['psnr_student']['mean']:.2f} | {q['psnr_teacher']['mean']:.2f} "
          f"| {q['degradation_db']['mean']:+.2f} | {q['psnr_vs_teacher']['mean']:.2f} "
          f"| {q['ssim_vs_teacher']['mean']:.4f} | {q['blend_mae']['mean']:.4f} "
          f"| {tr.get('flicker_ratio', float('nan')):.2f} | {tr.get('warp_ratio', float('nan')):.2f} |")
EOF
    exit 0
fi

mkdir -p "$OUT/$CELL"
DATA=("$ROOT/data/raw/DIV2K_train_HR" "$ROOT/data/raw/DAVIS/JPEGImages/Full-Resolution" \
      "$ROOT/data/train/proxy_proc")
SHAPE_V0="$ROOT/shapes/student_v0.json"
STEPS=${STEPS:-5000}         # 相对曲线 5K 步即可分形状;warmup 按比例缩
WARMUP=${WARMUP:-1000}       # 最终质量另跑长跑确认
TRAIN=(python3 "$T/train_distill.py" --teacher-weights "$ROOT/models/nr" --data "${DATA[@]}"
       --size 320 --batch 4 --steps "$STEPS" --warmup "$WARMUP")
EVAL=(python3 "$T/eval_quality.py" --teacher-weights "$ROOT/models/nr" --sizes 320
      --limit 8 -o "$OUT/$CELL")

case $CELL in
  shape_v0)     SHAPE="$SHAPE_V0" ;;
  shape_blocks) SHAPE="$ROOT/shapes/student_slim_blocks.json" ;;
  shape_mixed)  SHAPE="$ROOT/shapes/student_slim_mixed.json" ;;
  shape_deep)   SHAPE="$ROOT/shapes/student_slim_deep.json" ;;
  loss_nofeature|loss_nodetail|loss_temporal) SHAPE="$SHAPE_V0" ;;
  *) echo "unknown cell: $CELL"; exit 2 ;;
esac
case $CELL in
  loss_nofeature) EXTRA=(--loss-feature 0) ;;
  loss_nodetail)  EXTRA=(--loss-detail 0) ;;
  loss_temporal)  EXTRA=(--loss-temporal 0.5) ;;
  *)              EXTRA=() ;;
esac

echo "== [$CELL] train =="
"${TRAIN[@]}" --shape "$SHAPE" "${EXTRA[@]}" -o "$OUT/$CELL/run"
echo "== [$CELL] eval_quality =="
"${EVAL[@]}" --shape "$SHAPE" --images $IMAGES --checkpoint "$OUT/$CELL/run/ckpt.pt"
echo "== [$CELL] eval_temporal =="
python3 "$T/eval_temporal.py" --teacher-weights "$ROOT/models/nr" --shape "$SHAPE" \
    --seq "${SEQS[@]}" --checkpoint "$OUT/$CELL/run/ckpt.pt" --size 320 --max-frames 4 \
    -o "$OUT/$CELL"
echo "== [$CELL] done =="
