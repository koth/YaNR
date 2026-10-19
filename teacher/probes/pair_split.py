"""按是否配对样本统计 v45i 的损失,判断时序分支是否退化。"""
import csv
import sys

path = sys.argv[1]
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rows = [r for r in csv.DictReader(open(path)) if int(r['step']) > lo]
pair = [r for r in rows if float(r['temporal']) > 0.001]
non = [r for r in rows if float(r['temporal']) <= 0.001]


def mean(rs, key):
    return sum(float(r[key]) for r in rs) / max(len(rs), 1)


print(f'steps>{lo}  n={len(rows)}  pair={len(pair)} '
      f'({100 * len(pair) / max(len(rows), 1):.0f}%)  non={len(non)}')
for name, rs in (('pair', pair), ('non ', non)):
    if rs:
        print(f'{name}  out={mean(rs, "out"):.3f}  '
              f'logit={mean(rs, "out_logit"):.3f}  '
              f'psnr={mean(rs, "psnr_teacher"):.2f}')
