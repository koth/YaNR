"""比较两个训练日志在给定步数窗口内的中位数指标(容忍两种格式)。

用法: python3 logwin.py <lo> <hi> <log[:format]> [...]   format: csv | print(默认)
csv   = 本次 run 的 log.csv(step,lr,loss,out,detail,feature,temporal,psnr,ms)
print = 旧 run 的逐行 "step 19000 lr ... out ... detail ... feat ... psnr(t) ..."
"""
import statistics
import sys


def parse_csv(path, lo, hi):
    rows = []
    for line in open(path):
        parts = line.strip().split(',')
        if len(parts) < 8 or parts[0] == 'step':
            continue
        step = int(parts[0])
        if lo <= step <= hi:
            rows.append(dict(step=step, out=float(parts[3]), detail=float(parts[4]),
                             feature=float(parts[5]), psnr=float(parts[7])))
    return rows


def parse_print(path, lo, hi):
    rows = []
    for line in open(path, errors='replace'):
        parts = line.split()
        if not parts or parts[0] != 'step' or len(parts) < 4:
            continue
        try:
            step = int(parts[1])
        except ValueError:
            continue
        if not (lo <= step <= hi):
            continue
        row = dict(step=step)
        for i, token in enumerate(parts):
            if token in ('out', 'detail', 'feat') and i + 1 < len(parts):
                row[{'feat': 'feature'}.get(token, token)] = float(parts[i + 1])
            if token.startswith('psnr') and i + 1 < len(parts):
                row['psnr'] = float(parts[i + 1])
        rows.append(row)
    return rows


def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    for spec in sys.argv[3:]:
        path, _, fmt = spec.partition(':')
        rows = parse_csv(path, lo, hi) if fmt == 'csv' else parse_print(path, lo, hi)
        if not rows:
            print(f'{path} [{lo},{hi}]: 无数据')
            continue
        def med(key):
            values = [r[key] for r in rows if key in r]
            return statistics.median(values) if values else float('nan')
        print(f'{path.split("/")[-1]:24s} [{lo},{hi}] n={len(rows):3d}  '
              f'out={med("out"):.3f}  detail={med("detail"):.4f}  '
              f'feature={med("feature"):.3f}  psnr={med("psnr"):.2f}')


if __name__ == '__main__':
    main()
