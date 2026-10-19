"""按表头取任意列的中位数,支持多个文件/窗口对比。

logwin.py 只打固定的 out/detail/feature/psnr 四列;查 tex/stat 时不够用。
用法: python3 win.py <lo> <hi> <csv> [<lo2> <hi2> <csv2> ...]
"""
import csv
import statistics
import sys


def read(path):
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    return rows


def main():
    args = sys.argv[1:]
    groups = [args[i:i + 3] for i in range(0, len(args), 3)]
    cache = {}
    for lo, hi, path in groups:
        lo, hi = int(lo), int(hi)
        rows = cache.setdefault(path, read(path))
        sel = [r for r in rows if lo <= int(r['step']) < hi]
        if not sel:
            print(f'[{lo},{hi}] {path}: 无数据')
            continue
        tag = path.rsplit('/', 2)[-2] if path.count('/') >= 2 else path
        parts = []
        for key in ('lr', 'out', 'detail', 'feature', 'tex', 'stat', 'out_logit', 'psnr_teacher'):
            if key not in sel[0]:
                continue
            values = [float(r[key]) for r in sel]
            parts.append(f'{key}={statistics.median(values):.5g}')
        print(f'[{lo:5d},{hi:5d}] {tag:<10} n={len(sel):>3} ' + ' '.join(parts))


if __name__ == '__main__':
    main()
