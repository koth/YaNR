#!/usr/bin/env python3
"""Shape sweep (openspec task 1.8): find student shapes with predicted speedup >= 5x.

在教师形状上施加 宽度系数 x 深度系数 x 全分辨率策略 x ViT 块数 的网格,
用 cost_model 记账,输出 CSV 与 markdown 预算报告。

    python3 sweep_shapes.py                      # 默认网格
    python3 sweep_shapes.py --report ../tmp/cost_report.md

网格定义(以教师为基):
  宽度 w: 逐级通道 = 32*round(teacher_ch*w/32),下限 32;hidden = 32*max(1, round(4w))
  深度 d: 每侧块数 = max(1, round(teacher_enc*d))
  全分辨率策略: ffn-only | attn-8x8-h32 | attn-4x4(window_slots=16)
  ViT: 块数 {2,3,4},通道 = 32*round(1024*w/32),FFN = 2*通道
  split 级(d4): E = max(2, ch/64),bc = max(32, 32*round(ch/256)),mc = 4*bc
纯标准库。
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_model import account, teacher_shape                # noqa: E402
from nr_geometry import geometry_from_valid                  # noqa: E402

WIDTHS = (0.5, 0.6, 0.75)
DEPTHS = (0.4, 0.6, 0.8)
FULL_STRATEGIES = ('ffn-only', 'attn-8x8-h32', 'attn-4x4')
VIT_BLOCKS = (2, 3, 4)


def round32(value):
    return max(32, 32 * int(round(value / 32.0)))


def build_shape(w, d, strategy, vit_blocks):
    base = teacher_shape()
    hidden = 32 * max(1, int(round(4 * w)))
    levels = []
    for lv in base['levels']:
        ch = round32(lv['channels'] * w)
        enc = max(1, int(round(lv['blocks'] / 2 * d)))
        entry = {'level': lv['level'], 'kind': lv['kind'], 'channels': ch,
                 'hidden': hidden, 'blocks': 2 * enc, 'attn': True}
        if lv['level'] == 'full':
            entry['attn'] = strategy != 'ffn-only'
            entry['hidden'] = 32
            if strategy == 'attn-4x4':
                entry['window_slots'] = 16
        if lv['kind'] == 'split':
            entry['branch_channels'] = max(32, 32 * int(round(ch / 256.0)))
            entry['branches'] = ch // entry['branch_channels']       # E*bc == ch(分组切片语义)
            entry['middle_channels'] = 4 * entry['branch_channels']
        levels.append(entry)
    vit = {'channels': round32(1024 * w), 'ffn': 2 * round32(1024 * w),
           'blocks': vit_blocks, 'tokens_level': 'd5'}
    return {'name': f'w{w}-d{d}-{strategy}-v{vit_blocks}', 'levels': levels, 'vit': vit}


def run_grid(sizes=(320, 512)):
    teacher_totals = {}
    for size in sizes:
        g = geometry_from_valid(size, size)
        teacher_totals[size] = sum(r[4] for r in account(teacher_shape(), g))
    rows = []
    for w in WIDTHS:
        for d in DEPTHS:
            for strategy in FULL_STRATEGIES:
                for vb in VIT_BLOCKS:
                    shape = build_shape(w, d, strategy, vb)
                    row = {'name': shape['name'], 'width': w, 'depth': d,
                           'full': strategy, 'vit_blocks': vb}
                    for size in sizes:
                        g = geometry_from_valid(size, size)
                        stages = account(shape, g)
                        total = sum(r[4] for r in stages)
                        row[f'gmacc_{size}'] = round(total, 2)
                        row[f'speedup_{size}'] = round(teacher_totals[size] / total, 2)
                        for stage, _, _, _, gm in stages:
                            if stage in ('full', 'd0', 'd1', 'vit'):
                                row[f'{stage}_share_{size}'] = round(gm / total * 100, 1)
                    rows.append(row)
    return rows, teacher_totals


def main():
    parser = argparse.ArgumentParser(description='student shape sweep')
    parser.add_argument('--csv', default=None, help='CSV output path (default stdout summary only)')
    parser.add_argument('--report', default=None, help='markdown budget report path')
    parser.add_argument('--min-speedup', type=float, default=5.0)
    args = parser.parse_args()

    rows, teacher_totals = run_grid()
    rows.sort(key=lambda r: -r['speedup_512'])

    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'wrote {args.csv} ({len(rows)} shapes)')

    ok = [r for r in rows if r['speedup_512'] >= args.min_speedup]
    print(f"teacher totals: " + '  '.join(f"{k}² {v:.1f} GMAC" for k, v in teacher_totals.items()))
    print(f'{len(ok)}/{len(rows)} shapes reach {args.min_speedup}x at 512²; top 12:')
    for r in ok[:12]:
        print(f"  {r['name']:26s} 512² {r['gmacc_512']:6.2f} GMAC {r['speedup_512']:5.2f}x  "
              f"320² {r['speedup_320']:5.2f}x  full {r['full_share_512']:4.1f}% "
              f"d0 {r['d0_share_512']:4.1f}% vit {r['vit_share_512']:4.1f}%")

    if args.report:
        lines = ['# 学生形状扫描预算(cost_model 记账)', '',
                 f"teacher: " + '  '.join(f"{k}² {v:.2f} GMAC" for k, v in teacher_totals.items()),
                 f"网格: w={WIDTHS} d={DEPTHS} full={FULL_STRATEGIES} vit={VIT_BLOCKS}",
                 f"过 ≥{args.min_speedup}× 门禁: {len(ok)}/{len(rows)}", '',
                 '| 形状 | 512² GMAC | 512² 加速 | 320² 加速 | full 占比 | d0 占比 | ViT 占比 |',
                 '|---|---|---|---|---|---|---|']
        for r in rows:
            mark = ' ✅' if r['speedup_512'] >= args.min_speedup else ''
            lines.append(f"| {r['name']}{mark} | {r['gmacc_512']} | {r['speedup_512']}× | "
                         f"{r['speedup_320']}× | {r['full_share_512']}% | {r['d0_share_512']}% | "
                         f"{r['vit_share_512']}% |")
        with open(args.report, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print(f'wrote {args.report}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
