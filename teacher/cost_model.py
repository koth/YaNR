#!/usr/bin/env python3
"""MACs / latency cost model (openspec tasks 1.1-1.4, 1.6).

把"5× 从哪来"变成可查账的数字:逐块 GMAC 记账(FFN/expert/split/ViT、qkv/proj、
窗口注意力、过渡 gemm、adapter/head),按尺寸输出逐级预算表、总计与加速比。

算式(每像素 MACs/块,已用教师形状对实测锚定 512² ≈ 68.5 G):
  plain  FFN 块:  2*ch*h + 4*ch^2 + 128*ch        (h=hidden; 128*ch 是 8x8 窗口注意力)
  expert FFN 块:  ch^2*(h/32+1) + ch*h + 4*ch^2 + 128*ch   (专家数 ch/32,窄瓶颈 32)
  split 块:       6*ch^2 + 2*E*bc*mc + 128*ch      (E 分支:branch+contract 2ch^2、窄 bc->mc->bc、qkv/proj 4ch^2)
  ViT 块:         2*ch*ffn + 4*ch^2 + 128*ch       (tokens 很少,注意力项另算 2*T*Tpad*32*heads)
  过渡 gemm:      rows * (k*n)                     (enc ch->2ch, dec 2ch->ch, vit 512->1024/1024->512)

耗时预估:有效吞吐 --gops(默认 607 G products/s,由 3090 实测 512² 113ms 校准)。
纯标准库,任何 python3 可跑:
  python3 cost_model.py --shape teacher --sizes 320,512,768,1080
  python3 cost_model.py --shape shapes/student_v0.json --sizes 512 --json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_geometry import geometry_from_valid                      # noqa: E402  纯标准库

ATTN_MACS_PER_CH = 128.0        # 8x8 窗口(64 槽)qk+av,heads = ch/32 -> 4096 MACs/px/head
DEFAULT_GOPS = 607.0            # 3090 实测:512² 68.5G / 113ms
# 注:注意力 = 2*window_slots*ch;64 槽即 128*ch(ATTN_MACS_PER_CH),4x4 备选 16 槽即 32*ch。


# ---------------------------------------------------------------- shapes

def teacher_shape():
    """教师结构(71 块,nr_network.py 的布局)。块数 = enc+dec 合计。"""
    return {
        'name': 'teacher',
        'levels': [
            {'level': 'full', 'kind': 'ffn', 'channels': 32, 'hidden': 128, 'blocks': 2, 'attn': True},
            {'level': 'd0', 'kind': 'ffn', 'channels': 32, 'hidden': 128, 'blocks': 8, 'attn': True},
            {'level': 'd1', 'kind': 'expert', 'channels': 64, 'hidden': 128, 'blocks': 8, 'attn': True},
            {'level': 'd2', 'kind': 'expert', 'channels': 128, 'hidden': 128, 'blocks': 12, 'attn': True},
            {'level': 'd3', 'kind': 'expert', 'channels': 256, 'hidden': 128, 'blocks': 16, 'attn': True},
            {'level': 'd4', 'kind': 'split', 'channels': 512, 'hidden': 128, 'blocks': 16, 'attn': True,
             'branches': 8, 'branch_channels': 64, 'middle_channels': 256},
        ],
        'vit': {'channels': 1024, 'ffn': 4096, 'blocks': 8, 'tokens_level': 'd5'},
    }


def load_shape(path):
    with open(path) as fh:
        spec = json.load(fh)
    for level in spec['levels']:
        if level['channels'] % 32 or level['hidden'] % 32:
            raise ValueError(f"{level['level']}: channels/hidden must be multiples of 32")
        if level['kind'] not in ('ffn', 'expert', 'split'):
            raise ValueError(f"{level['level']}: unknown kind {level['kind']!r}")
        if level['kind'] == 'split':
            for key in ('branches', 'branch_channels', 'middle_channels'):
                if key not in level:
                    raise ValueError(f"{level['level']}: split needs {key}")
            if level['branches'] * level['branch_channels'] != level['channels']:
                raise ValueError(f"{level['level']}: split needs branches*branch_channels == channels")
    spec.setdefault('name', os.path.splitext(os.path.basename(path))[0])
    return spec


# ---------------------------------------------------------------- formulas

def block_macs_per_px(level):
    ch = level['channels']
    h = level['hidden']
    slots = level.get('window_slots', 64)                    # 8x8 窗 64 槽;4x4 备选 16 槽
    attn = 2.0 * slots * ch if level.get('attn') else 0.0    # qk+av: 2*slots*ch(64 槽 -> 128*ch)
    if level['kind'] == 'ffn':
        ffn = 2 * ch * h
    elif level['kind'] == 'expert':
        ffn = ch * ch * (h / 32.0 + 1.0) + ch * h          # expand + narrow(32) + project, E = ch/32
    else:  # split
        e, bc, mc = level['branches'], level['branch_channels'], level['middle_channels']
        ffn = ch * ch + 2 * e * bc * mc + ch * ch           # branch + 窄 FFN + contract
    return ffn + 4 * ch * ch + attn                          # + qkv 3ch^2 + proj ch^2


def vit_macs_per_token(level):
    ch, ffn = level['channels'], level['ffn']
    return 2 * ch * ffn + 4 * ch * ch


def vit_attn_macs(level, tokens, padded):
    heads = level['channels'] // 32
    return 2 * tokens * padded * 32 * heads * level['blocks']


def level_rows(g, name):
    if name == 'full':
        return g['full_rows']
    return g['levels'][int(name[1:])]['rows']


# ---------------------------------------------------------------- accounting

def account(shape, g):
    """逐块/逐级 GMAC + 过渡 gemm。返回 [(stage, kind, params_equiv, rows, gmacc)]。"""
    rows_out = []
    for level in shape['levels']:
        per_px = block_macs_per_px(level)
        rows = level_rows(g, level['level'])
        rows_out.append((level['level'], level['kind'], per_px, rows, per_px * rows * level['blocks'] / 1e9))
    vit = shape['vit']
    tokens = level_rows(g, vit.get('tokens_level', 'd5'))
    padded = (tokens + 63) & ~63
    gm_vit = vit_macs_per_token(vit) * tokens * vit['blocks'] / 1e9
    gm_vit += vit_attn_macs(vit, tokens, padded) / 1e9
    rows_out.append(('vit', 'vit', vit_macs_per_token(vit), tokens, gm_vit))

    # 过渡:enc 每级 ch->2ch、dec 2ch->ch、full->d0 纯下采样、vit 进出 512<->1024、head、adapter。
    channels = [lv['channels'] for lv in shape['levels']]          # full..d4
    trans = 0.0
    for i in range(len(channels) - 1):                             # full..d3 -> 下一级
        rows_next = level_rows(g, shape['levels'][i + 1]['level'])
        trans += rows_next * channels[i] * channels[i + 1] * 2 / 1e9      # enc ch->2ch + dec 2ch->ch
    rows_vit = level_rows(g, vit.get('tokens_level', 'd5'))
    ch_last = channels[-1]
    trans += rows_vit * ch_last * vit['channels'] * 2 / 1e9               # 512<->1024
    rows_full = g['full_rows']
    trans += rows_full * (16 * channels[0] + channels[0] * 4) / 1e9       # adapter + head
    rows_out.append(('transition', 'gemm', 0.0, 0, trans))
    return rows_out


def report(shape, sizes, gops=DEFAULT_GOPS, as_json=False):
    results = []
    for size in sizes:
        g = geometry_from_valid(size, size)
        rows = account(shape, g)
        total = sum(r[4] for r in rows)
        results.append({'size': size, 'field': [g['full_width'], g['full_height']],
                        'stages': [{'stage': s, 'kind': k, 'macs_per_px': round(p, 1), 'rows': r_,
                                    'gmacc': round(gm, 3)} for s, k, p, r_, gm in rows],
                        'total_gmacc': round(total, 2),
                        'predicted_ms': round(total * 1e9 / (gops * 1e9) * 1000, 1)})
    if as_json:
        print(json.dumps({'shape': shape['name'], 'gops': gops, 'results': results},
                         indent=1, sort_keys=True))
        return results
    for res in results:
        print(f"== {res['size']}²  field {res['field'][0]}x{res['field'][1]}  "
              f"total {res['total_gmacc']} GMAC  ~{res['predicted_ms']} ms @{gops:.0f} G/s")
        for s in res['stages']:
            share = s['gmacc'] / res['total_gmacc'] * 100
            print(f"  {s['stage']:10s} {s['kind']:7s} {s['macs_per_px']:10.1f} MAC/px  "
                  f"rows {s['rows']:7d}  {s['gmacc']:7.2f} GMAC  {share:5.1f}%")
    return results


def main():
    parser = argparse.ArgumentParser(description='student/teacher MACs and latency cost model')
    parser.add_argument('--shape', default='teacher', help='"teacher" or path to a shapes/*.json')
    parser.add_argument('--sizes', default='512', help='comma list of square sizes')
    parser.add_argument('--gops', type=float, default=DEFAULT_GOPS,
                        help='effective throughput in G products/s (calibrated on 3090)')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    shape = teacher_shape() if args.shape == 'teacher' else load_shape(args.shape)
    sizes = [int(s) for s in args.sizes.split(',') if s]
    report(shape, sizes, gops=args.gops, as_json=args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
