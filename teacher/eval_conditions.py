#!/usr/bin/env python3
"""条件控件扫描验收(style/tone/structure/skin/automask):教师全量程行为 vs 学生保真度。

背景(2026-10-10 探针结论):教师是条件增强滤镜,改动幅度由条件 lane 控制(all-max ≈ PSNR 20dB、
全关 ≈ 56dB);v1.1 学生只在近零档训练过(style lane ≤ 0.023),高档失控。本工具把
"教师每个档位改多少 / 学生跟得多紧"扫成表,作为 v1.2 重训与否的决策依据,也是重训后的验收口径。

    python3 eval_conditions.py --images a.png b.png --size 512 \
        --teacher-weights $NR_WEIGHTS --student weights/student_mixed -o eval_cond/

口径:每配置 × 每图,two 网络吃同一份 features;
  teacher(in)  = PSNR(教师合成图, proxy)      —— 教师改动幅度(越小改得越狠)
  student(in)  = PSNR(学生合成图, proxy)
  s-vs-t       = PSNR(学生合成图, 教师合成图)   —— 蒸馏保真度(核心指标)
  corr         = 学生/教师残差皮尔逊相关
配置行标注 IN/OOD:是否落在 v1.1 训练采样范围(style lane [0,0.023]、tone [0.3,0.7]、
structure [0.2,0.8]、skin=-1、automask=False)。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_geometry import geometry_from_valid                     # noqa: E402
from nr_model import Model                                      # noqa: E402
from nr_network import Network                                  # noqa: E402
from nr_package import is_student_package, load_student_package  # noqa: E402
from nr_student import StudentNetwork, load_student_shape       # noqa: E402
from run_image import build_features, save_png                  # noqa: E402

# v1.1 训练采样范围(train_distill.py 默认 cfg)—— 行标 IN/OOD 的判据
TRAIN_RANGE = {'style': (0.0, 3.0), 'tone': (0.3, 0.7), 'structure': (0.2, 0.8),
               'skin': (-1.0, -1.0), 'automask': False}


def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def in_train_range(cfg):
    for key in ('style', 'tone', 'structure'):
        lo, hi = TRAIN_RANGE[key]
        if not (lo <= cfg[key] <= hi):
            return False
    return (cfg['skin'] == TRAIN_RANGE['skin'][0]
            and cfg['automask'] == TRAIN_RANGE['automask'])


BASE = dict(style=1.5, tone=0.5, structure=0.35, skin=-1.0, automask=False)


def sweep_plan(base_style):
    """(名称, 条件覆盖 dict);单变量扫描 + 角点。tone/struct/skin 行挂在 base_style 工作点上
    (默认 1.5 = 训练区内,分离 style 之外的覆盖问题;64 可复现高档淹没效应)。"""
    plan = []
    for sid in (0, 2, 4, 8, 16, 32, 64, 96, 127):
        plan.append((f'style={sid}', dict(style=float(sid))))
    for tone in (0.0, 0.25, 0.5, 0.75, 1.0):
        plan.append((f'tone={tone}', dict(style=base_style, tone=tone)))
    for st in (0.0, 0.25, 0.5, 0.75, 1.0):
        plan.append((f'struct={st}', dict(style=base_style, structure=st)))
    for skin in (0.0, 0.5, 1.0):
        plan.append((f'skin={skin}+am', dict(style=base_style, skin=skin, automask=True)))
    plan.append(('all-min', dict(style=0.0, tone=0.0, structure=0.0, skin=0.0)))
    plan.append(('all-max', dict(style=127.0, tone=1.0, structure=1.0, skin=1.0)))
    return plan


def main():
    parser = argparse.ArgumentParser(description='conditioning sweep acceptance (6.x)')
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--base-style', type=float, default=1.5,
                        help='tone/struct/skin 单变量行的工作点 style ID(默认训练区内;64 复现高档淹没)')
    parser.add_argument('--teacher-weights', default=os.environ.get('NR_WEIGHTS'))
    parser.add_argument('--student', help='学生包目录')
    parser.add_argument('--shape', help='或 ckpt 口径:形状 DSL')
    parser.add_argument('--checkpoint', help='或 ckpt 口径:train_distill ckpt.pt')
    parser.add_argument('-o', '--out', default='eval_cond')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = geometry_from_valid(args.width, args.height)
    tmodel = Model(device).load(args.teacher_weights)
    teacher = Network(tmodel, g)
    if args.student and is_student_package(args.student):
        student, manifest = load_student_package(args.student, args.width, args.height, device)
        student_tag = manifest['model']['version']
    else:
        torch.manual_seed(0)
        student = StudentNetwork(load_student_shape(args.shape), g).to(device).eval()
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        student.load_state_dict(ckpt.get('ema', ckpt.get('model', ckpt)))
        student_tag = f"ckpt(step {ckpt.get('step', '?')})"
    print(f'student: {student_tag}  size {args.width}x{args.height}  images {len(args.images)}')

    files = []
    for pat in args.images:
        import glob as _glob
        files += sorted(_glob.glob(pat)) or ([pat] if os.path.isfile(pat) else [])
    rows = []
    os.makedirs(args.out, exist_ok=True)
    for name, over in sweep_plan(args.base_style):
        cfg = dict(BASE)
        cfg.update(over)
        per = []
        for path in files:
            proxy = np.asarray(Image.open(path).convert('RGB')
                               .resize((args.width, args.height), Image.LANCZOS),
                               dtype=np.float32) / 255.0
            feats = torch.from_numpy(build_features(
                proxy, g, args.seed, cfg['style'], cfg['tone'], cfg['structure'],
                cfg['skin'], cfg['automask'])).to(device)
            with torch.no_grad():
                outs = {}
                for tag, net in (('t', teacher), ('s', student)):
                    head = net.record(feats) if tag == 't' else net(feats)
                    head_np = head.float().cpu().numpy().reshape(g['full_height'],
                                                                 g['full_width'], 4)
                    valid = head_np[:args.height, :args.width]
                    outs[tag] = np.clip(proxy + valid[..., :3] / 4.0, 0, 1)
            res_t = (outs['t'] - proxy).ravel()
            res_s = (outs['s'] - proxy).ravel()
            corr = (float(np.corrcoef(res_t, res_s)[0, 1])
                    if res_t.std() > 0 and res_s.std() > 0 else 0.0)
            per.append({'image': os.path.basename(path),
                        'teacher_in': psnr(outs['t'], proxy), 'student_in': psnr(outs['s'], proxy),
                        's_vs_t': psnr(outs['s'], outs['t']), 'corr': corr})
        agg = {k: float(np.mean([p[k] for p in per]))
               for k in ('teacher_in', 'student_in', 's_vs_t', 'corr')}
        row = {'config': name, 'params': cfg,
               'ood': not in_train_range(cfg), 'aggregate': agg, 'per_image': per}
        rows.append(row)
        print(f"  {name:<12} {'OOD' if row['ood'] else 'IN ':>3}  "
              f"teacher(in) {agg['teacher_in']:6.2f}  student(in) {agg['student_in']:6.2f}  "
              f"s-vs-t {agg['s_vs_t']:6.2f}  corr {agg['corr']:.3f}")

    report = {'config': {'size': [args.width, args.height], 'seed': args.seed,
                         'student': student_tag, 'images': files,
                         'base_style': args.base_style, 'train_range': TRAIN_RANGE},
              'rows': rows}
    with open(os.path.join(args.out, 'eval_conditions.json'), 'w') as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(args.out, 'eval_conditions.md'), 'w') as fh:
        fh.write('| 配置 | 训练覆盖 | 教师 vs 输入 | 学生 vs 输入 | s-vs-t | corr |\n')
        fh.write('|---|---|---|---|---|---|\n')
        for r in rows:
            a = r['aggregate']
            fh.write(f"| {r['config']} | {'❌ OOD' if r['ood'] else '✅ IN'} "
                     f"| {a['teacher_in']:.2f}dB | {a['student_in']:.2f}dB "
                     f"| {a['s_vs_t']:.2f}dB | {a['corr']:.3f} |\n")
    print(f"wrote {args.out}/eval_conditions.json + .md")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
