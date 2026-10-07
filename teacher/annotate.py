#!/usr/bin/env python3
"""Teacher annotation: online teacher forward with capture points, and sequence rollout.

openspec fast-student-distillation 任务 4.5(按 D7 的"在线教师"决策):
训练每 step 现跑教师取 head 与蒸馏特征,不缓存中间激活(18MB/点会破 TB);
允许缓存的只有小输出 —— 逐帧 head f32、stored history f16 —— 供时序训练的
teacher-forcing(历史链 = 教师 rollout,见 design.md D10)。

  TeacherAnnotator.annotate(features) -> head [rows,4] f32 + 指定捕获点激活(原 dtype)
  rollout_sequence(...)               -> 逐帧 head / composite / weight / stored history

CLI:
  python3 annotate.py --probe input.png --width 512 --height 512 --weights <dir>
      # 列出全部捕获点的名称/形状/字节数(蒸馏对齐表 2.8 的原料)
  python3 annotate.py --rollout f0.png f1.png f2.png -o out/ --weights <dir> [--motion m1.npy ...]
      # 序列 rollout:rollout.npz(head f32 / stored f16 / weight f16)+ manifest.jsonl(sha256)

依赖:numpy、torch、PIL,以及本目录的 run_image / nr_history / nr_network。
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image

import nr_torch
from nr_geometry import geometry_from_valid
from nr_history import reproject_history, store_history
from nr_model import Model
from nr_network import Network
from run_image import build_features


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class TeacherAnnotator:
    """在线教师:一次前向产出 head 与任意捕获点激活。"""

    def __init__(self, weights, g):
        self.g = g
        self.model = Model(nr_torch.DEVICE).load(weights)
        self.network = Network(self.model, g)
        self.blend_scale = float(self.model.blend_scale())

    def forward(self, features_np):
        """跑一次教师,返回 (head [rows,4] f32 torch, boundaries dict 名->张量)。"""
        features = torch.from_numpy(np.ascontiguousarray(features_np, dtype=np.float32))
        features = features.to(nr_torch.DEVICE)
        head = self.network.record(features)
        torch.cuda.synchronize()
        return head, self.network.boundaries

    def annotate(self, features_np, capture=()):
        """head f32 numpy + 捕获点激活(原 buffer dtype,通常 f16)。"""
        head, boundaries = self.forward(features_np)
        out = {'head': head.float().cpu().numpy().reshape(self.g['full_rows'], 4)}
        for name in capture:
            tensor = boundaries.get(name)
            if tensor is None:
                raise KeyError(f'no capture {name!r}; run --probe to list the names')
            out[name] = tensor.cpu().numpy()
        self.network.boundaries = None                       # 释放
        return out

    def probe(self, features_np):
        """全部捕获点的形状清单(不拷贝数据)。"""
        _, boundaries = self.forward(features_np)
        table = {name: (list(t.shape), str(t.dtype), t.numel() * t.element_size())
                 for name, t in sorted(boundaries.items())}
        self.network.boundaries = None
        return table


def rollout_sequence(annotator, proxies, seeds, lane, motions=None, capture=()):
    """按 D10 时序语义跑教师序列:历史 = 上帧 composite(truncate_half 存储)+ 重投影。

    proxies: [T][h,w,3] code values;seeds: [T] int;motions: [T][h,w,2] 或 None(全恒等)。
    返回 [T] dict:head / neural / weight / stored(+ 捕获点激活)。
    """
    out = []
    prev = None
    for t, proxy in enumerate(proxies):
        h, w = proxy.shape[:2]
        motion = None if motions is None else motions[t]
        if prev is None:
            repro = None
        else:
            m = motion if motion is not None else np.zeros((h, w, 2), np.float32)
            repro = reproject_history(prev, m)
        features = build_features(proxy, annotator.g, seeds[t], lane['style'], lane['tone'],
                                  lane['structure'], lane.get('skin', -1.0),
                                  lane.get('auto_mask', False), history=repro)
        res = annotator.annotate(features, capture=capture)
        head_v = res['head'].reshape(annotator.g['full_height'], annotator.g['full_width'], 4)[:h, :w]
        neural = np.clip(proxy + head_v[..., :3] / 4.0, 0, 1)
        sigmoid = 1.0 / (1.0 + np.exp(-head_v[..., 3].astype(np.float64)))
        weight = np.clip(sigmoid * annotator.blend_scale, 0, 1)
        if repro is not None:
            neural = neural + (repro.astype(np.float64) - neural) * weight[..., None]
        stored = store_history(neural.astype(np.float32))
        frame = {k: v for k, v in res.items() if k != 'head'}
        frame.update({'head': head_v, 'neural': neural, 'weight': weight, 'stored': stored})
        out.append(frame)
        prev = stored
    return out


def load_proxy(path, width, height):
    image = Image.open(path).convert('RGB').resize((width, height), Image.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description='teacher annotation: probe capture points / rollout sequences')
    parser.add_argument('frames', nargs='*', help='proxy frames in order (rollout mode)')
    parser.add_argument('--probe', metavar='IMAGE', help='list capture points on one forward')
    parser.add_argument('--rollout', action='store_true', help='rollout the given frames as a sequence')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--weights', default=os.environ.get('NR_WEIGHTS'), help='model directory')
    parser.add_argument('-o', '--out', default='annotate_out', help='output directory (rollout)')
    parser.add_argument('--seed', type=int, default=12345, help='base noise seed; frame t uses seed + t')
    parser.add_argument('--style', type=float, default=0.0)
    parser.add_argument('--tone', type=float, default=0.5)
    parser.add_argument('--structure', type=float, default=0.5)
    parser.add_argument('--skin', type=float, default=-1.0)
    parser.add_argument('--automask', action='store_true')
    parser.add_argument('--motion', nargs='*', default=[],
                        help='per-step motion .npy files (frame 1..T-1); omit for zero motion')
    parser.add_argument('--capture', nargs='*', default=[],
                        help='capture points to store in the rollout npz (default: none, head only)')
    args = parser.parse_args()

    if not args.weights:
        print('no weights: pass --weights or set NR_WEIGHTS')
        return 2

    g = geometry_from_valid(args.width, args.height)
    annotator = TeacherAnnotator(args.weights, g)
    lane = {'style': args.style, 'tone': args.tone, 'structure': args.structure,
            'skin': args.skin, 'auto_mask': args.automask}

    if args.probe:
        proxy = load_proxy(args.probe, args.width, args.height)
        features = build_features(proxy, g, args.seed, **lane)
        table = annotator.probe(features)
        print(f'{len(table)} capture points (field {g["full_width"]}x{g["full_height"]}, '
              f'blend_scale {annotator.blend_scale}):')
        for name, (shape, dtype, nbytes) in table.items():
            print(f'  {name:28s} {str(shape):22s} {dtype:8s} {nbytes / 1024:9.1f} KB')
        return 0

    if args.rollout:
        if not args.frames:
            raise SystemExit('rollout mode needs frame paths')
        proxies = [load_proxy(p, args.width, args.height) for p in args.frames]
        seeds = [args.seed + t for t in range(len(proxies))]
        motions = None
        if args.motion:
            if len(args.motion) != len(proxies) - 1:
                raise SystemExit(f'--motion needs {len(proxies) - 1} files (frames 1..T-1)')
            motions = [np.zeros((args.height, args.width, 2), np.float32)] + \
                      [np.load(p).astype(np.float32) for p in args.motion]
        frames = rollout_sequence(annotator, proxies, seeds, lane,
                                  motions=motions, capture=args.capture)

        os.makedirs(args.out, exist_ok=True)
        np.savez(os.path.join(args.out, 'rollout.npz'),
                 head=np.stack([f['head'] for f in frames]).astype(np.float32),
                 stored=np.stack([f['stored'] for f in frames]).astype(np.float16),
                 weight=np.stack([f['weight'] for f in frames]).astype(np.float16),
                 neural=np.stack([f['neural'] for f in frames]).astype(np.float16),
                 **{f'capture_{name}': np.stack([f[name] for f in frames]).astype(np.float16)
                    for name in args.capture})
        record = {'generator': 'annotate.py v1', 'kind': 'rollout', 'frames': len(frames),
                  'size': [args.width, args.height], 'seeds': seeds, 'lane': lane,
                  'blend_scale': annotator.blend_scale, 'capture': args.capture,
                  'inputs': [{'path': p, 'sha256': sha256_file(p)} for p in args.frames],
                  'motion': [os.path.basename(m) for m in args.motion]}
        with open(os.path.join(args.out, 'manifest.jsonl'), 'w') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')
        for t, f in enumerate(frames):
            print(f'frame {t}: head rgb/4 mean {f["head"][..., :3].mean() / 4:+.4f}  '
                  f'weight mean {f["weight"].mean():.3f}  '
                  f'stored diff vs prev '
                  f'{0.0 if t == 0 else np.abs(f["stored"] - frames[t - 1]["stored"]).mean():.4f}')
        print('wrote', os.path.join(args.out, 'rollout.npz'))
        return 0

    raise SystemExit('choose --probe or --rollout')


if __name__ == '__main__':
    raise SystemExit(main())
