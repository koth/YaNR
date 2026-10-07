#!/usr/bin/env python3
"""Student benchmark (openspec tasks 2.12 / 3.1 / 3.2 / 3.5).

eager / torch.compile × f32 / autocast(f16) 四种组合,多个分辨率,对照 cost model 的
GMAC 记账与教师实测(3090:320² 57ms、512² 113ms、768² 231ms)。

    python3 bench_student.py --shape ../shapes/student_v0.json --sizes 320,512,768
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_model import account, teacher_shape                    # noqa: E402
from nr_geometry import geometry_from_valid                      # noqa: E402
from nr_student import StudentNetwork, load_student_shape        # noqa: E402
from run_image import build_features                             # noqa: E402

TEACHER_MS = {320: 57.0, 512: 113.0, 768: 231.0}


def make_features(g, size):
    proxy = np.random.default_rng(0).random((size, size, 3), dtype=np.float32)
    return torch.from_numpy(build_features(proxy, g, 12345, 0.0, 0.5, 0.5, -1.0, False))


def time_forward(model, features, device, repeats, autocast_dtype=None):
    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(3):                                  # warmup(编译/缓存)
            with torch.autocast(device, dtype=autocast_dtype) if autocast_dtype else _noop():
                model(features)
        sync()
        start = time.perf_counter()
        for _ in range(repeats):
            with torch.autocast(device, dtype=autocast_dtype) if autocast_dtype else _noop():
                model(features)
        sync()
    return (time.perf_counter() - start) / repeats * 1000


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main():
    parser = argparse.ArgumentParser(description='student benchmark')
    parser.add_argument('--shape', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        '..', 'shapes', 'student_v0.json'))
    parser.add_argument('--sizes', default='320,512,768')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--modes', default='eager32,eager16,compile32,compile16')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.set_float32_matmul_precision('high')          # f32 matmul 走 TF32 张量核
    shape = load_student_shape(args.shape)
    sizes = [int(s) for s in args.sizes.split(',') if s]
    modes = [m for m in args.modes.split(',') if m]
    params = None

    print(f"shape {shape.get('name', args.shape)}  device {device}  repeats {args.repeats}")
    print(f"{'size':>6} {'mode':>10} {'ms':>8} {'GMAC/s':>9} {'vs teacher':>11} "
          f"{'GMAC(cost)':>11} {'pred ms':>8}")
    for size in sizes:
        g = geometry_from_valid(size, size)
        features = make_features(g, size).to(device)
        gm = sum(r[4] for r in account(shape, g))
        teacher = TEACHER_MS.get(size)
        for mode in modes:
            torch.manual_seed(0)
            model = StudentNetwork(shape, g).to(device).eval()
            params = sum(p.numel() for p in model.parameters())
            autocast_dtype = torch.float16 if mode.endswith('16') else None
            fn = model
            if mode.startswith('compile'):
                try:
                    fn = torch.compile(model, mode='reduce-overhead')
                except Exception as exc:                     # noqa: BLE001
                    print(f'{size:>6} {mode:>10}  compile unavailable: {exc}')
                    continue
            ms = time_forward(fn, features, device, args.repeats, autocast_dtype)
            speed = gm * 1000 / ms
            vs = f'{teacher / ms:.2f}x' if teacher else '-'
            print(f'{size:>6} {mode:>10} {ms:>8.2f} {speed:>9.0f} {vs:>11} {gm:>11.2f} '
                  f'{gm * 1000 / 607:>8.1f}')
    print(f'params {params / 1e6:.2f}M   (cost model pred ms assumes teacher-calibrated 607 GMAC/s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
