#!/usr/bin/env python3
"""Elementwise parity: C++ lane pipeline vs run_image.build_features (openspec task 8.3).

契约(f16 口径 ≤1 LSB):C++ 的 libm 与 numpy 在 log2/cos/sin 上可能差 ULPs,
经 f16 舍入后偶尔落到相邻 f16 值 —— 允许相邻,不允许更大偏差。

    python3 check_lanes.py --image ../teacher/samples/lake.png --size 512 --seed 12345
    python3 check_lanes.py --image <png> --size 320 --automask --history-test
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_geometry import geometry_from_valid                  # noqa: E402
from run_image import build_features                        # noqa: E402


def f16_adjacency_ok(ref, got):
    """ref/got: 1D f32 数组(值在 f16 网格上)。返回 (最大 f16 步距, 越界元素数)。"""
    a = ref.astype(np.float16).view(np.uint16).astype(np.int32)
    b = got.astype(np.float16).view(np.uint16).astype(np.int32)
    steps = np.abs(a - b)
    # 位模式距离在正数区等于步距;负数区符号位在高位,取 f16 数值间距兜底
    bad = (np.abs(ref - got) > np.spacing(ref.astype(np.float16)).astype(np.float32) * 1.01)
    return int(steps.max()), int(bad.sum())


def main():
    parser = argparse.ArgumentParser(description='C++ vs python lane parity')
    parser.add_argument('--image', required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--style', type=float, default=1.5)
    parser.add_argument('--tone', type=float, default=0.5)
    parser.add_argument('--structure', type=float, default=0.35)
    parser.add_argument('--skin', type=float, default=-1.0)
    parser.add_argument('--automask', action='store_true')
    parser.add_argument('--history-test', action='store_true',
                        help='also test the history lane path with a shifted image')
    parser.add_argument('--build', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        '..', 'engine', 'build'))
    args = parser.parse_args()

    size = args.size
    g = geometry_from_valid(size, size)
    proxy = np.asarray(Image.open(args.image).convert('RGB').resize((size, size), Image.LANCZOS),
                       dtype=np.float32) / 255.0
    tool = os.path.join(args.build, 'cpu_lanes')
    if not os.path.exists(tool):
        raise SystemExit(f'build the engine first: {tool} not found')

    tmp = tempfile.mkdtemp(prefix='lanes_')
    proxy_path = os.path.join(tmp, 'proxy.bin')
    proxy.tofile(proxy_path)

    cases = [('first-frame', None)]
    if args.history_test:
        shifted = np.roll(proxy, 7, axis=1)
        hist_path = os.path.join(tmp, 'hist.bin')
        shifted.tofile(hist_path)
        cases.append(('with-history', (shifted, hist_path)))

    failures = 0
    for name, hist in cases:
        history = hist[0] if hist else None
        ref = build_features(proxy, g, args.seed, args.style, args.tone,
                             args.structure, args.skin, args.automask, history=history)
        out_path = os.path.join(tmp, 'lanes.bin')
        cmd = [tool, '--proxy', proxy_path, '--out', out_path,
               '--vw', str(size), '--vh', str(size), '--seed', str(args.seed),
               '--style', str(args.style), '--tone', str(args.tone),
               '--structure', str(args.structure), '--skin', str(args.skin)]
        if args.automask:
            cmd.append('--automask')
        if hist:
            cmd += ['--history', hist[1]]
        subprocess.run(cmd, check=True, capture_output=True)
        got = np.fromfile(out_path, dtype=np.float32).reshape(g['full_rows'], 16)

        diff = np.abs(ref - got)
        steps, bad = f16_adjacency_ok(ref.reshape(-1), got.reshape(-1))
        status = 'PASS' if bad == 0 and steps <= 1 else 'FAIL'
        failures += status == 'FAIL'
        print(f'[{name}] max|diff| {diff.max():.3e}  mean {diff.mean():.3e}  '
              f'max f16 步距 {steps}  越界元素 {bad}  -> {status}')
        if status == 'FAIL':
            worst = np.unravel_index(np.argmax(diff), diff.shape)
            print(f'    worst at {worst}: ref {ref[worst]:+.6f}  got {got[worst]:+.6f}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
