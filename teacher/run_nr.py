#!/usr/bin/env python3
# Runs the 71-block network on a real input and reports timings - the torch counterpart of
# ports/browser-webgpu/web/bench.html, with the same deterministic input so the two can be compared
# value for value.
#
#   NR_WEIGHTS=~/klss/models/nr-synth python3 run_nr.py --width 512 --height 512 --frames 5
#   python3 run_nr.py --width 320 --height 320 --frames 3 --dump out/
#
# The features are the bench's: a smooth field in lanes 0-3 and xorshift noise in lanes 4-6.

import argparse
import math
import os
import statistics
import sys
import time

import numpy as np
import torch

from nr_geometry import geometry_from_valid
from nr_model import Model
from nr_network import Network
import nr_torch


def bench_features(g):
    """The input bench.html generates: f32 [fullRows][16], the LCG seeded with 12345."""
    width, height = g['full_width'], g['full_height']
    features = np.zeros((width * height, 16), dtype=np.float32)
    seed = 12345
    for y in range(height):
        for x in range(width):
            at = y * width + x
            u = x / width
            v = y / height
            features[at, 0] = 0.4 + 0.4 * math.sin(u * 7) * math.cos(v * 5)
            features[at, 1] = 0.3 + 0.5 * u * v
            features[at, 2] = 0.2 + 0.6 * (1 - u) * v
            features[at, 3] = 1
            for lane in range(4, 7):
                seed = (seed * 1103515245 + 12345) & 0xffffffff
                features[at, lane] = (seed / 4294967296) * 2 - 1
    return features


def main():
    parser = argparse.ArgumentParser(description='run the OpenDLSS-NR network in torch')
    parser.add_argument('--width', type=int, default=512, help='valid width')
    parser.add_argument('--height', type=int, default=512, help='valid height')
    parser.add_argument('--frames', type=int, default=5, help='timed frames after one warm-up')
    parser.add_argument('--weights', default=os.environ.get('NR_WEIGHTS'), help='model directory')
    parser.add_argument('--shape', default='auto',
                        help="'auto'(默认,按包内 manifest 自动探测教师/学生)/ 'student'(要求学生包)")
    parser.add_argument('--dump', default=None, help='directory for features/head .bin dumps')
    parser.add_argument('--eager', action='store_true',
                        help='run frame by frame instead of replaying a captured CUDA graph')
    args = parser.parse_args()

    if not args.weights:
        print('no weights: pass --weights or set NR_WEIGHTS')
        return 2

    device = nr_torch.DEVICE
    print(f'run_nr: {args.width}x{args.height}, frames {args.frames}, device {device}')

    g = geometry_from_valid(args.width, args.height)
    print(f"field {g['full_width']}x{g['full_height']} ({g['full_rows']} rows), levels "
          + ' '.join(f"{l['width']}x{l['height']}" for l in g['levels'])
          + f", vit tokens {g['vit_tokens']} (padded {g['padded_vit_tokens']})")

    # ---- 模型加载(7.2):manifest 自动探测;--shape student 显式要求学生包。
    from nr_package import is_student_package, load_student_package
    student_mode = is_student_package(args.weights)
    if args.shape == 'student' and not student_mode:
        raise SystemExit(f'{args.weights} 不是学生模型包(缺 manifest model.kind=student)')
    if student_mode:
        model, manifest = load_student_package(args.weights, args.width, args.height, device)
        print(f"model: {manifest['model']['version']}  ({manifest['model']['params']:,} params)")
        boundaries = {}                                    # boundary 捕获是教师调试口径

        def forward(feats):
            with torch.no_grad():                           # nn.Module 参数会挂 autograd
                return model(feats)
    else:
        tmodel = Model(device).load(args.weights)
        print(f'model: teacher-nr-{tmodel.block_count}blk ({tmodel.block_count} blocks loaded)')
        network = Network(tmodel, g)
        boundaries = network.boundaries

        def forward(feats):
            return network.record(feats)

    features_np = bench_features(g)
    features = torch.from_numpy(features_np).to(nr_torch.DEVICE)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # The frame is a fixed sequence of launches, so capture it into a CUDA graph after warm-up: a replay
    # carries neither the Python per-call overhead nor the launch cost (unless --eager asks for both).
    # 学生前向的窗注意力含掩码布尔索引(nonzero 同步),不可捕获,退回 eager。
    use_graph = device.type == 'cuda' and not args.eager and not student_mode
    if student_mode and device.type == 'cuda' and not args.eager:
        print('note: 学生前向含掩码布尔索引,不支持 CUDA graph 捕获,退回 eager 计时')
    head = None
    if use_graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            forward(features)                              # warm-up: JITs, caches, the allocator
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            head = forward(features)

        def run():
            graph.replay()
            return head
    else:
        def run():
            return forward(features)

    print('warm-up frame')
    sync()
    head = run()
    sync()

    times = []
    for frame in range(args.frames):
        start = time.perf_counter()
        head = run()
        sync()
        times.append((time.perf_counter() - start) * 1000.0)
        print(f'  frame {frame + 1}: {times[-1]:.1f} ms')

    head_np = head.float().cpu().numpy()
    channel0 = head_np.reshape(-1, 4)[:, 0]
    finite = np.isfinite(channel0)
    nonfinite = int(channel0.size - finite.sum())
    values = channel0[finite]
    print('')
    print(f'frames: min {min(times):.2f} ms  median {statistics.median(times):.2f} ms  '
          f'mean {sum(times) / len(times):.2f} ms')
    if values.size:
        print(f'head R residual over {channel0.size} px: min {values.min():.3e}  max {values.max():.3e}  '
              f'mean {values.mean():.3e}  non-finite {nonfinite}')
    all_finite = bool(np.isfinite(head_np).all())
    print(f'head all finite: {all_finite}')

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        features_np.astype('<f4').tofile(os.path.join(args.dump, 'features.f32.bin'))
        head_np.astype('<f4').tofile(os.path.join(args.dump, 'head.f32.bin'))
        print(f'dumped features/head f32 to {args.dump}')
        bnd_dir = os.path.join(args.dump, 'bnd')
        os.makedirs(bnd_dir, exist_ok=True)
        for name, tensor in boundaries.items():
            values = tensor.half().cpu().numpy().astype('<f2')
            values.tofile(os.path.join(bnd_dir, f'{name}.f16bin'))
        print(f'dumped {len(boundaries)} boundaries to {bnd_dir}')

    ok = all_finite and nonfinite == 0
    print('RUN PASS' if ok else 'RUN FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
