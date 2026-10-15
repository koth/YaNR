#!/usr/bin/env python3
"""Check native CUDA against CPU/PyTorch, including eager captures and graph replay.

Use the existing uv environment. No CUDA-enabled torch is needed: torch is only
the CPU reference. Student packages are read directly, without weight conversion.
student_v4 only runs the default controls at 320x320 and 512x512.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from nr_package import load_student_package


def run(exe, args):
    return subprocess.run([str(exe), *map(str, args)], check=True,
                          capture_output=True, text=True).stdout


def compare(label, ref, got, tolerance):
    if ref.shape != got.shape or not np.isfinite(got).all():
        raise AssertionError(f'{label}: invalid shape or nonfinite output')
    diff = np.abs(ref - got)
    maximum = float(diff.max())
    print(f'{label}: max {maximum:.3e}, mean {diff.mean():.3e}')
    if maximum > tolerance:
        raise AssertionError(f'{label}: max error {maximum} > {tolerance}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', default='engine/build-cuda')
    parser.add_argument('--cpu-build', default='engine/build/Release')
    parser.add_argument('--baseline-gpu', type=Path,
                        help='optional previous GPU executable for bitwise optimization regression')
    parser.add_argument('--weights', action='append')
    parser.add_argument('--image', default='teacher/samples/lake.png')
    parser.add_argument('--tolerance', type=float, default=2e-3)
    parser.add_argument('--quick', action='store_true', help='only 320x320 style 64')
    args = parser.parse_args()
    suffix = '.exe' if os.name == 'nt' else ''
    cpu = Path(args.cpu_build) / ('cpu_engine' + suffix)
    gpu = Path(args.build) / ('gpu_engine' + suffix)
    torch.set_num_threads(8)
    packages = args.weights or ['weights/student_v3', 'weights/student_v4']
    for package in packages:
        shape = json.loads((Path(package) / 'shape.json').read_text(encoding='utf-8'))
        has_moe = any(level['kind'] == 'moe' for level in shape['levels'])
        default_only = Path(package).resolve().name.casefold() == 'student_v4'
        if default_only:
            cases = [(320, 320, 64)] if args.quick else [(320, 320, 64), (512, 512, 64)]
        else:
            cases = [(320, 320, 64)] if args.quick else [
                (320, 320, 64), (320, 320, 0), (512, 512, 1.5),
                (320, 512, 127 if has_moe else 3), (137, 201, 1.5)]
        for width, height, style in cases:
            print(f'\n{package}, {width}x{height}, style {style}', flush=True)
            with tempfile.TemporaryDirectory(prefix='yanr_cuda_') as tmp:
                tmp = Path(tmp)
                features = tmp / 'features.bin'
                cpu_head, gpu_head, graph_head = (tmp / n for n in ('cpu.bin', 'gpu.bin', 'graph.bin'))
                controls = [] if default_only else ['--style', style]
                run(cpu, [args.image, '--weights', package, '--width', width, '--height', height,
                          *controls, '--threads', 8, '--dump-features', features,
                          '--out', cpu_head, '--dump-dir', tmp / 'cpu', '-o', tmp / 'cpu.png'])
                common = ['--weights', package, '--width', width, '--height', height, '--features', features]
                run(gpu, common + ['--out', gpu_head, '--dump-dir', tmp / 'gpu'])
                # Sorted GPU tags allow locating the first semantic divergence.
                for capture in sorted((tmp / 'cpu').glob('*.bin')):
                    other = tmp / 'gpu' / capture.name
                    if not other.exists():
                        raise AssertionError(f'missing GPU capture: {capture.name}')
                    compare(capture.stem, np.fromfile(capture, np.float32),
                            np.fromfile(other, np.float32), args.tolerance)
                ref = np.fromfile(cpu_head, np.float32)
                eager = np.fromfile(gpu_head, np.float32)
                compare('CPU / CUDA eager head', ref, eager, args.tolerance)
                print(run(gpu, common + ['--out', graph_head, '--bench', '--warmup', 2, '--repeats', 3]).strip())
                graph = np.fromfile(graph_head, np.float32)
                compare('CUDA eager / graph head', eager, graph, 0)
                again = tmp / 'again.bin'
                run(gpu, common + ['--out', again, '--bench', '--warmup', 2, '--repeats', 3])
                compare('CUDA graph repeat determinism', graph, np.fromfile(again, np.float32), 0)
                if args.baseline_gpu:
                    baseline = tmp / 'baseline.bin'
                    run(args.baseline_gpu, common + ['--out', baseline, '--bench', '--warmup', 2, '--repeats', 3])
                    compare('Previous / optimized CUDA head', np.fromfile(baseline, np.float32), graph, 0)
                model, _ = load_student_package(package, width, height, 'cpu')
                feats = np.fromfile(features, np.float32).reshape(-1, 16)
                with torch.no_grad():
                    expected = model(torch.from_numpy(feats)).numpy().reshape(-1)
                compare('PyTorch / CUDA head', expected, graph, args.tolerance)
                del model
    print('\nCHECK GPU PASS')


if __name__ == '__main__':
    main()
