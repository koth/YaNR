#!/usr/bin/env python3
"""Separate memory diagnostic; explicitly distinguish own buffers/device sampling."""
import argparse
import json
import os
from pathlib import Path
import re

from bench_shared_moe import command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('weights/student_v44'))
    parser.add_argument('--features-root', type=Path, default=Path('tmp/shared_moe/fused_gate'))
    parser.add_argument('--output', type=Path, default=Path('tmp/shared_moe/memory'))
    parser.add_argument('--engine', type=Path, default=Path('engine/build-cuda/gpu_engine.exe'))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env['NR_CUDA_MEMORY'] = '1'; env.pop('NR_CUDA_PROFILE', None)
    if env.get('CUDA_PATH'):
        env['PATH'] = str(Path(env['CUDA_PATH']) / 'bin') + os.pathsep + env['PATH']
    results = []
    for size in ('320x320', '512x512', '1280x720'):
        width, height = size.split('x')
        for label, weights in [('baseline', args.baseline), ('candidate', args.candidate)]:
            output = command([args.engine, '--weights', weights, '--features', args.features_root / (size + '_features.bin'),
                              '--width', width, '--height', height, '--bench', '--warmup', 1, '--repeats', 1],
                             env, args.output / (size + '_' + label + '.log'))
            a = re.search(r'CUDA buffers: resident (\d+) bytes  allocation high-water (\d+) bytes', output)
            b = re.search(r'CUDA memory sample: total (\d+) initial-used (\d+) peak-used (\d+) delta (\d+) bytes', output)
            if not a or not b:
                raise ValueError('memory diagnostic missing')
            results.append(dict(size=size, model=label, own_resident_bytes=int(a[1]), own_allocation_high_water_bytes=int(a[2]),
                                device_total_bytes=int(b[1]), device_initial_used_bytes=int(b[2]),
                                device_sampled_peak_used_bytes=int(b[3]), device_sampled_peak_delta_bytes=int(b[4])))
    report = dict(note='Separate diagnostic, not forward timing. Own Buffer allocation high-water is exact but excludes CUDA/cuBLAS/Graph internal allocations. Device sampling includes other applications and samples allocation/operation boundaries; it is not an exclusive per-process complete VRAM peak.', results=results)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
