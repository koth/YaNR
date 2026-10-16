#!/usr/bin/env python3
"""Separate instrumented CUDA Graph profiles; do not use these times as the gate."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re

from bench_shared_moe import command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path('tmp/shared_moe/prototype'))
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--size', default='320x320')
    parser.add_argument('--output', type=Path, default=Path('tmp/shared_moe/profile'))
    parser.add_argument('--engine', type=Path, default=Path('engine/build-cuda/gpu_engine.exe'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env['NR_CUDA_PROFILE'] = '1'
    if env.get('CUDA_PATH'):
        env['PATH'] = str(Path(env['CUDA_PATH']) / 'bin') + os.pathsep + env['PATH']
    width, height = args.size.split('x')
    times = []
    for i in range(5):
        output = command([args.engine, '--weights', args.weights, '--features', args.features,
                          '--width', width, '--height', height, '--bench', '--warmup', 10, '--repeats', 10],
                         env, args.output / f'round{i}.log')
        events = {name: float(ms) for ms, name in re.findall(r'CUDA profile: ([0-9.]+) ms (.+)', output)}
        if not events:
            raise RuntimeError('profile events missing')
        times.append(events)
    import statistics
    median = {n: statistics.median(t[n] for t in times) for n in times[0]}
    grouped = defaultdict(float)
    for name, ms in median.items():
        grouped[name.split(':')[0]] += ms
    report = dict(note='Instrumented Graph, includes event-node overhead; nested block/route/transition totals overlap child categories. Gate is measured separately.',
                  operation_median_ms=median, category_sums_ms=dict(grouped), rounds=times)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(grouped, indent=2))


if __name__ == '__main__':
    main()
