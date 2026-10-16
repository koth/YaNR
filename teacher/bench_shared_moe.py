#!/usr/bin/env python3
"""Same-device native FP32 forward benchmark for v44 and shared-routed MoE.

Use --candidate after exporting the prototype. Without it, record the baseline.
Image decoding and feature construction happen once, outside measured forwards.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
import re
from pathlib import Path
import statistics
import subprocess

from audit_moe import audit_package, sha256
from nr_geometry import geometry_from_valid


ROOT = Path(__file__).resolve().parent.parent


def command(args, env, log=None):
    result = subprocess.run([str(arg) for arg in args], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding='utf-8', errors='replace')
    if log:
        log.write_text(result.stdout, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('command failed: ' + subprocess.list2cmdline([str(a) for a in args]) + '\n' + result.stdout)
    return result.stdout


def summarize(samples):
    values = sorted(samples)
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('invalid forward time samples')
    return dict(median_ms=statistics.median(values), min_ms=values[0],
                p95_ms=values[math.ceil(len(values) * .95) - 1], count=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, default=ROOT / 'engine/build-cuda/gpu_engine.exe')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'weights/student_v44')
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--image', type=Path, default=ROOT / 'teacher/samples/t1/t1.png')
    parser.add_argument('--sizes', default='320x320,512x512,1280x720')
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/shared_moe/baseline')
    parser.add_argument('--reference-python', type=Path, default=ROOT / 'tmp/teacher-cuda-venv/Scripts/python.exe')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.rounds) < 1:
        parser.error('warmup/repeats/rounds must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '8'
    for name in ('NR_INT8', 'NR_PROFILE', 'NR_ATTN_PROFILE', 'NR_CUDA_PROFILE', 'NR_CUDA_MEMORY'):
        env.pop(name, None)
    if env.get('CUDA_PATH'):
        env['PATH'] = str(Path(env['CUDA_PATH']) / 'bin') + os.pathsep + env['PATH']
    models = [('baseline', args.baseline.resolve())]
    if args.candidate:
        models.append(('candidate', args.candidate.resolve()))
    identities = {}
    for label, weights in models:
        audit = audit_package(weights, verify=True)
        identities[label] = {key: audit[key] for key in ('package', 'totals', 'budget', 'public_by_category')}
    sources = ['engine/src/engine_cuda.cu', 'engine/src/shared_moe_cuda.cuh', 'engine/src/engine_package.cpp',
               'engine/tools/engine_main.cpp', 'engine/src/lanes.cpp']
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  commit=command(['git', 'rev-parse', 'HEAD'], env).strip(),
                  source_sha256={path: sha256(ROOT / path) for path in sources},
                  engine=dict(path=str(args.engine.resolve()), sha256=sha256(args.engine)),
                  models=identities, warmup=args.warmup, repeats=args.repeats, rounds=args.rounds,
                  precision='FP32; TF32 off; native CUDA graph',
                  timing='Complete GPU forward CUDA events; excludes input/output transfers and image processing',
                  controls=dict(style=64, tone=1, structure=1, skin=1, automask=1, seed=12345),
                  image=dict(path=str(args.image.resolve()), sha256=sha256(args.image)),
                  gpu=command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                               '--format=csv,noheader'], env).strip(), measurements=[])
    if args.reference_python.is_file():
        report['reference_environment'] = command([args.reference_python, '-c',
            "import torch,numpy,PIL; print('torch',torch.__version__,'CUDA',torch.version.cuda,"
            "'numpy',numpy.__version__,'Pillow',PIL.__version__)"], env).strip()
    report_path = args.output / 'report.json'

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    save()
    for text in args.sizes.split(','):
        width, height = map(int, text.lower().split('x'))
        g = geometry_from_valid(width, height)
        features = args.output / (text + '_features.bin')
        command([args.engine, args.image, '--weights', args.baseline, '--width', width, '--height', height,
                 '--png', args.output / (text + '_baseline.png'), '--dump-features', features,
                 '--bench', '--warmup', 1, '--repeats', 1], env, args.output / (text + '_prepare.log'))
        features_hash = sha256(features)
        for round_index in range(args.rounds):
            for label, weights in models:
                tag = '{}_{}_{}'.format(text, round_index, label)
                times_path = args.output / (tag + '_times.txt')
                invocation = [args.engine, '--weights', weights, '--features', features,
                              '--width', width, '--height', height, '--bench', '--warmup', args.warmup,
                              '--repeats', args.repeats, '--bench-times', times_path]
                output = command(invocation, env, args.output / (tag + '.log'))
                samples = [float(v) for v in times_path.read_text().splitlines()]
                if len(samples) != args.repeats:
                    raise ValueError('incorrect timing sample count: ' + str(times_path))
                row = dict(model=label, round=round_index, valid=[width, height],
                           field=[g['full_width'], g['full_height']], features_sha256=features_hash,
                           command=[str(a) for a in invocation], samples_ms=samples, **summarize(samples))
                memory = re.search(r'CUDA buffers: resident (\d+) bytes  allocation high-water (\d+) bytes', output)
                if memory:
                    row['cuda_buffer_bytes'] = dict(resident=int(memory[1]), allocation_high_water=int(memory[2]),
                                                   excludes='CUDA/cuBLAS/Graph internal allocations; see separate memory diagnostics')
                report['measurements'].append(row)
                save()
                print('{} round {} {} median {:.3f} ms p95 {:.3f} ms'.format(
                    text, round_index, label, row['median_ms'], row['p95_ms']), flush=True)
                if 'TF32 off' not in output:
                    raise ValueError('expected native FP32 CUDA backend was not reported')
    if args.candidate:
        gates = []
        for text in args.sizes.split(','):
            valid = list(map(int, text.lower().split('x')))
            rows = [r for r in report['measurements'] if r['valid'] == valid]
            old = [r for r in rows if r['model'] == 'baseline']
            new = [r for r in rows if r['model'] == 'candidate']
            ratio = statistics.median(n['median_ms'] / o['median_ms'] for o, n in zip(old, new))
            p95_ratio = statistics.median(r['p95_ms'] for r in new) / statistics.median(r['p95_ms'] for r in old)
            gates.append(dict(valid=valid, median_ratio=ratio, p95_ratio=p95_ratio,
                              passed=ratio <= 1.0 and p95_ratio <= 1.1))
        budget = identities['candidate']['budget']
        report['gate'] = dict(by_size=gates, budget_pass=budget.get('pass', False),
                              protocol_pass=args.warmup >= 10 and args.repeats >= 100 and args.rounds >= 3,
                              required_sizes_present=all(size in [g['valid'] for g in gates]
                                                         for size in ([320, 320], [512, 512], [1280, 720])))
        report['gate']['passed'] = (all(g['passed'] for g in gates) and
                                    report['gate']['budget_pass'] and report['gate']['protocol_pass'] and
                                    report['gate']['required_sizes_present'])
        save()
        print(json.dumps(report['gate'], indent=2), flush=True)
    print('report:', report_path, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
