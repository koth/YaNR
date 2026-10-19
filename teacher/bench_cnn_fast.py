#!/usr/bin/env python3
"""Measured legacy CNN / same-weights grouped CNN / reduced Rep CNN comparison."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import statistics

import numpy as np
import torch

from bench_cnn_single import benchmark, clear_runtime, gpu_state, build_features_for_benchmark
from nr_cnn_unet_fast import CNNUNetFast, GroupedExpertFFN, load_fast_shape
from nr_geometry import geometry_from_valid
from nr_package import build_student, read_package
from single_frame_data import digest_file, read_rgb, RESAMPLE

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', default='320x320,512x512,1280x720')
    parser.add_argument('--train-sizes', default='320x320,512x512')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--train-repeats', type=int, default=5)
    parser.add_argument('--expert-execution', choices=('auto', 'split', 'tiled'),
                        help='explicit execution override for both fast variants')
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/cnn_unet_fast/speed.json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA Torch required')
    if os.name != 'nt' and os.environ.get('CUDA_VISIBLE_DEVICES') != '6':
        parser.error('remote GPU constraint: CUDA_VISIBLE_DEVICES=6')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    shape = ROOT / 'shapes/cnn_unet_fast.json'
    package = ROOT / 'weights/cnn_unet_moe8'
    g = geometry_from_valid(320, 320)
    original = build_student(package / 'shape.json', g)
    state, manifest = read_package(package, verify=True)
    original.load_state_dict(state)
    del state
    grouped = CNNUNetFast.from_legacy(original)
    torch.manual_seed(20261019)
    fast = CNNUNetFast(load_fast_shape(shape), g)
    if args.expert_execution:
        for model in (grouped, fast):
            model.cfg['expert_execution'] = args.expert_execution
            for module in model.modules():
                if isinstance(module, GroupedExpertFFN):
                    module.execution = args.expert_execution
    inference = copy.deepcopy(fast).eval().switch_to_deploy()
    cases = dict(legacy=original, grouped_legacy=grouped, fast=fast)
    image = read_rgb(ROOT / 'teacher/samples/t7/t7.jpg')
    report = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  gpu_before=gpu_state(), protocol=dict(backend='PyTorch eager', batch=1, precision='FP32',
                  tf32=False, deterministic='warn_only (legacy bilinear backward)', rounds=args.rounds,
                  repeats=args.repeats, warmup=args.warmup, train_warmup=2, train_repeats=args.train_repeats,
                  timing='synchronized full student only; no IO/transfers/teacher/optimizer/EMA',
                  backward='same raw-head MSE', fast_inference='Rep branches fused',
                  expert_execution=args.expert_execution or fast.cfg['expert_execution'],
                  fast_training='Rep branches unfused'),
                  weights=dict(legacy=manifest['model']['version'], grouped_legacy='identical legacy weights, packed',
                               fast='untrained, seeded 20261019'),
                  parameters=dict(grouped_legacy=grouped.parameter_ledger(), fast=fast.parameter_ledger()),
                  shape_sha256=digest_file(shape),
                  source_sha256={name: digest_file(ROOT / 'teacher' / name) for name in
                                 ('nr_cnn_unet_fast.py', 'nr_cnn_unet.py', 'nr_shared_moe.py',
                                  'bench_cnn_fast.py', 'bench_cnn_single.py')}, measurements=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    # Check a real saved network, not merely seeded single-block arithmetic.
    proxy = np.asarray(image.resize((320, 320), RESAMPLE), dtype=np.float32) / 255
    features = torch.from_numpy(build_features_for_benchmark(proxy, g)).cuda()
    heads = []
    for model in (original, grouped):
        model.cuda().eval()
        with torch.no_grad():
            heads.append(model(features).cpu())
        clear_runtime(model)
        model.cpu()
        torch.cuda.empty_cache()
    difference = (heads[0] - heads[1]).abs()
    report['same_weights_parity_320'] = dict(max_abs=float(difference.max()),
                                           mean_abs=float(difference.mean()),
                                           above_2e_3=int((difference > .002).sum()))
    print('PARITY', report['same_weights_parity_320'], flush=True)
    del features, heads
    save()
    for mode, sizes in (('forward', args.sizes), ('forward_backward', args.train_sizes)):
        for dimensions in filter(None, sizes.split(',')):
            w, h = map(int, dimensions.split('x'))
            g = geometry_from_valid(w, h)
            proxy = np.asarray(image.resize((w, h), RESAMPLE), dtype=np.float32) / 255
            features_np = build_features_for_benchmark(proxy, g)
            target_np = np.random.default_rng([20261019, w, h]).normal(0, .01, (g['full_rows'], 4)).astype('float32')
            for round_index in range(args.rounds):
                labels = list(cases)
                labels = labels[round_index % 3:] + labels[:round_index % 3]
                for label in labels:
                    model = inference if label == 'fast' and mode == 'forward' else cases[label]
                    model.g = g
                    row = dict(model=label, mode=mode, valid=[w, h],
                               field=[g['full_width'], g['full_height']], round=round_index)
                    print('BEGIN', dimensions, mode, round_index, label, flush=True)
                    features = target = None
                    try:
                        model.cuda()
                        features = torch.from_numpy(features_np).cuda()
                        target = torch.from_numpy(target_np).cuda()
                        row.update(benchmark(model, features, target, mode,
                                             args.warmup if mode == 'forward' else 2,
                                             args.repeats if mode == 'forward' else args.train_repeats))
                        print('DONE', label, '%.3fms %.3fGiB' %
                              (row['wall']['median'], row['peak_allocated_bytes'] / 2**30), flush=True)
                    except torch.cuda.OutOfMemoryError as error:
                        row['oom'] = str(error)
                    finally:
                        clear_runtime(model)
                        model.cpu()
                        del features, target
                        gc.collect()
                        torch.cuda.empty_cache()
                    report['measurements'].append(row)
                    save()
    summary = []
    keys = {(r['model'], r['mode'], tuple(r['valid'])) for r in report['measurements']}
    for label, mode, size in sorted(keys):
        rows = [r for r in report['measurements']
                if (r['model'], r['mode'], tuple(r['valid'])) == (label, mode, size)]
        successful = [r for r in rows if 'wall' in r]
        entry = dict(model=label, mode=mode, valid=list(size), oom_rounds=len(rows) - len(successful))
        if successful:
            entry.update(median_ms=statistics.median(r['wall']['median'] for r in successful),
                         round_medians_ms=[r['wall']['median'] for r in successful],
                         peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in successful))
        summary.append(entry)
    report.update(summary=summary, gpu_after=gpu_state())
    save()
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
