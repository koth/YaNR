#!/usr/bin/env python3
"""Same-device eager FP32 comparison: single56m, CNN U-Net8 and routed7 top-1.

Full student forwards and synthetic-loss forward/backward only. Excludes
teacher, image IO, transfers, optimizer, EMA and trainer-specific losses.
Uses saved packages where available; no model weight is changed by this probe.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import warnings

import numpy as np
import torch

from nr_geometry import geometry_from_valid
from nr_package import build_student, read_package
from single_frame_data import digest_file, read_rgb, RESAMPLE


ROOT = Path(__file__).resolve().parent.parent
CASES = (
    ('single56m', 'tmp/single_frame_pipeline_final/package', 'shapes/cnn_single_56m.json'),
    ('cnn_unet8', 'weights/cnn_unet_moe8', 'shapes/cnn_unet_moe8.json'),
    ('moe7_top1', 'weights/student_v45i', 'shapes/student_routed7_v1.json'),
)


def gpu_state():
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.used,utilization.gpu,'
                             'pstate,clocks.sm,temperature.gpu,power.draw', '--format=csv,noheader'],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def summarize(values):
    return dict(median=statistics.median(values), p95=float(np.percentile(values, 95)),
                minimum=min(values), maximum=max(values), count=len(values))


def active_ledger(model):
    if hasattr(model, 'parameter_ledger'):
        return {k: v for k, v in model.parameter_ledger().items() if k != 'blocks'}
    total = sum(p.numel() for p in model.parameters())
    routed, active = 0, 0
    if getattr(model, 'shared_routed', False):
        from nr_shared_moe import RoutedModule
        for module in model.modules():
            if isinstance(module, RoutedModule):
                for param in module.parameters(recurse=False):
                    if param.ndim >= 2 and param.shape[0] == 7:
                        routed += param.numel()
                        active += param.numel() // 7
    else:
        for module in model.modules():
            if module.__class__.__name__ == 'ExpertFFN':
                sizes = [sum(p.numel() for p in e.parameters()) for e in module.bank]
                routed += sum(sizes)
                active += sizes[0]
    return dict(total=total, public=total - routed, routed=routed,
                active=total - routed + active, active_routed=active,
                definition='all public parameters + one expert per block per spatial position')


def routing_summary(model):
    layers = []
    for name, module in model.named_modules():
        picks = None
        experts = 0
        if module.__class__.__name__ == 'Top1FFN' and module.last_stats is not None:
            counts = module.last_stats[0].detach().cpu().tolist()
        else:
            if module.__class__.__name__ == 'ExpertFFN' and module.last_stats is not None:
                picks, probabilities, _ = module.last_stats
                experts = probabilities.shape[1]
            elif module.__class__.__name__ in ('SharedRoutedBlock', 'SharedRoutedLinear') and module.moe_stats:
                picks, probabilities, _ = module.moe_stats[-1]
                experts = probabilities.shape[-1]
            if picks is None:
                continue
            counts = torch.bincount(picks.reshape(-1), minlength=experts).cpu().tolist()
        layers.append(dict(name=name, counts=counts, used=sum(c > 0 for c in counts),
                           max_fraction=max(counts) / max(sum(counts), 1)))
    return layers


def clear_runtime(model):
    for module in model.modules():
        if hasattr(module, 'last_stats'):
            module.last_stats = None
        for key in ('captures', 'moe_stats', 'training_metric_records', 'diversity_terms'):
            if hasattr(module, key):
                getattr(module, key).clear()
    model.zero_grad(set_to_none=True)


def benchmark(model, features, target, mode, warmup, repeats):
    model.train(mode == 'forward_backward')
    wall, events = [], []
    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    peak_allocated = peak_reserved = 0
    first_loss = None
    for step in range(warmup + repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        if mode == 'forward':
            # WindowAttention caches geometry tensors across calls; inference
            # tensors in that cache cannot later participate in autograd.
            with torch.no_grad():
                result = model(features, batch=1)
        else:
            model.zero_grad(set_to_none=True)
            result = model(features, batch=1)
            loss = (result - target).square().mean()
            loss.backward()
        end.record()
        end.synchronize()
        elapsed = (time.perf_counter() - begin) * 1000
        if step == 0:
            if not bool(torch.isfinite(result).all()):
                raise FloatingPointError('nonfinite forward')
            if mode != 'forward':
                if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('nonfinite gradient')
                first_loss = float(loss.detach())
        if step >= warmup:
            wall.append(elapsed)
            events.append(start.elapsed_time(end))
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
        del result
        if mode != 'forward':
            del loss
    row = dict(wall_ms=wall, cuda_event_ms=events, wall=summarize(wall), cuda_event=summarize(events),
               peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
               resident_before_bytes=baseline_bytes, routing=routing_summary(model), synthetic_loss=first_loss)
    clear_runtime(model)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', default='320x320,512x512,1280x720')
    parser.add_argument('--train-sizes', default='320x320,512x512')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--train-warmup', type=int, default=2)
    parser.add_argument('--train-repeats', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--tf32', type=int, choices=[0, 1], default=0)
    parser.add_argument('--fresh', action='store_true', help='use seeded initialization for all three models')
    parser.add_argument('--image', type=Path, default=ROOT / 'teacher/samples/t7/t7.jpg')
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/cnn_single_speed/report.json')
    args = parser.parse_args()
    if min(args.repeats, args.train_repeats, args.rounds) < 1 or min(args.warmup, args.train_warmup) < 0:
        parser.error('repeats/rounds must be positive, warmups nonnegative')
    if not torch.cuda.is_available():
        parser.error('requires CUDA Torch; Windows: tmp/teacher-cuda-venv/Scripts/python.exe')
    if os.name != 'nt' and os.environ.get('CUDA_VISIBLE_DEVICES') != '6':
        parser.error('remote GPU constraint: CUDA_VISIBLE_DEVICES=6')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Existing CNN uses CUDA bilinear backward, which lacks a deterministic
    # implementation. Report warnings; do not patch/replace its operators.
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings('once', message='.*does not have a deterministic implementation.*')
    original = read_rgb(args.image)
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), torch=torch.__version__,
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), gpu_before=gpu_state(),
                  protocol=dict(batch=1, backend='PyTorch eager', precision='FP32', matmul_tf32=bool(args.tf32),
                                cudnn_tf32=False, cudnn_benchmark=False, deterministic='warn_only',
                                inference_context='no_grad',
                                cpu_threads=8, rounds=args.rounds, warmup=args.warmup, repeats=args.repeats,
                                train_warmup=args.train_warmup, train_repeats=args.train_repeats,
                                backward_loss='same synthetic raw-head MSE, no optimizer/EMA/teacher/feature loss',
                                timing='per-call synchronized wall + CUDA events; complete model, no IO/H2D/D2H'),
                  image=dict(path=str(args.image), sha256=digest_file(args.image)), models={}, measurements=[])
    files = ('nr_cnn_single.py', 'nr_cnn_unet.py', 'nr_student.py', 'nr_shared_moe.py',
             'nr_geometry.py', 'nr_precision.py', 'bench_cnn_single.py')
    report['source_sha256'] = {name: digest_file(ROOT / 'teacher' / name) for name in files}
    models = {}
    for label, package, shape in CASES:
        package, shape = ROOT / package, ROOT / shape
        if not args.fresh:
            shape = package / 'shape.json'
        torch.manual_seed(20261019)
        model = build_student(shape, geometry_from_valid(320, 320))
        info = dict(shape=str(shape), shape_sha256=digest_file(shape))
        if args.fresh:
            info.update(weight_source='seeded initialization', seed=20261019)
        else:
            state, manifest = read_package(package, verify=True)
            model.load_state_dict(state)
            del state
            info.update(package=str(package), manifest_sha256=digest_file(package / 'manifest.json'),
                        weight_source=manifest['model']['version'], step=manifest['model']['step'])
        info['parameters'] = active_ledger(model)
        report['models'][label] = info
        models[label] = model
        print(label, json.dumps(info), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    save()
    for mode, sizes, warmup, repeats in (
            ('forward', args.sizes, args.warmup, args.repeats),
            ('forward_backward', args.train_sizes, args.train_warmup, args.train_repeats)):
        for dimensions in sizes.split(','):
            if not dimensions:
                continue
            w, h = map(int, dimensions.lower().split('x'))
            g = geometry_from_valid(w, h)
            proxy = np.asarray(original.resize((w, h), RESAMPLE), dtype=np.float32) / 255
            features_np = build_features_for_benchmark(proxy, g)
            features_hash = hashlib.sha256(features_np.tobytes()).hexdigest()
            target_np = np.random.default_rng([20261019, w, h]).normal(0, .01, (g['full_rows'], 4)).astype(np.float32)
            for round_index in range(args.rounds):
                # Rotate order; only one model resides on GPU at a time.
                labels = list(models)
                labels = labels[round_index % len(labels):] + labels[:round_index % len(labels)]
                for label in labels:
                    model = models[label]
                    row = dict(model=label, mode=mode, valid=[w, h],
                               field=[g['full_width'], g['full_height']], round=round_index,
                               features_sha256=features_hash)
                    print('BEGIN', mode, dimensions, 'round', round_index, label, flush=True)
                    features = target = None
                    try:
                        model.to('cuda')
                        model.g = g
                        model.capture_enabled = False
                        features = torch.from_numpy(features_np).cuda()
                        target = torch.from_numpy(target_np).cuda()
                        row.update(benchmark(model, features, target, mode, warmup, repeats))
                        print('DONE', label, mode, dimensions, 'median %.3f ms, peak %.3f GiB' %
                              (row['wall']['median'], row['peak_allocated_bytes'] / 2**30), flush=True)
                    except torch.cuda.OutOfMemoryError as exc:
                        row.update(oom=str(exc), peak_allocated_bytes=torch.cuda.max_memory_allocated())
                        print('OOM', label, mode, dimensions, flush=True)
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
    for label, mode, valid in sorted(keys):
        rows = [r for r in report['measurements'] if (r['model'], r['mode'], tuple(r['valid'])) == (label, mode, valid)]
        valid_rows = [r for r in rows if 'wall' in r]
        result = dict(model=label, mode=mode, valid=list(valid), oom_rounds=len(rows) - len(valid_rows))
        if valid_rows:
            result.update(median_ms=statistics.median(r['wall']['median'] for r in valid_rows),
                          round_medians_ms=[r['wall']['median'] for r in valid_rows],
                          cuda_event_median_ms=statistics.median(r['cuda_event']['median'] for r in valid_rows),
                          peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in valid_rows))
        summary.append(result)
    report.update(summary=summary, gpu_after=gpu_state())
    save()
    print('SUMMARY', json.dumps(summary), flush=True)
    print('report:', args.output, flush=True)


def build_features_for_benchmark(proxy, geometry):
    from run_image import build_features
    return build_features(proxy, geometry, 12345, 2., 1., 1., -1., True)


if __name__ == '__main__':
    main()
