#!/usr/bin/env python3
"""Reproducible FP32 student training time/peak-memory probe, without a teacher.

--baseline-ref replays the committed implementation without changing the tree.
Includes forward, task loss, backward, AdamW, EMA and bias controller. It does
not represent end-to-end online teacher/data throughput; use train_distill logs
for that. Run probes sequentially on an otherwise idle GPU.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape', default='shapes/student_shared_moe_v1.json')
    p.add_argument('--size', type=int, default=320)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--baseline-ref')
    p.add_argument('--tile', type=int, help='Diagnostic override for the committed tile implementation')
    p.add_argument('--telemetry', action='store_true')
    p.add_argument('--profile', action='store_true', help='Diagnostic operator profile, not a timing result')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261017)
    import nr_shared_moe
    import moe_training
    if args.baseline_ref:
        root = Path(__file__).resolve().parent.parent
        for filename, target in [('nr_shared_moe.py', nr_shared_moe), ('moe_training.py', moe_training)]:
            source = subprocess.check_output(['git', 'show', args.baseline_ref + ':teacher/' + filename],
                                             cwd=root).decode('utf-8')
            if args.tile and filename == 'nr_shared_moe.py':
                source = source.replace('tile = 4096', 'tile = ' + str(args.tile))
            module = types.ModuleType('baseline_' + filename[:-3])
            exec(compile(source, filename, 'exec'), module.__dict__)
            if filename == 'nr_shared_moe.py':
                nr_shared_moe.SharedRoutedBlock = module.SharedRoutedBlock
                nr_shared_moe.SharedRoutedLinear = module.SharedRoutedLinear
                nr_shared_moe.RoutedModule = module.RoutedModule
            else:
                moe_training.BiasController = module.BiasController
                moe_training.update_ema = module.update_ema
        student_source = subprocess.check_output(['git', 'show', args.baseline_ref + ':teacher/nr_student.py'], cwd=root).decode('utf-8')
        module = types.ModuleType('nr_student')
        module.__file__ = str(root / 'teacher/nr_student.py')
        exec(compile(student_source, module.__file__, 'exec'), module.__dict__)
        sys.modules['nr_student'] = module
    from nr_student import StudentNetwork, load_student_shape
    from nr_geometry import geometry_from_valid
    g = geometry_from_valid(args.size, args.size)
    model = StudentNetwork(load_student_shape(args.shape), g).cuda().train()
    controller = moe_training.BiasController(model)
    for module in controller.modules.values():
        module.training_metrics_enabled = args.telemetry or bool(args.baseline_ref)
    ema = {n: v.detach().clone() for n, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, **({'fused': True} if not args.baseline_ref else {}))
    rng = np.random.default_rng([20261017, args.size, args.batch])
    features = torch.from_numpy(rng.normal(0, .2, (args.batch * g['full_rows'], 16)).astype('float32')).cuda()
    target = torch.from_numpy(rng.normal(0, .1, (args.batch * g['full_rows'], 4)).astype('float32')).cuda()
    report = dict(size=args.size, batch=args.batch, field=[g['full_width'], g['full_height']],
                  reference=args.baseline_ref or 'working-tree', tile_override=args.tile,
                  telemetry=args.telemetry or bool(args.baseline_ref), precision='FP32 TF32 off', seed=20261017,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, samples=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for step in range(args.warmup + args.repeats):
            profiler = None
            if args.profile and step == args.warmup:
                profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA])
                profiler.__enter__()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            controller.begin(step)
            output = model(features, batch=args.batch)
            controller.collect()
            loss = (output - target).square().mean()
            loss.backward()
            if args.telemetry or args.baseline_ref:
                controller.capture_gradients()
            optimizer.step()
            controller.apply(step)
            moe_training.update_ema(ema, model.state_dict(), .99)
            torch.cuda.synchronize()
            row = dict(ms=(time.perf_counter() - t0) * 1000,
                       peak_allocated=torch.cuda.max_memory_allocated(),
                       peak_reserved=torch.cuda.max_memory_reserved(), loss=float(loss))
            print('step', step, row, flush=True)
            if profiler:
                profiler.__exit__(None, None, None)
                report['profile'] = profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=22)
                print(report['profile'], flush=True)
            if step >= args.warmup:
                report['samples'].append(row)
        report['median_ms'] = statistics.median(r['ms'] for r in report['samples'])
        report['peak_allocated'] = max(r['peak_allocated'] for r in report['samples'])
        report['peak_reserved'] = max(r['peak_reserved'] for r in report['samples'])
    except torch.cuda.OutOfMemoryError as exc:
        report['oom'] = str(exc)
        report['peak_allocated'] = torch.cuda.max_memory_allocated()
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
