#!/usr/bin/env python3
"""Layerwise FP32 Python/native CUDA parity for the untrained prototype.

Compares all routing decisions exactly and gates/FFN/QKV/attention/transitions/
head numerically. Also compares eager/Graph/repeat heads bit for bit. Input
features are shared between paths, never rebuilt independently.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from nr_geometry import geometry_from_valid
from nr_package import load_student_package
from audit_moe import sha256


def difference(a, b):
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(max_abs=float(error.max()), mean_abs=float(error.mean()),
                p99_abs=float(np.quantile(error, .99)),
                rms=float(np.sqrt(np.mean(error ** 2))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path('tmp/shared_moe/prototype'))
    parser.add_argument('--engine', type=Path, default=Path('engine/build-cuda/gpu_engine.exe'))
    parser.add_argument('--output', type=Path, default=Path('tmp/shared_moe/parity'))
    parser.add_argument('--sizes', default='320x320,512x512,320x512,137x201')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--head-only', action='store_true', help='Skip intermediate capture and eager check for large native images')
    parser.add_argument('--routes-only', action='store_true', help='Capture all router choices/logits/gates at large sizes without retaining feature maps')
    parser.add_argument('--atol', type=float, default=3e-5, help='Absolute FP32 tolerance, raw head before division by 4')
    parser.add_argument('--images', nargs='*', type=Path, help='Native images; build actual default-condition features once with the native image pipeline')
    parser.add_argument('--feature-weights', type=Path, default=Path('weights/student_v44'))
    parser.add_argument('--features-root', type=Path, help='Read SIZE_features.bin saved by the benchmark for exact load/cost diagnostics')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env.get('CUDA_PATH'):
        env['PATH'] = str(Path(env['CUDA_PATH']) / 'bin') + os.pathsep + env['PATH']
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if args.routes_only:
        env['NR_CUDA_CAPTURE'] = 'routes'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    reports = []
    failures = []
    sizes = args.sizes.split(',')
    if args.images:
        from PIL import Image
        sizes = ['{}x{}'.format(*Image.open(p).size) for p in args.images]
    for index, size in enumerate(sizes):
        width, height = map(int, size.split('x'))
        tag = f'{index}_{args.images[index].parent.name}_{size}' if args.images else size
        out = args.output / tag; out.mkdir(exist_ok=True)
        g = geometry_from_valid(width, height)
        rng = np.random.default_rng([args.seed, index])
        data = rng.normal(0, .2, (g['full_rows'], 16)).astype(np.float32)
        features = out / 'features.bin'; data.tofile(features)
        if args.features_root:
            data = np.fromfile(args.features_root / (size + '_features.bin'), np.float32).reshape(g['full_rows'], 16)
            data.tofile(features)
        proxy = None
        if args.images:
            image = args.images[index]
            proc = subprocess.run([str(args.engine), str(image), '--weights', str(args.feature_weights),
                                   '--dump-features', str(features), '--proxy-png', str(out / 'proxy.png'),
                                   '--png', str(out / 'feature_pipeline.png')], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
            (out / 'prepare.log').write_text(proc.stdout, encoding='utf-8')
            if proc.returncode:
                raise RuntimeError(proc.stdout)
            data = np.fromfile(features, np.float32).reshape(g['full_rows'], 16)
            proxy = np.asarray(Image.open(out / 'proxy.png').convert('RGB'), np.float32) / 255
            proxy.tofile(out / 'proxy.bin')
        cmd = [str(args.engine), '--weights', str(args.weights), '--features', str(features),
               '--width', str(width), '--height', str(height)]
        modes = [('eager', ['--dump-dir', str(out / 'capture')]),
                                ('graph', ['--bench', '--warmup', '2', '--repeats', '2']),
                                ('repeat', ['--bench', '--warmup', '2', '--repeats', '2'])]
        if args.head_only and not args.routes_only:
            modes = modes[1:]
        for label, options in modes:
            if proxy is not None and label == 'graph':
                options = options + ['--proxy', str(out / 'proxy.bin'), '--png', str(out / 'native.png')]
            proc = subprocess.run(cmd + options + ['--out', str(out / (label + '.bin'))], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
            (out / (label + '.log')).write_text(proc.stdout, encoding='utf-8')
            if proc.returncode:
                raise RuntimeError(proc.stdout)
        graph = np.fromfile(out / 'graph.bin', np.float32)
        eager = graph if args.head_only and not args.routes_only else np.fromfile(out / 'eager.bin', np.float32)
        repeated = np.fromfile(out / 'repeat.bin', np.float32)
        if not np.array_equal(eager, graph) or not np.array_equal(graph, repeated):
            failures.append(size + ': Graph/eager/repeat differs')
        model, _ = load_student_package(args.weights, width, height, 'cuda')
        model.capture_enabled = not (args.head_only or args.routes_only)
        for module in model.modules():
            if hasattr(module, 'router_bias'):
                module.capture_enabled = not args.head_only or args.routes_only
                if args.routes_only:
                    module.capture_filter = {'logits', 'pick', 'gate', 'assign'}
        with torch.inference_mode():
            result = model(torch.from_numpy(data).cuda())
        expected_head = result.cpu().numpy().reshape(-1)
        composite = None
        if proxy is not None:
            python_head = expected_head.reshape(g['full_height'], g['full_width'], 4)[:height, :width, :3]
            python_png = (np.clip(proxy.astype(np.float64) + python_head.astype(np.float64) * .25, 0, 1) * 255 + .5).astype(np.uint8)
            native_png = np.asarray(Image.open(out / 'native.png').convert('RGB'))
            delta = np.abs(python_png.astype(np.int16) - native_png.astype(np.int16))
            composite = dict(max_code_error=int(delta.max()), changed_channel_fraction=float((delta != 0).mean()))
            if delta.max() > 1:
                failures.append(size + ': composite differs by more than 1 code value')
        if not np.isfinite(graph).all() or not np.allclose(graph, expected_head, rtol=2e-4, atol=args.atol):
            failures.append(size + ': head differs')
        expected = {}
        loads = {}
        for name, value in model.captures.items():
            expected[name] = value
        labels = dict(logits='logits', pick='pick', gate='gate', ffn_shared='ffn_shared',
                      ffn_routed='ffn_routed', ffn='ffn', qkv='qkv', attention='att')
        for name, module in model.named_modules():
            if hasattr(module, 'router_bias') and (not args.head_only or args.routes_only):
                loads[name] = torch.bincount(module.captures['assign'], minlength=16).cpu().tolist()
                native_load = np.fromfile(out / 'capture' / ('load-' + name + '.bin'), np.float32).astype(np.int64).tolist()
                if loads[name] != native_load:
                    failures.append(size + ': ' + name + ' row load differs')
                for key, label in labels.items():
                    if key in module.captures:
                        expected[label + '-' + name] = module.captures[key]
                if not args.routes_only:
                    expected[('blk-' if 'blocks.' in name else 'output-') + name] = module.captures['output']
        layers = []
        for name, tensor in expected.items():
            native = np.fromfile(out / 'capture' / (name + '.bin'), np.float32)
            python = tensor.detach().cpu().numpy().reshape(-1)
            stats = difference(native, python)
            passed = (np.array_equal(native, python) if name.startswith('pick-') else
                      np.isfinite(native).all() and np.allclose(native, python, rtol=2e-4, atol=args.atol))
            layers.append(dict(name=name, passed=bool(passed), **stats))
            if not passed:
                failures.append(size + ': ' + name + ' ' + str(stats))
        from moe_cost import account_moe
        reports.append(dict(valid=[width, height], field=[g['full_width'], g['full_height']],
                            image=None if not args.images else str(args.images[index]), features_sha256=sha256(features),
                            graph_eager_bit_exact=None if args.head_only and not args.routes_only else bool(np.array_equal(eager, graph)),
                            repeat_bit_exact=bool(np.array_equal(graph, repeated)),
                            head=difference(graph, expected_head), composite=composite, layers=layers,
                            load_rows=loads, cost=None if not loads else account_moe(model.shape, g, loads)))
        print(size, 'head', reports[-1]['head'], 'failed', sum(not l['passed'] for l in layers), flush=True)
        del model, expected, result
        torch.cuda.empty_cache()
    (args.output / 'report.json').write_text(json.dumps(dict(rtol=2e-4, atol=args.atol, results=reports, failures=failures), indent=2) + '\n', encoding='utf-8')
    for message in failures:
        print(message)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
