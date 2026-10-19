#!/usr/bin/env python3
"""End-to-end FP32 parity for routed7 and CNN students (CPU, CUDA, graph replay).

Without --package, export deterministically initialized production shapes.
--package can be repeated to check trained packages instead. No CUDA torch is
needed; PyTorch runs the reference on CPU. Outputs stay in --work for diagnosis.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from nr_geometry import geometry_from_valid
from nr_package import build_student, load_student_package, write_package


def invoke(exe, args):
    result = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{exe}:\n{result.stdout}\n{result.stderr}')
    return result.stdout


def compare(label, expected, actual, tolerance):
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise AssertionError(f'{label}: shape mismatch / nonfinite output')
    delta = np.abs(expected - actual)
    maximum = float(delta.max(initial=0))
    if maximum > tolerance:
        raise AssertionError(f'{label}: max {maximum:.3e}, mean {delta.mean():.3e} > {tolerance:.3e}')
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=Path, default=Path('engine/build/Release/cpu_engine.exe'))
    parser.add_argument('--gpu', type=Path, default=Path('engine/build-cuda/gpu_engine.exe'))
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--package', type=Path, action='append')
    parser.add_argument('--shape', type=Path, action='append')
    parser.add_argument('--edge-cases', action='store_true', help='small dilated/ViT/tie and CNN channel-tail/zero-block fixtures')
    parser.add_argument('--work', type=Path, default=Path('tmp/native_students'))
    parser.add_argument('--sizes', nargs='+', default=['320x320', '512x512', '320x512', '137x201'])
    parser.add_argument('--tolerance', type=float, default=2e-4)
    args = parser.parse_args()
    torch.set_num_threads(8)
    args.work.mkdir(parents=True, exist_ok=True)
    packages = list(args.package or [])
    if not packages:
        shapes = list(args.shape or map(Path, ['shapes/student_routed7_v1.json', 'shapes/cnn_unet_moe8.json']))
        if args.edge_cases:
            routed = json.loads(Path('shapes/student_routed7_v1.json').read_text(encoding='utf-8'))
            for level in routed['levels']:
                level.update(channels=32, hidden=32, blocks=2)
                if level['level'] in ('d2', 'd3'):
                    level['window_dilation'] = 2
            routed['vit'].update(channels=32, ffn=64, blocks=1)
            cnn = json.loads(Path('shapes/cnn_unet_moe8.json').read_text(encoding='utf-8'))
            for name in cnn['channels']:
                cnn['channels'][name] = 24
                cnn['expert_hidden'][name] = 13
                cnn['blocks_enc'][name] = 0 if name == 'full' else 1
            for name in cnn['blocks_dec']:
                cnn['blocks_dec'][name] = 1 if name == 'full' else 0
            shapes = []
            for name, shape in [('routed7_dilated_vit_ties', routed), ('cnn_tails_zero_blocks', cnn)]:
                path = args.work / (name + '.json')
                path.write_text(json.dumps(shape), encoding='utf-8'); shapes.append(path)
        for shape in shapes:
            torch.manual_seed(179)
            model = build_student(shape, geometry_from_valid(320, 320)).eval()
            # Nondefault affine, residual scales and biases catch omitted/misplaced
            # parameters. Normal router weights retain pixel/window variation.
            with torch.no_grad():
                for name, value in model.named_parameters():
                    if 'aux' in name:
                        value.uniform_(.7, 1.2)
                    elif 'norm' in name and name.endswith('.weight'):
                        value.uniform_(.8, 1.2)
                    elif 'norm' in name and name.endswith('.bias'):
                        value.uniform_(-.05, .05)
                    elif name.endswith('.attention.prior'):
                        value.uniform_(-.1, .1)
                    elif name.endswith('.attention.scale'):
                        value.uniform_(.7, 1.2)
                for name, value in model.named_buffers():
                    if name.endswith('router_bias'):
                        value.copy_(torch.linspace(-.07, .09, value.numel()))
                if getattr(model, 'cfg', {}).get('architecture') == 'cnn_unet_moe8':
                    # Avoid discontinuous argmax boundaries in the end-to-end
                    # arithmetic test. Rotate the dominant expert across blocks
                    # so every bank and empty-expert grouping are exercised.
                    # Mixed routing and near ties are checked by shared_moe_check.
                    index = 0
                    for name, module in model.named_modules():
                        if name.endswith('.ffn.router'):
                            module.bias.zero_(); module.bias[index % 8] = 8
                            if index == 0 and shape.stem == 'cnn_tails_zero_blocks':
                                # Different logits that tie after FP32 softmax.
                                module.weight.zero_(); module.bias.fill_(-1e-8); module.bias[-1] = 0
                            index += 1
                if shape.stem == 'routed7_dilated_vit_ties':
                    for name, value in model.named_parameters():
                        if name.endswith('.router.1.weight'):
                            value.zero_()
                    for name, value in model.named_buffers():
                        if name.endswith('router_bias'):
                            value.zero_()
            package = args.work / shape.stem
            write_package(model, model.state_dict(), package, shape, extra={'untrainedPrototype': True})
            del model
            packages.append(package)
    results = []
    for package in packages:
        for size in args.sizes:
            width, height = map(int, size.split('x'))
            case = args.work / f'{package.name}-{size}'
            case.mkdir(parents=True, exist_ok=True)
            model, _ = load_student_package(package, width, height, 'cpu')
            model.capture_enabled = True
            for module in model.modules():
                if hasattr(module, 'capture_enabled'):
                    module.capture_enabled = True
            torch.manual_seed(811)
            features = torch.randn(model.g['full_rows'], 16) * .3
            features.numpy().tofile(case / 'features.bin')
            with torch.inference_mode():
                expected = model(features).numpy().reshape(-1)
            refs = {tag: tensor.detach().numpy().reshape(-1).copy() for tag, tensor in model.captures.items()}
            if getattr(model, 'shared_routed', False):
                for name, module in model.named_modules():
                    for tag, value in getattr(module, 'captures', {}).items():
                        if tag in ('logits', 'pick', 'gate', 'ffn', 'qkv', 'attention'):
                            refs[('att' if tag == 'attention' else tag) + '-' + name] = value.numpy().reshape(-1).copy()
            # CNN exposes plain pixel top-1 selections in last_stats. Record every
            # block so a single misrouted pixel cannot hide in the final head.
            if getattr(model, 'cfg', {}).get('architecture') == 'cnn_unet_moe8':
                for name, module in model.named_modules():
                    stats = getattr(module, 'last_stats', None)
                    if stats is not None:
                        pick, probs, logits = stats
                        refs['pick-' + name] = pick.numpy().reshape(-1).astype(np.float32)
                        refs['logits-' + name] = logits.permute(0, 2, 3, 1).reshape(-1).numpy().copy()
                        refs['gate-' + name] = probs.permute(0, 2, 3, 1).reshape(-1, 8).gather(1, pick[:, None]).numpy().reshape(-1)
            common = ['--weights', package, '--features', case / 'features.bin', '--width', width, '--height', height]
            summary = {'package': str(package), 'size': size}
            for backend, exe in [('cpu', args.cpu)] + ([] if args.cpu_only else [('cuda', args.gpu)]):
                invoke(exe, common + ['--out', case / f'{backend}.bin', '--dump-dir', case / backend, '--threads', 8])
                maximum = 0
                for tag, reference in refs.items():
                    if tag == 's-head':
                        continue  # Head is also checked via the public --out interface.
                    capture = case / backend / f'{tag}.bin'
                    if not capture.exists():
                        raise AssertionError(f'{backend}: missing {tag}')
                    maximum = max(maximum, compare(f'{backend}/{tag}', reference, np.fromfile(capture, np.float32),
                                                   0 if tag.startswith('pick-') else args.tolerance))
                eager = np.fromfile(case / f'{backend}.bin', np.float32)
                summary[backend] = {'head_max': compare(backend + '/head', expected, eager, args.tolerance), 'captures_max': maximum}
                if backend == 'cuda':
                    for repeat in range(2):
                        invoke(exe, common + ['--out', case / 'graph.bin', '--bench', '--warmup', 1, '--repeats', 2])
                        compare('eager/graph repeat', eager, np.fromfile(case / 'graph.bin', np.float32), 0)
            results.append(summary)
            (args.work / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            print(json.dumps(summary), flush=True)
            del model, refs
            gc.collect()
    print('NATIVE STUDENTS PASS', flush=True)


if __name__ == '__main__':
    main()
