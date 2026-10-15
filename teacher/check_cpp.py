#!/usr/bin/env python3
"""Compare shipped student packages with native CPU/CUDA using the same pixels.

    .venv/Scripts/python.exe teacher/check_cpp.py --cuda --bisect \
        --weights weights/student_v4 --report tmp/cpp_parity.json

FP32 head tolerance is 2e-3, as in check_engine.py. PNGs may differ by at most
one 8-bit level. Noise uses platform libm, so lane parity allows one adjacent
half value in lanes 0..2; all other lanes must match exactly. A pre-resized PNG
avoids conflating WIC Fant / PIL Lanczos differences with model differences.
Stage captures are diagnostic: discontinuous MoE picks can amplify rounding in
intermediate states even when the final head passes. History checks input lanes
and forward only; temporal output blending belongs to the calling application.
student_v4 is restricted to default controls, seed and static input; only the
320x320 and 512x512 fixed64 cases are allowed for that package.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image
import torch

from nr_geometry import geometry_from_valid
from nr_package import load_student_package
from run_image import build_features


def run(exe, args):
    env = dict(os.environ, OMP_NUM_THREADS='8')
    for key in ('NR_INT8', 'NR_PROFILE'):
        env.pop(key, None)  # Presence, including NR_INT8=0, enables INT8 in the CLI.
    result = subprocess.run([str(exe), *map(str, args)], env=env,
                            capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{exe}:\n{result.stdout}\n{result.stderr}')
    return result.stdout


def compare(ref, got, tolerance):
    if ref.shape != got.shape or not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise AssertionError('shape mismatch or nonfinite values')
    delta = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    stats = {'max': float(delta.max()), 'mean': float(delta.mean()),
             'different': int(np.count_nonzero(delta))}
    if tolerance is not None and stats['max'] > tolerance:
        raise AssertionError(f'{stats}, tolerance {tolerance}')
    return stats


def render(head, proxy, g):
    valid = head.reshape(g['full_height'], g['full_width'], 4)[:proxy.shape[0], :proxy.shape[1]]
    # Same f32 composite and round-to-nearest-byte as run_student/run_image.
    rgb = np.clip(proxy + valid[..., :3] / 4.0, 0, 1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def head_details(reference, actual, proxy, g, blend_scale):
    """Separate RGB, temporal confidence and padding from the raw head maximum."""
    height, width = proxy.shape[:2]
    ref = reference.reshape(g['full_height'], g['full_width'], 4)[:height, :width]
    got = actual.reshape(g['full_height'], g['full_width'], 4)[:height, :width]
    ref_rgb = np.clip(proxy + ref[..., :3] / 4.0, 0, 1)
    got_rgb = np.clip(proxy + got[..., :3] / 4.0, 0, 1)

    def confidence(logit):
        value = logit.astype(np.float64)
        return np.clip(blend_scale / (1.0 + np.exp(-np.clip(value, -700, 700))), 0, 1)

    delta = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    # Diagnostic only: the original absolute head pass/fail threshold is retained.
    return {
        'rgb_raw': compare(reference[:, :3], actual[:, :3], None),
        'blend_logit_raw': compare(reference[:, 3], actual[:, 3], None),
        'valid_rgb_raw': compare(ref[..., :3], got[..., :3], None),
        'valid_blend_logit_raw': compare(ref[..., 3], got[..., 3], None),
        'composited_rgb': compare(ref_rgb, got_rgb, None),
        'confidence': compare(confidence(ref[..., 3]), confidence(got[..., 3]), None),
        'reference_rgb_abs_max': float(np.abs(reference[:, :3]).max()),
        'reference_blend_logit_abs_max': float(np.abs(reference[:, 3]).max()),
        'outside_atol_2e3_rtol_1e5': int(np.count_nonzero(delta > 2e-3 + 1e-5 * np.abs(reference))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', action='append',
                        help='repeat for multiple packages; defaults to student_v3 and student_v4')
    parser.add_argument('--image', default='teacher/samples/lake.png')
    parser.add_argument('--cpu-build', default='engine/build/Release')
    parser.add_argument('--cuda-build', default='engine/build-cuda')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--bisect', action='store_true', help='compare every captured stage and block')
    parser.add_argument('--quick', action='store_true', help='only the default 320x320 working point')
    parser.add_argument('--case', action='append',
                        help='select named cases; student_v4 only permits fixed64-320/512')
    parser.add_argument('--tolerance', type=float, default=2e-3)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    suffix = '.exe' if os.name == 'nt' else ''
    cpu = Path(args.cpu_build) / ('cpu_engine' + suffix)
    gpu = Path(args.cuda_build) / ('gpu_engine' + suffix)
    lane_exe = Path(args.cpu_build) / ('cpu_lanes' + suffix)
    packages = args.weights or ['weights/student_v3', 'weights/student_v4']
    # Default controls are intentionally omitted to check the actual C++ defaults.
    cases = [('fixed64-320', 320, 320, 64, 1, 1, 1, True, False, 12345, [])]
    if not args.quick:
        cases += [
            ('fixed64-512', 512, 512, 64, 1, 1, 1, True, False, 12345, []),
            ('style0-default-controls', 320, 320, 0, 1, 1, 1, True, False, 12345, None),
            ('style127-default-controls', 320, 320, 127, 1, 1, 1, True, False, 12345, None),
            ('style0-no-mask', 320, 320, 0, .25, .75, -1, False, False, 0, None),
            ('style127-rect', 320, 512, 127, 0, 0, .25, True, False, 42, None),
            ('odd-skin-follow', 137, 201, 64, 1, .35, -1, True, False, 12345, None),
            ('history', 320, 320, 64, 1, 1, 1, True, True, 12345, []),
            ('half-rounding-controls', 320, 320, 64, .5002441406250001, .5002441406249999,
             .5002441406250001, True, False, 12345, None),
        ]
    if args.case:
        unknown = set(args.case) - {case[0] for case in cases}
        if unknown:
            parser.error(f'unknown cases: {", ".join(sorted(unknown))}')
        cases = [case for case in cases if case[0] in args.case]
    selections = []
    for package in packages:
        package_cases = cases
        if Path(package).resolve().name.casefold() == 'student_v4':
            package_cases = [case for case in cases if case[0] in ('fixed64-320', 'fixed64-512')]
            if len(package_cases) != len(cases):
                print(f'{package}: default parameters only; skipping nondefault cases', flush=True)
        if package_cases:
            selections.append((package, package_cases))
    if not selections:
        parser.error('student_v4 only permits default cases: fixed64-320 and fixed64-512')
    image = Image.open(args.image).convert('RGB')
    records = []
    for package, package_cases in selections:
        shape = json.loads((Path(package) / 'shape.json').read_text(encoding='utf-8'))
        has_moe = any(level['kind'] == 'moe' for level in shape['levels'])
        for name, width, height, style, tone, structure, skin, automask, history, seed, options in package_cases:
            if style == 127 and not has_moe:
                name, style = name.replace('style127-', 'style3-'), 3
            print(f'{package} / {name}', flush=True)
            g = geometry_from_valid(width, height)
            with tempfile.TemporaryDirectory(prefix='yanr_cpp_') as temp:
                tmp = Path(temp)
                resized = image.resize((width, height), Image.Resampling.LANCZOS)
                resized.save(tmp / 'input.png')
                proxy = np.asarray(resized, dtype=np.float32) / 255.0
                proxy.tofile(tmp / 'proxy.bin')
                previous = np.roll(proxy, 7, axis=1) if history else None
                expected_lanes = build_features(proxy, g, seed, style, tone, structure, skin, automask,
                                                history=previous)
                expected_lanes.tofile(tmp / 'python_features.bin')
                if options is None:
                    options = ['--style', style, '--tone', tone, '--structure', structure,
                               '--skin', skin, '--automask' if automask else '--no-automask']
                common = ['--weights', package, '--width', width, '--height', height, '--threads', 8]
                dump = ['--dump-dir', tmp / 'cpu'] if args.bisect else []
                if history:
                    previous.tofile(tmp / 'history.bin')
                    run(lane_exe, ['--proxy', tmp / 'proxy.bin', '--history', tmp / 'history.bin',
                                   '--out', tmp / 'features.bin', '--vw', width, '--vh', height,
                                   '--seed', seed, *options])
                    source = ['--features', tmp / 'features.bin', '--proxy', tmp / 'proxy.bin']
                else:
                    source = [tmp / 'input.png', '--seed', seed, *options,
                              '--dump-features', tmp / 'features.bin']
                run(cpu, [*source, *common, '--out', tmp / 'cpu.bin', '-o', tmp / 'cpu.png', *dump])
                lanes = np.fromfile(tmp / 'features.bin', np.float32).reshape(-1, 16)
                compare(expected_lanes[:, 3:], lanes[:, 3:], 0)
                # uint16 distances are valid here because a change of sign is a failure too.
                a = expected_lanes[:, :3].astype(np.float16).view(np.uint16).astype(np.int32)
                b = lanes[:, :3].astype(np.float16).view(np.uint16).astype(np.int32)
                noise_steps = int(np.abs(a - b).max())
                if noise_steps > 1:
                    raise AssertionError(f'noise differs by {noise_steps} half steps')
                model, manifest = load_student_package(package, width, height, 'cpu')
                model.capture_enabled = args.bisect
                hooks = []
                if args.bisect:
                    for block_name, block in model.named_modules():
                        if block_name.startswith(('enc_blocks.', 'dec_blocks.', 'vit_blocks.')) and block_name.count('.') in (1, 2):
                            if block_name.split('.')[-1].isdigit():
                                hooks.append(block.register_forward_hook(
                                    lambda _m, _i, out, tag='blk-' + block_name: model.capture(tag, out)))
                with torch.no_grad():
                    expected_head = model(torch.from_numpy(lanes)).numpy().copy()
                captures = {key: value.detach().numpy() for key, value in model.captures.items()}
                for hook in hooks:
                    hook.remove()
                row = {'package': package, 'version': manifest['model']['version'], 'case': name,
                       'size': [width, height], 'noise_max_half_steps': noise_steps,
                       'controls': {'style': style, 'tone': tone, 'structure': structure,
                                    'skin': skin, 'automask': automask, 'seed': seed, 'history': history},
                       'manifest_sha256': hashlib.sha256((Path(package) / 'manifest.json').read_bytes()).hexdigest(),
                       'backends': {}}
                for backend in (['cpu', 'cuda'] if args.cuda else ['cpu']):
                    if backend == 'cuda':
                        debug_args = ['--dump-dir', tmp / 'cuda'] if args.bisect else []
                        run(gpu, [*common, '--features', tmp / 'features.bin', '--proxy', tmp / 'proxy.bin',
                                  '--out', tmp / 'cuda.bin', '-o', tmp / 'cuda.png', *debug_args])
                    head = np.fromfile(tmp / (backend + '.bin'), np.float32).reshape(-1, 4)
                    stats = {'head': compare(expected_head, head, None)}
                    stats['details'] = head_details(expected_head, head, proxy, g,
                                                    float(model.blend_scale.detach()))
                    if args.bisect:
                        stage_stats = {}
                        for tag, reference in captures.items():
                            if tag == 's-head':
                                continue
                            path = tmp / backend / (tag + '.bin')
                            if not path.is_file():
                                raise AssertionError(f'missing {backend} capture: {tag}')
                            stage_stats[tag] = compare(reference.reshape(-1), np.fromfile(path, np.float32), None)
                        stats['stages'] = stage_stats
                        stats['stages_over_tolerance'] = [tag for tag, error in stage_stats.items()
                                                         if error['max'] > args.tolerance]
                        if stats['stages_over_tolerance']:
                            print(f'  {backend}: intermediate stage differences: '
                                  + ', '.join(stats['stages_over_tolerance']), flush=True)
                    expected_png = render(expected_head, proxy, g)
                    got_png = np.asarray(Image.open(tmp / (backend + '.png')).convert('RGB'))
                    stats['png_vs_own_head'] = compare(render(head, proxy, g), got_png, 1)
                    if backend == 'cuda':
                        # Graph is the production path; debug captures use eager execution.
                        run(gpu, [*common, '--features', tmp / 'features.bin', '--out', tmp / 'graph.bin',
                                  '--bench', '--warmup', 1, '--repeats', 2])
                        graph = np.fromfile(tmp / 'graph.bin', np.float32).reshape(-1, 4)
                        stats['eager_vs_graph'] = compare(head, graph, 0)
                    stats['png_vs_python'] = compare(expected_png, got_png, None)
                    stats['pass'] = stats['head']['max'] <= args.tolerance and stats['png_vs_python']['max'] <= 1
                    row['backends'][backend] = stats
                    print(f'  {backend}: head max {stats["head"]["max"]:.3e}, '
                          f'PNG max {stats["png_vs_python"]["max"]:.0f}/255, '
                          f'{"PASS" if stats["pass"] else "FAIL"}', flush=True)
                    detail = stats['details']
                    print(f'    RGB head {detail["valid_rgb_raw"]["max"]:.3e}, '
                          f'blend logit {detail["valid_blend_logit_raw"]["max"]:.3e}, '
                          f'composite RGB {detail["composited_rgb"]["max"]:.3e}, '
                          f'confidence {detail["confidence"]["max"]:.3e}', flush=True)
                # Also run Python's independent lane construction through the model.
                # This exposes noise rounding amplified by discontinuous MoE routing.
                model.capture_enabled = False
                model.captures.clear()
                with torch.no_grad():
                    independent = model(torch.from_numpy(expected_lanes)).numpy()
                row['independent_python_lanes'] = compare(independent, expected_head, None)
                row['lanes_bit_equal'] = bool(np.array_equal(expected_lanes, lanes))
                for backend, stats in row['backends'].items():
                    head = np.fromfile(tmp / (backend + '.bin'), np.float32).reshape(-1, 4)
                    got_png = np.asarray(Image.open(tmp / (backend + '.png')).convert('RGB'))
                    stats['independent_python_head'] = compare(independent, head, None)
                    stats['independent_python_png'] = compare(render(independent, proxy, g), got_png, None)
                row['independent_lanes_pass'] = all(
                    stats['independent_python_head']['max'] <= args.tolerance
                    and stats['independent_python_png']['max'] <= 1
                    for stats in row['backends'].values())
                row['common_input_pass'] = all(stats['pass'] for stats in row['backends'].values())
                print(f'  independent Python lanes: head delta '
                      f'{row["independent_python_lanes"]["max"]:.3e}, '
                      f'{"PASS" if row["independent_lanes_pass"] else "FAIL"}', flush=True)
                del model
                records.append(row)
                if args.report:
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    args.report.write_text(json.dumps({'status': 'running', 'tolerance': args.tolerance,
                                                      'cases': records}, indent=2), encoding='utf-8')
    common_pass = all(r['common_input_pass'] for r in records)
    if args.report:
        args.report.write_text(json.dumps({'common_input_status': 'pass' if common_pass else 'fail',
                                          'independent_lanes_status': 'pass' if all(r['independent_lanes_pass'] for r in records) else 'fail',
                                          'tolerance': args.tolerance,
                                          'cases': records}, indent=2), encoding='utf-8')
    independent_pass = all(r['independent_lanes_pass'] for r in records)
    print(f'CHECK CPP: common-input {"PASS" if common_pass else "FAIL"} ({len(records)} cases); independent lanes '
          f'{"PASS" if independent_pass else "FAIL"}', flush=True)
    return 0 if common_pass and independent_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
