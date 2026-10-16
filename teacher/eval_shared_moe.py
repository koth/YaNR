#!/usr/bin/env python3
"""Teacher-pinned texture/noise/branch/temporal diagnostics at native image size.

ROI is supplied explicitly (x0,y0,x1,y1); it is NOT an automatic skin detector.
Raw metrics are evidence, not an automatic quality pass. Prototype packages
require --smoke and are labelled pipeline checks, never trained quality results.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from audit_moe import sha256
from nr_geometry import geometry_from_valid
from nr_history import block_match_flow, reproject_history, store_history
from nr_package import load_student_package
from nr_shared_moe import RoutedModule
from run_image import build_features


def highpass(image):
    p = np.pad(image.astype(np.float64), ((1, 1), (1, 1), (0, 0)), mode='reflect')
    return 4 * p[1:-1, 1:-1] - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:]


def image_metrics(student, teacher, proxy, roi=None):
    sr, tr = student - proxy, teacher - proxy
    sh, th = highpass(sr), highpass(tr)
    if roi is not None:
        x0, y0, x1, y1 = roi
        student, teacher, sr, tr, sh, th = [v[y0:y1, x0:x1] for v in (student, teacher, sr, tr, sh, th)]
        if not student.size:
            raise ValueError('ROI is empty/outside the image')
    centered_s, centered_t = sh - sh.mean(), th - th.mean()
    denom = np.sqrt(np.sum(centered_s ** 2) * np.sum(centered_t ** 2))
    teacher_energy = float(np.sqrt(np.mean(th ** 2)))
    mse = float(np.mean((student.astype(np.float64) - teacher.astype(np.float64)) ** 2))
    return dict(image_mae=float(np.mean(np.abs(student - teacher))),
                image_psnr_db=None if mse == 0 else float(-10 * np.log10(mse)),
                residual_mae=float(np.mean(np.abs(sr - tr))),
                highpass_residual_mae=float(np.mean(np.abs(sh - th))),
                student_hf_rms=float(np.sqrt(np.mean(sh ** 2))), teacher_hf_rms=teacher_energy,
                hf_gain=None if teacher_energy < 1e-12 else float(np.sqrt(np.mean(sh ** 2)) / teacher_energy),
                hf_correlation=None if denom < 1e-12 else float(np.sum(centered_s * centered_t) / denom))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--teacher-weights', type=Path, required=True)
    parser.add_argument('--images', type=Path, nargs='+', required=True, help='Ordered frames if --temporal')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--roi', type=str, help='Explicit skin ROI x0,y0,x1,y1; shared across frames')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--style', type=float, default=64)
    parser.add_argument('--tone', type=float, default=1)
    parser.add_argument('--structure', type=float, default=1)
    parser.add_argument('--skin', type=float, default=1)
    parser.add_argument('--size', type=int, help='Explicit square diagnostic override; default native size')
    parser.add_argument('--temporal', action='store_true', help='Independent teacher/student history chains with block-matching motion')
    parser.add_argument('--smoke', action='store_true', help='Permit untrained package; no quality conclusion')
    args = parser.parse_args()
    # Optional prebuilt native teacher library, useful on Windows.
    if os.environ.get('NR_CUDA_LIBRARY'):
        import nr_cuda
        nr_cuda._SO = os.environ['NR_CUDA_LIBRARY']
    from nr_model import Model
    from nr_network import Network
    from eval_temporal import composite
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    roi = tuple(map(int, args.roi.split(','))) if args.roi else None
    manifest = json.loads((args.weights / 'manifest.json').read_text())
    prototype = manifest['model'].get('untrainedPrototype', False)
    validation_only = manifest['model'].get('validationOnly', False)
    if (prototype or validation_only) and not args.smoke:
        raise ValueError('random prototype: use --smoke for pipeline validation only')
    teacher_manifest_hash = sha256(args.teacher_weights / 'manifest.json')
    source_hash = manifest['model'].get('source', {}).get('teacher_manifest_sha256')
    if source_hash and source_hash != teacher_manifest_hash:
        raise ValueError('teacher weights differ from the checkpoint provenance')
    teacher = Model('cuda' if torch.cuda.is_available() else 'cpu').load(str(args.teacher_weights))
    student = None
    report = dict(kind='pipeline-smoke-only' if prototype or validation_only or args.smoke else 'trained-model-diagnostics',
                  student_manifest_sha256=sha256(args.weights / 'manifest.json'),
                  teacher_manifest_sha256=teacher_manifest_hash, seed=args.seed,
                  conditions=dict(style=args.style, tone=args.tone, structure=args.structure, skin=args.skin, automask=True),
                  roi=roi, native_size=args.size is None, temporal=args.temporal,
                  branch_ablation='Zero indicated branch, retain c=1/2, no renormalization; downstream routes may change with changed activations.',
                  note='ROI is user-supplied; no skin detection or quality gate is inferred from PSNR/load alone.', images=[])
    histories = {'teacher': None, 'full': None, 'remove_shared': None, 'remove_routed': None}
    previous = {}; prev_proxy = None
    for index, path in enumerate(args.images):
        image = Image.open(path).convert('RGB')
        if args.size:
            image = image.resize((args.size, args.size), Image.Resampling.LANCZOS)
        proxy = np.asarray(image, np.float32) / 255
        height, width = proxy.shape[:2]; g = geometry_from_valid(width, height)
        if roi and not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height):
            raise ValueError('ROI must fit every evaluated image')
        if student is None:
            student, _ = load_student_package(args.weights, width, height, teacher.device)
        student.g = g
        network = Network(teacher, g)
        if args.temporal and prev_proxy is not None and proxy.shape != prev_proxy.shape:
            raise ValueError('temporal frames must have matching sizes')
        motion = block_match_flow(prev_proxy, proxy) if args.temporal and prev_proxy is not None else None
        repros = {name: (reproject_history(history, motion / np.array([width, height], np.float32))
                         if history is not None and motion is not None else None) for name, history in histories.items()}
        seeds = [int(np.random.default_rng([args.seed, index, j]).integers(1 << 30)) for j in range(2)]
        outputs = {}
        with torch.inference_mode():
            for j, seed in enumerate(seeds):
                for mode in ('teacher', 'full', 'remove_shared', 'remove_routed'):
                    # Static/noise probe shares identical lanes; temporal uses each chain's own history.
                    feats = build_features(proxy, g, seed, args.style, args.tone, args.structure, args.skin, True,
                                           history=repros[mode])
                    x = torch.from_numpy(feats).to(teacher.device)
                    if mode == 'teacher':
                        head = network.record(x)
                        scale = teacher.blend_scale()
                        network.boundaries = None
                    else:
                        for module in student.modules():
                            if isinstance(module, RoutedModule):
                                module.branch_mode = mode if mode != 'full' else 'full'
                        head = student(x)
                        scale = float(student.blend_scale)
                    head = head.float().cpu().numpy().reshape(g['full_height'], g['full_width'], 4)
                    outputs[mode, j] = composite(head, proxy, height, width, repros[mode], scale)
        entry = dict(path=str(path), image_sha256=sha256(path), valid=[width, height],
                     field=[g['full_width'], g['full_height']], seeds=seeds, modes={})
        for mode in ('full', 'remove_shared', 'remove_routed'):
            out, target = outputs[mode, 0], outputs['teacher', 0]
            metrics = image_metrics(out, target, proxy)
            metrics['roi'] = None if roi is None else image_metrics(out, target, proxy, roi)
            metrics['noise_response_rms'] = float(np.sqrt(np.mean((outputs[mode, 1] - out) ** 2)))
            metrics['teacher_noise_response_rms'] = float(np.sqrt(np.mean((outputs['teacher', 1] - target) ** 2)))
            metrics['noise_response_error_rms'] = float(np.sqrt(np.mean(((outputs[mode, 1] - out) - (outputs['teacher', 1] - target)) ** 2)))
            if mode in previous:
                metrics['temporal_teacher_delta_error_rms'] = float(np.sqrt(np.mean(((out - previous[mode]) -
                                                                                                (target - previous['teacher'])) ** 2)))
            entry['modes'][mode] = metrics
        for mode in histories:
            histories[mode] = store_history(outputs[mode, 0]) if args.temporal else None
            previous[mode] = outputs[mode, 0]
        prev_proxy = proxy
        report['images'].append(entry)
        print(path, 'diagnostics complete', flush=True)
        del network
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
