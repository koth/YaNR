#!/usr/bin/env python3
"""Production-shape forward/backward smoke. No teacher weights or quality claim."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from nr_package import build_student
from nr_geometry import geometry_from_valid
from run_image import build_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', default=str(Path(__file__).resolve().parent.parent / 'shapes/cnn_single_56m.json'))
    parser.add_argument('--sizes', default='320x320,512x512,640x384,384x640,512x288')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--backward', action='store_true', help='backward on the first geometry')
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.device == 'cuda' and os.name != 'nt' and os.environ.get('CUDA_VISIBLE_DEVICES') != '6':
        parser.error('remote GPU constraint: set CUDA_VISIBLE_DEVICES=6')
    torch.set_num_threads(args.threads)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261019)
    sizes = [tuple(map(int, size.split('x'))) for size in args.sizes.split(',')]
    model = build_student(args.shape, geometry_from_valid(*sizes[0]), args.device).eval()
    report = dict(device=args.device, threads=args.threads, ledger=model.parameter_ledger(), forwards=[])
    for w, h in sizes:
        g = geometry_from_valid(w, h)
        model.g = g
        proxy = np.random.default_rng([20261019, w, h]).uniform(.1, .9, (h, w, 3)).astype(np.float32)
        x = torch.from_numpy(build_features(proxy, g, 91, 2., 1., 1., -1., True)).to(args.device)
        started = time.perf_counter()
        with torch.inference_mode():
            head = model(x)
            assert head.shape == (g['full_rows'], 4) and torch.isfinite(head).all()
        record = dict(valid=[w, h], field=[g['full_width'], g['full_height']],
                      rows=g['full_rows'], finite=True, seconds=time.perf_counter() - started)
        report['forwards'].append(record)
        print(json.dumps(record), flush=True)
    if args.backward:
        w, h = sizes[0]
        g = model.g = geometry_from_valid(w, h)
        x = torch.zeros((g['full_rows'], 16), device=args.device)
        x[:, 4:10] = .5
        model.train()
        head = model(x)
        aux, z = model.moe_losses()
        loss = head[:, :3].square().mean() + .01 * aux + .001 * z
        loss.backward()
        finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert finite and float(model.stem[0].weight.grad.abs().sum()) > 0
        report['backward'] = dict(valid=[w, h], loss=float(loss.detach()), gradients_finite=True)
        print(json.dumps(report['backward']), flush=True)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
