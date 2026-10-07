#!/usr/bin/env python3
"""Run the student on an image and render the result (openspec tasks 2.9-2.10).

与 run_image.py 同参同输出(proxy/neural/blend/side_by_side PNG、同 composite 语义),
只是网络换学生。权重默认随机(--seed 可复现),--checkpoint 加载训练产物(.pt state_dict)。

    python3 run_student.py input.png --width 512 --height 512 -o out/
    python3 run_student.py input.png --shape ../shapes/student_alt_depth.json --checkpoint v1.pt
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from nr_geometry import geometry_from_valid
from nr_history import reproject_history, store_history
from nr_student import StudentNetwork, load_student_shape, student_alignment
from run_image import build_features, save_png


def main():
    parser = argparse.ArgumentParser(description='run the student on an image and render the result')
    parser.add_argument('input', help='input image (any format PIL reads)')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--shape', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        '..', 'shapes', 'student_v0.json'))
    parser.add_argument('--checkpoint', help='trained weights (.pt state_dict); omit for random init')
    parser.add_argument('-o', '--out', default='out_student')
    parser.add_argument('--seed', type=int, default=12345, help='noise seed (and random-init seed)')
    parser.add_argument('--style', type=float, default=0.0)
    parser.add_argument('--tone', type=float, default=0.5)
    parser.add_argument('--structure', type=float, default=0.5)
    parser.add_argument('--skin', type=float, default=-1.0)
    parser.add_argument('--automask', action='store_true')
    parser.add_argument('--repeat', type=int, default=1, help='forward repeats for timing')
    parser.add_argument('--prev', help="previous frame's stored history (.npy, as written by "
                                       '--emit-history; D10 semantics)')
    parser.add_argument('--motion', help='motion field .npy [h,w,2], uv offsets y down; default zero')
    parser.add_argument('--emit-history', help="write the next frame's stored history (.npy)")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    shape = load_student_shape(args.shape)
    g = geometry_from_valid(args.width, args.height)

    image = Image.open(args.input).convert('RGB').resize((args.width, args.height), Image.LANCZOS)
    proxy = np.asarray(image, dtype=np.float32) / 255.0

    repro = None
    if args.prev:
        history = (np.load(args.prev).astype(np.float32) if args.prev.lower().endswith('.npy')
                   else np.asarray(Image.open(args.prev).convert('RGB'), dtype=np.float32) / 255.0)
        if history.shape != (args.height, args.width, 3):
            raise SystemExit(f'--prev shape {history.shape} != {(args.height, args.width, 3)}')
        motion = (np.load(args.motion).astype(np.float32) if args.motion
                  else np.zeros((args.height, args.width, 2), np.float32))
        repro = reproject_history(history, motion)
    elif args.motion:
        raise SystemExit('--motion needs --prev')

    features_np = build_features(proxy, g, args.seed, args.style, args.tone,
                                 args.structure, args.skin, args.automask, history=repro)
    features = torch.from_numpy(features_np).to(device)

    torch.manual_seed(args.seed)
    model = StudentNetwork(shape, g).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f'loaded {args.checkpoint}')
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"shape {shape.get('name', args.shape)}  params {params / 1e6:.2f}M  device {device}  "
          f"field {g['full_width']}x{g['full_height']}")

    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.repeat):
            head = model(features)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / args.repeat * 1000
    print(f'forward: {elapsed:.1f} ms  ({args.repeat} repeats)')

    head_np = head.float().cpu().numpy().reshape(g['full_height'], g['full_width'], 4)
    valid = head_np[:args.height, :args.width]
    rgb = valid[..., :3] / 4.0
    neural = np.clip(proxy + rgb, 0, 1)
    sigmoid = 1.0 / (1.0 + np.exp(-valid[..., 3].astype(np.float64)))
    blend = np.clip(sigmoid * float(model.blend_scale.detach().cpu()), 0, 1)
    if repro is not None:
        neural = neural + (repro.astype(np.float64) - neural) * blend[..., None]
    if args.emit_history:
        np.save(args.emit_history, store_history(neural.astype(np.float32)))
        print('  wrote', args.emit_history)

    print(f'head rgb/4: min {rgb.min():+.3f}  max {rgb.max():+.3f}  mean {rgb.mean():+.3f}')
    print(f'blend logit: min {valid[..., 3].min():+.3f}  max {valid[..., 3].max():+.3f}  '
          f'applied mean {blend.mean():.3f}')

    os.makedirs(args.out, exist_ok=True)
    save_png(os.path.join(args.out, 'proxy.png'), proxy)
    save_png(os.path.join(args.out, 'neural.png'), neural)
    save_png(os.path.join(args.out, 'blend.png'), np.repeat(blend[..., None], 3, axis=2))
    side = np.concatenate([proxy, neural, np.repeat(blend[..., None], 3, axis=2)], axis=1)
    save_png(os.path.join(args.out, 'side_by_side.png'), side)

    print('capture alignment (student -> teacher):')
    for s, t in student_alignment().items():
        print(f'  {s:12s} -> {t}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
