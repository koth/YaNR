#!/usr/bin/env python3
# Head-level arbitration against MLX-DLSS's independent implementation of the same network.
#
# One set of input features (built exactly as run_image.py builds them) goes through both networks:
#   theirs:  mlxdlss.pipeline.NeuralRenderingPipeline.run_features (float torch, recovered layout)
#   ours:    this runtime's Network (the fixed-point contract, native CUDA)
# and the f32 [rows, 4] heads are compared value for value. The layout question
# (NR_WEIGHT_LAYOUT=nr vs mlx) is settled by whichever side matches the reference head.
#
#   python3 check_vs_mlx_head.py prepare   samples/lake.png features.bin --width 512 --height 512
#   python3 check_vs_mlx_head.py theirs    features.bin head_ref.bin          # local (MPS/CPU)
#   python3 check_vs_mlx_head.py ours      features.bin head_ours.bin         # the CUDA box
#   python3 check_vs_mlx_head.py compare   head_ref.bin head_ours.bin

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tools', 'MLX-DLSS', 'python'))


def load_features(path):
    with open(path, 'rb') as handle:
        dims = np.fromfile(handle, dtype=np.int64, count=3)
    return np.fromfile(path, dtype=np.float32, offset=24).reshape(tuple(int(d) for d in dims))


def cmd_prepare(args):
    from PIL import Image
    from run_image import build_features
    from nr_geometry import geometry_from_valid
    image = Image.open(args.image).convert('RGB').resize((args.width, args.height), Image.LANCZOS)
    proxy = np.asarray(image, dtype=np.float32) / 255.0
    g = geometry_from_valid(args.width, args.height)
    features = build_features(proxy, g, args.seed, args.style, args.tone, args.structure,
                              args.skin, args.automask)
    features = features.reshape(g['full_height'], g['full_width'], 16)
    with open(args.out, 'wb') as handle:
        np.asarray(features.shape, dtype=np.int64).tofile(handle)
        features.astype(np.float32).tofile(handle)
    print(f'features {features.shape} -> {args.out}')


def cmd_theirs(args):
    from mlxdlss.pipeline import NeuralRenderingPipeline
    features = load_features(args.features)
    pipeline = NeuralRenderingPipeline.from_safetensors(args.weights, device=args.device,
                                                       precision='reference')
    head = pipeline.run_features(features)                     # (H, W, 4)
    with open(args.out, 'wb') as handle:
        np.asarray(head.shape, dtype=np.int64).tofile(handle)
        head.astype(np.float32).tofile(handle)
    print(f'mlxdlss head {head.shape} -> {args.out}  '
          f'rgb/4 range [{head[..., :3].min() / 4:+.4f}, {head[..., :3].max() / 4:+.4f}]')


def cmd_ours(args):
    import torch
    import nr_torch
    from nr_geometry import geometry_from_valid
    from nr_model import Model
    from nr_network import Network
    features = load_features(args.features)
    height, width = features.shape[:2]
    g = geometry_from_valid(width, height)
    if (g['full_height'], g['full_width']) != (height, width):
        raise ValueError(f'feature extent {height}x{width} does not match the geometry '
                         f"{g['full_height']}x{g['full_width']}")
    model = Model(nr_torch.DEVICE).load(args.weights)
    network = Network(model, g)
    head = network.record(torch.from_numpy(features.reshape(height * width, 16)).to(nr_torch.DEVICE))
    torch.cuda.synchronize()
    out = head.float().cpu().numpy().reshape(height, width, 4)
    with open(args.out, 'wb') as handle:
        np.asarray(out.shape, dtype=np.int64).tofile(handle)
        out.astype(np.float32).tofile(handle)
    print(f'our head {out.shape} -> {args.out}  '
          f'rgb/4 range [{out[..., :3].min() / 4:+.4f}, {out[..., :3].max() / 4:+.4f}]')


def cmd_compare(args):
    ref = load_features(args.ref)
    got = load_features(args.got)
    if ref.shape != got.shape:
        print(f'shape mismatch {ref.shape} vs {got.shape}')
        return 1
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    per_channel = d.reshape(-1, 4).mean(0)
    print(f'head MAE {d.mean():.6f}  max {d.max():.6f}   per-channel {np.round(per_channel, 6).tolist()}')
    print(f'  (reference rgb/4 std {ref[..., :3].std() / 4:.6f}; '
          f'a wrong weight layout gives MAE ~ 1.0)')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare'); p.add_argument('image'); p.add_argument('out')
    p.add_argument('--width', type=int, default=512); p.add_argument('--height', type=int, default=512)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--style', type=float, default=0.0); p.add_argument('--tone', type=float, default=0.5)
    p.add_argument('--structure', type=float, default=0.5); p.add_argument('--skin', type=float, default=-1.0)
    p.add_argument('--automask', action='store_true')
    p.set_defaults(func=cmd_prepare)
    p = sub.add_parser('theirs'); p.add_argument('features'); p.add_argument('out')
    p.add_argument('--weights', required=True); p.add_argument('--device', default='auto')
    p.set_defaults(func=cmd_theirs)
    p = sub.add_parser('ours'); p.add_argument('features'); p.add_argument('out')
    p.add_argument('--weights', required=True)
    p.set_defaults(func=cmd_ours)
    p = sub.add_parser('compare'); p.add_argument('ref'); p.add_argument('got')
    p.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    return args.func(args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
