#!/usr/bin/env python3
# Feed an image through the teacher and render what the network does to it.
#
# The 16 input lanes are built exactly as the frame pipeline builds them (docs/network.md,
# demo/shaders/nr_preprocess.comp): the three Gaussian noise lanes are Box-Muller off a hash of the padded
# pixel coordinate, lane 3 is 1, lanes 4-6 are the display proxy centred through three half roundings,
# lanes 7-9 are the previous frame's *output* (stored composite code value, truncated to the half grid,
# reprojected with a 5-tap Catmull-Rom, centred the same way; design.md D10) and copy lanes 4-6 when there
# is no history, lanes 10-15 are the style / tone / structure conditioning. The padded field mirrors the
# image (2 * valid - x - 2) while the noise keeps the padded coordinate. The head is composited the way the
# frame pipeline composites it:
#
#   neural = clamp(proxy + rgb / 4, 0, 1)        # head channels 0-2, in proxy code space
#   weight = clamp(sigmoid(head channel 3) * blend_scale, 0, 1)   # only with a history
#   neural = neural + (history - neural) * weight                  # the temporal blend
#
#   python3 run_image.py input.png --width 512 --height 512 -o out/
#   python3 run_image.py f1.png -o out1/ --emit-history out1/history.npy
#   python3 run_image.py f2.png -o out2/ --prev out1/history.npy --motion motion.npy

import argparse
import os

import numpy as np
import torch
from PIL import Image

import nr_torch
from nr_geometry import geometry_from_valid
from nr_history import reproject_history, store_history
from nr_model import Model
from nr_network import Network


def u32(x):
    return np.asarray(x, dtype=np.uint64).astype(np.uint32)   # wrap like the shader's uint math


def hash_uniform(value):
    mixed = u32(value)
    mixed = u32(np.right_shift(mixed, u32((mixed >> np.uint32(28)) + np.uint32(4))) ^ mixed)
    mixed = u32(mixed * np.uint32(0x108ef2d9))
    integer = u32(((mixed >> np.uint32(30)) ^ (mixed >> np.uint32(8))) + np.uint32(1))
    return integer.astype(np.float64) * (2.0 ** -24)          # uintBitsToFloat(0x33800000)


def gaussian3(x, y, seed):
    """The shader's three Gaussian lanes (Box-Muller off the padded-coordinate hash)."""
    base = u32(u32(u32(x * np.uint32(0x8da6b343)) ^ u32(y * np.uint32(0xd8163841)))
               ^ u32(np.uint64(np.uint64(seed) * np.uint64(0x9e3779b9)).astype(np.uint32))
               ^ np.uint32(0x243f6a88))
    base = u32(np.right_shift(base, u32((base >> np.uint32(28)) + np.uint32(4))) ^ base)
    base = u32(base * np.uint32(0x108ef2d9))
    base = u32(np.right_shift(base, np.uint32(22)) ^ base)
    u0 = hash_uniform(u32(base * np.uint32(0x2c9277b5) + np.uint32(0xac564b05)))
    u1 = hash_uniform(u32(base * np.uint32(0xfa6dc5f9) + np.uint32(0x4712a88e)))
    u2 = hash_uniform(u32(base * np.uint32(0xcaa5b80d) + np.uint32(0x21dd796b)))
    u3 = hash_uniform(u32(base * np.uint32(0x83232c31) + np.uint32(0x3463e0ac)))
    radius0 = np.sqrt(np.log2(u0) * np.log(2.0) * -2.0)
    radius1 = np.sqrt(np.log2(u2) * np.log(2.0) * -2.0)
    tau = 2.0 * np.pi

    def r16(v):
        return v.astype(np.float16).astype(np.float64)

    return (r16(radius0 * np.cos(u1 * tau)), r16(radius0 * np.sin(u1 * tau)),
            r16(radius1 * np.cos(u3 * tau)))


def center_proxy(code):
    """The shader's centring: roundF16(roundF16(roundF16(c) - 0.5) * 0.125)."""
    a = code.astype(np.float16)
    b = (a.astype(np.float32) - np.float32(0.5)).astype(np.float16)
    return (b.astype(np.float32) * np.float32(0.125)).astype(np.float16).astype(np.float32)


def build_features(proxy, g, seed, style, tone, structure, skin, auto_mask, history=None):
    """proxy [valid_h, valid_w, 3] code values 0..1 -> f32 [full_rows, 16] on the padded field.

    history: the reprojected previous-frame stored history in code values [valid_h, valid_w, 3]
    (already run through nr_history.reproject_history), or None for a first frame — lanes 7-9
    then copy lanes 4-6 (design.md D10).
    """
    vh, vw = proxy.shape[:2]
    full_w, full_h = g['full_width'], g['full_height']
    xs = np.arange(full_w, dtype=np.int64)
    ys = np.arange(full_h, dtype=np.int64)
    ref_x = np.where(xs < vw, xs, 2 * vw - xs - 2).clip(0, vw - 1)
    ref_y = np.where(ys < vh, ys, 2 * vh - ys - 2).clip(0, vh - 1)
    px = proxy[ref_y][:, ref_x]                                   # [full_h, full_w, 3], mirrored
    centered = center_proxy(px)

    gx, gy = np.meshgrid(np.arange(full_w, dtype=np.uint64), np.arange(full_h, dtype=np.uint64))
    n0, n1, n2 = gaussian3(u32(gx), u32(gy), seed)

    features = np.zeros((full_h * full_w, 16), dtype=np.float32)
    f = features.reshape(full_h, full_w, 16)
    f[..., 0] = n0
    f[..., 1] = n1
    f[..., 2] = n2
    f[..., 3] = 1.0
    f[..., 4:7] = centered
    if history is None:
        f[..., 7:10] = centered                                   # first frame: history = proxy
    else:
        hx = history[ref_y][:, ref_x]                             # mirrored the same way (frame.wgsl)
        f[..., 7:10] = center_proxy(hx)
    f[..., 10] = style / 128.0
    f[..., 11] = np.float16(tone).astype(np.float32)
    if auto_mask:
        f[..., 12] = np.float16(1.0).astype(np.float32)
        f[..., 13] = np.float16(structure if skin < 0 else skin).astype(np.float32)
        f[..., 14] = np.float16(structure).astype(np.float32)
    else:
        f[..., 12] = np.float16(structure).astype(np.float32)
        f[..., 13] = np.float16(-1.0).astype(np.float32)
        f[..., 14] = np.float16(-1.0).astype(np.float32)
    return features


def save_png(path, rgb01):
    Image.fromarray((np.clip(rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8)).save(path)
    print('  wrote', path)


def main():
    parser = argparse.ArgumentParser(description='run the teacher on an image and render the result')
    parser.add_argument('input', help='input image (any format PIL reads)')
    parser.add_argument('--width', type=int, default=512, help='valid width to resize to')
    parser.add_argument('--height', type=int, default=512, help='valid height to resize to')
    parser.add_argument('--weights', default=os.environ.get('NR_WEIGHTS'), help='model directory')
    parser.add_argument('-o', '--out', default='out', help='output directory')
    parser.add_argument('--seed', type=int, default=12345, help='noise seed (per frame)')
    parser.add_argument('--style', type=float, default=0.0, help='style id (lane 10 = style/128)')
    parser.add_argument('--tone', type=float, default=0.5, help='local tone 0..1 (lane 11)')
    parser.add_argument('--structure', type=float, default=0.5, help='local structure 0..1 (lane 12)')
    parser.add_argument('--skin', type=float, default=-1.0, help='skin structure (lane 13 with auto-mask)')
    parser.add_argument('--automask', action='store_true', help='auto-mask conditioning on')
    parser.add_argument('--prev', help="previous frame's stored history (.npy [h,w,3] code values, "
                                       'as written by --emit-history; 8-bit PNG also accepted)')
    parser.add_argument('--motion', help='motion field .npy [h,w,2], uv offsets y down (current -> previous); '
                                         'defaults to zero (static)')
    parser.add_argument('--history-source', choices=['output', 'proxy'], default='output',
                        help='what --prev holds: the previous composite output (default, per docs/network.md) '
                             'or the previous proxy (ablation)')
    parser.add_argument('--emit-history', help="write the next frame's stored history (.npy) here")
    args = parser.parse_args()

    if not args.weights:
        print('no weights: pass --weights or set NR_WEIGHTS')
        return 2

    image = Image.open(args.input).convert('RGB').resize((args.width, args.height), Image.LANCZOS)
    proxy = np.asarray(image, dtype=np.float32) / 255.0

    g = geometry_from_valid(args.width, args.height)
    print(f"image {args.width}x{args.height} -> field {g['full_width']}x{g['full_height']} "
          f"({g['full_rows']} rows)")

    # ---- The temporal history (design.md D10): stored previous output, reprojected, centred into 7-9.
    history = None
    repro = None
    if args.prev:
        if args.prev.lower().endswith('.npy'):
            history = np.load(args.prev).astype(np.float32)
        else:
            history = np.asarray(Image.open(args.prev).convert('RGB'), dtype=np.float32) / 255.0
        if history.shape != (args.height, args.width, 3):
            raise SystemExit(f'--prev shape {history.shape} != {(args.height, args.width, 3)}')
        motion = (np.load(args.motion).astype(np.float32) if args.motion
                  else np.zeros((args.height, args.width, 2), np.float32))
        if motion.shape != (args.height, args.width, 2):
            raise SystemExit(f'--motion shape {motion.shape} != {(args.height, args.width, 2)}')
        repro = reproject_history(history, motion)
        print(f'history: source={args.history_source}  mean |reprojected - prev| '
              f'{np.abs(repro - history).mean():.5f}')
    elif args.motion:
        raise SystemExit('--motion needs --prev')

    # --history-source only says what --prev holds (previous output vs previous proxy); either way the
    # lanes get centre(reprojected --prev), and with 'proxy' the composite blend is an ablation too.
    features_np = build_features(proxy, g, args.seed, args.style, args.tone,
                                 args.structure, args.skin, args.automask, history=repro)
    features = torch.from_numpy(features_np).to(nr_torch.DEVICE)

    model = Model(nr_torch.DEVICE).load(args.weights)
    network = Network(model, g)
    head = network.record(features)
    torch.cuda.synchronize()
    head_np = head.float().cpu().numpy().reshape(g['full_height'], g['full_width'], 4)
    valid = head_np[:args.height, :args.width]                    # the valid rect, in field coordinates
    rgb = valid[..., :3] / 4.0
    neural = np.clip(proxy + rgb, 0, 1)
    sigmoid = 1.0 / (1.0 + np.exp(-valid[..., 3].astype(np.float64)))
    blend = sigmoid
    if repro is not None:
        # The temporal blend, and the weight is the network's own (clipped by the model's blend scale).
        blend = np.clip(sigmoid * float(model.blend_scale()), 0, 1)
        neural = neural + (repro.astype(np.float64) - neural) * blend[..., None]

    print(f'head rgb/4: min {rgb.min():+.3f}  max {rgb.max():+.3f}  mean {rgb.mean():+.3f}')
    print(f'blend logit: min {valid[..., 3].min():+.3f}  max {valid[..., 3].max():+.3f}  '
          f'sigmoid mean {sigmoid.mean():.3f}  applied mean {blend.mean():.3f}')

    if args.emit_history:
        stored = store_history(neural.astype(np.float32))         # truncated toward zero, not rounded
        np.save(args.emit_history, stored)
        print('  wrote', args.emit_history)

    os.makedirs(args.out, exist_ok=True)
    save_png(os.path.join(args.out, 'proxy.png'), proxy)
    save_png(os.path.join(args.out, 'neural.png'), neural)
    save_png(os.path.join(args.out, 'blend.png'), np.repeat(blend[..., None], 3, axis=2))
    side = np.concatenate([proxy, neural, np.repeat(blend[..., None], 3, axis=2)], axis=1)
    save_png(os.path.join(args.out, 'side_by_side.png'), side)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
