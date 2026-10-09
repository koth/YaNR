#!/usr/bin/env python3
"""Parity + bench: C++ engine vs torch student forward (openspec tasks 8.4/8.8).

    python3 check_engine.py --image /tmp/hist_exp/f000.png --size 512
    python3 check_engine.py --image <png> --size 512 --checkpoint runs/v1/ckpt.pt --bench

流程:build_features -> torch 参考 head -> 写 features.bin -> cpu_engine -> 比对 head。
口径:C++ 是 f32 重排累加,与 torch 不逐位;PASS 线 = max|diff| ≤ 2e-3(head 原始刻度,
折合合成图 PSNR ≥ ~45dB)。
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_geometry import geometry_from_valid                  # noqa: E402
from nr_student import StudentNetwork, load_student_shape    # noqa: E402
from run_image import build_features                         # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='C++ engine vs torch parity')
    parser.add_argument('--image', required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--shape', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        '..', 'shapes', 'student_v0.json'))
    parser.add_argument('--checkpoint')
    parser.add_argument('--engine-build', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'engine', 'build'))
    parser.add_argument('--weights-dir', default=None,
                        help='where to dump weights (default: temp dir)')
    parser.add_argument('--bench', action='store_true', help='also run cpu_engine --bench')
    parser.add_argument('--bisect', action='store_true',
                        help='compare per-stage captures to locate the first divergent stage')
    parser.add_argument('--tolerance', type=float, default=2e-3)
    args = parser.parse_args()

    size = args.size
    g = geometry_from_valid(size, size)
    torch.manual_seed(0)
    shape = load_student_shape(args.shape)
    model = StudentNetwork(shape, g).eval()
    model.capture_enabled = True
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt.get('ema', ckpt.get('model', ckpt)))

    proxy = np.asarray(Image.open(args.image).convert('RGB').resize((size, size), Image.LANCZOS),
                       dtype=np.float32) / 255.0
    features = build_features(proxy, g, args.seed, 1.5, 0.5, 0.35, -1.0, False)

    hook_store = {}
    d0_blocks = list(model.enc_blocks['d0'])
    d0_blocks[0].register_forward_pre_hook(
        lambda mod, inp: hook_store.__setitem__('trans-d0', inp[0].detach().numpy()))
    for tag, blk in [('enc_blocks.d0.%d' % i, b) for i, b in enumerate(d0_blocks)] + \
                    [('vit_blocks.0', model.vit_blocks[0])]:
        blk.register_forward_hook(
            lambda mod, inp, out, tag=tag: hook_store.__setitem__('blk-' + tag,
                                                                 out.detach().numpy()))
        if getattr(blk, 'tail', None) is not None:
            blk.tail.register_forward_pre_hook(
                lambda mod, inp, tag=tag: hook_store.__setitem__('ffn-' + tag, inp[0].detach().numpy()))
            blk.tail.attention.register_forward_hook(
                lambda mod, inp, out, tag=tag: hook_store.__setitem__('att-' + tag,
                                                                     out.detach().numpy()))
        elif getattr(blk, 'attention', None) is not None:      # VitBlock
            blk.attention.register_forward_hook(
                lambda mod, inp, out, tag=tag: hook_store.__setitem__('att-' + tag,
                                                                     out.detach().numpy()))
    model.vit_in.register_forward_hook(
        lambda mod, inp, out: hook_store.__setitem__('trans-vit', out.detach().numpy()))

    with torch.no_grad():
        ref = model(torch.from_numpy(features)).numpy()
    captures = {k: v.numpy() for k, v in model.captures.items()} if model.capture_enabled else {}
    captures.update(hook_store)

    tmp = args.weights_dir or tempfile.mkdtemp(prefix='engine_')
    os.makedirs(tmp, exist_ok=True)
    prefix = os.path.join(tmp, 'student')
    dump = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dump_weights.py')
    subprocess.run([sys.executable, dump, '--shape', args.shape, '--size', str(size),
                    '-o', prefix] + (['--checkpoint', args.checkpoint] if args.checkpoint else []),
                   check=True, capture_output=True)

    feat_path = os.path.join(tmp, 'features.bin')
    features.astype(np.float32).tofile(feat_path)
    head_path = os.path.join(tmp, 'head.bin')
    tool = os.path.join(args.engine_build, 'cpu_engine')
    if args.bench:
        bench_cmd = [tool, '--idx', prefix + '.idx', '--features', feat_path,
                     '--bench', '--repeats', '30']
        print(subprocess.run(bench_cmd, check=True, capture_output=True, text=True).stdout.strip())
    cmd = [tool, '--idx', prefix + '.idx', '--features', feat_path, '--out', head_path]
    if args.bisect:
        os.makedirs(os.path.join(tmp, 'dumps'), exist_ok=True)
        cmd += ['--dump-dir', os.path.join(tmp, 'dumps')]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)

    if args.bisect:
        print('stage-by-stage:')
        for name in ('s-enc-full', 'trans-d0', 'ffn-enc_blocks.d0.0', 'att-enc_blocks.d0.0',
                     'blk-enc_blocks.d0.0', 'ffn-enc_blocks.d0.1', 'att-enc_blocks.d0.1',
                     'blk-enc_blocks.d0.1', 's-enc-d0', 's-enc-d1',
                     's-enc-d2', 's-enc-d3', 's-enc-d4',
                     'trans-vit', 'att-vit_blocks.0', 'blk-vit_blocks.0', 's-vit',
                     's-dec-d4', 's-dec-d3', 's-dec-d2', 's-dec-d1', 's-dec-d0'):
            path = os.path.join(tmp, 'dumps', name + '.bin')
            if not os.path.exists(path) or name not in captures:
                continue
            got_s = np.fromfile(path, dtype=np.float32)
            ref_s = np.asarray(captures[name]).reshape(-1)
            d = np.abs(ref_s - got_s)
            print(f'  {name:22s} max {d.max():.3e}  mean {d.mean():.3e}')
    got = np.fromfile(head_path, dtype=np.float32).reshape(g['full_rows'], 4)

    diff = np.abs(ref - got)
    mse = float(((ref[:, :3] - got[:, :3]) ** 2).mean())
    psnr = 10 * np.log10(1.0 / max(mse / 16.0, 1e-12))
    print(f'max|diff| {diff.max():.3e}  mean {diff.mean():.3e}  '
          f'composite-equivalent PSNR {psnr:.1f} dB  ({res.stdout.strip()})')
    ok = diff.max() <= args.tolerance
    print('PASS' if ok else f'FAIL (tolerance {args.tolerance})')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
