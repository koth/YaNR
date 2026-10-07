#!/usr/bin/env python3
"""ONNX export / parity / quantize / bench for the student (openspec tasks 8.1-8.2, 8.6).

学生算子全部标准(Linear/MatMul/softmax/where/index_put),导出即用。注意:窗口索引
是按几何在 trace 期固化的常量,所以**按边长导出固定形状模型**(部署性能最优),
每个目标边长一个 .onnx;blend_scale 导出为同名 .json 侧车(合成用)。

    python3 export_onnx.py export --shape ../shapes/student_v0.json --size 512 -o model512.onnx
    python3 export_onnx.py check  --onnx model512.onnx --size 512
    python3 export_onnx.py quantize --onnx model512.onnx -o model512.int8.onnx
    python3 export_onnx.py bench  --onnx model512.onnx --size 512 --repeats 30
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_geometry import geometry_from_valid                     # noqa: E402
from nr_student import StudentNetwork, load_student_shape      # noqa: E402
from run_image import build_features                           # noqa: E402


def build_model(shape_path, size):
    torch.manual_seed(0)                       # 随机初始化必须可复现,export/check 才是同一模型
    g = geometry_from_valid(size, size)
    shape = load_student_shape(shape_path)
    model = StudentNetwork(shape, g).eval()
    return model, g


def make_features(g, size, seed=7):
    proxy = np.random.default_rng(seed).random((size, size, 3), dtype=np.float32)
    return torch.from_numpy(build_features(proxy, g, seed, 0.0, 0.5, 0.5, -1.0, False))


def cmd_export(args):
    model, g = build_model(args.shape, args.size)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu',
                                         weights_only=False))
    features = make_features(g, args.size)
    torch.onnx.export(
        model, (features,), args.out, opset_version=17,
        input_names=['features'], output_names=['head'],
        dynamic_axes=None if not args.dynamic else {'features': {0: 'rows'}, 'head': {0: 'rows'}})
    side = os.path.splitext(args.out)[0] + '.json'
    with open(side, 'w') as fh:
        json.dump({'size': args.size, 'blend_scale': float(model.blend_scale.detach()),
                   'shape': args.shape, 'checkpoint': args.checkpoint}, fh, indent=1)
    print(f'wrote {args.out} (+ {side})')
    return 0


def cmd_check(args):
    import onnxruntime as ort
    size = args.size
    model, g = build_model(args.shape, size)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu',
                                         weights_only=False))
    features = make_features(g, size)
    with torch.no_grad():
        ref = model(features).numpy()
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    out = sess.run(['head'], {'features': features.numpy()})[0]
    diff = np.abs(ref - out)
    head = out.reshape(g['full_height'], g['full_width'], 4)[:size, :size]
    href = ref.reshape(g['full_height'], g['full_width'], 4)[:size, :size]
    mse = float(((head[..., :3] - href[..., :3]) ** 2).mean())
    psnr = 10 * np.log10(1.0 / max(mse / 16.0, 1e-12))          # rgb/4 注入后的合成图等效 PSNR
    print(f'max|diff| {diff.max():.6f}  mean {diff.mean():.6f}  composite-equivalent PSNR {psnr:.1f} dB')
    ok = diff.max() <= 1e-3
    print('PASS' if ok else 'FAIL (超过 1e-3)')
    return 0 if ok else 1


def cmd_quantize(args):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(args.onnx, args.out, weight_type=QuantType.QInt8)
    print(f'wrote {args.out}')
    return 0


def cmd_bench(args):
    import onnxruntime as ort
    size = args.size
    _, g = build_model(args.shape, size)
    features = make_features(g, size).numpy()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.onnx, sess_options=opts, providers=['CPUExecutionProvider'])
    for _ in range(3):
        sess.run(['head'], {'features': features})
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        sess.run(['head'], {'features': features})
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f'{args.onnx}  {size}²  threads {args.threads}  '
          f'median {times[len(times) // 2]:.2f} ms  min {times[0]:.2f} ms  '
          f'p90 {times[int(len(times) * 0.9)]:.2f} ms')
    return 0


def main():
    parser = argparse.ArgumentParser(description='student ONNX tooling')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('export')
    p.add_argument('--shape', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--size', type=int, required=True)
    p.add_argument('--dynamic', action='store_true')
    p.add_argument('-o', '--out', required=True)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser('check')
    p.add_argument('--shape', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   '..', 'shapes', 'student_v0.json'))
    p.add_argument('--checkpoint')
    p.add_argument('--onnx', required=True)
    p.add_argument('--size', type=int, required=True)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser('quantize')
    p.add_argument('--onnx', required=True)
    p.add_argument('-o', '--out', required=True)
    p.set_defaults(fn=cmd_quantize)

    p = sub.add_parser('bench')
    p.add_argument('--shape', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   '..', 'shapes', 'student_v0.json'))
    p.add_argument('--onnx', required=True)
    p.add_argument('--size', type=int, required=True)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--threads', type=int, default=0)
    p.set_defaults(fn=cmd_bench)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == '__main__':
    raise SystemExit(main())
