#!/usr/bin/env python3
"""打包校验(7.3):转换后的学生模型包加载 → 与训练导出的 torch 输出逐位一致(同输入同权重)。

    python3 check_package.py --package weights/student_mixed --checkpoint runs/v1/ckpt.pt \
        [--size 512] [--image in.png] [--seed 7]

口径:
  1. 权重往返:包内张量 vs ckpt(EMA 优先)张量,逐 tensor torch.equal;
  2. 前向往返:同一 features 同一设备,包重建的 StudentNetwork 与 ckpt 参考网络
     输出 torch.equal 逐位比对(严格,不做容差);
  3. sha256:read_package 默认逐 stage 校验。
任一不逐位一致即 FAIL。
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_geometry import geometry_from_valid                  # noqa: E402
from nr_package import load_student_package, read_package   # noqa: E402
from nr_student import StudentNetwork, load_student_shape   # noqa: E402
from run_image import build_features                        # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='student package round-trip check (7.3)')
    parser.add_argument('--package', required=True, help='convert_weights --shape student 的输出目录')
    parser.add_argument('--checkpoint', required=True, help='train_distill ckpt.pt(参考)')
    parser.add_argument('--shape', default=None, help='参考侧形状 DSL(默认从 ckpt config 探测)')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--image', default=None, help='输入图(默认确定性合成图)')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = geometry_from_valid(args.size, args.size)

    # ---- 参考:训练导出的 torch(EMA 优先,部署口径,同 dump_weights)。
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ref_state = ckpt.get('ema', ckpt.get('model', ckpt))
    shape_path = args.shape or (ckpt.get('config', {}) or {}).get('shape')
    if not shape_path or not os.path.isfile(shape_path):
        raise SystemExit(f'shape json not found: {shape_path!r}(pass --shape)')
    torch.manual_seed(0)
    ref_model = StudentNetwork(load_student_shape(shape_path), g).to(device).eval()
    ref_model.load_state_dict(ref_state)

    # ---- 包:sha256 校验 + 权重往返。
    pkg_state, manifest = read_package(args.package)
    pkg_model, _ = load_student_package(args.package, args.size, args.size, device)
    print(f"package: {manifest['model']['version']}  {manifest['model']['params']:,} params  "
          f"{len(manifest['tensors'])} tensors, sha256 全绿")

    bad = [name for name, tensor in pkg_state.items()
           if name not in ref_state or not torch.equal(tensor, ref_state[name].cpu())]
    bad += [name for name in ref_state if name not in pkg_state]
    print(f'权重往返: {len(pkg_state)} tensors, 不逐位一致 {len(bad)}'
          + (f'  <- {bad[:5]}' if bad else ''))

    # ---- 前向往返:同一 features(torch 侧确定性)。
    if args.image:
        from PIL import Image
        proxy = np.asarray(Image.open(args.image).convert('RGB')
                           .resize((args.size, args.size), Image.LANCZOS), dtype=np.float32) / 255.0
    else:
        yy, xx = np.mgrid[0:args.size, 0:args.size].astype(np.float32)
        proxy = np.stack([xx / (args.size - 1), yy / (args.size - 1),
                          (xx + yy) / (2 * (args.size - 1))], axis=2).astype(np.float32)
    features = torch.from_numpy(
        build_features(proxy, g, args.seed, 1.5, 0.5, 0.35, -1.0, False)).to(device)

    with torch.no_grad():
        ref_head = ref_model(features)
        pkg_head = pkg_model(features)
    max_diff = float((ref_head - pkg_head).abs().max())
    bit_equal = bool(torch.equal(ref_head, pkg_head))
    print(f'前向往返: head {tuple(ref_head.shape)}  torch.equal '
          f'{"逐位一致" if bit_equal else "不一致"}  max|diff| {max_diff:.3e}')

    ok = not bad and bit_equal
    print('PACKAGE PASS' if ok else 'PACKAGE FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
