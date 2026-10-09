#!/usr/bin/env python3
"""四联图管线(6.5):proxy / neural / diff×12 / blend,教师-学生并排版。

    python3 quad_compare.py --image in.jpg --size 512 --checkpoint v11.pt \
        --shape ../shapes/student_slim_mixed.json --teacher-weights <nr> -o ../teacher/samples/

输出:
  <name>_quad.png               2 行 × 4 列:上排教师、下排学生;列 = proxy / neural /
                                diff×12(|neural−proxy|×12,各自添加的细节)/ blend 权重图
  <name>_vs_teacher_diff_x12.png  学生−教师差 ×12(误差落点)

与 eval_temporal.composite、run_image.build_features 同口径(单帧,无历史)。
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_temporal import composite                      # noqa: E402
from nr_geometry import geometry_from_valid              # noqa: E402
from nr_model import Model                               # noqa: E402
from nr_network import Network                           # noqa: E402
from nr_student import StudentNetwork, load_student_shape  # noqa: E402
from nr_torch import DEVICE                              # noqa: E402
from run_image import build_features, save_png           # noqa: E402


def blend_map(head_np, h, w, scale):
    valid = head_np[:h, :w]
    sigmoid = 1.0 / (1.0 + np.exp(-valid[..., 3].astype(np.float64)))
    return np.clip(sigmoid * float(scale), 0, 1).astype(np.float32)


def gutter(a, b, px=4):
    pad = np.full((a.shape[0], px, 3), 1.0, np.float32)
    return np.concatenate([a, pad, b], axis=1)


def main():
    parser = argparse.ArgumentParser(description='teacher vs student quad figure (6.5)')
    parser.add_argument('--image', required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--shape', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--teacher-weights', default=os.environ.get('NR_WEIGHTS'))
    parser.add_argument('--name', default=None, help='输出文件名前缀(默认取图片名)')
    parser.add_argument('-o', '--out', default='.')
    args = parser.parse_args()

    size = args.size
    g = geometry_from_valid(size, size)
    name = args.name or os.path.splitext(os.path.basename(args.image))[0]

    proxy = np.asarray(Image.open(args.image).convert('RGB').resize((size, size), Image.LANCZOS),
                       dtype=np.float32) / 255.0
    feats = build_features(proxy, g, args.seed, 1.5, 0.5, 0.35, -1.0, False)
    f = torch.from_numpy(feats).to(DEVICE)

    tmodel = Model(DEVICE).load(args.teacher_weights)
    network = Network(tmodel, g)
    shape = load_student_shape(args.shape)
    torch.manual_seed(0)
    student = StudentNetwork(shape, g).to(DEVICE).eval()
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ckpt.get('ema', ckpt.get('model', ckpt)))

    with torch.no_grad():
        head_t = network.record(f).float().cpu().numpy().reshape(g['full_height'],
                                                                 g['full_width'], 4)
        head_s = student(f).float().cpu().numpy().reshape(g['full_height'], g['full_width'], 4)

    rows = []
    diffs = {}
    for tag, head_np, scale in (('teacher', head_t, tmodel.blend_scale()),
                                ('student', head_s,
                                 float(student.blend_scale.detach().cpu()))):
        neural = composite(head_np, proxy, size, size, None, scale)
        diff = np.clip(np.abs(neural - proxy) * 12, 0, 1).astype(np.float32)
        blend = blend_map(head_np, size, size, scale)
        diffs[tag] = neural
        rows.append(np.concatenate([
            gutter(proxy, neural), np.full((size, 4, 3), 1.0, np.float32),
            gutter(diff, np.repeat(blend[..., None], 3, axis=2))], axis=1))

    os.makedirs(args.out, exist_ok=True)
    quad = np.concatenate([rows[0], np.full((4, rows[0].shape[1], 3), 1.0, np.float32),
                           rows[1]], axis=0)
    save_png(os.path.join(args.out, f'{name}_quad.png'), quad)
    vs = np.clip(np.abs(diffs['student'] - diffs['teacher']) * 12, 0, 1)
    save_png(os.path.join(args.out, f'{name}_vs_teacher_diff_x12.png'), vs)
    print(f'wrote {args.out}/{name}_quad.png  ({quad.shape[1]}x{quad.shape[0]})  '
          f'+ {name}_vs_teacher_diff_x12.png')


if __name__ == '__main__':
    raise SystemExit(main())
