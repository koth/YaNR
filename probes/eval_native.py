"""原分辨率评估:铺块覆盖整图(不缩放),与引擎的 nr.png 口径对齐。

同时输出三条误差,用来分辨"引擎低"到底是哪一环:
  1. 学生 vs teacher.png      —— 与引擎 nr.png 同口径,可直接比
  2. 我的教师 vs teacher.png  —— 若这条不高,说明我的 proxy/控制项与生成
                                teacher.png 的那套不一致,后面的数都不可比
  3. 输入 vs teacher.png      —— 基线

seed 用 12345,与 engine/run_samples.sh 的默认值一致。
    CUDA_VISIBLE_DEVICES=0 python3 probes/eval_native.py <ckpt>
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path('/home/koth/klss')
sys.path.insert(0, str(ROOT / 'teacher'))

from annotate import TeacherAnnotator              # noqa: E402
from nr_cnn_unet_fast import CNNUNetFast, load_fast_shape  # noqa: E402
from nr_geometry import geometry_from_valid        # noqa: E402
from run_image import build_features               # noqa: E402
from single_frame_data import read_rgb, RESAMPLE   # noqa: E402

W, H = 512, 320
STEP_X, STEP_Y = W // 2, H // 2
SEED = 12345
CONTROLS = (2., 1., 1., -1., True)


def load_png(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else '/f/yanr/keep_tmp/run_cnn_fast_1/ckpt.pt'
    g = geometry_from_valid(W, H)
    ann = TeacherAnnotator(str(ROOT / 'models/nr'), g)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = CNNUNetFast(load_fast_shape(ROOT / 'shapes/cnn_unet_fast.json'), g)
    model.load_state_dict(ckpt['ema'] if 'ema' in ckpt else ckpt['model'])
    model.eval().switch_to_deploy().cuda()
    model.g = g
    print('ckpt step', ckpt.get('step'), '| 原分辨率铺块 | seed', SEED, flush=True)

    rows = []
    for directory in sorted((ROOT / 'teacher/samples').glob('t*')):
        teacher_png = directory / 'teacher.png'
        source = next((p for p in directory.iterdir()
                       if p.suffix.lower() in ('.png', '.jpg')
                       and p.stem not in ('teacher', 'nr')), None)
        if not teacher_png.exists() or source is None:
            continue
        ref = load_png(teacher_png)
        image = read_rgb(source)
        canvas = np.asarray(image, dtype=np.float32) / 255
        ih, iw = canvas.shape[:2]
        if (ih, iw) != ref.shape[:2]:
            print(f'{directory.name}: 尺寸不匹配 {iw}x{ih} vs ref,跳过')
            continue
        pad_w = (-(iw - W)) % STEP_X
        pad_h = (-(ih - H)) % STEP_Y
        padded = np.pad(canvas, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        se_s = se_t = se_b = 0.0
        blocks = 0
        for top in range(0, padded.shape[0] - H + 1, STEP_Y):
            for left in range(0, padded.shape[1] - W + 1, STEP_X):
                proxy = padded[top:top + H, left:left + W]
                feats = build_features(proxy, g, SEED, *CONTROLS)
                ann.g = ann.network.g = g
                with torch.inference_mode():
                    t_head = ann.annotate(feats.reshape(-1, 16), capture=(), batch=1)['head']
                with torch.inference_mode():
                    s_head = model(torch.from_numpy(feats).cuda()).cpu().numpy()
                t_c = np.clip(proxy + t_head.reshape(H, W, 4)[:, :, :3] / 4.0, 0, 1)
                s_c = np.clip(proxy + s_head.reshape(H, W, 4)[:, :, :3] / 4.0, 0, 1)
                r = ref[top:top + H, left:left + W]
                # 只统计与真实图像区域重叠的部分(边缘补齐区不计)
                valid_h = min(H, ih - top)
                valid_w = min(W, iw - left)
                t_c, s_c, rv = t_c[:valid_h, :valid_w], s_c[:valid_h, :valid_w], r[:valid_h, :valid_w]
                bv = proxy[:valid_h, :valid_w]
                se_s += float(np.sum((s_c - rv) ** 2))
                se_t += float(np.sum((t_c - rv) ** 2))
                se_b += float(np.sum((bv - rv) ** 2))
                blocks += s_c.size
        p = 10 * np.log10(1 / max(se_s / blocks, 1e-12))
        pt = 10 * np.log10(1 / max(se_t / blocks, 1e-12))
        pb = 10 * np.log10(1 / max(se_b / blocks, 1e-12))
        rows.append((directory.name, p, pt, pb))
        print(f'{directory.name:<4} {iw}x{ih:<5} 学生 {p:6.2f} | 我的教师 vs teacher.png '
              f'{pt:6.2f} | 输入 {pb:6.2f} | 净提升 {p - pb:+5.2f}', flush=True)

    if rows:
        arr = np.array([r[1:] for r in rows])
        print(f'均值  学生 {arr[:, 0].mean():6.2f} | 教师一致性 {arr[:, 1].mean():6.2f} | '
              f'输入 {arr[:, 2].mean():6.2f} | 净提升 {(arr[:, 0] - arr[:, 2]).mean():+5.2f}')


if __name__ == '__main__':
    main()
