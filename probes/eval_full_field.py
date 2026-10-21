"""按仓库正确协议评估:整图原生分辨率、一次前向。

对齐 teacher/render_samples.py 的做法(它能把 teacher.png 复现到 58.85 dB):
    g = geometry_from_valid(w, h)                     # 原生尺寸的几何
    feats = build_features(proxy, g, 12345, 2, 1, 1, -1, True)
    head  = model(feats)                              # 整图一次前向
    out   = clip(proxy + head[:h, :w, :3] / 4)

之前的 probes/eval_native.py 切成 512x320 小块,改变了 geometry,
多尺度 lanes 全部按错误尺度计算,教师只能复现到 30 dB —— 那是我的错。

    CUDA_VISIBLE_DEVICES=0 python3 probes/eval_full_field.py <ckpt>
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

RECIPE = dict(seed=12345, style=2.0, tone=1.0, structure=1.0, skin=-1.0, automask=True)


def load_png(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0


def psnr(a, b):
    return 10 * np.log10(1 / max(float(np.mean((a - b) ** 2)), 1e-12))


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else '/f/yanr/keep_tmp/run_cnn_fast_1/ckpt.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print('ckpt step', ckpt.get('step'), '| 整图一次前向 | seed', RECIPE['seed'], flush=True)

    rows = []
    for directory in sorted((ROOT / 'teacher/samples').glob('t*')):
        ref_path = directory / 'teacher.png'
        source = next((p for p in directory.iterdir()
                       if p.suffix.lower() in ('.png', '.jpg')
                       and p.stem not in ('teacher', 'nr')), None)
        if ref_path is None or not ref_path.exists() or source is None:
            continue
        image = Image.open(source).convert('RGB')
        w, h = image.size
        proxy = np.asarray(image, dtype=np.float32) / 255.0
        g = geometry_from_valid(w, h)
        feats = build_features(proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
                               RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])

        ann = TeacherAnnotator(str(ROOT / 'models/nr'), g)
        ann.g = ann.network.g = g
        with torch.inference_mode():
            t_head = ann.annotate(np.ascontiguousarray(feats), capture=(), batch=1)['head']
        t_head = t_head.reshape(g['full_height'], g['full_width'], 4)

        model = CNNUNetFast(load_fast_shape(ROOT / 'shapes/cnn_unet_fast.json'), g)
        model.load_state_dict(ckpt['ema'] if 'ema' in ckpt else ckpt['model'])
        model.eval().switch_to_deploy().cuda()
        model.g = g
        with torch.inference_mode():
            s_head = model(torch.from_numpy(np.ascontiguousarray(feats)).cuda())
        s_head = s_head.float().cpu().numpy().reshape(g['full_height'], g['full_width'], 4)
        del model
        torch.cuda.empty_cache()

        mine = np.clip(proxy + t_head[:h, :w, :3] / 4.0, 0, 1)
        stud = np.clip(proxy + s_head[:h, :w, :3] / 4.0, 0, 1)
        ref = load_png(ref_path)
        if ref.shape != mine.shape:
            print(f'{directory.name}: 参照尺寸不符 {ref.shape} vs {mine.shape}')
            continue
        rows.append((directory.name, psnr(stud, ref), psnr(mine, ref), psnr(proxy, ref)))
        print(f'{directory.name:<4} {w}x{h:<5} 学生 {rows[-1][1]:6.2f} | '
              f'我的教师 {rows[-1][2]:6.2f} | 输入 {rows[-1][3]:6.2f} | '
              f'净提升 {rows[-1][1] - rows[-1][3]:+5.2f}', flush=True)

    if rows:
        arr = np.array([r[1:] for r in rows])
        print(f'均值  学生 {arr[:, 0].mean():6.2f} | 教师一致性 {arr[:, 1].mean():6.2f} | '
              f'输入 {arr[:, 2].mean():6.2f} | 净提升 {(arr[:, 0] - arr[:, 2]).mean():+5.2f}')


if __name__ == '__main__':
    main()
