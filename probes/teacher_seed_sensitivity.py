"""教师输出对噪声 lane(seed)的敏感度。

lanes 0-2 是 gaussian3(gx, gy, seed),逐像素噪声。如果教师输出对 seed 很敏感,
说明它的输出里含有"逐像素、依赖噪声"的成分 —— 而学生的 CNN 结构要经过 5 次
降采样,几乎不可能把这种逐像素信息保留到输出。那就会表现成我们测到的
amp≈0.7 / corr≈0.6(回归到均值),并且构成拟合上限。

    CUDA_VISIBLE_DEVICES=0 python3 probes/teacher_seed_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path('/home/koth/klss')
sys.path.insert(0, str(ROOT / 'teacher'))

from annotate import TeacherAnnotator              # noqa: E402
from nr_geometry import geometry_from_valid        # noqa: E402
from run_image import build_features               # noqa: E402

CONTROLS = (2., 1., 1., -1., True)
SEEDS = (12345, 999)
SAMPLES = ['t1', 't3', 't7']


def psnr(a, b):
    return 10 * np.log10(1 / max(float(np.mean((a - b) ** 2)), 1e-12))


def main():
    rows = []
    for name in SAMPLES:
        directory = ROOT / 'teacher/samples' / name
        source = next(p for p in directory.iterdir()
                      if p.suffix.lower() in ('.png', '.jpg') and p.stem not in ('teacher', 'nr'))
        image = Image.open(source).convert('RGB')
        w, h = image.size
        proxy = np.asarray(image, dtype=np.float32) / 255.0
        g = geometry_from_valid(w, h)
        outs = []
        for seed in SEEDS:
            feats = np.ascontiguousarray(build_features(proxy, g, seed, *CONTROLS))
            ann = TeacherAnnotator(str(ROOT / 'models/nr'), g)
            ann.g = ann.network.g = g
            with torch.inference_mode():
                head = ann.annotate(feats, capture=(), batch=1)['head']
            head = head.reshape(g['full_height'], g['full_width'], 4)
            outs.append(np.clip(proxy + head[:h, :w, :3] / 4.0, 0, 1))
        d = psnr(outs[0], outs[1])
        delta = float(np.sqrt(np.mean((outs[0] - outs[1]) ** 2)) * 255)
        rows.append(d)
        print(f'{name:<4} {w}x{h:<5} 两个 seed 之间 {d:6.2f} dB  '
              f'(逐像素差异 {delta:.2f}/255)', flush=True)
    print(f'均值 {np.mean(rows):.2f} dB')


if __name__ == '__main__':
    main()
