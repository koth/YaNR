"""跨代受控对比:v45d 56k(shared-routed v1)对 student_v44_preview_46k(moe_v0)。

两者架构与训练预算都不同,不能只靠原图尺寸的 7 张目测。这里在**训练分辨率
320x512** 上用同一批图、同一配方、同一套指标比,教师输出读缓存
(texcache_samples320),不需要 GPU、不影响正在跑的训练。

    python3 compare_lineages.py
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/home/koth/klss/teacher')
from nr_geometry import geometry_from_valid                  # noqa: E402
from nr_package import load_student_package                  # noqa: E402
from nr_student import StudentNetwork, load_student_shape     # noqa: E402

W, H = 320, 512
CACHE = '/home/koth/keep_tmp/texcache_samples320'
CKPT = '/home/koth/keep_tmp/v45d_56k/ckpt.pt'
SHAPE = '/home/koth/klss/shapes/student_shared_moe_v1.json'
PKG = '/home/koth/klss/weights/student_v44_preview_46k'
SAMPLES = '/home/koth/klss/teacher/samples'
STEM = '.{}x{}.s2.0t1.0k1.0'


def psnr(a, b):
    return 10 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))


def pyramid(a):
    levels = [a]
    while min(levels[-1].shape[:2]) >= 8:
        h, w, c = levels[-1].shape
        h2, w2 = h // 2, w // 2
        levels.append(levels[-1][:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, c).mean((1, 3)))
    bands = []
    for i in range(len(levels) - 1):
        up = np.repeat(np.repeat(levels[i + 1], 2, 0), 2, 1)[:levels[i].shape[0],
                                                             :levels[i].shape[1]]
        bands.append(levels[i] - up)
    bands.append(levels[-1])
    return bands


def evaluate(model, feats, proxy, teacher, g):
    with torch.no_grad():
        head = model(torch.from_numpy(feats).float()).float().cpu().numpy()
    head = head.reshape(g['full_height'], g['full_width'], 4)
    out = np.clip(proxy + head[:H, :W, :3] / 4.0, 0, 1)
    dt, ds = teacher - proxy, out - proxy
    bands = []
    for bt, bs in zip(pyramid(dt), pyramid(ds)):
        st = bt.std()
        corr = float(np.corrcoef(bt.ravel(), bs.ravel())[0, 1]) if st > 1e-9 else float('nan')
        bands.append((bs.std() / max(st, 1e-12), corr))
    return psnr(out, teacher), bands


def main():
    g = geometry_from_valid(W, H)
    shape = load_student_shape(SHAPE)
    torch.manual_seed(0)
    model_a = StudentNetwork(shape, g).eval()
    model_a.load_state_dict(torch.load(CKPT, map_location='cpu',
                                       weights_only=False).get('ema'))
    model_b, _ = load_student_package(PKG, W, H, 'cpu')
    model_b.eval()
    print('A = v45d 56k(shared-routed v1)   B = v44_preview_46k(moe_v0)\n')

    rows = []
    for name in sorted(os.path.basename(d) for d in glob.glob(os.path.join(SAMPLES, 't*'))
                       if os.path.isdir(d)):
        src = [f for f in glob.glob(os.path.join(SAMPLES, name, '*'))
               if os.path.basename(f) not in ('nr.png', 'teacher.png')][0]
        base = os.path.basename(src) + STEM.format(W, H)
        fp = os.path.join(CACHE, base + '.feats.npy')
        tp = os.path.join(CACHE, base + '.teacher.npy')
        if not (os.path.exists(fp) and os.path.exists(tp)):
            print(f'{name}: 缓存缺失,跳过')
            continue
        feats, teacher = np.load(fp), np.load(tp)
        proxy = np.asarray(Image.open(src).convert('RGB').resize((W, H), Image.LANCZOS),
                           np.float32) / 255.0
        a_st, a_bands = evaluate(model_a, feats, proxy, teacher, g)
        b_st, b_bands = evaluate(model_b, feats, proxy, teacher, g)
        rows.append({'name': name, 't_in': psnr(teacher, proxy),
                     'a_st': a_st, 'b_st': b_st, 'a': a_bands, 'b': b_bands})

    nb = len(rows[0]['a'])
    labels = [f'{2 ** i}-{2 ** (i + 1)}px' for i in range(nb - 1)] + ['残余']

    print(f'{"目录":<5}{"t-in":>7}{"A s-t":>8}{"B s-t":>8}{"差":>7}')
    for r in rows:
        print(f'{r["name"]:<5}{r["t_in"]:7.2f}{r["a_st"]:8.2f}{r["b_st"]:8.2f}'
              f'{r["b_st"] - r["a_st"]:+7.2f}')
    ma = float(np.mean([r['a_st'] for r in rows]))
    mb = float(np.mean([r['b_st'] for r in rows]))
    print(f'{"均值":<5}{"":>7}{ma:8.2f}{mb:8.2f}{mb - ma:+7.2f}   (正 = B 更好)')

    print(f'\n各尺度 amp / corr(7 张均值)')
    print(f'{"尺度":<10}{"A amp":>8}{"B amp":>8}{"A corr":>9}{"B corr":>9}')
    for i, lab in enumerate(labels):
        a_amp = np.nanmean([r['a'][i][0] for r in rows])
        b_amp = np.nanmean([r['b'][i][0] for r in rows])
        a_c = np.nanmean([r['a'][i][1] for r in rows])
        b_c = np.nanmean([r['b'][i][1] for r in rows])
        print(f'{lab:<10}{a_amp:8.2f}{b_amp:8.2f}{a_c:9.2f}{b_c:9.2f}')


if __name__ == '__main__':
    main()
