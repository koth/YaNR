"""用学生渲染 teacher/samples/t1..t7,输出各目录下的 nr.png。

档位已由 teacher.png 反推确认(见 tmp/probe_sample_recipe.py):同一张图跑遍
候选组合,只有

    seed=12345  style=2.0  tone=1.0  structure=1.0  skin=-1.0  automask=True

复现 teacher.png 到 58.85 dB(maxdiff 0.002 = 8bit PNG 量化量级),其余组合
27-47 dB。所以这套组合就是当初生成 teacher.png 的配方,nr.png 才和它同口径。

全程不需要教师模型:build_features 是纯 numpy,teacher.png 直接读盘对比,
显存需求只有学生本身(约 1GB),训练在跑时也能用。

    python3 render_samples.py --samples ../teacher/samples --checkpoint xxx.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_geometry import geometry_from_valid                     # noqa: E402
from nr_student import StudentNetwork, load_student_shape       # noqa: E402
from run_image import build_features                            # noqa: E402

SHAPE = '/home/koth/klss/shapes/student_shared_moe_v1.json'
# 反推得到的配方(与 samples 里的 teacher.png 一致)
RECIPE = dict(seed=12345, style=2.0, tone=1.0, structure=1.0, skin=-1.0, automask=True)


def blur3(a):
    h, w, _ = a.shape
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    s = np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            s += p[i:i + h, j:j + w]
    return s / 9.0


def band(a, it):
    y = a
    for _ in range(it):
        y = blur3(y)
    return a - y


def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def texture(dt, ds):
    """coarse 频带(与训练 hf() 同口径)上的幅度比与相关性。"""
    return float(ds.std() / max(dt.std(), 1e-12)), float(np.corrcoef(dt.ravel(), ds.ravel())[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', default='/home/koth/klss/teacher/samples')
    parser.add_argument('--checkpoint', default='/home/koth/keep_tmp/v45d_56k/ckpt.pt')
    parser.add_argument('--name', default='nr.png', help='输出文件名')
    parser.add_argument('--only', nargs='+', help='只跑指定子目录')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dirs = sorted(d for d in glob.glob(os.path.join(args.samples, 't*')) if os.path.isdir(d))
    if args.only:
        dirs = [d for d in dirs if os.path.basename(d) in args.only]
    shape = load_student_shape(SHAPE)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('ema', ckpt.get('model'))
    print(f'checkpoint step {ckpt.get("step")}  device {device}  '
          f'recipe seed={RECIPE["seed"]} style={RECIPE["style"]} tone={RECIPE["tone"]} '
          f'struct={RECIPE["structure"]} skin={RECIPE["skin"]} am={RECIPE["automask"]}')
    print(f'{"目录":<6}{"尺寸":>11} {"psnr(t)":>8} {"amp":>6} {"corr":>6}')

    for d in dirs:
        files = [f for f in glob.glob(os.path.join(d, '*'))
                 if os.path.basename(f) not in (args.name, 'teacher.png')]
        if not files:
            print(f'{os.path.basename(d):<6} 找不到输入图')
            continue
        src = files[0]
        proxy_img = Image.open(src).convert('RGB')
        w, h = proxy_img.size
        proxy = np.asarray(proxy_img, dtype=np.float32) / 255.0
        g = geometry_from_valid(w, h)
        feats = torch.from_numpy(build_features(
            proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
            RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])).to(device)

        torch.manual_seed(0)
        student = StudentNetwork(shape, g).to(device).eval()
        student.load_state_dict(state)
        with torch.no_grad():
            head = student(feats)
            head = head.float().cpu().numpy().reshape(g['full_height'], g['full_width'], 4)
        out = np.clip(proxy + head[:h, :w, :3] / 4.0, 0, 1)
        del student, head, feats
        torch.cuda.empty_cache()

        Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)).save(os.path.join(d, args.name))

        line = f'{os.path.basename(d):<6}{w}x{h:<7}'
        tref = os.path.join(d, 'teacher.png')
        if os.path.exists(tref):
            t = np.asarray(Image.open(tref).convert('RGB'), dtype=np.float32) / 255.0
            dt = band(t, 9) - band(proxy, 9)
            ds = band(out, 9) - band(proxy, 9)
            amp, corr = texture(dt, ds)
            line += f' {psnr(out, t):8.2f} {amp:6.2f} {corr:6.2f}'
        print(line, flush=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
