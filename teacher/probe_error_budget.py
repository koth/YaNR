"""s-t 误差预算:在指定尺寸下把教师/学生改动的差拆成低频与高频,算各占多少 MSE。

回答两个问题:
  1. s-t 的差距里,低频(色调/结构)占多少、高频(纹理)占多少?
  2. 若低频被完全复刻,s-t 能到多少?(给出改进上限,避免瞎优化)

低频分界用 blur3^9(与训练 hf() 同口径)。尺寸可指定,320x512 即训练尺寸。

    python3 analyze_budget.py --size 320 512
    python3 analyze_budget.py --size 0 0      # 0 0 = 各图原尺寸
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/home/koth/klss/teacher')
from nr_geometry import geometry_from_valid                  # noqa: E402
from nr_model import Model                                    # noqa: E402
from nr_network import Network                                # noqa: E402
from nr_student import StudentNetwork, load_student_shape     # noqa: E402
from run_image import build_features                          # noqa: E402

CKPT = '/home/koth/keep_tmp/v45d_56k/ckpt.pt'
SHAPE = '/home/koth/klss/shapes/student_shared_moe_v1.json'
RECIPE = dict(seed=12345, style=2.0, tone=1.0, structure=1.0, skin=-1.0, automask=True)


def blur3(a):
    h, w, _ = a.shape
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    s = np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            s += p[i:i + h, j:j + w]
    return s / 9.0


def lowpass(a, it=9):
    y = a
    for _ in range(it):
        y = blur3(y)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, nargs=2, default=[320, 512])
    ap.add_argument('--samples', default='/home/koth/klss/teacher/samples')
    args = ap.parse_args()
    want_w, want_h = args.size

    shape = load_student_shape(SHAPE)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    state = ck.get('ema', ck.get('model'))
    print(f'{"目录":<6}{"尺寸":>11}{"s-t":>7}{"低误差":>9}{"高误差":>9}'
          f'{"低频占比":>9}{"alpha低":>9}{"alpha高":>9}{"若低频完美":>11}')
    for d in sorted(x for x in glob.glob(os.path.join(args.samples, 't*'))
                    if os.path.isdir(x)):
        src = [f for f in glob.glob(os.path.join(d, '*'))
               if os.path.basename(f) not in ('nr.png', 'teacher.png')][0]
        img = Image.open(src).convert('RGB')
        w, h = (want_w, want_h) if want_w else img.size
        proxy = np.asarray(img.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
        g = geometry_from_valid(w, h)
        feats = torch.from_numpy(build_features(
            proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
            RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])).to('cuda')

        teacher = Network(Model('cuda').load('/home/koth/klss/models/nr'), g)
        with torch.no_grad():
            head = teacher.record(feats).float().cpu().numpy().reshape(
                g['full_height'], g['full_width'], 4)
        tout = np.clip(proxy + head[:h, :w, :3] / 4.0, 0, 1)
        del teacher, head
        torch.cuda.empty_cache()

        torch.manual_seed(0)
        student = StudentNetwork(shape, g).to('cuda').eval()
        student.load_state_dict(state)
        with torch.no_grad():
            head = student(feats).float().cpu().numpy().reshape(
                g['full_height'], g['full_width'], 4)
        sout = np.clip(proxy + head[:h, :w, :3] / 4.0, 0, 1)
        del student, head, feats
        torch.cuda.empty_cache()

        dt, ds = tout - proxy, sout - proxy
        e_low = float(((lowpass(dt) - lowpass(ds)) ** 2).mean())
        e_high = float((((dt - lowpass(dt)) - (ds - lowpass(ds))) ** 2).mean())
        e_tot = float(((tout - sout) ** 2).mean())
        s_t = 10 * np.log10(1 / max(e_tot, 1e-12))
        # 若低频部分被完美复刻,只剩高频误差 -> 上限
        only_high = 10 * np.log10(1 / max(e_high, 1e-12))

        def alpha(a, b):
            x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
            return float((x * y).mean() / max((x * x).mean(), 1e-12))

        print(f'{os.path.basename(d):<6}{w}x{h:<7}{s_t:7.2f}{e_low * 1e4:9.2f}'
              f'{e_high * 1e4:9.2f}{100 * e_low / max(e_tot, 1e-12):8.0f}%'
              f'{alpha(lowpass(dt), lowpass(ds)):9.2f}{alpha(dt - lowpass(dt), ds - lowpass(ds)):9.2f}'
              f'{only_high:11.2f}', flush=True)


if __name__ == '__main__':
    main()
