"""把 teacher.png - input 与 nr.png - input 拆成低频(色调/结构)与高频(纹理),
分别看学生复刻了多少。

为什么要拆:全图 PSNR 显示学生只改了输入 3.4/255 RMS,而教师改了 12.9/255,
差 4 倍;但高频频带的幅度比却有 0.58-0.76。两者矛盾说明缺口主要在低频。
这里用 blur3^9 作分界(与训练 hf() 同口径),对两个分量分别算:

  ratio  学生改动标准差 / 教师改动标准差
  corr   两者的相关系数
  alpha  学生改动在教师改动方向上的投影系数 <ds,dt>/|dt|^2
         (1.0 = 完全复刻教师的改动;0 = 完全没动)

    python3 analyze_decomp.py <samples 目录>
"""
import glob
import os
import sys

import numpy as np
from PIL import Image


def load(p):
    return np.asarray(Image.open(p).convert('RGB'), dtype=np.float64) / 255.0


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


def stats(dt, ds, label):
    x, y = dt.ravel(), ds.ravel()
    xm, ym = x - x.mean(), y - y.mean()
    vx = float((xm * xm).mean())
    vy = float((ym * ym).mean())
    cov = float((xm * ym).mean())
    ratio = np.sqrt(vy / max(vx, 1e-12))
    corr = cov / np.sqrt(max(vx * vy, 1e-24))
    alpha = cov / max(vx, 1e-12)
    return (f'{label:<5} t={np.sqrt(vx) * 255:5.2f} s={np.sqrt(vy) * 255:5.2f} '
            f'ratio={ratio:5.2f} corr={corr:5.2f} alpha={alpha:5.2f}')


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '/home/koth/klss/teacher/samples'
    print(f'{"目录":<6}{"分量":<6}{"教师改动":>9}{"学生改动":>9}   比值  相关  投影')
    for d in sorted(x for x in glob.glob(os.path.join(root, 't*')) if os.path.isdir(x)):
        src = [f for f in glob.glob(os.path.join(d, '*'))
               if os.path.basename(f) not in ('nr.png', 'teacher.png')]
        if not src:
            continue
        proxy = load(src[0])
        teacher = load(os.path.join(d, 'teacher.png'))
        student = load(os.path.join(d, 'nr.png'))
        dt, ds = teacher - proxy, student - proxy
        dt_lo, ds_lo = lowpass(dt), lowpass(ds)
        name = os.path.basename(d)
        print(f'{name:<6}{stats(dt, ds, "全图")}')
        print(f'{"":<6}{stats(dt_lo, ds_lo, "低频")}')
        print(f'{"":<6}{stats(dt - dt_lo, ds - ds_lo, "高频")}', flush=True)


if __name__ == '__main__':
    main()
