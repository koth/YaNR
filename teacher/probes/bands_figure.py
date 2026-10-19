"""把"高低频"画出来:同一处改动拆成低频与高频,并排放对比。

面板布局(2 行 x 3 列):
  上行 = 教师:  输入 | 教师输出 | 教师改动 x8
  下行 = 拆解:  低频部分 x8 | 高频部分 x8 | 学生改动 x8
低频 = blur3^9(σ≈2.45px),高频 = 改动 - 低频。改动放大 8 倍才看得清。
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

W, H = 320, 512
GAIN = 8.0
SAMPLE = sys.argv[1] if len(sys.argv) > 1 else 't3'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/tmp/bands.png'
RECIPE_DIR = '/home/koth/klss/teacher/samples'


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


def to8(a):
    return np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)


def main():
    d = os.path.join(RECIPE_DIR, SAMPLE)
    src = [f for f in glob.glob(d + '/*')
           if os.path.basename(f) not in ('nr.png', 'teacher.png')][0]
    proxy = np.asarray(Image.open(src).convert('RGB').resize((W, H), Image.LANCZOS),
                       dtype=np.float32) / 255.0
    teacher = np.asarray(Image.open(os.path.join(d, 'teacher.png')).convert('RGB').resize(
        (W, H), Image.LANCZOS), dtype=np.float32) / 255.0
    student = np.asarray(Image.open(os.path.join(d, 'nr.png')).convert('RGB').resize(
        (W, H), Image.LANCZOS), dtype=np.float32) / 255.0

    dt = teacher - proxy
    ds = student - proxy
    lo, hi = lowpass(dt), dt - lowpass(dt)

    def amp(a):
        return np.clip(a * GAIN + 0.5, 0, 1)

    top = np.concatenate([proxy, teacher, amp(dt)], axis=1)
    bottom = np.concatenate([amp(lo), amp(hi), amp(ds)], axis=1)
    Image.fromarray(to8(np.concatenate([top, bottom], axis=0))).save(OUT)
    print(f'{OUT}  ({W * 3}x{H * 2})  放大 {GAIN:g}x  '
          f'教师改动 rms {dt.std() * 255:.2f}/255  学生 {ds.std() * 255:.2f}/255')


if __name__ == '__main__':
    main()
