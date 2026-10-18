"""分析 samples/t*/ 里 原图 / teacher.png / nr.png 三者的纹理关系。

只用读盘,不加载任何模型,CPU 几秒跑完。

render_samples.py 里那个 corr 是裸 np.corrcoef,有两个坑:
  1. 没扣 8bit 量化噪声 —— 教师改动小的时候相关系数会被噪声主导,甚至为负;
  2. 没打印教师的改动幅度 Dt —— 看不出这一点。
这里按 eval_texture 的口径重算:量化噪声用"均匀噪声过同一滤波器"标定,
再对协方差与方差做扣除。Dt 一起打出来,分不清的样本就不下结论。

    python3 analyze_samples.py <samples 目录>
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


def band(a, it):
    y = a
    for _ in range(it):
        y = blur3(y)
    return a - y


def noise_var(shape, it, rng):
    e = rng.uniform(-0.5 / 255, 0.5 / 255, shape)
    return float(band(e, it).var())


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '/home/koth/klss/teacher/samples'
    rng = np.random.default_rng(7)
    print(f'{"目录":<6}{"尺寸":>11}{"Dt":>7}{"a":>7}{"corr":>7}{"amp":>7}'
          f'{"m_dt":>7}{"m_ds":>7}{"sgn":>6}')
    for d in sorted(x for x in glob.glob(os.path.join(root, 't*')) if os.path.isdir(x)):
        src = [f for f in glob.glob(os.path.join(d, '*'))
               if os.path.basename(f) not in ('nr.png', 'teacher.png')]
        if not src:
            continue
        proxy = load(src[0])
        teacher = load(os.path.join(d, 'teacher.png'))
        student = load(os.path.join(d, 'nr.png'))
        h, w = proxy.shape[:2]
        hp = band(proxy, 2)
        tex = np.abs(hp).mean(2) >= np.percentile(np.abs(hp).mean(2), 75)
        dt = band(teacher, 9) - band(proxy, 9)
        ds = band(student, 9) - band(proxy, 9)
        v = noise_var(proxy.shape, 9, rng)
        x, y = dt[tex], ds[tex]
        xm, ym = x - x.mean(), y - y.mean()
        cov = float((xm * ym).mean()) - v
        vx = max(float((xm * xm).mean()) - 2 * v, 0.0)
        vy = max(float((ym * ym).mean()) - 2 * v, 0.0)
        a = cov / max(vx, 1e-12)
        corr = float('nan') if vy < 3 * v else cov / np.sqrt(vx * vy)
        amp = np.sqrt(vy / max(vx, 1e-12))
        aa, bb = x * 255, y * 255
        act = np.abs(aa) > 0.5
        sgn = float((np.sign(aa[act]) == np.sign(bb[act])).mean()) if act.any() else float('nan')
        print(f'{os.path.basename(d):<6}{w}x{h:<7}{x.std() * 255:7.2f}{a:7.2f}{corr:7.2f}'
              f'{amp:7.2f}{aa.mean():+7.2f}{bb.mean():+7.2f}{sgn:6.2f}', flush=True)


if __name__ == '__main__':
    main()
