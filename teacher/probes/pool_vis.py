"""把多个 vis 目录当一个样本集汇总:版本间比较靠"多张随机图的均值",不靠单张。

vis 每次是随机裁剪块,单张 corr/amp 的样本噪声很大(实测同一模型 0.35-0.84)。
但同一个版本的多张图放在一起求均值+标准误,可以压住噪声,给出可比的中枢值。
图片来自同一数据集、同一采样器,所以组间均值是可比的。

用法: python3 pool_vis.py <tag>=<dir1>,<dir2>,... [<tag2>=...]
"""
import sys

import numpy as np
from PIL import Image


def load(d, n):
    return np.asarray(Image.open(f'{d}/{n}.png').convert('RGB'), dtype=np.float64) / 255.0


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


def one(d, rng):
    proxy, teacher, student = (load(d, n) for n in ('proxy', 'teacher', 'student'))
    hpm = np.abs(band(proxy, 2)).mean(2)
    tex = hpm >= np.percentile(hpm, 75)
    out = {}
    for it, name in ((2, 'fine'), (9, 'coarse')):
        dt = band(teacher, it) - band(proxy, it)
        ds = band(student, it) - band(proxy, it)
        v = noise_var(proxy.shape, it, rng)
        x, y = dt[tex], ds[tex]
        xm, ym = x - x.mean(), y - y.mean()
        cov = float((xm * ym).mean()) - v
        vx = max(float((xm * xm).mean()) - 2 * v, 0.0)
        vy = max(float((ym * ym).mean()) - 2 * v, 0.0)
        out[name] = {'a': cov / max(vx, 1e-12),
                     'corr': float('nan') if vy < 3 * v else cov / np.sqrt(vx * vy),
                     'amp': np.sqrt(vy / max(vx, 1e-12)),
                     'Dt': float(x.std() * 255)}
        if name == 'coarse':
            a, b = x * 255, y * 255
            act = np.abs(a) > 0.5
            out['same'] = float((np.sign(a[act]) == np.sign(b[act])).mean()) if act.any() else float('nan')
            out['dt'] = float(a.mean())
            out['ds'] = float(b.mean())
    return out


def main():
    rng = np.random.default_rng(7)
    for arg in sys.argv[1:]:
        tag, _, dirs = arg.partition('=')
        rows = [one(d, rng) for d in dirs.split(',') if d]
        def agg(key, band_name=None):
            values = np.array([r[band_name][key] if band_name else r[key] for r in rows])
            values = values[~np.isnan(values)]
            n = len(values)
            se = values.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
            return values.mean(), se, n
        f = [agg(k, 'fine') for k in ('corr', 'amp')]
        c = [agg(k, 'coarse') for k in ('corr', 'amp')]
        same, dt, ds = agg('same'), agg('dt'), agg('ds')
        print(f'{tag:<6} n={len(rows):>2} | fine corr {f[0][0]:.2f}±{f[0][1]:.2f} '
              f'amp {f[1][0]:.2f}±{f[1][1]:.2f} | coarse corr {c[0][0]:.2f}±{c[0][1]:.2f} '
              f'amp {c[1][0]:.2f}±{c[1][1]:.2f} | same_sign {same[0]:.2f} | '
              f'mean_dt {dt[0]:+.2f} mean_ds {ds[0]:+.2f}')


if __name__ == '__main__':
    main()
