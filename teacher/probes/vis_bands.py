"""两个频带上量纹理复刻度,确认"平台"不是某个滤波口径的假象。

fine  = x - blur3²(x)      (sigma ≈ 1.15px,结构与我的 vis_modulation 一致)
coarse= x - blur3⁹(x)      (sigma ≈ 2.45px,与训练里 hf() 同口径)
每个带上:纹理区 Ds=a·Dt 的 a、corr(去量化噪声后),以及幅度比。

用法: python3 vis_bands.py <tag>=<dir> [...]
"""
import sys

import numpy as np
from PIL import Image


def load(d, n):
    return np.asarray(Image.open(f'{d}/{n}.png').convert('RGB'), dtype=np.float64) / 255.0


def blur3(a):
    h, w, c = a.shape
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    s = np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            s += p[i:i + h, j:j + w]
    return s / 9.0


def band(a, iterations):
    y = a
    for _ in range(iterations):
        y = blur3(y)
    return a - y


def noise_var(shape, iterations, rng):
    e = rng.uniform(-0.5 / 255, 0.5 / 255, shape)
    return float(band(e, iterations).var())


def stats(dt, ds, v_noise, mask):
    x, y = dt[mask], ds[mask]
    x = x - x.mean()
    y = y - y.mean()
    cov = float((x * y).mean())
    vx, vy = float((x * x).mean()), float((y * y).mean())
    cov_t = cov - v_noise
    vx_t, vy_t = max(vx - 2 * v_noise, 1e-12), max(vy - 2 * v_noise, 1e-12)
    return cov_t / vx_t, cov_t / np.sqrt(vx_t * vy_t), y.std() / x.std()


def main():
    rng = np.random.default_rng(7)
    for arg in sys.argv[1:]:
        tag, _, d = arg.partition('=')
        proxy, teacher, student = (load(d, n) for n in ('proxy', 'teacher', 'student'))
        hpm = np.abs(band(proxy, 2)).mean(2)
        tex = hpm >= np.percentile(hpm, 75)
        out = []
        for iters, name in ((2, 'fine'), (9, 'coarse')):
            dt = band(teacher, iters) - band(proxy, iters)
            ds = band(student, iters) - band(proxy, iters)
            v = noise_var(proxy.shape, iters, rng)
            a, c, ratio = stats(dt, ds, v, tex)
            out.append(f'{name}: Dt={dt[tex].std() * 255:5.2f} a={a:5.2f} corr={c:5.2f} '
                       f'amp={ratio:5.2f}')
        print(f'{tag}: ' + '  |  '.join(out))


if __name__ == '__main__':
    main()
