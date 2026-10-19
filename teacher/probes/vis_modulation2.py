"""vis 三元组的纹理复刻度,并对 8bit PNG 量化噪声做一阶修正。

模型(在 0-1 单位、高频域):
  Dt = hp(teacher) - hp(proxy) = dt + (e_t - e_p)
  Ds = hp(student) - hp(proxy) = ds + (e_s - e_p)
e_* 为各自 PNG 量化误差(独立、均匀 ±0.5LSB)。于是
  cov(Dt,Ds) = cov(dt,ds) + v'            (proxy 的误差两边共享 -> 抬高相关)
  var(Dt) = var(dt) + 2v',  var(Ds) = var(ds) + 2v'
v' 用同尺寸均匀噪声过同一高通核数值估计。
输出: a/corr(原始)、a/corr(去噪)、学生的改动里教师可解释的方差占比 corr^2。

用法: python3 vis_modulation2.py <tag>=<dir> [...]
"""
import sys

import numpy as np
from PIL import Image


def load(d, name):
    return np.asarray(Image.open(f'{d}/{name}.png').convert('RGB'), dtype=np.float64) / 255.0


def blur3(a):
    h, w, c = a.shape
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    s = np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            s += p[i:i + h, j:j + w]
    return s / 9.0


def hp(a):
    return a - blur3(blur3(a))


def noise_var(shape, rng):
    e = rng.uniform(-0.5 / 255, 0.5 / 255, shape)
    return float(hp(e).var())


def main():
    rng = np.random.default_rng(7)
    v_noise = None
    for arg in sys.argv[1:]:
        tag, _, d = arg.partition('=')
        proxy, teacher, student = (load(d, n) for n in ('proxy', 'teacher', 'student'))
        if v_noise is None:
            v_noise = noise_var(proxy.shape, rng)
        dt, ds = hp(teacher) - hp(proxy), hp(student) - hp(proxy)
        hpm = np.abs(hp(proxy)).mean(2)
        tex = hpm >= np.percentile(hpm, 75)
        out = []
        for name, m in (('全图', None), ('纹理区', tex)):
            x = dt if m is None else dt[m]
            y = ds if m is None else ds[m]
            x = x - x.mean()
            y = y - y.mean()
            cov = float((x * y).mean())
            vx, vy = float((x * x).mean()), float((y * y).mean())
            a0 = cov / vx
            c0 = cov / np.sqrt(vx * vy)
            cov_t = cov - v_noise
            vx_t, vy_t = max(vx - 2 * v_noise, 1e-12), max(vy - 2 * v_noise, 1e-12)
            a1 = cov_t / vx_t
            c1 = cov_t / np.sqrt(vx_t * vy_t)
            out.append(f'{name} a={a0:5.2f}->{a1:5.2f} corr={c0:5.2f}->{c1:5.2f} '
                       f'(可解释方差 {c1 ** 2 * 100:4.1f}%)')
        print(f'{tag}: Dt_rms={dt.std() * 255:5.2f} Ds_rms={ds.std() * 255:5.2f}/255  '
              + '  |  '.join(out))
    print(f'(量化噪声 v\'={v_noise:.3e} -> {np.sqrt(v_noise) * 255:.3f}/255 rms)')


if __name__ == '__main__':
    main()
