"""从 vis 三元组(proxy/teacher/student)里量"学生有没有学到教师对纹理的改动"。

只读分析,不需要 GPU。对每个 vis 目录:
  Dt = hp(teacher) - hp(proxy)   教师对这张图做了什么(高频域)
  Ds = hp(student) - hp(proxy)   学生做了什么
  回归 Ds = a * Dt + 残差,给出 a 与 corr;再按"输入高频高"的像素子集(纹理区)复算一遍。
另有:全图 mean|T-S|、以及高频能量 hf(1st-diff)。

用法: python3 vis_modulation.py <tag>=<dir> [...]
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


def hf(a):
    g = a.mean(2)
    return float(np.abs(np.diff(g, axis=1)).mean() * 255)


def fit(dt, ds, mask=None):
    if mask is not None:
        dt, ds = dt[mask], ds[mask]
    if dt.size < 64:
        return float('nan'), float('nan'), 0
    dt = dt - dt.mean()
    ds = ds - ds.mean()
    var = float((dt * dt).mean())
    a = float((dt * ds).mean() / var) if var > 0 else float('nan')
    sd = dt.std() * ds.std()
    corr = float((dt * ds).mean() / sd) if sd > 0 else float('nan')
    return a, corr, dt.size


def main():
    for arg in sys.argv[1:]:
        tag, _, d = arg.partition('=')
        try:
            proxy, teacher, student = (load(d, n) for n in ('proxy', 'teacher', 'student'))
        except Exception as exc:  # noqa: BLE001
            print(f'{tag}: ERR {exc}')
            continue
        dt, ds = hp(teacher) - hp(proxy), hp(student) - hp(proxy)
        res = np.abs(teacher - student) * 255
        # 纹理区:输入本身高频高的像素(前 25%)
        hpm = np.abs(hp(proxy)).mean(2)
        tex = hpm >= np.percentile(hpm, 75)
        a_all, c_all, _ = fit(dt, ds)
        a_tex, c_tex, n_tex = fit(dt, ds, tex)
        print(f'{tag}: |T-S|={res.mean():5.2f}/255  hf(px/t/s)={hf(proxy):5.2f}/{hf(teacher):5.2f}/'
              f'{hf(student):5.2f}  Dt_rms={dt.std() * 255:5.2f} Ds_rms={ds.std() * 255:5.2f}  '
              f'全图 a={a_all:5.2f} corr={c_all:5.2f}  |  纹理区 a={a_tex:5.2f} corr={c_tex:5.2f} '
              f'(n={n_tex})')


if __name__ == '__main__':
    main()
