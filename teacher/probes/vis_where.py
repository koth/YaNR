"""纹理区里,"哪儿改"对不对:改动与输入高频的相关,以及学生/教师改动的相关。

历史口径(2026-10 的结论):教师改动与输入高频负相关(把高频按输入强度压下去);
学生曾经"哪儿抹"学得像(corr 更负)但"抹成什么样"学不像(corr(Δs,Δt) 低)。
用法: python3 vis_where.py <tag>=<dir> [...]
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


def corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = x.std() * y.std()
    return float((x * y).mean() / d) if d > 0 else float('nan')


def main():
    for arg in sys.argv[1:]:
        tag, _, d = arg.partition('=')
        proxy, teacher, student = (load(d, n) for n in ('proxy', 'teacher', 'student'))
        p, dt, ds = hp(proxy), hp(teacher) - hp(proxy), hp(student) - hp(proxy)
        hpm = np.abs(p).mean(2)
        tex = hpm >= np.percentile(hpm, 75)
        print(f'{tag}: 纹理区 corr(教师改, 输入HF)={corr(dt[tex], p[tex]):5.2f}  '
              f'corr(学生改, 输入HF)={corr(ds[tex], p[tex]):5.2f}  '
              f'corr(学生改, 教师改)={corr(ds[tex], dt[tex]):5.2f}  '
              f'幅度比 |Ds|/|Dt|={ds[tex].std() / dt[tex].std():5.2f}')


if __name__ == '__main__':
    main()
