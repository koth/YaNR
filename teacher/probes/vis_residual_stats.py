"""对比各 run 的可视化目录:教师/学生残差与高频能量,并核对样本是否同一张。

用法: python3 vis_residual_stats.py <tag>=<vis-dir> [...]
只读统计,不写任何东西。
"""
import hashlib
import sys

import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)


def hf(a):
    g = a.mean(2)
    return float(np.abs(np.diff(g, axis=1)).mean())


def main():
    for arg in sys.argv[1:]:
        tag, _, d = arg.partition('=')
        try:
            t, s, p = (load(f'{d}/{n}.png') for n in ('teacher', 'student', 'proxy'))
        except Exception as exc:  # noqa: BLE001
            print(f'{tag}: ERR {exc}')
            continue
        res = np.abs(t - s)
        md5 = {n: hashlib.md5(open(f'{d}/{n}.png', 'rb').read()).hexdigest()[:10]
               for n in ('teacher', 'proxy')}
        print(f'{tag}: mean|T-S|={res.mean():6.2f}/255  p95={np.percentile(res, 95):5.2f}  '
              f'max={res.max():3.0f} | hf teacher={hf(t):5.2f} student={hf(s):5.2f} '
              f'proxy={hf(p):5.2f} | md5 teacher={md5["teacher"]} proxy={md5["proxy"]}')


if __name__ == '__main__':
    main()
