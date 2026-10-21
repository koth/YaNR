"""逐样本 PSNR:引擎/渲染产物 nr.png 对比 teacher.png(整图、8bit 域)。

这条口径是"C++ 引擎跑出来的图 vs 教师图",与 probes/eval_full_image.py 的
"float 域、512x320 铺块"不同,不能互相替代。
    python3 probes/psnr_png.py [samples 目录]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SAMPLES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/koth/klss/teacher/samples')


def load(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0


def main():
    rows = []
    for directory in sorted(SAMPLES.glob('t*')):
        teacher, student = directory / 'teacher.png', directory / 'nr.png'
        if not (teacher.exists() and student.exists()):
            print(f'{directory.name}: 缺文件')
            continue
        t, s = load(teacher), load(student)
        if t.shape != s.shape:
            print(f'{directory.name}: 尺寸不一致 {t.shape} vs {s.shape}')
            continue
        mse = float(np.mean((s - t) ** 2))
        psnr = 10 * np.log10(1 / max(mse, 1e-12))
        base = load(next(p for p in directory.iterdir()
                         if p.suffix.lower() in ('.png', '.jpg')
                         and p.stem not in ('teacher', 'nr')))
        base_mse = float(np.mean((base - t) ** 2))
        base_psnr = 10 * np.log10(1 / max(base_mse, 1e-12))
        rows.append((directory.name, psnr, base_psnr, t.shape))
        print(f'{directory.name:<4} {t.shape[1]}x{t.shape[0]:<5} '
              f'nr vs teacher {psnr:6.2f} dB | 输入 vs teacher {base_psnr:6.2f} | '
              f'净提升 {psnr - base_psnr:+5.2f}')
    if rows:
        arr = np.array([[r[1], r[2]] for r in rows])
        print(f'{"均值":<4} {"":<11} nr vs teacher {arr[:, 0].mean():6.2f} dB | '
              f'输入 {arr[:, 1].mean():6.2f} | 净提升 {(arr[:, 0] - arr[:, 1]).mean():+5.2f}')


if __name__ == '__main__':
    main()
