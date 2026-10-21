"""实验 A:教师输出对 batch 分组是否确定。

教师走 fp8 GEMM,GEMM 的分块随 batch 变化。如果同一张图在不同 batch 组合下
输出有差异,那么这条差异就是学生永远学不到(也无需学)的噪声,构成 PSNR 天花板。

只测 teacher,不加载学生。
    CUDA_VISIBLE_DEVICES=0 python3 probes/teacher_determinism.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/koth/klss')
sys.path.insert(0, str(ROOT / 'teacher'))

from annotate import TeacherAnnotator              # noqa: E402
from nr_geometry import geometry_from_valid        # noqa: E402
from run_image import build_features               # noqa: E402
from single_frame_data import read_rgb, RESAMPLE   # noqa: E402

W, H = 512, 320
IMAGES = ['teacher/samples/t1/t1.png', 'teacher/samples/t3/t3.png',
          'teacher/samples/t7/t7.jpg']


def features(path, seed=91):
    image = read_rgb(ROOT / path)
    proxy = np.asarray(image.resize((W, H), RESAMPLE), dtype=np.float32) / 255
    return build_features(proxy, geometry_from_valid(W, H), seed,
                          2., 1., 1., -1., True)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12)), mse


def main():
    g = geometry_from_valid(W, H)
    ann = TeacherAnnotator(str(ROOT / 'models/nr'), g)
    print('teacher loaded', flush=True)
    stacks = [features(p) for p in IMAGES]

    def run(batch_inputs):
        ann.g = ann.network.g = g
        blob = np.stack(batch_inputs).reshape(-1, 16)
        with torch.inference_mode():
            res = ann.annotate(blob, capture=(), batch=len(batch_inputs))
        return res['head'].reshape(len(batch_inputs), H, W, 4).copy()

    report = {}
    # 同一张图、batch=1,连跑两次 -> 纯随机性
    one = run([stacks[0]])
    one_again = run([stacks[0]])
    p, m = psnr(one[0][:, :, :3], one_again[0][:, :, :3])
    report['same_batch1_twice'] = dict(psnr=p, mse=m)
    print(f'batch1 重复两次        : {p:.2f} dB  (mse {m:.3e})', flush=True)

    # 同一张图放在 batch=2 / batch=3 里 -> GEMM 分块差异
    for size in (2, 3):
        block = run([stacks[0]] + stacks[1:size])
        p, m = psnr(one[0][:, :, :3], block[0][:, :, :3])
        report[f'batch1_vs_batch{size}'] = dict(psnr=p, mse=m)
        print(f'batch1 vs batch{size}        : {p:.2f} dB  (mse {m:.3e})', flush=True)

    # 顺序变化(把图换到 batch 的第 2 位)
    swapped = run([stacks[1], stacks[0]])
    p, m = psnr(one[0][:, :, :3], swapped[1][:, :, :3])
    report['position_swapped'] = dict(psnr=p, mse=m)
    print(f'batch 内位置从 0 换到 1 : {p:.2f} dB  (mse {m:.3e})', flush=True)

    Path('/tmp/teacher_determinism.json').write_text(json.dumps(report, indent=1))
    print('wrote /tmp/teacher_determinism.json')


if __name__ == '__main__':
    main()
