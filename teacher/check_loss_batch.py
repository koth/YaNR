"""批量损失口径钉(2026-10-14 向量化):output_losses / laplacian_l1 / sobel_l1 / hf
批量堆叠必须不改变任何样本的语义 —— 逐样本 B=1 重算与批量 (B,) 切片同值
(允许 ~1e-5 相对差:浮点求和序不同)。single / pair(repro + 逐样本 t_scale)都钉。

用法: python3 check_loss_batch.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_distill import hf, laplacian_l1, output_losses, sobel_l1  # noqa: E402

KW = dict(logit_w=1.0, act_floor=0.2, act_knee=0.01,
          tex_w=1.0, tex_floor=0.25, tex_knee=0.005)


def rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1e-12)


def main():
    torch.manual_seed(7)
    b, h, w = 8, 64, 48
    s_head = torch.randn(b, h, w, 4)
    t_head = torch.randn(b, h, w, 4)
    proxy = torch.rand(b, h, w, 3)
    repro = torch.rand(b, h, w, 3)
    t_scale = torch.rand(b)

    worst = 0.0
    for name, repro_b in (('single', None), ('pair', repro)):
        out_v, s_n, t_n, tex_v = output_losses(
            s_head, t_head, proxy, repro=repro_b, s_scale=0.8, t_scale=t_scale, **KW)
        det_v = laplacian_l1(s_n, t_n) + sobel_l1(s_n, t_n)
        hf_v = hf(s_n)
        for i in range(b):
            repro_i = repro_b[i:i + 1] if repro_b is not None else None
            out1, s1, t1, tex1 = output_losses(
                s_head[i:i + 1], t_head[i:i + 1], proxy[i:i + 1], repro=repro_i,
                s_scale=0.8, t_scale=t_scale[i:i + 1], **KW)
            det1 = laplacian_l1(s1, t1) + sobel_l1(s1, t1)
            checks = [('out', out_v[i], out1[0]), ('tex', tex_v[i], tex1[0]),
                      ('detail', det_v[i], det1[0])]
            for tag, got, want in checks:
                r = rel(got, want)
                worst = max(worst, r)
                if r > 1e-5:
                    raise SystemExit(f'FAIL {name}[{i}] {tag}: batch {float(got)} vs '
                                     f'single {float(want)} rel {r:.2e}')
            d = (hf_v[i:i + 1] - hf(s1)).abs().max()
            if float(d) > 1e-6:
                raise SystemExit(f'FAIL {name}[{i}] hf max|diff| {float(d):.2e}')
    print(f'LOSS BATCH PASS  worst rel {worst:.2e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
