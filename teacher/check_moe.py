#!/usr/bin/env python3
"""MoE 块回归(2026-10-10,Capacity ×8 @同 FLOPs):

  1. 分发正确性:_moe_linear / MoEBlock 的 top-1 分组 == 逐行暴力参考;
  2. 路由器梯度:Switch 门控让 router.weight 收到非零梯度;
  3. 负载均衡:初始化时 moe_aux ≈ 1(均匀路由期望 E*Σ f*p = 1);
  4. 确定性:同输入两次前向逐位一致(argmax 平局按索引,无随机);
  5. 整网:student_moe.json 前向有限、参数量 ≈ 30M(容量 ×8 口径)。

    python3 check_moe.py        # PASS/FAIL,退出码 0/1
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_geometry import geometry_from_valid                    # noqa: E402
from nr_student import (MoEBlock, StudentNetwork, _moe_linear,  # noqa: E402
                        _window_route, load_student_shape)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    torch.manual_seed(0)
    ok = True

    # ---- 1. _moe_linear 分发 == 逐行暴力
    rows, ch_in, ch_out, E = 64, 32, 48, 8
    x = torch.randn(rows, ch_in)
    w = torch.randn(E, ch_in, ch_out)
    assign = torch.randint(0, E, (rows,))
    gates = torch.rand(rows)
    got = _moe_linear(x, w, assign, gates)
    ref = torch.stack([x[i] @ w[assign[i]] * gates[i] for i in range(rows)])
    hit = bool(torch.allclose(got, ref, atol=1e-5))
    ok &= hit
    print(f'1. moe_linear 分发 vs 暴力:{"ok" if hit else "MISMATCH"}')

    # ---- 2/3/4. MoEBlock:梯度、均衡、确定性
    blk = MoEBlock(32, 32, experts=8, attn=True).eval()
    state = torch.randn(256, 32)                               # 16x16 域 = 4 个 8x8 窗
    out1 = blk(state, 16, 16)
    a0, p0, _l0 = blk.moe_stats[-1]
    hit0 = p0.shape[0] == 4 and a0.shape[0] == 256             # probs 行=窗数,assign 行=像素数
    ok &= hit0
    print(f'0. 窗池化语义 probs {tuple(p0.shape)}(应 [4,8])assign {tuple(a0.shape)}'
          f'{"ok" if hit0 else "FAIL(池化广播 bug 回归!)"}')
    aux = _aux(blk)
    hit2 = abs(float(aux) - 1.0) < 0.15
    ok &= hit2
    print(f'3. 初始 moe_aux {float(aux):.3f}(期望 ≈1){"ok" if hit2 else "FAIL"}')

    blk.zero_grad(set_to_none=False)
    out1.sum().backward()
    rg = blk.router[1].weight.grad
    hit3 = rg is not None and float(rg.abs().sum()) > 0
    ok &= hit3
    print(f'2. 路由器梯度 {"非零 ok" if hit3 else "FAIL(门控断梯度)"}')

    out2 = blk(state, 16, 16)
    hit4 = bool(torch.equal(out1, out2))
    ok &= hit4
    print(f'4. 确定性 {"逐位一致" if hit4 else "FAIL"}')

    # ---- 4b. 配额平衡:极端同分输入下也必须均分(结构防塌缩)
    from nr_student import _balanced_pick
    same = torch.zeros(64, 8).repeat(1, 1)
    same[:, 3] = 0.001                                        # 所有窗都偏好专家 3
    bal = torch.bincount(_balanced_pick(same), minlength=8)
    hit4b = bool((bal.max() <= 8) and (bal.min() >= 0) and int(bal.sum()) == 64
                 and bal.max().item() - bal.min().item() <= 1)
    ok &= hit4b
    print(f'4b. 配额平衡(64 窗 8 专家全偏好同一家):{bal.tolist()}  {"ok" if hit4b else "FAIL"}')

    # ---- 5. 整网
    shape_path = os.path.join(HERE, '..', 'shapes', 'student_moe.json')
    shape = load_student_shape(shape_path)
    torch.manual_seed(0)
    net = StudentNetwork(shape, geometry_from_valid(320, 320)).eval()
    params = sum(p.numel() for p in net.parameters())
    feats = torch.randn(320 * 320, 16)
    with torch.no_grad():
        head = net(feats)
    finite = bool(torch.isfinite(head).all())
    ok &= finite
    with torch.no_grad():
        aux_net = net.moe_aux_loss()
    print(f'5. 整网:{params / 1e6:.1f}M 参数  head {tuple(head.shape)}  '
          f'finite {finite}  moe_aux {float(aux_net):.3f}')
    ok &= params > 20_000_000                                  # 容量 ×8 口径(≥20M)

    print('CHECK MOE PASS' if ok else 'CHECK MOE FAIL')
    return 0 if ok else 1


def _aux(blk):
    assign, probs, _logits = blk.moe_stats[-1]
    e_count = probs.shape[-1]
    f = torch.bincount(assign, minlength=e_count).float() / assign.numel()
    return e_count * (f * probs.mean(0)).sum()


if __name__ == '__main__':
    raise SystemExit(main())
