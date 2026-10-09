#!/usr/bin/env python3
"""运动场约定回归(2026-10-10 双 bug 的钉子):

  1. 单位:block_match_flow 返回像素,reproject_history 吃 UV(÷w/h);
  2. 方向:motion 是**采样偏移**(prev 坐标 = 当前 + motion),不是内容位移。

平移样例自检:cur = roll(prev, d) 时,block_match_flow(prev, cur) 应 ≈ -d;
取反用(内容位移)会让重投影误差放大一个数量级以上。

    python3 check_motion.py        # PASS/FAIL,退出码 0/1
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_history import block_match_flow, reproject_history, warp_bilinear  # noqa: E402,F401

SHIFTS = [(3, 0), (-3, 0), (0, 4), (0, -4), (2, 2), (-5, 3)]


def main():
    rng = np.random.default_rng(0)
    prev = rng.random((64, 64, 3)).astype(np.float32)
    ok = True
    for dx, dy in SHIFTS:
        cur = np.roll(np.roll(prev, dy, 0), dx, 1)             # 内容位移 (dx, dy)
        motion = block_match_flow(prev, cur)
        want = np.array([-dx, -dy], np.float32)                 # 采样偏移 = -d
        got = motion[16:48, 16:48].reshape(-1, 2).mean(0)       # 中心区(避开回绕边)
        hit = bool(np.all(np.abs(got - want) <= 1.0))
        ok &= hit
        print(f'  shift d=({dx:+d},{dy:+d})  motion={got}  want={want}  '
              f'{"ok" if hit else "MISMATCH"}')

        # 重投影闭环(部署口径:上帧 -> 当前坐标):reproject(prev) 应贴上 cur
        uv = motion / np.array([64.0, 64.0], np.float32)
        aligned = reproject_history(prev, uv)
        err_fix = float(np.abs(aligned[16:48, 16:48] - cur[16:48, 16:48]).mean())
        err_inv = float(np.abs(reproject_history(prev, -uv)[16:48, 16:48]
                               - cur[16:48, 16:48]).mean())
        hit2 = err_fix < 0.02 and err_fix * 3 < err_inv
        ok &= hit2
        print(f'    reproject err 正向 {err_fix:.4f}  取反 {err_inv:.4f}  '
              f'{"ok" if hit2 else "FAIL"}')

        # warp_bilinear 同一约定(像素单位,上帧 -> 当前坐标)
        warped = warp_bilinear(prev, motion)
        err_w = float(np.abs(warped[16:48, 16:48] - cur[16:48, 16:48]).mean())
        hit3 = err_w < 0.02
        ok &= hit3
        print(f'    warp err {err_w:.4f}  {"ok" if hit3 else "FAIL"}')
    print('CHECK MOTION PASS' if ok else 'CHECK MOTION FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
