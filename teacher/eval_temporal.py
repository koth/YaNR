#!/usr/bin/env python3
"""时序稳定性评估(openspec 6.3):帧间闪烁 + 块匹配光流 warp error。

    python3 eval_temporal.py --seq 'data/DAVIS/.../bear/*.jpg' --checkpoint v1.pt \
        --size 320 --max-frames 8 -o eval_temporal/

协议(与真实帧管线同口径):
  * 逐帧推进,lanes 7-9 吃上帧输出(store_history 向零截断),t=0 无历史;
  * 运动用**块匹配光流近似**(8px 块,±8 搜索,SAD),一鱼两吃:历史重投影 +
    warp error 度量;
  * temporal blend 按 run_image/run_student 语义应用。

指标(逐序列 + 聚合,学生/教师成对):
  - flicker = std((out_t - out_{t-1}) - (in_t - in_{t-1}))   帧间闪烁
  - warp_err = mean|warp(out_{t-1}, flow) - out_t|           重投影一致性
门禁(6.3):学生 ≤ 1.5× 教师。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nr_torch                                          # noqa: E402
from nr_geometry import geometry_from_valid              # noqa: E402
from nr_history import reproject_history, store_history  # noqa: E402
from nr_model import Model                               # noqa: E402
from nr_network import Network                           # noqa: E402
from nr_student import StudentNetwork, load_student_shape  # noqa: E402
from run_image import build_features                     # noqa: E402


def block_match_flow(prev, cur, block=8, search=8):
    """块匹配光流:cur 的每个 block 在 prev 里搜 SAD 最小偏移(uv,y 向下,像素单位)。

    返回 motion [h][w][2]。暴力但向量化(逐搜索偏移整图算 SAD)。
    """
    h, w = cur.shape[:2]
    nb_y, nb_x = h // block, w // block
    best = np.full((nb_y, nb_x), np.inf, np.float64)
    mv = np.zeros((nb_y, nb_x, 2), np.float32)
    gray_p = prev.mean(2) if prev.ndim == 3 else prev
    gray_c = cur.mean(2) if cur.ndim == 3 else cur
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            ps = np.roll(np.roll(gray_p, dy, 0), dx, 1)
            sad = np.abs(gray_c - ps)
            # 按块聚合 SAD
            b = sad[:nb_y * block, :nb_x * block].reshape(nb_y, block, nb_x, block).sum((1, 3))
            better = b < best
            best[better] = b[better]
            mv[..., 0][better] = dx
            mv[..., 1][better] = dy
    motion = np.repeat(np.repeat(mv, block, 0), block, 1)
    return motion[:h, :w]


def warp_bilinear(img, motion):
    """按 motion [h][w][2](prev 坐标 = 当前 + motion)双线性采样 prev。"""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = xx + motion[..., 0]
    sy = yy + motion[..., 1]
    sx = np.clip(sx, 0, w - 1.001)
    sy = np.clip(sy, 0, h - 1.001)
    x0 = sx.astype(np.int32)
    y0 = sy.astype(np.int32)
    fx = (sx - x0)[..., None]
    fy = (sy - y0)[..., None]
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def composite(head_np, proxy, h, w, repro, blend_scale):
    """run_image/run_student 同口径(含 temporal blend)。"""
    valid = head_np[:h, :w]
    rgb = valid[..., :3] / 4.0
    neural = np.clip(proxy + rgb, 0, 1)
    sigmoid = 1.0 / (1.0 + np.exp(-valid[..., 3].astype(np.float64)))
    blend = sigmoid
    if repro is not None:
        blend = np.clip(sigmoid * float(blend_scale), 0, 1)
        neural = neural + (repro.astype(np.float64) - neural) * blend[..., None]
    return neural.astype(np.float32)


def run_sequence(paths, size, seed, network, student, tmodel, g, shape, vary_seed=False,
                 hist_src='self'):
    """教师/学生各是一条独立反馈环:features 按各自的上帧历史分别构建。
    hist_src='teacher'(诊断口径):学生的历史吃教师上帧输出,开环隔离反馈误差。"""
    outs_t, outs_s, proxies = [], [], []
    hist = {'t': None, 's': None}
    prev_proxy = None
    for ti, path in enumerate(paths):
        fs = seed + ti if vary_seed else seed
        proxy = np.asarray(Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS),
                           dtype=np.float32) / 255.0
        motion = block_match_flow(prev_proxy, proxy) if prev_proxy is not None else None
        result = {}
        for tag in ('t', 's'):
            htag = 't' if (tag == 's' and hist_src == 'teacher') else tag
            repro = (reproject_history(hist[htag], motion)
                     if hist[htag] is not None and motion is not None else None)
            feats = build_features(proxy, g, fs, 1.5, 0.5, 0.35, -1.0, False, history=repro)
            f = torch.from_numpy(feats).to(nr_torch.DEVICE)
            with torch.no_grad():
                head = (network.record(f) if tag == 't' else student(f))
            head_np = head.float().cpu().numpy().reshape(g['full_height'], g['full_width'], 4)
            scale = float(tmodel.blend_scale()) if tag == 't' \
                else float(student.blend_scale.detach().cpu())
            out = composite(head_np, proxy, size, size, repro, scale)
            hist[tag] = store_history(out)
            result[tag] = out
        outs_t.append(result['t'])
        outs_s.append(result['s'])
        proxies.append(proxy)
        prev_proxy = proxy
    return np.stack(proxies), np.stack(outs_t), np.stack(outs_s)


def metrics(proxies, outs, flow_fn):
    """闪烁与 warp error(该序列)。"""
    din = proxies[1:] - proxies[:-1]
    dout = outs[1:] - outs[:-1]
    flicker = float(np.std((dout - din).astype(np.float64)))
    warp_errs = []
    for t in range(1, len(outs)):
        flow = flow_fn(proxies[t - 1], proxies[t])
        warped = warp_bilinear(outs[t - 1], flow)
        warp_errs.append(float(np.mean(np.abs(warped - outs[t]))))
    return flicker, float(np.mean(warp_errs))


def main():
    parser = argparse.ArgumentParser(description='temporal stability eval (6.3)')
    parser.add_argument('--seq', nargs='+', required=True, help='序列帧 globs(每 glob 一序列)')
    parser.add_argument('--checkpoint')
    parser.add_argument('--shape', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        '..', 'shapes', 'student_v0.json'))
    parser.add_argument('--teacher-weights', default=os.environ.get('NR_WEIGHTS'))
    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('--max-frames', type=int, default=8)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--vary-seed', action='store_true',
                        help='逐帧变换噪声 seed(真实部署口径;默认固定 seed 便于复现)')
    parser.add_argument('--hist-src', choices=('self', 'teacher'), default='self',
                        help="学生的上帧历史来源:self=闭环(部署口径) teacher=开环(诊断)")
    parser.add_argument('-o', '--out', default='eval_temporal')
    args = parser.parse_args()

    device = nr_torch.DEVICE
    size = args.size
    g = geometry_from_valid(size, size)
    shape = load_student_shape(args.shape)
    torch.manual_seed(0)
    student = StudentNetwork(shape, g).to(device).eval()
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        student.load_state_dict(ckpt.get('ema', ckpt.get('model', ckpt)))
    tmodel = Model(device).load(args.teacher_weights)
    network = Network(tmodel, g)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for pat in args.seq:
        paths = sorted(glob.glob(pat))[:args.max_frames]
        if len(paths) < 2:
            print(f'skip(帧不足):{pat}')
            continue
        proxies, outs_t, outs_s = run_sequence(paths, size, args.seed, network, student,
                                               tmodel, g, shape, args.vary_seed,
                                               args.hist_src)
        fl_t, we_t = metrics(proxies, outs_t, block_match_flow)
        fl_s, we_s = metrics(proxies, outs_s, block_match_flow)
        name = os.path.basename(os.path.dirname(paths[0])) or pat
        row = {'sequence': name, 'frames': len(paths),
               'flicker_teacher': fl_t, 'flicker_student': fl_s,
               'flicker_ratio': fl_s / fl_t if fl_t > 0 else 0.0,
               'warp_teacher': we_t, 'warp_student': we_s,
               'warp_ratio': we_s / we_t if we_t > 0 else 0.0}
        rows.append(row)
        print(f"  {name:<24} flicker {fl_s:.5f} (t {fl_t:.5f}, x{row['flicker_ratio']:.2f})  "
              f"warp {we_s:.5f} (t {we_t:.5f}, x{row['warp_ratio']:.2f})")

    if not rows:
        raise SystemExit('no sequences evaluated')
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ('flicker_teacher', 'flicker_student', 'flicker_ratio',
                     'warp_teacher', 'warp_student', 'warp_ratio')}
    report = {'config': {'size': size, 'checkpoint': args.checkpoint,
                         'max_frames': args.max_frames, 'seed': args.seed,
                         'hist_src': args.hist_src, 'vary_seed': args.vary_seed},
              'aggregate': agg, 'per_sequence': rows}
    with open(os.path.join(args.out, 'eval_temporal.json'), 'w') as fh:
        json.dump(report, fh, indent=1)
    gate = 'PASS' if agg['flicker_ratio'] <= 1.5 and agg['warp_ratio'] <= 1.5 else 'FAIL'
    lines = ['# eval_temporal(6.3 时序稳定性)', '',
             '| 序列 | 帧 | flicker 学生 | flicker 教师 | × | warp 学生 | warp 教师 | × |',
             '|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['sequence']} | {r['frames']} | {r['flicker_student']:.5f} "
                     f"| {r['flicker_teacher']:.5f} | {r['flicker_ratio']:.2f} "
                     f"| {r['warp_student']:.5f} | {r['warp_teacher']:.5f} "
                     f"| {r['warp_ratio']:.2f} |")
    lines += ['', f"**6.3 门禁:闪烁 ×{agg['flicker_ratio']:.2f}、warp ×{agg['warp_ratio']:.2f}"
              f" 均 ≤ 1.5 → {gate}**"]
    with open(os.path.join(args.out, 'eval_temporal.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"== aggregate: flicker x{agg['flicker_ratio']:.2f}  warp x{agg['warp_ratio']:.2f}"
          f"  -> {gate}")
    print(f"wrote {args.out}/eval_temporal.json  +  eval_temporal.md")


if __name__ == '__main__':
    main()
