#!/usr/bin/env python3
"""皮肤纹理蒸馏探针:教师的 skin/automask 改动 vs 学生的同口径改动,拆低频(光线)/高频(纹理)。

背景(2026-10-13 用户观察):学生在 skin 档只复刻了教师的光线改动,皮肤纹理修改丢失。
lane 是常数图(掩码是模型内部内容驱动),嫌疑:训练池缺皮肤 / 动作加权对小幅度高频不利。

    python3 probe_skin.py --image face.png --student weights/student_v3 \
        --teacher-weights $NR_WEIGHTS -o probe_skin

口径:对每张图跑 教师/学生 × {基线(skin=-1,am=0), 皮肤档(skin=1.0,am=1)},取各自
效果差分 d = out(skin) − out(基线),拆 d = blur(d, σ=4)(光线) + (d − blur)(纹理),
报各通道能量;并输出 2x2 对比 PNG(基线/皮肤档/差分×8)。皮肤区域占比:高频差分
能量的空间集中度(Gini 前 10% 像素占的能量比)。
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_geometry import geometry_from_valid                     # noqa: E402
from nr_model import Model                                      # noqa: E402
from nr_network import Network                                  # noqa: E402
from nr_package import is_student_package, load_student_package  # noqa: E402
from nr_student import StudentNetwork, load_student_shape       # noqa: E402
from run_image import build_features, save_png                  # noqa: E402


def blur3(x, k=9):
    """3x3 均值 k 次 ≈ 高斯低通(σ≈√(k/3)·0.57),纯 numpy,免依赖。"""
    out = x
    for _ in range(k):
        p = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode='edge')
        out = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
               p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
               p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0
    return out


def spectrum(d, tag):
    low = blur3(d)
    high = d - low
    e_low = float(np.sqrt((low ** 2).mean()))
    e_high = float(np.sqrt((high ** 2).mean()))
    mag = np.sqrt((high ** 2).sum(-1)).ravel()
    top = np.sort(mag)[::-1][: max(1, len(mag) // 10)].sum()
    gini10 = float(top / max(mag.sum(), 1e-12))
    print(f"  {tag:16s} 光线(低频) {e_low*255:6.2f}/255   纹理(高频) {e_high*255:6.2f}/255"
          f"   纹理能量 top10%像素占 {gini10*100:4.1f}%")
    return e_low, e_high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--student', required=True)
    ap.add_argument('--shape', default='../shapes/student_moe_v0.json')
    ap.add_argument('--teacher-weights', default=os.environ.get('NR_WEIGHTS'))
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--style', type=float, default=1.5)
    ap.add_argument('--tone', type=float, default=0.5)
    ap.add_argument('--structure', type=float, default=0.35)
    ap.add_argument('--direct', action='store_true',
                    help='固定旋钮口径:直接比 out−proxy 的教师/学生差(无基线差分);'
                         '单工作点学生只能用这个(基线分支对其是分布外)')
    ap.add_argument('-o', '--out', default='probe_skin')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = Image.open(args.image).convert('RGB').resize((args.width, args.height), Image.LANCZOS)
    proxy = np.asarray(img, dtype=np.float32) / 255.0
    g = geometry_from_valid(args.width, args.height)

    from annotate import TeacherAnnotator
    t = TeacherAnnotator(args.teacher_weights, g)

    if is_student_package(args.student):
        s_model, _ = load_student_package(args.student, args.width, args.height, device)
    else:
        s_model = StudentNetwork(load_student_shape(args.shape), g).to(device)
        ck = torch.load(args.student, map_location=device, weights_only=False)
        s_model.load_state_dict(ck.get('ema', ck['model']))
    s_model.eval()

    def forward(kind, skin, am):
        feats = build_features(proxy, g, args.seed, args.style, args.tone,
                               args.structure, skin, am)
        if kind == 't':
            head = t.annotate(feats)['head'].reshape(g['full_height'], g['full_width'], 4)
        else:
            with torch.no_grad():
                head = s_model(torch.from_numpy(feats).to(device))
            head = head.view(g['full_height'], g['full_width'], 4).cpu().numpy()
        return np.clip(proxy + head[:args.height, :args.width, :3] / 4, 0, 1).astype(np.float32)

    skin = {'t': forward('t', 1.0, True), 's': forward('s', 1.0, True)}
    if args.direct:
        dt, ds = skin['t'] - proxy, skin['s'] - proxy
    else:
        base = {'t': forward('t', -1.0, False), 's': forward('s', -1.0, False)}
        dt, ds = skin['t'] - base['t'], skin['s'] - base['s']

    print(f"== 教师 skin 效果拆解(/{args.image})==")
    _, e_ht = spectrum(dt, 'teacher 效果')
    print(f"== 学生 skin 效果拆解 ==")
    _, e_hs = spectrum(ds, 'student 效果')
    print(f"== 学生−教师 效果残差 ==")
    spectrum(ds - dt, 'residual')

    def hf(a):
        return a - blur3(a)

    def corr(a, b):
        a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))

    base_img = proxy if args.direct else base['t']
    hb = hf(base_img)
    print(f"纹理复刻率(高频能量 学生/教师) {100 * e_hs / max(e_ht, 1e-12):5.1f}%"
          f"   方向 corr(教师Δ,基线HF) {corr(hf(dt), hb):+.2f}"
          f"  corr(学生Δ,基线HF) {corr(hf(ds), hb):+.2f}"
          f"   corr(学生Δ,教师Δ) {corr(hf(ds), hf(dt)):+.2f}")

    # 按输入自身纹理强度分区:滑区(低30%)看"造纹理",纹理区(高30%)看"抹纹理"
    hd, hsd = hf(dt), hf(ds)
    mag = np.sqrt((hb ** 2).sum(-1)).ravel()
    md = np.sqrt((hd ** 2).sum(-1)).ravel()
    ms = np.sqrt((hsd ** 2).sum(-1)).ravel()
    lo, hi = np.quantile(mag, [0.3, 0.7])
    for name, sel in (('滑区(输入HF低30%)', mag <= lo),
                      ('中区', (mag > lo) & (mag <= hi)),
                      ('纹理区(输入HF高30%)', mag > hi)):
        bt = hb.reshape(-1, 3)[sel]
        dt_g = hd.reshape(-1, 3)[sel]
        ds_g = hsd.reshape(-1, 3)[sel]
        print(f"  {name:18s} 教师Δ纹理 {md[sel].mean() * 255:5.2f}  学生Δ纹理 {ms[sel].mean() * 255:5.2f}"
              f"  复刻 {100 * ms[sel].mean() / max(md[sel].mean(), 1e-12):5.1f}%"
              f"  方向 corr(Δ,输入HF) t {corr(dt_g, bt):+.2f} s {corr(ds_g, bt):+.2f}")

    for name, arr in (('base', base_img), ('skin_t', skin['t']), ('dt_x8', np.clip(dt * 8 + .5, 0, 1)),
                      ('skin_s', skin['s']), ('ds_x8', np.clip(ds * 8 + .5, 0, 1)),
                      ('res_x8', np.clip((ds - dt) * 8 + .5, 0, 1))):
        save_png(os.path.join(args.out, f'{name}.png'), arr)
    print(f'wrote {args.out}/')


if __name__ == '__main__':
    main()
