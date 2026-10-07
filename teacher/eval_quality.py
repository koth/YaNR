#!/usr/bin/env python3
"""学生 vs 教师质量评估(openspec 6.1-6.2,并为 8.8 质量验收供数)。

    python3 eval_quality.py --images '/data/val/*.png' --checkpoint runs/v1/ckpt.pt \
        --sizes 320,512 --limit 24 -o eval_v1/

每图教师/学生吃**同一份 features**(同 seed 同噪声),输出口径与 run_image/run_student
一致:neural = clamp(proxy + head_rgb/4, 0, 1)。指标:

  - psnr_teacher / psnr_student : 输出合成图 vs 输入 proxy(6.2:退化 =
    psnr_teacher - psnr_student,均值 ≤ 1.0dB 达标)
  - psnr_vs_teacher / ssim_vs_teacher : 学生输出 vs 教师输出
  - residual_corr : rgb 残差(输出 - proxy)的皮尔逊相关
  - blend_mae : blend logit 平均绝对差(6.1)

聚合输出 mean / median / min / max;结果写 eval_quality.json 与 eval_quality.md。
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nr_torch                                          # noqa: E402
from nr_geometry import geometry_from_valid              # noqa: E402
from nr_model import Model                               # noqa: E402
from nr_network import Network                           # noqa: E402
from nr_student import StudentNetwork, load_student_shape  # noqa: E402
from run_image import build_features                     # noqa: E402


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def _gaussian_kernel(size=11, sigma=1.5):
    ax = np.arange(size) - size // 2
    k = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(k, k)
    return (k / k.sum()).astype(np.float64)


def ssim(a, b, kernel=None):
    """逐通道 SSIM(11x11 高斯窗),三通道均值。a/b: [H][W][3] in 0..1。"""
    k = kernel if kernel is not None else _gaussian_kernel()
    pad = k.shape[0] // 2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for c in range(a.shape[2]):
        x = np.pad(a[..., c].astype(np.float64), pad, mode='reflect')
        y = np.pad(b[..., c].astype(np.float64), pad, mode='reflect')
        mx = _conv2(x, k)
        my = _conv2(y, k)
        mxx = _conv2(x * x, k)
        myy = _conv2(y * y, k)
        mxy = _conv2(x * y, k)
        vx = mxx - mx * mx
        vy = myy - my * my
        vxy = mxy - mx * my
        s = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
        vals.append(float(s.mean()))
    return float(np.mean(vals))


def _conv2(x, k):
    """有效卷积(调用方先 reflect pad,输出 = 输入原尺寸)。"""
    kh, kw = k.shape
    out = np.zeros((x.shape[0] - kh + 1, x.shape[1] - kw + 1), np.float64)
    for dy in range(kh):
        for dx in range(kw):
            out += k[dy, dx] * x[dy:dy + out.shape[0], dx:dx + out.shape[1]]
    return out


def composite(head_np, proxy, h, w):
    """run_image/run_student 同款:neural = clamp(proxy + rgb/4)。返回 (neural, valid)。"""
    valid = head_np[:h, :w]
    rgb = valid[..., :3] / 4.0
    neural = np.clip(proxy + rgb, 0, 1)
    return neural, valid


def collect_images(patterns):
    files = []
    for p in patterns:
        hit = sorted(glob.glob(p))
        if hit:
            files.extend(hit)
        elif os.path.isfile(p):
            files.append(p)
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def aggregate(vals):
    a = np.asarray(vals, np.float64)
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'min': float(a.min()), 'max': float(a.max())}


def main():
    parser = argparse.ArgumentParser(description='student vs teacher quality eval')
    parser.add_argument('--images', nargs='+', required=True, help='image files / globs')
    parser.add_argument('--checkpoint', help='student checkpoint (.pt); omit for random-init smoke')
    parser.add_argument('--shape', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        '..', 'shapes', 'student_v0.json'))
    parser.add_argument('--teacher-weights', default=os.environ.get('NR_WEIGHTS'))
    parser.add_argument('--sizes', default='512', help='逗号分隔边长,如 320,512')
    parser.add_argument('--limit', type=int, default=0, help='最多评估 0=全部')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('-o', '--out', default='eval_out')
    args = parser.parse_args()

    files = collect_images(args.images)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit('no images matched')
    sizes = [int(s) for s in args.sizes.split(',') if s]
    device = nr_torch.DEVICE
    os.makedirs(args.out, exist_ok=True)

    shape = load_student_shape(args.shape)
    torch.manual_seed(0)
    student = None           # 每个 size 的几何不同,逐 size 重建(权重共享拷贝)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False) \
        if args.checkpoint else None

    tmodel = Model(device).load(args.teacher_weights)
    print(f'{len(files)} images  sizes {sizes}  device {device}  '
          f'checkpoint {args.checkpoint or "(random init)"}')

    per_image = []
    t_start = time.time()
    for size in sizes:
        g = geometry_from_valid(size, size)
        torch.manual_seed(0)
        student = StudentNetwork(shape, g).to(device).eval()
        if ckpt is not None:
            student.load_state_dict(ckpt.get('ema', ckpt.get('model', ckpt)))
        network = Network(tmodel, g)
        k = _gaussian_kernel()
        for path in files:
            proxy = np.asarray(Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS),
                               dtype=np.float32) / 255.0
            feats = build_features(proxy, g, args.seed, 1.5, 0.5, 0.35, -1.0, False)
            f = torch.from_numpy(feats).to(device)
            with torch.no_grad():
                t_head = network.record(f).float().cpu().numpy().reshape(
                    g['full_height'], g['full_width'], 4)
                s_head = student(f).float().cpu().numpy().reshape(
                    g['full_height'], g['full_width'], 4)
            n_t, v_t = composite(t_head, proxy, size, size)
            n_s, v_s = composite(s_head, proxy, size, size)
            res_t = (n_t - proxy).ravel()
            res_s = (n_s - proxy).ravel()
            corr = float(np.corrcoef(res_t, res_s)[0, 1]) if res_t.std() > 0 else 1.0
            row = {
                'image': path, 'size': size,
                'psnr_teacher': psnr(n_t, proxy),
                'psnr_student': psnr(n_s, proxy),
                'degradation_db': psnr(n_t, proxy) - psnr(n_s, proxy),
                'psnr_vs_teacher': psnr(n_s, n_t),
                'ssim_vs_teacher': ssim(n_s, n_t, k),
                'residual_corr': corr,
                'blend_mae': float(np.abs(v_s[..., 3] - v_t[..., 3]).mean()),
            }
            per_image.append(row)
            print(f"  [{size}] {os.path.basename(path):<28} "
                  f"psnr {row['psnr_student']:5.2f} (t {row['psnr_teacher']:5.2f}, "
                  f"degr {row['degradation_db']:+5.2f})  vs_t {row['psnr_vs_teacher']:6.2f}dB "
                  f"ssim {row['ssim_vs_teacher']:.4f}  corr {row['residual_corr']:.4f} "
                  f"blend_mae {row['blend_mae']:.4f}")

    agg = {}
    for size in sizes:
        rows = [r for r in per_image if r['size'] == size]
        agg[str(size)] = {key: aggregate([r[key] for r in rows])
                          for key in ('psnr_teacher', 'psnr_student', 'degradation_db',
                                      'psnr_vs_teacher', 'ssim_vs_teacher',
                                      'residual_corr', 'blend_mae')}

    report = {'config': {'images': args.images, 'n': len(files), 'sizes': sizes,
                         'checkpoint': args.checkpoint, 'shape': args.shape,
                         'seed': args.seed, 'elapsed_s': round(time.time() - t_start, 1)},
              'aggregate': agg, 'per_image': per_image}
    with open(os.path.join(args.out, 'eval_quality.json'), 'w') as fh:
        json.dump(report, fh, indent=1)

    lines = ['# eval_quality(学生 vs 教师)', '',
             f"- 图片 {len(files)} 张,sizes {sizes},checkpoint `{args.checkpoint or 'random'}`,"
             f"seed {args.seed}", '',
             '| size | 指标 | mean | median | min | max |', '|---|---|---|---|---|---|']
    for size in sizes:
        for key, a in agg[str(size)].items():
            lines.append(f"| {size} | {key} | {a['mean']:.4f} | {a['median']:.4f} "
                         f"| {a['min']:.4f} | {a['max']:.4f} |")
    for size in sizes:
        d = agg[str(size)]['degradation_db']['mean']
        gate = 'PASS' if d <= 1.0 else 'FAIL'
        lines += ['', f'**6.2 门禁(512 口径,{size}²):退化 {d:+.2f}dB ≤ 1.0dB → {gate}**']
    with open(os.path.join(args.out, 'eval_quality.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

    print('\n== aggregate ==')
    for size in sizes:
        a = agg[str(size)]
        print(f"  {size}²: psnr_student {a['psnr_student']['mean']:.2f}dB "
              f"(teacher {a['psnr_teacher']['mean']:.2f}, 退化 {a['degradation_db']['mean']:+.2f}dB)  "
              f"vs_teacher {a['psnr_vs_teacher']['mean']:.2f}dB / ssim "
              f"{a['ssim_vs_teacher']['mean']:.4f}  corr {a['residual_corr']['mean']:.4f}  "
              f"blend_mae {a['blend_mae']['mean']:.4f}")
    print(f"wrote {args.out}/eval_quality.json  +  eval_quality.md")


if __name__ == '__main__':
    main()
