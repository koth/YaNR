"""固定图集上的纹理复刻度评估:同一批图片、同一档位,横向比多个 checkpoint。

为什么需要这个脚本:训练里的 vis 每 2000 步抓的是 `chunk[-1]['proxy']`,
即**每步不同的随机裁剪块**。实测 22k/30k/42k 的 proxy.png 哈希全不同,
所以跨 checkpoint 比 corr/amp 是在比不同图片 —— 样本内容差异盖过了模型进步。
本脚本固定图片、固定条件,才能回答"纹理到底学到什么程度"。

口径与训练一致:style 2 / tone 1 / structure 1 / skin -1 / automask 1,
用到 `ema` 权重(与 eval_conditions 相同),条件由 build_features 注入。

指标(扣量化噪声,与 vis_bands 同):
  fine/coarse 两个频带上的 a(回归斜率)、corr、amp(幅度比)
  方向:mean_dt / mean_ds(带符号,看是"加"还是"减")、same_sign、add_frac

用法:
  python3 eval_texture.py --images a.png b.png --teacher-weights ~/klss/models/nr \
      --size 320 512 --ckpt base20k=/path/ckpt.pt --ckpt now=/path/ckpt.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_geometry import geometry_from_valid                     # noqa: E402
from nr_model import Model                                      # noqa: E402
from nr_network import Network                                  # noqa: E402
from nr_student import StudentNetwork, load_student_shape       # noqa: E402
from run_image import build_features                            # noqa: E402

# 训练目标档位(style/tone/structure/skin/automask)
CFG = dict(style=2.0, tone=1.0, structure=1.0, skin=-1.0, automask=True)
SHAPE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'shapes', 'student_shared_moe_v1.json')


def blur3(a):
    h, w, _ = a.shape
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    s = np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            s += p[i:i + h, j:j + w]
    return s / 9.0


def band(a, iterations):
    y = a
    for _ in range(iterations):
        y = blur3(y)
    return a - y


def noise_var(shape, iterations, rng):
    e = rng.uniform(-0.5 / 255, 0.5 / 255, shape)
    return float(band(e, iterations).var())


def band_stats(dt, ds, v_noise, mask):
    x, y = dt[mask], ds[mask]
    x = x - x.mean()
    y = y - y.mean()
    cov = float((x * y).mean())
    vx, vy = float((x * x).mean()), float((y * y).mean())
    cov_t = cov - v_noise
    vx_t, vy_t = max(vx - 2 * v_noise, 0.0), max(vy - 2 * v_noise, 0.0)
    a = cov_t / max(vx_t, 1e-12)
    # 学生几乎不产纹理时(vy 被噪声校正压到 0),corr 的分母趋零会炸出 73 这种
    # 假值。低于噪声量级就如实记 nan,不参与聚合。
    if vy_t < 3 * v_noise or vx_t < 3 * v_noise:
        corr = float('nan')
    else:
        corr = cov_t / np.sqrt(vx_t * vy_t)
    amp = np.sqrt(vy_t / max(vx_t, 1e-12))
    return a, corr, amp, float(dt[mask].std() * 255)


def signed_stats(dt, ds, mask):
    """方向诊断:教师加纹理的地方,学生是否也在加。"""
    a, b = dt[mask] * 255, ds[mask] * 255
    act = np.abs(a) > 0.5                       # 超过量化噪声,教师确实动了
    same = float((np.sign(a[act]) == np.sign(b[act])).mean()) if act.any() else float('nan')
    pos = act & (a > 0)
    amp_pos = float(b[pos].std() / a[pos].std()) if pos.sum() > 30 else float('nan')
    return {'mean_dt': float(a.mean()), 'mean_ds': float(b.mean()), 'same_sign': same,
            'add_t': float((a > 0).mean()), 'add_s': float((b > 0).mean()),
            'amp_pos': amp_pos}


def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def metrics(proxy, teacher_out, student_out, rng):
    mask = np.abs(band(proxy, 2)).mean(2) >= np.percentile(np.abs(band(proxy, 2)).mean(2), 75)
    row = {}
    for iters, name in ((2, 'fine'), (9, 'coarse')):
        dt = band(teacher_out, iters) - band(proxy, iters)
        ds = band(student_out, iters) - band(proxy, iters)
        v = noise_var(proxy.shape, iters, rng)
        a, c, amp, dstd = band_stats(dt, ds, v, mask)
        row[name] = {'a': round(a, 3), 'corr': round(c, 3), 'amp': round(amp, 3),
                     'Dt': round(dstd, 2)}
        if name == 'coarse':
            row['signed'] = {k: round(v2, 3) for k, v2 in signed_stats(dt, ds, mask).items()}
    # 整体保真度:学生合成图 vs 教师合成图(标准蒸馏口径),以及各自的改动幅度。
    row['psnr'] = {'s_vs_t': round(psnr(student_out, teacher_out), 3),
                   't_vs_in': round(psnr(teacher_out, proxy), 3),
                   's_vs_in': round(psnr(student_out, proxy), 3)}
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--teacher-weights', required=True)
    parser.add_argument('--ckpt', action='append', required=True,
                        help='标签=checkpoint 路径,可重复')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('-o', '--out', default='eval_texture')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = geometry_from_valid(args.width, args.height)
    rng = np.random.default_rng(7)
    print(f'device {device}  size {args.width}x{args.height}  '
          f'style {CFG["style"]} tone {CFG["tone"]} struct {CFG["structure"]} '
          f'skin {CFG["skin"]} automask {CFG["automask"]}')

    # 先把教师的输出全部算完并留在 CPU,后面每个 ckpt 复用(教师是确定性的)。
    teacher = Network(Model(device).load(args.teacher_weights), g)
    cache = []
    for path in args.images:
        proxy = np.asarray(Image.open(path).convert('RGB').resize(
            (args.width, args.height), Image.LANCZOS), dtype=np.float32) / 255.0
        feats = torch.from_numpy(build_features(
            proxy, g, args.seed, CFG['style'], CFG['tone'], CFG['structure'],
            CFG['skin'], CFG['automask'])).to(device)
        with torch.no_grad():
            head = teacher.record(feats)
            head = head.float().cpu().numpy().reshape(g['full_height'], g['full_width'], 4)
        tout = np.clip(proxy + head[:args.height, :args.width, :3] / 4.0, 0, 1)
        cache.append({'name': os.path.basename(path), 'proxy': proxy,
                      'teacher': tout, 'feats': feats.cpu()})
        del feats, head
        torch.cuda.empty_cache()
    del teacher
    torch.cuda.empty_cache()
    print(f'teacher done: {len(cache)} 张')

    shape = load_student_shape(SHAPE)
    report = {}
    for spec in args.ckpt:
        tag, _, path = spec.partition('=')
        torch.manual_seed(0)
        student = StudentNetwork(shape, g).to(device).eval()
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        student.load_state_dict(ckpt.get('ema', ckpt.get('model')))
        step = ckpt.get('step', '?')
        per = []
        for item in cache:
            with torch.no_grad():
                head = student(item['feats'].to(device))
                head = head.float().cpu().numpy().reshape(g['full_height'],
                                                         g['full_width'], 4)
            sout = np.clip(item['proxy'] + head[:args.height, :args.width, :3] / 4.0, 0, 1)
            m = metrics(item['proxy'], item['teacher'], sout, rng)
            m['image'] = item['name']
            per.append(m)
            del head
        del student, ckpt
        torch.cuda.empty_cache()
        agg = {}
        for band_name in ('fine', 'coarse'):
            for key in ('a', 'corr', 'amp', 'Dt'):
                agg[f'{band_name}_{key}'] = round(
                    float(np.nanmean([p[band_name][key] for p in per])), 3)
        for key in ('mean_dt', 'mean_ds', 'same_sign', 'add_t', 'add_s', 'amp_pos'):
            agg[key] = round(float(np.nanmean([p['signed'][key] for p in per])), 3)
        for key in ('s_vs_t', 't_vs_in', 's_vs_in'):
            agg['psnr_' + key] = round(float(np.nanmean([p['psnr'][key] for p in per])), 2)
        # 学生几乎不出纹理的图片(fine-band nan)单独计数,别让它们悄悄从均值里消失。
        agg['fine_defined'] = sum(
            1 for p in per if not np.isnan(p['fine']['corr']))
        report[tag] = {'step': step, 'aggregate': agg, 'per_image': per}
        print(f"{tag} (step {step}): "
              f"fine a={agg['fine_a']:.2f} corr={agg['fine_corr']:.2f} amp={agg['fine_amp']:.2f} | "
              f"coarse a={agg['coarse_a']:.2f} corr={agg['coarse_corr']:.2f} "
              f"amp={agg['coarse_amp']:.2f} | Dt={agg['coarse_Dt']:.2f} | "
              f"mean_dt={agg['mean_dt']:+.2f} mean_ds={agg['mean_ds']:+.2f} "
              f"same_sign={agg['same_sign']:.2f} amp_pos={agg['amp_pos']:.2f} | "
              f"s-vs-t {agg['psnr_s_vs_t']:.2f}dB")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'eval_texture.json'), 'w') as fh:
        json.dump({'config': vars(args), 'regime': CFG, 'report': report}, fh, indent=1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
