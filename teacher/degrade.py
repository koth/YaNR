#!/usr/bin/env python3
"""Proxy degradation pipeline (openspec task 4.3).

把"干净"帧变成 proxy —— 游戏低质量渲染的替身。真实管线里网络看到的不是
干净图,而是 TAA/FSR 上采样后的低质量渲染:残留混叠、模糊、锐化光晕、
噪声颗粒、色带、残影。本工具用参数化退化链模拟这些特征,产出
run_image.py / annotate.py 直接可用的 proxy 帧。

退化链(逐序列抽参,逐帧抽噪声;确定性可复现):
    降采样(混叠) -> 上采样(糊化) -> 非锐化掩模(光晕) -> 高斯模糊
    -> 高斯噪声(颗粒) -> 色带量化 -> 与上一帧 proxy 混合(TAA 式残影)

用法:
    python3 degrade.py data/proc -o data/train/proxy --seed 4242
    python3 degrade.py data/raw/DIV2K_train_HR -o data/train/proxy_div2k --seed 4242
    python3 degrade.py data/proc/seq-0000-translate -o tmp/deg --seed 1 --report

源可以是:含 meta.json + f*.png 的序列目录、单图、或任意目录(递归收集图片)。
输出镜像源布局;每个样本在 <out>/degrade_index.jsonl 记一行参数(含 seed/序号)。

--report 打印 clean vs proxy 的边缘能量(Sobel)与径向 FFT 能量,用于校准参数范围。
依赖:numpy + Pillow。
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

GENERATOR = 'degrade.py v1'
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp')


# ---------------------------------------------------------------- source listing

def collect_samples(src_roots):
    """返回 [(sample_id, kind, dir_or_file, [frame_paths])],顺序稳定(排序后)。"""
    samples = []
    for root in src_roots:
        if os.path.isfile(root):
            samples.append((os.path.basename(root), 'image', root, [root]))
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            frames = sorted(f for f in filenames if f.lower().endswith(IMAGE_EXTS))
            if not frames:
                continue
            if 'meta.json' in filenames:                      # 程序化序列目录
                rel = os.path.relpath(dirpath, root)
                sid = rel if rel != '.' else os.path.basename(os.path.abspath(dirpath))
                samples.append((sid.replace(os.sep, '_'), 'sequence', dirpath,
                                [os.path.join(dirpath, f) for f in frames]))
            else:                                             # 扁平图片目录:每图一个样本
                for f in frames:
                    samples.append((os.path.splitext(f)[0], 'image', dirpath,
                                    [os.path.join(dirpath, f)]))
    return samples


# ---------------------------------------------------------------- degradation

def sample_params(rng, mode):
    """一次抽完整条退化链的参数(逐序列固定)。"""
    identity = (mode == 'none') or (mode == 'random' and rng.random() < 0.08)
    if identity:
        return {'identity': True}
    strong = mode == 'hard'
    scale = float(rng.choice([1.5, 2.0, 2.5, 3.0], p=[0.3, 0.35, 0.2, 0.15]))
    down_kernel = str(rng.choice(['box', 'bilinear', 'bicubic', 'lanczos', 'nearest'],
                                 p=[0.25, 0.25, 0.25, 0.15, 0.10]))
    up_kernel = str(rng.choice(['bilinear', 'bicubic', 'nearest'], p=[0.45, 0.35, 0.20]))
    return {
        'identity': False,
        'scale': scale,
        'down_kernel': down_kernel,
        'up_kernel': up_kernel,
        'unsharp': float(rng.uniform(0.2, 1.4 if strong else 1.0)),
        'blur_sigma': float(rng.uniform(0.2, 1.2 if strong else 0.8)),
        'noise_sigma': float(rng.uniform(0.002, 0.02 if strong else 0.014)),
        'band_bits': int(rng.choice([8, 8, 7, 6], p=[0.5, 0.2, 0.2, 0.1])),
        'ghost_alpha': float(rng.uniform(0.0, 0.18 if strong else 0.12)),
    }


PIL_KERNELS = {'box': Image.BOX, 'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC,
               'lanczos': Image.LANCZOS, 'nearest': Image.NEAREST}


def to_pil(rgb01):
    return Image.fromarray((np.clip(rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8), 'RGB')


def to_np(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def degrade_frame(frame01, prev_proxy, params, rng):
    """一帧退化:frame01/prev_proxy 为 [h, w, 3] float32,prev_proxy 可为 None。"""
    if params['identity']:
        out = frame01
    else:
        h, w = frame01.shape[:2]
        img = to_pil(frame01)
        small = (max(8, int(round(w / params['scale']))), max(8, int(round(h / params['scale']))))
        down = img.resize(small, PIL_KERNELS[params['down_kernel']])          # 混叠
        up = down.resize((w, h), PIL_KERNELS[params['up_kernel']])            # 糊化/块状
        if params['unsharp'] > 0.05:                                          # 锐化光晕
            up = up.filter(ImageFilter.UnsharpMask(radius=2, percent=int(params['unsharp'] * 100),
                                                   threshold=2))
        if params['blur_sigma'] > 0.05:                                       # 镜头/TAA 软化
            up = up.filter(ImageFilter.GaussianBlur(radius=params['blur_sigma']))
        out = to_np(up)
        out = out + rng.normal(0.0, params['noise_sigma'], size=out.shape).astype(np.float32)
        bits = params['band_bits']
        if bits < 8:                                                          # 色带
            levels = float((1 << bits) - 1)
            out = np.round(np.clip(out, 0, 1) * levels) / levels
    if prev_proxy is not None and params.get('ghost_alpha', 0.0) > 0.01:      # TAA 式残影
        a = params['ghost_alpha']
        out = (1.0 - a) * out + a * prev_proxy
    return np.clip(out, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- stats report

def sobel_energy(rgb01):
    gray = rgb01.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return float(0.5 * (gx + gy))


def radial_fft_bands(rgb01, bands=8):
    gray = rgb01.mean(axis=2)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    out = []
    for b in range(bands):
        lo, hi = b / bands, (b + 1) / bands
        mask = (r >= lo) & (r < hi)
        out.append(float(spec[mask].mean()))
    return out


def print_report(src_files, out_files, limit=16):
    se_src = se_out = 0.0
    fft_src = None
    fft_out = None
    n = 0
    for sf, of in list(zip(src_files, out_files))[:limit]:
        a = to_np(Image.open(sf).convert('RGB'))
        b = to_np(Image.open(of).convert('RGB'))
        se_src += sobel_energy(a)
        se_out += sobel_energy(b)
        fa = np.array(radial_fft_bands(a))
        fb = np.array(radial_fft_bands(b))
        fft_src = fa if fft_src is None else fft_src + fa
        fft_out = fb if fft_out is None else fft_out + fb
        n += 1
    if not n:
        print('no paired files to report')
        return
    print(f'== clean vs proxy 统计 (n={n})')
    print(f'Sobel 边缘能量: clean {se_src / n:.5f}  proxy {se_out / n:.5f}  '
          f'比值 {se_out / se_src:.3f}')
    fft_src /= n
    fft_out /= n
    print('径向 FFT 能量(低频 -> 高频, 比值 proxy/clean):')
    for b, (s, o) in enumerate(zip(fft_src, fft_out)):
        print(f'  band {b}: clean {s:.4e}  proxy {o:.4e}  比值 {o / max(s, 1e-30):.3f}')


# ---------------------------------------------------------------- driver

def main():
    parser = argparse.ArgumentParser(description='proxy degradation pipeline for distillation data')
    parser.add_argument('src', nargs='+', help='source image files / sequence dirs / image dirs')
    parser.add_argument('-o', '--out', required=True, help='output root (mirrors source layout)')
    parser.add_argument('--seed', type=int, default=4242, help='root seed (deterministic)')
    parser.add_argument('--mode', choices=['random', 'none', 'hard'], default='random',
                        help='random=混合强度(8% 恒等);none=全恒等;hard=偏强退化')
    parser.add_argument('--limit', type=int, default=0, help='只处理前 N 个样本(调试)')
    parser.add_argument('--report', action='store_true', help='打印 clean vs proxy 统计')
    args = parser.parse_args()

    samples = collect_samples(args.src)
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        raise SystemExit('no source images found')

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, 'degrade_index.jsonl')
    written = []
    for ordinal, (sid, kind, base, frames) in enumerate(samples):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, ordinal]))
        params = sample_params(rng, args.mode)
        out_dir = os.path.join(args.out, os.path.relpath(base, args.src[0]) if len(args.src) == 1 and kind == 'sequence'
                              else ('seq-' + sid if kind == 'sequence' else ''))
        os.makedirs(out_dir, exist_ok=True)
        prev = None
        out_frames = []
        for frame_path in frames:
            frame = to_np(Image.open(frame_path).convert('RGB'))
            proxy = degrade_frame(frame, prev, params, rng)
            out_path = os.path.join(out_dir, os.path.basename(frame_path))
            to_pil(proxy).save(out_path)
            out_frames.append(out_path)
            prev = proxy
        record = {'generator': GENERATOR, 'sample': sid, 'kind': kind, 'ordinal': ordinal,
                  'seed': args.seed, 'mode': args.mode, 'params': params,
                  'source_frames': [os.path.relpath(f, args.src[0] if len(args.src) == 1 else base)
                                    for f in frames]}
        with open(index_path, 'a') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')
        written.append((frames, out_frames))
        label = 'identity' if params['identity'] else f"scale={params['scale']:.1f}"
        print(f'[{ordinal + 1}/{len(samples)}] {sid}  {label}')
    if args.report:
        src_all = [f for pair in written for f in pair[0]]
        out_all = [f for pair in written for f in pair[1]]
        print_report(src_all, out_all)
    print(f'index: {index_path}')


if __name__ == '__main__':
    main()
