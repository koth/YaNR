#!/usr/bin/env python3
"""Procedural sequence generator for the distillation dataset (openspec task 4.2).

产出"干净"RGB 帧序列(PNG),后续由 degrade.py 做 proxy 退化、annotate.py 跑教师标注。
内容刻意偏游戏渲染性格:硬边缘、细线、锯齿源(checker/条纹)、HUD 式矩形、粒子,
以及平移/缩放/旋转/流场四类运动 —— 网络的细节生成对这些高频结构最敏感。

确定性:同 --seed 逐位可复现。序列 i 的 rng = numpy SeedSequence([seed, i]),
与生成顺序、机器无关。

用法:
    python3 gen_sequences.py -o data/proc --count 200 --frames 8 --size 1920x1080 --seed 20261007
    python3 gen_sequences.py -o tmp/proc_test --count 3 --frames 2 --size 512x288 --seed 1 \\
        --types checker,particles

输出布局:
    <out>/seq-0000-checker/f000.png ... fNNN.png
    <out>/seq-0000-checker/meta.json   # 场景、参数、seed,可复现

依赖:numpy + Pillow。
"""
import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

SCENES = ['translate', 'zoom', 'rotate', 'particles', 'checker', 'flow']
GENERATOR = 'gen_sequences.py v1'


# ---------------------------------------------------------------- noise fields

def value_noise(h, w, cells_y, cells_x, rng):
    """低分辨率随机格点双三次上采样成 [0,1] 平滑噪声。"""
    grid = rng.random((max(2, cells_y) + 1, max(2, cells_x) + 1)).astype(np.float32)
    img = Image.fromarray((grid * 255.0 + 0.5).astype(np.uint8), 'L')
    return np.asarray(img.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0


def fbm(h, w, rng, octaves=5, base=4):
    """倍频程叠加的分形噪声,主能量在低频、细节在高频。"""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    cy = cx = base
    for _ in range(octaves):
        out += amp * value_noise(h, w, cy, cx, rng)
        total += amp
        amp *= 0.5
        cy *= 2
        cx *= 2
    return out / total


def checker_layer(h, w, cell_x, cell_y, phase_x, phase_y, xx, yy):
    """解析棋盘(亚像素相位 → 帧间漂移,故意制造混叠压力)。"""
    return ((np.floor((xx + phase_x) / cell_x) + np.floor((yy + phase_y) / cell_y)) % 2).astype(np.float32)


def stripe_layer(h, w, angle, period, phase, xx, yy):
    proj = xx * math.cos(angle) + yy * math.sin(angle)
    return (np.floor((proj + phase) / period) % 2).astype(np.float32)


# ---------------------------------------------------------------- texture build

def make_texture(h, w, rng):
    """一张带硬边内容的 RGB 纹理:fbm 底 + 条纹/棋盘补丁 + 细线 + HUD 矩形。"""
    f1 = fbm(h, w, rng, octaves=6, base=5)
    f2 = fbm(h, w, rng, octaves=6, base=5)
    f3 = fbm(h, w, rng, octaves=6, base=5)
    tint = rng.random(3).astype(np.float32)
    rgb = (np.stack([f1, f2, f3], axis=-1) * (0.45 + 0.55 * tint) + 0.18 * f1[..., None])

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(int(rng.integers(2, 5))):
        y0 = int(rng.integers(0, h - 8)); y1 = int(rng.integers(y0 + 8, h))
        x0 = int(rng.integers(0, w - 8)); x1 = int(rng.integers(x0 + 8, w))
        alpha = float(rng.uniform(0.25, 0.8))
        if rng.random() < 0.5:
            layer = checker_layer(h, w, int(rng.integers(2, 9)), int(rng.integers(2, 9)),
                                  rng.uniform(0, 8), rng.uniform(0, 8), xx, yy)
        else:
            layer = stripe_layer(h, w, rng.uniform(0, math.pi), float(rng.uniform(2.0, 9.0)),
                                 rng.uniform(0, 16), xx, yy)
        mask = np.zeros((h, w), np.float32)
        mask[y0:y1, x0:x1] = alpha
        rgb = rgb * (1.0 - mask[..., None]) + layer[..., None] * mask[..., None]

    img = Image.fromarray((np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8), 'RGB')
    draw = ImageDraw.Draw(img, 'RGBA')
    for _ in range(int(rng.integers(6, 14))):                      # 实心几何体("物体"硬边界)
        cx0, cy0 = rng.uniform(0, w), rng.uniform(0, h)
        s = float(rng.uniform(0.02, 0.16) * max(h, w))
        fill = tuple(int(v) for v in rng.integers(20, 235, size=3)) + (255,)
        border = tuple(int(v) for v in rng.integers(0, 255, size=3)) + (255,)
        kind = int(rng.integers(0, 3))
        if kind == 0:
            draw.ellipse([cx0 - s, cy0 - s * 0.7, cx0 + s, cy0 + s * 0.7],
                         fill=fill, outline=border, width=int(rng.integers(1, 3)))
        elif kind == 1:
            draw.rectangle([cx0 - s, cy0 - s * 0.6, cx0 + s, cy0 + s * 0.6],
                           fill=fill, outline=border, width=int(rng.integers(1, 3)))
        else:
            n = int(rng.integers(3, 7))
            pts = [(cx0 + s * math.cos(2 * math.pi * k / n + rng.uniform(0, 1)),
                    cy0 + s * math.sin(2 * math.pi * k / n + rng.uniform(0, 1))) for k in range(n)]
            draw.polygon(pts, fill=fill, outline=border)
    for _ in range(int(rng.integers(4, 12))):                      # 细线(亚像素边缘源)
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(0.1, 0.6) * max(h, w)
        width = int(rng.integers(1, 4))
        color = tuple(int(v) for v in rng.integers(20, 240, size=3)) + (255,)
        draw.line([x0, y0, x0 + length * math.cos(ang), y0 + length * math.sin(ang)],
                  fill=color, width=width)
    for _ in range(int(rng.integers(2, 6))):                       # HUD 式面板
        x0 = int(rng.integers(0, w - 40)); y0 = int(rng.integers(0, h - 30))
        x1 = int(rng.integers(x0 + 40, min(w, x0 + w // 2)))
        y1 = int(rng.integers(y0 + 30, min(h, y0 + h // 3)))
        border = tuple(int(v) for v in rng.integers(60, 255, size=3)) + (255,)
        fill = tuple(int(v) for v in rng.integers(0, 90, size=3)) + (int(rng.integers(40, 140)),)
        draw.rectangle([x0, y0, x1, y1], outline=border, width=int(rng.integers(1, 3)), fill=fill)
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------- sampling utils

def shift_subpixel(img, dx, dy):
    """整数 roll + 双线性分数混合(wrap 边界,合成内容可接受)。"""
    ix, iy = int(math.floor(dx)), int(math.floor(dy))
    fx, fy = dx - ix, dy - iy

    def roll(x, y):
        return np.roll(np.roll(img, y, axis=0), x, axis=1)

    return (roll(ix, iy) * (1 - fx) * (1 - fy) + roll(ix + 1, iy) * fx * (1 - fy) +
            roll(ix, iy + 1) * (1 - fx) * fy + roll(ix + 1, iy + 1) * fx * fy)


def bilinear_sample(img, xs, ys):
    """按坐标场采样 wrap 纹理(xs/ys 形状 [h, w])。"""
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(np.int64) % w
    y0 = np.floor(ys).astype(np.int64) % h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    fx = (xs - np.floor(xs)).astype(np.float32)[..., None]
    fy = (ys - np.floor(ys)).astype(np.float32)[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
            img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def center_crop(arr, h, w):
    ch, cw = arr.shape[:2]
    y0 = max(0, (ch - h) // 2)
    x0 = max(0, (cw - w) // 2)
    return arr[y0:y0 + h, x0:x0 + w]


# ---------------------------------------------------------------- scenes

def scene_translate(h, w, frames, rng):
    speed = float(rng.uniform(3.0, 9.0))
    ang = rng.uniform(0, 2 * math.pi)
    dx, dy = speed * math.cos(ang), speed * math.sin(ang)
    margin = int(math.ceil(speed * frames)) + 4
    tex = make_texture(h + margin, w + margin, rng)
    out = []
    for t in range(frames):
        crop = center_crop(tex, h, w)
        out.append(shift_subpixel(crop, dx * t % margin, dy * t % margin))
    return out, {'motion': 'translate', 'speed': speed, 'angle': float(ang)}


def scene_zoom(h, w, frames, rng):
    rate = float(rng.uniform(0.01, 0.05))          # 每帧缩放比例
    zoom_in = bool(rng.random() < 0.5)
    tex = make_texture(int(h * 1.6), int(w * 1.6), rng)
    pil = Image.fromarray((np.clip(tex, 0, 1) * 255.0 + 0.5).astype(np.uint8), 'RGB')
    out = []
    for t in range(frames):
        s = (1.0 + rate * t) if zoom_in else (1.0 / (1.0 + rate * t))
        nw, nh = max(8, int(round(w * 1.6 * s))), max(8, int(round(h * 1.6 * s)))
        frame = pil.resize((nw, nh), Image.BICUBIC)
        arr = np.asarray(frame, dtype=np.float32) / 255.0
        out.append(center_crop(arr, h, w))
    return out, {'motion': 'zoom', 'rate': rate, 'zoom_in': zoom_in}


def scene_rotate(h, w, frames, rng):
    speed = float(rng.uniform(0.3, 1.5))           # 度/帧
    tex = make_texture(h, w, rng)
    pil = Image.fromarray((np.clip(tex, 0, 1) * 255.0 + 0.5).astype(np.uint8), 'RGB')
    out = []
    for t in range(frames):
        frame = pil.rotate(speed * t, resample=Image.BICUBIC, expand=False)
        out.append(np.asarray(frame, dtype=np.float32) / 255.0)
    return out, {'motion': 'rotate', 'speed': speed}


def scene_particles(h, w, frames, rng):
    bg = make_texture(h, w, rng)
    n = int(rng.integers(30, 90))
    shapes = rng.integers(0, 3, size=n)            # 0 椭圆 1 矩形 2 三角
    px = rng.uniform(0, w, size=n)
    py = rng.uniform(0, h, size=n)
    vx = rng.uniform(-18, 18, size=n)
    vy = rng.uniform(-18, 18, size=n)
    size = rng.uniform(6, max(10.0, min(h, w) * 0.06), size=n)
    colors = rng.integers(30, 255, size=(n, 3))
    out = []
    for t in range(frames):
        img = Image.fromarray((np.clip(bg, 0, 1) * 255.0 + 0.5).astype(np.uint8), 'RGB')
        draw = ImageDraw.Draw(img)
        for i in range(n):
            x = float((px[i] + vx[i] * t) % (w + 60)) - 30
            y = float((py[i] + vy[i] * t) % (h + 60)) - 30
            s = float(size[i])
            c = tuple(int(v) for v in colors[i])
            if shapes[i] == 0:
                draw.ellipse([x - s, y - s, x + s, y + s], fill=c, outline=(255, 255, 255))
            elif shapes[i] == 1:
                draw.rectangle([x - s, y - s * 0.6, x + s, y + s * 0.6], fill=c, outline=(0, 0, 0))
            else:
                draw.polygon([(x, y - s), (x - s, y + s), (x + s, y + s)], fill=c, outline=(255, 255, 0))
        out.append(np.asarray(img, dtype=np.float32) / 255.0)
    return out, {'motion': 'particles', 'count': n}


def scene_checker(h, w, frames, rng):
    """混叠压力测试:亚像素漂移的棋盘 + 斜条纹 + fbm 底。"""
    bg = make_texture(h, w, rng)
    cell_x = float(rng.uniform(2.0, 7.0))
    cell_y = float(rng.uniform(2.0, 7.0))
    phase_x = float(rng.uniform(0, 20))
    phase_y = float(rng.uniform(0, 20))
    vx = float(rng.uniform(0.2, 1.7)) * (1 if rng.random() < 0.5 else -1)
    vy = float(rng.uniform(0.2, 1.7)) * (1 if rng.random() < 0.5 else -1)
    angle = rng.uniform(0, math.pi)
    period = float(rng.uniform(2.5, 9.0))
    blend = float(rng.uniform(0.35, 0.75))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = []
    for t in range(frames):
        check = checker_layer(h, w, cell_x, cell_y, phase_x + vx * t, phase_y + vy * t, xx, yy)
        stripe = stripe_layer(h, w, angle, period, phase_x * 0.7 + vx * t * 0.5, xx, yy)
        pattern = 0.6 * check + 0.4 * stripe
        out.append(np.clip(bg * (1 - blend) + pattern[..., None] * blend, 0, 1))
    return out, {'motion': 'checker', 'cell': [cell_x, cell_y], 'velocity': [vx, vy]}


def scene_flow(h, w, frames, rng):
    """流场 warp:整张纹理按低频速度场逐帧位移。"""
    tex = make_texture(h, w, rng)
    fx = (fbm(h, w, rng, octaves=3) - 0.5) * 2.0
    fy = (fbm(h, w, rng, octaves=3) - 0.5) * 2.0
    speed = float(rng.uniform(4.0, 14.0))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = []
    for t in range(frames):
        out.append(np.clip(bilinear_sample(tex, xx + fx * speed * t, yy + fy * speed * t), 0, 1))
    return out, {'motion': 'flow', 'speed': speed}


SCENE_FUNCS = {
    'translate': scene_translate,
    'zoom': scene_zoom,
    'rotate': scene_rotate,
    'particles': scene_particles,
    'checker': scene_checker,
    'flow': scene_flow,
}


# ---------------------------------------------------------------- driver

def save_png(path, rgb01):
    Image.fromarray((np.clip(rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8), 'RGB').save(path)


def generate_sequence(out_root, seq_index, scene, frames, width, height, seed_root):
    rng = np.random.default_rng(np.random.SeedSequence([seed_root, seq_index]))
    frames_rgb, params = SCENE_FUNCS[scene](height, width, frames, rng)
    seq_dir = os.path.join(out_root, f'seq-{seq_index:04d}-{scene}')
    os.makedirs(seq_dir, exist_ok=True)
    for t, frame in enumerate(frames_rgb):
        save_png(os.path.join(seq_dir, f'f{t:03d}.png'), frame)
    meta = {
        'generator': GENERATOR, 'seq_index': seq_index, 'scene': scene,
        'seed_root': seed_root, 'frames': frames, 'size': [width, height], 'params': params,
    }
    with open(os.path.join(seq_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    return seq_dir


def main():
    parser = argparse.ArgumentParser(description='procedural sequence generator for distillation data')
    parser.add_argument('-o', '--out', default='data/proc', help='output directory')
    parser.add_argument('--count', type=int, default=200, help='number of sequences')
    parser.add_argument('--frames', type=int, default=8, help='frames per sequence')
    parser.add_argument('--size', default='1920x1080', help='frame size WxH')
    parser.add_argument('--seed', type=int, default=20261007, help='root seed (deterministic)')
    parser.add_argument('--types', default=','.join(SCENES),
                        help='comma list of scenes; sequences cycle through them')
    parser.add_argument('--start-index', type=int, default=0,
                        help='first sequence index (for appending more sequences later)')
    args = parser.parse_args()

    width, height = (int(v) for v in args.size.lower().split('x'))
    types = [t.strip() for t in args.types.split(',') if t.strip()]
    unknown = [t for t in types if t not in SCENE_FUNCS]
    if unknown:
        raise SystemExit(f'unknown scene types: {unknown}; known: {SCENES}')
    if width < 32 or height < 32:
        raise SystemExit('size too small')

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, 'index.json')
    for i in range(args.count):
        seq_index = args.start_index + i
        scene = types[seq_index % len(types)]
        seq_dir = generate_sequence(args.out, seq_index, scene, args.frames,
                                    width, height, args.seed)
        print(f'[{i + 1}/{args.count}] {seq_dir}')

    summary = {'generator': GENERATOR, 'seed': args.seed, 'count': args.count,
               'frames': args.frames, 'size': [width, height], 'types': types,
               'start_index': args.start_index}
    with open(index_path, 'w') as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print(f'index: {index_path}')


if __name__ == '__main__':
    main()
