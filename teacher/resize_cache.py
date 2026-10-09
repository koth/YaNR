#!/usr/bin/env python3
"""训练数据解码缓存:一次性把大图缩到 max-side 1280(JPEG q95),镜像源目录树。

背景:v1.3 训练步长被同步数据加载卡死(2K PNG 解码 200-400ms/张,纯 CPU 准备 ~2.8s/步 >
GPU 计算)。裁剪采样最高用到 2×512=1024 的原生区域,1280 边长的缓存对训练语义无损
(512² 目标下 LANCZOS 重采样链等价),解码降 3-4×。

    python3 resize_cache.py src1 src2 ... --out-root /f/yanr/rs --workers 8

输出:<out-root>/<源根名>/... 镜像树;max side ≤ 1280 的图原样拷贝(不重编码)。
"""
import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp')
MAX_SIDE = 1280


def process(job):
    src, dst = job
    if os.path.exists(dst):
        return 'skip'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        img = Image.open(src).convert('RGB')
    except Exception as exc:                                    # noqa: BLE001
        return f'ERR {src}: {exc!r}'
    if max(img.size) <= MAX_SIDE and src.lower().endswith(('.jpg', '.jpeg')):
        shutil.copyfile(src, dst)                               # 已够小的 JPEG 原样保留
        return 'copy'
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    img.save(dst, 'JPEG', quality=95)
    return 'resize'


def main():
    parser = argparse.ArgumentParser(description='training image decode cache')
    parser.add_argument('roots', nargs='+', help='源图根目录(递归)')
    parser.add_argument('--out-root', required=True, help='缓存根目录(镜像源树)')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    jobs = []
    for root in args.roots:
        base = os.path.basename(os.path.normpath(root))
        for dirpath, _, filenames in os.walk(root):
            for name in sorted(filenames):
                if name.lower().endswith(IMAGE_EXTS):
                    rel = os.path.relpath(os.path.join(dirpath, name), root)
                    dst = os.path.join(args.out_root, base, rel)
                    dst = os.path.splitext(dst)[0] + '.jpg'
                    jobs.append((os.path.join(dirpath, name), dst))
    print(f'{len(jobs)} images -> {args.out_root}')
    counts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(process, jobs, chunksize=16)):
            key = result.split()[0]
            counts[key] = counts.get(key, 0) + 1
            if (i + 1) % 500 == 0:
                print(f'  {i + 1}/{len(jobs)}  {counts}')
    print('done:', counts)
    return 0


if __name__ == '__main__':
    sys.exit(main())
