"""按正确协议(整图原生分辨率、一次前向)在清单上评估 checkpoint。

修正了 probes/eval_paths.py 的错误:那里把图切成 512x320 小块,每块用块级
geometry 调 build_features,多尺度 lanes 全按错尺度算 —— 教师只能复现到 30 dB。

这里:
    w, h = 原图尺寸
    g = geometry_from_valid(w, h)
    feats = build_features(proxy, g, seed, style=2, tone=1, structure=1, skin=-1, automask=True)
    head  = model(feats)                    # 整图一次
    out   = clip(proxy + head[:h, :w, :3] / 4)
教师与学生同一套 features,在 float 域比 PSNR。

    CUDA_VISIBLE_DEVICES=0 python3 probes/eval_manifest_full.py <ckpt> <manifest.json>
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path('/home/koth/klss')
sys.path.insert(0, str(ROOT / 'teacher'))

from annotate import TeacherAnnotator              # noqa: E402
from nr_cnn_unet_fast import CNNUNetFast, load_fast_shape  # noqa: E402
from nr_geometry import geometry_from_valid        # noqa: E402
from run_image import build_features               # noqa: E402

SEED = 12345
CONTROLS = (2., 1., 1., -1., True)


def psnr(a, b):
    return 10 * np.log10(1 / max(float(np.mean((a - b) ** 2)), 1e-12))


def main():
    import json
    ckpt_path, manifest_path = sys.argv[1], sys.argv[2]
    # 第三个参数可选:shape 路径(默认原 cnn_unet_fast,可传 pixel-norm 版本)
    shape_path = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / 'shapes/cnn_unet_fast.json'
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    items = [(e['path'], e['source']) for e in manifest['entries'] if e['split'] == 'train']
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print('ckpt step', ckpt.get('step'), '| shape', shape_path.name,
          '| 图片', len(items), '| 整图一次前向', flush=True)

    fits, bases, per_source = [], [], {}
    for path, source in items:
        candidates = [Path(path)] if path.startswith('/') else [ROOT / path, Path('/f/yanr/rs') / path]
        full = next((c for c in candidates if c.exists()), None)
        if full is None:
            print('skip', path)
            continue
        image = Image.open(full).convert('RGB')
        w, h = image.size
        proxy = np.asarray(image, dtype=np.float32) / 255.0
        g = geometry_from_valid(w, h)
        feats = np.ascontiguousarray(build_features(
            proxy, g, SEED, *CONTROLS))

        ann = TeacherAnnotator(str(ROOT / 'models/nr'), g)
        ann.g = ann.network.g = g
        with torch.inference_mode():
            t_head = ann.annotate(feats, capture=(), batch=1)['head']
        t_head = t_head.reshape(g['full_height'], g['full_width'], 4)

        model = CNNUNetFast(load_fast_shape(shape_path), g)
        model.load_state_dict(ckpt['ema'] if 'ema' in ckpt else ckpt['model'])
        model.eval().switch_to_deploy().cuda()
        model.g = g
        with torch.inference_mode():
            s_head = model(torch.from_numpy(feats).cuda())
        s_head = s_head.float().cpu().numpy().reshape(g['full_height'], g['full_width'], 4)
        del model
        torch.cuda.empty_cache()

        teacher = np.clip(proxy + t_head[:h, :w, :3] / 4.0, 0, 1)
        student = np.clip(proxy + s_head[:h, :w, :3] / 4.0, 0, 1)
        fit, base = psnr(student, teacher), psnr(proxy, teacher)
        fits.append(fit); bases.append(base)
        per_source.setdefault(source, []).append((fit, base))
        print(f'  {Path(path).name[:36]:<36} {w}x{h:<5} 拟合 {fit:6.2f} | 基线 {base:6.2f}', flush=True)

    fits, bases = np.array(fits), np.array(bases)
    print(f'均值  拟合 {fits.mean():.2f} dB (最好 {fits.max():.2f} 最差 {fits.min():.2f}) | '
          f'基线 {bases.mean():.2f} | 净提升 {(fits - bases).mean():+.2f}')
    print('按源:')
    for name in sorted(per_source):
        rows = np.array(per_source[name])
        print(f'  {name:<18} n={len(rows):<3} 拟合 {rows[:, 0].mean():6.2f} | '
              f'基线 {rows[:, 1].mean():5.2f} | 净 {(rows[:, 0] - rows[:, 1]).mean():+5.2f}')


if __name__ == '__main__':
    main()
