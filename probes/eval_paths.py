"""实验 B 第 3 步:在指定图片清单上评估某个 checkpoint 的 s-vs-t。

用于过拟合测试:训练完在同一批图上量,看这套架构到底能不能拟合到 36-38 dB。
    CUDA_VISIBLE_DEVICES=0 python3 probes/eval_paths.py <ckpt> <paths.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/koth/klss')
sys.path.insert(0, str(ROOT / 'teacher'))

from annotate import TeacherAnnotator              # noqa: E402
from nr_cnn_unet_fast import CNNUNetFast, load_fast_shape  # noqa: E402
from nr_geometry import geometry_from_valid        # noqa: E402
from run_image import build_features               # noqa: E402
from single_frame_data import read_rgb, RESAMPLE   # noqa: E402

W, H = 512, 320
CONTROLS = (2., 1., 1., -1., True)


def main():
    ckpt_path, paths_file = sys.argv[1], sys.argv[2]
    raw = json.loads(Path(paths_file).read_text(encoding='utf-8'))
    if isinstance(raw, dict) and 'entries' in raw:      # 直接传 manifest
        items = [(e['path'], e['source']) for e in raw['entries'] if e['split'] == 'train']
    else:                                                # 传路径列表
        items = [(p, '?') for p in raw]
    paths = [p for p, _ in items]
    g = geometry_from_valid(W, H)
    ann = TeacherAnnotator(str(ROOT / 'models/nr'), g)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = CNNUNetFast(load_fast_shape(ROOT / 'shapes/cnn_unet_fast.json'), g)
    model.load_state_dict(ckpt['ema'] if 'ema' in ckpt else ckpt['model'])
    model.eval().switch_to_deploy().cuda()
    model.g = g
    print('ckpt step', ckpt.get('step'), '| 图片数', len(paths), flush=True)

    psnrs = []
    per_source = {}
    for path, source in items:
        # 依次尝试:绝对路径、项目根相对、数据根相对
        candidates = [Path(path)] if path.startswith('/') else [
            ROOT / path, Path('/f/yanr/rs') / path]
        full = next((c for c in candidates if c.exists()), None)
        if full is None:
            print('skip (not found)', path)
            continue
        try:
            image = read_rgb(full)
        except Exception as error:
            print('skip (decode)', path, error)
            continue
        # 居中取 320x512(与验证口径一致:先缩放到覆盖,再裁)
        ratio = max(W / image.width, H / image.height)
        resized = image.resize((max(W, round(image.width * ratio)),
                                max(H, round(image.height * ratio))), RESAMPLE)
        left = (resized.width - W) // 2
        top = (resized.height - H) // 2
        crop = resized.crop((left, top, left + W, top + H))
        proxy = np.asarray(crop, dtype=np.float32) / 255
        feats = build_features(proxy, g, 91, *CONTROLS)

        ann.g = ann.network.g = g
        with torch.inference_mode():
            t_head = ann.annotate(feats.reshape(-1, 16), capture=(), batch=1)['head']
        with torch.inference_mode():
            s_head = model(torch.from_numpy(feats).cuda()).cpu().numpy()
        t_head = t_head.reshape(H, W, 4)
        s_head = s_head.reshape(H, W, 4)
        student = np.clip(proxy + s_head[:, :, :3] / 4.0, 0, 1)
        teacher = np.clip(proxy + t_head[:, :, :3] / 4.0, 0, 1)
        base = proxy
        mse = float(np.mean((student - teacher) ** 2))
        fit = 10 * np.log10(1 / max(mse, 1e-12))
        b = 10 * np.log10(1 / max(float(np.mean((base - teacher) ** 2)), 1e-12))
        psnrs.append((fit, b))
        per_source.setdefault(source, []).append((fit, b))

    arr = np.array(psnrs)
    print(f'拟合 PSNR: {arr[:, 0].mean():.2f} dB (中位 {np.median(arr[:, 0]):.2f}, '
          f'最好 {arr[:, 0].max():.2f}, 最差 {arr[:, 0].min():.2f})')
    print(f'输入基线: {arr[:, 1].mean():.2f} dB')
    print(f'净提升:   {arr[:, 0].mean() - arr[:, 1].mean():+.2f} dB')
    print('按源:')
    for name in sorted(per_source):
        rows = np.array(per_source[name])
        fit = rows[:, 0]
        print(f'  {name:<18} n={len(fit):<3} 拟合 {fit.mean():6.2f} dB | '
              f'基线 {rows[:, 1].mean():5.2f} | 净提升 {fit.mean() - rows[:, 1].mean():+5.2f}')


if __name__ == '__main__':
    main()
