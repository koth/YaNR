"""整图铺块评估:s-vs-t 的全局 PSNR(按平方误差累加),不是单块采样。

eval_paths.py 只取中心 512x320,对 1920x1080 的图只覆盖约 22% 面积,数字会偏。
这里以 512x320 为窗口、按半窗步长滑动覆盖整图,累加平方误差后算一个全局 PSNR。

    CUDA_VISIBLE_DEVICES=0 python3 probes/eval_full_image.py <ckpt> <paths.json>
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
STEP_X, STEP_Y = W // 2, H // 2
CONTROLS = (2., 1., 1., -1., True)


def main():
    ckpt_path, paths_file = sys.argv[1], sys.argv[2]
    raw = json.loads(Path(paths_file).read_text(encoding='utf-8'))
    items = ([(e['path'], e.get('source', '?')) for e in raw['entries']
              if e['split'] == 'train'] if isinstance(raw, dict) else [(p, '?') for p in raw])

    g = geometry_from_valid(W, H)
    ann = TeacherAnnotator(str(ROOT / 'models/nr'), g)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = CNNUNetFast(load_fast_shape(ROOT / 'shapes/cnn_unet_fast.json'), g)
    model.load_state_dict(ckpt['ema'] if 'ema' in ckpt else ckpt['model'])
    model.eval().switch_to_deploy().cuda()
    model.g = g
    print('ckpt step', ckpt.get('step'), flush=True)

    for path, source in items:
        candidates = [Path(path)] if path.startswith('/') else [ROOT / path, Path('/f/yanr/rs') / path]
        full = next((c for c in candidates if c.exists()), None)
        if full is None:
            print('skip', path)
            continue
        image = read_rgb(full)
        ratio = max(W / image.width, H / image.height)
        resized = image.resize((max(W, round(image.width * ratio)),
                                max(H, round(image.height * ratio))), RESAMPLE)
        # 补足到整块网格
        pad_w = (-(resized.width - W)) % STEP_X
        pad_h = (-(resized.height - H)) % STEP_Y
        canvas = np.asarray(resized, dtype=np.float32) / 255
        if pad_w or pad_h:
            canvas = np.pad(canvas, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

        se_student = se_base = 0.0
        count = 0
        for top in range(0, canvas.shape[0] - H + 1, STEP_Y):
            for left in range(0, canvas.shape[1] - W + 1, STEP_X):
                proxy = canvas[top:top + H, left:left + W]
                feats = build_features(proxy, g, 91, *CONTROLS)
                ann.g = ann.network.g = g
                with torch.inference_mode():
                    t_head = ann.annotate(feats.reshape(-1, 16), capture=(), batch=1)['head']
                with torch.inference_mode():
                    s_head = model(torch.from_numpy(feats).cuda()).cpu().numpy()
                t_head = t_head.reshape(H, W, 4)
                s_head = s_head.reshape(H, W, 4)
                st = np.clip(proxy + s_head[:, :, :3] / 4.0, 0, 1)
                tt = np.clip(proxy + t_head[:, :, :3] / 4.0, 0, 1)
                se_student += float(np.sum((st - tt) ** 2))
                se_base += float(np.sum((proxy - tt) ** 2))
                count += st.size
        p = 10 * np.log10(1 / max(se_student / count, 1e-12))
        b = 10 * np.log10(1 / max(se_base / count, 1e-12))
        print(f'{Path(path).name:<12} 块数 {count // (W * H * 3):<3} '
              f'整图 PSNR {p:6.2f} dB | 基线 {b:6.2f} | 净提升 {p - b:+.2f}', flush=True)


if __name__ == '__main__':
    main()
