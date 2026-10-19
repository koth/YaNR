"""反推 samples/t*/teacher.png 用的条件档位与种子。

教师是确定性的:同一图 + 同一 (style, tone, structure, skin, automask, seed) 必然
给出同一张图。所以把候选组合各跑一遍,和 teacher.png 比 PSNR —— 命中者就是当初
的配方,后续学生的 nr.png 才能和 teacher.png 放在同一口径下比较。

用法: python3 probe_sample_recipe.py <teacher.png 路径> <原图路径>
"""
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/home/koth/klss/teacher')
from nr_geometry import geometry_from_valid          # noqa: E402
from nr_model import Model                            # noqa: E402
from nr_network import Network                        # noqa: E402
from run_image import build_features                  # noqa: E402


def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def main():
    ref_path, src_path = sys.argv[1], sys.argv[2]
    ref = np.asarray(Image.open(ref_path).convert('RGB'), dtype=np.float32) / 255.0
    h, w = ref.shape[:2]
    src = np.asarray(Image.open(src_path).convert('RGB').resize((w, h), Image.LANCZOS),
                     dtype=np.float32) / 255.0
    g = geometry_from_valid(w, h)
    print(f'{src_path}  {w}x{h} -> field {g["full_width"]}x{g["full_height"]} ({g["full_rows"]} rows)')

    teacher = Network(Model('cuda').load('/home/koth/klss/models/nr'), g)
    candidates = []
    for seed in (12345, 7, 0, 20261007):
        for cond in ((2.0, 1.0, 1.0, -1.0, True),      # 训练目标区
                     (2.0, 1.0, 1.0, -1.0, False),
                     (1.5, 0.5, 0.35, -1.0, False),    # eval_conditions BASE
                     (2.0, 0.5, 0.5, -1.0, True),
                     (0.0, 0.5, 0.5, -1.0, False)):
            candidates.append((seed, cond))

    best = None
    for seed, cond in candidates:
        style, tone, structure, skin, automask = cond
        feats = torch.from_numpy(build_features(src, g, seed, style, tone, structure,
                                                skin, automask)).to('cuda')
        with torch.no_grad():
            head = teacher.record(feats)
            head = head.float().cpu().numpy().reshape(g['full_height'], g['full_width'], 4)
        out = np.clip(src + head[:h, :w, :3] / 4.0, 0, 1)
        score = psnr(out, ref)
        diff = float(np.abs(out - ref).max())
        tag = (f'seed={seed:<9} style={style} tone={tone} struct={structure} '
               f'skin={skin} am={int(automask)}')
        print(f'  {tag}  psnr={score:6.2f}dB  maxdiff={diff:.4f}')
        if best is None or score > best[0]:
            best = (score, tag)
        del feats, head
        torch.cuda.empty_cache()
    print(f'\n最接近: {best[1]}  ({best[0]:.2f} dB)')


if __name__ == '__main__':
    main()
