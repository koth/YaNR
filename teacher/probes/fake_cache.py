"""用假教师数据预填评估缓存,只为验证"只跑学生"的显存占用与流程。

真实教师输出需要 1.5GB+ 显存(训练占满时加载不进来);这里用 proxy 加噪声冒充
教师输出,数值无意义,只用来跑通 student-only 路径。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, '/home/koth/klss/teacher')
from nr_geometry import geometry_from_valid                     # noqa: E402
from run_image import build_features                            # noqa: E402

W, H = 320, 512
CFG = dict(style=2.0, tone=1.0, structure=1.0, skin=-1.0, automask=True)
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
g = geometry_from_valid(W, H)
rng = np.random.default_rng(0)

for path in sys.argv[2:]:
    proxy = np.asarray(Image.open(path).convert('RGB').resize((W, H), Image.LANCZOS),
                       dtype=np.float32) / 255.0
    feats = build_features(proxy, g, 7, CFG['style'], CFG['tone'], CFG['structure'],
                           CFG['skin'], CFG['automask'])
    stem = (f'{os.path.basename(path)}.{W}x{H}.s{CFG["style"]}t{CFG["tone"]}'
            f'k{CFG["structure"]}')
    np.save(os.path.join(OUT, stem + '.feats.npy'), feats)
    fake = np.clip(proxy + rng.normal(0, 0.004, proxy.shape).astype(np.float32), 0, 1)
    np.save(os.path.join(OUT, stem + '.teacher.npy'), fake)
    print(f'{stem}: feats {feats.shape} {feats.dtype}')
