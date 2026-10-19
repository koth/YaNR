"""同一张图在不同尺寸下,教师/学生的全局色调偏移各是多少。

全局色调依赖大范围上下文:尺寸一变,同样的感受野覆盖的画面比例就变了。
若学生在训练尺寸(320x512)复现了压暗、在大尺寸没有,那就是尺寸泛化问题,
而不是"学不会色调"。
"""
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/home/koth/klss/teacher')
from nr_geometry import geometry_from_valid          # noqa: E402
from nr_model import Model                            # noqa: E402
from nr_network import Network                        # noqa: E402
from nr_student import StudentNetwork, load_student_shape  # noqa: E402
from run_image import build_features                  # noqa: E402

SRC = sys.argv[1]
SIZES = [(320, 512), (480, 768), (640, 1024), (960, 540), (1600, 900)]
CKPT = '/home/koth/keep_tmp/v45d_56k/ckpt.pt'
SHAPE = '/home/koth/klss/shapes/student_shared_moe_v1.json'
RECIPE = dict(seed=12345, style=2.0, tone=1.0, structure=1.0, skin=-1.0, automask=True)

shape = load_student_shape(SHAPE)
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
state = ck.get('ema', ck.get('model'))

print(f'{"尺寸":>11}{"教师 RGB 偏移":>26}{"学生 RGB 偏移":>26}{"psnr s-t":>10}')
for w, h in SIZES:
    proxy = np.asarray(Image.open(SRC).convert('RGB').resize((w, h), Image.LANCZOS),
                       dtype=np.float32) / 255.0
    g = geometry_from_valid(w, h)
    feats = torch.from_numpy(build_features(
        proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
        RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])).to('cuda')

    teacher = Network(Model('cuda').load('/home/koth/klss/models/nr'), g)
    with torch.no_grad():
        head = teacher.record(feats).float().cpu().numpy().reshape(
            g['full_height'], g['full_width'], 4)
    tout = np.clip(proxy + head[:h, :w, :3] / 4.0, 0, 1)
    del teacher, head
    torch.cuda.empty_cache()

    torch.manual_seed(0)
    student = StudentNetwork(shape, g).to('cuda').eval()
    student.load_state_dict(state)
    with torch.no_grad():
        head = student(feats).float().cpu().numpy().reshape(
            g['full_height'], g['full_width'], 4)
    sout = np.clip(proxy + head[:h, :w, :3] / 4.0, 0, 1)
    del student, head, feats
    torch.cuda.empty_cache()

    dt = (tout - proxy).mean((0, 1)) * 255
    ds = (sout - proxy).mean((0, 1)) * 255
    mse = float(((sout.astype(np.float64) - tout) ** 2).mean())
    print(f'{w}x{h:<7}[{dt[0]:+6.2f} {dt[1]:+6.2f} {dt[2]:+6.2f}]   '
          f'[{ds[0]:+6.2f} {ds[1]:+6.2f} {ds[2]:+6.2f}]   '
          f'{10 * np.log10(1 / max(mse, 1e-12)):8.2f}', flush=True)
