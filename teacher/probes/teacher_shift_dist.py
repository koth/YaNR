"""教师改动在训练分布上的分布:平均偏移是集中在 0 附近,还是系统性偏负?

如果训练分布上教师偏移接近 0 且离散,那么学生在 7 张样例上输出 ~0 就是 L1 的
条件中位数,属于"没学会自适应",不是"结构性不能产生直流";
如果训练分布本身就系统性偏负,学生输出 0 才是真正的失败。
"""
import glob
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/home/koth/klss/teacher')
from nr_geometry import geometry_from_valid                  # noqa: E402
from nr_model import Model                                    # noqa: E402
from nr_network import Network                                # noqa: E402
from run_image import build_features                          # noqa: E402

W, H = 320, 512
RECIPE = dict(seed=12345, style=2.0, tone=1.0, structure=1.0, skin=-1.0, automask=True)
ROOTS = ['/f/yanr/rs/DIV2K_train_HR', '/f/yanr/rs/Flickr2K', '/f/yanr/rs/gameplay',
         '/f/yanr/rs/portraits', '/f/yanr/rs/rgb']
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24

files = []
for root in ROOTS:
    files += [f for f in glob.glob(root + '/**/*.*', recursive=True)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.seed(20261007)
picked = random.sample(files, min(N, len(files)))
print(f'从 {len(files)} 张里抽 {len(picked)} 张,各随机裁剪 {W}x{H}')

g = geometry_from_valid(W, H)
teacher = Network(Model('cuda').load('/home/koth/klss/models/nr'), g)
shifts = []
print(f'{"#":>3}{"平均偏移 R/G/B":>26}{"改动标准差":>10}{"输入均值":>10}')
for i, path in enumerate(picked):
    img = Image.open(path).convert('RGB')
    if img.width < W or img.height < H:
        img = img.resize((max(W, img.width), max(H, img.height)), Image.LANCZOS)
    x = random.randint(0, img.width - W)
    y = random.randint(0, img.height - H)
    crop = img.crop((x, y, x + W, y + H))
    proxy = np.asarray(crop, dtype=np.float32) / 255.0
    feats = torch.from_numpy(build_features(
        proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
        RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])).to('cuda')
    with torch.no_grad():
        head = teacher.record(feats).float().cpu().numpy().reshape(
            g['full_height'], g['full_width'], 4)
    raw = head[:H, :W, :3] / 4.0
    shift = raw.mean((0, 1)) * 255
    shifts.append(shift)
    print(f'{i:>3}[{shift[0]:+6.2f} {shift[1]:+6.2f} {shift[2]:+6.2f}]'
          f'{raw.std() * 255:10.2f}{proxy.mean():10.3f}', flush=True)
    del feats, head
    torch.cuda.empty_cache()

a = np.array(shifts)
print(f'\n均值 [{a[:, 0].mean():+6.2f} {a[:, 1].mean():+6.2f} {a[:, 2].mean():+6.2f}]'
      f'   标准差 [{a[:, 0].std():5.2f} {a[:, 1].std():5.2f} {a[:, 2].std():5.2f}]')
print(f'通道平均: 均值 {a.mean():+6.2f}  标准差 {a.std():5.2f}  '
      f'负向占比 {100 * (a < 0).mean():.0f}%')
