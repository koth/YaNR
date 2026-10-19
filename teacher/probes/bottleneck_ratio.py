"""ViT 为何无效:量瓶颈通路与跳连的相对幅度。

解码时每级都 merge(来自瓶颈的 state, 来自编码器的 skip)。若瓶颈信号的幅度
远小于 skip,那么无论 ViT 算出什么都会被淹没 —— 这解释了消融里 vit-mean 与
baseline 完全一致。

在 merge 上挂 forward_pre_hook,记录两个输入的 rms 比值。
"""
import glob
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/home/koth/klss/teacher')
from nr_geometry import geometry_from_valid                  # noqa: E402
from nr_student import StudentNetwork, load_student_shape     # noqa: E402
from run_image import build_features                          # noqa: E402

W, H = 320, 512
CKPT = '/home/koth/keep_tmp/v45d_56k/ckpt.pt'
SHAPE = '/home/koth/klss/shapes/student_shared_moe_v1.json'
RECIPE = dict(seed=12345, style=2.0, tone=1.0, structure=1.0, skin=-1.0, automask=True)
ROOTS = ['/f/yanr/rs/DIV2K_train_HR', '/f/yanr/rs/Flickr2K', '/f/yanr/rs/gameplay',
         '/f/yanr/rs/portraits', '/f/yanr/rs/rgb']

g = geometry_from_valid(W, H)
torch.manual_seed(0)
student = StudentNetwork(load_student_shape(SHAPE), g).to('cuda').eval()
student.load_state_dict(torch.load(CKPT, map_location='cpu',
                                   weights_only=False).get('ema'))

rec = {}
def make_hook(name):
    def hook(module, args):
        state, skip = args[0], args[1]
        rec[name] = (float(state.std()), float(skip.std()))
    return hook

for name, module in student.merge.items():
    module.register_forward_pre_hook(make_hook(name))

random.seed(20261007)
files = []
for root in ROOTS:
    files += [f for f in glob.glob(root + '/**/*.*', recursive=True)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
picked = random.sample(files, 12)

acc = {}
for path in picked:
    img = Image.open(path).convert('RGB')
    if img.width < W or img.height < H:
        img = img.resize((max(W, img.width), max(H, img.height)), Image.LANCZOS)
    x = random.randint(0, img.width - W)
    y = random.randint(0, img.height - H)
    proxy = np.asarray(img.crop((x, y, x + W, y + H)), dtype=np.float32) / 255.0
    feats = torch.from_numpy(build_features(
        proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
        RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])).to('cuda')
    with torch.no_grad():
        student(feats)
    for k, v in rec.items():
        acc.setdefault(k, []).append(v)
    del feats
    torch.cuda.empty_cache()

print(f'{"merge 级":<10}{"瓶颈通路 rms":>14}{"跳连 rms":>11}{"瓶颈/跳连":>11}')
for name in ('d4', 'd3', 'd2', 'd1', 'd0'):
    if name not in acc:
        continue
    s = np.mean([a[0] for a in acc[name]])
    k = np.mean([a[1] for a in acc[name]])
    print(f'{name:<10}{s:14.4f}{k:11.4f}{s / max(k, 1e-9):10.3f}')
