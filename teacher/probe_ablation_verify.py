"""确认消融 hook 真的生效:逐条件打印 s-vit 的空间 std 与最终 head 的差异。

若 vit-mean 条件下 s-vit 的 std 降到 ~0 而 head 几乎不变,则"ViT 空间内容无用"
成立;若 s-vit 的 std 没降,说明 hook 没打到预期位置,消融结论作废。
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

g = geometry_from_valid(W, H)
torch.manual_seed(0)
student = StudentNetwork(load_student_shape(SHAPE), g).to('cuda').eval()
student.load_state_dict(torch.load(CKPT, map_location='cpu',
                                   weights_only=False).get('ema'))
student.capture_enabled = True

files = []
for r in ('DIV2K_train_HR', 'Flickr2K', 'gameplay', 'portraits', 'rgb'):
    files += [f for f in glob.glob(f'/f/yanr/rs/{r}/**/*.*', recursive=True)
              if f.lower().endswith(('.png', '.jpg'))]
random.seed(20261007)
path = random.sample(files, 1)[0]
img = Image.open(path).convert('RGB')
if img.width < W or img.height < H:
    img = img.resize((max(W, img.width), max(H, img.height)), Image.LANCZOS)
x = random.randint(0, img.width - W)
y = random.randint(0, img.height - H)
proxy = np.asarray(img.crop((x, y, x + W, y + H)), dtype=np.float32) / 255.0
feats = torch.from_numpy(build_features(
    proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
    RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])).to('cuda')

handles = []


def clear():
    for h in handles:
        h.remove()
    handles.clear()


def run(tag):
    with torch.no_grad():
        head = student(feats)
    v = student.captures['s-vit'].float()
    d4 = student.captures['s-dec-d4'].float()
    out = student.captures['s-head'].float()
    return tag, float(v.std()), float(v.mean().abs()), float(d4.std()), head.clone()


rows = [run('baseline')]
base_head = rows[0][4]

clear()
for block in student.vit_blocks:
    handles.append(block.register_forward_hook(lambda m, i, o: i[0]))
rows.append(run('no-vit'))
clear()

def mean_hook(m, i, o):
    return o.mean(0, keepdim=True).expand_as(o)

handles.append(student.vit_blocks[-1].register_forward_hook(mean_hook))
rows.append(run('vit-mean'))
clear()

rows.append(run('restored'))

print(f'{"条件":<10}{"s-vit std":>11}{"|mean|":>9}{"s-dec-d4 std":>14}'
      f'{"head L1 vs base":>17}{"head 相对变化":>14}')
b = base_head.abs().mean()
for tag, sd, mu, d4sd, head in rows:
    diff = float((head - base_head).abs().mean())
    print(f'{tag:<10}{sd:11.4f}{mu:9.4f}{d4sd:14.4f}{diff:17.6f}{diff / max(float(b), 1e-9):13.3f}')
