"""区分"教师真的压暗"还是"clip 截断造成的假压暗"。

教师/学生的合成都是 clip(proxy + head/4, 0, 1)。若原始改动均值≈0、但高光被
截断,clip 本身就会产生净负偏移 —— 那学生的"不压暗"就不算错。
这里同时给出未截断的原始改动均值、截断后均值、以及被截断的像素比例。
"""
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/home/koth/klss/teacher')
from nr_geometry import geometry_from_valid                  # noqa: E402
from nr_model import Model                                    # noqa: E402
from nr_network import Network                                # noqa: E402
from nr_student import StudentNetwork, load_student_shape     # noqa: E402
from run_image import build_features                          # noqa: E402

W, H = 320, 512
CKPT = '/home/koth/keep_tmp/v45d_56k/ckpt.pt'
SHAPE = '/home/koth/klss/shapes/student_shared_moe_v1.json'
RECIPE = dict(seed=12345, style=2.0, tone=1.0, structure=1.0, skin=-1.0, automask=True)

g = geometry_from_valid(W, H)
teacher = Network(Model('cuda').load('/home/koth/klss/models/nr'), g)
torch.manual_seed(0)
student = StudentNetwork(load_student_shape(SHAPE), g).to('cuda').eval()
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
student.load_state_dict(ck.get('ema', ck.get('model')))

print(f'{"目录":<6}{"原始均值 教师":>22}{"原始均值 学生":>22}{"截断%":>14}')
for name in ('t1', 't2', 't3', 't4', 't5', 't6', 't7'):
    import glob
    import os
    d = f'/home/koth/klss/teacher/samples/{name}'
    src = [f for f in glob.glob(d + '/*')
           if os.path.basename(f) not in ('nr.png', 'teacher.png')][0]
    proxy = np.asarray(Image.open(src).convert('RGB').resize((W, H), Image.LANCZOS),
                       dtype=np.float32) / 255.0
    feats = torch.from_numpy(build_features(
        proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
        RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])).to('cuda')
    rows = {}
    for tag, net in (('t', teacher), ('s', student)):
        with torch.no_grad():
            head = net.record(feats) if tag == 't' else net(feats)
            head = head.float().cpu().numpy().reshape(g['full_height'], g['full_width'], 4)
        raw = head[:H, :W, :3] / 4.0          # 未截断的改动
        rows[tag] = raw
    raw_t = rows['t'].mean((0, 1)) * 255
    raw_s = rows['s'].mean((0, 1)) * 255
    # 被截断的比例(教师侧):原始改动把值推出 [0,1] 的像素占比
    lo = np.clip(proxy + rows['t'], 0, 1) != (proxy + rows['t'])
    print(f'{name:<6}[{raw_t[0]:+6.2f} {raw_t[1]:+6.2f} {raw_t[2]:+6.2f}]   '
          f'[{raw_s[0]:+6.2f} {raw_s[1]:+6.2f} {raw_s[2]:+6.2f}]   '
          f'{100 * lo.mean():6.2f}% /{100 * ((np.clip(proxy + rows["s"], 0, 1)) != (proxy + rows["s"])).mean():5.2f}%',
          flush=True)
