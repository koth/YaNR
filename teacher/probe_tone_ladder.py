"""色调响应阶梯:固定输入、扫亮度,看教师和学生对亮度的响应曲线。

学生如果只是"幅度弱",响应曲线应随亮度单调变化且符号与教师一致;
如果"根本没学会",曲线会全程贴着 0。

两组输入:
  A 平灰:纯常数,隔离出纯色调映射(不受画面内容干扰)
  B 真实图:固定样例整体增益,看有内容时是否不同

    python3 tone_ladder.py [图片路径]
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
IMG = sys.argv[1] if len(sys.argv) > 1 else '/home/koth/klss/teacher/samples/t1/t1.png'


def run_pair(proxy, g, teacher, student):
    feats = torch.from_numpy(build_features(
        proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
        RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])).to('cuda')
    outs = {}
    for tag, net in (('t', teacher), ('s', student)):
        with torch.no_grad():
            head = net.record(feats) if tag == 't' else net(feats)
            head = head.float().cpu().numpy().reshape(g['full_height'], g['full_width'], 4)
        outs[tag] = np.clip(proxy + head[:H, :W, :3] / 4.0, 0, 1)
    return outs['t'], outs['s']


def main():
    g = geometry_from_valid(W, H)
    teacher = Network(Model('cuda').load('/home/koth/klss/models/nr'), g)
    torch.manual_seed(0)
    student = StudentNetwork(load_student_shape(SHAPE), g).to('cuda').eval()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    student.load_state_dict(ck.get('ema', ck.get('model')))

    print('=== A: 平灰阶梯(纯色调映射) ===')
    print(f'{"输入灰度":>9}{"教师响应":>26}{"学生响应":>26}')
    for level in (0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.98):
        proxy = np.full((H, W, 3), level, dtype=np.float32)
        t, s = run_pair(proxy, g, teacher, student)
        dt = (t - proxy).mean((0, 1)) * 255
        ds = (s - proxy).mean((0, 1)) * 255
        print(f'{level:9.2f}   [{dt[0]:+6.2f} {dt[1]:+6.2f} {dt[2]:+6.2f}]   '
              f'[{ds[0]:+6.2f} {ds[1]:+6.2f} {ds[2]:+6.2f}]', flush=True)

    print('\n=== B: 真实图整体增益 ===')
    base = np.asarray(Image.open(IMG).convert('RGB').resize((W, H), Image.LANCZOS),
                      dtype=np.float32) / 255.0
    print(f'{"增益":>6}{"输入均值":>9}{"教师响应":>26}{"学生响应":>26}')
    for gain in (0.4, 0.6, 0.8, 1.0, 1.25, 1.5):
        proxy = np.clip(base * gain, 0, 1).astype(np.float32)
        t, s = run_pair(proxy, g, teacher, student)
        dt = (t - proxy).mean((0, 1)) * 255
        ds = (s - proxy).mean((0, 1)) * 255
        print(f'{gain:6.2f}{proxy.mean():9.3f}   [{dt[0]:+6.2f} {dt[1]:+6.2f} {dt[2]:+6.2f}]   '
              f'[{ds[0]:+6.2f} {ds[1]:+6.2f} {ds[2]:+6.2f}]', flush=True)


if __name__ == '__main__':
    main()
