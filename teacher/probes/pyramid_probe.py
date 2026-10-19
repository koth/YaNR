"""全尺度金字塔:按 2 倍下采样把改动拆成真正的多尺度,算各尺度的复刻比与相关。

上一版用 blur3 重复 1..32 次,实际只覆盖 σ≈0.8-4.6 像素,标签却写成到"整幅",
是错的。这里用标准拉普拉斯金字塔:每层 2x2 均值下采样,相邻层相减得该尺度的
带通分量,最后一层剩直流。覆盖范围从 1px 一直到整幅画面。

  ratio  学生该尺度改动的标准差 / 教师
  corr   相关系数
  err    误差能量 / 教师该尺度能量 = 1 + ratio^2 - 2*ratio*corr
         (1.0 表示"等于什么都不做",<1 才是真的有帮助)

    python3 pyramid_probe.py
"""
import glob
import os
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


def downsample(a):
    h, w, c = a.shape
    h2, w2 = h // 2, w // 2
    a = a[:h2 * 2, :w2 * 2]
    return a.reshape(h2, 2, w2, 2, c).mean((1, 3))


def upsample(a, h, w):
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)[:h, :w]


def pyramid(a):
    """返回 [b0(最细), b1, ..., bN(最粗带), 残余直流]。"""
    levels = [a]
    while min(levels[-1].shape[:2]) >= 8:
        levels.append(downsample(levels[-1]))
    bands = []
    for i in range(len(levels) - 1):
        up = upsample(levels[i + 1], *levels[i].shape[:2])
        bands.append(levels[i] - up)
    bands.append(levels[-1])
    return bands


def main():
    g = geometry_from_valid(W, H)
    teacher = Network(Model('cuda').load('/home/koth/klss/models/nr'), g)
    torch.manual_seed(0)
    student = StudentNetwork(load_student_shape(SHAPE), g).to('cuda').eval()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    student.load_state_dict(ck.get('ema', ck.get('model')))

    acc = None
    names = ('t1', 't2', 't3', 't4', 't5', 't6', 't7')
    for name in names:
        d = f'/home/koth/klss/teacher/samples/{name}'
        src = [f for f in glob.glob(d + '/*')
               if os.path.basename(f) not in ('nr.png', 'teacher.png')][0]
        proxy = np.asarray(Image.open(src).convert('RGB').resize((W, H), Image.LANCZOS),
                           dtype=np.float32) / 255.0
        feats = torch.from_numpy(build_features(
            proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
            RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])).to('cuda')
        outs = {}
        for tag, net in (('t', teacher), ('s', student)):
            with torch.no_grad():
                head = net.record(feats) if tag == 't' else net(feats)
                head = head.float().cpu().numpy().reshape(g['full_height'],
                                                          g['full_width'], 4)
            outs[tag] = head[:H, :W, :3] / 4.0
        bt, bs = pyramid(outs['t']), pyramid(outs['s'])
        if acc is None:
            acc = [[] for _ in bt]
        for i, (a, b) in enumerate(zip(bt, bs)):
            st, ss = a.std(), b.std()
            ratio = ss / st if st > 1e-9 else np.nan
            corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if st > 1e-9 else np.nan
            acc[i].append((ratio, corr, st * 255))
        del feats
        torch.cuda.empty_cache()

    print(f'{"尺度":<12}{"教师幅度":>9}{"复刻比":>15}{"相关系数":>15}{"误差/教师":>11}')
    for i, rows in enumerate(acc):
        a = np.array(rows, dtype=float)
        ratio, corr, amp = np.nanmean(a[:, 0]), np.nanmean(a[:, 1]), np.nanmean(a[:, 2])
        err = 1 + ratio ** 2 - 2 * ratio * corr
        if i == len(acc) - 1:
            label = '残余(最低频)'
        else:
            lo, hi = 2 ** i, 2 ** (i + 1)
            label = f'{lo}-{hi}px'
        print(f'{label:<12}{amp:9.2f}{ratio:9.2f}±{np.nanstd(a[:, 0]):.2f}'
              f'{corr:9.2f}±{np.nanstd(a[:, 1]):.2f}{err:11.2f}')


if __name__ == '__main__':
    main()
