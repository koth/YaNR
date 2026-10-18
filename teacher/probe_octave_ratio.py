"""按倍频程拆改动,看学生的"复刻比"是各频段一致偏低,还是某个频段特有。

若各频段比值都在 0.3-0.6 左右 -> 单一原因:整体幅度不足(增益问题);
若只有某段低 -> 那个频段特有,需要单独处理。
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
OCTAVES = (1, 2, 4, 8, 16, 32)


def blur3(a):
    h, w, _ = a.shape
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    s = np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            s += p[i:i + h, j:j + w]
    return s / 9.0


def lows(a, upto):
    """返回 [a, blur(a), blur^2(a), ...] 到 upto 次。"""
    out = [a]
    for _ in range(upto):
        out.append(blur3(out[-1]))
    return out


def main():
    g = geometry_from_valid(W, H)
    teacher = Network(Model('cuda').load('/home/koth/klss/models/nr'), g)
    torch.manual_seed(0)
    student = StudentNetwork(load_student_shape(SHAPE), g).to('cuda').eval()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    student.load_state_dict(ck.get('ema', ck.get('model')))

    ratios = {k: [] for k in OCTAVES}
    corrs = {k: [] for k in OCTAVES}
    ratios['DC'] = []
    corrs['DC'] = []
    labels = {'DC': '直流(均值)'}
    for k in OCTAVES:
        labels[k] = f'~{2 ** (k / 2):.0f}px 尺度'

    for name in ('t1', 't2', 't3', 't4', 't5', 't6', 't7'):
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
            outs[tag] = head[:H, :W, :3] / 4.0          # 未截断的原始改动
        dt, ds = outs['t'], outs['s']
        ratios['DC'].append(ds.mean() / dt.mean() if abs(dt.mean()) > 1e-6 else np.nan)
        corrs['DC'].append(float(np.corrcoef(dt.ravel(), ds.ravel())[0, 1]))
        lo_t, lo_s = lows(dt, max(OCTAVES)), lows(ds, max(OCTAVES))
        for k in OCTAVES:
            bt = lo_t[k // 2] - lo_t[k]
            bs = lo_s[k // 2] - lo_s[k]
            ratios[k].append(bs.std() / bt.std() if bt.std() > 1e-9 else np.nan)
            corrs[k].append(float(np.corrcoef(bt.ravel(), bs.ravel())[0, 1]))
        del feats
        torch.cuda.empty_cache()

    print(f'{"分量":<14}{"复刻比(学生/教师)":>24}{"相关系数":>20}{"比值/corr":>12}')
    for key in ['DC'] + list(OCTAVES):
        v = np.array(ratios[key], dtype=float)
        c = np.array(corrs[key], dtype=float)
        m = ~np.isnan(v) & ~np.isnan(c)
        v, c = v[m], c[m]
        # 最小二乘最优增益就是 corr:若实测比值≈corr,说明当前的"幅度不足"
        # 已经是最优解,问题在可预测性而非权重。
        agg = float(np.nanmean(v / np.where(np.abs(c) > 1e-6, c, np.nan)))
        print(f'{labels[key]:<14}{v.mean():12.2f} ± {v.std():.2f}   '
              f'{c.mean():9.2f} ± {c.std():.2f}   {agg:9.2f}')


if __name__ == '__main__':
    main()
