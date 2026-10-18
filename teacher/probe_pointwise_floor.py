"""上界探针:用逐像素回归拟合"输入16通道 -> 教师改动",看学生是否连这个下限都没到。

思路:学生视觉上是个很大的网络,但它的输出本质上还是从输入 lanes 里算出来的。
若一个**逐像素**的线性/二次回归就能达到学生的相关系数,说明学生的瓶颈不在
"关系太非线性",而在优化或架构把能力浪费了;若逐像素回归远低于学生,说明
学生确实靠空间上下文拿到了逐像素做不到的信息。

逐像素模型是下限(没有空间上下文),学生应当显著超过它。

  python3 upper_bound_probe.py [裁剪数]
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
from nr_student import StudentNetwork, load_student_shape     # noqa: E402
from run_image import build_features                          # noqa: E402

W, H = 320, 512
CKPT = '/home/koth/keep_tmp/v45d_56k/ckpt.pt'
SHAPE = '/home/koth/klss/shapes/student_shared_moe_v1.json'
RECIPE = dict(seed=12345, style=2.0, tone=1.0, structure=1.0, skin=-1.0, automask=True)
ROOTS = ['/f/yanr/rs/DIV2K_train_HR', '/f/yanr/rs/Flickr2K', '/f/yanr/rs/gameplay',
         '/f/yanr/rs/portraits', '/f/yanr/rs/rgb']
SUBSAMPLE = 3000


def pyramid(a):
    levels = [a]
    while min(levels[-1].shape[:2]) >= 8:
        h, w, c = levels[-1].shape
        h2, w2 = h // 2, w // 2
        y = levels[-1][:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, c).mean((1, 3))
        levels.append(y)
    bands = []
    for i in range(len(levels) - 1):
        up = np.repeat(np.repeat(levels[i + 1], 2, 0), 2, 1)[:levels[i].shape[0],
                                                             :levels[i].shape[1]]
        bands.append(levels[i] - up)
    bands.append(levels[-1])
    return bands


def design(x, quad):
    if not quad:
        return x
    i, j = np.triu_indices(x.shape[1])
    return np.concatenate([x, x[:, i] * x[:, j]], axis=1)


def band_corr(dt, dp):
    out = []
    for bt, bp in zip(pyramid(dt), pyramid(dp)):
        st, sp = bt.std(), bp.std()
        out.append(float(np.corrcoef(bt.ravel(), bp.ravel())[0, 1])
                   if st > 1e-9 and sp > 1e-9 else np.nan)
    return out


def main():
    n_crops = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    files = []
    for root in ROOTS:
        files += [f for f in glob.glob(root + '/**/*.*', recursive=True)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.seed(20261007)
    picked = random.sample(files, n_crops)

    g = geometry_from_valid(W, H)
    teacher = Network(Model('cuda').load('/home/koth/klss/models/nr'), g)
    torch.manual_seed(0)
    student = StudentNetwork(load_student_shape(SHAPE), g).to('cuda').eval()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    student.load_state_dict(ck.get('ema', ck.get('model')))
    rng = np.random.default_rng(0)

    def forward(path):
        img = Image.open(path).convert('RGB')
        if img.width < W or img.height < H:
            img = img.resize((max(W, img.width), max(H, img.height)), Image.LANCZOS)
        x = random.randint(0, img.width - W)
        y = random.randint(0, img.height - H)
        proxy = np.asarray(img.crop((x, y, x + W, y + H)), dtype=np.float32) / 255.0
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
        del feats
        torch.cuda.empty_cache()
        return feats_np(proxy), outs['t'], outs['s']

    def feats_np(proxy):
        return build_features(proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
                              RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])

    # --- 拟合集:每张裁牌子采样若干像素
    keep = max(1, n_crops * 2 // 3)
    Xs, Ys = [], []
    for k, path in enumerate(picked):
        if k >= keep:
            break
        f, t, _ = forward(path)
        idx = rng.choice(f.shape[0], SUBSAMPLE, replace=False)
        Xs.append(f[idx])
        Ys.append(t.reshape(-1, 3)[idx])
        del f
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd

    models = {}
    for name, quad in (('线性', False), ('二次', True)):
        D = design(Xn, quad)
        A = D.T @ D + 1e-3 * np.eye(D.shape[1])
        Wc = np.linalg.solve(A, D.T @ Y)          # (feat, 3)
        models[name] = (quad, Wc)
        print(f'拟合 {name}: {D.shape[1]} 维特征, {X.shape[0]} 个像素')

    # --- 测试集:全分辨率逐像素预测,算各尺度相关
    labels = []
    acc = {k: [] for k in list(models) + ['学生']}
    for k, path in enumerate(picked):
        if k < keep:
            continue
        f, t, s = forward(path)
        Xt = (f - mu) / sd
        for name, (quad, Wc) in models.items():
            dt = (design(Xt, quad) @ Wc).reshape(H, W, 3)
            acc[name].append(band_corr(t, dt))
        acc['学生'].append(band_corr(t, s))
        del f, Xt
        if not labels:
            labels = [f'{2 ** i}-{2 ** (i + 1)}px' for i in range(len(acc['学生'][0]) - 1)]
            labels.append('残余')

    print(f'\n{"尺度":<10}' + ''.join(f'{n:>10}' for n in acc) + f'{"教师":>8}')
    for i, lab in enumerate(labels):
        row = ''.join(f'{np.nanmean([r[i] for r in acc[n]]):10.2f}' for n in acc)
        print(f'{lab:<10}{row}{1.0:8.2f}')
    print(f'\n测试裁剪 {len(acc["学生"])} 张')


if __name__ == '__main__':
    main()
