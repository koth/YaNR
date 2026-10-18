"""上界探针 v2:逐像素回归 vs 逐像素+多尺度 vs 学生,看粗尺度信息是否本来可达。

v1 结论:学生的各尺度相关(0.30-0.51)是逐像素线性回归(0.15-0.34)的 1.5-2.1 倍,
说明学生确实用了空间上下文,不是纯逐像素映射。

但粗尺度绝对相关仍然低。本版加一个对照:给逐像素回归补上**多尺度平滑后的原图**
(4/8/16/32px 尺度的低通),如果粗尺度相关明显抬起来,说明信息本来就在输入里、
只是需要"看大范围";学生金字塔在 d2-d4 恰有 8-32px 的特征,若它没吃到这个便宜,
瓶颈就在架构/优化而不是信息缺失。
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
SCALES = (4, 8, 16, 32)
SUBSAMPLE = 2500


def blur3(a):
    h, w, _ = a.shape
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    s = np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            s += p[i:i + h, j:j + w]
    return s / 9.0


def multiscale(proxy):
    """原图按若干尺度过模糊后的通道,作为"大范围上下文"喂给逐像素回归。"""
    out, y = [], proxy
    steps = sorted(SCALES)
    done = 0
    for k in steps:
        while done < k:
            y = blur3(y)
            done += 1
        out.append(y)
    return np.concatenate(out, axis=2).reshape(-1, len(steps) * 3)


def pyramid(a):
    levels = [a]
    while min(levels[-1].shape[:2]) >= 8:
        h, w, c = levels[-1].shape
        h2, w2 = h // 2, w // 2
        levels.append(levels[-1][:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, c).mean((1, 3)))
    bands = []
    for i in range(len(levels) - 1):
        up = np.repeat(np.repeat(levels[i + 1], 2, 0), 2, 1)[:levels[i].shape[0],
                                                             :levels[i].shape[1]]
        bands.append(levels[i] - up)
    bands.append(levels[-1])
    return bands


def band_corr(dt, dp):
    out = []
    for bt, bp in zip(pyramid(dt), pyramid(dp)):
        out.append(float(np.corrcoef(bt.ravel(), bp.ravel())[0, 1]))
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
    student.load_state_dict(torch.load(CKPT, map_location='cpu',
                                       weights_only=False).get('ema'))
    rng = np.random.default_rng(0)

    def forward(path, want_student):
        img = Image.open(path).convert('RGB')
        if img.width < W or img.height < H:
            img = img.resize((max(W, img.width), max(H, img.height)), Image.LANCZOS)
        x = random.randint(0, img.width - W)
        y = random.randint(0, img.height - H)
        proxy = np.asarray(img.crop((x, y, x + W, y + H)), dtype=np.float32) / 255.0
        f = build_features(proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
                           RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])
        feats = torch.from_numpy(f).to('cuda')
        outs = {}
        nets = [('t', teacher)] + ([('s', student)] if want_student else [])
        for tag, net in nets:
            with torch.no_grad():
                head = net.record(feats) if tag == 't' else net(feats)
                head = head.float().cpu().numpy().reshape(g['full_height'],
                                                          g['full_width'], 4)
            outs[tag] = head[:H, :W, :3] / 4.0
        del feats
        torch.cuda.empty_cache()
        return proxy, f, outs['t'], outs.get('s')

    keep = max(1, n_crops * 2 // 3)
    rows = {'逐像素': [], '逐像素+多尺度': []}
    Ys = []
    for k, path in enumerate(picked):
        if k >= keep:
            break
        proxy, f, t, _ = forward(path, False)
        idx = rng.choice(f.shape[0], SUBSAMPLE, replace=False)
        rows['逐像素'].append(f[idx])
        rows['逐像素+多尺度'].append(np.concatenate([f, multiscale(proxy)], axis=1)[idx])
        Ys.append(t.reshape(-1, 3)[idx])
    Y = np.concatenate(Ys)

    models = {}
    for name, X in rows.items():
        X = np.concatenate(X)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - mu) / sd
        A = Xn.T @ Xn + 1e-2 * np.eye(Xn.shape[1])
        Wc = np.linalg.solve(A, Xn.T @ Y)
        models[name] = (mu, sd, Wc)
        print(f'拟合 {name}: {X.shape[1]} 维, {X.shape[0]} 像素')

    acc = {k: [] for k in list(models) + ['学生']}
    labels = []
    for k, path in enumerate(picked):
        if k < keep:
            continue
        proxy, f, t, s = forward(path, True)
        extra = multiscale(proxy)
        for name, (mu, sd, Wc) in models.items():
            X = np.concatenate([f, extra], axis=1) if '多尺度' in name else f
            dt = (((X - mu) / sd) @ Wc).reshape(H, W, 3)
            acc[name].append(band_corr(t, dt))
        acc['学生'].append(band_corr(t, s))
        if not labels:
            labels = [f'{2 ** i}-{2 ** (i + 1)}px' for i in range(len(acc['学生'][0]) - 1)]
            labels.append('残余')

    print(f'\n{"尺度":<10}' + ''.join(f'{n:>14}' for n in acc))
    for i, lab in enumerate(labels):
        print(f'{lab:<10}' + ''.join(f'{np.nanmean([r[i] for r in acc[n]]):14.2f}' for n in acc))
    print(f'\n测试裁剪 {len(acc["学生"])} 张')


if __name__ == '__main__':
    main()
