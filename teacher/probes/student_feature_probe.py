"""学生内部探针:粗尺度信息到底在不在深层特征里?

若在,那么一个作用在深层特征上的线性读出就能预测教师的低频改动 —— 说明瓶颈在
解码器/头部,改动便宜;若不在,则深层特征本身缺这个信息,只能加深/加宽。

做法:跑训练好的学生并抓取各层特征(rows, C),对每个抓取点:
  - 目标:教师改动 dt 按该层的网格平均下采样(即该尺度的低频内容)
  - 在训练裁剪上用岭回归拟合 特征 -> 目标,在留出裁剪上算相关
  - 同时给出学生自己输出在同一网格上的相关(作为对照,等价于"解码器实际做到了多少")

    python3 student_feature_probe.py [裁剪数]
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
POINTS = ('s-enc-d4', 's-dec-d4', 's-dec-d3', 's-dec-d2')


def pool_to(a, gw, gh):
    """(H, W, C) -> (gh, gw, C) 的均值下采样(严格整除)。"""
    h, w, c = a.shape
    a = a[:gh * (h // gh), :gw * (w // gw)]
    return a.reshape(gh, h // gh, gw, w // gw, c).mean((1, 3))


def main():
    n_crops = int(sys.argv[1]) if len(sys.argv) > 1 else 60
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
    student.capture_enabled = True
    grids = {name: (lv['width'], lv['height'])
             for name, lv in ((f'd{i}', g['levels'][i]) for i in range(6))}

    rows = []
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
            _ = student(feats)
            head = teacher.record(feats).float().cpu().numpy().reshape(
                g['full_height'], g['full_width'], 4)
        dt = head[:H, :W, :3] / 4.0
        s = student.captures['s-head'].float().cpu().numpy().reshape(
            g['full_height'], g['full_width'], 4)[:H, :W, :3] / 4.0
        item = {'dt': dt, 'ds': s}
        for name in POINTS:
            level = name.split('-')[-1]
            gw, gh = grids[level]
            item[name] = student.captures[name].float().cpu().numpy().reshape(
                gh, gw, -1)
        rows.append(item)
        del feats
        torch.cuda.empty_cache()

    split = max(1, int(len(rows) * 0.7))
    print(f'裁剪 {len(rows)}:训练 {split} / 测试 {len(rows) - split}')
    print(f'{"抓取点":<12}{"网格":>10}{"线性读出":>10}{"学生输出":>10}')

    for name in POINTS:
        level = name.split('-')[-1]
        gw, gh = grids[level]

        def prep(items):
            X = np.stack([it[name].reshape(-1, it[name].shape[-1]) for it in items])
            yt = np.stack([pool_to(it['dt'], gw, gh).reshape(-1, 3) for it in items])
            ys = np.stack([pool_to(it['ds'], gw, gh).reshape(-1, 3) for it in items])
            return X, yt, ys

        Xtr, Ytr, _ = prep(rows[:split])
        Xte, Yte, Yse = prep(rows[split:])
        # 以"每张裁剪"为单位的特征标准化(通道维)
        mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
        sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
        A, B = [], []
        for x_, y_ in zip(Xtr, Ytr):
            A.append(((x_ - mu) / sd).reshape(-1, Xtr.shape[-1]))
            B.append(y_.reshape(-1, 3))
        A, B = np.concatenate(A), np.concatenate(B)
        Wc = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ B)

        acc_l, acc_s = [], []
        for x_, y_, ys_ in zip(Xte, Yte, Yse):
            pred = (((x_ - mu) / sd).reshape(-1, Xtr.shape[-1]) @ Wc).reshape(-1)
            acc_l.append(np.corrcoef(pred, y_.reshape(-1))[0, 1])
            acc_s.append(np.corrcoef(ys_.reshape(-1), y_.reshape(-1))[0, 1])
        print(f'{name:<12}{gw}x{gh:<7}{np.mean(acc_l):10.2f}{np.mean(acc_s):10.2f}')


if __name__ == '__main__':
    main()
