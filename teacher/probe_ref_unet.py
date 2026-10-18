"""可达上限参照:用普通卷积 UNet 回归教师改动,看各尺度相关最高能到多少。

学生各尺度相关只有 0.30-0.51,而逐像素回归(含多尺度)在粗尺度只能到 0.15-0.22。
本实验要回答:换成"更擅长做多尺度非线性"的结构,同一任务能做到多少?

- 输入与学生完全相同:build_features 产出的 16 条 lanes(320x512)
- 目标:教师的改动 head[...,:3] / 4
- 损失:L1(与学生一致)
- 评估:留出裁剪上按拉普拉斯金字塔算各尺度相关,并与学生同口径对比

    python3 ref_unet.py --prepare 300      # 生成数据(需教师)
    python3 ref_unet.py --train            # 训练 + 评估
"""
import argparse
import glob
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
DATA = '/f/yanr/keep_tmp/ref_unet'


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
    return [float(np.corrcoef(bt.ravel(), bp.ravel())[0, 1])
            for bt, bp in zip(pyramid(dt), pyramid(dp))]


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)

    def forward(self, x):
        x = F.silu(self.n1(self.c1(x)))
        return F.silu(self.n2(self.c2(x)))


class UNet(nn.Module):
    """16 lanes -> 3 通道改动。128 起底、4 级下采样,感受野覆盖上百像素。"""

    def __init__(self, cin=16, base=64):
        super().__init__()
        self.e0 = Block(cin, base)
        self.e1 = Block(base, base * 2)
        self.e2 = Block(base * 2, base * 4)
        self.e3 = Block(base * 4, base * 8)
        self.b = Block(base * 8, base * 8)
        self.d3 = Block(base * 16, base * 4)
        self.d2 = Block(base * 8, base * 2)
        self.d1 = Block(base * 4, base)
        self.d0 = Block(base * 2, base)
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(F.avg_pool2d(e0, 2))
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        b = self.b(F.avg_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(b, size=e3.shape[-2:], mode='nearest'), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode='nearest'), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode='nearest'), e1], 1))
        d0 = self.d0(torch.cat([F.interpolate(d1, size=e0.shape[-2:], mode='nearest'), e0], 1))
        return self.out(d0)


def prepare(n):
    os.makedirs(DATA, exist_ok=True)
    files = []
    for root in ROOTS:
        files += [f for f in glob.glob(root + '/**/*.*', recursive=True)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.seed(20261007)
    picked = random.sample(files, n)
    g = geometry_from_valid(W, H)
    teacher = Network(Model('cuda').load('/home/koth/klss/models/nr'), g)
    xs = np.lib.format.open_memmap(os.path.join(DATA, 'inputs.npy'), mode='w+',
                                   dtype=np.float16, shape=(n, 16, H, W))
    ys = np.lib.format.open_memmap(os.path.join(DATA, 'targets.npy'), mode='w+',
                                   dtype=np.float16, shape=(n, 3, H, W))
    for k, path in enumerate(picked):
        img = Image.open(path).convert('RGB')
        if img.width < W or img.height < H:
            img = img.resize((max(W, img.width), max(H, img.height)), Image.LANCZOS)
        x = random.randint(0, img.width - W)
        y = random.randint(0, img.height - H)
        proxy = np.asarray(img.crop((x, y, x + W, y + H)), dtype=np.float32) / 255.0
        f = build_features(proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
                           RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])
        with torch.no_grad():
            head = teacher.record(torch.from_numpy(f).to('cuda')).float().cpu().numpy()
        head = head.reshape(g['full_height'], g['full_width'], 4)
        xs[k] = f.reshape(H, W, 16).transpose(2, 0, 1).astype(np.float16)
        ys[k] = (head[:H, :W, :3] / 4.0).transpose(2, 0, 1).astype(np.float16)
        if k % 60 == 0:
            print(f'  准备 {k}/{n}', flush=True)
    xs.flush(); ys.flush()
    print(f'完成:{n} 张 -> {DATA}')


def train(steps=1500, batch=2, lr=2e-4):
    xin = np.load(os.path.join(DATA, 'inputs.npy'), mmap_mode='r')
    ytar = np.load(os.path.join(DATA, 'targets.npy'), mmap_mode='r')
    n = len(xin)
    split = max(1, int(n * 0.8))
    print(f'数据 {n} 张,训练 {split} / 测试 {n - split}')

    model = UNet().cuda()
    print(f'UNet 参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(0)
    model.train()
    for step in range(steps):
        idx = rng.integers(0, split, batch)
        x = torch.from_numpy(np.stack([xin[i] for i in idx])).float().cuda()
        y = torch.from_numpy(np.stack([ytar[i] for i in idx])).float().cuda()
        loss = (model(x) - y).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            print(f'  step {step:5d}  L1 {loss.item():.5f}  lr {sched.get_last_lr()[0]:.2e}',
                  flush=True)
    torch.save(model.state_dict(), os.path.join(DATA, 'unet.pt'))

    g = geometry_from_valid(W, H)
    torch.manual_seed(0)
    student = StudentNetwork(load_student_shape(SHAPE), g).to('cuda').eval()
    student.load_state_dict(torch.load(CKPT, map_location='cpu',
                                       weights_only=False).get('ema'))
    model.eval()
    acc = {'UNet': [], '学生': []}
    for i in range(split, n):
        x = torch.from_numpy(np.array(xin[i])).float().unsqueeze(0).cuda()
        y = np.array(ytar[i]).astype(np.float32).transpose(1, 2, 0)
        with torch.no_grad():
            pred = model(x)[0].float().cpu().numpy().transpose(1, 2, 0)
            # memmap 是 fp16,学生权重是 fp32 —— 必须显式转换
            head = student(torch.from_numpy(np.array(xin[i]).transpose(1, 2, 0)
                                            .reshape(-1, 16)).float().cuda())
            head = head.float().cpu().numpy().reshape(g['full_height'],
                                                      g['full_width'], 4)
        acc['UNet'].append(band_corr(y, pred))
        acc['学生'].append(band_corr(y, head[:H, :W, :3] / 4.0))
        del x
        torch.cuda.empty_cache()

    labels = [f'{2 ** i}-{2 ** (i + 1)}px' for i in range(len(acc['UNet'][0]) - 1)]
    labels.append('残余')
    print(f'\n{"尺度":<10}{"UNet":>10}{"学生":>10}')
    for i, lab in enumerate(labels):
        print(f'{lab:<10}{np.nanmean([r[i] for r in acc["UNet"]]):10.2f}'
              f'{np.nanmean([r[i] for r in acc["学生"]]):10.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', type=int, default=0)
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--steps', type=int, default=1500)
    a = ap.parse_args()
    if a.prepare:
        prepare(a.prepare)
    if a.train:
        train(steps=a.steps)
