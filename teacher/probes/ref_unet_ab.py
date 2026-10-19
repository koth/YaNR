"""感受野 A/B:用回归管线快速比较"再加一级下采样"与"空洞卷积"。

背景:ViT 消融显示中尺度(8-64px)结构来自卷积编码器而非 ViT,而普通 UNet
在同一任务上粗尺度能到 0.47-0.55、学生只有 0.42-0.44。本实验比较两种扩大
卷积感受野的方案,同样数据、同样步数、同样种子:

  base    down=4,瓶颈在 /16(dilation 1)      —— 对照
  deep    down=5,瓶颈在 /32                  —— 多一级下采样
  dilate  down=4,瓶颈 dilation=2             —— 空洞卷积扩感受野

评估同前:留出裁剪上按拉普拉斯金字塔算各尺度相关。

    python3 ref_unet_ab.py --variant deep --steps 6000
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/koth/klss/teacher')
from nr_geometry import geometry_from_valid                  # noqa: E402
from nr_student import StudentNetwork, load_student_shape     # noqa: E402

W, H = 320, 512
DATA = '/f/yanr/keep_tmp/ref_unet'
CKPT = '/home/koth/keep_tmp/v45d_56k/ckpt.pt'
SHAPE = '/home/koth/klss/shapes/student_shared_moe_v1.json'


class Block(nn.Module):
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)

    def forward(self, x):
        x = F.silu(self.n1(self.c1(x)))
        return F.silu(self.n2(self.c2(x)))


class UNet(nn.Module):
    """可配置下采样级数与瓶颈膨胀率。up() 保证尺寸对齐(奇偶都安全)。"""

    def __init__(self, cin=16, base=64, down=4, dilate=1):
        super().__init__()
        ch = [base * (2 ** i) for i in range(down)]
        self.down = down
        self.enc = nn.ModuleList()
        prev = cin
        for c in ch:
            self.enc.append(Block(prev, c, dilation=1))
            prev = c
        self.bott = Block(ch[-1], ch[-1], dilation=dilate)
        self.dec = nn.ModuleList()
        prev = ch[-1]
        for i in range(down - 1, -1, -1):
            self.dec.append(Block(prev + ch[i], ch[i]))
            prev = ch[i]
        self.out = nn.Conv2d(ch[0], 3, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            skips.append(x)
            if i != self.down - 1:
                x = F.avg_pool2d(x, 2)
        x = self.bott(F.avg_pool2d(x, 2))
        for j, blk in enumerate(self.dec):
            skip = skips[self.down - 1 - j]
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
            x = blk(torch.cat([x, skip], 1))
        return self.out(x)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=('base', 'deep', 'dilate'))
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-4)
    a = ap.parse_args()

    xin = np.load(os.path.join(DATA, 'inputs.npy'), mmap_mode='r')
    ytar = np.load(os.path.join(DATA, 'targets.npy'), mmap_mode='r')
    n = len(xin)
    split = max(1, int(n * 0.8))

    cfg = {'base': dict(down=4, dilate=1), 'deep': dict(down=5, dilate=1),
           'dilate': dict(down=4, dilate=2)}[a.variant]
    torch.manual_seed(1234)
    model = UNet(**cfg).cuda()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'[{a.variant}] down={cfg["down"]} dilate={cfg["dilate"]} '
          f'参数 {params:.1f}M  steps {a.steps}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    rng = np.random.default_rng(0)
    model.train()
    for step in range(a.steps):
        idx = rng.integers(0, split, a.batch)
        x = torch.from_numpy(np.stack([xin[i] for i in idx])).float().cuda()
        y = torch.from_numpy(np.stack([ytar[i] for i in idx])).float().cuda()
        loss = (model(x) - y).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0:
            print(f'  step {step:5d}  L1 {loss.item():.5f}', flush=True)

    g = geometry_from_valid(W, H)
    torch.manual_seed(0)
    student = StudentNetwork(load_student_shape(SHAPE), g).to('cuda').eval()
    student.load_state_dict(torch.load(CKPT, map_location='cpu',
                                       weights_only=False).get('ema'))
    model.eval()
    per = []
    for i in range(split, n):
        x = torch.from_numpy(np.array(xin[i])).float().unsqueeze(0).cuda()
        y = np.array(ytar[i]).astype(np.float32).transpose(1, 2, 0)
        with torch.no_grad():
            pred = model(x)[0].float().cpu().numpy().transpose(1, 2, 0)
            head = student(torch.from_numpy(np.array(xin[i]).transpose(1, 2, 0)
                                            .reshape(-1, 16)).float().cuda())
        head = head.float().cpu().numpy().reshape(g['full_height'], g['full_width'], 4)
        per.append((band_corr(y, pred), band_corr(y, head[:H, :W, :3] / 4.0)))
        del x
        torch.cuda.empty_cache()

    labels = [f'{2 ** i}-{2 ** (i + 1)}px' for i in range(len(per[0][0]) - 1)] + ['残余']
    print(f'\n[{a.variant}] {"尺度":<10}{"UNet":>9}{"学生":>9}')
    for k, lab in enumerate(labels):
        u = np.nanmean([p[0][k] for p in per])
        s = np.nanmean([p[1][k] for p in per])
        print(f'{"":<{len(a.variant) + 4}}{lab:<10}{u:9.2f}{s:9.2f}')


if __name__ == '__main__':
    main()
