"""损失权重的 A/B:高频项堆叠是否就是"细尺度好、粗尺度差"的原因?

损失清单(学生当前配置):
  带宽无关    rgb L1 (1.0) + 合成图 L1 (1.0)
  高通        tex 3.0(σ<2.45px 的 L1)+ texstat 2.5(高频幅度)+ detail 0.5(拉普拉斯+Sobel)
  低频        无

本实验用同一 UNet 回归管线,只换损失,看各尺度的相关怎么变:
  l1   纯 L1(带宽无关)              —— 对照
  hf   L1 + 3.0*tex + 2.5*texstat + 0.5*detail   —— 复刻学生的偏置
  lf   L1 + 2.0*低频项               —— 反向偏置

    python3 ref_unet_loss.py --loss hf --steps 4000
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
    def __init__(self, cin=16, base=64, down=4):
        super().__init__()
        ch = [base * (2 ** i) for i in range(down)]
        self.down = down
        self.enc = nn.ModuleList()
        prev = cin
        for c in ch:
            self.enc.append(Block(prev, c)); prev = c
        self.bott = Block(ch[-1], ch[-1])
        self.dec = nn.ModuleList()
        prev = ch[-1]
        for i in range(down - 1, -1, -1):
            self.dec.append(Block(prev + ch[i], ch[i])); prev = ch[i]
        self.out = nn.Conv2d(ch[0], 3, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x); skips.append(x)
            if i != self.down - 1:
                x = F.avg_pool2d(x, 2)
        x = self.bott(F.avg_pool2d(x, 2))
        for j, blk in enumerate(self.dec):
            skip = skips[self.down - 1 - j]
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
            x = blk(torch.cat([x, skip], 1))
        return self.out(x)


def blur(x, times):
    for _ in range(times):
        x = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
    return x


def hf(x):
    return x - blur(x, 9)


def local_amp(x):
    a = hf(x).abs().mean(1, keepdim=True)
    return F.avg_pool2d(F.pad(a, (6, 6, 6, 6), mode='replicate'), 13, stride=1)


def detail_loss(a, b):
    def lap(x):
        out = x
        for _ in range(2):
            hh, ww = out.shape[-2:]
            low = F.avg_pool2d(out, 2, ceil_mode=True)
            out = out - F.interpolate(low, size=(hh, ww), mode='bilinear',
                                      align_corners=False)
        return out
    la, lb = lap(a), lap(b)
    gray_a, gray_b = a.mean(1, keepdim=True), b.mean(1, keepdim=True)
    k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                     dtype=a.dtype, device=a.device).view(1, 1, 3, 3)
    sob = sum((F.conv2d(g, kt, padding=1) - F.conv2d(g2, kt, padding=1)).abs().mean()
              for g, g2 in ((gray_a, gray_b),) for kt in (k, k.transpose(-1, -2)))
    return (la - lb).abs().mean() + sob


def total_loss(pred, target, kind):
    base = (pred - target).abs().mean()
    if kind == 'l1':
        return base
    if kind == 'hf':
        stat = (local_amp(pred) - local_amp(target)).abs().mean()
        return (base + 3.0 * (hf(pred) - hf(target)).abs().mean()
                + 2.5 * stat + 0.5 * detail_loss(pred, target))
    lp = (blur(pred, 9) - blur(target, 9)).abs().mean()
    return base + 2.0 * lp


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
    ap.add_argument('--loss', required=True, choices=('l1', 'hf', 'lf'))
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=2)
    a = ap.parse_args()

    xin = np.load(os.path.join(DATA, 'inputs.npy'), mmap_mode='r')
    ytar = np.load(os.path.join(DATA, 'targets.npy'), mmap_mode='r')
    n = len(xin)
    split = max(1, int(n * 0.8))

    torch.manual_seed(1234)
    model = UNet().cuda()
    print(f'[loss={a.loss}] 参数 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M '
          f'steps {a.steps}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    rng = np.random.default_rng(0)
    model.train()
    for step in range(a.steps):
        idx = rng.integers(0, split, a.batch)
        x = torch.from_numpy(np.stack([xin[i] for i in idx])).float().cuda()
        y = torch.from_numpy(np.stack([ytar[i] for i in idx])).float().cuda()
        loss = total_loss(model(x), y, a.loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0:
            print(f'  step {step:5d}  loss {loss.item():.5f}', flush=True)

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
    print(f'\n[{a.loss}] {"尺度":<10}{"UNet":>9}{"学生":>9}')
    for k, lab in enumerate(labels):
        print(f'{"":<{len(a.loss) + 4}}{lab:<10}'
              f'{np.nanmean([p[0][k] for p in per]):9.2f}'
              f'{np.nanmean([p[1][k] for p in per]):9.2f}')


if __name__ == '__main__':
    main()
