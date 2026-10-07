#!/usr/bin/env python3
"""Distillation training (openspec tasks 5.1-5.9; numerics design.md D6/D11).

在线教师(D7:每 step 现跑,不缓存激活)提供 head 与逐级特征目标;学生在自己
的真实推理数值(bf16 AMP 训练 / f16 推理,无 QAT)下对齐。四类损失(5.3-5.6):
输出 L1(rgb/4 + blend logit SmoothL1 + 合成图)、细节(拉普拉斯 + Sobel 高频)、
特征(逐级 cosine+L2,形状不匹配处过 1x1 投影,投影随训练走)、时序(帧差一致,
序列模式)。

    python3 train_distill.py --data <dir> --size 320 --steps 2000 --overfit 8 -o runs/of1
    python3 train_distill.py --config config.json --resume runs/v1/ckpt.pt

ckpt 含 model/ema/optim/step/config;--resume 全恢复(5.9)。
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time

try:
    import comet_ml                                                        # noqa: F401  在 torch 前导入
    from comet_ml import Experiment
except ImportError:                                                        # pragma: no cover
    Experiment = None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate import TeacherAnnotator                                # noqa: E402
from nr_geometry import geometry_from_valid                          # noqa: E402
from nr_student import StudentNetwork, load_student_shape, student_alignment   # noqa: E402
from run_image import build_features, save_png                       # noqa: E402

DEFAULTS = {
    'shape': os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'shapes', 'student_v0.json'),
    'teacher_weights': os.environ.get('NR_WEIGHTS'),
    'size': 320, 'size_weights': None,
    'batch': 4,
    'steps': 200000,
    'overfit': 0,
    'lr': 2e-4, 'wd': 0.01, 'warmup': 2000, 'ema': 0.999, 'clip': 1.0,
    'loss_out': 1.0, 'loss_detail': 0.5, 'loss_feature': 0.5, 'loss_temporal': 0.0,
    'log_every': 100, 'vis_every': 2000, 'ckpt_every': 5000,
    'seed': 20261007,
    'style_range': [0.0, 3.0], 'tone_range': [0.3, 0.7], 'structure_range': [0.2, 0.8],
    'comet_project': 'dlss-student', 'comet_workspace': None, 'comet_key': None,
}


def comet_key(cfg):
    """key 来源优先级:config > COMET_API_KEY 环境变量 > teacher/.comet_key 文件。"""
    if cfg.get('comet_key'):
        return cfg['comet_key']
    if os.environ.get('COMET_API_KEY'):
        return os.environ['COMET_API_KEY']
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.comet_key')
    if os.path.exists(path):
        with open(path) as fh:
            return fh.read().strip()
    return None


def load_config(args):
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as fh:
            cfg.update(json.load(fh))
    for key in ('shape', 'teacher_weights', 'batch', 'steps', 'overfit', 'lr', 'wd',
                'warmup', 'ema', 'clip', 'loss_out', 'loss_detail', 'loss_feature',
                'loss_temporal', 'seed'):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if getattr(args, 'size', None) is not None:
        cfg['size'] = args.size
    size = cfg['size']
    cfg['size'] = [int(v) for v in str(size).split(',')]      # 支持多尺度:'320' 或 '320,512'
    if getattr(args, 'size_weights', None) is not None:
        cfg['size_weights'] = args.size_weights
    if cfg['size_weights'] is not None:
        cfg['size_weights'] = [float(v) for v in str(cfg['size_weights']).split(',')]
        if len(cfg['size_weights']) != len(cfg['size']):
            raise SystemExit('size_weights length must match --size list')
    return cfg


# ---------------------------------------------------------------- data

class FrameSource:
    """图片目录(递归,可多个根)即样本;overfit 模式固定前 N 个循环。lane 参数按 step 采样(确定性)。"""

    def __init__(self, roots, size, cfg, overfit=0):
        self.size = size
        self.cfg = cfg
        self.overfit = overfit
        self.files = []
        for root in roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames.sort()
                for name in sorted(filenames):
                    if name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        self.files.append(os.path.join(dirpath, name))
        if overfit:
            self.files = self.files[:overfit]
        if not self.files:
            raise SystemExit(f'no images under {roots}')

    def sample(self, step):
        rng = np.random.default_rng([self.cfg['seed'], step])
        idx = step % len(self.files) if self.overfit else int(rng.integers(len(self.files)))
        # 多分辨率采样:size 可为列表(如 [320, 512]),配 size_weights 做加权混合
        # (如 7:3);逐 step 随机取边长 —— 训练分布覆盖多尺度,跨分辨率推理更稳。
        size_cfg = self.size if isinstance(self.size, (list, tuple)) else [self.size]
        weights = self.cfg.get('size_weights')
        if weights:
            p = [w / sum(weights) for w in weights]
            size = int(size_cfg[int(rng.choice(len(size_cfg), p=p))])
        else:
            size = int(size_cfg[int(rng.integers(len(size_cfg)))])
        image = Image.open(self.files[idx]).convert('RGB')
        proxy = np.asarray(image.resize((size, size), Image.LANCZOS),
                           dtype=np.float32) / 255.0
        lane = {
            'style': float(rng.uniform(*self.cfg['style_range'])),
            'tone': float(rng.uniform(*self.cfg['tone_range'])),
            'structure': float(rng.uniform(*self.cfg['structure_range'])),
            'skin': -1.0, 'auto_mask': False,
        }
        noise_seed = int(rng.integers(1 << 30))
        return proxy, noise_seed, lane, self.files[idx]


# ---------------------------------------------------------------- losses

def laplacian_l1(a, b):
    """两级拉普拉斯"场之差"的 L1(5.4:让学生的高频形态对上教师,而不是抹掉高频)。

    注:初版写成 lap(a)+lap(b)(两图高频能量之和),没在比较师生、反而奖励模糊 —— 已修。
    """
    def lap(x):
        fields = []
        cur = x.permute(2, 0, 1).unsqueeze(0)
        for _ in range(2):
            h, w = cur.shape[-2:]
            low = F.avg_pool2d(cur, 2, ceil_mode=True)
            up = F.interpolate(low, size=(h, w), mode='bilinear', align_corners=False)
            fields.append(cur - up)
            cur = low
        return fields
    return sum((fa - fb).abs().mean() for fa, fb in zip(lap(a), lap(b)))


def sobel_l1(a, b):
    """Sobel 边缘场之差的 L1。"""
    def edges(x):
        gray = x.mean(dim=2, keepdim=True).permute(2, 0, 1).unsqueeze(0)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        ky = kx.transpose(-1, -2)
        return F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)
    gxa, gya = edges(a)
    gxb, gyb = edges(b)
    return (gxa - gxb).abs().mean() + (gya - gyb).abs().mean()


def output_losses(s_head, t_head, proxy):
    """输出损失(5.3):rgb/4 L1 + blend logit SmoothL1 + 合成图 L1。"""
    s_rgb, t_rgb = s_head[..., :3] / 4.0, t_head[..., :3] / 4.0
    out = F.l1_loss(s_rgb, t_rgb) + 0.5 * F.smooth_l1_loss(s_head[..., 3], t_head[..., 3])
    s_neural = torch.clamp(proxy + s_rgb, 0, 1)
    t_neural = torch.clamp(proxy + t_rgb, 0, 1)
    out = out + F.l1_loss(s_neural, t_neural)
    return out, s_neural, t_neural


class FeatureAligner(nn.Module):
    """形状不匹配的捕获点(d4/ViT)过 1x1 投影:teacher 维 -> student 维,随训练走(5.5)。"""

    def __init__(self, pairs):
        super().__init__()
        self.proj = nn.ModuleDict({name: nn.Linear(t_dim, s_dim, bias=False)
                                   for name, (s_dim, t_dim) in pairs.items()})

    def forward(self, name, teacher_act):
        key = name.replace('-', '_')
        if key in self.proj:
            return self.proj[key](teacher_act.float())
        return teacher_act.float()


def feature_losses(student_caps, teacher_caps, aligner, weights):
    """逐级 (1 - cosine) + 0.1* MSE,按点取加权均值(5.5)。

    注意:MSE 不做逐点方差归一化 —— 方差小的层会把比值炸成 1e4+,首次训练就是这样 NaN 的。
    """
    total = 0.0
    wsum = 0.0
    detail = {}
    for s_name, t_name in student_alignment().items():
        if s_name == 's-head':
            continue
        s = student_caps[s_name].float()
        t_act = torch.as_tensor(teacher_caps[t_name], device=s.device)
        t = aligner(s_name, t_act)
        cos = F.cosine_similarity(s, t, dim=-1).mean()
        mse = (s - t).pow(2).mean()
        value = (1.0 - cos) + 0.1 * mse
        w = weights.get(s_name, 1.0)
        total = total + w * value
        wsum += w
        detail[s_name] = float(value.detach())
    return total / max(wsum, 1e-6), detail


# ---------------------------------------------------------------- training

def lr_at(step, cfg):
    if step < cfg['warmup']:
        return cfg['lr'] * (step + 1) / cfg['warmup']
    span = max(1, cfg['steps'] - cfg['warmup'])
    t = (step - cfg['warmup']) / span
    return cfg['lr'] * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def main():
    parser = argparse.ArgumentParser(description='distillation training')
    parser.add_argument('--config', help='JSON config (CLI flags override)')
    parser.add_argument('--data', required=True, nargs='+',
                        help='image root(s), recursive')
    parser.add_argument('-o', '--out', required=True, help='run directory')
    parser.add_argument('--resume', help='checkpoint .pt')
    parser.add_argument('--shape')
    parser.add_argument('--teacher-weights', dest='teacher_weights')
    parser.add_argument('--size', type=str,
                        help='train edge length(s): "320" or multi-scale "320,512"')
    parser.add_argument('--size-weights', dest='size_weights', type=str,
                        help='per-size sampling weights, e.g. "0.7,0.3" for 7:3 mix')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--overfit', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--ema', type=float)
    parser.add_argument('--clip', type=float)
    parser.add_argument('--loss-out', dest='loss_out', type=float)
    parser.add_argument('--loss-detail', dest='loss_detail', type=float)
    parser.add_argument('--loss-feature', dest='loss_feature', type=float)
    parser.add_argument('--loss-temporal', dest='loss_temporal', type=float)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    cfg = load_config(args)
    if not cfg['teacher_weights']:
        raise SystemExit('no teacher weights: --teacher-weights or NR_WEIGHTS')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'config.json'), 'w') as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)

    experiment = None
    key = comet_key(cfg)
    if key and Experiment is not None:
        experiment = Experiment(api_key=key, project_name=cfg['comet_project'],
                                workspace=cfg.get('comet_workspace') or None,
                                log_code=False)
        experiment.log_parameters({k: v for k, v in cfg.items() if k != 'comet_key'})
        print('comet:', experiment.url)
    else:
        print('comet: disabled (no comet_ml package or no key)')

    sizes = cfg['size']
    g = geometry_from_valid(sizes[0], sizes[0])
    shape = load_student_shape(cfg['shape'])
    geo_cache = {}

    def geo_for(size):
        """每种训练边长一套 geometry + 在线教师(多尺度采样;教师模型按需懒加载)。"""
        found = geo_cache.get(size)
        if found is None:
            g_s = geometry_from_valid(size, size)
            found = (g_s, TeacherAnnotator(cfg['teacher_weights'], g_s))
            geo_cache[size] = found
        return found

    model = StudentNetwork(shape, g).to(device)
    model.capture_enabled = True

    # 形状不匹配的特征对(d4/ViT) -> 1x1 投影表。
    align_pairs = {}
    s_dims = {'s-enc-full': 32, 's-enc-d0': 32, 's-enc-d1': 64, 's-enc-d2': 128,
              's-enc-d3': 256, 's-enc-d4': shape['levels'][5]['channels'],
              's-vit': shape['vit']['channels'],
              's-dec-d4': shape['levels'][5]['channels'], 's-dec-d3': 256,
              's-dec-d2': 128, 's-dec-d1': 64, 's-dec-d0': 32}
    t_dims = {'s-enc-full': 32, 's-enc-d0': 32, 's-enc-d1': 64, 's-enc-d2': 128,
              's-enc-d3': 256, 's-enc-d4': 512, 's-vit': 1024,
              's-dec-d4': 512, 's-dec-d3': 256, 's-dec-d2': 128, 's-dec-d1': 64,
              's-dec-d0': 32}
    for s_name in student_alignment():
        if s_name == 's-head':
            continue
        if s_dims[s_name] != t_dims[s_name]:
            align_pairs[s_name.replace('-', '_')] = (s_dims[s_name], t_dims[s_name])
    aligner = FeatureAligner(align_pairs).to(device)

    params = list(model.parameters()) + list(aligner.parameters())
    optim = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'])
    ema = {k: p.detach().clone() for k, p in model.state_dict().items()}

    start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        aligner.load_state_dict(ckpt['aligner'])
        optim.load_state_dict(ckpt['optim'])
        ema = ckpt['ema']
        start = ckpt['step'] + 1
        print(f"resumed at step {start} from {args.resume}")

    source = FrameSource(args.data, sizes, cfg, overfit=cfg['overfit'])
    teacher_caps = tuple(t for t in student_alignment().values() if t != 'head')
    feature_weights = {name: 1.0 for name in student_alignment()}

    log_path = os.path.join(args.out, 'log.csv')
    new_log = not os.path.exists(log_path)
    log = open(log_path, 'a', newline='')
    writer = csv.writer(log)
    if new_log:
        writer.writerow(['step', 'lr', 'loss', 'out', 'detail', 'feature', 'psnr_teacher',
                         'ms'])

    print(f"train: {len(source.files)} files, {sizes}², {cfg['batch']} batch, "
          f"{cfg['steps']} steps, device {device}")
    model.train()
    for step in range(start, cfg['steps']):
        lr = lr_at(step, cfg)
        for group in optim.param_groups:
            group['lr'] = lr
        t0 = time.perf_counter()
        totals = {'out': 0.0, 'detail': 0.0, 'feature': 0.0}
        feat_detail = {}
        psnrs = []
        optim.zero_grad()
        for _ in range(cfg['batch']):
            proxy, noise_seed, lane, _ = source.sample(step * cfg['batch'] + len(psnrs))
            s = proxy.shape[0]
            g_s, teacher_s = geo_for(s)
            model.g = g_s                              # 学生算子与分辨率无关,换几何即换边长
            features_np = build_features(proxy, g_s, noise_seed, lane['style'], lane['tone'],
                                         lane['structure'], lane['skin'], lane['auto_mask'])
            with torch.no_grad():
                t_res = teacher_s.annotate(features_np, capture=teacher_caps)
            # head 是场分辨率(如 512² -> 场 576x512),裁到有效区再进损失。
            t_head = torch.from_numpy(t_res['head']).view(
                g_s['full_height'], g_s['full_width'], 4)[:s, :s].to(device)
            proxy_t = torch.from_numpy(proxy).to(device)

            with torch.autocast(device, dtype=torch.bfloat16):
                s_head = model(torch.from_numpy(features_np).to(device))
            s_head = s_head.view(g_s['full_height'], g_s['full_width'], 4)[:s, :s].float()
            out, s_neural, t_neural = output_losses(s_head, t_head, proxy_t)
            detail = laplacian_l1(s_neural, t_neural) + sobel_l1(s_neural, t_neural)
            feat, feat_detail = feature_losses(model.captures, t_res, aligner, feature_weights)
            loss = (cfg['loss_out'] * out + cfg['loss_detail'] * detail
                    + cfg['loss_feature'] * feat)
            if not torch.isfinite(loss):
                print(f'step {step}: non-finite sample loss, skipped')
                continue
            (loss / cfg['batch']).backward()
            totals['out'] += float(out.detach())
            totals['detail'] += float(detail.detach())
            totals['feature'] += float(feat.detach())
            psnrs.append(psnr(s_neural.detach(), t_neural))
        torch.nn.utils.clip_grad_norm_(params, cfg['clip'])
        optim.step()
        if cfg['ema'] > 0:
            # EMA 暖起:固定 decay 在短跑里会被早期半成品权重污染(3K 步 × 0.999 视界
            # 下 32% 质量来自 step 1000-2000),按步数渐升到 cfg['ema'](run v1 长跑自然退化为常数)。
            decay = min(cfg['ema'], (1.0 + step) / (100.0 + step))
            with torch.no_grad():
                for k, p in model.state_dict().items():
                    if p.dtype.is_floating_point:
                        ema[k].mul_(decay).add_(p.detach(), alpha=1 - decay)
        ms = (time.perf_counter() - t0) * 1000

        if experiment is not None:
            n = max(1, len(psnrs))
            experiment.log_metrics({
                'loss/total': (totals['out'] + totals['detail'] + totals['feature']) / n,
                'loss/out': totals['out'] / n,
                'loss/detail': totals['detail'] / n,
                'loss/feature': totals['feature'] / n,
                'psnr/teacher': sum(psnrs) / n,
                'train/lr': lr,
                'train/ms': ms,
            }, step=step)
            if feat_detail:
                experiment.log_metrics({f'feat/{k}': v for k, v in feat_detail.items()}, step=step)

        if step % cfg['log_every'] == 0 or step == cfg['steps'] - 1:
            n = cfg['batch']
            pn = max(1, len(psnrs))
            row = [step, f'{lr:.2e}', f"{(totals['out'] + totals['detail'] + totals['feature']) / n:.5f}",
                   f"{totals['out'] / n:.5f}", f"{totals['detail'] / n:.5f}",
                   f"{totals['feature'] / n:.5f}", f'{sum(psnrs) / pn:.2f}',
                   f'{ms:.0f}']
            writer.writerow(row)
            log.flush()
            print(f"step {step:6d}  lr {lr:.2e}  out {totals['out'] / n:.5f}  "
                  f"detail {totals['detail'] / n:.5f}  feat {totals['feature'] / n:.5f}  "
                  f"psnr(t) {sum(psnrs) / len(psnrs):.2f}  {ms:.0f} ms")

        if cfg['vis_every'] and step % cfg['vis_every'] == 0 and step > 0:
            vis = os.path.join(args.out, f'vis-{step:06d}')
            os.makedirs(vis, exist_ok=True)
            save_png(os.path.join(vis, 'proxy.png'), proxy)
            save_png(os.path.join(vis, 'teacher.png'), t_neural.detach().cpu().numpy())
            save_png(os.path.join(vis, 'student.png'), s_neural.detach().cpu().numpy())
            diff = np.abs(s_neural.detach().cpu().numpy()
                          - t_neural.detach().cpu().numpy()) * 12
            save_png(os.path.join(vis, 'diff_x12.png'), diff)
            if experiment is not None:
                experiment.log_image(proxy, 'proxy', step=step)
                experiment.log_image(t_neural.detach().cpu().numpy(), 'teacher', step=step)
                experiment.log_image(s_neural.detach().cpu().numpy(), 'student', step=step)
                experiment.log_image(np.clip(diff, 0, 1), 'diff_x12', step=step)

        if cfg['ckpt_every'] and step % cfg['ckpt_every'] == 0 and step > 0:
            torch.save({'step': step, 'model': model.state_dict(), 'aligner': aligner.state_dict(),
                        'optim': optim.state_dict(), 'ema': ema, 'config': cfg},
                       os.path.join(args.out, 'ckpt.pt'))
    torch.save({'step': cfg['steps'] - 1, 'model': model.state_dict(),
                'aligner': aligner.state_dict(), 'optim': optim.state_dict(), 'ema': ema,
                'config': cfg}, os.path.join(args.out, 'ckpt.pt'))
    log.close()
    if experiment is not None:
        experiment.end()
    print('done ->', os.path.join(args.out, 'ckpt.pt'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
