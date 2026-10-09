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
import threading
import time
from queue import Queue

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
from degrade import degrade_frame, sample_params as degrade_params   # noqa: E402
from nr_geometry import geometry_from_valid                          # noqa: E402
from nr_history import block_match_flow, reproject_history, store_history  # noqa: E402
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
    'loss_out': 1.0, 'loss_detail': 0.5, 'loss_feature': 0.5, 'loss_temporal': 1.0,
    'pair_fraction': 0.5,        # 时序对采样比例(历史腿:上帧教师输出 -> truncate_half -> 重投影)
    'crop_prob': 0.5,            # 随机原生裁剪(尺度抖动 1-2x);其余整图 resize
    'degrade_prob': 0.5,         # 在线退化(degrade.py 链;proxy_proc 已预退化的跳过)
    'moe_aux': 0.02,             # MoE 均衡正则(配额分配已结构保证均衡;此项仅防 p 漂移)
    'moe_z': 1e-2,               # router z-loss(ST-MoE,压 logits 幅值;router 前 LN 后配合生效)
    'log_every': 100, 'vis_every': 2000, 'ckpt_every': 5000,
    'seed': 20261007,
    'style_range': [0.0, 127.0], 'tone_range': [0.0, 1.0], 'structure_range': [0.0, 1.0],
    # v1.2 条件全覆盖(2026-10-10):style 是 ID(lane = ID/128,真实量程 [0,1));
    # v1.1 只采 [0,3] -> lane ≤0.023,导致高档失控(见 eval_conditions 探针)。
    'automask_prob': 0.3,        # automask 开时 skin ~ U[0,1](lane 13/14 分支),关时 skin=-1
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
                'loss_temporal', 'pair_fraction', 'seed'):
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
        self.file_root = []
        for root in roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames.sort()
                for name in sorted(filenames):
                    if name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        self.files.append(os.path.join(dirpath, name))
                        self.file_root.append(root)
        if overfit:
            self.files = self.files[:overfit]
            self.file_root = self.file_root[:overfit]
        if not self.files:
            raise SystemExit(f'no images under {roots}')
        # 序列分组(真实相邻帧对的原料):2..256 张图的目录 = 一个序列
        # (DAVIS <seq>/、proxy_proc seq-*/ 算;DIV2K/Flickr2K 平铺 800+ 张的目录不算)。
        by_dir = {}
        for i, f in enumerate(self.files):
            by_dir.setdefault(os.path.dirname(f), []).append(i)
        self.seq_groups = [idxs for _, idxs in sorted(by_dir.items()) if 2 <= len(idxs) <= 256]
        if not self.seq_groups:
            print('warn: no sequence groups found; pair batches will sample single images')

    def _draw_crop(self, rng, size, dims):
        """按 crop_prob 抽一个原生裁剪框(尺度抖动 1-2x);返回 (x,y,side) 或 None=整图 resize。"""
        w0, h0 = dims
        if rng.random() < self.cfg.get('crop_prob', 0.0) and min(w0, h0) >= size:
            side = min(int(size * float(rng.uniform(1.0, 2.0))), min(w0, h0))
            return (int(rng.integers(0, w0 - side + 1)), int(rng.integers(0, h0 - side + 1)), side)
        return None

    def _load_frame(self, path, size, rng, crop_box='draw', deg=None):
        """一帧 proxy:裁剪 + 在线退化(degrade.py 链)。

        预退化语料(proxy_proc)跳过在线退化;
        crop_box:'draw'=按 crop_prob 自抽;tuple=共用框(帧对);None=强制整图 resize;
        deg 三态:False=不做退化;dict=用这组参数(帧对语义:逐序列抽参、逐帧抽噪声);
        None=按 degrade_prob 自抽。
        """
        image = Image.open(path).convert('RGB')
        if crop_box == 'draw':
            crop_box = self._draw_crop(rng, size, image.size)
        if crop_box is not None:
            x, y, side = crop_box
            image = image.crop((x, y, x + side, y + side))
        proxy = np.asarray(image.resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.0
        predegraded = 'proxy_proc' in path
        if not predegraded and 'degrade_prob' in self.cfg:
            if deg is False:
                pass
            elif deg is not None:
                proxy = degrade_frame(proxy, None, deg, rng)
            elif rng.random() < self.cfg['degrade_prob']:
                proxy = degrade_frame(proxy, None, degrade_params(rng, 'random'), rng)
        return proxy, crop_box

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
        proxy, _ = self._load_frame(self.files[idx], size, rng)
        auto_mask = bool(rng.random() < self.cfg.get('automask_prob', 0.0))
        lane = {
            'style': float(rng.uniform(*self.cfg['style_range'])),
            'tone': float(rng.uniform(*self.cfg['tone_range'])),
            'structure': float(rng.uniform(*self.cfg['structure_range'])),
            'skin': float(rng.uniform(0.0, 1.0)) if auto_mask else -1.0,
            'auto_mask': auto_mask,
        }
        noise_seed = int(rng.integers(1 << 30))
        return proxy, noise_seed, lane, self.files[idx]

    def sample_pair(self, step):
        """时序对(历史腿):**真实相邻帧** + 块匹配运动(输出像素系 -> UV 归一化)。

        2026-10-10:flicker 修复 —— 合成视角对(同图两裁剪±平移)的统计迁不到真实视频,
        改为序列里 t/t+1 真实帧对,同 eval_temporal 口径;两帧共用裁剪框与退化参数
        (逐序列抽参,逐帧抽噪声),motion 由块匹配估计(归一化 UV,喂 reproject_history)。
        """
        rng = np.random.default_rng([self.cfg['seed'], step])
        size_cfg = self.size if isinstance(self.size, (list, tuple)) else [self.size]
        weights = self.cfg.get('size_weights')
        if weights:
            p = [w / sum(weights) for w in weights]
            size = int(size_cfg[int(rng.choice(len(size_cfg), p=p))])
        else:
            size = int(size_cfg[int(rng.integers(len(size_cfg)))])
        if self.seq_groups:
            group = self.seq_groups[int(rng.integers(len(self.seq_groups)))]
            i = int(rng.integers(len(group) - 1))
            path_prev, path_cur = self.files[group[i]], self.files[group[i + 1]]
        else:                                                   # 退化兜底:无序列时单图自对
            path_prev = path_cur = self.files[int(rng.integers(len(self.files)))]
        crop_box = self._draw_crop(rng, size, Image.open(path_prev).size)   # 两帧共用
        deg = False
        if 'degrade_prob' in self.cfg and rng.random() < self.cfg['degrade_prob']:
            deg = degrade_params(rng, 'random')                # 逐序列抽参,逐帧抽噪声
        prev_p, _ = self._load_frame(path_prev, size, rng, crop_box=crop_box, deg=deg)
        cur_p, _ = self._load_frame(path_cur, size, rng, crop_box=crop_box, deg=deg)
        h, w = cur_p.shape[:2]
        if path_prev == path_cur:
            motion = np.zeros((h, w, 2), np.float32)            # 静态兜底
        else:
            motion = block_match_flow(prev_p, cur_p).astype(np.float32)
            motion[..., 0] /= w                                  # 像素 -> UV(nr_history 口径)
            motion[..., 1] /= h
        auto_mask = bool(rng.random() < self.cfg.get('automask_prob', 0.0))
        lane = {
            'style': float(rng.uniform(*self.cfg['style_range'])),
            'tone': float(rng.uniform(*self.cfg['tone_range'])),
            'structure': float(rng.uniform(*self.cfg['structure_range'])),
            'skin': float(rng.uniform(0.0, 1.0)) if auto_mask else -1.0,
            'auto_mask': auto_mask,
        }
        return prev_p, cur_p, motion, lane, path_cur


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


def output_losses(s_head, t_head, proxy, repro=None, s_scale=1.0, t_scale=1.0):
    """输出损失(5.3):rgb/4 L1 + blend logit SmoothL1 + 合成图 L1。

    有历史(repro 非空)时合成走 temporal blend(run_image 语义:权重
    clamp(sigmoid(logit)*blend_scale)),blend logit 的监督此时才有意义;
    返回的 neural 是部署口径合成图(时序损失吃它)。
    """
    s_rgb, t_rgb = s_head[..., :3] / 4.0, t_head[..., :3] / 4.0
    out = F.l1_loss(s_rgb, t_rgb) + 0.5 * F.smooth_l1_loss(s_head[..., 3], t_head[..., 3])
    s_neural = torch.clamp(proxy + s_rgb, 0, 1)
    t_neural = torch.clamp(proxy + t_rgb, 0, 1)
    out = out + F.l1_loss(s_neural, t_neural)
    if repro is not None:
        s_w = torch.clamp(torch.sigmoid(s_head[..., 3]) * s_scale, 0, 1)
        t_w = torch.clamp(torch.sigmoid(t_head[..., 3]) * t_scale, 0, 1)
        s_neural = s_neural + (repro - s_neural) * s_w[..., None]
        t_neural = t_neural + (repro - t_neural) * t_w[..., None]
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


def moe_route_metrics(model):
    """路由健康度(逐块聚合):util_worst = 最偏块的最大专家占比(1.0=整块塌缩)、
    util_max = 各块最大占比均值、dead_frac = 死专家(<1% 利用)占比、entropy = 平均
    归一化利用熵(1=均匀)。取的是当前 moe_stats(主前向、时序二次前向之前)。"""
    stats = [s for m in model.modules() if hasattr(m, 'moe_stats') for s in m.moe_stats]
    if not stats:
        return None
    umax, dead, ent = [], [], []
    for assign, probs, _logits in stats:
        e_count = probs.shape[-1]
        f = torch.bincount(assign, minlength=e_count).float() / assign.numel()
        umax.append(float(f.max()))
        dead.append(float((f < 0.01).float().mean()))
        ent.append(float(-(f * (f + 1e-9).log()).sum() / math.log(e_count)))
    return {'util_max': sum(umax) / len(umax), 'util_worst': max(umax),
            'dead_frac': sum(dead) / len(dead), 'entropy': sum(ent) / len(ent)}


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
    parser.add_argument('--pair-fraction', dest='pair_fraction', type=float,
                        help='时序对采样比例(0 关闭历史腿)')
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

    # 形状不匹配的特征对(d4/ViT) -> 1x1 投影表。s 维从形状读 —— 曾经硬编码 v0,
    # 非 v0 形状(如 slim 系列 d3=128)会漏建投影,教师维度原样穿透直接崩。
    lv = shape['levels']
    s_dims = {'s-enc-full': lv[0]['channels'], 's-enc-d0': lv[1]['channels'],
              's-enc-d1': lv[2]['channels'], 's-enc-d2': lv[3]['channels'],
              's-enc-d3': lv[4]['channels'], 's-enc-d4': lv[5]['channels'],
              's-vit': shape['vit']['channels'],
              's-dec-d4': lv[5]['channels'], 's-dec-d3': lv[4]['channels'],
              's-dec-d2': lv[3]['channels'], 's-dec-d1': lv[2]['channels'],
              's-dec-d0': lv[1]['channels']}
    align_pairs = {}
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
        writer.writerow(['step', 'lr', 'loss', 'out', 'detail', 'feature', 'temporal',
                         'psnr_teacher', 'ms'])

    print(f"train: {len(source.files)} files, {sizes}², {cfg['batch']} batch, "
          f"{cfg['steps']} steps, device {device}")
    model.train()
    def prep_item(step, bi):
        """纯 CPU 数据准备(预取线程):采样/裁剪/退化/块匹配/特征构造。

        逐 (step,bi) 独立种子,与执行顺序无关 -> 结果确定;不碰模型/教师状态(零竞态)。
        帧对的 feats_b 依赖教师上帧输出,留在主循环。
        """
        item_pair = (cfg['pair_fraction'] > 0 and
                     np.random.default_rng([cfg['seed'], step, 7]).random() < cfg['pair_fraction'])
        if item_pair:
            prev_p, cur_p, motion, lane, _ = source.sample_pair(step * cfg['batch'] + bi)
            seed_a = int(np.random.default_rng([cfg['seed'], step, bi, 1]).integers(1 << 30))
            noise_seed = int(np.random.default_rng([cfg['seed'], step, bi, 2]).integers(1 << 30))
            g_s = geometry_from_valid(cur_p.shape[0], cur_p.shape[1])
            feats_a = build_features(prev_p, g_s, seed_a, lane['style'], lane['tone'],
                                     lane['structure'], lane['skin'], lane['auto_mask'])
            return ('pair', prev_p, cur_p, motion, lane, seed_a, noise_seed, feats_a)
        proxy, noise_seed, lane, _ = source.sample(step * cfg['batch'] + bi)
        g_s = geometry_from_valid(proxy.shape[0], proxy.shape[1])
        features_np = build_features(proxy, g_s, noise_seed, lane['style'], lane['tone'],
                                     lane['structure'], lane['skin'], lane['auto_mask'])
        return ('single', proxy, None, None, lane, None, noise_seed, features_np)

    prep_q = Queue(maxsize=3)

    def _prep_worker():
        try:
            for step_w in range(start, cfg['steps']):
                for bi_w in range(cfg['batch']):
                    prep_q.put(prep_item(step_w, bi_w))
        except Exception as exc:                                # noqa: BLE001
            print(f'prep worker died: {exc!r}')
    threading.Thread(target=_prep_worker, daemon=True).start()

    for step in range(start, cfg['steps']):
        lr = lr_at(step, cfg)
        for group in optim.param_groups:
            group['lr'] = lr
        t0 = time.perf_counter()
        totals = {'out': 0.0, 'detail': 0.0, 'feature': 0.0, 'temporal': 0.0}
        moe_tot = {}
        feat_detail = {}
        psnrs = []
        optim.zero_grad()
        pair_mode = (cfg['pair_fraction'] > 0 and
                     np.random.default_rng([cfg['seed'], step, 7]).random() < cfg['pair_fraction'])
        for bi in range(cfg['batch']):
            item = prep_q.get()                                 # 预取线程已备好纯 CPU 部分
            if item[0] == 'pair':
                # 历史腿:上帧教师输出(teacher-forcing)-> truncate_half 存史 ->
                # 已知运动重投影 -> 当前帧带真实历史 lanes(与部署管线同口径)。
                _, prev_p, cur_p, motion, lane, seed_a, noise_seed, feats_a = item
                s = cur_p.shape[0]
                g_s, teacher_s = geo_for(s)
                model.g = g_s
                with torch.no_grad():
                    t_res_a = teacher_s.annotate(feats_a)
                    t_head_a = torch.from_numpy(t_res_a['head']).view(
                        g_s['full_height'], g_s['full_width'], 4)[:s, :s].to(device)
                    prev_t = torch.from_numpy(prev_p).to(device)
                    t_neural_a = torch.clamp(prev_t + t_head_a[..., :3] / 4, 0, 1)
                    hist = store_history(t_neural_a.detach().cpu().numpy().astype(np.float32))
                    repro_np = reproject_history(hist, motion)
                proxy = cur_p
                features_np = build_features(cur_p, g_s, noise_seed, lane['style'], lane['tone'],
                                             lane['structure'], lane['skin'], lane['auto_mask'],
                                             history=repro_np)
            else:
                _, proxy, _, _, lane, _, noise_seed, features_np = item
                repro_np = None
                s = proxy.shape[0]
                g_s, teacher_s = geo_for(s)
                model.g = g_s                              # 学生算子与分辨率无关,换几何即换边长
            with torch.no_grad():
                t_res = teacher_s.annotate(features_np, capture=teacher_caps)
            # head 是场分辨率(如 512² -> 场 576x512),裁到有效区再进损失。
            t_head = torch.from_numpy(t_res['head']).view(
                g_s['full_height'], g_s['full_width'], 4)[:s, :s].to(device)
            proxy_t = torch.from_numpy(proxy).to(device)
            repro_t = torch.from_numpy(repro_np).to(device) if repro_np is not None else None

            with torch.autocast(device, dtype=torch.bfloat16):
                s_head = model(torch.from_numpy(features_np).to(device))
            s_head = s_head.view(g_s['full_height'], g_s['full_width'], 4)[:s, :s].float()
            out, s_neural, t_neural = output_losses(
                s_head, t_head, proxy_t, repro=repro_t,
                s_scale=model.blend_scale, t_scale=teacher_s.blend_scale)
            # ^ s_scale 传张量而非 float:blend_scale 是 nn.Parameter,曾被 float(detach())
            #   断梯度永远停在 1.0(教师实测 0.7397,历史混合量差 35%)。2026-10-10 修。
            detail = laplacian_l1(s_neural, t_neural) + sobel_l1(s_neural, t_neural)
            feat, feat_detail = feature_losses(model.captures, t_res, aligner, feature_weights)
            loss = (cfg['loss_out'] * out + cfg['loss_detail'] * detail
                    + cfg['loss_feature'] * feat)
            moe_aux = model.moe_aux_loss()                      # 在时序二次前向清统计前取
            aux_val = float(moe_aux.detach())
            if moe_aux.requires_grad:
                z = model.moe_z_loss()
                loss = loss + cfg.get('moe_aux', 0.1) * moe_aux
                loss = loss + cfg.get('moe_z', 1e-3) * z
                route = moe_route_metrics(model)                # 坍缩指标(Comet 上报)
                route['aux'] = aux_val
                route['z'] = float(z.detach())
                for k, v in route.items():
                    moe_tot[k] = moe_tot.get(k, 0.0) + v
            temporal = None
            if pair_mode and cfg['loss_temporal'] > 0:
                # 时序损失(真接上):帧差残差匹配 L1((s_B - s_A) - (t_B - t_A)) ——
                # 正是 eval_temporal 的 flicker 口径(输入差分抵消后)。
                with torch.no_grad():
                    s_head_a = model(torch.from_numpy(feats_a).to(device))
                s_head_a = s_head_a.view(g_s['full_height'], g_s['full_width'], 4)[:s, :s].float()
                _, s_neural_a, _ = output_losses(s_head_a, t_head_a, prev_t)
                temporal = F.l1_loss(s_neural - s_neural_a, t_neural - t_neural_a.detach())
                loss = loss + cfg['loss_temporal'] * temporal
            if not torch.isfinite(loss):
                print(f'step {step}: non-finite sample loss, skipped')
                continue
            (loss / cfg['batch']).backward()
            totals['out'] += float(out.detach())
            totals['detail'] += float(detail.detach())
            totals['feature'] += float(feat.detach())
            if temporal is not None:
                totals['temporal'] += float(temporal.detach())
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
            metrics = {
                'loss/total': (totals['out'] + totals['detail'] + totals['feature']) / n,
                'loss/out': totals['out'] / n,
                'loss/detail': totals['detail'] / n,
                'loss/feature': totals['feature'] / n,
                'psnr/teacher': sum(psnrs) / n,
                'train/lr': lr,
                'train/ms': ms,
            }
            if moe_tot:                                         # MoE 路由健康度(坍缩监控)
                nb = max(1, len(psnrs))
                metrics.update({f'moe/{k}': v / nb for k, v in moe_tot.items()})
            experiment.log_metrics(metrics, step=step)
            if feat_detail:
                experiment.log_metrics({f'feat/{k}': v for k, v in feat_detail.items()}, step=step)

        if step % cfg['log_every'] == 0 or step == cfg['steps'] - 1:
            n = cfg['batch']
            pn = max(1, len(psnrs))
            row = [step, f'{lr:.2e}',
                   f"{(totals['out'] + totals['detail'] + totals['feature'] + totals['temporal']) / n:.5f}",
                   f"{totals['out'] / n:.5f}", f"{totals['detail'] / n:.5f}",
                   f"{totals['feature'] / n:.5f}", f"{totals['temporal'] / n:.5f}",
                   f'{sum(psnrs) / pn:.2f}', f'{ms:.0f}']
            writer.writerow(row)
            log.flush()
            print(f"step {step:6d}  lr {lr:.2e}  out {totals['out'] / n:.5f}  "
                  f"detail {totals['detail'] / n:.5f}  feat {totals['feature'] / n:.5f}  "
                  f"temp {totals['temporal'] / n:.5f}  "
                  f"psnr(t) {sum(psnrs) / len(psnrs):.2f}"
                  + (f"  moe_aux {aux_val:.3f} umax {moe_tot.get('util_worst', 0.0) / max(1, len(psnrs)):.2f}"
                     if aux_val > 0 else "")
                  + f"  {ms:.0f} ms")

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
