#!/usr/bin/env python3
"""Distillation training (openspec tasks 5.1-5.9; numerics design.md D6/D11).

在线教师(D7:每 step 现跑,不缓存激活)提供 head 与逐级特征目标;学生在自己
的真实推理数值(bf16 AMP 训练 / f16 推理,无 QAT)下对齐。四类损失(5.3-5.6):
输出 L1(rgb/4 + blend logit SmoothL1 + 合成图,**按教师改动幅度加权** —— 强改动区
满权、弱改动区压到 action_floor,低水平修改不学)、细节(拉普拉斯 + Sobel 高频)、
特征(逐级 cosine+L2,形状不匹配处过 1x1 投影,投影随训练走)、时序(帧差一致,
序列模式)。LR:warmup -> 平段 -> 余弦降到 lr_floor 封底(lr_at)。

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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

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
from moe_training import (BiasController, configure, provenance, restore_gradients,
                          restore_rng, training_checkpoint, update_ema, validate_resume_sampling)

DEFAULTS = {
    'shape': os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'shapes', 'student_v0.json'),
    'teacher_weights': os.environ.get('NR_WEIGHTS'),
    'size': 320, 'size_weights': None,
    'batch': 4,
    'batch_rows': 4 * 512 * 512,  # 单次学生前向的实际 field 行预算(含 padding；至少容纳一张)
    'teacher_batch_rows': None,  # 默认沿用学生行预算；教师也必须分块，不能整步无限成批
    'teacher_overlap': True,    # 关闭后教师/学生串行，降低重叠峰值；CPU 准备仍预取
    'teacher_workers': 2,        # 兼容旧配置；GPU 教师当前由一个执行器串行调度，避免复制模型
    'steps': 200000,
    'overfit': 0,
    'lr': 2e-4, 'wd': 0.01, 'warmup': 2000, 'ema': 0.999, 'clip': 1.0,
    'lr_hold': 0.5,              # LR:warmup 后平段长度(占总步数;2026-10-12 起不再一降到底)
    'lr_decay': 0.25,            # 平段后余弦降到 lr_floor 的窗口(占总步数)
    'lr_floor': 1e-5,            # 尾段常数下限(降到这里就不降了,不归零)
    'loss_out': 1.0, 'loss_detail': 0.5, 'loss_feature': 0.5, 'loss_temporal': 1.0,
    'loss_logit': 1.0,           # blend logit 对齐权重(幅度阀专项;原 0.5 内联)
    'action_floor': 0.1,         # 教师改动加权:零改动处权重下限(低水平修改压到地板,不学)
    'action_knee': 0.02,         # 权重半衰点(教师 |合成−输入| 幅度 ≈ 5/255;改动越大权重越满)
    'loss_texture': 0.0,         # 高频专项权重(2026-10-14 v4.1):hf(out−proxy) 带单独监督,救滑区造纹理
    'texture_floor': 0.25,       # 高频专项:教师高频改动零处权重下限
    'texture_knee': 0.005,       # 高频专项:半衰点(≈1.3/255;滑区造纹理 ~2/255 落近满权区)
    'tex_boost': 0.0,            # 滑区加成(2026-10-14 v4.1b):输入平滑像素的高频权重倍增上限
    'tex_sigma': 0.008,          # 滑区加成宽度(|hf(输入)| 尺度 ≈2/255;平滑度高斯窗)
    'loss_texstat': 0.0,         # 纹理能量统计项(2026-10-15 v4.2):局部 |hf| 幅度匹配教师
    'texstat_boost': 3.0,        # 统计项滑区加成倍数(造纹理分支的主战场)
    'ema_every': 1,              # EMA 更新间隔步数(decay 取 ^ema_every,期望等效;省每步发射)
    'pair_fraction': 0.5,        # 时序对采样比例(历史腿:上帧教师输出 -> truncate_half -> 重投影)
    'crop_prob': 0.5,            # 随机原生裁剪(尺度抖动 1-2x);其余整图 resize
    'degrade_prob': 0.5,         # 在线退化(degrade.py 链;proxy_proc 已预退化的跳过)
    'moe_aux': 0.02,             # MoE 均衡正则(配额分配已结构保证均衡;此项仅防 p 漂移)
    'moe_z': 1e-2,               # router z-loss(ST-MoE,压 logits 幅值;router 前 LN 后配合生效)
    'student_precision': 'bf16', 'router_jitter': 0.1, 'student_compile': False,
    'optimizer_fused': None,     # 新 FP32 CUDA 学生默认 fused AdamW；旧结构保持既有优化器
    'moe_z_formula': 'legacy-logits-l2-v1', 'moe_bias_rate': 1e-3,
    'ema_bias_policy': 'legacy-average-float-buffers', 'comet_disabled': False,
    'log_every': 100, 'vis_every': 2000, 'ckpt_every': 5000,
    'seed': 20261007,
    'style_range': [0.0, 127.0], 'tone_range': [0.0, 1.0], 'structure_range': [0.0, 1.0],
    # v1.2 条件全覆盖(2026-10-10):style 是 ID(lane = ID/128,真实量程 [0,1));
    # v1.1 只采 [0,3] -> lane ≤0.023,导致高档失控(见 eval_conditions 探针)。
    'automask_prob': 0.3,        # automask 开时 skin ~ skin_range(lane 13/14 分支),关时 skin=-1
    'skin_range': [0.0, 1.0],    # skin 采样区间(置 [1,1] 即钉死 skin=1;2026-10-13 固定旋钮配方)
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
    explicit = set()
    if args.config:
        with open(args.config) as fh:
            provided = json.load(fh)
            cfg.update(provided)
            explicit.update(provided)
    for key in ('shape', 'teacher_weights', 'batch', 'steps', 'overfit', 'lr', 'wd',
                'warmup', 'ema', 'clip', 'loss_out', 'loss_detail', 'loss_feature',
                'loss_temporal', 'pair_fraction', 'seed', 'batch_rows', 'teacher_workers',
                'teacher_batch_rows', 'teacher_overlap',
                'lr_hold', 'lr_decay', 'lr_floor', 'loss_logit',
                'action_floor', 'action_knee'):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
            explicit.add(key)
    if getattr(args, 'size', None) is not None:
        cfg['size'] = args.size
    size = cfg['size']
    cfg['size'] = [int(v) for v in (size if isinstance(size, list) else str(size).split(','))]
    if getattr(args, 'size_weights', None) is not None:
        cfg['size_weights'] = args.size_weights
    if cfg['size_weights'] is not None:
        weights = cfg['size_weights']
        cfg['size_weights'] = [float(v) for v in (weights if isinstance(weights, list) else str(weights).split(','))]
        if len(cfg['size_weights']) != len(cfg['size']):
            raise SystemExit('size_weights length must match --size list')
    for key in ('batch_rows', 'teacher_batch_rows'):
        if cfg.get(key) is not None and int(cfg[key]) <= 0:
            raise SystemExit(key + ' must be positive')
    return configure(cfg, load_student_shape(cfg['shape']), explicit)


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
            'skin': float(rng.uniform(*self.cfg.get('skin_range', (0.0, 1.0)))) if auto_mask else -1.0,
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
            'skin': float(rng.uniform(*self.cfg.get('skin_range', (0.0, 1.0)))) if auto_mask else -1.0,
            'auto_mask': auto_mask,
        }
        return prev_p, cur_p, motion, lane, path_cur


# ---------------------------------------------------------------- losses

def laplacian_l1(a, b):
    """两级拉普拉斯"场之差"的 L1(5.4:让学生的高频形态对上教师,而不是抹掉高频)。
    批量口径:(B,H,W,C) -> 逐样本 (B,);B=1 与旧逐样本 .mean() 语义一致。

    注:初版写成 lap(a)+lap(b)(两图高频能量之和),没在比较师生、反而奖励模糊 —— 已修。
    """
    def lap(x):
        fields = []
        cur = x.permute(0, 3, 1, 2)
        for _ in range(2):
            h, w = cur.shape[-2:]
            low = F.avg_pool2d(cur, 2, ceil_mode=True)
            up = F.interpolate(low, size=(h, w), mode='bilinear', align_corners=False)
            fields.append(cur - up)
            cur = low
        return fields

    out = 0.0
    for fa, fb in zip(lap(a), lap(b)):
        out = out + (fa - fb).abs().reshape(a.shape[0], -1).mean(1)
    return out


def sobel_l1(a, b):
    """Sobel 边缘场之差的 L1,逐样本 (B,)。"""
    def edges(x):
        gray = x.mean(dim=3, keepdim=True).permute(0, 3, 1, 2)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        ky = kx.transpose(-1, -2)
        return F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)

    gxa, gya = edges(a)
    gxb, gyb = edges(b)
    return ((gxa - gxb).abs().reshape(a.shape[0], -1).mean(1)
            + (gya - gyb).abs().reshape(a.shape[0], -1).mean(1))


def hf(x):
    """高频带 x − blur3(x),blur3 = 9 次 3×3 均值(与 probe_skin.blur3 同口径)。
    批量口径:(B,H,W,C) -> (B,H,W,C)。"""
    y = x.permute(0, 3, 1, 2)
    for _ in range(9):
        y = F.avg_pool2d(F.pad(y, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
    return (x.permute(0, 3, 1, 2) - y).permute(0, 2, 3, 1)


def output_losses(s_head, t_head, proxy, repro=None, s_scale=1.0, t_scale=1.0,
                  logit_w=1.0, act_floor=0.1, act_knee=0.02,
                  tex_w=0.0, tex_floor=0.25, tex_knee=0.005,
                  tex_boost=0.0, tex_sigma=0.008,
                  texstat_w=0.0, texstat_boost=3.0):
    """输出损失(5.3):rgb/4 L1 + blend logit SmoothL1 + 合成图 L1,按教师改动幅度加权。

    2026-10-12 损失改造(用户口径:低水平修改不需要学习,学就学改动较大的):
    逐像素权重 w = floor + (1-floor)·Δ²/(Δ²+knee²),Δ = 教师 |合成 − 输入| 幅度 ——
    强改动区满权重,弱改动区压到 floor;**不做 w.mean() 归一化**(归一化会把整样本的
    弱改动再拉回满权,与"不学弱改动"相反),跨样本/跨像素的梯度占比就是改动幅度占比。
    blend logit 单独升为 loss_logit 项(幅度总阀,原 0.5 内联),同样吃 w。

    2026-10-14 高频专项(v4.1):滑区造纹理 ~2/255 的小改动在动作加权下只有 w≈0.25,
    梯度结构性流向光线(20K 实测滑区纹理复刻 47%,照片脸 91%)。tex_w > 0 时对高频带
    (probe_skin 同口径 hf())单独加权:权重按教师高频改动幅度(半衰 tex_knee ≈ 1.3/255,
    滑区造纹理落在近满权区),零高频改动处压到 tex_floor。

    2026-10-14 批量口径(向量化):张量 (B,H,W,·),返回逐样本向量 —— 逐样本 python
    循环是 GPU 气泡主源之一;B=1 与旧实现同语义(仅浮点求和序差异,~1e-7 级)。

    有历史(repro 非空)时合成走 temporal blend(run_image 语义:权重
    clamp(sigmoid(logit)*blend_scale)),blend logit 的监督此时才有意义;
    返回的 neural 是部署口径合成图(时序损失吃它)。t_scale 可为 (B,) 张量。
    返回 (out_v, s_neural, t_neural, tex_v):前三者含逐样本 L1 和,tex_v 为高频项原值。
    """
    def pm(z):                                          # per-sample mean over (H,W,C)
        return z.reshape(z.shape[0], -1).mean(1)

    s_rgb, t_rgb = s_head[..., :3] / 4.0, t_head[..., :3] / 4.0
    s_neural = torch.clamp(proxy + s_rgb, 0, 1)
    t_neural = torch.clamp(proxy + t_rgb, 0, 1)
    with torch.no_grad():
        delta = (t_neural - proxy).abs().mean(dim=3, keepdim=True)
        w = act_floor + (1.0 - act_floor) * delta.pow(2) / (delta.pow(2) + act_knee ** 2)

    def wl1(a, b, wm=None):
        ww = w if wm is None else wm
        return pm((a - b).abs() * ww)

    out = wl1(s_rgb, t_rgb) + logit_w * pm(
        F.smooth_l1_loss(s_head[..., 3], t_head[..., 3], reduction='none') * w[..., 0])
    out = out + wl1(s_neural, t_neural)
    if repro is not None:
        ts = torch.as_tensor(t_scale, dtype=s_head.dtype, device=s_head.device)
        ts = ts.view(-1, 1, 1) if ts.ndim else ts
        s_w = torch.clamp(torch.sigmoid(s_head[..., 3]) * s_scale, 0, 1)
        t_w = torch.clamp(torch.sigmoid(t_head[..., 3]) * ts, 0, 1)
        s_neural = s_neural + (repro - s_neural) * s_w[..., None]
        t_neural = t_neural + (repro - t_neural) * t_w[..., None]
        out = out + wl1(s_neural, t_neural)
    tex_v = torch.zeros(s_head.shape[0], dtype=s_head.dtype, device=s_head.device)
    if tex_w > 0:
        with torch.no_grad():
            d_hf = hf(t_neural - proxy).abs().mean(dim=3, keepdim=True)
            wt = tex_floor + (1.0 - tex_floor) * d_hf.pow(2) / (d_hf.pow(2) + tex_knee ** 2)
            if tex_boost > 0:
                # 滑区加成(v4.1b):皮肤滑区造纹理是稀有事件,全图均值梯度被大面积非
                # 皮肤高频淹没 —— 20K 实测滑区复刻 43%(10K 61% 被挤出)。输入自身
                # 高频越小(越平滑)加成越大,直接对准滑区复刻率验收指标。
                # (v4.2 起默认关:逐像素 L1 对自造纹理的最优解是"不造",加成反而
                #  加速收敛到零;造纹理分支交给下面的统计项。)
                mag0 = hf(proxy).abs().mean(dim=3, keepdim=True)
                wt = wt * (1.0 + tex_boost * torch.exp(-(mag0 / tex_sigma) ** 2))
        tex_v = pm((hf(s_neural) - hf(t_neural)).abs() * wt)
        out = out + tex_w * tex_v
    stat_v = torch.zeros(s_head.shape[0], dtype=s_head.dtype, device=s_head.device)
    if texstat_w > 0:
        # 纹理能量统计项(v4.2):教师造纹理是自造非还原(corr≈0),逐像素 L1 追
        # 不可预测纹理的最优解是"不造"(40K 实测滑区复刻 34-42%,权重越大掉得越快)。
        # 改为局部 |hf| 幅度统计匹配:学生造什么纹理由它,能量对上就给分 ——
        # 正对着"滑区纹理复刻率"验收口径。幅值用 Charbonnier 光滑,免零点梯度死角。
        def local_amp(z):
            a = torch.sqrt(hf(z) ** 2 + 1e-8).mean(dim=3, keepdim=True).permute(0, 3, 1, 2)
            return F.avg_pool2d(F.pad(a, (6, 6, 6, 6), mode='replicate'), 13,
                                stride=1).permute(0, 2, 3, 1)

        amp_s, amp_t = local_amp(s_neural - proxy), local_amp(t_neural - proxy)
        wstat = 1.0
        if texstat_boost > 0:
            with torch.no_grad():
                mag0 = hf(proxy).abs().mean(dim=3, keepdim=True)
                wstat = 1.0 + texstat_boost * torch.exp(-(mag0 / tex_sigma) ** 2)
        stat_v = pm((amp_s - amp_t).abs() * wstat)
        out = out + texstat_w * stat_v
    return out, s_neural, t_neural, tex_v, stat_v


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


def feature_losses(student_caps, teacher_caps, aligner, weights, diagnostics=True):
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
        if diagnostics:
            detail[s_name] = value.detach()
    if detail:
        values = torch.stack(list(detail.values())).cpu().tolist()
        detail = dict(zip(detail, values))
    return total / max(wsum, 1e-6), detail


# ---------------------------------------------------------------- training

def lr_at(step, cfg):
    """LR(2026-10-12 定型):warmup 线性升 -> 平段(lr_hold 占比)-> 余弦降到 lr_floor
    (lr_decay 占比)-> 常数 lr_floor 封底不归零(用户口径:warmup 后先平一阵,
    降低到 1e-5 就不降了)。"""
    warm = max(1, cfg['warmup'])
    if step < warm:
        return cfg['lr'] * (step + 1) / warm
    steps = max(warm + 1, cfg['steps'])
    hold_end = max(warm + 1, int(cfg.get('lr_hold', 0.5) * steps))
    floor_end = max(hold_end + 1, hold_end + int(cfg.get('lr_decay', 0.25) * steps))
    floor = cfg.get('lr_floor', 1e-5)
    if step < hold_end:
        return cfg['lr']
    if step < floor_end:
        t = (step - hold_end) / (floor_end - hold_end)
        return floor + (cfg['lr'] - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    return floor


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
    parser.add_argument('--batch-rows', dest='batch_rows', type=int,
                        help='单次学生前向的 field 行预算(含 padding；显存紧可调小)')
    parser.add_argument('--teacher-batch-rows', type=int,
                        help='单次教师前向的 field 行预算(默认沿用 batch_rows)')
    parser.add_argument('--no-teacher-overlap', dest='teacher_overlap', action='store_false', default=None,
                        help='教师和学生串行执行以降低显存峰值；CPU 准备仍预取')
    parser.add_argument('--teacher-workers', dest='teacher_workers', type=int,
                        help='兼容配置项；GPU 教师当前串行调度')
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
    parser.add_argument('--loss-logit', dest='loss_logit', type=float,
                        help='blend logit 对齐权重(幅度阀专项,默认 1.0)')
    parser.add_argument('--action-floor', dest='action_floor', type=float,
                        help='教师改动加权下限(默认 0.1;低水平修改压到地板)')
    parser.add_argument('--action-knee', dest='action_knee', type=float,
                        help='改动加权半衰点(默认 0.02 ≈ 5/255)')
    parser.add_argument('--lr-hold', dest='lr_hold', type=float,
                        help='warmup 后平段长度(占总步数,默认 0.5)')
    parser.add_argument('--lr-decay', dest='lr_decay', type=float,
                        help='降到 lr_floor 的余弦窗口(占总步数,默认 0.25)')
    parser.add_argument('--lr-floor', dest='lr_floor', type=float,
                        help='尾段常数 LR 下限(默认 1e-5)')
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
    shape = load_student_shape(cfg['shape'])
    from moe_shape import is_shared
    shared_routed = is_shared(shape)
    if shared_routed:
        cfg['source'] = provenance(cfg['shape'], cfg['teacher_weights'])
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'config.json'), 'w') as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)

    experiment = None
    key = None if cfg.get('comet_disabled', False) else comet_key(cfg)
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
    geo_cache = {}
    _tls = threading.local()
    _teacher_lock = threading.Lock()

    def geo_for(size):
        """每种训练边长一套 geometry(纯 CPU 数值,共享只读)。"""
        found = geo_cache.get(size)
        if found is None:
            found = geometry_from_valid(size, size)
            geo_cache[size] = found
        return found

    def teacher_for(size):
        """线程本地在线教师 + 线程本地 CUDA 流(2026-10-12 流水重叠)。

        annotate 内部会写 network.boundaries(可变状态),实例不能跨线程共享;
        教师核走独立流,与主循环学生前向/反向并发,重叠掉教师时间。
        """
        cache = getattr(_tls, 'teachers', None)
        if cache is None:
            cache = {}
            _tls.teachers = cache
        hit = cache.get(size)
        if hit is None:
            g_s = geo_for(size)
            with _teacher_lock:              # 串行加载:防 loader 竞态 + 显存尖峰
                teacher_s = TeacherAnnotator(cfg['teacher_weights'], g_s)
            stream = torch.cuda.Stream() if device == 'cuda' else None
            hit = (teacher_s, stream)
            cache[size] = hit
        return hit

    model = StudentNetwork(shape, g).to(device)
    model.capture_enabled = True
    controller = BiasController(model, cfg['moe_bias_rate']) if shared_routed else None
    if controller:
        for module in controller.modules.values():
            module.training_metrics_enabled = True
    student_fwd = model
    if cfg.get('student_compile'):
        # 瓦片路由(nr_shared_moe)已无 .tolist() 值 guard;capture_scalar_outputs 把
        # int(tiles.sum()) 符号化,cache 放开容纳 30 模块×尺寸×批次的编译帧。
        # 变输入实测 25 轮零重编译,262ms -> 200ms。
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.cache_size_limit = 1024
        student_fwd = torch.compile(model, dynamic=True)

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
    fused_adam = cfg['optimizer_fused']
    if fused_adam is None:
        fused_adam = bool(shared_routed and device == 'cuda')
    optim = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'], **({'fused': True} if fused_adam else {}))
    ema = {k: p.detach().clone() for k, p in model.state_dict().items()}

    start = 0
    if args.resume:
        # Loading the full model+EMA+Adam checkpoint directly on GPU retained
        # a second model/optimizer snapshot throughout the whole resumed run.
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if controller:
            old_source = ckpt.get('source', {})
            for key in ('teacher_manifest_sha256', 'shape_sha256', 'teacher_source_sha256'):
                if old_source.get(key) != cfg['source'].get(key):
                    raise ValueError('shared/routed replay provenance changed: ' + key)
            validate_resume_sampling(ckpt, cfg)
        model.load_state_dict(ckpt['model'])
        aligner.load_state_dict(ckpt['aligner'])
        optim.load_state_dict(ckpt['optim'])
        for group in optim.param_groups:
            group.update(fused=bool(fused_adam), foreach=False if fused_adam else None)
        for state in optim.state.values():
            if 'step' in state:
                state['step'] = state['step'].to(device if fused_adam else 'cpu')
        ema = {n: v.to(device) for n, v in ckpt['ema'].items()}
        start = ckpt['step'] + 1
        if controller:
            controller.load_state_dict(ckpt['routing_controller'])
            restore_rng(ckpt['rng'])
            if controller.step is not None:
                restore_gradients(ckpt, model, aligner)
                start = controller.step
        print(f"resumed at step {start} from {args.resume}")
        del ckpt

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
        """样本 CPU 准备(预取线程):采样/裁剪/退化/块匹配/上帧特征构造。

        逐 (step,bi) 独立种子,与执行顺序无关 -> 结果确定。教师标注移出本函数
        (2026-10-13 教师批量):见 prepare_step,整步成批走教师,与学生前向流水重叠。
        """
        item_pair = (cfg['pair_fraction'] > 0 and
                     np.random.default_rng([cfg['seed'], step, 7]).random() < cfg['pair_fraction'])
        if item_pair:
            prev_p, cur_p, motion, lane, _ = source.sample_pair(step * cfg['batch'] + bi)
            seed_a = int(np.random.default_rng([cfg['seed'], step, bi, 1]).integers(1 << 30))
            noise_seed = int(np.random.default_rng([cfg['seed'], step, bi, 2]).integers(1 << 30))
            s = cur_p.shape[0]
            g_s = geo_for(s)
            feats_a = build_features(prev_p, g_s, seed_a, lane['style'], lane['tone'],
                                     lane['structure'], lane['skin'], lane['auto_mask'])
            return {'kind': 'pair', 's': s, 'proxy': cur_p, 'prev': prev_p, 'motion': motion,
                    'lane': lane, 'noise_seed': noise_seed, 'feats_a': feats_a}
        proxy, noise_seed, lane, _ = source.sample(step * cfg['batch'] + bi)
        s = proxy.shape[0]
        g_s = geo_for(s)
        features_np = build_features(proxy, g_s, noise_seed, lane['style'], lane['tone'],
                                     lane['structure'], lane['skin'], lane['auto_mask'])
        return {'kind': 'single', 's': s, 'proxy': proxy, 'noise_seed': noise_seed,
                'lane': lane, 'feats': features_np, 'repro': None}

    def prepare_step(raw_items):
        """整步教师批量标注(单线程执行器 + 线程本地 CUDA 流,与学生前向流水重叠)。

        2026-10-13 教师批量:教师 GEMM 行主序批量、窗注意力/ViT 批内逐样本(nr_network
        batch 维)。pair 的历史链:上帧整批标注 -> 逐样本重投影 + 当前帧特征(CPU) ->
        当前帧整批标注(head + 特征捕获)。按尺寸分桶批处理,桶内顺序 = 输入顺序。
        返回逐样本 dict(键同旧版,消费者不动)。
        """
        def annotate_group(grp, s, key='feats', capture=()):
            g_s = geo_for(s)
            teacher_s, stream = teacher_for(s)
            x = np.stack([p[key] for p in grp]).reshape(-1, 16)
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                res = teacher_s.annotate(x, capture=capture, batch=len(grp))
            return g_s, teacher_s, res

        by_size = {}
        for p in raw_items:
            if p['kind'] == 'pair':
                by_size.setdefault(p['s'], []).append(p)
        def chunks(by_size):
            budget = cfg.get('teacher_batch_rows') or cfg['batch_rows']
            for s, items in by_size.items():
                limit = max(1, int(budget) // geo_for(s)['full_rows'])
                for i in range(0, len(items), limit):
                    yield s, items[i:i + limit]

        for s, grp in chunks(by_size):
            g_s, _, res = annotate_group(grp, s, key='feats_a')
            heads = res['head'].reshape(len(grp), g_s['full_height'], g_s['full_width'], 4)
            for i, p in enumerate(grp):
                h = heads[i][:s, :s]
                t_neural_a = np.clip(p['prev'] + h[..., :3] / 4, 0, 1).astype(np.float32)
                hist = store_history(t_neural_a)
                repro_np = reproject_history(hist, p['motion'])
                p['repro'] = repro_np
                p['t_head_a'] = h
                p['t_neural_a'] = t_neural_a
                p['feats'] = build_features(p['proxy'], g_s, p['noise_seed'], p['lane']['style'],
                                            p['lane']['tone'], p['lane']['structure'],
                                            p['lane']['skin'], p['lane']['auto_mask'],
                                            history=repro_np)

        out = []
        by_size = {}
        for p in raw_items:
            by_size.setdefault(p['s'], []).append(p)
        for s, grp in chunks(by_size):
            g_s, teacher_s, res = annotate_group(grp, s, capture=teacher_caps)   # 当前帧整批
            heads = res['head'].reshape(len(grp), g_s['full_height'], g_s['full_width'], 4)
            for i, p in enumerate(grp):
                p['t_head'] = heads[i][:s, :s]
                p['t_scale'] = teacher_s.blend_scale
                per = {}
                for name in teacher_caps:
                    arr = res[name]
                    n = arr.shape[0] // len(grp)
                    per[name] = arr[i * n:(i + 1) * n]
                p['t_res'] = per
                out.append(p)
        return out

    # 批内并行准备(2026-10-10):prep_item 是 (step,bi) 独立种子的纯函数,4 线程并行算、
    # 按 (step,bi) 顺序取回 -> 训练语义与串行版逐位一致;GPU 利用率 45% -> 75%+。
    # 2026-10-13 教师批量:CPU 原料与教师标注分两段 —— 教师段整步成批(单线程执行器,
    # 线程本地实例/流),与上/下一步的学生计算流水重叠;实例 2 套(320/512)而非 6 套。
    prep_pool = ThreadPoolExecutor(max_workers=4)
    teach_pool = ThreadPoolExecutor(max_workers=1)
    prep_futs = deque()
    step_futs = deque()
    prep_lock = threading.Lock()
    prep_seq = {'step': start, 'bi': 0}

    def _prep_submit():
        with prep_lock:
            s_, b_ = prep_seq['step'], prep_seq['bi']
            if s_ >= cfg['steps']:
                return False
            prep_seq['bi'] = b_ + 1
            if prep_seq['bi'] >= cfg['batch']:
                prep_seq['step'], prep_seq['bi'] = s_ + 1, 0
        prep_futs.append(prep_pool.submit(prep_item, s_, b_))
        return True

    def _collect_raw():
        """取回一步的 CPU 原料并补交预取;返回原料列表(顺序 = (step,bi) 序)。"""
        raw = []
        for _ in range(cfg['batch']):
            raw.append(prep_futs.popleft().result())
            _prep_submit()
        return raw

    for _ in range(8):
        if not _prep_submit():
            break
    step_futs.append(teach_pool.submit(prepare_step, _collect_raw()))

    for step in range(start, cfg['steps']):
        log_step = step % cfg['log_every'] == 0 or step == cfg['steps'] - 1
        if controller:
            for module in controller.modules.values():
                module.training_metrics_enabled = log_step
        lr = lr_at(step, cfg)
        for group in optim.param_groups:
            group['lr'] = lr
        t0 = time.perf_counter()
        totals = {'out': 0.0, 'detail': 0.0, 'feature': 0.0, 'temporal': 0.0,
                  'texture': 0.0, 'texstat': 0.0}
        moe_tot = {}
        feat_detail_acc = {}
        psnrs = []
        resumed_chunks = controller.microbatches if controller and controller.step == step else 0
        if not resumed_chunks:
            optim.zero_grad()
            if controller:
                controller.begin(step)
        chunk_cursor = 0
        # ---- 取回本步已备样本(教师批量段在上一步末尾提交,与上一步学生计算重叠),
        # 并把下一步的教师段压进流水(与本步学生计算重叠;末步不预取,防越界取料)。
        t_wait0 = time.perf_counter()
        prepared = step_futs.popleft().result()
        t_wait = time.perf_counter() - t_wait0
        next_raw = _collect_raw() if step + 1 < cfg['steps'] else None
        if next_raw is not None and cfg['teacher_overlap']:
            step_futs.append(teach_pool.submit(prepare_step, next_raw))

        # ---- 按 (边长, 模式) 分桶 + rows 预算切块,桶内真 batch 前向(2026-10-12 吞吐改造)。
        # 模型前向接受 batch 维行主序叠放(nr_student batch=1 与逐样本逐位一致;批量=逐样本
        # fp32 等价见钉2/钉3,同批同样本逐位自洽见钉4)。
        t_comp0 = time.perf_counter()
        buckets = {}
        for p in prepared:
            buckets.setdefault((p['s'], p['kind']), []).append(p)
        for (s, kind), group in buckets.items():
            g_s = geo_for(s)
            model.g = g_s                              # 学生算子与分辨率无关,换几何即换边长
            max_b = max(1, int(cfg.get('batch_rows', 4 * 512 * 512)) // g_s['full_rows'])
            for c0 in range(0, len(group), max_b):
                chunk_cursor += 1
                if chunk_cursor <= resumed_chunks:
                    continue
                chunk = group[c0:c0 + max_b]
                cnb = len(chunk)
                # 教师捕获按样本 stack 成整块布局喂特征损失。
                t_res_stack = {}
                for name in teacher_caps:
                    arr = np.stack([p['t_res'][name] for p in chunk])
                    t_res_stack[name] = arr.reshape(-1, arr.shape[-1])

                x = torch.from_numpy(np.stack([p['feats'] for p in chunk])).reshape(-1, 16).to(device)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=cfg['student_precision'] == 'bf16'):
                    s_head_all = student_fwd(x, batch=cnb)
                if controller:
                    controller.collect()
                s_head_all = s_head_all.view(
                    cnb, g_s['full_height'], g_s['full_width'], 4)[:, :s, :s].float()

                # 整块级损失:特征(整块捕获 ≈ 逐样本均值,行数同)与 MoE 均衡项 ——
                # 必须在时序二次前向(清 moe_stats/captures)之前取。
                feat, feat_detail = feature_losses(model.captures, t_res_stack,
                                                   aligner, feature_weights, diagnostics=log_step)
                need_moe = log_step or cfg['moe_aux'] != 0 or cfg['moe_z'] != 0
                moe_aux = model.moe_aux_loss() if need_moe else s_head_all.new_zeros(())
                aux_val = float(moe_aux.detach()) if log_step else 0.
                chunk_moe = 0.0
                if moe_aux.requires_grad:
                    z = model.moe_z_loss() if log_step or cfg['moe_z'] != 0 else moe_aux.new_zeros(())
                    chunk_moe = cfg.get('moe_aux', 0.1) * moe_aux + cfg.get('moe_z', 1e-3) * z
                    if log_step:
                        route = moe_route_metrics(model)
                        route['aux'] = aux_val
                        route['z'] = float(z.detach())
                        for k, v in route.items():
                            moe_tot[k] = moe_tot.get(k, 0.0) + v * cnb
                if log_step:
                    totals['feature'] += float(feat.detach()) * cnb
                for k, v in feat_detail.items():
                    feat_detail_acc[k] = feat_detail_acc.get(k, 0.0) + v * cnb

                # pair:上帧学生前向(teacher-forcing,no_grad)整块走,s_neural_a 批量合成。
                s_neural_a = None
                if kind == 'pair' and cfg['loss_temporal'] > 0:
                    xa = torch.from_numpy(
                        np.stack([p['feats_a'] for p in chunk])).reshape(-1, 16).to(device)
                    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=cfg['student_precision'] == 'bf16'):
                        sa_all = student_fwd(xa, batch=cnb)
                    sa_all = sa_all.view(
                        cnb, g_s['full_height'], g_s['full_width'], 4)[:, :s, :s].float()
                    prev_t = torch.from_numpy(np.stack([p['prev'] for p in chunk])).to(device)
                    s_neural_a = torch.clamp(prev_t + sa_all[..., :3] / 4.0, 0, 1)

                # 批量损失(2026-10-14 向量化):整块一次算、逐样本向量聚合 —— 旧逐样本
                # python 循环是 GPU 气泡主源之一(提速牌 7);B=1 语义与旧实现一致。
                chunk_loss = 0.0
                proxy = chunk[-1]['proxy']              # vis 块取图用(重构后保持旧名)
                proxy_t = torch.from_numpy(np.stack([p['proxy'] for p in chunk])).to(device)
                t_head_t = torch.from_numpy(np.stack([p['t_head'] for p in chunk])).to(device)
                repro_t = (torch.from_numpy(np.stack([p['repro'] for p in chunk])).to(device)
                           if kind == 'pair' else None)
                out_v, s_neural, t_neural, tex_v, stat_v = output_losses(
                    s_head_all, t_head_t, proxy_t, repro=repro_t,
                    s_scale=model.blend_scale,
                    t_scale=torch.tensor([p['t_scale'] for p in chunk], device=device),
                    logit_w=cfg['loss_logit'], act_floor=cfg['action_floor'],
                    act_knee=cfg['action_knee'], tex_w=cfg['loss_texture'],
                    tex_floor=cfg['texture_floor'], tex_knee=cfg['texture_knee'],
                    tex_boost=cfg['tex_boost'], tex_sigma=cfg['tex_sigma'],
                    texstat_w=cfg['loss_texstat'], texstat_boost=cfg['texstat_boost'])
                # ^ s_scale 传张量而非 float:blend_scale 是 nn.Parameter,曾被 float(detach())
                #   断梯度永远停在 1.0(教师实测 0.7397,历史混合量差 35%)。2026-10-10 修。
                detail_v = laplacian_l1(s_neural, t_neural) + sobel_l1(s_neural, t_neural)
                sample_v = cfg['loss_out'] * out_v + cfg['loss_detail'] * detail_v
                temporal_v = None
                if s_neural_a is not None:
                    # 时序损失(真接上):帧差残差匹配 L1((s_B - s_A) - (t_B - t_A)) ——
                    # 正是 eval_temporal 的 flicker 口径(输入差分抵消后)。
                    t_na_t = torch.from_numpy(
                        np.stack([p['t_neural_a'] for p in chunk])).to(device)
                    temporal_v = (s_neural - s_neural_a - (t_neural - t_na_t)
                                  ).abs().reshape(cnb, -1).mean(1)
                    sample_v = sample_v + cfg['loss_temporal'] * temporal_v
                finite = torch.isfinite(sample_v)
                if not bool(finite.all()):
                    print(f'step {step}: non-finite sample loss, '
                          f'skipped {int((~finite).sum())}')
                chunk_loss = chunk_loss + (
                    torch.where(finite, sample_v, torch.zeros_like(sample_v)).sum()
                    / cfg['batch'])
                if log_step:
                    fm = finite.to(out_v.dtype)
                    totals['out'] += float((out_v * fm).sum())
                    totals['detail'] += float((detail_v * fm).sum())
                    totals['texture'] += float((tex_v * fm).sum())
                    totals['texstat'] += float((stat_v * fm).sum())
                    if temporal_v is not None:
                        totals['temporal'] += float((temporal_v * fm).sum())
                    mse = ((s_neural.detach() - t_neural.detach()) ** 2).reshape(cnb, -1).mean(1)
                    psnr_v = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
                    psnrs.extend(psnr_v[finite].tolist())
                chunk_loss = chunk_loss + (cfg['loss_feature'] * feat + chunk_moe) * (
                    cnb / cfg['batch'])
                if torch.isfinite(chunk_loss):
                    if controller and (cfg['moe_aux'] != 0 or cfg['moe_z'] != 0):
                        controller.observe_regularizer(chunk_moe * (cnb / cfg['batch']))
                    chunk_loss.backward()
                else:
                    print(f'step {step}: non-finite chunk loss, skipped')
        t_comp = time.perf_counter() - t_comp0
        t_opt0 = time.perf_counter()
        if controller and log_step:
            controller.capture_gradients()
        torch.nn.utils.clip_grad_norm_(params, cfg['clip'])
        optim.step()
        if controller:
            controller.apply(step)
        if cfg['ema'] > 0:
            # EMA 暖起:固定 decay 在短跑里会被早期半成品权重污染(3K 步 × 0.999 视界
            # 下 32% 质量来自 step 1000-2000),按步数渐升到 cfg['ema'](run v1 长跑自然退化为常数)。
            decay = min(cfg['ema'], (1.0 + step) / (100.0 + step))
            with torch.no_grad():
                if controller:
                    update_ema(ema, model.state_dict(), decay)
                else:
                    for k, p in model.state_dict().items():
                        if p.dtype.is_floating_point:
                            ema[k].mul_(decay).add_(p.detach(), alpha=1 - decay)
        elif controller:
            update_ema(ema, model.state_dict(), 0.)
        t_opt = time.perf_counter() - t_opt0
        if next_raw is not None and not cfg['teacher_overlap']:
            step_futs.append(teach_pool.submit(prepare_step, next_raw))
        ms = (time.perf_counter() - t0) * 1000

        # comet 上报按 log_every 节流(2026-10-14 提速牌 2:每步 HTTP 上报是墙钟尾巴)。
        if experiment is not None and (step % cfg['log_every'] == 0 or step == cfg['steps'] - 1):
            n = max(1, len(psnrs))
            metrics = {
                'loss/total': (totals['out'] + totals['detail'] + totals['feature']) / n,
                'loss/out': totals['out'] / n,
                'loss/detail': totals['detail'] / n,
                'loss/feature': totals['feature'] / n,
                'loss/texture': totals['texture'] / n,
                'psnr/teacher': sum(psnrs) / n,
                'train/lr': lr,
                'train/ms': ms,
            }
            if moe_tot:                                         # MoE 路由健康度(坍缩监控)
                nb = max(1, len(psnrs))
                metrics.update({f'moe/{k}': v / nb for k, v in moe_tot.items()})
            experiment.log_metrics(metrics, step=step)
            if feat_detail_acc:
                nb2 = cfg['batch']
                experiment.log_metrics({f'feat/{k}': v / nb2 for k, v in feat_detail_acc.items()},
                                       step=step)

        if step % cfg['log_every'] == 0 or step == cfg['steps'] - 1:
            if controller:
                with open(os.path.join(args.out, 'routing.jsonl'), 'a', encoding='utf-8') as route_log:
                    json.dump(dict(step=step, gradient_source='task_only' if cfg['moe_aux'] == cfg['moe_z'] == 0 else 'task_plus_regularizers',
                                   layers=controller.diagnostics(model)), route_log)
                    route_log.write('\n')
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
                  f"temp {totals['temporal'] / n:.5f}  tex {totals['texture'] / n:.5f}"
                  f"  stat {totals['texstat'] / n:.5f}  "
                  f"psnr(t) {sum(psnrs) / len(psnrs):.2f}"
                  + (f"  moe_aux {aux_val:.3f} umax {moe_tot.get('util_worst', 0.0) / max(1, len(psnrs)):.2f}"
                     if aux_val > 0 else "")
                  + f"  {ms:.0f} ms"
                  + f"  wait {t_wait * 1000:.0f} comp {t_comp * 1000:.0f}"
                    f" opt {t_opt * 1000:.0f}")

        if cfg['vis_every'] and step % cfg['vis_every'] == 0 and step > 0:
            # 可视化只许看不许杀:2026-10-12 vis 块 NameError 杀过一次 28h 跑,套保险。
            # comet 不传 vis(2026-10-14 用户口径:没看过;本地产 PNG 保留)。
            try:
                vis = os.path.join(args.out, f'vis-{step:06d}')
                os.makedirs(vis, exist_ok=True)
                t_last = t_neural[-1].detach().cpu().numpy()
                s_last = s_neural[-1].detach().cpu().numpy()
                save_png(os.path.join(vis, 'proxy.png'), proxy)
                save_png(os.path.join(vis, 'teacher.png'), t_last)
                save_png(os.path.join(vis, 'student.png'), s_last)
                save_png(os.path.join(vis, 'diff_x12.png'), np.abs(s_last - t_last) * 12)
            except Exception as exc:                            # noqa: BLE001
                print(f'step {step}: vis failed ({exc!r}), skipped')

        if cfg['ckpt_every'] and step % cfg['ckpt_every'] == 0 and step > 0:
            torch.save(training_checkpoint(model, aligner, optim, ema, cfg, step, controller),
                       os.path.join(args.out, 'ckpt.pt'))
    torch.save(training_checkpoint(model, aligner, optim, ema, cfg, cfg['steps'] - 1, controller), os.path.join(args.out, 'ckpt.pt'))
    log.close()
    if experiment is not None:
        experiment.end()
    print('done ->', os.path.join(args.out, 'ckpt.pt'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
