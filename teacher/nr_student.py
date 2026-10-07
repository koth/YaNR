#!/usr/bin/env python3
"""Student network (openspec tasks 2.1-2.8; numerics per design.md D11).

学生是蒸馏目标的小模型:结构沿教师的语义骨架(逐级 FFN/expert/split 块 + 余弦窗注意力
+ 底部全局 ViT + post_blend + 32->4 head),数值完全自由 —— 标准 matmul + 常规 softmax,
不走 fp8 定点链(教师那套数值是复原的枷锁,不是快的来源),通道/hidden/窗口不欠 kernel 的债。
torch nn.Module,训练直接挂优化器。

块语义与教师同构(便于逐级特征对齐,design.md D2/D11):
  FFN:    y = x + contract(silu(expand(x))) * aux_ffn
  +attn:  z = y + proj(window_attn(qkv(y))) * aux_attn
  expert: expand(E 个全输入->h)-> 窄支(h->32)合并(E*32=ch)-> contract(ch->ch)(残差+aux)
  split:  branch(ch->ch)-> E(bc->mc->bc)-> contract(ch->ch)(残差+aux)-> qkv/attn/proj
  ViT:    expand->contract(残差)-> qkv-> 全 token 余弦注意力-> proj(残差)

窗注意力:8x8 窗(可 16 槽)、相位循环同 nr_geometry.PHASES,余弦归一化 q/k、每头可学习
scale 与相对先验 [heads, slots, slots]、填充位掩码 softmax。ViT:列掩码到 tokens,query 带 sqrt(32)。
"""
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_geometry import geometry_from_valid, window_phase                    # noqa: E402

HEAD_DIM = 32
WINDOW = 8
# 教师的相位计数器语义:每级按块序号 0..n 走 PHASES[i & 3](full=6 号计数器)。
# 学生在构造期把相位定死进各块,前向不再传 phase(也是 torch.compile 的图稳定条件)。
LEVEL_INDEX = {'d0': 0, 'd1': 1, 'd2': 2, 'd3': 3, 'd4': 4, 'full': 6}

# 蒸馏对齐表(2.8):学生捕获点 -> 教师捕获点(annotate.py --probe 的 79 点子集)。
ALIGNMENT = {
    's-enc-full': 'block-0', 's-enc-d0': 'block-4', 's-enc-d1': 'block-8',
    's-enc-d2': 'block-14', 's-enc-d3': 'block-22', 's-enc-d4': 'block-30',
    's-vit': 'block-38',
    's-dec-d4': 'block-47', 's-dec-d3': 'block-55', 's-dec-d2': 'block-61',
    's-dec-d1': 'block-65', 's-dec-d0': 'block-69',
    's-head': 'head',
}


def student_alignment():
    """学生捕获点 -> 教师捕获点。"""
    return dict(ALIGNMENT)


# ---------------------------------------------------------------- shapes

def load_student_shape(path):
    """shapes/*.json(与 cost_model 同一 DSL)。"""
    with open(path) as fh:
        spec = json.load(fh)
    for level in spec['levels']:
        if level['channels'] % 32 or level['hidden'] % 32:
            raise ValueError(f"{level['level']}: channels/hidden must be multiples of 32")
        if level['kind'] not in ('ffn', 'expert', 'split'):
            raise ValueError(f"{level['level']}: unknown kind {level['kind']!r}")
        if level['kind'] == 'split':
            for key in ('branches', 'branch_channels', 'middle_channels'):
                if key not in level:
                    raise ValueError(f"{level['level']}: split needs {key}")
            if level['branches'] * level['branch_channels'] != level['channels']:
                raise ValueError(f"{level['level']}: split needs branches*branch_channels == channels")
    return spec


def level_rows(g, name):
    if name == 'full':
        return g['full_rows']
    return g['levels'][int(name[1:])]['rows']


def level_size(g, name):
    if name == 'full':
        return g['full_width'], g['full_height']
    level = g['levels'][int(name[1:])]
    return level['width'], level['height']


# ---------------------------------------------------------------- attention

def cosine_norm(x):
    """单位长度(教师的 pair-square 树在这里就是 L2 归一化)。"""
    return x / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(x.dtype)


class WindowAttention(nn.Module):
    """8x8 移位窗注意力:余弦 q/k、每头 scale、相对先验、掩码 softmax。

    shift 在构造期定死(教师相位是逐块确定的:每级计数器按块序号 0..n 走,
    PHASES[i & 3]);前向不传 phase —— 既是语义事实,也让 torch.compile 图稳定。
    """

    def __init__(self, channels, window=WINDOW, shift=(0, 0)):
        super().__init__()
        assert channels % HEAD_DIM == 0
        self.heads = channels // HEAD_DIM
        self.window = window
        self.slots = window * window
        self.shift_x, self.shift_y = shift
        self.scale = nn.Parameter(torch.ones(self.heads))
        self.prior = nn.Parameter(torch.zeros(self.heads, self.slots, self.slots))

    def forward(self, qkv, width, height):
        rows = width * height
        heads = self.heads
        slots = self.slots
        win = self.window
        shift_x, shift_y = self.shift_x, self.shift_y
        windows_x = (width + shift_x + win - 1) // win
        windows_y = (height + shift_y + win - 1) // win
        windows = windows_x * windows_y

        qkv3 = qkv.view(rows, heads, 3, HEAD_DIM)
        q = cosine_norm(qkv3[:, :, 0]) * self.scale.view(1, heads, 1)
        k = cosine_norm(qkv3[:, :, 1])
        v = qkv3[:, :, 2]

        win_x = torch.arange(windows_x, device=qkv.device) * win - shift_x
        win_y = torch.arange(windows_y, device=qkv.device) * win - shift_y
        slot = torch.arange(slots, device=qkv.device)
        ox = win_x.view(-1, 1).expand(windows_x, windows_y).reshape(-1)     # [windows]
        oy = win_y.view(1, -1).expand(windows_x, windows_y).reshape(-1)
        fx = (ox.view(-1, 1) + (slot % win).view(1, -1)).reshape(-1)
        fy = (oy.view(-1, 1) + (slot // win).view(1, -1)).reshape(-1)
        valid = (fx >= 0) & (fx < width) & (fy >= 0) & (fy < height)
        index = fy.clamp(0, height - 1) * width + fx.clamp(0, width - 1)

        def gather(t):     # [rows, heads, 32] -> [windows, heads, slots, 32]
            out = t[index].view(windows, slots, heads, HEAD_DIM).permute(0, 2, 1, 3)
            return out * valid.view(windows, 1, slots, 1).to(t.dtype)

        scores = torch.einsum('whsd,whtd->whst', gather(q), gather(k))
        scores = scores + self.prior.view(1, heads, slots, slots)
        # NaN 安全的掩码 softmax:-inf 版在真实数值下反向会产出 NaN(诊断:bisect 显示
        # 首个 NaN 就在带注意力的级),用 finfo.min + where 清零,全程有限值。
        mask = valid.view(windows, 1, slots, 1)
        neg = torch.finfo(scores.dtype).min
        masked = scores.masked_fill(~mask, neg)
        shifted = masked - masked.amax(dim=-1, keepdim=True)
        exp = torch.exp(shifted) * mask.to(scores.dtype)
        attn = exp / exp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        out = torch.einsum('whst,whtd->whsd', attn, gather(v))

        result = qkv.new_zeros((rows, heads * HEAD_DIM))
        values = out.permute(0, 2, 1, 3).reshape(windows * slots, heads * HEAD_DIM)
        result[index[valid]] = values[valid]
        return result


class VitAttention(nn.Module):
    """全 token 注意力(底部 ViT):列掩码到 tokens,query 带 sqrt(32) 与每头 scale。"""

    def __init__(self, channels):
        super().__init__()
        assert channels % HEAD_DIM == 0
        self.heads = channels // HEAD_DIM
        self.scale = nn.Parameter(torch.ones(self.heads))

    def forward(self, qkv, tokens, padded):
        heads = self.heads
        qkv3 = qkv.view(tokens, heads, 3, HEAD_DIM)
        q = cosine_norm(qkv3[:, :, 0]) * self.scale.view(1, heads, 1) * math.sqrt(HEAD_DIM)
        k = cosine_norm(qkv3[:, :, 1])
        v = qkv3[:, :, 2]
        k_pad = qkv.new_zeros((padded, heads, HEAD_DIM))
        v_pad = qkv.new_zeros((padded, heads, HEAD_DIM))
        k_pad[:tokens] = k
        v_pad[:tokens] = v

        scores = torch.einsum('thd,phd->htp', q, k_pad)
        keep = (torch.arange(padded, device=qkv.device) < tokens).view(1, 1, padded)
        neg = torch.finfo(scores.dtype).min
        masked = scores.masked_fill(~keep, neg)
        shifted = masked - masked.amax(dim=-1, keepdim=True)
        exp = torch.exp(shifted) * keep.to(scores.dtype)
        attn = exp / exp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        out = torch.einsum('htp,phd->thd', attn, v_pad)
        return out.reshape(tokens, heads * HEAD_DIM)


# ---------------------------------------------------------------- blocks

class _AttentionTail(nn.Module):
    """qkv -> 窗注意力 -> proj(残差+aux),FFN 三类块共用。"""

    def __init__(self, channels, window, shift):
        super().__init__()
        self.qkv = nn.Linear(channels, channels * 3)
        self.attention = WindowAttention(channels, window, shift)
        self.proj = nn.Linear(channels, channels)
        self.aux_attn = nn.Parameter(torch.ones(channels))

    def forward(self, out, width, height):
        attended = self.attention(self.qkv(out), width, height)
        return out + self.proj(attended) * self.aux_attn


class FFNBlock(nn.Module):
    """expand->SiLU->contract(残差+aux),可选注意力尾(2.3 / 2.5)。"""

    def __init__(self, channels, hidden, attn=True, window=WINDOW, phase_index=0):
        super().__init__()
        self.expand = nn.Linear(channels, hidden)
        self.contract = nn.Linear(hidden, channels)
        self.aux_ffn = nn.Parameter(torch.ones(channels))
        self.tail = (_AttentionTail(channels, window, window_phase(phase_index))
                     if attn else None)

    def forward(self, state, width, height):
        out = self.contract(F.silu(self.expand(state))) + state * self.aux_ffn
        if self.tail is not None:
            out = self.tail(out, width, height)
        return out


class ExpertBlock(nn.Module):
    """教师的 expert FFN:E 个全输入->h 展开、h->32 窄支、合并 ch->ch 契约(残差+aux)(2.4)。"""

    def __init__(self, channels, hidden, expert_channels=32, attn=True, window=WINDOW,
                 phase_index=0):
        super().__init__()
        assert channels % expert_channels == 0
        self.experts = channels // expert_channels
        self.expand = nn.Parameter(torch.empty(self.experts, channels, hidden))
        self.narrow = nn.Parameter(torch.empty(self.experts, hidden, expert_channels))
        nn.init.kaiming_uniform_(self.expand, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.narrow, a=math.sqrt(5))
        self.contract = nn.Linear(channels, channels)
        self.aux_ffn = nn.Parameter(torch.ones(channels))
        self.tail = (_AttentionTail(channels, window, window_phase(phase_index))
                     if attn else None)

    def forward(self, state, width, height):
        ffn = F.silu(torch.einsum('rc,ech->erh', state, self.expand))
        merged = torch.einsum('erh,ehc->erc', ffn, self.narrow).reshape(state.shape[0], -1)
        out = self.contract(merged) + state * self.aux_ffn
        if self.tail is not None:
            out = self.tail(out, width, height)
        return out


class SplitBlock(nn.Module):
    """教师的 split 块:branch(ch->ch)、E 个 bc->mc->bc 窄支、contract(残差+aux),后接注意力(2.4)。"""

    def __init__(self, channels, branches, branch_channels, middle_channels,
                 attn=True, window=WINDOW, phase_index=0):
        super().__init__()
        assert branches * branch_channels == channels, 'split: E * bc must equal ch (grouped slices)'
        self.branches = branches
        self.bc = branch_channels
        self.mc = middle_channels
        self.branch = nn.Linear(channels, channels)
        self.w2 = nn.Parameter(torch.empty(branches, branch_channels, middle_channels))
        self.w3 = nn.Parameter(torch.empty(branches, middle_channels, branch_channels))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))
        self.contract = nn.Linear(channels, channels)
        self.aux_ffn = nn.Parameter(torch.ones(channels))
        self.tail = (_AttentionTail(channels, window, window_phase(phase_index))
                     if attn else None)

    def forward(self, state, width, height):
        rows = state.shape[0]
        b = self.branch(state).view(rows, self.branches, self.bc)
        mid = F.silu(torch.einsum('rbc,bcm->rbm', b, self.w2))
        narrow = torch.einsum('rbm,bmc->rbc', mid, self.w3).reshape(rows, -1)
        out = self.contract(narrow) + state * self.aux_ffn
        if self.tail is not None:
            out = self.tail(out, width, height)
        return out


class VitBlock(nn.Module):
    """ViT 块:expand->contract(残差)-> qkv->全 token 注意力-> proj(残差)(2.6)。"""

    def __init__(self, channels, ffn_channels):
        super().__init__()
        self.expand = nn.Linear(channels, ffn_channels)
        self.contract = nn.Linear(ffn_channels, channels)
        self.aux_ffn = nn.Parameter(torch.ones(channels))
        self.qkv = nn.Linear(channels, channels * 3)
        self.attention = VitAttention(channels)
        self.proj = nn.Linear(channels, channels)
        self.aux_attn = nn.Parameter(torch.ones(channels))

    def forward(self, state, tokens, padded):
        out = self.contract(F.silu(self.expand(state))) + state * self.aux_ffn
        attended = self.attention(self.qkv(out), tokens, padded)
        return out + self.proj(attended) * self.aux_attn


# ---------------------------------------------------------------- plumbing

def box_downsample(x, in_w, in_h, out_w, out_h):
    """2x2 box pool((a+b)+(c+d))*0.25,越界位 0 —— 教师 downsample 的自由数值版(2.7)。"""
    x = x.view(in_h, in_w, -1)
    sx = (torch.arange(out_w, device=x.device) * 2).clamp(max=max(in_w - 1, 0))
    sy = (torch.arange(out_h, device=x.device) * 2).clamp(max=max(in_h - 1, 0))
    sx1 = (sx + 1).clamp(max=in_w - 1)
    sy1 = (sy + 1).clamp(max=in_h - 1)
    valid = ((sx + 1 < in_w).view(1, -1)) & ((sy + 1 < in_h).view(-1, 1))
    top = x[sy][:, sx] + x[sy][:, sx1]
    bottom = x[sy1][:, sx] + x[sy1][:, sx1]
    values = (top + bottom) * 0.25
    return torch.where(valid.unsqueeze(-1), values,
                       torch.zeros_like(values)).reshape(-1, x.shape[-1])


def _upsample2(low, in_w, in_h, out_w, out_h):
    base = low.view(in_h, in_w, -1)
    src_y = (torch.arange(out_h, device=low.device) >> 1).clamp(max=in_h - 1)
    src_x = (torch.arange(out_w, device=low.device) >> 1).clamp(max=in_w - 1)
    return base[src_y][:, src_x]


class UpsampleMerge(nn.Module):
    """解码入口:最近邻 2x 上采样 + 编码跳连 * 每通道 aux(教师 upsample_residual 语义)。"""

    def __init__(self, channels):
        super().__init__()
        self.aux = nn.Parameter(torch.ones(channels))

    def forward(self, low, skip, in_w, in_h, out_w, out_h):
        up = _upsample2(low, in_w, in_h, out_w, out_h)
        merged = up + skip.view(out_h, out_w, -1) * self.aux.view(1, 1, -1)
        return merged.reshape(-1, merged.shape[-1])


class PostBlend(nn.Module):
    """block 70 前的最后合并:d0 解码输出上采样 *aux0 + full 级跳连 *aux1(2.7)。"""

    def __init__(self, channels):
        super().__init__()
        self.aux_pair = nn.Parameter(torch.ones(2, channels))

    def forward(self, low, skip, in_w, in_h, out_w, out_h):
        up = _upsample2(low, in_w, in_h, out_w, out_h) * self.aux_pair[0].view(1, 1, -1)
        merged = up + skip.view(out_h, out_w, -1) * self.aux_pair[1].view(1, 1, -1)
        return merged.reshape(-1, merged.shape[-1])


def make_block(level, phase_index):
    """按形状条目构造块(2.3-2.6);相位索引即教师的每级计数器值。"""
    kind = level['kind']
    ch = level['channels']
    attn = level.get('attn', True)
    window = level.get('window_slots', WINDOW)
    if kind == 'ffn':
        return FFNBlock(ch, level['hidden'], attn=attn, window=window, phase_index=phase_index)
    if kind == 'expert':
        return ExpertBlock(ch, level['hidden'], attn=attn, window=window, phase_index=phase_index)
    return SplitBlock(ch, level['branches'], level['branch_channels'],
                      level['middle_channels'], attn=attn, window=window,
                      phase_index=phase_index)


# ---------------------------------------------------------------- the network

class StudentNetwork(nn.Module):
    """学生整网(2.2 / 2.7):full(1+1)→ d0..d4 各级 enc/dec → ViT → post_blend → head。

    块计划:每级 enc = blocks//2、dec = 其余(full 级 1+1);相位计数与教师同规则
    (full=6,d0..d4=0..4,解码续计)。
    """

    def __init__(self, shape, g):
        super().__init__()
        self.g = g
        self.shape = shape
        self.blend_scale = nn.Parameter(torch.ones(1))
        levels = {lv['level']: lv for lv in shape['levels']}
        order = ('full', 'd0', 'd1', 'd2', 'd3', 'd4')
        self.channels = {name: levels[name]['channels'] for name in order}

        self.adapter = nn.Linear(16, self.channels['full'])
        self.head = nn.Linear(self.channels['full'], 4)

        self.enc_blocks = nn.ModuleDict()
        self.dec_blocks = nn.ModuleDict()
        self.plan = []
        for name in order:
            lv = levels[name]
            enc_count = max(1, lv['blocks'] // 2)
            dec_count = max(1, lv['blocks'] - enc_count)
            # 相位索引 = 教师的每级计数器:enc 块 0..enc-1,dec 块续计(enc..)。
            self.enc_blocks[name] = nn.ModuleList(
                make_block(lv, j) for j in range(enc_count))
            self.dec_blocks[name] = nn.ModuleList(
                make_block(lv, enc_count + j) for j in range(dec_count))
            self.plan.append({'level': name, 'kind': lv['kind'],
                              'enc': enc_count, 'dec': dec_count})

        # 过渡:enc 每级 ch->2ch(含 full->d0 的学习变换)、dec 对称、上采样合并、ViT 进出。
        self.enc_trans = nn.ModuleDict({
            name: nn.Linear(self.channels[low], self.channels[name])
            for low, name in zip(order, order[1:])})
        # 解码投影方向相反:高层(更大) -> 本层。
        self.dec_trans = nn.ModuleDict({
            name: nn.Linear(self.channels[higher], self.channels[name])
            for name, higher in (('d3', 'd4'), ('d2', 'd3'), ('d1', 'd2'), ('d0', 'd1'))})
        self.merge = nn.ModuleDict({name: UpsampleMerge(self.channels[name])
                                    for name in order[1:]})
        vit_ch = shape['vit']['channels']
        self.vit_in = nn.Linear(self.channels['d4'], vit_ch)
        self.vit_out = nn.Linear(vit_ch, self.channels['d4'])
        self.vit_blocks = nn.ModuleList(
            VitBlock(vit_ch, shape['vit']['ffn']) for _ in range(shape['vit']['blocks']))
        self.post = PostBlend(self.channels['full'])

        self.captures = {}
        self.capture_enabled = False

    def capture(self, name, tensor):
        if self.capture_enabled:
            self.captures[name] = tensor

    def forward(self, features):
        g = self.g
        full_w, full_h = g['full_width'], g['full_height']
        order = ('full', 'd0', 'd1', 'd2', 'd3', 'd4')

        # ---- 编码:full -> d4,每级块后池化 + 展宽(full->d0 也有学习变换,便宜且更自由)。
        state = self.adapter(features)
        skips = {}
        for i, name in enumerate(order):
            width, height = level_size(g, name)
            for j, block in enumerate(self.enc_blocks[name]):
                state = block(state, width, height)
                if j == len(self.enc_blocks[name]) - 1:
                    self.capture(f's-enc-{name}', state)
            skips[name] = state
            if name != 'd4':
                next_name = order[i + 1]
                nw, nh = level_size(g, next_name)
                state = self.enc_trans[next_name](box_downsample(state, width, height, nw, nh))

        # ---- ViT(d5 tokens)。
        tokens = g['vit_tokens']
        padded = g['padded_vit_tokens']
        dw, dh = level_size(g, 'd5')
        state = self.vit_in(box_downsample(state, *level_size(g, 'd4'), dw, dh))
        for block in self.vit_blocks:
            state = block(state, tokens, padded)
        self.capture('s-vit', state)

        # ---- 解码:d5 -> d4 -> d3 -> d2 -> d1 -> d0 -> full。
        state = self.vit_out(state)
        prev_name, prev_w, prev_h = 'd5', dw, dh
        for name in ('d4', 'd3', 'd2', 'd1', 'd0'):
            width, height = level_size(g, name)
            if prev_name != 'd5':
                state = self.dec_trans[name](state)
            state = self.merge[name](state, skips[name], prev_w, prev_h, width, height)
            for block in self.dec_blocks[name]:
                state = block(state, width, height)
            self.capture(f's-dec-{name}', state)
            prev_name, prev_w, prev_h = name, width, height

        state = self.post(state, skips['full'], prev_w, prev_h, full_w, full_h)
        for block in self.dec_blocks['full']:
            state = block(state, full_w, full_h)
        head = self.head(state)
        self.capture('s-head', head)
        return head


def build(shape_path=None):
    """便捷构造:形状 + 512x512 几何(调用方自备 geometry 时直接用 StudentNetwork)。"""
    if shape_path is None:
        shape_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'shapes', 'student_v0.json')
    shape = load_student_shape(shape_path)
    return shape
