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
    if 'architecture' in spec:
        from moe_shape import validate
        return validate(spec)
    for level in spec['levels']:
        if level['channels'] % 32 or level['hidden'] % 32:
            raise ValueError(f"{level['level']}: channels/hidden must be multiples of 32")
        if level['kind'] not in ('ffn', 'expert', 'split', 'moe'):
            raise ValueError(f"{level['level']}: unknown kind {level['kind']!r}")
        if level['kind'] == 'moe':
            if level.get('experts', 0) < 2:
                raise ValueError(f"{level['level']}: moe needs experts >= 2")
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

    def forward(self, qkv, width, height, batch=1):
        """batch>1(2026-10-12 吞吐):样本按行主序叠放(rows = width*height 每样本,
        总行数 batch*rows);窗索引加样本行偏移、valid 平铺,窗网格不跨样本。
        batch=1 走原语义,数值逐位不变。"""
        rows = width * height
        heads = self.heads
        slots = self.slots
        win = self.window
        shift_x, shift_y = self.shift_x, self.shift_y
        windows_x = (width + shift_x + win - 1) // win
        windows_y = (height + shift_y + win - 1) // win
        windows = windows_x * windows_y

        win_x = torch.arange(windows_x, device=qkv.device) * win - shift_x
        win_y = torch.arange(windows_y, device=qkv.device) * win - shift_y
        slot = torch.arange(slots, device=qkv.device)
        ox = win_x.view(-1, 1).expand(windows_x, windows_y).reshape(-1)     # [windows]
        oy = win_y.view(1, -1).expand(windows_x, windows_y).reshape(-1)
        fx = (ox.view(-1, 1) + (slot % win).view(1, -1)).reshape(-1)
        fy = (oy.view(-1, 1) + (slot // win).view(1, -1)).reshape(-1)
        valid = (fx >= 0) & (fx < width) & (fy >= 0) & (fy < height)
        index = fy.clamp(0, height - 1) * width + fx.clamp(0, width - 1)
        if batch > 1:
            off = (torch.arange(batch, device=qkv.device) * rows).view(-1, 1)
            index = (index.view(1, -1) + off).reshape(-1)
            valid = valid.view(1, -1).expand(batch, -1).reshape(-1)
            windows = windows * batch
            rows = rows * batch

        qkv3 = qkv.view(rows, heads, 3, HEAD_DIM)
        q = cosine_norm(qkv3[:, :, 0]) * self.scale.view(1, heads, 1)
        k = cosine_norm(qkv3[:, :, 1])
        v = qkv3[:, :, 2]

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

    def forward(self, out, width, height, batch=1):
        attended = self.attention(self.qkv(out), width, height, batch)
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

    def forward(self, state, width, height, batch=1):
        out = self.contract(F.silu(self.expand(state))) + state * self.aux_ffn
        if self.tail is not None:
            out = self.tail(out, width, height, batch)
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

    def forward(self, state, width, height, batch=1):
        ffn = F.silu(torch.einsum('rc,ech->erh', state, self.expand))
        merged = torch.einsum('erh,ehc->erc', ffn, self.narrow).reshape(state.shape[0], -1)
        out = self.contract(merged) + state * self.aux_ffn
        if self.tail is not None:
            out = self.tail(out, width, height, batch)
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

    def forward(self, state, width, height, batch=1):
        rows = state.shape[0]
        b = self.branch(state).view(rows, self.branches, self.bc)
        mid = F.silu(torch.einsum('rbc,bcm->rbm', b, self.w2))
        narrow = torch.einsum('rbm,bmc->rbc', mid, self.w3).reshape(rows, -1)
        out = self.contract(narrow) + state * self.aux_ffn
        if self.tail is not None:
            out = self.tail(out, width, height, batch)
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

def box_downsample(x, in_w, in_h, out_w, out_h, batch=1):
    """2x2 box pool((a+b)+(c+d))*0.25,越界位 0 —— 教师 downsample 的自由数值版(2.7)。

    batch>1:样本按行主序叠放((batch*in_h*in_w, C)),输出同布局;batch=1 原路径逐位不变。
    """
    sx = (torch.arange(out_w, device=x.device) * 2).clamp(max=max(in_w - 1, 0))
    sy = (torch.arange(out_h, device=x.device) * 2).clamp(max=max(in_h - 1, 0))
    sx1 = (sx + 1).clamp(max=in_w - 1)
    sy1 = (sy + 1).clamp(max=in_h - 1)
    valid = ((sx + 1 < in_w).view(1, -1)) & ((sy + 1 < in_h).view(-1, 1))
    if batch > 1:
        x4 = x.view(batch, in_h, in_w, -1)
        top = x4[:, sy][:, :, sx] + x4[:, sy][:, :, sx1]
        bottom = x4[:, sy1][:, :, sx] + x4[:, sy1][:, :, sx1]
        values = (top + bottom) * 0.25
        return torch.where(valid.view(1, out_h, out_w, 1), values,
                           torch.zeros_like(values)).reshape(-1, x.shape[-1])
    x = x.view(in_h, in_w, -1)
    top = x[sy][:, sx] + x[sy][:, sx1]
    bottom = x[sy1][:, sx] + x[sy1][:, sx1]
    values = (top + bottom) * 0.25
    return torch.where(valid.unsqueeze(-1), values,
                       torch.zeros_like(values)).reshape(-1, x.shape[-1])


def _upsample2(low, in_w, in_h, out_w, out_h, batch=1):
    src_y = (torch.arange(out_h, device=low.device) >> 1).clamp(max=in_h - 1)
    src_x = (torch.arange(out_w, device=low.device) >> 1).clamp(max=in_w - 1)
    if batch > 1:
        base = low.view(batch, in_h, in_w, -1)
        return base[:, src_y][:, :, src_x]
    base = low.view(in_h, in_w, -1)
    return base[src_y][:, src_x]


class UpsampleMerge(nn.Module):
    """解码入口:最近邻 2x 上采样 + 编码跳连 * 每通道 aux(教师 upsample_residual 语义)。"""

    def __init__(self, channels):
        super().__init__()
        self.aux = nn.Parameter(torch.ones(channels))

    def forward(self, low, skip, in_w, in_h, out_w, out_h, batch=1):
        up = _upsample2(low, in_w, in_h, out_w, out_h, batch=batch)
        if batch > 1:
            merged = up + skip.view(batch, out_h, out_w, -1) * self.aux.view(1, 1, 1, -1)
            return merged.reshape(-1, merged.shape[-1])
        merged = up + skip.view(out_h, out_w, -1) * self.aux.view(1, 1, -1)
        return merged.reshape(-1, merged.shape[-1])


class PostBlend(nn.Module):
    """block 70 前的最后合并:d0 解码输出上采样 *aux0 + full 级跳连 *aux1(2.7)。"""

    def __init__(self, channels):
        super().__init__()
        self.aux_pair = nn.Parameter(torch.ones(2, channels))

    def forward(self, low, skip, in_w, in_h, out_w, out_h, batch=1):
        if batch > 1:
            up = _upsample2(low, in_w, in_h, out_w, out_h, batch=batch) * self.aux_pair[0].view(1, 1, 1, -1)
            merged = up + skip.view(batch, out_h, out_w, -1) * self.aux_pair[1].view(1, 1, 1, -1)
            return merged.reshape(-1, merged.shape[-1])
        up = _upsample2(low, in_w, in_h, out_w, out_h) * self.aux_pair[0].view(1, 1, -1)
        merged = up + skip.view(out_h, out_w, -1) * self.aux_pair[1].view(1, 1, -1)
        return merged.reshape(-1, merged.shape[-1])


# ---------------------------------------------------------------- MoE(2026-10-10)
# top-1 硬路由、窗级(UNet)/Token 级(ViT)分组,整块专家化(FFN expand/narrow 与注意力
# qkv/proj 同一专家);计算量 = 单专家 = 与 dense 块逐像素相同,参数 ×E(容量杠杆)。
# Switch 式门控:softmax 概率乘 FFN 输出,给路由器留梯度;负载均衡走辅助损失
# (StudentNetwork.moe_aux_loss)。窗序布局不变 —— 引擎侧按窗分组 grouped GEMM 即可。

def _moe_group(assign, experts):
    """排序分组簿记:order(按专家聚序)、inv(还原置换)、bounds(逐专家段)。
    2026-10-10:替换逐专家 mask/gather/scatter(每步上万微 kernel + 同步点,吃掉
    GPU 一半利用率)-> 一次 argsort + 整段 matmul;逐行独立,结果与暴力参考一致。"""
    order = torch.argsort(assign, stable=True)
    counts = torch.bincount(assign, minlength=experts).tolist()
    bounds, acc = [], 0
    for n in counts:
        bounds.append((acc, acc + n))
        acc += n
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return order, inv, bounds


def _moe_mm(x_sorted, weight, bounds):
    """分段 matmul(x 已按专家聚序):x[s:t] · weight[e];空段 [0,in]@[in,out] 无害。"""
    return torch.cat([x_sorted[s:t] @ weight[e] for e, (s, t) in enumerate(bounds)], dim=0)


def _balanced_pick(scores):
    """配额平衡 top-1(塌缩的结构解):每专家名额 quota=⌈W/E⌉,均衡由构造保证。

    初始 argmax 后,超载专家把"margin 最弱"的窗让给"该窗分数最高"的欠载专家;
    W<E 的深块天然只用 ⌈W/E⌉ 个专家(结构,非塌缩)。确定性(stable 排序)。
    2026-10-10:Switch aux 在小 logits 区间失明(p≈均匀 ⇒ aux≡1 ⇒ 零梯度),
    用它防塌缩在数学上就不成立 —— 改为结构保证,专业化由分数驱动(梯度走门控)。
    """
    w_count, e_count = scores.shape
    quota = -(-w_count // e_count)
    pick = scores.argmax(-1)
    count = torch.bincount(pick, minlength=e_count)
    over = torch.nonzero(count > quota).flatten()
    if over.numel() == 0:
        return pick
    alt = scores.scatter(1, pick[:, None], float('-inf'))
    best_alt = alt.argmax(-1)
    margin = (scores.gather(1, pick[:, None])
              - scores.gather(1, best_alt[:, None])).squeeze(1)
    victims = []
    for e in over.tolist():
        mine = torch.nonzero(pick == e).flatten()
        k = int(count[e] - quota)
        weak = mine[torch.argsort(margin[mine], stable=True)[:k]]
        victims.append(weak)
    victims = torch.cat(victims)
    under = count < quota
    under_list = torch.nonzero(under).flatten()
    seats = torch.repeat_interleave(under_list, (quota - count[under_list]).clamp_min(0))
    # 受害者 = margin 最弱的窗(对去处最不敏感),按座位表精确填充 -> 均衡严格保证
    order = victims[torch.argsort(margin[victims], stable=True)]
    n = min(order.numel(), seats.numel())
    pick[order[:n]] = seats[:n]
    return pick


def _window_route(state, width, height, router, batch=1):
    """8x8 窗均值池化 -> router -> top-1;展开 rows 级 assign/gates。

    深层尺寸只保证 4 对齐(20x20、12x12…),末窗不满:越界槽屏蔽、按有效槽求均值
    (分组与 WindowAttention 的窗网格一致,但路由用不移位网格即可)。
    训练期 logit 加抖动(±0.1)防赢家通吃塌缩;eval 确定性不受影响。
    batch>1:样本按行主序叠放;窗网格逐样本、配额平衡按样本内窗集统计(batch=1
    口径逐窗一致,批大小不改路由语义)。
    """
    ch = state.shape[-1]
    win = 8
    nw = (width + win - 1) // win
    nh = (height + win - 1) // win
    device = state.device
    ys = torch.arange(nh * win, device=device)
    xs = torch.arange(nw * win, device=device)
    vmask = ((ys < height).view(-1, 1) & (xs < width).view(1, -1)).view(nh, win, nw, win)
    idx = (ys.clamp(0, height - 1).view(-1, 1) * width
           + xs.clamp(0, width - 1).view(1, -1)).reshape(-1)
    count = vmask.sum((1, 3)).clamp_min(1).to(state.dtype).unsqueeze(-1)   # [nh, nw, 1]
    if batch > 1:
        idx = (idx.view(1, -1)
               + (torch.arange(batch, device=device) * (width * height)).view(-1, 1)).reshape(-1)
        gathered = state[idx].view(batch, nh, win, nw, win, ch)
        pooled = (gathered * vmask.view(1, nh, win, nw, win, 1)).sum((2, 4))
        pooled = (pooled / count.view(1, nh, nw, 1)).reshape(-1, ch)
    else:
        gathered = state[idx].view(nh, win, nw, win, ch)
        pooled = (gathered * vmask.unsqueeze(-1)).sum((1, 3)) / count          # [nh, nw, ch]
    logits = router(pooled.reshape(-1, ch))              # [batch*nwin, E]
    if router.training:
        logits = logits + torch.rand_like(logits) * 0.2 - 0.1
    probs = torch.softmax(logits, dim=-1)
    if batch > 1:
        # 配额平衡按样本内窗集统计(与 batch=1 口径逐窗一致,批大小不改路由语义)。
        nwin = nh * nw
        pick = torch.cat([_balanced_pick(logits[b * nwin:(b + 1) * nwin])
                          for b in range(batch)], 0)
    else:
        pick = _balanced_pick(logits)                      # 配额平衡 top-1(结构防塌缩)
    g = probs.gather(1, pick[:, None]).squeeze(1)        # 门控标量(×E 归一:均匀时=1)
    win_id = ((torch.arange(height, device=device) // win).view(-1, 1) * nw
              + (torch.arange(width, device=device) // win).view(1, -1)).reshape(-1)
    if batch > 1:
        win_id = (win_id.view(1, -1)
                  + (torch.arange(batch, device=device) * (nh * nw)).view(-1, 1)).reshape(-1)
    return pick[win_id], g[win_id], probs, logits


class MoEBlock(nn.Module):
    """top-1 窗路由专家块:FFN(expand->SiLU->narrow,门控)+ 窗注意力(qkv/proj 同专家)。

    外契约与 FFNBlock 一致:残差 + aux、注意力尾同 _AttentionTail 的数值路径。
    """

    def __init__(self, channels, hidden, experts=8, attn=True, window=WINDOW, phase_index=0):
        super().__init__()
        self.experts = experts
        self.channels = channels
        # router 前置 LayerNorm:logits 幅值有界,防 softmax 饱和(z 失控的根因之一)。
        self.router = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, experts, bias=False))
        self.expand = nn.Parameter(torch.empty(experts, channels, hidden))
        self.narrow = nn.Parameter(torch.empty(experts, hidden, channels))
        nn.init.kaiming_uniform_(self.expand, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.narrow, a=math.sqrt(5))
        self.aux_ffn = nn.Parameter(torch.ones(channels))
        self.moe_stats = []
        self.attn = attn
        if attn:
            self.qkv = nn.Parameter(torch.empty(experts, channels, channels * 3))
            self.proj = nn.Parameter(torch.empty(experts, channels, channels))
            nn.init.kaiming_uniform_(self.qkv, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.proj, a=math.sqrt(5))
            self.attention = WindowAttention(channels, window, window_phase(phase_index))
            self.aux_attn = nn.Parameter(torch.ones(channels))

    def forward(self, state, width, height, batch=1):
        assign, gates, probs, logits = _window_route(state, width, height, self.router, batch)
        gates = gates * self.experts                        # 归一化门控:均匀路由时 = 1(dense 语义)
        self.moe_stats.append((assign.detach(), probs, logits))
        order, inv, bounds = _moe_group(assign, self.experts)
        gs = gates[order]
        h = F.silu(_moe_mm(state[order], self.expand, bounds))
        ffn = (_moe_mm(h, self.narrow, bounds) * gs.unsqueeze(1)).to(h.dtype)
        out = ffn[inv] + state * self.aux_ffn
        if self.attn:
            qkv = _moe_mm(out[order], self.qkv, bounds)[inv]
            attended = self.attention(qkv, width, height, batch)
            out = out + _moe_mm(attended[order], self.proj, bounds)[inv] * self.aux_attn
        return out


class MoEVitBlock(nn.Module):
    """Token 级 top-1 专家 ViT 块(外契约同 VitBlock:FFN 残差 -> 全 token 注意力残差)。"""

    def __init__(self, channels, ffn_channels, experts=8):
        super().__init__()
        self.experts = experts
        self.channels = channels
        self.router = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, experts, bias=False))
        self.expand = nn.Parameter(torch.empty(experts, channels, ffn_channels))
        self.contract = nn.Parameter(torch.empty(experts, ffn_channels, channels))
        nn.init.kaiming_uniform_(self.expand, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.contract, a=math.sqrt(5))
        self.aux_ffn = nn.Parameter(torch.ones(channels))
        self.qkv = nn.Parameter(torch.empty(experts, channels, channels * 3))
        self.proj = nn.Parameter(torch.empty(experts, channels, channels))
        nn.init.kaiming_uniform_(self.qkv, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proj, a=math.sqrt(5))
        self.attention = VitAttention(channels)
        self.aux_attn = nn.Parameter(torch.ones(channels))
        self.moe_stats = []

    def forward(self, state, tokens, padded):
        logits = self.router(state)                              # [rows, E] token 级
        if self.router.training:
            logits = logits + torch.rand_like(logits) * 0.2 - 0.1
        probs = torch.softmax(logits, dim=-1)
        pick = _balanced_pick(logits)                              # 配额平衡 top-1
        gates = probs.gather(1, pick[:, None]).squeeze(1) * self.experts   # 归一化门控
        self.moe_stats.append((pick.detach(), probs, logits))
        order, inv, bounds = _moe_group(pick, self.experts)
        gs = gates[order]
        h = F.silu(_moe_mm(state[order], self.expand, bounds))
        ffn = (_moe_mm(h, self.contract, bounds) * gs.unsqueeze(1)).to(h.dtype)
        out = ffn[inv] + state * self.aux_ffn
        qkv = _moe_mm(out[order], self.qkv, bounds)[inv]
        attended = self.attention(qkv, tokens, padded)
        return out + _moe_mm(attended[order], self.proj, bounds)[inv] * self.aux_attn


def make_block(level, phase_index):
    """按形状条目构造块(2.3-2.6);相位索引即教师的每级计数器值。"""
    kind = level['kind']
    ch = level['channels']
    attn = level.get('attn', True)
    window = level.get('window_slots', WINDOW)
    if kind == 'shared_routed_moe':
        from nr_shared_moe import SharedRoutedBlock
        return SharedRoutedBlock(ch, level['hidden'], attn=attn, phase_index=phase_index)
    if kind == 'ffn':
        return FFNBlock(ch, level['hidden'], attn=attn, window=window, phase_index=phase_index)
    if kind == 'expert':
        return ExpertBlock(ch, level['hidden'], attn=attn, window=window, phase_index=phase_index)
    if kind == 'moe':
        return MoEBlock(ch, level['hidden'], experts=level['experts'], attn=attn,
                        window=window, phase_index=phase_index)
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
        from moe_shape import is_shared, validate
        self.shared_routed = is_shared(shape)
        if 'architecture' in shape or any(lv['kind'] == 'shared_routed_moe' for lv in shape['levels']):
            validate(shape)
        if self.shared_routed:
            from nr_shared_moe import SharedRoutedLinear, SharedRoutedBlock
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
        transition = SharedRoutedLinear if self.shared_routed else nn.Linear
        self.enc_trans = nn.ModuleDict({
            name: transition(self.channels[low], self.channels[name])
            for low, name in zip(order, order[1:])})
        # 解码投影方向相反:高层(更大) -> 本层。
        self.dec_trans = nn.ModuleDict({
            name: transition(self.channels[higher], self.channels[name])
            for name, higher in (('d3', 'd4'), ('d2', 'd3'), ('d1', 'd2'), ('d0', 'd1'))})
        self.merge = nn.ModuleDict({name: UpsampleMerge(self.channels[name])
                                    for name in order[1:]})
        vit_ch = shape['vit']['channels']
        self.vit_in = (transition(self.channels['d4'], vit_ch, token=True) if self.shared_routed
                       else transition(self.channels['d4'], vit_ch))
        self.vit_out = (transition(vit_ch, self.channels['d4'], token=True) if self.shared_routed
                        else transition(vit_ch, self.channels['d4']))
        vit_experts = int(shape['vit'].get('experts', 0))
        self.vit_blocks = nn.ModuleList(
            (SharedRoutedBlock(vit_ch, shape['vit']['ffn'], token=True) if self.shared_routed else
             MoEVitBlock(vit_ch, shape['vit']['ffn'], experts=vit_experts) if vit_experts >= 2
             else VitBlock(vit_ch, shape['vit']['ffn'])) for _ in range(shape['vit']['blocks']))
        self.post = PostBlend(self.channels['full'])

        self.captures = {}
        self.capture_enabled = False

    def capture(self, name, tensor):
        if self.capture_enabled:
            self.captures[name] = tensor

    def transition(self, module, state, width, height, batch):
        return module(state, width, height, batch) if self.shared_routed else module(state)

    def moe_aux_loss(self):
        """Switch 负载均衡辅助损失:E * Σ_e f_e * mean(p_e)(f 常数、p 可导);无 MoE 块=0。"""
        stats = [s for m in self.modules() if hasattr(m, 'moe_stats') for s in m.moe_stats]
        if not stats:
            return torch.zeros((), device=self.blend_scale.device)
        total = self.blend_scale.new_zeros(())
        for assign, probs, _logits in stats:
            e_count = probs.shape[-1]
            f = torch.bincount(assign, minlength=e_count).to(probs.dtype) / assign.numel()
            total = total + e_count * (f * probs.mean(0)).sum()
        return total / len(stats)

    def moe_z_loss(self):
        """New shared/routed: ST-MoE logsumexp²; legacy preserves old logits L2."""
        stats = [s for m in self.modules() if hasattr(m, 'moe_stats') for s in m.moe_stats]
        if not stats:
            return torch.zeros((), device=self.blend_scale.device)
        total = self.blend_scale.new_zeros(())
        for _assign, _probs, logits in stats:
            total = total + (torch.logsumexp(logits.float(), dim=-1).square().mean()
                             if self.shared_routed else (logits ** 2).mean())
        return total / len(stats)

    def forward(self, features, batch=1):
        """batch>1(2026-10-12 吞吐):features 为样本行主序叠放 ((batch*rows, 16)),
        输出同布局 ((batch*rows, 4));ViT 在瓶颈 token 数小,逐样本跑(全局注意力
        跨样本互见的掩码复杂度不值当)。batch=1 与原实现逐位一致。"""
        for m in self.modules():
            if hasattr(m, 'moe_stats'):
                m.moe_stats.clear()
            if hasattr(m, 'training_metric_records'):
                m.training_metric_records.clear()
        g = self.g
        full_w, full_h = g['full_width'], g['full_height']
        order = ('full', 'd0', 'd1', 'd2', 'd3', 'd4')

        # ---- 编码:full -> d4,每级块后池化 + 展宽(full->d0 也有学习变换,便宜且更自由)。
        state = self.adapter(features)
        skips = {}
        for i, name in enumerate(order):
            width, height = level_size(g, name)
            for j, block in enumerate(self.enc_blocks[name]):
                state = block(state, width, height, batch)
                if j == len(self.enc_blocks[name]) - 1:
                    self.capture(f's-enc-{name}', state)
            skips[name] = state
            if name != 'd4':
                next_name = order[i + 1]
                nw, nh = level_size(g, next_name)
                state = self.transition(self.enc_trans[next_name],
                    box_downsample(state, width, height, nw, nh, batch=batch), nw, nh, batch)

        # ---- ViT(d5 tokens)。
        tokens = g['vit_tokens']
        padded = g['padded_vit_tokens']
        dw, dh = level_size(g, 'd5')
        state = self.transition(self.vit_in,
            box_downsample(state, *level_size(g, 'd4'), dw, dh, batch=batch), dw, dh, batch)
        if batch > 1:
            per_sample = state.view(batch, tokens, -1)
            outs = []
            for b in range(batch):
                x = per_sample[b]
                for block in self.vit_blocks:
                    x = block(x, tokens, padded)
                outs.append(x)
            state = torch.cat(outs, 0)
        else:
            for block in self.vit_blocks:
                state = block(state, tokens, padded)
        self.capture('s-vit', state)

        # ---- 解码:d5 -> d4 -> d3 -> d2 -> d1 -> d0 -> full。
        state = self.transition(self.vit_out, state, dw, dh, batch)
        prev_name, prev_w, prev_h = 'd5', dw, dh
        for name in ('d4', 'd3', 'd2', 'd1', 'd0'):
            width, height = level_size(g, name)
            if prev_name != 'd5':
                state = self.transition(self.dec_trans[name], state, prev_w, prev_h, batch)
            state = self.merge[name](state, skips[name], prev_w, prev_h, width, height,
                                     batch=batch)
            for block in self.dec_blocks[name]:
                state = block(state, width, height, batch)
            self.capture(f's-dec-{name}', state)
            prev_name, prev_w, prev_h = name, width, height

        state = self.post(state, skips['full'], prev_w, prev_h, full_w, full_h, batch=batch)
        for block in self.dec_blocks['full']:
            state = block(state, full_w, full_h, batch)
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
