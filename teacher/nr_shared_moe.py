"""FP32 reference for shared-routed-moe-v1; no quotas, drops, or routing jitter.

This deliberately simple grouped implementation is a correctness/training
reference. Native CUDA uses device-resident compact groups instead of Python
counts. Routing is per sample's unshifted window, or per ViT token.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


ROUTED_EXPERTS = 7    # 2026-10-18:去掉共享专家后,bank 轴 0..6 全是路由专家
TOP_K = 1             # 每 token 激活一个专家
# 旧设计是一支线性 shared(零均值)+ 一支 silu 专家,融合 (shared + g·E)·0.5;
# 现在两支都是 silu 专家(silu 非零均值,相加会把直流偏置翻倍,实测 0.5 时开局
# psnr 只有 6.9dB/gnorm 66)。按"激活专家数 × 系数 = 0.5"取,换 TOP_K 量级不跳变:
# top-2 -> 0.25(实测 step0 psnr 19.8 正常),top-1 -> 0.5。
FUSION = 0.5 / TOP_K


def route_tile_rows(rows):
    """Bound padding by the actual problem size, particularly deep ViT groups.

    At most 15 partial tiles: padding < 16*tile, instead of imposing 4096
    rows even on a group with one token. Sizes depend only on the input shape.
    """
    # Shape guards work with Dynamo's SymInt; Python int.bit_length() does not.
    for tile in (4096, 2048, 1024, 512, 256, 128, 64, 32, 16):
        if rows >= 16 * tile:
            return tile
    return 8


def row_counts(assign):
    # Fixed output shape; CUDA bincount/repeat_interleave otherwise need a host
    # read to discover their output size, even with minlength=16.
    return torch.zeros(ROUTED_EXPERTS, dtype=torch.int64, device=assign.device).scatter_add_(
        0, assign, torch.ones_like(assign))


class _Permutation(torch.autograd.Function):
    """The transpose of a known permutation is its inverse, never an atomic sum."""
    @staticmethod
    def forward(ctx, state, order, inverse):
        ctx.save_for_backward(inverse)
        return state.index_select(0, order)

    @staticmethod
    def backward(ctx, grad):
        inverse, = ctx.saved_tensors
        return grad.index_select(0, inverse), None, None


class _TilePack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, packed_rows, valid, slots):
        ctx.save_for_backward(slots)
        return state.index_select(0, packed_rows) * valid[:, None]

    @staticmethod
    def backward(ctx, grad):
        slots, = ctx.saved_tensors
        return grad.index_select(0, slots), None, None, None


class _TileUnpack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, slots, packed_rows, valid):
        ctx.save_for_backward(packed_rows, valid)
        return state.index_select(0, slots)

    @staticmethod
    def backward(ctx, grad):
        packed_rows, valid = ctx.saved_tensors
        return grad.index_select(0, packed_rows) * valid[:, None], None, None, None


class _ExpertWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bank, experts, tiles):
        ctx.save_for_backward(tiles)
        return bank.index_select(0, experts)

    @staticmethod
    def backward(ctx, grad):
        tiles, = ctx.saved_tensors
        # Tiles are contiguous and sorted by expert. This ordered reduction
        # replaces generic repeated-index backward (sort + atomic/scatter).
        return torch.segment_reduce(grad, 'sum', lengths=tiles, axis=0, unsafe=True), None, None


def select(logits, bias):
    """Selection uses frozen bias; differentiable gate uses unbiased FP32 logits.

    2026-10-18:去掉共享专家、改为 top-2。每个被选专家的门控仍沿用
    gate = 16 * softmax(z)[e](与 top-1 同一口径),这样两个专家各按自身置信度
    进入,路由支路幅度大约是原来的两倍 —— 正是为了解决"共享支路在 d3/d4
    占到 62-74% 却不专门化"的实测问题。

    选择用 argsort(stable=True) 而不是 topk:契约要求并列时取最小索引,
    torch.topk 不保证这一点。
    """
    if logits.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError('shared/routed router logits and bias must be FP32')
    probabilities = logits.softmax(-1)
    ranking = torch.argsort(-(logits + bias), dim=-1, stable=True)
    pick = ranking[:, :TOP_K]                       # [rows, 2],并列取最小索引
    gate = ROUTED_EXPERTS * probabilities.gather(1, pick)   # 均匀路由时 =1
    return pick, gate, probabilities


def pool_windows(state, width, height, batch):
    channels = state.shape[-1]
    nw, nh = (width + 7) // 8, (height + 7) // 8
    grid = state.view(batch, height, width, channels)
    padded = F.pad(grid.permute(0, 3, 1, 2), (0, nw * 8 - width, 0, nh * 8 - height))
    pooled = padded.view(batch, channels, nh, 8, nw, 8).sum((3, 5))
    ys = (height - torch.arange(nh, device=state.device) * 8).clamp(1, 8)
    xs = (width - torch.arange(nw, device=state.device) * 8).clamp(1, 8)
    pooled = (pooled / (ys[:, None] * xs[None, :])).permute(0, 2, 3, 1).reshape(-1, channels)
    row_window = ((torch.arange(height, device=state.device) // 8)[:, None] * nw +
                  (torch.arange(width, device=state.device) // 8)[None, :]).reshape(-1)
    row_window = (row_window[None, :] + torch.arange(batch, device=state.device)[:, None] * nw * nh).reshape(-1)
    return pooled, row_window


class RoutedModule(nn.Module):
    split_token_stats = False

    def __init__(self, channels, token=False):
        super().__init__()
        self.router = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, ROUTED_EXPERTS, bias=False))
        self.register_buffer('router_bias', torch.zeros(ROUTED_EXPERTS))
        self.token = token
        self.moe_stats = []
        self.capture_enabled = False
        self.capture_filter = None
        self.captures = {}
        self.training_metrics = {}
        self.training_metric_records = []
        self.training_metrics_enabled = False
        self.branch_mode = 'full'  # eval-only diagnostic; training must use full

    def capture(self, name, value):
        if self.capture_enabled and (self.capture_filter is None or name in self.capture_filter):
            self.captures[name] = value.detach()

    def route(self, state, width, height, batch):
        if state.dtype != torch.float32:
            raise TypeError('shared-routed-moe-v1 reference requires FP32 state')
        self.captures.clear()
        self.training_metrics = {}
        pooled, row_window = (state, None) if self.token else pool_windows(state, width, height, batch)
        # Disabling autocast is intentional: router selection/gate always remain FP32.
        with torch.autocast(device_type=state.device.type, enabled=False):
            logits = self.router(pooled.float())
            pick, gate, probs = select(logits, self.router_bias)
        assign, gates = (pick, gate) if row_window is None else (pick[row_window], gate[row_window])
        if self.training_metrics_enabled and self.training and torch.is_grad_enabled():
            self.training_metric_records.append(self.training_metrics)
            with torch.no_grad():
                top = logits.topk(2, dim=1).values
                self.training_metrics.update(gate_mean=gates.mean(), gate_max=gates.max(),
                                             score_margin_mean=(top[:, 0] - top[:, 1]).mean())
        # 负载统计按"每个被选中的(行, 专家)对"记:top-2 时一行贡献两条。
        flat_assign = assign.reshape(-1)
        if self.token and self.split_token_stats and batch > 1:
            # Keep the existing per-sample auxiliary-loss population, even
            # though the projections/attention now execute in one batch.
            self.moe_stats.extend(zip(assign.detach().chunk(batch), probs.chunk(batch), logits.chunk(batch)))
        else:
            self.moe_stats.append((flat_assign.detach(), probs, logits))
        for name, tensor in [('logits', logits), ('pick', pick), ('gate', gate), ('assign', assign)]:
            self.capture(name, tensor)
        primary = self._pack_route(assign[:, 0], gates[:, 0])
        if TOP_K == 1:
            return primary
        return (*primary, self._pack_route(assign[:, 1], gates[:, 1]))

    @staticmethod
    def _pack_route(assign, gate):
        """把一维 assign 编成固定瓦片分组布局;top-2 时对两个 pick 各调一次。

        返回 (order, inv, lay, gate[:, None]),`routed_mm` 直接消费。
        """
        # One scalar size read per route; reuse its compact layout in all four
        # projections. Never impose a 4096-row minimum on small/deep layers.
        order = torch.argsort(assign, stable=True)
        counts = row_counts(assign)
        tile = route_tile_rows(assign.numel())
        tiles = (counts + tile - 1) // tile
        tile_base = torch.cumsum(tiles, 0) - tiles
        group_start = torch.cumsum(counts, 0) - counts
        dev = assign.device
        expert = assign[order]
        pos = torch.arange(assign.numel(), device=dev) - group_start[expert]
        tile_id = tile_base[expert] + pos // tile
        tile_count = int(tiles.sum())
        tile_experts = torch.searchsorted(tile_base + tiles, torch.arange(tile_count, device=dev), right=True)
        packed = torch.arange(tile_count * tile, device=dev)
        packed_expert = tile_experts[packed // tile]
        packed_pos = (packed // tile - tile_base[packed_expert]) * tile + packed % tile
        packed_valid = packed_pos < counts[packed_expert]
        packed_rows = (group_start[packed_expert] + packed_pos).clamp_max(assign.numel() - 1)
        lay = dict(TILE=tile, B=tile_count, tiles=tiles, tile_id=tile_id, slot=pos % tile,
                   packed_rows=packed_rows, packed_valid=packed_valid, flat_slots=tile_id * tile + pos % tile,
                   expert_of_tile=tile_experts,
                   expert_per_row=expert)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        return order, inv, lay, gate[:, None]

    @staticmethod
    def routed_mm(state, bank, route, bias=None):
        # route 可能是 5 元组(top-2 的整体路由)或 4 元组(单个槽位),取前四项。
        order, inv, lay, _ = route[:4]
        sorted_x = _Permutation.apply(state, order, inv)
        out_f = bank.shape[2]
        # 固定瓦片分组 GEMM(2026-10-17 向量化):变长专家组折算进稠密 [B, TILE, k],
        # 16 专家一次 bmm;替代逐专家循环(33 碎内核/次 -> 8,无 Python 循环派发)。
        buf = _TilePack.apply(sorted_x, lay['packed_rows'], lay['packed_valid'], lay['flat_slots'])
        weights = _ExpertWeights.apply(bank[0:ROUTED_EXPERTS], lay['expert_of_tile'], lay['tiles'])
        out = torch.bmm(buf.view(lay['B'], lay['TILE'], state.shape[1]), weights).reshape(-1, out_f)
        seg = _TileUnpack.apply(out, lay['flat_slots'], lay['packed_rows'], lay['packed_valid'])
        if bias is not None:
            seg = seg + bias[0:ROUTED_EXPERTS][lay['expert_per_row']]
        return _Permutation.apply(seg, inv, order)

    def mixed_mm(self, state, bank, route, bias=None):
        first, second = self.expert_pair(state, bank, route, bias)
        if self.training_metrics_enabled and self.training and torch.is_grad_enabled():
            with torch.no_grad():
                self.training_metrics.update(
                    shared_projection_rms=first.square().mean().sqrt(),
                    routed_projection_rms=second.square().mean().sqrt())
        first, second = self.diagnostic_branches(first, second)
        if TOP_K > 1:
            return (route[3] * first + route[4][3] * second) * FUSION
        return route[3] * first * FUSION

    @staticmethod
    def expert_pair(state, bank, route, bias=None):
        """top-2、无共享专家:返回两个被选专家各自的输出。

        2026-10-18 去掉共享专家(bank 轴 0 现在也是路由专家)。原来的
        (F_shared + g·F_e)·0.5 里共享支路在 d3/d4 占 62-74% 却不专门化,
        实测是深层的容量黑洞。现在两路都来自被路由的专家。
        """
        first = RoutedModule.routed_mm(state, bank, route, bias)
        if TOP_K == 1:
            return first, first
        return first, RoutedModule.routed_mm(state, bank, route[4], bias)

    def diagnostic_branches(self, first, second):
        if self.branch_mode != 'full' and self.training:
            raise ValueError('branch removal is an eval-only diagnostic')
        if self.branch_mode == 'remove_shared':
            first = torch.zeros_like(first)
        elif self.branch_mode == 'remove_routed':
            second = torch.zeros_like(second)
        elif self.branch_mode != 'full':
            raise ValueError('unknown branch diagnostic mode')
        return first, second


def initialize_bank(bank):
    # Initialize each expert as an independent ordinary Linear, not a 3-D fan-in.
    bound = 1 / math.sqrt(bank.shape[1])
    nn.init.uniform_(bank, -bound, bound)


class SharedRoutedLinear(RoutedModule):
    def __init__(self, inputs, outputs, token=False):
        super().__init__(inputs, token)
        self.weight = nn.Parameter(torch.empty(ROUTED_EXPERTS, inputs, outputs))
        self.bias = nn.Parameter(torch.empty(ROUTED_EXPERTS, outputs))
        initialize_bank(self.weight)
        nn.init.uniform_(self.bias, -1 / math.sqrt(inputs), 1 / math.sqrt(inputs))

    def forward(self, state, width, height, batch=1):
        route = self.route(state, width, height, batch)
        out = self.mixed_mm(state, self.weight, route, self.bias)
        self.capture('output', out)
        return out


class SharedRoutedBlock(RoutedModule):
    split_token_stats = True

    def __init__(self, channels, hidden, attn=True, phase_index=0, token=False,
                 window=8, dilation=1):
        super().__init__(channels, token)
        from nr_student import WindowAttention, VitAttention
        from nr_geometry import window_phase
        self.expand = nn.Parameter(torch.empty(ROUTED_EXPERTS, channels, hidden))
        name = 'contract' if token else 'narrow'
        self.register_parameter(name, nn.Parameter(torch.empty(ROUTED_EXPERTS, hidden, channels)))
        self.contract_name = name
        initialize_bank(self.expand)
        initialize_bank(getattr(self, name))
        self.aux_ffn = nn.Parameter(torch.ones(channels))
        # 层内专家多样性(2026-10-18):top-2 的两个专家在同一批 token 上的
        # 隐藏激活如果高度相似,top-2 就只多花一倍算力、没有收益。这里把它们的
        # 余弦相似度记下来,由 trainer 汇总成一个惩罚项。
        # 参考:Synergistic Intra-/Cross-Layer Regularization for MoE Expert
        # Specialization(ICML 2026, arXiv 2602.14159)的 intra-layer loss。
        self.diversity_terms = []
        self.attn = attn
        if attn:
            self.qkv = nn.Parameter(torch.empty(ROUTED_EXPERTS, channels, 3 * channels))
            self.proj = nn.Parameter(torch.empty(ROUTED_EXPERTS, channels, channels))
            initialize_bank(self.qkv); initialize_bank(self.proj)
            self.aux_attn = nn.Parameter(torch.ones(channels))
            self.attention = (VitAttention(channels) if token else
                              WindowAttention(channels, window, window_phase(phase_index),
                                              dilation=dilation))

    def forward(self, state, width, height, batch=1):
        # For token blocks width=tokens, height=padded, matching existing ViT API.
        route = self.route(state, width, height, batch)      # top-2:route[4] 是第二个专家
        contract = getattr(self, self.contract_name)
        first = route
        expert_a = RoutedModule.routed_mm(
            F.silu(RoutedModule.routed_mm(state, self.expand, first)), contract, first)
        if TOP_K > 1:
            second = route[4]
            expert_b = RoutedModule.routed_mm(
                F.silu(RoutedModule.routed_mm(state, self.expand, second)), contract, second)
            # 同一批 token 上两个专家的隐藏激活相似度 —— 直接惩罚它,推动互补分工。
            if self.training and torch.is_grad_enabled():
                self.diversity_terms.append(
                    F.cosine_similarity(
                        RoutedModule.routed_mm(state, self.expand, first),
                        RoutedModule.routed_mm(state, self.expand, second), dim=-1).mean())
            if self.training_metrics_enabled and self.training and torch.is_grad_enabled():
                with torch.no_grad():
                    self.training_metrics.update(
                        shared_ffn_rms=expert_a.square().mean().sqrt(),
                        routed_ffn_rms=expert_b.square().mean().sqrt(),
                        gated_routed_ffn_rms=(second[3] * expert_b).square().mean().sqrt())
            expert_a, expert_b = self.diagnostic_branches(expert_a, expert_b)
            ffn = (first[3] * expert_a + second[3] * expert_b) * FUSION
            self.capture('ffn_shared', expert_a); self.capture('ffn_routed', expert_b)
        else:
            if self.training_metrics_enabled and self.training and torch.is_grad_enabled():
                with torch.no_grad():
                    self.training_metrics.update(
                        shared_ffn_rms=expert_a.square().mean().sqrt(),
                        routed_ffn_rms=expert_a.square().mean().sqrt(),
                        gated_routed_ffn_rms=(first[3] * expert_a).square().mean().sqrt())
            expert_a, _drop = self.diagnostic_branches(expert_a, expert_a)
            ffn = first[3] * expert_a * FUSION
            self.capture('ffn_shared', expert_a); self.capture('ffn_routed', expert_a)
        out = state * self.aux_ffn + ffn
        self.capture('ffn', out)
        if self.attn:
            qkv = self.mixed_mm(out, self.qkv, route)
            self.capture('qkv', qkv)
            attended = self.attention(qkv, width, height, batch)
            self.capture('attention', attended)
            projection = self.mixed_mm(attended, self.proj, route)
            out = out + self.aux_attn * projection
        self.capture('output', out)
        return out
