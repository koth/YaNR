"""FP32 reference for shared-routed-moe-v1; no quotas, drops, or routing jitter.

This deliberately simple grouped implementation is a correctness/training
reference. Native CUDA uses device-resident compact groups instead of Python
counts. Routing is per sample's unshifted window, or per ViT token.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def route_tile_rows(rows):
    """Bound padding by the actual problem size, particularly deep ViT groups.

    At most 15 partial tiles: padding < 16*tile, instead of imposing 4096
    rows even on a group with one token. Sizes depend only on the input shape.
    """
    return min(4096, max(8, 1 << max(0, (max(1, rows // 16)).bit_length() - 1)))


def row_counts(assign):
    # Fixed output shape; CUDA bincount/repeat_interleave otherwise need a host
    # read to discover their output size, even with minlength=16.
    return torch.zeros(16, dtype=torch.int64, device=assign.device).scatter_add_(
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
    """Selection uses frozen bias; differentiable gate uses unbiased FP32 logits."""
    if logits.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError('shared/routed router logits and bias must be FP32')
    probabilities = logits.softmax(-1)
    pick = (logits + bias).argmax(-1)  # torch argmax: first (smallest) index on ties.
    gate = 16 * probabilities.gather(1, pick[:, None]).squeeze(1)
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
    def __init__(self, channels, token=False):
        super().__init__()
        self.router = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 16, bias=False))
        self.register_buffer('router_bias', torch.zeros(16))
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
        self.moe_stats.append((assign.detach(), probs, logits))
        for name, tensor in [('logits', logits), ('pick', pick), ('gate', gate), ('assign', assign)]:
            self.capture(name, tensor)
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
        return order, inv, lay, gates[:, None]

    @staticmethod
    def routed_mm(state, bank, route, bias=None):
        order, inv, lay, _ = route
        sorted_x = _Permutation.apply(state, order, inv)
        out_f = bank.shape[2]
        # 固定瓦片分组 GEMM(2026-10-17 向量化):变长专家组折算进稠密 [B, TILE, k],
        # 16 专家一次 bmm;替代逐专家循环(33 碎内核/次 -> 8,无 Python 循环派发)。
        buf = _TilePack.apply(sorted_x, lay['packed_rows'], lay['packed_valid'], lay['flat_slots'])
        weights = _ExpertWeights.apply(bank[1:17], lay['expert_of_tile'], lay['tiles'])
        out = torch.bmm(buf.view(lay['B'], lay['TILE'], state.shape[1]), weights).reshape(-1, out_f)
        seg = _TileUnpack.apply(out, lay['flat_slots'], lay['packed_rows'], lay['packed_valid'])
        if bias is not None:
            seg = seg + bias[1:17][lay['expert_per_row']]
        return _Permutation.apply(seg, inv, order)

    def mixed_mm(self, state, bank, route, bias=None):
        shared, routed = self.expert_pair(state, state, bank, route, bias)
        if self.training_metrics_enabled and self.training and torch.is_grad_enabled():
            with torch.no_grad():
                self.training_metrics.update(shared_projection_rms=shared.square().mean().sqrt(),
                                             routed_projection_rms=routed.square().mean().sqrt())
        shared, routed = self.diagnostic_branches(shared, routed)
        return (shared + route[3] * routed) * 0.5

    @staticmethod
    def expert_pair(shared_state, routed_state, bank, route, bias=None):
        shared = shared_state @ bank[0]
        if bias is not None:
            shared = shared + bias[0]
        return shared, RoutedModule.routed_mm(routed_state, bank, route, bias)

    def diagnostic_branches(self, shared, routed):
        if self.branch_mode != 'full' and self.training:
            raise ValueError('branch removal is an eval-only diagnostic')
        if self.branch_mode == 'remove_shared':
            shared = torch.zeros_like(shared)
        elif self.branch_mode == 'remove_routed':
            routed = torch.zeros_like(routed)
        elif self.branch_mode != 'full':
            raise ValueError('unknown branch diagnostic mode')
        return shared, routed


def initialize_bank(bank):
    # Initialize each expert as an independent ordinary Linear, not a 3-D fan-in.
    bound = 1 / math.sqrt(bank.shape[1])
    nn.init.uniform_(bank, -bound, bound)


class SharedRoutedLinear(RoutedModule):
    def __init__(self, inputs, outputs, token=False):
        super().__init__(inputs, token)
        self.weight = nn.Parameter(torch.empty(17, inputs, outputs))
        self.bias = nn.Parameter(torch.empty(17, outputs))
        initialize_bank(self.weight)
        nn.init.uniform_(self.bias, -1 / math.sqrt(inputs), 1 / math.sqrt(inputs))

    def forward(self, state, width, height, batch=1):
        route = self.route(state, width, height, batch)
        out = self.mixed_mm(state, self.weight, route, self.bias)
        self.capture('output', out)
        return out


class SharedRoutedBlock(RoutedModule):
    def __init__(self, channels, hidden, attn=True, phase_index=0, token=False):
        super().__init__(channels, token)
        from nr_student import WindowAttention, VitAttention
        from nr_geometry import window_phase
        self.expand = nn.Parameter(torch.empty(17, channels, hidden))
        name = 'contract' if token else 'narrow'
        self.register_parameter(name, nn.Parameter(torch.empty(17, hidden, channels)))
        self.contract_name = name
        initialize_bank(self.expand)
        initialize_bank(getattr(self, name))
        self.aux_ffn = nn.Parameter(torch.ones(channels))
        self.attn = attn
        if attn:
            self.qkv = nn.Parameter(torch.empty(17, channels, 3 * channels))
            self.proj = nn.Parameter(torch.empty(17, channels, channels))
            initialize_bank(self.qkv); initialize_bank(self.proj)
            self.aux_attn = nn.Parameter(torch.ones(channels))
            self.attention = (VitAttention(channels) if token else
                              WindowAttention(channels, 8, window_phase(phase_index)))

    def forward(self, state, width, height, batch=1):
        # For token blocks width=tokens, height=padded, matching existing ViT API.
        route = self.route(state, width, height, batch)
        contract = getattr(self, self.contract_name)
        shared_hidden, routed_hidden = self.expert_pair(state, state, self.expand, route)
        shared, routed = self.expert_pair(F.silu(shared_hidden), F.silu(routed_hidden), contract, route)
        if self.training_metrics_enabled and self.training and torch.is_grad_enabled():
            with torch.no_grad():
                self.training_metrics.update(shared_ffn_rms=shared.square().mean().sqrt(),
                                             routed_ffn_rms=routed.square().mean().sqrt(),
                                             gated_routed_ffn_rms=(route[3] * routed).square().mean().sqrt())
        shared, routed = self.diagnostic_branches(shared, routed)
        ffn = (shared + route[3] * routed) * 0.5
        self.capture('ffn_shared', shared); self.capture('ffn_routed', routed)
        out = state * self.aux_ffn + ffn
        self.capture('ffn', out)
        if self.attn:
            qkv = self.mixed_mm(out, self.qkv, route)
            self.capture('qkv', qkv)
            attended = (self.attention(qkv, width, height) if self.token else
                        self.attention(qkv, width, height, batch))
            self.capture('attention', attended)
            projection = self.mixed_mm(attended, self.proj, route)
            out = out + self.aux_attn * projection
        self.capture('output', out)
        return out
