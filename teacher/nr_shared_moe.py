"""FP32 reference for shared-routed-moe-v1; no quotas, drops, or routing jitter.

This deliberately simple grouped implementation is a correctness/training
reference. Native CUDA uses device-resident compact groups instead of Python
counts. Routing is per sample's unshifted window, or per ViT token.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


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
        order = torch.argsort(assign, stable=True)
        counts = torch.bincount(assign, minlength=16).tolist()
        bounds = [0]
        for count in counts:
            bounds.append(bounds[-1] + count)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        return order, inv, bounds, gates[:, None]

    @staticmethod
    def routed_mm(state, bank, route, bias=None):
        order, inv, bounds, _ = route
        sorted_x = state[order]
        segments = []
        for e in range(16):
            result = sorted_x[bounds[e]:bounds[e + 1]] @ bank[e + 1]
            if bias is not None:
                result = result + bias[e + 1]
            segments.append(result)
        return torch.cat(segments, 0)[inv]

    def mixed_mm(self, state, bank, route, bias=None):
        shared = state @ bank[0]
        if bias is not None:
            shared = shared + bias[0]
        routed = self.routed_mm(state, bank, route, bias)
        if self.training_metrics_enabled and self.training and torch.is_grad_enabled():
            with torch.no_grad():
                self.training_metrics.update(shared_projection_rms=shared.square().mean().sqrt(),
                                             routed_projection_rms=routed.square().mean().sqrt())
        shared, routed = self.diagnostic_branches(shared, routed)
        return (shared + route[3] * routed) * 0.5

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
        shared = F.silu(state @ self.expand[0]) @ contract[0]
        hidden = F.silu(self.routed_mm(state, self.expand, route))
        routed = self.routed_mm(hidden, contract, route)
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
