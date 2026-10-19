"""Independent single-frame CNN U-Net, ~56M stored / ~7M active parameters.

The active ledger counts all public parameters and ONE expert per block and
spatial position, including biases and routers. It is not an image-wide union
of visited experts, nor a FLOP or training-memory estimate. No old model is
imported or subclassed. The 16-lane / four-channel head contract is retained;
single-frame training supervises only the RGB residual.
"""
import json
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from nr_precision import tf32_matmul


ARCHITECTURE = 'cnn-single-moe16-v1'
ORDER = ('full', 'd0', 'd1', 'd2', 'd3', 'd4', 'd5')
DECODE = tuple(reversed(ORDER[:-1]))


def load_single_shape(path):
    with open(path, encoding='utf-8') as handle:
        cfg = json.load(handle)
    if cfg.get('architecture') != ARCHITECTURE or cfg.get('experts') != 16:
        raise ValueError('expected cnn-single-moe16-v1 with 16 top-1 experts')
    for name in ORDER:
        dilations = cfg.get('dilations', {}).get(name, [1])
        if not dilations or any(int(d) != d or d < 1 for d in dilations):
            raise ValueError('dilations must be positive integers')
        if int(cfg['channels'][name]) < 2 or int(cfg['expert_hidden'][name]) < 1:
            raise ValueError('channels and expert_hidden must be positive')
        if int(cfg['blocks_enc'][name]) < 1:
            raise ValueError('each encoder stage needs at least one block')
        if name in DECODE:
            if int(cfg['blocks_dec'][name]) < 1 or cfg['merge_kernel'][name] not in (1, 3):
                raise ValueError('decoder needs blocks and a 1x1 or 3x3 merge')
    return cfg


def level_size(g, name):
    level = g if name == 'full' else g['levels'][int(name[1:])]
    prefix = 'full_' if name == 'full' else ''
    return level[prefix + 'width'], level[prefix + 'height']


@lru_cache(maxsize=64)
def _resize_axis(length, target, device, dtype):
    # A validation-first call must not cache inference tensors that autograd
    # cannot save when the same geometry is subsequently used for training.
    with torch.inference_mode(False), torch.no_grad():
        positions = (torch.arange(target, device=device, dtype=torch.float64) + .5) * (length / target) - .5
        lower = positions.floor()
        weight = (positions - lower).to(dtype)
        return lower.long().clamp(0, length - 1), (lower.long() + 1).clamp(0, length - 1), weight


def resize_bilinear(x, size):
    """Half-pixel bilinear resize with deterministic CUDA backward.

    Native interpolate(bilinear) backward throws under strict deterministic
    CUDA mode. Separable index_select uses PyTorch's deterministic reduction
    path; indices/weights are cached, carry no learned parameters or gradients.
    """
    for axis, target in ((-1, size[1]), (-2, size[0])):
        if x.shape[axis] == target:
            continue
        lower, upper, weight = _resize_axis(x.shape[axis], target, x.device, x.dtype)
        dims = [1] * x.ndim
        dims[axis] = target
        x = torch.lerp(x.index_select(axis, lower), x.index_select(axis, upper), weight.reshape(dims))
    return x


class PixelNorm(nn.Module):
    """LayerNorm over channels at each pixel; no spatial/batch statistics."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        y = F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],),
                         self.weight, self.bias, 1e-5)
        return y.permute(0, 3, 1, 2)


class Top1FFN(nn.Module):
    def __init__(self, channels, hidden, experts=16):
        super().__init__()
        self.experts = experts
        self.router = nn.Conv2d(channels, experts, 1)
        self.bank = nn.ModuleList([
            nn.Sequential(nn.Linear(channels, hidden), nn.SiLU(),
                          nn.Linear(hidden, channels)) for _ in range(experts)
        ])
        self.last_stats = None

    def forward(self, x):
        b, c, h, w = x.shape
        with tf32_matmul(False):
            logits = self.router(x).permute(0, 2, 3, 1).reshape(-1, self.experts)
        probs = logits.softmax(-1)
        pick = probs.argmax(-1)
        gate = probs.gather(1, pick[:, None])[:, 0]
        flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        out = torch.zeros_like(flat)
        for index, expert in enumerate(self.bank):
            rows = torch.nonzero(pick == index, as_tuple=False).flatten()
            if rows.numel():
                value = expert(flat.index_select(0, rows)) * gate[rows, None]
                out.index_copy_(0, rows, value)
        # Store reductions, not additional full-resolution routing tensors.
        self.last_stats = (torch.bincount(pick, minlength=self.experts).detach(),
                           probs.mean(0), logits.logsumexp(-1).square().mean())
        return out.view(b, h, w, c).permute(0, 3, 1, 2)


class SingleBlock(nn.Module):
    def __init__(self, channels, hidden, experts, dilation=1, residual_scale=.1):
        super().__init__()
        self.norm1 = PixelNorm(channels)
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels),
            nn.Conv2d(channels, channels, 1))
        self.norm2 = PixelNorm(channels)
        self.ffn = Top1FFN(channels, hidden, experts)
        self.spatial_scale = float(residual_scale)
        self.aux = nn.Parameter(torch.full((1, channels, 1, 1), float(residual_scale)))

    def forward(self, x):
        x = x + self.spatial(F.silu(self.norm1(x))) * self.spatial_scale
        return x + self.ffn(self.norm2(x)) * self.aux


class SingleFrameCNN(nn.Module):
    def __init__(self, cfg, geometry):
        super().__init__()
        self.cfg, self.g = cfg, geometry
        self.channels = {k: int(v) for k, v in cfg['channels'].items()}
        self.experts = int(cfg['experts'])
        def block(name, index):
            cycle = cfg.get('dilations', {}).get(name, [1])
            return SingleBlock(self.channels[name], cfg['expert_hidden'][name], self.experts,
                               int(cycle[index % len(cycle)]), cfg.get('residual_scale', .1))
        self.blend_scale = nn.Parameter(torch.ones(1))  # package contract only
        ch = self.channels
        self.stem = nn.Sequential(nn.Conv2d(16, ch['full'], 3, padding=1),
                                  PixelNorm(ch['full']), nn.SiLU())
        self.enc, self.enc_proj = nn.ModuleDict(), nn.ModuleDict()
        for index, name in enumerate(ORDER):
            if index:
                self.enc_proj[name] = nn.Sequential(
                    nn.Conv2d(ch[ORDER[index - 1]], ch[name], 1),
                    PixelNorm(ch[name]), nn.SiLU())
            self.enc[name] = nn.ModuleList([
                block(name, i) for i in range(cfg['blocks_enc'][name])])
        self.merge, self.dec = nn.ModuleDict(), nn.ModuleDict()
        for name in DECODE:
            higher = ORDER[ORDER.index(name) + 1]
            kernel = int(cfg['merge_kernel'][name])
            self.merge[name] = nn.Sequential(
                nn.Conv2d(ch[higher] + ch[name], ch[name], kernel, padding=kernel // 2),
                PixelNorm(ch[name]), nn.SiLU())
            self.dec[name] = nn.ModuleList([
                block(name, i) for i in range(cfg['blocks_dec'][name])])
        self.head_norm = PixelNorm(ch['full'])
        self.head = nn.Conv2d(ch['full'], 4, 1)
        nn.init.normal_(self.head.weight, std=.001)
        nn.init.zeros_(self.head.bias)
        self.capture_enabled = False
        self.captures = {}

    def forward(self, features, batch=1):
        self.captures.clear()
        fw, fh = level_size(self.g, 'full')
        x = features.reshape(batch, fh, fw, 16).permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)
        skips = {}
        for name in ORDER:
            w, h = level_size(self.g, name)
            if x.shape[-2:] != (h, w):
                x = resize_bilinear(x, (h, w))
            if name != 'full':
                x = self.enc_proj[name](x)
            for block in self.enc[name]:
                x = block(x)
            self._capture('s-vit' if name == 'd5' else 's-enc-' + name, x)
            if name != 'd5':
                skips[name] = x
        for name in DECODE:
            w, h = level_size(self.g, name)
            x = resize_bilinear(x, (h, w))
            x = self.merge[name](torch.cat((x, skips[name]), dim=1))
            for block in self.dec[name]:
                x = block(x)
            self._capture('s-dec-' + name, x)
        with tf32_matmul(False):
            head = self.head(F.silu(self.head_norm(x)))
        head = head.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.capture_enabled:
            self.captures['s-head'] = head
        return head

    def _capture(self, tag, x):
        if self.capture_enabled:
            self.captures[tag] = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])

    def moe_losses(self):
        aux, z = [], []
        for module in self.modules():
            if isinstance(module, Top1FFN) and module.last_stats is not None:
                counts, prob, logz = module.last_stats
                freq = counts.float() / counts.sum().clamp_min(1)
                aux.append(self.experts * (freq * prob).sum())
                z.append(logz)
        zero = self.head.weight.new_zeros(())
        return (torch.stack(aux).mean(), torch.stack(z).mean()) if aux else (zero, zero)

    def parameter_ledger(self):
        routed = 0
        active_routed = 0
        blocks = []
        for name, module in self.named_modules():
            if isinstance(module, Top1FFN):
                sizes = [sum(p.numel() for p in expert.parameters()) for expert in module.bank]
                assert len(set(sizes)) == 1
                routed += sum(sizes)
                active_routed += sizes[0]
                blocks.append(dict(name=name, bank=sum(sizes), active=sizes[0]))
        total = sum(p.numel() for p in self.parameters())
        public = total - routed
        return dict(total=total, public=public, routed=routed,
                    active=public + active_routed, active_routed=active_routed,
                    experts=self.experts, top_k=1,
                    definition='all public parameters + one expert per block per spatial position',
                    blocks=blocks)

    def routing_snapshot(self):
        """Last microbatch only; makes collapsed/dead experts visible in logs."""
        snapshot = {}
        for name, module in self.named_modules():
            if isinstance(module, Top1FFN) and module.last_stats is not None:
                counts, prob, _ = module.last_stats
                counts = counts.detach().cpu().tolist()
                total = max(sum(counts), 1)
                snapshot[name] = dict(counts=counts, used=sum(c > 0 for c in counts),
                                      max_fraction=max(counts) / total,
                                      mean_probability=prob.detach().cpu().tolist())
        return snapshot
