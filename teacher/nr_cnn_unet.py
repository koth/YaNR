"""Independent CNN U-Net student with 8-expert top-1 routed FFNs.

This file is deliberately self-contained: it does not import or subclass the
existing Swin/ViT student.  The training entry point is train_distill_cnn.py.

Contract:
  input:  features [batch * full_rows, 16]
  output: head     [batch * full_rows, 4]
  captures: same s-* names used by the distillation feature loss
"""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr_precision import tf32_matmul


ORDER = ('full', 'd0', 'd1', 'd2', 'd3', 'd4', 'd5')


def level_size(g, name):
    if name == 'full':
        return g['full_width'], g['full_height']
    level = g['levels'][int(name[1:])]
    return level['width'], level['height']


def load_cnn_shape(path):
    with open(path) as fh:
        cfg = json.load(fh)
    for name in ORDER:
        if name not in cfg['channels']:
            raise ValueError(f'missing channels for {name}')
        if cfg['channels'][name] % 8:
            raise ValueError(f'channels for {name} must be a multiple of 8')
    if int(cfg.get('experts', 8)) != 8:
        raise ValueError('this architecture is fixed to 8 experts')
    return cfg


class ExpertFFN(nn.Module):
    """Spatially routed 8-expert top-1 FFN.

    Each pixel picks one expert; only that expert runs on the pixel.  The
    selected expert output is multiplied by its softmax gate, which keeps a
    gradient on the router probabilities while the hard pick stays sparse.
    """

    def __init__(self, channels, hidden, experts=8):
        super().__init__()
        self.experts = experts
        self.router = nn.Conv2d(channels, experts, 1)
        self.bank = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, hidden),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, channels),
            )
            for _ in range(experts)
        ])
        self.last_stats = None

    def forward(self, x):
        b, c, h, w = x.shape
        with tf32_matmul(False):
            logits = self.router(x)
        probs = logits.softmax(dim=1)
        pick = probs.argmax(dim=1)                              # [b, h, w]
        gate = probs.gather(1, pick.unsqueeze(1)).squeeze(1)     # [b, h, w]
        flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        pick_flat = pick.reshape(-1)
        gate_flat = gate.reshape(-1)
        out = flat.new_zeros(flat.shape)
        for e, expert in enumerate(self.bank):
            idx = torch.nonzero(pick_flat == e, as_tuple=False).squeeze(1)
            if idx.numel():
                out.index_copy_(0, idx, expert(flat.index_select(0, idx)) * gate_flat[idx, None])
        self.last_stats = (pick_flat.detach(), probs, logits)
        return out.view(b, h, w, c).permute(0, 3, 1, 2)


class CNNBlock(nn.Module):
    """Residual 3x3 CNN block with a routed expert FFN."""

    def __init__(self, channels, hidden, experts=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.ffn = ExpertFFN(channels, hidden, experts)
        self.aux = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        y = x + self.conv(F.silu(self.norm1(x)))
        return y + self.ffn(self.norm2(y)) * self.aux


class CNNUNet(nn.Module):
    def __init__(self, cfg, g):
        super().__init__()
        self.g = g
        self.cfg = cfg
        self.experts = int(cfg.get('experts', 8))
        ch = cfg['channels']
        self.channels = {name: int(ch[name]) for name in ORDER}
        hidden = cfg['expert_hidden']
        enc_n = cfg['blocks_enc']
        dec_n = cfg['blocks_dec']
        self.blend_scale = nn.Parameter(torch.ones(1))

        self.stem = nn.Sequential(
            nn.Conv2d(16, self.channels['full'], 3, padding=1),
            nn.GroupNorm(8, self.channels['full']),
            nn.SiLU(inplace=True),
        )

        self.enc = nn.ModuleDict()
        self.enc_proj = nn.ModuleDict()
        prev_ch = self.channels['full']
        for name in ORDER:
            if name != 'full':
                self.enc_proj[name] = nn.Sequential(
                    nn.Conv2d(prev_ch, self.channels[name], 1),
                    nn.GroupNorm(8, self.channels[name]),
                    nn.SiLU(inplace=True),
                )
                prev_ch = self.channels[name]
            self.enc[name] = nn.ModuleList([
                CNNBlock(self.channels[name], int(hidden[name]), self.experts)
                for _ in range(int(enc_n[name]))
            ])

        self.merge = nn.ModuleDict()
        self.dec = nn.ModuleDict()
        for name in ('d4', 'd3', 'd2', 'd1', 'd0', 'full'):
            # The high-side channel count comes from the previous decoder level.
            if name == 'd4':
                high = self.channels['d5']
            elif name == 'd3':
                high = self.channels['d4']
            elif name == 'd2':
                high = self.channels['d3']
            elif name == 'd1':
                high = self.channels['d2']
            elif name == 'd0':
                high = self.channels['d1']
            else:
                high = self.channels['d0']
            self.merge[name] = nn.Sequential(
                nn.Conv2d(high + self.channels[name], self.channels[name], 3, padding=1),
                nn.GroupNorm(8, self.channels[name]),
                nn.SiLU(inplace=True),
            )
            self.dec[name] = nn.ModuleList([
                CNNBlock(self.channels[name], int(hidden[name]), self.experts)
                for _ in range(int(dec_n[name]))
            ])

        self.head_norm = nn.GroupNorm(8, self.channels['full'])
        self.head = nn.Conv2d(self.channels['full'], 4, 1)

        self.capture_enabled = False
        self.captures = {}
        self.moe_stats = []

    def capture(self, name, tensor):
        if self.capture_enabled:
            self.captures[name] = tensor

    def _capture_rows(self, name, x):
        self.capture(name, x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]))

    def forward(self, features, batch=1):
        self.captures.clear()
        self.moe_stats.clear()
        g = self.g
        fw, fh = level_size(g, 'full')
        x = features.view(batch, fh, fw, 16).permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)

        skips = {}
        for name in ORDER:
            w, h = level_size(g, name)
            if x.shape[-2:] != (h, w):
                x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            if name != 'full':
                x = self.enc_proj[name](x)
            for block in self.enc[name]:
                x = block(x)
            if name == 'd5':
                self._capture_rows('s-vit', x)
            else:
                self._capture_rows(f's-enc-{name}', x)
                skips[name] = x

        for name in ('d4', 'd3', 'd2', 'd1', 'd0', 'full'):
            w, h = level_size(g, name)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            x = self.merge[name](torch.cat([x, skips[name]], dim=1))
            for block in self.dec[name]:
                x = block(x)
            self._capture_rows(f's-dec-{name}', x)

        with tf32_matmul(False):
            head = self.head(F.silu(self.head_norm(x)))
        head = head.permute(0, 2, 3, 1).reshape(-1, 4)
        self.capture('s-head', head)
        self.moe_stats = [block.ffn.last_stats for name in ORDER
                          for block in list(self.enc[name])
                          + (list(self.dec[name]) if name in self.dec else [])
                          if block.ffn.last_stats is not None]
        return head

    def moe_aux_loss(self):
        if not self.moe_stats:
            return self.head.weight.new_zeros(())
        total = self.head.weight.new_zeros(())
        for pick, probs, _logits in self.moe_stats:
            counts = torch.bincount(pick, minlength=self.experts).float()
            f = counts / counts.sum().clamp_min(1)
            p = probs.mean(dim=(0, 2, 3))
            total = total + self.experts * (f * p).sum()
        return total / len(self.moe_stats)

    def moe_z_loss(self):
        if not self.moe_stats:
            return self.head.weight.new_zeros(())
        total = self.head.weight.new_zeros(())
        for _pick, _probs, logits in self.moe_stats:
            total = total + torch.logsumexp(logits, dim=1).square().mean()
        return total / len(self.moe_stats)
