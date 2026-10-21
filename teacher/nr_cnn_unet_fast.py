"""CNN U-Net optimization variant; the original model remains unchanged.

Eight pixel experts retain the old softmax gate. Sorted groups replace
per-expert nonzero loops; optional tiled GEMMs provide a second executor.
Small-scale architecture edits are explicit
in the shape; plain legacy shapes can also be converted without retraining.
Rep convolutions are fused before inference/native export. See CNN_UNET_FAST.md.
"""
import copy
from functools import lru_cache
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from nr_cnn_unet import CNNUNet, ORDER, level_size
from nr_precision import tf32_matmul
from nr_shared_moe import _Permutation, _TilePack, _TileUnpack, _ExpertWeights

ARCHITECTURE = 'cnn-unet-fast8-v1'
DECODE = tuple(reversed(ORDER[:-1]))


def load_fast_shape(path):
    with open(path, encoding='utf-8') as handle:
        cfg = json.load(handle)
    if cfg.get('architecture') != ARCHITECTURE or cfg.get('experts') != 8:
        raise ValueError('expected cnn-unet-fast8-v1 with 8 experts')
    for name in ORDER:
        if cfg['channels'][name] < 8 or cfg['channels'][name] % 8:
            raise ValueError('GroupNorm channels must be positive multiples of 8')
        if cfg['expert_hidden'][name] < 1 or cfg['blocks_enc'][name] < 1:
            raise ValueError('hidden widths and block counts must be positive')
        if name in DECODE and cfg['blocks_dec'][name] < 1:
            raise ValueError('decoder block counts must be positive')
    if not set(cfg.get('rep_levels', ())).issubset(ORDER):
        raise ValueError('unknown rep level')
    if cfg.get('expert_execution', 'split') not in ('auto', 'split', 'tiled'):
        raise ValueError('expert_execution must be auto, split or tiled')
    return cfg


def compact_route(assign, experts):
    """One scalar read per block, independent of how many experts are empty."""
    rows = assign.numel()
    order = torch.argsort(assign, stable=True)
    selected = assign.index_select(0, order)
    boundaries = torch.searchsorted(selected, torch.arange(experts + 1, device=assign.device))
    counts = boundaries[1:] - boundaries[:-1]
    tile = 8
    for candidate in (4096, 2048, 1024, 512, 256, 128, 64, 32, 16):
        if rows >= 16 * candidate:
            tile = candidate
            break
    tiles = (counts + tile - 1) // tile
    tile_base = tiles.cumsum(0) - tiles
    group_start = counts.cumsum(0) - counts
    position = torch.arange(rows, device=assign.device) - group_start[selected]
    tile_id = tile_base[selected] + position // tile
    tile_count = int(tiles.sum())
    tile_experts = torch.searchsorted(tile_base + tiles,
                                     torch.arange(tile_count, device=assign.device), right=True)
    packed = torch.arange(tile_count * tile, device=assign.device)
    packed_expert = tile_experts[packed // tile]
    packed_pos = (packed // tile - tile_base[packed_expert]) * tile + packed % tile
    valid = packed_pos < counts[packed_expert]
    packed_rows = (group_start[packed_expert] + packed_pos).clamp_max(rows - 1)
    slots = tile_id * tile + position % tile
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(rows, device=assign.device)
    return order, inverse, packed_rows, valid, slots, tile_experts, tiles, tile, counts


class GroupedExpertFFN(nn.Module):
    def __init__(self, channels, hidden, experts=8):
        super().__init__()
        self.experts = experts
        self.channels, self.hidden = channels, hidden
        self.router = nn.Conv2d(channels, experts, 1)
        # Preserve the legacy Linear's contiguous [out,in] slice layout.
        # This also keeps same-weight Linear arithmetic identical to the old
        # path, rather than relying on GEMMs with a different weight stride.
        self.w1 = nn.Parameter(torch.empty(experts, hidden, channels).transpose(1, 2))
        self.b1 = nn.Parameter(torch.empty(experts, hidden))
        self.w2 = nn.Parameter(torch.empty(experts, channels, hidden).transpose(1, 2))
        self.b2 = nn.Parameter(torch.empty(experts, channels))
        for weight, bias, fan_in in ((self.w1, self.b1, channels), (self.w2, self.b2, hidden)):
            nn.init.uniform_(weight, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
            nn.init.uniform_(bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
        self.last_stats = self.route_counts = None
        self.execution = 'split'  # compact 'tiled' and original 'loop' are references
        self.split_max_rows = 131072

    @classmethod
    def from_legacy(cls, ffn):
        module = cls(ffn.router.in_channels, ffn.bank[0][0].out_features, ffn.experts)
        module.router.load_state_dict(ffn.router.state_dict())
        with torch.no_grad():
            for index, expert in enumerate(ffn.bank):
                module.w1[index].copy_(expert[0].weight.T)
                module.b1[index].copy_(expert[0].bias)
                module.w2[index].copy_(expert[2].weight.T)
                module.b2[index].copy_(expert[2].bias)
        return module

    def forward(self, x):
        b, c, h, w = x.shape
        with tf32_matmul(False):
            logits = self.router(x)
        probs = logits.softmax(1)
        pick = probs.argmax(1)
        gate = probs.gather(1, pick[:, None]).reshape(-1, 1)
        flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        assign = pick.reshape(-1)
        execution = self.execution
        if execution == 'auto':
            execution = 'split' if assign.numel() <= self.split_max_rows else 'loop'
        if execution == 'loop':
            w1, b1, w2, b2 = (p.unbind(0) for p in (self.w1, self.b1, self.w2, self.b2))
            out = torch.zeros_like(flat)
            for expert in range(self.experts):
                index = torch.nonzero(assign == expert, as_tuple=False).flatten()
                if index.numel():
                    value = F.silu(F.linear(flat.index_select(0, index), w1[expert].T, b1[expert]), inplace=True)
                    value = F.linear(value, w2[expert].T, b2[expert])
                    out.index_copy_(0, index, value)
            # Do not add an atomic histogram to the large-image inference
            # path. Aux loss/logging can request counts separately.
            self.route_counts = None
        elif execution == 'split':
            # One host read for all eight lengths; the following projections
            # enqueue asynchronously. Views reuse sorted rows/weight banks,
            # avoiding per-tile weight copies and padding workspace.
            order = torch.argsort(assign, stable=True)
            sorted_assign = assign.index_select(0, order)
            boundaries = torch.searchsorted(sorted_assign, torch.arange(self.experts + 1, device=x.device))
            counts = boundaries[1:] - boundaries[:-1]
            lengths = counts.cpu().tolist()
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(assign.numel(), device=x.device)
            sorted_x = _Permutation.apply(flat, order, inverse)
            groups = sorted_x.split(lengths)
            # A single unbind backward assembles each bank once. Repeated
            # bank[e] selects otherwise allocate a full zero bank per expert.
            w1, b1, w2, b2 = (p.unbind(0) for p in (self.w1, self.b1, self.w2, self.b2))
            values = []
            for expert, length in enumerate(lengths):
                if length:
                    value = F.silu(F.linear(groups[expert], w1[expert].T, b1[expert]), inplace=True)
                    values.append(F.linear(value, w2[expert].T, b2[expert]))
            out = _Permutation.apply(torch.cat(values, 0), inverse, order)
            self.route_counts = counts.detach()
        elif execution == 'tiled':
            order, inverse, rows, valid, slots, expert, tiles, tile, counts = compact_route(assign, self.experts)
            sorted_x = _Permutation.apply(flat, order, inverse)
            packed = _TilePack.apply(sorted_x, rows, valid, slots).view(-1, tile, c)
            w1 = _ExpertWeights.apply(self.w1, expert, tiles)
            b1 = _ExpertWeights.apply(self.b1, expert, tiles)
            value = F.silu(torch.baddbmm(b1[:, None, :], packed, w1))
            w2 = _ExpertWeights.apply(self.w2, expert, tiles)
            b2 = _ExpertWeights.apply(self.b2, expert, tiles)
            value = torch.baddbmm(b2[:, None, :], value, w2)
            sorted_out = _TileUnpack.apply(value.reshape(-1, c), slots, rows, valid)
            out = _Permutation.apply(sorted_out, inverse, order)
            self.route_counts = counts.detach()
        else:
            raise ValueError('unknown expert execution: ' + self.execution)
        self.last_stats = (assign.detach(), probs, logits)
        return (out * gate).view(b, h, w, c).permute(0, 3, 1, 2)


class RepConv3(nn.Module):
    """Linear training branches -> one ordinary 3x3 convolution at deployment."""
    def __init__(self, channels, deploy=False):
        super().__init__()
        if deploy:
            self.reparam = nn.Conv2d(channels, channels, 3, padding=1)
        else:
            self.main = nn.Conv2d(channels, channels, 3, padding=1)
            self.point = nn.Conv2d(channels, channels, 1)
            self.horizontal = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
            self.vertical = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
            # Extra branches start as zero: initial spatial operator remains
            # exactly the main convolution, while branch weights get gradients.
            for branch in (self.point, self.horizontal, self.vertical):
                nn.init.zeros_(branch.weight)
                nn.init.zeros_(branch.bias)

    def forward(self, x):
        if hasattr(self, 'reparam'):
            return self.reparam(x)
        return self.main(x) + self.point(x) + self.horizontal(x) + self.vertical(x)

    def fuse(self):
        if hasattr(self, 'reparam'):
            return self
        weight = (self.main.weight + F.pad(self.point.weight, (1, 1, 1, 1)) +
                  F.pad(self.horizontal.weight, (0, 0, 1, 1)) +
                  F.pad(self.vertical.weight, (1, 1, 0, 0)))
        bias = self.main.bias + self.point.bias + self.horizontal.bias + self.vertical.bias
        self.reparam = nn.Conv2d(weight.shape[1], weight.shape[0], 3, padding=1).to(weight)
        with torch.no_grad():
            self.reparam.weight.copy_(weight)
            self.reparam.bias.copy_(bias)
        for name in ('main', 'point', 'horizontal', 'vertical'):
            delattr(self, name)
        return self


@lru_cache(maxsize=128)
def _resize_reverse_plan(source, target, device, dtype):
    # Construct CPU geometry once; validation-first must not cache inference tensors.
    with torch.inference_mode(False), torch.no_grad():
        coordinate = (torch.arange(target, dtype=torch.float64) + .5) * source / target - .5
        lower = coordinate.floor()
        weight = coordinate - lower
        index = torch.cat((lower.long(), lower.long() + 1)).clamp(0, source - 1)
        order = index.argsort(stable=True)
        counts = torch.bincount(index, minlength=source)
        weights = torch.cat((1 - weight, weight))
        return order.to(device), counts.to(device), weights.to(device=device, dtype=dtype)


class _DeterministicBilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, height, width):
        ctx.source = x.shape[-2:]
        return F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)

    @staticmethod
    def backward(ctx, gradient):
        # Ordered segmented sums avoid atomic scatter and retain native fast
        # bilinear forward. Two axis reductions are the transpose of resize.
        for axis, source in ((2, ctx.source[0]), (3, ctx.source[1])):
            target = gradient.shape[axis]
            if source == target:
                continue
            order, counts, weights = _resize_reverse_plan(source, target, gradient.device, gradient.dtype)
            moved = gradient.movedim(axis, 0)
            shape = moved.shape[1:]
            flat = moved.reshape(target, -1)
            values = torch.cat((flat, flat), dim=0) * weights[:, None]
            values = values.index_select(0, order)
            reduced = torch.segment_reduce(values, 'sum', lengths=counts, axis=0, unsafe=True)
            gradient = reduced.reshape(source, *shape).movedim(0, axis).contiguous()
        return gradient, None, None


def resize_bilinear(x, size):
    if x.shape[-2:] == tuple(size):
        return x
    return _DeterministicBilinear.apply(x, *size)



class DenseFFN(nn.Module):
    """稠密 FFN,用来替代逐像素 top-1 MoE 做消融。

    每像素 FLOPs 与"只激活一个专家"相同(hidden 取该级 expert_hidden),
    但全部像素共享同一份参数,不再把 FFN 切成 8 份、每份只服务约 1/8 的像素。
    因此这一替换是单变量对比:算力不变,只改"参数是否碎片化"。
    """

    def __init__(self, channels, hidden):
        super().__init__()
        self.channels, self.hidden = channels, hidden
        self.w1 = nn.Conv2d(channels, hidden, 1)
        self.w2 = nn.Conv2d(hidden, channels, 1)
        for layer, fan_in in ((self.w1, channels), (self.w2, hidden)):
            nn.init.uniform_(layer.weight, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
            nn.init.uniform_(layer.bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

    def extra_repr(self):
        return f'{self.channels}->{self.hidden}->{self.channels}'


class PixelNorm(nn.Module):
    """逐像素跨通道归一化:统计量只在 C 维上求,与 H×W 无关。

    替换 nn.GroupNorm(8, C) 的目的:GroupNorm 在整张特征图的 H×W 上求均值/方差,
    统计量随输入场尺寸变化 —— 320x512 裁剪与 1920x1080 整图的归一化行为不同,
    这正是"训练用裁剪、部署用整图"之间约 2.5 dB 落差的来源。

    state_dict 键与形状和 GroupNorm 完全一致(weight/bias 各 [C]),
    所以同名替换后参数名不变,旧 checkpoint 仍可加载(但语义不同,需重训)。
    """

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        y = x.permute(0, 2, 3, 1)
        mean = y.mean(-1, keepdim=True)
        var = y.var(-1, keepdim=True, unbiased=False)
        y = (y - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias
        return y.permute(0, 3, 1, 2)

    def extra_repr(self):
        return f'{self.channels}, eps={self.eps}'


def swap_group_norm(root):
    """把树里所有 nn.GroupNorm 就地换成 PixelNorm(保留参数名 weight/bias)。"""
    swapped = 0
    for module in list(root.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.GroupNorm):
                setattr(module, name, PixelNorm(child.num_channels, child.eps))
                swapped += 1
    return swapped


class CNNUNetFast(CNNUNet):
    def __init__(self, cfg, geometry):
        super().__init__(copy.deepcopy(cfg), geometry)
        # 2026-10-21:层级顺序可配置。shape 里给 "order" 就能去掉最深层(例如去掉 d5,
        # 让 d4 当瓶颈),用来验证"最深那级下采样是否在抹掉逐像素细节"。
        self.order = tuple(cfg['order']) if cfg.get('order') else ORDER
        self.decode = tuple(reversed(self.order[:-1]))
        for side in (self.enc, self.dec):
            for name, blocks in side.items():
                for block in blocks:
                    if cfg.get('ffn') == 'dense':
                        # 2026-10-21 消融:把逐像素 top-1 MoE 换成稠密 FFN。
                        # hidden 取该级专家 hidden,所以每像素 FLOPs 与一个专家相同,
                        # 但所有像素共享全部参数(不再把参数切成 8 份各管 1/8 像素)。
                        block.ffn = DenseFFN(self.channels[name], cfg['expert_hidden'][name])
                    else:
                        block.ffn = GroupedExpertFFN.from_legacy(block.ffn)
                        block.ffn.execution = cfg.get('expert_execution', 'split')
                        block.ffn.split_max_rows = int(cfg.get('split_max_rows', 131072))
                    if name in cfg.get('rep_levels', ()):
                        original = block.conv
                        block.conv = RepConv3(self.channels[name], bool(cfg.get('deploy', False)))
                        branch = block.conv.reparam if cfg.get('deploy', False) else block.conv.main
                        branch.load_state_dict(original.state_dict())
        # 2026-10-21:可选把 GroupNorm 换成逐像素归一化(shape 里 norm='pixel'),
        # 用来验证"归一化依赖场尺寸"是不是裁剪训练/整图部署落差的根源。
        if cfg.get('norm') == 'pixel':
            self.swapped_norms = swap_group_norm(self)
        if cfg.get('head_init_std') is not None:
            nn.init.normal_(self.head.weight, std=float(cfg['head_init_std']))
            nn.init.zeros_(self.head.bias)

    def forward(self, features, batch=1):
        self.captures.clear()
        self.moe_stats.clear()
        fw, fh = level_size(self.g, 'full')
        x = features.reshape(batch, fh, fw, 16).permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)
        skips = {}
        order, decode = self.order, self.decode
        for name in order:
            w, h = level_size(self.g, name)
            x = resize_bilinear(x, (h, w))
            if name != 'full':
                x = self.enc_proj[name](x)
            for block in self.enc[name]:
                x = block(x)
            self._capture_rows('s-vit' if name == order[-1] else 's-enc-' + name, x)
            if name != order[-1]:
                skips[name] = x
        for name in decode:
            w, h = level_size(self.g, name)
            x = resize_bilinear(x, (h, w))
            x = self.merge[name](torch.cat((x, skips[name]), dim=1))
            for block in self.dec[name]:
                x = block(x)
            self._capture_rows('s-dec-' + name, x)
        with tf32_matmul(False):
            head = self.head(F.silu(self.head_norm(x)))
        head = head.permute(0, 2, 3, 1).reshape(-1, 4)
        self.capture('s-head', head)
        return head

    def moe_losses(self):
        aux, z = [], []
        for module in self.modules():
            if isinstance(module, GroupedExpertFFN) and module.last_stats is not None:
                pick, probabilities, logits = module.last_stats
                counts = (module.route_counts if module.route_counts is not None else
                          torch.bincount(pick, minlength=self.experts))
                freq = counts.float() / counts.sum().clamp_min(1)
                aux.append(self.experts * (freq * probabilities.mean((0, 2, 3))).sum())
                z.append(logits.logsumexp(1).square().mean())
        zero = self.head.weight.new_zeros(())
        return (torch.stack(aux).mean(), torch.stack(z).mean()) if aux else (zero, zero)

    def parameter_ledger(self):
        routed = active_routed = extra = 0
        blocks = []
        for name, module in self.named_modules():
            if isinstance(module, GroupedExpertFFN):
                bank = sum(p.numel() for p in (module.w1, module.b1, module.w2, module.b2))
                routed += bank
                active_routed += bank // self.experts
                blocks.append(dict(name=name, bank=bank, active=bank // self.experts))
            if isinstance(module, RepConv3) and not hasattr(module, 'reparam'):
                extra += sum(p.numel() for branch in (module.point, module.horizontal, module.vertical)
                             for p in branch.parameters())
        total = sum(p.numel() for p in self.parameters())
        return dict(total=total, public=total - routed, routed=routed,
                    active=total - routed + active_routed, active_routed=active_routed,
                    deploy_total=total - extra, deploy_active=total - extra - routed + active_routed,
                    training_only_rep=extra, experts=self.experts, top_k=1,
                    definition='all public parameters + one expert per block per spatial position', blocks=blocks)

    def moe_aux_loss(self):
        return self.moe_losses()[0]

    def moe_z_loss(self):
        return self.moe_losses()[1]

    def routing_snapshot(self):
        result = {}
        for name, module in self.named_modules():
            if isinstance(module, GroupedExpertFFN) and module.last_stats is not None:
                tensor = (module.route_counts if module.route_counts is not None else
                          torch.bincount(module.last_stats[0], minlength=self.experts))
                counts = tensor.cpu().tolist()
                result[name] = dict(counts=counts, used=sum(c > 0 for c in counts),
                                    max_fraction=max(counts) / max(sum(counts), 1))
        return result

    def switch_to_deploy(self):
        if self.training:
            raise ValueError('call eval() before fusing training branches')
        for module in self.modules():
            if isinstance(module, RepConv3):
                module.fuse()
        self.cfg['deploy'] = True
        return self

    @classmethod
    def from_legacy(cls, legacy):
        cfg = copy.deepcopy(legacy.cfg)
        cfg.update(architecture=ARCHITECTURE, rep_levels=[], deploy=True)
        result = cls(cfg, legacy.g)
        state = legacy.state_dict()
        packed = result.state_dict()
        for name, value in packed.items():
            if name in state:
                value.copy_(state[name])
        for side in ('enc', 'dec'):
            for name, blocks in getattr(result, side).items():
                for index, block in enumerate(blocks):
                    prefix = '%s.%s.%d.ffn.' % (side, name, index)
                    for expert in range(result.experts):
                        for target, suffix, transpose in (('w1', '0.weight', True), ('b1', '0.bias', False),
                                                          ('w2', '2.weight', True), ('b2', '2.bias', False)):
                            value = state[prefix + 'bank.%d.' % expert + suffix]
                            packed[prefix + target][expert].copy_(value.T if transpose else value)
        result.load_state_dict(packed)
        return result

    def to_native_model(self):
        """Export ordinary CNN DSL/weights, already supported by C++ CPU/CUDA."""
        if self.training:
            raise ValueError('call eval() before native export')
        self.switch_to_deploy()
        cfg = copy.deepcopy(self.cfg)
        for key in ('rep_levels', 'deploy', 'parameter_targets'):
            cfg.pop(key, None)
        cfg['architecture'] = 'cnn_unet_moe8'
        native = CNNUNet(cfg, self.g)
        source = self.state_dict()
        target = native.state_dict()
        for name, value in target.items():
            if name in source:
                value.copy_(source[name])
            elif '.conv.' in name:
                value.copy_(source[name.replace('.conv.', '.conv.reparam.')])
            elif '.ffn.bank.' in name:
                prefix, tail = name.split('.ffn.bank.')
                expert, layer, kind = tail.split('.')
                key = ('w1' if layer == '0' else 'w2') if kind == 'weight' else ('b1' if layer == '0' else 'b2')
                matrix = source[prefix + '.ffn.' + key][int(expert)]
                value.copy_(matrix.T if kind == 'weight' else matrix)
            else:
                raise KeyError('cannot export ' + name)
        native.load_state_dict(target)
        return native.eval()
