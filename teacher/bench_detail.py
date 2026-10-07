#!/usr/bin/env python3
# Per-shape, per-phase breakdown: which GEMM shapes eat the frame, and how much of window_attention is
# torch prep versus the fused kernel. CUDA events around each backend call, aggregated by signature.
#
#   NR_WEIGHTS=... python3 bench_detail.py --width 512 --height 512

import argparse
import os
from collections import defaultdict

import torch

import nr_torch
import nr_cuda
from nr_geometry import geometry_from_valid
from nr_model import Model
from nr_network import Network
from run_nr import bench_features

parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--height', type=int, default=512)
args = parser.parse_args()

gemm_time = defaultdict(float)
gemm_count = defaultdict(int)
win_time = defaultdict(lambda: [0.0, 0.0])
win_count = defaultdict(int)
ffn_time = [0.0]
ffn_count = [0]


class _Timer:
    def __init__(self):
        self.ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))

    def __enter__(self):
        self.ev[0].record()
        return self

    def __exit__(self, *exc):
        self.ev[1].record()
        torch.cuda.synchronize()
        return False

    @property
    def ms(self):
        return self.ev[0].elapsed_time(self.ev[1])


orig_gemm = Network.gemm
orig_win = Network.window_attention


def timed_gemm(self, x, w, k, n, batches=1, broadcast=False, partition=0, silu=False,
               residual=None, aux=None, e4=True, half=False):
    key = (k, n, batches, broadcast, partition, silu, residual is not None, e4, half, x.shape[0])
    with _Timer() as t:
        out = orig_gemm(self, x, w, k, n, batches=batches, broadcast=broadcast, partition=partition,
                        silu=silu, residual=residual, aux=aux, e4=e4, half=half)
    gemm_time[key] += t.ms
    gemm_count[key] += 1
    return out


class _PrepTimer:
    """Times the torch prep of window_attention by wrapping the fused call it makes."""

    def __init__(self):
        self.kernel_ms = 0.0
        self.calls = 0


def timed_win(self, qkv, prior, scales, width, height, heads, phase):
    import nr_network as nn
    saved = nn.ntr.window_attention
    box = _PrepTimer()

    def timed_kernel(*a, **kw):
        with _Timer() as t:
            out = saved(*a, **kw)
        box.kernel_ms += t.ms
        box.calls += 1
        return out

    if nn.ntr is not None:
        nn.ntr.window_attention = timed_kernel
    try:
        with _Timer() as t:
            out = orig_win(self, qkv, prior, scales, width, height, heads, phase)
    finally:
        if nn.ntr is not None:
            nn.ntr.window_attention = saved
    key = (width, height, heads, qkv.shape[0])
    win_time[key][0] += t.ms - box.kernel_ms
    win_time[key][1] += box.kernel_ms
    win_count[key] += 1
    return out


Network.gemm = timed_gemm
Network.window_attention = timed_win

_orig_ffn = nr_cuda.block_ffn


def timed_ffn(*a, **kw):
    with _Timer() as t:
        out = _orig_ffn(*a, **kw)
    ffn_time[0] += t.ms
    ffn_count[0] += 1
    return out


nr_cuda.block_ffn = timed_ffn

g = geometry_from_valid(args.width, args.height)
model = Model(nr_torch.DEVICE).load(os.environ['NR_WEIGHTS'])
features = torch.from_numpy(bench_features(g)).to(nr_torch.DEVICE)
network = Network(model, g)

network.record(features)          # warm
torch.cuda.synchronize()
gemm_time.clear()
gemm_count.clear()
win_time.clear()
win_count.clear()
ffn_time[0] = 0.0
ffn_count[0] = 0

network.record(features)
torch.cuda.synchronize()

print('== gemm by (K, N, batches, broadcast, partition, silu, seeded, e4, half, rows) ==')
total = 0.0
for key, value in sorted(gemm_time.items(), key=lambda kv: -kv[1]):
    total += value
    print(f'{value:9.2f} ms {gemm_count[key]:4d}x  {value / gemm_count[key]:7.3f} ms/call  {key}')
print(f'{total:9.2f} ms gemm total')

print('== window_attention by (width, height, heads, rows): (prep, kernel) ==')
total = 0.0
for key, value in sorted(win_time.items(), key=lambda kv: -sum(kv[1])):
    total += sum(value)
    print(f'{sum(value):9.2f} ms {win_count[key]:4d}x  prep {value[0]:7.2f}  kernel {value[1]:7.2f}  {key}')
print(f'{total:9.2f} ms window total')
print(f'{ffn_time[0]:9.2f} ms block_ffn {ffn_count[0]}x')
