#!/usr/bin/env python3
# Where a frame's time goes: wraps the Network's stages with CUDA-synchronized timers and prints the
# breakdown. One frame at the given size.
#
#   NR_WEIGHTS=... python3 profile_nr.py --width 512 --height 512

import argparse
import os
import time

import torch

import nr_torch
from nr_geometry import geometry_from_valid
from nr_model import Model
from nr_network import Network
from run_nr import bench_features

parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--height', type=int, default=512)
args = parser.parse_args()

totals = {}
counts = {}


def wrap(name):
    original = getattr(Network, name)

    def timed(self, *a, **kw):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = original(self, *a, **kw)
        torch.cuda.synchronize()
        totals[name] = totals.get(name, 0.0) + (time.perf_counter() - start)
        counts[name] = counts.get(name, 0) + 1
        return out

    setattr(Network, name, timed)


for name in ('gemm', 'gemm_f16', 'window_attention', 'vit_normalize', 'vit_attend',
             'downsample', 'upsample_residual', 'post_blend'):
    wrap(name)

g = geometry_from_valid(args.width, args.height)
model = Model(nr_torch.DEVICE).load(os.environ['NR_WEIGHTS'])
features = torch.from_numpy(bench_features(g)).to(nr_torch.DEVICE)
network = Network(model, g)

# One warm frame first: the measured frame must not pay for Triton JIT or the allocator's first touch.
network.record(features)
torch.cuda.synchronize()
totals.clear()
counts.clear()

start = time.perf_counter()
head = network.record(features)
torch.cuda.synchronize()
total = time.perf_counter() - start

for name, value in sorted(totals.items(), key=lambda kv: -kv[1]):
    print(f'{value * 1000:9.1f} ms  {counts[name]:4d}x  {name}')
print(f'{total * 1000:9.1f} ms  total')
