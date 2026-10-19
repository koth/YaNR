#!/usr/bin/env python3
"""Print CNN U-Net parameter counts for a shape JSON."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_cnn_unet import CNNUNet, load_cnn_shape                       # noqa: E402
from nr_geometry import geometry_from_valid                           # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('shape')
    parser.add_argument('--size', type=int, default=320)
    args = parser.parse_args()
    cfg = load_cnn_shape(args.shape)
    g = geometry_from_valid(args.size, args.size)
    model = CNNUNet(cfg, g)
    total = sum(p.numel() for p in model.parameters())
    print(f'params: {total:,} ({total / 1e6:.3f}M)')
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        if n:
            print(f'  {name:10s} {n:,}')
    x = torch.randn(g['full_rows'], 16)
    with torch.no_grad():
        y = model(x)
    print('forward:', tuple(y.shape), 'finite:', bool(torch.isfinite(y).all()))


if __name__ == '__main__':
    main()
