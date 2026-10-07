#!/usr/bin/env python3
# Compares the boundary captures of the torch port against the WebGPU port's samples, boundary by boundary,
# and names the first one that diverges. The WebGPU side posts the first 4096 values of each boundary
# (web/dump.html); the torch side writes every value (run_nr.py --dump).
#
#   python3 compare_bnd.py /tmp/parity/bnd /tmp/parity/head.f32.bin.bnd

import os
import sys

import numpy as np


def order_key(name):
    # graph order: keep file listing deterministic and roughly chronological
    return name


def main():
    if len(sys.argv) != 3:
        print('usage: python3 compare_bnd.py <torchBndDir> <webgpuBndDir>')
        return 2
    torch_dir, webgpu_dir = sys.argv[1], sys.argv[2]
    names = sorted(os.listdir(webgpu_dir))
    bad = None
    for name in names:
        sample = np.fromfile(os.path.join(webgpu_dir, name), dtype='<f2')
        full_path = os.path.join(torch_dir, name)
        if not os.path.exists(full_path):
            print(f'{name}: MISSING on the torch side')
            bad = bad or name
            continue
        full = np.fromfile(full_path, dtype='<f2')
        if full.size < sample.size:
            print(f'{name}: size mismatch {full.size} vs sample {sample.size}')
            bad = bad or name
            continue
        a = sample.astype(np.float32)
        b = full[:sample.size].astype(np.float32)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        mismatch = int(a.size - same.sum())
        if mismatch:
            first = int(np.nonzero(~same)[0][0])
            print(f'{name}: FAIL {mismatch}/{a.size} mismatched, first at {first}: '
                  f'{a[first]} vs {b[first]}')
            bad = bad or name
        else:
            print(f'{name}: ok ({a.size} values)')
    print('COMPARE BND PASS' if bad is None else f'COMPARE BND FAIL (first bad: {bad})')
    return 0 if bad is None else 1


if __name__ == '__main__':
    sys.exit(main())
