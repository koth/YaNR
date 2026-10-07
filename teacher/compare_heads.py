#!/usr/bin/env python3
# Compares two head dumps (f32 [fullRows][4]) value for value - the end-to-end check between the WebGPU port
# and the torch port on identical features:
#
#   python3 compare_heads.py webgpu-head.f32.bin torch-head.f32.bin

import sys

import numpy as np


def main():
    if len(sys.argv) != 3:
        print('usage: python3 compare_heads.py <a.f32.bin> <b.f32.bin>')
        return 2
    a = np.fromfile(sys.argv[1], dtype='<f4')
    b = np.fromfile(sys.argv[2], dtype='<f4')
    if a.size != b.size:
        print(f'size mismatch: {a.size} vs {b.size} floats')
        return 1
    same = a == b
    nan_pairs = np.isnan(a) & np.isnan(b)
    matched = int(same.sum() + nan_pairs.sum())
    mismatch = int(a.size - matched)
    finite = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(a[finite] - b[finite]) if finite.any() else np.zeros(1)
    print(f'{a.size} values: {matched} bit-equal (of which {int(nan_pairs.sum())} NaN pairs), '
          f'{mismatch} mismatched')
    if diff.size:
        print(f'max |a-b| on the finite pairs: {diff.max():.3e}')
    if mismatch:
        bad = np.nonzero(~same & ~nan_pairs)[0][:8]
        for i in bad:
            print(f'  first mismatches: [{i}] {a[i]!r} vs {b[i]!r}')
    print('COMPARE PASS' if mismatch == 0 else 'COMPARE FAIL')
    return 0 if mismatch == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
