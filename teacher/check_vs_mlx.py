#!/usr/bin/env python3
# Cross-check our weight unpacking against MLX-DLSS's independent decode of the same DLL resource.
#
# Both implementations were written against the same packed WEIGHTS_HT layout (MMA fragments, 16-byte
# alignment pads); ours is validated against the C++ reference's addressing on synthetic weights, theirs
# is validated by running the real network. Comparing the decoded matrices of the real model settles
# whether the unpacking - and every byte offset the network uses - is right for the actual DLL.
#
#   python3 check_vs_mlx.py <model-dir> <dlssnr-weights-logical.safetensors>

import json
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nr_model as nm


def read_logical(path):
    with open(path, 'rb') as handle:
        n = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(n))
        data = handle.read()
    out = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        lo, hi = info['data_offsets']
        dtype = {'F16': '<f2', 'F32': '<f4', 'U16': '<u2', 'U8': 'u1'}[info['dtype']]
        out[name] = np.frombuffer(data[lo:hi], dtype=dtype).reshape(info['shape']).copy()
    return out


def unpack_e4m3(tensor, byte_offset, k, n):
    """nr_model.fp8_matrix without the bounded-half assertion (reported separately)."""
    k_index = np.arange(k, dtype=np.int64)[:, None]
    n_index = np.arange(n, dtype=np.int64)[None, :]
    index = nm.packed_weight_index_array(k_index, n_index, n)
    codes = np.asarray(tensor.bytes[byte_offset + index], dtype=np.int64)
    return nm._E4_DECODE[codes].astype(np.float16), np.asarray(tensor.bytes[byte_offset:byte_offset + k * n])


def compare(label, got, ref):
    if got.shape != ref.shape:
        print(f'FAIL {label}: shape {got.shape} vs {ref.shape}')
        return True
    ga, ra = got.astype(np.float32), ref.astype(np.float32)
    bad = int((got.view(np.uint16) != ref.view(np.uint16)).sum()) if got.dtype == ref.dtype \
        else int((ga != ra).sum())
    if bad:
        i = int(np.nonzero((ga != ra).ravel())[0][0])
        print(f'FAIL {label}: {bad}/{got.size} differ, first [{i}]: {ga.ravel()[i]} vs {ra.ravel()[i]}')
        return True
    print(f'ok   {label}: {got.size} values bit-equal')
    return False


def main():
    if len(sys.argv) != 3:
        print('usage: python3 check_vs_mlx.py <model-dir> <dlssnr-weights-logical.safetensors>')
        return 2
    model = nm.Model(None).load(sys.argv[1])
    logical = read_logical(sys.argv[2])
    failed = False

    # (tensor, byte offset, k, n, logical name) for the matrix families the network reads.
    cases = [
        ('block0.layer0.layer', 0, 32, 128, 'block0.layer0.weight1'),
        ('block0.layer0.layer', 4096, 128, 32, 'block0.layer0.weight2'),
        ('block0.layer0.layer', 8208, 16, 32, 'block0.layer0.input_adapter_weight'),
        ('block0.layer0.layer', 9312, 32, 96, 'block0.layer0.qkv_weight'),
        ('block0.layer0.layer', 20592, 32, 32, 'block0.layer0.projection_weight'),
        ('block1.layer0.layer', 0, 32, 128, 'block1.layer0.weight1'),
        ('block1.layer0.layer', 4096, 128, 32, 'block1.layer0.weight2'),
        ('block1.layer0.layer', 8288, 32, 96, 'block1.layer0.qkv_weight'),
        ('block1.layer0.layer', 19568, 32, 32, 'block1.layer0.projection_weight'),
        ('block31.layer0.layer', 0, 1024, 4096, 'block31.layer0.weight'),
    ]
    for name, offset, k, n, logical_name in cases:
        tensor = model.tensor(int(name.split('.')[0][5:]))
        f16_matrix = logical_name.endswith('adapter_weight')
        if f16_matrix:
            got = nm.Model.f16_matrix.__get__(model)(tensor, offset, k, n)
        else:
            got, raw = unpack_e4m3(tensor, offset, k, n)
            magnitude = raw & 0x7f
            over = int(((magnitude > 0x51) & (magnitude != 0x7f)).sum())
            if over:
                print(f'note {name} [{offset}:]: {over} codes above the bounded-half threshold '
                      f'(nr_model raises on this for synthetic weights)')
        failed |= compare(f'{logical_name} @ {name}[{offset}:]', got, logical[logical_name])

    print('CHECK VS MLX FAIL' if failed else 'CHECK VS MLX PASS')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
