#!/usr/bin/env python3
# Checks the weight loader against the reference's scalar addressing: every unpacked matrix is compared
# value for value with a straight port of the C++ index arithmetic (src/reference.cpp, src/model.js), and
# the block tensor sizes are checked against the byte layouts the graph expects.
#
#   python3 check_model.py [modelDir]        # default: $NR_WEIGHTS or /tmp/nr-synth

import os
import sys

import numpy as np

from nr_model import (Model, packed_weight_index, packed_f16_weight_index, packed_input_index,
                      tiled_token, inverse_tiled_token, load_relative_bias_scalar)
from nr_geometry import (fused_layout, pre_fused_layout, upsample_fused_layout, post_fused_layout)


def check_unpacked(model, tensor, offset, k, n, label):
    got = model.fp8_matrix(tensor, offset, k, n)
    stage = tensor.bytes
    from nr_numerics import e4m3_to_number
    if k * n <= 100000:
        positions = ((ki, ni) for ki in range(k) for ni in range(n))
    else:
        rng = np.random.default_rng(7)
        positions = ((int(a), int(b)) for a, b in
                     zip(rng.integers(0, k, 5000), rng.integers(0, n, 5000)))
    count = 0
    for ki, ni in positions:
        code = stage[offset + packed_weight_index(ki, ni, n)]
        expect = 0.0 if (code & 0x7f) == 0x7f else e4m3_to_number(code)
        if got[ki, ni] != np.float16(expect):
            raise SystemExit(f'{label}: mismatch at ({ki},{ni}): expected {expect} got {got[ki, ni]}')
        count += 1
    print(f'ok   {label}: {k}x{n} fp8 matrix unpacked ({count} values)')


def check_f16_unpacked(model, tensor, offset, k, n, label):
    got = model.f16_matrix(tensor, offset, k, n)
    stage = tensor.bytes
    from nr_numerics import f16_to_number
    for ki in range(k):
        for ni in range(n):
            half_index = (offset >> 1) + packed_f16_weight_index(ki, ni, n)
            code = int(stage[half_index * 2]) | (int(stage[half_index * 2 + 1]) << 8)
            if got[ki, ni] != np.float16(f16_to_number(code)):
                raise SystemExit(f'{label}: mismatch at ({ki},{ni})')
    print(f'ok   {label}: {k}x{n} f16 matrix unpacked')


def check_prior(model, tensor, offset, heads, label):
    got = model.relative_bias(tensor, offset, heads)
    for head in range(heads):
        for query in range(64):
            for key in range(64):
                expect = load_relative_bias_scalar(tensor.bytes, offset, head, query, key)
                if got[head, query, key] != np.float16(expect):
                    raise SystemExit(f'{label}: mismatch at h{head} q{query} k{key}: '
                                     f'expected {expect} got {got[head, query, key]}')
    print(f'ok   {label}: prior untangled for {heads} heads')


def main():
    args = sys.argv[1:]
    directory = args[0] if args else os.environ.get('NR_WEIGHTS', '/tmp/nr-synth')
    model = Model('cpu').load(directory)
    if model.block_count != 71:
        raise SystemExit(f'the model has {model.block_count} blocks; this graph is the 71-block network')

    # The tensor sizes the graph asserts on.
    pre = model.tensor(0)
    if pre.byte_length != pre_fused_layout()['end_without_padding'] + 16:
        raise SystemExit(f"block 0 layout: {pre.byte_length}")
    post = model.tensor(70)
    if post.byte_length != post_fused_layout()['end_without_padding']:
        raise SystemExit(f"block 70 layout: {post.byte_length}")
    for channels in (32, 64, 128, 256):
        layout = upsample_fused_layout(channels * 2, channels)
        tensor = model.tensor(66 if channels == 32 else 62 if channels == 64
                              else 56 if channels == 128 else 48)
        if tensor.byte_length != layout['end_without_padding'] + 16:
            raise SystemExit(f'upsample layout for {channels}: {tensor.byte_length} vs '
                             f"{layout['end_without_padding'] + 16}")
    print('ok   block 0 / 70 / upsample layouts match the tensor sizes')

    # One representative matrix of each shape family, against the scalar index arithmetic.
    layout = fused_layout(32)
    check_unpacked(model, pre, layout['expand'], 32, 128, 'block 0 expand')
    check_unpacked(model, pre, layout['qkv'], 32, 96, 'block 0 qkv')
    check_f16_unpacked(model, pre, pre_fused_layout()['input_adapter'], 16, 32, 'input adapter')
    check_f16_unpacked(model, post, post_fused_layout()['post_weights'], 32, 4, 'head')

    layout64 = fused_layout(64)
    t5 = model.tensor(5)
    check_unpacked(model, t5, layout64['expand'], 64 * 2, 128, 'block 5 expert expand')
    check_unpacked(model, t5, layout64['qkv'], 64, 192, 'block 5 qkv')

    t23 = model.tensor(23, 0)
    check_unpacked(model, t23, 0, 512, 512, 'block 23 split layer0')

    t31_expand = model.tensor(31, 0)
    check_unpacked(model, t31_expand, 0, 1024, 4096, 'block 31 vit expand')
    check_prior(model, pre, pre_fused_layout()['relative'], 1, 'block 0 prior')
    check_prior(model, t5, layout64['relative'], 2, 'block 5 prior')
    check_prior(model, model.tensor(23, 2), 512 * 512 * 3, 16, 'block 23 prior')

    # Skip scales and head scales, against the reference's loadHalf/auxF32.
    from nr_numerics import f16_to_number
    aux = model.aux_vector(pre, pre_fused_layout()['ffn_cos_skip'], 32)
    for i in range(32):
        at = pre_fused_layout()['ffn_cos_skip'] + i * 2
        expect = f16_to_number(int(pre.bytes[at]) | (int(pre.bytes[at + 1]) << 8))
        if aux[i] != np.float16(expect):
            raise SystemExit(f'aux mismatch at {i}')
    scales = model.head_scales(model.tensor(23, 2), 512 * 512 * 3 + 16 * 8192, 16)
    if scales.dtype != np.float32 or scales.shape != (16,):
        raise SystemExit('head scales shape')
    print('ok   skip scales / head scales')

    # packedInputIndex / inversePackedInputIndex are inverses of each other.
    for k in range(320):
        if inverse_packed_input_index_check(k) is False:
            raise SystemExit(f'packedInputIndex involution at {k}')
    print('CHECK MODEL PASS')
    return 0


def inverse_packed_input_index_check(k):
    from nr_model import inverse_packed_input_index
    return (inverse_packed_input_index(packed_input_index(k)) == k
            and packed_input_index(inverse_packed_input_index(k)) == k)


if __name__ == '__main__':
    sys.exit(main())
