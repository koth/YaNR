#!/usr/bin/env python3
# Drills into block 0: recomputes its first 64 rows (one 8x8 window at origin (0,0), phase 0) with the scalar
# oracle - straight from reference.cpp's semantics - and compares the block output against the torch port's
# boundary dump and the WebGPU port's boundary sample.
#
#   NR_WEIGHTS=... python3 check_block0.py /tmp/parity     (320x320 dumps)

import os
import sys

import numpy as np

import nr_numerics as num
from nr_model import Model, packed_input_index, load_relative_bias_scalar
from nr_geometry import geometry_from_valid, pre_fused_layout
from run_nr import bench_features


def gemm_fp8_scalar(row, w_kn, k, n, initial, partition=0):
    """reference.cpp gemmFp8Element for one element, with the row's A operand and a [K][N] weight matrix."""
    sums = num.round_f16(initial)
    part_sums = 0.0
    for kb in range(0, k, 32):
        a = [row[packed_input_index(kb + j)] for j in range(32)]
        b = [w_kn[kb + j][n] for j in range(32)]
        sums = num.ada_fp8_fdpa16(a[:16], b[:16], 16, sums)
        sums = num.ada_fp8_fdpa16(a[16:], b[16:], 16, sums)
        if partition and ((kb + 32) % partition == 0 or kb + 32 >= k):
            part_sums = sums if kb < partition else num.round_f16(part_sums + sums)
            sums = 0.0
    return part_sums if partition else sums


def gemm_f16_scalar(row, w_kn, k, n):
    """reference.cpp gemmF16Element for one element."""
    value = 0.0
    for base in range(0, k, 8):
        a = [num.round_f16(row[base + i]) for i in range(8)]
        b = [w_kn[base + i][n] for i in range(8)]
        value = num.ada_f16_fdpa8(a, b, 8, value)
    return value


def norm_of(x):
    """reference.cpp windowNormalizeRef's normOf: pair squares, half tree, 1/sqrt."""
    r = []
    for c in range(16):
        high_square = num.round_f16(x[c + 16] * x[c + 16])
        r.append(num.round_f16(x[c] * x[c] + high_square))
    for stride in (8, 4, 2, 1):
        for c in range(stride):
            r[c] = num.round_f16(r[c] + r[c + stride])
    return num.round_f16(1.0 / (r[0] ** 0.5))


def window_normalize(qkv_row, scale):
    q, k, v = qkv_row[0:32], qkv_row[32:64], qkv_row[64:96]
    q_norm = norm_of(q)
    k_norm = norm_of(k)
    scale_half = num.round_f16(scale)
    out = [num.fp8_domain(num.round_f16(num.round_f16(q[c] * q_norm) * scale_half)) for c in range(32)]
    out += [num.fp8_domain(num.round_f16(k[c] * k_norm)) for c in range(32)]
    out += [num.fp8_domain(v[c]) for c in range(32)]
    return out


def inverse_tiled(token):
    tile, within = token >> 4, token & 15
    return ((tile >> 1) * 4 + (within >> 2)) * 8 + (tile & 1) * 4 + (within & 3)


def main():
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/parity'
    width = int(os.environ.get('BND_WIDTH', 320))
    height = int(os.environ.get('BND_HEIGHT', 320))
    g = geometry_from_valid(width, height)
    fw = g['full_width']
    layout = pre_fused_layout()
    model = Model('cpu').load(os.environ.get('NR_WEIGHTS', '/tmp/nr-synth'))
    pre = model.tensor(0)

    features = bench_features(g)
    slots = [(x, y) for y in range(8) for x in range(8)]
    rows = [y * fw + x for (x, y) in slots]

    def plain(matrix):
        return [[float(v) for v in row] for row in matrix]

    w_adapter = plain(model.f16_matrix(pre, layout['input_adapter'], 16, 32))
    w_expand = plain(model.fp8_matrix(pre, layout['expand'], 32, 128))
    w_contract = plain(model.fp8_matrix(pre, layout['contract_weights'], 128, 32))
    w_qkv = plain(model.fp8_matrix(pre, layout['qkv'], 32, 96))
    w_proj = plain(model.fp8_matrix(pre, layout['projection'], 32, 32))
    ffn_aux = [float(v) for v in model.aux_vector(pre, layout['ffn_cos_skip'], 32)]
    attn_aux = [float(v) for v in model.aux_vector(pre, layout['attn_cos_skip'], 32)]
    scale = float(model.head_scales(pre, layout['scale'], 1)[0])
    prior = [[load_relative_bias_scalar(pre.bytes, layout['relative'], 0, qq, kk) for kk in range(64)]
             for qq in range(64)]

    contract_raw_by_row = {}
    normalized = {}

    # The input bundle for the C++ tie-breaker (check_block0.cpp): the packed block-0 tensor and the 64 rows.
    import struct
    with open(os.path.join(dump_dir, 'block0_input.bin'), 'wb') as handle:
        handle.write(struct.pack('<I', pre.byte_length))
        handle.write(pre.bytes.tobytes())
        handle.write(features[rows].astype('<f4').tobytes())

    for r in rows:
        f16_row = [num.round_f16(float(features[r, c])) for c in range(16)]
        adapter = [gemm_f16_scalar(f16_row, w_adapter, 16, n) for n in range(32)]
        adapter_e4 = [num.fp8_domain(v) for v in adapter]
        ffn = [num.fp8_domain(num.mp_cubic_silu(
            gemm_fp8_scalar(adapter_e4, w_expand, 32, m, 0.0))) for m in range(128)]
        contract_raw = []
        for n in range(32):
            initial = num.round_f16(adapter[n] * ffn_aux[n])
            contract_raw.append(gemm_fp8_scalar(ffn, w_contract, 128, n, initial))
        ffn_quantized = [num.fp8_domain(v) for v in contract_raw]
        qkv = [gemm_fp8_scalar(ffn_quantized, w_qkv, 32, j, 0.0) for j in range(96)]
        normalized[r] = window_normalize(qkv, scale)
        contract_raw_by_row[r] = contract_raw

    def field_row(x, y):
        return y * fw + x

    # Window attention for the query at slot 0 (field (0, 0)); the window sits at origin (0, 0).
    q = normalized[rows[0]][0:32]
    scores = []
    for slot, (kx, ky) in enumerate(slots):
        kvec = normalized[field_row(kx, ky)][32:64]
        score = num.ada_fp8_fdpa16(q[:16], kvec[:16], 16, prior[0][slot])
        score = num.ada_fp8_fdpa16(q[16:], kvec[16:], 16, score)
        scores.append(num.exp_weight(num.round_f16(score)))

    def pair(p, parity):
        key = p * 2 + parity
        get = lambda t: scores[inverse_tiled(t)]
        b01 = num.round_f16(get(key) + get(key + 8))
        b23 = num.round_f16(get(key + 16) + get(key + 24))
        b45 = num.round_f16(get(key + 32) + get(key + 40))
        b67 = num.round_f16(get(key + 48) + get(key + 56))
        return num.round_f16(num.round_f16(num.round_f16(b01 + b23) + b45) + b67)

    even = num.round_f16(num.round_f16(num.round_f16(pair(0, 0) + pair(1, 0)) + pair(2, 0)) + pair(3, 0))
    odd = num.round_f16(num.round_f16(num.round_f16(pair(0, 1) + pair(1, 1)) + pair(2, 1)) + pair(3, 1))
    total = num.round_f16(even + odd)
    reciprocal = num.round_f16(1.0 / total)
    for slot in range(64):
        scores[slot] = num.fp8_domain(num.round_f16(scores[slot] * reciprocal))

    attended = []
    for c in range(32):
        value = 0.0
        for group in range(4):
            w = [scores[inverse_tiled(group * 16 + i)] for i in range(16)]
            v = [normalized[field_row(*slots[inverse_tiled(group * 16 + i)])][64 + c] for i in range(16)]
            value = num.ada_fp8_fdpa16(w, v, 16, value)
        attended.append(num.fp8_domain(value))

    block0 = []
    for n in range(32):
        initial = num.round_f16(contract_raw_by_row[rows[0]][n] * attn_aux[n])
        block0.append(num.fp8_domain(gemm_fp8_scalar(attended, w_proj, 32, n, initial)))

    torch_path = os.path.join(dump_dir, 'bnd', 'block-0.f16bin')
    webgpu_path = os.path.join(dump_dir, 'head.webgpu.f32.bin.bnd', 'block-0.f16bin')
    torch_b = np.fromfile(torch_path, dtype='<f2')[:32].astype(np.float64) if os.path.exists(torch_path) else None
    webgpu_b = np.fromfile(webgpu_path, dtype='<f2')[:32].astype(np.float64) if os.path.exists(webgpu_path) else None

    print('block-0 output row 0, channel by channel:')
    print('  oracle: ' + ' '.join(f'{v:.6g}' for v in block0))
    if torch_b is not None:
        print('  torch : ' + ' '.join(f'{v:.6g}' for v in torch_b))
        t_bad = [i for i in range(32) if float(np.float16(block0[i])) != torch_b[i]]
        print(f'  torch vs oracle: {len(t_bad)}/32 mismatched' + (f', first at {t_bad[0]}' if t_bad else ''))
    if webgpu_b is not None:
        print('  webgpu: ' + ' '.join(f'{v:.6g}' for v in webgpu_b))
        w_bad = [i for i in range(32) if float(np.float16(block0[i])) != webgpu_b[i]]
        print(f'  webgpu vs oracle: {len(w_bad)}/32 mismatched' + (f', first at {w_bad[0]}' if w_bad else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
