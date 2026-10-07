#!/usr/bin/env python3
# Unit tests for the fused kernels against the step-by-step path, bit-exact or bust:
#   window_attention_raw  vs  the torch prep + the reference window kernel
#   block_ffn             vs  the three chained GEMMs (expand silu, contract seeded, qkv)
# The step-by-step side runs the verified kernels, so this isolates the fused kernels alone.
#
#   python3 check_fused.py

import numpy as np
import torch

import nr_torch
import nr_cuda
from nr_network import Network

DEVICE = nr_torch.DEVICE


def bits16(t):
    return t.detach().cpu().numpy().astype(np.float16).view(np.uint16)


def diff(name, a, b):
    an = a.detach().cpu().numpy().astype(np.float32)
    bn = b.detach().cpu().numpy().astype(np.float32)
    ba, bb = bits16(a), bits16(b)
    if ba.shape != bb.shape:
        print(f'FAIL {name}: shape {ba.shape} vs {bb.shape}')
        return True
    same = (ba == bb) | (np.isnan(an) & np.isnan(bn))
    bad = int(ba.size - same.sum())
    if bad:
        i = int(np.nonzero(~same.ravel())[0][0])
        print(f'FAIL {name}: {bad}/{ba.size} mismatched, first at [{i}]: '
              f'{an.ravel()[i]!r} (0x{ba.ravel()[i]:04x}) vs {bn.ravel()[i]!r} (0x{bb.ravel()[i]:04x})')
        return True
    print(f'ok   {name}: {ba.size} values bit-equal')
    return False


def f16np(rng, *shape):
    """A CPU numpy f16 array of plausible activations (weights/aux/prior scale)."""
    v = rng.standard_normal(shape) * 6
    v[rng.uniform(size=shape) < 0.05] *= 30
    return np.ascontiguousarray(v.astype(np.float16))


def f16dev(rng, *shape):
    return torch.from_numpy(f16np(rng, *shape)).to(DEVICE)


def run_window(rng, width, height, heads, phase=0):
    rows = width * height
    qkv = f16dev(rng, rows, heads, 96)
    prior = f16np(rng, heads, 64, 64)
    scales = np.ascontiguousarray((rng.standard_normal(heads) * 0.3).astype(np.float32))
    net = Network(None, None)
    saved = nr_cuda.FUSED_PREP
    nr_cuda.FUSED_PREP = False
    ref = net.window_attention(qkv, prior, scales, width, height, heads, phase)
    nr_cuda.FUSED_PREP = True
    got = net.window_attention(qkv, prior, scales, width, height, heads, phase)
    nr_cuda.FUSED_PREP = saved
    return diff(f'window_attention_raw {width}x{height} heads{heads} phase{phase}', got, ref)


def run_block(rng, rows):
    state = f16dev(rng, rows, 32)
    res = f16dev(rng, rows, 32)
    aux_np = f16np(rng, 32)
    w1_np, w2_np, w3_np = f16np(rng, 32, 128), f16np(rng, 128, 32), f16np(rng, 32, 96)
    net = Network(None, None)
    aux = net.np16(aux_np)
    w1, w2, w3 = net.np16(w1_np), net.np16(w2_np), net.np16(w3_np)
    ffn_e4, _ = net.gemm(state, w1_np, 32, 128, silu=True)
    ffn_q_e4, ffn_raw = net.gemm(ffn_e4, w2_np, 128, 32, residual=res, aux=aux, e4=True, half=True)
    _, qkv_ref = net.gemm(ffn_q_e4, w3_np, 32, 96, e4=False, half=True)
    got_raw = torch.zeros(rows, 32, dtype=torch.float16, device=DEVICE)
    got_qkv = torch.zeros(rows, 96, dtype=torch.float16, device=DEVICE)
    nr_cuda.block_ffn(state, w1, w2, w3, res, aux, got_raw, None, got_qkv)
    failed = diff(f'block_ffn raw rows={rows}', got_raw, ffn_raw)
    failed |= diff(f'block_ffn qkv rows={rows}', got_qkv, qkv_ref)
    return failed


def run_block_expert(rng, rows, ch):
    e = ch // 32
    state = f16dev(rng, rows, ch)
    res = f16dev(rng, rows, ch)
    aux_np = f16np(rng, ch)
    w1_np = f16np(rng, e * ch, 128)
    w2_np = f16np(rng, e * 128, 32)
    w3_np = f16np(rng, ch, ch)
    w4_np = f16np(rng, ch, 3 * ch)
    net = Network(None, None)
    aux = net.np16(aux_np)
    ffn, _ = net.gemm(state, w1_np, ch, 128, batches=e, broadcast=True, silu=True)
    ffn_narrow, _ = net.gemm(ffn, w2_np, 128, 32, batches=e)
    ffn_quantized, _ = net.gemm(ffn_narrow, w3_np, ch, ch, residual=res, aux=aux, e4=True, half=True)
    _, qkv_ref = net.gemm(ffn_quantized, w4_np, ch, 3 * ch, e4=False, half=True)
    w1 = net.np16(w1_np)
    w2 = net.np16(w2_np)
    w3 = net.np16(w3_np)
    w4 = net.np16(w4_np)
    got_e4 = torch.zeros(rows, ch, dtype=torch.float16, device=DEVICE)
    got_qkv = torch.zeros(rows, 3 * ch, dtype=torch.float16, device=DEVICE)
    nr_cuda.block_ffn_expert(state, w1, w2, w3, w4, res, aux, None, got_e4, got_qkv)
    failed = diff(f'block_ffn_expert e4 rows={rows} ch={ch}', got_e4, ffn_quantized)
    failed |= diff(f'block_ffn_expert qkv rows={rows} ch={ch}', got_qkv, qkv_ref)
    return failed


def run_vit(rng, tokens, padded, heads):
    qkv = f16dev(rng, tokens, heads, 96)
    scales = np.ascontiguousarray((rng.standard_normal(heads) * 0.3).astype(np.float32))
    net = Network(None, None)
    normalized = net.vit_normalize(qkv.reshape(tokens, heads * 96), scales, tokens, padded, heads)
    ref = net.vit_attend(normalized, tokens, padded, heads)          # [tokens, heads*32]
    learned = nr_torch.round_f16_t(net.np32(scales))
    got = nr_cuda.vit_attention_raw(qkv, learned, net.head_scale_const(), tokens, padded, heads)
    return diff(f'vit_attention_raw tokens={tokens} padded={padded} heads={heads}',
                got.reshape(tokens, heads * 32), ref)


def main():
    rng = np.random.RandomState(20240612)
    failed = False
    failed |= run_window(rng, 24, 16, 1)
    failed |= run_window(rng, 24, 16, 2)
    failed |= run_window(rng, 40, 24, 1)
    failed |= run_window(rng, 21, 13, 3)           # odd sizes exercise the window overhang
    failed |= run_window(rng, 24, 16, 1, phase=1)  # shifted windows wrap the field
    failed |= run_window(rng, 24, 16, 2, phase=2)
    for rows in (32, 33, 64, 96, 100, 257):
        failed |= run_block(rng, rows)
    for rows, ch in ((33, 64), (64, 128), (100, 256), (32, 64), (257, 256), (16, 128)):
        failed |= run_block_expert(rng, rows, ch)
    for tokens, padded, heads in ((96, 128, 32), (20, 64, 2), (130, 192, 4), (64, 64, 1), (7, 64, 8)):
        failed |= run_vit(rng, tokens, padded, heads)
    print('CHECK FUSED FAIL' if failed else 'CHECK FUSED PASS')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
