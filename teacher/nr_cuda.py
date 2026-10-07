# The native kernels, driven through ctypes.
#
# nr_kernels.cu is standalone CUDA C with a C ABI - no torch header, cudart linked statically - so it is
# built once with whatever nvcc is on the machine and loaded with ctypes. Tensors are passed as data
# pointers plus element strides (the kernels are fully strided, so the same permuted views the rest of the
# port uses go straight in). The tables are nr_triton's, built from the oracle.
#
# The Python API mirrors nr_triton: fp8_gemm / fp8_gemm_full / fp8_gemm_partitioned / f16_gemm /
# window_attention.

import ctypes
import os
import subprocess

import torch

import nr_numerics as oracle

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, 'nr_kernels.cu')
_SO = os.path.join(_HERE, 'nr_kernels.so')

_lib = None
_tables = {}


def tables():
    """The publication tables on the device: the E4M3 round trip, expWeight and vitExpWeight, all from
    the oracle."""
    device = torch.cuda.current_device()
    if device not in _tables:
        roundtrip = torch.empty(65536, dtype=torch.float16)
        expw = torch.empty(65536, dtype=torch.float16)
        vitexpw = torch.empty(65536, dtype=torch.float16)
        for chunk in range(0, 65536, 8192):
            lo, hi = chunk, chunk + 8192
            roundtrip[lo:hi] = torch.tensor(
                [oracle.e4m3_to_number(oracle.e4m3_from_f16_bits(b)) for b in range(lo, hi)],
                dtype=torch.float16)
            expw[lo:hi] = torch.tensor(
                [oracle.exp_weight(oracle.f16_to_number(b)) for b in range(lo, hi)],
                dtype=torch.float16)
            vitexpw[lo:hi] = torch.tensor(
                [oracle.vit_exp_weight(oracle.f16_to_number(b)) for b in range(lo, hi)],
                dtype=torch.float16)
        _tables[device] = (roundtrip.cuda(), expw.cuda(), vitexpw.cuda())
    return _tables[device][:2]


def vit_expw():
    """The ViT's expWeight variant table."""
    tables()
    return _tables[torch.cuda.current_device()][2]


def _build():
    nvcc = os.environ.get('NVCC')
    if not nvcc:
        for candidate in ('/usr/local/cuda-11.8/bin/nvcc', '/usr/local/cuda/bin/nvcc', 'nvcc'):
            if candidate == 'nvcc' or os.path.exists(candidate):
                nvcc = candidate
                break
    cmd = [nvcc, '-O3', '-std=c++14', '-arch=sm_86', '-cudart=static', '-shared',
           '-Xcompiler', '-fPIC', _SRC, '-o', _SO]
    print('building nr_kernels.so:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def lib():
    global _lib
    if _lib is None:
        if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(_SRC):
            _build()
        _lib = ctypes.CDLL(_SO)
        ptr = ctypes.c_void_p
        ll = ctypes.c_longlong
        i = ctypes.c_int
        _lib.nr_fp8_gemm.restype = i
        _lib.nr_fp8_gemm.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
                                     ll, ll, ll, ll] + [ll] * 18 + [i, ll, i, i, ptr]
        _lib.nr_f16_gemm.restype = i
        _lib.nr_f16_gemm.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr,
                                     ll, ll, ll, ll] + [ll] * 15 + [i, ptr]
        _lib.nr_window_attention.restype = i
        _lib.nr_window_attention.argtypes = [ptr, ptr, ptr, ptr, ptr,
                                             i, i, i, i, i, i] + [ll] * 9 + [ptr]
        _lib.nr_window_fused.restype = i
        _lib.nr_window_fused.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr,
                                         i, i, i, i, i, i] + [ll] * 9 + [ptr]
        _lib.nr_block_ffn.restype = i
        _lib.nr_block_ffn.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
                                      ll] + [ll] * 5 + [i, ptr]
        _lib.nr_block_ffn_expert.restype = i
        _lib.nr_block_ffn_expert.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
                                             ll, ll] + [ll] * 5 + [i, ptr]
        _lib.nr_vit_fused.restype = i
        _lib.nr_vit_fused.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr,
                                      i, i, i] + [ll] * 6 + [ptr]
        _lib.nr_block_debug.restype = i
        _lib.nr_block_debug.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
                                        ll, ll, ll, ptr]
        _lib.nr_block_expert_debug.restype = i
        _lib.nr_block_expert_debug.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
                                               ll, ll, ll, ptr]
    return _lib


def available():
    return torch.cuda.is_available()


def _p(t):
    return None if t is None else t.data_ptr()


def _s(t, k):
    return 0 if t is None else t.stride(k)


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _check(status, what):
    if status != 0:
        raise RuntimeError(f'{what}: CUDA error {status}')


# flags for nr_fp8_gemm: 1 seed, 2 residual, 4 swizzle A, 8 silu, 16 write raw, 32 write e4.
F_SEED, F_RES, F_SWIZZLE, F_SILU, F_RAW, F_E4 = 1, 2, 4, 8, 16, 32

# The fused entry points the network prefers over the step-by-step path (env switches for bisection).
FUSED_PREP = os.environ.get('NR_FUSED_PREP', '1') == '1'      # window_attention_raw
FUSED_BLOCK = os.environ.get('NR_FUSED_BLOCK', '1') == '1'    # block_ffn
FUSED_EXPERT = os.environ.get('NR_FUSED_EXPERT', '1') == '1'  # block_ffn_expert
FUSED_VIT = os.environ.get('NR_FUSED_VIT', '1') == '1'        # vit_attention_raw


# Kernel tuning knobs (for A/B experiments): K step of the chain loop, and the output tile.
_KSTEP = int(os.environ.get('NR_KSTEP', '64'))
_TILE = int(os.environ.get('NR_TILE', '0'))


def _fp8_launch(x, w, seed, res, aux, out_raw, out_e4, silu, swizzle, partition=0):
    table, _ = tables()
    b, rows, k = x.shape
    n = w.shape[2]
    flags = 0
    if seed is not None:
        flags |= F_SEED
    if res is not None:
        flags |= F_RES
    if swizzle:
        flags |= F_SWIZZLE
    if silu:
        flags |= F_SILU
    if out_raw is not None:
        flags |= F_RAW
    if out_e4 is not None:
        flags |= F_E4
    status = lib().nr_fp8_gemm(
        _p(x), _p(w), _p(seed), _p(res), _p(aux), _p(out_raw), _p(out_e4), _p(table),
        rows, k, n, b,
        _s(x, 0), _s(x, 1), _s(x, 2),
        _s(w, 0), _s(w, 1), _s(w, 2),
        _s(seed, 0), _s(seed, 1), _s(seed, 2),
        _s(res, 0), _s(res, 1), _s(res, 2),
        _s(out_raw, 0), _s(out_raw, 1), _s(out_raw, 2),
        _s(out_e4, 0), _s(out_e4, 1), _s(out_e4, 2),
        flags, partition, _KSTEP, _TILE, _stream())
    _check(status, 'nr_fp8_gemm')


def fp8_gemm(x, w, seed=None, partition=0, **_):
    """The plain chain (what the oracle checks): x [R, K]/[B, R, K], w [K, N]/[B, K, N], seed like out.
    `partition` runs independent chunk chains inside the kernel, combined with half adds."""
    batched = x.dim() == 3
    if not batched:
        x = x.unsqueeze(0)
        w = w.unsqueeze(0)
        seed = seed.unsqueeze(0) if seed is not None else None
    if partition:
        k = x.shape[2]
        assert k % partition == 0, f'partition {partition} must divide K {k}'
    out = torch.empty(x.shape[0], x.shape[1], w.shape[2], dtype=torch.float16, device=x.device)
    _fp8_launch(x, w, seed, None, None, out, None, False, False, partition)
    return out if batched else out.squeeze(0)


def fp8_gemm_full(x, w, residual=None, aux=None, silu=False, quantize=True, raw=False,
                  swizzle=True, out_e4=None, out_raw=None, partition=0, **_):
    """
    The GEMM as the graph calls it: x [B, R, K], w [B, K, N] (strided views are fine - the batch stride is
    the per-batch row/column offset), residual [B, R, N] seeded through the per-column aux scale, and the
    epilogue in the same launch. Returns (e4, raw).
    """
    if out_raw is None and raw:
        out_raw = torch.empty(x.shape[0], x.shape[1], w.shape[2], dtype=torch.float16, device=x.device)
    if out_e4 is None and quantize:
        out_e4 = torch.empty(x.shape[0], x.shape[1], w.shape[2], dtype=torch.float16, device=x.device)
    _fp8_launch(x, w, None, residual, aux, out_raw, out_e4, silu, swizzle, partition)
    return out_e4, out_raw


def fp8_gemm_partitioned(x, w, partition, residual=None, aux=None, silu=False,
                         quantize=True, raw=False, out_e4=None, out_raw=None, **_):
    """The ViT's partitioned GEMMs: independent chains per K chunk combined with half adds, epilogue last.
    All of it happens inside the kernel now (one launch, no torch combines)."""
    k = x.shape[2]
    assert k % partition == 0, f'partition {partition} must divide K {k}'
    if out_raw is None and raw:
        out_raw = torch.empty(x.shape[0], x.shape[1], w.shape[2], dtype=torch.float16, device=x.device)
    if out_e4 is None and quantize:
        out_e4 = torch.empty(x.shape[0], x.shape[1], w.shape[2], dtype=torch.float16, device=x.device)
    _fp8_launch(x, w, None, residual, aux, out_raw, out_e4, silu, False, partition)
    return out_e4, out_raw


def f16_gemm(x, w, seed=None, quantize=True, raw=True, out_raw=None, out_e4=None, **_):
    """The f16 matrix multiply: x [R, K]/[B, R, K] raw halves, w [K, N]/[B, K, N]. Returns (e4, raw)."""
    table, _ = tables()
    batched = x.dim() == 3
    if not batched:
        x = x.unsqueeze(0)
        w = w.unsqueeze(0)
        seed = seed.unsqueeze(0) if seed is not None else None
    if out_raw is None and raw:
        out_raw = torch.empty(x.shape[0], x.shape[1], w.shape[2], dtype=torch.float16, device=x.device)
    if out_e4 is None and quantize:
        out_e4 = torch.empty(x.shape[0], x.shape[1], w.shape[2], dtype=torch.float16, device=x.device)
    b, rows, k = x.shape
    n = w.shape[2]
    # The f16 kernel's flag bits are its own: 1 seed, 2 write raw, 4 write e4.
    flags = (1 if seed is not None else 0) | (2 if out_raw is not None else 0) \
            | (4 if out_e4 is not None else 0)
    status = lib().nr_f16_gemm(
        _p(x), _p(w), _p(seed), _p(out_raw), _p(out_e4), _p(table),
        rows, k, n, b,
        _s(x, 0), _s(x, 1), _s(x, 2),
        _s(w, 0), _s(w, 1), _s(w, 2),
        _s(seed, 0), _s(seed, 1), _s(seed, 2),
        _s(out_raw, 0), _s(out_raw, 1), _s(out_raw, 2),
        _s(out_e4, 0), _s(out_e4, 1), _s(out_e4, 2),
        flags, _stream())
    _check(status, 'nr_f16_gemm')
    if not batched:
        out_e4 = out_e4.squeeze(0) if out_e4 is not None else None
        out_raw = out_raw.squeeze(0) if out_raw is not None else None
    return out_e4, out_raw


def window_attention(qkv, prior, width, height, heads, shift_x, shift_y, out=None, **_):
    """
    The fused window attention: qkv [rows, heads, 96] f16 (cosine-normalized q/k/v published), prior
    [heads, 64, 64] f16 in natural order, out [rows, heads, 32] f16 (E4M3 values).
    """
    table, expw = tables()
    rows = qkv.shape[0]
    if out is None:
        out = torch.zeros((rows, heads, 32), dtype=torch.float16, device=qkv.device)
    windows_x = (width + shift_x + 7) // 8
    status = lib().nr_window_attention(
        _p(qkv), _p(prior), _p(table), _p(expw), _p(out),
        width, height, shift_x, shift_y, windows_x, heads,
        qkv.stride(0), qkv.stride(1), qkv.stride(2),
        prior.stride(0), prior.stride(1), prior.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        _stream())
    _check(status, 'nr_window_attention')
    return out


def window_attention_raw(qkv, prior, scales, width, height, heads, shift_x, shift_y, out=None, **_):
    """
    The fused window attention from RAW qkv: the cosine normalization and the q/k/v publications happen
    inside the kernel. qkv [rows, heads, 96] f16 raw, prior [heads, 64, 64] f16 in natural order,
    scales [heads] f16 (round_f16 of the learned per-head scales), out [rows, heads, 32] f16 (E4M3).
    """
    table, expw = tables()
    rows = qkv.shape[0]
    if out is None:
        out = torch.zeros((rows, heads, 32), dtype=torch.float16, device=qkv.device)
    windows_x = (width + shift_x + 7) // 8
    status = lib().nr_window_fused(
        _p(qkv), _p(prior), _p(scales), _p(table), _p(expw), _p(out),
        width, height, shift_x, shift_y, windows_x, heads,
        qkv.stride(0), qkv.stride(1), qkv.stride(2),
        prior.stride(0), prior.stride(1), prior.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        _stream())
    _check(status, 'nr_window_fused')
    return out


def block_ffn(state, w1, w2, w3, res, aux, out_raw, out_e4, out_qkv, **_):
    """
    The 32-channel block's FFN chain in one launch: expand 32->128 (SiLU, published), contract 128->32
    (the residual seeded through aux, raw + E4 out), qkv 32->96 (raw out). The intermediates stay in
    shared memory. All tensors f16, row-contiguous [rows, C].
    """
    table, _ = tables()
    rows = state.shape[0]
    flags = (1 if out_raw is not None else 0) | (2 if out_e4 is not None else 0)
    status = lib().nr_block_ffn(
        _p(state), _p(w1), _p(w2), _p(w3), _p(res), _p(aux), _p(table),
        _p(out_raw), _p(out_e4), _p(out_qkv),
        rows,
        state.stride(0), res.stride(0), out_raw.stride(0) if out_raw is not None else 0,
        out_e4.stride(0) if out_e4 is not None else 0, out_qkv.stride(0),
        flags, _stream())
    _check(status, 'nr_block_ffn')


def vit_attention_raw(qkv, learned, head_scale, tokens, padded, heads, out=None, **_):
    """
    The fused ViT attention from RAW qkv: the cosine norm and the q/k/v publications (three roundings on
    q), the scores with vitExpWeight, the chunked 64-wide softmax trees with the padding correction, and
    the value chain all inside one kernel. qkv [tokens, heads, 96] f16 raw, learned [heads] f16,
    head_scale a 0-dim f16 (round_f16(sqrt(32))), out [tokens, heads, 32] f16 (E4M3 values).
    """
    table, _ = tables()
    if out is None:
        out = torch.zeros((tokens, heads, 32), dtype=torch.float16, device=qkv.device)
    status = lib().nr_vit_fused(
        _p(qkv), _p(learned), _p(head_scale), _p(table), _p(vit_expw()), _p(out),
        tokens, padded, heads,
        qkv.stride(0), qkv.stride(1), qkv.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        _stream())
    _check(status, 'nr_vit_fused')
    return out


def block_ffn_expert(state, w1, w2, w3, w4, res, aux, out_raw, out_e4, out_qkv, **_):
    """
    The expert block's FFN chain in one launch: expand (E broadcast experts ch->128, SiLU, published),
    narrow (E x 128->32, published), contract (ch->ch, seeded, raw + E4), qkv (ch->3ch, raw). The
    intermediates stay in shared memory. All tensors f16, row-contiguous [rows, C].
    """
    table, _ = tables()
    rows, ch = state.shape[0], state.shape[1]
    flags = (1 if out_raw is not None else 0) | (2 if out_e4 is not None else 0)
    status = lib().nr_block_ffn_expert(
        _p(state), _p(w1), _p(w2), _p(w3), _p(w4), _p(res), _p(aux), _p(table),
        _p(out_raw), _p(out_e4), _p(out_qkv),
        rows, ch,
        state.stride(0), res.stride(0), out_raw.stride(0) if out_raw is not None else 0,
        out_e4.stride(0) if out_e4 is not None else 0, out_qkv.stride(0),
        flags, _stream())
    _check(status, 'nr_block_ffn_expert')


def block_debug(state, w1, w2, w3, res, aux, dbg):
    """Debug probe: the full block sequence with ff and cc dumped as float32."""
    table, _ = tables()
    rows = state.shape[0]
    status = lib().nr_block_debug(_p(state), _p(w1), _p(w2), _p(w3), _p(res), _p(aux), _p(table),
                                  _p(dbg), rows, state.stride(0), res.stride(0), _stream())
    _check(status, 'nr_block_debug')


def block_expert_debug(state, w1, w2, w3, res, aux, dbg):
    """Debug probe: the expert sequence with ff/nn/cc dumped as float32 (ch=64)."""
    table, _ = tables()
    rows = state.shape[0]
    status = lib().nr_block_expert_debug(_p(state), _p(w1), _p(w2), _p(w3), _p(res), _p(aux),
                                         _p(table), _p(dbg), rows, state.stride(0),
                                         res.stride(0), _stream())
    _check(status, 'nr_block_expert_debug')
