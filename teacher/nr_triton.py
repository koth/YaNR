# The fused kernels: the fixed-point GEMM with its whole F13 chain and epilogue in one Triton launch, and
# the window attention - scores, expWeight, the 64-wide softmax tree, the value fold, the publications -
# in another.
#
# The arithmetic is exactly the reference contract (docs/numerics.md, adaFp8Fdpa16 and friends): every
# product is exact in f32, the alignment and de-alignment are powers of two built as bit patterns (no exp2
# approximation - the trick shaders/numerics.wgsl and PRODUCTION_FDPA use), the terms truncate toward zero,
# the sum of 17 terms stays under 2^20 so f32 holds it exactly whatever the summation order, and each
# 16-product group rounds to half once and becomes the next group's accumulator.
#
# Publications are table lookups against the oracle's own tables (the E4M3 round trip, expWeight): a table
# built from nr_numerics cannot drift from it. The half bit pattern of a register value comes from a
# bitcast of its f16 form.
#
# K partitions (the ViT contract) are the reference's chunked accumulation: independent chains per chunk
# combined with half adds, so they are a loop around the kernel rather than something inside it.

import torch
import triton
import triton.language as tl

import nr_numerics as oracle

_tables = {}


def tables():
    """The publication tables on the device: the E4M3 round trip and expWeight, both from the oracle."""
    device = torch.cuda.current_device()
    if device not in _tables:
        roundtrip = torch.empty(65536, dtype=torch.float16)
        expw = torch.empty(65536, dtype=torch.float16)
        for chunk in range(0, 65536, 8192):
            lo, hi = chunk, chunk + 8192
            roundtrip[lo:hi] = torch.tensor(
                [oracle.e4m3_to_number(oracle.e4m3_from_f16_bits(b)) for b in range(lo, hi)],
                dtype=torch.float16)
            expw[lo:hi] = torch.tensor(
                [oracle.exp_weight(oracle.f16_to_number(b)) for b in range(lo, hi)],
                dtype=torch.float16)
        _tables[device] = (roundtrip.cuda(), expw.cuda())
    return _tables[device]


@triton.jit
def _trunc_f32(x):
    return tl.where(x < 0, tl.math.ceil(x), tl.math.floor(x))


@triton.jit
def _hadd(a, b):
    """One publication-grid addition: the exact f32 sum of two halves, rounded to half."""
    return (a + b).to(tl.float16)


@triton.jit
def _mp_cubic_silu(value):
    """MpCubicSiLU: five half publications, the inner products exact in f32."""
    bounded = tl.maximum(tl.minimum(value, 4.0), -4.0).to(tl.float16).to(tl.float32)
    absolute = tl.abs(bounded)
    inner = (absolute * -0.055908203125 + 0.447265625).to(tl.float16).to(tl.float32)
    polynomial = (bounded * inner + 0.89453125).to(tl.float16).to(tl.float32)
    return (value * polynomial).to(tl.float16).to(tl.float32)


@triton.jit
def _f16_bits(value):
    return value.to(tl.float16).to(tl.int16, bitcast=True) & 0xffff


@triton.jit
def _e4_exponent(v32):
    return tl.maximum((((v32.to(tl.int32, bitcast=True) >> 23) & 0xff) - 127), -6)


@triton.jit
def _fp8_group(acc, a, b):
    """One 16-product group of the FP8 chain. a [..., 16], b [..., 16, N], acc [..., N] (f32-held values);
    returns the new accumulator (f32 holding an f16 value)."""
    ea = _e4_exponent(a)
    eb = _e4_exponent(b)
    esum = ea[:, :, None] + eb[None, :, :]
    covered = (a[:, :, None] != 0.0) & (b[None, :, :] != 0.0)
    accbits = acc.to(tl.int32, bitcast=True)
    accexp = (((accbits >> 23) & 0xff) - 127)
    seeded = tl.where(acc == 0.0, -21, tl.maximum(accexp, -14))          # f13Start
    group_max = tl.max(tl.where(covered, esum, -21), axis=1)             # f13Cover
    mexp = tl.maximum(tl.minimum(tl.maximum(seeded, group_max), 16), -21)
    align = ((140 - mexp) << 23).to(tl.float32, bitcast=True)            # 2^(13 - maxexp)
    units = _trunc_f32(acc * align)                                      # f13Term of the accumulator
    prod = a[:, :, None] * b[None, :, :]                                 # exact in f32
    units += tl.sum(_trunc_f32(prod * align[:, None, :]), axis=1)        # exact integer sum
    de_align = ((mexp + 114) << 23).to(tl.float32, bitcast=True)         # 2^(maxexp - 13)
    stepped = (units * de_align).to(tl.float16).to(tl.float32)           # f13Finish
    finite = ((accbits >> 23) & 0xff) != 0xff                            # adaFp8Fdpa16's isFinite guard
    return tl.where(finite, stepped, acc)


# ------------------------------------------------------------------------------------------------------------
# The GEMM: the F13 chain plus the epilogue in one launch.
# ------------------------------------------------------------------------------------------------------------

@triton.jit
def fp8_chain_kernel(
    X, W, SEED, RES, AUX, OUT_RAW, OUT_E4, TABLE,
    R, K, N,
    sxb, sxr, sxk, swb, swk, swn,
    sdb, sdr, sdn, srb, srr, srn,
    srawb, srawr, srawn, se4b, se4r, se4n,
    HAS_SEED: tl.constexpr, HAS_RES: tl.constexpr,
    SWIZZLE_A: tl.constexpr, SILU: tl.constexpr, WRITE_RAW: tl.constexpr, WRITE_E4: tl.constexpr,
    TILES_N: tl.constexpr, TILES_R: tl.constexpr,
    TM: tl.constexpr, TN: tl.constexpr,
):
    """out[b, r, n] = the 16-product-group chain of x[b, r, :] against w[b, :, n] - one program per output
    tile, every program folded into grid.x (row tiles past 65535 do not fit grid.y). The batch strides are
    the per-batch row/column offsets of the strided views the caller passes."""
    pid = tl.program_id(0)
    batch = pid // (TILES_N * TILES_R)
    pid_r = (pid // TILES_N) % TILES_R
    ntile = pid % TILES_N
    offs_r = pid_r * TM + tl.arange(0, TM)
    offs_n = ntile * TN + tl.arange(0, TN)
    offs_k = tl.arange(0, 16)
    mask_r = offs_r < R
    mask_n = offs_n < N
    full = mask_r[:, None] & mask_n[None, :]

    acc = tl.zeros((TM, TN), dtype=tl.float32)
    if HAS_SEED:
        acc = tl.load(SEED + batch * sdb + offs_r[:, None] * sdr + offs_n[None, :] * sdn,
                      mask=full, other=0.0).to(tl.float32)
    if HAS_RES:
        res = tl.load(RES + batch * srb + offs_r[:, None] * srr + offs_n[None, :] * srn,
                      mask=full, other=0.0).to(tl.float32)
        aux = tl.load(AUX + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = ((res * aux[None, :]).to(tl.float16)).to(tl.float32)

    for kb in range(0, K, 16):
        kidx = kb + offs_k
        if SWIZZLE_A:
            # The A operand's within-32 index rotation, applied at load (packedInputIndex).
            within = kidx & 31
            kidx = (kidx & ~31) + (within & 16) + (((within & 15) >> 2) * 2) + (within & 1) \
                   + (((within >> 1) & 1) * 8)
        ax = tl.load(X + batch * sxb + offs_r[:, None] * sxr + kidx[None, :] * sxk,
                     mask=mask_r[:, None], other=0.0).to(tl.float32)      # [TM, 16]
        bw = tl.load(W + batch * swb + (kb + offs_k)[:, None] * swk + offs_n[None, :] * swn,
                     mask=mask_n[None, :], other=0.0).to(tl.float32)      # [16, TN]
        acc = _fp8_group(acc, ax, bw)

    value = acc
    if SILU:
        value = _mp_cubic_silu(value)
    if WRITE_RAW:
        tl.store(OUT_RAW + batch * srawb + offs_r[:, None] * srawr + offs_n[None, :] * srawn,
                 value.to(tl.float16), mask=full)
    if WRITE_E4:
        quantized = tl.load(TABLE + _f16_bits(value))
        tl.store(OUT_E4 + batch * se4b + offs_r[:, None] * se4r + offs_n[None, :] * se4n,
                 quantized, mask=full)


# ------------------------------------------------------------------------------------------------------------
# The window attention. Slots are natural row-major tokens of the 8x8 window; the keys, the softmax tree
# and the value fold walk the 4x4-tiled ("physical") key order, which is what they are specified in.
# ------------------------------------------------------------------------------------------------------------

@triton.jit
def _inverse_tiled_token(token):
    """Physical token -> natural token inside the 8x8 window."""
    tile = token >> 4
    within = token & 15
    x = (tile & 1) * 4 + (within & 3)
    y = (tile >> 1) * 4 + (within >> 2)
    return y * 8 + x


@triton.jit
def _node2(S, KO, A: tl.constexpr, B: tl.constexpr):
    """One softmax-tree node over two physical lanes: the pair sum, rounded to half."""
    return tl.sum(S * ((KO == A) | (KO == B)), axis=1).to(tl.float16)


@triton.jit
def window_attention_kernel(
    QKV, PRIOR, TABLE, EXPW, OUT,
    width, height, shift_x, shift_y, windows_x, heads,
    q_row_s, q_head_s, q_ch_s, p_head_s, p_q_s, p_k_s, o_row_s, o_head_s, o_ch_s,
    TQ: tl.constexpr,
):
    program = tl.program_id(0)
    qb = tl.program_id(1)
    task = program // heads
    head = program % heads
    wx = (task % windows_x) * 8 - shift_x
    wy = (task // windows_x) * 8 - shift_y

    # The queries: natural slots of the window rows qb*TQ .. qb*TQ+TQ-1.
    q_slot = qb * TQ + tl.arange(0, TQ)
    qx = wx + (q_slot & 7)
    qy = wy + (q_slot >> 3)
    q_valid = (qx >= 0) & (qx < width) & (qy >= 0) & (qy < height)
    q_row = qy * width + qx

    # The keys in physical order: physical p is the natural slot inv(p), at its field position.
    p_off = tl.arange(0, 64)
    nat = _inverse_tiled_token(p_off)
    kx = wx + (nat & 7)
    ky = wy + (nat >> 3)
    k_valid = (kx >= 0) & (kx < width) & (ky >= 0) & (ky < height)
    k_row = ky * width + kx

    # The score matrix multiply carries the learned prior as its accumulator; the prior column of
    # physical key p is the natural key inv(p). The prior is read for every key - out-of-field keys keep
    # it as their score (windowAttendRef), so they still count in the softmax denominator.
    prior = tl.load(PRIOR + head * p_head_s + q_slot[:, None] * p_q_s + nat[None, :] * p_k_s,
                    mask=q_valid[:, None], other=0.0).to(tl.float32)
    acc = prior.to(tl.float16).to(tl.float32)

    for cg in tl.static_range(2):                      # 32 channels in two 16-product groups
        ch = cg * 16 + tl.arange(0, 16)
        q = tl.load(QKV + q_row[:, None] * q_row_s + head * q_head_s + ch[None, :] * q_ch_s,
                    mask=q_valid[:, None], other=0.0).to(tl.float32)      # [TQ, 16]
        k = tl.load(QKV + k_row[:, None] * q_row_s + head * q_head_s + ch[None, :] * q_ch_s,
                    mask=k_valid[:, None], other=0.0).to(tl.float32)      # [64, 16]
        acc = _fp8_group(acc, q, tl.trans(k))

    scores = tl.load(EXPW + _f16_bits(acc))             # expWeight, one table lookup per score

    # The 64-wide softmax denominator: the pair tree over physical lanes, half adds throughout
    # (windowAttendRef's pair/even/odd/total).
    ko = tl.arange(0, 64)
    a0 = _node2(scores, ko, 0, 8);   a1 = _node2(scores, ko, 1, 9)
    a2 = _node2(scores, ko, 2, 10);  a3 = _node2(scores, ko, 3, 11)
    a4 = _node2(scores, ko, 4, 12);  a5 = _node2(scores, ko, 5, 13)
    a6 = _node2(scores, ko, 6, 14);  a7 = _node2(scores, ko, 7, 15)
    b0 = _node2(scores, ko, 16, 24); b1 = _node2(scores, ko, 17, 25)
    b2 = _node2(scores, ko, 18, 26); b3 = _node2(scores, ko, 19, 27)
    b4 = _node2(scores, ko, 20, 28); b5 = _node2(scores, ko, 21, 29)
    b6 = _node2(scores, ko, 22, 30); b7 = _node2(scores, ko, 23, 31)
    c0 = _node2(scores, ko, 32, 40); c1 = _node2(scores, ko, 33, 41)
    c2 = _node2(scores, ko, 34, 42); c3 = _node2(scores, ko, 35, 43)
    c4 = _node2(scores, ko, 36, 44); c5 = _node2(scores, ko, 37, 45)
    c6 = _node2(scores, ko, 38, 46); c7 = _node2(scores, ko, 39, 47)
    d0 = _node2(scores, ko, 48, 56); d1 = _node2(scores, ko, 49, 57)
    d2 = _node2(scores, ko, 50, 58); d3 = _node2(scores, ko, 51, 59)
    d4 = _node2(scores, ko, 52, 60); d5 = _node2(scores, ko, 53, 61)
    d6 = _node2(scores, ko, 54, 62); d7 = _node2(scores, ko, 55, 63)
    p0 = _hadd(_hadd(_hadd(a0, b0), c0), d0)
    p1 = _hadd(_hadd(_hadd(a1, b1), c1), d1)
    p2 = _hadd(_hadd(_hadd(a2, b2), c2), d2)
    p3 = _hadd(_hadd(_hadd(a3, b3), c3), d3)
    p4 = _hadd(_hadd(_hadd(a4, b4), c4), d4)
    p5 = _hadd(_hadd(_hadd(a5, b5), c5), d5)
    p6 = _hadd(_hadd(_hadd(a6, b6), c6), d6)
    p7 = _hadd(_hadd(_hadd(a7, b7), c7), d7)
    even = _hadd(_hadd(_hadd(p0, p2), p4), p6)
    odd = _hadd(_hadd(_hadd(p1, p3), p5), p7)
    total = _hadd(even, odd)
    reciprocal = (1.0 / total.to(tl.float32)).to(tl.float16).to(tl.float32)

    weights = tl.load(TABLE + _f16_bits((scores * reciprocal[:, None]).to(tl.float16)))

    # The value fold: four 16-key groups over physical keys, each group rounding through half.
    v = tl.load(QKV + k_row[:, None] * q_row_s + head * q_head_s + (64 + tl.arange(0, 32))[None, :] * q_ch_s,
                mask=k_valid[:, None], other=0.0).to(tl.float32)           # [64, 32]
    prodv = weights[:, :, None] * v[None, :, :]                            # [TQ, 64, 32]
    esumv = _e4_exponent(weights.to(tl.float32))[:, :, None] + _e4_exponent(v)[None, :, :]
    coveredv = (weights[:, :, None] != 0.0) & (v[None, :, :] != 0.0)
    accv = tl.zeros((TQ, 32), dtype=tl.float32)
    for g in tl.static_range(4):
        group = ((p_off >= g * 16) & (p_off < g * 16 + 16))[:, None]       # [64, 1]
        accbits = accv.to(tl.int32, bitcast=True)
        accexp = (((accbits >> 23) & 0xff) - 127)
        seeded = tl.where(accv == 0.0, -21, tl.maximum(accexp, -14))
        group_max = tl.max(tl.where(group & coveredv, esumv, -21), axis=1)  # [TQ, 32]
        mexp = tl.maximum(tl.minimum(tl.maximum(seeded, group_max), 16), -21)
        align = ((140 - mexp) << 23).to(tl.float32, bitcast=True)
        units = _trunc_f32(accv * align)
        units += tl.sum(_trunc_f32(tl.where(group, prodv, 0.0) * align[:, None, :]), axis=1)
        de_align = ((mexp + 114) << 23).to(tl.float32, bitcast=True)
        finite = ((accbits >> 23) & 0xff) != 0xff
        accv = tl.where(finite, (units * de_align).to(tl.float16).to(tl.float32), accv)

    out = tl.load(TABLE + _f16_bits(accv))
    tl.store(OUT + q_row[:, None] * o_row_s + head * o_head_s + tl.arange(0, 32)[None, :] * o_ch_s,
             out, mask=q_valid[:, None])


# ------------------------------------------------------------------------------------------------------------
# The f16 GEMM (the input adapter and the head): 8-product groups, 24 fractional bits, and the exact
# integer -> half publication (fixedToF16) instead of a rounding of the sum.
# ------------------------------------------------------------------------------------------------------------

@triton.jit
def _round_shift_right_even_t(value, shift):
    """Shift right with round-to-nearest-even, ties away from even - the reference's roundShiftRightEven."""
    s = tl.maximum(shift, 0)
    q = value >> s
    rem = value - (q << s)
    halfway = 1 << tl.maximum(s - 1, 0)
    up = (rem > halfway) | ((rem == halfway) & ((q & 1) == 1))
    return q + tl.where(up, 1, 0)


@triton.jit
def _fixed_to_f16(fixed, binexp):
    """Exact signed integer times 2^binaryExponent -> half, as the f16 bit pattern (reference's
    fixedToF16). Zero answers 0 before any of the field arithmetic sees it."""
    negative = fixed < 0
    mag = tl.where(negative, -fixed, fixed)
    msb = (((mag.to(tl.float64).to(tl.int64, bitcast=True) >> 52) & 0x7ff) - 1023).to(tl.int32)
    value_exp = msb + binexp

    sig = tl.where(msb > 10,
                   _round_shift_right_even_t(mag, msb - 10),
                   mag << tl.maximum(10 - msb, 0))
    carried = sig >= 2048
    sig = tl.where(carried, 1024, sig)
    value_exp = value_exp + tl.where(carried, 1, 0)
    normal_bits = tl.where(value_exp >= 16, 0x7c00,
                           ((value_exp + 15) << 10) | (sig - 1024))

    sub_scale = binexp + 24
    mantissa = tl.where(sub_scale >= 0,
                        mag << sub_scale,
                        _round_shift_right_even_t(mag, -sub_scale))
    sub_bits = tl.minimum(mantissa, 1024)

    half = tl.where(negative, 1, 0) << 15
    half = half | tl.where(value_exp >= -14, normal_bits, sub_bits)
    return tl.where(fixed == 0, 0, half)


@triton.jit
def f16_chain_kernel(
    X, W, SEED, OUT_RAW, OUT_E4, TABLE,
    R, K, N,
    sxb, sxr, sxk, swb, swk, swn, sdb, sdr, sdn, srawb, srawr, srawn, se4b, se4r, se4n,
    HAS_SEED: tl.constexpr, WRITE_E4: tl.constexpr,
    TILES_N: tl.constexpr, TILES_R: tl.constexpr,
    TM: tl.constexpr, TN: tl.constexpr,
):
    """out[b, r, n] = the 8-product-group F24 chain of x[b, r, :] against w[b, :, n]."""
    pid = tl.program_id(0)
    batch = pid // (TILES_N * TILES_R)
    pid_r = (pid // TILES_N) % TILES_R
    ntile = pid % TILES_N
    offs_r = pid_r * TM + tl.arange(0, TM)
    offs_n = ntile * TN + tl.arange(0, TN)
    offs_k = tl.arange(0, 8)
    mask_r = offs_r < R
    mask_n = offs_n < N
    full = mask_r[:, None] & mask_n[None, :]

    acc_bits = tl.zeros((TM, TN), dtype=tl.int32)
    if HAS_SEED:
        seed = tl.load(SEED + batch * sdb + offs_r[:, None] * sdr + offs_n[None, :] * sdn,
                       mask=full, other=0.0)
        acc_bits = seed.to(tl.int16, bitcast=True).to(tl.int32) & 0xffff

    for kb in range(0, K, 8):
        ax = tl.load(X + batch * sxb + offs_r[:, None] * sxr + (kb + offs_k)[None, :] * sxk,
                     mask=mask_r[:, None], other=0.0).to(tl.float32)      # [TM, 8]
        bw = tl.load(W + batch * swb + (kb + offs_k)[:, None] * swk + offs_n[None, :] * swn,
                     mask=mask_n[None, :], other=0.0).to(tl.float32)      # [8, TN]
        acc = (acc_bits.to(tl.int16).to(tl.float16, bitcast=True)).to(tl.float32)

        e16a = tl.maximum((((ax.to(tl.int32, bitcast=True) >> 23) & 0xff) - 127), -14)
        e16b = tl.maximum((((bw.to(tl.int32, bitcast=True) >> 23) & 0xff) - 127), -14)
        esum = e16a[:, :, None] + e16b[None, :, :]
        covered = (ax[:, :, None] != 0.0) & (bw[None, :, :] != 0.0)
        acc_is_zero = (acc_bits & 0x7fff) == 0
        accexp = tl.where((acc_bits & 0x7c00) == 0, -14, ((acc_bits >> 10) & 0x1f) - 15)
        seeded = tl.where(acc_is_zero, -21, accexp)                       # f13Start
        group_max = tl.max(tl.where(covered, esum, -21), axis=1)
        mexp = tl.maximum(seeded, group_max)

        align = ((151 - mexp) << 23).to(tl.float32, bitcast=True)         # 2^(24 - maxexp)
        units = _trunc_f32(acc * align).to(tl.int32)
        prod = ax[:, :, None] * bw[None, :, :]
        units += tl.sum(_trunc_f32(prod * align[:, None, :]).to(tl.int32), axis=1)

        stepped = _fixed_to_f16(units, mexp - 24)
        finite = ((acc_bits & 0x7c00) != 0x7c00)                          # the isFinite accumulator guard
        acc_bits = tl.where(finite, stepped, acc_bits)

    tl.store(OUT_RAW + batch * srawb + offs_r[:, None] * srawr + offs_n[None, :] * srawn,
             acc_bits.to(tl.int16).to(tl.float16, bitcast=True), mask=full)
    if WRITE_E4:
        quantized = tl.load(TABLE + (acc_bits & 0xffff))
        tl.store(OUT_E4 + batch * se4b + offs_r[:, None] * se4r + offs_n[None, :] * se4n,
                 quantized, mask=full)


# ------------------------------------------------------------------------------------------------------------
# Python API
# ------------------------------------------------------------------------------------------------------------

def available():
    return torch.cuda.is_available() and triton is not None


def _stride(t, i):
    return t.stride(i) if t is not None else 0


def auto_tiles(rows, n):
    """Tile choice by shape: tall row spaces want a wide row tile, column-heavy ones a tall column tile."""
    if rows >= 65536:
        return 32, 64, 8
    if n >= 256:
        return 8, 256, 8
    return 16, 64, 8


def _launch_gemm(x, w, seed, res, aux, out_raw, out_e4, table, silu, swizzle, tm, tn, num_warps):
    b, rows, k = x.shape
    n = w.shape[2]
    dummy = out_raw if out_raw is not None else out_e4
    tiles_n = triton.cdiv(n, tn)
    tiles_r = triton.cdiv(rows, tm)
    grid = (tiles_n * tiles_r * b,)
    fp8_chain_kernel[grid](
        x, w,
        seed if seed is not None else dummy,
        res if res is not None else dummy,
        aux if aux is not None else dummy,
        out_raw if out_raw is not None else dummy,
        out_e4 if out_e4 is not None else dummy,
        table,
        rows, k, n,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        _stride(seed, 0), _stride(seed, 1), _stride(seed, 2),
        _stride(res, 0), _stride(res, 1), _stride(res, 2),
        _stride(out_raw, 0), _stride(out_raw, 1), _stride(out_raw, 2),
        _stride(out_e4, 0), _stride(out_e4, 1), _stride(out_e4, 2),
        HAS_SEED=seed is not None, HAS_RES=res is not None, SWIZZLE_A=swizzle,
        SILU=silu, WRITE_RAW=out_raw is not None, WRITE_E4=out_e4 is not None,
        TILES_N=tiles_n, TILES_R=tiles_r, TM=tm, TN=tn, num_warps=num_warps)


def fp8_gemm(x, w, seed=None, partition=0, tm=None, tn=None, num_warps=None):
    """The plain chain (what the oracle checks): x [R, K]/[B, R, K], w [K, N]/[B, K, N], seed like out.
    Returns f16 [R, N] / [B, R, N]."""
    if tm is None or tn is None or num_warps is None:
        tm, tn, num_warps = auto_tiles(x.shape[-2], w.shape[-1])
    batched = x.dim() == 3
    if not batched:
        x = x.unsqueeze(0)
        w = w.unsqueeze(0)
        seed = seed.unsqueeze(0) if seed is not None else None
    if not partition:
        out = torch.empty(x.shape[0], x.shape[1], w.shape[2], dtype=torch.float16, device=x.device)
        _launch_gemm(x, w, seed, None, None, out, None, None, False, False, tm, tn, num_warps)
        return out if batched else out.squeeze(0)
    k = x.shape[2]
    assert k % partition == 0, f'partition {partition} must divide K {k}'
    combined = None
    for lo in range(0, k, partition):
        part = fp8_gemm(x[:, :, lo:lo + partition], w[:, lo:lo + partition, :],
                        seed if lo == 0 else None, partition=0, tm=tm, tn=tn, num_warps=num_warps)
        combined = part if combined is None else (combined.float() + part.float()).to(torch.float16)
    return combined if batched else combined.squeeze(0)


def fp8_gemm_full(x, w, residual=None, aux=None, silu=False, quantize=True, raw=False,
                  swizzle=True, out_e4=None, out_raw=None, tm=None, tn=None, num_warps=None):
    """
    The GEMM as the graph calls it: x [B, R, K], w [B, K, N] (strided views are fine - the batch stride is
    the per-batch row/column offset), residual [B, R, N] seeded through the per-column aux scale, and the
    epilogue (MpCubicSiLU, the E4M3 publication, the raw half) in the same launch. Returns (e4, raw).
    """
    if tm is None or tn is None or num_warps is None:
        tm, tn, num_warps = auto_tiles(x.shape[-2], w.shape[-1])
    table, _ = tables()
    if out_raw is None and raw:
        out_raw = torch.empty(x.shape[0], x.shape[1], w.shape[2], dtype=torch.float16, device=x.device)
    if out_e4 is None and quantize:
        out_e4 = torch.empty(x.shape[0], x.shape[1], w.shape[2], dtype=torch.float16, device=x.device)
    _launch_gemm(x, w, None, residual, aux, out_raw, out_e4, table, silu, swizzle, tm, tn, num_warps)
    return out_e4, out_raw


def fp8_gemm_partitioned(x, w, partition, residual=None, aux=None, silu=False,
                         quantize=True, raw=False, swizzle=True, out_e4=None, out_raw=None,
                         tm=None, tn=None, num_warps=None):
    """
    The ViT's partitioned GEMMs: independent chains per K chunk combined with half adds (the reference's
    partitionSums), then the epilogue - the rows there are small enough that the elementwise tail is free.
    When the caller passes out_e4/out_raw views, the results are copied into them.
    """
    if tm is None or tn is None or num_warps is None:
        tm, tn, num_warps = auto_tiles(x.shape[-2], w.shape[-1])
    k = x.shape[2]
    assert k % partition == 0, f'partition {partition} must divide K {k}'
    seed = None
    if residual is not None:
        seed = (residual.float() * aux.view(1, 1, -1).float()).to(torch.float16)
    combined = None
    for lo in range(0, k, partition):
        part = fp8_gemm(x[:, :, lo:lo + partition], w[:, lo:lo + partition, :],
                        seed if lo == 0 else None, partition=0, tm=tm, tn=tn, num_warps=num_warps)
        combined = part if combined is None else (combined.float() + part.float()).to(torch.float16)
    value = combined
    if silu:
        import nr_torch
        value = nr_torch.mp_cubic_silu_t(value)
    got_e4 = None
    if quantize:
        import nr_torch
        got_e4 = nr_torch.fp8_quant(value)
    got_raw = value if raw else None
    if out_e4 is not None and got_e4 is not None:
        out_e4.copy_(got_e4)
        got_e4 = out_e4
    if out_raw is not None and got_raw is not None:
        out_raw.copy_(got_raw)
        got_raw = out_raw
    return got_e4, got_raw


def f16_gemm(x, w, seed=None, quantize=True, raw=True, out_raw=None, out_e4=None,
             tm=None, tn=None, num_warps=None):
    """
    The f16 matrix multiply through the fused kernel: x [R, K] or [B, R, K] raw halves, w [K, N] or
    [B, K, N], optional seed like the output. Returns (out_e4 or None, out_raw or None).
    """
    if tm is None or tn is None or num_warps is None:
        tm, tn, num_warps = auto_tiles(x.shape[-2], w.shape[-1])
    batched = x.dim() == 3
    if not batched:
        x = x.unsqueeze(0)
        w = w.unsqueeze(0)
        seed = seed.unsqueeze(0) if seed is not None else None
    b, rows, k = x.shape
    n = w.shape[2]
    if out_raw is None and raw:
        out_raw = torch.empty(b, rows, n, dtype=torch.float16, device=x.device)
    if out_e4 is None and quantize:
        out_e4 = torch.empty(b, rows, n, dtype=torch.float16, device=x.device)
    table, _ = tables()
    dummy = out_raw if out_raw is not None else out_e4
    tiles_n = triton.cdiv(n, tn)
    tiles_r = triton.cdiv(rows, tm)
    f16_chain_kernel[(tiles_n * tiles_r * b,)](
        x, w,
        seed if seed is not None else dummy,
        out_raw if out_raw is not None else dummy,
        out_e4 if out_e4 is not None else dummy,
        table,
        rows, k, n,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        _stride(seed, 0), _stride(seed, 1), _stride(seed, 2),
        _stride(out_raw, 0), _stride(out_raw, 1), _stride(out_raw, 2),
        _stride(out_e4, 0), _stride(out_e4, 1), _stride(out_e4, 2),
        HAS_SEED=seed is not None, WRITE_E4=out_e4 is not None,
        TILES_N=tiles_n, TILES_R=tiles_r, TM=tm, TN=tn, num_warps=num_warps)
    if not batched:
        out_e4 = out_e4.squeeze(0) if out_e4 is not None else None
        out_raw = out_raw.squeeze(0) if out_raw is not None else None
    return out_e4, out_raw


def window_attention(qkv, prior, width, height, heads, shift_x, shift_y, out=None, num_warps=4):
    """
    The fused window attention. qkv [rows, heads, 96] f16 (cosine-normalized q/k/v published), prior
    [heads, 64, 64] f16 in natural order, out [rows, heads, 32] f16 (E4M3 values). One program per
    (window, head, 8 queries).
    """
    table, expw = tables()
    rows = qkv.shape[0]
    windows_x = (width + shift_x + 7) // 8
    windows_y = (height + shift_y + 7) // 8
    tasks = windows_x * windows_y
    tq = 8
    if out is None:
        out = torch.zeros((rows, heads, 32), dtype=torch.float16, device=qkv.device)
    grid = (tasks * heads, (64 + tq - 1) // tq)
    window_attention_kernel[grid](
        qkv, prior, table, expw, out,
        width, height, shift_x, shift_y, windows_x, heads,
        qkv.stride(0), qkv.stride(1), qkv.stride(2),
        prior.stride(0), prior.stride(1), prior.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        TQ=tq, num_warps=num_warps)
    return out
