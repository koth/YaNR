# The torch kernels: the same publication grid as nr_numerics.py, vectorized. Every function here is checked
# against the scalar oracle (check_numerics.py --torch) before it is allowed to run the network.
#
# The representation: activations and weights are torch.float16 tensors holding the *values* the buffers
# would hold - E4M3 tensors store f16 values that lie on the E4M3 grid, since every E4M3 value is exactly a
# half. `fp8_quant` is the publication (encode + decode through the exact tables), `round_f16` is .to(half).
#
# The dot products keep the tensor-core contract literally: per 16-product group, a shared exponent, the
# products aligned to 13 fractional bits, each term truncated toward zero, an exact sum, one rounding to
# half. The group result becomes the next group's accumulator. Nothing fuses, nothing reorders.
#
# The default backend runs each chain as one fused Triton kernel (nr_triton.py); the plain torch loops
# below stay as the reference the fused kernel is checked against (NR_TRITON=0 selects them).

import os

import torch

import nr_numerics as oracle

try:
    import nr_cuda as _cuda_backend
except Exception:                                    # no nvcc / build failure: next backend
    _cuda_backend = None

try:
    import nr_triton as _triton_backend
except Exception:                                    # triton missing or unusable: torch loops only
    _triton_backend = None

USE_TRITON = os.environ.get('NR_TRITON', '1') != '0' and (
    _cuda_backend is not None or _triton_backend is not None)


def fused_backend():
    """The fastest available kernel module (nr_cuda, else nr_triton), or None for the torch loops."""
    if not USE_TRITON:
        return None
    return _cuda_backend if _cuda_backend is not None else _triton_backend

_HALF_BITS = 65536


def _device():
    name = os.environ.get('NR_DEVICE')
    if name:
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DEVICE = _device()

_tables = {}


def _tables_for(device):
    key = str(device)
    if key not in _tables:
        encode = [oracle.e4m3_from_f16_bits(bits) for bits in range(_HALF_BITS)]
        decode = [oracle.e4m3_to_number(byte) for byte in range(256)]
        # The publication round trip as one table: f16 bits -> f16 value of the E4M3 quantization.
        roundtrip = [decode[encode[bits]] for bits in range(_HALF_BITS)]
        _tables[key] = {
            'encode': torch.tensor(encode, dtype=torch.uint8, device=device),
            'decode': torch.tensor(decode, dtype=torch.float16, device=device),
            'roundtrip': torch.tensor(roundtrip, dtype=torch.float16, device=device),
        }
    return _tables[key]


def to_device(*tensors):
    out = [t.to(DEVICE) if t is not None else None for t in tensors]
    return out[0] if len(out) == 1 else out


# ------------------------------------------------------------------------------------------------------------
# Bit-level helpers.
# ------------------------------------------------------------------------------------------------------------

def f16_bits_t(x16):
    """The half bit patterns of a f16 tensor, as int32 in 0..0xffff."""
    return x16.contiguous().view(torch.int16).to(torch.int32) & 0xffff


def f16_from_bits_t(bits):
    """f16 tensor from half bit patterns (int tensor)."""
    return (bits & 0xffff).to(torch.int16).view(torch.float16)


def round_f16_t(x):
    """One publication to the half grid."""
    return x.to(torch.float16)


def fp8_quant(x16):
    """The E4M3 publication as a value (fp8Domain): NaN -> +0, zeros keep their sign."""
    table = _tables_for(x16.device)['roundtrip']
    return table[f16_bits_t(x16)]


def e4m3_encode(x16):
    """The E4M3 publication as the byte a buffer stores."""
    table = _tables_for(x16.device)['encode']
    return table[f16_bits_t(x16)]


def e4m3_decode(code):
    """The f16 value of an E4M3 byte."""
    table = _tables_for(code.device)['decode']
    return table[code.to(torch.int64)]


# ------------------------------------------------------------------------------------------------------------
# Exponents: the shared exponent of each fixed-point group comes from these.
# ------------------------------------------------------------------------------------------------------------

def _exp_field(x16):
    """floor(log2|x|) of a f16 value, read off its f32 form (f16 subnormals are normal in f32)."""
    bits = x16.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xff) - 127


def e4m3_exponent_t(x16):
    return torch.maximum(_exp_field(x16), torch.full_like(_exp_field(x16), -6))


def f16_exponent_t(x16):
    return torch.maximum(_exp_field(x16), torch.full_like(_exp_field(x16), -14))


# ------------------------------------------------------------------------------------------------------------
# The elementwise publications.
# ------------------------------------------------------------------------------------------------------------

def mp_cubic_silu_t(x16):
    """MpCubicSiLU: five half publications; the inner products are exact in f32, so one f32 rounding is the
    fused multiply-add's own."""
    x = x16.to(torch.float32)
    bounded = torch.clamp(x, -4.0, 4.0).to(torch.float16)
    absolute = torch.abs(bounded)                                   # exact
    inner = (absolute.to(torch.float32) * -0.055908203125 + 0.447265625).to(torch.float16)
    polynomial = (bounded.to(torch.float32) * inner.to(torch.float32) + 0.89453125).to(torch.float16)
    return (x * polynomial.to(torch.float32)).to(torch.float16)


def _shifted_exponent_weight(x16, multiplier, bias, lo, hi, shift, increment):
    affine = (x16.to(torch.float32) * multiplier + bias).to(torch.float16)
    clamped = torch.clamp(affine, lo, hi)
    bits = f16_bits_t(clamped).to(torch.int64)
    pattern = ((bits << shift) + increment) & 0xffff
    return f16_from_bits_t(pattern)


def exp_weight_t(score16):
    """The window blocks' attention weight: an affine map dropped into the half exponent field."""
    return _shifted_exponent_weight(score16, 0.044921875, 1.30078125, 1.03125, 1.5693359375, 5, 0x8000)


def vit_exp_weight_t(score16):
    """The ViT's variant: a 4-bit shift, a different bias, and an affine evaluated in halves."""
    return _shifted_exponent_weight(score16, 0.08953857421875, 1.708984375, 1.439453125, 1.9775390625,
                                    4, 0x4000)


# ------------------------------------------------------------------------------------------------------------
# Fixed point -> half, for the f16 chains.
# ------------------------------------------------------------------------------------------------------------

def _round_shift_right_even_t(value, shift):
    """value >= 0 int64, shift >= 0 int64; round-to-nearest-even, ties away from even like the reference."""
    s = torch.clamp(shift, min=0)
    quotient = value >> s
    remainder = value - (quotient << s)
    halfway = torch.ones_like(value) << torch.clamp(s, min=1) - 1
    round_up = (remainder > halfway) | ((remainder == halfway) & ((quotient & 1) == 1))
    return quotient + round_up.to(value.dtype)


def fixed_to_f16_t(fixed_sum, binary_exponent):
    """Exact signed integer times 2^binaryExponent -> half, bit for bit as the reference's fixedToF16."""
    fixed_sum = fixed_sum.to(torch.int64)
    binary_exponent = torch.as_tensor(binary_exponent, dtype=torch.int64, device=fixed_sum.device)
    if binary_exponent.dim() == 0:
        binary_exponent = binary_exponent.expand_as(fixed_sum)

    zero = fixed_sum == 0
    negative = fixed_sum < 0
    magnitude = torch.abs(fixed_sum)

    # bit_length via frexp: value = m * 2^e with m in [0.5, 1), so msb = e - 1.
    _, exponent = torch.frexp(magnitude.to(torch.float64))
    msb = exponent.to(torch.int64) - 1

    value_exponent = msb + binary_exponent
    half_bits = negative.to(torch.int64) << 15

    # Normal path: value_exponent >= -14.
    wide = msb > 10
    shift = msb - 10
    significand = torch.where(wide,
                              _round_shift_right_even_t(magnitude, torch.clamp(shift, min=0)),
                              magnitude << torch.clamp(10 - msb, min=0))
    carried = significand >= 2048
    significand = torch.where(carried, torch.full_like(significand, 1024), significand)
    value_exponent = value_exponent + carried.to(torch.int64)
    normal_bits = torch.where(value_exponent >= 16, torch.full_like(value_exponent, 0x7c00),
                              ((value_exponent + 15) << 10) | (significand - 1024))

    # Subnormal path: value_exponent < -14.
    subnormal_scale = binary_exponent + 24
    mantissa = torch.where(subnormal_scale >= 0,
                           magnitude << torch.clamp(subnormal_scale, min=0),
                           _round_shift_right_even_t(magnitude, torch.clamp(-subnormal_scale, min=0)))
    subnormal_bits = torch.clamp(mantissa, max=1024)

    bits = half_bits | torch.where(value_exponent >= -14, normal_bits, subnormal_bits)
    bits = torch.where(zero, torch.zeros_like(bits), bits)
    return f16_from_bits_t(bits & 0xffff)


# ------------------------------------------------------------------------------------------------------------
# The dot products.
# ------------------------------------------------------------------------------------------------------------

MAX_GROUP_ELEMENTS = 1 << 25   # 32M products per group step: 128 MiB of f32 intermediates.


def _fp8_group_step(xg, xeg, wg, weg, acc, acce):
    """
    One 16-product group of the FP8 chain. xg [..., R, 16], wg [..., 16, N], acc [..., R, N] - all f16
    values (leading batch dims broadcast); xeg/weg/acce their exponents (int32). Returns the new
    accumulator [..., R, N] f16. A bare [R, 16] / [16, N] pair is accepted and returns [R, N].
    """
    flat = xg.dim() == 2
    if flat:
        xg, xeg = xg.unsqueeze(0), xeg.unsqueeze(0)
        wg, weg = wg.unsqueeze(0), weg.unsqueeze(0)
        acc, acce = acc.unsqueeze(0), acce.unsqueeze(0)
    prod = xg.unsqueeze(-1).to(torch.float32) * wg.unsqueeze(1).to(torch.float32)   # [B, R, 16, N]
    esum = xeg.unsqueeze(-1).to(torch.int32) + weg.unsqueeze(1).to(torch.int32)
    covered = (xg != 0).unsqueeze(-1) & (wg != 0).unsqueeze(1)
    esum = esum.masked_fill(~covered, -21)
    max_exponent = esum.amax(dim=2)                                                # [B, R, N]
    max_exponent = torch.maximum(max_exponent, acce)

    scale = torch.exp2((13 - max_exponent).to(torch.float32))
    units = torch.trunc(acc.to(torch.float32) * scale)
    units = units + torch.trunc(prod * scale.unsqueeze(2)).sum(dim=2)
    out = (units * torch.exp2((max_exponent - 13).to(torch.float32))).to(torch.float16)
    return out.squeeze(0) if flat else out


def fp8_dot_chain(x, w, seed=None, partition=0):
    """
    The FP8 matrix multiply: out[r, n] = the k-group chain of x[r, :] against w[:, n], every 16 products
    rounded through the half grid. x [R, K], w [K, N], seed [R, N] (the pre-rounded accumulator start).
    `partition` splits K into independently accumulated chunks combined with half adds, as the reference's
    gemmFp8Element does.
    """
    backend = fused_backend()
    if backend is not None:
        return backend.fp8_gemm(x, w, seed, partition=partition)
    return _fp8_dot_chain_torch(x, w, seed, partition=partition)


def _fp8_dot_chain_torch(x, w, seed=None, partition=0):
    R, K = x.shape
    N = w.shape[1]
    assert K % 16 == 0, f'K must be a multiple of 16, got {K}'
    if partition:
        assert K % partition == 0, f'partition {partition} must divide K {K}'

    xe = e4m3_exponent_t(x)
    we = e4m3_exponent_t(w)

    rows_per_chunk = max(1, MAX_GROUP_ELEMENTS // (16 * N))
    acc = seed.clone() if seed is not None else torch.zeros((R, N), dtype=torch.float16, device=x.device)
    part = None
    groups = K // 16
    for lo in range(0, R, rows_per_chunk):
        hi = min(lo + rows_per_chunk, R)
        a = acc[lo:hi]
        acce = torch.where(a == 0, torch.full_like(e4m3_exponent_t(a), -21), f16_exponent_t(a))
        for g in range(groups):
            a = _fp8_group_step(x[lo:hi, g * 16:(g + 1) * 16], xe[lo:hi, g * 16:(g + 1) * 16],
                                w[g * 16:(g + 1) * 16, :], we[g * 16:(g + 1) * 16, :],
                                a, acce)
            acce = torch.where(a == 0, torch.full_like(e4m3_exponent_t(a), -21), f16_exponent_t(a))
            if partition and ((g + 1) * 16) % partition == 0:
                if part is None:
                    part = torch.zeros((R, N), dtype=torch.float16, device=x.device)
                part[lo:hi] = (part[lo:hi].to(torch.float32) + a.to(torch.float32)).to(torch.float16)
                a = torch.zeros_like(a)
                acce = torch.full_like(acce, -21)
        acc[lo:hi] = a
    return part if partition else acc


def fp8_dot_chain_batched(x, w, seed=None):
    """
    The same chain with one matrix per batch: x [B, R, K], w [B, K, N], seed [B, R, N]. Used by the window
    attention, whose score matrix multiply carries the learned prior as its accumulator.
    """
    backend = fused_backend()
    if backend is not None:
        return backend.fp8_gemm(x, w, seed)
    return _fp8_dot_chain_batched_torch(x, w, seed)


def _fp8_dot_chain_batched_torch(x, w, seed=None):
    B, R, K = x.shape
    N = w.shape[2]
    assert K % 16 == 0
    xe = e4m3_exponent_t(x)
    we = e4m3_exponent_t(w)

    batches_per_chunk = max(1, MAX_GROUP_ELEMENTS // (R * 16 * N))
    acc = seed.clone() if seed is not None else torch.zeros((B, R, N), dtype=torch.float16, device=x.device)
    for lo in range(0, B, batches_per_chunk):
        hi = min(lo + batches_per_chunk, B)
        a = acc[lo:hi]
        acce = torch.where(a == 0, torch.full_like(e4m3_exponent_t(a), -21), f16_exponent_t(a))
        for g in range(K // 16):
            a = _fp8_group_step(x[lo:hi, :, g * 16:(g + 1) * 16], xe[lo:hi, :, g * 16:(g + 1) * 16],
                                w[lo:hi, g * 16:(g + 1) * 16, :], we[lo:hi, g * 16:(g + 1) * 16, :],
                                a, acce)
            acce = torch.where(a == 0, torch.full_like(e4m3_exponent_t(a), -21), f16_exponent_t(a))
        acc[lo:hi] = a
    return acc


def f16_dot_chain(x, w):
    """
    The f16 matrix multiply: out[r, n] = the 8-product F24 chain of x[r, :] against w[:, n]. x [R, K],
    w [K, N], K a multiple of 8. Only the input adapter and the head use this.
    """
    R, K = x.shape
    N = w.shape[1]
    assert K % 8 == 0
    xe = f16_exponent_t(x)
    we = f16_exponent_t(w)

    rows_per_chunk = max(1, MAX_GROUP_ELEMENTS // (8 * N))
    acc = torch.zeros((R, N), dtype=torch.float16, device=x.device)
    for lo in range(0, R, rows_per_chunk):
        hi = min(lo + rows_per_chunk, R)
        a = acc[lo:hi]
        acce = torch.where(a == 0, torch.full_like(e4m3_exponent_t(a), -21), f16_exponent_t(a))
        for g in range(K // 8):
            xg = x[lo:hi, g * 8:(g + 1) * 8]
            wg = w[g * 8:(g + 1) * 8, :]
            prod = xg.unsqueeze(-1).to(torch.float32) * wg.unsqueeze(0).to(torch.float32)   # [R, 8, N]
            esum = xe[lo:hi, g * 8:(g + 1) * 8].unsqueeze(-1).to(torch.int32) + \
                   we[g * 8:(g + 1) * 8, :].unsqueeze(0).to(torch.int32)
            covered = (xg != 0).unsqueeze(-1) & (wg != 0).unsqueeze(0)
            esum = esum.masked_fill(~covered, -21)
            max_exponent = esum.amax(dim=1)                    # over the 8 products
            max_exponent = torch.maximum(max_exponent, acce)

            scale = torch.exp2((24 - max_exponent).to(torch.float32))
            units = torch.trunc(a.to(torch.float32) * scale).to(torch.int64)
            units = units + torch.trunc(prod * scale.unsqueeze(1)).to(torch.int64).sum(dim=1)
            # A non-finite accumulator passes through unchanged, like the oracle's isFinite guard.
            stepped = fixed_to_f16_t(units, max_exponent - 24)
            a = torch.where(torch.isfinite(a), stepped, a)
            acce = torch.where(a == 0, torch.full_like(e4m3_exponent_t(a), -21), f16_exponent_t(a))
        acc[lo:hi] = a
    return acc


# ------------------------------------------------------------------------------------------------------------
# Checks against the scalar oracle. check_numerics.py --torch runs these.
# ------------------------------------------------------------------------------------------------------------

def check_against_oracle(cases):
    failed = False
    device = DEVICE
    print(f'  torch device: {device}')

    for entry in cases:
        name = entry['name']
        if name.startswith('f16Bits'):
            inputs = entry.get('torch_inputs')
            if inputs is None:
                continue
        if 'over every half' in name and 'e4m3' not in name and 'vit' not in name:
            every = torch.arange(_HALF_BITS, dtype=torch.int32, device=device)
            x16 = f16_from_bits_t(every)
            if 'mpCubicSilu' in name:
                got = f16_bits_t(mp_cubic_silu_t(x16)).cpu().tolist()
            elif 'expWeight' in name:
                got = f16_bits_t(exp_weight_t(x16)).cpu().tolist()
            else:
                continue
            mismatches = 0
            first = None
            for i in range(_HALF_BITS):
                if (i & 0x7c00) == 0x7c00:
                    continue
                if got[i] != entry['expected'][i]:
                    mismatches += 1
                    if first is None:
                        first = (i, entry['expected'][i], got[i])
            status = 'ok' if mismatches == 0 else 'FAIL'
            failed |= mismatches != 0
            line = f"  {status:4} torch {name}"
            if first:
                line += f", first mismatch at {first[0]}: expected 0x{first[1]:04x} got 0x{first[2]:04x}"
            print(line)
        elif name.startswith('adaFp8Fdpa16') or name.startswith('adaF16Fdpa8'):
            operands = entry['operands']
            width = entry['width']
            b_count = len(operands)
            xs = torch.zeros((b_count, 1, width), dtype=torch.float16, device=device)
            ws = torch.zeros((b_count, width, 1), dtype=torch.float16, device=device)
            seeds = torch.zeros((b_count, 1, 1), dtype=torch.float16, device=device)
            for i, (a, b, acc) in enumerate(operands):
                xs[i, 0, :] = torch.tensor(a, dtype=torch.float16)
                ws[i, :, 0] = torch.tensor(b, dtype=torch.float16)
                seeds[i, 0, 0] = acc
            results = []
            if width == 16:
                results.append(('torch', fp8_dot_chain_batched(xs, ws, seeds)[:, 0, 0]))
            else:
                results.append(('torch', _f16_group_batched(xs, ws, seeds)[:, 0, 0]))
                if _cuda_backend is not None:
                    _, raw = _cuda_backend.f16_gemm(xs, ws, seed=seeds, quantize=False, raw=True)
                    results.append(('cuda', raw[:, 0, 0]))
                if _triton_backend is not None:
                    _, raw = _triton_backend.f16_gemm(xs, ws, seed=seeds, quantize=False, raw=True)
                    results.append(('triton', raw[:, 0, 0]))
            for backend_name, got_t in results:
                got_bits = f16_bits_t(got_t).cpu().tolist()
                mismatches = 0
                nan_pairs = 0
                first = None
                for i in range(b_count):
                    e, g = entry['expected'][i], got_bits[i]
                    if e == g:
                        continue
                    if (e & 0x7c00) == 0x7c00 and (e & 0x3ff) and (g & 0x7c00) == 0x7c00 and (g & 0x3ff):
                        nan_pairs += 1
                        continue
                    mismatches += 1
                    if first is None:
                        first = (i, e, g)
                status = 'ok' if mismatches == 0 else 'FAIL'
                failed |= mismatches != 0
                line = f'  {status:4} {backend_name} {name}: {b_count} cases'
                if nan_pairs:
                    line += f', {nan_pairs} NaN pairs'
                if first:
                    line += f", first mismatch at {first[0]}: expected 0x{first[1]:04x} got 0x{first[2]:04x}"
                print(line)

    # The ViT exponential has no fixture entry; the torch form is pinned to the scalar oracle here.
    every = torch.arange(_HALF_BITS, dtype=torch.int32, device=DEVICE)
    x16 = f16_from_bits_t(every)
    got = f16_bits_t(vit_exp_weight_t(x16)).cpu().tolist()
    bad = [i for i in range(_HALF_BITS)
           if (i & 0x7c00) != 0x7c00 and got[i] != oracle.f16_bits(oracle.vit_exp_weight(oracle.f16_to_number(i)))]
    print(f"  {'ok  ' if not bad else 'FAIL'} torch vitExpWeight over every half"
          + ('' if not bad else f', {len(bad)} mismatches, first at {bad[0]}'))
    failed |= bool(bad)

    failed |= check_chain_against_oracle()
    return failed


def _f16_group_batched(xs, ws, seeds):
    """One 8-product f16 group over batched [B, 1, 8] shapes - the FD16 fixture's shape."""
    prod = xs.unsqueeze(-1).to(torch.float32) * ws.unsqueeze(1).to(torch.float32)
    esum = f16_exponent_t(xs).unsqueeze(-1).to(torch.int32) + f16_exponent_t(ws).unsqueeze(1).to(torch.int32)
    covered = (xs != 0).unsqueeze(-1) & (ws != 0).unsqueeze(1)
    esum = esum.masked_fill(~covered, -21)
    max_exponent = esum.amax(dim=2)
    acce = torch.where(seeds == 0, torch.full_like(e4m3_exponent_t(seeds), -21), f16_exponent_t(seeds))
    max_exponent = torch.maximum(max_exponent, acce)
    scale = torch.exp2((24 - max_exponent).to(torch.float32))
    units = torch.trunc(seeds.to(torch.float32) * scale).to(torch.int64)
    units = units + torch.trunc(prod * scale.unsqueeze(2)).to(torch.int64).sum(dim=2)
    return fixed_to_f16_t(units, max_exponent - 24)


def _finite_half_bits(draw):
    """Non-finite operands take separately listed paths in the kernels, so the random cases stay finite."""
    bits = draw & 0xffff
    if (bits & 0x7c00) == 0x7c00:
        bits &= 0x7bff
    return bits


def check_chain_against_oracle():
    """Random small GEMMs through the chained kernels vs the scalar reference semantics."""
    import random
    rng = random.Random(20240611)
    device = DEVICE
    failed = False

    def scalar_fp8(x_rows, w_kn, seed_row, partition):
        # reference gemmFp8Element, element by element
        out = []
        for r in range(len(x_rows)):
            row = []
            for n in range(len(w_kn[0])):
                sums = oracle.round_f16(seed_row[r][n])
                part_sums = 0.0
                for kb in range(0, len(x_rows[r]), 32):
                    a = x_rows[r][kb:kb + 32]
                    b = [w_kn[k][n] for k in range(kb, kb + 32)]
                    sums = oracle.ada_fp8_fdpa16(a[:16], b[:16], 16, sums)
                    sums = oracle.ada_fp8_fdpa16(a[16:], b[16:], 16, sums)
                    if partition and ((kb + 32) % partition == 0 or kb + 32 >= len(x_rows[r])):
                        part_sums = sums if kb < partition else oracle.round_f16(part_sums + sums)
                        sums = 0.0
                row.append(part_sums if partition else sums)
            out.append(row)
        return out

    def chain_backends():
        backends = [('torch', _fp8_dot_chain_torch)]
        if _cuda_backend is not None:
            backends.append(('cuda', _cuda_backend.fp8_gemm))
        if _triton_backend is not None:
            backends.append(('triton', _triton_backend.fp8_gemm))
        return backends

    def draw_code():
        # Half the trials lean on the largest E4M3 magnitudes so some chains overflow to inf and exercise
        # the isFinite accumulator passthrough the reference has.
        if rng.random() < 0.5:
            return rng.choice([0x7e, 0xfe, 0x76, 0xf6, rng.randrange(256)])
        return rng.randrange(256)

    for trial in range(12):
        R = rng.randint(1, 5)
        N = rng.randint(1, 5)
        K = rng.choice([32, 64, 96, 128])
        partition = rng.choice([0, 32, 64]) if K in (64, 128) else 0
        if partition and K % partition:
            partition = 0
        codes_x = [[draw_code() for _ in range(K)] for _ in range(R)]
        codes_w = [[draw_code() for _ in range(N)] for _ in range(K)]
        seed = [[oracle.f16_to_number(_finite_half_bits(rng.getrandbits(16))) for _ in range(N)]
                for _ in range(R)]
        x_rows = [[oracle.e4m3_to_number(c) for c in row] for row in codes_x]
        w_kn = [[oracle.e4m3_to_number(codes_w[k][n]) for n in range(N)] for k in range(K)]

        expected = scalar_fp8(x_rows, w_kn, seed, partition)
        xt = torch.tensor(x_rows, dtype=torch.float16, device=device)
        wt = torch.tensor([[oracle.e4m3_to_number(codes_w[k][n]) for n in range(N)] for k in range(K)],
                          dtype=torch.float16, device=device)
        seedt = torch.tensor(seed, dtype=torch.float16, device=device)
        for name, backend in chain_backends():
            got = backend(xt, wt, seedt, partition=partition).cpu().tolist()
            for r in range(R):
                for n in range(N):
                    e, gval = expected[r][n], got[r][n]
                    if e != gval and not (e != e and gval != gval):
                        print(f'  FAIL {name} fp8 chain: trial {trial} R{R} N{N} K{K} P{partition} '
                              f'at ({r},{n}): expected {e} got {gval}')
                        failed = True

    # Batched chains (the window attention's shape family), against the scalar reference per batch.
    for trial in range(4):
        B = rng.randint(2, 5)
        R = rng.randint(1, 3)
        N = rng.randint(1, 3)
        K = rng.choice([32, 64])
        bx = [[[draw_code() for _ in range(K)] for _ in range(R)] for _ in range(B)]
        bw = [[[draw_code() for _ in range(N)] for _ in range(K)] for _ in range(B)]
        bseed = [[[oracle.f16_to_number(_finite_half_bits(rng.getrandbits(16))) for _ in range(N)]
                  for _ in range(R)] for _ in range(B)]
        xt = torch.tensor([[[oracle.e4m3_to_number(c) for c in row] for row in batch] for batch in bx],
                          dtype=torch.float16, device=device)
        wt = torch.tensor([[[oracle.e4m3_to_number(bw[b][k][n]) for n in range(N)] for k in range(K)]
                           for b in range(B)], dtype=torch.float16, device=device)
        seedt = torch.tensor(bseed, dtype=torch.float16, device=device)
        results = [('torch', _fp8_dot_chain_batched_torch(xt, wt, seedt))]
        if _cuda_backend is not None:
            results.append(('cuda', _cuda_backend.fp8_gemm(xt, wt, seedt)))
        if _triton_backend is not None:
            results.append(('triton', _triton_backend.fp8_gemm(xt, wt, seedt)))
        for name, got_t in results:
            got = got_t.cpu().tolist()
            for b in range(B):
                x_rows = [[oracle.e4m3_to_number(c) for c in row] for row in bx[b]]
                w_kn = [[oracle.e4m3_to_number(bw[b][k][n]) for n in range(N)] for k in range(K)]
                expected = scalar_fp8(x_rows, w_kn, bseed[b], 0)
                for r in range(R):
                    for n in range(N):
                        e, gval = expected[r][n], got[b][r][n]
                        if e != gval and not (e != e and gval != gval):
                            print(f'  FAIL {name} batched chain: trial {trial} B{B} R{R} N{N} K{K} '
                                  f'at batch {b} ({r},{n}): expected {e} got {gval}')
                            failed = True

    for trial in range(8):
        R = rng.randint(1, 5)
        N = rng.randint(1, 5)
        K = rng.choice([8, 16, 24, 32])
        codes_x = [[_finite_half_bits(rng.getrandbits(16)) for _ in range(K)] for _ in range(R)]
        codes_w = [[_finite_half_bits(rng.getrandbits(16)) for _ in range(N)] for _ in range(K)]
        x_rows = [[oracle.f16_to_number(c) for c in row] for row in codes_x]
        w_kn = [[oracle.f16_to_number(codes_w[k][n]) for n in range(N)] for k in range(K)]

        expected = []
        for r in range(R):
            row = []
            for n in range(N):
                value = 0.0
                for base in range(0, K, 8):
                    a = [oracle.round_f16(x_rows[r][base + i]) for i in range(8)]
                    b = [w_kn[base + i][n] for i in range(8)]
                    value = oracle.ada_f16_fdpa8(a, b, 8, value)
                row.append(value)
            expected.append(row)
        xt = torch.tensor(x_rows, dtype=torch.float16, device=device)
        wt = torch.tensor([[oracle.f16_to_number(codes_w[k][n]) for n in range(N)] for k in range(K)],
                          dtype=torch.float16, device=device)
        results = [('torch', f16_dot_chain(xt, wt))]
        if _cuda_backend is not None:
            _, raw = _cuda_backend.f16_gemm(xt, wt, quantize=False, raw=True)
            results.append(('cuda', raw))
        if _triton_backend is not None:
            _, raw = _triton_backend.f16_gemm(xt, wt, quantize=False, raw=True)
            results.append(('triton', raw))
        for name, got_t in results:
            got = got_t.cpu().tolist()
            for r in range(R):
                for n in range(N):
                    e, gval = expected[r][n], got[r][n]
                    if e != gval and not (e != e and gval != gval):
                        print(f'  FAIL {name} f16 chain: trial {trial} R{R} N{N} K{K} '
                              f'at ({r},{n}): expected {e} got {gval}')
                        failed = True

    print(f"  {'ok  ' if not failed else 'FAIL'} torch chained GEMMs vs scalar reference")
    return failed
