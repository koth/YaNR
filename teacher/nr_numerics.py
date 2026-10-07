# The network's publication grid, in Python. A line-for-line port of ports/browser-webgpu/src/numerics.js,
# which mirrors src/numeric.h and src/reference.cpp of the Vulkan implementation. See docs/numerics.md for
# why the schedule of roundings, and not an error bound, is the contract.
#
# This module is the port's oracle: check_numerics.py pins it against web/fixtures/numerics.bin (recorded from
# the C++ reference) and the torch kernels are pinned against it in turn. It is written for legibility rather
# than speed, and it never takes a shortcut the shaders cannot take. All arithmetic happens in Python floats
# (IEEE doubles, like JavaScript) with explicit f32 roundings where the WGSL has f32 operations.

import math
import struct

_F32 = struct.Struct('<f')
_U32 = struct.Struct('<I')


def f32_bits(value):
    """f32 bit pattern of a number (double -> f32, round-to-nearest-even)."""
    try:
        return _U32.unpack(_F32.pack(value))[0]
    except OverflowError:
        return 0x7f800000 if value > 0 else 0xff800000


def f32_from_bits(bits):
    """Number from an f32 bit pattern."""
    return _F32.unpack(_U32.pack(bits & 0xffffffff))[0]


def to_f32(value):
    """Round an arbitrary double down to what an f32 would hold, which is where the shader side starts."""
    return f32_from_bits(f32_bits(value))


def round_shift_right_even(value, shift):
    """Shift right with round-to-nearest-even, the rounding every narrowing conversion here uses."""
    value = int(value) & 0xffffffff
    if shift <= 0:
        return value
    if shift > 31:
        return 0
    quotient = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    return quotient + (1 if (remainder > halfway or (remainder == halfway and (quotient & 1))) else 0)


def f16_bits(value):
    """IEEE binary16 bit pattern, round-to-nearest-even (via f32, as the WGSL f16() conversion does)."""
    bits = f32_bits(value)
    sign = (bits >> 16) & 0x8000
    exponent = (bits >> 23) & 0xff
    mantissa = bits & 0x7fffff
    if exponent == 0xff:
        return sign | (0x7e00 if mantissa else 0x7c00)
    half_exponent = exponent - 112
    if half_exponent >= 31:
        return sign | 0x7c00
    if half_exponent <= 0:
        if half_exponent < -10:
            return sign
        return sign | round_shift_right_even(mantissa | 0x800000, 14 - half_exponent)
    rounded = round_shift_right_even(mantissa, 13)
    if rounded == 0x400:
        rounded = 0
        half_exponent += 1
    if half_exponent >= 31:
        return sign | 0x7c00
    return sign | (half_exponent << 10) | rounded


def f16_to_number(bits):
    """Number from an IEEE binary16 bit pattern."""
    bits = int(bits) & 0xffff
    sign = -1.0 if (bits & 0x8000) else 1.0
    exponent = (bits >> 10) & 0x1f
    mantissa = bits & 0x3ff
    if exponent == 0:
        return sign * mantissa * 2.0 ** -24
    if exponent == 0x1f:
        return float('nan') if mantissa else sign * float('inf')
    return sign * (1.0 + mantissa / 1024.0) * 2.0 ** (exponent - 15)


def round_f16(value):
    """One publication to the half grid."""
    return f16_to_number(f16_bits(value))


def e4m3_from_f16_bits(half):
    """
    E4M3FN code of an already half-rounded value: RNE, finite saturation at 448, NaN -> +0. The sign of a
    zero survives, as the hardware conversion leaves it.
    """
    half = int(half) & 0xffff
    if (half & 0x7c00) == 0x7c00 and (half & 0x03ff) != 0:
        return 0
    if (half & 0x7fff) == 0:
        return (half >> 8) & 0x80
    negative = 0x80 if (half & 0x8000) else 0
    exponent = (half >> 10) & 0x1f
    mantissa = half & 0x3ff
    if exponent == 31:
        code = 0x7e
    elif exponent <= 8:
        # Below 2^-6 the E4M3 grid is subnormal: the half significand shifts down onto a fixed step of 2^-9.
        significand = mantissa if exponent == 0 else 1024 + mantissa
        code = min(round_shift_right_even(significand, 15 if exponent == 0 else 16 - exponent), 8)
    else:
        e4_exponent = exponent - 8
        e4_mantissa = round_shift_right_even(mantissa, 7)
        if e4_mantissa == 8:
            e4_mantissa = 0
            e4_exponent += 1
        code = 0x7e if (e4_exponent > 15 or (e4_exponent == 15 and e4_mantissa > 6)) \
               else (e4_exponent << 3) | e4_mantissa
    return negative | code


def e4m3_from_number(value):
    return e4m3_from_f16_bits(f16_bits(value))


def e4m3_to_number(byte):
    """
    E4M3FN value of a byte. The one NaN code reads as a signed zero, which is what the shaders do with it.
    """
    byte = int(byte) & 0xff
    sign = -1.0 if (byte & 0x80) else 1.0
    exponent = (byte >> 3) & 0xf
    mantissa = byte & 0x7
    if exponent == 0:
        return sign * mantissa * 2.0 ** -9
    if exponent == 0xf and mantissa == 0x7:
        return sign * 0.0
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def normal_exponent(value):
    return ((f32_bits(abs(value)) >> 23) & 0xff) - 127


def e4m3_exponent(value):
    return max(normal_exponent(value), -6)


def f16_exponent(value):
    return max(normal_exponent(value), -14)


# The tensor core's dot product, as fixed point: the terms are accumulated as exact integers in units of
# 2^(max - 13) so that no difference in summation order or fused multiply-add can drift the result apart.

def f13_start(accumulator):
    """The shared exponent of a step that starts from this accumulator; -21 is the empty value."""
    return f16_exponent(accumulator) if accumulator != 0 else -21


def f13_cover(maximum_exponent, a, b):
    """Widen the shared exponent to cover one product; a zero operand is skipped, not clamped."""
    if a == 0 or b == 0:
        return maximum_exponent
    return max(maximum_exponent, e4m3_exponent(a) + e4m3_exponent(b))


def f13_term(value, scale):
    """One term of the aligned sum, in units of 2^(max - 13)."""
    return math.trunc(to_f32(value * scale))


def f13_finish(units, maximum_exponent):
    """The aligned sum, published once to the half grid."""
    return round_f16(units * 2.0 ** (maximum_exponent - 13))


def round_shift_magnitude(magnitude, shift):
    return round_shift_right_even(magnitude, shift) if shift >= 0 else (magnitude << -shift)


def fixed_to_f16(fixed_sum, binary_exponent):
    """Exact signed integer times 2^binaryExponent -> half."""
    if fixed_sum == 0:
        return 0.0
    negative = fixed_sum < 0
    magnitude = -fixed_sum if negative else fixed_sum
    msb = magnitude.bit_length() - 1
    value_exponent = msb + binary_exponent
    half_bits = 0x8000 if negative else 0
    if value_exponent >= -14:
        significand = round_shift_right_even(magnitude, msb - 10) if msb > 10 else magnitude << (10 - msb)
        if significand >= 2048:
            significand = 1024
            value_exponent += 1
        if value_exponent >= 16:
            half_bits |= 0x7c00
        else:
            half_bits |= ((value_exponent + 15) << 10) | (significand - 1024)
    else:
        subnormal_scale = binary_exponent + 24
        mantissa = round_shift_magnitude(magnitude, -subnormal_scale)
        half_bits |= min(mantissa, 1024)
    return f16_to_number(half_bits)


def ada_fp8_fdpa16(a, b, count, accumulator):
    """One half of a k32 FP8 step: 16 products plus the incoming accumulator, aligned, truncated, summed."""
    if not math.isfinite(accumulator):
        return accumulator
    maximum_exponent = f13_start(accumulator)
    for i in range(count):
        maximum_exponent = f13_cover(maximum_exponent, a[i], b[i])
    scale = 2.0 ** (13 - maximum_exponent)
    units = f13_term(accumulator, scale)
    for i in range(count):
        units += f13_term(to_f32(a[i] * b[i]), scale)
    return f13_finish(units, maximum_exponent)


def ada_f16_fdpa8(a, b, count, accumulator):
    """The f16 step: 8 products against 24 fractional bits, for the chains that never narrow to E4M3."""
    if not math.isfinite(accumulator):
        return accumulator
    maximum_exponent = f13_start(accumulator)
    for i in range(count):
        if a[i] != 0 and b[i] != 0:
            maximum_exponent = max(maximum_exponent, f16_exponent(a[i]) + f16_exponent(b[i]))
    scale = 2.0 ** (24 - maximum_exponent)
    units = math.trunc(to_f32(accumulator * scale))
    for i in range(count):
        units += math.trunc(to_f32(to_f32(a[i] * b[i]) * scale))
    return fixed_to_f16(units, maximum_exponent - 24)


# The three functions below each round in a particular precision. Python's doubles are wider than all of
# them, so every f32-level step is put through to_f32. Where native fuses a multiply and an add, the product
# happens to be exact in f32, so one rounding over the whole expression is that fused operation.


def mp_cubic_silu(value):
    """MpCubicSiLU: five half publications, the two inner steps f32 fused multiply-adds with exact products."""
    bounded = round_f16(min(max(value, -4.0), 4.0))
    absolute = round_f16(abs(bounded))
    inner = round_f16(to_f32(-0.055908203125 * absolute + 0.447265625))
    polynomial = round_f16(to_f32(bounded * inner + 0.89453125))
    return round_f16(to_f32(value * polynomial))


def exp_weight(score):
    """The window blocks' attention weight: an affine map dropped into the half exponent field."""
    affine = round_f16(to_f32(score * 0.044921875 + 1.30078125))
    clamped = min(max(affine, 1.03125), 1.5693359375)
    return f16_to_number((((f16_bits(clamped) << 5) + 0x8000) & 0xffff))


def vit_exp_weight(score):
    """The ViT's variant: a 4-bit shift instead of 5, a different bias, and an affine evaluated in halves."""
    affine = round_f16(score * 0.08953857421875 + 1.708984375)
    clamped = min(max(affine, 1.439453125), 1.9775390625)
    return f16_to_number((((f16_bits(clamped) << 4) + 0x4000) & 0xffff))


def fp8_domain(value):
    """The E4M3 publication as a value: quantize to the fp8 grid (NaN -> +0, zeros keep their sign)."""
    return e4m3_to_number(e4m3_from_f16_bits(f16_bits(value)))


def silu_table():
    """The SiLU lookup table the fused kernels index by half bit pattern, built once."""
    return [f16_bits(mp_cubic_silu(f16_to_number(bits))) for bits in range(65536)]


# ------------------------------------------------------------------------------------------------------------
# The fixed-point schedules, for reference and for tests:
#
#   adaFp8Fdpa16: k32 steps = two 16-product groups, shared exponent, truncate to 13 fractional bits (F13),
#                 exact integer sum in i32, round to half once per group.
#   adaF16Fdpa8:  8 products, F24 truncation, exact integer -> half (fixedToF16).
# ------------------------------------------------------------------------------------------------------------
