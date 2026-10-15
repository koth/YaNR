// f16 conversions (round-to-nearest-even), the numeric base of the lane pipeline.
// The teacher's numerics round through half everywhere (center_proxy, gaussian3's r16),
// so the C++ side needs the same RNE behaviour as numpy/PyTorch float16.
#pragma once
#include <cstdint>
#include <cstring>

inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t field = (x >> 23) & 0xffu;
    uint32_t mant = x & 0x7fffffu;
    if (field == 0xffu) {                       // inf / nan -> half inf / qnan
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    }
    int32_t exp = (int32_t)field - 112;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);          // overflow -> inf
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;                  // underflow -> 0
        mant |= 0x800000u;                                     // subnormal half
        uint32_t shift = (uint32_t)(14 - exp);                 // 14..24
        uint32_t bias = (1u << (shift - 1)) - 1u + ((mant >> shift) & 1u);
        return (uint16_t)(sign | ((mant + bias) >> shift));    // carry promotes to normal
    }
    uint32_t bias = 0x0fffu + ((mant >> 13) & 1u);             // RNE on 23->10 bits
    // 尾数进位用 + 而不是 |:进位位 0x400 必须真的加进指数域(exp<<10 的 bit10 可能
    // 已是 1,OR 会把进位静默吞掉 -> 1.9998 会错舍入成 1.0)。
    return (uint16_t)(sign | (((uint32_t)exp << 10) + ((mant + bias) >> 13)));
}

// numpy converts its f64 Box-Muller values and CLI controls directly to half.
// Going through float first can round a value onto a half midpoint and then
// choose the wrong even neighbour. MoE routing can amplify that one-half error.
inline uint16_t f64_to_f16(double f) {
    uint64_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint16_t sign = static_cast<uint16_t>((x >> 48) & 0x8000u);
    const uint64_t field = (x >> 52) & 0x7ffu;
    uint64_t mant = x & 0xfffffffffffffull;
    if (field == 0x7ffu)
        return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    const int exp = static_cast<int>(field) - 1008;
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant |= uint64_t(1) << 52;
        const int shift = 43 - exp;
        const uint64_t bias = (uint64_t(1) << (shift - 1)) - 1 + ((mant >> shift) & 1u);
        return static_cast<uint16_t>(sign | ((mant + bias) >> shift));
    }
    const uint64_t bias = (uint64_t(1) << 41) - 1 + ((mant >> 42) & 1u);
    return static_cast<uint16_t>(sign | ((static_cast<uint64_t>(exp) << 10) + ((mant + bias) >> 42)));
}

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {                                               // subnormal half -> normal float
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            out = sign | (exp << 23) | ((mant & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

// numpy: code.astype(np.float16) -> back to float32
inline float round_f16(float v) { return f16_to_f32(f32_to_f16(v)); }
inline float round_f16(double v) { return f16_to_f32(f64_to_f16(v)); }
