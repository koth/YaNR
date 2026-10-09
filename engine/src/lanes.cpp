#include "lanes.h"

#include <cmath>
#include <cstring>

#include "f16.h"

namespace {

inline uint32_t u32_mul(uint32_t a, uint32_t b) { return a * b; }   // natural wrap

double hash_uniform(uint32_t value) {
    uint32_t mixed = value;
    mixed = (mixed >> ((mixed >> 28) + 4)) ^ mixed;
    mixed = u32_mul(mixed, 0x108ef2d9u);
    uint32_t integer = ((mixed >> 30) ^ (mixed >> 8)) + 1u;
    return (double)integer * 5.960464477539063e-08;                 // 2^-24
}

void gaussian3(uint32_t x, uint32_t y, uint32_t seed, double out[3]) {
    uint32_t base = u32_mul(x, 0x8da6b343u) ^ u32_mul(y, 0xd8163841u);
    base ^= seed;                                                   // (seed*0x9e3779b9) low 32 == seed mix below
    base ^= 0x243f6a88u;
    base = (base >> ((base >> 28) + 4)) ^ base;
    base = u32_mul(base, 0x108ef2d9u);
    base = (base >> 22) ^ base;
    double u0 = hash_uniform(u32_mul(base, 0x2c9277b5u) + 0xac564b05u);
    double u1 = hash_uniform(u32_mul(base, 0xfa6dc5f9u) + 0x4712a88e);
    double u2 = hash_uniform(u32_mul(base, 0xcaa5b80du) + 0x21dd796bu);
    double u3 = hash_uniform(u32_mul(base, 0x83232c31u) + 0x3463e0acu);
    const double kLn2 = 0.6931471805599453;
    const double kTau = 6.283185307179586;
    double radius0 = std::sqrt(std::log2(u0) * kLn2 * -2.0);
    double radius1 = std::sqrt(std::log2(u2) * kLn2 * -2.0);
    out[0] = round_f16((float)(radius0 * std::cos(u1 * kTau)));
    out[1] = round_f16((float)(radius0 * std::sin(u1 * kTau)));
    out[2] = round_f16((float)(radius1 * std::cos(u3 * kTau)));
}

uint32_t seed_mix(uint64_t seed) {
    // numpy: uint64(seed) * uint64(0x9e3779b9) -> take low 32 bits
    return (uint32_t)((seed * 0x9e3779b9ull) & 0xffffffffull);
}

inline float center_proxy(float code) {
    float a = f16_to_f32(f32_to_f16(code));                 // roundF16(c)
    float b = f16_to_f32(f32_to_f16(a - 0.5f));             // roundF16(that - 0.5)
    return f16_to_f32(f32_to_f16(b * 0.125f));              // roundF16(that * 0.125)
}

}  // namespace

void build_lanes(const float* proxy, int vw, int vh, const Geometry& g,
                 const LaneParams& p, float* out) {
    const int full_w = g.full_width, full_h = g.full_height;
    const uint32_t seed = seed_mix((uint64_t)p.seed);
    const float style_lane = (float)(p.style / 128.0);
    const float tone_lane = f16_to_f32(f32_to_f16((float)p.tone));
    float s12, s13, s14;
    if (p.auto_mask) {
        s12 = f16_to_f32(f32_to_f16(1.0f));
        s13 = f16_to_f32(f32_to_f16((float)(p.skin < 0.0 ? p.structure : p.skin)));
        s14 = f16_to_f32(f32_to_f16((float)p.structure));
    } else {
        s12 = f16_to_f32(f32_to_f16((float)p.structure));
        s13 = f16_to_f32(f32_to_f16(-1.0f));
        s14 = f16_to_f32(f32_to_f16(-1.0f));
    }

    // 每像素独立、无跨像素归约 —— 按行并行且逐位不变(check_lanes 对账不受影响)。
#pragma omp parallel for schedule(static)
    for (int y = 0; y < full_h; y++) {
        int ry = y < vh ? y : 2 * vh - y - 2;
        if (ry < 0) ry = 0;
        if (ry > vh - 1) ry = vh - 1;
        for (int x = 0; x < full_w; x++) {
            int rx = x < vw ? x : 2 * vw - x - 2;
            if (rx < 0) rx = 0;
            if (rx > vw - 1) rx = vw - 1;
            const float* px = proxy + ((size_t)ry * vw + rx) * 3;
            float* f = out + ((size_t)y * full_w + x) * 16;

            double noise[3];
            gaussian3((uint32_t)x, (uint32_t)y, seed, noise);
            f[0] = (float)noise[0];
            f[1] = (float)noise[1];
            f[2] = (float)noise[2];
            f[3] = 1.0f;
            for (int c = 0; c < 3; c++) {
                float centered = center_proxy(px[c]);
                f[4 + c] = centered;
                f[7 + c] = p.history ? center_proxy(p.history[((size_t)ry * vw + rx) * 3 + c])
                                     : centered;
            }
            f[10] = style_lane;
            f[11] = tone_lane;
            f[12] = s12;
            f[13] = s13;
            f[14] = s14;
            f[15] = 0.0f;
        }
    }
}
