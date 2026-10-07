// hash_probe: print the gaussian3/hash_uniform intermediates for one coordinate.
// Debug tool for lane parity (task 8.3): compares against run_image.gaussian3.
//
//   hash_probe --x 324 --y 0 --seed 12345
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <string>

#include "f16.h"

namespace {

uint32_t u32_mul(uint32_t a, uint32_t b) { return a * b; }

double hash_uniform(uint32_t value) {
    uint32_t mixed = value;
    mixed = (mixed >> ((mixed >> 28) + 4)) ^ mixed;
    mixed = u32_mul(mixed, 0x108ef2d9u);
    uint32_t integer = ((mixed >> 30) ^ (mixed >> 8)) + 1u;
    return (double)integer * 5.960464477539063e-08;
}

}  // namespace

int main(int argc, char** argv) {
    uint32_t x = 0, y = 0, seed_in = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return argv[++i]; };
        if (a == "--x") x = (uint32_t)std::strtoul(next(), nullptr, 10);
        else if (a == "--y") y = (uint32_t)std::strtoul(next(), nullptr, 10);
        else if (a == "--seed") seed_in = (uint32_t)std::strtoul(next(), nullptr, 10);
    }
    uint32_t seed = (uint32_t)(((uint64_t)seed_in * 0x9e3779b9ull) & 0xffffffffull);

    uint32_t base = u32_mul(x, 0x8da6b343u) ^ u32_mul(y, 0xd8163841u);
    base ^= seed;
    base ^= 0x243f6a88u;
    base = (base >> ((base >> 28) + 4)) ^ base;
    base = u32_mul(base, 0x108ef2d9u);
    base = (base >> 22) ^ base;
    double u0 = hash_uniform(u32_mul(base, 0x2c9277b5u) + 0xac564b05u);
    double u1 = hash_uniform(u32_mul(base, 0xfa6dc5f9u) + 0x4712a88eu);
    double u2 = hash_uniform(u32_mul(base, 0xcaa5b80du) + 0x21dd796bu);
    double u3 = hash_uniform(u32_mul(base, 0x83232c31u) + 0x3463e0acu);
    const double kLn2 = 0.6931471805599453;
    const double kTau = 6.283185307179586;
    double radius0 = std::sqrt(std::log2(u0) * kLn2 * -2.0);
    double radius1 = std::sqrt(std::log2(u2) * kLn2 * -2.0);
    double n0r = radius0 * std::cos(u1 * kTau);
    double n1r = radius0 * std::sin(u1 * kTau);
    double n2r = radius1 * std::cos(u3 * kTau);

    std::printf("base   0x%08x\n", base);
    std::printf("u0 %.17g\nu1 %.17g\nu2 %.17g\nu3 %.17g\n", u0, u1, u2, u3);
    std::printf("radius0 %.17g  radius1 %.17g\n", radius0, radius1);
    std::printf("raw n0 %.17g\nraw n1 %.17g\nraw n2 %.17g\n", n0r, n1r, n2r);
    std::printf("f16 n0 %.9g\nf16 n1 %.9g\nf16 n2 %.9g\n",
                round_f16((float)n0r), round_f16((float)n1r), round_f16((float)n2r));
    std::printf("direct-double->f16 n1 %.9g\n", f16_to_f32(f32_to_f16((float)n1r)));
    float v = (float)n1r;
    uint32_t vb;
    std::memcpy(&vb, &v, 4);
    std::printf("n1 f32 bits 0x%08x  f16 bits 0x%04x\n", vb, (unsigned)f32_to_f16(v));
    {
        uint32_t xx = vb;
        uint32_t field = (xx >> 23) & 0xffu;
        uint32_t mant = xx & 0x7fffffu;
        int32_t e = (int32_t)field - 112;
        uint32_t bias = 0x0fffu + ((mant >> 13) & 1u);
        std::printf("  field %u  mant %u (0x%x)  exp %d  bias %u  mant+bias %u  >>13 %u\n",
                    field, mant, mant, e, bias, mant + bias, (mant + bias) >> 13);
        std::printf("  OR test: %04x\n", (unsigned)((0u | ((uint32_t)e << 10) | ((mant + bias) >> 13))));
    }
    std::printf("sanity 1.0->%04x  1.5->%04x  2.0->%04x  1.9998159->%04x\n",
                (unsigned)f32_to_f16(1.0f), (unsigned)f32_to_f16(1.5f),
                (unsigned)f32_to_f16(2.0f), (unsigned)f32_to_f16(1.9998159408569336f));
    return 0;
}
