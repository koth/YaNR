// Baseline ISA only. Specialized instructions are confined to the dot kernels.
#include "gemm.h"
#include "gemm_vnni.h"
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace {
struct Features { bool avx2 = false, avx_vnni = false, avx512_vnni = false; };

void cpuid(unsigned leaf, unsigned sub, unsigned out[4]) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int v[4]; __cpuidex(v, leaf, sub);
    for (int i = 0; i < 4; ++i) out[i] = static_cast<unsigned>(v[i]);
#elif defined(__x86_64__) || defined(__i386__)
    __cpuid_count(leaf, sub, out[0], out[1], out[2], out[3]);
#else
    for (int i = 0; i < 4; ++i) out[i] = 0;
#endif
}

uint64_t xcr0() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return _xgetbv(0);
#elif defined(__x86_64__) || defined(__i386__)
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#else
    return 0;
#endif
}

const Features& features() {
    static const Features f = [] {
        Features v;
        unsigned c[4]; cpuid(0, 0, c);
        if (c[0] < 7) return v;
        cpuid(1, 0, c);
        // OSXSAVE + AVX + FMA, with XMM/YMM state enabled by the OS.
        constexpr unsigned avx = (1u << 27) | (1u << 28) | (1u << 12);
        if ((c[2] & avx) != avx) return v;
        const uint64_t state = xcr0();
        if ((state & 6) != 6) return v;
        cpuid(7, 0, c);
        const unsigned max_sub = c[0];
        v.avx2 = (c[1] & (1u << 5)) != 0;
        // Match the complete /arch:AVX512 compiler ISA, plus VNNI. XCR0
        // additionally needs opmask and both ZMM state components.
        constexpr unsigned avx512 = (1u << 16) | (1u << 17) | (1u << 28) |
                                     (1u << 30) | (1u << 31);
        v.avx512_vnni = v.avx2 && (c[1] & avx512) == avx512 &&
                         (c[2] & (1u << 11)) && (state & 0xe6) == 0xe6;
        if (max_sub >= 1) {
            cpuid(7, 1, c);
            v.avx_vnni = v.avx2 && (c[0] & (1u << 4));
        }
        return v;
    }();
    return f;
}
}

VnniDot vnni_kernel(VnniIsa isa) {
#if defined(NR_BUILD_AVX_VNNI)
    if (isa == VnniIsa::avx && features().avx_vnni) return vnni_dot_avx;
#endif
#if defined(NR_BUILD_AVX512_VNNI)
    if (isa == VnniIsa::avx512 && features().avx512_vnni) return vnni_dot_avx512;
#endif
    (void)isa;
    return nullptr;
}

VnniIsa selected_vnni_isa() {
    const char* value = std::getenv("NR_VNNI");
    if (value && std::strcmp(value, "0") == 0) return VnniIsa::none;
    if (value && std::strcmp(value, "avx") == 0)
        return vnni_kernel(VnniIsa::avx) ? VnniIsa::avx : VnniIsa::none;
    if (value && std::strcmp(value, "avx512") == 0)
        return vnni_kernel(VnniIsa::avx512) ? VnniIsa::avx512 : VnniIsa::none;
    if (vnni_kernel(VnniIsa::avx512)) return VnniIsa::avx512;
    if (vnni_kernel(VnniIsa::avx)) return VnniIsa::avx;
    return VnniIsa::none;
}

bool gemm_vnni_available() { return selected_vnni_isa() != VnniIsa::none; }
const char* gemm_i8_backend() {
    switch (selected_vnni_isa()) {
        case VnniIsa::avx: return "avx-vnni";
        case VnniIsa::avx512: return "avx512-vnni";
        default: break;
    }
#if defined(NR_BUILD_AVX2)
    if (features().avx2) return "avx2";
#endif
    return "scalar";
}
