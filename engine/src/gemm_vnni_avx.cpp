#include "gemm_vnni.h"
#include <immintrin.h>
#include <cstring>

namespace {
inline __m256i broadcast_u8(const int8_t* p) {
    int32_t x; std::memcpy(&x, p, sizeof(x));
    // XOR the sign bit in every byte: u8 = s8 + 128.
    return _mm256_set1_epi32(x ^ static_cast<int32_t>(0x80808080u));
}
}

void vnni_dot_avx(const int8_t* A, const int8_t* panel,
                  const int32_t* correction, int Kp, int32_t* out) {
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(correction));
    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(correction + 8));
    __m256i c00 = lo, c01 = hi, c10 = lo, c11 = hi;
    __m256i c20 = lo, c21 = hi, c30 = lo, c31 = hi;
    for (int k = 0; k < Kp; k += 4, panel += 64) {
        const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(panel));
        const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(panel + 32));
        const __m256i a0 = broadcast_u8(A + k), a1 = broadcast_u8(A + Kp + k);
        const __m256i a2 = broadcast_u8(A + 2 * Kp + k), a3 = broadcast_u8(A + 3 * Kp + k);
        // Use the VEX intrinsic, not the EVEX/AVX-512VL intrinsic with a similar name.
        c00 = _mm256_dpbusd_avx_epi32(c00, a0, w0); c01 = _mm256_dpbusd_avx_epi32(c01, a0, w1);
        c10 = _mm256_dpbusd_avx_epi32(c10, a1, w0); c11 = _mm256_dpbusd_avx_epi32(c11, a1, w1);
        c20 = _mm256_dpbusd_avx_epi32(c20, a2, w0); c21 = _mm256_dpbusd_avx_epi32(c21, a2, w1);
        c30 = _mm256_dpbusd_avx_epi32(c30, a3, w0); c31 = _mm256_dpbusd_avx_epi32(c31, a3, w1);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), c00);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 8), c01);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 16), c10);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 24), c11);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 32), c20);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 40), c21);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 48), c30);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 56), c31);
}
