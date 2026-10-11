#include "gemm_vnni.h"
#include <immintrin.h>
#include <cstring>

namespace {
inline __m512i broadcast_u8(const int8_t* p) {
    int32_t x; std::memcpy(&x, p, sizeof(x));
    return _mm512_set1_epi32(x ^ static_cast<int32_t>(0x80808080u));
}
}

void vnni_dot_avx512(const int8_t* A, const int8_t* panel,
                     const int32_t* correction, int Kp, int32_t* out) {
    const __m512i bias = _mm512_loadu_si512(correction);
    __m512i c0 = bias, c1 = bias, c2 = bias, c3 = bias;
    for (int k = 0; k < Kp; k += 4, panel += 64) {
        const __m512i w = _mm512_loadu_si512(panel);
        c0 = _mm512_dpbusd_epi32(c0, broadcast_u8(A + k), w);
        c1 = _mm512_dpbusd_epi32(c1, broadcast_u8(A + Kp + k), w);
        c2 = _mm512_dpbusd_epi32(c2, broadcast_u8(A + 2 * Kp + k), w);
        c3 = _mm512_dpbusd_epi32(c3, broadcast_u8(A + 3 * Kp + k), w);
    }
    _mm512_storeu_si512(out, c0); _mm512_storeu_si512(out + 16, c1);
    _mm512_storeu_si512(out + 32, c2); _mm512_storeu_si512(out + 48, c3);
}
