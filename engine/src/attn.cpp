#include "attn.h"

#include <chrono>
#include <cmath>
#include <vector>

#include "gemm.h"   // exp_vec

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define NR_X86 1
#include <immintrin.h>
#endif

namespace {

// 核心内部微剖析(粗:多线程直接累加,只看占比):norms / scores / softmax / av。
double g_aprof[4] = {0, 0, 0, 0};
bool g_aprof_on = false;

#if defined(NR_X86) && defined(__AVX2__)

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

inline float cosine_norm32(float* x) {
    __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8);
    __m256 v2 = _mm256_loadu_ps(x + 16), v3 = _mm256_loadu_ps(x + 24);
    __m256 s = _mm256_mul_ps(v0, v0);
    s = _mm256_fmadd_ps(v1, v1, s);
    s = _mm256_fmadd_ps(v2, v2, s);
    s = _mm256_fmadd_ps(v3, v3, s);
    float inv = 1.0f / std::fmax(std::sqrt(hsum256(s)), 1e-6f);
    __m256 iv = _mm256_set1_ps(inv);
    _mm256_storeu_ps(x, _mm256_mul_ps(v0, iv));
    _mm256_storeu_ps(x + 8, _mm256_mul_ps(v1, iv));
    _mm256_storeu_ps(x + 16, _mm256_mul_ps(v2, iv));
    _mm256_storeu_ps(x + 24, _mm256_mul_ps(v3, iv));
    return inv;
}

inline void scale32(float* x, float s) {
    __m256 sv = _mm256_set1_ps(s);
    for (int d = 0; d < 32; d += 8) {
        _mm256_storeu_ps(x + d, _mm256_mul_ps(_mm256_loadu_ps(x + d), sv));
    }
}

inline float dot32(const float* a, const float* b) {
    // 双链累加:4-FMA 单链是延迟受限(每 dot ~16 周期),两条 2-FMA 链并行减半。
    __m256 s0 = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8), s0);
    __m256 s1 = _mm256_mul_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24), s1);
    return hsum256(_mm256_add_ps(s0, s1));
}

// 8 通道部分和(双链):4 个点积攒一起再 hsum。
inline __m256 dot32_partial(const float* a, const float* b) {
    __m256 s0 = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8), s0);
    __m256 s1 = _mm256_mul_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24), s1);
    return _mm256_add_ps(s0, s1);
}

// out[32] = Σ_j sr[j]*inv * vrows[j](vrows[j] == nullptr 的槽贡献 0,跳过)。
inline void av32(const float* sr, int slots, float inv, const float* const* vrows,
                 float* out) {
    // j 奇偶双链:acc 链深 64 -> 32(av 相位也是延迟受限)。
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
    __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
    for (int j = 0; j + 1 < slots; j += 2) {
        float a = sr[j] * inv;
        const float* vr = vrows[j];
        if (vr && a != 0.0f) {
            __m256 av = _mm256_set1_ps(a);
            a0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(vr), a0);
            a1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(vr + 8), a1);
            a2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(vr + 16), a2);
            a3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(vr + 24), a3);
        }
        a = sr[j + 1] * inv;
        vr = vrows[j + 1];
        if (vr && a != 0.0f) {
            __m256 av = _mm256_set1_ps(a);
            b0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(vr), b0);
            b1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(vr + 8), b1);
            b2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(vr + 16), b2);
            b3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(vr + 24), b3);
        }
    }
    if (slots & 1) {
        float a = sr[slots - 1] * inv;
        const float* vr = vrows[slots - 1];
        if (vr && a != 0.0f) {
            __m256 av = _mm256_set1_ps(a);
            a0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(vr), a0);
            a1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(vr + 8), a1);
            a2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(vr + 16), a2);
            a3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(vr + 24), a3);
        }
    }
    a0 = _mm256_add_ps(a0, b0); a1 = _mm256_add_ps(a1, b1);
    a2 = _mm256_add_ps(a2, b2); a3 = _mm256_add_ps(a3, b3);
    _mm256_storeu_ps(out, a0);
    _mm256_storeu_ps(out + 8, a1);
    _mm256_storeu_ps(out + 16, a2);
    _mm256_storeu_ps(out + 24, a3);
}

#else  // 标量回落(非 x86 / 未开 AVX2)

inline float cosine_norm32(float* x) {
    float s = 0.0f;
    for (int d = 0; d < 32; d++) s += x[d] * x[d];
    float inv = 1.0f / std::fmax(std::sqrt(s), 1e-6f);
    for (int d = 0; d < 32; d++) x[d] *= inv;
    return inv;
}

inline void scale32(float* x, float s) {
    for (int d = 0; d < 32; d++) x[d] *= s;
}

inline float dot32(const float* a, const float* b) {
    float acc = 0.0f;
    for (int d = 0; d < 32; d++) acc += a[d] * b[d];
    return acc;
}

inline void av32(const float* sr, int slots, float inv, const float* const* vrows,
                 float* out) {
    for (int d = 0; d < 32; d++) out[d] = 0.0f;
    for (int j = 0; j < slots; j++) {
        float a = sr[j] * inv;
        const float* vr = vrows[j];
        if (!vr || a == 0.0f) continue;
        for (int d = 0; d < 32; d++) out[d] += a * vr[d];
    }
}

#endif

}  // namespace

const double* attn_profile() { return g_aprof; }
void attn_profile_reset() { for (double& v : g_aprof) v = 0.0; g_aprof_on = true; }

void attn_window_head(float* qkv, int row_stride, const int* index,
                      const unsigned char* valid, int slots, float scale,
                      const float* prior, float* scores, float* out, int out_stride) {
    std::chrono::steady_clock::time_point t0, t1, t2, t3;
    if (g_aprof_on) t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < slots; s++) {
        if (!valid[s]) continue;
        float* qr = qkv + (size_t)s * row_stride;
        cosine_norm32(qr);
        scale32(qr, scale);
        cosine_norm32(qr + 32);
    }
    if (g_aprof_on) {
        t1 = std::chrono::steady_clock::now();
        g_aprof[0] += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    // 打分:先把 k 转置成 kT[32][slots](非法键列清零 = k 为 0 的掩码语义),
    // 双查询 x 4 带累加向量化 —— 每 MAC 的加载口压力从 ~2 降到 ~0.25(scores
    // 相位原是加载墙:dot32 每点积 8 次 ymm 加载,只产 8 MAC)。
#if defined(NR_X86) && defined(__AVX2__)
    {
        static thread_local std::vector<float> kT_buf;
        kT_buf.resize((size_t)32 * slots);
        float* kT = kT_buf.data();
        for (int j = 0; j < slots; j++) {
            const float* kr = valid[j] ? qkv + (size_t)j * row_stride + 32 : nullptr;
            for (int d = 0; d < 32; d++) kT[(size_t)d * slots + j] = kr ? kr[d] : 0.0f;
        }
        for (int s = 0; s + 1 < slots; s += 2) {
            const float* q0 = qkv + (size_t)s * row_stride;
            const float* q1 = qkv + (size_t)(s + 1) * row_stride;
            float* r0 = scores + (size_t)s * slots;
            float* r1 = scores + (size_t)(s + 1) * slots;
            const float* pr0 = prior + (size_t)s * slots;
            const float* pr1 = prior + (size_t)(s + 1) * slots;
            for (int jg = 0; jg < slots; jg += 8) {
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
                __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
                for (int d = 0; d < 32; d += 4) {
                    __m256 k0 = _mm256_loadu_ps(kT + (size_t)d * slots + jg);
                    __m256 k1 = _mm256_loadu_ps(kT + (size_t)(d + 1) * slots + jg);
                    __m256 k2 = _mm256_loadu_ps(kT + (size_t)(d + 2) * slots + jg);
                    __m256 k3 = _mm256_loadu_ps(kT + (size_t)(d + 3) * slots + jg);
                    a0 = _mm256_fmadd_ps(_mm256_set1_ps(q0[d]), k0, a0);
                    a1 = _mm256_fmadd_ps(_mm256_set1_ps(q0[d + 1]), k1, a1);
                    a2 = _mm256_fmadd_ps(_mm256_set1_ps(q0[d + 2]), k2, a2);
                    a3 = _mm256_fmadd_ps(_mm256_set1_ps(q0[d + 3]), k3, a3);
                    b0 = _mm256_fmadd_ps(_mm256_set1_ps(q1[d]), k0, b0);
                    b1 = _mm256_fmadd_ps(_mm256_set1_ps(q1[d + 1]), k1, b1);
                    b2 = _mm256_fmadd_ps(_mm256_set1_ps(q1[d + 2]), k2, b2);
                    b3 = _mm256_fmadd_ps(_mm256_set1_ps(q1[d + 3]), k3, b3);
                }
                __m256 s0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
                __m256 s1 = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
                _mm256_storeu_ps(r0 + jg, _mm256_add_ps(s0, _mm256_loadu_ps(pr0 + jg)));
                _mm256_storeu_ps(r1 + jg, _mm256_add_ps(s1, _mm256_loadu_ps(pr1 + jg)));
            }
            float mx0 = -INFINITY, mx1 = -INFINITY;
            for (int j2 = 0; j2 < slots; j2++) {
                if (r0[j2] > mx0) mx0 = r0[j2];
                if (r1[j2] > mx1) mx1 = r1[j2];
            }
            for (int j2 = 0; j2 < slots; j2++) {
                r0[j2] -= mx0;
                r1[j2] -= mx1;
            }
        }
        if (slots & 1) {                       // 奇数槽尾(当前 win=8/slots=64 不触发)
            int s = slots - 1;
            const float* q0 = qkv + (size_t)s * row_stride;
            float* r0 = scores + (size_t)s * slots;
            for (int j = 0; j < slots; j++) {
                r0[j] = (valid[j] ? dot32(q0, qkv + (size_t)j * row_stride + 32) : 0.0f)
                      + prior[(size_t)s * slots + j];
            }
            float mx = -INFINITY;
            for (int j2 = 0; j2 < slots; j2++) if (r0[j2] > mx) mx = r0[j2];
            for (int j2 = 0; j2 < slots; j2++) r0[j2] -= mx;
        }
    }
#else
    for (int s = 0; s < slots; s++) {
        if (!valid[s]) continue;
        const float* qr = qkv + (size_t)s * row_stride;
        float* sr = scores + (size_t)s * slots;
        const float* pr = prior + (size_t)s * slots;
        float mx = -INFINITY;
        for (int j = 0; j < slots; j++) {
            float acc = (valid[j] ? dot32(qr, qkv + (size_t)j * row_stride + 32) : 0.0f) + pr[j];
            sr[j] = acc;
            if (acc > mx) mx = acc;
        }
        for (int j = 0; j < slots; j++) sr[j] -= mx;
    }
#endif
    if (g_aprof_on) {
        t2 = std::chrono::steady_clock::now();
        g_aprof[1] += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    // softmax:整块一次 exp 调用(无效查询行是垃圾但无人读,不影响输出)
    exp_vec(scores, (size_t)slots * slots);
    if (g_aprof_on) {
        t3 = std::chrono::steady_clock::now();
        g_aprof[2] += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }
    // av:双查询共享 v 加载(每 j 一次 4 ymm 加载喂两个查询),4 d 组双 acc。
    {
        static thread_local std::vector<const float*> vrows_buf;
        vrows_buf.resize(slots);
        for (int j = 0; j < slots; j++) {
            vrows_buf[j] = valid[j] ? qkv + (size_t)j * row_stride + 64 : nullptr;
        }
#if defined(NR_X86) && defined(__AVX2__)
        int s = 0;
        for (; s + 1 < slots; s += 2) {
            const float* sr0 = scores + (size_t)s * slots;
            const float* sr1 = scores + (size_t)(s + 1) * slots;
            float den0 = 0.0f, den1 = 0.0f;
            for (int j = 0; j < slots; j++) { den0 += sr0[j]; den1 += sr1[j]; }
            float inv0 = den0 > 0.0f ? 1.0f / den0 : 0.0f;
            float inv1 = den1 > 0.0f ? 1.0f / den1 : 0.0f;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
            __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
            for (int j = 0; j < slots; j++) {
                const float* vr = vrows_buf[j];
                if (!vr) continue;
                float w0 = sr0[j] * inv0, w1 = sr1[j] * inv1;
                if (w0 == 0.0f && w1 == 0.0f) continue;
                __m256 v0 = _mm256_loadu_ps(vr), v1 = _mm256_loadu_ps(vr + 8);
                __m256 v2 = _mm256_loadu_ps(vr + 16), v3 = _mm256_loadu_ps(vr + 24);
                if (w0 != 0.0f) {
                    __m256 x = _mm256_set1_ps(w0);
                    a0 = _mm256_fmadd_ps(x, v0, a0); a1 = _mm256_fmadd_ps(x, v1, a1);
                    a2 = _mm256_fmadd_ps(x, v2, a2); a3 = _mm256_fmadd_ps(x, v3, a3);
                }
                if (w1 != 0.0f) {
                    __m256 x = _mm256_set1_ps(w1);
                    b0 = _mm256_fmadd_ps(x, v0, b0); b1 = _mm256_fmadd_ps(x, v1, b1);
                    b2 = _mm256_fmadd_ps(x, v2, b2); b3 = _mm256_fmadd_ps(x, v3, b3);
                }
            }
            if (valid[s]) {
                float* orow = out + (size_t)index[s] * out_stride;
                _mm256_storeu_ps(orow, a0); _mm256_storeu_ps(orow + 8, a1);
                _mm256_storeu_ps(orow + 16, a2); _mm256_storeu_ps(orow + 24, a3);
            }
            if (valid[s + 1]) {
                float* orow = out + (size_t)index[s + 1] * out_stride;
                _mm256_storeu_ps(orow, b0); _mm256_storeu_ps(orow + 8, b1);
                _mm256_storeu_ps(orow + 16, b2); _mm256_storeu_ps(orow + 24, b3);
            }
        }
        for (; s < slots; s++) {
            if (!valid[s]) continue;
            const float* sr = scores + (size_t)s * slots;
            float den = 0.0f;
            for (int j = 0; j < slots; j++) den += sr[j];
            float inv = den > 0.0f ? 1.0f / den : 0.0f;
            av32(sr, slots, inv, vrows_buf.data(), out + (size_t)index[s] * out_stride);
        }
#else
        for (int s = 0; s < slots; s++) {
            if (!valid[s]) continue;
            const float* sr = scores + (size_t)s * slots;
            float den = 0.0f;
            for (int j = 0; j < slots; j++) den += sr[j];
            float inv = den > 0.0f ? 1.0f / den : 0.0f;
            av32(sr, slots, inv, vrows_buf.data(), out + (size_t)index[s] * out_stride);
        }
#endif
    }
    if (g_aprof_on) {
        auto t4 = std::chrono::steady_clock::now();
        g_aprof[3] += std::chrono::duration<double, std::milli>(t4 - t3).count();
    }
}

void attn_vit_head(float* qkv, int row_stride, int tokens, int padded, float scale,
                   float* scores, float* out, int out_stride) {
    // k 的 cosine 归一只做一次(见 attn.h;原实现逐查询重复归一同一原始 k)。
    for (int t2 = 0; t2 < tokens; t2++) {
        cosine_norm32(qkv + (size_t)t2 * row_stride + 32);
    }
    static thread_local std::vector<const float*> vrows_buf;
    vrows_buf.resize(tokens);
    for (int t2 = 0; t2 < tokens; t2++) {
        vrows_buf[t2] = qkv + (size_t)t2 * row_stride + 64;
    }
    for (int ti = 0; ti < tokens; ti++) {
        float* qr = qkv + (size_t)ti * row_stride;
        cosine_norm32(qr);
        scale32(qr, scale);
        float* sr = scores + (size_t)ti * padded;
        int t2 = 0;
#if defined(NR_X86) && defined(__AVX2__)
        for (; t2 + 4 <= tokens; t2 += 4) {
            __m256 p0 = dot32_partial(qr, qkv + (size_t)t2 * row_stride + 32);
            __m256 p1 = dot32_partial(qr, qkv + (size_t)(t2 + 1) * row_stride + 32);
            __m256 p2 = dot32_partial(qr, qkv + (size_t)(t2 + 2) * row_stride + 32);
            __m256 p3 = dot32_partial(qr, qkv + (size_t)(t2 + 3) * row_stride + 32);
            sr[t2] = hsum256(p0); sr[t2 + 1] = hsum256(p1);
            sr[t2 + 2] = hsum256(p2); sr[t2 + 3] = hsum256(p3);
        }
#endif
        for (; t2 < tokens; t2++) {
            sr[t2] = dot32(qr, qkv + (size_t)t2 * row_stride + 32);
        }
        float mx = -INFINITY;
        for (int t2 = 0; t2 < tokens; t2++) {
            if (sr[t2] > mx) mx = sr[t2];
        }
        for (int t2 = 0; t2 < tokens; t2++) sr[t2] -= mx;
    }
    exp_vec(scores, (size_t)tokens * padded);
    for (int t2 = 0; t2 < tokens; t2++) {
        float* sr = scores + (size_t)t2 * padded;
        for (int t3 = tokens; t3 < padded; t3++) sr[t3] = 0.0f;
    }
    for (int ti = 0; ti < tokens; ti++) {
        const float* sr = scores + (size_t)ti * padded;
        float den = 0.0f;
        for (int t2 = 0; t2 < tokens; t2++) den += sr[t2];
        float inv = den > 0.0f ? 1.0f / den : 0.0f;
        av32(sr, tokens, inv, vrows_buf.data(), out + (size_t)ti * out_stride);
    }
}
