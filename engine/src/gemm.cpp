#include "gemm.h"
#include "gemm_vnni.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define NR_X86 1
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#endif

namespace {

#if defined(_MSC_VER)
#define NR_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define NR_FORCE_INLINE inline __attribute__((always_inline))
#else
#define NR_FORCE_INLINE inline
#endif

// 边角 tile(mr x nr 尾块):Bp 行距 Ns,按 k 跨步累加(head N=4 全走这)。
void tile_edge(const float* A, int K, int Ns, int mr, int nr, const float* Bp, float* C,
               const float* bias, bool act, const float* X, const float* aux) {
    for (int r = 0; r < mr; r++) {
        const float* ar = A + (size_t)r * K;
        float* cr = C + (size_t)r * Ns;
        for (int n = 0; n < nr; n++) {
            float acc = bias ? bias[n] : 0.0f;
            const float* b = Bp + n;
            int k = 0;
            for (; k + 3 < K; k += 4) {
                acc += ar[k] * b[(size_t)k * Ns] + ar[k + 1] * b[(size_t)(k + 1) * Ns]
                     + ar[k + 2] * b[(size_t)(k + 2) * Ns]
                     + ar[k + 3] * b[(size_t)(k + 3) * Ns];
            }
            for (; k < K; k++) acc += ar[k] * b[(size_t)k * Ns];
            if (X) acc += X[(size_t)r * Ns + n] * aux[n];
            if (act) acc = acc / (1.0f + std::exp(-acc));
            cr[n] = acc;
        }
    }
}

// 行指针版尾块(gemm_kn_rows 用)。
void tile_edge_rows(const float* A, int K, int Ns, int mr, int nr, const float* Bp,
                    float* const* crows, const float* bias) {
    for (int r = 0; r < mr; r++) {
        const float* ar = A + (size_t)r * K;
        float* cr = crows[r];
        for (int n = 0; n < nr; n++) {
            float acc = bias ? bias[n] : 0.0f;
            const float* b = Bp + n;
            int k = 0;
            for (; k + 3 < K; k += 4) {
                acc += ar[k] * b[(size_t)k * Ns] + ar[k + 1] * b[(size_t)(k + 1) * Ns]
                     + ar[k + 2] * b[(size_t)(k + 2) * Ns]
                     + ar[k + 3] * b[(size_t)(k + 3) * Ns];
            }
            for (; k < K; k++) acc += ar[k] * b[(size_t)k * Ns];
            cr[n] = acc;
        }
    }
}

#if defined(NR_X86) && defined(__AVX2__)
// The RGB/blend head has only four columns; the 16-column kernel cannot
// handle it. Vectorize its outputs and reuse each weight across eight rows.
NR_FORCE_INLINE void kernel_8x4(const float* A, const float* B, int K, float* C, const float* bias) {
    __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps();
    __m128 a2 = _mm_setzero_ps(), a3 = _mm_setzero_ps();
    __m128 a4 = _mm_setzero_ps(), a5 = _mm_setzero_ps();
    __m128 a6 = _mm_setzero_ps(), a7 = _mm_setzero_ps();
    for (int k = 0; k < K; ++k) {
        __m128 b = _mm_loadu_ps(B + k * 4);
        a0 = _mm_fmadd_ps(_mm_set1_ps(A[k]), b, a0);
        a1 = _mm_fmadd_ps(_mm_set1_ps(A[K + k]), b, a1);
        a2 = _mm_fmadd_ps(_mm_set1_ps(A[2 * K + k]), b, a2);
        a3 = _mm_fmadd_ps(_mm_set1_ps(A[3 * K + k]), b, a3);
        a4 = _mm_fmadd_ps(_mm_set1_ps(A[4 * K + k]), b, a4);
        a5 = _mm_fmadd_ps(_mm_set1_ps(A[5 * K + k]), b, a5);
        a6 = _mm_fmadd_ps(_mm_set1_ps(A[6 * K + k]), b, a6);
        a7 = _mm_fmadd_ps(_mm_set1_ps(A[7 * K + k]), b, a7);
    }
    if (bias) {
        __m128 b = _mm_loadu_ps(bias);
        a0 = _mm_add_ps(a0, b); a1 = _mm_add_ps(a1, b);
        a2 = _mm_add_ps(a2, b); a3 = _mm_add_ps(a3, b);
        a4 = _mm_add_ps(a4, b); a5 = _mm_add_ps(a5, b);
        a6 = _mm_add_ps(a6, b); a7 = _mm_add_ps(a7, b);
    }
    _mm_storeu_ps(C, a0); _mm_storeu_ps(C + 4, a1);
    _mm_storeu_ps(C + 8, a2); _mm_storeu_ps(C + 12, a3);
    _mm_storeu_ps(C + 16, a4); _mm_storeu_ps(C + 20, a5);
    _mm_storeu_ps(C + 24, a6); _mm_storeu_ps(C + 28, a7);
}

// exp(x):x = k*ln2 + r(|r| <= ln2/2),exp = 2^k * poly6(r);6 阶 Taylor
// 尾项 r^7/7! <= 1.2e-7 相对,再加系数舍入 ~1e-7 —— 对 2e-3 的 parity 线无感。
__m256 exp_ps(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));    // 2^k 指数域守卫(k <= 127)
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
    __m256 kf = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(-2.1219444030165678E-4f), r);
    __m256 p = _mm256_set1_ps(1.0f / 720.0f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    __m256i ik = _mm256_cvtps_epi32(kf);
    __m256i ip = _mm256_slli_epi32(_mm256_add_epi32(ik, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(ip));
}

void silu_vec_avx2(float* v, size_t n) {
    size_t i = 0;
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(v + i);
        __m256 e = exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
        _mm256_storeu_ps(v + i, _mm256_div_ps(x, _mm256_add_ps(one, e)));
    }
    for (; i < n; i++) v[i] = v[i] / (1.0f + std::exp(-v[i]));
}

// GEMM 出口融合的 silu(与 silu_vec_avx2 逐位同式)。
inline __m256 silu_ps(__m256 x) {
    __m256 e = exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

// 4x16 微内核:acc8 个 ymm(每行两个覆盖 16 列);每 k 两次 B 行加载 + 4 次
// 广播 FMA —— 广播:FMA = 1:2,比 8x8 的 1:1 少负载口压力;14 个 ymm 寄存器
// 零溢出(6x16 的 20 个会爆寄存器,实测慢 6x)。真实形状的 M 全是 4 的倍数。
NR_FORCE_INLINE void kernel_4x16(const float* A, const float* Bp, int K, int Ns, float* C,
                 const float* bias, bool act, const float* X, const float* aux) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
    for (int k = 0; k < K; k++) {
        const float* b = Bp + (size_t)k * Ns;
        __m256 v0 = _mm256_loadu_ps(b), v1 = _mm256_loadu_ps(b + 8);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k), v0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k), v1, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + K + k), v0, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + K + k), v1, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 2 * K + k), v0, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 2 * K + k), v1, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 3 * K + k), v0, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 3 * K + k), v1, acc7);
    }
    if (bias) {
        __m256 b0 = _mm256_loadu_ps(bias), b1 = _mm256_loadu_ps(bias + 8);
        acc0 = _mm256_add_ps(acc0, b0); acc1 = _mm256_add_ps(acc1, b1);
        acc2 = _mm256_add_ps(acc2, b0); acc3 = _mm256_add_ps(acc3, b1);
        acc4 = _mm256_add_ps(acc4, b0); acc5 = _mm256_add_ps(acc5, b1);
        acc6 = _mm256_add_ps(acc6, b0); acc7 = _mm256_add_ps(acc7, b1);
    }
    if (X) {                                        // 残差融合:C += X*aux
        __m256 a0 = _mm256_loadu_ps(aux), a1 = _mm256_loadu_ps(aux + 8);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(X), a0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(X + 8), a1, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(X + Ns), a0, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(X + Ns + 8), a1, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(X + 2 * Ns), a0, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(X + 2 * Ns + 8), a1, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(X + 3 * Ns), a0, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(X + 3 * Ns + 8), a1, acc7);
    }
    if (act) {
        acc0 = silu_ps(acc0); acc1 = silu_ps(acc1); acc2 = silu_ps(acc2); acc3 = silu_ps(acc3);
        acc4 = silu_ps(acc4); acc5 = silu_ps(acc5); acc6 = silu_ps(acc6); acc7 = silu_ps(acc7);
    }
    _mm256_storeu_ps(C, acc0);              _mm256_storeu_ps(C + 8, acc1);
    _mm256_storeu_ps(C + Ns, acc2);         _mm256_storeu_ps(C + Ns + 8, acc3);
    _mm256_storeu_ps(C + 2 * Ns, acc4);     _mm256_storeu_ps(C + 2 * Ns + 8, acc5);
    _mm256_storeu_ps(C + 3 * Ns, acc6);     _mm256_storeu_ps(C + 3 * Ns + 8, acc7);
}

// 行指针版 4x16(gemm_kn_rows 用:qkv 槽序写入,消 attention gather)。
NR_FORCE_INLINE void kernel_4x16_rows(const float* A, const float* Bp, int K, int Ns,
                      float* const* crows, const float* bias) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
    for (int k = 0; k < K; k++) {
        const float* b = Bp + (size_t)k * Ns;
        __m256 v0 = _mm256_loadu_ps(b), v1 = _mm256_loadu_ps(b + 8);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k), v0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k), v1, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + K + k), v0, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + K + k), v1, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 2 * K + k), v0, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 2 * K + k), v1, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 3 * K + k), v0, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 3 * K + k), v1, acc7);
    }
    if (bias) {
        __m256 b0 = _mm256_loadu_ps(bias), b1 = _mm256_loadu_ps(bias + 8);
        acc0 = _mm256_add_ps(acc0, b0); acc1 = _mm256_add_ps(acc1, b1);
        acc2 = _mm256_add_ps(acc2, b0); acc3 = _mm256_add_ps(acc3, b1);
        acc4 = _mm256_add_ps(acc4, b0); acc5 = _mm256_add_ps(acc5, b1);
        acc6 = _mm256_add_ps(acc6, b0); acc7 = _mm256_add_ps(acc7, b1);
    }
    _mm256_storeu_ps(crows[0], acc0);       _mm256_storeu_ps(crows[0] + 8, acc1);
    _mm256_storeu_ps(crows[1], acc2);       _mm256_storeu_ps(crows[1] + 8, acc3);
    _mm256_storeu_ps(crows[2], acc4);       _mm256_storeu_ps(crows[2] + 8, acc5);
    _mm256_storeu_ps(crows[3], acc6);       _mm256_storeu_ps(crows[3] + 8, acc7);
}

void exp_vec_avx2(float* v, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(v + i, exp_ps(_mm256_loadu_ps(v + i)));
    for (; i < n; i++) v[i] = std::exp(v[i]);
}

// Small MoE experts often have only 1-3 tokens. Keep these rows vectorized
// instead of sending their large K/N matrices through the scalar edge tile.
template<int MR>
NR_FORCE_INLINE void kernel_short_x16(const float* A, const float* B, int K, int N,
                                      float* const* crows, bool act) {
    __m256 lo[MR], hi[MR];
    for (int r = 0; r < MR; ++r) lo[r] = hi[r] = _mm256_setzero_ps();
    for (int k = 0; k < K; ++k) {
        __m256 b0 = _mm256_loadu_ps(B + static_cast<size_t>(k) * N);
        __m256 b1 = _mm256_loadu_ps(B + static_cast<size_t>(k) * N + 8);
        for (int r = 0; r < MR; ++r) {
            __m256 a = _mm256_broadcast_ss(A + static_cast<size_t>(r) * K + k);
            lo[r] = _mm256_fmadd_ps(a, b0, lo[r]);
            hi[r] = _mm256_fmadd_ps(a, b1, hi[r]);
        }
    }
    for (int r = 0; r < MR; ++r) {
        if (act) { lo[r] = silu_ps(lo[r]); hi[r] = silu_ps(hi[r]); }
        _mm256_storeu_ps(crows[r], lo[r]);
        _mm256_storeu_ps(crows[r] + 8, hi[r]);
    }
}
#endif

// 运行期守卫:按 AVX2 编译的二进制落到老 CPU 上退回标量,避免非法指令。
bool cpu_has_avx2() {
#if defined(NR_X86) && defined(__AVX2__) && defined(_MSC_VER)
    // clang-cl's __builtin_cpu_supports requires a compiler-rt CPU model
    // symbol on Windows. Use the MSVC-compatible intrinsics on both compilers.
    static const bool v = [] {
        int info[4];
        __cpuid(info, 1);
        constexpr int required = (1 << 27) | (1 << 28) | (1 << 12); // OSXSAVE, AVX, FMA
        if ((info[2] & required) != required || (_xgetbv(0) & 6) != 6) return false;
        __cpuidex(info, 7, 0);
        return (info[1] & (1 << 5)) != 0;
    }();
    return v;
#elif defined(NR_X86) && defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__))
    static const bool v = __builtin_cpu_supports("avx2") != 0;
    return v;
#else
    return false;
#endif
}

}  // namespace

void pack_weight_kn(const float* w_nk, int N, int K, float* out_kn) {
    for (int n = 0; n < N; n++) {
        const float* wr = w_nk + (size_t)n * K;
        for (int k = 0; k < K; k++) out_kn[(size_t)k * N + n] = wr[k];
    }
}

void gemm_kn_rows(const float* A, int M, int K, int N, const float* Bp, float* C,
                  const float* bias, const int* row_map) {
    // 行排列输出(纯 + bias):kernel 按行指针存。分块同 gemm_kn。
    const bool avx = cpu_has_avx2();
    int MC = 64;
    while (M / MC < 16 && MC > 4) MC /= 2;
    const int mblocks = (M + MC - 1) / MC;
    const bool par = (long long)M * K * N > 2000000 && mblocks > 1;
#pragma omp parallel for schedule(static) if (par)
    for (int bi = 0; bi < mblocks; bi++) {
        const int m0 = bi * MC;
        const int mend = std::min(m0 + MC, M);
        for (int n0 = 0; n0 < N; n0 += 16) {
            const int nr = std::min(16, N - n0);
            const float* bp = Bp + n0;
            const float* bs = bias ? bias + n0 : nullptr;
            for (int m = m0; m < mend; m += 4) {
                const int mr = std::min(4, mend - m);
                float* crows[4] = {nullptr, nullptr, nullptr, nullptr};
                for (int r = 0; r < mr; r++) {
                    crows[r] = C + (size_t)row_map[m + r] * N + n0;
                }
#if defined(NR_X86) && defined(__AVX2__)
                if (avx && mr == 4 && nr == 16) {
                    kernel_4x16_rows(A + (size_t)m * K, bp, K, N, crows, bs);
                    continue;
                }
#endif
                tile_edge_rows(A + (size_t)m * K, K, N, mr, nr, bp, crows, bs);
            }
        }
    }
    (void)avx;
}

void gemm_kn(const float* A, int M, int K, int N, const float* Bp, float* C,
             const float* bias, bool act, const float* X, const float* aux) {
    const bool avx = cpu_has_avx2();
    int MC = 64;                // 行大块:Bp 面板在块内 L1 复用(大 K 关键)
    while (M / MC < 16 && MC > 6) MC /= 2;   // 小 M 提高块数,保住并行度
    const int mblocks = (M + MC - 1) / MC;
    const bool par = (long long)M * K * N > 2000000 && mblocks > 1;
#pragma omp parallel for schedule(static) if (par)
    for (int bi = 0; bi < mblocks; bi++) {
        const int m0 = bi * MC;
        const int mend = std::min(m0 + MC, M);
#if defined(NR_X86) && defined(__AVX2__)
        if (avx && N == 4 && !act && !X) {
            int m = m0;
            for (; m + 8 <= mend; m += 8)
                kernel_8x4(A + static_cast<size_t>(m) * K, Bp, K, C + m * 4, bias);
            for (; m < mend; m += 4)
                tile_edge(A + static_cast<size_t>(m) * K, K, N, std::min(4, mend - m), N,
                          Bp, C + m * 4, bias, false, nullptr, nullptr);
            continue;
        }
#endif
        for (int n0 = 0; n0 < N; n0 += 16) {
            const int nr = std::min(16, N - n0);
            const float* bp = Bp + n0;
            const float* bs = bias ? bias + n0 : nullptr;
            const float* axs = aux ? aux + n0 : nullptr;
            for (int m = m0; m < mend; m += 4) {
                const int mr = std::min(4, mend - m);
                float* c = C + (size_t)m * N + n0;
                const float* xr = X ? X + (size_t)m * N + n0 : nullptr;
#if defined(NR_X86) && defined(__AVX2__)
                if (avx && mr == 4 && nr == 16) {
                    kernel_4x16(A + (size_t)m * K, bp, K, N, c, bs, act, xr, axs);
                    continue;
                }
#endif
                tile_edge(A + (size_t)m * K, K, N, mr, nr, bp, c, bs, act, xr, axs);
            }
        }
    }
    (void)avx;
}

void gemm_kn_grouped(const float* A, int K, int N, const float* weights, float* C,
                     const int* offsets, int experts, bool act, const int* row_map) {
    const bool avx = cpu_has_avx2();
    struct Task { int expert, row, count, column, end_column; };
    std::vector<Task> tasks;
    constexpr int MC = 32;
    for (int e = 0; e < experts; ++e) {
        const int count = offsets[e + 1] - offsets[e];
        // Split output columns as well for token experts: M alone cannot
        // expose enough parallelism when each expert receives only 3 tokens.
        const int NC = count < MC ? 64 : N;
        for (int m = offsets[e]; m < offsets[e + 1]; m += MC)
            for (int n = 0; n < N; n += NC)
                tasks.push_back({e, m, std::min(MC, offsets[e + 1] - m), n,
                                 std::min(n + NC, N)});
    }
    const int task_count = static_cast<int>(tasks.size());
    const bool par = static_cast<long long>(offsets[experts]) * K * N > 1000000
                  && task_count > 1;
#pragma omp parallel for schedule(static) if (par)
    for (int ti = 0; ti < task_count; ++ti) {
        const auto& task = tasks[ti];
        const float* expert_w = weights + static_cast<size_t>(task.expert) * K * N;
        for (int n = task.column; n < task.end_column; n += 16) {
            const int nr = std::min(16, task.end_column - n);
            for (int m = task.row; m < task.row + task.count; m += 4) {
                const int mr = std::min(4, task.row + task.count - m);
                const float* a = A + static_cast<size_t>(m) * K;
                const float* b = expert_w + n;
                float* crows[4];
                for (int r = 0; r < mr; ++r)
                    crows[r] = C + static_cast<size_t>(row_map ? row_map[m + r] : m + r) * N + n;
#if defined(NR_X86) && defined(__AVX2__)
                if (avx && nr == 16) {
                    if (mr == 4 && !row_map)
                        kernel_4x16(a, b, K, N, crows[0], nullptr, act, nullptr, nullptr);
                    else if (mr == 4 && !act)
                        kernel_4x16_rows(a, b, K, N, crows, nullptr);
                    else if (mr == 1) kernel_short_x16<1>(a, b, K, N, crows, act);
                    else if (mr == 2) kernel_short_x16<2>(a, b, K, N, crows, act);
                    else if (mr == 3) kernel_short_x16<3>(a, b, K, N, crows, act);
                    else { // mapped activation is uncommon; still support it.
                        for (int r = 0; r < mr; ++r) {
                            float* cr = crows[r];
                            kernel_short_x16<1>(a + static_cast<size_t>(r) * K, b, K, N, &cr, act);
                        }
                    }
                    continue;
                }
#endif
                for (int r = 0; r < mr; ++r)
                    tile_edge(a + static_cast<size_t>(r) * K, K, N, 1, nr, b, crows[r],
                              nullptr, act, nullptr, nullptr);
            }
        }
    }
    (void)avx;
}

void ffn_kn(const float* state, int rows, int channels, int hidden,
            const float* expand, const float* expand_bias,
            const float* contract, const float* contract_bias,
            const float* aux, float* out, int outputs, int experts) {
    const bool avx = cpu_has_avx2();
    if (!outputs) outputs = channels;
    // The intermediate stays in each worker's L1/L2 instead of traversing a
    // full-resolution buffer twice. Use one team for both matrix multiplies.
    constexpr int MC = 32;
    const int row_blocks = (rows + MC - 1) / MC;
    const int blocks = row_blocks * experts;
    const bool par = static_cast<long long>(rows) * channels * hidden * experts > 1000000 && blocks > 1;
#pragma omp parallel if (par)
    {
        static thread_local std::vector<float> intermediate;
        intermediate.resize(static_cast<size_t>(MC) * hidden);
#pragma omp for schedule(static)
        for (int bi = 0; bi < blocks; ++bi) {
            int expert = bi / row_blocks;
            int m0 = (bi % row_blocks) * MC, count = std::min(MC, rows - m0);
            const float* we = expand + static_cast<size_t>(expert) * channels * hidden;
            const float* wc = contract + static_cast<size_t>(expert) * hidden * outputs;
            float* oe = out + static_cast<size_t>(expert) * rows * outputs;
            for (int n0 = 0; n0 < hidden; n0 += 16) {
                int nr = std::min(16, hidden - n0);
                const float* bias = expand_bias ? expand_bias + expert * hidden + n0 : nullptr;
                for (int m = 0; m < count; m += 4) {
                    int mr = std::min(4, count - m);
                    const float* a = state + static_cast<size_t>(m0 + m) * channels;
                    float* c = intermediate.data() + static_cast<size_t>(m) * hidden + n0;
#if defined(NR_X86) && defined(__AVX2__)
                    if (avx && mr == 4 && nr == 16) {
                        kernel_4x16(a, we + n0, channels, hidden, c, bias, true, nullptr, nullptr);
                        continue;
                    }
#endif
                    tile_edge(a, channels, hidden, mr, nr, we + n0, c, bias, true, nullptr, nullptr);
                }
            }
            // Expand has consumed this tile before contract overwrites state.
            for (int n0 = 0; n0 < outputs; n0 += 16) {
                int nr = std::min(16, outputs - n0);
                const float* bias = contract_bias ? contract_bias + expert * outputs + n0 : nullptr;
                for (int m = 0; m < count; m += 4) {
                    int mr = std::min(4, count - m);
                    const float* a = intermediate.data() + static_cast<size_t>(m) * hidden;
                    const float* x = aux ? state + static_cast<size_t>(m0 + m) * channels + n0 : nullptr;
                    const float* axs = aux ? aux + n0 : nullptr;
                    float* c = oe + static_cast<size_t>(m0 + m) * outputs + n0;
#if defined(NR_X86) && defined(__AVX2__)
                    if (avx && mr == 4 && nr == 16) {
                        kernel_4x16(a, wc + n0, hidden, outputs, c, bias, false, x, axs);
                        continue;
                    }
#endif
                    tile_edge(a, hidden, outputs, mr, nr, wc + n0, c, bias, false, x, axs);
                }
            }
        }
    }
    (void)avx;
}

void gemm_kn_serial(const float* A, int M, int K, int N, const float* Bp, float* C,
                    const float* bias, bool act, const float* X, const float* aux) {
    const bool avx = cpu_has_avx2();
    for (int n = 0; n < N; n += 16) {
        int nr = std::min(16, N - n);
        for (int m = 0; m < M; m += 4) {
            int mr = std::min(4, M - m);
            const float* bs = bias ? bias + n : nullptr;
            const float* xr = X ? X + static_cast<size_t>(m) * N + n : nullptr;
            const float* ax = aux ? aux + n : nullptr;
            float* c = C + static_cast<size_t>(m) * N + n;
#if defined(NR_X86) && defined(__AVX2__)
            if (avx && mr == 4 && nr == 16) {
                kernel_4x16(A + static_cast<size_t>(m) * K, Bp + n, K, N, c, bs, act, xr, ax);
                continue;
            }
#endif
            tile_edge(A + static_cast<size_t>(m) * K, K, N, mr, nr, Bp + n, c, bs, act, xr, ax);
        }
    }
    (void)avx;
}


namespace {

void silu_range(float* v, size_t n) {
#if defined(NR_X86) && defined(__AVX2__)
    if (cpu_has_avx2()) { silu_vec_avx2(v, n); return; }
#endif
    for (size_t i = 0; i < n; i++) v[i] = v[i] / (1.0f + std::exp(-v[i]));
}

void exp_range(float* v, size_t n) {
#if defined(NR_X86) && defined(__AVX2__)
    if (cpu_has_avx2()) { exp_vec_avx2(v, n); return; }
#endif
    for (size_t i = 0; i < n; i++) v[i] = std::exp(v[i]);
}

// 大缓冲分块并行(这些 pass 以前是单线程,512² 下 silu 一项就 20ms+)。
constexpr size_t kActChunk = 64 * 1024;

}  // namespace

void silu_vec(float* v, size_t n) {
    if (n > kActChunk) {
        const int nb = (int)((n + kActChunk - 1) / kActChunk);
#pragma omp parallel for schedule(static)
        for (int b = 0; b < nb; b++) {
            size_t off = (size_t)b * kActChunk;
            silu_range(v + off, std::min(kActChunk, n - off));
        }
        return;
    }
    silu_range(v, n);
}

void exp_vec(float* v, size_t n) {
    if (n > kActChunk) {
        const int nb = (int)((n + kActChunk - 1) / kActChunk);
#pragma omp parallel for schedule(static)
        for (int b = 0; b < nb; b++) {
            size_t off = (size_t)b * kActChunk;
            exp_range(v + off, std::min(kActChunk, n - off));
        }
        return;
    }
    exp_range(v, n);
}

void field_downsample2(const float* x, float* out, int in_w, int in_h,
                       int out_w, int out_h, int channels) {
    const bool avx = cpu_has_avx2();
    const bool par = static_cast<size_t>(out_w) * out_h * channels > 65536;
#pragma omp parallel for schedule(static) if (par)
    for (int y = 0; y < out_h; ++y) {
        int sy = std::min(y * 2, in_h - 1);
        for (int xx = 0; xx < out_w; ++xx) {
            int sx = std::min(xx * 2, in_w - 1);
            float* o = out + (static_cast<size_t>(y) * out_w + xx) * channels;
            if (sy + 1 >= in_h || sx + 1 >= in_w) {
                std::fill(o, o + channels, 0.0f);
                continue;
            }
            const float* a = x + (static_cast<size_t>(sy) * in_w + sx) * channels;
            const float* b = a + channels;
            const float* c = a + static_cast<size_t>(in_w) * channels;
            const float* d = c + channels;
            int ch = 0;
#if defined(NR_X86) && defined(__AVX2__)
            if (avx) for (; ch + 8 <= channels; ch += 8) {
                __m256 sum = _mm256_add_ps(_mm256_loadu_ps(a + ch), _mm256_loadu_ps(b + ch));
                sum = _mm256_add_ps(sum, _mm256_loadu_ps(c + ch));
                sum = _mm256_add_ps(sum, _mm256_loadu_ps(d + ch));
                _mm256_storeu_ps(o + ch, _mm256_mul_ps(sum, _mm256_set1_ps(0.25f)));
            }
#endif
            for (; ch < channels; ++ch) o[ch] = (a[ch] + b[ch] + c[ch] + d[ch]) * 0.25f;
        }
    }
    (void)avx;
}

void field_upsample_merge(const float* x, const float* skip, const float* aux0,
                          const float* aux1, float* out, int in_w, int in_h,
                          int out_w, int out_h, int channels) {
    const bool avx = cpu_has_avx2();
    const bool par = static_cast<size_t>(out_w) * out_h * channels > 65536;
#pragma omp parallel for schedule(static) if (par)
    for (int y = 0; y < out_h; ++y) {
        int sy = std::min(y / 2, in_h - 1);
        for (int xx = 0; xx < out_w; ++xx) {
            int sx = std::min(xx / 2, in_w - 1);
            const float* s = x + (static_cast<size_t>(sy) * in_w + sx) * channels;
            const float* sk = skip + (static_cast<size_t>(y) * out_w + xx) * channels;
            float* o = out + (static_cast<size_t>(y) * out_w + xx) * channels;
            int ch = 0;
#if defined(NR_X86) && defined(__AVX2__)
            if (avx) for (; ch + 8 <= channels; ch += 8) {
                __m256 a = _mm256_loadu_ps(s + ch);
                if (aux0) a = _mm256_mul_ps(a, _mm256_loadu_ps(aux0 + ch));
                __m256 b = _mm256_mul_ps(_mm256_loadu_ps(sk + ch), _mm256_loadu_ps(aux1 + ch));
                // Keep multiply + add separate, matching the baseline field pass.
                _mm256_storeu_ps(o + ch, _mm256_add_ps(a, b));
            }
#endif
            for (; ch < channels; ++ch)
                o[ch] = (aux0 ? s[ch] * aux0[ch] : s[ch]) + sk[ch] * aux1[ch];
        }
    }
    (void)avx;
}

// ---- int8(任务 8.2) ------------------------------------------------------

namespace {

#if defined(NR_X86) && defined(__AVX2__)
inline int hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
    return _mm_cvtsi128_si32(lo);
}

// 2 行 x 4 列微内核:vpmovsxbw 展开 i16 + vpmaddwd 进 i32(无饱和路径)。
// W4: [4][Kp] i16(解包一次,行对循环复用);A 行距 Kp;Kp 是 16 的倍数。
void kernel_i8_2x4(const int8_t* A, const int16_t* W4, int Kp, int32_t out[8]) {
    __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
    __m256i acc4 = _mm256_setzero_si256(), acc5 = _mm256_setzero_si256();
    __m256i acc6 = _mm256_setzero_si256(), acc7 = _mm256_setzero_si256();
    for (int k = 0; k < Kp; k += 16) {
        __m256i a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(A + k)));
        __m256i a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(A + Kp + k)));
        __m256i w0 = _mm256_loadu_si256((const __m256i*)(W4 + k));
        __m256i w1 = _mm256_loadu_si256((const __m256i*)(W4 + Kp + k));
        __m256i w2 = _mm256_loadu_si256((const __m256i*)(W4 + 2 * Kp + k));
        __m256i w3 = _mm256_loadu_si256((const __m256i*)(W4 + 3 * Kp + k));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a0, w0));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a0, w1));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(a0, w2));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(a0, w3));
        acc4 = _mm256_add_epi32(acc4, _mm256_madd_epi16(a1, w0));
        acc5 = _mm256_add_epi32(acc5, _mm256_madd_epi16(a1, w1));
        acc6 = _mm256_add_epi32(acc6, _mm256_madd_epi16(a1, w2));
        acc7 = _mm256_add_epi32(acc7, _mm256_madd_epi16(a1, w3));
    }
    out[0] = hsum_epi32(acc0); out[4] = hsum_epi32(acc4);
    out[1] = hsum_epi32(acc1); out[5] = hsum_epi32(acc5);
    out[2] = hsum_epi32(acc2); out[6] = hsum_epi32(acc6);
    out[3] = hsum_epi32(acc3); out[7] = hsum_epi32(acc7);
}
#endif

// 边角 tile(mr<=2, nr<=4)标量版本;out 布局 [2][4]。
void tile_i8_scalar(const int8_t* A, int mr, const int8_t* W, int nr, int Kp, int32_t out[8]) {
    for (int r = 0; r < 2; r++) {
        for (int n = 0; n < 4; n++) out[r * 4 + n] = 0;
    }
    for (int r = 0; r < mr; r++) {
        const int8_t* ar = A + (size_t)r * Kp;
        for (int n = 0; n < nr; n++) {
            const int8_t* wr = W + (size_t)n * Kp;
            int32_t s = 0;
            for (int k = 0; k < Kp; k++) s += (int32_t)ar[k] * wr[k];
            out[r * 4 + n] = s;
        }
    }
}

inline int8_t quant_sym(float x, float inv) {
    float t = x * inv;
    int v = (int)(t + (t >= 0.0f ? 0.5f : -0.5f));
    return (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
}

// 量化一行:round + clamp 向量化(AVX2),打包按标量走(K 多为 16~256 的短行)。
void quantize_row(const float* x, int K, float inv, int8_t* out, int Kp) {
    int k = 0;
#if defined(NR_X86) && defined(__AVX2__)
    if (cpu_has_avx2()) {
        const __m256 iv = _mm256_set1_ps(inv);
        const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
        int32_t tmp[8];
        for (; k + 8 <= K; k += 8) {
            __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + k), iv);
            v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i i32 = _mm256_cvtps_epi32(v);
            i32 = _mm256_max_epi32(i32, lo);
            i32 = _mm256_min_epi32(i32, hi);
            _mm256_storeu_si256((__m256i*)tmp, i32);
            for (int j = 0; j < 8; j++) out[k + j] = (int8_t)tmp[j];
        }
    }
#endif
    for (; k < K; k++) out[k] = quant_sym(x[k], inv);
    for (; k < Kp; k++) out[k] = 0;
}

// 反量化 4 个输出:acc i32 -> f32 -> 乘 sA*sW -> 加 bias(SSE);出口融合同 gemm_kn。
inline void dequant4(const int32_t* acc, float sa, const float* sW, const float* bias,
                     const float* xr, const float* aux, float* cr, bool act) {
#if defined(NR_X86) && defined(__AVX2__)
    if (cpu_has_avx2()) {
        __m128 f4 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)acc));
        f4 = _mm_mul_ps(f4, _mm_mul_ps(_mm_set1_ps(sa), _mm_loadu_ps(sW)));
        if (bias) f4 = _mm_add_ps(f4, _mm_loadu_ps(bias));
        if (xr) f4 = _mm_fmadd_ps(_mm_loadu_ps(xr), _mm_loadu_ps(aux), f4);
        if (act) {
            __m256 v = _mm256_castps128_ps256(f4);
            f4 = _mm256_castps256_ps128(silu_ps(v));
        }
        _mm_storeu_ps(cr, f4);
        return;
    }
#endif
    for (int n = 0; n < 4; n++) {
        float x = acc[n] * (sa * sW[n]) + (bias ? bias[n] : 0.0f);
        if (xr) x += xr[n] * aux[n];
        if (act) x = x / (1.0f + std::exp(-x));
        cr[n] = x;
    }
}

}  // namespace

void quantize_w_nk(const float* w_nk, int N, int K, int Kp, int8_t* out, float* sW) {
    for (int n = 0; n < N; n++) {
        const float* wr = w_nk + (size_t)n * K;
        float mx = 0.0f;
        for (int k = 0; k < K; k++) mx = std::fmax(mx, std::fabs(wr[k]));
        const float s = mx > 0.0f ? mx * (1.0f / 127.0f) : 1.0f;
        sW[n] = s;
        int8_t* o = out + (size_t)n * Kp;
        const float inv = 1.0f / s;
        for (int k = 0; k < K; k++) o[k] = quant_sym(wr[k], inv);
        for (int k = K; k < Kp; k++) o[k] = 0;
    }
}

void quantize_w_kn(const float* w_kn, int K, int N, int Kp, int8_t* out, float* sW) {
    for (int n = 0; n < N; n++) {
        float mx = 0.0f;
        for (int k = 0; k < K; k++) mx = std::fmax(mx, std::fabs(w_kn[(size_t)k * N + n]));
        const float s = mx > 0.0f ? mx * (1.0f / 127.0f) : 1.0f;
        sW[n] = s;
        int8_t* o = out + (size_t)n * Kp;
        const float inv = 1.0f / s;
        for (int k = 0; k < K; k++) o[k] = quant_sym(w_kn[(size_t)k * N + n], inv);
        for (int k = K; k < Kp; k++) o[k] = 0;
    }
}

I8VnniWeights pack_weight_i8_vnni(const int8_t* Wq, int N, int Kp) {
    if (N <= 0 || Kp <= 0 || Kp % 4 != 0)
        throw std::invalid_argument("invalid VNNI weight shape");
    I8VnniWeights p;
    p.N = N; p.Kp = Kp;
    const int Np = (N + 15) & ~15;
    p.w.resize(static_cast<size_t>(Np) * Kp, 0);
    p.correction.resize(Np, 0);
    for (int n = 0; n < N; ++n) {
        const int8_t* row = Wq + static_cast<size_t>(n) * Kp;
        int32_t sum = 0;
        for (int k = 0; k < Kp; ++k) sum += row[k];
        p.correction[n] = -128 * sum;
        for (int k = 0; k < Kp; k += 4) {
            int8_t* dst = p.w.data() + static_cast<size_t>(n / 16) * Kp * 16
                        + static_cast<size_t>(k / 4) * 64 + (n % 16) * 4;
            for (int j = 0; j < 4; ++j) dst[j] = row[k + j];
        }
    }
    return p;
}

void gemm_kn_i8(const float* A, int M, int K, int N, const int8_t* Wq, int Kp,
                const float* sW, float* C, const float* bias, bool silu_act,
                const float* X, const float* aux, const I8VnniWeights* vnni) {
    const bool avx = cpu_has_avx2();
    const VnniDot dot = vnni ? vnni_kernel(selected_vnni_isa()) : nullptr;
    if (dot && (vnni->N != N || vnni->Kp != Kp ||
                vnni->w.size() != static_cast<size_t>((N + 15) & ~15) * Kp ||
                vnni->correction.size() != static_cast<size_t>((N + 15) & ~15)))
        throw std::invalid_argument("VNNI packed weight shape mismatch");
    int MT = 48;                                   // 行块 = 线程本地量化缓冲
    // Low-resolution/VIT layers often have only 64-144 rows. Keep enough
    // four-row tiles to use the worker team instead of giving it 2-3 blocks.
    if (dot) while (M / MT < 16 && MT > 4) MT = std::max(4, (MT / 2) & ~3);
    const int blocks = (M + MT - 1) / MT;
    const bool par = (long long)M * K * N > 2000000 && blocks > 1;
#pragma omp parallel if (par)
    {
        // thread_local:跨调用复用,steady-state 零 malloc。
        static thread_local std::vector<int8_t> aq;
        static thread_local std::vector<float> sA;
        static thread_local std::vector<int16_t> w4;    // W 列块解包 scratch(跨行对复用)
        aq.resize((size_t)MT * Kp);
        sA.resize(MT);
        if (!dot) w4.resize((size_t)4 * Kp);
#pragma omp for schedule(static)
        for (int bt = 0; bt < blocks; bt++) {
            const int m0 = bt * MT;
            const int mr_all = std::min(MT, M - m0);
            // 1) per-row max -> scale -> 量化(K 尾补零)
            for (int r = 0; r < mr_all; r++) {
                const float* ar = A + (size_t)(m0 + r) * K;
                float mx = 0.0f;
                for (int k = 0; k < K; k++) mx = std::fmax(mx, std::fabs(ar[k]));
                const float s = mx > 0.0f ? mx * (1.0f / 127.0f) : 1.0f;
                sA[r] = s;
                quantize_row(ar, K, 1.0f / s, aq.data() + (size_t)r * Kp, Kp);
            }
            if (dot && (mr_all & 3)) {
                const int padded_rows = (mr_all + 3) & ~3;
                std::fill(aq.begin() + static_cast<size_t>(mr_all) * Kp,
                          aq.begin() + static_cast<size_t>(padded_rows) * Kp, int8_t(0));
            }
            // Reuse the same quantization and dequant/SiLU as the AVX2 path.
            // MT is divisible by four; unused tail rows stay inside aq's arena.
            if (dot) {
                for (int n0 = 0; n0 < N; n0 += 16) {
                    const int nr = std::min(16, N - n0);
                    const int8_t* panel = vnni->w.data() + static_cast<size_t>(n0) * Kp;
                    for (int r0 = 0; r0 < mr_all; r0 += 4) {
                        int32_t acc32[64];
                        dot(aq.data() + static_cast<size_t>(r0) * Kp, panel,
                            vnni->correction.data() + n0, Kp, acc32);
                        for (int r = 0; r < std::min(4, mr_all - r0); ++r) {
                            const float sa = sA[r0 + r];
                            float* cr = C + static_cast<size_t>(m0 + r0 + r) * N + n0;
                            const float* xr = X ? X + static_cast<size_t>(m0 + r0 + r) * N + n0 : nullptr;
                            int n = 0;
                            for (; n + 4 <= nr; n += 4)
                                dequant4(acc32 + r * 16 + n, sa, sW + n0 + n,
                                         bias ? bias + n0 + n : nullptr, xr ? xr + n : nullptr,
                                         aux ? aux + n0 + n : nullptr, cr + n, silu_act);
                            for (; n < nr; ++n) {
                                float x = acc32[r * 16 + n] * (sa * sW[n0 + n])
                                        + (bias ? bias[n0 + n] : 0.0f);
                                if (xr) x += xr[n] * aux[n0 + n];
                                if (silu_act) x = x / (1.0f + std::exp(-x));
                                cr[n] = x;
                            }
                        }
                    }
                }
                continue;
            }
            // 2) int8 GEMM + 反量化 C = sA*sW*acc + bias
            for (int n0 = 0; n0 < N; n0 += 4) {
                const int nr = std::min(4, N - n0);
                for (int n = 0; n < nr; n++) {
                    const int8_t* wr = Wq + (size_t)(n0 + n) * Kp;
                    int16_t* o = w4.data() + (size_t)n * Kp;
                    for (int k = 0; k < Kp; k++) o[k] = wr[k];
                }
                for (int r0 = 0; r0 < mr_all; r0 += 2) {
                    const int rr = std::min(2, mr_all - r0);
                    int32_t acc32[8];
#if defined(NR_X86) && defined(__AVX2__)
                    if (avx && rr == 2 && nr == 4) {
                        kernel_i8_2x4(aq.data() + (size_t)r0 * Kp, w4.data(), Kp, acc32);
                    } else
#endif
                    {
                        tile_i8_scalar(aq.data() + (size_t)r0 * Kp, rr,
                                       Wq + (size_t)n0 * Kp, nr, Kp, acc32);
                    }
                    for (int r = 0; r < rr; r++) {
                        const float sa = sA[r0 + r];
                        float* cr = C + (size_t)(m0 + r0 + r) * N + n0;
                        const float* xr = X ? X + (size_t)(m0 + r0 + r) * N + n0 : nullptr;
                        const float* axs = aux ? aux + n0 : nullptr;
                        if (nr == 4) {
                            dequant4(acc32 + r * 4, sa, sW + n0, bias ? bias + n0 : nullptr,
                                     xr, axs, cr, silu_act);
                        } else {
                            for (int n = 0; n < nr; n++) {
                                float x = acc32[r * 4 + n] * (sa * sW[n0 + n])
                                        + (bias ? bias[n0 + n] : 0.0f);
                                if (xr) x += xr[n] * axs[n];
                                if (silu_act) x = x / (1.0f + std::exp(-x));
                                cr[n] = x;
                            }
                        }
                    }
                }
            }
        }
    }
    (void)avx;
}
