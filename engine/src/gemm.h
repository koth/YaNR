// GEMM 微内核(openspec 8.9 / M3 主路径):
//
//   C[M,N] = A[M,K] * Bp[K,N] + bias[N]        (bias 可为 nullptr)
//
// Bp 是"打包"权重布局 [K][N] 行主序:torch Linear 权重 [N][K] 经 pack_weight_kn
// 转置得到;专家 einsum 切片(expand[e][ch][h] / narrow[e][h][nc])本身就是 [K][N]。
// 累加沿 k 升序(与参考循环同一求和顺序);AVX2 路径用 FMA,与标量只差舍入。
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

// w_nk: [N][K] 行主序 -> out_kn: [K][N] 行主序(加载期一次性转置)。
void pack_weight_kn(const float* w_nk, int N, int K, float* out_kn);

// Cache-local FFN: expand+silu+contract+residual within one row tile.
// Packed weights use the same [K][N] layout as gemm_kn. out may alias state.
// aux=nullptr omits residual. Multiple experts write [expert][rows][outputs],
// preserving the trained expert-major reshape before the final contract.
void ffn_kn(const float* state, int rows, int channels, int hidden,
            const float* expand, const float* expand_bias,
            const float* contract, const float* contract_bias,
            const float* aux, float* out, int outputs = 0, int experts = 1);


// C: [M][N] 行主序;A: [M][K];Bp: [K][N]。
// 出口融合(与独立 pass 逐位同式):
//   silu_act = true            -> C = silu(bias + A*Bp)(expand 后的激活)
//   X/aux 非空                 -> C = bias + A*Bp + X[m][n]*aux[n](残差;X 可与 C
//                                同缓冲,C 每行先算完再读写,别名安全)
// 两者互斥。
void gemm_kn(const float* A, int M, int K, int N, const float* Bp, float* C,
             const float* bias, bool silu_act = false,
             const float* X = nullptr, const float* aux = nullptr);

// For cache-local window pipelines already inside a worker team.
void gemm_kn_serial(const float* A, int M, int K, int N, const float* Bp, float* C,
                    const float* bias, bool act = false,
                    const float* X = nullptr, const float* aux = nullptr);

// 输出行排列版(qkv 槽序写入,消 gather):输出行 m 写到 C + row_map[m]*N。
// 只支持纯路径(无残差/无 silu;qkv 专用)。
void gemm_kn_rows(const float* A, int M, int K, int N, const float* Bp, float* C,
                  const float* bias, const int* row_map);

// Top-1 MoE: A is grouped by expert, offsets delimit each expert's rows,
// weights are [experts][K][N]. One worker team covers all experts, including
// few-token experts. Optional row_map scatters output directly to field slots.
void gemm_kn_grouped(const float* A, int K, int N, const float* weights, float* C,
                     const int* offsets, int experts, bool act = false,
                     const int* row_map = nullptr);

// silu(v) = v / (1 + exp(-v)),与 teacher/nr_student.py 逐元素同式。
void silu_vec(float* v, size_t n);

// 逐元素 exp(softmax 用);AVX2 路径用 6 阶多项式,相对误差 ~3e-7。
void exp_vec(float* v, size_t n);

// Field passes: SIMD channels, with serial execution for small pyramid levels.
void field_downsample2(const float* x, float* out, int in_w, int in_h,
                       int out_w, int out_h, int channels);
void field_upsample_merge(const float* x, const float* skip, const float* aux0,
                          const float* aux1, float* out, int in_w, int in_h,
                          int out_w, int out_h, int channels);

// ---- int8(任务 8.2):动态 A 量化(per-row scale)+ 权重 per-output-channel
// 量化(加载期一次)。安全内核:i8 经 vpmovsxbw 展开 i16、vpmaddwd 进 i32 累加,
// 全程无饱和。语义:C[m][n] = sA[m]*sW[n] * Σ_k Aq[m][k]*Wq[n][k] + bias[n]。
// Wq: [N][Kp] int8(Kp = K 向上对齐 16,补零;线性层就是 torch 权重布局)。

// f32 权重 [N][K] -> i8 [N][Kp](每行 scale 进 sW[N])。
void quantize_w_nk(const float* w_nk, int N, int K, int Kp, int8_t* out, float* sW);
// f32 权重切片 [K][N](专家 einsum 布局)-> i8 [N][Kp](转置 + 量化)。
void quantize_w_kn(const float* w_kn, int K, int N, int Kp, int8_t* out, float* sW);

// VNNI panel: [ceil(N/16)][Kp/4][16][4], padded columns are zero.
// correction[n] = -128 * sum(Wq[n]), converting u8*s8 back to s8*s8.
// Prepare once at load time; both VNNI ISAs share this layout.
struct I8VnniWeights {
    std::vector<int8_t> w;
    std::vector<int32_t> correction;
    int N = 0, Kp = 0;
};
I8VnniWeights pack_weight_i8_vnni(const int8_t* Wq, int N, int Kp);

// C: [M][N] f32;A 动态量化(per-row);出口融合语义同 gemm_kn。
// vnni is optional; without it the existing AVX2/scalar implementation is used.
void gemm_kn_i8(const float* A, int M, int K, int N, const int8_t* Wq, int Kp,
                const float* sW, float* C, const float* bias, bool silu_act = false,
                const float* X = nullptr, const float* aux = nullptr,
                const I8VnniWeights* vnni = nullptr);

// Runtime-dispatched INT8 backend. VNNI is compiled in a separate translation
// unit so binaries remain safe on AVX2-only machines.
bool gemm_vnni_available();
const char* gemm_i8_backend();
