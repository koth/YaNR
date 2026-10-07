// The student CPU engine (openspec tasks 8.4/8.9). Forward pass mirrors
// teacher/nr_student.py block by block; the block/attention math follows the CUDA
// engine's field-addressed window walk (no gather/scatter).
#pragma once
#include <cstdint>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

#include "geometry.h"

struct BlockCfg {
    std::string kind;              // ffn / expert / split / vit
    bool attn = false;
    int hidden = 0, experts = 0, narrow = 0;
    int branches = 0, bc = 0, mc = 0;
    int window = 0, shift_x = 0, shift_y = 0;
};

struct Tensor {
    std::vector<int> shape;
    const float* data = nullptr;   // points into the mapped bin
};

// 帧间复用暂存(M3 arena):容量只增不减,steady-state 前向零 malloc。
// ffn_block/vit_block 用 tmp/aux/merge/res/qkv/att/proj;forward 用 sk/x 等槽位。
struct Scratch {
    std::vector<float> tmp, aux, merge, res, qkv, att, proj;
    std::vector<float> sk[6], x, pooled, vstate, up;
};

// int8 权重(任务 8.2):[N][Kp] int8 + 每输出通道 scale;专家 3D 张量扁平存
// [e*N][Kp](每专家切片转置成 [N][K] 再量化)。
struct Wq {
    std::vector<int8_t> w;
    std::vector<float> sw;
    int N = 0, Kp = 0;
};

class Engine {
public:
    explicit Engine(const std::string& idx_path);   // loads <idx> + sibling .bin
    ~Engine();

    // features: [full_rows][16] -> head: [full_rows][4]
    void forward(const float* features, float* head);

    // 二分调试:转储各级中间状态(engine_main --dump-dir 写盘,check_engine --bisect 对账)。
    void set_debug(bool on) { debug_on_ = on; }
    const std::map<std::string, std::vector<float>>& debug() const { return debug_; }

    // 粗粒度剖析(M3):gemm / silu / attn / attn-gather 累计毫秒;其余 = 总时间 - 之和。
    void set_profile(bool on) { prof_on_ = on; for (double& v : prof_) v = 0.0; }
    const double* profile() const { return prof_; }

    // int8 GEMM 路径(任务 8.2):动态 A 量化 + per-channel 权重量化。
    void set_int8(bool on) { int8_ = on; }

    const Geometry& geometry() const { return geom_; }
    float blend_scale() const { return blend_scale_; }
    int valid_size() const { return valid_size_; }   // GEOM 的有效边长(composite 口径)
    const std::vector<BlockCfg>& enc_blocks(const std::string& level) const;
    const std::vector<BlockCfg>& dec_blocks(const std::string& level) const;

private:
    const Tensor& t(const std::string& name) const;
    const float* w(const std::string& name) const { return t(name).data; }
    int dim(const std::string& name, int i) const { return t(name).shape[i]; }

    void linear(const float* x, int rows, int K, int N, const std::string& name, float* out) const;
    // X/aux 非空时出口融合残差 out = A*W + bias + X*aux(X 可与 out 同缓冲)。
    // row_map 非空时输出行 m 写到 out + row_map[m]*N(qkv 槽序写入,消 gather)。
    void linear_bias(const float* x, int rows, int K, int N, const std::string& name,
                     float* out, bool silu_act = false,
                     const float* X = nullptr, const float* aux = nullptr,
                     const int* row_map = nullptr) const;
    void ffn_block(const BlockCfg& cfg, const std::string& prefix, float* state,
                   int rows, int width, int height) const;
    void vit_block(const BlockCfg& cfg, const std::string& prefix, float* state,
                   int tokens, int padded) const;
    // qkv 可写:注意力核心直接在行上就地归一(掩码/复刻语义见 attn.h)。
    void window_attention(float* qkv, float* out, int rows, int width, int height,
                          int heads, const BlockCfg& cfg, const std::string& prefix) const;
    void vit_attention(float* qkv, float* out, int tokens, int padded, int heads,
                       const std::string& prefix) const;
    void box_downsample(const float* x, float* out, int in_w, int in_h, int out_w,
                        int out_h, int ch) const;
    void upsample2(const float* x, float* out, int in_w, int in_h, int out_w,
                   int out_h, int ch) const;
    // upsample2 + 跳连残差一趟融合:out = up(x)*(aux0 或 1) + skip*aux1。
    void upsample2_merge(const float* x, const float* skip, const float* aux0,
                         const float* aux1, float* out, int in_w, int in_h, int out_w,
                         int out_h, int ch) const;
    // 窗口槽序排列(几何决定,按 key 缓存):field 行 r 的 qkv 输出行 = perm[r]。
    const std::vector<int>& window_perm(int width, int height, const BlockCfg& cfg) const;

    std::vector<char> bin_;        // whole weight file
    std::unordered_map<std::string, Tensor> tensors_;
    std::unordered_map<std::string, std::vector<float>> packed_;   // [K][N] 打包权重(gemm.h)
    std::unordered_map<std::string, Wq> packed_i8_;                // 线性 [N][Kp] int8
    std::unordered_map<std::string, Wq> expert_i8_;                // 专家 [e*N][Kp] int8
    mutable std::map<uint64_t, std::vector<int>> perm_cache_;      // 窗口槽序排列缓存
    bool int8_ = false;
    mutable Scratch sc_;
    std::unordered_map<std::string, std::vector<BlockCfg>> blocks_[2];   // 0 enc, 1 dec
    std::vector<BlockCfg> vit_blocks_;
    Geometry geom_;
    float blend_scale_ = 1.0f;
    int valid_size_ = 0;
    int vit_tokens_ = 0, vit_padded_ = 0;
    bool debug_on_ = false;
    bool prof_on_ = false;
    mutable double prof_[4] = {0, 0, 0, 0};         // gemm / silu / attn / attn-gather
    mutable std::map<std::string, std::vector<float>> debug_;   // mutable:ffn_block 里也埋点
};
