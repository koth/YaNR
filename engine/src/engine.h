// The student CPU engine (openspec tasks 8.4/8.9). Forward pass mirrors
// teacher/nr_student.py block by block; the block/attention math follows the CUDA
// engine's field-addressed window walk (no gather/scatter).
#pragma once
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

class Engine {
public:
    explicit Engine(const std::string& idx_path);   // loads <idx> + sibling .bin
    ~Engine();

    // features: [full_rows][16] -> head: [full_rows][4]
    void forward(const float* features, float* head);

    // 二分调试:转储各级中间状态(engine_main --dump-dir 写盘,check_engine --bisect 对账)。
    void set_debug(bool on) { debug_on_ = on; }
    const std::map<std::string, std::vector<float>>& debug() const { return debug_; }

    const Geometry& geometry() const { return geom_; }
    float blend_scale() const { return blend_scale_; }
    const std::vector<BlockCfg>& enc_blocks(const std::string& level) const;
    const std::vector<BlockCfg>& dec_blocks(const std::string& level) const;

private:
    const Tensor& t(const std::string& name) const;
    const float* w(const std::string& name) const { return t(name).data; }
    int dim(const std::string& name, int i) const { return t(name).shape[i]; }

    void linear(const float* x, int rows, int K, int N, const std::string& name, float* out) const;
    void linear_bias(const float* x, int rows, int K, int N, const std::string& name, float* out) const;
    void ffn_block(const BlockCfg& cfg, const std::string& prefix, float* state,
                   int rows, int width, int height) const;
    void vit_block(const BlockCfg& cfg, const std::string& prefix, float* state,
                   int tokens, int padded) const;
    void window_attention(const float* qkv, float* out, int rows, int width, int height,
                          int heads, const BlockCfg& cfg, const std::string& prefix) const;
    void vit_attention(const float* qkv, float* out, int tokens, int padded, int heads,
                       const std::string& prefix) const;
    void box_downsample(const float* x, float* out, int in_w, int in_h, int out_w,
                        int out_h, int ch) const;
    void upsample2(const float* x, float* out, int in_w, int in_h, int out_w,
                   int out_h, int ch) const;

    std::vector<char> bin_;        // whole weight file
    std::unordered_map<std::string, Tensor> tensors_;
    std::unordered_map<std::string, std::vector<BlockCfg>> blocks_[2];   // 0 enc, 1 dec
    std::vector<BlockCfg> vit_blocks_;
    Geometry geom_;
    float blend_scale_ = 1.0f;
    int vit_tokens_ = 0, vit_padded_ = 0;
    bool debug_on_ = false;
    mutable std::map<std::string, std::vector<float>> debug_;   // mutable:ffn_block 里也埋点
};
