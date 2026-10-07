#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

constexpr int kHeadDim = 32;

inline float silu(float v) { return v / (1.0f + std::exp(-v)); }

inline float cosine_norm(float x[32]) {
    float s = 0.0f;
    for (int d = 0; d < 32; d++) s += x[d] * x[d];
    float inv = 1.0f / std::fmax(std::sqrt(s), 1e-6f);
    for (int d = 0; d < 32; d++) x[d] *= inv;
    return inv;
}

}  // namespace

Engine::Engine(const std::string& idx_path) {
    std::ifstream in(idx_path);
    if (!in) throw std::runtime_error("cannot open " + idx_path);
    std::string bin_name;
    std::string line;
    struct Raw { std::vector<int> shape; long offset; long bytes; };
    std::vector<std::pair<std::string, Raw>> order;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "BIN") {
            ss >> bin_name;
        } else if (tag == "GEOM") {
            int size, fw, fh;
            ss >> size >> fw >> fh;
            geom_ = geometry_from_valid(size, size);
            (void)fw; (void)fh;
            for (int i = 0; i < 6; i++) { int w, h; ss >> w >> h; (void)w; (void)h; }
            ss >> vit_tokens_ >> vit_padded_;
        } else if (tag == "BLEND_SCALE") {
            ss >> blend_scale_;
        } else if (tag == "T") {
            std::string name;
            int ndim;
            ss >> name >> ndim;
            Raw r;
            r.shape.resize(ndim);
            for (int i = 0; i < ndim; i++) ss >> r.shape[i];
            ss >> r.offset >> r.bytes;
            order.emplace_back(name, r);
        } else if (tag == "BLOCK") {
            std::string key, kind;
            BlockCfg cfg;
            int attn;
            ss >> key >> kind >> attn >> cfg.hidden >> cfg.experts >> cfg.narrow
               >> cfg.branches >> cfg.bc >> cfg.mc >> cfg.window >> cfg.shift_x >> cfg.shift_y;
            cfg.kind = kind;
            cfg.attn = attn != 0;
            int side = key.rfind("vit_blocks", 0) == 0 ? 2
                     : (key.rfind("enc_", 0) == 0 ? 0 : 1);
            if (side == 2) {
                vit_blocks_.push_back(cfg);
            } else {
                // key: enc_blocks.<level>.<i>
                std::string rest = key.substr(key.find('.') + 1);
                std::string level = rest.substr(0, rest.find('.'));
                blocks_[side][level].push_back(cfg);
            }
        }
    }

    std::string dir = idx_path.substr(0, idx_path.find_last_of('/') + 1);
    std::ifstream bin(dir + bin_name, std::ios::binary);
    if (!bin) throw std::runtime_error("cannot open " + dir + bin_name);
    bin_.assign(std::istreambuf_iterator<char>(bin), std::istreambuf_iterator<char>());
    for (auto& kv : order) {
        Tensor t;
        t.shape = kv.second.shape;
        t.data = reinterpret_cast<const float*>(bin_.data() + kv.second.offset);
        tensors_[kv.first] = t;
    }
}

Engine::~Engine() = default;

const Tensor& Engine::t(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("missing tensor " + name);
    return it->second;
}

const std::vector<BlockCfg>& Engine::enc_blocks(const std::string& level) const {
    return blocks_[0].at(level);
}
const std::vector<BlockCfg>& Engine::dec_blocks(const std::string& level) const {
    return blocks_[1].at(level);
}

// out[r*N+n] = bias[n] + sum_k x[r*K+k] * W_t[n*K+k]   (torch Linear: weight [N,K])
void Engine::linear(const float* x, int rows, int K, int N, const std::string& name,
                    float* out) const {
    const float* wt = w(name + ".weight");
    const float* bias = tensors_.count(name + ".bias") ? w(name + ".bias") : nullptr;
    linear_bias(x, rows, K, N, name, out);
    (void)wt; (void)bias;
}

void Engine::linear_bias(const float* x, int rows, int K, int N, const std::string& name,
                         float* out) const {
    const float* wt = w(name + ".weight");
    const float* bias = tensors_.count(name + ".bias") ? w(name + ".bias") : nullptr;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * K;
        float* orow = out + (size_t)r * N;
        for (int n = 0; n < N; n++) {
            const float* wr = wt + (size_t)n * K;
            float acc = bias ? bias[n] : 0.0f;
            int k = 0;
            for (; k + 3 < K; k += 4) {
                acc += xr[k] * wr[k] + xr[k + 1] * wr[k + 1]
                     + xr[k + 2] * wr[k + 2] + xr[k + 3] * wr[k + 3];
            }
            for (; k < K; k++) acc += xr[k] * wr[k];
            orow[n] = acc;
        }
    }
}

void Engine::box_downsample(const float* x, float* out, int in_w, int in_h, int out_w,
                            int out_h, int ch) const {
#pragma omp parallel for schedule(static)
    for (int y = 0; y < out_h; y++) {
        int sy = std::min(2 * y, in_h - 1);
        int sy1 = std::min(sy + 1, in_h - 1);
        bool vy = sy + 1 < in_h;
        for (int xx = 0; xx < out_w; xx++) {
            int sx = std::min(2 * xx, in_w - 1);
            int sx1 = std::min(sx + 1, in_w - 1);
            bool valid = vy && (sx + 1 < in_w);
            const float* a = x + ((size_t)sy * in_w + sx) * ch;
            const float* b = x + ((size_t)sy * in_w + sx1) * ch;
            const float* c = x + ((size_t)sy1 * in_w + sx) * ch;
            const float* d = x + ((size_t)sy1 * in_w + sx1) * ch;
            float* o = out + ((size_t)y * out_w + xx) * ch;
            for (int chn = 0; chn < ch; chn++) {
                o[chn] = valid ? (a[chn] + b[chn] + c[chn] + d[chn]) * 0.25f : 0.0f;
            }
        }
    }
}

void Engine::upsample2(const float* x, float* out, int in_w, int in_h, int out_w,
                       int out_h, int ch) const {
#pragma omp parallel for schedule(static)
    for (int y = 0; y < out_h; y++) {
        int sy = std::min(y >> 1, in_h - 1);
        for (int xx = 0; xx < out_w; xx++) {
            int sx = std::min(xx >> 1, in_w - 1);
            const float* s = x + ((size_t)sy * in_w + sx) * ch;
            float* o = out + ((size_t)y * out_w + xx) * ch;
            for (int chn = 0; chn < ch; chn++) o[chn] = s[chn];
        }
    }
}

void Engine::window_attention(const float* qkv, float* out, int rows, int width, int height,
                              int heads, const BlockCfg& cfg, const std::string& prefix) const {
    const int win = cfg.window;
    const int slots = win * win;
    const float* prior = w(prefix + ".attention.prior");
    const float* scale = w(prefix + ".attention.scale");
    const int windows_x = (width + cfg.shift_x + win - 1) / win;
    const int windows_y = (height + cfg.shift_y + win - 1) / win;
    const int windows = windows_x * windows_y;

#pragma omp parallel
    {
        std::vector<float> q((size_t)slots * 32), k((size_t)slots * 32), v((size_t)slots * 32);
        std::vector<float> scores((size_t)slots * slots);
        std::vector<int> valid(slots), index(slots);
#pragma omp for schedule(static)
        for (int widx = 0; widx < windows; widx++) {
            int wx = widx / windows_y, wy = widx % windows_y;
            int ox = wx * win - cfg.shift_x, oy = wy * win - cfg.shift_y;
            for (int s = 0; s < slots; s++) {
                int fx = ox + (s % win), fy = oy + (s / win);
                valid[s] = (fx >= 0 && fx < width && fy >= 0 && fy < height);
                index[s] = valid[s] ? fy * width + fx : 0;
            }
            for (int h = 0; h < heads; h++) {
                for (int s = 0; s < slots; s++) {
                    const float* row = qkv + (size_t)index[s] * (heads * 96) + h * 96;
                    float* qs = q.data() + (size_t)s * 32;
                    float* ks = k.data() + (size_t)s * 32;
                    float* vs = v.data() + (size_t)s * 32;
                    for (int d = 0; d < 32; d++) {
                        qs[d] = valid[s] ? row[d] : 0.0f;
                        ks[d] = valid[s] ? row[32 + d] : 0.0f;
                        vs[d] = valid[s] ? row[64 + d] : 0.0f;
                    }
                    if (valid[s]) {
                        cosine_norm(qs);
                        cosine_norm(ks);
                        float sc = scale[h];
                        for (int d = 0; d < 32; d++) qs[d] *= sc;
                    }
                }
                const float* pr = prior + (size_t)h * slots * slots;
                for (int s = 0; s < slots; s++) {
                    if (!valid[s]) continue;                   // 查询维掩码:非法查询行整体丢弃
                    const float* qs = q.data() + (size_t)s * 32;
                    float mx = -INFINITY;
                    for (int j = 0; j < slots; j++) {
                        const float* ks = k.data() + (size_t)j * 32;
                        float acc = 0.0f;
                        for (int d = 0; d < 32; d++) acc += qs[d] * ks[d];
                        acc += pr[s * slots + j];
                        scores[(size_t)s * slots + j] = acc;   // 注意:非法键不掩码 —— 学生
                        if (acc > mx) mx = acc;                // (torch)的掩码在查询维;非法键
                    }                                          // 带 prior 分吸概率、值为 0
                    float den = 0.0f;
                    for (int j = 0; j < slots; j++) {
                        float e = std::exp(scores[(size_t)s * slots + j] - mx);
                        scores[(size_t)s * slots + j] = e;
                        den += e;
                    }
                    float inv = den > 0.0f ? 1.0f / den : 0.0f;
                    float* orow = out + (size_t)index[s] * (heads * 32) + h * 32;
                    for (int d = 0; d < 32; d++) orow[d] = 0.0f;
                    for (int j = 0; j < slots; j++) {
                        float a = scores[(size_t)s * slots + j] * inv;
                        if (a == 0.0f) continue;
                        const float* vs = v.data() + (size_t)j * 32;
                        for (int d = 0; d < 32; d++) orow[d] += a * vs[d];
                    }
                }
            }
        }
    }
    (void)rows;
}

void Engine::vit_attention(const float* qkv, float* out, int tokens, int padded, int heads,
                           const std::string& prefix) const {
    const float* scale = w(prefix + ".attention.scale");
    const float kSqrt = std::sqrt(32.0f);
#pragma omp parallel
    {
        std::vector<float> k((size_t)padded * 32), v((size_t)padded * 32);
        std::vector<float> scores((size_t)tokens * padded);
#pragma omp for schedule(static)
        for (int h = 0; h < heads; h++) {
            std::fill(k.begin(), k.end(), 0.0f);
            std::fill(v.begin(), v.end(), 0.0f);
            for (int ti = 0; ti < tokens; ti++) {
                const float* row = qkv + (size_t)ti * (heads * 96) + h * 96;
                for (int d = 0; d < 32; d++) {
                    k[(size_t)ti * 32 + d] = row[32 + d];
                    v[(size_t)ti * 32 + d] = row[64 + d];
                }
            }
            for (int ti = 0; ti < tokens; ti++) {
                float q[32];
                const float* row = qkv + (size_t)ti * (heads * 96) + h * 96;
                for (int d = 0; d < 32; d++) q[d] = row[d];
                cosine_norm(q);
                float sc = scale[h] * kSqrt;
                for (int d = 0; d < 32; d++) q[d] *= sc;
                float ks[32];
                for (int t2 = 0; t2 < tokens; t2++) {
                    for (int d = 0; d < 32; d++) ks[d] = k[(size_t)t2 * 32 + d];
                    cosine_norm(ks);
                    float acc = 0.0f;
                    for (int d = 0; d < 32; d++) acc += q[d] * ks[d];
                    scores[(size_t)ti * padded + t2] = acc;
                }
                float mx = -INFINITY;
                for (int t2 = 0; t2 < tokens; t2++) {
                    float s = scores[(size_t)ti * padded + t2];
                    if (s > mx) mx = s;
                }
                float den = 0.0f;
                for (int p = 0; p < padded; p++) {
                    float e = 0.0f;
                    if (p < tokens) {
                        e = std::exp(scores[(size_t)ti * padded + p] - mx);
                        den += e;
                    }
                    scores[(size_t)ti * padded + p] = e;
                }
                float inv = den > 0.0f ? 1.0f / den : 0.0f;
                float* orow = out + (size_t)ti * (heads * 32) + h * 32;
                for (int d = 0; d < 32; d++) orow[d] = 0.0f;
                for (int p = 0; p < tokens; p++) {
                    float a = scores[(size_t)ti * padded + p] * inv;
                    if (a == 0.0f) continue;
                    for (int d = 0; d < 32; d++) orow[d] += a * v[(size_t)p * 32 + d];
                }
            }
        }
    }
}

void Engine::ffn_block(const BlockCfg& cfg, const std::string& prefix, float* state,
                       int rows, int width, int height) const {
    int ch = dim(prefix + ".contract.weight", 0);
    std::vector<float> tmp((size_t)rows * std::max(cfg.hidden, 1));
    std::vector<float> merged((size_t)rows * ch);
    std::vector<float> out((size_t)rows * ch);

    if (cfg.kind == "ffn") {
        int h = cfg.hidden;
        linear_bias(state, rows, ch, h, prefix + ".expand", tmp.data());
        for (size_t i = 0; i < tmp.size(); i++) tmp[i] = silu(tmp[i]);
        linear_bias(tmp.data(), rows, h, ch, prefix + ".contract", merged.data());
    } else if (cfg.kind == "expert") {
        int e = cfg.experts, h = cfg.hidden, nc = cfg.narrow;
        // torch: ffn = silu(einsum('rc,ech->erh'))     -> [e][r][h]
        //        merged = einsum('erh,ehc->erc').reshape(rows, -1)
        // reshape 之后 [e][r][c] 的扁平缓冲被直接当作 [rows][ch] 用 —— 这个 e-外层的
        // 置乱是训练语义的一部分,C++ 必须逐位复刻(写成自然序会偏 ~1%)。
        std::vector<float> ffn((size_t)e * rows * h);
        const float* we = w(prefix + ".expand");
        const float* wn = w(prefix + ".narrow");
        // 张量布局同 torch einsum 约定:expand[e][c][h](in 外 out 内)、
        // narrow[e][h][c] —— 外积式累加,内层连续。
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            for (int ei = 0; ei < e; ei++) {
                float* dst = ffn.data() + ((size_t)ei * rows + r) * h;
                for (int n = 0; n < h; n++) dst[n] = 0.0f;
                for (int c = 0; c < ch; c++) {
                    float xv = state[(size_t)r * ch + c];
                    const float* wcol = we + ((size_t)ei * ch + c) * h;
                    for (int n = 0; n < h; n++) dst[n] += xv * wcol[n];
                }
                for (int n = 0; n < h; n++) dst[n] = silu(dst[n]);
            }
        }
        merged.resize((size_t)e * rows * nc);
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            for (int ei = 0; ei < e; ei++) {
                const float* src = ffn.data() + ((size_t)ei * rows + r) * h;
                float* dst = merged.data() + ((size_t)ei * rows + r) * nc;
                for (int n = 0; n < nc; n++) dst[n] = 0.0f;
                for (int c = 0; c < h; c++) {
                    const float* wrow = wn + ((size_t)ei * h + c) * nc;
                    float xv = src[c];
                    for (int n = 0; n < nc; n++) dst[n] += xv * wrow[n];
                }
            }
        }
        linear_bias(merged.data(), rows, ch, ch, prefix + ".contract", out.data());
        for (int r = 0; r < rows; r++) {
            const float* xr = state + (size_t)r * ch;
            float* orow = out.data() + (size_t)r * ch;
            const float* aux = w(prefix + ".aux_ffn");
            for (int c = 0; c < ch; c++) orow[c] += xr[c] * aux[c];
        }
        std::swap(merged, out);
    } else {  // split
        int e = cfg.branches, bc = cfg.bc, mc = cfg.mc;
        std::vector<float> bvec((size_t)rows * ch);
        linear_bias(state, rows, ch, ch, prefix + ".branch", bvec.data());
        const float* w2 = w(prefix + ".w2");
        const float* w3 = w(prefix + ".w3");
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            float* dst = merged.data() + (size_t)r * ch;
            const float* br = bvec.data() + (size_t)r * ch;
            for (int ei = 0; ei < e; ei++) {
                const float* b = br + ei * bc;
                float mid[512];
                const float* w2e = w2 + (size_t)ei * bc * mc;
                for (int m = 0; m < mc; m++) mid[m] = 0.0f;
                for (int c = 0; c < bc; c++) {                  // w2[e][c][m] 外积式
                    const float* wrow = w2e + (size_t)c * mc;
                    float xv = b[c];
                    for (int m = 0; m < mc; m++) mid[m] += xv * wrow[m];
                }
                for (int m = 0; m < mc; m++) mid[m] = silu(mid[m]);
                const float* w3e = w3 + (size_t)ei * mc * bc;
                float* d = dst + ei * bc;
                for (int c = 0; c < bc; c++) d[c] = 0.0f;
                for (int m = 0; m < mc; m++) {                  // w3[e][m][c]
                    const float* wrow = w3e + (size_t)m * bc;
                    float xv = mid[m];
                    for (int c = 0; c < bc; c++) d[c] += xv * wrow[c];
                }
            }
        }
        linear_bias(merged.data(), rows, ch, ch, prefix + ".contract", out.data());
        const float* aux = w(prefix + ".aux_ffn");
        for (int r = 0; r < rows; r++) {
            const float* xr = state + (size_t)r * ch;
            float* orow = out.data() + (size_t)r * ch;
            for (int c = 0; c < ch; c++) orow[c] += xr[c] * aux[c];
        }
        std::swap(merged, out);
    }

    if (cfg.kind == "ffn") {
        const float* aux = w(prefix + ".aux_ffn");
        for (int r = 0; r < rows; r++) {
            const float* xr = state + (size_t)r * ch;
            float* orow = merged.data() + (size_t)r * ch;
            for (int c = 0; c < ch; c++) orow[c] += xr[c] * aux[c];
        }
    }

    if (cfg.attn) {
        if (debug_on_) debug_["ffn-" + prefix] = merged;
        int heads = ch / kHeadDim;
        std::vector<float> qkv((size_t)rows * ch * 3);
        std::vector<float> att((size_t)rows * ch);
        std::vector<float> proj((size_t)rows * ch);
        linear_bias(merged.data(), rows, ch, ch * 3, prefix + ".tail.qkv", qkv.data());
        window_attention(qkv.data(), att.data(), rows, width, height, heads, cfg, prefix + ".tail");
        if (debug_on_) debug_["att-" + prefix] = att;
        linear_bias(att.data(), rows, ch, ch, prefix + ".tail.proj", proj.data());
        const float* aux = w(prefix + ".tail.aux_attn");
        for (int r = 0; r < rows; r++) {
            const float* xr = merged.data() + (size_t)r * ch;
            float* orow = proj.data() + (size_t)r * ch;
            for (int c = 0; c < ch; c++) orow[c] = xr[c] + orow[c] * aux[c];
        }
        std::memcpy(state, proj.data(), (size_t)rows * ch * sizeof(float));
    } else {
        std::memcpy(state, merged.data(), (size_t)rows * ch * sizeof(float));
    }
    if (debug_on_) debug_["blk-" + prefix] = std::vector<float>(state, state + (size_t)rows * ch);
}

void Engine::vit_block(const BlockCfg& cfg, const std::string& prefix, float* state,
                       int tokens, int padded) const {
    int ch = dim(prefix + ".contract.weight", 0);
    int ffn = cfg.hidden ? cfg.hidden : dim(prefix + ".expand.weight", 0);
    std::vector<float> tmp((size_t)tokens * ffn);
    std::vector<float> y((size_t)tokens * ch);
    linear_bias(state, tokens, ch, ffn, prefix + ".expand", tmp.data());
    for (size_t i = 0; i < tmp.size(); i++) tmp[i] = silu(tmp[i]);
    linear_bias(tmp.data(), tokens, ffn, ch, prefix + ".contract", y.data());
    const float* aux = w(prefix + ".aux_ffn");
    for (int r = 0; r < tokens; r++) {
        const float* xr = state + (size_t)r * ch;
        float* orow = y.data() + (size_t)r * ch;
        for (int c = 0; c < ch; c++) orow[c] += xr[c] * aux[c];
    }
    std::vector<float> qkv((size_t)tokens * ch * 3), att((size_t)tokens * ch),
        proj((size_t)tokens * ch);
    linear_bias(y.data(), tokens, ch, ch * 3, prefix + ".qkv", qkv.data());
    vit_attention(qkv.data(), att.data(), tokens, padded, ch / kHeadDim, prefix);
    if (debug_on_) debug_["att-" + prefix] = att;
    linear_bias(att.data(), tokens, ch, ch, prefix + ".proj", proj.data());
    const float* aux2 = w(prefix + ".aux_attn");
    for (int r = 0; r < tokens; r++) {
        const float* xr = y.data() + (size_t)r * ch;
        float* orow = proj.data() + (size_t)r * ch;
        for (int c = 0; c < ch; c++) orow[c] = xr[c] + orow[c] * aux2[c];
    }
    std::memcpy(state, proj.data(), (size_t)tokens * ch * sizeof(float));
    if (debug_on_) debug_["blk-" + prefix] = std::vector<float>(state, state + (size_t)tokens * ch);
}

void Engine::forward(const float* features, float* head) {
    const char* order[6] = {"full", "d0", "d1", "d2", "d3", "d4"};
    int ch_full = dim("adapter.weight", 0);
    int dims[6][2], rows[6], chs[6];
    for (int i = 0; i < 6; i++) {
        if (i == 0) { dims[0][0] = geom_.full_width; dims[0][1] = geom_.full_height; }
        else { dims[i][0] = geom_.levels[i - 1].width; dims[i][1] = geom_.levels[i - 1].height; }
        rows[i] = dims[i][0] * dims[i][1];
    }
    for (int i = 0; i < 6; i++) {
        const auto& bl = blocks_[0].at(order[i]);
        std::string p = std::string("enc_blocks.") + order[i] + ".0";
        (void)bl;
        chs[i] = t(p + ".contract.weight").shape[0];
    }

    std::vector<float> state[6], skip[6];
    for (int i = 0; i < 6; i++) {
        state[i].resize((size_t)rows[i] * chs[i]);
        skip[i].resize((size_t)rows[i] * chs[i]);
    }
    std::vector<float> x((size_t)rows[0] * ch_full);
    linear_bias(features, rows[0], 16, ch_full, "adapter", x.data());

    for (int i = 0; i < 6; i++) {
        int ch = chs[i];
        std::vector<float> cur = x;
        cur.resize((size_t)rows[i] * ch);
        const auto& bl = blocks_[0].at(order[i]);
        for (size_t j = 0; j < bl.size(); j++) {
            std::string p = std::string("enc_blocks.") + order[i] + "." + std::to_string(j);
            ffn_block(bl[j], p, cur.data(), rows[i], dims[i][0], dims[i][1]);
        }
        skip[i] = cur;
        if (debug_on_) debug_["s-enc-" + std::string(order[i])] = cur;
        if (i < 5) {
            std::vector<float> pooled((size_t)rows[i + 1] * ch);
            box_downsample(cur.data(), pooled.data(), dims[i][0], dims[i][1],
                           dims[i + 1][0], dims[i + 1][1], ch);
            std::string tn = std::string("enc_trans.") + order[i + 1];
            x.assign((size_t)rows[i + 1] * chs[i + 1], 0.0f);
            linear_bias(pooled.data(), rows[i + 1], ch, chs[i + 1], tn, x.data());
            if (debug_on_) debug_["trans-" + std::string(order[i + 1])] = x;
        }
    }

    // ---- ViT
    int d5w = geom_.levels[5].width, d5h = geom_.levels[5].height;
    int rows5 = d5w * d5h;
    int ch4 = chs[5];
    int vit_ch = dim("vit_in.weight", 0);
    std::vector<float> pooled((size_t)rows5 * ch4);
    box_downsample(skip[5].data(), pooled.data(), dims[5][0], dims[5][1], d5w, d5h, ch4);
    std::vector<float> vstate((size_t)rows5 * vit_ch);
    linear_bias(pooled.data(), rows5, ch4, vit_ch, "vit_in", vstate.data());
    if (debug_on_) debug_["trans-vit"] = vstate;
    for (size_t j = 0; j < vit_blocks_.size(); j++) {
        std::string p = "vit_blocks." + std::to_string(j);
        vit_block(vit_blocks_[j], p, vstate.data(), vit_tokens_, vit_padded_);
    }
    if (debug_on_) debug_["s-vit"] = vstate;
    x.assign((size_t)rows5 * ch4, 0.0f);
    linear_bias(vstate.data(), rows5, vit_ch, ch4, "vit_out", x.data());

    // ---- 解码:d5 -> d4 -> d3 -> d2 -> d1 -> d0 -> full
    const char* dec[5] = {"d4", "d3", "d2", "d1", "d0"};
    int prev_w = d5w, prev_h = d5h, prev_rows = rows5;
    for (int i = 0; i < 5; i++) {
        const char* name = dec[i];
        int idx = 5 - i;                                  // d4 -> 5, d3 -> 4 ...
        int lw = dims[idx][0], lh = dims[idx][1], r = rows[idx], ch = chs[idx];
        if (i > 0) {
            std::string tn = std::string("dec_trans.") + name;
            std::vector<float> proj((size_t)r * ch, 0.0f);
            linear_bias(x.data(), r, chs[idx + 1], ch, tn, proj.data());
            x = std::move(proj);
        }
        std::vector<float> up((size_t)r * ch);
        upsample2(x.data(), up.data(), prev_w, prev_h, lw, lh, ch);
        const float* aux = w(std::string("merge.") + name + ".aux");
        for (int rr = 0; rr < r; rr++) {
            const float* sk = skip[idx].data() + (size_t)rr * ch;
            float* o = up.data() + (size_t)rr * ch;
            for (int c = 0; c < ch; c++) o[c] += sk[c] * aux[c];
        }
        x = std::move(up);
        const auto& bl = blocks_[1].at(name);
        for (size_t j = 0; j < bl.size(); j++) {
            std::string p = std::string("dec_blocks.") + name + "." + std::to_string(j);
            ffn_block(bl[j], p, x.data(), r, lw, lh);
        }
        if (debug_on_) debug_["s-dec-" + std::string(name)] = x;
        prev_w = lw; prev_h = lh; prev_rows = r;
    }

    // ---- post_blend + full 级解码块 + head
    {
        int lw = dims[0][0], lh = dims[0][1], r = rows[0], ch = chs[0];
        std::vector<float> up((size_t)r * ch);
        upsample2(x.data(), up.data(), prev_w, prev_h, lw, lh, ch);
        const float* pair = w("post.aux_pair");               // [2, ch]
        for (int rr = 0; rr < r; rr++) {
            const float* sk = skip[0].data() + (size_t)rr * ch;
            float* o = up.data() + (size_t)rr * ch;
            for (int c = 0; c < ch; c++) o[c] = o[c] * pair[c] + sk[c] * pair[ch + c];
        }
        const auto& bl = blocks_[1].at("full");
        for (size_t j = 0; j < bl.size(); j++) {
            std::string p = "dec_blocks.full." + std::to_string(j);
            ffn_block(bl[j], p, up.data(), r, lw, lh);
        }
        linear_bias(up.data(), r, ch, 4, "head", head);
    }
    (void)prev_rows;
}
