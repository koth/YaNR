#include "engine.h"

#include "attn.h"
#include "gemm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace {

constexpr int kHeadDim = 32;

inline float silu(float v) { return v / (1.0f + std::exp(-v)); }

struct MoeRoute {
    std::vector<int> assign;
    std::vector<float> gate;
};

// Match nr_student.py::_balanced_pick: initial argmax, then move the weakest
// margins from overloaded experts into their underloaded quota seats.
void balance_moe(const std::vector<float>& logits, int items, int experts,
                 std::vector<int>& pick) {
    pick.resize(items);
    std::vector<int> count(experts, 0);
    for (int i = 0; i < items; ++i) {
        int best = 0;
        for (int e = 1; e < experts; ++e)
            if (logits[(size_t)i * experts + e] > logits[(size_t)i * experts + best]) best = e;
        pick[i] = best; ++count[best];
    }
    const int quota = (items + experts - 1) / experts;
    std::vector<int> under;
    for (int e = 0; e < experts; ++e) if (count[e] < quota) {
        for (int n = count[e]; n < quota; ++n) under.push_back(e);
    }
    std::vector<std::pair<float, int>> victims;
    for (int e = 0; e < experts; ++e) if (count[e] > quota) {
        std::vector<std::pair<float, int>> mine;
        for (int i = 0; i < items; ++i) if (pick[i] == e) {
            float alt = -INFINITY;
            for (int a = 0; a < experts; ++a) if (a != e)
                alt = std::max(alt, logits[(size_t)i * experts + a]);
            mine.emplace_back(logits[(size_t)i * experts + e] - alt, i);
        }
        std::stable_sort(mine.begin(), mine.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        for (int n = 0; n < count[e] - quota; ++n) victims.push_back(mine[n]);
    }
    std::stable_sort(victims.begin(), victims.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    const int moves = std::min(static_cast<int>(victims.size()), static_cast<int>(under.size()));
    for (int n = 0; n < moves; ++n) pick[victims[n].second] = under[n];
}

MoeRoute route_moe(const float* state, int rows, int width, int height, int channels,
                   int experts, const float* gamma, const float* beta,
                   const float* router_weight, bool windows) {
    const int nw = windows ? (width + 7) / 8 : 0;
    const int nh = windows ? (height + 7) / 8 : 0;
    const int items = windows ? nw * nh : rows;
    std::vector<float> logits(static_cast<size_t>(items) * experts);
    std::vector<float> pooled(channels);
    for (int item = 0; item < items; ++item) {
        if (windows) {
            std::fill(pooled.begin(), pooled.end(), 0.0f);
            int count = 0;
            const int wy = item / nw, wx = item % nw;
            for (int y = wy * 8; y < std::min(height, wy * 8 + 8); ++y)
                for (int x = wx * 8; x < std::min(width, wx * 8 + 8); ++x) {
                    const float* row = state + (static_cast<size_t>(y) * width + x) * channels;
                    for (int c = 0; c < channels; ++c) pooled[c] += row[c];
                    ++count;
                }
            const float inv_count = 1.0f / std::max(1, count);
            for (float& v : pooled) v *= inv_count;
        } else {
            const float* row = state + static_cast<size_t>(item) * channels;
            std::copy(row, row + channels, pooled.begin());
        }
        float mean = 0.0f;
        for (float v : pooled) mean += v;
        mean /= channels;
        float var = 0.0f;
        for (float v : pooled) { float d = v - mean; var += d * d; }
        const float inv = 1.0f / std::sqrt(var / channels + 1e-5f);
        for (int e = 0; e < experts; ++e) {
            const float* wr = router_weight + static_cast<size_t>(e) * channels;
            float score = 0.0f;
            for (int c = 0; c < channels; ++c)
                score += ((pooled[c] - mean) * inv * gamma[c] + beta[c]) * wr[c];
            logits[(size_t)item * experts + e] = score;
        }
    }
    std::vector<int> item_pick;
    balance_moe(logits, items, experts, item_pick);
    MoeRoute result;
    result.assign.resize(rows);
    result.gate.resize(rows);
    std::vector<float> probs(experts);
    for (int item = 0; item < items; ++item) {
        float mx = -INFINITY, den = 0.0f;
        for (int e = 0; e < experts; ++e) mx = std::max(mx, logits[(size_t)item * experts + e]);
        for (int e = 0; e < experts; ++e) { probs[e] = std::exp(logits[(size_t)item * experts + e] - mx); den += probs[e]; }
        const float gate = experts * probs[item_pick[item]] / std::max(den, 1e-9f);
        if (windows) {
            const int wy = item / nw, wx = item % nw;
            for (int y = wy * 8; y < std::min(height, wy * 8 + 8); ++y)
                for (int x = wx * 8; x < std::min(width, wx * 8 + 8); ++x) {
                    const int row = y * width + x;
                    result.assign[row] = item_pick[item]; result.gate[row] = gate;
                }
        } else {
            result.assign[item] = item_pick[item]; result.gate[item] = gate;
        }
    }
    return result;
}

// 粗粒度剖析计时器:acc 为空时近零开销(prof_on_ 门控)。
struct ProfTimer {
    double* acc;
    double* layer;
    std::chrono::steady_clock::time_point t0;
    explicit ProfTimer(double* a, double* l = nullptr) : acc(a), layer(l) {
        if (acc || layer) t0 = std::chrono::steady_clock::now();
    }
    ~ProfTimer() {
        if (acc || layer) {
            double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
            if (acc) *acc += ms;
            if (layer) *layer += ms;
        }
    }
};

}  // namespace

Engine::Engine(const std::string& idx_path, bool cpu_pack) {
    std::ifstream in(std::filesystem::u8path(idx_path));
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
            valid_size_ = size;
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

    // std::filesystem handles both '/' and '\\' on Windows.  The old string
    // split only recognized '/', so an absolute Windows idx path silently
    // looked for the sibling bin in the current working directory.
    const std::filesystem::path idx_fs = std::filesystem::u8path(idx_path);
    const std::filesystem::path bin_path = idx_fs.parent_path() / bin_name;
    std::ifstream bin(bin_path, std::ios::binary);
    if (!bin) throw std::runtime_error("cannot open " + bin_path.string());
    bin_.assign(std::istreambuf_iterator<char>(bin), std::istreambuf_iterator<char>());
    for (auto& kv : order) {
        Tensor t;
        t.shape = kv.second.shape;
        t.data = reinterpret_cast<const float*>(bin_.data() + kv.second.offset);
        tensors_[kv.first] = t;
    }
    if (cpu_pack) pack_weights();
}

void Engine::pack_weights() {
    // 打包 [N][K] Linear 权重 -> [K][N](GEMM 微内核的 B 布局);专家 einsum 切片
    // (expand[e][ch][h] / narrow[e][h][nc])本身就是 [K][N],直接切指针用。
    for (const auto& kv : tensors_) {
        const std::string& name = kv.first;
        const auto& sh = kv.second.shape;
        if (sh.size() != 2 || name.size() < 7) continue;
        if (name.compare(name.size() - 7, 7, ".weight") != 0) continue;
        int N = sh[0], K = sh[1];
        std::vector<float>& pk = packed_[name];
        pk.resize((size_t)N * K);
        pack_weight_kn(t(name).data, N, K, pk.data());
    }
    // int8 权重(任务 8.2):线性 [N][K] -> [N][Kp](Kp 对齐 16);专家 3D [e][K][N]
    // 每专家切片转置量化成 [N][Kp]。逐输出通道 scale,加载期一次。
    for (const auto& kv : tensors_) {
        const std::string& name = kv.first;
        const auto& sh = kv.second.shape;
        if (sh.size() == 2 && name.size() >= 7 &&
            name.compare(name.size() - 7, 7, ".weight") == 0) {
            int N = sh[0], K = sh[1], Kp = (K + 15) & ~15;
            Wq& q = packed_i8_[name];
            q.N = N; q.Kp = Kp;
            q.w.resize((size_t)N * Kp);
            q.sw.resize(N);
            quantize_w_nk(t(name).data, N, K, Kp, q.w.data(), q.sw.data());
            if (gemm_vnni_available()) q.vnni.push_back(pack_weight_i8_vnni(q.w.data(), N, Kp));
        } else if (sh.size() == 3 && name.size() >= 7 &&
                   (name.compare(name.size() - 7, 7, ".expand") == 0 ||
                    name.compare(name.size() - 7, 7, ".narrow") == 0 ||
                    name.compare(name.size() - 4, 4, ".qkv") == 0 ||
                    name.compare(name.size() - 5, 5, ".proj") == 0 ||
                    name.compare(name.size() - 8, 8, ".contract") == 0)) {
            int E = sh[0], K = sh[1], N = sh[2], Kp = (K + 15) & ~15;
            Wq& q = expert_i8_[name];
            q.N = E * N; q.Kp = Kp;
            q.w.resize((size_t)E * N * Kp);
            q.sw.resize((size_t)E * N);
            const bool vnni = gemm_vnni_available();
            for (int e = 0; e < E; e++) {
                quantize_w_kn(t(name).data + (size_t)e * K * N, K, N, Kp,
                              q.w.data() + (size_t)e * N * Kp,
                              q.sw.data() + (size_t)e * N);
                if (vnni) q.vnni.push_back(pack_weight_i8_vnni(q.w.data() + (size_t)e * N * Kp, N, Kp));
            }
        }
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

// out[r*N+n] = bias[n] + sum_k x[r*K+k] * W_t[n*K+k]   (torch Linear: weight [N,K])
// 实际走打包布局 Bp[k][n] = W_t[n][k](ctor 期一次性转置),见 gemm.h。
// int8 只吃大 K(任务 8.2 实测,K=1152..294912 行全形状微基准):K>=256 时
// i8 内核 2-3x 于 fp32;K=128 打平;K<=64 因 hsum/量化开销 3-5x 慢,留给
// fp32 微内核(小 K 上它已到 300+ GMAC/s)。
constexpr int kInt8MinK = 256;

void Engine::linear_bias(const float* x, int rows, int K, int N, const std::string& name,
                         float* out, bool silu_act, const float* X, const float* aux,
                         const int* row_map) const {
    ProfTimer tm(prof_on_ ? &prof_[0] : nullptr, prof_on_ ? &layer_prof_[name] : nullptr);
    const float* bias = tensors_.count(name + ".bias") ? w(name + ".bias") : nullptr;
    if (int8_ && K >= kInt8MinK && !row_map) {
        auto it = packed_i8_.find(name + ".weight");
        if (it == packed_i8_.end()) throw std::runtime_error("missing int8 weight " + name);
        gemm_kn_i8(x, rows, K, N, it->second.w.data(), it->second.Kp,
                   it->second.sw.data(), out, bias, silu_act, X, aux,
                   it->second.vnni.empty() ? nullptr : &it->second.vnni[0]);
        return;
    }
    auto it = packed_.find(name + ".weight");
    if (it == packed_.end()) throw std::runtime_error("missing packed weight " + name);
    if (row_map) {
        gemm_kn_rows(x, rows, K, N, it->second.data(), out, bias, row_map);
    } else {
        gemm_kn(x, rows, K, N, it->second.data(), out, bias, silu_act, X, aux);
    }
}

void Engine::box_downsample(const float* x, float* out, int in_w, int in_h, int out_w,
                            int out_h, int ch) const {
    ProfTimer tm(nullptr, prof_on_ ? &layer_prof_["pool:" + std::to_string(in_w) + "x" + std::to_string(in_h)] : nullptr);
    field_downsample2(x, out, in_w, in_h, out_w, out_h, ch);
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

void Engine::upsample2_merge(const float* x, const float* skip, const float* aux0,
                             const float* aux1, float* out, int in_w, int in_h, int out_w,
                             int out_h, int ch) const {
    ProfTimer tm(nullptr, prof_on_ ? &layer_prof_["merge:" + std::to_string(out_w) + "x" + std::to_string(out_h)] : nullptr);
    field_upsample_merge(x, skip, aux0, aux1, out, in_w, in_h, out_w, out_h, ch);
}

// 窗口槽序排列:field 行 r 的 qkv 输出行 = perm[r] = widx*slots + s。与
// window_attention 的窗口游走同公式(widx = wx*windows_y + wy),两边必须一致。
const std::vector<int>& Engine::window_perm(int width, int height, const BlockCfg& cfg) const {
    const int win = cfg.window;
    const int slots = win * win;
    const int wx = (width + cfg.shift_x + win - 1) / win;
    const int wy = (height + cfg.shift_y + win - 1) / win;
    uint64_t key = (uint64_t)width | (uint64_t)height << 16 | (uint64_t)win << 32
                 | (uint64_t)(cfg.shift_x + 8) << 40 | (uint64_t)(cfg.shift_y + 8) << 48;
    auto it = perm_cache_.find(key);
    if (it != perm_cache_.end()) return it->second;
    std::vector<int> perm((size_t)wx * wy * slots, -1);
    for (int widx = 0; widx < wx * wy; widx++) {
        int mx = widx / wy, my = widx % wy;
        int ox = mx * win - cfg.shift_x, oy = my * win - cfg.shift_y;
        for (int s = 0; s < slots; s++) {
            int fx = ox + (s % win), fy = oy + (s / win);
            if (fx >= 0 && fx < width && fy >= 0 && fy < height) {
                perm[(size_t)fy * width + fx] = widx * slots + s;
            }
        }
    }
    return perm_cache_.emplace(key, std::move(perm)).first->second;
}

void Engine::moe_block(const BlockCfg& cfg, const std::string& prefix, float* state,
                       int rows, int width, int height) const {
    const int channels = dim(prefix + ".expand", 1);
    const int hidden = dim(prefix + ".expand", 2);
    const int experts = cfg.experts;
    const float* gamma = w(prefix + ".router.0.weight");
    const float* beta = w(prefix + ".router.0.bias");
    const float* router = w(prefix + ".router.1.weight");
    const MoeRoute route = route_moe(state, rows, width, height, channels, experts,
                                     gamma, beta, router, true);
    // Keep all rows for one expert contiguous.  The old implementation
    // copied every expert input and output at each of expand/narrow/qkv/proj;
    // this layout lets the first three operators reuse one packed ordering.
    std::vector<int> group_offsets(experts + 1, 0), group_rows(rows);
    for (int r = 0; r < rows; ++r) ++group_offsets[route.assign[r] + 1];
    for (int e = 0; e < experts; ++e) group_offsets[e + 1] += group_offsets[e];
    std::vector<int> cursor = group_offsets;
    for (int r = 0; r < rows; ++r) group_rows[cursor[route.assign[r]]++] = r;

    auto expert_linear = [&](const std::string& name, int expert, const float* input,
                             int count, int K, int N, float* output, bool act) {
        auto qi = expert_i8_.find(name);
        const bool use_i8 = int8_ && K >= kInt8MinK && qi != expert_i8_.end();
        if (use_i8) {
            const Wq& q = qi->second;
            const size_t offset = static_cast<size_t>(expert) * N * q.Kp;
            gemm_kn_i8(input, count, K, N, q.w.data() + offset, q.Kp,
                       q.sw.data() + static_cast<size_t>(expert) * N,
                       output, nullptr, act, nullptr, nullptr,
                       q.vnni.empty() ? nullptr : &q.vnni[expert]);
        } else {
            const float* weights = t(name).data + static_cast<size_t>(expert) * K * N;
            gemm_kn(input, count, K, N, weights, output, nullptr, act);
        }
    };

    FloatBuffer& input_buf = sc_.tmp;
    FloatBuffer& hidden_buf = sc_.aux;
    FloatBuffer& ffn_buf = sc_.merge;
    FloatBuffer& residual_buf = sc_.res;
    input_buf.resize(static_cast<size_t>(rows) * channels);
    hidden_buf.resize(static_cast<size_t>(rows) * hidden);
    ffn_buf.resize(static_cast<size_t>(rows) * channels);
    residual_buf.resize(static_cast<size_t>(rows) * channels);
    float* grouped_input = input_buf.data();
    float* grouped_hidden = hidden_buf.data();
    float* grouped_ffn = ffn_buf.data();
    float* grouped_residual = residual_buf.data();
    for (int i = 0; i < rows; ++i)
        std::memcpy(grouped_input + static_cast<size_t>(i) * channels,
                    state + static_cast<size_t>(group_rows[i]) * channels,
                    static_cast<size_t>(channels) * sizeof(float));
    const float* aux_ffn = w(prefix + ".aux_ffn");
    if (!int8_) {
        ProfTimer tm(prof_on_ ? &prof_[0] : nullptr,
                     prof_on_ ? &layer_prof_[prefix + ".ffn"] : nullptr);
        gemm_kn_grouped(grouped_input, channels, hidden, w(prefix + ".expand"), grouped_hidden,
                        group_offsets.data(), experts, true);
        gemm_kn_grouped(grouped_hidden, hidden, channels, w(prefix + ".narrow"), grouped_ffn,
                        group_offsets.data(), experts);
    }
    for (int e = 0; e < experts; ++e) {
        const int start = group_offsets[e];
        const int count = group_offsets[e + 1] - start;
        if (!count) continue;
        if (int8_) {
        expert_linear(prefix + ".expand", e, grouped_input + static_cast<size_t>(start) * channels,
                      count, channels, hidden, grouped_hidden + static_cast<size_t>(start) * hidden, true);
        expert_linear(prefix + ".narrow", e, grouped_hidden + static_cast<size_t>(start) * hidden,
                      count, hidden, channels, grouped_ffn + static_cast<size_t>(start) * channels, false);
        }
    }
#pragma omp parallel for schedule(static) if (rows > 1024)
    for (int i = 0; i < rows; ++i) {
            const int row = group_rows[i];
            const float gate = route.gate[row];
            float* dst = grouped_residual + static_cast<size_t>(i) * channels;
            const float* src = grouped_ffn + static_cast<size_t>(i) * channels;
            const float* original = grouped_input + static_cast<size_t>(i) * channels;
            for (int c = 0; c < channels; ++c)
                dst[c] = src[c] * gate + original[c] * aux_ffn[c];
    }
    if (!cfg.attn) {
        for (int i = 0; i < rows; ++i)
            std::memcpy(state + static_cast<size_t>(group_rows[i]) * channels,
                        grouped_residual + static_cast<size_t>(i) * channels,
                        static_cast<size_t>(channels) * sizeof(float));
        if (debug_on_) debug_["blk-" + prefix] = std::vector<float>(state, state + (size_t)rows * channels);
        return;
    }

    const int heads = channels / kHeadDim;
    const int qkv_channels = channels * 3;
    const std::vector<int>& perm = window_perm(width, height, cfg);
    std::vector<int> group_perm(rows);
    for (int i = 0; i < rows; ++i) group_perm[i] = perm[group_rows[i]];
    FloatBuffer& grouped_qkv_buf = sc_.qkv;
    float* grouped_qkv = nullptr;
    if (int8_) {
        grouped_qkv_buf.resize(static_cast<size_t>(rows) * qkv_channels);
        grouped_qkv = grouped_qkv_buf.data();
    }
    FloatBuffer& qkv_slot_buf = sc_.att;
    qkv_slot_buf.resize(static_cast<size_t>(perm.size()) * qkv_channels);
    float* qkv_slot = qkv_slot_buf.data();
    std::fill(qkv_slot, qkv_slot + qkv_slot_buf.size(), 0.0f);
    if (!int8_) {
        ProfTimer tm(prof_on_ ? &prof_[0] : nullptr,
                     prof_on_ ? &layer_prof_[prefix + ".qkv"] : nullptr);
        gemm_kn_grouped(grouped_residual, channels, qkv_channels, w(prefix + ".qkv"),
                        qkv_slot, group_offsets.data(), experts, false, group_perm.data());
    } else for (int e = 0; e < experts; ++e) {
        const int start = group_offsets[e];
        const int count = group_offsets[e + 1] - start;
        if (!count) continue;
            expert_linear(prefix + ".qkv", e,
                          grouped_residual + static_cast<size_t>(start) * channels,
                          count, channels, qkv_channels,
                          grouped_qkv + static_cast<size_t>(start) * qkv_channels, false);
    }

    if (int8_) {
        for (int i = 0; i < rows; ++i) {
            const int row = group_rows[i];
            if (perm[row] < 0) continue;
            std::memcpy(qkv_slot + static_cast<size_t>(perm[row]) * qkv_channels,
                        grouped_qkv + static_cast<size_t>(i) * qkv_channels,
                        static_cast<size_t>(qkv_channels) * sizeof(float));
        }
    }
    std::vector<int> group_pos(rows);
    for (int i = 0; i < rows; ++i) group_pos[group_rows[i]] = i;
    FloatBuffer& attended_buf = sc_.proj;
    attended_buf.resize(static_cast<size_t>(rows) * channels);
    float* attended = attended_buf.data();
    window_attention(qkv_slot, attended, rows, width, height, heads, cfg, prefix,
                     group_pos.data());

    // Attention writes directly in expert order, so projection consumes it
    // without another full-field gather.
    if (!int8_) {
        ProfTimer tm(prof_on_ ? &prof_[0] : nullptr,
                     prof_on_ ? &layer_prof_[prefix + ".proj"] : nullptr);
        gemm_kn_grouped(attended, channels, channels, w(prefix + ".proj"), grouped_ffn,
                        group_offsets.data(), experts);
    } else for (int e = 0; e < experts; ++e) {
        const int start = group_offsets[e];
        const int count = group_offsets[e + 1] - start;
        if (!count) continue;
        expert_linear(prefix + ".proj", e, attended + static_cast<size_t>(start) * channels,
                      count, channels, channels,
                      grouped_ffn + static_cast<size_t>(start) * channels, false);
    }
    const float* aux_attn = w(prefix + ".aux_attn");
#pragma omp parallel for schedule(static) if (rows > 1024)
    for (int i = 0; i < rows; ++i) {
        const int row = group_rows[i];
        const float* residual = grouped_residual + static_cast<size_t>(i) * channels;
        const float* projected = grouped_ffn + static_cast<size_t>(i) * channels;
        float* dst = state + static_cast<size_t>(row) * channels;
        for (int c = 0; c < channels; ++c) dst[c] = residual[c] + projected[c] * aux_attn[c];
    }
    if (debug_on_) debug_["blk-" + prefix] = std::vector<float>(state, state + (size_t)rows * channels);
}

void Engine::window_attention(float* qkv, float* out, int rows, int width, int height,
                              int heads, const BlockCfg& cfg, const std::string& prefix,
                              const int* output_map) const {
    const int win = cfg.window;
    const int slots = win * win;
    const float* prior = w(prefix + ".attention.prior");
    const float* scale = w(prefix + ".attention.scale");
    const int windows_x = (width + cfg.shift_x + win - 1) / win;
    const int windows_y = (height + cfg.shift_y + win - 1) / win;
    const int windows = windows_x * windows_y;

#pragma omp parallel
    {
        std::vector<float> scores((size_t)slots * slots);
        std::vector<unsigned char> valid(slots);
        std::vector<int> index(slots);
#pragma omp for schedule(static)
        for (int widx = 0; widx < windows; widx++) {
            int wx = widx / windows_y, wy = widx % windows_y;
            int ox = wx * win - cfg.shift_x, oy = wy * win - cfg.shift_y;
            for (int s = 0; s < slots; s++) {
                int fx = ox + (s % win), fy = oy + (s / win);
                valid[s] = (fx >= 0 && fx < width && fy >= 0 && fy < height) ? 1 : 0;
                index[s] = valid[s] ? fy * width + fx : 0;
            }
            for (int h = 0; h < heads; h++) {
                // qkv 已是槽序(window_perm/gemm_kn_rows 写入):零 gather。
                attn_window_head(qkv + (size_t)widx * slots * (heads * 96) + h * 96,
                                 heads * 96, index.data(), valid.data(),
                                 slots, scale[h], prior + (size_t)h * slots * slots,
                                 scores.data(), out + h * 32, heads * 32, output_map);
            }
        }
    }
    (void)rows;
}

void Engine::vit_attention(float* qkv, float* out, int tokens, int padded, int heads,
                           const std::string& prefix, const int* output_map) const {
    const float* scale = w(prefix + ".attention.scale");
    const float kSqrt = std::sqrt(32.0f);
    // 整行 gather 一次(各 head 切片一起),head 循环零拷贝。注意:不能用
    // thread_local —— OpenMP 工作线程各有自己的 TLS 副本,主线程 resize 的
    // 缓冲它们看不见(踩过:worker 拿到 nullptr)。栈 vector 只按指针共享。
    std::vector<float> gbuf((size_t)tokens * heads * 96);
    for (int ti = 0; ti < tokens; ti++) {
        std::memcpy(gbuf.data() + (size_t)ti * heads * 96,
                    qkv + (size_t)ti * (heads * 96), (size_t)heads * 96 * sizeof(float));
    }
#pragma omp parallel for schedule(static)
    for (int h = 0; h < heads; h++) {
        static thread_local std::vector<float> scores;
        scores.resize((size_t)tokens * padded);
        attn_vit_head(gbuf.data() + h * 96, heads * 96, tokens, padded, scale[h] * kSqrt,
                      scores.data(), out + h * 32, heads * 32, output_map);
    }
}

void Engine::ffn_window_block(const BlockCfg& cfg, const std::string& prefix, float* state,
                              int width, int height) const {
    const int ch = dim(prefix + ".contract.weight", 0), hidden = cfg.hidden;
    const int win = cfg.window, slots = win * win, heads = ch / 32;
    const int wx = (width + cfg.shift_x + win - 1) / win;
    const int wy = (height + cfg.shift_y + win - 1) / win;
    const float* we = packed_.at(prefix + ".expand.weight").data();
    const float* be = w(prefix + ".expand.bias");
    const float* wc = packed_.at(prefix + ".contract.weight").data();
    const float* bc = w(prefix + ".contract.bias");
    const float* aux = w(prefix + ".aux_ffn");
    const float* wq = packed_.at(prefix + ".tail.qkv.weight").data();
    const float* bq = w(prefix + ".tail.qkv.bias");
    const float* wp = packed_.at(prefix + ".tail.proj.weight").data();
    const float* bp = w(prefix + ".tail.proj.bias");
    const float* ap = w(prefix + ".tail.aux_attn");
    const float* scale = w(prefix + ".tail.attention.scale");
    const float* prior = w(prefix + ".tail.attention.prior");
    // Windows are disjoint even when shifted. Keep the entire block in each
    // worker's cache, avoiding field-sized FFN/qkv/attention/projection passes.
#pragma omp parallel
    {
        static thread_local std::vector<float> workspace;
        workspace.resize(static_cast<size_t>(slots) * (hidden + 7 * ch) + slots * slots);
        float* input = workspace.data();
        float* temp = input + slots * ch;
        float* res = temp + slots * hidden;
        float* qkv = res + slots * ch;
        float* att = qkv + slots * ch * 3;
        float* proj = att + slots * ch;
        float* scores = proj + slots * ch;
        std::vector<int> field(slots), local(slots);
        std::vector<unsigned char> valid(slots);
#pragma omp for schedule(static)
        for (int wi = 0; wi < wx * wy; ++wi) {
            int ox = (wi / wy) * win - cfg.shift_x, oy = (wi % wy) * win - cfg.shift_y;
            for (int s = 0; s < slots; ++s) {
                int x = ox + s % win, y = oy + s / win;
                valid[s] = x >= 0 && x < width && y >= 0 && y < height;
                field[s] = valid[s] ? y * width + x : 0;
                local[s] = s;
                if (valid[s]) std::memcpy(input + s * ch, state + static_cast<size_t>(field[s]) * ch, ch * sizeof(float));
                else std::fill(input + s * ch, input + (s + 1) * ch, 0.0f);
            }
            gemm_kn_serial(input, slots, ch, hidden, we, temp, be, true);
            gemm_kn_serial(temp, slots, hidden, ch, wc, res, bc, false, input, aux);
            gemm_kn_serial(res, slots, ch, ch * 3, wq, qkv, bq);
            for (int h = 0; h < heads; ++h)
                attn_window_head(qkv + h * 96, ch * 3, local.data(), valid.data(), slots,
                                  scale[h], prior + static_cast<size_t>(h) * slots * slots,
                                  scores, att + h * 32, ch);
            // Invalid output rows are not written by attention. Zero them so
            // projection never consumes uninitialized temporary values.
            for (int s = 0; s < slots; ++s) if (!valid[s]) std::fill(att + s * ch, att + (s + 1) * ch, 0.0f);
            gemm_kn_serial(att, slots, ch, ch, wp, proj, bp);
            for (int s = 0; s < slots; ++s) if (valid[s]) {
                float* dst = state + static_cast<size_t>(field[s]) * ch;
                for (int c = 0; c < ch; ++c) dst[c] = res[s * ch + c] + proj[s * ch + c] * ap[c];
            }
        }
    }
}

void Engine::ffn_block(const BlockCfg& cfg, const std::string& prefix, float* state,
                       int rows, int width, int height) const {
    ProfTimer block_tm(nullptr, prof_on_ ? &layer_prof_["block:" + prefix] : nullptr);
    if (cfg.kind == "moe") {
        moe_block(cfg, prefix, state, rows, width, height);
        return;
    }
    int ch = dim(prefix + ".contract.weight", 0);
    if (cfg.kind == "ffn" && cfg.attn && ch <= 64 &&
        !debug_on_ && !prof_on_ && (!int8_ || (ch < kInt8MinK && cfg.hidden < kInt8MinK))) {
        ffn_window_block(cfg, prefix, state, width, height);
        return;
    }
    FloatBuffer& tmp = sc_.tmp;
    FloatBuffer& aux = sc_.aux;
    FloatBuffer& merge = sc_.merge;
    FloatBuffer& res = sc_.res;
    const float* auxv = w(prefix + ".aux_ffn");   // 残差融合:contract 出口 += state*aux

    if (cfg.kind == "ffn") {
        int h = cfg.hidden;
        if (cfg.attn) res.resize((size_t)rows * ch);
        if (!int8_ || (ch < kInt8MinK && h < kInt8MinK)) {
            ProfTimer tm(prof_on_ ? &prof_[0] : nullptr, prof_on_ ? &layer_prof_[prefix + ".ffn_fused"] : nullptr);
            ffn_kn(state, rows, ch, h, packed_.at(prefix + ".expand.weight").data(),
                   w(prefix + ".expand.bias"), packed_.at(prefix + ".contract.weight").data(),
                   w(prefix + ".contract.bias"), auxv, cfg.attn ? res.data() : state);
        } else {
            tmp.resize((size_t)rows * std::max(h, 1));
            linear_bias(state, rows, ch, h, prefix + ".expand", tmp.data(), true);
            linear_bias(tmp.data(), rows, h, ch, prefix + ".contract",
                        cfg.attn ? res.data() : state, false, state, auxv);
        }
    } else if (cfg.kind == "expert") {
        int e = cfg.experts, h = cfg.hidden, nc = cfg.narrow;
        // torch: ffn = silu(einsum('rc,ech->erh'))     -> [e][r][h]
        //        merged = einsum('erh,ehc->erc').reshape(rows, -1)
        // reshape 之后 [e][r][c] 的扁平缓冲被直接当作 [rows][ch] 用 —— 这个 e-外层的
        // 置乱是训练语义的一部分,C++ 必须逐位复刻(写成自然序会偏 ~1%)。
        merge.resize((size_t)e * rows * nc);
        res.resize((size_t)rows * ch);
        const float* we = w(prefix + ".expand");   // [e][ch][h]:每专家切片即 [K][N]
        const float* wn = w(prefix + ".narrow");   // [e][h][nc]
        if (!int8_ || (ch < kInt8MinK && h < kInt8MinK)) {
            ProfTimer tm(prof_on_ ? &prof_[0] : nullptr, prof_on_ ? &layer_prof_[prefix + ".expert_fused"] : nullptr);
            ffn_kn(state, rows, ch, h, we, nullptr, wn, nullptr, nullptr, merge.data(), nc, e);
        } else {
        aux.resize((size_t)e * rows * h);
        {
            ProfTimer tm(prof_on_ ? &prof_[0] : nullptr, prof_on_ ? &layer_prof_[prefix + ".expert_expand"] : nullptr);
            if (int8_ && ch >= kInt8MinK) {
                const Wq& wq = expert_i8_.at(prefix + ".expand");
                for (int ei = 0; ei < e; ei++) {
                    gemm_kn_i8(state, rows, ch, h,
                               wq.w.data() + (size_t)ei * h * wq.Kp, wq.Kp,
                               wq.sw.data() + (size_t)ei * h,
                               aux.data() + (size_t)ei * rows * h, nullptr, true, nullptr, nullptr,
                               wq.vnni.empty() ? nullptr : &wq.vnni[ei]);
                }
            } else {
                for (int ei = 0; ei < e; ei++) {
                    gemm_kn(state, rows, ch, h, we + (size_t)ei * ch * h,
                            aux.data() + (size_t)ei * rows * h, nullptr, true);
                }
            }
        }
        {
            ProfTimer tm(prof_on_ ? &prof_[0] : nullptr, prof_on_ ? &layer_prof_[prefix + ".expert_narrow"] : nullptr);
            if (int8_ && h >= kInt8MinK) {
                const Wq& wq = expert_i8_.at(prefix + ".narrow");
                for (int ei = 0; ei < e; ei++) {
                    gemm_kn_i8(aux.data() + (size_t)ei * rows * h, rows, h, nc,
                               wq.w.data() + (size_t)ei * nc * wq.Kp, wq.Kp,
                               wq.sw.data() + (size_t)ei * nc,
                               merge.data() + (size_t)ei * rows * nc, nullptr, false, nullptr, nullptr,
                               wq.vnni.empty() ? nullptr : &wq.vnni[ei]);
                }
            } else {
                for (int ei = 0; ei < e; ei++) {
                    gemm_kn(aux.data() + (size_t)ei * rows * h, rows, h, nc,
                            wn + (size_t)ei * h * nc, merge.data() + (size_t)ei * rows * nc, nullptr);
                }
            }
        }
        }
        linear_bias(merge.data(), rows, ch, ch, prefix + ".contract",
                    cfg.attn ? res.data() : state, false, state, auxv);
    } else {  // split
        int e = cfg.branches, bc = cfg.bc, mc = cfg.mc;
        aux.resize((size_t)rows * ch);
        merge.resize((size_t)rows * ch);
        res.resize((size_t)rows * ch);
        linear_bias(state, rows, ch, ch, prefix + ".branch", aux.data());
        const float* w2 = w(prefix + ".w2");
        const float* w3 = w(prefix + ".w3");
        {
        ProfTimer tm(prof_on_ ? &prof_[0] : nullptr, prof_on_ ? &layer_prof_[prefix + ".split"] : nullptr);
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            float* dst = merge.data() + (size_t)r * ch;
            const float* br = aux.data() + (size_t)r * ch;
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
        }
        linear_bias(merge.data(), rows, ch, ch, prefix + ".contract",
                    cfg.attn ? res.data() : state, false, state, auxv);
    }

    // 残差已随 contract 出口融合(FFN + state*aux_ffn):attn 路径在 res 里作 qkv
    // 输入与残差基;非 attn 路径已写穿 state,零后处理。
    if (cfg.attn) {
        if (debug_on_) debug_["ffn-" + prefix] = res.snapshot();
        int heads = ch / kHeadDim;
        FloatBuffer& qkv = sc_.qkv;
        FloatBuffer& att = sc_.att;
        FloatBuffer& proj = sc_.proj;
        // qkv 按窗口槽序写输出行(排列按几何缓存,gemm_kn_rows)—— attention
        // gather 归零(原占 attention 40%)。
        const std::vector<int>& perm = window_perm(width, height, cfg);
        qkv.resize((size_t)perm.size() * (heads * 96));
        att.resize((size_t)rows * ch);
        proj.resize((size_t)rows * ch);
        linear_bias(res.data(), rows, ch, ch * 3, prefix + ".tail.qkv", qkv.data(),
                    false, nullptr, nullptr, perm.data());
        {
            ProfTimer tm(prof_on_ ? &prof_[2] : nullptr, prof_on_ ? &layer_prof_[prefix + ".attention"] : nullptr);
            window_attention(qkv.data(), att.data(), rows, width, height, heads, cfg, prefix + ".tail");
        }
        if (debug_on_) debug_["att-" + prefix] = att.snapshot();
        linear_bias(att.data(), rows, ch, ch, prefix + ".tail.proj", proj.data());
        const float* aux2 = w(prefix + ".tail.aux_attn");
#pragma omp parallel for schedule(static) if (rows > 4096)
        for (int r = 0; r < rows; r++) {
            const float* rr = res.data() + (size_t)r * ch;
            const float* pr = proj.data() + (size_t)r * ch;
            float* orow = state + (size_t)r * ch;
            for (int c = 0; c < ch; c++) orow[c] = rr[c] + pr[c] * aux2[c];
        }
    }
    if (debug_on_) debug_["blk-" + prefix] = std::vector<float>(state, state + (size_t)rows * ch);
}

void Engine::vit_block(const BlockCfg& cfg, const std::string& prefix, float* state,
                       int tokens, int padded) const {
    ProfTimer block_tm(nullptr, prof_on_ ? &layer_prof_["block:" + prefix] : nullptr);
    if (cfg.kind == "moe_vit") {
        moe_vit_block(cfg, prefix, state, tokens, padded);
        return;
    }
    int ch = dim(prefix + ".contract.weight", 0);
    int ffn = cfg.hidden ? cfg.hidden : dim(prefix + ".expand.weight", 0);
    FloatBuffer& tmp = sc_.tmp;
    FloatBuffer& y = sc_.res;
    tmp.resize((size_t)tokens * ffn);
    y.resize((size_t)tokens * ch);
    linear_bias(state, tokens, ch, ffn, prefix + ".expand", tmp.data(), true);  // silu 融合出口
    const float* aux = w(prefix + ".aux_ffn");
    // contract 出口融合残差 y = FFN + state*aux_ffn
    linear_bias(tmp.data(), tokens, ffn, ch, prefix + ".contract", y.data(), false, state, aux);
    FloatBuffer& qkv = sc_.qkv;
    FloatBuffer& att = sc_.att;
    FloatBuffer& proj = sc_.proj;
    qkv.resize((size_t)tokens * ch * 3);
    att.resize((size_t)tokens * ch);
    proj.resize((size_t)tokens * ch);
    linear_bias(y.data(), tokens, ch, ch * 3, prefix + ".qkv", qkv.data());
    {
        ProfTimer tm(prof_on_ ? &prof_[2] : nullptr, prof_on_ ? &layer_prof_[prefix + ".attention"] : nullptr);
        vit_attention(qkv.data(), att.data(), tokens, padded, ch / kHeadDim, prefix);
    }
    if (debug_on_) debug_["att-" + prefix] = att.snapshot();
    linear_bias(att.data(), tokens, ch, ch, prefix + ".proj", proj.data());
    const float* aux2 = w(prefix + ".aux_attn");
    for (int r = 0; r < tokens; r++) {
        const float* yr = y.data() + (size_t)r * ch;
        const float* pr = proj.data() + (size_t)r * ch;
        float* orow = state + (size_t)r * ch;
        for (int c = 0; c < ch; c++) orow[c] = yr[c] + pr[c] * aux2[c];
    }
    if (debug_on_) debug_["blk-" + prefix] = std::vector<float>(state, state + (size_t)tokens * ch);
}

void Engine::moe_vit_block(const BlockCfg& cfg, const std::string& prefix, float* state,
                           int tokens, int padded) const {
    const int channels = dim(prefix + ".expand", 1);
    const int hidden = dim(prefix + ".expand", 2);
    const int experts = cfg.experts;
    const float* gamma = w(prefix + ".router.0.weight");
    const float* beta = w(prefix + ".router.0.bias");
    const float* router = w(prefix + ".router.1.weight");
    const MoeRoute route = route_moe(state, tokens, tokens, 1, channels, experts,
                                     gamma, beta, router, false);
    std::vector<int> group_offsets(experts + 1, 0), group_rows(tokens);
    for (int r = 0; r < tokens; ++r) ++group_offsets[route.assign[r] + 1];
    for (int e = 0; e < experts; ++e) group_offsets[e + 1] += group_offsets[e];
    std::vector<int> cursor = group_offsets;
    for (int r = 0; r < tokens; ++r) group_rows[cursor[route.assign[r]]++] = r;

    auto expert_linear = [&](const std::string& name, int expert, const float* input,
                             int count, int K, int N, float* output, bool act) {
        auto qi = expert_i8_.find(name);
        const bool use_i8 = int8_ && K >= kInt8MinK && qi != expert_i8_.end();
        if (use_i8) {
            const Wq& q = qi->second;
            gemm_kn_i8(input, count, K, N,
                       q.w.data() + static_cast<size_t>(expert) * N * q.Kp, q.Kp,
                       q.sw.data() + static_cast<size_t>(expert) * N,
                       output, nullptr, act, nullptr, nullptr,
                       q.vnni.empty() ? nullptr : &q.vnni[expert]);
        } else {
            const float* weights = t(name).data + static_cast<size_t>(expert) * K * N;
            gemm_kn(input, count, K, N, weights, output, nullptr, act);
        }
    };

    FloatBuffer& input_buf = sc_.tmp;
    FloatBuffer& hidden_buf = sc_.aux;
    FloatBuffer& ffn_buf = sc_.merge;
    FloatBuffer& residual_buf = sc_.res;
    input_buf.resize(static_cast<size_t>(tokens) * channels);
    hidden_buf.resize(static_cast<size_t>(tokens) * hidden);
    ffn_buf.resize(static_cast<size_t>(tokens) * channels);
    residual_buf.resize(static_cast<size_t>(tokens) * channels);
    float* grouped_input = input_buf.data();
    float* grouped_hidden = hidden_buf.data();
    float* grouped_ffn = ffn_buf.data();
    float* grouped_residual = residual_buf.data();
    for (int i = 0; i < tokens; ++i)
        std::memcpy(grouped_input + static_cast<size_t>(i) * channels,
                    state + static_cast<size_t>(group_rows[i]) * channels,
                    static_cast<size_t>(channels) * sizeof(float));
    const float* aux_ffn = w(prefix + ".aux_ffn");
    if (!int8_) {
        ProfTimer tm(prof_on_ ? &prof_[0] : nullptr,
                     prof_on_ ? &layer_prof_[prefix + ".ffn"] : nullptr);
        gemm_kn_grouped(grouped_input, channels, hidden, w(prefix + ".expand"), grouped_hidden,
                        group_offsets.data(), experts, true);
        gemm_kn_grouped(grouped_hidden, hidden, channels, w(prefix + ".contract"), grouped_ffn,
                        group_offsets.data(), experts);
    }
    for (int e = 0; e < experts; ++e) {
        const int start = group_offsets[e];
        const int count = group_offsets[e + 1] - start;
        if (!count) continue;
        if (int8_) {
        expert_linear(prefix + ".expand", e, grouped_input + static_cast<size_t>(start) * channels,
                      count, channels, hidden, grouped_hidden + static_cast<size_t>(start) * hidden, true);
        expert_linear(prefix + ".contract", e, grouped_hidden + static_cast<size_t>(start) * hidden,
                      count, hidden, channels, grouped_ffn + static_cast<size_t>(start) * channels, false);
        }
        for (int i = 0; i < count; ++i) {
            const int row = group_rows[start + i];
            const float gate = route.gate[row];
            float* dst = grouped_residual + static_cast<size_t>(start + i) * channels;
            const float* src = grouped_ffn + static_cast<size_t>(start + i) * channels;
            const float* original = state + static_cast<size_t>(row) * channels;
            for (int c = 0; c < channels; ++c)
                dst[c] = src[c] * gate + original[c] * aux_ffn[c];
        }
    }

    const int qkv_channels = channels * 3;
    FloatBuffer& qkv_buf = sc_.qkv;
    qkv_buf.resize(static_cast<size_t>(tokens) * qkv_channels);
    float* grouped_qkv = qkv_buf.data();
    if (!int8_) {
        ProfTimer tm(prof_on_ ? &prof_[0] : nullptr,
                     prof_on_ ? &layer_prof_[prefix + ".qkv"] : nullptr);
        gemm_kn_grouped(grouped_residual, channels, qkv_channels, w(prefix + ".qkv"),
                        grouped_qkv, group_offsets.data(), experts);
    } else for (int e = 0; e < experts; ++e) {
        const int start = group_offsets[e];
        const int count = group_offsets[e + 1] - start;
        if (!count) continue;
        expert_linear(prefix + ".qkv", e, grouped_residual + static_cast<size_t>(start) * channels,
                      count, channels, qkv_channels,
                      grouped_qkv + static_cast<size_t>(start) * qkv_channels, false);
    }
    FloatBuffer& qkv_spatial_buf = sc_.att;
    qkv_spatial_buf.resize(static_cast<size_t>(tokens) * qkv_channels);
    float* qkv_spatial = qkv_spatial_buf.data();
    for (int i = 0; i < tokens; ++i)
        std::memcpy(qkv_spatial + static_cast<size_t>(group_rows[i]) * qkv_channels,
                    grouped_qkv + static_cast<size_t>(i) * qkv_channels,
                    static_cast<size_t>(qkv_channels) * sizeof(float));
    FloatBuffer& attended_buf = sc_.proj;
    attended_buf.resize(static_cast<size_t>(tokens) * channels);
    float* attended = attended_buf.data();
    std::vector<int> group_pos(tokens);
    for (int i = 0; i < tokens; ++i) group_pos[group_rows[i]] = i;
    vit_attention(qkv_spatial, attended, tokens, padded, channels / kHeadDim, prefix,
                  group_pos.data());
    if (!int8_) {
        ProfTimer tm(prof_on_ ? &prof_[0] : nullptr,
                     prof_on_ ? &layer_prof_[prefix + ".proj"] : nullptr);
        gemm_kn_grouped(attended, channels, channels, w(prefix + ".proj"), grouped_ffn,
                        group_offsets.data(), experts);
    } else for (int e = 0; e < experts; ++e) {
        const int start = group_offsets[e];
        const int count = group_offsets[e + 1] - start;
        if (!count) continue;
        expert_linear(prefix + ".proj", e, attended + static_cast<size_t>(start) * channels,
                      count, channels, channels,
                      grouped_ffn + static_cast<size_t>(start) * channels, false);
    }
    const float* aux_attn = w(prefix + ".aux_attn");
    for (int i = 0; i < tokens; ++i) {
        const int row = group_rows[i];
        const float* residual = grouped_residual + static_cast<size_t>(i) * channels;
        const float* projected = grouped_ffn + static_cast<size_t>(i) * channels;
        float* dst = state + static_cast<size_t>(row) * channels;
        for (int c = 0; c < channels; ++c) dst[c] = residual[c] + projected[c] * aux_attn[c];
    }
    if (debug_on_) debug_["blk-" + prefix] = std::vector<float>(state, state + (size_t)tokens * channels);
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
        const std::string expand_name = tensors_.count(p + ".expand") ? p + ".expand" : p + ".expand.weight";
        chs[i] = tensors_.count(expand_name) ? t(expand_name).shape[1]
                                             : t(p + ".contract.weight").shape[0];
    }

    FloatBuffer* skip = sc_.sk;
    for (int i = 0; i < 6; i++) skip[i].resize((size_t)rows[i] * chs[i]);
    FloatBuffer& x = sc_.x;
    x.resize((size_t)rows[0] * ch_full);
    linear_bias(features, rows[0], 16, ch_full, "adapter", x.data());

    for (int i = 0; i < 6; i++) {
        int ch = chs[i];
        x.resize((size_t)rows[i] * ch);            // full 级 ch_full == chs[0]
        const auto& bl = blocks_[0].at(order[i]);
        for (size_t j = 0; j < bl.size(); j++) {
            std::string p = std::string("enc_blocks.") + order[i] + "." + std::to_string(j);
            ffn_block(bl[j], p, x.data(), rows[i], dims[i][0], dims[i][1]);
        }
        if (debug_on_) debug_["s-enc-" + std::string(order[i])] = x.snapshot();
        if (i < 5) {
            FloatBuffer& pooled = sc_.pooled;
            pooled.resize((size_t)rows[i + 1] * ch);
            box_downsample(x.data(), pooled.data(), dims[i][0], dims[i][1],
                           dims[i + 1][0], dims[i + 1][1], ch);
            skip[i].swap(x);                       // 跳连零拷贝:x 复用旧 skip 缓冲
            std::string tn = std::string("enc_trans.") + order[i + 1];
            x.resize((size_t)rows[i + 1] * chs[i + 1]);
            linear_bias(pooled.data(), rows[i + 1], ch, chs[i + 1], tn, x.data());
            if (debug_on_) debug_["trans-" + std::string(order[i + 1])] = x.snapshot();
        } else {
            skip[i].swap(x);
        }
    }

    // ---- ViT
    int d5w = geom_.levels[5].width, d5h = geom_.levels[5].height;
    int rows5 = d5w * d5h;
    int ch4 = chs[5];
    int vit_ch = dim("vit_in.weight", 0);
    FloatBuffer& pooled = sc_.pooled;
    pooled.resize((size_t)rows5 * ch4);
    box_downsample(skip[5].data(), pooled.data(), dims[5][0], dims[5][1], d5w, d5h, ch4);
    FloatBuffer& vstate = sc_.vstate;
    vstate.resize((size_t)rows5 * vit_ch);
    linear_bias(pooled.data(), rows5, ch4, vit_ch, "vit_in", vstate.data());
    if (debug_on_) debug_["trans-vit"] = vstate.snapshot();
    for (size_t j = 0; j < vit_blocks_.size(); j++) {
        std::string p = "vit_blocks." + std::to_string(j);
        vit_block(vit_blocks_[j], p, vstate.data(), vit_tokens_, vit_padded_);
    }
    if (debug_on_) debug_["s-vit"] = vstate.snapshot();
    x.resize((size_t)rows5 * ch4);
    linear_bias(vstate.data(), rows5, vit_ch, ch4, "vit_out", x.data());

    // ---- 解码:d5 -> d4 -> d3 -> d2 -> d1 -> d0 -> full
    const char* dec[5] = {"d4", "d3", "d2", "d1", "d0"};
    int prev_w = d5w, prev_h = d5h, prev_rows = rows5;
    FloatBuffer& up = sc_.up;
    for (int i = 0; i < 5; i++) {
        const char* name = dec[i];
        int idx = 5 - i;                                  // d4 -> 5, d3 -> 4 ...
        int lw = dims[idx][0], lh = dims[idx][1], r = rows[idx], ch = chs[idx];
        if (i > 0) {
            // torch:dec_trans 作用在上一级网格行上(prev_rows 行,行数不变),
            // merge 再 2x 上采样到本级。旧实现把行数写成 r 并越界读 x —— 输出的
            // 前 prev_rows 行恰好等价、垃圾行无人消费才没炸;这里改成 torch 的行数。
            std::string tn = std::string("dec_trans.") + name;
            up.resize((size_t)prev_rows * ch);
            linear_bias(x.data(), prev_rows, chs[idx + 1], ch, tn, up.data());
            std::swap(x, up);
        }
        up.resize((size_t)r * ch);
        upsample2_merge(x.data(), skip[idx].data(), nullptr,
                        w(std::string("merge.") + name + ".aux"), up.data(),
                        prev_w, prev_h, lw, lh, ch);
        const auto& bl = blocks_[1].at(name);
        for (size_t j = 0; j < bl.size(); j++) {
            std::string p = std::string("dec_blocks.") + name + "." + std::to_string(j);
            ffn_block(bl[j], p, up.data(), r, lw, lh);
        }
        if (debug_on_) debug_["s-dec-" + std::string(name)] = up.snapshot();
        std::swap(x, up);
        prev_w = lw; prev_h = lh; prev_rows = r;
    }

    // ---- post_blend + full 级解码块 + head
    {
        int lw = dims[0][0], lh = dims[0][1], r = rows[0], ch = chs[0];
        up.resize((size_t)r * ch);
        const float* pair = w("post.aux_pair");               // [2, ch]
        upsample2_merge(x.data(), skip[0].data(), pair, pair + ch, up.data(),
                        prev_w, prev_h, lw, lh, ch);
        const auto& bl = blocks_[1].at("full");
        for (size_t j = 0; j < bl.size(); j++) {
            std::string p = "dec_blocks.full." + std::to_string(j);
            ffn_block(bl[j], p, up.data(), r, lw, lh);
        }
        linear_bias(up.data(), r, ch, 4, "head", head);
    }
    (void)prev_rows;
}
