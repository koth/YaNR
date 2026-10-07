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
#include <sstream>
#include <stdexcept>

namespace {

constexpr int kHeadDim = 32;

inline float silu(float v) { return v / (1.0f + std::exp(-v)); }

// 粗粒度剖析计时器:acc 为空时近零开销(prof_on_ 门控)。
struct ProfTimer {
    double* acc;
    std::chrono::steady_clock::time_point t0;
    explicit ProfTimer(double* a) : acc(a), t0(std::chrono::steady_clock::now()) {}
    ~ProfTimer() {
        if (acc) {
            *acc += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        }
    }
};

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
    // 打包 [N][K] Linear 权重 -> [K][N](GEMM 微内核的 B 布局);专家 einsum 切片
    // (expand[e][ch][h] / narrow[e][h][nc])本身就是 [K][N],直接切指针用。
    for (const auto& kv : order) {
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
    for (const auto& kv : order) {
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
        } else if (sh.size() == 3 && name.size() >= 7 &&
                   (name.compare(name.size() - 7, 7, ".expand") == 0 ||
                    name.compare(name.size() - 7, 7, ".narrow") == 0)) {
            int E = sh[0], K = sh[1], N = sh[2], Kp = (K + 15) & ~15;
            Wq& q = expert_i8_[name];
            q.N = E * N; q.Kp = Kp;
            q.w.resize((size_t)E * N * Kp);
            q.sw.resize((size_t)E * N);
            for (int e = 0; e < E; e++) {
                quantize_w_kn(t(name).data + (size_t)e * K * N, K, N, Kp,
                              q.w.data() + (size_t)e * N * Kp,
                              q.sw.data() + (size_t)e * N);
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
    ProfTimer tm(prof_on_ ? &prof_[0] : nullptr);
    const float* bias = tensors_.count(name + ".bias") ? w(name + ".bias") : nullptr;
    if (int8_ && K >= kInt8MinK && !row_map) {
        auto it = packed_i8_.find(name + ".weight");
        if (it == packed_i8_.end()) throw std::runtime_error("missing int8 weight " + name);
        gemm_kn_i8(x, rows, K, N, it->second.w.data(), it->second.Kp,
                   it->second.sw.data(), out, bias, silu_act, X, aux);
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

void Engine::upsample2_merge(const float* x, const float* skip, const float* aux0,
                             const float* aux1, float* out, int in_w, int in_h, int out_w,
                             int out_h, int ch) const {
#pragma omp parallel for schedule(static)
    for (int y = 0; y < out_h; y++) {
        int sy = std::min(y >> 1, in_h - 1);
        for (int xx = 0; xx < out_w; xx++) {
            int sx = std::min(xx >> 1, in_w - 1);
            const float* s = x + ((size_t)sy * in_w + sx) * ch;
            const float* sk = skip + ((size_t)y * out_w + xx) * ch;
            float* o = out + ((size_t)y * out_w + xx) * ch;
            if (aux0) {
                for (int c = 0; c < ch; c++) o[c] = s[c] * aux0[c] + sk[c] * aux1[c];
            } else {
                for (int c = 0; c < ch; c++) o[c] = s[c] + sk[c] * aux1[c];
            }
        }
    }
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

void Engine::window_attention(float* qkv, float* out, int rows, int width, int height,
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
                                 scores.data(), out + h * 32, heads * 32);
            }
        }
    }
    (void)rows;
}

void Engine::vit_attention(float* qkv, float* out, int tokens, int padded, int heads,
                           const std::string& prefix) const {
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
                      scores.data(), out + h * 32, heads * 32);
    }
}

void Engine::ffn_block(const BlockCfg& cfg, const std::string& prefix, float* state,
                       int rows, int width, int height) const {
    int ch = dim(prefix + ".contract.weight", 0);
    std::vector<float>& tmp = sc_.tmp;
    std::vector<float>& aux = sc_.aux;
    std::vector<float>& merge = sc_.merge;
    std::vector<float>& res = sc_.res;
    const float* auxv = w(prefix + ".aux_ffn");   // 残差融合:contract 出口 += state*aux

    if (cfg.kind == "ffn") {
        int h = cfg.hidden;
        tmp.resize((size_t)rows * std::max(h, 1));
        res.resize((size_t)rows * ch);
        linear_bias(state, rows, ch, h, prefix + ".expand", tmp.data(), true);  // silu 融合出口
        linear_bias(tmp.data(), rows, h, ch, prefix + ".contract",
                    cfg.attn ? res.data() : state, false, state, auxv);
    } else if (cfg.kind == "expert") {
        int e = cfg.experts, h = cfg.hidden, nc = cfg.narrow;
        // torch: ffn = silu(einsum('rc,ech->erh'))     -> [e][r][h]
        //        merged = einsum('erh,ehc->erc').reshape(rows, -1)
        // reshape 之后 [e][r][c] 的扁平缓冲被直接当作 [rows][ch] 用 —— 这个 e-外层的
        // 置乱是训练语义的一部分,C++ 必须逐位复刻(写成自然序会偏 ~1%)。
        aux.resize((size_t)e * rows * h);
        merge.resize((size_t)e * rows * nc);
        res.resize((size_t)rows * ch);
        const float* we = w(prefix + ".expand");   // [e][ch][h]:每专家切片即 [K][N]
        const float* wn = w(prefix + ".narrow");   // [e][h][nc]
        {
            ProfTimer tm(prof_on_ ? &prof_[0] : nullptr);
            if (int8_ && ch >= kInt8MinK) {
                const Wq& wq = expert_i8_.at(prefix + ".expand");
                for (int ei = 0; ei < e; ei++) {
                    gemm_kn_i8(state, rows, ch, h,
                               wq.w.data() + (size_t)ei * h * wq.Kp, wq.Kp,
                               wq.sw.data() + (size_t)ei * h,
                               aux.data() + (size_t)ei * rows * h, nullptr, true);
                }
            } else {
                for (int ei = 0; ei < e; ei++) {
                    gemm_kn(state, rows, ch, h, we + (size_t)ei * ch * h,
                            aux.data() + (size_t)ei * rows * h, nullptr, true);
                }
            }
        }
        {
            ProfTimer tm(prof_on_ ? &prof_[0] : nullptr);
            if (int8_ && h >= kInt8MinK) {
                const Wq& wq = expert_i8_.at(prefix + ".narrow");
                for (int ei = 0; ei < e; ei++) {
                    gemm_kn_i8(aux.data() + (size_t)ei * rows * h, rows, h, nc,
                               wq.w.data() + (size_t)ei * nc * wq.Kp, wq.Kp,
                               wq.sw.data() + (size_t)ei * nc,
                               merge.data() + (size_t)ei * rows * nc, nullptr);
                }
            } else {
                for (int ei = 0; ei < e; ei++) {
                    gemm_kn(aux.data() + (size_t)ei * rows * h, rows, h, nc,
                            wn + (size_t)ei * h * nc, merge.data() + (size_t)ei * rows * nc, nullptr);
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
        ProfTimer tm(prof_on_ ? &prof_[0] : nullptr);
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
        if (debug_on_) debug_["ffn-" + prefix] = res;
        int heads = ch / kHeadDim;
        std::vector<float>& qkv = sc_.qkv;
        std::vector<float>& att = sc_.att;
        std::vector<float>& proj = sc_.proj;
        // qkv 按窗口槽序写输出行(排列按几何缓存,gemm_kn_rows)—— attention
        // gather 归零(原占 attention 40%)。
        const std::vector<int>& perm = window_perm(width, height, cfg);
        qkv.resize((size_t)perm.size() * (heads * 96));
        att.resize((size_t)rows * ch);
        proj.resize((size_t)rows * ch);
        linear_bias(res.data(), rows, ch, ch * 3, prefix + ".tail.qkv", qkv.data(),
                    false, nullptr, nullptr, perm.data());
        {
            ProfTimer tm(prof_on_ ? &prof_[2] : nullptr);
            window_attention(qkv.data(), att.data(), rows, width, height, heads, cfg, prefix + ".tail");
        }
        if (debug_on_) debug_["att-" + prefix] = att;
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
    int ch = dim(prefix + ".contract.weight", 0);
    int ffn = cfg.hidden ? cfg.hidden : dim(prefix + ".expand.weight", 0);
    std::vector<float>& tmp = sc_.tmp;
    std::vector<float>& y = sc_.res;
    tmp.resize((size_t)tokens * ffn);
    y.resize((size_t)tokens * ch);
    linear_bias(state, tokens, ch, ffn, prefix + ".expand", tmp.data(), true);  // silu 融合出口
    const float* aux = w(prefix + ".aux_ffn");
    // contract 出口融合残差 y = FFN + state*aux_ffn
    linear_bias(tmp.data(), tokens, ffn, ch, prefix + ".contract", y.data(), false, state, aux);
    std::vector<float>& qkv = sc_.qkv;
    std::vector<float>& att = sc_.att;
    std::vector<float>& proj = sc_.proj;
    qkv.resize((size_t)tokens * ch * 3);
    att.resize((size_t)tokens * ch);
    proj.resize((size_t)tokens * ch);
    linear_bias(y.data(), tokens, ch, ch * 3, prefix + ".qkv", qkv.data());
    {
        ProfTimer tm(prof_on_ ? &prof_[2] : nullptr);
        vit_attention(qkv.data(), att.data(), tokens, padded, ch / kHeadDim, prefix);
    }
    if (debug_on_) debug_["att-" + prefix] = att;
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

    std::vector<float>* skip = sc_.sk;
    for (int i = 0; i < 6; i++) skip[i].resize((size_t)rows[i] * chs[i]);
    std::vector<float>& x = sc_.x;
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
        if (debug_on_) debug_["s-enc-" + std::string(order[i])] = x;
        if (i < 5) {
            std::vector<float>& pooled = sc_.pooled;
            pooled.resize((size_t)rows[i + 1] * ch);
            box_downsample(x.data(), pooled.data(), dims[i][0], dims[i][1],
                           dims[i + 1][0], dims[i + 1][1], ch);
            skip[i].swap(x);                       // 跳连零拷贝:x 复用旧 skip 缓冲
            std::string tn = std::string("enc_trans.") + order[i + 1];
            x.resize((size_t)rows[i + 1] * chs[i + 1]);
            linear_bias(pooled.data(), rows[i + 1], ch, chs[i + 1], tn, x.data());
            if (debug_on_) debug_["trans-" + std::string(order[i + 1])] = x;
        } else {
            skip[i].swap(x);
        }
    }

    // ---- ViT
    int d5w = geom_.levels[5].width, d5h = geom_.levels[5].height;
    int rows5 = d5w * d5h;
    int ch4 = chs[5];
    int vit_ch = dim("vit_in.weight", 0);
    std::vector<float>& pooled = sc_.pooled;
    pooled.resize((size_t)rows5 * ch4);
    box_downsample(skip[5].data(), pooled.data(), dims[5][0], dims[5][1], d5w, d5h, ch4);
    std::vector<float>& vstate = sc_.vstate;
    vstate.resize((size_t)rows5 * vit_ch);
    linear_bias(pooled.data(), rows5, ch4, vit_ch, "vit_in", vstate.data());
    if (debug_on_) debug_["trans-vit"] = vstate;
    for (size_t j = 0; j < vit_blocks_.size(); j++) {
        std::string p = "vit_blocks." + std::to_string(j);
        vit_block(vit_blocks_[j], p, vstate.data(), vit_tokens_, vit_padded_);
    }
    if (debug_on_) debug_["s-vit"] = vstate;
    x.resize((size_t)rows5 * ch4);
    linear_bias(vstate.data(), rows5, vit_ch, ch4, "vit_out", x.data());

    // ---- 解码:d5 -> d4 -> d3 -> d2 -> d1 -> d0 -> full
    const char* dec[5] = {"d4", "d3", "d2", "d1", "d0"};
    int prev_w = d5w, prev_h = d5h, prev_rows = rows5;
    std::vector<float>& up = sc_.up;
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
        if (debug_on_) debug_["s-dec-" + std::string(name)] = up;
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
