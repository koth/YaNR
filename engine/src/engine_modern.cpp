// Native FP32 implementations of the current routed and CNN students.
// Spatial tensors use row-major NHWC; GroupNorm still reduces C/8 * H * W.
#include "engine.h"
#include "attn.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace {
float activation(float v) { return v / (1.f + std::exp(-v)); }
void resize_bilinear(const float* in, float* out, int iw, int ih, int ow, int oh, int c) {
#pragma omp parallel for schedule(static)
    for (int y = 0; y < oh; ++y) {
        float fy = std::max(0.f, (y + .5f) * (float(ih) / oh) - .5f);
        int y0 = int(fy), y1 = std::min(y0 + 1, ih - 1); float ty = fy - y0;
        for (int x = 0; x < ow; ++x) {
            float fx = std::max(0.f, (x + .5f) * (float(iw) / ow) - .5f);
            int x0 = int(fx), x1 = std::min(x0 + 1, iw - 1); float tx = fx - x0;
            for (int k = 0; k < c; ++k) {
                auto at = [&](int xx, int yy) { return in[(size_t(yy) * iw + xx) * c + k]; };
                out[(size_t(y) * ow + x) * c + k] =
                    (1 - ty) * ((1 - tx) * at(x0, y0) + tx * at(x1, y0)) +
                    ty * ((1 - tx) * at(x0, y1) + tx * at(x1, y1));
            }
        }
    }
}
}

Engine::Route Engine::route(const float* state, int width, int height, int c,
                            const std::string& p, bool windows) const {
    int rows = width * height, nw = (width + 7) / 8;
    int items = windows ? nw * ((height + 7) / 8) : rows, E = routed_experts_;
    Route r; r.pick.resize(rows); r.gate.resize(rows); r.offsets.assign(E + 1, 0);
    std::vector<float> logits(size_t(items) * E), gates(items), pick(items), pooled(c), norm(c);
    for (int item = 0; item < items; ++item) {
        if (windows) {
            std::fill(pooled.begin(), pooled.end(), 0.f); int n = 0;
            for (int y = item / nw * 8; y < std::min(height, item / nw * 8 + 8); ++y)
                for (int x = item % nw * 8; x < std::min(width, item % nw * 8 + 8); ++x) {
                    for (int k = 0; k < c; ++k) pooled[k] += state[(size_t(y) * width + x) * c + k];
                    ++n;
                }
            for (float& v : pooled) v /= n;
        } else std::copy_n(state + size_t(item) * c, c, pooled.data());
        float mean = std::accumulate(pooled.begin(), pooled.end(), 0.f) / c, var = 0;
        for (float v : pooled) var += (v - mean) * (v - mean);
        float inv = 1.f / std::sqrt(var / c + 1e-5f);
        for (int k = 0; k < c; ++k) norm[k] = (pooled[k] - mean) * inv * w(p + ".router.0.weight")[k] + w(p + ".router.0.bias")[k];
        int best = 0; float mx = -INFINITY, den = 0;
        for (int e = 0; e < E; ++e) {
            float v = 0; for (int k = 0; k < c; ++k) v = std::fma(norm[k], w(p + ".router.1.weight")[e * c + k], v);
            logits[item * E + e] = v; mx = std::max(mx, v);
            if (v + w(p + ".router_bias")[e] > logits[item * E + best] + w(p + ".router_bias")[best]) best = e;
        }
        for (int e = 0; e < E; ++e) den += std::exp(logits[item * E + e] - mx);
        pick[item] = float(best); gates[item] = E * std::exp(logits[item * E + best] - mx) / den;
    }
    for (int row = 0; row < rows; ++row) {
        int item = windows ? row / width / 8 * nw + row % width / 8 : row;
        r.pick[row] = int(pick[item]); r.gate[row] = gates[item]; ++r.offsets[r.pick[row] + 1];
    }
    std::partial_sum(r.offsets.begin(), r.offsets.end(), r.offsets.begin());
    auto cursor = r.offsets; r.order.resize(rows);
    for (int row = 0; row < rows; ++row) r.order[cursor[r.pick[row]]++] = row;
    if (debug_on_) { debug_["logits-" + p] = logits; debug_["pick-" + p] = pick; debug_["gate-" + p] = gates; }
    return r;
}

std::vector<float> Engine::routed_projection(const float* in, int rows, int K, int N,
        const std::string& key, const Route& r, const std::string& contract) const {
    const int O = contract.empty() ? N : dim(contract, 2);
    std::vector<float> grouped(size_t(rows) * K), hidden(size_t(rows) * N), result(size_t(rows) * O);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) std::copy_n(in + size_t(r.order[i]) * K, K, grouped.data() + size_t(i) * K);
    gemm_kn_grouped(grouped.data(), K, N, w(key) + size_t(shared_experts_) * K * N,
                    hidden.data(), r.offsets.data(), routed_experts_, !contract.empty());
    std::vector<float> output;
    const float* values = hidden.data();
    if (!contract.empty()) {
        output.resize(size_t(rows) * O);
        gemm_kn_grouped(hidden.data(), N, O, w(contract) + size_t(shared_experts_) * N * O,
                        output.data(), r.offsets.data(), routed_experts_);
        values = output.data();
    }
    const std::string bias_key = key.size() >= 7 && key.substr(key.size() - 7) == ".weight" ? key.substr(0, key.size() - 7) + ".bias" : "";
    const float* bias = tensors_.count(bias_key) ? w(bias_key) : nullptr;
    if (shared_experts_) {
        std::vector<float> shared(size_t(rows) * N);
        gemm_kn(in, rows, K, N, w(key), shared.data(), bias, !contract.empty());
        if (contract.empty()) result.swap(shared);
        else gemm_kn(shared.data(), rows, N, O, w(contract), result.data(), nullptr);
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        int row = r.order[i], expert = r.pick[row] + shared_experts_;
        for (int c = 0; c < O; ++c)
            result[size_t(row) * O + c] = .5f * (result[size_t(row) * O + c] +
                r.gate[row] * (values[size_t(i) * O + c] + (bias ? bias[expert * O + c] : 0)));
    }
    return result;
}

void Engine::routed_transition(const float* in, float* out, int width, int height, int K, int N,
                               const std::string& p, bool windows) const {
    auto r = route(in, width, height, K, p, windows);
    auto values = routed_projection(in, width * height, K, N, p + ".weight", r);
    std::copy(values.begin(), values.end(), out);
    if (debug_on_) debug_["output-" + p] = values;
}

void Engine::routed_block(const BlockCfg& cfg, const std::string& p, float* state,
                          int width, int height, bool global) const {
    int rows = width * height, c = dim(p + ".expand", 1);
    auto r = route(state, width, height, c, p, !global);
    auto value = routed_projection(state, rows, c, cfg.hidden, p + ".expand", r, p + (global ? ".contract" : ".narrow"));
    for (size_t i = 0; i < value.size(); ++i) value[i] += state[i] * w(p + ".aux_ffn")[i % c];
    if (debug_on_) debug_["ffn-" + p] = value;
    if (cfg.attn) {
        auto qkv = routed_projection(value.data(), rows, c, c * 3, p + ".qkv", r);
        if (debug_on_) debug_["qkv-" + p] = qkv;
        std::vector<float> att(size_t(rows) * c);
        if (global) vit_attention(qkv.data(), att.data(), rows, vit_padded_, c / 32, p);
        else {
            // The existing attention primitive consumes slot-major QKV.
            int extent = (cfg.window - 1) * cfg.dilation + 1;
            int wx = (width + cfg.shift_x + extent - 1) / extent;
            int wy = (height + cfg.shift_y + extent - 1) / extent;
            std::vector<float> slots(size_t(wx) * wy * cfg.window * cfg.window * c * 3, 0.f);
            for (int xx = 0; xx < wx; ++xx) for (int yy = 0; yy < wy; ++yy)
                for (int s = 0; s < cfg.window * cfg.window; ++s) {
                    int x = xx * extent - cfg.shift_x + s % cfg.window * cfg.dilation;
                    int y = yy * extent - cfg.shift_y + s / cfg.window * cfg.dilation;
                    if (x >= 0 && x < width && y >= 0 && y < height)
                        std::copy_n(qkv.data() + (size_t(y) * width + x) * c * 3, c * 3,
                                    slots.data() + ((size_t(xx) * wy + yy) * cfg.window * cfg.window + s) * c * 3);
                }
            // attn_window_head normalizes Q/K in place.
            window_attention(slots.data(), att.data(), rows, width, height, c / 32, cfg, p);
        }
        if (debug_on_) debug_["att-" + p] = att;
        auto projection = routed_projection(att.data(), rows, c, c, p + ".proj", r);
        for (size_t i = 0; i < value.size(); ++i) value[i] += projection[i] * w(p + ".aux_attn")[i % c];
    }
    std::copy(value.begin(), value.end(), state);
    if (debug_on_) debug_["blk-" + p] = value;
}

void Engine::prepare_cnn_weights() {
    for (const auto& kv : tensors_) {
        const auto& s = kv.second.shape;
        if (s.size() != 4 || kv.first.find(".weight") == std::string::npos) continue;
        int O = s[0], I = s[1], k = s[2];
        auto& out = cnn_weights_[kv.first]; out.resize(size_t(O) * I * k * k);
        for (int o = 0; o < O; ++o) for (int i = 0; i < I; ++i) for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x)
            out[((size_t(y) * k + x) * I + i) * O + o] = kv.second.data[((size_t(o) * I + i) * k + y) * k + x];
    }
    for (int side = 0; side < 2; ++side) for (const auto& level : blocks_[side]) for (size_t j = 0; j < level.second.size(); ++j) {
        std::string p = std::string(side ? "dec." : "enc.") + level.first + "." + std::to_string(j) + ".ffn";
        int c = dim(p + ".bank.0.0.weight", 1), h = dim(p + ".bank.0.0.weight", 0);
        for (auto part : {std::make_pair("0", "expand"), std::make_pair("2", "narrow")}) {
            int K = part.first == std::string("0") ? c : h, N = part.first == std::string("0") ? h : c;
            auto& bank = cnn_weights_[p + "." + part.second]; bank.resize(size_t(8) * K * N);
            auto& bias = cnn_weights_[p + "." + part.second + "_bias"]; bias.resize(size_t(8) * N);
            for (int e = 0; e < 8; ++e) {
                std::string b = p + ".bank." + std::to_string(e) + "." + part.first;
                pack_weight_kn(w(b + ".weight"), N, K, bank.data() + size_t(e) * K * N);
                std::copy_n(w(b + ".bias"), N, bias.data() + e * N);
            }
        }
    }
}

void Engine::cnn_conv(const float* in, float* out, int width, int height, const std::string& p) const {
    int O = dim(p + ".weight", 0), I = dim(p + ".weight", 1), k = dim(p + ".weight", 2);
    int K = k * k * I, rows = width * height, tile = 1024;
    std::vector<float> columns(size_t(std::min(tile, rows)) * K);
    for (int start = 0; start < rows; start += tile) {
        int n = std::min(tile, rows - start);
#pragma omp parallel for schedule(static)
        for (int r = 0; r < n; ++r) for (int y = 0; y < k; ++y) for (int x = 0; x < k; ++x) {
            int sx = (start + r) % width + x - k / 2, sy = (start + r) / width + y - k / 2;
            float* dst = columns.data() + (size_t(r) * k * k + y * k + x) * I;
            if (sx < 0 || sx >= width || sy < 0 || sy >= height) std::fill_n(dst, I, 0.f);
            else std::copy_n(in + (size_t(sy) * width + sx) * I, I, dst);
        }
        gemm_kn(columns.data(), n, K, O, cnn_weights_.at(p + ".weight").data(), out + size_t(start) * O, w(p + ".bias"));
    }
}

void Engine::cnn_norm(const float* in, float* out, int rows, int c, const std::string& p, bool act) const {
#pragma omp parallel for schedule(static)
    for (int g = 0; g < 8; ++g) {
        int cg = c / 8; double sum = 0, variance = 0;
        for (int row = 0; row < rows; ++row) for (int k = g * cg; k < (g + 1) * cg; ++k) sum += in[size_t(row) * c + k];
        float mean = float(sum / (size_t(rows) * cg));
        for (int row = 0; row < rows; ++row) for (int k = g * cg; k < (g + 1) * cg; ++k) {
            double d = double(in[size_t(row) * c + k]) - mean; variance += d * d;
        }
        float inv = 1.f / std::sqrt(float(variance / (size_t(rows) * cg)) + 1e-5f);
        for (int row = 0; row < rows; ++row) for (int k = g * cg; k < (g + 1) * cg; ++k) {
            float v = (in[size_t(row) * c + k] - mean) * inv * w(p + ".weight")[k] + w(p + ".bias")[k];
            out[size_t(row) * c + k] = act ? activation(v) : v;
        }
    }
}

void Engine::cnn_block(const std::string& p, float* state, int width, int height) const {
    int rows = width * height, c = dim(p + ".conv.weight", 0), h = dim(p + ".ffn.bank.0.0.weight", 0);
    std::vector<float> norm(size_t(rows) * c), conv(norm.size()), logits(size_t(rows) * 8);
    cnn_norm(state, norm.data(), rows, c, p + ".norm1", true); cnn_conv(norm.data(), conv.data(), width, height, p + ".conv");
    for (size_t i = 0; i < conv.size(); ++i) state[i] += conv[i];
    cnn_norm(state, norm.data(), rows, c, p + ".norm2", false);
    cnn_conv(norm.data(), logits.data(), width, height, p + ".ffn.router");
    std::vector<int> pick(rows), offsets(9, 0), order(rows); std::vector<float> gate(rows);
    for (int r = 0; r < rows; ++r) {
        float mx = *std::max_element(logits.data() + r * 8, logits.data() + r * 8 + 8), den = 0, probabilities[8];
        for (int e = 0; e < 8; ++e) { probabilities[e] = std::exp(logits[r * 8 + e] - mx); den += probabilities[e]; }
        int best = 0;
        for (int e = 0; e < 8; ++e) { probabilities[e] /= den; if (probabilities[e] > probabilities[best]) best = e; }
        pick[r] = best; gate[r] = probabilities[best]; ++offsets[best + 1];
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin()); auto cursor = offsets;
    for (int r = 0; r < rows; ++r) order[cursor[pick[r]]++] = r;
    std::vector<float> grouped(norm.size()), hidden(size_t(rows) * h), output(norm.size());
    for (int r = 0; r < rows; ++r) std::copy_n(norm.data() + size_t(order[r]) * c, c, grouped.data() + size_t(r) * c);
    std::string f = p + ".ffn";
    gemm_kn_grouped(grouped.data(), c, h, cnn_weights_.at(f + ".expand").data(), hidden.data(), offsets.data(), 8);
    for (int e = 0; e < 8; ++e) for (int r = offsets[e]; r < offsets[e + 1]; ++r) for (int k = 0; k < h; ++k)
        hidden[size_t(r) * h + k] = activation(hidden[size_t(r) * h + k] + cnn_weights_.at(f + ".expand_bias")[e * h + k]);
    gemm_kn_grouped(hidden.data(), h, c, cnn_weights_.at(f + ".narrow").data(), output.data(), offsets.data(), 8);
    for (int r = 0; r < rows; ++r) for (int k = 0; k < c; ++k) {
        int row = order[r]; state[size_t(row) * c + k] +=
            (output[size_t(r) * c + k] + cnn_weights_.at(f + ".narrow_bias")[pick[row] * c + k]) * gate[row] * w(p + ".aux")[k];
    }
    if (debug_on_) {
        debug_["blk-" + p] = {state, state + size_t(rows) * c};
        debug_["logits-" + f] = logits;
        auto& picks = debug_["pick-" + f]; picks.resize(rows);
        for (int row = 0; row < rows; ++row) picks[row] = float(pick[row]);
        debug_["gate-" + f] = gate;
    }
}

void Engine::cnn_forward(const float* features, float* head) {
    if (int8_) throw std::runtime_error("CNN inference requires FP32; disable NR_INT8");
    const char* names[] = {"full", "d0", "d1", "d2", "d3", "d4", "d5"};
    int widths[7], heights[7];
    for (int i = 0; i < 7; ++i) { widths[i] = i ? geom_.levels[i-1].width : geom_.full_width; heights[i] = i ? geom_.levels[i-1].height : geom_.full_height; }
    std::vector<float> x(size_t(geom_.full_rows) * cnn_channels_[0]), next, skip[6];
    cnn_conv(features, x.data(), widths[0], heights[0], "stem.0"); cnn_norm(x.data(), x.data(), geom_.full_rows, cnn_channels_[0], "stem.1", true);
    for (int i = 0; i < 7; ++i) {
        int rows = widths[i] * heights[i], c = cnn_channels_[i];
        if (i) {
            next.resize(size_t(rows) * cnn_channels_[i-1]); resize_bilinear(x.data(), next.data(), widths[i-1], heights[i-1], widths[i], heights[i], cnn_channels_[i-1]);
            x.resize(size_t(rows) * c); cnn_conv(next.data(), x.data(), widths[i], heights[i], std::string("enc_proj.") + names[i] + ".0");
            cnn_norm(x.data(), x.data(), rows, c, std::string("enc_proj.") + names[i] + ".1", true);
        }
        const auto& bs = blocks_[0].at(names[i]);
        for (size_t j = 0; j < bs.size(); ++j) cnn_block(std::string("enc.") + names[i] + "." + std::to_string(j), x.data(), widths[i], heights[i]);
        if (debug_on_) debug_[i == 6 ? "s-vit" : std::string("s-enc-") + names[i]] = x;
        if (i < 6) skip[i] = x;
    }
    for (int i = 5; i >= 0; --i) {
        int rows = widths[i] * heights[i], ci = cnn_channels_[i+1], c = cnn_channels_[i];
        next.resize(size_t(rows) * ci); resize_bilinear(x.data(), next.data(), widths[i+1], heights[i+1], widths[i], heights[i], ci);
        std::vector<float> cat(size_t(rows) * (ci + c));
        for (int row = 0; row < rows; ++row) {
            std::copy_n(next.data() + size_t(row) * ci, ci, cat.data() + size_t(row) * (ci+c));
            std::copy_n(skip[i].data() + size_t(row) * c, c, cat.data() + size_t(row) * (ci+c) + ci);
        }
        x.resize(size_t(rows) * c); cnn_conv(cat.data(), x.data(), widths[i], heights[i], std::string("merge.") + names[i] + ".0");
        cnn_norm(x.data(), x.data(), rows, c, std::string("merge.") + names[i] + ".1", true);
        const auto& bs = blocks_[1].at(names[i]);
        for (size_t j = 0; j < bs.size(); ++j) cnn_block(std::string("dec.") + names[i] + "." + std::to_string(j), x.data(), widths[i], heights[i]);
        if (debug_on_) debug_[std::string("s-dec-") + names[i]] = x;
    }
    cnn_norm(x.data(), x.data(), geom_.full_rows, cnn_channels_[0], "head_norm", true);
    cnn_conv(x.data(), head, widths[0], heights[0], "head");
    if (debug_on_) debug_["s-head"] = {head, head + size_t(geom_.full_rows) * 4};
}
