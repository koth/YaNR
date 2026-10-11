// Regression for tiled FFNs, expert-major output, in-place residuals, N=4 head,
// and field padding/tail channels. Run on both NR_AVX2=ON and OFF builds.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#include <omp.h>
#include "gemm.h"

namespace {
std::mt19937 rng(7);
std::vector<float> random_values(size_t n, float scale = 1) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
}
void exact(const std::vector<float>& a, const std::vector<float>& b, const char* tag) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)))
        throw std::runtime_error(tag);
}
void check_ffn(int rows, int channels, int hidden, int outputs, int experts, bool residual) {
    auto input = random_values(static_cast<size_t>(rows) * channels);
    auto we = random_values(static_cast<size_t>(experts) * channels * hidden, 0.1f);
    auto wc = random_values(static_cast<size_t>(experts) * hidden * outputs, 0.1f);
    auto be = random_values(experts * hidden, 0.01f), bc = random_values(experts * outputs, 0.01f);
    auto aux = random_values(channels);
    std::vector<float> ref(static_cast<size_t>(experts) * rows * outputs), got(ref.size());
    std::vector<float> temp(static_cast<size_t>(rows) * hidden);
    for (int e = 0; e < experts; ++e) {
        gemm_kn(input.data(), rows, channels, hidden, we.data() + e * channels * hidden,
                temp.data(), be.data() + e * hidden, true);
        gemm_kn(temp.data(), rows, hidden, outputs, wc.data() + e * hidden * outputs,
                ref.data() + static_cast<size_t>(e) * rows * outputs, bc.data() + e * outputs,
                false, residual ? input.data() : nullptr, residual ? aux.data() : nullptr);
    }
    ffn_kn(input.data(), rows, channels, hidden, we.data(), be.data(), wc.data(), bc.data(),
           residual ? aux.data() : nullptr, got.data(), outputs, experts);
    exact(ref, got, "fused FFN / expert layout mismatch");
    if (residual) {
        auto inplace = input;
        ffn_kn(inplace.data(), rows, channels, hidden, we.data(), be.data(), wc.data(), bc.data(),
               aux.data(), inplace.data());
        exact(ref, inplace, "in-place FFN mismatch");
    }
}
void check_head(int rows, int k, bool bias) {
    auto a = random_values(static_cast<size_t>(rows) * k);
    auto b = random_values(k * 4, 0.1f), bs = random_values(4, 0.01f);
    std::vector<float> out(rows * 4);
    gemm_kn(a.data(), rows, k, 4, b.data(), out.data(), bias ? bs.data() : nullptr);
    for (int r = 0; r < rows; ++r) for (int c = 0; c < 4; ++c) {
        double ref = bias ? bs[c] : 0;
        for (int i = 0; i < k; ++i) ref += static_cast<double>(a[r * k + i]) * b[i * 4 + c];
        if (std::abs(out[r * 4 + c] - ref) > 1e-5 * (1 + std::abs(ref)))
            throw std::runtime_error("four-column head mismatch");
    }
}
void check_grouped(int k, int n, bool act, bool mapped) {
    // Empty experts, 1/2/3-row SIMD tails, uneven groups, row tiles and column
    // tails occur together. The mapped case leaves holes to detect bad stores.
    const std::vector<int> offsets = {0, 0, 1, 3, 6, 10, 15, 48, 81};
    const int rows = offsets.back(), experts = static_cast<int>(offsets.size()) - 1;
    auto a = random_values(static_cast<size_t>(rows) * k);
    auto w = random_values(static_cast<size_t>(experts) * k * n, 0.1f);
    std::vector<int> row_map(rows);
    for (int r = 0; r < rows; ++r) row_map[r] = rows - r + 1;
    std::vector<float> out(static_cast<size_t>(rows + 3) * n, 123.0f);
    gemm_kn_grouped(a.data(), k, n, w.data(), out.data(), offsets.data(), experts,
                    act, mapped ? row_map.data() : nullptr);
    for (int e = 0; e < experts; ++e)
        for (int r = offsets[e]; r < offsets[e + 1]; ++r)
            for (int c = 0; c < n; ++c) {
                double ref = 0;
                for (int i = 0; i < k; ++i)
                    ref += static_cast<double>(a[static_cast<size_t>(r) * k + i])
                         * w[(static_cast<size_t>(e) * k + i) * n + c];
                if (act) ref /= 1 + std::exp(-ref);
                float got = out[static_cast<size_t>(mapped ? row_map[r] : r) * n + c];
                if (!std::isfinite(got) || std::abs(got - ref) > 2e-5 * (1 + std::abs(ref)))
                    throw std::runtime_error("grouped MoE GEMM / short rows / map mismatch");
            }
    for (int r : (mapped ? std::vector<int>{0, 1, rows + 2}
                         : std::vector<int>{rows, rows + 1, rows + 2}))
        for (int c = 0; c < n; ++c)
            if (out[static_cast<size_t>(r) * n + c] != 123.0f)
                throw std::runtime_error("grouped MoE GEMM overwrote an unmapped row");
}
void check_field(int iw, int ih, int ow, int oh, int channels) {
    auto x = random_values(static_cast<size_t>(iw) * ih * channels);
    std::vector<float> ref(static_cast<size_t>(ow) * oh * channels), got(ref.size());
    for (int y = 0; y < oh; ++y) for (int xx = 0; xx < ow; ++xx) {
        int sy = std::min(2 * y, ih - 1), sx = std::min(2 * xx, iw - 1);
        for (int c = 0; c < channels; ++c) {
            size_t index = (static_cast<size_t>(sy) * iw + sx) * channels + c;
            ref[(static_cast<size_t>(y) * ow + xx) * channels + c] =
                sy + 1 < ih && sx + 1 < iw ?
                (x[index] + x[index + channels] + x[index + iw * channels] +
                 x[index + (iw + 1) * channels]) * 0.25f : 0;
        }
    }
    field_downsample2(x.data(), got.data(), iw, ih, ow, oh, channels);
    exact(ref, got, "downsample padding/channel tail mismatch");
    auto skip = random_values(ref.size()), aux0 = random_values(channels), aux1 = random_values(channels);
    for (bool weighted : {false, true}) {
        for (int y = 0; y < oh; ++y) for (int xx = 0; xx < ow; ++xx) {
            int sy = std::min(y / 2, ih - 1), sx = std::min(xx / 2, iw - 1);
            for (int c = 0; c < channels; ++c) {
                size_t src = (static_cast<size_t>(sy) * iw + sx) * channels + c;
                size_t dst = (static_cast<size_t>(y) * ow + xx) * channels + c;
                ref[dst] = (weighted ? x[src] * aux0[c] : x[src]) + skip[dst] * aux1[c];
            }
        }
        field_upsample_merge(x.data(), skip.data(), weighted ? aux0.data() : nullptr, aux1.data(),
                             got.data(), iw, ih, ow, oh, channels);
        exact(ref, got, "upsample merge/channel tail mismatch");
    }
}
}
int main() {
    try {
        omp_set_dynamic(0); omp_set_num_threads(8);
        for (int rows : {1, 3, 4, 7, 8, 31, 32, 33, 257}) {
            for (int channels : {4, 16, 32, 64}) for (int hidden : {7, 16, 32, 48})
                check_ffn(rows, channels, hidden, channels, 1, true);
            check_ffn(rows, 64, 48, 32, 2, false);
            check_ffn(rows, 192, 64, 32, 6, false);
        }
        check_ffn(2053, 64, 64, 32, 4, false); // parallel + last tile
        for (int rows : {3, 8, 17, 65, 1024}) for (int k : {16, 32, 64, 256})
            for (bool bias : {false, true}) check_head(rows, k, bias);
        for (bool act : {false, true}) for (bool mapped : {false, true}) {
            check_grouped(32, 32, act, mapped);
            check_grouped(37, 29, act, mapped);
            check_grouped(512, 1024, act, mapped);
        }
        for (int c : {3, 7, 8, 32, 64}) {
            check_field(3, 5, 4, 4, c);
            check_field(16, 12, 8, 8, c);
            check_field(13, 17, 28, 36, c);
        }
        check_field(320, 320, 160, 160, 32);
        std::puts("PASS: FFN/expert fusion, grouped MoE short rows/maps, in-place/tails, head, field padding/merge");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
