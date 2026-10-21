// Exercise parallel two-pass GroupNorm (including aliasing/low variance) and
// compact pixel packing independently of model routing and image quantization.
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../src/cnn_cuda.cuh"
#include "../src/shared_moe_cuda.cuh"

static void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Device() { cudaFree(p); }
};
static unsigned next(unsigned& seed) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed;
}
static void norm(int rows, int channels, int distribution, bool act, bool alias, bool coalesced) {
    size_t n = size_t(rows) * channels, count = n / 8;
    int partial_size = coalesced ? 512 : 4096;
    int parts = int((count + partial_size - 1) / partial_size);
    std::vector<float> input(n), gamma(channels), beta(channels), ref(n), got(n);
    unsigned seed = 811;
    for (float& v : input) {
        float noise = (int(next(seed) % 20001) - 10000) / 10000.f;
        v = distribution == 0 ? noise : distribution == 1 ? 8192.f + noise * .01f : 7.f;
    }
    for (int c = 0; c < channels; ++c) { gamma[c] = .8f + .01f * c; beta[c] = -.1f + .001f * c; }
    Device<float> x(n), y(n), reference(n), g(channels), b(channels), stats(16);
    Device<double> partial(size_t(parts) * 8);
    check(cudaMemcpy(x.p, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(g.p, gamma.data(), channels * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(b.p, beta.data(), channels * sizeof(float), cudaMemcpyHostToDevice));
    cnn_group_norm<<<8, 256>>>(x.p, reference.p, rows, channels, g.p, b.p, act);
    auto launch_partial = [&](bool variance) {
        if (coalesced) {
            switch (channels) {
#define CHECK_NORM(C) case C: \
                if (variance) cnn_norm_partial_coalesced<C, true><<<parts, 256>>>(x.p, partial.p, stats.p, rows, parts); \
                else cnn_norm_partial_coalesced<C, false><<<parts, 256>>>(x.p, partial.p, stats.p, rows, parts); \
                break
                CHECK_NORM(32); CHECK_NORM(64); CHECK_NORM(128); CHECK_NORM(256);
#undef CHECK_NORM
            default: throw std::runtime_error("unsupported coalesced fixture");
            }
        } else {
            if (variance) cnn_norm_partial<true><<<dim3(parts, 8), 256>>>(x.p, partial.p, stats.p, rows, channels, parts);
            else cnn_norm_partial<false><<<dim3(parts, 8), 256>>>(x.p, partial.p, stats.p, rows, channels, parts);
        }
    };
    launch_partial(false);
    cnn_norm_stats<false><<<8, 256>>>(partial.p, stats.p, parts, count);
    launch_partial(true);
    cnn_norm_stats<true><<<8, 256>>>(partial.p, stats.p, parts, count);
    float* output = alias ? x.p : y.p;
    cnn_norm_apply<<<int((n + 255) / 256), 256>>>(x.p, output, stats.p, g.p, b.p, rows, channels, act);
    check(cudaMemcpy(ref.data(), reference.p, n * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(got.data(), output, n * sizeof(float), cudaMemcpyDeviceToHost));
    float maximum = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i])) throw std::runtime_error("GroupNorm nonfinite");
        maximum = std::max(maximum, std::abs(got[i] - ref[i]));
    }
    if (maximum > 1e-6f) throw std::runtime_error("GroupNorm differs from two-pass reference");
    std::printf("norm rows=%d channels=%d distribution=%d act=%d alias=%d max=%g\n",
                rows, channels, distribution, act, alias, maximum);
}
static void pack(int rows, int channels) {
    size_t n = size_t(rows) * channels;
    std::vector<int> ids(rows), map(rows);
    std::vector<float> input(n), got(n);
    for (int i = 0; i < rows; ++i) ids[i] = rows - i - 1;
    for (size_t i = 0; i < n; ++i) input[i] = float(i);
    Device<int> order(rows), inverse(rows);
    Device<float> x(n), y(n);
    check(cudaMemcpy(order.p, ids.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(x.p, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    cnn_pack_pixels<<<int((n + 255) / 256), 256>>>(x.p, y.p, inverse.p, order.p, rows, channels);
    check(cudaMemcpy(map.data(), inverse.p, rows * sizeof(int), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(got.data(), y.p, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int r = 0; r < rows; ++r) {
        if (map[ids[r]] != r) throw std::runtime_error("pixel inverse permutation");
        for (int c = 0; c < channels; ++c)
            if (got[size_t(r) * channels + c] != input[size_t(ids[r]) * channels + c])
                throw std::runtime_error("pixel pack");
    }
}
static void pick(int rows, bool ties) {
    std::vector<float> logits(size_t(rows) * 8), expected_gates(rows), gates(rows);
    std::vector<int> expected(rows), actual(rows), counts(8), reference_counts(8);
    unsigned seed = 17;
    for (float& v : logits) v = (int(next(seed) % 1001) - 500) * .01f;
    if (ties) for (int row = 0; row < rows; ++row) {
        for (int e = 0; e < 8; ++e) logits[size_t(row) * 8 + e] = -1e-8f;
        logits[size_t(row) * 8 + 7] = 0;  // probabilities round to ties
    }
    Device<float> x(logits.size()), g0(rows), g1(rows);
    Device<int> p0(rows), p1(rows), keys(rows), ids(rows), c0(8), c1(8);
    check(cudaMemcpy(x.p, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemset(c0.p, 0, 8 * sizeof(int))); check(cudaMemset(c1.p, 0, 8 * sizeof(int)));
    sr_pick<<<(rows + 255) / 256, 256>>>(x.p, nullptr, p0.p, g0.p, keys.p, ids.p, c0.p, rows, rows, 1, false, 8, 1.f, true);
    cnn_pick_pixels<<<(rows + 255) / 256, 256>>>(x.p, p1.p, g1.p, keys.p, ids.p, c1.p, rows);
    check(cudaMemcpy(expected.data(), p0.p, rows * sizeof(int), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(), p1.p, rows * sizeof(int), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(expected_gates.data(), g0.p, rows * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(gates.data(), g1.p, rows * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(reference_counts.data(), c0.p, 8 * sizeof(int), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(counts.data(), c1.p, 8 * sizeof(int), cudaMemcpyDeviceToHost));
    if (expected != actual || expected_gates != gates || counts != reference_counts)
        throw std::runtime_error("pixel routing/gate/count mismatch");
}
static void ffn(int rows, bool collapsed) {
    std::vector<float> input(size_t(rows) * 32), expand(8 * 2048), narrow(expand.size()), b1(8 * 64), b2(8 * 32);
    unsigned seed = 31;
    for (auto* data : {&input, &expand, &narrow, &b1, &b2})
        for (float& v : *data) v = (int(next(seed) % 2001) - 1000) * .0002f;
    std::vector<int> offsets(9), tiles(9);
    for (int e = 0; e < 8; ++e) {
        int count = collapsed ? (e == 7 ? rows : 0) : rows / 8 + (e < rows % 8);
        offsets[e + 1] = offsets[e] + count;
        tiles[e + 1] = tiles[e] + (count + 31) / 32;
    }
    Device<float> x(input.size()), w1(expand.size()), w2(narrow.size()), bias1(b1.size()), bias2(b2.size());
    Device<float> hidden(size_t(rows) * 64), ref(size_t(rows) * 32), actual(size_t(rows) * 32);
    Device<int> os(9), ts(9);
    check(cudaMemcpy(x.p, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(w1.p, expand.data(), expand.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(w2.p, narrow.data(), narrow.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(bias1.p, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(bias2.p, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(os.p, offsets.data(), 9 * sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(ts.p, tiles.data(), 9 * sizeof(int), cudaMemcpyHostToDevice));
    int blocks = (rows + 31) / 32 + 7;
    sr_grouped_gemm<64><<<blocks, 256>>>(x.p, w1.p, hidden.p, os.p, ts.p, 32, 64, true, 8, 0, bias1.p);
    sr_grouped_gemm<64><<<blocks, 256>>>(hidden.p, w2.p, ref.p, os.p, ts.p, 64, 32, false, 8, 0, bias2.p);
    cnn_ffn32_64<<<blocks, 256>>>(x.p, w1.p, w2.p, bias1.p, bias2.p, actual.p, os.p, ts.p);
    std::vector<float> expected(input.size()), got(input.size());
    check(cudaMemcpy(expected.data(), ref.p, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(got.data(), actual.p, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (std::memcmp(expected.data(), got.data(), got.size() * sizeof(float)))
        throw std::runtime_error("fused FFN differs from ordered FP32 GEMMs");
    std::vector<int> permutation(rows), inverse(rows);
    std::vector<float> gate(rows), aux(32);
    for (int r = 0; r < rows; ++r) { permutation[r] = rows - r - 1; inverse[r] = rows - r - 1; gate[r] = .1f + .8f * (r % 11) / 11; }
    for (int c = 0; c < 32; ++c) aux[c] = .7f + c * .01f;
    Device<int> perm(rows), inv(rows);
    Device<float> gates(rows), scale(32), state0(input.size()), state1(input.size());
    check(cudaMemcpy(perm.p, permutation.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(inv.p, inverse.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(gates.p, gate.data(), rows * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(scale.p, aux.data(), 32 * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(state0.p, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(state1.p, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    cnn_ffn_add<<<int((input.size() + 255) / 256), 256>>>(state0.p, ref.p, inv.p, gates.p, scale.p, rows, 32);
    cnn_ffn32_64<<<blocks, 256>>>(x.p, w1.p, w2.p, bias1.p, bias2.p, state1.p, os.p, ts.p, perm.p, gates.p, scale.p);
    check(cudaMemcpy(expected.data(), state0.p, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(got.data(), state1.p, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (std::memcmp(expected.data(), got.data(), got.size() * sizeof(float)))
        throw std::runtime_error("fused FFN scatter/gate/residual");
}
template<int C> static void im2col() {
    constexpr int width = 37, height = 29, count = width * height;
    std::vector<float> input(size_t(count) * C), expected(size_t(count) * C * 9), actual(expected.size());
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(i);
    Device<float> x(input.size()), ref(expected.size()), out(actual.size());
    check(cudaMemcpy(x.p, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    cnn_im2col<<<int((expected.size() + 255) / 256), 256>>>(x.p, ref.p, 0, count, width, height, C, 3);
    cnn_im2col3_vec<C><<<int((expected.size() / 4 + 255) / 256), 256>>>(
        reinterpret_cast<const float4*>(x.p), reinterpret_cast<float4*>(out.p), 0, count, width, height);
    check(cudaMemcpy(expected.data(), ref.p, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(), out.p, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (expected != actual) throw std::runtime_error("vectorized im2col padding/layout");
}
int main() {
    try {
        const std::pair<int, int> shapes[] = {{1, 24}, {137, 32}, {4097, 24}, {32771, 64}, {70003, 32}, {1057, 128}, {517, 256}, {33, 480}};
        for (auto shape : shapes)
            for (int distribution = 0; distribution < 3; ++distribution)
                for (bool act : {false, true}) {
                    norm(shape.first, shape.second, distribution, act, distribution != 0, false);
                    if (shape.second == 32 || shape.second == 64 || shape.second == 128 || shape.second == 256)
                        norm(shape.first, shape.second, distribution, act, distribution != 0, true);
                }
        for (int rows : {1, 257, 4097}) for (int channels : {24, 32, 480}) pack(rows, channels);
        for (int rows : {1, 33, 257, 4097, 10003}) for (bool ties : {false, true}) pick(rows, ties);
        for (int rows : {1, 33, 257, 4097}) for (bool collapsed : {false, true}) ffn(rows, collapsed);
        im2col<16>(); im2col<32>(); im2col<96>(); im2col<480>(); im2col<736>();
        check(cudaDeviceSynchronize());
        std::puts("CNN CUDA CHECK PASS");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
