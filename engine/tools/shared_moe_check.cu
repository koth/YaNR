// Device-resident routing/grouped GEMM adversarial fixtures and microbench.
// Synthetic values are exact deterministic integer formulas, not random state.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "../src/shared_moe_cuda.cuh"
int repetitions = 100;

void check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
template<class T> struct Device {
    T* p; int n;
    explicit Device(int count) : n(count) { check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void upload(const std::vector<T>& values) { check(cudaMemcpy(p, values.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> download() { std::vector<T> values(n); check(cudaMemcpy(values.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return values; }
};
__global__ void gather_rows(const float* x, float* packed, const int* map, int rows, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * K) packed[map[i / K] * K + i % K] = x[i];
}

void fixture(int width, int height, bool windows, int K, int N) {
    int rows = width * height, nw = (width + 7) / 8;
    int items = windows ? nw * ((height + 7) / 8) : rows;
    std::vector<float> x(rows * K), weight(17 * K * N), bias(16, 0);
    for (int i = 0; i < rows * K; ++i) x[i] = ((i * 17 % 127) - 63) / 128.f;
    for (int i = 0; i < 17 * K * N; ++i) weight[i] = ((i * 19 % 131) - 65) / 512.f;
    Device<float> dx(rows * K), dw(17 * K * N), dl(items * 16), db(16), dg(items), packed(rows * K), out(rows * N);
    Device<float> ffn_routed(rows * N), ffn_shared(rows * N);
    Device<int> pick(items), keys(items), ids(items), counts(16), map(rows), offsets(17), tiles(17);
    dx.upload(x); dw.upload(weight); db.upload(bias);
    cudaStream_t stream; check(cudaStreamCreate(&stream));
    auto launch = [&]() {
        check(cudaMemsetAsync(counts.p, 0, 16 * sizeof(int), stream));
        sr_pick<<<(items + 255) / 256, 256, 0, stream>>>(dl.p, db.p, pick.p, dg.p, keys.p, ids.p, counts.p, items, width, height, windows);
        sr_compact_map<<<16, 256, 0, stream>>>(pick.p, counts.p, map.p, offsets.p, tiles.p, items, width, height, windows);
        gather_rows<<<(rows * K + 255) / 256, 256, 0, stream>>>(dx.p, packed.p, map.p, rows, K);
        dim3 grid((rows + 31) / 32 + 15, (N + 63) / 64);
        if (N == 32) sr_grouped_gemm<32><<<grid, 256, 0, stream>>>(packed.p, dw.p, out.p, offsets.p, tiles.p, K, N, false);
        else sr_grouped_gemm<64><<<grid, 256, 0, stream>>>(packed.p, dw.p, out.p, offsets.p, tiles.p, K, N, false);
        if (K == 32 && N == 32) {
            sr_ffn32<true><<<(rows + 31) / 32 + 15, 256, 0, stream>>>(packed.p, dw.p, dw.p, ffn_routed.p, offsets.p, tiles.p, rows);
            sr_ffn32<false><<<(rows + 31) / 32, 256, 0, stream>>>(dx.p, dw.p, dw.p, ffn_shared.p, nullptr, nullptr, rows);
        }
    };
    // Capture once; rewrite logits between replays to test fresh counts/maps.
    std::vector<float> logits(items * 16, 0); dl.upload(logits);
    launch(); check(cudaStreamSynchronize(stream));
    cudaGraph_t graph; cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); launch();
    check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int mode = 0; mode < 7; ++mode) {
        std::fill(bias.begin(), bias.end(), 0.f);
        if (mode == 6) bias[9] = 2;
        db.upload(bias);
        std::vector<int> expected(items), expected_counts(16, 0), expected_map(rows, -1);
        std::fill(logits.begin(), logits.end(), -2.f);
        for (int i = 0; i < items; ++i) {
            int e = mode == 0 ? i % 16 : mode == 1 ? 7 : mode == 2 ? (i % 10 ? 3 : i % 16) : mode == 3 ? i % 3 : mode == 4 ? 0 : 9;
            expected[i] = e;
            if (mode == 4 || mode == 6) { for (int j = 0; j < 16; ++j) logits[i * 16 + j] = 0; }
            else logits[i * 16 + e] = mode == 5 ? -2.f + 1e-6f : 1.f;
            int n = windows ? std::min(8, width - i % nw * 8) * std::min(8, height - i / nw * 8) : 1;
            expected_counts[e] += n;
        }
        dl.upload(logits);
        check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
        if (pick.download() != expected || counts.download() != expected_counts) throw std::runtime_error("pick/count mismatch");
        auto actual_map = map.download();
        int next = 0;
        for (int e = 0; e < 16; ++e) for (int i = 0; i < items; ++i) if (expected[i] == e) {
            int wx = windows ? i % nw * 8 : i, wy = windows ? i / nw * 8 : 0;
            int cw = windows ? std::min(8, width - wx) : 1, ch = windows ? std::min(8, height - wy) : 1;
            for (int y = 0; y < ch; ++y) for (int xx = 0; xx < cw; ++xx) expected_map[(wy + y) * width + wx + xx] = next++;
        }
        if (next != rows || actual_map != expected_map) throw std::runtime_error("unstable or incomplete compact map");
        auto result = out.download(), gates = dg.download();
        float max_error = 0;
        for (int row = 0; row < rows; ++row) {
            int item = windows ? row / width / 8 * nw + row % width / 8 : row;
            int e = expected[item];
            double denominator = 0; for (int j = 0; j < 16; ++j) denominator += std::exp(double(logits[item * 16 + j]));
            float gate = float(16 * std::exp(double(logits[item * 16 + e])) / denominator);
            if (std::abs(gate - gates[item]) > 2e-6f) throw std::runtime_error("gate mismatch");
            for (int c = 0; c < N; ++c) {
                float value = 0;
                for (int k = 0; k < K; ++k) value = std::fma(x[row * K + k], weight[((e + 1) * K + k) * N + c], value);
                max_error = std::max(max_error, std::abs(result[actual_map[row] * N + c] - value));
            }
        }
        if (max_error != 0) throw std::runtime_error("GEMM differs from scalar ordered FMA");
        if (K == 32 && N == 32) {
            auto routed = ffn_routed.download(), shared = ffn_shared.download();
            for (int row = 0; row < rows; ++row) for (int branch = 0; branch < 2; ++branch) {
                int item = windows ? row / width / 8 * nw + row % width / 8 : row;
                int e = branch ? expected[item] + 1 : 0;
                float hidden[32] = {};
                for (int c = 0; c < 32; ++c) {
                    for (int k = 0; k < 32; ++k) hidden[c] = std::fma(x[row * 32 + k], weight[e * 1024 + k * 32 + c], hidden[c]);
                    hidden[c] = hidden[c] / (1 + std::exp(-hidden[c]));
                }
                for (int c = 0; c < 32; ++c) {
                    float value = 0;
                    for (int k = 0; k < 32; ++k) value = std::fma(hidden[k], weight[e * 1024 + k * 32 + c], value);
                    float actual = branch ? routed[actual_map[row] * 32 + c] : shared[row * 32 + c];
                    if (std::abs(value - actual) > 3e-7f) throw std::runtime_error("fused FFN mismatch");
                }
            }
        }
        auto first = out.download();
        check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
        if (out.download() != first) throw std::runtime_error("repeat differs");
        // Replay microbenchmark covers router selection, compact packing and GEMM.
        cudaEvent_t begin, end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        check(cudaEventRecord(begin, stream));
        for (int i = 0; i < repetitions; ++i) check(cudaGraphLaunch(executable, stream));
        check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
        float ms; check(cudaEventElapsedTime(&ms, begin, end));
        int extra = 0; for (int n : expected_counts) if (n) extra += (32 - n % 32) % 32;
        std::printf("PASS width=%d height=%d windows=%d K=%d N=%d mode=%d %.6f ms alignment_extra_rows=%d\n", width, height, windows, K, N, mode, ms / repetitions, extra);
        cudaEventDestroy(begin); cudaEventDestroy(end);
    }
    cudaGraphExecDestroy(executable); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
}

// Current routed-only banks and CNN's biased pixel FFNs share packing/GEMM,
// but deliberately use different gates. Exercise ties, controller bias,
// empty experts, and K/N tails independently of model argmax sensitivity.
void modern_fixture(int E, int K, int N, bool cnn) {
    const int rows = 117;
    std::vector<float> input(rows * K), bank(E * K * N), bias(E * N), z(rows * E), rb(E);
    for (int i = 0; i < rows * K; ++i) input[i] = (i * 17 % 127 - 63) / 128.f;
    for (int i = 0; i < E * K * N; ++i) bank[i] = (i * 19 % 131 - 65) / 512.f;
    for (int i = 0; i < E * N; ++i) bias[i] = (i * 7 % 17 - 8) / 32.f;
    Device<float> dx(rows * K), dw(E * K * N), db(E * N), dz(rows * E), drb(E), gates(rows), packed(rows * K), output(rows * N);
    Device<int> pick(rows), keys(rows), ids(rows), counts(E), map(rows), offsets(E + 1), tiles(E + 1);
    dx.upload(input); dw.upload(bank); db.upload(bias);
    for (int mode = 0; mode < 6; ++mode) {
        std::fill(rb.begin(), rb.end(), 0.f);
        if (mode == 3 && !cnn) rb[E - 1] = 2;
        for (int i = 0; i < rows; ++i) for (int e = 0; e < E; ++e)
            z[i * E + e] = mode == 0 || mode == 3 ? 0.f : e == (mode == 1 ? E - 1 : i % E) ? 1.f : -2.f;
        if (mode == 4) for (int i = 0; i < rows; ++i) z[i * E + (i + 1) % E] = 1.f - 1e-6f;
        if (mode == 5) for (int i = 0; i < rows; ++i) for (int e = 0; e < E; ++e) z[i * E + e] = e == E - 1 ? 0.f : -1e-8f;
        dz.upload(z); drb.upload(rb); check(cudaMemset(counts.p, 0, E * sizeof(int)));
        sr_pick<<<1, 256>>>(dz.p, cnn ? nullptr : drb.p, pick.p, gates.p, keys.p, ids.p, counts.p, rows, rows, 1, false, E, cnn ? 1.f : float(E), cnn);
        sr_compact_map<<<E, 256>>>(pick.p, counts.p, map.p, offsets.p, tiles.p, rows, rows, 1, false, E);
        gather_rows<<<(rows * K + 255) / 256, 256>>>(dx.p, packed.p, map.p, rows, K);
        dim3 tasks((rows + 31) / 32 + E - 1, (N + 63) / 64);
        sr_grouped_gemm<64><<<tasks, 256>>>(packed.p, dw.p, output.p, offsets.p, tiles.p, K, N, cnn, E, 0, cnn ? db.p : nullptr);
        check(cudaDeviceSynchronize());
        auto picks = pick.download(), mapping = map.download(); auto actual = output.download(), actual_gates = gates.download();
        for (int i = 0; i < rows; ++i) {
            int best = 0;
            for (int e = 1; e < E; ++e) if (z[i * E + e] + rb[e] > z[i * E + best] + rb[best]) best = e;
            if (mode == 5 && cnn) best = 0; // softmax rounds every probability to 1/E
            if (picks[i] != best) throw std::runtime_error("modern pick/tie mismatch");
            double den = 0; for (int e = 0; e < E; ++e) den += std::exp(double(z[i * E + e]));
            float gate = float((cnn ? 1 : E) * std::exp(double(z[i * E + best])) / den);
            if (std::abs(gate - actual_gates[i]) > 2e-6f) throw std::runtime_error("modern gate mismatch");
            for (int c = 0; c < N; ++c) {
                float value = 0;
                for (int k = 0; k < K; ++k) value = std::fma(input[i * K + k], bank[(best * K + k) * N + c], value);
                if (cnn) { value += bias[best * N + c]; value /= 1.f + std::exp(-value); }
                if (std::abs(value - actual[mapping[i] * N + c]) > 3e-7f) throw std::runtime_error("modern GEMM/bias/activation mismatch");
            }
        }
    }
    std::printf("PASS modern experts=%d K=%d N=%d cnn=%d\n", E, K, N, cnn);
}

int main(int argc, char**) {
    if (argc > 1) repetitions = 1; // --quick for compute-sanitizer, correctness unchanged
    try {
        fixture(9, 7, true, 32, 32);     // partial windows, fewer units than experts
        fixture(37, 1, false, 96, 64);   // non-divisible token rows and empty groups
        fixture(320, 320, true, 32, 32);
        fixture(20, 20, true, 192, 192);
        fixture(100, 1, false, 384, 384);
        modern_fixture(7, 32, 32, false);
        modern_fixture(8, 24, 13, true);
        modern_fixture(8, 13, 24, true);
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
