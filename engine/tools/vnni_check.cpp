// Packed INT8 reference runs everywhere. Native ISA checks are conditional;
// --require-vnni rejects a run that cannot actually execute either VNNI kernel.
#include "gemm.h"
#include "gemm_vnni.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace {
std::mt19937 rng(19);
void select(const char* mode) {
#ifdef _WIN32
    _putenv_s("NR_VNNI", mode);
#else
    setenv("NR_VNNI", mode, 1);
#endif
}
void require(bool condition, const char* tag) {
    if (!condition) throw std::runtime_error(tag);
}
std::vector<float> floats(size_t count, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(count);
    for (float& x : v) x = dist(rng);
    return v;
}
void exact(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float)),
            "INT8 GEMM backend/fusion mismatch");
}

void check_panel(int N, int K, int Kp) {
    std::uniform_int_distribution<int> dist(-128, 127);
    std::vector<int8_t> a(4 * Kp), w(static_cast<size_t>(N) * Kp);
    for (int r = 0; r < 4; ++r) for (int k = 0; k < K; ++k) a[r * Kp + k] = static_cast<int8_t>(dist(rng));
    for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) w[n * Kp + k] = static_cast<int8_t>(dist(rng));
    // Extreme products, nonzero correction, and a zero activation row.
    std::fill(a.begin(), a.begin() + K, int8_t(-128));
    std::fill(a.begin() + Kp, a.begin() + Kp + K, int8_t(127));
    std::fill(a.begin() + 2 * Kp, a.begin() + 3 * Kp, int8_t(0));
    std::fill(w.begin(), w.begin() + K, int8_t(127));
    const auto packed = pack_weight_i8_vnni(w.data(), N, Kp);
    const int Np = (N + 15) & ~15;
    for (int n0 = 0; n0 < Np; n0 += 16) {
        int32_t ref[64] = {}, corrected[64] = {};
        const int8_t* panel = packed.w.data() + static_cast<size_t>(n0) * Kp;
        for (int r = 0; r < 4; ++r) for (int n = 0; n < 16; ++n) {
            int32_t sum = packed.correction[n0 + n];
            for (int k = 0; k < Kp; ++k) {
                const int8_t pw = panel[(k / 4) * 64 + n * 4 + k % 4];
                const int8_t expected = n0 + n < N ? w[(n0 + n) * Kp + k] : int8_t(0);
                require(pw == expected, "VNNI packing/padded column mismatch");
                sum += (static_cast<int>(a[r * Kp + k]) + 128) * pw;
                if (n0 + n < N) ref[r * 16 + n] += static_cast<int>(a[r * Kp + k]) * expected;
            }
            corrected[r * 16 + n] = sum;
        }
        require(!std::memcmp(ref, corrected, sizeof(ref)), "VNNI signed offset correction mismatch");
        for (VnniIsa isa : {VnniIsa::avx, VnniIsa::avx512}) {
            if (VnniDot dot = vnni_kernel(isa)) {
                int32_t native[64];
                dot(a.data(), panel, packed.correction.data() + n0, Kp, native);
                require(!std::memcmp(ref, native, sizeof(ref)), "native VNNI integer dot mismatch");
            }
        }
    }
}

void check_gemm(int M, int K, int N, const std::vector<std::string>& modes) {
    const int Kp = (K + 15) & ~15;
    auto a = floats(static_cast<size_t>(M) * K, 2.0f), w = floats(static_cast<size_t>(N) * K, 0.1f);
    auto bias = floats(N, 0.02f), residual = floats(static_cast<size_t>(M) * N, 0.5f), aux = floats(N, 0.3f);
    std::fill(a.begin(), a.begin() + K, 0.0f); // per-row zero scale case
    std::vector<int8_t> wq(static_cast<size_t>(N) * Kp);
    std::vector<float> sw(N), ref(static_cast<size_t>(M) * N), got(ref.size());
    quantize_w_nk(w.data(), N, K, Kp, wq.data(), sw.data());
    const auto packed = pack_weight_i8_vnni(wq.data(), N, Kp);
    for (int epilogue = 0; epilogue < 3; ++epilogue) for (bool with_bias : {false, true}) {
        const float* bs = with_bias ? bias.data() : nullptr;
        const float* xr = epilogue == 2 ? residual.data() : nullptr;
        const float* ax = epilogue == 2 ? aux.data() : nullptr;
        select("0");
        gemm_kn_i8(a.data(), M, K, N, wq.data(), Kp, sw.data(), ref.data(), bs, epilogue == 1, xr, ax);
        for (const auto& mode : modes) {
            select(mode.c_str());
            gemm_kn_i8(a.data(), M, K, N, wq.data(), Kp, sw.data(), got.data(), bs, epilogue == 1, xr, ax, &packed);
            exact(ref, got);
        }
    }
}

void check_inplace(int M, int N, const std::vector<std::string>& modes) {
    const int Kp = (N + 15) & ~15;
    auto a = floats(static_cast<size_t>(M) * N, 2.0f), w = floats(N * N, 0.1f);
    auto aux = floats(N, 0.3f), bias = floats(N, 0.1f);
    std::vector<int8_t> wq(static_cast<size_t>(N) * Kp);
    std::vector<float> sw(N), ref(a.size());
    quantize_w_nk(w.data(), N, N, Kp, wq.data(), sw.data());
    const auto packed = pack_weight_i8_vnni(wq.data(), N, Kp);
    select("0");
    gemm_kn_i8(a.data(), M, N, N, wq.data(), Kp, sw.data(), ref.data(), bias.data(), false, a.data(), aux.data());
    for (const auto& mode : modes) {
        auto got = a;
        select(mode.c_str());
        gemm_kn_i8(got.data(), M, N, N, wq.data(), Kp, sw.data(), got.data(), bias.data(), false,
                   got.data(), aux.data(), &packed);
        exact(ref, got);
    }
}
}

int main(int argc, char** argv) {
    try {
        const bool require_vnni = argc == 2 && std::strcmp(argv[1], "--require-vnni") == 0;
        if (argc > 1 && !require_vnni) throw std::runtime_error("usage: vnni_check [--require-vnni]");
        omp_set_dynamic(0); omp_set_num_threads(4);
        const bool avx = vnni_kernel(VnniIsa::avx) != nullptr, avx512 = vnni_kernel(VnniIsa::avx512) != nullptr;
        std::printf("VNNI executable on this CPU: AVX=%s AVX512=%s\n", avx ? "yes" : "no", avx512 ? "yes" : "no");
        require(!require_vnni || avx || avx512, "no executable VNNI kernel on this CPU/build");
        std::vector<std::string> modes = {"0", "auto", "avx", "avx512"};
        select("0"); require(!gemm_vnni_available(), "NR_VNNI=0 dispatch mismatch");
        select("avx"); require(gemm_vnni_available() == avx, "AVX-VNNI runtime guard mismatch");
        select("avx512"); require(gemm_vnni_available() == avx512, "AVX512 VNNI runtime guard mismatch");
        for (int N : {1, 4, 7, 15, 16, 17, 31, 32, 33, 129})
            for (int K : {1, 15, 16, 17, 31, 64, 256, 513}) check_panel(N, K, (K + 15) & ~15);
        const int cases[][3] = {{1,15,3}, {3,17,17}, {4,32,16}, {47,31,33},
                               {48,64,32}, {49,256,16}, {97,257,129}, {129,512,128}};
        for (const auto& c : cases) check_gemm(c[0], c[1], c[2], modes);
        for (int N : {16, 33, 256}) check_inplace(49, N, modes);
        std::puts("PASS: INT8 packing/correction, zero rows, row/column/K tails, bias/SiLU/residual/in-place, dispatch");
        if (!avx && !avx512) std::puts("SKIP: native VNNI instructions (CPU/build unavailable)");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
