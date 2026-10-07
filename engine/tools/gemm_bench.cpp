// GEMM 微基准(openspec 8.9):分离内核吞吐与量化/反量化开销,形状取自真实层。
//
//   cpu_gemm_bench [--repeats 50]
//
// 每形状报 fp32 与 int8 的 median 毫秒与有效 GMAC/s。
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "gemm.h"

namespace {

double time_ms(void (*fn)(void*), void* ctx, int repeats) {
    std::vector<double> t;
    for (int i = 0; i < repeats + 1; i++) {
        auto t0 = std::chrono::steady_clock::now();
        fn(ctx);
        auto t1 = std::chrono::steady_clock::now();
        if (i > 0) t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

struct Case {
    int M, K, N;
    std::vector<float> A, Bp, C, bias;
    std::vector<int8_t> Wq;
    std::vector<float> sW;
    int Kp;
};

void run_f32(void* p) {
    Case* c = (Case*)p;
    gemm_kn(c->A.data(), c->M, c->K, c->N, c->Bp.data(), c->C.data(), c->bias.data());
}

void run_i8(void* p) {
    Case* c = (Case*)p;
    gemm_kn_i8(c->A.data(), c->M, c->K, c->N, c->Wq.data(), c->Kp,
               c->sW.data(), c->C.data(), c->bias.data());
}

}  // namespace

int main(int argc, char** argv) {
    int repeats = 50;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[++i]);
    }
    const int shapes[][3] = {
        {102400, 32, 32},    // full 级 expand/contract
        {30720, 32, 96},     // d0 级 qkv
        {25600, 32, 32},     // d0 级 expand/contract
        {18432, 64, 192},    // d1 级 qkv
        {1600, 128, 384},    // d2 级 qkv(320²)
        {1152, 256, 256},    // d3 级 contract
        {1152, 256, 768},    // d3 级 qkv
        {400, 256, 768},     // d3 级 qkv(320²)
        {144, 256, 256},     // d4 级 contract(320²)
        {64, 512, 1024},     // vit expand
        {64, 512, 1536},     // vit qkv
    };
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (const auto& sh : shapes) {
        Case c{sh[0], sh[1], sh[2], {}, {}, {}, {}, {}, {}, 0};
        c.A.resize((size_t)c.M * c.K);
        c.Bp.resize((size_t)c.N * c.K);
        c.C.resize((size_t)c.M * c.N);
        c.bias.resize(c.N);
        for (auto& v : c.A) v = dist(rng);
        for (auto& v : c.Bp) v = dist(rng) * 0.1f;
        for (auto& v : c.bias) v = dist(rng) * 0.01f;
        c.Kp = (c.K + 15) & ~15;
        c.Wq.resize((size_t)c.N * c.Kp);
        c.sW.resize(c.N);
        // 从同一份 [N][K] 权重准备两种布局(打包/量化都不计时)
        std::vector<float> Wf((size_t)c.N * c.K);
        for (auto& v : Wf) v = dist(rng) * 0.1f;
        pack_weight_kn(Wf.data(), c.N, c.K, c.Bp.data());
        quantize_w_nk(Wf.data(), c.N, c.K, c.Kp, c.Wq.data(), c.sW.data());

        double tf = time_ms(run_f32, &c, repeats);
        double ti = time_ms(run_i8, &c, repeats);
        double macs = (double)c.M * c.K * c.N / 1e9;
        std::printf("M=%6d K=%4d N=%4d | fp32 %7.3f ms %7.1f GMAC/s | i8 %7.3f ms %7.1f GMAC/s | x%.2f\n",
                    c.M, c.K, c.N, tf, macs / (tf / 1e3), ti, macs / (ti / 1e3), tf / ti);
    }
    return 0;
}
