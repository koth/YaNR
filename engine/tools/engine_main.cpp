// cpu_engine: run the student forward from a features dump (parity/bench tool).
//
//   cpu_engine --idx student512.idx --features feat.bin --out head.bin
//   cpu_engine --idx student512.idx --features feat.bin --bench --repeats 30
//
// features: raw f32 [full_rows][16]; head: raw f32 [full_rows][4].
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine.h"
#include "png_write.h"
#include "../src/attn.h"

int main(int argc, char** argv) {
    std::string idx_path, feat_path, out_path, dump_dir;
    std::string proxy_path, png_path, blend_path;
    int repeats = 30;
    bool bench = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return argv[++i]; };
        if (a == "--idx") idx_path = next();
        else if (a == "--features") feat_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--dump-dir") dump_dir = next();
        else if (a == "--proxy") proxy_path = next();      // 有效区 proxy raw f32 [h][w][3]
        else if (a == "--png") png_path = next();          // composite PNG 输出(8.4)
        else if (a == "--blend-png") blend_path = next();  // blend 权重图 PNG
        else if (a == "--bench") bench = true;
        else if (a == "--repeats") repeats = std::atoi(next());
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (idx_path.empty() || feat_path.empty() || (!bench && out_path.empty())) {
        std::fprintf(stderr, "usage: cpu_engine --idx m.idx --features f.bin --out h.bin | --bench\n");
        return 2;
    }

    Engine engine(idx_path);
    const Geometry& g = engine.geometry();
    std::vector<float> features((size_t)g.full_rows * 16);
    FILE* f = std::fopen(feat_path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", feat_path.c_str()); return 1; }
    size_t got = std::fread(features.data(), 4, features.size(), f);
    std::fclose(f);
    if (got != features.size()) {
        std::fprintf(stderr, "features size mismatch: %zu / %zu\n", got, features.size());
        return 1;
    }
    std::vector<float> head((size_t)g.full_rows * 4);
    engine.set_debug(!dump_dir.empty());
    engine.set_profile(std::getenv("NR_PROFILE") != nullptr);
    engine.set_int8(std::getenv("NR_INT8") != nullptr);

    if (bench) {
        engine.forward(features.data(), head.data());                 // warmup
        engine.set_profile(std::getenv("NR_PROFILE") != nullptr);     // 清零,只统计计时段
        attn_profile_reset();
        std::vector<double> times;
        for (int i = 0; i < repeats; i++) {
            auto t0 = std::chrono::steady_clock::now();
            engine.forward(features.data(), head.data());
            auto t1 = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(times.begin(), times.end());
        std::printf("cpu_engine %dx%d  median %.2f ms  min %.2f ms  p90 %.2f ms\n",
                    g.full_width, g.full_height, times[times.size() / 2],
                    times.front(), times[(int)(times.size() * 0.9)]);
        if (std::getenv("NR_PROFILE")) {
            const double* p = engine.profile();
            const double* ap = attn_profile();
            double inv = 1.0 / repeats;
            std::printf("profile:  gemm %6.2f ms  silu %6.2f ms  attn %6.2f ms "
                        "(gather %5.2f)  other %6.2f ms\n",
                        p[0] * inv, p[1] * inv, p[2] * inv, p[3] * inv,
                        times[times.size() / 2] - (p[0] + p[1] + p[2]) * inv);
            std::printf("attn-core(累加口径): norms %6.2f  scores %6.2f  softmax %6.2f "
                        " av %6.2f\n", ap[0] * inv, ap[1] * inv, ap[2] * inv, ap[3] * inv);
        }
    } else {
        auto t0 = std::chrono::steady_clock::now();
        engine.forward(features.data(), head.data());
        auto t1 = std::chrono::steady_clock::now();
        FILE* o = std::fopen(out_path.c_str(), "wb");
        std::fwrite(head.data(), 4, head.size(), o);
        std::fclose(o);
        for (const auto& kv : engine.debug()) {
            std::string path = dump_dir + "/" + kv.first + ".bin";
            FILE* d = std::fopen(path.c_str(), "wb");
            std::fwrite(kv.second.data(), 4, kv.second.size(), d);
            std::fclose(d);
        }
        std::printf("wrote %s  forward %.2f ms  blend_scale %.4f\n", out_path.c_str(),
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    engine.blend_scale());
    }

    // composite + PNG + 统计段(8.4):neural = clamp(proxy + head_rgb/4),
    // blend = clamp(sigmoid(logit) * blend_scale)。口径同 run_image/run_student。
    if (!proxy_path.empty()) {
        const Geometry& gg = engine.geometry();
        const int vs = engine.valid_size();
        std::vector<float> proxy((size_t)vs * vs * 3);
        FILE* pf = std::fopen(proxy_path.c_str(), "rb");
        if (!pf) { std::fprintf(stderr, "cannot open %s\n", proxy_path.c_str()); return 1; }
        size_t got = std::fread(proxy.data(), 4, proxy.size(), pf);
        std::fclose(pf);
        if (got != proxy.size()) {
            std::fprintf(stderr, "proxy size mismatch: %zu / %zu\n", got, proxy.size());
            return 1;
        }
        std::vector<uint8_t> rgb((size_t)vs * vs * 3), bw((size_t)vs * vs * 3);
        double rmin = 1e30, rmax = -1e30, rsum = 0.0, lmin = 1e30, lmax = -1e30, bsum = 0.0;
        for (int y = 0; y < vs; y++) {
            for (int x = 0; x < vs; x++) {
                const float* hv = head.data() + ((size_t)y * gg.full_width + x) * 4;
                for (int c = 0; c < 3; c++) {
                    double r = hv[c] / 4.0;
                    double v = proxy[((size_t)y * vs + x) * 3 + c] + r;
                    v = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
                    rgb[((size_t)y * vs + x) * 3 + c] = (uint8_t)(v * 255.0 + 0.5);
                    if (r < rmin) rmin = r;
                    if (r > rmax) rmax = r;
                    rsum += r;
                }
                double lg = hv[3];
                double bl = 1.0 / (1.0 + std::exp(-lg)) * engine.blend_scale();
                bl = bl < 0.0 ? 0.0 : (bl > 1.0 ? 1.0 : bl);
                uint8_t b = (uint8_t)(bl * 255.0 + 0.5);
                for (int c = 0; c < 3; c++) bw[((size_t)y * vs + x) * 3 + c] = b;
                if (lg < lmin) lmin = lg;
                if (lg > lmax) lmax = lg;
                bsum += bl;
            }
        }
        double n3 = (double)vs * vs * 3, n1 = (double)vs * vs;
        std::printf("composite %dx%d: rgb/4 min %+.3f max %+.3f mean %+.3f | "
                    "blend logit min %+.3f max %+.3f | blend applied mean %.3f\n",
                    vs, vs, rmin, rmax, rsum / n3, lmin, lmax, bsum / n1);
        if (!png_path.empty()) {
            pngw::write_rgb8(png_path, vs, vs, rgb.data());
            std::printf("wrote %s\n", png_path.c_str());
        }
        if (!blend_path.empty()) {
            pngw::write_rgb8(blend_path, vs, vs, bw.data());
            std::printf("wrote %s\n", blend_path.c_str());
        }
    }
    return 0;
}
