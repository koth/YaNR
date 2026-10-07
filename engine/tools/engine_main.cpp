// cpu_engine: run the student forward from a features dump (parity/bench tool).
//
//   cpu_engine --idx student512.idx --features feat.bin --out head.bin
//   cpu_engine --idx student512.idx --features feat.bin --bench --repeats 30
//
// features: raw f32 [full_rows][16]; head: raw f32 [full_rows][4].
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine.h"

int main(int argc, char** argv) {
    std::string idx_path, feat_path, out_path, dump_dir;
    int repeats = 30;
    bool bench = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return argv[++i]; };
        if (a == "--idx") idx_path = next();
        else if (a == "--features") feat_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--dump-dir") dump_dir = next();
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

    if (bench) {
        engine.forward(features.data(), head.data());                 // warmup
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
    return 0;
}
