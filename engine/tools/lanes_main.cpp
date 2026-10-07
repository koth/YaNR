// cpu_lanes: build the 16-lane feature field from a raw proxy dump (task 8.3 parity tool).
//
//   cpu_lanes --proxy proxy.bin --out lanes.bin --vw W --vh H --seed N
//             [--style x --tone y --structure z --skin s] [--automask] [--history hist.bin]
//
// proxy/history: raw f32 [vh][vw][3] code values. lanes: raw f32 [full_rows][16].
// The parity check (teacher/check_lanes.py) compares this byte stream against
// run_image.build_features elementwise.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "geometry.h"
#include "lanes.h"

int main(int argc, char** argv) {
    std::string proxy_path, history_path, out_path;
    int vw = 0, vh = 0;
    LaneParams p;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--proxy") proxy_path = next();
        else if (a == "--history") history_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--vw") vw = std::atoi(next());
        else if (a == "--vh") vh = std::atoi(next());
        else if (a == "--seed") p.seed = (uint32_t)std::strtoul(next(), nullptr, 10);
        else if (a == "--style") p.style = std::atof(next());
        else if (a == "--tone") p.tone = std::atof(next());
        else if (a == "--structure") p.structure = std::atof(next());
        else if (a == "--skin") p.skin = std::atof(next());
        else if (a == "--automask") p.auto_mask = true;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (proxy_path.empty() || out_path.empty() || vw <= 0 || vh <= 0) {
        std::fprintf(stderr, "usage: cpu_lanes --proxy in.bin --out out.bin --vw W --vh H --seed N ...\n");
        return 2;
    }

    auto read_f32 = [](const std::string& path, size_t expect) {
        std::vector<float> buf(expect);
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
        size_t got = std::fread(buf.data(), 4, expect, f);
        std::fclose(f);
        if (got != expect) { std::fprintf(stderr, "%s: expected %zu floats, got %zu\n", path.c_str(), expect, got); std::exit(1); }
        return buf;
    };

    std::vector<float> proxy = read_f32(proxy_path, (size_t)vw * vh * 3);
    std::vector<float> history;
    if (!history_path.empty()) {
        history = read_f32(history_path, (size_t)vw * vh * 3);
        p.history = history.data();
    }

    Geometry g = geometry_from_valid(vw, vh);
    std::vector<float> lanes((size_t)g.full_rows * 16);
    build_lanes(proxy.data(), vw, vh, g, p, lanes.data());

    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    std::fwrite(lanes.data(), 4, lanes.size(), f);
    std::fclose(f);
    std::printf("wrote %s  field %dx%d  rows %d\n", out_path.c_str(), g.full_width, g.full_height, g.full_rows);
    return 0;
}
