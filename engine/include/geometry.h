// Field geometry, a C++ port of nr_geometry.geometry_from_valid (the field/padding
// rules decide which tokens exist - nothing here is a free choice).
#pragma once
#include <cstdint>

struct Level { int width, height, rows; };

struct Geometry {
    int valid_width, valid_height;
    int full_width, full_height;
    int full_rows;
    Level levels[6];
    int vit_tokens;
    int padded_vit_tokens;
};

inline int align_up(int v, int a) { return (v + a - 1) / a * a; }

inline Geometry geometry_from_valid(int valid_width, int valid_height) {
    auto alignment = [](int valid) {
        int reductions = 0;
        int size = valid;
        for (int level = 0; level < 6; level++) {
            int half = align_up((size + 1) / 2, 4);
            if (half < size) reductions++;
            if (level == 0 && half % 8 != 0) reductions++;
            size = half;
        }
        return 1 << reductions;
    };
    int aw = alignment(valid_width), ah = alignment(valid_height);
    Geometry g{};
    g.valid_width = valid_width;
    g.valid_height = valid_height;
    g.full_width = align_up(valid_width, aw) > 320 ? align_up(valid_width, aw) : 320;
    g.full_height = align_up(valid_height, ah) > 320 ? align_up(valid_height, ah) : 320;
    if (g.full_width % (4 * aw) == 0 && g.full_height % (4 * ah) == 0) g.full_width += aw;
    int w = g.full_width, h = g.full_height;
    for (int i = 0; i < 6; i++) {
        w = align_up((w + 1) / 2, 4);
        h = align_up((h + 1) / 2, 4);
        g.levels[i] = {w, h, w * h};
    }
    g.full_rows = g.full_width * g.full_height;
    g.vit_tokens = g.levels[5].rows;
    g.padded_vit_tokens = (g.vit_tokens + 63) & ~63;
    return g;
}
