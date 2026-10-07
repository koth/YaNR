// The 16-lane input pipeline, a C++ port of run_image.build_features (task 8.3).
// Every numeric detail mirrors the python/shader semantics: u32 wraparound hashing,
// Box-Muller noise rounded through half, center_proxy's three half roundings, and the
// padded field's mirror addressing (2*valid - x - 2) with the noise hashing the padded
// coordinate instead.
#pragma once
#include <cstdint>
#include "geometry.h"

struct LaneParams {
    uint32_t seed = 0;
    double style = 0.0;        // lane 10 = style / 128 (f64 division, stored f32)
    double tone = 0.5;         // lane 11 = roundF16(tone)
    double structure = 0.5;    // lanes 12-14, auto_mask semantics below
    double skin = -1.0;
    bool auto_mask = false;
    const float* history = nullptr;   // [vh][vw][3] code values, or nullptr = first frame
};

// out: [g.full_rows][16] contiguous f32.
void build_lanes(const float* proxy, int vw, int vh, const Geometry& g,
                 const LaneParams& p, float* out);
