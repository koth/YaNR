#pragma once
#include <cstdint>

enum class VnniIsa { none, avx, avx512 };
// One signed INT8 tile: A[4][Kp], panel[Kp/4][16][4], out[4][16].
// All four rows must be accessible, including unused rows in a tail tile.
using VnniDot = void (*)(const int8_t* A, const int8_t* panel,
                        const int32_t* correction, int Kp, int32_t* out);

VnniIsa selected_vnni_isa();
VnniDot vnni_kernel(VnniIsa isa); // nullptr unless compiled + CPU + OS supported

void vnni_dot_avx(const int8_t*, const int8_t*, const int32_t*, int, int32_t*);
void vnni_dot_avx512(const int8_t*, const int8_t*, const int32_t*, int, int32_t*);
