// CPU reference of the network's exact arithmetic (FP8 GEMM accumulation order, half publications, window
// attention). Used to bisect the GPU kernels one by one: each check feeds the GPU's own inputs so a mismatch names
// the responsible kernel rather than a compounded downstream effect.
#pragma once
#include <cstdint>
#include <vector>

#include "nr_model.h"

namespace ref {

float roundF16(float value);
float fp8Domain(float value);   // E4M3 RNE saturating publication, as a value
float mpCubicSilu(float value);
// 65536-entry f16 -> f16 SiLU table, which is the form the GLSL kernels use.
std::vector<uint16_t> siluTable();

// Ada FP8 m16n8k32 half of a k32 step: 16 products + incoming half accumulator,
// shared exponent, F=13 truncation, half RNE result.
float adaFp8Fdpa16(const float* a, const float* b, int count, float accumulator);
// Ada FP16 m16n8k16 half step: 8 products, F=24 truncation, integer publication.
float adaF16Fdpa8(const float* a, const float* b, int count, float accumulator);

struct GemmRef {
  const nr::Tensor* tensor = nullptr;
  uint32_t weightByteOffset = 0;
  uint32_t K = 0;
  uint32_t Nmatrix = 0;
  uint32_t weightColumnOffset = 0;
  uint32_t partition = 0;
  bool swizzleInput = true;
};
// One FP8 GEMM output element with the literal accumulation contract. `input` holds
// the activation row values (E4M3-decoded) in natural channel order.
float gemmFp8Element(const GemmRef& gemm, const float* inputRow, uint32_t column, float initial);

float gemmF16Element(const nr::Tensor& tensor, uint32_t weightByteOffset, uint32_t K, uint32_t N,
                     const float* inputRow, uint32_t column);

// Window attention reference for one (window origin, head): returns the 64x32
// attended E4M3 values (as floats) in query-local order, NaN for OOB queries.
// `normalized` is the E4M3-decoded [tokens][channels*3] tensor.
void windowAttendRef(const std::vector<float>& normalized, uint32_t width, uint32_t height, uint32_t channels,
                     uint32_t head, int windowX, int windowY, const nr::Tensor& tensor, uint32_t relativeByteOffset,
                     float* out /*64*32*/);
// Normalized Q/K/V (E4M3 values) for one token/head from raw half QKV.
void windowNormalizeRef(const float* qkvRow /*channels*3*/, uint32_t head, float scale, float* out /*96*/);

}  // namespace ref
