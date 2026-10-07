#include "reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "numeric.h"

namespace ref {

namespace {
int normalExponent(float value) {
  uint32_t bits = num::f32Bits(std::fabs(value));
  return (int)((bits >> 23) & 0xff) - 127;
}
int e4m3Exponent(float value) { return std::max(normalExponent(value), -6); }
int f16Exponent(float value) { return std::max(normalExponent(value), -14); }

uint32_t roundShiftRightEven(uint32_t value, uint32_t shift) {
  if (shift == 0) return value;
  if (shift > 31) return 0;
  uint32_t truncated = value >> shift;
  uint32_t remainder = value & ((1u << shift) - 1);
  uint32_t halfway = 1u << (shift - 1);
  return truncated + ((remainder > halfway || (remainder == halfway && (truncated & 1))) ? 1u : 0u);
}

int countLeadingZeros(uint32_t value) {
  int count = 0;
  for (int bit = 31; bit >= 0; --bit) { if (value & (1u << bit)) break; ++count; }
  return count;
}

// Exact signed F24 integer -> IEEE half.
float fixedToF16(int64_t fixedSum, int binaryExponent) {
  if (fixedSum == 0) return 0.0f;
  bool negative = fixedSum < 0;
  uint32_t magnitude = (uint32_t)(negative ? -fixedSum : fixedSum);
  uint32_t msb = 31 - (uint32_t)countLeadingZeros(magnitude);
  int valueExponent = (int)msb + binaryExponent;
  uint32_t halfBits = negative ? 0x8000 : 0;
  if (valueExponent >= -14) {
    uint32_t significand = msb > 10 ? roundShiftRightEven(magnitude, msb - 10) : magnitude << (10 - msb);
    if (significand >= 2048) { significand = 1024; valueExponent += 1; }
    if (valueExponent >= 16) halfBits |= 0x7c00;
    else { halfBits |= (uint32_t)(valueExponent + 15) << 10; halfBits |= significand - 1024; }
  } else {
    int subnormalScale = binaryExponent + 24;
    uint32_t mantissa = subnormalScale >= 0 ? magnitude << subnormalScale
                                            : roundShiftRightEven(magnitude, (uint32_t)-subnormalScale);
    halfBits |= std::min(mantissa, 1024u);
  }
  return num::f16ToF32((uint16_t)halfBits);
}

uint32_t inverseTiledToken(uint32_t token) {
  uint32_t tile = token >> 4, within = token & 15;
  uint32_t x = (tile & 1) * 4 + (within & 3);
  uint32_t y = (tile >> 1) * 4 + (within >> 2);
  return y * 8 + x;
}

uint32_t mapRelativeTokenTiled(uint32_t token) {
  uint32_t x = token & 7, y = token >> 3;
  return (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3);
}

float loadHalf(const nr::Tensor& tensor, uint32_t byteOffset) {
  return num::f16ToF32((uint16_t)(tensor.bytes[byteOffset] | (tensor.bytes[byteOffset + 1] << 8)));
}

float loadRelativeBias(const nr::Tensor& tensor, uint32_t relativeByteOffset, uint32_t head, uint32_t queryLocal,
                       uint32_t keyLocal) {
  uint32_t q = mapRelativeTokenTiled(queryLocal), k = mapRelativeTokenTiled(keyLocal);
  uint32_t m = q & 15, n = k & 15;
  uint32_t lane = ((m & 7) << 2) | ((n & 7) >> 1);
  uint32_t fragment = (m >= 8 ? 2 : 0) + (n & 1);
  uint32_t tileOffset = (q >> 4) * 1024 + (k >> 4) * 256;
  uint32_t laneOffset = lane * 8 + (n >> 3) * 4;
  uint32_t halfIndex = tileOffset + laneOffset + fragment;
  return loadHalf(tensor, relativeByteOffset + head * 8192 + halfIndex * 2);
}

float expWeight(float score) {
  float fusedAffine = roundF16(std::fma(score, 0.044921875f, 1.30078125f));
  float affine = std::min(std::max(fusedAffine, 1.03125f), 1.5693359375f);
  uint32_t bits = num::f16Bits(affine);
  uint32_t exponentialBits = ((bits << 5) + 0x8000u) & 0xffffu;
  return num::f16ToF32((uint16_t)exponentialBits);
}

float addF16(float a, float b) { return roundF16((float)((double)a + (double)b)); }
}  // namespace

float roundF16(float value) { return num::roundF16(value); }

float fp8Domain(float value) {
  if (value != value) return 0.0f;
  float magnitude = std::min(std::fabs(value), 448.0f);
  if (magnitude == 0.0f) return 0.0f;
  float quantized;
  if (magnitude < 0.015625f) {
    quantized = std::nearbyint(magnitude * 512.0f) / 512.0f;  // nearbyint: RNE in default rounding mode
  } else {
    float step = std::ldexp(1.0f, (int)std::floor(std::log2(magnitude)) - 3);
    quantized = std::min(std::nearbyint(magnitude / step) * step, 448.0f);
  }
  return value < 0.0f ? -quantized : quantized;
}

float mpCubicSilu(float value) {
  float bounded = roundF16(std::min(std::max(value, -4.0f), 4.0f));
  float absolute = roundF16(std::fabs(bounded));
  float inner = roundF16(-0.055908203125f * absolute + 0.447265625f);
  float polynomial = roundF16(bounded * inner + 0.89453125f);
  return roundF16(value * polynomial);
}

std::vector<uint16_t> siluTable() {
  std::vector<uint16_t> table(65536);
  for (uint32_t bits = 0; bits < 65536; ++bits) table[bits] = num::f16Bits(mpCubicSilu(num::f16ToF32((uint16_t)bits)));
  return table;
}

float adaFp8Fdpa16(const float* a, const float* b, int count, float accumulator) {
  if (std::isinf(accumulator) || std::isnan(accumulator)) return accumulator;
  int maximumExponent = -21;
  if (accumulator != 0.0f) maximumExponent = f16Exponent(accumulator);
  for (int i = 0; i < count; ++i) {
    if (a[i] != 0.0f && b[i] != 0.0f) maximumExponent = std::max(maximumExponent, e4m3Exponent(a[i]) + e4m3Exponent(b[i]));
  }
  double fused = 0.0;
  if (accumulator != 0.0f) {
    int exponent = f16Exponent(accumulator);
    float significand = accumulator * std::ldexp(1.0f, -exponent);
    fused += std::trunc(significand * std::ldexp(1.0f, exponent - maximumExponent + 13)) * 0.0001220703125;
  }
  for (int i = 0; i < count; ++i) {
    if (a[i] != 0.0f && b[i] != 0.0f) {
      int ea = e4m3Exponent(a[i]), eb = e4m3Exponent(b[i]);
      float sa = a[i] * std::ldexp(1.0f, -ea), sb = b[i] * std::ldexp(1.0f, -eb);
      fused += std::trunc(sa * sb * std::ldexp(1.0f, ea + eb - maximumExponent + 13)) * 0.0001220703125;
    }
  }
  return roundF16((float)(fused * std::ldexp(1.0, maximumExponent)));
}

float adaF16Fdpa8(const float* a, const float* b, int count, float accumulator) {
  int maximumExponent = -21;
  if (accumulator != 0.0f) maximumExponent = f16Exponent(accumulator);
  for (int i = 0; i < count; ++i) {
    if (a[i] != 0.0f && b[i] != 0.0f) maximumExponent = std::max(maximumExponent, f16Exponent(a[i]) + f16Exponent(b[i]));
  }
  int64_t fixedSum = 0;
  if (accumulator != 0.0f) {
    int exponent = f16Exponent(accumulator);
    float significand = accumulator * std::ldexp(1.0f, -exponent);
    fixedSum += (int64_t)std::trunc(significand * std::ldexp(1.0f, exponent - maximumExponent + 24));
  }
  for (int i = 0; i < count; ++i) {
    if (a[i] != 0.0f && b[i] != 0.0f) {
      int ea = f16Exponent(a[i]), eb = f16Exponent(b[i]);
      float sa = a[i] * std::ldexp(1.0f, -ea), sb = b[i] * std::ldexp(1.0f, -eb);
      fixedSum += (int64_t)std::trunc(sa * sb * std::ldexp(1.0f, ea + eb - maximumExponent + 24));
    }
  }
  return fixedToF16(fixedSum, maximumExponent - 24);
}

float gemmFp8Element(const GemmRef& gemm, const float* inputRow, uint32_t column, float initial) {
  float sums = roundF16(initial);
  float partitionSums = 0.0f;
  float a[32], b[32];
  for (uint32_t kb = 0; kb < gemm.K; kb += 32) {
    for (uint32_t k = 0; k < 32; ++k) {
      uint32_t logical = kb + k;
      uint32_t inputK = gemm.swizzleInput ? nr::packedInputIndex(logical) : logical;
      a[k] = inputRow[inputK];
      uint8_t code = gemm.tensor->bytes[gemm.weightByteOffset +
                                         nr::packedWeightIndex(logical, column + gemm.weightColumnOffset, gemm.Nmatrix)];
      b[k] = (code & 0x7f) == 0x7f ? 0.0f : num::e4m3ToF32(code);
    }
    sums = adaFp8Fdpa16(a, b, 16, sums);
    sums = adaFp8Fdpa16(a + 16, b + 16, 16, sums);
    if (gemm.partition && (((kb + 32) % gemm.partition) == 0 || kb + 32 >= gemm.K)) {
      partitionSums = kb < gemm.partition ? sums : roundF16(partitionSums + sums);
      sums = 0.0f;
    }
  }
  return gemm.partition ? partitionSums : sums;
}

float gemmF16Element(const nr::Tensor& tensor, uint32_t weightByteOffset, uint32_t K, uint32_t N,
                     const float* inputRow, uint32_t column) {
  const uint32_t nTiles = (N + 15) / 16;
  auto weightIndex = [&](uint32_t k, uint32_t n) -> uint32_t {   // = nr_model.cpp packedF16WeightIndex
    uint32_t tile = (k >> 4) * nTiles + (n >> 4);
    uint32_t kk = k & 15, nn = n & 15;
    uint32_t lane = ((nn & 7) << 2) | ((kk & 7) >> 1);
    uint32_t fragment = (kk >= 8 ? 2 : 0) + (kk & 1);
    return tile * 256 + lane * 8 + ((nn >> 3) & 1) * 4 + fragment;
  };
  float value = 0.0f;
  float a[8], b[8];
  for (uint32_t base = 0; base < K; base += 8) {
    for (uint32_t i = 0; i < 8; ++i) {
      a[i] = roundF16(inputRow[base + i]);
      b[i] = loadHalf(tensor, weightByteOffset + weightIndex(base + i, column) * 2);
    }
    value = adaF16Fdpa8(a, b, 8, value);
  }
  return value;
}

void windowNormalizeRef(const float* qkvRow, uint32_t head, float scale, float* out) {
  const float* q = qkvRow + head * 96;
  const float* k = q + 32;
  const float* v = q + 64;
  auto normOf = [](const float* x) {
    float r[16];
    for (int c = 0; c < 16; ++c) {
      float highSquare = roundF16(x[c + 16] * x[c + 16]);
      r[c] = roundF16((float)((double)x[c] * (double)x[c] + (double)highSquare));  // exact half fma
    }
    for (int stride = 8; stride > 0; stride >>= 1)
      for (int c = 0; c < stride; ++c) r[c] = addF16(r[c], r[c + stride]);
    return roundF16(1.0f / std::sqrt(r[0]));
  };
  float qNorm = normOf(q), kNorm = normOf(k);
  float scaleHalf = roundF16(scale);
  for (int c = 0; c < 32; ++c) {
    out[c] = fp8Domain(roundF16(roundF16(q[c] * qNorm) * scaleHalf));
    out[32 + c] = fp8Domain(roundF16(k[c] * kNorm));
    out[64 + c] = fp8Domain(v[c]);
  }
}

void windowAttendRef(const std::vector<float>& normalized, uint32_t width, uint32_t height, uint32_t channels,
                     uint32_t head, int windowX, int windowY, const nr::Tensor& tensor, uint32_t relativeByteOffset,
                     float* out) {
  const uint32_t stride3 = channels * 3;
  const uint32_t headBase = head * 96;
  for (uint32_t queryLocal = 0; queryLocal < 64; ++queryLocal) {
    int qx = windowX + (int)(queryLocal & 7), qy = windowY + (int)(queryLocal >> 3);
    if (qx < 0 || qy < 0 || qx >= (int)width || qy >= (int)height) {
      for (int c = 0; c < 32; ++c) out[queryLocal * 32 + c] = NAN;
      continue;
    }
    const float* q = &normalized[((uint32_t)qy * width + (uint32_t)qx) * stride3 + headBase];
    float scores[64];
    float keys[64][32];
    float values[64][32];
    for (uint32_t keyLocal = 0; keyLocal < 64; ++keyLocal) {
      int kx = windowX + (int)(keyLocal & 7), ky = windowY + (int)(keyLocal >> 3);
      bool valid = kx >= 0 && ky >= 0 && kx < (int)width && ky < (int)height;
      for (int c = 0; c < 32; ++c) {
        keys[keyLocal][c] = valid ? normalized[((uint32_t)ky * width + (uint32_t)kx) * stride3 + headBase + 32 + c] : 0.0f;
        values[keyLocal][c] = valid ? normalized[((uint32_t)ky * width + (uint32_t)kx) * stride3 + headBase + 64 + c] : 0.0f;
      }
      float prior = loadRelativeBias(tensor, relativeByteOffset, head, queryLocal, keyLocal);
      float score = adaFp8Fdpa16(q, keys[keyLocal], 16, prior);
      score = adaFp8Fdpa16(q + 16, keys[keyLocal] + 16, 16, score);
      scores[keyLocal] = expWeight(roundF16(score));
    }
    auto physical = [&](uint32_t p) { return scores[inverseTiledToken(p)]; };
    auto pair = [&](uint32_t pairIndex, uint32_t parity) {
      uint32_t key = pairIndex * 2 + parity;
      float b01 = addF16(physical(key), physical(key + 8));
      float b23 = addF16(physical(key + 16), physical(key + 24));
      float b45 = addF16(physical(key + 32), physical(key + 40));
      float b67 = addF16(physical(key + 48), physical(key + 56));
      return addF16(addF16(addF16(b01, b23), b45), b67);
    };
    float even = addF16(addF16(addF16(pair(0, 0), pair(1, 0)), pair(2, 0)), pair(3, 0));
    float odd = addF16(addF16(addF16(pair(0, 1), pair(1, 1)), pair(2, 1)), pair(3, 1));
    float total = addF16(even, odd);
    float reciprocal = roundF16(1.0f / total);
    for (uint32_t key = 0; key < 64; ++key) scores[key] = fp8Domain(roundF16(scores[key] * reciprocal));
    for (int c = 0; c < 32; ++c) {
      float value = 0.0f;
      for (uint32_t group = 0; group < 4; ++group) {
        float w[16], v[16];
        for (uint32_t i = 0; i < 16; ++i) {
          uint32_t keyLocal = inverseTiledToken(group * 16 + i);
          w[i] = scores[keyLocal];
          v[i] = values[keyLocal][c];
        }
        value = adaFp8Fdpa16(w, v, 16, value);
      }
      out[queryLocal * 32 + c] = fp8Domain(value);
    }
  }
}

}  // namespace ref
