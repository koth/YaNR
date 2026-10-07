// Block 0, row 0, from the Vulkan implementation's own CPU reference (src/reference.cpp) - the tie-breaker
// when the torch port and the WebGPU port disagree about what the network computes. It reads the block-0
// input bundle check_block0.py writes (the packed block-0 tensor plus the 64 feature rows of the window at
// origin (0,0)) and prints the 32 block output values of row 0.
//
//   cp <repo>/src/reference.cpp <repo>/src/reference.h <repo>/src/numeric.h shim/
//   g++ -std=c++17 -O2 -I shim check_block0.cpp -o check_block0
//   ./check_block0 block0_input.bin

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vendor/reference/numeric.h"
#include "vendor/reference/reference.cpp"

namespace nr {
uint32_t packedInputIndex(uint32_t k) {
  uint32_t base = k & ~31u;
  uint32_t within = k & 31u;
  uint32_t half = within & 16u;
  uint32_t quarter = within & 15u;
  return base + half + (quarter >> 2) * 2 + (quarter & 1) + (((quarter & 2) != 0) ? 8 : 0);
}
uint32_t inversePackedInputIndex(uint32_t k) {
  uint32_t base = k & ~31u;
  uint32_t within = k & 31u;
  return base + (within & 17u) + ((within & 2u) << 1) + ((within & 4u) << 1) + ((within & 8u) >> 2);
}
uint32_t packedWeightIndex(uint32_t k, uint32_t n, uint32_t outputChannels) {
  uint32_t kTile = k >> 5, kIn = k & 31;
  uint32_t nTile = n >> 7, nIn = n & 127;
  uint32_t nHalf = nIn >> 6, nGroup = (nIn & 63) >> 4, nInGroup = nIn & 15;
  uint32_t lane = ((nInGroup & 7) << 2) | ((kIn & 15) >> 2);
  uint32_t byteInLane = ((nInGroup >> 3) << 3) | ((kIn >> 4) << 2) | (kIn & 3);
  return kTile * outputChannels * 32 + nTile * 4096 + nHalf * 2048 + nGroup * 512 + lane * 16 + byteInLane;
}
}  // namespace nr

namespace {

// pre_fused_layout() from ports/browser-webgpu/src/geometry.js.
constexpr uint32_t kExpand = 0;
constexpr uint32_t kContract = 4096;
constexpr uint32_t kInputAdapter = 8208;
constexpr uint32_t kFfnCosSkip = 9232;
constexpr uint32_t kQkv = 9312;
constexpr uint32_t kRelative = 12384;
constexpr uint32_t kScale = 20576;
constexpr uint32_t kProjection = 20592;
constexpr uint32_t kAttnCosSkip = 21616;

uint16_t auxHalf(const nr::Tensor& tensor, uint32_t byteOffset, uint32_t column) {
  uint32_t at = byteOffset + column * 2;
  return (uint16_t)(tensor.bytes[at] | (tensor.bytes[at + 1] << 8));
}

float auxF32(const nr::Tensor& tensor, uint32_t byteOffset) {
  float value;
  std::memcpy(&value, tensor.bytes + byteOffset, 4);
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s block0_input.bin [width]\n", argv[0]);
    return 2;
  }
  uint32_t width = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 320;
  FILE* file = std::fopen(argv[1], "rb");
  if (!file) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  uint32_t preSize = 0;
  if (std::fread(&preSize, 4, 1, file) != 1) return 2;
  std::vector<uint8_t> stage(preSize);
  if (std::fread(stage.data(), 1, preSize, file) != preSize) return 2;
  std::vector<float> features(64 * 16);
  if (std::fread(features.data(), 4, 64 * 16, file) != 64 * 16) return 2;
  std::fclose(file);

  nr::Tensor pre;
  pre.bytes = stage.data();
  pre.byteLength = preSize;

  ref::GemmRef expand{&pre, kExpand, 32, 128, 0, 0, true};
  ref::GemmRef contract{&pre, kContract, 128, 32, 0, 0, true};
  ref::GemmRef qkvGemm{&pre, kQkv, 32, 96, 0, 0, true};
  ref::GemmRef proj{&pre, kProjection, 32, 32, 0, 0, true};
  float scale = auxF32(pre, kScale);

  // The window sits at origin (0,0): its 64 slots are pixels (x, y), y in 0..7, x in 0..7.
  std::vector<float> normalized((size_t)width * width * 96, 0.0f);
  std::vector<float> contractRawRow0(32, 0.0f);
  const uint32_t stride3 = 32 * 3;

  for (uint32_t slot = 0; slot < 64; ++slot) {
    uint32_t x = slot & 7, y = slot >> 3;
    uint32_t row = y * width + x;
    const float* featureRow = features.data() + slot * 16;

    float adapter[32];
    for (uint32_t n = 0; n < 32; ++n) {
      adapter[n] = ref::gemmF16Element(pre, kInputAdapter, 16, 32, featureRow, n);
    }
    float adapterE4[32];
    for (uint32_t n = 0; n < 32; ++n) adapterE4[n] = ref::fp8Domain(adapter[n]);

    float ffn[128];
    for (uint32_t m = 0; m < 128; ++m) {
      float preValue = ref::gemmFp8Element(expand, adapterE4, m, 0.0f);
      ffn[m] = ref::fp8Domain(ref::mpCubicSilu(preValue));
    }

    float contractRaw[32];
    for (uint32_t n = 0; n < 32; ++n) {
      float aux = num::f16ToF32(auxHalf(pre, kFfnCosSkip, n));
      contractRaw[n] = ref::gemmFp8Element(contract, ffn, n, adapter[n] * aux);
    }
    float ffnQuantized[32];
    for (uint32_t n = 0; n < 32; ++n) ffnQuantized[n] = ref::fp8Domain(contractRaw[n]);

    float qkv[96];
    for (uint32_t j = 0; j < 96; ++j) {
      qkv[j] = ref::gemmFp8Element(qkvGemm, ffnQuantized, j, 0.0f);
    }
    float out96[96];
    ref::windowNormalizeRef(qkv, 0, scale, out96);
    for (uint32_t c = 0; c < 96; ++c) normalized[(size_t)row * stride3 + c] = out96[c];
    if (slot == 0) {
      for (uint32_t n = 0; n < 32; ++n) contractRawRow0[n] = contractRaw[n];
    }
  }

  std::vector<float> attended(64 * 32, 0.0f);
  ref::windowAttendRef(normalized, width, width, 32, 0, 0, 0, pre, kRelative, attended.data());

  const float* q0 = attended.data();   // query local 0 = field (0, 0)
  for (uint32_t n = 0; n < 32; ++n) {
    float aux = num::f16ToF32(auxHalf(pre, kAttnCosSkip, n));
    float raw = ref::gemmFp8Element(proj, q0, n, contractRawRow0[n] * aux);
    std::printf("%.9g%s", ref::fp8Domain(raw), n + 1 == 32 ? "\n" : " ");
  }
  return 0;
}
