// Minimal stand-in for src/nr_model.h so reference.cpp compiles without Vulkan: the CPU reference only ever
// touches Tensor's bytes. The real header drags in vk_context.h / volk; this shim is on the include path
// before src/ when building check_block0.cpp.
#pragma once
#include <cstdint>
#include <string>

namespace nr {

struct Tensor {
  std::string name;
  int block = 0;
  int layer = 0;
  std::string parameter;
  std::string stage;
  uint32_t stageOffset = 0;
  uint32_t byteLength = 0;
  const uint8_t* bytes = nullptr;
};

uint32_t packedInputIndex(uint32_t k);
uint32_t inversePackedInputIndex(uint32_t k);
uint32_t packedWeightIndex(uint32_t k, uint32_t n, uint32_t outputChannels);

}  // namespace nr
