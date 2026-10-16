#pragma once
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>

class Engine;

struct CudaBgraTimings {
    double prepare_ms = 0, transfer_ms = 0, forward_ms = 0, output_ms = 0;
    double device_forward_ms = 0;
};

// Owns a resident model, features, scratch and CUDA graph. The host image/lane
// pipeline is shared with the CPU entry. No CUDA headers leak into C++ callers.
class CudaEngine {
public:
    CudaEngine(const Engine& model, int device = 0);
    ~CudaEngine();
    CudaEngine(const CudaEngine&) = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;
    const std::string& device_name() const;
    void upload(const float* features);
    void download(float* head);
    void download_features(float* features); // Diagnostic copy of all input lanes.
    // Synchronized CUDA-event time, including routing and every network op,
    // excluding H2D/D2H. Debug mode executes eagerly and captures CPU-style tags.
    double forward(bool capture_debug = false);
    // Same-size temporal path. upload() must initialize all invariant lanes
    // first. Strides are bytes; outputs are tightly packed BGRA8 and confidence.
    // The engine owns pinned staging, retains FP32 forward and synchronizes once.
    CudaBgraTimings infer_bgra(const std::uint8_t* previous, std::size_t previous_stride,
                             const std::uint8_t* current, std::size_t current_stride,
                             float blend_scale, std::uint8_t* bgra, std::uint8_t* confidence);
    const std::map<std::string, std::vector<float>>& debug() const;
    // Diagnostic mode NR_CUDA_PROFILE=1 inserts Graph event nodes; instrumentation
    // is excluded from normal Graph benchmarks. Last forward, per operation.
    const std::map<std::string, double>& profile() const;
    std::pair<std::size_t, std::size_t> buffer_bytes() const; // resident, allocation high-water
    // NR_CUDA_MEMORY=1 samples device memory at allocations/operation boundaries.
    // Device-wide totals include other applications; run separately from timing.
    std::vector<std::size_t> memory_sample_bytes() const; // total, initial used, sampled peak used, delta
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
