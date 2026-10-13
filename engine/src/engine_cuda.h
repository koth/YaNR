#pragma once
#include <memory>
#include <map>
#include <string>
#include <vector>

class Engine;

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
    // Synchronized CUDA-event time, including routing and every network op,
    // excluding H2D/D2H. Debug mode executes eagerly and captures CPU-style tags.
    double forward(bool capture_debug = false);
    const std::map<std::string, std::vector<float>>& debug() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
