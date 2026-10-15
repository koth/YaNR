#include "engine.h"
#include "engine_cuda.h"
#include "f16.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <cub/device/device_radix_sort.cuh>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
void cu(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}
void blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(static_cast<int>(status)));
}
template<class T> struct Buffer {
    T* p = nullptr;
    size_t capacity = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (p) cudaFree(p); }
    void resize(size_t n) {
        if (n <= capacity) return;
        T* next = nullptr;
        cu(cudaMalloc(reinterpret_cast<void**>(&next), n * sizeof(T)));
        if (p) cu(cudaFree(p));
        p = next; capacity = n;
    }
};
template<class T> struct HostBuffer {
    T* p = nullptr;
    HostBuffer() = default;
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    ~HostBuffer() { if (p) cudaFreeHost(p); }
    void allocate(size_t n) { cu(cudaMallocHost(reinterpret_cast<void**>(&p), n * sizeof(T))); }
};
int grid(size_t n) { return static_cast<int>((n + 255) / 256); }

__global__ void temporal_bgra(const unsigned char* frames, float* features, const float* centered,
                             int width, int height, int full_width, int full_height) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(full_width) * full_height * 6) return;
    int row = i / 6, c = i % 6, x = row % full_width, y = row / full_width;
    int rx = max(0, min(x < width ? x : 2 * width - x - 2, width - 1));
    int ry = max(0, min(y < height ? y : 2 * height - y - 2, height - 1));
    size_t base = c < 3 ? static_cast<size_t>(width) * height * 4 : 0;
    unsigned char code = frames[base + (static_cast<size_t>(ry) * width + rx) * 4 + 2 - c % 3];
    features[static_cast<size_t>(row) * 16 + 4 + c] = centered[code];
}
__global__ void redraw_bgra(const unsigned char* current, const float* head, const float* rgb,
                           unsigned char* output, int width, int height, int full_width, float blend_scale) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(width) * height) return;
    size_t row = i / width * full_width + i % width;
    for (int c = 0; c < 3; ++c) {
        float v = fmaxf(0.0f, fminf(1.0f, rgb[current[i * 4 + 2 - c]] + head[row * 4 + c] * 0.25f));
        output[i * 4 + 2 - c] = static_cast<unsigned char>(v * 255.0f + 0.5f);
    }
    output[i * 4 + 3] = 255;
    float confidence = fmaxf(0.0f, fminf(1.0f, (1.0f / (1.0f + expf(-head[row * 4 + 3]))) * blend_scale));
    output[static_cast<size_t>(width) * height * 4 + i] = static_cast<unsigned char>(confidence * 255.0f + 0.5f);
}

__global__ void epilogue(float* out, int rows, int channels, const float* bias,
                         bool act, const float* residual, const float* aux) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(rows) * channels) return;
    float v = out[i] + (bias ? bias[i % channels] : 0.0f);
    if (act) v = v / (1.0f + expf(-v));
    if (residual) v += residual[i] * aux[i % channels];
    out[i] = v;
}
__global__ void pool2(const float* x, float* out, int iw, int ih, int ow, int oh, int ch) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(ow) * oh * ch) return;
    int row = static_cast<int>(i / ch), c = i % ch;
    int sx = min((row % ow) * 2, iw - 1), sy = min((row / ow) * 2, ih - 1);
    if (sx + 1 >= iw || sy + 1 >= ih) { out[i] = 0; return; }
    size_t a = (static_cast<size_t>(sy) * iw + sx) * ch + c;
    out[i] = (x[a] + x[a + ch] + x[a + iw * ch] + x[a + (iw + 1) * ch]) * 0.25f;
}
__global__ void merge2(const float* x, const float* skip, float* out, const float* a0,
                       const float* a1, int iw, int ih, int ow, int oh, int ch) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(ow) * oh * ch) return;
    int row = static_cast<int>(i / ch), c = i % ch;
    size_t src = (static_cast<size_t>(min(row / ow / 2, ih - 1)) * iw
                 + min(row % ow / 2, iw - 1)) * ch + c;
    out[i] = (a0 ? x[src] * a0[c] : x[src]) + skip[i] * a1[c];
}
__global__ void add_attention(const float* residual, const float* proj, const float* aux,
                              float* state, int rows, int ch) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(rows) * ch) state[i] = residual[i] + proj[i] * aux[i % ch];
}

// Router pooling keeps the same per-channel sum order as the CPU implementation.
// The scalar LN reduction is intentional: tiny logit changes can alter top-1 picks.
__global__ void router_input(const float* state, float* norm, int width, int height, int ch,
                             const float* gamma, const float* beta, bool windows) {
    __shared__ float pooled[1024];
    __shared__ float mean, inv;
    int item = blockIdx.x, c = threadIdx.x;
    int nw = (width + 7) / 8;
    for (int j = c; j < ch; j += blockDim.x) {
        float v = 0; int count = 0;
        if (windows) {
            int wx = item % nw, wy = item / nw;
            for (int y = wy * 8; y < min(height, wy * 8 + 8); ++y)
                for (int x = wx * 8; x < min(width, wx * 8 + 8); ++x) {
                    v += state[(static_cast<size_t>(y) * width + x) * ch + j]; ++count;
                }
            v *= 1.0f / count;
        } else v = state[static_cast<size_t>(item) * ch + j];
        pooled[j] = v;
    }
    __syncthreads();
    if (!c) {
        float m = 0, var = 0;
        for (int j = 0; j < ch; ++j) m += pooled[j];
        m /= ch;
        for (int j = 0; j < ch; ++j) { float d = pooled[j] - m; var += d * d; }
        mean = m; inv = 1.0f / sqrtf(var / ch + 1e-5f);
    }
    __syncthreads();
    for (int j = c; j < ch; j += blockDim.x)
        norm[static_cast<size_t>(item) * ch + j] = (pooled[j] - mean) * inv * gamma[j] + beta[j];
}
__global__ void initial_pick(const float* logits, int items, int experts, int* pick,
                             int* counts, unsigned long long* keys, int* ids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= items) return;
    int best = 0;
    for (int e = 1; e < experts; ++e)
        if (logits[i * experts + e] > logits[i * experts + best]) best = e;
    float alt = -CUDART_INF_F;
    for (int e = 0; e < experts; ++e) if (e != best) alt = fmaxf(alt, logits[i * experts + e]);
    float margin = fmaxf(0.0f, logits[i * experts + best] - alt);
    pick[i] = best; atomicAdd(counts + best, 1); ids[i] = i;
    keys[i] = (static_cast<unsigned long long>(best) << 32) | __float_as_uint(margin);
}
__global__ void victim_keys(const unsigned long long* sorted, const int* counts,
                            unsigned long long* keys, int items, int experts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= items) return;
    int expert = static_cast<int>(sorted[i] >> 32), base = 0;
    for (int e = 0; e < expert; ++e) base += counts[e];
    int excess = max(0, counts[expert] - (items + experts - 1) / experts);
    keys[i] = i - base < excess ? (sorted[i] & 0xffffffffULL) : 0xffffffffffffffffULL;
}
__global__ void balanced_assign(const int* victims, const unsigned long long* keys,
                                const int* counts, int* pick, int items, int experts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= items || keys[i] == 0xffffffffffffffffULL) return;
    int quota = (items + experts - 1) / experts, seat = i;
    for (int e = 0; e < experts; ++e) {
        int n = max(0, quota - counts[e]);
        if (seat < n) { pick[victims[i]] = e; return; }
        seat -= n;
    }
}
__global__ void route_gate(const float* logits, const int* pick, float* gates, int items, int experts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= items) return;
    float mx = -CUDART_INF_F, den = 0;
    for (int e = 0; e < experts; ++e) mx = fmaxf(mx, logits[i * experts + e]);
    for (int e = 0; e < experts; ++e) den += expf(logits[i * experts + e] - mx);
    gates[i] = experts * expf(logits[i * experts + pick[i]] - mx) / fmaxf(den, 1e-9f);
}
// One block per route item. Reserve a whole 8x8 tile so capacities stay bounded
// by quota*64, including partial windows; no host readback of group sizes.
__global__ void make_group(const float* state, float* grouped, int* row_group,
                           const int* pick, int* counters, int cap, int width, int height,
                           int ch, bool windows) {
    __shared__ int base;
    int item = blockIdx.x, tid = threadIdx.x;
    if (!tid) base = pick[item] * cap + atomicAdd(counters + pick[item], windows ? 64 : 1);
    __syncthreads();
    int nw = (width + 7) / 8;
    for (int i = tid; i < (windows ? 64 : 1) * ch; i += blockDim.x) {
        int slot = i / ch, c = i % ch;
        int x = windows ? item % nw * 8 + slot % 8 : item;
        int y = windows ? item / nw * 8 + slot / 8 : 0;
        if (x < width && y < height) {
            int row = y * width + x;
            grouped[static_cast<size_t>(base + slot) * ch + c] = state[static_cast<size_t>(row) * ch + c];
            if (!c) row_group[row] = base + slot;
        }
    }
}
__global__ void moe_residual(const float* grouped_ffn, const float* state, float* residual,
                             float* grouped_res, const int* row_group, const float* gates,
                             const float* aux, int width, int height, int ch, bool windows) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(width) * height * ch) return;
    int row = i / ch, c = i % ch;
    int item = windows ? (row / width / 8) * ((width + 7) / 8) + row % width / 8 : row;
    float v = grouped_ffn[static_cast<size_t>(row_group[row]) * ch + c] * gates[item] + state[i] * aux[c];
    residual[i] = v; grouped_res[static_cast<size_t>(row_group[row]) * ch + c] = v;
}
__global__ void group_copy(const float* in, float* out, const int* row_group, int rows, int ch, bool gather) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(rows) * ch) return;
    size_t g = static_cast<size_t>(row_group[i / ch]) * ch + i % ch;
    if (gather) out[g] = in[i]; else out[i] = in[g];
}

// q/k/v are packed [window, head, slot, channel]. Invalid window keys stay zero,
// but remain in the softmax denominator: the trained window mask is query-only.
template<int Window>
__global__ void window_qkv(const float* qkv, float* packed, int width, int height,
                           int heads, int runtime_win, int sx, int sy, const float* scale, int tasks) {
    const int win = Window ? Window : runtime_win;
    int item = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (item >= tasks) return; // Whole warps are inactive; normalization stays warp-wide.
    int slots = win * win, wy = (height + sy + win - 1) / win;
    int slot = item % slots, wh = item / slots, head = wh % heads, window = wh / heads;
    int x = window / wy * win - sx + slot % win, y = window % wy * win - sy + slot / win;
    int c = threadIdx.x % 32;
    float q = 0, k = 0, v = 0;
    if (x >= 0 && x < width && y >= 0 && y < height) {
        const float* src = qkv + (static_cast<size_t>(y) * width + x) * heads * 96 + head * 96;
        q = src[c]; k = src[32 + c]; v = src[64 + c];
    }
    float qn = q * q, kn = k * k;
    for (int d = 16; d; d >>= 1) { qn += __shfl_down_sync(0xffffffff, qn, d); kn += __shfl_down_sync(0xffffffff, kn, d); }
    qn = fmaxf(sqrtf(__shfl_sync(0xffffffff, qn, 0)), 1e-6f);
    kn = fmaxf(sqrtf(__shfl_sync(0xffffffff, kn, 0)), 1e-6f);
    size_t base = static_cast<size_t>(wh) * slots * 96 + slot * 32 + c;
    packed[base] = q / qn * scale[head];
    packed[base + slots * 32] = k / kn;
    packed[base + slots * 64] = v;
}
__device__ float warp_sum(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    return __shfl_sync(0xffffffff, v, 0);
}
__device__ float warp_max(float v) {
    for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, d));
    return __shfl_sync(0xffffffff, v, 0);
}
template<int Window>
__global__ void window_attend(const float* packed, float* out, const float* prior,
                              int width, int height, int heads, int runtime_win, int sx, int sy) {
    extern __shared__ float shared[];
    const int win = Window ? Window : runtime_win;
    int wh = blockIdx.x, head = wh % heads, window = wh / heads;
    int slots = win * win, wy = (height + sy + win - 1) / win;
    int ox = window / wy * win - sx, oy = window % wy * win - sy;
    float* queries = shared;
    float* values = queries + slots * 32;
    float* keys = values + slots * 32;
    // Score lanes address different keys at the same channel. Row-major keys
    // cause 32-way shared-bank conflicts; transpose with a padded pitch so both
    // the transpose stores and score loads are conflict-free for 4/8 windows.
    int key_pitch = slots + 1;
    for (int i = threadIdx.x; i < slots * 32; i += blockDim.x) {
        size_t base = static_cast<size_t>(wh) * slots * 96;
        queries[i] = packed[base + i];
        keys[(i % 32) * key_pitch + i / 32] = packed[base + slots * 32 + i];
        values[i] = packed[base + slots * 64 + i];
    }
    __syncthreads();
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int query = warp; query < slots; query += blockDim.x / 32) {
        int x = ox + query % win, y = oy + query / win;
        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        float scores[Window ? (Window * Window + 31) / 32 : 8], mx = -CUDART_INF_F;
#pragma unroll
        for (int t = lane; t < slots; t += 32) {
            float score = 0;
#pragma unroll
            for (int c = 0; c < 32; ++c)
                score += queries[query * 32 + c] * keys[c * key_pitch + t];
            score += prior[(head * slots + query) * slots + t];
            scores[t / 32] = score; mx = fmaxf(mx, score);
        }
        mx = warp_max(mx);
        float den = 0;
#pragma unroll
        for (int t = lane; t < slots; t += 32) { scores[t / 32] = expf(scores[t / 32] - mx); den += scores[t / 32]; }
        den = fmaxf(warp_sum(den), 1e-9f);
        // Normalize each key once, before broadcasting it to the value lanes.
        // The denominator is identical across the warp; division and the
        // sequential value accumulation retain the reference rounding order.
#pragma unroll
        for (int t = lane; t < slots; t += 32) scores[t / 32] /= den;
        float v = 0;
#pragma unroll
        for (int t = 0; t < slots; ++t) {
            float prob = __shfl_sync(0xffffffff, scores[t / 32], t % 32);
            v += prob * values[t * 32 + lane];
        }
        out[(static_cast<size_t>(y) * width + x) * heads * 32 + head * 32 + lane] = v;
    }
}
__global__ void vit_qkv(const float* qkv, float* packed, int tokens, int padded, int heads, const float* scale) {
    int t = blockIdx.x % padded, h = blockIdx.x / padded, c = threadIdx.x;
    float q = 0, k = 0, v = 0;
    if (t < tokens) {
        const float* src = qkv + (static_cast<size_t>(t) * heads + h) * 96;
        q = src[c]; k = src[32 + c]; v = src[64 + c];
    }
    float qn = q * q, kn = k * k;
    for (int d = 16; d; d >>= 1) { qn += __shfl_down_sync(0xffffffff, qn, d); kn += __shfl_down_sync(0xffffffff, kn, d); }
    qn = fmaxf(sqrtf(__shfl_sync(0xffffffff, qn, 0)), 1e-6f);
    kn = fmaxf(sqrtf(__shfl_sync(0xffffffff, kn, 0)), 1e-6f);
    size_t base = static_cast<size_t>(h) * padded * 96 + t * 32 + c;
    packed[base] = q / qn * scale[h] * 5.65685424949238f;
    packed[base + padded * 32] = k / kn;
    packed[base + padded * 64] = v;
}
__global__ void vit_attend(const float* packed, float* out, int tokens, int padded, int heads) {
    extern __shared__ float scores[];
    __shared__ float reductions[128];
    int query = blockIdx.x % tokens, head = blockIdx.x / tokens, tid = threadIdx.x;
    const float* q = packed + static_cast<size_t>(head) * padded * 96 + query * 32;
    const float* k = packed + static_cast<size_t>(head) * padded * 96 + padded * 32;
    const float* v = k + padded * 32;
    float mx = -CUDART_INF_F;
    for (int t = tid; t < padded; t += blockDim.x) {
        float s = -CUDART_INF_F;
        if (t < tokens) {
            s = 0;
            for (int c = 0; c < 32; ++c) s += q[c] * k[t * 32 + c];
        }
        scores[t] = s; mx = fmaxf(mx, s);
    }
    reductions[tid] = mx; __syncthreads();
    for (int d = blockDim.x / 2; d; d >>= 1) {
        if (tid < d) reductions[tid] = fmaxf(reductions[tid], reductions[tid + d]); __syncthreads();
    }
    mx = reductions[0];
    // All warps must read the maximum before reusing the reduction storage.
    __syncthreads();
    float den = 0;
    for (int t = tid; t < padded; t += blockDim.x) { scores[t] = expf(scores[t] - mx); den += scores[t]; }
    reductions[tid] = den; __syncthreads();
    for (int d = blockDim.x / 2; d; d >>= 1) {
        if (tid < d) reductions[tid] += reductions[tid + d]; __syncthreads();
    }
    den = fmaxf(reductions[0], 1e-9f);
    if (tid < 32) {
        float result = 0;
        for (int t = 0; t < tokens; ++t) result += scores[t] / den * v[t * 32 + tid];
        out[(static_cast<size_t>(query) * heads + head) * 32 + tid] = result;
    }
}
// Split blocks use branch-major batched GEMMs but merge back to row-major.
__global__ void split_layout(const float* src, float* dst, int rows, int e, int bc, bool gather) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(rows) * e * bc) return;
    int c = i % bc, r = (i / bc) % rows, expert = i / (bc * rows);
    size_t row = (static_cast<size_t>(r) * e + expert) * bc + c;
    if (gather) dst[i] = src[row]; else dst[row] = src[i];
}
} // namespace

struct CudaEngine::Impl {
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaEvent_t io_events[4]{};
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    int device = 0;
    std::string name;
    Geometry g{};
    std::map<std::string, std::unique_ptr<Buffer<float>>> weights;
    std::map<std::string, std::vector<int>> shapes;
    std::unordered_map<std::string, std::vector<BlockCfg>> blocks[2];
    std::vector<BlockCfg> vit;
    Buffer<float> features, head, x, up, trans, pooled, vstate, skip[6];
    Buffer<float> byte_luts;
    Buffer<unsigned char> bgra_frames, bgra_output;
    HostBuffer<unsigned char> host_frames, host_output;
    Buffer<float> tmp, aux, merged, residual, qkv, attended, projected, packed;
    Buffer<float> route_norm, logits, gates, group_in, group_hidden, group_ffn, group_res, group_qkv, group_att;
    Buffer<int> pick, counts, counters, ids0, ids1, ids2, row_group;
    Buffer<unsigned long long> keys0, keys1, keys2;
    Buffer<unsigned char> sort_temp;
    std::map<std::string, std::vector<float>> captures;
    bool debugging = false, uploaded = false;

    ~Impl() {
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (handle) cublasDestroy(handle);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        for (auto event : io_events) if (event) cudaEventDestroy(event);
        if (stream) cudaStreamDestroy(stream);
    }
    const float* w(const std::string& key) const {
        auto it = weights.find(key);
        if (it == weights.end()) throw std::runtime_error("CUDA missing tensor " + key);
        return it->second->p;
    }
    void ensure_graph() {
        if (executable) return;
        run(); cu(cudaStreamSynchronize(stream));
        cu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        try { run(); }
        catch (...) {
            cudaGraph_t abandoned = nullptr; cudaStreamEndCapture(stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
            throw;
        }
        cu(cudaStreamEndCapture(stream, &graph));
        cu(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    }
    const float* optional(const std::string& key) const { return weights.count(key) ? w(key) : nullptr; }
    int dim(const std::string& key, int d) const { return shapes.at(key).at(d); }
    void record(const std::string& tag, const float* data, size_t n) {
        if (!debugging) return;
        auto& dst = captures[tag]; dst.resize(n);
        cu(cudaMemcpyAsync(dst.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cu(cudaStreamSynchronize(stream));
    }
    void linear(const float* in, float* out, int rows, int K, int N, const std::string& key,
                bool act = false, const float* original = nullptr, const float* scale = nullptr) {
        const float one = 1, zero = 0;
        blas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, rows, K, &one,
                         w(key + ".weight"), K, in, K, &zero, out, N));
        epilogue<<<grid(static_cast<size_t>(rows) * N), 256, 0, stream>>>
            (out, rows, N, optional(key + ".bias"), act, original, scale);
    }
    void batch(const float* in, float* out, int rows, int K, int N, int E, const std::string& key, bool act = false) {
        const float one = 1, zero = 0;
        blas(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, rows, K, &one,
             w(key), N, static_cast<long long>(K) * N, in, K, static_cast<long long>(rows) * K,
             &zero, out, N, static_cast<long long>(rows) * N, E));
        if (act) epilogue<<<grid(static_cast<size_t>(E) * rows * N), 256, 0, stream>>>(out, E * rows, N, nullptr, true, nullptr, nullptr);
    }
    void down(const float* in, float* out, int iw, int ih, int ow, int oh, int ch) {
        pool2<<<grid(static_cast<size_t>(ow) * oh * ch), 256, 0, stream>>>(in, out, iw, ih, ow, oh, ch);
    }
    void merge(const float* in, const float* sk, float* out, const float* a0, const float* a1,
               int iw, int ih, int ow, int oh, int ch) {
        merge2<<<grid(static_cast<size_t>(ow) * oh * ch), 256, 0, stream>>>(in, sk, out, a0, a1, iw, ih, ow, oh, ch);
    }
    void attention(const BlockCfg& cfg, const std::string& p, int width, int height, int ch, bool global) {
        int heads = ch / 32, rows = width * height;
        attended.resize(static_cast<size_t>(rows) * ch);
        if (global) {
            int padded = g.padded_vit_tokens;
            if (padded > 8192) throw std::runtime_error("CUDA ViT supports up to 8192 padded tokens; resize the image");
            packed.resize(static_cast<size_t>(heads) * padded * 96);
            vit_qkv<<<heads * padded, 32, 0, stream>>>(qkv.p, packed.p, rows, padded, heads, w(p + ".attention.scale"));
            vit_attend<<<heads * rows, 128, padded * sizeof(float), stream>>>(packed.p, attended.p, rows, padded, heads);
        } else {
            int win = cfg.window;
            if (win < 1 || win > 8) throw std::runtime_error("CUDA window size must be between 1 and 8");
            int windows = ((width + cfg.shift_x + win - 1) / win) * ((height + cfg.shift_y + win - 1) / win);
            packed.resize(static_cast<size_t>(windows) * heads * win * win * 96);
            int tasks = windows * heads * win * win;
            auto scale = w(p + ".attention.scale");
            int qkv_blocks = grid(static_cast<size_t>(tasks) * 32);
            // Eight independent normalization warps per block avoid the low
            // occupancy of one 32-thread block per slot, without changing sums.
            if (win == 4)
                window_qkv<4><<<qkv_blocks, 256, 0, stream>>>
                    (qkv.p, packed.p, width, height, heads, win, cfg.shift_x, cfg.shift_y, scale, tasks);
            else if (win == 8)
                window_qkv<8><<<qkv_blocks, 256, 0, stream>>>
                    (qkv.p, packed.p, width, height, heads, win, cfg.shift_x, cfg.shift_y, scale, tasks);
            else
                window_qkv<0><<<qkv_blocks, 256, 0, stream>>>
                    (qkv.p, packed.p, width, height, heads, win, cfg.shift_x, cfg.shift_y, scale, tasks);
            // Constant windows keep softmax scores in registers and eliminate
            // dynamic array indexing in the common trained 4x4/8x8 kernels.
            auto prior = w(p + ".attention.prior");
            int shared_bytes = (win * win * 96 + 32) * sizeof(float);
            if (win == 4)
                window_attend<4><<<windows * heads, 128, shared_bytes, stream>>>
                    (packed.p, attended.p, prior, width, height, heads, win, cfg.shift_x, cfg.shift_y);
            else if (win == 8)
                window_attend<8><<<windows * heads, 512, shared_bytes, stream>>>
                    (packed.p, attended.p, prior, width, height, heads, win, cfg.shift_x, cfg.shift_y);
            else
                window_attend<0><<<windows * heads, 128, shared_bytes, stream>>>
                    (packed.p, attended.p, prior, width, height, heads, win, cfg.shift_x, cfg.shift_y);
        }
    }
    int route(const float* state, int width, int height, int ch, int E, const std::string& p, bool windows) {
        if (ch > 1024) throw std::runtime_error("CUDA router supports at most 1024 channels");
        int items = windows ? ((width + 7) / 8) * ((height + 7) / 8) : width * height;
        int cap = (items + E - 1) / E * (windows ? 64 : 1);
        route_norm.resize(static_cast<size_t>(items) * ch); logits.resize(static_cast<size_t>(items) * E);
        pick.resize(items); counts.resize(E); counters.resize(E); gates.resize(items);
        ids0.resize(items); ids1.resize(items); ids2.resize(items);
        keys0.resize(items); keys1.resize(items); keys2.resize(items);
        row_group.resize(static_cast<size_t>(width) * height);
        router_input<<<items, 128, 0, stream>>>(state, route_norm.p, width, height, ch,
             w(p + ".router.0.weight"), w(p + ".router.0.bias"), windows);
        const float one = 1, zero = 0;
        blas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, E, items, ch, &one,
                        w(p + ".router.1.weight"), ch, route_norm.p, ch, &zero, logits.p, E));
        cu(cudaMemsetAsync(counts.p, 0, E * sizeof(int), stream));
        initial_pick<<<grid(items), 256, 0, stream>>>(logits.p, items, E, pick.p, counts.p, keys0.p, ids0.p);
        // Keys contain a nonnegative float margin plus an expert ID. Sort only
        // the populated bits; the victim pass needs just the low 32 bits because
        // its all-ones sentinel sorts after every nonnegative float margin.
        int key_bits = 32;
        for (int e = E - 1; e; e >>= 1) ++key_bits;
        size_t bytes = 0;
        cu(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys0.p, keys1.p, ids0.p, ids1.p, items, 0, 64, stream));
        sort_temp.resize(bytes);
        cu(cub::DeviceRadixSort::SortPairs(sort_temp.p, bytes, keys0.p, keys1.p, ids0.p, ids1.p, items, 0, key_bits, stream));
        victim_keys<<<grid(items), 256, 0, stream>>>(keys1.p, counts.p, keys2.p, items, E);
        cu(cub::DeviceRadixSort::SortPairs(sort_temp.p, bytes, keys2.p, keys0.p, ids1.p, ids2.p, items, 0, 32, stream));
        balanced_assign<<<grid(items), 256, 0, stream>>>(ids2.p, keys0.p, counts.p, pick.p, items, E);
        route_gate<<<grid(items), 256, 0, stream>>>(logits.p, pick.p, gates.p, items, E);
        group_in.resize(static_cast<size_t>(E) * cap * ch);
        cu(cudaMemsetAsync(counters.p, 0, E * sizeof(int), stream));
        cu(cudaMemsetAsync(group_in.p, 0, static_cast<size_t>(E) * cap * ch * sizeof(float), stream));
        make_group<<<items, 128, 0, stream>>>(state, group_in.p, row_group.p, pick.p, counters.p, cap, width, height, ch, windows);
        return cap;
    }
    void moe(const BlockCfg& cfg, const std::string& p, float* state, int width, int height, bool global) {
        int ch = dim(p + ".expand", 1), hidden = cfg.hidden, E = cfg.experts, rows = width * height;
        int cap = route(state, width, height, ch, E, p, !global);
        size_t grouped = static_cast<size_t>(E) * cap;
        group_hidden.resize(grouped * hidden); group_ffn.resize(grouped * ch);
        group_res.resize(grouped * ch); residual.resize(static_cast<size_t>(rows) * ch);
        batch(group_in.p, group_hidden.p, cap, ch, hidden, E, p + ".expand", true);
        batch(group_hidden.p, group_ffn.p, cap, hidden, ch, E, p + (global ? ".contract" : ".narrow"));
        cu(cudaMemsetAsync(group_res.p, 0, grouped * ch * sizeof(float), stream));
        moe_residual<<<grid(static_cast<size_t>(rows) * ch), 256, 0, stream>>>
            (group_ffn.p, state, residual.p, group_res.p, row_group.p, gates.p, w(p + ".aux_ffn"), width, height, ch, !global);
        if (cfg.attn) {
            record("ffn-" + p, residual.p, static_cast<size_t>(rows) * ch);
            group_qkv.resize(grouped * ch * 3); qkv.resize(static_cast<size_t>(rows) * ch * 3);
            batch(group_res.p, group_qkv.p, cap, ch, ch * 3, E, p + ".qkv");
            group_copy<<<grid(static_cast<size_t>(rows) * ch * 3), 256, 0, stream>>>(group_qkv.p, qkv.p, row_group.p, rows, ch * 3, false);
            attention(cfg, p, width, height, ch, global);
            record("att-" + p, attended.p, static_cast<size_t>(rows) * ch);
            group_att.resize(grouped * ch);
            cu(cudaMemsetAsync(group_att.p, 0, grouped * ch * sizeof(float), stream));
            group_copy<<<grid(static_cast<size_t>(rows) * ch), 256, 0, stream>>>(attended.p, group_att.p, row_group.p, rows, ch, true);
            batch(group_att.p, group_ffn.p, cap, ch, ch, E, p + ".proj");
            projected.resize(static_cast<size_t>(rows) * ch);
            group_copy<<<grid(static_cast<size_t>(rows) * ch), 256, 0, stream>>>(group_ffn.p, projected.p, row_group.p, rows, ch, false);
            add_attention<<<grid(static_cast<size_t>(rows) * ch), 256, 0, stream>>>(residual.p, projected.p, w(p + ".aux_attn"), state, rows, ch);
        } else cu(cudaMemcpyAsync(state, residual.p, static_cast<size_t>(rows) * ch * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        record("blk-" + p, state, static_cast<size_t>(rows) * ch);
    }
    void block(const BlockCfg& cfg, const std::string& p, float* state, int width, int height, bool global = false) {
        if (cfg.kind == "moe" || cfg.kind == "moe_vit") { moe(cfg, p, state, width, height, global); return; }
        int ch = dim(p + ".contract.weight", 0), rows = width * height;
        residual.resize(static_cast<size_t>(rows) * ch);
        if (cfg.kind == "ffn" || cfg.kind == "vit") {
            tmp.resize(static_cast<size_t>(rows) * cfg.hidden);
            linear(state, tmp.p, rows, ch, cfg.hidden, p + ".expand", true);
            linear(tmp.p, residual.p, rows, cfg.hidden, ch, p + ".contract", false, state, w(p + ".aux_ffn"));
        } else if (cfg.kind == "expert") {
            int E = cfg.experts, hidden = cfg.hidden, nc = cfg.narrow;
            aux.resize(static_cast<size_t>(E) * rows * hidden); merged.resize(static_cast<size_t>(E) * rows * nc);
            const float one = 1, zero = 0;
            blas(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden, rows, ch, &one,
                w(p + ".expand"), hidden, static_cast<long long>(ch) * hidden, state, ch, 0,
                &zero, aux.p, hidden, static_cast<long long>(rows) * hidden, E));
            epilogue<<<grid(static_cast<size_t>(E) * rows * hidden), 256, 0, stream>>>(aux.p, E * rows, hidden, nullptr, true, nullptr, nullptr);
            batch(aux.p, merged.p, rows, hidden, nc, E, p + ".narrow");
            // The expert-major flattened tensor is the trained legacy contract.
            linear(merged.p, residual.p, rows, ch, ch, p + ".contract", false, state, w(p + ".aux_ffn"));
        } else if (cfg.kind == "split") {
            int E = cfg.branches, bc = cfg.bc, mc = cfg.mc;
            tmp.resize(static_cast<size_t>(rows) * ch); merged.resize(static_cast<size_t>(rows) * ch);
            aux.resize(static_cast<size_t>(E) * rows * bc); projected.resize(static_cast<size_t>(E) * rows * mc);
            linear(state, tmp.p, rows, ch, ch, p + ".branch");
            split_layout<<<grid(static_cast<size_t>(rows) * ch), 256, 0, stream>>>(tmp.p, aux.p, rows, E, bc, true);
            batch(aux.p, projected.p, rows, bc, mc, E, p + ".w2", true);
            batch(projected.p, aux.p, rows, mc, bc, E, p + ".w3");
            split_layout<<<grid(static_cast<size_t>(rows) * ch), 256, 0, stream>>>(aux.p, merged.p, rows, E, bc, false);
            linear(merged.p, residual.p, rows, ch, ch, p + ".contract", false, state, w(p + ".aux_ffn"));
        } else throw std::runtime_error("CUDA unsupported block kind: " + cfg.kind);
        if (cfg.attn) {
            record("ffn-" + p, residual.p, static_cast<size_t>(rows) * ch);
            std::string tail = global ? p : p + ".tail";
            qkv.resize(static_cast<size_t>(rows) * ch * 3);
            linear(residual.p, qkv.p, rows, ch, ch * 3, tail + ".qkv");
            attention(cfg, tail, width, height, ch, global);
            record("att-" + p, attended.p, static_cast<size_t>(rows) * ch);
            projected.resize(static_cast<size_t>(rows) * ch);
            linear(attended.p, projected.p, rows, ch, ch, tail + ".proj");
            add_attention<<<grid(static_cast<size_t>(rows) * ch), 256, 0, stream>>>(residual.p, projected.p, w(tail + ".aux_attn"), state, rows, ch);
        } else cu(cudaMemcpyAsync(state, residual.p, static_cast<size_t>(rows) * ch * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        record("blk-" + p, state, static_cast<size_t>(rows) * ch);
    }
    void run() {
        const char* names[] = {"full", "d0", "d1", "d2", "d3", "d4"};
        int widths[6], heights[6], channels[6], rows[6];
        size_t max_state = 0;
        for (int i = 0; i < 6; ++i) {
            widths[i] = i ? g.levels[i - 1].width : g.full_width;
            heights[i] = i ? g.levels[i - 1].height : g.full_height;
            rows[i] = widths[i] * heights[i];
            std::string p = std::string("enc_blocks.") + names[i] + ".0";
            channels[i] = shapes.count(p + ".expand") ? dim(p + ".expand", 1) : dim(p + ".contract.weight", 0);
            skip[i].resize(static_cast<size_t>(rows[i]) * channels[i]);
            max_state = std::max(max_state, static_cast<size_t>(rows[i]) * channels[i]);
        }
        x.resize(max_state); up.resize(max_state); trans.resize(max_state);
        linear(features.p, x.p, rows[0], 16, channels[0], "adapter");
        for (int i = 0; i < 6; ++i) {
            const auto& bs = blocks[0].at(names[i]);
            for (size_t j = 0; j < bs.size(); ++j)
                block(bs[j], std::string("enc_blocks.") + names[i] + "." + std::to_string(j), x.p, widths[i], heights[i]);
            record(std::string("s-enc-") + names[i], x.p, static_cast<size_t>(rows[i]) * channels[i]);
            cu(cudaMemcpyAsync(skip[i].p, x.p, static_cast<size_t>(rows[i]) * channels[i] * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            if (i < 5) {
                pooled.resize(static_cast<size_t>(rows[i + 1]) * channels[i]);
                down(x.p, pooled.p, widths[i], heights[i], widths[i + 1], heights[i + 1], channels[i]);
                linear(pooled.p, x.p, rows[i + 1], channels[i], channels[i + 1], std::string("enc_trans.") + names[i + 1]);
                record(std::string("trans-") + names[i + 1], x.p, static_cast<size_t>(rows[i + 1]) * channels[i + 1]);
            }
        }
        int vw = g.levels[5].width, vh = g.levels[5].height, vc = dim("vit_in.weight", 0);
        pooled.resize(static_cast<size_t>(vw) * vh * channels[5]); vstate.resize(static_cast<size_t>(vw) * vh * vc);
        down(skip[5].p, pooled.p, widths[5], heights[5], vw, vh, channels[5]);
        linear(pooled.p, vstate.p, vw * vh, channels[5], vc, "vit_in");
        record("trans-vit", vstate.p, static_cast<size_t>(vw) * vh * vc);
        for (size_t j = 0; j < vit.size(); ++j) block(vit[j], "vit_blocks." + std::to_string(j), vstate.p, vw * vh, 1, true);
        record("s-vit", vstate.p, static_cast<size_t>(vw) * vh * vc);
        linear(vstate.p, x.p, vw * vh, vc, channels[5], "vit_out");
        int pw = vw, ph = vh;
        for (int i = 5; i >= 1; --i) {
            const float* in = x.p;
            if (i < 5) {
                linear(x.p, trans.p, pw * ph, channels[i + 1], channels[i], std::string("dec_trans.") + names[i]);
                in = trans.p;
            }
            merge(in, skip[i].p, up.p, nullptr, w(std::string("merge.") + names[i] + ".aux"), pw, ph, widths[i], heights[i], channels[i]);
            const auto& bs = blocks[1].at(names[i]);
            for (size_t j = 0; j < bs.size(); ++j)
                block(bs[j], std::string("dec_blocks.") + names[i] + "." + std::to_string(j), up.p, widths[i], heights[i]);
            record(std::string("s-dec-") + names[i], up.p, static_cast<size_t>(rows[i]) * channels[i]);
            cu(cudaMemcpyAsync(x.p, up.p, static_cast<size_t>(rows[i]) * channels[i] * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            pw = widths[i]; ph = heights[i];
        }
        const float* pair = w("post.aux_pair");
        merge(x.p, skip[0].p, up.p, pair, pair + channels[0], pw, ph, widths[0], heights[0], channels[0]);
        const auto& bs = blocks[1].at("full");
        for (size_t j = 0; j < bs.size(); ++j) block(bs[j], "dec_blocks.full." + std::to_string(j), up.p, widths[0], heights[0]);
        linear(up.p, head.p, rows[0], channels[0], 4, "head");
        record("s-head", head.p, static_cast<size_t>(rows[0]) * 4);
        cu(cudaGetLastError());
    }
};

CudaEngine::CudaEngine(const Engine& model, int device) : impl_(std::make_unique<Impl>()) {
    auto& p = *impl_; p.device = device;
    cu(cudaSetDevice(device));
    cudaDeviceProp properties{}; cu(cudaGetDeviceProperties(&properties, device));
    p.name = properties.name; p.g = model.geom_; p.vit = model.vit_blocks_;
    p.blocks[0] = model.blocks_[0]; p.blocks[1] = model.blocks_[1];
    cu(cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking));
    cu(cudaEventCreate(&p.start)); cu(cudaEventCreate(&p.stop));
    for (auto& event : p.io_events) cu(cudaEventCreate(&event));
    blas(cublasCreate(&p.handle)); blas(cublasSetStream(p.handle, p.stream));
    // Preserve f32 precision. TF32 silently changes router logits and is disabled.
    blas(cublasSetMathMode(p.handle, CUBLAS_PEDANTIC_MATH));
    blas(cublasSetAtomicsMode(p.handle, CUBLAS_ATOMICS_NOT_ALLOWED));
    for (const auto& kv : model.tensors_) {
        size_t n = 1; for (int d : kv.second.shape) n *= d;
        auto buffer = std::make_unique<Buffer<float>>(); buffer->resize(n);
        cu(cudaMemcpyAsync(buffer->p, kv.second.data, n * sizeof(float), cudaMemcpyHostToDevice, p.stream));
        p.weights.emplace(kv.first, std::move(buffer)); p.shapes.emplace(kv.first, kv.second.shape);
    }
    p.features.resize(static_cast<size_t>(p.g.full_rows) * 16);
    p.head.resize(static_cast<size_t>(p.g.full_rows) * 4);
    size_t visible = static_cast<size_t>(p.g.valid_width) * p.g.valid_height;
    p.bgra_frames.resize(visible * 8); p.bgra_output.resize(visible * 5);
    p.host_frames.allocate(visible * 8); p.host_output.allocate(visible * 5);
    std::array<float, 512> luts{};
    for (int i = 0; i < 256; ++i) {
        luts[i] = round_f16(round_f16(round_f16(i / 255.0f) - 0.5f) * 0.125f);
        luts[256 + i] = i / 255.0f;
    }
    p.byte_luts.resize(luts.size());
    cu(cudaMemcpyAsync(p.byte_luts.p, luts.data(), luts.size() * sizeof(float), cudaMemcpyHostToDevice, p.stream));
    cu(cudaStreamSynchronize(p.stream));
}
CudaEngine::~CudaEngine() = default;
const std::string& CudaEngine::device_name() const { return impl_->name; }
void CudaEngine::upload(const float* features) {
    auto& p = *impl_; cu(cudaSetDevice(p.device));
    cu(cudaMemcpyAsync(p.features.p, features, static_cast<size_t>(p.g.full_rows) * 16 * sizeof(float), cudaMemcpyHostToDevice, p.stream));
    cu(cudaStreamSynchronize(p.stream)); p.uploaded = true;
}
void CudaEngine::download(float* head) {
    auto& p = *impl_; cu(cudaSetDevice(p.device));
    cu(cudaMemcpyAsync(head, p.head.p, static_cast<size_t>(p.g.full_rows) * 4 * sizeof(float), cudaMemcpyDeviceToHost, p.stream));
    cu(cudaStreamSynchronize(p.stream));
}
void CudaEngine::download_features(float* features) {
    auto& p = *impl_; cu(cudaSetDevice(p.device));
    cu(cudaMemcpyAsync(features, p.features.p, static_cast<size_t>(p.g.full_rows) * 16 * sizeof(float), cudaMemcpyDeviceToHost, p.stream));
    cu(cudaStreamSynchronize(p.stream));
}
double CudaEngine::forward(bool debug) {
    auto& p = *impl_; cu(cudaSetDevice(p.device));
    if (!p.uploaded) throw std::runtime_error("CUDA features must be uploaded before forward");
    p.debugging = debug; p.captures.clear();
    if (!debug) p.ensure_graph();
    cu(cudaEventRecord(p.start, p.stream));
    if (debug) p.run(); else cu(cudaGraphLaunch(p.executable, p.stream));
    cu(cudaEventRecord(p.stop, p.stream)); cu(cudaEventSynchronize(p.stop));
    float ms = 0; cu(cudaEventElapsedTime(&ms, p.start, p.stop)); return ms;
}
const std::map<std::string, std::vector<float>>& CudaEngine::debug() const { return impl_->captures; }

CudaBgraTimings CudaEngine::infer_bgra(const std::uint8_t* previous, size_t previous_stride,
                                     const std::uint8_t* current, size_t current_stride,
                                     float blend_scale, std::uint8_t* bgra, std::uint8_t* confidence) {
    auto begin = std::chrono::steady_clock::now();
    auto& p = *impl_; cu(cudaSetDevice(p.device));
    size_t row_bytes = static_cast<size_t>(p.g.valid_width) * 4;
    if (!p.uploaded || !previous || !current || !bgra || !confidence ||
        previous_stride < row_bytes || current_stride < row_bytes || !std::isfinite(blend_scale))
        throw std::runtime_error("CUDA temporal BGRA requires initialized lanes, valid pointers, strides and blend scale");
    size_t pixels = static_cast<size_t>(p.g.valid_width) * p.g.valid_height;
    for (int y = 0; y < p.g.valid_height; ++y) {
        std::memcpy(p.host_frames.p + y * row_bytes, previous + y * previous_stride, row_bytes);
        std::memcpy(p.host_frames.p + pixels * 4 + y * row_bytes, current + y * current_stride, row_bytes);
    }
    auto prepared = std::chrono::steady_clock::now();
    p.debugging = false; p.captures.clear(); p.ensure_graph();
    cu(cudaEventRecord(p.start, p.stream));
    cu(cudaMemcpyAsync(p.bgra_frames.p, p.host_frames.p, pixels * 8, cudaMemcpyHostToDevice, p.stream));
    cu(cudaEventRecord(p.io_events[0], p.stream));
    temporal_bgra<<<grid(static_cast<size_t>(p.g.full_rows) * 6), 256, 0, p.stream>>>
        (p.bgra_frames.p, p.features.p, p.byte_luts.p, p.g.valid_width, p.g.valid_height, p.g.full_width, p.g.full_height);
    cu(cudaEventRecord(p.io_events[1], p.stream));
    cu(cudaGraphLaunch(p.executable, p.stream));
    cu(cudaEventRecord(p.io_events[2], p.stream));
    redraw_bgra<<<grid(pixels), 256, 0, p.stream>>>
        (p.bgra_frames.p + pixels * 4, p.head.p, p.byte_luts.p + 256, p.bgra_output.p,
         p.g.valid_width, p.g.valid_height, p.g.full_width, blend_scale);
    cu(cudaEventRecord(p.io_events[3], p.stream));
    cu(cudaMemcpyAsync(p.host_output.p, p.bgra_output.p, pixels * 5, cudaMemcpyDeviceToHost, p.stream));
    cu(cudaEventRecord(p.stop, p.stream)); cu(cudaEventSynchronize(p.stop));
    auto downloaded = std::chrono::steady_clock::now();
    std::memcpy(bgra, p.host_output.p, pixels * 4);
    std::memcpy(confidence, p.host_output.p + pixels * 4, pixels);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    auto gpu_elapsed = [](cudaEvent_t a, cudaEvent_t b) {
        float ms = 0; cu(cudaEventElapsedTime(&ms, a, b)); return static_cast<double>(ms);
    };
    CudaBgraTimings timing;
    timing.prepare_ms = elapsed(begin, prepared) + gpu_elapsed(p.io_events[0], p.io_events[1]);
    timing.transfer_ms = gpu_elapsed(p.start, p.io_events[0]) + gpu_elapsed(p.io_events[3], p.stop);
    timing.output_ms = gpu_elapsed(p.io_events[2], p.io_events[3]) + elapsed(downloaded, end);
    timing.device_forward_ms = gpu_elapsed(p.io_events[1], p.io_events[2]);
    // Attribute submission/wait/graph initialization overhead to forward so
    // the stages still sum to actual host elapsed time rather than GPU time only.
    timing.forward_ms = std::max(0.0, elapsed(begin, end) - timing.prepare_ms - timing.transfer_ms - timing.output_ms);
    return timing;
}
