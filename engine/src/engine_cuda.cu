#include "engine.h"
#include "engine_cuda.h"
#include "f16.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
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
thread_local size_t allocated_bytes = 0, high_water_bytes = 0;
thread_local bool sample_device_memory = false;
thread_local size_t device_total_bytes = 0, device_initial_free = 0, device_min_free = 0;
void sample_memory() {
    if (!sample_device_memory) return;
    size_t free_bytes, total;
    cu(cudaMemGetInfo(&free_bytes, &total));
    device_min_free = std::min(device_min_free, free_bytes);
}
template<class T> struct Buffer {
    T* p = nullptr;
    size_t capacity = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (p) { cudaFree(p); allocated_bytes -= capacity * sizeof(T); } }
    void resize(size_t n) {
        if (n <= capacity) return;
        T* next = nullptr;
        cu(cudaMalloc(reinterpret_cast<void**>(&next), n * sizeof(T)));
        allocated_bytes += n * sizeof(T);
        high_water_bytes = std::max(high_water_bytes, allocated_bytes);
        sample_memory();
        if (p) { cu(cudaFree(p)); allocated_bytes -= capacity * sizeof(T); }
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

#include "shared_moe_cuda.cuh"
#include "cnn_cuda.cuh"

// q/k/v are packed [window, head, slot, channel]. Invalid window keys stay zero,
// but remain in the softmax denominator: the trained window mask is query-only.
template<int Window>
__global__ void window_qkv(const float* qkv, float* packed, int width, int height,
                           int heads, int runtime_win, int sx, int sy, const float* scale, int tasks, int dilation = 1) {
    const int win = Window ? Window : runtime_win;
    int item = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (item >= tasks) return; // Whole warps are inactive; normalization stays warp-wide.
    int extent = (win - 1) * dilation + 1;
    int slots = win * win, wy = (height + sy + extent - 1) / extent;
    int slot = item % slots, wh = item / slots, head = wh % heads, window = wh / heads;
    int x = window / wy * extent - sx + slot % win * dilation, y = window % wy * extent - sy + slot / win * dilation;
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
                              int width, int height, int heads, int runtime_win, int sx, int sy, int dilation = 1) {
    extern __shared__ float shared[];
    const int win = Window ? Window : runtime_win;
    int wh = blockIdx.x, head = wh % heads, window = wh / heads;
    int extent = (win - 1) * dilation + 1;
    int slots = win * win, wy = (height + sy + extent - 1) / extent;
    int ox = window / wy * extent - sx, oy = window % wy * extent - sy;
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
        int x = ox + query % win * dilation, y = oy + query / win * dilation;
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
    Buffer<unsigned char> sr_sort_temp, sr_scan_temp;
    Buffer<int> sr_keys0, sr_keys1, sr_lengths_buf, sr_item_offsets, sr_offsets, sr_tiles;
    Buffer<float> shared_hidden, shared_out, shared_qkv, int_capture;
    std::map<std::string, std::vector<float>> captures;
    bool debugging = false, uploaded = false;
    bool capture_routes_only = false;
    bool shared_routed = false;
    bool cnn = false;
    int shared_experts = 0, routed_experts = 0;
    int cnn_channels[7]{};
    Buffer<float> cnn_columns, cnn_normed, cnn_conv_out, cnn_cat;
    bool profiling = false;
    bool capturing = false;
    struct EventPair { std::string name; cudaEvent_t begin, end; };
    std::vector<EventPair> profile_events;
    size_t profile_cursor = 0;
    std::map<std::string, double> profile_times;
    struct Scope {
        Impl& owner; EventPair event{}; bool active;
        Scope(Impl& p, const std::string& name) : owner(p), active(p.profiling) {
            if (active) {
                if (p.profile_cursor == p.profile_events.size()) {
                    event.name = name; cu(cudaEventCreate(&event.begin)); cu(cudaEventCreate(&event.end));
                    p.profile_events.push_back(event);
                } else event = p.profile_events[p.profile_cursor];
                ++p.profile_cursor;
                if (p.capturing) cu(cudaEventRecordWithFlags(event.begin, p.stream, cudaEventRecordExternal));
                else cu(cudaEventRecord(event.begin, p.stream));
            }
        }
        ~Scope() {
            if (active) {
                if (owner.capturing) cudaEventRecordWithFlags(event.end, owner.stream, cudaEventRecordExternal);
                else cudaEventRecord(event.end, owner.stream);
            }
            sample_memory();
        }
    };
    void finish_profile() {
        profile_times.clear();
        for (const auto& event : profile_events) {
            float ms = 0; cu(cudaEventElapsedTime(&ms, event.begin, event.end));
            profile_times[event.name] += ms;
        }
    }

    ~Impl() {
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        for (const auto& event : profile_events) { cudaEventDestroy(event.begin); cudaEventDestroy(event.end); }
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
        capturing = true;
        try { run(); }
        catch (...) {
            cudaGraph_t abandoned = nullptr; cudaStreamEndCapture(stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
            capturing = false;
            throw;
        }
        cu(cudaStreamEndCapture(stream, &graph));
        capturing = false;
        cu(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    }
    const float* optional(const std::string& key) const { return weights.count(key) ? w(key) : nullptr; }
    int dim(const std::string& key, int d) const { return shapes.at(key).at(d); }
    void record(const std::string& tag, const float* data, size_t n) {
        if (!debugging) return;
        if (capture_routes_only && tag.rfind("pick-", 0) && tag.rfind("logits-", 0) && tag.rfind("gate-", 0) && tag.rfind("load-", 0)) return;
        auto& dst = captures[tag]; dst.resize(n);
        cu(cudaMemcpyAsync(dst.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cu(cudaStreamSynchronize(stream));
    }
    void record_int(const std::string& tag, const int* data, int n) {
        if (!debugging) return;
        int_capture.resize(n);
        sr_int_float<<<grid(n), 256, 0, stream>>>(data, int_capture.p, n);
        record(tag, int_capture.p, n);
    }
    void shared_mm(const float* in, float* out, int rows, int K, int N, const std::string& key, bool act = false) {
        Scope profile(*this, "shared_gemm:" + key);
        const float one = 1, zero = 0;
        blas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, rows, K, &one,
                        w(key), N, in, K, &zero, out, N));
        if (act) epilogue<<<grid(static_cast<size_t>(rows) * N), 256, 0, stream>>>(out, rows, N, nullptr, true, nullptr, nullptr);
    }
    void routed_mm(const float* in, float* out, int rows, int K, int N, const std::string& key, bool act = false) {
        Scope profile(*this, "routed_gemm:" + key);
        if (N == 32) {
            dim3 blocks((rows + 31) / 32 + 15, 1);
            sr_grouped_gemm<32><<<blocks, 256, 0, stream>>>(in, w(key), out, sr_offsets.p, sr_tiles.p, K, N, act, routed_experts, shared_experts);
        } else {
            dim3 blocks((rows + 31) / 32 + 15, (N + 63) / 64);
            sr_grouped_gemm<64><<<blocks, 256, 0, stream>>>(in, w(key), out, sr_offsets.p, sr_tiles.p, K, N, act, routed_experts, shared_experts);
        }
    }
    void shared_route(const float* state, int width, int height, int ch, const std::string& p, bool windows) {
        Scope profile(*this, "route:" + p);
        int rows = width * height, items = windows ? ((width + 7) / 8) * ((height + 7) / 8) : rows;
        const int E = routed_experts;
        route_norm.resize(static_cast<size_t>(items) * ch); logits.resize(static_cast<size_t>(items) * E);
        pick.resize(items); gates.resize(items); counts.resize(E); ids0.resize(items); ids1.resize(items);
        sr_keys0.resize(items); sr_keys1.resize(items); sr_lengths_buf.resize(items); sr_item_offsets.resize(items);
        sr_offsets.resize(E + 1); sr_tiles.resize(E + 1); row_group.resize(rows); group_in.resize(static_cast<size_t>(rows) * ch);
        {
        Scope math(*this, "router_math:" + p);
        cu(cudaMemsetAsync(counts.p, 0, E * sizeof(int), stream));
        sr_router<<<items, 512, 0, stream>>>(state, logits.p, w(p + ".router.0.weight"), w(p + ".router.0.bias"),
            w(p + ".router.1.weight"), w(p + ".router_bias"), pick.p, gates.p, counts.p, width, height, ch, windows, E);
        }
        {
        Scope grouping(*this, "grouping:" + p);
        if (items <= 4096) {
            sr_compact_map<<<E, 256, 0, stream>>>(pick.p, counts.p, row_group.p, sr_offsets.p, sr_tiles.p,
                                                  items, width, height, windows, E);
            group_copy<<<grid(static_cast<size_t>(rows) * ch), 256, 0, stream>>>(state, group_in.p, row_group.p, rows, ch, true);
        } else {
        // Radix sort is asymptotically better for large full-resolution grids.
        // sr_pick also supplies stable unit IDs for this path.
        cu(cudaMemsetAsync(counts.p, 0, E * sizeof(int), stream));
        sr_pick<<<grid(items), 256, 0, stream>>>(logits.p, w(p + ".router_bias"), pick.p, gates.p,
            sr_keys0.p, ids0.p, counts.p, items, width, height, windows, E, float(E));
        size_t sort_bytes = 0, scan_bytes = 0;
        cu(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, sr_keys0.p, sr_keys1.p, ids0.p, ids1.p, items, 0, 4, stream));
        sr_sort_temp.resize(sort_bytes);
        cu(cub::DeviceRadixSort::SortPairs(sr_sort_temp.p, sort_bytes, sr_keys0.p, sr_keys1.p, ids0.p, ids1.p, items, 0, 4, stream));
        sr_lengths<<<grid(items), 256, 0, stream>>>(ids1.p, sr_lengths_buf.p, items, width, height, windows);
        cu(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, sr_lengths_buf.p, sr_item_offsets.p, items, stream));
        sr_scan_temp.resize(scan_bytes);
        cu(cub::DeviceScan::ExclusiveSum(sr_scan_temp.p, scan_bytes, sr_lengths_buf.p, sr_item_offsets.p, items, stream));
        sr_descriptors<<<1, 32, 0, stream>>>(counts.p, sr_offsets.p, sr_tiles.p, E);
        sr_pack<<<items, 128, 0, stream>>>(state, group_in.p, row_group.p, ids1.p, sr_item_offsets.p, width, height, ch, windows);
        }
        }
        record("logits-" + p, logits.p, static_cast<size_t>(items) * E);
        record("gate-" + p, gates.p, items); record_int("pick-" + p, pick.p, items);
        record_int("load-" + p, counts.p, E);
        record_int("row-group-" + p, row_group.p, rows);
    }
    void mix(const float* shared, const float* routed, float* out, const std::string& p,
             int width, int height, int channels, bool windows, const float* bias = nullptr,
             const float* original = nullptr, const float* aux_scale = nullptr, bool attention_tail = false) {
        Scope profile(*this, "fusion:" + p);
        sr_mix<<<grid(static_cast<size_t>(width) * height * channels), 256, 0, stream>>>
            (shared_experts ? shared : nullptr, routed, out, row_group.p, pick.p, gates.p, bias, original, aux_scale,
             width, height, channels, windows, attention_tail, shared_experts);
    }
    void shared_transition(const float* in, float* out, int width, int height, int K, int N,
                           const std::string& p, bool windows) {
        Scope profile(*this, "transition:" + p);
        int rows = width * height;
        shared_route(in, width, height, K, p, windows);
        shared_out.resize(static_cast<size_t>(rows) * N); group_ffn.resize(static_cast<size_t>(rows) * N);
        if (shared_experts) shared_mm(in, shared_out.p, rows, K, N, p + ".weight");
        routed_mm(group_in.p, group_ffn.p, rows, K, N, p + ".weight");
        mix(shared_out.p, group_ffn.p, out, p, width, height, N, windows, w(p + ".bias"));
        record("output-" + p, out, static_cast<size_t>(rows) * N);
    }
    void shared_block(const BlockCfg& cfg, const std::string& p, float* state, int width, int height, bool global) {
        Scope profile(*this, "block:" + p);
        int ch = dim(p + ".expand", 1), rows = width * height, hidden = cfg.hidden;
        shared_route(state, width, height, ch, p, !global);
        shared_out.resize(static_cast<size_t>(rows) * ch); group_ffn.resize(static_cast<size_t>(rows) * ch);
        residual.resize(static_cast<size_t>(rows) * ch);
        std::string contract = p + (global ? ".contract" : ".narrow");
        if (ch == 32 && hidden == 32) {
            {
                Scope profile(*this, "shared_gemm:" + p + ".fused_ffn");
                if (shared_experts) sr_ffn32<false><<<(rows + 31) / 32, 256, 0, stream>>>(state, w(p + ".expand"), w(contract), shared_out.p, nullptr, nullptr, rows);
            }
            {
                Scope profile(*this, "routed_gemm:" + p + ".fused_ffn");
                sr_ffn32<true><<<(rows + 31) / 32 + 15, 256, 0, stream>>>(group_in.p, w(p + ".expand"), w(contract), group_ffn.p, sr_offsets.p, sr_tiles.p, rows, routed_experts, shared_experts);
            }
        } else {
        shared_hidden.resize(static_cast<size_t>(rows) * hidden); group_hidden.resize(static_cast<size_t>(rows) * hidden);
        if (shared_experts) shared_mm(state, shared_hidden.p, rows, ch, hidden, p + ".expand", true);
        routed_mm(group_in.p, group_hidden.p, rows, ch, hidden, p + ".expand", true);
        if (shared_experts) shared_mm(shared_hidden.p, shared_out.p, rows, hidden, ch, contract);
        routed_mm(group_hidden.p, group_ffn.p, rows, hidden, ch, contract);
        }
        if (shared_experts) record("ffn_shared-" + p, shared_out.p, static_cast<size_t>(rows) * ch);
        if (debugging) {
            projected.resize(static_cast<size_t>(rows) * ch);
            group_copy<<<grid(static_cast<size_t>(rows) * ch), 256, 0, stream>>>(group_ffn.p, projected.p, row_group.p, rows, ch, false);
            record("ffn_routed-" + p, projected.p, static_cast<size_t>(rows) * ch);
        }
        mix(shared_out.p, group_ffn.p, residual.p, p, width, height, ch, !global, nullptr, state, w(p + ".aux_ffn"));
        record("ffn-" + p, residual.p, static_cast<size_t>(rows) * ch);
        if (cfg.attn) {
            group_res.resize(static_cast<size_t>(rows) * ch); group_qkv.resize(static_cast<size_t>(rows) * ch * 3);
            shared_qkv.resize(static_cast<size_t>(rows) * ch * 3); qkv.resize(static_cast<size_t>(rows) * ch * 3);
            group_copy<<<grid(static_cast<size_t>(rows) * ch), 256, 0, stream>>>(residual.p, group_res.p, row_group.p, rows, ch, true);
            if (shared_experts) shared_mm(residual.p, shared_qkv.p, rows, ch, ch * 3, p + ".qkv");
            routed_mm(group_res.p, group_qkv.p, rows, ch, ch * 3, p + ".qkv");
            mix(shared_qkv.p, group_qkv.p, qkv.p, p, width, height, ch * 3, !global);
            record("qkv-" + p, qkv.p, static_cast<size_t>(rows) * ch * 3);
            attention(cfg, p, width, height, ch, global);
            record("att-" + p, attended.p, static_cast<size_t>(rows) * ch);
            group_att.resize(static_cast<size_t>(rows) * ch);
            group_copy<<<grid(static_cast<size_t>(rows) * ch), 256, 0, stream>>>(attended.p, group_att.p, row_group.p, rows, ch, true);
            if (shared_experts) shared_mm(attended.p, shared_out.p, rows, ch, ch, p + ".proj");
            routed_mm(group_att.p, group_ffn.p, rows, ch, ch, p + ".proj");
            mix(shared_out.p, group_ffn.p, state, p, width, height, ch, !global, nullptr, residual.p, w(p + ".aux_attn"), true);
        } else cu(cudaMemcpyAsync(state, residual.p, static_cast<size_t>(rows) * ch * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        record("blk-" + p, state, static_cast<size_t>(rows) * ch);
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
        Scope profile(*this, "attention:" + p);
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
            int extent = (win - 1) * cfg.dilation + 1;
            int windows = ((width + cfg.shift_x + extent - 1) / extent) * ((height + cfg.shift_y + extent - 1) / extent);
            if (cfg.dilation != 1) cu(cudaMemsetAsync(attended.p, 0, size_t(rows) * ch * sizeof(float), stream));
            packed.resize(static_cast<size_t>(windows) * heads * win * win * 96);
            int tasks = windows * heads * win * win;
            auto scale = w(p + ".attention.scale");
            int qkv_blocks = grid(static_cast<size_t>(tasks) * 32);
            // Eight independent normalization warps per block avoid the low
            // occupancy of one 32-thread block per slot, without changing sums.
            if (win == 4)
                window_qkv<4><<<qkv_blocks, 256, 0, stream>>>
                    (qkv.p, packed.p, width, height, heads, win, cfg.shift_x, cfg.shift_y, scale, tasks, cfg.dilation);
            else if (win == 8)
                window_qkv<8><<<qkv_blocks, 256, 0, stream>>>
                    (qkv.p, packed.p, width, height, heads, win, cfg.shift_x, cfg.shift_y, scale, tasks, cfg.dilation);
            else
                window_qkv<0><<<qkv_blocks, 256, 0, stream>>>
                    (qkv.p, packed.p, width, height, heads, win, cfg.shift_x, cfg.shift_y, scale, tasks, cfg.dilation);
            // Constant windows keep softmax scores in registers and eliminate
            // dynamic array indexing in the common trained 4x4/8x8 kernels.
            auto prior = w(p + ".attention.prior");
            int shared_bytes = (win * win * 96 + 32) * sizeof(float);
            if (win == 4)
                window_attend<4><<<windows * heads, 128, shared_bytes, stream>>>
                    (packed.p, attended.p, prior, width, height, heads, win, cfg.shift_x, cfg.shift_y, cfg.dilation);
            else if (win == 8)
                window_attend<8><<<windows * heads, 512, shared_bytes, stream>>>
                    (packed.p, attended.p, prior, width, height, heads, win, cfg.shift_x, cfg.shift_y, cfg.dilation);
            else
                window_attend<0><<<windows * heads, 128, shared_bytes, stream>>>
                    (packed.p, attended.p, prior, width, height, heads, win, cfg.shift_x, cfg.shift_y, cfg.dilation);
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
        if (cfg.kind == "shared_routed_moe") { shared_block(cfg, p, state, width, height, global); return; }
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
    void cnn_conv(const float* in, float* out, int width, int height, const std::string& p) {
        Scope profile(*this, "convolution:" + p);
        int O = dim(p + ".weight", 0), I = dim(p + ".weight", 1), k = dim(p + ".weight", 2);
        int rows = width * height, K = I * k * k, tile = k == 1 ? rows : 4096;
        if (k != 1) cnn_columns.resize(size_t(std::min(tile, rows)) * K);
        const float one = 1, zero = 0;
        for (int start = 0; start < rows; start += tile) {
            int count = std::min(tile, rows - start);
            const float* columns = in;
            if (k != 1) {
                cnn_im2col<<<grid(size_t(count) * K), 256, 0, stream>>>(in, cnn_columns.p, start, count, width, height, I, k);
                columns = cnn_columns.p;
            }
            blas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, O, count, K, &one,
                            w("cnn:" + p + ".weight"), O, columns, K, &zero, out + size_t(start) * O, O));
        }
        epilogue<<<grid(size_t(rows) * O), 256, 0, stream>>>(out, rows, O, w(p + ".bias"), false, nullptr, nullptr);
    }
    void cnn_norm(const float* in, float* out, int rows, int channels, const std::string& p, bool act) {
        Scope profile(*this, "group_norm:" + p);
        cnn_group_norm<<<8, 256, 0, stream>>>(in, out, rows, channels, w(p + ".weight"), w(p + ".bias"), act);
    }
    void cnn_block(const std::string& p, int width, int height) {
        Scope profile(*this, "block:" + p);
        int rows = width * height, ch = dim(p + ".conv.weight", 0), h = dim(p + ".ffn.bank.0.0.weight", 0);
        size_t size = size_t(rows) * ch;
        cnn_normed.resize(size); cnn_conv_out.resize(size);
        cnn_norm(x.p, cnn_normed.p, rows, ch, p + ".norm1", true);
        cnn_conv(cnn_normed.p, cnn_conv_out.p, width, height, p + ".conv");
        cnn_add<<<grid(size), 256, 0, stream>>>(x.p, cnn_conv_out.p, size);
        cnn_norm(x.p, cnn_normed.p, rows, ch, p + ".norm2", false);
        logits.resize(size_t(rows) * 8);
        cnn_conv(cnn_normed.p, logits.p, width, height, p + ".ffn.router");
        pick.resize(rows); gates.resize(rows); counts.resize(8); ids0.resize(rows); ids1.resize(rows);
        sr_keys0.resize(rows); sr_keys1.resize(rows); sr_offsets.resize(9); sr_tiles.resize(9);
        sr_lengths_buf.resize(rows); sr_item_offsets.resize(rows); row_group.resize(rows); group_in.resize(size);
        group_hidden.resize(size_t(rows) * h); group_ffn.resize(size);
        cu(cudaMemsetAsync(counts.p, 0, 8 * sizeof(int), stream));
        sr_pick<<<grid(rows), 256, 0, stream>>>(logits.p, nullptr, pick.p, gates.p, sr_keys0.p, ids0.p, counts.p, rows, rows, 1, false, 8, 1.f, true);
        if (rows <= 4096) {
            sr_compact_map<<<8, 256, 0, stream>>>(pick.p, counts.p, row_group.p, sr_offsets.p, sr_tiles.p, rows, rows, 1, false, 8);
        } else {
            size_t sort_bytes = 0;
            cu(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, sr_keys0.p, sr_keys1.p, ids0.p, ids1.p, rows, 0, 3, stream));
            sr_sort_temp.resize(sort_bytes);
            cu(cub::DeviceRadixSort::SortPairs(sr_sort_temp.p, sort_bytes, sr_keys0.p, sr_keys1.p, ids0.p, ids1.p, rows, 0, 3, stream));
            // Pixel routes have unit length; sorted positions already are offsets.
            sr_lengths<<<grid(rows), 256, 0, stream>>>(ids1.p, sr_lengths_buf.p, rows, rows, 1, false);
            size_t scan_bytes = 0;
            cu(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, sr_lengths_buf.p, sr_item_offsets.p, rows, stream));
            sr_scan_temp.resize(scan_bytes);
            cu(cub::DeviceScan::ExclusiveSum(sr_scan_temp.p, scan_bytes, sr_lengths_buf.p, sr_item_offsets.p, rows, stream));
            sr_descriptors<<<1, 32, 0, stream>>>(counts.p, sr_offsets.p, sr_tiles.p, 8);
            sr_pack<<<rows, 128, 0, stream>>>(cnn_normed.p, group_in.p, row_group.p, ids1.p, sr_item_offsets.p, rows, 1, ch, false);
        }
        if (rows <= 4096) group_copy<<<grid(size), 256, 0, stream>>>(cnn_normed.p, group_in.p, row_group.p, rows, ch, true);
        const std::string f = "cnn:" + p + ".ffn";
        dim3 expand_grid((rows + 31) / 32 + 7, (h + 63) / 64), narrow_grid((rows + 31) / 32 + 7, (ch + 63) / 64);
        sr_grouped_gemm<64><<<expand_grid, 256, 0, stream>>>(group_in.p, w(f + ".expand"), group_hidden.p, sr_offsets.p, sr_tiles.p, ch, h, true, 8, 0, w(f + ".expand_bias"));
        sr_grouped_gemm<64><<<narrow_grid, 256, 0, stream>>>(group_hidden.p, w(f + ".narrow"), group_ffn.p, sr_offsets.p, sr_tiles.p, h, ch, false, 8, 0, w(f + ".narrow_bias"));
        cnn_ffn_add<<<grid(size), 256, 0, stream>>>(x.p, group_ffn.p, row_group.p, gates.p, w(p + ".aux"), rows, ch);
        record("blk-" + p, x.p, size); record("logits-" + p + ".ffn", logits.p, size_t(rows) * 8);
        record("gate-" + p + ".ffn", gates.p, rows); record_int("pick-" + p + ".ffn", pick.p, rows);
    }
    void run_cnn() {
        profile_cursor = 0;
        const char* names[] = {"full", "d0", "d1", "d2", "d3", "d4", "d5"};
        int widths[7], heights[7], rows[7]; size_t max_state = 0, max_up = 0;
        for (int i = 0; i < 7; ++i) {
            widths[i] = i ? g.levels[i-1].width : g.full_width;
            heights[i] = i ? g.levels[i-1].height : g.full_height;
            rows[i] = widths[i] * heights[i];
            max_state = std::max(max_state, size_t(rows[i]) * cnn_channels[i]);
            if (i < 6) { skip[i].resize(size_t(rows[i]) * cnn_channels[i]); max_up = std::max(max_up, size_t(rows[i]) * cnn_channels[i+1]); }
        }
        x.resize(max_state); up.resize(std::max(max_state, max_up));
        cnn_conv(features.p, x.p, widths[0], heights[0], "stem.0"); cnn_norm(x.p, x.p, rows[0], cnn_channels[0], "stem.1", true);
        for (int i = 0; i < 7; ++i) {
            if (i) {
                cnn_resize<<<grid(size_t(rows[i]) * cnn_channels[i-1]), 256, 0, stream>>>(x.p, up.p, widths[i-1], heights[i-1], widths[i], heights[i], cnn_channels[i-1]);
                cnn_conv(up.p, x.p, widths[i], heights[i], std::string("enc_proj.") + names[i] + ".0");
                cnn_norm(x.p, x.p, rows[i], cnn_channels[i], std::string("enc_proj.") + names[i] + ".1", true);
            }
            const auto& bs = blocks[0].at(names[i]);
            for (size_t j = 0; j < bs.size(); ++j) cnn_block(std::string("enc.") + names[i] + "." + std::to_string(j), widths[i], heights[i]);
            record(i == 6 ? "s-vit" : std::string("s-enc-") + names[i], x.p, size_t(rows[i]) * cnn_channels[i]);
            if (i < 6) cu(cudaMemcpyAsync(skip[i].p, x.p, size_t(rows[i]) * cnn_channels[i] * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
        for (int i = 5; i >= 0; --i) {
            int ci = cnn_channels[i+1], c = cnn_channels[i];
            cnn_resize<<<grid(size_t(rows[i]) * ci), 256, 0, stream>>>(x.p, up.p, widths[i+1], heights[i+1], widths[i], heights[i], ci);
            cnn_cat.resize(size_t(rows[i]) * (ci + c));
            cnn_concat<<<grid(size_t(rows[i]) * (ci + c)), 256, 0, stream>>>(up.p, skip[i].p, cnn_cat.p, rows[i], ci, c);
            cnn_conv(cnn_cat.p, x.p, widths[i], heights[i], std::string("merge.") + names[i] + ".0");
            cnn_norm(x.p, x.p, rows[i], c, std::string("merge.") + names[i] + ".1", true);
            const auto& bs = blocks[1].at(names[i]);
            for (size_t j = 0; j < bs.size(); ++j) cnn_block(std::string("dec.") + names[i] + "." + std::to_string(j), widths[i], heights[i]);
            record(std::string("s-dec-") + names[i], x.p, size_t(rows[i]) * c);
        }
        cnn_norm(x.p, x.p, rows[0], cnn_channels[0], "head_norm", true);
        cnn_conv(x.p, head.p, widths[0], heights[0], "head"); record("s-head", head.p, size_t(rows[0]) * 4);
        cu(cudaGetLastError());
    }
    void run() {
        if (cnn) { run_cnn(); return; }
        profile_cursor = 0;
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
                if (shared_routed) shared_transition(pooled.p, x.p, widths[i + 1], heights[i + 1], channels[i], channels[i + 1], std::string("enc_trans.") + names[i + 1], true);
                else linear(pooled.p, x.p, rows[i + 1], channels[i], channels[i + 1], std::string("enc_trans.") + names[i + 1]);
                record(std::string("trans-") + names[i + 1], x.p, static_cast<size_t>(rows[i + 1]) * channels[i + 1]);
            }
        }
        int vw = g.levels[5].width, vh = g.levels[5].height, vc = dim("vit_in.weight", shared_routed ? 2 : 0);
        pooled.resize(static_cast<size_t>(vw) * vh * channels[5]); vstate.resize(static_cast<size_t>(vw) * vh * vc);
        down(skip[5].p, pooled.p, widths[5], heights[5], vw, vh, channels[5]);
        if (shared_routed) shared_transition(pooled.p, vstate.p, vw * vh, 1, channels[5], vc, "vit_in", false);
        else linear(pooled.p, vstate.p, vw * vh, channels[5], vc, "vit_in");
        record("trans-vit", vstate.p, static_cast<size_t>(vw) * vh * vc);
        for (size_t j = 0; j < vit.size(); ++j) block(vit[j], "vit_blocks." + std::to_string(j), vstate.p, vw * vh, 1, true);
        record("s-vit", vstate.p, static_cast<size_t>(vw) * vh * vc);
        if (shared_routed) shared_transition(vstate.p, x.p, vw * vh, 1, vc, channels[5], "vit_out", false);
        else linear(vstate.p, x.p, vw * vh, vc, channels[5], "vit_out");
        int pw = vw, ph = vh;
        for (int i = 5; i >= 1; --i) {
            const float* in = x.p;
            if (i < 5) {
                if (shared_routed) shared_transition(x.p, trans.p, pw, ph, channels[i + 1], channels[i], std::string("dec_trans.") + names[i], true);
                else linear(x.p, trans.p, pw * ph, channels[i + 1], channels[i], std::string("dec_trans.") + names[i]);
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
    p.profiling = std::getenv("NR_CUDA_PROFILE") && std::string(std::getenv("NR_CUDA_PROFILE")) == "1";
    p.capture_routes_only = std::getenv("NR_CUDA_CAPTURE") && std::string(std::getenv("NR_CUDA_CAPTURE")) == "routes";
    cu(cudaSetDevice(device));
    cudaDeviceProp properties{}; cu(cudaGetDeviceProperties(&properties, device));
    p.name = properties.name; p.g = model.geom_; p.vit = model.vit_blocks_;
    p.blocks[0] = model.blocks_[0]; p.blocks[1] = model.blocks_[1];
    p.shared_routed = model.shared_routed_; p.cnn = model.cnn_;
    p.shared_experts = model.shared_experts_; p.routed_experts = model.routed_experts_;
    std::copy_n(model.cnn_channels_, 7, p.cnn_channels);
    cu(cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking));
    cu(cudaEventCreate(&p.start)); cu(cudaEventCreate(&p.stop));
    for (auto& event : p.io_events) cu(cudaEventCreate(&event));
    blas(cublasCreate(&p.handle)); blas(cublasSetStream(p.handle, p.stream));
    // Preserve f32 precision. TF32 silently changes router logits and is disabled.
    blas(cublasSetMathMode(p.handle, CUBLAS_PEDANTIC_MATH));
    blas(cublasSetAtomicsMode(p.handle, CUBLAS_ATOMICS_NOT_ALLOWED));
    sample_device_memory = std::getenv("NR_CUDA_MEMORY") && std::string(std::getenv("NR_CUDA_MEMORY")) == "1";
    if (sample_device_memory) {
        cu(cudaMemGetInfo(&device_initial_free, &device_total_bytes));
        device_min_free = device_initial_free;
    }
    for (const auto& kv : model.tensors_) {
        size_t n = 1; for (int d : kv.second.shape) n *= d;
        auto buffer = std::make_unique<Buffer<float>>(); buffer->resize(n);
        cu(cudaMemcpyAsync(buffer->p, kv.second.data, n * sizeof(float), cudaMemcpyHostToDevice, p.stream));
        p.weights.emplace(kv.first, std::move(buffer)); p.shapes.emplace(kv.first, kv.second.shape);
    }
    for (const auto& kv : model.cnn_weights_) {
        const std::string key = "cnn:" + kv.first;
        auto buffer = std::make_unique<Buffer<float>>(); buffer->resize(kv.second.size());
        cu(cudaMemcpyAsync(buffer->p, kv.second.data(), kv.second.size() * sizeof(float), cudaMemcpyHostToDevice, p.stream));
        p.weights.emplace(key, std::move(buffer));
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
    float ms = 0; cu(cudaEventElapsedTime(&ms, p.start, p.stop));
    if (p.profiling) p.finish_profile();
    return ms;
}
const std::map<std::string, std::vector<float>>& CudaEngine::debug() const { return impl_->captures; }
const std::map<std::string, double>& CudaEngine::profile() const { return impl_->profile_times; }
std::pair<size_t, size_t> CudaEngine::buffer_bytes() const { return {allocated_bytes, high_water_bytes}; }
std::vector<size_t> CudaEngine::memory_sample_bytes() const {
    if (!sample_device_memory) return {};
    sample_memory();
    return {device_total_bytes, device_total_bytes - device_initial_free,
            device_total_bytes - device_min_free, device_initial_free - device_min_free};
}

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
