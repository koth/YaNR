// shared-routed-moe-v1 device kernels. Included inside engine_cuda.cu's
// anonymous namespace. No capacity quotas and no CPU group descriptors.
__global__ void sr_router(const float* state, float* logits, const float* gamma, const float* beta,
                         const float* weight, const float* bias, int* pick, float* gates, int* counts,
                         int width, int height, int channels, bool windows) {
    __shared__ float pooled[1024];
    __shared__ float normalized[1024];
    __shared__ float scores[16];
    __shared__ float mean, inv;
    int tid = threadIdx.x, item = blockIdx.x, nw = (width + 7) / 8;
    for (int c = tid; c < channels; c += blockDim.x) {
        float sum = 0; int n = 0;
        if (windows) {
            for (int y = item / nw * 8; y < min(height, item / nw * 8 + 8); ++y)
                for (int x = item % nw * 8; x < min(width, item % nw * 8 + 8); ++x) {
                    sum += state[(static_cast<size_t>(y) * width + x) * channels + c]; ++n;
                }
            sum *= 1.f / n;
        } else sum = state[static_cast<size_t>(item) * channels + c];
        pooled[c] = sum;
    }
    __syncthreads();
    if (!tid) {
        float m = 0, var = 0;
        for (int c = 0; c < channels; ++c) m += pooled[c];
        m /= channels;
        for (int c = 0; c < channels; ++c) { float v = pooled[c] - m; var += v * v; }
        mean = m; inv = 1.f / sqrtf(var / channels + 1e-5f);
    }
    __syncthreads();
    for (int c = tid; c < channels; c += blockDim.x)
        normalized[c] = (pooled[c] - mean) * inv * gamma[c] + beta[c];
    __syncthreads();
    int e = tid / 32, lane = tid % 32;
    float sum = 0;
    for (int c = lane; c < channels; c += 32)
        sum = __fmaf_rn(normalized[c], weight[e * channels + c], sum);
    for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffff, sum, d);
    if (!lane) scores[e] = logits[item * 16 + e] = sum;
    __syncthreads();
    if (!tid) {
        int best = 0; float mx = scores[0], den = 0;
        for (int j = 1; j < 16; ++j) {
            if (scores[j] + bias[j] > scores[best] + bias[best]) best = j;
            mx = fmaxf(mx, scores[j]);
        }
        for (int j = 0; j < 16; ++j) den += expf(scores[j] - mx);
        pick[item] = best; gates[item] = 16 * expf(scores[best] - mx) / den;
        int rows = windows ? min(8, width - item % nw * 8) * min(8, height - item / nw * 8) : 1;
        atomicAdd(counts + best, rows);
    }
}

// 16 CTAs scan routing units independently, preserving unit/pixel order.
// Counts are actual field rows. Avoid radix sort/scan launch chains on small
// deep grids, and never stage a largest-group-sized capacity per expert.
__global__ void sr_compact_map(const int* pick, const int* counts, int* row_group,
                              int* offsets, int* tiles, int items, int width, int height, bool windows) {
    using Scan = cub::BlockScan<int, 256>;
    __shared__ typename Scan::TempStorage storage;
    __shared__ int running, base;
    int e = blockIdx.x, tid = threadIdx.x, nw = (width + 7) / 8;
    if (!tid) {
        base = 0; running = 0;
        for (int j = 0; j < e; ++j) base += counts[j];
        offsets[e] = base;
        int t = 0;
        for (int j = 0; j < e; ++j) t += (counts[j] + 31) / 32;
        tiles[e] = t;
        if (e == 15) { offsets[16] = base + counts[e]; tiles[16] = t + (counts[e] + 31) / 32; }
    }
    __syncthreads();
    for (int i = 0; i < items; i += 256) {
        int item = i + tid;
        int cw = windows ? min(8, width - item % nw * 8) : 1;
        int ch = windows ? min(8, height - item / nw * 8) : 1;
        int n = item < items && pick[item] == e ? cw * ch : 0;
        int prefix, aggregate;
        Scan(storage).ExclusiveSum(n, prefix, aggregate);
        int dst = base + running + prefix;
        if (n) {
            int wx = windows ? item % nw * 8 : item, wy = windows ? item / nw * 8 : 0;
            for (int y = 0; y < ch; ++y)
                for (int x = 0; x < cw; ++x)
                    row_group[(wy + y) * width + wx + x] = dst + y * cw + x;
        }
        __syncthreads();
        if (!tid) running += aggregate;
        __syncthreads();
    }
}
__global__ void sr_pick(const float* logits, const float* bias, int* pick,
                       float* gates, int* keys, int* ids, int* counts,
                       int items, int width, int height, bool windows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= items) return;
    int best = 0;
    float mx = logits[i * 16];
    for (int e = 1; e < 16; ++e) {
        if (logits[i * 16 + e] + bias[e] > logits[i * 16 + best] + bias[best]) best = e;
        mx = fmaxf(mx, logits[i * 16 + e]);
    }
    float den = 0;
    for (int e = 0; e < 16; ++e) den += expf(logits[i * 16 + e] - mx);
    pick[i] = best; keys[i] = best; ids[i] = i;
    gates[i] = 16 * expf(logits[i * 16 + best] - mx) / den;
    int nw = (width + 7) / 8;
    int rows = windows ? min(8, width - i % nw * 8) * min(8, height - i / nw * 8) : 1;
    atomicAdd(counts + best, rows);
}

__global__ void sr_lengths(const int* ids, int* lengths, int items, int width, int height, bool windows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= items) return;
    int id = ids[i], nw = (width + 7) / 8;
    lengths[i] = windows ? min(8, width - id % nw * 8) * min(8, height - id / nw * 8) : 1;
}

__global__ void sr_descriptors(const int* counts, int* offsets, int* tiles) {
    if (threadIdx.x) return;
    offsets[0] = 0; tiles[0] = 0;
    for (int e = 0; e < 16; ++e) {
        offsets[e + 1] = offsets[e] + counts[e];
        tiles[e + 1] = tiles[e] + (counts[e] + 31) / 32;
    }
}

// Stable sorted routing units, then row-major valid pixels within each unit.
// Partial windows contribute only actual field rows; there are no hole rows.
__global__ void sr_pack(const float* state, float* grouped, int* row_group,
                       const int* ids, const int* offsets, int width, int height,
                       int channels, bool windows) {
    int item = ids[blockIdx.x], base = offsets[blockIdx.x];
    int nw = (width + 7) / 8;
    int wx = windows ? item % nw * 8 : item, wy = windows ? item / nw * 8 : 0;
    int cw = windows ? min(8, width - wx) : 1;
    int ch = windows ? min(8, height - wy) : 1;
    for (int j = threadIdx.x; j < cw * ch * channels; j += blockDim.x) {
        int slot = j / channels, c = j % channels;
        int row = (wy + slot / cw) * width + wx + slot % cw;
        grouped[static_cast<size_t>(base + slot) * channels + c] = state[static_cast<size_t>(row) * channels + c];
        if (!c) row_group[row] = base + slot;
    }
}

// Fixed upper-bound launch, dynamic device descriptors. At most 15 unused row
// tiles globally, independent of load imbalance. Each valid row is evaluated;
// tile masking adds arithmetic only, never padding rows in packed storage.
template<int BN> __global__ void sr_grouped_gemm(const float* in, const float* bank, float* out,
                              const int* offsets, const int* tiles,
                              int K, int N, bool act) {
    int task = blockIdx.x, e = 0;
    while (e < 16 && task >= tiles[e + 1]) ++e;
    if (e == 16) return;
    int start = offsets[e], count = offsets[e + 1] - start;
    int tile_m = (task - tiles[e]) * 32, tile_n = blockIdx.y * BN;
    int tid = threadIdx.x, r = tid / (BN / 4), c = (tid % (BN / 4)) * 4;
    constexpr int RPT = BN / 32, STRIDE = 256 / (BN / 4);
    __shared__ float a[32][16];
    __shared__ float b[16][BN];
    float acc[RPT][4] = {};
    const float* weight = bank + static_cast<size_t>(e + 1) * K * N;
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int j = tid; j < 32 * 16; j += 256) {
            int mr = j / 16, kc = j % 16;
            a[mr][kc] = tile_m + mr < count ? in[static_cast<size_t>(start + tile_m + mr) * K + k0 + kc] : 0;
        }
        for (int j = tid; j < 16 * BN; j += 256) {
            int kr = j / BN, nc = j % BN;
            b[kr][nc] = tile_n + nc < N ? weight[static_cast<size_t>(k0 + kr) * N + tile_n + nc] : 0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            #pragma unroll
            for (int i = 0; i < RPT; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[i][j] = __fmaf_rn(a[r + i * STRIDE][k], b[k][c + j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < RPT; ++i) {
        int mr = tile_m + r + STRIDE * i;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int nc = tile_n + c + j;
            if (mr < count && nc < N) {
                float value = acc[i][j];
                if (act) value = value / (1 + expf(-value));
                out[static_cast<size_t>(start + mr) * N + nc] = value;
            }
        }
    }
}

__global__ void sr_mix(const float* shared, const float* routed, float* out,
                      const int* row_group, const int* pick, const float* gates,
                      const float* bias, const float* original, const float* aux,
                      int width, int height, int channels, bool windows, bool attention) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(width) * height * channels) return;
    int row = i / channels, c = i % channels;
    int item = windows ? row / width / 8 * ((width + 7) / 8) + row % width / 8 : row;
    float s = shared[i] + (bias ? bias[c] : 0);
    float r = routed[static_cast<size_t>(row_group[row]) * channels + c] +
              (bias ? bias[(pick[item] + 1) * channels + c] : 0);
    float value = (s + gates[item] * r) * 0.5f;
    if (original) value = attention ? original[i] + aux[c] * value : original[i] * aux[c] + value;
    out[i] = value;
}

__global__ void sr_int_float(const int* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = static_cast<float>(input[i]);
}

// Two FP32 matrix products with the independent SiLU kept between them.
// Specialization for C=H=32 avoids writing the hidden matrix to global memory.
// Shared branch uses all field rows directly; routed branch uses compact rows.
template<bool ROUTED> __global__ void sr_ffn32(const float* in, const float* expand,
                                            const float* contract, float* out,
                                            const int* offsets, const int* tiles, int rows) {
    int task = blockIdx.x, e = 0, start = 0, count = rows, tile = task * 32;
    if (ROUTED) {
        while (e < 16 && task >= tiles[e + 1]) ++e;
        if (e == 16) return;
        start = offsets[e]; count = offsets[e + 1] - start;
        tile = (task - tiles[e]) * 32;
        ++e;
    }
    __shared__ float a[32][32], b[32][32];
    int tid = threadIdx.x, row = tid / 8, col = tid % 8 * 4;
    for (int j = tid; j < 1024; j += 256) {
        a[j / 32][j % 32] = tile + j / 32 < count ? in[(static_cast<size_t>(start + tile + j / 32) * 32) + j % 32] : 0;
        b[j / 32][j % 32] = expand[e * 1024 + j];
    }
    __syncthreads();
    float h[4] = {};
    #pragma unroll
    for (int k = 0; k < 32; ++k)
        #pragma unroll
        for (int j = 0; j < 4; ++j) h[j] = __fmaf_rn(a[row][k], b[k][col + j], h[j]);
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 4; ++j) a[row][col + j] = h[j] / (1 + expf(-h[j]));
    for (int j = tid; j < 1024; j += 256) b[j / 32][j % 32] = contract[e * 1024 + j];
    __syncthreads();
    float result[4] = {};
    #pragma unroll
    for (int k = 0; k < 32; ++k)
        #pragma unroll
        for (int j = 0; j < 4; ++j) result[j] = __fmaf_rn(a[row][k], b[k][col + j], result[j]);
    if (tile + row < count)
        #pragma unroll
        for (int j = 0; j < 4; ++j) out[static_cast<size_t>(start + tile + row) * 32 + col + j] = result[j];
}
