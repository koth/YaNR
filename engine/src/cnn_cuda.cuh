// NHWC kernels for nr_cnn_unet.py. Convolution banks use [ky,kx,in,out].
__global__ void cnn_im2col(const float* in, float* out, int start, int count,
                          int width, int height, int channels, int kernel) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int K = kernel * kernel * channels;
    if (i >= size_t(count) * K) return;
    int row = start + int(i / K), k = int(i % K), c = k % channels;
    int sx = row % width + k / channels % kernel - kernel / 2;
    int sy = row / width + k / channels / kernel - kernel / 2;
    out[i] = sx < 0 || sy < 0 || sx >= width || sy >= height ? 0.f : in[(size_t(sy) * width + sx) * channels + c];
}

// Vectorize copies without changing GEMM shape or its accumulation order.
template<int CHANNELS>
__global__ void cnn_im2col3_vec(const float4* in, float4* out, int start, int count,
                               int width, int height) {
    constexpr int C = CHANNELS / 4, K = C * 9;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(count) * K) return;
    int row = start + int(i / K), k = int(i % K);
    int x = row % width + k / C % 3 - 1, y = row / width + k / C / 3 - 1;
    out[i] = x < 0 || y < 0 || x >= width || y >= height
        ? make_float4(0, 0, 0, 0) : in[(size_t(y) * width + x) * C + k % C];
}

// Statistics span spatial positions AND the channels of each group. All
// reductions finish before writes, so input/output may alias.
__global__ void cnn_group_norm(const float* in, float* out, int rows, int channels,
                               const float* gamma, const float* beta, bool act) {
    __shared__ double sums[256];
    __shared__ float mean, inv;
    int cg = channels / 8, group = blockIdx.x, tid = threadIdx.x;
    size_t count = size_t(rows) * cg;
    double sum = 0;
    for (size_t i = tid; i < count; i += blockDim.x)
        sum += in[i / cg * channels + group * cg + i % cg];
    sums[tid] = sum; __syncthreads();
    for (int step = 128; step; step /= 2) { if (tid < step) sums[tid] += sums[tid + step]; __syncthreads(); }
    if (!tid) mean = float(sums[0] / count);
    __syncthreads(); sum = 0;
    for (size_t i = tid; i < count; i += blockDim.x) {
        double d = double(in[i / cg * channels + group * cg + i % cg]) - mean; sum += d * d;
    }
    sums[tid] = sum; __syncthreads();
    for (int step = 128; step; step /= 2) { if (tid < step) sums[tid] += sums[tid + step]; __syncthreads(); }
    if (!tid) inv = 1.f / sqrtf(float(sums[0] / count) + 1e-5f);
    __syncthreads();
    for (size_t i = tid; i < count; i += blockDim.x) {
        int c = group * cg + i % cg; size_t pos = i / cg * channels + c;
        float value = (in[pos] - mean) * inv * gamma[c] + beta[c];
        out[pos] = act ? value / (1.f + expf(-value)) : value;
    }
}

// Large fields need many CTAs per group. Keep the two-pass double statistics
// and the FP32 mean/variance rounding of cnn_group_norm; E[x*x]-E[x]^2 would
// lose accuracy on nearly constant inputs. Separate apply also permits in==out.
__device__ double cnn_reduce_sum(double value) {
    __shared__ double warps[8];
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    for (int step = 16; step; step /= 2)
        value += __shfl_down_sync(0xffffffff, value, step);
    if (!lane) warps[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warps[lane] : 0;
    if (!warp)
        for (int step = 16; step; step /= 2)
            value += __shfl_down_sync(0xffffffff, value, step);
    return value;
}

template<bool VARIANCE>
__global__ void cnn_norm_partial(const float* in, double* partial, const float* stats,
                                 int rows, int channels, int parts) {
    int group = blockIdx.y, cg = channels / 8;
    size_t count = size_t(rows) * cg;
    size_t begin = size_t(blockIdx.x) * 4096, end = min(begin + 4096, count);
    double mean = VARIANCE ? stats[group] : 0, sum = 0;
    for (size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        double value = in[i / cg * channels + group * cg + i % cg];
        if (VARIANCE) { value -= mean; value *= value; }
        sum += value;
    }
    sum = cnn_reduce_sum(sum);
    if (!threadIdx.x) partial[group * parts + blockIdx.x] = sum;
}

// For power-of-two channel groups, one CTA reads every channel of its spatial
// tile contiguously. Avoid eight strided passes through the NHWC cache lines.
template<int CHANNELS, bool VARIANCE>
__global__ void cnn_norm_partial_coalesced(const float* in, double* partial, const float* stats,
                                          int rows, int parts) {
    constexpr int CG = CHANNELS / 8, TILE_ROWS = 512 / CG;
    __shared__ double sums[256 / CG];
    int tid = threadIdx.x, c = tid % CHANNELS, group = c / CG;
    size_t begin = size_t(blockIdx.x) * TILE_ROWS * CHANNELS;
    size_t end = min(begin + size_t(TILE_ROWS) * CHANNELS, size_t(rows) * CHANNELS);
    double sum = 0, mean = VARIANCE ? stats[group] : 0;
    for (size_t i = begin + tid; i < end; i += 256) {
        double value = in[i];
        if (VARIANCE) { value -= mean; value *= value; }
        sum += value;
    }
    for (int step = CG / 2; step; step /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, step, CG);
    if (tid % CG == 0) sums[tid / CG] = sum;
    __syncthreads();
    if (tid < 8) {
        sum = 0;
        #pragma unroll
        for (int j = tid; j < 256 / CG; j += 8) sum += sums[j];
        partial[tid * parts + blockIdx.x] = sum;
    }
}

template<bool VARIANCE>
__global__ void cnn_norm_stats(const double* partial, float* stats, int parts, size_t count) {
    double sum = 0;
    for (int i = threadIdx.x; i < parts; i += blockDim.x)
        sum += partial[blockIdx.x * parts + i];
    sum = cnn_reduce_sum(sum);
    if (!threadIdx.x) {
        float value = float(sum / count);
        stats[blockIdx.x + (VARIANCE ? 8 : 0)] = VARIANCE ? 1.f / sqrtf(value + 1e-5f) : value;
    }
}

__global__ void cnn_norm_apply(const float* in, float* out, const float* stats,
                               const float* gamma, const float* beta,
                               int rows, int channels, bool act) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * channels) return;
    int c = i % channels, group = c / (channels / 8);
    float value = (in[i] - stats[group]) * stats[group + 8] * gamma[c] + beta[c];
    out[i] = act ? value / (1.f + expf(-value)) : value;
}

__global__ void cnn_resize(const float* in, float* out, int iw, int ih, int ow, int oh, int channels) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(ow) * oh * channels) return;
    int row = i / channels, c = i % channels;
    float fx = fmaxf(0.f, (row % ow + .5f) * (float(iw) / ow) - .5f);
    float fy = fmaxf(0.f, (row / ow + .5f) * (float(ih) / oh) - .5f);
    int x0 = int(fx), y0 = int(fy), x1 = min(x0 + 1, iw - 1), y1 = min(y0 + 1, ih - 1);
    float tx = fx - x0, ty = fy - y0;
    out[i] = (1 - ty) * ((1 - tx) * in[(size_t(y0) * iw + x0) * channels + c] + tx * in[(size_t(y0) * iw + x1) * channels + c]) +
             ty * ((1 - tx) * in[(size_t(y1) * iw + x0) * channels + c] + tx * in[(size_t(y1) * iw + x1) * channels + c]);
}
__global__ void cnn_concat(const float* high, const float* skip, float* out, int rows, int hi, int lo) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * (hi + lo)) return;
    int row = i / (hi + lo), c = i % (hi + lo);
    out[i] = c < hi ? high[size_t(row) * hi + c] : skip[size_t(row) * lo + c - hi];
}
__global__ void cnn_add(float* state, const float* value, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) state[i] += value[i];
}
__global__ void cnn_ffn_add(float* state, const float* grouped, const int* row_group,
                           const float* gates, const float* aux, int rows, int channels) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * channels) return;
    int row = i / channels, c = i % channels;
    state[i] += grouped[size_t(row_group[row]) * channels + c] * gates[row] * aux[c];
}

// Pixel routing units have length one: sorted positions are already offsets.
// Pack multiple pixels per CTA instead of launching a mostly idle CTA per pixel.
__global__ void cnn_pack_pixels(const float* in, float* grouped, int* row_group,
                                const int* ids, int rows, int channels) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * channels) return;
    int dst = i / channels, c = i % channels, src = ids[dst];
    grouped[i] = in[size_t(src) * channels + c];
    if (!c) row_group[src] = dst;
}

__global__ void cnn_inverse_map(const int* row_group, int* ids, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) ids[row_group[row]] = row;
}

__global__ void cnn_pick_pixels(const float* logits, int* pick, float* gates,
                                int* keys, int* ids, int* counts, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float values[8], exps[8], mx = logits[row * 8], den = 0;
    #pragma unroll
    for (int e = 0; e < 8; ++e) { values[e] = logits[row * 8 + e]; mx = fmaxf(mx, values[e]); }
    #pragma unroll
    for (int e = 0; e < 8; ++e) { exps[e] = expf(values[e] - mx); den += exps[e]; }
    int best = 0; float maximum = exps[0] / den;
    #pragma unroll
    for (int e = 1; e < 8; ++e) {
        float probability = exps[e] / den;
        if (probability > maximum) { best = e; maximum = probability; }
    }
    pick[row] = best; keys[row] = best; ids[row] = row; gates[row] = maximum;
    unsigned peers = __match_any_sync(__activemask(), best);
    if ((threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(counts + best, __popc(peers));
}

// Full/d0 FFNs are 32 -> 64 -> 32. Keep the hidden tile on chip while
// retaining the same ordered FP32 FMAs, bias and independent SiLU as two GEMMs.
__global__ void cnn_ffn32_64(const float* in, const float* expand, const float* narrow,
                            const float* b1, const float* b2, float* out,
                            const int* offsets, const int* tiles,
                            const int* original_rows = nullptr,
                            const float* gates = nullptr, const float* aux = nullptr) {
    int task = blockIdx.x, expert = 0;
    while (expert < 8 && task >= tiles[expert + 1]) ++expert;
    if (expert == 8) return;
    int start = offsets[expert], count = offsets[expert + 1] - start;
    int base = (task - tiles[expert]) * 32, tid = threadIdx.x;
    __shared__ float a[32][65], weight[2048];
    for (int j = tid; j < 1024; j += 256)
        a[j / 32][j % 32] = base + j / 32 < count ? in[size_t(start + base + j / 32) * 32 + j % 32] : 0.f;
    for (int j = tid; j < 2048; j += 256) weight[j] = expand[expert * 2048 + j];
    __syncthreads();
    int r = tid / 16, c = (tid % 16) * 4;
    float hidden[2][4] = {};
    #pragma unroll
    for (int k = 0; k < 32; ++k)
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                hidden[i][j] = __fmaf_rn(a[r + i * 16][k], weight[k * 64 + c + j], hidden[i][j]);
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float value = hidden[i][j] + b1[expert * 64 + c + j];
            a[r + i * 16][c + j] = value / (1.f + expf(-value));
        }
    for (int j = tid; j < 2048; j += 256) weight[j] = narrow[expert * 2048 + j];
    __syncthreads();
    r = tid / 8; c = (tid % 8) * 4;
    float result[4] = {};
    #pragma unroll
    for (int k = 0; k < 64; ++k)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            result[j] = __fmaf_rn(a[r][k], weight[k * 32 + c + j], result[j]);
    if (base + r < count) {
        int row = start + base + r, dst = original_rows ? original_rows[row] : row;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float value = result[j] + b2[expert * 32 + c + j];
            size_t pos = size_t(dst) * 32 + c + j;
            if (original_rows) out[pos] += value * gates[dst] * aux[c + j];
            else out[pos] = value;
        }
    }
}
