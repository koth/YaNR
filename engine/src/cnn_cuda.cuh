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
