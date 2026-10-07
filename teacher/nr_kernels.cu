// The native kernels: the fixed-point GEMM (the F13 chain plus the epilogue), the f16 GEMM (the F24
// chain with the exact integer -> half publication), the window attention (fused, from raw qkv), and the
// block FFN chain (expand -> contract -> qkv in one launch, intermediates in shared memory). Standalone
// CUDA C with a C ABI - it touches no torch header and links cudart statically, so it is built once with
// whatever nvcc is present and driven through ctypes:
//
//   nvcc -O3 -std=c++14 -arch=sm_86 -cudart=static -shared -Xcompiler -fPIC nr_kernels.cu -o nr_kernels.so
//
// The arithmetic is the reference contract (docs/numerics.md, src/reference.cpp) exactly as the oracle and
// the Triton kernels implement it: products exact in f32, powers of two built as bit patterns, terms
// truncated toward zero, exact integer sums, one rounding to half per 16-product group. Everything is
// element-strided so the callers can pass the same permuted views the rest of the port uses.

#include <cuda_fp16.h>
#include <math.h>

// ----------------------------------------------------------------------------------------------------
// The publication grid.
// ----------------------------------------------------------------------------------------------------

__device__ __forceinline__ int f32_exp(float v) {
  return (int)((__float_as_uint(v) >> 23) & 0xff) - 127;
}

__device__ __forceinline__ float pow2(int exponent) {   // an exact power of two, as an f32 bit pattern
  return __int_as_float((exponent + 127) << 23);
}

__device__ __forceinline__ unsigned short f16_bits(float v) {
  return __half_as_ushort(__float2half_rn(v));
}

__device__ __forceinline__ float f16_value(unsigned short bits) {
  return __half2float(__ushort_as_half(bits));
}

// One publication-grid add: the exact f32 sum of two halves, rounded to half (addF16).
__device__ __forceinline__ float hadd(float a, float b) {
  return f16_value(f16_bits(a + b));
}

// One 16-product group of the FP8 chain (adaFp8Fdpa16): shared exponent, 13 fractional bits,
// truncation toward zero, exact integer sum, one rounding to half; the group result chains on.
__device__ __forceinline__ float fp8_group(float acc, const float* a, const float* b) {
  if (!isfinite(acc)) return acc;
  int maxexp = (acc != 0.0f) ? max(f32_exp(acc), -14) : -21;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float av = a[i], bv = b[i];
    if (av != 0.0f && bv != 0.0f) {
      int e = max(f32_exp(av), -6) + max(f32_exp(bv), -6);
      maxexp = max(maxexp, e);
    }
  }
  maxexp = max(-21, min(16, maxexp));
  float align = pow2(13 - maxexp);
  int units = (int)truncf(acc * align);
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    units += (int)truncf(a[i] * b[i] * align);
  }
  float stepped = (float)units * pow2(maxexp - 13);
  return f16_value(f16_bits(stepped));
}

// The same group with the operand exponents passed in (fp8_exp): a row/column of operands feeds two
// outputs, so extracting each operand's exponent once per element instead of once per product halves
// the scan's work. A zero operand carries the sentinel -42, whose sums can never win the max (the floor
// is -21 and every real clamped sum is at least -12), exactly as the zero test does.
__device__ __forceinline__ int fp8_exp(float v) {
  return v != 0.0f ? max(f32_exp(v), -6) : -42;
}

__device__ __forceinline__ float fp8_group_ex(float acc, const float* a, const float* b,
                                              const int* ea, const int* eb) {
  if (!isfinite(acc)) return acc;
  int maxexp = (acc != 0.0f) ? max(f32_exp(acc), -14) : -21;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    maxexp = max(maxexp, ea[i] + eb[i]);
  }
  maxexp = max(-21, min(16, maxexp));
  float align = pow2(13 - maxexp);
  int units = (int)truncf(acc * align);
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    units += (int)truncf(a[i] * b[i] * align);
  }
  float stepped = (float)units * pow2(maxexp - 13);
  return f16_value(f16_bits(stepped));
}

// Four groups at once (a 2x2 thread tile): the operand exponents feed two outputs each, so one scan
// pass over the operands covers all four chains - four extractions per product pair instead of eight,
// with no extra live state. Each output keeps its own accumulator chain and rounding.
__device__ __forceinline__ void fp8_group_quad(float* accp,
                                               const float* a0, const float* a1,
                                               const float* b0, const float* b1) {
  float (&acc)[2][2] = *(float (*)[2][2])accp;
  int m[2][2];
  #pragma unroll
  for (int di = 0; di < 2; ++di)
    #pragma unroll
    for (int dj = 0; dj < 2; ++dj)
      m[di][dj] = (acc[di][dj] != 0.0f) ? max(f32_exp(acc[di][dj]), -14) : -21;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    int ea0 = fp8_exp(a0[i]), ea1 = fp8_exp(a1[i]);
    int eb0 = fp8_exp(b0[i]), eb1 = fp8_exp(b1[i]);
    m[0][0] = max(m[0][0], ea0 + eb0);
    m[0][1] = max(m[0][1], ea0 + eb1);
    m[1][0] = max(m[1][0], ea1 + eb0);
    m[1][1] = max(m[1][1], ea1 + eb1);
  }
  const float* arow[2] = {a0, a1};
  const float* brow[2] = {b0, b1};
  #pragma unroll
  for (int di = 0; di < 2; ++di) {
    #pragma unroll
    for (int dj = 0; dj < 2; ++dj) {
      if (!isfinite(acc[di][dj])) continue;
      int maxexp = max(-21, min(16, m[di][dj]));
      float align = pow2(13 - maxexp);
      int units = (int)truncf(acc[di][dj] * align);
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        units += (int)truncf(arow[di][i] * brow[dj][i] * align);
      }
      float stepped = (float)units * pow2(maxexp - 13);
      acc[di][dj] = f16_value(f16_bits(stepped));
    }
  }
}

// roundShiftRightEven: shift right, round to nearest, ties away from even.
__device__ __forceinline__ unsigned int rshift_even(unsigned int value, int shift) {
  if (shift <= 0) return value;
  if (shift > 31) return 0;
  unsigned int quotient = value >> shift;
  unsigned int remainder = value & ((1u << shift) - 1);
  unsigned int halfway = 1u << (shift - 1);
  return quotient + ((remainder > halfway || (remainder == halfway && (quotient & 1))) ? 1 : 0);
}

// fixedToF16: exact signed integer times 2^binaryExponent -> half, as its bit pattern.
__device__ __forceinline__ unsigned short fixed_to_f16(int fixed_sum, int binary_exponent) {
  if (fixed_sum == 0) return 0;
  bool negative = fixed_sum < 0;
  unsigned int magnitude = (unsigned int)(negative ? -fixed_sum : fixed_sum);
  int msb = 31 - __clz(magnitude);
  int value_exponent = msb + binary_exponent;
  unsigned int half_bits = negative ? 0x8000u : 0u;
  if (value_exponent >= -14) {
    unsigned int significand = msb > 10 ? rshift_even(magnitude, msb - 10)
                                        : (magnitude << (10 - msb));
    if (significand >= 2048) { significand = 1024; value_exponent += 1; }
    if (value_exponent >= 16) half_bits |= 0x7c00u;
    else half_bits |= ((unsigned int)(value_exponent + 15) << 10) | (significand - 1024);
  } else {
    int subnormal_scale = binary_exponent + 24;
    unsigned int mantissa = subnormal_scale >= 0 ? (magnitude << subnormal_scale)
                                                : rshift_even(magnitude, -subnormal_scale);
    half_bits |= min(mantissa, 1024u);
  }
  return (unsigned short)half_bits;
}

// One 8-product group of the f16 chain (adaF16Fdpa8): 24 fractional bits, exact integer -> half.
__device__ __forceinline__ float f16_group(float acc, const float* a, const float* b) {
  if (!isfinite(acc)) return acc;
  int maxexp = (acc != 0.0f) ? max(f32_exp(acc), -14) : -21;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float av = a[i], bv = b[i];
    if (av != 0.0f && bv != 0.0f) {
      int e = max(f32_exp(av), -14) + max(f32_exp(bv), -14);
      maxexp = max(maxexp, e);
    }
  }
  float align = pow2(24 - maxexp);
  int units = (int)truncf(acc * align);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    units += (int)truncf(a[i] * b[i] * align);
  }
  return f16_value(fixed_to_f16(units, maxexp - 24));
}

// MpCubicSiLU: five half publications, the inner products exact in f32.
__device__ __forceinline__ float mp_cubic_silu(float value) {
  float bounded = f16_value(f16_bits(fminf(fmaxf(value, -4.0f), 4.0f)));
  float absolute = f16_value(f16_bits(fabsf(bounded)));
  float inner = f16_value(f16_bits(__fmaf_rn(absolute, -0.055908203125f, 0.447265625f)));
  float polynomial = f16_value(f16_bits(__fmaf_rn(bounded, inner, 0.89453125f)));
  return f16_value(f16_bits(__fmaf_rn(value, polynomial, 0.0f)));
}

// The A operand's within-32 index rotation (packedInputIndex), applied at load.
__device__ __forceinline__ int swizzle_index(int k) {
  int base = k & ~31;
  int within = k & 31;
  return base + (within & 16) + ((within & 15) >> 2) * 2 + (within & 1) + ((within & 2) ? 8 : 0);
}

// The same rotation as a compile-time constant of the within-32 index (the K loop walks in steps of 32,
// so the rotation of kb + j is just kb plus this).
__host__ __device__ constexpr int swz_const(int within) {
  return (within & 16) + ((within & 15) >> 2) * 2 + (within & 1) + ((within & 2) ? 8 : 0);
}

// Physical token -> natural token inside the 8x8 window (inverseTiledToken).
__device__ __forceinline__ int inverse_tiled(int token) {
  int tile = token >> 4, within = token & 15;
  return (((tile >> 1) * 4 + (within >> 2)) * 8) + (tile & 1) * 4 + (within & 3);
}

// The cosine normalization of one 32-channel head: pair squares, the half tree, 1/sqrt in f32, one
// rounding to half (Network._cosine_norm).
__device__ __forceinline__ float cosine_norm32(const float* x) {
  float r[16];
  #pragma unroll
  for (int j = 0; j < 16; ++j) {
    float lo = x[j], hi = x[16 + j];
    float hs = f16_value(f16_bits(hi * hi));
    r[j] = f16_value(f16_bits(lo * lo + hs));
  }
  float t0 = hadd(r[0], r[8]),  t1 = hadd(r[1], r[9]);
  float t2 = hadd(r[2], r[10]), t3 = hadd(r[3], r[11]);
  float t4 = hadd(r[4], r[12]), t5 = hadd(r[5], r[13]);
  float t6 = hadd(r[6], r[14]), t7 = hadd(r[7], r[15]);
  float u0 = hadd(t0, t4), u1 = hadd(t1, t5), u2 = hadd(t2, t6), u3 = hadd(t3, t7);
  float v0 = hadd(u0, u2), v1 = hadd(u1, u3);
  float total = hadd(v0, v1);
  return f16_value(f16_bits(1.0f / sqrtf(total)));
}

// ----------------------------------------------------------------------------------------------------
// The FP8 GEMM: the chain and the epilogue in one launch. flags: 1 seed, 2 residual, 4 swizzle A,
// 8 silu, 16 write raw, 32 write e4. partition > 0 splits K into independent chunk chains combined
// with half adds (the ViT's adaFp8Fdpa partitions); the seed feeds chunk 0 only.
// ----------------------------------------------------------------------------------------------------

template <int TR, int TC, int TX, int TY>
__global__ void __launch_bounds__((TR / TX) * (TC / TY), 3)
fp8_chain_kernel_v2(
    const __half* __restrict__ X, const __half* __restrict__ W,
    const __half* __restrict__ SEED, const __half* __restrict__ RES, const __half* __restrict__ AUX,
    __half* __restrict__ OUT_RAW, __half* __restrict__ OUT_E4, const __half* __restrict__ TABLE,
    long long R, long long K, long long N,
    long long sxb, long long sxr, long long sxk,
    long long swb, long long swk, long long swn,
    long long sdb, long long sdr, long long sdn,
    long long srb, long long srr, long long srn,
    long long srawb, long long srawr, long long srawn,
    long long se4b, long long se4r, long long se4n,
    int flags, long long partition, int kstep) {
  constexpr int THREADS = (TR / TX) * (TC / TY);
  constexpr int TCOLS = TC / TY;
  __shared__ float As[TR][65];        // [tile row][k within the 64 step], padded against bank conflicts
  __shared__ float BsT[TC][65];       // [tile col][k], transposed so each thread's products stay contiguous

  long long tiles_n = (N + TC - 1) / TC;
  long long tiles_r = (R + TR - 1) / TR;
  long long pid = blockIdx.x;
  int ntile = (int)(pid % tiles_n); pid /= tiles_n;
  int rtile = (int)(pid % tiles_r); pid /= tiles_r;
  long long batch = pid;

  int t = threadIdx.x;
  int ti = t / TCOLS, tj = t % TCOLS;
  long long row0 = (long long)rtile * TR + ti * TX;
  long long col0 = (long long)ntile * TC + tj * TY;

  float total[TX][TY];
  #pragma unroll
  for (int di = 0; di < TX; ++di)
    #pragma unroll
    for (int dj = 0; dj < TY; ++dj) total[di][dj] = 0.0f;

  long long step = partition > 0 ? partition : K;
  bool first = true;
  for (long long lo = 0; lo < K; lo += step) {
    long long k_end = lo + step;
    float acc[TX][TY];
    #pragma unroll
    for (int di = 0; di < TX; ++di) {
      #pragma unroll
      for (int dj = 0; dj < TY; ++dj) {
        float v = 0.0f;
        long long row = row0 + di, col = col0 + dj;
        if (row < R && col < N && lo == 0) {
          if (flags & 1) v = f16_value(__half_as_ushort(SEED[batch * sdb + row * sdr + col * sdn]));
          if (flags & 2) {
            float res = f16_value(__half_as_ushort(RES[batch * srb + row * srr + col * srn]));
            float aux = f16_value(__half_as_ushort(AUX[col]));
            v = f16_value(f16_bits(res * aux));
          }
        }
        acc[di][dj] = v;
      }
    }

    for (long long kb = lo; kb < k_end; kb += kstep) {
      for (int i = t; i < TR * 64; i += THREADS) {
        int row = i >> 6, k = i & 63;
        long long grow = (long long)rtile * TR + row;
        long long ka = kb + k;
        if (flags & 4) ka = kb + swz_const(k & 31) + (k & 32);
        As[row][k] = (grow < R && k < kstep && kb + k < K)
            ? __half2float(X[batch * sxb + grow * sxr + ka * sxk]) : 0.0f;
      }
      for (int i = t; i < TC * 64; i += THREADS) {
        int col = i >> 6, k = i & 63;
        long long gcol = (long long)ntile * TC + col;
        BsT[col][k] = (gcol < N && k < kstep && kb + k < K)
            ? __half2float(W[batch * swb + (kb + k) * swk + gcol * swn]) : 0.0f;
      }
      __syncthreads();

      // Always four 16-product groups per step: out-of-range operands staged as zero contribute exactly
      // zero and never move the shared exponent, so a padded tail group is the same as none at all.
      #pragma unroll
      for (int g = 0; g < 4; ++g) {
        float a[TX][16], b[TY][16];
        #pragma unroll
        for (int di = 0; di < TX; ++di)
          #pragma unroll
          for (int j = 0; j < 16; ++j) a[di][j] = As[ti * TX + di][g * 16 + j];
        #pragma unroll
        for (int dj = 0; dj < TY; ++dj)
          #pragma unroll
          for (int j = 0; j < 16; ++j) b[dj][j] = BsT[tj * TY + dj][g * 16 + j];
        if (TX == 2 && TY == 2) {
          fp8_group_quad(&acc[0][0], a[0], a[1], b[0], b[1]);
        } else {
          #pragma unroll
          for (int di = 0; di < TX; ++di)
            #pragma unroll
            for (int dj = 0; dj < TY; ++dj) acc[di][dj] = fp8_group(acc[di][dj], a[di], b[dj]);
        }
      }
      __syncthreads();
    }

    #pragma unroll
    for (int di = 0; di < TX; ++di)
      #pragma unroll
      for (int dj = 0; dj < TY; ++dj)
        total[di][dj] = first ? acc[di][dj] : hadd(total[di][dj], acc[di][dj]);
    first = false;
  }

  #pragma unroll
  for (int di = 0; di < TX; ++di) {
    #pragma unroll
    for (int dj = 0; dj < TY; ++dj) {
      long long row = row0 + di, col = col0 + dj;
      if (row < R && col < N) {
        float value = total[di][dj];
        if (flags & 8) value = mp_cubic_silu(value);
        if (flags & 16) OUT_RAW[batch * srawb + row * srawr + col * srawn] = __float2half_rn(value);
        if (flags & 32) OUT_E4[batch * se4b + row * se4r + col * se4n] = TABLE[f16_bits(value)];
      }
    }
  }
}

// ----------------------------------------------------------------------------------------------------
// The f16 GEMM: 8-product groups, F24, fixedToF16. flags: 1 seed, 2 write raw, 4 write e4.
// ----------------------------------------------------------------------------------------------------

__global__ void f16_chain_kernel(
    const __half* __restrict__ X, const __half* __restrict__ W,
    const __half* __restrict__ SEED,
    __half* __restrict__ OUT_RAW, __half* __restrict__ OUT_E4, const __half* __restrict__ TABLE,
    long long R, long long K, long long N,
    long long sxb, long long sxr, long long sxk,
    long long swb, long long swk, long long swn,
    long long sdb, long long sdr, long long sdn,
    long long srawb, long long srawr, long long srawn,
    long long se4b, long long se4r, long long se4n,
    int flags) {
  long long tiles_n = (N + 15) >> 4;
  long long tiles_r = (R + 15) >> 4;
  long long pid = blockIdx.x;
  int ntile = (int)(pid % tiles_n); pid /= tiles_n;
  int rtile = (int)(pid % tiles_r); pid /= tiles_r;
  long long batch = pid;

  long long row = (long long)rtile * 16 + (threadIdx.x >> 4);
  long long col = (long long)ntile * 16 + (threadIdx.x & 15);
  bool row_ok = row < R;
  bool col_ok = col < N;

  float acc = 0.0f;
  if ((flags & 1) && row_ok && col_ok) {
    acc = f16_value(__half_as_ushort(SEED[batch * sdb + row * sdr + col * sdn]));
  }

  for (long long kb = 0; kb < K; kb += 8) {
    float a[8], b[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      a[i] = row_ok ? f16_value(__half_as_ushort(X[batch * sxb + row * sxr + (kb + i) * sxk])) : 0.0f;
      b[i] = col_ok ? f16_value(__half_as_ushort(W[batch * swb + (kb + i) * swk + col * swn])) : 0.0f;
    }
    acc = f16_group(acc, a, b);
  }

  if (row_ok && col_ok) {
    if (flags & 2) OUT_RAW[batch * srawb + row * srawr + col * srawn] = __float2half_rn(acc);
    if (flags & 4) OUT_E4[batch * se4b + row * se4r + col * se4n] = TABLE[f16_bits(acc)];
  }
}

// ----------------------------------------------------------------------------------------------------
// The window attention (reference form): scores with the prior as the accumulator, expWeight, the
// 64-wide softmax tree over the 4x4-tiled key order, the four-group value fold, the E4M3 publication -
// one block per (window, head, 8 queries), reading and writing the field directly. Used by the numeric
// checks; the network runs window_fused_kernel below.
// ----------------------------------------------------------------------------------------------------

__global__ void window_attention_kernel(
    const __half* __restrict__ QKV, const __half* __restrict__ PRIOR,
    const __half* __restrict__ TABLE, const __half* __restrict__ EXPW, __half* __restrict__ OUT,
    int width, int height, int shift_x, int shift_y, int windows_x, int heads,
    long long q_row_s, long long q_head_s, long long q_ch_s,
    long long p_head_s, long long p_q_s, long long p_k_s,
    long long o_row_s, long long o_head_s, long long o_ch_s) {
  __shared__ __half sq[8][32];
  __shared__ __half sk[64][32];
  __shared__ __half sv[64][32];
  __shared__ float sprior[8][64];
  __shared__ float sscores[8][64];
  __shared__ float srecip[8];

  long long program = blockIdx.x;
  int task = (int)(program / heads);
  int head = (int)(program % heads);
  int qb = blockIdx.y;
  int wx = (task % windows_x) * 8 - shift_x;
  int wy = (task / windows_x) * 8 - shift_y;
  int t = threadIdx.x;

  // The 8 queries (natural slots) and their field rows.
  int q_slot_of[8];
  int q_row_of[8];
  bool q_ok[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    int slot = qb * 8 + i;
    int qx = wx + (slot & 7), qy = wy + (slot >> 3);
    q_ok[i] = (qx >= 0) && (qx < width) && (qy >= 0) && (qy < height);
    q_slot_of[i] = slot;
    q_row_of[i] = qy * width + qx;
  }

  // The keys in physical order -> their natural slots -> field rows.
  int k_nat[64];
  int k_row[64];
  bool k_ok[64];
  #pragma unroll
  for (int p = 0; p < 64; p += 32) {
    for (int j = 0; j < 32; ++j) {
      int key = p + j;
      int nat = inverse_tiled(key);
      int kx = wx + (nat & 7), ky = wy + (nat >> 3);
      k_ok[key] = (kx >= 0) && (kx < width) && (ky >= 0) && (ky < height);
      k_nat[key] = nat;
      k_row[key] = ky * width + kx;
    }
  }

  // Load q/k/v (zeros off the field) and the prior (every key keeps its prior; out-of-field keys score
  // expWeight(prior) and still count in the denominator, as windowAttendRef has it).
  for (int i = t; i < 256; i += 128) {
    int qi = i >> 5, c = i & 31;
    sq[qi][c] = q_ok[qi]
        ? QKV[(long long)q_row_of[qi] * q_row_s + head * q_head_s + c * q_ch_s] : __float2half(0.0f);
  }
  for (int i = t; i < 2048; i += 128) {
    int key = i >> 5, c = i & 31;
    if (k_ok[key]) {
      long long at = (long long)k_row[key] * q_row_s + head * q_head_s;
      sk[key][c] = QKV[at + (32 + c) * q_ch_s];
      sv[key][c] = QKV[at + (64 + c) * q_ch_s];
    } else {
      sk[key][c] = __float2half(0.0f);
      sv[key][c] = __float2half(0.0f);
    }
  }
  for (int i = t; i < 512; i += 128) {
    int qi = i >> 6, key = i & 63;
    sprior[qi][key] = f16_value(__half_as_ushort(
        PRIOR[head * p_head_s + (long long)q_slot_of[qi] * p_q_s + k_nat[key] * p_k_s]));
  }
  __syncthreads();

  // Scores: two 16-channel groups per (query, key), the prior as the accumulator.
  for (int i = t; i < 512; i += 128) {
    int qi = i >> 6, key = i & 63;
    float a[16], b[16];
    float acc = sprior[qi][key];
    for (int g = 0; g < 2; ++g) {
      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        a[j] = f16_value(__half_as_ushort(sq[qi][g * 16 + j]));
        b[j] = f16_value(__half_as_ushort(sk[key][g * 16 + j]));
      }
      acc = fp8_group(acc, a, b);
    }
    sscores[qi][key] = f16_value(__half_as_ushort(EXPW[f16_bits(acc)]));
  }
  __syncthreads();

  // The 64-wide softmax denominator: the pair tree over physical lanes, half adds throughout.
  if (t < 8) {
    float* s = sscores[t];
    float a0 = hadd(s[0], s[8]),   a1 = hadd(s[1], s[9]);
    float a2 = hadd(s[2], s[10]),  a3 = hadd(s[3], s[11]);
    float a4 = hadd(s[4], s[12]),  a5 = hadd(s[5], s[13]);
    float a6 = hadd(s[6], s[14]),  a7 = hadd(s[7], s[15]);
    float b0 = hadd(s[16], s[24]), b1 = hadd(s[17], s[25]);
    float b2 = hadd(s[18], s[26]), b3 = hadd(s[19], s[27]);
    float b4 = hadd(s[20], s[28]), b5 = hadd(s[21], s[29]);
    float b6 = hadd(s[22], s[30]), b7 = hadd(s[23], s[31]);
    float c0 = hadd(s[32], s[40]), c1 = hadd(s[33], s[41]);
    float c2 = hadd(s[34], s[42]), c3 = hadd(s[35], s[43]);
    float c4 = hadd(s[36], s[44]), c5 = hadd(s[37], s[45]);
    float c6 = hadd(s[38], s[46]), c7 = hadd(s[39], s[47]);
    float d0 = hadd(s[48], s[56]), d1 = hadd(s[49], s[57]);
    float d2 = hadd(s[50], s[58]), d3 = hadd(s[51], s[59]);
    float d4 = hadd(s[52], s[60]), d5 = hadd(s[53], s[61]);
    float d6 = hadd(s[54], s[62]), d7 = hadd(s[55], s[63]);
    float p0 = hadd(hadd(hadd(a0, b0), c0), d0);
    float p1 = hadd(hadd(hadd(a1, b1), c1), d1);
    float p2 = hadd(hadd(hadd(a2, b2), c2), d2);
    float p3 = hadd(hadd(hadd(a3, b3), c3), d3);
    float p4 = hadd(hadd(hadd(a4, b4), c4), d4);
    float p5 = hadd(hadd(hadd(a5, b5), c5), d5);
    float p6 = hadd(hadd(hadd(a6, b6), c6), d6);
    float p7 = hadd(hadd(hadd(a7, b7), c7), d7);
    float even = hadd(hadd(hadd(p0, p2), p4), p6);
    float odd = hadd(hadd(hadd(p1, p3), p5), p7);
    float total = hadd(even, odd);
    srecip[t] = f16_value(f16_bits(1.0f / total));
  }
  __syncthreads();

  // The weights: scores * reciprocal, rounded once, published E4M3 (unnormalized; the reciprocal reaches
  // the result through the value fold).
  for (int i = t; i < 512; i += 128) {
    int qi = i >> 6, key = i & 63;
    float w = sscores[qi][key] * srecip[qi];
    sscores[qi][key] = f16_value(__half_as_ushort(TABLE[f16_bits(w)]));
  }
  __syncthreads();

  // The value fold: four 16-key groups over physical keys, each group rounding through half.
  for (int i = t; i < 256; i += 128) {
    int qi = i >> 5, c = i & 31;
    float acc = 0.0f;
    float a[16], b[16];
    for (int g = 0; g < 4; ++g) {
      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        int key = g * 16 + j;
        a[j] = sscores[qi][key];
        b[j] = f16_value(__half_as_ushort(sv[key][c]));
      }
      acc = fp8_group(acc, a, b);
    }
    if (q_ok[qi]) {
      OUT[(long long)q_row_of[qi] * o_row_s + head * o_head_s + c * o_ch_s] = TABLE[f16_bits(acc)];
    }
  }
}

// ----------------------------------------------------------------------------------------------------
// The fused window attention: raw qkv in, one block per (window, head) covering all 64 queries. The
// cosine normalization and the q/k/v publications happen inside (they used to be ~10 full-tensor torch
// passes before the kernel); k is staged transposed so the score phase reads without bank conflicts.
// ----------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 3) window_fused_kernel(
    const __half* __restrict__ QKV, const __half* __restrict__ PRIOR, const __half* __restrict__ SCALES,
    const __half* __restrict__ TABLE, const __half* __restrict__ EXPW, __half* __restrict__ OUT,
    int width, int height, int shift_x, int shift_y, int windows_x, int heads,
    long long q_row_s, long long q_head_s, long long q_ch_s,
    long long p_head_s, long long p_q_s, long long p_k_s,
    long long o_row_s, long long o_head_s, long long o_ch_s) {
  __shared__ __half raw[64][96];      // natural slot row: q [0:32], k [32:64], v [64:96] (published in place)
  __shared__ __half kt[32][64];       // published k transposed: [channel][natural slot]
  __shared__ __half pri[64][64];      // prior [query slot][key slot], natural order
  __shared__ __half scr[64][64];      // scores/weights [query slot][physical key]
  __shared__ float srecip[64];

  long long program = blockIdx.x;
  int task = (int)(program / heads);
  int head = (int)(program % heads);
  int wx = (task % windows_x) * 8 - shift_x;
  int wy = (task / windows_x) * 8 - shift_y;
  int t = threadIdx.x;

  // Load raw qkv for the 64 window rows (natural slots), zeros off the field.
  for (int i = t; i < 64 * 96; i += 256) {
    int slot = i / 96, c = i % 96;
    int qx = wx + (slot & 7), qy = wy + (slot >> 3);
    bool ok = (qx >= 0) && (qx < width) && (qy >= 0) && (qy < height);
    raw[slot][c] = ok
        ? QKV[(long long)(qy * width + qx) * q_row_s + head * q_head_s + (long long)c * q_ch_s]
        : __float2half(0.0f);
  }
  __syncthreads();

  // The cosine normalization and the q/k/v publications, one task per (slot, part).
  if (t < 192) {
    int part = t / 64, slot = t % 64;
    int qx = wx + (slot & 7), qy = wy + (slot >> 3);
    bool ok = (qx >= 0) && (qx < width) && (qy >= 0) && (qy < height);
    if (!ok) {
      if (part == 0) for (int c = 0; c < 32; ++c) raw[slot][c] = __float2half(0.0f);
      if (part == 1) for (int c = 0; c < 32; ++c) kt[c][slot] = __float2half(0.0f);
      if (part == 2) for (int c = 0; c < 32; ++c) raw[slot][64 + c] = __float2half(0.0f);
    } else if (part == 0) {
      float x[32];
      for (int c = 0; c < 32; ++c) x[c] = f16_value(__half_as_ushort(raw[slot][c]));
      float qn = cosine_norm32(x);
      float scale_h = f16_value(__half_as_ushort(SCALES[head]));
      for (int c = 0; c < 32; ++c) {
        float q1 = f16_value(f16_bits(x[c] * qn));
        raw[slot][c] = TABLE[f16_bits(q1 * scale_h)];
      }
    } else if (part == 1) {
      float x[32];
      for (int c = 0; c < 32; ++c) x[c] = f16_value(__half_as_ushort(raw[slot][32 + c]));
      float kn = cosine_norm32(x);
      for (int c = 0; c < 32; ++c) {
        float k1 = f16_value(f16_bits(x[c] * kn));
        kt[c][slot] = TABLE[f16_bits(k1)];
      }
    } else {
      for (int c = 0; c < 32; ++c) {
        float v = f16_value(__half_as_ushort(raw[slot][64 + c]));
        raw[slot][64 + c] = TABLE[f16_bits(v)];
      }
    }
  }

  // The prior, the same 64x64 for every window.
  for (int i = t; i < 4096; i += 256) {
    int qi = i >> 6, nat = i & 63;
    pri[qi][nat] = PRIOR[head * p_head_s + (long long)qi * p_q_s + (long long)nat * p_k_s];
  }
  __syncthreads();

  // Scores: two 16-channel groups per (query slot, physical key), the prior as the accumulator.
  for (int i = t; i < 4096; i += 256) {
    int qi = i >> 6, p = i & 63;
    int nat = inverse_tiled(p);
    float a[16], b[16];
    float acc = f16_value(__half_as_ushort(pri[qi][nat]));
    for (int g = 0; g < 2; ++g) {
      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        a[j] = f16_value(__half_as_ushort(raw[qi][g * 16 + j]));
        b[j] = f16_value(__half_as_ushort(kt[g * 16 + j][nat]));
      }
      acc = fp8_group(acc, a, b);
    }
    scr[qi][p] = EXPW[f16_bits(acc)];
  }
  __syncthreads();

  // The 64-wide softmax denominator: the pair tree over physical lanes, half adds throughout.
  if (t < 64) {
    float s[64];
    for (int j = 0; j < 64; ++j) s[j] = f16_value(__half_as_ushort(scr[t][j]));
    float a0 = hadd(s[0], s[8]),   a1 = hadd(s[1], s[9]);
    float a2 = hadd(s[2], s[10]),  a3 = hadd(s[3], s[11]);
    float a4 = hadd(s[4], s[12]),  a5 = hadd(s[5], s[13]);
    float a6 = hadd(s[6], s[14]),  a7 = hadd(s[7], s[15]);
    float b0 = hadd(s[16], s[24]), b1 = hadd(s[17], s[25]);
    float b2 = hadd(s[18], s[26]), b3 = hadd(s[19], s[27]);
    float b4 = hadd(s[20], s[28]), b5 = hadd(s[21], s[29]);
    float b6 = hadd(s[22], s[30]), b7 = hadd(s[23], s[31]);
    float c0 = hadd(s[32], s[40]), c1 = hadd(s[33], s[41]);
    float c2 = hadd(s[34], s[42]), c3 = hadd(s[35], s[43]);
    float c4 = hadd(s[36], s[44]), c5 = hadd(s[37], s[45]);
    float c6 = hadd(s[38], s[46]), c7 = hadd(s[39], s[47]);
    float d0 = hadd(s[48], s[56]), d1 = hadd(s[49], s[57]);
    float d2 = hadd(s[50], s[58]), d3 = hadd(s[51], s[59]);
    float d4 = hadd(s[52], s[60]), d5 = hadd(s[53], s[61]);
    float d6 = hadd(s[54], s[62]), d7 = hadd(s[55], s[63]);
    float p0 = hadd(hadd(hadd(a0, b0), c0), d0);
    float p1 = hadd(hadd(hadd(a1, b1), c1), d1);
    float p2 = hadd(hadd(hadd(a2, b2), c2), d2);
    float p3 = hadd(hadd(hadd(a3, b3), c3), d3);
    float p4 = hadd(hadd(hadd(a4, b4), c4), d4);
    float p5 = hadd(hadd(hadd(a5, b5), c5), d5);
    float p6 = hadd(hadd(hadd(a6, b6), c6), d6);
    float p7 = hadd(hadd(hadd(a7, b7), c7), d7);
    float even = hadd(hadd(hadd(p0, p2), p4), p6);
    float odd = hadd(hadd(hadd(p1, p3), p5), p7);
    float total = hadd(even, odd);
    srecip[t] = f16_value(f16_bits(1.0f / total));
  }
  __syncthreads();

  // The weights: scores * reciprocal, rounded once, published E4M3 (unnormalized).
  for (int i = t; i < 4096; i += 256) {
    int qi = i >> 6, p = i & 63;
    float w = f16_value(__half_as_ushort(scr[qi][p])) * srecip[qi];
    scr[qi][p] = TABLE[f16_bits(w)];
  }
  __syncthreads();

  // The value fold: four 16-key groups over physical keys, each group rounding through half.
  for (int i = t; i < 64 * 32; i += 256) {
    int qi = i >> 5, c = i & 31;
    float acc = 0.0f;
    float a[16], b[16];
    for (int g = 0; g < 4; ++g) {
      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        int p = g * 16 + j;
        int nat = inverse_tiled(p);
        a[j] = f16_value(__half_as_ushort(scr[qi][p]));
        b[j] = f16_value(__half_as_ushort(raw[nat][64 + c]));
      }
      acc = fp8_group(acc, a, b);
    }
    int qx = wx + (qi & 7), qy = wy + (qi >> 3);
    if (qx >= 0 && qx < width && qy >= 0 && qy < height) {
      OUT[(long long)(qy * width + qx) * o_row_s + head * o_head_s + (long long)c * o_ch_s]
          = TABLE[f16_bits(acc)];
    }
  }
}

// ----------------------------------------------------------------------------------------------------
// The block FFN chain: expand (SiLU, published) -> contract (residual seed, raw + E4) -> qkv (raw), one
// launch per block, the intermediates never leaving shared memory. Fixed to the 32-channel blocks
// (hidden 128, qkv 96): the A operand's swizzle is applied at every stage's load.
// ----------------------------------------------------------------------------------------------------

__device__ __forceinline__ void ffn_stage_32(
    const float* __restrict__ Asrc, int astr, long long K,
    const __half* __restrict__ W, long long wk, long long wn, long long N, long long n0,
    const __half* __restrict__ RES, long long sres, const __half* __restrict__ AUX,
    const __half* __restrict__ TABLE, float* __restrict__ BsT,
    float* __restrict__ Dst, int dstr, bool silu,
    __half* __restrict__ g_raw, long long sraw,
    __half* __restrict__ g_e4, long long se4,
    long long row0, long long rows, int t) {
  int ti = t >> 4, tj = t & 15;
  float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  #pragma unroll
  for (int di = 0; di < 2; ++di) {
    #pragma unroll
    for (int dj = 0; dj < 2; ++dj) {
      long long row = row0 + 2 * ti + di, col = n0 + 2 * tj + dj;
      if (RES != 0 && row < rows && col < N) {
        float res = f16_value(__half_as_ushort(RES[row * sres + col]));
        float aux = f16_value(__half_as_ushort(AUX[col]));
        acc[di][dj] = f16_value(f16_bits(res * aux));
      }
    }
  }

  for (long long kb = 0; kb < K; kb += 32) {
    for (int i = t; i < 32 * 32; i += 256) {
      int k = i >> 5, col = i & 31;
      long long gcol = n0 + col;
      BsT[col * 33 + k] = (gcol < N && kb + k < K)
          ? __half2float(W[(kb + k) * wk + gcol * wn]) : 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int g = 0; g < 2; ++g) {
      float a[2][16], b[2][16];
      #pragma unroll
      for (int di = 0; di < 2; ++di)
        #pragma unroll
        for (int j = 0; j < 16; ++j)
          a[di][j] = Asrc[(2 * ti + di) * astr + kb + swz_const(g * 16 + j)];
      #pragma unroll
      for (int dj = 0; dj < 2; ++dj)
        #pragma unroll
        for (int j = 0; j < 16; ++j) b[dj][j] = BsT[(2 * tj + dj) * 33 + g * 16 + j];
      fp8_group_quad(&acc[0][0], a[0], a[1], b[0], b[1]);
    }
    __syncthreads();
  }

  #pragma unroll
  for (int di = 0; di < 2; ++di) {
    #pragma unroll
    for (int dj = 0; dj < 2; ++dj) {
      long long row = row0 + 2 * ti + di, col = n0 + 2 * tj + dj;
      if (row < rows && col < N) {
        float value = acc[di][dj];
        if (silu) value = mp_cubic_silu(value);
        if (Dst != 0) Dst[(2 * ti + di) * dstr + (n0 + 2 * tj + dj)] = __half2float(TABLE[f16_bits(value)]);
        if (g_raw != 0) g_raw[row * sraw + col] = __float2half_rn(value);
        if (g_e4 != 0) g_e4[row * se4 + col] = TABLE[f16_bits(value)];
      }
    }
  }
}

__global__ void __launch_bounds__(256, 3) block_ffn_kernel(
    const __half* __restrict__ STATE, const __half* __restrict__ W1, const __half* __restrict__ W2,
    const __half* __restrict__ W3, const __half* __restrict__ RES, const __half* __restrict__ AUX,
    const __half* __restrict__ TABLE,
    __half* __restrict__ OUT_RAW, __half* __restrict__ OUT_E4, __half* __restrict__ OUT_QKV,
    long long rows,
    long long s_state, long long s_res, long long s_raw, long long s_e4, long long s_qkv,
    int flags) {
  __shared__ float sx[32][32];        // the state rows
  __shared__ float ff[32][128];       // expand, SiLU'd and published
  __shared__ float cc[32][32];        // contract, published
  __shared__ float BsT[32][33];

  long long row0 = (long long)blockIdx.x * 32;
  int t = threadIdx.x;

  for (int i = t; i < 32 * 32; i += 256) {
    int rb = i >> 5, k = i & 31;
    long long row = row0 + rb;
    sx[rb][k] = (row < rows) ? __half2float(STATE[row * s_state + k]) : 0.0f;
  }
  __syncthreads();

  // expand: 32 -> 128, SiLU, published.
  for (long long n0 = 0; n0 < 128; n0 += 32) {
    ffn_stage_32(&sx[0][0], 32, 32, W1, 128, 1, 128, n0,
                 0, 0, 0, TABLE, &BsT[0][0], &ff[0][0], 128, true, 0, 0, 0, 0, row0, rows, t);
    __syncthreads();
  }

  // contract: 128 -> 32 with the residual seed, raw + E4 out.
  ffn_stage_32(&ff[0][0], 128, 128, W2, 32, 1, 32, 0,
               RES, s_res, AUX, TABLE, &BsT[0][0], &cc[0][0], 32, false,
               (flags & 1) ? OUT_RAW : 0, s_raw, (flags & 2) ? OUT_E4 : 0, s_e4, row0, rows, t);
  __syncthreads();

  // qkv: 32 -> 96, raw out.
  for (long long n0 = 0; n0 < 96; n0 += 32) {
    ffn_stage_32(&cc[0][0], 32, 32, W3, 96, 1, 96, n0,
                 0, 0, 0, TABLE, &BsT[0][0], 0, 0, false, OUT_QKV, s_qkv, 0, 0, row0, rows, t);
    __syncthreads();
  }
}

// ----------------------------------------------------------------------------------------------------
// Debug probe: the full block sequence with the intermediates dumped as floats (ff first, then cc).
// ----------------------------------------------------------------------------------------------------

__global__ void block_debug_kernel(
    const __half* __restrict__ STATE, const __half* __restrict__ W1, const __half* __restrict__ W2,
    const __half* __restrict__ W3, const __half* __restrict__ RES, const __half* __restrict__ AUX,
    const __half* __restrict__ TABLE, float* __restrict__ DBG,
    long long rows, long long s_state, long long s_res) {
  __shared__ float sx[32][32];
  __shared__ float ff[32][128];
  __shared__ float cc[32][32];
  __shared__ float BsT[32][33];
  int t = threadIdx.x;
  for (int i = t; i < 32 * 32; i += 256) {
    int rb = i >> 5, k = i & 31;
    sx[rb][k] = __half2float(STATE[rb * s_state + k]);
  }
  __syncthreads();
  for (long long n0 = 0; n0 < 128; n0 += 32) {
    ffn_stage_32(&sx[0][0], 32, 32, W1, 128, 1, 128, n0,
                 0, 0, 0, TABLE, &BsT[0][0], &ff[0][0], 128, true, 0, 0, 0, 0, 0LL, rows, t);
    __syncthreads();
  }
  for (int i = t; i < 32 * 128; i += 256) DBG[i] = ff[i / 128][i % 128];
  __syncthreads();
  ffn_stage_32(&ff[0][0], 128, 128, W2, 32, 1, 32, 0,
               RES, s_res, AUX, TABLE, &BsT[0][0], &cc[0][0], 32, false, 0, 0, 0, 0, 0LL, rows, t);
  __syncthreads();
  for (int i = t; i < 32 * 32; i += 256) DBG[4096 + i] = cc[i / 32][i % 32];
}

// ----------------------------------------------------------------------------------------------------
// The expert block FFN chain: expand (E broadcast experts x ch->128, SiLU, published) -> narrow
// (E x 128->32, published) -> contract (ch->ch with the residual seed, E4 out) -> qkv (ch->3ch, raw),
// one launch per block. The batched "expert" gemms are the same chains over per-expert weight rows and
// output column blocks. TR rows per block (32, or 16 for the 256-channel blocks' smem budget).
// ----------------------------------------------------------------------------------------------------

template <int TR>
__device__ __forceinline__ void expert_stage(
    const __half* __restrict__ Asrc, long long abase, long long astr, long long K,
    const __half* __restrict__ W, long long wrow, long long wk, long long wn, long long N,
    long long n0, long long wcol, const __half* __restrict__ RES, long long sres,
    const __half* __restrict__ AUX,
    const __half* __restrict__ TABLE, float* __restrict__ BsT,
    __half* __restrict__ Dst, long long dstr, bool silu,
    __half* __restrict__ g_raw, long long sraw,
    __half* __restrict__ g_e4, long long se4,
    long long row0, long long rows, int t) {
  int ti = t / 16, tj = t % 16;
  float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  #pragma unroll
  for (int di = 0; di < 2; ++di) {
    #pragma unroll
    for (int dj = 0; dj < 2; ++dj) {
      long long row = row0 + 2 * ti + di, col = n0 + 2 * tj + dj;
      if (RES != 0 && row < rows && col < N) {
        float res = f16_value(__half_as_ushort(RES[row * sres + col]));
        float aux = f16_value(__half_as_ushort(AUX[col]));
        acc[di][dj] = f16_value(f16_bits(res * aux));
      }
    }
  }

  for (long long kb = 0; kb < K; kb += 32) {
    for (int i = t; i < 32 * 32; i += TR * 8) {
      int k = i >> 5, col = i & 31;
      long long gcol = n0 + col;
      BsT[col * 33 + k] = (gcol < N && kb + k < K)
          ? __half2float(W[(wrow + kb + k) * wk + (wcol + col) * wn]) : 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int g = 0; g < 2; ++g) {
      float a[2][16], b[2][16];
      #pragma unroll
      for (int di = 0; di < 2; ++di)
        #pragma unroll
        for (int j = 0; j < 16; ++j)
          a[di][j] = __half2float(Asrc[abase + (2 * ti + di) * astr + kb + swz_const(g * 16 + j)]);
      #pragma unroll
      for (int dj = 0; dj < 2; ++dj)
        #pragma unroll
        for (int j = 0; j < 16; ++j) b[dj][j] = BsT[(2 * tj + dj) * 33 + g * 16 + j];
      fp8_group_quad(&acc[0][0], a[0], a[1], b[0], b[1]);
    }
    __syncthreads();
  }

  #pragma unroll
  for (int di = 0; di < 2; ++di) {
    #pragma unroll
    for (int dj = 0; dj < 2; ++dj) {
      long long row = row0 + 2 * ti + di, col = n0 + 2 * tj + dj;
      if (row < rows && col < N) {
        float value = acc[di][dj];
        if (silu) value = mp_cubic_silu(value);
        if (Dst != 0) Dst[(2 * ti + di) * dstr + col] = TABLE[f16_bits(value)];
        if (g_raw != 0) g_raw[row * sraw + col] = __float2half_rn(value);
        if (g_e4 != 0) g_e4[row * se4 + col] = TABLE[f16_bits(value)];
      }
    }
  }
}

template <int TR, int CH>
__global__ void __launch_bounds__(TR * 8, 3) block_ffn_expert_kernel(
    const __half* __restrict__ STATE, const __half* __restrict__ W1, const __half* __restrict__ W2,
    const __half* __restrict__ W3, const __half* __restrict__ W4,
    const __half* __restrict__ RES, const __half* __restrict__ AUX, const __half* __restrict__ TABLE,
    __half* __restrict__ OUT_RAW, __half* __restrict__ OUT_E4, __half* __restrict__ OUT_QKV,
    long long rows,
    long long s_state, long long s_res, long long s_raw, long long s_e4, long long s_qkv,
    int flags) {
  constexpr int E = CH / 32;
  // The intermediates: the expand of one expert (128 wide) is consumed by that expert's narrow at once,
  // so the chain streams expert by expert and only 128 columns of expand are ever resident. Everything
  // fits the default shared budget again (24-36 KiB), so several blocks sit on each SM.
  extern __shared__ __half smem[];
  __half* sx = smem;
  __half* ff = sx + TR * CH;
  __half* nn = ff + TR * 128;
  __half* cc = nn + TR * CH;
  float* BsT = (float*)(cc + TR * CH);

  long long row0 = (long long)blockIdx.x * TR;
  int t = threadIdx.x;
  const int threads = TR * 8;

  for (int i = t; i < TR * CH; i += threads) {
    int rb = i / CH, k = i % CH;
    long long row = row0 + rb;
    sx[rb * CH + k] = (row < rows) ? STATE[row * s_state + k] : __float2half(0.0f);
  }
  __syncthreads();

  // expand(e) -> narrow(e), streamed expert by expert: the expand's SiLU'd publications land in ff
  // (that expert's own 128 columns) and the narrow consumes them before the next expert.
  for (int e = 0; e < E; ++e) {
    for (long long n0 = 0; n0 < 128; n0 += 32) {
      expert_stage<TR>((const __half*)sx, 0, CH, CH,
                       W1, (long long)e * CH, 128, 1, 128, n0, n0,
                       0, 0, 0, TABLE, BsT, (__half*)ff, 128, true, 0, 0, 0, 0, row0, rows, t);
      __syncthreads();
    }
    expert_stage<TR>((const __half*)ff, 0, 128, 128,
                     W2, (long long)e * 128, 32, 1, CH, (long long)e * 32, 0,
                     0, 0, 0, TABLE, BsT, (__half*)nn, CH, false, 0, 0, 0, 0, row0, rows, t);
    __syncthreads();
  }

  // contract: ch -> ch with the residual seed; the E4 feeds the projection's residual (raw unused).
  for (long long n0 = 0; n0 < CH; n0 += 32) {
    expert_stage<TR>((const __half*)nn, 0, CH, CH,
                     W3, 0, CH, 1, CH, n0, n0,
                     RES, s_res, AUX, TABLE, BsT, (__half*)cc, CH, false,
                     (flags & 1) ? OUT_RAW : 0, s_raw, (flags & 2) ? OUT_E4 : 0, s_e4,
                     row0, rows, t);
    __syncthreads();
  }

  // qkv: ch -> 3ch, raw out.
  for (long long n0 = 0; n0 < 3 * CH; n0 += 32) {
    expert_stage<TR>((const __half*)cc, 0, CH, CH,
                     W4, 0, 3 * CH, 1, 3 * CH, n0, n0,
                     0, 0, 0, TABLE, BsT, 0, 0, false, OUT_QKV, s_qkv, 0, 0, row0, rows, t);
    __syncthreads();
  }
}

// ----------------------------------------------------------------------------------------------------
// The fused ViT attention: raw qkv in, one block per (head, 8 queries). The cosine normalization and the
// q/k/v publications (three half roundings on q: norm, sqrt(32), learned scale) happen inside; the scores
// carry no prior seed; the 64-wide softmax tree runs per key chunk with the running half adds of
// vitNormalize/vitAttend, the padding keys contribute vitExpWeight(0) and are subtracted once, and the
// value chain runs continuously across chunks with the accumulator kept in registers.
// ----------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 2) vit_fused_kernel(
    const __half* __restrict__ QKV, const __half* __restrict__ LEARNED, const __half* __restrict__ HSCALE,
    const __half* __restrict__ TABLE, const __half* __restrict__ VITEXPW, __half* __restrict__ OUT,
    int tokens, int padded, int heads,
    long long q_row_s, long long q_head_s, long long q_ch_s,
    long long o_row_s, long long o_head_s, long long o_ch_s) {
  extern __shared__ __half smem[];
  __half* qpub = smem;                       // [8][32]  the queries, published in place
  __half* kt = qpub + 8 * 32;                // [32][64] the current chunk's k, transposed
  __half* vpub = kt + 32 * 64;               // [64][32] the current chunk's v
  __half* scr = vpub + 64 * 32;              // [8][64]  the chunk's vitExpWeight values
  __half* w = scr + 8 * 64;                  // [8][padded] the E4M3 weights, kept across chunks
  float* total = (float*)(w + 8LL * padded);
  float* recip = total + 8;

  long long program = blockIdx.x;
  int task = (int)(program / heads);
  int head = (int)(program % heads);
  int qbase = task * 8;
  int t = threadIdx.x;
  float hs = f16_value(__half_as_ushort(HSCALE[0]));
  float learned = f16_value(__half_as_ushort(LEARNED[head]));

  // The eight queries' raw q, then their cosine norm and the three-rounding publication.
  for (int i = t; i < 8 * 32; i += 256) {
    int qi = i >> 5, c = i & 31;
    int tok = qbase + qi;
    qpub[qi * 32 + c] = tok < tokens
        ? QKV[(long long)tok * q_row_s + head * q_head_s + (long long)c * q_ch_s]
        : __float2half(0.0f);
  }
  __syncthreads();
  if (t < 8) {
    int tok = qbase + t;
    if (tok < tokens) {
      float x[32];
      for (int c = 0; c < 32; ++c) x[c] = f16_value(__half_as_ushort(qpub[t * 32 + c]));
      float qn = cosine_norm32(x);
      for (int c = 0; c < 32; ++c) {
        float q1 = f16_value(f16_bits(x[c] * qn));
        float q2 = f16_value(f16_bits(q1 * hs));
        float q3 = f16_value(f16_bits(q2 * learned));
        qpub[t * 32 + c] = TABLE[f16_bits(q3)];
      }
    }
  }
  if (t < 8) total[t] = 0.0f;
  __syncthreads();

  int qi_t = t >> 5, c_t = t & 31;
  float v_acc = 0.0f;

  for (int chunk = 0; chunk < padded; chunk += 64) {
    // The chunk's keys: k normalized and published transposed, v published; padding rows stay zero.
    if (t < 64) {
      int tok = chunk + t;
      if (tok < tokens) {
        float x[32];
        for (int c = 0; c < 32; ++c)
          x[c] = f16_value(__half_as_ushort(
              QKV[(long long)tok * q_row_s + head * q_head_s + (long long)(32 + c) * q_ch_s]));
        float kn = cosine_norm32(x);
        for (int c = 0; c < 32; ++c) {
          float k1 = f16_value(f16_bits(x[c] * kn));
          kt[c * 64 + t] = TABLE[f16_bits(k1)];
        }
        for (int c = 0; c < 32; ++c) {
          float v = f16_value(__half_as_ushort(
              QKV[(long long)tok * q_row_s + head * q_head_s + (long long)(64 + c) * q_ch_s]));
          vpub[t * 32 + c] = TABLE[f16_bits(v)];
        }
      } else {
        for (int c = 0; c < 32; ++c) {
          kt[c * 64 + t] = __float2half(0.0f);
          vpub[t * 32 + c] = __float2half(0.0f);
        }
      }
    }
    __syncthreads();

    // The scores: two 16-channel groups per (query, key), no accumulator seed.
    for (int i = t; i < 8 * 64; i += 256) {
      int qi = i >> 6, key = i & 63;
      float a[16], b[16];
      float acc = 0.0f;
      for (int g = 0; g < 2; ++g) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
          a[j] = f16_value(__half_as_ushort(qpub[qi * 32 + g * 16 + j]));
          b[j] = f16_value(__half_as_ushort(kt[(g * 16 + j) * 64 + key]));
        }
        acc = fp8_group(acc, a, b);
      }
      scr[qi * 64 + key] = VITEXPW[f16_bits(acc)];
    }
    __syncthreads();

    // The 64-wide tree per query, the running half add, and the E4M3 weights of the chunk.
    if (t < 8) {
      float s[64];
      for (int j = 0; j < 64; ++j) s[j] = f16_value(__half_as_ushort(scr[t * 64 + j]));
      float a0 = hadd(s[0], s[8]),   a1 = hadd(s[1], s[9]);
      float a2 = hadd(s[2], s[10]),  a3 = hadd(s[3], s[11]);
      float a4 = hadd(s[4], s[12]),  a5 = hadd(s[5], s[13]);
      float a6 = hadd(s[6], s[14]),  a7 = hadd(s[7], s[15]);
      float b0 = hadd(s[16], s[24]), b1 = hadd(s[17], s[25]);
      float b2 = hadd(s[18], s[26]), b3 = hadd(s[19], s[27]);
      float b4 = hadd(s[20], s[28]), b5 = hadd(s[21], s[29]);
      float b6 = hadd(s[22], s[30]), b7 = hadd(s[23], s[31]);
      float c0 = hadd(s[32], s[40]), c1 = hadd(s[33], s[41]);
      float c2 = hadd(s[34], s[42]), c3 = hadd(s[35], s[43]);
      float c4 = hadd(s[36], s[44]), c5 = hadd(s[37], s[45]);
      float c6 = hadd(s[38], s[46]), c7 = hadd(s[39], s[47]);
      float d0 = hadd(s[48], s[56]), d1 = hadd(s[49], s[57]);
      float d2 = hadd(s[50], s[58]), d3 = hadd(s[51], s[59]);
      float d4 = hadd(s[52], s[60]), d5 = hadd(s[53], s[61]);
      float d6 = hadd(s[54], s[62]), d7 = hadd(s[55], s[63]);
      float p0 = hadd(hadd(hadd(a0, b0), c0), d0);
      float p1 = hadd(hadd(hadd(a1, b1), c1), d1);
      float p2 = hadd(hadd(hadd(a2, b2), c2), d2);
      float p3 = hadd(hadd(hadd(a3, b3), c3), d3);
      float p4 = hadd(hadd(hadd(a4, b4), c4), d4);
      float p5 = hadd(hadd(hadd(a5, b5), c5), d5);
      float p6 = hadd(hadd(hadd(a6, b6), c6), d6);
      float p7 = hadd(hadd(hadd(a7, b7), c7), d7);
      float even = hadd(hadd(hadd(p0, p2), p4), p6);
      float odd = hadd(hadd(hadd(p1, p3), p5), p7);
      total[t] = f16_value(f16_bits(total[t] + hadd(even, odd)));
    }
    for (int i = t; i < 8 * 64; i += 256) {
      int qi = i >> 6, key = i & 63;
      w[qi * (long long)padded + chunk + key] = TABLE[__half_as_ushort(scr[qi * 64 + key])];
    }
    __syncthreads();

    // The value fold over the chunk's 64 keys; the chain continues across chunks.
    {
      float a[16], b[16];
      for (int g = 0; g < 4; ++g) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
          int key = g * 16 + j;
          a[j] = f16_value(__half_as_ushort(w[qi_t * (long long)padded + chunk + key]));
          b[j] = f16_value(__half_as_ushort(vpub[key * 32 + c_t]));
        }
        v_acc = fp8_group(v_acc, a, b);
      }
    }
    __syncthreads();
  }

  // The padding correction (each padding key adds vitExpWeight(0) to the denominator), then the
  // reciprocal and the single rounding of value * reciprocal.
  if (t < 8) {
    int padding = padded - tokens;
    float tv = total[t];
    if (padding > 0) {
      float zero_w = f16_value(__half_as_ushort(VITEXPW[0]));
      float correction = f16_value(f16_bits(zero_w * (float)padding));
      tv = f16_value(f16_bits(tv - correction));
    }
    recip[t] = f16_value(f16_bits(1.0f / tv));
  }
  __syncthreads();
  int tok = qbase + qi_t;
  if (tok < tokens) {
    float outv = f16_value(f16_bits(v_acc * recip[qi_t]));
    OUT[(long long)tok * o_row_s + head * o_head_s + (long long)c_t * o_ch_s] = TABLE[f16_bits(outv)];
  }
}

// ----------------------------------------------------------------------------------------------------
// Debug probe: the expert sequence with the intermediates dumped as floats (ff, then nn, then cc).
// ----------------------------------------------------------------------------------------------------

__global__ void block_expert_debug_kernel(
    const __half* __restrict__ STATE, const __half* __restrict__ W1, const __half* __restrict__ W2,
    const __half* __restrict__ W3, const __half* __restrict__ RES, const __half* __restrict__ AUX,
    const __half* __restrict__ TABLE, float* __restrict__ DBG,
    long long rows, long long s_state, long long s_res) {
  __shared__ __half sx[32][64];
  __shared__ __half ff[32][256];
  __shared__ __half nn[32][64];
  __shared__ __half cc[32][64];
  __shared__ float BsT[32][33];
  long long row0 = (long long)blockIdx.x * 32;
  int t = threadIdx.x;
  for (int i = t; i < 32 * 64; i += 256) {
    int rb = i / 64, k = i % 64;
    sx[rb][k] = (row0 + rb < rows) ? STATE[(row0 + rb) * s_state + k] : __float2half(0.0f);
  }
  __syncthreads();
  for (int e = 0; e < 2; ++e) {
    for (long long n0 = 0; n0 < 128; n0 += 32) {
      expert_stage<32>((const __half*)sx, 0, 64, 64,
                       W1, (long long)e * 64, 128, 1, 256, (long long)e * 128 + n0, n0,
                       0, 0, 0, TABLE, &BsT[0][0], (__half*)ff, 256, true, 0, 0, 0, 0, row0, rows, t);
      __syncthreads();
    }
  }
  for (int i = t; i < 32 * 256; i += 256) DBG[i] = __half2float(ff[i / 256][i % 256]);
  __syncthreads();
  for (int e = 0; e < 2; ++e) {
    expert_stage<32>((const __half*)ff, (long long)e * 128, 256, 128,
                     W2, (long long)e * 128, 32, 1, 64, (long long)e * 32, 0,
                     0, 0, 0, TABLE, &BsT[0][0], (__half*)nn, 64, false, 0, 0, 0, 0, row0, rows, t);
    __syncthreads();
  }
  for (int i = t; i < 32 * 64; i += 256) DBG[8192 + i] = __half2float(nn[i / 64][i % 64]);
  __syncthreads();
  for (long long n0 = 0; n0 < 64; n0 += 32) {
    expert_stage<32>((const __half*)nn, 0, 64, 64,
                     W3, 0, 64, 1, 64, n0, n0,
                     RES, s_res, AUX, TABLE, &BsT[0][0], (__half*)cc, 64, false, 0, 0, 0, 0,
                     row0, rows, t);
    __syncthreads();
  }
  for (int i = t; i < 32 * 64; i += 256) DBG[10240 + i] = __half2float(cc[i / 64][i % 64]);
}

// ----------------------------------------------------------------------------------------------------
// Launch wrappers (C ABI).
// ----------------------------------------------------------------------------------------------------
#define NR_THREADS 256

extern "C" {

int nr_fp8_gemm(
    const void* x, const void* w, const void* seed, const void* res, const void* aux,
    void* out_raw, void* out_e4, const void* table,
    long long R, long long K, long long N, long long B,
    long long sxb, long long sxr, long long sxk,
    long long swb, long long swk, long long swn,
    long long sdb, long long sdr, long long sdn,
    long long srb, long long srr, long long srn,
    long long srawb, long long srawr, long long srawn,
    long long se4b, long long se4r, long long se4n,
    int flags, long long partition, int kstep, int force_tile, void* stream) {
  long long TR = force_tile ? force_tile : ((R >= 256 && N >= 64) ? 32 : 16);
  long long blocks = ((N + TR - 1) / TR) * ((R + TR - 1) / TR) * B;
  if (blocks > 0x7fffffffLL) return 9;   // cudaErrorInvalidConfiguration
  if (TR == 32) {
    fp8_chain_kernel_v2<32, 32, 2, 2><<<(unsigned int)blocks, NR_THREADS, 0, (cudaStream_t)stream>>>(
        (const __half*)x, (const __half*)w, (const __half*)seed, (const __half*)res, (const __half*)aux,
        (__half*)out_raw, (__half*)out_e4, (const __half*)table,
        R, K, N, sxb, sxr, sxk, swb, swk, swn, sdb, sdr, sdn, srb, srr, srn,
        srawb, srawr, srawn, se4b, se4r, se4n, flags, partition, kstep);
  } else {
    fp8_chain_kernel_v2<16, 16, 1, 1><<<(unsigned int)blocks, NR_THREADS, 0, (cudaStream_t)stream>>>(
        (const __half*)x, (const __half*)w, (const __half*)seed, (const __half*)res, (const __half*)aux,
        (__half*)out_raw, (__half*)out_e4, (const __half*)table,
        R, K, N, sxb, sxr, sxk, swb, swk, swn, sdb, sdr, sdn, srb, srr, srn,
        srawb, srawr, srawn, se4b, se4r, se4n, flags, partition, kstep);
  }
  return (int)cudaGetLastError();
}

int nr_f16_gemm(
    const void* x, const void* w, const void* seed,
    void* out_raw, void* out_e4, const void* table,
    long long R, long long K, long long N, long long B,
    long long sxb, long long sxr, long long sxk,
    long long swb, long long swk, long long swn,
    long long sdb, long long sdr, long long sdn,
    long long srawb, long long srawr, long long srawn,
    long long se4b, long long se4r, long long se4n,
    int flags, void* stream) {
  long long blocks = ((N + 15) >> 4) * ((R + 15) >> 4) * B;
  if (blocks > 0x7fffffffLL) return 9;
  f16_chain_kernel<<<(unsigned int)blocks, NR_THREADS, 0, (cudaStream_t)stream>>>(
      (const __half*)x, (const __half*)w, (const __half*)seed,
      (__half*)out_raw, (__half*)out_e4, (const __half*)table,
      R, K, N, sxb, sxr, sxk, swb, swk, swn, sdb, sdr, sdn,
      srawb, srawr, srawn, se4b, se4r, se4n, flags);
  return (int)cudaGetLastError();
}

int nr_window_attention(
    const void* qkv, const void* prior, const void* table, const void* expw, void* out,
    int width, int height, int shift_x, int shift_y, int windows_x, int heads,
    long long q_row_s, long long q_head_s, long long q_ch_s,
    long long p_head_s, long long p_q_s, long long p_k_s,
    long long o_row_s, long long o_head_s, long long o_ch_s,
    void* stream) {
  long long tasks = (long long)windows_x * ((height + shift_y + 7) / 8);
  dim3 grid((unsigned int)(tasks * heads), 8, 1);
  window_attention_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>(
      (const __half*)qkv, (const __half*)prior, (const __half*)table, (const __half*)expw,
      (__half*)out, width, height, shift_x, shift_y, windows_x, heads,
      q_row_s, q_head_s, q_ch_s, p_head_s, p_q_s, p_k_s, o_row_s, o_head_s, o_ch_s);
  return (int)cudaGetLastError();
}

int nr_window_fused(
    const void* qkv, const void* prior, const void* scales, const void* table, const void* expw,
    void* out,
    int width, int height, int shift_x, int shift_y, int windows_x, int heads,
    long long q_row_s, long long q_head_s, long long q_ch_s,
    long long p_head_s, long long p_q_s, long long p_k_s,
    long long o_row_s, long long o_head_s, long long o_ch_s,
    void* stream) {
  long long tasks = (long long)windows_x * ((height + shift_y + 7) / 8);
  dim3 grid((unsigned int)(tasks * heads), 1, 1);
  window_fused_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
      (const __half*)qkv, (const __half*)prior, (const __half*)scales,
      (const __half*)table, (const __half*)expw, (__half*)out,
      width, height, shift_x, shift_y, windows_x, heads,
      q_row_s, q_head_s, q_ch_s, p_head_s, p_q_s, p_k_s, o_row_s, o_head_s, o_ch_s);
  return (int)cudaGetLastError();
}

int nr_block_ffn(
    const void* state, const void* w1, const void* w2, const void* w3,
    const void* res, const void* aux, const void* table,
    void* out_raw, void* out_e4, void* out_qkv,
    long long rows,
    long long s_state, long long s_res, long long s_raw, long long s_e4, long long s_qkv,
    int flags, void* stream) {
  long long blocks = (rows + 31) / 32;
  if (blocks > 0x7fffffffLL) return 9;
  block_ffn_kernel<<<(unsigned int)blocks, NR_THREADS, 0, (cudaStream_t)stream>>>(
      (const __half*)state, (const __half*)w1, (const __half*)w2, (const __half*)w3,
      (const __half*)res, (const __half*)aux, (const __half*)table,
      (__half*)out_raw, (__half*)out_e4, (__half*)out_qkv,
      rows, s_state, s_res, s_raw, s_e4, s_qkv, flags);
  return (int)cudaGetLastError();
}

int nr_vit_fused(
    const void* qkv, const void* learned, const void* hscale,
    const void* table, const void* vitexpw, void* out,
    int tokens, int padded, int heads,
    long long q_row_s, long long q_head_s, long long q_ch_s,
    long long o_row_s, long long o_head_s, long long o_ch_s,
    void* stream) {
  static bool tuned = false;
  if (!tuned) {
    cudaFuncSetAttribute((void*)vit_fused_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    tuned = true;
  }
  long long blocks = (long long)heads * ((tokens + 7) / 8);
  if (blocks > 0x7fffffffLL) return 9;
  size_t smem = sizeof(__half) * (8 * 32 + 32 * 64 + 64 * 32 + 8 * 64 + 8 * (size_t)padded)
              + sizeof(float) * 16;
  vit_fused_kernel<<<(unsigned int)blocks, 256, smem, (cudaStream_t)stream>>>(
      (const __half*)qkv, (const __half*)learned, (const __half*)hscale,
      (const __half*)table, (const __half*)vitexpw, (__half*)out,
      tokens, padded, heads, q_row_s, q_head_s, q_ch_s, o_row_s, o_head_s, o_ch_s);
  return (int)cudaGetLastError();
}

int nr_block_debug(const void* state, const void* w1, const void* w2, const void* w3,
                   const void* res, const void* aux, const void* table, void* dbg,
                   long long rows, long long s_state, long long s_res, void* stream) {
  block_debug_kernel<<<1, NR_THREADS, 0, (cudaStream_t)stream>>>(
      (const __half*)state, (const __half*)w1, (const __half*)w2, (const __half*)w3,
      (const __half*)res, (const __half*)aux, (const __half*)table, (float*)dbg,
      rows, s_state, s_res);
  return (int)cudaGetLastError();
}

int nr_block_ffn_expert(
    const void* state, const void* w1, const void* w2, const void* w3, const void* w4,
    const void* res, const void* aux, const void* table,
    void* out_raw, void* out_e4, void* out_qkv,
    long long rows, long long ch,
    long long s_state, long long s_res, long long s_raw, long long s_e4, long long s_qkv,
    int flags, void* stream) {
  static bool tuned = false;
  if (!tuned) {
    cudaFuncSetAttribute((void*)block_ffn_expert_kernel<32, 64>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    cudaFuncSetAttribute((void*)block_ffn_expert_kernel<32, 128>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    cudaFuncSetAttribute((void*)block_ffn_expert_kernel<16, 256>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    tuned = true;
  }
  if (ch == 64) {
    long long blocks = (rows + 31) / 32;
    if (blocks > 0x7fffffffLL) return 9;
    block_ffn_expert_kernel<32, 64><<<(unsigned int)blocks, 256, sizeof(__half) * 32 * (64 + 128 + 64 + 64) + 32 * 33 * 4, (cudaStream_t)stream>>>(
        (const __half*)state, (const __half*)w1, (const __half*)w2, (const __half*)w3,
        (const __half*)w4, (const __half*)res, (const __half*)aux, (const __half*)table,
        (__half*)out_raw, (__half*)out_e4, (__half*)out_qkv,
        rows, s_state, s_res, s_raw, s_e4, s_qkv, flags);
  } else if (ch == 128) {
    long long blocks = (rows + 31) / 32;
    if (blocks > 0x7fffffffLL) return 9;
    block_ffn_expert_kernel<32, 128><<<(unsigned int)blocks, 256, sizeof(__half) * 32 * (128 + 128 + 128 + 128) + 32 * 33 * 4, (cudaStream_t)stream>>>(
        (const __half*)state, (const __half*)w1, (const __half*)w2, (const __half*)w3,
        (const __half*)w4, (const __half*)res, (const __half*)aux, (const __half*)table,
        (__half*)out_raw, (__half*)out_e4, (__half*)out_qkv,
        rows, s_state, s_res, s_raw, s_e4, s_qkv, flags);
  } else if (ch == 256) {
    long long blocks = (rows + 15) / 16;
    if (blocks > 0x7fffffffLL) return 9;
    block_ffn_expert_kernel<16, 256><<<(unsigned int)blocks, 128, sizeof(__half) * 16 * (256 + 128 + 256 + 256) + 32 * 33 * 4, (cudaStream_t)stream>>>(
        (const __half*)state, (const __half*)w1, (const __half*)w2, (const __half*)w3,
        (const __half*)w4, (const __half*)res, (const __half*)aux, (const __half*)table,
        (__half*)out_raw, (__half*)out_e4, (__half*)out_qkv,
        rows, s_state, s_res, s_raw, s_e4, s_qkv, flags);
  } else {
    return 22;   // cudaErrorInvalidValue: only the 64/128/256-channel expert blocks
  }
  return (int)cudaGetLastError();
}

int nr_block_expert_debug(const void* state, const void* w1, const void* w2, const void* w3,
                          const void* res, const void* aux, const void* table, void* dbg,
                          long long rows, long long s_state, long long s_res, void* stream) {
  block_expert_debug_kernel<<<1, 256, 0, (cudaStream_t)stream>>>(
      (const __half*)state, (const __half*)w1, (const __half*)w2, (const __half*)w3,
      (const __half*)res, (const __half*)aux, (const __half*)table, (float*)dbg,
      rows, s_state, s_res);
  return (int)cudaGetLastError();
}

}  // extern "C"
