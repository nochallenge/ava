// TESTING / DESIGN ONLY — fused LIF scan kernels for ava_gpu (compiled to PTX by build.rs).
//
// One thread per (batch row b, channel c) runs the whole time loop for that neuron, so a layer's scan
// is one launch instead of ~10 small kernels per time step. Tensors are [B, T, D] row-major; for a
// fixed (b, t) consecutive threads read consecutive channels, so every load and store is coalesced.
//
// Forward, per channel:  v_t = beta·v_{t-1} + I_t - theta·s_{t-1};  s_t = [v_t - theta >= 0]
// Backward (reset detached, fast-sigmoid surrogate ds/dv = 1/(1 + slope·|v - theta|)^2):
//   gv_t = gs_t · sur(v_t) + beta · gv_{t+1};  gI_t = gv_t;  gbeta += gv_t · v_{t-1}
// The arithmetic uses explicit _rn intrinsics in the same order as the unrolled candle graph, so the
// compiler cannot contract it into FMAs and the spikes match the reference exactly.

// bf16 stores (speed round 7): round to nearest even, as __float2bfloat16_rn for finite values.
// v52 fix_S: and a NaN stays a NaN (quiet, its sign kept), as __float2bfloat16_rn does. Without this test the rounding
// add carried a NaN's mantissa into its exponent and sign: CUDA's NaN 0x7FFFFFFF became bf16 -0.0, so every bf16 matmul
// operand silently turned NaN into zero (runs/v52/fix_S.md). Infinities and every finite value are unchanged.
__device__ __forceinline__ unsigned short f2bf(float f) {
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (unsigned short)((u >> 16) | 0x0040u);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
__device__ __forceinline__ void put(float* p, size_t i, float v) { p[i] = v; }
__device__ __forceinline__ void put(unsigned short* p, size_t i, float v) { p[i] = f2bf(v); }

// v40: the scans are latency-bound (one thread per neuron runs T dependent steps; at ctx 2048 x batch 4 only 1,536
// threads exist), so the inputs of the next LIF_G steps are loaded while this group is worked on (indices clamped to
// the last step, so the loads need no branches): the loads are always in flight and the time per step is the
// arithmetic's own chain. The arithmetic is unchanged, step for step; emit(t, v, s) receives every step.
#define LIF_G 32
template <typename F>
__device__ __forceinline__ void lif_scan_fwd(const float* __restrict__ x, int T, int D, float bt, float theta, F emit) {
    float v = 0.0f, sp = 0.0f;
    const int tl = T - 1;
    float xa[LIF_G];
#pragma unroll
    for (int u = 0; u < LIF_G; ++u) xa[u] = x[(size_t)min(u, tl) * D];
    int t0 = 0;
    for (; t0 + LIF_G <= T; t0 += LIF_G) {
        float xb[LIF_G];
#pragma unroll
        for (int u = 0; u < LIF_G; ++u) xb[u] = x[(size_t)min(t0 + LIF_G + u, tl) * D];
#pragma unroll
        for (int u = 0; u < LIF_G; ++u) {
            v = __fsub_rn(__fadd_rn(__fmul_rn(bt, v), xa[u]), __fmul_rn(theta, sp));
            sp = __fsub_rn(v, theta) >= 0.0f ? 1.0f : 0.0f;
            emit(t0 + u, v, sp);
        }
#pragma unroll
        for (int u = 0; u < LIF_G; ++u) xa[u] = xb[u];
    }
#pragma unroll
    for (int u = 0; u < LIF_G; ++u) {
        if (t0 + u < T) {
            v = __fsub_rn(__fadd_rn(__fmul_rn(bt, v), xa[u]), __fmul_rn(theta, sp));
            sp = __fsub_rn(v, theta) >= 0.0f ? 1.0f : 0.0f;
            emit(t0 + u, v, sp);
        }
    }
}

template <typename O>
__device__ __forceinline__ void lif_fwd_impl(const float* __restrict__ cur, const float* __restrict__ beta, O* __restrict__ s,
                                             int B, int T, int D, float theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * D) return;
    int b = i / D, c = i - b * D;
    const float bt = beta[c];
    const float* x = cur + (size_t)b * T * D + c;
    O* o = s + (size_t)b * T * D + c;
    lif_scan_fwd(x, T, D, bt, theta, [&](int t, float, float st) { put(o, (size_t)t * D, st); });
}
extern "C" __global__ void lif_fwd(const float* __restrict__ cur, const float* __restrict__ beta, float* __restrict__ s, int B,
                                   int T, int D, float theta) {
    lif_fwd_impl(cur, beta, s, B, T, D, theta);
}
extern "C" __global__ void lif_fwd_b(const float* __restrict__ cur, const float* __restrict__ beta, unsigned short* __restrict__ s,
                                     int B, int T, int D, float theta) {
    lif_fwd_impl(cur, beta, s, B, T, D, theta);
}

// out is [B, T+1, D]: rows 0..T-1 receive dL/dI, row T receives this batch row's share of dL/dbeta
// (summed over b by the caller, so the reduction is deterministic). Rows 0..T-1 first hold the
// recomputed membrane v_t, which the reverse pass overwrites in place with gv_t.
extern "C" __global__ void lif_bwd(const float* __restrict__ cur, const float* __restrict__ beta,
                                   const float* __restrict__ gs, float* __restrict__ out, int B, int T,
                                   int D, float theta, float slope) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * D) return;
    int b = i / D, c = i - b * D;
    const float bt = beta[c];
    const float* x = cur + (size_t)b * T * D + c;
    const float* g = gs + (size_t)b * T * D + c;
    float* o = out + (size_t)b * (T + 1) * D + c;
    float v = 0.0f, sp = 0.0f;
    for (int t = 0; t < T; ++t) {
        v = __fsub_rn(__fadd_rn(__fmul_rn(bt, v), x[(size_t)t * D]), __fmul_rn(theta, sp));
        sp = __fsub_rn(v, theta) >= 0.0f ? 1.0f : 0.0f;
        o[(size_t)t * D] = v;
    }
    float gv_next = 0.0f, gb = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        float vt = o[(size_t)t * D];
        float a = 1.0f + slope * fabsf(vt - theta);
        float gv = g[(size_t)t * D] / (a * a) + bt * gv_next;
        if (t > 0) gb += gv * o[(size_t)(t - 1) * D];
        o[(size_t)t * D] = gv;
        gv_next = gv;
    }
    o[(size_t)T * D] = gb;
}

// kWTA mask: one block per row of u [N, H]. Finds the K-th largest value exactly (radix select over the
// order-preserving uint32 image of each float, 8 bits per pass, 4 passes), then writes
// m = [u - kth >= 0] as 1/0, the same expression model::kwta_mask computes after a full sort, so the
// mask is identical (ties at the K-th value are all kept). Dynamic shared memory: H uint32 keys.
extern "C" __global__ void kwta_mask_rows(const float* __restrict__ u, float* __restrict__ m, int N, int H, int K) {
    extern __shared__ unsigned int keys[];
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix, s_k;
    const int row = blockIdx.x;
    if (row >= N) return;
    const float* x = u + (size_t)row * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        unsigned int b = __float_as_uint(x[i]);
        keys[i] = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    }
    if (threadIdx.x == 0) { s_prefix = 0u; s_k = (unsigned int)K; }
    __syncthreads();
    unsigned int fixed = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        const unsigned int prefix = s_prefix;
        for (int i = threadIdx.x; i < H; i += blockDim.x) {
            const unsigned int kk = keys[i];
            if ((kk & fixed) == prefix) atomicAdd(&hist[(kk >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int k = s_k, cum = 0u;
            int bin = 255;
            for (; bin > 0; --bin) {
                if (cum + hist[bin] >= k) break;
                cum += hist[bin];
            }
            s_k = k - cum;
            s_prefix = prefix | ((unsigned int)bin << shift);
        }
        __syncthreads();
        fixed |= 255u << shift;
    }
    const unsigned int kk = s_prefix;
    const float kth = __uint_as_float((kk & 0x80000000u) ? (kk ^ 0x80000000u) : ~kk);
    float* o = m + (size_t)row * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) o[i] = __fsub_rn(x[i], kth) >= 0.0f ? 1.0f : 0.0f;
}

// Fused cross-entropy. Forward, one block per row of logits x [N, V]: out[row] = (loss, lse) with
// lse = log sum exp(x) and loss = lse - x[target] (0 when target >= V). Backward, one block per row:
// gx = (exp(x - lse) - [i == target]) * gout[row], with aux[row] = (gout, lse).
__device__ float block_max(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : __int_as_float(0xff800000);
    if (w == 0) for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}
__device__ float block_sum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}
extern "C" __global__ void ce_fwd(const float* __restrict__ x, const unsigned int* __restrict__ tg, float* __restrict__ out, int N, int V) {
    __shared__ float sh[32];
    const int row = blockIdx.x;
    if (row >= N) return;
    const float* r = x + (size_t)row * V;
    float m = __int_as_float(0xff800000);
    for (int i = threadIdx.x; i < V; i += blockDim.x) m = fmaxf(m, r[i]);
    m = block_max(m, sh);
    float s = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) s += expf(r[i] - m);
    s = block_sum(s, sh);
    if (threadIdx.x == 0) {
        const float lse = m + logf(s);
        const unsigned int t = tg[row];
        out[2 * row] = (t < (unsigned int)V) ? lse - r[t] : 0.0f;
        out[2 * row + 1] = lse;
    }
}
extern "C" __global__ void ce_bwd(const float* __restrict__ x, const unsigned int* __restrict__ tg, const float* __restrict__ aux, float* __restrict__ gx, int N, int V) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const unsigned int t = tg[row];
    const float g = (t < (unsigned int)V) ? aux[2 * row] : 0.0f, lse = aux[2 * row + 1];
    const float* r = x + (size_t)row * V;
    float* o = gx + (size_t)row * V;
    for (int i = threadIdx.x; i < V; i += blockDim.x) o[i] = (expf(r[i] - lse) - (i == (int)t ? 1.0f : 0.0f)) * g;
}

// ---- speed round 2: fused ops (rounding differs from candle's primitive chains in the last bits) ----

// kWTA + relu in one pass: the same radix select as kwta_mask_rows, then a = u where (u - kth >= 0 and
// u > 0), else 0, which equals relu(u * mask).
// v47: the select (the row's K-th largest value) is kwta_kth, shared with kwta_relu2_impl; the arithmetic is unchanged.
__device__ __forceinline__ float kwta_kth(const float* __restrict__ x, int H, int K, unsigned int* keys, unsigned int* hist,
                                          unsigned int& s_prefix, unsigned int& s_k) {
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        unsigned int b = __float_as_uint(x[i]);
        keys[i] = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    }
    if (threadIdx.x == 0) { s_prefix = 0u; s_k = (unsigned int)K; }
    __syncthreads();
    unsigned int fixed = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        const unsigned int prefix = s_prefix;
        for (int i = threadIdx.x; i < H; i += blockDim.x) {
            const unsigned int kk = keys[i];
            if ((kk & fixed) == prefix) atomicAdd(&hist[(kk >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            // Speed round 13c: the walk down the bins (from 255: stop at the first bin where the count from the top
            // reaches k, else bin 0) by one warp: lane l holds bins 255-8l .. 248-8l, a prefix sum over lanes picks
            // the lane, that lane walks its 8 bins. Integer counts, so the same bin and remainder as one thread.
            const int l = threadIdx.x;
            const unsigned int k = s_k;
            unsigned int c[8], sum = 0u;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                c[j] = hist[255 - 8 * l - j];
                sum += c[j];
            }
            unsigned int incl = sum;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const unsigned int y = __shfl_up_sync(0xffffffffu, incl, off);
                if (l >= off) incl += y;
            }
            const unsigned int ballot = __ballot_sync(0xffffffffu, incl >= k);
            const int lsel = ballot ? __ffs(ballot) - 1 : 31;
            if (l == lsel) {
                unsigned int cum = incl - sum;
                int bin = 255 - 8 * l;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    bin = 255 - 8 * l - j;
                    if (bin == 0 || cum + c[j] >= k) break;
                    cum += c[j];
                }
                s_k = k - cum;
                s_prefix = prefix | ((unsigned int)bin << shift);
            }
        }
        __syncthreads();
        fixed |= 255u << shift;
    }
    const unsigned int kk = s_prefix;
    return __uint_as_float((kk & 0x80000000u) ? (kk ^ 0x80000000u) : ~kk);
}
template <typename O>
__device__ __forceinline__ void kwta_relu_impl(const float* __restrict__ u, O* __restrict__ a, int N, int H, int K, unsigned int* keys,
                                               unsigned int* hist, unsigned int& s_prefix, unsigned int& s_k) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const float* x = u + (size_t)row * H;
    const float kth = kwta_kth(x, H, K, keys, hist, s_prefix, s_k);
    O* o = a + (size_t)row * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float xi = x[i];
        put(o, i, (__fsub_rn(xi, kth) >= 0.0f && xi > 0.0f) ? xi : 0.0f);
    }
}
extern "C" __global__ void kwta_relu_rows(const float* __restrict__ u, float* __restrict__ a, int N, int H, int K) {
    extern __shared__ unsigned int keys[];
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix, s_k;
    kwta_relu_impl(u, a, N, H, K, keys, hist, s_prefix, s_k);
}
extern "C" __global__ void kwta_relu_rows_b(const float* __restrict__ u, unsigned short* __restrict__ a, int N, int H, int K) {
    extern __shared__ unsigned int keys[];
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix, s_k;
    kwta_relu_impl(u, a, N, H, K, keys, hist, s_prefix, s_k);
}

// Backward of kwta_relu: gu = ga where a > 0, else 0.
extern "C" __global__ void relu_mask_bwd(const float* __restrict__ a, const float* __restrict__ ga, float* __restrict__ gu, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gu[i] = a[i] > 0.0f ? ga[i] : 0.0f;
}

// rmsnorm, one block per row of x [N, D]: y = (x * inv) * g, inv = 1 / sqrt(sum(x^2) / D + eps).
template <typename O>
__device__ __forceinline__ void rms_fwd_impl(const float* __restrict__ x, const float* __restrict__ g, O* __restrict__ y, int N, int D,
                                             float eps, float* sh) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const float* xr = x + (size_t)row * D;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, sh);
    const float inv = 1.0f / sqrtf(ss / (float)D + eps);
    O* yr = y + (size_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) put(yr, i, (xr[i] * inv) * g[i]);
}
extern "C" __global__ void rms_fwd(const float* __restrict__ x, const float* __restrict__ g, float* __restrict__ y, int N, int D, float eps) {
    __shared__ float sh[32];
    rms_fwd_impl(x, g, y, N, D, eps, sh);
}
extern "C" __global__ void rms_fwd_b(const float* __restrict__ x, const float* __restrict__ g, unsigned short* __restrict__ y, int N, int D,
                                     float eps) {
    __shared__ float sh[32];
    rms_fwd_impl(x, g, y, N, D, eps, sh);
}

// rmsnorm backward, one block per row: dx = inv * (gy*g - xhat * sum(gy*g*xhat) / D), xhat = x * inv.
// Also stores inv per row for rms_bwd_gain.
extern "C" __global__ void rms_bwd_rows(const float* __restrict__ x, const float* __restrict__ g, const float* __restrict__ gy,
                                        float* __restrict__ dx, float* __restrict__ invs, int N, int D, float eps) {
    __shared__ float sh[32];
    const int row = blockIdx.x;
    if (row >= N) return;
    const float* xr = x + (size_t)row * D;
    const float* gr = gy + (size_t)row * D;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, sh);
    const float inv = 1.0f / sqrtf(ss / (float)D + eps);
    float dot = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) dot += gr[i] * g[i] * (xr[i] * inv);
    dot = block_sum(dot, sh) / (float)D;
    float* o = dx + (size_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) o[i] = inv * (gr[i] * g[i] - (xr[i] * inv) * dot);
    if (threadIdx.x == 0) invs[row] = inv;
}

// rmsnorm gain gradient, partial sums: block p sums rows [p*R, p*R + R) of gy * x * inv per column.
// Fixed order (no atomics), so the result is deterministic. part [P, D].
extern "C" __global__ void rms_bwd_gain(const float* __restrict__ x, const float* __restrict__ gy, const float* __restrict__ invs,
                                        float* __restrict__ part, int N, int D, int R) {
    const int p = blockIdx.x;
    const int r0 = p * R, r1 = min(N, r0 + R);
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float acc = 0.0f;
        for (int r = r0; r < r1; ++r) acc += gy[(size_t)r * D + c] * (x[(size_t)r * D + c] * invs[r]);
        part[(size_t)p * D + c] = acc;
    }
}

// AdamW, one thread per element; out = [w | m | v] (3n): m = b1 m + (1-b1) g s; v = b2 v + (1-b2) (g s)^2;
// w = w * shrink - step * m / (sqrt(v) * ibc2 + eps). s = the clipping scale.
extern "C" __global__ void adamw_step(const float* __restrict__ w, const float* __restrict__ g, const float* __restrict__ m,
                                      const float* __restrict__ v, float* __restrict__ out, int n, float s, float b1, float c1,
                                      float b2, float c2, float eps, float step, float ibc2, float shrink) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gi = g[i] * s;
    const float mi = m[i] * b1 + gi * c1;
    const float vi = v[i] * b2 + (gi * gi) * c2;
    const float den = sqrtf(vi) * ibc2 + eps;
    out[i] = w[i] * shrink - (mi / den) * step;
    out[n + i] = mi;
    out[2 * n + i] = vi;
}

// ---- speed round 4: fused causal attention for the T yardstick (head size 32) ----
// q, k, v [BH, T, 32]. Same maths as softmax(q k^T * scale + causal mask) v, without storing the [T, T] scores.
// Each block owns a tile of 32 rows (4 warps x 8 rows, row = w + 4r) and walks 32-wide tiles of the other side
// staged in shared memory. Forward keeps a running max and sum per row (online softmax).
// out [BH, T, 33]: o (32) then lse = m + log(l). The backward recomputes P from lse; fixed sum order, no atomics.
#define AHS 32
#define AT 32
#define AW 4
#define ARPW 8
__device__ __forceinline__ float wsum(float v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); return v; }
__device__ __forceinline__ float wmax(float v) { for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o)); return v; }

// Stage rows r0..r0+31 of a [T, 32] matrix into a shared tile (row stride S), zeros past T.
template <int S>
__device__ __forceinline__ void stage(float (*dst)[S], const float* __restrict__ src, int r0, int T) {
    for (int e = threadIdx.x; e < AT * AHS; e += AW * 32) {
        const int r = e / AHS, d = e % AHS;
        dst[r][d] = (r0 + r < T) ? src[(size_t)(r0 + r) * AHS + d] : 0.0f;
    }
}

extern "C" __global__ void __launch_bounds__(128) attn_fwd(const float* __restrict__ q, const float* __restrict__ k,
                                    const float* __restrict__ v, float* __restrict__ out, int BH, int T, float scale) {
    __shared__ float Qs[AT][AHS];
    __shared__ float Ks[AT][AHS + 1];
    __shared__ float Vs[AT][AHS];
    __shared__ float Ps[AW][ARPW][AT];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.y, i0 = blockIdx.x * AT;
    const size_t base = (size_t)bh * T * AHS;
    stage<AHS>(Qs, q + base, i0, T);
    float m[ARPW], l[ARPW], acc[ARPW];
#pragma unroll
    for (int r = 0; r < ARPW; ++r) { m[r] = -INFINITY; l[r] = 0.0f; acc[r] = 0.0f; }
    for (int j0 = 0; j0 <= i0; j0 += AT) {
        __syncthreads();
        stage<AHS + 1>(Ks, k + base, j0, T);
        stage<AHS>(Vs, v + base, j0, T);
        __syncthreads();
        float kr[AHS];
#pragma unroll
        for (int d = 0; d < AHS; ++d) kr[d] = Ks[lane][d];
        const int j = j0 + lane;
#pragma unroll
        for (int r = 0; r < ARPW; ++r) {
            const int rl = w + AW * r, i = i0 + rl;
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < AHS; ++d) s += Qs[rl][d] * kr[d];
            const bool ok = j <= i && j < T;
            s = ok ? s * scale : -INFINITY;
            const float nm = fmaxf(m[r], wmax(s)); // finite: key j0 <= i0 <= i is always present
            const float p = ok ? expf(s - nm) : 0.0f;
            const float corr = expf(m[r] - nm);
            l[r] = l[r] * corr + wsum(p);
            acc[r] *= corr;
            m[r] = nm;
            Ps[w][r][lane] = p;
        }
        __syncwarp();
        for (int jj = 0; jj < AT; ++jj) {
            const float vv = Vs[jj][lane];
#pragma unroll
            for (int r = 0; r < ARPW; ++r) acc[r] += Ps[w][r][jj] * vv;
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int i = i0 + w + AW * r;
        if (i < T) {
            float* o = out + ((size_t)bh * T + i) * (AHS + 1);
            o[lane] = acc[r] / l[r];
            if (lane == 0) o[AHS] = m[r] + logf(l[r]);
        }
    }
}

// Backward 1, a block per 32 query rows: D_i = sum(dO_i * O_i) (also written to dvec);
// dQ_i = scale * sum_j P_ij (dO_i . v_j - D_i) k_j.
extern "C" __global__ void __launch_bounds__(128) attn_bwd_q(const float* __restrict__ q, const float* __restrict__ k,
                                      const float* __restrict__ v, const float* __restrict__ out, const float* __restrict__ dout,
                                      float* __restrict__ dq, float* __restrict__ dvec, int BH, int T, float scale) {
    __shared__ float Qs[AT][AHS];
    __shared__ float DOs[AT][AHS];
    __shared__ float Ks[AT][AHS + 1];
    __shared__ float Vs[AT][AHS + 1];
    __shared__ float Ls[AT], Ds[AT];
    __shared__ float Ss[AW][ARPW][AT];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.y, i0 = blockIdx.x * AT;
    const size_t base = (size_t)bh * T * AHS;
    stage<AHS>(Qs, q + base, i0, T);
    stage<AHS>(DOs, dout + base, i0, T);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int rl = w + AW * r, i = i0 + rl;
        const float* orow = out + ((size_t)bh * T + i) * (AHS + 1);
        const float Di = wsum(i < T ? DOs[rl][lane] * orow[lane] : 0.0f);
        if (lane == 0) {
            Ds[rl] = Di;
            Ls[rl] = i < T ? orow[AHS] : 0.0f;
            if (i < T) dvec[(size_t)bh * T + i] = Di;
        }
    }
    __syncwarp();
    float acc[ARPW];
#pragma unroll
    for (int r = 0; r < ARPW; ++r) acc[r] = 0.0f;
    for (int j0 = 0; j0 <= i0; j0 += AT) {
        __syncthreads();
        stage<AHS + 1>(Ks, k + base, j0, T);
        stage<AHS + 1>(Vs, v + base, j0, T);
        __syncthreads();
        float kr[AHS], vr[AHS];
#pragma unroll
        for (int d = 0; d < AHS; ++d) { kr[d] = Ks[lane][d]; vr[d] = Vs[lane][d]; }
        const int j = j0 + lane;
#pragma unroll
        for (int r = 0; r < ARPW; ++r) {
            const int rl = w + AW * r, i = i0 + rl;
            float s = 0.0f, dp = 0.0f;
#pragma unroll
            for (int d = 0; d < AHS; ++d) { s += Qs[rl][d] * kr[d]; dp += DOs[rl][d] * vr[d]; }
            const bool ok = j <= i && j < T && i < T;
            Ss[w][r][lane] = ok ? expf(s * scale - Ls[rl]) * (dp - Ds[rl]) : 0.0f;
        }
        __syncwarp();
        for (int jj = 0; jj < AT; ++jj) {
            const float kv = Ks[jj][lane];
#pragma unroll
            for (int r = 0; r < ARPW; ++r) acc[r] += Ss[w][r][jj] * kv;
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int i = i0 + w + AW * r;
        if (i < T) dq[base + (size_t)i * AHS + lane] = acc[r] * scale;
    }
}

// Backward 2, a block per 32 key rows: dV_j = sum_{i>=j} P_ij dO_i; dK_j = scale * sum_{i>=j} P_ij (dO_i . v_j - D_i) q_i.
extern "C" __global__ void __launch_bounds__(128) attn_bwd_kv(const float* __restrict__ q, const float* __restrict__ k,
                                       const float* __restrict__ v, const float* __restrict__ out, const float* __restrict__ dout,
                                       const float* __restrict__ dvec, float* __restrict__ dk, float* __restrict__ dv,
                                       int BH, int T, float scale) {
    __shared__ float Ks[AT][AHS];
    __shared__ float Vs[AT][AHS];
    __shared__ float Qs[AT][AHS + 1];
    __shared__ float DOs[AT][AHS + 1];
    __shared__ float Ls[AT], Ds[AT];
    __shared__ float Ps[AW][ARPW][AT];
    __shared__ float DSs[AW][ARPW][AT];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.y, j0 = blockIdx.x * AT;
    const size_t base = (size_t)bh * T * AHS;
    stage<AHS>(Ks, k + base, j0, T);
    stage<AHS>(Vs, v + base, j0, T);
    float gk[ARPW], gv[ARPW];
#pragma unroll
    for (int r = 0; r < ARPW; ++r) { gk[r] = 0.0f; gv[r] = 0.0f; }
    for (int i0 = j0; i0 < T; i0 += AT) {
        __syncthreads();
        stage<AHS + 1>(Qs, q + base, i0, T);
        stage<AHS + 1>(DOs, dout + base, i0, T);
        if (threadIdx.x < AT) {
            const int i = i0 + threadIdx.x;
            Ls[threadIdx.x] = i < T ? out[((size_t)bh * T + i) * (AHS + 1) + AHS] : 0.0f;
            Ds[threadIdx.x] = i < T ? dvec[(size_t)bh * T + i] : 0.0f;
        }
        __syncthreads();
        float qr[AHS], dor[AHS];
#pragma unroll
        for (int d = 0; d < AHS; ++d) { qr[d] = Qs[lane][d]; dor[d] = DOs[lane][d]; }
        const int i = i0 + lane;
        const float Li = Ls[lane], Di = Ds[lane];
#pragma unroll
        for (int r = 0; r < ARPW; ++r) {
            const int rl = w + AW * r, j = j0 + rl;
            float s = 0.0f, dp = 0.0f;
#pragma unroll
            for (int d = 0; d < AHS; ++d) { s += qr[d] * Ks[rl][d]; dp += dor[d] * Vs[rl][d]; }
            const bool ok = i >= j && i < T && j < T;
            const float p = ok ? expf(s * scale - Li) : 0.0f;
            Ps[w][r][lane] = p;
            DSs[w][r][lane] = p * (dp - Di);
        }
        __syncwarp();
        for (int ii = 0; ii < AT; ++ii) {
            const float dov = DOs[ii][lane], qv = Qs[ii][lane];
#pragma unroll
            for (int r = 0; r < ARPW; ++r) { gv[r] += Ps[w][r][ii] * dov; gk[r] += DSs[w][r][ii] * qv; }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int j = j0 + w + AW * r;
        if (j < T) {
            dk[base + (size_t)j * AHS + lane] = gk[r] * scale;
            dv[base + (size_t)j * AHS + lane] = gv[r];
        }
    }
}

// ---- speed round 5: embedding backward and GELU ----
// Embedding backward: out[v] = base[v] + sum over n with ids[n] == v of dx[n], summed in increasing n, which is the
// order of candle's index_add (one thread per column looping over every id), so the result is bit-identical; but
// here every vocabulary row is its own block. The block first lists its matching n in order (warp ballots) in
// dynamic shared memory (N ints), then each thread sums its columns.
extern "C" __global__ void embed_bwd(const float* __restrict__ base, const unsigned int* __restrict__ ids,
                                     const float* __restrict__ dx, float* __restrict__ out, int N, int d) {
    extern __shared__ int list[];
    __shared__ int wc[32];
    __shared__ int cnt;
    const int v = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nw = blockDim.x >> 5;
    if (tid == 0) cnt = 0;
    __syncthreads();
    for (int c0 = 0; c0 < N; c0 += blockDim.x) {
        const int n = c0 + tid;
        const bool hit = n < N && ids[n] == (unsigned int)v;
        const unsigned int mask = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) wc[w] = __popc(mask);
        __syncthreads();
        int off = cnt;
        for (int k = 0; k < w; ++k) off += wc[k];
        if (hit) list[off + __popc(mask & ((1u << lane) - 1u))] = n;
        __syncthreads();
        if (tid == 0) {
            int s = 0;
            for (int k = 0; k < nw; ++k) s += wc[k];
            cnt += s;
        }
        __syncthreads();
    }
    const int c_n = cnt;
    for (int c = tid; c < d; c += blockDim.x) {
        float acc = base[(size_t)v * d + c];
        for (int k = 0; k < c_n; ++k) acc += dx[(size_t)list[k] * d + c];
        out[(size_t)v * d + c] = acc;
    }
}

// GELU (tanh form) exactly as candle's: forward 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))); backward with
// candle's formula: 0.5 t + (0.0535161 x^3 + 0.398942 x)(1 - t^2) + 0.5, t = tanh(0.0356774 x^3 + 0.797885 x).
extern "C" __global__ void gelu_fwd(const float* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    const float cube = v * v * v;
    const float alpha = v + 0.044715f * cube;
    y[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * alpha));
}

extern "C" __global__ void gelu_bwd(const float* __restrict__ x, const float* __restrict__ gy, float* __restrict__ gx, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    const float cube = powf(v, 3.0f);
    const float t = tanhf(0.0356774f * cube + 0.797885f * v);
    const float g = 0.5f * t + (0.0535161f * cube + 0.398942f * v) * (1.0f - powf(t, 2.0f)) + 0.5f;
    gx[i] = gy[i] * g;
}

// ---- speed round 7: bf16 ----
// f32 -> bf16 (f2bf above), 4 values per thread.
extern "C" __global__ void cast_bf16(const float* __restrict__ x, unsigned short* __restrict__ y, int n) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        if (i + j < n) y[i + j] = f2bf(x[i + j]);
}
// a + b (f32) and its bf16 copy (the residual gradient after a block's backward, which feeds two bf16 matmuls).
extern "C" __global__ void add_cast_bf16(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y,
                                         unsigned short* __restrict__ yb, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = a[i] + b[i];
    y[i] = v;
    yb[i] = f2bf(v);
}

// ---- speed round 7b: kernels that write their matmul operand in bf16 directly (same f32 value, then f2bf) ----

// lif_bwd with dL/dI written as bf16 to gb [B, T, D] (it only feeds bf16 matmuls); scratch [B, T+1, D] f32 holds the
// recomputed membrane and, in row T, this batch row's share of dL/dbeta, as lif_bwd's out does.
extern "C" __global__ void lif_bwd_b(const float* __restrict__ cur, const float* __restrict__ beta, const float* __restrict__ gs,
                                     float* __restrict__ scratch, unsigned short* __restrict__ gb, int B, int T, int D, float theta,
                                     float slope) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * D) return;
    int b = i / D, c = i - b * D;
    const float bt = beta[c];
    const float* x = cur + (size_t)b * T * D + c;
    const float* g = gs + (size_t)b * T * D + c;
    float* o = scratch + (size_t)b * (T + 1) * D + c;
    unsigned short* ob = gb + (size_t)b * T * D + c;
    // v40: both passes keep the next LIF_G steps' inputs in flight (see lif_scan_fwd); the reverse pass works out a
    // group's quotients (independent of the recurrence) before its chain. The same operations as v39, with the
    // contractions nvcc chose there spelled out: a = fma(|v - theta|, slope, 1), gv = fma(beta, gv', gs / (a a)),
    // gbeta = fma(gv, v_{t-1}, gbeta), in the same order.
    lif_scan_fwd(x, T, D, bt, theta, [&](int t, float v, float) { o[(size_t)t * D] = v; });
    float gv_next = 0.0f, gbeta = 0.0f;
    // the group from t1 down: va[u] = v_{t1-u}, ga[u] = gs_{t1-u} (indices clamped at 0); vb, gn: the next group's
    float va[LIF_G], ga[LIF_G];
#pragma unroll
    for (int u = 0; u < LIF_G; ++u) {
        const size_t t = (size_t)max(T - 1 - u, 0);
        va[u] = o[t * D];
        ga[u] = g[t * D];
    }
    int t1 = T - 1;
    for (; t1 + 1 >= LIF_G; t1 -= LIF_G) {
        float vb[LIF_G], gn[LIF_G], q[LIF_G];
#pragma unroll
        for (int u = 0; u < LIF_G; ++u) {
            const size_t t = (size_t)max(t1 - LIF_G - u, 0);
            vb[u] = o[t * D];
            gn[u] = g[t * D];
        }
#pragma unroll
        for (int u = 0; u < LIF_G; ++u) {
            const float a = __fmaf_rn(fabsf(__fsub_rn(va[u], theta)), slope, 1.0f);
            q[u] = __fdiv_rn(ga[u], __fmul_rn(a, a));
        }
#pragma unroll
        for (int u = 0; u < LIF_G; ++u) {
            const int t = t1 - u;
            const float gv = __fmaf_rn(bt, gv_next, q[u]);
            if (t > 0) gbeta = __fmaf_rn(gv, u + 1 < LIF_G ? va[u + 1] : vb[0], gbeta);
            ob[(size_t)t * D] = f2bf(gv);
            gv_next = gv;
        }
#pragma unroll
        for (int u = 0; u < LIF_G; ++u) {
            va[u] = vb[u];
            ga[u] = gn[u];
        }
    }
#pragma unroll
    for (int u = 0; u < LIF_G - 1; ++u) {
        const int t = t1 - u;
        if (t >= 0) {
            const float a = __fmaf_rn(fabsf(__fsub_rn(va[u], theta)), slope, 1.0f);
            const float gv = __fmaf_rn(bt, gv_next, __fdiv_rn(ga[u], __fmul_rn(a, a)));
            if (t > 0) gbeta = __fmaf_rn(gv, va[u + 1], gbeta);
            ob[(size_t)t * D] = f2bf(gv);
            gv_next = gv;
        }
    }
    o[(size_t)T * D] = gbeta;
}

// Backward of kwta_relu from the bf16 activation: gu = ga where a > 0 (bf16 sign bit clear and nonzero), as bf16.
extern "C" __global__ void relu_mask_bwd_b(const unsigned short* __restrict__ a, const float* __restrict__ ga, unsigned short* __restrict__ gu,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const unsigned short ai = a[i];
        gu[i] = ((ai & 0x8000u) == 0u && ai != 0u) ? f2bf(ga[i]) : (unsigned short)0;
    }
}

// ce_bwd writing bf16.
extern "C" __global__ void ce_bwd_b(const float* __restrict__ x, const unsigned int* __restrict__ tg, const float* __restrict__ aux,
                                    unsigned short* __restrict__ gx, int N, int V) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const unsigned int t = tg[row];
    const float g = (t < (unsigned int)V) ? aux[2 * row] : 0.0f, lse = aux[2 * row + 1];
    const float* r = x + (size_t)row * V;
    unsigned short* o = gx + (size_t)row * V;
    for (int i = threadIdx.x; i < V; i += blockDim.x) o[i] = f2bf((expf(r[i] - lse) - (i == (int)t ? 1.0f : 0.0f)) * g);
}

// GELU forward / backward writing bf16 (same formulas as gelu_fwd / gelu_bwd).
extern "C" __global__ void gelu_fwd_b(const float* __restrict__ x, unsigned short* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    const float cube = v * v * v;
    const float alpha = v + 0.044715f * cube;
    y[i] = f2bf(0.5f * v * (1.0f + tanhf(0.7978845608028654f * alpha)));
}
extern "C" __global__ void gelu_bwd_b(const float* __restrict__ x, const float* __restrict__ gy, unsigned short* __restrict__ gx, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    const float cube = powf(v, 3.0f);
    const float t = tanhf(0.0356774f * cube + 0.797885f * v);
    const float g = 0.5f * t + (0.0535161f * cube + 0.398942f * v) * (1.0f - powf(t, 2.0f)) + 0.5f;
    gx[i] = f2bf(gy[i] * g);
}

// Causal attention reading q, k, v in place from the fused projection [B*T, 3*D] (head h of q at column h*32, of k
// at D + h*32, of v at 2D + h*32; row stride ld = 3D) and writing o [B*T, D] (f32 and bf16) and lse [B*H, T]:
// no head transposes. Same arithmetic as attn_fwd / attn_bwd_q / attn_bwd_kv (only the addressing differs).
__device__ __forceinline__ void stage_ld(float (*dst)[AHS + 1], const float* __restrict__ src, int r0, int T, int ld) {
    for (int e = threadIdx.x; e < AT * AHS; e += AW * 32) {
        const int r = e / AHS, d = e % AHS;
        dst[r][d] = (r0 + r < T) ? src[(size_t)(r0 + r) * ld + d] : 0.0f;
    }
}
__device__ __forceinline__ void stage_ld32(float (*dst)[AHS], const float* __restrict__ src, int r0, int T, int ld) {
    for (int e = threadIdx.x; e < AT * AHS; e += AW * 32) {
        const int r = e / AHS, d = e % AHS;
        dst[r][d] = (r0 + r < T) ? src[(size_t)(r0 + r) * ld + d] : 0.0f;
    }
}

extern "C" __global__ void __launch_bounds__(128) attn_fwd_s(const float* __restrict__ qkv, float* __restrict__ o,
                                    unsigned short* __restrict__ ob, float* __restrict__ lse, int H, int T, int D, float scale) {
    __shared__ float Qs[AT][AHS];
    __shared__ float Ks[AT][AHS + 1];
    __shared__ float Vs[AT][AHS];
    __shared__ float Ps[AW][ARPW][AT];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.y, i0 = blockIdx.x * AT;
    const int b = bh / H, h = bh % H, ld = 3 * D;
    const float* qb = qkv + (size_t)b * T * ld + h * AHS;
    const float* kb = qb + D;
    const float* vb = qb + 2 * D;
    stage_ld32(Qs, qb, i0, T, ld);
    float m[ARPW], l[ARPW], acc[ARPW];
#pragma unroll
    for (int r = 0; r < ARPW; ++r) { m[r] = -INFINITY; l[r] = 0.0f; acc[r] = 0.0f; }
    for (int j0 = 0; j0 <= i0; j0 += AT) {
        __syncthreads();
        stage_ld(Ks, kb, j0, T, ld);
        stage_ld32(Vs, vb, j0, T, ld);
        __syncthreads();
        float kr[AHS];
#pragma unroll
        for (int d = 0; d < AHS; ++d) kr[d] = Ks[lane][d];
        const int j = j0 + lane;
#pragma unroll
        for (int r = 0; r < ARPW; ++r) {
            const int rl = w + AW * r, i = i0 + rl;
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < AHS; ++d) s += Qs[rl][d] * kr[d];
            const bool ok = j <= i && j < T;
            s = ok ? s * scale : -INFINITY;
            const float nm = fmaxf(m[r], wmax(s));
            const float p = ok ? expf(s - nm) : 0.0f;
            const float corr = expf(m[r] - nm);
            l[r] = l[r] * corr + wsum(p);
            acc[r] *= corr;
            m[r] = nm;
            Ps[w][r][lane] = p;
        }
        __syncwarp();
        for (int jj = 0; jj < AT; ++jj) {
            const float vv = Vs[jj][lane];
#pragma unroll
            for (int r = 0; r < ARPW; ++r) acc[r] += Ps[w][r][jj] * vv;
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int i = i0 + w + AW * r;
        if (i < T) {
            const size_t at = ((size_t)b * T + i) * D + h * AHS + lane;
            const float val = acc[r] / l[r];
            o[at] = val;
            ob[at] = f2bf(val);
            if (lane == 0) lse[(size_t)bh * T + i] = m[r] + logf(l[r]);
        }
    }
}

// dQ into the fused gradient dqkv [B*T, 3D] (bf16, columns 0..D), and D_i = sum(dO_i * O_i) into dvec [B*H, T].
// dout [B*T, D] f32.
template <typename OT>
__device__ __forceinline__ void attn_bwd_q_s_impl(const float* __restrict__ qkv, const float* __restrict__ o,
                                      const float* __restrict__ lse, const float* __restrict__ dout, OT* __restrict__ dqkv,
                                      float* __restrict__ dvec, int H, int T, int D, float scale) {
    __shared__ float Qs[AT][AHS];
    __shared__ float DOs[AT][AHS];
    __shared__ float Ks[AT][AHS + 1];
    __shared__ float Vs[AT][AHS + 1];
    __shared__ float Ls[AT], Ds[AT];
    __shared__ float Ss[AW][ARPW][AT];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.y, i0 = blockIdx.x * AT;
    const int b = bh / H, h = bh % H, ld = 3 * D;
    const float* qb = qkv + (size_t)b * T * ld + h * AHS;
    const float* kb = qb + D;
    const float* vb = qb + 2 * D;
    const float* ob = o + (size_t)b * T * D + h * AHS;
    const float* dob = dout + (size_t)b * T * D + h * AHS;
    stage_ld32(Qs, qb, i0, T, ld);
    stage_ld32(DOs, dob, i0, T, D);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int rl = w + AW * r, i = i0 + rl;
        const float Di = wsum(i < T ? DOs[rl][lane] * ob[(size_t)i * D + lane] : 0.0f);
        if (lane == 0) {
            Ds[rl] = Di;
            Ls[rl] = i < T ? lse[(size_t)bh * T + i] : 0.0f;
            if (i < T) dvec[(size_t)bh * T + i] = Di;
        }
    }
    __syncwarp();
    float acc[ARPW];
#pragma unroll
    for (int r = 0; r < ARPW; ++r) acc[r] = 0.0f;
    for (int j0 = 0; j0 <= i0; j0 += AT) {
        __syncthreads();
        stage_ld(Ks, kb, j0, T, ld);
        stage_ld(Vs, vb, j0, T, ld);
        __syncthreads();
        float kr[AHS], vr[AHS];
#pragma unroll
        for (int d = 0; d < AHS; ++d) { kr[d] = Ks[lane][d]; vr[d] = Vs[lane][d]; }
        const int j = j0 + lane;
#pragma unroll
        for (int r = 0; r < ARPW; ++r) {
            const int rl = w + AW * r, i = i0 + rl;
            float s = 0.0f, dp = 0.0f;
#pragma unroll
            for (int d = 0; d < AHS; ++d) { s += Qs[rl][d] * kr[d]; dp += DOs[rl][d] * vr[d]; }
            const bool ok = j <= i && j < T && i < T;
            Ss[w][r][lane] = ok ? expf(s * scale - Ls[rl]) * (dp - Ds[rl]) : 0.0f;
        }
        __syncwarp();
        for (int jj = 0; jj < AT; ++jj) {
            const float kv = Ks[jj][lane];
#pragma unroll
            for (int r = 0; r < ARPW; ++r) acc[r] += Ss[w][r][jj] * kv;
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int i = i0 + w + AW * r;
        if (i < T) put(dqkv, ((size_t)b * T + i) * ld + h * AHS + lane, acc[r] * scale);
    }
}
// v51 fast1: the bf16 store as before, and an f32 store (see attn_bwd_kv_sf)
extern "C" __global__ void __launch_bounds__(128) attn_bwd_q_s(const float* __restrict__ qkv, const float* __restrict__ o,
                                      const float* __restrict__ lse, const float* __restrict__ dout, unsigned short* __restrict__ dqkv,
                                      float* __restrict__ dvec, int H, int T, int D, float scale) {
    attn_bwd_q_s_impl(qkv, o, lse, dout, dqkv, dvec, H, T, D, scale);
}
extern "C" __global__ void __launch_bounds__(128) attn_bwd_q_sf(const float* __restrict__ qkv, const float* __restrict__ o,
                                       const float* __restrict__ lse, const float* __restrict__ dout, float* __restrict__ dqkv,
                                       float* __restrict__ dvec, int H, int T, int D, float scale) {
    attn_bwd_q_s_impl(qkv, o, lse, dout, dqkv, dvec, H, T, D, scale);
}

// dK, dV into dqkv columns D..2D and 2D..3D (bf16).
template <typename OT>
__device__ __forceinline__ void attn_bwd_kv_s_impl(const float* __restrict__ qkv, const float* __restrict__ lse,
                                       const float* __restrict__ dout, const float* __restrict__ dvec, OT* __restrict__ dqkv,
                                       int H, int T, int D, float scale) {
    __shared__ float Ks[AT][AHS];
    __shared__ float Vs[AT][AHS];
    __shared__ float Qs[AT][AHS + 1];
    __shared__ float DOs[AT][AHS + 1];
    __shared__ float Ls[AT], Ds[AT];
    __shared__ float Ps[AW][ARPW][AT];
    __shared__ float DSs[AW][ARPW][AT];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.y, j0 = blockIdx.x * AT;
    const int b = bh / H, h = bh % H, ld = 3 * D;
    const float* qb = qkv + (size_t)b * T * ld + h * AHS;
    const float* kb = qb + D;
    const float* vb = qb + 2 * D;
    const float* dob = dout + (size_t)b * T * D + h * AHS;
    stage_ld32(Ks, kb, j0, T, ld);
    stage_ld32(Vs, vb, j0, T, ld);
    float gk[ARPW], gv[ARPW];
#pragma unroll
    for (int r = 0; r < ARPW; ++r) { gk[r] = 0.0f; gv[r] = 0.0f; }
    for (int i0 = j0; i0 < T; i0 += AT) {
        __syncthreads();
        stage_ld(Qs, qb, i0, T, ld);
        stage_ld(DOs, dob, i0, T, D);
        if (threadIdx.x < AT) {
            const int i = i0 + threadIdx.x;
            Ls[threadIdx.x] = i < T ? lse[(size_t)bh * T + i] : 0.0f;
            Ds[threadIdx.x] = i < T ? dvec[(size_t)bh * T + i] : 0.0f;
        }
        __syncthreads();
        float qr[AHS], dor[AHS];
#pragma unroll
        for (int d = 0; d < AHS; ++d) { qr[d] = Qs[lane][d]; dor[d] = DOs[lane][d]; }
        const int i = i0 + lane;
        const float Li = Ls[lane], Di = Ds[lane];
#pragma unroll
        for (int r = 0; r < ARPW; ++r) {
            const int rl = w + AW * r, j = j0 + rl;
            float s = 0.0f, dp = 0.0f;
#pragma unroll
            for (int d = 0; d < AHS; ++d) { s += qr[d] * Ks[rl][d]; dp += dor[d] * Vs[rl][d]; }
            const bool ok = i >= j && i < T && j < T;
            const float p = ok ? expf(s * scale - Li) : 0.0f;
            Ps[w][r][lane] = p;
            DSs[w][r][lane] = p * (dp - Di);
        }
        __syncwarp();
        for (int ii = 0; ii < AT; ++ii) {
            const float dov = DOs[ii][lane], qv = Qs[ii][lane];
#pragma unroll
            for (int r = 0; r < ARPW; ++r) { gv[r] += Ps[w][r][ii] * dov; gk[r] += DSs[w][r][ii] * qv; }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int j = j0 + w + AW * r;
        if (j < T) {
            const size_t row = ((size_t)b * T + j) * ld + h * AHS + lane;
            put(dqkv, row + D, gk[r] * scale);
            put(dqkv, row + 2 * D, gv[r]);
        }
    }
}
// v51 fast1: the bf16 store as before, and an f32 store for the f32 training step (its engram gradient lost ~1e-3 to
// the bf16 store; runs/v51/prereg_v51.md Amendment F-1)
extern "C" __global__ void __launch_bounds__(128) attn_bwd_kv_s(const float* __restrict__ qkv, const float* __restrict__ lse,
                                       const float* __restrict__ dout, const float* __restrict__ dvec, unsigned short* __restrict__ dqkv,
                                       int H, int T, int D, float scale) {
    attn_bwd_kv_s_impl(qkv, lse, dout, dvec, dqkv, H, T, D, scale);
}
extern "C" __global__ void __launch_bounds__(128) attn_bwd_kv_sf(const float* __restrict__ qkv, const float* __restrict__ lse,
                                        const float* __restrict__ dout, const float* __restrict__ dvec, float* __restrict__ dqkv,
                                        int H, int T, int D, float scale) {
    attn_bwd_kv_s_impl(qkv, lse, dout, dvec, dqkv, H, T, D, scale);
}

// ---- speed round 9: causal attention on the tensor cores (heads of 32, bf16 inputs, f32 accumulate) ----
// FlashAttention-2 layout: 64 query rows per block (4 warps x 16), key tiles of 64, online softmax in registers,
// mma.sync m16n8k16 bf16. q/k/v are read in place from the fused projection qkv [B*T, 3D] (bf16): head h of q at
// column h*32, k at D + h*32, v at 2D + h*32. lse is natural-log (the same meaning as attn_fwd_s's).
// Accumulator layout (m16n8, f32): c0, c1 = row g, cols 2t, 2t+1; c2, c3 = row g + 8 (g = lane/4, t = lane%4).
#define TCB 64
#define TKP 40 // smem row stride (bf16) of a [64][32] tile: 80 bytes, conflict-free ldmatrix rows

__device__ __forceinline__ void tc_mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void tc_ldsm4(unsigned* r, const unsigned short* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void tc_ldsm4t(unsigned* r, const unsigned short* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ unsigned tc_pack(float lo, float hi) { return (unsigned)f2bf(lo) | ((unsigned)f2bf(hi) << 16); }
// A fragment (16 rows from r0, k-slab k0..k0+15) of a row-major [rows][k] smem tile.
__device__ __forceinline__ void tc_lda(unsigned* a, const unsigned short* S, int r0, int k0) {
    const int L = threadIdx.x & 31;
    tc_ldsm4(a, S + (r0 + (L & 7) + ((L >> 3) & 1) * 8) * TKP + k0 + (L >> 4) * 8);
}
// B fragments for n-tiles n0 and n0 + 8, k-slab k0, from a smem tile stored [n][k] (rows = n).
__device__ __forceinline__ void tc_ldb_nk(unsigned* b, const unsigned short* S, int n0, int k0) {
    const int L = threadIdx.x & 31;
    tc_ldsm4(b, S + (n0 + (L & 7) + (L >> 4) * 8) * TKP + k0 + ((L >> 3) & 1) * 8);
}
// B fragments for n-tiles n0 and n0 + 8, k-slab k0, from a smem tile stored [k][n] (rows = k).
__device__ __forceinline__ void tc_ldb_kn(unsigned* b, const unsigned short* S, int k0, int n0) {
    const int L = threadIdx.x & 31;
    tc_ldsm4t(b, S + (k0 + (L & 7) + ((L >> 3) & 1) * 8) * TKP + n0 + (L >> 4) * 8);
}
// 64 rows x 32 bf16 from a row-major source with row stride ld (rows >= T read as zeros) into S[64][TKP].
__device__ __forceinline__ void tc_tile(unsigned short* S, const unsigned short* __restrict__ src, int r0, int T, int ld) {
    for (int v = threadIdx.x; v < 256; v += blockDim.x) {
        const int r = v >> 2, c = (v & 3) * 8;
        uint4 x = make_uint4(0u, 0u, 0u, 0u);
        if (r0 + r < T) x = __ldg((const uint4*)(src + (size_t)(r0 + r) * ld + c));
        *(uint4*)(S + r * TKP + c) = x;
    }
}
// Row-reduce over the 4 lanes that share a row.
__device__ __forceinline__ float tc_rmax(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float tc_rsum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// Forward: ob [B*T, D] (bf16) and lse [B*H, T]. Grid (ceil(T/64), B*H), 128 threads.
extern "C" __global__ void __launch_bounds__(128) attn_fwd_tc(const unsigned short* __restrict__ qkv, unsigned short* __restrict__ ob,
                                                              float* __restrict__ lse, int H, int T, int D, float scale) {
    __shared__ __align__(16) unsigned short Qs[TCB * TKP];
    __shared__ __align__(16) unsigned short Ks[TCB * TKP];
    __shared__ __align__(16) unsigned short Vs[TCB * TKP];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    const int bh = blockIdx.y, b = bh / H, h = bh % H, ld = 3 * D;
    const int qt = gridDim.x - 1 - blockIdx.x, i0 = qt * TCB; // longest query tiles first
    const unsigned short* base = qkv + (size_t)b * T * ld + h * 32;
    tc_tile(Qs, base, i0, T, ld);
    __syncthreads();
    unsigned qa[2][4];
    tc_lda(qa[0], Qs, 16 * w, 0);
    tc_lda(qa[1], Qs, 16 * w, 16);
    float o[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float c = scale * 1.4426950408889634f; // scores in log2 units
    const int r0 = i0 + 16 * w + g, r1 = r0 + 8;
    for (int kt = 0; kt <= qt; ++kt) {
        const int j0 = kt * TCB;
        __syncthreads();
        tc_tile(Ks, base + D, j0, T, ld);
        tc_tile(Vs, base + 2 * D, j0, T, ld);
        __syncthreads();
        float s[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                unsigned bb[4];
                tc_ldb_nk(bb, Ks, 16 * np, 16 * kk);
                tc_mma(s[2 * np], qa[kk], bb[0], bb[1]);
                tc_mma(s[2 * np + 1], qa[kk], bb[2], bb[3]);
            }
        }
        const bool edge = kt == qt;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const int col = j0 + 8 * n + 2 * tq;
            if (edge) {
                s[n][0] = (col <= r0 && col < T) ? s[n][0] * c : -INFINITY;
                s[n][1] = (col + 1 <= r0 && col + 1 < T) ? s[n][1] * c : -INFINITY;
                s[n][2] = (col <= r1 && col < T) ? s[n][2] * c : -INFINITY;
                s[n][3] = (col + 1 <= r1 && col + 1 < T) ? s[n][3] * c : -INFINITY;
            } else {
                s[n][0] *= c; s[n][1] *= c; s[n][2] *= c; s[n][3] *= c;
            }
        }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
            mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
        }
        mx0 = tc_rmax(mx0);
        mx1 = tc_rmax(mx1);
        const float b0 = mx0 == -INFINITY ? 0.0f : mx0, b1 = mx1 == -INFINITY ? 0.0f : mx1;
        const float k0 = exp2f(m0 - b0), k1 = exp2f(m1 - b1);
        m0 = mx0;
        m1 = mx1;
        l0 *= k0;
        l1 *= k1;
#pragma unroll
        for (int n = 0; n < 4; ++n) { o[n][0] *= k0; o[n][1] *= k0; o[n][2] *= k1; o[n][3] *= k1; }
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            s[n][0] = exp2f(s[n][0] - b0); s[n][1] = exp2f(s[n][1] - b0);
            s[n][2] = exp2f(s[n][2] - b1); s[n][3] = exp2f(s[n][3] - b1);
            l0 += s[n][0] + s[n][1];
            l1 += s[n][2] + s[n][3];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned pa[4] = {tc_pack(s[2 * j][0], s[2 * j][1]), tc_pack(s[2 * j][2], s[2 * j][3]),
                                    tc_pack(s[2 * j + 1][0], s[2 * j + 1][1]), tc_pack(s[2 * j + 1][2], s[2 * j + 1][3])};
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned bb[4];
                tc_ldb_kn(bb, Vs, 16 * j, 16 * np);
                tc_mma(o[2 * np], pa, bb[0], bb[1]);
                tc_mma(o[2 * np + 1], pa, bb[2], bb[3]);
            }
        }
    }
    l0 = tc_rsum(l0);
    l1 = tc_rsum(l1);
    const float i0v = 1.0f / l0, i1v = 1.0f / l1;
    if (r0 < T) {
        unsigned short* dst = ob + ((size_t)b * T + r0) * D + h * 32 + 2 * tq;
#pragma unroll
        for (int n = 0; n < 4; ++n) *(unsigned*)(dst + 8 * n) = tc_pack(o[n][0] * i0v, o[n][1] * i0v);
        if (tq == 0) lse[(size_t)bh * T + r0] = (m0 + log2f(l0)) * 0.6931471805599453f;
    }
    if (r1 < T) {
        unsigned short* dst = ob + ((size_t)b * T + r1) * D + h * 32 + 2 * tq;
#pragma unroll
        for (int n = 0; n < 4; ++n) *(unsigned*)(dst + 8 * n) = tc_pack(o[n][2] * i1v, o[n][3] * i1v);
        if (tq == 0) lse[(size_t)bh * T + r1] = (m1 + log2f(l1)) * 0.6931471805599453f;
    }
}

// Backward, part 1: dq (into dqkv's q columns, bf16) and dvec = rowsum(dout * out) [B*H, T] for part 2.
// dob = dL/d(attention output) [B*T, D] (bf16), ob = the forward's output (bf16). Grid (ceil(T/64), B*H), 128 threads.
extern "C" __global__ void __launch_bounds__(128) attn_bwd_q_tc(const unsigned short* __restrict__ qkv, const unsigned short* __restrict__ ob,
                                                                const float* __restrict__ lse, const unsigned short* __restrict__ dob,
                                                                unsigned short* __restrict__ dqkv, float* __restrict__ dvec, int H, int T, int D,
                                                                float scale) {
    __shared__ __align__(16) unsigned short Qs[TCB * TKP];
    __shared__ __align__(16) unsigned short Ds[TCB * TKP];
    __shared__ __align__(16) unsigned short Ks[TCB * TKP];
    __shared__ __align__(16) unsigned short Vs[TCB * TKP];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    const int bh = blockIdx.y, b = bh / H, h = bh % H, ld = 3 * D;
    const int qt = gridDim.x - 1 - blockIdx.x, i0 = qt * TCB;
    const unsigned short* base = qkv + (size_t)b * T * ld + h * 32;
    tc_tile(Qs, base, i0, T, ld);
    tc_tile(Ds, dob + (size_t)b * T * D + h * 32, i0, T, D);
    __syncthreads();
    unsigned qa[2][4], da[2][4];
    tc_lda(qa[0], Qs, 16 * w, 0);
    tc_lda(qa[1], Qs, 16 * w, 16);
    tc_lda(da[0], Ds, 16 * w, 0);
    tc_lda(da[1], Ds, 16 * w, 16);
    const int r0 = i0 + 16 * w + g, r1 = r0 + 8;
    // dvec for rows r0, r1: lanes of a row split the 32 dims 8 each
    float dv0 = 0.0f, dv1 = 0.0f;
    {
        const unsigned short* d0 = Ds + (16 * w + g) * TKP + 8 * tq;
        const unsigned short* d1 = d0 + 8 * TKP;
        uint4 o0 = make_uint4(0u, 0u, 0u, 0u), o1 = o0;
        if (r0 < T) o0 = __ldg((const uint4*)(ob + ((size_t)b * T + r0) * D + h * 32 + 8 * tq));
        if (r1 < T) o1 = __ldg((const uint4*)(ob + ((size_t)b * T + r1) * D + h * 32 + 8 * tq));
        const unsigned short* p0 = (const unsigned short*)&o0;
        const unsigned short* p1 = (const unsigned short*)&o1;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            dv0 += __uint_as_float((unsigned)d0[e] << 16) * __uint_as_float((unsigned)p0[e] << 16);
            dv1 += __uint_as_float((unsigned)d1[e] << 16) * __uint_as_float((unsigned)p1[e] << 16);
        }
        dv0 = tc_rsum(dv0);
        dv1 = tc_rsum(dv1);
        if (tq == 0) {
            if (r0 < T) dvec[(size_t)bh * T + r0] = dv0;
            if (r1 < T) dvec[(size_t)bh * T + r1] = dv1;
        }
    }
    const float L0 = r0 < T ? lse[(size_t)bh * T + r0] * 1.4426950408889634f : 0.0f;
    const float L1 = r1 < T ? lse[(size_t)bh * T + r1] * 1.4426950408889634f : 0.0f;
    const float c = scale * 1.4426950408889634f;
    float dq[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) dq[n][0] = dq[n][1] = dq[n][2] = dq[n][3] = 0.0f;
    for (int kt = 0; kt <= qt; ++kt) {
        const int j0 = kt * TCB;
        __syncthreads();
        tc_tile(Ks, base + D, j0, T, ld);
        tc_tile(Vs, base + 2 * D, j0, T, ld);
        __syncthreads();
        float s[8][4], dp[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            dp[n][0] = dp[n][1] = dp[n][2] = dp[n][3] = 0.0f;
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                unsigned bb[4];
                tc_ldb_nk(bb, Ks, 16 * np, 16 * kk);
                tc_mma(s[2 * np], qa[kk], bb[0], bb[1]);
                tc_mma(s[2 * np + 1], qa[kk], bb[2], bb[3]);
                tc_ldb_nk(bb, Vs, 16 * np, 16 * kk);
                tc_mma(dp[2 * np], da[kk], bb[0], bb[1]);
                tc_mma(dp[2 * np + 1], da[kk], bb[2], bb[3]);
            }
        }
        const bool edge = kt == qt;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const int col = j0 + 8 * n + 2 * tq;
            float p0 = exp2f(s[n][0] * c - L0), p1 = exp2f(s[n][1] * c - L0);
            float p2 = exp2f(s[n][2] * c - L1), p3 = exp2f(s[n][3] * c - L1);
            if (edge) {
                if (!(col <= r0 && col < T)) p0 = 0.0f;
                if (!(col + 1 <= r0 && col + 1 < T)) p1 = 0.0f;
                if (!(col <= r1 && col < T)) p2 = 0.0f;
                if (!(col + 1 <= r1 && col + 1 < T)) p3 = 0.0f;
            }
            s[n][0] = p0 * (dp[n][0] - dv0);
            s[n][1] = p1 * (dp[n][1] - dv0);
            s[n][2] = p2 * (dp[n][2] - dv1);
            s[n][3] = p3 * (dp[n][3] - dv1);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned pa[4] = {tc_pack(s[2 * j][0], s[2 * j][1]), tc_pack(s[2 * j][2], s[2 * j][3]),
                                    tc_pack(s[2 * j + 1][0], s[2 * j + 1][1]), tc_pack(s[2 * j + 1][2], s[2 * j + 1][3])};
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned bb[4];
                tc_ldb_kn(bb, Ks, 16 * j, 16 * np);
                tc_mma(dq[2 * np], pa, bb[0], bb[1]);
                tc_mma(dq[2 * np + 1], pa, bb[2], bb[3]);
            }
        }
    }
    if (r0 < T) {
        unsigned short* dst = dqkv + ((size_t)b * T + r0) * ld + h * 32 + 2 * tq;
#pragma unroll
        for (int n = 0; n < 4; ++n) *(unsigned*)(dst + 8 * n) = tc_pack(dq[n][0] * scale, dq[n][1] * scale);
    }
    if (r1 < T) {
        unsigned short* dst = dqkv + ((size_t)b * T + r1) * ld + h * 32 + 2 * tq;
#pragma unroll
        for (int n = 0; n < 4; ++n) *(unsigned*)(dst + 8 * n) = tc_pack(dq[n][2] * scale, dq[n][3] * scale);
    }
}

// Backward, part 2: dk and dv (into dqkv's k and v columns). Each warp owns 16 keys; the block walks the query
// tiles from its own diagonal to the end. Needs part 1's dvec. Grid (ceil(T/64), B*H), 128 threads.
extern "C" __global__ void __launch_bounds__(128) attn_bwd_kv_tc(const unsigned short* __restrict__ qkv, const float* __restrict__ lse,
                                                                 const unsigned short* __restrict__ dob, const float* __restrict__ dvec,
                                                                 unsigned short* __restrict__ dqkv, int H, int T, int D, float scale) {
    __shared__ __align__(16) unsigned short Ks[TCB * TKP];
    __shared__ __align__(16) unsigned short Vs[TCB * TKP];
    __shared__ __align__(16) unsigned short Qs[TCB * TKP];
    __shared__ __align__(16) unsigned short Ds[TCB * TKP];
    __shared__ float Ls[TCB], Dv[TCB];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    const int bh = blockIdx.y, b = bh / H, h = bh % H, ld = 3 * D;
    const int kt = blockIdx.x, j0 = kt * TCB; // the longest walks (first key tiles) start first
    const int nq = gridDim.x;
    const unsigned short* base = qkv + (size_t)b * T * ld + h * 32;
    tc_tile(Ks, base + D, j0, T, ld);
    tc_tile(Vs, base + 2 * D, j0, T, ld);
    __syncthreads();
    unsigned ka[2][4], va[2][4];
    tc_lda(ka[0], Ks, 16 * w, 0);
    tc_lda(ka[1], Ks, 16 * w, 16);
    tc_lda(va[0], Vs, 16 * w, 0);
    tc_lda(va[1], Vs, 16 * w, 16);
    const int key0 = j0 + 16 * w + g, key1 = key0 + 8;
    const float c = scale * 1.4426950408889634f;
    float dk[4][4], dv[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        dk[n][0] = dk[n][1] = dk[n][2] = dk[n][3] = 0.0f;
        dv[n][0] = dv[n][1] = dv[n][2] = dv[n][3] = 0.0f;
    }
    for (int qt = kt; qt < nq; ++qt) {
        const int i0 = qt * TCB;
        __syncthreads();
        tc_tile(Qs, base, i0, T, ld);
        tc_tile(Ds, dob + (size_t)b * T * D + h * 32, i0, T, D);
        if (threadIdx.x < TCB) {
            const int i = i0 + threadIdx.x;
            Ls[threadIdx.x] = i < T ? lse[(size_t)bh * T + i] * 1.4426950408889634f : 0.0f;
            Dv[threadIdx.x] = i < T ? dvec[(size_t)bh * T + i] : 0.0f;
        }
        __syncthreads();
        float st[8][4], dpt[8][4]; // [key][query]: rows key0/key1, cols i0 + 8n + 2t (+1)
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            st[n][0] = st[n][1] = st[n][2] = st[n][3] = 0.0f;
            dpt[n][0] = dpt[n][1] = dpt[n][2] = dpt[n][3] = 0.0f;
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                unsigned bb[4];
                tc_ldb_nk(bb, Qs, 16 * np, 16 * kk);
                tc_mma(st[2 * np], ka[kk], bb[0], bb[1]);
                tc_mma(st[2 * np + 1], ka[kk], bb[2], bb[3]);
                tc_ldb_nk(bb, Ds, 16 * np, 16 * kk);
                tc_mma(dpt[2 * np], va[kk], bb[0], bb[1]);
                tc_mma(dpt[2 * np + 1], va[kk], bb[2], bb[3]);
            }
        }
        const bool edge = qt == kt || i0 + TCB > T;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const int cl = 8 * n + 2 * tq, col = i0 + cl;
            float p0 = exp2f(st[n][0] * c - Ls[cl]), p1 = exp2f(st[n][1] * c - Ls[cl + 1]);
            float p2 = exp2f(st[n][2] * c - Ls[cl]), p3 = exp2f(st[n][3] * c - Ls[cl + 1]);
            if (edge) {
                if (!(col >= key0 && col < T)) p0 = 0.0f;
                if (!(col + 1 >= key0 && col + 1 < T)) p1 = 0.0f;
                if (!(col >= key1 && col < T)) p2 = 0.0f;
                if (!(col + 1 >= key1 && col + 1 < T)) p3 = 0.0f;
            }
            st[n][0] = p0; st[n][1] = p1; st[n][2] = p2; st[n][3] = p3;
            dpt[n][0] = p0 * (dpt[n][0] - Dv[cl]);
            dpt[n][1] = p1 * (dpt[n][1] - Dv[cl + 1]);
            dpt[n][2] = p2 * (dpt[n][2] - Dv[cl]);
            dpt[n][3] = p3 * (dpt[n][3] - Dv[cl + 1]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned pa[4] = {tc_pack(st[2 * j][0], st[2 * j][1]), tc_pack(st[2 * j][2], st[2 * j][3]),
                                    tc_pack(st[2 * j + 1][0], st[2 * j + 1][1]), tc_pack(st[2 * j + 1][2], st[2 * j + 1][3])};
            const unsigned sa[4] = {tc_pack(dpt[2 * j][0], dpt[2 * j][1]), tc_pack(dpt[2 * j][2], dpt[2 * j][3]),
                                    tc_pack(dpt[2 * j + 1][0], dpt[2 * j + 1][1]), tc_pack(dpt[2 * j + 1][2], dpt[2 * j + 1][3])};
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned bb[4];
                tc_ldb_kn(bb, Ds, 16 * j, 16 * np);
                tc_mma(dv[2 * np], pa, bb[0], bb[1]);
                tc_mma(dv[2 * np + 1], pa, bb[2], bb[3]);
                tc_ldb_kn(bb, Qs, 16 * j, 16 * np);
                tc_mma(dk[2 * np], sa, bb[0], bb[1]);
                tc_mma(dk[2 * np + 1], sa, bb[2], bb[3]);
            }
        }
    }
    if (key0 < T) {
        unsigned short* dst = dqkv + ((size_t)b * T + key0) * ld + h * 32 + 2 * tq;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            *(unsigned*)(dst + D + 8 * n) = tc_pack(dk[n][0] * scale, dk[n][1] * scale);
            *(unsigned*)(dst + 2 * D + 8 * n) = tc_pack(dv[n][0], dv[n][1]);
        }
    }
    if (key1 < T) {
        unsigned short* dst = dqkv + ((size_t)b * T + key1) * ld + h * 32 + 2 * tq;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            *(unsigned*)(dst + D + 8 * n) = tc_pack(dk[n][2] * scale, dk[n][3] * scale);
            *(unsigned*)(dst + 2 * D + 8 * n) = tc_pack(dv[n][2], dv[n][3]);
        }
    }
}

// ---- speed round 10: R's fast-weight read (decayed linear attention), f32 ----
// y_i = sum_{j<=i} scale (q_i . k_j) exp(c_i - c_j) v_j, with q, k [BH, T, 32] (R's k already multiplied by its
// write strength beta) and vc [BH, T, 33] = v then c (the running sum of log alpha, so c_i - c_j <= 0 for j <= i).
// The same tiling as attn_fwd with no softmax. Replaces R's chain of [T, T] tensors (scores, decay matrix, products)
// and their backward. Backward: g_ij = dy_i . v_j, E_ij = exp(c_i - c_j), w_ij = scale (q_i . k_j) E_ij;
// dv_j = sum_i w_ij dy_i; dq_i = scale sum_j g_ij E_ij k_j; dk_j = scale sum_i g_ij E_ij q_i;
// dc_i = sum_j g_ij w_ij (as the later token) - sum_i' g_i'i w_i'i (as the earlier one). Fixed order, no atomics.
#define GVC 33
extern "C" __global__ void __launch_bounds__(128) gla_fwd(const float* __restrict__ q, const float* __restrict__ k,
                                                          const float* __restrict__ vc, float* __restrict__ y, int BH, int T, float scale) {
    __shared__ float Qs[AT][AHS];
    __shared__ float Ks[AT][AHS + 1];
    __shared__ float Vs[AT][AHS];
    __shared__ float Ck[AT], Cq[AT];
    __shared__ float Ps[AW][ARPW][AT];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.y, i0 = blockIdx.x * AT;
    const size_t base = (size_t)bh * T * AHS, basec = (size_t)bh * T * GVC;
    stage_ld32(Qs, q + base, i0, T, AHS);
    if (threadIdx.x < AT) Cq[threadIdx.x] = i0 + (int)threadIdx.x < T ? vc[basec + (size_t)(i0 + threadIdx.x) * GVC + AHS] : 0.0f;
    float acc[ARPW];
#pragma unroll
    for (int r = 0; r < ARPW; ++r) acc[r] = 0.0f;
    for (int j0 = 0; j0 <= i0; j0 += AT) {
        __syncthreads();
        stage_ld(Ks, k + base, j0, T, AHS);
        stage_ld32(Vs, vc + basec, j0, T, GVC);
        if (threadIdx.x < AT) Ck[threadIdx.x] = j0 + (int)threadIdx.x < T ? vc[basec + (size_t)(j0 + threadIdx.x) * GVC + AHS] : 0.0f;
        __syncthreads();
        float kr[AHS];
#pragma unroll
        for (int d = 0; d < AHS; ++d) kr[d] = Ks[lane][d];
        const int j = j0 + lane;
        const float cj = Ck[lane];
#pragma unroll
        for (int r = 0; r < ARPW; ++r) {
            const int rl = w + AW * r, i = i0 + rl;
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < AHS; ++d) s += Qs[rl][d] * kr[d];
            const bool ok = j <= i && j < T;
            Ps[w][r][lane] = ok ? s * scale * expf(Cq[rl] - cj) : 0.0f;
        }
        __syncwarp();
        for (int jj = 0; jj < AT; ++jj) {
            const float vv = Vs[jj][lane];
#pragma unroll
            for (int r = 0; r < ARPW; ++r) acc[r] += Ps[w][r][jj] * vv;
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int i = i0 + w + AW * r;
        if (i < T) y[base + (size_t)i * AHS + lane] = acc[r];
    }
}

// Backward 1, a block per 32 query rows: dq and rowp_i = sum_j g_ij w_ij.
extern "C" __global__ void __launch_bounds__(128) gla_bwd_q(const float* __restrict__ q, const float* __restrict__ k,
                                                            const float* __restrict__ vc, const float* __restrict__ dy,
                                                            float* __restrict__ dq, float* __restrict__ rowp, int BH, int T, float scale) {
    __shared__ float Qs[AT][AHS];
    __shared__ float DYs[AT][AHS];
    __shared__ float Ks[AT][AHS + 1];
    __shared__ float Vs[AT][AHS + 1];
    __shared__ float Ck[AT], Cq[AT];
    __shared__ float Ss[AW][ARPW][AT];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.y, i0 = blockIdx.x * AT;
    const size_t base = (size_t)bh * T * AHS, basec = (size_t)bh * T * GVC;
    stage_ld32(Qs, q + base, i0, T, AHS);
    stage_ld32(DYs, dy + base, i0, T, AHS);
    if (threadIdx.x < AT) Cq[threadIdx.x] = i0 + (int)threadIdx.x < T ? vc[basec + (size_t)(i0 + threadIdx.x) * GVC + AHS] : 0.0f;
    float acc[ARPW], rp[ARPW];
#pragma unroll
    for (int r = 0; r < ARPW; ++r) { acc[r] = 0.0f; rp[r] = 0.0f; }
    for (int j0 = 0; j0 <= i0; j0 += AT) {
        __syncthreads();
        stage_ld(Ks, k + base, j0, T, AHS);
        stage_ld(Vs, vc + basec, j0, T, GVC);
        if (threadIdx.x < AT) Ck[threadIdx.x] = j0 + (int)threadIdx.x < T ? vc[basec + (size_t)(j0 + threadIdx.x) * GVC + AHS] : 0.0f;
        __syncthreads();
        float kr[AHS], vr[AHS];
#pragma unroll
        for (int d = 0; d < AHS; ++d) { kr[d] = Ks[lane][d]; vr[d] = Vs[lane][d]; }
        const int j = j0 + lane;
        const float cj = Ck[lane];
#pragma unroll
        for (int r = 0; r < ARPW; ++r) {
            const int rl = w + AW * r, i = i0 + rl;
            float s = 0.0f, dp = 0.0f;
#pragma unroll
            for (int d = 0; d < AHS; ++d) { s += Qs[rl][d] * kr[d]; dp += DYs[rl][d] * vr[d]; }
            const bool ok = j <= i && j < T && i < T;
            const float e = ok ? expf(Cq[rl] - cj) : 0.0f;
            Ss[w][r][lane] = dp * e;
            rp[r] += wsum(dp * s * scale * e);
        }
        __syncwarp();
        for (int jj = 0; jj < AT; ++jj) {
            const float kv = Ks[jj][lane];
#pragma unroll
            for (int r = 0; r < ARPW; ++r) acc[r] += Ss[w][r][jj] * kv;
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int i = i0 + w + AW * r;
        if (i < T) {
            dq[base + (size_t)i * AHS + lane] = acc[r] * scale;
            if (lane == 0) rowp[(size_t)bh * T + i] = rp[r];
        }
    }
}

// Backward 2, a block per 32 key rows: dk, and dvc = (dv, dc) with dc_j = rowp_j - sum_i g_ij w_ij. Needs rowp.
extern "C" __global__ void __launch_bounds__(128) gla_bwd_kv(const float* __restrict__ q, const float* __restrict__ k,
                                                             const float* __restrict__ vc, const float* __restrict__ dy,
                                                             const float* __restrict__ rowp, float* __restrict__ dk,
                                                             float* __restrict__ dvc, int BH, int T, float scale) {
    __shared__ float Ks[AT][AHS];
    __shared__ float Vs[AT][AHS];
    __shared__ float Qs[AT][AHS + 1];
    __shared__ float DYs[AT][AHS + 1];
    __shared__ float Ck[AT], Cq[AT];
    __shared__ float Ps[AW][ARPW][AT];
    __shared__ float DSs[AW][ARPW][AT];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.y, j0 = blockIdx.x * AT;
    const size_t base = (size_t)bh * T * AHS, basec = (size_t)bh * T * GVC;
    stage_ld32(Ks, k + base, j0, T, AHS);
    stage_ld32(Vs, vc + basec, j0, T, GVC);
    if (threadIdx.x < AT) Ck[threadIdx.x] = j0 + (int)threadIdx.x < T ? vc[basec + (size_t)(j0 + threadIdx.x) * GVC + AHS] : 0.0f;
    float gk[ARPW], gv[ARPW], cp[ARPW];
#pragma unroll
    for (int r = 0; r < ARPW; ++r) { gk[r] = 0.0f; gv[r] = 0.0f; cp[r] = 0.0f; }
    for (int i0 = j0; i0 < T; i0 += AT) {
        __syncthreads();
        stage_ld(Qs, q + base, i0, T, AHS);
        stage_ld(DYs, dy + base, i0, T, AHS);
        if (threadIdx.x < AT) Cq[threadIdx.x] = i0 + (int)threadIdx.x < T ? vc[basec + (size_t)(i0 + threadIdx.x) * GVC + AHS] : 0.0f;
        __syncthreads();
        float qr[AHS], dyr[AHS];
#pragma unroll
        for (int d = 0; d < AHS; ++d) { qr[d] = Qs[lane][d]; dyr[d] = DYs[lane][d]; }
        const int i = i0 + lane;
        const float ci = Cq[lane];
#pragma unroll
        for (int r = 0; r < ARPW; ++r) {
            const int rl = w + AW * r, j = j0 + rl;
            float s = 0.0f, dp = 0.0f;
#pragma unroll
            for (int d = 0; d < AHS; ++d) { s += qr[d] * Ks[rl][d]; dp += dyr[d] * Vs[rl][d]; }
            const bool ok = i >= j && i < T && j < T;
            const float e = ok ? expf(ci - Ck[rl]) : 0.0f;
            const float wv = s * scale * e;
            Ps[w][r][lane] = wv;
            DSs[w][r][lane] = dp * e;
            cp[r] += wsum(dp * wv);
        }
        __syncwarp();
        for (int ii = 0; ii < AT; ++ii) {
            const float dyv = DYs[ii][lane], qv = Qs[ii][lane];
#pragma unroll
            for (int r = 0; r < ARPW; ++r) { gv[r] += Ps[w][r][ii] * dyv; gk[r] += DSs[w][r][ii] * qv; }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int j = j0 + w + AW * r;
        if (j < T) {
            dk[base + (size_t)j * AHS + lane] = gk[r] * scale;
            dvc[basec + (size_t)j * GVC + lane] = gv[r];
            if (lane == 0) dvc[basec + (size_t)j * GVC + AHS] = rowp[(size_t)bh * T + j] - cp[r];
        }
    }
}

// ---- speed round 11: D's gated delta rule as a direct recurrence, f32 ----
// Per head, fast weights F [32 x 32] (F_ij, i = key dim, j = value dim): p = F^T k_t, e = v_t - alpha_t p,
// F <- alpha_t F + beta_t k_t e^T  (= alpha (I - beta k k^T) F + beta k v^T),  y_t = F^T q_t.
// q, k [BH, T, 32] (unit length); vg [BH, T, 34] = v, then log alpha, then beta. One warp per head; lane j holds
// column j of F in registers, so p_j and y_j need no cross-lane sums. The same values as the chunked WY form
// (model.rs delta_rule) up to float rounding. Backward: delta_states writes the state before every step (F_{t-1})
// to states [BH, T, 32 (i), 32 (j)], then delta_bwd walks t backwards carrying dF.
#define DVG 34
template <bool WRITE_Y, bool WRITE_S>
__device__ __forceinline__ void delta_scan(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ vg,
                                           float* __restrict__ y, float* __restrict__ states, int BH, int T) {
    __shared__ float Qs[4][32], Ks[4][32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.x * 4 + w;
    if (bh >= BH) return; // the whole warp leaves together (no block barriers below)
    const float* qb = q + (size_t)bh * T * 32;
    const float* kb = k + (size_t)bh * T * 32;
    const float* gb = vg + (size_t)bh * T * DVG;
    float F[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) F[i] = 0.0f;
    float qn = qb[lane], kn = kb[lane], vn = gb[lane], lan = gb[32], ben = gb[33];
    for (int t = 0; t < T; ++t) {
        const float qc = qn, kc = kn, vc = vn, lac = lan, bec = ben;
        if (t + 1 < T) {
            qn = qb[(size_t)(t + 1) * 32 + lane];
            kn = kb[(size_t)(t + 1) * 32 + lane];
            vn = gb[(size_t)(t + 1) * DVG + lane];
            lan = gb[(size_t)(t + 1) * DVG + 32];
            ben = gb[(size_t)(t + 1) * DVG + 33];
        }
        if (WRITE_S) {
            float* s = states + ((size_t)bh * T + t) * 1024 + lane;
#pragma unroll
            for (int i = 0; i < 32; ++i) s[i * 32] = F[i];
        }
        Qs[w][lane] = qc;
        Ks[w][lane] = kc;
        __syncwarp();
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i += 4) {
            p0 += Ks[w][i] * F[i];
            p1 += Ks[w][i + 1] * F[i + 1];
            p2 += Ks[w][i + 2] * F[i + 2];
            p3 += Ks[w][i + 3] * F[i + 3];
        }
        const float a = expf(lac);
        const float be = bec * (vc - a * ((p0 + p1) + (p2 + p3)));
        float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i += 4) {
            F[i] = a * F[i] + Ks[w][i] * be;
            F[i + 1] = a * F[i + 1] + Ks[w][i + 1] * be;
            F[i + 2] = a * F[i + 2] + Ks[w][i + 2] * be;
            F[i + 3] = a * F[i + 3] + Ks[w][i + 3] * be;
            y0 += Qs[w][i] * F[i];
            y1 += Qs[w][i + 1] * F[i + 1];
            y2 += Qs[w][i + 2] * F[i + 2];
            y3 += Qs[w][i + 3] * F[i + 3];
        }
        if (WRITE_Y) y[((size_t)bh * T + t) * 32 + lane] = (y0 + y1) + (y2 + y3);
        __syncwarp();
    }
}
extern "C" __global__ void __launch_bounds__(128) delta_fwd(const float* __restrict__ q, const float* __restrict__ k,
                                                            const float* __restrict__ vg, float* __restrict__ y, int BH, int T) {
    delta_scan<true, false>(q, k, vg, y, nullptr, BH, T);
}
extern "C" __global__ void __launch_bounds__(128) delta_states(const float* __restrict__ q, const float* __restrict__ k,
                                                               const float* __restrict__ vg, float* __restrict__ states, int BH, int T) {
    delta_scan<false, true>(q, k, vg, nullptr, states, BH, T);
}

// Backward: dq, dk [BH, T, 32] and dvg [BH, T, 34] = (dv, d log alpha, d beta) for dy [BH, T, 32].
extern "C" __global__ void __launch_bounds__(128) delta_bwd(const float* __restrict__ q, const float* __restrict__ k,
                                                            const float* __restrict__ vg, const float* __restrict__ states,
                                                            const float* __restrict__ dy, float* __restrict__ dq,
                                                            float* __restrict__ dk, float* __restrict__ dvg, int BH, int T) {
    __shared__ float Qs[4][32], Ks[4][32];
    __shared__ float Rq[4][32][33], Rk[4][32][33];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.x * 4 + w;
    if (bh >= BH) return;
    const float* qb = q + (size_t)bh * T * 32;
    const float* kb = k + (size_t)bh * T * 32;
    const float* gb = vg + (size_t)bh * T * DVG;
    const float* yb = dy + (size_t)bh * T * 32;
    float dF[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) dF[i] = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        const float qc = qb[(size_t)t * 32 + lane], kc = kb[(size_t)t * 32 + lane], vc = gb[(size_t)t * DVG + lane];
        const float lac = gb[(size_t)t * DVG + 32], bec = gb[(size_t)t * DVG + 33], dyc = yb[(size_t)t * 32 + lane];
        float F[32];
        const float* s = states + ((size_t)bh * T + t) * 1024 + lane;
#pragma unroll
        for (int i = 0; i < 32; ++i) F[i] = s[i * 32];
        Qs[w][lane] = qc;
        Ks[w][lane] = kc;
        __syncwarp();
        float p = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) p += Ks[w][i] * F[i];
        const float a = expf(lac);
        const float e = vc - a * p;
        float sk = 0.0f, da = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            dF[i] += Qs[w][i] * dyc; // dF' += q dy^T (y read the updated F)
            sk += dF[i] * Ks[w][i];  // (dF'^T k)_j
            da += dF[i] * F[i];      // d alpha, direct term (column j's share)
        }
        const float de = bec * sk;   // dL/de_j (= dL/dv_j)
        const float dp = -a * de;    // dL/dp_j
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            Rq[w][lane][i] = (a * F[i] + bec * Ks[w][i] * e) * dyc; // F'_ij dy_j
            Rk[w][lane][i] = bec * dF[i] * e + F[i] * dp;           // beta dF'_ij e_j + F_ij dp_j
        }
        __syncwarp();
        float gq = 0.0f, gk = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            gq += Rq[w][j][lane];
            gk += Rk[w][j][lane];
        }
        const float dalpha = wsum(da - de * p);
        const float dbeta = wsum(e * sk);
        dq[((size_t)bh * T + t) * 32 + lane] = gq;
        dk[((size_t)bh * T + t) * 32 + lane] = gk;
        dvg[((size_t)bh * T + t) * DVG + lane] = de;
        if (lane == 0) {
            dvg[((size_t)bh * T + t) * DVG + 32] = dalpha * a;
            dvg[((size_t)bh * T + t) * DVG + 33] = dbeta;
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) dF[i] = a * dF[i] + Ks[w][i] * dp; // dF before the step
        __syncwarp();
    }
}

// ---- speed round 12: D's whole fast-weight block (after the input projection) in one kernel each way ----
// P [N, ld] f32 (row n = b*T + t) = h . [Wq | Wk | Wv | Wg | Wa | Wb]: columns q (d), k (d), v (d), g (d), za (H), zb (H);
// cst [3H] = (ba, bb, off). Per head h (lane j = channel j of the 32):
//   q^ = q / sqrt(|q|^2 + 1e-6), k^ likewise; alpha = sigmoid(za + ba + off); beta = 2 sigmoid(zb + bb)
//   p = F^T k^, e = v - alpha p, F <- alpha F + beta k^ e^T, y = F^T q^          (the gated delta rule)
//   O = y / sqrt(mean(y^2) + eps) * silu(g)                                       (head norm, output gate)
// O [N, d] f32. The same maths as model.rs hebb() for D (l2_norm, hebb_gates, delta_rule, head_norm, out_gate).
// The forward also saves the state F every HB_C steps (ck [B*H, ceil(T/HB_C), 32, 32]); the backward recomputes the
// states inside each chunk from its checkpoint (the same arithmetic, so the same values) into a small per-warp
// scratch instead of reading a saved state for every step (round 12 first saved all of them: 400 MB per layer at
// Ava-B's shape).
#define HB_C 16
__device__ __forceinline__ float hb_sig(float z) { return z >= 0.0f ? 1.0f / (1.0f + expf(-z)) : expf(z) / (1.0f + expf(z)); }

// One step of the delta rule on the state F (lane = column), as the forward does it. Ks = k^ in shared memory.
__device__ __forceinline__ void hb_update(float (&F)[32], const float* Ks, float a, float be, float v) {
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
        p0 += Ks[i] * F[i]; p1 += Ks[i + 1] * F[i + 1]; p2 += Ks[i + 2] * F[i + 2]; p3 += Ks[i + 3] * F[i + 3];
    }
    const float bee = be * (v - a * ((p0 + p1) + (p2 + p3)));
#pragma unroll
    for (int i = 0; i < 32; ++i) F[i] = a * F[i] + Ks[i] * bee;
}

// v40: the body is shared by hebb_fwd (O f32) and hebb_fwd_b (O as bf16, f2bf of the same values: the matmul operand
// written directly instead of cast from f32 afterwards).
// v44 K2: SG = true is the sigmoid output gate, O = y r sigmoid(g) (g already h·Wd·Wu); SG = false compiles today's code.
template <typename OT, bool SG = false>
__device__ __forceinline__ void hebb_fwd_body(const float* __restrict__ P, const float* __restrict__ cst, OT* __restrict__ O,
                                              float* __restrict__ ck, int B, int H, int T, int ld, float eps) {
    __shared__ float Qs[4][32], Ks[4][32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.x * 4 + w;
    if (bh >= B * H) return;
    const int b = bh / H, h = bh % H, d = H * 32, nck = (T + HB_C - 1) / HB_C;
    const float ca = cst[h] + cst[2 * H + h], cb = cst[H + h];
    const float* rowp = P + (size_t)b * T * ld;
    float F[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) F[i] = 0.0f;
    float qn = rowp[h * 32 + lane], kn = rowp[d + h * 32 + lane], vn = rowp[2 * d + h * 32 + lane], gn = rowp[3 * d + h * 32 + lane];
    float zan = rowp[4 * d + h], zbn = rowp[4 * d + H + h];
    for (int t = 0; t < T; ++t) {
        const float q = qn, k = kn, v = vn, g = gn, za = zan, zb = zbn;
        if (t + 1 < T) {
            const float* r = rowp + (size_t)(t + 1) * ld;
            qn = r[h * 32 + lane]; kn = r[d + h * 32 + lane]; vn = r[2 * d + h * 32 + lane]; gn = r[3 * d + h * 32 + lane];
            zan = r[4 * d + h]; zbn = r[4 * d + H + h];
        }
        if (t % HB_C == 0) {
            float* s = ck + ((size_t)bh * nck + t / HB_C) * 1024 + lane;
#pragma unroll
            for (int i = 0; i < 32; ++i) s[i * 32] = F[i];
        }
        const float qh = q * rsqrtf(wsum(q * q) + 1e-6f), kh = k * rsqrtf(wsum(k * k) + 1e-6f);
        const float a = hb_sig(za + ca), be = 2.0f * hb_sig(zb + cb);
        Qs[w][lane] = qh;
        Ks[w][lane] = kh;
        __syncwarp();
        hb_update(F, Ks[w], a, be, v);
        float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i += 4) {
            y0 += Qs[w][i] * F[i]; y1 += Qs[w][i + 1] * F[i + 1]; y2 += Qs[w][i + 2] * F[i + 2]; y3 += Qs[w][i + 3] * F[i + 3];
        }
        const float y = (y0 + y1) + (y2 + y3);
        const float r = rsqrtf(wsum(y * y) * (1.0f / 32.0f) + eps);
        put(O, ((size_t)b * T + t) * d + h * 32 + lane, y * r * (SG ? hb_sig(g) : g * hb_sig(g)));
        __syncwarp();
    }
}

extern "C" __global__ void __launch_bounds__(128) hebb_fwd(const float* __restrict__ P, const float* __restrict__ cst, float* __restrict__ O,
                                                           float* __restrict__ ck, int B, int H, int T, int ld, float eps) {
    hebb_fwd_body<float>(P, cst, O, ck, B, H, T, ld, eps);
}

extern "C" __global__ void __launch_bounds__(128) hebb_fwd_b(const float* __restrict__ P, const float* __restrict__ cst, unsigned short* __restrict__ O,
                                                             float* __restrict__ ck, int B, int H, int T, int ld, float eps) {
    hebb_fwd_body<unsigned short>(P, cst, O, ck, B, H, T, ld, eps);
}

// v44 K2: hebb_fwd and hebb_fwd_b with the sigmoid output gate.
extern "C" __global__ void __launch_bounds__(128) hebb_fwd_sg(const float* __restrict__ P, const float* __restrict__ cst, float* __restrict__ O,
                                                              float* __restrict__ ck, int B, int H, int T, int ld, float eps) {
    hebb_fwd_body<float, true>(P, cst, O, ck, B, H, T, ld, eps);
}

extern "C" __global__ void __launch_bounds__(128) hebb_fwd_b_sg(const float* __restrict__ P, const float* __restrict__ cst, unsigned short* __restrict__ O,
                                                                float* __restrict__ ck, int B, int H, int T, int ld, float eps) {
    hebb_fwd_body<unsigned short, true>(P, cst, O, ck, B, H, T, ld, eps);
}

// Backward of hebb_fwd: dP [N, ld] (every column written) for dO [N, d]. Chunks of HB_C steps from the last: the
// chunk's states are recomputed from its checkpoint into scratch [B*H, HB_C, 32, 32] (per warp, read back by the
// thread that wrote it), then walked back through.
// v40: the body is shared by hebb_bwd (NW = 4 heads per block of 128 threads) and hebb_bwd_w1 (one head per block of
// 32 threads, at most 168 registers): the same code per head, so the same bits.
template <int NW, typename OT, bool SG = false>
__device__ __forceinline__ void hebb_bwd_body(const float* __restrict__ P, const float* __restrict__ cst,
                                              const float* __restrict__ ck, float* __restrict__ scratch,
                                              const float* __restrict__ dO, OT* __restrict__ dP, float* __restrict__ dz,
                                              int B, int H, int T, int ld, float eps) {
    __shared__ float Qs[NW][32], Ks[NW][32];
    __shared__ float Rq[NW][32][33], Rk[NW][32][33];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.x * NW + w;
    if (bh >= B * H) return;
    const int b = bh / H, h = bh % H, d = H * 32, nck = (T + HB_C - 1) / HB_C;
    const float ca = cst[h] + cst[2 * H + h], cb = cst[H + h];
    float* scr = scratch + (size_t)bh * HB_C * 1024 + lane;
    float dF[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) dF[i] = 0.0f;
    // the inputs of the next step are loaded while this one is worked on (as in the forward); same arithmetic
    float qn, kn, vn, gn, zan, zbn, don;
    {
        const size_t m = (size_t)b * T + (T - 1);
        const float* r = P + m * ld;
        qn = r[h * 32 + lane]; kn = r[d + h * 32 + lane]; vn = r[2 * d + h * 32 + lane]; gn = r[3 * d + h * 32 + lane];
        zan = r[4 * d + h]; zbn = r[4 * d + H + h]; don = dO[m * d + h * 32 + lane];
    }
    for (int c = nck - 1; c >= 0; --c) {
        const int t0 = c * HB_C, t1 = min(T, t0 + HB_C);
        {
            float F[32];
            const float* s = ck + ((size_t)bh * nck + c) * 1024 + lane;
#pragma unroll
            for (int i = 0; i < 32; ++i) F[i] = s[i * 32];
            const float* r0 = P + ((size_t)b * T + t0) * ld;
            float kr = r0[d + h * 32 + lane], vr = r0[2 * d + h * 32 + lane], zar = r0[4 * d + h], zbr = r0[4 * d + H + h];
            for (int t = t0; t < t1; ++t) {
#pragma unroll
                for (int i = 0; i < 32; ++i) scr[(t - t0) * 1024 + i * 32] = F[i];
                if (t + 1 < t1) {
                    const float k = kr, v = vr, za = zar, zb = zbr;
                    if (t + 2 < t1) {
                        const float* r = P + ((size_t)b * T + t + 1) * ld;
                        kr = r[d + h * 32 + lane]; vr = r[2 * d + h * 32 + lane]; zar = r[4 * d + h]; zbr = r[4 * d + H + h];
                    }
                    const float kh = k * rsqrtf(wsum(k * k) + 1e-6f);
                    const float a = hb_sig(za + ca), be = 2.0f * hb_sig(zb + cb);
                    Ks[w][lane] = kh;
                    __syncwarp();
                    hb_update(F, Ks[w], a, be, v);
                    __syncwarp();
                }
            }
        }
        float Fn[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) Fn[i] = scr[(t1 - 1 - t0) * 1024 + i * 32];
        for (int t = t1 - 1; t >= t0; --t) {
            const size_t n = (size_t)b * T + t;
            const float q = qn, k = kn, v = vn, g = gn, za = zan, zb = zbn, dov = don;
            if (t > 0) {
                const float* r = P + (n - 1) * ld;
                qn = r[h * 32 + lane]; kn = r[d + h * 32 + lane]; vn = r[2 * d + h * 32 + lane]; gn = r[3 * d + h * 32 + lane];
                zan = r[4 * d + h]; zbn = r[4 * d + H + h]; don = dO[(n - 1) * d + h * 32 + lane];
            }
            float F[32];
#pragma unroll
            for (int i = 0; i < 32; ++i) F[i] = Fn[i];
            if (t > t0) {
#pragma unroll
                for (int i = 0; i < 32; ++i) Fn[i] = scr[(t - 1 - t0) * 1024 + i * 32];
            }
            const float iq = rsqrtf(wsum(q * q) + 1e-6f), ik = rsqrtf(wsum(k * k) + 1e-6f);
            const float qh = q * iq, kh = k * ik;
            const float a = hb_sig(za + ca), be = 2.0f * hb_sig(zb + cb);
            Qs[w][lane] = qh;
            Ks[w][lane] = kh;
            __syncwarp();
            float p = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; ++i) p += Ks[w][i] * F[i];
            const float e = v - a * p;
            float y = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; ++i) y += Qs[w][i] * (a * F[i] + be * Ks[w][i] * e);
            // O = y r silu(g): back through the gate and the head norm
            const float rr = rsqrtf(wsum(y * y) * (1.0f / 32.0f) + eps);
            // v44 K2 (SG): O = y r sigmoid(g), dg = dO yn sg (1 - sg)
            const float yn = y * rr, sg = hb_sig(g), silu = SG ? sg : g * sg;
            const float dg = SG ? dov * yn * sg * (1.0f - sg) : dov * yn * sg * (1.0f + g * (1.0f - sg));
            const float dyn = dov * silu;
            const float dy = rr * (dyn - yn * wsum(dyn * yn) * (1.0f / 32.0f));
            // the delta rule (as delta_bwd), with dy
            float sk = 0.0f, da = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                dF[i] += Qs[w][i] * dy;
                sk += dF[i] * Ks[w][i];
                da += dF[i] * F[i];
            }
            const float de = be * sk, dp = -a * de;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                Rq[w][lane][i] = (a * F[i] + be * Ks[w][i] * e) * dy;
                Rk[w][lane][i] = be * dF[i] * e + F[i] * dp;
            }
            __syncwarp();
            float gq = 0.0f, gk = 0.0f;
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                gq += Rq[w][j][lane];
                gk += Rk[w][j][lane];
            }
            const float dalpha = wsum(da - de * p), dbeta = wsum(e * sk);
            // unit-length q and k: dx = (dx^ - x^ (x^ . dx^)) / |x|
            const float dq = iq * (gq - qh * wsum(qh * gq)), dk = ik * (gk - kh * wsum(kh * gk));
            OT* o = dP + n * ld;
            put(o, h * 32 + lane, dq);
            put(o, d + h * 32 + lane, dk);
            put(o, 2 * d + h * 32 + lane, de);
            put(o, 3 * d + h * 32 + lane, dg);
            if (lane == 0) {
                const float za = dalpha * a * (1.0f - a), zb = dbeta * be * (1.0f - 0.5f * be);
                put(o, 4 * d + h, za);     // alpha = sigmoid(z)
                put(o, 4 * d + H + h, zb); // beta = 2 sigmoid(z)
                if (sizeof(OT) == 2) {     // v40: bf16 dP: dza, dzb also in f32 (dz [N, 2H]) for the bias gradients
                    dz[n * 2 * H + h] = za;
                    dz[n * 2 * H + H + h] = zb;
                }
            }
#pragma unroll
            for (int i = 0; i < 32; ++i) dF[i] = a * dF[i] + Ks[w][i] * dp;
            __syncwarp();
        }
    }
}

extern "C" __global__ void __launch_bounds__(128) hebb_bwd(const float* __restrict__ P, const float* __restrict__ cst,
                                                           const float* __restrict__ ck, float* __restrict__ scratch,
                                                           const float* __restrict__ dO, float* __restrict__ dP,
                                                           int B, int H, int T, int ld, float eps) {
    hebb_bwd_body<4, float>(P, cst, ck, scratch, dO, dP, nullptr, B, H, T, ld, eps);
}

// v40: one head per block, 9 blocks per SM: B·H = 384 heads (D at 32 x 256) fit the RTX 4070's 46 SMs in one wave
// (hebb_bwd's 4-head blocks at 246 registers fit 92 blocks of the 96, so a second wave ran for the last 4).
extern "C" __global__ void __launch_bounds__(32, 9) hebb_bwd_w1(const float* __restrict__ P, const float* __restrict__ cst,
                                                               const float* __restrict__ ck, float* __restrict__ scratch,
                                                               const float* __restrict__ dO, float* __restrict__ dP,
                                                               int B, int H, int T, int ld, float eps) {
    hebb_bwd_body<1, float>(P, cst, ck, scratch, dO, dP, nullptr, B, H, T, ld, eps);
}

// v40: hebb_bwd and hebb_bwd_w1 writing dP as bf16 (f2bf of the same values: the matmul operand) and dza, dzb in f32
// at dz [N, 2H].
extern "C" __global__ void __launch_bounds__(128) hebb_bwd_bf(const float* __restrict__ P, const float* __restrict__ cst,
                                                              const float* __restrict__ ck, float* __restrict__ scratch,
                                                              const float* __restrict__ dO, unsigned short* __restrict__ dP,
                                                              float* __restrict__ dz, int B, int H, int T, int ld, float eps) {
    hebb_bwd_body<4, unsigned short>(P, cst, ck, scratch, dO, dP, dz, B, H, T, ld, eps);
}

extern "C" __global__ void __launch_bounds__(32, 9) hebb_bwd_w1_bf(const float* __restrict__ P, const float* __restrict__ cst,
                                                                  const float* __restrict__ ck, float* __restrict__ scratch,
                                                                  const float* __restrict__ dO, unsigned short* __restrict__ dP,
                                                                  float* __restrict__ dz, int B, int H, int T, int ld, float eps) {
    hebb_bwd_body<1, unsigned short>(P, cst, ck, scratch, dO, dP, dz, B, H, T, ld, eps);
}

// v44 K2: the four backward kernels above with the sigmoid output gate.
extern "C" __global__ void __launch_bounds__(128) hebb_bwd_sg(const float* __restrict__ P, const float* __restrict__ cst,
                                                              const float* __restrict__ ck, float* __restrict__ scratch,
                                                              const float* __restrict__ dO, float* __restrict__ dP,
                                                              int B, int H, int T, int ld, float eps) {
    hebb_bwd_body<4, float, true>(P, cst, ck, scratch, dO, dP, nullptr, B, H, T, ld, eps);
}

extern "C" __global__ void __launch_bounds__(32, 9) hebb_bwd_w1_sg(const float* __restrict__ P, const float* __restrict__ cst,
                                                                  const float* __restrict__ ck, float* __restrict__ scratch,
                                                                  const float* __restrict__ dO, float* __restrict__ dP,
                                                                  int B, int H, int T, int ld, float eps) {
    hebb_bwd_body<1, float, true>(P, cst, ck, scratch, dO, dP, nullptr, B, H, T, ld, eps);
}

extern "C" __global__ void __launch_bounds__(128) hebb_bwd_bf_sg(const float* __restrict__ P, const float* __restrict__ cst,
                                                                 const float* __restrict__ ck, float* __restrict__ scratch,
                                                                 const float* __restrict__ dO, unsigned short* __restrict__ dP,
                                                                 float* __restrict__ dz, int B, int H, int T, int ld, float eps) {
    hebb_bwd_body<4, unsigned short, true>(P, cst, ck, scratch, dO, dP, dz, B, H, T, ld, eps);
}

extern "C" __global__ void __launch_bounds__(32, 9) hebb_bwd_w1_bf_sg(const float* __restrict__ P, const float* __restrict__ cst,
                                                                     const float* __restrict__ ck, float* __restrict__ scratch,
                                                                     const float* __restrict__ dO, unsigned short* __restrict__ dP,
                                                                     float* __restrict__ dz, int B, int H, int T, int ld, float eps) {
    hebb_bwd_body<1, unsigned short, true>(P, cst, ck, scratch, dO, dP, dz, B, H, T, ld, eps);
}

// ---- speed round 13: AdamW in place (w, m, v overwritten), the same arithmetic as adamw_step ----
extern "C" __global__ void adamw_inplace(float* __restrict__ w, const float* __restrict__ g, float* __restrict__ m, float* __restrict__ v,
                                         int n, float s, float b1, float c1, float b2, float c2, float eps, float step, float ibc2,
                                         float shrink) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gi = g[i] * s;
    const float mi = m[i] * b1 + gi * c1;
    const float vi = v[i] * b2 + (gi * gi) * c2;
    const float den = sqrtf(vi) * ibc2 + eps;
    w[i] = w[i] * shrink - (mi / den) * step;
    m[i] = mi;
    v[i] = vi;
}
// The same with the clipping factor read from the device (an f64 scalar, rounded to f32 as the host would).
extern "C" __global__ void adamw_inplace_s(float* __restrict__ w, const float* __restrict__ g, float* __restrict__ m, float* __restrict__ v,
                                           const double* __restrict__ sp, int n, float b1, float c1, float b2, float c2, float eps,
                                           float step, float ibc2, float shrink) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = __double2float_rn(sp[0]);
    if (!isfinite(s)) return; // v52 fix_S: a non-finite gradient norm (optim::clip_scale_t gives NaN) skips the update
    const float gi = g[i] * s;
    const float mi = m[i] * b1 + gi * c1;
    const float vi = v[i] * b2 + (gi * gi) * c2;
    const float den = sqrtf(vi) * ibc2 + eps;
    w[i] = w[i] * shrink - (mi / den) * step;
    m[i] = mi;
    v[i] = vi;
}

// ---- v40 (a faster block): the optimiser step in a few launches instead of ~700 small ones ----
// Squared gradient norm with exactly the grouping of AdamW::grad_norm_t's candle ops: per tensor sqr, then
// sum_keepdim(-1) (candle's fast_sum: one block of bd = next_pow2(min(1024, cols)) threads per row; thread v adds
// elements v, v + bd, v + 2 bd, ... in order, then a tree s = bd/2 .. 1), then sum_all of the row sums (the same with
// bd over the rows), then the tensors' sums added in order (f32), then the f64 sqrt. A rank-1 tensor is one row (its
// sum_all is that block). One warp stands in for one such block: virtual thread v = lane + 32 j keeps its partial
// sum in acc[j]; tree steps s >= 32 stay inside the lane, s < 32 are shuffles. f32 addition is commutative, so the
// grouping (candle's) fixes every rounding: the same bits as the per-tensor ops.
// gn_tab[4 i ..] = (data pointer, rows, cols, index of the tensor's first row).
template <int J, bool SQ>
__device__ __forceinline__ float gn_block_sum(const float* __restrict__ x, int n, int bd) {
    const int lane = threadIdx.x & 31;
    float acc[J];
#pragma unroll
    for (int j = 0; j < J; ++j) {
        float a = 0.0f;
        for (int i = lane + 32 * j; i < n; i += bd) {
            const float v = x[i];
            a = __fadd_rn(a, SQ ? __fmul_rn(v, v) : v);
        }
        acc[j] = a;
    }
#pragma unroll
    for (int s = J / 2; s >= 1; s >>= 1) {
#pragma unroll
        for (int j = 0; j < s; ++j) acc[j] = __fadd_rn(acc[j], acc[j + s]);
    }
    float r = acc[0];
    for (int s = (bd < 32 ? bd : 32) / 2; s >= 1; s >>= 1) r = __fadd_rn(r, __shfl_down_sync(0xffffffffu, r, s));
    return r;
}

__device__ __forceinline__ float gn_sum_any(const float* __restrict__ x, int n, bool sq) {
    int bd = 1;
    while (bd < n && bd < 1024) bd <<= 1;
    switch (bd > 32 ? bd >> 5 : 1) {
    case 1: return sq ? gn_block_sum<1, true>(x, n, bd) : gn_block_sum<1, false>(x, n, bd);
    case 2: return sq ? gn_block_sum<2, true>(x, n, bd) : gn_block_sum<2, false>(x, n, bd);
    case 4: return sq ? gn_block_sum<4, true>(x, n, bd) : gn_block_sum<4, false>(x, n, bd);
    case 8: return sq ? gn_block_sum<8, true>(x, n, bd) : gn_block_sum<8, false>(x, n, bd);
    case 16: return sq ? gn_block_sum<16, true>(x, n, bd) : gn_block_sum<16, false>(x, n, bd);
    default: return sq ? gn_block_sum<32, true>(x, n, bd) : gn_block_sum<32, false>(x, n, bd);
    }
}

// Stage 1, one warp per row of every tensor: rs[row] = the row's sum of squares.
extern "C" __global__ void gn_rows(const unsigned long long* __restrict__ tab, int ntens, int nrows, float* __restrict__ rs) {
    const int w = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (w >= nrows) return;
    int lo = 0, hi = ntens - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if ((long long)tab[4 * mid + 3] <= (long long)w) lo = mid;
        else hi = mid - 1;
    }
    const int cols = (int)tab[4 * lo + 2];
    const float* x = reinterpret_cast<const float*>(tab[4 * lo]) + (size_t)(w - (long long)tab[4 * lo + 3]) * cols;
    const float r = gn_sum_any(x, cols, true);
    if ((threadIdx.x & 31) == 0) rs[w] = r;
}

// Stage 2, one block: each warp sums one tensor's row sums (ts[i]); then thread 0 adds the tensors' sums in order
// and writes sqrt as f64 (the cast to f64 is exact; sqrt is correctly rounded, as candle's usqrt_f64).
extern "C" __global__ void gn_final(const unsigned long long* __restrict__ tab, int ntens, const float* __restrict__ rs, float* __restrict__ ts,
                                    double* __restrict__ out) {
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int i = w; i < ntens; i += nw) {
        const float s = gn_sum_any(rs + tab[4 * i + 3], (int)tab[4 * i + 1], false);
        if ((threadIdx.x & 31) == 0) ts[i] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = ts[0];
        for (int i = 1; i < ntens; ++i) acc = __fadd_rn(acc, ts[i]);
        out[0] = sqrt((double)acc);
    }
}

// AdamW over many tensors in one launch, element for element the arithmetic of adamw_inplace_s. Block b updates
// elements (b - first) * blockDim.x + tid of the tensor whose block range holds b. aw_tab[6 i ..] = (w, g, m, v
// pointers, n, first block | shrink bits << 32).
extern "C" __global__ void adamw_multi(const unsigned long long* __restrict__ tab, int ntens, const double* __restrict__ sp, float b1, float c1,
                                       float b2, float c2, float eps, float step, float ibc2) {
    const long long b = blockIdx.x;
    int lo = 0, hi = ntens - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if ((long long)(tab[6 * mid + 5] & 0xffffffffull) <= b) lo = mid;
        else hi = mid - 1;
    }
    const unsigned long long* e = tab + 6 * lo;
    const long long i = (b - (long long)(e[5] & 0xffffffffull)) * blockDim.x + threadIdx.x;
    if (i >= (long long)e[4]) return;
    float* __restrict__ w = reinterpret_cast<float*>(e[0]);
    const float* __restrict__ g = reinterpret_cast<const float*>(e[1]);
    float* __restrict__ m = reinterpret_cast<float*>(e[2]);
    float* __restrict__ v = reinterpret_cast<float*>(e[3]);
    const float shrink = __uint_as_float((unsigned int)(e[5] >> 32));
    const float s = __double2float_rn(sp[0]);
    if (!isfinite(s)) return; // v52 fix_S: a non-finite gradient norm skips the update (as adamw_inplace_s)
    // the contractions nvcc chose for adamw_inplace_s (read from its SASS), spelled out so they cannot differ
    const float gi = __fmul_rn(g[i], s);
    const float mi = __fmaf_rn(m[i], b1, __fmul_rn(gi, c1));
    const float vi = __fmaf_rn(v[i], b2, __fmul_rn(__fmul_rn(gi, gi), c2));
    const float den = __fmaf_rn(sqrtf(vi), ibc2, eps);
    w[i] = __fmaf_rn(w[i], shrink, -__fmul_rn(mi / den, step));
    m[i] = mi;
    v[i] = vi;
}

// ==== v38: chunk form of D's fast-weight block (the same maths as hebb_fwd / hebb_bwd, in chunks of HC_C tokens) ====
// The gated delta rule reordered as in DeltaNet / Gated DeltaNet, so that most of the work runs in parallel over
// chunks instead of one token after another. In a chunk of C = HC_C tokens that starts from state F (32 x 32, F_ij:
// key i, value j), with lg_t = sum_{s<=t} log alpha_s from the chunk start and gamma_t = exp(lg_t), the written
// vectors delta_t = beta_t e_t solve
//   (I + A) Delta = diag(beta) R,   R_t = v_t - gamma_t F^T k^_t,   A_ts = beta_t exp(lg_t - lg_s) k^_t . k^_s (s < t),
// so Delta = U - W F with (I + A) U = diag(beta) V and (I + A) W = diag(beta gamma) K^ (forward substitution), and
//   read-out    y_t = gamma_t F^T q^_t + sum_{s<=t} exp(lg_t - lg_s) (q^_t . k^_s) delta_s
//   next state  F'  = gamma_C F + sum_s exp(lg_C - lg_s) k^_s delta_s^T.
// Forward: hc_prep (U, W, Kd = exp(lg_C - lg) k^, gC = gamma_C per chunk), hc_scan (per head: the state entering
// every chunk, saved as ck exactly like hebb_fwd's HB_C checkpoints, and Delta), hc_out (O). Backward: hc_bout (the
// WY form again, then back through the head norm and gate: dY, dg; X1 = M^T dY, X2 = Q^T diag(gamma) dY; W, Kd, gC
// for the scan), hc_bscan (per head, from the last chunk: dDelta and the gradient of each chunk's end state),
// hc_bmain (everything else). The per-chunk kernels run one warp per chunk, lane = channel; the two scans one block
// of 128 threads per head, each column of the state split over four neighbouring lanes (columns never mix). Rows
// past the end of the sequence are zero with alpha = 1, beta = 0, so they change nothing. Tiles in shared memory
// have rows of HC_P floats (16-byte aligned: float4 reads without bank conflicts); a lane's 8 entries of a 16 x 16
// matrix are row t = lane / 2, columns s = 2m + lane % 2 (m = 0..7).
#define HC_C 16
#define HC_P 36
#define HC_Q 20
static_assert(HC_C == HB_C, "ck is shared with hebb_fwd / hebb_bwd");

__device__ __forceinline__ float hc_logsig(float z) { return z >= 0.0f ? -log1pf(expf(-z)) : z - log1pf(expf(z)); }

// x . F for a 32-float row x in shared memory (read as float4 broadcasts) and F in registers.
__device__ __forceinline__ float hc_rowF(const float* x, const float (&F)[32]) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 p = x4[i];
        a0 += p.x * F[4 * i];
        a1 += p.y * F[4 * i + 1];
        a2 += p.z * F[4 * i + 2];
        a3 += p.w * F[4 * i + 3];
    }
    return (a0 + a1) + (a2 + a3);
}

__device__ __forceinline__ void hc_row(const float* x, float (&r)[32]) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 p = x4[i];
        r[4 * i] = p.x;
        r[4 * i + 1] = p.y;
        r[4 * i + 2] = p.z;
        r[4 * i + 3] = p.w;
    }
}

__device__ __forceinline__ float hc_sq(const float* x) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 p = x4[i];
        a0 += p.x * p.x;
        a1 += p.y * p.y;
        a2 += p.z * p.z;
        a3 += p.w * p.w;
    }
    return (a0 + a1) + (a2 + a3);
}

// The sums over the 32 lanes of v[0..16) (v is used up): returns the sum for row t = lane / 2 (lanes 2t and 2t + 1).
// One halving step of hc_rsum16: lanes with bit 2W set keep v[W..2W), the others v[0..W); the kept half gets the
// partner lane's copy of it added, and lands in v[0..W).
template <int W>
__device__ __forceinline__ void hc_half(float (&v)[16], int lane) {
    const bool hi = lane & (2 * W);
#pragma unroll
    for (int i = 0; i < W; ++i) {
        const float send = hi ? v[i] : v[i + W], keep = hi ? v[i + W] : v[i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, 2 * W);
    }
}

__device__ __forceinline__ float hc_rsum16(float (&v)[16]) {
    const int lane = threadIdx.x & 31;
    hc_half<8>(v, lane);
    hc_half<4>(v, lane);
    hc_half<2>(v, lane);
    hc_half<1>(v, lane);
    return v[0] + __shfl_xor_sync(0xffffffffu, v[0], 1);
}

// This lane's 8 Gram entries (t = lane / 2, s = 2m + lane % 2): g1[m] = X1_t . Y_s and, if TWO, g2[m] = X2_t . Y_s.
template <bool TWO>
__device__ __forceinline__ void hc_gram(const float (*X1)[HC_P], const float (*X2)[HC_P], const float (*Y)[HC_P], float (&g1)[8], float (&g2)[8]) {
    const int lane = threadIdx.x & 31, t = lane >> 1, p = lane & 1;
    float a[32], b[32];
    hc_row(X1[t], a);
    if (TWO) hc_row(X2[t], b);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        const float4* y4 = reinterpret_cast<const float4*>(Y[2 * m + p]);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 y = y4[i];
            a0 += a[4 * i] * y.x;
            a1 += a[4 * i + 1] * y.y;
            a2 += a[4 * i + 2] * y.z;
            a3 += a[4 * i + 3] * y.w;
            if (TWO) {
                b0 += b[4 * i] * y.x;
                b1 += b[4 * i + 1] * y.y;
                b2 += b[4 * i + 2] * y.z;
                b3 += b[4 * i + 3] * y.w;
            }
        }
        g1[m] = (a0 + a1) + (a2 + a3);
        g2[m] = TWO ? (b0 + b1) + (b2 + b3) : 0.0f;
    }
}

// Chunk c of head h of sequence b: q^ (if LQ) and k^ (unit length, as hebb_fwd: x rsqrt(|x|^2 + 1e-6)) into the
// tiles q and k, raw v into registers (if LV), and per row lg, be (beta), al (alpha), iq = 1/|q|, ik = 1/|k|.
// Returns the number of real rows.
template <bool LQ, bool LV>
__device__ __forceinline__ int hc_load(const float* __restrict__ P, const float* __restrict__ cst, int b, int h, int c, int H, int T, int ld,
                                       float (*q)[HC_P], float (*k)[HC_P], float (&v)[HC_C], float* lg, float* be, float* al, float* iq,
                                       float* ik) {
    const int lane = threadIdx.x & 31, d = H * 32, t0 = c * HC_C, n = min(HC_C, T - t0);
    const float* rowp = P + ((size_t)b * T + t0) * ld;
    float qr[HC_C], kr[HC_C];
#pragma unroll
    for (int r = 0; r < HC_C; ++r) {
        const float* rp = rowp + (size_t)r * ld + h * 32 + lane;
        const bool in = r < n;
        if (LQ) qr[r] = in ? rp[0] : 0.0f;
        kr[r] = in ? rp[d] : 0.0f;
        if (LV) v[r] = in ? rp[2 * d] : 0.0f;
    }
#pragma unroll
    for (int r = 0; r < HC_C; ++r) {
        if (LQ) q[r][lane] = qr[r];
        k[r][lane] = kr[r];
    }
    __syncwarp();
    float la = 0.0f;
    {
        const int r = lane & 15;
        const float(*x)[HC_P] = (LQ && lane < 16) ? q : k;
        const float inv = rsqrtf(hc_sq(x[r]) + 1e-6f);
        if (lane >= 16) ik[r] = inv;
        else if (LQ) iq[r] = inv;
        if (lane < 16) {
            float a = 1.0f, bb = 0.0f;
            if (r < n) {
                const float* gp = rowp + (size_t)r * ld + 4 * d;
                const float za = gp[h] + (cst[h] + cst[2 * H + h]), zb = gp[H + h] + cst[H + h];
                a = hb_sig(za);
                la = hc_logsig(za);
                bb = 2.0f * hb_sig(zb);
            }
            al[r] = a;
            be[r] = bb;
        }
    }
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) {
        const float u = __shfl_up_sync(0xffffffffu, la, o, 16);
        if ((lane & 15) >= o) la += u;
    }
    if (lane < 16) lg[lane] = la;
    __syncwarp();
#pragma unroll
    for (int r = 0; r < HC_C; ++r) {
        if (LQ) q[r][lane] = qr[r] * iq[r];
        k[r][lane] = kr[r] * ik[r];
    }
    __syncwarp();
    return n;
}

// This lane's column of U and W from A (in shared memory): (I + A) U = diag(beta) V, (I + A) W = diag(beta gamma) K^.
__device__ __forceinline__ void hc_subst(const float (*k)[HC_P], const float (&v)[HC_C], const float (*A)[HC_C + 1], const float* lg, const float* be,
                                         float (&u)[HC_C], float (&w)[HC_C]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int t = 0; t < HC_C; ++t) {
        float uu = be[t] * v[t], ww = be[t] * expf(lg[t]) * k[t][lane];
#pragma unroll
        for (int s = 0; s < t; ++s) {
            const float a = A[t][s];
            uu -= a * u[s];
            ww -= a * w[s];
        }
        u[t] = uu;
        w[t] = ww;
    }
}

// This lane's column of y: y_t = gamma_t F^T q^_t + sum_{s<=t} M_ts delta_s.
__device__ __forceinline__ void hc_y(const float (*q)[HC_P], const float (*M)[HC_C + 1], const float* lg, const float (&F)[32], const float (&dl)[HC_C],
                                     float (&y)[HC_C]) {
#pragma unroll
    for (int t = 0; t < HC_C; ++t) {
        float x = expf(lg[t]) * hc_rowF(q[t], F);
#pragma unroll
        for (int s = 0; s <= t; ++s) x += M[t][s] * dl[s];
        y[t] = x;
    }
}

// rsqrt(mean_j y_t^2 + eps) for row t = lane / 2.
__device__ __forceinline__ float hc_rr(const float (&y)[HC_C], float eps) {
    float sq[HC_C];
#pragma unroll
    for (int t = 0; t < HC_C; ++t) sq[t] = y[t] * y[t];
    return rsqrtf(hc_rsum16(sq) * (1.0f / 32.0f) + eps);
}

// Forward 1, one warp per chunk: U, W of the WY form, Kd = exp(lg_C - lg) k^ and gC = gamma_C.
extern "C" __global__ void __launch_bounds__(32) hc_prep(const float* __restrict__ P, const float* __restrict__ cst, float* __restrict__ U,
                                                         float* __restrict__ W, float* __restrict__ Kd, float* __restrict__ gC,
                                                         int B, int H, int T, int ld) {
    __shared__ __align__(16) float k[HC_C][HC_P];
    __shared__ float A[HC_C][HC_C + 1], lg[HC_C], be[HC_C], al[HC_C], ik[HC_C];
    const int lane = threadIdx.x & 31, nC = (T + HC_C - 1) / HC_C, cid = blockIdx.x, bh = cid / nC, c = cid % nC, t2 = lane >> 1;
    if (bh >= B * H) return;
    float v[HC_C];
    hc_load<false, true>(P, cst, bh / H, bh % H, c, H, T, ld, nullptr, k, v, lg, be, al, nullptr, ik);
    {
        float g[8], g2[8];
        hc_gram<false>(k, k, k, g, g2);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int s = 2 * m + (lane & 1);
            A[t2][s] = s < t2 ? be[t2] * expf(lg[t2] - lg[s]) * g[m] : 0.0f;
        }
    }
    __syncwarp();
    float u[HC_C], w[HC_C];
    hc_subst(k, v, A, lg, be, u, w);
    const float lgC = lg[HC_C - 1];
    const size_t o = (size_t)cid * (HC_C * 32);
#pragma unroll
    for (int t = 0; t < HC_C; ++t) {
        U[o + t * 32 + lane] = u[t];
        W[o + t * 32 + lane] = w[t];
        Kd[o + t * 32 + lane] = expf(lgC - lg[t]) * k[t][lane];
    }
    if (lane == 0) gC[cid] = expf(lgC);
}

// Forward 2, one block of 128 threads per head; thread (j = tid / 4, r = tid % 4) holds rows 8r..8r+7 of column j of
// the state. Delta = U - W F, F' = gC F + Kd^T Delta. Saves F_c (the state entering chunk c) at
// ck[(bh nC + c) 1024 + i 32 + j], the layout of hebb_fwd's checkpoints, and Delta_c at DL[(bh nC + c) 512 + t 32 + j].
// The next chunk's U, W, Kd are read while this one is worked on.
extern "C" __global__ void __launch_bounds__(128) hc_scan(const float* __restrict__ U, const float* __restrict__ W, const float* __restrict__ Kd,
                                                          const float* __restrict__ gC, float* __restrict__ ck, float* __restrict__ DL, int nC) {
    __shared__ __align__(16) float sb[2][3][HC_C][32];
    const int tid = threadIdx.x, j = tid >> 2, r = tid & 3;
    const size_t base = (size_t)blockIdx.x * nC;
    const float4* src[3] = {reinterpret_cast<const float4*>(U), reinterpret_cast<const float4*>(W), reinterpret_cast<const float4*>(Kd)};
    float F[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) F[i] = 0.0f;
    float4 nx[3];
#pragma unroll
    for (int m = 0; m < 3; ++m) nx[m] = src[m][base * 128 + tid];
    float g = gC[base], gn = 0.0f;
#pragma unroll
    for (int m = 0; m < 3; ++m) reinterpret_cast<float4*>(&sb[0][m][0][0])[tid] = nx[m];
    __syncthreads();
    for (int c = 0; c < nC; ++c) {
        const size_t cid = base + c;
        const int cur = c & 1;
        if (c + 1 < nC) {
#pragma unroll
            for (int m = 0; m < 3; ++m) nx[m] = src[m][(cid + 1) * 128 + tid];
            gn = gC[cid + 1];
        }
        float* s = ck + cid * 1024 + j;
#pragma unroll
        for (int i = 0; i < 8; ++i) s[(8 * r + i) * 32] = F[i];
        const float(*Us)[32] = sb[cur][0];
        const float(*Ws)[32] = sb[cur][1];
        const float(*Ks)[32] = sb[cur][2];
        float dl[HC_C];
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            const float4 a = reinterpret_cast<const float4*>(Ws[t])[2 * r], bq = reinterpret_cast<const float4*>(Ws[t])[2 * r + 1];
            float x = a.x * F[0];
            x += a.y * F[1];
            x += a.z * F[2];
            x += a.w * F[3];
            x += bq.x * F[4];
            x += bq.y * F[5];
            x += bq.z * F[6];
            x += bq.w * F[7];
            dl[t] = x;
        }
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            dl[t] += __shfl_xor_sync(0xffffffffu, dl[t], 1);
            dl[t] += __shfl_xor_sync(0xffffffffu, dl[t], 2);
        }
        float* dp = DL + cid * 512 + j;
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            dl[t] = Us[t][j] - dl[t];
            if ((t & 3) == r) dp[t * 32] = dl[t];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) F[i] *= g;
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            const float4 a = reinterpret_cast<const float4*>(Ks[t])[2 * r], bq = reinterpret_cast<const float4*>(Ks[t])[2 * r + 1];
            F[0] += a.x * dl[t];
            F[1] += a.y * dl[t];
            F[2] += a.z * dl[t];
            F[3] += a.w * dl[t];
            F[4] += bq.x * dl[t];
            F[5] += bq.y * dl[t];
            F[6] += bq.z * dl[t];
            F[7] += bq.w * dl[t];
        }
        if (c + 1 < nC) {
#pragma unroll
            for (int m = 0; m < 3; ++m) reinterpret_cast<float4*>(&sb[cur ^ 1][m][0][0])[tid] = nx[m];
        }
        g = gn;
        __syncthreads();
    }
}

// Forward 3, one warp per chunk (lane = value channel j): y from the chunk's start state (ck) and Delta (from
// hc_scan), then O = y rsqrt(mean y^2 + eps) silu(g).
template <typename OT, bool SG = false>
__device__ __forceinline__ void hc_out_body(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ DL,
                                            const float* __restrict__ ck, OT* __restrict__ O, int B, int H, int T, int ld, float eps) {
    __shared__ __align__(16) float q[HC_C][HC_P], k[HC_C][HC_P];
    __shared__ float M[HC_C][HC_C + 1], lg[HC_C], be[HC_C], al[HC_C], iq[HC_C], ik[HC_C];
    const int lane = threadIdx.x & 31, nC = (T + HC_C - 1) / HC_C, cid = blockIdx.x, bh = cid / nC, c = cid % nC, t2 = lane >> 1;
    if (bh >= B * H) return;
    const int b = bh / H, h = bh % H, d = H * 32;
    float v[HC_C];
    const int n = hc_load<true, false>(P, cst, b, h, c, H, T, ld, q, k, v, lg, be, al, iq, ik);
    float F[32], dl[HC_C];
    const size_t o = (size_t)cid * (HC_C * 32);
#pragma unroll
    for (int i = 0; i < 32; ++i) F[i] = ck[(size_t)cid * 1024 + i * 32 + lane];
#pragma unroll
    for (int t = 0; t < HC_C; ++t) dl[t] = DL[o + t * 32 + lane];
    {
        float g[8], g2[8];
        hc_gram<false>(q, q, k, g, g2);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int s = 2 * m + (lane & 1);
            M[t2][s] = s <= t2 ? expf(lg[t2] - lg[s]) * g[m] : 0.0f;
        }
    }
    __syncwarp();
    float y[HC_C];
    hc_y(q, M, lg, F, dl, y);
    const float rr = hc_rr(y, eps);
#pragma unroll
    for (int t = 0; t < HC_C; ++t) {
        const float rt = __shfl_sync(0xffffffffu, rr, 2 * t);
        if (t < n) {
            const size_t row = (size_t)b * T + c * HC_C + t;
            const float g = P[row * ld + 3 * d + h * 32 + lane];
            put(O, row * d + h * 32 + lane, y[t] * rt * (SG ? hb_sig(g) : g * hb_sig(g)));
        }
    }
}

extern "C" __global__ void __launch_bounds__(32) hc_out(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ DL,
                                                        const float* __restrict__ ck, float* __restrict__ O, int B, int H, int T, int ld, float eps) {
    hc_out_body<float>(P, cst, DL, ck, O, B, H, T, ld, eps);
}

// v40: hc_out writing O as bf16 (the matmul operand).
extern "C" __global__ void __launch_bounds__(32) hc_out_b(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ DL,
                                                          const float* __restrict__ ck, unsigned short* __restrict__ O, int B, int H, int T, int ld,
                                                          float eps) {
    hc_out_body<unsigned short>(P, cst, DL, ck, O, B, H, T, ld, eps);
}

// v44 K2: hc_out and hc_out_b with the sigmoid output gate.
extern "C" __global__ void __launch_bounds__(32) hc_out_sg(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ DL,
                                                           const float* __restrict__ ck, float* __restrict__ O, int B, int H, int T, int ld, float eps) {
    hc_out_body<float, true>(P, cst, DL, ck, O, B, H, T, ld, eps);
}

extern "C" __global__ void __launch_bounds__(32) hc_out_b_sg(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ DL,
                                                             const float* __restrict__ ck, unsigned short* __restrict__ O, int B, int H, int T, int ld,
                                                             float eps) {
    hc_out_body<unsigned short, true>(P, cst, DL, ck, O, B, H, T, ld, eps);
}

// Backward 1, one warp per chunk (lane = value channel j): the WY form and y again (as hc_prep and hc_out), then dO
// back through the gate and head norm (dg into dP, dY), X1 = M^T dY and X2 = Q^T diag(gamma) dY; W, Kd, gC for
// hc_bscan.
template <typename OT, bool SG = false>
__device__ __forceinline__ void hc_bout_body(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ ck,
                                             const float* __restrict__ dO, float* __restrict__ W, float* __restrict__ Kd,
                                             float* __restrict__ gC, OT* __restrict__ dP, float* __restrict__ DY,
                                             float* __restrict__ X1, float* __restrict__ X2, int B, int H, int T, int ld, float eps) {
    __shared__ __align__(16) float q[HC_C][HC_P], k[HC_C][HC_P], w[HC_C][HC_P];
    __shared__ float A[HC_C][HC_C + 1], M[HC_C][HC_C + 1], lg[HC_C], be[HC_C], al[HC_C], iq[HC_C], ik[HC_C];
    const int lane = threadIdx.x & 31, nC = (T + HC_C - 1) / HC_C, cid = blockIdx.x, bh = cid / nC, c = cid % nC, t2 = lane >> 1;
    if (bh >= B * H) return;
    const int b = bh / H, h = bh % H, d = H * 32;
    float v[HC_C];
    const int n = hc_load<true, true>(P, cst, b, h, c, H, T, ld, q, k, v, lg, be, al, iq, ik);
    {
        float ga[8], gq[8];
        hc_gram<true>(k, q, k, ga, gq);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int s = 2 * m + (lane & 1);
            const float el = expf(lg[t2] - lg[s]);
            A[t2][s] = s < t2 ? be[t2] * el * ga[m] : 0.0f;
            M[t2][s] = s <= t2 ? el * gq[m] : 0.0f;
        }
    }
    __syncwarp();
    const size_t o = (size_t)cid * (HC_C * 32);
    float dl[HC_C];
    {
        float u[HC_C], wc[HC_C];
        hc_subst(k, v, A, lg, be, u, wc);
        const float lgC = lg[HC_C - 1];
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            W[o + t * 32 + lane] = wc[t];
            Kd[o + t * 32 + lane] = expf(lgC - lg[t]) * k[t][lane];
            w[t][lane] = wc[t];
        }
        if (lane == 0) gC[cid] = expf(lgC);
        __syncwarp();
        float F[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) F[i] = ck[(size_t)cid * 1024 + i * 32 + lane];
#pragma unroll
        for (int t = 0; t < HC_C; ++t) dl[t] = u[t] - hc_rowF(w[t], F);
        float y[HC_C];
        hc_y(q, M, lg, F, dl, y);
        const float rr = hc_rr(y, eps);
        float yn[HC_C], dyn[HC_C], pm[HC_C];
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            const float rt = __shfl_sync(0xffffffffu, rr, 2 * t);
            yn[t] = 0.0f;
            dyn[t] = 0.0f;
            if (t < n) {
                const size_t row = (size_t)b * T + c * HC_C + t;
                const float g = P[row * ld + 3 * d + h * 32 + lane], dov = dO[row * d + h * 32 + lane];
                const float sg = hb_sig(g), silu = SG ? sg : g * sg; // v44 K2 (SG): the sigmoid gate
                yn[t] = y[t] * rt;
                put(dP, row * ld + 3 * d + h * 32 + lane, SG ? dov * yn[t] * sg * (1.0f - sg) : dov * yn[t] * sg * (1.0f + g * (1.0f - sg)));
                dyn[t] = dov * silu;
            }
            pm[t] = dyn[t] * yn[t];
        }
        const float mm = hc_rsum16(pm) * (1.0f / 32.0f);
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            const float rt = __shfl_sync(0xffffffffu, rr, 2 * t), mt = __shfl_sync(0xffffffffu, mm, 2 * t);
            dl[t] = rt * (dyn[t] - yn[t] * mt); // dl holds dY from here
            DY[o + t * 32 + lane] = dl[t];
        }
    }
#pragma unroll
    for (int s = 0; s < HC_C; ++s) {
        float x = 0.0f;
#pragma unroll
        for (int t = s; t < HC_C; ++t) x += M[t][s] * dl[t];
        X1[o + s * 32 + lane] = x;
    }
    float gm[HC_C];
#pragma unroll
    for (int t = 0; t < HC_C; ++t) gm[t] = expf(lg[t]) * dl[t];
    float* x2 = X2 + (size_t)cid * 1024 + lane;
#pragma unroll 2
    for (int i4 = 0; i4 < 8; ++i4) {
        float x0 = 0.0f, x1 = 0.0f, x2v = 0.0f, x3 = 0.0f;
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            const float4 qv = reinterpret_cast<const float4*>(q[t])[i4];
            x0 += qv.x * gm[t];
            x1 += qv.y * gm[t];
            x2v += qv.z * gm[t];
            x3 += qv.w * gm[t];
        }
        x2[(4 * i4) * 32] = x0;
        x2[(4 * i4 + 1) * 32] = x1;
        x2[(4 * i4 + 2) * 32] = x2v;
        x2[(4 * i4 + 3) * 32] = x3;
    }
}

extern "C" __global__ void __launch_bounds__(32) hc_bout(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ ck,
                                                         const float* __restrict__ dO, float* __restrict__ W, float* __restrict__ Kd,
                                                         float* __restrict__ gC, float* __restrict__ dP, float* __restrict__ DY,
                                                         float* __restrict__ X1, float* __restrict__ X2, int B, int H, int T, int ld, float eps) {
    hc_bout_body<float>(P, cst, ck, dO, W, Kd, gC, dP, DY, X1, X2, B, H, T, ld, eps);
}

// v40: hc_bout writing dP's dg columns as bf16 (into hc_bmain_b's bf16 dP).
extern "C" __global__ void __launch_bounds__(32) hc_bout_b(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ ck,
                                                           const float* __restrict__ dO, float* __restrict__ W, float* __restrict__ Kd,
                                                           float* __restrict__ gC, unsigned short* __restrict__ dP, float* __restrict__ DY,
                                                           float* __restrict__ X1, float* __restrict__ X2, int B, int H, int T, int ld, float eps) {
    hc_bout_body<unsigned short>(P, cst, ck, dO, W, Kd, gC, dP, DY, X1, X2, B, H, T, ld, eps);
}

// v44 K2: hc_bout and hc_bout_b with the sigmoid output gate (hc_bscan, hc_bmain do not read the gate).
extern "C" __global__ void __launch_bounds__(32) hc_bout_sg(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ ck,
                                                            const float* __restrict__ dO, float* __restrict__ W, float* __restrict__ Kd,
                                                            float* __restrict__ gC, float* __restrict__ dP, float* __restrict__ DY,
                                                            float* __restrict__ X1, float* __restrict__ X2, int B, int H, int T, int ld, float eps) {
    hc_bout_body<float, true>(P, cst, ck, dO, W, Kd, gC, dP, DY, X1, X2, B, H, T, ld, eps);
}

extern "C" __global__ void __launch_bounds__(32) hc_bout_b_sg(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ ck,
                                                              const float* __restrict__ dO, float* __restrict__ W, float* __restrict__ Kd,
                                                              float* __restrict__ gC, unsigned short* __restrict__ dP, float* __restrict__ DY,
                                                              float* __restrict__ X1, float* __restrict__ X2, int B, int H, int T, int ld, float eps) {
    hc_bout_body<unsigned short, true>(P, cst, ck, dO, W, Kd, gC, dP, DY, X1, X2, B, H, T, ld, eps);
}

// Backward 2, one block of 128 threads per head from the last chunk (thread layout as hc_scan): G = gradient of the
// state at the end of chunk c (0 after the last, saved at dFn); dDelta_c = X1 + Kd G (saved at DD); the gradient of
// the state entering chunk c = gC G + X2 - W^T dDelta_c.
extern "C" __global__ void __launch_bounds__(128) hc_bscan(const float* __restrict__ W, const float* __restrict__ Kd, const float* __restrict__ gC,
                                                           const float* __restrict__ X1, const float* __restrict__ X2, float* __restrict__ dFn,
                                                           float* __restrict__ DD, int nC) {
    __shared__ __align__(16) float sb[2][3][HC_C][32];
    const int tid = threadIdx.x, j = tid >> 2, r = tid & 3;
    const size_t base = (size_t)blockIdx.x * nC;
    const float4* src[3] = {reinterpret_cast<const float4*>(W), reinterpret_cast<const float4*>(Kd), reinterpret_cast<const float4*>(X1)};
    float G[8], x2n[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) G[i] = 0.0f;
    const size_t last = base + nC - 1;
    float4 nx[3];
#pragma unroll
    for (int m = 0; m < 3; ++m) nx[m] = src[m][last * 128 + tid];
#pragma unroll
    for (int i = 0; i < 8; ++i) x2n[i] = X2[last * 1024 + (8 * r + i) * 32 + j];
    float g = gC[last], gn = 0.0f;
#pragma unroll
    for (int m = 0; m < 3; ++m) reinterpret_cast<float4*>(&sb[0][m][0][0])[tid] = nx[m];
    __syncthreads();
    for (int c = nC - 1, it = 0; c >= 0; --c, ++it) {
        const size_t cid = base + c;
        const int cur = it & 1;
        float x2[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) x2[i] = x2n[i];
        if (c > 0) {
#pragma unroll
            for (int m = 0; m < 3; ++m) nx[m] = src[m][(cid - 1) * 128 + tid];
#pragma unroll
            for (int i = 0; i < 8; ++i) x2n[i] = X2[(cid - 1) * 1024 + (8 * r + i) * 32 + j];
            gn = gC[cid - 1];
        }
        float* s = dFn + cid * 1024 + j;
#pragma unroll
        for (int i = 0; i < 8; ++i) s[(8 * r + i) * 32] = G[i];
        const float(*Ws)[32] = sb[cur][0];
        const float(*Ks)[32] = sb[cur][1];
        const float(*Xs)[32] = sb[cur][2];
        float dd[HC_C];
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            const float4 a = reinterpret_cast<const float4*>(Ks[t])[2 * r], bq = reinterpret_cast<const float4*>(Ks[t])[2 * r + 1];
            float x = a.x * G[0];
            x += a.y * G[1];
            x += a.z * G[2];
            x += a.w * G[3];
            x += bq.x * G[4];
            x += bq.y * G[5];
            x += bq.z * G[6];
            x += bq.w * G[7];
            dd[t] = x;
        }
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            dd[t] += __shfl_xor_sync(0xffffffffu, dd[t], 1);
            dd[t] += __shfl_xor_sync(0xffffffffu, dd[t], 2);
        }
        float* dp = DD + cid * 512 + j;
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            dd[t] += Xs[t][j];
            if ((t & 3) == r) dp[t * 32] = dd[t];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) G[i] = g * G[i] + x2[i];
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            const float4 a = reinterpret_cast<const float4*>(Ws[t])[2 * r], bq = reinterpret_cast<const float4*>(Ws[t])[2 * r + 1];
            G[0] -= a.x * dd[t];
            G[1] -= a.y * dd[t];
            G[2] -= a.z * dd[t];
            G[3] -= a.w * dd[t];
            G[4] -= bq.x * dd[t];
            G[5] -= bq.y * dd[t];
            G[6] -= bq.z * dd[t];
            G[7] -= bq.w * dd[t];
        }
        if (c > 0) {
#pragma unroll
            for (int m = 0; m < 3; ++m) reinterpret_cast<float4*>(&sb[cur ^ 1][m][0][0])[tid] = nx[m];
        }
        g = gn;
        __syncthreads();
    }
}

// Backward 3, one warp per chunk: with the chunk's start state (ck), the gradient of its end state (dFn), dDelta (DD)
// and dY (DY), the gradients of q, k, v, za and zb (dg came from hc_bout).
template <typename OT>
__device__ __forceinline__ void hc_bmain_body(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ ck,
                                              const float* __restrict__ dFn, const float* __restrict__ DD, const float* __restrict__ DY,
                                              OT* __restrict__ dP, float* __restrict__ dz, int B, int H, int T, int ld) {
    __shared__ __align__(16) float q[HC_C][HC_P], k[HC_C][HC_P], dl[HC_C][HC_P], Z[HC_C][HC_P], dy[HC_C][HC_P];
    __shared__ __align__(16) float L[HC_C][HC_Q], M[HC_C][HC_Q], DL[HC_C][HC_Q], DM[HC_C][HC_Q];
    __shared__ float lg[HC_C], be[HC_C], al[HC_C], iq[HC_C], ik[HC_C], dlg[HC_C], dbe[HC_C], qd[HC_C], kq[HC_C];
    const int lane = threadIdx.x & 31, nC = (T + HC_C - 1) / HC_C, cid = blockIdx.x, bh = cid / nC, c = cid % nC;
    if (bh >= B * H) return;
    const int b = bh / H, h = bh % H, d = H * 32, t2 = lane >> 1, p2 = lane & 1;
    float v[HC_C];
    const int n = hc_load<true, true>(P, cst, b, h, c, H, T, ld, q, k, v, lg, be, al, iq, ik);
    const size_t o512 = (size_t)cid * (HC_C * 32), o1024 = (size_t)cid * 1024;
    // lane = value channel j: gkf = gamma F^T k^ and R = v - gkf (this lane's column); s2a = dY_t . y1_t (row lane / 2)
    // with y1 = gamma F^T q^
    float R[HC_C], gkf[HC_C], s2a;
    {
        float F[32], dyv[HC_C], py[HC_C];
#pragma unroll
        for (int t = 0; t < HC_C; ++t) dyv[t] = DY[o512 + t * 32 + lane];
#pragma unroll
        for (int i = 0; i < 32; ++i) F[i] = ck[o1024 + i * 32 + lane];
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            const float g = expf(lg[t]), kf = g * hc_rowF(k[t], F);
            gkf[t] = kf;
            R[t] = v[t] - kf;
            dy[t][lane] = dyv[t];
            py[t] = dyv[t] * (g * hc_rowF(q[t], F));
        }
        s2a = hc_rsum16(py);
    }
    {   // L_ts = exp(lg_t - lg_s) k^_t . k^_s (s < t), M_ts = exp(lg_t - lg_s) q^_t . k^_s (s <= t)
        float gk[8], gq[8];
        hc_gram<true>(k, q, k, gk, gq);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int s = 2 * m + p2;
            const float el = expf(lg[t2] - lg[s]);
            L[t2][s] = s < t2 ? el * gk[m] : 0.0f;
            M[t2][s] = s <= t2 ? el * gq[m] : 0.0f;
        }
    }
    __syncwarp();
    {   // Delta: delta_t = beta_t (R_t - sum_{s<t} L_ts delta_s)
        float de[HC_C];
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            float x = R[t];
#pragma unroll
            for (int s = 0; s < t; ++s) x -= L[t][s] * de[s];
            de[t] = be[t] * x;
            dl[t][lane] = de[t];
        }
    }
    // Z = (I + A)^-T dDelta: Z_s = dDelta_s - sum_{t>s} beta_t L_ts Z_t; s1 = Z_t . R_t, s2b = (beta Z)_t . gkf_t
    // (row lane / 2); dv = dR = beta Z
    float s1, s2b;
    {
        float z[HC_C];
#pragma unroll
        for (int s = HC_C - 1; s >= 0; --s) {
            float x = DD[o512 + s * 32 + lane];
#pragma unroll
            for (int t = s + 1; t < HC_C; ++t) x -= be[t] * L[t][s] * z[t];
            z[s] = x;
            Z[s][lane] = x;
        }
        float p1[HC_C], pg[HC_C];
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            const float bz = be[t] * z[t];
            p1[t] = z[t] * R[t];
            pg[t] = bz * gkf[t];
            if (t < n) put(dP, ((size_t)b * T + c * HC_C + t) * ld + 2 * d + h * 32 + lane, bz);
        }
        s1 = hc_rsum16(p1);
        s2b = hc_rsum16(pg);
    }
    __syncwarp();
    // this lane's entries: dA_ts = -Z_t . delta_s (s < t), dM_ts = dY_t . delta_s (s <= t): DL = dL, DM = dM, and E
    // (the d lg terms, by row and by column) over M; re, rb = the row sums of E and BL
    float re = 0.0f, rb = 0.0f;
    {
        float gy[8], gz[8];
        hc_gram<true>(dy, Z, dl, gy, gz);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int s = 2 * m + p2;
            float bl = 0.0f, dlv = 0.0f, dmv = 0.0f, ee = 0.0f;
            if (s <= t2) {
                const float el = expf(lg[t2] - lg[s]), zm = gy[m];
                dmv = zm * el;
                ee = zm * M[t2][s];
                if (s < t2) {
                    const float za = -gz[m], dLv = be[t2] * za;
                    bl = za * L[t2][s];
                    dlv = dLv * el;
                    ee += dLv * L[t2][s];
                }
            }
            DL[t2][s] = dlv;
            DM[t2][s] = dmv;
            M[t2][s] = ee;
            re += ee;
            rb += bl;
        }
    }
    re += __shfl_xor_sync(0xffffffffu, re, 1);
    rb += __shfl_xor_sync(0xffffffffu, rb, 1);
    __syncwarp();
    // d lg_t = sum_s E_ts - sum_s E_st + dY_t . y1_t - (beta Z)_t . gkf_t;  d beta_t = Z_t . R_t + sum_s BL_ts
    float ce = 0.0f;
    if (lane < 16) {
#pragma unroll
        for (int t = 0; t < HC_C; ++t) ce += M[t][lane];
    }
    if (p2 == 0) {
        dbe[t2] = s1 + rb;
        dlg[t2] = re + (s2a - s2b);
    }
    __syncwarp();
    if (lane < 16) dlg[lane] -= ce;
    // dR = diag(beta) Z (rows of Z), DL + DL^T over L, DM^T over M
#pragma unroll
    for (int t = 0; t < HC_C; ++t) Z[t][lane] *= be[t];
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int s = 2 * m + p2;
        L[t2][s] = DL[t2][s] + DL[s][t2];
        M[t2][s] = DM[s][t2];
    }
    __syncwarp();
    // lane = key channel i from here
    const float lgC = lg[HC_C - 1];
    float dq[HC_C], dk[HC_C], ccs = 0.0f, scr, fgp = 0.0f;
    {   // rows i of F (the start state) and G (the gradient of the end state); fgp = this row's part of F . G
        float Fr[32], Gr[32], cc[HC_C];
        const float4* f4 = reinterpret_cast<const float4*>(ck + o1024 + lane * 32);
        const float4* g4 = reinterpret_cast<const float4*>(dFn + o1024 + lane * 32);
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
            const float4 x = f4[jj], y = g4[jj];
            Fr[4 * jj] = x.x;
            Fr[4 * jj + 1] = x.y;
            Fr[4 * jj + 2] = x.z;
            Fr[4 * jj + 3] = x.w;
            Gr[4 * jj] = y.x;
            Gr[4 * jj + 1] = y.y;
            Gr[4 * jj + 2] = y.z;
            Gr[4 * jj + 3] = y.w;
        }
#pragma unroll
        for (int jj = 0; jj < 32; ++jj) fgp += Fr[jj] * Gr[jj];
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            const float g = expf(lg[t]), gd = expf(lgC - lg[t]) * hc_rowF(dl[t], Gr);
            dq[t] = g * hc_rowF(dy[t], Fr);
            dk[t] = -(g * hc_rowF(Z[t], Fr)) + gd;
            cc[t] = gd * k[t][lane];
            ccs += cc[t];
        }
        scr = hc_rsum16(cc);
    }
    float kc[HC_C], qc[HC_C];
#pragma unroll
    for (int s = 0; s < HC_C; ++s) {
        kc[s] = k[s][lane];
        qc[s] = q[s][lane];
    }
#pragma unroll
    for (int t = 0; t < HC_C; ++t) {
        const float4* m4 = reinterpret_cast<const float4*>(DM[t]);
        const float4* mt4 = reinterpret_cast<const float4*>(M[t]);
        const float4* l4 = reinterpret_cast<const float4*>(L[t]);
        float a = dq[t], bb = dk[t];
#pragma unroll
        for (int s4 = 0; s4 < 4; ++s4) {
            const float4 x = m4[s4], y = mt4[s4], zz = l4[s4];
            a += x.x * kc[4 * s4];
            a += x.y * kc[4 * s4 + 1];
            a += x.z * kc[4 * s4 + 2];
            a += x.w * kc[4 * s4 + 3];
            bb += y.x * qc[4 * s4] + zz.x * kc[4 * s4];
            bb += y.y * qc[4 * s4 + 1] + zz.y * kc[4 * s4 + 1];
            bb += y.z * qc[4 * s4 + 2] + zz.z * kc[4 * s4 + 2];
            bb += y.w * qc[4 * s4 + 3] + zz.w * kc[4 * s4 + 3];
        }
        dq[t] = a;
        dk[t] = bb;
    }
    {   // qd_t = q^_t . dq_t, kq_t = k^_t . dk_t; the end-state terms of d lg
        float p3[HC_C], p4[HC_C];
#pragma unroll
        for (int t = 0; t < HC_C; ++t) {
            p3[t] = qc[t] * dq[t];
            p4[t] = kc[t] * dk[t];
        }
        const float sq = hc_rsum16(p3), sk = hc_rsum16(p4), csum = wsum(ccs), fg = wsum(fgp);
        if (p2 == 0) {
            qd[t2] = sq;
            kq[t2] = sk;
            float x = dlg[t2] - scr;
            if (t2 == HC_C - 1) x += csum + expf(lgC) * fg;
            dlg[t2] = x;
        }
    }
    __syncwarp();
    // d log alpha_r = sum_{t >= r} d lg_t; alpha = sigmoid(za), beta = 2 sigmoid(zb)
    float x = lane < 16 ? dlg[lane] : 0.0f;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) {
        const float u = __shfl_down_sync(0xffffffffu, x, o, 16);
        if ((lane & 15) + o < 16) x += u;
    }
    if (lane < n) {
        const size_t row = (size_t)b * T + c * HC_C + lane;
        const float za = x * (1.0f - al[lane]), zb = dbe[lane] * be[lane] * (1.0f - 0.5f * be[lane]);
        put(dP, row * ld + 4 * d + h, za);
        put(dP, row * ld + 4 * d + H + h, zb);
        if (sizeof(OT) == 2) { // v40: bf16 dP: dza, dzb also in f32 (dz [N, 2H]) for the bias gradients
            dz[row * 2 * H + h] = za;
            dz[row * 2 * H + H + h] = zb;
        }
    }
#pragma unroll
    for (int t = 0; t < HC_C; ++t) {
        if (t < n) {
            const size_t row = (size_t)b * T + c * HC_C + t;
            put(dP, row * ld + h * 32 + lane, iq[t] * (dq[t] - qc[t] * qd[t]));
            put(dP, row * ld + d + h * 32 + lane, ik[t] * (dk[t] - kc[t] * kq[t]));
        }
    }
}

extern "C" __global__ void __launch_bounds__(32) hc_bmain(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ ck,
                                                          const float* __restrict__ dFn, const float* __restrict__ DD, const float* __restrict__ DY,
                                                          float* __restrict__ dP, int B, int H, int T, int ld) {
    hc_bmain_body<float>(P, cst, ck, dFn, DD, DY, dP, nullptr, B, H, T, ld);
}

// v40: hc_bmain writing dP as bf16 (the matmul operand) and dza, dzb in f32 at dz [N, 2H].
extern "C" __global__ void __launch_bounds__(32) hc_bmain_b(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ ck,
                                                            const float* __restrict__ dFn, const float* __restrict__ DD, const float* __restrict__ DY,
                                                            unsigned short* __restrict__ dP, float* __restrict__ dz, int B, int H, int T, int ld) {
    hc_bmain_body<unsigned short>(P, cst, ck, dFn, DD, DY, dP, dz, B, H, T, ld);
}

// ---------------------------------------------------------------- v39: the glance's place ids
// out[i] = the place of token i in its row of length t (ids are [rows, t] row-major): for a digit (ids 48..57) at
// index k of its run, if the token before the run is a length mark 4096 + L − 1 with L <= 15 and k < L, the place
// is L − 1 − k (0 = units); everything else is 16 (none). The look-back stops at the row start and after 16
// digits. Same rule as ava_data::numbers::place_ids (tested against it).
extern "C" __global__ void place_ids(const unsigned int* __restrict__ ids, unsigned int* __restrict__ out, int n, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int tok = ids[i];
    unsigned int p = 16u;
    if (tok >= 48u && tok <= 57u) {
        int pos = i % t;
        int row0 = i - pos;
        int k = 0;
        while (k < 16 && pos - k - 1 >= 0) {
            unsigned int u = ids[row0 + pos - k - 1];
            if (u < 48u || u > 57u) break;
            k++;
        }
        if (k < 16 && pos - k - 1 >= 0) {
            unsigned int m = ids[row0 + pos - k - 1];
            if (m >= 4096u && m < 4096u + 15u) {
                int len = (int)(m - 4096u) + 1;
                if (k < len) p = (unsigned int)(len - 1 - k);
            }
        }
    }
    out[i] = p;
}

// v40: the magnitude code ("how much"), the rule of src/mag.rs: one thread per token, double precision.
// out [n, 14]: [1, m/16, sin(2 pi m/P), cos(2 pi m/P) for P = 1, 2, 4, 8, 16, 32] where m is the token's place on
// the log10 number line (a length mark: L - 0.5; a digit: the middle of what the number can still be), or all 0.
extern "C" __global__ void mag_feats(const unsigned int* __restrict__ ids, float* __restrict__ out, int n, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int tok = ids[i];
    int pos = i % t;
    int row0 = i - pos;
    bool has = false;
    double m = 0.0;
    if (tok >= 4096u && tok < 4112u) {
        int len = (int)(tok - 4096u) + 1;
        if (len <= 15) { m = (double)len - 0.5; has = true; }
    } else if (tok >= 48u && tok <= 57u) {
        int k = 0;
        while (k < 16 && pos - k - 1 >= 0) {
            unsigned int u = ids[row0 + pos - k - 1];
            if (u < 48u || u > 57u) break;
            k++;
        }
        if (pos - k - 1 >= 0 && k < 16) {
            unsigned int mk = ids[row0 + pos - k - 1];
            if (mk >= 4096u && mk < 4112u) {
                int len = (int)(mk - 4096u) + 1;
                if (len <= 15 && k < len) {
                    double v = 0.0;
                    for (int j = pos - k; j <= pos; ++j) v = v * 10.0 + (double)(ids[row0 + j] - 48u);
                    int r = len - 1 - k;
                    m = (r == 0) ? log10(v + 1.0) : log10((v + 0.5) * pow(10.0, (double)r) + 1.0);
                    has = true;
                }
            }
        }
    }
    float* o = out + (size_t)i * 14;
    if (!has) {
        for (int j = 0; j < 14; ++j) o[j] = 0.0f;
        return;
    }
    const double periods[6] = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
    o[0] = 1.0f;
    o[1] = (float)(m / 16.0);
    for (int j = 0; j < 6; ++j) {
        double a = 2.0 * 3.14159265358979323846 * m / periods[j];
        o[2 + 2 * j] = (float)sin(a);
        o[3 + 2 * j] = (float)cos(a);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// v42 the general LIF scan (src/lif_gen.rs): E1 graded spikes (L levels each side of 0), E4 adaptive thresholds (a trace
// with learned rho and gain), E8 per-neuron thresholds and O1's carried state, one thread per (b, c) as lif_fwd. The
// arithmetic is arch::lif_ref's, op for op in the same order with _rn intrinsics, so the spikes are the same:
//     a = a*rho + |s_prev|*(1 - rho);  th = theta + a*gain          (E4; else th = theta)
//     v = (v*beta + I) - s_prev*th_prev
//     s = sum_{k=1..L} [v - k*th >= 0] - [-v - k*th >= 0]          (binary: [v - th >= 0])
// ps = [beta | theta | rho | gain] (each D), then [v0 | s0 | a0] (each B*D, row i = b*D + c).
// Forward out (flat): spikes [B, T, D], then v_T [B, D], a_T [B, D].
// Backward out (flat): dL/dI [B, T, D], then dL/dv0, dL/da0, and the row's parts of dL/dbeta, dL/drho, dL/dgain (each
// [B, D], summed over b by the caller). Surrogate per edge 1/(1 + S|x|)^2; the reset and the trace's spikes are detached.
template <typename F>
__device__ __forceinline__ void lif_pf(const float* __restrict__ x, int T, int D, F step) {
    const int tl = T - 1;
    float xa[LIF_G];
#pragma unroll
    for (int u = 0; u < LIF_G; ++u) xa[u] = x[(size_t)min(u, tl) * D];
    int t0 = 0;
    for (; t0 + LIF_G <= T; t0 += LIF_G) {
        float xb[LIF_G];
#pragma unroll
        for (int u = 0; u < LIF_G; ++u) xb[u] = x[(size_t)min(t0 + LIF_G + u, tl) * D];
#pragma unroll
        for (int u = 0; u < LIF_G; ++u) step(t0 + u, xa[u]);
#pragma unroll
        for (int u = 0; u < LIF_G; ++u) xa[u] = xb[u];
    }
#pragma unroll
    for (int u = 0; u < LIF_G; ++u) {
        if (t0 + u < T) step(t0 + u, xa[u]);
    }
}
template <int L, bool SG>
__device__ __forceinline__ float lifg_spike(float v, float th) {
    if (!SG) return __fsub_rn(v, th) >= 0.0f ? 1.0f : 0.0f;
    float h = 0.0f;
#pragma unroll
    for (int k = 1; k <= L; ++k) {
        const float kth = __fmul_rn(th, (float)k);
        h += (__fsub_rn(v, kth) >= 0.0f ? 1.0f : 0.0f) - (__fsub_rn(-v, kth) >= 0.0f ? 1.0f : 0.0f);
    }
    return h;
}
__device__ __forceinline__ float lifg_sur(float x, float slope) {
    const float a = 1.0f + slope * fabsf(x);
    return 1.0f / (a * a);
}
// ds/dv and ds/dth of one step's surrogate
template <int L, bool SG>
__device__ __forceinline__ void lifg_dspike(float v, float th, float slope, float& dv, float& dth) {
    if (!SG) {
        const float g = lifg_sur(v - th, slope);
        dv = g;
        dth = -g;
        return;
    }
    dv = 0.0f;
    dth = 0.0f;
#pragma unroll
    for (int k = 1; k <= L; ++k) {
        const float kth = th * (float)k;
        const float gu = lifg_sur(v - kth, slope), gd = lifg_sur(-v - kth, slope);
        dv += gu + gd;
        dth += (float)k * (gd - gu);
    }
}
template <int L, bool SG, bool AL>
__device__ __forceinline__ void lifg_step(float x, float bt, float th0, float rho, float omr, float gain, float& v, float& sp,
                                          float& a, float& thp) {
    float tht = th0;
    if (AL) {
        a = __fadd_rn(__fmul_rn(a, rho), __fmul_rn(fabsf(sp), omr));
        tht = __fadd_rn(th0, __fmul_rn(a, gain));
    }
    v = __fsub_rn(__fadd_rn(__fmul_rn(v, bt), x), __fmul_rn(sp, thp));
    sp = lifg_spike<L, SG>(v, tht);
    thp = tht;
}
template <int L, bool SG, bool AL>
__device__ __forceinline__ void lifg_fwd_impl(const float* __restrict__ cur, const float* __restrict__ ps, float* __restrict__ out,
                                              int B, int T, int D) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * D) return;
    const int b = i / D, c = i - b * D;
    const size_t BD = (size_t)B * D, BTD = BD * T;
    const float bt = ps[c], th0 = ps[D + c], rho = ps[2 * D + c], gain = ps[3 * D + c];
    const float omr = __fsub_rn(1.0f, rho);
    const float* st = ps + 4 * (size_t)D;
    float v = st[i], sp = st[BD + i], a = st[2 * BD + i];
    float thp = AL ? __fadd_rn(th0, __fmul_rn(a, gain)) : th0;
    const float* x = cur + (size_t)b * T * D + c;
    float* o = out + (size_t)b * T * D + c;
    lif_pf(x, T, D, [&](int t, float xt) {
        lifg_step<L, SG, AL>(xt, bt, th0, rho, omr, gain, v, sp, a, thp);
        o[(size_t)t * D] = sp;
    });
    out[BTD + i] = v;
    out[BTD + BD + i] = a;
}
// v52 spike fix: the threshold the reset path carries in the backward: th_t, or 2 beta / (ds/dv) where th_t ds/dv > 2 beta,
// so the per-step factor beta - th_t ds/dv stays in [-beta, beta] (the cap passes no gradient)
__device__ __forceinline__ float lifg_te(float th, float dv, float bt) {
    const float q = th * dv, cap = 2.0f * bt;
    return q > cap ? cap / dv : th;
}
// ah: [B, T, D] scratch for the trace (E4 only). The membrane is kept in out's dL/dI rows, overwritten in the reverse pass.
template <int L, bool SG, bool AL>
__device__ __forceinline__ void lifg_bwd_impl(const float* __restrict__ cur, const float* __restrict__ ps, const float* __restrict__ g,
                                              float* __restrict__ out, float* __restrict__ ah, int B, int T, int D, float slope, int rgi) {
    // v52 spike fix (--lif-reset-grad, rgi = 1): the gradient also flows through the reset v_{t+1} -= s_t th_t (to the
    // spikes and, with ALIF, to the threshold), its carry capped by lifg_te; rgi = 0: the reset detached, as before
    const bool rg = rgi != 0;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * D) return;
    const int b = i / D, c = i - b * D;
    const size_t BD = (size_t)B * D, BTD = BD * T;
    const float bt = ps[c], th0 = ps[D + c], rho = ps[2 * D + c], gain = ps[3 * D + c];
    const float omr = __fsub_rn(1.0f, rho);
    const float* st = ps + 4 * (size_t)D;
    const float v0 = st[i], s0 = st[BD + i], a0 = st[2 * BD + i];
    float v = v0, sp = s0, a = a0;
    float thp = AL ? __fadd_rn(th0, __fmul_rn(a0, gain)) : th0;
    const size_t off = (size_t)b * T * D + c;
    const float* x = cur + off;
    float* o = out + off;
    float* h = ah + (AL ? off : 0);
    lif_pf(x, T, D, [&](int t, float xt) {
        lifg_step<L, SG, AL>(xt, bt, th0, rho, omr, gain, v, sp, a, thp);
        o[(size_t)t * D] = v;
        if (AL) h[(size_t)t * D] = a;
    });
    const float* gs = g + off;
    float cv = g[BTD + i], ca = AL ? g[BTD + BD + i] : 0.0f, gb = 0.0f, gr = 0.0f, gg = 0.0f;
    float gvn = 0.0f; // dL/dv_{t+1}
    for (int t = T - 1; t >= 0; --t) {
        const float vt = o[(size_t)t * D];
        const float at = AL ? h[(size_t)t * D] : 0.0f;
        const float tht = AL ? __fadd_rn(th0, __fmul_rn(at, gain)) : th0;
        float dv, dth;
        lifg_dspike<L, SG>(vt, tht, slope, dv, dth);
        const float gst = rg ? gs[(size_t)t * D] - lifg_te(tht, dv, bt) * gvn : gs[(size_t)t * D];
        const float gv = gst * dv + cv;
        const float vp = t > 0 ? o[(size_t)(t - 1) * D] : v0;
        gb += gv * vp;
        if (AL) {
            const float gth = rg ? gst * dth - lifg_spike<L, SG>(vt, tht) * gvn : gst * dth;
            const float ga = gth * gain + ca;
            gg += gth * at;
            const float ap = t > 0 ? h[(size_t)(t - 1) * D] : a0;
            const float spv = t > 0 ? lifg_spike<L, SG>(vp, __fadd_rn(th0, __fmul_rn(ap, gain))) : s0;
            gr += ga * (ap - fabsf(spv));
            ca = rho * ga;
        }
        o[(size_t)t * D] = gv;
        gvn = gv;
        cv = bt * gv;
    }
    if (AL && rg) { // the first step's reset used the initial threshold th0 + a0 gain
        const float gth0 = -s0 * gvn;
        gg += gth0 * a0;
        ca += gth0 * gain;
    }
    out[BTD + i] = cv;
    out[BTD + BD + i] = ca;
    out[BTD + 2 * BD + i] = gb;
    out[BTD + 3 * BD + i] = gr;
    out[BTD + 4 * BD + i] = gg;
}
#define LIFG_KERNELS(NAME, L, SG, AL)                                                                                        \
    extern "C" __global__ void lifg_fwd_##NAME(const float* __restrict__ cur, const float* __restrict__ ps,                  \
                                               float* __restrict__ out, int B, int T, int D) {                               \
        lifg_fwd_impl<L, SG, AL>(cur, ps, out, B, T, D);                                                                     \
    }                                                                                                                        \
    extern "C" __global__ void lifg_bwd_##NAME(const float* __restrict__ cur, const float* __restrict__ ps,                  \
                                               const float* __restrict__ g, float* __restrict__ out, float* __restrict__ ah, \
                                               int B, int T, int D, float slope, int rgi) {                                         \
        lifg_bwd_impl<L, SG, AL>(cur, ps, g, out, ah, B, T, D, slope, rgi);                                                     \
    }
LIFG_KERNELS(b0, 1, false, false)
LIFG_KERNELS(b1, 1, false, true)
LIFG_KERNELS(t0, 1, true, false)
LIFG_KERNELS(t1, 1, true, true)
LIFG_KERNELS(i0, 2, true, false)
LIFG_KERNELS(i1, 2, true, true)

// ---------------------------------------------------------------------------------------------------------------------
// v42 the general fast-weight memory (src/mem_gen.rs): P2 per-channel gates and erase key, P5 momentum, P1 head size 64.
// v45 W1 (KDA = true, --gates kda): read key alpha*e and erase key b*e instead (the fade first, then erase and write).
// Per head, F [HS x HS] (i = key row, j = value column); thread j holds column j (F[i] in registers), and (P5) S too:
//   r_j  = sum_i e_i F_ij                                   (what the memory recalls for the erase key)
//   U_ij = (b_i k_i) v_j - (alpha_i b_i e_i) r_j
//   S <- M S + U (P5);  F <- Diag(alpha) F + S              (no P5: F <- Diag(alpha) F + U)
//   y_j  = sum_i q_i F_ij
// qke [BH, T, 3 HS] = q | k | e (unit length); vg [BH, T, HS + 2G] = v | log alpha | b, G = 1 (per head) or HS (P2).
// The rule of arch::delta_rec (the same values up to rounding). One block of HS threads per head.
// Backward: the forward is rerun saving F (and S) every C steps (ck; C = MG_C unless the host passes another chunk length); each chunk's states are then recomputed from its
// checkpoint into a per-head scratch (scr) and walked back, carrying dF (and dS) in registers.
#define MG_C 16
template <int HS, bool PC>
struct MgIn {
    float q, k, e, v, a, b; // this thread's channel of the step's inputs (a = alpha, per head: the head's)
    __device__ __forceinline__ void load(const float* __restrict__ qr, const float* __restrict__ gr, int j) {
        const int G = PC ? HS : 1;
        q = qr[j];
        k = qr[HS + j];
        e = qr[2 * HS + j];
        v = gr[j];
        a = expf(gr[HS + (PC ? j : 0)]);
        b = gr[HS + G + (PC ? j : 0)];
    }
};
template <int HS>
struct MgSh {
    float Q[HS], W[HS], E[HS], AE[HS], A[HS];
};
template <int HS, bool PC, bool KDA>
__device__ __forceinline__ void memg_share(const MgIn<HS, PC>& x, MgSh<HS>& sh, int j) {
    sh.Q[j] = x.q;
    sh.W[j] = x.b * x.k;
    if (KDA) {
        // v45 W1: read with the faded key, erase with b*e (e = k)
        sh.E[j] = x.a * x.e;
        sh.AE[j] = x.b * x.e;
    } else {
        sh.E[j] = x.e;
        sh.AE[j] = x.a * x.b * x.e;
    }
    sh.A[j] = x.a;
}
template <int HS>
__device__ __forceinline__ float memg_dot(const float* __restrict__ s, const float (&F)[HS]) {
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
#pragma unroll
    for (int i = 0; i < HS; i += 4) {
        p0 += s[i] * F[i];
        p1 += s[i + 1] * F[i + 1];
        p2 += s[i + 2] * F[i + 2];
        p3 += s[i + 3] * F[i + 3];
    }
    return (p0 + p1) + (p2 + p3);
}
// v52 fix_S: the bound on the state. Every element of F stays in [-cap, cap] (cap = mem_gen::MEM_CAP unless a test sets
// another): with momentum (P5) and a write strength up to 2 the rule is not stable for keys that keep changing (the
// heavy-ball iteration grows from b ~ 0.6 at M = 0.5), and on a repeating text one head's F grew past f32 (runs/v52/
// fix_S.md). Below cap this is the identity (the old bits); where the bound holds an element its gradient is zero.
__device__ __forceinline__ float mg_cap(float x, float cap) { return fminf(fmaxf(x, -cap), cap); }
// one step of the rule on column j (v = v_j); returns nothing, updates F (and S)
template <int HS, bool MOM>
__device__ __forceinline__ void memg_step(float (&F)[HS], float (&S)[MOM ? HS : 1], const MgSh<HS>& sh, float v, float M,
                                          float cap) {
    const float r = memg_dot<HS>(sh.E, F);
#pragma unroll
    for (int i = 0; i < HS; ++i) {
        const float u = sh.W[i] * v - sh.AE[i] * r;
        if (MOM) {
            S[i] = M * S[i] + u;
            F[i] = mg_cap(sh.A[i] * F[i] + S[i], cap);
        } else {
            F[i] = mg_cap(sh.A[i] * F[i] + u, cap);
        }
    }
}
template <int HS, bool PC, bool MOM, bool KDA>
__device__ __forceinline__ void memg_fwd_impl(const float* __restrict__ qke, const float* __restrict__ vg, float* __restrict__ y,
                                              int BH, int T, float M, float cap) {
    __shared__ MgSh<HS> sh;
    const int j = threadIdx.x, bh = blockIdx.x;
    if (bh >= BH) return;
    const int LQ = 3 * HS, LG = HS + 2 * (PC ? HS : 1);
    const float* qb = qke + (size_t)bh * T * LQ;
    const float* gb = vg + (size_t)bh * T * LG;
    float F[HS], S[MOM ? HS : 1];
#pragma unroll
    for (int i = 0; i < HS; ++i) F[i] = 0.0f;
#pragma unroll
    for (int i = 0; i < (MOM ? HS : 1); ++i) S[i] = 0.0f;
    MgIn<HS, PC> nx;
    nx.load(qb, gb, j);
    for (int t = 0; t < T; ++t) {
        const MgIn<HS, PC> x = nx;
        if (t + 1 < T) nx.load(qb + (size_t)(t + 1) * LQ, gb + (size_t)(t + 1) * LG, j);
        memg_share<HS, PC, KDA>(x, sh, j);
        __syncthreads();
        memg_step<HS, MOM>(F, S, sh, x.v, M, cap);
        y[((size_t)bh * T + t) * HS + j] = memg_dot<HS>(sh.Q, F);
        __syncthreads();
    }
}
// dqke [BH, T, 3 HS], dvg [BH, T, HS + 2G]; ck [BH, ceil(T/C), NS, HS, HS], scr [BH, C, NS, HS, HS], NS = 2 with P5.
template <int HS, bool PC, bool MOM, bool KDA>
__device__ __forceinline__ void memg_bwd_impl(const float* __restrict__ qke, const float* __restrict__ vg, const float* __restrict__ dy,
                                              float* __restrict__ dqke, float* __restrict__ dvg, float* __restrict__ ck,
                                              float* __restrict__ scr, int BH, int T, float M, int C, float cap) {
    __shared__ MgSh<HS> sh;
    __shared__ float2 R[HS][HS + 1];
    __shared__ float red[2][HS];
    const int j = threadIdx.x, bh = blockIdx.x;
    if (bh >= BH) return;
    const int G = PC ? HS : 1, LQ = 3 * HS, LG = HS + 2 * G;
    constexpr int NS = MOM ? 2 : 1;
    constexpr size_t SQ = (size_t)HS * HS;
    const int nck = (T + C - 1) / C;
    const float* qb = qke + (size_t)bh * T * LQ;
    const float* gb = vg + (size_t)bh * T * LG;
    float* cb = ck + (size_t)bh * nck * NS * SQ;
    float* sb = scr + (size_t)bh * C * NS * SQ;
    float F[HS], S[MOM ? HS : 1];
#pragma unroll
    for (int i = 0; i < HS; ++i) F[i] = 0.0f;
#pragma unroll
    for (int i = 0; i < (MOM ? HS : 1); ++i) S[i] = 0.0f;
    // 1. the forward, saving the state at the start of every chunk
    for (int t = 0; t < T; ++t) {
        if (t % C == 0) {
            float* c = cb + (size_t)(t / C) * NS * SQ + j;
#pragma unroll
            for (int i = 0; i < HS; ++i) c[i * HS] = F[i];
            if (MOM) {
#pragma unroll
                for (int i = 0; i < (MOM ? HS : 1); ++i) c[SQ + i * HS] = S[i];
            }
        }
        MgIn<HS, PC> x;
        x.load(qb + (size_t)t * LQ, gb + (size_t)t * LG, j);
        memg_share<HS, PC, KDA>(x, sh, j);
        __syncthreads();
        memg_step<HS, MOM>(F, S, sh, x.v, M, cap);
        __syncthreads();
    }
    // 2. chunk by chunk, last first
    float dF[HS], dS[MOM ? HS : 1];
#pragma unroll
    for (int i = 0; i < HS; ++i) dF[i] = 0.0f;
#pragma unroll
    for (int i = 0; i < (MOM ? HS : 1); ++i) dS[i] = 0.0f;
    for (int c = nck - 1; c >= 0; --c) {
        const int t0 = c * C, t1 = min(T, t0 + C);
        {
            const float* cc = cb + (size_t)c * NS * SQ + j;
#pragma unroll
            for (int i = 0; i < HS; ++i) F[i] = cc[i * HS];
            if (MOM) {
#pragma unroll
                for (int i = 0; i < (MOM ? HS : 1); ++i) S[i] = cc[SQ + i * HS];
            }
        }
        for (int t = t0; t < t1; ++t) {
            float* s = sb + (size_t)(t - t0) * NS * SQ + j;
#pragma unroll
            for (int i = 0; i < HS; ++i) s[i * HS] = F[i];
            if (MOM) {
#pragma unroll
                for (int i = 0; i < (MOM ? HS : 1); ++i) s[SQ + i * HS] = S[i];
            }
            MgIn<HS, PC> x;
            x.load(qb + (size_t)t * LQ, gb + (size_t)t * LG, j);
            memg_share<HS, PC, KDA>(x, sh, j);
            __syncthreads();
            memg_step<HS, MOM>(F, S, sh, x.v, M, cap);
            __syncthreads();
        }
        for (int t = t1 - 1; t >= t0; --t) {
            const float* s = sb + (size_t)(t - t0) * NS * SQ + j;
#pragma unroll
            for (int i = 0; i < HS; ++i) F[i] = s[i * HS]; // the state before step t
            MgIn<HS, PC> x;
            x.load(qb + (size_t)t * LQ, gb + (size_t)t * LG, j);
            memg_share<HS, PC, KDA>(x, sh, j);
            __syncthreads();
            const float dyj = dy[((size_t)bh * T + t) * HS + j];
            const float r = memg_dot<HS>(sh.E, F);
            // dF' = carried + q dy^T; du = dS' (P5) or dF'; round 1 of the row sums: (F'_ij dy_j, dF'_ij F_ij)
            float dv = 0.0f, dr = 0.0f;
#pragma unroll
            for (int i = 0; i < HS; ++i) {
                const float u = sh.W[i] * x.v - sh.AE[i] * r;
                float fn;
                if (MOM) fn = sh.A[i] * F[i] + (M * s[SQ + i * HS] + u);
                else fn = sh.A[i] * F[i] + u;
                dF[i] += sh.Q[i] * dyj;
                if (!(fabsf(fn) < cap)) dF[i] = 0.0f; // v52 fix_S: the bound holds F here, so no gradient flows through it
                fn = mg_cap(fn, cap);
                R[j][i] = make_float2(fn * dyj, dF[i] * F[i]);
                float du;
                if (MOM) {
                    dS[(MOM ? i : 0)] += dF[i];
                    du = dS[(MOM ? i : 0)];
                } else {
                    du = dF[i];
                }
                dv += du * sh.W[i];
                dr -= du * sh.AE[i];
            }
            __syncthreads();
            float r1 = 0.0f, r2 = 0.0f;
#pragma unroll 8
            for (int jj = 0; jj < HS; ++jj) {
                const float2 p = R[jj][j];
                r1 += p.x;
                r2 += p.y;
            }
            __syncthreads();
            // round 2: (du_ij v_j, du_ij r_j)
#pragma unroll
            for (int i = 0; i < HS; ++i) {
                const float du = MOM ? dS[(MOM ? i : 0)] : dF[i];
                R[j][i] = make_float2(du * x.v, du * r);
            }
            __syncthreads();
            float r3 = 0.0f, r4 = 0.0f;
#pragma unroll 8
            for (int jj = 0; jj < HS; ++jj) {
                const float2 p = R[jj][j];
                r3 += p.x;
                r4 += p.y;
            }
            __syncthreads();
            // round 3: dr_j F_ij; then the carried gradients move to the state before the step
#pragma unroll
            for (int i = 0; i < HS; ++i) {
                R[j][i].x = dr * F[i];
                dF[i] = sh.A[i] * dF[i] + sh.E[i] * dr;
                if (MOM) dS[(MOM ? i : 0)] *= M;
            }
            __syncthreads();
            float r5 = 0.0f;
#pragma unroll 8
            for (int jj = 0; jj < HS; ++jj) r5 += R[jj][j].x;
            // row j's gradients (r4 = -dL/d(ae_j))
            // v45 W1 (KDA): E = a e and AE = b e, so dE adds r5 e to dalpha and r5 a to de; dAE gives -r4 b and -r4 e
            const float dalpha = KDA ? r2 + r5 * x.e : r2 - r4 * x.b * x.e;
            const float dbr = KDA ? r3 * x.k - r4 * x.e : r3 * x.k - r4 * x.a * x.e;
            float* dq = dqke + ((size_t)bh * T + t) * LQ;
            float* dg = dvg + ((size_t)bh * T + t) * LG;
            dq[j] = r1;
            dq[HS + j] = r3 * x.b;
            dq[2 * HS + j] = KDA ? r5 * x.a - r4 * x.b : r5 - r4 * x.a * x.b;
            dg[j] = dv;
            if (PC) {
                dg[HS + j] = dalpha * x.a;
                dg[2 * HS + j] = dbr;
            } else {
                red[0][j] = dalpha * x.a;
                red[1][j] = dbr;
            }
            __syncthreads();
            if (!PC && j == 0) {
                float sa = 0.0f, sbb = 0.0f;
                for (int i = 0; i < HS; ++i) {
                    sa += red[0][i];
                    sbb += red[1][i];
                }
                dg[HS] = sa;
                dg[HS + 1] = sbb;
            }
        }
    }
}
#define MEMG_KERNELS(NAME, HS, PC, MOM, KDA)                                                                                \
    extern "C" __global__ void __launch_bounds__(HS) memg_fwd_##NAME(const float* __restrict__ qke, const float* __restrict__ vg, \
                                                                     float* __restrict__ y, int BH, int T, float M,         \
                                                                     float cap) {                                           \
        memg_fwd_impl<HS, PC, MOM, KDA>(qke, vg, y, BH, T, M, cap);                                                         \
    }                                                                                                                       \
    extern "C" __global__ void __launch_bounds__(HS) memg_bwd_##NAME(                                                      \
        const float* __restrict__ qke, const float* __restrict__ vg, const float* __restrict__ dy, float* __restrict__ dqke, \
        float* __restrict__ dvg, float* __restrict__ ck, float* __restrict__ scr, int BH, int T, float M, int C,               \
        float cap) {                                                                                                        \
        memg_bwd_impl<HS, PC, MOM, KDA>(qke, vg, dy, dqke, dvg, ck, scr, BH, T, M, C >= 1 ? C : MG_C, cap);                 \
    }
MEMG_KERNELS(32h0, 32, false, false, false)
MEMG_KERNELS(32h1, 32, false, true, false)
MEMG_KERNELS(32c0, 32, true, false, false)
MEMG_KERNELS(32c1, 32, true, true, false)
MEMG_KERNELS(64h0, 64, false, false, false)
MEMG_KERNELS(64h1, 64, false, true, false)
MEMG_KERNELS(64c0, 64, true, false, false)
MEMG_KERNELS(64c1, 64, true, true, false)
// v45 W1 (--gates kda): per-channel fades, KDA's keys
MEMG_KERNELS(32k0, 32, true, false, true)
MEMG_KERNELS(32k1, 32, true, true, true)
MEMG_KERNELS(64k0, 64, true, false, true)
MEMG_KERNELS(64k1, 64, true, true, true)
// v47 Amendment 3: memg_bwd4, the scan backward with each column's rows split over 4 threads. Thread tid = 4 j + p owns
// rows i = 4 ii + p of column j (HS / 4 of them), so a head has 4 HS threads and a quarter of the registers each. The
// column sums (r, dv, dr) are 4-lane shuffles; the recomputed r keeps memg_dot's order (rows by i mod 4, then
// (p0 + p1) + (p2 + p3)), so the states are the old kernel's bits. The five row sums over columns go through two
// shared buffers [HS][HS + 8] (conflict-free writes and reads): 8 lanes per row each add 8 columns, then a 3-step
// shuffle. ck / scr keep their sizes; the element (ii, tid) of a state sits at ii * 4 HS + tid.
template <int HS, bool PC, bool KDA>
__device__ __forceinline__ void memg4_load(const float* __restrict__ qr, const float* __restrict__ gr, MgSh<HS>& sh,
                                           float (&xr)[4][HS], int tid) {
    if (tid < HS) {
        MgIn<HS, PC> x;
        x.load(qr, gr, tid);
        memg_share<HS, PC, KDA>(x, sh, tid);
        xr[0][tid] = x.k;
        xr[1][tid] = x.e;
        xr[2][tid] = x.a;
        xr[3][tid] = x.b;
    }
}
template <int HS>
__device__ __forceinline__ float memg4_r(const float* __restrict__ s, const float (&F)[HS / 4], int p) {
    float a = 0.0f;
#pragma unroll
    for (int ii = 0; ii < HS / 4; ++ii) a += s[4 * ii + p] * F[ii];
    a += __shfl_xor_sync(0xffffffffu, a, 1);
    a += __shfl_xor_sync(0xffffffffu, a, 2);
    return a;
}
// v52 fix_S: TRK = also note in `sat` whether the bound held any element of F this step (memgk_scan's forward walk)
template <int HS, bool MOM, bool TRK>
__device__ __forceinline__ void memg4_step_t(float (&F)[HS / 4], float (&S)[MOM ? HS / 4 : 1], const MgSh<HS>& sh, float v,
                                             float M, int p, float cap, bool& sat) {
    const float r = memg4_r<HS>(sh.E, F, p);
#pragma unroll
    for (int ii = 0; ii < HS / 4; ++ii) {
        const int i = 4 * ii + p;
        const float u = sh.W[i] * v - sh.AE[i] * r;
        float fn;
        if (MOM) {
            S[(MOM ? ii : 0)] = M * S[(MOM ? ii : 0)] + u;
            fn = sh.A[i] * F[ii] + S[(MOM ? ii : 0)];
        } else {
            fn = sh.A[i] * F[ii] + u;
        }
        if (TRK) sat |= !(fabsf(fn) < cap);
        F[ii] = mg_cap(fn, cap);
    }
}
template <int HS, bool MOM>
__device__ __forceinline__ void memg4_step(float (&F)[HS / 4], float (&S)[MOM ? HS / 4 : 1], const MgSh<HS>& sh, float v,
                                           float M, int p, float cap) {
    bool unused = false;
    memg4_step_t<HS, MOM, false>(F, S, sh, v, M, p, cap, unused);
}
// the row sums: lane (g, sl) of warp w adds columns sl + 8 m of rows 4 (2 w + h) + g, h = 0, 1; all 8 lanes get the total
template <int HS>
__device__ __forceinline__ void memg4_rows(const float* __restrict__ B, float (&o)[2], int w, int g, int sl) {
    constexpr int LD = HS + 8;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const float* b = B + (4 * (2 * w + h) + g) * LD + sl;
        float a = 0.0f;
#pragma unroll
        for (int m = 0; m < HS / 8; ++m) a += b[8 * m];
        a += __shfl_xor_sync(0xffffffffu, a, 1);
        a += __shfl_xor_sync(0xffffffffu, a, 2);
        a += __shfl_xor_sync(0xffffffffu, a, 4);
        o[h] = a;
    }
}
template <int HS, bool PC, bool MOM, bool KDA>
__device__ __forceinline__ void memg4_bwd_impl(const float* __restrict__ qke, const float* __restrict__ vg, const float* __restrict__ dy,
                                               float* __restrict__ dqke, float* __restrict__ dvg, float* __restrict__ ck,
                                               float* __restrict__ scr, int BH, int T, float M, int C, float cap) {
    constexpr int NT = 4 * HS, RP = HS / 4, LD = HS + 8, NW = NT / 32;
    constexpr int NS = MOM ? 2 : 1;
    constexpr size_t SQ = (size_t)HS * HS;
    __shared__ MgSh<HS> sh;
    __shared__ float xr[4][HS]; // the step's k, e, alpha, b by row
    __shared__ float B0[HS * LD], B1[HS * LD];
    __shared__ float red[2][NW];
    const int tid = threadIdx.x, bh = blockIdx.x;
    if (bh >= BH) return;
    const int j = tid >> 2, p = tid & 3, w = tid >> 5, g = (tid & 31) >> 3, sl = tid & 7;
    const int G = PC ? HS : 1, LQ = 3 * HS, LG = HS + 2 * G;
    const int nck = (T + C - 1) / C;
    const float* qb = qke + (size_t)bh * T * LQ;
    const float* gb = vg + (size_t)bh * T * LG;
    float* cb = ck + (size_t)bh * nck * NS * SQ + tid;
    float* sb = scr + (size_t)bh * C * NS * SQ + tid;
    float F[RP], S[MOM ? RP : 1];
#pragma unroll
    for (int ii = 0; ii < RP; ++ii) F[ii] = 0.0f;
#pragma unroll
    for (int ii = 0; ii < (MOM ? RP : 1); ++ii) S[ii] = 0.0f;
    // 1. the forward, saving the state at the start of every chunk
    for (int t = 0; t < T; ++t) {
        if (t % C == 0) {
            float* c = cb + (size_t)(t / C) * NS * SQ;
#pragma unroll
            for (int ii = 0; ii < RP; ++ii) c[ii * NT] = F[ii];
            if (MOM) {
#pragma unroll
                for (int ii = 0; ii < (MOM ? RP : 1); ++ii) c[SQ + ii * NT] = S[ii];
            }
        }
        memg4_load<HS, PC, KDA>(qb + (size_t)t * LQ, gb + (size_t)t * LG, sh, xr, tid);
        const float v = gb[(size_t)t * LG + j];
        __syncthreads();
        memg4_step<HS, MOM>(F, S, sh, v, M, p, cap);
        __syncthreads();
    }
    // 2. chunk by chunk, last first
    float dF[RP], dS[MOM ? RP : 1];
#pragma unroll
    for (int ii = 0; ii < RP; ++ii) dF[ii] = 0.0f;
#pragma unroll
    for (int ii = 0; ii < (MOM ? RP : 1); ++ii) dS[ii] = 0.0f;
    for (int c = nck - 1; c >= 0; --c) {
        const int t0 = c * C, t1 = min(T, t0 + C);
        {
            const float* cc = cb + (size_t)c * NS * SQ;
#pragma unroll
            for (int ii = 0; ii < RP; ++ii) F[ii] = cc[ii * NT];
            if (MOM) {
#pragma unroll
                for (int ii = 0; ii < (MOM ? RP : 1); ++ii) S[ii] = cc[SQ + ii * NT];
            }
        }
        for (int t = t0; t < t1; ++t) {
            float* s = sb + (size_t)(t - t0) * NS * SQ;
#pragma unroll
            for (int ii = 0; ii < RP; ++ii) s[ii * NT] = F[ii];
            if (MOM) {
#pragma unroll
                for (int ii = 0; ii < (MOM ? RP : 1); ++ii) s[SQ + ii * NT] = S[ii];
            }
            memg4_load<HS, PC, KDA>(qb + (size_t)t * LQ, gb + (size_t)t * LG, sh, xr, tid);
            const float v = gb[(size_t)t * LG + j];
            __syncthreads();
            memg4_step<HS, MOM>(F, S, sh, v, M, p, cap);
            __syncthreads();
        }
        for (int t = t1 - 1; t >= t0; --t) {
            const float* s = sb + (size_t)(t - t0) * NS * SQ;
#pragma unroll
            for (int ii = 0; ii < RP; ++ii) F[ii] = s[ii * NT]; // the state before step t
            memg4_load<HS, PC, KDA>(qb + (size_t)t * LQ, gb + (size_t)t * LG, sh, xr, tid);
            const float v = gb[(size_t)t * LG + j];
            const float dyj = dy[((size_t)bh * T + t) * HS + j];
            __syncthreads();
            const float r = memg4_r<HS>(sh.E, F, p);
            // dF' = carried + q dy^T; du = dS' (momentum) or dF'; round 1 of the row sums: (F'_ij dy_j, dF'_ij F_ij)
            float dv = 0.0f, dr = 0.0f;
#pragma unroll
            for (int ii = 0; ii < RP; ++ii) {
                const int i = 4 * ii + p;
                const float u = sh.W[i] * v - sh.AE[i] * r;
                float fn;
                if (MOM) fn = sh.A[i] * F[ii] + (M * s[SQ + ii * NT] + u);
                else fn = sh.A[i] * F[ii] + u;
                dF[ii] += sh.Q[i] * dyj;
                if (!(fabsf(fn) < cap)) dF[ii] = 0.0f; // v52 fix_S: the bound holds F here, so no gradient flows through it
                fn = mg_cap(fn, cap);
                B0[i * LD + j] = fn * dyj;
                B1[i * LD + j] = dF[ii] * F[ii];
                float du;
                if (MOM) {
                    dS[(MOM ? ii : 0)] += dF[ii];
                    du = dS[(MOM ? ii : 0)];
                } else {
                    du = dF[ii];
                }
                dv += du * sh.W[i];
                dr -= du * sh.AE[i];
            }
            dv += __shfl_xor_sync(0xffffffffu, dv, 1);
            dv += __shfl_xor_sync(0xffffffffu, dv, 2);
            dr += __shfl_xor_sync(0xffffffffu, dr, 1);
            dr += __shfl_xor_sync(0xffffffffu, dr, 2);
            // this lane's two rows' k, e, alpha, b (read now: the next step rewrites xr after the last barrier here)
            float rk[2], re[2], ra[2], rb[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int i = 4 * (2 * w + h) + g;
                rk[h] = xr[0][i];
                re[h] = xr[1][i];
                ra[h] = xr[2][i];
                rb[h] = xr[3][i];
            }
            __syncthreads();
            float r1[2], r2[2], r3[2], r4[2], r5[2];
            memg4_rows<HS>(B0, r1, w, g, sl);
            memg4_rows<HS>(B1, r2, w, g, sl);
            __syncthreads();
            // round 2: (du_ij v_j, du_ij r_j)
#pragma unroll
            for (int ii = 0; ii < RP; ++ii) {
                const int i = 4 * ii + p;
                const float du = MOM ? dS[(MOM ? ii : 0)] : dF[ii];
                B0[i * LD + j] = du * v;
                B1[i * LD + j] = du * r;
            }
            __syncthreads();
            memg4_rows<HS>(B0, r3, w, g, sl);
            memg4_rows<HS>(B1, r4, w, g, sl);
            __syncthreads();
            // round 3: dr_j F_ij; then the carried gradients move to the state before the step
#pragma unroll
            for (int ii = 0; ii < RP; ++ii) {
                const int i = 4 * ii + p;
                B0[i * LD + j] = dr * F[ii];
                dF[ii] = sh.A[i] * dF[ii] + sh.E[i] * dr;
                if (MOM) dS[(MOM ? ii : 0)] *= M;
            }
            __syncthreads();
            memg4_rows<HS>(B0, r5, w, g, sl);
            float* dq = dqke + ((size_t)bh * T + t) * LQ;
            float* dg = dvg + ((size_t)bh * T + t) * LG;
            if (p == 0) dg[j] = dv;
            float sa = 0.0f, sbb = 0.0f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                // row i's gradients (r4 = -dL/d(ae_i)); as memg_bwd_impl
                const int i = 4 * (2 * w + h) + g;
                const float dalpha = KDA ? r2[h] + r5[h] * re[h] : r2[h] - r4[h] * rb[h] * re[h];
                const float dbr = KDA ? r3[h] * rk[h] - r4[h] * re[h] : r3[h] * rk[h] - r4[h] * ra[h] * re[h];
                if (sl == 0) {
                    dq[i] = r1[h];
                    dq[HS + i] = r3[h] * rb[h];
                    dq[2 * HS + i] = KDA ? r5[h] * ra[h] - r4[h] * rb[h] : r5[h] - r4[h] * ra[h] * rb[h];
                    if (PC) {
                        dg[HS + i] = dalpha * ra[h];
                        dg[2 * HS + i] = dbr;
                    }
                }
                sa += dalpha * ra[h];
                sbb += dbr;
            }
            if (!PC) {
                // per head: the sum over rows (lanes sl = 0 of each row group carry it)
                if (sl != 0) sa = sbb = 0.0f;
                sa += __shfl_xor_sync(0xffffffffu, sa, 8);
                sa += __shfl_xor_sync(0xffffffffu, sa, 16);
                sbb += __shfl_xor_sync(0xffffffffu, sbb, 8);
                sbb += __shfl_xor_sync(0xffffffffu, sbb, 16);
                if ((tid & 31) == 0) {
                    red[0][w] = sa;
                    red[1][w] = sbb;
                }
                __syncthreads();
                if (tid == 0) {
                    float ta = 0.0f, tb = 0.0f;
                    for (int q = 0; q < NW; ++q) {
                        ta += red[0][q];
                        tb += red[1][q];
                    }
                    dg[HS] = ta;
                    dg[HS + 1] = tb;
                }
            }
        }
    }
}
#define MEMG4_KERNELS(NAME, HS, PC, MOM, KDA)                                                                               \
    extern "C" __global__ void __launch_bounds__(4 * HS, 2) memg_bwd4_##NAME(                                              \
        const float* __restrict__ qke, const float* __restrict__ vg, const float* __restrict__ dy, float* __restrict__ dqke, \
        float* __restrict__ dvg, float* __restrict__ ck, float* __restrict__ scr, int BH, int T, float M, int C,          \
        float cap) {                                                                                                        \
        memg4_bwd_impl<HS, PC, MOM, KDA>(qke, vg, dy, dqke, dvg, ck, scr, BH, T, M, C >= 1 ? C : MG_C, cap);              \
    }
MEMG4_KERNELS(32h0, 32, false, false, false)
MEMG4_KERNELS(32h1, 32, false, true, false)
MEMG4_KERNELS(32c0, 32, true, false, false)
MEMG4_KERNELS(32c1, 32, true, true, false)
MEMG4_KERNELS(64h0, 64, false, false, false)
MEMG4_KERNELS(64h1, 64, false, true, false)
MEMG4_KERNELS(64c0, 64, true, false, false)
MEMG4_KERNELS(64c1, 64, true, true, false)
MEMG4_KERNELS(32k0, 32, true, false, true)
MEMG4_KERNELS(32k1, 32, true, true, true)
MEMG4_KERNELS(64k0, 64, true, false, true)
MEMG4_KERNELS(64k1, 64, true, true, true)

// ---------------------------------------------------------------------------------------------------------------------
// v49 (runs/v47/prereg_fast_winners.md, Amendment 6): the memg scan split in time, with memg_fwd's and memg_bwd4's bits.
// memg_bwd4 walks one head's whole sequence in one block (24 blocks at ctx 2048 batch 4, 6 at 1 x 8192: most of the GPU
// idle, every step's latency exposed). The rule is linear in the state, so the carried gradients (dF, dS) follow the
// transposed rule alone: they need the step inputs and dy, not the forward's states. 4 HS threads per head throughout
// (memg_bwd4's layout: thread tid = 4 j + p owns rows i = 4 ii + p of column j):
//   memgk_scan  2 BH blocks, both walks at once: blocks < BH rerun the forward from zero (memg4_step, memg_bwd4's phase
//               1) saving (F, S) at the start of every item of L steps into ck; blocks >= BH walk (dF, dS) back from the
//               end with memg_bwd4's own update ops, saving them as they enter every item (before its last step) in ca.
//   memgk_bwd   persistent blocks, one (head, item) pair after another: load the item's (F, S) and (dF, dS); with L > C
//               rerun the item saving (F, S) every C steps in the block's own c2; then each C-step piece, last first,
//               exactly as memg_bwd4's phase 2 (recompute into the block's scratch, walk back with its per-step code).
//   memgk_fwd   the forward with memg4_step and memg4_r (memg_fwd's sums in its order), so y is memg_fwd's bits.
// The next step's inputs are loaded while the current step runs (no sum changes). memgk_fwd and memgk_scan split a head's
// columns over NCG blocks (the columns are independent recurrences; the same ops per column). ck, ca [BH, ceil(T/L), NS, HS, HS];
// scr [gridDim, L/C + C, NS, HS, HS]; a state's element (ii, tid) sits at ii * 4 HS + tid, as in memg_bwd4.
template <int HS, bool PC>
struct MgkIn {
    MgIn<HS, PC> x; // channel tid's inputs (threads tid < HS)
    float v, dy;    // column j's v and dy
};
template <int HS, bool PC>
__device__ __forceinline__ void mgk_load(MgkIn<HS, PC>& n, const float* __restrict__ qb, const float* __restrict__ gb,
                                         const float* __restrict__ dyb, int t, int tid, int j, bool wdy) {
    constexpr int G = PC ? HS : 1, LQ = 3 * HS, LG = HS + 2 * G;
    if (tid < HS) n.x.load(qb + (size_t)t * LQ, gb + (size_t)t * LG, tid);
    n.v = gb[(size_t)t * LG + j];
    if (wdy) n.dy = dyb[(size_t)t * HS + j];
}
// memg4_load's shared writes (xr only when XR) from inputs loaded ahead
template <int HS, bool PC, bool KDA, bool XR>
__device__ __forceinline__ void mgk_share(const MgkIn<HS, PC>& n, MgSh<HS>& sh, float (&xr)[4][HS], int tid) {
    if (tid < HS) {
        memg_share<HS, PC, KDA>(n.x, sh, tid);
        if (XR) {
            xr[0][tid] = n.x.k;
            xr[1][tid] = n.x.e;
            xr[2][tid] = n.x.a;
            xr[3][tid] = n.x.b;
        }
    }
}
template <int HS, bool MOM>
__device__ __forceinline__ void mgk_put(float* __restrict__ c, const float (&X)[HS / 4], const float (&Y)[MOM ? HS / 4 : 1]) {
    constexpr int NT = 4 * HS;
    constexpr size_t SQ = (size_t)HS * HS;
#pragma unroll
    for (int ii = 0; ii < HS / 4; ++ii) c[ii * NT] = X[ii];
    if (MOM) {
#pragma unroll
        for (int ii = 0; ii < (MOM ? HS / 4 : 1); ++ii) c[SQ + ii * NT] = Y[ii];
    }
}
template <int HS, bool MOM>
__device__ __forceinline__ void mgk_get(const float* __restrict__ c, float (&X)[HS / 4], float (&Y)[MOM ? HS / 4 : 1]) {
    constexpr int NT = 4 * HS;
    constexpr size_t SQ = (size_t)HS * HS;
#pragma unroll
    for (int ii = 0; ii < HS / 4; ++ii) X[ii] = c[ii * NT];
    if (MOM) {
#pragma unroll
        for (int ii = 0; ii < (MOM ? HS / 4 : 1); ++ii) Y[ii] = c[SQ + ii * NT];
    }
}
template <int HS, bool PC, bool MOM, bool KDA>
__device__ __forceinline__ void memgk_fwd_impl(const float* __restrict__ qke, const float* __restrict__ vg, float* __restrict__ y,
                                               int BH, int T, float M, int NCG, float cap) {
    constexpr int RP = HS / 4, G = PC ? HS : 1, LQ = 3 * HS, LG = HS + 2 * G;
    __shared__ MgSh<HS> sh;
    __shared__ float xr[4][HS];
    // NCG blocks per head, HS / NCG columns each (columns are independent: the same ops per column)
    const int tid = threadIdx.x, bh = blockIdx.x / NCG, j = (blockIdx.x % NCG) * (HS / NCG) + (tid >> 2), p = tid & 3;
    if (bh >= BH) return;
    const float* qb = qke + (size_t)bh * T * LQ;
    const float* gb = vg + (size_t)bh * T * LG;
    float F[RP], S[MOM ? RP : 1];
#pragma unroll
    for (int ii = 0; ii < RP; ++ii) F[ii] = 0.0f;
#pragma unroll
    for (int ii = 0; ii < (MOM ? RP : 1); ++ii) S[ii] = 0.0f;
    MgkIn<HS, PC> nx;
    mgk_load<HS, PC>(nx, qb, gb, nullptr, 0, tid, j, false);
    for (int t = 0; t < T; ++t) {
        mgk_share<HS, PC, KDA, false>(nx, sh, xr, tid);
        const float v = nx.v;
        if (t + 1 < T) mgk_load<HS, PC>(nx, qb, gb, nullptr, t + 1, tid, j, false);
        __syncthreads();
        memg4_step<HS, MOM>(F, S, sh, v, M, p, cap);
        const float yv = memg4_r<HS>(sh.Q, F, p);
        if (p == 0) y[((size_t)bh * T + t) * HS + j] = yv;
        __syncthreads();
    }
}
template <int HS, bool PC, bool MOM, bool KDA>
__device__ __forceinline__ void memgk_scan_impl(const float* __restrict__ qke, const float* __restrict__ vg, const float* __restrict__ dy,
                                                float* __restrict__ ck, float* __restrict__ ca, int BH, int T, float M, int L,
                                                int NCG, float cap, float* __restrict__ sat) {
    constexpr int RP = HS / 4, NS = MOM ? 2 : 1, G = PC ? HS : 1, LQ = 3 * HS, LG = HS + 2 * G;
    constexpr size_t SQ = (size_t)HS * HS;
    __shared__ MgSh<HS> sh;
    __shared__ float xr[4][HS];
    // NCG blocks per head and walk, HS / NCG columns each; gt = 4 j + p, the thread's place in memg_bwd4's layout
    const int tid = threadIdx.x, p = tid & 3, blk = blockIdx.x / NCG, cg = blockIdx.x % NCG;
    const int j = cg * (HS / NCG) + (tid >> 2), gt = 4 * j + p;
    const bool adj = blk >= BH;
    const int bh = adj ? blk - BH : blk;
    if (bh >= BH) return;
    const int nit = (T + L - 1) / L;
    const float* qb = qke + (size_t)bh * T * LQ;
    const float* gb = vg + (size_t)bh * T * LG;
    const float* dyb = dy + (size_t)bh * T * HS;
    float X[RP], Y[MOM ? RP : 1];
#pragma unroll
    for (int ii = 0; ii < RP; ++ii) X[ii] = 0.0f;
#pragma unroll
    for (int ii = 0; ii < (MOM ? RP : 1); ++ii) Y[ii] = 0.0f;
    MgkIn<HS, PC> nx;
    if (!adj) {
        // the forward (F, S) from zero, saved at the start of every item
        float* cb = ck + (size_t)bh * nit * NS * SQ + gt;
        mgk_load<HS, PC>(nx, qb, gb, dyb, 0, tid, j, false);
        bool hit = false;
        for (int t = 0; t < T; ++t) {
            if (t % L == 0) mgk_put<HS, MOM>(cb + (size_t)(t / L) * NS * SQ, X, Y);
            mgk_share<HS, PC, KDA, false>(nx, sh, xr, tid);
            const float v = nx.v;
            if (t + 1 < T) mgk_load<HS, PC>(nx, qb, gb, dyb, t + 1, tid, j, false);
            __syncthreads();
            memg4_step_t<HS, MOM, true>(X, Y, sh, v, M, p, cap, hit);
            __syncthreads();
        }
        // v52 fix_S: whether the bound held any element of this block's columns. The adjoint walk below cannot zero the
        // gradient where it did (it runs without the states), so the host then runs memg_bwd4 instead (fwin_ops::memg_bwd_k).
        const int any = __syncthreads_or(hit);
        if (tid == 0) sat[blockIdx.x] = any ? 1.0f : 0.0f;
    } else {
        // the carried gradients (dF, dS) from the end, saved as they enter every item: memg_bwd4's ops for them only
        float* cb = ca + (size_t)bh * nit * NS * SQ + gt;
        mgk_load<HS, PC>(nx, qb, gb, dyb, T - 1, tid, j, true);
        for (int t = T - 1; t >= 0; --t) {
            if (t == T - 1 || (t + 1) % L == 0) mgk_put<HS, MOM>(cb + (size_t)(t / L) * NS * SQ, X, Y);
            mgk_share<HS, PC, KDA, false>(nx, sh, xr, tid);
            const float dyj = nx.dy;
            if (t > 0) mgk_load<HS, PC>(nx, qb, gb, dyb, t - 1, tid, j, true);
            __syncthreads();
            float dr = 0.0f;
#pragma unroll
            for (int ii = 0; ii < RP; ++ii) {
                const int i = 4 * ii + p;
                X[ii] += sh.Q[i] * dyj;
                float du;
                if (MOM) {
                    Y[(MOM ? ii : 0)] += X[ii];
                    du = Y[(MOM ? ii : 0)];
                } else {
                    du = X[ii];
                }
                dr -= du * sh.AE[i];
            }
            dr += __shfl_xor_sync(0xffffffffu, dr, 1);
            dr += __shfl_xor_sync(0xffffffffu, dr, 2);
#pragma unroll
            for (int ii = 0; ii < RP; ++ii) {
                const int i = 4 * ii + p;
                X[ii] = sh.A[i] * X[ii] + sh.E[i] * dr;
                if (MOM) Y[(MOM ? ii : 0)] *= M;
            }
            __syncthreads();
        }
    }
}
template <int HS, bool PC, bool MOM, bool KDA>
__device__ __forceinline__ void memgk_bwd_impl(const float* __restrict__ qke, const float* __restrict__ vg, const float* __restrict__ dy,
                                               float* __restrict__ dqke, float* __restrict__ dvg, const float* __restrict__ ck,
                                               const float* __restrict__ ca, float* __restrict__ scr, int BH, int T, float M,
                                               int L, int C, float cap) {
    constexpr int NT = 4 * HS, RP = HS / 4, LD = HS + 8, NW = NT / 32;
    constexpr int NS = MOM ? 2 : 1, G = PC ? HS : 1, LQ = 3 * HS, LG = HS + 2 * G;
    constexpr size_t SQ = (size_t)HS * HS;
    __shared__ MgSh<HS> sh;
    __shared__ float xr[4][HS]; // the step's k, e, alpha, b by row
    __shared__ float B0[HS * LD], B1[HS * LD];
    __shared__ float red[2][NW];
    const int tid = threadIdx.x;
    const int j = tid >> 2, p = tid & 3, w = tid >> 5, g = (tid & 31) >> 3, sl = tid & 7;
    const int nit = (T + L - 1) / L, npc = L / C;
    float* c2 = scr + (size_t)blockIdx.x * (npc + C) * NS * SQ + tid; // the item's pieces' starts (L > C)
    float* sb = c2 + (size_t)npc * NS * SQ;                            // one piece's states
    for (int it = blockIdx.x; it < BH * nit; it += gridDim.x) {
        const int bh = it % BH, c = it / BH;
        const int t0 = c * L, t1 = min(T, t0 + L);
        const float* qb = qke + (size_t)bh * T * LQ;
        const float* gb = vg + (size_t)bh * T * LG;
        const float* dyb = dy + (size_t)bh * T * HS;
        float F[RP], S[MOM ? RP : 1], dF[RP], dS[MOM ? RP : 1];
        mgk_get<HS, MOM>(ck + ((size_t)bh * nit + c) * NS * SQ + tid, F, S);
        mgk_get<HS, MOM>(ca + ((size_t)bh * nit + c) * NS * SQ + tid, dF, dS);
        MgkIn<HS, PC> nx;
        if (npc > 1) {
            mgk_load<HS, PC>(nx, qb, gb, dyb, t0, tid, j, false);
            for (int t = t0; t < t1; ++t) {
                if ((t - t0) % C == 0) mgk_put<HS, MOM>(c2 + (size_t)((t - t0) / C) * NS * SQ, F, S);
                mgk_share<HS, PC, KDA, false>(nx, sh, xr, tid);
                const float v = nx.v;
                if (t + 1 < t1) mgk_load<HS, PC>(nx, qb, gb, dyb, t + 1, tid, j, false);
                __syncthreads();
                memg4_step<HS, MOM>(F, S, sh, v, M, p, cap);
                __syncthreads();
            }
        }
        const int np = (t1 - t0 + C - 1) / C;
        for (int pc = np - 1; pc >= 0; --pc) {
            const int u0 = t0 + pc * C, u1 = min(t1, u0 + C);
            if (npc > 1) mgk_get<HS, MOM>(c2 + (size_t)pc * NS * SQ, F, S);
            // memg_bwd4's phase 2 on this piece: its states into the scratch, then the walk back
            mgk_load<HS, PC>(nx, qb, gb, dyb, u0, tid, j, false);
            for (int t = u0; t < u1; ++t) {
                mgk_put<HS, MOM>(sb + (size_t)(t - u0) * NS * SQ, F, S);
                mgk_share<HS, PC, KDA, false>(nx, sh, xr, tid);
                const float v = nx.v;
                if (t + 1 < u1) mgk_load<HS, PC>(nx, qb, gb, dyb, t + 1, tid, j, false);
                __syncthreads();
                memg4_step<HS, MOM>(F, S, sh, v, M, p, cap);
                __syncthreads();
            }
            mgk_load<HS, PC>(nx, qb, gb, dyb, u1 - 1, tid, j, true);
            for (int t = u1 - 1; t >= u0; --t) {
                const float* s = sb + (size_t)(t - u0) * NS * SQ;
#pragma unroll
                for (int ii = 0; ii < RP; ++ii) F[ii] = s[ii * NT]; // the state before step t
                mgk_share<HS, PC, KDA, true>(nx, sh, xr, tid);
                const float v = nx.v, dyj = nx.dy;
                if (t > u0) mgk_load<HS, PC>(nx, qb, gb, dyb, t - 1, tid, j, true);
                __syncthreads();
                // from here to the end of the step: memg_bwd4's code
                const float r = memg4_r<HS>(sh.E, F, p);
                float dv = 0.0f, dr = 0.0f;
#pragma unroll
                for (int ii = 0; ii < RP; ++ii) {
                    const int i = 4 * ii + p;
                    const float u = sh.W[i] * v - sh.AE[i] * r;
                    float fn;
                    if (MOM) fn = sh.A[i] * F[ii] + (M * s[SQ + ii * NT] + u);
                    else fn = sh.A[i] * F[ii] + u;
                    dF[ii] += sh.Q[i] * dyj;
                    if (!(fabsf(fn) < cap)) dF[ii] = 0.0f; // v52 fix_S: the bound holds F here, so no gradient flows through it
                    fn = mg_cap(fn, cap);
                    B0[i * LD + j] = fn * dyj;
                    B1[i * LD + j] = dF[ii] * F[ii];
                    float du;
                    if (MOM) {
                        dS[(MOM ? ii : 0)] += dF[ii];
                        du = dS[(MOM ? ii : 0)];
                    } else {
                        du = dF[ii];
                    }
                    dv += du * sh.W[i];
                    dr -= du * sh.AE[i];
                }
                dv += __shfl_xor_sync(0xffffffffu, dv, 1);
                dv += __shfl_xor_sync(0xffffffffu, dv, 2);
                dr += __shfl_xor_sync(0xffffffffu, dr, 1);
                dr += __shfl_xor_sync(0xffffffffu, dr, 2);
                float rk[2], re[2], ra[2], rb[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int i = 4 * (2 * w + h) + g;
                    rk[h] = xr[0][i];
                    re[h] = xr[1][i];
                    ra[h] = xr[2][i];
                    rb[h] = xr[3][i];
                }
                __syncthreads();
                float r1[2], r2[2], r3[2], r4[2], r5[2];
                memg4_rows<HS>(B0, r1, w, g, sl);
                memg4_rows<HS>(B1, r2, w, g, sl);
                __syncthreads();
#pragma unroll
                for (int ii = 0; ii < RP; ++ii) {
                    const int i = 4 * ii + p;
                    const float du = MOM ? dS[(MOM ? ii : 0)] : dF[ii];
                    B0[i * LD + j] = du * v;
                    B1[i * LD + j] = du * r;
                }
                __syncthreads();
                memg4_rows<HS>(B0, r3, w, g, sl);
                memg4_rows<HS>(B1, r4, w, g, sl);
                __syncthreads();
#pragma unroll
                for (int ii = 0; ii < RP; ++ii) {
                    const int i = 4 * ii + p;
                    B0[i * LD + j] = dr * F[ii];
                    dF[ii] = sh.A[i] * dF[ii] + sh.E[i] * dr;
                    if (MOM) dS[(MOM ? ii : 0)] *= M;
                }
                __syncthreads();
                memg4_rows<HS>(B0, r5, w, g, sl);
                float* dq = dqke + ((size_t)bh * T + t) * LQ;
                float* dg = dvg + ((size_t)bh * T + t) * LG;
                if (p == 0) dg[j] = dv;
                float sa = 0.0f, sbb = 0.0f;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int i = 4 * (2 * w + h) + g;
                    const float dalpha = KDA ? r2[h] + r5[h] * re[h] : r2[h] - r4[h] * rb[h] * re[h];
                    const float dbr = KDA ? r3[h] * rk[h] - r4[h] * re[h] : r3[h] * rk[h] - r4[h] * ra[h] * re[h];
                    if (sl == 0) {
                        dq[i] = r1[h];
                        dq[HS + i] = r3[h] * rb[h];
                        dq[2 * HS + i] = KDA ? r5[h] * ra[h] - r4[h] * rb[h] : r5[h] - r4[h] * ra[h] * rb[h];
                        if (PC) {
                            dg[HS + i] = dalpha * ra[h];
                            dg[2 * HS + i] = dbr;
                        }
                    }
                    sa += dalpha * ra[h];
                    sbb += dbr;
                }
                if (!PC) {
                    if (sl != 0) sa = sbb = 0.0f;
                    sa += __shfl_xor_sync(0xffffffffu, sa, 8);
                    sa += __shfl_xor_sync(0xffffffffu, sa, 16);
                    sbb += __shfl_xor_sync(0xffffffffu, sbb, 8);
                    sbb += __shfl_xor_sync(0xffffffffu, sbb, 16);
                    if ((tid & 31) == 0) {
                        red[0][w] = sa;
                        red[1][w] = sbb;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        float ta = 0.0f, tb = 0.0f;
                        for (int q = 0; q < NW; ++q) {
                            ta += red[0][q];
                            tb += red[1][q];
                        }
                        dg[HS] = ta;
                        dg[HS + 1] = tb;
                    }
                }
            }
        }
    }
}
#define MEMGK_KERNELS(NAME, HS, PC, MOM, KDA)                                                                               \
    extern "C" __global__ void __launch_bounds__(4 * HS) memgk_fwd_##NAME(const float* __restrict__ qke,                   \
                                                                          const float* __restrict__ vg, float* __restrict__ y, \
                                                                          int BH, int T, float M, int NCG, float cap) {    \
        memgk_fwd_impl<HS, PC, MOM, KDA>(qke, vg, y, BH, T, M, NCG, cap);                                                   \
    }                                                                                                                       \
    extern "C" __global__ void __launch_bounds__(4 * HS) memgk_scan_##NAME(                                                \
        const float* __restrict__ qke, const float* __restrict__ vg, const float* __restrict__ dy, float* __restrict__ ck,  \
        float* __restrict__ ca, int BH, int T, float M, int L, int NCG, float cap, float* __restrict__ sat) {                \
        memgk_scan_impl<HS, PC, MOM, KDA>(qke, vg, dy, ck, ca, BH, T, M, L, NCG, cap, sat);                                 \
    }                                                                                                                       \
    extern "C" __global__ void __launch_bounds__(4 * HS, 2) memgk_bwd_##NAME(                                              \
        const float* __restrict__ qke, const float* __restrict__ vg, const float* __restrict__ dy, float* __restrict__ dqke, \
        float* __restrict__ dvg, const float* __restrict__ ck, const float* __restrict__ ca, float* __restrict__ scr,       \
        int BH, int T, float M, int L, int C, float cap) {                                                                  \
        memgk_bwd_impl<HS, PC, MOM, KDA>(qke, vg, dy, dqke, dvg, ck, ca, scr, BH, T, M, L, C, cap);                         \
    }
MEMGK_KERNELS(32h0, 32, false, false, false)
MEMGK_KERNELS(32h1, 32, false, true, false)
MEMGK_KERNELS(32c0, 32, true, false, false)
MEMGK_KERNELS(32c1, 32, true, true, false)
MEMGK_KERNELS(64h0, 64, false, false, false)
MEMGK_KERNELS(64h1, 64, false, true, false)
MEMGK_KERNELS(64c0, 64, true, false, false)
MEMGK_KERNELS(64c1, 64, true, true, false)

// ---------------------------------------------------------------------------------------------------------------------
// v42 three Wave 5 scans fused (src/scan_fast.rs), one thread per channel, the parts' ops in the same order:
// E7 resonators (ava_parts::resonate::membrane), E9 short-term plasticity (ava_parts::stp), B9 working memory
// (ava_parts::wm::SlotMemory::scan). Backward by hand; each recomputes its forward into its output buffer first.

// E7: re' = (re*rc - im*rs) + I,  im' = re*rs + im*rc;  out = re'. ps = [rc | rs] (each C).
extern "C" __global__ void res_fwd(const float* __restrict__ cur, const float* __restrict__ ps, float* __restrict__ out, int B, int T,
                                   int C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * C) return;
    const int b = i / C, c = i - b * C;
    const float rc = ps[c], rs = ps[C + c];
    const size_t off = (size_t)b * T * C + c;
    float* o = out + off;
    float re = 0.0f, im = 0.0f;
    lif_pf(cur + off, T, C, [&](int t, float x) {
        const float re2 = __fadd_rn(__fsub_rn(__fmul_rn(re, rc), __fmul_rn(im, rs)), x);
        im = __fadd_rn(__fmul_rn(re, rs), __fmul_rn(im, rc));
        re = re2;
        o[(size_t)t * C] = re;
    });
}
// out (flat): dL/dI [B, T, C], then the row's dL/drc, dL/drs [B, C] each. ih: [B, T, C] scratch for Im z.
extern "C" __global__ void res_bwd(const float* __restrict__ cur, const float* __restrict__ ps, const float* __restrict__ g,
                                   float* __restrict__ out, float* __restrict__ ih, int B, int T, int C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * C) return;
    const int b = i / C, c = i - b * C;
    const float rc = ps[c], rs = ps[C + c];
    const size_t off = (size_t)b * T * C + c, BC = (size_t)B * C, BTC = BC * T;
    float* o = out + off;
    float* h = ih + off;
    float re = 0.0f, im = 0.0f;
    lif_pf(cur + off, T, C, [&](int t, float x) {
        const float re2 = __fadd_rn(__fsub_rn(__fmul_rn(re, rc), __fmul_rn(im, rs)), x);
        im = __fadd_rn(__fmul_rn(re, rs), __fmul_rn(im, rc));
        re = re2;
        o[(size_t)t * C] = re;
        h[(size_t)t * C] = im;
    });
    const float* gs = g + off;
    float gr = 0.0f, gi = 0.0f, grc = 0.0f, grs = 0.0f; // gr, gi: dL/dz_{t+1}
    for (int t = T - 1; t >= 0; --t) {
        const float ret = o[(size_t)t * C], imt = h[(size_t)t * C];
        grc += gr * ret + gi * imt;
        grs += gi * ret - gr * imt;
        const float nr = gs[(size_t)t * C] + rc * gr + rs * gi;
        const float ni = rc * gi - rs * gr;
        gr = nr;
        gi = ni;
        o[(size_t)t * C] = gr;
    }
    out[BTC + i] = grc;
    out[BTC + BC + i] = grs;
}

// E9, per channel from u = 0, x = 1:  up = u + ((1 - u)*U)*s;  e = up*x;  eff = e/U;  xu = x - e*s;
//   u <- up*df;  x <- 1 - (1 - xu)*dd.   ps = [U | df | dd] (each C). Output: eff [B, T, C].
__device__ __forceinline__ void stp_step(float s, float U, float df, float dd, float& u, float& x, float& eff) {
    const float up = __fadd_rn(u, __fmul_rn(__fmul_rn(__fsub_rn(1.0f, u), U), s));
    const float e = __fmul_rn(up, x);
    eff = __fdiv_rn(e, U);
    const float xu = __fsub_rn(x, __fmul_rn(e, s));
    u = __fmul_rn(up, df);
    x = __fsub_rn(1.0f, __fmul_rn(__fsub_rn(1.0f, xu), dd));
}
extern "C" __global__ void stp_fwd(const float* __restrict__ s, const float* __restrict__ ps, float* __restrict__ out, int B, int T,
                                   int C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * C) return;
    const int b = i / C, c = i - b * C;
    const float U = ps[c], df = ps[C + c], dd = ps[2 * C + c];
    const size_t off = (size_t)b * T * C + c;
    float* o = out + off;
    float u = 0.0f, x = 1.0f;
    lif_pf(s + off, T, C, [&](int t, float st) {
        float eff;
        stp_step(st, U, df, dd, u, x, eff);
        o[(size_t)t * C] = eff;
    });
}
// out (flat): dL/ds [B, T, C], then the row's dL/dU, dL/ddf, dL/ddd [B, C] each. xh: [B, T, C] scratch.
extern "C" __global__ void stp_bwd(const float* __restrict__ s, const float* __restrict__ ps, const float* __restrict__ g,
                                   float* __restrict__ out, float* __restrict__ xh, int B, int T, int C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * C) return;
    const int b = i / C, c = i - b * C;
    const float U = ps[c], df = ps[C + c], dd = ps[2 * C + c];
    const size_t off = (size_t)b * T * C + c, BC = (size_t)B * C, BTC = BC * T;
    float* o = out + off;
    float* h = xh + off;
    const float* sb = s + off;
    float u = 0.0f, x = 1.0f;
    // the state at the start of every step: u in o, x in h
    lif_pf(sb, T, C, [&](int t, float st) {
        o[(size_t)t * C] = u;
        h[(size_t)t * C] = x;
        float eff;
        stp_step(st, U, df, dd, u, x, eff);
    });
    const float* gs = g + off;
    float gu = 0.0f, gx = 0.0f, gU = 0.0f, gdf = 0.0f, gdd = 0.0f; // gu, gx: dL/du, dL/dx at the next step's start
    for (int t = T - 1; t >= 0; --t) {
        const float ut = o[(size_t)t * C], xt = h[(size_t)t * C], st = sb[(size_t)t * C], ge = gs[(size_t)t * C];
        const float a = 1.0f - ut;
        const float up = ut + a * U * st;
        const float e = up * xt;
        const float xu = xt - e * st;
        const float gxu = gx * dd;
        gdd -= gx * (1.0f - xu);
        float gup = gu * df;
        gdf += gu * up;
        float gxt = gxu;
        float gee = -gxu * st + ge / U;
        float gst = -gxu * e;
        gU -= ge * e / (U * U);
        gup += gee * xt;
        gxt += gee * up;
        gU += gup * a * st;
        gst += gup * a * U;
        gu = gup - gup * U * st;
        gx = gxt;
        o[(size_t)t * C] = gst;
    }
    out[BTC + i] = gU;
    out[BTC + BC + i] = gdf;
    out[BTC + 2 * BC + i] = gdd;
}

// B9: m <- m*(1 - g) + v*g per (b, slot k, channel c); v [B, T, K, DK], g [B, T, K]; out m [B, T, K, DK].
extern "C" __global__ void wm_fwd(const float* __restrict__ v, const float* __restrict__ g, float* __restrict__ out, int B, int T, int K,
                                  int DK) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * K * DK) return;
    const int c = i % DK, k = (i / DK) % K, b = i / (DK * K);
    const size_t row = (size_t)K * DK;
    const float* vb = v + (size_t)b * T * row + (size_t)k * DK + c;
    const float* gb = g + (size_t)b * T * K + k;
    float* o = out + (size_t)b * T * row + (size_t)k * DK + c;
    float m = 0.0f;
    float vn = vb[0], gn = gb[0];
    for (int t = 0; t < T; ++t) {
        const float vt = vn, gt = gn;
        if (t + 1 < T) {
            vn = vb[(size_t)(t + 1) * row];
            gn = gb[(size_t)(t + 1) * K];
        }
        m = __fadd_rn(__fmul_rn(m, __fsub_rn(1.0f, gt)), __fmul_rn(vt, gt));
        o[(size_t)t * row] = m;
    }
}
// out (flat): dL/dv [B, T, K, DK], then each element's share of dL/dg [B, T, K, DK] (summed over DK by the caller).
extern "C" __global__ void wm_bwd(const float* __restrict__ v, const float* __restrict__ g, const float* __restrict__ gm,
                                  float* __restrict__ out, int B, int T, int K, int DK) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * K * DK) return;
    const int c = i % DK, k = (i / DK) % K, b = i / (DK * K);
    const size_t row = (size_t)K * DK, n = (size_t)B * T * row;
    const size_t off = (size_t)b * T * row + (size_t)k * DK + c;
    const float* vb = v + off;
    const float* gb = g + (size_t)b * T * K + k;
    const float* mb = gm + off;
    float* dv = out + off;
    float* dg = out + n + off;
    float m = 0.0f;
    for (int t = 0; t < T; ++t) {
        const float gt = gb[(size_t)t * K];
        m = __fadd_rn(__fmul_rn(m, __fsub_rn(1.0f, gt)), __fmul_rn(vb[(size_t)t * row], gt));
        dv[(size_t)t * row] = m;
    }
    float carry = 0.0f; // dL/dm_t through m_{t+1}
    for (int t = T - 1; t >= 0; --t) {
        const float gt = gb[(size_t)t * K];
        const float G = mb[(size_t)t * row] + carry;
        const float mp = t > 0 ? dv[(size_t)(t - 1) * row] : 0.0f;
        dg[(size_t)t * row] = G * (vb[(size_t)t * row] - mp);
        dv[(size_t)t * row] = G * gt;
        carry = G * (1.0f - gt);
    }
}

// ---- v42 B10 (the structure read): causal attention over strictly earlier keys (j < i), copies of attn_fwd,
// attn_bwd_q and attn_bwd_kv with that mask. Row 0 has no key: its output is 0 (and its log-sum-exp slot 0, unused).
extern "C" __global__ void __launch_bounds__(128) attn_fwd_x(const float* __restrict__ q, const float* __restrict__ k,
                                    const float* __restrict__ v, float* __restrict__ out, int BH, int T, float scale) {
    __shared__ float Qs[AT][AHS];
    __shared__ float Ks[AT][AHS + 1];
    __shared__ float Vs[AT][AHS];
    __shared__ float Ps[AW][ARPW][AT];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.y, i0 = blockIdx.x * AT;
    const size_t base = (size_t)bh * T * AHS;
    stage<AHS>(Qs, q + base, i0, T);
    float m[ARPW], l[ARPW], acc[ARPW];
#pragma unroll
    for (int r = 0; r < ARPW; ++r) { m[r] = -INFINITY; l[r] = 0.0f; acc[r] = 0.0f; }
    for (int j0 = 0; j0 <= i0; j0 += AT) {
        __syncthreads();
        stage<AHS + 1>(Ks, k + base, j0, T);
        stage<AHS>(Vs, v + base, j0, T);
        __syncthreads();
        float kr[AHS];
#pragma unroll
        for (int d = 0; d < AHS; ++d) kr[d] = Ks[lane][d];
        const int j = j0 + lane;
#pragma unroll
        for (int r = 0; r < ARPW; ++r) {
            const int rl = w + AW * r, i = i0 + rl;
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < AHS; ++d) s += Qs[rl][d] * kr[d];
            const bool ok = j < i && j < T;
            s = ok ? s * scale : -INFINITY;
            const float nm = fmaxf(m[r], wmax(s)); // -inf only on row 0 (no earlier key)
            const float p = ok ? expf(s - nm) : 0.0f;
            const float corr = nm == -INFINITY ? 1.0f : expf(m[r] - nm);
            l[r] = l[r] * corr + wsum(p);
            acc[r] *= corr;
            m[r] = nm;
            Ps[w][r][lane] = p;
        }
        __syncwarp();
        for (int jj = 0; jj < AT; ++jj) {
            const float vv = Vs[jj][lane];
#pragma unroll
            for (int r = 0; r < ARPW; ++r) acc[r] += Ps[w][r][jj] * vv;
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int i = i0 + w + AW * r;
        if (i < T) {
            float* o = out + ((size_t)bh * T + i) * (AHS + 1);
            o[lane] = l[r] > 0.0f ? acc[r] / l[r] : 0.0f;
            if (lane == 0) o[AHS] = l[r] > 0.0f ? m[r] + logf(l[r]) : 0.0f;
        }
    }
}

// Backward 1, a block per 32 query rows: D_i = sum(dO_i * O_i) (also written to dvec);
// dQ_i = scale * sum_j P_ij (dO_i . v_j - D_i) k_j.
extern "C" __global__ void __launch_bounds__(128) attn_bwd_q_x(const float* __restrict__ q, const float* __restrict__ k,
                                      const float* __restrict__ v, const float* __restrict__ out, const float* __restrict__ dout,
                                      float* __restrict__ dq, float* __restrict__ dvec, int BH, int T, float scale) {
    __shared__ float Qs[AT][AHS];
    __shared__ float DOs[AT][AHS];
    __shared__ float Ks[AT][AHS + 1];
    __shared__ float Vs[AT][AHS + 1];
    __shared__ float Ls[AT], Ds[AT];
    __shared__ float Ss[AW][ARPW][AT];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.y, i0 = blockIdx.x * AT;
    const size_t base = (size_t)bh * T * AHS;
    stage<AHS>(Qs, q + base, i0, T);
    stage<AHS>(DOs, dout + base, i0, T);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int rl = w + AW * r, i = i0 + rl;
        const float* orow = out + ((size_t)bh * T + i) * (AHS + 1);
        const float Di = wsum(i < T ? DOs[rl][lane] * orow[lane] : 0.0f);
        if (lane == 0) {
            Ds[rl] = Di;
            Ls[rl] = i < T ? orow[AHS] : 0.0f;
            if (i < T) dvec[(size_t)bh * T + i] = Di;
        }
    }
    __syncwarp();
    float acc[ARPW];
#pragma unroll
    for (int r = 0; r < ARPW; ++r) acc[r] = 0.0f;
    for (int j0 = 0; j0 <= i0; j0 += AT) {
        __syncthreads();
        stage<AHS + 1>(Ks, k + base, j0, T);
        stage<AHS + 1>(Vs, v + base, j0, T);
        __syncthreads();
        float kr[AHS], vr[AHS];
#pragma unroll
        for (int d = 0; d < AHS; ++d) { kr[d] = Ks[lane][d]; vr[d] = Vs[lane][d]; }
        const int j = j0 + lane;
#pragma unroll
        for (int r = 0; r < ARPW; ++r) {
            const int rl = w + AW * r, i = i0 + rl;
            float s = 0.0f, dp = 0.0f;
#pragma unroll
            for (int d = 0; d < AHS; ++d) { s += Qs[rl][d] * kr[d]; dp += DOs[rl][d] * vr[d]; }
            const bool ok = j < i && j < T && i < T;
            Ss[w][r][lane] = ok ? expf(s * scale - Ls[rl]) * (dp - Ds[rl]) : 0.0f;
        }
        __syncwarp();
        for (int jj = 0; jj < AT; ++jj) {
            const float kv = Ks[jj][lane];
#pragma unroll
            for (int r = 0; r < ARPW; ++r) acc[r] += Ss[w][r][jj] * kv;
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int i = i0 + w + AW * r;
        if (i < T) dq[base + (size_t)i * AHS + lane] = acc[r] * scale;
    }
}

// Backward 2, a block per 32 key rows: dV_j = sum_{i>=j} P_ij dO_i; dK_j = scale * sum_{i>=j} P_ij (dO_i . v_j - D_i) q_i.
extern "C" __global__ void __launch_bounds__(128) attn_bwd_kv_x(const float* __restrict__ q, const float* __restrict__ k,
                                       const float* __restrict__ v, const float* __restrict__ out, const float* __restrict__ dout,
                                       const float* __restrict__ dvec, float* __restrict__ dk, float* __restrict__ dv,
                                       int BH, int T, float scale) {
    __shared__ float Ks[AT][AHS];
    __shared__ float Vs[AT][AHS];
    __shared__ float Qs[AT][AHS + 1];
    __shared__ float DOs[AT][AHS + 1];
    __shared__ float Ls[AT], Ds[AT];
    __shared__ float Ps[AW][ARPW][AT];
    __shared__ float DSs[AW][ARPW][AT];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int bh = blockIdx.y, j0 = blockIdx.x * AT;
    const size_t base = (size_t)bh * T * AHS;
    stage<AHS>(Ks, k + base, j0, T);
    stage<AHS>(Vs, v + base, j0, T);
    float gk[ARPW], gv[ARPW];
#pragma unroll
    for (int r = 0; r < ARPW; ++r) { gk[r] = 0.0f; gv[r] = 0.0f; }
    for (int i0 = j0; i0 < T; i0 += AT) {
        __syncthreads();
        stage<AHS + 1>(Qs, q + base, i0, T);
        stage<AHS + 1>(DOs, dout + base, i0, T);
        if (threadIdx.x < AT) {
            const int i = i0 + threadIdx.x;
            Ls[threadIdx.x] = i < T ? out[((size_t)bh * T + i) * (AHS + 1) + AHS] : 0.0f;
            Ds[threadIdx.x] = i < T ? dvec[(size_t)bh * T + i] : 0.0f;
        }
        __syncthreads();
        float qr[AHS], dor[AHS];
#pragma unroll
        for (int d = 0; d < AHS; ++d) { qr[d] = Qs[lane][d]; dor[d] = DOs[lane][d]; }
        const int i = i0 + lane;
        const float Li = Ls[lane], Di = Ds[lane];
#pragma unroll
        for (int r = 0; r < ARPW; ++r) {
            const int rl = w + AW * r, j = j0 + rl;
            float s = 0.0f, dp = 0.0f;
#pragma unroll
            for (int d = 0; d < AHS; ++d) { s += qr[d] * Ks[rl][d]; dp += dor[d] * Vs[rl][d]; }
            const bool ok = i > j && i < T && j < T;
            const float p = ok ? expf(s * scale - Li) : 0.0f;
            Ps[w][r][lane] = p;
            DSs[w][r][lane] = p * (dp - Di);
        }
        __syncwarp();
        for (int ii = 0; ii < AT; ++ii) {
            const float dov = DOs[ii][lane], qv = Qs[ii][lane];
#pragma unroll
            for (int r = 0; r < ARPW; ++r) { gv[r] += Ps[w][r][ii] * dov; gk[r] += DSs[w][r][ii] * qv; }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
        const int j = j0 + w + AW * r;
        if (j < T) {
            dk[base + (size_t)j * AHS + lane] = gk[r] * scale;
            dv[base + (size_t)j * AHS + lane] = gv[r];
        }
    }
}


// =====================================================================================================================
// v47: model D with Stage 3's winners on a hand-written training step (src/model/fwin.rs; wrappers src/fwin_ops.rs).
// bf16 helpers: loads from either operand type as f32 (a bf16 value is exact in f32).
__device__ __forceinline__ float ldf(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float ldf(const unsigned short* p, size_t i) { return __uint_as_float(((unsigned int)p[i]) << 16); }

// --mlp-act relu2: kwta_relu, then squared. a = the kwta_relu value (for the backward), a2 = a * a (the w2 operand);
// both written as O. The select is kwta_relu's (kwta_kth), so a is bit for bit kwta_relu's.
template <typename O>
__device__ __forceinline__ void kwta_relu2_impl(const float* __restrict__ u, O* __restrict__ a, O* __restrict__ a2, int N, int H, int K,
                                                unsigned int* keys, unsigned int* hist, unsigned int& s_prefix, unsigned int& s_k) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const float* x = u + (size_t)row * H;
    const float kth = kwta_kth(x, H, K, keys, hist, s_prefix, s_k);
    O* o = a + (size_t)row * H;
    O* o2 = a2 + (size_t)row * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float xi = x[i];
        const float v = (__fsub_rn(xi, kth) >= 0.0f && xi > 0.0f) ? xi : 0.0f;
        put(o, i, v);
        put(o2, i, __fmul_rn(v, v));
    }
}
extern "C" __global__ void kwta_relu2_rows(const float* __restrict__ u, float* __restrict__ a, float* __restrict__ a2, int N, int H, int K) {
    extern __shared__ unsigned int keys[];
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix, s_k;
    kwta_relu2_impl(u, a, a2, N, H, K, keys, hist, s_prefix, s_k);
}
extern "C" __global__ void kwta_relu2_rows_b(const float* __restrict__ u, unsigned short* __restrict__ a, unsigned short* __restrict__ a2,
                                             int N, int H, int K) {
    extern __shared__ unsigned int keys[];
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix, s_k;
    kwta_relu2_impl(u, a, a2, N, H, K, keys, hist, s_prefix, s_k);
}
// Backward of relu2 after kWTA: gu = (a ga) 2 where a > 0, else 0 (candle's sqr backward, then the relu mask).
template <typename O>
__device__ __forceinline__ void relu2_bwd_impl(const O* __restrict__ a, const float* __restrict__ ga, O* __restrict__ gu, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float ai = ldf(a, i);
    put(gu, i, ai > 0.0f ? __fmul_rn(__fmul_rn(ai, ga[i]), 2.0f) : 0.0f);
}
extern "C" __global__ void relu2_bwd(const float* __restrict__ a, const float* __restrict__ ga, float* __restrict__ gu, int n) {
    relu2_bwd_impl(a, ga, gu, n);
}
extern "C" __global__ void relu2_bwd_b(const unsigned short* __restrict__ a, const float* __restrict__ ga, unsigned short* __restrict__ gu, int n) {
    relu2_bwd_impl(a, ga, gu, n);
}

// The fast-weight memory around the memg scan (head size HS = 32 or 64). P [N, ld] f32 (row n = b T + t) =
// h . [Wq | Wk | Wv | Wg | Wa | Wb], cst [3H] = (ba, bb, off), as hebb_fwd's. One warp per (row, head); lane j holds
// channels j, j + 32, ... The scan's layout: qke [B H, T, 3 HS] = (q^, k^, e = k^), vg [B H, T, HS + 2] = (v, log alpha,
// beta), y [B H, T, HS]. The maths of arch.rs mem_arch for D with per-head gates:
//   q^ = q / sqrt(|q|^2 + 1e-6), k^ likewise;  log alpha = logsigmoid((za + ba) + off);  beta = 2 sigmoid(zb + bb)
//   --value-residual (mode 1, blocks after the first): v = v (1 + p0) + v0 p1, v0 the first block's v (its P: P0)
//   O = y / sqrt(mean(y^2) + eps) * silu(g)
template <int HS>
__device__ __forceinline__ void wprep_impl(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ vres,
                                           const float* __restrict__ P0, float* __restrict__ qke, float* __restrict__ vg, int N, int T,
                                           int H, int ld, int mode) {
    constexpr int R = HS / 32;
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= N * H) return;
    const int n = w / H, h = w - n * H, b = n / T, t = n - b * T, d = H * HS;
    const float* r = P + (size_t)n * ld;
    float q[R], k[R], v[R], sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int c = h * HS + lane + 32 * u;
        q[u] = r[c];
        k[u] = r[d + c];
        v[u] = r[2 * d + c];
        sq += q[u] * q[u];
        sk += k[u] * k[u];
    }
    const float qs = rsqrtf(wsum(sq) + 1e-6f), ks = rsqrtf(wsum(sk) + 1e-6f);
    if (mode == 1) {
        const float p0 = vres[0] + 1.0f, p1 = vres[1];
        const float* r0 = P0 + (size_t)n * ld + 2 * d + h * HS;
#pragma unroll
        for (int u = 0; u < R; ++u) v[u] = __fadd_rn(__fmul_rn(v[u], p0), __fmul_rn(r0[lane + 32 * u], p1));
    }
    const size_t row = (size_t)(b * H + h) * T + t;
    float* oq = qke + row * 3 * HS;
    float* ov = vg + row * (HS + 2);
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int c = lane + 32 * u;
        const float kh = k[u] * ks;
        oq[c] = q[u] * qs;
        oq[HS + c] = kh;
        oq[2 * HS + c] = kh;
        ov[c] = v[u];
    }
    if (lane == 0) {
        ov[HS] = hc_logsig((r[4 * d + h] + cst[h]) + cst[2 * H + h]);
        ov[HS + 1] = 2.0f * hb_sig(r[4 * d + H + h] + cst[H + h]);
    }
}
template <int HS, typename OT>
__device__ __forceinline__ void wpost_impl(const float* __restrict__ P, const float* __restrict__ Y, OT* __restrict__ O, int N, int T, int H,
                                           int ld, float eps) {
    constexpr int R = HS / 32;
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= N * H) return;
    const int n = w / H, h = w - n * H, b = n / T, t = n - b * T, d = H * HS;
    const float* yr = Y + ((size_t)(b * H + h) * T + t) * HS;
    float y[R], ss = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        y[u] = yr[lane + 32 * u];
        ss += y[u] * y[u];
    }
    const float rr = rsqrtf(wsum(ss) * (1.0f / HS) + eps);
    const float* g = P + (size_t)n * ld + 3 * d + h * HS;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const float gg = g[lane + 32 * u];
        put(O, (size_t)n * d + h * HS + lane + 32 * u, y[u] * rr * (gg * hb_sig(gg)));
    }
}
// Backward of wpost: dY [B H, T, HS] (the scan's dy) and dP's gate columns (written as OT into dP [N, ld]).
template <int HS, typename OT>
__device__ __forceinline__ void wpost_bwd_impl(const float* __restrict__ P, const float* __restrict__ Y, const float* __restrict__ dO,
                                               float* __restrict__ dY, OT* __restrict__ dP, int N, int T, int H, int ld, float eps) {
    constexpr int R = HS / 32;
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= N * H) return;
    const int n = w / H, h = w - n * H, b = n / T, t = n - b * T, d = H * HS;
    const size_t row = (size_t)(b * H + h) * T + t;
    const float* yr = Y + row * HS;
    float y[R], ss = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        y[u] = yr[lane + 32 * u];
        ss += y[u] * y[u];
    }
    const float rr = rsqrtf(wsum(ss) * (1.0f / HS) + eps);
    const float* g = P + (size_t)n * ld + 3 * d + h * HS;
    const float* go = dO + (size_t)n * d + h * HS;
    float dyn[R], yn[R], dot = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int c = lane + 32 * u;
        const float gg = g[c], s = hb_sig(gg), od = go[c];
        yn[u] = y[u] * rr;
        dyn[u] = od * (gg * s);
        dot += dyn[u] * yn[u];
        put(dP, (size_t)n * ld + 3 * d + h * HS + c, od * yn[u] * (s * (1.0f + gg * (1.0f - s))));
    }
    dot = wsum(dot) * (1.0f / HS);
#pragma unroll
    for (int u = 0; u < R; ++u) dY[row * HS + lane + 32 * u] = rr * (dyn[u] - yn[u] * dot);
}
// Backward of wprep from the scan's dqke, dvg: dP's q, k, v, za, zb columns (OT), dz [N, 2H] f32 (za, zb columns, for
// the bias gradients). mode 1 (value residual): dP_v = dv (1 + p0), accout = accin + dv p1 (the first block's share;
// accin absent when has_acc = 0), part [N H, 2] = per (row, head) sums of dv v and dv v0 (for dp0, dp1). mode 2 (the
// first block, when later blocks read its v): dP_v = dv + accin. mode 0: dP_v = dv.
template <int HS, typename OT>
__device__ __forceinline__ void wprep_bwd_impl(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ vres,
                                               const float* __restrict__ P0, const float* __restrict__ accin, const float* __restrict__ dqke,
                                               const float* __restrict__ dvg, OT* __restrict__ dP, float* __restrict__ dz,
                                               float* __restrict__ accout, float* __restrict__ part, int N, int T, int H, int ld, int mode,
                                               int has_acc) {
    constexpr int R = HS / 32;
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= N * H) return;
    const int n = w / H, h = w - n * H, b = n / T, t = n - b * T, d = H * HS;
    const float* r = P + (size_t)n * ld;
    const size_t row = (size_t)(b * H + h) * T + t;
    const float* gq = dqke + row * 3 * HS;
    const float* gv = dvg + row * (HS + 2);
    float q[R], k[R], sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int c = h * HS + lane + 32 * u;
        q[u] = r[c];
        k[u] = r[d + c];
        sq += q[u] * q[u];
        sk += k[u] * k[u];
    }
    const float qs = rsqrtf(wsum(sq) + 1e-6f), ks = rsqrtf(wsum(sk) + 1e-6f);
    float dqh[R], dkh[R], sdq = 0.0f, sdk = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int c = lane + 32 * u;
        dqh[u] = gq[c];
        dkh[u] = gq[HS + c] + gq[2 * HS + c];
        q[u] *= qs;
        k[u] *= ks;
        sdq += dqh[u] * q[u];
        sdk += dkh[u] * k[u];
    }
    sdq = wsum(sdq);
    sdk = wsum(sdk);
    OT* o = dP + (size_t)n * ld;
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int c = lane + 32 * u, cg = h * HS + c;
        put(o, cg, qs * (dqh[u] - q[u] * sdq));
        put(o, d + cg, ks * (dkh[u] - k[u] * sdk));
        const float dv = gv[c];
        if (mode == 1) {
            const float p0 = vres[0] + 1.0f, p1 = vres[1];
            const float v0 = P0[(size_t)n * ld + 2 * d + cg];
            put(o, 2 * d + cg, dv * p0);
            const float prev = has_acc ? accin[(size_t)n * d + cg] : 0.0f;
            accout[(size_t)n * d + cg] = prev + dv * p1;
            s0 += dv * r[2 * d + cg];
            s1 += dv * v0;
        } else if (mode == 2) {
            put(o, 2 * d + cg, has_acc ? dv + accin[(size_t)n * d + cg] : dv);
        } else {
            put(o, 2 * d + cg, dv);
        }
    }
    if (mode == 1) {
        s0 = wsum(s0);
        s1 = wsum(s1);
        if (lane == 0) {
            part[(size_t)w * 2] = s0;
            part[(size_t)w * 2 + 1] = s1;
        }
    }
    if (lane == 0) {
        const float sa = hb_sig((r[4 * d + h] + cst[h]) + cst[2 * H + h]);
        const float sb = hb_sig(r[4 * d + H + h] + cst[H + h]);
        const float dza = gv[HS] * (1.0f - sa), dzb = gv[HS + 1] * 2.0f * sb * (1.0f - sb);
        put(o, 4 * d + h, dza);
        put(o, 4 * d + H + h, dzb);
        dz[(size_t)n * 2 * H + h] = dza;
        dz[(size_t)n * 2 * H + H + h] = dzb;
    }
}
#define WMEM_KERNELS(HS)                                                                                                     \
    extern "C" __global__ void wprep_##HS(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ vres, \
                                          const float* __restrict__ P0, float* __restrict__ qke, float* __restrict__ vg, int N, int T, \
                                          int H, int ld, int mode) {                                                           \
        wprep_impl<HS>(P, cst, vres, P0, qke, vg, N, T, H, ld, mode);                                                         \
    }                                                                                                                        \
    extern "C" __global__ void wpost_##HS(const float* __restrict__ P, const float* __restrict__ Y, float* __restrict__ O, int N, int T, \
                                          int H, int ld, float eps) {                                                          \
        wpost_impl<HS, float>(P, Y, O, N, T, H, ld, eps);                                                                     \
    }                                                                                                                        \
    extern "C" __global__ void wpost_b_##HS(const float* __restrict__ P, const float* __restrict__ Y, unsigned short* __restrict__ O, int N, \
                                            int T, int H, int ld, float eps) {                                                 \
        wpost_impl<HS, unsigned short>(P, Y, O, N, T, H, ld, eps);                                                            \
    }                                                                                                                        \
    extern "C" __global__ void wpost_bwd_##HS(const float* __restrict__ P, const float* __restrict__ Y, const float* __restrict__ dO, \
                                              float* __restrict__ dY, float* __restrict__ dP, int N, int T, int H, int ld, float eps) { \
        wpost_bwd_impl<HS, float>(P, Y, dO, dY, dP, N, T, H, ld, eps);                                                        \
    }                                                                                                                        \
    extern "C" __global__ void wpost_bwd_b_##HS(const float* __restrict__ P, const float* __restrict__ Y, const float* __restrict__ dO, \
                                                float* __restrict__ dY, unsigned short* __restrict__ dP, int N, int T, int H, int ld, \
                                                float eps) {                                                                   \
        wpost_bwd_impl<HS, unsigned short>(P, Y, dO, dY, dP, N, T, H, ld, eps);                                               \
    }                                                                                                                        \
    extern "C" __global__ void wprep_bwd_##HS(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ vres, \
                                              const float* __restrict__ P0, const float* __restrict__ accin, const float* __restrict__ dqke, \
                                              const float* __restrict__ dvg, float* __restrict__ dP, float* __restrict__ dz,       \
                                              float* __restrict__ accout, float* __restrict__ part, int N, int T, int H, int ld,    \
                                              int mode, int has_acc) {                                                         \
        wprep_bwd_impl<HS, float>(P, cst, vres, P0, accin, dqke, dvg, dP, dz, accout, part, N, T, H, ld, mode, has_acc);     \
    }                                                                                                                        \
    extern "C" __global__ void wprep_bwd_b_##HS(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ vres, \
                                                const float* __restrict__ P0, const float* __restrict__ accin, const float* __restrict__ dqke, \
                                                const float* __restrict__ dvg, unsigned short* __restrict__ dP, float* __restrict__ dz, \
                                                float* __restrict__ accout, float* __restrict__ part, int N, int T, int H, int ld,  \
                                                int mode, int has_acc) {                                                       \
        wprep_bwd_impl<HS, unsigned short>(P, cst, vres, P0, accin, dqke, dvg, dP, dz, accout, part, N, T, H, ld, mode, has_acc); \
    }
WMEM_KERNELS(32)
WMEM_KERNELS(64)

// --residual dense (DenseFormer). S [R, M]: the streams as rows (row 0 the embedding, row j + 1 block j's raw output);
// wf: every block's mixing weights in one vector, block i's i + 2 weights from i (i + 3) / 2.
// dense_mix (block i): raw = xa + xb (the block's last residual add) goes to row i + 1 of S, and
//   out = raw + sum_{j = 0..i+1} w_ij S_j, added in arch.rs's order (j = 0 first), each product and sum rounded alone.
extern "C" __global__ void dense_mix(const float* __restrict__ xa, const float* __restrict__ xb, const float* __restrict__ wf,
                                     float* __restrict__ S, float* __restrict__ out, int i, long long M) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= M) return;
    const float* w = wf + i * (i + 3) / 2;
    const float raw = __fadd_rn(xa[e], xb[e]);
    S[(size_t)(i + 1) * M + e] = raw;
    float x = raw;
    for (int j = 0; j <= i; ++j) x = __fadd_rn(x, __fmul_rn(S[(size_t)j * M + e], w[j]));
    out[e] = __fadd_rn(x, __fmul_rn(raw, w[i + 1]));
}
// dense_back: out = base + sum_{k = i0..L-1} w_k[col] G_k (G [L, M]: row k = dL/d(the mix after block k)), as f32 and,
// in dense_back_b, also as bf16; part [gridDim, nd] = this thread block's share of sum_e base_e S_{j, e} (j < nd).
// Block i: base = G_i, col = i + 1, i0 = i, nd = i + 2 (dL/d raw_{i+1} and dw_i). Embedding: base = dL/dx0, col 0, i0 0, nd 0.
#define DENSE_MAXD 32
template <typename OB>
__device__ __forceinline__ void dense_back_impl(const float* __restrict__ S, const float* __restrict__ G, const float* __restrict__ wf,
                                                const float* __restrict__ base, float* __restrict__ out, OB* __restrict__ outb,
                                                float* __restrict__ part, int col, int i0, int L, int nd, long long M) {
    __shared__ float red[8][DENSE_MAXD];
    float acc[DENSE_MAXD];
#pragma unroll
    for (int j = 0; j < DENSE_MAXD; ++j) acc[j] = 0.0f;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < M; e += stride) {
        const float g = base[e];
        float x = g;
        for (int k = i0; k < L; ++k) x = __fadd_rn(x, __fmul_rn(wf[k * (k + 3) / 2 + col], G[(size_t)k * M + e]));
        out[e] = x;
        if (outb != nullptr) put(outb, (size_t)e, x);
#pragma unroll
        for (int j = 0; j < DENSE_MAXD; ++j)
            if (j < nd) acc[j] += g * S[(size_t)j * M + e];
    }
    if (nd == 0) return;
    const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < DENSE_MAXD; ++j) {
        if (j < nd) {
            const float s = wsum(acc[j]);
            if (lane == 0) red[wi][j] = s;
        }
    }
    __syncthreads();
    if ((int)threadIdx.x < nd) {
        float s = 0.0f;
        for (int k = 0; k < (int)(blockDim.x >> 5); ++k) s += red[k][threadIdx.x];
        part[(size_t)blockIdx.x * nd + threadIdx.x] = s;
    }
}
extern "C" __global__ void __launch_bounds__(256) dense_back(const float* __restrict__ S, const float* __restrict__ G, const float* __restrict__ wf,
                                                             const float* __restrict__ base, float* __restrict__ out, float* __restrict__ part,
                                                             int col, int i0, int L, int nd, long long M) {
    dense_back_impl<float>(S, G, wf, base, out, (float*)nullptr, part, col, i0, L, nd, M);
}
extern "C" __global__ void __launch_bounds__(256) dense_back_b(const float* __restrict__ S, const float* __restrict__ G, const float* __restrict__ wf,
                                                               const float* __restrict__ base, float* __restrict__ out,
                                                               unsigned short* __restrict__ outb, float* __restrict__ part, int col, int i0,
                                                               int L, int nd, long long M) {
    dense_back_impl<unsigned short>(S, G, wf, base, out, outb, part, col, i0, L, nd, M);
}

// v47 SLOW (--gates channel): the prep around the memg scan with P2's per-channel gates and erase key. P [N, ld] =
// h . [Wq | Wk | Wv | Wg | Wa | Wb | We | Wa1 | Wb1] (ld = 5d + 2H + 2 GATE_RANK), cst [2H + d] = (ba, bb, the forget
// offset per channel), Z [N, 2d] = (h Wa1 Wa2 | h Wb1 Wb2). arch.rs mem_arch with channel gates:
//   log alpha_c = logsigmoid(((za_h + ba_h) + off_c) + Za_c);  beta_c = 2 sigmoid((zb_h + bb_h) + Zb_c)
//   e = unit(k + h We) (k before its norm);  qke = (q^, k^, e), vg [B H, T, 3 HS] = (v, log alpha, beta)
template <int HS>
__device__ __forceinline__ void wprepc_impl(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ vres,
                                            const float* __restrict__ P0, const float* __restrict__ Z, float* __restrict__ qke,
                                            float* __restrict__ vg, int N, int T, int H, int ld, int mode) {
    constexpr int R = HS / 32;
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= N * H) return;
    const int n = w / H, h = w - n * H, b = n / T, t = n - b * T, d = H * HS;
    const float* r = P + (size_t)n * ld;
    const float* z = Z + (size_t)n * 2 * d;
    float q[R], k[R], e[R], v[R], sq = 0.0f, sk = 0.0f, se = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int c = h * HS + lane + 32 * u;
        q[u] = r[c];
        k[u] = r[d + c];
        v[u] = r[2 * d + c];
        e[u] = __fadd_rn(k[u], r[4 * d + 2 * H + c]);
        sq += q[u] * q[u];
        sk += k[u] * k[u];
        se += e[u] * e[u];
    }
    const float qs = rsqrtf(wsum(sq) + 1e-6f), ks = rsqrtf(wsum(sk) + 1e-6f), es = rsqrtf(wsum(se) + 1e-6f);
    if (mode == 1) {
        const float p0 = vres[0] + 1.0f, p1 = vres[1];
        const float* r0 = P0 + (size_t)n * ld + 2 * d + h * HS;
#pragma unroll
        for (int u = 0; u < R; ++u) v[u] = __fadd_rn(__fmul_rn(v[u], p0), __fmul_rn(r0[lane + 32 * u], p1));
    }
    const size_t row = (size_t)(b * H + h) * T + t;
    float* oq = qke + row * 3 * HS;
    float* ov = vg + row * 3 * HS;
    const float zah = r[4 * d + h] + cst[h], zbh = r[4 * d + H + h] + cst[H + h];
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int c = lane + 32 * u, cg = h * HS + c;
        oq[c] = q[u] * qs;
        oq[HS + c] = k[u] * ks;
        oq[2 * HS + c] = e[u] * es;
        ov[c] = v[u];
        ov[HS + c] = hc_logsig((zah + cst[2 * H + cg]) + z[cg]);
        ov[2 * HS + c] = 2.0f * hb_sig(zbh + z[d + cg]);
    }
}
// The backward of wprepc: the q, k, v, za, zb and e columns of dP (OT; the Wa1, Wb1 columns are filled after, from dZ),
// dz [N, 2H] f32 (the head gates' sums, for the bias gradients), dZ [N, 2d] (OT); the value residual as in wprep_bwd.
template <int HS, typename OT>
__device__ __forceinline__ void wprepc_bwd_impl(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ vres,
                                                const float* __restrict__ P0, const float* __restrict__ Z, const float* __restrict__ accin,
                                                const float* __restrict__ dqke, const float* __restrict__ dvg, OT* __restrict__ dP,
                                                float* __restrict__ dz, OT* __restrict__ dZ, float* __restrict__ accout, float* __restrict__ part,
                                                int N, int T, int H, int ld, int mode, int has_acc) {
    constexpr int R = HS / 32;
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= N * H) return;
    const int n = w / H, h = w - n * H, b = n / T, t = n - b * T, d = H * HS;
    const float* r = P + (size_t)n * ld;
    const float* z = Z + (size_t)n * 2 * d;
    const size_t row = (size_t)(b * H + h) * T + t;
    const float* gq = dqke + row * 3 * HS;
    const float* gv = dvg + row * 3 * HS;
    float q[R], k[R], e[R], sq = 0.0f, sk = 0.0f, se = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int c = h * HS + lane + 32 * u;
        q[u] = r[c];
        k[u] = r[d + c];
        e[u] = __fadd_rn(k[u], r[4 * d + 2 * H + c]);
        sq += q[u] * q[u];
        sk += k[u] * k[u];
        se += e[u] * e[u];
    }
    const float qs = rsqrtf(wsum(sq) + 1e-6f), ks = rsqrtf(wsum(sk) + 1e-6f), es = rsqrtf(wsum(se) + 1e-6f);
    float dqh[R], dkh[R], deh[R], sdq = 0.0f, sdk = 0.0f, sde = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int c = lane + 32 * u;
        dqh[u] = gq[c];
        dkh[u] = gq[HS + c];
        deh[u] = gq[2 * HS + c];
        q[u] *= qs;
        k[u] *= ks;
        e[u] *= es;
        sdq += dqh[u] * q[u];
        sdk += dkh[u] * k[u];
        sde += deh[u] * e[u];
    }
    sdq = wsum(sdq);
    sdk = wsum(sdk);
    sde = wsum(sde);
    OT* o = dP + (size_t)n * ld;
    OT* oz = dZ + (size_t)n * 2 * d;
    const float zah = r[4 * d + h] + cst[h], zbh = r[4 * d + H + h] + cst[H + h];
    float s0 = 0.0f, s1 = 0.0f, ga = 0.0f, gb = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int c = lane + 32 * u, cg = h * HS + c;
        const float du = es * (deh[u] - e[u] * sde);
        put(o, cg, qs * (dqh[u] - q[u] * sdq));
        put(o, d + cg, ks * (dkh[u] - k[u] * sdk) + du);
        put(o, 4 * d + 2 * H + cg, du);
        const float dv = gv[c];
        if (mode == 1) {
            const float p0 = vres[0] + 1.0f, p1 = vres[1];
            const float v0 = P0[(size_t)n * ld + 2 * d + cg];
            put(o, 2 * d + cg, dv * p0);
            const float prev = has_acc ? accin[(size_t)n * d + cg] : 0.0f;
            accout[(size_t)n * d + cg] = prev + dv * p1;
            s0 += dv * r[2 * d + cg];
            s1 += dv * v0;
        } else if (mode == 2) {
            put(o, 2 * d + cg, has_acc ? dv + accin[(size_t)n * d + cg] : dv);
        } else {
            put(o, 2 * d + cg, dv);
        }
        const float sa = hb_sig((zah + cst[2 * H + cg]) + z[cg]);
        const float sb = hb_sig(zbh + z[d + cg]);
        const float dza = gv[HS + c] * (1.0f - sa), dzb = gv[2 * HS + c] * 2.0f * sb * (1.0f - sb);
        put(oz, cg, dza);
        put(oz, d + cg, dzb);
        ga += dza;
        gb += dzb;
    }
    if (mode == 1) {
        s0 = wsum(s0);
        s1 = wsum(s1);
        if (lane == 0) {
            part[(size_t)w * 2] = s0;
            part[(size_t)w * 2 + 1] = s1;
        }
    }
    ga = wsum(ga);
    gb = wsum(gb);
    if (lane == 0) {
        put(o, 4 * d + h, ga);
        put(o, 4 * d + H + h, gb);
        dz[(size_t)n * 2 * H + h] = ga;
        dz[(size_t)n * 2 * H + H + h] = gb;
    }
}
#define WMEMC_KERNELS(HS)                                                                                                     \
    extern "C" __global__ void wprepc_##HS(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ vres, \
                                           const float* __restrict__ P0, const float* __restrict__ Z, float* __restrict__ qke,     \
                                           float* __restrict__ vg, int N, int T, int H, int ld, int mode) {                     \
        wprepc_impl<HS>(P, cst, vres, P0, Z, qke, vg, N, T, H, ld, mode);                                                     \
    }                                                                                                                         \
    extern "C" __global__ void wprepc_bwd_##HS(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ vres, \
                                               const float* __restrict__ P0, const float* __restrict__ Z, const float* __restrict__ accin, \
                                               const float* __restrict__ dqke, const float* __restrict__ dvg, float* __restrict__ dP, \
                                               float* __restrict__ dz, float* __restrict__ dZ, float* __restrict__ accout,         \
                                               float* __restrict__ part, int N, int T, int H, int ld, int mode, int has_acc) {    \
        wprepc_bwd_impl<HS, float>(P, cst, vres, P0, Z, accin, dqke, dvg, dP, dz, dZ, accout, part, N, T, H, ld, mode, has_acc); \
    }                                                                                                                         \
    extern "C" __global__ void wprepc_bwd_b_##HS(const float* __restrict__ P, const float* __restrict__ cst, const float* __restrict__ vres, \
                                                 const float* __restrict__ P0, const float* __restrict__ Z, const float* __restrict__ accin, \
                                                 const float* __restrict__ dqke, const float* __restrict__ dvg, unsigned short* __restrict__ dP, \
                                                 float* __restrict__ dz, unsigned short* __restrict__ dZ, float* __restrict__ accout, \
                                                 float* __restrict__ part, int N, int T, int H, int ld, int mode, int has_acc) {  \
        wprepc_bwd_impl<HS, unsigned short>(P, cst, vres, P0, Z, accin, dqke, dvg, dP, dz, dZ, accout, part, N, T, H, ld, mode, has_acc); \
    }
WMEMC_KERNELS(32)
WMEMC_KERNELS(64)

// =====================================================================================================================
// v51 fused N (src/fwin_nm.rs): --neuromod 3 (NE, ACh, DA; src/v50.rs) on the hand-written WIN training step. Each kernel
// has a candle-op twin in fwin_nm.rs (the CPU tests check those against autograd; the GPU tests these against them).
//
// nm_ach_fwd: the memory scan's write strengths (vg [B H, T, W], W = HS + 2G: the last G columns of each row
// (b H + h) T + t) times the token's ACh signal s[b T + t], kept in [0, 2] (candle's clamp: minimum(maximum(y, 0), 2)),
// in place; braw [B H, T, G] gets the unscaled values. One thread per strength (E = B H T G).
extern "C" __global__ void nm_ach_fwd(float* __restrict__ vg, const float* __restrict__ s, float* __restrict__ braw, long long E, int T, int H,
                                      int W, int G) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const long long row = e / G;
    const int c = (int)(e - row * G);
    const int bh = (int)(row / T), t = (int)(row - (long long)bh * T), b = bh / H;
    float* p = vg + row * W + (W - G) + c;
    const float x = *p;
    braw[e] = x;
    const float y = __fmul_rn(x, s[(size_t)b * T + t]);
    *p = fminf(fmaxf(y, 0.0f), 2.0f);
}
// nm_ach_bwd: dvg's strength columns hold dL/d(scaled); they become dL/d(unscaled) = (dL/d(scaled) m) s, m candle's clamp
// gradient (1 inside (0, 2), 1/2 on either bound, 0 outside); part[n H + h] = the sum over the head's G columns of
// (dL/d(scaled) m) x (x unscaled): the ACh signal's gradient per (token, head). One warp per (token, head).
extern "C" __global__ void nm_ach_bwd(float* __restrict__ dvg, const float* __restrict__ braw, const float* __restrict__ s, float* __restrict__ part,
                                      int N, int T, int H, int W, int G) {
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= N * H) return;
    const int n = w / H, h = w - n * H, b = n / T, t = n - b * T;
    const size_t row = (size_t)(b * H + h) * T + t;
    const float sn = s[n];
    float acc = 0.0f;
    for (int c = lane; c < G; c += 32) {
        float* p = dvg + row * W + (W - G) + c;
        const float x = braw[row * G + c];
        const float y = __fmul_rn(x, sn);
        const float m = (y > 0.0f && y < 2.0f) ? 1.0f : ((y == 0.0f || y == 2.0f) ? 0.5f : 0.0f);
        const float gd = __fmul_rn(*p, m);
        *p = __fmul_rn(gd, sn);
        acc = fmaf(gd, x, acc);
    }
    acc = wsum(acc);
    if (lane == 0) part[w] = acc;
}
// nm_rows_bwd(_b): the backward of y = x s with one s per row (NE on the LIF's input current, DA on the MLP's output),
// dy, x [N, D] f32: dx = dy s (OT: the matmul operand, f32 or bf16 by f2bf), ds[n] = sum_j dy x. One warp per row.
template <typename OT>
__device__ __forceinline__ void nm_rows_bwd_impl(const float* __restrict__ dy, const float* __restrict__ x, const float* __restrict__ s,
                                                 OT* __restrict__ dx, float* __restrict__ ds, int N, int D) {
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= N) return;
    const size_t o = (size_t)w * D;
    const float sn = s[w];
    float acc = 0.0f;
    for (int j = lane; j < D; j += 32) {
        const float g = dy[o + j];
        put(dx, o + j, __fmul_rn(g, sn));
        acc = fmaf(g, x[o + j], acc);
    }
    acc = wsum(acc);
    if (lane == 0) ds[w] = acc;
}
extern "C" __global__ void nm_rows_bwd(const float* __restrict__ dy, const float* __restrict__ x, const float* __restrict__ s, float* __restrict__ dx,
                                       float* __restrict__ ds, int N, int D) {
    nm_rows_bwd_impl<float>(dy, x, s, dx, ds, N, D);
}
extern "C" __global__ void nm_rows_bwd_b(const float* __restrict__ dy, const float* __restrict__ x, const float* __restrict__ s,
                                         unsigned short* __restrict__ dx, float* __restrict__ ds, int N, int D) {
    nm_rows_bwd_impl<unsigned short>(dy, x, s, dx, ds, N, D);
}

// =====================================================================================================================
// v51 fast2 (src/model/fwin_f2.rs): R --letters and CB --cerebellum-head on the hand-written WIN training step (W
// --columns and TH --thalamus need no kernels of their own). Each kernel has a candle-op twin in fwin_f2.rs (the CPU tests
// check the twins against autograd; the GPU tests these kernels against the twins).
//
// f2_let_fwd: R's letter vectors of the whole vocabulary, out [V, 64] = inv[v] * sum_j mask[v, j] tanh((E[b_vj] + Q[e_vj])
// + P[j]) (v51::letter_core's ops), the sum over the L = 16 places in candle's order (its reduce kernel for a dim of 16:
// one value per thread from 0, then the tree s = 8, 4, 2, 1). One thread per (v, c), c of the 64 columns fastest.
#define F2_LD 64
#define F2_LM 16
extern "C" __global__ void f2_let_fwd(const float* __restrict__ E, const float* __restrict__ P, const float* __restrict__ Q,
                                      const unsigned int* __restrict__ ids, const unsigned int* __restrict__ eidx,
                                      const float* __restrict__ mask, const float* __restrict__ inv, float* __restrict__ out, int V) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V * F2_LD) return;
    const int v = i / F2_LD, c = i - v * F2_LD;
    float t[F2_LM];
#pragma unroll
    for (int j = 0; j < F2_LM; ++j) {
        const size_t p = (size_t)v * F2_LM + j;
        const float z = __fadd_rn(__fadd_rn(E[(size_t)ids[p] * F2_LD + c], Q[(size_t)eidx[p] * F2_LD + c]), P[j * F2_LD + c]);
        t[j] = __fadd_rn(0.0f, __fmul_rn(tanhf(z), mask[p]));
    }
#pragma unroll
    for (int s = F2_LM / 2; s > 0; s >>= 1) {
#pragma unroll
        for (int j = 0; j < s; ++j) t[j] = __fadd_rn(t[j], t[j + s]);
    }
    out[i] = __fmul_rn(t[0], inv[v]);
}
// f2_let_bwd_part: R's backward, part 1. dz[p, c] = (dL[v, c] inv[v]) (1 - s^2), s = tanh((E[b_p] + Q[e_p]) + P[j]) recomputed,
// for the entries p = v L + j that hold a byte (mask 1), listed by table row (fwin_f2::LetCsr: rows 0..255 the bytes,
// 256..271 the places, 272..287 the places from the end), in chunks of at most F2_CH entries of one row each:
// part[chunk, c] = the chunk's sum, one entry after another from 0. One block of 64 threads per chunk.
extern "C" __global__ void f2_let_bwd_part(const float* __restrict__ E, const float* __restrict__ P, const float* __restrict__ Q,
                                           const unsigned int* __restrict__ ids, const unsigned int* __restrict__ eidx,
                                           const float* __restrict__ inv, const float* __restrict__ dl, const unsigned int* __restrict__ pos,
                                           const unsigned int* __restrict__ cbeg, float* __restrict__ part, int NCH) {
    const int ch = blockIdx.x, c = threadIdx.x;
    if (ch >= NCH || c >= F2_LD) return;
    float acc = 0.0f;
    for (unsigned int e = cbeg[ch]; e < cbeg[ch + 1]; ++e) {
        const unsigned int p = pos[e];
        const unsigned int v = p / F2_LM, j = p - v * F2_LM;
        const float z = __fadd_rn(__fadd_rn(E[(size_t)ids[p] * F2_LD + c], Q[(size_t)eidx[p] * F2_LD + c]), P[j * F2_LD + c]);
        const float s = tanhf(z);
        const float g = __fmul_rn(dl[(size_t)v * F2_LD + c], inv[v]);
        acc = __fadd_rn(acc, __fmul_rn(g, __fsub_rn(1.0f, __fmul_rn(s, s))));
    }
    part[(size_t)ch * F2_LD + c] = acc;
}
// f2_let_bwd_red: part 2. out[r, c] = the sum of row r's chunks (rbeg[r] .. rbeg[r + 1]), in order from 0.
extern "C" __global__ void f2_let_bwd_red(const float* __restrict__ part, const unsigned int* __restrict__ rbeg, float* __restrict__ out, int R) {
    const int r = blockIdx.x, c = threadIdx.x;
    if (r >= R || c >= F2_LD) return;
    float acc = 0.0f;
    for (unsigned int ch = rbeg[r]; ch < rbeg[r + 1]; ++ch) acc = __fadd_rn(acc, part[(size_t)ch * F2_LD + c]);
    out[(size_t)r * F2_LD + c] = acc;
}

// f2_cb_code: CB's granule code (v51b::cb_code from its mossy-fibre input u [N, U], U = 2d: her final normed state, then
// her word's normed embedding). One block per row, blockDim = B = min(1024, G) rounded up to a power of two (candle's
// reduce block for a row of G), dynamic shared memory G uint32 keys:
//   h_g = (u[i0] w0 + u[i2] w2) + (u[i1] w1 + u[i3] w3)   (F = 4 fixed inputs: candle's sum over a dim of 4, its tree)
//   kth = the K-th largest h (kwta_mask_rows's radix select over the order-preserving keys; ties all kept)
//   a_g = relu(h_g [h_g - kth >= 0])                       (h.mul(mask).relu())
//   len = sqrt(sum_g a_g^2) + 1e-6                          (the sum in candle's order: thread t from 0 over g = t, t+B,
//                                                            .., then the tree s = B/2 .. 1)
//   c_g = a_g / len, written in full (0 for the losers).
extern "C" __global__ void f2_cb_code(const float* __restrict__ u, const unsigned int* __restrict__ idx, const float* __restrict__ sgn,
                                      float* __restrict__ out, int N, int U, int G, int K) {
    extern __shared__ unsigned int keys[];
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix, s_k;
    __shared__ float red[1024];
    const int row = blockIdx.x;
    if (row >= N) return;
    const float* x = u + (size_t)row * U;
    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        const unsigned int* id = idx + (size_t)g * 4;
        const float* w = sgn + (size_t)g * 4;
        const float p0 = __fmul_rn(x[id[0]], w[0]), p1 = __fmul_rn(x[id[1]], w[1]);
        const float p2 = __fmul_rn(x[id[2]], w[2]), p3 = __fmul_rn(x[id[3]], w[3]);
        const float h = __fadd_rn(__fadd_rn(__fadd_rn(0.0f, p0), __fadd_rn(0.0f, p2)), __fadd_rn(__fadd_rn(0.0f, p1), __fadd_rn(0.0f, p3)));
        const unsigned int b = __float_as_uint(h);
        keys[g] = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    }
    if (threadIdx.x == 0) { s_prefix = 0u; s_k = (unsigned int)K; }
    __syncthreads();
    // the radix walk of kwta_kth (lif.cu, speed round 13c), on the keys already made
    unsigned int fixed = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        const unsigned int prefix = s_prefix;
        for (int i = threadIdx.x; i < G; i += blockDim.x) {
            const unsigned int kk = keys[i];
            if ((kk & fixed) == prefix) atomicAdd(&hist[(kk >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            const int l = threadIdx.x;
            const unsigned int k = s_k;
            unsigned int c[8], sum = 0u;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                c[j] = hist[255 - 8 * l - j];
                sum += c[j];
            }
            unsigned int incl = sum;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const unsigned int y = __shfl_up_sync(0xffffffffu, incl, off);
                if (l >= off) incl += y;
            }
            const unsigned int ballot = __ballot_sync(0xffffffffu, incl >= k);
            const int lsel = ballot ? __ffs(ballot) - 1 : 31;
            if (l == lsel) {
                unsigned int cum = incl - sum;
                int bin = 255 - 8 * l;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    bin = 255 - 8 * l - j;
                    if (bin == 0 || cum + c[j] >= k) break;
                    cum += c[j];
                }
                s_k = k - cum;
                s_prefix = prefix | ((unsigned int)bin << shift);
            }
        }
        __syncthreads();
        fixed |= 255u << shift;
    }
    const unsigned int kk = s_prefix;
    const float kth = __uint_as_float((kk & 0x80000000u) ? (kk ^ 0x80000000u) : ~kk);
    // a_g from the keys (the inverse map gives h back bit for bit), its square summed in candle's order
    float ss = 0.0f;
    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        const unsigned int k2 = keys[g];
        const float h = __uint_as_float((k2 & 0x80000000u) ? (k2 ^ 0x80000000u) : ~k2);
        const float a = (__fsub_rn(h, kth) >= 0.0f && h > 0.0f) ? h : 0.0f;
        ss = __fadd_rn(ss, __fmul_rn(a, a));
    }
    red[threadIdx.x] = ss;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s) red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + s]);
    }
    __syncthreads();
    const float len = __fadd_rn(__fmul_rn(sqrtf(red[0]), 1.0f), 1e-6f);
    float* o = out + (size_t)row * G;
    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        const unsigned int k2 = keys[g];
        const float h = __uint_as_float((k2 & 0x80000000u) ? (k2 ^ 0x80000000u) : ~k2);
        const float a = (__fsub_rn(h, kth) >= 0.0f && h > 0.0f) ? h : 0.0f;
        o[g] = __fdiv_rn(a, len);
    }
}
