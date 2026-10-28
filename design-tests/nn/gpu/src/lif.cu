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
__device__ __forceinline__ unsigned short f2bf(float f) {
    unsigned int u = __float_as_uint(f);
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
template <typename O>
__device__ __forceinline__ void kwta_relu_impl(const float* __restrict__ u, O* __restrict__ a, int N, int H, int K, unsigned int* keys,
                                               unsigned int* hist, unsigned int& s_prefix, unsigned int& s_k) {
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
    const float kth = __uint_as_float((kk & 0x80000000u) ? (kk ^ 0x80000000u) : ~kk);
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
extern "C" __global__ void __launch_bounds__(128) attn_bwd_q_s(const float* __restrict__ qkv, const float* __restrict__ o,
                                      const float* __restrict__ lse, const float* __restrict__ dout, unsigned short* __restrict__ dqkv,
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
        if (i < T) dqkv[((size_t)b * T + i) * ld + h * AHS + lane] = f2bf(acc[r] * scale);
    }
}

// dK, dV into dqkv columns D..2D and 2D..3D (bf16).
extern "C" __global__ void __launch_bounds__(128) attn_bwd_kv_s(const float* __restrict__ qkv, const float* __restrict__ lse,
                                       const float* __restrict__ dout, const float* __restrict__ dvec, unsigned short* __restrict__ dqkv,
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
            dqkv[row + D] = f2bf(gk[r] * scale);
            dqkv[row + 2 * D] = f2bf(gv[r]);
        }
    }
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
template <typename OT>
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
        put(O, ((size_t)b * T + t) * d + h * 32 + lane, y * r * (g * hb_sig(g)));
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

// Backward of hebb_fwd: dP [N, ld] (every column written) for dO [N, d]. Chunks of HB_C steps from the last: the
// chunk's states are recomputed from its checkpoint into scratch [B*H, HB_C, 32, 32] (per warp, read back by the
// thread that wrote it), then walked back through.
// v40: the body is shared by hebb_bwd (NW = 4 heads per block of 128 threads) and hebb_bwd_w1 (one head per block of
// 32 threads, at most 168 registers): the same code per head, so the same bits.
template <int NW, typename OT>
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
            const float yn = y * rr, sg = hb_sig(g), silu = g * sg;
            const float dg = dov * yn * sg * (1.0f + g * (1.0f - sg));
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
template <typename OT>
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
            put(O, row * d + h * 32 + lane, y[t] * rt * (g * hb_sig(g)));
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

// Backward 1, one warp per chunk (lane = value channel j): the WY form and y again (as hc_prep and hc_out), then dO
// back through the gate and head norm (dg into dP, dY), X1 = M^T dY and X2 = Q^T diag(gamma) dY; W, Kd, gC for
// hc_bscan.
template <typename OT>
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
                const float sg = hb_sig(g), silu = g * sg;
                yn[t] = y[t] * rt;
                put(dP, row * ld + 3 * d + h * 32 + lane, dov * yn[t] * sg * (1.0f + g * (1.0f - sg)));
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
