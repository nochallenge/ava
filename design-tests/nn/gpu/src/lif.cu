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

extern "C" __global__ void lif_fwd(const float* __restrict__ cur, const float* __restrict__ beta,
                                   float* __restrict__ s, int B, int T, int D, float theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * D) return;
    int b = i / D, c = i - b * D;
    const float bt = beta[c];
    const float* x = cur + (size_t)b * T * D + c;
    float* o = s + (size_t)b * T * D + c;
    float v = 0.0f, sp = 0.0f;
    for (int t = 0; t < T; ++t) {
        v = __fsub_rn(__fadd_rn(__fmul_rn(bt, v), x[(size_t)t * D]), __fmul_rn(theta, sp));
        float st = __fsub_rn(v, theta) >= 0.0f ? 1.0f : 0.0f;
        o[(size_t)t * D] = st;
        sp = st;
    }
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
