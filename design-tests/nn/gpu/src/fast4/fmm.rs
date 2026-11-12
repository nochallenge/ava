// TESTING / DESIGN ONLY — ava_gpu::fast4::fmm: the f32 one-row products of the fused step with int8 off, in the order
// gemm 0.19 computes them on an AVX2 + FMA CPU (candle's CPU matmul hands every product to gemm):
// * x · W (W [k, n] row-major, the weights as the State path multiplies them): gemm's column-major gemv, each output
//   x0·w0 and then fma(w_j, x_j, out) for j = 1, 2, ...; k = 2: (x1·w1) + (x0·w0); k = 1: x0·w0 (its gevv). Every
//   output's chain is its own, so the outputs can be split over threads in any way (here: column ranges).
// * x · E^T (E [V, d] row-major, the tied head as `int8::head` multiplies it with int8 off): gemm's row-major gemv,
//   each output a dot product over d with eight 8-lane accumulators over 64-wide steps (fma), summed pairwise
//   ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7)), then fma over the remaining 8-wide steps, the 8 lanes summed one after
//   another from lane 0, then fma over the last d mod 8 values. Rows split over threads in any way.
// Checked bit for bit against candle's `matmul` in fast4/fmm_tests.rs (many shapes, k = 1, 2, n mod 8 != 0).

/// Products of fewer multiply-adds than this run on the calling thread (the pool's wake-up costs more).
const PAR_MIN: usize = 1 << 16;

#[derive(Clone, Copy)]
struct Ptr(*mut f32);
// SAFETY: each thread writes its own disjoint range of the output through it (see `par_ranges`)
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

/// f(a, b) over disjoint ranges covering 0..n (each a multiple of `align` long but the last), on up to `threads`
/// threads of candle's barrier pool (the pool the int8 products use), or on the calling thread.
pub(crate) fn par_ranges(n: usize, align: usize, threads: usize, work: usize, f: impl Fn(usize, usize) + Sync) {
    let pool = candle_core::utils::barrier_pool();
    let nt = threads.min(pool.n_workers() + 1).min(n.div_ceil(align.max(1))).max(1);
    if nt <= 1 || work < PAR_MIN {
        f(0, n);
        return;
    }
    let per = n.div_ceil(nt).div_ceil(align) * align;
    pool.execute(|tid| {
        if tid < nt {
            let (a, b) = ((tid * per).min(n), ((tid + 1) * per).min(n));
            if a < b {
                f(a, b);
            }
        }
    });
}

/// out = x · W for W [x.len(), n] row-major (gemm's one-row order, above), on up to `threads` threads.
pub(crate) fn xw(x: &[f32], w: &[f32], n: usize, threads: usize) -> Vec<f32> {
    let k = x.len();
    assert!(w.len() >= k * n, "fmm::xw: weights [{}] for [{k}, {n}]", w.len());
    let mut out = vec![0f32; n];
    let op = Ptr(out.as_mut_ptr());
    par_ranges(n, 64, threads, k * n, |a, b| {
        let op = op;
        // SAFETY: [a, b) is this call's own range of `out`
        let o = unsafe { std::slice::from_raw_parts_mut(op.0.add(a), b - a) };
        xw_cols(x, w, n, a, o);
    });
    out
}

/// Outputs c0 .. c0 + o.len() of x · W (W's rows n long).
pub(crate) fn xw_cols(x: &[f32], w: &[f32], n: usize, c0: usize, o: &mut [f32]) {
    let k = x.len();
    match k {
        0 => o.fill(0.0),
        1 => {
            for (j, v) in o.iter_mut().enumerate() {
                *v = w[c0 + j] * x[0];
            }
        }
        2 => {
            for (j, v) in o.iter_mut().enumerate() {
                let c = c0 + j;
                *v = w[n + c] * x[1] + w[c] * x[0];
            }
        }
        _ => {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: the fused step runs only where AVX2 + FMA were detected (Model::fused_blocker)
            unsafe {
                avx::cols(x, w, n, c0, o)
            }
            #[cfg(not(target_arch = "x86_64"))]
            for (j, v) in o.iter_mut().enumerate() {
                let c = c0 + j;
                let mut acc = w[c] * x[0];
                for i in 1..k {
                    acc = w[i * n + c].mul_add(x[i], acc);
                }
                *v = acc;
            }
        }
    }
}

/// out = x · E^T for E [v, x.len()] row-major (gemm's row-major gemv, above), on up to `threads` threads.
pub(crate) fn xet(x: &[f32], e: &[f32], v: usize, threads: usize) -> Vec<f32> {
    let d = x.len();
    assert!(e.len() >= v * d, "fmm::xet: [{}] for [{v}, {d}]", e.len());
    let mut out = vec![0f32; v];
    let op = Ptr(out.as_mut_ptr());
    par_ranges(v, 16, threads, v * d, |a, b| {
        let op = op;
        for r in a..b {
            // SAFETY: row r is in this call's own range
            unsafe { *op.0.add(r) = dot_rm(&e[r * d..(r + 1) * d], x) };
        }
    });
    out
}

/// One output of gemm's row-major gemv: sum_j a_j b_j in its order (above).
pub(crate) fn dot_rm(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: as `xw_cols`
    unsafe {
        avx::dot(a, b)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let k = a.len();
        let mut acc = [[0f32; 8]; 8];
        let mut j = 0;
        while j + 64 <= k {
            for (u, ac) in acc.iter_mut().enumerate() {
                for l in 0..8 {
                    ac[l] = a[j + 8 * u + l].mul_add(b[j + 8 * u + l], ac[l]);
                }
            }
            j += 64;
        }
        let add = |p: [f32; 8], q: [f32; 8]| -> [f32; 8] { std::array::from_fn(|l| p[l] + q[l]) };
        let mut s = add(add(add(acc[0], acc[1]), add(acc[2], acc[3])), add(add(acc[4], acc[5]), add(acc[6], acc[7])));
        while j + 8 <= k {
            for l in 0..8 {
                s[l] = a[j + l].mul_add(b[j + l], s[l]);
            }
            j += 8;
        }
        let mut r = s[0];
        for v in &s[1..] {
            r += *v;
        }
        while j < k {
            r = a[j].mul_add(b[j], r);
            j += 1;
        }
        r * 1.0
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::*;

    /// `xw_cols` for k >= 3: 64 outputs at a time (eight chains side by side), then 32, then 8, then the last ones in
    /// one masked vector (each output's chain is the same whatever runs beside it).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn cols(x: &[f32], w: &[f32], n: usize, c0: usize, o: &mut [f32]) {
        unsafe {
            let m = o.len();
            assert!(x.len() >= 1 && c0 + m <= n && w.len() >= x.len() * n);
            let wp = w.as_ptr().add(c0);
            let op = o.as_mut_ptr();
            let mut c = 0;
            while c + 64 <= m {
                block::<8>(x, wp, n, c, op);
                c += 64;
            }
            while c + 32 <= m {
                block::<4>(x, wp, n, c, op);
                c += 32;
            }
            while c + 8 <= m {
                block::<1>(x, wp, n, c, op);
                c += 8;
            }
            let t = m - c;
            if t > 0 {
                let lanes: [i32; 8] = std::array::from_fn(|i| if i < t { -1 } else { 0 });
                let mask = _mm256_loadu_si256(lanes.as_ptr() as *const __m256i);
                let p = wp.add(c);
                let mut acc = _mm256_mul_ps(_mm256_maskload_ps(p, mask), _mm256_set1_ps(x[0]));
                for (j, &xj) in x.iter().enumerate().skip(1) {
                    acc = _mm256_fmadd_ps(_mm256_maskload_ps(p.add(j * n), mask), _mm256_set1_ps(xj), acc);
                }
                _mm256_maskstore_ps(op.add(c), mask, acc);
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn block<const V: usize>(x: &[f32], wp: *const f32, n: usize, c: usize, op: *mut f32) {
        unsafe {
            let x0 = _mm256_set1_ps(x[0]);
            let mut a: [__m256; V] = std::array::from_fn(|v| _mm256_mul_ps(_mm256_loadu_ps(wp.add(c + 8 * v)), x0));
            for (j, &xj) in x.iter().enumerate().skip(1) {
                let xj = _mm256_set1_ps(xj);
                let r = wp.add(j * n + c);
                for (v, av) in a.iter_mut().enumerate() {
                    *av = _mm256_fmadd_ps(_mm256_loadu_ps(r.add(8 * v)), xj, *av);
                }
            }
            for (v, av) in a.iter().enumerate() {
                _mm256_storeu_ps(op.add(c + 8 * v), *av);
            }
        }
    }

    /// `super::dot_rm` (gemm's row-major gemv for one output).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let k = a.len().min(b.len());
            let (ap, bp) = (a.as_ptr(), b.as_ptr());
            let mut acc = [_mm256_setzero_ps(); 8];
            let mut j = 0;
            while j + 64 <= k {
                for (u, ac) in acc.iter_mut().enumerate() {
                    *ac = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j + 8 * u)), _mm256_loadu_ps(bp.add(j + 8 * u)), *ac);
                }
                j += 64;
            }
            let a01 = _mm256_add_ps(acc[0], acc[1]);
            let a23 = _mm256_add_ps(acc[2], acc[3]);
            let a45 = _mm256_add_ps(acc[4], acc[5]);
            let a67 = _mm256_add_ps(acc[6], acc[7]);
            let a03 = _mm256_add_ps(a01, a23);
            let a47 = _mm256_add_ps(a45, a67);
            let mut s = _mm256_add_ps(a03, a47);
            while j + 8 <= k {
                s = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(j)), _mm256_loadu_ps(bp.add(j)), s);
                j += 8;
            }
            let mut l = [0f32; 8];
            _mm256_storeu_ps(l.as_mut_ptr(), s);
            let mut r = l[0];
            for v in &l[1..] {
                r += *v;
            }
            while j < k {
                r = (*ap.add(j)).mul_add(*bp.add(j), r);
                j += 1;
            }
            r * 1.0
        }
    }
}
