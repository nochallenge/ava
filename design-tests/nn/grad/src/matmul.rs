// TESTING / DESIGN ONLY — ava_grad::matmul
// The GEMM kernels behind Graph::matmul / matmul_t and their backward passes, plus the thread-count
// setting and a small row-parallel helper used by the heavier row-wise ops.
//
// Design:
// * One core operation: C[m,n] += op(A)[m,k] · op(B)[k,n], op = identity or transpose.
// * B is packed once per call into column panels NR = 16 wide ([panel][k][16], zero padded), so the
//   micro-kernel reads B contiguously. A transposed A is materialised (cheap: m·k) before the call.
// * The micro-kernel keeps an MR×NR = 4×16 tile of C in registers and walks k: per step one 16-wide B
//   row and four broadcast A values → 4×16 multiply-adds. It is plain Rust over fixed-size arrays so
//   LLVM vectorises it; on x86-64 a copy compiled with AVX2+FMA is chosen at run time.
// * Cache blocking: k in blocks of KC = 256 (a panel slice is 16 KB, L1), rows in blocks of MC = 64
//   (an A block is 64 KB, L2).
// * Threads: std::thread::scope, output rows split in multiples of MR, one contiguous row range per thread.
//   Every output element is produced by exactly one thread with the same k order, so results are
//   bit-identical for any thread count (they can differ between CPUs with and without FMA).

use std::sync::OnceLock;

pub const MR: usize = 4;
pub const NR: usize = 16;
const KC: usize = 256;
const MC: usize = 64;
/// Minimum multiply-adds per thread before another thread is worth spawning.
const MIN_MACS_PER_THREAD: usize = 1 << 20;
/// Minimum estimated work (rows × work_per_row, roughly flops) per thread before the row-parallel helpers
/// (`par_rows`, `par_rows2`) spawn another thread; spawning an OS thread costs tens of microseconds.
pub(crate) const ROW_WORK_PER_THREAD: usize = 1 << 18;

/// Upper bound on worker threads (`threads()` and `Graph::set_threads` clamp to 1..=MAX_THREADS).
pub const MAX_THREADS: usize = 32;

/// Worker threads for matmul and row-parallel ops: env AVA_THREADS, default 4, clamped to 1..=32.
pub fn threads() -> usize {
    static T: OnceLock<usize> = OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("AVA_THREADS").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(4).clamp(1, MAX_THREADS)
    })
}

fn plan_threads(rows: usize, macs: usize, max_threads: usize) -> usize {
    let by_work = (macs / MIN_MACS_PER_THREAD).max(1);
    let by_rows = rows.div_ceil(MR).max(1);
    max_threads.max(1).min(by_work).min(by_rows)
}

/// C[m,n] += op(A) · op(B).
/// A is stored [m,k] (or [k,m] if `trans_a`), B is stored [k,n] (or [n,k] if `trans_b`), C is [m,n].
#[allow(clippy::too_many_arguments)]
pub fn gemm(m: usize, k: usize, n: usize, a: &[f32], trans_a: bool, b: &[f32], trans_b: bool, c: &mut [f32], threads: usize) {
    assert_eq!(a.len(), m * k, "gemm: A has {} values, expected {}", a.len(), m * k);
    assert_eq!(b.len(), k * n, "gemm: B has {} values, expected {}", b.len(), k * n);
    assert_eq!(c.len(), m * n, "gemm: C has {} values, expected {}", c.len(), m * n);
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let t = plan_threads(m, m * k * n, threads);
    let a_t;
    let a = if trans_a {
        a_t = transpose2d_par(a, k, m, t);
        &a_t[..]
    } else {
        a
    };
    let packed = pack_b(b, k, n, trans_b, t);
    if t <= 1 {
        gemm_rows(a, &packed, c, m, k, n);
        return;
    }
    let rows_per = m.div_ceil(MR).div_ceil(t) * MR; // a multiple of MR
    let packed = &packed[..];
    std::thread::scope(|s| {
        let mut chunks: Vec<(usize, &mut [f32])> = c.chunks_mut(rows_per * n).enumerate().collect();
        let last = chunks.pop();
        for (ci, chunk) in chunks {
            let rows = chunk.len() / n;
            let a_rows = &a[ci * rows_per * k..(ci * rows_per + rows) * k];
            s.spawn(move || gemm_rows(a_rows, packed, chunk, rows, k, n));
        }
        if let Some((ci, chunk)) = last {
            let rows = chunk.len() / n;
            gemm_rows(&a[ci * rows_per * k..(ci * rows_per + rows) * k], packed, chunk, rows, k, n);
        }
    });
}

/// Batched GEMM: for each b in 0..batch, C_b += op(A_b) · op(B_b), with A_b, B_b, C_b consecutive blocks
/// of m·k, k·n and m·n values. Threads go across batches when there are enough of them, otherwise
/// across the rows of each product.
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    trans_a: bool,
    b: &[f32],
    trans_b: bool,
    c: &mut [f32],
    threads: usize,
) {
    let (sa, sb, sc) = (m * k, k * n, m * n);
    assert_eq!(a.len(), batch * sa, "gemm_batched: A size");
    assert_eq!(b.len(), batch * sb, "gemm_batched: B size");
    assert_eq!(c.len(), batch * sc, "gemm_batched: C size");
    if batch == 0 || sc == 0 || k == 0 {
        return;
    }
    let t = threads.max(1).min((batch * m * k * n / MIN_MACS_PER_THREAD).max(1));
    if t <= 1 || batch < t {
        for bi in 0..batch {
            gemm(m, k, n, &a[bi * sa..(bi + 1) * sa], trans_a, &b[bi * sb..(bi + 1) * sb], trans_b, &mut c[bi * sc..(bi + 1) * sc], t);
        }
        return;
    }
    let per = batch.div_ceil(t);
    std::thread::scope(|s| {
        for (ci, chunk) in c.chunks_mut(per * sc).enumerate() {
            let nb = chunk.len() / sc;
            let b0 = ci * per;
            let a = &a[b0 * sa..(b0 + nb) * sa];
            let b = &b[b0 * sb..(b0 + nb) * sb];
            s.spawn(move || {
                for i in 0..nb {
                    gemm(m, k, n, &a[i * sa..(i + 1) * sa], trans_a, &b[i * sb..(i + 1) * sb], trans_b, &mut chunk[i * sc..(i + 1) * sc], 1);
                }
            });
        }
    });
}

/// out[c][r] = x[r][c] for x stored [rows, cols].
pub fn transpose2d(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    transpose2d_par(x, rows, cols, 1)
}

/// Copying work below this many elements stays on one thread.
const PAR_COPY_MIN: usize = 1 << 18;

fn transpose2d_par(x: &[f32], rows: usize, cols: usize, threads: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * cols);
    let mut out = vec![0.0f32; x.len()];
    if x.is_empty() {
        return out;
    }
    let t = if x.len() >= PAR_COPY_MIN { threads.clamp(1, cols) } else { 1 };
    let per = cols.div_ceil(t); // output rows (= input columns) per thread
    let work = |c0: usize, dst: &mut [f32]| {
        let c1 = c0 + dst.len() / rows;
        const B: usize = 32;
        for r0 in (0..rows).step_by(B) {
            for cb in (c0..c1).step_by(B) {
                for r in r0..(r0 + B).min(rows) {
                    for c in cb..(cb + B).min(c1) {
                        dst[(c - c0) * rows + r] = x[r * cols + c];
                    }
                }
            }
        }
    };
    if t <= 1 {
        work(0, &mut out);
    } else {
        let work = &work;
        std::thread::scope(|s| {
            for (ci, chunk) in out.chunks_mut(per * rows).enumerate() {
                s.spawn(move || work(ci * per, chunk));
            }
        });
    }
    out
}

/// Pack op(B) ([k,n]) into ceil(n/NR) panels laid out [panel][k][NR], zero padded on the right.
fn pack_b(b: &[f32], k: usize, n: usize, trans: bool, threads: usize) -> Vec<f32> {
    let panels = n.div_ceil(NR);
    let mut out = vec![0.0f32; panels * k * NR];
    let t = if k * n >= PAR_COPY_MIN { threads.clamp(1, panels.max(1)) } else { 1 };
    if t <= 1 {
        pack_panels(b, k, n, trans, 0, &mut out);
    } else {
        let per = panels.div_ceil(t);
        std::thread::scope(|s| {
            for (ci, chunk) in out.chunks_mut(per * k * NR).enumerate() {
                s.spawn(move || pack_panels(b, k, n, trans, ci * per, chunk));
            }
        });
    }
    out
}

/// Pack the panels p0.. that fit in `dst_all` (whole panels of k·NR values each).
fn pack_panels(b: &[f32], k: usize, n: usize, trans: bool, p0: usize, dst_all: &mut [f32]) {
    for (pi, dst) in dst_all.chunks_mut(k * NR).enumerate() {
        let j0 = (p0 + pi) * NR;
        let w = NR.min(n - j0);
        if !trans {
            for kk in 0..k {
                dst[kk * NR..kk * NR + w].copy_from_slice(&b[kk * n + j0..kk * n + j0 + w]);
            }
        } else {
            // b stored [n, k]
            for jj in 0..w {
                let row = &b[(j0 + jj) * k..(j0 + jj + 1) * k];
                for (kk, &v) in row.iter().enumerate() {
                    dst[kk * NR + jj] = v;
                }
            }
        }
    }
}

/// Single-threaded: c[rows, n] += a[rows, k] · B (packed). Picks the AVX2+FMA build when available.
fn gemm_rows(a: &[f32], packed: &[f32], c: &mut [f32], rows: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), rows * k);
    debug_assert_eq!(c.len(), rows * n);
    debug_assert_eq!(packed.len(), n.div_ceil(NR) * k * NR);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: the CPU supports AVX2 and FMA (checked at run time).
            unsafe { gemm_rows_avx2(a, packed, c, rows, k, n) };
            return;
        }
    }
    gemm_rows_generic::<false>(a, packed, c, rows, k, n);
}

#[cfg(target_arch = "x86_64")]
fn has_avx2_fma() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_rows_avx2(a: &[f32], packed: &[f32], c: &mut [f32], rows: usize, k: usize, n: usize) {
    gemm_rows_generic::<true>(a, packed, c, rows, k, n);
}

/// The blocked loop nest. `FMA` selects fused multiply-add (only used where the CPU has it).
#[inline(always)]
fn gemm_rows_generic<const FMA: bool>(a: &[f32], packed: &[f32], c: &mut [f32], rows: usize, k: usize, n: usize) {
    let panels = n.div_ceil(NR);
    let ap = a.as_ptr();
    let bp = packed.as_ptr();
    let cp = c.as_mut_ptr();
    let mut kb = 0;
    while kb < k {
        let kc = KC.min(k - kb);
        let mut ib = 0;
        while ib < rows {
            let iend = (ib + MC).min(rows);
            for p in 0..panels {
                let j0 = p * NR;
                let w = NR.min(n - j0);
                // SAFETY: every pointer below stays inside its slice: rows i..i+R < rows, columns
                // kb..kb+kc ≤ k of A, panel p rows kb..kb+kc of the packed B, columns j0..j0+w ≤ n of C.
                unsafe {
                    let bpanel = bp.add((p * k + kb) * NR);
                    let mut i = ib;
                    while i + MR <= iend {
                        kernel::<MR, FMA>(kc, ap.add(i * k + kb), k, bpanel, cp.add(i * n + j0), n, w);
                        i += MR;
                    }
                    match iend - i {
                        3 => kernel::<3, FMA>(kc, ap.add(i * k + kb), k, bpanel, cp.add(i * n + j0), n, w),
                        2 => kernel::<2, FMA>(kc, ap.add(i * k + kb), k, bpanel, cp.add(i * n + j0), n, w),
                        1 => kernel::<1, FMA>(kc, ap.add(i * k + kb), k, bpanel, cp.add(i * n + j0), n, w),
                        _ => {}
                    }
                }
            }
            ib = iend;
        }
        kb += kc;
    }
}

/// R×16 register tile: acc = Σ_kk A[r, kk]·Bpanel[kk, :], then C[r, 0..w] += acc[r, 0..w].
#[inline(always)]
unsafe fn kernel<const R: usize, const FMA: bool>(kc: usize, a: *const f32, lda: usize, b: *const f32, c: *mut f32, ldc: usize, w: usize) {
    let mut acc = [[0.0f32; NR]; R];
    for kk in 0..kc {
        let brow: &[f32; NR] = &*(b.add(kk * NR) as *const [f32; NR]);
        for (r, accr) in acc.iter_mut().enumerate() {
            let av = *a.add(r * lda + kk);
            for j in 0..NR {
                accr[j] = if FMA { av.mul_add(brow[j], accr[j]) } else { av * brow[j] + accr[j] };
            }
        }
    }
    for (r, accr) in acc.iter().enumerate() {
        let crow = c.add(r * ldc);
        for (j, &v) in accr.iter().enumerate().take(w) {
            *crow.add(j) += v;
        }
    }
}

/// Run `f(first_row, rows_chunk)` over `out` split into whole rows of `row_len`, in parallel when the
/// estimated work (rows × `work_per_row`) is large enough. Each row is handled by exactly one call, so
/// results never depend on the thread count.
pub(crate) fn par_rows<F>(out: &mut [f32], row_len: usize, work_per_row: usize, max_threads: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    if row_len == 0 || out.is_empty() {
        return;
    }
    let rows = out.len() / row_len;
    let work = rows.saturating_mul(work_per_row.max(1));
    let t = max_threads.max(1).min((work / ROW_WORK_PER_THREAD).max(1)).min(rows.max(1));
    if t <= 1 {
        f(0, out);
        return;
    }
    let per = rows.div_ceil(t);
    let f = &f;
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * row_len).enumerate() {
            s.spawn(move || f(ci * per, chunk));
        }
    });
}

/// Like `par_rows`, but over two row-aligned buffers at once (`a` with `a_len` values per row, `b` with
/// `b_len` per row, the same number of rows): `f(first_row, a_rows, b_rows)`. Used by ops that produce an
/// output and a per-row side result (norm statistics, the kWTA mask).
#[allow(clippy::too_many_arguments)]
pub(crate) fn par_rows2<A, B, F>(a: &mut [A], a_len: usize, b: &mut [B], b_len: usize, work_per_row: usize, max_threads: usize, f: F)
where
    A: Send,
    B: Send,
    F: Fn(usize, &mut [A], &mut [B]) + Sync,
{
    if a_len == 0 || a.is_empty() {
        return;
    }
    let rows = a.len() / a_len;
    debug_assert_eq!(b.len(), rows * b_len, "par_rows2: row counts differ");
    let work = rows.saturating_mul(work_per_row.max(1));
    let t = max_threads.max(1).min((work / ROW_WORK_PER_THREAD).max(1)).min(rows.max(1));
    if t <= 1 || b_len == 0 {
        f(0, a, b);
        return;
    }
    let per = rows.div_ceil(t);
    let f = &f;
    std::thread::scope(|s| {
        for (ci, (ca, cb)) in a.chunks_mut(per * a_len).zip(b.chunks_mut(per * b_len)).enumerate() {
            s.spawn(move || f(ci * per, ca, cb));
        }
    });
}

/// Elementwise kernels that are a few flops per element (add, mul, scale, relu, axpy) need this many
/// elements per thread before threading pays; transcendental kernels (gelu, tanh, sigmoid, exp) use
/// `ELEM_COSTLY`.
pub(crate) const ELEM_CHEAP: usize = 1 << 18;
pub(crate) const ELEM_COSTLY: usize = 1 << 15;

/// Run `f(first_index, chunk)` over `out` split into contiguous element ranges, in parallel when every
/// thread gets at least `min_per_thread` elements. Each element is written by exactly one call and the
/// per-element arithmetic does not depend on the split, so results never depend on the thread count.
pub(crate) fn par_chunks<F>(out: &mut [f32], min_per_thread: usize, max_threads: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let n = out.len();
    let t = max_threads.max(1).min((n / min_per_thread.max(1)).max(1));
    if t <= 1 {
        f(0, out);
        return;
    }
    let per = n.div_ceil(t);
    let f = &f;
    std::thread::scope(|s| {
        let mut chunks = out.chunks_mut(per).enumerate();
        let first = chunks.next();
        for (ci, chunk) in chunks {
            s.spawn(move || f(ci * per, chunk));
        }
        // the calling thread takes the first chunk instead of idling
        if let Some((ci, chunk)) = first {
            f(ci * per, chunk);
        }
    });
}

/// dst += alpha · src, threaded over element ranges.
pub(crate) fn axpy_par(dst: &mut [f32], alpha: f32, src: &[f32], max_threads: usize) {
    debug_assert_eq!(dst.len(), src.len());
    par_chunks(dst, ELEM_CHEAP, max_threads, |i0, chunk| {
        let n = chunk.len();
        for (d, s) in chunk.iter_mut().zip(&src[i0..i0 + n]) {
            *d += alpha * *s;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_path_matches_naive() {
        // exercise the non-FMA path directly (the public entry may pick AVX2)
        let (m, k, n) = (13, 300, 37);
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1).collect();
        let packed = pack_b(&b, k, n, false, 1);
        let mut c = vec![0.0f32; m * n];
        gemm_rows_generic::<false>(&a, &packed, &mut c, m, k, n);
        for i in 0..m {
            for j in 0..n {
                let want: f64 = (0..k).map(|kk| a[i * k + kk] as f64 * b[kk * n + j] as f64).sum();
                assert!((c[i * n + j] as f64 - want).abs() < 1e-3);
            }
        }
    }
}
