// TESTING / DESIGN ONLY — ava_gpu::int8 row blocking: a batch's dense rows 6 at a time through one f32 copy of each int8 tile.
//
// Batched int8 products (a prompt read in one pass, a scored window) were slower than candle's f32 matmul: the per-row
// kernel of src/int8.rs converts every int8 weight to f32 again for every row (8 sign-extends and 8 int-to-float
// converts per input j for 64 columns, against 8 multiply-adds), so a batch paid the conversion once per row. Here a
// work unit is one 64-column tile and up to CHUNK consecutive rows. The tile is converted to f32 once per thread (kept
// while the thread's next unit reads the same tile), laid out as [4 sub-tiles][din][16], and the unit's dense rows run
// 6 at a time through an AVX2 6-row x 16-column kernel (12 accumulators, 2 weight loads and 6 broadcasts per j), the
// last 1..5 rows through the same kernel with fewer rows. The unit's sparse rows keep the per-row event sums (E2) on the
// int8 tile, which is 4 times smaller than the f32 copy and so stays nearer for their scattered reads. In a batch a
// row runs as an event sum only up to BLOCK_EVENT_MAX non-zero inputs (the spikes); denser rows (the kWTA'd MLP units)
// are cheaper through the blocked kernel. Units are handed out one at a time from a shared counter, so the faster
// cores of a hybrid CPU take more of them.
//
// The same bits as the per-row kernel: int8 to f32 is exact, and every output still starts from +0, adds
// fma(x[j], w[j][c], acc) for j = 0, 1, ... (all j for a dense row, the listed non-zero j in increasing order for a
// sparse one, which gives the dense sum's bits, see src/int8.rs) and is multiplied by its column's scale at the end.
// Only the loop order across rows and columns changes, never the order of one output's sum. Used when a product has
// at least BLOCK_MIN rows that run dense and the CPU has AVX2 + FMA; else the per-row path.

use super::{Lists, QW, PAR_MIN, TILE};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

/// Rows per call of the blocked kernel.
pub const MR: usize = 6;
/// Columns per call of the blocked kernel: a quarter of a tile.
pub const SUB: usize = 16;
const SUBS: usize = TILE / SUB;
/// Rows per work unit.
pub const CHUNK: usize = 48;
/// A product runs blocked when at least this many of its rows run dense.
pub const BLOCK_MIN: usize = 6;
/// In a blocked product, a row runs as an event sum only when at most this share of its inputs is non-zero.
pub const BLOCK_EVENT_MAX: f64 = 0.1;

static BLOCK_EVENT: AtomicU64 = AtomicU64::new(u64::MAX);

/// The share in use (BLOCK_EVENT_MAX unless `set_block_event_max` changed it).
pub fn block_event_max() -> f64 {
    match BLOCK_EVENT.load(Relaxed) {
        u64::MAX => BLOCK_EVENT_MAX,
        v => f64::from_bits(v),
    }
}

/// Set the share for this process (benchmarks). It changes the speed only, never a bit.
pub fn set_block_event_max(v: f64) {
    BLOCK_EVENT.store(v.to_bits(), Relaxed);
}

thread_local! {
    /// One f32 tile, [SUBS][din][SUB], per thread (candle's pool threads live on, so the buffer is reused).
    static BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

impl QW {
    /// y [m, dout] = x [m, din] · W through the row-blocked kernel (the caller checked `super::simd_ok`); `lists`:
    /// each row's non-zero inputs when it runs as an event sum (None: it runs through the 6-row kernel).
    pub(super) fn matmul_blocked(&self, x: &[f32], m: usize, y: &mut [f32], lists: &Lists, threads: usize) {
        let din = self.din;
        let dense: Vec<bool> = (0..m).map(|r| lists.get(r).is_none()).collect();
        let work: usize = lists.steps(din) * self.tiles * TILE;
        let nch = m.div_ceil(CHUNK);
        let units = self.tiles * nch;
        let yp = y.as_mut_ptr() as usize;
        let next = AtomicUsize::new(0);
        // units in tile-major order (unit u: tile u / nch, the rows of chunk u % nch), taken one at a time
        let run = || {
            BUF.with(|cell| {
                let mut buf = cell.borrow_mut();
                if buf.len() < din * TILE {
                    buf.resize(din * TILE, 0.0);
                }
                let mut have = usize::MAX;
                loop {
                    let u = next.fetch_add(1, Relaxed);
                    if u >= units {
                        break;
                    }
                    let (t, ch) = (u / nch, u % nch);
                    let (r0, r1) = (ch * CHUNK, ((ch + 1) * CHUNK).min(m));
                    if t != have && dense[r0..r1].iter().any(|&d| d) {
                        convert(&self.q[t * din * TILE..(t + 1) * din * TILE], din, &mut buf[..din * TILE]);
                        have = t;
                    }
                    // SAFETY: AVX2 + FMA were checked by the caller; the unit writes only rows r0..r1 of tile t's
                    // columns, which no other unit writes; the lengths of x and y were checked by matmul_rows_with
                    unsafe { self.chunk(x, lists, &dense, &buf[..din * TILE], t, r0, r1, yp as *mut f32) };
                }
            })
        };
        let want = threads.min(work / PAR_MIN).min(units);
        if want <= 1 {
            run();
        } else {
            let pool = candle_core::utils::barrier_pool();
            let nt = want.min(pool.n_workers() + 1).max(1);
            pool.execute(|tid| {
                if tid < nt {
                    run();
                }
            });
        }
    }

    /// Rows r0..r1 of tile t; the tile is in `buf` as f32 [SUBS][din][SUB] when some of these rows run dense.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn chunk(&self, x: &[f32], lists: &Lists, dense: &[bool], buf: &[f32], t: usize, r0: usize, r1: usize, y: *mut f32) {
        unsafe {
            let (din, dout) = (self.din, self.dout);
            let c0 = t * TILE;
            let n_out = TILE.min(dout - c0);
            let sc = self.scale.as_ptr().add(c0);
            let qt = &self.q[t * din * TILE..(t + 1) * din * TILE];
            let bp = buf.as_ptr();
            let mut rows_d = [0usize; CHUNK];
            let mut nd = 0;
            for r in r0..r1 {
                if dense[r] {
                    rows_d[nd] = r;
                    nd += 1;
                } else {
                    let out = std::slice::from_raw_parts_mut(y.add(r * dout + c0), n_out);
                    super::avx::tile(&x[r * din..(r + 1) * din], lists.get(r), qt, &self.scale[c0..c0 + TILE], out);
                }
            }
            let mut i = 0;
            while i + MR <= nd {
                rows::<MR>(&rows_d[i..i + MR], x, bp, din, sc, y, dout, c0, n_out);
                i += MR;
            }
            let rest = &rows_d[i..nd];
            match rest.len() {
                0 => {}
                1 => rows::<1>(rest, x, bp, din, sc, y, dout, c0, n_out),
                2 => rows::<2>(rest, x, bp, din, sc, y, dout, c0, n_out),
                3 => rows::<3>(rest, x, bp, din, sc, y, dout, c0, n_out),
                4 => rows::<4>(rest, x, bp, din, sc, y, dout, c0, n_out),
                _ => rows::<5>(rest, x, bp, din, sc, y, dout, c0, n_out),
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[allow(clippy::too_many_arguments)]
    unsafe fn chunk(&self, _: &[f32], _: &Lists, _: &[bool], _: &[f32], _: usize, _: usize, _: usize, _: *mut f32) {
        unreachable!("the blocked kernel needs AVX2 + FMA")
    }
}

/// R dense rows (indices `rs`) of one tile: every sub-tile that holds output columns.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn rows<const R: usize>(rs: &[usize], x: &[f32], bp: *const f32, din: usize, sc: *const f32, y: *mut f32, dout: usize, c0: usize, n_out: usize) {
    unsafe {
        let mut xs = [std::ptr::null::<f32>(); R];
        let mut outs = [std::ptr::null_mut::<f32>(); R];
        for k in 0..R {
            xs[k] = x.as_ptr().add(rs[k] * din);
            outs[k] = y.add(rs[k] * dout + c0);
        }
        let mut s = 0;
        while s < SUBS && s * SUB < n_out {
            let ncols = SUB.min(n_out - s * SUB);
            avx::mk::<R>(&xs, din, bp.add(s * din * SUB), sc.add(s * SUB), &outs, s * SUB, ncols);
            s += 1;
        }
    }
}

/// One int8 tile [din][TILE] as f32 [SUBS][din][SUB] (exact).
fn convert(qt: &[i8], din: usize, buf: &mut [f32]) {
    for j in 0..din {
        let row = &qt[j * TILE..(j + 1) * TILE];
        for s in 0..SUBS {
            let dst = &mut buf[(s * din + j) * SUB..(s * din + j + 1) * SUB];
            for (d, &q) in dst.iter_mut().zip(&row[s * SUB..(s + 1) * SUB]) {
                *d = q as f32;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use super::SUB;
    use std::arch::x86_64::*;

    /// R rows x 16 columns: out_k[off + c] = sc[c] · sum_j x_k[j] w[j][c] for c < ncols, j = 0..din in order, w the
    /// sub-tile [din][16] at bp.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub unsafe fn mk<const R: usize>(xs: &[*const f32; R], din: usize, bp: *const f32, sc: *const f32, outs: &[*mut f32; R], off: usize, ncols: usize) {
        unsafe {
            let mut a0 = [_mm256_setzero_ps(); R];
            let mut a1 = [_mm256_setzero_ps(); R];
            for j in 0..din {
                let b0 = _mm256_loadu_ps(bp.add(j * SUB));
                let b1 = _mm256_loadu_ps(bp.add(j * SUB + 8));
                for k in 0..R {
                    let xb = _mm256_set1_ps(*xs[k].add(j));
                    a0[k] = _mm256_fmadd_ps(xb, b0, a0[k]);
                    a1[k] = _mm256_fmadd_ps(xb, b1, a1[k]);
                }
            }
            let (s0, s1) = (_mm256_loadu_ps(sc), _mm256_loadu_ps(sc.add(8)));
            for k in 0..R {
                let (v0, v1) = (_mm256_mul_ps(a0[k], s0), _mm256_mul_ps(a1[k], s1));
                let o = outs[k].add(off);
                if ncols == SUB {
                    _mm256_storeu_ps(o, v0);
                    _mm256_storeu_ps(o.add(8), v1);
                } else {
                    let mut buf = [0f32; SUB];
                    _mm256_storeu_ps(buf.as_mut_ptr(), v0);
                    _mm256_storeu_ps(buf.as_mut_ptr().add(8), v1);
                    std::ptr::copy_nonoverlapping(buf.as_ptr(), o, ncols);
                }
            }
        }
    }
}
