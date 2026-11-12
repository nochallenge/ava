// TESTING / DESIGN ONLY — ava_gpu::int8: E10 int8 weights and E2 event-driven sums for Ava's talking path on the CPU.
//
// E10 (the reference math is ava_parts::quant): each weight matrix W [d_in, d_out] (applied x · W) keeps one f32
// scale per output column and its weights as int8, symmetric, round to nearest (ava_parts::quant::quantize_int8
// itself). The activations stay f32 (weights-only int8): y[c] = scale[c] · sum_j x[j] · q[j][c], the products of
// f32 inputs and exact small integers summed in f32 with fused multiply-adds in the order j = 0, 1, ..., d_in - 1.
// Every output row is computed on its own in that one order, so a row gives the same bits whether it is read alone
// (`Model::step`) or inside a batch (`Model::forward`), with any number of threads.
//
// E2 (ava_parts::event): for a row that is mostly zeros (the LIF spikes, the kWTA'd MLP units), the same sum runs over
// the non-zero inputs only. fma(0, q, acc) = acc exactly (acc starts at +0 and a sum of round-to-nearest adds is
// never -0), so the event sum gives the dense int8 sum bit for bit; it is chosen per row by its density
// (`event_max`), which changes only the speed.
//
// The kernel: the int8 weights are packed in column tiles of 64 (q[(tile · d_in + j) · 64 + c] = Wq[j][64·tile + c],
// columns past d_out are 0 with scale 1), so one row of one tile is one 64-byte cache line and a whole tile is read
// in order. AVX2 + FMA when the CPU has them (checked at run time; the build needs no target flags), else the same
// operations in scalar code (f32::mul_add): the two give the same bits. Threads: candle's own barrier pool
// (CANDLE_NUM_THREADS, which --threads sets), the (tile, row) work split among them; small products run on one.
//
// Row blocking (src/int8/block.rs, a batch of 6 or more dense rows, `set_block` / AVA_INT8_BLOCK=0 to switch off): each
// 64-column tile is turned into f32 once per work unit and the dense rows run 6 at a time through a 6-row x 16-column
// kernel, each output still summed in the order j = 0, 1, ... from +0 and scaled at the end, so the same bits; the
// mostly-zero rows keep their event sums on the int8 tile.
//
// Why not candle's quantized Q8_0 (candle_core::quantized): it quantizes the activations too (blocks of 32, an f16
// scale each), and its SIMD dot products are compiled in only with target-feature avx2, which this build does not set,
// so its CPU path here is scalar. `int8_bench` measures both.
//
// Use: `Model::set_int8(true)` builds a `Q8Set` (every block's LIF, memory, attention, engram and MLP matrices and the
// output head) once; the model's forward, step and state paths then route those matmuls through `mm` / `head` while
// a `Scope` holds the set on this thread. With int8 off (the default) `mm(x, w)` is `x.matmul(w)` and `head(x, e)` is
// `x.matmul(&e.t()?)`, the very ops of before, so every output is bit for bit unchanged.

use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor, TensorId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Output columns per packed tile.
pub const TILE: usize = 64;
/// Multiply-adds per thread: a product of fewer runs on the calling thread only, and a bigger one wakes one thread of
/// candle's pool per this many (waking a thread costs about a microsecond).
pub const PAR_MIN: usize = 1 << 16;

/// One weight matrix in int8: W [din, dout] (x · W), one f32 scale per output column, packed in column tiles.
pub struct QW {
    pub din: usize,
    pub dout: usize,
    tiles: usize,
    /// [tiles][din][TILE]
    q: Vec<i8>,
    /// [tiles · TILE]; 1 past dout
    scale: Vec<f32>,
}

impl QW {
    /// W [din, dout] row-major (as applied x · W).
    pub fn from_rows(w: &[f32], din: usize, dout: usize) -> QW {
        let r = ava_parts::quant::quantize_int8(w, din, dout);
        let tiles = dout.div_ceil(TILE);
        let mut q = vec![0i8; tiles * din * TILE];
        let mut scale = vec![1f32; tiles * TILE];
        scale[..dout].copy_from_slice(&r.scale);
        for j in 0..din {
            for c in 0..dout {
                let (t, cc) = (c / TILE, c % TILE);
                q[(t * din + j) * TILE + cc] = r.q[j * dout + c];
            }
        }
        QW { din, dout, tiles, q, scale }
    }

    /// The int8 copy of a weight tensor W [din, dout].
    pub fn from_tensor(w: &Tensor) -> Result<QW> {
        let (din, dout) = w.dims2()?;
        let v: Vec<f32> = w.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
        Ok(QW::from_rows(&v, din, dout))
    }

    /// The output head of a tied embedding E [V, d]: W = E^T [d, V], so one scale per vocabulary row of E.
    pub fn from_tensor_t(e: &Tensor) -> Result<QW> {
        QW::from_tensor(&e.t()?.contiguous()?)
    }

    /// The columns of `parts` side by side, [din, the douts summed]: whole tiles copied, so every output column
    /// is computed exactly as in its own part (the same bits; a row's event choice depends on the row alone). None
    /// unless every part has the same din and whole tiles (dout a multiple of TILE).
    pub fn concat(parts: &[&QW]) -> Option<QW> {
        let din = parts.first()?.din;
        if parts.iter().any(|p| p.din != din || p.dout % TILE != 0) {
            return None;
        }
        let dout: usize = parts.iter().map(|p| p.dout).sum();
        let mut q = Vec::with_capacity(din * dout);
        let mut scale = Vec::with_capacity(dout);
        for p in parts {
            q.extend_from_slice(&p.q);
            scale.extend_from_slice(&p.scale);
        }
        Some(QW { din, dout, tiles: dout / TILE, q, scale })
    }

    /// The dequantized weights [din, dout] row-major (q · scale, each rounded to f32): what the int8 copy stands for.
    pub fn dequant(&self) -> Vec<f32> {
        let mut w = vec![0f32; self.din * self.dout];
        for j in 0..self.din {
            for c in 0..self.dout {
                let (t, cc) = (c / TILE, c % TILE);
                w[j * self.dout + c] = self.q[(t * self.din + j) * TILE + cc] as f32 * self.scale[c];
            }
        }
        w
    }

    /// Bytes held (int8 weights with their padding, and the scales).
    pub fn bytes(&self) -> usize {
        self.q.len() + 4 * self.scale.len()
    }

    /// y [m, dout] = x [m, din] · W. `event_max`: a row whose share of non-zero inputs is at most this runs as an
    /// event sum (E2; the same bits as the dense sum). `threads`: at most this many (1 = the calling thread only).
    /// A batch runs row-blocked (src/int8/block.rs, the same bits) unless `set_block(false)` / AVA_INT8_BLOCK=0.
    pub fn matmul_rows(&self, x: &[f32], m: usize, y: &mut [f32], event_max: f64, threads: usize) {
        self.matmul_rows_with(x, m, y, event_max, threads, block_on())
    }

    /// E2: the event lists of x [m, din]: the rows with at most `event_max` · din non-zero inputs, each with its
    /// non-zero inputs (the other rows run dense); a big batch's rows are read on up to `threads` threads.
    #[cfg(test)]
    pub(crate) fn event_lists(&self, x: &[f32], m: usize, event_max: f64, threads: usize) -> Lists {
        let counts = self.nnz_rows(x, m, threads);
        self.lists_for(x, m, &counts, event_max, threads)
    }

    /// The number of non-zero inputs of each row of x [m, din].
    fn nnz_rows(&self, x: &[f32], m: usize, threads: usize) -> Vec<u32> {
        let din = self.din;
        par_rows(m, threads.min(m * din / PAR_MIN), |r| nnz(&x[r * din..(r + 1) * din]) as u32)
    }

    /// The event lists of the rows with at most `event_max` · din non-zero inputs (`counts` from `nnz_rows`).
    fn lists_for(&self, x: &[f32], m: usize, counts: &[u32], event_max: f64, threads: usize) -> Lists {
        let din = self.din;
        let lim = event_max * din as f64;
        let ev: Vec<bool> = counts.iter().map(|&c| c as f64 <= lim).collect();
        let mut off = Vec::with_capacity(m + 1);
        off.push(0usize);
        for r in 0..m {
            off.push(off[r] + if ev[r] { counts[r] as usize } else { 0 });
        }
        let mut idx = vec![0u32; off[m]];
        let ip = idx.as_mut_ptr() as usize;
        let scan = ev.iter().filter(|&&e| e).count() * din;
        par_for(m, threads.min(scan / PAR_MIN), |r| {
            if ev[r] {
                // SAFETY: row r alone writes its slots off[r]..off[r + 1], which lie inside idx
                let out = unsafe { std::slice::from_raw_parts_mut((ip as *mut u32).add(off[r]), off[r + 1] - off[r]) };
                nonzeros(&x[r * din..(r + 1) * din], out);
            }
        });
        Lists { idx, off, ev }
    }

    /// `matmul_rows` with the row blocking chosen here (`block` false: always the per-row kernel).
    pub fn matmul_rows_with(&self, x: &[f32], m: usize, y: &mut [f32], event_max: f64, threads: usize, block: bool) {
        assert_eq!(x.len(), m * self.din, "int8 matmul: input length");
        assert_eq!(y.len(), m * self.dout, "int8 matmul: output length");
        let din = self.din;
        let simd = simd_ok();
        let counts = self.nnz_rows(x, m, threads);
        // row blocking: a row runs as an event sum only up to the (lower) blocked share; the rest through the 6-row kernel
        if block && simd && m >= block::BLOCK_MIN {
            let lim = event_max.min(block::block_event_max());
            if counts.iter().filter(|&&c| c as f64 > lim * din as f64).count() >= block::BLOCK_MIN {
                let lists = self.lists_for(x, m, &counts, lim, threads);
                return self.matmul_blocked(x, m, y, &lists, threads);
            }
        }
        let lists = self.lists_for(x, m, &counts, event_max, threads);
        let work: usize = lists.steps(din) * self.tiles * TILE;
        let units = m * self.tiles;
        let yp = y.as_mut_ptr() as usize;
        // unit u: tile u / m, row u % m (tile-major, so a tile is reused across a batch's rows from the cache)
        let run = |u: usize| {
            let (t, r) = (u / m, u % m);
            let c0 = t * TILE;
            let n_out = TILE.min(self.dout - c0);
            let xr = &x[r * din..(r + 1) * din];
            let qt = &self.q[t * din * TILE..(t + 1) * din * TILE];
            let sc = &self.scale[c0..c0 + TILE];
            // SAFETY: each unit writes its own n_out outputs of row r, inside y (checked lengths above)
            let out = unsafe { std::slice::from_raw_parts_mut((yp as *mut f32).add(r * self.dout + c0), n_out) };
            tile(xr, lists.get(r), qt, sc, out, simd);
        };
        // one thread per PAR_MIN multiply-adds, at most `threads` (waking more costs more than they save)
        let want = threads.min(work / PAR_MIN).min(units);
        if want <= 1 {
            (0..units).for_each(run);
        } else {
            let pool = candle_core::utils::barrier_pool();
            let nt = want.min(pool.n_workers() + 1).max(1);
            let per = units.div_ceil(nt);
            pool.execute(|tid| {
                if tid < nt {
                    let (a, b) = (tid * per, ((tid + 1) * per).min(units));
                    (a..b).for_each(&run);
                }
            });
        }
    }
}

/// E2: a batch's event lists (`QW::event_lists`), in one buffer: row r runs as an event sum over
/// idx[off[r]..off[r + 1]] when ev[r], else dense.
#[derive(Debug, PartialEq)]
pub(crate) struct Lists {
    idx: Vec<u32>,
    off: Vec<usize>,
    ev: Vec<bool>,
}

impl Lists {
    /// Row r's non-zero inputs (increasing) when it runs as an event sum; None when it runs dense.
    pub(crate) fn get(&self, r: usize) -> Option<&[u32]> {
        if self.ev[r] { Some(&self.idx[self.off[r]..self.off[r + 1]]) } else { None }
    }

    /// The product's multiply-adds per output column: din per dense row, the list's length per sparse one.
    pub(crate) fn steps(&self, din: usize) -> usize {
        self.ev.iter().filter(|&&e| !e).count() * din + self.idx.len()
    }
}

/// f(r) for r in 0..n, on up to `nt` threads of candle's pool (1 or less: the calling thread), each thread a range.
fn par_for(n: usize, nt: usize, f: impl Fn(usize) + Sync) {
    let nt = nt.min(n);
    if nt <= 1 {
        (0..n).for_each(f);
        return;
    }
    let pool = candle_core::utils::barrier_pool();
    let nt = nt.min(pool.n_workers() + 1);
    let per = n.div_ceil(nt);
    pool.execute(|tid| {
        if tid < nt {
            ((tid * per).min(n)..((tid + 1) * per).min(n)).for_each(&f);
        }
    });
}

/// f(r) for r in 0..n into a Vec, on up to `nt` threads (`par_for`).
fn par_rows<T: Send + Copy + Default>(n: usize, nt: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let mut out = vec![T::default(); n];
    let op = out.as_mut_ptr() as usize;
    // SAFETY: each r < n is written by one thread only
    par_for(n, nt, |r| unsafe { *(op as *mut T).add(r) = f(r) });
    out
}

/// The number of inputs != 0 (-0 counts as 0, NaN as non-zero, as `v != 0.0`).
fn nnz(row: &[f32]) -> usize {
    #[cfg(target_arch = "x86_64")]
    if simd_ok() {
        // SAFETY: AVX2 was detected
        return unsafe { avx::nnz(row) };
    }
    row.iter().filter(|&&v| v != 0.0).count()
}

/// Write the indices of the inputs != 0, in increasing order, into `out` (of exactly `nnz(row)` slots).
fn nonzeros(row: &[f32], out: &mut [u32]) {
    #[cfg(target_arch = "x86_64")]
    if simd_ok() {
        // SAFETY: AVX2 was detected
        let n = unsafe { avx::nonzeros(row, out) };
        assert_eq!(n, out.len(), "int8 event list: count");
        return;
    }
    let mut n = 0;
    for (j, _) in row.iter().enumerate().filter(|(_, &v)| v != 0.0) {
        out[n] = j as u32;
        n += 1;
    }
    assert_eq!(n, out.len(), "int8 event list: count");
}

/// Whether the AVX2 + FMA kernel can run on this CPU.
pub fn simd_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *OK.get_or_init(|| std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// One row x [din] times one packed tile qt [din, 64]: out[c] = sc[c] · sum_j x[j] q[j][c] over every j (`idx` None) or
/// over the listed non-zero j in increasing order (the same bits).
fn tile(x: &[f32], idx: Option<&[u32]>, qt: &[i8], sc: &[f32], out: &mut [f32], simd: bool) {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: AVX2 and FMA were detected; qt holds din rows of TILE and sc TILE values; out has <= TILE slots
        unsafe { avx::tile(x, idx, qt, sc, out) };
        return;
    }
    let _ = simd;
    tile_scalar(x, idx, qt, sc, out);
}

/// `tile` in scalar code: fused multiply-adds in the same order as the AVX2 kernel, so the same bits.
pub fn tile_scalar(x: &[f32], idx: Option<&[u32]>, qt: &[i8], sc: &[f32], out: &mut [f32]) {
    let mut acc = [0f32; TILE];
    let mut add = |j: usize| {
        let xj = x[j];
        let row = &qt[j * TILE..(j + 1) * TILE];
        for (a, &q) in acc.iter_mut().zip(row) {
            *a = xj.mul_add(q as f32, *a);
        }
    };
    match idx {
        Some(js) => js.iter().for_each(|&j| add(j as usize)),
        None => (0..x.len()).for_each(add),
    }
    for (c, o) in out.iter_mut().enumerate() {
        *o = acc[c] * sc[c];
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use super::TILE;
    use std::arch::x86_64::*;

    /// `super::nnz`: 8 inputs per compare (not-equal, unordered: NaN counts, -0 does not).
    #[target_feature(enable = "avx2")]
    pub unsafe fn nnz(row: &[f32]) -> usize {
        unsafe {
            let (p, n) = (row.as_ptr(), row.len());
            let z = _mm256_setzero_ps();
            let (mut c, mut j) = (0usize, 0usize);
            while j + 8 <= n {
                c += _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_NEQ_UQ>(_mm256_loadu_ps(p.add(j)), z)).count_ones() as usize;
                j += 8;
            }
            c + row[j..].iter().filter(|&&v| v != 0.0).count()
        }
    }

    /// `super::nonzeros`: the set bits of each 8-input compare, lowest first; returns the number written (out holds
    /// at least that many: indexing checks it).
    #[target_feature(enable = "avx2")]
    pub unsafe fn nonzeros(row: &[f32], out: &mut [u32]) -> usize {
        unsafe {
            let (p, n) = (row.as_ptr(), row.len());
            let z = _mm256_setzero_ps();
            let (mut j, mut k) = (0usize, 0usize);
            while j + 8 <= n {
                let mut bits = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_NEQ_UQ>(_mm256_loadu_ps(p.add(j)), z)) as u32;
                while bits != 0 {
                    out[k] = j as u32 + bits.trailing_zeros();
                    k += 1;
                    bits &= bits - 1;
                }
                j += 8;
            }
            for (i, &v) in row[j..].iter().enumerate() {
                if v != 0.0 {
                    out[k] = (j + i) as u32;
                    k += 1;
                }
            }
            k
        }
    }

    /// 8 int8 weights at p, as 8 f32.
    macro_rules! w8 {
        ($p:expr) => {
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64($p as *const __m128i)))
        };
    }

    #[target_feature(enable = "avx2,fma")]
    #[allow(unused_unsafe)]
    pub unsafe fn tile(x: &[f32], idx: Option<&[u32]>, qt: &[i8], sc: &[f32], out: &mut [f32]) {
        unsafe {
            let (mut a0, mut a1, mut a2, mut a3) = (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
            let (mut a4, mut a5, mut a6, mut a7) = (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
            let q = qt.as_ptr();
            let xp = x.as_ptr();
            macro_rules! row {
                ($j:expr) => {{
                    let j = $j;
                    let xb = _mm256_set1_ps(*xp.add(j));
                    let p = q.add(j * TILE);
                    a0 = _mm256_fmadd_ps(xb, w8!(p), a0);
                    a1 = _mm256_fmadd_ps(xb, w8!(p.add(8)), a1);
                    a2 = _mm256_fmadd_ps(xb, w8!(p.add(16)), a2);
                    a3 = _mm256_fmadd_ps(xb, w8!(p.add(24)), a3);
                    a4 = _mm256_fmadd_ps(xb, w8!(p.add(32)), a4);
                    a5 = _mm256_fmadd_ps(xb, w8!(p.add(40)), a5);
                    a6 = _mm256_fmadd_ps(xb, w8!(p.add(48)), a6);
                    a7 = _mm256_fmadd_ps(xb, w8!(p.add(56)), a7);
                }};
            }
            match idx {
                Some(js) => {
                    for &j in js {
                        row!(j as usize);
                    }
                }
                None => {
                    for j in 0..x.len() {
                        row!(j);
                    }
                }
            }
            let mut buf = [0f32; TILE];
            let s = sc.as_ptr();
            let b = buf.as_mut_ptr();
            _mm256_storeu_ps(b, _mm256_mul_ps(a0, _mm256_loadu_ps(s)));
            _mm256_storeu_ps(b.add(8), _mm256_mul_ps(a1, _mm256_loadu_ps(s.add(8))));
            _mm256_storeu_ps(b.add(16), _mm256_mul_ps(a2, _mm256_loadu_ps(s.add(16))));
            _mm256_storeu_ps(b.add(24), _mm256_mul_ps(a3, _mm256_loadu_ps(s.add(24))));
            _mm256_storeu_ps(b.add(32), _mm256_mul_ps(a4, _mm256_loadu_ps(s.add(32))));
            _mm256_storeu_ps(b.add(40), _mm256_mul_ps(a5, _mm256_loadu_ps(s.add(40))));
            _mm256_storeu_ps(b.add(48), _mm256_mul_ps(a6, _mm256_loadu_ps(s.add(48))));
            _mm256_storeu_ps(b.add(56), _mm256_mul_ps(a7, _mm256_loadu_ps(s.add(56))));
            out.copy_from_slice(&buf[..out.len()]);
        }
    }
}

/// E2 in f32 (ava_parts::event, for the timings of `int8_bench`): y [d_out] = x · W for one row x [din] and W [din, d_out]
/// row-major, summing W's rows over the non-zero inputs only (exact up to the order of the float sums).
pub fn event_f32(x: &[f32], w: &[f32], dout: usize, y: &mut [f32]) {
    y.iter_mut().for_each(|v| *v = 0.0);
    for (j, &xj) in x.iter().enumerate() {
        if xj == 0.0 {
            continue;
        }
        let row = &w[j * dout..(j + 1) * dout];
        if xj == 1.0 {
            y.iter_mut().zip(row).for_each(|(o, &r)| *o += r);
        } else {
            y.iter_mut().zip(row).for_each(|(o, &r)| *o += xj * r);
        }
    }
}

// ------------------------------------------------------------------------------------------- the model's int8 set

/// The share of non-zero inputs at or below which a row runs as an event sum (E2). AVA_INT8_EVENT=0 turns E2 off
/// (always dense), =1 always on; a number in (0, 1) sets the share. Default `EVENT_MAX`.
pub const EVENT_MAX: f64 = 0.5;

/// The bits of the share in use (u64::MAX: not read from the environment yet).
static EVENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

pub fn event_max() -> f64 {
    use std::sync::atomic::Ordering::Relaxed;
    let v = EVENT.load(Relaxed);
    if v != u64::MAX {
        return f64::from_bits(v);
    }
    let e = match std::env::var("AVA_INT8_EVENT").ok().and_then(|v| v.parse::<f64>().ok()) {
        Some(v) if v <= 0.0 => -1.0,
        Some(v) => v.min(1.0),
        None => EVENT_MAX,
    };
    EVENT.store(e.to_bits(), Relaxed);
    e
}

/// Set the event share for this process (benchmarks; -1: E2 off). It changes the speed only, never a bit.
pub fn set_event_max(v: f64) {
    EVENT.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// Row blocking of batched products (src/int8/block.rs): 0 off, 1 on, 2 not read from AVA_INT8_BLOCK yet (default on;
/// AVA_INT8_BLOCK=0 turns it off). It changes the speed only, never a bit.
static BLOCK: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

pub fn block_on() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match BLOCK.load(Relaxed) {
        2 => {
            let on = std::env::var("AVA_INT8_BLOCK").map_or(true, |v| v.trim() != "0");
            BLOCK.store(u8::from(on), Relaxed);
            on
        }
        v => v == 1,
    }
}

/// Set the row blocking for this process (benchmarks).
pub fn set_block(on: bool) {
    BLOCK.store(u8::from(on), std::sync::atomic::Ordering::Relaxed);
}

/// A model's int8 copies, by the id of the f32 weight tensor they stand for.
pub struct Q8Set {
    w: HashMap<TensorId, QW>,
    /// The output head: the tied embedding's id and E^T in int8.
    head: Option<(TensorId, QW)>,
    pub threads: usize,
    /// Copies of matrices that read the same input, side by side (`merged`), built on first use.
    merged: std::sync::Mutex<HashMap<Vec<TensorId>, Arc<QW>>>,
}

impl Q8Set {
    pub fn new(threads: usize) -> Q8Set {
        Q8Set { w: HashMap::new(), head: None, threads: threads.max(1), merged: Default::default() }
    }

    /// The int8 copies of `ws` side by side (`QW::concat`): one product for several matrices that read the same
    /// input, each output the bits of its own matrix's product. Built on first use and kept (not in `size`); None
    /// when one is missing or they do not concatenate.
    pub(crate) fn merged(&self, ws: &[&Tensor]) -> Option<Arc<QW>> {
        let key: Vec<TensorId> = ws.iter().map(|w| w.id()).collect();
        let mut map = self.merged.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(q) = map.get(&key) {
            return Some(q.clone());
        }
        let parts: Option<Vec<&QW>> = ws.iter().map(|w| self.get(w)).collect();
        let q = Arc::new(QW::concat(&parts?)?);
        map.insert(key, q.clone());
        Some(q)
    }

    /// Add the int8 copy of W [din, dout] (a no-op when it is already there).
    pub fn add(&mut self, w: &Tensor) -> Result<()> {
        if !self.w.contains_key(&w.id()) {
            self.w.insert(w.id(), QW::from_tensor(w)?);
        }
        Ok(())
    }

    /// Set the output head from the tied embedding E [V, d].
    pub fn set_head(&mut self, e: &Tensor) -> Result<()> {
        self.head = Some((e.id(), QW::from_tensor_t(e)?));
        Ok(())
    }

    /// The int8 copy of the weight tensor w, if the set holds one.
    pub(crate) fn get(&self, w: &Tensor) -> Option<&QW> {
        self.w.get(&w.id())
    }

    /// The int8 output head when it was made from the tied embedding e.
    pub(crate) fn head_of(&self, e: &Tensor) -> Option<&QW> {
        self.head.as_ref().filter(|(id, _)| *id == e.id()).map(|(_, q)| q)
    }

    /// Matrices held (the head counted) and their bytes.
    pub fn size(&self) -> (usize, usize) {
        let n = self.w.len() + usize::from(self.head.is_some());
        let b = self.w.values().chain(self.head.as_ref().map(|h| &h.1)).map(|q| q.bytes()).sum();
        (n, b)
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Arc<Q8Set>>> = const { RefCell::new(None) };
}

/// Holds a model's int8 set on this thread while its forward or step runs (None: plain f32); the one before comes back
/// on drop, so nested passes (--modulate's shadow pass) see the same set.
pub struct Scope {
    prev: Option<Arc<Q8Set>>,
}

pub fn scope(set: Option<Arc<Q8Set>>) -> Scope {
    Scope { prev: ACTIVE.with(|a| a.replace(set)) }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let prev = self.prev.take();
        ACTIVE.with(|a| *a.borrow_mut() = prev);
    }
}

fn active() -> Option<Arc<Q8Set>> {
    ACTIVE.with(|a| a.borrow().clone())
}

/// candle op: x [.., din] (f32, CPU) · the int8 W → [.., dout].
struct Apply<'a> {
    w: &'a QW,
    threads: usize,
}

impl CustomOp1 for Apply<'_> {
    fn name(&self) -> &'static str {
        "ava-int8-matmul"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let x = match s {
            CpuStorage::F32(v) => v.as_slice(),
            _ => candle_core::bail!("int8 matmul: f32 input only"),
        };
        let (a, b) = l.contiguous_offsets().ok_or_else(|| candle_core::Error::Msg("int8 matmul: contiguous input only".into()))?;
        let dims = l.dims();
        let din = *dims.last().ok_or_else(|| candle_core::Error::Msg("int8 matmul: a scalar input".into()))?;
        if din != self.w.din {
            candle_core::bail!("int8 matmul: input width {din}, the weights take {}", self.w.din);
        }
        let m = (b - a) / din;
        let mut y = vec![0f32; m * self.w.dout];
        self.w.matmul_rows(&x[a..b], m, &mut y, event_max(), self.threads);
        let mut shape = dims.to_vec();
        *shape.last_mut().expect("rank >= 1") = self.w.dout;
        Ok((CpuStorage::F32(y), Shape::from(shape)))
    }
}

fn apply(x: &Tensor, w: &QW, threads: usize) -> Result<Tensor> {
    let x = if x.is_contiguous() { x.clone() } else { x.contiguous()? };
    x.apply_op1_no_bwd(&Apply { w, threads })
}

/// Off-by-default timer over `mm` and `head` (both paths, int8 and f32): calls and nanoseconds, for profiles
/// (src/bin/fstep_bench.rs). Off it costs one relaxed load per call.
static MM_PROF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static MM_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MM_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Switch the `mm` / `head` timer on or off (and zero it).
pub fn set_mm_prof(on: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    MM_CALLS.store(0, Relaxed);
    MM_NS.store(0, Relaxed);
    MM_PROF.store(on, Relaxed);
}

/// (calls, nanoseconds) of `mm` and `head` since the timer was switched on.
pub fn mm_prof() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (MM_CALLS.load(Relaxed), MM_NS.load(Relaxed))
}

/// The `mm` / `head` timer for other one-row products (the fused step's): whether it is on, and add one call.
pub(crate) fn mm_prof_on() -> bool {
    MM_PROF.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn mm_prof_add(ns: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    MM_NS.fetch_add(ns, Relaxed);
    MM_CALLS.fetch_add(1, Relaxed);
}

fn timed(f: impl FnOnce() -> Result<Tensor>) -> Result<Tensor> {
    use std::sync::atomic::Ordering::Relaxed;
    if !MM_PROF.load(Relaxed) {
        return f();
    }
    let t0 = std::time::Instant::now();
    let r = f();
    MM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    MM_CALLS.fetch_add(1, Relaxed);
    r
}

/// x · w: through w's int8 copy when the active set has one, else exactly `x.matmul(w)`.
pub fn mm(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    timed(|| {
        if let Some(set) = active() {
            if let Some(q) = set.w.get(&w.id()) {
                if x.dtype() == candle_core::DType::F32 && x.device().is_cpu() && x.rank() == 2 {
                    return apply(x, q, set.threads);
                }
            }
        }
        x.matmul(w)
    })
}

/// The output head x · e^T for the tied embedding e [V, d]: int8 when the active set holds it, else exactly
/// `x.matmul(&e.t()?)`.
pub fn head(x: &Tensor, e: &Tensor) -> Result<Tensor> {
    timed(|| {
        if let Some(set) = active() {
            if let Some((id, q)) = &set.head {
                if *id == e.id() && x.dtype() == candle_core::DType::F32 && x.device().is_cpu() && x.rank() == 2 {
                    return apply(x, q, set.threads);
                }
            }
        }
        x.matmul(&e.t()?)
    })
}

/// Whether an int8 set is active on this thread (tests).
pub fn is_active() -> bool {
    active().is_some()
}

mod block;
pub use block::{set_block_event_max, BLOCK_EVENT_MAX};

#[cfg(test)]
mod tests;
