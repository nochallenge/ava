// TESTING / DESIGN ONLY — ava_gpu::model::fstep: Ava's one-token CPU step as plain Rust loops (`Model::set_fused`).
//
// Talking one token at a time, `Model::step` runs about a hundred candle ops per block (each norm is six, the LIF
// update seven, the fast-weight memory's gates and delta rule fifty), and each op costs a few microseconds of dispatch
// and allocation whatever its size. At d128 those costs are nearly the whole step, and they stay half of it at d384
// even with int8 weights. This module runs the same step for one token in plain loops: one pass over each vector, the
// int8 matmuls straight into the kernel of src/int8.rs, no tensor per intermediate.
//
// The same bits as `Model::step` with int8 on. Every float operation is the one candle performs, in the same order:
// * elementwise ops as candle's CPU kernels write them (affine is v·mul + add with both cast to f32, so `x + eps`
//   is x·1 + eps; sigmoid is 1 / (exp(-x)·1 + 1); silu is v / (1 + exp(-v)); relu is v.max(0));
// * sums (the norms' means, softmax's denominator) one value after another from 0, as candle's `vec_sum` does when
//   the build has no avx2 / neon / simd128 target feature (with one, candle sums in SIMD lanes; the fused step then
//   switches itself off);
// * the int8 products through `QW::matmul_rows` itself (q, k, v and the output gate's input, which read the same
//   vector, as one product of their side-by-side copy, `Q8Set::merged`: each column's sum is its own matrix's);
// * the small f32 products that stay candle matmuls in `Model::step` (the gates' projections, the fast weights'
//   recall and read, the engram's scores and mix, the magnitude code) in the order gemm 0.19 computes a one-row
//   product on an AVX2 + FMA CPU: out = x0·w0, then out = fma(w_j, x_j, out) for j = 1, 2, ... (its column-major
//   gemv); with 2 inputs (x1·w1) + (x0·w0) and with 1 the product (its gevv); a K = 1 outer product is the plain
//   product of the two factors. With one head (d = 32) gemm takes another kernel for the gates' projection [1, d] x
//   [d, 1], so the fused step needs 2 heads or more; on a CPU without FMA, or with AVX-512 (kernels not checked
//   here), it is off too (`fused_blocker` says why);
// * the kWTA threshold is the k-th largest value, found by selection instead of a full sort: the same value up to
//   the sign of a zero, which gives the same mask.
// The carried state stays in the Stream's own tensors in their own shapes, so the fused and the plain step can take
// turns on one stream (and `state_tensors`, `step_trace` see the same state). Tests: src/model/fstep/tests.rs
// (max |dlogit| = 0 and equal states over 200 steps, every kind); bench: src/bin/fstep_bench.rs.

use super::{Model, Stream, HEAD_SIZE, LIF_THRESHOLD, NORM_EPS, PLACES};
use crate::int8::QW;
use candle_core::{CpuStorage, Result, Storage, Tensor};

impl Model {
    /// Talk through the fused one-token CPU step (off by default). It applies on the CPU with int8 on, to today's
    /// block and (src/model/fstep/arch.rs) to Ava 0.4's block options on a State-backed stream, and gives `step` its
    /// bits exactly; where it does not apply (`fused_blocker`), `step` runs as before.
    pub fn set_fused(&self, on: bool) {
        self.fused.set(on);
    }

    /// Whether the fused step is switched on (it may still not apply: `fused_blocker`).
    pub fn fused(&self) -> bool {
        self.fused.get()
    }

    /// v51 fast4 (src/fast4.rs): the fused step for talking with an online session. With it on, the block-options
    /// step also runs with int8 off (f32 products in gemm's one-row order, src/fast4/fmm.rs: `forward_with_state`'s f32
    /// bits), takes the cerebellum's online table (one row) and is the base of the e-prop session step
    /// (`Model::fast4_eprop_step`). Off by default: everything runs as before.
    pub fn set_fast4(&self, on: bool) {
        self.fast4.set(on);
    }

    /// Whether `set_fast4` is on.
    pub fn fast4(&self) -> bool {
        self.fast4.get()
    }

    /// Why the fused step does not apply to this model as it is set now; None: it applies.
    pub fn fused_blocker(&self) -> Option<&'static str> {
        if cfg!(any(target_feature = "avx2", target_feature = "neon", target_feature = "simd128")) {
            return Some("built with SIMD target features: candle then sums in SIMD lanes, another order");
        }
        if !self.tok.device().is_cpu() {
            return Some("the model is not on the CPU");
        }
        if !self.cfg.kind.is_ava() {
            return Some("T has no one-token step");
        }
        if self.q8.borrow().is_none() {
            // v51 fast4: with `set_fast4` the block-options step also runs with int8 off (f32 products)
            if !self.fast4.get() {
                return Some("int8 is off: the fused step is the int8 talking path");
            }
            if !self.arch_path() {
                return Some("int8 is off: the f32 fused step covers block-options models only");
            }
            if let Some(why) = crate::fast4::f32_blocker(self) {
                return Some(why);
            }
        }
        if !crate::int8::simd_ok() {
            return Some("no AVX2 + FMA: gemm's arithmetic differs");
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            return Some("AVX-512: gemm's kernels there are not checked against the fused step");
        }
        if self.arch_path() {
            // block options: the State-backed step (src/model/fstep/arch.rs)
            return arch::blocker(self);
        }
        if self.cfg.heads() < 2 {
            return Some("one head: gemm takes another kernel for the gates' projection");
        }
        if self.blocks.iter().any(|b| b.mem.is_none()) {
            return Some("a block without its spiking memory");
        }
        None
    }
}

/// A CPU f32 tensor's values (row-major, contiguous), read in place.
struct Ro<'a> {
    g: std::sync::RwLockReadGuard<'a, Storage>,
    a: usize,
    b: usize,
}

impl Ro<'_> {
    fn s(&self) -> &[f32] {
        match &*self.g {
            Storage::Cpu(CpuStorage::F32(v)) => &v[self.a..self.b],
            _ => unreachable!("checked in ro"),
        }
    }
}

fn ro(t: &Tensor) -> Result<Ro<'_>> {
    let (g, l) = t.storage_and_layout();
    let (a, b) = l.contiguous_offsets().ok_or_else(|| candle_core::Error::Msg("fused step: a non-contiguous tensor".into()))?;
    if !matches!(&*g, Storage::Cpu(CpuStorage::F32(_))) {
        candle_core::bail!("fused step: CPU f32 tensors only");
    }
    Ok(Ro { g, a, b })
}

/// The values of a tensor (copied; for the carried state, which may come from either step).
fn vals(t: &Tensor) -> Result<Vec<f32>> {
    let t = t.contiguous()?;
    let r = ro(&t)?;
    Ok(r.s().to_vec())
}

/// The tensor itself when contiguous (shared, nothing copied), else a contiguous copy: for reading in place.
fn contig(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

/// The fused step's section clock, on and zeroed with `crate::int8::set_mm_prof` (whose timer counts the int8
/// products): nanoseconds in the kWTA threshold, the fast weights and attention + engram, their int8 products
/// excluded. Off it costs one relaxed load per section.
static SECS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// The section names of `take_secs`.
pub const SECTIONS: [&str; 5] =
    ["kWTA threshold", "fast weights (no int8)", "attention + engram (no int8)", "  of the fast weights: gates", "  of the fast weights: heads"];

/// Nanoseconds per section since the last call (zeroes them).
pub fn take_secs() -> [u64; 5] {
    std::array::from_fn(|c| SECS[c].swap(0, std::sync::atomic::Ordering::Relaxed))
}

fn clock<T>(c: usize, f: impl FnOnce() -> T) -> T {
    if !crate::int8::mm_prof_on() {
        return f();
    }
    let t0 = std::time::Instant::now();
    let r = f();
    SECS[c].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    r
}

/// The int8 product of one row: exactly what `crate::int8::mm` runs for x [1, din].
fn mm(q: &QW, x: &[f32], threads: usize) -> Vec<f32> {
    let mut y = vec![0f32; q.dout];
    let t0 = crate::int8::mm_prof_on().then(std::time::Instant::now);
    q.matmul_rows(x, 1, &mut y, crate::int8::event_max(), threads);
    if let Some(t0) = t0 {
        crate::int8::mm_prof_add(t0.elapsed().as_nanos() as u64);
    }
    y
}

/// x · W for each of `ws` (all reading the same x): one product of their side-by-side int8 copy when they
/// concatenate (`Q8Set::merged`; one dispatch instead of several), else one each; the same bits either way.
fn mm_many(set: &crate::int8::Q8Set, ws: &[&Tensor], x: &[f32], threads: usize) -> Vec<Vec<f32>> {
    let parts: Vec<&QW> = ws.iter().map(|w| set.get(w).expect("checked in step")).collect();
    match set.merged(ws) {
        Some(q) => {
            let all = mm(&q, x, threads);
            let mut c = 0;
            parts
                .iter()
                .map(|p| {
                    c += p.dout;
                    all[c - p.dout..c].to_vec()
                })
                .collect()
        }
        None => parts.iter().map(|p| mm(p, x, threads)).collect(),
    }
}

fn add_into(x: &mut [f32], o: &[f32]) {
    for (a, &b) in x.iter_mut().zip(o) {
        *a = *a + b;
    }
}

/// `model::rmsnorm`: x · (mean(x²) + eps)^-1/2 · gain, as candle computes it.
fn rmsnorm(x: &[f32], gain: &[f32]) -> Vec<f32> {
    let inv = inv_rms(x);
    x.iter().zip(gain).map(|(&v, &g)| (v * inv) * g).collect()
}

/// (mean(x²) + eps)^-1/2: sqr, sum from 0, the mean's affine (sum · (1/n) + 0), affine(1, eps), powf(-0.5).
fn inv_rms(x: &[f32]) -> f32 {
    let mut s = 0f32;
    for &v in x {
        s += v * v;
    }
    let ms = s * ((1f64 / x.len() as f64) as f32) + 0.0;
    (ms * 1.0 + NORM_EPS as f32).powf(-0.5f64 as f32)
}

/// One-row f32 product out[c] = sum_j x[j] w[j][c] (w [k, n] row-major, n >= 2) in gemm's order (see the top).
fn gemv(x: &[f32], w: &[f32], n: usize, out: &mut [f32]) {
    let k = x.len();
    debug_assert!(w.len() >= k * n && out.len() >= n);
    match k {
        0 => out[..n].fill(0.0),
        1 => {
            for c in 0..n {
                out[c] = w[c] * x[0];
            }
        }
        2 => {
            for c in 0..n {
                out[c] = w[n + c] * x[1] + w[c] * x[0];
            }
        }
        _ => gemv_fma(x, w, n, out),
    }
}

#[cfg(target_arch = "x86_64")]
fn gemv_fma(x: &[f32], w: &[f32], n: usize, out: &mut [f32]) {
    // SAFETY: the fused step runs only where AVX2 + FMA were detected (fused_blocker)
    unsafe { avx::gemv(x, w, n, out) }
}

#[cfg(not(target_arch = "x86_64"))]
fn gemv_fma(x: &[f32], w: &[f32], n: usize, out: &mut [f32]) {
    for c in 0..n {
        let mut acc = w[c] * x[0];
        for j in 1..x.len() {
            acc = w[j * n + c].mul_add(x[j], acc);
        }
        out[c] = acc;
    }
}

/// candle's sigmoid as `model::sigmoid` builds it: (exp(-x) · 1 + 1)^-1.
pub(crate) fn sigmoid(x: f32) -> f32 {
    ((-x).exp() * 1.0 + 1.0).recip()
}

/// `model::log_sigmoid`: -(relu(-x) + ln(exp(-|x|) · 1 + 1)).
fn log_sigmoid(x: f32) -> f32 {
    let soft = ((-x.abs()).exp() * 1.0 + 1.0).ln();
    -((-x).max(0f32) + soft)
}

/// Each head's slice (hs values) times (sum(x²) · 1 + 1e-6)^-1/2: `model::l2_norm`.
fn l2_heads(x: &mut [f32], hs: usize) {
    for h in x.chunks_mut(hs) {
        let mut s = 0f32;
        for &v in h.iter() {
            s += v * v;
        }
        let inv = (s * 1.0 + 1e-6f64 as f32).powf(-0.5f64 as f32);
        for v in h.iter_mut() {
            *v = *v * inv;
        }
    }
}

/// `model::head_norm` per head.
fn head_norm(y: &mut [f32], hs: usize) {
    for h in y.chunks_mut(hs) {
        let inv = inv_rms(h);
        for v in h.iter_mut() {
            *v = *v * inv;
        }
    }
}

/// The k-th largest value of u (the kWTA threshold of lif.rs's CPU mask, which sorts; see the top).
pub(crate) fn kth_largest(u: &[f32], k: usize) -> f32 {
    let mut b = u.to_vec();
    let (_, kth, _) = b.select_nth_unstable_by(k - 1, |a, c| c.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    *kth
}

/// One head's causal read over its keys and values so far (`kc`, `vc`: n rows of hs; the scores scaled, softmax,
/// mix), as Model::step's attention and engram chains compute it with gemm.
fn attend(q: &[f32], kc: &[f32], vc: &[f32], n: usize, hs: usize, out: &mut [f32]) {
    let scale = (1.0 / (HEAD_SIZE as f64).sqrt()) as f32;
    // scores q · K^T: per key the one-row product over hs (gemm transposes [1, n] to n outputs of a column-major
    // gemv: out_j = K[j][0] q0, then fma over the dims), scaled; for n = 1 gemm takes a dot-product kernel, but
    // softmax of one score is 1 for any finite score
    let mut sc = vec![0f32; n];
    scores(q, kc, hs, scale, &mut sc);
    // softmax: max (candle's fold: replace while val < s), x - max, exp, sum from 0, divide
    let mut mx = sc[0];
    for &s in &sc {
        if mx < s {
            mx = s;
        }
    }
    let mut den = 0f32;
    for s in sc.iter_mut() {
        *s = (*s - mx).exp();
        den += *s;
    }
    for s in sc.iter_mut() {
        *s = *s / den;
    }
    // p · V: the one-row product [1, n] x [n, hs]
    gemv(&sc, vc, hs, out);
}

/// out[j] = (K[j][0]·q0, then fma(K[j][a], q[a], ·) for a = 1, 2, ...) · scale + 0 for the n = out.len() keys: one
/// chain per key in gemm's order, 8 keys' chains side by side (the same bits as one key at a time).
#[inline(always)]
fn scores_rows(q: &[f32], kc: &[f32], hs: usize, scale: f32, out: &mut [f32]) {
    let n = out.len();
    let mut j = 0;
    while j + 8 <= n {
        let kb = &kc[j * hs..(j + 8) * hs];
        let mut acc: [f32; 8] = std::array::from_fn(|i| kb[i * hs] * q[0]);
        for a in 1..hs {
            for (i, s) in acc.iter_mut().enumerate() {
                *s = kb[i * hs + a].mul_add(q[a], *s);
            }
        }
        for (o, &s) in out[j..j + 8].iter_mut().zip(&acc) {
            *o = s * scale + 0.0;
        }
        j += 8;
    }
    for (jj, o) in out.iter_mut().enumerate().skip(j) {
        let kr = &kc[jj * hs..(jj + 1) * hs];
        let mut s = kr[0] * q[0];
        for a in 1..hs {
            s = kr[a].mul_add(q[a], s);
        }
        *o = s * scale + 0.0;
    }
}

#[cfg(target_arch = "x86_64")]
fn scores(q: &[f32], kc: &[f32], hs: usize, scale: f32, out: &mut [f32]) {
    // SAFETY: the fused step runs only where AVX2 + FMA were detected (fused_blocker)
    unsafe { avx::scores(q, kc, hs, scale, out) }
}

#[cfg(not(target_arch = "x86_64"))]
fn scores(q: &[f32], kc: &[f32], hs: usize, scale: f32, out: &mut [f32]) {
    scores_rows(q, kc, hs, scale, out)
}

/// How one head's fast weights take this token (see `step`).
#[derive(Clone, Copy)]
enum Rule {
    /// D: ((F - (k p^T)·beta)·alpha) + (k v^T)·beta, p = k^T F
    Delta(f32),
    /// G, R: (F·alpha) + (k v^T)·beta
    Gated(f32),
    /// F: (F·gamma) + k v^T
    Gamma,
}

/// One head's fast-weight update: its rule, the old weights `f` [hs, hs], k, p = k^T f (Delta only), v, alpha, and
/// the read's q.
struct Fw<'a> {
    rule: Rule,
    f: &'a [f32],
    k: &'a [f32],
    p: &'a [f32],
    v: &'a [f32],
    al: f32,
    q: &'a [f32],
    hs: usize,
}

/// One head's new fast weights `out` [hs, hs] from the old, row by row (elementwise: the same ops in the same order
/// whatever the vector width), and its read y = q · F_new [hs] as the rows are made, in gemv's order for the one-row
/// product (hs >= 3): y = F[0]·q0, then y = fma(F[a], q[a], y) for a = 1, 2, ...
#[inline(always)]
fn fw_rows(w: &Fw, out: &mut [std::mem::MaybeUninit<f32>], y: &mut [f32]) {
    let hs = w.hs;
    assert!(out.len() == hs * hs && w.f.len() == hs * hs && w.k.len() >= hs && w.q.len() >= hs && w.v.len() >= hs && y.len() >= hs);
    let (v, y) = (&w.v[..hs], &mut y[..hs]);
    let mut row = vec![0f32; hs];
    for (a, ((orow, frow), &ka)) in out.chunks_exact_mut(hs).zip(w.f.chunks_exact(hs)).zip(w.k).enumerate() {
        let al = w.al;
        match w.rule {
            Rule::Delta(b) => {
                for (((r, &fv), &pv), &vv) in row.iter_mut().zip(frow).zip(&w.p[..hs]).zip(v) {
                    let corr = (ka * pv) * b;
                    *r = ((fv - corr) * al) + ((ka * vv) * b);
                }
            }
            Rule::Gated(b) => {
                for ((r, &fv), &vv) in row.iter_mut().zip(frow).zip(v) {
                    *r = (fv * al) + ((ka * vv) * b);
                }
            }
            Rule::Gamma => {
                for ((r, &fv), &vv) in row.iter_mut().zip(frow).zip(v) {
                    *r = (fv * al) + (ka * vv);
                }
            }
        }
        for (o, &r) in orow.iter_mut().zip(&row) {
            o.write(r);
        }
        let qa = w.q[a];
        if a == 0 {
            for (yc, &r) in y.iter_mut().zip(&row) {
                *yc = r * qa;
            }
        } else {
            for (yc, &r) in y.iter_mut().zip(&row) {
                *yc = r.mul_add(qa, *yc);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn fw_head(w: &Fw, out: &mut [std::mem::MaybeUninit<f32>], y: &mut [f32]) {
    // SAFETY: the fused step runs only where AVX2 + FMA were detected (fused_blocker)
    unsafe { avx::fw_head(w, out, y) }
}

#[cfg(not(target_arch = "x86_64"))]
fn fw_head(w: &Fw, out: &mut [std::mem::MaybeUninit<f32>], y: &mut [f32]) {
    fw_rows(w, out, y)
}

/// The keys or values so far [nh, n, hs] with this token's row [nh, hs] appended: [nh, n + 1, hs].
fn append(old: Option<&[f32]>, n: usize, row: &[f32], nh: usize, hs: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(nh * (n + 1) * hs);
    for h in 0..nh {
        if let Some(o) = old {
            out.extend_from_slice(&o[h * n * hs..(h + 1) * n * hs]);
        }
        out.extend_from_slice(&row[h * hs..(h + 1) * hs]);
    }
    out
}

/// `Model::step` for one token, fused; None (nothing changed) where it does not apply.
pub(super) fn step(m: &Model, st: &mut Stream, tok: u32) -> Result<Option<Tensor>> {
    if m.fused_blocker().is_some() || st.st.is_some() {
        return Ok(None);
    }
    let Some(set) = m.q8.borrow().clone() else { return Ok(None) };
    let cfg = m.cfg;
    let (d, nh, hs) = (cfg.d, cfg.heads(), HEAD_SIZE);
    let th = set.threads;
    let dev = m.tok.device().clone();
    let nb = m.blocks.len();
    // everything checked before the stream changes, so a None leaves it as it was
    let Some(head) = set.head_of(&m.tok) else { return Ok(None) };
    if st.v.len() != nb || st.s.len() != nb || st.beta.len() != nb || st.kv.len() != nb || (tok as usize) >= cfg.vocab {
        return Ok(None);
    }
    for (i, blk) in m.blocks.iter().enumerate() {
        let mem = blk.mem.as_ref().expect("checked in fused_blocker");
        let mut ws = vec![&mem.win, &mem.wout, &blk.mlp.w1, &blk.mlp.w2];
        if let Some(hb) = &blk.hebb {
            ws.extend([&hb.wq, &hb.wk, &hb.wv, &hb.wo]);
            ws.extend(hb.wg.iter());
            let delta_ok = hb.gates.is_some() || st.gamma.is_some();
            if !delta_ok || st.fw.len() <= i {
                return Ok(None);
            }
        }
        for a in blk.attn.iter().chain(blk.eng.iter()) {
            ws.extend([&a.wq, &a.wk, &a.wv, &a.wo]);
        }
        if ws.iter().any(|w| set.get(w).is_none()) {
            return Ok(None);
        }
    }
    let q8 = |w: &Tensor| set.get(w).expect("checked above");
    let (place_off, mag_off, eng_off) = m.lesion.get();

    // the token's embedding, its place (v39 glance) and its magnitude code (v40)
    let mut x: Vec<f32> = {
        let e = ro(&m.tok)?;
        e.s()[tok as usize * d..(tok as usize + 1) * d].to_vec()
    };
    let p = st.next_place(tok) as usize;
    if let (Some(place), false) = (&m.place, place_off) {
        let pt = ro(place)?;
        if p < PLACES {
            add_into(&mut x, &pt.s()[p * d..(p + 1) * d]);
        } else {
            // the table's zero row ("none")
            x.iter_mut().for_each(|v| *v = *v + 0.0);
        }
    }
    st.hist.push(tok);
    if st.hist.len() > 17 {
        st.hist.remove(0);
    }
    if let (Some(mw), false) = (&m.mag_w, mag_off) {
        if let Some(mv) = crate::mag::position(&st.hist, st.hist.len() - 1) {
            let f = crate::mag::code(mv);
            let w = ro(mw)?;
            let mut o = vec![0f32; d];
            gemv(&f, w.s(), d, &mut o);
            add_into(&mut x, &o);
        }
    }

    for (i, blk) in m.blocks.iter().enumerate() {
        // the spiking memory: ((beta·v + I) - theta·s), spike where v - theta >= 0
        let mem = blk.mem.as_ref().expect("checked in fused_blocker");
        let h = rmsnorm(&x, ro(&mem.norm)?.s());
        let cur = mm(q8(&mem.win), &h, th);
        let (v0, s0, beta) = (vals(&st.v[i])?, vals(&st.s[i])?, vals(&st.beta[i])?);
        let thr = LIF_THRESHOLD as f32;
        let mut v = vec![0f32; d];
        let mut s = vec![0f32; d];
        for j in 0..d {
            let vv = ((v0[j] * beta[j]) + cur[j]) - (s0[j] * thr + 0.0);
            v[j] = vv;
            s[j] = if vv * 1.0 + (-LIF_THRESHOLD as f32) >= 0.0 { 1.0 } else { 0.0 };
        }
        add_into(&mut x, &mm(q8(&mem.wout), &s, th));
        st.v[i] = Tensor::from_vec(v, (1, d), &dev)?;
        st.s[i] = Tensor::from_vec(s, (1, d), &dev)?;

        if let Some(hb) = &blk.hebb {
            // the fast weights: F <- alpha (F - beta k (k^T F)) + beta k v^T (D), alpha F + beta k v^T (G, R),
            // gamma F + k v^T (F); read q·F
            let h = rmsnorm(&x, ro(&hb.norm)?.s());
            // q, k, v (and the output gate's input) read the same h: one product of their side-by-side copy
            let mut ws = vec![&hb.wq, &hb.wk, &hb.wv];
            ws.extend(hb.wg.iter());
            let mut outs = mm_many(&set, &ws, &h, th).into_iter();
            let (mut q, mut k, vv) = (outs.next().expect("q"), outs.next().expect("k"), outs.next().expect("v"));
            let gq = outs.next();
            if hb.qk_norm {
                l2_heads(&mut q, hs);
                l2_heads(&mut k, hs);
            }
            let (y, f1) = clock(1, || -> Result<(Vec<f32>, Vec<f32>)> {
                let ft = contig(&st.fw[i])?;
                let fr = ro(&ft)?;
                let f0 = fr.s();
                let len = nh * hs * hs;
                if f0.len() != len || hs < 3 {
                    candle_core::bail!("fused step: fast weights of {} values, want {len} (head size {hs})", f0.len());
                }
                // per head (alpha, beta) of the gates, or (gamma, none)
                let (alpha, beta): (Vec<f32>, Option<Vec<f32>>) = clock(3, || -> Result<_> { Ok(match (&hb.gates, &st.gamma) {
                    (Some(g), _) => {
                        let (wa, ba, off, wb, bb) = (ro(&g.wa)?, ro(&g.ba)?, ro(&g.off)?, ro(&g.wb)?, ro(&g.bb)?);
                        let (mut pa, mut pb) = (vec![0f32; nh], vec![0f32; nh]);
                        gemv(&h, wa.s(), nh, &mut pa);
                        gemv(&h, wb.s(), nh, &mut pb);
                        let la: Vec<f32> = (0..nh).map(|e| log_sigmoid((pa[e] + ba.s()[e]) + off.s()[e])).collect();
                        let be: Vec<f32> = (0..nh).map(|e| sigmoid(pb[e] + bb.s()[e]) * 2.0 + 0.0).collect();
                        (la.iter().map(|v| v.exp()).collect(), Some(be))
                    }
                    (None, Some(gamma)) => (vals(gamma)?, None),
                    (None, None) => unreachable!("checked above"),
                }) })?;
                // the new fast weights written once (no zero fill), the read q·F taken row by row as they are made
                let mut f1: Vec<f32> = Vec::with_capacity(len);
                let mut y = vec![0f32; d];
                {
                    let spare = &mut f1.spare_capacity_mut()[..len];
                    clock(4, || {
                        for ((e, out), ye) in spare.chunks_exact_mut(hs * hs).enumerate().zip(y.chunks_exact_mut(hs)) {
                            let (kh, vh) = (&k[e * hs..(e + 1) * hs], &vv[e * hs..(e + 1) * hs]);
                            let fh = &f0[e * hs * hs..(e + 1) * hs * hs];
                            let mut pred = Vec::new();
                            let rule = match (&beta, hb.delta) {
                                (Some(be), true) => {
                                    pred = vec![0f32; hs];
                                    gemv(kh, fh, hs, &mut pred);
                                    Rule::Delta(be[e])
                                }
                                (Some(be), false) => Rule::Gated(be[e]),
                                (None, _) => Rule::Gamma,
                            };
                            let fw = Fw { rule, f: fh, k: kh, p: &pred, v: vh, al: alpha[e], q: &q[e * hs..(e + 1) * hs], hs };
                            fw_head(&fw, out, ye);
                        }
                    });
                }
                // SAFETY: fw_head wrote every one of the len values (each head's hs rows of hs)
                unsafe { f1.set_len(len) };
                if !hb.delta {
                    let sc = (1.0 / (HEAD_SIZE as f64).sqrt()) as f32;
                    y.iter_mut().for_each(|v| *v = *v * sc + 0.0);
                }
                head_norm(&mut y, hs);
                Ok((y, f1))
            })?;
            let mut y = y;
            if let Some(g) = &gq {
                for (a, &gv) in y.iter_mut().zip(g.iter()) {
                    *a = *a * (gv / (1.0 + (-gv).exp()));
                }
            } else if let Some((wd, wu)) = &hb.wgl {
                // v44 K2: y times sigmoid(h·Wd·Wu), the two small products in f32 as `step` takes them (gemm's one-row order)
                let (wd, wu) = (ro(wd)?, ro(wu)?);
                let r = wd.s().len() / d;
                let mut z1 = vec![0f32; r];
                gemv(&h, wd.s(), r, &mut z1);
                let mut z = vec![0f32; d];
                gemv(&z1, wu.s(), d, &mut z);
                for (a, &zv) in y.iter_mut().zip(z.iter()) {
                    *a = *a * sigmoid(zv);
                }
            }
            add_into(&mut x, &mm(q8(&hb.wo), &y, th));
            st.fw[i] = Tensor::from_vec(f1, (nh, hs, hs), &dev)?;
        }

        // attention over every token so far (A's attention blocks, st.kv) and the engram (v40, st.ekv; it keeps
        // storing while lesioned and then adds nothing)
        for (a, is_eng) in blk.attn.iter().map(|a| (a, false)).chain(blk.eng.iter().map(|e| (e, true))) {
            let h = rmsnorm(&x, ro(&a.norm)?.s());
            let mut qkv = mm_many(&set, &[&a.wq, &a.wk, &a.wv], &h, th).into_iter();
            let (q, k, v) = (qkv.next().expect("q"), qkv.next().expect("k"), qkv.next().expect("v"));
            let slot = if is_eng { st.ekv.take() } else { st.kv[i].take() };
            let read = !(is_eng && eng_off);
            let (n, kc, vc, o) = clock(2, || -> Result<(usize, Vec<f32>, Vec<f32>, Vec<f32>)> {
                // the keys and values so far, read in place, and this token's appended
                let (n0, kt, vt) = match &slot {
                    None => (0, None, None),
                    Some((kp, vp)) => (kp.dim(1)?, Some(contig(kp)?), Some(contig(vp)?)),
                };
                let (kr, vr) = (kt.as_ref().map(ro).transpose()?, vt.as_ref().map(ro).transpose()?);
                let kc = append(kr.as_ref().map(|r| r.s()), n0, &k, nh, hs);
                let vc = append(vr.as_ref().map(|r| r.s()), n0, &v, nh, hs);
                let n = n0 + 1;
                let mut o = Vec::new();
                if read {
                    o = vec![0f32; d];
                    for e in 0..nh {
                        let span = e * n * hs..(e + 1) * n * hs;
                        attend(&q[e * hs..(e + 1) * hs], &kc[span.clone()], &vc[span], n, hs, &mut o[e * hs..(e + 1) * hs]);
                    }
                }
                Ok((n, kc, vc, o))
            })?;
            if read {
                add_into(&mut x, &mm(q8(&a.wo), &o, th));
            }
            let kv = (Tensor::from_vec(kc, (nh, n, hs), &dev)?, Tensor::from_vec(vc, (nh, n, hs), &dev)?);
            if is_eng {
                st.ekv = Some(kv);
            } else {
                st.kv[i] = Some(kv);
            }
        }

        // the sparse MLP: the k largest of h·W1 kept, relu, ·W2
        let l = &blk.mlp;
        let h = rmsnorm(&x, ro(&l.norm)?.s());
        let u = mm(q8(&l.w1), &h, th);
        let kk = cfg.kwta_k();
        let a: Vec<f32> = if kk >= u.len() {
            u.iter().map(|&v| (v * 1.0).max(0f32)).collect()
        } else {
            let kth = clock(0, || kth_largest(&u, kk));
            u.iter().map(|&v| (v * if v - kth >= 0.0 { 1.0 } else { 0.0 }).max(0f32)).collect()
        };
        add_into(&mut x, &mm(q8(&l.w2), &a, th));
    }
    st.seen += 1;
    let x = rmsnorm(&x, ro(&m.final_norm)?.s());
    let logits = mm(head, &x, th);
    #[cfg(test)]
    RAN.with(|r| r.set(r.get() + 1));
    Ok(Some(Tensor::from_vec(logits, cfg.vocab, &dev)?))
}

#[cfg(test)]
thread_local! {
    /// Tests: fused steps completed on this thread.
    pub(crate) static RAN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::*;

    /// `super::gemv` for k >= 3: each output x0·w0 and then fused multiply-adds over j, 64 outputs at a time (eight
    /// chains side by side), then 32, then 8, then the last ones in one masked vector; each output's chain is the
    /// same whatever runs beside it, so the same bits as one output at a time.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemv(x: &[f32], w: &[f32], n: usize, out: &mut [f32]) {
        unsafe {
            let k = x.len();
            assert!(k >= 1 && w.len() >= k * n && out.len() >= n);
            let mut c = 0;
            while c + 64 <= n {
                cols::<8>(x, w.as_ptr(), n, c, out.as_mut_ptr());
                c += 64;
            }
            while c + 32 <= n {
                cols::<4>(x, w.as_ptr(), n, c, out.as_mut_ptr());
                c += 32;
            }
            while c + 8 <= n {
                cols::<1>(x, w.as_ptr(), n, c, out.as_mut_ptr());
                c += 8;
            }
            let t = n - c;
            if t > 0 {
                // the last t < 8 outputs: lanes past t neither read nor written
                let lanes: [i32; 8] = std::array::from_fn(|i| if i < t { -1 } else { 0 });
                let mask = _mm256_loadu_si256(lanes.as_ptr() as *const __m256i);
                let wp = w.as_ptr().add(c);
                let mut acc = _mm256_mul_ps(_mm256_maskload_ps(wp, mask), _mm256_set1_ps(x[0]));
                for (j, &xj) in x.iter().enumerate().skip(1) {
                    acc = _mm256_fmadd_ps(_mm256_maskload_ps(wp.add(j * n), mask), _mm256_set1_ps(xj), acc);
                }
                _mm256_maskstore_ps(out.as_mut_ptr().add(c), mask, acc);
            }
        }
    }

    /// Outputs c .. c + 8V of `gemv` (in range: checked by the caller).
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn cols<const V: usize>(x: &[f32], wp: *const f32, n: usize, c: usize, op: *mut f32) {
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

    /// `super::scores_rows` compiled for AVX2 + FMA (mul_add becomes one instruction).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn scores(q: &[f32], kc: &[f32], hs: usize, scale: f32, out: &mut [f32]) {
        super::scores_rows(q, kc, hs, scale, out)
    }

    /// `super::fw_rows` compiled for AVX2 + FMA (8 lanes; elementwise, so the same bits).
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn fw_head(w: &super::Fw, out: &mut [std::mem::MaybeUninit<f32>], y: &mut [f32]) {
        super::fw_rows(w, out, y)
    }
}

// the per-op-class profile of the plain step (src/bin/fstep_bench.rs)
pub mod prof;

// the fused step for block options (Ava 0.4's recipe): forward_with_state for one token on a one-row State
pub mod arch;

#[cfg(test)]
mod tests;
