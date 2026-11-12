// TESTING / DESIGN ONLY — ava_gpu::model::fstep::arch: the fused one-token CPU step for block options (Ava 0.4's recipe).
//
// A model with block options talks through a one-row State (src/state.rs): every token is one `forward_with_state`
// of one token, about 3,100 candle ops (the LIF with int2 spikes and ALIF, the per-channel gates, the chunked memory
// with momentum, the dense residual mixes), each costing dispatch and an allocation whatever its size. This module runs
// the same token in plain loops on the State itself (`Model::step` calls it when `set_fused` is on), so a stream can
// mix both (the batched `feed` of a prompt, then fused steps).
//
// The same bits as `forward_with_state` for one token with int8 on. Every float operation is the one candle performs,
// in the same order, as in src/model/fstep.rs (whose helpers this uses: the norms, gemm's one-row product order, the
// int8 products through `QW::matmul_rows`, the engram's read); what is new here:
// * the LIF of `arch::lif_ref` (binary, ternary or int2 spikes, the ALIF trace, the homeostatic threshold) or, for
//   binary spikes without ALIF, of `state::lif_infer`; the spikes' value is soft + (hard - soft), as candle forms it;
// * the memory of `arch::delta_chunk` at t = 1 (one chunk of one token), per channel gates, erase key, momentum. Its
//   chunk algebra at c = 1 has factors that are exactly 1 or 0 for finite inputs (the decay ratios exp(cum - cum),
//   the kernel K, the triangular inverse, the previous-row kernel); those are taken as their values (a non-finite
//   input sends the token back to the plain path). The products that stay are formed as gemm forms them: the
//   recall r = e^T F, the read q (alpha) F and the momentum reads in its one-row order, K = 1 products as plain
//   products, the channel sums one after another from 0 (candle's sum);
// * the value residual, the dense residual mixes and the per-channel gates' low-rank products (gemm's one-row order).
// The memory heads (gates, update, norm, output gate) and the engram's heads run on up to the int8 thread count of
// candle's pool (`par_heads`): each head reads and writes only its own values, so any split gives the same bits.
// It covers model D with --gates channel and any of: --head-size, --timescales, --value-residual, --kwta, --mlp-act,
// --mem-momentum, --resid-fix, --residual dense, --lif-init, --lif-scale, --surrogate-slope, --spikes, --alif,
// --homeostasis, --gate silu|sigmoid; the engram (no cap or an oldest-first cap), the glance and the magnitude code,
// and the lesions; v51 fast1 --neuromod, v51 fast2 R --letters and TH --thalamus (arch_f2.rs), v51 fast3 S --stp, M --mood,
// Pr --prune and CB --cerebellum-head (arch_f3.rs); f4ext P --dendrite, DA --dopamine and Y --rhythm (arch_f4.rs). Anything else
// (`blocker`, `state_ok`) runs `forward_with_state` as before.
// Tests: src/model/fstep/arch_tests.rs (every logit and the whole State bit for bit over 300 tokens).

use super::{add_into, append, attend, contig, gemv, head_norm, kth_largest, l2_heads, log_sigmoid, mm, mm_many, rmsnorm, ro, sigmoid, vals};
use crate::arch::{EngGate, Gating, LifState, MemState, OutGate, Spikes};
use crate::memory::Kv;
use crate::model::{Kind, Model, HEAD_SIZE, LIF_THRESHOLD, PLACES};
use crate::state::{Evict, State, HIST};
use crate::v50::{ACH, DA, NE};
use candle_core::{Result, Tensor};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

/// Why the fused step does not apply to this block-options model; None: it applies (the checks common to every
/// model, CPU, int8 and the CPU's kernels, are `Model::fused_blocker`'s).
pub(crate) fn blocker(m: &Model) -> Option<&'static str> {
    let Some(ext) = m.ext.as_ref() else { return Some("forced onto the block-options path without block options") };
    let (cfg, ar) = (m.cfg, m.cfg.arch);
    if cfg.kind != Kind::D || m.pos.is_some() {
        return Some("block options: the fused step covers model D");
    }
    if ar.gates != Gating::Channel {
        return Some("block options: the fused step covers --gates channel (per-head and KDA gates run another memory)");
    }
    if let Some(why) = crate::v51::fused_why(&arch_f4::strip(arch_f3::strip(arch_f2::strip(ar.v51)))) {
        return Some(why); // v51 (src/v51.rs; v51 fast2: R runs here; v51 fast3: M and Pr; f4ext: P, DA and Y)
    }
    if ar.feedback || ar.attn_res || ar.token_shift || ar.bonus || ar.eng_gate != EngGate::Off {
        return Some("block options: --feedback, --residual attn, --token-shift, --bonus and --eng-gate are not in the fused step");
    }
    if ar.dale > 0.0 || ar.loops > 1 {
        return Some("v50: --dale and --loop are not in the fused step");
    }
    // v51 fast1: --neuromod is (the nucleus after block 0; NE, ACh and DA in the later blocks)
    if ar.neuromod > 0 && ext.v50.nucleus.as_ref().is_none_or(|(_, w, _)| w.dims().len() != 2 || w.dims()[1] < 2) {
        return Some("v50 --neuromod without its nucleus (or one signal: gemm takes another kernel)");
    }
    if let Some(why) = arch_f2::v51b_why(m) {
        return Some(why); // v51 fast2: TH runs here (its --loop gate is v50's --loop, refused above)
    }
    // merge (runs/v51/fast4.md): v51e's shared trunk (B-4) replaces blocks 0..k of forward_with_state; not in the fused step
    if m.trunk_replay_k().is_some() || m.trunk_recording() {
        return Some("v51 B-4: the shared trunk's replay or recording is on (forward_with_state takes the trunk)");
    }
    if ar.head_size < 3 || cfg.d % ar.head_size != 0 || cfg.d / ar.head_size < 2 {
        return Some("one memory head: gemm takes another kernel for the gates' projection");
    }
    if ar.gate == OutGate::Sigmoid && ar.gate_rank < 2 {
        return Some("a rank-1 sigmoid gate: gemm takes another kernel");
    }
    if crate::arch::delta_rec_on() {
        return Some("AVA_DELTA_REC=1: the memory runs token by token (another order)");
    }
    let mx = &ext.mem;
    if mx.pkm.is_some() || ar.mem.modulate || ar.mem.eng_topk > 0 {
        return Some("block options: the memory parts (--pk-mem, --modulate, --eng-topk) are not in the fused step");
    }
    if mx.eng.as_ref().is_some_and(|e| e.tau.is_some() || e.ss.is_some() || e.btsp.is_some()) {
        return Some("block options: the engram's options are not in the fused step");
    }
    if mx.parts.iter().any(|p| p.dend.is_some() || p.delays.is_some() || p.wm.is_some()) {
        return Some("block options: the in-block parts are not in the fused step");
    }
    let w5 = |w: &crate::wave5::W5Block| w.res.is_some() || w.modl.is_some() || w.tem.is_some() || w.halt.is_some() || w.cb.is_some() || w.ws.is_some() || w.holo.is_some() || w.sync.is_some();
    if mx.w5.blocks.iter().any(w5) {
        return Some("block options: the Wave 5 parts are not in the fused step");
    }
    if ext.blocks.len() != m.blocks.len() {
        return Some("block options: a block without its extra parts");
    }
    for (blk, bx) in m.blocks.iter().zip(&ext.blocks) {
        if blk.mem.is_none() || blk.attn.is_some() {
            return Some("a block without its spiking memory, or with attention");
        }
        let Some(hb) = &blk.hebb else { return Some("a block without its fast-weight memory") };
        if hb.gates.is_none() || !hb.delta || (hb.wg.is_none() && hb.wgl.is_none()) {
            return Some("a memory without D's gates, delta rule or output gate");
        }
        if bx.chan.is_none() || bx.kda.is_some() || bx.shift.is_some() || bx.bonus.is_some() || bx.eng_gate.is_some() || bx.attn_res.is_some() {
            return Some("a block's extra parts are not the fused step's");
        }
    }
    None
}

/// The carried state is one the fused step continues (else the token runs `forward_with_state`).
fn state_ok(m: &Model, st: &State) -> bool {
    let mom = m.cfg.arch.mem_momentum > 0.0;
    st.rows() == 1
        && st.hist.len() == 1
        && st.fb.is_none()
        && st.modc.is_none()
        && st.shadow.is_none()
        && !st.cap.is_some_and(|c| c.rule == Evict::LeastRead)
        && st.lif.len() == m.blocks.len()
        && st.lif.iter().all(|l| l.is_some())
        && st.mem.len() == m.blocks.len()
        && st.mem.iter().all(|x| x.as_ref().is_some_and(|x| x.s.is_some() == mom && x.prev.is_none()))
        && st.kv.iter().all(|k| k.is_none())
        && st.eng.as_ref().is_none_or(|e| e.pad.is_none() && e.logp.is_none() && e.ktail.is_none() && e.reads.is_none() && e.age.is_none() && e.valid.len() == 1)
        && (m.cfg.engram.is_some() || st.eng.is_none())
        && arch_f3::state_ok(m, st) // v51 fast3: S's (u, x), M's stream, surprise and mood
}

/// The fused step's section clock (on and zeroed with `crate::int8::set_mm_prof`, whose timer counts the int8
/// products): nanoseconds in each section, its int8 products excluded.
static SECS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// The section names of `take_secs`.
pub const SECTIONS: [&str; 6] = ["input (embedding, glance, magnitude)", "LIF (no int8)", "memory gate projections (no int8)", "memory heads (gates, update, norm, out gate)", "engram read (no int8)", "MLP kWTA + norms + mixes + rest"];

/// Nanoseconds per section since the last call (zeroes them).
pub fn take_secs() -> [u64; 6] {
    std::array::from_fn(|c| SECS[c].swap(0, Relaxed))
}

/// Section timer: the time since `t0` minus the int8 time spent meanwhile goes to section `c`.
struct Clock {
    on: bool,
    t: std::time::Instant,
    mm: u64,
}

impl Clock {
    fn new() -> Clock {
        Clock { on: crate::int8::mm_prof_on(), t: std::time::Instant::now(), mm: crate::int8::mm_prof().1 }
    }

    fn lap(&mut self, c: usize) {
        if !self.on {
            return;
        }
        let (now, mm) = (std::time::Instant::now(), crate::int8::mm_prof().1);
        let ns = (now - self.t).as_nanos() as u64;
        SECS[c].fetch_add(ns.saturating_sub(mm - self.mm), Relaxed);
        self.t = now;
        self.mm = mm;
    }
}

fn t1(v: Vec<f32>, d: usize, dev: &candle_core::Device) -> Result<Tensor> {
    Tensor::from_vec(v, (1, d), dev)
}

/// One LIF update of one row (see `arch::lif_ref`): returns (spikes, membrane, trace).
#[allow(clippy::too_many_arguments)]
fn lif(cur: &[f32], beta: &[f32], theta: Option<&[f32]>, alif: Option<(&[f32], &[f32])>, spikes: Spikes, slope: f32, v0: &[f32], s0: &[f32], a0: Option<&[f32]>) -> (Vec<f32>, Vec<f32>, Option<Vec<f32>>) {
    let d = cur.len();
    let one = LIF_THRESHOLD as f32;
    let th = |c: usize| theta.map_or(one, |t| t[c]);
    let (mut s, mut v) = (vec![0f32; d], vec![0f32; d]);
    if spikes == Spikes::Binary && alif.is_none() {
        // state::lif_infer: v = beta·v + I - th·s; s = H(v - th)
        for c in 0..d {
            let vv = beta[c] * v0[c] + cur[c] - th(c) * s0[c];
            v[c] = vv;
            s[c] = if vv - th(c) >= 0.0 { 1.0 } else { 0.0 };
        }
        return (s, v, None);
    }
    let soft_step = |u: f32| u / (u.abs() * slope + 1.0);
    let heav = |u: f32| if u >= 0.0 { 1f32 } else { 0f32 };
    let mut a = alif.map(|_| vec![0f32; d]);
    for c in 0..d {
        // the threshold the last spikes crossed, then the trace and this step's threshold
        let (th_prev, th_t) = match (alif, a.as_mut()) {
            (Some((rho, gain)), Some(a)) => {
                let a_prev = a0.map_or(0.0, |x| x[c]);
                let th_prev = th(c) + a_prev * gain[c];
                let an = a_prev * rho[c] + s0[c].abs() * (rho[c] * -1.0 + 1.0);
                a[c] = an;
                (th_prev, th(c) + an * gain[c])
            }
            _ => (th(c), th(c)),
        };
        let vv = v0[c] * beta[c] + cur[c] - s0[c] * th_prev;
        v[c] = vv;
        s[c] = match spikes {
            Spikes::Binary => {
                let u = vv - th_t;
                let sg = soft_step(u);
                sg + (heav(u) - sg)
            }
            Spikes::Ternary | Spikes::Int2 => {
                let (mut soft, mut hard) = (0f32, 0f32);
                for k in 1..=spikes.levels() {
                    let kth = th_t * k as f32 + 0.0;
                    let (up, down) = (vv - kth, -vv - kth);
                    let s1 = soft_step(up) - soft_step(down);
                    let h1 = heav(up) - heav(down);
                    if k == 1 {
                        (soft, hard) = (s1, h1);
                    } else {
                        (soft, hard) = (soft + s1, hard + h1);
                    }
                }
                soft + (hard - soft)
            }
        };
    }
    (s, v, a)
}

/// One memory head's inputs at one token (chunk_core with c = 1; see the top): the read query q, the erase key e,
/// the values v, the write key w = b*k, the erase key ah = e*(b*alpha), q*alpha, alpha, the sums q·w and q·ah, the
/// recall's zero term ke, and with momentum M: q*M and the zero row p2 = e*0.
struct HeadIn<'a> {
    hs: usize,
    e: &'a [f32],
    v: &'a [f32],
    w: &'a [f32],
    ah: &'a [f32],
    qg: &'a [f32],
    ex: &'a [f32],
    qw: f32,
    qa: f32,
    ke: f32,
    tinv: f32,
    mom: Option<(f32, &'a [f32], &'a [f32])>,
    f: &'a [f32],
    s: Option<&'a [f32]>,
}

/// One head: y [hs] (before the head norm), the new fast weights and momentum ([hs, hs] each), as chunk_core's loop:
///     rr = (e F + r0) + p2 S                      (the recall; gemm's one-row order)
///     y  = ((q alpha) F + (q.w) v) - (q.ah) rr + (q M) S
///     F' = ((F * alpha_row + w v^T) - ah rr^T) + S * M
///     S' = ((S * M + 0) + w v^T) - ah rr^T
#[inline(always)]
fn head_rows(h: &HeadIn, y: &mut [f32], fo: &mut [std::mem::MaybeUninit<f32>], so: Option<&mut [std::mem::MaybeUninit<f32>]>) {
    let hs = h.hs;
    assert!(h.e.len() >= hs && h.v.len() >= hs && h.w.len() >= hs && h.ah.len() >= hs && h.qg.len() >= hs && h.ex.len() >= hs);
    assert!(h.f.len() == hs * hs && fo.len() == hs * hs && y.len() >= hs);
    let (v, y) = (&h.v[..hs], &mut y[..hs]);
    let mut rr = vec![0f32; hs];
    let mut t1 = vec![0f32; hs];
    for (a, frow) in h.f.chunks_exact(hs).enumerate() {
        let (pa, ga) = (h.e[a], h.qg[a]);
        if a == 0 {
            for ((r, t), &fv) in rr.iter_mut().zip(t1.iter_mut()).zip(frow) {
                *r = fv * pa;
                *t = fv * ga;
            }
        } else {
            for ((r, t), &fv) in rr.iter_mut().zip(t1.iter_mut()).zip(frow) {
                *r = fv.mul_add(pa, *r);
                *t = fv.mul_add(ga, *t);
            }
        }
    }
    // r0 = tinv (ke v): the recall's (exactly zero) term from inside the chunk
    for (r, &vb) in rr.iter_mut().zip(v) {
        *r = *r + h.tinv * (h.ke * vb);
    }
    let mut tq = Vec::new();
    if let (Some((_, qs, p2)), Some(s)) = (h.mom, h.s) {
        assert!(s.len() == hs * hs && qs.len() >= hs && p2.len() >= hs);
        let mut z = vec![0f32; hs];
        tq = vec![0f32; hs];
        for (a, srow) in s.chunks_exact(hs).enumerate() {
            let (pa, qa) = (p2[a], qs[a]);
            if a == 0 {
                for ((zz, t), &sv) in z.iter_mut().zip(tq.iter_mut()).zip(srow) {
                    *zz = sv * pa;
                    *t = sv * qa;
                }
            } else {
                for ((zz, t), &sv) in z.iter_mut().zip(tq.iter_mut()).zip(srow) {
                    *zz = sv.mul_add(pa, *zz);
                    *t = sv.mul_add(qa, *t);
                }
            }
        }
        for (r, &zz) in rr.iter_mut().zip(&z) {
            *r = *r + zz;
        }
    }
    for b in 0..hs {
        y[b] = (t1[b] + h.qw * v[b]) - h.qa * rr[b];
    }
    if !tq.is_empty() {
        for (yb, &t) in y.iter_mut().zip(&tq) {
            *yb = *yb + t;
        }
    }
    match (h.mom, h.s, so) {
        (Some((m, _, _)), Some(s), Some(so)) => {
            assert!(so.len() == hs * hs);
            for a in 0..hs {
                let (wa, aa, xa) = (h.w[a], h.ah[a], h.ex[a]);
                let (frow, srow) = (&h.f[a * hs..(a + 1) * hs], &s[a * hs..(a + 1) * hs]);
                let (fo, so) = (&mut fo[a * hs..(a + 1) * hs], &mut so[a * hs..(a + 1) * hs]);
                for b in 0..hs {
                    let (wv, ar) = (wa * v[b], aa * rr[b]);
                    fo[b].write(((frow[b] * xa + wv) - ar) + srow[b] * m);
                    so[b].write(((srow[b] * m + 0.0) + wv) - ar);
                }
            }
        }
        _ => {
            for a in 0..hs {
                let (wa, aa, xa) = (h.w[a], h.ah[a], h.ex[a]);
                let frow = &h.f[a * hs..(a + 1) * hs];
                let fo = &mut fo[a * hs..(a + 1) * hs];
                for b in 0..hs {
                    fo[b].write((frow[b] * xa + wa * v[b]) - aa * rr[b]);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn head(h: &HeadIn, y: &mut [f32], fo: &mut [std::mem::MaybeUninit<f32>], so: Option<&mut [std::mem::MaybeUninit<f32>]>) {
    // SAFETY: the fused step runs only where AVX2 + FMA were detected (Model::fused_blocker)
    unsafe { avx::head(h, y, fo, so) }
}

#[cfg(not(target_arch = "x86_64"))]
fn head(h: &HeadIn, y: &mut [f32], fo: &mut [std::mem::MaybeUninit<f32>], so: Option<&mut [std::mem::MaybeUninit<f32>]>) {
    head_rows(h, y, fo, so)
}

#[cfg(target_arch = "x86_64")]
mod avx {
    /// `super::head_rows` compiled for AVX2 + FMA (elementwise over the columns: the same bits at any width).
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn head(h: &super::HeadIn, y: &mut [f32], fo: &mut [std::mem::MaybeUninit<f32>], so: Option<&mut [std::mem::MaybeUninit<f32>]>) {
        super::head_rows(h, y, fo, so)
    }
}

/// f(e) for every head e in 0..nh, on up to `threads` threads of candle's pool (as `QW::matmul_rows` uses it), each
/// thread a range of heads; one thread: one after another on the calling thread. Never called from inside the pool.
fn par_heads(nh: usize, threads: usize, f: impl Fn(usize) + Sync) {
    let pool = candle_core::utils::barrier_pool();
    let nt = threads.min(nh).min(pool.n_workers() + 1);
    if nt <= 1 {
        (0..nh).for_each(f);
        return;
    }
    let per = nh.div_ceil(nt);
    pool.execute(|tid| {
        if tid < nt {
            ((tid * per).min(nh)..((tid + 1) * per).min(nh)).for_each(&f);
        }
    });
}


/// v51 fast4: the step's one-row products: int8 (`QW::matmul_rows`, int8 on) or f32 in gemm's one-row order (int8
/// off with `Model::set_fast4`: x · W as `int8::mm` then multiplies, the head as `int8::head`; src/fast4/fmm.rs).
pub(crate) enum Prods<'a> {
    Q8 { set: &'a crate::int8::Q8Set, head: &'a crate::int8::QW, th: usize },
    F32 { tok: &'a Tensor, th: usize },
}

impl Prods<'_> {
    /// x · W.
    pub(crate) fn xw(&self, w: &Tensor, x: &[f32]) -> Result<Vec<f32>> {
        match self {
            Prods::Q8 { set, th, .. } => Ok(mm(set.get(w).expect("checked in step_rec"), x, *th)),
            Prods::F32 { th, .. } => {
                let n = w.dim(1)?;
                Ok(crate::fast4::fmm::xw(x, ro(w)?.s(), n, *th))
            }
        }
    }

    /// x · W for each of `ws` (int8: one product of their side-by-side copy; f32: each its own, the same bits as one).
    fn many(&self, ws: &[&Tensor], x: &[f32]) -> Result<Vec<Vec<f32>>> {
        match self {
            Prods::Q8 { set, th, .. } => Ok(mm_many(set, ws, x, *th)),
            Prods::F32 { .. } => ws.iter().map(|w| self.xw(w, x)).collect(),
        }
    }

    /// The head x · E^T.
    pub(crate) fn head(&self, x: &[f32]) -> Result<Vec<f32>> {
        match self {
            Prods::Q8 { head, th, .. } => Ok(mm(head, x, *th)),
            Prods::F32 { tok, th } => {
                let v = tok.dim(0)?;
                Ok(crate::fast4::fmm::xet(x, ro(tok)?.s(), v, *th))
            }
        }
    }
}
/// All finite (the chunk's exact 1s and 0s hold only for finite values).
fn finite(x: &[f32]) -> bool {
    x.iter().all(|v| v.is_finite())
}

/// v49 (the chat app): one fused token's activity, for `Model::step_trace` (src/model.rs). Every value is a copy of
/// what the step computed, or a size computed on the side from those copies; nothing here feeds back, so the logits
/// and the State are the untraced step's bit for bit (src/model/fstep/trace_tests.rs checks it).
#[derive(Default)]
pub(crate) struct TokTrace {
    /// per block, in order
    pub(crate) layers: Vec<TokLayer>,
    /// the glance's place code (16 = none) and the magnitude code's position (None = it added nothing)
    pub(crate) place: u32,
    pub(crate) mag: Option<f64>,
    /// RMS of what entered her stream for this token (embedding + glance + magnitude code), before block 0
    pub(crate) input_rms: f32,
}

/// One block of `TokTrace`.
#[derive(Default)]
pub(crate) struct TokLayer {
    /// the LIF spikes [d] as they reach Wout (int2: -2..2)
    pub(crate) spikes: Vec<f32>,
    /// the kWTA winners [hidden] (0/1) and the MLP's units after kWTA, relu (and relu2) [hidden], as they reach W2
    pub(crate) winners: Vec<f32>,
    pub(crate) hidden: Vec<f32>,
    /// the fast-weight memory per head [heads]: write = |F' - alpha F| (Frobenius, alpha per row), read = |y| (the
    /// head's read before its norm), keep = mean alpha, gate = mean write strength b over the head's channels
    pub(crate) mem_write: Vec<f32>,
    pub(crate) mem_read: Vec<f32>,
    pub(crate) mem_keep: Vec<f32>,
    pub(crate) mem_gate: Vec<f32>,
    /// the engram (when not lesioned): its attention [heads, seen] (this token last) and its read [d] as added
    pub(crate) engram: Option<(Vec<f32>, usize, Vec<f32>)>,
}

/// The engram's attention weights of one head over its n keys, on the side: `attend`'s scores and softmax.
fn attn_row(q: &[f32], kc: &[f32], n: usize, out: &mut [f32]) {
    let scale = (1.0 / (HEAD_SIZE as f64).sqrt()) as f32;
    super::scores(q, kc, HEAD_SIZE, scale, out);
    let mut mx = out[0];
    for &s in out.iter() {
        if mx < s {
            mx = s;
        }
    }
    let mut den = 0f32;
    for s in out.iter_mut() {
        *s = (*s - mx).exp();
        den += *s;
    }
    for s in out.iter_mut() {
        *s = *s / den;
    }
    debug_assert_eq!(out.len(), n);
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

/// `forward_with_state` for one token on a one-row State, fused; None (nothing changed) where it does not apply.
pub(crate) fn step(m: &Model, st: &mut State, tok: u32) -> Result<Option<Tensor>> {
    step_rec(m, st, tok, None)
}

/// v49: `step` that also fills `rec` with the token's activity (the same logits and State bit for bit). On None the
/// record is incomplete and must not be used.
pub(crate) fn step_traced(m: &Model, st: &mut State, tok: u32, rec: &mut TokTrace) -> Result<Option<Tensor>> {
    *rec = TokTrace::default();
    step_rec(m, st, tok, Some(rec))
}

fn step_rec(m: &Model, st: &mut State, tok: u32, rec: Option<&mut TokTrace>) -> Result<Option<Tensor>> {
    match step_inner(m, st, tok, rec, None)? {
        Some(Done::Logits(l)) => Ok(Some(l)),
        _ => Ok(None),
    }
}

/// v51 fast4: what the fused step hands back: the logits, or (`trunk`) the stream after the first k blocks.
enum Done {
    Logits(Tensor),
    Trunk(Trunk),
}

/// v51 fast4 (src/fast4/eprop.rs): blocks 0..k of the fused step, the input of block k: x, the dense streams so far
/// (the input first), block 0's values (value residual; None when k = 0) and the history with this token.
pub(crate) struct Trunk {
    pub(crate) x: Vec<f32>,
    pub(crate) streams: Vec<Vec<f32>>,
    pub(crate) v_first: Option<Vec<f32>>,
    pub(crate) full: Vec<u32>,
}

/// v51 fast4: the fused step's blocks 0..k only (the e-prop session's frozen trunk): their LIF and memory states
/// (and the engram store, S and M) are committed; the token count and history are not (the caller moves them on
/// after the remaining blocks). None: the fused step does not take this token (nothing is committed). v52: `rec` (the
/// chat app's viewer, on the side) gets the input's and blocks 0..k's activity, as `step_traced` records them.
pub(crate) fn trunk(m: &Model, st: &mut State, tok: u32, k: usize, rec: Option<&mut TokTrace>) -> Result<Option<Trunk>> {
    if k > m.blocks.len() {
        candle_core::bail!("fused trunk: {k} blocks of {}", m.blocks.len());
    }
    match step_inner(m, st, tok, rec, Some(k))? {
        Some(Done::Trunk(t)) => Ok(Some(t)),
        _ => Ok(None),
    }
}

fn step_inner(m: &Model, st: &mut State, tok: u32, mut rec: Option<&mut TokTrace>, stop: Option<usize>) -> Result<Option<Done>> {
    if !m.arch_path() || m.fused_blocker().is_some() || !state_ok(m, st) {
        return Ok(None);
    }
    let tracing = rec.is_some();
    let set = m.q8.borrow().clone();
    let (cfg, ar) = (m.cfg, m.cfg.arch);
    let ext = m.ext.as_ref().expect("checked in blocker");
    let (d, hs) = (cfg.d, ar.head_size);
    let nh = d / hs;
    let th = set.as_ref().map_or_else(candle_core::utils::get_num_threads, |s| s.threads);
    let dev = m.tok.device().clone();
    if (tok as usize) >= cfg.vocab {
        return Ok(None);
    }
    for (blk, bx) in m.blocks.iter().zip(&ext.blocks) {
        let (mem, hb, c) = (blk.mem.as_ref().expect("checked"), blk.hebb.as_ref().expect("checked"), bx.chan.as_ref().expect("checked"));
        let mut ws = vec![&mem.win, &mem.wout, &hb.wq, &hb.wk, &hb.wv, &hb.wo, &c.we, &blk.mlp.w1, &blk.mlp.w2];
        ws.extend(hb.wg.iter());
        if let Some(e) = &blk.eng {
            ws.extend([&e.wq, &e.wk, &e.wv, &e.wo]);
        }
        let ok = |w: &&Tensor| match &set {
            Some(s) => s.get(w).is_some(),
            None => w.rank() == 2 && w.is_contiguous(), // v51 fast4 f32: gemm's one-row order needs W [k, n] row-major
        };
        if !ws.iter().all(ok) {
            return Ok(None);
        }
    }
    // v51 fast4: the one-row products, int8 or (int8 off, `set_fast4`) f32 in gemm's order
    let pr = match &set {
        Some(s) => match s.head_of(&m.tok) {
            Some(head) => Prods::Q8 { set: s, head, th },
            None => return Ok(None),
        },
        None if m.tok.is_contiguous() => Prods::F32 { tok: &m.tok, th },
        None => return Ok(None),
    };
    let (place_off, mag_off, eng_off) = m.lesion.get();
    let mom = ar.mem_momentum > 0.0;
    let mf = ar.mem_momentum;
    let mut clk = Clock::new();

    // the input: the token's embedding, its place (v39 glance) and its magnitude code (v40), over the history kept
    let full: Vec<u32> = st.hist[0].iter().copied().chain(std::iter::once(tok)).collect();
    let h0 = full.len() - 1;
    let mut x: Vec<f32> = {
        let e = ro(&m.tok)?;
        e.s()[tok as usize * d..(tok as usize + 1) * d].to_vec()
    };
    if let (Some(place), false) = (&m.place, place_off) {
        let p = ava_data::numbers::place_ids(&full)[h0] as usize;
        if p < PLACES {
            add_into(&mut x, &ro(place)?.s()[p * d..(p + 1) * d]);
        } else {
            // the table's zero row ("none")
            x.iter_mut().for_each(|v| *v = *v + 0.0);
        }
    }
    if let (Some(mw), false) = (&m.mag_w, mag_off) {
        let fe = &crate::mag::feats_row(&full)[h0 * crate::mag::MAG_F..];
        let mut o = vec![0f32; d];
        gemv(fe, ro(mw)?.s(), d, &mut o);
        add_into(&mut x, &o);
    }
    if let Some(o) = arch_f2::letters_add(m, tok)? {
        add_into(&mut x, &o); // v51 fast2 R: the letters
    }
    // f4ext Y: the rhythms at this token, from the input after the letters (v51::begin); their gates in every block
    let rhythm = arch_f4::Rhythm::begin(m, st, &x)?;
    if let Some(r) = rec.as_deref_mut() {
        // on the side: what the viewer shows of the input (the same rules forward_with_state adds by)
        r.input_rms = rms(&x);
        r.place = match (&m.place, place_off) {
            (Some(_), false) => ava_data::numbers::place_ids(&full)[h0].min(PLACES as u32),
            _ => PLACES as u32,
        };
        r.mag = match (&m.mag_w, mag_off) {
            (Some(_), false) => crate::mag::position(&full, h0),
            _ => None,
        };
    }
    clk.lap(0);

    let nb = m.blocks.len();
    let mut lifs: Vec<LifState> = Vec::with_capacity(nb);
    let mut mems: Vec<MemState> = Vec::with_capacity(nb);
    let mut eng_new: Option<Kv> = None;
    let mut streams: Vec<Vec<f32>> = if ar.dense { vec![x.clone()] } else { Vec::new() };
    let mut v_first: Option<Vec<f32>> = None;
    // v51 fast1 --neuromod: the signals (NE, ACh, DA) after block 0; None in block 0 and without the option
    let mut sig: Option<Vec<f32>> = None;
    let mut relay = arch_f2::Relay::begin(m, &x)?; // v51 fast2 TH: the relay reads the blocks' input
    // v51 fast3: each block's new synapse state (S), and block 0's surprise and mood (M; None in block 0)
    let mut stps: Vec<Option<(Vec<f32>, Vec<f32>)>> = Vec::with_capacity(nb);
    let mut mood: Option<arch_f3::Mood> = None;
    for (i, (blk, bx)) in m.blocks.iter().zip(&ext.blocks).enumerate() {
        if stop == Some(i) {
            break; // v51 fast4: the trunk ends before block k
        }
        let x_in = relay.as_ref().map(|_| x.clone());
        if let Some(r) = relay.as_mut() {
            r.block(i); // TH: this block's gates
        }
        // f4ext Y: this block's rhythm gains per head (the write strength's, the read's)
        let rg = match rhythm.as_ref() {
            Some(r) => r.gates(m, i, nh)?,
            None => None,
        };
        // the spiking memory (block_state: lif_scale, theta, ALIF, the spike kind)
        let mem = blk.mem.as_ref().expect("checked");
        let h = rmsnorm(&x, ro(&mem.norm)?.s());
        clk.lap(5);
        let mut cur = pr.xw(&mem.win, &h)?;
        let beta: Vec<f32> = ro(&mem.beta)?.s().iter().map(|&b| sigmoid(b)).collect();
        if ar.lif_scale {
            for (c, &b) in cur.iter_mut().zip(&beta) {
                *c = *c * (b * -1.0 + 1.0);
            }
        }
        if let Some(s) = sig.as_ref() {
            // v51 fast1 --neuromod NE: the input current times its signal (v50_ne, after --lif-scale)
            cur.iter_mut().for_each(|c| *c = *c * s[NE]);
        }
        arch_f3::Mood::gain(mood.as_ref(), m, i, &mut cur)?; // v51 fast3 M: the mood's gain (v51::lif_gain, after NE)
        let ach = sig.as_ref().map(|s| s[ACH]);
        let ls = st.lif[i].as_ref().expect("checked");
        let (v0, s0) = (vals(&ls.v)?, vals(&ls.s)?);
        let a0 = match (&bx.alif, &ls.a) {
            (Some(_), Some(a)) => Some(vals(a)?),
            _ => None,
        };
        let theta = bx.theta.as_ref().map(vals).transpose()?;
        let alif = match &bx.alif {
            Some((r, g)) => Some((ro(r)?.s().iter().map(|&x| sigmoid(x)).collect::<Vec<f32>>(), vals(g)?)),
            None => None,
        };
        if v0.len() != d || s0.len() != d || a0.as_ref().is_some_and(|a| a.len() != d) {
            return Ok(None);
        }
        let (s, v1, a1) = lif(&cur, &beta, theta.as_deref(), alif.as_ref().map(|(r, g)| (r.as_slice(), g.as_slice())), ar.spikes, ar.slope, &v0, &s0, a0.as_deref());
        // v51 fast3 S: the spikes through the synapse (what reaches W_out; the LIF keeps s)
        let syn = arch_f3::stp(m, st, i, &s)?;
        let sw = syn.as_ref().map_or(&s, |(o, _)| o);
        clk.lap(1);
        let mut lay = TokLayer::default();
        if tracing {
            lay.spikes = sw.clone();
        }
        add_into(&mut x, &pr.xw(&mem.wout, sw)?);
        stps.push(syn.map(|(_, n)| n));
        lifs.push(LifState { v: t1(v1, d, &dev)?, s: t1(s, d, &dev)?, a: a1.map(|a| t1(a, d, &dev)).transpose()? });

        // the fast-weight memory (mem_arch with per-channel gates, delta_chunk at t = 1)
        let hb = blk.hebb.as_ref().expect("checked");
        let g = hb.gates.as_ref().expect("checked");
        let c = bx.chan.as_ref().expect("checked");
        let h = rmsnorm(&x, ro(&hb.norm)?.s());
        clk.lap(5);
        let mut ws = vec![&hb.wq, &hb.wk, &hb.wv, &c.we];
        ws.extend(hb.wg.iter());
        let mut outs = pr.many(&ws, &h)?.into_iter();
        let (mut q, kraw, mut v, weo) = (outs.next().expect("q"), outs.next().expect("k"), outs.next().expect("v"), outs.next().expect("we"));
        let gq = outs.next();
        l2_heads(&mut q, hs);
        let mut k = kraw.clone();
        l2_heads(&mut k, hs);
        if ar.value_residual {
            match (&v_first, &bx.vres) {
                (None, _) => v_first = Some(v.clone()),
                (Some(vf), Some(p)) => {
                    // (1 + p0) v + p1 v0
                    let p = vals(p)?;
                    let p0 = p[0] * 1.0 + 1.0;
                    for (a, &b) in v.iter_mut().zip(vf) {
                        *a = *a * p0 + b * p[1];
                    }
                }
                (Some(_), None) => {}
            }
        }
        // the gates' projections: za = h·Wa + ba, zb = h·Wb + bb (one per head), the low-rank h·Wa1·Wa2, h·Wb1·Wb2
        let r = c.wa1.dim(1)?;
        let (mut za, mut zb) = (vec![0f32; nh], vec![0f32; nh]);
        gemv(&h, ro(&g.wa)?.s(), nh, &mut za);
        gemv(&h, ro(&g.wb)?.s(), nh, &mut zb);
        add_into(&mut za, ro(&g.ba)?.s());
        add_into(&mut zb, ro(&g.bb)?.s());
        let (mut z1, mut lra, mut lrb) = (vec![0f32; r], vec![0f32; d], vec![0f32; d]);
        gemv(&h, ro(&c.wa1)?.s(), r, &mut z1);
        gemv(&z1, ro(&c.wa2)?.s(), d, &mut lra);
        gemv(&h, ro(&c.wb1)?.s(), r, &mut z1);
        gemv(&z1, ro(&c.wb2)?.s(), d, &mut lrb);
        // the sigmoid output gate's projection h·Wd·Wu (it reads h only; the silu gate's is the int8 product gq)
        let zgate: Option<Vec<f32>> = match (&gq, &hb.wgl) {
            (None, Some((wd, wu))) => {
                let (wd, wu) = (ro(wd)?, ro(wu)?);
                let rk = wd.s().len() / d;
                let mut z1 = vec![0f32; rk];
                gemv(&h, wd.s(), rk, &mut z1);
                let mut z = vec![0f32; d];
                gemv(&z1, wu.s(), d, &mut z);
                Some(z)
            }
            _ => None,
        };
        let off = ro(&c.off)?;
        let off = off.s();
        let ms = st.mem[i].as_ref().expect("checked");
        let (ft, smt) = (contig(&ms.f)?, ms.s.as_ref().map(contig).transpose()?);
        let (fr, sr) = (ro(&ft)?, smt.as_ref().map(ro).transpose()?);
        let len = nh * hs * hs;
        if fr.s().len() != len || sr.as_ref().is_some_and(|s| s.s().len() != len) || off.len() != d {
            return Ok(None);
        }
        clk.lap(2);
        let mut y = vec![0f32; d];
        let mut fnew: Vec<f32> = Vec::with_capacity(len);
        let mut snew: Vec<f32> = Vec::with_capacity(if mom { len } else { 0 });
        let bad = std::sync::atomic::AtomicBool::new(false);
        // v49 trace, on the side (only when tracing): per head its read size, mean keep and mean write strength, and
        // every channel's keep alpha (for the write size below); each head writes only its own slots
        let side: Vec<AtomicU32> = (0..if tracing { 3 * nh + d } else { 0 }).map(|_| AtomicU32::new(0)).collect();
        {
            // Each head from its gates to its gated output: it reads its own channels and writes its own outputs, so on
            // up to `th` threads the heads give the same bits as one after another.
            let (yp, fp, spp) = (y.as_mut_ptr() as usize, fnew.as_mut_ptr() as usize, snew.as_mut_ptr() as usize);
            let (fs, ss) = (fr.s(), sr.as_ref().map(|s| s.s()));
            let (q, k, v, kraw, weo) = (&q, &k, &v, &kraw, &weo);
            let run = |e: usize| {
                let sp = e * hs..(e + 1) * hs;
                // the gates: log alpha = log_sigmoid((za + off_ch) + lra), b = 2 sigmoid(zb + lrb)
                let la: Vec<f32> = sp.clone().map(|ch| log_sigmoid((za[e] + off[ch]) + lra[ch])).collect();
                let be: Vec<f32> = sp.clone().map(|ch| sigmoid(zb[e] + lrb[ch]) * 2.0 + 0.0).collect();
                // f4ext Y: the rhythms gate the write strength (v51::write_gate, before ACh's clamp as mem_arch runs it)
                let be: Vec<f32> = match &rg {
                    Some((gw, _)) => be.iter().map(|&b| b * gw[e]).collect(),
                    None => be,
                };
                // v51 fast1 --neuromod ACh: the write strength times its signal, kept in [0, 2] (v50::ach_scale: candle's
                // clamp is maximum then minimum)
                let be: Vec<f32> = match ach {
                    Some(s) => be.iter().map(|&b| clamp02(b * s)).collect(),
                    None => be,
                };
                // the erase key: unit(k_raw + h·We)
                let mut ek: Vec<f32> = kraw[sp.clone()].iter().zip(&weo[sp.clone()]).map(|(&a, &b)| a + b).collect();
                l2_heads(&mut ek, hs);
                let (qh, kh, vh) = (&q[sp.clone()], &k[sp.clone()], &v[sp.clone()]);
                // the chunk's per-channel terms (delta_chunk: w = k*b, ah = e*(b*alpha); chunk_core at c = 1)
                let ex: Vec<f32> = la.iter().map(|l| l.exp()).collect();
                let w: Vec<f32> = kh.iter().zip(&be).map(|(&a, &b)| a * b).collect();
                let ah: Vec<f32> = ek.iter().zip(be.iter().zip(&ex)).map(|(&e, (&b, &x))| e * (b * x)).collect();
                let qg: Vec<f32> = qh.iter().zip(&ex).map(|(&a, &b)| a * b).collect();
                if !finite(&la) || !finite(&ek) || !finite(&w) || !finite(&ah) || !finite(qh) || !finite(vh) {
                    bad.store(true, Relaxed);
                    return;
                }
                let qs: Vec<f32> = if mom { qh.iter().map(|&a| a * (1.0 * mf + 0.0)).collect() } else { Vec::new() };
                let gsx = 0f32 * mf + 0.0;
                let p2: Vec<f32> = if mom { ek.iter().map(|&e| e * gsx).collect() } else { Vec::new() };
                // the pair sums (pair2: q·(K w), q·(K ah) with K = 1; e·(Kp w), e·(Kp ah) with Kp = 0), each one after
                // another from 0 (candle's sum)
                let sum0 = |a: &[f32], b: &[f32]| {
                    let mut s = 0f32;
                    for (&x, &y) in a.iter().zip(b) {
                        s += x * y;
                    }
                    s
                };
                let e0: Vec<f32> = ek.iter().map(|&x| x * 0.0).collect();
                let (qw, qa) = (sum0(qh, &w), sum0(qh, &ah));
                let (ke, ae) = (sum0(&e0, &w), sum0(&e0, &ah));
                // inv_unit_lower at c = 1: -(ae · 0) + 1
                let tinv = -(ae * 0.0) + 1.0;
                if tinv != 1.0 {
                    bad.store(true, Relaxed);
                    return;
                }
                let hin = HeadIn {
                    hs,
                    e: &ek,
                    v: vh,
                    w: &w,
                    ah: &ah,
                    qg: &qg,
                    ex: &ex,
                    qw,
                    qa,
                    ke,
                    tinv,
                    mom: if mom { Some((mf, &qs, &p2)) } else { None },
                    f: &fs[e * hs * hs..(e + 1) * hs * hs],
                    s: ss.map(|s| &s[e * hs * hs..(e + 1) * hs * hs]),
                };
                type U = std::mem::MaybeUninit<f32>;
                // SAFETY: head e writes only y[e hs..(e+1) hs] (y has d = nh hs values) and values e hs^2..(e+1) hs^2
                // of fnew and (with momentum) snew, whose capacity is len = nh hs^2; no other head touches them
                let (ye, fo, so) = unsafe {
                    (
                        std::slice::from_raw_parts_mut((yp as *mut f32).add(e * hs), hs),
                        std::slice::from_raw_parts_mut((fp as *mut U).add(e * hs * hs), hs * hs),
                        if mom { Some(std::slice::from_raw_parts_mut((spp as *mut U).add(e * hs * hs), hs * hs)) } else { None },
                    )
                };
                head(&hin, ye, fo, so);
                if tracing {
                    let put = |j: usize, v: f32| side[j].store(v.to_bits(), Relaxed);
                    put(e, ye.iter().map(|v| v * v).sum::<f32>().sqrt());
                    put(nh + e, ex.iter().sum::<f32>() / hs as f32);
                    put(2 * nh + e, be.iter().sum::<f32>() / hs as f32);
                    for (a, &xa) in ex.iter().enumerate() {
                        put(3 * nh + e * hs + a, xa);
                    }
                }
                // the head norm, then the output gate (silu of the int8 product, or the low-rank sigmoid)
                head_norm(ye, hs);
                if let Some((_, gr)) = &rg {
                    // f4ext Y: and the read (v51::read_gate, after the head norm, before the output gate)
                    for a in ye.iter_mut() {
                        *a = *a * gr[e];
                    }
                }
                if let Some(gv) = &gq {
                    for (a, &g) in ye.iter_mut().zip(&gv[sp.clone()]) {
                        *a = *a * (g / (1.0 + (-g).exp()));
                    }
                } else if let Some(z) = &zgate {
                    for (a, &zv) in ye.iter_mut().zip(&z[sp.clone()]) {
                        *a = *a * sigmoid(zv);
                    }
                }
            };
            par_heads(nh, th, run);
        }
        if bad.load(Relaxed) {
            // a non-finite value: the plain path takes the token (nothing is committed yet)
            return Ok(None);
        }
        // SAFETY: `head` wrote every one of the len values of each (every head's hs rows of hs; no head bailed out)
        unsafe {
            fnew.set_len(len);
            if mom {
                snew.set_len(len);
            }
        }
        if tracing {
            // the write: |F' - alpha F| per head (alpha per row a, as the update scales F's rows), from the copies
            let got = |j: usize| f32::from_bits(side[j].load(Relaxed));
            let fs = fr.s();
            lay.mem_write = (0..nh)
                .map(|e| {
                    let mut s2 = 0f32;
                    for a in 0..hs {
                        let xa = got(3 * nh + e * hs + a);
                        let base = e * hs * hs + a * hs;
                        for b in 0..hs {
                            let dlt = fnew[base + b] - fs[base + b] * xa;
                            s2 += dlt * dlt;
                        }
                    }
                    s2.sqrt()
                })
                .collect();
            lay.mem_read = (0..nh).map(got).collect();
            lay.mem_keep = (0..nh).map(|e| got(nh + e)).collect();
            lay.mem_gate = (0..nh).map(|e| got(2 * nh + e)).collect();
        }
        clk.lap(3);
        // f4ext P: the memory's read (before TH's gate, as v51::put_ctx keeps it), the context of this block's dendrites
        let mut ctx: Option<Vec<f32>> = None;
        {
            let mut yo = pr.xw(&hb.wo, &y)?;
            if arch_f4::has_dendrites(m, i) {
                ctx = Some(yo.clone());
            }
            if let Some(r) = &relay {
                r.mem(&mut yo); // TH: the memory gate
            }
            add_into(&mut x, &yo);
        }
        mems.push(MemState { f: Tensor::from_vec(fnew, (nh, hs, hs), &dev)?, s: if mom { Some(Tensor::from_vec(snew, (nh, hs, hs), &dev)?) } else { None }, prev: None });

        // the engram (memory::read for one token: every earlier token's key and value, heads of 32; a lesioned engram
        // keeps writing its store and adds nothing)
        if let Some(e) = &blk.eng {
            let h = rmsnorm(&x, ro(&e.norm)?.s());
            clk.lap(5);
            let enh = d / HEAD_SIZE;
            let mut qkv = pr.many(&[&e.wq, &e.wk, &e.wv], &h)?.into_iter();
            let (q, k, v) = (qkv.next().expect("q"), qkv.next().expect("k"), qkv.next().expect("v"));
            let old = st.eng.as_ref();
            let (n0, kt, vt) = match old {
                None => (0, None, None),
                Some(kv) => (kv.k.dim(2)?, Some(contig(&kv.k)?), Some(contig(&kv.v)?)),
            };
            let (kr, vr) = (kt.as_ref().map(ro).transpose()?, vt.as_ref().map(ro).transpose()?);
            let kc = append(kr.as_ref().map(|r| r.s()), n0, &k, enh, HEAD_SIZE);
            let vc = append(vr.as_ref().map(|r| r.s()), n0, &v, enh, HEAD_SIZE);
            let n = n0 + 1;
            let mut o = Vec::new();
            if !eng_off {
                o = vec![0f32; d];
                // the heads read independently (each its own keys, values and outputs): on up to `th` threads
                let op = o.as_mut_ptr() as usize;
                let (q, kc, vc) = (&q, &kc, &vc);
                par_heads(enh, th, |hh| {
                    let span = hh * n * HEAD_SIZE..(hh + 1) * n * HEAD_SIZE;
                    // SAFETY: head hh writes only o[hh 32..(hh+1) 32], inside o's d = enh 32 values
                    let out = unsafe { std::slice::from_raw_parts_mut((op as *mut f32).add(hh * HEAD_SIZE), HEAD_SIZE) };
                    attend(&q[hh * HEAD_SIZE..(hh + 1) * HEAD_SIZE], &kc[span.clone()], &vc[span], n, HEAD_SIZE, out);
                });
            }
            // v49 trace: the read's attention weights [heads, n], on the side from the same q and keys
            let eng_attn: Option<Vec<f32>> = (tracing && !eng_off).then(|| {
                let mut w = vec![0f32; enh * n];
                for (hh, row) in w.chunks_exact_mut(n).enumerate() {
                    attn_row(&q[hh * HEAD_SIZE..(hh + 1) * HEAD_SIZE], &kc[hh * n * HEAD_SIZE..(hh + 1) * n * HEAD_SIZE], n, row);
                }
                w
            });
            let valid = match old {
                None => vec![1],
                Some(kv) => kv.valid.iter().map(|x| x + 1).collect(),
            };
            drop((kr, vr));
            let mut store = Some(Kv {
                k: Tensor::from_vec(kc, (1, enh, n, HEAD_SIZE), &dev)?,
                v: Tensor::from_vec(vc, (1, enh, n, HEAD_SIZE), &dev)?,
                pad: None,
                valid,
                logp: None,
                ktail: None,
                reads: None,
                age: None,
            });
            if let Some(cap) = st.cap {
                crate::state::evict(&mut store, cap)?;
            }
            eng_new = store;
            clk.lap(4);
            if !eng_off {
                let r = pr.xw(&e.wo, &o)?;
                add_into(&mut x, &r);
                if let Some(w) = eng_attn {
                    lay.engram = Some((w, n, r));
                }
            }
        }

        // the sparse MLP: the k largest of h·W1 kept, relu (squared with --mlp-act relu2), ·W2
        let l = &blk.mlp;
        let h = rmsnorm(&x, ro(&l.norm)?.s());
        let mut u = pr.xw(&l.w1, &h)?;
        // v51 fast3 Pr: the cut units at -1e9 and k from the live units (v51::mlp)
        let kk = arch_f3::prune(m, i, &mut u)?.unwrap_or(cfg.kwta_k());
        let kth = if kk >= u.len() { None } else { Some(kth_largest(&u, kk)) };
        let mut a: Vec<f32> = match kth {
            None => u.iter().map(|&v| (v * 1.0).max(0f32)).collect(),
            Some(kth) => u.iter().map(|&v| (v * if v - kth >= 0.0 { 1.0 } else { 0.0 }).max(0f32)).collect(),
        };
        if ar.relu2 {
            a.iter_mut().for_each(|v| *v = *v * *v);
        }
        if let Some(c) = ctx.as_ref() {
            arch_f4::dendrites(m, i, c, &mut a)?; // f4ext P: the apical dendrites scale the units (v51::mlp)
        }
        let mut y2 = pr.xw(&l.w2, &a)?;
        arch_f4::dopamine(m, i, arch_f3::Mood::z(mood.as_ref()), &mut y2)?; // f4ext DA: the surprise scales the drive (v51::mlp, before v50's DA)
        if let Some(s) = sig.as_ref() {
            // v51 fast1 --neuromod DA: the MLP's output times its signal (v50_da)
            y2.iter_mut().for_each(|v| *v = *v * s[DA]);
        }
        add_into(&mut x, &y2);
        if let Some(r) = rec.as_deref_mut() {
            // the kWTA mask (as above), and the units as they reach W2
            lay.winners = match kth {
                None => vec![1.0; u.len()],
                Some(kth) => u.iter().map(|&v| if v - kth >= 0.0 { 1.0 } else { 0.0 }).collect(),
            };
            lay.hidden = a.clone();
            r.layers.push(std::mem::take(&mut lay));
        }

        if let (Some(r), Some(xi)) = (&relay, &x_in) {
            r.out(m, i, xi, &mut x); // TH: the residual gate
        }
        // R1 dense: the stream after block i is x + sum_j w_j x_j over the streams so far (this block's output last)
        if let Some(wt) = &bx.dense {
            let wv = vals(wt)?;
            streams.push(x.clone());
            let mut mix = x.clone();
            for (j, sj) in streams.iter().enumerate() {
                for (mv, &sv) in mix.iter_mut().zip(sj) {
                    *mv = *mv + sv * wv[j];
                }
            }
            x = mix;
        }
        if let Some(r) = relay.as_mut() {
            r.after(i, &x); // v51 fast2 TH: the relay reads this block's output
        }
        if i == 0 {
            // v51 fast1 --neuromod: the nucleus reads block 0's output (v50::nucleus_signals: 2 sigmoid(norm(x) W + b))
            sig = nucleus(m, &x)?;
            // merge_f: the chat app's midbrain signals of this token, as v51b_th_block(1) keeps them on the State path
            if let (Some(e), Some(s)) = (m.v51b(), sig.as_ref()) {
                if e.sig_on.get() && ar.neuromod > 0 {
                    e.sig.borrow_mut().nm = Some([s[NE], s[ACH], s[DA]]);
                }
            }
            // v51 fast3 M: the surprise and the mood (v51::after0, after the nucleus as forward_with_state runs them)
            mood = arch_f3::Mood::after0(m, st, &x, tok, &|v: &[f32]| pr.head(v))?;
        }
        clk.lap(5);
    }
    if stop.is_some() {
        // v51 fast4: the trunk's blocks move on; the token count and history wait for the rest of the token
        for (i, (l, ms)) in lifs.into_iter().zip(mems).enumerate() {
            st.lif[i] = Some(l);
            st.mem[i] = Some(ms);
        }
        if cfg.engram.is_some() {
            st.eng = eng_new;
        }
        arch_f3::commit(st, stps, mood, rhythm.as_ref().map(|r| r.a.as_slice()), d, &dev)?;
        return Ok(Some(Done::Trunk(Trunk { x, streams, v_first, full })));
    }
    let x = rmsnorm(&x, ro(&m.final_norm)?.s());
    let x = arch_f3::cb(m, x, tok)?; // v51 fast3 CB: the cerebellum's correction (v51b_cb; x itself without it)
    let logits = pr.head(&x)?;
    clk.lap(5);

    // everything computed: the State moves on (nothing above changed it)
    for (i, (l, ms)) in lifs.into_iter().zip(mems).enumerate() {
        st.lif[i] = Some(l);
        st.mem[i] = Some(ms);
    }
    if cfg.engram.is_some() {
        st.eng = eng_new;
    }
    arch_f3::commit(st, stps, mood, rhythm.as_ref().map(|r| r.a.as_slice()), d, &dev)?; // v51 fast3: S's (u, x); M's stream, surprise and mood; f4ext: Y's times
    st.seen[0] += 1;
    st.hist[0] = full[full.len().saturating_sub(HIST)..].to_vec();
    #[cfg(test)]
    RAN.with(|r| r.set(r.get() + 1));
    Ok(Some(Done::Logits(Tensor::from_vec(logits, cfg.vocab, &dev)?)))
}

/// Every tensor of a one-row State the fused step carries (the LIF membranes, spikes and traces, the fast weights and
/// momentum, the engram store; v51 fast3: the synapses' (u, x) and v51's carried signals), in a fixed order, and its counts
/// (tokens read, the history, the engram's valid slots):
/// for comparing two States bit for bit (tests, src/bin/talk_bench.rs).
pub fn state_tensors(st: &State) -> (Vec<Tensor>, Vec<usize>, Vec<Vec<u32>>) {
    let mut t = Vec::new();
    for l in st.lif.iter().flatten() {
        t.push(l.v.clone());
        t.push(l.s.clone());
        t.extend(l.a.iter().cloned());
    }
    for x in st.mem.iter().flatten() {
        t.push(x.f.clone());
        t.extend(x.s.iter().cloned());
        t.extend(x.prev.iter().cloned());
    }
    let mut counts = st.seen.clone();
    if let Some(e) = &st.eng {
        t.push(e.k.clone());
        t.push(e.v.clone());
        counts.extend(e.valid.iter().copied());
    }
    // v51 fast3: S's (u, x) per block, and v51's carried signals (M's stream, valid, surprise, mood; Y's times)
    for (u, x) in st.parts.iter().filter_map(|p| p.stp.as_ref()) {
        t.push(u.clone());
        t.push(x.clone());
    }
    let v = &st.v51;
    t.extend([&v.prev, &v.pvalid, &v.last_s, &v.mood, &v.rh].into_iter().flatten().cloned());
    (t, counts, st.hist.clone())
}


/// v51 fast1 --neuromod: the nucleus's signals for block 0's output x [d] as forward_with_state forms them
/// (v50::nucleus_signals on one row: the rms norm, gemm's one-row product, + b, candle's sigmoid, affine(2, 0));
/// None without the option.
fn nucleus(m: &Model, x: &[f32]) -> Result<Option<Vec<f32>>> {
    let Some((g, w, b)) = m.ext.as_ref().and_then(|e| e.v50.nucleus.as_ref()) else { return Ok(None) };
    let hn = rmsnorm(x, ro(g)?.s());
    let k = w.dim(1)?;
    let mut z = vec![0f32; k];
    gemv(&hn, ro(w)?.s(), k, &mut z);
    add_into(&mut z, ro(b)?.s());
    Ok(Some(z.into_iter().map(|v| sigmoid(v) * 2.0 + 0.0).collect()))
}

/// v51 fast1 --neuromod ACh's clamp to [0, 2] as candle's clamp computes it: maximum (v < 0 ? 0 : v), then minimum.
fn clamp02(v: f32) -> f32 {
    let v = if v < 0.0 { 0.0 } else { v };
    if v > 2.0 {
        2.0
    } else {
        v
    }
}

#[path = "arch_f2.rs"]
mod arch_f2;

// v52 (the chat app's live learning): the cerebellum's two additions, for the e-prop session step's tape
pub(crate) use arch_f3::cb_parts;

#[path = "arch_f3.rs"]
mod arch_f3;

// f4ext (runs/v52/f4ext/report.md): v51 P --dendrite, DA --dopamine and Y --rhythm
#[path = "arch_f4.rs"]
mod arch_f4;

#[cfg(test)]
thread_local! {
    /// Tests: fused block-options steps completed on this thread.
    pub(crate) static RAN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
#[path = "arch_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "trace_tests.rs"]
mod trace_tests;

#[cfg(test)]
#[path = "f32_tests.rs"]
mod f32_tests;
