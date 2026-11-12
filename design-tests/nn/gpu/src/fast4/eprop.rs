// TESTING / DESIGN ONLY — ava_gpu::fast4::eprop: one token of an e-prop session (v51 E2, one row) with the same bits as
// the State path's token (`forward_with_state`, `cross_entropy(..).affine(1)`, `loss.backward()`,
// `plastic_learn_rows(.., Rule::EProp)`, `State::detach_all`), without candle's per-op costs:
// * blocks 0..k (k the first plastic block) run the fused one-token step (f32, `Model::set_fast4`; its trunk mode,
//   src/model/fstep/arch.rs `trunk`), which is forward_with_state's values bit for bit;
// * blocks k.., the final norm, the head and the loss run on fast4's tape (src/fast4/tape.rs), op for op as
//   `block_state`, `lif_ref`, `mem_arch`, `delta_chunk` / `chunk_core` (c = 1 at one token), `plastic_mm` and
//   candle_nn's cross_entropy build them, so the graph, its values and its backward are candle's;
// * the probes' gradients then drive the learning step in place: per recorded site, in the record order, W_in's trace
//   e = e·beta_j + x_i and dW = dW·f + (e·l)·m, the other sites dW = dW·f + (x_i·l_j)·m (a line with x_i = 0 only
//   fades), eprop.rs's arithmetic, the lines split over the barrier pool's threads instead of new threads per op.
// Scope (else `blocker` says why and the caller runs the State path): one row, probes on, int8 off, no record
// pending; block options as the fused step takes them, with no v51 option but --letters, no --neuromod, no thalamus,
// no engram after block k, the silu output gate, and lif_ref's spikes (not --spikes binary without ALIF).
// Tests: fast4/eprop_tests.rs (logits, State, changes, traces bit for bit over 300 tokens).
// v52 (the chat app's live learning, src/fast4/live.rs): the cerebellum head and its online table (one row) on the
// tape (its correction and the table's share added to the final normed state, its code detached), the chat app's part
// signals, a target chosen from the logits (`Target`: she draws her next word there while she speaks) and the viewer's
// record (blocks k.. from the tape's values). Tests: fast4/live_tests.rs (against the State path, bit for bit).

use super::tape::{Bin, Tape, Un, T};
use crate::arch::{LifState, MemState, Spikes};
use crate::eprop::Site;
use crate::model::{Model, LIF_THRESHOLD, NORM_EPS};
use crate::state::{State, HIST};
use candle_core::{bail, CpuStorage, Layout, Result, Storage, Tensor};
use std::cell::Cell;
use std::collections::HashMap;

thread_local! {
    static SESSION: Cell<bool> = const { Cell::new(false) };
}

/// Whether this thread is inside the e-prop session step (the fused step's blockers then let a session through).
pub(crate) fn session_on() -> bool {
    SESSION.with(|s| s.get())
}

struct On;

impl On {
    fn new() -> On {
        SESSION.with(|s| s.set(true));
        On
    }
}

impl Drop for On {
    fn drop(&mut self) {
        SESSION.with(|s| s.set(false));
    }
}

/// A read guard on a contiguous CPU f32 tensor's values.
struct Rd<'a> {
    g: std::sync::RwLockReadGuard<'a, Storage>,
    a: usize,
    b: usize,
}

impl Rd<'_> {
    fn s(&self) -> &[f32] {
        match &*self.g {
            Storage::Cpu(CpuStorage::F32(v)) => &v[self.a..self.b],
            _ => unreachable!("checked in rd"),
        }
    }
}

fn rd(t: &Tensor) -> Result<Rd<'_>> {
    let (g, l) = t.storage_and_layout();
    let (a, b) = l.contiguous_offsets().ok_or_else(|| candle_core::Error::Msg("fast4 e-prop: a non-contiguous tensor".into()))?;
    if !matches!(&*g, Storage::Cpu(CpuStorage::F32(_))) {
        bail!("fast4 e-prop: CPU f32 tensors only");
    }
    Ok(Rd { g, a, b })
}

/// A tensor's values (contiguous copy) and dims.
fn own(t: &Tensor) -> Result<(Vec<f32>, Vec<usize>)> {
    let c = t.contiguous()?;
    Ok((super::with_f32(&c, |v| v.to_vec())?, t.dims().to_vec()))
}

/// The first plastic block (None: no session or no plastic site).
fn first_plastic(m: &Model) -> Option<usize> {
    let e = m.v51b()?;
    let g = e.plastic.borrow();
    g.as_ref()?.delta.keys().map(|k| k.0).min()
}

/// Why the e-prop session step does not take this token (None: it does). Call with the session flag on.
fn blocker(m: &Model) -> Option<&'static str> {
    let (cfg, ar) = (m.cfg, m.cfg.arch);
    if m.q8.borrow().is_some() {
        return Some("e-prop step: int8 is on (the plastic blocks run in f32)");
    }
    if !m.fast4() {
        return Some("e-prop step: `set_fast4` is off");
    }
    let Some(e) = m.v51b() else { return Some("e-prop step: no block options") };
    {
        let g = e.plastic.borrow();
        let Some(p) = g.as_ref() else { return Some("e-prop step: no plastic session") };
        if p.rows != 1 || !p.probes || p.rec_len() != 0 || p.half.len() != 1 {
            return Some("e-prop step: one row with its probes on and no record pending");
        }
        if p.delta.is_empty() {
            return Some("e-prop step: no plastic site");
        }
    }
    let mut o = ar.v51;
    o.letters = false;
    if o.any() || ar.neuromod > 0 || ar.dale > 0.0 || ar.loops > 1 {
        return Some("e-prop step: a v51 option other than --letters, or --neuromod, --dale, --loop");
    }
    // v52 (the chat app's live learning): the cerebellum head (its code detached, its two additions on the tape) and
    // its online table of one row run here, and so do the chat app's part signals (in this scope only the
    // cerebellum's: the thalamus and --neuromod stay out)
    if e.th.is_some() || e.cb_online.borrow().as_ref().is_some_and(|o| o.rows != 1) {
        return Some("e-prop step: the thalamus, or a cerebellum online table of more than one row");
    }
    if m.tracing() || !cfg.kind.is_ava() {
        return Some("e-prop step: tracing on, or not an Ava model");
    }
    let Some(k) = first_plastic(m) else { return Some("e-prop step: no plastic site") };
    let Some(ext) = m.ext.as_ref() else { return Some("e-prop step: no block options") };
    for i in k..m.blocks.len() {
        let (blk, bx) = (&m.blocks[i], &ext.blocks[i]);
        if blk.eng.is_some() || ext.mem.parts.get(i).is_some_and(|p| p.stp.is_some()) {
            return Some("e-prop step: an engram or synapse part after the first plastic block");
        }
        if blk.hebb.as_ref().is_none_or(|h| h.wg.is_none()) {
            return Some("e-prop step: the plastic blocks need the silu output gate");
        }
        if ar.spikes == Spikes::Binary && bx.alif.is_none() {
            return Some("e-prop step: --spikes binary without ALIF runs lif_infer");
        }
    }
    if let Some(why) = m.fused_blocker() {
        return Some(why);
    }
    None
}

/// One recorded probe: block, site, the input x (its values), the probe's node.
struct Rec {
    i: usize,
    site: Site,
    x: Vec<f32>,
    pr: T,
}

/// The block program on the tape: the lent changes and the records.
struct Prog<'t, 'a> {
    tp: &'t mut Tape<'a>,
    dws: HashMap<(usize, Site), &'a [f32]>,
    rec: Vec<Rec>,
}

/// The lent values of `t` (one of the tensors `rds` guards, by `idx`).
fn lent<'r>(rds: &'r [Rd<'_>], idx: &HashMap<usize, usize>, t: &Tensor) -> Result<&'r [f32]> {
    let j = *idx.get(&(t as *const Tensor as usize)).ok_or_else(|| candle_core::Error::Msg("fast4 e-prop: a weight not lent".into()))?;
    Ok(rds[j].s())
}

impl<'a> Prog<'_, 'a> {
    fn last(&self, x: T) -> usize {
        self.tp.shape(x).len() - 1
    }

    fn leaf_of(&mut self, t: &Tensor) -> Result<T> {
        let (v, sh) = own(t)?;
        Ok(self.tp.leaf(v, &sh))
    }

    /// `model::rmsnorm`: x · (mean(x²) + eps)^-1/2 · gain.
    fn rmsnorm(&mut self, x: T, gain: &Tensor) -> Result<T> {
        let g = self.leaf_of(gain)?;
        let l = self.last(x);
        let sq = self.tp.un(Un::Sqr, x)?;
        let ms = self.tp.mean_keepdim(sq, &[l])?;
        let e = self.tp.affine(ms, 1.0, NORM_EPS)?;
        let inv = self.tp.powf(e, -0.5)?;
        let a = self.tp.bbin(Bin::Mul, x, inv)?;
        self.tp.bbin(Bin::Mul, a, g)
    }

    /// `model::sigmoid`: (exp(-x) + 1)^-1.
    fn sigmoid(&mut self, x: T) -> Result<T> {
        let n = self.tp.un(Un::Neg, x)?;
        let e = self.tp.un(Un::Exp, n)?;
        let a = self.tp.affine(e, 1.0, 1.0)?;
        self.tp.un(Un::Recip, a)
    }

    /// `model::log_sigmoid`: -(relu(-x) + ln(exp(-|x|) + 1)).
    fn log_sigmoid(&mut self, x: T) -> Result<T> {
        let ab = self.tp.un(Un::Abs, x)?;
        let ng = self.tp.un(Un::Neg, ab)?;
        let ex = self.tp.un(Un::Exp, ng)?;
        let a1 = self.tp.affine(ex, 1.0, 1.0)?;
        let soft = self.tp.un(Un::Log, a1)?;
        let n2 = self.tp.un(Un::Neg, x)?;
        let r = self.tp.un(Un::Relu, n2)?;
        let s = self.tp.add(r, soft)?;
        self.tp.un(Un::Neg, s)
    }

    /// `model::l2_norm`: x · (sum(x²) + 1e-6)^-1/2 over the last dim.
    fn l2_norm(&mut self, x: T) -> Result<T> {
        let l = self.last(x);
        let sq = self.tp.un(Un::Sqr, x)?;
        let s = self.tp.sum_keepdim(sq, &[l])?;
        let a = self.tp.affine(s, 1.0, 1e-6)?;
        let p = self.tp.powf(a, -0.5)?;
        self.tp.bbin(Bin::Mul, x, p)
    }

    /// `model::head_norm`: y · (mean(y²) + eps)^-1/2 over the last dim.
    fn head_norm(&mut self, y: T) -> Result<T> {
        let l = self.last(y);
        let sq = self.tp.un(Un::Sqr, y)?;
        let ms = self.tp.mean_keepdim(sq, &[l])?;
        let a = self.tp.affine(ms, 1.0, NORM_EPS)?;
        let p = self.tp.powf(a, -0.5)?;
        self.tp.bbin(Bin::Mul, y, p)
    }

    /// arch.rs `soft_step`: u / (slope·|u| + 1).
    fn soft_step(&mut self, u: T, slope: f64) -> Result<T> {
        let ab = self.tp.un(Un::Abs, u)?;
        let a = self.tp.affine(ab, slope, 1.0)?;
        self.tp.div(u, a)
    }

    /// x [n, d] -> [1, nh, n, w] (mem_arch's `heads` / `per_head` at b = 1).
    fn per_head(&mut self, x: T, nh: usize, w: usize) -> Result<T> {
        let n = self.tp.shape(x)[0];
        let r = self.tp.reshape(x, &[1, n, nh, w])?;
        let t = self.tp.transpose(r, 1, 2)?;
        self.tp.contiguous(t)
    }

    /// A lent weight (its values `s`, the dims of `t`).
    fn w(&mut self, s: &'a [f32], t: &Tensor) -> T {
        self.tp.leaf_ext(s, t.dims())
    }

    /// `plastic_mm`: x · W, plus the row's change (RowMm) and a zero probe when the site is plastic.
    fn pmm(&mut self, i: usize, site: Site, x: T, w: T) -> Result<T> {
        let y = self.tp.matmul(x, w)?;
        let Some(&dw) = self.dws.get(&(i, site)) else { return Ok(y) };
        let (n, din) = (self.tp.shape(x)[0], self.tp.shape(x)[1]);
        let dout = self.tp.shape(w)[1];
        if n != 1 {
            bail!("fast4 e-prop: one token");
        }
        let x3 = self.tp.reshape(x, &[1, 1, din])?;
        let x3 = self.tp.contiguous(x3)?;
        let dy = self.tp.rowmm(x3, dw, 1, 1, din, dout)?;
        let dy = self.tp.reshape(dy, &[n, dout])?;
        let y = self.tp.add(y, dy)?;
        let pr = self.tp.var(&[n, dout]);
        let y = self.tp.add(y, pr)?;
        let xv = self.tp.vals(x);
        self.rec.push(Rec { i, site, x: xv, pr });
        Ok(y)
    }

    /// arch.rs `lif_ref` at one token (b = 1): the spikes [1, 1, d] and the new (v, s, a).
    #[allow(clippy::too_many_arguments)]
    fn lif(&mut self, cur: T, beta: T, theta: Option<T>, alif: Option<(T, T)>, spikes: Spikes, slope: f64, init: &LifState) -> Result<(T, T, T, Option<T>)> {
        let d = self.tp.shape(cur)[2];
        let th = match theta {
            Some(t) => t,
            None => self.tp.leaf(vec![LIF_THRESHOLD as f32; d], &[d]),
        };
        let v = self.leaf_of(&init.v)?;
        let sp = self.leaf_of(&init.s)?;
        let mut a = match (alif, init.a.as_ref()) {
            (Some(_), Some(x)) => Some(self.leaf_of(x)?),
            (Some(_), None) => Some(self.tp.leaf(vec![0f32; d], &[1, d])),
            (None, _) => None,
        };
        let tb = self.tp.bcast(th, &[1, d])?;
        let mut th_prev = self.tp.contiguous(tb)?;
        if let (Some((_, gain)), Some(a0)) = (alif, a) {
            let p = self.tp.bbin(Bin::Mul, a0, gain)?;
            th_prev = self.tp.add(th_prev, p)?;
        }
        let i_t = self.tp.narrow(cur, 1, 0, 1)?;
        let i_t = self.tp.squeeze(i_t, 1)?;
        let th_t = match (alif, a.as_mut()) {
            (Some((rho, gain)), Some(a)) => {
                let p1 = self.tp.bbin(Bin::Mul, *a, rho)?;
                let ab = self.tp.un(Un::Abs, sp)?;
                let ab = self.tp.detach(ab);
                let r1 = self.tp.affine(rho, -1.0, 1.0)?;
                let p2 = self.tp.bbin(Bin::Mul, ab, r1)?;
                *a = self.tp.add(p1, p2)?;
                let ag = self.tp.bbin(Bin::Mul, *a, gain)?;
                self.tp.bbin(Bin::Add, th, ag)?
            }
            _ => {
                let b = self.tp.bcast(th, &[1, d])?;
                self.tp.contiguous(b)?
            }
        };
        let vb = self.tp.bbin(Bin::Mul, v, beta)?;
        let va = self.tp.add(vb, i_t)?;
        let st = self.tp.mul(sp, th_prev)?;
        let st = self.tp.detach(st);
        let v = self.tp.sub(va, st)?;
        let ge = |x: &[f32]| -> Vec<f32> { x.iter().map(|&u| if u >= 0.0 { 1.0 } else { 0.0 }).collect() };
        let s = match spikes {
            Spikes::Binary => {
                let u = self.tp.sub(v, th_t)?;
                let sg = self.soft_step(u, slope)?;
                let hard = ge(&self.tp.vals(u));
                let hard = self.tp.leaf(hard, &[1, d]);
                let hs = self.tp.sub(hard, sg)?;
                let hs = self.tp.detach(hs);
                self.tp.add(sg, hs)?
            }
            Spikes::Ternary | Spikes::Int2 => {
                let mut soft: Option<T> = None;
                let mut hard: Option<Vec<f32>> = None;
                for k in 1..=spikes.levels() {
                    let kth = self.tp.affine(th_t, k as f64, 0.0)?;
                    let up = self.tp.sub(v, kth)?;
                    let nv = self.tp.un(Un::Neg, v)?;
                    let down = self.tp.sub(nv, kth)?;
                    let s_up = self.soft_step(up, slope)?;
                    let s_down = self.soft_step(down, slope)?;
                    let s1 = self.tp.sub(s_up, s_down)?;
                    let (hu, hd) = (ge(&self.tp.vals(up)), ge(&self.tp.vals(down)));
                    let h1: Vec<f32> = hu.iter().zip(&hd).map(|(p, q)| p - q).collect();
                    soft = Some(match soft {
                        None => s1,
                        Some(x) => self.tp.add(x, s1)?,
                    });
                    hard = Some(match hard {
                        None => h1,
                        Some(x) => x.iter().zip(&h1).map(|(p, q)| p + q).collect(),
                    });
                }
                let (soft, hard) = (soft.expect("levels >= 1"), hard.expect("levels >= 1"));
                let hard = self.tp.leaf(hard, &[1, d]);
                let hs = self.tp.sub(hard, soft)?;
                let hs = self.tp.detach(hs);
                self.tp.add(soft, hs)?
            }
        };
        let out = self.tp.unsqueeze(s, 1)?;
        let sp = self.tp.detach(s);
        Ok((out, v, sp, a))
    }

    /// candle's `cumsum` (rank > 1): transpose(dim, last), broadcast_matmul with the upper triangle, transpose back.
    fn cumsum(&mut self, x: T, dim: usize) -> Result<T> {
        let sh = self.tp.shape(x).to_vec();
        let n = sh[dim];
        let triu: Vec<f32> = (0..n * n).map(|q| if q % n >= q / n { 1.0 } else { 0.0 }).collect();
        let triu = self.tp.leaf(triu, &[n, n]);
        let last = sh.len() - 1;
        let t = self.tp.transpose(x, dim, last)?;
        let t = self.tp.broadcast_matmul(t, triu)?;
        self.tp.transpose(t, dim, last)
    }

    /// chunk_core's `pair2` (g > 1): sums over channels of x_i K_ij y_j for two right-hand keys.
    fn pair2(&mut self, x: T, kk: T, y1: T, y2: T) -> Result<(T, T)> {
        let xu = self.tp.unsqueeze(x, 3)?;
        let xk = self.tp.bbin(Bin::Mul, xu, kk)?;
        let u1 = self.tp.unsqueeze(y1, 2)?;
        let p1 = self.tp.bbin(Bin::Mul, xk, u1)?;
        let s1 = self.tp.sum(p1, &[4])?;
        let u2 = self.tp.unsqueeze(y2, 2)?;
        let p2 = self.tp.bbin(Bin::Mul, xk, u2)?;
        let s2 = self.tp.sum(p2, &[4])?;
        Ok((s1, s2))
    }

    /// x.narrow(1, r, 1).squeeze(1) (chunk_core's `at`).
    fn at(&mut self, x: T, r: usize) -> Result<T> {
        let n = self.tp.narrow(x, 1, r, 1)?;
        self.tp.squeeze(n, 1)
    }

    /// arch.rs `delta_chunk` + `chunk_core` (q, k, e, v [1, nh, t, hs]; log alpha and b [1, nh, t, g], g > 1): the
    /// read y [1, nh, t, hs] and the new fast weights and momentum.
    #[allow(clippy::too_many_arguments)]
    fn delta_chunk(&mut self, q: T, k: T, e: T, v: T, la: T, b: T, m: f64, init: &MemState) -> Result<(T, T, Option<T>)> {
        let sh = self.tp.shape(q).to_vec();
        let (bs, nh, t, hs) = (sh[0], sh[1], sh[2], sh[3]);
        let g = self.tp.shape(la)[3];
        if g == 1 {
            bail!("fast4 e-prop: per-channel gates only");
        }
        let bn = bs * nh;
        let mut c = 16;
        while t % c != 0 {
            c /= 2;
        }
        let nc = t / c;
        let q = self.tp.reshape(q, &[bn, nc, c, hs])?;
        let k = self.tp.reshape(k, &[bn, nc, c, hs])?;
        let e = self.tp.reshape(e, &[bn, nc, c, hs])?;
        let v = self.tp.reshape(v, &[bn, nc, c, hs])?;
        let la = self.tp.reshape(la, &[bn, nc, c, g])?;
        let bw = self.tp.reshape(b, &[bn, nc, c, g])?;
        let w = self.tp.bbin(Bin::Mul, k, bw)?;
        let lae = self.tp.un(Un::Exp, la)?;
        let bl = self.tp.mul(bw, lae)?;
        let ah = self.tp.bbin(Bin::Mul, e, bl)?;
        // chunk_core
        let cum = self.cumsum(la, 2)?;
        let cumex = self.tp.sub(cum, la)?;
        let mask: Vec<f32> = (0..c * c).map(|q| if q % c > q / c { -1e9 } else { 0.0 }).collect();
        let mask = self.tp.leaf(mask, &[c, c, 1]);
        let c3 = self.tp.unsqueeze(cum, 3)?;
        let c2 = self.tp.unsqueeze(cum, 2)?;
        let dd = self.tp.bbin(Bin::Sub, c3, c2)?;
        let dd = self.tp.bbin(Bin::Add, dd, mask)?;
        let dm = self.tp.un(Un::Exp, dd)?;
        let mom = m > 0.0;
        let kk = if mom {
            let mt: Vec<f32> = (0..c * c).map(|q| if q / c >= q % c { m.powi((q / c - q % c) as i32) as f32 } else { 0.0 }).collect();
            let mt = self.tp.leaf(mt, &[c, c]);
            let t1 = self.tp.transpose(dm, 3, 4)?;
            let t1 = self.tp.contiguous(t1)?;
            let t1 = self.tp.reshape(t1, &[bn * nc * c * g, c])?;
            let t1 = self.tp.matmul(t1, mt)?;
            let t1 = self.tp.reshape(t1, &[bn, nc, c, g, c])?;
            let t1 = self.tp.transpose(t1, 3, 4)?;
            self.tp.contiguous(t1)?
        } else {
            dm
        };
        let z = self.tp.leaf(vec![0f32; bn * nc * c * g], &[bn, nc, 1, c, g]);
        let kn = self.tp.narrow(kk, 2, 0, c - 1)?;
        let kp = self.tp.cat(&[z, kn], 2)?;
        let (qw, qa) = self.pair2(q, kk, w, ah)?;
        let (ke, ae) = self.pair2(e, kp, w, ah)?;
        // model::inv_unit_lower(ae, c)
        let level = |p: &mut Self, s: usize| -> Result<T> {
            let mv: Vec<f32> = (0..c * c)
                .map(|n| {
                    let (i, j) = (n / c, n % c);
                    if i / (2 * s) == j / (2 * s) && (i / s) % 2 == 1 && (j / s) % 2 == 0 { 1.0 } else { 0.0 }
                })
                .collect();
            let mv = p.tp.leaf(mv, &[c, c]);
            p.tp.bbin(Bin::Mul, ae, mv)
        };
        let l1 = level(self, 1)?;
        let ng = self.tp.un(Un::Neg, l1)?;
        let eye: Vec<f32> = (0..c * c).map(|n| if n / c == n % c { 1.0 } else { 0.0 }).collect();
        let eye = self.tp.leaf(eye, &[c, c]);
        let mut tinv = self.tp.bbin(Bin::Add, ng, eye)?;
        let mut s = 2;
        while s < c {
            let ls = level(self, s)?;
            let a1 = self.tp.matmul(tinv, ls)?;
            let a2 = self.tp.matmul(a1, tinv)?;
            tinv = self.tp.sub(tinv, a2)?;
            s *= 2;
        }
        let cx = self.tp.un(Un::Exp, cumex)?;
        let ec = self.tp.bbin(Bin::Mul, e, cx)?;
        let p1 = self.tp.matmul(tinv, ec)?;
        let kv = self.tp.matmul(ke, v)?;
        let r0 = self.tp.matmul(tinv, kv)?;
        let ce = self.tp.un(Un::Exp, cum)?;
        let qg = self.tp.bbin(Bin::Mul, q, ce)?;
        let qwv = self.tp.matmul(qw, v)?;
        let kl = self.tp.narrow(kk, 2, c - 1, 1)?;
        let klast = self.tp.squeeze(kl, 2)?;
        let wk = self.tp.bbin(Bin::Mul, w, klast)?;
        let wk = self.tp.transpose(wk, 2, 3)?;
        let wk = self.tp.contiguous(wk)?;
        let wlv = self.tp.matmul(wk, v)?;
        let ak = self.tp.bbin(Bin::Mul, ah, klast)?;
        let ak = self.tp.transpose(ak, 2, 3)?;
        let al = self.tp.contiguous(ak)?;
        let cn = self.tp.narrow(cum, 2, c - 1, 1)?;
        let cn = self.tp.un(Un::Exp, cn)?;
        let gc = self.tp.transpose(cn, 2, 3)?;
        let ms = if mom {
            let k0 = self.tp.narrow(kk, 3, 0, 1)?;
            let k0 = self.tp.squeeze(k0, 3)?;
            let gs = self.tp.affine(k0, m, 0.0)?;
            let p0 = self.tp.narrow(kp, 3, 0, 1)?;
            let p0 = self.tp.squeeze(p0, 3)?;
            let gsx = self.tp.affine(p0, m, 0.0)?;
            let mu: Vec<f32> = (0..c).map(|p| m.powi((c - 1 - p) as i32) as f32).collect();
            let mu = self.tp.leaf(mu, &[1, 1, c, 1]);
            let wmu = self.tp.bbin(Bin::Mul, w, mu)?;
            let wmu = self.tp.transpose(wmu, 2, 3)?;
            let wmu = self.tp.contiguous(wmu)?;
            let wm = self.tp.matmul(wmu, v)?;
            let amu = self.tp.bbin(Bin::Mul, ah, mu)?;
            let amu = self.tp.transpose(amu, 2, 3)?;
            let am = self.tp.contiguous(amu)?;
            let eg = self.tp.bbin(Bin::Mul, e, gsx)?;
            let p2 = self.tp.matmul(tinv, eg)?;
            let qs = self.tp.bbin(Bin::Mul, q, gs)?;
            let gn = self.tp.narrow(gs, 2, c - 1, 1)?;
            let gsc = self.tp.transpose(gn, 2, 3)?;
            Some((p2, qs, gsc, wm, am))
        } else {
            None
        };
        let mut f = self.leaf_of(&init.f)?;
        let mut sm = if mom {
            Some(match init.s.as_ref() {
                Some(x) => self.leaf_of(x)?,
                None => self.tp.leaf(vec![0f32; bn * hs * hs], &[bn, hs, hs]),
            })
        } else {
            None
        };
        let mc = m.powi(c as i32);
        let mut ys = Vec::with_capacity(nc);
        for r in 0..nc {
            let a = self.at(p1, r)?;
            let pf = self.tp.matmul(a, f)?;
            let a = self.at(r0, r)?;
            let mut rr = self.tp.add(pf, a)?;
            if let (Some((p2, ..)), Some(s)) = (ms, sm) {
                let a = self.at(p2, r)?;
                let ps = self.tp.matmul(a, s)?;
                rr = self.tp.add(rr, ps)?;
            }
            let a = self.at(qg, r)?;
            let qf = self.tp.matmul(a, f)?;
            let a = self.at(qwv, r)?;
            let y0 = self.tp.add(qf, a)?;
            let a = self.at(qa, r)?;
            let qr = self.tp.matmul(a, rr)?;
            let mut y = self.tp.sub(y0, qr)?;
            let a = self.at(gc, r)?;
            let fg = self.tp.bbin(Bin::Mul, f, a)?;
            let a = self.at(wlv, r)?;
            let f1 = self.tp.add(fg, a)?;
            let a = self.at(al, r)?;
            let ar = self.tp.matmul(a, rr)?;
            let mut fnew = self.tp.sub(f1, ar)?;
            if let (Some((_, qs, gsc, wm, am)), Some(s)) = (ms, sm.as_mut()) {
                let a = self.at(qs, r)?;
                let qss = self.tp.matmul(a, *s)?;
                y = self.tp.add(y, qss)?;
                let a = self.at(gsc, r)?;
                let sg = self.tp.bbin(Bin::Mul, *s, a)?;
                fnew = self.tp.add(fnew, sg)?;
                let sa = self.tp.affine(*s, mc, 0.0)?;
                let a = self.at(wm, r)?;
                let s1 = self.tp.add(sa, a)?;
                let a = self.at(am, r)?;
                let amr = self.tp.matmul(a, rr)?;
                *s = self.tp.sub(s1, amr)?;
            }
            ys.push(y);
            f = fnew;
        }
        let y = self.tp.cat(&ys, 1)?;
        let y = self.tp.reshape(y, &[bs, nh, t, hs])?;
        Ok((y, f, sm))
    }
}

/// What the plastic part hands back: logits, each block's new LIF (v, s, a) and memory (F, S) values, the target it
/// learned from with the records' (block, site, x, l) (None: no target, no backward), and (tracing) blocks k..'s
/// activity for the viewer.
struct Out {
    logits: Vec<f32>,
    lifs: Vec<(Vec<f32>, Vec<f32>, Option<Vec<f32>>)>,
    mems: Vec<(Vec<f32>, Option<Vec<f32>>)>,
    learn: Option<(u32, Vec<(usize, Site, Vec<f32>, Vec<f32>)>)>,
    layers: Vec<crate::model::fstep::arch::TokLayer>,
}

/// The target of a session step: given the token's logits [V], the word to learn to predict (None: nothing is learned
/// at this token). v52: the chat app draws her next word here, from these very logits, while she speaks.
pub(crate) type Target<'f> = &'f mut dyn FnMut(&[f32]) -> Result<Option<u32>>;

/// v52 (tracing): the nodes of one plastic block the viewer reads (spikes, the kWTA mask, the MLP's units, the memory's
/// read before its head norm, log alpha and b per channel, the new fast weights), and the old fast weights' values.
struct Seen {
    sp: T,
    mask: Vec<f32>,
    a: T,
    y: T,
    la: T,
    b: T,
    f: T,
    f_old: Vec<f32>,
}

/// v52 (tracing): one plastic block's activity as the fused step's record keeps it (src/model/fstep/arch.rs `TokLayer`):
/// spikes, winners, units, and per head |F' - alpha F| (alpha per row), |y|, mean alpha and mean b. On the side only.
fn seen_layer(tp: &Tape, s: Seen, nh: usize, hs: usize) -> crate::model::fstep::arch::TokLayer {
    let (y, la, b, f) = (tp.vals(s.y), tp.vals(s.la), tp.vals(s.b), tp.vals(s.f));
    let ex: Vec<f32> = la.iter().map(|l| l.exp()).collect();
    let mut lay = crate::model::fstep::arch::TokLayer { spikes: tp.vals(s.sp), winners: s.mask, hidden: tp.vals(s.a), ..Default::default() };
    for e in 0..nh {
        let sp = e * hs..(e + 1) * hs;
        lay.mem_read.push(y[sp.clone()].iter().map(|v| v * v).sum::<f32>().sqrt());
        lay.mem_keep.push(ex[sp.clone()].iter().sum::<f32>() / hs as f32);
        lay.mem_gate.push(b[sp.clone()].iter().sum::<f32>() / hs as f32);
        let mut s2 = 0f32;
        for a in 0..hs {
            let base = e * hs * hs + a * hs;
            for c in 0..hs {
                let dlt = f[base + c] - s.f_old[base + c] * ex[e * hs + a];
                s2 += dlt * dlt;
            }
        }
        lay.mem_write.push(s2.sqrt());
    }
    lay
}

/// Blocks k.., the head and the loss on the tape for the trunk's output, then the backward. The target comes from
/// `target` once the logits are known (the forward does not depend on it); without one there is no loss and no
/// backward. v52: with the cerebellum head, its correction (and its online table's share) is added to the final normed
/// state as `v51b_cb` adds them (its code reads that state detached, so nothing flows back through it).
#[allow(clippy::too_many_arguments)]
fn plastic_part(m: &Model, st: &State, trunk: &crate::model::fstep::arch::Trunk, k: usize, tok: u32, th: usize, tracing: bool, target: Target) -> Result<Out> {
    let (cfg, ar) = (m.cfg, m.cfg.arch);
    let ext = m.ext.as_ref().expect("checked in blocker");
    let e = m.v51b().expect("checked in blocker");
    let pg = e.plastic.borrow();
    let p = pg.as_ref().expect("checked in blocker");
    let (d, hs) = (cfg.d, ar.head_size);
    let nh = d / hs;
    // the lent tensors: the big products' weights, the head, the changes
    let mut held: Vec<&Tensor> = vec![&m.tok];
    for i in k..m.blocks.len() {
        let (blk, bx) = (&m.blocks[i], &ext.blocks[i]);
        let (mem, hb) = (blk.mem.as_ref().expect("checked"), blk.hebb.as_ref().expect("checked"));
        let c = bx.chan.as_ref().expect("checked");
        held.extend([&mem.win, &mem.wout, &hb.wq, &hb.wk, &hb.wv, &c.we, hb.wg.as_ref().expect("checked"), &hb.wo, &blk.mlp.w1, &blk.mlp.w2]);
    }
    let mut keys: Vec<(usize, Site)> = p.delta.keys().copied().collect();
    keys.sort();
    let deltas: Vec<&Tensor> = keys.iter().map(|k| &p.delta[k]).collect();
    let rds: Vec<Rd> = held.iter().chain(deltas.iter()).map(|t| rd(t)).collect::<Result<_>>()?;
    let idx: HashMap<usize, usize> = held.iter().enumerate().map(|(j, t)| (*t as *const Tensor as usize, j)).collect();
    let mut tp = Tape::new(th);
    let dws: HashMap<(usize, Site), &[f32]> = keys.iter().enumerate().map(|(j, k)| (*k, rds[held.len() + j].s())).collect();
    let mut pg_ = Prog { tp: &mut tp, dws, rec: Vec::new() };
    let pr = &mut pg_;
    let rds_ref: &[Rd] = &rds;
    // the trunk's stream, the dense streams so far and block 0's values
    let mut x = pr.tp.leaf(trunk.x.clone(), &[1, d]);
    let mut streams: Vec<T> = trunk.streams.iter().map(|s| pr.tp.leaf(s.clone(), &[1, d])).collect();
    let mut v0: Option<T> = trunk.v_first.as_ref().map(|v| pr.tp.leaf(v.clone(), &[1, nh, 1, hs]));
    let mut lifs = Vec::new();
    let mut mems = Vec::new();
    let mut seen: Vec<Seen> = Vec::new(); // v52: tracing only
    for i in k..m.blocks.len() {
        let (blk, bx) = (&m.blocks[i], &ext.blocks[i]);
        let mem = blk.mem.as_ref().expect("checked");
        // the spiking memory
        let h = pr.rmsnorm(x, &mem.norm)?;
        let w = pr.w(lent(rds_ref, &idx, &mem.win)?, &mem.win);
        let cur = pr.pmm(i, Site::Win, h, w)?;
        let mut cur = pr.tp.reshape(cur, &[1, 1, d])?;
        let bp = pr.leaf_of(&mem.beta)?;
        let beta = pr.sigmoid(bp)?;
        if ar.lif_scale {
            let b1 = pr.tp.affine(beta, -1.0, 1.0)?;
            cur = pr.tp.bbin(Bin::Mul, cur, b1)?;
        }
        let theta = match &bx.theta {
            Some(t) => Some(pr.leaf_of(t)?),
            None => None,
        };
        let alif = match &bx.alif {
            Some((r, g)) => {
                let rl = pr.leaf_of(r)?;
                let rho = pr.sigmoid(rl)?;
                Some((rho, pr.leaf_of(g)?))
            }
            None => None,
        };
        let init = st.lif[i].as_ref().expect("checked in state_ok");
        let (s, v, sp, a) = pr.lif(cur, beta, theta, alif, ar.spikes, ar.slope as f64, init)?;
        lifs.push((v, sp, a));
        let seen_sp = sp;
        let s2 = pr.tp.reshape(s, &[1, d])?;
        let w = pr.w(lent(rds_ref, &idx, &mem.wout)?, &mem.wout);
        let y = pr.pmm(i, Site::Wout, s2, w)?;
        x = pr.tp.add(x, y)?;
        // the fast-weight memory (mem_arch: per-channel gates, the delta rule with momentum)
        let hb = blk.hebb.as_ref().expect("checked");
        let h = pr.rmsnorm(x, &hb.norm)?;
        let wq = pr.w(lent(rds_ref, &idx, &hb.wq)?, &hb.wq);
        let ql = pr.tp.matmul(h, wq)?;
        let qh = pr.per_head(ql, nh, hs)?;
        let q = pr.l2_norm(qh)?;
        let wk = pr.w(lent(rds_ref, &idx, &hb.wk)?, &hb.wk);
        let kl = pr.tp.matmul(h, wk)?;
        let kraw = pr.per_head(kl, nh, hs)?;
        let kq = pr.l2_norm(kraw)?;
        let wv = pr.w(lent(rds_ref, &idx, &hb.wv)?, &hb.wv);
        let vl = pr.tp.matmul(h, wv)?;
        let mut v = pr.per_head(vl, nh, hs)?;
        if ar.value_residual {
            match (v0, &bx.vres) {
                (None, _) => v0 = Some(v),
                (Some(v0t), Some(pv)) => {
                    let pl = pr.leaf_of(pv)?;
                    let p0 = pr.tp.narrow(pl, 0, 0, 1)?;
                    let p0 = pr.tp.affine(p0, 1.0, 1.0)?;
                    let vb = pr.tp.bbin(Bin::Mul, v, p0)?;
                    let p1 = pr.tp.narrow(pl, 0, 1, 1)?;
                    let v0b = pr.tp.bbin(Bin::Mul, v0t, p1)?;
                    v = pr.tp.add(vb, v0b)?;
                }
                (Some(_), None) => {}
            }
        }
        let g = hb.gates.as_ref().expect("checked");
        let c = bx.chan.as_ref().expect("checked");
        let n = 1;
        let wa = pr.leaf_of(&g.wa)?;
        let za = pr.tp.matmul(h, wa)?;
        let ba = pr.leaf_of(&g.ba)?;
        let za = pr.tp.bbin(Bin::Add, za, ba)?;
        let wb = pr.leaf_of(&g.wb)?;
        let zb = pr.tp.matmul(h, wb)?;
        let bb = pr.leaf_of(&g.bb)?;
        let zb = pr.tp.bbin(Bin::Add, zb, bb)?;
        let zar = pr.tp.reshape(za, &[n, nh, 1])?;
        let off = pr.leaf_of(&c.off)?;
        let offr = pr.tp.reshape(off, &[1, nh, hs])?;
        let za = pr.tp.bbin(Bin::Add, zar, offr)?;
        let wa1 = pr.leaf_of(&c.wa1)?;
        let l1 = pr.tp.matmul(h, wa1)?;
        let wa2 = pr.leaf_of(&c.wa2)?;
        let l2 = pr.tp.matmul(l1, wa2)?;
        let l2 = pr.tp.reshape(l2, &[n, nh, hs])?;
        let za = pr.tp.add(za, l2)?;
        let zbr = pr.tp.reshape(zb, &[n, nh, 1])?;
        let wb1 = pr.leaf_of(&c.wb1)?;
        let l1 = pr.tp.matmul(h, wb1)?;
        let wb2 = pr.leaf_of(&c.wb2)?;
        let l2 = pr.tp.matmul(l1, wb2)?;
        let l2 = pr.tp.reshape(l2, &[n, nh, hs])?;
        let zb = pr.tp.bbin(Bin::Add, zbr, l2)?;
        let zaf = pr.tp.reshape(za, &[n, d])?;
        let la = pr.log_sigmoid(zaf)?;
        let zbf = pr.tp.reshape(zb, &[n, d])?;
        let sb = pr.sigmoid(zbf)?;
        let beta_w = pr.tp.affine(sb, 2.0, 0.0)?;
        let we = pr.w(lent(rds_ref, &idx, &c.we)?, &c.we);
        let el = pr.tp.matmul(h, we)?;
        let eh = pr.per_head(el, nh, hs)?;
        let ek = pr.tp.add(kraw, eh)?;
        let ek = pr.l2_norm(ek)?;
        let la = pr.per_head(la, nh, hs)?;
        let beta_w = pr.per_head(beta_w, nh, hs)?;
        let ms0 = st.mem[i].as_ref().expect("checked in state_ok");
        let (y, fnew, snew) = pr.delta_chunk(q, kq, ek, v, la, beta_w, ar.mem_momentum as f64, ms0)?;
        let seen_mem = (y, la, beta_w, fnew, if tracing { own(&ms0.f)?.0 } else { Vec::new() });
        mems.push((fnew, snew));
        let y = pr.head_norm(y)?;
        let y = pr.tp.transpose(y, 1, 2)?;
        let y = pr.tp.contiguous(y)?;
        let y = pr.tp.reshape(y, &[n, d])?;
        let wg = pr.w(lent(rds_ref, &idx, hb.wg.as_ref().expect("checked"))?, hb.wg.as_ref().expect("checked"));
        let gq = pr.tp.matmul(h, wg)?;
        let gs = pr.tp.un(Un::Silu, gq)?;
        let yg = pr.tp.mul(y, gs)?;
        let wo = pr.w(lent(rds_ref, &idx, &hb.wo)?, &hb.wo);
        let y = pr.tp.matmul(yg, wo)?;
        x = pr.tp.add(x, y)?;
        // the sparse MLP
        let l = &blk.mlp;
        let h = pr.rmsnorm(x, &l.norm)?;
        let w1 = pr.w(lent(rds_ref, &idx, &l.w1)?, &l.w1);
        let u = pr.pmm(i, Site::W1, h, w1)?;
        let uv = pr.tp.vals(u);
        let kw = cfg.kwta_k();
        let mask: Vec<f32> = if kw >= uv.len() {
            vec![1.0; uv.len()]
        } else {
            let kth = crate::model::fstep::kth_largest(&uv, kw);
            uv.iter().map(|&v| if v - kth >= 0.0 { 1.0 } else { 0.0 }).collect()
        };
        let mask_seen = if tracing { mask.clone() } else { Vec::new() };
        let wm = pr.tp.leaf(mask, &[1, uv.len()]);
        let a = pr.tp.mul(u, wm)?;
        let mut a = pr.tp.un(Un::Relu, a)?;
        if ar.relu2 {
            a = pr.tp.un(Un::Sqr, a)?;
        }
        let w2 = pr.w(lent(rds_ref, &idx, &l.w2)?, &l.w2);
        if tracing {
            let (y, la, b, f, f_old) = seen_mem;
            seen.push(Seen { sp: seen_sp, mask: mask_seen, a, y, la, b, f, f_old });
        }
        let y2 = pr.pmm(i, Site::W2, a, w2)?;
        x = pr.tp.add(x, y2)?;
        // R1 dense: x + sum_j w_j x_j over the streams so far (this block's output last)
        if let Some(wt) = &bx.dense {
            let wl = pr.leaf_of(wt)?;
            streams.push(x);
            let mut mix = x;
            for j in 0..streams.len() {
                let wj = pr.tp.narrow(wl, 0, j, 1)?;
                let sw = pr.tp.bbin(Bin::Mul, streams[j], wj)?;
                mix = pr.tp.add(mix, sw)?;
            }
            x = mix;
        }
    }
    // the final norm, the tied head, candle_nn's cross_entropy (log_softmax, gather, sum, -1/1) times 1
    let mut xo = pr.rmsnorm(x, &m.final_norm)?;
    // v52: the cerebellum's correction, then its online table's share (v51b_cb: out = xf + corr, out + add; its code
    // is computed on the side from xf's values, detached, and keeps the code for `cb_learn`)
    if let Some((corr, online)) = crate::model::fstep::arch::cb_parts(m, &pr.tp.vals(xo), tok, lent(rds_ref, &idx, &m.tok)?)? {
        let c = pr.tp.leaf(corr, &[1, d]);
        xo = pr.tp.add(xo, c)?;
        if let Some(add) = online {
            let a = pr.tp.leaf(add, &[1, d]);
            xo = pr.tp.add(xo, a)?;
        }
    }
    let tk = pr.w(lent(rds_ref, &idx, &m.tok)?, &m.tok);
    let tt = pr.tp.t(tk)?;
    let lg = pr.tp.matmul(xo, tt)?;
    let logits = pr.tp.vals(lg);
    let layers = seen.into_iter().map(|s| seen_layer(pr.tp, s, nh, hs)).collect();
    let lifs: Vec<_> = lifs.into_iter().map(|(v, s, a)| (pr.tp.vals(v), pr.tp.vals(s), a.map(|a| pr.tp.vals(a)))).collect();
    let mems: Vec<_> = mems.into_iter().map(|(f, s)| (pr.tp.vals(f), s.map(|s| pr.tp.vals(s)))).collect();
    // the target, now that the logits are known (v52: the chat app draws her next word here while she speaks)
    let Some(next) = target(&logits)? else { return Ok(Out { logits, lifs, mems, learn: None, layers }) };
    if (next as usize) >= cfg.vocab {
        bail!("fast4 e-prop: target {next} past the vocabulary");
    }
    let mx = pr.tp.max_keepdim(lg, 1)?;
    let diff = pr.tp.bbin(Bin::Sub, lg, mx)?;
    let ex = pr.tp.un(Un::Exp, diff)?;
    let se = pr.tp.sum_keepdim(ex, &[1])?;
    let lse = pr.tp.un(Un::Log, se)?;
    let ls = pr.tp.bbin(Bin::Sub, diff, lse)?;
    let gt = pr.tp.gather_last(ls, &[next as usize])?;
    let sa = pr.tp.sum(gt, &[0, 1])?;
    let l1 = pr.tp.affine(sa, -1.0, 0.0)?;
    let loss = pr.tp.affine(l1, 1.0, 0.0)?;
    let grads = pr.tp.backward(loss)?;
    let rec = std::mem::take(&mut pr.rec);
    let recs = rec
        .into_iter()
        .map(|r| {
            let l = grads[r.pr].clone().unwrap_or_else(|| vec![0f32; pr.tp.shape(r.pr).iter().product()]);
            (r.i, r.site, r.x, l)
        })
        .collect();
    Ok(Out { logits, lifs, mems, learn: Some((next, recs)), layers })
}

#[derive(Clone, Copy)]
struct Ptr(*mut f32);
// SAFETY: the threads write disjoint lines through it
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

/// f(line g, line) over the `din` lines of `dout` values of a [1, din, dout] tensor on the threads; the per-line f64
/// results summed as eprop.rs's `par_lines` sums them (by its chunks, each from 0 in line order, then the chunks').
fn lines(data: &mut [f32], dout: usize, th: usize, f: &(dyn Fn(usize, &mut [f32]) -> f64 + Sync)) -> f64 {
    let n = data.len() / dout.max(1);
    let mut per_line = vec![0f64; n];
    let (dp, sp) = (Ptr(data.as_mut_ptr()), per_line.as_mut_ptr() as usize);
    super::fmm::par_ranges(n, 1, th, n * dout, |a, b| {
        let dp = dp;
        for g in a..b {
            // SAFETY: line g (and its result slot) belongs to this range only
            let line = unsafe { std::slice::from_raw_parts_mut(dp.0.add(g * dout), dout) };
            let s = f(g, line);
            unsafe { *(sp as *mut f64).add(g) = s };
        }
    });
    let tr = candle_core::utils::get_num_threads().clamp(1, 8).min(n.max(1));
    let per = n.div_ceil(tr).max(1);
    per_line
        .chunks(per)
        .map(|c| {
            let mut s = 0f64;
            for &x in c {
                s += x;
            }
            s
        })
        .sum()
}

fn f32s_mut<'s>(s: &'s mut CpuStorage, l: &Layout) -> Result<&'s mut [f32]> {
    let CpuStorage::F32(v) = s else { bail!("fast4 e-prop: f32 tensors only") };
    let Some((a, z)) = l.contiguous_offsets() else { bail!("fast4 e-prop: contiguous tensors only") };
    Ok(&mut v[a..z])
}

fn f32s<'s>(s: &'s CpuStorage, l: &Layout) -> Result<&'s [f32]> {
    let CpuStorage::F32(v) = s else { bail!("fast4 e-prop: f32 tensors only") };
    let Some((a, z)) = l.contiguous_offsets() else { bail!("fast4 e-prop: contiguous tensors only") };
    Ok(&v[a..z])
}

/// W_in's trace in place: e[i, j] = e[i, j] · beta_j + x_i (eprop.rs `TraceStep`).
struct TraceStep<'x> {
    x: &'x [f32],
    beta: &'x [f32],
    th: usize,
}

impl candle_core::InplaceOp1 for TraceStep<'_> {
    fn name(&self) -> &'static str {
        "fast4-eprop-trace-step"
    }
    fn cpu_fwd(&self, s: &mut CpuStorage, l: &Layout) -> Result<()> {
        let e = f32s_mut(s, l)?;
        let (din, dout) = (self.x.len(), self.beta.len());
        if e.len() != din * dout {
            bail!("fast4 trace step: sizes");
        }
        lines(e, dout, self.th, &|g, line| {
            let xi = self.x[g];
            for (v, &bj) in line.iter_mut().zip(self.beta) {
                *v = *v * bj + xi;
            }
            0.0
        });
        Ok(())
    }
}

/// W_in's change in place from its trace: dW = dW · f + (e · l) · m, or dW += (e · l) · m (eprop.rs `TraceAdd`).
struct TraceAdd<'x> {
    l: &'x [f32],
    m: f32,
    f: Option<f32>,
    th: usize,
    sq: Cell<f64>,
}

impl candle_core::InplaceOp2 for TraceAdd<'_> {
    fn name(&self) -> &'static str {
        "fast4-eprop-trace-add"
    }
    fn cpu_fwd(&self, s1: &mut CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<()> {
        let (dw, e) = (f32s_mut(s1, l1)?, f32s(s2, l2)?);
        let dout = self.l.len();
        if dw.len() != e.len() || dw.len() % dout != 0 {
            bail!("fast4 trace add: sizes");
        }
        let (m, l) = (self.m, self.l);
        let sq = match self.f {
            Some(f) => lines(dw, dout, self.th, &|g, line| {
                let mut s = 0f64;
                for ((w, &ej), &lj) in line.iter_mut().zip(&e[g * dout..(g + 1) * dout]).zip(l) {
                    let c = (ej * lj) * m;
                    *w = *w * f + c;
                    s += (c * c) as f64;
                }
                s
            }),
            None => lines(dw, dout, self.th, &|g, line| {
                let mut s = 0f64;
                for ((w, &ej), &lj) in line.iter_mut().zip(&e[g * dout..(g + 1) * dout]).zip(l) {
                    let c = (ej * lj) * m;
                    *w += c;
                    s += (c * c) as f64;
                }
                s
            }),
        };
        self.sq.set(sq);
        Ok(())
    }
}

/// Another site's change in place: dW[i] = dW[i] · f + (x_i · l) · m (a line with x_i = 0 only fades), or without
/// fading dW[i] += (x_i · l) · m (eprop.rs `OuterAdd` at one token).
struct OuterAdd<'x> {
    x: &'x [f32],
    l: &'x [f32],
    m: f32,
    f: Option<f32>,
    th: usize,
    sq: Cell<f64>,
}

impl candle_core::InplaceOp1 for OuterAdd<'_> {
    fn name(&self) -> &'static str {
        "fast4-eprop-outer-add"
    }
    fn cpu_fwd(&self, s: &mut CpuStorage, l: &Layout) -> Result<()> {
        let dw = f32s_mut(s, l)?;
        let (din, dout) = (self.x.len(), self.l.len());
        if dw.len() != din * dout {
            bail!("fast4 outer add: sizes");
        }
        let (m, lv, x) = (self.m, self.l, self.x);
        let sq = match self.f {
            Some(f) => lines(dw, dout, self.th, &|g, line| {
                let xi = x[g];
                if xi == 0.0 {
                    for w in line.iter_mut() {
                        *w *= f;
                    }
                    return 0.0;
                }
                let mut s = 0f64;
                for (w, &lj) in line.iter_mut().zip(lv) {
                    let c = (xi * lj) * m;
                    *w = *w * f + c;
                    s += (c * c) as f64;
                }
                s
            }),
            None => lines(dw, dout, self.th, &|g, line| {
                let xi = x[g];
                if xi == 0.0 {
                    return 0.0;
                }
                let mut s = 0f64;
                for (w, &lj) in line.iter_mut().zip(lv) {
                    let c = (xi * lj) * m;
                    *w += c;
                    s += (c * c) as f64;
                }
                s
            }),
        };
        self.sq.set(sq);
        Ok(())
    }
}

/// v52: what one e-prop session step gives back: the logits [V], the learning step's rms change and the target it
/// learned from (both None when the target was None: nothing learned, nothing faded).
pub(crate) struct Stepped {
    pub(crate) logits: Vec<f32>,
    pub(crate) rms: Option<f64>,
    pub(crate) next: Option<u32>,
}

/// One token of a one-row e-prop session, fused (see `Model::fast4_eprop_step`), with the target from `target` once
/// the logits are known (v52: the chat app draws her next word there while she speaks; None: the token is read and
/// the State moves on, nothing is learned and nothing fades) and (v52) with `rec` the viewer's record of the token
/// (blocks 0..k from the fused trunk, blocks k.. from the tape's values, on the side). None: the fused step does not
/// take this token (nothing changed). A target past the vocabulary is an error after the trunk has moved on.
pub(crate) fn eprop_live(m: &Model, st: &mut State, tok: u32, lr: f64, mut rec: Option<&mut crate::model::fstep::arch::TokTrace>, target: Target) -> Result<Option<Stepped>> {
    let _on = On::new();
    if blocker(m).is_some() || (tok as usize) >= m.cfg.vocab {
        return Ok(None);
    }
    let k = first_plastic(m).expect("checked in blocker");
    let th = candle_core::utils::get_num_threads();
    if let Some(r) = rec.as_deref_mut() {
        *r = Default::default();
    }
    let Some(trunk) = crate::model::fstep::arch::trunk(m, st, tok, k, rec.as_deref_mut())? else { return Ok(None) };
    let out = plastic_part(m, st, &trunk, k, tok, th, rec.is_some(), target)?;
    if let Some(r) = rec {
        r.layers.extend(out.layers);
    }
    // the State moves on (the State path's blocks k.. after detach_all), the token is counted
    let (d, dev) = (m.cfg.d, m.tok.device().clone());
    let (hs, nh) = (m.cfg.arch.head_size, d / m.cfg.arch.head_size);
    for (j, ((v, s, a), (f, sm))) in out.lifs.into_iter().zip(out.mems).enumerate() {
        let i = k + j;
        st.lif[i] = Some(LifState { v: Tensor::from_vec(v, (1, d), &dev)?, s: Tensor::from_vec(s, (1, d), &dev)?, a: a.map(|a| Tensor::from_vec(a, (1, d), &dev)).transpose()? });
        st.mem[i] = Some(MemState { f: Tensor::from_vec(f, (nh, hs, hs), &dev)?, s: sm.map(|s| Tensor::from_vec(s, (nh, hs, hs), &dev)).transpose()?, prev: None });
    }
    st.seen[0] += 1;
    st.hist[0] = trunk.full[trunk.full.len().saturating_sub(HIST)..].to_vec();
    let Some((next, recs)) = out.learn else { return Ok(Some(Stepped { logits: out.logits, rms: None, next: None })) };
    // the learning step, in the record order
    let e = m.v51b().expect("checked in blocker");
    let mut pg = e.plastic.borrow_mut();
    let p = pg.as_mut().expect("checked in blocker");
    let mrate = (-lr) as f32;
    let f = crate::eprop::fade(p.half[0], 1);
    let f = if f == 1.0 { None } else { Some(f) };
    let (mut sq, mut cnt) = (0f64, 0usize);
    for (i, site, x, l) in recs {
        let dw = p.delta.get(&(i, site)).expect("a recorded site is plastic");
        cnt += dw.elem_count();
        if site == Site::Win {
            let beta: Vec<f32> = own(&m.blocks[i].mem.as_ref().expect("D block").beta)?.0.iter().map(|&b| crate::model::fstep::sigmoid(b)).collect();
            let tr = p.trace.get(&i).ok_or_else(|| candle_core::Error::Msg(format!("e-prop: no trace for block {i}")))?;
            tr.inplace_op1(&TraceStep { x: &x, beta: &beta, th })?;
            let op = TraceAdd { l: &l, m: mrate, f, th, sq: Cell::new(0.0) };
            dw.inplace_op2(tr, &op)?;
            sq += op.sq.get();
        } else {
            let op = OuterAdd { x: &x, l: &l, m: mrate, f, th, sq: Cell::new(0.0) };
            dw.inplace_op1(&op)?;
            sq += op.sq.get();
        }
    }
    let rms = (sq / cnt.max(1) as f64).sqrt();
    Ok(Some(Stepped { logits: out.logits, rms: Some(rms), next: Some(next) }))
}

impl Model {
    /// One token of a one-row e-prop session (reading `tok`, learning to predict `next` at rate `lr`): the State path's
    /// `forward_with_state` + `cross_entropy(..).affine(1)` + `backward` + `plastic_learn_rows(.., &[lr], EProp)` +
    /// `detach_all`, bit for bit (logits, State, changes, traces), fused. Returns the logits [1, V] and the learning
    /// step's rms change (as `plastic_learn_rows` returns it); None: this token takes the State path (nothing changed).
    pub fn fast4_eprop_step(&self, st: &mut State, tok: u32, next: u32, lr: f64) -> Result<Option<(Tensor, f64)>> {
        {
            let _on = On::new();
            if blocker(self).is_some() || (tok as usize) >= self.cfg.vocab {
                return Ok(None);
            }
        }
        if (next as usize) >= self.cfg.vocab {
            bail!("fast4_eprop_step: target {next} past the vocabulary");
        }
        let Some(s) = eprop_live(self, st, tok, lr, None, &mut |_| Ok(Some(next)))? else { return Ok(None) };
        let rms = s.rms.expect("a target was given");
        Ok(Some((Tensor::from_vec(s.logits, (1, self.cfg.vocab), self.tok.device())?, rms)))
    }

    /// Why `fast4_eprop_step` would not take a token now (None: it would), for tests and the stream's report.
    pub fn fast4_eprop_blocker(&self) -> Option<&'static str> {
        let _on = On::new();
        blocker(self)
    }
}
