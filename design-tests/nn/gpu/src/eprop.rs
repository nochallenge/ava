// TESTING / DESIGN ONLY — ava_gpu::eprop: v51 Part B, E (runs/v51/prereg_v51.md): learning while she reads and talks, on
// top of her trained brain (backprop builds her, e-prop learns live), and the reference it is judged against (online
// backprop: dynamic evaluation).
//
// A plastic session (`Model::plastic_begin`) makes chosen weights of chosen blocks plastic for B stream rows at once:
// each row r reads with W + dW_r, its own change, so B documents learn side by side in one batched pass (the State
// path, src/state.rs, where the hook `plastic_mm` sits on four matmuls of each block: the LIF input W_in, its read-out
// W_out, and the MLP's W1 and W2). With probes on, each plastic matmul's output also gets a zero variable added (its
// gradient is the error signal at that output), and the matmul's input is kept.
//
// E, e-prop (Bellec et al. 2020), one token at a time, no stored history. For a synapse W_ij from input i to unit j:
//   learning signal  L_j(t) = dE_t / d(out_j(t)): her output error at this word (cross-entropy of the word that came),
//                    carried back through the blocks above within this word only (the probes' gradients); nothing
//                    flows back in time
//   eligibility      the synapse's own trace of its recent activity:
//                    LIF input W_in (u = h W_in, the membrane v_t = beta v_{t-1} + (1 - beta) u_t - reset, the reset and
//                    the ALIF threshold detached as in training): e_ij(t) = beta_j e_ij(t-1) + h_i(t), one per synapse
//                    ([d, d] per row and block): exactly dv_j(t)/dW_ij / (1 - beta_j)
//                    W_out, W1, W2 (no state of their own): e_ij(t) = x_i(t), the present input (no memory to fade)
//   change           dW_ij -= lr · L_j(t) · e_ij(t), after each word
// Where the theory says it is exact (no path back in time but the unit's own membrane: the learned block is the last
// one and its memory read is silent) the summed e-prop changes equal backprop through time; eprop/tests.rs checks
// it on a tiny model. Elsewhere it drops the paths through the fast-weight memory and the other units (its known
// approximation).
//
// Dynamic evaluation (Krause et al. 2018), the strong reference: the same plastic weights, plain SGD on segments of S
// tokens: score the segment, then dW_r -= lr · sum over its tokens of x(t)^T L(t), where L now comes from backprop
// through time within the segment (the probes' gradients of the segment's summed loss: the membranes and memory
// inside the segment included), the state at its start detached.
//
// v51 E2 (runs/v51/prereg_v51.md, Amendment B-4): the changes fade back to her saved weights. Row r's changes have a
// half-life of h_r tokens (`Model::plastic_set_half_lives`; f64::INFINITY, the default, is E itself: nothing fades). A
// learning step over n tokens per row is then
//   dW_r <- f_r · dW_r - lr_r · step_r,   f_r = 2^(-n / h_r)
// (every element of dW_r, also those the step leaves at 0), so after T tokens a change made k tokens ago weighs
// 2^(-k / h_r). With h_r infinite, f_r = 1 and the old arithmetic runs unchanged (bit for bit). The traces belong to
// the membranes: `Model::plastic_new_documents` zeroes them (a new text starts from a fresh state) and keeps dW.

use crate::model::Model;
use crate::state::State;
use candle_core::backprop::GradStore;
use candle_core::{DType, Result, Tensor, Var};
use std::collections::HashMap;

/// The plastic matmuls of a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Site {
    /// the LIF's input W_in [d, d]
    Win,
    /// the spikes' read-out W_out [d, d]
    Wout,
    /// the MLP's W1 [d, hidden]
    W1,
    /// the MLP's W2 [hidden, d]
    W2,
}

impl Site {
    pub const ALL: [Site; 4] = [Site::Win, Site::Wout, Site::W1, Site::W2];

    /// The variable's name in block i.
    pub fn var_name(self, i: usize) -> String {
        match self {
            Site::Win => format!("b{i}.mem.win"),
            Site::Wout => format!("b{i}.mem.wout"),
            Site::W1 => format!("b{i}.mlp.w1"),
            Site::W2 => format!("b{i}.mlp.w2"),
        }
    }
}

/// How a learning step turns the error signals into changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// e-prop: W_in through its eligibility traces, the others by their present input (one token per row per step)
    EProp,
    /// online backprop over the forward's tokens: sum_t x(t)^T L(t)
    Backprop,
}

/// A plastic session (see the top of the file).
pub struct Plastic {
    pub rows: usize,
    /// per (block, site): the rows' changes [B, in, out]
    pub delta: HashMap<(usize, Site), Tensor>,
    /// e-prop: per learned block, W_in's eligibility traces [B, d, d]
    pub trace: HashMap<usize, Tensor>,
    /// add probes and keep inputs in the next forwards (a learning forward)
    pub probes: bool,
    rec: Vec<(usize, Site, Tensor, Var)>,
    /// v51 E2: per row, the changes' half-life in tokens (f64::INFINITY: they never fade)
    pub half: Vec<f64>,
}

impl Plastic {
    /// v51 fast4: the learning forwards' records not learned from yet (src/fast4/eprop.rs runs only on none).
    pub(crate) fn rec_len(&self) -> usize {
        self.rec.len()
    }
}

/// v51 E2: the fading factor of a learning step over `n` tokens for a half-life of `h` tokens: 2^(-n / h), exactly 1
/// for an infinite half-life.
pub fn fade(h: f64, n: usize) -> f32 {
    if h.is_infinite() {
        1.0
    } else {
        (-(n as f64) / h).exp2() as f32
    }
}

/// Whether a plastic session is on (the fused step refuses it).
pub fn active(m: &Model) -> bool {
    m.v51b().is_some_and(|e| e.plastic.borrow().is_some())
}

impl Model {
    /// The State path's matmul x · W of block i's `site` (x [b·t, in]): with a plastic session, plus each row's change
    /// x_r · dW_r and, with probes on, a zero probe (its input kept). Without a session: `int8::mm` itself.
    pub(crate) fn plastic_mm(&self, i: usize, site: Site, x: &Tensor, w: &Tensor, b: usize, t: usize) -> Result<Tensor> {
        let y = crate::int8::mm(x, w)?;
        let Some(e) = self.v51b() else { return Ok(y) };
        let mut g = e.plastic.borrow_mut();
        let Some(p) = g.as_mut() else { return Ok(y) };
        let Some(dw) = p.delta.get(&(i, site)) else { return Ok(y) };
        let (n, din) = x.dims2()?;
        if n != b * t || dw.dim(0)? != b {
            candle_core::bail!("plastic_mm: {n} rows for a session of {} rows x {t} tokens", dw.dim(0)?);
        }
        let x3 = x.reshape((b, t, din))?;
        let dout = dw.dim(2)?;
        let dy = if x.device().is_cpu() { x3.contiguous()?.apply_op2(dw, inplace::RowMm { b, t, din, dout })? } else { x3.matmul(dw)? }.reshape((n, ()))?;
        let mut y = y.add(&dy)?;
        if p.probes {
            let pr = Var::zeros(y.dims(), DType::F32, y.device())?;
            y = y.add(pr.as_tensor())?;
            p.rec.push((i, site, x.detach(), pr));
        }
        Ok(y)
    }

    /// Start a plastic session for `rows` rows: `sites` of `blocks` plastic, every change 0 (replacing any session).
    pub fn plastic_begin(&self, rows: usize, blocks: &[usize], sites: &[Site]) -> Result<()> {
        let Some(e) = self.v51b() else { candle_core::bail!("plastic_begin: a model with block options") };
        // merge (runs/v51/merge_b.md): v51's MLP options (--dendrite, --dopamine, --columns, --prune) and wave5's
        // --halting run their own MLP, which does not pass through `plastic_mm`: W1 and W2 would never learn there
        let ar = self.cfg.arch;
        if (ar.v51.mlp() || ar.mem.halting > 0) && sites.iter().any(|s| matches!(s, Site::W1 | Site::W2)) {
            candle_core::bail!("plastic_begin: W1 / W2 are not plastic under v51's MLP options or --halting (their MLP has no plastic hook)");
        }
        let dev = self.tok.device();
        let mut delta = HashMap::new();
        let mut trace = HashMap::new();
        for &i in blocks {
            let blk = self.blocks.get(i).ok_or_else(|| candle_core::Error::Msg(format!("plastic_begin: no block {i}")))?;
            for &s in sites {
                let w = match s {
                    Site::Win => &blk.mem.as_ref().expect("D block").win,
                    Site::Wout => &blk.mem.as_ref().expect("D block").wout,
                    Site::W1 => &blk.mlp.w1,
                    Site::W2 => &blk.mlp.w2,
                };
                let (din, dout) = w.dims2()?;
                delta.insert((i, s), Tensor::zeros((rows, din, dout), DType::F32, dev)?);
                if s == Site::Win {
                    trace.insert(i, Tensor::zeros((rows, din, dout), DType::F32, dev)?);
                }
            }
        }
        *e.plastic.borrow_mut() = Some(Plastic { rows, delta, trace, probes: false, rec: Vec::new(), half: vec![f64::INFINITY; rows] });
        Ok(())
    }

    /// v51 E2: each row's half-life of its changes, in tokens (f64::INFINITY: they never fade, the default).
    pub fn plastic_set_half_lives(&self, h: &[f64]) -> Result<()> {
        let Some(e) = self.v51b() else { candle_core::bail!("plastic_set_half_lives: no session") };
        let mut g = e.plastic.borrow_mut();
        let Some(p) = g.as_mut() else { candle_core::bail!("plastic_set_half_lives: no session") };
        if h.len() != p.rows || h.iter().any(|&x| !(x > 0.0)) {
            candle_core::bail!("plastic_set_half_lives: {} half-lives (each > 0) for {} rows", h.len(), p.rows);
        }
        p.half = h.to_vec();
        Ok(())
    }

    /// v51 E2: a new text starts in every row (from a fresh state): every eligibility trace back to 0, the changes kept.
    pub fn plastic_new_documents(&self) -> Result<()> {
        let Some(e) = self.v51b() else { return Ok(()) };
        let mut g = e.plastic.borrow_mut();
        let Some(p) = g.as_mut() else { return Ok(()) };
        for v in p.trace.values_mut() {
            *v = v.zeros_like()?;
        }
        Ok(())
    }

    /// v51 E2: every (block, site)'s changes of all rows [B, in, out] (copies), e.g. to save a stream's brain.
    pub fn plastic_deltas(&self) -> Result<Vec<((usize, Site), Tensor)>> {
        let Some(e) = self.v51b() else { return Ok(Vec::new()) };
        let g = e.plastic.borrow();
        let Some(p) = g.as_ref() else { return Ok(Vec::new()) };
        let mut keys: Vec<(usize, Site)> = p.delta.keys().copied().collect();
        keys.sort();
        keys.into_iter().map(|k| Ok((k, p.delta[&k].copy()?))).collect()
    }

    /// v51 E2: set a (block, site)'s changes of all rows [B, in, out] (a saved session resumed).
    pub fn plastic_set_delta(&self, i: usize, site: Site, d: &Tensor) -> Result<()> {
        let Some(e) = self.v51b() else { candle_core::bail!("plastic_set_delta: no session") };
        let mut g = e.plastic.borrow_mut();
        let Some(p) = g.as_mut() else { candle_core::bail!("plastic_set_delta: no session") };
        let Some(dw) = p.delta.get_mut(&(i, site)) else { candle_core::bail!("plastic_set_delta: b{i} {site:?} is not plastic") };
        if dw.dims() != d.dims() {
            candle_core::bail!("plastic_set_delta: b{i} {site:?} is {:?}, given {:?}", dw.dims(), d.dims());
        }
        *dw = d.to_dtype(DType::F32)?.to_device(dw.device())?.contiguous()?.copy()?;
        Ok(())
    }

    /// Probes on (the next forwards are learning forwards) or off (scoring only, the changes still apply).
    pub fn plastic_probes(&self, on: bool) {
        if let Some(e) = self.v51b() {
            if let Some(p) = e.plastic.borrow_mut().as_mut() {
                p.probes = on;
            }
        }
    }

    /// Clear row r's changes and traces (a new document starts in that row).
    pub fn plastic_reset_row(&self, r: usize) -> Result<()> {
        let Some(e) = self.v51b() else { return Ok(()) };
        let mut g = e.plastic.borrow_mut();
        let Some(p) = g.as_mut() else { return Ok(()) };
        let keep: Vec<f32> = (0..p.rows).map(|j| if j == r { 0.0 } else { 1.0 }).collect();
        let m = Tensor::from_vec(keep, (p.rows, 1, 1), self.tok.device())?;
        for v in p.delta.values_mut().chain(p.trace.values_mut()) {
            *v = v.broadcast_mul(&m)?;
        }
        Ok(())
    }
    /// The learning step after a learning forward and the backward of its summed loss (`grads`): each plastic
    /// matmul's error signal L (its probe's gradient) becomes each row's change, by `rule` (see the top of the file),
    /// times -lr. The records are then dropped. Returns the rms of the step's changes over every plastic weight.
    pub fn plastic_learn(&self, grads: &GradStore, lr: f64, rule: Rule) -> Result<f64> {
        let rows = self.v51b().and_then(|e| e.plastic.borrow().as_ref().map(|p| p.rows)).unwrap_or(0);
        self.plastic_learn_rows(grads, &vec![lr; rows], rule)
    }

    /// `plastic_learn` with a learning rate per row (`lrs`, one per row). On the CPU the changes and traces are
    /// updated in place (one pass over each, the same arithmetic as the tensor path, src/eprop.rs's `inplace`); on
    /// CUDA by tensor ops.
    pub fn plastic_learn_rows(&self, grads: &GradStore, lrs: &[f64], rule: Rule) -> Result<f64> {
        let Some(e) = self.v51b() else { candle_core::bail!("plastic_learn: no session") };
        let mut g = e.plastic.borrow_mut();
        let Some(p) = g.as_mut() else { candle_core::bail!("plastic_learn: no session") };
        let rec = std::mem::take(&mut p.rec);
        let b = p.rows;
        if lrs.len() != b {
            candle_core::bail!("plastic_learn: {} rates for {b} rows", lrs.len());
        }
        let m: Vec<f32> = lrs.iter().map(|lr| (-lr) as f32).collect();
        let half = p.half.clone();
        let (mut sq, mut cnt) = (0f64, 0usize);
        for (i, site, x, pr) in rec {
            let (n, din) = x.dims2()?;
            let t = n / b;
            // v51 E2: this step's fading factor per row (None: nothing fades, the old arithmetic)
            let f: Vec<f32> = half.iter().map(|&h| fade(h, t)).collect();
            let f = if f.iter().all(|&x| x == 1.0) { None } else { Some(f) };
            let l = match grads.get(pr.as_tensor()) {
                Some(l) => l.contiguous()?,
                None => Tensor::zeros(pr.dims(), DType::F32, x.device())?, // no path to the loss
            };
            let dout = l.dim(1)?;
            let x = x.contiguous()?;
            let eprop_win = rule == Rule::EProp && site == Site::Win;
            if eprop_win && t != 1 {
                candle_core::bail!("e-prop: one token per row per step ({t} here)");
            }
            let beta = if eprop_win { Some(crate::model::sigmoid(&self.blocks[i].mem.as_ref().expect("D block").beta)?.contiguous()?) } else { None };
            let dw = p.delta.get_mut(&(i, site)).expect("a recorded site is plastic");
            cnt += dw.elem_count();
            if x.device().is_cpu() {
                if let Some(beta) = &beta {
                    let tr = p.trace.get(&i).ok_or_else(|| candle_core::Error::Msg(format!("e-prop: no trace for block {i}")))?;
                    tr.inplace_op3(&beta, &x, &inplace::TraceStep { b, din, dout })?;
                    let op = inplace::TraceAdd { b, din, dout, m: m.clone(), f: f.clone(), sq: Default::default() };
                    dw.inplace_op3(tr, &l, &op)?;
                    sq += op.sq.get();
                } else {
                    let op = inplace::OuterAdd { b, t, din, dout, m: m.clone(), f: f.clone(), sq: Default::default() };
                    dw.inplace_op3(&x, &l, &op)?;
                    sq += op.sq.get();
                }
                continue;
            }
            let step = match &beta {
                Some(beta) => {
                    let tr = p.trace.get_mut(&i).ok_or_else(|| candle_core::Error::Msg(format!("e-prop: no trace for block {i}")))?;
                    // e(t) = beta ⊙ e(t-1) + h(t) (h over the inputs i, beta over the units j)
                    *tr = tr.broadcast_mul(&beta.reshape((1, 1, dout))?)?.broadcast_add(&x.reshape((b, din, 1))?)?;
                    tr.broadcast_mul(&l.reshape((b, 1, dout))?)?
                }
                None => x.reshape((b, t, din))?.transpose(1, 2)?.contiguous()?.matmul(&l.reshape((b, t, dout))?)?,
            };
            let ch = if lrs.iter().all(|&r| r == lrs[0]) {
                step.affine(-lrs[0], 0.0)?
            } else {
                step.broadcast_mul(&Tensor::from_vec(m.clone(), (b, 1, 1), step.device())?)?
            };
            sq += ch.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            *dw = match &f {
                // v51 E2: dW <- f · dW + change
                Some(f) => dw.broadcast_mul(&Tensor::from_vec(f.clone(), (b, 1, 1), dw.device())?)?.add(&ch)?,
                None => dw.add(&ch)?,
            };
        }
        Ok((sq / cnt.max(1) as f64).sqrt())
    }

    /// e-prop on a token without a learning signal (no target): the traces still take this token's input.
    pub fn plastic_trace_only(&self) -> Result<()> {
        let Some(e) = self.v51b() else { return Ok(()) };
        let mut g = e.plastic.borrow_mut();
        let Some(p) = g.as_mut() else { return Ok(()) };
        let rec = std::mem::take(&mut p.rec);
        let b = p.rows;
        for (i, site, x, pr) in rec {
            if site != Site::Win {
                continue;
            }
            let (_, din) = x.dims2()?;
            let dout = pr.dim(1)?;
            let beta = crate::model::sigmoid(&self.blocks[i].mem.as_ref().expect("D block").beta)?.contiguous()?;
            let x = x.contiguous()?;
            if let Some(tr) = p.trace.get_mut(&i) {
                if x.device().is_cpu() {
                    tr.inplace_op3(&beta, &x, &inplace::TraceStep { b, din, dout })?;
                } else {
                    *tr = tr.broadcast_mul(&beta.reshape((1, 1, dout))?)?.broadcast_add(&x.reshape((b, din, 1))?)?;
                }
            }
        }
        Ok(())
    }

    /// The sum over rows of block i's `site` changes [in, out] (None if not plastic): every row's change kept at once.
    pub fn plastic_delta_sum(&self, i: usize, site: Site) -> Result<Option<Tensor>> {
        let Some(e) = self.v51b() else { return Ok(None) };
        let g = e.plastic.borrow();
        let Some(p) = g.as_ref() else { return Ok(None) };
        p.delta.get(&(i, site)).map(|d| d.sum(0)).transpose()
    }

    /// Row r's change of block i's `site` [in, out] (None if not plastic).
    pub fn plastic_delta(&self, i: usize, site: Site, r: usize) -> Result<Option<Tensor>> {
        let Some(e) = self.v51b() else { return Ok(None) };
        let g = e.plastic.borrow();
        let Some(p) = g.as_ref() else { return Ok(None) };
        p.delta.get(&(i, site)).map(|d| d.get(r)).transpose()
    }

    /// End the session (its changes are dropped).
    pub fn plastic_end(&self) {
        if let Some(e) = self.v51b() {
            *e.plastic.borrow_mut() = None;
        }
    }
}

impl State {
    /// Every carried tensor detached (after a learning step: the probes' graph must not carry into the next word).
    pub fn detach_all(&mut self) {
        for l in self.lif.iter_mut().flatten() {
            l.v = l.v.detach();
            l.s = l.s.detach();
            l.a = l.a.as_ref().map(|a| a.detach());
        }
        for m in self.mem.iter_mut().flatten() {
            m.f = m.f.detach();
            m.s = m.s.as_ref().map(|s| s.detach());
            m.prev = m.prev.as_ref().map(|s| s.detach());
        }
        for kv in self.kv.iter_mut().flatten().chain(self.eng.as_mut()) {
            kv.k = kv.k.detach();
            kv.v = kv.v.detach();
            kv.logp = kv.logp.as_ref().map(|x| x.detach());
            kv.ktail = kv.ktail.as_ref().map(|x| x.detach());
        }
        self.fb = self.fb.as_ref().map(|f| f.detach());
        if let Some(s) = self.loop2.as_mut() {
            s.detach_all();
        }
    }
}

/// The CPU learning step in place (no new tensors per word): each op makes one pass over the changes (or traces),
/// split over the threads by (row, input) lines of `dout` values, with the tensor path's arithmetic in the same order.
mod inplace {
    use candle_core::{bail, CpuStorage, InplaceOp3, Layout, Result};
    use std::cell::Cell;

    fn f32s<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
        let CpuStorage::F32(v) = s else { bail!("eprop: f32 tensors only") };
        let Some((a, z)) = l.contiguous_offsets() else { bail!("eprop: contiguous tensors only") };
        Ok(&v[a..z])
    }

    fn f32s_mut<'a>(s: &'a mut CpuStorage, l: &Layout) -> Result<&'a mut [f32]> {
        let CpuStorage::F32(v) = s else { bail!("eprop: f32 tensors only") };
        let Some((a, z)) = l.contiguous_offsets() else { bail!("eprop: contiguous tensors only") };
        Ok(&mut v[a..z])
    }

    /// f(line index, line) over every line of `len` values, on the threads; the sum of what f returns.
    fn par_lines(data: &mut [f32], len: usize, f: &(dyn Fn(usize, &mut [f32]) -> f64 + Sync)) -> f64 {
        let lines = data.len() / len.max(1);
        let th = candle_core::utils::get_num_threads().clamp(1, 8).min(lines.max(1));
        let per = lines.div_ceil(th).max(1);
        std::thread::scope(|s| {
            let hs: Vec<_> = data
                .chunks_mut(per * len)
                .enumerate()
                .map(|(c, chunk)| {
                    s.spawn(move || {
                        let mut sq = 0f64;
                        for (k, line) in chunk.chunks_mut(len).enumerate() {
                            sq += f(c * per + k, line);
                        }
                        sq
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("eprop thread")).sum()
        })
    }

    /// dw[r] += m_r · sum_t x[r, t]^T l[r, t]  (dw [b, din, dout], x [b·t, din], l [b·t, dout]); keeps the sum of the
    /// squared changes. v51 E2: with `f`, dw[r] <- f_r · dw[r] + m_r · ... (every line fades, those with x = 0 too).
    pub(super) struct OuterAdd {
        pub b: usize,
        pub t: usize,
        pub din: usize,
        pub dout: usize,
        pub m: Vec<f32>,
        pub f: Option<Vec<f32>>,
        pub sq: Cell<f64>,
    }

    /// v51 E2: line <- line · f
    fn fade_line(line: &mut [f32], f: f32) {
        for w in line.iter_mut() {
            *w *= f;
        }
    }

    impl InplaceOp3 for OuterAdd {
        fn name(&self) -> &'static str {
            "eprop-outer-add"
        }
        fn cpu_fwd(&self, s1: &mut CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout, s3: &CpuStorage, l3: &Layout) -> Result<()> {
            let (b, t, din, dout) = (self.b, self.t, self.din, self.dout);
            let (dw, x, l) = (f32s_mut(s1, l1)?, f32s(s2, l2)?, f32s(s3, l3)?);
            if dw.len() != b * din * dout || x.len() != b * t * din || l.len() != b * t * dout {
                bail!("eprop-outer-add: sizes {} {} {} for b {b} t {t} din {din} dout {dout}", dw.len(), x.len(), l.len());
            }
            let mm = &self.m;
            let ff = self.f.as_deref();
            let sq = par_lines(dw, dout, &|g, line| {
                let (r, i) = (g / din, g % din);
                let m = mm[r];
                let mut s = 0f64;
                if let Some(ff) = ff {
                    // v51 E2: the line fades, then takes the step (the same step as below)
                    let f = ff[r];
                    if t == 1 {
                        let xi = x[r * din + i];
                        if xi == 0.0 {
                            fade_line(line, f);
                            return 0.0;
                        }
                        for (w, &lj) in line.iter_mut().zip(&l[r * dout..(r + 1) * dout]) {
                            let c = (xi * lj) * m;
                            *w = *w * f + c;
                            s += (c * c) as f64;
                        }
                        return s;
                    }
                    let mut acc = vec![0f32; dout];
                    let mut any = false;
                    for k in 0..t {
                        let xi = x[(r * t + k) * din + i];
                        if xi == 0.0 {
                            continue;
                        }
                        any = true;
                        for (a, &lj) in acc.iter_mut().zip(&l[(r * t + k) * dout..(r * t + k + 1) * dout]) {
                            *a += xi * lj;
                        }
                    }
                    if !any {
                        fade_line(line, f);
                        return 0.0;
                    }
                    for (w, a) in line.iter_mut().zip(&acc) {
                        let c = a * m;
                        *w = *w * f + c;
                        s += (c * c) as f64;
                    }
                    return s;
                }
                if t == 1 {
                    let xi = x[r * din + i];
                    if xi == 0.0 {
                        return 0.0;
                    }
                    for (w, &lj) in line.iter_mut().zip(&l[r * dout..(r + 1) * dout]) {
                        let c = (xi * lj) * m;
                        *w += c;
                        s += (c * c) as f64;
                    }
                    return s;
                }
                let mut acc = vec![0f32; dout];
                let mut any = false;
                for k in 0..t {
                    let xi = x[(r * t + k) * din + i];
                    if xi == 0.0 {
                        continue;
                    }
                    any = true;
                    for (a, &lj) in acc.iter_mut().zip(&l[(r * t + k) * dout..(r * t + k + 1) * dout]) {
                        *a += xi * lj;
                    }
                }
                if !any {
                    return 0.0;
                }
                for (w, a) in line.iter_mut().zip(&acc) {
                    let c = a * m;
                    *w += c;
                    s += (c * c) as f64;
                }
                s
            });
            self.sq.set(sq);
            Ok(())
        }
    }

    /// e-prop's eligibility traces: e[r, i, j] = e[r, i, j] · beta_j + x[r, i]  (e [b, din, dout], beta [dout], x [b, din]).
    pub(super) struct TraceStep {
        pub b: usize,
        pub din: usize,
        pub dout: usize,
    }

    impl InplaceOp3 for TraceStep {
        fn name(&self) -> &'static str {
            "eprop-trace-step"
        }
        fn cpu_fwd(&self, s1: &mut CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout, s3: &CpuStorage, l3: &Layout) -> Result<()> {
            let (b, din, dout) = (self.b, self.din, self.dout);
            let (e, beta, x) = (f32s_mut(s1, l1)?, f32s(s2, l2)?, f32s(s3, l3)?);
            if e.len() != b * din * dout || beta.len() != dout || x.len() != b * din {
                bail!("eprop-trace-step: sizes {} {} {} for b {b} din {din} dout {dout}", e.len(), beta.len(), x.len());
            }
            par_lines(e, dout, &|g, line| {
                let xi = x[g];
                for (v, &bj) in line.iter_mut().zip(beta) {
                    *v = *v * bj + xi;
                }
                0.0
            });
            Ok(())
        }
    }

    /// dw[r, i, j] += m_r · (e[r, i, j] · l[r, j])  (dw, e [b, din, dout], l [b, dout]); keeps the sum of the squared
    /// changes. v51 E2: with `f`, dw[r, i, j] <- f_r · dw[r, i, j] + m_r · (e · l).
    pub(super) struct TraceAdd {
        pub b: usize,
        pub din: usize,
        pub dout: usize,
        pub m: Vec<f32>,
        pub f: Option<Vec<f32>>,
        pub sq: Cell<f64>,
    }

    impl InplaceOp3 for TraceAdd {
        fn name(&self) -> &'static str {
            "eprop-trace-add"
        }
        fn cpu_fwd(&self, s1: &mut CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout, s3: &CpuStorage, l3: &Layout) -> Result<()> {
            let (b, din, dout) = (self.b, self.din, self.dout);
            let (dw, e, l) = (f32s_mut(s1, l1)?, f32s(s2, l2)?, f32s(s3, l3)?);
            if dw.len() != b * din * dout || e.len() != dw.len() || l.len() != b * dout {
                bail!("eprop-trace-add: sizes {} {} {} for b {b} din {din} dout {dout}", dw.len(), e.len(), l.len());
            }
            let mm = &self.m;
            let ff = self.f.as_deref();
            let sq = par_lines(dw, dout, &|g, line| {
                let r = g / din;
                let m = mm[r];
                let mut s = 0f64;
                if let Some(ff) = ff {
                    let f = ff[r];
                    for ((w, &ej), &lj) in line.iter_mut().zip(&e[g * dout..(g + 1) * dout]).zip(&l[r * dout..(r + 1) * dout]) {
                        let c = (ej * lj) * m;
                        *w = *w * f + c;
                        s += (c * c) as f64;
                    }
                    return s;
                }
                for ((w, &ej), &lj) in line.iter_mut().zip(&e[g * dout..(g + 1) * dout]).zip(&l[r * dout..(r + 1) * dout]) {
                    let c = (ej * lj) * m;
                    *w += c;
                    s += (c * c) as f64;
                }
                s
            });
            self.sq.set(sq);
            Ok(())
        }
    }

    /// The rows' own changes in the forward: y[r, k] = x[r, k] · dw[r]  (x [b, t, din], dw [b, din, dout] -> y [b, t,
    /// dout]) on the threads, skipping the inputs that are 0 (spikes, the kWTA's losers). Its backward gives x's
    /// gradient (RowMmT); dw is not a variable (the changes are learned by the rule, not by autograd).
    pub(super) struct RowMm {
        pub b: usize,
        pub t: usize,
        pub din: usize,
        pub dout: usize,
    }

    impl candle_core::CustomOp2 for RowMm {
        fn name(&self) -> &'static str {
            "eprop-row-mm"
        }
        fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, candle_core::Shape)> {
            let (b, t, din, dout) = (self.b, self.t, self.din, self.dout);
            let (x, w) = (f32s(s1, l1)?, f32s(s2, l2)?);
            if x.len() != b * t * din || w.len() != b * din * dout {
                bail!("eprop-row-mm: sizes {} {} for b {b} t {t} din {din} dout {dout}", x.len(), w.len());
            }
            let mut y = vec![0f32; b * t * dout];
            par_lines(&mut y, dout, &|g, line| {
                let r = g / t;
                let xr = &x[g * din..(g + 1) * din];
                for (i, &xi) in xr.iter().enumerate() {
                    if xi == 0.0 {
                        continue;
                    }
                    for (o, &wj) in line.iter_mut().zip(&w[(r * din + i) * dout..(r * din + i + 1) * dout]) {
                        *o += xi * wj;
                    }
                }
                0.0
            });
            Ok((CpuStorage::F32(y), candle_core::Shape::from((b, t, dout))))
        }
        fn bwd(&self, _x: &candle_core::Tensor, w: &candle_core::Tensor, _y: &candle_core::Tensor, gy: &candle_core::Tensor) -> Result<(Option<candle_core::Tensor>, Option<candle_core::Tensor>)> {
            let gx = gy.contiguous()?.apply_op2_no_bwd(&w.contiguous()?, &RowMmT { b: self.b, t: self.t, din: self.din, dout: self.dout })?;
            Ok((Some(gx), None))
        }
    }

    /// x's gradient of RowMm: gx[r, k, i] = sum_j gy[r, k, j] dw[r, i, j]  (gy [b, t, dout], dw [b, din, dout]).
    pub(super) struct RowMmT {
        pub b: usize,
        pub t: usize,
        pub din: usize,
        pub dout: usize,
    }

    impl candle_core::CustomOp2 for RowMmT {
        fn name(&self) -> &'static str {
            "eprop-row-mm-t"
        }
        fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, candle_core::Shape)> {
            let (b, t, din, dout) = (self.b, self.t, self.din, self.dout);
            let (gy, w) = (f32s(s1, l1)?, f32s(s2, l2)?);
            if gy.len() != b * t * dout || w.len() != b * din * dout {
                bail!("eprop-row-mm-t: sizes {} {} for b {b} t {t} din {din} dout {dout}", gy.len(), w.len());
            }
            let mut gx = vec![0f32; b * t * din];
            par_lines(&mut gx, din, &|g, line| {
                let r = g / t;
                let gr = &gy[g * dout..(g + 1) * dout];
                for (i, o) in line.iter_mut().enumerate() {
                    *o = gr.iter().zip(&w[(r * din + i) * dout..(r * din + i + 1) * dout]).map(|(a, c)| a * c).sum();
                }
                0.0
            });
            Ok((CpuStorage::F32(gx), candle_core::Shape::from((b, t, din))))
        }
    }
}

#[cfg(test)]
mod tests;
