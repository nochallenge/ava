// TESTING / DESIGN ONLY — ava_gpu::v50: round 1 of "a more brain-like Ava" (runs/v50/prereg_v50.md). Three block
// options for model D, each off by default (off: every number is what it was, src/v50_golden.rs checks it bit for bit):
//
// --dale FRAC   Dale's law on the MLP's outgoing weights W2 [hidden, d]: hidden unit i is excitatory (E) when
//               i < round(FRAC·hidden), else inhibitory (I), fixed by index; its outgoing weights are
//               w_ij = sign_i · softplus(v_ij), v learned (the variable b{i}.mlp.w2s, in W2's place: the same count).
//               Init: today's random draws w0 for W2 are made (same order, so every other weight gets today's init)
//               and v = softplus^-1(|w0|), so the starting |W2| is exactly today's (only the signs follow E/I; no E/I
//               balancing). Weight decay: AdamW's own decay would pull v toward 0 (softplus(0) = 0.69, far bigger
//               weights), so v is not in AdamW's decayed set; instead `Model::dale_decay(lr·wd)` after each optimizer
//               step shrinks the effective weights as AdamW shrinks every other matrix, w <- w·(1 - lr·wd), to first
//               order in lr·wd: v <- v - lr·wd·softplus(v)/sigmoid(v). The MLP's int8 copy (talking) is not used for
//               W2 under --dale (the effective weight is a new tensor; that matmul stays f32).
// --loop N      recurrent processing with top-down feedback (N = 2 in v50): pass 1 is the normal forward; its last
//               block's output, normed with the final norm's gain, through a learned projection W [d, d] (loop.wo,
//               zero at init, decayed as every .wo), is added to block 0's input; pass 2 runs the same blocks (shared
//               weights) and gives the logits. Each pass has its own memory and LIF states (the State carries pass 2's
//               in State::loop2), so the parallel training forward and the one-token stream compute the same thing.
//               About N times the compute.
// --neuromod 3  a small learned nucleus reads block 0's output per token (norm with its own gain nm.norm [d], then a
//               linear read-out nm.wn [d, 3] + nm.bn [3], both 0 at init) and broadcasts three signals
//               g = 2·sigmoid(.) in (0, 2), exactly 1 at init, to every later block (1..L-1), per token:
//                 NE  (noradrenaline): a gain on the LIF's input drive (the current into the membrane, after
//                     --lif-scale; the same as a threshold scale 1/g for the spike decision, but the reset stays theta)
//                 ACh (acetylcholine): a scale on the fast-weight memory's write strength b (kept in [0, 2], the delta
//                     rule's stable range, as wave5's B8 does)
//                 DA  (dopamine): a scale on the MLP's output (value / drive)
//               Per token, no state: the stream computes it from block 0's output of the same token.
//
// All three run on the block-options path (Model::forward_arch, candle's autograd) only: the hand-written fused WIN
// training step (src/model/fwin.rs, step_ok) and the fused CPU talking step (src/model/fstep/arch.rs, blocker) refuse
// them, so v50 runs every arm, the base included, with AVA_WIN_FUSED=0 (one code path for all arms).
// v51 fused N (src/fwin_nm.rs): --neuromod 3 now trains on the fused WIN step too (--dale and --loop still do not); the
// ops above are unchanged, the fused step does the same maths with its own backward.

use crate::arch::Arch;
use crate::model::{norm, sigmoid, Config, Model};
use candle_core::{DType, Result, Tensor, Var};
use candle_nn::{Init, VarBuilder};
use std::cell::{Cell, RefCell};

/// The three signals' columns.
pub const NE: usize = 0;
pub const ACH: usize = 1;
pub const DA: usize = 2;

/// Whether any v50 option is on.
pub fn any(a: &Arch) -> bool {
    a.dale > 0.0 || a.loops > 1 || a.neuromod > 0
}

/// The v50 options that are on, as " key=value ..." (Arch::signature).
pub fn signature(a: &Arch) -> String {
    let mut s = String::new();
    if a.dale > 0.0 {
        s += &format!(" dale={}", a.dale);
        if a.dale_proj {
            s += " dale_form=proj";
        }
    }
    if a.loops > 1 {
        s += &format!(" loop={}", a.loops);
    }
    if a.neuromod > 0 {
        s += &format!(" neuromod={}", a.neuromod);
    }
    s
}

/// The flags that set them (Arch::flags, arch.txt).
pub fn flags(a: &Arch) -> Vec<String> {
    let mut f = Vec::new();
    if a.dale > 0.0 {
        f.push(format!("--dale {}", a.dale));
        if a.dale_proj {
            f.push("--dale-form proj".into());
        }
    }
    if a.loops > 1 {
        f.push(format!("--loop {}", a.loops));
    }
    if a.neuromod > 0 {
        f.push(format!("--neuromod {}", a.neuromod));
    }
    f
}

/// The options checked against the rest of the config (Arch::validate; model D only is checked there).
pub fn validate(a: &Arch, c: &Config) -> std::result::Result<(), String> {
    if !(a.dale >= 0.0 && a.dale <= 1.0) {
        return Err(format!("--dale {}: the excitatory share, in (0, 1] (0 = off)", a.dale));
    }
    if a.dale_proj && a.dale <= 0.0 {
        return Err("--dale-form proj: needs --dale FRAC".into());
    }
    if a.dale > 0.0 && a.mem.halting > 0 {
        return Err("--dale: --halting's MLP reads W2 directly".into());
    }
    if a.loops == 0 || a.loops > 4 {
        return Err(format!("--loop {}: 1 to 4 passes (1 = off)", a.loops));
    }
    if a.loops > 1 && (a.feedback || a.mem.modulate) {
        return Err("--loop: not with --feedback (its own top-down loop) or --modulate (a shadow pass of its own)".into());
    }
    if a.neuromod != 0 && a.neuromod != 3 {
        return Err(format!("--neuromod {}: 3 (NE, ACh, DA) or 0 (off)", a.neuromod));
    }
    if a.neuromod > 0 && c.layers < 2 {
        return Err("--neuromod: block 0 drives the later blocks, so at least 2 blocks".into());
    }
    Ok(())
}

/// The extra weights the options make (Config::weight_counts; --dale's v takes W2's place and adds none).
pub fn extra_weights(cfg: &Config) -> usize {
    let (a, d) = (cfg.arch, cfg.d);
    let mut n = 0;
    if a.loops > 1 {
        n += d * d;
    }
    if a.neuromod > 0 {
        n += d + d * a.neuromod + a.neuromod;
    }
    n
}

/// The options' parts of a model (ModelExt::v50).
pub(crate) struct V50Ext {
    /// --dale: each hidden unit's sign [hidden, 1] (+1 E, -1 I), a constant
    pub(crate) dale_sign: Option<Tensor>,
    /// --dale: the effective W2 of every block, kept while `Model::set_dale_cache(true)` (talking: no weight changes)
    pub(crate) dale_cache: RefCell<Option<Vec<Tensor>>>,
    pub(crate) dale_cache_on: Cell<bool>,
    /// --loop: the top-down projection [d, d] (0 at init)
    pub(crate) loop_w: Option<Tensor>,
    /// --neuromod: the nucleus (norm gain [d], read-out [d, k], bias [k]; read-out and bias 0 at init)
    pub(crate) nucleus: Option<(Tensor, Tensor, Tensor)>,
    /// --neuromod: this pass's signals [n, k], set after block 0 (None before it, so block 0 is never modulated)
    pub(crate) sig: RefCell<Option<Tensor>>,
    /// --loop: on during a pass before the last (the trace and record hooks stay quiet)
    pub(crate) quiet: Cell<bool>,
}

/// Make the options' new tensors (called last in arch::build_ext: all start at constants, no random draws, so every
/// other weight gets today's init).
pub(crate) fn build(cfg: &Config, vb: &VarBuilder) -> Result<V50Ext> {
    let (a, d, hid) = (cfg.arch, cfg.d, cfg.hidden());
    let dev = vb.device();
    let dale_sign = if a.dale > 0.0 {
        let ne = dale_e(a.dale, hid);
        let s: Vec<f32> = (0..hid).map(|i| if i < ne { 1.0 } else { -1.0 }).collect();
        Some(Tensor::from_vec(s, (hid, 1), dev)?)
    } else {
        None
    };
    let loop_w = if a.loops > 1 { Some(vb.get_with_hints((d, d), "loop.wo", Init::Const(0.0))?) } else { None };
    let nucleus = if a.neuromod > 0 {
        let k = a.neuromod;
        let vn = vb.pp("nm");
        Some((vn.get_with_hints(d, "norm", Init::Const(1.0))?, vn.get_with_hints((d, k), "wn", Init::Const(0.0))?, vn.get_with_hints(k, "bn", Init::Const(0.0))?))
    } else {
        None
    };
    Ok(V50Ext { dale_sign, dale_cache: RefCell::new(None), dale_cache_on: Cell::new(false), loop_w, nucleus, sig: RefCell::new(None), quiet: Cell::new(false) })
}

/// --dale: how many of the `hid` hidden units are excitatory.
pub fn dale_e(frac: f32, hid: usize) -> usize {
    ((frac as f64 * hid as f64).round() as usize).min(hid)
}

/// The MLP's W2 variable [hid, d] of one block (Model::new): today's `w2`, or under --dale its v (`w2s`) after
/// drawing today's random numbers for W2 (dropped once v = softplus^-1(|w0|) is set from them).
pub(crate) fn mlp_w2_param(cfg: &Config, vl: &VarBuilder, rs: f64) -> Result<Tensor> {
    let (hid, d) = (cfg.hidden(), cfg.d);
    let randn = Init::Randn { mean: 0.0, stdev: rs };
    if cfg.arch.dale <= 0.0 {
        return vl.get_with_hints((hid, d), "w2", randn);
    }
    let w0 = randn.var((hid, d), DType::F32, vl.device())?;
    let fresh = !vl.contains_tensor("w2s");
    let v = vl.get_with_hints((hid, d), "w2s", Init::Const(0.0))?;
    if fresh {
        let y: Vec<f32> = w0.as_tensor().flatten_all()?.to_vec1()?;
        let v0: Vec<f32> = if cfg.arch.dale_proj {
            // D2: v = |w0| · (E 0.5, I e/(hid-e) · 0.5): each output's expected E and I drive cancel, mean square kept
            let (ne, gi) = proj_balance(cfg.arch.dale, hid);
            y.iter().enumerate().map(|(k, &w)| w.abs() * if k / d < ne { gi.0 } else { gi.1 }).collect()
        } else {
            y.iter().map(|&w| softplus_inv(w.abs() as f64) as f32).collect()
        };
        Var::from_tensor(&v)?.set(&Tensor::from_vec(v0, (hid, d), vl.device())?)?;
    }
    Ok(v)
}

/// D2 (--dale-form proj): the number of E units and the start scales (E, I) of |w0|: E·sE = I·sI (E/I balance of the
/// expected drive) and (E·sE² + I·sI²)/hid = 1 (today's mean square). For 0.8: (0.5, 2.0). All E: (1, 1).
pub fn proj_balance(frac: f32, hid: usize) -> (usize, (f32, f32)) {
    let ne = dale_e(frac, hid);
    let ni = hid - ne;
    if ni == 0 || ne == 0 {
        return (ne, (1.0, 1.0));
    }
    let r = ne as f64 / ni as f64; // sI = r·sE
    let se = (hid as f64 / (ne as f64 + ni as f64 * r * r)).sqrt();
    (ne, (se as f32, (se * r) as f32))
}

/// softplus^-1(y) = ln(e^y - 1) for y > 0 (y below 1e-12 is taken as 1e-12: v = -27.6).
pub fn softplus_inv(y: f64) -> f64 {
    let y = y.max(1e-12);
    if y > 30.0 {
        y + (-(-y).exp()).ln_1p()
    } else {
        y.exp_m1().ln()
    }
}

/// softplus(x) = max(x, 0) + ln(1 + e^-|x|), in candle ops (autograd: d/dx = sigmoid(x), exact up to rounding).
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    x.relu()?.add(&x.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)
}

/// --dale: the effective weights sign ⊙ softplus(v), v [hid, d], sign [hid, 1].
pub fn dale_w2(v: &Tensor, sign: &Tensor) -> Result<Tensor> {
    softplus(v)?.broadcast_mul(sign)
}

/// D2 (--dale-form proj): the effective weights sign ⊙ v, v >= 0 (kept by Model::dale_decay's projection).
pub fn dale_w2_proj(v: &Tensor, sign: &Tensor) -> Result<Tensor> {
    v.broadcast_mul(sign)
}

/// The effective W2 for either form.
pub fn dale_w2_form(v: &Tensor, sign: &Tensor, proj: bool) -> Result<Tensor> {
    if proj {
        dale_w2_proj(v, sign)
    } else {
        dale_w2(v, sign)
    }
}

/// --loop: pass 2's input: the input of pass 1 plus the top-down signal norm(top) · W (top = pass 1's last block's
/// output, normed with the final norm's gain).
pub fn loop_add(x_in: &Tensor, top: &Tensor, gain: &Tensor, w: &Tensor) -> Result<Tensor> {
    x_in.add(&norm(top, gain)?.matmul(w)?)
}

/// --neuromod: the signals 2·sigmoid(norm(x, g) · W + b) [n, k] for block 0's output x [n, d].
pub fn nucleus_signals(x: &Tensor, g: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    Ok(sigmoid(&norm(x, g)?.matmul(w)?.broadcast_add(b)?)?.affine(2.0, 0.0)?)
}

/// --neuromod ACh: the write strength b [B, nh, T, g] times the signal [B·T, 1], kept in [0, 2].
pub fn ach_scale(beta: &Tensor, s: &Tensor) -> Result<Tensor> {
    let (b, _, t, _) = beta.dims4()?;
    beta.broadcast_mul(&s.reshape((b, 1, t, 1))?)?.clamp(0f32, 2f32)
}

impl Model {
    fn v50(&self) -> Option<&V50Ext> {
        self.ext.as_ref().map(|e| &e.v50)
    }

    /// --dale: keep the effective W2 of every block from now on (talking, where the weights do not change), or drop
    /// it (off, the default: every forward recomputes it from v). Nothing without --dale.
    pub fn set_dale_cache(&self, on: bool) {
        if let Some(x) = self.v50() {
            x.dale_cache_on.set(on);
            *x.dale_cache.borrow_mut() = None;
        }
    }

    /// The MLP's outgoing weights of block `i` as used: today's W2 (the variable itself), or under --dale
    /// sign ⊙ softplus(v).
    pub(crate) fn mlp_w2(&self, i: usize) -> Result<Tensor> {
        let l = &self.blocks[i].mlp;
        let Some((x, sign)) = self.v50().and_then(|x| x.dale_sign.as_ref().map(|s| (x, s))) else { return Ok(l.w2.clone()) };
        let proj = self.cfg.arch.dale_proj;
        if !x.dale_cache_on.get() {
            return dale_w2_form(&l.w2, sign, proj);
        }
        if x.dale_cache.borrow().is_none() {
            let ws: Vec<Tensor> = self.blocks.iter().map(|b| dale_w2_form(&b.mlp.w2.detach(), sign, proj)).collect::<Result<_>>()?;
            *x.dale_cache.borrow_mut() = Some(ws);
        }
        Ok(x.dale_cache.borrow().as_ref().expect("filled above")[i].clone())
    }

    /// --dale, after an optimizer step at learning rate lr with weight decay wd (c = lr·wd): every block's effective
    /// W2 shrinks by (1 - c) as AdamW's decayed matrices do, to first order in c: v <- v - c·softplus(v)/sigmoid(v)
    /// (softplus/sigmoid -> 1 + e^v/2 for v < -8, where the direct form loses precision). Nothing without --dale.
    pub fn dale_decay(&self, c: f64) -> Result<()> {
        let Some(x) = self.v50() else { return Ok(()) };
        if x.dale_sign.is_none() {
            return Ok(());
        }
        if self.cfg.arch.dale_proj {
            // D2: AdamW's decoupled decay on v itself, then the projection v >= 0 (also when c = 0)
            for b in &self.blocks {
                let v = b.mlp.w2.detach();
                Var::from_tensor(&b.mlp.w2)?.set(&v.affine(1.0 - c, 0.0)?.relu()?)?;
            }
            *x.dale_cache.borrow_mut() = None;
            return Ok(());
        }
        if c == 0.0 {
            return Ok(());
        }
        for b in &self.blocks {
            let v = b.mlp.w2.detach();
            let vc = v.maximum(-8f32)?;
            let hi = softplus(&vc)?.mul(&vc.neg()?.exp()?.affine(1.0, 1.0)?)?;
            let lo = v.exp()?.affine(0.5, 1.0)?;
            let ratio = v.lt(-8f32)?.where_cond(&lo, &hi)?;
            Var::from_tensor(&b.mlp.w2)?.set(&v.sub(&ratio.affine(c, 0.0)?)?)?;
        }
        *x.dale_cache.borrow_mut() = None;
        Ok(())
    }

    /// --loop: the number of passes (1 without it).
    pub(crate) fn v50_passes(&self) -> usize {
        self.cfg.arch.loops.max(1)
    }

    /// --loop: pass `pass + 1`'s input from the window's input and pass `pass`'s top output.
    pub(crate) fn v50_loop_input(&self, x_in: &Tensor, top: &Tensor) -> Result<Tensor> {
        let w = self.v50().and_then(|x| x.loop_w.as_ref()).ok_or_else(|| candle_core::Error::Msg("--loop: no projection".into()))?;
        loop_add(x_in, top, &self.final_norm, w)
    }

    /// A pass starts: no signals yet (block 0 is never modulated), and the hooks are quiet unless it is the last.
    pub(crate) fn v50_begin_pass(&self, last: bool) {
        if let Some(x) = self.v50() {
            *x.sig.borrow_mut() = None;
            x.quiet.set(!last);
        }
    }

    /// The window's passes are done (nothing kept).
    pub(crate) fn v50_end(&self) {
        if let Some(x) = self.v50() {
            *x.sig.borrow_mut() = None;
            x.quiet.set(false);
        }
    }

    /// Whether this is a pass before the last (--loop): the trace and record hooks stay quiet.
    pub(crate) fn v50_quiet(&self) -> bool {
        self.v50().is_some_and(|x| x.quiet.get())
    }

    /// --neuromod: the nucleus reads block 0's output `x` [n, d] (call after block 0; nothing without the option).
    pub(crate) fn v50_after_block0(&self, x: &Tensor) -> Result<()> {
        if let Some((vx, (g, w, b))) = self.v50().and_then(|vx| vx.nucleus.as_ref().map(|n| (vx, n))) {
            *vx.sig.borrow_mut() = Some(nucleus_signals(x, g, w, b)?);
        }
        Ok(())
    }

    /// --neuromod: signal `k` [n, 1] for this block, if any (None in block 0 and without the option).
    pub(crate) fn v50_signal(&self, k: usize) -> Result<Option<Tensor>> {
        match self.v50().and_then(|x| x.sig.borrow().clone()) {
            Some(s) => Ok(Some(s.narrow(1, k, 1)?)),
            None => Ok(None),
        }
    }

    /// --neuromod NE on the LIF's input current [b, t, d] (unchanged without a signal).
    pub(crate) fn v50_ne(&self, cur: Tensor, b: usize, t: usize) -> Result<Tensor> {
        match self.v50_signal(NE)? {
            Some(g) => cur.broadcast_mul(&g.reshape((b, t, 1))?),
            None => Ok(cur),
        }
    }

    /// --neuromod ACh on the memory's write strength [b, nh, t, g] (unchanged without a signal).
    pub(crate) fn v50_ach(&self, beta: Tensor) -> Result<Tensor> {
        match self.v50_signal(ACH)? {
            Some(g) => ach_scale(&beta, &g),
            None => Ok(beta),
        }
    }

    /// --neuromod DA on the MLP's output [n, d] (unchanged without a signal).
    pub(crate) fn v50_da(&self, y: Tensor) -> Result<Tensor> {
        match self.v50_signal(DA)? {
            Some(g) => y.broadcast_mul(&g),
            None => Ok(y),
        }
    }
}

#[cfg(test)]
mod tests;
