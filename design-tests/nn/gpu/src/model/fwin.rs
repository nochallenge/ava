// TESTING / DESIGN ONLY — v47: model D with Stage 3's winning block options on a hand-written training step.
//
// Stage 3's winners (--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense
// --resid-fix --lif-init logspaced --lif-scale) put D on the block-options path (arch.rs), which trained through candle's
// f32 autograd, 4x slower than plain D's hand-written step. This is that step (`loss_and_grads_a_wr`'s scheme: every
// forward op a kernel, every backward the matching kernel, no autograd; bf16 matmul operands under mm.bf16) for D with
// any subset of these options and nothing else:
//   --resid-fix, --lif-init logspaced, --timescales fixed   init only (the weights' start values, the constant offsets)
//   --kwta F                                                 k = hidden / 4 in the same kWTA kernel
//   --mlp-act relu2                                          fwin_ops::kwta_relu2 (a and a^2), backward du = 2 a da
//   --lif-scale                                              cur = h (Win diag(1 - beta)): the scale lives in the
//                                                            weight (d x d), the LIF kernels are unchanged; backward
//                                                            dWin = dW' diag(1 - beta), d(1 - beta) = colsum(Win * dW')
//   --head-size 32|64, --value-residual                      fwin_ops::wmem_fwd / wmem_bwd (prep -> memg scan -> post)
//   --residual dense                                         fwin_ops::dense_mix / dense_back
// v47 SLOW (Ava 0.4's other five; the same step, any subset of these too):
//   --gates channel                                          P2's per-channel gates and erase key: fwin_ops::wmem_* in
//                                                            their channel form (wprepc), the low-rank gate inputs as
//                                                            one matmul Z = (h Wa1 | h Wb1) [[Wa2, 0], [0, Wb2]]
//   --mem-momentum M                                         the memg scan's momentum form (its own kernels)
//   --spikes ternary|int2, --alif, --homeostasis R          the LIF through lif_gen's own kernels (lifg_*), called
//                                                            directly; homeostasis: the firing rates for
//                                                            homeostasis_step (outside the step, as before)
// v51 fused N (src/fwin_nm.rs; runs/v51/fused_n_check.sh):
//   --neuromod 3                                             the nucleus on block 0's output, its three signals on
//                                                            blocks 1..L-1: NE on the LIF's input current, ACh on the
//                                                            memory's write strengths (between the prep and the scan,
//                                                            fwin_ops::wmem_fwd_nm / wmem_bwd_nm), DA on the MLP's
//                                                            output; their gradients summed over the blocks, then the
//                                                            nucleus's backward into block 0's output gradient
// v51 fast1 (src/fwin_r2.rs; runs/v51/fast1_check.sh), any subset, with or without --neuromod 3:
//   --dendrite, --dopamine, --mood, --rhythm, --prune (v51)  each part's forward op where the reference path has it,
//   --stp (v42 E9)                                           its backward next to the part it scales (fwin_r2's notes);
//                                                            Y's write gate inside fwin_ops::wmem_fwd_g / wmem_bwd_g
// v51 fast2 (src/model/fwin_f2.rs, the `f2` hooks; runs/v51/fast2_check.sh):
//   --letters, --columns K, --thalamus, --cerebellum-head G  R's letter vectors, W's column-wired MLP, TH's relay and
//                                                            gates, CB's granule code and read-out (fwin_f2.rs says how)
// Both sets combine (runs/v51/merge_f.md, runs/v51/fast_merged_check.sh), as on the reference path: R's letters join
// the embedding before Y's rhythms read it; P's context is the memory read before TH's memory gate; W's per-column
// winners and Pr's cut units are never on together (Arch::validate).
// Checked against autograd in tests/fast_winners.rs (runs/v47/prereg_fast_winners.md). AVA_WIN_FUSED=0 keeps the
// autograd path for these models; AVA_MANUAL_DA=0 does too (as for plain D).

#[path = "fwin_f2.rs"]
pub mod f2;

use super::*;
use crate::arch::{Arch, Gating, Spikes, Timescales};
use crate::fwin_nm::{self as nmo, ACH, DA, NE};
use crate::fwin_ops::{self as wo, MemShape, Vres, GATE_RANK};

/// The step is on unless AVA_WIN_FUSED=0 (read once).
/// v47: the forward keeps the memory scan's inputs for the backward (no second prep; same bits) unless
/// AVA_WIN_KEEP_PREP=0.
fn keep_prep() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("AVA_WIN_KEEP_PREP").is_ok_and(|v| v == "0"))
}

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("AVA_WIN_FUSED").is_ok_and(|v| v == "0"))
}

/// Whether `cfg` can train on this step: model D (v47 Amendment 5: with or without the engram read, --engram L),
/// whose block options are among the nine above, the slow five, (v51 fused N) --neuromod 3 and round 2's ten parts (v51
/// fast1: P, DA, M, Y, Pr, S; v51 fast2: R, W, TH, CB, when f2::shapes_ok holds).
pub fn step_ok(cfg: &Config) -> bool {
    let a = cfg.arch;
    // v51 fast2's four cleared first (f2::strip: R and W in a.v51, TH and CB in a.v51b), then fast1's five from that
    // result (merge_f: the explicit `v51` field below must start from a2, or it would put R and W back)
    let a2 = f2::strip(a);
    let rest = Arch {
        neuromod: 0, // v51 fused N (Arch::validate allows 0 or 3, with at least 2 blocks)
        v51: crate::fwin_r2::off(a2.v51), // v51 fast1: P, DA, M, Y, Pr (v51 fast2: R and W, cleared in a2)
        mem: crate::memory::MemOpts { stp: false, ..a.mem }, // v51 fast1: S
        head_size: HEAD_SIZE,
        timescales: Timescales::Index,
        value_residual: false,
        kwta_div: KWTA_DIV,
        relu2: false,
        resid_fix: false,
        dense: false,
        lif_logspaced: false,
        lif_scale: false,
        gates: Gating::Head,
        mem_momentum: 0.0,
        spikes: Spikes::Binary,
        alif: false,
        lif_reset_grad: false, // v52 spike fix: the lifg backward carries the reset when asked
        homeostasis: 0.0,
        ..a2 // v51 fast2: R, W, TH and CB cleared
    };
    cfg.kind == Kind::D
        && f2::shapes_ok(cfg)
        && crate::fused::ATTN_HS == HEAD_SIZE
        && rest == Arch::default()
        && matches!(a.gates, Gating::Head | Gating::Channel)
        && crate::mem_gen::supports(a.head_size)
        && cfg.layers + 1 <= wo::DENSE_MAXD
}

thread_local! {
    /// v51 fused N (the GPU check's logits comparison): Some while capture_logits(true) is on, holding the last step's
    /// logits [N, V] f32. Off (None), the default: nothing kept.
    static LOGITS: std::cell::RefCell<Option<Option<Tensor>>> = const { std::cell::RefCell::new(None) };
}

/// v51 fused N (tests): keep the logits of every later step on this thread (`on`), or stop and drop them.
pub fn capture_logits(on: bool) {
    LOGITS.with(|l| *l.borrow_mut() = if on { Some(None) } else { None });
}

/// v51 fused N (tests): the last step's logits since capture_logits(true), taken out.
pub fn take_logits() -> Option<Tensor> {
    LOGITS.with(|l| l.borrow_mut().as_mut().and_then(|x| x.take()))
}

fn keep_logits(logits: &Tensor) {
    LOGITS.with(|l| {
        if let Some(slot) = l.borrow_mut().as_mut() {
            *slot = Some(logits.clone());
        }
    });
}

/// v52 fasteval: what the step's body returns: the training step's (loss, activity, gradients), or, forward only, the
/// logits [N, V] f32.
pub(crate) enum WinOut {
    Step(Tensor, Activity, GradStore),
    Logits(Tensor),
}

/// merge_f (runs/v51/merge_f.md): dL/d(block i's memory read, after its output projection) as the matmul operand,
/// from dL/dx_m2 `dx` (f32; `dx_m` its operand). v51 fast2 TH: through the memory gate (dx itself without TH). v51 fast1
/// P: plus the dendrites' share `dctx` (they read the read before TH's gate), added in f32 before the operand is made.
/// With neither part: `dx_m`.
pub(crate) fn dread_op(f2: &mut f2::F2, mm: &Mm, i: usize, dx: &Tensor, dx_m: &Tensor, dctx: Option<&Tensor>) -> Result<Tensor> {
    match dctx {
        Some(dc) => Ok(mm.add(&f2.mem_back_as(i, dx, dx, false)?, dc)?.1),
        None => f2.mem_back(i, dx, dx_m),
    }
}

/// Whether a model of `cfg` with block options trains on this step on a CUDA device (`cuda`), the switches allowing:
/// what Model::win_step decides, for the training programs' first lines.
pub fn would_take(cfg: &Config, cuda: bool) -> bool {
    on() && crate::manual_da_on() && crate::fast_path() && cuda && step_ok(cfg)
}

/// v51 cbfix (runs/v51/cbfix.md): would_take on the device `dev` itself: also CB's G within what that device's shared
/// memory holds (f2::device_ok; a smaller card's larger G keeps autograd instead of failing at the launch).
pub fn would_take_on(cfg: &Config, dev: &candle_core::Device) -> bool {
    would_take(cfg, dev.is_cuda()) && f2::device_ok(cfg, dev)
}

impl Model {
    /// Whether this model's block options train on the hand-written winners' step (CUDA and the switches allowing).
    pub fn win_step(&self, ids: &Tensor) -> bool {
        self.ext.is_some() && would_take_on(&self.cfg, ids.device()) && f2::model_ok(self)
    }

    /// v47: the training step of D with Stage 3's winners (see the top of src/model/fwin.rs), with
    /// `loss_and_grads_a_wr`'s target weights: None the mean over every target; `raw` sum(w·loss); else sum(w·loss)/sum(w).
    pub fn loss_and_grads_win(&self, ids: &Tensor, targets: &Tensor, wt: Option<&Tensor>, raw: bool, mm: &Mm) -> Result<(Tensor, Activity, GradStore)> {
        match self.win_pass(ids, Some(targets), wt, raw, mm)? {
            WinOut::Step(loss, act, grads) => Ok((loss, act, grads)),
            WinOut::Logits(_) => candle_core::bail!("loss_and_grads_win: the pass stopped at the logits"),
        }
    }

    /// v52 fasteval (runs/v52/fasteval.md): the forward half of the step above, for scoring: the same ops on the same
    /// operands up to the logits [N, V] f32, then stop (no loss, no backward). Nothing is kept for a backward (no Saved
    /// values, no scan inputs), no activity is tracked and nothing is written for a training step (homeostasis rates,
    /// Pr's use statistics), as the reference forward does outside training. `mm`: f32 (`Mm { bf16: false }`) gives the
    /// reference forward's numbers up to rounding (the scoring tools use it); bf16 is the training step's precision.
    pub fn logits_win(&self, ids: &Tensor, mm: &Mm) -> Result<Tensor> {
        match self.win_pass(ids, None, None, false, mm)? {
            WinOut::Logits(l) => Ok(l),
            WinOut::Step(..) => candle_core::bail!("logits_win: the pass did not stop at the logits"),
        }
    }

    /// v52 fasteval: why this model cannot score on the fused forward on `dev` (None: it can), as `would_take` says it
    /// for the training step: CUDA; the switches (AVA_WIN_FUSED, AVA_MANUAL_DA, AVA_FAST); a model the fused steps take
    /// (block options of the fused WIN step, or today's block D: loss_and_grads_a's step); and none of the inspection
    /// hooks only the reference forward has (int8 copies, traces, injections, state and ReDo records, training forwards).
    pub fn fused_eval_why(&self, dev: &candle_core::Device) -> Option<String> {
        if !dev.is_cuda() {
            return Some("the fused kernels are CUDA only; on the CPU the reference forward scores".into());
        }
        for (off, what) in [(!on(), "AVA_WIN_FUSED=0"), (!crate::manual_da_on(), "AVA_MANUAL_DA=0"), (!crate::fast_path(), "AVA_FAST=0")] {
            if off {
                return Some(format!("{what} switches the fused steps off"));
            }
        }
        let ext = self.ext.as_ref();
        let hooks = [
            (self.q8.borrow().is_some(), "int8 copies are on"),
            (self.tracing(), "a forward trace is recording"),
            (self.train_fwd.get(), "a training forward"),
            (self.force_arch.get(), "force_arch is on (tests)"),
            (ext.is_some_and(|e| e.inject.borrow().is_some()), "an injection is on (introspect)"),
            (ext.is_some_and(|e| e.state_rec.borrow().is_some()), "the block states are recording (sae)"),
            (ext.is_some_and(|e| e.redo_rec.borrow().is_some()), "ReDo's activity record is on"),
        ];
        if let Some((_, what)) = hooks.iter().find(|h| h.0) {
            return Some(format!("{what}: only the reference forward has that hook"));
        }
        if ext.is_some() {
            if !step_ok(&self.cfg) {
                return Some(format!("block options the fused WIN step does not take ({})", self.cfg.arch.signature()));
            }
            if !f2::device_ok(&self.cfg, dev) {
                return Some("the cerebellum head's G is past this device's shared memory".into());
            }
            if !f2::model_ok(self) {
                return Some("a talk-time part's state is on (CB online, e-prop or the part signals)".into());
            }
            return None;
        }
        if self.cfg.kind != Kind::D {
            return Some(format!("model {:?} (today's block): only D's fused step forward is checked", self.cfg.kind));
        }
        if HEAD_SIZE != crate::fused::ATTN_HS || self.pos.is_some() {
            return Some("today's block D without the step's shapes (head size, no position table)".into());
        }
        None
    }

    /// v52 fasteval: which fused forward this model scores on (for the tools' first lines).
    pub fn fused_eval_name(&self) -> &'static str {
        if self.ext.is_some() {
            "the fused WIN step's forward (f32)"
        } else {
            "the fused D step's forward (loss_and_grads_a, f32)"
        }
    }

    /// v52 fasteval: the logits [N, V] f32 of `ids` [B, T] on the fused forward (logits_win with block options, else
    /// logits_a), f32 matmuls as the reference forward's; an error when fused_eval_why gives a reason.
    pub fn logits_fused(&self, ids: &Tensor) -> Result<Tensor> {
        if let Some(why) = self.fused_eval_why(ids.device()) {
            candle_core::bail!("logits_fused: {why}");
        }
        let mm = Mm { bf16: false };
        if self.ext.is_some() {
            self.logits_win(ids, &mm)
        } else {
            self.logits_a(ids, &mm)
        }
    }

    /// The step's body: `targets` None = the forward only, up to the logits (logits_win).
    fn win_pass(&self, ids: &Tensor, targets: Option<&Tensor>, wt: Option<&Tensor>, raw: bool, mm: &Mm) -> Result<WinOut> {
        use crate::fused::{kwta_relu_b, kwta_relu_raw, lif_bwd_b, lif_fwd_b, relu_mask_bwd_b, relu_mask_bwd_raw, rms_bwd_raw};
        use crate::lif::{ce_bwd_raw, ce_fwd_raw, lif_bwd_raw, lif_fwd_raw};
        if !step_ok(&self.cfg) || !ids.device().is_cuda() || !f2::device_ok(&self.cfg, ids.device()) {
            candle_core::bail!("loss_and_grads_win: model D on CUDA with only the winners' options (got {})", self.cfg.arch.signature());
        }
        // v52 fasteval: forward only (no targets): nothing saved, tracked or written for training
        let fwd = targets.is_none();
        let ar = self.cfg.arch;
        let (b, t) = ids.dims2()?;
        let (d, n, k, nh, hs) = (self.cfg.d, b * t, self.cfg.kwta_k(), self.cfg.heads(), ar.head_size);
        let nl = self.blocks.len();
        let ms = MemShape { b, t, nh, hs, chan: ar.gates == Gating::Channel, mom: ar.mem_momentum };
        // SLOW: the LIF through lif_gen's kernels unless it is the plain binary one; --homeostasis: each block's rates
        let lgen = !ar.lif_fused_ok();
        let mut rates: Vec<Option<Tensor>> = Vec::new();
        let bf = mm.bf16;
        let dev = ids.device();
        let track = self.track.get() && !fwd;
        let bxs = self.ext.as_ref().map(|e| &e.blocks);
        let bx = |i: usize| bxs.map(|v| &v[i]);
        let mut act = Activity { spike_rate: Vec::new(), kwta_post: Vec::new() };
        let tok = self.tok.detach();
        let tok_b = mm.op(&tok)?;
        let idf = ids.flatten_all()?;
        // v52 fasteval: block 0's projection, the only value a later block reads (--value-residual), when nothing is saved
        let mut p0_fwd: Option<Tensor> = None;
        // v47 Amendment 5: the engram read's saved values, as loss_and_grads_a's
        struct Eng {
            x_pre: Tensor,
            h: Tensor,
            qkv: Tensor,
            o: Tensor,
            ob: Tensor,
            lse: Tensor,
            wqkv: Tensor,
            wo: Tensor,
        }
        let scale = 1.0 / (crate::fused::ATTN_HS as f64).sqrt();
        let tc = bf && crate::attn_tc_on();
        struct Saved {
            x_in: Tensor,
            h1: Tensor,
            cur: Tensor,
            beta: Tensor,
            /// --lif-scale: 1 - beta
            omb: Option<Tensor>,
            /// SLOW's LIF (lif_gen's kernels): its flat parameters [beta | theta | rho | gain | 0 (3 B d)]
            lif_ps: Option<Tensor>,
            s: Tensor,
            x_m1: Tensor,
            h: Tensor,
            p: Tensor,
            cst: Tensor,
            y: Tensor,
            pre: Option<(Tensor, Tensor)>,
            o: Tensor,
            wcat: Tensor,
            /// --gates channel: (Z f32 [N, 2d], the low-rank inputs (h Wa1 | h Wb1) and [[Wa2, 0], [0, Wb2]] as operands)
            chz: Option<(Tensor, Tensor, Tensor)>,
            wo: Tensor,
            /// v47 Amendment 5: the engram read (--engram L) after the memory, before the MLP
            eng: Option<Eng>,
            /// the MLP's input (after the engram read when there is one)
            x_m2: Tensor,
            h2: Tensor,
            /// --mlp-act relu2: the relu value a (a_m is then a^2)
            a_r: Option<Tensor>,
            a_m: Tensor,
            w: [Tensor; 4], // Win (times diag(1 - beta) with --lif-scale), Wout, W1, W2 as matmul operands
            /// v51 fused N (--neuromod, blocks after 0): the LIF's input current before NE's gain [B, T, d], the
            /// memory's unscaled write strengths [B, H, T, G] (when the scan inputs are kept), the MLP's output before
            /// DA's scale [N, d]
            cur0: Option<Tensor>,
            braw: Option<Tensor>,
            mlp0: Option<Tensor>,
            /// v51 fast1 (src/fwin_r2.rs): the round 2 parts' saved values (empty without them)
            r2: crate::fwin_r2::Blk,
        }
        let mut saved: Vec<Saved> = Vec::with_capacity(nl);
        // v51 fused N: the nucleus (norm gain, read-out, bias) under --neuromod, and its signals once block 0 has run
        let nm = match (ar.neuromod > 0, self.ext.as_ref().and_then(|e| e.v50.nucleus.as_ref())) {
            (false, _) => None,
            (true, Some(n)) => Some(n),
            (true, None) => candle_core::bail!("loss_and_grads_win: --neuromod without the nucleus"),
        };
        let mut nu: Option<nmo::Nucleus> = None;
        let mut x = tok.embedding(&idf)?; // [N, d]
        let (place_off, mag_off, eng_off) = self.lesion.get();
        let pid = if self.place.is_some() && !place_off { Some(crate::fused::place_ids(ids)?) } else { None };
        if let (Some(place), Some(pid)) = (&self.place, &pid) {
            x = x.add(&Self::place_table(&place.detach())?.embedding(pid)?)?;
        }
        let mfeat = match (&self.mag_w, mag_off) {
            (Some(mw), false) => {
                let f = crate::fused::mag_feats(ids)?;
                x = x.add(&f.matmul(&mw.detach())?)?;
                Some(f)
            }
            _ => None,
        };
        let mut f2 = f2::F2::new(self, &idf, mm)?; // v51 fast2 (nothing without R, W, TH and CB)
        x = f2.embed(x)?; // R: the letters
        // v51 fast1 Y (--rhythm): the rhythms' phases from the embedding, R's letters included (v51::begin runs after
        // v51::embed); the forward's signals (DA and M after block 0)
        let mut r2f = crate::fwin_r2::begin(self, &x, b, t)?;
        // --residual dense: every block's weights in one vector, and the streams (row 0 the embedding) as rows
        let dense = if ar.dense {
            let ws: Vec<Tensor> = (0..nl).map(|i| bx(i).and_then(|e| e.dense.as_ref()).map(|w| w.detach()).ok_or_else(|| candle_core::Error::Msg(format!("dense weights of block {i}")))).collect::<Result<_>>()?;
            let wf = Tensor::cat(&ws, 0)?;
            let s = wo::empty(&x, (nl + 1, n * d), DType::F32)?;
            s.slice_set(&x.reshape((1, n * d))?, 0, 0)?;
            Some((wf, s))
        } else {
            None
        };
        f2.relay_begin(&x)?; // TH: the relay reads the blocks' input
        for (i, blk) in self.blocks.iter().enumerate() {
            let m = blk.mem.as_ref().expect("D block has SpikeMem");
            let l = &blk.mlp;
            let hb = blk.hebb.as_ref().expect("D block has fast weights");
            f2.block_begin(i)?; // TH: this block's gates
            let beta = sigmoid(&m.beta.detach())?;
            let omb = if ar.lif_scale { Some(beta.affine(-1.0, 1.0)?) } else { None };
            let win = match &omb {
                Some(o) => m.win.detach().broadcast_mul(o)?,
                None => m.win.detach(),
            };
            let w = [mm.op(&win)?, mm.op(&m.wout.detach())?, mm.op(&f2.w1(i, &l.w1)?)?, mm.op(&f2.w2(i, &l.w2)?)?]; // W: folded
            let h1 = mm.rms(&x, &m.norm.detach())?;
            let cur = mm.mm(&h1, false, &w[0], false)?.reshape((b, t, d))?;
            // v51 fused N: this block's signals (none in block 0); NE's gain on the LIF's input drive (block_arch's op)
            let sg = if i > 0 { nu.as_ref() } else { None };
            let (cur, cur0) = match sg {
                Some(nu) => (cur.broadcast_mul(&nu.cols[NE].reshape((b, t, 1))?)?, Some(cur)),
                None => (cur, None),
            };
            // v51 fast1: this block's round 2 values; M: the mood's gain on the drive (after NE, as block_arch)
            let mut r2 = crate::fwin_r2::Blk::default();
            let cur = crate::fwin_r2::mood_fwd(self, i, &r2f, cur, &mut r2)?;
            let (s, lif_ps) = if lgen {
                // SLOW: lif_gen's scan (graded spikes, adaptive and per-neuron thresholds), no carried state
                let th = match bx(i).and_then(|e| e.theta.as_ref()) {
                    Some(th) => th.detach(),
                    None => Tensor::full(LIF_THRESHOLD as f32, d, dev)?,
                };
                let (rho, gain) = match bx(i).and_then(|e| e.alif.as_ref()) {
                    Some((r, g)) => (sigmoid(&r.detach())?, g.detach()),
                    None => (Tensor::zeros(d, DType::F32, dev)?, Tensor::zeros(d, DType::F32, dev)?),
                };
                let ps = Tensor::cat(&[&beta, &th, &rho, &gain, &Tensor::zeros(3 * b * d, DType::F32, dev)?], 0)?;
                let s = wo::lifg_fwd(&cur, &ps, ar.spikes, ar.alif)?.narrow(0, 0, n * d)?.reshape((n, d))?;
                // v51 fast1 S (--stp): the synapse on the spikes (what reaches Wout, and the rates' input)
                let s = crate::fwin_r2::stp_fwd(self, i, s.reshape((b, t, d))?, &mut r2)?.reshape((n, d))?;
                if track {
                    act.spike_rate.push(mean_rows_first(&s.abs()?)?);
                }
                if ar.homeostasis > 0.0 && !fwd {
                    rates.push(Some(s.abs()?.sum(0)?.affine(1.0 / n as f64, 0.0)?));
                }
                (mm.op(&s)?, Some(ps))
            } else {
                let s3 = if bf { lif_fwd_b(&cur, &beta)? } else { lif_fwd_raw(&cur, &beta)? };
                let s3 = match crate::fwin_r2::stp_of(self, i) {
                    // v51 fast1 S (--stp): the synapse on the 0/1 spikes (f32), then as the operand
                    Some(_) => mm.op(&crate::fwin_r2::stp_fwd(self, i, s3.to_dtype(DType::F32)?, &mut r2)?)?,
                    None => s3,
                };
                if track {
                    act.spike_rate.push(mean_rows_first(&s3.to_dtype(DType::F32)?)?);
                }
                (s3.reshape((n, d))?, None)
            };
            let x_m1 = x.add(&mm.mm(&s, false, &w[1], false)?)?;
            // the fast-weight memory
            let g = hb.gates.as_ref().expect("D has gates");
            let wg = hb.wg.as_ref().expect("D has the silu output gate here").detach();
            let chan = bx(i).and_then(|e| e.chan.as_ref());
            let mut wl = vec![hb.wq.detach(), hb.wk.detach(), hb.wv.detach(), wg, g.wa.detach(), g.wb.detach()];
            if let Some(c) = chan {
                wl.extend([c.we.detach(), c.wa1.detach(), c.wb1.detach()]);
            }
            let wcat = mm.op(&Tensor::cat(&wl, 1)?)?;
            drop(wl);
            let off = chan.map_or(&g.off, |c| &c.off);
            let cst = Tensor::cat(&[&g.ba.detach(), &g.bb.detach(), off], 0)?;
            let h = mm.rms(&x_m1, &hb.norm.detach())?;
            let p = mm.mm(&h, false, &wcat, false)?; // f32 [N, ms.ld()]
            let chz = match chan {
                Some(c) => {
                    // the channel gates' low-rank parts: Z = (h Wa1 | h Wb1) [[Wa2, 0], [0, Wb2]] (f32 [N, 2d])
                    let a = mm.op(&p.narrow(1, 5 * d + 2 * nh, 2 * GATE_RANK)?.contiguous()?)?;
                    let zr = Tensor::zeros((GATE_RANK, d), DType::F32, dev)?;
                    let w2 = Tensor::cat(&[&Tensor::cat(&[&c.wa2.detach(), &zr], 1)?, &Tensor::cat(&[&zr, &c.wb2.detach()], 1)?], 0)?;
                    let w2 = mm.op(&w2)?;
                    Some((mm.mm(&a, false, &w2, false)?, a, w2))
                }
                None => None,
            };
            let vres_w = bx(i).and_then(|e| e.vres.as_ref()).map(|v| v.detach());
            // v52 fasteval: block 0's projection from Saved, or (forward only) from p0_fwd
            let p0 = if fwd { p0_fwd.as_ref() } else { saved.first().map(|s| &s.p) };
            let vr = match (&vres_w, ar.value_residual && i > 0, p0) {
                (Some(vw), true, Some(p0)) => Vres::Mix { p: vw, p0 },
                (Some(_), true, None) => candle_core::bail!("loss_and_grads_win: --value-residual without block 0's projection"),
                _ => Vres::Off,
            };
            // v51 fused N: ACh's scale on the write strengths, between the prep and the scan
            let ach = sg.map(|nu| &nu.cols[ACH]);
            // v51 fast1 Y (--rhythm): the rhythms' gates on the write strengths (before ACh) and on the read (per head,
            // on the f32 read-out, then the operand)
            let ywg = crate::fwin_r2::wgate(self, i, &r2f, nh, &mut r2)?;
            let yrg = crate::fwin_r2::rgate(self, i, &r2f)?;
            // v52 fasteval: the scan's inputs are kept only for a backward
            let (o, y, pre, braw, bung) = wo::wmem_fwd_g(&p, &cst, chz.as_ref().map(|z| &z.0), vr, ms, NORM_EPS, bf && yrg.is_none(), keep_prep() && !fwd, ach, ywg.as_ref())?;
            r2.bung = bung;
            let o = match yrg {
                Some((gr, th)) => {
                    let og = mm.op(&crate::fwin_r2::rgate_fwd(&o, &gr)?)?;
                    r2.rg = Some((gr, th, o));
                    og
                }
                None => o,
            };
            let wo_ = mm.op(&hb.wo.detach())?;
            let mread = mm.mm(&o, false, &wo_, false)?;
            if crate::fwin_r2::wants_ctx(self, i) {
                // v51 fast1 P: the dendrites' context, the memory read (before TH's memory gate, as block_arch's put_ctx)
                r2.ctx = Some(mread.clone());
            }
            let x_m2 = x_m1.add(&f2.mem_out(i, mread)?)?; // v51 fast2 TH: times the memory gate
            // v47 Amendment 5: x_m3 = x_m2 + engram(rms(x_m2)) (--engram L), loss_and_grads_a's code; the MLP reads x_m3
            let (eng, x_m2) = match (&blk.eng, eng_off) {
                (Some(e), false) => {
                    let wqkv = mm.op(&Tensor::cat(&[&e.wq.detach(), &e.wk.detach(), &e.wv.detach()], 1)?)?;
                    let wo = mm.op(&e.wo.detach())?;
                    let h = mm.rms(&x_m2, &e.norm.detach())?;
                    let (qkv, o, ob, lse) = if tc {
                        let qkv = crate::fused::mm_bf16_b(&h, false, &wqkv, false)?;
                        let (ob, lse) = crate::fused::attn_fwd_tc(&qkv, b, t, scale)?;
                        (qkv, ob.clone(), ob, lse)
                    } else {
                        let qkv = mm.mm(&h, false, &wqkv, false)?;
                        let (o, ob, lse) = crate::fused::attn_fwd_s(&qkv, b, t, scale)?;
                        let ob = if bf { ob } else { o.clone() };
                        (qkv, o, ob, lse)
                    };
                    let x_m3 = x_m2.add(&mm.mm(&ob, false, &wo, false)?)?;
                    (Some(Eng { x_pre: x_m2, h, qkv, o, ob, lse, wqkv, wo }), x_m3)
                }
                _ => (None, x_m2),
            };
            // the kWTA MLP (relu or relu^2); v51 fast1 Pr: the cut units at -1e9, k of the live ones; P: the units in f32
            // (scaled by the dendrites), then the operand; Pr measures the units for the cuts as they are (v52 fix_Pr:
            // in the step's precision, as B's, so Pr with no unit cut is B's step bit for bit)
            let h2 = mm.rms(&x_m2, &l.norm.detach())?;
            let u = mm.mm(&h2, false, &w[2], false)?;
            let (u, kb) = crate::fwin_r2::prune_u(self, i, u, k)?;
            let (u, kb) = f2.cols_in(u, kb)?; // v51 fast2 W: the winners per column (never with Pr: Arch::validate)
            let f32u = crate::fwin_r2::mlp_f32(self, i);
            let bu = bf && !f32u;
            let (a_r, a_m) = if ar.relu2 {
                let (a, a2) = wo::kwta_relu2(&u, kb, bu)?;
                (Some(a), a2)
            } else {
                (None, if bu { kwta_relu_b(&u, kb)? } else { kwta_relu_raw(&u, kb)? })
            };
            drop(u);
            let (a_r, a_m) = f2.cols_out(a_r, a_m, n)?;
            if track {
                act.kwta_post.push(mean_rows_first(&a_r.as_ref().unwrap_or(&a_m).gt(0.0)?.to_dtype(DType::F32)?)?);
            }
            let a_m = if f32u {
                let a = crate::fwin_r2::dend_fwd(self, i, a_m, &mut r2)?; // P: the dendrites' gain
                if !fwd {
                    crate::fwin_r2::prune_stats(self, i, &a)?; // Pr: the units' mean |a| (after P, as v51::mlp)
                }
                mm.op(&a)?
            } else {
                if !fwd {
                    crate::fwin_r2::prune_stats(self, i, &a_m)?; // Pr: the units' mean |a| (bf16 units read in f32)
                }
                a_m
            };
            let mlp = mm.mm(&a_m, false, &w[3], false)?;
            // v51 fast1 DA (--dopamine): the surprise's gain on the MLP's output (before v50's DA, as v51::mlp)
            let mlp = crate::fwin_r2::da_fwd(self, i, &r2f, mlp, &mut r2)?;
            // v51 fused N: DA's scale on the MLP's output
            let (mlp, mlp0) = match sg {
                Some(nu) => (mlp.broadcast_mul(&nu.cols[DA])?, Some(mlp)),
                None => (mlp, None),
            };
            let (xa, xb) = f2.res_in(i, &x, x_m2.clone(), mlp)?; // TH: the residual gate (x_m2, mlp without it)
            let x_out = match &dense {
                Some((wf, sa)) => wo::dense_mix(&xa, &xb, wf, sa, i)?,
                None => xa.add(&xb)?,
            };
            drop((xa, xb));
            f2.relay_after(i, &x_out)?; // TH: the relay reads this block's output
            if let (0, Some((g, wn, bn))) = (i, nm) {
                // v51 fused N: the nucleus reads block 0's output (after its mix): the signals of blocks 1..L-1
                nu = Some(nmo::nucleus_fwd(&x_out, &g.detach(), &wn.detach(), &bn.detach())?);
            }
            if i == 0 {
                crate::fwin_r2::after0(self, &mut r2f, &x_out, ids, &tok)?; // v51 fast1 DA, M: the surprise and the mood
            }
            if fwd {
                // v52 fasteval: nothing kept for a backward; block 0's projection for the later blocks' value residual
                if i == 0 && ar.value_residual {
                    p0_fwd = Some(p);
                }
            } else {
                saved.push(Saved { x_in: x, h1, cur, beta, omb, lif_ps, s, x_m1, h, p, cst, y, pre, o, wcat, chz, wo: wo_, eng, x_m2, h2, a_r, a_m, w, cur0, braw, mlp0, r2 });
            }
            x = x_out;
        }
        if let (true, false, Some(e)) = (ar.homeostasis > 0.0, fwd, self.ext.as_ref()) {
            // E8: the rates homeostasis_step reads after the optimizer step (as the autograd forward leaves them)
            *e.rates.borrow_mut() = rates;
        }
        let gf = self.final_norm.detach();
        let hf = f2.head_in(&x, &gf, mm.rms(&x, &gf)?)?; // CB: plus the cerebellum's correction
        let logits = mm.mm(&hf, false, &tok_b, true)?; // [N, V]
        let Some(targets) = targets else {
            return Ok(WinOut::Logits(logits)); // v52 fasteval: the forward stops here
        };
        let tg = targets.flatten_all()?;
        keep_logits(&logits);
        let ce = ce_fwd_raw(&logits, &tg)?; // [N, 2] = (loss, lse)
        let (loss, row_scale) = match wt {
            None => (ce.narrow(1, 0, 1)?.mean_all()?, Tensor::full((1.0 / n as f64) as f32, (n, 1), dev)?),
            Some(w) if raw => {
                let w = w.reshape((n, 1))?.contiguous()?;
                (ce.narrow(1, 0, 1)?.mul(&w)?.sum_all()?, w)
            }
            Some(w) => {
                let w = w.reshape((n, 1))?;
                let ws = w.broadcast_div(&w.sum_all()?)?;
                (ce.narrow(1, 0, 1)?.mul(&ws)?.sum_all()?, ws)
            }
        };
        // ---- backward
        let mut grads = Tensor::zeros((), DType::F32, dev)?.backward()?;
        let aux = Tensor::cat(&[&row_scale, &ce.narrow(1, 1, 1)?], 1)?;
        let dlogits = if bf { crate::fused::ce_bwd_b(&logits, &tg, &aux)? } else { ce_bwd_raw(&logits, &tg, &aux)? };
        drop(logits);
        let dhf = mm.mm(&dlogits, false, &tok_b, false)?;
        f2.head_bwd(&dhf, &mut grads)?; // CB: the read-out's gradient (xf's passes unchanged)
        let dtok_out = mm.mm(&dlogits, true, &hf, false)?; // [V, d]
        drop(dlogits);
        let (dx0, dgf) = rms_bwd_raw(&x, &gf, &dhf, NORM_EPS)?;
        grads.insert(&self.final_norm, dgf);
        let mut dx = dx0; // dL/d(the stream after the block about to be walked back)
        let mut dx_m = if dense.is_some() { dx.clone() } else { mm.op(&dx)? };
        // dense: G [L, N·d], row i = dL/d(the mix after block i)
        let gall = match &dense {
            Some(_) => Some(wo::empty(&dx, (nl, n * d), DType::F32)?),
            None => None,
        };
        // value residual: the first block's v gradient from the later blocks so far
        let mut acc: Option<Tensor> = None;
        // v51 fused N: dL/d(signal) per signal (NE, ACh, DA), summed over blocks 1..L-1
        let mut dsig: [Option<Tensor>; 3] = [None, None, None];
        // v51 fast1: df (Y) and dm (M), summed over the blocks
        let mut b2 = crate::fwin_r2::Bwd::default();
        for i in (0..nl).rev() {
            let blk = &self.blocks[i];
            let sv = &saved[i];
            let m = blk.mem.as_ref().expect("D block has SpikeMem");
            let l = &blk.mlp;
            let hb = blk.hebb.as_ref().expect("D block has fast weights");
            let sg = if i > 0 { nu.as_ref() } else { None }; // v51 fused N: this block's signals
            if let (0, Some(nu), Some((g, wn, bn))) = (i, nu.as_ref(), nm) {
                // v51 fused N: the nucleus read block 0's output (block 1's input x_in): its gradient joins the one
                // that block 1 sent back, before block 0's mix is walked back
                let ds = nmo::dsig_cat(&dsig, n, &dx)?;
                let (dxn, dg, dw, db) = nmo::nucleus_bwd(nu, &saved[1].x_in, &g.detach(), &wn.detach(), &ds)?;
                grads.insert(g, dg);
                grads.insert(wn, dw);
                grads.insert(bn, db);
                (dx, dx_m) = mm.add(&dx, &dxn)?;
            }
            if let (Some((wf, sa)), Some(ga)) = (&dense, &gall) {
                // the mix after block i: dw_i, and the gradient of block i's raw output (this mix's and the later ones')
                ga.slice_set(&dx.reshape((1, n * d))?, 0, i)?;
                let (out, outb, dw) = wo::dense_back(sa, ga, wf, &ga.narrow(0, i, 1)?, i + 1, i, i + 2, bf)?;
                let wv = bx(i).and_then(|e| e.dense.as_ref()).expect("dense weights");
                grads.insert(wv, dw.expect("dense dw"));
                dx = out.reshape((n, d))?;
                dx_m = match outb {
                    Some(o) => o.reshape((n, d))?,
                    None => dx.clone(),
                };
            }
            if let Some(r) = f2.res_back(i, &dx)? {
                // TH: the residual gate: dx becomes dL/d(the raw output)
                (dx, dx_m) = mm.add(&dx, &r)?;
            }
            // x_out = x_m2 + act(rms(x_m2) w1) w2 (v51 fused N: times DA's signal; dx is dL/d(the raw output))
            // v51 fast1 DA: then v51's gain, in f32, before the operand
            let r2da = sv.r2.da.is_some();
            let dmlp = match (&sv.mlp0, sg) {
                (Some(m0), Some(nu)) => {
                    let (dm, ds) = nmo::rows_bwd(&dx, m0, &nu.cols[DA], bf && !r2da)?;
                    nmo::acc(&mut dsig[DA], ds)?;
                    dm
                }
                _ if r2da => dx.clone(),
                _ => dx_m.clone(),
            };
            let dmlp = if r2da { mm.op(&crate::fwin_r2::da_bwd(self, i, &r2f, &sv.r2, &dmlp, &mut grads)?)? } else { dmlp };
            let da = mm.mm(&dmlp, false, &sv.w[3], true)?;
            f2.put_w(i, true, &l.w2, mm.mm(&sv.a_m, true, &dmlp, false)?, &mut grads)?; // W: unfolded
            drop(dmlp);
            // v51 fast1 P: the dendrites' backward: dL/d(the units before their gain), and the memory read's share
            let (da, dctx) = match crate::fwin_r2::dend_bwd(self, i, &sv.r2, &da, &mut grads)? {
                Some((da, dc)) => (da, Some(dc)),
                None => (da, None),
            };
            let du = match (&sv.a_r, crate::fwin_r2::dend_units(&sv.r2)) {
                (Some(a), _) => wo::relu2_bwd(a, &da)?,
                (None, Some(a0)) => relu_mask_bwd_raw(a0, &da)?,
                (None, None) if bf => relu_mask_bwd_b(&sv.a_m, &da)?,
                (None, None) => relu_mask_bwd_raw(&sv.a_m, &da)?,
            };
            let du = if bf && crate::fwin_r2::mlp_f32(self, i) { mm.op(&du)? } else { du }; // P: f32 units (v52 fix_Pr: not Pr)
            drop(da);
            let dh2 = mm.mm(&du, false, &sv.w[2], true)?;
            f2.put_w(i, false, &l.w1, mm.mm(&sv.h2, true, &du, false)?, &mut grads)?;
            drop(du);
            let (dxm, dg2) = rms_bwd_raw(&sv.x_m2, &l.norm.detach(), &dh2, NORM_EPS)?;
            grads.insert(&l.norm, dg2);
            (dx, dx_m) = mm.add(&dx, &dxm)?;
            // v47 Amendment 5: x_m3 = x_m2 + engram(rms(x_m2)), loss_and_grads_a's backward
            if let Some(es) = &sv.eng {
                let e = blk.eng.as_ref().expect("engram block");
                grads.insert(&e.wo, mm.mm(&es.ob, true, &dx_m, false)?);
                let dqkv = if tc {
                    let dout = crate::fused::mm_bf16_b(&dx_m, false, &es.wo, true)?;
                    crate::fused::attn_bwd_tc(&es.qkv, &es.ob, &es.lse, &dout, b, t, scale)?
                } else {
                    let dout = mm.mm(&dx_m, false, &es.wo, true)?;
                    // v51 fast1 (F-1): the f32 step takes the f32 store (the bf16 one cost ~1e-3 of these gradients)
                    if bf { crate::fused::attn_bwd_s(&es.qkv, &es.o, &es.lse, &dout, b, t, scale)? } else { crate::fused::attn_bwd_sf(&es.qkv, &es.o, &es.lse, &dout, b, t, scale)? }
                };
                let dw = mm.mm(&es.h, true, &dqkv, false)?;
                grads.insert(&e.wq, dw.narrow(1, 0, d)?.contiguous()?);
                grads.insert(&e.wk, dw.narrow(1, d, d)?.contiguous()?);
                grads.insert(&e.wv, dw.narrow(1, 2 * d, d)?.contiguous()?);
                let dh = mm.mm(&dqkv, false, &es.wqkv, true)?;
                let (dxi, dgn) = rms_bwd_raw(&es.x_pre, &e.norm.detach(), &dh, NORM_EPS)?;
                grads.insert(&e.norm, dgn);
                (dx, dx_m) = mm.add(&dx, &dxi)?;
            }
            // x_m2 = x_m1 + memory(rms(x_m1)) wo
            let g = hb.gates.as_ref().expect("D has gates");
            // v51 fast1 P + v51 fast2 TH: dL/d(the memory read) (dread_op)
            let dread_m = dread_op(&mut f2, mm, i, &dx, &dx_m, dctx.as_ref())?;
            grads.insert(&hb.wo, mm.mm(&sv.o, true, &dread_m, false)?);
            let d_o = mm.mm(&dread_m, false, &sv.wo, true)?; // f32 [N, d]
            drop(dread_m);
            // v51 fast1 Y: the read gate's backward (dL/d(the read-out before it); df and its weight's gradient)
            let d_o = crate::fwin_r2::rgate_bwd(self, i, &r2f, &sv.r2, &d_o, &mut b2, &mut grads)?;
            let vres_w = bx(i).and_then(|e| e.vres.as_ref());
            let vr = match (vres_w, ar.value_residual, i) {
                (Some(vw), true, i) if i > 0 => Vres::Mix { p: vw, p0: &saved[0].p },
                (_, true, 0) => Vres::First,
                _ => Vres::Off,
            };
            let ach = sg.map(|nu| (&nu.cols[ACH], sv.braw.as_ref())); // v51 fused N: ACh's scale on the write strengths
            let ywg = sv.r2.wg.as_ref().map(|(gt, _)| (gt, sv.r2.bung.as_ref())); // v51 fast1 Y: the write gate
            let wg = wo::wmem_bwd_g(&sv.p, &sv.cst, sv.chz.as_ref().map(|z| &z.0), vr, &sv.y, &d_o, acc.as_ref(), ms, NORM_EPS, bf, sv.pre.as_ref().map(|(a, b)| (a, b)), ach, ywg)?;
            drop(d_o);
            if let Some(ds) = wg.dach {
                nmo::acc(&mut dsig[ACH], ds)?;
            }
            if let Some(dgt) = &wg.dwgate {
                crate::fwin_r2::wgate_bwd_params(self, i, &r2f, &sv.r2, dgt, &mut b2, &mut grads)?;
            }
            let chan = bx(i).and_then(|e| e.chan.as_ref());
            if let (Some(c), Some((_, a, w2)), Some(dzc)) = (chan, &sv.chz, &wg.dzc) {
                // Z = A W2: dW2 = A^T dZ (its two diagonal blocks), and dA = dZ W2^T into dP's Wa1, Wb1 columns
                let dw2 = mm.mm(a, true, dzc, false)?; // [2 R, 2d]
                grads.insert(&c.wa2, dw2.narrow(0, 0, GATE_RANK)?.narrow(1, 0, d)?.contiguous()?);
                grads.insert(&c.wb2, dw2.narrow(0, GATE_RANK, GATE_RANK)?.narrow(1, d, d)?.contiguous()?);
                let da = mm.op(&mm.mm(dzc, false, w2, true)?)?; // [N, 2 R]
                wg.dp.slice_set(&da, 1, 5 * d + 2 * nh)?;
            }
            if let (Some(vw), Some(dv)) = (vres_w, wg.dvres) {
                grads.insert(vw, dv);
            }
            match vr {
                Vres::Mix { .. } => acc = wg.acc,
                Vres::First => acc = None,
                Vres::Off => {}
            }
            let dw = mm.mm(&sv.h, true, &wg.dp, false)?; // [d, 4d + 2nh]
            grads.insert(&hb.wq, dw.narrow(1, 0, d)?.contiguous()?);
            grads.insert(&hb.wk, dw.narrow(1, d, d)?.contiguous()?);
            grads.insert(&hb.wv, dw.narrow(1, 2 * d, d)?.contiguous()?);
            grads.insert(hb.wg.as_ref().expect("silu gate"), dw.narrow(1, 3 * d, d)?.contiguous()?);
            grads.insert(&g.wa, dw.narrow(1, 4 * d, nh)?.contiguous()?);
            grads.insert(&g.wb, dw.narrow(1, 4 * d + nh, nh)?.contiguous()?);
            grads.insert(&g.ba, wg.dz.narrow(1, 0, nh)?.sum(0)?);
            grads.insert(&g.bb, wg.dz.narrow(1, nh, nh)?.sum(0)?);
            if let Some(c) = chan {
                let c0 = 4 * d + 2 * nh;
                grads.insert(&c.we, dw.narrow(1, c0, d)?.contiguous()?);
                grads.insert(&c.wa1, dw.narrow(1, c0 + d, GATE_RANK)?.contiguous()?);
                grads.insert(&c.wb1, dw.narrow(1, c0 + d + GATE_RANK, GATE_RANK)?.contiguous()?);
            }
            let dh = mm.mm(&wg.dp, false, &sv.wcat, true)?;
            drop(wg.dp);
            let (dxi, dgn) = rms_bwd_raw(&sv.x_m1, &hb.norm.detach(), &dh, NORM_EPS)?;
            grads.insert(&hb.norm, dgn);
            (dx, dx_m) = mm.add(&dx, &dxi)?;
            // x_m1 = x_in + lif(rms(x_in) win') wout
            let ds = mm.mm(&dx_m, false, &sv.w[1], true)?;
            grads.insert(&m.wout, mm.mm(&sv.s, true, &dx_m, false)?);
            // v51 fast1 S: dL/d(the LIF's spikes) through the synapse
            let ds = match &sv.r2.stp {
                Some(_) => crate::fwin_r2::stp_bwd(self, i, &sv.r2, &ds, &mut grads)?.reshape((n, d))?,
                None => ds,
            };
            let (dcur, dbeta) = match &sv.lif_ps {
                Some(ps) => {
                    // SLOW: lif_gen's backward kernel; the final state's gradients are 0 (no carried state)
                    let (btd, bd) = (n * d, b * d);
                    let gfl = Tensor::cat(&[&ds.flatten_all()?, &Tensor::zeros(2 * bd, DType::F32, dev)?], 0)?;
                    let pk = wo::lifg_bwd(&sv.cur, ps, &gfl, ar.spikes, ar.alif, ar.slope, ar.lif_reset_grad)?;
                    let part = |j: usize| pk.narrow(0, btd + (2 + j) * bd, bd)?.reshape((b, d))?.sum(0);
                    if let Some((r, g)) = bx(i).and_then(|e| e.alif.as_ref()) {
                        // rho = sigmoid(r): dr = drho rho (1 - rho); the gain as it is
                        let rho = ps.narrow(0, 2 * d, d)?;
                        grads.insert(r, part(1)?.mul(&rho)?.mul(&rho.affine(-1.0, 1.0)?)?);
                        grads.insert(g, part(2)?);
                    }
                    (pk.narrow(0, 0, btd)?.reshape((n, d))?, part(0)?)
                }
                None => {
                    let ds3 = ds.reshape((b, t, d))?;
                    let (dcur, dbeta) = if bf { lif_bwd_b(&sv.cur, &sv.beta, &ds3)? } else { lif_bwd_raw(&sv.cur, &sv.beta, &ds3)? };
                    (dcur.reshape((n, d))?, dbeta)
                }
            };
            // v51 fast1 M: the mood's gain (in f32): dL/d(the current after NE), dm and G's gradient
            let mood = sv.r2.mood.is_some();
            let dcur = if mood { crate::fwin_r2::mood_bwd(self, i, &r2f, &sv.r2, &dcur.to_dtype(DType::F32)?, &mut b2, &mut grads)? } else { dcur };
            let dcur = match (&sv.cur0, sg) {
                // v51 fused N: NE's gain: dL/d(the current before it) as the operand, and the signal's gradient
                (Some(c0), Some(nu)) => {
                    let (dc, ds) = nmo::rows_bwd(&dcur, c0, &nu.cols[NE], bf)?;
                    nmo::acc(&mut dsig[NE], ds)?;
                    dc
                }
                _ if bf && (sv.lif_ps.is_some() || mood) => mm.op(&dcur)?,
                _ => dcur,
            };
            let dh1 = mm.mm(&dcur, false, &sv.w[0], true)?;
            let dwin = mm.mm(&sv.h1, true, &dcur, false)?; // dL/d(Win diag(1 - beta)) with --lif-scale
            let (dwin, dbeta) = match &sv.omb {
                Some(omb) => {
                    let domb = m.win.detach().mul(&dwin)?.sum(0)?;
                    (dwin.broadcast_mul(omb)?, dbeta.sub(&domb)?)
                }
                None => (dwin, dbeta),
            };
            grads.insert(&m.win, dwin);
            // beta = sigmoid(param): dparam = dbeta * beta * (1 - beta)
            grads.insert(&m.beta, dbeta.mul(&sv.beta)?.mul(&sv.beta.affine(-1.0, 1.0)?)?);
            let (dxi, dg1) = rms_bwd_raw(&sv.x_in, &m.norm.detach(), &dh1, NORM_EPS)?;
            grads.insert(&m.norm, dg1);
            (dx, dx_m) = mm.add(&dx, &dxi)?;
            if let Some(r) = f2.block_back(i)? {
                // TH: the residual gate's share of the block's input; its gates into the relay, then the relay's read
                // of this input
                (dx, dx_m) = mm.add(&dx, &r)?;
            }
            if let Some(r) = f2.read_grad(i, &sv.x_in)? {
                (dx, dx_m) = mm.add(&dx, &r)?;
            }
        }
        if let (Some((wf, sa)), Some(ga)) = (&dense, &gall) {
            // the embedding is stream 0 too: its share of every mix
            dx = wo::dense_back(sa, ga, wf, &dx, 0, 0, 0, false)?.0.reshape((n, d))?;
        }
        // v51 fast1: M's scan and Y's rhythms into their weights; Y's share of the embedding's gradient
        if let Some(dxr) = crate::fwin_r2::finish(self, &r2f, b2, &saved[0].x_in, &mut grads)? {
            dx = dx.add(&dxr)?;
        }
        if let (Some(place), Some(pid)) = (&self.place, &pid) {
            grads.insert(place, crate::fused::place_bwd(pid, &dx, PLACES)?);
        }
        if let (Some(mw), Some(f)) = (&self.mag_w, &mfeat) {
            grads.insert(mw, f.t()?.contiguous()?.matmul(&dx)?);
        }
        f2.finish(&dx, &mut grads)?; // R's tables, TH's weights
        grads.insert(&self.tok, crate::fused::embed_bwd_add(&dtok_out, &idf, &dx)?);
        Ok(WinOut::Step(loss, act, grads))
    }
}

/// merge_f (runs/v51/merge_f.md): the ten parts with --neuromod 3 together, the CPU half (routing and the pieces).
#[cfg(test)]
#[path = "fwin_merge_tests.rs"]
mod merge_tests;
