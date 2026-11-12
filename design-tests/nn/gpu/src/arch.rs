// TESTING / DESIGN ONLY — ava_gpu::arch: v42 (Wave 2B) block options for model D, and the plain ops behind them.
//
// Every option defaults to today's D block. `Arch::default()` builds exactly the old model and runs the old code
// (the fused kernels and the hand-written training step). A model with any option on runs `Model::forward_arch`
// below with candle's autograd, on the CPU or the GPU: the fused kernels assume today's block, so the options use
// plain ops here (exact, but slower on the GPU; fast kernels for them are Wave 2C, O3). The one exception is
// --resid-fix, which only changes the init and keeps the old fast path.
//
// P1  --head-size N         the fast-weight memory's head size (32 today; attention and the engram stay at 32)
//     --timescales fixed    forget rates log-spaced over the heads between d128's extremes (1 - gamma from 2^-5 to
//                           2^-8), instead of gamma_h = 1 - 2^-(5+h) tied to the head number (`index`, today)
// P2  --gates channel       a forget rate and a write strength per key channel (a vector per head, not one number),
//                           and a separate erase key e = unit(k + h·We), as in Kimi Linear's KDA and RWKV-7:
//                           F_t = Diag(alpha)(F - (b*e)(e^T F)) + (b*k) v^T. The new parts start at 0, so the block
//                           starts as today's (alpha and b the same in every channel, e = k).
// P3  --value-residual      v_l <- (1 + p0) v_l + p1 v_0, v_0 the first memory block's values (p starts at 0)
// P4  --kwta F (0.125 today, or 0.25), --mlp-act relu2 (the kWTA winners' relu, squared)
// P5  --mem-momentum M      S_t = M S_{t-1} + U_t, F_t = alpha F_{t-1} + S_t (U_t today's write and erase)
// O1  --feedback            the window runs in chunks of 64 tokens; the top block's state at the end of a chunk,
//                           through a projection that starts at 0, is added to block 0's input for the next chunk.
//                           Every block carries its state from chunk to chunk (no engram with feedback).
// R1  --resid-fix           the init std of the weights that write into the stream counts D's 3 writes per block
//                           (4 in the engram block) instead of 2.  --residual dense: DenseFormer, the stream after
//                           block i is x_{i+1} + sum_j w_ij x_j over the earlier streams (w starts at 0: plain).
// R3  --lif-init logspaced  leak time constants log-spaced from 2 to 512 tokens over the channels (today all 0.9)
//     --lif-scale           the membrane input is scaled by (1 - beta)
//     --surrogate-slope S   ds/dv = 1 / (1 + S·|v - theta|)^2 (S = 4 today)
// E1  --spikes ternary|int2 signed integer spikes in one step: s = sign(v)·min(floor(|v|/theta), L), L = 1 or 2;
//                           reset by theta·s; the surrogate has one bump per step edge
// E4  --alif                an adaptive threshold: a second, slower trace a_t = rho a_{t-1} + (1 - rho)|s_{t-1}|
//                           raises it: theta_t = theta + g·a_t (rho starts at 0.98, g at 1; both learned)
// E8  --homeostasis RATE    a per-neuron threshold, moved after every optimizer step toward the target firing rate
//                           (theta += 0.01·(rate - RATE)), so dead neurons come back and busy ones calm down
// R2  --top-weight W        (train_gpu) Token Order Prediction: one extra unembedding head and a listwise ranking
//                           loss over the next 16 tokens (nearer tokens rank higher)
// v44 (two ideas from Moonshot's open Kimi models; both off by default):
// K1  --residual attn       Attention Residuals: each block's input (and the final stream) is a per-token softmax mix
//                           of the earlier sources v_0 = the embedding, v_j = block j-1's output minus its input:
//                           h = n · sum_i softmax_i(w · rms(v_i)) v_i, one learned query w [d] per block (used after
//                           it), keys rms(v) without a gain (a gain would only rescale w). Kimi's form times the
//                           number of sources n: w starts at 0, every weight is then 1/n and the mix is today's plain
//                           residual sum (up to rounding), the start closest to today's D (Kimi's own, a plain
//                           average, scales each block's input down by n against its in-block residual adds).
// K2  --gate sigmoid        the memory's output gate sigmoid(h·Wd·Wu) (Wd [d, R], Wu [R, d], --gate-rank R, 32 by
//     --gate-rank R         default) instead of silu(h·Wg) (Wg [d, d]), as in Kimi Linear. Wg's random numbers are still
//                           drawn (and dropped), so every other weight gets today's init for the same seed; Wd and Wu
//                           are made last (Model::new). Not a path option: a plain D model with it keeps the fused
//                           training step, inference kernel and one-token CPU step (lif.cu _sg kernels, fstep.rs); with
//                           other options it runs here (out_gate).
// v45 (four ideas from RWKV-7 and Qwen3-Next; all off by default, each starting exactly as today's block):
// W1  --gates kda           Kimi Delta Attention's order: fade per channel first, then erase and write with one key:
//                           F_t = (I - b k k^T) Diag(alpha) F + b k v^T (with --mem-momentum: U_t = b k (v - F^T(alpha*k))^T,
//                           S_t = M S + U_t, F_t = Diag(alpha) F + S_t). log alpha per (head, channel) as P2's (the same
//                           W_a1, W_a2 and offsets, W_a2 at 0); b = 2 sigmoid(h·w_b + b_b) one per head; erase key e = k.
//                           P2's W_b1's random numbers are drawn and dropped (W_b2 and W_e are not made), so the shared
//                           weights start as P2's. At W_a2 = 0 it is today's per-head D. --gates head is today's.
// W2  --bonus               RWKV-7's current-token bonus: o_t = headnorm(y_t) + ((q_t * rho) · k_t) v_t per head, before
//                           the output gate; rho [d] starts at 0.
// W3  --token-shift         RWKV's 2-tap mix: q, k, v and the gate input (forget, write and output gates; P2's erase key
//                           goes with k) read x_t = h_t + mu ⊙ (h_{t-1} - h_t), mu [4, d] at 0, h before the text 0. The
//                           talking path carries h_{t-1} in MemState::prev.
// W4  --eng-gate head|elem  Qwen's gated attention on the engram read: W_o(attn(h) ⊙ 2 sigmoid(h·W_th)), W_th [d, heads]
//                           (head) or [d, d] (elem) at 0, so the gate starts at 1. Needs --engram L.
//
// New tensors are made after all of today's, so the random init of today's weights is drawn in the same order.
// Token-by-token generation (Model::stream / step) does not know these options yet and refuses them.
// The Wave 5 parts (src/memory.rs, src/wave5.rs) enter `block_arch` too; see those files.

use crate::model::{
    head_norm, l2_norm, log_sigmoid, logit, mean_rows_first, norm, out_gate, rmsnorm, sigmoid, Activity, Block, Config, Hebb, Kind, Model, INIT_STD, LIF_SLOPE,
    LIF_THRESHOLD,
};
use candle_core::{DType, Device, Result, Tensor, Var, D};
use candle_nn::{Init, VarBuilder};
use std::cell::RefCell;

/// O1: the chunk length of the feedback loop.
pub const FB_CHUNK: usize = 64;
/// P2: the rank of the per-channel gate projections (h · W1 [d, R] · W2 [R, d]).
pub const GATE_RANK: usize = 32;
/// E8: how far one optimizer step moves a threshold per unit of rate error.
pub const HOMEO_STEP: f64 = 0.01;
/// E8: the thresholds stay in this range.
pub const HOMEO_RANGE: (f64, f64) = (0.05, 20.0);
/// E4: the adaptation trace's start (rho) and the threshold gain's start.
pub const ALIF_RHO0: f64 = 0.98;
pub const ALIF_GAIN0: f64 = 1.0;
/// R3: the log-spaced leak time constants span these many tokens.
pub const LIF_TAU: (f64, f64) = (2.0, 512.0);
/// P1: the fixed timescales: 1 - gamma from 2^-5 to 2^-8 (d128's four heads today).
pub const FIXED_TS: (f64, f64) = (5.0, 8.0);
/// R2: how many upcoming tokens the order loss ranks.
pub const TOP_WINDOW: usize = 16;
/// K2: the sigmoid output gate's default rank (today's head size).
pub const OUT_GATE_RANK: usize = 32;

/// K2: the fast-weight memory's output gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutGate {
    /// today's silu(h·Wg), Wg [d, d]
    Silu,
    /// sigmoid(h·Wd·Wu), rank `Arch::gate_rank`
    Sigmoid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Timescales {
    Index,
    Fixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gating {
    Head,
    Channel,
    /// v45 W1: per-channel fade first, then erase and write with one key (KDA)
    Kda,
}

/// v45 W4: the engram read's output gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngGate {
    Off,
    /// one gate per head (W_th [d, heads])
    Head,
    /// one gate per number (W_th [d, d])
    Elem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spikes {
    Binary,
    Ternary,
    Int2,
}

impl Spikes {
    /// How many spike levels each side of 0 (binary: one, positive only).
    pub fn levels(self) -> usize {
        match self {
            Spikes::Binary | Spikes::Ternary => 1,
            Spikes::Int2 => 2,
        }
    }
}

/// The block options (see the top of this file). `Arch::default()` is today's D.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arch {
    pub head_size: usize,
    pub timescales: Timescales,
    pub gates: Gating,
    pub value_residual: bool,
    /// kWTA keeps hidden / kwta_div units (8 today)
    pub kwta_div: usize,
    pub relu2: bool,
    pub mem_momentum: f32,
    pub feedback: bool,
    pub resid_fix: bool,
    pub dense: bool,
    pub lif_logspaced: bool,
    pub lif_scale: bool,
    pub slope: f32,
    pub spikes: Spikes,
    pub alif: bool,
    /// v52 spike fix (--lif-reset-grad): the surrogate backward also carries the gradient through the LIF reset
    /// (v -= s·θ); off = the reset detached (as before). Only the general scan (lif_gen / lifg kernels) has it.
    pub lif_reset_grad: bool,
    /// E8: the target firing rate (0 = off)
    pub homeostasis: f32,
    /// R2: the order head exists (train_gpu sets it when --top-weight > 0)
    pub top: bool,
    /// v42 Wave 2C: the memory options and the in-block parts (src/memory.rs); default all off
    pub mem: crate::memory::MemOpts,
    /// v44 K1: --residual attn (Attention Residuals)
    pub attn_res: bool,
    /// v44 K2: the memory's output gate and (sigmoid) its rank
    pub gate: OutGate,
    pub gate_rank: usize,
    /// v45 W2: RWKV-7's current-token bonus
    pub bonus: bool,
    /// v45 W3: RWKV's token shift on the memory's inputs
    pub token_shift: bool,
    /// v45 W4: the engram read's output gate
    pub eng_gate: EngGate,
    /// v50 (src/v50.rs, runs/v50/prereg_v50.md): Dale's law on the MLP's W2 (the E share of the hidden units; 0 = off),
    /// the number of bottom-up/top-down passes (1 = off) and the neuromodulator signals (0 = off, 3 = NE, ACh, DA)
    pub dale: f32,
    /// v50 Amendment 3 (D2): --dale-form proj: w = sign · v, v >= 0 by projection, E/I-balanced start (false = softplus)
    pub dale_proj: bool,
    pub loops: usize,
    pub neuromod: usize,
    /// v51 (src/v51.rs): round 2's parts P, R, DA, M, W, Pr, Y; default all off
    pub v51: crate::v51::V51Opts,
    /// v51 Part B2 (src/v51b.rs, runs/v51/prereg_v51.md): --thalamus and --cerebellum-head G
    pub v51b: crate::v51b::Opts,
}

// The float fields are checked finite by `validate`, so equality is an equivalence.
impl Eq for Arch {}

impl Default for Arch {
    fn default() -> Arch {
        Arch {
            head_size: crate::model::HEAD_SIZE,
            timescales: Timescales::Index,
            gates: Gating::Head,
            value_residual: false,
            kwta_div: crate::model::KWTA_DIV,
            relu2: false,
            mem_momentum: 0.0,
            feedback: false,
            resid_fix: false,
            dense: false,
            lif_logspaced: false,
            lif_scale: false,
            slope: LIF_SLOPE as f32,
            spikes: Spikes::Binary,
            alif: false,
            lif_reset_grad: false,
            homeostasis: 0.0,
            top: false,
            mem: crate::memory::MemOpts::default(),
            attn_res: false,
            gate: OutGate::Silu,
            gate_rank: OUT_GATE_RANK,
            bonus: false,
            token_shift: false,
            eng_gate: EngGate::Off,
            dale: 0.0,
            dale_proj: false,
            loops: 1,
            neuromod: 0,
            v51: crate::v51::V51Opts::default(),
            v51b: Default::default(),
        }
    }
}

pub const FLAG_USAGE: &str = "[--head-size 32] [--timescales index|fixed] [--gates head|channel|kda] [--value-residual] [--kwta 0.125|0.25]\n\
    \x20            [--mlp-act relu|relu2] [--mem-momentum 0] [--feedback] [--resid-fix] [--residual plain|dense|attn]\n\
    \x20            [--lif-init const|logspaced] [--lif-scale] [--surrogate-slope 4] [--spikes binary|ternary|int2] [--alif] [--lif-reset-grad]\n\
    \x20            [--homeostasis RATE] [--gate silu|sigmoid] [--gate-rank 32] [--bonus] [--token-shift] [--eng-gate off|head|elem]\n\
    \x20            [--dale FRAC] [--dale-form proj] [--loop N] [--neuromod 3]";

impl Arch {
    pub fn is_default(&self) -> bool {
        *self == Arch::default()
    }

    /// Whether the model needs the plain-op forward (`Model::forward_arch`). --resid-fix alone does not: it only
    /// changes the init.
    pub fn needs_path(&self) -> bool {
        // v44 K2: the sigmoid gate lives in the memory block itself and runs on every path (the fused ones too)
        Arch { resid_fix: false, gate: OutGate::Silu, gate_rank: OUT_GATE_RANK, ..*self } != Arch::default()
    }

    /// The slope the general scan's backward is given: negative asks it to carry the gradient through the reset.
    pub fn lif_bwd_slope(&self) -> f32 {
        if self.lif_reset_grad {
            -self.slope
        } else {
            self.slope
        }
    }

    /// The LIF can run the fused scan (binary spikes, one fixed threshold, no adaptation, no carried state).
    pub fn lif_fused_ok(&self) -> bool {
        self.spikes == Spikes::Binary && !self.alif && self.homeostasis == 0.0 && !self.feedback && !self.lif_reset_grad
    }

    /// One block flag at `args[i]`: how many arguments it used, or None when it is not a block flag.
    pub fn take_flag(&mut self, args: &[String], i: usize) -> std::result::Result<Option<usize>, String> {
        // v51 (src/v51.rs): round 2's flags
        if let Some(n) = self.v51.take_flag(args, i)? {
            return Ok(Some(n));
        }
        let key = args[i].as_str();
        match key {
            "--value-residual" => self.value_residual = true,
            "--feedback" => self.feedback = true,
            "--resid-fix" => self.resid_fix = true,
            "--lif-scale" => self.lif_scale = true,
            "--alif" => self.alif = true,
            "--lif-reset-grad" => self.lif_reset_grad = true,
            "--bonus" => self.bonus = true,
            "--token-shift" => self.token_shift = true,
            _ => {
                // v51 Part B2 (src/v51b.rs)
                if let Some(n) = crate::v51b::take_flag(&mut self.v51b, args, i)? {
                    return Ok(Some(n));
                }
                let known = [
                    "--head-size",
                    "--timescales",
                    "--gates",
                    "--kwta",
                    "--mlp-act",
                    "--mem-momentum",
                    "--residual",
                    "--lif-init",
                    "--surrogate-slope",
                    "--spikes",
                    "--homeostasis",
                    "--gate",
                    "--gate-rank",
                    "--eng-gate",
                    "--dale",
                    "--dale-form",
                    "--loop",
                    "--neuromod",
                ];
                if !known.contains(&key) {
                    // v42 Wave 2C (src/memory.rs)
                    return self.mem.take_flag(args, i);
                }
                let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value"))?.as_str();
                let fl = |v: &str| v.parse::<f64>().map_err(|_| format!("bad number {v:?} for {key}"));
                match key {
                    "--head-size" => self.head_size = v.parse().map_err(|_| format!("bad integer {v:?} for {key}"))?,
                    "--timescales" => {
                        self.timescales = match v {
                            "index" => Timescales::Index,
                            "fixed" => Timescales::Fixed,
                            _ => return Err(format!("{key} {v:?}: index or fixed")),
                        }
                    }
                    "--gates" => {
                        self.gates = match v {
                            "head" => Gating::Head,
                            "channel" => Gating::Channel,
                            "kda" => Gating::Kda,
                            _ => return Err(format!("{key} {v:?}: head, channel or kda")),
                        }
                    }
                    "--eng-gate" => {
                        self.eng_gate = match v {
                            "off" => EngGate::Off,
                            "head" => EngGate::Head,
                            "elem" => EngGate::Elem,
                            _ => return Err(format!("{key} {v:?}: off, head or elem")),
                        }
                    }
                    "--kwta" => {
                        let f = match v.split_once('/') {
                            Some((a, b)) => fl(a)? / fl(b)?,
                            None => fl(v)?,
                        };
                        if !(f > 0.0 && f <= 1.0) {
                            return Err(format!("{key} {v}: a fraction in (0, 1]"));
                        }
                        let div = (1.0 / f).round();
                        if (1.0 / div - f).abs() > 1e-9 {
                            return Err(format!("{key} {v}: must be 1/n (0.125 or 0.25)"));
                        }
                        self.kwta_div = div as usize;
                    }
                    "--mlp-act" => {
                        self.relu2 = match v {
                            "relu" => false,
                            "relu2" => true,
                            _ => return Err(format!("{key} {v:?}: relu or relu2")),
                        }
                    }
                    "--mem-momentum" => self.mem_momentum = fl(v)? as f32,
                    "--residual" => {
                        (self.dense, self.attn_res) = match v {
                            "plain" => (false, false),
                            "dense" => (true, false),
                            "attn" => (false, true),
                            _ => return Err(format!("{key} {v:?}: plain, dense or attn")),
                        }
                    }
                    "--gate" => {
                        self.gate = match v {
                            "silu" => OutGate::Silu,
                            "sigmoid" => OutGate::Sigmoid,
                            _ => return Err(format!("{key} {v:?}: silu or sigmoid")),
                        }
                    }
                    "--gate-rank" => self.gate_rank = v.parse().map_err(|_| format!("bad integer {v:?} for {key}"))?,
                    "--lif-init" => {
                        self.lif_logspaced = match v {
                            "const" => false,
                            "logspaced" => true,
                            _ => return Err(format!("{key} {v:?}: const or logspaced")),
                        }
                    }
                    "--surrogate-slope" => self.slope = fl(v)? as f32,
                    "--spikes" => {
                        self.spikes = match v {
                            "binary" => Spikes::Binary,
                            "ternary" => Spikes::Ternary,
                            "int2" => Spikes::Int2,
                            _ => return Err(format!("{key} {v:?}: binary, ternary or int2")),
                        }
                    }
                    "--homeostasis" => self.homeostasis = fl(v)? as f32,
                    // v50 (src/v50.rs)
                    "--dale" => self.dale = fl(v)? as f32,
                    "--dale-form" => {
                        self.dale_proj = match v {
                            "softplus" => false,
                            "proj" => true,
                            _ => return Err(format!("{key} {v:?}: softplus or proj")),
                        }
                    }
                    "--loop" => self.loops = v.parse().map_err(|_| format!("bad integer {v:?} for {key}"))?,
                    "--neuromod" => self.neuromod = v.parse().map_err(|_| format!("bad integer {v:?} for {key}"))?,
                    _ => unreachable!(),
                }
                return Ok(Some(2));
            }
        }
        Ok(Some(1))
    }

    /// The options that are not today's, as " key=value ..." (empty for the default): for resume signatures and logs.
    pub fn signature(&self) -> String {
        let z = Arch::default();
        let mut s = String::new();
        if self.head_size != z.head_size {
            s += &format!(" head_size={}", self.head_size);
        }
        if self.timescales != z.timescales {
            s += " timescales=fixed";
        }
        match self.gates {
            Gating::Head => {}
            Gating::Channel => s += " gates=channel",
            Gating::Kda => s += " gates=kda",
        }
        if self.value_residual {
            s += " value_residual=1";
        }
        if self.kwta_div != z.kwta_div {
            s += &format!(" kwta=1/{}", self.kwta_div);
        }
        if self.relu2 {
            s += " mlp_act=relu2";
        }
        if self.mem_momentum != 0.0 {
            s += &format!(" mem_momentum={}", self.mem_momentum);
        }
        if self.feedback {
            s += " feedback=1";
        }
        if self.resid_fix {
            s += " resid_fix=1";
        }
        if self.dense {
            s += " residual=dense";
        }
        if self.lif_logspaced {
            s += " lif_init=logspaced";
        }
        if self.lif_scale {
            s += " lif_scale=1";
        }
        if self.slope != z.slope {
            s += &format!(" surrogate_slope={}", self.slope);
        }
        match self.spikes {
            Spikes::Binary => {}
            Spikes::Ternary => s += " spikes=ternary",
            Spikes::Int2 => s += " spikes=int2",
        }
        if self.alif {
            s += " alif=1";
        }
        if self.lif_reset_grad {
            s += " lif_reset_grad=1";
        }
        if self.homeostasis != 0.0 {
            s += &format!(" homeostasis={}", self.homeostasis);
        }
        if self.attn_res {
            s += " residual=attn";
        }
        if self.gate == OutGate::Sigmoid {
            s += " gate=sigmoid";
        }
        if self.gate_rank != OUT_GATE_RANK {
            s += &format!(" gate_rank={}", self.gate_rank);
        }
        if self.bonus {
            s += " bonus=1";
        }
        if self.token_shift {
            s += " token_shift=1";
        }
        match self.eng_gate {
            EngGate::Off => {}
            EngGate::Head => s += " eng_gate=head",
            EngGate::Elem => s += " eng_gate=elem",
        }
        s += &self.mem.signature();
        s += &crate::v50::signature(self);
        s += &self.v51.signature();
        s += &crate::v51b::signature(self);
        if self.top {
            s += " top=1";
        }
        s
    }

    /// The options that are not today's as the flags that set them ("" for the default). train_gpu writes this line to
    /// DIR/arch.txt, so a tool that loads the model can build the same block with `from_flags`. The order head (R2)
    /// is left out: it is used only in training and does not change what the model computes.
    pub fn flags(&self) -> String {
        let z = Arch::default();
        let mut f: Vec<String> = Vec::new();
        if self.head_size != z.head_size {
            f.push(format!("--head-size {}", self.head_size));
        }
        if self.timescales != z.timescales {
            f.push("--timescales fixed".into());
        }
        match self.gates {
            Gating::Head => {}
            Gating::Channel => f.push("--gates channel".into()),
            Gating::Kda => f.push("--gates kda".into()),
        }
        if self.value_residual {
            f.push("--value-residual".into());
        }
        if self.kwta_div != z.kwta_div {
            f.push(format!("--kwta 1/{}", self.kwta_div));
        }
        if self.relu2 {
            f.push("--mlp-act relu2".into());
        }
        if self.mem_momentum != 0.0 {
            f.push(format!("--mem-momentum {}", self.mem_momentum));
        }
        if self.feedback {
            f.push("--feedback".into());
        }
        if self.resid_fix {
            f.push("--resid-fix".into());
        }
        if self.dense {
            f.push("--residual dense".into());
        }
        if self.lif_logspaced {
            f.push("--lif-init logspaced".into());
        }
        if self.lif_scale {
            f.push("--lif-scale".into());
        }
        if self.slope != z.slope {
            f.push(format!("--surrogate-slope {}", self.slope));
        }
        match self.spikes {
            Spikes::Binary => {}
            Spikes::Ternary => f.push("--spikes ternary".into()),
            Spikes::Int2 => f.push("--spikes int2".into()),
        }
        if self.alif {
            f.push("--alif".into());
        }
        if self.lif_reset_grad {
            f.push("--lif-reset-grad".into());
        }
        if self.homeostasis != 0.0 {
            f.push(format!("--homeostasis {}", self.homeostasis));
        }
        if self.attn_res {
            f.push("--residual attn".into());
        }
        if self.gate == OutGate::Sigmoid {
            f.push("--gate sigmoid".into());
        }
        if self.gate_rank != OUT_GATE_RANK {
            f.push(format!("--gate-rank {}", self.gate_rank));
        }
        if self.bonus {
            f.push("--bonus".into());
        }
        if self.token_shift {
            f.push("--token-shift".into());
        }
        match self.eng_gate {
            EngGate::Off => {}
            EngGate::Head => f.push("--eng-gate head".into()),
            EngGate::Elem => f.push("--eng-gate elem".into()),
        }
        f.extend(self.mem.flags());
        f.extend(crate::v50::flags(self));
        f.extend(self.v51.flags());
        f.extend(crate::v51b::flags(self));
        f.join(" ")
    }

    /// The options a `flags` line (or any run of block flags) sets; anything else in the line is refused.
    pub fn from_flags(line: &str) -> std::result::Result<Arch, String> {
        let args: Vec<String> = line.split_whitespace().map(String::from).collect();
        let mut a = Arch::default();
        let mut i = 0;
        while i < args.len() {
            match a.take_flag(&args, i)? {
                Some(n) => i += n,
                None => return Err(format!("{:?} is not a block flag", args[i])),
            }
        }
        Ok(a)
    }

    /// The options checked against the rest of the config.
    pub fn validate(&self, c: &Config) -> std::result::Result<(), String> {
        if c.kind.has_hebb() && (self.head_size == 0 || c.d % self.head_size != 0) {
            return Err(format!("--head-size {}: must divide d = {}", self.head_size, c.d));
        }
        if c.kind.is_ava() && (self.kwta_div == 0 || c.hidden() % self.kwta_div != 0) {
            return Err(format!("kWTA 1/{}: must divide the hidden width {}", self.kwta_div, c.hidden()));
        }
        if !(self.mem_momentum >= 0.0 && self.mem_momentum < 1.0) {
            return Err(format!("--mem-momentum {}: must be in [0, 1)", self.mem_momentum));
        }
        if !(self.slope > 0.0 && self.slope.is_finite()) {
            return Err(format!("--surrogate-slope {}: must be positive", self.slope));
        }
        if !(self.homeostasis >= 0.0 && self.homeostasis < 1.0) {
            return Err(format!("--homeostasis {}: a target rate in [0, 1)", self.homeostasis));
        }
        if self.dense && self.attn_res {
            return Err("--residual: dense or attn, not both".into());
        }
        if self.gate == OutGate::Sigmoid && (self.gate_rank == 0 || self.gate_rank > c.d) {
            return Err(format!("--gate-rank {}: from 1 to d = {}", self.gate_rank, c.d));
        }
        if self.gate_rank != OUT_GATE_RANK && self.gate != OutGate::Sigmoid {
            return Err(format!("--gate-rank {}: for --gate sigmoid only", self.gate_rank));
        }
        if self.gate == OutGate::Sigmoid && c.kind != Kind::D {
            return Err(format!("--gate sigmoid is for model D (this is {})", c.kind.name()));
        }
        if self.needs_path() && c.kind != Kind::D {
            return Err(format!("the block options{} are for model D (this is {})", self.signature(), c.kind.name()));
        }
        if self.eng_gate != EngGate::Off && c.engram.is_none() {
            return Err("--eng-gate gates the engram read: it needs --engram L".into());
        }
        self.mem.validate(c)?;
        crate::v50::validate(self, c)?;
        self.v51.validate(c)?;
        crate::v51b::validate(self, c)?;
        if self.feedback && c.engram.is_some() {
            return Err("--feedback runs the window in chunks and cannot carry the engram's store yet".into());
        }
        Ok(())
    }
}

/// The forget rates of the memory's `nh` heads: today's gamma_h = 1 - 2^-(5+h), or log-spaced between FIXED_TS.
pub fn gate_gammas(nh: usize, ts: Timescales) -> Vec<f64> {
    match ts {
        Timescales::Index => crate::model::hebb_gammas(nh),
        Timescales::Fixed => (0..nh)
            .map(|h| {
                let f = if nh > 1 { h as f64 / (nh - 1) as f64 } else { 0.0 };
                1.0 - 2f64.powf(-(FIXED_TS.0 + (FIXED_TS.1 - FIXED_TS.0) * f))
            })
            .collect(),
    }
}

/// R3: the leak parameters (logit beta) of `d` channels with time constants log-spaced over LIF_TAU.
pub fn logspaced_beta_params(d: usize) -> Vec<f32> {
    (0..d)
        .map(|c| {
            let f = if d > 1 { c as f64 / (d - 1) as f64 } else { 0.0 };
            let tau = LIF_TAU.0 * (LIF_TAU.1 / LIF_TAU.0).powf(f);
            logit(1.0 - 1.0 / tau) as f32
        })
        .collect()
}

/// K1 (--residual attn): the mix of the sources v_i, each [N, d]: n · sum_i a_i v_i with a = softmax over the sources
/// of q · rms(v_i) per row (rms without a gain), n the number of sources. q = 0: every a_i = 1/n, the plain sum.
pub fn attn_res_mix(srcs: &[Tensor], q: &Tensor) -> Result<Tensor> {
    let s = srcs.len();
    let v = Tensor::stack(srcs, 0)?; // [S, N, d]
    let (_, n, d) = v.dims3()?;
    let scores = head_norm(&v)?.reshape((s * n, d))?.matmul(&q.reshape((d, 1))?)?.reshape((s, n))?;
    let a = candle_nn::ops::softmax(&scores, 0)?; // [S, N]
    v.broadcast_mul(&a.unsqueeze(2)?)?.sum(0)?.affine(s as f64, 0.0)
}

/// v45 W3 (--token-shift): the four mixed inputs x_j = h + mu_j ⊙ (h_{t-1} - h) (j = q, k, v, gate) of the normed
/// input h [b·t, d] (rows token-major per text row), `prev` [b, d] the token before the window (None: zeros), and the
/// window's last token [b, d] (the next call's `prev`). mu [4, d]; mu = 0 gives h exactly.
pub fn token_shift(h: &Tensor, mu: &Tensor, b: usize, t: usize, prev: Option<&Tensor>) -> Result<([Tensor; 4], Tensor)> {
    let d = h.dim(1)?;
    let h3 = h.reshape((b, t, d))?;
    let first = match prev {
        Some(p) => p.reshape((b, 1, d))?,
        None => Tensor::zeros((b, 1, d), h.dtype(), h.device())?,
    };
    let hp = if t > 1 { Tensor::cat(&[&first, &h3.narrow(1, 0, t - 1)?], 1)? } else { first };
    let delta = hp.sub(&h3)?.reshape((b * t, d))?;
    let mix = |j: usize| -> Result<Tensor> { h.add(&delta.broadcast_mul(&mu.narrow(0, j, 1)?)?) };
    let last = h3.narrow(1, t - 1, 1)?.squeeze(1)?;
    Ok(([mix(0)?, mix(1)?, mix(2)?, mix(3)?], last))
}

/// v45 W4 (--eng-gate): the attention output o [n, d] (heads concatenated) times the gate g [n, w] (w = heads: one per
/// head, broadcast over its channels; w = d: one per number).
pub fn eng_gate_apply(o: &Tensor, g: &Tensor) -> Result<Tensor> {
    let (n, d) = o.dims2()?;
    let w = g.dim(1)?;
    if w == d {
        o.mul(g)
    } else {
        o.reshape((n, w, d / w))?.broadcast_mul(&g.reshape((n, w, 1))?)?.reshape((n, d))
    }
}

/// v45 W4: the gate 2·sigmoid(h·W_th) for the normed input h [n, d] (W_th = 0: exactly 1).
pub fn eng_gate_of(h: &Tensor, w: &Tensor) -> Result<Tensor> {
    sigmoid(&h.matmul(w)?)?.affine(2.0, 0.0)
}

/// v45 W2 (--bonus): y [b, nh, t, hs] (after the head norm) + ((q ⊙ rho) · k) v per head, rho [d].
pub fn bonus_add(y: &Tensor, q: &Tensor, k: &Tensor, v: &Tensor, rho: &Tensor) -> Result<Tensor> {
    let (_, nh, _, hs) = q.dims4()?;
    let s = q.broadcast_mul(&rho.reshape((1, nh, 1, hs))?)?.mul(k)?.sum_keepdim(3)?; // [b, nh, t, 1]
    y.add(&v.broadcast_mul(&s)?)
}

/// The per-channel gate parts of one memory block (P2).
pub(crate) struct ChanGates {
    pub wa1: Tensor,
    pub wa2: Tensor,
    pub wb1: Tensor,
    pub wb2: Tensor,
    pub we: Tensor,
    /// the forget offsets per channel [nh·hs] (each head's offset repeated)
    pub off: Tensor,
}

/// v45 W1 (--gates kda): the per-channel fade's parts, P2's names (W_a2 starts at 0).
pub(crate) struct KdaGates {
    pub wa1: Tensor,
    pub wa2: Tensor,
    /// the forget offsets per channel [nh·hs] (each head's offset repeated)
    pub off: Tensor,
}

/// One block's extra parts.
pub(crate) struct BlockExt {
    /// v45 W1 (--gates kda)
    pub kda: Option<KdaGates>,
    /// v45 W2 (--bonus): rho [d], starts at 0
    pub bonus: Option<Tensor>,
    /// v45 W3 (--token-shift): mu [4, d] for q, k, v and the gate input, starts at 0
    pub shift: Option<Tensor>,
    /// v45 W4 (--eng-gate, the engram's block only): W_th [d, heads] or [d, d], starts at 0
    pub eng_gate: Option<Tensor>,
    /// E8: the per-neuron threshold [d] (a variable without gradient, moved by `homeostasis_step`)
    pub theta: Option<Tensor>,
    /// E4: the trace's rho (as a logit) and the threshold gain, each [d]
    pub alif: Option<(Tensor, Tensor)>,
    pub chan: Option<ChanGates>,
    /// P3: (p0, p1) [2], blocks after the first
    pub vres: Option<Tensor>,
    /// R1 dense: the weights over the streams x_0 .. x_{i+1} [i + 2]
    pub dense: Option<Tensor>,
    /// v44 K1 (--residual attn): the query [d] that mixes the sources after this block (starts at 0)
    pub attn_res: Option<Tensor>,
}

/// A model's extra parts (Model::ext): Some exactly when `cfg.arch.needs_path()`.
pub struct ModelExt {
    pub(crate) blocks: Vec<BlockExt>,
    /// O1: the feedback projection [d, d], starting at 0
    pub(crate) fb: Option<Tensor>,
    /// R2: the order head [V, d]
    pub(crate) top: Option<Tensor>,
    /// E8: each block's per-neuron firing rate [d] in the last forward
    pub(crate) rates: RefCell<Vec<Option<Tensor>>>,
    /// v42 Wave 2C: the memory options' and the parts' weights (all None/empty when those are off)
    pub(crate) mem: crate::memory::MemExt,
    /// I3 (ReDo, src/upkeep.rs): Some only during a measurement forward; each block's [sum over rows of |spikes| per
    /// channel [d], of |MLP output| per unit [hidden]]
    pub(crate) redo_rec: RefCell<Option<Vec<[Option<Tensor>; 2]>>>,
    /// A13 (src/sae.rs): Some only while Model::record_states is on; each block's output state and xf
    pub(crate) state_rec: RefCell<Option<crate::sae::StateRec>>,
    /// F10 (src/introspect.rs): Some only while Model::set_injection is on; (block, vector [d] added to its output)
    pub(crate) inject: RefCell<Option<(usize, Tensor)>>,
    /// v50 (src/v50.rs): --dale, --loop and --neuromod's parts (all None when those are off)
    pub(crate) v50: crate::v50::V50Ext,
    /// v51 (src/v51.rs): round 2's parts (empty when they are off; made last, after every other weight)
    pub(crate) v51: crate::v51::V51Ext,
    /// v51 Part B2 (src/v51b.rs): --thalamus and --cerebellum-head's parts (None when off), and the online learning
    /// sessions (src/eprop.rs, the cerebellum's online table)
    pub(crate) v51b: crate::v51b::V51bExt,
}

/// I3: add |x| summed over rows into block i's slot k of the ReDo record (nothing unless a measurement is on).
fn redo_add(m: &Model, i: usize, k: usize, x: &Tensor) -> Result<()> {
    let Some(e) = m.ext.as_ref() else { return Ok(()) };
    let mut rec = e.redo_rec.borrow_mut();
    let Some(r) = rec.as_mut() else { return Ok(()) };
    add_opt(&mut r[i][k], x.detach().to_dtype(DType::F32)?.abs()?.sum(0)?)
}

/// Make the extra parts in `vb` (after every one of today's tensors, see the top of the file).
pub(crate) fn build_ext(cfg: &Config, vb: &VarBuilder) -> Result<Option<ModelExt>> {
    let ar = cfg.arch;
    if !ar.needs_path() {
        return Ok(None);
    }
    let (d, v, hs) = (cfg.d, cfg.vocab, ar.head_size);
    let nh = d / hs;
    let randn = |std: f64| Init::Randn { mean: 0.0, stdev: std };
    let mut blocks = Vec::with_capacity(cfg.layers);
    for i in 0..cfg.layers {
        let vm = vb.pp(format!("b{i}.mem"));
        let vh = vb.pp(format!("b{i}.hebb"));
        let theta = if ar.homeostasis > 0.0 { Some(vm.get_with_hints(d, "theta", Init::Const(LIF_THRESHOLD))?) } else { None };
        let alif = if ar.alif {
            Some((vm.get_with_hints(d, "arho", Init::Const(logit(ALIF_RHO0)))?, vm.get_with_hints(d, "again", Init::Const(ALIF_GAIN0))?))
        } else {
            None
        };
        let chan = if ar.gates == Gating::Channel {
            let off: Vec<f32> = gate_gammas(nh, ar.timescales).into_iter().flat_map(|g| std::iter::repeat(logit(g) as f32).take(hs)).collect();
            Some(ChanGates {
                wa1: vh.get_with_hints((d, GATE_RANK), "wa1", randn(INIT_STD))?,
                wa2: vh.get_with_hints((GATE_RANK, d), "wa2", Init::Const(0.0))?,
                wb1: vh.get_with_hints((d, GATE_RANK), "wb1", randn(INIT_STD))?,
                wb2: vh.get_with_hints((GATE_RANK, d), "wb2", Init::Const(0.0))?,
                we: vh.get_with_hints((d, d), "we", Init::Const(0.0))?,
                off: Tensor::from_vec(off, d, vb.device())?.to_dtype(vb.dtype())?,
            })
        } else {
            None
        };
        // v45 W1: P2's W_a1 and W_a2 under the same names and in the same place; W_b1's random numbers are drawn and
        // dropped (W_b2 and W_e start at 0 and draw none), so every later weight gets P2's init for the same seed
        let kda = if ar.gates == Gating::Kda {
            let off: Vec<f32> = gate_gammas(nh, ar.timescales).into_iter().flat_map(|g| std::iter::repeat(logit(g) as f32).take(hs)).collect();
            let wa1 = vh.get_with_hints((d, GATE_RANK), "wa1", randn(INIT_STD))?;
            let wa2 = vh.get_with_hints((GATE_RANK, d), "wa2", Init::Const(0.0))?;
            drop(randn(INIT_STD).var((d, GATE_RANK), DType::F32, vb.device())?);
            Some(KdaGates { wa1, wa2, off: Tensor::from_vec(off, d, vb.device())?.to_dtype(vb.dtype())? })
        } else {
            None
        };
        let vres = if ar.value_residual && i > 0 { Some(vh.get_with_hints(2, "vres", Init::Const(0.0))?) } else { None };
        let dense = if ar.dense { Some(vb.get_with_hints(i + 2, &format!("b{i}.dense"), Init::Const(0.0))?) } else { None };
        blocks.push(BlockExt { kda, bonus: None, shift: None, eng_gate: None, theta, alif, chan, vres, dense, attn_res: None });
    }
    let fb = if ar.feedback { Some(vb.get_with_hints((d, d), "fb.wo", Init::Const(0.0))?) } else { None };
    let top = if ar.top { Some(vb.get_with_hints((v, d), "top_emb", randn(INIT_STD))?) } else { None };
    // v42 Wave 2C: after every tensor above, so the options before it draw their init as they did
    let mem = crate::memory::build(cfg, vb)?;
    // v44 K1: after everything else, for the same reason (K2's gate is made after these, in Model::new)
    for (i, bx) in blocks.iter_mut().enumerate() {
        if ar.attn_res {
            bx.attn_res = Some(vb.get_with_hints(d, &format!("b{i}.ares"), Init::Const(0.0))?);
        }
    }
    // v45 W2-W4: after everything else (all start at 0 and draw no random numbers)
    for (i, bx) in blocks.iter_mut().enumerate() {
        let vh = vb.pp(format!("b{i}.hebb"));
        if ar.bonus {
            bx.bonus = Some(vh.get_with_hints(d, "bonus", Init::Const(0.0))?);
        }
        if ar.token_shift {
            bx.shift = Some(vh.get_with_hints((4, d), "shift", Init::Const(0.0))?);
        }
        if ar.eng_gate != EngGate::Off && cfg.engram == Some(i) {
            let w = if ar.eng_gate == EngGate::Head { d / crate::model::HEAD_SIZE } else { d };
            bx.eng_gate = Some(vb.get_with_hints((d, w), &format!("b{i}.eng.wth"), Init::Const(0.0))?);
        }
    }
    // v50: after everything else (constants only, no random draws)
    let v50 = crate::v50::build(cfg, vb)?;
    // v51: last of all (after v50's constants, so its random draws come in the order they had without v50)
    let v51 = crate::v51::build(cfg, vb)?;
    // v51b: after everything else (constants and seeded values only, no random draws, so v51's draws keep their order)
    let v51b = crate::v51b::build(cfg, vb)?;
    Ok(Some(ModelExt { blocks, fb, top, rates: RefCell::new(Vec::new()), mem, redo_rec: RefCell::new(None), state_rec: RefCell::new(None), inject: RefCell::new(None), v50, v51, v51b }))
}

/// The number of extra weights `build_ext` makes (for Config::weight_counts), and how many of them are the order
/// head (an unembedding table, counted with the embeddings).
pub fn extra_weights(cfg: &Config) -> (usize, usize) {
    let ar = cfg.arch;
    if !ar.needs_path() {
        return (0, 0);
    }
    let (d, l) = (cfg.d, cfg.layers);
    let mut n = 0;
    if ar.homeostasis > 0.0 {
        n += l * d;
    }
    if ar.alif {
        n += l * 2 * d;
    }
    if ar.gates == Gating::Channel {
        n += l * (4 * d * GATE_RANK + d * d);
    }
    if ar.value_residual {
        n += 2 * (l - 1);
    }
    if ar.dense {
        n += (0..l).map(|i| i + 2).sum::<usize>();
    }
    if ar.feedback {
        n += d * d;
    }
    if ar.attn_res {
        n += l * d;
    }
    // v45 W1-W4
    if ar.gates == Gating::Kda {
        n += l * 2 * d * GATE_RANK;
    }
    if ar.bonus {
        n += l * d;
    }
    if ar.token_shift {
        n += l * 4 * d;
    }
    match (ar.eng_gate, cfg.engram) {
        (EngGate::Head, Some(_)) => n += d * (d / crate::model::HEAD_SIZE),
        (EngGate::Elem, Some(_)) => n += d * d,
        _ => {}
    }
    n += ar.mem.extra_weights(cfg);
    n += crate::v50::extra_weights(cfg);
    n += ar.v51.extra_weights(cfg); // v51 (src/v51.rs)
    n += crate::v51b::extra_weights(cfg);
    let top = if ar.top { cfg.vocab * d } else { 0 };
    // src/wave5.rs: N1's vocabulary tables count with the embeddings too
    (n + top, top + ar.mem.extra_emb_weights(cfg))
}

// ------------------------------------------------------------------------------------------- the LIF, in plain ops

/// The LIF's carried state between chunks (O1): membrane v, last spikes s and (E4) the adaptation trace, each [B, d].
#[derive(Clone)]
pub struct LifState {
    pub v: Tensor,
    pub s: Tensor,
    pub a: Option<Tensor>,
}

/// u / (1 + S·|u|): the fast sigmoid whose slope is the surrogate gradient 1 / (1 + S·|u|)^2.
fn soft_step(u: &Tensor, slope: f64) -> Result<Tensor> {
    u.div(&u.abs()?.affine(slope, 1.0)?)
}

/// The LIF scan over the time axis of `cur` [B, T, d], with every option, in plain candle ops (autograd: the
/// straight-through estimator of model::lif_scan_unrolled). `beta` [d] the leak in (0, 1); `theta` [d] the
/// threshold (None: 1); `alif` (rho in (0, 1), gain), each [d]. Per step:
///     a = rho·a + (1 - rho)|s_prev| (ALIF; detached spikes),   th = theta + gain·a
///     v = beta·v + I - (th_prev·s_prev)        (the reset detached; a negative slope carries it, capped as lifg_te)
///     s = sum_{k=1..L} H(v - k·th) - H(-v - k·th)   (binary: H(v - th) only), surrogate per edge
/// With the defaults this is the fused scan of lif.rs (the same values; its backward up to rounding).
pub fn lif_ref(
    cur: &Tensor,
    beta: &Tensor,
    theta: Option<&Tensor>,
    alif: Option<(&Tensor, &Tensor)>,
    spikes: Spikes,
    slope: f64,
    init: Option<&LifState>,
) -> Result<(Tensor, LifState)> {
    // a negative slope: the gradient also flows through the reset v -= s·θ, its carry capped (--lif-reset-grad; the
    // trace stays detached)
    let rg = slope < 0.0;
    let slope = slope.abs();
    let (b, t, d) = cur.dims3()?;
    let dev = cur.device();
    let th = match theta {
        Some(x) => x.detach(),
        None => Tensor::full(LIF_THRESHOLD as f32, d, dev)?.to_dtype(cur.dtype())?,
    };
    let zeros = || Tensor::zeros((b, d), cur.dtype(), dev);
    let mut v = match init {
        Some(s) => s.v.clone(),
        None => zeros()?,
    };
    let mut sp = match init {
        Some(s) => s.s.detach(),
        None => zeros()?,
    };
    let mut a = match (alif, init.and_then(|s| s.a.as_ref())) {
        (Some(_), Some(x)) => Some(x.clone()),
        (Some(_), None) => Some(zeros()?),
        (None, _) => None,
    };
    // the threshold the last spikes crossed (for their reset)
    let mut th_prev = th.broadcast_as((b, d))?.contiguous()?;
    if let (Some((_, gain)), Some(a0)) = (alif, a.as_ref()) {
        th_prev = th_prev.add(&a0.broadcast_mul(gain)?)?;
    }
    let mut out = Vec::with_capacity(t);
    for step in 0..t {
        let i_t = cur.narrow(1, step, 1)?.squeeze(1)?; // [B, d]
        let th_t = match (alif, a.as_mut()) {
            (Some((rho, gain)), Some(a)) => {
                *a = a.broadcast_mul(rho)?.add(&sp.abs()?.detach().broadcast_mul(&rho.affine(-1.0, 1.0)?)?)?;
                th.broadcast_add(&a.broadcast_mul(gain)?)?
            }
            _ => th.broadcast_as((b, d))?.contiguous()?,
        };
        let reset = sp.mul(&th_prev)?;
        v = v.broadcast_mul(beta)?.add(&i_t)?.sub(&if rg { reset } else { reset.detach() })?;
        let s = match spikes {
            Spikes::Binary => {
                let u = v.sub(&th_t)?;
                let sg = soft_step(&u, slope)?;
                let hard = u.ge(0.0)?.to_dtype(u.dtype())?;
                sg.add(&hard.sub(&sg)?.detach())?
            }
            Spikes::Ternary | Spikes::Int2 => {
                let mut soft: Option<Tensor> = None;
                let mut hard: Option<Tensor> = None;
                for k in 1..=spikes.levels() {
                    let kth = th_t.affine(k as f64, 0.0)?;
                    let up = v.sub(&kth)?;
                    let down = v.neg()?.sub(&kth)?;
                    let s1 = soft_step(&up, slope)?.sub(&soft_step(&down, slope)?)?;
                    let h1 = up.ge(0.0)?.to_dtype(up.dtype())?.sub(&down.ge(0.0)?.to_dtype(up.dtype())?)?;
                    soft = Some(match soft {
                        None => s1,
                        Some(x) => x.add(&s1)?,
                    });
                    hard = Some(match hard {
                        None => h1,
                        Some(x) => x.add(&h1)?,
                    });
                }
                let (soft, hard) = (soft.expect("levels >= 1"), hard.expect("levels >= 1"));
                soft.add(&hard.sub(&soft)?.detach())?
            }
        };
        out.push(s.clone());
        sp = if rg {
            // the next reset carries th_t ds/dv capped at 2 beta (lif.cu's lifg_te): the spike's gradient into it is
            // scaled by min(1, 2 beta / (th_t ds/dv)), worked out detached (the cap passes no gradient)
            let (vd, thd) = (v.detach(), th_t.detach());
            let sur = |x: Tensor| -> Result<Tensor> { x.abs()?.affine(slope, 1.0)?.sqr()?.recip() };
            let dv = match spikes {
                Spikes::Binary => sur(vd.sub(&thd)?)?,
                Spikes::Ternary | Spikes::Int2 => {
                    let mut acc = vd.zeros_like()?;
                    for k in 1..=spikes.levels() {
                        let kth = thd.affine(k as f64, 0.0)?;
                        acc = acc.add(&sur(vd.sub(&kth)?)?.add(&sur(vd.neg()?.sub(&kth)?)?)?)?;
                    }
                    acc
                }
            };
            let cap = beta.detach().broadcast_as((b, d))?.affine(2.0, 0.0)?;
            let q = thd.mul(&dv)?;
            let sc = q.gt(&cap)?.where_cond(&cap.div(&q)?, &q.ones_like()?)?; // only where th ds/dv > 2 beta, as lifg_te
            let sd = s.detach();
            sd.add(&s.sub(&sd)?.mul(&sc)?)?
        } else {
            s.detach()
        };
        th_prev = th_t;
    }
    Ok((Tensor::stack(&out, 1)?, LifState { v, s: sp.detach(), a }))
}

// ------------------------------------------------------------------------------------------- the memory, recurrent

/// The fast-weight memory's carried state (O1): F [B·nh, hs, hs], (P5) the momentum S and (v45 W3, --token-shift) the
/// last token's normed input h [B, d] (None: the start of the text, zeros).
#[derive(Clone)]
pub struct MemState {
    pub f: Tensor,
    pub s: Option<Tensor>,
    pub prev: Option<Tensor>,
}

/// The memory token by token, with every option, in plain candle ops: q, k, e (erase key), v [B, nh, T, hs] (q, k, e
/// unit length), log alpha and the write strength b [B, nh, T, g] with g = 1 (per head) or hs (per channel):
///     U_t = (b*k) v^T - Diag(alpha)(b*e)(e^T F_{t-1})
///     S_t = M S_{t-1} + U_t   (P5; M = 0: S_t = U_t)
///     F_t = clamp(Diag(alpha) F_{t-1} + S_t, -cap, cap),   y_t = q_t F_t   (v52 fix_S: cap = mem_gen::cap(), MEM_CAP)
/// With per-head gates, e = k and M = 0 this is model::delta_rule (today's D), up to rounding. The CPU fallback for
/// the options the fused kernel lacks; the fast kernel is O3.
pub fn delta_rec(q: &Tensor, k: &Tensor, e: &Tensor, v: &Tensor, la: &Tensor, b: &Tensor, m: f64, init: Option<&MemState>) -> Result<(Tensor, MemState)> {
    let (bs, nh, t, hs) = q.dims4()?;
    let g = la.dim(3)?;
    let bn = bs * nh;
    let dev = q.device();
    let r = |x: &Tensor, w: usize| x.reshape((bn, t, w));
    let (q, k, e, v) = (r(q, hs)?, r(k, hs)?, r(e, hs)?, r(v, hs)?);
    let alpha = r(la, g)?.exp()?;
    let bw = r(b, g)?;
    let mut f = match init {
        Some(s) => s.f.clone(),
        None => Tensor::zeros((bn, hs, hs), q.dtype(), dev)?,
    };
    let mut sm = if m > 0.0 {
        Some(match init.and_then(|s| s.s.as_ref()) {
            Some(x) => x.clone(),
            None => Tensor::zeros((bn, hs, hs), q.dtype(), dev)?,
        })
    } else {
        None
    };
    let cap = crate::mem_gen::cap() as f64;
    let mut ys = Vec::with_capacity(t);
    for i in 0..t {
        let at = |x: &Tensor| x.narrow(1, i, 1); // [bn, 1, w]
        let (qi, ki, ei, vi) = (at(&q)?, at(&k)?, at(&e)?, at(&v)?);
        let ai = at(&alpha)?.transpose(1, 2)?; // [bn, g, 1]: one rate per key row
        let bi = at(&bw)?; // [bn, 1, g]
        let kb = ki.broadcast_mul(&bi)?.transpose(1, 2)?; // [bn, hs, 1]
        let eb = ei.broadcast_mul(&bi)?.transpose(1, 2)?;
        let pred = ei.matmul(&f)?; // [bn, 1, hs]: what the memory recalls for e
        let upd = kb.matmul(&vi)?.sub(&eb.matmul(&pred)?.broadcast_mul(&ai)?)?;
        let fa = f.broadcast_mul(&ai)?;
        f = match sm.as_mut() {
            Some(s) => {
                *s = s.affine(m, 0.0)?.add(&upd)?;
                fa.add(s)?
            }
            None => fa.add(&upd)?,
        };
        f = f.clamp(-cap, cap)?; // v52 fix_S: the bound on the state (mem_gen::MEM_CAP; lif.cu mg_cap)
        ys.push(qi.matmul(&f)?);
    }
    let y = Tensor::cat(&ys, 1)?.reshape((bs, nh, t, hs))?;
    Ok((y, MemState { f, s: sm, prev: None }))
}

/// O3 for P2 and P5: the same rule as `delta_rec` (same inputs, same result up to rounding) in chunks of c tokens
/// (v52 fix_S: but without the bound on the state, which a closed form per chunk cannot hold: the same while every
/// element stays below mem_gen::MEM_CAP, i.e. in every normal head; a runaway head can still overflow here)
/// instead of one token at a time. AVA_DELTA_REC=1 goes back to the token loop.
/// Inside a chunk (positions 0..c, rates as logs, cum = inclusive sums, Gamma_i = exp(cum_i)), with the write key
/// w = b*k, the erase key a = alpha*b*e and the per-channel kernel K_ij = sum_{m=j..i} M^(m-j) exp(cum_i - cum_m):
///     F_i = Diag(Gamma_i) F0 + Diag(M K_i0) S0 + sum_{j<=i} Diag(K_ij) (w_j v_j^T - a_j r_j^T),  r_j = e_j^T F_{j-1}
///     S_i = M^(i+1) S0 + sum_{j<=i} M^(i-j) (w_j v_j^T - a_j r_j^T)
/// The recalls R solve one unit lower-triangular system, (I + Ae) R = (e*Gamma_ex) F0 + (e*M Kp_0) S0 + Ke V, with
/// Kp the kernel one row back. Then Y = (q*Gamma) F0 + (q*M K_0) S0 + Qw V - Qa R. Here Qw_ij = sum_ch q_i K_ij w_j,
/// and Qa, Ke, Ae are formed the same way. Every decay is a ratio <= 1, formed pair by pair (no division by Gamma),
/// so strong forgetting cannot overflow. Per-channel gates cost [c, c, hs] per chunk, hence c <= 16; per-head
/// gates (g = 1) need only [c, c], so c <= 64.
pub fn delta_chunk(q: &Tensor, k: &Tensor, e: &Tensor, v: &Tensor, la: &Tensor, b: &Tensor, m: f64, init: Option<&MemState>) -> Result<(Tensor, MemState)> {
    let (bs, nh, t, hs) = q.dims4()?;
    let g = la.dim(3)?;
    let bn = bs * nh;
    let c = chunk_len(t, g);
    let nc = t / c;
    let ch = |x: &Tensor, w: usize| x.reshape((bn, nc, c, w));
    let (q, k, e, v) = (ch(q, hs)?, ch(k, hs)?, ch(e, hs)?, ch(v, hs)?);
    let la = ch(la, g)?;
    let bw = ch(b, g)?;
    let w = k.broadcast_mul(&bw)?; // write key
    let ah = e.broadcast_mul(&bw.mul(&la.exp()?)?)?; // erase key alpha*b*e
    chunk_core(&q, &e, &v, &la, &w, &ah, m, init, bs)
}

/// The chunk length of `delta_chunk` for t tokens and g gates per head.
fn chunk_len(t: usize, g: usize) -> usize {
    let mut c = if g == 1 { 64 } else { 16 };
    while t % c != 0 {
        c /= 2;
    }
    c
}

/// v45 W1 (--gates kda): KDA's rule token by token in plain ops: q, k, v [B, nh, T, hs] (q, k unit length), log alpha
/// [B, nh, T, hs] (per channel) and the write strength b [B, nh, T, 1] (per head):
///     r_t = (alpha*k)^T F_{t-1},  U_t = b k (v - r)^T,  S_t = M S_{t-1} + U_t (P5),  F_t = Diag(alpha) F_{t-1} + S_t (or + U_t)
///     y_t = q_t F_t
/// i.e. F_t = (I - b k k^T) Diag(alpha) F_{t-1} + b k v^T without momentum: the fade first, then erase and write with
/// the same key, so reading k right after a write with b = 1 gives v exactly. Works in any float type (f64 for the
/// finite-difference test).
pub fn kda_rec(q: &Tensor, k: &Tensor, v: &Tensor, la: &Tensor, b: &Tensor, m: f64, init: Option<&MemState>) -> Result<(Tensor, MemState)> {
    let (bs, nh, t, hs) = q.dims4()?;
    let bn = bs * nh;
    let (dev, dt) = (q.device(), q.dtype());
    let r = |x: &Tensor, w: usize| x.reshape((bn, t, w));
    let (q, k, v) = (r(q, hs)?, r(k, hs)?, r(v, hs)?);
    let alpha = r(la, hs)?.exp()?;
    let bw = r(b, 1)?;
    let mut f = match init {
        Some(s) => s.f.clone(),
        None => Tensor::zeros((bn, hs, hs), dt, dev)?,
    };
    let mut sm = if m > 0.0 {
        Some(match init.and_then(|s| s.s.as_ref()) {
            Some(x) => x.clone(),
            None => Tensor::zeros((bn, hs, hs), dt, dev)?,
        })
    } else {
        None
    };
    let cap = crate::mem_gen::cap() as f64;
    let mut ys = Vec::with_capacity(t);
    for i in 0..t {
        let at = |x: &Tensor| x.narrow(1, i, 1); // [bn, 1, w]
        let (qi, ki, vi, ai) = (at(&q)?, at(&k)?, at(&v)?, at(&alpha)?);
        let pred = ki.mul(&ai)?.matmul(&f)?; // [bn, 1, hs]: what the faded memory recalls for k
        let kb = ki.broadcast_mul(&at(&bw)?)?.transpose(1, 2)?; // [bn, hs, 1]
        let upd = kb.matmul(&vi.sub(&pred)?)?;
        let fa = f.broadcast_mul(&ai.transpose(1, 2)?)?;
        f = match sm.as_mut() {
            Some(s) => {
                *s = s.affine(m, 0.0)?.add(&upd)?;
                fa.add(s)?
            }
            None => fa.add(&upd)?,
        };
        f = f.clamp(-cap, cap)?; // v52 fix_S: the bound on the state (mem_gen::MEM_CAP; lif.cu mg_cap)
        ys.push(qi.matmul(&f)?);
    }
    let y = Tensor::cat(&ys, 1)?.reshape((bs, nh, t, hs))?;
    Ok((y, MemState { f, s: sm, prev: None }))
}

/// v45 W1: `kda_rec` in chunks (the same inputs and result up to rounding): `delta_chunk`'s algebra with the write and
/// erase keys both b*k and the read key alpha*k. AVA_DELTA_REC=1 goes back to the token loop.
pub fn kda_chunk(q: &Tensor, k: &Tensor, v: &Tensor, la: &Tensor, b: &Tensor, m: f64, init: Option<&MemState>) -> Result<(Tensor, MemState)> {
    let (bs, nh, t, hs) = q.dims4()?;
    let bn = bs * nh;
    let c = chunk_len(t, hs);
    let nc = t / c;
    let ch = |x: &Tensor, w: usize| x.reshape((bn, nc, c, w));
    let (q, k, v) = (ch(q, hs)?, ch(k, hs)?, ch(v, hs)?);
    let la = ch(la, hs)?;
    let w = k.broadcast_mul(&ch(b, 1)?)?; // write and erase key b*k
    let er = k.mul(&la.exp()?)?; // read key alpha*k
    chunk_core(&q, &er, &v, &la, &w, &w, m, init, bs)
}

/// The chunked memory (see `delta_chunk`) on chunk-shaped inputs [bn, nc, c, w]: q, the read key e, v, log alpha, the
/// write key w and the erase key ah (bs rows).
#[allow(clippy::too_many_arguments)]
fn chunk_core(q: &Tensor, e: &Tensor, v: &Tensor, la: &Tensor, w: &Tensor, ah: &Tensor, m: f64, init: Option<&MemState>, bs: usize) -> Result<(Tensor, MemState)> {
    let (bn, nc, c, hs) = q.dims4()?;
    let (nh, t, g, dev) = (bn / bs, nc * c, la.dim(3)?, q.device());
    let (q, e, v, la, w, ah) = (q.clone(), e.clone(), v.clone(), la.clone(), w.clone(), ah.clone());
    let cum = la.cumsum(2)?; // [bn, nc, c, g]
    let cumex = cum.sub(&la)?; // the sums before each position
    // D_im = exp(cum_i - cum_m) on/below the diagonal, 0 above: [bn, nc, c(i), c(m), g]
    let mask = crate::model::causal_mask(c, dev)?.to_dtype(q.dtype())?.reshape((c, c, 1))?;
    let dm = cum.unsqueeze(3)?.broadcast_sub(&cum.unsqueeze(2)?)?.broadcast_add(&mask)?.exp()?;
    let mom = m > 0.0;
    let kk = if mom {
        // K_ij = sum_m D_im Mt_mj with Mt_mj = M^(m-j) for m >= j: one matmul over m
        let mt: Vec<f32> = (0..c * c).map(|n| if n / c >= n % c { m.powi((n / c - n % c) as i32) as f32 } else { 0.0 }).collect();
        let mt = Tensor::from_vec(mt, (c, c), dev)?.to_dtype(q.dtype())?;
        dm.transpose(3, 4)?.contiguous()?.reshape((bn * nc * c * g, c))?.matmul(&mt)?.reshape((bn, nc, c, g, c))?.transpose(3, 4)?.contiguous()?
    } else {
        dm
    };
    let kp = Tensor::cat(&[&Tensor::zeros((bn, nc, 1, c, g), q.dtype(), dev)?, &kk.narrow(2, 0, c - 1)?], 2)?; // K_{i-1, j}
    // sum over channels of x_i K_ij y_j for two right-hand keys at once
    let pair2 = |x: &Tensor, kk: &Tensor, y1: &Tensor, y2: &Tensor| -> Result<(Tensor, Tensor)> {
        if g == 1 {
            let kk = kk.squeeze(4)?;
            Ok((x.matmul(&y1.transpose(2, 3)?.contiguous()?)?.mul(&kk)?, x.matmul(&y2.transpose(2, 3)?.contiguous()?)?.mul(&kk)?))
        } else {
            let xk = x.unsqueeze(3)?.broadcast_mul(kk)?;
            Ok((xk.broadcast_mul(&y1.unsqueeze(2)?)?.sum(4)?, xk.broadcast_mul(&y2.unsqueeze(2)?)?.sum(4)?))
        }
    };
    let (qw, qa) = pair2(&q, &kk, &w, &ah)?; // [bn, nc, c, c]
    let (ke, ae) = pair2(&e, &kp, &w, &ah)?;
    let tinv = crate::model::inv_unit_lower(&ae, c)?;
    let p1 = tinv.matmul(&e.broadcast_mul(&cumex.exp()?)?)?; // R's F0 term [bn, nc, c, hs]
    let r0 = tinv.matmul(&ke.matmul(&v)?)?;
    let qg = q.broadcast_mul(&cum.exp()?)?;
    let qwv = qw.matmul(&v)?;
    let klast = kk.narrow(2, c - 1, 1)?.squeeze(2)?; // K_{c-1, j} [bn, nc, c, g]
    let wlv = w.broadcast_mul(&klast)?.transpose(2, 3)?.contiguous()?.matmul(&v)?; // [bn, nc, hs, hs]
    let al = ah.broadcast_mul(&klast)?.transpose(2, 3)?.contiguous()?; // [bn, nc, hs, c]
    let gc = cum.narrow(2, c - 1, 1)?.exp()?.transpose(2, 3)?; // Gamma_c as row scales [bn, nc, g, 1]
    // momentum terms (P5)
    let ms = if mom {
        let gs = kk.narrow(3, 0, 1)?.squeeze(3)?.affine(m, 0.0)?; // M K_i0 [bn, nc, c, g]
        let gsx = kp.narrow(3, 0, 1)?.squeeze(3)?.affine(m, 0.0)?;
        let mu: Vec<f32> = (0..c).map(|p| m.powi((c - 1 - p) as i32) as f32).collect();
        let mu = Tensor::from_vec(mu, (1, 1, c, 1), dev)?.to_dtype(q.dtype())?;
        let wm = w.broadcast_mul(&mu)?.transpose(2, 3)?.contiguous()?.matmul(&v)?;
        let am = ah.broadcast_mul(&mu)?.transpose(2, 3)?.contiguous()?;
        Some((tinv.matmul(&e.broadcast_mul(&gsx)?)?, q.broadcast_mul(&gs)?, gs.narrow(2, c - 1, 1)?.transpose(2, 3)?, wm, am))
    } else {
        None
    };
    let mut f = match init {
        Some(s) => s.f.clone(),
        None => Tensor::zeros((bn, hs, hs), q.dtype(), dev)?,
    };
    let mut sm = if mom {
        Some(match init.and_then(|s| s.s.as_ref()) {
            Some(x) => x.clone(),
            None => Tensor::zeros((bn, hs, hs), q.dtype(), dev)?,
        })
    } else {
        None
    };
    let mc = m.powi(c as i32);
    let mut ys = Vec::with_capacity(nc);
    for r in 0..nc {
        let at = |x: &Tensor| -> Result<Tensor> { x.narrow(1, r, 1)?.squeeze(1) };
        let mut rr = at(&p1)?.matmul(&f)?.add(&at(&r0)?)?;
        if let (Some((p2, _, _, _, _)), Some(s)) = (ms.as_ref(), sm.as_ref()) {
            rr = rr.add(&at(p2)?.matmul(s)?)?;
        }
        let mut y = at(&qg)?.matmul(&f)?.add(&at(&qwv)?)?.sub(&at(&qa)?.matmul(&rr)?)?;
        let mut fnew = f.broadcast_mul(&at(&gc)?)?.add(&at(&wlv)?)?.sub(&at(&al)?.matmul(&rr)?)?;
        if let (Some((_, qs, gsc, wm, am)), Some(s)) = (ms.as_ref(), sm.as_mut()) {
            y = y.add(&at(qs)?.matmul(s)?)?;
            fnew = fnew.add(&s.broadcast_mul(&at(gsc)?)?)?;
            *s = s.affine(mc, 0.0)?.add(&at(wm)?)?.sub(&at(am)?.matmul(&rr)?)?;
        }
        ys.push(y);
        f = fnew;
    }
    let y = Tensor::cat(&ys, 1)?.reshape((bs, nh, t, hs))?;
    Ok((y, MemState { f, s: sm, prev: None }))
}

/// AVA_DELTA_REC=1: P2 and P5 use the token loop `delta_rec` instead of `delta_chunk` (for checks).
pub fn delta_rec_on() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| std::env::var("AVA_DELTA_REC").is_ok_and(|v| v == "1"))
}

// ------------------------------------------------------------------------------------------- R2 order targets

/// Token Order Prediction targets for `targets` [B, T] (row-major; IGNORE or ids >= vocab are not tokens): for each
/// position, the distinct tokens among its next `w` targets (its own target first), scored w - distance + 1 at their
/// nearest place, and the softmax of those scores as the target distribution. Returned as `w` (token, weight) slots
/// per position, [B·T·w] each; unused slots are (0, 0), and a position with no upcoming token has all zeros.
pub fn top_targets(targets: &[u32], b: usize, t: usize, vocab: usize, w: usize) -> (Vec<u32>, Vec<f32>) {
    let mut idx = vec![0u32; b * t * w];
    let mut p = vec![0f32; b * t * w];
    for bi in 0..b {
        for ti in 0..t {
            let base = (bi * t + ti) * w;
            let mut toks: Vec<(u32, f64)> = Vec::with_capacity(w);
            for dist in 1..=w {
                let j = ti + dist - 1;
                if j >= t {
                    break;
                }
                let tok = targets[bi * t + j];
                if (tok as usize) >= vocab || toks.iter().any(|x| x.0 == tok) {
                    continue;
                }
                toks.push((tok, (w - dist + 1) as f64));
            }
            if toks.is_empty() {
                continue;
            }
            let mx = toks.iter().map(|x| x.1).fold(f64::MIN, f64::max);
            let z: f64 = toks.iter().map(|x| (x.1 - mx).exp()).sum();
            for (k, (tok, s)) in toks.iter().enumerate() {
                idx[base + k] = *tok;
                p[base + k] = ((s - mx).exp() / z) as f32;
            }
        }
    }
    (idx, p)
}

// ------------------------------------------------------------------------------------------- the forward

/// Activity sums of one forward over the chunks.
struct Stats {
    spikes: Vec<Option<Tensor>>,
    kwta: Vec<Option<Tensor>>,
    rates: Vec<Option<Tensor>>,
}

/// AVA_NANCHECK=1 (debugging): print where a block's values first go non-finite or very large.
pub(crate) fn nan_check(tag: &str, i: usize, t: &Tensor) -> Result<()> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("AVA_NANCHECK").is_ok()) {
        return Ok(());
    }
    let t = t.to_dtype(DType::F32)?;
    let (m, s) = (t.abs()?.max_all()?.to_scalar::<f32>()?, t.sum_all()?.to_scalar::<f32>()?);
    if !m.is_finite() || !s.is_finite() || m > 1e4 {
        eprintln!("nancheck block {i} {tag}: max |x| {m}, sum {s}");
    }
    Ok(())
}

fn add_opt(slot: &mut Option<Tensor>, x: Tensor) -> Result<()> {
    *slot = Some(match slot.take() {
        None => x,
        Some(s) => s.add(&x)?,
    });
    Ok(())
}

impl Model {
    /// The final normed hidden state [B·T, d] of `ids` [B, T] (logits = it · tok_emb^T) and the activity, for a model
    /// with block options (and, under `set_force_arch`, today's model, which it computes as `forward` does).
    pub(crate) fn forward_arch(&self, ids: &Tensor) -> Result<(Tensor, Activity)> {
        let cfg = self.cfg;
        let ar = cfg.arch;
        if self.blocks.iter().any(|b| b.attn.is_some()) {
            candle_core::bail!("forward_arch: model D only");
        }
        let (b, t) = ids.dims2()?;
        let (d, n, nb) = (cfg.d, b * t, self.blocks.len());
        // src/wave5.rs: `main` is false inside --modulate's shadow pass (no modulators, nothing stashed)
        let w5 = self.ext.as_ref().map(|e| &e.mem.w5);
        let main = w5.map_or(true, |w| !w.mod_off.get());
        if main && self.train_fwd.get() {
            if let Some(w) = w5 {
                w.clear_stash();
            }
        }
        let msig = match w5 {
            Some(w) if ar.mem.modulate && main => Some(self.mod_shadow(ids, w)?),
            _ => None,
        };
        let fast = crate::fast_path() && ids.device().is_cuda();
        let mut x = if fast { crate::fused::embedding(&self.tok, ids)? } else { self.tok.embedding(&ids.flatten_all()?)? };
        let (place_off, mag_off, _) = self.lesion.get();
        if let (Some(place), false) = (&self.place, place_off) {
            let pid = crate::fused::place_ids(ids)?;
            x = x.add(&Model::place_table(place)?.embedding(&pid)?)?;
        }
        if let (Some(mw), false) = (&self.mag_w, mag_off) {
            x = x.add(&crate::fused::mag_feats(ids)?.to_dtype(mw.dtype())?.matmul(mw)?)?;
        }
        let x = crate::v51::embed(self, ids, x)?; // v51 R: the letters (x itself without --letters)
        crate::v51::begin(self, &x, b, t, None)?; // v51 Y: the rhythms
        let ext = self.ext.as_ref();
        let chunk = if ar.feedback { FB_CHUNK.min(t) } else { t };
        // v42 (src/arch/ckpt.rs, AVA_CKPT=1): each block recomputed in the backward instead of keeping its inner values
        // (not with carried state, nor with --recall-loss, whose read keeps its own graph; v50: nor with its options,
        // whose signals live between the blocks; nor with v51's or v51b's parts)
        let ck = main && chunk == t && !ar.feedback && ar.mem.recall_loss == 0.0 && !crate::v50::any(&ar) && !ar.v51.any() && !crate::v51b::any(&ar) && ckpt::on();
        // v50 --loop: the passes over the window (1 without it); activity and rates average over all of them
        let np = self.v50_passes();
        if ck {
            ckpt::check_merged()?;
        }
        let mut lif_st: Vec<Option<LifState>> = vec![None; nb];
        let mut mem_st: Vec<Option<MemState>> = vec![None; nb];
        let mut st = Stats { spikes: vec![None; nb], kwta: vec![None; nb], rates: vec![None; nb] };
        let x3 = if chunk < t { Some(x.reshape((b, t, d))?) } else { None };
        let mut outs: Vec<Tensor> = Vec::new();
        let mut fb: Option<Tensor> = None; // [b, d]
        let mut t0 = 0;
        while t0 < t {
            let tc = chunk.min(t - t0);
            let mut xc = match &x3 {
                Some(x3) => x3.narrow(1, t0, tc)?.contiguous()?.reshape((b * tc, d))?,
                None => x.clone(),
            };
            if let Some(f) = &fb {
                xc = xc.reshape((b, tc, d))?.broadcast_add(&f.unsqueeze(1)?)?.reshape((b * tc, d))?;
            }
            // v50 (--loop N, src/v50.rs): the blocks run N times on the same input; from the second pass on, the
            // last pass's top output (normed, through loop.wo) is added to it. One pass without the option.
            let x_in = xc.clone();
            for pass in 0..np {
                if pass > 0 {
                    xc = self.v51b_th_td(&x_in, self.v50_loop_input(&x_in, &xc)?)?; // v51b --thalamus: the relay's top-down gate
                }
                let last = pass + 1 == np;
                self.v50_begin_pass(last);
                self.v51b_th_begin(&xc)?; // v51b --thalamus: the relay reads the blocks' input
                let mut streams: Vec<Tensor> = if ar.dense || ar.attn_res { vec![xc.clone()] } else { Vec::new() };
                let mut v0: Option<Tensor> = None;
                for (i, blk) in self.blocks.iter().enumerate() {
                    let bx = ext.map(|e| &e.blocks[i]);
                    let xin = xc.clone(); // K1: the block's input (its output minus this is the block's source)
                    self.v51b_th_block(i)?; // v51b --thalamus: this block's gates
                    xc = if ck {
                        self.block_ck(i, &xc, b, tc, &mut v0, &mut st, msig.as_ref())?
                    } else {
                        self.block_arch(i, blk, bx, &xc, b, tc, &mut lif_st[i], &mut mem_st[i], &mut v0, &mut st, msig.as_ref())?
                    };
                    xc = self.v51b_th_out(i, &xin, xc)?; // v51b --thalamus: the residual gate
                    if !main {
                        // --modulate's shadow pass: its result is detached, so no block's graph needs to be kept
                        xc = xc.detach();
                    }
                    if let Some(w) = bx.and_then(|e| e.dense.as_ref()) {
                        // DenseFormer: x_{i+1} + sum_j w_j x_j (w = 0: the plain stream, exactly)
                        streams.push(xc.clone());
                        let mut mix = xc.clone();
                        for (j, sj) in streams.iter().enumerate() {
                            mix = mix.add(&sj.broadcast_mul(&w.narrow(0, j, 1)?)?)?;
                        }
                        xc = mix;
                    }
                    if let Some(q) = bx.and_then(|e| e.attn_res.as_ref()) {
                        // K1: Attention Residuals (q = 0: the plain stream, up to rounding)
                        streams.push(xc.sub(&xin)?);
                        xc = attn_res_mix(&streams, q)?;
                    }
                    if last {
                        xc = crate::introspect::inject_block(self, i, xc)?; // F10: nothing unless Model::set_injection is on
                        if main {
                            crate::sae::rec_block(self, i, &xc)?; // A13: nothing unless Model::record_states is on
                        }
                    }
                    self.v51b_th_after(i, &xc)?; // v51b --thalamus: the relay reads this block's output
                    if i == 0 {
                        self.v50_after_block0(&xc)?; // v50 --neuromod: the nucleus reads block 0's output
                        crate::v51::after0(self, &xc, ids, b, tc, None)?; // v51 DA, M: the surprise and the mood
                    }
                }
            }
            self.v50_end();
            self.v51b_th_end();
            if let Some(wf) = ext.and_then(|e| e.fb.as_ref()) {
                // O1: the top state at the chunk's last token, normed, through the feedback projection
                let top = xc.reshape((b, tc, d))?.narrow(1, tc - 1, 1)?.squeeze(1)?;
                fb = Some(rmsnorm(&top, &self.final_norm)?.matmul(wf)?);
            }
            outs.push(xc);
            t0 += tc;
        }
        crate::v51::end(self);
        let xo = if outs.len() == 1 {
            outs.pop().expect("one chunk")
        } else {
            let parts: Vec<Tensor> = outs.iter().map(|o| o.reshape((b, o.dim(0)? / b, d))).collect::<Result<_>>()?;
            Tensor::cat(&parts, 1)?.reshape((n, d))?
        };
        let mut act = Activity { spike_rate: Vec::new(), kwta_post: Vec::new() };
        if self.track.get() {
            for i in 0..nb {
                if let Some(s) = st.spikes[i].take() {
                    act.spike_rate.push(s.affine(1.0 / (n * np) as f64, 0.0)?);
                }
                if let Some(k) = st.kwta[i].take() {
                    act.kwta_post.push(k.affine(1.0 / (n * np) as f64, 0.0)?);
                }
            }
        }
        if let Some(e) = ext {
            if ar.homeostasis > 0.0 && !w5.is_some_and(|w| w.fast.get()) {
                let rates: Vec<Option<Tensor>> = st.rates.into_iter().map(|r| r.map(|r| r.affine(1.0 / (n * np) as f64, 0.0)).transpose()).collect::<Result<_>>()?;
                *e.rates.borrow_mut() = rates;
            }
        }
        let xf = norm(&xo, &self.final_norm)?;
        let xf = self.v51b_cb(xf, ids)?; // v51b --cerebellum-head: its correction to her state (xf itself without it)
        if main {
            crate::sae::rec_final(self, b, &xf)?;
        }
        if let Some(w) = w5.filter(|w| main && self.train_fwd.get() && w.wants_xf()) {
            // src/wave5.rs: the auxiliary heads read the final state (Model::aux_terms takes it)
            w.stash.borrow_mut().xf = Some((xf.clone(), b, t));
            if ar.mem.reflex > 0.0 {
                w.stash.borrow_mut().ids = Some(ids.clone()); // H3b: for the one-pass forward
            }
        }
        Ok((xf, act))
    }

    /// B8/F13 (--modulate): the shadow pass (this model with the modulators off, activity not tracked) and the
    /// signals from its surprise.
    fn mod_shadow(&self, ids: &Tensor, w: &crate::wave5::W5Ext) -> Result<crate::wave5::ModSig> {
        w.mod_off.set(true);
        let tr = self.track.replace(false);
        let r = self.forward_arch(ids);
        w.mod_off.set(false);
        self.track.set(tr);
        let (xs, _) = r?;
        let logits = crate::int8::head(&xs, &self.tok)?.detach();
        crate::wave5::mod_signals(&logits, ids, None)
    }

    /// src/wave5.rs: keep the logits of a training forward (detached) for the critic and the confidence probe.
    pub(crate) fn stash_logits(&self, logits: &Tensor) {
        if let Some(w) = self.ext.as_ref().map(|e| &e.mem.w5) {
            if self.train_fwd.get() && !w.mod_off.get() && w.wants_logits() {
                w.stash.borrow_mut().logits = Some(logits.detach());
            }
        }
    }

    /// One block on the rows of a chunk (b rows of t tokens), carrying its state when the window runs in chunks.
    #[allow(clippy::too_many_arguments)]
    fn block_arch(
        &self,
        i: usize,
        blk: &Block,
        bx: Option<&BlockExt>,
        x: &Tensor,
        b: usize,
        t: usize,
        lif: &mut Option<LifState>,
        mem: &mut Option<MemState>,
        v0: &mut Option<Tensor>,
        st: &mut Stats,
        msig: Option<&crate::wave5::ModSig>,
    ) -> Result<Tensor> {
        let ar = self.cfg.arch;
        let (d, n) = (self.cfg.d, b * t);
        let carry = ar.feedback;
        // src/wave5.rs parts of this block, and B8/F13's modulators for this read (None in the shadow pass)
        let w5 = self.ext.as_ref().map(|e| &e.mem.w5);
        let main = w5.map_or(true, |w| !w.mod_off.get());
        // v49: forward_trace's sink (src/model.rs): copies of what this block computed, never fed back
        let tr = main && self.tracing() && !self.v50_quiet(); // v50 --loop: the last pass only
        if tr {
            self.trace_begin(i);
        }
        crate::v51::at_block(self, i);
        let wb = w5.and_then(|w| w.blocks.get(i));
        let mods = match (wb.and_then(|w| w.modl.as_ref()), msig) {
            (Some(md), Some(sg)) => Some(crate::wave5::block_mods(md, sg)?),
            _ => None,
        };
        let xm;
        let x = match &mods {
            // F13: mood and arousal colour the block's input (w starts at 0)
            Some((_, add)) => {
                xm = x.add(add)?;
                &xm
            }
            None => x,
        };
        let m = blk.mem.as_ref().expect("D block has SpikeMem");
        let h = norm(x, &m.norm)?;
        let mut cur = crate::int8::mm(&h, &m.win)?.reshape((b, t, d))?;
        let beta = sigmoid(&m.beta)?;
        if ar.lif_scale {
            cur = cur.broadcast_mul(&beta.affine(-1.0, 1.0)?)?;
        }
        cur = self.v50_ne(cur, b, t)?; // v50 --neuromod NE: the gain on the input drive (blocks after 0)
        if let Some((md, _)) = &mods {
            // B8 acetylcholine-like: the weight of new input
            cur = cur.broadcast_mul(&md.input.unsqueeze(2)?)?;
        }
        cur = crate::v51::lif_gain(self, i, cur)?; // v51 M: the mood's gain
        // v42 Wave 2C parts (src/memory.rs): dendrites gate and delays shift the input current, stp acts on the spikes
        let px = self.ext.as_ref().and_then(|e| e.mem.parts.get(i));
        if let Some(dn) = px.and_then(|p| p.dend.as_ref()) {
            cur = cur.mul(&dn.gate(&h.reshape((b, t, d))?)?)?;
        }
        if let Some(dl) = px.and_then(|p| p.delays.as_ref()) {
            cur = dl.forward(&cur)?;
        }
        let s = if ar.lif_fused_ok() && cur.dtype() == DType::F32 {
            crate::lif::lif_slope(&cur, &beta, ar.slope)?
        } else {
            let theta = bx.and_then(|e| e.theta.as_ref());
            let alif = match bx.and_then(|e| e.alif.as_ref()) {
                Some((r, g)) => Some((sigmoid(r)?, g.clone())),
                None => None,
            };
            // v42: the fused general scan (src/lif_gen.rs, the same spikes as lif_ref); AVA_LIF_REF=1 runs lif_ref
            let scan = if crate::lif_gen::on() { crate::lif_gen::lif_gen } else { lif_ref };
            let (s, ls) = scan(&cur, &beta, theta, alif.as_ref().map(|(r, g)| (r, g)), ar.spikes, ar.lif_bwd_slope() as f64, lif.as_ref())?;
            if carry {
                *lif = Some(ls);
            }
            s
        };
        nan_check("current", i, &cur)?;
        nan_check("spikes", i, &s)?;
        let s = match wb.and_then(|w| w.res.as_ref()) {
            // E7: the resonators on the same current, mixed in (m starts at 0)
            Some(rs) => crate::wave5::res_mix(&s, &crate::scan_fast::res_forward(&rs.part, &cur, LIF_THRESHOLD, ar.slope as f64)?, &rs.mix)?,
            None => s,
        };
        let s = match px.and_then(|p| p.stp.as_ref()) {
            Some(sp) => crate::scan_fast::stp_forward(sp, &s)?,
            None => s,
        };
        nan_check("spikes after resonate/stp", i, &s)?;
        let s2 = s.reshape((n, d))?;
        if self.track.get() {
            // the mean |spike| (the rate for binary spikes), summed over the chunks' rows
            add_opt(&mut st.spikes[i], mean_rows_first(&s2.detach().abs()?)?.affine(n as f64, 0.0)?)?;
        }
        if ar.homeostasis > 0.0 {
            add_opt(&mut st.rates[i], s2.detach().abs()?.sum(0)?)?;
        }
        if main {
            redo_add(self, i, 0, &s2)?; // I3: what reaches wout, per channel
        }
        if tr {
            self.trace_put(|p| p.spikes = Some(s2.detach())); // v49: what reaches wout
        }
        let mut x = x.add(&crate::int8::mm(&s2, &m.wout)?)?;
        if let Some(hb) = &blk.hebb {
            let h = norm(&x, &hb.norm)?;
            let md = mods.as_ref().map(|(md, _)| (&md.write, &md.reset));
            let y = self.mem_arch(&h, hb, bx, b, t, mem, v0, carry, md)?;
            crate::v51::put_ctx(self, &y); // v51 P: the dendrites' context
            nan_check("memory read", i, &y)?;
            if tr {
                self.trace_put(|p| p.hebb_read = Some(y.detach()));
            }
            x = x.add(&self.v51b_th_mem(y)?)?; // v51b --thalamus: the memory path's gate (y itself without it)
        }
        let (_, _, eng_off) = self.lesion.get();
        if let (Some(e), false) = (&blk.eng, eng_off) {
            let h = norm(&x, &e.norm)?;
            let mo = ar.mem;
            // v45 W4: the engram's output gate (None without --eng-gate)
            let eg = bx.and_then(|e| e.eng_gate.as_ref()).map(|w| eng_gate_of(&h, w)).transpose()?;
            if mo.eng_any() || mo.recall_loss > 0.0 {
                // v42 Wave 2C: the engram read with its options (R4, L3), and its output kept for --recall-loss (O2)
                let mx = self.ext.as_ref().map(|e| &e.mem);
                let r = crate::memory::read(&h, e, mx.and_then(|m| m.eng.as_ref()), &mo, b, t, None, Default::default(), false, eg.as_ref())?.expect("not quiet");
                if mo.recall_loss > 0.0 && self.train_fwd.get() {
                    if let Some(m) = mx {
                        *m.recall.borrow_mut() = Some(r.clone());
                    }
                }
                if tr {
                    self.trace_put(|p| p.engram_read = Some(r.detach()));
                }
                x = x.add(&r)?;
            } else {
                let r = self.engram_read_g(&h, e, b, t, eg.as_ref())?;
                if tr {
                    // v49: the plain read's attention weights, on the side from the same normed input
                    let w = crate::model::attn_weights(&h, e, b, t)?;
                    self.trace_put(|p| {
                        p.engram_attn = Some(w);
                        p.engram_read = Some(r.detach());
                    });
                }
                x = x.add(&r)?;
            }
        }
        if let Some((nm, wm)) = px.and_then(|p| p.wm.as_ref()) {
            // B9: the gated working memory (its read-out starts at 0)
            let h = norm(&x, nm)?;
            x = x.add(&crate::scan_fast::wm_forward(wm, &h.reshape((b, t, d))?)?.reshape((n, d))?)?;
        }
        if let Some((nm, ws)) = wb.and_then(|w| w.ws.as_ref()) {
            // B3: the shared workspace (only in wave5::ws_block; its read-out starts at 0)
            let h = norm(&x, nm)?.reshape((b, t, d))?;
            if let (Some(sc), Some(w)) = (wb.and_then(|w| w.schema.as_ref()), w5.filter(|_| main && self.train_fwd.get())) {
                // F12: the schema learns to guess the competition (it only watches)
                let l = crate::wave5::schema_loss(sc, ws, &h)?.affine(ar.mem.schema as f64, 0.0)?;
                w.stash.borrow_mut().extra.push(l);
            }
            x = x.add(&crate::scan_fast::wm_forward(ws, &h)?.reshape((n, d))?)?;
        }
        if let Some((nm, hm)) = wb.and_then(|w| w.holo.as_ref()) {
            // D2: the holographic binding memory (its read-out starts at 0)
            let h = norm(&x, nm)?.reshape((b, t, d))?;
            x = x.add(&hm.forward(&h)?.reshape((n, d))?)?;
        }
        nan_check("after memory/engram/wm", i, &x)?;
        if let Some(tp) = wb.and_then(|w| w.tem.as_ref()) {
            // B10: the structure read (its output starts at 0)
            let h = norm(&x, &tp.norm)?;
            x = x.add(&crate::wave5::tem_read(&h, tp, b, t, None, None)?)?;
        }
        nan_check("after tem", i, &x)?;
        if let Some((nm, sy)) = wb.and_then(|w| w.sync.as_ref()) {
            // L5: binding by synchrony, one token at a time (its read-out starts at 0)
            x = x.add(&sy.forward(&norm(&x, nm)?)?)?;
        }
        let l = &blk.mlp;
        let x = if let Some(hl) = wb.and_then(|w| w.halt.as_ref()) {
            // A7: the MLP step pondered over up to N passes (the first pass's activity is the one tracked)
            // H3b: the one-pass (habit) forward of --reflex takes the first pass only
            let passes = if w5.is_some_and(|w| w.fast.get()) { 1 } else { ar.mem.halting };
            let (y, a, p, _) = crate::wave5::halt_mlp(self, l, hl, &x, passes)?;
            if tr {
                // v49: the first pass's units; no mask of its own here, so its winners are its non-zero units
                let w = a.detach().ne(0f32)?.to_dtype(DType::F32)?;
                self.trace_put(|q| {
                    q.winners = Some(w);
                    q.hidden = Some(a.detach());
                });
            }
            if self.track.get() {
                add_opt(&mut st.kwta[i], mean_rows_first(&a.detach().gt(0.0)?.to_dtype(DType::F32)?)?.affine(n as f64, 0.0)?)?;
            }
            if main {
                redo_add(self, i, 1, &a)?; // I3: the first pass's units
            }
            if let Some(w) = w5.filter(|_| main && self.train_fwd.get()) {
                w.stash.borrow_mut().halt.push(p);
                if ar.mem.topo > 0.0 {
                    let tl = crate::wave5::topo_loss(w, &a)?.affine(ar.mem.topo as f64 / self.cfg.layers as f64, 0.0)?;
                    w.stash.borrow_mut().extra.push(tl);
                }
            }
            y
        } else if ar.v51.mlp() {
            // v51 P, DA, W, Pr (src/v51.rs); `collect`: Pr's use statistics, from training forwards only
            let collect = main && self.train_fwd.get();
            let (a, winners, y) = crate::v51::mlp(self, i, &norm(&x, &l.norm)?, l, tr, collect)?.expect("a v51 MLP option is on");
            if tr {
                self.trace_put(|p| {
                    p.winners = winners;
                    p.hidden = Some(a.detach());
                });
            }
            if self.track.get() {
                add_opt(&mut st.kwta[i], mean_rows_first(&a.detach().gt(0.0)?.to_dtype(DType::F32)?)?.affine(n as f64, 0.0)?)?;
            }
            if main {
                redo_add(self, i, 1, &a)?;
            }
            x.add(&y)?
        } else {
            let h = norm(&x, &l.norm)?;
            let u = crate::int8::mm(&h, &l.w1)?;
            let k = self.cfg.kwta_k();
            let mut winners = None; // v49: the kWTA mask, kept for forward_trace only
            let mut a = if crate::fast_path() && u.device().is_cuda() {
                if tr {
                    winners = Some(crate::model::kwta_mask(&u, k)?.detach()); // on the side: the fused op makes its own
                }
                crate::fused::kwta_relu(&u, k)?
            } else {
                let wm = crate::model::kwta_mask(&u, k)?;
                let a = u.mul(&wm)?.relu()?;
                if tr {
                    winners = Some(wm.detach());
                }
                a
            };
            if ar.relu2 {
                a = a.sqr()?;
            }
            if tr {
                self.trace_put(|p| {
                    p.winners = winners;
                    p.hidden = Some(a.detach()); // what reaches W2
                });
            }
            if self.track.get() {
                add_opt(&mut st.kwta[i], mean_rows_first(&a.detach().gt(0.0)?.to_dtype(DType::F32)?)?.affine(n as f64, 0.0)?)?;
            }
            if main {
                redo_add(self, i, 1, &a)?; // I3: the units after kWTA (and relu2)
            }
            if let Some(w) = w5.filter(|_| main && self.train_fwd.get() && ar.mem.topo > 0.0) {
                // H12: this block's spatial loss (its share of the weight), for wave5::aux_loss
                let tl = crate::wave5::topo_loss(w, &a)?.affine(ar.mem.topo as f64 / self.cfg.layers as f64, 0.0)?;
                w.stash.borrow_mut().extra.push(tl);
            }
            // v50: --dale's effective W2 (today's w2 without it) and --neuromod DA on the output
            x.add(&self.v50_da(crate::int8::mm(&a, &self.mlp_w2(i)?)?)?)?
        };
        nan_check("after mlp/halting", i, &x)?;
        // H11: the cerebellum's correction (its read-out starts at 0)
        let x = match (wb.and_then(|w| w.cb.as_ref()), w5.and_then(|w| w.ones.as_ref())) {
            (Some(cb), Some(ones)) => crate::wave5::cb_add(cb, ones, &x)?,
            _ => x,
        };
        // A9: the product-key memory after this block's MLP (its values start at 0)
        match self.ext.as_ref().and_then(|e| e.mem.pkm.as_ref()) {
            Some(pk) if ar.mem.pk_block(self.cfg.layers) == Some(i) => x.add(&pk.forward(&x)?),
            _ => Ok(x),
        }
    }

    /// The fast-weight memory's read for the normed input h [n, d] with the block options (today's `hebb` for D when
    /// every memory option is off).
    #[allow(clippy::too_many_arguments)]
    /// `md`: B8's write and reset modulators [b, t] (src/wave5.rs), None without --modulate.
    pub(crate) fn mem_arch(
        &self,
        h: &Tensor,
        hb: &Hebb,
        bx: Option<&BlockExt>,
        b: usize,
        t: usize,
        state: &mut Option<MemState>,
        v0: &mut Option<Tensor>,
        carry: bool,
        md: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let ar = self.cfg.arch;
        let (d, n, hs) = (self.cfg.d, b * t, ar.head_size);
        let nh = d / hs;
        let heads = |x: &Tensor| -> Result<Tensor> { x.reshape((b, t, nh, hs))?.transpose(1, 2)?.contiguous() }; // [b, nh, t, hs]
        // v45 W3 (--token-shift): q, k, v and the gates read their own mix of h and the token before (mu = 0: h)
        let (hq, hk, hv, hg, last) = match bx.and_then(|e| e.shift.as_ref()) {
            Some(mu) => {
                let prev = state.as_ref().and_then(|s| s.prev.clone());
                let ([a, c, e, g], last) = token_shift(h, mu, b, t, prev.as_ref())?;
                (a, c, e, g, Some(last))
            }
            None => (h.clone(), h.clone(), h.clone(), h.clone(), None),
        };
        let h = &hg; // the gate input (the forget, write and output gates)
        let q = l2_norm(&heads(&crate::int8::mm(&hq, &hb.wq)?)?)?;
        let kraw = heads(&crate::int8::mm(&hk, &hb.wk)?)?;
        let k = l2_norm(&kraw)?;
        let mut v = heads(&crate::int8::mm(&hv, &hb.wv)?)?;
        if ar.value_residual {
            match (v0.as_ref(), bx.and_then(|e| e.vres.as_ref())) {
                (None, _) => *v0 = Some(v.clone()),
                (Some(v0t), Some(p)) => {
                    // (1 + p0) v + p1 v0: with p = 0 the values exactly
                    v = v.broadcast_mul(&p.narrow(0, 0, 1)?.affine(1.0, 1.0)?)?.add(&v0t.broadcast_mul(&p.narrow(0, 1, 1)?)?)?;
                }
                (Some(_), None) => {}
            }
        }
        let g = hb.gates.as_ref().expect("D has gates");
        let za = h.matmul(&g.wa)?.broadcast_add(&g.ba)?; // [n, nh]
        let zb = h.matmul(&g.wb)?.broadcast_add(&g.bb)?;
        let per_head = |x: Tensor, w: usize| -> Result<Tensor> { x.reshape((b, t, nh, w))?.transpose(1, 2)?.contiguous() }; // [b, nh, t, w]
        let chan = bx.and_then(|e| e.chan.as_ref());
        let kda = bx.and_then(|e| e.kda.as_ref());
        let (la, beta, e) = match (chan, kda) {
            (None, Some(c)) => {
                // v45 W1: P2's per-channel fade, today's per-head write strength, e = k
                let za = za.reshape((n, nh, 1))?.broadcast_add(&c.off.reshape((1, nh, hs))?)?.add(&h.matmul(&c.wa1)?.matmul(&c.wa2)?.reshape((n, nh, hs))?)?;
                let la = log_sigmoid(&za.reshape((n, d))?)?;
                let beta = sigmoid(&zb)?.affine(2.0, 0.0)?;
                (per_head(la, hs)?, per_head(beta, 1)?, k.clone())
            }
            (None, None) => {
                let la = log_sigmoid(&za.broadcast_add(&g.off)?)?;
                let beta = sigmoid(&zb)?.affine(2.0, 0.0)?;
                (per_head(la, 1)?, per_head(beta, 1)?, k.clone())
            }
            (Some(c), _) => {
                let za = za.reshape((n, nh, 1))?.broadcast_add(&c.off.reshape((1, nh, hs))?)?.add(&h.matmul(&c.wa1)?.matmul(&c.wa2)?.reshape((n, nh, hs))?)?;
                let zb = zb.reshape((n, nh, 1))?.broadcast_add(&h.matmul(&c.wb1)?.matmul(&c.wb2)?.reshape((n, nh, hs))?)?;
                let la = log_sigmoid(&za.reshape((n, d))?)?;
                let beta = sigmoid(&zb.reshape((n, d))?)?.affine(2.0, 0.0)?;
                // the erase key goes with k (v45 W3: the key's input)
                let e = l2_norm(&kraw.add(&heads(&crate::int8::mm(&hk, &c.we)?)?)?)?;
                (per_head(la, hs)?, per_head(beta, hs)?, e)
            }
        };
        let (la, beta) = match md {
            // B8: dopamine-like write strength and noradrenaline-like reset (both exactly as given at the start)
            Some((w, r)) => crate::wave5::mod_gates(la, beta, w, r)?,
            None => (la, beta),
        };
        let beta = crate::v51::write_gate(self, beta)?; // v51 Y: the rhythms gate the write strength
        // (v51's gate first, so v50's [0, 2] clamp, the delta rule's stable range, comes last when both are on)
        let beta = self.v50_ach(beta)?; // v50 --neuromod ACh: the write strength (blocks after 0), kept in [0, 2]
        if self.tracing() && !self.v50_quiet() && self.ext.as_ref().map_or(true, |e| !e.mem.w5.mod_off.get()) {
            // v49 (forward_trace): the gates used, [n, nh·g] head-major, on the side
            let rows = |x: &Tensor| -> Result<Tensor> {
                let g = x.dim(3)?;
                x.detach().transpose(1, 2)?.contiguous()?.reshape((n, nh * g))
            };
            let (keep, write) = (rows(&la.exp()?)?, rows(&beta)?);
            self.trace_put(|p| {
                p.hebb_keep = Some(keep);
                p.hebb_write = Some(write);
            });
        }
        let y = if kda.is_some() {
            // v45 W1: the fused general memory's KDA form on CUDA (src/mem_gen.rs), else the chunked plain ops
            if !carry && crate::mem_gen::on() && h.device().is_cuda() && crate::mem_gen::supports(hs) {
                crate::mem_gen::mem_gen_kda(&q, &k, &v, &la, &beta, ar.mem_momentum as f64)?
            } else {
                let run = if delta_rec_on() { kda_rec } else { kda_chunk };
                let (y, ms) = run(&q, &k, &v, &la, &beta, ar.mem_momentum as f64, state.as_ref())?;
                if carry {
                    *state = Some(ms);
                }
                y
            }
        } else if chan.is_none() && ar.mem_momentum == 0.0 {
            let (la3, b3) = (la.squeeze(3)?, beta.squeeze(3)?); // [b, nh, t]
            if ar.mem.mem_reset > 0 && !carry && self.train_fwd.get() {
                // O2: in training forwards the fast memory forgets at every mem_reset-token boundary
                crate::memory::delta_reset(&q, &k, &v, &la3, &b3, ar.mem.mem_reset)?
            } else if !carry && crate::delta_on() && h.device().is_cuda() && hs == crate::fused::ATTN_HS {
                let vg = Tensor::cat(&[&v, &la3.unsqueeze(3)?, &b3.unsqueeze(3)?], 3)?;
                crate::fused::delta(&q, &k, &vg)?
            } else if !carry && crate::mem_gen::on() && h.device().is_cuda() && crate::mem_gen::supports(hs) {
                // v42 (P1, head size 64): the general fused memory with e = k, per-head gates, no momentum
                crate::mem_gen::mem_gen(&q, &k, &k, &v, &la, &beta, 0.0)?
            } else {
                let f0 = state.as_ref().map(|s| s.f.clone());
                let (y, f) = crate::model::delta_rule_from(&q, &k, &v, &la3, &b3, f0.as_ref())?;
                if carry {
                    *state = Some(MemState { f, s: None, prev: None });
                }
                y
            }
        } else {
            let run = if delta_rec_on() { delta_rec } else { delta_chunk };
            if !carry && crate::mem_gen::on() && h.device().is_cuda() && crate::mem_gen::supports(hs) {
                // v42 (P2, P5): the general fused memory (src/mem_gen.rs), the same rule as delta_rec
                crate::mem_gen::mem_gen(&q, &k, &e, &v, &la, &beta, ar.mem_momentum as f64)?
            } else {
                let (y, ms) = run(&q, &k, &e, &v, &la, &beta, ar.mem_momentum as f64, state.as_ref())?;
                if carry {
                    *state = Some(ms);
                }
                y
            }
        };
        if carry {
            // v45 W3: the window's last input is the next window's token before (after any state the memory set)
            if let Some(s) = state.as_mut() {
                s.prev = last;
            }
        }
        let mut y = crate::v51::read_gate(self, head_norm(&y)?)?; // v51 Y: and the read
        if let Some(rho) = bx.and_then(|e| e.bonus.as_ref()) {
            // v45 W2: the current token's bonus (rho = 0: nothing added)
            y = bonus_add(&y, &q, &k, &v, rho)?;
        }
        let y = y.transpose(1, 2)?.contiguous()?.reshape((n, d))?;
        // v44 K2: out_gate applies the sigmoid gate too
        crate::int8::mm(&out_gate(&y, h, hb)?, &hb.wo)
    }

    /// E8: after an optimizer step, move every block's thresholds toward the target rate using the rates of the last
    /// forward: theta += HOMEO_STEP · (rate - target), kept in HOMEO_RANGE. Does nothing without --homeostasis.
    pub fn homeostasis_step(&self) -> Result<()> {
        crate::v51::after_step(self)?; // v51 Pr: the use averages and the cuts (nothing without --prune)
        let (Some(e), target) = (self.ext.as_ref(), self.cfg.arch.homeostasis as f64) else { return Ok(()) };
        // v49 (D2): AVA_HOMEO_OFF=1 keeps the thresholds (and every forward pass) but skips the nudge
        if target <= 0.0 || std::env::var("AVA_HOMEO_OFF").is_ok_and(|v| v == "1") {
            return Ok(());
        }
        let rates = e.rates.borrow();
        for (bx, r) in e.blocks.iter().zip(rates.iter()) {
            if let (Some(th), Some(r)) = (&bx.theta, r) {
                let new = th.add(&r.affine(HOMEO_STEP, -HOMEO_STEP * target)?)?.clamp(HOMEO_RANGE.0, HOMEO_RANGE.1)?;
                Var::from_tensor(th)?.set(&new)?;
            }
        }
        Ok(())
    }

    /// K4 (many GPUs): the rates `homeostasis_step` will use, flattened in block order, so the ranks can average
    /// them (each rank measured only its own rows). `set_homeostasis_rates` puts the average back.
    pub fn homeostasis_rates(&self) -> Result<Vec<f32>> {
        let Some(e) = self.ext.as_ref() else { return Ok(Vec::new()) };
        let mut out = Vec::new();
        for r in e.rates.borrow().iter().flatten() {
            out.extend(r.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?);
        }
        Ok(out)
    }

    pub fn set_homeostasis_rates(&self, flat: &[f32]) -> Result<()> {
        let Some(e) = self.ext.as_ref() else { return Ok(()) };
        let mut off = 0;
        for r in e.rates.borrow_mut().iter_mut().flatten() {
            let n = r.elem_count();
            *r = Tensor::from_slice(&flat[off..off + n], r.dims(), r.device())?.to_dtype(r.dtype())?;
            off += n;
        }
        Ok(())
    }

    /// R2: the order loss for the final hidden state `xf` [N, d] and the targets of `top_targets` (`idx` [N, w] u32,
    /// `p` [N, w] f32): -sum p · log softmax(xf · top_emb^T)[idx], averaged over the positions that have targets.
    pub fn top_loss(&self, xf: &Tensor, idx: &Tensor, p: &Tensor, rows: usize) -> Result<Tensor> {
        let top = self.ext.as_ref().and_then(|e| e.top.as_ref()).ok_or_else(|| candle_core::Error::Msg("this model has no order head (--top-weight)".into()))?;
        let lsm = candle_nn::ops::log_softmax(&xf.matmul(&top.t()?)?, D::Minus1)?;
        let got = lsm.gather(&idx.contiguous()?, 1)?;
        got.mul(p)?.sum_all()?.affine(-1.0 / rows.max(1) as f64, 0.0)
    }

    /// R2: (total = CE + weight · order loss, CE, order loss, activity) for `ids` and `targets` [B, T], with autograd.
    pub fn loss_top(&self, ids: &Tensor, targets: &Tensor, weight: f64) -> Result<(Tensor, Tensor, Tensor, Activity)> {
        let (b, t) = ids.dims2()?;
        let (xf, act) = self.forward_arch(ids)?;
        let logits = xf.matmul(&self.tok.t()?)?;
        self.stash_logits(&logits);
        let tg = targets.flatten_all()?;
        let ce = crate::lif::cross_entropy_fused(&logits, &tg)?;
        let tv: Vec<u32> = tg.to_vec1()?;
        let (idx, p) = top_targets(&tv, b, t, self.cfg.vocab, TOP_WINDOW);
        let rows = p.chunks(TOP_WINDOW).filter(|r| r.iter().any(|&x| x > 0.0)).count();
        let dev = ids.device();
        let idx = Tensor::from_vec(idx, (b * t, TOP_WINDOW), dev)?;
        let p = Tensor::from_vec(p, (b * t, TOP_WINDOW), dev)?;
        let top = self.top_loss(&xf, &idx, &p, rows)?;
        let total = ce.add(&top.affine(weight, 0.0)?)?;
        Ok((total, ce, top, act))
    }
}

/// A block's LIF leak parameters at fresh init with --lif-init logspaced (used by Model::new).
pub(crate) fn set_logspaced_beta(beta: &Tensor, d: usize, dev: &Device) -> Result<()> {
    Var::from_tensor(beta)?.set(&Tensor::from_vec(logspaced_beta_params(d), d, dev)?.to_dtype(beta.dtype())?)
}

pub mod ckpt;

#[cfg(test)]
mod tests;
