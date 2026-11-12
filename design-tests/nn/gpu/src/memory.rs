// TESTING / DESIGN ONLY — ava_gpu::memory: v42 Wave 2C memory options and the in-block brain parts (model D).
//
// Every option is off by default (MemOpts::default()), and then nothing here runs: the model computes exactly what it
// did. They are block options (src/arch.rs: `Arch::mem`), so a model with any of them on runs `Model::forward_arch`
// with candle's autograd, and train_gpu/grow_gpu write them to DIR/arch.txt like the other block flags.
// Every new weight starts so that the part changes nothing (the part is a no-op at init).
//
// The engram read (R4 sharper read, L3 BTSP write; needs --engram L):
//   --eng-topk K     only the K highest scores of each query are kept (the rest get -1e9 before the softmax)
//   --eng-temp       a learned log-temperature per head: scores · exp(tau_h); tau starts at 0 (today's 1/sqrt(32))
//   --eng-ssmax      scalable softmax: scores · (1 + s_h · ln n), n = the keys the query sees; s starts at 0, so a
//                    long store can learn not to blur into an average
//   --btsp W         behavioural-timescale write (after Wu & Maass 2025, adapted to the key-value store): a plateau
//                    signal p_j = sigmoid(h_j·wp + bp) per token gates its slot (score bias g_h · ln p_j: a slot
//                    written without a plateau is not found), and the key written is bound to what came just
//                    before: k_j + m_h · (mean(k_{j-W+1..j}) − k_j), the eligibility window of W tokens.
//                    g and m start at 0, so the write is today's until training uses it. One pass, no iteration:
//                    the write is one-shot.
// Getting new parts used (O2; the research note's three ways: train from the start = a run choice; the two below;
// and logging each part's gradient share, `grad_shares`):
//   --mem-reset N    in training forwards only, the fast-weight memory forgets everything at every N-token
//                    boundary, so recall over more than N tokens must go through the engram (or the spiking state)
//   --recall-loss W  the engram's own recall loss: its read, normed and through the tied embedding (both detached),
//                    must predict the next token too; total = CE + W · CE(engram read)
// A sparse memory layer (A9; Lample et al. 2019 product keys, Lin et al. 2025 sparse memory finetuning):
//   --pk-mem N [--pk-layer L] [--pk-topk K]  N slots (a square), read by product keys after block L's MLP
//                    (default the last block): two half-queries each pick their top K sub-keys, the K·K pairs are
//                    ranked and the top K slots' values (start at 0) are mixed by softmax. `SlotCounts` + `sparse_ft_rows`
//                    pick the slots most specific to new data (TF-IDF against background reading); only those rows
//                    of the values train during sparse finetuning (grow_gpu --sparse-ft T).
// Parts from design-tests/nn/parts (each starts as a no-op):
//   --dendrites N    E6 active dendrites: N segments per channel read the block's normed input and gate the spiking
//                    layer's input current: cur · (1 + a ⊙ tanh(max_j h·S_j)), a starts at 0
//   --stp            E9 short-term plasticity on the spikes (mix starts at 0)
//   --delays K       E5 learned synaptic delays of up to K−1 tokens on the input current (gain starts at 0)
//   --wm N           B9 gated working memory: N slots after the engram, its read-out starting at 0
// More parts (src/wave5.rs): --resonate P, --modulate, --tem K, --halting N, --cerebellum G, --workspace N, --holo DB,
// --sync N, and the
// auxiliary losses --jepa W, --diffusion W, --critic W, --confidence W, --topo W, --schema W.
// Two records of a grown shape (src/grow.rs, grow_exact; A15):
//   --ws-layer L     the shared workspace (--workspace) sits in block L instead of the middle block (layers/2), so a
//                    model grown deeper keeps it where it was trained
//   --mlp-mult M     the MLP's hidden width is M·d instead of model::MLP_MULT·d (4d), for a model whose MLPs were
//                    widened; Config::hidden() (src/model.rs) reads it, so a model never builds 4d
//                    MLPs while its file says otherwise

use crate::model::{causal_mask, kwta_mask, log_sigmoid, rmsnorm, Attn, Config, Kind, HEAD_SIZE};
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};
use std::cell::RefCell;

/// The additive mask value (as model.rs: exp underflows to exactly 0).
pub(crate) const NEG: f32 = -1e9;
/// A9: the default top-k and the width of each half-query.
pub const PK_TOPK: usize = 32;
pub const PK_HALF: usize = 32;
/// E5: the width of the delay kernels' bumps (in tokens).
pub const DELAY_SIGMA: f64 = 1.0;
/// B9: the width of each working-memory slot.
pub const WM_DK: usize = 32;

/// The Wave 2C options (see the top of the file). Default: all off.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct MemOpts {
    pub eng_topk: usize,
    pub eng_temp: bool,
    pub eng_ssmax: bool,
    pub btsp: usize,
    pub mem_reset: usize,
    pub recall_loss: f32,
    pub pk_slots: usize,
    pub pk_layer: Option<usize>,
    pub pk_topk: usize,
    pub dendrites: usize,
    pub stp: bool,
    pub delays: usize,
    pub wm: usize,
    /// src/wave5.rs: E7 the resonators' longest period (0 = off)
    pub resonate: usize,
    /// B8 + F13
    pub modulate: bool,
    /// B10: the phases (0 = off)
    pub tem: usize,
    /// A7: the most MLP passes (0 = off)
    pub halting: usize,
    /// H11: the granule units (0 = off)
    pub cerebellum: usize,
    /// B3: the shared workspace's slots (0 = off)
    pub workspace: usize,
    /// D2: the holographic binding memory's width in every block (0 = off)
    pub holo: usize,
    /// L5: the synchrony oscillators per token in every block (0 = off)
    pub sync: usize,
    /// the auxiliary losses' weights (0 = off): B1, N1, B2, C8/F9
    pub jepa: f32,
    pub diffusion: f32,
    pub critic: f32,
    pub confidence: f32,
    /// H12: the topographic loss's weight (0 = off)
    pub topo: f32,
    /// F12: the attention schema's weight (0 = off; needs --workspace)
    pub schema: f32,
    /// F9 (needs --confidence): in an answer-only (weighted) step the confidence probe learns only where the answer is
    /// scored, so at the question's last token it learns "will my answer be right" (src/wave5.rs)
    pub fok_qa: bool,
    /// H3: practice to reflex, the weight (0 = off; needs --halting): the first MLP pass learns the pondered answer,
    /// each token times its success² (src/wave5.rs)
    pub reflex: f32,
    /// F10 follow-up (needs --confidence): each training step the probe also learns on her final state pushed by a
    /// random direction (strength uniform in [-S, S] times the mean state length), labelled by the pushed guess (src/wave5.rs)
    pub fok_push: f32,
    /// A15 (src/grow.rs): the workspace's block (None: wave5::ws_block, the middle one)
    pub ws_layer: Option<usize>,
    /// A15 (src/grow.rs): the MLP's hidden width in multiples of d (0: model::MLP_MULT)
    pub mlp_mult: usize,
}

// recall_loss and the auxiliary weights are checked finite by `validate`.
impl Eq for MemOpts {}

pub const FLAG_USAGE: &str = "[--eng-topk K] [--eng-temp] [--eng-ssmax] [--btsp W] [--mem-reset N] [--recall-loss W]\n\
    \x20            [--pk-mem N [--pk-layer L] [--pk-topk K]] [--dendrites N] [--stp] [--delays K] [--wm N]\n\
    \x20            [--resonate P] [--modulate] [--tem K] [--halting N] [--cerebellum G] [--workspace N] [--holo DB] [--sync N]\n\
    \x20            [--jepa W] [--diffusion W] [--critic W] [--confidence W] [--topo W] [--schema W] [--reflex W] [--fok-qa] [--fok-push S]\n\
    \x20            [--ws-layer L] [--mlp-mult M]";

impl MemOpts {
    pub fn is_default(&self) -> bool {
        *self == MemOpts::default()
    }

    /// Any option of the engram read (they need --engram).
    pub fn eng_any(&self) -> bool {
        self.eng_topk > 0 || self.eng_temp || self.eng_ssmax || self.btsp > 0
    }

    /// An auxiliary loss joins the training loss (src/wave5.rs; A7's KL included): train through `Model::loss_aux`.
    pub fn aux_any(&self) -> bool {
        self.halting > 0 || self.jepa > 0.0 || self.diffusion > 0.0 || self.critic > 0.0 || self.confidence > 0.0 || self.topo > 0.0 || self.schema > 0.0
    }

    /// The product-key memory's block for a model of `layers` blocks.
    pub fn pk_block(&self, layers: usize) -> Option<usize> {
        (self.pk_slots > 0).then(|| self.pk_layer.unwrap_or(layers.saturating_sub(1)))
    }

    /// The product-key top-k: --pk-topk, else PK_TOPK (at most sqrt(slots)).
    pub fn pk_k(&self) -> usize {
        if self.pk_topk == 0 {
            PK_TOPK.min((self.pk_slots as f64).sqrt().round() as usize)
        } else {
            self.pk_topk
        }
    }

    /// One Wave 2C flag at `args[i]`: how many arguments it used, or None when it is not one of them.
    pub fn take_flag(&mut self, args: &[String], i: usize) -> std::result::Result<Option<usize>, String> {
        let key = args[i].as_str();
        match key {
            "--eng-temp" => self.eng_temp = true,
            "--eng-ssmax" => self.eng_ssmax = true,
            "--stp" => self.stp = true,
            "--modulate" => self.modulate = true,
            "--fok-qa" => self.fok_qa = true,
            _ => {
                let known = [
                    "--eng-topk",
                    "--btsp",
                    "--mem-reset",
                    "--recall-loss",
                    "--pk-mem",
                    "--pk-layer",
                    "--pk-topk",
                    "--dendrites",
                    "--delays",
                    "--wm",
                    "--resonate",
                    "--tem",
                    "--halting",
                    "--cerebellum",
                    "--workspace",
                    "--holo",
                    "--sync",
                    "--topo",
                    "--schema",
                    "--jepa",
                    "--reflex",
                    "--fok-push",
                    "--diffusion",
                    "--critic",
                    "--confidence",
                    "--ws-layer",
                    "--mlp-mult",
                ];
                if !known.contains(&key) {
                    return Ok(None);
                }
                let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value"))?.as_str();
                let int = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
                let num = |v: &str| v.parse::<f32>().map_err(|_| format!("bad number {v:?} for {key}"));
                match key {
                    "--eng-topk" => self.eng_topk = int(v)?,
                    "--btsp" => self.btsp = int(v)?,
                    "--mem-reset" => self.mem_reset = int(v)?,
                    "--recall-loss" => self.recall_loss = num(v)?,
                    "--pk-mem" => self.pk_slots = int(v)?,
                    "--pk-layer" => self.pk_layer = Some(int(v)?),
                    "--pk-topk" => self.pk_topk = int(v)?,
                    "--dendrites" => self.dendrites = int(v)?,
                    "--delays" => self.delays = int(v)?,
                    "--wm" => self.wm = int(v)?,
                    "--resonate" => self.resonate = int(v)?,
                    "--tem" => self.tem = int(v)?,
                    "--halting" => self.halting = int(v)?,
                    "--cerebellum" => self.cerebellum = int(v)?,
                    "--workspace" => self.workspace = int(v)?,
                    "--holo" => self.holo = int(v)?,
                    "--sync" => self.sync = int(v)?,
                    "--topo" => self.topo = num(v)?,
                    "--schema" => self.schema = num(v)?,
                    "--jepa" => self.jepa = num(v)?,
                    "--reflex" => self.reflex = num(v)?,
                    "--fok-push" => self.fok_push = num(v)?,
                    "--diffusion" => self.diffusion = num(v)?,
                    "--critic" => self.critic = num(v)?,
                    "--confidence" => self.confidence = num(v)?,
                    "--ws-layer" => self.ws_layer = Some(int(v)?),
                    "--mlp-mult" => self.mlp_mult = int(v)?,
                    _ => unreachable!(),
                }
                return Ok(Some(2));
            }
        }
        Ok(Some(1))
    }

    /// The flags that set these options (empty for the default), in a fixed order.
    pub fn flags(&self) -> Vec<String> {
        let mut f = Vec::new();
        if self.eng_topk > 0 {
            f.push(format!("--eng-topk {}", self.eng_topk));
        }
        if self.eng_temp {
            f.push("--eng-temp".into());
        }
        if self.eng_ssmax {
            f.push("--eng-ssmax".into());
        }
        if self.btsp > 0 {
            f.push(format!("--btsp {}", self.btsp));
        }
        if self.mem_reset > 0 {
            f.push(format!("--mem-reset {}", self.mem_reset));
        }
        if self.recall_loss != 0.0 {
            f.push(format!("--recall-loss {}", self.recall_loss));
        }
        if self.pk_slots > 0 {
            f.push(format!("--pk-mem {}", self.pk_slots));
            if let Some(l) = self.pk_layer {
                f.push(format!("--pk-layer {l}"));
            }
            if self.pk_topk > 0 {
                f.push(format!("--pk-topk {}", self.pk_topk));
            }
        }
        if self.dendrites > 0 {
            f.push(format!("--dendrites {}", self.dendrites));
        }
        if self.stp {
            f.push("--stp".into());
        }
        if self.delays > 0 {
            f.push(format!("--delays {}", self.delays));
        }
        if self.wm > 0 {
            f.push(format!("--wm {}", self.wm));
        }
        if self.resonate > 0 {
            f.push(format!("--resonate {}", self.resonate));
        }
        if self.modulate {
            f.push("--modulate".into());
        }
        if self.tem > 0 {
            f.push(format!("--tem {}", self.tem));
        }
        if self.halting > 0 {
            f.push(format!("--halting {}", self.halting));
        }
        if self.cerebellum > 0 {
            f.push(format!("--cerebellum {}", self.cerebellum));
        }
        if self.workspace > 0 {
            f.push(format!("--workspace {}", self.workspace));
            if let Some(l) = self.ws_layer {
                f.push(format!("--ws-layer {l}"));
            }
        }
        if self.holo > 0 {
            f.push(format!("--holo {}", self.holo));
        }
        if self.sync > 0 {
            f.push(format!("--sync {}", self.sync));
        }
        for (key, w) in [("--jepa", self.jepa), ("--diffusion", self.diffusion), ("--critic", self.critic), ("--confidence", self.confidence), ("--topo", self.topo), ("--schema", self.schema), ("--reflex", self.reflex), ("--fok-push", self.fok_push)] {
            if w != 0.0 {
                f.push(format!("{key} {w}"));
            }
        }
        if self.fok_qa {
            f.push("--fok-qa".into());
        }
        if self.mlp_mult > 0 {
            f.push(format!("--mlp-mult {}", self.mlp_mult));
        }
        f
    }

    /// " key=value ..." for resume signatures and logs ("" for the default).
    pub fn signature(&self) -> String {
        self.flags().iter().map(|f| format!(" {}", f.trim_start_matches("--").replacen(' ', "=", 1).replace('-', "_"))).collect()
    }

    pub fn validate(&self, c: &Config) -> std::result::Result<(), String> {
        if self.is_default() {
            return Ok(());
        }
        if c.kind != Kind::D {
            return Err(format!("the Wave 2C options {} are for model D", self.flags().join(" ")));
        }
        if self.eng_any() && c.engram.is_none() {
            return Err("--eng-topk, --eng-temp, --eng-ssmax and --btsp change the engram read: they need --engram L".into());
        }
        if self.recall_loss != 0.0 && (c.engram.is_none() || !(self.recall_loss > 0.0 && self.recall_loss.is_finite())) {
            return Err(format!("--recall-loss {}: a positive weight, and it needs --engram L", self.recall_loss));
        }
        if self.pk_slots > 0 {
            let n1 = (self.pk_slots as f64).sqrt().round() as usize;
            if n1 * n1 != self.pk_slots || n1 < 2 {
                return Err(format!("--pk-mem {}: a square number of slots (e.g. 1024 = 32 x 32)", self.pk_slots));
            }
            if self.pk_k() > n1 {
                return Err(format!("--pk-topk {}: at most sqrt(slots) = {n1}", self.pk_k()));
            }
            if self.pk_block(c.layers).is_some_and(|l| l >= c.layers) {
                return Err(format!("--pk-layer: must be below {}", c.layers));
            }
        } else if self.pk_layer.is_some() || self.pk_topk > 0 {
            return Err("--pk-layer and --pk-topk need --pk-mem N".into());
        }
        if self.delays == 1 {
            return Err("--delays K: K >= 2 taps (0 = off)".into());
        }
        if self.resonate == 1 {
            return Err("--resonate P: the longest period, P >= 2 tokens (0 = off)".into());
        }
        if self.halting == 1 {
            return Err("--halting N: N >= 2 passes (0 = off; one pass is today's MLP)".into());
        }
        if self.fok_qa && self.confidence == 0.0 {
            return Err("--fok-qa teaches the confidence probe on answers: it needs --confidence W".into());
        }
        if self.fok_push != 0.0 && self.confidence == 0.0 {
            return Err("--fok-push S teaches the confidence probe on pushed states: it needs --confidence W".into());
        }
        if self.reflex != 0.0 && self.halting == 0 {
            return Err("--reflex W teaches the first MLP pass the pondered answer: it needs --halting N".into());
        }
        if self.cerebellum > 0 && self.cerebellum < crate::wave5::CB_SPARSITY {
            return Err(format!("--cerebellum G: at least {} granule units (the top G/{} are kept)", crate::wave5::CB_SPARSITY, crate::wave5::CB_SPARSITY));
        }
        if self.workspace > 0 && self.workspace <= crate::wave5::WS_TOPK {
            return Err(format!("--workspace N: more than {} slots (the top {} write each token)", crate::wave5::WS_TOPK, crate::wave5::WS_TOPK));
        }
        if self.holo == 1 {
            return Err("--holo DB: a binding width of 2 or more (0 = off)".into());
        }
        if self.schema != 0.0 && self.workspace == 0 {
            return Err("--schema W models the workspace's attention: it needs --workspace N".into());
        }
        if let Some(l) = self.ws_layer {
            if self.workspace == 0 || l >= c.layers {
                return Err(format!("--ws-layer {l}: needs --workspace N and a block below {}", c.layers));
            }
        }
        if self.mlp_mult > 0 && c.hidden() != self.mlp_mult * c.d {
            return Err(format!(
                "--mlp-mult {}: the MLP would be {} wide, not {} x d = {}: Config::hidden() (src/model.rs) does not read --mlp-mult yet",
                self.mlp_mult,
                c.hidden(),
                self.mlp_mult,
                self.mlp_mult * c.d
            ));
        }
        for (key, w) in [("--jepa", self.jepa), ("--diffusion", self.diffusion), ("--critic", self.critic), ("--confidence", self.confidence), ("--topo", self.topo), ("--schema", self.schema), ("--reflex", self.reflex), ("--fok-push", self.fok_push)] {
            if !(w >= 0.0 && w.is_finite()) {
                return Err(format!("{key} {w}: a weight of 0 or more"));
            }
        }
        let ar = c.arch;
        if (self.stp || self.delays > 0 || self.wm > 0 || self.workspace > 0 || self.holo > 0) && ar.feedback {
            return Err("--stp, --delays, --wm, --workspace and --holo keep a state over time that --feedback's chunks do not carry yet".into());
        }
        if self.sync > 0 && ar.feedback {
            return Err("--sync is not wired for --feedback's chunks yet".into());
        }
        if self.reflex != 0.0 && ar.feedback {
            return Err("--reflex is not wired for --feedback's chunks yet (its token weights are for the whole window)".into());
        }
        if (self.resonate > 0 || self.modulate || self.tem > 0 || self.halting > 0) && ar.feedback {
            return Err("--resonate, --modulate, --tem and --halting are not wired for --feedback's chunks yet".into());
        }
        if self.mem_reset > 0 && (ar.gates != crate::arch::Gating::Head || ar.mem_momentum != 0.0 || ar.feedback) {
            return Err("--mem-reset is for today's memory write (no --gates channel, --mem-momentum or --feedback)".into());
        }
        Ok(())
    }

    /// The new weights these options add (for Config::weight_counts).
    pub fn extra_weights(&self, c: &Config) -> usize {
        if self.is_default() {
            return 0;
        }
        let (d, l) = (c.d, c.layers);
        let nh = d / HEAD_SIZE;
        let mut n = 0;
        if self.eng_temp {
            n += nh;
        }
        if self.eng_ssmax {
            n += nh;
        }
        if self.btsp > 0 {
            n += d + 1 + 2 * nh;
        }
        if self.pk_slots > 0 {
            let n1 = (self.pk_slots as f64).sqrt().round() as usize;
            n += d + d * 2 * PK_HALF + 2 * n1 * PK_HALF + self.pk_slots * d;
        }
        if self.dendrites > 0 {
            n += l * (d * self.dendrites * d + d);
        }
        if self.stp {
            n += l * 4 * d;
        }
        if self.delays > 0 {
            n += l * 2 * d;
        }
        if self.wm > 0 {
            let k = self.wm;
            n += l * (d + d * k + k + d * k * WM_DK + d * WM_DK + WM_DK * d + d);
        }
        n + crate::wave5::extra_weights(self, c).0
    }

    /// How many of `extra_weights` are vocabulary tables (counted with the embeddings): N1's.
    pub fn extra_emb_weights(&self, c: &Config) -> usize {
        crate::wave5::extra_weights(self, c).1
    }
}

// ------------------------------------------------------------------------------------------- the new weights

/// L3: the plateau gate and the eligibility window's mix (engram block).
pub(crate) struct Btsp {
    pub wp: Tensor,
    pub bp: Tensor,
    pub g: Tensor,
    pub mix: Tensor,
    pub w: usize,
}

/// R4 and L3 weights of the engram block.
pub(crate) struct EngExt {
    pub tau: Option<Tensor>,
    pub ss: Option<Tensor>,
    pub btsp: Option<Btsp>,
}

/// A9: the product-key memory.
pub(crate) struct Pkm {
    pub norm: Tensor,
    pub wq: Tensor,
    pub k1: Tensor,
    pub k2: Tensor,
    pub values: Tensor,
    pub n1: usize,
    pub topk: usize,
    /// When Some: every forward adds how often each slot was read (sparse memory finetuning).
    pub counts: RefCell<Option<Vec<u64>>>,
}

/// The parts of one block.
pub(crate) struct PartsExt {
    pub dend: Option<ava_parts::dendrites::Dendrites>,
    pub stp: Option<ava_parts::stp::Stp>,
    pub delays: Option<ava_parts::delays::Delays>,
    pub wm: Option<(Tensor, ava_parts::wm::SlotMemory)>,
}

/// Everything Wave 2C adds to a model (inside arch::ModelExt).
pub(crate) struct MemExt {
    pub eng: Option<EngExt>,
    pub pkm: Option<Pkm>,
    pub parts: Vec<PartsExt>,
    /// O2: the engram read of the last training forward (for --recall-loss), [B·T, d]
    pub recall: RefCell<Option<Tensor>>,
    /// src/wave5.rs: the more parts and the auxiliary heads
    pub w5: crate::wave5::W5Ext,
}

/// Make the Wave 2C weights in `vb`, after every other tensor (so the random init of the rest is drawn as before).
pub(crate) fn build(cfg: &Config, vb: &VarBuilder) -> Result<MemExt> {
    let o = cfg.arch.mem;
    let d = cfg.d;
    let nh = d / HEAD_SIZE;
    let randn = |std: f64| Init::Randn { mean: 0.0, stdev: std };
    let eng = match cfg.engram {
        Some(l) if o.eng_any() => {
            let ve = vb.pp(format!("b{l}.eng"));
            Some(EngExt {
                tau: if o.eng_temp { Some(ve.get_with_hints(nh, "tau", Init::Const(0.0))?) } else { None },
                ss: if o.eng_ssmax { Some(ve.get_with_hints(nh, "ss", Init::Const(0.0))?) } else { None },
                btsp: if o.btsp > 0 {
                    Some(Btsp {
                        wp: ve.get_with_hints((d, 1), "wp", randn(crate::model::INIT_STD))?,
                        bp: ve.get_with_hints(1, "bp", Init::Const(0.0))?,
                        g: ve.get_with_hints(nh, "btsp_g", Init::Const(0.0))?,
                        mix: ve.get_with_hints(nh, "btsp_mix", Init::Const(0.0))?,
                        w: o.btsp,
                    })
                } else {
                    None
                },
            })
        }
        _ => None,
    };
    let pkm = match o.pk_block(cfg.layers) {
        Some(l) => {
            let vp = vb.pp(format!("b{l}.pk"));
            let n1 = (o.pk_slots as f64).sqrt().round() as usize;
            Some(Pkm {
                norm: vp.get_with_hints(d, "norm", Init::Const(1.0))?,
                wq: vp.get_with_hints((d, 2 * PK_HALF), "wq", randn(1.0 / (d as f64).sqrt()))?,
                k1: vp.get_with_hints((n1, PK_HALF), "k1", randn(1.0 / (PK_HALF as f64).sqrt()))?,
                k2: vp.get_with_hints((n1, PK_HALF), "k2", randn(1.0 / (PK_HALF as f64).sqrt()))?,
                values: vp.get_with_hints((o.pk_slots, d), "values", Init::Const(0.0))?,
                n1,
                topk: o.pk_k(),
                counts: RefCell::new(None),
            })
        }
        None => None,
    };
    let mut parts = Vec::with_capacity(cfg.layers);
    for i in 0..cfg.layers {
        let vp = vb.pp(format!("b{i}.parts"));
        let dend = if o.dendrites > 0 { Some(ava_parts::dendrites::Dendrites::new(vp.pp("dend"), d, d, o.dendrites)?) } else { None };
        let stp = if o.stp { Some(ava_parts::stp::Stp::new(vp.pp("stp"), d)?) } else { None };
        let delays = if o.delays > 0 { Some(ava_parts::delays::Delays::new(vp.pp("delays"), d, o.delays, DELAY_SIGMA)?) } else { None };
        let wm = if o.wm > 0 {
            let norm = vp.get_with_hints(d, "wm.norm", Init::Const(1.0))?;
            Some((norm, ava_parts::wm::SlotMemory::new(vp.pp("wm"), d, o.wm, WM_DK, ava_parts::wm::Gating::Independent)?))
        } else {
            None
        };
        parts.push(PartsExt { dend, stp, delays, wm });
    }
    // src/wave5.rs: after every tensor above, so the options before it draw their init as they did
    let w5 = crate::wave5::build(cfg, vb)?;
    Ok(MemExt { eng, pkm, parts, recall: RefCell::new(None), w5 })
}

// ------------------------------------------------------------------------------------------- the read

/// What a stateful read keeps of one attention or engram block (state.rs): every token's key and value so far.
#[derive(Clone)]
pub struct Kv {
    /// [B, nh, S, hs]: the keys as written (with --btsp, the window-bound keys) and the values
    pub k: Tensor,
    pub v: Tensor,
    /// [B, S] additive: 0 for a slot, NEG for padding (rows of different lengths, see State::cat); None: no padding
    pub pad: Option<Tensor>,
    /// how many real slots each row holds
    pub valid: Vec<usize>,
    /// L3: each slot's log plateau [B, S]
    pub logp: Option<Tensor>,
    /// L3: the last W − 1 raw keys [B, nh, W − 1, hs] (the next tokens' eligibility window)
    pub ktail: Option<Tensor>,
    /// S2 least-read: the attention mass each slot received [B, S] and how many later queries could see it [B, S]
    pub reads: Option<Tensor>,
    pub age: Option<Tensor>,
}

/// How a read updates its store.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StoreUse {
    /// keep read counts (for the least-read eviction rule)
    pub count_reads: bool,
}

/// The engram (or attention) read of the normed input h [b·t, d] over the store and the new tokens, heads of 32:
/// q·k^T/sqrt(32) (R4: · exp(tau_h), · (1 + s_h ln n)), + g_h ln p_j (L3), causal mask (+ padding), top-k (R4),
/// softmax, · v, then Wo. With every option off and no store this is model::attention's chain exactly (the same
/// ops in the same order). `store`: Some for a stateful read (the new keys and values are appended); `quiet`: write
/// the store but return None (a lesioned engram keeps writing, as Model::step does).
#[allow(clippy::too_many_arguments)]
pub(crate) fn read(
    h: &Tensor,
    e: &Attn,
    ext: Option<&EngExt>,
    o: &MemOpts,
    b: usize,
    t: usize,
    store: Option<&mut Option<Kv>>,
    usage: StoreUse,
    quiet: bool,
    gate: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    // v45 W4 (`gate`, --eng-gate): the output gate [b·t, heads or d] on the heads before Wo (None: no gate)
    let d = h.dim(1)?;
    let nh = d / HEAD_SIZE;
    let dev = h.device();
    let heads = |w: &Tensor| -> Result<Tensor> { crate::int8::mm(&h, w)?.reshape((b, t, nh, HEAD_SIZE))?.transpose(1, 2)?.contiguous() };
    let (q, k, v) = (heads(&e.wq)?, heads(&e.wk)?, heads(&e.wv)?);
    let mut old = store.as_ref().and_then(|s| (*s).clone());
    // L3: the window-bound keys and each token's log plateau
    let bt = ext.and_then(|x| x.btsp.as_ref());
    let (kw, logp_new, ktail_new) = match bt {
        Some(bt) => {
            let tail = old.as_ref().and_then(|s| s.ktail.clone());
            let (kw, tail_next) = btsp_keys(&k, tail.as_ref(), &bt.mix, bt.w)?;
            let lp = log_sigmoid(&h.matmul(&bt.wp)?.broadcast_add(&bt.bp)?)?.reshape((b, t))?;
            (kw, Some(lp), tail_next)
        }
        None => (k.clone(), None, None),
    };
    let s0 = old.as_ref().map_or(0, |s| s.k.dim(2).unwrap_or(0));
    let (kc, vc) = match &old {
        None => (kw.clone(), v.clone()),
        Some(s) => (Tensor::cat(&[&s.k, &kw], 2)?, Tensor::cat(&[&s.v, &v], 2)?),
    };
    let sn = s0 + t;
    let valid: Vec<usize> = match &old {
        None => vec![t; b],
        Some(s) => s.valid.iter().map(|x| x + t).collect(),
    };
    let pad = match old.as_ref().and_then(|s| s.pad.as_ref()) {
        None => None,
        Some(p) => Some(Tensor::cat(&[p, &Tensor::zeros((b, t), DType::F32, dev)?], 1)?),
    };
    let logp = match (&logp_new, old.as_ref().and_then(|s| s.logp.as_ref())) {
        (None, _) => None,
        (Some(n), None) if s0 == 0 => Some(n.clone()),
        (Some(n), None) => Some(Tensor::cat(&[&Tensor::zeros((b, s0), DType::F32, dev)?, n], 1)?),
        (Some(n), Some(p)) => Some(Tensor::cat(&[p, n], 1)?),
    };
    let out = if quiet {
        None
    } else {
        let scale = 1.0 / (HEAD_SIZE as f64).sqrt();
        let mut sc = q.matmul(&kc.t()?.contiguous()?)?.affine(scale, 0.0)?; // [b, nh, t, sn]
        if let Some(tau) = ext.and_then(|x| x.tau.as_ref()) {
            sc = sc.broadcast_mul(&tau.exp()?.reshape((1, nh, 1, 1))?)?;
        }
        if let Some(ss) = ext.and_then(|x| x.ss.as_ref()) {
            // n = the keys query i sees: the row's earlier slots + i + 1
            let mut ln = Vec::with_capacity(b * t);
            for r in 0..b {
                let before = valid[r] - t;
                ln.extend((0..t).map(|i| ((before + i + 1) as f32).ln()));
            }
            let ln = Tensor::from_vec(ln, (b, 1, t, 1), dev)?;
            sc = sc.mul(&ln.broadcast_mul(&ss.reshape((1, nh, 1, 1))?)?.affine(1.0, 1.0)?.broadcast_as(sc.shape())?.contiguous()?)?;
        }
        if let (Some(bt), Some(lp)) = (bt, &logp) {
            sc = sc.broadcast_add(&lp.reshape((b, 1, 1, sn))?.broadcast_mul(&bt.g.reshape((1, nh, 1, 1))?)?)?;
        }
        let mask = if s0 == 0 { causal_mask(t, dev)? } else { Tensor::cat(&[&Tensor::zeros((t, s0), DType::F32, dev)?, &causal_mask(t, dev)?], 1)? };
        let mut sm = sc.broadcast_add(&mask)?;
        if let Some(p) = &pad {
            sm = sm.broadcast_add(&p.reshape((b, 1, 1, sn))?)?;
        }
        if o.eng_topk > 0 && o.eng_topk < sn && ext.is_some() {
            let flat = sm.reshape((b * nh * t, sn))?;
            let keep = kwta_mask(&flat, o.eng_topk)?.reshape(sm.shape())?; // 0/1, detached
            sm = sm.add(&keep.affine(-(NEG as f64), NEG as f64)?)?;
        }
        let p = candle_nn::ops::softmax(&sm, D::Minus1)?;
        let y = p.matmul(&vc)?;
        Some((y, p))
    };
    let (y, probs) = match out {
        Some((y, p)) => (Some(y), Some(p)),
        None => (None, None),
    };
    if let Some(st) = store {
        let (mut reads, mut age) = (None, None);
        if usage.count_reads {
            let (r_old, a_old) = match old.as_mut() {
                Some(s) => (s.reads.take(), s.age.take()),
                None => (None, None),
            };
            let zeros = |n: usize| Tensor::zeros((b, n), DType::F32, dev);
            // mass each slot got from the new queries (the diagonal and later tokens of this call left out)
            let got = match &probs {
                Some(p) => {
                    let seen: Vec<f32> = (0..t).flat_map(|i| (0..sn).map(move |j| if j < s0 || j - s0 < i { 1.0 } else { 0.0 })).collect();
                    let seen = Tensor::from_vec(seen, (t, sn), dev)?;
                    p.sum(1)?.affine(1.0 / nh as f64, 0.0)?.broadcast_mul(&seen)?.sum(1)? // [b, sn]
                }
                None => zeros(sn)?,
            };
            let new_age: Vec<f32> = (0..t).map(|j| (t - 1 - j) as f32).collect();
            let new_age = Tensor::from_vec(new_age, (1, t), dev)?.broadcast_as((b, t))?.contiguous()?;
            // a store made without counts (or a fresh one) starts its old slots at 0 reads and 0 age
            let a_all = if s0 > 0 {
                let a_old = match a_old {
                    Some(a) => a,
                    None => zeros(s0)?,
                };
                Tensor::cat(&[&a_old.affine(1.0, t as f64)?, &new_age], 1)?
            } else {
                new_age
            };
            let r_all = if s0 > 0 {
                let r_old = match r_old {
                    Some(r) => r,
                    None => zeros(s0)?,
                };
                Tensor::cat(&[&r_old, &zeros(t)?], 1)?.add(&got)?
            } else {
                got
            };
            reads = Some(r_all);
            age = Some(a_all);
        }
        *st = Some(Kv { k: kc, v: vc, pad, valid, logp, ktail: ktail_new, reads, age });
    }
    match y {
        None => Ok(None),
        Some(y) => {
            let y = y.transpose(1, 2)?.contiguous()?.reshape((b * t, d))?;
            let y = match gate {
                Some(g) => crate::arch::eng_gate_apply(&y, g)?,
                None => y,
            };
            Ok(Some(crate::int8::mm(&y, &e.wo)?))
        }
    }
}

/// L3: the keys bound to their eligibility window: k_j + mix_h · (mean of the raw keys j−W+1..j − k_j), where the
/// window reaches back into `tail` (the raw keys before this call, [b, nh, W−1, hs]) and is shorter at the start.
/// Each window is summed directly (W adds), so a sequence read in pieces gives the same keys as one pass. Returns the
/// keys and the new tail (the last W − 1 raw keys).
pub(crate) fn btsp_keys(k: &Tensor, tail: Option<&Tensor>, mix: &Tensor, w: usize) -> Result<(Tensor, Option<Tensor>)> {
    let (b, nh, t, hs) = k.dims4()?;
    let dev = k.device();
    let tl = tail.map_or(0, |x| x.dim(2).unwrap_or(0));
    let kk = match tail {
        Some(x) if tl > 0 => Tensor::cat(&[x, k], 2)?,
        _ => k.clone(),
    };
    let total = tl + t;
    let mut sum: Option<Tensor> = None;
    for s in 0..w {
        // the key s places back, for every new position whose window reaches it (from `first` on)
        let first = s.saturating_sub(tl);
        if first >= t {
            break;
        }
        let src = kk.narrow(2, tl + first - s, t - first)?;
        let part = if first > 0 { Tensor::cat(&[&Tensor::zeros((b, nh, first, hs), DType::F32, dev)?, &src], 2)? } else { src };
        sum = Some(match sum {
            None => part,
            Some(a) => a.add(&part)?,
        });
    }
    let sum = sum.expect("w >= 1");
    let counts: Vec<f32> = (0..t).map(|j| w.min(tl + j + 1) as f32).collect();
    let mean = sum.broadcast_div(&Tensor::from_vec(counts, (1, 1, t, 1), dev)?)?;
    let kw = k.add(&mean.sub(k)?.broadcast_mul(&mix.reshape((1, nh, 1, 1))?)?)?;
    let keep = (w - 1).min(total);
    let tail_next = if keep == 0 { None } else { Some(kk.narrow(2, total - keep, keep)?.contiguous()?) };
    Ok((kw, tail_next))
}

// ------------------------------------------------------------------------------------------- A9 product keys

impl Pkm {
    /// The memory's output for x [n, d] (before the residual add): normed, product-key top-k, softmax-mixed values.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let h = rmsnorm(x, &self.norm)?;
        let q = h.matmul(&self.wq)?; // [n, 2·half]
        let (q1, q2) = (q.narrow(1, 0, PK_HALF)?, q.narrow(1, PK_HALF, PK_HALF)?);
        let s1 = q1.matmul(&self.k1.t()?)?; // [n, n1]
        let s2 = q2.matmul(&self.k2.t()?)?;
        let k = self.topk;
        let (v1, i1) = topk_last(&s1, k)?; // [n, k]
        let (v2, i2) = topk_last(&s2, k)?;
        // all k·k pairs: score v1_a + v2_b, slot i1_a·n1 + i2_b
        let pv = v1.unsqueeze(2)?.broadcast_add(&v2.unsqueeze(1)?)?.reshape((n, k * k))?;
        let pi = i1.to_dtype(DType::F32)?.affine(self.n1 as f64, 0.0)?.unsqueeze(2)?.broadcast_add(&i2.to_dtype(DType::F32)?.unsqueeze(1)?)?.reshape((n, k * k))?;
        let (bv, bi) = topk_last(&pv, k)?; // [n, k] (positions within the pairs)
        let slots = pi.gather(&bi, 1)?.to_dtype(DType::U32)?; // [n, k]
        let w = candle_nn::ops::softmax(&bv, D::Minus1)?; // [n, k]
        if let Some(c) = self.counts.borrow_mut().as_mut() {
            for s in slots.flatten_all()?.to_vec1::<u32>()? {
                c[s as usize] += 1;
            }
        }
        let vals = self.values.embedding(&slots.flatten_all()?)?.reshape((n, k, self.values.dim(1)?))?;
        vals.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)
    }
}

/// The k largest values of each row of x [n, m] and their column indices (u32), highest first.
pub(crate) fn topk_last(x: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let idx = x.detach().arg_sort_last_dim(false)?.narrow(1, 0, k)?.contiguous()?;
    Ok((x.contiguous()?.gather(&idx, 1)?, idx))
}

/// A9 sparse memory finetuning: how often each slot was read, on the new data and on background reading.
#[derive(Clone, Debug, Default)]
pub struct SlotCounts {
    pub counts: Vec<u64>,
}

/// A9: the rows of the values to train on new data: the `top` slots by TF-IDF, tf = the slot's share of the new
/// data's reads, idf = ln((background reads + 1) / (the slot's background reads + 1)). Slots the new data never
/// read are never picked. Sorted by slot.
pub fn sparse_ft_rows(new: &[u64], background: &[u64], top: usize) -> Vec<usize> {
    let tot_new: u64 = new.iter().sum();
    let tot_bg: u64 = background.iter().sum();
    if tot_new == 0 {
        return Vec::new();
    }
    let mut score: Vec<(f64, usize)> = new
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| {
            let tf = c as f64 / tot_new as f64;
            let idf = ((tot_bg as f64 + 1.0) / (background.get(i).copied().unwrap_or(0) as f64 + 1.0)).ln();
            (tf * idf, i)
        })
        .collect();
    score.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut rows: Vec<usize> = score.into_iter().take(top).map(|x| x.1).collect();
    rows.sort_unstable();
    rows
}

/// A9: a [slots, 1] 0/1 mask with ones at `rows` (multiply the values' gradient by it).
pub fn row_mask(rows: &[usize], slots: usize, dev: &candle_core::Device) -> Result<Tensor> {
    let mut m = vec![0f32; slots];
    for &r in rows {
        if r < slots {
            m[r] = 1.0;
        }
    }
    Tensor::from_vec(m, (slots, 1), dev)
}

// ------------------------------------------------------------------------------------------- O2 gradient shares

/// O2: each part's share of the squared gradient norm, from the parameter names: the embeddings, and per kind of part
/// summed over the blocks: spiking layer (`.mem.`), fast-weight memory (`.hebb.`), engram (`.eng.`), attention,
/// MLP, product-key memory (`.pk.`), the brain parts (`.parts.`), and the rest. Sorted by name; shares sum to 1.
pub fn grad_shares(params: &[(String, Tensor)], grads: &candle_core::backprop::GradStore) -> Result<Vec<(String, f64)>> {
    let mut acc: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    let mut total = 0f64;
    for (name, p) in params {
        let Some(g) = grads.get(p) else { continue };
        let s = g.to_dtype(DType::F32)?.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let part = part_of(name);
        *acc.entry(part.to_string()).or_insert(0.0) += s;
        total += s;
    }
    Ok(acc.into_iter().map(|(k, v)| (k, if total > 0.0 { v / total } else { 0.0 })).collect())
}

/// The part a parameter belongs to (see `grad_shares`).
pub fn part_of(name: &str) -> &'static str {
    let tags = [(".eng.", "engram"), (".hebb.", "fast_memory"), (".mem.", "spiking"), (".attn.", "attention"), (".mlp.", "mlp"), (".pk.", "pk_memory"), (".parts.", "parts")];
    for (tag, part) in tags {
        if name.contains(tag) {
            return part;
        }
    }
    if name.starts_with("aux.") {
        // src/wave5.rs: the auxiliary heads
        return "aux_heads";
    }
    if name == crate::model::TOK_EMB || name == crate::model::POS_EMB || name == crate::model::PLACE_EMB || name == crate::model::MAG_W {
        return "embedding";
    }
    "other"
}

/// O2 --recall-loss: the cross-entropy (mean over the targets) of the engram read `r` [N, d] alone, through the final
/// norm and the tied embedding (both detached, so only the engram learns from it).
pub(crate) fn recall_ce(r: &Tensor, final_norm: &Tensor, tok: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let x = rmsnorm(r, &final_norm.detach())?;
    let logits = x.matmul(&tok.detach().t()?)?;
    crate::lif::cross_entropy_fused(&logits, &targets.flatten_all()?)
}

/// O2 --mem-reset N: the fast-weight memory's read with its state dropped at every N-token boundary (the chunks are
/// read from F = 0 each). q, k, v [b, nh, t, hs]; la, beta [b, nh, t].
pub(crate) fn delta_reset(q: &Tensor, k: &Tensor, v: &Tensor, la: &Tensor, beta: &Tensor, n: usize) -> Result<Tensor> {
    let t = q.dim(2)?;
    let mut ys = Vec::new();
    let mut t0 = 0;
    while t0 < t {
        let tc = n.min(t - t0);
        let c = |x: &Tensor| x.narrow(2, t0, tc).and_then(|x| x.contiguous());
        let (y, _) = crate::model::delta_rule_from(&c(q)?, &c(k)?, &c(v)?, &c(la)?, &c(beta)?, None)?;
        ys.push(y);
        t0 += tc;
    }
    Tensor::cat(&ys, 2)
}

#[cfg(test)]
mod tests;
