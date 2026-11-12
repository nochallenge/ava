// TESTING / DESIGN ONLY — ava_gpu::wave5: more Wave 5 brain parts (design-tests/nn/parts) as block options of model D.
//
// Like src/memory.rs's --dendrites, --stp, --delays and --wm, every option here is off by default (then nothing in this
// file runs), is set by a block flag (memory::MemOpts, so train_gpu/grow_gpu write it to DIR/arch.txt and
// ckpt::load_auto rebuilds the model), and every new weight starts where the part changes nothing. The one exception is
// --halting: its stop bias starts at 4 as the part has it, so at the start 1.8% of each token goes on to a second pass.
// The carried state of the parts that have one (forward_with_state) is in src/state.rs.
//
// In the block (src/arch.rs `block_arch`, src/state.rs `block_state`):
//   --modulate       B8 neuromodulation + F13 mood. A shadow pass (the same model with the modulators off, detached)
//                    gives each token's surprise s_t = -log p(x_t | x_<t); its running z-score z, slow average z̄
//                    (`ModRow`, the part's `zscores` with a carried start), mood (EMA of -z, v42) and arousal (EMA of |z|)
//                    drive, in each block: the fast-weight write strength b·relu(1 + a_w tanh z) (kept in [0, 2]),
//                    the forget rate alpha^relu(1 + a_r sigmoid(z - 2)) (a push to forget), the spiking layer's input
//                    current · (1 + a_i tanh z̄), and mood·w0 + arousal·w1 added to the block's input. a and w start at
//                    0. Costs one more forward pass (no backward) per pass.
//   --resonate P     E7 resonate-and-fire neurons beside the LIF, on the same input current (periods log-spaced from 2
//                    to P tokens): s = s_lif + m ⊙ (s_res - s_lif), m [d] starts at 0
//   --tem K          B10 structure apart from content: moves θ_t = h_t·Wθ (K phases) are path-integrated over the
//                    tokens, φ_t = Σ_{s<t} θ_s, codes g_t = [cos φ_t, sin φ_t]/√K, and the earlier tokens' normed inputs
//                    are read by structural similarity only (softmax over τ < t of 30·g_t·g_τ), then Wo (starts at 0)
//   --halting N      A7 PonderNet on the block's MLP: the step x + MLP(x) runs N times on its own output and the output is
//                    Σ p_n x_n, with a learned stop λ_n = sigmoid(x_n·w + b) (w = 0, b = 4 at the start); in training
//                    0.01 · KL(p ‖ geometric(0.8)) joins the loss
//   --cerebellum G   H11, after the MLP: the normed stream (detached) through a fixed sparse expansion to G granule units
//                    (fan-in 4, the top G/16 kept, seeded per block, not saved) and a learned read-out [G, d] (starts at
//                    0) added to the stream
//   --workspace N    B3, one shared global workspace in the middle block (layers/2), after the working memory: N slots
//                    that compete to be written (a softmax over salience, only the top 2 per token, times a write
//                    strength), read by attention and broadcast into the stream for every later block (read-out at 0);
//                    --ws-layer L (src/memory.rs, written by grow_exact) puts it in block L instead
//   --schema W      F12 with B3: an attention schema (W_s [d, slots] at 0) guesses, from the workspace's detached input,
//                    which slots win each token (softmax(h·W_s)); W · KL(shares ‖ guess), the shares detached. It only
//                    watches: the KL is a "distracted" signal to be read, and nothing it learns changes the model
//   --holo DB        D2, in every block after the workspace: a holographic memory of role-filler bindings (Plate 1995;
//                    ava_parts::holo). Each token binds a unit role r and a filler f by circular convolution into a
//                    fading trace T_t = λ·T_{t-1} + r_t ⊛ f_t (λ learned, 0.95 at the start) and reads the trace before
//                    it with its unit query q_t's inverse; the read through Wo (starts at 0). DB = the binding width
//   --sync N         L5, in every block before the MLP: binding by synchrony (Kuramoto, AKOrN-style; ava_parts::sync).
//                    N oscillators per token start from the normed stream, couple through a learned J for 4 steps
//                    (pulled by the token, each at its own learned speed), and the phases reached join through Wo (at 0)
// Auxiliary losses on the final normed state, in training forwards only (Model::loss_aux, loss_and_grads_autograd). The
// heads never change what the model computes, and the logged next-token loss is the plain one:
//   --jepa W         B1: a predictor (h + MLP(h), the MLP's output at 0) guesses the state 4 tokens on; W · mean(1 - cos)
//                    against the state itself, detached (the stop-gradient form, without an EMA copy of the network)
//   --diffusion W    N1: a block denoiser fills the next 8 tokens, a random share (10% to 100%) of them hidden, from the
//                    state just before them, on blocks every 16 tokens; W · the masked-diffusion loss
//   --critic W       B2: a value head V(h) learns the λ-return (γ 0.9, λ 0.95, its own values detached in the target,
//                    the window's last value bootstrapped) of the reward -(1 - γ)·CE per token; W · mean squared error
//   --confidence W   C8/F9: a feeling-of-knowing probe on the detached state learns whether the top-1 guess is right;
//                    W · BCE. Its input is detached, so it cannot change the model; it is there to be read.
// And one on the MLPs' activity:
//   --topo W         H12 topography: each block's MLP units sit on a ⌈√units⌉-wide sheet, and W · the mean over blocks
//                    of the part's spatial loss (1 − pearson of the units' correlations with 1/(1 + distance)), on at
//                    most 1,024 of the batch's tokens (every s-th), pulls neighbours toward responding alike
// D2/L5 binding is not wired: the part is a library of host functions (HRR bind/unbind, Kuramoto steps) without a
// module or weights; a block layer for it is still to be designed and tested in the parts crate.

use crate::memory::{Kv, MemOpts, NEG, WM_DK};
use crate::model::{kwta_mask, norm, rmsnorm, Config, Mlp, Model};
use ava_parts::{cerebellum::Cerebellum, confidence::Probe, critic::Critic, diffusion::BlockDenoiser, halting::Halting, jepa::Predictor, resonate::Resonate};
use ava_parts::modulate::{Mood, Neuromod};
use ava_parts::wm::{Gating, SlotMemory};
use ava_parts::schema::Schema;
use ava_parts::holo::HoloMem;
use ava_parts::sync::Oscillators;
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};
use std::cell::{Cell, RefCell};

/// B8: the z-score's EMA rate; F13: mood's and arousal's.
pub const MOD_RHO: f32 = 0.9;
pub const MOOD_RHO: f32 = 0.98;
/// B10: the sharpness of the structural read, and the spread of one move (radians) at the start.
pub const TEM_BETA: f64 = 30.0;
pub const TEM_STEP: f64 = 0.5;
/// A7: the geometric prior's rate and the KL's weight.
pub const HALT_PRIOR: f64 = 0.8;
pub const HALT_KL: f64 = 0.01;
/// H11: inputs per granule unit, the share kept (1 in 16), and the wiring seed.
pub const CB_FAN_IN: usize = 4;
pub const CB_SPARSITY: usize = 16;
pub const CB_SEED: u64 = 0xCE7E_B311;
/// B1: how many tokens on the predictor guesses, and its hidden width (× d).
pub const JEPA_K: usize = 4;
pub const JEPA_HIDDEN: usize = 2;
/// N1: the block length, the distance between blocks, and the least share hidden.
pub const DIFF_K: usize = 8;
pub const DIFF_STRIDE: usize = 16;
pub const DIFF_MIN_SHARE: f32 = 0.1;
/// B2: the discount and the λ of the λ-return.
pub const CRITIC_GAMMA: f64 = 0.9;
pub const CRITIC_LAMBDA: f64 = 0.95;
/// B3: how many slots each token may write.
pub const WS_TOPK: usize = 2;
/// H12: the most tokens the spatial loss looks at per block.
pub const TOPO_ROWS: usize = 1024;

/// B3: the block that holds the shared workspace.
pub fn ws_block(layers: usize) -> usize {
    layers / 2
}

/// B3: the workspace's block for the options `o`: --ws-layer when a grown model recorded one (src/grow.rs), else
/// `ws_block`.
pub fn ws_block_of(o: &MemOpts, layers: usize) -> usize {
    o.ws_layer.unwrap_or(ws_block(layers))
}

/// E7: the resonators and their mix into the spikes.
pub(crate) struct Res {
    pub part: Resonate,
    pub mix: Tensor,
}

/// B8 + F13: the gains of the three modulators and mood's vector.
pub(crate) struct Modl {
    pub nm: Neuromod,
    pub mood: Mood,
}

/// B10: the norm of the read's input, the moves' projection [d, K] and the output [d, d].
pub(crate) struct Tem {
    pub norm: Tensor,
    pub wth: Tensor,
    pub wo: Tensor,
    pub k: usize,
}

/// The parts of one block.
pub(crate) struct W5Block {
    pub res: Option<Res>,
    pub modl: Option<Modl>,
    pub tem: Option<Tem>,
    pub halt: Option<Halting>,
    pub cb: Option<Cerebellum>,
    /// B3 (in ws_block only): the read's input norm and the competing slots
    pub ws: Option<(Tensor, SlotMemory)>,
    /// F12 (with B3, in ws_block only): the attention schema, a model of the workspace's competition
    pub schema: Option<Schema>,
    /// D2: the read's input norm and the binding memory
    pub holo: Option<(Tensor, HoloMem)>,
    /// L5: the oscillators' input norm and the oscillators
    pub sync: Option<(Tensor, Oscillators)>,
}

/// The auxiliary heads (one per model).
pub(crate) struct Heads {
    pub jepa: Option<Predictor>,
    pub diff: Option<BlockDenoiser>,
    pub critic: Option<Critic>,
    pub conf: Option<Probe>,
}

/// What the last training forward left for the auxiliary losses (taken, so used once).
#[derive(Default)]
pub(crate) struct Stash {
    /// the final normed state [B·T, d], B and T
    pub xf: Option<(Tensor, usize, usize)>,
    /// its logits, detached [B·T, V]
    pub logits: Option<Tensor>,
    /// A7: each halting block's stop distribution [B·T, N]
    pub halt: Vec<Tensor>,
    /// H3b (--reflex): the training forward's token ids [b, t], for the one-pass (habit) forward
    pub ids: Option<Tensor>,
    /// F9 (--fok-qa): 1 where an answer-only step scores the target, else 0 [B·T]; None in an ordinary step
    pub scored: Option<Tensor>,
    /// per-block auxiliary losses, each a scalar already times its weight (H12, F12)
    pub extra: Vec<Tensor>,
}

/// Everything this file adds to a model (inside memory::MemExt).
pub(crate) struct W5Ext {
    pub blocks: Vec<W5Block>,
    pub heads: Heads,
    /// B8: on during the shadow pass (the modulators then do nothing and nothing is stashed)
    pub mod_off: Cell<bool>,
    /// H3b: on during the one-pass forward of --reflex (halting takes one pass; it runs as an evaluation forward, so
    /// nothing is stashed, and the firing rates are left alone)
    pub fast: Cell<bool>,
    pub stash: RefCell<Stash>,
    /// H11: the norm's unit gain [d] (a constant, not a weight)
    pub ones: Option<Tensor>,
    /// B2: the λ-return matrix of the last window length (T, [T, T])
    pub lam: RefCell<Option<(usize, Tensor)>>,
    /// H12: the sheet's pair weights, made at the first use
    pub topo: RefCell<Option<TopoPairs>>,
    /// H3b: --reflex is on (its target is the pondered logits)
    pub reflex: bool,
}

impl W5Ext {
    /// Whether a training forward should keep the final state (an auxiliary head is on).
    pub fn wants_xf(&self) -> bool {
        self.heads.jepa.is_some() || self.heads.diff.is_some() || self.heads.critic.is_some() || self.heads.conf.is_some() || self.reflex
    }

    pub fn wants_logits(&self) -> bool {
        self.heads.critic.is_some() || self.heads.conf.is_some() || self.reflex
    }

    pub fn clear_stash(&self) {
        *self.stash.borrow_mut() = Stash::default();
    }
}

/// Make this file's weights in `vb`, after every other tensor of the block options (so their random init is drawn as
/// before): the blocks' parts first, then the heads.
pub(crate) fn build(cfg: &Config, vb: &VarBuilder) -> Result<W5Ext> {
    let o = cfg.arch.mem;
    let d = cfg.d;
    let mut blocks = Vec::with_capacity(cfg.layers);
    for i in 0..cfg.layers {
        let vp = vb.pp(format!("b{i}.parts"));
        let res = if o.resonate > 0 {
            Some(Res { part: Resonate::new(vp.pp("res"), d, o.resonate as f64)?, mix: vp.get_with_hints(d, "res.mix", Init::Const(0.0))? })
        } else {
            None
        };
        let modl = if o.modulate { Some(Modl { nm: Neuromod::new(vp.pp("mod"))?, mood: Mood::new(vp.pp("mood"), d)? }) } else { None };
        let tem = if o.tem > 0 {
            Some(Tem {
                norm: vp.get_with_hints(d, "tem.norm", Init::Const(1.0))?,
                wth: vp.get_with_hints((d, o.tem), "tem.wth", Init::Randn { mean: 0.0, stdev: TEM_STEP / (d as f64).sqrt() })?,
                wo: vp.get_with_hints((d, d), "tem.wo", Init::Const(0.0))?,
                k: o.tem,
            })
        } else {
            None
        };
        let halt = if o.halting > 0 { Some(Halting::new(vp.pp("halt"), d)?) } else { None };
        let cb = if o.cerebellum > 0 {
            Some(Cerebellum::new(vp.pp("cb"), d, o.cerebellum, CB_FAN_IN, o.cerebellum / CB_SPARSITY, CB_SEED.wrapping_add(i as u64 * 7919))?)
        } else {
            None
        };
        let ws = if o.workspace > 0 && i == ws_block_of(&o, cfg.layers) {
            let norm = vp.get_with_hints(d, "ws.norm", Init::Const(1.0))?;
            Some((norm, SlotMemory::new(vp.pp("ws"), d, o.workspace, WM_DK, Gating::Compete { k: WS_TOPK })?))
        } else {
            None
        };
        let schema = if o.schema > 0.0 && ws.is_some() { Some(Schema::new(vp.pp("schema"), d, o.workspace)?) } else { None };
        let holo = if o.holo > 0 { Some((vp.get_with_hints(d, "holo.norm", Init::Const(1.0))?, HoloMem::new(vp.pp("holo"), d, o.holo)?)) } else { None };
        let sync = if o.sync > 0 { Some((vp.get_with_hints(d, "sync.norm", Init::Const(1.0))?, Oscillators::new(vp.pp("sync"), d, o.sync)?)) } else { None };
        blocks.push(W5Block { res, modl, tem, halt, cb, ws, schema, holo, sync });
    }
    let va = vb.pp("aux");
    let heads = Heads {
        jepa: if o.jepa > 0.0 { Some(Predictor::new(va.pp("jepa"), d, JEPA_HIDDEN * d)?) } else { None },
        diff: if o.diffusion > 0.0 { Some(BlockDenoiser::new(va.pp("diff"), cfg.vocab, DIFF_K, d)?) } else { None },
        critic: if o.critic > 0.0 { Some(Critic::new(va.pp("critic"), d)?) } else { None },
        conf: if o.confidence > 0.0 { Some(Probe::new(va.pp("conf"), d)?) } else { None },
    };
    let ones = if o.cerebellum > 0 { Some(Tensor::ones(d, DType::F32, vb.device())?) } else { None };
    Ok(W5Ext { blocks, heads, mod_off: Cell::new(false), fast: Cell::new(false), stash: RefCell::new(Stash::default()), ones, lam: RefCell::new(None), topo: RefCell::new(None), reflex: o.reflex > 0.0 })
}

/// The weights these options add (total, and how many of them are vocabulary tables: N1's embedding and output).
pub(crate) fn extra_weights(o: &MemOpts, c: &Config) -> (usize, usize) {
    let (d, l, v) = (c.d, c.layers, c.vocab);
    let mut n = 0;
    if o.resonate > 0 {
        n += l * 3 * d;
    }
    if o.modulate {
        n += l * (3 + 2 * d);
    }
    if o.tem > 0 {
        n += l * (d + d * o.tem + d * d);
    }
    if o.halting > 0 {
        n += l * (d + 1);
    }
    if o.cerebellum > 0 {
        n += l * o.cerebellum * d;
    }
    if o.workspace > 0 {
        let (k, dk) = (o.workspace, WM_DK);
        n += d + d * k + k + d * k * dk + d * dk + dk * d + d;
    }
    if o.schema > 0.0 && o.workspace > 0 {
        n += d * o.workspace;
    }
    if o.holo > 0 {
        n += l * (d + 3 * d * o.holo + o.holo * d + 1);
    }
    if o.sync > 0 {
        let k = o.sync;
        n += l * (d + 2 * d * 2 * k + k * k + k + 2 * k * d);
    }
    if o.jepa > 0.0 {
        n += 2 * d * JEPA_HIDDEN * d;
    }
    let mut emb = 0;
    if o.diffusion > 0.0 {
        emb = (v + 1) * d + d * v;
        n += emb + DIFF_K * d + 4 * d * d + 2 * d * 4 * d;
    }
    if o.critic > 0.0 {
        n += d + 1;
    }
    if o.confidence > 0.0 {
        n += d + 1;
    }
    (n, emb)
}

// ------------------------------------------------------------------------------------------- B8 + F13 signals

/// One row's running statistics (B8 z-score, F13 mood and arousal).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ModRow {
    /// the shadow pass's last logits of this row are in ModCarry::prev (the next token's surprise comes from them)
    pub has_prev: bool,
    /// the z-score's mean and variance have started (at the first token with a surprise)
    pub started: bool,
    pub mean: f32,
    pub var: f32,
    pub slow: f32,
    pub mood: f32,
    pub arousal: f32,
}

/// v42 (Amendment 15): F13's valence was −surprise in nats (about −6, not centred), so mood's weight acted as a bias
/// with a six-times input and carried almost all of the gradient. Valence is now −z, the surprise against its own running
/// mean and spread (what was expected). AVA_MOOD_RAW=1 restores −s.
fn mood_raw() -> bool {
    static RAW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *RAW.get_or_init(|| std::env::var("AVA_MOOD_RAW").is_ok_and(|v| v == "1"))
}

impl ModRow {
    /// One token with its surprise (None: nothing predicted it, the text's first token) → (z, z̄, mood, arousal). On a
    /// row whose every token has a surprise this is ava_parts::modulate::zscores and mood (valence -z, |z|) exactly.
    pub fn step(&mut self, s: Option<f32>) -> (f32, f32, f32, f32) {
        let Some(s) = s else { return (0.0, self.slow, self.mood, self.arousal) };
        if !self.started {
            self.mean = s;
            self.var = 1.0;
            self.slow = 0.0;
            self.started = true;
        }
        let (rho, rm) = (MOD_RHO, MOOD_RHO);
        let zt = (s - self.mean) / self.var.sqrt().max(1e-3);
        self.slow = rho * self.slow + (1.0 - rho) * zt;
        let dd = s - self.mean;
        self.mean += (1.0 - rho) * dd;
        self.var = rho * (self.var + (1.0 - rho) * dd * dd);
        let valence = if mood_raw() { -s } else { -zt };
        self.mood = rm * self.mood + (1.0 - rm) * valence;
        self.arousal = rm * self.arousal + (1.0 - rm) * zt.abs();
        (zt, self.slow, self.mood, self.arousal)
    }

    /// The row as 7 numbers (for State::save) and back.
    pub fn to_vec(&self) -> [f32; 7] {
        [self.has_prev as u8 as f32, self.started as u8 as f32, self.mean, self.var, self.slow, self.mood, self.arousal]
    }

    pub fn from_slice(v: &[f32]) -> ModRow {
        ModRow { has_prev: v[0] > 0.5, started: v[1] > 0.5, mean: v[2], var: v[3], slow: v[4], mood: v[5], arousal: v[6] }
    }
}

/// The modulators' carried state of B rows: each row's statistics and the shadow pass's last logits [B, V].
#[derive(Clone)]
pub struct ModCarry {
    pub rows: Vec<ModRow>,
    pub prev: Tensor,
}

/// The signals of one read, each [b, t].
#[derive(Clone)]
pub(crate) struct ModSig {
    pub z: Tensor,
    pub zbar: Tensor,
    pub mood: Tensor,
    pub arousal: Tensor,
}

/// B8/F13: the signals of reading `ids` [b, t] from the shadow pass's logits [b·t, V] (detached), after `carry` (None: a
/// fresh text; else it moves on). Token i's surprise is row i − 1's cross-entropy against it; the first token's comes
/// from the carried last logits, or is None at a text's start.
pub(crate) fn mod_signals(logits: &Tensor, ids: &Tensor, carry: Option<&mut ModCarry>) -> Result<ModSig> {
    let (b, t) = ids.dims2()?;
    let dev = ids.device();
    let v = logits.dim(1)?;
    let idv: Vec<Vec<u32>> = ids.to_vec2()?;
    let next: Vec<u32> = idv.iter().flat_map(|r| r[1..].iter().copied().chain(std::iter::once(v as u32))).collect();
    let ce: Vec<f32> = crate::lif::cross_entropy_rows(&logits.contiguous()?, &Tensor::from_vec(next, b * t, dev)?)?.to_vec1()?;
    let mut fresh = vec![ModRow::default(); b];
    let (rows, first): (&mut [ModRow], Option<Vec<f32>>) = match carry {
        Some(c) => {
            let first = if c.rows.iter().any(|r| r.has_prev) {
                Some(crate::lif::cross_entropy_rows(&c.prev.contiguous()?, &ids.narrow(1, 0, 1)?.flatten_all()?.contiguous()?)?.to_vec1()?)
            } else {
                None
            };
            c.prev = logits.reshape((b, t, v))?.narrow(1, t - 1, 1)?.squeeze(1)?.contiguous()?;
            (&mut c.rows, first)
        }
        None => (&mut fresh, None),
    };
    let mut out = [Vec::with_capacity(b * t), Vec::with_capacity(b * t), Vec::with_capacity(b * t), Vec::with_capacity(b * t)];
    for (r, row) in rows.iter_mut().enumerate() {
        for i in 0..t {
            let s = if i > 0 {
                Some(ce[r * t + i - 1])
            } else if row.has_prev {
                first.as_ref().map(|f| f[r])
            } else {
                None
            };
            let (z, zb, m, a) = row.step(s);
            out[0].push(z);
            out[1].push(zb);
            out[2].push(m);
            out[3].push(a);
        }
        row.has_prev = true;
    }
    let [z, zbar, mood, arousal] = out;
    let mk = |x: Vec<f32>| Tensor::from_vec(x, (b, t), dev);
    Ok(ModSig { z: mk(z)?, zbar: mk(zbar)?, mood: mk(mood)?, arousal: mk(arousal)? })
}

/// One block's modulators for a read: (the part's Mods: write, reset, input [b, t]; mood's addition [b·t, d]).
pub(crate) fn block_mods(md: &Modl, sig: &ModSig) -> Result<(ava_parts::modulate::Mods, Tensor)> {
    let mods = md.nm.forward(&sig.z, &sig.zbar)?;
    let add = md.mood.forward(&sig.mood, &sig.arousal)?;
    let (b, t, d) = add.dims3()?;
    Ok((mods, add.reshape((b * t, d))?))
}

/// B8 in the fast-weight memory: log alpha · relu(1 + reset) and the write strength · relu(write), kept in [0, 2] (the
/// delta rule's stable range). `la` and `beta` [b, nh, t, g]; write and reset [b, t]. With the gains at 0 both come
/// back exactly as given.
pub(crate) fn mod_gates(la: Tensor, beta: Tensor, write: &Tensor, reset: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, t) = write.dims2()?;
    let r = reset.affine(1.0, 1.0)?.relu()?.reshape((b, 1, t, 1))?;
    let w = write.relu()?.reshape((b, 1, t, 1))?;
    Ok((la.broadcast_mul(&r)?, beta.broadcast_mul(&w)?.clamp(0f32, 2f32)?))
}

// ------------------------------------------------------------------------------------------- E7 with a start state

/// E7: the resonators' membrane Re z [B, T, C] for `cur` [B, T, C] from a carried (Re z, Im z) [B, C]: the part's own
/// `membrane` (the same ops in the same order) with its start state given. Returns the membrane and the end state.
pub(crate) fn res_from(p: &Resonate, cur: &Tensor, init: &(Tensor, Tensor)) -> Result<(Tensor, (Tensor, Tensor))> {
    let t = cur.dim(1)?;
    let r = candle_nn::ops::sigmoid(&p.r_param)?;
    let om = p.omega()?;
    let (rc, rs) = ((&r * om.cos()?)?, (&r * om.sin()?)?);
    let (mut re, mut im) = (init.0.clone(), init.1.clone());
    let mut out = Vec::with_capacity(t);
    for k in 0..t {
        let i = cur.narrow(1, k, 1)?.squeeze(1)?;
        let re2 = ((re.broadcast_mul(&rc)? - im.broadcast_mul(&rs)?)? + i)?;
        im = (re.broadcast_mul(&rs)? + im.broadcast_mul(&rc)?)?;
        re = re2;
        out.push(re.clone());
    }
    Ok((Tensor::stack(&out, 1)?, (re, im)))
}

/// E7: the spikes mixed in: s + m ⊙ (s_res − s) (m = 0: s exactly).
pub(crate) fn res_mix(s: &Tensor, s_res: &Tensor, mix: &Tensor) -> Result<Tensor> {
    s.add(&s_res.sub(s)?.broadcast_mul(mix)?)
}

// ------------------------------------------------------------------------------------------- B10 the structure read

/// B10: the structure read of the normed input h [b·t, d] (see the top of the file) after the earlier tokens in
/// `store` (S1; None: a fresh text) and the phase reached so far [b, K] (None: 0). Returns the output [b·t, d] (before
/// the residual add); the store and the phase move on when given. With no store this is ava_parts::tem::recall on the
/// codes of the moves (the first token reads nothing).
pub(crate) fn tem_read(h: &Tensor, tp: &Tem, b: usize, t: usize, store: Option<&mut Option<Kv>>, phase: Option<&mut Tensor>) -> Result<Tensor> {
    if tem_diag().self_read && store.is_none() && phase.is_none() {
        return tem_values(h).matmul(&tp.wo);
    }
    let (n, d) = h.dims2()?;
    let k = tp.k;
    let dev = h.device();
    if store.is_none() && phase.is_none() && tem_fast(dev, k, d) {
        return tem_read_fast(h, tp, b, t);
    }
    let th = tem_moves(h, tp)?.reshape((b, t, k))?;
    let cs = th.cumsum(1)?; // the moves up to and including each token
    let z = Tensor::zeros((b, 1, k), DType::F32, dev)?;
    let before = if t > 1 { Tensor::cat(&[&z, &cs.narrow(1, 0, t - 1)?], 1)? } else { z };
    let last = cs.narrow(1, t - 1, 1)?.squeeze(1)?;
    let phi = match phase {
        Some(p) => {
            let phi = before.broadcast_add(&p.unsqueeze(1)?)?;
            *p = p.add(&last)?;
            phi
        }
        None => before,
    };
    let g = (Tensor::cat(&[&phi.cos()?, &phi.sin()?], 2)? / (k as f64).sqrt())?; // [b, t, 2K]
    let x = tem_values(h).reshape((b, t, d))?;
    let old: Option<Kv> = store.as_ref().and_then(|s| (**s).clone());
    let s0 = old.as_ref().map_or(0, |s| s.k.dim(2).unwrap_or(0));
    let (keys, vals) = match &old {
        None => (g.clone(), x.clone()),
        Some(s) => (Tensor::cat(&[&s.k.squeeze(1)?, &g], 1)?, Tensor::cat(&[&s.v.squeeze(1)?, &x], 1)?),
    };
    let sc = g.matmul(&keys.t()?.contiguous()?)?.affine(tem_diag().beta, 0.0)?; // [b, t, s0 + t]
    // only strictly earlier tokens (the store's slots, and the new tokens before this one)
    let m: Vec<f32> = (0..t).flat_map(|i| (0..s0 + t).map(move |j| if j < s0 || j - s0 < i { 0.0 } else { NEG })).collect();
    let mut sm = sc.broadcast_add(&Tensor::from_vec(m, (t, s0 + t), dev)?)?;
    let pad = match old.as_ref().and_then(|s| s.pad.as_ref()) {
        Some(p) => Some(Tensor::cat(&[p, &Tensor::zeros((b, t), DType::F32, dev)?], 1)?),
        None => None,
    };
    if let Some(p) = &pad {
        sm = sm.broadcast_add(&p.unsqueeze(1)?)?;
    }
    let p = candle_nn::ops::softmax(&sm, D::Minus1)?;
    let valid_old: Vec<usize> = old.as_ref().map_or_else(|| vec![0; b], |s| s.valid.clone());
    // a token with nothing earlier to read reads nothing (the text's first token)
    let vo = &valid_old;
    let seen: Vec<f32> = (0..b).flat_map(|r| (0..t).map(move |i| if vo[r] + i > 0 { 1.0 } else { 0.0 })).collect();
    let y = p.matmul(&vals)?.broadcast_mul(&Tensor::from_vec(seen, (b, t, 1), dev)?)?;
    if let Some(st) = store {
        let (nk, nv) = (g.unsqueeze(1)?, x.unsqueeze(1)?);
        let (kc, vc) = match &old {
            None => (nk.contiguous()?, nv.contiguous()?),
            Some(s) => (Tensor::cat(&[&s.k, &nk], 2)?, Tensor::cat(&[&s.v, &nv], 2)?),
        };
        let valid = valid_old.iter().map(|v| v + t).collect();
        *st = Some(Kv { k: kc, v: vc, pad, valid, logp: None, ktail: None, reads: None, age: None });
    }
    y.reshape((n, d))?.matmul(&tp.wo)
}

/// v42 (Amendment 13): what B10's moves read. Every token's phase is a running sum of the moves, so one move's gradient
/// gathers terms from every later token (sharpened by TEM_BETA); passed on into h, that large noisy gradient reached
/// every upstream weight and, in a d128 run, cost 0.58 nats at step 500. So the moves read h detached: they still learn
/// (through wth), the values path still passes its gradient, and a read without gradient (talking, eval) is unchanged.
/// AVA_TEM_PHASE_GRAD=1 restores the old gradient.
fn tem_moves(h: &Tensor, tp: &Tem) -> Result<Tensor> {
    let dg = tem_diag();
    let h = if dg.phase_grad { h.clone() } else { h.detach() };
    let w = if dg.freeze_moves { tp.wth.detach() } else { tp.wth.clone() };
    h.matmul(&w)
}

/// v42 (Amendment 13b): switches for finding what in B10 still costs loss, read once from the environment.
/// AVA_TEM_FREEZE_MOVES=1 keeps the moves' weights at their init; AVA_TEM_DETACH_VALUES=1 stops the values' gradient
/// into h (so B10 sends no gradient upstream at all); AVA_TEM_BETA sets the read's sharpness (default TEM_BETA);
/// AVA_TEM_SELF=1 (Amendment 13d, a control) reads only the token itself, so B10 is just an extra layer h·wo.
struct TemDiag {
    phase_grad: bool,
    freeze_moves: bool,
    detach_values: bool,
    self_read: bool,
    beta: f64,
}

fn tem_diag() -> &'static TemDiag {
    static D: std::sync::OnceLock<TemDiag> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        let on = |n: &str| std::env::var(n).is_ok_and(|v| v == "1");
        TemDiag {
            phase_grad: on("AVA_TEM_PHASE_GRAD"),
            freeze_moves: on("AVA_TEM_FREEZE_MOVES"),
            detach_values: on("AVA_TEM_DETACH_VALUES"),
            self_read: on("AVA_TEM_SELF"),
            beta: std::env::var("AVA_TEM_BETA").ok().and_then(|v| v.parse().ok()).unwrap_or(TEM_BETA),
        }
    })
}

fn tem_values(h: &Tensor) -> Tensor {
    if tem_diag().detach_values {
        h.detach()
    } else {
        h.clone()
    }
}

/// Whether B10's training read can run fused: CUDA, codes of the attention kernels' width (2K = 32), the values in
/// whole 32-wide heads, and the fused scans on (AVA_SCAN_REF=1 turns this off too).
fn tem_fast(dev: &candle_core::Device, k: usize, d: usize) -> bool {
    use crate::fused::ATTN_HS;
    dev.is_cuda() && 2 * k == ATTN_HS && d % ATTN_HS == 0 && crate::scan_fast::on()
}

/// v42: B10's read in training (no store, no phase) without the [t, t] scores: the same codes g [b, t, 2K] are the
/// query and key of every value head, the values h split into d/32 heads, through the fused causal attention over
/// strictly earlier tokens (row 0 reads 0, as tem_read's `seen`). The same maths as tem_read up to rounding.
pub(crate) fn tem_read_fast(h: &Tensor, tp: &Tem, b: usize, t: usize) -> Result<Tensor> {
    use crate::fused::ATTN_HS;
    let (n, d) = h.dims2()?;
    let k = tp.k;
    let th = tem_moves(h, tp)?.reshape((b, t, k))?;
    let cs = th.cumsum(1)?;
    let z = Tensor::zeros((b, 1, k), DType::F32, h.device())?;
    let before = if t > 1 { Tensor::cat(&[&z, &cs.narrow(1, 0, t - 1)?], 1)? } else { z };
    let g = (Tensor::cat(&[&before.cos()?, &before.sin()?], 2)? / (k as f64).sqrt())?; // [b, t, 2K]
    let nh = d / ATTN_HS;
    let gq = g.unsqueeze(1)?.broadcast_as((b, nh, t, 2 * k))?.contiguous()?;
    let v = tem_values(h).reshape((b, t, nh, ATTN_HS))?.transpose(1, 2)?.contiguous()?;
    let y = crate::fused::causal_attention_strict(&gq, &gq, &v, tem_diag().beta)?; // [b, nh, t, 32]
    y.transpose(1, 2)?.reshape((n, d))?.matmul(&tp.wo)
}

// ------------------------------------------------------------------------------------------- A7 and H11 in the block

/// One MLP step of block D: x + MLP(norm(x)) with kWTA (relu² with --mlp-act relu2), and the activations (the same ops
/// as the block's own MLP).
pub(crate) fn mlp_step(m: &Model, l: &Mlp, x: &Tensor) -> Result<(Tensor, Tensor)> {
    let h = norm(x, &l.norm)?;
    let u = crate::int8::mm(&h, &l.w1)?;
    let k = m.cfg.kwta_k();
    let mut a = if crate::fast_path() && u.device().is_cuda() { crate::fused::kwta_relu(&u, k)? } else { u.mul(&kwta_mask(&u, k)?)?.relu()? };
    if m.cfg.arch.relu2 {
        a = a.sqr()?;
    }
    Ok((x.add(&crate::int8::mm(&a, &l.w2)?)?, a))
}

/// A7: the MLP step pondered by the part (Halting::run, N passes): (output, the first pass's activations, p [n, N],
/// the first pass's output (H3)).
pub(crate) fn halt_mlp(m: &Model, l: &Mlp, hl: &Halting, x: &Tensor, n_max: usize) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let first: RefCell<Option<(Tensor, Tensor)>> = RefCell::new(None);
    let step = |x: &Tensor| -> Result<Tensor> {
        let (y, a) = mlp_step(m, l, x)?;
        if first.borrow().is_none() {
            *first.borrow_mut() = Some((a, y.clone()));
        }
        Ok(y)
    };
    let pd = hl.run(x, &step, n_max)?;
    let (a, y1) = first.into_inner().ok_or_else(|| candle_core::Error::Msg("halting: no pass".into()))?;
    Ok((pd.output, a, pd.p, y1))
}

/// H11: the cerebellum's correction added to x [n, d]: the part's sparse code of the normed, detached stream (top k by
/// the model's kWTA mask, which has no width limit on the GPU), through the read-out.
pub(crate) fn cb_add(cb: &Cerebellum, ones: &Tensor, x: &Tensor) -> Result<Tensor> {
    let h = rmsnorm(&x.detach(), ones)?;
    let u = h.matmul(&cb.expand)?.relu()?;
    let code = u.mul(&kwta_mask(&u, cb.k)?)?;
    x.add(&code.matmul(&cb.readout)?)
}

// ------------------------------------------------------------------------------------------- the auxiliary losses

/// B2: λ-return targets [b, t] for rewards r and values v [b, t] (both detached), the value after the window
/// bootstrapped by the last one: G_t = r_t + γ((1 − λ)v_{t+1} + λ G_{t+1}), G_T = v_{T−1}. As
/// ava_parts::critic::lambda_returns with v[T] = v[T − 1], in one matmul (the [T, T] matrix is kept per length).
pub fn lambda_targets(r: &Tensor, v: &Tensor, gamma: f64, lambda: f64, cache: Option<&RefCell<Option<(usize, Tensor)>>>) -> Result<Tensor> {
    let (_, t) = r.dims2()?;
    let dev = r.device();
    let gl = gamma * lambda;
    let v_next = if t > 1 { Tensor::cat(&[&v.narrow(1, 1, t - 1)?, &v.narrow(1, t - 1, 1)?], 1)? } else { v.clone() };
    let u = r.add(&v_next.affine(gamma * (1.0 - lambda), 0.0)?)?;
    let build = || -> Result<Tensor> {
        // up[s, i] = (γλ)^(s − i) for s >= i
        let m: Vec<f32> = (0..t).flat_map(|s| (0..t).map(move |i| if s >= i { gl.powi((s - i) as i32) as f32 } else { 0.0 })).collect();
        Tensor::from_vec(m, (t, t), dev)
    };
    let up = match cache {
        Some(c) => {
            let hit = c.borrow().as_ref().filter(|(n, m)| *n == t && m.device().same_device(dev)).map(|(_, m)| m.clone());
            match hit {
                Some(m) => m,
                None => {
                    let m = build()?;
                    *c.borrow_mut() = Some((t, m.clone()));
                    m
                }
            }
        }
        None => build()?,
    };
    let tail: Vec<f32> = (0..t).map(|i| gl.powi((t - i) as i32) as f32).collect();
    let tail = Tensor::from_vec(tail, (1, t), dev)?;
    u.matmul(&up)?.add(&v.narrow(1, t - 1, 1)?.broadcast_mul(&tail)?)
}

/// N1: the blocks of a window: for every row, the state at p (after reading x_0..x_p) and the next DIFF_K tokens
/// targets[p..p + K), at p = K − 1, K − 1 + DIFF_STRIDE, ..; blocks with a target outside the vocabulary are left out.
/// Returns (row-major positions into [b·t], the blocks).
pub fn diffusion_blocks(tg: &[Vec<u32>], t: usize, vocab: usize) -> (Vec<u32>, Vec<Vec<u32>>) {
    let (mut at, mut blocks) = (Vec::new(), Vec::new());
    for (r, row) in tg.iter().enumerate() {
        let mut p = DIFF_K - 1;
        while p + DIFF_K <= t {
            let blk = &row[p..p + DIFF_K];
            if blk.iter().all(|&x| (x as usize) < vocab) {
                at.push((r * t + p) as u32);
                blocks.push(blk.to_vec());
            }
            p += DIFF_STRIDE;
        }
    }
    (at, blocks)
}

/// A seed from the window's targets (FNV-1a), so the hidden tokens are the same whenever the same batch comes back.
fn batch_seed(tg: &[Vec<u32>]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in tg.iter().flat_map(|r| r.iter().take(64)) {
        for byte in x.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// K4 (many GPUs): the whole batch's target rows and which of them this rank holds, in order. The hidden tokens are
/// drawn from the whole batch (its seed, its block order), so a rank must draw them for the whole batch and keep its
/// own rows' blocks; then W ranks compute what one process computes. None (the default): this process has it all.
pub struct RowSplit {
    pub full: Vec<Vec<u32>>,
    pub rows: Vec<usize>,
}

thread_local! {
    static ROW_SPLIT: RefCell<Option<RowSplit>> = const { RefCell::new(None) };
}

/// K4: set (or clear) the batch split for this thread's next forward passes.
pub fn set_row_split(s: Option<RowSplit>) {
    ROW_SPLIT.with(|c| *c.borrow_mut() = s);
}

fn diffusion_loss(dn: &BlockDenoiser, x3: &Tensor, targets: &Tensor) -> Result<Option<Tensor>> {
    let (b, t, d) = x3.dims3()?;
    let dev = x3.device();
    let local: Vec<Vec<u32>> = targets.to_vec2()?;
    // K4: the split applies only to the batch it was set for (its rows are exactly these); any other forward (ReDo,
    // probes) holds a whole batch of its own
    let split = ROW_SPLIT.with(|c| {
        c.borrow().as_ref().filter(|s| s.rows.len() == b && s.rows.iter().zip(&local).all(|(&r, row)| s.full.get(r) == Some(row))).map(|s| (s.full.clone(), s.rows.clone()))
    });
    let (tg, rows) = match split {
        Some((full, rows)) => (full, Some(rows)),
        None => (local, None),
    };
    let (at, blocks) = diffusion_blocks(&tg, t, dn.vocab);
    if blocks.is_empty() {
        return Ok(None);
    }
    let nb = blocks.len();
    let seed = batch_seed(&tg);
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let share: Vec<f32> = (0..nb)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            DIFF_MIN_SHARE + (1.0 - DIFF_MIN_SHARE) * ((s >> 11) as f64 / (1u64 << 53) as f64) as f32
        })
        .collect();
    let (blk, masked) = dn.mask(&blocks, &share, seed ^ 0x5bd1_e995)?;
    // (factor: a rank's mean over its blocks, times its share of the whole batch's blocks times W, so that the ranks'
    // average is the whole batch's mean even when blocks with a token outside the vocabulary left the rows uneven)
    let nb_full = nb;
    let (at, blocks, share, blk, masked, factor) = match rows {
        None => (at, blocks, share, blk, masked, 1.0),
        Some(rows) => {
            let w = tg.len() as f64 / rows.len() as f64;
            // K4: keep this rank's blocks, with their positions moved into its own rows
            let (mut keep, mut at_here) = (Vec::new(), Vec::new());
            for (k, &p) in at.iter().enumerate() {
                if let Some(i) = rows.iter().position(|&r| r == p as usize / t) {
                    keep.push(k as u32);
                    at_here.push((i * t + p as usize % t) as u32);
                }
            }
            if keep.is_empty() {
                return Ok(None);
            }
            let kt = Tensor::from_vec(keep.clone(), keep.len(), dev)?;
            let bl: Vec<Vec<u32>> = keep.iter().map(|&k| blocks[k as usize].clone()).collect();
            let sh: Vec<f32> = keep.iter().map(|&k| share[k as usize]).collect();
            let f = keep.len() as f64 * w / nb_full as f64;
            (at_here, bl, sh, blk.index_select(&kt, 0)?, masked.index_select(&kt, 0)?, f)
        }
    };
    let nb = blocks.len();
    let h = x3.reshape((b * t, d))?.index_select(&Tensor::from_vec(at, nb, dev)?, 0)?;
    let logits = dn.logits(&h, &blk)?;
    let tgt = Tensor::from_vec(blocks.concat(), (nb, DIFF_K), dev)?;
    let loss = ava_parts::diffusion::mdlm_loss(&logits, &tgt, &masked, &Tensor::from_vec(share, nb, dev)?)?;
    Ok(Some(if factor == 1.0 { loss } else { loss.affine(factor, 0.0)? }))
}

/// The auxiliary losses of the last training forward, each times its weight, summed (None: none is on, or nothing was
/// stashed). The stash is always taken (so a forward's graph is not kept); the losses are computed only with `on`.
pub(crate) fn aux_loss(m: &Model, w5: &W5Ext, targets: &Tensor, on: bool) -> Result<Option<Tensor>> {
    let st = std::mem::take(&mut *w5.stash.borrow_mut());
    if !on {
        return Ok(None);
    }
    let o = m.cfg.arch.mem;
    let mut terms: Vec<Tensor> = Vec::new();
    for p in &st.halt {
        terms.push(ava_parts::halting::kl_to_geometric(p, HALT_PRIOR)?.mean_all()?.affine(HALT_KL, 0.0)?);
    }
    terms.extend(st.extra.iter().cloned());
    if let Some((xf, b, t)) = &st.xf {
        let (b, t) = (*b, *t);
        let d = xf.dim(1)?;
        let x3 = xf.reshape((b, t, d))?;
        if let Some(p) = &w5.heads.jepa {
            if t > JEPA_K {
                terms.push(ava_parts::jepa::jepa_loss(&p.forward(&x3)?, &x3, JEPA_K)?.affine(o.jepa as f64, 0.0)?);
            }
        }
        if let Some(dn) = &w5.heads.diff {
            if let Some(l) = diffusion_loss(dn, &x3, targets)? {
                terms.push(l.affine(o.diffusion as f64, 0.0)?);
            }
        }
        if w5.wants_logits() {
            let logits = match &st.logits {
                Some(l) => l.clone(),
                None => xf.detach().matmul(&m.tok.detach().t()?)?,
            };
            let tg = targets.flatten_all()?.contiguous()?;
            if let (true, Some(ids)) = (o.reflex > 0.0, &st.ids) {
                // H3b, practice to habit: the same text again in one pass (each pondering block takes its first pass
                // only), and that quick answer learns the pondered one: KL(pondered ‖ one pass) per token, times the
                // pondered answer's success (its right token's probability, detached), so a wrong answer is never made
                // a habit; targets outside the vocabulary count 0. The pondered side is the target (detached).
                // (H3's first design pulled the first MLP pass toward the pondered block output; it failed, prereg 17j.)
                w5.fast.set(true);
                let trf = m.train_fwd.replace(false);
                let trk = m.track.replace(false);
                let r = m.forward_arch(ids);
                w5.fast.set(false);
                m.train_fwd.set(trf);
                m.track.set(trk);
                let (xs, _) = r?;
                let fast = crate::int8::head(&xs, &m.tok)?;
                let v = logits.dim(1)?;
                let lp = candle_nn::ops::log_softmax(&logits.detach(), D::Minus1)?;
                let lq = candle_nn::ops::log_softmax(&fast, D::Minus1)?;
                let kl = lp.exp()?.mul(&lp.sub(&lq)?)?.sum(D::Minus1)?;
                let s = crate::lif::cross_entropy_rows(&logits.contiguous()?, &tg)?.neg()?.exp()?;
                let w = s.mul(&tg.lt(v as u32)?.to_dtype(DType::F32)?)?.detach();
                terms.push(kl.mul(&w)?.mean_all()?.affine(o.reflex as f64, 0.0)?);
            }
            if let Some(c) = &w5.heads.critic {
                let v = c.value(&x3)?; // [b, t]
                let r = crate::lif::cross_entropy_rows(&logits.contiguous()?, &tg)?.reshape((b, t))?.affine(-(1.0 - CRITIC_GAMMA), 0.0)?;
                let g = lambda_targets(&r, &v.detach(), CRITIC_GAMMA, CRITIC_LAMBDA, Some(&w5.lam))?;
                terms.push(v.sub(&g)?.sqr()?.mean_all()?.affine(o.critic as f64, 0.0)?);
            }
            if let Some(pr) = &w5.heads.conf {
                let right = logits.argmax(D::Minus1)?.eq(&tg)?.to_dtype(DType::F32)?;
                // F9 (--fok-qa): only where the answer is scored, so the question's last token learns "will my answer
                // be right" (BCE as the probe's, averaged over the scored positions); otherwise the probe's own loss
                let probe_term = |x: &Tensor, right: &Tensor| -> Result<Tensor> {
                    match &st.scored {
                        Some(m) => {
                            let z = pr.logits(x)?;
                            let bce = z.relu()?.add(&z.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?.sub(&right.mul(&z)?)?;
                            let n = m.sum_all()?.to_scalar::<f32>()?.max(1.0) as f64;
                            bce.mul(m)?.sum_all()?.affine(o.confidence as f64 / n, 0.0)
                        }
                        None => pr.loss(x, right)?.affine(o.confidence as f64, 0.0),
                    }
                };
                terms.push(probe_term(xf, &right)?);
                if o.fok_push > 0.0 {
                    // F10 follow-up: the same lesson on her final state pushed off its usual range (one random unit
                    // direction per step, strength uniform in [-S, S] times the mean state length), labelled by
                    // whether the pushed state's own guess is right. Detached: only the probe learns from it. The
                    // direction is seeded by the batch's targets, so a recomputed or resumed step pushes the same way.
                    let k = tg.to_vec1::<u32>()?.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3));
                    let mut r = crate::sae::Rng::new(0xF10_5EED ^ k);
                    let d = xf.dim(1)?;
                    let u: Vec<f64> = (0..d).map(|_| r.normal()).collect();
                    let un = u.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
                    let s = (2.0 * r.uniform() - 1.0) * o.fok_push as f64;
                    let u = Tensor::from_vec(u.iter().map(|x| (x / un * s) as f32).collect::<Vec<_>>(), d, xf.device())?;
                    let xd = xf.detach();
                    let len = xd.sqr()?.sum(D::Minus1)?.sqrt()?.mean_all()?;
                    let xp = xd.broadcast_add(&u.broadcast_mul(&len)?)?;
                    let right_p = xp.matmul(&m.tok.detach().t()?)?.argmax(D::Minus1)?.eq(&tg)?.to_dtype(DType::F32)?;
                    terms.push(probe_term(&xp, &right_p)?);
                }
            }
        }
    }
    let mut it = terms.into_iter();
    let Some(first) = it.next() else { return Ok(None) };
    it.try_fold(first, |a, x| a.add(&x)).map(Some)
}

// ------------------------------------------------------------------------------------------- F12 attention schema

/// F12: KL(the workspace's write shares ‖ the schema's guess), the mean over tokens; h [b, t, d] is the workspace's
/// normed input. The shares are the gates over their sum (the part's competing softmax before the write strength).
pub(crate) fn schema_loss(sc: &Schema, ws: &SlotMemory, h: &Tensor) -> Result<Tensor> {
    let g = ws.gates(&h.detach())?;
    let a = g.broadcast_div(&(g.sum_keepdim(D::Minus1)? + 1e-9)?)?;
    ava_parts::schema::surprise(&a, &sc.predict(h)?)?.mean_all()
}

// ------------------------------------------------------------------------------------------- H12 topography

/// H12: the pair weights of `u` units on a topo_width(u)-wide sheet: the pairs i < j (a mask), and 1/(1 + distance)
/// centred over those pairs (0 elsewhere) with its sum of squares, so each step needs no host work.
pub(crate) struct TopoPairs {
    u: usize,
    mask: Tensor,
    dc: Tensor,
    dss: f64,
    pairs: f64,
}

pub fn topo_width(u: usize) -> usize {
    (u as f64).sqrt().ceil() as usize
}

fn topo_pairs(u: usize, dev: &candle_core::Device) -> Result<TopoPairs> {
    let inv = ava_parts::topo::inverse_distance(u, topo_width(u), dev)?;
    let m: Vec<f32> = (0..u * u).map(|k| if k / u < k % u { 1.0 } else { 0.0 }).collect();
    let mask = Tensor::from_vec(m, (u, u), dev)?;
    let pairs = (u * (u - 1) / 2) as f64;
    let mean = inv.mul(&mask)?.sum_all()?.to_scalar::<f32>()? as f64 / pairs;
    let dc = inv.affine(1.0, -mean)?.mul(&mask)?;
    let dss = dc.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    Ok(TopoPairs { u, mask, dc, dss, pairs })
}

/// H12: ava_parts::topo::spatial_loss of activations a [n, u] on every s-th of its rows (at most TOPO_ROWS), with the
/// pair weights cached in `w5`. The same maths as the part: pearson over the pairs i < j of the units' correlations
/// and the inverse distances, written as sums over the masked pairs.
pub(crate) fn topo_loss(w5: &W5Ext, a: &Tensor) -> Result<Tensor> {
    let (n, u) = a.dims2()?;
    if w5.topo.borrow().as_ref().is_none_or(|p| p.u != u) {
        *w5.topo.borrow_mut() = Some(topo_pairs(u, a.device())?);
    }
    let pb = w5.topo.borrow();
    let tp = pb.as_ref().expect("made above");
    let s = n.div_ceil(TOPO_ROWS);
    let r = if s == 1 {
        a.clone()
    } else if n % s == 0 {
        a.reshape((n / s, s, u))?.narrow(1, 0, 1)?.squeeze(1)?
    } else {
        let idx: Vec<u32> = (0..n).step_by(s).map(|i| i as u32).collect();
        let k = idx.len();
        a.index_select(&Tensor::from_vec(idx, k, a.device())?, 0)?
    };
    topo_from(&r, tp)
}

fn topo_from(r: &Tensor, tp: &TopoPairs) -> Result<Tensor> {
    let rc = r.broadcast_sub(&r.mean_keepdim(0)?)?;
    let sd = (rc.sqr()?.sum_keepdim(0)? + 1e-8)?.sqrt()?;
    let z = rc.broadcast_div(&sd)?;
    let corr = z.t()?.matmul(&z)?;
    let cm = corr.mul(&tp.mask)?;
    let (sc, scc) = (cm.sum_all()?, cm.sqr()?.sum_all()?);
    let num = corr.mul(&tp.dc)?.sum_all()?;
    let var_c = (scc - (sc.sqr()? / tp.pairs)?)?;
    let den = var_c.affine(tp.dss, 0.0)?.sqrt()?;
    num.div(&(den + 1e-12)?)?.affine(-1.0, 1.0)
}

impl Model {
    /// F9: her feeling of knowing at every position of `ids` [B, T]: the C8 probe's estimate, from her state before
    /// she guesses, that her top-1 guess of the next token will be right ([B·T], in (0, 1)). None for a model trained
    /// without --confidence. A read only: one forward, no gradient, nothing stashed for a loss.
    pub fn feeling_of_knowing(&self, ids: &Tensor) -> Result<Option<Tensor>> {
        let Some(pr) = self.ext.as_ref().and_then(|e| e.mem.w5.heads.conf.as_ref()) else {
            return Ok(None);
        };
        let _q8 = self.int8_scope();
        let (xf, _) = self.forward_arch(ids)?;
        Ok(Some(candle_nn::ops::sigmoid(&pr.logits(&xf)?)?))
    }
}

#[cfg(test)]
mod tests;
