// TESTING / DESIGN ONLY — ava_gpu::model: Ava-S (and the T yardstick) on candle tensors.
//
// Mirrors brain/src/model.rs (Kind::S and Kind::T) in structure, shapes, init distributions and parameter
// names (tok_emb, pos_emb, b{i}.mem.{norm,win,beta,wout}, b{i}.attn.{norm,wq,wk,wv,wo},
// b{i}.mlp.{norm,w1,w2}, final.norm); weight matrices are [d_in, d_out] applied as h · W.  Known
// differences from the reference: the init RNG (candle's, same distributions), kWTA ties (every value >=
// the k-th largest is kept, the reference keeps the lowest index), and the LIF surrogate realised as a
// fused CUDA kernel with a hand-written backward (lif.rs) instead of the CPU op.
//
// S  block: x += SpikeMem(rmsnorm(x)); x += SparseMLP(rmsnorm(x)).  No attention, no position embedding.
// T  block: x += Attn(rmsnorm(x)); x += MLP(rmsnorm(x)).  Causal multi-head attention, head size 32,
//    q/k/v/o d→d without biases, scores · 1/sqrt(32); MLP d→4d, gelu (tanh form), 4d→d; learned
//    position embedding [ctx, d].  The yardstick, never part of her.
// F  block: x += SpikeMem(rmsnorm(x)); x += Hebb(rmsnorm(x)); x += SparseMLP(rmsnorm(x)).  Hebb = fast
//    Hebbian synapses per head (head size 32): F_t = gamma_h·F_{t-1} + k_t v_t^T, read y_t = q_t·F_t / sqrt(32),
//    each head's read-out rmsnormed, then Wo.  gamma_h = 1 - 2^-(5+h): several forgetting timescales.
//    Trained in the parallel form (decay-masked q·k^T times v); generation carries F token to token.
// G  block: as F, but forgetting and imprinting are gated per token and head (neuromodulation):
//    F_t = alpha_t·F_{t-1} + beta_t·k_t v_t^T, alpha = sigmoid(h·Wa + ba + logit(gamma_h)), beta = 2·sigmoid(h·Wb + bb);
//    Wa, ba, Wb, bb start at 0, so G starts exactly as F.
// R  block: as G, plus unit-length q and k (every imprint equally strong) and an output gate: the read-out
//    times silu(h·Wg) before Wo, so the memory can stay quiet when it recalls nothing useful.
// D  block: as R, but the write is the gated delta rule: F_t = alpha_t (I - beta_t k_t k_t^T) F_{t-1} + beta_t k_t v_t^T,
//    read y_t = q_t·F_t.  Each write first erases what the memory already recalls for k_t, so it stores the
//    error (what was new), not the raw value.  Trained in chunks of 64 (`delta_rule`); generation steps it.
//
// LIF over the time axis, per channel: v_t = beta·v_{t-1} + I_t - theta·s_{t-1} (reset term detached),
// s_t = H(v_t - theta), beta = sigmoid(beta_param), theta = 1.  Backward uses the fast-sigmoid surrogate
// ds/dv = 1 / (1 + 4·|v - theta|)^2.

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

pub const VOCAB: usize = 4096;
pub const NORM_EPS: f64 = 1e-5;
pub const INIT_STD: f64 = 0.02;
pub const LIF_THRESHOLD: f64 = 1.0;
pub const LIF_SLOPE: f64 = 4.0;
pub const LIF_BETA0: f64 = 0.9;
pub const KWTA_DIV: usize = 8;
pub const MLP_MULT: usize = 4;
pub const HEAD_SIZE: usize = 32;
pub const TOK_EMB: &str = "tok_emb";
pub const POS_EMB: &str = "pos_emb";
/// Added to the scores above the diagonal before the softmax (exp underflows to exactly 0).
const MASK_NEG: f32 = -1e9;

/// S's carried LIF state for token-by-token generation (`Model::stream`, `Model::step`): per layer the
/// leak beta, the membrane v and the last spikes, each [1, d], and the number of tokens fed so far.
pub struct Stream {
    beta: Vec<Tensor>,
    v: Vec<Tensor>,
    s: Vec<Tensor>,
    /// F only: each layer's fast weights [nh, hs, hs] and the per-head forgetting rates [nh, 1, 1].
    fw: Vec<Tensor>,
    gamma: Option<Tensor>,
    /// A only: each attention block's keys and values so far, [nh, seen, hs] (None before the first token).
    kv: Vec<Option<(Tensor, Tensor)>>,
    pub seen: usize,
}

/// logit(p) = ln(p / (1 - p)).
pub fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// S = Ava (spiking memory + sparse MLP); F = S plus a Hebbian fast-weight memory in every block; G gates it;
/// R adds unit keys and an output gate; D makes the write error-correcting (delta rule); T = the transformer yardstick.
/// A = the hybrid: D's blocks, except that every 4th block counting back from the last (the last, then 4 before it,
/// ...) has causal softmax attention in place of the fast-weight memory. No position table: the spiking and
/// fast-weight layers carry order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    S,
    T,
    F,
    G,
    R,
    D,
    A,
}

impl Kind {
    pub fn parse(s: &str) -> std::result::Result<Kind, String> {
        match s {
            "S" | "s" => Ok(Kind::S),
            "T" | "t" => Ok(Kind::T),
            "F" | "f" => Ok(Kind::F),
            "G" | "g" => Ok(Kind::G),
            "R" | "r" => Ok(Kind::R),
            "D" | "d" => Ok(Kind::D),
            "A" | "a" => Ok(Kind::A),
            o => Err(format!("unknown model kind {o:?} (expected S, F, G, R, D, A or T)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::S => "S",
            Kind::T => "T",
            Kind::F => "F",
            Kind::G => "G",
            Kind::R => "R",
            Kind::D => "D",
            Kind::A => "A",
        }
    }

    /// Hebbian fast weights in every block: F (fixed forgetting) and G (gated by what she reads).
    pub fn has_hebb(self) -> bool {
        matches!(self, Kind::F | Kind::G | Kind::R | Kind::D | Kind::A)
    }

    /// Ava's own parts (LIF memory, kWTA sparse MLP, recurrent generation): S and F.
    pub fn is_ava(self) -> bool {
        self != Kind::T
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub kind: Kind,
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub ctx: usize,
}

impl Config {
    /// An S (Ava) config.
    pub fn new(vocab: usize, d: usize, layers: usize, ctx: usize) -> Config {
        Config::with_kind(Kind::S, vocab, d, layers, ctx)
    }

    pub fn with_kind(kind: Kind, vocab: usize, d: usize, layers: usize, ctx: usize) -> Config {
        let c = Config { kind, vocab, d, layers, ctx };
        c.validate().unwrap_or_else(|e| panic!("bad model config: {e}"));
        c
    }

    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.vocab == 0 || self.d == 0 || self.layers == 0 || self.ctx == 0 {
            return Err(format!("vocab, d, layers and ctx must all be positive: {self:?}"));
        }
        if self.kind.is_ava() && (MLP_MULT * self.d) % KWTA_DIV != 0 {
            return Err(format!("4d = {} must be divisible by {KWTA_DIV} for kWTA", MLP_MULT * self.d));
        }
        if self.kind != Kind::S && self.d % HEAD_SIZE != 0 {
            return Err(format!("d = {} must be a multiple of the head size {HEAD_SIZE} for model {}", self.d, self.kind.name()));
        }
        Ok(())
    }

    pub fn hidden(&self) -> usize {
        MLP_MULT * self.d
    }

    pub fn heads(&self) -> usize {
        self.d / HEAD_SIZE
    }

    pub fn kwta_k(&self) -> usize {
        self.hidden() / KWTA_DIV
    }

    /// Init std of projections that write into the residual stream.
    pub fn resid_std(&self) -> f64 {
        INIT_STD / ((2 * self.layers) as f64).sqrt()
    }

    /// A only: whether block `i` has attention (every 4th block counting back from the last) instead of the
    /// fast-weight memory.
    pub fn attn_layer(&self, i: usize) -> bool {
        self.kind == Kind::A && (self.layers - 1 - i) % 4 == 0
    }

    /// A only: how many blocks have attention.
    pub fn attn_layers(&self) -> usize {
        (0..self.layers).filter(|&i| self.attn_layer(i)).count()
    }

    /// (all weights, weights excluding the token and position embeddings).
    pub fn weight_counts(&self) -> (usize, usize) {
        let (v, d, h) = (self.vocab, self.d, self.hidden());
        let mlp = d + d * h + h * d;
        let spike = d + d * d + d + d * d;
        let mem_r = d + 5 * d * d + 2 * (d * self.heads() + self.heads());
        let (layers_total, pos) = match self.kind {
            Kind::S => (self.layers * (spike + mlp), 0),
            Kind::F => (self.layers * (spike + (d + 4 * d * d) + mlp), 0),
            Kind::G => (self.layers * (spike + (d + 4 * d * d + 2 * (d * self.heads() + self.heads())) + mlp), 0),
            Kind::R | Kind::D => (self.layers * (spike + mem_r + mlp), 0),
            Kind::A => {
                let na = self.attn_layers();
                (na * (spike + (d + 4 * d * d) + mlp) + (self.layers - na) * (spike + mem_r + mlp), 0)
            }
            Kind::T => (self.layers * (d + 4 * d * d + mlp), self.ctx * d),
        };
        let total = v * d + pos + layers_total + d;
        (total, total - v * d - pos)
    }
}

struct SpikeMem {
    norm: Tensor,
    win: Tensor,
    beta: Tensor,
    wout: Tensor,
}

struct Attn {
    norm: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
}

struct Mlp {
    norm: Tensor,
    w1: Tensor,
    w2: Tensor,
}

/// Hebbian fast weights (F, G): per head a matrix of fast synapses that every token strengthens by the
/// outer product of its key and value and that fades per token; the query reads it.
struct Hebb {
    norm: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    gates: Option<Gates>,
    /// R only: the output gate Wg [d, d] (read-out times silu(h·Wg), so the memory can stay quiet)
    /// and unit-length queries and keys (every imprint has the same strength).
    wg: Option<Tensor>,
    qk_norm: bool,
    /// D only: the gated delta rule (`delta_rule`): each write first erases what the memory recalls for k.
    delta: bool,
}

/// G's neuromodulation, per head and per token, read from the same normalised input h:
/// forget alpha = sigmoid(h·Wa + ba + logit(gamma_h)), write strength beta = 2·sigmoid(h·Wb + bb) in (0, 2).
/// Wa, ba, Wb, bb start at 0, so at init alpha = F's gamma_h and beta = 1 for every token:
/// G starts exactly as F and learns when to forget and when to imprint more or less strongly.
struct Gates {
    wa: Tensor,
    ba: Tensor,
    wb: Tensor,
    bb: Tensor,
    /// logit(gamma_h) [nh], a constant (not trained).
    off: Tensor,
}

struct Block {
    mem: Option<SpikeMem>,
    attn: Option<Attn>,
    hebb: Option<Hebb>,
    mlp: Mlp,
}

pub struct Model {
    pub cfg: Config,
    tok: Tensor,
    pos: Option<Tensor>,
    blocks: Vec<Block>,
    final_norm: Tensor,
    /// Whether forward computes the logged activity means (off on steps that are not logged: they never
    /// feed the loss, and cost a pass over every hidden layer).
    track: std::cell::Cell<bool>,
    /// v37: inference only (no backward will follow): D's fast-weight blocks use the training step's fused kernel
    /// (hebb_fwd_raw: a state saved every 16 tokens) instead of the autograd op that keeps one per token.
    infer: std::cell::Cell<bool>,
}

/// Per-layer activity of one forward pass (means over every position), as 0-d tensors on the device.
/// Empty for T.
pub struct Activity {
    pub spike_rate: Vec<Tensor>,
    pub kwta_post: Vec<Tensor>,
}

impl Activity {
    /// Host copies, "r0,r1,..." formatted.
    pub fn fmt(v: &[Tensor]) -> String {
        v.iter()
            .map(|t| t.to_dtype(DType::F32).and_then(|t| t.to_scalar::<f32>()).map(|x| format!("{x:.4}")).unwrap_or_else(|_| "?".into()))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// x / sqrt(mean(x^2) + eps) · gain over the last dim.
/// rmsnorm for the training forward: the fused kernel on CUDA (speed round 2), else the primitive chain.
fn norm(x: &Tensor, gain: &Tensor) -> Result<Tensor> {
    if crate::fast_path() && x.device().is_cuda() {
        crate::fused::rms_norm(x, gain, NORM_EPS)
    } else {
        rmsnorm(x, gain)
    }
}

pub fn rmsnorm(x: &Tensor, gain: &Tensor) -> Result<Tensor> {
    let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
    let inv = (ms + NORM_EPS)?.powf(-0.5)?;
    x.broadcast_mul(&inv)?.broadcast_mul(gain)
}

/// sigmoid(x) from primitives (keeps the backward inside candle's standard autograd).
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    (x.neg()?.exp()? + 1.0)?.recip()
}

/// LIF scan over the time axis of `cur` [B, T, d] with leak sigmoid(beta_param) [d]; returns spikes
/// [B, T, d] whose forward values are 0/1 and whose backward is the fast-sigmoid surrogate (reset detached).
/// The fused op of `lif.rs`: one kernel launch per direction.
pub fn lif_scan(cur: &Tensor, beta_param: &Tensor) -> Result<Tensor> {
    crate::lif::lif(cur, &sigmoid(beta_param)?)
}

/// The same scan unrolled into standard candle ops (straight-through estimator), ~10 kernel launches
/// per time step. The reference that tests/smoke.rs checks the fused op against.
pub fn lif_scan_unrolled(cur: &Tensor, beta_param: &Tensor) -> Result<Tensor> {
    let (_b, t, _d) = cur.dims3()?;
    let beta = sigmoid(beta_param)?; // [d]
    let mut v: Option<Tensor> = None;
    let mut s_prev: Option<Tensor> = None;
    let mut out = Vec::with_capacity(t);
    for step in 0..t {
        let i_t = cur.narrow(1, step, 1)?.squeeze(1)?; // [B, d]
        let mut vt = match &v {
            None => i_t,
            Some(vp) => vp.broadcast_mul(&beta)?.add(&i_t)?,
        };
        if let Some(sp) = &s_prev {
            vt = vt.sub(&sp.detach().affine(LIF_THRESHOLD, 0.0)?)?;
        }
        let u = vt.affine(1.0, -LIF_THRESHOLD)?;
        let sg = u.div(&u.abs()?.affine(LIF_SLOPE, 1.0)?)?;
        let hard = u.ge(0.0)?.to_dtype(DType::F32)?;
        let s = sg.add(&hard.sub(&sg)?.detach())?;
        out.push(s.clone());
        v = Some(vt);
        s_prev = Some(s);
    }
    Tensor::stack(&out, 1)
}

/// 0/1 mask (f32, no gradient) of the k largest entries of every row of `u` [N, H] (all values >= the
/// k-th largest are kept).
/// The k largest per row by radix select (lif.cu); same mask as kwta_mask_sorted.
pub fn kwta_mask(u: &Tensor, k: usize) -> Result<Tensor> {
    if k >= u.dim(D::Minus1)? {
        return u.detach().ones_like();
    }
    crate::lif::kwta_mask_fast(u, k)
}

/// The original sort-based mask (kept as the reference for tests and benches).
pub fn kwta_mask_sorted(u: &Tensor, k: usize) -> Result<Tensor> {
    let h = u.dim(D::Minus1)?;
    let u = u.detach();
    if k >= h {
        return u.ones_like();
    }
    let (sorted, _idx) = u.sort_last_dim(false)?;
    let kth = sorted.narrow(D::Minus1, k - 1, 1)?; // [N, 1]
    u.broadcast_sub(&kth)?.ge(0.0)?.to_dtype(DType::F32)
}


/// [t, t] additive causal mask: 0 on and below the diagonal, MASK_NEG above it.
pub fn causal_mask(t: usize, dev: &Device) -> Result<Tensor> {
    let m: Vec<f32> = (0..t).flat_map(|i| (0..t).map(move |j| if j > i { MASK_NEG } else { 0.0 })).collect();
    Tensor::from_vec(m, (t, t), dev)
}

/// Causal multi-head self-attention on h [b·t, d] → [b·t, d] (before the residual add).
fn attention(h: &Tensor, a: &Attn, b: usize, t: usize, mask: &Tensor) -> Result<Tensor> {
    let d = h.dim(1)?;
    let nh = d / HEAD_SIZE;
    let heads = |w: &Tensor| -> Result<Tensor> { h.matmul(w)?.reshape((b, t, nh, HEAD_SIZE))?.transpose(1, 2)?.contiguous() };
    let (q, k, v) = (heads(&a.wq)?, heads(&a.wk)?, heads(&a.wv)?);
    let scale = 1.0 / (HEAD_SIZE as f64).sqrt();
    // Speed round 4: one fused kernel (no [t, t] score tensor) on CUDA; same maths as the chain below.
    let o = if crate::fast_path() && h.device().is_cuda() && HEAD_SIZE == crate::fused::ATTN_HS {
        crate::fused::causal_attention(&q, &k, &v, scale)?
    } else {
        let sc = q.matmul(&k.t()?.contiguous()?)?.affine(scale, 0.0)?;
        candle_nn::ops::softmax(&sc.broadcast_add(mask)?, D::Minus1)?.matmul(&v)?
    };
    let o = o.transpose(1, 2)?.contiguous()?.reshape((b * t, d))?;
    o.matmul(&a.wo)
}

/// Forgetting rates of the fast weights, one per head: gamma_h = 1 - 2^-(5 + h), so the memory's
/// half-life doubles from head to head (about 22, 44, 88, ... tokens): several timescales at once.
pub fn hebb_gammas(nh: usize) -> Vec<f64> {
    (0..nh).map(|h| 1.0 - 2f64.powi(-(5 + h as i32))).collect()
}

/// [nh, t, t]: gamma_h^(i-j) where j <= i, else 0 (how much of token j's imprint is left at token i).
pub fn hebb_decay(nh: usize, t: usize, dev: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; nh * t * t];
    for (h, g) in hebb_gammas(nh).into_iter().enumerate() {
        for i in 0..t {
            for j in 0..=i {
                m[(h * t + i) * t + j] = g.powi((i - j) as i32) as f32;
            }
        }
    }
    Tensor::from_vec(m, (nh, t, t), dev)
}

/// rmsnorm over the last dim without a gain (each head's read-out is normalised on its own).
fn head_norm(y: &Tensor) -> Result<Tensor> {
    let inv = (y.sqr()?.mean_keepdim(D::Minus1)? + NORM_EPS)?.powf(-0.5)?;
    y.broadcast_mul(&inv)
}

/// x / |x| over the last dim.
fn l2_norm(x: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.powf(-0.5)?)
}

/// R: the read-out y [n, d] times silu(h·Wg); F and G: y unchanged.
fn out_gate(y: &Tensor, h: &Tensor, m: &Hebb) -> Result<Tensor> {
    match &m.wg {
        Some(wg) => y.mul(&candle_nn::ops::silu(&h.matmul(wg)?)?),
        None => Ok(y.clone()),
    }
}

/// log(sigmoid(x)) = -(relu(-x) + ln(1 + exp(-|x|))), stable for any x.
fn log_sigmoid(x: &Tensor) -> Result<Tensor> {
    let soft = (x.abs()?.neg()?.exp()? + 1.0)?.log()?;
    x.neg()?.relu()?.add(&soft)?.neg()
}

/// G's gates for rows h [n, d]: (log alpha, beta), each [n, nh].
fn hebb_gates(h: &Tensor, g: &Gates) -> Result<(Tensor, Tensor)> {
    let la = log_sigmoid(&h.matmul(&g.wa)?.broadcast_add(&g.ba)?.broadcast_add(&g.off)?)?;
    let beta = sigmoid(&h.matmul(&g.wb)?.broadcast_add(&g.bb)?)?.affine(2.0, 0.0)?;
    Ok((la, beta))
}

/// The fast-weight memory over a window, parallel form: y_i = sum_{j<=i} D_ij beta_j (q_i.k_j / sqrt(hs)) v_j,
/// which equals reading q_i from F_i = alpha_i F_{i-1} + beta_i k_i v_i^T (the recurrent form `Model::step`
/// uses). F: D_ij = gamma^(i-j) (`decay` = hebb_decay) and beta = 1. G: D_ij = prod_{l=j+1..i} alpha_l
/// = exp(c_i - c_j) with c the running sum of log alpha (`decay` = the causal mask, which zeroes j > i).
/// v37: D's fast-weight block for inference, by the training step's fused kernel (the same maths as `hebb` for D:
/// one projection h·[Wq|Wk|Wv|Wg|Wa|Wb], then unit q/k, gates, delta rule, head norm and silu gate in one kernel).
fn hebb_infer(h: &Tensor, m: &Hebb, b: usize, t: usize) -> Result<Tensor> {
    let g = m.gates.as_ref().expect("D has gates");
    let wg = m.wg.as_ref().expect("D has an output gate");
    let nh = h.dim(1)? / HEAD_SIZE;
    let wcat = Tensor::cat(&[&m.wq.detach(), &m.wk.detach(), &m.wv.detach(), &wg.detach(), &g.wa.detach(), &g.wb.detach()], 1)?;
    let cst = Tensor::cat(&[&g.ba.detach(), &g.bb.detach(), &g.off], 0)?;
    let p = h.detach().matmul(&wcat)?;
    let (o, _ck) = crate::fused::hebb_fwd_raw(&p, &cst, b, t, nh, NORM_EPS)?;
    o.matmul(&m.wo.detach())
}

fn hebb(h: &Tensor, m: &Hebb, b: usize, t: usize, decay: &Tensor) -> Result<Tensor> {
    let d = h.dim(1)?;
    let nh = d / HEAD_SIZE;
    let heads = |w: &Tensor| -> Result<Tensor> { h.matmul(w)?.reshape((b, t, nh, HEAD_SIZE))?.transpose(1, 2)?.contiguous() };
    let (mut q, mut k, v) = (heads(&m.wq)?, heads(&m.wk)?, heads(&m.wv)?);
    if m.qk_norm {
        q = l2_norm(&q)?;
        k = l2_norm(&k)?;
    }
    if m.delta {
        let g = m.gates.as_ref().expect("D has gates");
        let (la, beta) = hebb_gates(h, g)?;
        let per_head = |x: Tensor| -> Result<Tensor> { x.reshape((b, t, nh))?.transpose(1, 2)?.contiguous() }; // [b, nh, t]
        let (la, beta) = (per_head(la)?, per_head(beta)?);
        let y = if crate::delta_on() && h.device().is_cuda() && HEAD_SIZE == crate::fused::ATTN_HS {
            // Speed round 11: the same rule as one fused recurrence (F in registers), no chunk loop.
            let vg = Tensor::cat(&[&v, &la.unsqueeze(3)?, &beta.unsqueeze(3)?], 3)?;
            crate::fused::delta(&q, &k, &vg)?
        } else {
            delta_rule(&q, &k, &v, &la, &beta)?
        };
        let y = head_norm(&y)?;
        return out_gate(&y.transpose(1, 2)?.contiguous()?.reshape((b * t, d))?, h, m)?.matmul(&m.wo);
    }
    if let Some(g) = m.gates.as_ref().filter(|_| crate::gla_on() && h.device().is_cuda() && HEAD_SIZE == crate::fused::ATTN_HS) {
        // Speed round 10: the same read as below in one fused op, no [T, T] tensors (beta folded into k).
        let (la, beta) = hebb_gates(h, g)?;
        let per_head = |x: Tensor| -> Result<Tensor> { x.reshape((b, t, nh))?.transpose(1, 2)?.contiguous() }; // [b, nh, t]
        let c = per_head(la)?.cumsum(2)?;
        let kb = k.broadcast_mul(&per_head(beta)?.unsqueeze(3)?)?;
        let vc = Tensor::cat(&[&v, &c.unsqueeze(3)?], 3)?;
        let y = head_norm(&crate::fused::gla(&q, &kb, &vc, 1.0 / (HEAD_SIZE as f64).sqrt())?)?;
        return out_gate(&y.transpose(1, 2)?.contiguous()?.reshape((b * t, d))?, h, m)?.matmul(&m.wo);
    }
    let sc = q.matmul(&k.t()?.contiguous()?)?.affine(1.0 / (HEAD_SIZE as f64).sqrt(), 0.0)?;
    let w = match &m.gates {
        None => sc.broadcast_mul(decay)?,
        Some(g) => {
            let (la, beta) = hebb_gates(h, g)?;
            let per_head = |x: Tensor| -> Result<Tensor> { x.reshape((b, t, nh))?.transpose(1, 2)?.contiguous() }; // [b, nh, t]
            let c = per_head(la)?.cumsum(2)?;
            let dmat = c.unsqueeze(3)?.broadcast_sub(&c.unsqueeze(2)?)?.broadcast_add(decay)?.exp()?; // [b, nh, t, t]
            sc.mul(&dmat)?.broadcast_mul(&per_head(beta)?.unsqueeze(2)?)?
        }
    };
    let y = head_norm(&w.matmul(&v)?)?; // [b, nh, t, hs]
    out_gate(&y.transpose(1, 2)?.contiguous()?.reshape((b * t, d))?, h, m)?.matmul(&m.wo)
}

/// Mean of every element, as the mean of the row means. candle's mean_all reduces with one GPU block (1 of
/// 46 SMs): on Ava-M2's [8192, 1536] activity it took 6.6 ms; rows first spreads the work over every SM.
/// Same value up to float rounding. Used for the logged activity only (it never feeds the loss).
fn mean_rows_first(x: &Tensor) -> Result<Tensor> {
    x.mean_keepdim(D::Minus1)?.mean_all()
}

/// Chunk length for the delta rule: the largest power of two <= 64 that divides t.
fn chunk_len(t: usize) -> usize {
    let mut c = 64;
    while t % c != 0 {
        c /= 2;
    }
    c
}

/// [c, c] 0/1 masks: strictly below the diagonal, and on or below it.
fn tri_masks(c: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let strict: Vec<f32> = (0..c * c).map(|i| if i % c < i / c { 1.0 } else { 0.0 }).collect();
    let incl: Vec<f32> = (0..c * c).map(|i| if i % c <= i / c { 1.0 } else { 0.0 }).collect();
    Ok((Tensor::from_vec(strict, (c, c), dev)?, Tensor::from_vec(incl, (c, c), dev)?))
}

/// (I + A)^-1 for strictly lower-triangular A [.., c, c], by recursive block doubling: with T the inverse
/// of the diagonal s-blocks, the inverse of the diagonal 2s-blocks is T - T E T, where E holds A's
/// lower-left s-block inside each 2s-block (E T E = 0 exactly). log2(c) levels of two matmuls. Unlike
/// the Neumann series it never forms (-A)^n, whose entries can be huge and cancel when keys repeat.
fn inv_unit_lower(a: &Tensor, c: usize) -> Result<Tensor> {
    let dev = a.device();
    let level = |s: usize| -> Result<Tensor> {
        let m: Vec<f32> = (0..c * c)
            .map(|n| {
                let (i, j) = (n / c, n % c);
                if i / (2 * s) == j / (2 * s) && (i / s) % 2 == 1 && (j / s) % 2 == 0 { 1.0 } else { 0.0 }
            })
            .collect();
        a.broadcast_mul(&Tensor::from_vec(m, (c, c), dev)?)
    };
    let mut t = level(1)?.neg()?.broadcast_add(&Tensor::eye(c, DType::F32, dev)?)?;
    let mut s = 2;
    while s < c {
        t = t.sub(&t.matmul(&level(s)?)?.matmul(&t)?)?;
        s *= 2;
    }
    Ok(t)
}

/// D: the gated delta rule over a window, per head, with fast weights F [dk, dv]:
///     F_t = alpha_t (I - beta_t k_t k_t^T) F_{t-1} + beta_t k_t v_t^T,   read y_t = q_t F_t.
/// Before writing, the memory's own prediction for k_t (k_t^T F) is subtracted: it stores the error.
/// Computed in chunks of c tokens (the WY form of DeltaNet with the gate folded in as ratios <= 1).
/// Inside a chunk, Gamma_i = prod_{l<=i} alpha_l, D_ij = Gamma_i / Gamma_j, M = on/below the diagonal:
///     W = (I + A)^-1 diag(beta) K,   A_ij = beta_i k_i.k_j for j < i
///     U = (I + A o D)^-1 diag(beta) V
///     Y = diag(Gamma) (Q F - (QK^T o M) W F) + (QK^T o D o M) U
///     F <- Gamma_c (F - K^T W F) + K^T diag(Gamma_c / Gamma) U
/// q, k, v [b, nh, t, hs] (q and k unit length); log alpha and beta [b, nh, t]. Returns y [b, nh, t, hs].
/// Heads are folded into the batch (CUDA batched matmul takes at most two batch dimensions).
pub fn delta_rule(q: &Tensor, k: &Tensor, v: &Tensor, la: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let (b, nh, t, hs) = q.dims4()?;
    let dev = q.device();
    let (c, bn) = (chunk_len(t), b * nh);
    let nc = t / c;
    let ch = |x: &Tensor| x.reshape((bn, nc, c, hs));
    let (q, k, v) = (ch(q)?, ch(k)?, ch(v)?);
    let bc = beta.reshape((bn, nc, c))?.unsqueeze(3)?; // [bn, nc, c, 1]
    let cum = la.reshape((bn, nc, c))?.cumsum(2)?; // log Gamma, restarting every chunk
    let (strict, incl) = tri_masks(c, dev)?;
    // D o M: exp(cum_i - cum_j) on/below the diagonal, 0 above (exp of a large negative)
    let dmat = cum.unsqueeze(3)?.broadcast_sub(&cum.unsqueeze(2)?)?.broadcast_add(&causal_mask(c, dev)?)?.exp()?;
    let kt = k.transpose(2, 3)?.contiguous()?; // [bn, nc, hs, c]
    let a = k.matmul(&kt)?.broadcast_mul(&bc)?.broadcast_mul(&strict)?;
    let w = inv_unit_lower(&a, c)?.matmul(&k.broadcast_mul(&bc)?)?;
    let u = inv_unit_lower(&a.mul(&dmat)?, c)?.matmul(&v.broadcast_mul(&bc)?)?;
    let qk = q.matmul(&kt)?;
    let (qkm, qkd) = (qk.broadcast_mul(&incl)?, qk.mul(&dmat)?);
    let gam = cum.exp()?.unsqueeze(3)?; // [bn, nc, c, 1]
    let last = cum.narrow(2, c - 1, 1)?; // [bn, nc, 1]
    let tail = last.broadcast_sub(&cum)?.exp()?.unsqueeze(3)?; // Gamma_c / Gamma_j, [bn, nc, c, 1]
    let gam_c = last.exp()?.unsqueeze(3)?; // [bn, nc, 1, 1]
    let mut f = Tensor::zeros((bn, hs, hs), DType::F32, dev)?;
    let mut ys = Vec::with_capacity(nc);
    for r in 0..nc {
        let at = |x: &Tensor| -> Result<Tensor> { x.narrow(1, r, 1)?.squeeze(1) };
        let ktr = at(&kt)?;
        let wf = at(&w)?.matmul(&f)?;
        let y = at(&q)?.matmul(&f)?.sub(&at(&qkm)?.matmul(&wf)?)?.broadcast_mul(&at(&gam)?)?;
        ys.push(y.add(&at(&qkd)?.matmul(&at(&u)?)?)?);
        let fresh = ktr.matmul(&at(&u)?.broadcast_mul(&at(&tail)?)?)?;
        f = f.sub(&ktr.matmul(&wf)?)?.broadcast_mul(&at(&gam_c)?)?.add(&fresh)?;
    }
    Tensor::cat(&ys, 1)?.reshape((b, nh, t, hs))
}

impl Model {
    /// Build the model's parameters in `vb` (fresh init when the VarMap is empty; the saved values when
    /// the VarMap was loaded first).  For S the parameters are created in the same order as before T
    /// existed, so an S run consumes the init RNG exactly as it did.
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Model> {
        cfg.validate().map_err(candle_core::Error::msg)?;
        let (v, d, hid) = (cfg.vocab, cfg.d, cfg.hidden());
        let rs = cfg.resid_std();
        let randn = |std: f64| Init::Randn { mean: 0.0, stdev: std };
        let tok = vb.get_with_hints((v, d), TOK_EMB, randn(INIT_STD))?;
        let pos = if cfg.kind == Kind::T { Some(vb.get_with_hints((cfg.ctx, d), POS_EMB, randn(INIT_STD))?) } else { None };
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let mem = if cfg.kind.is_ava() {
                let vm = vb.pp(format!("b{i}.mem"));
                Some(SpikeMem {
                    norm: vm.get_with_hints(d, "norm", Init::Const(1.0))?,
                    win: vm.get_with_hints((d, d), "win", randn(1.0 / (d as f64).sqrt()))?,
                    beta: vm.get_with_hints(d, "beta", Init::Const(logit(LIF_BETA0)))?,
                    wout: vm.get_with_hints((d, d), "wout", randn(rs))?,
                })
            } else {
                None
            };
            let attn = if cfg.kind == Kind::T || cfg.attn_layer(i) {
                let va = vb.pp(format!("b{i}.attn"));
                Some(Attn {
                    norm: va.get_with_hints(d, "norm", Init::Const(1.0))?,
                    wq: va.get_with_hints((d, d), "wq", randn(INIT_STD))?,
                    wk: va.get_with_hints((d, d), "wk", randn(INIT_STD))?,
                    wv: va.get_with_hints((d, d), "wv", randn(INIT_STD))?,
                    wo: va.get_with_hints((d, d), "wo", randn(rs))?,
                })
            } else {
                None
            };
            let hebb = if cfg.kind.has_hebb() && !cfg.attn_layer(i) {
                let vh = vb.pp(format!("b{i}.hebb"));
                let nh = cfg.heads();
                Some(Hebb {
                    norm: vh.get_with_hints(d, "norm", Init::Const(1.0))?,
                    wq: vh.get_with_hints((d, d), "wq", randn(INIT_STD))?,
                    wk: vh.get_with_hints((d, d), "wk", randn(INIT_STD))?,
                    wv: vh.get_with_hints((d, d), "wv", randn(INIT_STD))?,
                    wo: vh.get_with_hints((d, d), "wo", randn(rs))?,
                    gates: if matches!(cfg.kind, Kind::G | Kind::R | Kind::D | Kind::A) {
                        let off: Vec<f32> = hebb_gammas(nh).into_iter().map(|g| logit(g) as f32).collect();
                        Some(Gates {
                            wa: vh.get_with_hints((d, nh), "wa", Init::Const(0.0))?,
                            ba: vh.get_with_hints(nh, "ba", Init::Const(0.0))?,
                            wb: vh.get_with_hints((d, nh), "wb", Init::Const(0.0))?,
                            bb: vh.get_with_hints(nh, "bb", Init::Const(0.0))?,
                            off: Tensor::from_vec(off, nh, vb.device())?,
                        })
                    } else {
                        None
                    },
                    wg: if matches!(cfg.kind, Kind::R | Kind::D | Kind::A) { Some(vh.get_with_hints((d, d), "wg", randn(INIT_STD))?) } else { None },
                    qk_norm: matches!(cfg.kind, Kind::R | Kind::D | Kind::A),
                    delta: matches!(cfg.kind, Kind::D | Kind::A),
                })
            } else {
                None
            };
            let vl = vb.pp(format!("b{i}.mlp"));
            let mlp = Mlp {
                norm: vl.get_with_hints(d, "norm", Init::Const(1.0))?,
                w1: vl.get_with_hints((d, hid), "w1", randn(INIT_STD))?,
                w2: vl.get_with_hints((hid, d), "w2", randn(rs))?,
            };
            blocks.push(Block { mem, attn, hebb, mlp });
        }
        let final_norm = vb.get_with_hints(d, "final.norm", Init::Const(1.0))?;
        Ok(Model { cfg, tok, pos, blocks, final_norm, track: std::cell::Cell::new(true), infer: std::cell::Cell::new(false) })
    }

    /// Compute the logged activity means in forward (default on).
    pub fn set_track_activity(&self, on: bool) {
        self.track.set(on);
    }

    /// v37: forward for inference only (default off; scoring and generation tools turn it on). D's fast-weight
    /// blocks then run the fused kernel of the training step, which needs far less memory on long inputs.
    pub fn set_inference(&self, on: bool) {
        self.infer.set(on);
    }

    /// Whether the parameter called `name` gets weight decay (the weight matrices only).
    pub fn decays(name: &str) -> bool {
        [".win", ".wout", ".w1", ".w2", ".wq", ".wk", ".wv", ".wo"].iter().any(|s| name.ends_with(s))
    }

    /// Logits [B·T, V] for `ids` [B, T] (u32), plus per-layer activity means (S only).
    pub fn forward_with_activity(&self, ids: &Tensor) -> Result<(Tensor, Activity)> {
        let (b, t) = ids.dims2()?;
        let d = self.cfg.d;
        let n = b * t;
        let mut act = Activity { spike_rate: Vec::new(), kwta_post: Vec::new() };
        let fast = crate::fast_path() && ids.device().is_cuda();
        // Speed round 5: same values; the backward scatters with one block per vocabulary row (bit-identical).
        let mut x = if fast { crate::fused::embedding(&self.tok, ids)? } else { self.tok.embedding(&ids.flatten_all()?)? }; // [N, d]
        if let Some(pos) = &self.pos {
            if t > self.cfg.ctx {
                candle_core::bail!("sequence length {t} exceeds the position table ({})", self.cfg.ctx);
            }
            x = x.reshape((b, t, d))?.broadcast_add(&pos.narrow(0, 0, t)?)?.reshape((n, d))?;
        }
        let mask = if matches!(self.cfg.kind, Kind::T | Kind::A) { Some(causal_mask(t, x.device())?) } else { None };
        let decay = match self.cfg.kind {
            Kind::F => Some(hebb_decay(self.cfg.heads(), t, x.device())?),
            Kind::G | Kind::R => Some(causal_mask(t, x.device())?),
            Kind::D | Kind::A => Some(Tensor::zeros(1, DType::F32, x.device())?), // unused: delta_rule builds its own masks
            _ => None,
        };
        for blk in &self.blocks {
            if let Some(m) = &blk.mem {
                let h = norm(&x, &m.norm)?;
                let cur = h.matmul(&m.win)?.reshape((b, t, d))?;
                let s = lif_scan(&cur, &m.beta)?;
                if self.track.get() {
                    act.spike_rate.push(mean_rows_first(&s.detach())?);
                }
                let o = s.reshape((n, d))?.matmul(&m.wout)?;
                x = x.add(&o)?;
            }
            if let (Some(hb), Some(decay)) = (&blk.hebb, &decay) {
                let h = norm(&x, &hb.norm)?;
                if self.infer.get() && hb.delta && h.device().is_cuda() && HEAD_SIZE == crate::fused::ATTN_HS {
                    x = x.add(&hebb_infer(&h, hb, b, t)?)?;
                } else {
                    x = x.add(&hebb(&h, hb, b, t, decay)?)?;
                }
            }
            if let (Some(a), Some(mask)) = (&blk.attn, &mask) {
                let h = norm(&x, &a.norm)?;
                x = x.add(&attention(&h, a, b, t, mask)?)?;
            }
            let l = &blk.mlp;
            let h = norm(&x, &l.norm)?;
            let u = h.matmul(&l.w1)?;
            let a = if self.cfg.kind.is_ava() {
                let a = if crate::fast_path() && u.device().is_cuda() {
                    crate::fused::kwta_relu(&u, self.cfg.kwta_k())?
                } else {
                    u.mul(&kwta_mask(&u, self.cfg.kwta_k())?)?.relu()?
                };
                if self.track.get() {
                    act.kwta_post.push(mean_rows_first(&a.detach().gt(0.0)?.to_dtype(DType::F32)?)?);
                }
                a
            } else if fast {
                crate::fused::gelu(&u)? // speed round 5: candle's formulas, one kernel each way
            } else {
                u.gelu()?
            };
            x = x.add(&a.matmul(&l.w2)?)?;
        }
        let x = norm(&x, &self.final_norm)?;
        let logits = x.matmul(&self.tok.t()?)?;
        Ok((logits, act))
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        Ok(self.forward_with_activity(ids)?.0)
    }

    /// A fresh carried state for `step` (every kind but T): every membrane, spike and fast weight at 0, the
    /// state at the start of every training window.
    pub fn stream(&self) -> Result<Stream> {
        if !self.cfg.kind.is_ava() {
            candle_core::bail!("stream: only Ava (every kind but T) carries a recurrent state (T needs its window)");
        }
        let dev = self.tok.device();
        let nh = self.cfg.heads();
        let gamma = if self.cfg.kind == Kind::F {
            let g: Vec<f32> = hebb_gammas(nh).into_iter().map(|g| g as f32).collect();
            Some(Tensor::from_vec(g, (nh, 1, 1), dev)?)
        } else {
            None
        };
        let mut st = Stream { beta: Vec::new(), v: Vec::new(), s: Vec::new(), fw: Vec::new(), gamma, kv: Vec::new(), seen: 0 };
        for blk in &self.blocks {
            let m = blk.mem.as_ref().expect("Ava block has SpikeMem");
            st.beta.push(sigmoid(&m.beta)?.unsqueeze(0)?); // [1, d], the value lif_scan uses
            st.v.push(Tensor::zeros((1, self.cfg.d), DType::F32, dev)?);
            st.s.push(Tensor::zeros((1, self.cfg.d), DType::F32, dev)?);
            st.kv.push(None);
            if self.cfg.kind.has_hebb() {
                // one slot per block (A's attention blocks keep an unused one, so st.fw[i] is block i's)
                st.fw.push(Tensor::zeros((nh, HEAD_SIZE, HEAD_SIZE), DType::F32, dev)?);
            }
        }
        Ok(st)
    }

    /// Feed one token and return the logits [V] for the next one. The LIF membranes carry over from the
    /// previous call, so a token costs one position of work instead of a whole window. Same arithmetic as
    /// `forward` over the full history (no window cut): the membrane update repeats the fused kernel's
    /// rounding step for step; matmuls of one row may round differently from a whole window's.
    pub fn step(&self, st: &mut Stream, tok: u32) -> Result<Tensor> {
        let dev = self.tok.device();
        let mut x = self.tok.embedding(&Tensor::new(&[tok], dev)?)?; // [1, d]
        for (i, blk) in self.blocks.iter().enumerate() {
            let m = blk.mem.as_ref().expect("S block has SpikeMem");
            let h = rmsnorm(&x, &m.norm)?;
            let cur = h.matmul(&m.win)?;
            // ((beta·v + I) − theta·s), each step rounded, as lif_fwd does.
            let v = st.v[i].mul(&st.beta[i])?.add(&cur)?.sub(&st.s[i].affine(LIF_THRESHOLD, 0.0)?)?;
            let s = v.affine(1.0, -LIF_THRESHOLD)?.ge(0.0)?.to_dtype(DType::F32)?;
            x = x.add(&s.matmul(&m.wout)?)?;
            st.v[i] = v;
            st.s[i] = s;
            if let Some(hb) = &blk.hebb {
                // F <- alpha·F + beta·k v^T (Hebbian imprint), then read q·F / sqrt(hs).
                let nh = self.cfg.heads();
                let h = rmsnorm(&x, &hb.norm)?;
                let (mut q, mut k) = (h.matmul(&hb.wq)?.reshape((nh, 1, HEAD_SIZE))?, h.matmul(&hb.wk)?.reshape((nh, 1, HEAD_SIZE))?);
                if hb.qk_norm {
                    q = l2_norm(&q)?;
                    k = l2_norm(&k)?;
                }
                let k_row = k.clone();
                let k = k.reshape((nh, HEAD_SIZE, 1))?;
                let v = h.matmul(&hb.wv)?.reshape((nh, 1, HEAD_SIZE))?;
                let imprint = k.matmul(&v)?;
                let fw = match (&hb.gates, &st.gamma) {
                    (Some(g), _) if hb.delta => {
                        // F <- alpha (F - beta k (k^T F)) + beta k v^T: store the error, not the value
                        let (la, beta) = hebb_gates(&h, g)?;
                        let (alpha, beta) = (la.exp()?.reshape((nh, 1, 1))?, beta.reshape((nh, 1, 1))?);
                        let pred = k_row.matmul(&st.fw[i])?; // [nh, 1, hs]: what the memory recalls for k
                        let corr = k.matmul(&pred)?.broadcast_mul(&beta)?;
                        st.fw[i].sub(&corr)?.broadcast_mul(&alpha)?.add(&imprint.broadcast_mul(&beta)?)?
                    }
                    (Some(g), _) => {
                        let (la, beta) = hebb_gates(&h, g)?;
                        let alpha = la.exp()?.reshape((nh, 1, 1))?;
                        st.fw[i].broadcast_mul(&alpha)?.add(&imprint.broadcast_mul(&beta.reshape((nh, 1, 1))?)?)?
                    }
                    (None, Some(gamma)) => st.fw[i].broadcast_mul(gamma)?.add(&imprint)?,
                    (None, None) => candle_core::bail!("step: fast weights without forgetting rates"),
                };
                let read = q.matmul(&fw)?;
                let y = head_norm(&if hb.delta { read } else { read.affine(1.0 / (HEAD_SIZE as f64).sqrt(), 0.0)? })?;
                x = x.add(&out_gate(&y.reshape((1, self.cfg.d))?, &h, hb)?.matmul(&hb.wo)?)?;
                st.fw[i] = fw;
            }
            if let Some(a) = &blk.attn {
                // A: causal attention over every token so far (keys and values cached), the same maths as `attention`.
                let nh = self.cfg.heads();
                let h = rmsnorm(&x, &a.norm)?;
                let head = |w: &Tensor| -> Result<Tensor> { h.matmul(w)?.reshape((nh, 1, HEAD_SIZE)) };
                let (q, k, v) = (head(&a.wq)?, head(&a.wk)?, head(&a.wv)?);
                let (kc, vc) = match st.kv[i].take() {
                    None => (k, v),
                    Some((kp, vp)) => (Tensor::cat(&[&kp, &k], 1)?, Tensor::cat(&[&vp, &v], 1)?),
                };
                let sc = q.matmul(&kc.t()?.contiguous()?)?.affine(1.0 / (HEAD_SIZE as f64).sqrt(), 0.0)?; // [nh, 1, seen]
                let o = candle_nn::ops::softmax(&sc, D::Minus1)?.matmul(&vc)?; // [nh, 1, hs]
                x = x.add(&o.reshape((1, self.cfg.d))?.matmul(&a.wo)?)?;
                st.kv[i] = Some((kc, vc));
            }
            let l = &blk.mlp;
            let h = rmsnorm(&x, &l.norm)?;
            let u = h.matmul(&l.w1)?;
            let a = u.mul(&kwta_mask(&u, self.cfg.kwta_k())?)?.relu()?;
            x = x.add(&a.matmul(&l.w2)?)?;
        }
        st.seen += 1;
        let x = rmsnorm(&x, &self.final_norm)?;
        x.matmul(&self.tok.t()?)?.squeeze(0)
    }

    /// Mean cross-entropy (f32 nats/token) of `ids` [B, T] against `targets` [B, T].
    /// Speed round 3: the S training step (and, since round 6, T via loss_and_grads_t) with a hand-written backward
    /// (no candle autograd). Returns the mean loss, the activity (when tracked) and the gradient of every parameter,
    /// or None when this path does not apply (not S or T, not CUDA, AVA_FAST=0, AVA_MANUAL_BWD=0, or candle's
    /// cross-entropy requested); then use `loss` + `backward`. Same maths as that path: every op is the fused kernel
    /// the forward uses, and its backward is the kernel the autograd path calls; candle's autograd would add a
    /// zeros + add pass for every gradient (~19 of 74 ms per step at d384 L8). Checked against autograd in
    /// tests/smoke.rs. Speed round 7: with AVA_BF16=1 every matmul here takes bf16 operands (tensor cores, f32
    /// accumulate, f32 result); norms, LIF, kWTA, attention, softmax, cross-entropy and the weights stay f32.
    pub fn loss_and_grads(&self, ids: &Tensor, targets: &Tensor) -> Result<Option<(Tensor, Activity, GradStore)>> {
        self.loss_and_grads_with(ids, targets, crate::bf16_on())
    }

    /// loss_and_grads with the matmul precision given (bf16 = true: bf16 operands) instead of read from AVA_BF16.
    pub fn loss_and_grads_with(&self, ids: &Tensor, targets: &Tensor, bf16: bool) -> Result<Option<(Tensor, Activity, GradStore)>> {
        if !matches!(self.cfg.kind, Kind::S | Kind::T | Kind::D | Kind::A)
            || !crate::fast_path()
            || std::env::var("AVA_MANUAL_BWD").is_ok_and(|v| v == "0")
            || !ids.device().is_cuda()
            || std::env::var_os("AVA_GPU_CANDLE_CE").is_some()
        {
            return Ok(None);
        }
        if matches!(self.cfg.kind, Kind::D | Kind::A) {
            // speed round 12: bf16 only; otherwise the autograd path
            return if crate::manual_da_on() { Ok(Some(self.loss_and_grads_a(ids, targets, &Mm { bf16 })?)) } else { Ok(None) };
        }
        let mm = Mm { bf16 };
        if self.cfg.kind == Kind::T {
            if mm.bf16 {
                return Ok(Some(self.loss_and_grads_t_bf16(ids, targets)?));
            }
            return Ok(Some(self.loss_and_grads_t(ids, targets, &mm)?));
        }
        use crate::fused::{kwta_relu_raw, relu_mask_bwd_raw, rms_bwd_raw};
        use crate::lif::{ce_bwd_raw, ce_fwd_raw, lif_bwd_raw, lif_fwd_raw};
        let (b, t) = ids.dims2()?;
        let (d, n, k) = (self.cfg.d, b * t, self.cfg.kwta_k());
        let track = self.track.get();
        let mut act = Activity { spike_rate: Vec::new(), kwta_post: Vec::new() };
        let tok = self.tok.detach();
        let tok_m = mm.op(&tok)?;
        let idf = ids.flatten_all()?;
        let tg = targets.flatten_all()?;
        // ---- forward, keeping what the backward needs (h1, s, h2, a also as matmul operands: bf16 under AVA_BF16)
        struct Saved {
            x_in: Tensor,
            h1: Tensor,
            cur: Tensor,
            beta: Tensor,
            s: Tensor,
            x_mid: Tensor,
            h2: Tensor,
            a_m: Tensor,
            w: [Tensor; 4], // win, wout, w1, w2 as matmul operands
        }
        let mut saved: Vec<Saved> = Vec::with_capacity(self.blocks.len());
        let mut x = tok.embedding(&idf)?; // [N, d]
        for blk in &self.blocks {
            let m = blk.mem.as_ref().expect("S block has SpikeMem");
            let l = &blk.mlp;
            let w = [mm.op(&m.win.detach())?, mm.op(&m.wout.detach())?, mm.op(&l.w1.detach())?, mm.op(&l.w2.detach())?];
            let h1 = mm.rms(&x, &m.norm.detach())?;
            let cur = mm.mm(&h1, false, &w[0], false)?.reshape((b, t, d))?;
            let beta = sigmoid(&m.beta.detach())?;
            let s3 = if mm.bf16 { crate::fused::lif_fwd_b(&cur, &beta)? } else { lif_fwd_raw(&cur, &beta)? };
            if track {
                act.spike_rate.push(mean_rows_first(&s3.to_dtype(DType::F32)?)?);
            }
            let s = s3.reshape((n, d))?;
            drop(s3);
            let x_mid = x.add(&mm.mm(&s, false, &w[1], false)?)?;
            let h2 = mm.rms(&x_mid, &l.norm.detach())?;
            let u = mm.mm(&h2, false, &w[2], false)?;
            let a_m = if mm.bf16 { crate::fused::kwta_relu_b(&u, k)? } else { kwta_relu_raw(&u, k)? };
            drop(u);
            let x_out = x_mid.add(&mm.mm(&a_m, false, &w[3], false)?)?;
            if track {
                act.kwta_post.push(mean_rows_first(&a_m.gt(0.0)?.to_dtype(DType::F32)?)?);
            }
            saved.push(Saved { x_in: x, h1, cur, beta, s, x_mid, h2, a_m, w });
            x = x_out;
        }
        let gf = self.final_norm.detach();
        let hf = mm.rms(&x, &gf)?;
        let logits = mm.mm(&hf, false, &tok_m, true)?; // [N, V]
        let ce = ce_fwd_raw(&logits, &tg)?; // [N, 2] = (loss, lse)
        let loss = ce.narrow(1, 0, 1)?.mean_all()?;
        // ---- backward
        let mut grads = Tensor::zeros((), DType::F32, ids.device())?.backward()?; // an empty-ish GradStore
        let aux = Tensor::cat(&[&Tensor::full((1.0 / n as f64) as f32, (n, 1), ids.device())?, &ce.narrow(1, 1, 1)?], 1)?;
        let dlogits = if mm.bf16 { crate::fused::ce_bwd_b(&logits, &tg, &aux)? } else { ce_bwd_raw(&logits, &tg, &aux)? };
        drop(logits);
        let dhf = mm.mm(&dlogits, false, &tok_m, false)?;
        let dtok_out = mm.mm(&dlogits, true, &hf, false)?; // [V, d]
        drop(dlogits);
        let (dx0, dgf) = rms_bwd_raw(&x, &gf, &dhf, NORM_EPS)?;
        grads.insert(&self.final_norm, dgf);
        let mut dx_m = mm.op(&dx0)?;
        let mut dx = dx0;
        for (blk, sv) in self.blocks.iter().zip(saved.iter()).rev() {
            let m = blk.mem.as_ref().expect("S block has SpikeMem");
            let l = &blk.mlp;
            // x_out = x_mid + kwta_relu(rms(x_mid) w1) w2
            let da = mm.mm(&dx_m, false, &sv.w[3], true)?;
            grads.insert(&l.w2, mm.mm(&sv.a_m, true, &dx_m, false)?);
            let du = if mm.bf16 { crate::fused::relu_mask_bwd_b(&sv.a_m, &da)? } else { relu_mask_bwd_raw(&sv.a_m, &da)? };
            drop(da);
            let dh2 = mm.mm(&du, false, &sv.w[2], true)?;
            grads.insert(&l.w1, mm.mm(&sv.h2, true, &du, false)?);
            drop(du);
            let (dxm, dg2) = rms_bwd_raw(&sv.x_mid, &l.norm.detach(), &dh2, NORM_EPS)?;
            grads.insert(&l.norm, dg2);
            (dx, dx_m) = mm.add(&dx, &dxm)?;
            // x_mid = x_in + lif(rms(x_in) win) wout
            let ds = mm.mm(&dx_m, false, &sv.w[1], true)?;
            grads.insert(&m.wout, mm.mm(&sv.s, true, &dx_m, false)?);
            let ds3 = ds.reshape((b, t, d))?;
            let (dcur, dbeta) = if mm.bf16 { crate::fused::lif_bwd_b(&sv.cur, &sv.beta, &ds3)? } else { lif_bwd_raw(&sv.cur, &sv.beta, &ds3)? };
            // beta = sigmoid(param): dparam = dbeta * beta * (1 - beta)
            grads.insert(&m.beta, dbeta.mul(&sv.beta)?.mul(&sv.beta.affine(-1.0, 1.0)?)?);
            let dcur = dcur.reshape((n, d))?;
            let dh1 = mm.mm(&dcur, false, &sv.w[0], true)?;
            grads.insert(&m.win, mm.mm(&sv.h1, true, &dcur, false)?);
            let (dxi, dg1) = rms_bwd_raw(&sv.x_in, &m.norm.detach(), &dh1, NORM_EPS)?;
            grads.insert(&m.norm, dg1);
            (dx, dx_m) = mm.add(&dx, &dxi)?;
        }
        // embedding: x0 = tok[ids], so dtok[ids] += dx (index_add, as candle's embedding backward)
        grads.insert(&self.tok, crate::fused::embed_bwd_add(&dtok_out, &idf, &dx)?);
        Ok(Some((loss, act, grads)))
    }

    /// Speed round 6: the T training step with a hand-written backward (same scheme as S's round 3 above): every
    /// forward op is the kernel the autograd path uses and every backward the kernel its autograd calls, without
    /// candle's zeros + add per gradient. Called by loss_and_grads (which also picks bf16 or f32 matmuls).
    fn loss_and_grads_t(&self, ids: &Tensor, targets: &Tensor, mm: &Mm) -> Result<(Tensor, Activity, GradStore)> {
        use crate::fused::{attn_bwd_raw, causal_attention_raw, gelu_bwd_raw, rms_bwd_raw, rms_fwd_raw, ATTN_HS};
        use crate::lif::{ce_bwd_raw, ce_fwd_raw};
        let (b, t) = ids.dims2()?;
        let (d, n) = (self.cfg.d, b * t);
        let nh = d / HEAD_SIZE;
        if HEAD_SIZE != ATTN_HS {
            candle_core::bail!("loss_and_grads_t: head size {HEAD_SIZE} != {ATTN_HS}");
        }
        let scale = 1.0 / (HEAD_SIZE as f64).sqrt();
        let dev = ids.device();
        let tok = self.tok.detach();
        let tok_m = mm.op(&tok)?;
        let pos = self.pos.as_ref().expect("T has a position table");
        if t > self.cfg.ctx {
            candle_core::bail!("sequence length {t} exceeds the position table ({})", self.cfg.ctx);
        }
        let idf = ids.flatten_all()?;
        let tg = targets.flatten_all()?;
        let heads = |x: &Tensor| -> Result<Tensor> { x.reshape((b, t, nh, HEAD_SIZE))?.transpose(1, 2)?.contiguous() };
        let unheads = |x: &Tensor| -> Result<Tensor> { x.transpose(1, 2)?.contiguous()?.reshape((n, d)) };
        struct Saved {
            x_in: Tensor,
            h1: Tensor,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            res: Tensor,
            o: Tensor,
            x_mid: Tensor,
            h2: Tensor,
            u: Tensor,
            g: Tensor,
            w: [Tensor; 6], // wq, wk, wv, wo, w1, w2 as matmul operands
        }
        let mut saved: Vec<Saved> = Vec::with_capacity(self.blocks.len());
        let mut x = tok.embedding(&idf)?.reshape((b, t, d))?.broadcast_add(&pos.detach().narrow(0, 0, t)?)?.reshape((n, d))?;
        for blk in &self.blocks {
            let a = blk.attn.as_ref().expect("T block has attention");
            let l = &blk.mlp;
            let w = [
                mm.op(&a.wq.detach())?,
                mm.op(&a.wk.detach())?,
                mm.op(&a.wv.detach())?,
                mm.op(&a.wo.detach())?,
                mm.op(&l.w1.detach())?,
                mm.op(&l.w2.detach())?,
            ];
            let h1 = mm.op(&rms_fwd_raw(&x, &a.norm.detach(), NORM_EPS)?)?;
            let (q, k, v) = (heads(&mm.mm(&h1, false, &w[0], false)?)?, heads(&mm.mm(&h1, false, &w[1], false)?)?, heads(&mm.mm(&h1, false, &w[2], false)?)?);
            let res = causal_attention_raw(&q, &k, &v, scale)?;
            let o = mm.op(&unheads(&res.narrow(3, 0, HEAD_SIZE)?)?)?;
            let x_mid = x.add(&mm.mm(&o, false, &w[3], false)?)?;
            let h2 = mm.op(&rms_fwd_raw(&x_mid, &l.norm.detach(), NORM_EPS)?)?;
            let u = mm.mm(&h2, false, &w[4], false)?;
            let g = mm.op(&crate::fused::gelu(&u)?)?;
            let x_out = x_mid.add(&mm.mm(&g, false, &w[5], false)?)?;
            saved.push(Saved { x_in: x, h1, q, k, v, res, o, x_mid, h2, u, g, w });
            x = x_out;
        }
        let gf = self.final_norm.detach();
        let hf = mm.op(&rms_fwd_raw(&x, &gf, NORM_EPS)?)?;
        let logits = mm.mm(&hf, false, &tok_m, true)?; // [N, V]
        let ce = ce_fwd_raw(&logits, &tg)?; // [N, 2] = (loss, lse)
        let loss = ce.narrow(1, 0, 1)?.mean_all()?;
        // ---- backward
        let mut grads = Tensor::zeros((), DType::F32, dev)?.backward()?;
        let aux = Tensor::cat(&[&Tensor::full((1.0 / n as f64) as f32, (n, 1), dev)?, &ce.narrow(1, 1, 1)?], 1)?;
        let dlogits = mm.op(&ce_bwd_raw(&logits, &tg, &aux)?)?;
        drop(logits);
        let dhf = mm.mm(&dlogits, false, &tok_m, false)?;
        let dtok_out = mm.mm(&dlogits, true, &hf, false)?; // [V, d]
        drop(dlogits);
        let (dx0, dgf) = rms_bwd_raw(&x, &gf, &dhf, NORM_EPS)?;
        grads.insert(&self.final_norm, dgf);
        let mut dx_m = mm.op(&dx0)?;
        let mut dx = dx0;
        for (blk, sv) in self.blocks.iter().zip(saved.iter()).rev() {
            let a = blk.attn.as_ref().expect("T block has attention");
            let l = &blk.mlp;
            // x_out = x_mid + gelu(rms(x_mid) w1) w2
            let dg = mm.mm(&dx_m, false, &sv.w[5], true)?;
            grads.insert(&l.w2, mm.mm(&sv.g, true, &dx_m, false)?);
            let du = mm.op(&gelu_bwd_raw(&sv.u, &dg)?)?;
            drop(dg);
            let dh2 = mm.mm(&du, false, &sv.w[4], true)?;
            grads.insert(&l.w1, mm.mm(&sv.h2, true, &du, false)?);
            drop(du);
            let (dxm, dg2) = rms_bwd_raw(&sv.x_mid, &l.norm.detach(), &dh2, NORM_EPS)?;
            grads.insert(&l.norm, dg2);
            (dx, dx_m) = mm.add(&dx, &dxm)?;
            // x_mid = x_in + attn(q, k, v) wo, q/k/v = rms(x_in) wq/wk/wv
            let dout = mm.mm(&dx_m, false, &sv.w[3], true)?;
            grads.insert(&a.wo, mm.mm(&sv.o, true, &dx_m, false)?);
            let (dq, dk, dv) = attn_bwd_raw(&sv.q, &sv.k, &sv.v, &sv.res, &heads(&dout)?, scale)?;
            let (dq, dk, dv) = (mm.op(&unheads(&dq)?)?, mm.op(&unheads(&dk)?)?, mm.op(&unheads(&dv)?)?);
            grads.insert(&a.wq, mm.mm(&sv.h1, true, &dq, false)?);
            grads.insert(&a.wk, mm.mm(&sv.h1, true, &dk, false)?);
            grads.insert(&a.wv, mm.mm(&sv.h1, true, &dv, false)?);
            let dh1 = mm.mm(&dq, false, &sv.w[0], true)?.add(&mm.mm(&dk, false, &sv.w[1], true)?)?.add(&mm.mm(&dv, false, &sv.w[2], true)?)?;
            let (dxi, dg1) = rms_bwd_raw(&sv.x_in, &a.norm.detach(), &dh1, NORM_EPS)?;
            grads.insert(&a.norm, dg1);
            (dx, dx_m) = mm.add(&dx, &dxi)?;
        }
        // x0 = tok[ids] + pos[0..t] (broadcast over the batch)
        let dpos = dx.reshape((b, t, d))?.sum(0)?;
        let ctx = pos.dim(0)?;
        let dpos = if t < ctx { Tensor::cat(&[&dpos, &Tensor::zeros((ctx - t, d), DType::F32, dev)?], 0)? } else { dpos };
        grads.insert(pos, dpos);
        grads.insert(&self.tok, crate::fused::embed_bwd_add(&dtok_out, &idf, &dx)?);
        Ok((loss, Activity { spike_rate: Vec::new(), kwta_post: Vec::new() }, grads))
    }

    /// The T step under AVA_BF16=1 (speed round 7b): one fused q/k/v projection, attention read in place from it
    /// (no head transposes), and kernels that write their matmul operands as bf16 directly (no separate casts).
    fn loss_and_grads_t_bf16(&self, ids: &Tensor, targets: &Tensor) -> Result<(Tensor, Activity, GradStore)> {
        use crate::fused::{add_cast_bf16, attn_bwd_s, attn_fwd_s, ce_bwd_b, gelu_b, gelu_bwd_b, mm_bf16, rms_bwd_raw, rms_fwd_b, to_bf16, ATTN_HS};
        use crate::lif::ce_fwd_raw;
        let (b, t) = ids.dims2()?;
        let (d, n) = (self.cfg.d, b * t);
        if HEAD_SIZE != ATTN_HS {
            candle_core::bail!("loss_and_grads_t_bf16: head size {HEAD_SIZE} != {ATTN_HS}");
        }
        let scale = 1.0 / (HEAD_SIZE as f64).sqrt();
        let tc = crate::attn_tc_on(); // speed round 9: attention on the tensor cores (qkv and dout as bf16)
        let dev = ids.device();
        let tok = self.tok.detach();
        let tok_b = to_bf16(&tok)?;
        let pos = self.pos.as_ref().expect("T has a position table");
        if t > self.cfg.ctx {
            candle_core::bail!("sequence length {t} exceeds the position table ({})", self.cfg.ctx);
        }
        let idf = ids.flatten_all()?;
        let tg = targets.flatten_all()?;
        struct Saved {
            x_in: Tensor,
            h1: Tensor,  // bf16 [N, D]
            qkv: Tensor, // [N, 3D]: f32, or bf16 under the tensor-core attention
            o: Tensor,   // [N, D]: f32, or the bf16 output under the tensor-core attention
            ob: Tensor,  // bf16 [N, D]
            lse: Tensor, // [B*H, T]
            x_mid: Tensor,
            h2: Tensor, // bf16
            u: Tensor,  // f32
            g: Tensor,  // bf16
            wqkv: Tensor,
            wo: Tensor,
            w1: Tensor,
            w2: Tensor,
        }
        let mut saved: Vec<Saved> = Vec::with_capacity(self.blocks.len());
        let mut x = tok.embedding(&idf)?.reshape((b, t, d))?.broadcast_add(&pos.detach().narrow(0, 0, t)?)?.reshape((n, d))?;
        for blk in &self.blocks {
            let a = blk.attn.as_ref().expect("T block has attention");
            let l = &blk.mlp;
            let wqkv = to_bf16(&Tensor::cat(&[&a.wq.detach(), &a.wk.detach(), &a.wv.detach()], 1)?)?;
            let (wo, w1, w2) = (to_bf16(&a.wo.detach())?, to_bf16(&l.w1.detach())?, to_bf16(&l.w2.detach())?);
            let h1 = rms_fwd_b(&x, &a.norm.detach(), NORM_EPS)?;
            let (qkv, o, ob, lse) = if tc {
                let qkv = crate::fused::mm_bf16_b(&h1, false, &wqkv, false)?;
                let (ob, lse) = crate::fused::attn_fwd_tc(&qkv, b, t, scale)?;
                (qkv, ob.clone(), ob, lse)
            } else {
                let qkv = mm_bf16(&h1, false, &wqkv, false)?;
                let (o, ob, lse) = attn_fwd_s(&qkv, b, t, scale)?;
                (qkv, o, ob, lse)
            };
            let x_mid = x.add(&mm_bf16(&ob, false, &wo, false)?)?;
            let h2 = rms_fwd_b(&x_mid, &l.norm.detach(), NORM_EPS)?;
            let u = mm_bf16(&h2, false, &w1, false)?;
            let g = gelu_b(&u)?;
            let x_out = x_mid.add(&mm_bf16(&g, false, &w2, false)?)?;
            saved.push(Saved { x_in: x, h1, qkv, o, ob, lse, x_mid, h2, u, g, wqkv, wo, w1, w2 });
            x = x_out;
        }
        let gf = self.final_norm.detach();
        let hf = rms_fwd_b(&x, &gf, NORM_EPS)?;
        let logits = mm_bf16(&hf, false, &tok_b, true)?; // [N, V]
        let ce = ce_fwd_raw(&logits, &tg)?; // [N, 2] = (loss, lse)
        let loss = ce.narrow(1, 0, 1)?.mean_all()?;
        // ---- backward
        let mut grads = Tensor::zeros((), DType::F32, dev)?.backward()?;
        let aux = Tensor::cat(&[&Tensor::full((1.0 / n as f64) as f32, (n, 1), dev)?, &ce.narrow(1, 1, 1)?], 1)?;
        let dlogits = ce_bwd_b(&logits, &tg, &aux)?;
        drop(logits);
        let dhf = mm_bf16(&dlogits, false, &tok_b, false)?;
        let dtok_out = mm_bf16(&dlogits, true, &hf, false)?; // [V, d]
        drop(dlogits);
        let (dx0, dgf) = rms_bwd_raw(&x, &gf, &dhf, NORM_EPS)?;
        grads.insert(&self.final_norm, dgf);
        let mut dx_m = to_bf16(&dx0)?;
        let mut dx = dx0;
        for (blk, sv) in self.blocks.iter().zip(saved.iter()).rev() {
            let a = blk.attn.as_ref().expect("T block has attention");
            let l = &blk.mlp;
            let dg = mm_bf16(&dx_m, false, &sv.w2, true)?;
            grads.insert(&l.w2, mm_bf16(&sv.g, true, &dx_m, false)?);
            let du = gelu_bwd_b(&sv.u, &dg)?;
            drop(dg);
            let dh2 = mm_bf16(&du, false, &sv.w1, true)?;
            grads.insert(&l.w1, mm_bf16(&sv.h2, true, &du, false)?);
            drop(du);
            let (dxm, dg2) = rms_bwd_raw(&sv.x_mid, &l.norm.detach(), &dh2, NORM_EPS)?;
            grads.insert(&l.norm, dg2);
            (dx, dx_m) = add_cast_bf16(&dx, &dxm)?;
            grads.insert(&a.wo, mm_bf16(&sv.ob, true, &dx_m, false)?);
            let dqkv = if tc {
                let dout = crate::fused::mm_bf16_b(&dx_m, false, &sv.wo, true)?;
                crate::fused::attn_bwd_tc(&sv.qkv, &sv.ob, &sv.lse, &dout, b, t, scale)?
            } else {
                let dout = mm_bf16(&dx_m, false, &sv.wo, true)?;
                attn_bwd_s(&sv.qkv, &sv.o, &sv.lse, &dout, b, t, scale)?
            }; // bf16 [N, 3D]
            let dw = mm_bf16(&sv.h1, true, &dqkv, false)?; // [D, 3D]
            grads.insert(&a.wq, dw.narrow(1, 0, d)?.contiguous()?);
            grads.insert(&a.wk, dw.narrow(1, d, d)?.contiguous()?);
            grads.insert(&a.wv, dw.narrow(1, 2 * d, d)?.contiguous()?);
            let dh1 = mm_bf16(&dqkv, false, &sv.wqkv, true)?;
            let (dxi, dg1) = rms_bwd_raw(&sv.x_in, &a.norm.detach(), &dh1, NORM_EPS)?;
            grads.insert(&a.norm, dg1);
            (dx, dx_m) = add_cast_bf16(&dx, &dxi)?;
        }
        let dpos = dx.reshape((b, t, d))?.sum(0)?;
        let ctx = pos.dim(0)?;
        let dpos = if t < ctx { Tensor::cat(&[&dpos, &Tensor::zeros((ctx - t, d), DType::F32, dev)?], 0)? } else { dpos };
        grads.insert(pos, dpos);
        grads.insert(&self.tok, crate::fused::embed_bwd_add(&dtok_out, &idf, &dx)?);
        Ok((loss, Activity { spike_rate: Vec::new(), kwta_post: Vec::new() }, grads))
    }

    /// Speed round 12: the D and A training step with a hand-written backward (the scheme of S's step; bf16 matmul
    /// operands under `mm.bf16`). D's fast-weight block = one projection h·[Wq|Wk|Wv|Wg|Wa|Wb], one kernel
    /// (fused::hebb_fwd_raw: unit q/k, gates, delta rule, head norm, output gate) and Wo; its backward one kernel
    /// (hebb_bwd_raw) and the matmul gradients. A's attention blocks are T's fused attention (no position table).
    /// Spiking memory, kWTA MLP, cross-entropy and the embedding as S. The f32 form is there for the exact test.
    pub fn loss_and_grads_a(&self, ids: &Tensor, targets: &Tensor, mm: &Mm) -> Result<(Tensor, Activity, GradStore)> {
        self.loss_and_grads_a_w(ids, targets, None, mm)
    }

    /// v37 (teaching her to answer): the D/A step with a weight per target, `wt` [B·T] f32 (1 = scored, 0 = not:
    /// question tokens and padding). The loss is the weighted mean, sum(w·loss) / sum(w), and each row's gradient
    /// scale is w / sum(w). `wt = None` is the plain mean over every target, with the same ops as before.
    pub fn loss_and_grads_a_w(&self, ids: &Tensor, targets: &Tensor, wt: Option<&Tensor>, mm: &Mm) -> Result<(Tensor, Activity, GradStore)> {
        use crate::fused::{
            attn_bwd_s, attn_fwd_s, hebb_bwd_raw, hebb_fwd_raw, kwta_relu_b, kwta_relu_raw, lif_bwd_b, lif_fwd_b, relu_mask_bwd_b, relu_mask_bwd_raw,
            rms_bwd_raw, ATTN_HS,
        };
        use crate::lif::{ce_bwd_raw, ce_fwd_raw, lif_bwd_raw, lif_fwd_raw};
        let (b, t) = ids.dims2()?;
        let (d, n, k, nh) = (self.cfg.d, b * t, self.cfg.kwta_k(), self.cfg.heads());
        if HEAD_SIZE != ATTN_HS {
            candle_core::bail!("loss_and_grads_a: head size {HEAD_SIZE} != {ATTN_HS}");
        }
        let scale = 1.0 / (HEAD_SIZE as f64).sqrt();
        let bf = mm.bf16;
        let tc = bf && crate::attn_tc_on();
        let dev = ids.device();
        let track = self.track.get();
        let mut act = Activity { spike_rate: Vec::new(), kwta_post: Vec::new() };
        let tok = self.tok.detach();
        let tok_b = mm.op(&tok)?;
        let idf = ids.flatten_all()?;
        let tg = targets.flatten_all()?;
        enum Mid {
            // h: the normed input (an operand), p: the projection (f32), ck: saved states, o: the output (an operand)
            Hebb { h: Tensor, p: Tensor, cst: Tensor, ck: Tensor, o: Tensor, wcat: Tensor, wo: Tensor },
            // o: attention output f32, ob: the same as an operand
            Attn { h: Tensor, qkv: Tensor, o: Tensor, ob: Tensor, lse: Tensor, wqkv: Tensor, wo: Tensor },
        }
        struct Saved {
            x_in: Tensor,
            h1: Tensor,
            cur: Tensor,
            beta: Tensor,
            s: Tensor,
            x_m1: Tensor,
            mid: Mid,
            x_m2: Tensor,
            h2: Tensor,
            a_m: Tensor,
            w: [Tensor; 4], // win, wout, w1, w2 as matmul operands
        }
        let mut saved: Vec<Saved> = Vec::with_capacity(self.blocks.len());
        let mut x = tok.embedding(&idf)?; // [N, d]
        for blk in &self.blocks {
            let m = blk.mem.as_ref().expect("D/A block has SpikeMem");
            let l = &blk.mlp;
            let w = [mm.op(&m.win.detach())?, mm.op(&m.wout.detach())?, mm.op(&l.w1.detach())?, mm.op(&l.w2.detach())?];
            let h1 = mm.rms(&x, &m.norm.detach())?;
            let cur = mm.mm(&h1, false, &w[0], false)?.reshape((b, t, d))?;
            let beta = sigmoid(&m.beta.detach())?;
            let s3 = if bf { lif_fwd_b(&cur, &beta)? } else { lif_fwd_raw(&cur, &beta)? };
            if track {
                act.spike_rate.push(mean_rows_first(&s3.to_dtype(DType::F32)?)?);
            }
            let s = s3.reshape((n, d))?;
            drop(s3);
            let x_m1 = x.add(&mm.mm(&s, false, &w[1], false)?)?;
            let (mid, x_m2) = if let Some(hb) = &blk.hebb {
                let g = hb.gates.as_ref().expect("D has gates");
                let wg = hb.wg.as_ref().expect("D has an output gate");
                let wcat = mm.op(&Tensor::cat(&[&hb.wq.detach(), &hb.wk.detach(), &hb.wv.detach(), &wg.detach(), &g.wa.detach(), &g.wb.detach()], 1)?)?;
                let cst = Tensor::cat(&[&g.ba.detach(), &g.bb.detach(), &g.off], 0)?;
                let h = mm.rms(&x_m1, &hb.norm.detach())?;
                let p = mm.mm(&h, false, &wcat, false)?; // f32 [N, 4d + 2H]
                let (o, ck) = hebb_fwd_raw(&p, &cst, b, t, nh, NORM_EPS)?;
                let o = mm.op(&o)?;
                let wo = mm.op(&hb.wo.detach())?;
                let x_m2 = x_m1.add(&mm.mm(&o, false, &wo, false)?)?;
                (Mid::Hebb { h, p, cst, ck, o, wcat, wo }, x_m2)
            } else {
                let a = blk.attn.as_ref().expect("A block has attention or fast weights");
                let wqkv = mm.op(&Tensor::cat(&[&a.wq.detach(), &a.wk.detach(), &a.wv.detach()], 1)?)?;
                let wo = mm.op(&a.wo.detach())?;
                let h = mm.rms(&x_m1, &a.norm.detach())?;
                let (qkv, o, ob, lse) = if tc {
                    let qkv = crate::fused::mm_bf16_b(&h, false, &wqkv, false)?;
                    let (ob, lse) = crate::fused::attn_fwd_tc(&qkv, b, t, scale)?;
                    (qkv, ob.clone(), ob, lse)
                } else {
                    let qkv = mm.mm(&h, false, &wqkv, false)?;
                    let (o, ob, lse) = attn_fwd_s(&qkv, b, t, scale)?;
                    let ob = if bf { ob } else { o.clone() };
                    (qkv, o, ob, lse)
                };
                let x_m2 = x_m1.add(&mm.mm(&ob, false, &wo, false)?)?;
                (Mid::Attn { h, qkv, o, ob, lse, wqkv, wo }, x_m2)
            };
            let h2 = mm.rms(&x_m2, &l.norm.detach())?;
            let u = mm.mm(&h2, false, &w[2], false)?;
            let a_m = if bf { kwta_relu_b(&u, k)? } else { kwta_relu_raw(&u, k)? };
            drop(u);
            let x_out = x_m2.add(&mm.mm(&a_m, false, &w[3], false)?)?;
            if track {
                act.kwta_post.push(mean_rows_first(&a_m.gt(0.0)?.to_dtype(DType::F32)?)?);
            }
            saved.push(Saved { x_in: x, h1, cur, beta, s, x_m1, mid, x_m2, h2, a_m, w });
            x = x_out;
        }
        let gf = self.final_norm.detach();
        let hf = mm.rms(&x, &gf)?;
        let logits = mm.mm(&hf, false, &tok_b, true)?; // [N, V]
        let ce = ce_fwd_raw(&logits, &tg)?; // [N, 2] = (loss, lse)
        let (loss, row_scale) = match wt {
            None => (ce.narrow(1, 0, 1)?.mean_all()?, Tensor::full((1.0 / n as f64) as f32, (n, 1), dev)?),
            Some(w) => {
                let w = w.reshape((n, 1))?;
                let ws = w.broadcast_div(&w.sum_all()?)?; // [N, 1] = w / sum(w)
                (ce.narrow(1, 0, 1)?.mul(&ws)?.sum_all()?, ws)
            }
        };
        // ---- backward
        let mut grads = Tensor::zeros((), DType::F32, dev)?.backward()?;
        let aux = Tensor::cat(&[&row_scale, &ce.narrow(1, 1, 1)?], 1)?;
        let dlogits = if bf { crate::fused::ce_bwd_b(&logits, &tg, &aux)? } else { ce_bwd_raw(&logits, &tg, &aux)? };
        drop(logits);
        let dhf = mm.mm(&dlogits, false, &tok_b, false)?;
        let dtok_out = mm.mm(&dlogits, true, &hf, false)?; // [V, d]
        drop(dlogits);
        let (dx0, dgf) = rms_bwd_raw(&x, &gf, &dhf, NORM_EPS)?;
        grads.insert(&self.final_norm, dgf);
        let mut dx_m = mm.op(&dx0)?;
        let mut dx = dx0;
        for (blk, sv) in self.blocks.iter().zip(saved.iter()).rev() {
            let m = blk.mem.as_ref().expect("D/A block has SpikeMem");
            let l = &blk.mlp;
            // x_out = x_m2 + kwta_relu(rms(x_m2) w1) w2
            let da = mm.mm(&dx_m, false, &sv.w[3], true)?;
            grads.insert(&l.w2, mm.mm(&sv.a_m, true, &dx_m, false)?);
            let du = if bf { relu_mask_bwd_b(&sv.a_m, &da)? } else { relu_mask_bwd_raw(&sv.a_m, &da)? };
            drop(da);
            let dh2 = mm.mm(&du, false, &sv.w[2], true)?;
            grads.insert(&l.w1, mm.mm(&sv.h2, true, &du, false)?);
            drop(du);
            let (dxm, dg2) = rms_bwd_raw(&sv.x_m2, &l.norm.detach(), &dh2, NORM_EPS)?;
            grads.insert(&l.norm, dg2);
            (dx, dx_m) = mm.add(&dx, &dxm)?;
            // x_m2 = x_m1 + (fast weights or attention)(rms(x_m1))
            match &sv.mid {
                Mid::Hebb { h, p, cst, ck, o, wcat, wo } => {
                    let hb = blk.hebb.as_ref().expect("fast-weight block");
                    let g = hb.gates.as_ref().expect("D has gates");
                    let wg = hb.wg.as_ref().expect("D has an output gate");
                    grads.insert(&hb.wo, mm.mm(o, true, &dx_m, false)?);
                    let d_o = mm.mm(&dx_m, false, wo, true)?; // f32 [N, d]
                    let dp = hebb_bwd_raw(p, cst, ck, &d_o, b, t, nh, NORM_EPS)?; // f32 [N, 4d + 2H]
                    drop(d_o);
                    let dp_b = mm.op(&dp)?;
                    let dw = mm.mm(h, true, &dp_b, false)?; // [d, 4d + 2H]
                    grads.insert(&hb.wq, dw.narrow(1, 0, d)?.contiguous()?);
                    grads.insert(&hb.wk, dw.narrow(1, d, d)?.contiguous()?);
                    grads.insert(&hb.wv, dw.narrow(1, 2 * d, d)?.contiguous()?);
                    grads.insert(wg, dw.narrow(1, 3 * d, d)?.contiguous()?);
                    grads.insert(&g.wa, dw.narrow(1, 4 * d, nh)?.contiguous()?);
                    grads.insert(&g.wb, dw.narrow(1, 4 * d + nh, nh)?.contiguous()?);
                    grads.insert(&g.ba, dp.narrow(1, 4 * d, nh)?.sum(0)?);
                    grads.insert(&g.bb, dp.narrow(1, 4 * d + nh, nh)?.sum(0)?);
                    let dh = mm.mm(&dp_b, false, wcat, true)?;
                    let (dxi, dgn) = rms_bwd_raw(&sv.x_m1, &hb.norm.detach(), &dh, NORM_EPS)?;
                    grads.insert(&hb.norm, dgn);
                    (dx, dx_m) = mm.add(&dx, &dxi)?;
                }
                Mid::Attn { h, qkv, o, ob, lse, wqkv, wo } => {
                    let a = blk.attn.as_ref().expect("attention block");
                    grads.insert(&a.wo, mm.mm(ob, true, &dx_m, false)?);
                    let dqkv = if tc {
                        let dout = crate::fused::mm_bf16_b(&dx_m, false, wo, true)?;
                        crate::fused::attn_bwd_tc(qkv, ob, lse, &dout, b, t, scale)?
                    } else {
                        let dout = mm.mm(&dx_m, false, wo, true)?;
                        let dqkv = attn_bwd_s(qkv, o, lse, &dout, b, t, scale)?; // bf16
                        if bf { dqkv } else { dqkv.to_dtype(DType::F32)? }
                    };
                    let dw = mm.mm(h, true, &dqkv, false)?; // [d, 3d]
                    grads.insert(&a.wq, dw.narrow(1, 0, d)?.contiguous()?);
                    grads.insert(&a.wk, dw.narrow(1, d, d)?.contiguous()?);
                    grads.insert(&a.wv, dw.narrow(1, 2 * d, d)?.contiguous()?);
                    let dh = mm.mm(&dqkv, false, wqkv, true)?;
                    let (dxi, dgn) = rms_bwd_raw(&sv.x_m1, &a.norm.detach(), &dh, NORM_EPS)?;
                    grads.insert(&a.norm, dgn);
                    (dx, dx_m) = mm.add(&dx, &dxi)?;
                }
            }
            // x_m1 = x_in + lif(rms(x_in) win) wout
            let ds = mm.mm(&dx_m, false, &sv.w[1], true)?;
            grads.insert(&m.wout, mm.mm(&sv.s, true, &dx_m, false)?);
            let ds3 = ds.reshape((b, t, d))?;
            let (dcur, dbeta) = if bf { lif_bwd_b(&sv.cur, &sv.beta, &ds3)? } else { lif_bwd_raw(&sv.cur, &sv.beta, &ds3)? };
            // beta = sigmoid(param): dparam = dbeta * beta * (1 - beta)
            grads.insert(&m.beta, dbeta.mul(&sv.beta)?.mul(&sv.beta.affine(-1.0, 1.0)?)?);
            let dcur = dcur.reshape((n, d))?;
            let dh1 = mm.mm(&dcur, false, &sv.w[0], true)?;
            grads.insert(&m.win, mm.mm(&sv.h1, true, &dcur, false)?);
            let (dxi, dg1) = rms_bwd_raw(&sv.x_in, &m.norm.detach(), &dh1, NORM_EPS)?;
            grads.insert(&m.norm, dg1);
            (dx, dx_m) = mm.add(&dx, &dxi)?;
        }
        // embedding: x0 = tok[ids], so dtok[ids] += dx
        grads.insert(&self.tok, crate::fused::embed_bwd_add(&dtok_out, &idf, &dx)?);
        Ok((loss, act, grads))
    }

    pub fn loss(&self, ids: &Tensor, targets: &Tensor) -> Result<(Tensor, Activity)> {
        let (logits, act) = self.forward_with_activity(ids)?;
        let tg = targets.flatten_all()?;
        // Fused kernel by default; AVA_GPU_CANDLE_CE=1 restores candle's chain (the v34 scaling test used it).
        let loss = if std::env::var_os("AVA_GPU_CANDLE_CE").is_some() {
            candle_nn::loss::cross_entropy(&logits, &tg)?
        } else {
            crate::lif::cross_entropy_fused(&logits, &tg)?
        };
        Ok((loss, act))
    }
}

impl Module for Model {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        Model::forward(self, ids)
    }
}

/// Matmul operands of the hand-written training steps: f32 (TF32 when AVA_TF32=1) or, under AVA_BF16=1, bf16
/// (speed round 7: tensor cores, f32 accumulate, f32 result via fused::mm_bf16).
pub struct Mm {
    pub bf16: bool,
}

impl Mm {
    /// `x` as a matmul operand: its bf16 copy under bf16, else `x` itself.
    fn op(&self, x: &Tensor) -> Result<Tensor> {
        if self.bf16 {
            crate::fused::to_bf16(x)
        } else {
            Ok(x.clone())
        }
    }

    /// op(a) · op(b) (op = transpose when the flag is set), f32 result.
    fn mm(&self, a: &Tensor, ta: bool, b: &Tensor, tb: bool) -> Result<Tensor> {
        if self.bf16 {
            return crate::fused::mm_bf16(a, ta, b, tb);
        }
        let a = if ta { a.t()? } else { a.clone() };
        let b = if tb { b.t()? } else { b.clone() };
        a.matmul(&b)
    }

    /// rmsnorm forward, as a matmul operand.
    fn rms(&self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        if self.bf16 {
            crate::fused::rms_fwd_b(x, g, NORM_EPS)
        } else {
            crate::fused::rms_fwd_raw(x, g, NORM_EPS)
        }
    }

    /// (a + b, the same as a matmul operand).
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
        if self.bf16 {
            crate::fused::add_cast_bf16(a, b)
        } else {
            let s = a.add(b)?;
            Ok((s.clone(), s))
        }
    }
}
