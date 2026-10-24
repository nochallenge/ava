// TESTING / DESIGN ONLY — ava_brain::model: the three pre-registered language models.
//
// Shared by all three: vocab V, token embedding E [V, d] tied to the output (logits = x · Eᵀ via
// matmul_t), pre-norm residual blocks with rmsnorm (gain 1, eps 1e-5), a final rmsnorm before the
// logits. Weight matrices are stored [d_in, d_out] and applied as h · W.
// Init: every weight matrix N(0, 0.02) except the projections that write into the residual stream
// (attention output, MLP down, spiking-memory output): N(0, 0.02 / sqrt(2·layers)); embeddings
// N(0, 0.02); SpikeMem input Win N(0, 1/sqrt(d)); LIF leak parameter logit(0.9) in every channel.
// Weight decay only on the weight matrices (not embeddings, gains or LIF leak parameters).
//
//   T  yardstick transformer (never part of Ava): tok + learned absolute position embedding [ctx, d];
//      block: x += Attn(rmsnorm(x)); x += MLP(rmsnorm(x)).
//      Attn = causal multi-head self-attention, head size 32, q/k/v/o d→d without biases,
//      scores · 1/sqrt(32), softmax_last(causal). MLP = d→4d, gelu, 4d→d.
//   S  Ava spiking (no attention, no position embedding);
//      block: x += SpikeMem(rmsnorm(x)); x += SparseMLP(rmsnorm(x)).
//      SpikeMem: current = h·Win [d,d]; spikes = lif_scan(current [B,T,d], beta_param [d], θ = 1,
//      Surrogate{slope 4}, reset detached); out = spikes·Wout.
//      SparseMLP: u = h·W1 (d→4d); a = relu(kwta(u, k = 4d/8)); out = a·W2 (4d→d).
//   H  Ava hybrid: position embedding as in T;
//      block: x += SpikeMem(rmsnorm(x)); x += Attn(rmsnorm(x)); x += SparseMLP(rmsnorm(x)).
//
// Parameter names: tok_emb, pos_emb, b{i}.mem.{norm,win,beta,wout}, b{i}.attn.{norm,wq,wk,wv,wo},
// b{i}.mlp.{norm,w1,w2}, final.norm. Tensors are created (and the init RNG consumed) in exactly
// this order, so a seed fixes every initial value.
//
// The forward pass also measures the wall time of each op category (forward only; the backward pass
// is one ava_grad call and is profiled separately by the `profile` binary).

use ava_grad::{Graph, ParamId, Params, Rng, SpikeFn, Tensor, Var};
use std::time::Instant;

pub const HEAD_SIZE: usize = 32;
pub const NORM_EPS: f32 = 1e-5;
pub const INIT_STD: f32 = 0.02;
pub const LIF_THRESHOLD: f32 = 1.0;
pub const LIF_SLOPE: f32 = 4.0;
/// Initial leak β = σ(beta_param) = 0.9.
pub const LIF_BETA0: f32 = 0.9;
/// kWTA keeps 1/KWTA_DIV of the 4d hidden units.
pub const KWTA_DIV: usize = 8;
pub const MLP_MULT: usize = 4;

pub const TOK_EMB: &str = "tok_emb";
pub const POS_EMB: &str = "pos_emb";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Yardstick transformer.
    T,
    /// Ava spiking: SpikeMem + SparseMLP, no attention, no position embedding.
    S,
    /// Ava hybrid: SpikeMem + Attn + SparseMLP, position embedding.
    H,
}

impl Kind {
    pub fn parse(s: &str) -> Result<Kind, String> {
        match s.trim() {
            "T" | "t" => Ok(Kind::T),
            "S" | "s" => Ok(Kind::S),
            "H" | "h" => Ok(Kind::H),
            _ => Err(format!("unknown model {s:?} (expected T, S or H)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::T => "T",
            Kind::S => "S",
            Kind::H => "H",
        }
    }

    pub fn has_pos(self) -> bool {
        matches!(self, Kind::T | Kind::H)
    }

    pub fn has_attn(self) -> bool {
        matches!(self, Kind::T | Kind::H)
    }

    pub fn has_spike_mem(self) -> bool {
        matches!(self, Kind::S | Kind::H)
    }

    /// kWTA + relu MLP (S, H) instead of the dense gelu MLP (T).
    pub fn has_sparse_mlp(self) -> bool {
        matches!(self, Kind::S | Kind::H)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub kind: Kind,
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    /// Maximum sequence length (the size of the position table for T and H).
    pub ctx: usize,
}

impl Config {
    pub fn new(kind: Kind, vocab: usize, d: usize, layers: usize, ctx: usize) -> Config {
        let c = Config { kind, vocab, d, layers, ctx };
        c.validate().unwrap_or_else(|e| panic!("bad model config: {e}"));
        c
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.vocab == 0 || self.d == 0 || self.layers == 0 || self.ctx == 0 {
            return Err(format!("vocab, d, layers and ctx must all be positive: {self:?}"));
        }
        if self.kind.has_attn() && !self.d.is_multiple_of(HEAD_SIZE) {
            return Err(format!("d = {} must be a multiple of the head size {HEAD_SIZE} for model {}", self.d, self.kind.name()));
        }
        if self.kind.has_sparse_mlp() && !(MLP_MULT * self.d).is_multiple_of(KWTA_DIV) {
            return Err(format!("4d = {} must be divisible by {KWTA_DIV} for kWTA", MLP_MULT * self.d));
        }
        Ok(())
    }

    pub fn heads(&self) -> usize {
        self.d / HEAD_SIZE
    }

    pub fn hidden(&self) -> usize {
        MLP_MULT * self.d
    }

    pub fn kwta_k(&self) -> usize {
        self.hidden() / KWTA_DIV
    }

    /// Init std of projections that write into the residual stream.
    pub fn resid_std(&self) -> f32 {
        INIT_STD / ((2 * self.layers) as f32).sqrt()
    }
}

#[derive(Clone, Debug)]
struct SpikeMem {
    norm: ParamId,
    win: ParamId,
    beta: ParamId,
    wout: ParamId,
}

#[derive(Clone, Debug)]
struct Attn {
    norm: ParamId,
    wq: ParamId,
    wk: ParamId,
    wv: ParamId,
    wo: ParamId,
}

#[derive(Clone, Debug)]
struct Mlp {
    norm: ParamId,
    w1: ParamId,
    w2: ParamId,
}

#[derive(Clone, Debug)]
struct Block {
    mem: Option<SpikeMem>,
    attn: Option<Attn>,
    mlp: Mlp,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub cfg: Config,
    tok: ParamId,
    pos: Option<ParamId>,
    blocks: Vec<Block>,
    final_norm: ParamId,
}

/// Forward-pass op categories whose wall time is measured.
pub const OP_CATS: [&str; 9] = ["embed", "rmsnorm", "proj_matmul", "attn_core", "lif", "kwta", "act", "resid_add", "logits"];
const C_EMBED: usize = 0;
const C_NORM: usize = 1;
const C_PROJ: usize = 2;
const C_ATTN: usize = 3;
const C_LIF: usize = 4;
const C_KWTA: usize = 5;
const C_ACT: usize = 6;
const C_ADD: usize = 7;
const C_LOGITS: usize = 8;

#[derive(Clone, Debug, Default)]
pub struct OpTimes {
    pub secs: [f64; OP_CATS.len()],
}

impl OpTimes {
    pub fn add(&mut self, o: &OpTimes) {
        for (a, b) in self.secs.iter_mut().zip(&o.secs) {
            *a += b;
        }
    }

    pub fn total(&self) -> f64 {
        self.secs.iter().sum()
    }
}

/// What one forward pass produced: the logits [B·T, V] plus the activity tensors used for reporting.
pub struct Forward {
    pub logits: Var,
    /// LIF output of every SpikeMem, [B, T, d], one per layer (S, H).
    pub spikes: Vec<Var>,
    /// kWTA output before relu, [B·T, 4d], one per layer (S, H).
    pub kwta_pre: Vec<Var>,
    /// After relu, [B·T, 4d], one per layer (S, H).
    pub kwta_post: Vec<Var>,
    pub times: OpTimes,
}

fn timed<R>(times: &mut OpTimes, cat: usize, f: impl FnOnce() -> R) -> R {
    let t0 = Instant::now();
    let r = f();
    times.secs[cat] += t0.elapsed().as_secs_f64();
    r
}

/// logit(p) = ln(p / (1 − p)).
pub fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

impl Model {
    /// Build the model and a fresh parameter store initialised from `seed`.
    pub fn new(cfg: Config, seed: u64) -> (Model, Params) {
        cfg.validate().unwrap_or_else(|e| panic!("bad model config: {e}"));
        let mut rng = Rng::new(seed);
        let mut p = Params::new();
        let (v, d, hid) = (cfg.vocab, cfg.d, cfg.hidden());
        let rs = cfg.resid_std();
        let tok = p.add(TOK_EMB, Tensor::randn(&[v, d], INIT_STD, &mut rng), false);
        let pos = if cfg.kind.has_pos() { Some(p.add(POS_EMB, Tensor::randn(&[cfg.ctx, d], INIT_STD, &mut rng), false)) } else { None };
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let mem = if cfg.kind.has_spike_mem() {
                let norm = p.add(&format!("b{i}.mem.norm"), Tensor::full(&[d], 1.0), false);
                let win = p.add(&format!("b{i}.mem.win"), Tensor::randn(&[d, d], 1.0 / (d as f32).sqrt(), &mut rng), true);
                let beta = p.add(&format!("b{i}.mem.beta"), Tensor::full(&[d], logit(LIF_BETA0)), false);
                let wout = p.add(&format!("b{i}.mem.wout"), Tensor::randn(&[d, d], rs, &mut rng), true);
                Some(SpikeMem { norm, win, beta, wout })
            } else {
                None
            };
            let attn = if cfg.kind.has_attn() {
                let norm = p.add(&format!("b{i}.attn.norm"), Tensor::full(&[d], 1.0), false);
                let wq = p.add(&format!("b{i}.attn.wq"), Tensor::randn(&[d, d], INIT_STD, &mut rng), true);
                let wk = p.add(&format!("b{i}.attn.wk"), Tensor::randn(&[d, d], INIT_STD, &mut rng), true);
                let wv = p.add(&format!("b{i}.attn.wv"), Tensor::randn(&[d, d], INIT_STD, &mut rng), true);
                let wo = p.add(&format!("b{i}.attn.wo"), Tensor::randn(&[d, d], rs, &mut rng), true);
                Some(Attn { norm, wq, wk, wv, wo })
            } else {
                None
            };
            let norm = p.add(&format!("b{i}.mlp.norm"), Tensor::full(&[d], 1.0), false);
            let w1 = p.add(&format!("b{i}.mlp.w1"), Tensor::randn(&[d, hid], INIT_STD, &mut rng), true);
            let w2 = p.add(&format!("b{i}.mlp.w2"), Tensor::randn(&[hid, d], rs, &mut rng), true);
            blocks.push(Block { mem, attn, mlp: Mlp { norm, w1, w2 } });
        }
        let final_norm = p.add("final.norm", Tensor::full(&[d], 1.0), false);
        (Model { cfg, tok, pos, blocks, final_norm }, p)
    }

    /// (all weights, weights excluding the token and position embeddings).
    pub fn weight_counts(params: &Params) -> (usize, usize) {
        let total = params.count();
        let emb = params.count_where(|n| n == TOK_EMB || n == POS_EMB);
        (total, total - emb)
    }

    /// Forward pass over `b` sequences of `t` tokens (`ids` is [b·t], sequence-major). Returns the
    /// logits [b·t, V] and the activity tensors.
    pub fn forward(&self, g: &mut Graph, ids: &[u32], b: usize, t: usize) -> Forward {
        let cfg = &self.cfg;
        assert_eq!(ids.len(), b * t, "forward: {} ids for b={b} × t={t}", ids.len());
        assert!(t >= 1 && t <= cfg.ctx, "forward: t = {t} outside 1..={}", cfg.ctx);
        let n = b * t;
        let mut tm = OpTimes::default();
        let mut spikes = Vec::new();
        let mut kwta_pre = Vec::new();
        let mut kwta_post = Vec::new();

        let te = g.param(self.tok);
        let mut x = timed(&mut tm, C_EMBED, || g.embedding(te, ids));
        if let Some(pos) = self.pos {
            let pe = g.param(pos);
            let pos_ids: Vec<u32> = (0..n).map(|i| (i % t) as u32).collect();
            x = timed(&mut tm, C_EMBED, || {
                let xp = g.embedding(pe, &pos_ids);
                g.add(x, xp)
            });
        }

        for blk in &self.blocks {
            if let Some(m) = &blk.mem {
                let gain = g.param(m.norm);
                let h = timed(&mut tm, C_NORM, || g.rmsnorm(x, gain, NORM_EPS));
                let win = g.param(m.win);
                let cur = timed(&mut tm, C_PROJ, || g.matmul(h, win));
                let cur = timed(&mut tm, C_LIF, || g.reshape(cur, &[b, t, cfg.d]));
                let beta = g.param(m.beta);
                let s = timed(&mut tm, C_LIF, || g.lif_scan(cur, beta, LIF_THRESHOLD, SpikeFn::Surrogate { slope: LIF_SLOPE }));
                spikes.push(s);
                let s2 = timed(&mut tm, C_LIF, || g.reshape(s, &[n, cfg.d]));
                let wout = g.param(m.wout);
                let o = timed(&mut tm, C_PROJ, || g.matmul(s2, wout));
                x = timed(&mut tm, C_ADD, || g.add(x, o));
            }
            if let Some(a) = &blk.attn {
                let gain = g.param(a.norm);
                let h = timed(&mut tm, C_NORM, || g.rmsnorm(x, gain, NORM_EPS));
                let o = self.attention(g, a, h, b, t, &mut tm);
                x = timed(&mut tm, C_ADD, || g.add(x, o));
            }
            let m = &blk.mlp;
            let gain = g.param(m.norm);
            let h = timed(&mut tm, C_NORM, || g.rmsnorm(x, gain, NORM_EPS));
            let w1 = g.param(m.w1);
            let u = timed(&mut tm, C_PROJ, || g.matmul(h, w1));
            let a = if cfg.kind.has_sparse_mlp() {
                let k = cfg.kwta_k();
                let w = timed(&mut tm, C_KWTA, || g.kwta(u, k));
                let a = timed(&mut tm, C_ACT, || g.relu(w));
                kwta_pre.push(w);
                kwta_post.push(a);
                a
            } else {
                timed(&mut tm, C_ACT, || g.gelu(u))
            };
            let w2 = g.param(m.w2);
            let o = timed(&mut tm, C_PROJ, || g.matmul(a, w2));
            x = timed(&mut tm, C_ADD, || g.add(x, o));
        }

        let gf = g.param(self.final_norm);
        let x = timed(&mut tm, C_NORM, || g.rmsnorm(x, gf, NORM_EPS));
        let logits = timed(&mut tm, C_LOGITS, || g.matmul_t(x, te));
        Forward { logits, spikes, kwta_pre, kwta_post, times: tm }
    }

    /// Causal multi-head self-attention on h [b·t, d] → [b·t, d] (before the residual add).
    fn attention(&self, g: &mut Graph, a: &Attn, h: Var, b: usize, t: usize, tm: &mut OpTimes) -> Var {
        let d = self.cfg.d;
        let nh = self.cfg.heads();
        let heads_of =|g: &mut Graph, w: ParamId, tm: &mut OpTimes| {
            let wv = g.param(w);
            let y = timed(tm, C_PROJ, || g.matmul(h, wv));
            timed(tm, C_ATTN, || {
                let y = g.reshape(y, &[b, t, nh, HEAD_SIZE]);
                g.transpose(y, 1, 2) // [b, nh, t, 32]
            })
        };
        let q = heads_of(g, a.wq, tm);
        let k = heads_of(g, a.wk, tm);
        let v = heads_of(g, a.wv, tm);
        let o = timed(tm, C_ATTN, || {
            let sc = g.matmul_t(q, k); // [b, nh, t, t]
            let sc = g.scale(sc, 1.0 / (HEAD_SIZE as f32).sqrt());
            let att = g.softmax_last(sc, true);
            let o = g.matmul(att, v); // [b, nh, t, 32]
            let o = g.transpose(o, 1, 2); // [b, t, nh, 32]
            g.reshape(o, &[b * t, d])
        });
        let wo = g.param(a.wo);
        timed(tm, C_PROJ, || g.matmul(o, wo))
    }

    /// Mean cross-entropy (nats/token, IGNORE targets skipped) of one batch, plus the Forward.
    pub fn loss(&self, g: &mut Graph, ids: &[u32], targets: &[u32], b: usize, t: usize) -> (Var, Forward) {
        let f = self.forward(g, ids, b, t);
        let l = g.cross_entropy(f.logits, targets);
        (l, f)
    }
}
