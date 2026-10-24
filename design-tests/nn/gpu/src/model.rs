// TESTING / DESIGN ONLY — ava_gpu::model: Ava-S on candle tensors.
//
// Mirrors brain/src/model.rs (Kind::S) in structure, shapes, init distributions and parameter names
// (tok_emb, b{i}.mem.{norm,win,beta,wout}, b{i}.mlp.{norm,w1,w2}, final.norm); weight matrices are
// [d_in, d_out] applied as h · W.  Known differences from the reference: the init RNG (candle's, same
// distributions), kWTA ties (every value >= the k-th largest is kept, the reference keeps the lowest
// index), and the LIF surrogate realised as a straight-through estimator instead of a custom backward.
//
// LIF over the time axis, per channel: v_t = beta·v_{t-1} + I_t - theta·s_{t-1} (reset term detached),
// s_t = H(v_t - theta), beta = sigmoid(beta_param), theta = 1.  Backward uses the fast-sigmoid surrogate
// ds/dv = 1 / (1 + 4·|v - theta|)^2, from sg(u) = u / (1 + 4|u|) and s = sg + (H(u) - sg).detach().

use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

pub const VOCAB: usize = 4096;
pub const NORM_EPS: f64 = 1e-5;
pub const INIT_STD: f64 = 0.02;
pub const LIF_THRESHOLD: f64 = 1.0;
pub const LIF_SLOPE: f64 = 4.0;
pub const LIF_BETA0: f64 = 0.9;
pub const KWTA_DIV: usize = 8;
pub const MLP_MULT: usize = 4;
pub const TOK_EMB: &str = "tok_emb";

/// logit(p) = ln(p / (1 - p)).
pub fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub ctx: usize,
}

impl Config {
    pub fn new(vocab: usize, d: usize, layers: usize, ctx: usize) -> Config {
        let c = Config { vocab, d, layers, ctx };
        c.validate().unwrap_or_else(|e| panic!("bad model config: {e}"));
        c
    }

    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.vocab == 0 || self.d == 0 || self.layers == 0 || self.ctx == 0 {
            return Err(format!("vocab, d, layers and ctx must all be positive: {self:?}"));
        }
        if (MLP_MULT * self.d) % KWTA_DIV != 0 {
            return Err(format!("4d = {} must be divisible by {KWTA_DIV} for kWTA", MLP_MULT * self.d));
        }
        Ok(())
    }

    pub fn hidden(&self) -> usize {
        MLP_MULT * self.d
    }

    pub fn kwta_k(&self) -> usize {
        self.hidden() / KWTA_DIV
    }

    /// Init std of projections that write into the residual stream.
    pub fn resid_std(&self) -> f64 {
        INIT_STD / ((2 * self.layers) as f64).sqrt()
    }

    /// (all weights, weights excluding the token embedding).
    pub fn weight_counts(&self) -> (usize, usize) {
        let (v, d, h) = (self.vocab, self.d, self.hidden());
        let per_layer = d + d * d + d + d * d + d + d * h + h * d;
        let total = v * d + self.layers * per_layer + d;
        (total, total - v * d)
    }
}

struct SpikeMem {
    norm: Tensor,
    win: Tensor,
    beta: Tensor,
    wout: Tensor,
}

struct Mlp {
    norm: Tensor,
    w1: Tensor,
    w2: Tensor,
}

struct Block {
    mem: SpikeMem,
    mlp: Mlp,
}

pub struct Model {
    pub cfg: Config,
    tok: Tensor,
    blocks: Vec<Block>,
    final_norm: Tensor,
}

/// Per-layer activity of one forward pass (means over every position), as 0-d tensors on the device.
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
pub fn kwta_mask(u: &Tensor, k: usize) -> Result<Tensor> {
    let h = u.dim(D::Minus1)?;
    let u = u.detach();
    if k >= h {
        return u.ones_like();
    }
    let (sorted, _idx) = u.sort_last_dim(false)?;
    let kth = sorted.narrow(D::Minus1, k - 1, 1)?; // [N, 1]
    u.broadcast_sub(&kth)?.ge(0.0)?.to_dtype(DType::F32)
}

impl Model {
    /// Build the model's parameters in `vb` (fresh init when the VarMap is empty; the saved values when
    /// the VarMap was loaded first).
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Model> {
        cfg.validate().map_err(candle_core::Error::msg)?;
        let (v, d, hid) = (cfg.vocab, cfg.d, cfg.hidden());
        let rs = cfg.resid_std();
        let randn = |std: f64| Init::Randn { mean: 0.0, stdev: std };
        let tok = vb.get_with_hints((v, d), TOK_EMB, randn(INIT_STD))?;
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let vm = vb.pp(format!("b{i}.mem"));
            let mem = SpikeMem {
                norm: vm.get_with_hints(d, "norm", Init::Const(1.0))?,
                win: vm.get_with_hints((d, d), "win", randn(1.0 / (d as f64).sqrt()))?,
                beta: vm.get_with_hints(d, "beta", Init::Const(logit(LIF_BETA0)))?,
                wout: vm.get_with_hints((d, d), "wout", randn(rs))?,
            };
            let vl = vb.pp(format!("b{i}.mlp"));
            let mlp = Mlp {
                norm: vl.get_with_hints(d, "norm", Init::Const(1.0))?,
                w1: vl.get_with_hints((d, hid), "w1", randn(INIT_STD))?,
                w2: vl.get_with_hints((hid, d), "w2", randn(rs))?,
            };
            blocks.push(Block { mem, mlp });
        }
        let final_norm = vb.get_with_hints(d, "final.norm", Init::Const(1.0))?;
        Ok(Model { cfg, tok, blocks, final_norm })
    }

    /// Whether the parameter called `name` gets weight decay (the weight matrices only).
    pub fn decays(name: &str) -> bool {
        name.ends_with(".win") || name.ends_with(".wout") || name.ends_with(".w1") || name.ends_with(".w2")
    }

    /// Logits [B·T, V] for `ids` [B, T] (u32), plus per-layer activity means.
    pub fn forward_with_activity(&self, ids: &Tensor) -> Result<(Tensor, Activity)> {
        let (b, t) = ids.dims2()?;
        let d = self.cfg.d;
        let n = b * t;
        let mut act = Activity { spike_rate: Vec::new(), kwta_post: Vec::new() };
        let mut x = self.tok.embedding(&ids.flatten_all()?)?; // [N, d]
        for blk in &self.blocks {
            let m = &blk.mem;
            let h = rmsnorm(&x, &m.norm)?;
            let cur = h.matmul(&m.win)?.reshape((b, t, d))?;
            let s = lif_scan(&cur, &m.beta)?;
            act.spike_rate.push(s.detach().mean_all()?);
            let o = s.reshape((n, d))?.matmul(&m.wout)?;
            x = x.add(&o)?;

            let l = &blk.mlp;
            let h = rmsnorm(&x, &l.norm)?;
            let u = h.matmul(&l.w1)?;
            let mask = kwta_mask(&u, self.cfg.kwta_k())?;
            let a = u.mul(&mask)?.relu()?;
            act.kwta_post.push(a.detach().gt(0.0)?.to_dtype(DType::F32)?.mean_all()?);
            let o = a.matmul(&l.w2)?;
            x = x.add(&o)?;
        }
        let x = rmsnorm(&x, &self.final_norm)?;
        let logits = x.matmul(&self.tok.t()?)?;
        Ok((logits, act))
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        Ok(self.forward_with_activity(ids)?.0)
    }

    /// Mean cross-entropy (f32 nats/token) of `ids` [B, T] against `targets` [B, T].
    pub fn loss(&self, ids: &Tensor, targets: &Tensor) -> Result<(Tensor, Activity)> {
        let (logits, act) = self.forward_with_activity(ids)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &targets.flatten_all()?)?;
        Ok((loss, act))
    }
}

impl Module for Model {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        Model::forward(self, ids)
    }
}
