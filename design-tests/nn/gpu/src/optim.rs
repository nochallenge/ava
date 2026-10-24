// TESTING / DESIGN ONLY — ava_gpu::optim: AdamW with per-parameter weight decay, global-norm
// gradient clipping and the warmup + cosine LR schedule, all matching ava_grad::optim.
//
// AdamW step t (1-based): m = b1·m + (1-b1)·g; v = b2·v + (1-b2)·g^2;
// w = w·(1 - lr·wd·[decay]) - (lr / (1 - b1^t)) · m / (sqrt(v) / sqrt(1 - b2^t) + eps).
// Clipping: if |g| > clip, every gradient is scaled by clip / (|g| + 1e-6) before the step.

use candle_core::backprop::GradStore;
use candle_core::{DType, Result, Tensor, Var};

pub struct Param {
    pub name: String,
    pub var: Var,
    pub decay: bool,
}

pub struct AdamW {
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub t: u64,
    params: Vec<Param>,
    // Vars so each step writes into the same buffers (Var::set copies in place). Fresh moment tensors every
    // step would live across the next step's transient allocations and pin the CUDA pool's chunks: the pool's
    // reserved memory grew ~17 MB/step until the card thrashed (v33 Ava-M run, step ~650).
    m: Vec<Var>,
    v: Vec<Var>,
}

impl AdamW {
    /// b1 = 0.9, b2 = 0.95, eps = 1e-8.
    pub fn new(params: Vec<Param>, weight_decay: f64) -> Result<AdamW> {
        let mut m = Vec::with_capacity(params.len());
        let mut v = Vec::with_capacity(params.len());
        for p in &params {
            m.push(Var::zeros(p.var.shape(), p.var.dtype(), p.var.device())?);
            v.push(Var::zeros(p.var.shape(), p.var.dtype(), p.var.device())?);
        }
        Ok(AdamW { beta1: 0.9, beta2: 0.95, eps: 1e-8, weight_decay, t: 0, params, m, v })
    }

    pub fn params(&self) -> &[Param] {
        &self.params
    }

    /// Global L2 norm of the gradients of all parameters (one device-to-host copy).
    pub fn grad_norm(&self, grads: &GradStore) -> Result<f64> {
        let mut acc: Option<Tensor> = None;
        for p in &self.params {
            if let Some(g) = grads.get(p.var.as_tensor()) {
                let s = g.sqr()?.sum_all()?;
                acc = Some(match acc {
                    None => s,
                    Some(a) => a.add(&s)?,
                });
            }
        }
        match acc {
            None => Ok(0.0),
            Some(a) => Ok(a.to_dtype(DType::F64)?.to_scalar::<f64>()?.sqrt()),
        }
    }

    /// One AdamW step; `grad_scale` multiplies every gradient first (the clipping factor).
    pub fn step(&mut self, grads: &GradStore, lr: f64, grad_scale: f64) -> Result<()> {
        self.t += 1;
        let (b1, b2) = (self.beta1, self.beta2);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        let step_size = lr / bc1;
        let inv_sqrt_bc2 = 1.0 / bc2.sqrt();
        for (i, p) in self.params.iter().enumerate() {
            let Some(g) = grads.get(p.var.as_tensor()) else { continue };
            let g = if grad_scale != 1.0 { g.affine(grad_scale, 0.0)? } else { g.clone() };
            // detach g: the moments must not keep an op chain into the step's graph
            let g = g.detach();
            let m = (self.m[i].affine(b1, 0.0)? + g.affine(1.0 - b1, 0.0)?)?;
            let v = (self.v[i].affine(b2, 0.0)? + g.sqr()?.affine(1.0 - b2, 0.0)?)?;
            let denom = (v.sqrt()?.affine(inv_sqrt_bc2, 0.0)? + self.eps)?;
            let shrink = if p.decay { 1.0 - lr * self.weight_decay } else { 1.0 };
            let w = (p.var.as_tensor().affine(shrink, 0.0)? - m.div(&denom)?.affine(step_size, 0.0)?)?;
            p.var.set(&w)?;
            self.m[i].set(&m)?;
            self.v[i].set(&v)?;
        }
        Ok(())
    }
}

/// Clipping factor for a gradient norm: clip / (norm + 1e-6) when norm > clip, else 1.
pub fn clip_scale(norm: f64, clip: f64) -> f64 {
    if clip > 0.0 && norm > clip && norm > 0.0 {
        clip / (norm + 1e-6)
    } else {
        1.0
    }
}

/// Linear warmup over `warmup` steps (step 0 gets max_lr/warmup, step warmup-1 gets max_lr), then cosine
/// decay from max_lr at step `warmup` to min_lr at step `total`; min_lr afterwards. Same arithmetic as
/// ava_grad::optim::cosine_lr.
pub fn cosine_lr(step: usize, warmup: usize, total: usize, max_lr: f32, min_lr: f32) -> f32 {
    if step < warmup {
        return max_lr * (step + 1) as f32 / warmup as f32;
    }
    if step >= total || total <= warmup {
        return min_lr;
    }
    let progress = (step - warmup) as f64 / (total - warmup) as f64;
    let cos = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    (min_lr as f64 + (max_lr - min_lr) as f64 * cos) as f32
}
