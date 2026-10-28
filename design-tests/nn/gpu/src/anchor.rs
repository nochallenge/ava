// TESTING / DESIGN ONLY — ava_gpu::anchor: v41 synaptic consolidation (like EWC, and like the brain keeping the
// synapses an old skill leans on).
//
// Before sleep or reward, each weight's importance to her usual reading is measured: F = the mean of g² over a few
// reading batches (g = the batch's reading gradient), divided by the mean of F over every weight (so F averages 1).
// After every optimizer step each weight is pulled back toward where it started:
//     w ← w − min(lr · strength · F, 1) · (w − w0)
// Weights her reading leans on hardly move; the others learn freely. A decoupled pull, like AdamW's weight decay,
// so it does not touch the Adam moments.

use crate::optim::Param;
use candle_core::backprop::GradStore;
use candle_core::{DType, Result, Tensor, Var};

/// Running sums of g² per parameter (None: the parameter got no gradient yet).
pub struct Importance {
    vars: Vec<Var>,
    acc: Vec<Option<Tensor>>,
    batches: usize,
}

impl Importance {
    pub fn new(params: &[Param]) -> Importance {
        Importance { vars: params.iter().map(|p| p.var.clone()).collect(), acc: vec![None; params.len()], batches: 0 }
    }

    /// Add one batch's gradient.
    pub fn add(&mut self, grads: &GradStore) -> Result<()> {
        for (k, v) in self.vars.iter().enumerate() {
            let Some(g) = grads.get(v.as_tensor()) else { continue };
            let sq = g.to_dtype(DType::F32)?.sqr()?;
            self.acc[k] = Some(match self.acc[k].take() {
                Some(s) => s.add(&sq)?,
                None => sq,
            });
        }
        self.batches += 1;
        Ok(())
    }

    /// The anchor at `strength`, holding every parameter that got a gradient at its current value. Returns None when
    /// no gradient was seen or the mean of g² is 0 or not finite.
    pub fn anchor(self, strength: f64) -> Result<Option<Anchor>> {
        let (mut sum, mut count) = (0f64, 0usize);
        for x in self.acc.iter().flatten() {
            sum += x.sum_all()?.to_scalar::<f32>()? as f64;
            count += x.elem_count();
        }
        let mean = sum / count.max(1) as f64;
        if count == 0 || !(mean.is_finite() && mean > 0.0) {
            return Ok(None);
        }
        let (mut vars, mut w0, mut imp) = (Vec::new(), Vec::new(), Vec::new());
        for (v, x) in self.vars.into_iter().zip(self.acc) {
            let Some(x) = x else { continue }; // a weight her reading never touches is free
            imp.push(x.affine(1.0 / mean, 0.0)?.to_dtype(v.as_tensor().dtype())?);
            w0.push(v.as_tensor().copy()?);
            vars.push(v);
        }
        Ok(Some(Anchor { vars, w0, imp, strength, weights: count }))
    }
}

pub struct Anchor {
    vars: Vec<Var>,
    w0: Vec<Tensor>,
    imp: Vec<Tensor>,
    pub strength: f64,
    /// the number of anchored weights
    pub weights: usize,
}

impl Anchor {
    /// The pull after an optimizer step at learning rate `lr`.
    pub fn pull(&self, lr: f64) -> Result<()> {
        let s = lr * self.strength;
        for ((v, w0), f) in self.vars.iter().zip(&self.w0).zip(&self.imp) {
            let w = v.as_tensor();
            let c = f.affine(s, 0.0)?.clamp(0f32, 1f32)?;
            let next = w.sub(&c.mul(&w.sub(w0)?)?)?;
            v.set(&next)?;
        }
        Ok(())
    }

    /// The importance-weighted mean squared distance from where she started (logged: how far she has moved).
    pub fn drift(&self) -> Result<f64> {
        let (mut s, mut n) = (0f64, 0usize);
        for ((v, w0), f) in self.vars.iter().zip(&self.w0).zip(&self.imp) {
            let d = v.as_tensor().sub(w0)?.sqr()?.mul(f)?.to_dtype(DType::F32)?;
            s += d.sum_all()?.to_scalar::<f32>()? as f64;
            n += d.elem_count();
        }
        Ok(s / n.max(1) as f64)
    }

    /// Each anchored parameter's importance (in the order of the parameters that got a gradient).
    pub fn importance(&self) -> &[Tensor] {
        &self.imp
    }
}
