// TESTING / DESIGN ONLY
//! B1: JEPA, predicting in the space of meaning, not of words (LeCun 2022; I-JEPA, Assran et al. 2023;
//! for language, LLM-JEPA, Huang et al. 2025). Besides the next token, she guesses what her own
//! state will be k steps on. The target is an EMA copy of her network (slow, detached), so the guess cannot
//! collapse to a trick. Brains are thought to work this way: predicting their own coming states, with
//! learning driven by the error.
//!
//! - `Predictor`: pred = h + MLP(h); the MLP's output weights start at 0, so pred starts as h.
//! - `jepa_loss(pred, target, k)`: mean over t of 1 − cos(pred_t, target_{t+k}), with target detached.
//! - `ema_update`: target ← decay·target + (1 − decay)·online, for each pair of weights.

use crate::linear;
use candle_core::{Result, Tensor, Var, D};
use candle_nn::{Init, VarBuilder};

pub struct Predictor {
    pub w1: Tensor,
    pub w2: Tensor,
}

impl Predictor {
    pub fn new(vb: VarBuilder, d: usize, hidden: usize) -> Result<Predictor> {
        Ok(Predictor {
            w1: vb.get_with_hints((d, hidden), "w1", Init::Randn { mean: 0.0, stdev: (1.0 / d as f64).sqrt() })?,
            w2: vb.get_with_hints((hidden, d), "w2", Init::Const(0.0))?,
        })
    }

    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        h + linear(&linear(h, &self.w1)?.gelu_erf()?, &self.w2)?
    }
}

/// pred, target [B, T, d]: the guess at t against the target state at t + k.
pub fn jepa_loss(pred: &Tensor, target: &Tensor, k: usize) -> Result<Tensor> {
    let t = pred.dim(1)?;
    let p = pred.narrow(1, 0, t - k)?;
    let g = target.detach().narrow(1, k, t - k)?;
    let dot = (&p * &g)?.sum(D::Minus1)?;
    let n = ((p.sqr()?.sum(D::Minus1)? * g.sqr()?.sum(D::Minus1)?)? + 1e-12)?.sqrt()?;
    (dot / n)?.affine(-1.0, 1.0)?.mean_all()
}

/// target ← decay·target + (1 − decay)·online.
pub fn ema_update(target: &[Var], online: &[Tensor], decay: f64) -> Result<()> {
    for (t, o) in target.iter().zip(online) {
        t.set(&((t.as_tensor() * decay)? + (o.detach() * (1.0 - decay))?)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_core::Device;

    #[test]
    fn loss_gradients_and_ema() {
        let (vm, vb) = vm();
        let p = Predictor::new(vb, 8, 16).unwrap();
        let h = randn(&[2, 10, 8], 3);
        assert!(max_diff(&p.forward(&h).unwrap(), &h) < 1e-7);

        // A perfect guess (the target shifted back by k) scores 0.
        let target = randn(&[2, 12, 8], 4);
        let perfect = target.narrow(1, 2, 10).unwrap().pad_with_zeros(1, 0, 2).unwrap();
        let l = jepa_loss(&perfect.narrow(1, 0, 12).unwrap(), &target, 2).unwrap().to_scalar::<f32>().unwrap();
        assert!(l.abs() < 1e-5, "{l}");

        // The predictor learns; the target gets no gradient.
        let tv = Var::from_tensor(&randn(&[2, 10, 8], 5)).unwrap();
        set(&vm, "w2", &randn(&[16, 8], 6));
        let loss = jepa_loss(&p.forward(&h).unwrap(), tv.as_tensor(), 1).unwrap();
        let g = loss.backward().unwrap();
        assert!(g.get(tv.as_tensor()).is_none());
        assert!(g.get(var(&vm, "w2").as_tensor()).is_some());

        // EMA moves the target 10% of the way.
        let a = Var::from_tensor(&Tensor::new(&[0f32, 10.0], &Device::Cpu).unwrap()).unwrap();
        ema_update(&[a.clone()], &[Tensor::new(&[10f32, 0.0], &Device::Cpu).unwrap()], 0.9).unwrap();
        let v = a.as_tensor().to_vec1::<f32>().unwrap();
        assert!((v[0] - 1.0).abs() < 1e-5 && (v[1] - 9.0).abs() < 1e-5, "{v:?}");
    }
}
