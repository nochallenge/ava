// TESTING / DESIGN ONLY
//! F12: an attention schema (Graziano's attention schema theory; AST-1). The brain keeps a simple model
//! of its own attention, and uses it to steer that attention, much as the body schema helps steer the
//! arm. Here a small head predicts, from her state, where her attention (or her memory gates) will go:
//!   â_t = softmax(h_t·W_s)      trained by KL(a_t ‖ â_t), a_t detached
//! Two uses:
//! - awareness: KL(a_t ‖ â_t) is high when her attention is pulled somewhere she did not expect (a
//!   distraction), a signal she can report ("I got distracted") and that modulation (B8) can use;
//! - control: a' ∝ a^(1−β) · â^β pulls attention back toward what the schema expects. β defaults to 0
//!   (no change), and W_s starts at 0 (a uniform guess).

use crate::linear;
use candle_core::{Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

pub struct Schema {
    pub w: Tensor,
}

impl Schema {
    pub fn new(vb: VarBuilder, d: usize, items: usize) -> Result<Schema> {
        Ok(Schema { w: vb.get_with_hints((d, items), "w", Init::Const(0.0))? })
    }

    /// The predicted attention [.., items] from h [.., d] (h detached: the schema only watches).
    pub fn predict(&self, h: &Tensor) -> Result<Tensor> {
        candle_nn::ops::softmax(&linear(&h.detach(), &self.w)?, D::Minus1)
    }
}

/// KL(a ‖ â) per position (a detached), the training loss and the awareness signal.
pub fn surprise(actual: &Tensor, predicted: &Tensor) -> Result<Tensor> {
    let a = actual.detach();
    (&a * ((&a + 1e-9)?.log()? - (predicted + 1e-9)?.log()?)?)?.sum(D::Minus1)
}

/// a' ∝ a^(1−β)·â^β.
pub fn steer(actual: &Tensor, predicted: &Tensor, beta: f64) -> Result<Tensor> {
    let l = (((actual + 1e-9)?.log()? * (1.0 - beta))? + ((predicted + 1e-9)?.log()? * beta)?)?;
    candle_nn::ops::softmax(&l, D::Minus1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_nn::{AdamW, Optimizer, ParamsAdamW};

    #[test]
    fn learns_her_attention_notices_distraction_and_steers_back() {
        let (d, items) = (6, 5);
        let h = randn(&[128, d], 1);
        let rule = randn(&[d, items], 2);
        let attn = candle_nn::ops::softmax(&h.matmul(&rule).unwrap(), D::Minus1).unwrap();
        let (vm, vb) = vm();
        let s = Schema::new(vb, d, items).unwrap();
        let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 0.05, weight_decay: 0.0, ..Default::default() }).unwrap();
        let loss = |s: &Schema| surprise(&attn, &s.predict(&h).unwrap()).unwrap().mean_all().unwrap();
        let first = loss(&s).to_scalar::<f32>().unwrap();
        for _ in 0..300 {
            opt.backward_step(&loss(&s)).unwrap();
        }
        let last = loss(&s).to_scalar::<f32>().unwrap();
        assert!(last < 0.1 * first, "{first} -> {last}");
        // A distractor grabs item 0 (+4 on its logit): the schema did not expect it.
        let mut bump = vec![0f32; items];
        bump[0] = 4.0;
        let bump = candle_core::Tensor::from_vec(bump, items, h.device()).unwrap();
        let grabbed = candle_nn::ops::softmax(&h.matmul(&rule).unwrap().broadcast_add(&bump).unwrap(), D::Minus1).unwrap();
        let pred = s.predict(&h).unwrap();
        let aware = surprise(&grabbed, &pred).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(aware > 10.0 * last, "{aware} vs {last}");
        // Steering toward the schema takes attention back off the distractor; β = 0 changes nothing.
        let on0 = |a: &Tensor| a.narrow(1, 0, 1).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(max_diff(&steer(&grabbed, &pred, 0.0).unwrap(), &grabbed) < 1e-5);
        assert!(on0(&steer(&grabbed, &pred, 0.5).unwrap()) < on0(&grabbed) - 0.1);
    }
}
