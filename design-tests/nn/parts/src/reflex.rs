// TESTING / DESIGN ONLY
//! H3: practice turns slow thinking into a reflex (System 2 → System 1; ACT-R production compilation,
//! Anderson; the shift from effortful to automatic skill, Fitts & Posner). The first times she solves
//! something she uses her slow path: several passes (`crate::halting`) or a written thought
//! (`crate::rlp`). Once a skill has been practised often and gone right, her fast single pass is trained
//! to give the slow path's answer directly:
//!   loss = w(count, success) · KL(slow ‖ fast)        (the slow answer detached; temperature τ)
//!   w = success² · (1 − exp(−count / n0))             (graded: no switch, no fixed count)
//! A wrong slow answer is never compiled into a reflex: w goes to 0 as success goes to 0.

use candle_core::{Result, Tensor, D};

/// KL(teacher ‖ student) over the last dimension, mean over the other positions; the teacher is detached.
pub fn distill_loss(student: &Tensor, teacher: &Tensor, temperature: f64) -> Result<Tensor> {
    let s = candle_nn::ops::log_softmax(&(student / temperature)?, D::Minus1)?;
    let t = candle_nn::ops::log_softmax(&(teacher.detach() / temperature)?, D::Minus1)?;
    let kl = (t.exp()? * (t - s)?)?.sum(D::Minus1)?;
    kl.mean_all()? * (temperature * temperature)
}

pub fn compile_weight(count: u64, success: f32, n0: f32) -> f32 {
    success.clamp(0.0, 1.0).powi(2) * (1.0 - (-(count as f32) / n0).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_nn::{AdamW, Optimizer, ParamsAdamW};

    #[test]
    fn the_fast_path_learns_the_slow_answer() {
        let x = randn(&[64, 6], 1);
        // The slow path: three passes of a fixed map (what pondering would compute).
        let m = (randn(&[6, 6], 2) * 0.5).unwrap();
        let mut h = x.clone();
        for _ in 0..3 {
            h = h.matmul(&m).unwrap().tanh().unwrap();
        }
        let teacher = h.matmul(&randn(&[6, 5], 3)).unwrap();
        let w = candle_core::Var::from_tensor(&(randn(&[6, 5], 4) * 0.1).unwrap()).unwrap();
        assert!(distill_loss(&teacher, &teacher, 2.0).unwrap().to_scalar::<f32>().unwrap().abs() < 1e-6);
        let mut opt = AdamW::new(vec![w.clone()], ParamsAdamW { lr: 0.05, weight_decay: 0.0, ..Default::default() }).unwrap();
        let first = distill_loss(&x.matmul(w.as_tensor()).unwrap(), &teacher, 1.0).unwrap().to_scalar::<f32>().unwrap();
        for _ in 0..300 {
            opt.backward_step(&distill_loss(&x.matmul(w.as_tensor()).unwrap(), &teacher, 1.0).unwrap()).unwrap();
        }
        let last = distill_loss(&x.matmul(w.as_tensor()).unwrap(), &teacher, 1.0).unwrap().to_scalar::<f32>().unwrap();
        assert!(last < 0.3 * first, "{first} -> {last}");
    }

    #[test]
    fn compiling_grows_with_practice_and_success() {
        assert_eq!(compile_weight(0, 1.0, 20.0), 0.0);
        assert!(compile_weight(5, 1.0, 20.0) < compile_weight(50, 1.0, 20.0));
        assert!(compile_weight(200, 1.0, 20.0) > 0.99);
        assert!(compile_weight(200, 0.5, 20.0) < 0.26);
        assert_eq!(compile_weight(200, 0.0, 20.0), 0.0);
    }
}
