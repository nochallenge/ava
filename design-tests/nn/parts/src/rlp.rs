// TESTING / DESIGN ONLY
//! M2: RLP, reinforcement learning as pre-training (Hatamizadeh et al., NVIDIA, 2025), on Ava's scale.
//! Before predicting what comes next, she may think: write a short thought first. The thought is rewarded
//! by how much it helps her predict the real continuation:
//!   reward = log p(next tokens | context, thought) − log p(next tokens | context, no thought)
//! where the no-thought score comes from a slow EMA copy of her (a fixed baseline, so she cannot make the
//! baseline worse to look better). There is no verifier and no labels: any text can teach her to think.
//! Several thoughts per context are compared with each other (group-relative advantages, as in GRPO):
//!   A_i = (r_i − mean r) / (std r + eps)
//! Thoughts are tagged imagined (C2), so their content never enters memory as fact.

/// The rewards for a group of thoughts: the log-probs of the true continuation with each thought, and
/// without any thought under the EMA baseline.
pub fn rewards(with_thought: &[f32], baseline: f32) -> Vec<f32> {
    with_thought.iter().map(|&l| l - baseline).collect()
}

/// Group-relative advantages.
pub fn advantages(r: &[f32], eps: f32) -> Vec<f32> {
    let n = r.len() as f32;
    let m = r.iter().sum::<f32>() / n;
    let sd = (r.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n).sqrt();
    r.iter().map(|x| (x - m) / (sd + eps)).collect()
}

/// The policy-gradient loss for the thoughts' log-probabilities (sum over each thought's tokens):
/// −mean(A_i · logp_i). Minimising it makes helpful thoughts likelier.
pub fn pg_loss(adv: &[f32], thought_logp: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
    let a = candle_core::Tensor::from_vec(adv.to_vec(), adv.len(), thought_logp.device())?;
    (thought_logp * a)?.mean_all()?.neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn helpful_thoughts_get_positive_advantage_and_become_likelier() {
        // Three thoughts: one helps (−2 → −1 nats), one does nothing, one misleads.
        let r = rewards(&[-1.0, -2.0, -3.0], -2.0);
        assert_eq!(r, vec![1.0, 0.0, -1.0]);
        let a = advantages(&r, 1e-6);
        assert!(a[0] > 1.2 && a[1].abs() < 1e-5 && a[2] < -1.2);
        assert!(a.iter().sum::<f32>().abs() < 1e-5);
        // A gradient step on the loss raises the helpful thought's log-prob and lowers the misleading one's.
        let lp = Var::new(&[-5f32, -5.0, -5.0], &Device::Cpu).unwrap();
        let g = pg_loss(&a, lp.as_tensor()).unwrap().backward().unwrap();
        let g = g.get(lp.as_tensor()).unwrap().to_vec1::<f32>().unwrap();
        assert!(g[0] < 0.0 && g[2] > 0.0);
    }
}
