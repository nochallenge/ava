// TESTING / DESIGN ONLY
//! B2: a critic that learns by temporal differences, like the dopamine system (Schultz, Dayan & Montague
//! 1997). It guesses how good things will turn out, V(h), and learns from the surprise in what happens next:
//!   δ_t = r_t + γ·V_{t+1} − V_t   (the reward prediction error: phasic dopamine)
//! The λ-return mixes one-step and long-run targets:
//!   G_t^λ = r_t + γ((1 − λ)V_{t+1} + λ·G_{t+1}^λ)
//! GAE advantages A_t = Σ_k (γλ)^k δ_{t+k} say how much better an action was than expected. They are what
//! reward learning (Wave 2A's U4 reward step, M2 RLP) should use instead of raw rewards.
//! The value head V(h) = h·w + b starts at 0.

use crate::linear;
use candle_core::{Result, Tensor};
use candle_nn::{Init, VarBuilder};

pub struct Critic {
    pub w: Tensor,
    pub b: Tensor,
}

impl Critic {
    pub fn new(vb: VarBuilder, d: usize) -> Result<Critic> {
        Ok(Critic { w: vb.get_with_hints((d, 1), "w", Init::Const(0.0))?, b: vb.get_with_hints(1, "b", Init::Const(0.0))? })
    }
    /// V [..] for h [.., d].
    pub fn value(&self, h: &Tensor) -> Result<Tensor> {
        linear(h, &self.w)?.broadcast_add(&self.b)?.squeeze(candle_core::D::Minus1)
    }
}

/// TD errors for one episode: rewards r[0..T], values v[0..=T] (v[T] = value after the end; 0 if it ended).
pub fn td_errors(r: &[f32], v: &[f32], gamma: f32) -> Vec<f32> {
    (0..r.len()).map(|t| r[t] + gamma * v[t + 1] - v[t]).collect()
}

/// λ-returns (targets for V).
pub fn lambda_returns(r: &[f32], v: &[f32], gamma: f32, lambda: f32) -> Vec<f32> {
    let t = r.len();
    let mut g = vec![0f32; t];
    let mut next = v[t];
    for i in (0..t).rev() {
        g[i] = r[i] + gamma * ((1.0 - lambda) * v[i + 1] + lambda * next);
        next = g[i];
    }
    g
}

/// GAE advantages.
pub fn gae(r: &[f32], v: &[f32], gamma: f32, lambda: f32) -> Vec<f32> {
    let d = td_errors(r, v, gamma);
    let mut a = vec![0f32; d.len()];
    let mut acc = 0.0;
    for i in (0..d.len()).rev() {
        acc = d[i] + gamma * lambda * acc;
        a[i] = acc;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn td_lambda_and_gae_by_hand() {
        let (r, v) = ([0.0f32, 0.0, 1.0], [0.5f32, 0.5, 0.5, 0.0]);
        let d = td_errors(&r, &v, 0.9);
        assert!((d[0] - (0.45 - 0.5)).abs() < 1e-6 && (d[2] - 0.5).abs() < 1e-6);
        // λ = 1: Monte-Carlo returns; λ = 0: one-step targets.
        let mc = lambda_returns(&r, &v, 0.9, 1.0);
        assert!((mc[0] - 0.81).abs() < 1e-6 && (mc[2] - 1.0).abs() < 1e-6);
        let one = lambda_returns(&r, &v, 0.9, 0.0);
        assert!((one[0] - 0.45).abs() < 1e-6);
        // GAE at λ = 1 equals return − value.
        let a = gae(&r, &v, 0.9, 1.0);
        for i in 0..3 {
            assert!((a[i] - (mc[i] - v[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn the_value_head_starts_at_zero() {
        let (_vm, vb) = crate::testkit::vm();
        let c = Critic::new(vb, 5).unwrap();
        let v = c.value(&crate::testkit::randn(&[2, 7, 5], 1)).unwrap();
        assert_eq!(v.dims(), &[2, 7]);
        assert_eq!(v.abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap(), 0.0);
    }
}
