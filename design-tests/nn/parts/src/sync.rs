// TESTING / DESIGN ONLY
//! L5 as a layer: binding by synchrony (von der Malsburg 1981; as neurons, AKOrN: Miyato et al. 2025).
//!
//! Each token starts n oscillators (unit 2-vectors) from its hidden state and runs a few coupled steps,
//!   x_i ← unit( R(γ·ω_i)·x_i + γ·P_i(Σ_j J_ij x_j + c_i) ),   P_i(y) = y − ⟨x_i, y⟩·x_i,
//! where J is a learned coupling, ω_i a learned own frequency and c_i a pull the token makes. In angles this is the
//! Kuramoto model, dθ_i = ω_i + Σ_j J_ij sin(θ_j − θ_i) + |c_i| sin(arg c_i − θ_i): oscillators that couple
//! positively fall into step, so which features are in step says which belong together. The final phases, x·Wo, join
//! the stream; Wo starts at 0, so the layer starts as no change. It looks at one token at a time (no state to carry).

use crate::linear;
use candle_core::{Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

pub const SYNC_STEPS: usize = 4;
pub const SYNC_GAMMA: f64 = 0.5;

pub struct Oscillators {
    /// [d, 2n]: the starting phases
    pub wx: Tensor,
    /// [d, 2n]: the pull
    pub wc: Tensor,
    /// [n, n]
    pub j: Tensor,
    /// [n]
    pub omega: Tensor,
    /// [2n, d], starts at 0
    pub wo: Tensor,
    pub n: usize,
    pub steps: usize,
    pub gamma: f64,
}

/// Scales each oscillator (axis 1 of [N, 2, n]) to unit length.
fn unit2(x: &Tensor) -> Result<Tensor> {
    x.broadcast_div(&(x.sqr()?.sum_keepdim(1)? + 1e-6)?.sqrt()?)
}

impl Oscillators {
    pub fn new(vb: VarBuilder, d: usize, n: usize) -> Result<Oscillators> {
        let sd = (1.0 / d as f64).sqrt();
        Ok(Oscillators {
            wx: vb.get_with_hints((d, 2 * n), "wx", Init::Randn { mean: 0.0, stdev: sd })?,
            wc: vb.get_with_hints((d, 2 * n), "wc", Init::Randn { mean: 0.0, stdev: sd })?,
            j: vb.get_with_hints((n, n), "j", Init::Randn { mean: 0.0, stdev: (1.0 / n as f64).sqrt() })?,
            omega: vb.get_with_hints(n, "omega", Init::Const(0.0))?,
            wo: vb.get_with_hints((2 * n, d), "wo", Init::Const(0.0))?,
            n,
            steps: SYNC_STEPS,
            gamma: SYNC_GAMMA,
        })
    }

    /// The oscillators after the steps, [N, 2, n] (axis 1: cos, sin), from starting phases x0 and pulls c (both
    /// [N, 2, n]).
    pub fn settle(&self, x0: &Tensor, c: &Tensor) -> Result<Tensor> {
        let g = self.gamma;
        let phi = self.omega.affine(g, 0.0)?;
        let (cs, sn) = (phi.cos()?, phi.sin()?);
        let jt = self.j.t()?.contiguous()?;
        let mut x = unit2(x0)?;
        for _ in 0..self.steps {
            let y = (linear(&x, &jt)? + c)?; // Σ_j J_ij x_j + c_i
            let p = (&y - x.broadcast_mul(&(&x * &y)?.sum_keepdim(1)?)?)?;
            let (a, b) = (x.narrow(1, 0, 1)?, x.narrow(1, 1, 1)?);
            let ra = (a.broadcast_mul(&cs)? - b.broadcast_mul(&sn)?)?;
            let rb = (a.broadcast_mul(&sn)? + b.broadcast_mul(&cs)?)?;
            x = unit2(&(Tensor::cat(&[ra, rb], 1)? + p.affine(g, 0.0)?)?)?;
        }
        Ok(x)
    }

    /// h [.., d] → what joins the stream [.., d].
    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let dims = h.dims().to_vec();
        let d = *dims.last().unwrap();
        let rows = h.elem_count() / d;
        let hh = h.reshape((rows, d))?;
        let x0 = linear(&hh, &self.wx)?.reshape((rows, 2, self.n))?;
        let c = linear(&hh, &self.wc)?.reshape((rows, 2, self.n))?;
        let x = self.settle(&x0, &c)?.reshape((rows, 2 * self.n))?;
        linear(&x, &self.wo)?.reshape(dims)
    }
}

/// How in step oscillators `idx` are in each row of x [N, 2, n] → [N] (the Kuramoto order parameter).
pub fn order(x: &Tensor, idx: &[u32]) -> Result<Tensor> {
    let ix = Tensor::new(idx, x.device())?;
    let m = x.index_select(&ix, 2)?.mean(D::Minus1)?; // [N, 2]
    m.sqr()?.sum(D::Minus1)?.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::kuramoto;
    use crate::testkit::*;
    use candle_core::Device;

    fn phases(x: &Tensor) -> Vec<Vec<f32>> {
        let v: Vec<Vec<Vec<f32>>> = x.to_vec3().unwrap();
        v.iter().map(|r| r[0].iter().zip(&r[1]).map(|(c, s)| s.atan2(*c)).collect()).collect()
    }

    fn from_angles(th: &[Vec<f32>]) -> Tensor {
        let n = th[0].len();
        let mut v = Vec::new();
        for r in th {
            v.extend(r.iter().map(|t| t.cos()));
            v.extend(r.iter().map(|t| t.sin()));
        }
        Tensor::from_vec(v, (th.len(), 2, n), &Device::Cpu).unwrap()
    }

    fn wrap(a: f32) -> f32 {
        let p = std::f32::consts::PI;
        (a + p).rem_euclid(2.0 * p) - p
    }

    /// A small step is a Kuramoto step: dθ_i = γ(ω_i + Σ_j J_ij sin(θ_j − θ_i) + |c_i| sin(arg c_i − θ_i)).
    #[test]
    fn a_small_step_is_a_kuramoto_step() {
        let (vm, vb) = vm();
        let n = 6;
        let s = Oscillators::new(vb, 4, n).unwrap();
        let jv: Vec<f32> = randn(&[n * n], 1).to_vec1().unwrap();
        let om: Vec<f32> = randn(&[n], 2).to_vec1().unwrap();
        set(&vm, "j", &Tensor::from_vec(jv.clone(), (n, n), &Device::Cpu).unwrap());
        set(&vm, "omega", &Tensor::from_vec(om.clone(), n, &Device::Cpu).unwrap());
        let g = 1e-3;
        let s = Oscillators { steps: 1, gamma: g, ..s };
        let th0: Vec<f32> = randn(&[n], 3).to_vec1().unwrap();
        let x0 = from_angles(&[th0.clone()]);
        // no pull: the host Kuramoto step (its k/n = 1 here)
        let zero = Tensor::zeros((1, 2, n), candle_core::DType::F32, &Device::Cpu).unwrap();
        let got = phases(&s.settle(&x0, &zero).unwrap())[0].clone();
        let mut want = th0.clone();
        kuramoto(&mut want, &om, &jv, n as f32, g as f32, 1);
        for i in 0..n {
            assert!(wrap(got[i] - want[i]).abs() < 3e-5, "osc {i}: {} vs {}", got[i], want[i]);
        }
        // a pull alone (J = 0, ω = 0) turns each oscillator toward its pull by |c| sin(arg c − θ)
        set(&vm, "j", &Tensor::zeros((n, n), candle_core::DType::F32, &Device::Cpu).unwrap());
        set(&vm, "omega", &Tensor::zeros(n, candle_core::DType::F32, &Device::Cpu).unwrap());
        let ca: Vec<f32> = randn(&[n], 4).to_vec1().unwrap();
        let cm: Vec<f32> = (0..n).map(|i| 0.5 + i as f32 * 0.3).collect();
        let mut cv = Vec::new();
        cv.extend(ca.iter().zip(&cm).map(|(a, m)| m * a.cos()));
        cv.extend(ca.iter().zip(&cm).map(|(a, m)| m * a.sin()));
        let c = Tensor::from_vec(cv, (1, 2, n), &Device::Cpu).unwrap();
        let got = phases(&s.settle(&x0, &c).unwrap())[0].clone();
        for i in 0..n {
            let w = th0[i] + g as f32 * cm[i] * (ca[i] - th0[i]).sin();
            assert!(wrap(got[i] - w).abs() < 3e-5, "pull {i}: {} vs {w}", got[i]);
        }
    }

    /// Two groups that pull together inside and apart across fall into step inside and out of step across.
    #[test]
    fn coupled_groups_fall_into_step_apart() {
        let (vm, vb) = vm();
        let n = 8;
        let s = Oscillators::new(vb, 4, n).unwrap();
        let grp = |i: usize| i < 4;
        let jv: Vec<f32> = (0..n * n).map(|k| if grp(k / n) == grp(k % n) { 1.0 } else { -0.5 }).collect();
        set(&vm, "j", &Tensor::from_vec(jv, (n, n), &Device::Cpu).unwrap());
        let s = Oscillators { steps: 60, gamma: 0.2, ..s };
        let th0: Vec<Vec<f32>> = (0..5).map(|k| randn(&[n], 10 + k).affine(3.0, 0.0).unwrap().to_vec1().unwrap()).collect();
        let x0 = from_angles(&th0);
        let zero = Tensor::zeros((5, 2, n), candle_core::DType::F32, &Device::Cpu).unwrap();
        let x = s.settle(&x0, &zero).unwrap();
        let before_a: Vec<f32> = order(&x0, &[0, 1, 2, 3]).unwrap().to_vec1().unwrap();
        let a: Vec<f32> = order(&x, &[0, 1, 2, 3]).unwrap().to_vec1().unwrap();
        let b: Vec<f32> = order(&x, &[4, 5, 6, 7]).unwrap().to_vec1().unwrap();
        let all: Vec<f32> = order(&x, &[0, 1, 2, 3, 4, 5, 6, 7]).unwrap().to_vec1().unwrap();
        for r in 0..5 {
            assert!(a[r] > 0.99 && b[r] > 0.99, "row {r}: in step {} {} (from {})", a[r], b[r], before_a[r]);
            assert!(all[r] < 0.3, "row {r}: the groups should end apart, all {}", all[r]);
        }
    }

    /// Starts as no change; every weight learns.
    #[test]
    fn starts_silent_and_learns() {
        let (vm, vb) = vm();
        let d = 10;
        let s = Oscillators::new(vb, d, 6).unwrap();
        let h = randn(&[2, 5, d], 7);
        assert_eq!(s.forward(&h).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap(), 0.0);
        seed(&vm, &[("wo", 1.0), ("omega", 0.5)], 8);
        let loss = s.forward(&h).unwrap().sqr().unwrap().sum_all().unwrap();
        let g = grad_norms(&vm, &loss, &["wx", "wc", "j", "omega", "wo"]);
        assert!(g.iter().all(|&x| x > 1e-6 && x.is_finite()), "{g:?}");
    }
}
