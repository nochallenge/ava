// TESTING / DESIGN ONLY
//! A7: latent depth with learned halting (PonderNet, Banino et al. 2021; like ACT, Graves 2016). People
//! think longer about hard questions. Here a block can be run again on its own output up to `n_max`
//! times, and after each pass a halting unit says how likely it is to stop:
//!   λ_n = sigmoid(h_n·w + b),  p_n = λ_n Π_{j<n} (1 − λ_j)   (the last pass takes what is left)
//!   output = Σ_n p_n·h_n,  expected passes = Σ_n n·p_n
//! A KL term to a geometric prior with rate `prior` (PonderNet) keeps the depth from growing without need.
//! The bias b starts high (sigmoid(4) = 0.98), so at the start nearly every token stops after one pass,
//! as today.

use crate::linear;
use candle_core::{Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

pub struct Halting {
    pub w: Tensor,
    pub b: Tensor,
}

pub struct Pondered {
    /// [.., d]
    pub output: Tensor,
    /// [.., n_max]: the chance of stopping after each pass (sums to 1).
    pub p: Tensor,
    /// [..]
    pub expected_passes: Tensor,
}

impl Halting {
    pub fn new(vb: VarBuilder, d: usize) -> Result<Halting> {
        Ok(Halting { w: vb.get_with_hints((d, 1), "w", Init::Const(0.0))?, b: vb.get_with_hints(1, "b", Init::Const(4.0))? })
    }

    pub fn run(&self, h0: &Tensor, step: &dyn Fn(&Tensor) -> Result<Tensor>, n_max: usize) -> Result<Pondered> {
        let mut h = h0.clone();
        let mut left = h0.narrow(D::Minus1, 0, 1)?.ones_like()?;
        let (mut out, mut ps) = (None::<Tensor>, Vec::with_capacity(n_max));
        for n in 0..n_max {
            h = step(&h)?;
            let lam = if n + 1 == n_max { left.ones_like()? } else { candle_nn::ops::sigmoid(&linear(&h, &self.w)?.broadcast_add(&self.b)?)? };
            let p = (&left * &lam)?;
            left = (&left * lam.affine(-1.0, 1.0)?)?;
            let term = h.broadcast_mul(&p)?;
            out = Some(match out {
                None => term,
                Some(o) => (o + term)?,
            });
            ps.push(p);
        }
        let p = Tensor::cat(&ps, D::Minus1)?;
        let idx = Tensor::arange(1f32, n_max as f32 + 1.0, p.device())?;
        let expected_passes = p.broadcast_mul(&idx)?.sum(D::Minus1)?;
        Ok(Pondered { output: out.unwrap(), p, expected_passes })
    }
}

/// KL(p ‖ geometric(prior)) per position, truncated to n_max passes and renormalised.
pub fn kl_to_geometric(p: &Tensor, prior: f64) -> Result<Tensor> {
    let n = p.dim(D::Minus1)?;
    let g: Vec<f32> = (0..n).map(|k| (prior * (1.0 - prior).powi(k as i32)) as f32).collect();
    let s: f32 = g.iter().sum();
    let g = Tensor::from_vec(g.iter().map(|x| x / s).collect::<Vec<f32>>(), n, p.device())?;
    let pl = (p + 1e-9)?.log()?;
    (p * pl.broadcast_sub(&g.log()?)?)?.sum(D::Minus1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_core::Device;

    #[test]
    fn halting_mixes_passes_and_learns_when_to_stop() {
        let (vm, vb) = vm();
        let hl = Halting::new(vb, 4).unwrap();
        let h0 = randn(&[3, 4], 1);
        let step = |h: &Tensor| (h * 0.5)? + 1.0;
        let r = hl.run(&h0, &step, 4).unwrap();
        let p = r.p.to_vec2::<f32>().unwrap();
        for row in &p {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5 && row[0] > 0.97);
        }
        let one = step(&h0).unwrap();
        assert!(max_diff(&r.output, &one) < 0.1);
        let e = r.expected_passes.to_vec1::<f32>().unwrap();
        assert!(e.iter().all(|&x| x > 1.0 && x < 1.1));

        // With the bias at 0, stopping is 50/50 each pass: p = 1/2, 1/4, 1/8, then 1/8 left over.
        set(&vm, "b", &Tensor::new(&[0f32], &Device::Cpu).unwrap());
        let r = hl.run(&h0, &step, 4).unwrap();
        let p = r.p.to_vec2::<f32>().unwrap();
        for (a, b) in p[0].iter().zip([0.5f32, 0.25, 0.125, 0.125]) {
            assert!((a - b).abs() < 1e-6);
        }
        // The KL to a geometric prior with rate 1/2 is small but not zero (the tail is cut), and the halting
        // unit gets a gradient from the task loss and the KL.
        let kl = kl_to_geometric(&r.p, 0.5).unwrap().mean_all().unwrap();
        assert!(kl.to_scalar::<f32>().unwrap() >= 0.0);
        let loss = (r.output.sqr().unwrap().mean_all().unwrap() + kl).unwrap();
        let g = grad_norms(&vm, &loss, &["w", "b"]);
        assert!(g[0] > 1e-5 && g[1] > 1e-5, "{g:?}");
    }
}
