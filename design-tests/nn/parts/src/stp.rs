// TESTING / DESIGN ONLY
//! E9: short-term synaptic plasticity (Tsodyks & Markram 1997; Mongillo, Barak & Tsodyks 2008). A real
//! synapse's strength changes from spike to spike over tens to hundreds of milliseconds:
//! - depression: each spike uses up some of the ready resources x, which come back slowly (time τd);
//! - facilitation: each spike raises the release fraction u, which then fades (time τf).
//! Mongillo et al. proposed that facilitation holds working memory "silently", in the synapses, without
//! constant firing; it is a cheap memory with no extra activity.
//!
//! Per channel, for spikes s_t (0/1, or graded), with a = events(s) (= s for 0/1 spikes):
//!   u⁺ = u + U(1 − u)a        (facilitation jump; u rests at 0)
//!   e  = u⁺ x                 (efficacy)
//!   x  ← x − e·a              (resources used)
//!   then, to the next step: u ← u⁺·df, x ← 1 − (1 − x)·dd    (df = e^{−1/τf}, dd = e^{−1/τd})
//! The output is the spike times its efficacy relative to a lone spike at rest (e/U, so 1 for a lone spike):
//!   out = s + mix·(s·e/U − s)
//! mix starts at 0, so the part starts as no change. U, df and dd are learned per channel through
//! sigmoids. They start at U 0.2, τf 20 steps and τd 5 steps, i.e. facilitating.
//!
//! v42: the rule holds only for events in [0, 1]. A signed or multi-level spike (`--spikes int2` gives −2..2) fed in
//! directly drives u below 0 and x above 1, and the scan then grows without bound (it reached 3e17 in a d128 run with
//! every part on). So the synapse is driven by the event size a = min(|s|, 1); the output still scales s itself.

use crate::logit;
use candle_core::{Result, Tensor};
use candle_nn::ops::sigmoid;
use candle_nn::{Init, VarBuilder};

pub struct Stp {
    pub u_param: Tensor,
    pub f_param: Tensor,
    pub d_param: Tensor,
    pub mix: Tensor,
}

/// The per-step efficacies and the output, for inspection.
pub struct StpOut {
    pub out: Tensor,
    /// e/U at each step (whether or not there was a spike), [B, T, C].
    pub efficacy: Tensor,
}

/// The event size that drives the synapse: min(|s|, 1), with slope 1 on [0, 1] (so 0/1 spikes are unchanged in value
/// and gradient), −1 on [−1, 0) and 0 beyond. Written as s·k + (min(|s|, 1) − s·k) with the slope k detached, which is
/// exactly min(|s|, 1) in value (k = ±1 gives s·k = |s| for |s| <= 1; k = 0 beyond).
pub fn events(s: &Tensor) -> Result<Tensor> {
    let f = |m: Tensor| m.to_dtype(s.dtype());
    let pos = f(s.ge(0.0)?)?.mul(&f(s.le(1.0)?)?)?;
    let neg = f(s.lt(0.0)?)?.mul(&f(s.ge(-1.0)?)?)?;
    let k = pos.sub(&neg)?.detach();
    let sk = s.mul(&k)?;
    let a = s.abs()?.clamp(0f32, 1f32)?;
    sk.add(&a.sub(&sk)?.detach())
}

impl Stp {
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Stp> {
        Ok(Stp {
            u_param: vb.get_with_hints(channels, "u_param", Init::Const(logit(0.2)))?,
            f_param: vb.get_with_hints(channels, "f_param", Init::Const(logit((-1.0f64 / 20.0).exp())))?,
            d_param: vb.get_with_hints(channels, "d_param", Init::Const(logit((-1.0f64 / 5.0).exp())))?,
            mix: vb.get_with_hints(channels, "mix", Init::Const(0.0))?,
        })
    }

    /// spikes [B, T, C] -> out [B, T, C].
    pub fn forward(&self, s: &Tensor) -> Result<StpOut> {
        let (b, t, c) = s.dims3()?;
        let (uu, df, dd) = (sigmoid(&self.u_param)?, sigmoid(&self.f_param)?, sigmoid(&self.d_param)?);
        let mut u = Tensor::zeros((b, c), s.dtype(), s.device())?;
        let mut x = Tensor::ones((b, c), s.dtype(), s.device())?;
        let a = events(s)?;
        let mut effs = Vec::with_capacity(t);
        for k in 0..t {
            let sk = a.narrow(1, k, 1)?.squeeze(1)?;
            let u_plus = (&u + (u.affine(-1.0, 1.0)?.broadcast_mul(&uu)? * &sk)?)?;
            let e = (&u_plus * &x)?;
            effs.push(e.broadcast_div(&uu)?);
            let x_used = (&x - (&e * &sk)?)?;
            u = u_plus.broadcast_mul(&df)?;
            x = x_used.affine(-1.0, 1.0)?.broadcast_mul(&dd)?.affine(-1.0, 1.0)?;
        }
        let efficacy = Tensor::stack(&effs, 1)?;
        let out = (s + ((s * &efficacy)? - s)?.broadcast_mul(&self.mix)?)?;
        Ok(StpOut { out, efficacy })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_core::Device;

    fn train(n: usize) -> Tensor {
        Tensor::ones((1, n, 1), candle_core::DType::F32, &Device::Cpu).unwrap()
    }

    fn effs(st: &Stp, n: usize) -> Vec<f32> {
        st.forward(&train(n)).unwrap().efficacy.flatten_all().unwrap().to_vec1().unwrap()
    }

    #[test]
    fn depresses_facilitates_and_starts_as_no_change() {
        let (vm, vb) = vm();
        let st = Stp::new(vb, 1).unwrap();
        let s = randn(&[2, 9, 1], 4).ge(0.3f64).unwrap().to_dtype(candle_core::DType::F32).unwrap();
        assert!(max_diff(&st.forward(&s).unwrap().out, &s) < 1e-7);

        // Depressing synapse: high U, no facilitation memory, slow recovery. Each spike in a train is weaker.
        let p = |x: f64| Tensor::new(&[logit(x) as f32], &Device::Cpu).unwrap();
        set(&vm, "u_param", &p(0.8));
        set(&vm, "f_param", &p(1e-4));
        set(&vm, "d_param", &p(0.99));
        let e = effs(&st, 5);
        assert!((e[0] - 1.0).abs() < 1e-4 && e.windows(2).all(|w| w[1] < w[0]), "{e:?}");

        // Facilitating synapse: low U, long facilitation, fast recovery. Each spike is stronger.
        set(&vm, "u_param", &p(0.1));
        set(&vm, "f_param", &p(0.95));
        set(&vm, "d_param", &p(1e-4));
        let e = effs(&st, 5);
        assert!((e[0] - 1.0).abs() < 1e-4 && e.windows(2).all(|w| w[1] > w[0]) && e[4] > 2.0, "{e:?}");

        // With mix 1 the output is the efficacy-weighted spikes, and every parameter learns.
        set(&vm, "mix", &Tensor::new(&[1f32], &Device::Cpu).unwrap());
        let o = st.forward(&s).unwrap();
        let want = (&s * &o.efficacy).unwrap();
        assert!(max_diff(&o.out, &want) < 1e-6);
        let loss = o.out.sqr().unwrap().sum_all().unwrap();
        let g = grad_norms(&vm, &loss, &["u_param", "f_param", "mix"]);
        assert!(g.iter().all(|&x| x > 1e-5), "{g:?}");
    }

    /// v42: signed multi-level spikes (int2: −2..2) used to drive u below 0 and x above 1, and the scan grew without
    /// bound. Through events() every efficacy stays finite and at most 1/U; 0/1 spikes pass through events() unchanged.
    #[test]
    fn signed_multilevel_spikes_keep_the_rule_bounded() {
        let (vm, vb) = vm();
        let st = Stp::new(vb, 3).unwrap();
        let b = randn(&[2, 50, 3], 7).ge(0.3f64).unwrap().to_dtype(candle_core::DType::F32).unwrap();
        assert_eq!(max_diff(&events(&b).unwrap(), &b), 0.0);
        // a facilitating synapse with a small U, the case that grew fastest
        let p = |x: f64| Tensor::new(&[logit(x) as f32; 3], &Device::Cpu).unwrap();
        set(&vm, "u_param", &p(0.05));
        set(&vm, "f_param", &p(0.99));
        set(&vm, "d_param", &p(0.5));
        // random int2 spikes, then a sustained run of −2 (the pattern that made u fall by (1 + 2U)·df > 1 a step)
        let rnd = randn(&[2, 600, 3], 8).affine(1.5, 0.0).unwrap().round().unwrap().clamp(-2f32, 2f32).unwrap();
        let neg = Tensor::full(-2f32, (1, 300, 3), &Device::Cpu).unwrap();
        for s in [&rnd, &neg] {
            let e: Vec<f32> = st.forward(s).unwrap().efficacy.flatten_all().unwrap().to_vec1().unwrap();
            let mx = e.iter().fold(0f32, |m, x| m.max(x.abs()));
            assert!(e.iter().all(|x| x.is_finite()) && mx <= 1.0 / 0.05 + 1e-3, "max efficacy {mx}");
        }
        // the old rule (s fed in directly) on the sustained run leaves that range
        let (uu, df, dd) = (0.05f32, 0.99f32, 0.5f32);
        let sv: Vec<f32> = neg.flatten_all().unwrap().to_vec1().unwrap();
        let (mut u, mut x, mut worst) = (0f32, 1f32, 0f32);
        for k in 0..300 {
            let sk = sv[k * 3];
            let up = u + (1.0 - u) * uu * sk;
            let ek = up * x;
            worst = worst.max((ek / uu).abs());
            let xu = x - ek * sk;
            u = up * df;
            x = 1.0 - (1.0 - xu) * dd;
        }
        assert!(!worst.is_finite() || worst > 1.0 / uu, "the old rule stayed bounded ({worst}): the test no longer shows the bug");
    }
}
