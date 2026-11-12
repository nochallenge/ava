// TESTING / DESIGN ONLY
//! E7: resonate-and-fire neurons (Izhikevich 2001), an option in place of the LIF scan. A LIF neuron
//! only integrates. A resonator is a damped oscillator, so it answers most to inputs that repeat at its own
//! period: it can pick out rhythms and timing that a leaky integrator smears together. Recent spiking
//! models train them well with surrogate gradients (Higuchi et al., "Balanced resonate-and-fire neurons",
//! ICML 2024).
//!
//! Each channel is a complex state z_t = r·e^{iω}·z_{t-1} + I_t, and it spikes when Re z_t ≥ θ. There is
//! no reset, so the membrane is a linear filter of the input:
//!   Re z_t = Σ_k r^k cos(kω) I_{t-k}
//! That allows a parallel (FFT or chunked) form on the GPU later; here it is the exact step-by-step scan.
//! r = sigmoid(r_param) starts at 0.9 (LIF's leak). The periods 2π/ω start log-spaced from 2 to
//! `max_period` steps over the channels.

use crate::{logit, spike};
use candle_core::{Result, Tensor};
use candle_nn::{Init, VarBuilder};

pub struct Resonate {
    /// [C], r = sigmoid(r_param).
    pub r_param: Tensor,
    /// [C], fixed: the starting angles per step (radians), periods log-spaced from 2 to `max_period`.
    pub omega_base: Tensor,
    /// [C], learned, starts at 0: ω = omega_base + omega_delta.
    pub omega_delta: Tensor,
}

impl Resonate {
    pub fn new(vb: VarBuilder, channels: usize, max_period: f64) -> Result<Resonate> {
        let r_param = vb.get_with_hints(channels, "r_param", Init::Const(logit(0.9)))?;
        let omega_delta = vb.get_with_hints(channels, "omega_delta", Init::Const(0.0))?;
        let w: Vec<f32> = (0..channels)
            .map(|c| {
                let f = if channels > 1 { c as f64 / (channels - 1) as f64 } else { 0.0 };
                (2.0 * std::f64::consts::PI / (2.0 * (max_period / 2.0).powf(f))) as f32
            })
            .collect();
        let omega_base = Tensor::from_vec(w, channels, omega_delta.device())?;
        Ok(Resonate { r_param, omega_base, omega_delta })
    }

    pub fn omega(&self) -> Result<Tensor> {
        &self.omega_base + &self.omega_delta
    }

    /// The membrane Re z [B, T, C] for input current [B, T, C].
    pub fn membrane(&self, cur: &Tensor) -> Result<Tensor> {
        let (b, t, c) = cur.dims3()?;
        let r = candle_nn::ops::sigmoid(&self.r_param)?;
        let om = self.omega()?;
        let (rc, rs) = ((&r * om.cos()?)?, (&r * om.sin()?)?);
        let mut re = Tensor::zeros((b, c), cur.dtype(), cur.device())?;
        let mut im = re.clone();
        let mut out = Vec::with_capacity(t);
        for k in 0..t {
            let i = cur.narrow(1, k, 1)?.squeeze(1)?;
            let re2 = ((re.broadcast_mul(&rc)? - im.broadcast_mul(&rs)?)? + i)?;
            im = (re.broadcast_mul(&rs)? + im.broadcast_mul(&rc)?)?;
            re = re2;
            out.push(re.clone());
        }
        Tensor::stack(&out, 1)
    }

    /// Spikes [B, T, C]: 1 where the membrane reaches `theta`, with the fast-sigmoid surrogate.
    pub fn forward(&self, cur: &Tensor, theta: f64, slope: f64) -> Result<Tensor> {
        spike(&(self.membrane(cur)? - theta)?, slope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_core::Device;

    #[test]
    fn impulse_rings_at_its_period_and_resonance_wins() {
        let (vm, vb) = vm();
        let n = Resonate::new(vb, 3, 32.0).unwrap();
        let om = n.omega().unwrap().to_vec1::<f32>().unwrap();
        let periods: Vec<f64> = om.iter().map(|&w| 2.0 * std::f64::consts::PI / w as f64).collect();
        assert!((periods[0] - 2.0).abs() < 1e-3 && (periods[1] - 8.0).abs() < 1e-3 && (periods[2] - 32.0).abs() < 1e-2, "{periods:?}");

        // Impulse response = r^k cos(k ω), exactly.
        let mut imp = vec![0f32; 20 * 3];
        imp[0] = 1.0;
        imp[1] = 1.0;
        imp[2] = 1.0;
        let m = n.membrane(&Tensor::from_vec(imp, (1, 20, 3), &Device::Cpu).unwrap()).unwrap().squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        for k in 0..20 {
            for c in 0..3 {
                let want = 0.9f64.powi(k as i32) * (k as f64 * om[c] as f64).cos();
                assert!((m[k][c] as f64 - want).abs() < 1e-4, "k {k} c {c}: {} vs {want}", m[k][c]);
            }
        }

        // A drive at period 8 makes the period-8 channel swing far more than the others.
        let t = 64;
        let drive: Vec<f32> = (0..t).flat_map(|k| {
            let v = (2.0 * std::f32::consts::PI * k as f32 / 8.0).cos();
            [v, v, v]
        }).collect();
        let m = n.membrane(&Tensor::from_vec(drive, (1, t, 3), &Device::Cpu).unwrap()).unwrap().squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        let amp = |c: usize| m[32..].iter().map(|r| r[c].abs()).fold(0f32, f32::max);
        assert!(amp(1) > 2.0 * amp(0) && amp(1) > 2.0 * amp(2), "{} {} {}", amp(0), amp(1), amp(2));

        // Spikes are differentiable in both the leak and the frequency.
        let loss = n.forward(&randn(&[2, 16, 3], 9), 0.5, 4.0).unwrap().sum_all().unwrap();
        let g = grad_norms(&vm, &loss, &["r_param", "omega_delta"]);
        assert!(g[0] > 1e-4 && g[1] > 1e-4, "{g:?}");
    }
}
