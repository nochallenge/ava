// TESTING / DESIGN ONLY
//! E5: learned synaptic delays. Real axons and synapses delay signals by different amounts, and learned
//! delays help spiking networks with timing (DCLS: Hammouamri, Khalfaoui-Hassani & Masquelier, ICLR 2024).
//!
//! Each channel c reads its input through a causal kernel over the last K steps. The kernel is a
//! normalised Gaussian bump centred at a learned delay d_c in [0, K-1] with width sigma, so the delay
//! itself gets a gradient:
//!   y = x + g ⊙ (kernel_c * x)
//! The gain g starts at 0, so the part starts as no change. The centres start spread over [0, K-1].

use candle_core::{Result, Tensor};
use candle_nn::{Init, VarBuilder};

pub struct Delays {
    /// [C], the delay of each channel in steps.
    pub center: Tensor,
    /// [C], how much of the delayed signal is added.
    pub gain: Tensor,
    pub k: usize,
    pub sigma: f64,
}

impl Delays {
    /// Weights `center` and `gain` under `vb`.
    pub fn new(vb: VarBuilder, channels: usize, k: usize, sigma: f64) -> Result<Delays> {
        let center = vb.get_with_hints(channels, "center", Init::Uniform { lo: 0.0, up: (k - 1) as f64 })?;
        let gain = vb.get_with_hints(channels, "gain", Init::Const(0.0))?;
        Ok(Delays { center, gain, k, sigma })
    }

    /// The kernels [C, K]; tap j is the weight of x_{t-j}. Each row sums to 1.
    pub fn kernel(&self) -> Result<Tensor> {
        let dev = self.center.device();
        let taps = Tensor::arange(0f32, self.k as f32, dev)?.unsqueeze(0)?;
        let c = self.center.clamp(0f32, (self.k - 1) as f32)?.unsqueeze(1)?;
        let e = (taps.broadcast_sub(&c)?.sqr()? * (-0.5 / (self.sigma * self.sigma)))?.exp()?;
        e.broadcast_div(&e.sum_keepdim(1)?)
    }

    /// x [B, T, C] -> [B, T, C].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(1)?;
        let w = self.kernel()?;
        let xp = x.pad_with_zeros(1, self.k - 1, 0)?;
        let mut acc: Option<Tensor> = None;
        for j in 0..self.k {
            let tap = xp.narrow(1, self.k - 1 - j, t)?.broadcast_mul(&w.narrow(1, j, 1)?.squeeze(1)?)?;
            acc = Some(match acc {
                None => tap,
                Some(a) => (a + tap)?,
            });
        }
        x + acc.unwrap().broadcast_mul(&self.gain)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_core::Device;

    #[test]
    fn starts_as_no_change_then_delays_and_learns() {
        let (vm, vb) = vm();
        let d = Delays::new(vb, 2, 8, 0.3).unwrap();
        let x = randn(&[1, 12, 2], 3);
        assert!(max_diff(&d.forward(&x).unwrap(), &x) < 1e-6);

        // Gain 1 and a delay of 3 on channel 0: an impulse at t = 0 comes back at t = 3.
        set(&vm, "center", &Tensor::new(&[3f32, 0.0], &Device::Cpu).unwrap());
        set(&vm, "gain", &Tensor::new(&[1f32, 0.0], &Device::Cpu).unwrap());
        let mut imp = vec![0f32; 24];
        imp[0] = 1.0;
        let y = d.forward(&Tensor::from_vec(imp, (1, 12, 2), &Device::Cpu).unwrap()).unwrap();
        let ch0: Vec<f32> = y.squeeze(0).unwrap().narrow(1, 0, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert!((ch0[0] - 1.0).abs() < 1e-3 && ch0[3] > 0.98 && ch0[2] < 0.01 && ch0[4] < 0.01, "{ch0:?}");

        // Causal: changing the input at t = 6 never changes the output before t = 6.
        let mut x2 = x.to_vec3::<f32>().unwrap();
        x2[0][6][0] += 5.0;
        let x2 = Tensor::new(x2, &Device::Cpu).unwrap();
        let (a, b) = (d.forward(&x).unwrap().narrow(1, 0, 6).unwrap(), d.forward(&x2).unwrap().narrow(1, 0, 6).unwrap());
        assert!(max_diff(&a, &b) < 1e-6);

        // The delay itself gets a gradient, so it can be learned.
        set(&vm, "center", &Tensor::new(&[2.6f32, 1.3], &Device::Cpu).unwrap());
        set(&vm, "gain", &Tensor::new(&[0.5f32, 0.5], &Device::Cpu).unwrap());
        let sigma_wide = Delays { sigma: 1.0, ..d };
        let loss = (sigma_wide.forward(&x).unwrap() * &x.affine(1.0, 0.3).unwrap()).unwrap().sum_all().unwrap();
        let g = grad_norms(&vm, &loss, &["center", "gain"]);
        assert!(g[0] > 1e-4 && g[1] > 1e-4, "{g:?}");
    }
}
