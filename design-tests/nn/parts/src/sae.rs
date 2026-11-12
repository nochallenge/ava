// TESTING / DESIGN ONLY
//! A13: a sparse autoencoder to see what her neurons mean. TopK SAEs (Gao et al., "Scaling and evaluating
//! sparse autoencoders", 2024) take a layer's activity and find a larger set of directions (features),
//! with only k of them active at a time, that rebuilds it. Each feature tends to mean one thing. This lets
//! us read her mind, and later check what a memory or a skill is made of.
//!
//!   z = TopK(relu((x − b_dec)·W_enc + b_enc)),   x̂ = z·D + b_dec,   D's rows kept at unit length
//! Trained on saved activity with AdamW on the reconstruction error.

use crate::linear;
use candle_core::{DType, Device, Result, Tensor, Var, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

pub struct Sae {
    pub enc: Var,
    pub b_enc: Var,
    pub dec: Var,
    pub b_dec: Var,
    pub k: usize,
}

impl Sae {
    /// d inputs, m features. The decoder starts as the encoder's transpose (a common, stable start).
    pub fn new(d: usize, m: usize, k: usize, seed: u64, dev: &Device) -> Result<Sae> {
        let enc = (crate::randn_seeded(d * m, seed, dev)?.reshape((d, m))? / (d as f64).sqrt())?;
        let dec = enc.t()?.contiguous()?;
        Ok(Sae {
            enc: Var::from_tensor(&enc)?,
            b_enc: Var::zeros(m, DType::F32, dev)?,
            dec: Var::from_tensor(&dec)?,
            b_dec: Var::zeros(d, DType::F32, dev)?,
            k,
        })
    }

    fn unit_dec(&self) -> Result<Tensor> {
        let d = self.dec.as_tensor();
        d.broadcast_div(&(d.sqr()?.sum_keepdim(1)? + 1e-12)?.sqrt()?)
    }

    /// The sparse code [n, m]: k non-zero values per row.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let pre = linear(&x.broadcast_sub(self.b_dec.as_tensor())?, self.enc.as_tensor())?.broadcast_add(self.b_enc.as_tensor())?.relu()?;
        let kth = pre.sort_last_dim(false)?.0.narrow(D::Minus1, self.k - 1, 1)?;
        let keep = pre.broadcast_ge(&kth)?.to_dtype(pre.dtype())?.detach();
        pre * keep
    }

    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        linear(z, &self.unit_dec()?)?.broadcast_add(self.b_dec.as_tensor())
    }

    /// Mean squared reconstruction error.
    pub fn loss(&self, x: &Tensor) -> Result<Tensor> {
        (self.decode(&self.encode(x)?)? - x)?.sqr()?.mean_all()
    }

    /// Full-batch AdamW steps. Returns the loss before and after.
    pub fn train(&self, x: &Tensor, steps: usize, lr: f64) -> Result<(f32, f32)> {
        let vars = vec![self.enc.clone(), self.b_enc.clone(), self.dec.clone(), self.b_dec.clone()];
        let mut opt = AdamW::new(vars, ParamsAdamW { lr, weight_decay: 0.0, ..Default::default() })?;
        let first = self.loss(x)?.to_scalar::<f32>()?;
        for _ in 0..steps {
            opt.backward_step(&self.loss(x)?)?;
        }
        Ok((first, self.loss(x)?.to_scalar::<f32>()?))
    }

    /// The decoder's unit-length feature directions [m, d].
    pub fn features(&self) -> Result<Vec<Vec<f32>>> {
        self.unit_dec()?.to_vec2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_planted_features() {
        let (d, f, m, n) = (16, 12, 24, 2048);
        let dev = Device::Cpu;
        // 12 true feature directions; each sample mixes 2 of them with positive strengths.
        let truth: Vec<Vec<f32>> = (0..f)
            .map(|i| {
                let v = crate::testkit::randn(&[d], 100 + i as u64).to_vec1::<f32>().unwrap();
                let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / nrm).collect()
            })
            .collect();
        let pick = crate::testkit::randn(&[n * 4], 7).to_vec1::<f32>().unwrap();
        let mut x = vec![0f32; n * d];
        for s in 0..n {
            let a = ((pick[4 * s].abs() * 1000.0) as usize) % f;
            let b = (a + 1 + ((pick[4 * s + 1].abs() * 1000.0) as usize) % (f - 1)) % f;
            let (ca, cb) = (0.5 + pick[4 * s + 2].abs().min(1.0), 0.5 + pick[4 * s + 3].abs().min(1.0));
            for k in 0..d {
                x[s * d + k] = ca * truth[a][k] + cb * truth[b][k];
            }
        }
        let x = Tensor::from_vec(x, (n, d), &dev).unwrap();
        let sae = Sae::new(d, m, 2, 5, &dev).unwrap();
        let (first, last) = sae.train(&x, 800, 3e-3).unwrap();
        assert!(last < 0.1 * first, "{first} -> {last}");
        let feats = sae.features().unwrap();
        let found = truth
            .iter()
            .filter(|t| feats.iter().map(|g| g.iter().zip(t.iter()).map(|(a, b)| a * b).sum::<f32>()).fold(f32::MIN, f32::max) > 0.9)
            .count();
        assert!(found >= 10, "found {found} of {f}");
        // Codes are sparse: at most k active per row.
        let z = sae.encode(&x.narrow(0, 0, 50).unwrap()).unwrap().to_vec2::<f32>().unwrap();
        assert!(z.iter().all(|r| r.iter().filter(|&&v| v > 0.0).count() <= 2));
    }
}
