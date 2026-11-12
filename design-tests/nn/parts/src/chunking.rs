// TESTING / DESIGN ONLY
//! A14: learned chunking instead of a fixed tokenizer (dynamic chunking from H-Net, Hwang, Wang & Gu
//! 2025). Readers do not see letters one by one; they group them into words and phrases where the
//! meaning shifts. A router marks a boundary where the next piece looks unlike the last one:
//!   q_t = h_t·W_q,  k_t = h_t·W_k,  p_t = ½(1 − cos(q_t, k_{t−1})),  p_0 = 1,  boundary b_t = [p_t ≥ ½]
//! The boundary pieces go up to the slower, deeper level; on the way back each chunk's output is
//! smoothed (z̄_c = P_c·z_c + (1 − P_c)·z̄_{c−1}, P_c the chunk's boundary probability) and every
//! position takes its chunk's value, times a confidence that is 1 in the forward pass but carries the
//! router's gradient (c_t = p_t if a boundary else 1 − p_t, used as c + (1 − c).detach()).
//! A ratio loss keeps about one boundary per N positions:
//!   L = N/(N−1) · ((N−1)·F·G + (1−F)·(1−G)),  F = share of boundaries, G = mean p  (1 at F = G = 1/N)
//! W_q and W_k start as the identity (H-Net's choice). Unlike the other parts this is not a no-op: a
//! chunked model is a new architecture, for a new run only.

use crate::linear;
use candle_core::{Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

pub struct Router {
    pub wq: Tensor,
    pub wk: Tensor,
}

pub struct Chunks {
    /// [T] boundary probabilities.
    pub p: Tensor,
    /// The boundary positions.
    pub starts: Vec<usize>,
}

fn unit(x: &Tensor) -> Result<Tensor> {
    x.broadcast_div(&(x.sqr()?.sum_keepdim(D::Minus1)? + 1e-12)?.sqrt()?)
}

impl Router {
    pub fn new(vb: VarBuilder, d: usize) -> Result<Router> {
        let eye = Tensor::eye(d, candle_core::DType::F32, vb.device())?;
        let wq = vb.get_with_hints((d, d), "wq", Init::Const(0.0))?;
        let wk = vb.get_with_hints((d, d), "wk", Init::Const(0.0))?;
        // Identity start: the hints only allow constants, so the identity is added here.
        Ok(Router { wq: (wq + &eye)?, wk: (wk + eye)? })
    }

    /// One sequence h [T, d].
    pub fn route(&self, h: &Tensor) -> Result<Chunks> {
        let t = h.dim(0)?;
        let q = unit(&linear(h, &self.wq)?)?;
        let k = unit(&linear(h, &self.wk)?)?;
        let cos = (q.narrow(0, 1, t - 1)? * k.narrow(0, 0, t - 1)?)?.sum(D::Minus1)?;
        let p_rest = cos.affine(-0.5, 0.5)?;
        let p = Tensor::cat(&[Tensor::ones(1, h.dtype(), h.device())?, p_rest], 0)?;
        let pv = p.to_vec1::<f32>()?;
        let starts = (0..t).filter(|&i| pv[i] >= 0.5).collect();
        Ok(Chunks { p, starts })
    }
}

impl Chunks {
    /// The boundary rows of h [T, d] → [C, d], for the slower level.
    pub fn down(&self, h: &Tensor) -> Result<Tensor> {
        let idx = Tensor::from_vec(self.starts.iter().map(|&i| i as u32).collect::<Vec<u32>>(), self.starts.len(), h.device())?;
        h.index_select(&idx, 0)
    }

    /// The slower level's outputs z [C, d] back to every position [T, d].
    pub fn up(&self, z: &Tensor) -> Result<Tensor> {
        let t = self.p.dim(0)?;
        let mut smooth: Vec<Tensor> = Vec::with_capacity(self.starts.len());
        for (c, &s) in self.starts.iter().enumerate() {
            let pc = self.p.narrow(0, s, 1)?;
            let zc = z.narrow(0, c, 1)?;
            let v = match smooth.last() {
                None => zc,
                Some(prev) => (zc.broadcast_mul(&pc)? + prev.broadcast_mul(&pc.affine(-1.0, 1.0)?)?)?,
            };
            smooth.push(v);
        }
        let pv = self.p.to_vec1::<f32>()?;
        let mut rows = Vec::with_capacity(t);
        let mut c = 0usize;
        for i in 0..t {
            if c + 1 < self.starts.len() && self.starts[c + 1] == i {
                c += 1;
            }
            let pi = self.p.narrow(0, i, 1)?;
            let conf = if pv[i] >= 0.5 { pi } else { pi.affine(-1.0, 1.0)? };
            let ste = (conf.affine(-1.0, 1.0)?.detach() + conf)?;
            rows.push(smooth[c].broadcast_mul(&ste)?);
        }
        Tensor::cat(&rows, 0)
    }

    pub fn ratio_loss(&self, n: f64) -> Result<Tensor> {
        let t = self.p.dim(0)? as f64;
        let f = self.starts.len() as f64 / t;
        let g = self.p.mean_all()?;
        let a = (g.clone() * ((n - 1.0) * f))?;
        let b = (g.affine(-1.0, 1.0)? * (1.0 - f))?;
        (a + b)? * (n / (n - 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_core::Device;

    #[test]
    fn boundaries_fall_where_the_content_changes() {
        let d = 8;
        // Five "words" of lengths 3, 5, 2, 4, 3; word c is e_c − e_{c−1}, so neighbours have cosine −½.
        let lens = [3usize, 5, 2, 4, 3];
        let mut rows = Vec::new();
        let mut want = Vec::new();
        for (c, &l) in lens.iter().enumerate() {
            want.push(rows.len() / d);
            let mut v = vec![0f32; d];
            v[c + 1] = 1.0;
            v[c] -= 1.0;
            for _ in 0..l {
                rows.extend_from_slice(&v);
            }
        }
        let t = rows.len() / d;
        let h = Tensor::from_vec(rows, (t, d), &Device::Cpu).unwrap();
        let (vm, vb) = vm();
        let r = Router::new(vb, d).unwrap();
        let ch = r.route(&h).unwrap();
        assert_eq!(ch.starts, want);
        // Up after down (with an identity slower level) gives each position its smoothed chunk value.
        let up = ch.up(&ch.down(&h).unwrap()).unwrap();
        assert_eq!(up.dims(), &[t, d]);
        assert!(max_diff(&up.narrow(0, 0, 3).unwrap(), &h.narrow(0, 0, 3).unwrap()) < 1e-6);
        // The router learns through the confidence and the ratio loss.
        let loss = (up.sqr().unwrap().mean_all().unwrap() + ch.ratio_loss(4.0).unwrap()).unwrap();
        let g = grad_norms(&vm, &loss, &["wq", "wk"]);
        assert!(g[0] > 1e-6 && g[1] > 1e-6, "{g:?}");
    }

    #[test]
    fn ratio_loss_is_one_at_the_target_rate() {
        let p = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 8, &Device::Cpu).unwrap();
        let at = Chunks { p, starts: vec![0, 4] }.ratio_loss(4.0).unwrap().to_scalar::<f32>().unwrap();
        assert!((at - 1.0).abs() < 1e-6, "{at}");
        let p = Tensor::ones(8, candle_core::DType::F32, &Device::Cpu).unwrap();
        let all = Chunks { p, starts: (0..8).collect() }.ratio_loss(4.0).unwrap().to_scalar::<f32>().unwrap();
        assert!(all > 1.3, "{all}");
    }
}
