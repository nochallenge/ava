// TESTING / DESIGN ONLY
//! H11: a cerebellum. The cerebellum holds most of the brain's neurons. It is wide and shallow:
//! - mossy fibres fan out onto a huge number of granule cells, through fixed, sparse, random wiring;
//! - only a few granule cells are active at once (a sparse, high-dimensional code);
//! - Purkinje cells learn a linear readout from that code, taught by error signals (climbing fibres).
//! Marr-Albus theory: this makes it a fast, stable learner of corrections that cortex then uses, for
//! timing, prediction, and (in newer work) thought as well as movement.
//!
//! Here: the input h [.., d] goes through a fixed random expansion to G granule units, which are sparse
//! (each takes `fan_in` random inputs) and fixed by a seed, so the weights need not be saved. Then kWTA
//! keeps the top `k`, and a learned readout [G, d] gives a correction added to the stream. The readout
//! starts at 0 (no change). The expansion is not trained; only the readout learns, which keeps it fast
//! and its features stable.

use candle_core::{Device, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

pub struct Cerebellum {
    /// [d, G], fixed: each granule column has `fan_in` non-zero entries of ±1/sqrt(fan_in).
    pub expand: Tensor,
    /// [G, d], learned, starts at 0.
    pub readout: Tensor,
    pub k: usize,
}

/// The fixed sparse expansion from a seed (the same seed always gives the same wiring).
pub fn expansion(d: usize, g: usize, fan_in: usize, seed: u64, dev: &Device) -> Result<Tensor> {
    let mut w = vec![0f32; d * g];
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let amp = 1.0 / (fan_in as f32).sqrt();
    for col in 0..g {
        let mut picked = Vec::with_capacity(fan_in);
        while picked.len() < fan_in.min(d) {
            let r = (next() % d as u64) as usize;
            if !picked.contains(&r) {
                picked.push(r);
            }
        }
        for r in picked {
            w[r * g + col] = if next() & 1 == 0 { amp } else { -amp };
        }
    }
    Tensor::from_vec(w, (d, g), dev)
}

impl Cerebellum {
    pub fn new(vb: VarBuilder, d: usize, granules: usize, fan_in: usize, k: usize, seed: u64) -> Result<Cerebellum> {
        let readout = vb.get_with_hints((granules, d), "readout", Init::Const(0.0))?;
        let expand = expansion(d, granules, fan_in, seed, readout.device())?;
        Ok(Cerebellum { expand, readout, k })
    }

    /// The sparse granule code [.., G]: the top k units' rectified values, the rest 0. It is detached:
    /// the expansion is fixed, and no gradient goes back into the stream through it.
    pub fn code(&self, h: &Tensor) -> Result<Tensor> {
        let u = crate::linear(&h.detach(), &self.expand)?.relu()?;
        let kth = u.sort_last_dim(false)?.0.narrow(D::Minus1, self.k - 1, 1)?;
        let keep = u.broadcast_ge(&kth)?.to_dtype(u.dtype())?;
        u * keep
    }

    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        crate::linear(&self.code(h)?, &self.readout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;

    #[test]
    fn sparse_code_starts_silent_and_learns_a_mapping_fast() {
        let (d, g, k) = (16, 512, 24);
        let (vm, vb) = vm();
        let cb = Cerebellum::new(vb, d, g, 4, k, 42).unwrap();
        // The same seed gives the same wiring.
        assert!(max_diff(&cb.expand, &expansion(d, g, 4, 42, &Device::Cpu).unwrap()) == 0.0);
        let x = randn(&[64, d], 31);
        let c = cb.code(&x).unwrap().to_vec2::<f32>().unwrap();
        for row in &c {
            let on = row.iter().filter(|&&v| v > 0.0).count();
            assert!(on <= k && on >= k - 2, "{on}");
        }
        let y0 = cb.forward(&x).unwrap();
        assert!(max_diff(&y0, &y0.zeros_like().unwrap()) == 0.0);

        // Plain gradient steps on the readout alone learn a fixed non-linear target well.
        let target = (x.sqr().unwrap() - 1.0).unwrap().tanh().unwrap();
        let readout = var(&vm, "readout");
        let lr = 0.5;
        let loss_of = |cb: &Cerebellum| (cb.forward(&x).unwrap() - &target).unwrap().sqr().unwrap().mean_all().unwrap();
        let first = loss_of(&cb).to_scalar::<f32>().unwrap();
        for _ in 0..200 {
            let loss = loss_of(&cb);
            let gr = loss.backward().unwrap();
            let gw = gr.get(readout.as_tensor()).unwrap();
            readout.set(&(readout.as_tensor() - (gw * lr).unwrap()).unwrap()).unwrap();
        }
        let last = loss_of(&cb).to_scalar::<f32>().unwrap();
        assert!(last < 0.4 * first, "{first} -> {last}");
    }
}
