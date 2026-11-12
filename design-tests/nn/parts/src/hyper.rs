// TESTING / DESIGN ONLY
//! Weights made by a smaller network.
//!
//! H4, the genomic bottleneck (Shuvaev, Lachi, Koulakov & Zador 2024). A genome is far too small to list
//! every synapse; it holds rules that grow them. Here a small network G gives the weight between neuron i
//! of one layer and neuron j of the next from codes of their places in the layer:
//!   W[i, j] = G(code(u_i), code(u_j)),  u = (i + ½)/n,  code(u) = [sin 2πfu, cos 2πfu]_{f=1..F}
//! The genome has far fewer numbers than the matrix, and it can make the matrix at any width, which gives
//! a grown model (A15) a prior instead of noise.
//!
//! M6, writing to weights in one pass (like Doc-to-LoRA, 2026). Sleep replays a day many times; some
//! things are learned at once. A writer turns a summary s of a day (or a document) into a low-rank change
//! of a weight, with no gradient steps at write time:
//!   ΔW(s) = A(s)·B,  A(s) = reshape(s·W_a, [d, r]),  layer output h + h·ΔW(s)
//! W_a starts at 0, so nothing is written until the writer has been trained.

use crate::linear;
use candle_core::{Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

pub struct Genome {
    pub freqs: usize,
    pub w1: Tensor,
    pub b1: Tensor,
    pub w2: Tensor,
    pub b2: Tensor,
}

fn codes(n: usize, freqs: usize, dev: &candle_core::Device) -> Result<Tensor> {
    let mut v = Vec::with_capacity(n * 2 * freqs);
    for i in 0..n {
        let u = (i as f64 + 0.5) / n as f64;
        for f in 1..=freqs {
            let a = std::f64::consts::TAU * f as f64 * u;
            v.push(a.sin() as f32);
            v.push(a.cos() as f32);
        }
    }
    Tensor::from_vec(v, (n, 2 * freqs), dev)
}

impl Genome {
    pub fn new(vb: VarBuilder, freqs: usize, hidden: usize) -> Result<Genome> {
        let i = 4 * freqs;
        Ok(Genome {
            freqs,
            w1: vb.get_with_hints((i, hidden), "w1", Init::Randn { mean: 0.0, stdev: 1.0 / (i as f64).sqrt() })?,
            b1: vb.get_with_hints(hidden, "b1", Init::Const(0.0))?,
            w2: vb.get_with_hints((hidden, 1), "w2", Init::Randn { mean: 0.0, stdev: 1.0 / (hidden as f64).sqrt() })?,
            b2: vb.get_with_hints(1, "b2", Init::Const(0.0))?,
        })
    }

    /// The weight matrix [n_in, n_out] at any width.
    pub fn weights(&self, n_in: usize, n_out: usize) -> Result<Tensor> {
        let dev = self.w1.device();
        let ci = codes(n_in, self.freqs, dev)?.unsqueeze(1)?.broadcast_as((n_in, n_out, 2 * self.freqs))?;
        let co = codes(n_out, self.freqs, dev)?.unsqueeze(0)?.broadcast_as((n_in, n_out, 2 * self.freqs))?;
        let x = Tensor::cat(&[ci, co], D::Minus1)?;
        let h = linear(&x, &self.w1)?.broadcast_add(&self.b1)?.tanh()?;
        linear(&h, &self.w2)?.broadcast_add(&self.b2)?.squeeze(D::Minus1)
    }

    pub fn size(&self) -> usize {
        self.w1.elem_count() + self.b1.elem_count() + self.w2.elem_count() + self.b2.elem_count()
    }
}

pub struct Writer {
    pub d: usize,
    pub rank: usize,
    pub wa: Tensor,
    pub b: Tensor,
}

impl Writer {
    pub fn new(vb: VarBuilder, d_s: usize, d: usize, rank: usize) -> Result<Writer> {
        Ok(Writer {
            d,
            rank,
            wa: vb.get_with_hints((d_s, d * rank), "wa", Init::Const(0.0))?,
            b: vb.get_with_hints((rank, d), "b", Init::Randn { mean: 0.0, stdev: 1.0 / (rank as f64).sqrt() })?,
        })
    }

    /// ΔW [N, d, d] for summaries s [N, d_s].
    pub fn delta(&self, s: &Tensor) -> Result<Tensor> {
        let n = s.dim(0)?;
        let a = linear(s, &self.wa)?.reshape((n, self.d, self.rank))?;
        a.broadcast_matmul(&self.b.unsqueeze(0)?.broadcast_as((n, self.rank, self.d))?.contiguous()?)
    }

    /// h [N, T, d] → h + h·ΔW(s), each sequence with its own written change.
    pub fn apply(&self, h: &Tensor, s: &Tensor) -> Result<Tensor> {
        h + h.matmul(&self.delta(s)?)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_core::Device;
    use candle_nn::{AdamW, Optimizer, ParamsAdamW};

    fn band(n_in: usize, n_out: usize) -> Tensor {
        let mut v = Vec::with_capacity(n_in * n_out);
        for i in 0..n_in {
            for j in 0..n_out {
                let (a, b) = ((i as f64 + 0.5) / n_in as f64, (j as f64 + 0.5) / n_out as f64);
                v.push((-(a - b).powi(2) / 0.02).exp() as f32);
            }
        }
        Tensor::from_vec(v, (n_in, n_out), &Device::Cpu).unwrap()
    }

    #[test]
    fn a_small_genome_grows_the_weights_at_any_width() {
        let (vm, vb) = vm();
        let g = Genome::new(vb, 4, 32).unwrap();
        seed(&vm, &[("w1", 0.25), ("w2", 1.0 / 32f64.sqrt())], 31);
        let target = band(32, 32);
        let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 0.01, weight_decay: 0.0, ..Default::default() }).unwrap();
        for _ in 0..600 {
            let loss = (g.weights(32, 32).unwrap() - &target).unwrap().sqr().unwrap().mean_all().unwrap();
            opt.backward_step(&loss).unwrap();
        }
        assert!(g.size() < 32 * 32, "genome {} numbers", g.size());
        let rel = |w: &Tensor, t: &Tensor| {
            let e = (w - t).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            (e / t.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap()).sqrt()
        };
        let fit = rel(&g.weights(32, 32).unwrap(), &target);
        // Grown at twice the width, never trained there.
        let grown = rel(&g.weights(64, 64).unwrap(), &band(64, 64));
        eprintln!("genome {} numbers for a 32×32 matrix: fit {fit:.3}, at 64×64 {grown:.3}", g.size());
        assert!(fit < 0.1 && grown < 0.15, "{fit} {grown}");
    }

    #[test]
    fn the_writer_learns_to_write_a_change_in_one_pass() {
        let (d, r, d_s) = (8, 3, 12);
        let u = randn(&[d, r], 1);
        let v = (randn(&[r, d], 2) * 0.5).unwrap();
        let e = randn(&[r, d_s], 3);
        // Each "document" defines its own linear change U·diag(α)·V; the writer sees only s = α·E.
        let target = |alpha: &Tensor| -> Tensor {
            let n = alpha.dim(0).unwrap();
            let ua = u.unsqueeze(0).unwrap().broadcast_mul(&alpha.unsqueeze(1).unwrap()).unwrap();
            ua.matmul(&v.unsqueeze(0).unwrap().broadcast_as((n, r, d)).unwrap().contiguous().unwrap()).unwrap()
        };
        let (vm, vb) = vm();
        let w = Writer::new(vb, d_s, d, r).unwrap();
        seed(&vm, &[("b", 1.0 / (r as f64).sqrt())], 21);
        let x = randn(&[16, 5, d], 4);
        let s0 = randn(&[16, r], 5);
        assert!(max_diff(&w.apply(&x, &s0.matmul(&e).unwrap()).unwrap(), &x) < 1e-7, "writes nothing at the start");
        let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 0.02, weight_decay: 0.0, ..Default::default() }).unwrap();
        let loss_on = |alpha: &Tensor| {
            let s = alpha.matmul(&e).unwrap();
            let want = x.matmul(&target(alpha)).unwrap();
            (x.matmul(&w.delta(&s).unwrap()).unwrap() - &want).unwrap().sqr().unwrap().mean_all().unwrap()
        };
        for step in 0..500 {
            let alpha = randn(&[16, r], 100 + step);
            opt.backward_step(&loss_on(&alpha)).unwrap();
        }
        // New documents, never seen: one pass writes the right change.
        let alpha = randn(&[16, r], 9999);
        let want = x.matmul(&target(&alpha)).unwrap();
        let err = loss_on(&alpha).to_scalar::<f32>().unwrap() / want.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(err < 0.05, "relative error {err}");
    }
}
