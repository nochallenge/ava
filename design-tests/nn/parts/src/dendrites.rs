// TESTING / DESIGN ONLY
//! E6: active dendrites. A cortical neuron's distal dendrites detect patterns in their own input (often
//! top-down context), and a matching segment makes the cell fire more easily. In networks this lets one
//! unit take part in different tasks, depending on context, without the tasks overwriting each other
//! (Iyer et al., "Avoiding catastrophe: active dendrites", Frontiers 2022).
//!
//! Each output unit has `n` segments that read a context vector. The best-matching segment sets a gain:
//!   gate = 1 + a ⊙ tanh(max_j ctx · S_j)
//! The caller multiplies a layer's output (e.g. the MLP's hidden units) by `gate`. The context is the layer's
//! own input, or a top-down signal from a higher block. `a` starts at 0, so the gate starts at exactly 1.

use crate::linear;
use candle_core::{Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

pub struct Dendrites {
    /// [d_ctx, n·units]
    pub seg: Tensor,
    /// [units]
    pub a: Tensor,
    pub n: usize,
    pub units: usize,
}

impl Dendrites {
    pub fn new(vb: VarBuilder, d_ctx: usize, units: usize, n: usize) -> Result<Dendrites> {
        let std = (1.0 / d_ctx as f64).sqrt();
        let seg = vb.get_with_hints((d_ctx, n * units), "seg", Init::Randn { mean: 0.0, stdev: std })?;
        let a = vb.get_with_hints(units, "a", Init::Const(0.0))?;
        Ok(Dendrites { seg, a, n, units })
    }

    /// ctx [.., d_ctx] -> gate [.., units].
    pub fn gate(&self, ctx: &Tensor) -> Result<Tensor> {
        let mut dims = ctx.dims().to_vec();
        dims.pop();
        let mut shaped = dims.clone();
        shaped.extend([self.n, self.units]);
        let m = linear(ctx, &self.seg)?.reshape(shaped)?.max(D::Minus2)?;
        m.tanh()?.broadcast_mul(&self.a)? + 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_core::Device;

    #[test]
    fn starts_at_one_and_context_selects_units() {
        let (vm, vb) = vm();
        let den = Dendrites::new(vb, 4, 2, 3).unwrap();
        let ctx = randn(&[2, 5, 4], 7);
        let g = den.gate(&ctx).unwrap();
        assert_eq!(g.dims(), &[2, 5, 2]);
        assert!(max_diff(&g, &g.ones_like().unwrap()) < 1e-7);

        // Unit 0's segments all look for context A = e0 and unit 1's for context B = e1. With a = 1, context A
        // opens unit 0 more than unit 1, and B the other way round.
        let mut s = vec![0f32; 4 * 6];
        for j in 0..3 {
            s[j * 2] = 3.0; // row 0 (A) -> segment j of unit 0
            s[6 + j * 2 + 1] = 3.0; // row 1 (B) -> segment j of unit 1
        }
        set(&vm, "seg", &Tensor::from_vec(s, (4, 6), &Device::Cpu).unwrap());
        set(&vm, "a", &Tensor::new(&[1f32, 1.0], &Device::Cpu).unwrap());
        let ab = Tensor::new(&[[1f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], &Device::Cpu).unwrap();
        let g = den.gate(&ab).unwrap().to_vec2::<f32>().unwrap();
        assert!(g[0][0] > 1.9 && g[0][1] < 1.1 && g[1][1] > 1.9 && g[1][0] < 1.1, "{g:?}");

        // Both the segments and the gain learn.
        set(&vm, "a", &Tensor::new(&[0.5f32, 0.5], &Device::Cpu).unwrap());
        let loss = den.gate(&ctx).unwrap().sqr().unwrap().sum_all().unwrap();
        let gn = grad_norms(&vm, &loss, &["seg", "a"]);
        assert!(gn[0] > 1e-4 && gn[1] > 1e-4, "{gn:?}");
    }
}
