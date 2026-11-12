// TESTING / DESIGN ONLY
//! B9 gated working memory and B3 a global workspace: a few slots that hold content across time, with gates.
//!
//! - B9 (PBWM, O'Reilly & Frank 2006): the basal ganglia gate what enters prefrontal working memory. Each
//!   slot has its own input gate g = sigmoid(h·Wg + bg). A closed gate holds the slot's content
//!   unchanged however long the gap, unlike a decaying trace:
//!     slot_k ← (1 − g_k)·slot_k + g_k·v_k,  v_k = (h·Wv)_k
//! - B3 (global workspace; Baars; Dehaene's ignition; Goyal et al., ICLR 2022): the same slots, but they
//!   compete. Per token, the write gates are a softmax over slot salience, times an overall write strength,
//!   so the slots share a limited budget (capacity is the bottleneck), and only the top `k` slots may write
//!   at all.
//!
//! Every token reads the slots by attention (query h·Wq) and adds r·Wo to its stream. Wo starts at 0, so
//! the part starts as no change. The scan is a gated linear recurrence; this is the exact step form.

use crate::linear;
use candle_core::{Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gating {
    /// B9: an independent sigmoid gate per slot.
    Independent,
    /// B3: slots compete (softmax over salience, only the top k write).
    Compete { k: usize },
}

pub struct SlotMemory {
    pub wg: Tensor,
    pub bg: Tensor,
    pub wv: Tensor,
    pub wq: Tensor,
    pub wo: Tensor,
    /// For `Compete`: [d, 1] overall write strength.
    pub ws: Tensor,
    pub slots: usize,
    pub dk: usize,
    pub gating: Gating,
}

impl SlotMemory {
    pub fn new(vb: VarBuilder, d: usize, slots: usize, dk: usize, gating: Gating) -> Result<SlotMemory> {
        let std = (1.0 / d as f64).sqrt();
        let r = |n: &str, s: (usize, usize)| vb.get_with_hints(s, n, Init::Randn { mean: 0.0, stdev: std });
        Ok(SlotMemory {
            wg: r("wg", (d, slots))?,
            // Gates start mostly closed (sigmoid(-2) = 0.12): content is kept unless a token opens them.
            bg: vb.get_with_hints(slots, "bg", Init::Const(-2.0))?,
            wv: r("wv", (d, slots * dk))?,
            wq: r("wq", (d, dk))?,
            wo: vb.get_with_hints((dk, d), "wo", Init::Const(0.0))?,
            ws: r("ws", (d, 1))?,
            slots,
            dk,
            gating,
        })
    }

    /// The write gates [B, T, K] for h [B, T, d].
    pub fn gates(&self, h: &Tensor) -> Result<Tensor> {
        let sal = linear(h, &self.wg)?.broadcast_add(&self.bg)?;
        match self.gating {
            Gating::Independent => candle_nn::ops::sigmoid(&sal),
            Gating::Compete { k } => {
                let k = k.min(self.slots).max(1);
                // Only the top k slots by salience may write (ties: all at the k-th value).
                let kth = sal.sort_last_dim(false)?.0.narrow(D::Minus1, k - 1, 1)?;
                let keep = sal.broadcast_ge(&kth)?.to_dtype(sal.dtype())?;
                let masked = (sal.broadcast_mul(&keep)? + ((keep.affine(-1.0, 1.0)?) * -1e9)?)?;
                let share = candle_nn::ops::softmax(&masked, D::Minus1)?;
                share.broadcast_mul(&candle_nn::ops::sigmoid(&linear(h, &self.ws)?)?)
            }
        }
    }

    /// Slot contents after each token, [B, T, K, dk], from gates [B, T, K] and values [B, T, K, dk].
    pub fn scan(gates: &Tensor, values: &Tensor) -> Result<Tensor> {
        let (b, t, k, dk) = values.dims4()?;
        let mut m = Tensor::zeros((b, k, dk), values.dtype(), values.device())?;
        let mut out = Vec::with_capacity(t);
        for i in 0..t {
            let g = gates.narrow(1, i, 1)?.squeeze(1)?.unsqueeze(2)?;
            let v = values.narrow(1, i, 1)?.squeeze(1)?;
            m = (m.broadcast_mul(&g.affine(-1.0, 1.0)?)? + v.broadcast_mul(&g)?)?;
            out.push(m.clone());
        }
        Tensor::stack(&out, 1)
    }

    /// What each token adds to its stream, [B, T, d].
    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let (b, t, _) = h.dims3()?;
        let g = self.gates(h)?;
        let v = linear(h, &self.wv)?.reshape((b, t, self.slots, self.dk))?;
        let m = Self::scan(&g, &v)?;
        let q = linear(h, &self.wq)?.unsqueeze(2)?;
        let att = (m.broadcast_mul(&q)?.sum(D::Minus1)? / (self.dk as f64).sqrt())?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?.unsqueeze(3)?;
        let r = m.broadcast_mul(&att)?.sum(2)?;
        linear(&r, &self.wo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_core::Device;

    #[test]
    fn a_closed_gate_holds_content_exactly() {
        // Slot 0 opens at t = 0 only; slot 1 is half open every step.
        let g = Tensor::new(&[[[1f32, 0.5], [0.0, 0.5], [0.0, 0.5], [0.0, 0.5]]], &Device::Cpu).unwrap();
        let v = randn(&[1, 4, 2, 3], 5);
        let m = SlotMemory::scan(&g, &v).unwrap();
        let held = m.narrow(1, 3, 1).unwrap().narrow(2, 0, 1).unwrap().flatten_all().unwrap();
        let first = v.narrow(1, 0, 1).unwrap().narrow(2, 0, 1).unwrap().flatten_all().unwrap();
        assert!(max_diff(&held, &first) < 1e-7);
        // The open slot is a running average that favours recent values.
        let s1 = m.narrow(1, 3, 1).unwrap().narrow(2, 1, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vv = v.narrow(2, 1, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for c in 0..3 {
            let want = 0.5 * vv[9 + c] + 0.25 * vv[6 + c] + 0.125 * vv[3 + c] + 0.0625 * vv[c];
            assert!((s1[c] - want).abs() < 1e-6);
        }
    }

    #[test]
    fn starts_as_no_change_and_learns_both_kinds() {
        for gating in [Gating::Independent, Gating::Compete { k: 2 }] {
            let (vm, vb) = vm();
            let wm = SlotMemory::new(vb, 8, 4, 4, gating).unwrap();
            let h = randn(&[2, 6, 8], 6);
            let y = wm.forward(&h).unwrap();
            assert_eq!(y.dims(), &[2, 6, 8]);
            assert!(max_diff(&y, &y.zeros_like().unwrap()) < 1e-9);
            if let Gating::Compete { .. } = gating {
                // At most 2 of the 4 slots write per token, and the shares are a limited budget (sum <= 1).
                let g = wm.gates(&h).unwrap().flatten_to(1).unwrap().to_vec2::<f32>().unwrap();
                for row in &g {
                    assert!(row.iter().filter(|&&x| x > 1e-6).count() <= 2 && row.iter().sum::<f32>() <= 1.0 + 1e-5, "{row:?}");
                }
            }
            set(&vm, "wo", &randn(&[4, 8], 7));
            let loss = wm.forward(&h).unwrap().sqr().unwrap().sum_all().unwrap();
            let gn = grad_norms(&vm, &loss, &["wg", "wv", "wq", "wo"]);
            assert!(gn.iter().all(|&x| x > 1e-6), "{gating:?} {gn:?}");
        }
    }
}
