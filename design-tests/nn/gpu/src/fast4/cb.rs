// TESTING / DESIGN ONLY — ava_gpu::fast4::cb: the cerebellum's online learning step for one row (`Model::cb_learn` on
// the fused step's logits), in place, with the same bits:
// * p = candle_nn's softmax as candle runs it on the CPU: the max as its reduce keeps it (the first value, replaced
//   by each larger one), each value minus it, exp, their sum from 0 in order, each value divided by the sum;
// * err = p - onehot(next) (p - 0 and p_y - 1), delta = err · E (E the tied embedding [V, d]; gemm's column-major
//   gemv over V, `fmm::xw`, its outputs split over threads);
// * the table: the row's scale fades by 2^(-1 / half) (folded below 1e-18), then for each unit of the kept code
//   s = lr · x (÷ scale), w -= s · delta, as `cb_learn` writes it.
// The forward's part (the table's share of the correction and the kept code) runs in the fused step itself
// (src/model/fstep/arch_f3.rs `cb`). Tests: fast4/cb_tests.rs.

use crate::model::Model;
use candle_core::Result;

impl Model {
    /// `cb_learn` for one row from the logits' values [V] (the fused step's logits, or any one row's): the same table,
    /// scale and update count bit for bit, and the same returned nats (the row's -ln p of `next`; NaN for IGNORE).
    pub fn fast4_cb_learn(&self, logits: &[f32], next: u32) -> Result<f64> {
        Ok(self.fast4_cb_learn_sized(logits, next)?.0)
    }

    /// v52 (the chat app's live learning): `fast4_cb_learn`, also returning the size of the change it made, on the
    /// side: the rms over the whole table [G, d] of the step as it acts (each winner u's row moves by lr x_u delta), 0 for
    /// IGNORE. The same table, scale and returned nats bit for bit.
    pub fn fast4_cb_learn_sized(&self, logits: &[f32], next: u32) -> Result<(f64, f64)> {
        let Some(e) = self.v51b() else { candle_core::bail!("cb_learn: no block options") };
        let mut guard = e.cb_online.borrow_mut();
        let Some(on) = guard.as_mut() else { candle_core::bail!("cb_learn: no online table (Model::cb_online_begin)") };
        if on.last.len() != 1 || on.rows != 1 {
            candle_core::bail!("fast4_cb_learn: one row ({} rows, {} kept codes)", on.rows, on.last.len());
        }
        let (d, v) = (self.cfg.d, logits.len());
        if v != self.tok.dim(0)? {
            candle_core::bail!("fast4_cb_learn: {v} logits for a vocabulary of {}", self.tok.dim(0)?);
        }
        if next == u32::MAX {
            return Ok((f64::NAN, 0.0));
        }
        // the softmax (candle_nn::ops::softmax: max_keepdim, broadcast_sub, exp, sum_keepdim, broadcast_div)
        let mut mx = logits[0];
        for &z in logits {
            if mx < z {
                mx = z;
            }
        }
        let mut p: Vec<f32> = logits.iter().map(|&z| (z - mx).exp()).collect();
        let mut den = 0f32;
        for &x in &p {
            den += x;
        }
        p.iter_mut().for_each(|x| *x = *x / den);
        let y = next as usize;
        let ce = -(p[y].max(1e-30) as f64).ln();
        // err = p - onehot, then err · E
        p.iter_mut().enumerate().for_each(|(j, x)| *x = *x - if j == y { 1.0 } else { 0.0 });
        let delta = crate::fast4::with_f32(&self.tok, |emb| crate::fast4::fmm::xw(&p, emb, d, candle_core::utils::get_num_threads()))?;
        let g = e.cb.as_ref().map(|c| c.g).unwrap_or(0);
        let tab = &mut on.delta[0];
        if tab.is_empty() {
            tab.resize(g * d, 0.0);
        }
        if on.half[0].is_finite() {
            on.scale[0] *= (-1.0 / on.half[0]).exp2();
            if on.scale[0] < crate::v51b::CB_SCALE_FOLD {
                let sc = on.scale[0];
                tab.iter_mut().for_each(|v| *v = (*v as f64 * sc) as f32);
                on.scale[0] = 1.0;
            }
        }
        let sc = on.scale[0];
        let mut s2 = 0f64; // v52: the step's size as it acts, on the side
        for &(u, x) in &on.last[0] {
            let w = &mut tab[u as usize * d..(u as usize + 1) * d];
            let s = on.lrs[0] * x;
            s2 += (s as f64) * (s as f64);
            let s = if sc == 1.0 { s } else { (s as f64 / sc) as f32 };
            for (wi, di) in w.iter_mut().zip(&delta) {
                *wi -= s * di;
            }
        }
        on.updates[0] += 1;
        let d2: f64 = delta.iter().map(|&v| (v as f64) * (v as f64)).sum();
        Ok((ce, (s2 * d2 / tab.len().max(1) as f64).sqrt()))
    }
}
