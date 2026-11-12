// TESTING / DESIGN ONLY — f4ext (runs/v52/f4ext/report.md): v51's P --dendrite, DA --dopamine and Y --rhythm in the
// fused one-token talking step (src/model/fstep/arch.rs), with the same bits as `forward_with_state` (int8 on, or f32
// with `Model::set_fast4`): every float operation the one candle performs on the CPU, in its order; the small products
// in gemm's one-row order (fstep.rs's `gemv`):
//   Y   before the blocks (v51::begin at t = 1, x the input after the letters): pre = head_norm(x) Wr + br (gemv into 2),
//       L = log_sigmoid(-pre), then v51::rhythm_scan at c = 1 op for op: the triangular product (K = 1) L·1, before =
//       L·1 - L, diff = L·1 - before, w = exp(min(diff, 0))·1 (candle's minimum: v > 0 ? 0 : v), a = w + exp(L·1)·a0 (a0
//       the State's time, 0 without one; the sum over one place is the value itself); the phases ph = a exp(lw), the
//       features f = [cos ph0, sin ph0, cos ph1, sin ph1] (v51::rhythm_feats' stack order); in every block the write
//       strength b (its 2 sigmoid + 0, before --neuromod's ACh clamp, as mem_arch gates it) times tanh(f Gw)·1 + 1 per
//       head, and the head-normed read times tanh(f Gr)·1 + 1 per head (before the output gate) (gemv into the heads)
//   DA  after block 0 the surprise z = s·(1/ln V) + 0 (arch_f3.rs `Mood::after0`, M's own, now also without --mood); in
//       every later block the MLP's output times tanh(z w + c)·1 + 1 per channel (v51::mlp), before --neuromod's DA
//   P   each block's memory read y (W_o's output, before TH's gate: v51::put_ctx) as the dendrites' context: the MLP's
//       units (after kWTA, relu and relu2) times tanh(head_norm(y) A B)·1 + 1 (gemv into 32, then into the hidden width)
// The State moves on only when the whole token is computed: Y's times with the rest (arch_f3.rs `commit`).
// Not here: W --columns (the blocker still refuses it). Tests: src/model/fstep/arch_f4_tests.rs.

use super::super::{gemv, head_norm, log_sigmoid, ro, vals};
use crate::model::Model;
use crate::state::State;
use crate::v51::{V51Ext, V51Opts};
use candle_core::Result;

/// The v51 options this step takes off the blocker's list (P, DA, Y); with arch_f2's R and arch_f3's M and Pr.
pub(super) fn strip(o: V51Opts) -> V51Opts {
    V51Opts { dendrite: false, dopamine: false, rhythm: false, ..o }
}

fn v51(m: &Model) -> Option<&V51Ext> {
    m.ext.as_ref().map(|e| &e.v51).filter(|v| v.on)
}

/// 1 + tanh(x) as v51::gain computes it (tanh, then affine(1, 1)).
fn gain(x: f32) -> f32 {
    x.tanh() * 1.0 + 1.0
}

/// Y: this token's rhythm features and the times the State keeps after it.
pub(super) struct Rhythm {
    /// [cos ph0, sin ph0, cos ph1, sin ph1]
    f: Vec<f32>,
    /// the time since each rhythm's last reset [2] (the State's `v51.rh` after this token)
    pub(super) a: Vec<f32>,
}

impl Rhythm {
    /// Before the blocks, from the input x [d] (after the letters): the rhythms at this token (None without --rhythm).
    pub(super) fn begin(m: &Model, st: &State, x: &[f32]) -> Result<Option<Rhythm>> {
        let Some(rp) = v51(m).and_then(|v| v.rhythm.as_ref()) else { return Ok(None) };
        let d = x.len();
        let mut hn = x.to_vec();
        head_norm(&mut hn, d);
        let (wr, br, lw) = (ro(&rp.wr)?, ro(&rp.br)?, ro(&rp.lw)?);
        let (wr, br, lw) = (wr.s(), br.s(), lw.s());
        if wr.len() != d * 2 || br.len() != 2 || lw.len() != 2 {
            candle_core::bail!("fused step: the rhythms' sizes");
        }
        let mut pre = vec![0f32; 2];
        gemv(&hn, wr, 2, &mut pre);
        for (p, &b) in pre.iter_mut().zip(br) {
            *p = *p + b;
        }
        let a0 = match st.v51.rh.as_ref() {
            Some(t) => vals(t)?,
            None => vec![0f32; 2],
        };
        if a0.len() != 2 {
            candle_core::bail!("fused step: the rhythms' carried times");
        }
        let a: Vec<f32> = (0..2)
            .map(|o| {
                let l = log_sigmoid(-pre[o]); // ln(1 - r)
                let big = l * 1.0; // the triangular product at c = 1 (K = 1)
                let before = big - l;
                let diff = big - before;
                let w = (if diff > 0.0 { 0.0 } else { diff }).exp() * 1.0;
                w + big.exp() * a0[o]
            })
            .collect();
        let mut f = Vec::with_capacity(4);
        for o in 0..2 {
            let ph = a[o] * lw[o].exp();
            f.push(ph.cos());
            f.push(ph.sin());
        }
        Ok(Some(Rhythm { f, a }))
    }

    /// Block i's gains per head: (the write strength's, the read's), each tanh(f G)·1 + 1 [heads]; None when the block
    /// has no rhythm gates.
    pub(super) fn gates(&self, m: &Model, i: usize, nh: usize) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        let Some((gw, gr)) = v51(m).and_then(|v| v.blocks.get(i)).and_then(|b| b.rh.as_ref()) else { return Ok(None) };
        let (gw, gr) = (ro(gw)?, ro(gr)?);
        if gw.s().len() != 4 * nh || gr.s().len() != 4 * nh {
            candle_core::bail!("fused step: the rhythm gates' sizes");
        }
        let (mut zw, mut zr) = (vec![0f32; nh], vec![0f32; nh]);
        gemv(&self.f, gw.s(), nh, &mut zw);
        gemv(&self.f, gr.s(), nh, &mut zr);
        Ok(Some((zw.into_iter().map(gain).collect(), zr.into_iter().map(gain).collect())))
    }
}

/// P: whether block i's MLP has dendrites (its memory read is then kept as their context).
pub(super) fn has_dendrites(m: &Model, i: usize) -> bool {
    m.cfg.arch.v51.dendrite && v51(m).and_then(|v| v.blocks.get(i)).is_some_and(|b| b.dend.is_some())
}

/// P: block i's MLP units a [hidden] times their dendrites' gain tanh(head_norm(ctx) A B)·1 + 1 (ctx [d] the block's
/// memory read); nothing without --dendrite.
pub(super) fn dendrites(m: &Model, i: usize, ctx: &[f32], a: &mut [f32]) -> Result<()> {
    let Some((da, db)) = v51(m).and_then(|v| v.blocks.get(i)).and_then(|b| b.dend.as_ref()) else { return Ok(()) };
    let (d, h) = (ctx.len(), a.len());
    let (da, db) = (ro(da)?, ro(db)?);
    let r = crate::v51::DEND_RANK;
    if da.s().len() != d * r || db.s().len() != r * h {
        candle_core::bail!("fused step: the dendrites' sizes");
    }
    let mut hn = ctx.to_vec();
    head_norm(&mut hn, d);
    let mut z1 = vec![0f32; r];
    gemv(&hn, da.s(), r, &mut z1);
    let mut z = vec![0f32; h];
    gemv(&z1, db.s(), h, &mut z);
    for (av, zv) in a.iter_mut().zip(z) {
        *av = *av * gain(zv);
    }
    Ok(())
}

/// DA: block i's MLP output y [d] times tanh(z w + c)·1 + 1 (z this token's surprise / ln V); nothing in block 0 or
/// without --dopamine.
pub(super) fn dopamine(m: &Model, i: usize, z: Option<f32>, y: &mut [f32]) -> Result<()> {
    let (Some((w, c)), Some(z)) = (v51(m).and_then(|v| v.blocks.get(i)).and_then(|b| b.da.as_ref()), z) else { return Ok(()) };
    let (w, c) = (ro(w)?, ro(c)?);
    if w.s().len() != y.len() || c.s().len() != y.len() {
        candle_core::bail!("fused step: the dopamine gate's sizes");
    }
    for ((yv, &wv), &cv) in y.iter_mut().zip(w.s()).zip(c.s()) {
        *yv = *yv * gain(z * wv + cv);
    }
    Ok(())
}

#[cfg(test)]
#[path = "arch_f4_tests.rs"]
mod tests;
