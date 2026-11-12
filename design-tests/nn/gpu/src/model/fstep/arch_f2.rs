// TESTING / DESIGN ONLY — v51 fast2 in the fused one-token talking step (src/model/fstep/arch.rs): R --letters and TH
// --thalamus, with the same bits as `forward_with_state` with int8 on (every float operation the one candle performs on
// the CPU, in its order; the small products in gemm's one-row order, fstep.rs's `gemv`):
//   R   the token's letter vector l = (0 + m_0 tanh((E[b_0] + Q[e_0]) + P[0]) + ...) inv (v51::letter_core on one row:
//       candle's sum over a middle dim runs from 0 in place order), then l Wo (gemv), added to the input
//   TH  the relay's reads rmsnorm(x) R_j (gemv into 16), their sum, each block's gates 2 sigmoid(tanh(acc / sqrt(reads))
//       G_i + c_i) (gemv into 2), the memory read times g_mem, the residual rule x + (x - x_in)(g_res · 1 + -1), the
//       chat app's part signals (as v51b_th_out records them)
// Not here (the talking step still refuses them, src/model/fstep/arch.rs `blocker`): W --columns (its per-column
// products and kWTA in the MLP) and the cerebellum's online table (CB itself: v51 fast3, arch_f3.rs); runs/v51/fast2.md
// says what W would take.

use super::super::{gemv, rmsnorm, ro, sigmoid};
use crate::model::Model;
use candle_core::Result;

/// The four v51/v51b options this step takes off the blocker's list (R, TH); the others stay.
pub(super) fn strip(o: crate::v51::V51Opts) -> crate::v51::V51Opts {
    crate::v51::V51Opts { letters: false, ..o }
}

/// v51b's options the talking step does not run (an online session: e-prop; the cerebellum's online table for more
/// than one row), as the blocker's reason. v51 fast3: CB itself runs here (arch_f3.rs `cb`); v51 fast4: its online
/// table too (one row, with `Model::set_fast4`), and e-prop through src/fast4.rs's session step (not this plain step).
pub(super) fn v51b_why(m: &Model) -> Option<&'static str> {
    if crate::eprop::active(m) && !crate::fast4::eprop::session_on() {
        return Some("v51b: an e-prop session runs in src/fast4.rs's session step, not the plain fused step");
    }
    if m.cb_online_with(|o| o.rows != 1 || !m.fast4()).unwrap_or(false) {
        return Some("v51b: the cerebellum's online table is in the fused step only with `set_fast4` and one row");
    }
    None
}

/// R: the letters' share of token `tok`'s input [d] (None without --letters).
pub(super) fn letters_add(m: &Model, tok: u32) -> Result<Option<Vec<f32>>> {
    let Some(lt) = m.ext.as_ref().map(|e| &e.v51).filter(|v| v.on).and_then(|v| v.letters.as_ref()) else { return Ok(None) };
    let (dl, lm, d) = (crate::v51::LETTER_DIM, crate::v51::LETTER_MAX, m.cfg.d);
    let t = tok as usize;
    let ids: Vec<u32> = lt.ids.get(t)?.to_vec1()?;
    let eidx: Vec<u32> = lt.eidx.get(t)?.to_vec1()?;
    let mask: Vec<f32> = lt.mask.get(t)?.to_vec1()?;
    let inv: f32 = lt.inv.get(t)?.get(0)?.to_scalar()?;
    let (e, p, q) = (ro(&lt.byte)?, ro(&lt.pos)?, ro(&lt.end)?);
    let (e, p, q) = (e.s(), p.s(), q.s());
    let mut l = vec![0f32; dl];
    for j in 0..lm {
        let (b, k) = (ids[j] as usize, eidx[j] as usize);
        for c in 0..dl {
            let s = ((e[b * dl + c] + q[k * dl + c]) + p[j * dl + c]).tanh();
            l[c] += s * mask[j];
        }
    }
    for v in l.iter_mut() {
        *v *= inv;
    }
    let wo = ro(&lt.wo)?;
    let mut out = vec![0f32; d];
    gemv(&l, wo.s(), d, &mut out);
    Ok(Some(out))
}

/// TH: the relay of one token's pass over the blocks.
pub(super) struct Relay {
    d: usize,
    nr: usize,
    read: Vec<f32>,
    gate: Vec<f32>,
    bias: Vec<f32>,
    ones: Vec<f32>,
    acc: Vec<f32>,
    reads: usize,
    /// the current block's gates (residual, memory)
    g: [f32; 2],
}

impl Relay {
    /// The relay of `m` (None without --thalamus), reading the blocks' input `x0`.
    pub(super) fn begin(m: &Model, x0: &[f32]) -> Result<Option<Relay>> {
        let Some(th) = m.v51b().and_then(|e| e.th.as_ref()) else { return Ok(None) };
        let d = m.cfg.d;
        let mut r = Relay {
            d,
            nr: th.read.dim(0)?,
            read: ro(&th.read)?.s().to_vec(),
            gate: ro(&th.gate)?.s().to_vec(),
            bias: ro(&th.bias)?.s().to_vec(),
            ones: vec![1.0; d],
            acc: Vec::new(),
            reads: 0,
            g: [1.0, 1.0],
        };
        r.acc = r.read_of(0, x0);
        r.reads = 1;
        Ok(Some(r))
    }

    /// v51b::th_read(x, ones, R_j): rmsnorm(x) · R_j [16].
    fn read_of(&self, j: usize, x: &[f32]) -> Vec<f32> {
        let k = crate::v51b::TH_K;
        let xn = rmsnorm(x, &self.ones);
        let mut out = vec![0f32; k];
        gemv(&xn, &self.read[j * self.d * k..(j + 1) * self.d * k], k, &mut out);
        out
    }

    /// Block i starts: its gates (v51b::th_gates).
    pub(super) fn block(&mut self, i: usize) {
        let k = crate::v51b::TH_K;
        let mul = (1.0 / (self.reads as f64).sqrt()) as f32;
        let f: Vec<f32> = self.acc.iter().map(|&a| (a * mul + 0.0).tanh()).collect();
        let mut z = [0f32; 2];
        gemv(&f, &self.gate[i * k * 2..(i + 1) * k * 2], 2, &mut z);
        for (c, zc) in z.iter().enumerate() {
            self.g[c] = sigmoid(zc + self.bias[i * 2 + c]) * 2.0 + 0.0;
        }
    }

    /// The memory read y [d] (after its output projection) times the memory gate.
    pub(super) fn mem(&self, y: &mut [f32]) {
        for v in y.iter_mut() {
            *v *= self.g[1];
        }
    }

    /// Block i's output x [d] (before its dense mix) under the residual rule from its input `x_in`; the chat app's part
    /// signals as v51b_th_out keeps them.
    pub(super) fn out(&self, m: &Model, i: usize, x_in: &[f32], x: &mut [f32]) {
        let gm1 = self.g[0] * 1.0 + -1.0;
        for (o, &xi) in x.iter_mut().zip(x_in) {
            *o = *o + (*o - xi) * gm1;
        }
        if let Some(e) = m.v51b() {
            if e.sig_on.get() && !m.v50_quiet() {
                let mut s = e.sig.borrow_mut();
                if s.th.len() <= i {
                    s.th.resize(i + 1, [1.0, 1.0]);
                }
                s.th[i] = self.g;
            }
        }
    }

    /// After block i's output x is final (its dense mix included): the relay reads it if a later gate needs it.
    pub(super) fn after(&mut self, i: usize, x: &[f32]) {
        if i + 1 < self.nr {
            let r = self.read_of(i + 1, x);
            for (a, b) in self.acc.iter_mut().zip(&r) {
                *a = *a + b;
            }
            self.reads += 1;
        }
    }
}

#[cfg(test)]
#[path = "arch_f2_tests.rs"]
mod tests;
