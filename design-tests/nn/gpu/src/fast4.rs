// TESTING / DESIGN ONLY — ava_gpu::fast4: v51 fast4 (runs/v51/fast4.md): the fused one-token talking step with an
// online session on (e-prop with its fading, the cerebellum's online table with its fading), with the same bits as
// the State path's session (`forward_with_state`, `cb_learn`, `loss.backward()` + `plastic_learn_rows`).
// * fmm: the f32 one-row products in gemm's order (the fused step with int8 off, `Model::set_fast4`);
// * cb: the cerebellum's online learning step for one row, in place (its forward part is in the fused step);
// * eprop: the e-prop session step (the frozen blocks fused, the plastic blocks on a small tape that mirrors candle's
//   graph and backward, the changes and traces updated in place);
// * live (v52): the chat app's live learning, one token of a session (e-prop and / or the cerebellum's table) with the
//   target drawn from its own logits and the viewer's record (src/bin/ava_live.rs --learn).

pub(crate) mod cb;
pub(crate) mod eprop;
pub(crate) mod fmm;
// v52: the chat app's live learning (one token of a session with what the chat app needs beside it)
pub mod live;
pub(crate) mod tape;

use crate::model::Model;
use candle_core::{CpuStorage, Result, Storage, Tensor};

/// Why the f32 fused step (int8 off, `Model::set_fast4`) does not apply beyond the int8 step's own checks; None: it
/// applies.
pub(crate) fn f32_blocker(m: &Model) -> Option<&'static str> {
    if m.cfg.vocab <= 256 {
        return Some("f32 fused step: a vocabulary of 256 or fewer (gemm takes its small kernel for the head)");
    }
    None
}

/// f(the values of a contiguous CPU f32 tensor), read in place.
pub(crate) fn with_f32<R>(t: &Tensor, f: impl FnOnce(&[f32]) -> R) -> Result<R> {
    let (g, l) = t.storage_and_layout();
    let Some((a, b)) = l.contiguous_offsets() else { candle_core::bail!("fast4: a non-contiguous tensor") };
    match &*g {
        Storage::Cpu(CpuStorage::F32(v)) => Ok(f(&v[a..b])),
        _ => candle_core::bail!("fast4: not a CPU f32 tensor"),
    }
}

impl Model {
    /// v51 fast4: one token through the fused step on a one-row State (the cerebellum's online table in it when on);
    /// None where the fused step does not apply (`fused_blocker`, the State's shape): the caller runs
    /// `forward_with_state`. The logits [1, V], as `forward_with_state` gives them for one token.
    pub fn fast4_state_step(&self, st: &mut crate::state::State, tok: u32) -> Result<Option<Tensor>> {
        let _q8 = self.int8_scope();
        match crate::model::fstep::arch::step(self, st, tok)? {
            Some(l) => Ok(Some(l.unsqueeze(0)?)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod cb_tests;
#[cfg(test)]
mod eprop_tests;
#[cfg(test)]
mod fmm_tests;
#[cfg(test)]
mod live_tests;
#[cfg(test)]
mod prof_tests;
