// TESTING / DESIGN ONLY — ava_gpu::fast4::live: v52, the chat app's live learning (src/bin/ava_live.rs --learn). One
// token of a one-row session through fast4's fused session step only, with what the chat app needs beside it:
// * E2, e-prop (`Model::plastic_begin` etc.): src/fast4/eprop.rs's session step (`eprop_live`), the cerebellum head
//   and its online table (CB2) on its tape when both learn;
// * CB2 alone: the fused step (`fstep::arch::step` / `step_traced`) with the online table in it, then
//   `fast4_cb_learn_sized` on its logits;
// * the target is chosen from this token's own logits (`Target`): while she reads it is the next word of the text,
//   while she speaks the word she draws there; None: nothing is learned at this token (the turn ends);
// * the viewer's record of the token (`StepTrace`, as `Model::step_trace` gives it), on the side.
// Bit for bit with fast4's own session calls (`fast4_eprop_step`; `fast4_state_step` + `fast4_cb_learn`) and with the
// State path (`forward_with_state` + backward + `plastic_learn_rows` / `cb_learn`): fast4/live_tests.rs.

use crate::model::fstep::arch::{self, TokTrace};
use crate::model::{Model, StepLayer, StepTrace};
use crate::state::State;
use candle_core::{Result, Tensor};

/// One token of a live session.
pub struct LiveStep {
    /// the logits [V] after this token
    pub logits: Tensor,
    /// the token's activity for the viewer (asked for with `trace`)
    pub trace: Option<StepTrace>,
    /// the word this token learned to predict (None: nothing was learned at this token)
    pub next: Option<u32>,
    /// e-prop's learning step: the rms of the change over every plastic weight (as `plastic_learn_rows` reports it)
    pub eprop_rms: Option<f64>,
    /// the cerebellum's online step: (the nats of `next` before it, the rms of its change over the whole table)
    pub cb: Option<(f64, f64)>,
}

impl Model {
    /// Why `fast4_live_step` would not take a token of this session now (None: it would): `set_fast4` off, or the
    /// e-prop session step's blocker (with an e-prop session on), or the fused step's (without one).
    pub fn fast4_live_blocker(&self) -> Option<&'static str> {
        if !self.fast4() {
            return Some("live learning: `set_fast4` is off");
        }
        if self.cb_online_with(|o| o.rows != 1).unwrap_or(false) {
            return Some("live learning: one row");
        }
        if crate::eprop::active(self) {
            if self.v51b().is_some_and(|e| e.plastic.borrow().as_ref().is_some_and(|p| p.rows != 1 || p.half.len() != 1)) {
                return Some("live learning: one row");
            }
            return self.fast4_eprop_blocker();
        }
        self.fused_blocker()
    }

    /// v52: one token `tok` of a one-row live session on `st`, fused. With an e-prop session (probes on) it learns at
    /// rate `eprop_lr` (fast4's e-prop session step); with the cerebellum's online table on, the table learns too
    /// (`fast4_cb_learn`). Both learn the target `target` picks from this token's logits [V]. `trace`: also the
    /// token's record for the viewer. None: the fused step does not take this token (nothing changed; the caller
    /// stops learning rather than run the State path).
    pub fn fast4_live_step(&self, st: &mut State, tok: u32, eprop_lr: f64, trace: bool, target: &mut dyn FnMut(&[f32]) -> Result<Option<u32>>) -> Result<Option<LiveStep>> {
        if !self.fast4() {
            return Ok(None);
        }
        let _q8 = self.int8_scope();
        let mut rec = trace.then(TokTrace::default);
        let (logits, eprop_rms, next) = if crate::eprop::active(self) {
            match super::eprop::eprop_live(self, st, tok, eprop_lr, rec.as_mut(), target)? {
                None => return Ok(None),
                Some(s) => (s.logits, s.rms, s.next),
            }
        } else {
            let l = match rec.as_mut() {
                Some(r) => arch::step_traced(self, st, tok, r)?,
                None => arch::step(self, st, tok)?,
            };
            let Some(l) = l else { return Ok(None) };
            let v = l.to_vec1::<f32>()?;
            let next = target(&v)?;
            (v, None, next)
        };
        let cb = match next {
            Some(y) if self.cb_online_on() => Some(self.fast4_cb_learn_sized(&logits, y)?),
            _ => None,
        };
        let trace = match rec {
            Some(r) => Some(self.live_trace(r, st)?),
            None => None,
        };
        let n = logits.len();
        Ok(Some(LiveStep { logits: Tensor::from_vec(logits, n, self.tok.device())?, trace, next, eprop_rms, cb }))
    }

    /// The fused record as `Model::step_trace` hands it out (its State-backed fused branch, the same fields).
    fn live_trace(&self, rec: TokTrace, st: &State) -> Result<StepTrace> {
        let dev = self.tok.device().clone();
        let t1 = |v: Vec<f32>| -> Result<Tensor> {
            let n = v.len();
            Tensor::from_vec(v, (1, n), &dev)
        };
        let t0 = |v: Vec<f32>| -> Result<Option<Tensor>> {
            if v.is_empty() {
                return Ok(None);
            }
            let n = v.len();
            Tensor::from_vec(v, n, &dev).map(Some)
        };
        let mut layers = Vec::with_capacity(rec.layers.len());
        for ly in rec.layers {
            let engram = match ly.engram {
                Some((w, n, r)) => {
                    let nh = w.len() / n.max(1);
                    (Some(Tensor::from_vec(w, (nh, n), &dev)?), Some(t1(r)?))
                }
                None => (None, None),
            };
            layers.push(StepLayer {
                spikes: t1(ly.spikes)?,
                winners: t1(ly.winners)?,
                hidden: t1(ly.hidden)?,
                mem_write: t0(ly.mem_write)?,
                mem_read: t0(ly.mem_read)?,
                mem_keep: t0(ly.mem_keep)?,
                mem_gate: t0(ly.mem_gate)?,
                engram_attn: engram.0,
                engram_read: engram.1,
            });
        }
        Ok(StepTrace { layers, place: rec.place, mag: rec.mag, input_rms: rec.input_rms, v51: crate::v51::trace(self, &st.v51)? })
    }
}
