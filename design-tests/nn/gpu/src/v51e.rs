// TESTING / DESIGN ONLY — ava_gpu::v51e: v51 Part B's one redesign (runs/v51/prereg_v51.md, Amendment B-4): the
// forgetting test's shared trunk.
//
// The redesigned forgetting test scores many brains that differ only in their last blocks (E2 / dyn2: blocks 6 and 7)
// or only in the cerebellum's read-out after the last block (CB2), each on the same windows of text. The blocks below
// the first changed one compute exactly the same thing for every brain, so they run once:
//   record   `Model::trunk_record(&[k, ..])`: the next forwards (the State path, `forward_with_state`) run as always and
//            keep, per call and per k, what block k reads: the stream after block k-1, the dense residual's list of
//            earlier streams and the value residual's first values (only the stream when k is the number of blocks:
//            the final norm reads it). `Model::trunk_take_records` hands the records over.
//   replay   `Model::trunk_replay(rec)` on a brain with the same blocks 0..k-1 (its own later blocks): each forward
//            skips blocks 0..k-1 and starts block k from the next recorded piece, so the same calls in the same order
//            give the same numbers as the full forward, bit for bit (the same ops on the same values; v51e/tests.rs).
//            A replay with g times the recorded rows reads each piece g times over (row g·b + r reads recorded row r).
// Only for models whose blocks have no side effects outside the block loop's own values (no TH, no v50 nucleus, no
// v51a options, no --loop, --feedback or --modulate): `trunk_record` / `trunk_replay` refuse the others. Off (no
// record, no replay), the block loop runs exactly as before.

use crate::model::Model;
use candle_core::{Result, Tensor};

/// One forward's input to block k.
#[derive(Clone)]
pub struct Piece {
    /// the stream block k reads [b·t, d]
    pub x: Tensor,
    /// the dense residual's list of earlier streams (empty when k is the number of blocks)
    pub streams: Vec<Tensor>,
    /// the value residual's first values [b, heads, t, head size] (None when k is the number of blocks)
    pub v0: Option<Tensor>,
    pub b: usize,
    pub t: usize,
}

/// A trunk record: what block k read, call by call.
pub struct Trunk {
    pub k: usize,
    pub pieces: Vec<Piece>,
    /// replay: the next piece
    pub next: usize,
}

impl Trunk {
    /// The bytes the record holds.
    pub fn bytes(&self) -> usize {
        self.pieces.iter().map(|p| (p.x.elem_count() + p.streams.iter().map(|s| s.elem_count()).sum::<usize>() + p.v0.as_ref().map_or(0, |v| v.elem_count())) * 4).sum()
    }
}

fn tile(x: &Tensor, g: usize) -> Result<Tensor> {
    if g == 1 {
        Ok(x.clone())
    } else {
        Tensor::cat(&vec![x; g], 0)
    }
}

impl Model {
    fn trunk_check(&self, k: usize) -> Result<()> {
        let ar = self.cfg.arch;
        if self.v51b().is_none() {
            candle_core::bail!("trunk: a model with block options");
        }
        if k == 0 || k > self.blocks.len() {
            candle_core::bail!("trunk: k = {k} for {} blocks", self.blocks.len());
        }
        if ar.v51b.thalamus || ar.neuromod > 0 || ar.v51.any() || ar.loops > 1 || ar.feedback || ar.mem.modulate {
            candle_core::bail!("trunk: not for --thalamus, --neuromod, v51a's options, --loop, --feedback or --modulate");
        }
        Ok(())
    }

    /// Record, for each k of `ks`, what block k reads in the next State-path forwards (see the top of the file).
    pub fn trunk_record(&self, ks: &[usize]) -> Result<()> {
        for &k in ks {
            self.trunk_check(k)?;
        }
        *self.v51b().expect("checked").trunk_rec.borrow_mut() = ks.iter().map(|&k| Trunk { k, pieces: Vec::new(), next: 0 }).collect();
        Ok(())
    }

    /// End the records and hand them over (in the order of `trunk_record`'s ks).
    pub fn trunk_take_records(&self) -> Vec<Trunk> {
        self.v51b().map(|e| std::mem::take(&mut *e.trunk_rec.borrow_mut())).unwrap_or_default()
    }

    /// Replay a record: the next State-path forwards start at block `rec.k` from its pieces, in order.
    pub fn trunk_replay(&self, mut rec: Trunk) -> Result<()> {
        self.trunk_check(rec.k)?;
        rec.next = 0;
        *self.v51b().expect("checked").trunk.borrow_mut() = Some(rec);
        Ok(())
    }

    /// End a replay and hand the record back.
    pub fn trunk_take(&self) -> Option<Trunk> {
        self.v51b().and_then(|e| e.trunk.borrow_mut().take())
    }

    /// The State path's block loop: Some(k) while a replay is on.
    pub(crate) fn trunk_replay_k(&self) -> Option<usize> {
        self.v51b().and_then(|e| e.trunk.borrow().as_ref().map(|t| t.k))
    }

    /// The State path's block loop: whether any record is on.
    pub(crate) fn trunk_recording(&self) -> bool {
        self.v51b().is_some_and(|e| !e.trunk_rec.borrow().is_empty())
    }

    /// Record: block i has run; keep what block i + 1 reads for the records with k = i + 1. Each tensor is kept as
    /// its own copy of exactly its values (detached, contiguous): a view would keep its whole parent storage alive,
    /// and a tensor of the forward keeps the whole graph of that forward alive (the weights are variables).
    pub(crate) fn trunk_put(&self, i: usize, x: &Tensor, streams: &[Tensor], v0: &Option<Tensor>, b: usize, t: usize) -> Result<()> {
        if let Some(e) = self.v51b() {
            let last = i + 1 == self.blocks.len();
            let own = |t: &Tensor| t.detach().force_contiguous();
            for tr in e.trunk_rec.borrow_mut().iter_mut().filter(|tr| tr.k == i + 1) {
                let streams = if last { Vec::new() } else { streams.iter().map(own).collect::<Result<Vec<_>>>()? };
                let v0 = if last { None } else { v0.as_ref().map(own).transpose()? };
                tr.pieces.push(Piece { x: own(x)?, streams, v0, b, t });
            }
        }
        Ok(())
    }

    /// Replay: the next piece for a forward of b rows of t tokens (the recorded rows tiled b / recorded times).
    pub(crate) fn trunk_get(&self, b: usize, t: usize) -> Result<(Tensor, Vec<Tensor>, Option<Tensor>)> {
        let e = self.v51b().ok_or_else(|| candle_core::Error::Msg("trunk: no record".into()))?;
        let mut g = e.trunk.borrow_mut();
        let tr = g.as_mut().ok_or_else(|| candle_core::Error::Msg("trunk: no record".into()))?;
        let p = tr.pieces.get(tr.next).ok_or_else(|| candle_core::Error::Msg(format!("trunk: replay asks for piece {} of {}", tr.next, tr.pieces.len())))?;
        if p.t != t || b % p.b != 0 {
            candle_core::bail!("trunk: piece {} was {} rows x {} tokens, the forward is {b} x {t}", tr.next, p.b, p.t);
        }
        let gr = b / p.b;
        let x = tile(&p.x, gr)?;
        let streams = p.streams.iter().map(|s| tile(s, gr)).collect::<Result<Vec<_>>>()?;
        let v0 = p.v0.as_ref().map(|v| tile(v, gr)).transpose()?;
        tr.next += 1;
        Ok((x, streams, v0))
    }
}

#[cfg(test)]
mod tests;
