// TESTING / DESIGN ONLY
//! N1: an any-order objective, block masked denoising on top of her recurrent state (block diffusion;
//! LLaDA and masked diffusion LMs, 2025). Predicting only the next token teaches one direction: a model
//! trained on "A is B" often cannot answer "what is B?" (the reversal curse), and it gets less from each
//! repeat of the data. Here a small head looks at her state h after the context plus the next K tokens
//! with a random share of them hidden, and fills in the hidden ones from both sides:
//!   x_i = embed(block_i or MASK) + pos_i + h,   one bidirectional attention layer inside the block + MLP,
//!   loss = Σ_masked −log p(token_i) / (share masked)        (the masked-diffusion ELBO weighting)
//! It is an extra loss in replay (sleep), from a separate head, so the model's normal forward pass is
//! unchanged; its weight defaults to 0.

use crate::linear;
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

pub struct BlockDenoiser {
    pub vocab: usize,
    pub k: usize,
    pub d: usize,
    pub emb: Tensor,
    pub pos: Tensor,
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub w1: Tensor,
    pub w2: Tensor,
    pub out: Tensor,
}

impl BlockDenoiser {
    /// Token id `vocab` is MASK.
    pub fn new(vb: VarBuilder, vocab: usize, k: usize, d: usize) -> Result<BlockDenoiser> {
        let s = 1.0 / (d as f64).sqrt();
        let r = |n: &str, shape: (usize, usize), sd: f64| vb.get_with_hints(shape, n, Init::Randn { mean: 0.0, stdev: sd });
        Ok(BlockDenoiser {
            vocab,
            k,
            d,
            emb: r("emb", (vocab + 1, d), 1.0)?,
            pos: r("pos", (k, d), 1.0)?,
            wq: r("wq", (d, d), s)?,
            wk: r("wk", (d, d), s)?,
            wv: r("wv", (d, d), s)?,
            wo: r("wo", (d, d), s)?,
            w1: r("w1", (d, 4 * d), s)?,
            w2: r("w2", (4 * d, d), 0.5 * s)?,
            out: r("out", (d, vocab), s)?,
        })
    }

    /// h [B, d] (her state after the context), block [B, K] token ids with MASK → logits [B, K, V].
    pub fn logits(&self, h: &Tensor, block: &Tensor) -> Result<Tensor> {
        let (b, k) = block.dims2()?;
        let x = self.emb.index_select(&block.flatten_all()?, 0)?.reshape((b, k, self.d))?;
        let x = x.broadcast_add(&self.pos)?.broadcast_add(&h.unsqueeze(1)?)?;
        let (q, kk, v) = (linear(&x, &self.wq)?, linear(&x, &self.wk)?, linear(&x, &self.wv)?);
        let att = (q.matmul(&kk.transpose(1, 2)?.contiguous()?)? / (self.d as f64).sqrt())?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        let x = (&x + linear(&att.matmul(&v)?, &self.wo)?)?;
        // An MLP after the attention: it combines where a token sits with what it attended to.
        let x = (&x + linear(&linear(&x, &self.w1)?.gelu_erf()?, &self.w2)?)?;
        linear(&x, &self.out)
    }

    /// Hides each token of `targets` [B, K] with probability `share[b]` (at least one per row), from a
    /// seed. Returns (block with MASK, masked as 0/1 f32 [B, K]).
    pub fn mask(&self, targets: &[Vec<u32>], share: &[f32], seed: u64) -> Result<(Tensor, Tensor)> {
        let dev = self.emb.device();
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32
        };
        let (mut blk, mut m) = (Vec::new(), Vec::new());
        for (row, &p) in targets.iter().zip(share) {
            let mut hide: Vec<bool> = row.iter().map(|_| u() < p).collect();
            if !hide.iter().any(|&x| x) {
                let i = ((u() * row.len() as f32) as usize).min(row.len() - 1);
                hide[i] = true;
            }
            for (&t, &hd) in row.iter().zip(&hide) {
                blk.push(if hd { self.vocab as u32 } else { t });
                m.push(if hd { 1f32 } else { 0.0 });
            }
        }
        let (b, k) = (targets.len(), self.k);
        Ok((Tensor::from_vec(blk, (b, k), dev)?, Tensor::from_vec(m, (b, k), dev)?))
    }
}

/// Σ over masked positions of −log p(target) / share, averaged over the B·K positions.
pub fn mdlm_loss(logits: &Tensor, targets: &Tensor, masked: &Tensor, share: &Tensor) -> Result<Tensor> {
    let lp = candle_nn::ops::log_softmax(logits, D::Minus1)?;
    let nll = lp.gather(&targets.unsqueeze(D::Minus1)?, D::Minus1)?.squeeze(D::Minus1)?.neg()?;
    let w = masked.broadcast_div(&share.unsqueeze(1)?)?;
    (nll * w)?.mean_all()
}

/// The share of masked positions predicted right (argmax).
pub fn masked_accuracy(logits: &Tensor, targets: &Tensor, masked: &Tensor) -> Result<f32> {
    let pred = logits.argmax(D::Minus1)?;
    let right = pred.eq(targets)?.to_dtype(DType::F32)?;
    let n = masked.sum_all()?.to_scalar::<f32>()?;
    Ok((right * masked)?.sum_all()?.to_scalar::<f32>()? / n.max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_core::Device;
    use candle_nn::{AdamW, Optimizer, ParamsAdamW};

    /// Blocks [a, a+1, a+2, a+3] mod V from a random start a; the context says nothing (h = 0).
    fn blocks(n: usize, v: u32, seed: u64) -> Vec<Vec<u32>> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let a = ((s >> 33) % v as u64) as u32;
                (0..4).map(|i| (a + i) % v).collect()
            })
            .collect()
    }

    #[test]
    fn fills_hidden_tokens_from_either_side() {
        let (vocab, k, d, b) = (12usize, 4usize, 32usize, 64usize);
        let (vm, vb) = vm();
        let m = BlockDenoiser::new(vb, vocab, k, d).unwrap();
        let s = 1.0 / (d as f64).sqrt();
        seed(&vm, &[("emb", 1.0), ("pos", 1.0), ("wq", s), ("wk", s), ("wv", s), ("wo", s), ("w1", s), ("w2", 0.5 * s), ("out", s)], 11);
        let dev = Device::Cpu;
        let h = Tensor::zeros((b, d), DType::F32, &dev).unwrap();
        let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 3e-3, weight_decay: 0.0, ..Default::default() }).unwrap();
        for step in 0..400u64 {
            let rows = blocks(b, vocab as u32, step + 1);
            let share: Vec<f32> = (0..b).map(|i| 0.25 + 0.75 * ((i * 37 + step as usize * 11) % 64) as f32 / 64.0).collect();
            let (blk, masked) = m.mask(&rows, &share, step + 7).unwrap();
            let tg = Tensor::from_vec(rows.concat(), (b, k), &dev).unwrap();
            let sh = Tensor::from_vec(share, b, &dev).unwrap();
            opt.backward_step(&mdlm_loss(&m.logits(&h, &blk).unwrap(), &tg, &masked, &sh).unwrap()).unwrap();
        }
        // Hide only the FIRST token: it must be inferred backwards from the later ones.
        let rows = blocks(b, vocab as u32, 99_999);
        let tg = Tensor::from_vec(rows.concat(), (b, k), &dev).unwrap();
        let mut blk = rows.clone();
        blk.iter_mut().for_each(|r| r[0] = vocab as u32);
        let blk = Tensor::from_vec(blk.concat(), (b, k), &dev).unwrap();
        let mut mk = vec![0f32; b * k];
        (0..b).for_each(|i| mk[i * k] = 1.0);
        let mk = Tensor::from_vec(mk, (b, k), &dev).unwrap();
        let back = masked_accuracy(&m.logits(&h, &blk).unwrap(), &tg, &mk).unwrap();
        // Hide everything: with no context there is nothing to go on (chance is 1/12).
        let all = Tensor::full(vocab as u32, (b, k), &dev).unwrap();
        let none = masked_accuracy(&m.logits(&h, &all).unwrap(), &tg, &Tensor::ones((b, k), DType::F32, &dev).unwrap()).unwrap();
        eprintln!("first token from the later ones: {back:.2}; all hidden: {none:.2}");
        assert!(back > 0.9 && none < 0.3, "{back} {none}");
    }
}
