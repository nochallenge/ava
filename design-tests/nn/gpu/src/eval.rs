// TESTING / DESIGN ONLY — ava_gpu::eval: validation bits per byte, the protocol of brain/src/eval.rs.
//
// BPB = (sum of cross-entropy in nats over scored targets) / ln 2 / (sum of bytes of those target
// tokens); token bytes = bpe.token_bytes(id).len(), EOS = 1 byte, PAD targets skipped. Windows of
// ctx + 1 tokens come in order from ShardSet::iter_sequential(ctx); inputs = w[..ctx], targets = w[1..];
// the scan stops when `eval_tokens` targets have been scored. The per-row -log softmax is computed in
// f64 on the device from the f32 logits (the reference does the same in f64 on the host).

use crate::model::Model;
use crate::IGNORE;
use ava_data::bpe::{Bpe, EOS, PAD};
use candle_core::{DType, Device, Result, Tensor, D};

/// Bytes per token id: token_bytes(id).len(), with EOS = 1 and PAD = 0 (PAD is never scored).
pub fn byte_lengths(bpe: &Bpe) -> Vec<u32> {
    (0..bpe.vocab_size() as u32)
        .map(|id| match id {
            EOS => 1,
            PAD => 0,
            _ => bpe.token_bytes(id).len() as u32,
        })
        .collect()
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BpbAcc {
    pub nats: f64,
    pub bytes: u64,
    pub tokens: u64,
}

impl BpbAcc {
    pub fn nats_per_token(&self) -> f64 {
        if self.tokens == 0 {
            f64::NAN
        } else {
            self.nats / self.tokens as f64
        }
    }

    pub fn bpb(&self) -> f64 {
        if self.bytes == 0 {
            f64::NAN
        } else {
            self.nats / std::f64::consts::LN_2 / self.bytes as f64
        }
    }
}

/// Split one window of ctx + 1 tokens into (inputs [ctx], targets [ctx]). PAD targets become IGNORE;
/// at most `*remaining` targets are kept (the rest become IGNORE) and `*remaining` is decreased.
pub fn window_io(window: &[u32], remaining: &mut u64) -> (Vec<u32>, Vec<u32>) {
    assert!(window.len() >= 2, "window_io: window of {} tokens", window.len());
    let ctx = window.len() - 1;
    let inputs = window[..ctx].to_vec();
    let mut targets = Vec::with_capacity(ctx);
    for &t in &window[1..] {
        if t == PAD || *remaining == 0 {
            targets.push(IGNORE);
        } else {
            targets.push(t);
            *remaining -= 1;
        }
    }
    (inputs, targets)
}

/// Training batch from sampler windows: (inputs [b·ctx], targets [b·ctx]).
pub fn train_io(windows: &[Vec<u32>]) -> (Vec<u32>, Vec<u32>) {
    let mut ids = Vec::new();
    let mut tg = Vec::new();
    for w in windows {
        let ctx = w.len() - 1;
        ids.extend_from_slice(&w[..ctx]);
        tg.extend_from_slice(&w[1..]);
    }
    (ids, tg)
}

/// Score one batch: per-row -log softmax(logits)[target] in f64 for rows whose target is not IGNORE.
fn score_batch(model: &Model, ids: &[u32], tg: &[u32], b: usize, t: usize, byte_len: &[u32], dev: &Device, acc: &mut BpbAcc) -> Result<()> {
    let ids_t = Tensor::from_slice(ids, (b, t), dev)?;
    let logits = model.forward(&ids_t)?; // [b·t, V] f32
    let safe: Vec<u32> = tg.iter().map(|&x| if x == IGNORE { 0 } else { x }).collect();
    let tg_t = Tensor::from_slice(&safe, (b * t, 1), dev)?;
    let lsm = candle_nn::ops::log_softmax(&logits.to_dtype(DType::F64)?, D::Minus1)?;
    let nll = lsm.gather(&tg_t, D::Minus1)?.neg()?.flatten_all()?.to_vec1::<f64>()?;
    for (r, &target) in tg.iter().enumerate() {
        if target == IGNORE || target == PAD {
            continue;
        }
        acc.nats += nll[r];
        acc.bytes += byte_len[target as usize] as u64;
        acc.tokens += 1;
    }
    Ok(())
}

/// Evaluate on `windows` (each ctx + 1 tokens) until `eval_tokens` targets are scored or the windows run
/// out; `batch` windows per forward pass. Returns the accumulator and the number of windows used.
pub fn evaluate<I>(model: &Model, windows: I, batch: usize, eval_tokens: u64, byte_len: &[u32], dev: &Device) -> Result<(BpbAcc, usize)>
where
    I: IntoIterator<Item = Vec<u32>>,
{
    let mut acc = BpbAcc::default();
    let mut remaining = eval_tokens;
    let mut it = windows.into_iter();
    let mut n_windows = 0usize;
    while remaining > 0 {
        let mut ids = Vec::new();
        let mut tg = Vec::new();
        let (mut b, mut t) = (0usize, 0usize);
        while b < batch && remaining > 0 {
            let Some(w) = it.next() else { break };
            let (i, tt) = window_io(&w, &mut remaining);
            if b == 0 {
                t = i.len();
            }
            assert_eq!(i.len(), t, "evaluate: windows of different lengths");
            ids.extend(i);
            tg.extend(tt);
            b += 1;
        }
        if b == 0 {
            break;
        }
        n_windows += b;
        score_batch(model, &ids, &tg, b, t, byte_len, dev, &mut acc)?;
    }
    Ok((acc, n_windows))
}
