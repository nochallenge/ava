// TESTING / DESIGN ONLY — preq (v42 I1): the prequential (online) score, the yardstick of a brain that never stops
// learning: the bits she pays on new text before she learns it.
//
// Text comes in order, a batch at a time. For each batch: (1) score it with her weights as they are now (the code length
// she pays for it), then (2) take one learning step on exactly that batch (predict, then learn). The total of (1) over
// the stream is the prequential code length (MDL; Rissanen, Blier & Ollivier 2018). Beside it, a frozen copy of the
// starting weights scores the same batches (the static score), so gain = static − prequential is what learning on the
// fly saved. Both are scored by the same batched pass (probe::token_nll, f32 log-softmax, PAD never scored), in nats,
// and per byte (the byte lengths of eval::byte_lengths_n; EOS = 1 byte, marks 0).
//
// A step is the weighted-mean loss over the scored targets (a batch's padding and unscored prompt tokens weigh 0): on
// CUDA, D and A take grow_gpu's hand-written step (Model::loss_and_grads_a_w, bf16 operands as AVA_BF16 says), any other
// case candle's autograd (Model::loss_and_grads_autograd); then global-norm clipping and AdamW (betas 0.9/0.95, decay on
// the matrices only, Model::decays) at a constant LR after a linear warm-up. Scoring runs in inference mode, the step not.

use crate::model::{Kind, Mm, Model};
use crate::optim::{clip_scale, AdamW, Param};
use ava_data::bpe::PAD;
use candle_core::{Device, Tensor};
use candle_nn::VarMap;

#[derive(Clone, Copy, Debug)]
pub struct PreqOpts {
    pub lr: f64,
    pub warmup: usize,
    pub wd: f64,
    pub clip: f64,
    /// tokens per scoring pass at most (rows · longest row)
    pub batch_tokens: usize,
}

/// One batch of the stream: tokens and bytes scored, and the nats she paid (prequential) and the frozen copy paid.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PreqStep {
    pub step: usize,
    pub tokens: u64,
    pub bytes: u64,
    pub preq: f64,
    pub stat: f64,
    pub loss: f32,
}

/// An AdamW over every weight of `varmap` (sorted by name; decay on the matrices only).
pub fn optimizer(varmap: &VarMap, wd: f64) -> Result<AdamW, String> {
    let mut names: Vec<(String, candle_core::Var)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    names.sort_by(|x, y| x.0.cmp(&y.0));
    let params = names.into_iter().map(|(name, var)| Param { decay: Model::decays(&name), name, var }).collect();
    AdamW::new(params, wd).map_err(|e| format!("optimizer: {e}"))
}

/// Rows right-padded with PAD: (ids, targets, weights, b, t); weight 1 for each target at index >= from.
pub fn padded(rows: &[(Vec<u32>, usize)]) -> (Vec<u32>, Vec<u32>, Vec<f32>, usize, usize) {
    let t = rows.iter().map(|r| r.0.len()).max().unwrap_or(2).saturating_sub(1).max(1);
    let b = rows.len();
    let (mut ids, mut tg, mut wt) = (Vec::with_capacity(b * t), Vec::with_capacity(b * t), Vec::with_capacity(b * t));
    for (s, from) in rows {
        for p in 0..t {
            let real = p + 1 < s.len();
            ids.push(if real { s[p] } else { PAD });
            tg.push(if real { s[p + 1] } else { 0 });
            wt.push(if real && p + 1 >= *from { 1f32 } else { 0f32 });
        }
    }
    (ids, tg, wt, b, t)
}

/// One learning step on `rows` at `lr` (see the top of the file). Returns the loss before the step.
pub fn learn_step(model: &Model, opt: &mut AdamW, dev: &Device, rows: &[(Vec<u32>, usize)], lr: f64, clip: f64) -> Result<f32, String> {
    let e = |e: candle_core::Error| format!("learn step: {e}");
    let (ids, tg, wt, b, t) = padded(rows);
    let ids = Tensor::from_vec(ids, (b, t), dev).map_err(e)?;
    let tg = Tensor::from_vec(tg, (b, t), dev).map_err(e)?;
    let wt = Tensor::from_vec(wt, b * t, dev).map_err(e)?;
    let (loss, _, grads) = if dev.is_cuda() && matches!(model.cfg.kind, Kind::D | Kind::A) {
        model.loss_and_grads_a_w(&ids, &tg, Some(&wt), &Mm { bf16: crate::bf16_on() })
    } else {
        model.loss_and_grads_autograd(&ids, &tg, Some(&wt), false)
    }
    .map_err(e)?;
    let loss = loss.to_dtype(candle_core::DType::F32).and_then(|l| l.to_scalar::<f32>()).map_err(e)?;
    if !loss.is_finite() {
        return Err(format!("learn step: the loss is {loss}"));
    }
    let gn = opt.grad_norm(&grads).map_err(e)?;
    opt.step(&grads, lr, clip_scale(gn, clip)).map_err(e)?;
    Ok(loss)
}

/// (nats, scored tokens, bytes) of `rows` under `model` (inference mode while scoring).
pub fn score(model: &Model, dev: &Device, rows: &[(Vec<u32>, usize)], byte_len: &[u32], batch_tokens: usize) -> Result<(f64, u64, u64), String> {
    let seqs: Vec<Vec<u32>> = rows.iter().map(|r| r.0.clone()).collect();
    let from: Vec<usize> = rows.iter().map(|r| r.1).collect();
    model.set_inference(true);
    let nll = crate::probe::token_nll(model, dev, &seqs, &from, batch_tokens).map_err(|e| format!("score: {e}"));
    model.set_inference(false);
    let nll = nll?;
    let (mut nats, mut toks, mut bytes) = (0f64, 0u64, 0u64);
    for (i, v) in nll.iter().enumerate() {
        nats += v.iter().map(|&x| x as f64).sum::<f64>();
        toks += v.len() as u64;
        let s = &seqs[i];
        bytes += s[from[i].max(1)..].iter().map(|&t| byte_len.get(t as usize).copied().unwrap_or(0) as u64).sum::<u64>();
    }
    Ok((nats, toks, bytes))
}

/// The prequential run over `batches` in order (see the top of the file); `frozen` (the same starting weights, never
/// trained) gives the static score, None skips it (stat = NaN). `on_step` sees each batch's numbers as they come.
pub fn prequential(
    model: &Model,
    varmap: &VarMap,
    frozen: Option<&Model>,
    dev: &Device,
    batches: &[Vec<(Vec<u32>, usize)>],
    byte_len: &[u32],
    o: &PreqOpts,
    on_step: &mut dyn FnMut(&PreqStep),
) -> Result<Vec<PreqStep>, String> {
    let mut opt = optimizer(varmap, o.wd)?;
    let mut out = Vec::with_capacity(batches.len());
    for (step, rows) in batches.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let (preq, tokens, bytes) = score(model, dev, rows, byte_len, o.batch_tokens)?;
        let stat = match frozen {
            Some(f) => score(f, dev, rows, byte_len, o.batch_tokens)?.0,
            None => f64::NAN,
        };
        let lr = if o.warmup > 0 && step < o.warmup { o.lr * (step + 1) as f64 / o.warmup as f64 } else { o.lr };
        let loss = learn_step(model, &mut opt, dev, rows, lr, o.clip)?;
        let s = PreqStep { step, tokens, bytes, preq, stat, loss };
        on_step(&s);
        out.push(s);
    }
    Ok(out)
}

/// Bits per byte of a nats total.
pub fn bpb(nats: f64, bytes: u64) -> f64 {
    nats / std::f64::consts::LN_2 / bytes.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{byte_bpe, tiny};
    use ava_data::bpe::EOS;

    fn rows_of(text: &str, copies: usize) -> Vec<Vec<(Vec<u32>, usize)>> {
        let mut s = vec![EOS];
        s.extend(text.bytes().map(|b| b as u32));
        s.push(EOS);
        (0..copies).map(|_| vec![(s.clone(), 1), (s[..s.len() / 2].to_vec(), 1)]).collect()
    }

    #[test]
    fn learning_on_the_fly_pays_less_than_the_frozen_copy() {
        let bpe = byte_bpe();
        let bl = crate::eval::byte_lengths_n(&bpe, 258);
        let (_c, vm, model) = tiny(Kind::D, 21);
        let (_c2, vm2, frozen) = tiny(Kind::D, 21); // the same fixed weights
        let _ = &vm2;
        let dev = Device::Cpu;
        let text = "Ava reads the same sentence again and again.";
        let batches = rows_of(text, 6);
        let o = PreqOpts { lr: 1e-2, warmup: 0, wd: 0.0, clip: 1.0, batch_tokens: 1 << 16 };
        let mut seen = 0;
        let st = prequential(&model, &vm, Some(&frozen), &dev, &batches, &bl, &o, &mut |_| seen += 1).unwrap();
        assert_eq!((st.len(), seen), (6, 6));
        // before any step both weights are the same, so the first batch costs the same
        assert_eq!(st[0].preq, st[0].stat);
        // the frozen copy pays the same for every copy; she pays less and less
        assert!(st.iter().all(|s| (s.stat - st[0].stat).abs() < 1e-9));
        assert!(st[5].preq < st[0].preq * 0.95, "{:?}", st.iter().map(|s| s.preq).collect::<Vec<_>>());
        let (p, s): (f64, f64) = (st.iter().map(|x| x.preq).sum(), st.iter().map(|x| x.stat).sum());
        assert!(p < s);
        // the text's bytes + the closing EOS, and the first half of the row again (bytes only)
        let n = text.len() as u64;
        assert_eq!(st[0].bytes, (n + 1) + ((n + 2) / 2 - 1));
        assert!(bpb(p, st.iter().map(|x| x.bytes).sum()) > 0.0);
    }

    #[test]
    fn at_lr_zero_nothing_is_learned_and_the_scores_agree() {
        let bl = crate::eval::byte_lengths_n(&byte_bpe(), 258);
        let (_c, vm, model) = tiny(Kind::D, 22);
        let (_c2, _vm2, frozen) = tiny(Kind::D, 22);
        let batches = rows_of("abc abc abc", 3);
        let o = PreqOpts { lr: 0.0, warmup: 0, wd: 0.1, clip: 1.0, batch_tokens: 64 };
        let st = prequential(&model, &vm, Some(&frozen), &Device::Cpu, &batches, &bl, &o, &mut |_| {}).unwrap();
        assert!(st.iter().all(|s| s.preq == s.stat), "{st:?}");
    }
}
