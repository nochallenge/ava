// TESTING / DESIGN ONLY — ava_brain::eval: validation bits per byte and activity statistics.
//
// Bits per byte over a set of scored targets:
//     BPB = (Σ cross-entropy in nats over scored targets) / ln 2 / (Σ bytes of those target tokens)
// A token's bytes = bpe.token_bytes(id).len(); EOS counts as 1 byte (it stands for the newline
// between documents); PAD targets are skipped entirely (neither nats nor bytes nor count).
// The per-row cross-entropy is computed here in f64 from the logits (log-sum-exp with the row max
// subtracted), not taken from Graph::cross_entropy (which returns a mean in f32).
//
// Evaluation reads windows of ctx + 1 tokens in order (ShardSet::iter_sequential), takes inputs =
// w[..ctx] and targets = w[1..], marks PAD targets as IGNORE, and stops exactly when `eval_tokens`
// targets have been scored (targets past the limit inside the last window are marked IGNORE too).
// Windows are batched (forward only) and every batch is independent of the others, so the result
// does not depend on the batch size up to float rounding of the logits.
//
// Activity: for every SpikeMem the mean spike rate, and for every SparseMLP the kWTA active fraction
// before and after the relu, each averaged over the scored positions only.

use crate::model::{Forward, Model};
use ava_data::bpe::{Bpe, EOS, PAD};
use ava_grad::{Graph, Params, Tensor, IGNORE};

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

/// −log softmax(row)[target] in f64.
pub fn row_nll(row: &[f32], target: usize) -> f64 {
    let mx = row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x)) as f64;
    let s: f64 = row.iter().map(|&x| (x as f64 - mx).exp()).sum();
    mx + s.ln() - row[target] as f64
}

/// Running sums for nats/token and bits/byte.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BpbAcc {
    pub nats: f64,
    pub bytes: u64,
    pub tokens: u64,
}

impl BpbAcc {
    /// Score every row of `logits` ([rows, vocab], row-major) whose target is neither IGNORE nor PAD.
    pub fn add_rows(&mut self, logits: &[f32], vocab: usize, targets: &[u32], byte_len: &[u32]) {
        assert_eq!(logits.len(), targets.len() * vocab, "add_rows: {} logits for {} rows × {vocab}", logits.len(), targets.len());
        for (r, &t) in targets.iter().enumerate() {
            if t == IGNORE || t == PAD {
                continue;
            }
            let t = t as usize;
            assert!(t < vocab, "add_rows: target {t} out of range for vocab {vocab}");
            self.nats += row_nll(&logits[r * vocab..(r + 1) * vocab], t);
            self.bytes += byte_len[t] as u64;
            self.tokens += 1;
        }
    }

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
/// at most `*remaining` targets are kept (the rest become IGNORE) and `*remaining` is decreased by the
/// number kept.
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

/// Training batch from sampler windows: inputs and targets (PAD → IGNORE, never present in practice).
pub fn train_io(windows: &[Vec<u32>]) -> (Vec<u32>, Vec<u32>) {
    let mut ids = Vec::new();
    let mut tg = Vec::new();
    for w in windows {
        let ctx = w.len() - 1;
        ids.extend_from_slice(&w[..ctx]);
        tg.extend(w[1..].iter().map(|&t| if t == PAD { IGNORE } else { t }));
    }
    (ids, tg)
}

/// Mean of one per-layer activity: (non-zero entries over scored rows, entries over scored rows).
#[derive(Clone, Debug, Default)]
pub struct Activity {
    pub nonzero: Vec<u64>,
    pub total: Vec<u64>,
}

impl Activity {
    pub fn new(layers: usize) -> Activity {
        Activity { nonzero: vec![0; layers], total: vec![0; layers] }
    }

    /// Count the non-zero entries of `t` (rows of its last dim, in target order) on rows with mask true.
    pub fn add(&mut self, layer: usize, t: &Tensor, mask: &[bool]) {
        let d = t.last_dim();
        assert_eq!(t.numel(), mask.len() * d, "Activity::add: tensor {:?} vs {} rows", t.shape, mask.len());
        for (row, &m) in t.data.chunks(d).zip(mask) {
            if m {
                self.nonzero[layer] += row.iter().filter(|&&v| v != 0.0).count() as u64;
                self.total[layer] += d as u64;
            }
        }
    }

    pub fn fractions(&self) -> Vec<f64> {
        self.nonzero.iter().zip(&self.total).map(|(&a, &b)| if b == 0 { f64::NAN } else { a as f64 / b as f64 }).collect()
    }
}

#[derive(Clone, Debug)]
pub struct EvalResult {
    pub acc: BpbAcc,
    /// Mean spike rate per SpikeMem layer.
    pub spike_rates: Vec<f64>,
    /// kWTA active fraction per SparseMLP layer before relu (1/8 by construction) and after relu.
    pub kwta_pre: Vec<f64>,
    pub kwta_post: Vec<f64>,
    pub windows: usize,
}

/// Accumulate a forward pass's activity on the scored rows.
pub fn add_activity(g: &Graph, f: &Forward, mask: &[bool], spikes: &mut Activity, pre: &mut Activity, post: &mut Activity) {
    for (i, &s) in f.spikes.iter().enumerate() {
        spikes.add(i, g.value(s), mask);
    }
    for (i, &k) in f.kwta_pre.iter().enumerate() {
        pre.add(i, g.value(k), mask);
    }
    for (i, &k) in f.kwta_post.iter().enumerate() {
        post.add(i, g.value(k), mask);
    }
}

/// Evaluate on `windows` (each ctx + 1 tokens, e.g. ShardSet::iter_sequential(ctx)) until
/// `eval_tokens` targets have been scored or the windows run out. Forward only, `batch` windows at a
/// time, `threads` worker threads.
pub fn evaluate<I>(model: &Model, params: &Params, windows: I, batch: usize, eval_tokens: u64, byte_len: &[u32], threads: usize) -> EvalResult
where
    I: IntoIterator<Item = Vec<u32>>,
{
    let layers = model.cfg.layers;
    let vocab = model.cfg.vocab;
    let (mut spikes, mut pre, mut post) = (Activity::new(layers), Activity::new(layers), Activity::new(layers));
    let mut acc = BpbAcc::default();
    let mut remaining = eval_tokens;
    let mut it = windows.into_iter();
    let mut n_windows = 0usize;
    while remaining > 0 {
        let mut ids = Vec::new();
        let mut tg = Vec::new();
        let mut b = 0usize;
        let mut t = 0usize;
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
        let mut g = Graph::new(params);
        g.set_threads(threads);
        let f = model.forward(&mut g, &ids, b, t);
        acc.add_rows(&g.value(f.logits).data, vocab, &tg, byte_len);
        let mask: Vec<bool> = tg.iter().map(|&x| x != IGNORE).collect();
        add_activity(&g, &f, &mask, &mut spikes, &mut pre, &mut post);
    }
    let keep = |a: &Activity, used: bool| if used { a.fractions() } else { Vec::new() };
    EvalResult {
        acc,
        spike_rates: keep(&spikes, model.cfg.kind.has_spike_mem()),
        kwta_pre: keep(&pre, model.cfg.kind.has_sparse_mlp()),
        kwta_post: keep(&post, model.cfg.kind.has_sparse_mlp()),
        windows: n_windows,
    }
}

/// "0.1234,0.2345" (or "-" when empty).
pub fn fmt_list(v: &[f64]) -> String {
    if v.is_empty() {
        "-".to_string()
    } else {
        v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",")
    }
}
