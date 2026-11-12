// TESTING / DESIGN ONLY — ava_gpu::sae::probe: A13 linear probes over her states, and labels made from the text alone.
//
// A probe is multinomial logistic regression on standardized features (each feature's mean and spread from the
// training rows): minibatch Adam, a cosine learning rate, a small L2 penalty. It is scored on rows it never saw, against
// the majority-class baseline (the training rows' most common class, guessed for every test row); binary probes also
// get the ROC AUC of their score, which does not depend on a threshold.
//
// The labels (one per input position p of a window of ctx + 1 tokens; token p + 1 exists for every p):
//   next_is_digit  the next token of text (length marks skipped) begins with a digit (a length mark always does)
//   in_number      token p is a length mark or all digits, or a '.' or ',' with digit tokens on both sides
//   sentence_end   token p ends in . ! or ? (after its own closing quotes and brackets), is not in a number, and what
//                  follows, past any closing quotes and brackets (" ' ) ] ” ’, also as tokens of their own), is the
//                  end of the document or white space (white space at the end of token p itself counts too)
//   in_quotes      after token p the text is inside double quotes: a straight quote toggles, “ opens, ” closes (read
//                  byte by byte across tokens, so a curly quote split over byte tokens counts); the count starts afresh
//                  at the window's start and at every end of document (she sees no further)

use super::Rng;
use ava_data::bpe::{Bpe, EOS, PAD};
use ava_data::numbers::is_mark;

/// The text labels' names, in the order `text_labels` returns them.
pub const TEXT_TASKS: [&str; 4] = ["next_is_digit", "in_number", "sentence_end", "in_quotes"];

fn bytes_of<'a>(bpe: &'a Bpe, t: u32) -> &'a [u8] {
    if is_mark(t) || t == EOS || t == PAD || t as usize >= bpe.vocab_size() {
        &[]
    } else {
        bpe.token_bytes(t)
    }
}

fn all_digits(b: &[u8]) -> bool {
    !b.is_empty() && b.iter().all(|c| c.is_ascii_digit())
}

const CLOSE_DQ: &[u8] = "”".as_bytes();
const CLOSE_SQ: &[u8] = "’".as_bytes();

/// The byte length of the closing quote or bracket at the start of `b` (" ' ) ] ” ’), 0 for none.
fn closer_at(b: &[u8]) -> usize {
    match b.first() {
        Some(b'"' | b'\'' | b')' | b']') => 1,
        _ if b.starts_with(CLOSE_DQ) || b.starts_with(CLOSE_SQ) => 3,
        _ => 0,
    }
}

/// Whether the text after token p is a gap: past any closing quotes and brackets (bytes read across tokens, so a
/// curly quote split over byte tokens still counts), an end of document or white space. A length mark is not a gap
/// (digits follow); the window running out first is not one either (unknown).
fn gap_after(bpe: &Bpe, w: &[u32], p: usize) -> bool {
    let mut buf: Vec<u8> = Vec::new();
    let mut i = 0;
    for &q in w.iter().skip(p + 1).take(16) {
        if q == EOS {
            return true;
        }
        if is_mark(q) {
            return false;
        }
        buf.extend_from_slice(bytes_of(bpe, q));
        while i < buf.len() {
            let c = closer_at(&buf[i..]);
            if c > 0 {
                i += c;
            } else if buf.len() - i < 3 && (CLOSE_DQ.starts_with(&buf[i..]) || CLOSE_SQ.starts_with(&buf[i..])) {
                break; // the start of a curly quote: read the next token
            } else {
                return buf[i].is_ascii_whitespace();
            }
        }
    }
    false
}

/// The four text labels (0/1) of the first `ctx` positions of window `w` (ctx + 1 tokens).
pub fn text_labels(bpe: &Bpe, w: &[u32], ctx: usize) -> [Vec<u32>; 4] {
    assert!(w.len() > ctx, "a window holds ctx + 1 tokens");
    let digit_tok = |t: u32| all_digits(bytes_of(bpe, t));
    let mut out: [Vec<u32>; 4] = Default::default();
    let mut inside = false;
    // the last two bytes read (for curly quotes split over tokens); reset with `inside`
    let mut prev = [0u8; 2];
    for p in 0..ctx {
        let t = w[p];
        // next_is_digit: skip marks (a mark is always followed by digits)
        let nxt = w[p + 1];
        let next_digit = is_mark(nxt) || bytes_of(bpe, nxt).first().is_some_and(|c| c.is_ascii_digit());
        // in_number
        let b = bytes_of(bpe, t);
        let prev_digit = (0..p).rev().map(|q| w[q]).find(|&q| !is_mark(q)).is_some_and(digit_tok);
        let next_dig_tok = (p + 1..w.len()).map(|q| w[q]).find(|&q| !is_mark(q)).is_some_and(digit_tok);
        let in_num = is_mark(t) || all_digits(b) || ((b == b"." || b == b",") && prev_digit && next_dig_tok);
        // sentence_end: the token's own closing quotes, brackets and white space come off its end first (white space
        // there is the gap itself); otherwise the gap must follow (gap_after skips closing quotes and brackets)
        let mut e = b;
        let mut own_gap = false;
        while let Some((&last, rest)) = e.split_last() {
            if matches!(last, b'"' | b'\'' | b')' | b']') {
                e = rest;
            } else if last.is_ascii_whitespace() {
                own_gap = true;
                e = rest;
            } else if e.ends_with(CLOSE_DQ) || e.ends_with(CLOSE_SQ) {
                e = &e[..e.len() - 3];
            } else {
                break;
            }
        }
        let ends = matches!(e.last(), Some(b'.' | b'!' | b'?')) && !in_num && (own_gap || gap_after(bpe, w, p));
        // in_quotes (bytes read across tokens, so a curly quote split over byte tokens is still seen)
        if t == EOS {
            inside = false;
            prev = [0, 0];
        }
        for &c in b {
            if c == b'"' {
                inside = !inside;
            } else if prev == [0xE2, 0x80] && c == 0x9C {
                inside = true;
            } else if prev == [0xE2, 0x80] && c == 0x9D {
                inside = false;
            }
            prev = [prev[1], c];
        }
        out[0].push(next_digit as u32);
        out[1].push(in_num as u32);
        out[2].push(ends as u32);
        out[3].push(inside as u32);
    }
    out
}

/// A trained linear probe: standardization, then softmax(W x + b).
#[derive(Clone, Debug)]
pub struct Probe {
    pub d: usize,
    pub c: usize,
    mu: Vec<f32>,
    sd: Vec<f32>,
    /// [c, d]
    w: Vec<f32>,
    b: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct ProbeCfg {
    pub epochs: usize,
    pub batch: usize,
    pub lr: f32,
    pub l2: f32,
    pub seed: u64,
}

impl Default for ProbeCfg {
    fn default() -> ProbeCfg {
        ProbeCfg { epochs: 20, batch: 1024, lr: 0.01, l2: 1e-4, seed: 1 }
    }
}

impl Probe {
    /// Class scores (logits) of one standardized row.
    fn logits(&self, z: &[f32], out: &mut [f32]) {
        for (c, o) in out.iter_mut().enumerate() {
            *o = self.b[c] + self.w[c * self.d..(c + 1) * self.d].iter().zip(z).map(|(a, b)| a * b).sum::<f32>();
        }
    }

    fn standardize(&self, x: &[f32], z: &mut [f32]) {
        for i in 0..self.d {
            z[i] = (x[i] - self.mu[i]) / self.sd[i];
        }
    }

    /// Fit on rows `x` (n × d) with labels `y` in 0..c (single-threaded: callers run many probes side by side).
    pub fn fit(x: &[f32], y: &[u32], d: usize, c: usize, cfg: &ProbeCfg) -> Probe {
        let n = y.len();
        assert_eq!(x.len(), n * d, "rows and labels");
        let mut mu = vec![0f64; d];
        for r in x.chunks_exact(d) {
            mu.iter_mut().zip(r).for_each(|(m, &v)| *m += v as f64);
        }
        mu.iter_mut().for_each(|m| *m /= n.max(1) as f64);
        let mut var = vec![0f64; d];
        for r in x.chunks_exact(d) {
            var.iter_mut().zip(r).zip(&mu).for_each(|((s, &v), m)| *s += (v as f64 - m).powi(2));
        }
        let sd: Vec<f32> = var.iter().map(|v| ((v / n.max(1) as f64).sqrt() as f32).max(1e-6)).collect();
        let mut p = Probe { d, c, mu: mu.iter().map(|&m| m as f32).collect(), sd, w: vec![0.0; c * d], b: vec![0.0; c] };
        let mut z: Vec<f32> = vec![0.0; n * d];
        for (zr, xr) in z.chunks_exact_mut(d).zip(x.chunks_exact(d)) {
            p.standardize(xr, zr);
        }
        let bs = cfg.batch.min(n).max(1);
        let steps_per = n.div_ceil(bs);
        let total = (steps_per * cfg.epochs).max(1);
        let (mut mw, mut vw) = (vec![0f32; c * d], vec![0f32; c * d]);
        let (mut mb, mut vb) = (vec![0f32; c], vec![0f32; c]);
        let mut rng = Rng::new(cfg.seed);
        let mut order: Vec<u32> = (0..n as u32).collect();
        let (mut gw, mut gb) = (vec![0f32; c * d], vec![0f32; c]);
        let mut lg = vec![0f32; c];
        let mut t = 0usize;
        for _ in 0..cfg.epochs {
            rng.shuffle(&mut order);
            for chunk in order.chunks(bs) {
                gw.iter_mut().for_each(|g| *g = 0.0);
                gb.iter_mut().for_each(|g| *g = 0.0);
                for &r in chunk {
                    let zr = &z[r as usize * d..(r as usize + 1) * d];
                    p.logits(zr, &mut lg);
                    let mx = lg.iter().copied().fold(f32::MIN, f32::max);
                    let s: f32 = lg.iter_mut().map(|v| {
                        *v = (*v - mx).exp();
                        *v
                    }).sum();
                    for cc in 0..c {
                        let g = lg[cc] / s - (y[r as usize] as usize == cc) as u32 as f32;
                        gb[cc] += g;
                        gw[cc * d..(cc + 1) * d].iter_mut().zip(zr).for_each(|(a, &b)| *a += g * b);
                    }
                }
                let inv = 1.0 / chunk.len() as f32;
                t += 1;
                let lr = cfg.lr * 0.5 * (1.0 + (std::f32::consts::PI * (t - 1) as f32 / total as f32).cos());
                let (c1, c2) = (1.0 - 0.9f32.powi(t as i32), 1.0 - 0.999f32.powi(t as i32));
                for i in 0..c * d {
                    let g = gw[i] * inv + cfg.l2 * p.w[i];
                    mw[i] = 0.9 * mw[i] + 0.1 * g;
                    vw[i] = 0.999 * vw[i] + 0.001 * g * g;
                    p.w[i] -= lr * (mw[i] / c1) / ((vw[i] / c2).sqrt() + 1e-8);
                }
                for i in 0..c {
                    let g = gb[i] * inv;
                    mb[i] = 0.9 * mb[i] + 0.1 * g;
                    vb[i] = 0.999 * vb[i] + 0.001 * g * g;
                    p.b[i] -= lr * (mb[i] / c1) / ((vb[i] / c2).sqrt() + 1e-8);
                }
            }
        }
        p
    }

    /// The predicted class of each row of `x`, and (for 2 classes) the score of class 1 (its logit minus class 0's).
    pub fn predict(&self, x: &[f32]) -> (Vec<u32>, Vec<f32>) {
        let mut z = vec![0f32; self.d];
        let mut lg = vec![0f32; self.c];
        let mut cls = Vec::with_capacity(x.len() / self.d);
        let mut score = Vec::with_capacity(x.len() / self.d);
        for r in x.chunks_exact(self.d) {
            self.standardize(r, &mut z);
            self.logits(&z, &mut lg);
            let best = (0..self.c).max_by(|&a, &b| lg[a].total_cmp(&lg[b])).unwrap_or(0);
            cls.push(best as u32);
            score.push(if self.c == 2 { lg[1] - lg[0] } else { 0.0 });
        }
        (cls, score)
    }
}

/// ROC AUC of `score` for labels `y` (0/1), ties counted as half; None without both classes.
pub fn auc(score: &[f32], y: &[u32]) -> Option<f64> {
    let mut o: Vec<usize> = (0..y.len()).collect();
    o.sort_by(|&a, &b| score[a].total_cmp(&score[b]));
    let (mut rank_sum, mut pos) = (0f64, 0f64);
    let mut i = 0;
    while i < o.len() {
        let mut j = i;
        while j + 1 < o.len() && score[o[j + 1]] == score[o[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &q in &o[i..=j] {
            if y[q] == 1 {
                rank_sum += avg;
                pos += 1.0;
            }
        }
        i = j + 1;
    }
    let neg = y.len() as f64 - pos;
    if pos == 0.0 || neg == 0.0 {
        return None;
    }
    Some((rank_sum - pos * (pos + 1.0) / 2.0) / (pos * neg))
}

/// One probe's result on held-out rows.
#[derive(Clone, Debug)]
pub struct Score {
    pub acc: f64,
    /// the majority class of the training rows, guessed for every test row
    pub majority: f64,
    pub auc: Option<f64>,
    /// mean recall over the classes present in the test rows
    pub balanced: f64,
}

/// Fit on (xa, ya), score on (xb, yb).
pub fn fit_and_score(xa: &[f32], ya: &[u32], xb: &[f32], yb: &[u32], d: usize, c: usize, cfg: &ProbeCfg) -> Score {
    let p = Probe::fit(xa, ya, d, c, cfg);
    let (pred, sc) = p.predict(xb);
    let n = yb.len().max(1) as f64;
    let acc = pred.iter().zip(yb).filter(|(a, b)| a == b).count() as f64 / n;
    let mut counts = vec![0usize; c];
    ya.iter().for_each(|&y| counts[y as usize] += 1);
    let maj = (0..c).max_by_key(|&k| counts[k]).unwrap_or(0) as u32;
    let majority = yb.iter().filter(|&&y| y == maj).count() as f64 / n;
    let (mut hit, mut tot) = (vec![0usize; c], vec![0usize; c]);
    for (&p, &y) in pred.iter().zip(yb) {
        tot[y as usize] += 1;
        hit[y as usize] += (p == y) as usize;
    }
    let present: Vec<usize> = (0..c).filter(|&k| tot[k] > 0).collect();
    let balanced = present.iter().map(|&k| hit[k] as f64 / tot[k] as f64).sum::<f64>() / present.len().max(1) as f64;
    Score { acc, majority, auc: if c == 2 { auc(&sc, yb) } else { None }, balanced }
}
