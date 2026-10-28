// TESTING / DESIGN ONLY — v40: multiple-choice items and their scoring, shared by mc_eval, qa_eval and grow_gpu.
//
// An item is a prompt and 2+ choices, one right. Scored as v37's qa_eval does: each choice is a sequence
// [EOS] prompt choice [EOS] and only the choice's tokens and the closing EOS are scored (total NLL, no length
// normalisation). Her pick is the lowest; she is right only when the right choice is strictly lowest (a tie is
// wrong). Margin = second-lowest − lowest; with a threshold tau she answers only when margin >= tau.
// Glance models (vocab 4112) get their length marks on the prompt and on each choice separately (the same rule
// in teaching, in scoring and in reward learning; a run of digits never spans the two, as an answer starts with
// a space).
//
// Item files (fact and sleep tests, written by the data crate): TSV `id kind bin right prompt c0 c1 ...`, with
// \t, \n and \\ escaped in every text field; the choices are scored with a leading space.

use ava_data::bpe::{Bpe, EOS, PAD};
use ava_data::numbers::insert_marks;
use candle_core::{Device, Tensor, D};
use std::path::Path;

use crate::model::Model;

/// One multiple-choice item.
#[derive(Clone, Debug)]
pub struct Item {
    pub id: String,
    pub kind: String,
    pub bin: String,
    pub prompt: String,
    pub choices: Vec<String>,
    pub right: usize,
}

/// `\t`, `\n` and `\\` back to tab, newline and backslash.
pub fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('\\') => out.push('\\'),
                Some(o) => {
                    out.push('\\');
                    out.push(o);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Read an item TSV (a first line starting with `id\t` is a header).
pub fn read_items(path: &Path) -> Result<Vec<Item>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        if line.trim().is_empty() || (ln == 0 && line.starts_with("id\t")) {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 7 {
            return Err(format!("{} line {}: {} fields, need id kind bin right prompt and 2+ choices", path.display(), ln + 1, f.len()));
        }
        let right: usize = f[3].parse().map_err(|_| format!("{} line {}: bad right {:?}", path.display(), ln + 1, f[3]))?;
        let choices: Vec<String> = f[5..].iter().map(|c| unescape(c)).collect();
        if right >= choices.len() {
            return Err(format!("{} line {}: right {right} but {} choices", path.display(), ln + 1, choices.len()));
        }
        out.push(Item { id: f[0].into(), kind: f[1].into(), bin: f[2].into(), prompt: unescape(f[4]), choices, right });
    }
    Ok(out)
}

/// v37's question sets as items (kind = the set name, bin = "").
pub fn from_qa(name: &str, qs: &[crate::qa::Mc], passage: bool) -> Vec<Item> {
    qs.iter()
        .enumerate()
        .map(|(i, q)| Item {
            id: format!("{name}-{i}"),
            kind: name.into(),
            bin: String::new(),
            prompt: crate::qa::prompt_text(if passage { &q.support } else { "" }, &q.question),
            choices: q.choices.iter().map(|c| c.trim().to_string()).collect(),
            right: q.right,
        })
        .collect()
}

/// The token sequences of one item: [EOS] prompt choice [EOS] per choice, and where scoring starts (the prompt's
/// length). With `glance`, marks go on the prompt and on each choice.
pub fn item_seqs(bpe: &Bpe, it: &Item, glance: bool) -> (Vec<Vec<u32>>, Vec<usize>) {
    let mut p = vec![EOS];
    let pt = bpe.encode(&it.prompt);
    p.extend(if glance { insert_marks(&pt) } else { pt });
    let (mut seqs, mut from) = (Vec::new(), Vec::new());
    for c in &it.choices {
        let ct = bpe.encode(&format!(" {}", c.trim()));
        let mut s = p.clone();
        s.extend(if glance { insert_marks(&ct) } else { ct });
        s.push(EOS);
        from.push(p.len());
        seqs.push(s);
    }
    (seqs, from)
}

/// (total NLL in nats, scored tokens) of every sequence; targets at index >= from[i] are scored. Sequences are
/// sorted by length and packed into batches of at most `batch` tokens (rows · longest row), right-padded with PAD.
pub fn scored_nll(model: &Model, dev: &Device, seqs: &[Vec<u32>], from: &[usize], batch: usize) -> candle_core::Result<Vec<(f64, usize)>> {
    let mut out = vec![(0.0f64, 0usize); seqs.len()];
    let mut order: Vec<usize> = (0..seqs.len()).collect();
    order.sort_by_key(|&i| seqs[i].len());
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for i in order {
        if !cur.is_empty() && (cur.len() + 1) * seqs[i].len() > batch.max(1) {
            chunks.push(std::mem::take(&mut cur));
        }
        cur.push(i);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    for chunk in &chunks {
        let t = chunk.iter().map(|&i| seqs[i].len()).max().unwrap_or(2).saturating_sub(1).max(1);
        let b = chunk.len();
        let (mut ids, mut tg, mut keep) = (Vec::with_capacity(b * t), Vec::with_capacity(b * t), Vec::with_capacity(b * t));
        for &i in chunk {
            let s = &seqs[i];
            for p in 0..t {
                let real = p + 1 < s.len();
                ids.push(if real { s[p] } else { PAD });
                tg.push(if real { s[p + 1] } else { 0 });
                keep.push(if real && p + 1 >= from[i] { 1f32 } else { 0f32 });
            }
        }
        let logits = model.forward(&Tensor::from_vec(ids, (b, t), dev)?)?; // [b·t, V]
        let lp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        drop(logits);
        let picked = lp.gather(&Tensor::from_vec(tg, (b * t, 1), dev)?, 1)?.squeeze(1)?;
        let nll: Vec<f32> = picked.neg()?.mul(&Tensor::from_vec(keep, b * t, dev)?)?.reshape((b, t))?.sum(1)?.to_vec1()?;
        for (r, &i) in chunk.iter().enumerate() {
            out[i] = (nll[r] as f64, seqs[i].len() - from[i]);
        }
    }
    Ok(out)
}

/// v41: how a choice's NLL becomes its score (lower is better). `Total` is v37's rule (the NLL of the choice's
/// tokens and the closing EOS); `Byte` divides it by the choice's bytes (marks are 0 bytes; the closing EOS counts
/// as 1, as eval::byte_lengths_n has it), so a long right answer is not penalised for its length (v40: total NLL put
/// every model below chance on OpenBookQA).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    Total,
    Byte,
}

impl Norm {
    pub fn parse(s: &str) -> Result<Norm, String> {
        match s {
            "total" => Ok(Norm::Total),
            "byte" => Ok(Norm::Byte),
            _ => Err(format!("unknown norm {s:?} (total, byte)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Norm::Total => "total",
            Norm::Byte => "byte",
        }
    }
}

/// The bytes of each sequence's scored part `s[from..]` (`bytes`: per-token byte lengths, `eval::byte_lengths_n`).
pub fn scored_bytes(bytes: &[u32], seqs: &[Vec<u32>], from: &[usize]) -> Vec<f64> {
    seqs.iter()
        .zip(from)
        .map(|(s, &f)| s[f.min(s.len())..].iter().map(|&t| bytes.get(t as usize).copied().unwrap_or(0) as f64).sum())
        .collect()
}

/// Scores from NLLs under `norm` (`Byte`: NLL / max(bytes, 1)).
pub fn normed(nll: &[(f64, usize)], bytes: &[f64], norm: Norm) -> Vec<f64> {
    match norm {
        Norm::Total => nll.iter().map(|x| x.0).collect(),
        Norm::Byte => nll.iter().zip(bytes).map(|(x, &b)| x.0 / b.max(1.0)).collect(),
    }
}

/// One graded item.
#[derive(Clone, Debug)]
pub struct Graded {
    pub right: bool,
    pub margin: f64,
    pub pick: usize,
    /// every choice's score (its NLL under the grading norm; lower is better)
    pub nll: Vec<f64>,
}

/// Grade a set of items in one batched pass, scoring each choice under `norm`.
pub fn grade(model: &Model, dev: &Device, bpe: &Bpe, items: &[Item], glance: bool, batch: usize, norm: Norm) -> Result<Vec<Graded>, String> {
    let (mut seqs, mut from) = (Vec::new(), Vec::new());
    for it in items {
        let (s, f) = item_seqs(bpe, it, glance);
        seqs.extend(s);
        from.extend(f);
    }
    let nll = scored_nll(model, dev, &seqs, &from, batch).map_err(|e| format!("score: {e}"))?;
    let bytes = scored_bytes(&crate::eval::byte_lengths_n(bpe, model.cfg.vocab), &seqs, &from);
    let score = normed(&nll, &bytes, norm);
    let mut out = Vec::with_capacity(items.len());
    let mut k = 0;
    for it in items {
        let tot: Vec<f64> = score[k..k + it.choices.len()].to_vec();
        k += it.choices.len();
        out.push(grade_one(&tot, it.right));
    }
    Ok(out)
}

/// Pick, margin and rightness from the choices' total NLLs.
pub fn grade_one(tot: &[f64], right: usize) -> Graded {
    let mut idx: Vec<usize> = (0..tot.len()).collect();
    idx.sort_by(|&a, &b| tot[a].total_cmp(&tot[b]));
    let margin = if idx.len() > 1 { tot[idx[1]] - tot[idx[0]] } else { f64::INFINITY };
    let right_ok = (0..tot.len()).filter(|&j| j != right).all(|j| tot[right] < tot[j]);
    Graded { right: right_ok, margin, pick: idx[0], nll: tot.to_vec() }
}

/// (right, wrong, abstained) when she answers only with margin >= tau.
pub fn with_tau(g: &[Graded], tau: f64) -> (usize, usize, usize) {
    let (mut r, mut w, mut a) = (0, 0, 0);
    for x in g {
        if x.margin < tau {
            a += 1;
        } else if x.right {
            r += 1;
        } else {
            w += 1;
        }
    }
    (r, w, a)
}

/// The tau with the most right − wrong (ties: the smaller tau); candidates are 0 and every margin.
pub fn best_tau(g: &[Graded]) -> f64 {
    let mut cands: Vec<f64> = g.iter().map(|x| x.margin).filter(|m| m.is_finite()).collect();
    cands.push(0.0);
    cands.sort_by(f64::total_cmp);
    let mut best = (i64::MIN, 0.0);
    for &t in &cands {
        let (r, w, _) = with_tau(g, t);
        let s = r as i64 - w as i64;
        if s > best.0 {
            best = (s, t);
        }
    }
    best.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_norm_divides_by_the_choice_bytes() {
        // token t has t bytes; the scored part starts at `from`
        let bytes: Vec<u32> = (0..10).collect();
        let seqs = vec![vec![1, 2, 3], vec![1, 9, 9, 0]];
        let b = scored_bytes(&bytes, &seqs, &[1, 1]);
        assert_eq!(b, vec![5.0, 18.0]);
        let nll = [(10.0, 2), (18.0, 3)];
        assert_eq!(normed(&nll, &b, Norm::Total), vec![10.0, 18.0]);
        assert_eq!(normed(&nll, &b, Norm::Byte), vec![2.0, 1.0]);
        // the longer choice loses on total NLL but wins per byte
        assert_eq!(grade_one(&normed(&nll, &b, Norm::Total), 1).right, false);
        assert_eq!(grade_one(&normed(&nll, &b, Norm::Byte), 1).right, true);
        assert_eq!(normed(&[(3.0, 1)], &[0.0], Norm::Byte), vec![3.0]); // 0 bytes divides by 1
        assert_eq!(Norm::parse("byte"), Ok(Norm::Byte));
        assert!(Norm::parse("pmi").is_err());
    }

    #[test]
    fn unescape_round_trips_the_three_escapes() {
        assert_eq!(unescape(r"a\tb\nc\\d"), "a\tb\nc\\d");
        assert_eq!(unescape(r"x\q"), r"x\q");
    }

    #[test]
    fn grading_counts_ties_as_wrong() {
        let g = grade_one(&[1.0, 2.0, 3.0], 0);
        assert!(g.right && g.pick == 0 && (g.margin - 1.0).abs() < 1e-12);
        assert!(!grade_one(&[1.0, 1.0, 3.0], 0).right);
        let gs = vec![grade_one(&[1.0, 5.0], 0), grade_one(&[1.0, 1.5], 1), grade_one(&[2.0, 2.1], 0)];
        assert_eq!(with_tau(&gs, 0.0), (2, 1, 0));
        assert_eq!(with_tau(&gs, 1.0), (1, 0, 2));
        assert_eq!(best_tau(&gs), 0.0); // 2 - 1 at tau 0 is not beaten
        let gs = vec![grade_one(&[1.0, 5.0], 0), grade_one(&[1.0, 1.5], 1), grade_one(&[2.0, 3.0], 0)];
        assert_eq!(best_tau(&gs), 1.0); // abstaining on the 0.5-margin wrong one: 2 - 0
    }
}
