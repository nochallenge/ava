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
        let logits = crate::eval::score_logits(model, &Tensor::from_vec(ids, (b, t), dev)?)?; // [b·t, V] (v52: the switch)
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
/// v42: `Pmi` scores NLL(choice | prompt) − NLL(choice | PMI_PROMPT), the same scored tokens after a bare neutral
/// prompt: minus the pointwise mutual information of question and answer. A small model's choice NLL mostly says how
/// common the answer string is; PMI takes that out, so what is left is what the question adds. It costs one extra
/// batched pass over the distinct choices (`grade` does it; `normed` alone cannot).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    Total,
    Byte,
    Pmi,
}

impl Norm {
    pub fn parse(s: &str) -> Result<Norm, String> {
        match s {
            "total" => Ok(Norm::Total),
            "byte" => Ok(Norm::Byte),
            "pmi" => Ok(Norm::Pmi),
            _ => Err(format!("unknown norm {s:?} (total, byte, pmi)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Norm::Total => "total",
            Norm::Byte => "byte",
            Norm::Pmi => "pmi",
        }
    }
}

/// v42: the neutral prompt of `Norm::Pmi` (her taught format puts every answer right after "Answer:").
pub const PMI_PROMPT: &str = "Answer:";

/// v42: the base sequence of every scored sequence (`seqs`, `from` as item_seqs makes them): [EOS], the neutral
/// prompt's tokens `prompt`, then exactly the same scored tokens `s[from..]` (the choice and its closing EOS).
/// Identical base sequences are kept once (many items share choices like "yes" or "no"). Returns (the distinct base
/// sequences, where scoring starts in each, and for every input sequence the index of its base sequence).
pub fn base_seqs(prompt: &[u32], seqs: &[Vec<u32>], from: &[usize]) -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>) {
    let mut head = vec![EOS];
    head.extend_from_slice(prompt);
    let mut seen: std::collections::HashMap<Vec<u32>, usize> = std::collections::HashMap::new();
    let (mut out, mut out_from, mut map) = (Vec::new(), Vec::new(), Vec::with_capacity(seqs.len()));
    for (s, &f) in seqs.iter().zip(from) {
        let mut b = head.clone();
        b.extend_from_slice(&s[f.min(s.len())..]);
        let next = out.len();
        let i = *seen.entry(b.clone()).or_insert(next);
        if i == next {
            out.push(b);
            out_from.push(head.len());
        }
        map.push(i);
    }
    (out, out_from, map)
}

/// v42: PMI scores (lower is better): each sequence's NLL minus the NLL of its base sequence (`map` from base_seqs).
pub fn pmi_scores(nll: &[(f64, usize)], base_nll: &[(f64, usize)], map: &[usize]) -> Vec<f64> {
    nll.iter().zip(map).map(|(x, &j)| x.0 - base_nll[j].0).collect()
}

/// The bytes of each sequence's scored part `s[from..]` (`bytes`: per-token byte lengths, `eval::byte_lengths_n`).
pub fn scored_bytes(bytes: &[u32], seqs: &[Vec<u32>], from: &[usize]) -> Vec<f64> {
    seqs.iter()
        .zip(from)
        .map(|(s, &f)| s[f.min(s.len())..].iter().map(|&t| bytes.get(t as usize).copied().unwrap_or(0) as f64).sum())
        .collect()
}

/// Scores from NLLs under `norm` (`Byte`: NLL / max(bytes, 1)). `Pmi` needs the base pass: use `pmi_scores`
/// (`grade` does); asking `normed` for it is a bug and panics.
pub fn normed(nll: &[(f64, usize)], bytes: &[f64], norm: Norm) -> Vec<f64> {
    match norm {
        Norm::Total => nll.iter().map(|x| x.0).collect(),
        Norm::Byte => nll.iter().zip(bytes).map(|(x, &b)| x.0 / b.max(1.0)).collect(),
        Norm::Pmi => panic!("Norm::Pmi needs the base pass: use mc::pmi_scores (mc::grade does)"),
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
    let score = if norm == Norm::Pmi {
        // one extra batched pass: every distinct choice after the bare neutral prompt
        let pt = bpe.encode(PMI_PROMPT);
        let pt = if glance { insert_marks(&pt) } else { pt };
        let (bs, bf, map) = base_seqs(&pt, &seqs, &from);
        let base = scored_nll(model, dev, &bs, &bf, batch).map_err(|e| format!("score (pmi base): {e}"))?;
        pmi_scores(&nll, &base, &map)
    } else {
        let bytes = scored_bytes(&crate::eval::byte_lengths_n(bpe, model.cfg.vocab), &seqs, &from);
        normed(&nll, &bytes, norm)
    };
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

// ------------------------------------------------------------------------------------- F9 feeling of knowing first

/// F9: her feeling of knowing on each item's question before she sees any choice: the C8 probe at the prompt's last
/// token ([EOS] prompt, the position that guesses the answer's first token). Prompts are sorted by length and packed
/// into batches of at most `batch` tokens, right-padded with PAD (a causal model: padding after a position cannot
/// change it). None for a model trained without --confidence.
pub fn fok(model: &Model, dev: &Device, bpe: &Bpe, items: &[Item], glance: bool, batch: usize) -> Result<Option<Vec<f32>>, String> {
    let prompts: Vec<Vec<u32>> = items
        .iter()
        .map(|it| {
            let (s, f) = item_seqs(bpe, it, glance);
            s.first().map(|s| s[..f[0]].to_vec()).ok_or_else(|| format!("item {} has no choices", it.id))
        })
        .collect::<Result<_, _>>()?;
    let mut order: Vec<usize> = (0..prompts.len()).collect();
    order.sort_by_key(|&i| prompts[i].len());
    let mut out = vec![0f32; prompts.len()];
    let mut k = 0;
    while k < order.len() {
        let mut end = k + 1;
        while end < order.len() && (end - k + 1) * prompts[order[end]].len() <= batch.max(1) {
            end += 1;
        }
        let chunk = &order[k..end];
        let t = prompts[chunk[chunk.len() - 1]].len();
        let pr = &prompts;
        let ids: Vec<u32> = chunk.iter().flat_map(|&i| (0..t).map(move |p| pr[i].get(p).copied().unwrap_or(PAD))).collect();
        let ids = Tensor::from_vec(ids, (chunk.len(), t), dev).map_err(|e| format!("fok ids: {e}"))?;
        let Some(c) = model.feeling_of_knowing(&ids).map_err(|e| format!("fok: {e}"))? else {
            return Ok(None);
        };
        let c: Vec<f32> = c.to_vec1().map_err(|e| format!("fok: {e}"))?;
        for (r, &i) in chunk.iter().enumerate() {
            out[i] = c[r * t + prompts[i].len() - 1];
        }
        k = end;
    }
    Ok(Some(out))
}

/// (right, wrong, abstained) when she answers only with a feeling of knowing >= thr.
pub fn with_fok(g: &[Graded], fok: &[f32], thr: f32) -> (usize, usize, usize) {
    let (mut r, mut w, mut a) = (0, 0, 0);
    for (x, &c) in g.iter().zip(fok) {
        if c < thr {
            a += 1;
        } else if x.right {
            r += 1;
        } else {
            w += 1;
        }
    }
    (r, w, a)
}

/// The share of (right, wrong) pairs in which the right item had the higher feeling of knowing (ties count half):
/// 0.5 is no knowing of knowing, 1 is perfect (the area under the ROC curve).
pub fn fok_auc(g: &[Graded], fok: &[f32]) -> f64 {
    let (r, w): (Vec<f32>, Vec<f32>) = (g.iter().zip(fok).filter(|x| x.0.right).map(|x| *x.1).collect(), g.iter().zip(fok).filter(|x| !x.0.right).map(|x| *x.1).collect());
    if r.is_empty() || w.is_empty() {
        return f64::NAN;
    }
    let mut s = 0.0;
    for &a in &r {
        for &b in &w {
            s += if a > b { 1.0 } else if a == b { 0.5 } else { 0.0 };
        }
    }
    s / (r.len() * w.len()) as f64
}

// ------------------------------------------------------------------------------- C19 several minds, A6 study first

/// C19: how sure a mind is of its own pick: softmax over the choices of −score (scores are NLL-like, lower better).
pub fn sureness(g: &Graded) -> f32 {
    let lo = g.nll.iter().copied().fold(f64::INFINITY, f64::min);
    let z: f64 = g.nll.iter().map(|s| (lo - s).exp()).sum();
    ((lo - g.nll[g.pick]).exp() / z) as f32
}

/// C19: each voter's weight, ln((K − 1)·c / (1 − c)) (ava_parts::debate's rule).
fn vote_weight(c: f32, k: usize) -> f64 {
    let c = c.clamp(1e-4, 1.0 - 1e-4) as f64;
    ((k - 1) as f64 * c / (1.0 - c)).ln()
}

/// C19: several minds' grades of the same items → the group's. Each mind's sureness weights its vote; `rounds`
/// debate rounds first (ava_parts::debate::debate_round: a mind moves only when the others' weighted support for
/// another answer beats its own). The group's pick is the weighted vote; it is right only when the right choice's
/// support is strictly the largest; margin = the winner's support − the runner-up's; `nll` = −support.
pub fn group(minds: &[Vec<Graded>], rights: &[usize], rounds: usize) -> Vec<Graded> {
    (0..rights.len())
        .map(|i| {
            let k = minds[0][i].nll.len();
            let conf: Vec<f32> = minds.iter().map(|m| sureness(&m[i])).collect();
            let mut ans: Vec<usize> = minds.iter().map(|m| m[i].pick).collect();
            for _ in 0..rounds {
                ans = ava_parts::debate::debate_round(&ans, &conf, k);
            }
            let mut s = vec![0f64; k];
            for (&a, &c) in ans.iter().zip(&conf) {
                s[a] += vote_weight(c, k);
            }
            let neg: Vec<f64> = s.iter().map(|x| -x).collect();
            let mut g = grade_one(&neg, rights[i]);
            g.pick = ava_parts::debate::weighted_vote(&ans, &conf, k);
            g
        })
        .collect()
}

/// A6: study the prompt before answering. `steps` plain gradient steps of size `lr` (ava_parts::ttt::adapt) on the
/// next-token loss of `prompt` (each token predicted from the ones before it), for `vars` only. Returns the snapshot
/// to restore after answering, so nothing from one question stays for the next.
pub fn study(model: &Model, dev: &Device, vars: &[candle_core::Var], prompt: &[u32], steps: usize, lr: f64) -> Result<Vec<Tensor>, String> {
    if prompt.len() < 2 || steps == 0 {
        return ava_parts::ttt::snapshot(vars).map_err(|e| e.to_string());
    }
    let n = prompt.len() - 1;
    let ids = Tensor::from_vec(prompt[..n].to_vec(), (1, n), dev).map_err(|e| e.to_string())?;
    let tg = Tensor::from_vec(prompt[1..].to_vec(), n, dev).map_err(|e| e.to_string())?;
    let loss = || -> candle_core::Result<Tensor> {
        let lg = model.forward(&ids)?;
        let v = lg.dim(D::Minus1)?;
        crate::lif::cross_entropy_rows(&lg.reshape((n, v))?, &tg)?.mean_all()
    };
    ava_parts::ttt::adapt(vars, &loss, steps, lr).map_err(|e| format!("study: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gr(nll: &[f64], right: usize) -> Graded {
        grade_one(nll, right)
    }

    /// Sureness is the softmax of −score at the pick; one mind is itself; three minds out-vote one, and a sure
    /// mind out-weighs two unsure ones; the group's pick is the part's weighted vote.
    #[test]
    fn several_minds_vote_by_how_sure_they_are() {
        let g = gr(&[1.0, 2.0, 3.0], 0);
        let want = 1.0 / (1.0 + (-1f64).exp() + (-2f64).exp());
        assert!((sureness(&g) as f64 - want).abs() < 1e-6);
        let a = vec![gr(&[1.0, 2.0, 3.0], 0), gr(&[3.0, 1.0, 2.0], 0)];
        let one = group(&[a.clone()], &[0, 0], 0);
        assert_eq!(one.iter().map(|x| (x.pick, x.right)).collect::<Vec<_>>(), vec![(0, true), (1, false)]);
        // two minds pick 1 mildly, one picks 0 very surely: the sure one wins
        let mild = gr(&[1.05, 1.0, 1.1], 0);
        let sure = gr(&[0.0, 9.0, 9.0], 0);
        let gs = group(&[vec![mild.clone()], vec![mild.clone()], vec![sure.clone()]], &[0], 0);
        assert_eq!((gs[0].pick, gs[0].right), (0, true));
        // three equally sure minds, two say 1: the majority wins
        let p1 = gr(&[2.0, 0.0, 2.0], 1);
        let p0 = gr(&[0.0, 2.0, 2.0], 1);
        let gs = group(&[vec![p1.clone()], vec![p1.clone()], vec![p0.clone()]], &[1], 0);
        assert_eq!((gs[0].pick, gs[0].right), (1, true));
        assert!(gs[0].margin > 0.0);
        // a debate round moves the lone dissenter to the majority, so the vote stays and the margin grows
        let gd = group(&[vec![p1.clone()], vec![p1.clone()], vec![p0]], &[1], 1);
        assert_eq!(gd[0].pick, 1);
        assert!(gd[0].margin > gs[0].margin);
    }

    /// A6: studying lowers the prompt's own loss, moves only the chosen weights, and restoring gives the exact
    /// weights back.
    #[test]
    fn studying_the_prompt_helps_and_is_undone() {
        use candle_nn::{VarBuilder, VarMap};
        let cfg = crate::model::Config::with_kind(crate::model::Kind::D, 258, 32, 2, 64);
        let vm = VarMap::new();
        let m = Model::new(cfg, VarBuilder::from_varmap(&vm, candle_core::DType::F32, &Device::Cpu)).unwrap();
        let all: Vec<(String, candle_core::Var)> = vm.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let vars: Vec<candle_core::Var> = all.iter().filter(|(k, _)| k.starts_with("b1.mlp")).map(|(_, v)| v.clone()).collect();
        assert!(!vars.is_empty());
        let before: Vec<Tensor> = all.iter().map(|(_, v)| v.as_tensor().copy().unwrap()).collect();
        let prompt: Vec<u32> = (0..30).map(|i| (i * 7 % 50) as u32 + 3).collect();
        let loss = |m: &Model| -> f32 {
            let n = prompt.len() - 1;
            let ids = Tensor::from_vec(prompt[..n].to_vec(), (1, n), &Device::Cpu).unwrap();
            let tg = Tensor::from_vec(prompt[1..].to_vec(), n, &Device::Cpu).unwrap();
            let lg = m.forward(&ids).unwrap().reshape((n, 258)).unwrap();
            crate::lif::cross_entropy_rows(&lg, &tg).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap()
        };
        let l0 = loss(&m);
        let saved = study(&m, &Device::Cpu, &vars, &prompt, 5, 0.05).unwrap();
        let l1 = loss(&m);
        assert!(l1 < l0, "studying should lower the prompt's loss: {l0} -> {l1}");
        let d = |a: &Tensor, b: &Tensor| (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        let mut moved = 0;
        for ((k, v), b) in all.iter().zip(&before) {
            if k.starts_with("b1.mlp") {
                moved += (d(v.as_tensor(), b) > 0.0) as usize;
            } else {
                assert_eq!(d(v.as_tensor(), b), 0.0, "{k} must not move");
            }
        }
        assert!(moved > 0);
        ava_parts::ttt::restore(&vars, &saved).unwrap();
        for ((k, v), b) in all.iter().zip(&before) {
            assert_eq!(d(v.as_tensor(), b), 0.0, "{k} restored");
        }
        assert_eq!(loss(&m), l0);
    }

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
        assert!(Norm::parse("bits").is_err());
    }

    #[test]
    fn pmi_takes_out_how_common_the_answer_is() {
        // choice 0 is right but rare (high NLL anywhere); choice 1 is a common string (low NLL anywhere)
        let nll = [(5.0, 2), (3.0, 1)];
        let base = [(9.0, 2), (2.5, 1)];
        assert_eq!(Norm::parse("pmi"), Ok(Norm::Pmi));
        assert_eq!(Norm::Pmi.name(), "pmi");
        assert_eq!(grade_one(&normed(&nll, &[2.0, 1.0], Norm::Total), 0).right, false);
        let s = pmi_scores(&nll, &base, &[0, 1]);
        assert_eq!(s, vec![-4.0, 0.5]); // the question makes choice 0 far likelier, choice 1 a bit less likely
        let g = grade_one(&s, 0);
        assert!(g.right && g.pick == 0 && (g.margin - 4.5).abs() < 1e-12);
        // equal question effects tie (wrong), whatever the strings' base rates
        assert!(!grade_one(&pmi_scores(&[(7.0, 1), (2.0, 1)], &[(8.0, 1), (3.0, 1)], &[0, 1]), 0).right);
    }

    #[test]
    fn pmi_base_sequences_score_the_same_tokens_once() {
        // two items, [EOS] prompt choice [EOS]; the choice " yes" (tokens 7 8) appears in both
        let seqs = vec![vec![EOS, 1, 2, 7, 8, EOS], vec![EOS, 1, 2, 9, EOS], vec![EOS, 3, 7, 8, EOS], vec![EOS, 3, 5, EOS]];
        let from = vec![3, 3, 2, 2];
        let (bs, bf, map) = base_seqs(&[40, 41], &seqs, &from);
        assert_eq!(bs, vec![vec![EOS, 40, 41, 7, 8, EOS], vec![EOS, 40, 41, 9, EOS], vec![EOS, 40, 41, 5, EOS]]);
        assert_eq!(bf, vec![3, 3, 3]);
        assert_eq!(map, vec![0, 1, 0, 2]);
        // every base sequence scores exactly its original's scored tokens
        for (i, s) in seqs.iter().enumerate() {
            assert_eq!(&bs[map[i]][bf[map[i]]..], &s[from[i]..]);
        }
        let s = pmi_scores(&[(1.0, 3), (2.0, 2), (3.0, 3), (4.0, 2)], &[(0.5, 3), (1.0, 2), (1.5, 2)], &map);
        assert_eq!(s, vec![0.5, 1.0, 2.5, 2.5]);
    }

    #[test]
    #[should_panic(expected = "base pass")]
    fn normed_refuses_pmi_without_its_base_pass() {
        normed(&[(1.0, 1)], &[1.0], Norm::Pmi);
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

    /// F9: the feeling of knowing is read at the question's last token, the same whether the prompts go in one batch
    /// or one at a time (padding after a position cannot change it), equals the probe on a lone forward of that prompt,
    /// and is None for a model without the C8 probe. with_fok and fok_auc count as they say.
    #[test]
    fn feeling_of_knowing_reads_the_question_before_the_choices() {
        use crate::arch::Arch;
        let bpe = crate::testkit::byte_bpe();
        let items: Vec<Item> = ["What is two and two?", "Name a red fruit.", "Is ice cold?", "Why?"]
            .iter()
            .enumerate()
            .map(|(k, p)| Item { id: format!("q{k}"), kind: String::new(), bin: String::new(), prompt: p.to_string(), choices: vec!["yes".into(), "no".into()], right: k % 2 })
            .collect();
        let cfg = crate::model::Config::with_kind(crate::model::Kind::D, 258, 32, 2, 64).with_arch(Arch::from_flags("--confidence 0.5").unwrap()).unwrap();
        let (_, m) = crate::state::tests::build_awake(cfg, 5);
        let all = fok(&m, &Device::Cpu, &bpe, &items, false, 1 << 16).unwrap().expect("the probe is there");
        let lone = fok(&m, &Device::Cpu, &bpe, &items, false, 1).unwrap().unwrap();
        for (k, it) in items.iter().enumerate() {
            let (s, f) = item_seqs(&bpe, it, false);
            let p = &s[0][..f[0]];
            let c: Vec<f32> = m.feeling_of_knowing(&Tensor::from_vec(p.to_vec(), (1, p.len()), &Device::Cpu).unwrap()).unwrap().unwrap().to_vec1().unwrap();
            assert!((all[k] - c[p.len() - 1]).abs() < 1e-6 && (lone[k] - c[p.len() - 1]).abs() < 1e-6, "{k}: {} {} {}", all[k], lone[k], c[p.len() - 1]);
            assert!(all[k] > 0.0 && all[k] < 1.0);
        }
        let plain = Model::new(crate::model::Config::with_kind(crate::model::Kind::D, 258, 32, 2, 64), candle_nn::VarBuilder::zeros(candle_core::DType::F32, &Device::Cpu)).unwrap();
        assert!(fok(&plain, &Device::Cpu, &bpe, &items, false, 1 << 16).unwrap().is_none());
        let g = vec![grade_one(&[0.0, 1.0], 0), grade_one(&[0.0, 1.0], 1), grade_one(&[1.0, 0.0], 1)];
        assert_eq!(with_fok(&g, &[0.9, 0.2, 0.6], 0.5), (2, 0, 1));
        assert_eq!(fok_auc(&g, &[0.9, 0.2, 0.6]), 1.0);
        assert_eq!(fok_auc(&g, &[0.1, 0.8, 0.6]), 0.0);
    }
}
