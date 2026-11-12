// TESTING / DESIGN ONLY — v42 (Gr): Ava's greedy continuations of a file of prompts, batched (for fluency_gen).
//
// The file format is the one pythia_eval gen reads and writes and evalsets' fluency_score reads: rows
// `id<TAB>...<TAB>prompt` (the prompt in column `col`), text escaped as \\ \t \n; `#` lines and empty lines skipped.
// Each prompt is read as EOS + its tokens (with length marks for a glance model), and she writes greedily (argmax)
// until EOS or `max_new` tokens; her own length marks are not printed.
//
// Batched: `state::generate` sorts the prompts by length, reads every equal-length run in one pass and then one
// position per new token with carried state; a group holds as many rows as fit `batch_tokens` (prompt + new tokens
// per row). Kinds without a batched state (F, G, R) re-read the whole text for every token instead, rows of equal
// length together (the same tokens; slower).

use crate::model::Model;
use ava_data::bpe::{Bpe, EOS};
use ava_data::numbers::{insert_marks, strip_marks};
use candle_core::{Device, Result, Tensor, D};
use std::time::Instant;

/// `\\`, tab and newline written as `\\`, `\t`, `\n` (evalsets' esc).
pub fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\t', "\\t").replace('\n', "\\n")
}

/// The inverse of `escape` (a backslash before any other character keeps that character).
pub fn unescape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            o.push(c);
            continue;
        }
        match it.next() {
            Some('t') => o.push('\t'),
            Some('n') => o.push('\n'),
            Some(x) => o.push(x),
            None => o.push('\\'),
        }
    }
    o
}

/// (id, prompt) of every row with a column `col` (unescaped); `#` lines and empty lines are skipped.
pub fn read_prompts(text: &str, col: usize) -> Vec<(String, String)> {
    text.lines()
        .map(|l| l.trim_end_matches('\r'))
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Some((f[0].to_string(), unescape(f.get(col)?)))
        })
        .collect()
}

/// What a run cost: prompts, tokens written, tokens through the model, seconds.
#[derive(Clone, Debug, Default)]
pub struct GenStats {
    pub prompts: usize,
    pub new_tokens: usize,
    pub seconds: f64,
    pub batched_state: bool,
}

/// The token ids of a prompt: EOS + its tokens, with length marks for a glance model.
pub fn prompt_ids(model: &Model, bpe: &Bpe, prompt: &str) -> Vec<u32> {
    let t = bpe.encode(prompt);
    let mut ids = vec![EOS];
    ids.extend(if model.cfg.glance { insert_marks(&t) } else { t });
    ids
}

fn argmax(logits: &Tensor) -> Result<u32> {
    logits.argmax(D::Minus1)?.to_scalar::<u32>()
}

/// Greedy token continuations of `prompts` (token ids), at most `max_new` tokens each, ending before EOS.
pub fn greedy_ids(model: &Model, dev: &Device, prompts: &[Vec<u32>], max_new: usize, batch_tokens: usize) -> Result<(Vec<Vec<u32>>, bool)> {
    let longest = prompts.iter().map(Vec::len).max().unwrap_or(1);
    let rows = (batch_tokens / (longest + max_new)).max(1);
    if model.state(1).is_ok() {
        let out = crate::state::generate(model, dev, prompts, max_new, rows, &mut |_, l| argmax(l), &|_| false)?;
        return Ok((out, true));
    }
    Ok((reforward(model, dev, prompts, max_new, batch_tokens)?, false))
}

/// The slow way (kinds without a batched state): the whole text again for every new token, equal lengths together.
fn reforward(model: &Model, dev: &Device, prompts: &[Vec<u32>], max_new: usize, batch_tokens: usize) -> Result<Vec<Vec<u32>>> {
    let mut out = vec![Vec::new(); prompts.len()];
    let mut by_len: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (i, p) in prompts.iter().enumerate() {
        by_len.entry(p.len()).or_default().push(i);
    }
    for (len, idx) in by_len {
        let per = (batch_tokens / (len + max_new)).max(1);
        for chunk in idx.chunks(per) {
            let mut seqs: Vec<Vec<u32>> = chunk.iter().map(|&i| prompts[i].clone()).collect();
            let mut live = vec![true; chunk.len()];
            for _ in 0..max_new {
                let t = seqs[0].len();
                let flat: Vec<u32> = seqs.iter().flatten().copied().collect();
                let logits = model.forward(&Tensor::from_vec(flat, (seqs.len(), t), dev)?)?; // [B·T, V]
                let at: Vec<u32> = (0..seqs.len()).map(|r| (r * t + t - 1) as u32).collect();
                let best = logits.index_select(&Tensor::from_vec(at, seqs.len(), dev)?, 0)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
                for (r, &b) in best.iter().enumerate() {
                    if live[r] && b == EOS {
                        live[r] = false;
                    }
                    if live[r] {
                        out[chunk[r]].push(b);
                    }
                    seqs[r].push(if live[r] { b } else { EOS });
                }
                if !live.iter().any(|&l| l) {
                    break;
                }
            }
        }
    }
    Ok(out)
}

/// Her greedy continuation of each prompt as text (her length marks not printed), and what it cost.
pub fn greedy_texts(model: &Model, dev: &Device, bpe: &Bpe, prompts: &[String], max_new: usize, batch_tokens: usize) -> Result<(Vec<String>, GenStats)> {
    let t0 = Instant::now();
    let ids: Vec<Vec<u32>> = prompts.iter().map(|p| prompt_ids(model, bpe, p)).collect();
    let (gen, batched) = greedy_ids(model, dev, &ids, max_new, batch_tokens)?;
    let texts = gen.iter().map(|g| bpe.decode(&strip_marks(g))).collect();
    let st = GenStats { prompts: prompts.len(), new_tokens: gen.iter().map(Vec::len).sum(), seconds: t0.elapsed().as_secs_f64(), batched_state: batched };
    Ok((texts, st))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Kind;
    use crate::testkit::{byte_bpe, tiny};

    #[test]
    fn escaping_round_trips_and_rows_read() {
        for s in ["plain", "a\tb\nc\\d", "\\t is not a tab", "end\\"] {
            assert_eq!(unescape(&escape(s)), s);
        }
        let file = "# note\nx1\tcat\tAnimals:\\n\n\nx2\tdog\tSay \\\"hi\\\"\\tnow\r\nx3\tshort\n";
        let rows = read_prompts(file, 2);
        assert_eq!(rows, vec![("x1".to_string(), "Animals:\n".to_string()), ("x2".to_string(), "Say \"hi\"\tnow".to_string())]);
        assert_eq!(read_prompts(file, 1).len(), 3);
    }

    /// The carried-state path writes the same tokens as re-reading the whole text, whatever the batch size; a kind
    /// without a batched state (G) takes the slow path and agrees with itself across batch sizes.
    #[test]
    fn greedy_is_the_same_batched_or_not_and_by_either_path() {
        let bpe = byte_bpe();
        let prompts: Vec<String> = ["Animals:", "Here is a list of fruits:", "Colors", "Name as many tools as you can.", "Birds:"].iter().map(|s| s.to_string()).collect();
        for kind in [Kind::D, Kind::S, Kind::A] {
            let (_, _, m) = tiny(kind, 3);
            let ids: Vec<Vec<u32>> = prompts.iter().map(|p| prompt_ids(&m, &bpe, p)).collect();
            let (one, batched) = greedy_ids(&m, &Device::Cpu, &ids, 12, 1).unwrap();
            assert!(batched, "{kind:?}");
            let (many, _) = greedy_ids(&m, &Device::Cpu, &ids, 12, 4096).unwrap();
            assert_eq!(one, many, "{kind:?}: batch size changed the text");
            let slow = reforward(&m, &Device::Cpu, &ids, 12, 4096).unwrap();
            assert_eq!(one, slow, "{kind:?}: carried state and re-reading disagree");
            assert!(one.iter().all(|g| g.len() <= 12 && !g.contains(&EOS)));
        }
        let (_, _, g) = tiny(Kind::G, 3);
        let ids: Vec<Vec<u32>> = prompts.iter().map(|p| prompt_ids(&g, &bpe, p)).collect();
        let (a, batched) = greedy_ids(&g, &Device::Cpu, &ids, 8, 1).unwrap();
        assert!(!batched);
        assert_eq!(a, greedy_ids(&g, &Device::Cpu, &ids, 8, 4096).unwrap().0);
        let (texts, st) = greedy_texts(&g, &Device::Cpu, &bpe, &prompts, 8, 256).unwrap();
        assert_eq!((texts.len(), st.prompts), (5, 5));
    }

}
