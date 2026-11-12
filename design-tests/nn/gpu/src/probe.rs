// TESTING / DESIGN ONLY — probe (v42 Wave 4): the gpu side of the always-online learner (design-tests/nn/online).
//
// The learner (`ava_learner run`) calls commands and reads plain files back; this module is what they share:
// - gate_probe (its --gate-cmd, placeholders {old} {new} {out}): per-skill dev probe items scored on both brains, and
//   greedy answers to a fixed list of self-questions; rows `skill<TAB>name<TAB>old<TAB>new` (one per item) and
//   `self<TAB>index<TAB>old answer<TAB>new answer` (one per question, answers escaped as the learner unescapes).
// - think (its --dream-cmd, placeholders {model} {out}): free text from the live brain (dreams) and short answers to
//   given prompts (thinking back on recent episodes, or self-questions), blank-line separated. The learner keeps all of
//   it as dreamt, never as fact.
// - episode_surprise: each text of an episode file and its mean surprise (nats per token) under the live brain.
//
// PROBE FILES. Multiple-choice TSV (src/mc.rs) from dev halves only: every NAME_dev.tsv in a folder whose name starts
// with evalsets_v42, under the probe root (data/nn). A file or folder whose name contains sealed, test, spent or
// heldout is refused (`refused`), and the retired sets (evalsets_v42's first instruct, emotion and bias files, never
// scored) are left out. Each skill uses a small fixed sample: the `per_skill` items with the lowest FNV-1a hash of
// "skill/id". The same items come up at every gate, so old and new are paired and gates over time compare like with like.
//
// GENERATION (`generate`). Many prompts at once: each step is one forward pass over every unfinished row (sorted by
// length, packed by tokens, right-padded with PAD; every model here is causal, so padding never changes a row's logits),
// and each row takes its next token at its own last position with diary::pick (greedy at temperature <= 0, else
// temperature, top-k and one draw from the row's own rng). A row stops as diary::reply stops: at EOS (not kept), when
// the decoded reply starts a new question (diary::cut_reply), or after max_new tokens. The tests check it against
// diary::reply's token-by-token stream.
//
// SURPRISE (`token_nll`, `surprise`). A text is read as [EOS] text [EOS], the form sleep replays (grow_gpu's episodes),
// and every token after the first EOS is scored, the closing EOS too. Batched as mc::scored_nll (sorted by length,
// packed by tokens); the per-text sums equal scored_nll's (checked in the tests).

use crate::diary;
use crate::mc::{Graded, Item};
use crate::model::Model;
use ava_data::bpe::{Bpe, EOS, PAD};
use ava_data::numbers::strip_marks;
use ava_data::rng::SplitMix64;
use candle_core::{Device, Tensor, D};
use std::path::{Path, PathBuf};

/// Words that mark a file or folder no probe may read.
pub const REFUSED: [&str; 4] = ["sealed", "test", "spent", "heldout"];

/// True when the file's name or its folder's name contains sealed, test, spent or heldout (any case).
pub fn refused(p: &Path) -> bool {
    let name = |q: Option<&Path>| q.and_then(|q| q.file_name()).map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    let (file, dir) = (name(Some(p)), name(p.parent()));
    REFUSED.iter().any(|w| file.contains(w) || dir.contains(w))
}

/// Err when `refused`: a probe never reads a sealed half.
pub fn check_path(p: &Path, what: &str) -> Result<(), String> {
    if refused(p) {
        return Err(format!("{what} {}: a name with sealed, test, spent or heldout is never read by a probe", p.display()));
    }
    Ok(())
}

/// Sets that are retired unopened and never scored: (folder, family).
pub const RETIRED: [(&str, &str); 3] = [("evalsets_v42", "instruct"), ("evalsets_v42", "emotion"), ("evalsets_v42", "bias")];

/// The dev probe files under `root`: (skill, path) for every NAME_dev.tsv in a folder named evalsets_v42*, sorted by
/// skill; retired and refused ones are left out. Only names are read here, never a file's contents. Two folders holding
/// the same skill name is an error (the gate's rows are keyed by skill).
pub fn dev_probe_files(root: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let rd = std::fs::read_dir(root).map_err(|e| format!("probe root {}: {e}", root.display()))?;
    let mut dirs: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("evalsets_v42")))
        .collect();
    dirs.sort();
    let mut out: Vec<(String, PathBuf)> = Vec::new();
    for d in dirs {
        let folder = d.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let mut files: Vec<PathBuf> = std::fs::read_dir(&d).map_err(|e| format!("{}: {e}", d.display()))?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        files.sort();
        for f in files {
            let Some(skill) = f.file_name().and_then(|n| n.to_str()).and_then(|n| n.strip_suffix("_dev.tsv")).map(String::from) else { continue };
            if refused(&f) || RETIRED.iter().any(|(fo, fa)| *fo == folder && *fa == skill) {
                continue;
            }
            if out.iter().any(|(s, _)| *s == skill) {
                return Err(format!("skill {skill:?} is in two probe folders; pick the files with --items"));
            }
            out.push((skill, f));
        }
    }
    out.sort();
    Ok(out)
}

/// FNV-1a, 64 bit.
pub fn fnv(data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The fixed sample of a skill: the `n` items with the lowest FNV-1a of "skill/id" (ties by id), in that order.
pub fn pick_fixed(items: Vec<Item>, skill: &str, n: usize) -> Vec<Item> {
    let mut keyed: Vec<(u64, Item)> = items.into_iter().map(|it| (fnv(format!("{skill}/{}", it.id).as_bytes()), it)).collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.id.cmp(&b.1.id)));
    keyed.truncate(n);
    keyed.into_iter().map(|(_, it)| it).collect()
}

/// How one probe item is scored in a gate row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Score {
    /// 1 when the right choice is strictly best, else 0 (mc::grade_one's rule; a tie is wrong)
    Acc,
    /// the right choice's share of softmax(−score) over the choices: in [0, 1] like accuracy, but it moves before the
    /// pick flips, so a small paired sample sees a drop sooner
    Prob,
}

impl Score {
    pub fn parse(s: &str) -> Result<Score, String> {
        match s {
            "acc" => Ok(Score::Acc),
            "prob" => Ok(Score::Prob),
            _ => Err(format!("unknown score {s:?} (acc, prob)")),
        }
    }
}

/// One item's score under `s` (`right` = the item's right choice).
pub fn item_score(g: &Graded, right: usize, s: Score) -> f64 {
    match s {
        Score::Acc => f64::from(u8::from(g.right)),
        Score::Prob => {
            let mn = g.nll.iter().copied().fold(f64::INFINITY, f64::min);
            let z: f64 = g.nll.iter().map(|&x| (mn - x).exp()).sum();
            (mn - g.nll[right]).exp() / z
        }
    }
}

/// Gate rows `skill<TAB>name<TAB>old<TAB>new`, one per item, in item order.
pub fn skill_rows(skill: &str, items: &[Item], old: &[Graded], new: &[Graded], s: Score) -> Vec<String> {
    items
        .iter()
        .zip(old.iter().zip(new))
        .map(|(it, (a, b))| {
            let (x, y) = (item_score(a, it.right, s), item_score(b, it.right, s));
            match s {
                Score::Acc => format!("skill\t{skill}\t{x}\t{y}"),
                Score::Prob => format!("skill\t{skill}\t{x:.6}\t{y:.6}"),
            }
        })
        .collect()
}

/// Gate rows `self<TAB>index<TAB>old answer<TAB>new answer` (answers escaped: backslash, tab, newline, CR).
pub fn self_rows(old: &[String], new: &[String]) -> Vec<String> {
    old.iter().zip(new).enumerate().map(|(i, (a, b))| format!("self\t{i}\t{}\t{}", diary::escape(a), diary::escape(b))).collect()
}

/// The default self-questions of the identity gate. The learner keys its anchor by index, so this list only ever grows
/// at the end; never reorder or edit it.
pub const SELF_QUESTIONS: [&str; 8] = [
    "What is your name?",
    "Who are you?",
    "What are you?",
    "What do you like to do?",
    "How do you learn new things?",
    "Do you dream?",
    "What do you do when you do not know an answer?",
    "How old are you?",
];

/// Questions from a file: one per line, or `ava_learner quiz` output (a header `id<TAB>recall<TAB>question<TAB>answer`,
/// fields escaped), whose question column is taken. Empty lines are skipped.
pub fn read_questions(path: &Path) -> Result<Vec<String>, String> {
    check_path(path, "questions")?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty()).peekable();
    let quiz = lines.peek().is_some_and(|l| l.starts_with("id\trecall\tquestion"));
    if quiz {
        lines.next();
    }
    let qs: Vec<String> = lines
        .filter_map(|l| if quiz { l.split('\t').nth(2).map(diary::unescape) } else { Some(l.trim().to_string()) })
        .filter(|q| !q.trim().is_empty())
        .collect();
    if qs.is_empty() {
        return Err(format!("{}: no questions", path.display()));
    }
    Ok(qs)
}

/// Episodes of a file: blocks separated by blank lines; a file without blank lines has one per line (grow_gpu's rule).
pub fn read_episodes(path: &Path) -> Result<Vec<String>, String> {
    check_path(path, "episodes")?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(diary::grow_episodes(&text))
}

/// A generated text as one block of the blank-line separated output: trimmed, CR dropped, runs of blank lines made one
/// newline (so the learner's split on "\n\n" keeps it whole).
pub fn one_block(s: &str) -> String {
    let lines: Vec<&str> = s.split('\n').map(|l| l.trim_end_matches('\r').trim_end()).filter(|l| !l.trim().is_empty()).collect();
    lines.join("\n").trim().to_string()
}

/// The first half of a text's first sentence, on one line, at most `words` words (think's prompt for going on from a
/// heard text): cut mid-sentence, so what follows is hers (a whole sentence mostly made her stop at once).
pub fn opening(text: &str, words: usize) -> String {
    let flat = diary::one_line(text);
    let end = flat.find(['.', '!', '?']).map_or(flat.len(), |i| i + 1);
    let w: Vec<&str> = flat[..end].split_whitespace().collect();
    w[..w.len().div_ceil(2).min(words)].join(" ")
}

/// The batch budget (rows · longest row) the tools use with --cpu when none is given. The CPU forward of D keeps
/// about 2 MB per batch token at d384 (its delta-rule chunk loop), so the GPU default of 32768 would need some 60 GB;
/// 256 kept a whole gate_probe run under 1.1 GB.
pub const CPU_BATCH_TOKENS: usize = 256;

/// Generation settings.
#[derive(Clone, Copy, Debug)]
pub struct GenOpts {
    pub max_new: usize,
    /// <= 0: greedy
    pub temperature: f32,
    pub top_k: usize,
    /// rows · longest row per forward pass at most
    pub batch_tokens: usize,
}

/// Rows sorted by `len`, packed so that rows · longest row <= budget.
fn pack(rows: &[usize], len: &dyn Fn(usize) -> usize, budget: usize) -> Vec<Vec<usize>> {
    let mut order = rows.to_vec();
    order.sort_by_key(|&r| len(r));
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for r in order {
        if !cur.is_empty() && (cur.len() + 1) * len(r) > budget.max(1) {
            chunks.push(std::mem::take(&mut cur));
        }
        cur.push(r);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// Continuations of every prompt at once (see the top of the file). `rngs[r]` is row r's sampler state; `stop(tokens)`
/// ends a row after the token just added. Returns each row's tokens (EOS not included).
pub fn generate(model: &Model, dev: &Device, prompts: &[Vec<u32>], o: &GenOpts, rngs: &mut [SplitMix64], stop: &dyn Fn(&[u32]) -> bool) -> Result<Vec<Vec<u32>>, String> {
    if rngs.len() != prompts.len() {
        return Err(format!("generate: {} prompts but {} samplers", prompts.len(), rngs.len()));
    }
    if prompts.iter().any(Vec::is_empty) {
        return Err("generate: an empty prompt".into());
    }
    let e = |e: candle_core::Error| format!("generate: {e}");
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    let mut live: Vec<usize> = (0..prompts.len()).collect();
    for _ in 0..o.max_new {
        if live.is_empty() {
            break;
        }
        let len = |r: usize| prompts[r].len() + out[r].len();
        let mut next: Vec<(usize, u32)> = Vec::with_capacity(live.len());
        for chunk in pack(&live, &len, o.batch_tokens) {
            let t = chunk.iter().map(|&r| len(r)).max().unwrap_or(1);
            let b = chunk.len();
            let mut ids = Vec::with_capacity(b * t);
            for &r in &chunk {
                ids.extend_from_slice(&prompts[r]);
                ids.extend_from_slice(&out[r]);
                ids.resize(ids.len() + t - len(r), PAD);
            }
            let logits = model.forward(&Tensor::from_vec(ids, (b, t), dev).map_err(e)?).map_err(e)?; // [b·t, V]
            let at: Vec<u32> = chunk.iter().enumerate().map(|(i, &r)| (i * t + len(r) - 1) as u32).collect();
            let last = logits.index_select(&Tensor::from_vec(at, b, dev).map_err(e)?, 0).map_err(e)?; // [b, V]
            drop(logits);
            for (i, &r) in chunk.iter().enumerate() {
                let row = last.get(i).map_err(e)?;
                next.push((r, diary::pick(&row, o.temperature, o.top_k, &mut rngs[r]).map_err(e)?));
            }
        }
        for (r, tok) in next {
            if tok == EOS {
                live.retain(|&x| x != r);
                continue;
            }
            out[r].push(tok);
            if stop(&out[r]) {
                live.retain(|&x| x != r);
            }
        }
    }
    Ok(out)
}

/// Her replies to `questions` in her taught format (diary::prompt_ids, no memory), all at once: the tokens until EOS, a
/// new question or `o.max_new`, decoded and cut by diary::cut_reply (what ava_chat would show).
pub fn answers(model: &Model, dev: &Device, bpe: &Bpe, questions: &[String], o: &GenOpts, rngs: &mut [SplitMix64]) -> Result<Vec<String>, String> {
    let glance = model.cfg.glance;
    let prompts: Vec<Vec<u32>> = questions.iter().map(|q| diary::prompt_ids(bpe, glance, &[], q, 0).0).collect();
    let stop = |toks: &[u32]| diary::cut_reply(&bpe.decode(&strip_marks(toks))).1;
    let outs = generate(model, dev, &prompts, o, rngs, &stop)?;
    Ok(outs.iter().map(|t| diary::cut_reply(&bpe.decode(&strip_marks(t))).0).collect())
}

/// Per-token NLL (nats) of every sequence for the targets at index >= from[i] (target p is s[p], read after s[..p]).
/// Sequences sorted by length and packed into batches of at most `batch_tokens` tokens, right-padded with PAD.
pub fn token_nll(model: &Model, dev: &Device, seqs: &[Vec<u32>], from: &[usize], batch_tokens: usize) -> candle_core::Result<Vec<Vec<f32>>> {
    let mut out = vec![Vec::new(); seqs.len()];
    let rows: Vec<usize> = (0..seqs.len()).filter(|&i| seqs[i].len() >= 2).collect();
    for chunk in pack(&rows, &|i| seqs[i].len(), batch_tokens) {
        let t = chunk.iter().map(|&i| seqs[i].len()).max().unwrap_or(2) - 1;
        let b = chunk.len();
        let (mut ids, mut tg) = (Vec::with_capacity(b * t), Vec::with_capacity(b * t));
        for &i in &chunk {
            let s = &seqs[i];
            for p in 0..t {
                let real = p + 1 < s.len();
                ids.push(if real { s[p] } else { PAD });
                tg.push(if real { s[p + 1] } else { 0 });
            }
        }
        let logits = model.forward(&Tensor::from_vec(ids, (b, t), dev)?)?; // [b·t, V]
        let lp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        drop(logits);
        let nll: Vec<Vec<f32>> = lp.gather(&Tensor::from_vec(tg, (b * t, 1), dev)?, 1)?.neg()?.reshape((b, t))?.to_vec2()?;
        for (r, &i) in chunk.iter().enumerate() {
            let s = &seqs[i];
            out[i] = (0..t).filter(|&p| p + 1 < s.len() && p + 1 >= from[i]).map(|p| nll[r][p]).collect();
        }
    }
    Ok(out)
}

/// [EOS] text [EOS] for each text (length marks with the glance), scored from index 1.
pub fn episode_seqs(bpe: &Bpe, glance: bool, texts: &[String]) -> (Vec<Vec<u32>>, Vec<usize>) {
    let seqs: Vec<Vec<u32>> = texts
        .iter()
        .map(|t| {
            let mut s = vec![EOS];
            s.extend(diary::encode(bpe, glance, t));
            s.push(EOS);
            s
        })
        .collect();
    let from = vec![1; seqs.len()];
    (seqs, from)
}

/// Each text's mean surprise in nats per token and its scored tokens, under `model` as it is (no lesion).
pub fn surprise(model: &Model, dev: &Device, bpe: &Bpe, texts: &[String], batch_tokens: usize) -> Result<Vec<(f64, usize)>, String> {
    let (seqs, from) = episode_seqs(bpe, model.cfg.glance, texts);
    let nll = token_nll(model, dev, &seqs, &from, batch_tokens).map_err(|e| format!("surprise: {e}"))?;
    Ok(nll.iter().map(|v| (v.iter().map(|&x| x as f64).sum::<f64>() / v.len().max(1) as f64, v.len())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{byte_bpe, scratch, tiny};

    #[test]
    fn refused_names_and_the_dev_files() {
        assert!(refused(Path::new("data/nn/evalsets_v42/facts_sealed.tsv")));
        assert!(refused(Path::new("data/v41/sealed/sleep_facts.txt")));
        assert!(refused(Path::new("x/test_templates.txt")));
        assert!(refused(Path::new("x/spent/a.tsv")) && refused(Path::new("x/HeldOut_a.tsv")));
        assert!(!refused(Path::new("data/nn/evalsets_v42/facts_dev.tsv")));
        // only the file and its folder are judged, so a repo folder such as design-tests above them is fine
        assert!(!refused(Path::new("design-tests/nn/x/evalsets_v42/facts_dev.tsv")));
        assert!(check_path(Path::new("a/sealed/b.tsv"), "--items").unwrap_err().contains("never read"));

        let root = scratch("probe_root");
        for (dir, files) in [
            ("evalsets_v42", &["facts_dev.tsv", "facts_sealed.tsv", "bias_dev.tsv", "instruct_dev.tsv", "emotion_dev.tsv", "MANIFEST.tsv", "test_templates.txt"][..]),
            ("evalsets_v42_social", &["bias2_dev.tsv", "bias2_sealed.tsv"][..]),
            ("other", &["words_dev.tsv"][..]),
        ] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
            for f in files {
                std::fs::write(root.join(dir).join(f), "never read here\n").unwrap();
            }
        }
        let got = dev_probe_files(&root).unwrap();
        let names: Vec<&str> = got.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(names, vec!["bias2", "facts"]);
        std::fs::write(root.join("evalsets_v42_social").join("facts_dev.tsv"), "x\n").unwrap();
        assert!(dev_probe_files(&root).unwrap_err().contains("two probe folders"));
    }

    fn item(id: &str, right: usize) -> Item {
        Item { id: id.into(), kind: "k".into(), bin: "b".into(), prompt: "Q: x?\nA:".into(), choices: vec!["a".into(), "b".into()], right }
    }

    #[test]
    fn the_fixed_sample_is_the_same_whatever_the_order() {
        let items: Vec<Item> = (0..50).map(|i| item(&format!("f-{i:03}"), 0)).collect();
        let a = pick_fixed(items.clone(), "facts", 8);
        let mut rev = items.clone();
        rev.reverse();
        let b = pick_fixed(rev, "facts", 8);
        assert_eq!(a.iter().map(|x| &x.id).collect::<Vec<_>>(), b.iter().map(|x| &x.id).collect::<Vec<_>>());
        assert_eq!(a.len(), 8);
        // another skill name draws another sample
        let c = pick_fixed(items, "words", 8);
        assert_ne!(a.iter().map(|x| &x.id).collect::<Vec<_>>(), c.iter().map(|x| &x.id).collect::<Vec<_>>());
    }

    #[test]
    fn gate_rows_read_back_as_the_learner_reads_them() {
        let its = vec![item("a", 0), item("b", 1)];
        let old = vec![crate::mc::grade_one(&[1.0, 2.0], 0), crate::mc::grade_one(&[1.0, 2.0], 1)];
        let new = vec![crate::mc::grade_one(&[1.0, 2.0], 0), crate::mc::grade_one(&[3.0, 2.0], 1)];
        assert_eq!(skill_rows("facts", &its, &old, &new, Score::Acc), vec!["skill\tfacts\t1\t1", "skill\tfacts\t0\t1"]);
        let p = skill_rows("facts", &its, &old, &new, Score::Prob);
        let f: Vec<&str> = p[1].split('\t').collect();
        let (x, y): (f64, f64) = (f[2].parse().unwrap(), f[3].parse().unwrap());
        assert!((x - 1.0 / (1.0 + 1f64.exp())).abs() < 1e-6 && (y - 1.0 / (1.0 + (-1f64).exp())).abs() < 1e-6);
        let rows = self_rows(&["My name is Ava.\tok".into()], &["two\nlines \\ here".into()]);
        let f: Vec<&str> = rows[0].split('\t').collect();
        assert_eq!(f.len(), 4);
        assert_eq!((f[0], f[1]), ("self", "0"));
        assert_eq!(diary::unescape(f[2]), "My name is Ava.\tok");
        assert_eq!(diary::unescape(f[3]), "two\nlines \\ here");
    }

    #[test]
    fn questions_from_lines_or_from_a_quiz() {
        let d = scratch("probe_q");
        let a = d.join("self.txt");
        std::fs::write(&a, "What is your name?\n\nWho are you?\n").unwrap();
        assert_eq!(read_questions(&a).unwrap(), vec!["What is your name?", "Who are you?"]);
        let b = d.join("quiz.tsv");
        std::fs::write(&b, "id\trecall\tquestion\tanswer\nlesson/x#1\t0.400\tWhat is 7\\ttimes 8?\t56.\n").unwrap();
        assert_eq!(read_questions(&b).unwrap(), vec!["What is 7\ttimes 8?"]);
        assert_eq!(one_block("  a\r\n\n\nb  \n\n"), "a\nb");
        assert_eq!(opening("I saw a\nfox today. It ran.", 24), "I saw a");
        assert_eq!(opening("one two three four five six", 2), "one two");
        assert_eq!(opening("Hello.", 24), "Hello.");
        assert_eq!(opening("", 24), "");
    }

    #[test]
    fn batched_generation_matches_the_token_by_token_stream() {
        let bpe = byte_bpe();
        let (_cfg, _vm, model) = tiny(crate::model::Kind::D, 7);
        let dev = Device::Cpu;
        let qs: Vec<String> = ["What is your name?", "Hi", "Why is the sky blue on a clear day?", "2+2"].iter().map(|s| s.to_string()).collect();
        let o = GenOpts { max_new: 12, temperature: 0.0, top_k: 1, batch_tokens: 48 }; // small: several chunks per step
        let mut rngs: Vec<SplitMix64> = (0..qs.len()).map(|i| SplitMix64::new(i as u64)).collect();
        let got = answers(&model, &dev, &bpe, &qs, &o, &mut rngs).unwrap();
        for (q, g) in qs.iter().zip(&got) {
            let ids = diary::prompt_ids(&bpe, false, &[], q, 0).0;
            let (want, _) = diary::reply(&model, &bpe, &ids, 12, 0.0, 1, &mut SplitMix64::new(0)).unwrap();
            assert_eq!(g, &want, "question {q:?}");
        }
        // sampling: each row draws from its own rng, so the batch packing does not change what a row says
        let o2 = GenOpts { temperature: 0.9, top_k: 20, ..o };
        let prompts: Vec<Vec<u32>> = vec![vec![EOS]; 3];
        let run = |bt: usize| {
            let mut r: Vec<SplitMix64> = (0..3).map(|i| SplitMix64::new(100 + i)).collect();
            generate(&model, &dev, &prompts, &GenOpts { batch_tokens: bt, ..o2 }, &mut r, &|_: &[u32]| false).unwrap()
        };
        let (a, b) = (run(1), run(4096));
        assert_eq!(a, b);
        assert!(a.iter().all(|x| x.len() <= 12));
    }

    #[test]
    fn surprise_sums_match_the_mc_scorer() {
        let bpe = byte_bpe();
        let (_cfg, _vm, model) = tiny(crate::model::Kind::D, 3);
        let dev = Device::Cpu;
        let texts: Vec<String> = ["Question: What is 2 and 3?\nAnswer: 5", "a", "The cat sat on the mat and looked at the bird."].iter().map(|s| s.to_string()).collect();
        let (seqs, from) = episode_seqs(&bpe, false, &texts);
        let per = token_nll(&model, &dev, &seqs, &from, 64).unwrap();
        let want = crate::mc::scored_nll(&model, &dev, &seqs, &from, 64).unwrap();
        for (i, v) in per.iter().enumerate() {
            assert_eq!(v.len(), want[i].1);
            let s: f64 = v.iter().map(|&x| x as f64).sum();
            assert!((s - want[i].0).abs() < 1e-3 * want[i].0.max(1.0), "text {i}: {s} vs {}", want[i].0);
        }
        let m = surprise(&model, &dev, &bpe, &texts, 1 << 20).unwrap();
        assert_eq!(m[1].1, 2); // "a" then the closing EOS
        assert!((m[0].0 - want[0].0 / want[0].1 as f64).abs() < 1e-4);
    }
}
