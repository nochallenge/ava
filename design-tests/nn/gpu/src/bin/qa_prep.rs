// TESTING / DESIGN ONLY — qa_prep: turn question-answer datasets into Ava's teaching examples (v37; v40 adds OpenBookQA and held-out long answers).
//
// qa_prep [--qa-dir data/qa] [--arc-dir data/ai2-arc/ARC-V1-Feb2018-2] [--bpe FILE] [--max-len 512] [--out FILE]
//         [--obqa] [--obqa-dir DIR] [--heldout FILE] [--exclude FILE]
//
// Every example is one token sequence  [EOS] prompt answer [EOS]  where
//   prompt = "Question: {q}\nAnswer:"                      (closed book)
//   prompt = "Context: {c}\nQuestion: {q}\nAnswer:"         (when the item has a passage)
//   answer = " {a}"
// The prompt and the answer are encoded separately (so the split point is a token boundary), and only the
// answer tokens and the closing EOS are scored when she is taught: she learns to answer, not to ask.
//
// Sources (train splits only; SciQ's test file lives in data/qa/sciq/sealed/ and is never read here, nor is
// OpenBookQA's Main/test.jsonl):
//   0 SciQ train, closed book (question -> correct answer)          data/qa/sciq/train.json
//   1 SciQ train, with its support passage when it has one
//   2 Dolly 15k, every item (with its context when it has one)      data/qa/dolly/databricks-dolly-15k.jsonl
//   3 ARC Easy + Challenge train, closed book (question -> the right choice's text)
//   4 OpenBookQA Main train, closed book (question stem -> the right choice's text); only with --obqa or
//     --obqa-dir (v40), appended after source 3
// A sequence longer than --max-len has its passage cut (by characters, from the end) until it fits; if it
// still does not fit, it is dropped and counted.
//
// HELD-OUT LONG ANSWERS (v40, only with --heldout FILE): a Dolly item whose answer " {a}" encodes to 40 or more
// tokens (EOS not counted) and whose instruction text (the raw "instruction" string) has 64-bit FNV-1a hash
// mod 20 == 0 is left out of the teaching file and written to FILE instead (same format, source 2).
// Without the new flags the output is byte for byte what v37 wrote.
//
// HELD-OUT QUESTIONS (v41, only with --exclude FILE): FILE holds normalised question stems, one per line
// (gpu's qa_split writes data/v41/qa_exclude.txt: v41's dev and sealed QA slices of the train splits). Every item
// of any source whose question (Dolly: instruction) normalises (ava_data::qa_stems::normalise_stem) to one of
// them is dropped before anything else and counted. The written teaching file is then read back, every prompt
// decoded, its question re-normalised and checked against the stems: the run fails unless none remains.
//
// Output (AVASFT01, little-endian): magic, u64 count, then per example: u8 source, u32 prompt length (tokens
// before the answer, the opening EOS included), u32 length, u16 tokens.

use ava_data::bpe::{Bpe, EOS};
use ava_data::qa_stems::{normalise_stem, read_stems};
use ava_gpu::qa::{load_arc, load_obqa, prompt_text, read_sft, str_of as s, ARC_DIR, OBQA_DIR, QA_DIR};
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: qa_prep [--qa-dir DIR] [--arc-dir DIR] [--bpe FILE] [--max-len 512] [--out FILE] [--obqa] [--obqa-dir DIR] \
[--heldout FILE] [--exclude FILE]";
const SOURCES: [&str; 5] = ["sciq-closed", "sciq-passage", "dolly", "arc-train", "obqa-train"];
/// A Dolly answer of at least this many tokens can be held out.
const HELDOUT_MIN_ANSWER_TOKENS: usize = 40;
const HELDOUT_MOD: u64 = 20;

fn main() {
    if let Err(e) = run() {
        eprintln!("qa_prep: {e}");
        std::process::exit(1);
    }
}

/// One teaching item before tokenisation.
struct Item {
    src: u8,
    context: String,
    question: String,
    answer: String,
}

fn read(p: &Path) -> Result<String, String> {
    std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))
}

fn sciq(path: &Path, out: &mut Vec<Item>) -> Result<(), String> {
    let v: Value = serde_json::from_str(&read(path)?).map_err(|e| format!("{}: {e}", path.display()))?;
    for q in v.as_array().ok_or("SciQ: not an array")? {
        let (question, answer, support) = (s(q, "question"), s(q, "correct_answer"), s(q, "support"));
        if question.trim().is_empty() || answer.trim().is_empty() {
            continue;
        }
        out.push(Item { src: 0, context: String::new(), question: question.clone(), answer: answer.clone() });
        if !support.trim().is_empty() {
            out.push(Item { src: 1, context: support, question, answer });
        }
    }
    Ok(())
}

fn dolly(path: &Path, out: &mut Vec<Item>) -> Result<(), String> {
    for (i, line) in read(path)?.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| format!("{} line {}: {e}", path.display(), i + 1))?;
        let (question, answer) = (s(&v, "instruction"), s(&v, "response"));
        if question.trim().is_empty() || answer.trim().is_empty() {
            continue;
        }
        out.push(Item { src: 2, context: s(&v, "context"), question, answer });
    }
    Ok(())
}

/// 64-bit FNV-1a of the bytes of `s`.
fn fnv1a(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The answer tokens " {a}" (EOS not included).
fn answer_tokens(bpe: &Bpe, answer: &str) -> Vec<u32> {
    bpe.encode(&format!(" {}", answer.trim()))
}

/// Is this Dolly item one of the held-out long answers?
fn held_out(bpe: &Bpe, it: &Item) -> bool {
    it.src == 2 && fnv1a(&it.question) % HELDOUT_MOD == 0 && answer_tokens(bpe, &it.answer).len() >= HELDOUT_MIN_ANSWER_TOKENS
}

/// Encoded examples, per-source statistics, and the AVASFT01 body.
#[derive(Default)]
struct Sft {
    kept: [usize; 5],
    cut: [usize; 5],
    dropped: [usize; 5],
    tokens: [u64; 5],
    scored: [u64; 5],
    buf: Vec<u8>,
    count: u64,
}

impl Sft {
    fn add(&mut self, bpe: &Bpe, it: &Item, max_len: usize) -> Result<(), String> {
        let mut answer_ids = answer_tokens(bpe, &it.answer);
        answer_ids.push(EOS);
        let mut ctx = it.context.trim().to_string();
        let mut was_cut = false;
        let seq = loop {
            let mut seq = vec![EOS];
            seq.extend(bpe.encode(&prompt_text(&ctx, &it.question)));
            let np = seq.len();
            seq.extend_from_slice(&answer_ids);
            if seq.len() <= max_len {
                break Some((seq, np));
            }
            if ctx.is_empty() {
                break None;
            }
            // cut the passage to 80% (on a char boundary) and try again; drop it once it is under 40 chars
            let keep = ctx.chars().count() * 4 / 5;
            ctx = if keep < 40 { String::new() } else { ctx.chars().take(keep).collect() };
            was_cut = true;
        };
        let k = it.src as usize;
        let Some((seq, np)) = seq else {
            self.dropped[k] += 1;
            return Ok(());
        };
        if seq.iter().any(|&t| t > u16::MAX as u32) {
            return Err("token id does not fit in u16".into());
        }
        self.kept[k] += 1;
        self.cut[k] += was_cut as usize;
        self.tokens[k] += seq.len() as u64;
        self.scored[k] += (seq.len() - np) as u64;
        self.buf.push(it.src);
        self.buf.extend_from_slice(&(np as u32).to_le_bytes());
        self.buf.extend_from_slice(&(seq.len() as u32).to_le_bytes());
        for &t in &seq {
            self.buf.extend_from_slice(&(t as u16).to_le_bytes());
        }
        self.count += 1;
        Ok(())
    }

    fn write(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        }
        let mut f = std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
        f.write_all(b"AVASFT01")
            .and_then(|_| f.write_all(&self.count.to_le_bytes()))
            .and_then(|_| f.write_all(&self.buf))
            .map_err(|e| e.to_string())
    }

    fn print(&self, rows: usize, max_len: usize, path: &Path) {
        println!("{:<14} {:>7} {:>7} {:>7} {:>11} {:>11}", "source", "kept", "cut", "dropped", "tokens", "scored");
        for k in 0..rows {
            println!(
                "{:<14} {:>7} {:>7} {:>7} {:>11} {:>11}",
                SOURCES[k], self.kept[k], self.cut[k], self.dropped[k], self.tokens[k], self.scored[k]
            );
        }
        println!(
            "total {} examples, {} tokens ({} scored), max length {max_len} -> {}",
            self.count,
            self.tokens.iter().sum::<u64>(),
            self.scored.iter().sum::<u64>(),
            path.display()
        );
    }
}

fn run() -> Result<(), String> {
    let (mut qa_dir, mut arc_dir) = (PathBuf::from(QA_DIR), PathBuf::from(ARC_DIR));
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let mut max_len = 512usize;
    let mut out_path: Option<PathBuf> = None;
    let mut obqa_dir: Option<PathBuf> = None;
    let mut heldout_path: Option<PathBuf> = None;
    let mut exclude_path: Option<PathBuf> = None;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        if key == "--obqa" {
            obqa_dir.get_or_insert_with(|| PathBuf::from(OBQA_DIR));
            i += 1;
            continue;
        }
        let val = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        match key {
            "--qa-dir" => qa_dir = PathBuf::from(val),
            "--arc-dir" => arc_dir = PathBuf::from(val),
            "--bpe" => bpe_path = PathBuf::from(val),
            "--max-len" => max_len = val.parse().map_err(|_| format!("bad integer {val:?}"))?,
            "--out" => out_path = Some(PathBuf::from(val)),
            "--obqa-dir" => obqa_dir = Some(PathBuf::from(val)),
            "--heldout" => heldout_path = Some(PathBuf::from(val)),
            "--exclude" => exclude_path = Some(PathBuf::from(val)),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let out_path = out_path.unwrap_or_else(|| qa_dir.join("sft/train.sft"));
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;

    let mut items = Vec::new();
    sciq(&qa_dir.join("sciq/train.json"), &mut items)?;
    dolly(&qa_dir.join("dolly/databricks-dolly-15k.jsonl"), &mut items)?;
    for f in ["ARC-Easy/ARC-Easy-Train.jsonl", "ARC-Challenge/ARC-Challenge-Train.jsonl"] {
        for q in load_arc(&arc_dir.join(f))? {
            items.push(Item { src: 3, context: String::new(), question: q.question, answer: q.choices[q.right].clone() });
        }
    }
    if let Some(dir) = &obqa_dir {
        // the train split only: Main/test.jsonl is sealed
        for q in load_obqa(&dir.join("train.jsonl"))? {
            items.push(Item { src: 4, context: String::new(), question: q.question, answer: q.choices[q.right].clone() });
        }
    }
    let stems = match &exclude_path {
        Some(p) => {
            let stems = read_stems(p).map_err(|e| format!("read {}: {e}", p.display()))?;
            let mut dropped = [0usize; 5];
            items.retain(|it| {
                let keep = !stems.contains(&normalise_stem(&it.question));
                dropped[it.src as usize] += !keep as usize;
                keep
            });
            let per: Vec<String> = SOURCES.iter().zip(dropped).map(|(s, d)| format!("{s} {d}")).collect();
            println!("excluded: {} held-out stems from {}; items dropped: {}", stems.len(), p.display(), per.join(", "));
            Some(stems)
        }
        None => None,
    };

    let mut teach = Sft::default();
    let mut held = Sft::default();
    for it in &items {
        if heldout_path.is_some() && held_out(&bpe, it) {
            held.add(&bpe, it, max_len)?;
        } else {
            teach.add(&bpe, it, max_len)?;
        }
    }
    teach.write(&out_path)?;
    teach.print(if obqa_dir.is_some() { 5 } else { 4 }, max_len, &out_path);
    if let Some(p) = &heldout_path {
        held.write(p)?;
        println!(
            "held out: {} Dolly examples with answers of {HELDOUT_MIN_ANSWER_TOKENS}+ tokens and fnv1a(instruction) % {HELDOUT_MOD} == 0 \
({} cut, {} dropped; {} tokens, {} scored) -> {}",
            held.count,
            held.cut[2],
            held.dropped[2],
            held.tokens[2],
            held.scored[2],
            p.display()
        );
    }
    if let Some(stems) = &stems {
        // read the teaching file back: no prompt's question may have a held-out stem
        let mut files = vec![out_path.clone()];
        files.extend(heldout_path.iter().cloned());
        for f in files {
            let ex = read_sft(&f)?;
            let mut left = 0usize;
            for e in &ex {
                let p = bpe.decode(&e.toks[1..e.np]);
                let p = p.strip_suffix("\nAnswer:").ok_or("a prompt without \\nAnswer:")?;
                let q = match p.strip_prefix("Question: ") {
                    Some(q) => q,
                    None => p.rfind("\nQuestion: ").map(|k| &p[k + 11..]).ok_or("a prompt without Question:")?,
                };
                left += stems.contains(&normalise_stem(q)) as usize;
            }
            println!("check {}: {} examples read back, {left} with a held-out stem", f.display(), ex.len());
            if left > 0 {
                return Err(format!("{left} held-out questions remain in {}", f.display()));
            }
        }
    }
    Ok(())
}
