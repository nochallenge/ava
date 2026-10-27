// TESTING / DESIGN ONLY — qa_prep: turn question-answer datasets into Ava's teaching examples (v37).
//
// qa_prep [--qa-dir data/qa] [--arc-dir data/ai2-arc/ARC-V1-Feb2018-2] [--bpe FILE] [--max-len 512] [--out FILE]
//
// Every example is one token sequence  [EOS] prompt answer [EOS]  where
//   prompt = "Question: {q}\nAnswer:"                      (closed book)
//   prompt = "Context: {c}\nQuestion: {q}\nAnswer:"         (when the item has a passage)
//   answer = " {a}"
// The prompt and the answer are encoded separately (so the split point is a token boundary), and only the
// answer tokens and the closing EOS are scored when she is taught: she learns to answer, not to ask.
//
// Sources (train splits only; SciQ's test file lives in data/qa/sciq/sealed/ and is never read here):
//   0 SciQ train, closed book (question -> correct answer)          data/qa/sciq/train.json
//   1 SciQ train, with its support passage when it has one
//   2 Dolly 15k, every item (with its context when it has one)      data/qa/dolly/databricks-dolly-15k.jsonl
//   3 ARC Easy + Challenge train, closed book (question -> the right choice's text)
// A sequence longer than --max-len has its passage cut (by characters, from the end) until it fits; if it
// still does not fit, it is dropped and counted.
//
// Output (AVASFT01, little-endian): magic, u64 count, then per example: u8 source, u32 prompt length (tokens
// before the answer, the opening EOS included), u32 length, u16 tokens.

use ava_data::bpe::{Bpe, EOS};
use ava_gpu::qa::{load_arc, prompt_text, str_of as s, ARC_DIR, QA_DIR};
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: qa_prep [--qa-dir DIR] [--arc-dir DIR] [--bpe FILE] [--max-len 512] [--out FILE]";
const SOURCES: [&str; 4] = ["sciq-closed", "sciq-passage", "dolly", "arc-train"];

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



fn run() -> Result<(), String> {
    let (mut qa_dir, mut arc_dir) = (PathBuf::from(QA_DIR), PathBuf::from(ARC_DIR));
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let mut max_len = 512usize;
    let mut out_path: Option<PathBuf> = None;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        let val = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        match key {
            "--qa-dir" => qa_dir = PathBuf::from(val),
            "--arc-dir" => arc_dir = PathBuf::from(val),
            "--bpe" => bpe_path = PathBuf::from(val),
            "--max-len" => max_len = val.parse().map_err(|_| format!("bad integer {val:?}"))?,
            "--out" => out_path = Some(PathBuf::from(val)),
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

    let mut kept = [0usize; 4];
    let mut cut = [0usize; 4];
    let mut dropped = [0usize; 4];
    let mut tokens = [0u64; 4];
    let mut scored = [0u64; 4];
    let mut buf: Vec<u8> = Vec::new();
    let mut count = 0u64;
    for it in &items {
        let mut answer_ids = bpe.encode(&format!(" {}", it.answer.trim()));
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
            dropped[k] += 1;
            continue;
        };
        if seq.iter().any(|&t| t > u16::MAX as u32) {
            return Err("token id does not fit in u16".into());
        }
        kept[k] += 1;
        cut[k] += was_cut as usize;
        tokens[k] += seq.len() as u64;
        scored[k] += (seq.len() - np) as u64;
        buf.push(it.src);
        buf.extend_from_slice(&(np as u32).to_le_bytes());
        buf.extend_from_slice(&(seq.len() as u32).to_le_bytes());
        for &t in &seq {
            buf.extend_from_slice(&(t as u16).to_le_bytes());
        }
        count += 1;
    }
    if let Some(dir) = out_path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    let mut f = std::fs::File::create(&out_path).map_err(|e| format!("create {}: {e}", out_path.display()))?;
    f.write_all(b"AVASFT01").and_then(|_| f.write_all(&count.to_le_bytes())).and_then(|_| f.write_all(&buf)).map_err(|e| e.to_string())?;
    println!("{:<14} {:>7} {:>7} {:>7} {:>11} {:>11}", "source", "kept", "cut", "dropped", "tokens", "scored");
    for k in 0..4 {
        println!("{:<14} {:>7} {:>7} {:>7} {:>11} {:>11}", SOURCES[k], kept[k], cut[k], dropped[k], tokens[k], scored[k]);
    }
    println!(
        "total {count} examples, {} tokens ({} scored), max length {max_len} -> {}",
        tokens.iter().sum::<u64>(),
        scored.iter().sum::<u64>(),
        out_path.display()
    );
    Ok(())
}
