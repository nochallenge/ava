// TESTING / DESIGN ONLY — qa: the question-answer format and the dataset readers shared by qa_prep, sft_gpu,
// qa_eval and generate_gpu --qa (v37, teaching Ava to answer questions).

use serde_json::Value;
use std::path::{Path, PathBuf};

/// Where the question-answer data lives (git-ignored, never published).
pub const QA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/qa");
pub const ARC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/ai2-arc/ARC-V1-Feb2018-2");

/// The prompt Ava is taught with and asked with. The answer follows as " {answer}" and ends with EOS.
pub fn prompt_text(context: &str, question: &str) -> String {
    if context.trim().is_empty() {
        format!("Question: {}\nAnswer:", question.trim())
    } else {
        format!("Context: {}\nQuestion: {}\nAnswer:", context.trim(), question.trim())
    }
}

/// One multiple-choice question. `right` indexes `choices`.
pub struct Mc {
    pub question: String,
    pub support: String,
    pub choices: Vec<String>,
    pub right: usize,
}

pub fn str_of(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or("").to_string()
}

fn read(p: &Path) -> Result<String, String> {
    std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))
}

/// SciQ (a JSON array of {question, correct_answer, distractor1..3, support}). Choices are [correct, d1, d2, d3];
/// each choice is scored on its own, so the order does not matter, and qa_eval counts a tie as wrong.
pub fn load_sciq(path: &Path) -> Result<Vec<Mc>, String> {
    let v: Value = serde_json::from_str(&read(path)?).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for q in v.as_array().ok_or("SciQ: not an array")? {
        let choices: Vec<String> = ["correct_answer", "distractor1", "distractor2", "distractor3"].iter().map(|k| str_of(q, k)).collect();
        if str_of(q, "question").trim().is_empty() || choices.iter().any(|c| c.trim().is_empty()) {
            continue;
        }
        out.push(Mc { question: str_of(q, "question"), support: str_of(q, "support"), choices, right: 0 });
    }
    Ok(out)
}

/// ARC jsonl: {"question": {"stem", "choices": [{"text", "label"}]}, "answerKey"}.
pub fn load_arc(path: &Path) -> Result<Vec<Mc>, String> {
    let mut out = Vec::new();
    for (i, line) in read(path)?.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| format!("{} line {}: {e}", path.display(), i + 1))?;
        let q = &v["question"];
        let key = str_of(&v, "answerKey");
        let Some(cs) = q["choices"].as_array() else { continue };
        let Some(right) = cs.iter().position(|c| str_of(c, "label") == key) else { continue };
        out.push(Mc { question: str_of(q, "stem"), support: String::new(), choices: cs.iter().map(|c| str_of(c, "text")).collect(), right });
    }
    Ok(out)
}

/// The evaluation sets by name. `sciq-test` is the sealed set: it is opened once, by a pre-registered test.
pub fn eval_set(name: &str) -> Result<Vec<Mc>, String> {
    let qa = PathBuf::from(QA_DIR);
    let arc = PathBuf::from(ARC_DIR);
    match name {
        "sciq-dev" => load_sciq(&qa.join("sciq/valid.json")),
        "sciq-test" => load_sciq(&qa.join("sciq/sealed/test.json")),
        "arc-easy-dev" => load_arc(&arc.join("ARC-Easy/ARC-Easy-Dev.jsonl")),
        "arc-challenge-dev" => load_arc(&arc.join("ARC-Challenge/ARC-Challenge-Dev.jsonl")),
        _ => Err(format!("unknown set {name:?} (sciq-dev, sciq-test, arc-easy-dev, arc-challenge-dev)")),
    }
}

/// One teaching example from an AVASFT01 file: tokens [EOS] prompt answer [EOS]; the first `np` tokens are
/// the prompt (positions that predict them are not scored).
pub struct Example {
    pub src: u8,
    pub np: usize,
    pub toks: Vec<u32>,
}

pub fn read_sft(path: &Path) -> Result<Vec<Example>, String> {
    let b = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if b.len() < 16 || &b[..8] != b"AVASFT01" {
        return Err(format!("{}: not an AVASFT01 file", path.display()));
    }
    let n = u64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(n);
    let mut p = 16usize;
    let u32_at = |p: usize| -> Result<usize, String> { Ok(u32::from_le_bytes(b.get(p..p + 4).ok_or("truncated file")?.try_into().unwrap()) as usize) };
    for _ in 0..n {
        let src = *b.get(p).ok_or("truncated file")?;
        let np = u32_at(p + 1)?;
        let len = u32_at(p + 5)?;
        p += 9;
        let body = b.get(p..p + 2 * len).ok_or("truncated file")?;
        let toks: Vec<u32> = body.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect();
        p += 2 * len;
        if np == 0 || np >= len {
            return Err(format!("bad example: prompt {np} of {len} tokens"));
        }
        out.push(Example { src, np, toks });
    }
    Ok(out)
}
