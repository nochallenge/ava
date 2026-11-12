// TESTING / DESIGN ONLY
//! ava_online: the plumbing for Ava's always-online learning (Wave 4 of "ava v42 - plan").
//!
//! She never goes offline to learn. A separate learner process reads what happened (the diary),
//! keeps episodes, picks what to rehearse, trains a candidate copy of her weights in the background,
//! checks it, and swaps it in only if it passes. Every decision is written to a checksummed,
//! append-only log first, so a crash or a power cut never loses or half-writes anything, and the
//! whole state can be rebuilt from the logs.
//!
//! - `log`: checksummed append-only logs and atomic file writes (G2).
//! - `json`, `safet`: read a safetensors header; check a new brain still reads the old one's codes (F4).
//! - `ckpt`: the checkpoint registry: candidates, the live brain, rollback, lineage (K6, G2, G6).
//! - `gpu_share`: sharing the PC with games (G5).
//! - `clock`: dates and "now" (C9).
//! - `episodes`: memory of experiences (L1, G4, C4, C7, C1, F2, G6).
//! - `replay`: what to rehearse and when (F7, H10, I9, B11, C2, M1, B12).
//! - `gates`: regression, identity and forgetting checks before a swap (G1, F3, L4).
//! - `drives`: what to do next, and what she can and cannot do yet (F6, F11).
//! - `diary`: a small reader for the diary files the chat writes.
//! - `feedback`: natural feedback in the diary (thanks, "no, it's X", asking again) as graded trust (C3).
//! - `mind`: reflection on her day, her life story, self-quizzing, a model of each person, lessons (F8, F11, F14, B13, J8).
//! - `goals`: standing goals over days, brought to mind by cue words, competing by pull (F5).
//! - `care`: how much a person leans on her, so a reply can gently turn them outward (J9).

pub mod ckpt;
pub mod care;
pub mod clock;
pub mod diary;
pub mod drives;
pub mod feedback;
pub mod episodes;
pub mod gates;
pub mod goals;
pub mod gpu_share;
pub mod json;
pub mod log;
pub mod mind;
pub mod replay;
pub mod safet;

/// Escapes a record field: backslash, tab, newline and CR.
pub fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => o.push_str("\\\\"),
            '\t' => o.push_str("\\t"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            _ => o.push(c),
        }
    }
    o
}

/// Undoes `esc`.
pub fn unesc(s: &str) -> String {
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
            Some('r') => o.push('\r'),
            Some(x) => o.push(x),
            None => o.push('\\'),
        }
    }
    o
}

/// One log record from fields (escaped, tab-separated, UTF-8).
pub fn rec(fields: &[&str]) -> Vec<u8> {
    fields.iter().map(|f| esc(f)).collect::<Vec<_>>().join("\t").into_bytes()
}

/// The fields of one log record.
pub fn fields(r: &[u8]) -> Vec<String> {
    String::from_utf8_lossy(r).split('\t').map(unesc).collect()
}

/// FNV-1a, 64 bit.
pub fn fnv(data: &[u8]) -> u64 {
    fnv_more(0xcbf2_9ce4_8422_2325, data)
}

/// Continues an FNV-1a hash (for hashing a file in pieces).
pub fn fnv_more(mut h: u64, data: &[u8]) -> u64 {
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Lowercase words (letters, digits, apostrophes) without the commonest little words.
pub fn content_words(s: &str) -> Vec<String> {
    const STOP: [&str; 32] = [
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "and", "in", "on", "it", "i", "you", "what",
        "who", "which", "that", "this", "do", "does", "did", "be", "for", "at", "with", "as", "by", "or", "my", "me",
    ];
    s.split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .filter(|w| !STOP.contains(&w.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip_any_text() {
        let f = ["ep", "a\tb", "line\nnext", "back\\slash", "cr\r", ""];
        let got = fields(&rec(&f));
        assert_eq!(got, f.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    }

    #[test]
    fn content_words_drop_little_words() {
        assert_eq!(content_words("What is the capital of France?"), vec!["capital", "france"]);
    }
}
