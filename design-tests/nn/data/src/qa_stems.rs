//! TESTING / DESIGN ONLY — qa_stems: v41's held-out question slices: one normalised form of a question, and the split hash.
//!
//! v41's sealed QA test is a slice of the SciQ, ARC and OpenBookQA TRAIN splits held out from teaching (v41
//! starts from v39, which never saw them). gpu's `qa_split` picks the slices; `qa_prep --exclude FILE` leaves out
//! every teaching example whose question has a held-out stem. Both use `normalise_stem`, so they agree:
//!   lowercase; every run of characters that are not letters or digits becomes one space; trimmed.
//! The split order is `split_hash(stem)` (splitmix64 of FNV-1a of the stem's bytes, with a label), then the stem.

use crate::rng::mix64;
use std::collections::HashSet;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

const SPLIT_LABEL: u64 = 0x7634_315f_7161;

/// Lowercase, non-alphanumeric runs to one space, trimmed.
pub fn normalise_stem(q: &str) -> String {
    let mut out = String::with_capacity(q.len());
    let mut gap = false;
    for c in q.chars() {
        if c.is_alphanumeric() {
            if gap && !out.is_empty() {
                out.push(' ');
            }
            gap = false;
            out.extend(c.to_lowercase());
        } else {
            gap = true;
        }
    }
    out
}

/// 64-bit FNV-1a of the bytes of `s`.
pub fn fnv1a(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The order key of a normalised stem in the split.
pub fn split_hash(stem: &str) -> u64 {
    mix64(fnv1a(stem) ^ SPLIT_LABEL)
}

/// The stems of an exclusion file (one per line; empty lines skipped; each line normalised again, so a
/// hand-edited file still matches).
pub fn read_stems(path: &Path) -> io::Result<HashSet<String>> {
    let mut s = HashSet::new();
    for l in BufReader::new(std::fs::File::open(path)?).lines() {
        let n = normalise_stem(&l?);
        if !n.is_empty() {
            s.insert(n);
        }
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalise_cases() {
        assert_eq!(normalise_stem("  What is H2O?  "), "what is h2o");
        assert_eq!(normalise_stem("Which -- of these\n is TRUE ?"), "which of these is true");
        assert_eq!(normalise_stem("(A) a, b; c."), "a a b c");
        assert_eq!(normalise_stem("Ünïcode Straße"), "ünïcode straße");
        assert_eq!(normalise_stem("?!"), "");
        assert_eq!(normalise_stem("x"), normalise_stem(" X. "));
        // idempotent
        for s in ["What is H2O?", "a  b\tc", "Ünïcode", ""] {
            assert_eq!(normalise_stem(&normalise_stem(s)), normalise_stem(s));
            assert!(!normalise_stem(s).contains(['\n', '\t', '\r']));
        }
    }

    #[test]
    fn split_hash_and_stems_file() {
        assert_eq!(split_hash("what is h2o"), split_hash("what is h2o"));
        assert_ne!(split_hash("what is h2o"), split_hash("what is h2"));
        let d = std::env::temp_dir().join(format!("ava_qa_stems-{}", std::process::id()));
        std::fs::write(&d, "what is h2o\n\nWhich, of these?\n").unwrap();
        let s = read_stems(&d).unwrap();
        assert_eq!(s.len(), 2);
        assert!(s.contains("which of these") && s.contains("what is h2o"));
        let _ = std::fs::remove_file(&d);
    }
}
