// TESTING / DESIGN ONLY
//! dyck: nested-bracket (Dyck-k) sequences as content-free text for pre-pretraining (H5).
//!
//! A SEQUENCE has exactly n matched pairs of k bracket types and never nests deeper than `max_depth`.
//! It is made left to right: with opens left and room to nest, open a random type with probability
//! `p_open`, else close the innermost open bracket; at depth 0 always open, at the depth cap always
//! close, and with no opens left close. So every sequence is balanced, by construction.
//!
//! ALPHABETS. `Brackets` (k <= 4): ( ) [ ] { } < >. `Letters` (k <= 26): type t opens with the
//! lowercase letter t and closes with the uppercase one ("a b B A"). Symbols are written with one
//! space between them, so each symbol is a short token of the existing byte-level vocabulary.
//!
//! DOCUMENTS hold sequences, one per line, until a document has at least `doc_symbols` symbols.
//! The idea (pre-pretraining on formal languages before natural text) is that matching a closing
//! symbol to its opener at any distance exercises exactly the look-back her memory has to learn.

use crate::rng::SplitMix64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alphabet {
    Brackets,
    Letters,
}

impl Alphabet {
    pub fn max_k(self) -> usize {
        match self {
            Alphabet::Brackets => 4,
            Alphabet::Letters => 26,
        }
    }

    pub fn symbol(self, open: bool, t: u8) -> char {
        match self {
            Alphabet::Brackets => {
                let pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')];
                let (o, c) = pairs[t as usize];
                if open {
                    o
                } else {
                    c
                }
            }
            Alphabet::Letters => {
                if open {
                    (b'a' + t) as char
                } else {
                    (b'A' + t) as char
                }
            }
        }
    }

    /// (open, type) of a symbol, or None.
    pub fn parse(self, c: char) -> Option<(bool, u8)> {
        for t in 0..self.max_k() as u8 {
            if self.symbol(true, t) == c {
                return Some((true, t));
            }
            if self.symbol(false, t) == c {
                return Some((false, t));
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DyckParams {
    pub k: usize,
    pub min_pairs: usize,
    pub max_pairs: usize,
    pub max_depth: usize,
    pub p_open: f64,
    pub alphabet: Alphabet,
    pub doc_symbols: usize,
}

impl Default for DyckParams {
    fn default() -> Self {
        DyckParams { k: 16, min_pairs: 8, max_pairs: 128, max_depth: 16, p_open: 0.5, alphabet: Alphabet::Letters, doc_symbols: 512 }
    }
}

/// One balanced sequence of (open, type).
pub fn sequence(r: &mut SplitMix64, p: &DyckParams) -> Vec<(bool, u8)> {
    assert!(p.k >= 1 && p.k <= p.alphabet.max_k() && p.max_depth >= 1 && p.min_pairs <= p.max_pairs);
    let n = p.min_pairs + r.below((p.max_pairs - p.min_pairs + 1) as u64) as usize;
    let mut out = Vec::with_capacity(2 * n);
    let mut stack: Vec<u8> = Vec::new();
    let mut opened = 0;
    while out.len() < 2 * n {
        let can_open = opened < n && stack.len() < p.max_depth;
        let open = if !can_open {
            false
        } else if stack.is_empty() {
            true
        } else {
            (r.next_u64() as f64 / u64::MAX as f64) < p.p_open
        };
        if open {
            let t = r.below(p.k as u64) as u8;
            stack.push(t);
            opened += 1;
            out.push((true, t));
        } else {
            let t = stack.pop().expect("a close always has an opener");
            out.push((false, t));
        }
    }
    out
}

pub fn render(seq: &[(bool, u8)], a: Alphabet) -> String {
    let mut s = String::with_capacity(seq.len() * 2);
    for (i, &(o, t)) in seq.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push(a.symbol(o, t));
    }
    s
}

/// Read a rendered sequence back (None on an unknown symbol).
pub fn parse(text: &str, a: Alphabet) -> Option<Vec<(bool, u8)>> {
    text.split_whitespace().map(|w| if w.chars().count() == 1 { a.parse(w.chars().next().unwrap()) } else { None }).collect()
}

/// Balanced (every close matches the innermost open, nothing left open), and the deepest nesting.
pub fn check(seq: &[(bool, u8)]) -> (bool, usize) {
    let mut stack = Vec::new();
    let mut deepest = 0;
    for &(o, t) in seq {
        if o {
            stack.push(t);
            deepest = deepest.max(stack.len());
        } else if stack.pop() != Some(t) {
            return (false, deepest);
        }
    }
    (stack.is_empty(), deepest)
}

/// One document: sequences, one per line, until it has at least `doc_symbols` symbols.
pub fn document(r: &mut SplitMix64, p: &DyckParams) -> String {
    let mut lines = Vec::new();
    let mut n = 0;
    while n < p.doc_symbols {
        let s = sequence(r, p);
        n += s.len();
        lines.push(render(&s, p.alphabet));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequences_are_balanced_capped_and_sized() {
        for alphabet in [Alphabet::Brackets, Alphabet::Letters] {
            let p = DyckParams { k: alphabet.max_k(), min_pairs: 5, max_pairs: 60, max_depth: 6, alphabet, ..Default::default() };
            let mut r = SplitMix64::new(11);
            for _ in 0..500 {
                let s = sequence(&mut r, &p);
                let (ok, depth) = check(&s);
                assert!(ok);
                assert!(depth <= 6 && depth >= 1);
                assert!(s.len() % 2 == 0 && s.len() / 2 >= 5 && s.len() / 2 <= 60);
                assert!(s.iter().all(|&(_, t)| (t as usize) < p.k));
                assert_eq!(parse(&render(&s, alphabet), alphabet).unwrap(), s);
            }
        }
    }

    #[test]
    fn seeded_and_uses_every_type() {
        let p = DyckParams::default();
        let a = sequence(&mut SplitMix64::new(5), &p);
        assert_eq!(a, sequence(&mut SplitMix64::new(5), &p));
        let mut r = SplitMix64::new(6);
        let mut seen = [false; 16];
        for _ in 0..50 {
            for (_, t) in sequence(&mut r, &p) {
                seen[t as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn check_catches_mistakes() {
        assert_eq!(check(&[(true, 0), (true, 1), (false, 1), (false, 0)]), (true, 2));
        assert!(!check(&[(true, 0), (false, 1)]).0);
        assert!(!check(&[(true, 0)]).0);
        assert!(!check(&[(false, 0)]).0);
        assert_eq!(parse("a b 7", Alphabet::Letters), None);
        assert_eq!(parse("a ab A", Alphabet::Letters), None);
        assert_eq!(render(&[(true, 2), (false, 2)], Alphabet::Brackets), "{ }");
    }

    #[test]
    fn documents_reach_their_size() {
        let p = DyckParams { doc_symbols: 300, ..Default::default() };
        let mut r = SplitMix64::new(1);
        let d = document(&mut r, &p);
        let n: usize = d.lines().map(|l| l.split(' ').count()).sum();
        assert!(n >= 300);
        for l in d.lines() {
            assert!(check(&parse(l, p.alphabet).unwrap()).0);
        }
    }
}
