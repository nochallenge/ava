//! TESTING / DESIGN ONLY — num_practice: child-style number practice text for v39 and v40 (seeded).
//!
//! One document is 3–6 sentences joined by spaces, each from a random template, worded
//! differently from the number test:
//!
//!   Let's count: 4, 5, 6, 7.                     3–8 numbers, start 0..=990
//!   Count down: 10, 9, 8.                        3–8 numbers, ending >= 0
//!   Which is bigger, 12 or 7? 12 is bigger.      a != b
//!   23 and 45 make 68.                           a, b in 1..=99
//!   There are 5 birds. 2 fly away. Now there are 3.   1 <= k < n <= 20 (birds/apples/cars/books)
//!   We say twenty-one for 21.                    n in 0..=999
//!   Look: # # #. That is 3.                      n in 1..=20, symbol `#` only
//!
//! Numbers are child-like where the template allows: 70% in 0..=100, 25% in 101..=999, 5% in
//! 1000..=9999 (a tier outside the template's range falls back to the next lower tier).
//! Grammar follows the count ("1 flies away", "Now there is 1.").
//!
//! EXCLUSION. A sentence is redrawn when it would teach a test item: a comparison whose
//! `cmp:{min}<{max}` key, a sum whose `sum:{min}+{max}` key, or a words sentence whose
//! `word:{n}` key is excluded, and a counting run (up or down) holding an adjacent pair k, k+1
//! with `succ:{k}` excluded. Stars cannot collide (the test uses `* o x +`, practice `#`).
//! The caller passes the keys of `items(Dev, 400)` ∪ `items(Test, 400)` (`exclusion_keys`).
//!
//! KEYS (v40). `practice_docs_with_keys` returns every doc with the test keys it teaches, in
//! `num_items::key` format: `succ:{k}` per adjacent pair of a count, `cmp:` per comparison,
//! `sum:` per sum, `word:` per words sentence, `stars:#{n}` per look sentence, and, to be safe,
//! `sum:{k}+{n-k}` per take-away sentence (n - k = m is the fact k + m = n). It draws exactly
//! what `practice_docs` draws, so v39's text is reproduced byte for byte (`V39_PRACTICE_SHA256`).
//!
//! VERSION 2 (v40, opt-in: `practice_docs_v2`). The seven templates above plus an eighth, a
//! comparison that never uses the test's wording (" is more than " / " is less than "):
//!
//!   Which is bigger, 34 or 43? It is 43.         Which is smaller, 120 or 95? It is 95.
//!   Between 7 and 12, the bigger one is 12.      Between 7 and 12, the smaller one is 7.
//!
//! Half of its pairs have the same number of digits (1–4 digits, weights 15/35/35/15; 1-digit
//! numbers include 0), half differ by one digit (the shorter has 1–3 digits); of those, half
//! are traps: the shorter number has the larger first digit (95 vs 102). Version 2 also redraws
//! a take-away sentence whose sum key is excluded. Its exclusion is Dev ∪ Test ∪ Test2
//! (`exclusion_keys_v40`), and it uses its own random stream (so seed 3 of v2 is not v39's).
//!
//! VERSION 3 (v41, opt-in: `practice_docs_v3`). Version 2's eight templates plus four comparisons in both
//! directions and several wordings, now INCLUDING the test's own wording (every key the tests use is excluded):
//!
//!   8  one statement, x < y:  "{y} is more than {x}."  "{x} is less than {y}."  "{y} is greater than {x}."
//!                             "{x} is smaller than {y}."  "{y} is bigger than {x}."  "{x} is lower than {y}."
//!   9  a yes/no question:     "Is {a} more than {b}? Yes, {a} is more than {b}."  "Is {a} less than {b}? No, {a} is
//!                             more than {b}."  (both questions, both answers)
//!   10 counted things:        "There are {a} cats and {b} dogs. There are more cats than dogs." (or "fewer"; either
//!                             kind of thing may be the subject)
//!   11 an ordered list:       "From small to big: 3, 17, 240."  "From big to small: 240, 17, 3."  (3-5 numbers)
//!
//! Their numbers (`pair_v3`): 35% the same length (1-5 digits, 1-digit numbers include 0), 35% lengths differing by
//! one (the shorter 1-4 digits; half of them traps, the shorter number having the larger first digit), 30% lengths
//! differing by 2-4 (far ratios, up to 6 digits). Since "more than" always has the smaller number second and "less
//! than" the bigger one, and the pairs run over all lengths, true "more than" and "less than" come with small and
//! with big second numbers alike (v40's comparisons across lengths and far ratios came out reversed). Every pair
//! of a sentence teaches its `cmp:` key (a list teaches all its pairs); a sentence with an excluded key is redrawn.
//! Its exclusion is Dev, Test, Test2 and Test3 (`exclusion_keys_v41`); its own random stream; seed 5 by default
//! (v41's data/num/practice_v41.txt).

use crate::num_items::{self, cmp_key, digit_range, leading_digit, succ_key, sum_key, uniform, word_key, Set};
use crate::numbers::number_words;
use crate::rng::{derive_seed, SplitMix64};
use std::collections::HashSet;

/// The practice text's default seed (the test sets use 1 and 2).
pub const DEFAULT_SEED: u64 = 3;
/// Version 2's default seed (v40's data/num/practice_v40.txt).
pub const DEFAULT_SEED_V2: u64 = 4;
/// Version 3's default seed (v41's data/num/practice_v41.txt).
pub const DEFAULT_SEED_V3: u64 = 5;
/// The test size the exclusion is built from.
pub const EXCLUDE_N_PER_KIND: usize = 400;
/// v39's practice text data/num/practice.txt: seed 3, 300,000 docs, one per line ("\n").
pub const V39_PRACTICE_SEED: u64 = 3;
pub const V39_PRACTICE_DOCS: usize = 300_000;
pub const V39_PRACTICE_SHA256: &str = "308b1be369d1619f1a487ecc843cf6819b8e832b9373c8713fc4e1c5cafb1507";
/// v40's practice text data/num/practice_v40.txt: version 2, seed 4, 300,000 docs, v40's exclusion (v40's mix
/// read only its first 45,690 docs; every key of all 300,000 is counted as used).
pub const V40_PRACTICE_SEED: u64 = 4;
pub const V40_PRACTICE_DOCS: usize = 300_000;
pub const V40_PRACTICE_SHA256: &str = "a82c6c296264001338a942aaaca8d41289a5ad91e991c7a1407044d4e8a5539f";

const N_TEMPLATES: u64 = 7;
const N_TEMPLATES_V2: u64 = 8;
const N_TEMPLATES_V3: u64 = 12;
const TRIES_PER_TEMPLATE: usize = 200;
const STREAM_LABEL: u64 = 0x7072_6163_7469_6365;

/// (plural noun, verb phrase for many, verb phrase for one)
const THINGS: [(&str, &str, &str); 4] = [
    ("birds", "fly away", "flies away"),
    ("apples", "are eaten", "is eaten"),
    ("cars", "drive away", "drives away"),
    ("books", "are given away", "is given away"),
];

/// v3's pairs of counted things.
const PAIRS: [(&str, &str); 8] = [
    ("cats", "dogs"),
    ("apples", "pears"),
    ("boats", "cars"),
    ("birds", "fish"),
    ("books", "pens"),
    ("cups", "plates"),
    ("girls", "boys"),
    ("red balls", "blue balls"),
];

/// Which practice generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Version {
    /// v39's seven templates (byte-identical to v39's text for the same seed).
    V1,
    /// v40: the seven plus the new comparison template, and take-away sentences checked too.
    V2,
    /// v41: version 2 plus four comparison templates in both directions (see the module docs).
    V3,
}

/// The keys of `items(Dev, n_per_kind)` ∪ `items(Test, n_per_kind)`, generated in memory (the
/// sealed file is never read).
pub fn exclusion_keys(n_per_kind: usize) -> HashSet<String> {
    let mut keys = HashSet::new();
    for set in [Set::Dev, Set::Test] {
        for it in num_items::items(set, n_per_kind) {
            keys.insert(num_items::key(&it));
        }
    }
    keys
}

/// v40's exclusion: the keys of `items(s, n_per_kind)` for s in Dev, Test and Test2.
pub fn exclusion_keys_v40(n_per_kind: usize) -> HashSet<String> {
    let mut keys = exclusion_keys(n_per_kind);
    for it in num_items::items(Set::Test2, n_per_kind) {
        keys.insert(num_items::key(&it));
    }
    keys
}

/// v41's exclusion: the keys of `items(s, n_per_kind)` for s in Dev, Test, Test2 and Test3.
pub fn exclusion_keys_v41(n_per_kind: usize) -> HashSet<String> {
    let mut keys = exclusion_keys_v40(n_per_kind);
    for it in num_items::items(Set::Test3, n_per_kind) {
        keys.insert(num_items::key(&it));
    }
    keys
}

/// Every key v40's practice text teaches (version 2, seed 4, all 300,000 docs, v40's exclusion).
pub fn v40_practice_keys() -> HashSet<String> {
    let ex = exclusion_keys_v40(EXCLUDE_N_PER_KIND);
    let mut keys = HashSet::new();
    let mut rng = doc_rng(Version::V2, V40_PRACTICE_SEED);
    let mut k = Vec::new();
    for _ in 0..V40_PRACTICE_DOCS {
        doc_sentences(Version::V2, &mut rng, &ex, &mut k);
        keys.extend(k.drain(..));
    }
    keys
}

/// Every key v39's practice text teaches (seed 3, 300,000 docs, v39's exclusion).
pub fn v39_practice_keys() -> HashSet<String> {
    let ex = exclusion_keys(EXCLUDE_N_PER_KIND);
    let mut keys = HashSet::new();
    let mut rng = doc_rng(Version::V1, V39_PRACTICE_SEED);
    let mut k = Vec::new();
    for _ in 0..V39_PRACTICE_DOCS {
        doc_sentences(Version::V1, &mut rng, &ex, &mut k);
        keys.extend(k.drain(..));
    }
    keys
}

/// A child-like number in `lo..=hi`: tier 0..=100 (70%), 101..=999 (25%), 1000..=9999 (5%),
/// each clamped to `lo..=hi`; an empty tier falls back to the next lower one.
fn child_number(rng: &mut SplitMix64, lo: u64, hi: u64) -> u64 {
    const TIERS: [(u64, u64); 3] = [(0, 100), (101, 999), (1000, 9999)];
    let r = rng.below(100);
    let mut t = if r < 70 { 0 } else if r < 95 { 1 } else { 2 };
    loop {
        let (a, b) = (TIERS[t].0.max(lo), TIERS[t].1.min(hi));
        if a <= b {
            return a + rng.below(b - a + 1);
        }
        if t == 0 {
            return lo + rng.below(hi - lo + 1);
        }
        t -= 1;
    }
}

fn join(nums: &[u64]) -> String {
    nums.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")
}

/// A version-2 comparison pair (see the module docs), in the order the sentence says them.
fn pair_v2(rng: &mut SplitMix64) -> (u64, u64) {
    if rng.below(2) == 0 {
        let d = match rng.below(100) {
            0..=14 => 1,
            15..=49 => 2,
            50..=84 => 3,
            _ => 4,
        };
        let (lo, hi) = if d == 1 { (0, 9) } else { digit_range(d) };
        (uniform(rng, lo, hi), uniform(rng, lo, hi))
    } else {
        let ds = 1 + rng.below(3) as u32;
        let trap = rng.below(2) == 0;
        let (slo, shi) = digit_range(ds);
        let (llo, lhi) = digit_range(ds + 1);
        let (mut s, mut l) = (uniform(rng, slo, shi), uniform(rng, llo, lhi));
        while (leading_digit(s) > leading_digit(l)) != trap {
            s = uniform(rng, slo, shi);
            l = uniform(rng, llo, lhi);
        }
        if rng.below(2) == 0 {
            (s, l)
        } else {
            (l, s)
        }
    }
}

/// A version-3 pair (x, y) with x <= y (see the module docs).
fn pair_v3(rng: &mut SplitMix64) -> (u64, u64) {
    let r = rng.below(100);
    let (a, b) = if r < 35 {
        let d = match rng.below(100) {
            0..=9 => 1,
            10..=34 => 2,
            35..=59 => 3,
            60..=84 => 4,
            _ => 5,
        };
        let (lo, hi) = if d == 1 { (0, 9) } else { digit_range(d) };
        (uniform(rng, lo, hi), uniform(rng, lo, hi))
    } else if r < 70 {
        let ds = 1 + rng.below(4) as u32;
        let trap = rng.below(2) == 0;
        let (slo, shi) = digit_range(ds);
        let (llo, lhi) = digit_range(ds + 1);
        let (mut s, mut l) = (uniform(rng, slo, shi), uniform(rng, llo, lhi));
        while (leading_digit(s) > leading_digit(l)) != trap {
            s = uniform(rng, slo, shi);
            l = uniform(rng, llo, lhi);
        }
        (s, l)
    } else {
        let ds = 1 + rng.below(3) as u32;
        let dl = (ds + 2 + rng.below(3) as u32).min(6);
        let (slo, shi) = if ds == 1 { (0, 9) } else { digit_range(ds) };
        let (llo, lhi) = digit_range(dl);
        (uniform(rng, slo, shi), uniform(rng, llo, lhi))
    };
    (a.min(b), a.max(b))
}

/// One sentence from template `t`, or None when it would be excluded. On success the keys it
/// teaches are appended to `keys`. Version 1 draws exactly what v39 drew.
fn sentence(v: Version, t: u64, rng: &mut SplitMix64, ex: &HashSet<String>, keys: &mut Vec<String>) -> Option<String> {
    match t {
        0 => {
            let len = 3 + rng.below(6);
            let start = child_number(rng, 0, 990);
            let nums: Vec<u64> = (start..start + len).collect();
            if nums.windows(2).any(|w| ex.contains(&succ_key(w[0]))) {
                return None;
            }
            keys.extend(nums.windows(2).map(|w| succ_key(w[0])));
            Some(format!("Let's count: {}.", join(&nums)))
        }
        1 => {
            let len = 3 + rng.below(6);
            let start = child_number(rng, len - 1, 9999);
            let nums: Vec<u64> = (0..len).map(|i| start - i).collect();
            if nums.windows(2).any(|w| ex.contains(&succ_key(w[1]))) {
                return None;
            }
            keys.extend(nums.windows(2).map(|w| succ_key(w[1])));
            Some(format!("Count down: {}.", join(&nums)))
        }
        2 => {
            let a = child_number(rng, 0, 9999);
            let b = child_number(rng, 0, 9999);
            if a == b || ex.contains(&cmp_key(a, b)) {
                return None;
            }
            keys.push(cmp_key(a, b));
            Some(format!("Which is bigger, {a} or {b}? {} is bigger.", a.max(b)))
        }
        3 => {
            let a = child_number(rng, 1, 99);
            let b = child_number(rng, 1, 99);
            if ex.contains(&sum_key(a, b)) {
                return None;
            }
            keys.push(sum_key(a, b));
            Some(format!("{a} and {b} make {}.", a + b))
        }
        4 => {
            let n = 2 + rng.below(19);
            let k = 1 + rng.below(n - 1);
            let (things, many, one) = THINGS[rng.below(THINGS.len() as u64) as usize];
            let verb = if k == 1 { one } else { many };
            let left = n - k;
            if v != Version::V1 && ex.contains(&sum_key(k, left)) {
                return None;
            }
            keys.push(sum_key(k, left));
            let now = if left == 1 { "Now there is 1.".to_string() } else { format!("Now there are {left}.") };
            Some(format!("There are {n} {things}. {k} {verb}. {now}"))
        }
        5 => {
            let n = child_number(rng, 0, 999);
            if ex.contains(&word_key(n)) {
                return None;
            }
            keys.push(word_key(n));
            Some(format!("We say {} for {n}.", number_words(n as u32)))
        }
        6 => {
            let n = 1 + rng.below(20);
            keys.push(format!("stars:#{n}"));
            Some(format!("Look: {}. That is {n}.", vec!["#"; n as usize].join(" ")))
        }
        7 => {
            let (a, b) = pair_v2(rng);
            let form = rng.below(4);
            if a == b || ex.contains(&cmp_key(a, b)) {
                return None;
            }
            keys.push(cmp_key(a, b));
            let (hi, lo) = (a.max(b), a.min(b));
            Some(match form {
                0 => format!("Which is bigger, {a} or {b}? It is {hi}."),
                1 => format!("Which is smaller, {a} or {b}? It is {lo}."),
                2 => format!("Between {a} and {b}, the bigger one is {hi}."),
                _ => format!("Between {a} and {b}, the smaller one is {lo}."),
            })
        }
        8 => {
            let (x, y) = pair_v3(rng);
            if x == y || ex.contains(&cmp_key(x, y)) {
                return None;
            }
            keys.push(cmp_key(x, y));
            Some(match rng.below(6) {
                0 => format!("{y} is more than {x}."),
                1 => format!("{x} is less than {y}."),
                2 => format!("{y} is greater than {x}."),
                3 => format!("{x} is smaller than {y}."),
                4 => format!("{y} is bigger than {x}."),
                _ => format!("{x} is lower than {y}."),
            })
        }
        9 => {
            let (x, y) = pair_v3(rng);
            if x == y || ex.contains(&cmp_key(x, y)) {
                return None;
            }
            keys.push(cmp_key(x, y));
            let (a, b) = if rng.below(2) == 0 { (x, y) } else { (y, x) };
            let asked = if rng.below(2) == 0 { "more" } else { "less" };
            let truth = if a > b { "more" } else { "less" };
            let yes = if asked == truth { "Yes" } else { "No" };
            Some(format!("Is {a} {asked} than {b}? {yes}, {a} is {truth} than {b}."))
        }
        10 => {
            let (x, y) = pair_v3(rng);
            if x == y || ex.contains(&cmp_key(x, y)) {
                return None;
            }
            keys.push(cmp_key(x, y));
            let (t1, t2) = PAIRS[rng.below(PAIRS.len() as u64) as usize];
            let (a, b) = if rng.below(2) == 0 { (x, y) } else { (y, x) };
            let (s, o, sn, on) = if rng.below(2) == 0 { (t1, t2, a, b) } else { (t2, t1, b, a) };
            let word = if sn > on { "more" } else { "fewer" };
            Some(format!("There are {a} {t1} and {b} {t2}. There are {word} {s} than {o}."))
        }
        _ => {
            let n = 3 + rng.below(3) as usize;
            let mut v: Vec<u64> = Vec::with_capacity(n);
            while v.len() < n {
                let (x, y) = pair_v3(rng);
                let z = if rng.below(2) == 0 { x } else { y };
                if !v.contains(&z) {
                    v.push(z);
                }
            }
            v.sort_unstable();
            for i in 0..n {
                for j in i + 1..n {
                    if ex.contains(&cmp_key(v[i], v[j])) {
                        return None;
                    }
                }
            }
            for i in 0..n {
                for j in i + 1..n {
                    keys.push(cmp_key(v[i], v[j]));
                }
            }
            if rng.below(2) == 0 {
                Some(format!("From small to big: {}.", join(&v)))
            } else {
                v.reverse();
                Some(format!("From big to small: {}.", join(&v)))
            }
        }
    }
}

/// The sentences of one document (3–6); the keys they teach are appended to `keys`.
fn doc_sentences(v: Version, rng: &mut SplitMix64, ex: &HashSet<String>, keys: &mut Vec<String>) -> Vec<String> {
    let n_templates = match v {
        Version::V1 => N_TEMPLATES,
        Version::V2 => N_TEMPLATES_V2,
        Version::V3 => N_TEMPLATES_V3,
    };
    let n = 3 + rng.below(4) as usize;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let t = rng.below(n_templates);
        // retry the same template (keeps the template mix uniform); a template that keeps
        // failing (an exclusion set covering it) gives way to another draw
        for _ in 0..TRIES_PER_TEMPLATE {
            if let Some(s) = sentence(v, t, rng, ex, keys) {
                out.push(s);
                break;
            }
        }
    }
    out
}

fn doc_rng(v: Version, seed: u64) -> SplitMix64 {
    SplitMix64::new(match v {
        Version::V1 => derive_seed(seed, STREAM_LABEL, 0),
        Version::V2 => derive_seed(seed, STREAM_LABEL, 2),
        Version::V3 => derive_seed(seed, STREAM_LABEL, 3),
    })
}

/// `n_docs` documents of version `v` with the keys each teaches (sorted, deduplicated).
pub fn docs_with_keys(v: Version, seed: u64, n_docs: usize, exclude: &HashSet<String>) -> Vec<(String, Vec<String>)> {
    let mut rng = doc_rng(v, seed);
    (0..n_docs)
        .map(|_| {
            let mut keys = Vec::new();
            let doc = doc_sentences(v, &mut rng, exclude, &mut keys).join(" ");
            keys.sort();
            keys.dedup();
            (doc, keys)
        })
        .collect()
}

/// `n_docs` practice documents (one string each, no newlines), seeded and deterministic.
/// No sentence teaches an excluded key (see the module docs).
pub fn practice_docs(seed: u64, n_docs: usize, exclude: &HashSet<String>) -> Vec<String> {
    let mut rng = doc_rng(Version::V1, seed);
    let mut keys = Vec::new();
    (0..n_docs)
        .map(|_| {
            keys.clear();
            doc_sentences(Version::V1, &mut rng, exclude, &mut keys).join(" ")
        })
        .collect()
}

/// v39's generator with the keys each doc teaches; the docs are exactly `practice_docs`'.
pub fn practice_docs_with_keys(seed: u64, n_docs: usize, exclude: &HashSet<String>) -> Vec<(String, Vec<String>)> {
    docs_with_keys(Version::V1, seed, n_docs, exclude)
}

/// Version 2 (v40): the v39 templates plus the new comparisons (see the module docs).
pub fn practice_docs_v2(seed: u64, n_docs: usize, exclude: &HashSet<String>) -> Vec<String> {
    let mut rng = doc_rng(Version::V2, seed);
    let mut keys = Vec::new();
    (0..n_docs)
        .map(|_| {
            keys.clear();
            doc_sentences(Version::V2, &mut rng, exclude, &mut keys).join(" ")
        })
        .collect()
}

/// Version 3 (v41): version 2 plus four comparison templates in both directions (see the module docs).
pub fn practice_docs_v3(seed: u64, n_docs: usize, exclude: &HashSet<String>) -> Vec<String> {
    let mut rng = doc_rng(Version::V3, seed);
    let mut keys = Vec::new();
    (0..n_docs)
        .map(|_| {
            keys.clear();
            doc_sentences(Version::V3, &mut rng, exclude, &mut keys).join(" ")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What one parsed sentence teaches.
    #[derive(Debug)]
    enum Fact {
        Count(Vec<u64>),
        Bigger(u64, u64),
        Sum(u64, u64),
        Take(u64, u64),
        Word(u64),
        Look(u64),
    }

    fn num(w: &str) -> u64 {
        w.trim_end_matches(['.', ',', '?']).parse().unwrap_or_else(|_| panic!("not a number: {w:?}"))
    }

    /// Re-parse a document word by word, checking every sentence is a template and true.
    fn parse_doc(doc: &str) -> Vec<Fact> {
        let w: Vec<&str> = doc.split(' ').collect();
        let mut facts = Vec::new();
        let mut i = 0;
        while i < w.len() {
            match w[i] {
                "Let's" | "Count" => {
                    let up = w[i] == "Let's";
                    assert_eq!(w[i + 1], if up { "count:" } else { "down:" });
                    i += 2;
                    let mut nums = Vec::new();
                    loop {
                        nums.push(num(w[i]));
                        let end = w[i].ends_with('.');
                        assert!(end || w[i].ends_with(','), "{doc}");
                        i += 1;
                        if end {
                            break;
                        }
                    }
                    assert!((3..=8).contains(&nums.len()), "{doc}");
                    for p in nums.windows(2) {
                        assert_eq!(p[1], if up { p[0] + 1 } else { p[0] - 1 }, "{doc}");
                    }
                    if up {
                        assert!(nums[0] <= 990);
                    }
                    facts.push(Fact::Count(nums));
                }
                "Which" => {
                    assert_eq!(w[i + 1], "is");
                    let bigger = match w[i + 2] {
                        "bigger," => true,
                        "smaller," => false,
                        x => panic!("{x:?} in {doc}"),
                    };
                    let (a, b) = (num(w[i + 3]), num(w[i + 5]));
                    assert_eq!(w[i + 4], "or");
                    assert!(w[i + 5].ends_with('?'));
                    assert_ne!(a, b);
                    let want = if bigger { a.max(b) } else { a.min(b) };
                    if w[i + 6] == "It" {
                        assert_eq!(w[i + 7], "is");
                        assert!(w[i + 8].ends_with('.'));
                        assert_eq!(num(w[i + 8]), want, "{doc}");
                    } else {
                        assert!(bigger);
                        assert_eq!(num(w[i + 6]), want, "{doc}");
                        assert_eq!(&w[i + 7..i + 9], ["is", "bigger."]);
                    }
                    facts.push(Fact::Bigger(a, b));
                    i += 9;
                }
                "Between" => {
                    let (a, b) = (num(w[i + 1]), num(w[i + 3]));
                    assert_eq!(w[i + 2], "and");
                    assert!(w[i + 3].ends_with(','));
                    assert_eq!(w[i + 4], "the");
                    let bigger = match w[i + 5] {
                        "bigger" => true,
                        "smaller" => false,
                        x => panic!("{x:?} in {doc}"),
                    };
                    assert_eq!(&w[i + 6..i + 8], ["one", "is"]);
                    assert!(w[i + 8].ends_with('.'));
                    assert_ne!(a, b);
                    assert_eq!(num(w[i + 8]), if bigger { a.max(b) } else { a.min(b) }, "{doc}");
                    facts.push(Fact::Bigger(a, b));
                    i += 9;
                }
                "There" => {
                    assert_eq!(w[i + 1], "are");
                    let n = num(w[i + 2]);
                    assert!(THINGS.iter().any(|t| w[i + 3] == format!("{}.", t.0)));
                    let k = num(w[i + 4]);
                    assert!(1 <= k && k < n && n <= 20, "{doc}");
                    i += 5;
                    while !w[i].ends_with('.') {
                        i += 1;
                    }
                    i += 1;
                    assert_eq!(&w[i..i + 2], ["Now", "there"]);
                    assert_eq!(w[i + 2], if n - k == 1 { "is" } else { "are" });
                    assert_eq!(num(w[i + 3]), n - k, "{doc}");
                    facts.push(Fact::Take(k, n - k));
                    i += 4;
                }
                "We" => {
                    assert_eq!(w[i + 1], "say");
                    let f = (i + 2..w.len()).find(|&j| w[j] == "for").unwrap();
                    let n = num(w[f + 1]);
                    assert_eq!(w[i + 2..f].join(" "), number_words(n as u32), "{doc}");
                    facts.push(Fact::Word(n));
                    i = f + 2;
                }
                "Look:" => {
                    i += 1;
                    let mut n = 0;
                    loop {
                        n += 1;
                        let end = w[i] == "#.";
                        assert!(end || w[i] == "#");
                        i += 1;
                        if end {
                            break;
                        }
                    }
                    assert_eq!(&w[i..i + 2], ["That", "is"]);
                    assert_eq!(num(w[i + 2]), n);
                    facts.push(Fact::Look(n));
                    i += 3;
                }
                first => {
                    let a = num(first);
                    assert_eq!(w[i + 1], "and");
                    let b = num(w[i + 2]);
                    assert_eq!(w[i + 3], "make");
                    assert_eq!(num(w[i + 4]), a + b, "{doc}");
                    assert!((1..=99).contains(&a) && (1..=99).contains(&b));
                    facts.push(Fact::Sum(a, b));
                    i += 5;
                }
            }
        }
        facts
    }

    /// The keys the parsed facts teach, sorted and deduplicated (the generator's key rule).
    fn fact_keys(facts: &[Fact]) -> Vec<String> {
        let mut k = Vec::new();
        for f in facts {
            match f {
                Fact::Count(v) => k.extend(v.windows(2).map(|p| succ_key(p[0].min(p[1])))),
                Fact::Bigger(a, b) => k.push(cmp_key(*a, *b)),
                Fact::Sum(a, b) | Fact::Take(a, b) => k.push(sum_key(*a, *b)),
                Fact::Word(n) => k.push(word_key(*n)),
                Fact::Look(n) => k.push(format!("stars:#{n}")),
            }
        }
        k.sort();
        k.dedup();
        k
    }

    /// Excluded keys taught; take-away sentences count only with `take` (v1 never checked them).
    fn hits(facts: &[Fact], ex: &HashSet<String>, take: bool) -> usize {
        facts
            .iter()
            .map(|f| match f {
                Fact::Count(v) => v.windows(2).filter(|p| ex.contains(&succ_key(p[0].min(p[1])))).count(),
                Fact::Bigger(a, b) => ex.contains(&cmp_key(*a, *b)) as usize,
                Fact::Sum(a, b) => ex.contains(&sum_key(*a, *b)) as usize,
                Fact::Take(a, b) => (take && ex.contains(&sum_key(*a, *b))) as usize,
                Fact::Word(n) => ex.contains(&word_key(*n)) as usize,
                Fact::Look(_) => 0,
            })
            .sum()
    }

    #[test]
    fn no_excluded_key_ever_and_sentences_are_true() {
        let ex = exclusion_keys(EXCLUDE_N_PER_KIND);
        assert_eq!(ex.len(), 2 * (5 * 400 + 40));
        let docs = practice_docs(DEFAULT_SEED, 4000, &ex);
        let mut total = 0;
        for d in &docs {
            assert!(!d.contains('\n') && !d.contains('\t'));
            let facts = parse_doc(d);
            assert!((3..=6).contains(&facts.len()), "{d}");
            assert_eq!(hits(&facts, &ex, false), 0, "{d}");
            total += facts.len();
        }
        assert!(total > 4000 * 3);
        // the check is not vacuous: without exclusion, the same templates do hit test keys
        let free = practice_docs(DEFAULT_SEED, 4000, &HashSet::new());
        let h: usize = free.iter().map(|d| hits(&parse_doc(d), &ex, false)).sum();
        assert!(h > 100, "only {h} hits without exclusion");
    }

    #[test]
    fn seeded_and_deterministic() {
        let ex = exclusion_keys(50);
        assert_eq!(practice_docs(3, 300, &ex), practice_docs(3, 300, &ex));
        assert_ne!(practice_docs(3, 300, &ex), practice_docs(4, 300, &ex));
        assert_eq!(practice_docs(3, 0, &ex).len(), 0);
        // a longer run starts with the shorter one
        assert_eq!(practice_docs(3, 50, &ex)[..], practice_docs(3, 300, &ex)[..50]);
    }

    #[test]
    fn three_to_six_sentences_and_all_templates() {
        let ex = HashSet::new();
        let mut rng = SplitMix64::new(9);
        let mut seen = [0usize; 7];
        let mut keys = Vec::new();
        for _ in 0..2000 {
            let s = doc_sentences(Version::V1, &mut rng, &ex, &mut keys);
            assert!((3..=6).contains(&s.len()));
            for x in &s {
                let t = match x.split(' ').next().unwrap() {
                    "Let's" => 0,
                    "Count" => 1,
                    "Which" => 2,
                    "There" => 4,
                    "We" => 5,
                    "Look:" => 6,
                    _ => 3,
                };
                seen[t] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c > 800), "{seen:?}");
    }

    #[test]
    fn child_numbers_follow_the_tiers() {
        let mut r = SplitMix64::new(5);
        let mut tiers = [0usize; 3];
        for _ in 0..20000 {
            let n = child_number(&mut r, 0, 9999);
            tiers[if n <= 100 { 0 } else if n <= 999 { 1 } else { 2 }] += 1;
        }
        assert!((13500..14500).contains(&tiers[0]), "{tiers:?}");
        assert!((4600..5400).contains(&tiers[1]), "{tiers:?}");
        assert!((800..1200).contains(&tiers[2]), "{tiers:?}");
        for _ in 0..2000 {
            let n = child_number(&mut r, 1, 99);
            assert!((1..=99).contains(&n));
            assert!(child_number(&mut r, 0, 990) <= 990);
            assert!(child_number(&mut r, 7, 9999) >= 7);
        }
    }

    /// The proof that `practice_docs_with_keys` regenerates v39's data/num/practice.txt: the
    /// sha256 of its 300,000 docs, one per line, equals the file's.
    #[test]
    fn v39_practice_text_reproduced_exactly() {
        let ex = exclusion_keys(EXCLUDE_N_PER_KIND);
        let docs = practice_docs_with_keys(V39_PRACTICE_SEED, V39_PRACTICE_DOCS, &ex);
        let mut h = crate::sha256::Sha256::new();
        for (d, _) in &docs {
            h.update(d.as_bytes());
            h.update(b"\n");
        }
        assert_eq!(h.hex(), V39_PRACTICE_SHA256);
        // and the plain generator gives the same docs
        let plain = practice_docs(V39_PRACTICE_SEED, 2000, &ex);
        assert!(plain.iter().zip(&docs).all(|(a, (b, _))| a == b));
        // every key set is what the doc's sentences teach
        for (d, k) in docs.iter().take(20_000) {
            assert_eq!(&fact_keys(&parse_doc(d)), k, "{d}");
        }
        // v39_practice_keys is their union
        let all: HashSet<String> = docs.iter().flat_map(|(_, k)| k.iter().cloned()).collect();
        assert_eq!(all, v39_practice_keys());
        assert!(all.is_disjoint(&ex.iter().filter(|k| !k.starts_with("sum:")).cloned().collect()));
    }

    #[test]
    fn v2_keeps_clear_of_dev_test_and_test2_and_of_the_test_wording() {
        let ex = exclusion_keys_v40(EXCLUDE_N_PER_KIND);
        assert!(ex.len() > 2 * (5 * 400 + 40));
        let docs = docs_with_keys(Version::V2, DEFAULT_SEED_V2, 6000, &ex);
        let (mut n7, mut diff_len, mut traps) = (0usize, 0usize, 0usize);
        for (d, k) in &docs {
            assert!(!d.contains('\n') && !d.contains('\t'));
            assert!(!d.contains(" is more than ") && !d.contains(" is less than "), "{d}");
            let facts = parse_doc(d);
            assert!((3..=6).contains(&facts.len()), "{d}");
            assert_eq!(hits(&facts, &ex, true), 0, "{d}");
            assert_eq!(&fact_keys(&facts), k, "{d}");
            assert!(k.iter().all(|x| !ex.contains(x)), "{d}");
            for s in d.split(". ").filter(|s| s.contains("? It is ") || s.starts_with("Between ")) {
                let v = num_items::numbers_in(s);
                let (a, b) = (v[0].max(v[1]), v[0].min(v[1]));
                n7 += 1;
                if a.to_string().len() != b.to_string().len() {
                    diff_len += 1;
                    traps += (leading_digit(b) > leading_digit(a)) as usize;
                }
            }
        }
        // about half of the new comparisons differ in length, about half of those are traps
        assert!(n7 > 2000, "{n7}");
        let f = diff_len as f64 / n7 as f64;
        assert!((0.44..0.56).contains(&f), "different-length share {f}");
        let t = traps as f64 / diff_len as f64;
        assert!((0.42..0.58).contains(&t), "trap share {t}");
    }

    #[test]
    fn v2_deterministic_and_v1_untouched() {
        let ex = exclusion_keys(50);
        assert_eq!(practice_docs_v2(4, 300, &ex), practice_docs_v2(4, 300, &ex));
        assert_ne!(practice_docs_v2(4, 300, &ex), practice_docs_v2(5, 300, &ex));
        assert_ne!(practice_docs_v2(3, 300, &ex), practice_docs(3, 300, &ex));
        let with_keys: Vec<String> = docs_with_keys(Version::V2, 4, 300, &ex).into_iter().map(|x| x.0).collect();
        assert_eq!(with_keys, practice_docs_v2(4, 300, &ex));
        // v2 uses all eight templates
        let docs = practice_docs_v2(4, 2000, &HashSet::new());
        let all = docs.join(" ");
        for needle in ["Let's count", "Count down", "is bigger.", "make", "There are", "We say", "Look:", "? It is ", "Between "] {
            assert!(all.contains(needle), "{needle}");
        }
    }

    /// The proof that `v40_practice_keys` covers v40's data/num/practice_v40.txt: its 300,000 docs regenerated,
    /// one per line, have the file's sha256, and their keys are the set's.
    #[test]
    fn v40_practice_text_reproduced_exactly() {
        let ex = exclusion_keys_v40(EXCLUDE_N_PER_KIND);
        let docs = docs_with_keys(Version::V2, V40_PRACTICE_SEED, V40_PRACTICE_DOCS, &ex);
        let mut h = crate::sha256::Sha256::new();
        for (d, _) in &docs {
            h.update(d.as_bytes());
            h.update(b"\n");
        }
        assert_eq!(h.hex(), V40_PRACTICE_SHA256);
        let all: HashSet<String> = docs.iter().flat_map(|(_, k)| k.iter().cloned()).collect();
        assert_eq!(all, v40_practice_keys());
    }

    /// v3's new sentences re-read and checked: Some(keys) for templates 8–11, None for the older ones.
    fn check_v3(s: &str) -> Option<Vec<String>> {
        let w: Vec<&str> = s.split(' ').collect();
        let v = num_items::numbers_in(s);
        let is_num = |x: &str| x.trim_end_matches(['.', ',', '?']).parse::<u64>().is_ok();
        if w.len() == 5 && is_num(w[0]) && w[1] == "is" && w[3] == "than" {
            let (p, q) = (num(w[0]), num(w[4]));
            assert!(w[4].ends_with('.'), "{s}");
            match w[2] {
                "more" | "greater" | "bigger" => assert!(p > q, "{s}"),
                "less" | "smaller" | "lower" => assert!(p < q, "{s}"),
                x => panic!("{x:?} in {s}"),
            }
            return Some(vec![cmp_key(p, q)]);
        }
        if w[0] == "Is" {
            assert_eq!(w.len(), 11, "{s}");
            let (a, b) = (num(w[1]), num(w[4]));
            assert!(w[4].ends_with('?') && w[3] == "than" && w[7] == "is" && w[9] == "than" && w[10].ends_with('.'), "{s}");
            assert_eq!((num(w[6]), num(w[10])), (a, b), "{s}");
            let truth = if a > b { "more" } else { "less" };
            assert_eq!(w[8], truth, "{s}");
            assert_eq!(w[5], if w[2] == truth { "Yes," } else { "No," }, "{s}");
            return Some(vec![cmp_key(a, b)]);
        }
        if s.starts_with("There are ") && s.contains(" and ") {
            let (a, b) = (v[0], v[1]);
            let ok = PAIRS.iter().any(|&(t1, t2)| {
                [(t1, t2, a, b), (t2, t1, b, a)].iter().any(|&(x, y, xn, yn)| {
                    let word = if xn > yn { "more" } else { "fewer" };
                    s == format!("There are {a} {t1} and {b} {t2}. There are {word} {x} than {y}.")
                })
            });
            assert!(ok, "{s}");
            return Some(vec![cmp_key(a, b)]);
        }
        if let Some(rest) = s.strip_prefix("From ") {
            let up = rest.starts_with("small to big: ");
            assert!(up || rest.starts_with("big to small: "), "{s}");
            assert!((3..=5).contains(&v.len()), "{s}");
            for p in v.windows(2) {
                assert!(if up { p[0] < p[1] } else { p[0] > p[1] }, "{s}");
            }
            let mut k = Vec::new();
            for i in 0..v.len() {
                for j in i + 1..v.len() {
                    k.push(cmp_key(v[i], v[j]));
                }
            }
            return Some(k);
        }
        None
    }

    #[test]
    fn v3_keeps_clear_of_every_test_key_and_is_true_both_ways() {
        let ex = exclusion_keys_v41(EXCLUDE_N_PER_KIND);
        let ex40 = exclusion_keys_v40(EXCLUDE_N_PER_KIND);
        assert!(ex.len() > ex40.len() + 800, "Test3's keys are in");
        let mut rng = doc_rng(Version::V3, DEFAULT_SEED_V3);
        let mut seen = [0usize; 12];
        // [more, less] statements x [second number < 100, >= 1000]; and different-length pairs either way round
        let mut dir = [[0usize; 2]; 2];
        let (mut diff_len_big_first, mut diff_len_small_first) = (0usize, 0usize);
        for _ in 0..8000 {
            let mut keys = Vec::new();
            let sents = doc_sentences(Version::V3, &mut rng, &ex, &mut keys);
            assert!((3..=6).contains(&sents.len()));
            let mut mine = Vec::new();
            for s in &sents {
                assert!(!s.contains('\n') && !s.contains('\t'));
                match check_v3(s) {
                    Some(k) => {
                        let t = if s.starts_with("Is ") {
                            9
                        } else if s.starts_with("There are ") {
                            10
                        } else if s.starts_with("From ") {
                            11
                        } else {
                            8
                        };
                        seen[t] += 1;
                        if t == 8 || t == 9 {
                            let v = num_items::numbers_in(s);
                            let second = *v.last().unwrap();
                            let more = s.ends_with(&format!(" more than {second}.")) || s.ends_with(&format!(" greater than {second}.")) || s.ends_with(&format!(" bigger than {second}."));
                            let less = s.ends_with(&format!(" less than {second}.")) || s.ends_with(&format!(" smaller than {second}.")) || s.ends_with(&format!(" lower than {second}."));
                            if more || less {
                                let col = if second < 100 { Some(0) } else if second >= 1000 { Some(1) } else { None };
                                if let Some(c) = col {
                                    dir[less as usize][c] += 1;
                                }
                                let first = v[v.len() - 2];
                                if first.to_string().len() != second.to_string().len() {
                                    if first > second {
                                        diff_len_big_first += 1;
                                    } else {
                                        diff_len_small_first += 1;
                                    }
                                }
                            }
                        }
                        mine.extend(k);
                    }
                    None => {
                        let facts = parse_doc(s);
                        assert_eq!(facts.len(), 1, "{s}");
                        assert_eq!(hits(&facts, &ex, true), 0, "{s}");
                        seen[match &facts[0] {
                            Fact::Count(_) => 0,
                            Fact::Bigger(..) => 7,
                            Fact::Sum(..) => 3,
                            Fact::Take(..) => 4,
                            Fact::Word(_) => 5,
                            Fact::Look(_) => 6,
                        }] += 1;
                        mine.extend(fact_keys(&facts));
                    }
                }
            }
            mine.sort();
            mine.dedup();
            keys.sort();
            keys.dedup();
            assert_eq!(mine, keys, "{sents:?}");
            assert!(keys.iter().all(|k| !ex.contains(k)), "{sents:?}");
        }
        assert!(seen.iter().enumerate().all(|(t, &c)| c > 1000 || t == 1 || t == 2), "{seen:?}");
        // true "more than" and "less than" come with small and with big second numbers
        assert!(dir.iter().flatten().all(|&c| c > 300), "{dir:?}");
        assert!(diff_len_big_first > 1000 && diff_len_small_first > 1000, "{diff_len_big_first} {diff_len_small_first}");
        // the check is not vacuous: without the exclusion the new templates hit test keys
        let free = docs_with_keys(Version::V3, DEFAULT_SEED_V3, 20_000, &HashSet::new());
        let h = free.iter().flat_map(|(_, k)| k.iter()).filter(|k| k.starts_with("cmp:") && ex.contains(*k)).count();
        assert!(h > 100, "only {h} comparison hits without exclusion");
    }

    #[test]
    fn v3_deterministic_and_older_versions_untouched() {
        let ex = exclusion_keys(50);
        assert_eq!(practice_docs_v3(5, 300, &ex), practice_docs_v3(5, 300, &ex));
        assert_ne!(practice_docs_v3(5, 300, &ex), practice_docs_v3(6, 300, &ex));
        assert_ne!(practice_docs_v3(4, 300, &ex), practice_docs_v2(4, 300, &ex));
        let with_keys: Vec<String> = docs_with_keys(Version::V3, 5, 300, &ex).into_iter().map(|x| x.0).collect();
        assert_eq!(with_keys, practice_docs_v3(5, 300, &ex));
        assert_eq!(practice_docs_v3(5, 50, &ex)[..], practice_docs_v3(5, 300, &ex)[..50]);
    }
}
