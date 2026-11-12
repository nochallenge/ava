// TESTING / DESIGN ONLY
//! reason: the Wave 5 reasoning curricula (v42), procedurally generated training text. Rust generators
//! only: no model writes any of it. Each family teaches a skill that a pre-registered test set measures
//! later, with its own wording, vocabulary, rules and layout, never the test's (see `reason_guard`).
//!
//! FAMILIES (one docs file each, `data/nn/docs/reason_<name>.docs`, written by the `v42_reason` bin):
//! - `search`  (C6): Countdown and grid-maze search traces (Stream of Search): states tried, dead ends,
//!   backing up, the solution, or an honest "no way" after an exhaustive search.
//! - `causal`  (C10, D3): made-up rule worlds; chains, branches, common causes, interventions and
//!   counterfactuals, every answer with a short why.
//! - `bayes`   (D8): evidence over several turns, beliefs updated each turn in words and exact fractions.
//! - `rules`   (A6): examples, then a rule found by checking candidates, then the rule applied.
//! - `state`   (C5): tracking positions, stacks, hand-offs and switch banks over 2 to 64 steps.
//! - `analogy` (D6): letter-string changes and word relations, none of the test's.
//! - `library` (D11): a tiny list language; tasks solved by enumeration, a library grown by compression.
//!
//! Document `i` of a family depends only on (seed, family, i): `doc(f, seed, i)` is reproducible and can
//! be made on any thread in any order.

use crate::rng::{derive_seed, SplitMix64};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Search,
    Causal,
    Bayes,
    Rules,
    State,
    Analogy,
    Library,
}

pub const FAMILIES: [Family; 7] = [Family::Search, Family::Causal, Family::Bayes, Family::Rules, Family::State, Family::Analogy, Family::Library];

impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Family::Search => "search",
            Family::Causal => "causal",
            Family::Bayes => "bayes",
            Family::Rules => "rules",
            Family::State => "state",
            Family::Analogy => "analogy",
            Family::Library => "library",
        }
    }

    pub fn parse(s: &str) -> Option<Family> {
        FAMILIES.iter().copied().find(|f| f.name() == s)
    }

    /// The source name in the shard builder, and the docs file stem.
    pub fn source_name(self) -> String {
        format!("reason_{}", self.name())
    }

    /// Documents per family by default (each family lands at roughly 4 to 7M tokens).
    pub fn default_docs(self) -> u64 {
        match self {
            Family::Search => 11_000,
            Family::Causal => 18_000,
            Family::Bayes => 12_000,
            Family::Rules => 22_000,
            Family::State => 9_000,
            Family::Analogy => 24_000,
            Family::Library => 3_000,
        }
    }

    fn id(self) -> u64 {
        FAMILIES.iter().position(|&f| f == self).unwrap() as u64 + 1
    }
}

/// Document `i` of family `f`.
pub fn doc(f: Family, seed: u64, i: u64) -> String {
    let mut r = SplitMix64::new(derive_seed(seed, 0x5245_4153_4f4e ^ f.id(), i));
    match f {
        Family::Search => crate::reason_search::doc(&mut r),
        Family::Causal => crate::reason_causal::doc(&mut r),
        Family::Bayes => crate::reason_bayes::doc(&mut r),
        Family::Rules => crate::reason_rules::doc(&mut r),
        Family::State => crate::reason_state::doc(&mut r),
        Family::Analogy => crate::reason_analogy::doc(&mut r),
        Family::Library => crate::reason_library::doc(&mut r),
    }
}

// ---------------------------------------------------------------------------------------------------
// Small helpers shared by the families.
// ---------------------------------------------------------------------------------------------------

/// A uniform pick.
pub fn pick<'a, T>(r: &mut SplitMix64, v: &'a [T]) -> &'a T {
    &v[r.below(v.len() as u64) as usize]
}

/// Uniform in lo..=hi.
pub fn range(r: &mut SplitMix64, lo: i64, hi: i64) -> i64 {
    assert!(lo <= hi);
    lo + r.below((hi - lo + 1) as u64) as i64
}

/// True with probability num/den.
pub fn chance(r: &mut SplitMix64, num: u64, den: u64) -> bool {
    r.below(den) < num
}

/// First letter upper case.
pub fn cap(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

/// "a", "a and b", "a, b and c".
pub fn and_list(items: &[String]) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].clone(),
        n => format!("{} and {}", items[..n - 1].join(", "), items[n - 1]),
    }
}

/// k distinct indices of 0..n, in random order.
pub fn sample_k(r: &mut SplitMix64, n: usize, k: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    r.shuffle(&mut v);
    v.truncate(k);
    v
}

pub fn gcd(a: u128, b: u128) -> u128 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Number words for 0..=20, digits above.
pub fn num_word(n: u64) -> String {
    const W: [&str; 21] = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    ];
    if n <= 20 {
        W[n as usize].to_string()
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reason_guard::{longest_number_run, FrameGuard};
    use crate::wording_guard::longest_bit_run;

    #[test]
    fn docs_are_reproducible_and_differ() {
        for f in FAMILIES {
            assert_eq!(doc(f, 7, 3), doc(f, 7, 3), "{f:?}");
            assert_ne!(doc(f, 7, 3), doc(f, 7, 4), "{f:?}");
            assert_ne!(doc(f, 7, 3), doc(f, 8, 3), "{f:?}");
        }
    }

    #[test]
    fn every_family_is_clean_on_a_sample() {
        let g = FrameGuard::new();
        for f in FAMILIES {
            let n = if f == Family::Library { 40 } else { 300 };
            for i in 0..n {
                let d = doc(f, 42, i);
                assert!(!d.trim().is_empty());
                let h = g.hits(&d);
                assert!(h.is_empty(), "{f:?} doc {i}: {h:?}\n{d}");
                assert!(longest_bit_run(&d) <= 12, "{f:?} doc {i}: bit run");
                assert!(longest_number_run(&d) <= 12, "{f:?} doc {i}: number run {}\n{d}", longest_number_run(&d));
                assert!(!d.contains('\u{1e}'));
            }
        }
    }

    #[test]
    fn helpers() {
        assert_eq!(and_list(&["a".into(), "b".into(), "c".into()]), "a, b and c");
        assert_eq!(and_list(&["a".into()]), "a");
        assert_eq!(cap("the gate"), "The gate");
        assert_eq!(num_word(7), "seven");
        assert_eq!(num_word(27), "27");
        let mut r = SplitMix64::new(1);
        for _ in 0..100 {
            let x = range(&mut r, -2, 3);
            assert!((-2..=3).contains(&x));
        }
        let s = sample_k(&mut r, 10, 4);
        assert_eq!(s.len(), 4);
        assert!(s.iter().all(|&x| x < 10));
    }
}
