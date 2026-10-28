//! TESTING / DESIGN ONLY — num_items: v39's generated number test (a dev set and a sealed test set).
//!
//! Six kinds of multiple-choice items; every choice is a full sentence and exactly one is true.
//! An item is scored by sentence likelihood (lowest NLL wins), so the wording is fixed:
//!
//! | kind          | true sentence                         | choices                    | bins              |
//! |---------------|---------------------------------------|----------------------------|-------------------|
//! | compare_same  | `{a} is more than {b}.` same length    | more, less (same order)    | near / mid / far  |
//! | compare_mixed | the same, lengths differ by one        | more, less (same order)    | near / mid / far  |
//! | successor     | `After {n} comes {n+1}.`               | n+1, n+2, n+10, n-1 (n+3)  | carry / no_carry  |
//! | sum           | `{a} plus {b} is {a+b}.`               | c, c+1, c-1, c+10          | carry / no_carry  |
//! | words         | `{Words} is written as {n}.`           | n, n+1, n+10, n-1 (n+2)    | d1 / d2 / d3      |
//! | stars         | `* * * makes 3 stars.`                 | n, n+1, n+2, n-1 (n+3)     | small / large     |
//!
//! Compare bins are the ratio a/b of the larger to the smaller number: near <= 1.25, mid <= 2,
//! far > 2. compare_mixed puts the larger leading digit on the shorter number in exactly half
//! its items (95 vs 102, "trap" items); in the other half the shorter number's leading digit is
//! not larger, which forces a ratio above 5, so those items are all `far`.
//!
//! SETS. `Set::Dev` (seed 1) and `Set::Test` (seed 2, sealed) never share an item: every item
//! has a key (its operands, e.g. `sum:37+48`) and a key belongs to Dev when
//! `mix64(fnv1a(key))` is even, to Test when it is odd. Stars is a small domain and is split by
//! symbol instead (Dev `*` and `o`, Test `x` and `+`, n = 1..=20 each, 40 items per set).
//! Inside a set a key is used once. The right answer's position is shuffled per item.
//!
//! TEST2 (v40). v39's sealed test was opened once, so v40 gets `Set::Test2`: seed 3, the Test
//! hash half, stars symbols `=` and `~`, and NO key of `items(Test, 400)`, `items(Dev, 400)` or
//! v39's practice text (`num_practice::v39_practice_keys`: seed 3, 300,000 docs, every key a
//! doc teaches). Excluded keys are simply pre-loaded into the set's used-key list, so a draw
//! that hits one is a failed try, and Dev and Test are unchanged. v39's practice drew sums
//! (1..=99 + 1..=99) and number words (0..=999) so often that it taught every key of those
//! domains: Test2 has NO sum and NO words items, and few successor items (only 4-digit numbers
//! the practice's count-downs missed); their cells run dry and close, as described below.
//!
//! TEST3 (v41). v40's Test2 was opened once, so v41 gets `Set::Test3`: seed 4, the Test hash half, stars
//! symbols `^` and `|`, and NO key of `items(Dev, 400)`, `items(Test, 400)`, `items(Test2, 400)`, v39's practice
//! text or v40's practice text (`num_practice::v40_practice_keys`: version 2, seed 4, all 300,000 docs of
//! data/num/practice_v40.txt, of which v40's mix read the first 45,690). Like Test2 it has no sum and no words
//! items; its successor items are the 4-digit numbers neither practice counted through.
//!
//! SAMPLING. Each kind draws its items cell by cell (a cell is e.g. "2 digits, near"): every
//! item gets a planned cell, and a cell whose domain runs dry (no new key of this set's half in
//! 2,000 draws, e.g. the 36 one-digit comparison pairs) is closed; its items move to another
//! open cell of the same group (compare_same: the same bin; compare_mixed: trap or not;
//! successor: carry or not). All randomness is `SplitMix64`, one stream per (set, kind).

use crate::numbers::number_words;
use crate::rng::{derive_seed, mix64, SplitMix64};
use std::collections::HashSet;

/// Which half of the number test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Set {
    /// The practice set: look at it, tune on it.
    Dev,
    /// The sealed test set: written once, read once by the pre-registered call (v39; SPENT).
    Test,
    /// v40's fresh sealed test set (seed 3, Test's hash half, clear of Dev, Test and v39's
    /// practice text).
    Test2,
    /// v41's fresh sealed test set (seed 4, Test's hash half, clear of Dev, Test, Test2 and the practice
    /// texts of v39 and v40).
    Test3,
}

impl Set {
    fn seed(self) -> u64 {
        match self {
            Set::Dev => 1,
            Set::Test => 2,
            Set::Test2 => 3,
            Set::Test3 => 4,
        }
    }

    fn half(self) -> u64 {
        match self {
            Set::Dev => 0,
            Set::Test | Set::Test2 | Set::Test3 => 1,
        }
    }

    /// (symbol, plural noun, singular noun) of the stars kind.
    fn star_symbols(self) -> [(&'static str, &'static str, &'static str); 2] {
        match self {
            Set::Dev => [("*", "stars", "star"), ("o", "rings", "ring")],
            Set::Test => [("x", "crosses", "cross"), ("+", "pluses", "plus")],
            Set::Test2 => [("=", "equals signs", "equals sign"), ("~", "waves", "wave")],
            Set::Test3 => [("^", "carets", "caret"), ("|", "bars", "bar")],
        }
    }
}

/// One multiple-choice item: `choices[right]` is the one true sentence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    pub id: u32,
    pub kind: &'static str,
    pub bin: &'static str,
    pub right: usize,
    pub choices: Vec<String>,
}

/// The kinds, in output order.
pub const KINDS: [&str; 6] = ["compare_same", "compare_mixed", "successor", "sum", "words", "stars"];

/// Stars items per set (2 symbols x n = 1..=20); independent of `n_per_kind`.
pub const STARS_PER_SET: usize = 40;

const MAX_TRIES: usize = 2000;
const BINS_RATIO: [&str; 3] = ["near", "mid", "far"];

// ---------------------------------------------------------------- keys

pub(crate) fn cmp_key(a: u64, b: u64) -> String {
    format!("cmp:{}<{}", a.min(b), a.max(b))
}

pub(crate) fn succ_key(n: u64) -> String {
    format!("succ:{n}")
}

pub(crate) fn sum_key(a: u64, b: u64) -> String {
    format!("sum:{}+{}", a.min(b), a.max(b))
}

pub(crate) fn word_key(n: u64) -> String {
    format!("word:{n}")
}

fn fnv1a(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The half a key belongs to: 0 = Dev, 1 = Test (stars keys are split by symbol instead).
pub fn key_half(key: &str) -> u64 {
    mix64(fnv1a(key)) & 1
}

/// Every maximal run of ASCII digits in `text`, as numbers.
pub(crate) fn numbers_in(text: &str) -> Vec<u64> {
    let mut out = Vec::new();
    let mut cur: Option<u64> = None;
    for b in text.bytes() {
        if b.is_ascii_digit() {
            cur = Some(cur.unwrap_or(0) * 10 + (b - b'0') as u64);
        } else if let Some(n) = cur.take() {
            out.push(n);
        }
    }
    out.extend(cur);
    out
}

/// The item's key, re-read from its sentences: `cmp:{min}<{max}`, `succ:{n}`,
/// `sum:{min}+{max}`, `word:{n}`, `stars:{symbol}{n}`. Panics on an unknown kind.
pub fn key(item: &Item) -> String {
    let first = &item.choices[0];
    match item.kind {
        "compare_same" | "compare_mixed" => {
            let v = numbers_in(first);
            cmp_key(v[0], v[1])
        }
        "successor" => succ_key(numbers_in(first)[0]),
        "sum" => {
            let v = numbers_in(first);
            sum_key(v[0], v[1])
        }
        "words" => word_key(numbers_in(&item.choices[item.right])[0]),
        "stars" => {
            let words: Vec<&str> = first.split(' ').collect();
            let n = words.iter().position(|&w| w == "makes").expect("stars sentence has 'makes'");
            format!("stars:{}{n}", words[0])
        }
        k => panic!("num_items::key: unknown kind {k:?}"),
    }
}

/// One TSV line (no newline): `id \t kind \t bin \t right \t choice0 \t choice1 [...]`.
pub fn tsv_line(item: &Item) -> String {
    let mut s = format!("{}\t{}\t{}\t{}", item.id, item.kind, item.bin, item.right);
    for c in &item.choices {
        s.push('\t');
        s.push_str(c);
    }
    s
}

// ---------------------------------------------------------------- generation

/// A drawn item before its choices are shuffled: `truth` first, then the wrong sentences.
struct Cand {
    key: String,
    bin: &'static str,
    truth: String,
    wrong: Vec<String>,
}

/// Draw one candidate per planned cell (see the module docs). `group[c]` is cell c's group,
/// `plan[i]` item i's planned cell. A drawn key counts only if it is in this set's half and
/// unused; `draw` returning None (e.g. an empty range) counts as a failed try.
fn fill(
    group: &[usize],
    plan: &[usize],
    set: Set,
    rng: &mut SplitMix64,
    seen: &mut HashSet<String>,
    mut draw: impl FnMut(usize, &mut SplitMix64) -> Option<Cand>,
) -> Vec<Cand> {
    let mut closed = vec![false; group.len()];
    let mut out = Vec::with_capacity(plan.len());
    for &planned in plan {
        let mut cell = planned;
        loop {
            if closed[cell] {
                let open = |same: bool| -> Vec<usize> {
                    (0..group.len()).filter(|&c| !closed[c] && (!same || group[c] == group[planned])).collect()
                };
                let mut pool = open(true);
                if pool.is_empty() {
                    pool = open(false);
                }
                if pool.is_empty() {
                    return out; // every cell ran dry
                }
                cell = pool[rng.below(pool.len() as u64) as usize];
            }
            let mut got = None;
            for _ in 0..MAX_TRIES {
                if let Some(c) = draw(cell, rng) {
                    if key_half(&c.key) == set.half() && !seen.contains(&c.key) {
                        got = Some(c);
                        break;
                    }
                }
            }
            match got {
                Some(c) => {
                    seen.insert(c.key.clone());
                    out.push(c);
                    break;
                }
                None => closed[cell] = true,
            }
        }
    }
    out
}

/// `n` flags, exactly `round(n * num / den)` of them true, in shuffled order.
fn exact_flags(n: usize, num: usize, den: usize, rng: &mut SplitMix64) -> Vec<bool> {
    let t = (n * num + den / 2) / den;
    let mut v: Vec<bool> = (0..n).map(|i| i < t).collect();
    rng.shuffle(&mut v);
    v
}

pub(crate) fn uniform(rng: &mut SplitMix64, lo: u64, hi: u64) -> u64 {
    lo + rng.below(hi - lo + 1)
}

/// The numbers with `d` digits (1-digit numbers start at 1 so every ratio is finite).
pub(crate) fn digit_range(d: u32) -> (u64, u64) {
    if d == 1 {
        (1, 9)
    } else {
        (10u64.pow(d - 1), 10u64.pow(d) - 1)
    }
}

pub(crate) fn leading_digit(mut n: u64) -> u64 {
    while n >= 10 {
        n /= 10;
    }
    n
}

/// The ratio bin of a > b > 0: near when a/b <= 1.25, mid when a/b <= 2, far above.
fn ratio_bin(a: u64, b: u64) -> &'static str {
    if 4 * a <= 5 * b {
        "near"
    } else if a <= 2 * b {
        "mid"
    } else {
        "far"
    }
}

/// The values of a (> b) that put a/b into ratio bin `bin`, intersected with `lo..=hi`.
fn bin_range(bin: usize, b: u64, lo: u64, hi: u64) -> Option<(u64, u64)> {
    let (a0, a1) = match bin {
        0 => (b + 1, 5 * b / 4),
        1 => (5 * b / 4 + 1, 2 * b),
        _ => (2 * b + 1, u64::MAX),
    };
    let (a0, a1) = (a0.max(lo), a1.min(hi));
    (a0 <= a1).then_some((a0, a1))
}

/// A comparison is a minimal pair: the same two numbers in the same order, only "more" or "less" differs, so how
/// common each number is cannot pick the answer (with the numbers swapped instead, the sentence that starts with
/// the smaller, more common number won: Ava scored 22% on far dev items). Which number comes first is bit 1 of the
/// key's hash (bit 0 picks the set), not the RNG, so every other item is unchanged; about half go each way.
fn compare_cand(big: u64, small: u64) -> Cand {
    let key = cmp_key(big, small);
    let (a, b) = if (mix64(fnv1a(&key)) >> 1) & 1 == 0 { (big, small) } else { (small, big) };
    let (t, f) = if a > b { ("more", "less") } else { ("less", "more") };
    Cand {
        key,
        bin: ratio_bin(big, small),
        truth: format!("{a} is {t} than {b}."),
        wrong: vec![format!("{a} is {f} than {b}.")],
    }
}

fn gen_compare_same(n: usize, set: Set, rng: &mut SplitMix64, seen: &mut HashSet<String>) -> Vec<Cand> {
    // cell = (digits - 1) * 3 + bin; group = bin
    let group: Vec<usize> = (0..12).map(|c| c % 3).collect();
    let plan: Vec<usize> = (0..n).map(|_| rng.below(12) as usize).collect();
    fill(&group, &plan, set, rng, seen, |cell, r| {
        let (d, bin) = (cell as u32 / 3 + 1, cell % 3);
        let (lo, hi) = digit_range(d);
        let b = uniform(r, lo, hi);
        let (a0, a1) = bin_range(bin, b, lo, hi)?;
        let a = uniform(r, a0, a1);
        debug_assert_eq!(ratio_bin(a, b), BINS_RATIO[bin]);
        Some(compare_cand(a, b))
    })
}

fn gen_compare_mixed(n: usize, set: Set, rng: &mut SplitMix64, seen: &mut HashSet<String>) -> Vec<Cand> {
    // cells: (short digits 1..=3, trap, bin); plain (non-trap) items are always far
    let mut cells: Vec<(u32, bool, usize)> = Vec::new();
    for trap in [true, false] {
        for ds in 1..=3 {
            if trap {
                for bin in 0..3 {
                    cells.push((ds, true, bin));
                }
            } else {
                cells.push((ds, false, 2));
            }
        }
    }
    let group: Vec<usize> = cells.iter().map(|c| c.1 as usize).collect();
    let traps = exact_flags(n, 1, 2, rng);
    let plan: Vec<usize> = traps
        .iter()
        .map(|&trap| {
            let ds = 1 + rng.below(3) as u32;
            let bin = if trap { rng.below(3) as usize } else { 2 };
            cells.iter().position(|&c| c == (ds, trap, bin)).unwrap()
        })
        .collect();
    fill(&group, &plan, set, rng, seen, |cell, r| {
        let (ds, trap, bin) = cells[cell];
        let (slo, shi) = digit_range(ds);
        let (llo, lhi) = digit_range(ds + 1);
        let s = uniform(r, slo, shi);
        let (l0, l1) = bin_range(bin, s, llo, lhi)?;
        let l = uniform(r, l0, l1);
        if (leading_digit(s) > leading_digit(l)) != trap {
            return None;
        }
        debug_assert_eq!(ratio_bin(l, s), BINS_RATIO[bin]);
        Some(compare_cand(l, s))
    })
}

fn gen_successor(n: usize, set: Set, rng: &mut SplitMix64, seen: &mut HashSet<String>) -> Vec<Cand> {
    // cell 0 = no_carry, 1 = carry (n ends in 9); exactly 30% planned carry
    let plan: Vec<usize> = exact_flags(n, 3, 10, rng).into_iter().map(|c| c as usize).collect();
    fill(&[0, 1], &plan, set, rng, seen, |cell, r| {
        let n = if cell == 1 {
            10 * r.below(1000) + 9
        } else {
            let n = r.below(10_000);
            if n % 10 == 9 {
                return None;
            }
            n
        };
        let s = |m: u64| format!("After {n} comes {m}.");
        let last = if n == 0 { n + 3 } else { n - 1 };
        Some(Cand {
            key: succ_key(n),
            bin: if n % 10 == 9 { "carry" } else { "no_carry" },
            truth: s(n + 1),
            wrong: vec![s(n + 2), s(n + 10), s(last)],
        })
    })
}

fn gen_sum(n: usize, set: Set, rng: &mut SplitMix64, seen: &mut HashSet<String>) -> Vec<Cand> {
    fill(&[0], &vec![0; n], set, rng, seen, |_, r| {
        let (a, b) = (uniform(r, 1, 99), uniform(r, 1, 99));
        let c = a + b;
        let s = |m: u64| format!("{a} plus {b} is {m}.");
        Some(Cand {
            key: sum_key(a, b),
            bin: if a % 10 + b % 10 >= 10 { "carry" } else { "no_carry" },
            truth: s(c),
            wrong: vec![s(c + 1), s(c - 1), s(c + 10)],
        })
    })
}

fn capitalised(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

fn gen_words(n: usize, set: Set, rng: &mut SplitMix64, seen: &mut HashSet<String>) -> Vec<Cand> {
    // cells d1 / d2 / d3, one group (a dry cell's items go to any open cell)
    let plan: Vec<usize> = (0..n).map(|_| rng.below(3) as usize).collect();
    fill(&[0, 0, 0], &plan, set, rng, seen, |cell, r| {
        let n = match cell {
            0 => r.below(10),
            1 => uniform(r, 10, 99),
            _ => uniform(r, 100, 999),
        };
        let w = capitalised(&number_words(n as u32));
        let s = |m: u64| format!("{w} is written as {m}.");
        let last = if n == 0 { n + 2 } else { n - 1 };
        Some(Cand {
            key: word_key(n),
            bin: ["d1", "d2", "d3"][cell],
            truth: s(n),
            wrong: vec![s(n + 1), s(n + 10), s(last)],
        })
    })
}

fn gen_stars(set: Set, seen: &mut HashSet<String>) -> Vec<Cand> {
    let mut out = Vec::with_capacity(STARS_PER_SET);
    for (sym, plural, single) in set.star_symbols() {
        for n in 1..=20u64 {
            let marks = vec![sym; n as usize].join(" ");
            let s = |m: u64| format!("{marks} makes {m} {}.", if m == 1 { single } else { plural });
            let last = if n == 1 { n + 3 } else { n - 1 };
            let key = format!("stars:{sym}{n}");
            assert!(seen.insert(key.clone()));
            out.push(Cand {
                key,
                bin: if n <= 4 { "small" } else { "large" },
                truth: s(n),
                wrong: vec![s(n + 1), s(n + 2), s(last)],
            });
        }
    }
    out
}

/// The number test: `n_per_kind` items of each of the first five kinds and the 40 stars items,
/// in `KINDS` order, ids from 0. Deterministic per set; Dev and Test share no key. A kind
/// whose (half-)domain is smaller than `n_per_kind` gives fewer items (never at 400 for Dev
/// and Test; Test2 runs short, see the module docs). Test2 excludes `test2_exclusion()`.
pub fn items(set: Set, n_per_kind: usize) -> Vec<Item> {
    match set {
        Set::Test2 => items_excluding(set, n_per_kind, test2_exclusion()),
        Set::Test3 => items_excluding(set, n_per_kind, test3_exclusion()),
        _ => items_excluding(set, n_per_kind, &HashSet::new()),
    }
}

/// The keys Test2 must not use: `items(Dev, 400)` ∪ `items(Test, 400)` ∪ every key of v39's
/// practice text (`num_practice::v39_practice_keys`). Built once per process (about a second).
pub fn test2_exclusion() -> &'static HashSet<String> {
    static EX: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
    EX.get_or_init(|| {
        let mut ex = HashSet::new();
        for set in [Set::Dev, Set::Test] {
            for it in items(set, 400) {
                ex.insert(key(&it));
            }
        }
        ex.extend(crate::num_practice::v39_practice_keys());
        ex
    })
}

/// The keys Test3 must not use: `items(s, 400)` for s in Dev, Test and Test2, and every key of v39's and v40's
/// practice texts. Built once per process (a few seconds).
pub fn test3_exclusion() -> &'static HashSet<String> {
    static EX: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
    EX.get_or_init(|| {
        let mut ex = HashSet::new();
        for set in [Set::Dev, Set::Test, Set::Test2] {
            for it in items(set, 400) {
                ex.insert(key(&it));
            }
        }
        ex.extend(crate::num_practice::v39_practice_keys());
        ex.extend(crate::num_practice::v40_practice_keys());
        ex
    })
}

/// `items` with the keys in `exclude` treated as already used (a draw that hits one is a
/// failed try). With an empty `exclude` this is exactly `items(Dev | Test, n)`.
pub fn items_excluding(set: Set, n_per_kind: usize, exclude: &HashSet<String>) -> Vec<Item> {
    let mut seen: HashSet<String> = exclude.clone();
    let mut out = Vec::new();
    for (k, &kind) in KINDS.iter().enumerate() {
        let mut rng = SplitMix64::new(derive_seed(set.seed(), 0x6E75_6D5F_6974_656D, k as u64));
        let cands = match kind {
            "compare_same" => gen_compare_same(n_per_kind, set, &mut rng, &mut seen),
            "compare_mixed" => gen_compare_mixed(n_per_kind, set, &mut rng, &mut seen),
            "successor" => gen_successor(n_per_kind, set, &mut rng, &mut seen),
            "sum" => gen_sum(n_per_kind, set, &mut rng, &mut seen),
            "words" => gen_words(n_per_kind, set, &mut rng, &mut seen),
            _ => gen_stars(set, &mut seen),
        };
        for c in cands {
            let mut order: Vec<usize> = (0..=c.wrong.len()).collect();
            rng.shuffle(&mut order);
            let mut all = Vec::with_capacity(order.len());
            all.push(c.truth);
            all.extend(c.wrong);
            let right = order.iter().position(|&o| o == 0).unwrap();
            let choices = order.iter().map(|&o| all[o].clone()).collect();
            out.push(Item { id: out.len() as u32, kind, bin: c.bin, right, choices });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Re-derive whether a sentence is true from its own numbers (never from `right`).
    fn is_true(kind: &str, s: &str, words: &HashMap<String, u64>) -> bool {
        let v = numbers_in(s);
        match kind {
            "compare_same" | "compare_mixed" => {
                assert!(s.ends_with('.') && (s.contains(" is more than ") || s.contains(" is less than ")));
                if s.contains(" is more than ") {
                    v[0] > v[1]
                } else {
                    v[0] < v[1]
                }
            }
            "successor" => {
                assert!(s.starts_with("After ") && s.contains(" comes "));
                v[1] == v[0] + 1
            }
            "sum" => {
                assert!(s.contains(" plus ") && s.contains(" is "));
                v[0] + v[1] == v[2]
            }
            "words" => {
                let (w, _) = s.split_once(" is written as ").unwrap();
                assert!(v.len() == 1);
                words[w] == v[0]
            }
            "stars" => {
                let toks: Vec<&str> = s.split(' ').collect();
                let n = toks.iter().position(|&t| t == "makes").unwrap();
                assert!(toks[..n].iter().all(|&t| t == toks[0]));
                n as u64 == v[0]
            }
            k => panic!("unknown kind {k}"),
        }
    }

    fn words_map() -> HashMap<String, u64> {
        (0..=999u32).map(|n| (capitalised(&number_words(n)), n as u64)).collect()
    }

    #[test]
    fn deterministic_and_sets_differ() {
        assert_eq!(items(Set::Dev, 50), items(Set::Dev, 50));
        assert_eq!(items(Set::Test, 50), items(Set::Test, 50));
        assert_ne!(items(Set::Dev, 50), items(Set::Test, 50));
    }

    #[test]
    fn counts_kinds_and_ids() {
        for set in [Set::Dev, Set::Test] {
            for n in [0usize, 50, 400] {
                let it = items(set, n);
                assert_eq!(it.len(), 5 * n + 40, "{set:?} n={n}");
                for (i, x) in it.iter().enumerate() {
                    assert_eq!(x.id, i as u32);
                }
                for kind in KINDS {
                    let c = it.iter().filter(|x| x.kind == kind).count();
                    assert_eq!(c, if kind == "stars" { 40 } else { n });
                }
            }
        }
    }

    #[test]
    fn exactly_one_true_choice_and_right_points_at_it() {
        let words = words_map();
        for set in [Set::Dev, Set::Test] {
            for it in items(set, 400) {
                let n_choices = if it.kind.starts_with("compare") { 2 } else { 4 };
                assert_eq!(it.choices.len(), n_choices, "{it:?}");
                let truths: Vec<usize> =
                    (0..it.choices.len()).filter(|&i| is_true(it.kind, &it.choices[i], &words)).collect();
                assert_eq!(truths, vec![it.right], "{it:?}");
                let mut c = it.choices.clone();
                c.sort();
                c.dedup();
                assert_eq!(c.len(), n_choices, "duplicate choices {it:?}");
                for s in &it.choices {
                    assert!(!s.contains('\t') && !s.contains('\n') && !s.contains('\r'), "{s:?}");
                }
            }
        }
    }

    #[test]
    fn bins_match_the_numbers() {
        for set in [Set::Dev, Set::Test] {
            for it in items(set, 400) {
                let truth = &it.choices[it.right];
                let v = numbers_in(truth);
                let want = match it.kind {
                    "compare_same" | "compare_mixed" => {
                        let (big, small) = (v[0].max(v[1]), v[0].min(v[1]));
                        let (ld, sd) = (big.to_string().len(), small.to_string().len());
                        if it.kind == "compare_same" {
                            assert_eq!(ld, sd);
                        } else {
                            assert_eq!(ld, sd + 1);
                            if leading_digit(small) <= leading_digit(big) {
                                assert_eq!(ratio_bin(big, small), "far");
                            }
                        }
                        ratio_bin(big, small)
                    }
                    "successor" => ["no_carry", "carry"][(v[0] % 10 == 9) as usize],
                    "sum" => ["no_carry", "carry"][(v[0] % 10 + v[1] % 10 >= 10) as usize],
                    "words" => ["d1", "d2", "d3"][v[0].to_string().len() - 1],
                    _ => ["large", "small"][(v[0] <= 4) as usize],
                };
                assert_eq!(it.bin, want, "{it:?}");
            }
        }
    }

    #[test]
    fn mixed_half_traps() {
        let it = items(Set::Dev, 400);
        let mixed: Vec<&Item> = it.iter().filter(|x| x.kind == "compare_mixed").collect();
        let traps = mixed
            .iter()
            .filter(|x| {
                let v = numbers_in(&x.choices[x.right]);
                leading_digit(v[0].min(v[1])) > leading_digit(v[0].max(v[1]))
            })
            .count();
        assert_eq!(traps, 200);
        let succ = it.iter().filter(|x| x.kind == "successor" && x.bin == "carry").count();
        assert_eq!(succ, 120);
    }

    #[test]
    fn comparisons_are_minimal_pairs_and_balanced() {
        for set in [Set::Dev, Set::Test] {
            let (mut n, mut big_first, mut more_true) = (0usize, 0usize, 0usize);
            for it in items(set, 400).iter().filter(|x| x.kind.starts_with("compare")) {
                assert_eq!(it.choices.len(), 2);
                let (c0, c1) = (&it.choices[0], &it.choices[1]);
                assert_eq!(numbers_in(c0), numbers_in(c1), "{it:?}");
                assert_eq!(c0.replace(" less ", " more "), c1.replace(" less ", " more "), "{it:?}");
                assert_ne!(c0, c1);
                let v = numbers_in(c0);
                n += 1;
                big_first += (v[0] > v[1]) as usize;
                more_true += it.choices[it.right].contains(" more ") as usize;
            }
            assert_eq!(big_first, more_true);
            assert!(n == 800 && (320..=480).contains(&big_first), "{set:?}: {big_first} of {n} put the larger number first");
        }
    }

    #[test]
    fn keys_unique_in_a_set_and_never_shared() {
        let dev = items(Set::Dev, 400);
        let test = items(Set::Test, 400);
        let mut keys = [HashSet::new(), HashSet::new()];
        for (h, set) in [&dev, &test].into_iter().enumerate() {
            for it in set {
                let k = key(it);
                if it.kind != "stars" {
                    assert_eq!(key_half(&k), h as u64, "{k} in the wrong half");
                }
                assert!(keys[h].insert(k.clone()), "duplicate key {k}");
            }
        }
        assert_eq!(keys[0].intersection(&keys[1]).count(), 0);
        assert!(keys[0].contains("stars:*1") && keys[1].contains("stars:+20"));
        assert!(key(&dev[0]).starts_with("cmp:"));
    }

    #[test]
    fn tsv_lines_have_the_right_fields() {
        for it in items(Set::Dev, 20) {
            let line = tsv_line(&it);
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 4 + it.choices.len());
            assert_eq!(f[0], it.id.to_string());
            assert_eq!(f[1], it.kind);
            assert_eq!(f[3], it.right.to_string());
        }
    }

    #[test]
    fn numbers_in_reads_runs() {
        assert_eq!(numbers_in("95 is more than 102."), vec![95, 102]);
        assert_eq!(numbers_in("x7y"), vec![7]);
        assert_eq!(numbers_in("none"), Vec::<u64>::new());
        assert_eq!(numbers_in("10000"), vec![10000]);
    }

    fn tsv_sha(it: &[Item]) -> String {
        let mut h = crate::sha256::Sha256::new();
        for x in it {
            h.update(tsv_line(x).as_bytes());
            h.update(b"\n");
        }
        h.hex()
    }

    /// v40 must not change v39's sets: data/num/dev.tsv and the (spent) sealed test.tsv by hash.
    #[test]
    fn dev_and_test_are_unchanged() {
        assert_eq!(tsv_sha(&items(Set::Dev, 400)), "4ed4ad493f16a9f837f07fbf900725f8000677bf0ae973893e3a8eafd007b067");
        assert_eq!(tsv_sha(&items(Set::Test, 400)), "dc0866309cbf0823071dc5590b3081228893ab1dbb9cfcff862c245f06617f23");
        let none = HashSet::new();
        assert!(items_excluding(Set::Dev, 50, &none) == items(Set::Dev, 50));
    }

    /// The bin an item's true sentence implies (no messages: Test2 is sealed).
    fn implied_bin(kind: &str, truth: &str) -> &'static str {
        let v = numbers_in(truth);
        match kind {
            "compare_same" | "compare_mixed" => {
                let (big, small) = (v[0].max(v[1]), v[0].min(v[1]));
                let (ld, sd) = (big.to_string().len(), small.to_string().len());
                assert!(if kind == "compare_same" { ld == sd } else { ld == sd + 1 });
                if kind == "compare_mixed" && leading_digit(small) <= leading_digit(big) {
                    assert!(ratio_bin(big, small) == "far");
                }
                ratio_bin(big, small)
            }
            "successor" => ["no_carry", "carry"][(v[0] % 10 == 9) as usize],
            "sum" => ["no_carry", "carry"][(v[0] % 10 + v[1] % 10 >= 10) as usize],
            "words" => ["d1", "d2", "d3"][v[0].to_string().len() - 1],
            _ => ["large", "small"][(v[0] <= 4) as usize],
        }
    }

    /// Test2 is SEALED: every message here carries ids and counts only, never an item.
    #[test]
    fn test2_is_fresh_deterministic_and_well_formed() {
        let t2 = items(Set::Test2, 400);
        assert!(t2 == items(Set::Test2, 400), "Test2 not deterministic");
        assert_eq!(tsv_sha(&t2), "9f11d9b196b773a354b0d2efcb63619ed3c23cd9685417b03dee21d07e6dbcda");
        let count = |k: &str| t2.iter().filter(|x| x.kind == k).count();
        // v39's practice taught every sum and every number word of the test's domains
        assert_eq!(KINDS.map(count), [400, 400, 269, 0, 0, 40]);
        let mut old: HashSet<String> = HashSet::new();
        for set in [Set::Dev, Set::Test] {
            old.extend(items(set, 400).iter().map(key));
        }
        let practice = crate::num_practice::v39_practice_keys();
        assert!(practice.len() > 10_000, "{} practice keys", practice.len());
        let words = words_map();
        let mut seen = HashSet::new();
        let (mut big_first, mut more_true, mut n_cmp) = (0usize, 0usize, 0usize);
        let mut right_pos = [0usize; 4];
        for (i, it) in t2.iter().enumerate() {
            assert!(it.id as usize == i, "id {i}");
            let k = key(it);
            assert!(!old.contains(&k), "item {i}: a Dev/Test key");
            assert!(!practice.contains(&k), "item {i}: a v39 practice key");
            assert!(seen.insert(k.clone()), "item {i}: duplicate key");
            if it.kind == "stars" {
                assert!(k.starts_with("stars:=") || k.starts_with("stars:~"), "item {i}: stars symbol");
            } else {
                assert!(key_half(&k) == 1, "item {i}: wrong hash half");
            }
            let n_choices = if it.kind.starts_with("compare") { 2 } else { 4 };
            assert!(it.choices.len() == n_choices, "item {i}: choice count");
            let truths: Vec<usize> = (0..n_choices).filter(|&c| is_true(it.kind, &it.choices[c], &words)).collect();
            assert!(truths == vec![it.right], "item {i}: not exactly one true choice at right");
            let mut c = it.choices.clone();
            c.sort();
            c.dedup();
            assert!(c.len() == n_choices, "item {i}: duplicate choices");
            assert!(it.choices.iter().all(|s| !s.contains(['\t', '\n', '\r'])), "item {i}: control chars");
            assert!(implied_bin(it.kind, &it.choices[it.right]) == it.bin, "item {i}: bin");
            right_pos[it.right] += 1;
            if it.kind.starts_with("compare") {
                // v39's fix: a minimal pair, same numbers in the same order, only more/less differs
                let (c0, c1) = (&it.choices[0], &it.choices[1]);
                assert!(numbers_in(c0) == numbers_in(c1), "item {i}: not the same numbers");
                assert!(c0.replace(" less ", " more ") == c1.replace(" less ", " more ") && c0 != c1, "item {i}: not a minimal pair");
                let v = numbers_in(c0);
                n_cmp += 1;
                big_first += (v[0] > v[1]) as usize;
                more_true += it.choices[it.right].contains(" more ") as usize;
            }
        }
        assert_eq!(big_first, more_true);
        assert!(n_cmp == 800 && (320..=480).contains(&big_first), "{big_first} of {n_cmp} larger first");
        // 800 two-choice items (positions 0, 1) and 309 four-choice items (0..=3), shuffled
        assert!(right_pos[2] > 40 && right_pos[3] > 40 && right_pos[0] > 360 && right_pos[1] > 360, "right positions {right_pos:?}");
        // the trap half of compare_mixed survived the closed cells
        let traps = t2
            .iter()
            .filter(|x| x.kind == "compare_mixed")
            .filter(|x| {
                let v = numbers_in(&x.choices[x.right]);
                leading_digit(v[0].min(v[1])) > leading_digit(v[0].max(v[1]))
            })
            .count();
        assert_eq!(traps, 200);
    }

    /// Test3 is SEALED: every message here carries ids and counts only, never an item.
    #[test]
    fn test3_is_fresh_deterministic_and_well_formed() {
        let t3 = items(Set::Test3, 400);
        assert!(t3 == items(Set::Test3, 400), "Test3 not deterministic");
        assert_eq!(tsv_sha(&t3), TEST3_SHA256);
        let count = |k: &str| t3.iter().filter(|x| x.kind == k).count();
        // v40's practice counted through all but two successor keys of the test's half
        assert_eq!(KINDS.map(count), [400, 400, 2, 0, 0, 40]);
        let mut old: HashSet<String> = HashSet::new();
        for set in [Set::Dev, Set::Test, Set::Test2] {
            old.extend(items(set, 400).iter().map(key));
        }
        let p39 = crate::num_practice::v39_practice_keys();
        let p40 = crate::num_practice::v40_practice_keys();
        assert!(p40.len() > 10_000, "{} v40 practice keys", p40.len());
        let words = words_map();
        let mut seen = HashSet::new();
        for (i, it) in t3.iter().enumerate() {
            assert!(it.id as usize == i, "id {i}");
            let k = key(it);
            assert!(!old.contains(&k), "item {i}: a Dev/Test/Test2 key");
            assert!(!p39.contains(&k) && !p40.contains(&k), "item {i}: a practice key");
            assert!(seen.insert(k.clone()), "item {i}: duplicate key");
            if it.kind == "stars" {
                assert!(k.starts_with("stars:^") || k.starts_with("stars:|"), "item {i}: stars symbol");
            } else {
                assert!(key_half(&k) == 1, "item {i}: wrong hash half");
            }
            let n_choices = if it.kind.starts_with("compare") { 2 } else { 4 };
            assert!(it.choices.len() == n_choices, "item {i}: choice count");
            let truths: Vec<usize> = (0..n_choices).filter(|&c| is_true(it.kind, &it.choices[c], &words)).collect();
            assert!(truths == vec![it.right], "item {i}: not exactly one true choice at right");
            assert!(implied_bin(it.kind, &it.choices[it.right]) == it.bin, "item {i}: bin");
            assert!(it.choices.iter().all(|s| !s.contains(['\t', '\n', '\r'])), "item {i}: control chars");
        }
        let traps = t3
            .iter()
            .filter(|x| x.kind == "compare_mixed")
            .filter(|x| {
                let v = numbers_in(&x.choices[x.right]);
                leading_digit(v[0].min(v[1])) > leading_digit(v[0].max(v[1]))
            })
            .count();
        assert_eq!(traps, 200);
    }

    /// sha256 of `items(Test3, 400)` as written by `num_items --set test3` (data/num/sealed/test3.tsv).
    const TEST3_SHA256: &str = "de78c1b61f8a52f5192f9bf797845e99b5fa0b56b31744a6886da0831e0621ef";
}
