// TESTING / DESIGN ONLY
//! reason_rules (A6): rule induction. A few examples of a list rule, candidate rules checked against them
//! one by one (each rejected with the example that breaks it), the rule that fits, then the rule applied
//! to new lists. A quarter of the rules are two rules in a row ("drop the last number, then add 3").
//!
//! AWAY FROM THE TEST. The A6 test uses ten rules (sort up, sort down, reverse, plus one with wrap-around,
//! rotate left, double, drop first, repeat last, swap first two, keep even), three examples, digit lists
//! of 3 or 4, and the layout "3 5 2 -> 2 3 5". The rules here are a DISJOINT set (a unit test runs each of
//! them, and each two-rule chain, against copies of the test's ten on random lists and asserts it differs
//! from every one of them), lists are 3 to 7 long with numbers up to 20, there are 2 to 5 examples, and
//! the layout is "in [4, 9, 2] out [4, 9]" or a sentence; "->" never appears. A list never holds more
//! than 7 numbers, so no 13 numbers ever stand in a row (no 13-word overlap with the digit-only test items).

use crate::reason::{pick, range};
use crate::reason_guard::{test_rule, TEST_RULES};
use crate::rng::SplitMix64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    DropLast,
    KeepOdd,
    AddK(u32),
    SubK(u32),
    Triple,
    RotateRight,
    SwapLastTwo,
    DupFirst,
    Dedupe,
    AppendSum,
    DropMax,
    KeepAbove(u32),
    HalveEvens,
    Mirror,
    DupEach,
    FirstLast,
    MaxToFront,
    NeighbourSums,
    ZeroBetween,
    TenMinus,
    Middle,
    AddPosition,
}

/// Every rule, with its parameters filled in.
pub fn all_rules() -> Vec<Rule> {
    use Rule::*;
    let mut v = vec![DropLast, KeepOdd, Triple, RotateRight, SwapLastTwo, DupFirst, Dedupe, AppendSum, DropMax, HalveEvens, Mirror, DupEach, FirstLast, MaxToFront, NeighbourSums, ZeroBetween, TenMinus, Middle, AddPosition];
    v.extend((2..=5).map(AddK));
    v.extend((1..=3).map(SubK));
    v.extend((3..=7).map(KeepAbove));
    v
}

impl Rule {
    pub fn apply(&self, x: &[u32]) -> Option<Vec<u32>> {
        use Rule::*;
        if x.is_empty() {
            return None;
        }
        let v: Vec<u32> = match *self {
            DropLast => x[..x.len() - 1].to_vec(),
            KeepOdd => x.iter().copied().filter(|d| d % 2 == 1).collect(),
            AddK(k) => x.iter().map(|d| d + k).collect(),
            SubK(k) => x.iter().map(|&d| d.checked_sub(k)).collect::<Option<Vec<u32>>>()?,
            Triple => x.iter().map(|d| d * 3).collect(),
            RotateRight => {
                let mut v = x.to_vec();
                v.rotate_right(1);
                v
            }
            SwapLastTwo => {
                if x.len() < 2 {
                    return None;
                }
                let mut v = x.to_vec();
                let n = v.len();
                v.swap(n - 1, n - 2);
                v
            }
            DupFirst => std::iter::once(x[0]).chain(x.iter().copied()).collect(),
            Dedupe => {
                let mut v: Vec<u32> = Vec::new();
                for &d in x {
                    if !v.contains(&d) {
                        v.push(d);
                    }
                }
                v
            }
            AppendSum => x.iter().copied().chain(std::iter::once(x.iter().sum())).collect(),
            DropMax => {
                let m = *x.iter().max()?;
                let i = x.iter().position(|&d| d == m)?;
                x.iter().enumerate().filter(|&(k, _)| k != i).map(|(_, &d)| d).collect()
            }
            KeepAbove(k) => x.iter().copied().filter(|&d| d > k).collect(),
            HalveEvens => x.iter().map(|&d| if d % 2 == 0 { d / 2 } else { d }).collect(),
            Mirror => x.iter().copied().chain(x.iter().rev().copied()).collect(),
            DupEach => x.iter().flat_map(|&d| [d, d]).collect(),
            FirstLast => vec![x[0], x[x.len() - 1]],
            MaxToFront => {
                let m = *x.iter().max()?;
                let i = x.iter().position(|&d| d == m)?;
                std::iter::once(m).chain(x.iter().enumerate().filter(|&(k, _)| k != i).map(|(_, &d)| d)).collect()
            }
            NeighbourSums => x.windows(2).map(|w| w[0] + w[1]).collect(),
            ZeroBetween => {
                let mut v = Vec::new();
                for (i, &d) in x.iter().enumerate() {
                    if i > 0 {
                        v.push(0);
                    }
                    v.push(d);
                }
                v
            }
            TenMinus => x.iter().map(|&d| 10u32.checked_sub(d)).collect::<Option<Vec<u32>>>()?,
            Middle => {
                if x.len() < 3 {
                    return None;
                }
                x[1..x.len() - 1].to_vec()
            }
            AddPosition => x.iter().enumerate().map(|(i, &d)| d + i as u32 + 1).collect(),
        };
        if v.is_empty() || v.len() > 7 {
            None
        } else {
            Some(v)
        }
    }

    pub fn describe(&self) -> String {
        use Rule::*;
        match *self {
            DropLast => "drop the last number".into(),
            KeepOdd => "keep only the odd numbers".into(),
            AddK(k) => format!("add {k} to every number"),
            SubK(k) => format!("take {k} from every number"),
            Triple => "multiply every number by 3".into(),
            RotateRight => "move the last number to the front".into(),
            SwapLastTwo => "swap the last two numbers".into(),
            DupFirst => "write the first number twice".into(),
            Dedupe => "remove repeats, keeping each number the first time it appears".into(),
            AppendSum => "add the total of all the numbers at the end".into(),
            DropMax => "remove the biggest number".into(),
            KeepAbove(k) => format!("keep only the numbers bigger than {k}"),
            HalveEvens => "halve the even numbers and leave the odd ones".into(),
            Mirror => "write the list, then the same list backwards".into(),
            DupEach => "write every number twice".into(),
            FirstLast => "keep only the first and the last number".into(),
            MaxToFront => "move the biggest number to the front".into(),
            NeighbourSums => "add each pair of neighbours".into(),
            ZeroBetween => "put a 0 between every two numbers".into(),
            TenMinus => "replace every number by 10 minus it".into(),
            Middle => "remove the first and the last number".into(),
            AddPosition => "add 1 to the first number, 2 to the second, and so on".into(),
        }
    }
}

/// One rule or two in a row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chain(pub Vec<Rule>);

impl Chain {
    pub fn apply(&self, x: &[u32]) -> Option<Vec<u32>> {
        let mut v = x.to_vec();
        for r in &self.0 {
            v = r.apply(&v)?;
        }
        Some(v)
    }
    pub fn describe(&self) -> String {
        self.0.iter().map(|r| r.describe()).collect::<Vec<_>>().join(", then ")
    }
}

fn input(r: &mut SplitMix64, big: bool) -> Vec<u32> {
    let n = range(r, 3, 6) as usize;
    let hi = if big { 20 } else { 9 };
    (0..n).map(|_| range(r, 0, hi) as u32).collect()
}

fn show(v: &[u32]) -> String {
    format!("[{}]", v.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
}

fn example_line(style: usize, x: &[u32], y: &[u32]) -> String {
    match style {
        0 => format!("in {} out {}", show(x), show(y)),
        1 => format!("Take {} and it becomes {}.", show(x), show(y)),
        _ => format!("The machine takes {} and gives back {}.", show(x), show(y)),
    }
}

/// The first example (by index) a chain gets wrong, with what it would give.
fn breaks(c: &Chain, ex: &[(Vec<u32>, Vec<u32>)]) -> Option<(usize, Option<Vec<u32>>)> {
    ex.iter().enumerate().find_map(|(i, (x, y))| {
        let got = c.apply(x);
        (got.as_ref() != Some(y)).then_some((i, got))
    })
}

fn add_or_sub(q: Rule) -> bool {
    matches!(q, Rule::AddK(_) | Rule::SubK(_))
}

/// Does a chain act like one of the test's rules on every one of 200 random lists of some length 3..=7
/// (digits, and numbers up to 20)? Such chains are never used.
pub fn acts_like_a_test_rule(c: &Chain) -> bool {
    for top in [9i64, 20] {
        for len in 3..=7usize {
            let mut r = SplitMix64::new(0x7e57 ^ (len as u64) << 8 ^ top as u64);
            let lists: Vec<Vec<u32>> = (0..200).map(|_| (0..len).map(|_| range(&mut r, 0, top) as u32).collect()).collect();
            if TEST_RULES.iter().any(|t| lists.iter().all(|x| c.apply(x) == test_rule(t, x))) {
                return true;
            }
        }
    }
    false
}

/// The chains a document may use: every rule alone, and every two different rules in a row, except two
/// add / take-away steps (they merge into one) and any chain that acts like a test rule.
pub fn allowed_chains() -> &'static (Vec<Chain>, Vec<Chain>) {
    static ALLOWED: std::sync::OnceLock<(Vec<Chain>, Vec<Chain>)> = std::sync::OnceLock::new();
    ALLOWED.get_or_init(|| {
        let rules = all_rules();
        let singles: Vec<Chain> = rules.iter().map(|&q| Chain(vec![q])).filter(|c| !acts_like_a_test_rule(c)).collect();
        let mut pairs = Vec::new();
        for &a in &rules {
            for &b in &rules {
                let c = Chain(vec![a, b]);
                if a != b && !(add_or_sub(a) && add_or_sub(b)) && !acts_like_a_test_rule(&c) {
                    pairs.push(c);
                }
            }
        }
        (singles, pairs)
    })
}

pub fn doc(r: &mut SplitMix64) -> String {
    let rules = all_rules();
    let (singles, pairs) = allowed_chains();
    loop {
        let target = if r.below(4) == 0 { pick(r, pairs).clone() } else { pick(r, singles).clone() };
        let big = r.below(3) == 0;
        let n_ex = range(r, 2, 5) as usize;
        let mut ex: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        let mut tries = 0;
        while ex.len() < n_ex && tries < 50 {
            tries += 1;
            let x = input(r, big);
            if let Some(y) = target.apply(&x) {
                if y != x && !ex.iter().any(|e| e.0 == x) {
                    ex.push((x, y));
                }
            }
        }
        if ex.len() < n_ex {
            continue;
        }
        // the examples must leave only the target among all single rules (and the target's own parts)
        let fits_other = rules.iter().any(|&q| {
            let c = Chain(vec![q]);
            c != target && breaks(&c, &ex).is_none()
        });
        if fits_other {
            continue;
        }
        let mut news: Vec<Vec<u32>> = Vec::new();
        for _ in 0..30 {
            let x = input(r, big);
            if target.apply(&x).is_some() && !ex.iter().any(|e| e.0 == x) && !news.contains(&x) {
                news.push(x);
                if news.len() == 2 {
                    break;
                }
            }
        }
        if news.is_empty() {
            continue;
        }
        // no test rule may explain what this document shows
        let shown: Vec<(Vec<u32>, Vec<u32>)> = ex.iter().cloned().chain(news.iter().map(|x| (x.clone(), target.apply(x).unwrap()))).collect();
        if TEST_RULES.iter().any(|t| shown.iter().all(|(x, y)| test_rule(t, x).as_ref() == Some(y))) {
            continue;
        }
        // wrong candidates: single rules (the target's parts first, when it is a chain)
        let mut cands: Vec<Chain> = Vec::new();
        if target.0.len() == 2 {
            cands.push(Chain(vec![target.0[0]]));
            cands.push(Chain(vec![target.0[1]]));
        }
        let mut others: Vec<Rule> = rules.iter().copied().filter(|q| !target.0.contains(q)).collect();
        r.shuffle(&mut others);
        for q in others.into_iter().take(range(r, 1, 3) as usize) {
            cands.push(Chain(vec![q]));
        }
        r.shuffle(&mut cands);
        let style = r.below(3) as usize;
        let mut out = vec![pick(r, &["Find the rule.", "What does the machine do?", "A rule hides behind these examples.", "Spot the pattern in the lists."]).to_string()];
        for (x, y) in &ex {
            out.push(example_line(style, x, y));
        }
        for c in &cands {
            let (i, got) = breaks(c, &ex).expect("only the target fits");
            let why = match got {
                Some(g) => format!("example {} would give {}, not {}", i + 1, show(&g), show(&ex[i].1)),
                None => format!("it cannot even be used on example {}", i + 1),
            };
            out.push(format!("Could it be \"{}\"? No: {why}.", c.describe()));
        }
        out.push(format!("\"{}\" fits every example.", cap_first(&target.describe())));
        out.push(format!("Rule: {}.", target.describe()));
        for x in &news {
            let y = target.apply(x).unwrap();
            out.push(match style {
                0 => format!("New: in {} out {}", show(x), show(&y)),
                1 => format!("So {} becomes {}", show(x), show(&y)),
                _ => format!("Given {}, the machine gives back {}.", show(x), show(&y)),
            });
        }
        return out.join("\n");
    }
}

fn cap_first(s: &str) -> String {
    crate::reason::cap(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_lists(seed: u64, n: usize, lo: usize, hi: usize, top: i64) -> Vec<Vec<u32>> {
        let mut r = SplitMix64::new(seed);
        (0..n).map(|_| (0..range(&mut r, lo as i64, hi as i64)).map(|_| range(&mut r, 0, top) as u32).collect()).collect()
    }

    #[test]
    fn rules_apply() {
        assert_eq!(Rule::DropLast.apply(&[4, 9, 2]), Some(vec![4, 9]));
        assert_eq!(Rule::RotateRight.apply(&[1, 2, 3]), Some(vec![3, 1, 2]));
        assert_eq!(Rule::SwapLastTwo.apply(&[1, 2, 3]), Some(vec![1, 3, 2]));
        assert_eq!(Rule::Dedupe.apply(&[2, 2, 5, 2]), Some(vec![2, 5]));
        assert_eq!(Rule::NeighbourSums.apply(&[1, 2, 3]), Some(vec![3, 5]));
        assert_eq!(Rule::SubK(2).apply(&[1, 5]), None);
        assert_eq!(Rule::TenMinus.apply(&[3, 12]), None);
        assert_eq!(Rule::AddPosition.apply(&[5, 5, 5]), Some(vec![6, 7, 8]));
        assert_eq!(Chain(vec![Rule::DropLast, Rule::AddK(3)]).apply(&[1, 2, 3]), Some(vec![4, 5]));
    }

    /// The disjointness check: no training rule, and no chain a document may use, gives the same output as
    /// any of the A6 test's ten rules on every one of 300 random lists, for lists of the test's own kind
    /// (digits, exactly 3 long, then exactly 4 long) and of this curriculum's kind (3 to 6 long, up to 20).
    #[test]
    fn no_training_rule_equals_a_test_rule() {
        let (singles, pairs) = allowed_chains();
        assert_eq!(singles.len(), all_rules().len(), "every single rule differs from the test's");
        let n = all_rules().len();
        let dropped = n * (n - 1) - pairs.len();
        assert!(dropped < 60, "{dropped} chains dropped");
        for (seed, lo, hi, top) in [(1u64, 3usize, 3usize, 9i64), (2, 4, 4, 9), (3, 3, 6, 20)] {
            let lists = random_lists(seed, 300, lo, hi, top);
            for c in singles.iter().chain(pairs.iter()) {
                for t in TEST_RULES {
                    let same = lists.iter().all(|x| c.apply(x) == test_rule(t, x));
                    assert!(!same, "{:?} acts like the test rule {t}", c);
                }
            }
        }
        // the chains that would act like a test rule on some list length are among the dropped ones
        let c = Chain(vec![Rule::RotateRight, Rule::SwapLastTwo]);
        assert!(acts_like_a_test_rule(&c), "on 3 numbers it reverses");
        assert!(!pairs.contains(&c));
    }

    #[test]
    fn docs_pin_the_rule_and_keep_runs_short() {
        let mut r = SplitMix64::new(8);
        for _ in 0..300 {
            let d = doc(&mut r);
            assert!(d.contains("Rule: "));
            assert!(!d.contains("->"));
            assert!(crate::reason_guard::longest_number_run(&d) <= 8, "{d}");
        }
    }
}
