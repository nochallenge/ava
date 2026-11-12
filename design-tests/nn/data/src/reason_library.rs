// TESTING / DESIGN ONLY
//! reason_library (D11): a growing program library, DreamCoder style (Ellis et al. 2021), on a tiny list
//! language.
//!
//! LANGUAGE. Steps on a list of whole numbers: map(+k) for k 2..5, map(-k) for k 1..3 (fails below 0),
//! map(*k) for k 3..5, map(sq), map(%k) for k 3 and 5, filter(>k), filter(<k), filter(odd), filter(x3)
//! (multiples of 3), take(k) and drop(k) for k 2..4 and 2..3, dedup. A program is steps joined by " | ",
//! and may end in a fold to one number: fold(+), fold(*), fold(max), fold(min), fold(count).
//!
//! A DOCUMENT is one session: 8 to 12 tasks, each given by four input/output examples. The tasks share
//! two hidden motifs (two-step pieces), so some need 4 or 5 steps. Round 1 searches every program of up
//! to 3 steps, smallest first (bottom-up, programs that give the same outputs on the examples counted
//! once), and checks each find on a held-out list ("fits the examples but fails the check" is said when it
//! happens). Then COMPRESSION: the two-step piece that appears most often in the solutions becomes a new
//! named library word (f1, f2, ...) with its meaning in plain words, and the search reruns with it as one
//! step: solutions get shorter and tasks beyond 3 steps become reachable. Up to three rounds.
//!
//! AWAY FROM THE A6 TEST (rules from examples on digit lists): no step coincides with a test rule. There is
//! no +1, no *2, no keep-even, no drop(1), no reverse, sort or rotate; a target, a found program or a
//! library word that acts like one of the test's rules on lists of 3 or 4 digits is never used (checked
//! against copies of the test's rules, which only ever reject).

use crate::reason::{pick, range};
use crate::reason_guard::{test_rule, TEST_RULES};
use crate::rng::SplitMix64;
use std::collections::{HashMap, HashSet};

const CAP_VALUE: u32 = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prim {
    Add(u32),
    Sub(u32),
    Mul(u32),
    Square,
    Mod(u32),
    Gt(u32),
    Lt(u32),
    Odd,
    Mult3,
    Take(usize),
    Drop(usize),
    Dedup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fold {
    Sum,
    Product,
    Max,
    Min,
    Count,
}

pub const FOLDS: [Fold; 5] = [Fold::Sum, Fold::Product, Fold::Max, Fold::Min, Fold::Count];

/// Every primitive step.
pub fn prims() -> Vec<Prim> {
    use Prim::*;
    let mut v = Vec::new();
    v.extend((2..=5).map(Add));
    v.extend((1..=3).map(Sub));
    v.extend((3..=5).map(Mul));
    v.push(Square);
    v.extend([3, 5].map(Mod));
    v.extend((2..=8).map(Gt));
    v.extend((3..=9).map(Lt));
    v.push(Odd);
    v.push(Mult3);
    v.extend((2..=4).map(Take));
    v.extend((2..=3).map(Drop));
    v.push(Dedup);
    v
}

impl Prim {
    pub fn apply(&self, x: &[u32]) -> Option<Vec<u32>> {
        use Prim::*;
        let cap = |v: u32| (v <= CAP_VALUE).then_some(v);
        Some(match *self {
            Add(k) => x.iter().map(|&d| cap(d + k)).collect::<Option<_>>()?,
            Sub(k) => x.iter().map(|&d| d.checked_sub(k)).collect::<Option<_>>()?,
            Mul(k) => x.iter().map(|&d| d.checked_mul(k).and_then(cap)).collect::<Option<_>>()?,
            Square => x.iter().map(|&d| d.checked_mul(d).and_then(cap)).collect::<Option<_>>()?,
            Mod(k) => x.iter().map(|&d| d % k).collect(),
            Gt(k) => x.iter().copied().filter(|&d| d > k).collect(),
            Lt(k) => x.iter().copied().filter(|&d| d < k).collect(),
            Odd => x.iter().copied().filter(|&d| d % 2 == 1).collect(),
            Mult3 => x.iter().copied().filter(|&d| d % 3 == 0).collect(),
            Take(k) => x.iter().copied().take(k).collect(),
            Drop(k) => x.iter().copied().skip(k).collect(),
            Dedup => {
                let mut v: Vec<u32> = Vec::new();
                for &d in x {
                    if !v.contains(&d) {
                        v.push(d);
                    }
                }
                v
            }
        })
    }

    pub fn code(&self) -> String {
        use Prim::*;
        match *self {
            Add(k) => format!("map(+{k})"),
            Sub(k) => format!("map(-{k})"),
            Mul(k) => format!("map(*{k})"),
            Square => "map(sq)".into(),
            Mod(k) => format!("map(%{k})"),
            Gt(k) => format!("filter(>{k})"),
            Lt(k) => format!("filter(<{k})"),
            Odd => "filter(odd)".into(),
            Mult3 => "filter(x3)".into(),
            Take(k) => format!("take({k})"),
            Drop(k) => format!("drop({k})"),
            Dedup => "dedup".into(),
        }
    }

    pub fn words(&self) -> String {
        use Prim::*;
        match *self {
            Add(k) => format!("add {k} to each number"),
            Sub(k) => format!("take {k} away from each number"),
            Mul(k) => format!("multiply each number by {k}"),
            Square => "square each number".into(),
            Mod(k) => format!("replace each number by its remainder after dividing by {k}"),
            Gt(k) => format!("keep the numbers above {k}"),
            Lt(k) => format!("keep the numbers below {k}"),
            Odd => "keep the odd numbers".into(),
            Mult3 => "keep the multiples of 3".into(),
            Take(k) => format!("keep the first {k}"),
            Drop(k) => format!("leave out the first {k}"),
            Dedup => "remove repeats".into(),
        }
    }
}

impl Fold {
    pub fn apply(&self, x: &[u32]) -> Option<u32> {
        match self {
            Fold::Sum => x.iter().try_fold(0u32, |a, &b| a.checked_add(b)),
            Fold::Product => x.iter().try_fold(1u32, |a, &b| a.checked_mul(b).filter(|&v| v <= CAP_VALUE)),
            Fold::Max => x.iter().copied().max(),
            Fold::Min => x.iter().copied().min(),
            Fold::Count => Some(x.len() as u32),
        }
    }
    pub fn code(&self) -> &'static str {
        match self {
            Fold::Sum => "fold(+)",
            Fold::Product => "fold(*)",
            Fold::Max => "fold(max)",
            Fold::Min => "fold(min)",
            Fold::Count => "fold(count)",
        }
    }
    pub fn words(&self) -> &'static str {
        match self {
            Fold::Sum => "add them all up",
            Fold::Product => "multiply them all together",
            Fold::Max => "take the largest",
            Fold::Min => "take the smallest",
            Fold::Count => "count them",
        }
    }
}

/// One list step: a primitive or a library word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Step {
    P(Prim),
    L(usize),
}

/// A library word: its definition in steps (which may use earlier words) and its primitives.
#[derive(Clone, Debug)]
pub struct Word {
    pub def: Vec<Step>,
    pub flat: Vec<Prim>,
}

#[derive(Clone, Debug, Default)]
pub struct Library {
    pub words: Vec<Word>,
}

impl Library {
    pub fn flatten(&self, steps: &[Step]) -> Vec<Prim> {
        steps
            .iter()
            .flat_map(|s| match s {
                Step::P(p) => vec![*p],
                Step::L(i) => self.words[*i].flat.clone(),
            })
            .collect()
    }
    pub fn apply(&self, s: &Step, x: &[u32]) -> Option<Vec<u32>> {
        match s {
            Step::P(p) => p.apply(x),
            Step::L(i) => run_prims(&self.words[*i].flat, x),
        }
    }
    pub fn code(&self, s: &Step) -> String {
        match s {
            Step::P(p) => p.code(),
            Step::L(i) => format!("f{}", i + 1),
        }
    }
    pub fn words_of(&self, prims: &[Prim]) -> String {
        prims.iter().map(|p| p.words()).collect::<Vec<_>>().join(", then ")
    }
}

pub fn run_prims(ps: &[Prim], x: &[u32]) -> Option<Vec<u32>> {
    let mut v = x.to_vec();
    for p in ps {
        v = p.apply(&v)?;
    }
    Some(v)
}

/// A program: list steps, then maybe a fold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prog {
    pub steps: Vec<Step>,
    pub fold: Option<Fold>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Out {
    List(Vec<u32>),
    Num(u32),
}

impl Out {
    fn text(&self) -> String {
        match self {
            Out::List(v) => format!("[{}]", v.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
            Out::Num(n) => n.to_string(),
        }
    }
}

impl Prog {
    pub fn size(&self) -> usize {
        self.steps.len() + self.fold.is_some() as usize
    }
    pub fn run(&self, lib: &Library, x: &[u32]) -> Option<Out> {
        let mut v = x.to_vec();
        for s in &self.steps {
            v = lib.apply(s, &v)?;
        }
        Some(match self.fold {
            Some(f) => Out::Num(f.apply(&v)?),
            None => Out::List(v),
        })
    }
    pub fn code(&self, lib: &Library) -> String {
        let mut parts: Vec<String> = self.steps.iter().map(|s| lib.code(s)).collect();
        if let Some(f) = self.fold {
            parts.push(f.code().to_string());
        }
        if parts.is_empty() {
            "id".into()
        } else {
            parts.join(" | ")
        }
    }
}

/// Does a list program act like one of the A6 test's rules on 3- or 4-digit lists?
pub fn acts_like_a_test_rule(ps: &[Prim]) -> bool {
    for len in [3usize, 4] {
        let mut r = SplitMix64::new(0x11b ^ len as u64);
        let lists: Vec<Vec<u32>> = (0..150).map(|_| (0..len).map(|_| r.below(10) as u32).collect()).collect();
        if TEST_RULES.iter().any(|t| lists.iter().all(|x| run_prims(ps, x) == test_rule(t, x))) {
            return true;
        }
    }
    false
}

pub struct Task {
    pub inputs: Vec<Vec<u32>>,
    pub outputs: Vec<Out>,
    pub check_in: Vec<u32>,
    pub check_out: Out,
}

/// What a search returned.
pub struct Found {
    pub prog: Option<Prog>,
    /// Programs tried (before removing ones with the same outputs).
    pub tried: usize,
    /// Programs that fit the examples but failed the held-out check.
    pub near_misses: Vec<Prog>,
}

/// Bottom-up search, smallest first, up to `max` steps (a fold counts as one).
pub fn search(lib: &Library, t: &Task, max: usize) -> Found {
    let mut alphabet: Vec<Step> = prims().into_iter().map(Step::P).collect();
    alphabet.extend((0..lib.words.len()).map(Step::L));
    let all_in: Vec<Vec<u32>> = t.inputs.iter().cloned().chain(std::iter::once(t.check_in.clone())).collect();
    let n_ex = t.inputs.len();
    let want_num = matches!(t.outputs[0], Out::Num(_));
    let mut tried = 0usize;
    let mut near = Vec::new();
    let mut seen: HashSet<Vec<Vec<u32>>> = HashSet::new();
    let mut level: Vec<(Vec<Step>, Vec<Vec<u32>>)> = vec![(Vec::new(), all_in.clone())];
    seen.insert(all_in);
    // does a value vector fit the examples / the check?
    let fits = |vals: &[Out]| vals[..n_ex] == t.outputs[..];
    for size in 1..=max {
        let mut next: Vec<(Vec<Step>, Vec<Vec<u32>>)> = Vec::new();
        // folds end a program: list programs of size-1, then one fold
        if want_num {
            for (steps, vals) in &level {
                for f in FOLDS {
                    tried += 1;
                    let outs: Option<Vec<Out>> = vals.iter().map(|v| f.apply(v).map(Out::Num)).collect();
                    let Some(outs) = outs else { continue };
                    if fits(&outs) {
                        let p = Prog { steps: steps.clone(), fold: Some(f) };
                        if outs[n_ex] == t.check_out {
                            return Found { prog: Some(p), tried, near_misses: near };
                        }
                        near.push(p);
                    }
                }
            }
        }
        for (steps, vals) in &level {
            for s in &alphabet {
                tried += 1;
                let nv: Option<Vec<Vec<u32>>> = vals.iter().map(|v| lib.apply(s, v)).collect();
                let Some(nv) = nv else { continue };
                if !want_num {
                    let outs: Vec<Out> = nv.iter().map(|v| Out::List(v.clone())).collect();
                    if fits(&outs) {
                        let mut st = steps.clone();
                        st.push(*s);
                        let p = Prog { steps: st, fold: None };
                        if outs[n_ex] == t.check_out {
                            return Found { prog: Some(p), tried, near_misses: near };
                        }
                        near.push(p);
                        continue;
                    }
                }
                // a longer program is only needed below the cap (a fold may still follow)
                let room = if want_num { size < max } else { size < max };
                if room && seen.insert(nv.clone()) {
                    let mut st = steps.clone();
                    st.push(*s);
                    next.push((st, nv));
                }
            }
        }
        level = next;
    }
    Found { prog: None, tried, near_misses: near }
}

fn input(r: &mut SplitMix64) -> Vec<u32> {
    let n = range(r, 4, 6) as usize;
    (0..n).map(|_| range(r, 0, 9) as u32).collect()
}

/// A hidden target: 1 to 3 parts, each a motif (two steps) or one primitive; 2 to 5 primitives in all;
/// a fold at the end 30% of the time.
fn target(r: &mut SplitMix64, motifs: &[Vec<Prim>], ps: &[Prim]) -> (Vec<Prim>, Option<Fold>) {
    loop {
        let parts = range(r, 1, 3);
        let mut v = Vec::new();
        for _ in 0..parts {
            if r.below(10) < 6 {
                v.extend(pick(r, motifs).iter().copied());
            } else {
                v.push(*pick(r, ps));
            }
        }
        if (2..=5).contains(&v.len()) {
            let fold = (r.below(10) < 3).then(|| *pick(r, &FOLDS));
            return (v, fold);
        }
    }
}

fn make_task(r: &mut SplitMix64, body: &[Prim], fold: Option<Fold>) -> Option<Task> {
    let lib = Library::default();
    let prog = Prog { steps: body.iter().map(|&p| Step::P(p)).collect(), fold };
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut tries = 0;
    while inputs.len() < 5 && tries < 40 {
        tries += 1;
        let x = input(r);
        let Some(y) = prog.run(&lib, &x) else { continue };
        if let Out::List(v) = &y {
            if v.is_empty() || v.len() > 6 || *v == x {
                continue;
            }
        }
        if !inputs.contains(&x) {
            inputs.push(x);
            outputs.push(y);
        }
    }
    if inputs.len() < 5 {
        return None;
    }
    let check_in = inputs.pop().unwrap();
    let check_out = outputs.pop().unwrap();
    // outputs must vary (a constant answer teaches nothing)
    if outputs.iter().all(|o| *o == outputs[0]) {
        return None;
    }
    Some(Task { inputs, outputs, check_in, check_out })
}

/// The most useful two-step piece in the solutions (count >= 2), as surface steps.
fn compress(lib: &Library, sols: &[Option<Prog>]) -> Option<Vec<Step>> {
    let mut count: HashMap<Vec<Step>, usize> = HashMap::new();
    let mut order: Vec<Vec<Step>> = Vec::new();
    for p in sols.iter().flatten() {
        let mut in_this: HashSet<Vec<Step>> = HashSet::new();
        for w in p.steps.windows(2) {
            let k = w.to_vec();
            if in_this.insert(k.clone()) {
                if !count.contains_key(&k) {
                    order.push(k.clone());
                }
                *count.entry(k).or_insert(0) += 1;
            }
        }
    }
    let ps = prims();
    order
        .into_iter()
        .filter(|k| count[k] >= 2)
        .filter(|k| {
            let flat = lib.flatten(k);
            // not one primitive in disguise, not a test rule
            let mut r = SplitMix64::new(77);
            let lists: Vec<Vec<u32>> = (0..60).map(|_| input(&mut r)).collect();
            !ps.iter().any(|p| lists.iter().all(|x| p.apply(x) == run_prims(&flat, x))) && !acts_like_a_test_rule(&flat)
        })
        .max_by_key(|k| (count[k], std::cmp::Reverse(0)))
}

fn show_list(v: &[u32]) -> String {
    format!("[{}]", v.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
}

pub fn doc(r: &mut SplitMix64) -> String {
    let ps = prims();
    loop {
        // two hidden motifs
        let mut motifs: Vec<Vec<Prim>> = Vec::new();
        while motifs.len() < 2 {
            let m = vec![*pick(r, &ps), *pick(r, &ps)];
            if m[0] != m[1] && !acts_like_a_test_rule(&m) && !motifs.contains(&m) {
                motifs.push(m);
            }
        }
        let n_tasks = range(r, 8, 12) as usize;
        let mut tasks: Vec<Task> = Vec::new();
        let mut guard = 0;
        while tasks.len() < n_tasks && guard < 200 {
            guard += 1;
            let (body, fold) = target(r, &motifs, &ps);
            if acts_like_a_test_rule(&body) {
                continue;
            }
            if let Some(t) = make_task(r, &body, fold) {
                tasks.push(t);
            }
        }
        if tasks.len() < 6 {
            continue;
        }
        let mut out: Vec<String> = Vec::new();
        out.push("Program search with a growing library.".to_string());
        out.push(
            "Steps: map(+k), map(-k), map(*k), map(sq), map(%k), filter(>k), filter(<k), filter(odd), filter(x3), take(k), drop(k), dedup; a program may end with fold(+), fold(*), fold(max), fold(min) or fold(count). Steps are joined with |."
                .to_string(),
        );
        for (i, t) in tasks.iter().enumerate() {
            let ex: Vec<String> = t.inputs.iter().zip(&t.outputs).map(|(x, y)| format!("{} gives {}", show_list(x), y.text())).collect();
            out.push(format!("Task {}: {}.", i + 1, ex.join("; ")));
        }
        let mut lib = Library::default();
        let max = 3;
        let mut sols: Vec<Option<Prog>> = vec![None; tasks.len()];
        let mut bad = false;
        for round in 1..=3 {
            let names: Vec<String> = (0..lib.words.len()).map(|i| format!("f{}", i + 1)).collect();
            out.push(if round == 1 {
                format!("Round 1. Library: the built-in steps only. Search every program of up to {max} steps, smallest first, and check each find on a held-out list.")
            } else {
                format!("Round {round}. Library now also has {}. Search again, up to {max} steps, a library word counting as one step.", names.join(", "))
            });
            let mut new_sols = Vec::new();
            for (i, t) in tasks.iter().enumerate() {
                let f = search(&lib, t, max);
                if let Some(p) = &f.prog {
                    if acts_like_a_test_rule(&lib.flatten(&p.steps)) && p.fold.is_none() {
                        bad = true;
                    }
                }
                let prev = &sols[i];
                let line = match (&f.prog, prev) {
                    (Some(p), None) => {
                        let mut s = format!("Task {}: {} ({} {}), after {} programs.", i + 1, p.code(&lib), p.size(), if p.size() == 1 { "step" } else { "steps" }, f.tried);
                        if let Some(m) = f.near_misses.first() {
                            s.push_str(&format!(
                                " Before it, {} fit the examples but gave {} on the check list {}, not {}.",
                                m.code(&lib),
                                m.run(&lib, &t.check_in).map(|o| o.text()).unwrap_or_else(|| "nothing".into()),
                                show_list(&t.check_in),
                                t.check_out.text()
                            ));
                        }
                        if round > 1 {
                            s.push_str(" Not solved before.");
                        }
                        s
                    }
                    (Some(p), Some(q)) if p.size() < q.size() => format!("Task {}: now {} ({} steps instead of {}).", i + 1, p.code(&lib), p.size(), q.size()),
                    (Some(_), Some(_)) => format!("Task {}: same as before.", i + 1),
                    (None, _) => format!("Task {}: nothing within {max} steps.", i + 1),
                };
                out.push(line);
                new_sols.push(f.prog.or_else(|| prev.clone()));
            }
            sols = new_sols;
            let solved = sols.iter().filter(|s| s.is_some()).count();
            out.push(format!("Solved after round {round}: {solved} of {}.", tasks.len()));
            if round == 3 || solved == tasks.len() && round > 1 {
                break;
            }
            match compress(&lib, &sols) {
                Some(piece) => {
                    let flat = lib.flatten(&piece);
                    let uses = sols.iter().flatten().filter(|p| p.steps.windows(2).any(|w| w == &piece[..])).count();
                    let def = piece.iter().map(|s| lib.code(s)).collect::<Vec<_>>().join(" | ");
                    let name = format!("f{}", lib.words.len() + 1);
                    out.push(format!(
                        "Compress: the piece {def} appears in {uses} solutions. New library word {name} = {def}. In plain words, {name} means: {}.",
                        lib.words_of(&flat)
                    ));
                    lib.words.push(Word { def: piece, flat });
                }
                None => {
                    out.push("Compress: no piece appears in two solutions, so the library stays as it is.".to_string());
                    break;
                }
            }
        }
        if bad {
            continue;
        }
        if !lib.words.is_empty() {
            let defs: Vec<String> = lib
                .words
                .iter()
                .enumerate()
                .map(|(i, w)| format!("f{} = {} ({})", i + 1, w.def.iter().map(|s| lib.code(s)).collect::<Vec<_>>().join(" | "), lib.words_of(&w.flat)))
                .collect();
            out.push(format!("Library learned: {}.", defs.join("; ")));
        }
        return out.join("\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_and_folds() {
        assert_eq!(Prim::Gt(4).apply(&[3, 8, 5]), Some(vec![8, 5]));
        assert_eq!(Prim::Sub(2).apply(&[1, 5]), None);
        assert_eq!(Prim::Dedup.apply(&[2, 2, 3, 2]), Some(vec![2, 3]));
        assert_eq!(Fold::Max.apply(&[]), None);
        assert_eq!(Fold::Count.apply(&[4, 4]), Some(2));
        assert_eq!(Prog { steps: vec![Step::P(Prim::Mul(3)), Step::P(Prim::Gt(10))], fold: Some(Fold::Sum) }.run(&Library::default(), &[2, 4, 5]), Some(Out::Num(27)));
    }

    /// No single step does what an A6 test rule does (on the test's 3- and 4-digit lists).
    #[test]
    fn no_step_is_a_test_rule() {
        for p in prims() {
            assert!(!acts_like_a_test_rule(&[p]), "{p:?}");
        }
        // the checker itself works: +1 would act like plus-one on lists without a 9? no: the test wraps 9 to 0
        assert!(!acts_like_a_test_rule(&[Prim::Add(2)]));
    }

    #[test]
    fn search_finds_the_smallest_fitting_program() {
        let lib = Library::default();
        let body = [Prim::Gt(4), Prim::Mul(3)];
        let p = Prog { steps: body.iter().map(|&q| Step::P(q)).collect(), fold: None };
        let ins = vec![vec![3, 8, 1, 6], vec![5, 2, 9, 7], vec![9, 9, 0, 4], vec![6, 1, 2, 8]];
        let outs: Vec<Out> = ins.iter().map(|x| p.run(&lib, x).unwrap()).collect();
        let t = Task { inputs: ins, outputs: outs, check_in: vec![7, 3, 5], check_out: p.run(&lib, &[7, 3, 5]).unwrap() };
        let f = search(&lib, &t, 3);
        let got = f.prog.unwrap();
        assert_eq!(got.size(), 2);
        for (x, y) in t.inputs.iter().zip(&t.outputs) {
            assert_eq!(got.run(&lib, x).as_ref(), Some(y));
        }
        // with a library word for the piece it is one step
        let lib2 = Library { words: vec![Word { def: vec![Step::P(Prim::Gt(4)), Step::P(Prim::Mul(3))], flat: body.to_vec() }] };
        assert_eq!(search(&lib2, &t, 3).prog.unwrap().size(), 1);
    }

    #[test]
    fn sessions_grow_a_library() {
        let mut r = SplitMix64::new(6);
        let mut grew = 0;
        let mut later = 0;
        for _ in 0..12 {
            let d = doc(&mut r);
            if d.contains("New library word f1") {
                grew += 1;
            }
            if d.contains("Not solved before.") || d.contains("steps instead of") {
                later += 1;
            }
            assert!(!d.contains("->"));
        }
        assert!(grew >= 6, "{grew}");
        assert!(later >= 3, "{later}");
    }
}
