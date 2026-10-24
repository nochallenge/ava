// TESTING / DESIGN ONLY.
//
// CHAIN — composition where it can actually show up (v19 §3 said the dialogue exam has no
// room for it: copy is 1.64 points, class pooling is noise).
//
//   cargo run --release --example chain -- [noise-percent] [withheld-percent]
//
// The morphology wug test works because the answer is an OPERATION ("add -ed") applied to
// something never seen. This is the same test for knowledge. A synthetic family tree:
// ava stores every MOTHER and FATHER fact, plus grandparent facts for HALF the people.
// Then she is asked for the grandparents of the OTHER half — keys she has never seen.
//
//   direct recall    : the key has no address, so the general route answers. ~0% expected.
//   composed         : induce a PROGRAM (a chain of base relations, up to 3 hops) from her
//                      own stored examples, then run it through her memory, one recall per
//                      hop. the program's image holds (query, answer) pairs no memory holds,
//                      which is v16's definition of composition.
//
// Three of the four faculties are in this one loop: COMPOSITION (the program), GOALS (a
// multi-step plan that knows where it is going), SELF-MODELLING (a hop with no address
// means "I don't know" — she abstains instead of guessing). A control relation with NO
// rule (godparent, assigned at random) must find no program and abstain everywhere.
use design_tests::seq;
use design_tests::v6::*;
use std::env;
use std::time::Instant;

const GENS: usize = 6;
const PER_GEN: usize = 2000;
const PER: usize = 2;
const BASE: [&str; 2] = ["mother", "father"];
/// derived relations and their TRUE programs. the truth is used only to generate facts
/// and to grade; the induction never sees it.
const DERIVED: [(&str, &[usize]); 5] = [
    ("maternal_grandmother", &[0, 0]),
    ("maternal_grandfather", &[0, 1]),
    ("paternal_grandmother", &[1, 0]),
    ("paternal_grandfather", &[1, 1]),
    ("great_grandmother", &[0, 0, 0]),
];
/// a program is adopted when, among her stored examples where it ran to completion, it
/// reproduces the stored answer at least ADOPT of the time AND beats the runner-up program
/// by MARGIN — "explains the examples far better than any alternative". (the first version
/// used a flat 0.9: at 10% wrong examples the true rule fits 89.5% and was rejected.)
const ADOPT: f64 = 0.5;
const MARGIN: f64 = 0.2;
const MIN_SUPPORT: usize = 50;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
    fn chance(&mut self, pct: f64) -> bool { ((self.next() % 10_000) as f64) < pct * 100.0 }
}

struct Mem { fac: Factored, ro: OpReadout }
impl Mem {
    fn key(rel: &str, x: &str) -> (Vec<String>, Vec<String>) {
        (vec![format!("R:{rel}|X:{x}")], vec![format!("R:{rel}")])
    }
    fn settle(&mut self, rel: &str, x: &str) {
        let (id, g) = Mem::key(rel, x);
        self.fac.learn_feats_each(&id, &g, PER);
    }
    fn code(&self, rel: &str, x: &str) -> Vec<NeuronId> {
        let (id, g) = Mem::key(rel, x);
        let (idc, gc) = self.fac.query_feats_each(&id, &g, PER);
        seq::code_of(idc, &gc)
    }
    fn write(&mut self, rel: &str, x: &str, y: &str) {
        let c = self.code(rel, x);
        self.ro.learn(&c, y);
    }
    /// recall through the neurons. None = no address exists: she knows she does not know.
    fn recall(&self, rel: &str, x: &str) -> Option<String> {
        if !self.fac.id.feat_id.contains_key(&format!("R:{rel}|X:{x}")) { return None; }
        self.ro.decode(&self.code(rel, x), 1).into_iter().next().map(|(w, _)| w)
    }
    /// what she says if made to answer anyway — the general route, for unseen keys.
    fn guess(&self, rel: &str, x: &str) -> Option<String> {
        self.ro.decode(&self.code(rel, x), 1).into_iter().next().map(|(w, _)| w)
    }
}

/// every base recall she can make, read once through the neurons and cached — the
/// program search asks the same hop thousands of times.
type Memo = std::collections::HashMap<String, [Option<String>; 2]>;
fn run(memo: &Memo, prog: &[usize], x: &str) -> Option<String> {
    let mut cur = x.to_string();
    for &b in prog { cur = memo.get(&cur)?[b].clone()?; }
    Some(cur)
}

fn show(prog: &[usize]) -> String {
    // read inside-out: [mother, father] = the father of the mother
    prog.iter().rev().map(|&b| BASE[b]).collect::<Vec<_>>().join(" of ")
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let noise: f64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let withheld: f64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let t0 = Instant::now();
    let mut rng = Rng(0xC4A1_2026_1013_0001);

    // ---- the world. person g*PER_GEN + i; even i female, odd i male.
    let n = GENS * PER_GEN;
    let name: Vec<String> = (0..n).map(|p| format!("p{}_{}", p / PER_GEN, p % PER_GEN)).collect();
    let mut parent = vec![[usize::MAX; 2]; n];
    for g in 1..GENS {
        for i in 0..PER_GEN {
            let p = g * PER_GEN + i;
            parent[p][0] = (g - 1) * PER_GEN + 2 * rng.below(PER_GEN / 2);
            parent[p][1] = (g - 1) * PER_GEN + 2 * rng.below(PER_GEN / 2) + 1;
        }
    }
    let truth = |prog: &[usize], x: usize| -> Option<usize> {
        let mut c = x;
        for &b in prog { c = parent[c][b]; if c == usize::MAX { return None; } }
        Some(c)
    };
    let train_person: Vec<bool> = (0..n).map(|_| rng.below(2) == 0).collect();

    // ---- what she is told.
    let mut facts: Vec<(&str, usize, String)> = Vec::new();
    let mut held_base = 0usize;
    for p in PER_GEN..n {
        for b in 0..2 {
            if rng.chance(withheld) { held_base += 1; continue; }   // a missing link
            facts.push((BASE[b], p, name[parent[p][b]].clone()));
        }
    }
    let n_base = facts.len();
    let mut n_noisy = 0usize;
    for (rel, prog) in DERIVED {
        for p in 0..n {
            if !train_person[p] { continue; }
            let Some(y) = truth(prog, p) else { continue };
            let y = if rng.chance(noise) {
                n_noisy += 1;
                (y / PER_GEN) * PER_GEN + rng.below(PER_GEN)               // a wrong person, same generation
            } else { y };
            facts.push((rel, p, name[y].clone()));
        }
    }
    let mut godparent = vec![usize::MAX; n];
    for p in PER_GEN..n {
        godparent[p] = (p / PER_GEN - 1) * PER_GEN + rng.below(PER_GEN);
        if train_person[p] { facts.push(("godparent", p, name[godparent[p]].clone())); }
    }
    println!("== chain: {n} people, {GENS} generations ==");
    println!("   told {} base facts ({held_base} withheld = missing links), {} derived facts \
for the training half ({n_noisy} deliberately wrong)", n_base, facts.len() - n_base);

    // ---- learn, exactly as deep.rs does: two settle passes, then a frozen write.
    let mut m = Mem { fac: Factored::new(2_000_000, 100_000, 6, 2), ro: OpReadout::default() };
    for _ in 0..2 { for (r, p, _) in &facts { m.settle(r, &name[*p]); } }
    for (r, p, y) in &facts { m.write(r, &name[*p], y); }
    println!("   learned at {:?}", t0.elapsed());
    let ok = facts.iter().filter(|(r, p, y)| m.recall(r, &name[*p]).as_deref() == Some(y.as_str())).count();
    println!("   recall of what she was told: {:.2}% ({ok}/{})\n", ok as f64 / facts.len() as f64 * 100.0, facts.len());

    let memo: Memo = name.iter().map(|x| (x.clone(), [m.recall(BASE[0], x), m.recall(BASE[1], x)])).collect();

    // ---- induce, then answer the held-out half.
    let mut progs: Vec<Vec<usize>> = Vec::new();
    for len in 1..=3 {
        for code in 0..(1usize << len) { progs.push((0..len).map(|k| (code >> k) & 1).collect()); }
    }
    println!("   {:<21} {:<34} {:>9} {:>8}   held out: {:>7} {:>9} {:>8} {:>7}",
        "relation", "induced rule", "train-fit", "support", "direct", "composed", "answered", "wrong");
    let rels: Vec<(&str, Option<&[usize]>)> = DERIVED.iter().map(|(r, p)| (*r, Some(*p)))
        .chain(std::iter::once(("godparent", None))).collect();
    let mut tot = [0usize; 5];   // queries, direct right, composed right, answered, wrong
    for (rel, true_prog) in &rels {
        // her OWN stored examples, read back through recall (not the list she was given).
        let examples: Vec<(usize, String)> = facts.iter().filter(|f| f.0 == *rel)
            .filter_map(|f| m.recall(rel, &name[f.1]).map(|s| (f.1, s))).collect();
        // score every program on her OWN stored examples, read back through recall.
        let mut fits: Vec<(usize, f64, usize)> = Vec::new();
        for (pi, prog) in progs.iter().enumerate() {
            let (mut ran, mut agree) = (0usize, 0usize);
            for (p, stored) in &examples {
                if let Some(got) = run(&memo, prog, &name[*p]) {
                    ran += 1;
                    if &got == stored { agree += 1; }
                }
            }
            if ran < MIN_SUPPORT { continue; }
            fits.push((pi, agree as f64 / ran as f64, ran));
        }
        fits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let best = fits.first().copied();
        let second = fits.get(1).map_or(0.0, |s| s.1);
        let adopted = best.filter(|b| b.1 >= ADOPT && b.1 - second >= MARGIN).map(|b| &progs[b.0]);
        let truth_of = |p: usize| -> Option<usize> {
            match true_prog { Some(tp) => truth(tp, p), None => if godparent[p] == usize::MAX { None } else { Some(godparent[p]) } }
        };
        let mut c = [0usize; 5];
        for p in 0..n {
            if train_person[p] { continue; }
            let Some(y) = truth_of(p) else { continue };
            c[0] += 1;
            if m.guess(rel, &name[p]).as_deref() == Some(name[y].as_str()) { c[1] += 1; }
            if let Some(prog) = adopted {
                if let Some(got) = run(&memo, prog, &name[p]) {
                    c[3] += 1;
                    if got == name[y] { c[2] += 1; } else { c[4] += 1; }
                }
            }
        }
        if true_prog.is_some() { for i in 0..5 { tot[i] += c[i]; } }   // the control is reported, not pooled
        let pc = |x: usize| x as f64 / c[0].max(1) as f64 * 100.0;
        let rule = match (adopted, best) {
            (Some(p), _) => format!("{}{}", show(p),
                if true_prog.map_or(false, |t| t == p.as_slice()) { "  (correct)" } else { "  (WRONG)" }),
            (None, Some(b)) => format!("none (best fit {:.0}%)", b.1 * 100.0),
            (None, None) => "none".into(),
        };
        println!("   {:<21} {:<34} {:>8.1}% {:>8}   {:>16.2}% {:>8.2}% {:>7.1}% {:>7}",
            rel, rule, best.map_or(0.0, |b| b.1 * 100.0), best.map_or(0, |b| b.2),
            pc(c[1]), pc(c[2]), pc(c[3]), c[4]);
    }
    let pc = |x: usize| x as f64 / tot[0].max(1) as f64 * 100.0;
    println!("\n   all {} held-out queries on the five real relations: direct recall {:.2}%  ->  composed {:.2}%  \
(answered {:.1}%, precision {:.2}%, {} wrong answers)", tot[0], pc(tot[1]), pc(tot[2]), pc(tot[3]),
        tot[2] as f64 / tot[3].max(1) as f64 * 100.0, tot[4]);

    // ---- a few, in words.
    println!();
    for p in [3 * PER_GEN + 7, 4 * PER_GEN + 12, 5 * PER_GEN + 33] {
        let q = &name[p];
        for (rel, prog) in [("maternal_grandfather", &[0usize, 1][..]), ("great_grandmother", &[0, 0, 0][..])] {
            let said = run(&memo, prog, q).unwrap_or_else(|| "I don't know".into());
            let real = truth(prog, p).map_or("-".into(), |y| name[y].clone());
            println!("   you> who is {q}'s {rel}?   ava> {said}   (truth {real}{})",
                if train_person[p] { ", taught" } else { ", never taught" });
        }
    }
    println!("\nTSV\tchain\t{noise}\t{withheld}\t{:.2}\t{:.2}\t{:.1}\t{}", pc(tot[1]), pc(tot[2]), pc(tot[3]), tot[4]);
    families(withheld);
}

// ---------------------------------------------------------------- part 2: families
//
// Everything above is functional: one mother, one answer. Real relations are SETS. Here
// people form couples and couples have 0..n children, so `child` has many answers, and
// the rules need an INVERSE (child) and an EXCLUSION ("my siblings are my mother's
// children, minus me"). The alphabet grows to {mother, father, child} x up to 3 hops x
// {exclude nobody, exclude the start, exclude the node after hop 1}: 117 programs.
// Missing links are now silent: a withheld child fact does not remove an address, it
// shrinks a set, so the answer is incomplete rather than refused.

const F_GENS: usize = 5;
const COUPLES: usize = 600;
const F_REL: [&str; 3] = ["mother", "father", "child"];
/// (relation, true program, excluded step) — for grading only, never shown to induction.
const F_DERIVED: [(&str, &[usize], Option<usize>); 4] = [
    ("grandchild", &[2, 2], None),
    ("sibling", &[0, 2], Some(0)),
    ("maternal_aunt_or_uncle", &[0, 0, 2], Some(1)),
    ("paternal_aunt_or_uncle", &[1, 0, 2], Some(1)),
];

type Set = Vec<usize>;
fn apply(memo: &[[Option<Set>; 3]], prog: &[usize], excl: Option<usize>, x: usize) -> Option<Set> {
    let mut cur: Set = vec![x];
    let mut drop: Set = Vec::new();
    for (k, &r) in prog.iter().enumerate() {
        if excl == Some(k) { drop = cur.clone(); }
        let mut next: Set = Vec::new();
        for &c in &cur {
            match (&memo[c][r], r) {
                (Some(s), _) => next.extend(s.iter().copied()),
                (None, 2) => {}               // no child on record: no children (closed world)
                (None, _) => return None,     // no mother on record: unknown, so refuse
            }
        }
        next.sort_unstable();
        next.dedup();
        cur = next;
    }
    cur.retain(|c| !drop.contains(c));
    Some(cur)
}

fn families(withheld: f64) {
    let t0 = Instant::now();
    let mut rng = Rng(0xFA31_2026_1013_0002);
    let per = 2 * COUPLES;
    let n = F_GENS * per;
    let name: Vec<String> = (0..n).map(|p| format!("f{}_{}", p / per, p % per)).collect();
    // person g*per + i; couple c of a generation is (2c, 2c+1): mother, father.
    let mut par = vec![[usize::MAX; 2]; n];
    let mut kids: Vec<Set> = vec![Vec::new(); n];
    for g in 1..F_GENS {
        for i in 0..per {
            let p = g * per + i;
            let c = rng.below(COUPLES);
            par[p] = [(g - 1) * per + 2 * c, (g - 1) * per + 2 * c + 1];
            kids[par[p][0]].push(p);
            kids[par[p][1]].push(p);
        }
    }
    let base = |r: usize, x: usize| -> Set {
        if r < 2 { if par[x][r] == usize::MAX { vec![] } else { vec![par[x][r]] } } else { kids[x].clone() }
    };
    let truth = |prog: &[usize], excl: Option<usize>, x: usize| -> Set {
        let mut cur: Set = vec![x];
        let mut drop: Set = Vec::new();
        for (k, &r) in prog.iter().enumerate() {
            if excl == Some(k) { drop = cur.clone(); }
            let mut next: Set = cur.iter().flat_map(|&c| base(r, c)).collect();
            next.sort_unstable();
            next.dedup();
            cur = next;
        }
        cur.retain(|c| !drop.contains(c));
        cur
    };
    let train_person: Vec<bool> = (0..n).map(|_| rng.below(2) == 0).collect();

    let mut facts: Vec<(&str, usize, usize)> = Vec::new();
    let mut held = 0usize;
    for x in 0..n {
        for r in 0..3 {
            for y in base(r, x) {
                if rng.chance(withheld) { held += 1; continue; }
                facts.push((F_REL[r], x, y));
            }
        }
    }
    let n_base = facts.len();
    for (rel, prog, excl) in F_DERIVED {
        for x in 0..n {
            if !train_person[x] { continue; }
            for y in truth(prog, excl, x) { facts.push((rel, x, y)); }
        }
    }
    // the control: two random "friends" in the same generation. no rule produces them.
    let mut friends: Vec<Set> = vec![Vec::new(); n];
    for x in 0..n {
        let g = x / per;
        let mut f: Set = (0..2).map(|_| g * per + rng.below(per)).filter(|&y| y != x).collect();
        f.sort_unstable();
        f.dedup();
        if train_person[x] { for &y in &f { facts.push(("friend", x, y)); } }
        friends[x] = f;
    }
    println!("\n== families: {n} people, {F_GENS} generations, {COUPLES} couples each; relations are SETS ==");
    println!("   told {n_base} base facts ({held} withheld, which silently shrink sets), {} derived facts \
for the training half", facts.len() - n_base);

    let mut m = Mem { fac: Factored::new(2_000_000, 100_000, 6, 2), ro: OpReadout::default() };
    for _ in 0..2 { for (r, x, _) in &facts { m.settle(r, &name[*x]); } }
    for (r, x, y) in &facts { m.write(r, &name[*x], &name[*y]); }
    // SET RECALL: every answer scoring at least half the best. a key with three children
    // splits its votes three ways and all three clear the bar; the general route's crowd
    // does not.
    let id_of: std::collections::HashMap<&str, usize> = name.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let recall_set = |rel: &str, x: &str, keyed: bool| -> Option<Set> {
        if keyed && !m.fac.id.feat_id.contains_key(&format!("R:{rel}|X:{x}")) { return None; }
        let top = m.ro.decode(&m.code(rel, x), 32);
        let best = top.first()?.1;
        let mut s: Set = top.iter().filter(|(_, v)| *v >= 0.5 * best)
            .filter_map(|(w, _)| id_of.get(w.as_str()).copied()).collect();
        s.sort_unstable();
        Some(s)
    };
    let memo: Vec<[Option<Set>; 3]> = (0..n).map(|x| {
        [recall_set("mother", &name[x], true), recall_set("father", &name[x], true), recall_set("child", &name[x], true)]
    }).collect();
    let told: Vec<(usize, usize, usize)> = facts.iter().take(n_base)
        .map(|(r, x, y)| (F_REL.iter().position(|q| q == r).unwrap(), *x, *y)).collect();
    let ok = told.iter().filter(|(r, x, y)| memo[*x][*r].as_ref().map_or(false, |s| s.contains(y))).count();
    println!("   learned at {:?}; set recall holds {:.2}% of the base facts she was told", t0.elapsed(),
        ok as f64 / told.len() as f64 * 100.0);

    let mut progs: Vec<(Vec<usize>, Option<usize>)> = Vec::new();
    for len in 1..=3usize {
        let mut code = vec![0usize; len];
        loop {
            for ex in [None, Some(0), Some(1)] {
                if ex == Some(1) && len < 2 { continue; }
                progs.push((code.clone(), ex));
            }
            let mut k = 0;
            while k < len { code[k] += 1; if code[k] < 3 { break; } code[k] = 0; k += 1; }
            if k == len { break; }
        }
    }
    let show = |p: &(Vec<usize>, Option<usize>)| -> String {
        let s = p.0.iter().rev().map(|&r| F_REL[r]).collect::<Vec<_>>().join(" of ");
        match p.1 { None => s, Some(0) => format!("{s}, not me"),
                    Some(k) => format!("{s}, not my {}", p.0[..k].iter().rev().map(|&r| F_REL[r]).collect::<Vec<_>>().join(" of ")) }
    };
    println!("   {} candidate programs\n", progs.len());
    println!("   {:<24} {:<40} {:>8}   held out: {:>6} {:>8} {:>9} {:>7}",
        "relation", "induced rule", "fit", "direct", "exact", "precision", "recall");
    let rels: Vec<(&str, Option<(&[usize], Option<usize>)>)> = F_DERIVED.iter().map(|(r, p, e)| (*r, Some((*p, *e))))
        .chain(std::iter::once(("friend", None))).collect();
    for (rel, tp) in &rels {
        let mut xs: Vec<usize> = facts.iter().filter(|f| f.0 == *rel).map(|f| f.1).collect();
        xs.dedup();
        let examples: Vec<(usize, Set)> = xs.iter().filter_map(|&x| recall_set(rel, &name[x], true).map(|s| (x, s))).collect();
        let outs: Vec<Vec<Option<Set>>> = progs.iter()
            .map(|p| examples.iter().map(|(x, _)| apply(&memo, &p.0, p.1, *x)).collect()).collect();
        // OVERLAP (jaccard) of two answer sets. with facts missing, an exact-set score is
        // too harsh: the true rule loses a name here and there and scores 41% at 15%
        // missing. overlap gives partial credit, and a wrong rule still scores ~0.
        let jac = |x: &Option<Set>, y: &Option<Set>| -> f64 {
            match (x, y) {
                (None, None) => 1.0,
                (Some(a), Some(b)) => {
                    let i = a.iter().filter(|v| b.contains(v)).count();
                    let u = a.len() + b.len() - i;
                    if u == 0 { 1.0 } else { i as f64 / u as f64 }
                }
                _ => 0.0,
            }
        };
        let mut fits: Vec<(usize, f64)> = outs.iter().enumerate().filter_map(|(pi, o)| {
            let (mut ran, mut ok) = (0usize, 0f64);
            for (got, (_, s)) in o.iter().zip(&examples) {
                if got.is_some() { ran += 1; ok += jac(got, &Some(s.clone())); }
            }
            if ran < MIN_SUPPORT { None } else { Some((pi, ok / ran as f64)) }
        }).collect();
        fits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let best = fits.first().copied();
        // the runner-up must be a DIFFERENT hypothesis. "mother's children, not me" and
        // "father's children, not me" are one rule spelled twice — they differ only where a
        // withheld fact hit one spelling and not the other — and a margin against your own
        // synonym can never be won. a RIVAL is a program that disagrees with the best on at
        // least half its answers on average — it predicts mostly DIFFERENT people.
        let agree = |a: usize, b: usize| outs[a].iter().zip(&outs[b]).map(|(x, y)| jac(x, y)).sum::<f64>()
            / examples.len().max(1) as f64;
        let second = fits.iter().skip(1)
            .find(|s| best.map_or(true, |b| agree(s.0, b.0) < 0.5)).map_or(0.0, |s| s.1);
        let adopted = best.filter(|b| b.1 >= ADOPT && b.1 - second >= MARGIN).map(|b| &progs[b.0]);
        let (mut q, mut direct, mut exact, mut hit, mut said, mut want) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        for x in 0..n {
            if train_person[x] { continue; }
            let t = match tp { Some((p, e)) => truth(p, *e, x), None => friends[x].clone() };
            if t.is_empty() { continue; }
            q += 1;
            want += t.len();
            if recall_set(rel, &name[x], false).as_ref() == Some(&t) { direct += 1; }
            if let Some(p) = adopted {
                if let Some(o) = apply(&memo, &p.0, p.1, x) {
                    if o == t { exact += 1; }
                    said += o.len();
                    hit += o.iter().filter(|y| t.contains(y)).count();
                }
            }
        }
        let rule = match (adopted, best) {
            (Some(p), _) => {
                let same = tp.map_or(false, |(tp, te)| tp == p.0.as_slice() && te == p.1);
                // a different spelling of the true rule: same answers on every example
                let synonym = tp.map_or(false, |(tp, te)| progs.iter().position(|q| q.0 == tp && q.1 == te)
                    .map_or(false, |ti| best.map_or(false, |b| agree(ti, b.0) >= 0.5)));
                format!("{}{}", show(p), if same { "  (correct)" } else if synonym { "  (= true rule)" } else { "" })
            }
            (None, Some(b)) => format!("none (best fit {:.0}%)", b.1 * 100.0),
            (None, None) => "none".into(),
        };
        let pc = |a: usize, b: usize| a as f64 / b.max(1) as f64 * 100.0;
        println!("   {:<24} {:<40} {:>7.1}%   {:>15.2}% {:>7.2}% {:>8.2}% {:>6.2}%",
            rel, rule, best.map_or(0.0, |b| b.1 * 100.0), pc(direct, q), pc(exact, q), pc(hit, said), pc(hit, want));
        println!("TSV\tfamily\t{rel}\t{:.2}\t{:.2}\t{:.2}\t{:.2}", pc(direct, q), pc(exact, q), pc(hit, said), pc(hit, want));
    }
}
