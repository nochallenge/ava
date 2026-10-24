// TESTING / DESIGN ONLY. (v29 WP10b) Reading a word she never saw, by its parts.
//
//   cargo run --release --example morph_read -- <split 0..9>
//
// An LLM reads a new word through its subword pieces; for her an unknown word was silent. People
// decompose what they read ("corner" primes "corn"). This is her morphology run BACKWARDS: from a
// written form alone to its lemma and its grammar ("inbent" -> inbend, V;PST).
//
// Her units, as in exam.rs but cued by the FORM: an identity unit "F:form" (she has seen this
// word) and grammar units for its last 1-5 letters plus one any-word unit, 2M grammar units, 3 per
// feature. The answer is one joint label: the edit from form to lemma, and the tag set.
//   S  her summed reading (every unit, specificity-weighted)
//   B  the WP5b reading reversed: where she has seen the form, the memory alone; otherwise her
//      ending units most specific first (count-min over a feature's units, Witten-Bell trust)
// split k: every 10th lemma (sorted) from k is withheld with ALL its forms; the test is each
// distinct form of a withheld lemma, right if it matches one of that form's withheld analyses.
// Pre-registered in "ava v29 - in depth" (WP10b): develop on split 0, confirm once on split 2.
use design_tests::living::Fact;
use design_tests::v6::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::time::Instant;

const CORPUS: &str = "../data/unimorph-eng.tsv";
const N_ID: u32 = 10_000_000;
const N_GRAM: u32 = 2_000_000;
const K_ID: usize = 10;
const PER_TAG: usize = 3;
const TRAIN: usize = 652_477;
const ENDS: usize = 5;

fn tagset(tags: &[String]) -> String {
    let mut ts: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
    ts.sort();
    ts.join(",")
}

fn ending(form: &str, n: usize) -> Option<String> {
    let ch: Vec<char> = form.chars().collect();
    (ch.len() >= n).then(|| ch[ch.len() - n..].iter().collect())
}

/// the grammar features of a written form: its endings, and the any-word unit
fn feats(form: &str) -> Vec<String> {
    let mut v = vec!["R0".to_string()];
    for n in 1..=ENDS { if let Some(e) = ending(form, n) { v.push(format!("R{n}:{e}")); } }
    v
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn pct(a: usize, b: usize) -> f64 { if b == 0 { 0.0 } else { a as f64 / b as f64 * 100.0 } }

/// (lemma, tag set) from a joint label "edit#tagset" applied to the form
fn analysis(form: &str, label: &str) -> Option<(String, String)> {
    let (op, ts) = label.split_once('#')?;
    Some((apply_op(form, op)?, ts.to_string()))
}

struct Her { fac: Factored, ro: OpReadout }

impl Her {
    fn cue(&self, form: &str) -> (Vec<NeuronId>, Vec<NeuronId>) {
        self.fac.query_feats(&[format!("F:{form}")], &feats(form), K_ID)
    }
    fn summed(&self, form: &str) -> Option<String> {
        let (idc, gc) = self.cue(form);
        let mut c = idc;
        c.extend_from_slice(&gc);
        self.ro.decode(&c, 1).into_iter().next().map(|(l, _)| l)
    }
    fn memory(&self, form: &str) -> Option<String> {
        let (idc, _) = self.cue(form);
        if idc.is_empty() { return None; }
        self.ro.decode(&idc, 1).into_iter().next().map(|(l, _)| l)
    }
    /// count-min over each feature's units (morph_backoff `level`)
    fn level(&self, feats: &[Vec<NeuronId>]) -> (HashMap<u32, f64>, f64) {
        let (mut cnt, mut tot) = (HashMap::new(), 0.0);
        for units in feats {
            let Some(first) = units.first().and_then(|u| self.ro.votes.get(u)) else { continue };
            for &o in first.keys() {
                let m = units.iter().map(|u| self.ro.votes.get(u).and_then(|v| v.get(&o)).copied().unwrap_or(0)).min().unwrap_or(0);
                if m > 0 { *cnt.entry(o).or_insert(0.0) += m as f64; tot += m as f64; }
            }
        }
        (cnt, tot)
    }
    fn read(&self, form: &str) -> Option<String> {
        if let Some(l) = self.memory(form) { return Some(l); }
        let mut levels: Vec<Vec<String>> = (1..=ENDS).rev().filter_map(|n| ending(form, n).map(|e| vec![format!("R{n}:{e}")])).collect();
        levels.push(vec!["R0".to_string()]);
        let mut p: HashMap<u32, f64> = HashMap::new();
        for feats in levels.iter().rev() {
            let units: Vec<Vec<NeuronId>> = feats.iter().map(|f| self.fac.gram.units(f)).collect();
            let (cnt, tot) = self.level(&units);
            if tot <= 0.0 { continue; }
            let lam = tot / (tot + cnt.len() as f64);
            for v in p.values_mut() { *v *= 1.0 - lam; }
            for (o, c) in cnt { *p.entry(o).or_insert(0.0) += lam * c / tot; }
        }
        // the best label that can be applied to this form
        let mut v: Vec<(u32, f64)> = p.into_iter().collect();
        v.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v.into_iter().map(|(o, _)| self.ro.ops[o as usize].clone()).find(|l| analysis(form, l).is_some())
    }
}

fn main() -> std::io::Result<()> {
    let split: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let raw = fs::read_to_string(CORPUS)?;
    let all: Vec<Fact> = raw.lines().filter_map(Fact::parse).take(TRAIN).collect();
    let mut lemmas: Vec<&str> = all.iter().map(|f| f.lemma.as_str()).collect();
    lemmas.sort_unstable();
    lemmas.dedup();
    let held: HashSet<&str> = lemmas.iter().enumerate().filter(|(i, _)| i % 10 == split % 10).map(|(_, l)| *l).collect();
    let train: Vec<&Fact> = all.iter().filter(|f| !held.contains(f.lemma.as_str())).collect();
    // each distinct withheld form and all its withheld analyses (sorted, so the order is fixed)
    let mut gold: BTreeMap<String, HashSet<(String, String)>> = BTreeMap::new();
    for f in all.iter().filter(|f| held.contains(f.lemma.as_str())) {
        gold.entry(f.form.clone()).or_default().insert((f.lemma.clone(), tagset(&f.tags)));
    }
    println!("split {split}: training on {} facts; reading {} distinct forms of {} withheld lemmas", train.len(), gold.len(), held.len());

    let t0 = Instant::now();
    let mut fac = Factored::new(N_ID, N_GRAM, K_ID, PER_TAG);
    let mut ro = OpReadout::default();
    for _ in 0..2 {
        for f in &train { fac.learn_feats(&[format!("F:{}", f.form)], &feats(&f.form)); }
    }
    for f in &train {
        let (idc, gc) = fac.query_feats(&[format!("F:{}", f.form)], &feats(&f.form), K_ID);
        let mut full = idc;
        full.extend_from_slice(&gc);
        ro.learn(&full, &format!("{}#{}", edit_op(&f.form, &f.lemma), tagset(&f.tags)));
    }
    println!("trained in {:.1?}", t0.elapsed());
    let her = Her { fac, ro };

    // [all, seen as another word's form (homograph), never seen] x [n, S lemma, S full, B lemma, B full, own-lemma, memory full]
    let mut t = [[0usize; 7]; 3];
    let mut diff: Vec<i64> = Vec::with_capacity(gold.len());
    let mut shown: Vec<String> = Vec::new();
    let t1 = Instant::now();
    for (form, g) in &gold {
        let lem_ok = |a: &Option<(String, String)>| a.as_ref().map_or(false, |(l, _)| g.iter().any(|(gl, _)| gl == l));
        let full_ok = |a: &Option<(String, String)>| a.as_ref().map_or(false, |x| g.contains(x));
        let s = her.summed(form).and_then(|l| analysis(form, &l));
        let b = her.read(form).and_then(|l| analysis(form, &l));
        let m = her.memory(form).and_then(|l| analysis(form, &l));
        let own = g.iter().any(|(gl, _)| gl == form);
        let seen = !her.cue(form).0.is_empty();
        for k in [0, if seen { 1 } else { 2 }] {
            t[k][0] += 1;
            t[k][1] += lem_ok(&s) as usize;
            t[k][2] += full_ok(&s) as usize;
            t[k][3] += lem_ok(&b) as usize;
            t[k][4] += full_ok(&b) as usize;
            t[k][5] += own as usize;
            t[k][6] += full_ok(&m) as usize;
        }
        diff.push(full_ok(&b) as i64 - full_ok(&s) as i64);
        if !seen && !own && shown.len() < 30 && lem_ok(&b) {
            let (l, ts) = b.clone().unwrap();
            shown.push(format!("{form} -> {l} {ts}{}", if full_ok(&b) { "" } else { "  (tags wrong)" }));
        }
    }
    let n = diff.len();
    let mut x: u64 = 1;
    let mut boots: Vec<i64> = (0..20_000).map(|_| (0..n).map(|_| diff[(splitmix(&mut x) % n as u64) as usize]).sum()).collect();
    boots.sort_unstable();
    println!("read {} forms in {:.2?} ({:.0} forms/s)\n", n, t1.elapsed(), n as f64 / t1.elapsed().as_secs_f64());
    println!("{:<40} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}", "", "S lemma", "S full", "B lemma", "B full", "own lemma", "memory");
    for (k, name) in ["all withheld forms", "  seen as another word's form", "  never seen"].iter().enumerate() {
        let c = t[k];
        println!("{:<40} {:>8.2}% {:>8.2}% {:>8.2}% {:>8.2}% {:>8.2}% {:>8.2}%", format!("{name} ({})", c[0]),
                 pct(c[1], c[0]), pct(c[2], c[0]), pct(c[3], c[0]), pct(c[4], c[0]), pct(c[5], c[0]), pct(c[6], c[0]));
    }
    println!("B vs S, full analysis: {:+} forms, 95% CI {:+} .. {:+} (paired bootstrap, 20,000, seed 1)", diff.iter().sum::<i64>(), boots[500], boots[19_499]);
    println!("\nunseen words she read (not their own lemma), first 30:");
    for s in &shown { println!("  {s}"); }
    Ok(())
}
