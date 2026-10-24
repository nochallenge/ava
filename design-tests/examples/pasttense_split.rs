// TESTING / DESIGN ONLY. Her 92.6% on withheld past tenses, split open.
//
//   cargo run --release --example pasttense_split
//
// exam.rs scores her on the past-tense forms of 1 lemma in 10, withheld from training, and
// she gets 92.6%. English past tense is nearly a spelling rule, so the number means little
// until it is split into REGULAR forms (a rule could do them) and IRREGULAR ones (only memory
// or analogy can), and set beside rules that know nothing:
//   R0 modal op      the commonest edit for the tag set in training (data-derived)
//   R1 add -ed       e -> +d, consonant+y -> ied, else +ed (the textbook spelling rule)
//   R2 suffix analogy the commonest edit among training lemmas sharing the longest final
//                    letters with the same tag set (data-derived, counting only)
// Training is exam.rs's, line for line, so her generalization number must come out the same.
use design_tests::living::Fact;
use design_tests::v4::{self, Params};
use design_tests::v6::*;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::time::Instant;

const CORPUS: &str = "../data/unimorph-eng.tsv";
const N_ID: u32 = 10_000_000;
const N_GRAM: u32 = 200_000;
const K_ID: usize = 10;
const PER_TAG: usize = 3;
const TRAIN: usize = 652_477;

fn tagkey(tags: &[String]) -> String {
    let mut ts: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
    ts.sort();
    ts.join(",")
}

fn vowel(c: char) -> bool { "aeiou".contains(c) }

/// the regular spellings of a past form: +ed, +d after e, y -> ied after a consonant, and a
/// doubled final consonant (stop -> stopped). any other form is irregular.
fn regular_forms(l: &str) -> Vec<String> {
    let c: Vec<char> = l.chars().collect();
    let mut v = vec![format!("{l}ed")];
    if let Some(&last) = c.last() {
        if last == 'e' { v.push(format!("{l}d")); }
        if last == 'y' && c.len() >= 2 && !vowel(c[c.len() - 2]) { v.push(format!("{}ied", &l[..l.len() - 1])); }
        if !vowel(last) && last.is_alphabetic() { v.push(format!("{l}{last}ed")); }
    }
    v
}

/// R1, the textbook rule (no doubling: knowing when to double needs stress)
fn add_ed(l: &str) -> String {
    let c: Vec<char> = l.chars().collect();
    match c.last() {
        Some('e') => format!("{l}d"),
        Some('y') if c.len() >= 2 && !vowel(c[c.len() - 2]) => format!("{}ied", &l[..l.len() - 1]),
        _ => format!("{l}ed"),
    }
}

fn pct(a: usize, b: usize) -> f64 { if b == 0 { 0.0 } else { a as f64 / b as f64 * 100.0 } }

fn main() -> std::io::Result<()> {
    let raw = fs::read_to_string(CORPUS)?;
    let all: Vec<Fact> = raw.lines().filter_map(Fact::parse).take(TRAIN).collect();
    // exam.rs's split, exactly: every 10th lemma in sorted order, its PST forms withheld
    let mut lemmas: Vec<&str> = all.iter().map(|f| f.lemma.as_str()).collect();
    lemmas.sort_unstable();
    lemmas.dedup();
    let mut held: HashSet<String> = HashSet::new();
    for (i, l) in lemmas.iter().enumerate() { if i % 10 == 0 { held.insert((*l).to_string()); } }
    let is_held = |f: &Fact| held.contains(&f.lemma) && f.tags.iter().any(|t| t == "PST");
    let train: Vec<&Fact> = all.iter().filter(|f| !is_held(f)).collect();
    let test: Vec<&Fact> = all.iter().filter(|f| is_held(f)).collect();
    println!("training on {} facts ({} held back)", train.len(), test.len());

    // exam.rs's training, line for line
    let t0 = Instant::now();
    let mut fac = Factored::new(N_ID, N_GRAM, K_ID, PER_TAG);
    let mut ro = OpReadout::default();
    let mut brain = v4::Brain::new(fac.n_total, Params::default());
    for _ in 0..2 {
        for f in &train {
            fac.learn_code(&Fields { lemma: &f.lemma, form: Some(&f.form), tags: &f.tags });
        }
    }
    for f in &train {
        let (idc, gc) = fac.query_code(&Fields { lemma: &f.lemma, form: None, tags: &f.tags }, K_ID);
        let mut full = idc; full.extend_from_slice(&gc);
        brain.learn_bulk(&full, 1);
        ro.learn(&full, &edit_op(&f.lemma, &f.form));
    }
    brain.compact();
    println!("trained in {:.1?}", t0.elapsed());
    let ask = |lemma: &str, tags: &[String]| -> Option<String> {
        let (idc, gc) = fac.query_code(&Fields { lemma, form: None, tags }, K_ID);
        let mut cue = idc; cue.extend_from_slice(&gc);
        ro.decode(&cue, 1).into_iter().filter_map(|(op, _)| apply_op(lemma, &op)).next()
    };

    // R0: commonest op per tag set; R2: commonest op per (tag set, final letters), longest first
    let mut modal: HashMap<String, HashMap<String, u32>> = HashMap::new();
    let mut by_suffix: HashMap<(String, String), HashMap<String, u32>> = HashMap::new();
    for f in &train {
        let (k, op) = (tagkey(&f.tags), edit_op(&f.lemma, &f.form));
        *modal.entry(k.clone()).or_default().entry(op.clone()).or_insert(0) += 1;
        let c: Vec<char> = f.lemma.chars().collect();
        for n in 1..=4.min(c.len()) {
            let suf: String = c[c.len() - n..].iter().collect();
            *by_suffix.entry((k.clone(), suf)).or_default().entry(op.clone()).or_insert(0) += 1;
        }
    }
    // ties broken by the op string, so the result never depends on hash order
    let best = |m: &HashMap<String, u32>| m.iter().max_by(|x, y| x.1.cmp(y.1).then(y.0.cmp(x.0))).map(|(op, _)| op.clone());
    let r0 = |l: &str, tags: &[String]| modal.get(&tagkey(tags)).and_then(|m| best(m)).and_then(|op| apply_op(l, &op));
    let r2 = |l: &str, tags: &[String]| -> Option<String> {
        let c: Vec<char> = l.chars().collect();
        for n in (1..=4.min(c.len())).rev() {
            let suf: String = c[c.len() - n..].iter().collect();
            if let Some(m) = by_suffix.get(&(tagkey(tags), suf)) { return best(m).and_then(|op| apply_op(l, &op)); }
        }
        r0(l, tags)
    };

    // a (lemma, tag set) with two or more withheld forms (dreamed / dreamt): a single answer
    // can match only one of them, so those are reported apart
    let mut n_forms: HashMap<(String, String), usize> = HashMap::new();
    for f in &test { *n_forms.entry((f.lemma.clone(), tagkey(&f.tags))).or_default() += 1; }
    // leak check: is the withheld form present in training as another form of the same lemma?
    let mut lemma_forms: HashMap<&str, HashSet<&str>> = HashMap::new();
    for f in &train { lemma_forms.entry(f.lemma.as_str()).or_default().insert(f.form.as_str()); }

    // [regular, irregular, variant] x [count, ava, r0, r1, r2, leaked]
    let mut t = [[0usize; 6]; 3];
    let (mut irr_right, mut irr_wrong) = (Vec::new(), Vec::new());
    let mut ava_ok = 0;
    for f in &test {
        let cat = if n_forms[&(f.lemma.clone(), tagkey(&f.tags))] > 1 { 2 }
                  else if regular_forms(&f.lemma).contains(&f.form) { 0 } else { 1 };
        let a = ask(&f.lemma, &f.tags);
        let ok = [a.as_deref() == Some(f.form.as_str()),
                  r0(&f.lemma, &f.tags).as_deref() == Some(f.form.as_str()),
                  add_ed(&f.lemma) == f.form,
                  r2(&f.lemma, &f.tags).as_deref() == Some(f.form.as_str()),
                  lemma_forms.get(f.lemma.as_str()).map_or(false, |s| s.contains(f.form.as_str()))];
        t[cat][0] += 1;
        for (k, o) in ok.iter().enumerate() { if *o { t[cat][k + 1] += 1; } }
        if ok[0] { ava_ok += 1; }
        if cat == 1 {
            let line = format!("{} {} -> {} (said {})", f.lemma, tagkey(&f.tags), f.form, a.as_deref().unwrap_or("-"));
            if ok[0] { irr_right.push(line) } else { irr_wrong.push(line) }
        }
    }
    println!("\nher generalization: {ava_ok}/{} = {:.2}%  (exam.rs reports 92.6%)", test.len(), pct(ava_ok, test.len()));
    println!("\n{:<10} {:>6}  {:>8} {:>10} {:>9} {:>12}   {}", "forms", "count", "Ava", "R0 modal", "R1 -ed", "R2 analogy", "form seen elsewhere");
    let names = ["regular", "irregular", "variants"];
    let mut tot = [0usize; 6];
    for (c, row) in t.iter().enumerate() {
        for k in 0..6 { tot[k] += row[k]; }
        println!("{:<10} {:>6}  {:>7.1}% {:>9.1}% {:>8.1}% {:>11.1}%   {:>5}", names[c], row[0], pct(row[1], row[0]), pct(row[2], row[0]), pct(row[3], row[0]), pct(row[4], row[0]), row[5]);
    }
    println!("{:<10} {:>6}  {:>7.1}% {:>9.1}% {:>8.1}% {:>11.1}%   {:>5}", "all", tot[0], pct(tot[1], tot[0]), pct(tot[2], tot[0]), pct(tot[3], tot[0]), pct(tot[4], tot[0]), tot[5]);
    println!("\nirregular she got right ({}), first 15:", irr_right.len());
    for l in irr_right.iter().take(15) { println!("  {l}"); }
    println!("irregular she got wrong ({}), first 25:", irr_wrong.len());
    for l in irr_wrong.iter().take(25) { println!("  {l}"); }
    Ok(())
}
