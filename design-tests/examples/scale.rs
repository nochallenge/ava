// TESTING / DESIGN ONLY — not part of ava's real source.
//
// Does making the population bigger make her smarter? Separate the two variables
// and measure, instead of assuming.
//
//   cargo run --release --example scale -- <N_ID> <N_GRAM> <TRAIN_FACTS>
//
//   cargo run --release --example scale -- 200000    50000  120000   <- today
//   cargo run --release --example scale -- 100000000 50000  120000   <- neurons only
//   cargo run --release --example scale -- 200000    50000  652477   <- data only
//   cargo run --release --example scale -- 100000000 200000 652477   <- both
//
// Reports RAM, training time, and the four exam skills that can move, PLUS the
// number that decides whether a bigger population is doing anything at all:
// how many neurons are even REACHABLE. A unit that is in no feature's candidate
// pool can never be recruited, so it is not part of the brain in any sense that
// matters — it is 48 bytes of address space.
use design_tests::living::Fact;
use design_tests::v4::{self, Params};
use design_tests::v6::*;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::time::Instant;

const CORPUS: &str = "../data/unimorph-eng.tsv";
const K_ID: usize = 10;
const PER_TAG: usize = 3;

const REGULAR: &[(&str, &[&str], &str)] = &[
    ("walk", &["V", "PST"], "walked"),
    ("play", &["V", "PST"], "played"),
    ("bake", &["V", "PST"], "baked"),
    ("jump", &["V", "PST"], "jumped"),
];
const NONCE: &[(&str, &[&str], &str)] = &[
    ("wug", &["N", "PL"], "wugs"),
    ("tove", &["N", "PL"], "toves"),
    ("gorp", &["N", "PL"], "gorps"),
    ("zib", &["N", "PL"], "zibs"),
    ("dax", &["N", "PL"], "daxes"),
    ("blick", &["V", "PST"], "blicked"),
    ("florp", &["V", "PST"], "florped"),
    ("nace", &["V", "PST"], "naced"),
];

fn rss_gb() -> f64 {
    use std::process::Command;
    let pid = std::process::id();
    Command::new("powershell")
        .args(["-NoProfile", "-Command", &format!("(Get-Process -Id {pid}).WorkingSet64")])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|b| b as f64 / 1e9)
        .unwrap_or(0.0)
}

fn derive_irregulars(train: &[&Fact]) -> Vec<(String, Vec<String>, String)> {
    let mut modal: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for f in train {
        let mut ts: Vec<&str> = f.tags.iter().map(|s| s.as_str()).collect();
        ts.sort();
        *modal.entry(ts.join(",")).or_default()
            .entry(edit_op(&f.lemma, &f.form)).or_insert(0) += 1;
    }
    let common: HashMap<String, String> = modal.iter()
        .filter_map(|(k, m)| m.iter().max_by_key(|(_, &c)| c).map(|(op, _)| (k.clone(), op.clone())))
        .collect();
    let mut out = Vec::new();
    for f in train {
        let mut ts: Vec<&str> = f.tags.iter().map(|s| s.as_str()).collect();
        ts.sort();
        if common.get(&ts.join(",")) != Some(&edit_op(&f.lemma, &f.form)) {
            out.push((f.lemma.clone(), f.tags.clone(), f.form.clone()));
        }
    }
    out
}

fn pct(a: usize, b: usize) -> f64 { if b == 0 { 0.0 } else { a as f64 / b as f64 * 100.0 } }

fn main() -> std::io::Result<()> {
    let a: Vec<String> = env::args().collect();
    let n_id: u32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let n_gram: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let limit: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(120_000);

    let base = rss_gb();
    let raw = fs::read_to_string(CORPUS)?;
    let all: Vec<Fact> = raw.lines().filter_map(Fact::parse).take(limit).collect();
    drop(raw);

    // same split as the exam: 1 lemma in 10, past tense only, held back
    let lemmas: HashSet<&str> = all.iter().map(|f| f.lemma.as_str()).collect();
    let mut held: HashSet<String> = HashSet::new();
    for (i, l) in lemmas.iter().enumerate() { if i % 10 == 0 { held.insert((*l).to_string()); } }
    let is_held = |f: &Fact| held.contains(&f.lemma) && f.tags.iter().any(|t| t == "PST");
    let train: Vec<&Fact> = all.iter().filter(|f| !is_held(f)).collect();
    let test: Vec<&Fact> = all.iter().filter(|f| is_held(f)).collect();

    println!("== N_ID {n_id}  N_GRAM {n_gram}  =  {} neurons, {} facts ==",
        n_id as u64 + n_gram as u64, train.len());

    let t0 = Instant::now();
    let mut fac = Factored::new(n_id, n_gram, K_ID, PER_TAG);
    let mut ro = OpReadout::default();
    let mut brain = v4::Brain::new(fac.n_total, Params::default());
    let alloc_t = t0.elapsed();
    let after_alloc = rss_gb();
    println!("  allocated the empty population in {alloc_t:?}  ({:.2} GB)", after_alloc - base);

    let t0 = Instant::now();
    for pass in 0..2 {
        for f in &train {
            fac.learn_code(&Fields { lemma: &f.lemma, form: Some(&f.form), tags: &f.tags });
        }
        println!("  settle pass {} done at {:?}", pass + 1, t0.elapsed());
    }
    for f in &train {
        let (idc, gc) = fac.query_code(&Fields { lemma: &f.lemma, form: None, tags: &f.tags }, K_ID);
        let mut full = idc; full.extend_from_slice(&gc);
        brain.learn_bulk(&full, 1);
        ro.learn(&full, &edit_op(&f.lemma, &f.form));
    }
    println!("  write pass done at {:?}, compacting...", t0.elapsed());
    brain.compact();
    let train_t = t0.elapsed();
    let peak = rss_gb();

    // --- is the bigger population actually being used? ---
    // A unit only ever gets recruited if some feature has it in a candidate pool.
    let mut reachable: HashSet<u32> = HashSet::new();
    for pool in &fac.id.cand { for &(u, _) in pool.iter() { reachable.insert(u); } }
    let used = brain.units.iter().filter(|u| !u.out.is_empty()).count();
    let syn = brain.total_synapses();

    println!("\n  --- population ---");
    println!("  identity features interned : {}", fac.id.feat_id.len());
    println!("  identity neurons REACHABLE : {} of {n_id}  ({:.1}% — the rest can never be recruited)",
        reachable.len(), pct(reachable.len(), n_id as usize));
    println!("  neurons actually wired     : {used}  ({:.2}% of the population)",
        pct(used, (n_id + n_gram) as usize));
    println!("  synapses                   : {syn}");
    println!("  RAM                        : {:.2} GB   (trained in {train_t:?})", peak - base);

    let ask = |lemma: &str, tags: &[String]| -> Option<String> {
        let f = Fields { lemma, form: None, tags };
        let (idc, gc) = fac.query_code(&f, K_ID);
        let mut cue = idc; cue.extend_from_slice(&gc);
        ro.decode(&cue, 1).into_iter()
            .filter_map(|(op, _)| apply_op(lemma, &op)).next()
    };
    let ask_s = |l: &str, t: &[&str]| -> Option<String> {
        ask(l, &t.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    };

    println!("\n  --- what she can do ---");
    let sample: Vec<&&Fact> = train.iter().step_by((train.len() / 3000).max(1)).take(3000).collect();
    let m = pct(sample.iter().filter(|f| ask(&f.lemma, &f.tags).as_deref() == Some(f.form.as_str())).count(), sample.len());
    println!("  1 memory          {m:>6.1}%   ({} sampled)", sample.len());

    let g = pct(test.iter().filter(|f| ask(&f.lemma, &f.tags).as_deref() == Some(f.form.as_str())).count(), test.len());
    println!("  2 generalization  {g:>6.1}%   ({} held-out real words)", test.len());

    let r = pct(REGULAR.iter().filter(|(l, t, w)| ask_s(l, t).as_deref() == Some(*w)).count(), REGULAR.len());
    println!("  3 regular rules   {r:>6.1}%");

    let irr = derive_irregulars(&train);
    let irs: Vec<&(String, Vec<String>, String)> =
        irr.iter().step_by((irr.len() / 800).max(1)).take(800).collect();
    let i = pct(irs.iter().filter(|(l, t, w)| ask(l, t).as_deref() == Some(w.as_str())).count(), irs.len());
    println!("  4 irregulars      {i:>6.1}%   ({} irregulars in the corpus)", irr.len());

    let w = pct(NONCE.iter().filter(|(l, t, w)| ask_s(l, t).as_deref() == Some(*w)).count(), NONCE.len());
    println!("  5 wug test        {w:>6.1}%");

    let t = Instant::now();
    for j in 0..5000 { let f = train[j * 7 % train.len()]; let _ = ask(&f.lemma, &f.tags); }
    let qps = 5000.0 / t.elapsed().as_secs_f64();
    println!("  7 speed           {qps:>6.0}    answers/sec");

    println!("\nTSV\t{n_id}\t{n_gram}\t{}\t{}\t{}\t{:.2}\t{:.1}\t{m:.1}\t{g:.1}\t{r:.1}\t{i:.1}\t{w:.1}\t{qps:.0}",
        train.len(), reachable.len(), syn, peak - base, train_t.as_secs_f64());
    Ok(())
}
