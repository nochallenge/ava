// ava's report card. Seven separate skills, scored independently, so progress is
// visible as a profile rather than one number that hides what is broken.
//
//   cargo run --release --example exam
//
// Each run appends a line to ../brain/exam-history.tsv so scores can be compared
// across versions.
use design_tests::living::Fact;
use design_tests::v4::{self, Params};
use design_tests::v6::*;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::env;
use std::fs;
use std::io::Write as _;
use std::time::Instant;

const CORPUS: &str = "../data/unimorph-eng.tsv";
const HISTORY: &str = "../brain/exam-history.tsv";
const N_ID: u32 = 10_000_000;
/// v29 WP5b: 10x the grammar space (was 200_000), so endings of 4 and 5 letters don't collide
const N_GRAM: u32 = 2_000_000;
/// v29 WP5b: stem endings up to 5 letters (was 3), crossed with the tag set like the others
const ENDS: usize = 5;
const K_ID: usize = 10;
const PER_TAG: usize = 3;
const TRAIN: usize = 652_477;

fn tagset(tags: &[String]) -> String {
    let mut ts: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
    ts.sort();
    ts.join(",")
}

/// her grammar features (v6 Fields::grammar: tags, tag set, endings of 1-3 letters) plus the
/// 4- and 5-letter endings; with ENDS = 3 this is exactly the old code
fn gram_feats(f: &Fields, ends: usize) -> Vec<String> {
    let mut v = f.grammar();
    let ch: Vec<char> = f.lemma.chars().collect();
    let ts = tagset(f.tags);
    for n in 4..=ends {
        if ch.len() >= n { v.push(format!("E{n}:{}|A:{ts}", ch[ch.len() - n..].iter().collect::<String>())); }
    }
    v
}

/// perfectly regular controls, all verified present in UniMorph.
/// v29: "play" was NOT (UniMorph has no past tense of play), so it was never a taught control;
/// "look" replaces it. Checked per split by morph_backoff's AVA_CHECK_CONTROLS.
const REGULAR: &[(&str, &[&str], &str)] = &[
    ("walk", &["V", "PST"], "walked"),
    ("look", &["V", "PST"], "looked"),
    ("bake", &["V", "PST"], "baked"),
    ("jump", &["V", "PST"], "jumped"),
];

/// nonce words. what a real english speaker would say.
const NONCE: &[(&str, &[&str], &str)] = &[
    ("wug", &["N", "PL"], "wugs"),
    ("tove", &["N", "PL"], "toves"),
    ("gorp", &["N", "PL"], "gorps"),
    ("zib", &["N", "PL"], "zibs"),
    ("dax", &["N", "PL"], "daxes"),
    // v29: "blick" is IN UniMorph (blicked), so it tested memory, not invention; "glorp" is absent
    ("glorp", &["V", "PST"], "glorped"),
    ("florp", &["V", "PST"], "florped"),
    ("nace", &["V", "PST"], "naced"),
];

/// Irregulars are DERIVED FROM THE DATA, not from my memory of english.
/// The first version of this exam hand-listed child->children, foot->feet,
/// woman->women, tooth->teeth. UniMorph contains none of those plurals, so the
/// test scored 0% on facts the corpus never taught and I nearly reported an
/// architecture failure that was a test bug. An irregular here is a fact whose
/// edit operation is NOT the most common operation for its tag-set.
fn derive_irregulars(train: &[&Fact]) -> Vec<(String, Vec<String>, String)> {
    let mut modal: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for f in train {
        let mut ts: Vec<&str> = f.tags.iter().map(|s| s.as_str()).collect();
        ts.sort();
        *modal.entry(ts.join(",")).or_default()
            .entry(edit_op(&f.lemma, &f.form)).or_insert(0) += 1;
    }
    let common: HashMap<String, String> = modal.iter()
        // ties broken by the op string: `max_by_key` over a HashMap broke them by the
        // random iteration order, so which facts counted as irregular varied run to run.
        .filter_map(|(k, m)| m.iter()
            .max_by(|x, y| x.1.cmp(y.1).then(y.0.cmp(x.0)))
            .map(|(op, _)| (k.clone(), op.clone())))
        .collect();
    let mut out = Vec::new();
    for f in train {
        let mut ts: Vec<&str> = f.tags.iter().map(|s| s.as_str()).collect();
        ts.sort();
        let key = ts.join(",");
        if common.get(&key) != Some(&edit_op(&f.lemma, &f.form)) {
            out.push((f.lemma.clone(), f.tags.clone(), f.form.clone()));
        }
    }
    out
}

struct Ava {
    fac: Factored,
    ro: OpReadout,
    /// same-moment synapses: written, never read by any skill here (v26 found the same in compose);
    /// off by default since v29, AVA_SYNAPSES=1 writes them as before
    _brain: Option<v4::Brain>,
}

impl Ava {
    fn cue(&self, lemma: &str, tags: &[String]) -> (Vec<NeuronId>, Vec<NeuronId>) {
        let f = Fields { lemma, form: None, tags };
        self.fac.query_feats(&f.identity(), &gram_feats(&f, ENDS), K_ID)
    }
    fn ask(&self, lemma: &str, tags: &[String], top: usize) -> Vec<(String, f32)> {
        let (idc, gc) = self.cue(lemma, tags);
        let mut cue = idc;
        cue.extend_from_slice(&gc);
        self.ro.decode(&cue, top)
            .into_iter()
            .filter_map(|(op, c)| apply_op(lemma, &op).map(|w| (w, c)))
            .collect()
    }
    fn top1(&self, lemma: &str, tags: &[&str]) -> Option<String> {
        let t: Vec<String> = tags.iter().map(|s| s.to_string()).collect();
        self.answer(lemma, &t)
    }
    /// v29: HER ANSWER. Where she remembers the form (identity units), her own summed reading,
    /// as before. Where she remembers nothing, her grammar units read MOST SPECIFIC FIRST:
    /// longest stem ending, shorter endings, tag set, bare tags; a feature's answer counts only as
    /// far as all its hashed units carry it (coincidence, count-min), and each level is trusted by
    /// its experience (Witten-Bell, c / (c + d)). Summed, the common "+ed" drowned the few units
    /// that know "-end -> -ent" ("resended", "swinged"). Confirmed on a fresh split (vault
    /// "ava v29 - in depth" WP10a): withheld past tense 92.31 -> 93.80, irregular 11.8 -> 44.6,
    /// regular -0.44, memory unchanged. `ask` (the summed reading with its scores) stays for the
    /// calibration and speed lines.
    ///
    /// v29 WP5b: endings up to 5 letters in a 2M grammar space, and where she remembers the form
    /// THE MEMORY SPEAKS ALONE (her identity units only; words and rules). Summed with every
    /// grammar unit, a remembered "ate" was outvoted once longer endings added units. Confirmed on
    /// fresh split 1: withheld past tense 93.18 -> 95.70 (+122, CI +94..+150), irregular
    /// 48.0 -> 79.3, regular +0.6, memory 99.93 -> 99.97, exceptions 99.75 -> 99.88.
    fn answer(&self, lemma: &str, tags: &[String]) -> Option<String> {
        let (idc, _) = self.cue(lemma, tags);
        if !idc.is_empty() {
            return self.ro.decode(&idc, 1).into_iter().filter_map(|(op, _)| apply_op(lemma, &op)).next();
        }
        let ts = tagset(tags);
        let ch: Vec<char> = lemma.chars().collect();
        let mut levels: Vec<Vec<String>> = (1..=ENDS).rev().filter(|n| ch.len() >= *n)
            .map(|n| vec![format!("E{n}:{}|A:{ts}", ch[ch.len() - n..].iter().collect::<String>())]).collect();
        levels.push(vec![format!("A:{ts}")]);
        levels.push(tags.iter().map(|t| format!("T:{t}")).collect());
        let mut p: HashMap<u32, f64> = HashMap::new();
        for feats in levels.iter().rev() {
            let (mut cnt, mut tot) = (HashMap::<u32, f64>::new(), 0.0);
            for feat in feats {
                let units = self.fac.gram.units(feat);
                let Some(first) = units.first().and_then(|u| self.ro.votes.get(u)) else { continue };
                for &o in first.keys() {
                    let m = units.iter().map(|u| self.ro.votes.get(u).and_then(|v| v.get(&o)).copied().unwrap_or(0)).min().unwrap_or(0);
                    if m > 0 { *cnt.entry(o).or_insert(0.0) += m as f64; tot += m as f64; }
                }
            }
            if tot <= 0.0 { continue; }
            let lam = tot / (tot + cnt.len() as f64);
            for v in p.values_mut() { *v *= 1.0 - lam; }
            for (o, c) in cnt { *p.entry(o).or_insert(0.0) += lam * c / tot; }
        }
        let mut v: Vec<(u32, f64)> = p.into_iter().collect();
        v.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v.first().and_then(|(o, _)| apply_op(lemma, &self.ro.ops[*o as usize]))
    }
}

fn pct(a: usize, b: usize) -> f64 { if b == 0 { 0.0 } else { a as f64 / b as f64 * 100.0 } }

fn bar(p: f64) -> String {
    let n = (p / 5.0).round() as usize;
    format!("{}{}", "#".repeat(n.min(20)), ".".repeat(20 - n.min(20)))
}

fn main() -> std::io::Result<()> {
    let raw = fs::read_to_string(CORPUS)?;
    let all: Vec<Fact> = raw.lines().filter_map(Fact::parse).take(TRAIN).collect();

    // hold out 1 lemma in 10, past tense only, for the generalization test
    // SORTED, not a HashSet. Rust seeds HashSet iteration order randomly per process, so
    // "every 10th lemma" was a different tenth on every run and the generalization score
    // was never reproducible — 92.19% was one draw. Found 2026-10-12, before publishing it.
    let mut lemmas: Vec<&str> = all.iter().map(|f| f.lemma.as_str()).collect();
    lemmas.sort_unstable();
    lemmas.dedup();
    let mut held: HashSet<String> = HashSet::new();
    for (i, l) in lemmas.iter().enumerate() { if i % 10 == 0 { held.insert((*l).to_string()); } }
    let is_held = |f: &Fact| held.contains(&f.lemma) && f.tags.iter().any(|t| t == "PST");
    let train: Vec<&Fact> = all.iter().filter(|f| !is_held(f)).collect();
    let test: Vec<&Fact> = all.iter().filter(|f| is_held(f)).collect();

    println!("training on {} facts ({} held back)\n", train.len(), test.len());
    let t0 = Instant::now();
    let mut fac = Factored::new(N_ID, N_GRAM, K_ID, PER_TAG);
    let mut ro = OpReadout::default();
    let mut brain = (env::var("AVA_SYNAPSES").as_deref() == Ok("1")).then(|| v4::Brain::new(fac.n_total, Params::default()));
    // PASS 1 and 2: let the competition run and the weights settle.
    for _ in 0..2 {
        for f in &train {
            let fl = Fields { lemma: &f.lemma, form: Some(&f.form), tags: &f.tags };
            fac.learn_feats(&fl.identity(), &gram_feats(&fl, ENDS));
        }
    }
    let t_alloc = t0.elapsed();
    let (mut t_syn, mut t_read) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
    // PASS 3: allocator FROZEN. Encode with the exact function a query will use, and
    // build the memory from those codes.
    //
    // This pass is not an optimisation, it is a correctness fix. Competitive learning
    // is NON-STATIONARY: a code allocated in pass 2 was produced by weights that later
    // facts kept changing, so by the end of training the same query produced a
    // DIFFERENT code and every early memory was unreachable. Symptom: she answered
    // "walks" for walk+PST and "eats" for eat+PRS.PTCP -- everything collapsed onto
    // the single most common operation. Freeze first, then write.
    // v29 WP5d: the counting is split across cores (AVA_THREADS, default 4). Counts add up in any
    // order, so each core counts its share of the facts and the shares are summed: the same
    // numbers as one core. Op ids are handed out first, in fact order, so even their numbering
    // is unchanged. With synapses on (AVA_SYNAPSES=1) or AVA_THREADS=1 it runs on one core.
    let threads: usize = env::var("AVA_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(4).max(1);
    if brain.is_none() && threads > 1 {
        let tr = Instant::now();
        let ops: Vec<u32> = train.iter().map(|f| ro.op(&edit_op(&f.lemma, &f.form))).collect();
        let chunk = train.len().div_ceil(threads);
        let fac_ref = &fac;
        let shares: Vec<(HashMap<NeuronId, HashMap<u32, i32>>, HashMap<NeuronId, u32>)> = std::thread::scope(|s| {
            let hs: Vec<_> = train.chunks(chunk).zip(ops.chunks(chunk)).map(|(fs, os)| s.spawn(move || {
                let (mut votes, mut total) = (HashMap::<NeuronId, HashMap<u32, i32>>::new(), HashMap::<NeuronId, u32>::new());
                for (f, &o) in fs.iter().zip(os) {
                    let fl = Fields { lemma: &f.lemma, form: None, tags: &f.tags };
                    let (idc, gc) = fac_ref.query_feats(&fl.identity(), &gram_feats(&fl, ENDS), K_ID);
                    for &u in idc.iter().chain(gc.iter()) {
                        *votes.entry(u).or_default().entry(o).or_insert(0) += 1;
                        *total.entry(u).or_insert(0) += 1;
                    }
                }
                (votes, total)
            })).collect();
            hs.into_iter().map(|h| h.join().expect("a counting thread failed")).collect()
        });
        for (votes, total) in shares {
            for (u, m) in votes { let e = ro.votes.entry(u).or_default(); for (o, c) in m { *e.entry(o).or_insert(0) += c; } }
            for (u, t) in total { *ro.total.entry(u).or_insert(0) += t; }
        }
        ro.n_learned += train.len() as u32;
        t_read = tr.elapsed();
        println!("  (pass 3 counted on {threads} cores)");
    } else {
        for f in &train {
            let fl = Fields { lemma: &f.lemma, form: None, tags: &f.tags };
            let (idc, gc) = fac.query_feats(&fl.identity(), &gram_feats(&fl, ENDS), K_ID);
            let mut full = idc; full.extend_from_slice(&gc);
            let ts = Instant::now();
            if let Some(b) = brain.as_mut() { b.learn_bulk(&full, 1); }
            t_syn += ts.elapsed();
            let tr = Instant::now();
            ro.learn(&full, &edit_op(&f.lemma, &f.form));
            t_read += tr.elapsed();
        }
    }
    if let Some(b) = brain.as_mut() { b.compact(); }
    let train_time = t0.elapsed();
    println!("  (training time: allocation passes {:.1?}, synapse writes {:.1?} + compaction, readout counting {:.1?})", t_alloc, t_syn, t_read);
    let ava = Ava { fac, ro, _brain: brain };

    let mut scores: Vec<(&str, f64, String)> = Vec::new();
    println!("{:<26} {:>7}  {}", "SKILL", "SCORE", "");

    // 1. memory
    let sample: Vec<&&Fact> = train.iter().step_by(train.len() / 3000).take(3000).collect();
    let ok = sample.iter().filter(|f| {
        let t: Vec<String> = f.tags.clone();
        ava.answer(&f.lemma, &t).map(|w| w == f.form).unwrap_or(false)
    }).count();
    let p = pct(ok, sample.len());
    println!("{:<26} {:>6.1}%  {}  facts she was taught", "1 memory", p, bar(p));
    scores.push(("memory", p, format!("{ok}/{}", sample.len())));

    // 2. generalization to held-out real words
    let ok = test.iter().filter(|f| {
        ava.answer(&f.lemma, &f.tags).map(|w| w == f.form).unwrap_or(false)
    }).count();
    let p = pct(ok, test.len());
    println!("{:<26} {:>6.1}%  {}  real words, past tense withheld", "2 generalization", p, bar(p));
    scores.push(("generalization", p, format!("{ok}/{}", test.len())));

    // 3. regular inflection
    let ok = REGULAR.iter().filter(|(l, t, want)| ava.top1(l, t).as_deref() == Some(*want)).count();
    let p = pct(ok, REGULAR.len());
    println!("{:<26} {:>6.1}%  {}  walk->walked, cat->cats", "3 regular rules", p, bar(p));
    scores.push(("regular", p, format!("{ok}/{}", REGULAR.len())));

    // 4. irregulars — the over-regularization test, on irregulars found IN the data
    let irr = derive_irregulars(&train);
    let irr_sample: Vec<&(String, Vec<String>, String)> =
        irr.iter().step_by((irr.len() / 800).max(1)).take(800).collect();
    let ok = irr_sample.iter().filter(|(l, t, want)| {
        ava.answer(l, t).map(|w| &w == want).unwrap_or(false)
    }).count();
    let p = pct(ok, irr_sample.len());
    println!("{:<26} {:>6.1}%  {}  {} found in the corpus, e.g. eat->ate",
        "4 irregular exceptions", p, bar(p), irr.len());
    scores.push(("irregular", p, format!("{ok}/{}", irr_sample.len())));

    // 5. nonce words
    let ok = NONCE.iter().filter(|(l, t, want)| ava.top1(l, t).as_deref() == Some(*want)).count();
    let p = pct(ok, NONCE.len());
    println!("{:<26} {:>6.1}%  {}  wug->wugs, words that do not exist", "5 wug test", p, bar(p));
    scores.push(("wug", p, format!("{ok}/{}", NONCE.len())));

    // 6. does she know when she does not know?
    let known_conf: Vec<f32> = sample.iter().take(400)
        .filter_map(|f| ava.ask(&f.lemma, &f.tags, 1).first().map(|(_, c)| *c)).collect();
    let junk: Vec<f32> = ["qxzt", "vbnm", "zzzq", "xkcd", "pfft"].iter()
        .filter_map(|w| ava.ask(w, &["N".into(), "PL".into()], 1).first().map(|(_, c)| *c)).collect();
    let mk = known_conf.iter().sum::<f32>() / known_conf.len().max(1) as f32;
    let jk = junk.iter().sum::<f32>() / junk.len().max(1) as f32;
    let calib = if mk > jk { (1.0 - jk / mk) * 100.0 } else { 0.0 } as f64;
    println!("{:<26} {:>6.1}%  {}  confident on known ({mk:.1}) vs junk ({jk:.1})",
        "6 knows what it knows", calib, bar(calib));
    scores.push(("calibration", calib, format!("{mk:.1} vs {jk:.1}")));

    // 7. speed
    let t = Instant::now();
    let n = 5000;
    for i in 0..n {
        let f = train[i * 7 % train.len()];
        let _ = ava.ask(&f.lemma, &f.tags, 1);
    }
    let qps = n as f64 / t.elapsed().as_secs_f64();
    let p = (qps / 200.0).min(100.0);
    println!("{:<26} {:>6.0}   {}  answers per second", "7 speed", qps, bar(p));
    scores.push(("speed_qps", qps, format!("{qps:.0}")));

    let overall: f64 = scores[..6].iter().map(|s| s.1).sum::<f64>() / 6.0;
    println!("\n{:<26} {:>6.1}%  {}", "OVERALL", overall, bar(overall));
    println!("(trained in {train_time:?})");

    // what is she getting wrong?
    println!("\n--- where she fails ---");
    for (l, t, want) in REGULAR.iter().chain(NONCE.iter()) {
        if let Some(got) = ava.top1(l, t) {
            if &got != want {
                println!("  {:>8} + {:<10} said {:<12} wanted {}", l, t.join(";"), got, want);
            }
        }
    }
    let mut shown = 0;
    for (l, t, want) in irr_sample.iter() {
        if shown >= 6 { break; }
        if let Some(got) = ava.answer(l, t) {
            if &got != want {
                println!("  {:>8} + {:<10} said {:<12} wanted {}  (irregular)",
                    l, t.join(";"), got, want);
                shown += 1;
            }
        }
    }

    // history
    let mut line = String::new();
    write!(line, "v29b").unwrap();
    for (_, v, _) in &scores { write!(line, "\t{v:.2}").unwrap(); }
    writeln!(line, "\t{overall:.2}").unwrap();
    fs::create_dir_all("../brain").ok();
    let exists = std::path::Path::new(HISTORY).exists();
    let mut fh = fs::OpenOptions::new().create(true).append(true).open(HISTORY)?;
    if !exists {
        writeln!(fh, "version\tmemory\tgeneralization\tregular\tirregular\twug\tcalibration\tspeed\toverall")?;
    }
    fh.write_all(line.as_bytes())?;
    println!("\nappended to {HISTORY}");
    let _: HashMap<u8, u8> = HashMap::new();
    Ok(())
}
