// TESTING / DESIGN ONLY. (v28 idea 10) Past tense read most-specific-first.
//
//   cargo run --release --example morph_backoff -- <split 0..9>       AVA_ENDS=<3..6> (default 3)
//
// For a verb whose past tense she never saw, her identity units are empty, so her answer is the
// SUM of every grammar unit the cue reaches: the tag units, the tag-set units and the stem-ending
// units (last 1, 2, 3 letters, each crossed with the tag set). Summed, the common "+ed" floods the
// few units that know "-end -> -ent" or "-ing -> -ung", and she says "resended", "swinged".
// (pasttense_split.rs: regular 97.0%, irregular 22.1%; plain suffix analogy gets 69.9%.)
//
// Here the SAME units are read most specific first: identity, then the longest ending, then
// shorter endings, the tag set, the bare tags. A feature's answer counts only as far as all its
// units carry it (coincidence; see `level`). Each level is trusted by its own experience,
// Witten-Bell: lambda = c / (c + d), c = the votes its units have collected, d = how many
// different answers they have seen. A level that has seen much and agreed with itself speaks
// with authority; one that has seen little, or everything, passes the word to the level below.
// No constant is set by hand. AVA_ENDS > 3 also gives her longer ending units (4..ENDS letters,
// crossed with the tag set the same way) when she learns; 3 is exactly exam.rs.
//
// split k: the withheld lemmas are every 10th in sorted order starting at k (exam.rs is 0).
// Pre-registered in "ava v28 - in depth" §9d: screen on split 0, confirm once on split 5.
use design_tests::living::Fact;
use design_tests::v4::{self, Params};
use design_tests::v6::*;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::time::Instant;

const CORPUS: &str = "../data/unimorph-eng.tsv";
const N_ID: u32 = 10_000_000;
const N_GRAM: u32 = 200_000; // AVA_NGRAM overrides (v29 WP5 screen: a bigger grammar space for longer endings)
const K_ID: usize = 10;
const PER_TAG: usize = 3;
const TRAIN: usize = 652_477;

// AVA_CONTROLS=2 (v29 §WP10): exam.rs's lists were wrong in two places, found in v28 §9e.
// UniMorph has NO past tense of "play", so it was never a taught control; "blick" IS in UniMorph
// (blicked), so it never tested invention. Set 2 swaps them for "look" (looked is in UniMorph)
// and "glorp" (absent). Set 1 is exam.rs's, kept for the v28 record.
const REGULAR: &[(&str, &[&str], &str)] = &[
    ("walk", &["V", "PST"], "walked"), ("play", &["V", "PST"], "played"),
    ("bake", &["V", "PST"], "baked"), ("jump", &["V", "PST"], "jumped"),
];
const NONCE: &[(&str, &[&str], &str)] = &[
    ("wug", &["N", "PL"], "wugs"), ("tove", &["N", "PL"], "toves"), ("gorp", &["N", "PL"], "gorps"),
    ("zib", &["N", "PL"], "zibs"), ("dax", &["N", "PL"], "daxes"), ("blick", &["V", "PST"], "blicked"),
    ("florp", &["V", "PST"], "florped"), ("nace", &["V", "PST"], "naced"),
];
const REGULAR2: &[(&str, &[&str], &str)] = &[
    ("walk", &["V", "PST"], "walked"), ("look", &["V", "PST"], "looked"),
    ("bake", &["V", "PST"], "baked"), ("jump", &["V", "PST"], "jumped"),
];
const NONCE2: &[(&str, &[&str], &str)] = &[
    ("wug", &["N", "PL"], "wugs"), ("tove", &["N", "PL"], "toves"), ("gorp", &["N", "PL"], "gorps"),
    ("zib", &["N", "PL"], "zibs"), ("dax", &["N", "PL"], "daxes"), ("glorp", &["V", "PST"], "glorped"),
    ("florp", &["V", "PST"], "florped"), ("nace", &["V", "PST"], "naced"),
];

fn tagset(tags: &[String]) -> String {
    let mut ts: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
    ts.sort();
    ts.join(",")
}

/// her grammar features, plus ending units of 4..=ends letters (none when ends = 3)
fn gram_feats(f: &Fields, ends: usize) -> Vec<String> {
    let mut v = f.grammar();
    let ch: Vec<char> = f.lemma.chars().collect();
    let ts = tagset(f.tags);
    for n in 4..=ends {
        if ch.len() >= n { v.push(format!("E{n}:{}|A:{ts}", ch[ch.len() - n..].iter().collect::<String>())); }
    }
    v
}

/// the levels of a query, most specific first: each a list of feature names (identity aside)
fn levels(lemma: &str, tags: &[String], ends: usize) -> Vec<Vec<String>> {
    let ch: Vec<char> = lemma.chars().collect();
    let ts = tagset(tags);
    let mut out = Vec::new();
    for n in (1..=ends).rev() {
        if ch.len() >= n { out.push(vec![format!("E{n}:{}|A:{ts}", ch[ch.len() - n..].iter().collect::<String>())]); }
    }
    out.push(vec![format!("A:{ts}")]);
    out.push(tags.iter().map(|t| format!("T:{t}")).collect());
    out
}

fn regular_forms(l: &str) -> Vec<String> {
    let vowel = |c: char| "aeiou".contains(c);
    let c: Vec<char> = l.chars().collect();
    let mut v = vec![format!("{l}ed")];
    if let Some(&last) = c.last() {
        if last == 'e' { v.push(format!("{l}d")); }
        if last == 'y' && c.len() >= 2 && !vowel(c[c.len() - 2]) { v.push(format!("{}ied", &l[..l.len() - 1])); }
        if !vowel(last) && last.is_alphabetic() { v.push(format!("{l}{last}ed")); }
    }
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

struct Her { fac: Factored, ro: OpReadout, ends: usize }

impl Her {
    fn cue(&self, lemma: &str, tags: &[String]) -> (Vec<NeuronId>, Vec<NeuronId>) {
        let f = Fields { lemma, form: None, tags };
        self.fac.query_feats(&f.identity(), &gram_feats(&f, self.ends), K_ID)
    }
    /// her own reading: the specificity-weighted sum over every unit (OpReadout::decode)
    fn summed(&self, lemma: &str, tags: &[String]) -> Option<String> {
        let (idc, gc) = self.cue(lemma, tags);
        let mut c = idc; c.extend_from_slice(&gc);
        self.ro.decode(&c, 1).into_iter().filter_map(|(op, _)| apply_op(lemma, &op)).next()
    }
    /// what one level has seen: votes per answer, and their total. A feature's units are
    /// hashed into a shared space, so one of them can also serve a busy feature of another word
    /// class; summed, that stranger's votes would speak for this feature (v28 §9d, first try:
    /// "tarmac" -> "tarmaier"). So an answer counts only as far as ALL the feature's units
    /// carry it (coincidence: count-min over its units, as v21 read her hashed gram units).
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
    /// most specific first, each level trusted by its experience (Witten-Bell).
    ///
    /// MEMORY BLOCKS ANALOGY (AVA_MORPH_BLOCK, default on). A form she was taught has identity
    /// units; one she never met has none and, by her design, falls through to her grammar units
    /// (v6.rs, Fields::identity). Read as one more Witten-Bell level, a fact met once is trusted
    /// only 1 / (1 + 1) = 1/2, and a strong analogy then overrides her own memory: taught
    /// "play -> played" came out "plaid" (lay -> laid), §9d. People don't do that: a stored form
    /// blocks the analogy (words and rules, Pinker & Ullman). So when her memory answers, it is her
    /// own reading that speaks; the backoff reads only where she remembers nothing.
    /// AVA_MORPH_BLOCK=0 keeps identity as a backoff level, for the record.
    /// AVA_MORPH_BLOCK=id (v29 WP5b): the memory speaks ALONE. Summed with every grammar unit, a
    /// remembered exception ("ate") is outvoted once longer endings add more grammar units (WP5a:
    /// exceptions -1.5 points); a retrieved word should block the rule, not be one vote among many.
    fn backoff(&self, lemma: &str, tags: &[String]) -> Option<String> {
        let (idc, _) = self.cue(lemma, tags);
        let block = std::env::var("AVA_MORPH_BLOCK").unwrap_or_default();
        if !idc.is_empty() && block == "id" {
            return self.ro.decode(&idc, 1).into_iter().filter_map(|(op, _)| apply_op(lemma, &op)).next();
        }
        if !idc.is_empty() && block != "0" {
            return self.summed(lemma, tags);
        }
        let mut lv: Vec<Vec<Vec<NeuronId>>> = vec![if idc.is_empty() { vec![] } else { vec![idc] }];
        for feats in levels(lemma, tags, self.ends) {
            lv.push(feats.iter().map(|f| self.fac.gram.units(f)).collect());
        }
        // from the most general level up: p = lambda * p_level + (1 - lambda) * p_below
        let mut p: HashMap<u32, f64> = HashMap::new();
        for feats in lv.iter().rev() {
            let (cnt, tot) = self.level(feats);
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

fn main() -> std::io::Result<()> {
    let split: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let ends: usize = std::env::var("AVA_ENDS").ok().and_then(|s| s.parse().ok()).unwrap_or(3).clamp(3, 6);
    let raw = fs::read_to_string(CORPUS)?;
    let all: Vec<Fact> = raw.lines().filter_map(Fact::parse).take(TRAIN).collect();
    let mut lemmas: Vec<&str> = all.iter().map(|f| f.lemma.as_str()).collect();
    lemmas.sort_unstable();
    lemmas.dedup();
    let mut held: HashSet<String> = HashSet::new();
    for (i, l) in lemmas.iter().enumerate() { if i % 10 == split % 10 { held.insert((*l).to_string()); } }
    let is_held = |f: &Fact| held.contains(&f.lemma) && f.tags.iter().any(|t| t == "PST");
    let set2 = std::env::var("AVA_CONTROLS").map_or(false, |v| v == "2");
    let (regular, nonce) = if set2 { (REGULAR2, NONCE2) } else { (REGULAR, NONCE) };
    // what each control really is in THIS split: taught (her memory), withheld, or never in the data
    for (l, t, w) in regular.iter().chain(nonce.iter()) {
        let mut ts: Vec<String> = t.iter().map(|s| s.to_string()).collect();
        ts.sort();
        let fact = all.iter().find(|f| f.lemma == *l && { let mut a = f.tags.clone(); a.sort(); a == ts });
        let status = match fact { None => "never in the data", Some(f) if is_held(f) => "WITHHELD in this split", Some(f) if f.form == *w => "taught", Some(_) => "taught, other form" };
        println!("  control {l} {} -> {w}: {status}", ts.join(","));
    }
    if std::env::var("AVA_CHECK_CONTROLS").map_or(false, |v| v == "1") { return Ok(()); }
    let train: Vec<&Fact> = all.iter().filter(|f| !is_held(f)).collect();
    let test: Vec<&Fact> = all.iter().filter(|f| is_held(f)).collect();
    println!("split {split}, endings up to {ends} letters: training on {} facts ({} held back)", train.len(), test.len());

    // exam.rs's training; with ends = 3 the features, and so every code, are exactly its own
    let t0 = Instant::now();
    let n_gram: u32 = std::env::var("AVA_NGRAM").ok().and_then(|s| s.parse().ok()).unwrap_or(N_GRAM);
    if n_gram != N_GRAM { println!("grammar space: {n_gram} units (AVA_NGRAM)"); }
    let mut fac = Factored::new(N_ID, n_gram, K_ID, PER_TAG);
    let mut ro = OpReadout::default();
    let mut brain = v4::Brain::new(fac.n_total, Params::default());
    for _ in 0..2 {
        for f in &train {
            let fl = Fields { lemma: &f.lemma, form: Some(&f.form), tags: &f.tags };
            fac.learn_feats(&fl.identity(), &gram_feats(&fl, ends));
        }
    }
    for f in &train {
        let fl = Fields { lemma: &f.lemma, form: None, tags: &f.tags };
        let (idc, gc) = fac.query_feats(&fl.identity(), &gram_feats(&fl, ends), K_ID);
        let mut full = idc; full.extend_from_slice(&gc);
        brain.learn_bulk(&full, 1);
        ro.learn(&full, &edit_op(&f.lemma, &f.form));
    }
    brain.compact();
    println!("trained in {:.1?}", t0.elapsed());
    let her = Her { fac, ro, ends };

    // withheld past tenses, by kind, both readings, paired
    let mut kinds = [[0usize; 3]; 2]; // [regular, irregular] x [count, summed, backoff]
    let mut diff: Vec<i32> = Vec::with_capacity(test.len());
    let mut changed: Vec<String> = Vec::new();
    // AVA_MORPH_DUMP=<file>: one line per withheld form (backoff reading), so two runs can be compared pairwise
    let mut dump = String::from("lemma\ttags\tgold\tkind\tbackoff\tok\n");
    for f in &test {
        let k = if regular_forms(&f.lemma).contains(&f.form) { 0 } else { 1 };
        let (s, b) = (her.summed(&f.lemma, &f.tags), her.backoff(&f.lemma, &f.tags));
        let (so, bo) = (s.as_deref() == Some(f.form.as_str()), b.as_deref() == Some(f.form.as_str()));
        kinds[k][0] += 1;
        if so { kinds[k][1] += 1; }
        if bo { kinds[k][2] += 1; }
        diff.push(bo as i32 - so as i32);
        dump.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\n", f.lemma, tagset(&f.tags), f.form, if k == 0 { "regular" } else { "irregular" }, b.as_deref().unwrap_or("-"), bo as u8));
        if so != bo && changed.len() < 30 {
            changed.push(format!("{} {} -> {}: summed {} | backoff {}", f.lemma, tagset(&f.tags), f.form, s.as_deref().unwrap_or("-"), b.as_deref().unwrap_or("-")));
        }
    }
    if let Ok(p) = std::env::var("AVA_MORPH_DUMP") { fs::write(&p, &dump)?; println!("per-form dump: {p}"); }
    let (sum_ok, bo_ok) = (kinds[0][1] + kinds[1][1], kinds[0][2] + kinds[1][2]);
    let n = test.len();
    // paired bootstrap over withheld forms: 20,000 resamples, splitmix64 seed 1
    let mut x: u64 = 1;
    let mut boots: Vec<i64> = (0..20_000).map(|_| (0..n).map(|_| diff[(splitmix(&mut x) % n as u64) as usize] as i64).sum()).collect();
    boots.sort_unstable();
    let (lo, hi) = (boots[500], boots[19_499]);

    // memory, in-data irregulars, controls: both readings
    let sample: Vec<&&Fact> = train.iter().step_by(train.len() / 3000).take(3000).collect();
    let mem = |g: &dyn Fn(&str, &[String]) -> Option<String>| sample.iter().filter(|f| g(&f.lemma, &f.tags).as_deref() == Some(f.form.as_str())).count();
    let mut modal: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for f in &train { *modal.entry(tagset(&f.tags)).or_default().entry(edit_op(&f.lemma, &f.form)).or_insert(0) += 1; }
    let common: HashMap<String, String> = modal.iter()
        .filter_map(|(k, m)| m.iter().max_by(|x, y| x.1.cmp(y.1).then(y.0.cmp(x.0))).map(|(op, _)| (k.clone(), op.clone()))).collect();
    let irr: Vec<&&Fact> = train.iter().filter(|f| common.get(&tagset(&f.tags)) != Some(&edit_op(&f.lemma, &f.form))).collect();
    let irr_sample: Vec<&&Fact> = irr.iter().step_by((irr.len() / 800).max(1)).take(800).copied().collect();
    let irq = |g: &dyn Fn(&str, &[String]) -> Option<String>| irr_sample.iter().filter(|f| g(&f.lemma, &f.tags).as_deref() == Some(f.form.as_str())).count();
    let ctl = |set: &[(&str, &[&str], &str)], g: &dyn Fn(&str, &[String]) -> Option<String>| set.iter()
        .filter(|(l, t, w)| g(l, &t.iter().map(|s| s.to_string()).collect::<Vec<_>>()).as_deref() == Some(*w)).count();
    let sm = |l: &str, t: &[String]| her.summed(l, t);
    let bk = |l: &str, t: &[String]| her.backoff(l, t);

    println!("\n{:<34} {:>12} {:>12}", "", "summed (hers)", "backoff");
    println!("{:<34} {:>11.2}% {:>11.2}%", format!("withheld past tense ({n})"), pct(sum_ok, n), pct(bo_ok, n));
    println!("{:<34} {:>11.2}% {:>11.2}%", format!("  regular ({})", kinds[0][0]), pct(kinds[0][1], kinds[0][0]), pct(kinds[0][2], kinds[0][0]));
    println!("{:<34} {:>11.2}% {:>11.2}%", format!("  irregular ({})", kinds[1][0]), pct(kinds[1][1], kinds[1][0]), pct(kinds[1][2], kinds[1][0]));
    println!("  change: {:+} forms, 95% CI {lo:+} .. {hi:+} (paired bootstrap, 20,000, seed 1)", bo_ok as i64 - sum_ok as i64);
    println!("{:<34} {:>11.2}% {:>11.2}%", format!("memory ({} taught)", sample.len()), pct(mem(&sm), sample.len()), pct(mem(&bk), sample.len()));
    println!("{:<34} {:>11.2}% {:>11.2}%", format!("irregular exceptions ({})", irr_sample.len()), pct(irq(&sm), irr_sample.len()), pct(irq(&bk), irr_sample.len()));
    println!("{:<34} {:>12} {:>12}", "wug test", format!("{}/8", ctl(nonce, &sm)), format!("{}/8", ctl(nonce, &bk)));
    println!("{:<34} {:>12} {:>12}", "regular controls", format!("{}/4", ctl(regular, &sm)), format!("{}/4", ctl(regular, &bk)));
    for (l, t, w) in regular.iter().chain(nonce.iter()) {
        let t: Vec<String> = t.iter().map(|s| s.to_string()).collect();
        let (s, b) = (sm(l, &t), bk(l, &t));
        if s.as_deref() != Some(*w) || b.as_deref() != Some(*w) {
            println!("  control {l} {} -> {w}: summed {} | backoff {}", tagset(&t), s.as_deref().unwrap_or("-"), b.as_deref().unwrap_or("-"));
        }
    }
    println!("\nfirst 30 withheld forms the two readings disagree on:");
    for c in &changed { println!("  {c}"); }
    Ok(())
}
