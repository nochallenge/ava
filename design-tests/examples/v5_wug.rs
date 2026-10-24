// The wug test. A child who has learned English can pluralise a word they have never
// heard: "this is a wug, now there are two ___" -> "wugs". That is the difference
// between learning a RULE and memorising instances.
//
// Hold out facts entirely, then ask for them. If accuracy on held-out facts is at
// chance, the network is a lookup table with extra steps, however good it looks on
// facts it was trained on.
use design_tests::living::Fact;
use design_tests::v4::{Brain, Params, overlap};
use design_tests::v5::*;
use std::collections::{HashMap, HashSet};
use std::fs;

const PATH: &str = "../data/unimorph-eng.tsv";
const N: u32 = 200_000;
const K: usize = 12;

fn query_features(f: &Fact) -> Vec<String> {
    let mut v = vec![format!("L:{}", f.lemma)];
    for t in &f.tags { v.push(format!("T:{t}")); }
    for t in &f.tags { v.push(format!("L:{}|T:{}", f.lemma, t)); }
    v
}
fn full_features(f: &Fact) -> Vec<String> {
    let mut v = query_features(f);
    v.push(format!("F:{}", f.form));
    v
}

/// hashed codes, for comparison
fn hashed(f: &Fact) -> Vec<NeuronId> {
    let mut s: HashMap<u32, f32> = HashMap::new();
    for feat in full_features(f) {
        for i in 0..8u64 { *s.entry((fnv1a(feat.as_bytes(), i) % N as u64) as u32).or_insert(0.0) += 1.0; }
    }
    let mut v: Vec<(u32, f32)> = s.into_iter().collect();
    v.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(K);
    let mut o: Vec<u32> = v.into_iter().map(|(u, _)| u).collect();
    o.sort_unstable(); o
}

fn main() {
    let raw = fs::read_to_string(PATH).expect("run from design-tests/");
    let all: Vec<Fact> = raw.lines().filter_map(Fact::parse).take(80_000).collect();

    // hold out every V;PST fact for 1 lemma in 10, entirely
    let lemmas: HashSet<&str> = all.iter().map(|f| f.lemma.as_str()).collect();
    let mut heldout_lemmas: HashSet<String> = HashSet::new();
    for (i, l) in lemmas.iter().enumerate() {
        if i % 10 == 0 { heldout_lemmas.insert((*l).to_string()); }
    }
    let is_held = |f: &Fact| heldout_lemmas.contains(&f.lemma) && f.tags.iter().any(|t| t == "PST");

    let train: Vec<&Fact> = all.iter().filter(|f| !is_held(f)).collect();
    let test: Vec<&Fact> = all.iter().filter(|f| is_held(f)).collect();
    println!("train {} facts, held out {} V;PST facts across {} lemmas",
        train.len(), test.len(), heldout_lemmas.len());
    println!("the held-out lemmas ARE seen in other tenses. only their PAST TENSE is hidden.\n");

    for mode in ["hashed", "earned"] {
        let mut alloc = Allocator::new(N, K);
        let codes: Vec<Vec<NeuronId>> = if mode == "earned" {
            let fs_: Vec<Vec<String>> = train.iter().map(|f| full_features(f)).collect();
            for f in &fs_ { alloc.allocate_learn(f); }
            fs_.iter().map(|f| alloc.allocate_learn(f)).collect()
        } else {
            train.iter().map(|f| hashed(f)).collect()
        };

        let mut brain = Brain::new(N, Params::default());
        let mut dec = Decoder::new(N);
        for (i, c) in codes.iter().enumerate() {
            brain.learn(c, 1);
            dec.learn(c, &train[i].form);
        }

        // accuracy on facts it WAS taught
        let seen: Vec<usize> = (0..1000).map(|i| i * 71 % train.len()).collect();
        let mut ok_seen = 0;
        for &i in &seen {
            let q = query_features(train[i]);
            let cue = if mode == "earned" { alloc.cue(&q, 7) } else {
                let mut c = hashed_query(&q); c.truncate(7); c.sort_unstable(); c };
            if cue.is_empty() { continue; }
            let fired: Vec<NeuronId> = brain.recall(&cue, 1).into_iter().map(|(_, u)| u).collect();
            if dec.decode(&fired, 1).first().map(|(w, _)| w == &train[i].form).unwrap_or(false) { ok_seen += 1; }
        }

        // accuracy on facts it was NEVER taught
        let (mut ok_held, mut top3_held, mut silent) = (0usize, 0usize, 0usize);
        for f in &test {
            let q = query_features(f);
            let cue = if mode == "earned" { alloc.cue(&q, 7) } else {
                let mut c = hashed_query(&q); c.truncate(7); c.sort_unstable(); c };
            if cue.is_empty() { silent += 1; continue; }
            let fired: Vec<NeuronId> = brain.recall(&cue, 1).into_iter().map(|(_, u)| u).collect();
            let ans = dec.decode(&fired, 3);
            if ans.is_empty() { silent += 1; continue; }
            if ans[0].0 == f.form { ok_held += 1; }
            if ans.iter().any(|(w, _)| w == &f.form) { top3_held += 1; }
        }

        println!("=== {mode} ===");
        println!("  on facts it WAS taught      : {:.1}%", ok_seen as f64 / seen.len() as f64 * 100.0);
        println!("  on facts it was NEVER taught: {:.1}% top-1, {:.1}% top-3, {silent} silent",
            ok_held as f64 / test.len() as f64 * 100.0, top3_held as f64 / test.len() as f64 * 100.0);

        // what does it actually say for a held-out word?
        print!("  examples: ");
        for f in test.iter().take(4) {
            let q = query_features(f);
            let cue = if mode == "earned" { alloc.cue(&q, 7) } else {
                let mut c = hashed_query(&q); c.truncate(7); c.sort_unstable(); c };
            let fired: Vec<NeuronId> = brain.recall(&cue, 1).into_iter().map(|(_, u)| u).collect();
            let ans = dec.decode(&fired, 1);
            print!("{}+PST -> {}   ", f.lemma, ans.first().map(|(w, _)| w.clone()).unwrap_or("(silence)".into()));
        }
        println!("\n     (wanted: {})\n", test.iter().take(4).map(|f| f.form.clone()).collect::<Vec<_>>().join(", "));
        let _ = overlap(&[], &[]);
    }
}

fn hashed_query(q: &[String]) -> Vec<NeuronId> {
    let mut s: HashMap<u32, f32> = HashMap::new();
    for feat in q {
        for i in 0..8u64 { *s.entry((fnv1a(feat.as_bytes(), i) % N as u64) as u32).or_insert(0.0) += 1.0; }
    }
    let mut v: Vec<(u32, f32)> = s.into_iter().collect();
    v.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v.into_iter().map(|(u, _)| u).collect()
}
