// Two questions:
//   1. do EARNED codes beat HASHED codes?
//   2. can she answer a question? "what is the past tense of eat?"
//
// The question is posed by giving the network everything about the fact EXCEPT the
// answer: lemma + tags, but not the surface form. If it completes to the right
// pattern and the readout says "ate", that is a question answered.
use design_tests::living::Fact;
use design_tests::v4::{Brain, Params, overlap};
use design_tests::v5::*;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Instant;

const PATH: &str = "../data/unimorph-eng.tsv";

/// features WITHOUT the surface form — this is the question
fn query_features(f: &Fact) -> Vec<String> {
    let mut v = vec![format!("L:{}", f.lemma)];
    for t in &f.tags { v.push(format!("T:{t}")); }
    for t in &f.tags { v.push(format!("L:{}|T:{}", f.lemma, t)); }
    v
}

/// features WITH the form — this is the full fact
fn full_features(f: &Fact) -> Vec<String> {
    let mut v = query_features(f);
    v.push(format!("F:{}", f.form));
    v
}

fn hashed_code(f: &Fact, n: u32, k: usize) -> Vec<NeuronId> {
    let mut score: HashMap<u32, f32> = HashMap::new();
    for feat in full_features(f) {
        for i in 0..8u64 {
            *score.entry((fnv1a(feat.as_bytes(), i) % n as u64) as u32).or_insert(0.0) += 1.0;
        }
    }
    let mut v: Vec<(u32, f32)> = score.into_iter().collect();
    v.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(k);
    let mut o: Vec<u32> = v.into_iter().map(|(u, _)| u).collect();
    o.sort_unstable();
    o
}

fn spread(codes: &[Vec<NeuronId>], n: u32) -> (f64, usize, u32) {
    let mut use_count: HashMap<NeuronId, u32> = HashMap::new();
    for c in codes { for &u in c { *use_count.entry(u).or_insert(0) += 1; } }
    let used = use_count.len();
    let max = use_count.values().copied().max().unwrap_or(0);
    let mean = codes.len() as f64 * codes[0].len() as f64 / used.max(1) as f64;
    let _ = n;
    (mean, used, max)
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let limit: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(60_000);
    let n: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let k: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(12);

    let raw = fs::read_to_string(PATH).expect("run from design-tests/");
    let facts: Vec<Fact> = raw.lines().filter_map(Fact::parse).take(limit).collect();
    println!("{} facts, N={n}, k={k}\n", facts.len());

    // ---------------- earned allocation ----------------
    let t = Instant::now();
    let mut alloc = Allocator::new(n, k);
    let feats: Vec<Vec<String>> = facts.iter().map(full_features).collect();
    // two passes: the first lets the competition settle, the second is the code we keep
    for f in &feats { alloc.allocate_learn(f); }
    let earned: Vec<Vec<NeuronId>> = feats.iter().map(|f| alloc.allocate_learn(f)).collect();
    println!("earned allocation in {:?} ({} features interned)", t.elapsed(), alloc.feat_id.len());

    let hashed: Vec<Vec<NeuronId>> = facts.iter().map(|f| hashed_code(f, n, k)).collect();

    println!("\n=== earned vs hashed ===");
    for (label, codes) in [("hashed", &hashed), ("earned", &earned)] {
        let (mean, used, max) = spread(codes, n);
        // pairwise overlap on a sample
        let s: Vec<usize> = (0..1500).map(|i| i * 37 % codes.len()).collect();
        let (mut mx, mut tot, mut cnt) = (0usize, 0u64, 0u64);
        for i in 0..s.len() { for j in i+1..s.len().min(i+40) {
            let o = overlap(&codes[s[i]], &codes[s[j]]); mx = mx.max(o); tot += o as u64; cnt += 1;
        }}
        println!("  {label}: units used {used}/{n}, facts/unit mean {mean:.1}, busiest unit holds {max}");
        println!("           pairwise overlap mean {:.3}, max {mx}", tot as f64 / cnt as f64);
    }

    // does earned allocation capture STRUCTURE? facts sharing a tag should be closer.
    println!("\n=== do codes reflect grammatical category, or only string identity? ===");
    let mut by_tag: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, f) in facts.iter().enumerate() {
        for t in &f.tags { by_tag.entry(t.clone()).or_default().push(i); }
    }
    println!("{:>10} {:>16} {:>16}", "tag", "hashed within", "earned within");
    for tag in ["PST", "PL", "V.PTCP"] {
        if let Some(idx) = by_tag.get(tag) {
            let sample: Vec<usize> = idx.iter().copied().step_by((idx.len() / 300).max(1)).take(300).collect();
            let mut res = [0f64; 2];
            for (ci, codes) in [&hashed, &earned].iter().enumerate() {
                let (mut tot, mut cnt) = (0u64, 0u64);
                for i in 0..sample.len() { for j in i+1..sample.len().min(i+30) {
                    // only compare facts with DIFFERENT lemmas, so shared lemma cannot explain it
                    if facts[sample[i]].lemma == facts[sample[j]].lemma { continue; }
                    tot += overlap(&codes[sample[i]], &codes[sample[j]]) as u64; cnt += 1;
                }}
                res[ci] = tot as f64 / cnt.max(1) as f64;
            }
            println!("{:>10} {:>16.4} {:>16.4}", tag, res[0], res[1]);
        }
    }
    println!("  (different lemmas only, so any difference is the TAG being learned)");

    // ---------------- store, with a readout ----------------
    println!("\n=== teaching ava, with a word-neuron readout ===");
    let t = Instant::now();
    let mut brain = Brain::new(n, Params::default());
    let mut dec = Decoder::new(n);
    for (i, code) in earned.iter().enumerate() {
        brain.learn(code, 1);
        dec.learn(code, &facts[i].form);
    }
    println!("stored {} facts in {:?}, {} synapses, {} word neurons",
        facts.len(), t.elapsed(), brain.total_synapses(), dec.words.len());

    // ---------------- ASK HER ----------------
    println!("\n=== asking questions: lemma + tags given, FORM withheld ===");
    let probes = ["eat", "go", "child", "run", "be", "mouse", "think", "swim"];
    for lemma in probes {
        for (i, f) in facts.iter().enumerate() {
            if f.lemma != lemma { continue; }
            let q = query_features(f);
            let cue = alloc.cue(&q, 7);
            if cue.is_empty() { continue; }
            let fired: Vec<NeuronId> = brain.recall(&cue, 1).into_iter().map(|(_, u)| u).collect();
            let ans = dec.decode(&fired, 3);
            let ov = overlap(&{ let mut g = fired.clone(); g.sort_unstable(); g.dedup(); g }, &earned[i]);
            let correct = ans.first().map(|(w, _)| w == &f.form).unwrap_or(false);
            println!("  {:>8} + {:<22} -> {:<14} {}  [code {}/{}]",
                f.lemma, f.tags.join(";"),
                ans.first().map(|(w, _)| w.clone()).unwrap_or("(silence)".into()),
                if correct { "CORRECT".to_string() } else { format!("WRONG, wanted {}", f.form) },
                ov, earned[i].len());
            break;
        }
    }

    // accuracy over many questions
    println!("\n=== accuracy over 2000 questions ===");
    let trials: Vec<usize> = (0..2000).map(|i| i * 131 % facts.len()).collect();
    let (mut top1, mut top3, mut silent) = (0usize, 0usize, 0usize);
    let t = Instant::now();
    for &ti in &trials {
        let q = query_features(&facts[ti]);
        let cue = alloc.cue(&q, 7);
        if cue.is_empty() { silent += 1; continue; }
        let fired: Vec<NeuronId> = brain.recall(&cue, 1).into_iter().map(|(_, u)| u).collect();
        let ans = dec.decode(&fired, 3);
        if ans.is_empty() { silent += 1; continue; }
        if ans[0].0 == facts[ti].form { top1 += 1; }
        if ans.iter().any(|(w, _)| w == &facts[ti].form) { top3 += 1; }
    }
    let d = t.elapsed();
    println!("  top-1 correct: {top1}/{} ({:.1}%)", trials.len(), top1 as f64 / trials.len() as f64 * 100.0);
    println!("  top-3 correct: {top3}/{} ({:.1}%)", trials.len(), top3 as f64 / trials.len() as f64 * 100.0);
    println!("  silent:        {silent}");
    println!("  {:.0} questions/sec", trials.len() as f64 / d.as_secs_f64());
}
