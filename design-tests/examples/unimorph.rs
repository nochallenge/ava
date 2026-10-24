// Run the real UniMorph English corpus through the encoder, then store it in
// synapses only - no chunk table, no tag, no index anywhere.
//
// The test that matters: give the network a PARTIAL cue and see whether it
// reconstructs the rest of the pattern, when there is nothing in the data
// structure that could possibly be looked up.
use design_tests::living::*;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Instant;

const PATH: &str = "../data/unimorph-eng.tsv";

fn main() {
    let args: Vec<String> = env::args().collect();
    let limit: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let n: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(24);

    let t0 = Instant::now();
    let raw = fs::read_to_string(PATH).expect("run from design-tests/, data at ../data/");
    let all: Vec<Fact> = raw.lines().filter_map(Fact::parse).collect();
    println!("loaded {} facts in {:?}", all.len(), t0.elapsed());
    println!("example: {:?}\n", all.iter().find(|f| f.lemma == "eat" && f.tags[0] == "V").map(|f| (&f.lemma, &f.form, &f.tags)));

    // ---------- encode the WHOLE corpus, to see what the encoder does at scale
    println!("=== encoding all {} facts, N={n}, k={k} ===", all.len());
    let t = Instant::now();
    let enc_all = Encoder::fit(&all, n, k, 8);
    println!("IDF fitted over {} distinct features in {:?}", enc_all.idf.len(), t.elapsed());

    let t = Instant::now();
    let codes_all: Vec<Vec<NeuronId>> = all.iter().map(|f| enc_all.encode(f)).collect();
    let dur = t.elapsed();
    println!("encoded {} facts in {:?} = {:.0} facts/sec",
        codes_all.len(), dur, codes_all.len() as f64 / dur.as_secs_f64());

    // how spread out is the population actually being used?
    let mut usage: HashMap<NeuronId, u32> = HashMap::new();
    for c in &codes_all { for &u in c { *usage.entry(u).or_insert(0) += 1; } }
    let used = usage.len();
    let mut counts: Vec<u32> = usage.values().copied().collect();
    counts.sort_unstable();
    let mean = counts.iter().map(|&c| c as f64).sum::<f64>() / counts.len() as f64;
    println!("units used: {used} of {n} ({:.1}%)", used as f64 / n as f64 * 100.0);
    println!("facts per unit: mean {:.1}, median {}, p99 {}, max {}",
        mean, counts[counts.len()/2], counts[counts.len()*99/100], counts[counts.len()-1]);
    println!("^ this is the 'shared neurons' property, measured. one unit takes part");
    println!("  in {:.0} different memories on average.\n", mean);

    // sample overlaps
    let sample: Vec<usize> = (0..2000).map(|i| i * 313 % codes_all.len()).collect();
    let (mut mx, mut tot, mut cnt) = (0usize, 0u64, 0u64);
    for i in 0..sample.len() {
        for j in i+1..sample.len().min(i+60) {
            let o = overlap(&codes_all[sample[i]], &codes_all[sample[j]]);
            mx = mx.max(o); tot += o as u64; cnt += 1;
        }
    }
    println!("pairwise overlap over {cnt} sampled pairs: mean {:.3}, max {mx}", tot as f64 / cnt as f64);

    // ---------- now build the LIVING brain on a subset
    println!("\n=== storing {limit} facts as synaptic strengths only ===");
    let facts: Vec<Fact> = all.into_iter().take(limit).collect();
    let enc = Encoder::fit(&facts, n, k, 8);
    let codes: Vec<Vec<NeuronId>> = facts.iter().map(|f| enc.encode(f)).collect();

    let t = Instant::now();
    let mut brain = Brain::new(n);
    for c in &codes { brain.learn(c, 1); }
    println!("learned in {:?}. synapses: {} ({:.1} MB)",
        t.elapsed(), brain.total_synapses(), brain.total_synapses() as f64 * 6.0 / 1e6);
    println!("there is no chunk table. Unit has no `tag` field. the only thing that");
    println!("changed is connection strength.\n");

    // superposition: one unit's weights carry many facts at once
    let busiest = (0..n).max_by_key(|&i| brain.units[i as usize].degree()).unwrap();
    let u = &brain.units[busiest as usize];
    println!("busiest unit {busiest}: wired to {} partners, total strength {}",
        u.degree(), u.load());
    let strong = u.out.iter().filter(|&&(_, w)| w > 1).count();
    println!("  synapses strengthened more than once (shared by several facts): {strong}");
    println!("  ^ superposition. no way to point at 'where fact 47 is'.\n");

    // ---------- THE TEST: recall from a partial cue, no lookup possible
    println!("=== recall from a partial cue ===");
    println!("{:>6} {:>9} {:>9} {:>9}  {}", "cue", "recalled", "of k", "precision", "reconstructed the fact?");
    let trials: Vec<usize> = (0..200).map(|i| i * 97 % codes.len()).collect();
    for &cue_n in &[6usize, 8, 12, 16] {
        let (mut hit, mut prec, mut perfect) = (0f64, 0f64, 0usize);
        for &ti in &trials {
            let full = &codes[ti];
            if full.len() < cue_n + 2 { continue; }
            let cue: Vec<NeuronId> = full.iter().take(cue_n).copied().collect();
            let got = brain.recall(&cue, 0.55, k * 2);
            let ov = overlap(&got, full);
            hit += ov as f64 / full.len() as f64;
            prec += if got.is_empty() { 0.0 } else { ov as f64 / got.len() as f64 };
            if ov >= full.len() - 1 { perfect += 1; }
        }
        let t = trials.len() as f64;
        println!("{:>6} {:>8.0}% {:>9} {:>8.0}%  {} of {} rebuilt within 1 unit",
            cue_n, hit / t * 100.0, k, prec / t * 100.0, perfect, trials.len());
    }

    // ---------- repetition strengthens: the memory is alive, not written once
    println!("\n=== does repetition strengthen it? (learning the same fact again) ===");
    let probe = &codes[7];
    let (a, b) = (probe[0], probe[1]);
    println!("  strength {a}->{b} after 1 exposure: {}", brain.units[a as usize].strength_to(b));
    for _ in 0..4 { brain.learn(probe, 1); }
    println!("  after 5 exposures:                  {}", brain.units[a as usize].strength_to(b));
    println!("  ^ same synapse, no new record created. that is what 'living' means here.");
}
