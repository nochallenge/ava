// How far up does this scale, and what does it cost?
use design_tests::living::{Encoder, Fact};
use design_tests::v4::*;
use std::env;
use std::fs;
use std::time::Instant;

const PATH: &str = "../data/unimorph-eng.tsv";

fn rss_mb() -> f64 {
    // rough: count what we allocated rather than asking the OS
    0.0
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let limit: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(652_477);
    let k: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);

    let raw = fs::read_to_string(PATH).expect("run from design-tests/");
    let facts: Vec<Fact> = raw.lines().filter_map(Fact::parse).take(limit).collect();
    println!("{} facts, k={k}\n", facts.len());
    println!("{:>10} {:>9} {:>10} {:>9} {:>8} {:>9} {:>10} {:>12}",
        "N", "encode", "learn", "synapses", "RAM MB", "recall", "precision", "recalls/sec");

    for &n in &[300_000u32, 1_000_000, 3_000_000] {
        let t = Instant::now();
        let enc = Encoder::fit(&facts, n, k, 8);
        let codes: Vec<Vec<u32>> = facts.iter().map(|f| enc.encode(f)).collect();
        let enc_t = t.elapsed();

        let t = Instant::now();
        let mut brain = Brain::new(n, Params::default());
        for c in &codes { brain.learn(c, 1); }
        let learn_t = t.elapsed();
        let syn = brain.total_synapses();
        // (u32,u16) padded to 8 bytes in a Vec, plus 32B per Vec header, plus Unit fields
        let ram = (syn as f64 * 8.0 + n as f64 * 48.0) / 1e6;

        let trials: Vec<usize> = (0..500).map(|i| i * 1013 % codes.len()).collect();
        let (mut rec, mut prec, mut cnt) = (0f64, 0f64, 0f64);
        let t = Instant::now();
        let mut spikes = 0usize;
        for &ti in &trials {
            let full = &codes[ti];
            if full.len() < 8 { continue; }
            let cue: Vec<u32> = full.iter().take(6).copied().collect();
            let sp = brain.recall(&cue, 1);
            spikes += sp.len();
            let mut got: Vec<u32> = sp.iter().map(|&(_, id)| id).collect();
            got.sort_unstable(); got.dedup();
            let ov = overlap(&got, full);
            rec += ov as f64 / full.len() as f64;
            prec += if got.is_empty() { 0.0 } else { ov as f64 / got.len() as f64 };
            cnt += 1.0;
        }
        let rt = t.elapsed();
        println!("{:>10} {:>8.1}s {:>9.1}s {:>9} {:>8.0} {:>8.0}% {:>9.0}% {:>12.0}",
            n, enc_t.as_secs_f64(), learn_t.as_secs_f64(), syn, ram,
            rec / cnt * 100.0, prec / cnt * 100.0, cnt / rt.as_secs_f64());
        let _ = spikes;
        let _ = rss_mb();
    }

    println!("\nprojection to a full english curriculum:");
    let per_fact_syn = (k * (k - 1)) as f64;
    for &(label, m) in &[("UniMorph only", 652_477f64),
                          ("+ Tatoeba sentences", 652_477.0 + 2_036_641.0),
                          ("+ WordNet synsets", 652_477.0 + 2_036_641.0 + 120_068.0)] {
        let syn = m * per_fact_syn;
        let n = m * 2.0;
        println!("  {label:<22} {m:>10.0} facts -> N={:>10.0}, {:>12.0} synapses, {:>7.1} GB, learn ~{:.0}s",
            n, syn, (syn * 8.0 + n * 48.0) / 1e9, m / 70_000.0);
    }
}
