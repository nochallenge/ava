// Does spiking recall return the same answer as v3's one-shot recall?
// That is the only honest test of the merge. If the event-driven version disagrees
// with the closed-form version, one of them is wrong.
use design_tests::living::{Encoder, Fact};
use design_tests::v4::*;
use std::env;
use std::fs;
use std::time::Instant;

const PATH: &str = "../data/unimorph-eng.tsv";

fn main() {
    let a: Vec<String> = env::args().collect();
    let limit: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let n: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(300_000);
    let k: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(12);

    let raw = fs::read_to_string(PATH).expect("run from design-tests/");
    let facts: Vec<Fact> = raw.lines().filter_map(Fact::parse).take(limit).collect();
    let enc = Encoder::fit(&facts, n, k, 8);
    let codes: Vec<Vec<u32>> = facts.iter().map(|f| enc.encode(f)).collect();
    println!("{} facts, N={n}, k={k}", facts.len());

    let mut brain = Brain::new(n, Params::default());
    let t = Instant::now();
    for c in &codes { brain.learn(c, 1); }
    println!("learned in {:?}, {} synapses\n", t.elapsed(), brain.total_synapses());

    let trials: Vec<usize> = (0..300).map(|i| i * 89 % codes.len()).collect();
    let cue_n = 6usize;

    // ---- does jitter matter? without it every input lands on one tick ----
    println!("=== does delay jitter create the temporal code? ===");
    println!("{:>8} {:>9} {:>10} {:>11} {:>13}", "jitter", "recall", "precision", "vs oneshot", "spread(ms)");
    for &j in &[0u64, 1, 2, 4, 8] {
        let mut p = Params::default();
        p.jitter = j * MS;
        brain.p = p;
        let (mut rec, mut prec, mut agree, mut spread, mut cnt) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for &ti in &trials {
            let full = &codes[ti];
            if full.len() < cue_n + 2 { continue; }
            let cue: Vec<u32> = full.iter().take(cue_n).copied().collect();
            let sp = brain.recall(&cue, 1);
            let mut got: Vec<u32> = sp.iter().map(|&(_, id)| id).collect();
            got.sort_unstable(); got.dedup();
            let one = brain.recall_oneshot(&cue, 0.55, k * 3);
            let ov = overlap(&got, full);
            rec += ov as f64 / full.len() as f64;
            prec += if got.is_empty() { 0.0 } else { ov as f64 / got.len() as f64 };
            let inter = overlap(&got, &one);
            let uni = got.len() + one.len() - inter;
            agree += if uni == 0 { 1.0 } else { inter as f64 / uni as f64 };
            if sp.len() > 1 {
                spread += (sp.iter().map(|&(t, _)| t).max().unwrap()
                    - sp.iter().map(|&(t, _)| t).min().unwrap()) as f64 / MS as f64;
            }
            cnt += 1.0;
        }
        println!("{:>7}ms {:>8.0}% {:>9.0}% {:>10.0}% {:>13.1}",
            j, rec / cnt * 100.0, prec / cnt * 100.0, agree / cnt * 100.0, spread / cnt);
    }

    // ---- timing: are strongly driven units firing FIRST? ----
    println!("\n=== rank ordering: do pattern members fire before spurious units? ===");
    brain.p = Params::default();
    let mut in_first = 0usize;
    let mut tot = 0usize;
    let mut member_t = 0f64;
    let mut other_t = 0f64;
    let (mut mc, mut oc) = (0f64, 0f64);
    for &ti in &trials {
        let full = &codes[ti];
        if full.len() < cue_n + 2 { continue; }
        let cue: Vec<u32> = full.iter().take(cue_n).copied().collect();
        let sp = brain.recall(&cue, 1);
        for (rank, &(t, id)) in sp.iter().enumerate() {
            let is_member = full.binary_search(&id).is_ok();
            if is_member { member_t += t as f64 / MS as f64; mc += 1.0;
                if rank < full.len() { in_first += 1; } }
            else { other_t += t as f64 / MS as f64; oc += 1.0; }
        }
        tot += full.len();
    }
    println!("  pattern members landing in the first k spikes: {in_first} of {tot} ({:.0}%)",
        in_first as f64 / tot as f64 * 100.0);
    println!("  mean first-spike time, pattern members: {:.2} ms", member_t / mc.max(1.0));
    println!("  mean first-spike time, spurious units : {:.2} ms", other_t / oc.max(1.0));

    // ---- speed ----
    println!("\n=== speed ===");
    let t = Instant::now();
    let mut spikes = 0usize;
    for &ti in &trials {
        let full = &codes[ti];
        if full.len() < cue_n + 2 { continue; }
        let cue: Vec<u32> = full.iter().take(cue_n).copied().collect();
        spikes += brain.recall(&cue, 1).len();
    }
    let d = t.elapsed();
    println!("  {} spiking recalls in {:?} = {:.0} recalls/sec, {:.1} spikes each",
        trials.len(), d, trials.len() as f64 / d.as_secs_f64(), spikes as f64 / trials.len() as f64);

    let t = Instant::now();
    for &ti in &trials {
        let full = &codes[ti];
        if full.len() < cue_n + 2 { continue; }
        let cue: Vec<u32> = full.iter().take(cue_n).copied().collect();
        let _ = brain.recall_oneshot(&cue, 0.55, k * 3);
    }
    let d2 = t.elapsed();
    println!("  {} one-shot recalls in {:?} = {:.0} recalls/sec",
        trials.len(), d2, trials.len() as f64 / d2.as_secs_f64());
    println!("  spiking costs {:.1}x the one-shot version", d.as_secs_f64() / d2.as_secs_f64());
}
