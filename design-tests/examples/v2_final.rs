// End to end validation of the fixed 8.1 design.
//   IDF encoder (separation) + global feedback inhibition (k-WTA, termination)
//   + two synapse types (completion within a chunk, sequence between chunks).
//
// NOTE ON METRICS: the first version of this file summed leaked units across ALL
// rival chunks and compared the total to a fixed threshold. With 1000 rivals that
// reports "leaking" even when each rival gets 0.4 units, which is fine. And a rival
// that merely SHARES units with the cued chunk scores a false hit. Both are fixed
// here: leak is measured per rival, and sequence timing uses only units that are
// EXCLUSIVE to one chunk.
use design_tests::v2::*;
use std::collections::HashMap;

const WINDOW: Tick = 400 * MS;
const POP: usize = 20_000;
const KK: usize = 40;

fn corpus(c: usize) -> Vec<String> {
    let topic = ["plural","tense","article","pronoun","adverb","clause","vowel","gerund",
                 "modal","passive","prefix","suffix","stress","idiom","dative","aspect"];
    (0..c).map(|i| format!("rule {i} states that the {} form of {} changes when {} follows",
        topic[i % 16], token(i, 1), token(i, 2))).collect()
}

fn token(i: usize, s: u64) -> String {
    let h = fnv1a(&i.to_le_bytes(), s);
    let c = ["b","d","f","g","k","l","m","n","p","r","s","t","v","z"];
    let v = ["a","e","i","o","u"];
    format!("{}{}{}{}{}", c[(h % 14) as usize], v[(h / 14 % 5) as usize],
        c[(h / 70 % 14) as usize], v[(h / 980 % 5) as usize], c[(h / 4900 % 14) as usize])
}

fn make(codes: &[Vec<NeuronId>], cue_n: usize, drive: f64) -> Brain {
    let mut b = Brain::new(POP, Inhibition::new(15.0 * MS as f64, 1.0, 3.0), 2 * MS);
    for (i, c) in codes.iter().enumerate() {
        install_chunk_drive(&mut b, i as u32, c, cue_n, drive);
    }
    b
}

/// units belonging to exactly one chunk — the only honest evidence a chunk fired
fn exclusive(codes: &[Vec<NeuronId>]) -> Vec<Vec<NeuronId>> {
    let mut count: HashMap<NeuronId, usize> = HashMap::new();
    for c in codes { for &u in c { *count.entry(u).or_insert(0) += 1; } }
    codes.iter().map(|c| c.iter().copied().filter(|u| count[u] == 1).collect()).collect()
}

fn main() {
    println!("v2: IDF separation + global feedback inhibition, N={POP}, k={KK}\n");

    println!("=== recall vs stored chunks. leak measured PER RIVAL, not summed ===");
    println!("{:>7} {:>5} {:>9} {:>10} {:>10} {:>8}  {}",
        "chunks", "cue", "recalled", "worstRival", "meanRival", "spikes", "verdict");
    for &c in &[50usize, 200, 1000] {
        let texts = corpus(c);
        let enc = Encoder::fit(&texts, POP, KK);
        let codes: Vec<Vec<NeuronId>> = texts.iter().map(|t| enc.encode(t)).collect();
        let excl = exclusive(&codes);
        for &cue_n in &[10usize, 20, 30] {
            let mut b = make(&codes, cue_n, 1.8);
            let seed: Vec<NeuronId> = codes[0].iter().take(cue_n).copied().collect();
            cue(&mut b, &seed, 0);
            let spikes = b.run(WINDOW, 500_000);
            let mut who: Vec<NeuronId> = spikes.iter().map(|&(_, id)| id).collect();
            who.sort_unstable();
            who.dedup();
            let recalled = overlap(&who, &codes[0]);
            // rivals judged on their EXCLUSIVE units only
            let rival: Vec<f64> = (1..codes.len())
                .map(|i| if excl[i].is_empty() { 0.0 } else {
                    overlap(&who, &excl[i]) as f64 / excl[i].len() as f64 * KK as f64 })
                .collect();
            let worst = rival.iter().cloned().fold(0.0f64, f64::max);
            let mean = rival.iter().sum::<f64>() / rival.len() as f64;
            let v = if recalled < 35 { "incomplete" }
                else if worst > 10.0 { "leaks" }
                else if spikes.len() > 6 * KK { "runaway" }
                else { "CLEAN" };
            println!("{:>7} {:>5} {:>9} {:>10.1} {:>10.2} {:>8}  {}",
                c, cue_n, recalled, worst, mean, spikes.len(), v);
        }
        println!();
    }

    println!("=== chain of thought: do chunks fire in ORDER? (exclusive units only) ===");
    let texts = corpus(200);
    let enc = Encoder::fit(&texts, POP, KK);
    let codes: Vec<Vec<NeuronId>> = texts.iter().map(|t| enc.encode(t)).collect();
    let excl = exclusive(&codes);
    for &d in &[30u64, 150] {
        let mut b = make(&codes, 20, 1.8);
        for i in 0..4 { install_sequence(&mut b, &codes[i], &codes[i + 1], d * MS, 1.6); }
        let seed: Vec<NeuronId> = codes[0].iter().take(20).copied().collect();
        cue(&mut b, &seed, 0);
        let spikes = b.run(WINDOW, 500_000);
        println!("\n  sequence delay {d} ms, {} spikes total", spikes.len());
        let mut prev = 0i64;
        let mut ordered = true;
        for ci in 0..5 {
            let times: Vec<Tick> = spikes.iter()
                .filter(|(_, id)| excl[ci].binary_search(id).is_ok())
                .map(|&(t, _)| t).collect();
            if times.is_empty() { println!("    chunk {ci}: never fired"); ordered = false; continue; }
            let first = times[0] as i64 / MS as i64;
            if first < prev { ordered = false; }
            prev = first;
            println!("    chunk {ci}: {:>3} spikes on {:>2} exclusive units, first at {:>4} ms",
                times.len(), excl[ci].len(), first);
        }
        println!("    -> {}", if ordered { "IN ORDER" } else { "OUT OF ORDER" });
    }
}
