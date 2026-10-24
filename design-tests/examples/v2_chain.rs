// Chain of thought under the fixed config: IDF separation + one spike per unit per
// recall + inhibition. Each unit may fire once, so a chain must walk through
// DISTINCT chunks - which is exactly what a train of thought is.
use design_tests::v2::*;
use std::collections::HashMap;

const POP: usize = 20_000;
const KK: usize = 40;
const WINDOW: Tick = 2_000 * MS;

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

fn exclusive(codes: &[Vec<NeuronId>]) -> Vec<Vec<NeuronId>> {
    let mut n: HashMap<NeuronId, usize> = HashMap::new();
    for c in codes { for &u in c { *n.entry(u).or_insert(0) += 1; } }
    codes.iter().map(|c| c.iter().copied().filter(|u| n[u] == 1).collect()).collect()
}

fn main() {
    let texts = corpus(500);
    let enc = Encoder::fit(&texts, POP, KK);
    let codes: Vec<Vec<NeuronId>> = texts.iter().map(|t| enc.encode(t)).collect();
    let excl = exclusive(&codes);

    // chain over chunks that do NOT overlap each other, the sensible case
    let mut chain = vec![0usize];
    for i in 1..codes.len() {
        if chain.len() == 6 { break; }
        if chain.iter().all(|&c| overlap(&codes[c], &codes[i]) <= 2) { chain.push(i); }
    }
    println!("chain over chunks {:?} (pairwise overlap <= 2)", chain);
    println!("cue 20 units of the first, then follow the sequence links\n");

    println!("{:>9} {:>8} {:>10}  {}", "delay_ms", "spikes", "in order?", "first-spike time of each chunk (ms)");
    for &d in &[20u64, 50, 100, 200] {
        let mut b = Brain::new(POP, Inhibition::new(15.0 * MS as f64, 1.0, 3.0), WINDOW + 1);
        for (i, c) in codes.iter().enumerate() { install_chunk_drive(&mut b, i as u32, c, 20, 1.8); }
        for w in chain.windows(2) {
            install_sequence(&mut b, &codes[w[0]], &codes[w[1]], d * MS, 1.6);
        }
        let seed: Vec<NeuronId> = codes[chain[0]].iter().take(20).copied().collect();
        cue(&mut b, &seed, 0);
        let spikes = b.run(WINDOW, 500_000);

        let mut firsts = Vec::new();
        let mut ok = true;
        let mut prev = -1i64;
        for &ci in &chain {
            let t: Vec<Tick> = spikes.iter()
                .filter(|(_, id)| excl[ci].binary_search(id).is_ok())
                .map(|&(t, _)| t).collect();
            if t.is_empty() { firsts.push(-1); ok = false; continue; }
            let f = t[0] as i64 / MS as i64;
            if f < prev { ok = false; }
            prev = f;
            firsts.push(f);
        }
        let complete = chain.iter().all(|&ci| overlap(
            &{ let mut w: Vec<NeuronId> = spikes.iter().map(|&(_, i)| i).collect(); w.sort_unstable(); w.dedup(); w },
            &codes[ci]) >= 35);
        println!("{:>9} {:>8} {:>10}  {:?}", d, spikes.len(),
            if ok && complete { "YES" } else if complete { "no" } else { "broken" }, firsts);
    }

    println!("\nspikes per unit at delay 100 ms (1.0 means each unit fired exactly once):");
    let mut b = Brain::new(POP, Inhibition::new(15.0 * MS as f64, 1.0, 3.0), WINDOW + 1);
    for (i, c) in codes.iter().enumerate() { install_chunk_drive(&mut b, i as u32, c, 20, 1.8); }
    for w in chain.windows(2) { install_sequence(&mut b, &codes[w[0]], &codes[w[1]], 100 * MS, 1.6); }
    let seed: Vec<NeuronId> = codes[chain[0]].iter().take(20).copied().collect();
    cue(&mut b, &seed, 0);
    let spikes = b.run(WINDOW, 500_000);
    let mut uniq: Vec<NeuronId> = spikes.iter().map(|&(_, i)| i).collect();
    uniq.sort_unstable(); uniq.dedup();
    println!("  {} spikes over {} distinct units = {:.2}",
        spikes.len(), uniq.len(), spikes.len() as f64 / uniq.len() as f64);
    println!("  units expected if only the 6 chain chunks fired: {}", 6 * KK);
}
