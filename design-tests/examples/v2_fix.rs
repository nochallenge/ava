// Root cause found in v2_worst: the cued chunk's units RE-FIRE about 6 times each
// (2 ms refractory, 400 ms window). Every re-fire re-delivers current to any rival
// sharing units, and with a 10 ms membrane time constant that current ACCUMULATES
// until the rival crosses threshold. A rival needs 14 shared units to complete even
// though 14 * w = 126 < V_th = 150 on any single pass.
//
// v1 experiment 2 tested a long refractory WITHOUT inhibition and it failed.
// This tests long refractory WITH inhibition, which was never tried.
use design_tests::v2::*;

const POP: usize = 20_000;
const KK: usize = 40;
const WINDOW: Tick = 400 * MS;

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

fn run(codes: &[Vec<NeuronId>], cue_n: usize, refractory: Tick, inh_on: bool) -> (usize, usize, usize, f64) {
    let inh = if inh_on { Inhibition::new(15.0 * MS as f64, 1.0, 3.0) }
              else { Inhibition::new(1.0, 0.0, 0.0) };
    let mut b = Brain::new(POP, inh, refractory);
    for (i, c) in codes.iter().enumerate() { install_chunk_drive(&mut b, i as u32, c, cue_n, 1.8); }
    let seed: Vec<NeuronId> = codes[0].iter().take(cue_n).copied().collect();
    cue(&mut b, &seed, 0);
    let spikes = b.run(WINDOW, 500_000);
    let mut who: Vec<NeuronId> = spikes.iter().map(|&(_, id)| id).collect();
    who.sort_unstable(); who.dedup();
    let recalled = overlap(&who, &codes[0]);
    let hits: Vec<usize> = (1..codes.len()).map(|i| overlap(&who, &codes[i])).collect();
    let full = hits.iter().filter(|&&h| h >= 35).count();
    let worst = *hits.iter().max().unwrap_or(&0);
    let per_unit = spikes.len() as f64 / recalled.max(1) as f64;
    (recalled, full, worst, per_unit)
}

fn main() {
    let texts = corpus(200);
    let enc = Encoder::fit(&texts, POP, KK);
    let codes: Vec<Vec<NeuronId>> = texts.iter().map(|t| enc.encode(t)).collect();
    println!("200 chunks, N={POP}, k={KK}, cue=30. max code overlap with chunk 0 = {}",
        (1..codes.len()).map(|i| overlap(&codes[0], &codes[i])).max().unwrap());
    println!("target: recalled 40, zero rivals complete, ~1 spike per unit\n");

    println!("{:>12} {:>5} {:>10} {:>16} {:>12} {:>12}", "refractory", "inh", "recalled", "rivals complete", "worst rival", "spikes/unit");
    for &(label, refr) in &[("2 ms", 2 * MS), ("20 ms", 20 * MS), ("100 ms", 100 * MS), ("whole window", WINDOW + 1)] {
        for &inh_on in &[false, true] {
            let (r, full, worst, per) = run(&codes, 30, refr, inh_on);
            println!("{:>12} {:>5} {:>10} {:>16} {:>12} {:>12.1}",
                label, if inh_on { "on" } else { "off" }, r, full, worst, per);
        }
    }

    println!("\n=== with one-spike-per-recall + inhibition, how far does it scale? ===");
    println!("{:>7} {:>5} {:>10} {:>16} {:>12} {:>12}", "chunks", "cue", "recalled", "rivals complete", "worst rival", "spikes/unit");
    for &c in &[200usize, 1000, 3000] {
        let texts = corpus(c);
        let enc = Encoder::fit(&texts, POP, KK);
        let codes: Vec<Vec<NeuronId>> = texts.iter().map(|t| enc.encode(t)).collect();
        for &cue_n in &[20usize, 30] {
            let (r, full, worst, per) = run(&codes, cue_n, WINDOW + 1, true);
            println!("{:>7} {:>5} {:>10} {:>16} {:>12} {:>12.1}", c, cue_n, r, full, worst, per);
        }
    }
}
