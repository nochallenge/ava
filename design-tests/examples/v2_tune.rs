// find the operating regime for the inhibition fix, then test the chain of thought.
use design_tests::v2::*;

const WINDOW: Tick = 200 * MS;

#[derive(Clone, Copy)]
struct Cfg {
    inh_tau: f64,
    per_spike: f64,
    gain: f64,
    drive: f64,
    refractory: Tick,
}

fn build(cfg: &Cfg, codes: &[Vec<NeuronId>], expected_cue: usize) -> Brain {
    let mut b = Brain::new(N, Inhibition::new(cfg.inh_tau, cfg.per_spike, cfg.gain), cfg.refractory);
    for (i, c) in codes.iter().enumerate() {
        install_chunk_drive(&mut b, i as u32, c, expected_cue, cfg.drive);
    }
    b
}

fn trial(cfg: &Cfg, codes: &[Vec<NeuronId>], cue_n: usize) -> (usize, Vec<usize>) {
    let mut b = build(cfg, codes, cue_n);
    let seed: Vec<NeuronId> = codes[0].iter().take(cue_n).copied().collect();
    cue(&mut b, &seed, 0);
    let spikes = b.run(WINDOW, 200_000);
    (spikes.len(), readout(&spikes, codes))
}

fn verdict(rec: &[usize], spikes: usize) -> &'static str {
    let leak: usize = rec[1..].iter().sum();
    if rec[0] < 35 { "incomplete" }
    else if leak > 4 { "leaks" }
    else if spikes > 4 * K { "runaway" }
    else { "CLEAN" }
}

fn sweep(label: &str, codes: &[Vec<NeuronId>]) -> Option<Cfg> {
    let mo = (1..codes.len()).map(|i| overlap(&codes[0], &codes[i])).max().unwrap_or(0);
    println!("\n=== {label} | max overlap chunk0 vs rivals = {mo} ===");
    println!("{:>7} {:>6} {:>5} {:>6} {:>7}  {:>18}  {}", "inhtau", "spike", "gain", "drive", "spikes", "[c0,c1,c2]", "verdict");
    let mut best: Option<(f64, Cfg)> = None;
    let mut shown = 0;
    for &inh_tau in &[2.0 * MS as f64, 5.0 * MS as f64, 15.0 * MS as f64] {
        for &per_spike in &[1.0, 2.0] {
            for &gain in &[1.0, 3.0, 8.0, 20.0] {
                for &drive in &[1.3, 1.8, 2.5, 4.0] {
                    let cfg = Cfg { inh_tau, per_spike, gain, drive, refractory: 2 * MS };
                    let (spikes, rec) = trial(&cfg, codes, 10);
                    let v = verdict(&rec, spikes);
                    if v == "CLEAN" {
                        let leak: usize = rec[1..].iter().sum();
                        let score = rec[0] as f64 - leak as f64 * 5.0 - spikes as f64 * 0.02;
                        if best.as_ref().map_or(true, |(s, _)| score > *s) { best = Some((score, cfg)); }
                        if shown < 8 {
                            println!("{:>7.0} {:>6.1} {:>5.1} {:>6.1} {:>7}  {:>18}  {}",
                                inh_tau / MS as f64, per_spike, gain, drive, spikes, format!("{:?}", rec), v);
                            shown += 1;
                        }
                    }
                }
            }
        }
    }
    match best {
        Some((_, c)) => {
            println!("-> best: inh_tau={} ms, per_spike={}, gain={}, drive={}",
                c.inh_tau / MS as f64, c.per_spike, c.gain, c.drive);
            Some(c)
        }
        None => { println!("-> NO CLEAN CONFIGURATION"); None }
    }
}

fn main() {
    let texts = [
        "the verb agrees with the subject in number and person",
        "a comma separates items in a list of three or more",
        "the past tense of irregular verbs must be memorised",
    ];

    // the hard case first: separation OFF, so overlap is high. can inhibition alone cope?
    let raw: Vec<Vec<NeuronId>> = texts.iter().map(|t| encode_raw(t)).collect();
    let raw_cfg = sweep("inhibition ONLY, separation disabled", &raw);

    // and with separation on, which is the actual design
    let sep: Vec<Vec<NeuronId>> = texts.iter().map(|t| encode(t)).collect();
    let cfg = match sweep("inhibition + separation (the real v2)", &sep) {
        Some(c) => c,
        None => return,
    };
    if raw_cfg.is_none() {
        println!("\n**inhibition alone is NOT sufficient. separation is still required.**");
    } else {
        println!("\n**inhibition alone was sufficient even at high overlap.**");
    }

    println!("\n--- capacity at the chosen config ---");
    let words = ["syntax","clause","vowel","tense","plural","adverb","pronoun","gerund",
                 "article","subject","object","modal","passive","phoneme","idiom","prefix",
                 "suffix","stress","schwa","accent","dative","aspect","mood","voice"];
    let corpus: Vec<String> = (0..400)
        .map(|i| format!("lesson {i} explains {} before {} because {} governs {}",
            words[i % 24], words[(i * 5 + 7) % 24], words[(i * 11 + 3) % 24], words[(i * 17 + 13) % 24]))
        .collect();
    let many: Vec<Vec<NeuronId>> = corpus.iter().map(|t| encode(t)).collect();
    for load in [10usize, 50, 100, 200, 400] {
        let subset = many[..load].to_vec();
        let (spikes, rec) = trial(&cfg, &subset, 10);
        let leak: usize = rec[1..].iter().sum();
        println!("{load:>4} chunks | cued {:>2}/40 | leaked {leak:>4} units over {:>3} rivals | {spikes:>5} spikes",
            rec[0], load - 1);
    }

    println!("\n--- chain of thought: A -> B -> C, sweeping the sequence delay ---");
    let chain = [
        "nouns name people places and things",
        "verbs describe what nouns do",
        "adverbs modify how verbs act",
    ];
    let cc: Vec<Vec<NeuronId>> = chain.iter().map(|t| encode(t)).collect();
    println!("{:>9} {:>7}  {:>18}  {}", "delay_ms", "spikes", "[A,B,C]", "verdict");
    for &d in &[1u64, 5, 10, 20, 40, 80, 150] {
        let mut b = build(&cfg, &cc, 10);
        for i in 0..cc.len() - 1 {
            install_sequence(&mut b, &cc[i], &cc[i + 1], d * MS, 1.6);
        }
        let seed: Vec<NeuronId> = cc[0].iter().take(10).copied().collect();
        cue(&mut b, &seed, 0);
        let spikes = b.run(WINDOW, 200_000);
        let rec = readout(&spikes, &cc);
        let v = if rec[1] < 30 { "dies at B" }
            else if rec[2] < 30 { "reaches B, dies at C" }
            else if spikes.len() > 12 * K { "runaway" }
            else { "FULL CHAIN" };
        println!("{:>9} {:>7}  {:>18}  {}", d, spikes.len(), format!("{:?}", rec), v);
    }
}
