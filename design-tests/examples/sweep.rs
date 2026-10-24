// design experiment, not part of ava. finds the usable window for the recurrent
// weight w_mv: too small and a partial cue never completes the pattern, too big
// and the population runs away and everything fires.
use design_tests::*;

fn main() {
    let text = "the verb agrees with the subject in number and person";
    let code = encode(text);
    let cue: Vec<NeuronId> = code.iter().take(code.len() / 4).copied().collect();

    println!("code size k = {}, cue size = {}, V_th = 15 mV", code.len(), cue.len());
    println!("predicted minimum w = V_th / cue = {:.2} mV\n", 15.0 / cue.len() as f64);
    println!("{:>5} {:>10} {:>12} {:>10}", "w_mv", "recovered", "total_spikes", "verdict");

    for w in 1..=6 {
        let mut brain = Brain::new(N, 10.0 * MS as f64, 15.0);
        install(&mut brain, 0, &code, w, MS);
        for &id in &cue {
            brain.queue.push(Event { at: 0, target: id, weight_mv: 100 });
        }
        let spikes = brain.run(50 * MS, 100_000);
        let mut who: Vec<NeuronId> = spikes.iter().map(|&(_, id)| id).collect();
        who.sort_unstable();
        who.dedup();
        let recovered = overlap(&who, &code);
        let verdict = if recovered <= cue.len() {
            "dead"
        } else if spikes.len() > 20 * code.len() {
            "runaway"
        } else {
            "ok"
        };
        println!("{:>5} {:>10} {:>12} {:>10}", w, recovered, spikes.len(), verdict);
    }

    // interference: three unrelated chunks in one population, cue only the first
    println!("\n--- interference, 3 chunks in one population of {N} ---");
    let texts = [
        "the verb agrees with the subject in number and person",
        "a comma separates items in a list of three or more",
        "the past tense of irregular verbs must be memorised",
    ];
    let codes: Vec<Vec<NeuronId>> = texts.iter().map(|t| encode(t)).collect();
    for (i, a) in codes.iter().enumerate() {
        for (j, b) in codes.iter().enumerate().skip(i + 1) {
            println!("overlap(chunk {i}, chunk {j}) = {}", overlap(a, b));
        }
    }

    let mut brain = Brain::new(N, 10.0 * MS as f64, 15.0);
    for (i, c) in codes.iter().enumerate() {
        install(&mut brain, i as u32, c, 2, MS);
    }
    let cue0: Vec<NeuronId> = codes[0].iter().take(10).copied().collect();
    let out = recall(&mut brain, &cue0, 0, 50 * MS);
    for (i, c) in codes.iter().enumerate() {
        println!("cue from chunk 0 -> recovered {} of chunk {i}", overlap(&out, c));
    }

    // provenance: which chunks does one unit hold a piece of?
    let shared: Vec<NeuronId> = (0..N).filter(|&i| brain.units[i].tag.len() > 1).collect();
    println!("\nunits holding a piece of more than one chunk: {}", shared.len());
    if let Some(&u) = shared.first() {
        println!("unit {u} carries tags {:?}", brain.units[u].tag);
    }
}
