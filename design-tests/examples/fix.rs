// design experiment 2. the sweep showed pure all-to-all excitation has NO usable
// window: w=1 is dead, w>=2 reverberates forever and every stored chunk fires.
// this tests one fix -- one spike per unit per presentation, i.e. make the
// refractory period as long as the recall window. that IS time-to-first-spike
// coding from section 3, so the architecture's own principle is the fix.
use design_tests::*;

const WINDOW: Tick = 50 * MS;

fn trial(refractory: Tick, w: i32, codes: &[Vec<NeuronId>], cue: &[NeuronId]) -> (usize, Vec<usize>) {
    let mut brain = Brain::new(N, 10.0 * MS as f64, 15.0);
    brain.set_refractory(refractory);
    for (i, c) in codes.iter().enumerate() {
        install(&mut brain, i as u32, c, w, MS);
    }
    for &id in cue {
        brain.queue.push(Event { at: 0, target: id, weight_mv: 100 });
    }
    let spikes = brain.run(WINDOW, 100_000);
    let mut who: Vec<NeuronId> = spikes.iter().map(|&(_, id)| id).collect();
    who.sort_unstable();
    who.dedup();
    (spikes.len(), codes.iter().map(|c| overlap(&who, c)).collect())
}

fn main() {
    let texts = [
        "the verb agrees with the subject in number and person",
        "a comma separates items in a list of three or more",
        "the past tense of irregular verbs must be memorised",
    ];
    let codes: Vec<Vec<NeuronId>> = texts.iter().map(|t| encode(t)).collect();
    let cue: Vec<NeuronId> = codes[0].iter().take(10).copied().collect();

    println!("3 chunks of k=40 in a population of {N}, cue = 10 units of chunk 0");
    println!("target: recover ~40 of chunk 0, near 0 of chunks 1 and 2\n");
    println!("{:>12} {:>4} {:>8}  {:>26}", "refractory", "w", "spikes", "recovered [c0, c1, c2]");

    for &(label, refr) in &[("2 ms", 2 * MS), ("10 ms", 10 * MS), ("window", WINDOW + 1)] {
        for w in [2, 3] {
            let (spikes, rec) = trial(refr, w, &codes, &cue);
            println!("{:>12} {:>4} {:>8}  {:>26}", label, w, spikes, format!("{:?}", rec));
        }
    }

    println!("\n--- capacity at one-spike-per-unit, w=2 ---");
    let corpus: Vec<String> = (0..60)
        .map(|i| format!("grammar rule number {i} explains how clause {i} attaches to its head"))
        .collect();
    let many: Vec<Vec<NeuronId>> = corpus.iter().map(|t| encode(t)).collect();
    for load in [5usize, 20, 40, 60] {
        let subset = &many[..load];
        let cue: Vec<NeuronId> = subset[0].iter().take(10).copied().collect();
        let (_, rec) = trial(WINDOW + 1, 2, subset, &cue);
        let target = rec[0];
        let others: usize = rec[1..].iter().sum();
        let spurious = others as f64 / (load - 1) as f64;
        println!(
            "{load:>3} chunks stored -> recovered {target:>2}/40 of the cued chunk, \
             mean {spurious:>5.2} units leaked per other chunk"
        );
    }
}
