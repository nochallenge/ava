// design experiment 3. experiments 1 and 2 both failed for the SAME reason and it
// is not the refractory period. it is arithmetic.
//
//   a cued chunk completes only if   cue_size * w >= V_th
//   an uncued chunk stays silent if  max_overlap * w <  V_th
//
// so a usable weight exists IFF  max_overlap < cue_size, and then
//   V_th / cue_size  <=  w  <  V_th / max_overlap
//
// in experiment 2: max_overlap = 9, cue = 10, V_th = 15 -> w must be in [1.5, 1.67).
// weights were whole millivolts. there is no integer in that interval. that is the
// entire bug. this run gives the weight 10x finer resolution (V_th = 150) and checks
// the predicted window.
use design_tests::*;

const WINDOW: Tick = 50 * MS;
const V_TH: f64 = 150.0;

fn trial(w: i32, codes: &[Vec<NeuronId>], cue: &[NeuronId]) -> Vec<usize> {
    let mut brain = Brain::new(N, 10.0 * MS as f64, V_TH);
    brain.set_refractory(WINDOW + 1); // one spike per unit per presentation
    for (i, c) in codes.iter().enumerate() {
        install(&mut brain, i as u32, c, w, MS);
    }
    for &id in cue {
        brain.queue.push(Event { at: 0, target: id, weight_mv: 1000 });
    }
    let mut who: Vec<NeuronId> = brain.run(WINDOW, 100_000).into_iter().map(|(_, id)| id).collect();
    who.sort_unstable();
    who.dedup();
    codes.iter().map(|c| overlap(&who, c)).collect()
}

fn main() {
    let texts = [
        "the verb agrees with the subject in number and person",
        "a comma separates items in a list of three or more",
        "the past tense of irregular verbs must be memorised",
    ];
    let codes: Vec<Vec<NeuronId>> = texts.iter().map(|t| encode(t)).collect();
    let cue: Vec<NeuronId> = codes[0].iter().take(10).copied().collect();

    let max_ov = (1..codes.len()).map(|i| overlap(&codes[0], &codes[i])).max().unwrap();
    println!("k = 40, cue = {}, max overlap with a rival chunk = {max_ov}", cue.len());
    println!(
        "predicted window:  {:.1} <= w < {:.1}\n",
        V_TH / cue.len() as f64,
        V_TH / max_ov as f64
    );
    println!("{:>5}  {:>26}  {}", "w", "recovered [c0, c1, c2]", "verdict");
    for w in [10, 14, 15, 16, 17, 20, 30] {
        let rec = trial(w, &codes, &cue);
        let leak: usize = rec[1..].iter().sum();
        let verdict = match (rec[0] >= 38, leak <= 2 * max_ov) {
            (false, _) => "incomplete",
            (true, true) => "CLEAN",
            (true, false) => "leaks",
        };
        println!("{:>5}  {:>26}  {}", w, format!("{:?}", rec), verdict);
    }

    // capacity: how many chunks fit before a cue starts dragging in neighbours
    println!("\n--- capacity, w = 16, cue = 10 of 40 ---");
    let words = [
        "syntax", "clause", "vowel", "tense", "plural", "adverb", "pronoun", "gerund",
        "article", "subject", "object", "modal", "passive", "phoneme", "idiom", "prefix",
    ];
    let corpus: Vec<String> = (0..120)
        .map(|i| {
            let a = words[i % words.len()];
            let b = words[(i * 7 + 3) % words.len()];
            let c = words[(i * 11 + 5) % words.len()];
            format!("{a} {b} {c} lesson {i} covers {a} and how {b} relates to {c} in speech")
        })
        .collect();
    let many: Vec<Vec<NeuronId>> = corpus.iter().map(|t| encode(t)).collect();
    for load in [5usize, 15, 40, 80, 120] {
        let subset = &many[..load];
        let cue: Vec<NeuronId> = subset[0].iter().take(10).copied().collect();
        let rec = trial(16, subset, &cue);
        let leaked: usize = rec[1..].iter().sum();
        println!(
            "{load:>4} chunks -> cued chunk {:>2}/40, total units leaked into the other {} chunks: {leaked}",
            rec[0],
            load - 1
        );
    }
}
