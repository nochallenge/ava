// design experiment 4. experiment 3 showed the analytic window
//   V_th/cue <= w < V_th/max_overlap
// is NECESSARY but not SUFFICIENT: with cue=10 and max_overlap=9 the interval is
// almost empty, and cascades through a third chunk close it completely.
//
// two levers actually widen it:
//   1. cue with more of the pattern (raises the lower bound's denominator)
//   2. lower the overlap itself -- that is PATTERN SEPARATION, the job the dentate
//      gyrus does in the hippocampus before anything reaches CA3.
// this run measures both.
use design_tests::*;

const WINDOW: Tick = 50 * MS;
const V_TH: f64 = 150.0;

const STOP: &[&str] = &[
    "the", "a", "an", "of", "in", "to", "and", "or", "is", "it", "its", "on", "with", "for",
    "that", "this", "as", "at", "by", "be", "are", "was", "must", "how", "more", "than",
];

/// same encoder as the library, plus stopword removal and word-pair features.
/// both are pattern separation: they make two similar sentences land on more
/// distinguishable unit sets.
fn encode_separated(chunk: &str, bigrams: bool) -> Vec<NeuronId> {
    let mut score = vec![0u32; N];
    let kept: Vec<String> = chunk
        .split_whitespace()
        .map(normalize)
        .filter(|w| !w.is_empty() && !STOP.contains(&w.as_str()))
        .collect();
    let mut feats: Vec<String> = kept.clone();
    if bigrams {
        for pair in kept.windows(2) {
            feats.push(format!("{}_{}", pair[0], pair[1]));
        }
    }
    for f in &feats {
        for i in 0..FANOUT {
            score[(fnv1a(f.as_bytes(), i as u64) % N as u64) as usize] += 1;
        }
    }
    top_k(&score, K)
}

fn max_pairwise_overlap(codes: &[Vec<NeuronId>]) -> usize {
    let mut m = 0;
    for i in 0..codes.len() {
        for j in i + 1..codes.len() {
            m = m.max(overlap(&codes[i], &codes[j]));
        }
    }
    m
}

fn trial(w: i32, codes: &[Vec<NeuronId>], cue: &[NeuronId]) -> Vec<usize> {
    let mut brain = Brain::new(N, 10.0 * MS as f64, V_TH);
    brain.set_refractory(WINDOW + 1);
    for (i, c) in codes.iter().enumerate() {
        install(&mut brain, i as u32, c, w, MS);
    }
    for &id in cue {
        brain.queue.push(Event { at: 0, target: id, weight_mv: 1000 });
    }
    let mut who: Vec<NeuronId> = brain.run(WINDOW, 200_000).into_iter().map(|(_, id)| id).collect();
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

    println!("=== lever 2 first: does separation actually lower overlap? ===");
    let plain: Vec<Vec<NeuronId>> = texts.iter().map(|t| encode(t)).collect();
    let nostop: Vec<Vec<NeuronId>> = texts.iter().map(|t| encode_separated(t, false)).collect();
    let bigram: Vec<Vec<NeuronId>> = texts.iter().map(|t| encode_separated(t, true)).collect();
    println!("max pairwise overlap, plain unigram encoder : {}", max_pairwise_overlap(&plain));
    println!("max pairwise overlap, stopwords removed     : {}", max_pairwise_overlap(&nostop));
    println!("max pairwise overlap, + word pairs          : {}", max_pairwise_overlap(&bigram));

    for (name, codes) in [("plain", &plain), ("separated", &bigram)] {
        let mo = max_pairwise_overlap(codes);
        println!("\n=== {name} encoder, max overlap {mo} ===");
        println!("{:>4} {:>4}  {:>26}  {}", "cue", "w", "recovered [c0, c1, c2]", "verdict");
        for cue_n in [10usize, 16, 20, 28] {
            let lo = (V_TH / cue_n as f64).ceil() as i32;
            let hi = (V_TH / mo.max(1) as f64).floor() as i32;
            if lo > hi {
                println!("{:>4} {:>4}  {:>26}  window empty ({lo} > {hi})", cue_n, "-", "-");
                continue;
            }
            let w = (lo + hi) / 2;
            let cue: Vec<NeuronId> = codes[0].iter().take(cue_n).copied().collect();
            let rec = trial(w, codes, &cue);
            let leak: usize = rec[1..].iter().sum();
            let verdict = match (rec[0] >= 38, leak <= mo) {
                (false, _) => "incomplete",
                (true, true) => "CLEAN",
                (true, false) => "leaks",
            };
            println!("{:>4} {:>4}  {:>26}  {}", cue_n, w, format!("{:?}", rec), verdict);
        }
    }

    println!("\n=== capacity with the separated encoder, cue = half the pattern ===");
    let words = [
        "syntax", "clause", "vowel", "tense", "plural", "adverb", "pronoun", "gerund",
        "article", "subject", "object", "modal", "passive", "phoneme", "idiom", "prefix",
    ];
    let corpus: Vec<String> = (0..200)
        .map(|i| {
            let a = words[i % words.len()];
            let b = words[(i * 7 + 3) % words.len()];
            let c = words[(i * 11 + 5) % words.len()];
            format!("lesson {i} covers {a} showing how {b} relates to {c} when speaking")
        })
        .collect();
    let many: Vec<Vec<NeuronId>> = corpus.iter().map(|t| encode_separated(t, true)).collect();
    for load in [5usize, 20, 50, 100, 200] {
        let subset = &many[..load].to_vec();
        let mo = max_pairwise_overlap(subset);
        let cue: Vec<NeuronId> = subset[0].iter().take(20).copied().collect();
        let lo = (V_TH / 20.0).ceil() as i32;
        let hi = (V_TH / mo.max(1) as f64).floor() as i32;
        let w = if lo > hi { lo } else { (lo + hi) / 2 };
        let rec = trial(w, subset, &cue);
        let leaked: usize = rec[1..].iter().sum();
        println!(
            "{load:>4} chunks | max overlap {mo:>2} | w={w:>2} | cued chunk {:>2}/40 | \
             leaked {leaked:>5} units across {} rivals",
            rec[0],
            load - 1
        );
    }
}
