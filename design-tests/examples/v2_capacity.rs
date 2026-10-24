// is the capacity collapse the architecture, or my degenerate test corpus?
// recall works iff max_pairwise_overlap < cue_size. so measure max overlap directly,
// no simulation needed, and find what N a real curriculum would need.
use design_tests::v2::*;

fn max_overlap(codes: &[Vec<NeuronId>]) -> (usize, f64) {
    let mut mx = 0usize;
    let mut tot = 0f64;
    let mut n = 0f64;
    for i in 0..codes.len() {
        for j in i + 1..codes.len() {
            let o = overlap(&codes[i], &codes[j]);
            mx = mx.max(o);
            tot += o as f64;
            n += 1.0;
        }
    }
    (mx, if n > 0.0 { tot / n } else { 0.0 })
}

/// degenerate: 24 content words recombined. sentences really are near-duplicates.
fn template_corpus(c: usize) -> Vec<String> {
    let w = ["syntax","clause","vowel","tense","plural","adverb","pronoun","gerund",
             "article","subject","object","modal","passive","phoneme","idiom","prefix",
             "suffix","stress","schwa","accent","dative","aspect","mood","voice"];
    (0..c).map(|i| format!("lesson {i} explains {} before {} because {} governs {}",
        w[i % 24], w[(i * 5 + 7) % 24], w[(i * 11 + 3) % 24], w[(i * 17 + 13) % 24])).collect()
}

/// diverse: every sentence has its own vocabulary, like a real curriculum would.
fn diverse_corpus(c: usize) -> Vec<String> {
    (0..c).map(|i| {
        let a = fnv1a(&i.to_le_bytes(), 1);
        let b = fnv1a(&i.to_le_bytes(), 2);
        let d = fnv1a(&i.to_le_bytes(), 3);
        format!("rule{i} states that {} forms combine with {} endings unless {} applies",
            word(a), word(b), word(d))
    }).collect()
}

fn word(h: u64) -> String {
    let c = ["b","d","f","g","k","l","m","n","p","r","s","t","v","z"];
    let v = ["a","e","i","o","u","ai","ee","ou"];
    format!("{}{}{}{}{}", c[(h % 14) as usize], v[(h / 14 % 8) as usize],
        c[(h / 112 % 14) as usize], v[(h / 1568 % 8) as usize], c[(h / 12544 % 14) as usize])
}

fn main() {
    println!("recall is clean only while  max_overlap < cue_size.  cue = 10 here.\n");

    for (name, gen) in [("template (24 words recycled)", template_corpus as fn(usize) -> Vec<String>),
                        ("diverse (unique vocabulary)", diverse_corpus as fn(usize) -> Vec<String>)] {
        println!("=== {name}, N=2000 k=40 ===");
        println!("{:>7} {:>12} {:>12} {}", "chunks", "max overlap", "mean", "usable at cue=10");
        for &c in &[10usize, 50, 100, 200, 400] {
            let corpus = gen(c);
            let codes: Vec<Vec<NeuronId>> = corpus.iter().map(|t| encode_in(t, 2000, 40)).collect();
            let (mx, mean) = max_overlap(&codes);
            println!("{:>7} {:>12} {:>12.2} {}", c, mx, mean, if mx < 10 { "yes" } else { "NO" });
        }
        println!();
    }

    println!("=== how big does N need to be? diverse corpus, k=40, cue=10 ===");
    println!("{:>7} {:>8} {:>12} {:>10} {}", "chunks", "N", "max overlap", "k^2/N", "usable");
    for &c in &[100usize, 500, 2000] {
        let corpus = diverse_corpus(c);
        for &n in &[2_000usize, 10_000, 50_000, 200_000] {
            let codes: Vec<Vec<NeuronId>> = corpus.iter().map(|t| encode_in(t, n, 40)).collect();
            let (mx, _) = max_overlap(&codes);
            println!("{:>7} {:>8} {:>12} {:>10.3} {}", c, n, mx, 1600.0 / n as f64,
                if mx < 10 { "yes" } else { "NO" });
        }
        println!();
    }

    println!("=== synapse cost, all-to-all within a chunk ===");
    println!("{:>8} {:>6} {:>16} {:>14}", "chunks", "k", "synapses", "RAM at 16B ea");
    for &(c, k) in &[(1_000usize, 40usize), (50_000, 40), (50_000, 100), (500_000, 40)] {
        let syn = c as u64 * (k as u64) * (k as u64 - 1);
        println!("{:>8} {:>6} {:>16} {:>13.1}GB", c, k, syn, syn as f64 * 16.0 / 1e9);
    }
}
