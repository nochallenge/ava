// Previous run showed mean overlap 15/40 even at N=200,000, which is impossible for
// random codes (expected k^2/N = 0.008). Diagnosis: my "diverse" corpus wasn't -
// every sentence shared states/forms/combine/endings/applies. With k/FANOUT ~ 5
// feature slots, the SHARED words filled the code and the distinctive ones were
// dropped.
//
// That is the real lesson. Stopword removal is a crude hand-written version of the
// right idea. The principled version is IDF: weight every feature by how rare it is
// across the corpus, so a code is built from what makes a chunk DIFFERENT.
use design_tests::v2::*;
use std::collections::HashMap;

fn max_mean_overlap(codes: &[Vec<NeuronId>]) -> (usize, f64) {
    let (mut mx, mut tot, mut n) = (0usize, 0f64, 0f64);
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

/// control: every sentence entirely unique tokens. proves the hash itself is fine.
fn truly_unique(c: usize) -> Vec<String> {
    (0..c).map(|i| (0..8).map(|j| format!("w{i}x{j}")).collect::<Vec<_>>().join(" ")).collect()
}

/// realistic: a shared frame plus a few distinctive words, like real curriculum text.
fn realistic(c: usize) -> Vec<String> {
    let topic = ["plural","tense","article","pronoun","adverb","clause","vowel","gerund",
                 "modal","passive","prefix","suffix","stress","idiom","dative","aspect"];
    (0..c).map(|i| {
        format!("rule {i} states that the {} form of {} changes when {} follows",
            topic[i % 16], token(i, 1), token(i, 2))
    }).collect()
}

fn token(i: usize, s: u64) -> String {
    let h = fnv1a(&i.to_le_bytes(), s);
    let c = ["b","d","f","g","k","l","m","n","p","r","s","t","v","z"];
    let v = ["a","e","i","o","u"];
    format!("{}{}{}{}{}", c[(h % 14) as usize], v[(h / 14 % 5) as usize],
        c[(h / 70 % 14) as usize], v[(h / 980 % 5) as usize], c[(h / 4900 % 14) as usize])
}

/// IDF over the corpus. rare features get big scores, ubiquitous ones get ~0.
fn idf_table(corpus: &[String]) -> HashMap<String, f64> {
    let n = corpus.len() as f64;
    let mut df: HashMap<String, f64> = HashMap::new();
    for text in corpus {
        let mut seen: Vec<String> = features(text);
        seen.sort();
        seen.dedup();
        for f in seen {
            *df.entry(f).or_insert(0.0) += 1.0;
        }
    }
    df.into_iter().map(|(k, d)| (k, (n / d).ln().max(0.0))).collect()
}

fn encode_idf(text: &str, idf: &HashMap<String, f64>, n: usize, k: usize) -> Vec<NeuronId> {
    let mut score = vec![0f64; n];
    for f in features(text) {
        let w = *idf.get(&f).unwrap_or(&1.0);
        if w <= 0.0 { continue; }
        for i in 0..FANOUT {
            score[(fnv1a(f.as_bytes(), i as u64) % n as u64) as usize] += w;
        }
    }
    // top-k on f64 scores. f64 is not Ord, so total_cmp - the section 5.2 escape hatch.
    let mut idx: Vec<usize> = (0..n).filter(|&i| score[i] > 0.0).collect();
    idx.sort_unstable_by(|&a, &b| {
        score[b].total_cmp(&score[a])
            .then_with(|| fnv1a(&a.to_le_bytes(), 0xA11).cmp(&fnv1a(&b.to_le_bytes(), 0xA11)))
    });
    idx.truncate(k);
    idx.sort_unstable();
    idx
}

fn main() {
    println!("clean recall needs max_overlap < cue_size (cue = 10)\n");

    println!("=== control: sentences with ZERO shared words, N=2000 k=40 ===");
    for &c in &[100usize, 400] {
        let corpus = truly_unique(c);
        let codes: Vec<Vec<NeuronId>> = corpus.iter().map(|t| encode_in(t, 2000, 40)).collect();
        let (mx, mean) = max_mean_overlap(&codes);
        println!("  {c:>4} chunks -> max {mx:>3}, mean {mean:>5.2}   (k^2/N predicts {:.2})", 1600.0 / 2000.0);
    }

    println!("\n=== realistic corpus: shared frame + distinctive words ===");
    println!("{:>7} {:>8} {:>6}  {:>11} {:>11}  {:>11} {:>11}", "chunks", "N", "k", "plain max", "plain mean", "IDF max", "IDF mean");
    for &c in &[100usize, 400] {
        let corpus = realistic(c);
        let idf = idf_table(&corpus);
        for &(n, k) in &[(2_000usize, 40usize), (20_000, 40), (20_000, 100)] {
            let plain: Vec<Vec<NeuronId>> = corpus.iter().map(|t| encode_in(t, n, k)).collect();
            let widf: Vec<Vec<NeuronId>> = corpus.iter().map(|t| encode_idf(t, &idf, n, k)).collect();
            let (pm, pmean) = max_mean_overlap(&plain);
            let (im, imean) = max_mean_overlap(&widf);
            println!("{:>7} {:>8} {:>6}  {:>11} {:>11.2} {:>11} {:>11.2}  {}",
                c, n, k, pm, pmean, im, imean, if im < 10 { "IDF usable" } else { "" });
        }
    }

    println!("\n=== the feature-slot ceiling: k / FANOUT ===");
    println!("FANOUT is currently {FANOUT}, so a k=40 code has room for ~{} features", 40 / FANOUT);
    let corpus = realistic(400);
    let idf = idf_table(&corpus);
    let nfeat = features(&corpus[0]).len();
    println!("but a realistic sentence produces {nfeat} features. the excess is dropped.");
    println!("\n{:>8} {:>8} {:>11} {:>11}", "FANOUT", "slots", "IDF max", "IDF mean");
    for fanout_k in [(40usize, 4usize), (40, 8), (100, 4), (100, 10)] {
        let (k, _f) = fanout_k;
        // emulate a different fanout by scaling k, since FANOUT is a const
        let widf: Vec<Vec<NeuronId>> = corpus.iter().map(|t| encode_idf(t, &idf, 20_000, k)).collect();
        let (im, imean) = max_mean_overlap(&widf);
        println!("{:>8} {:>8} {:>11} {:>11.2}", FANOUT, k / FANOUT, im, imean);
    }
}
