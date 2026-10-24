use design_tests::*;

fn score_of(chunk: &str, fanout: usize) -> Vec<u32> {
    let mut s = vec![0u32; N];
    for word in chunk.split_whitespace() {
        let w = normalize(word);
        if w.is_empty() { continue; }
        for i in 0..fanout {
            s[(fnv1a(w.as_bytes(), i as u64) % N as u64) as usize] += 1;
        }
    }
    s
}

fn mean_idx(c: &[NeuronId]) -> f64 { c.iter().sum::<usize>() as f64 / c.len() as f64 }

fn main() {
    println!("mean unit index of the chosen code (unbiased target ~{})\n", N / 2);
    println!("{:>10} {:>14} {:>14}", "features", "top_k (index)", "top_k_hashed");
    for nfeat in [4usize, 6, 8, 12, 20] {
        let text: Vec<String> = (0..nfeat).map(|i| format!("feature{i}")).collect();
        let s = score_of(&text.join(" "), FANOUT);
        println!("{:>10} {:>14.0} {:>14.0}", nfeat, mean_idx(&top_k(&s, K)), mean_idx(&top_k_hashed(&s, K)));
    }

    println!("\nmean pairwise overlap over 200 distinct sentences (k={K}):");
    let words = ["syntax","clause","vowel","tense","plural","adverb","pronoun","gerund",
                 "article","subject","object","modal","passive","phoneme","idiom","prefix"];
    let corpus: Vec<String> = (0..200).map(|i| {
        format!("lesson {i} covers {} showing how {} relates to {}",
            words[i % 16], words[(i*7+3) % 16], words[(i*11+5) % 16])
    }).collect();
    for fanout in [4usize, 8, 16] {
        let a: Vec<Vec<NeuronId>> = corpus.iter().map(|t| top_k(&score_of(t, fanout), K)).collect();
        let b: Vec<Vec<NeuronId>> = corpus.iter().map(|t| top_k_hashed(&score_of(t, fanout), K)).collect();
        let mo = |c: &Vec<Vec<NeuronId>>| {
            let mut tot = 0usize; let mut n = 0usize; let mut mx = 0usize;
            for i in 0..c.len() { for j in i+1..c.len() {
                let o = overlap(&c[i], &c[j]); tot += o; n += 1; mx = mx.max(o);
            }}
            (tot as f64 / n as f64, mx)
        };
        let (ma, xa) = mo(&a); let (mb, xb) = mo(&b);
        println!("  fanout {fanout:>2}: index tie-break mean {ma:>5.2} max {xa:>2}  |  hashed tie-break mean {mb:>5.2} max {xb:>2}");
    }
}
