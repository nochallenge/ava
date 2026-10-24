// design experiment 5. checking my own top_k for a bug before trusting experiment 4's
// capacity numbers. top_k breaks score ties with `.then(a.cmp(&b))` -- lowest INDEX
// wins. if most slots are tied (which happens as soon as a chunk has many features
// and few hash collisions), the code is not "the top 40 units", it is "the 40
// lowest-numbered units that got hit at all". every chunk would then pile onto the
// same low indices and overlap would be an artifact of the tie-break, not of meaning.
use design_tests::*;

fn score_of(chunk: &str) -> Vec<u32> {
    let mut score = vec![0u32; N];
    for word in chunk.split_whitespace() {
        let w = normalize(word);
        if w.is_empty() {
            continue;
        }
        for i in 0..FANOUT {
            score[(fnv1a(w.as_bytes(), i as u64) % N as u64) as usize] += 1;
        }
    }
    score
}

fn main() {
    let short = "verbs agree with subjects";
    let long = "lesson 42 covers tense showing how pronoun relates to phoneme when speaking";

    for (name, text) in [("short (4 content words)", short), ("long (11 words)", long)] {
        let s = score_of(text);
        let hit: Vec<u32> = s.iter().copied().filter(|&v| v > 0).collect();
        let ones = hit.iter().filter(|&&v| v == 1).count();
        let code = top_k(&s, K);
        let mean_idx = code.iter().sum::<usize>() as f64 / code.len() as f64;
        let cut = s[*code.last().unwrap()];
        println!("{name}");
        println!("  slots hit: {}, of which score==1: {ones}", hit.len());
        println!("  score at the k-th place (the cut): {cut}");
        println!("  mean index of chosen units: {mean_idx:.0}  (unbiased would be ~{})", N / 2);
        println!(
            "  ties AT the cut: {} slots share score {cut}\n",
            s.iter().filter(|&&v| v == cut).count()
        );
    }

    // how many features can a k-of-n code actually carry before it saturates?
    println!("features vs how many of them survive into the code (k={K}, fanout={FANOUT}):");
    for nfeat in [2usize, 4, 5, 6, 8, 12, 20] {
        let text: Vec<String> = (0..nfeat).map(|i| format!("feature{i}")).collect();
        let joined = text.join(" ");
        let s = score_of(&joined);
        let code = top_k(&s, K);
        let mean_idx = code.iter().sum::<usize>() as f64 / code.len() as f64;
        let cut = s[*code.last().unwrap()];
        let tied_at_cut = s.iter().filter(|&&v| v == cut).count();
        println!(
            "  {nfeat:>3} features -> {:>4} slots hit, cut score {cut}, {tied_at_cut:>4} tied at the cut, mean idx {mean_idx:>4.0}",
            s.iter().filter(|&&v| v > 0).count()
        );
    }
}
