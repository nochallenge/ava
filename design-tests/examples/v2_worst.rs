// worstRival was 40.0 while meanRival was 0.08. Exactly one rival fires completely.
// Which one, and why?
use design_tests::v2::*;

const POP: usize = 20_000;
const KK: usize = 40;

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

fn main() {
    let texts = corpus(200);
    let enc = Encoder::fit(&texts, POP, KK);
    let codes: Vec<Vec<NeuronId>> = texts.iter().map(|t| enc.encode(t)).collect();

    // how similar is each rival's CODE to chunk 0's?
    let mut sims: Vec<(usize, usize)> = (1..codes.len())
        .map(|i| (overlap(&codes[0], &codes[i]), i)).collect();
    sims.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    println!("chunk 0: {}", texts[0]);
    println!("  code size {}\n", codes[0].len());
    println!("most code-similar rivals:");
    for &(o, i) in sims.iter().take(5) {
        println!("  overlap {o:>2} with chunk {i:>3}: {}", texts[i]);
    }

    // are any two source sentences literally identical?
    let mut dupes = 0;
    for i in 0..texts.len() {
        for j in i + 1..texts.len() {
            if texts[i] == texts[j] { dupes += 1; }
        }
    }
    println!("\nidentical source sentences in the corpus: {dupes}");

    // how many rivals share ANY unit with chunk 0's cue?
    let cue_n = 30;
    let seed: Vec<NeuronId> = codes[0].iter().take(cue_n).copied().collect();
    let touched = (1..codes.len()).filter(|&i| overlap(&seed, &codes[i]) > 0).count();
    println!("rivals sharing >=1 unit with the {cue_n}-unit cue: {touched} of {}", codes.len() - 1);

    // simulate and find who actually fired completely
    let mut b = Brain::new(POP, Inhibition::new(15.0 * MS as f64, 1.0, 3.0), 2 * MS);
    for (i, c) in codes.iter().enumerate() { install_chunk_drive(&mut b, i as u32, c, cue_n, 1.8); }
    cue(&mut b, &seed, 0);
    let spikes = b.run(400 * MS, 500_000);
    let mut who: Vec<NeuronId> = spikes.iter().map(|&(_, id)| id).collect();
    who.sort_unstable(); who.dedup();
    println!("\ntotal distinct units that fired: {} (cued chunk is {} units)", who.len(), KK);
    let mut hits: Vec<(usize, usize)> = (1..codes.len())
        .map(|i| (overlap(&who, &codes[i]), i)).collect();
    hits.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    println!("rivals that fired most:");
    for &(h, i) in hits.iter().take(5) {
        println!("  {h:>2}/40 units of chunk {i:>3} fired (code overlap with chunk 0 = {})",
            overlap(&codes[0], &codes[i]));
    }
    let fully = hits.iter().filter(|&&(h, _)| h >= 35).count();
    println!("\nrivals that fired essentially completely: {fully}");
}
