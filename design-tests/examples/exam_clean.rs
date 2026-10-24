// TESTING / DESIGN ONLY. Which of her conversation-exam exchanges did she also hear in training?
//
//   cargo run --release --example exam_clean -- <n> <out-mask.tsv>      (n = compose's first arg)
//
// The exam compose holds out is the first 1,200 short exchanges of the corpus (plus the long
// ones among them); she trains on the stretch after it, up to the n-th short exchange. The
// corpus is DailyDialog, which repeats itself: an exam exchange can sit, word for word, in the
// stretch she trains on, and then a right answer is memory, not generalisation.
//
// This reads the corpus exactly as compose does (same tokenizer, same short/long split, same
// held-out region; AVA_CORPUS and AVA_SHORT_ONLY are not supported and refused) and writes one
// line per held-out exchange, in compose's exam order:
//   index  domain  pair_heard  reply_heard  prompt_heard  said_heard  reply_tokens
// pair_heard   = the same (prompt, reply) is a training exchange
// reply_heard  = the same reply follows some prompt in training
// prompt_heard = the same prompt is a training prompt
// said_heard   = the reply was heard as a training PROMPT (she heard those words said, as a turn)
// "clean" = neither the pair nor the reply was heard; "strict" = nor said as a prompt. Nothing here changes how she is scored;
// it is the mask a later switch can use to report the clean part of the exam beside the whole.
use design_tests::seq;
use std::collections::HashSet;
use std::fs;

const CORPUS: &str = "../data/triples.tsv";
const TEST: usize = 1200;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 { eprintln!("usage: exam_clean <n> <out-mask.tsv>"); std::process::exit(1); }
    let n: usize = a[1].parse().expect("n must be a whole number");
    for v in ["AVA_CORPUS", "AVA_TEST", "AVA_SHORT_ONLY"] {
        if std::env::var(v).is_ok() { eprintln!("{v} is set: only the default exam is supported"); std::process::exit(1); }
    }
    let raw = fs::read_to_string(CORPUS).expect("read the corpus");
    // compose's reading loop, the dialogue-only case (triples.tsv has no domain tags)
    let (mut pairs, mut long) = (Vec::<(Vec<String>, Vec<String>)>::new(), Vec::<bool>::new());
    let (mut seen_short, mut n_region) = (0usize, 0usize);
    for line in raw.lines() {
        let mut p = line.split('\t');
        let (Some(f0), Some(q), Some(r)) = (p.next(), p.next(), p.next()) else { continue };
        if f0.starts_with('@') { eprintln!("tagged corpus lines are not supported"); std::process::exit(1); }
        let (qt, rt) = (seq::tokenize(q), seq::tokenize(r));
        let short = rt.len() <= 16 && qt.len() <= 20;
        if qt.is_empty() || rt.is_empty() { continue; }
        let held = seen_short < TEST;
        if held { n_region += 1; }
        if short { seen_short += 1; }
        pairs.push((qt, rt));
        long.push(!short);
        if seen_short >= TEST + n { break; }
    }
    let n_test = n_region.min(pairs.len().saturating_sub(1));
    let (test, train) = pairs.split_at(n_test);
    let heard_pair: HashSet<(&[String], &[String])> = train.iter().map(|(q, r)| (q.as_slice(), r.as_slice())).collect();
    let heard_reply: HashSet<&[String]> = train.iter().map(|(_, r)| r.as_slice()).collect();
    let heard_prompt: HashSet<&[String]> = train.iter().map(|(q, _)| q.as_slice()).collect();

    let mut out = String::from("index\tdomain\tpair_heard\treply_heard\tprompt_heard\tsaid_heard\treply_tokens\n");
    // [short, long] x [all, pair, reply, clean, strict], in exchanges and in next-word positions
    let (mut ex, mut pos) = ([[0usize; 5]; 2], [[0usize; 5]; 2]);
    for (i, (q, r)) in test.iter().enumerate() {
        let (hp, hr, hq) = (heard_pair.contains(&(q.as_slice(), r.as_slice())), heard_reply.contains(r.as_slice()), heard_prompt.contains(q.as_slice()));
        let hs = heard_prompt.contains(r.as_slice());
        let d = long[i] as usize;
        out.push_str(&format!("{i}\t{}\t{}\t{}\t{}\t{}\t{}\n", if long[i] { "dialogue long" } else { "dialogue" }, hp as u8, hr as u8, hq as u8, hs as u8, r.len()));
        // compose scores every reply word and the end of the reply: len + 1 positions
        for (k, on) in [true, hp, hr, !hp && !hr, !hp && !hr && !hs].into_iter().enumerate() {
            if on { ex[d][k] += 1; pos[d][k] += r.len() + 1; }
        }
    }
    fs::write(&a[2], out).expect("write the mask");
    println!("  corpus {CORPUS}: {} train / {} held out (n = {n})", train.len(), test.len());
    for (d, name) in ["dialogue", "dialogue long"].iter().enumerate() {
        let pc = |k: usize| 100.0 * ex[d][k] as f64 / ex[d][0].max(1) as f64;
        println!("  {name:<14} {:>5} exchanges: pair heard {:>5.1}%  reply heard {:>5.1}%  clean {:>5.1}% ({} exchanges, {} of {} positions)  strict {:>5.1}% ({}, {} positions)",
                 ex[d][0], pc(1), pc(2), pc(3), ex[d][3], pos[d][3], pos[d][0], pc(4), ex[d][4], pos[d][4]);
    }
    println!("  mask written to {}", a[2]);
    // the training stretch repeats itself too: how much of what she hears is a repeat?
    let mut seen: HashSet<(&[String], &[String])> = HashSet::new();
    let (mut rep, mut rep_pos, mut all_pos) = (0usize, 0usize, 0usize);
    let mut top: std::collections::HashMap<(&[String], &[String]), usize> = std::collections::HashMap::new();
    for (q, r) in train {
        all_pos += r.len() + 1;
        if !seen.insert((q.as_slice(), r.as_slice())) { rep += 1; rep_pos += r.len() + 1; *top.entry((q.as_slice(), r.as_slice())).or_default() += 1; }
    }
    println!("  training: {} exchanges, {} repeats of an earlier exchange ({:.1}%, {:.1}% of reply positions)",
             train.len(), rep, 100.0 * rep as f64 / train.len() as f64, 100.0 * rep_pos as f64 / all_pos as f64);
    let mut t: Vec<_> = top.into_iter().collect();
    t.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for ((q, r), n) in t.iter().take(5) { println!("    x{:<4} {} -> {}", n + 1, q.join(" "), r.join(" ")); }
}
