// TESTING / DESIGN ONLY. (Rust since v27, replacing hippo/ceiling.py line for line.)
// Two CEILINGS for Ava's planned hippocampus, measured offline on her own data.
//
//   cargo run --release --example ceiling -- <path/to/triples.tsv>
//
// 1. CONVERSATION-MEMORY CEILING. On her 1200 test exchanges, every reply position (each
//    reply token plus one end-of-reply) is asked: is the target word (a) in the previous turn
//    but NOT in the prompt (only a memory of the conversation could supply it), (b) in the
//    prompt, (c) in neither? This bounds what a previous-turn memory could ever add.
//
// 2. NOVELTY CEILING ("have I seen these words together?"). Familiarity of a prompt = the
//    share of its content-word pairs (and, separately, of its adjacent bigrams) that occur in
//    her 30000 training prompts. AUC of familiarity, normal vs the v23 exam's REVERSED and
//    MIXED-UP prompts (built exactly as compose.rs builds them).
//
// Nothing here constrains Ava; it only measures the data. Her own tokenizer is used
// (design_tests::seq::tokenize), so the old "could Python and Rust tokenize differently"
// worry is gone; the count of such characters is still printed for comparison with old runs.
#[path = "common/school_util.rs"]
mod util;
use design_tests::seq::{tokenize as seq_tok, EOS};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use util::*;

const TEST_N: usize = 1200;
const TRAIN_N: usize = 30_000;

type Trip = (String, String, String);

/// the same pairs compose.rs reads from triples.tsv (same filter, same order), plus field 0,
/// the previous turn
fn dialogue3(text: &str) -> (Vec<Trip>, Vec<Trip>) {
    let mut trip = Vec::new();
    for line in splitlines(text) {
        let p: Vec<&str> = line.split('\t').collect();
        if p.len() < 3 { continue; }
        let (qt, rt) = (seq_tok(p[1]), seq_tok(p[2]));
        if !qt.is_empty() && !rt.is_empty() && rt.len() <= 16 && qt.len() <= 20 {
            trip.push((p[0].to_string(), p[1].to_string(), p[2].to_string()));
            if trip.len() >= TEST_N + TRAIN_N { break; }
        }
    }
    let train = trip.split_off(TEST_N.min(trip.len()));
    (trip, train)
}

/// cross-check: compose.rs reads with read_to_string + str::lines (split on '\n', one trailing
/// '\r' stripped, no newline translation). The universal-newline read + splitlines above
/// splits on more characters; this confirms the two agree on the kept block.
fn rust_lines(raw: &str, want: usize) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for line in raw.lines() {
        let p: Vec<&str> = line.split('\t').collect();
        if p.len() < 3 { continue; }
        let (qt, rt) = (seq_tok(p[1]), seq_tok(p[2]));
        if !qt.is_empty() && !rt.is_empty() && rt.len() <= 16 && qt.len() <= 20 {
            pairs.push((p[1].to_string(), p[2].to_string()));
            if pairs.len() >= want { break; }
        }
    }
    pairs
}

/// port of compose.rs auc(): rank-based, ties averaged; true = positive (normal)
fn auc(mut v: Vec<(f64, bool)>) -> f64 {
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (mut i, mut s, mut pos) = (0, 0.0, 0usize);
    while i < v.len() {
        let mut j = i;
        while j < v.len() && v[j].0 == v[i].0 { j += 1; }
        let avg = (i + j + 1) as f64 / 2.0;
        for k in i..j { if v[k].1 { s += avg; pos += 1; } }
        i = j;
    }
    let neg = v.len() - pos;
    if pos == 0 || neg == 0 { return 0.5; }
    (s - (pos * (pos + 1)) as f64 / 2.0) / (pos * neg) as f64
}

fn pct(a: usize, n: usize) -> String {
    if n > 0 { format!("{:6.2}%", 100.0 * a as f64 / n as f64) } else { "    - ".into() }
}

/// a non-ASCII letter, number or mark: where a Python tokenizer could have split differently
/// (approximate: Rust's alphanumeric plus the common combining-mark blocks)
fn risky(c: char) -> bool {
    let mark = matches!(c as u32, 0x300..=0x36F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F);
    (c as u32) > 127 && (c.is_alphanumeric() || mark)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 { eprintln!("usage: ceiling <path/to/triples.tsv>"); std::process::exit(1); }
    let data = Path::new(&a[1]);
    let shown = if cfg!(windows) { data.display().to_string().replace('/', "\\") } else { data.display().to_string() };
    // ------------------------------------------------------------------ data, as Ava reads it
    let (test3, train3) = dialogue3(&read_text(data).expect("read triples"));
    let raw = String::from_utf8(std::fs::read(data).expect("read triples")).expect("triples is UTF-8");
    let rl = rust_lines(&raw, TEST_N + TRAIN_N);
    let rust_same = rl.len() == test3.len() + train3.len()
        && rl.iter().zip(test3.iter().chain(&train3)).all(|(x, t)| x.0 == t.1 && x.1 == t.2);
    assert!(test3.len() == TEST_N && train3.len() == TRAIN_N, "({}, {})", test3.len(), train3.len());

    let test: Vec<(Vec<String>, Vec<String>, Vec<String>)> = test3.iter().map(|(v, q, r)| (seq_tok(v), seq_tok(q), seq_tok(r))).collect();
    let train: Vec<(Vec<String>, Vec<String>)> = train3.iter().map(|(_, q, r)| (seq_tok(q), seq_tok(r))).collect();
    let all3 = || test3.iter().chain(&train3).flat_map(|(v, q, r)| [v, q, r]);
    let nonascii = all3().filter(|s| !s.is_ascii()).count();
    let risky_n = all3().flat_map(|s| s.chars()).filter(|&c| risky(c)).collect::<HashSet<char>>().len();

    // the top-N training tokens exactly as compose.rs builds `common`: prompt + reply tokens of
    // the training exchanges, count descending, ties by spelling
    let mut f: HashMap<&str, usize> = HashMap::new();
    for (q, r) in &train { for w in q.iter().chain(r) { *f.entry(w).or_default() += 1; } }
    let mut ranked: Vec<(&str, usize)> = f.iter().map(|(w, c)| (*w, *c)).collect();
    ranked.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(y.0)));
    let top200: HashSet<&str> = ranked[..200].iter().map(|x| x.0).collect();
    let top1000: HashSet<&str> = ranked[..1000].iter().map(|x| x.0).collect();

    println!("CEILINGS FOR THE HIPPOCAMPUS  (offline, her own data)");
    println!("  file: {shown}");
    println!("  test exchanges {}, training exchanges {}; rust str::lines read gives the same pairs: {}",
        test.len(), train.len(), if rust_same { "True" } else { "False" });
    println!("  non-ASCII fields in the block: {nonascii}; non-ASCII letters/marks/numbers among them (where Python and Rust tokenizers could differ): {risky_n}");
    println!("  training vocabulary {} types; #200 = {} ({}), #1000 = {} ({})",
        f.len(), py_repr(ranked[199].0), ranked[199].1, py_repr(ranked[999].0), ranked[999].1);
    println!("  test exchanges with an empty previous turn: {}", test.iter().filter(|t| t.0.is_empty()).count());

    // ------------------------------------------------------------------ 1. conversation memory
    let groups: Vec<(&str, Box<dyn Fn(&str) -> bool>)> = vec![
        ("all positions", Box::new(|_| true)),
        ("content words (not top-200)", Box::new(|w| w != EOS && !top200.contains(w))),
        ("function words (top-200)", Box::new(|w| top200.contains(w))),
        ("end-of-reply", Box::new(|w| w == EOS)),
        ("content + end (compose.rs 'rare')", Box::new(|w| !top200.contains(w))),
        ("rare words (not top-1000)", Box::new(|w| w != EOS && !top1000.contains(w))),
        ("rare alphabetic (not top-1000)", Box::new(|w| w != EOS && !top1000.contains(w) && w.chars().any(|c| c.is_alphabetic()))),
    ];
    let mut cnt = vec![[0usize; 3]; groups.len()]; // a, b, c
    for (v, q, r) in &test {
        let (qs, vs): (HashSet<&str>, HashSet<&str>) = (q.iter().map(|x| x.as_str()).collect(), v.iter().map(|x| x.as_str()).collect());
        for w in r.iter().map(|x| x.as_str()).chain([EOS]) {
            let cat = if qs.contains(w) { 1 } else if vs.contains(w) { 0 } else { 2 };
            for (g, (_, keep)) in groups.iter().enumerate() { if keep(w) { cnt[g][cat] += 1; } }
        }
    }
    println!("\n1. CONVERSATION-MEMORY CEILING  (target word of each reply position)");
    println!("   (a) previous turn only = only a conversation memory could supply it");
    println!("   {:<36} {:>6} {:>14} {:>14} {:>12} {:>9}   counts a/b/c", "positions", "n", "(a) prev only", "(b) in prompt", "(c) neither", "a/(a+c)");
    for (g, (name, _)) in groups.iter().enumerate() {
        let [ca, cb, cc] = cnt[g];
        let n = ca + cb + cc;
        println!("   {:<36} {:>6} {:>14} {:>14} {:>12} {:>9}   {}/{}/{}", name, n, pct(ca, n), pct(cb, n), pct(cc, n), pct(ca, ca + cc), ca, cb, cc);
    }
    println!("   a/(a+c) = of the targets NOT in the prompt, the share the previous turn holds.");
    // how often the previous turn holds the target at all (prompt or not), for scale
    let (mut any_n, mut in_prev) = (0, 0);
    for (v, _, r) in &test {
        let vs: HashSet<&str> = v.iter().map(|x| x.as_str()).collect();
        for w in r { if !top200.contains(w.as_str()) { any_n += 1; if vs.contains(w.as_str()) { in_prev += 1; } } }
    }
    println!("   content targets present in the previous turn at all (prompt or not): {in_prev} / {any_n} = {}", pct(in_prev, any_n).trim());

    // ------------------------------------------------------------------ 2. novelty
    let mut ids: HashMap<String, u64> = HashMap::new();
    let mut wid = |w: &str| -> u64 { let n = ids.len() as u64; *ids.entry(w.to_string()).or_insert(n) };
    const B: u64 = 1 << 21; // pair key = lo * B + hi; vocab is far below 2^21 (checked below)
    let (mut pairs_seen, mut bigrams_seen, mut train_prompts) = (HashSet::new(), HashSet::new(), HashSet::new());
    let cwords = |q: &[String], wid: &mut dyn FnMut(&str) -> u64| -> Vec<u64> {
        let mut cw: Vec<u64> = q.iter().filter(|w| !top200.contains(w.as_str())).map(|w| wid(w)).collect();
        cw.sort_unstable();
        cw.dedup();
        cw
    };
    for (q, _) in &train {
        let x: Vec<u64> = q.iter().map(|w| wid(w)).collect();
        train_prompts.insert(q.clone());
        let cw = cwords(q, &mut wid);
        for i in 0..cw.len() { for j in i + 1..cw.len() { pairs_seen.insert(cw[i] * B + cw[j]); } }
        for k in 0..x.len().saturating_sub(1) { bigrams_seen.insert(x[k] * B + x[k + 1]); }
    }
    // (pair familiarity, bigram familiarity, had pairs?, had bigrams?)
    let mut fam = |q: &[String]| -> (f64, f64, bool, bool) {
        let cw = cwords(q, &mut wid);
        let tot = cw.len() * cw.len().saturating_sub(1) / 2;
        let fp = if tot > 0 {
            let mut seen = 0;
            for i in 0..cw.len() { for j in i + 1..cw.len() { if pairs_seen.contains(&(cw[i] * B + cw[j])) { seen += 1; } } }
            seen as f64 / tot as f64
        } else { 1.0 };
        let x: Vec<u64> = q.iter().map(|w| wid(w)).collect();
        let fb = if x.len() >= 2 {
            (0..x.len() - 1).filter(|&k| bigrams_seen.contains(&(x[k] * B + x[k + 1]))).count() as f64 / (x.len() - 1) as f64
        } else { 1.0 };
        (fp, fb, tot > 0, x.len() >= 2)
    };

    let tq: Vec<Vec<String>> = test.iter().map(|t| t.1.clone()).collect();
    let flat: Vec<&String> = tq.iter().flatten().collect();
    let versions: Vec<(&str, Vec<Vec<String>>)> = vec![
        ("normal", tq.clone()),
        ("reversed", tq.iter().map(|q| q.iter().rev().cloned().collect()).collect()),
        ("mixed-up", tq.iter().enumerate().map(|(i, q)| (0..q.len()).map(|j| flat[(i * 7919 + j * 104_729) % flat.len()].clone()).collect()).collect()),
    ];
    let ff: Vec<(&str, Vec<(f64, f64, bool, bool)>)> = versions.iter().map(|(k, qs)| (*k, qs.iter().map(|q| fam(q)).collect())).collect();
    assert!((ids.len() as u64) < B);
    let fv = |k: &str| &ff.iter().find(|x| x.0 == k).unwrap().1;
    // each feature as a per-prompt score: pairs, bigrams, their mean
    let feat: [fn(&(f64, f64, bool, bool)) -> f64; 3] = [|t| t.0, |t| t.1, |t| (t.0 + t.1) / 2.0];

    println!("\n2. NOVELTY CEILING  ('have I seen these words together?' over the 30000 training prompts)");
    println!("   seen in training: {} distinct content-word pairs, {} distinct bigrams", pairs_seen.len(), bigrams_seen.len());
    println!("   test prompts whose exact token sequence is also a training prompt: {} / {}", tq.iter().filter(|q| train_prompts.contains(*q)).count(), tq.len());
    println!("   {:<10} {:>16} {:>18} {:>15} {:>17}", "version", "no pairs (=1.0)", "no bigrams (=1.0)", "mean fam pairs", "mean fam bigrams");
    for (k, fs) in &ff {
        let n = fs.len();
        let (npair, nbig) = (fs.iter().filter(|t| !t.2).count(), fs.iter().filter(|t| !t.3).count());
        let (mut s0, mut s1) = (0.0, 0.0);
        for t in fs { s0 += t.0; s1 += t.1; }
        println!("   {:<10} {:>7} ({:>6}) {:>8} ({:>6}) {:>15.4} {:>17.4}", k, npair, pct(npair, n).trim(), nbig, pct(nbig, n).trim(), s0 / n as f64, s1 / n as f64);
    }

    println!("\n   AUC of familiarity, normal vs strange (0.5 = can't tell; goal mixed-up >= 0.85; her confidence: 0.66)");
    println!("   {:<12} {:>8} {:>8} {:>8}", "", "pairs", "bigrams", "mean");
    for k in ["reversed", "mixed-up"] {
        let row: Vec<f64> = feat.iter().map(|fun| {
            auc(fv("normal").iter().map(|t| (fun(t), true)).chain(fv(k).iter().map(|t| (fun(t), false))).collect())
        }).collect();
        println!("   {:<12} {:>8.4} {:>8.4} {:>8.4}", k, row[0], row[1], row[2]);
    }
    // supplementary: the pair AUC among prompts that HAVE pairs (each side keeps its own)
    println!("   supplementary, pairs AUC with no-pair prompts dropped from each side:");
    for k in ["reversed", "mixed-up"] {
        let xs: Vec<(f64, bool)> = fv("normal").iter().filter(|t| t.2).map(|t| (t.0, true)).chain(fv(k).iter().filter(|t| t.2).map(|t| (t.0, false))).collect();
        println!("     {:<10} {:.4}   (normal n={}, {} n={})", k, auc(xs), fv("normal").iter().filter(|t| t.2).count(), k, fv(k).iter().filter(|t| t.2).count());
    }
    // supplementary: a third of the test prompts are verbatim training prompts ("what ?");
    // does the separation survive on the prompts she has never seen word for word?
    let fresh: Vec<usize> = (0..tq.len()).filter(|&i| !train_prompts.contains(&tq[i])).collect();
    println!("   supplementary, only the {} prompts that are NOT a verbatim training prompt (both sides: prompt i and its strange version i):", fresh.len());
    println!("   {:<12} {:>8} {:>8} {:>8}", "", "pairs", "bigrams", "mean");
    for k in ["reversed", "mixed-up"] {
        let row: Vec<f64> = feat.iter().map(|fun| {
            auc(fresh.iter().map(|&i| (fun(&fv("normal")[i]), true)).chain(fresh.iter().map(|&i| (fun(&fv(k)[i]), false))).collect())
        }).collect();
        println!("   {:<12} {:>8.4} {:>8.4} {:>8.4}", k, row[0], row[1], row[2]);
    }

    println!("\n   the cost: prompts scored familiarity 0 (no pair / no bigram ever seen)");
    println!("   {:<10} {:>16} {:>16} {:>16}", "version", "pairs = 0", "bigrams = 0", "mean = 0");
    for (k, fs) in &ff {
        let n = fs.len();
        let z: Vec<String> = feat.iter().map(|fun| { let c = fs.iter().filter(|t| fun(t) == 0.0).count(); format!("{:>6} ({:>7})", c, pct(c, n).trim()) }).collect();
        println!("   {:<10} {}", k, z.join(" "));
    }
}
