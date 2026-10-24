// TESTING / DESIGN ONLY. (v28 idea 1) Learn word meaning from reading, as sparse codes.
//
//   cargo run --release --example meaning_build -- <ava-root> <out-file>
//
// RANDOM INDEXING, a counting form of "a word is known by the company it keeps". Every word has
// a fixed sparse random signature (8 of D units, ±1). Reading a sentence, each word's meaning
// trace gains the signatures of the words within 3 places of it, a rare neighbour weighing more
// (ln(T / count), the same rarity she weighs everywhere). One pass, only additions: no gradient,
// no training loop. Afterwards the common direction every trace shares is removed and each
// word keeps its K most active units: its sparse code (common/meaning.rs).
//
// What she reads (all already on disk, nothing downloaded):
//   data/eng_sentences.tsv  English sentences (Tatoeba)
//   data/cornell_lines.txt  movie lines
//   data/triples.tsv        her dialogue, WITHOUT the 1,200 exam exchanges
//   Rust reference + nomicon pages (main text) and Rust by Example (markdown)
// Never read: the fact files (their questions, answers and quotes are what she is tested on).
#[path = "common/meaning.rs"]
mod meaning;
use design_tests::seq::tokenize;
use meaning::{Meaning, D, K};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

const WINDOW: usize = 3;
const MIN_COUNT: u32 = 3;
const SIGN_UNITS: usize = 8;

fn words(s: &str) -> Vec<String> {
    tokenize(s).into_iter().filter(|w| w.chars().any(|c| c.is_alphanumeric())).collect()
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// a word's fixed random signature: SIGN_UNITS distinct units, each +1 or -1
fn signature(w: &str) -> [(u16, f32); SIGN_UNITS] {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in w.bytes() { h = (h ^ b as u64).wrapping_mul(0x100000001b3); }
    let mut out = [(0u16, 0f32); SIGN_UNITS];
    let mut n = 0;
    while n < SIGN_UNITS {
        let r = splitmix(&mut h);
        let u = (r % D as u64) as u16;
        if out[..n].iter().any(|x| x.0 == u) { continue; }
        out[n] = (u, if (r >> 32) & 1 == 0 { 1.0 } else { -1.0 });
        n += 1;
    }
    out
}

/// the main text of an mdBook page: inside <main>, tags removed, common entities decoded
fn html_text(html: &str) -> String {
    let body = match (html.find("<main"), html.rfind("</main>")) { (Some(a), Some(b)) if a < b => &html[a..b], _ => html };
    let mut out = String::with_capacity(body.len());
    let mut tag = false;
    for c in body.chars() {
        match c { '<' => { tag = true; out.push(' '); } '>' => tag = false, _ if !tag => out.push(c), _ => {} }
    }
    out.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'").replace("&nbsp;", " ")
}

fn files(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() { files(&p, ext, out); } else if p.extension().map_or(false, |x| x == ext) { out.push(p); }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 { eprintln!("usage: meaning_build <ava-root> <out-file>"); std::process::exit(1); }
    let (root, out) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]));
    let data = root.join("data");
    let t0 = std::time::Instant::now();

    // ---------------------------------------------------------------- what she reads
    let mut sents: Vec<Vec<String>> = Vec::new();
    let mut n_src: Vec<(&str, usize)> = Vec::new();
    let tat = String::from_utf8_lossy(&std::fs::read(data.join("eng_sentences.tsv")).expect("eng_sentences.tsv")).into_owned();
    let before = sents.len();
    for l in tat.lines() { if let Some(s) = l.split('\t').nth(2) { sents.push(words(s)); } }
    n_src.push(("english sentences", sents.len() - before));
    drop(tat);
    let corn = String::from_utf8_lossy(&std::fs::read(data.join("cornell_lines.txt")).expect("cornell_lines.txt")).into_owned();
    let before = sents.len();
    for l in corn.lines() { if let Some(s) = l.rsplit(" +++$+++ ").next() { sents.push(words(s)); } }
    n_src.push(("movie lines", sents.len() - before));
    drop(corn);
    // the dialogue exam is never read: its 1,200 exchanges are the first lines of school/dialogue.tsv.
    // (v29) Every exam turn, question or reply, is withheld wherever it appears: as a triple's
    // neighbouring turn, or as an identical sentence in any other source. Skipping only the exam's
    // own (question, reply) triple, as v28 did, still read 393 exam questions and 361 replies.
    let exam_turns: HashSet<Vec<String>> = std::fs::read_to_string(data.join("school/dialogue.tsv")).unwrap_or_default()
        .lines().take(1200).flat_map(|l| { let p: Vec<&str> = l.split('\t').collect(); if p.len() >= 3 { vec![words(p[1]), words(p[2])] } else { vec![] } })
        .filter(|w| !w.is_empty()).collect();
    let trip = std::fs::read_to_string(data.join("triples.tsv")).expect("triples.tsv");
    let before = sents.len();
    let mut seen_turn: HashSet<&str> = HashSet::new();
    for l in trip.lines() {
        let p: Vec<&str> = l.split('\t').collect();
        if p.len() < 3 { continue; }
        // each turn appears in up to three triples; read it once
        for t in &p[..3] { if !t.is_empty() && seen_turn.insert(t) { sents.push(words(t)); } }
    }
    n_src.push(("dialogue turns", sents.len() - before));
    let dl = data.join("school/downloads");
    let before = sents.len();
    let mut pages = Vec::new();
    files(&dl.join("reference"), "html", &mut pages);
    files(&dl.join("nomicon"), "html", &mut pages);
    pages.sort();
    for p in &pages {
        let t = html_text(&String::from_utf8_lossy(&std::fs::read(p).expect("page")));
        for s in t.split(['.', '\n', '?', '!']) { let w = words(s); if w.len() > 1 { sents.push(w); } }
    }
    let mut md = Vec::new();
    files(&dl.join("rust-by-example"), "md", &mut md);
    md.sort();
    for p in &md {
        let t = String::from_utf8_lossy(&std::fs::read(p).expect("md")).into_owned();
        for s in t.split(['.', '\n', '?', '!']) { let w = words(s); if w.len() > 1 { sents.push(w); } }
    }
    n_src.push(("rust book sentences", sents.len() - before));
    let n_read = sents.len();
    sents.retain(|s| !exam_turns.contains(s));
    let skipped = n_read - sents.len();
    let total: usize = sents.iter().map(|s| s.len()).sum();
    for (k, n) in &n_src { println!("  read {k:<20} {n:>9}"); }
    println!("  {} sentences, {} words ({} sentences equal to a dialogue exam turn withheld) in {:.1}s", sents.len(), total, skipped, t0.elapsed().as_secs_f32());

    // ---------------------------------------------------------------- counts, then traces
    let mut count: HashMap<&str, u32> = HashMap::new();
    for s in &sents { for w in s { *count.entry(w.as_str()).or_default() += 1; } }
    let mut vocab: Vec<&str> = count.iter().filter(|x| *x.1 >= MIN_COUNT).map(|x| *x.0).collect();
    vocab.sort_unstable();
    let id: HashMap<&str, usize> = vocab.iter().enumerate().map(|(i, w)| (*w, i)).collect();
    let sig: Vec<[(u16, f32); SIGN_UNITS]> = vocab.iter().map(|w| signature(w)).collect();
    let rare: Vec<f32> = vocab.iter().map(|w| (total as f32 / count[w] as f32).ln().max(0.0)).collect();
    println!("  vocabulary: {} words seen at least {MIN_COUNT} times (of {})", vocab.len(), count.len());
    let mut trace = vec![0f32; vocab.len() * D];
    for s in &sents {
        let ix: Vec<Option<usize>> = s.iter().map(|w| id.get(w.as_str()).copied()).collect();
        for (i, wi) in ix.iter().enumerate() {
            let Some(wi) = *wi else { continue };
            let row = &mut trace[wi * D..(wi + 1) * D];
            for j in i.saturating_sub(WINDOW)..(i + WINDOW + 1).min(ix.len()) {
                if j == i { continue; }
                let Some(c) = ix[j] else { continue };
                for &(u, sgn) in &sig[c] { row[u as usize] += sgn * rare[c]; }
            }
        }
    }
    // remove the direction every trace shares (what all words have in common says nothing about
    // any one of them), then keep each word's K most active units
    let mut mean = vec![0f64; D];
    for r in trace.chunks(D) {
        let n = r.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt().max(1e-9);
        for (m, x) in mean.iter_mut().zip(r) { *m += *x as f64 / n; }
    }
    for m in mean.iter_mut() { *m /= vocab.len() as f64; }
    let mut code: Vec<[u16; K]> = Vec::with_capacity(vocab.len());
    for r in trace.chunks(D) {
        let n = r.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt().max(1e-9);
        let mut v: Vec<(f64, u16)> = r.iter().enumerate().map(|(u, x)| (*x as f64 / n - mean[u], u as u16)).collect();
        v.select_nth_unstable_by(K - 1, |x, y| y.0.total_cmp(&x.0));
        let mut c = [0u16; K];
        for (k, x) in v[..K].iter().enumerate() { c[k] = x.1; }
        c.sort_unstable();
        code.push(c);
    }
    let m = Meaning { id: vocab.iter().enumerate().map(|(i, w)| (w.to_string(), i as u32)).collect(), code };
    m.save(&out).expect("write the meaning file");
    println!("  saved {} codes ({} of {} units each) to {} in {:.1}s", vocab.len(), K, D, out.display(), t0.elapsed().as_secs_f32());

    // ---------------------------------------------------------------- a look at what she learned
    let common: Vec<&str> = { let mut v: Vec<(&&str, &u32)> = count.iter().filter(|x| *x.1 >= 50).collect(); v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0))); v.into_iter().take(30000).map(|x| *x.0).collect() };
    for w in ["car", "happy", "begin", "big", "mother", "year", "variable", "mutable", "function", "network", "designed", "invented", "memory", "neuron", "gate"] {
        let mut nb: Vec<(f32, &str)> = common.iter().filter(|c| **c != w).map(|c| (m.overlap(w, c), *c)).collect();
        nb.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(b.1)));
        println!("  {w:<10} {}", nb.iter().take(6).map(|(s, c)| format!("{c} {s:.2}")).collect::<Vec<_>>().join(", "));
    }
    let pairs = [("begin", "start"), ("big", "large"), ("car", "automobile"), ("designed", "created"), ("inspired", "modelled"),
                 ("happy", "glad"), ("big", "small"), ("car", "happy"), ("year", "when"), ("person", "who")];
    println!("  pairs: {}", pairs.iter().map(|(x, y)| format!("{x}~{y} {:.2}", m.overlap(x, y))).collect::<Vec<_>>().join(", "));
}
