// TESTING / DESIGN ONLY.
// (v29 WP6b) Grammar that reaches back: BLiMP with working memory, word classes and word forms.
//
//   cargo run --release --example blimp_wm -- check                streamed == stored for every reading; every mixture sums to 1
//   cargo run --release --example blimp_wm -- dev [devlog.md]      the 8 subsets on dev (pairID % 5 == 0); the run is appended to devlog
//   cargo run --release --example blimp_wm -- dryrun <report.txt>  the frozen test routine, run on dev first
//   cargo run --release --example blimp_wm -- test <report.txt>    the frozen build on test, ONCE (refuses to overwrite)
//   cargo run --release --example blimp_wm -- arc <report.txt>     report only: the ARC science corpus as a bigger reading
//
// Started as a copy of blimp.rs (WP6, frozen sha256 765629c8...): the same reading (Tatoeba, movie
// lines, her dialogue turns, every dialogue-exam turn withheld, shuffled by sentence with splitmix64
// seed 81), the same words (her tokenizer's cuts, cased), the same sentence score (sum of ln P of each
// word and the end; the good sentence must score strictly higher, |good - bad| <= 1e-9 is a tie and a
// tie is wrong), the same streamed counting (only what the scored sentences use is kept, which gives
// the same numbers as storing everything; `check` compares the two).
//
// Her next word is a weighted MIXTURE OF READINGS. Each reading is a Witten-Bell backoff, as in WP6:
// read from the most general level up, p = lambda * count(context, x) / c + (1 - lambda) * p_below,
// lambda = c / (c + d) (c = times she read the context, d = different things she saw follow it); a
// context she never read is passed over.
//   W  words: WP6's frozen reading, up to 3 previous words, cased. Under it the any-word level: an even
//      share over every word read plus one never met.
//   M  working memory: each EARLIER word u of the sentence beyond the last 3 (4 or more places back,
//      at any distance) votes with its own Witten-Bell estimate of "x comes later in a sentence that
//      holds u": counted from every (u, x) she read 4 or more places apart (x may be the end), backed
//      off to W's any-word level. The votes are averaged with weight decay^(age), age 0 for the
//      nearest voter; the decay (0..1) is LEARNED from her reading: the one that makes M's own
//      predictions of the held-out slice most likely. A word with no earlier word beyond the last 3
//      has no voter: M is silent there, and the readings that speak share its weight in proportion.
//   C  word classes: a word's class is the strongest unit of its meaning code (brain/meaning-v29.bin).
//      That file keeps each code's 32 units sorted by unit number, not by strength, so the strengths
//      are recomputed here exactly as meaning_build computes them (the same reading, the same random
//      indexing), the recomputed codes are checked to equal the file's codes for every word, and the
//      class is the code unit with the highest strength. A word with no code is its own class.
//      P(x | before) = P(class(x) | previous classes, up to 3) * P(x | class(x)), the second the share
//      of the class's tokens that were x. A word never met falls in a class of its own, never met.
//   F  word forms: a word's grammar features from her UniMorph morphology (data/unimorph-eng.tsv):
//      number (SG, PL), person (1, 2, 3), tense (PRS, PST); a form with several analyses carries all of
//      them (the union); a word with no analysis gets the feature "unknown". The set of features is
//      the word's form class, and P(x | before) = P(form(x) | previous forms, up to 3) * P(x | form(x)),
//      exactly as C.
// Classes are looked up lower-cased (her meaning codes are lower-case); forms by the word as written, else lower-cased
// (so an abbreviation such as UniMorph's "IS", N;SG, does not give "is" a number); the word
// itself stays cased, so every reading is a distribution over the same things: every (cased) word she
// has read, the end of a sentence, and one word never met.
//
// THE WEIGHTS of W, M, C and F are learned by EM on a held-out slice of her READING: the last 5% of the
// words of each reading (at 10M: the sentences after the first 9.5M words), which is NOT counted. Where
// M is silent the weights of the others are renormalised; the EM step is then the standard one
// generalised (lambda_k = sum of responsibilities / sum of 1 / (weights speaking)), a minorise-maximise
// update that never lowers the held-out likelihood (asserted). BLiMP is never used for any weight.
// W alone has nothing to learn and counts its whole reading (that is WP6's reading, exactly).
//
// Dev (pairID % 5 == 0) chooses ONLY which of M, C, F join W (8 subsets): the best at 10M words; a tie
// goes to fewer readings, then to the earlier one in the order of SUBSETS. Test is scored once, by the
// frozen build: rule 1 overall >= 60.0% at 10M words; rule 2 higher than WP6's frozen reading (W alone)
// on the same test pairs, paired bootstrap 95% CI above 0 (20,000 resamples, splitmix64 seed 1).
//
// Pre-registered in "ava v29 - in depth" (WP6b).
#[path = "common/meaning.rs"]
mod meaning;
use design_tests::seq::tokenize;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DATA: &str = "../data";
const MEANING_FILE: &str = "../brain/meaning-v29.bin";
const UNIMORPH: &str = "../data/unimorph-eng.tsv";
const ARC_CORPUS: &str = "../data/ai2-arc/ARC-V1-Feb2018-2/ARC_Corpus.txt";
const SOURCE: &str = include_str!("blimp_wm.rs");
const SEED: u64 = 81;
const MAIN: u64 = 10_000_000;
const MARKS: [u64; 4] = [1_000_000, 2_500_000, 5_000_000, MAIN];
const MAXN: usize = 3; // W, C, F: up to 3 previous words / classes / forms (WP6 frozen: 3 words, cased)
const GAP: usize = 4; // M: an earlier word beyond the last 3 stands 4 or more places back
const HOLD: f64 = 0.05; // the last 5% of a reading: EM's held-out slice, not counted
const TIE: f64 = 1e-9;
const SLOT: usize = 21; // bits per context word in a context key
const ARC_SHARDS: u64 = 16; // arc: M's different-later-word sets are counted in 16 slices of the voters
const EM_MAX_IT: usize = 5000;
const NAMES: [&str; 4] = ["W", "M", "C", "F"];
/// which of M (1), C (2), F (4) join W; the order also breaks dev ties
const SUBSETS: [u8; 8] = [0, 1, 2, 4, 3, 5, 6, 7];

// ---------------------------------------------------------------- frozen setting (test, dryrun, arc)
// Chosen on dev before any test pair was scored (scratchpad v29/blimp_wm_DEVLOG.md, dev run 1, source
// sha256 ee29c378...). Rule, fixed before the dev run: the best dev accuracy at 10M words; a tie goes to
// fewer readings, then to the earlier in SUBSETS. Dev at 10M words (13,400 pairs): W 55.65%, W+M 55.60,
// W+C 55.10, W+F 54.09, W+M+C 55.38, W+M+F 54.41, W+C+F 54.47, W+M+C+F 54.61 -> W alone (WP6's
// reading). The readings that reach back raise the held-out likelihood but did not help on dev.
const FROZEN_SUBSET: u8 = 0;

const BOS: u32 = 0;
const EOS: u32 = 1;
/// class readings: the class of a word never met (never counted)
const NEW: u32 = 2;
const NONE: u32 = u32::MAX;
/// the end of a sentence, as a word key (no word's hash is this, short of a 1 in 2^64 accident)
const EOS_KEY: u64 = 0x0e05_0e05_0e05_0e05;
const GOLD: u64 = 0x9e37_79b9_7f4a_7c15;
const UNKNOWN: u8 = 128;

// ---------------------------------------------------------------- small tools
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(GOLD);
    mix64(*x)
}
#[derive(Default)]
struct Mix(u64);
impl Hasher for Mix {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, b: &[u8]) { for &x in b { self.0 = mix64(self.0 ^ x as u64 ^ GOLD); } }
    fn write_u64(&mut self, x: u64) { self.0 = mix64(self.0 ^ x); }
    fn write_u32(&mut self, x: u32) { self.write_u64(x as u64) }
}
type MixMap<K, V> = HashMap<K, V, BuildHasherDefault<Mix>>;
type MixSet<K> = HashSet<K, BuildHasherDefault<Mix>>;

/// a word's identity: FNV-1a over its bytes, then mixed
fn wkey(w: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in w.as_bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3); }
    mix64(h)
}

fn die(m: &str) -> ! { eprintln!("{m}"); std::process::exit(2) }
fn pct(a: u64, b: u64) -> f64 { if b == 0 { 0.0 } else { a as f64 / b as f64 * 100.0 } }
fn th(n: u64) -> String {
    let s = n.to_string();
    let mut o = String::new();
    for (i, c) in s.chars().enumerate() { if i > 0 && (s.len() - i) % 3 == 0 { o.push(','); } o.push(c); }
    o
}
fn dec(d: f64) -> String { if d.is_nan() { "-".into() } else { format!("{d:.4}") } }
fn set_name(s: u8) -> String {
    let mut n = String::from("W");
    for (b, x) in [(1u8, "M"), (2, "C"), (4, "F")] { if s & b != 0 { n.push('+'); n.push_str(x); } }
    n
}
/// is reading k (0 W, 1 M, 2 C, 3 F) in the subset?
fn on(set: u8, k: usize) -> bool { k == 0 || set & (1 << (k - 1)) != 0 }
macro_rules! say { ($o:expr, $($t:tt)*) => {{ let s = format!($($t)*); println!("{s}"); $o.push_str(&s); $o.push('\n'); }} }

#[cfg(windows)]
mod win {
    #[repr(C)]
    pub struct MemStatusEx { pub len: u32, pub load: u32, pub total_phys: u64, pub avail_phys: u64, pub total_pf: u64,
        pub avail_pf: u64, pub total_virt: u64, pub avail_virt: u64, pub avail_ext: u64 }
    #[repr(C)]
    pub struct Pmc { pub cb: u32, pub faults: u32, pub peak_ws: usize, pub ws: usize, pub qpp: usize, pub qp: usize,
        pub qpnp: usize, pub qnp: usize, pub pf: usize, pub peak_pf: usize }
    #[link(name = "kernel32")]
    extern "system" {
        pub fn GlobalMemoryStatusEx(p: *mut MemStatusEx) -> i32;
        pub fn GetCurrentProcess() -> isize;
        pub fn K32GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32;
    }
}
/// the whole machine's free physical memory, GB
fn free_gb() -> f64 {
    #[cfg(windows)]
    unsafe {
        let mut m: win::MemStatusEx = std::mem::zeroed();
        m.len = std::mem::size_of::<win::MemStatusEx>() as u32;
        if win::GlobalMemoryStatusEx(&mut m) != 0 { return m.avail_phys as f64 / 1e9; }
    }
    f64::INFINITY
}
/// this process's peak working set so far, GB
fn peak_gb() -> f64 {
    #[cfg(windows)]
    unsafe {
        let mut p: win::Pmc = std::mem::zeroed();
        p.cb = std::mem::size_of::<win::Pmc>() as u32;
        if win::K32GetProcessMemoryInfo(win::GetCurrentProcess(), &mut p, p.cb) != 0 { return p.peak_ws as f64 / 1e9; }
    }
    0.0
}
fn ram_guard(at: &str) {
    let f = free_gb();
    if f < 3.5 { eprintln!("!! stopping at {at}: free RAM {f:.2} GB is under 3.5 GB"); std::process::exit(3); }
}

/// SHA-256, so a report names the exact source its build was compiled from
fn sha256(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut m = data.to_vec();
    m.push(0x80);
    while m.len() % 64 != 56 { m.push(0); }
    m.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for ch in m.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes([ch[4 * i], ch[4 * i + 1], ch[4 * i + 2], ch[4 * i + 3]]); }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut a = h;
        for i in 0..64 {
            let s1 = a[4].rotate_right(6) ^ a[4].rotate_right(11) ^ a[4].rotate_right(25);
            let chs = (a[4] & a[5]) ^ (!a[4] & a[6]);
            let t1 = a[7].wrapping_add(s1).wrapping_add(chs).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a[0].rotate_right(2) ^ a[0].rotate_right(13) ^ a[0].rotate_right(22);
            let maj = (a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]);
            let t2 = s0.wrapping_add(maj);
            a = [t1.wrapping_add(t2), a[0], a[1], a[2], a[3].wrapping_add(t1), a[4], a[5], a[6]];
        }
        for i in 0..8 { h[i] = h[i].wrapping_add(a[i]); }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

// ---------------------------------------------------------------- her eyes: words
/// her tokenizer (seq::tokenize) without lower-casing: the same cuts, the case kept
fn tokenize_cased(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() || c == '\'' { cur.push(c); } else {
            if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
            if !c.is_whitespace() { out.push(c.to_string()); }
        }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}
/// the words of a sentence: tokens with a letter or digit (meaning_build's `words`)
fn words(s: &str, lower: bool) -> Vec<String> {
    let t = if lower { tokenize(s) } else { tokenize_cased(s) };
    t.into_iter().filter(|w| w.chars().any(|c| c.is_alphanumeric())).collect()
}
/// lower-cased the way her tokenizer lower-cases (char by char)
fn lower(w: &str) -> String { w.chars().flat_map(|c| c.to_lowercase()).collect() }

// ---------------------------------------------------------------- BLiMP
struct Pair { uid: usize, term: usize, id: u32, good: String, bad: String }

/// one field of a flat JSON line (BLiMP's lines are flat; strings may hold escapes)
fn field(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\": ");
    let rest = &line[line.find(&pat)? + pat.len()..];
    let Some(r) = rest.strip_prefix('"') else { return Some(rest.split([',', '}']).next()?.trim().to_string()) };
    let mut out = String::new();
    let mut it = r.chars();
    while let Some(c) = it.next() {
        match c {
            '"' => return Some(out),
            '\\' => match it.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'u' => { let h: String = (&mut it).take(4).collect(); out.push(char::from_u32(u32::from_str_radix(&h, 16).ok()?)?); }
                o => out.push(o),
            },
            _ => out.push(c),
        }
    }
    None
}

fn load_blimp() -> (Vec<Pair>, Vec<String>, Vec<String>) {
    let dir = format!("{DATA}/blimp");
    let mut files: Vec<String> = std::fs::read_dir(&dir).unwrap_or_else(|e| die(&format!("cannot read {dir}: {e}")))
        .flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|f| f.ends_with(".jsonl")).collect();
    files.sort();
    let (mut pairs, mut uids, mut terms) = (Vec::new(), Vec::<String>::new(), Vec::<String>::new());
    for f in &files {
        let raw = std::fs::read_to_string(format!("{dir}/{f}")).unwrap_or_else(|e| die(&format!("{f}: {e}")));
        for line in raw.lines().filter(|l| !l.trim().is_empty()) {
            let get = |k: &str| field(line, k).unwrap_or_else(|| die(&format!("{f}: no {k} in {line}")));
            let (uid, term) = (get("UID"), get("linguistics_term"));
            let ui = uids.iter().position(|u| *u == uid).unwrap_or_else(|| { uids.push(uid); uids.len() - 1 });
            let ti = terms.iter().position(|t| *t == term).unwrap_or_else(|| { terms.push(term); terms.len() - 1 });
            let id = get("pairID").parse().unwrap_or_else(|_| die(&format!("{f}: bad pairID")));
            pairs.push(Pair { uid: ui, term: ti, id, good: get("sentence_good"), bad: get("sentence_bad") });
        }
    }
    (pairs, uids, terms)
}

// ---------------------------------------------------------------- what she reads (WP6, unchanged)
struct Reading { sents: Vec<Box<str>>, exam: HashSet<Vec<String>>, lens: Vec<u32> }

fn reading(out: &mut String) -> Reading {
    let t0 = Instant::now();
    let read = |f: &str| String::from_utf8_lossy(&std::fs::read(format!("{DATA}/{f}")).unwrap_or_else(|e| die(&format!("{f}: {e}")))).into_owned();
    // the dialogue exam: every question and reply among the first 1,200 lines, as meaning_build
    let exam: HashSet<Vec<String>> = read("school/dialogue.tsv").lines().take(1200)
        .flat_map(|l| { let p: Vec<&str> = l.split('\t').collect(); if p.len() >= 3 { vec![words(p[1], true), words(p[2], true)] } else { vec![] } })
        .filter(|w| !w.is_empty()).collect();
    if exam.is_empty() { die("no dialogue exam turns found"); }
    let mut cand: Vec<(usize, Box<str>)> = Vec::new();
    let tat = read("eng_sentences.tsv");
    for l in tat.lines() { if let Some(s) = l.split('\t').nth(2) { cand.push((0, s.into())); } }
    drop(tat);
    let corn = read("cornell_lines.txt");
    for l in corn.lines() { if let Some(s) = l.rsplit(" +++$+++ ").next() { cand.push((1, s.into())); } }
    drop(corn);
    let trip = read("triples.tsv");
    let mut seen_turn: HashSet<&str> = HashSet::new();
    for l in trip.lines() {
        let p: Vec<&str> = l.split('\t').collect();
        if p.len() < 3 { continue; }
        // each turn appears in up to three triples; read it once
        for t in &p[..3] { if !t.is_empty() && seen_turn.insert(t) { cand.push((2, (*t).into())); } }
    }
    drop(seen_turn);
    let mut st = [[0u64; 4]; 3]; // per source: lines, no word, exam turn, words kept
    let mut sents: Vec<Box<str>> = Vec::with_capacity(cand.len());
    for (src, s) in cand {
        let w = words(&s, true);
        st[src][0] += 1;
        if w.is_empty() { st[src][1] += 1; } else if exam.contains(&w) { st[src][2] += 1; } else { st[src][3] += w.len() as u64; sents.push(s); }
    }
    drop(trip);
    // shuffled by sentence: Fisher-Yates from the end, splitmix64 seed 81
    let mut x = SEED;
    for i in (1..sents.len()).rev() { let j = (splitmix(&mut x) % (i as u64 + 1)) as usize; sents.swap(i, j); }
    let total: u64 = st.iter().map(|s| s[3]).sum();
    say!(out, "what she reads ({} dialogue-exam turns withheld wherever they appear):", exam.len());
    for (k, name) in ["Tatoeba sentences", "movie lines", "dialogue turns"].iter().enumerate() {
        say!(out, "  {name:<18} {:>10} lines, {:>7} with no word, {:>5} equal to an exam turn; {:>11} words kept",
             th(st[k][0]), th(st[k][1]), th(st[k][2]), th(st[k][3]));
    }
    say!(out, "  {} sentences, {} words, shuffled (splitmix64 seed {SEED}) in {:.1}s", th(sents.len() as u64), th(total), t0.elapsed().as_secs_f64());
    if total <= MAIN { die("her reading has no more than 10M words"); }
    // the words of each sentence, cased (the same count as lower-cased: the same cuts)
    let lens: Vec<u32> = sents.iter().map(|s| words(s, false).len() as u32).collect();
    if lens.iter().map(|&l| l as u64).sum::<u64>() != total { die("cased and lower-cased word counts differ"); }
    Reading { sents, exam, lens }
}

/// the reading at `limit` words: every sentence before the first one that would take her past it
fn prefix_end(lens: &[u32], limit: u64) -> (usize, u64) {
    let mut cum = 0u64;
    for (k, &l) in lens.iter().enumerate() {
        if cum + l as u64 > limit { return (k, cum); }
        cum += l as u64;
    }
    (lens.len(), cum)
}

// ---------------------------------------------------------------- her lexicon: classes and forms
/// meaning_build's random signature for a word (copied: the strengths must be recomputed exactly)
fn signature(w: &str) -> [(u16, f32); 8] {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in w.bytes() { h = (h ^ b as u64).wrapping_mul(0x100000001b3); }
    let mut out = [(0u16, 0f32); 8];
    let mut n = 0;
    while n < 8 {
        let r = splitmix(&mut h);
        let u = (r % meaning::D as u64) as u16;
        if out[..n].iter().any(|x| x.0 == u) { continue; }
        out[n] = (u, if (r >> 32) & 1 == 0 { 1.0 } else { -1.0 });
        n += 1;
    }
    out
}
/// the main text of an mdBook page (meaning_build's, copied)
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

/// each coded word's class: the strongest unit of its meaning code. The strengths are recomputed as
/// meaning_build computes them (its reading, in its order; its signatures, window, rarity, centring),
/// and every recomputed code must equal the code in meaning-v29.bin.
fn strongest_units(out: &mut String) -> HashMap<String, u16> {
    let t0 = Instant::now();
    let data = Path::new(DATA);
    let exam: HashSet<Vec<String>> = std::fs::read_to_string(data.join("school/dialogue.tsv")).unwrap_or_default()
        .lines().take(1200).flat_map(|l| { let p: Vec<&str> = l.split('\t').collect(); if p.len() >= 3 { vec![words(p[1], true), words(p[2], true)] } else { vec![] } })
        .filter(|w| !w.is_empty()).collect();
    let mut tmp: HashMap<String, u32> = HashMap::new();
    let mut sents: Vec<Box<[u32]>> = Vec::new();
    let mut withheld = 0usize;
    {
        let mut push = |w: Vec<String>| {
            if exam.contains(&w) { withheld += 1; return; }
            let ids: Box<[u32]> = w.into_iter().map(|x| { let n = tmp.len() as u32; *tmp.entry(x).or_insert(n) }).collect();
            sents.push(ids);
        };
        let tat = String::from_utf8_lossy(&std::fs::read(data.join("eng_sentences.tsv")).expect("eng_sentences.tsv")).into_owned();
        for l in tat.lines() { if let Some(s) = l.split('\t').nth(2) { push(words(s, true)); } }
        drop(tat);
        let corn = String::from_utf8_lossy(&std::fs::read(data.join("cornell_lines.txt")).expect("cornell_lines.txt")).into_owned();
        for l in corn.lines() { if let Some(s) = l.rsplit(" +++$+++ ").next() { push(words(s, true)); } }
        drop(corn);
        let trip = std::fs::read_to_string(data.join("triples.tsv")).expect("triples.tsv");
        let mut seen_turn: HashSet<&str> = HashSet::new();
        for l in trip.lines() {
            let p: Vec<&str> = l.split('\t').collect();
            if p.len() < 3 { continue; }
            for t in &p[..3] { if !t.is_empty() && seen_turn.insert(t) { push(words(t, true)); } }
        }
        drop(seen_turn);
        let dl = data.join("school/downloads");
        let mut pages = Vec::new();
        files(&dl.join("reference"), "html", &mut pages);
        files(&dl.join("nomicon"), "html", &mut pages);
        pages.sort();
        for p in &pages {
            let t = html_text(&String::from_utf8_lossy(&std::fs::read(p).expect("page")));
            for s in t.split(['.', '\n', '?', '!']) { let w = words(s, true); if w.len() > 1 { push(w); } }
        }
        let mut md = Vec::new();
        files(&dl.join("rust-by-example"), "md", &mut md);
        md.sort();
        for p in &md {
            let t = String::from_utf8_lossy(&std::fs::read(p).expect("md")).into_owned();
            for s in t.split(['.', '\n', '?', '!']) { let w = words(s, true); if w.len() > 1 { push(w); } }
        }
    }
    let mut count = vec![0u32; tmp.len()];
    let mut total = 0usize;
    for s in &sents { for &w in s.iter() { count[w as usize] += 1; } total += s.len(); }
    let mut vocab: Vec<(&str, u32)> = tmp.iter().filter(|(_, i)| count[**i as usize] >= 3).map(|(w, &i)| (w.as_str(), i)).collect();
    vocab.sort_unstable_by(|a, b| a.0.cmp(b.0));
    let mut vid = vec![u32::MAX; tmp.len()];
    for (k, (_, i)) in vocab.iter().enumerate() { vid[*i as usize] = k as u32; }
    let sig: Vec<[(u16, f32); 8]> = vocab.iter().map(|(w, _)| signature(w)).collect();
    let rare: Vec<f32> = vocab.iter().map(|(_, i)| (total as f32 / count[*i as usize] as f32).ln().max(0.0)).collect();
    let dd = meaning::D;
    let mut trace = vec![0f32; vocab.len() * dd];
    let mut ix: Vec<Option<usize>> = Vec::new();
    for s in &sents {
        ix.clear();
        ix.extend(s.iter().map(|&w| { let v = vid[w as usize]; if v == u32::MAX { None } else { Some(v as usize) } }));
        for (i, wi) in ix.iter().enumerate() {
            let Some(wi) = *wi else { continue };
            let row = &mut trace[wi * dd..(wi + 1) * dd];
            for j in i.saturating_sub(3)..(i + 3 + 1).min(ix.len()) {
                if j == i { continue; }
                let Some(c) = ix[j] else { continue };
                for &(u, sgn) in &sig[c] { row[u as usize] += sgn * rare[c]; }
            }
        }
    }
    let nsent = sents.len();
    drop(sents);
    let mut mean = vec![0f64; dd];
    for r in trace.chunks(dd) {
        let n = r.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt().max(1e-9);
        for (m, x) in mean.iter_mut().zip(r) { *m += *x as f64 / n; }
    }
    for m in mean.iter_mut() { *m /= vocab.len() as f64; }
    let file = meaning::Meaning::load(Path::new(MEANING_FILE)).unwrap_or_else(|e| die(&format!("{MEANING_FILE}: {e}")));
    let (mut mism, mut missing) = (0usize, 0usize);
    let mut unit: HashMap<String, u16> = HashMap::with_capacity(vocab.len());
    let mut size = vec![0u32; dd];
    for (k, r) in trace.chunks(dd).enumerate() {
        let n = r.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt().max(1e-9);
        let mut v: Vec<(f64, u16)> = r.iter().enumerate().map(|(u, x)| (*x as f64 / n - mean[u], u as u16)).collect();
        v.select_nth_unstable_by(meaning::K - 1, |x, y| y.0.total_cmp(&x.0));
        let mut c = [0u16; meaning::K];
        for (j, x) in v[..meaning::K].iter().enumerate() { c[j] = x.1; }
        c.sort_unstable();
        match file.id.get(vocab[k].0) { Some(&fi) => if file.code[fi as usize] != c { mism += 1; }, None => missing += 1 }
        // the strongest of the code's units
        let best = v[..meaning::K].iter().fold((f64::NEG_INFINITY, 0u16), |b, x| if x.0 > b.0 { *x } else { b });
        unit.insert(vocab[k].0.to_string(), best.1);
        size[best.1 as usize] += 1;
    }
    say!(out, "her word classes: {} sentences of meaning_build's reading ({} withheld as exam turns), {} words; {} coded words recomputed in {:.1}s",
         th(nsent as u64), th(withheld as u64), th(total as u64), th(vocab.len() as u64), t0.elapsed().as_secs_f64());
    say!(out, "  recomputed codes vs {MEANING_FILE}: {} words in the file, {mism} codes differ, {missing} words missing", th(file.id.len() as u64));
    if mism > 0 || missing > 0 || file.id.len() != vocab.len() { die("the recomputed meaning codes do not equal meaning-v29.bin: the strongest unit cannot be named"); }
    let used = size.iter().filter(|&&s| s > 0).count();
    let mut big: Vec<(u32, usize)> = size.iter().enumerate().map(|(u, &s)| (s, u)).collect();
    big.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    say!(out, "  {used} of {dd} units are some word's strongest; largest classes: {}",
         big.iter().take(5).map(|(s, u)| format!("unit {u} ({s} words)")).collect::<Vec<_>>().join(", "));
    unit
}

/// her UniMorph morphology: a form as written -> its features (number, person, tense), every analysis carried
fn unimorph() -> HashMap<String, u8> {
    let raw = std::fs::read_to_string(UNIMORPH).unwrap_or_else(|e| die(&format!("{UNIMORPH}: {e}")));
    let mut m: HashMap<String, u8> = HashMap::new();
    for l in raw.lines() {
        let p: Vec<&str> = l.split('\t').collect();
        if p.len() < 3 || p[1].contains(' ') { continue; }
        let mut bits = 0u8;
        for t in p[2].split(';') {
            bits |= match t { "SG" => 1, "PL" => 2, "1" => 4, "2" => 8, "3" => 16, "PRS" => 32, "PST" => 64, _ => 0 };
        }
        *m.entry(p[1].to_string()).or_insert(0) |= bits;
    }
    m
}
fn bundle_name(b: u8) -> String {
    if b == UNKNOWN { return "unknown".into(); }
    let f: Vec<&str> = ["SG", "PL", "1", "2", "3", "PRS", "PST"].iter().enumerate().filter(|(i, _)| b & (1 << i) != 0).map(|(_, x)| *x).collect();
    if f.is_empty() { "none".into() } else { f.join(";") }
}

struct Lex { unit: HashMap<String, u16>, form: HashMap<String, u8> }
impl Lex {
    fn load(out: &mut String) -> Lex {
        let unit = strongest_units(out);
        let form = unimorph();
        let mut by = [0usize; 129];
        for &b in form.values() { by[b as usize] += 1; }
        say!(out, "her word forms: {} UniMorph forms; {} distinct feature sets (number, person, tense; several analyses carried together)",
             th(form.len() as u64), by.iter().filter(|&&n| n > 0).count());
        let lx = Lex { unit, form };
        say!(out, "  e.g. {}", ["dog", "dogs", "walks", "walked", "is", "are", "himself", "the", "ever", "Derek"].iter().map(|w| {
            let lw = lower(w);
            format!("{w}: {} / {}", lx.unit.get(&lw).map(|u| format!("unit {u}")).unwrap_or("own class".into()), bundle_name(lx.bundle(w)))
        }).collect::<Vec<_>>().join(", "));
        lx
    }
    fn ckey(&self, lw: &str) -> u64 {
        match self.unit.get(lw) { Some(&u) => mix64(0xc1a5_5000_0000 ^ u as u64), None => mix64(wkey(lw) ^ 0x0c1a_55ed_0000_0000) }
    }
    /// a word's form class: its analyses as written, else lower-cased; no analysis: unknown
    fn bundle(&self, w: &str) -> u8 { self.form.get(w).or_else(|| self.form.get(&lower(w))).copied().unwrap_or(UNKNOWN) }
    fn fkey(&self, w: &str) -> u64 { mix64(0xf0f0_0000_0000 ^ self.bundle(w) as u64) }
    fn cname(&self, lw: &str) -> String { match self.unit.get(lw) { Some(&u) => format!("u{u}"), None => format!("w:{lw}") } }
    fn fname(&self, w: &str) -> String { format!("b{}", self.bundle(w)) }
}

// ---------------------------------------------------------------- her counts
/// the keys of the contexts before position i, shortest first (1..=maxn; <s> before item 0).
/// A context is its items' ids + 1 in 21-bit slots, the nearest lowest.
fn ctx_keys(ids: &[u32], i: usize, maxn: usize, out: &mut Vec<u64>) {
    out.clear();
    let mut key = 0u64;
    for n in 1..=maxn.min(i + 1) {
        let h = if n <= i { ids[i - n] } else { BOS };
        if h == NONE { break; }
        key |= (h as u64 + 1) << (SLOT * (n - 1));
        out.push(key);
    }
}
fn seq_key(keys: &[u64]) -> u64 { keys.iter().fold(mix64(keys.len() as u64), |h, &k| mix64(h ^ k)) }

/// the any-thing level: counts of each kept item, how many items read, how many different
#[derive(Default)]
struct Uni { n: u64, types: MixSet<u64>, cnt: Vec<u64> }
impl Uni {
    fn add(&mut self, k: u64, id: u32) {
        self.n += 1;
        self.types.insert(k);
        if id != NONE { self.cnt[id as usize] += 1; }
    }
    /// an even share for everything read plus one never met, and Witten-Bell trust in the counts
    fn p0(&self, t: u32) -> f64 {
        let (nn, v) = (self.n as f64, self.types.len() as f64);
        let lam0 = if nn > 0.0 { nn / (nn + v) } else { 0.0 };
        let mut p = (1.0 - lam0) / (v + 1.0);
        if nn > 0.0 { p += lam0 * self.cnt[t as usize] as f64 / nn; }
        p
    }
}

/// a Witten-Bell n-gram that keeps only the contexts and pairs the scored sentences register
struct Ng { maxn: usize, salt: u64, ctx: MixMap<u64, u32>, c: Vec<u64>, d: Vec<u64>, follow: MixSet<u64>, pair: MixMap<u64, u64> }
impl Ng {
    fn new(maxn: usize, salt: u64) -> Ng {
        Ng { maxn, salt, ctx: MixMap::default(), c: Vec::new(), d: Vec::new(), follow: MixSet::default(), pair: MixMap::default() }
    }
    fn register(&mut self, ids: &[u32], i: usize, t: u32, buf: &mut Vec<u64>) {
        ctx_keys(ids, i, self.maxn, buf);
        for &k in buf.iter() {
            let next = self.ctx.len() as u32;
            let s = *self.ctx.entry(k).or_insert(next);
            self.pair.entry((s as u64) << 32 | t as u64).or_insert(0);
        }
    }
    fn ready(&mut self) { self.c = vec![0; self.ctx.len()]; self.d = vec![0; self.ctx.len()]; }
    fn count(&mut self, ids: &[u32], i: usize, tk: u64, tid: u32, buf: &mut Vec<u64>) {
        ctx_keys(ids, i, self.maxn, buf);
        for &k in buf.iter() {
            // a longer context holds this one, so if this one is not kept, neither is it
            let Some(&s) = self.ctx.get(&k) else { break };
            self.c[s as usize] += 1;
            if self.follow.insert(mix64(tk ^ self.salt ^ (s as u64 + 1).wrapping_mul(GOLD))) { self.d[s as usize] += 1; }
            if tid != NONE { if let Some(x) = self.pair.get_mut(&((s as u64) << 32 | tid as u64)) { *x += 1; } }
        }
    }
    /// from the level below (p), up through every context read
    fn prob(&self, ids: &[u32], i: usize, t: u32, mut p: f64, buf: &mut Vec<u64>) -> f64 {
        ctx_keys(ids, i, self.maxn, buf);
        for k in buf.iter() {
            if let Some(&s) = self.ctx.get(k) {
                let (c, d) = (self.c[s as usize] as f64, self.d[s as usize] as f64);
                if c > 0.0 {
                    let lam = c / (c + d);
                    let ct = self.pair.get(&((s as u64) << 32 | t as u64)).copied().unwrap_or(0) as f64;
                    p = lam * ct / c + (1.0 - lam) * p;
                }
            }
        }
        p
    }
}

/// a class reading (C or F): the class sequence (Witten-Bell), then the word within its class
struct ClassRead { uni: Uni, ng: Ng }
impl ClassRead {
    /// wn: how often she read the word itself (0: never met, so its class is the never-met one)
    fn p(&self, cids: &[u32], i: usize, ct: u32, wn: u64, buf: &mut Vec<u64>) -> f64 {
        if wn == 0 { return self.ng.prob(cids, i, NEW, self.uni.p0(NEW), buf); }
        self.ng.prob(cids, i, ct, self.uni.p0(ct), buf) * (wn as f64 / self.uni.cnt[ct as usize] as f64)
    }
}

/// working memory: for each earlier word u, how often anything came 4 or more places later (c),
/// how many different things did (d), and how often each scored x did
struct Mem { c: Vec<u64>, d: Vec<u64>, follow: MixSet<u64>, pair: MixMap<u64, u64>, distinct: bool }
fn mkey(u: u32, tk: u64) -> u64 { mix64(tk ^ 0x6d65_6d6f_7279_0000 ^ (u as u64 + 1).wrapping_mul(GOLD)) }
impl Mem {
    fn register(&mut self, ids: &[u32], i: usize, t: u32) {
        if i < GAP { return; }
        for j in 0..=i - GAP { self.pair.entry((ids[j] as u64) << 32 | t as u64).or_insert(0); }
    }
    fn count(&mut self, ids: &[u32], i: usize, tk: u64, tid: u32) {
        if i < GAP { return; }
        for j in 0..=i - GAP {
            let u = ids[j];
            if u == NONE { continue; }
            self.c[u as usize] += 1;
            if self.distinct && self.follow.insert(mkey(u, tk)) { self.d[u as usize] += 1; }
            if tid != NONE { if let Some(x) = self.pair.get_mut(&((u as u64) << 32 | tid as u64)) { *x += 1; } }
        }
    }
    /// u's own Witten-Bell estimate that t comes later, over the any-word level
    fn vote(&self, u: u32, t: u32, pany: f64) -> f64 {
        let (c, d) = (self.c[u as usize] as f64, self.d[u as usize] as f64);
        if c == 0.0 { return pany; }
        let lam = c / (c + d);
        let ct = self.pair.get(&((u as u64) << 32 | t as u64)).copied().unwrap_or(0) as f64;
        lam * ct / c + (1.0 - lam) * pany
    }
}

struct Sent { ids: Vec<u32>, keys: Vec<u64>, cids: Vec<u32>, fids: Vec<u32> }
#[derive(Clone, Copy)]
struct Tok { ck: u64, fk: u64, cid: u32, fid: u32 }
#[derive(Default)]
struct Scratch { keys: Vec<u64>, ids: Vec<u32>, cids: Vec<u32>, fids: Vec<u32>, cks: Vec<u64>, fks: Vec<u64>, buf: Vec<u64> }

struct Her<'a> {
    lex: &'a Lex,
    vocab: MixMap<u64, u32>,  // a registered word's key -> id (0 = <s>, 1 = </s>)
    cvocab: MixMap<u64, u32>, // a registered class -> id (0 <s>, 1 </s>, 2 never met)
    fvocab: MixMap<u64, u32>, // a registered form class -> id (the same)
    cache: MixMap<u64, Tok>,  // any word read -> its class and form
    sents: Vec<Sent>,         // the registered sentences: BLiMP (2k good, 2k+1 bad), then the held-out slice
    nscored: usize,
    whole: MixMap<u64, Vec<u32>>,
    wu: Uni, wn: Ng, c: ClassRead, f: ClassRead, m: Mem,
    words: u64, nsent: u64,
    seen: Vec<bool>,
    need: u8, // which of M (1), C (2), F (4) this run counts (W always)
}

impl<'a> Her<'a> {
    fn new(lex: &'a Lex, scored: &[Vec<String>], held: &[Vec<String>], m_distinct: bool, need: u8) -> Her<'a> {
        let mut h = Her {
            lex, vocab: MixMap::default(), cvocab: MixMap::default(), fvocab: MixMap::default(), cache: MixMap::default(),
            sents: Vec::with_capacity(scored.len() + held.len()), nscored: scored.len(), whole: MixMap::default(),
            wu: Uni::default(), wn: Ng::new(MAXN, 0),
            c: ClassRead { uni: Uni::default(), ng: Ng::new(MAXN, 0x00c1_a550_0000_0000) },
            f: ClassRead { uni: Uni::default(), ng: Ng::new(MAXN, 0x00f0_4a50_0000_0000) },
            m: Mem { c: Vec::new(), d: Vec::new(), follow: MixSet::default(), pair: MixMap::default(), distinct: m_distinct },
            words: 0, nsent: 0, seen: vec![false; scored.len()], need,
        };
        for w in scored.iter().chain(held.iter()) {
            let keys: Vec<u64> = w.iter().map(|x| wkey(x)).collect();
            let ids: Vec<u32> = keys.iter().map(|k| { let n = h.vocab.len() as u32 + 2; *h.vocab.entry(*k).or_insert(n) }).collect();
            let (mut cids, mut fids) = (Vec::with_capacity(w.len()), Vec::with_capacity(w.len()));
            for x in w {
                let lw = lower(x);
                let (ck, fk) = (lex.ckey(&lw), lex.fkey(x));
                let n = h.cvocab.len() as u32 + 3;
                cids.push(*h.cvocab.entry(ck).or_insert(n));
                let n = h.fvocab.len() as u32 + 3;
                fids.push(*h.fvocab.entry(fk).or_insert(n));
            }
            h.sents.push(Sent { ids, keys, cids, fids });
        }
        if h.vocab.len() + 4 >= (1 << SLOT) - 1 || h.cvocab.len() + 4 >= (1 << SLOT) - 1 { die("vocabulary too big for 21-bit context slots"); }
        for (si, s) in h.sents[..h.nscored].iter().enumerate() { h.whole.entry(seq_key(&s.keys)).or_default().push(si as u32); }
        let mut buf = Vec::new();
        for s in &h.sents {
            for i in 0..=s.ids.len() {
                let (t, ct, ft) = if i < s.ids.len() { (s.ids[i], s.cids[i], s.fids[i]) } else { (EOS, EOS, EOS) };
                h.wn.register(&s.ids, i, t, &mut buf);
                if need & 2 != 0 { h.c.ng.register(&s.cids, i, ct, &mut buf); }
                if need & 4 != 0 { h.f.ng.register(&s.fids, i, ft, &mut buf); }
                if need & 1 != 0 { h.m.register(&s.ids, i, t); }
            }
        }
        h.wn.ready();
        h.c.ng.ready();
        h.f.ng.ready();
        h.wu.cnt = vec![0; h.vocab.len() + 2];
        h.c.uni.cnt = vec![0; h.cvocab.len() + 3];
        h.f.uni.cnt = vec![0; h.fvocab.len() + 3];
        h.m.c = vec![0; h.vocab.len() + 2];
        h.m.d = vec![0; h.vocab.len() + 2];
        h
    }

    fn tok(&mut self, k: u64, w: &str) -> Tok {
        if let Some(t) = self.cache.get(&k) { return *t; }
        let lw = lower(w);
        let (ck, fk) = (self.lex.ckey(&lw), self.lex.fkey(w));
        let t = Tok { ck, fk, cid: self.cvocab.get(&ck).copied().unwrap_or(NONE), fid: self.fvocab.get(&fk).copied().unwrap_or(NONE) };
        self.cache.insert(k, t);
        t
    }

    fn prep(&mut self, w: &[String], sc: &mut Scratch) {
        sc.keys.clear(); sc.ids.clear(); sc.cids.clear(); sc.fids.clear(); sc.cks.clear(); sc.fks.clear();
        for x in w {
            let k = wkey(x);
            sc.keys.push(k);
            sc.ids.push(self.vocab.get(&k).copied().unwrap_or(NONE));
            let t = self.tok(k, x);
            sc.cks.push(t.ck); sc.fks.push(t.fk); sc.cids.push(t.cid); sc.fids.push(t.fid);
        }
    }

    /// read one sentence: every word and the end, in every reading
    fn read(&mut self, w: &[String], sc: &mut Scratch) {
        self.prep(w, sc);
        self.nsent += 1;
        self.words += w.len() as u64;
        let l = w.len();
        for i in 0..=l {
            let (tk, tid, ck, cid, fk, fid) = if i < l { (sc.keys[i], sc.ids[i], sc.cks[i], sc.cids[i], sc.fks[i], sc.fids[i]) } else { (EOS_KEY, EOS, EOS_KEY, EOS, EOS_KEY, EOS) };
            self.wu.add(tk, tid);
            self.wn.count(&sc.ids, i, tk, tid, &mut sc.buf);
            if self.need & 2 != 0 {
                self.c.uni.add(ck, cid);
                self.c.ng.count(&sc.cids, i, ck, cid, &mut sc.buf);
            }
            if self.need & 4 != 0 {
                self.f.uni.add(fk, fid);
                self.f.ng.count(&sc.fids, i, fk, fid, &mut sc.buf);
            }
            if self.need & 1 != 0 { self.m.count(&sc.ids, i, tk, tid); }
        }
        if let Some(list) = self.whole.get(&seq_key(&sc.keys)) {
            for &si in list { if self.sents[si as usize].keys == sc.keys { self.seen[si as usize] = true; } }
        }
    }

    /// M's different-later-words for the voters with id % shards == shard only (arc: memory in slices)
    fn mem_distinct_part(&mut self, w: &[String], sc: &mut Scratch, shards: u64, shard: u64) {
        sc.keys.clear(); sc.ids.clear();
        for x in w { let k = wkey(x); sc.keys.push(k); sc.ids.push(self.vocab.get(&k).copied().unwrap_or(NONE)); }
        let l = w.len();
        for i in GAP..=l {
            let tk = if i < l { sc.keys[i] } else { EOS_KEY };
            for j in 0..=i - GAP {
                let u = sc.ids[j];
                if u == NONE || u as u64 % shards != shard { continue; }
                if self.m.follow.insert(mkey(u, tk)) { self.m.d[u as usize] += 1; }
            }
        }
    }

    /// every reading's P of each word and end of the registered sentences lo..hi, as counted so far
    fn probs(&self, lo: usize, hi: usize) -> Probs {
        let mut p = Probs::default();
        let mut buf = Vec::new();
        for s in &self.sents[lo..hi] {
            p.soff.push(p.pw.len());
            let l = s.ids.len();
            for i in 0..=l {
                let (t, ct, ft) = if i < l { (s.ids[i], s.cids[i], s.fids[i]) } else { (EOS, EOS, EOS) };
                let pany = self.wu.p0(t);
                p.pw.push(self.wn.prob(&s.ids, i, t, pany, &mut buf));
                let wn = self.wu.cnt[t as usize];
                p.pc.push(if self.need & 2 != 0 { self.c.p(&s.cids, i, ct, wn, &mut buf) } else { f64::NAN });
                p.pf.push(if self.need & 4 != 0 { self.f.p(&s.fids, i, ft, wn, &mut buf) } else { f64::NAN });
                p.voff.push(p.age.len());
                if self.need & 1 != 0 && i >= GAP {
                    for j in 0..=i - GAP {
                        p.age.push((i - GAP - j) as u32);
                        p.pv.push(self.m.vote(s.ids[j], t, pany));
                    }
                }
            }
        }
        p.soff.push(p.pw.len());
        p.voff.push(p.age.len());
        p
    }

    /// W alone (WP6's reading): each scored sentence's sum of ln P
    fn w_scores(&self) -> Vec<f64> {
        let mut buf = Vec::new();
        self.sents[..self.nscored].iter().map(|s| {
            let mut sum = 0.0;
            for i in 0..=s.ids.len() {
                let t = if i < s.ids.len() { s.ids[i] } else { EOS };
                sum += self.wn.prob(&s.ids, i, t, self.wu.p0(t), &mut buf).ln();
            }
            sum
        }).collect()
    }
}

/// per token of some sentences: P under W, C, F, and M's voters (their ages and votes)
#[derive(Default)]
struct Probs { soff: Vec<usize>, pw: Vec<f64>, pc: Vec<f64>, pf: Vec<f64>, voff: Vec<usize>, age: Vec<u32>, pv: Vec<f64> }

/// M's P per token for a decay: votes weighted decay^age; -1 where M is silent (no voter)
fn mem_mix(p: &Probs, delta: f64) -> Vec<f64> {
    let maxage = p.age.iter().copied().max().unwrap_or(0) as usize;
    let mut pw = vec![1.0f64; maxage + 1];
    for a in 1..=maxage { pw[a] = pw[a - 1] * delta; }
    (0..p.pw.len()).map(|t| {
        let (a, b) = (p.voff[t], p.voff[t + 1]);
        if a == b { return -1.0; }
        let (mut num, mut den) = (0.0, 0.0);
        for v in a..b { let w = pw[p.age[v] as usize]; num += w * p.pv[v]; den += w; }
        num / den
    }).collect()
}

/// the decay learned from her reading: the one that makes M's own predictions of the held-out slice
/// most likely (a grid of 0, 0.05, .., 1, then golden-section search around the best)
fn learn_decay(held: &Probs) -> (f64, f64, usize) {
    let ll = |d: f64| -> f64 { mem_mix(held, d).iter().filter(|&&x| x >= 0.0).map(|x| x.ln()).sum() };
    let speaks = mem_mix(held, 0.5).iter().filter(|&&x| x >= 0.0).count();
    if speaks == 0 { return (1.0, 0.0, 0); }
    let mut best = (f64::NEG_INFINITY, 0.0);
    for k in 0..=20 { let d = k as f64 / 20.0; let v = ll(d); if v > best.0 { best = (v, d); } }
    let (mut a, mut b) = ((best.1 - 0.05f64).max(0.0), (best.1 + 0.05f64).min(1.0));
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let (mut x1, mut x2) = (b - g * (b - a), a + g * (b - a));
    let (mut f1, mut f2) = (ll(x1), ll(x2));
    for _ in 0..40 {
        if f1 >= f2 { b = x2; x2 = x1; f2 = f1; x1 = b - g * (b - a); f1 = ll(x1); } else { a = x1; x1 = x2; f1 = f2; x2 = a + g * (b - a); f2 = ll(x2); }
    }
    for (f, x) in [(f1, x1), (f2, x2)] { if f > best.0 { best = (f, x); } }
    (best.1, best.0 / speaks as f64, speaks)
}

/// the mixture's P of one token: the readings in the subset that speak, weighted, renormalised
fn mix_p(lam: &[f64; 4], set: u8, p: [f64; 4]) -> f64 {
    let (mut s, mut m) = (0.0, 0.0);
    for k in 0..4 { if on(set, k) && !(k == 1 && p[1] < 0.0) { s += lam[k]; m += lam[k] * p[k]; } }
    m / s
}

/// EM for the subset's weights on the held-out tokens (W always in). Returns weights, mean ln P, steps.
fn em(set: u8, pw: &[f64], pm: &[f64], pc: &[f64], pf: &[f64]) -> ([f64; 4], f64, usize) {
    let nk = (0..4).filter(|&k| on(set, k)).count() as f64;
    let mut lam = [0.0; 4];
    for k in 0..4 { if on(set, k) { lam[k] = 1.0 / nk; } }
    let n = pw.len() as f64;
    let mut prev = f64::NEG_INFINITY;
    for it in 0..EM_MAX_IT {
        let (mut num, mut den, mut ll) = ([0.0f64; 4], [0.0f64; 4], 0.0f64);
        for t in 0..pw.len() {
            let p = [pw[t], pm[t], pc[t], pf[t]];
            let (mut s, mut m) = (0.0, 0.0);
            for k in 0..4 { if on(set, k) && !(k == 1 && p[1] < 0.0) { s += lam[k]; m += lam[k] * p[k]; } }
            ll += (m / s).ln();
            for k in 0..4 { if on(set, k) && !(k == 1 && p[1] < 0.0) { num[k] += lam[k] * p[k] / m; den[k] += 1.0 / s; } }
        }
        if ll < prev - 1e-9 * prev.abs() { die(&format!("EM lowered the held-out likelihood ({} -> {} at step {it})", prev, ll)); }
        if ll - prev <= 1e-9 * n || set == 0 { return (lam, ll / n, it); }
        prev = ll;
        for k in 0..4 { if on(set, k) && den[k] > 0.0 { lam[k] = num[k] / den[k]; } }
        let z: f64 = lam.iter().sum();
        for x in lam.iter_mut() { *x /= z; }
    }
    die("EM did not settle in 5,000 steps")
}

/// each sentence's sum of ln P under a mixture
fn mix_scores(p: &Probs, pm: &[f64], lam: &[f64; 4], set: u8) -> Vec<f64> {
    (0..p.soff.len() - 1).map(|s| {
        let mut sum = 0.0;
        for t in p.soff[s]..p.soff[s + 1] { sum += mix_p(lam, set, [p.pw[t], pm[t], p.pc[t], p.pf[t]]).ln(); }
        sum
    }).collect()
}

// ---------------------------------------------------------------- one reading, all readings
struct Fit { set: u8, lam: [f64; 4], ll: f64, steps: usize }
struct MarkRes {
    label: String,
    x_words: u64, x_sents: usize, c_words: u64, c_sents: usize, h_sents: usize, h_tokens: usize,
    decay: f64, m_ll: f64, m_speaks: usize, w_ll: f64,
    fits: Vec<Fit>,
    /// per pair, good - bad: [k] for SUBSETS[k] (NaN if not run); [8] W alone on the counted part only
    margin: Vec<[f64; 9]>,
    seen: Vec<bool>, oov: u64, tok: u64, types: u64, follow: [u64; 4],
    arc_alone: Option<(Vec<f64>, u64, u64)>, // ARC corpus alone, W only: margins, words, sentences
}

/// the ARC corpus, line by line, as WP6 reads it (cased; an exam turn withheld)
fn each_arc(r: &Reading, f: &mut dyn FnMut(&[String])) -> (u64, [u64; 2]) {
    let fh = std::fs::File::open(ARC_CORPUS).unwrap_or_else(|e| die(&format!("{ARC_CORPUS}: {e}")));
    let mut rd = BufReader::with_capacity(1 << 20, fh);
    let (mut line, mut lines, mut skipped) = (Vec::new(), 0u64, [0u64; 2]);
    loop {
        line.clear();
        if rd.read_until(b'\n', &mut line).unwrap_or_else(|e| die(&format!("read: {e}"))) == 0 { break; }
        lines += 1;
        let s = String::from_utf8_lossy(&line);
        let w = words(&s, false);
        if w.is_empty() { skipped[0] += 1; continue; }
        if r.exam.contains(&words(&s, true)) { skipped[1] += 1; continue; }
        f(&w);
        if lines % 1_000_000 == 0 { ram_guard("ARC corpus"); }
    }
    (lines, skipped)
}

/// one reading: count its first 95% (ARC first, when asked), take every reading's P of the scored
/// sentences and of the held-out last 5%, learn the decay and each subset's weights on the held-out
/// slice, then read the held-out slice too for W alone (WP6's reading at the full mark).
fn run_mark(r: &Reading, lex: &Lex, scored: &[Vec<String>], x: Option<u64>, arc: bool, sets: &[u8], out: &mut String) -> MarkRes {
    let t0 = Instant::now();
    let total: u64 = r.lens.iter().map(|&l| l as u64).sum();
    let xw = x.unwrap_or(total);
    let (end_x, words_x) = prefix_end(&r.lens, xw);
    let cut = (xw as f64 * (1.0 - HOLD)).floor() as u64;
    let (end_c, words_c) = prefix_end(&r.lens, cut);
    let label = match (x, arc) { (_, true) => "ARC corpus + her reading".to_string(), (Some(v), _) => format!("{} words", th(v)), (None, _) => format!("all ({} words)", th(total)) };
    let held: Vec<Vec<String>> = r.sents[end_c..end_x].iter().map(|s| words(s, false)).collect();
    let need = sets.iter().fold(0u8, |a, &s| a | s);
    let mut her = Her::new(lex, scored, &held, !arc, need);
    let h_tokens: usize = held.iter().map(|w| w.len() + 1).sum();
    drop(held);
    println!("  [{label}] {} scored + {} held-out sentences registered: {} words, {} classes, {} forms; W {} / C {} / F {} contexts, M {} pairs ({:.1}s)",
             th(her.nscored as u64), th((her.sents.len() - her.nscored) as u64), th(her.vocab.len() as u64), th(her.cvocab.len() as u64), th(her.fvocab.len() as u64),
             th(her.wn.ctx.len() as u64), th(her.c.ng.ctx.len() as u64), th(her.f.ng.ctx.len() as u64), th(her.m.pair.len() as u64), t0.elapsed().as_secs_f64());
    let mut sc = Scratch::default();
    let mut arc_alone = None;
    if arc {
        let (lines, sk) = each_arc(r, &mut |w: &[String]| her.read(w, &mut sc));
        say!(out, "  ARC corpus: {} lines, {} with no word, {} equal to an exam turn; {} sentences, {} words read in {:.0}s",
             th(lines), th(sk[0]), th(sk[1]), th(her.nsent), th(her.words), t0.elapsed().as_secs_f64());
        let w = her.w_scores();
        let m: Vec<f64> = (0..w.len() / 2).map(|k| w[2 * k] - w[2 * k + 1]).collect();
        arc_alone = Some((m, her.words, her.nsent));
    }
    for (k, s) in r.sents[..end_c].iter().enumerate() {
        her.read(&words(s, false), &mut sc);
        if (k + 1) % 500_000 == 0 { ram_guard("reading"); }
    }
    if arc && need & 1 != 0 {
        // M's different-later-words, in slices of the voters, over the same counted reading
        her.m.d.iter_mut().for_each(|x| *x = 0);
        for shard in 0..ARC_SHARDS {
            her.m.follow = MixSet::default();
            let her_ref = &mut her;
            each_arc(r, &mut |w: &[String]| her_ref.mem_distinct_part(w, &mut sc, ARC_SHARDS, shard));
            for s in &r.sents[..end_c] { her.mem_distinct_part(&words(s, false), &mut sc, ARC_SHARDS, shard); }
            println!("    M slice {}/{ARC_SHARDS}: {} pairs, {:.0}s, peak {:.2} GB, free {:.1} GB", shard + 1, th(her.m.follow.len() as u64), t0.elapsed().as_secs_f64(), peak_gb(), free_gb());
        }
        her.m.follow = MixSet::default();
    }
    let follow = [her.wn.follow.len() as u64, her.m.follow.len() as u64, her.c.ng.follow.len() as u64, her.f.ng.follow.len() as u64];
    let (c_words, c_sents) = (her.words, her.nsent as usize);
    let pb = her.probs(0, her.nscored);
    let ph = her.probs(her.nscored, her.sents.len());
    let npairs = her.nscored / 2;
    let w95: Vec<f64> = (0..her.nscored).map(|s| { let mut x = 0.0; for t in pb.soff[s]..pb.soff[s + 1] { x += pb.pw[t].ln(); } x }).collect();
    for s in &r.sents[end_c..end_x] { her.read(&words(s, false), &mut sc); }
    let wfull = her.w_scores();
    let (mut oov, mut tok) = (0u64, 0u64);
    for s in &her.sents[..her.nscored] { for &i in &s.ids { tok += 1; if her.wu.cnt[i as usize] == 0 { oov += 1; } } }
    let types = her.wu.types.len() as u64;
    let seen = her.seen.clone();
    let (x_words, x_sents) = (her.words, her.nsent as usize);
    if !arc && (x_words != words_x || c_words != words_c) { die("reading marks disagree"); }
    let read_secs = t0.elapsed().as_secs_f64();
    drop(her);

    // the decay, then each subset's weights, all on the held-out slice
    let (decay, m_ll, m_speaks) = if need & 1 != 0 { learn_decay(&ph) } else { (f64::NAN, f64::NAN, 0) };
    let hm = mem_mix(&ph, decay);
    let bm = mem_mix(&pb, decay);
    let w_ll = ph.pw.iter().map(|x| x.ln()).sum::<f64>() / ph.pw.len().max(1) as f64;
    let todo: Vec<u8> = sets.iter().copied().filter(|&s| s != 0).collect();
    let mut fits: Vec<Fit> = Vec::new();
    for chunk in todo.chunks(4) {
        let got: Vec<Fit> = std::thread::scope(|scope| {
            let hs: Vec<_> = chunk.iter().map(|&set| {
                let (ph, hm) = (&ph, &hm);
                scope.spawn(move || { let (lam, ll, steps) = em(set, &ph.pw, hm, &ph.pc, &ph.pf); Fit { set, lam, ll, steps } })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| die("EM thread failed"))).collect()
        });
        fits.extend(got);
    }
    let mut margin = vec![[f64::NAN; 9]; npairs];
    for k in 0..npairs { margin[k][8] = w95[2 * k] - w95[2 * k + 1]; }
    for (ki, &set) in SUBSETS.iter().enumerate() {
        if !sets.contains(&set) { continue; }
        if set == 0 {
            for k in 0..npairs { margin[k][ki] = wfull[2 * k] - wfull[2 * k + 1]; }
        } else {
            let f = fits.iter().find(|f| f.set == set).unwrap();
            let sc = mix_scores(&pb, &bm, &f.lam, set);
            for k in 0..npairs { margin[k][ki] = sc[2 * k] - sc[2 * k + 1]; }
        }
    }
    println!("  [{label}] counted {} words ({} sentences), held out {} sentences ({} tokens), read to {} words; {:.1}s reading, {:.1}s in all; peak RAM {:.2} GB",
             th(c_words), th(c_sents as u64), th((end_x - end_c) as u64), th(h_tokens as u64), th(x_words), read_secs, t0.elapsed().as_secs_f64(), peak_gb());
    MarkRes { label, x_words, x_sents, c_words, c_sents, h_sents: end_x - end_c, h_tokens, decay, m_ll, m_speaks, w_ll, fits, margin, seen, oov, tok, types, follow, arc_alone }
}

/// (right, ties, pairs) over the given pairs, for column k of the margins
fn tally(m: &[[f64; 9]], which: impl Iterator<Item = usize>, k: usize) -> (u64, u64, u64) {
    let (mut r, mut t, mut n) = (0, 0, 0);
    for i in which {
        let x = m[i][k];
        n += 1;
        if x.abs() <= TIE { t += 1; } else if x > 0.0 { r += 1; }
    }
    (r, t, n)
}

/// paired bootstrap of a per-pair difference (sum), 20,000 resamples, splitmix64 seed 1: 95% interval
fn boot(diff: &[i64]) -> (i64, i64) {
    let n = diff.len();
    let mut x: u64 = 1;
    let mut b: Vec<i64> = (0..20_000).map(|_| (0..n).map(|_| diff[(splitmix(&mut x) % n as u64) as usize]).sum()).collect();
    b.sort_unstable();
    (b[500], b[19_499])
}

fn scored_words(pairs: &[&Pair]) -> Vec<Vec<String>> {
    pairs.iter().flat_map(|p| [words(&p.good, false), words(&p.bad, false)]).collect()
}

fn fit_line(res: &MarkRes, set: u8) -> String {
    match res.fits.iter().find(|f| f.set == set) {
        None => format!("{:<9} (no weights: W alone)", set_name(set)),
        Some(f) => format!("{:<9} {}  held-out ln P/token {:.4} ({} EM steps)", set_name(set),
                           (0..4).map(|k| if on(set, k) { format!("{} {:.4}", NAMES[k], f.lam[k]) } else { format!("{} ------", NAMES[k]) }).collect::<Vec<_>>().join("  "),
                           f.ll, f.steps),
    }
}

// ---------------------------------------------------------------- check: streaming == storing everything
/// every n-gram stored, the obvious way, for `check`
struct NaiveNg { uni: HashMap<String, u64>, tab: HashMap<String, HashMap<String, u64>>, n: u64 }
impl NaiveNg {
    fn new() -> NaiveNg { NaiveNg { uni: HashMap::new(), tab: HashMap::new(), n: 0 } }
    /// a sentence's items (words, classes or forms), <s> first, </s> last
    fn add(&mut self, items: &[String]) {
        for i in 1..items.len() {
            self.n += 1;
            *self.uni.entry(items[i].clone()).or_insert(0) += 1;
            for n in 1..=MAXN.min(i) {
                *self.tab.entry(items[i - n..i].join(" ")).or_default().entry(items[i].clone()).or_insert(0) += 1;
            }
        }
    }
    fn p0(&self, t: &str) -> f64 {
        let (nn, v) = (self.n as f64, self.uni.len() as f64);
        let l0 = nn / (nn + v);
        l0 * *self.uni.get(t).unwrap_or(&0) as f64 / nn + (1.0 - l0) / (v + 1.0)
    }
    fn p(&self, hist: &[String], t: &str) -> f64 {
        let mut p = self.p0(t);
        for n in 1..=MAXN {
            if n <= hist.len() {
                if let Some(m) = self.tab.get(&hist[hist.len() - n..].join(" ")) {
                    let c: u64 = m.values().sum();
                    let lam = c as f64 / (c as f64 + m.len() as f64);
                    p = lam * *m.get(t).unwrap_or(&0) as f64 / c as f64 + (1.0 - lam) * p;
                }
            }
        }
        p
    }
}
struct Naive { w: NaiveNg, c: NaiveNg, f: NaiveNg, m: HashMap<String, HashMap<String, u64>> }
const NEVER: &str = "\u{1}never met";
impl Naive {
    fn items(lex: &Lex, s: &[String]) -> [Vec<String>; 3] {
        let mut w = vec!["<s>".to_string()];
        let mut c = vec!["<s>".to_string()];
        let mut f = vec!["<s>".to_string()];
        for x in s { let lw = lower(x); w.push(x.clone()); c.push(lex.cname(&lw)); f.push(lex.fname(x)); }
        w.push("</s>".into()); c.push("</s>".into()); f.push("</s>".into());
        [w, c, f]
    }
    fn new(lex: &Lex, sents: &[Vec<String>]) -> Naive {
        let mut nv = Naive { w: NaiveNg::new(), c: NaiveNg::new(), f: NaiveNg::new(), m: HashMap::new() };
        for s in sents {
            let [w, c, f] = Naive::items(lex, s);
            nv.w.add(&w); nv.c.add(&c); nv.f.add(&f);
            // working memory: u at word j, x at item i (word or end), i - j >= 4
            for i in GAP..=s.len() {
                let x = if i < s.len() { s[i].as_str() } else { "</s>" };
                for j in 0..=i - GAP { *nv.m.entry(s[j].clone()).or_default().entry(x.to_string()).or_insert(0) += 1; }
            }
        }
        nv
    }
    /// every reading's P of word t (or "</s>") after the sentence's first words `prev`; M = -1 when silent
    fn all(&self, lex: &Lex, prev: &[String], t: &str, decay: f64) -> ([f64; 4], Vec<(u32, f64)>) {
        let [w, c, f] = Naive::items(lex, prev);
        let (w, c, f) = (&w[..w.len() - 1], &c[..c.len() - 1], &f[..f.len() - 1]);
        let pw = self.w.p(w, t);
        let read = self.w.uni.get(t).copied().unwrap_or(0) > 0;
        let (tc, tf) = if t == "</s>" { ("</s>".to_string(), "</s>".to_string()) } else { let lt = lower(t); (lex.cname(&lt), lex.fname(t)) };
        let cls = |ng: &NaiveNg, hist: &[String], tc: &str| -> f64 {
            if !read { return ng.p(hist, NEVER); }
            ng.p(hist, tc) * (self.w.uni[t] as f64 / ng.uni[tc] as f64)
        };
        let pc = cls(&self.c, c, &tc);
        let pf = cls(&self.f, f, &tf);
        let i = prev.len();
        let pany = self.w.p0(t);
        let mut votes = Vec::new();
        if i >= GAP {
            for j in 0..=i - GAP {
                let v = match self.m.get(&prev[j]) {
                    None => pany,
                    Some(m) => {
                        let cu: u64 = m.values().sum();
                        let lam = cu as f64 / (cu as f64 + m.len() as f64);
                        lam * *m.get(t).unwrap_or(&0) as f64 / cu as f64 + (1.0 - lam) * pany
                    }
                };
                votes.push(((i - GAP - j) as u32, v));
            }
        }
        let pm = if votes.is_empty() { -1.0 } else {
            let (mut a, mut b) = (0.0, 0.0);
            for &(age, v) in &votes { let w = decay.powi(age as i32); a += w * v; b += w; }
            a / b
        };
        ([pw, pm, pc, pf], votes)
    }
}

fn check(pairs: &[Pair]) {
    let mut out = String::new();
    say!(out, "source sha256 (compiled in): {}", sha256(SOURCE.as_bytes()));
    let r = reading(&mut out);
    let lex = Lex::load(&mut out);
    let dev: Vec<&Pair> = pairs.iter().filter(|p| p.id % 5 == 0).take(1000).collect();
    let slice: Vec<Vec<String>> = r.sents.iter().take(30_000).map(|s| words(s, false)).collect();
    let cut = 28_500;
    let scored = scored_words(&dev);
    let mut her = Her::new(&lex, &scored, &slice[cut..], true, 7);
    let mut sc = Scratch::default();
    for w in &slice[..cut] { her.read(w, &mut sc); }
    let pb = her.probs(0, her.nscored);
    let ph = her.probs(her.nscored, her.sents.len());
    let nv = Naive::new(&lex, &slice[..cut]);
    say!(out, "cased slice: first {} sentences counted ({} words), the next {} held out; {} scored sentences; N={} V={} vs N={} V={}",
         th(cut as u64), th(her.words), th((slice.len() - cut) as u64), th(her.nscored as u64), her.wu.n, her.wu.types.len(), nv.w.n, nv.w.uni.len());
    // the decay and weights the slice teaches (to check M and the mixtures at a learned point)
    let (decay, _, _) = learn_decay(&ph);
    let hm = mem_mix(&ph, decay);
    let fits: Vec<(u8, [f64; 4])> = SUBSETS.iter().filter(|&&s| s != 0).map(|&s| { let (lam, _, _) = em(s, &ph.pw, &hm, &ph.pc, &ph.pf); (s, lam) }).collect();
    let mut worst = [0f64; 6]; // W, M votes, M mixed, C, F, full mixture
    let mut ntok = 0usize;
    let full = fits.iter().find(|f| f.0 == 7).unwrap().1;
    for (ws, p) in [(&scored[..], &pb), (&slice[cut..], &ph)] {
        let pm = mem_mix(p, decay);
        for (si, w) in ws.iter().enumerate() {
            for i in 0..=w.len() {
                let t = if i < w.len() { w[i].as_str() } else { "</s>" };
                let tix = p.soff[si] + i;
                let (q, votes) = nv.all(&lex, &w[..i], t, decay);
                let a = [p.pw[tix], pm[tix], p.pc[tix], p.pf[tix]];
                worst[0] = worst[0].max((a[0].ln() - q[0].ln()).abs());
                if (a[1] < 0.0) != (q[1] < 0.0) { die("M speaks in one count and not the other"); }
                if a[1] >= 0.0 { worst[2] = worst[2].max((a[1].ln() - q[1].ln()).abs()); }
                worst[3] = worst[3].max((a[2].ln() - q[2].ln()).abs());
                worst[4] = worst[4].max((a[3].ln() - q[3].ln()).abs());
                let (va, vb) = (p.voff[tix], p.voff[tix + 1]);
                if vb - va != votes.len() { die("different numbers of voters"); }
                for (k, &(age, v)) in votes.iter().enumerate() {
                    if p.age[va + k] != age { die("voter ages differ"); }
                    worst[1] = worst[1].max((p.pv[va + k].ln() - v.ln()).abs());
                }
                worst[5] = worst[5].max((mix_p(&full, 7, a).ln() - mix_p(&full, 7, q).ln()).abs());
                ntok += 1;
            }
        }
    }
    say!(out, "streamed vs stored, largest |difference| in ln P over {} tokens (BLiMP + held-out): W {:.3e}, M votes {:.3e}, M {:.3e}, C {:.3e}, F {:.3e}, W+M+C+F {:.3e}",
         th(ntok as u64), worst[0], worst[1], worst[2], worst[3], worst[4], worst[5]);
    // W alone at the whole slice, against a store of the whole slice (WP6's check)
    for w in &slice[cut..] { her.read(w, &mut sc); }
    let ws = her.w_scores();
    let nv_all = Naive::new(&lex, &slice);
    let mut wworst = 0f64;
    for (si, w) in scored.iter().enumerate() {
        let mut s = 0.0;
        for i in 0..=w.len() { let t = if i < w.len() { w[i].as_str() } else { "</s>" }; s += nv_all.all(&lex, &w[..i], t, 0.5).0[0].ln(); }
        wworst = wworst.max((s - ws[si]).abs());
    }
    say!(out, "W alone over the whole slice, largest |difference| in a sentence's ln P: {wworst:.3e}");
    // M's different-later-words in slices (the arc path) equal counting them at once
    let mut h2 = Her::new(&lex, &scored, &slice[cut..], false, 7);
    for w in &slice[..cut] { h2.read(w, &mut sc); }
    for shard in 0..3 { h2.m.follow = MixSet::default(); for w in &slice[..cut] { h2.mem_distinct_part(w, &mut sc, 3, shard); } }
    let mut h1 = Her::new(&lex, &scored, &slice[cut..], true, 7);
    for w in &slice[..cut] { h1.read(w, &mut sc); }
    let dd = h1.m.d.iter().zip(&h2.m.d).filter(|(a, b)| a != b).count();
    say!(out, "M's different later words counted in 3 slices vs at once: {dd} of {} voters differ (sum {} vs {})", th(h1.m.d.len() as u64), h2.m.d.iter().sum::<u64>(), h1.m.d.iter().sum::<u64>());
    if dd > 0 || worst.iter().any(|&x| x > 1e-9) || wworst > 1e-9 { say!(out, "!! CHECK FAILED"); }
    // every reading and every mixture is a distribution: over every word read + one never met, it sums to 1
    say!(out, "slice decay {decay:.4}; slice weights: {}", fits.iter().map(|(s, l)| format!("{} [{}]", set_name(*s), l.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(" "))).collect::<Vec<_>>().join(", "));
    let mut vocab: Vec<String> = nv.w.uni.keys().cloned().collect();
    vocab.sort();
    vocab.push(NEVER.to_string());
    say!(out, "sum of P over every word read ({}) + the end + one never met:", th(vocab.len() as u64 - 2));
    let mut worst_sum = 0f64;
    let mut shown = 0;
    for p in dev.iter() {
        let w = words(&p.good, false);
        if w.len() < 7 { continue; }
        for i in [2usize, 6] {
            let mut sums = [0f64; 4];
            let mut msum = vec![0f64; fits.len()];
            for t in &vocab {
                let (q, _) = nv.all(&lex, &w[..i], t, decay);
                for k in 0..4 { if !(k == 1 && q[1] < 0.0) { sums[k] += q[k]; } }
                for (fi, (s, lam)) in fits.iter().enumerate() { msum[fi] += mix_p(lam, *s, q); }
            }
            let silent = i < GAP;
            for k in 0..4 { if !(k == 1 && silent) { worst_sum = worst_sum.max((sums[k] - 1.0).abs()); } }
            for x in &msum { worst_sum = worst_sum.max((x - 1.0).abs()); }
            say!(out, "  after \"{}\" (M {}): W {:.12} M {} C {:.12} F {:.12}; mixtures {}", w[..i].join(" "), if silent { "silent" } else { "speaks" },
                 sums[0], if silent { "------".to_string() } else { format!("{:.12}", sums[1]) }, sums[2], sums[3],
                 msum.iter().map(|x| format!("{x:.12}")).collect::<Vec<_>>().join(" "));
        }
        shown += 1;
        if shown == 5 { break; }
    }
    say!(out, "largest |sum - 1| over every reading and mixture: {worst_sum:.3e}{}", if worst_sum > 1e-9 { "  !! NOT A DISTRIBUTION" } else { "" });
}

// ---------------------------------------------------------------- main
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(|s| s.as_str()).unwrap_or("");
    let t0 = Instant::now();
    let (pairs, uids, terms) = load_blimp();
    if pairs.len() != 67_000 || uids.len() != 67 { die(&format!("expected 67 sets x 1,000 pairs, found {} sets, {} pairs", uids.len(), pairs.len())); }
    match mode {
        "check" => check(&pairs),
        "dev" => dev(&pairs, a.get(2).map(|s| s.as_str())),
        "test" | "arc" | "dryrun" => {
            let path = a.get(2).unwrap_or_else(|| die("usage: blimp_wm test|arc|dryrun <report.txt>"));
            if FROZEN_SUBSET > 7 { die("FROZEN_SUBSET is not set: make the dev choice first"); }
            // the test is scored once: this build never writes over a report
            let mut fh = std::fs::OpenOptions::new().write(true).create_new(true).open(path)
                .unwrap_or_else(|e| die(&format!("refusing: {path} exists or cannot be created ({e}); the test is scored once")));
            let rep = test(&pairs, &uids, &terms, mode == "arc", mode == "dryrun");
            fh.write_all(rep.as_bytes()).unwrap_or_else(|e| die(&format!("write {path}: {e}")));
        }
        _ => die("usage: blimp_wm check | dev [devlog.md] | dryrun <report.txt> | test <report.txt> | arc <report.txt>"),
    }
    println!("total {:.1}s, peak RAM {:.2} GB", t0.elapsed().as_secs_f64(), peak_gb());
}

fn dev(pairs: &[Pair], log: Option<&str>) {
    let t0 = Instant::now();
    let mut out = String::new();
    say!(out, "source sha256 (compiled in): {}", sha256(SOURCE.as_bytes()));
    let dev: Vec<&Pair> = pairs.iter().filter(|p| p.id % 5 == 0).collect();
    say!(out, "DEV: {} pairs (pairID % 5 == 0); test pairs are not read", th(dev.len() as u64));
    let r = reading(&mut out);
    let lex = Lex::load(&mut out);
    let scored = scored_words(&dev);
    let mut res: Vec<MarkRes> = Vec::new();
    for x in MARKS.iter().map(|&m| Some(m)).chain([None]) {
        res.push(run_mark(&r, &lex, &scored, x, false, &SUBSETS, &mut out));
    }
    let n = dev.len();
    say!(out, "\nDEV, share of pairs right (ties, counted wrong). W alone counts its whole reading (WP6's reading); a mixture counts the first 95% and learns its weights on the last 5%.");
    let mut head = format!("  {:<24}", "reading");
    for &s in &SUBSETS { head.push_str(&format!(" {:>15}", set_name(s))); }
    head.push_str(&format!(" | {:>15}", "W, 95% counted"));
    say!(out, "{head}");
    let mut best: Option<(u64, usize, usize)> = None; // right, readings, index in SUBSETS
    for m in &res {
        let mut row = format!("  {:<24}", m.label);
        for k in 0..9 {
            let (rt, ti, kk) = tally(&m.margin, 0..n, k);
            if k == 8 { row.push_str(" |"); }
            row.push_str(&format!(" {:>7.2}% ({:>3} t)", pct(rt, kk), ti));
            if k < 8 && m.label.starts_with("10,000,000") {
                // the rule, fixed before the dev run: best at 10M; a tie goes to fewer readings, then the earlier in SUBSETS
                let nr = SUBSETS[k].count_ones() as usize;
                let better = match best { None => true, Some((b, bn, _)) => rt > b || (rt == b && nr < bn) };
                if better { best = Some((rt, nr, k)); }
            }
        }
        say!(out, "{row}");
    }
    say!(out, "\nlearned from the reading (held-out last 5%):");
    for m in &res {
        say!(out, "  {}: counted {} words, held out {} sentences / {} tokens; decay {:.4} (M speaks at {} held-out tokens, its own ln P/token {:.4}); W alone ln P/token {:.4}",
             m.label, th(m.c_words), th(m.h_sents as u64), th(m.h_tokens as u64), m.decay, th(m.m_speaks as u64), m.m_ll, m.w_ll);
        for &s in &SUBSETS { say!(out, "    {}", fit_line(m, s)); }
    }
    say!(out, "\nunseen BLiMP words and verbatim reading (dev):");
    for m in &res {
        let gs = (0..n).filter(|&k| m.seen[2 * k]).count();
        let bs = (0..n).filter(|&k| m.seen[2 * k + 1]).count();
        say!(out, "  {:<24} unseen {:.2}% of BLiMP word tokens; read word for word: {gs} good / {bs} bad; {} types; follower pairs W {} M {} C {} F {}",
             m.label, pct(m.oov, m.tok), th(m.types), th(m.follow[0]), th(m.follow[1]), th(m.follow[2]), th(m.follow[3]));
    }
    if let Some((rt, _, k)) = best {
        say!(out, "\nDEV CHOICE (best at 10M words; a tie goes to fewer readings, then the earlier in {}): {} = {}/{} = {:.2}%",
             SUBSETS.iter().map(|&s| set_name(s)).collect::<Vec<_>>().join(", "), set_name(SUBSETS[k]), rt, n, pct(rt, n as u64));
    }
    say!(out, "dev run {:.1}s, peak RAM {:.2} GB", t0.elapsed().as_secs_f64(), peak_gb());
    if let Some(path) = log {
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let mut fh = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
        let txt = format!("\n## dev run (unix time {stamp}), source sha256 {}\n\n```\n{out}```\n", sha256(SOURCE.as_bytes()));
        fh.write_all(txt.as_bytes()).unwrap_or_else(|e| die(&format!("write {path}: {e}")));
    }
}

/// the frozen routine: on test (once), or on dev (`dry`: the same code, run first so the one test
/// run cannot stop half-way; nothing is chosen from it)
fn test(pairs: &[Pair], uids: &[String], terms: &[String], arc: bool, dry: bool) -> String {
    let mut out = String::new();
    let fz = FROZEN_SUBSET;
    let kf = SUBSETS.iter().position(|&s| s == fz).unwrap_or_else(|| die("FROZEN_SUBSET is not one of the 8 subsets"));
    let what = if dry { "DRY RUN of the test routine, on DEV" } else if arc { "REPORT ONLY: the ARC science corpus as a bigger reading (test pairs)" } else { "TEST (scored once)" };
    say!(out, "BLiMP WP6b {what} -- frozen build, source sha256 (compiled in): {}", sha256(SOURCE.as_bytes()));
    say!(out, "frozen: {} (W = WP6's frozen reading, 3 words, cased); weights by EM and M's decay learned from the held-out last 5% of the reading; ties (|good - bad| <= {TIE:e} nats) count wrong", set_name(fz));
    let test: Vec<&Pair> = pairs.iter().filter(|p| (p.id % 5 == 0) == dry).collect();
    say!(out, "{}: {} pairs (pairID % 5 {} 0)", if dry { "dev" } else { "test" }, th(test.len() as u64), if dry { "==" } else { "!=" });
    let r = reading(&mut out);
    let lex = Lex::load(&mut out);
    let scored = scored_words(&test);
    let sets: Vec<u8> = if fz == 0 { vec![0] } else { vec![0, fz] };
    let n = test.len();
    let all = || 0..n;
    let mut res: Vec<MarkRes> = Vec::new();
    if arc {
        res.push(run_mark(&r, &lex, &scored, None, true, &sets, &mut out));
    } else {
        for x in MARKS.iter().map(|&m| Some(m)).chain([None]) { res.push(run_mark(&r, &lex, &scored, x, false, &sets, &mut out)); }
    }

    say!(out, "\n{}", if arc { "READINGS (report only, no bar)" } else { "LEARNING CURVE" });
    say!(out, "  {:<34} {:>12} {:>12} {:>8} {:>16} {:>16} {:>9} {:>10}", "reading", "words read", "counted", "decay", set_name(fz), "W alone (WP6)", "unseen", "types");
    for m in &res {
        let (rt, ti, k) = tally(&m.margin, all(), kf);
        let (wr, wt, _) = tally(&m.margin, all(), 0);
        say!(out, "  {:<34} {:>12} {:>12} {:>8} {:>8.2}% ({:>3} t) {:>8.2}% ({:>3} t) {:>8.2}% {:>10}", m.label, th(m.x_words), th(m.c_words), dec(m.decay),
             pct(rt, k), ti, pct(wr, k), wt, pct(m.oov, m.tok), th(m.types));
        if let Some((am, aw, asn)) = &m.arc_alone {
            let (r0, t0, _) = am.iter().fold((0u64, 0u64, 0u64), |(r, t, k), &x| if x.abs() <= TIE { (r, t + 1, k + 1) } else if x > 0.0 { (r + 1, t, k + 1) } else { (r, t, k + 1) });
            say!(out, "  {:<34} {:>12} {:>12} {:>8} {:>16} {:>8.2}% ({:>3} t)   ({} sentences)", "ARC corpus alone (W only)", th(*aw), th(*aw), "", "", pct(r0, n as u64), t0, th(*asn));
        }
    }
    say!(out, "\nweights learned from the reading (EM on the held-out last 5%; BLiMP never used):");
    for m in &res {
        say!(out, "  {:<34} decay {}; {}", m.label, dec(m.decay), fit_line(m, fz));
    }
    let mains: Vec<&MarkRes> = res.iter().filter(|m| arc || m.label.starts_with("10,000,000")).collect();
    for m in mains {
        let (rt, ti, k) = tally(&m.margin, all(), kf);
        let (wr, wt, _) = tally(&m.margin, all(), 0);
        let acc = pct(rt, k);
        let right = |i: usize, c: usize| m.margin[i][c] > TIE;
        say!(out, "\n== {} ==", m.label);
        say!(out, "{}: {rt}/{k} right = {acc:.2}%, {ti} ties (counted wrong)", set_name(fz));
        say!(out, "W alone (WP6's frozen reading): {wr}/{k} right = {:.2}%, {wt} ties", pct(wr, k));
        let vs_chance: Vec<i64> = (0..n).map(|i| if right(i, kf) { 1 } else { -1 }).collect();
        let (lo, hi) = boot(&vs_chance);
        say!(out, "{} - chance: {:+.1} pairs ({:+.2} points), 95% CI {:+.1} .. {:+.1} pairs (paired bootstrap, 20,000, seed 1)",
             set_name(fz), rt as f64 - k as f64 / 2.0, acc - 50.0, lo as f64 / 2.0, hi as f64 / 2.0);
        let diff: Vec<i64> = (0..n).map(|i| right(i, kf) as i64 - right(i, 0) as i64).collect();
        let (dlo, dhi) = boot(&diff);
        say!(out, "{} - W alone: {:+} pairs ({:+.2} points), 95% CI {:+} .. {:+} pairs ({:+.2} .. {:+.2} points) (paired bootstrap, 20,000, seed 1)",
             set_name(fz), rt as i64 - wr as i64, pct(rt, k) - pct(wr, k), dlo, dhi, dlo as f64 / k as f64 * 100.0, dhi as f64 / k as f64 * 100.0);
        if !arc {
            let r1 = acc >= 60.0;
            let r2 = rt > wr && dlo > 0;
            say!(out, "RULE 1  overall >= 60.0%:                          {acc:.2}%  -> {}", if r1 { "PASS" } else { "FAIL" });
            say!(out, "RULE 2  higher than WP6's reading, CI above 0:     {:+} pairs, lower bound {:+} -> {}", rt as i64 - wr as i64, dlo, if r2 { "PASS" } else { "FAIL" });
            say!(out, "VERDICT: {}", if r1 && r2 { "PASS" } else { "FAIL" });
        }
        say!(out, "\nphenomena (linguistics_term)          pairs  {:>9}   ties   W alone   ties", set_name(fz));
        let mut rows: Vec<(String, usize)> = terms.iter().enumerate().map(|(t, name)| (name.clone(), t)).collect();
        rows.sort();
        for (name, t) in &rows {
            let (a, at, kk) = tally(&m.margin, all().filter(|&i| test[i].term == *t), kf);
            let (b, bt, _) = tally(&m.margin, all().filter(|&i| test[i].term == *t), 0);
            say!(out, "  {name:<34} {kk:>6} {:>8.2}% {at:>6} {:>8.2}% {bt:>6}", pct(a, kk), pct(b, kk));
        }
        if let (Some(a0), Some(s0)) = (terms.iter().position(|t| t == "argument_structure"), terms.iter().position(|t| t == "s-selection")) {
            let f = |i: &usize| test[*i].term == a0 || test[*i].term == s0;
            let (a, at, kk) = tally(&m.margin, all().filter(f), kf);
            let (b, bt, _) = tally(&m.margin, all().filter(f), 0);
            say!(out, "  {:<34} {kk:>6} {:>8.2}% {at:>6} {:>8.2}% {bt:>6}", "(paper's 12) argument structure + s-sel.", pct(a, kk), pct(b, kk));
        }
        let mut sets67: Vec<(f64, String, u64, u64, f64, u64, usize)> = uids.iter().enumerate().map(|(u, name)| {
            let (a, at, kk) = tally(&m.margin, all().filter(|&i| test[i].uid == u), kf);
            let (b, _, _) = tally(&m.margin, all().filter(|&i| test[i].uid == u), 0);
            (pct(a, kk), name.clone(), kk, at, pct(b, kk), 0, u)
        }).collect();
        sets67.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        say!(out, "\nall 67 sets, best first                                       pairs {:>9}   ties   W alone  phenomenon", set_name(fz));
        for (acc, name, kk, at, wacc, _, u) in &sets67 {
            let t = test.iter().find(|p| p.uid == *u).map(|p| terms[p.term].as_str()).unwrap_or("");
            say!(out, "  {name:<58} {kk:>6} {acc:>8.2}% {at:>6} {wacc:>8.2}%  {t}");
        }
        let gs = (0..n).filter(|&i| m.seen[2 * i]).count() as u64;
        let bs = (0..n).filter(|&i| m.seen[2 * i + 1]).count() as u64;
        let only_g = (0..n).filter(|&i| m.seen[2 * i] && !m.seen[2 * i + 1]).count();
        let only_b = (0..n).filter(|&i| !m.seen[2 * i] && m.seen[2 * i + 1]).count();
        let (nr, nt, nk) = tally(&m.margin, all().filter(|&i| !m.seen[2 * i] && !m.seen[2 * i + 1]), kf);
        say!(out, "\nread word for word (the whole reading): {gs} good and {bs} bad sentences ({} of {} sentences); good only {only_g} pairs, bad only {only_b}",
             gs + bs, th(2 * n as u64));
        say!(out, "  pairs with neither read: {nr}/{nk} = {:.2}% ({nt} ties)", pct(nr, nk));
        say!(out, "BLiMP word tokens she never read: {}/{} = {:.2}%; counted {} words / {} sentences, held out {} sentences ({} tokens), read {} sentences",
             th(m.oov), th(m.tok), pct(m.oov, m.tok), th(m.c_words), th(m.c_sents as u64), th(m.h_sents as u64), th(m.h_tokens as u64), th(m.x_sents as u64));
    }
    out
}
