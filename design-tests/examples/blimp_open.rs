// TESTING / DESIGN ONLY. (v30) Open expectations: a copy of blimp.rs (WP6, frozen there) with one new
// reading, used by `open-dev` only (dev pairs, pairID % 5 == 0; this copy is never run on test).
//
// OPENERS. Some words open an expectation that later words depend on (a "not" lets an "ever" in, a
// "who" leaves a gap to close). She finds them in her own reading, not in BLiMP: among her 150 most
// frequent words, an opener is one whose having been read earlier in the sentence (2 or more places
// back) makes the next word, given the word before it, more predictable on held-out text (fitted on
// her first 9M words, gain measured on the next 1M, Witten-Bell both ways). The 15 with the largest
// gain are her openers.
// HER READING WITH OPEN EXPECTATIONS. The state of a place in a sentence is the most recent opener 2
// or more places back (or none). Her next word, most specific first: the previous 3, 2, 1 words WITH
// the state, then the previous word alone, then any word: the same Witten-Bell backoff, so where a
// state was rarely read it is trusted little, and an expectation can lower a word as well as raise it.
//
// ZORRO (fresh confirmation C2, a loader added after the freeze; the scoring is the frozen open_test):
//   cargo run --release --example blimp_open -- zorro-load               the loader alone (no score)
//   cargo run --release --example blimp_open -- zorro-test <report.txt>  plain vs open on Zorro, ONCE
//
// (WP6 header follows.)
// (v29 WP6) Grammar from reading: BLiMP at 10M words.
//
//   cargo run --release --example blimp -- check               streaming counts == storing everything (no BLiMP scores)
//   cargo run --release --example blimp -- dev                 the two choices, made on dev (pairID % 5 == 0)
//   cargo run --release --example blimp -- test <report.txt>   the frozen build on test, ONCE (refuses to overwrite)
//   cargo run --release --example blimp -- dryrun <report.txt> the frozen test routine, run on dev first
//   cargo run --release --example blimp -- arc <report.txt>    report only: the ARC science corpus as a bigger reading
//
// Children reach most of English grammar from about 10M words heard. BLiMP (67 sets of 1,000
// sentence pairs that differ in one point of grammar) asks whether a reader finds the good sentence
// more expected than the bad one. She has no grammar rules: she reads, then scores each sentence by
// how expected each word is, given the words before it.
//
// Her next word, most specific first: the previous 4, 3, 2 and 1 words, then any word. Each level is
// trusted by its own experience, Witten-Bell, lambda = c / (c + d): c = how often she has read that
// context, d = how many different words she has seen follow it (morph_read `read`, the same backoff
// she uses for word forms). Read from the most general level up,
//     p = lambda * count(context, word) / c + (1 - lambda) * p_below,
// and a context she never read is passed over. Under the any-word level is an even share over every
// word she has read plus one more for a word she never met, so an unseen word is very unexpected but
// never impossible. A sentence opens with one start mark <s> (it can be a context word) and its end
// </s> is predicted like a word. Sentence score = the sum of ln p over its words and its end. The good
// sentence must score strictly higher; |good - bad| <= 1e-9 nats is a tie, and a tie is wrong.
//
// Words are her tokenizer's (seq::tokenize) tokens that hold a letter or digit, as meaning_build
// reads; punctuation is not a word. Lower-casing is one of the two dev choices: the cased reading
// makes exactly the same cuts, without lower-casing.
//
// What she reads: Tatoeba sentences, movie lines, and each distinct dialogue turn once, WITHOUT any
// sentence whose words equal a dialogue-exam question or reply (the first 1,200 lines of
// school/dialogue.tsv, withheld exactly as meaning_build withholds them). No Rust books. A line or a
// turn is one sentence; one with no word is dropped. Shuffled by sentence (Fisher-Yates from the
// end, j = splitmix64 % (i + 1), seed 81), then read in that order. The reading at W words is every
// sentence before the first one that would take her past W.
//
// Counting is streamed: only the contexts and (context, word) pairs the scored BLiMP sentences use
// are kept, plus how many different words follow each kept context. That gives the same numbers as
// storing every n-gram (`check` compares the two on a slice of the reading). Words outside BLiMP's
// vocabulary are told apart by a 64-bit hash.
//
// Pre-registered in "ava v29 - in depth" (WP6): the longest context (1-4) and lower-casing are
// chosen on dev; test is scored once, by the frozen build, at 10M words.
use design_tests::seq::tokenize;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

const DATA: &str = "../data";
const ARC_CORPUS: &str = "../data/ai2-arc/ARC-V1-Feb2018-2/ARC_Corpus.txt";
const SOURCE: &str = include_str!("blimp_open.rs");
const SEED: u64 = 81;
const MAIN: u64 = 10_000_000;
const MARKS: [u64; 4] = [1_000_000, 2_500_000, 5_000_000, MAIN];
const MAX_N: usize = 4;
const TIE: f64 = 1e-9;

// ---------------------------------------------------------------- frozen settings (test, arc)
// Chosen on dev before any test pair was scored (scratchpad v29/blimp_dev1.txt). Rule, fixed before
// the dev run: the best dev accuracy at 10M words; a tie goes to the shorter context, then to
// lower-casing. Dev at 10M words, contexts of 1/2/3/4 words: lower-cased 53.69 / 55.37 / 55.54 /
// 55.54%, cased 54.74 / 55.30 / 55.65 / 55.62% -> 3 words, cased.
const FROZEN_N: usize = 3;
const FROZEN_LOWER: bool = false;

const BOS: u32 = 0;
const EOS: u32 = 1;
const NONE: u32 = u32::MAX;
/// the end of a sentence, as a word key (no word's hash is this, short of a 1 in 2^64 accident)
const EOS_KEY: u64 = 0x0e05_0e05_0e05_0e05;

// ---------------------------------------------------------------- small tools
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    mix64(*x)
}
#[derive(Default)]
struct Mix(u64);
impl Hasher for Mix {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, b: &[u8]) { for &x in b { self.0 = mix64(self.0 ^ x as u64 ^ 0x9e37_79b9_7f4a_7c15); } }
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

// ---------------------------------------------------------------- Zorro (v30 fresh confirmation C2: a loader only)
/// Zorro (Huebner et al. 2021), its babyberta sentence files: one set per file, its lines in pairs,
/// the ungrammatical sentence first (seen on the first pair of every file, before any score). The
/// written form is restored to match her reading: the first letter capitalised and the lone word "i"
/// written "I". Every pair is a test pair: no id is a multiple of 5, so open_test keeps them all.
fn load_zorro() -> (Vec<Pair>, Vec<String>) {
    let dir = format!("{DATA}/zorro");
    let mut files: Vec<String> = std::fs::read_dir(&dir).unwrap_or_else(|e| die(&format!("cannot read {dir}: {e}")))
        .flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|f| f.ends_with(".txt")).collect();
    files.sort();
    let written = |s: &str| -> String {
        let s = s.split(' ').map(|w| if w == "i" { "I" } else { w }).collect::<Vec<_>>().join(" ");
        let mut c = s.chars();
        c.next().map_or(String::new(), |f| f.to_uppercase().collect::<String>() + c.as_str())
    };
    let (mut pairs, mut uids) = (Vec::new(), Vec::new());
    for (ui, f) in files.iter().enumerate() {
        let raw = std::fs::read_to_string(format!("{dir}/{f}")).unwrap_or_else(|e| die(&format!("{f}: {e}")));
        let lines: Vec<&str> = raw.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        if lines.len() % 2 != 0 { die(&format!("{f}: an odd number of lines")); }
        for (k, ch) in lines.chunks(2).enumerate() {
            pairs.push(Pair { uid: ui, term: 0, id: 5 * k as u32 + 1, good: written(ch[1]), bad: written(ch[0]) });
        }
        uids.push(f.trim_end_matches(".txt").to_string());
    }
    (pairs, uids)
}

// ---------------------------------------------------------------- what she reads
struct Reading { sents: Vec<Box<str>>, exam: HashSet<Vec<String>> }

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
    Reading { sents, exam }
}

// ---------------------------------------------------------------- her counts
/// the keys of the contexts before position i, shortest first (1..=maxn words; <s> before word 0).
/// A context is its words' ids + 1 in 16-bit slots, the nearest word lowest.
/// st: the state of each place (empty: the plain reading). With states, the levels are the previous
/// word alone, then the previous 1..=maxn words with the state (bit 63 marks them, the state in 48..56).
fn ctx_keys(ids: &[u32], st: &[u8], i: usize, maxn: usize, out: &mut Vec<u64>) {
    out.clear();
    let mut key = 0u64;
    for n in 1..=maxn.min(i + 1) {
        let h = if n <= i { ids[i - n] } else { BOS };
        if h == NONE { break; }
        key |= (h as u64 + 1) << (16 * (n - 1));
        if st.is_empty() { out.push(key); continue; }
        if n == 1 { out.push(key); }
        out.push(key | 1u64 << 63 | (st[i] as u64) << 48);
    }
}

/// which opener(s) a place remembers (dev choice, AVA_OPEN_MODE): "last" the most recent 2 or more
/// places back, "first" the earliest, "both" the pair (first, last). Index + 1 into trig; 0 none.
fn open_mode() -> String { std::env::var("AVA_OPEN_MODE").unwrap_or_else(|_| "last".into()) }

fn states(keys: &[u64], trig: &[u64]) -> Vec<u8> {
    if trig.is_empty() { return Vec::new(); }
    let mode = open_mode();
    let mut out = Vec::with_capacity(keys.len() + 1);
    let (mut first, mut last) = (0u8, 0u8);
    for i in 0..=keys.len() {
        if i >= 2 { if let Some(p) = trig.iter().position(|&t| t == keys[i - 2]) { last = p as u8 + 1; if first == 0 { first = last; } } }
        out.push(match mode.as_str() { "first" => first, "both" => first * 16 + last, _ => last });
    }
    out
}

struct Her {
    maxn: usize,
    trig: Vec<u64>,               // her openers (empty: the plain reading)
    st: Vec<Vec<u8>>,             // per scored sentence: the state of each place
    vocab: MixMap<u64, u32>,      // a BLiMP word's key -> its id (0 = <s>, 1 = </s>)
    sents: Vec<(Vec<u32>, Vec<u64>)>, // the scored sentences (2k good, 2k+1 bad): ids, keys
    whole: MixMap<u64, Vec<u32>>, // a scored sentence's word sequence -> which sentences
    ctx: MixMap<u64, u32>,        // a context the scored sentences use -> its slot
    c: Vec<u64>,                  // per slot: times she read the context
    d: Vec<u64>,                  // per slot: different words she saw follow it
    follow: MixSet<u64>,          // (slot, following word) met
    pair: MixMap<u64, u64>,       // (slot, BLiMP word id) -> times read
    uni: Vec<u64>,                // per BLiMP word id: times read
    n: u64,                       // words and sentence ends read
    types: MixSet<u64>,           // different words read (and the end)
    words: u64,
    nsent: u64,
    seen: Vec<bool>,              // scored sentence read word for word
}

fn seq_key(keys: &[u64]) -> u64 { keys.iter().fold(mix64(keys.len() as u64), |h, &k| mix64(h ^ k)) }

impl Her {
    fn new(maxn: usize, lower: bool, pairs: &[&Pair]) -> Her { Her::new_open(maxn, lower, pairs, Vec::new()) }

    fn new_open(maxn: usize, lower: bool, pairs: &[&Pair], trig: Vec<u64>) -> Her {
        let mut vocab: MixMap<u64, u32> = MixMap::default();
        let mut sents = Vec::with_capacity(pairs.len() * 2);
        for p in pairs {
            for s in [&p.good, &p.bad] {
                let w = words(s, lower);
                let keys: Vec<u64> = w.iter().map(|x| wkey(x)).collect();
                let ids: Vec<u32> = keys.iter().map(|k| { let next = vocab.len() as u32 + 2; *vocab.entry(*k).or_insert(next) }).collect();
                sents.push((ids, keys));
            }
        }
        if vocab.len() + 2 >= 65_535 { die("BLiMP vocabulary too big for 16-bit context slots"); }
        let (mut ctx, mut pair): (MixMap<u64, u32>, MixMap<u64, u64>) = (MixMap::default(), MixMap::default());
        let mut whole: MixMap<u64, Vec<u32>> = MixMap::default();
        let mut buf = Vec::new();
        let st: Vec<Vec<u8>> = sents.iter().map(|(_, keys)| states(keys, &trig)).collect();
        for (si, (ids, keys)) in sents.iter().enumerate() {
            whole.entry(seq_key(keys)).or_default().push(si as u32);
            for i in 0..=ids.len() {
                let t = if i < ids.len() { ids[i] } else { EOS };
                ctx_keys(ids, &st[si], i, maxn, &mut buf);
                for &k in &buf {
                    let next = ctx.len() as u32;
                    let s = *ctx.entry(k).or_insert(next);
                    pair.entry((s as u64) << 32 | t as u64).or_insert(0);
                }
            }
        }
        let (nctx, nv, ns) = (ctx.len(), vocab.len() + 2, sents.len());
        Her { maxn, trig, st, vocab, sents, whole, ctx, c: vec![0; nctx], d: vec![0; nctx], follow: MixSet::default(), pair,
              uni: vec![0; nv], n: 0, types: MixSet::default(), words: 0, nsent: 0, seen: vec![false; ns] }
    }

    /// read one sentence: every word and the end, after each of its contexts the BLiMP sentences use
    fn read(&mut self, w: &[String], ids: &mut Vec<u32>, keys: &mut Vec<u64>, buf: &mut Vec<u64>) {
        keys.clear();
        ids.clear();
        for x in w { let k = wkey(x); keys.push(k); ids.push(self.vocab.get(&k).copied().unwrap_or(NONE)); }
        self.nsent += 1;
        self.words += w.len() as u64;
        let stv = states(keys, &self.trig);
        for i in 0..=keys.len() {
            let (tk, tid) = if i < keys.len() { (keys[i], ids[i]) } else { (EOS_KEY, EOS) };
            self.n += 1;
            self.types.insert(tk);
            if tid != NONE { self.uni[tid as usize] += 1; }
            ctx_keys(ids, &stv, i, self.maxn, buf);
            for &k in buf.iter() {
                // a longer context holds this one, so if this one is not used, neither is it
                let Some(&s) = self.ctx.get(&k) else { break };
                self.c[s as usize] += 1;
                if self.follow.insert(mix64(tk ^ (s as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))) { self.d[s as usize] += 1; }
                if tid != NONE { if let Some(x) = self.pair.get_mut(&((s as u64) << 32 | tid as u64)) { *x += 1; } }
            }
        }
        if let Some(list) = self.whole.get(&seq_key(keys)) {
            for &si in list { if self.sents[si as usize].1 == *keys { self.seen[si as usize] = true; } }
        }
    }

    /// ln P of a scored sentence's words and end, summed, for the longest context 1..=maxn: out[n - 1]
    fn score(&self, si: usize, buf: &mut Vec<u64>) -> [f64; MAX_N] {
        let ids = &self.sents[si].0;
        let nlev = self.maxn + !self.trig.is_empty() as usize;
        let (nn, v) = (self.n as f64, self.types.len() as f64);
        let lam0 = if nn > 0.0 { nn / (nn + v) } else { 0.0 };
        let mut out = [0.0; MAX_N];
        for i in 0..=ids.len() {
            let t = if i < ids.len() { ids[i] } else { EOS };
            // any word, over an even share for every word read plus one never met
            let mut p = (1.0 - lam0) / (v + 1.0);
            if nn > 0.0 { p += lam0 * self.uni[t as usize] as f64 / nn; }
            ctx_keys(ids, &self.st[si], i, self.maxn, buf);
            for n in 1..=nlev {
                if let Some(&s) = buf.get(n - 1).and_then(|k| self.ctx.get(k)) {
                    let (c, d) = (self.c[s as usize] as f64, self.d[s as usize] as f64);
                    if c > 0.0 {
                        let lam = c / (c + d);
                        let ct = self.pair.get(&((s as u64) << 32 | t as u64)).copied().unwrap_or(0) as f64;
                        p = lam * ct / c + (1.0 - lam) * p;
                    }
                }
                out[n - 1] += p.ln();
            }
        }
        out
    }

    fn snap(&self, label: String) -> Snap {
        let mut buf = Vec::new();
        let margin = (0..self.sents.len() / 2).map(|k| {
            let (g, b) = (self.score(2 * k, &mut buf), self.score(2 * k + 1, &mut buf));
            let mut m = [0.0; MAX_N];
            for n in 0..MAX_N { m[n] = g[n] - b[n]; }
            m
        }).collect();
        let (mut oov, mut tok) = (0u64, 0u64);
        for (ids, _) in &self.sents { for &i in ids { tok += 1; if self.uni[i as usize] == 0 { oov += 1; } } }
        Snap { label, words: self.words, sents: self.nsent, margin, seen: self.seen.clone(), oov, tok,
               types: self.types.len() as u64, follow: self.follow.len() as u64 }
    }
}

struct Snap { label: String, words: u64, sents: u64, margin: Vec<[f64; MAX_N]>, seen: Vec<bool>, oov: u64, tok: u64, types: u64, follow: u64 }

impl Snap {
    /// (right, ties) over the given pairs (indices into the scored pairs), longest context n
    fn tally(&self, which: impl Iterator<Item = usize>, n: usize) -> (u64, u64, u64) {
        let (mut r, mut t, mut k) = (0, 0, 0);
        for i in which {
            let m = self.margin[i][n - 1];
            k += 1;
            if m.abs() <= TIE { t += 1; } else if m > 0.0 { r += 1; }
        }
        (r, t, k)
    }
}

/// her reading, streamed; ARC first when asked. Snapshots at each mark she reaches, and at the end.
fn stream(r: &Reading, pairs: &[&Pair], lower: bool, maxn: usize, arc: bool, out: &mut String) -> Vec<Snap> { stream_open(r, pairs, lower, maxn, arc, out, &[]) }

fn stream_open(r: &Reading, pairs: &[&Pair], lower: bool, maxn: usize, arc: bool, out: &mut String, trig: &[u64]) -> Vec<Snap> {
    let t0 = Instant::now();
    let mut her = Her::new_open(maxn, lower, pairs, trig.to_vec());
    say!(out, "  [{}] {} scored sentences, {} BLiMP word types, {} contexts and {} (context, word) pairs to count",
         if lower { "lower-cased" } else { "cased" }, th(her.sents.len() as u64), th(her.vocab.len() as u64), th(her.ctx.len() as u64), th(her.pair.len() as u64));
    let (mut ids, mut keys, mut buf) = (Vec::new(), Vec::new(), Vec::new());
    let mut snaps = Vec::new();
    if arc {
        let f = std::fs::File::open(ARC_CORPUS).unwrap_or_else(|e| die(&format!("{ARC_CORPUS}: {e}")));
        let mut rd = BufReader::with_capacity(1 << 20, f);
        let (mut line, mut lines, mut skipped) = (Vec::new(), 0u64, [0u64; 2]);
        loop {
            line.clear();
            if rd.read_until(b'\n', &mut line).unwrap_or_else(|e| die(&format!("read: {e}"))) == 0 { break; }
            lines += 1;
            let s = String::from_utf8_lossy(&line);
            let w = words(&s, lower);
            if w.is_empty() { skipped[0] += 1; continue; }
            if if lower { r.exam.contains(&w) } else { r.exam.contains(&words(&s, true)) } { skipped[1] += 1; continue; }
            her.read(&w, &mut ids, &mut keys, &mut buf);
            if lines % 1_000_000 == 0 {
                ram_guard("ARC corpus");
                println!("    ARC {} lines, {} words, {} follower pairs, {:.0}s, peak {:.2} GB, free {:.1} GB",
                         th(lines), th(her.words), th(her.follow.len() as u64), t0.elapsed().as_secs_f64(), peak_gb(), free_gb());
            }
        }
        say!(out, "  ARC corpus: {} lines, {} with no word, {} equal to an exam turn; {} sentences, {} words read in {:.0}s",
             th(lines), th(skipped[0]), th(skipped[1]), th(her.nsent), th(her.words), t0.elapsed().as_secs_f64());
        snaps.push(her.snap("ARC corpus alone".into()));
    }
    let base = her.words;
    let mut next = if arc { MARKS.len() } else { 0 };
    for (k, s) in r.sents.iter().enumerate() {
        let w = words(s, lower);
        while next < MARKS.len() && her.words - base + w.len() as u64 > MARKS[next] {
            snaps.push(her.snap(format!("{} words", th(MARKS[next]))));
            next += 1;
        }
        her.read(&w, &mut ids, &mut keys, &mut buf);
        if (k + 1) % 500_000 == 0 { ram_guard("reading"); }
    }
    snaps.push(her.snap(if arc { "ARC corpus + all her reading".into() } else { format!("all ({} words)", th(her.words)) }));
    say!(out, "  read {} sentences, {} words in {:.1}s; {} contexts read at least once; peak RAM {:.2} GB",
         th(her.nsent), th(her.words), t0.elapsed().as_secs_f64(), th(her.c.iter().filter(|&&c| c > 0).count() as u64), peak_gb());
    snaps
}

/// paired bootstrap of (right - chance) over pairs, 20,000 resamples, splitmix64 seed 1: 95% interval, in pairs
fn boot(right: &[bool]) -> (f64, f64) {
    let n = right.len();
    let diff: Vec<i64> = right.iter().map(|&r| if r { 1 } else { -1 }).collect(); // 2 * (right - 1/2)
    let mut x: u64 = 1;
    let mut b: Vec<i64> = (0..20_000).map(|_| (0..n).map(|_| diff[(splitmix(&mut x) % n as u64) as usize]).sum()).collect();
    b.sort_unstable();
    (b[500] as f64 / 2.0, b[19_499] as f64 / 2.0)
}

// ---------------------------------------------------------------- check: streaming == storing everything
/// every n-gram stored, the obvious way, for `check`
struct Naive { uni: HashMap<String, u64>, tab: HashMap<String, HashMap<String, u64>>, n: u64 }

impl Naive {
    fn new(sents: &[Vec<String>]) -> Naive {
        let mut nv = Naive { uni: HashMap::new(), tab: HashMap::new(), n: 0 };
        for s in sents {
            let hist: Vec<&str> = std::iter::once("<s>").chain(s.iter().map(|x| x.as_str())).collect();
            for i in 0..=s.len() {
                let t = if i < s.len() { s[i].as_str() } else { "</s>" };
                nv.n += 1;
                *nv.uni.entry(t.to_string()).or_insert(0) += 1;
                for n in 1..=MAX_N.min(i + 1) {
                    *nv.tab.entry(hist[i + 1 - n..=i].join(" ")).or_default().entry(t.to_string()).or_insert(0) += 1;
                }
            }
        }
        nv
    }
    fn p(&self, hist: &[&str], t: &str) -> [f64; MAX_N] {
        let (nn, v) = (self.n as f64, self.uni.len() as f64);
        let l0 = nn / (nn + v);
        let mut p = l0 * *self.uni.get(t).unwrap_or(&0) as f64 / nn + (1.0 - l0) / (v + 1.0);
        let mut out = [0.0; MAX_N];
        for n in 1..=MAX_N {
            if n <= hist.len() {
                if let Some(m) = self.tab.get(&hist[hist.len() - n..].join(" ")) {
                    let c: u64 = m.values().sum();
                    let lam = c as f64 / (c as f64 + m.len() as f64);
                    p = lam * *m.get(t).unwrap_or(&0) as f64 / c as f64 + (1.0 - lam) * p;
                }
            }
            out[n - 1] = p;
        }
        out
    }
    fn score(&self, s: &[String]) -> [f64; MAX_N] {
        let hist: Vec<&str> = std::iter::once("<s>").chain(s.iter().map(|x| x.as_str())).collect();
        let mut out = [0.0; MAX_N];
        for i in 0..=s.len() {
            let p = self.p(&hist[..=i], if i < s.len() { &s[i] } else { "</s>" });
            for n in 0..MAX_N { out[n] += p[n].ln(); }
        }
        out
    }
}

fn check(pairs: &[Pair]) {
    let mut out = String::new();
    say!(out, "source sha256 (compiled in): {}", sha256(SOURCE.as_bytes()));
    let r = reading(&mut out);
    let dev: Vec<&Pair> = pairs.iter().filter(|p| p.id % 5 == 0).take(1000).collect();
    for lower in [true, false] {
        let slice: Vec<Vec<String>> = r.sents.iter().take(30_000).map(|s| words(s, lower)).collect();
        let mut her = Her::new(MAX_N, lower, &dev);
        let (mut ids, mut keys, mut buf) = (Vec::new(), Vec::new(), Vec::new());
        for w in &slice { her.read(w, &mut ids, &mut keys, &mut buf); }
        let nv = Naive::new(&slice);
        let mut worst = 0f64;
        for (k, p) in dev.iter().enumerate() {
            for (j, s) in [&p.good, &p.bad].iter().enumerate() {
                let (a, b) = (her.score(2 * k + j, &mut buf), nv.score(&words(s, lower)));
                for n in 0..MAX_N { worst = worst.max((a[n] - b[n]).abs()); }
            }
        }
        // the readout is a distribution: over every word read, plus one never met, it sums to 1
        let mut sums = Vec::new();
        for p in dev.iter().take(5) {
            let w = words(&p.good, lower);
            let hist: Vec<&str> = std::iter::once("<s>").chain(w.iter().map(|x| x.as_str())).take(4).collect();
            let unseen = nv.p(&hist, "\u{1}never met");
            let mut s = unseen;
            for t in nv.uni.keys() { let q = nv.p(&hist, t); for n in 0..MAX_N { s[n] += q[n]; } }
            sums.push(format!("[{}]", s.iter().map(|x| format!("{x:.12}")).collect::<Vec<_>>().join(" ")));
        }
        say!(out, "{}: 30,000 sentences, {} words; {} sentences ({} types) N={} V={} vs N={} V={}",
             if lower { "lower-cased" } else { "cased" }, th(her.words), th(her.sents.len() as u64), th(her.vocab.len() as u64), her.n, her.types.len(), nv.n, nv.uni.len());
        say!(out, "  streamed vs stored, largest |difference| in a sentence's ln P (n = 1..4): {worst:.3e}");
        say!(out, "  sum of P over every word read + one never met, 5 contexts (n = 1..4): {}", sums.join(" "));
    }
}

// ---------------------------------------------------------------- main
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(|s| s.as_str()).unwrap_or("");
    let t0 = Instant::now();
    if mode == "zorro-load" || mode == "zorro-test" {
        let (pairs, uids) = load_zorro();
        if pairs.len() != 46_000 || uids.len() != 23 { die(&format!("expected 23 Zorro sets x 2,000 pairs, found {} sets, {} pairs", uids.len(), pairs.len())); }
        if mode == "zorro-load" {
            // the loader alone: counts and each set's first pair as she will see them (no score)
            println!("Zorro: {} sets, {} pairs", uids.len(), th(pairs.len() as u64));
            for (u, name) in uids.iter().enumerate() {
                let p = pairs.iter().find(|p| p.uid == u).unwrap();
                println!("  {name:<52} good: {}  |  bad: {}", p.good, p.bad);
            }
        } else {
            let path = a.get(2).unwrap_or_else(|| die("usage: blimp_open zorro-test <report.txt>"));
            // the confirmation is scored once: this build never writes over a report
            let mut fh = std::fs::OpenOptions::new().write(true).create_new(true).open(path)
                .unwrap_or_else(|e| die(&format!("refusing: {path} exists or cannot be created ({e}); the test is scored once")));
            let rep = open_test(&pairs, &uids, false);
            fh.write_all(format!("ZORRO, v30 fresh confirmation C2: 23 sets x 2,000 pairs, every pair scored\n{rep}").as_bytes())
                .unwrap_or_else(|e| die(&format!("write {path}: {e}")));
        }
        println!("total {:.1}s, peak RAM {:.2} GB", t0.elapsed().as_secs_f64(), peak_gb());
        return;
    }
    let (pairs, uids, terms) = load_blimp();
    if pairs.len() != 67_000 || uids.len() != 67 { die(&format!("expected 67 sets x 1,000 pairs, found {} sets, {} pairs", uids.len(), pairs.len())); }
    match mode {
        "check" => check(&pairs),
        "dev" => dev(&pairs),
        "open-dev" => open_dev(&pairs, &uids),
        "open-test" | "open-dryrun" => {
            let path = a.get(2).unwrap_or_else(|| die("usage: blimp_open open-test|open-dryrun <report.txt>"));
            // the test is scored once: this build never writes over a report
            let mut fh = std::fs::OpenOptions::new().write(true).create_new(true).open(path)
                .unwrap_or_else(|e| die(&format!("refusing: {path} exists or cannot be created ({e}); the test is scored once")));
            let rep = open_test(&pairs, &uids, mode == "open-dryrun");
            fh.write_all(rep.as_bytes()).unwrap_or_else(|e| die(&format!("write {path}: {e}")));
        }
        "test" | "arc" | "dryrun" => {
            let path = a.get(2).unwrap_or_else(|| die("usage: blimp test|arc|dryrun <report.txt>"));
            // the test is scored once: this build never writes over a report
            let mut fh = std::fs::OpenOptions::new().write(true).create_new(true).open(path)
                .unwrap_or_else(|e| die(&format!("refusing: {path} exists or cannot be created ({e}); the test is scored once")));
            let rep = test(&pairs, &uids, &terms, mode == "arc", mode == "dryrun");
            fh.write_all(rep.as_bytes()).unwrap_or_else(|e| die(&format!("write {path}: {e}")));
        }
        _ => die("usage: blimp check | dev | dryrun <report.txt> | test <report.txt> | arc <report.txt>"),
    }
    println!("total {:.1}s, peak RAM {:.2} GB", t0.elapsed().as_secs_f64(), peak_gb());
}

fn dev(pairs: &[Pair]) {
    let mut out = String::new();
    say!(out, "source sha256 (compiled in): {}", sha256(SOURCE.as_bytes()));
    let dev: Vec<&Pair> = pairs.iter().filter(|p| p.id % 5 == 0).collect();
    say!(out, "DEV: {} pairs (pairID % 5 == 0); test pairs are not read", th(dev.len() as u64));
    let r = reading(&mut out);
    let mut best: Option<(u64, usize, bool)> = None;
    for lower in [true, false] {
        let snaps = stream(&r, &dev, lower, MAX_N, false, &mut out);
        say!(out, "  {:<22} {:>16} {:>16} {:>16} {:>16}   unseen BLiMP words, read verbatim (good/bad)", "words read",
             "1 word", "2 words", "3 words", "4 words");
        for sn in &snaps {
            let mut row = format!("  {:<22}", sn.label);
            for n in 1..=MAX_N {
                let (rt, ti, k) = sn.tally(0..dev.len(), n);
                row.push_str(&format!(" {:>7.2}% ({:>3} t)", pct(rt, k), ti));
                if sn.label.starts_with("10,000,000") {
                    // the rule, fixed before this run: best at 10M; a tie goes to the shorter context, then lower-casing
                    let better = match best {
                        None => true,
                        Some((b, bn, bl)) => rt > b || (rt == b && (n < bn || (n == bn && lower && !bl))),
                    };
                    if better { best = Some((rt, n, lower)); }
                }
            }
            let gs = (0..dev.len()).filter(|&k| sn.seen[2 * k]).count();
            let bs = (0..dev.len()).filter(|&k| sn.seen[2 * k + 1]).count();
            row.push_str(&format!("   {:.2}%, {gs}/{bs}", pct(sn.oov, sn.tok)));
            say!(out, "{row}");
        }
    }
    if let Some((rt, n, lower)) = best {
        say!(out, "\nDEV CHOICE (best at 10M words): longest context {n} word(s), {}; {}/{} = {:.2}%",
             if lower { "lower-cased" } else { "cased" }, rt, dev.len(), pct(rt, dev.len() as u64));
    }
}

/// the frozen routine: on test (once), or on dev (`dry`: the same code, run first so the one test
/// run cannot stop half-way; nothing is chosen from it)
fn test(pairs: &[Pair], uids: &[String], terms: &[String], arc: bool, dry: bool) -> String {
    let mut out = String::new();
    if !(1..=MAX_N).contains(&FROZEN_N) { die("FROZEN_N is not set: make the dev choice first"); }
    let what = if dry { "DRY RUN of the test routine, on DEV" } else if arc { "REPORT ONLY: the ARC science corpus as a bigger reading (test pairs)" } else { "TEST (scored once)" };
    say!(out, "BLiMP {what} -- frozen build, source sha256 (compiled in): {}", sha256(SOURCE.as_bytes()));
    say!(out, "frozen: longest context {FROZEN_N} word(s), {}; ties: |good - bad| <= {TIE:e} nats count wrong", if FROZEN_LOWER { "lower-cased" } else { "cased" });
    let test: Vec<&Pair> = pairs.iter().filter(|p| (p.id % 5 == 0) == dry).collect();
    say!(out, "{}: {} pairs (pairID % 5 {} 0)", if dry { "dev" } else { "test" }, th(test.len() as u64), if dry { "==" } else { "!=" });
    let r = reading(&mut out);
    let snaps = stream(&r, &test, FROZEN_LOWER, FROZEN_N, arc, &mut out);
    let n = test.len();
    let all = || 0..n;

    say!(out, "\n{}", if arc { "READINGS (report only, no bar)" } else { "LEARNING CURVE (test)" });
    say!(out, "  {:<30} {:>12} {:>11} {:>9} {:>6} {:>14} {:>10}", "reading", "words", "sentences", "right", "ties", "unseen words", "types");
    for sn in &snaps {
        let (rt, ti, k) = sn.tally(all(), FROZEN_N);
        say!(out, "  {:<30} {:>12} {:>11} {:>8.2}% {:>6} {:>13.2}% {:>10}", sn.label, th(sn.words), th(sn.sents), pct(rt, k), ti, pct(sn.oov, sn.tok), th(sn.types));
    }
    // in full: the main reading, 10M words (test); both readings (arc report)
    let mains: Vec<&Snap> = snaps.iter().filter(|s| arc || s.label.starts_with("10,000,000")).collect();
    for sn in mains {
        let (rt, ti, k) = sn.tally(all(), FROZEN_N);
        let right: Vec<bool> = (0..n).map(|i| sn.margin[i][FROZEN_N.saturating_sub(1)] > TIE).collect();
        let (lo, hi) = boot(&right);
        let acc = pct(rt, k);
        say!(out, "\n== {} ==", sn.label);
        say!(out, "overall: {rt}/{k} right = {acc:.2}%, {ti} ties (counted wrong)");
        say!(out, "right - chance: {:+.1} pairs ({:+.2} points), 95% CI {:+.1} .. {:+.1} pairs ({:+.2} .. {:+.2} points) (paired bootstrap, 20,000, seed 1)",
             rt as f64 - k as f64 / 2.0, acc - 50.0, lo, hi, lo / k as f64 * 100.0, hi / k as f64 * 100.0);
        if !arc {
            say!(out, "RULE 1  overall >= 60.0%:            {:.2}%  -> {}", acc, if acc >= 60.0 { "PASS" } else { "FAIL" });
            say!(out, "RULE 2  95% CI vs chance above 0:    lower bound {:+.1} pairs -> {}", lo, if lo > 0.0 { "PASS" } else { "FAIL" });
            say!(out, "VERDICT: {}", if acc >= 60.0 && lo > 0.0 { "PASS" } else { "FAIL" });
        }
        // phenomena (BLiMP's linguistics_term; the paper's 12 put s-selection inside argument structure)
        say!(out, "\nphenomena (linguistics_term)          pairs    right   ties");
        let mut rows: Vec<(String, u64, u64, u64)> = terms.iter().enumerate().map(|(t, name)| {
            let (rt, ti, k) = sn.tally(all().filter(|&i| test[i].term == t), FROZEN_N);
            (name.clone(), k, rt, ti)
        }).collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, k, rt, ti) in &rows { say!(out, "  {name:<34} {k:>6} {:>7.2}% {ti:>6}", pct(*rt, *k)); }
        if let (Some(a), Some(s)) = (terms.iter().position(|t| t == "argument_structure"), terms.iter().position(|t| t == "s-selection")) {
            let (rt, ti, k) = sn.tally(all().filter(|&i| test[i].term == a || test[i].term == s), FROZEN_N);
            say!(out, "  {:<34} {k:>6} {:>7.2}% {ti:>6}", "(paper's 12) argument structure + s-sel.", pct(rt, k));
        }
        // all 67 sets
        let mut sets: Vec<(f64, String, u64, u64, u64)> = uids.iter().enumerate().map(|(u, name)| {
            let (rt, ti, k) = sn.tally(all().filter(|&i| test[i].uid == u), FROZEN_N);
            (pct(rt, k), name.clone(), k, rt, ti)
        }).collect();
        sets.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        say!(out, "\nall 67 sets, best first                                       pairs    right   ties  phenomenon");
        for (acc, name, k, _, ti) in &sets {
            let u = uids.iter().position(|x| x == name).unwrap();
            let t = test.iter().find(|p| p.uid == u).map(|p| terms[p.term].as_str()).unwrap_or("");
            say!(out, "  {name:<58} {k:>6} {acc:>7.2}% {ti:>6}  {t}");
        }
        // what she read word for word
        let gs = (0..n).filter(|&i| sn.seen[2 * i]).count() as u64;
        let bs = (0..n).filter(|&i| sn.seen[2 * i + 1]).count() as u64;
        let only_g = (0..n).filter(|&i| sn.seen[2 * i] && !sn.seen[2 * i + 1]).count();
        let only_b = (0..n).filter(|&i| !sn.seen[2 * i] && sn.seen[2 * i + 1]).count();
        let (nr, nt, nk) = sn.tally(all().filter(|&i| !sn.seen[2 * i] && !sn.seen[2 * i + 1]), FROZEN_N);
        say!(out, "\nread word for word: {gs} good and {bs} bad test sentences ({} of {} sentences); good only {only_g} pairs, bad only {only_b}",
             gs + bs, th(2 * n as u64));
        say!(out, "  pairs with neither read: {nr}/{nk} = {:.2}% ({nt} ties)", pct(nr, nk));
        say!(out, "BLiMP word tokens she never read: {}/{} = {:.2}%; follower pairs kept {}", th(sn.oov), th(sn.tok), pct(sn.oov, sn.tok), th(sn.follow));
    }
    out
}

// ---------------------------------------------------------------- v30: openers
/// her openers, learned from her reading (the sentences inside her first 10M words): fitted on the
/// first 9M words, gain measured on the next 1M (see the header). Returns the 15 best, best first.
fn learn_openers(r: &Reading, out: &mut String) -> Vec<u64> {
    const NCAND: usize = 150;
    const NOPEN: usize = 15;
    let t0 = Instant::now();
    let (mut sents, mut n) = (Vec::new(), 0u64);
    for s in &r.sents {
        let w = words(s, false);
        if n + w.len() as u64 > MAIN { break; }
        n += w.len() as u64;
        sents.push(w.iter().map(|x| wkey(x)).collect::<Vec<u64>>());
    }
    let mut split = sents.len();
    let mut acc = 0u64;
    for (k, s) in sents.iter().enumerate() { acc += s.len() as u64; if acc > MAIN * 9 / 10 { split = k; break; } }
    let (fit, held) = sents.split_at(split);
    // her most frequent words in the fitted part are the candidates
    let mut freq: MixMap<u64, u64> = MixMap::default();
    for s in fit { for &k in s { *freq.entry(k).or_insert(0) += 1; } }
    let total: u64 = freq.values().sum();
    let mut fv: Vec<(u64, u64)> = freq.iter().map(|(&k, &c)| (c, k)).collect();
    fv.sort_unstable_by(|a, b| b.cmp(a));
    let cand: Vec<u64> = fv.iter().take(NCAND).map(|x| x.1).collect();
    let cidx: MixMap<u64, usize> = cand.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let vtypes = freq.len() as f64 + 1.0;
    // counts: (prev, next) and, per candidate open 2+ places back, (cand, prev, next); prev of word 0 is <s>
    const BOS_KEY: u64 = 0x0b05_0b05_0b05_0b05;
    let pk = |a: u64, b: u64| mix64(a ^ b.rotate_left(29) ^ 0x51ed_270b_27a8_6d3f);
    let tk = |t: usize, a: u64| mix64(a ^ (t as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let (mut c2, mut c1, mut d1): (MixMap<u64, u32>, MixMap<u64, u32>, MixMap<u64, u32>) = Default::default();
    let (mut ct2, mut ct1, mut dt1): (MixMap<u64, u32>, MixMap<u64, u32>, MixMap<u64, u32>) = Default::default();
    let open_at = |s: &[u64], i: usize, seen: &mut Vec<usize>| { if i >= 2 { if let Some(&c) = cidx.get(&s[i - 2]) { if !seen.contains(&c) { seen.push(c); } } } };
    let mut seen: Vec<usize> = Vec::new();
    for s in fit {
        seen.clear();
        for i in 0..=s.len() {
            open_at(s, i, &mut seen);
            let prev = if i == 0 { BOS_KEY } else { s[i - 1] };
            let next = if i < s.len() { s[i] } else { EOS_KEY };
            let e = c2.entry(pk(prev, next)).or_insert(0);
            if *e == 0 { *d1.entry(prev).or_insert(0) += 1; }
            *e += 1;
            *c1.entry(prev).or_insert(0) += 1;
            for &t in &seen {
                let e = ct2.entry(pk(tk(t, prev), next)).or_insert(0);
                if *e == 0 { *dt1.entry(tk(t, prev)).or_insert(0) += 1; }
                *e += 1;
                *ct1.entry(tk(t, prev)).or_insert(0) += 1;
            }
        }
    }
    ram_guard("openers");
    let uni = |x: u64| (freq.get(&x).copied().unwrap_or(0) as f64 + 1.0) / (total as f64 + vtypes);
    let p_prev = |prev: u64, x: u64| {
        let (c, d) = (c1.get(&prev).copied().unwrap_or(0) as f64, d1.get(&prev).copied().unwrap_or(0) as f64);
        if c == 0.0 { return uni(x); }
        let lam = c / (c + d);
        lam * c2.get(&pk(prev, x)).copied().unwrap_or(0) as f64 / c + (1.0 - lam) * uni(x)
    };
    let mut gain = vec![0f64; NCAND];
    let mut uses = vec![0u64; NCAND];
    for s in held {
        seen.clear();
        for i in 0..=s.len() {
            open_at(s, i, &mut seen);
            if seen.is_empty() { continue; }
            let prev = if i == 0 { BOS_KEY } else { s[i - 1] };
            let next = if i < s.len() { s[i] } else { EOS_KEY };
            let base = p_prev(prev, next);
            for &t in &seen {
                let (c, d) = (ct1.get(&tk(t, prev)).copied().unwrap_or(0) as f64, dt1.get(&tk(t, prev)).copied().unwrap_or(0) as f64);
                let p = if c == 0.0 { base } else { let lam = c / (c + d); lam * ct2.get(&pk(tk(t, prev), next)).copied().unwrap_or(0) as f64 / c + (1.0 - lam) * base };
                gain[t] += (p / base).ln();
                uses[t] += 1;
            }
        }
    }
    // dev choice AVA_OPEN_PICK: "gain" (total held-out gain) or "rate" (gain per place, among openers
    // met at 10,000 or more held-out places)
    let rate = std::env::var("AVA_OPEN_PICK").map_or(false, |v| v == "rate");
    let val = |t: usize| if rate { gain[t] / uses[t].max(1) as f64 } else { gain[t] };
    let mut order: Vec<usize> = (0..NCAND).filter(|&t| gain[t] > 0.0 && (!rate || uses[t] >= 10_000)).collect();
    order.sort_by(|&a, &b| val(b).partial_cmp(&val(a)).unwrap());
    let names: MixMap<u64, String> = r.sents.iter().take(300_000).flat_map(|s| words(s, false)).map(|w| (wkey(&w), w)).collect();
    let name = |k: u64| names.get(&k).cloned().unwrap_or_else(|| "?".into());
    say!(out, "openers learned from her reading ({} candidates; fitted on {} sentences, gain on {} held-out sentences; {:.1}s):",
         NCAND, th(fit.len() as u64), th(held.len() as u64), t0.elapsed().as_secs_f64());
    for &t in order.iter().take(NOPEN) {
        say!(out, "  {:<10} held-out gain {:>9.1} nats over {:>9} places ({:+.4} per place)", name(cand[t]), gain[t], th(uses[t]), gain[t] / uses[t].max(1) as f64);
    }
    say!(out, "  (next 5, not used: {})", order.iter().skip(NOPEN).take(5).map(|&t| name(cand[t])).collect::<Vec<_>>().join(", "));
    order.iter().take(NOPEN).map(|&t| cand[t]).collect()
}

/// dev only: her plain reading (3 previous words, cased: the WP6 frozen reading) against the same with
/// open expectations, at 10M words, overall and set by set; paired bootstrap on the dev pairs
fn open_dev(pairs: &[Pair], uids: &[String]) {
    let mut out = String::new();
    say!(out, "source sha256 (compiled in): {}", sha256(SOURCE.as_bytes()));
    let dev: Vec<&Pair> = pairs.iter().filter(|p| p.id % 5 == 0).collect();
    say!(out, "open mode {}, openers picked by {}", open_mode(), std::env::var("AVA_OPEN_PICK").unwrap_or_else(|_| "gain".into()));
    say!(out, "DEV: {} pairs (pairID % 5 == 0); test pairs are not read", th(dev.len() as u64));
    let r = reading(&mut out);
    let trig = learn_openers(&r, &mut out);
    let plain = stream_open(&r, &dev, false, 3, false, &mut out, &[]);
    let open = stream_open(&r, &dev, false, 3, false, &mut out, &trig);
    let at = |s: &[Snap]| s.iter().position(|x| x.label.starts_with("10,000,000")).unwrap_or_else(|| die("no 10M snapshot"));
    let (p, o) = (&plain[at(&plain)], &open[at(&open)]);
    let right = |s: &Snap, n: usize| -> Vec<bool> { (0..dev.len()).map(|k| s.margin[k][n - 1] > TIE).collect() };
    let (rp, ro) = (right(p, 3), right(o, 4));
    let np = rp.iter().filter(|&&x| x).count() as u64;
    let no = ro.iter().filter(|&&x| x).count() as u64;
    say!(out, "\nat 10M words: plain {:.2}%  open expectations {:.2}%  ({} vs {} of {})", pct(np, dev.len() as u64), pct(no, dev.len() as u64), np, no, dev.len());
    // paired bootstrap of the difference (splitmix64 seed 1, 20,000 resamples)
    let d: Vec<f64> = (0..dev.len()).map(|k| ro[k] as u8 as f64 - rp[k] as u8 as f64).collect();
    let mut x = 1u64;
    let mut ms: Vec<f64> = (0..20_000).map(|_| (0..d.len()).map(|_| d[(splitmix(&mut x) % d.len() as u64) as usize]).sum::<f64>() / d.len() as f64).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    say!(out, "difference {:+.2} points, 95% CI {:+.2} .. {:+.2}", 100.0 * d.iter().sum::<f64>() / d.len() as f64, 100.0 * ms[499], 100.0 * ms[19_499]);
    let mut rows: Vec<(f64, String)> = (0..uids.len()).map(|u| {
        let ks: Vec<usize> = (0..dev.len()).filter(|&k| dev[k].uid == u).collect();
        let (a, b) = (ks.iter().filter(|&&k| rp[k]).count() as u64, ks.iter().filter(|&&k| ro[k]).count() as u64);
        (pct(b, ks.len() as u64) - pct(a, ks.len() as u64), format!("  {:<52} {:>6.1}% -> {:>6.1}%", uids[u], pct(a, ks.len() as u64), pct(b, ks.len() as u64)))
    }).collect();
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    say!(out, "set by set (plain -> open), most helped first:");
    for (_, l) in &rows { say!(out, "{l}"); }
}

/// v30 frozen dev choice (open_dev on dev pairs, 2026-10-15): the most recent opener, openers picked
/// by held-out gain per place among those met at 10,000+ places (56.97% dev vs 55.65% plain)
const FROZEN_OPEN_MODE: &str = "last";
const FROZEN_OPEN_PICK: &str = "rate";

/// the one test run (pre-registered in "ava v30 - in depth"): plain (WP6 frozen reading, 3 words,
/// cased) against open expectations (frozen choice above), at 10M words, on the TEST pairs
/// (pairID % 5 != 0). PASS if open beats plain with the paired bootstrap 95% CI above 0.
fn open_test(pairs: &[Pair], uids: &[String], dry: bool) -> String {
    let mut out = String::new();
    if std::env::var("AVA_OPEN_MODE").map_or(false, |v| v != FROZEN_OPEN_MODE) || std::env::var("AVA_OPEN_PICK").map_or(false, |v| v != FROZEN_OPEN_PICK) {
        die("the test runs the frozen choice only: unset AVA_OPEN_MODE / AVA_OPEN_PICK");
    }
    std::env::set_var("AVA_OPEN_MODE", FROZEN_OPEN_MODE);
    std::env::set_var("AVA_OPEN_PICK", FROZEN_OPEN_PICK);
    let what = if dry { "DRY RUN of the test routine, on DEV" } else { "TEST (scored once)" };
    say!(out, "BLiMP open expectations, {what} -- source sha256 (compiled in): {}", sha256(SOURCE.as_bytes()));
    let set: Vec<&Pair> = pairs.iter().filter(|p| if dry { p.id % 5 == 0 } else { p.id % 5 != 0 }).collect();
    say!(out, "{} pairs", th(set.len() as u64));
    let r = reading(&mut out);
    let trig = learn_openers(&r, &mut out);
    let plain = stream_open(&r, &set, false, 3, false, &mut out, &[]);
    let open = stream_open(&r, &set, false, 3, false, &mut out, &trig);
    let at = |s: &[Snap]| s.iter().position(|x| x.label.starts_with("10,000,000")).unwrap_or_else(|| die("no 10M snapshot"));
    let (p, o) = (&plain[at(&plain)], &open[at(&open)]);
    let rp: Vec<bool> = (0..set.len()).map(|k| p.margin[k][2] > TIE).collect();
    let ro: Vec<bool> = (0..set.len()).map(|k| o.margin[k][3] > TIE).collect();
    let (np, no) = (rp.iter().filter(|&&x| x).count() as u64, ro.iter().filter(|&&x| x).count() as u64);
    let d: Vec<f64> = (0..set.len()).map(|k| ro[k] as u8 as f64 - rp[k] as u8 as f64).collect();
    let mut x = 1u64;
    let mut ms: Vec<f64> = (0..20_000).map(|_| (0..d.len()).map(|_| d[(splitmix(&mut x) % d.len() as u64) as usize]).sum::<f64>() / d.len() as f64).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (lo, hi) = (100.0 * ms[499], 100.0 * ms[19_499]);
    say!(out, "\nat 10M words: plain {:.2}%  open expectations {:.2}%  ({np} vs {no} of {})", pct(np, set.len() as u64), pct(no, set.len() as u64), set.len());
    say!(out, "difference {:+.2} points, paired bootstrap 95% CI {lo:+.2} .. {hi:+.2} (20,000 resamples, splitmix64 seed 1)", 100.0 * d.iter().sum::<f64>() / d.len() as f64);
    say!(out, "VERDICT (rule: CI above 0): {}", if lo > 0.0 { "PASS" } else { "FAIL" });
    let mut rows: Vec<(f64, String)> = (0..uids.len()).map(|u| {
        let ks: Vec<usize> = (0..set.len()).filter(|&k| set[k].uid == u).collect();
        let (a, b) = (ks.iter().filter(|&&k| rp[k]).count() as u64, ks.iter().filter(|&&k| ro[k]).count() as u64);
        (pct(b, ks.len() as u64) - pct(a, ks.len() as u64), format!("  {:<52} {:>6.1}% -> {:>6.1}%", uids[u], pct(a, ks.len() as u64), pct(b, ks.len() as u64)))
    }).collect();
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    say!(out, "set by set (plain -> open), most helped first:");
    for (_, l) in &rows { say!(out, "{l}"); }
    out
}
