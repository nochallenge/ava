// TESTING / DESIGN ONLY. (v31) The v31 copy of blimp_open.rs (v30, frozen there at sha256 7d416a81...).
// Lane B's grammar work goes here; blimp_open.rs is never edited.
//
// DEV ONLY. This copy has no test-scoring mode: test, dryrun, arc, open-test, open-dryrun, zorro-load and
// zorro-test were removed (with their loaders and frozen-routine code), so no BLiMP test pair (pairID % 5
// != 0) and no Zorro pair can be scored by accident. The BLiMP loader also keeps only the dev pairs. A
// test mode is added back only at the v31 freeze, under a pre-registration.
//
//   cargo run --release --example blimp_class -- check                    streaming == storing (no scores)
//   cargo run --release --example blimp_class -- dev                      WP6's two dev choices
//   cargo run --release --example blimp_class -- open-dev                 v30: plain vs open (AVA_OPEN_PICK=rate)
//   cargo run --release --example blimp_class -- probe [base new] [w.tsv] v31 dev tools, all three below
//   cargo run --release --example blimp_class -- halves|mirror|attrib [base new] [w.tsv]  one of them
//
// V31 DEV TOOLS (B0). Each compares two of her readings at 10M words (base and new; default plain and
// open, the v30 pair) on the dev pairs only:
//   halves  Dev split once, for good: half A = dev pairs with (pairID / 5) even, half B = odd. Every
//           report gives dev, half A and half B (choose on A, estimate on B).
//   mirror  Mirror groups: sets with one template whose contrasted word is in the good sentence on one
//           side and in the bad sentence on the other (list and rule in MIRRORS, written before any
//           score). A pure bias toward the word moves the sides in opposite directions, so the change in
//           (a1 + a2 - 100) / 2 is net discrimination and (a1 - a2) / 2 is the lean.
//   attrib  Per-word attribution: each sentence is cut against its partner (shared start, swapped
//           words, the next 3 shared places, later shared words, the end), and the change in each
//           word's ln P (new - base) says which positions moved the margin. With a path, every word of
//           every dev pair is written there as a TSV.
//
// FIX AFTER REVIEW (B0/B1). An end fewer than 3 places after the swap is now in next 3 (its context
// still sees the swap). The B1 table also splits the margin change by her opener state: the swap at
// places with no state and with a held one, next 3 where the two sentences' states match and where they
// differ, so the share where a held expectation acts is shown as a range. New descriptive prints: the
// filler-gap sets' pairs grouped by the openers among their swapped words, all their pairs with no
// selection by lost or won, and the mean ln P change per place by her opener state. Every table gives
// dev, half A and half B. The dead ARC-corpus reading and unused fields were removed (no output changes).
//
// V31 B2: WORD KINDS AND WORD ENDINGS AS CONTEXT. Each sits behind its own switch, off by default; with both off nothing
// below runs and every output is unchanged.
//   AVA_CLASS=1    her word kinds: K = 128 kinds by exchange clustering (Kneser & Ney 1993) of the words she read 3 or
//                  more times in her first 9M words (the words after 9M, to 10M, are held out of it); a rarer word shares
//                  one rare kind.
//   AVA_ENDINGS=1  her word endings, from her own word forms: an ending of 1-5 letters is hers once 100 different stems of
//                  3+ letters take it (stem and stem + ending both read in her first 9M words). A word's kind is the
//                  ending it carries over a stem she read (the one most stems take), or "none". No ending is listed.
//   Kinds are context only: she still predicts the next WORD. Each switch adds one block of Witten-Bell levels that read
//   the next word after the pattern of her previous 1..5 kinds, under every word context. The chain, most general first:
//   any word, the ENDINGS block (1..5 kinds), the CLASS block (1..5 kinds), then the word contexts exactly as before.
//   cargo run --release --example blimp_class -- kinds-check   the pre-registered check on her own reading (no BLiMP)
//   cargo run --release --example blimp_class -- kinds-dev     plain, open, open + each block that is on (both on: each
//                                                              alone and both), dev / half A / half B
//   The probe tools also read open+class, open+endings and open+both when their switches are on.
// TOOL FIXES (B2, from the lead's ruling on B0/B1; measuring only, no switch): word-matched mirror nets (pairs grouped by
// her openers among the swapped words, read from the contrasted word's side), printed after the pooled nets; and probe
// refuses a per-word path that does not end in .tsv (so "probe plain" cannot write a file named "plain").
//
// (v30 header follows.)
// (v30) Open expectations: a copy of blimp.rs (WP6, frozen there) with one new
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
// (The v30 file's Zorro modes, zorro-load and zorro-test, are not in this copy.)
//
// (WP6 header follows; its test, dryrun and arc modes are not in this copy.)
// (v29 WP6) Grammar from reading: BLiMP at 10M words.
//
//   cargo run --release --example blimp -- check               streaming counts == storing everything (no BLiMP scores)
//   cargo run --release --example blimp -- dev                 the two choices, made on dev (pairID % 5 == 0)
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
use std::io::{BufWriter, Write};
use std::rc::Rc;
use std::time::Instant;

const DATA: &str = "../data";
const SOURCE: &str = include_str!("blimp_class.rs");
const SEED: u64 = 81;
const MAIN: u64 = 10_000_000;
const MARKS: [u64; 4] = [1_000_000, 2_500_000, 5_000_000, MAIN];
const MAX_N: usize = 4;
const TIE: f64 = 1e-9;

// ---------------------------------------------------------------- frozen settings (WP6: the plain reading)
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
struct Pair { uid: usize, id: u32, good: String, bad: String }

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

fn load_blimp() -> (Vec<Pair>, Vec<String>) {
    let dir = format!("{DATA}/blimp");
    let mut files: Vec<String> = std::fs::read_dir(&dir).unwrap_or_else(|e| die(&format!("cannot read {dir}: {e}")))
        .flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|f| f.ends_with(".jsonl")).collect();
    files.sort();
    let (mut pairs, mut uids) = (Vec::new(), Vec::<String>::new());
    for f in &files {
        let raw = std::fs::read_to_string(format!("{dir}/{f}")).unwrap_or_else(|e| die(&format!("{f}: {e}")));
        for line in raw.lines().filter(|l| !l.trim().is_empty()) {
            let get = |k: &str| field(line, k).unwrap_or_else(|| die(&format!("{f}: no {k} in {line}")));
            let uid = get("UID");
            let ui = uids.iter().position(|u| *u == uid).unwrap_or_else(|| { uids.push(uid); uids.len() - 1 });
            let id: u32 = get("pairID").parse().unwrap_or_else(|_| die(&format!("{f}: bad pairID")));
            // v31: only dev pairs are kept (pairID % 5 == 0); a test pair is never held, so none can be scored
            if id % 5 != 0 { continue; }
            pairs.push(Pair { uid: ui, id, good: get("sentence_good"), bad: get("sentence_bad") });
        }
    }
    (pairs, uids)
}

// ---------------------------------------------------------------- what she reads
struct Reading { sents: Vec<Box<str>> }

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
    Reading { sents }
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
    blocks: Vec<Rc<KindMap>>,     // v31 B2: her kind blocks under the word contexts, most general first (empty: off)
    kst: Vec<Vec<Vec<u8>>>,       // v31 B2: per block, per scored sentence: the kind of each word
    kbuf: Vec<u64>,               // v31 B2: scratch for the kind keys of a place
}

fn seq_key(keys: &[u64]) -> u64 { keys.iter().fold(mix64(keys.len() as u64), |h, &k| mix64(h ^ k)) }

impl Her {
    fn new(maxn: usize, lower: bool, pairs: &[&Pair]) -> Her { Her::new_open(maxn, lower, pairs, Vec::new()) }

    fn new_open(maxn: usize, lower: bool, pairs: &[&Pair], trig: Vec<u64>) -> Her { Her::new_kinds(maxn, lower, pairs, trig, Vec::new()) }

    /// v31 B2: as new_open, with kind blocks under the word contexts (none: exactly new_open's reading)
    fn new_kinds(maxn: usize, lower: bool, pairs: &[&Pair], trig: Vec<u64>, blocks: Vec<Rc<KindMap>>) -> Her {
        if !blocks.is_empty() && maxn > 3 { die("kind blocks need word contexts of at most 3 words (their keys use bits 61 and 62)"); }
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
        let kst: Vec<Vec<Vec<u8>>> = blocks.iter().map(|b| sents.iter().map(|(_, keys)| keys.iter().map(|&k| b.kind(k)).collect()).collect()).collect();
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
                // v31 B2: the kind patterns before this place, block by block (none when off)
                for (b, blk) in blocks.iter().enumerate() {
                    kind_keys(&kst[b][si], i, blk.mark, &mut buf);
                    for &k in &buf {
                        let next = ctx.len() as u32;
                        let s = *ctx.entry(k).or_insert(next);
                        pair.entry((s as u64) << 32 | t as u64).or_insert(0);
                    }
                }
            }
        }
        let (nctx, nv, ns) = (ctx.len(), vocab.len() + 2, sents.len());
        Her { maxn, trig, st, vocab, sents, whole, ctx, c: vec![0; nctx], d: vec![0; nctx], follow: MixSet::default(), pair,
              uni: vec![0; nv], n: 0, types: MixSet::default(), words: 0, nsent: 0, seen: vec![false; ns], blocks, kst, kbuf: Vec::new() }
    }

    /// read one sentence: every word and the end, after each of its contexts the BLiMP sentences use
    fn read(&mut self, w: &[String], ids: &mut Vec<u32>, keys: &mut Vec<u64>, buf: &mut Vec<u64>) {
        keys.clear();
        ids.clear();
        for x in w { let k = wkey(x); keys.push(k); ids.push(self.vocab.get(&k).copied().unwrap_or(NONE)); }
        self.nsent += 1;
        self.words += w.len() as u64;
        let stv = states(keys, &self.trig);
        // v31 B2: each block's kind of each word (no block, nothing: the reading is exactly as before)
        let kv: Vec<Vec<u8>> = self.blocks.iter().map(|b| keys.iter().map(|&k| b.kind(k)).collect()).collect();
        let mut kb = std::mem::take(&mut self.kbuf);
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
            // v31 B2: the kind patterns, block by block; within a block a longer pattern holds the shorter one
            for b in 0..kv.len() {
                let mark = self.blocks[b].mark;
                kind_keys(&kv[b], i, mark, &mut kb);
                for &k in kb.iter() {
                    let Some(&s) = self.ctx.get(&k) else { break };
                    self.c[s as usize] += 1;
                    if self.follow.insert(mix64(tk ^ (s as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))) { self.d[s as usize] += 1; }
                    if tid != NONE { if let Some(x) = self.pair.get_mut(&((s as u64) << 32 | tid as u64)) { *x += 1; } }
                }
            }
        }
        self.kbuf = kb;
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
        let mut kb = Vec::new();
        for i in 0..=ids.len() {
            let t = if i < ids.len() { ids[i] } else { EOS };
            // any word, over an even share for every word read plus one never met
            let mut p = (1.0 - lam0) / (v + 1.0);
            if nn > 0.0 { p += lam0 * self.uni[t as usize] as f64 / nn; }
            p = self.kinds_p(si, i, t, p, &mut kb);
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

    /// v31 (attribution): ln P of each word and the end of a scored sentence, at her longest level.
    /// The same loop as `score`, kept word by word; summed in order it is score's last level exactly.
    fn word_lnp(&self, si: usize, buf: &mut Vec<u64>) -> Vec<f64> {
        let ids = &self.sents[si].0;
        let nlev = self.maxn + !self.trig.is_empty() as usize;
        let (nn, v) = (self.n as f64, self.types.len() as f64);
        let lam0 = if nn > 0.0 { nn / (nn + v) } else { 0.0 };
        let mut out = Vec::with_capacity(ids.len() + 1);
        let mut kb = Vec::new();
        for i in 0..=ids.len() {
            let t = if i < ids.len() { ids[i] } else { EOS };
            let mut p = (1.0 - lam0) / (v + 1.0);
            if nn > 0.0 { p += lam0 * self.uni[t as usize] as f64 / nn; }
            p = self.kinds_p(si, i, t, p, &mut kb);
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
            }
            out.push(p.ln());
        }
        out
    }

    /// v31 B2: P of word t at place i of scored sentence si after the kind blocks, read up from p (the any-word level):
    /// each block's patterns of 1..5 kinds, most general first, by the same Witten-Bell step as the word contexts. A
    /// pattern she never read is passed over. With no block, p comes back untouched.
    fn kinds_p(&self, si: usize, i: usize, t: u32, mut p: f64, kb: &mut Vec<u64>) -> f64 {
        for b in 0..self.blocks.len() {
            kind_keys(&self.kst[b][si], i, self.blocks[b].mark, kb);
            for k in kb.iter() {
                if let Some(&s) = self.ctx.get(k) {
                    let (c, d) = (self.c[s as usize] as f64, self.d[s as usize] as f64);
                    if c > 0.0 {
                        let lam = c / (c + d);
                        let ct = self.pair.get(&((s as u64) << 32 | t as u64)).copied().unwrap_or(0) as f64;
                        p = lam * ct / c + (1.0 - lam) * p;
                    }
                }
            }
        }
        p
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
        Snap { label, margin, seen: self.seen.clone(), oov, tok }
    }
}

struct Snap { label: String, margin: Vec<[f64; MAX_N]>, seen: Vec<bool>, oov: u64, tok: u64 }

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

/// her reading, streamed. Snapshots at each mark she reaches, and at the end. (v31: the ARC-corpus
/// reading of the WP6 `arc` mode is not in this copy.)
fn stream(r: &Reading, pairs: &[&Pair], lower: bool, maxn: usize, out: &mut String) -> Vec<Snap> { stream_open(r, pairs, lower, maxn, out, &[]) }

fn stream_open(r: &Reading, pairs: &[&Pair], lower: bool, maxn: usize, out: &mut String, trig: &[u64]) -> Vec<Snap> {
    let t0 = Instant::now();
    let mut her = Her::new_open(maxn, lower, pairs, trig.to_vec());
    say!(out, "  [{}] {} scored sentences, {} BLiMP word types, {} contexts and {} (context, word) pairs to count",
         if lower { "lower-cased" } else { "cased" }, th(her.sents.len() as u64), th(her.vocab.len() as u64), th(her.ctx.len() as u64), th(her.pair.len() as u64));
    let (mut ids, mut keys, mut buf) = (Vec::new(), Vec::new(), Vec::new());
    let mut snaps = Vec::new();
    let mut next = 0;
    for (k, s) in r.sents.iter().enumerate() {
        let w = words(s, lower);
        while next < MARKS.len() && her.words + w.len() as u64 > MARKS[next] {
            snaps.push(her.snap(format!("{} words", th(MARKS[next]))));
            next += 1;
        }
        her.read(&w, &mut ids, &mut keys, &mut buf);
        if (k + 1) % 500_000 == 0 { ram_guard("reading"); }
    }
    snaps.push(her.snap(format!("all ({} words)", th(her.words))));
    say!(out, "  read {} sentences, {} words in {:.1}s; {} contexts read at least once; peak RAM {:.2} GB",
         th(her.nsent), th(her.words), t0.elapsed().as_secs_f64(), th(her.c.iter().filter(|&&c| c > 0).count() as u64), peak_gb());
    snaps
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
    // v31 B2: the pre-registered check reads her own reading only; no BLiMP pair is loaded for it
    if mode == "kinds-check" {
        kinds_check();
        println!("total {:.1}s, peak RAM {:.2} GB", t0.elapsed().as_secs_f64(), peak_gb());
        return;
    }
    let (pairs, uids) = load_blimp();
    if pairs.len() != 13_400 || uids.len() != 67 { die(&format!("expected 67 sets x 200 dev pairs, found {} sets, {} pairs", uids.len(), pairs.len())); }
    match mode {
        "check" => check(&pairs),
        "dev" => dev(&pairs),
        "open-dev" => open_dev(&pairs, &uids),
        "kinds-dev" => kinds_dev(&pairs, &uids),
        "probe" | "halves" | "mirror" | "attrib" => {
            // [base new] [per-word.tsv]: two reading names (default plain and open), then a path
            let mut rest: Vec<&str> = a.iter().skip(2).map(|s| s.as_str()).collect();
            let (mut base, mut new) = (Reader::Plain, Reader::Open);
            if rest.len() >= 2 {
                if let (Some(b), Some(n)) = (Reader::named(rest[0]), Reader::named(rest[1])) { base = b; new = n; rest.drain(..2); }
            }
            if rest.len() > 1 { die("usage: blimp_class probe|halves|mirror|attrib [base new] [per-word.tsv]"); }
            // v31 B2 tool fix: the per-word path must end in .tsv, so a lone reading name ("probe plain") is never
            // taken for a path and written over
            if let Some(p) = rest.first() {
                if !p.ends_with(".tsv") { die(&format!("blimp_class {mode}: the per-word path must end in .tsv (got \"{p}\"); give two reading names, or none")); }
            }
            probe(&pairs, &uids, mode, base, new, rest.first().copied());
        }
        _ => die("usage: blimp_class check | dev | open-dev | probe|halves|mirror|attrib [base new] [per-word.tsv] | kinds-check | kinds-dev   (dev pairs only)"),
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
        let snaps = stream(&r, &dev, lower, MAX_N, &mut out);
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
    let plain = stream_open(&r, &dev, false, 3, &mut out, &[]);
    let open = stream_open(&r, &dev, false, 3, &mut out, &trig);
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

// ---------------------------------------------------------------- v31 dev tools (B0): halves, mirror groups, per-word attribution
/// the readings the dev tools compare: plain (WP6's frozen reading: 3 words, cased) and open (v30's
/// frozen dev choice). Later v31 steps add their readings here, each behind its own switch.
/// v31 B2: open+class, open+endings and open+both (v30's open reading with her kind blocks under it); each name
/// exists only when its switches are on (AVA_CLASS=1, AVA_ENDINGS=1, both).
#[derive(Clone, Copy, PartialEq)]
enum Reader { Plain, Open, Class, Endings, Both }

impl Reader {
    fn named(s: &str) -> Option<Reader> {
        match s {
            "plain" => Some(Reader::Plain),
            "open" => Some(Reader::Open),
            "open+class" if class_on() => Some(Reader::Class),
            "open+endings" if endings_on() => Some(Reader::Endings),
            "open+both" if class_on() && endings_on() => Some(Reader::Both),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self { Reader::Plain => "plain", Reader::Open => "open", Reader::Class => "open+class", Reader::Endings => "open+endings", Reader::Both => "open+both" }
    }
    /// v31 B2: its kind blocks, most general first (ENDINGS under CLASS)
    fn blocks(self, k: &Kinds) -> Vec<Rc<KindMap>> {
        let need = |m: &Option<Rc<KindMap>>| m.clone().unwrap_or_else(|| die("a kinds reading needs its kinds learned"));
        match self {
            Reader::Plain | Reader::Open => Vec::new(),
            Reader::Class => vec![need(&k.class)],
            Reader::Endings => vec![need(&k.end)],
            Reader::Both => vec![need(&k.end), need(&k.class)],
        }
    }
    fn needs_class(self) -> bool { matches!(self, Reader::Class | Reader::Both) }
    fn needs_endings(self) -> bool { matches!(self, Reader::Endings | Reader::Both) }
}

/// the fixed dev halves (B0): half A = dev pairs with (pairID / 5) even, half B = odd
fn half_b(p: &Pair) -> bool { (p.id / 5) % 2 == 1 }
/// the three parts every dev report gives: all of dev, half A, half B
const PARTS: [&str; 3] = ["dev", "half A", "half B"];
fn in_part(part: usize, p: &Pair) -> bool { part == 0 || (part == 2) == half_b(p) }

/// Mirror groups (B0; the list and its rule were written to the lane log before this tool gave any
/// score). Two sets mirror when their names mark them as two forms of one paradigm, they share a
/// template, and the contrasted word or word kind is in the GOOD sentence in one (side 1) and in the
/// BAD sentence in the other (side 2). A list used only to measure: nothing she reads or learns sees it.
const MIRRORS: [(&str, &[&str], &[&str]); 5] = [
    ("M1 wh vs that", &["wh_vs_that_with_gap"], &["wh_vs_that_no_gap"]),
    ("M2 wh vs that, long", &["wh_vs_that_with_gap_long_distance"], &["wh_vs_that_no_gap_long_distance"]),
    ("M3 tough vs raising", &["tough_vs_raising_1"], &["tough_vs_raising_2"]),
    ("M4 irregular forms", &["irregular_past_participle_adjectives"], &["irregular_past_participle_verbs"]),
    ("M5 transitivity", &["intransitive"], &["transitive"]),
];
/// the wide filler-gap group (the wh-word in the good sentence, or in the bad one): reported apart and
/// not in the total, because its two sides do not share one template
const MIRROR_WIDE: [(&str, &[&str], &[&str]); 1] = [
    ("W filler-gap, all 7", &["wh_vs_that_with_gap", "wh_vs_that_with_gap_long_distance"],
     &["wh_vs_that_no_gap", "wh_vs_that_no_gap_long_distance", "wh_questions_object_gap", "wh_questions_subject_gap",
       "wh_questions_subject_gap_long_distance"]),
];

/// the regions of a sentence cut against its partner: the shared words at the start (they add 0), the
/// swapped words, the next FROZEN_N shared places (their context still sees the swap), the later shared
/// words (the plain reading reads them the same in both, so only an opener state can move them), the end
const REGIONS: [&str; 5] = ["before", "swap", "next 3", "later", "end"];

/// the region of word i (i == len: the end) of a sentence of len words that shares its first `pre` and
/// last `suf` words with its partner. The end is predicted like a word, so an end fewer than FROZEN_N
/// places after the swap is in next 3: its context still sees the swap (v31 fix, after review; before
/// it, every end was labelled "end").
fn region(i: usize, len: usize, pre: usize, suf: usize) -> usize {
    if i < pre { 0 } else if i < len - suf { 1 } else if i - (len - suf) < FROZEN_N { 2 } else if i == len { 4 } else { 3 }
}

/// how many words two sentences share at the start, and then at the end (the two never overlap)
fn shared(g: &[String], b: &[String]) -> (usize, usize) {
    let pre = g.iter().zip(b).take_while(|(x, y)| x == y).count();
    let room = g.len().min(b.len()) - pre;
    let suf = g.iter().rev().zip(b.iter().rev()).take(room).take_while(|(x, y)| x == y).count();
    (pre, suf)
}

/// one reading of every dev sentence at 10M words (2k good, 2k + 1 bad): ln P of each word and the end,
/// and the opener state of each place (empty for the plain reading)
struct Lnp { w: Vec<Vec<f64>>, st: Vec<Vec<u8>> }

/// her reading up to the 10M-word mark, the same sentences as stream_open's 10M snapshot (every
/// sentence before the first that would take her past 10M words), then every scored word's ln P
fn read_main(r: &Reading, dev: &[&Pair], reader: Reader, trig: &[u64], kinds: &Kinds, out: &mut String) -> Lnp {
    let t0 = Instant::now();
    // every reading but plain holds v30's openers (v31 B2: the kinds readings are open + kind blocks)
    let tr = if reader != Reader::Plain { trig.to_vec() } else { Vec::new() };
    let mut her = Her::new_kinds(FROZEN_N, FROZEN_LOWER, dev, tr, reader.blocks(kinds));
    let (mut ids, mut keys, mut buf) = (Vec::new(), Vec::new(), Vec::new());
    for (k, s) in r.sents.iter().enumerate() {
        let w = words(s, FROZEN_LOWER);
        if her.words + w.len() as u64 > MAIN { break; }
        her.read(&w, &mut ids, &mut keys, &mut buf);
        if (k + 1) % 500_000 == 0 { ram_guard("reading"); }
    }
    say!(out, "  [{}] {} contexts to count; read {} sentences, {} words in {:.1}s; peak RAM {:.2} GB", reader.name(),
         th(her.ctx.len() as u64), th(her.nsent), th(her.words), t0.elapsed().as_secs_f64(), peak_gb());
    let w = (0..her.sents.len()).map(|si| her.word_lnp(si, &mut buf)).collect();
    Lnp { w, st: std::mem::take(&mut her.st) }
}

/// a paired bootstrap of a mean difference, as open_dev does it (20,000 resamples, splitmix64 seed 1):
/// the 95% interval, in points
fn boot_mean(d: &[f64]) -> (f64, f64) {
    let mut x = 1u64;
    let mut ms: Vec<f64> = (0..20_000).map(|_| (0..d.len()).map(|_| d[(splitmix(&mut x) % d.len() as u64) as usize]).sum::<f64>() / d.len() as f64).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (100.0 * ms[499], 100.0 * ms[19_499])
}

/// mirror groups on one part of dev: per group the net discrimination (a1 + a2 - 100) / 2 and the lean
/// (a1 - a2) / 2 of base and new (a1, a2 the side means), and a paired bootstrap of their changes (each
/// set's pairs resampled within the set; 20,000 resamples, splitmix64 seed 1). With `total`, the mean
/// net over the groups too. Returns the lines.
fn mirror_part(groups: &[(&str, &[&str], &[&str])], total: bool, uids: &[String], dev: &[&Pair], rb: &[bool], rn: &[bool], part: usize) -> Vec<String> {
    let uid = |name: &str| uids.iter().position(|u| u == name).unwrap_or_else(|| die(&format!("no BLiMP set {name}")));
    let gs: Vec<(Vec<usize>, Vec<usize>)> = groups.iter().map(|g| (g.1.iter().map(|s| uid(s)).collect(), g.2.iter().map(|s| uid(s)).collect())).collect();
    let mut sets: Vec<usize> = gs.iter().flat_map(|g| g.0.iter().chain(&g.1).copied()).collect();
    sets.sort_unstable();
    sets.dedup();
    let idx: Vec<Vec<usize>> = sets.iter().map(|&u| (0..dev.len()).filter(|&k| dev[k].uid == u && in_part(part, dev[k])).collect()).collect();
    let at = |u: usize| sets.iter().position(|&s| s == u).unwrap();
    // per group [net base, net new, lean base, lean new], from each set's (base, new) accuracy
    let measure = |acc: &[(f64, f64)]| -> Vec<[f64; 4]> {
        gs.iter().map(|(s1, s2)| {
            let mean = |s: &[usize], new: bool| s.iter().map(|&u| if new { acc[at(u)].1 } else { acc[at(u)].0 }).sum::<f64>() / s.len() as f64;
            let (b1, b2, n1, n2) = (mean(s1, false), mean(s2, false), mean(s1, true), mean(s2, true));
            [(b1 + b2 - 100.0) / 2.0, (n1 + n2 - 100.0) / 2.0, (b1 - b2) / 2.0, (n1 - n2) / 2.0]
        }).collect()
    };
    let acc_of = |ks: &[usize]| (pct(ks.iter().filter(|&&k| rb[k]).count() as u64, ks.len() as u64), pct(ks.iter().filter(|&&k| rn[k]).count() as u64, ks.len() as u64));
    let point = measure(&idx.iter().map(|ks| acc_of(ks)).collect::<Vec<_>>());
    let ng = gs.len();
    let (mut bnet, mut blean): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (vec![Vec::with_capacity(20_000); ng + 1], vec![Vec::with_capacity(20_000); ng]);
    let mut x = 1u64;
    let mut acc = vec![(0.0, 0.0); sets.len()];
    for _ in 0..20_000 {
        for (j, ks) in idx.iter().enumerate() {
            let (mut cb, mut cn) = (0u64, 0u64);
            for _ in 0..ks.len() { let k = ks[(splitmix(&mut x) % ks.len() as u64) as usize]; cb += rb[k] as u64; cn += rn[k] as u64; }
            acc[j] = (pct(cb, ks.len() as u64), pct(cn, ks.len() as u64));
        }
        let m = measure(&acc);
        for g in 0..ng { bnet[g].push(m[g][1] - m[g][0]); blean[g].push(m[g][3] - m[g][2]); }
        bnet[ng].push(m.iter().map(|v| v[1] - v[0]).sum::<f64>() / ng as f64);
    }
    let ci = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); (v[499], v[19_499]) };
    let mut lines = Vec::new();
    for (g, grp) in groups.iter().enumerate() {
        let ((nl, nh), (ll, lh)) = (ci(&mut bnet[g]), ci(&mut blean[g]));
        let p = point[g];
        lines.push(format!("    {:<22} net {:+6.2} -> {:+6.2}  change {:+6.2} (95% CI {:+6.2} .. {:+6.2})   lean {:+6.2} -> {:+6.2}  change {:+6.2} ({:+6.2} .. {:+6.2})",
                           grp.0, p[0], p[1], p[1] - p[0], nl, nh, p[2], p[3], p[3] - p[2], ll, lh));
    }
    if total {
        let (tl, thi) = ci(&mut bnet[ng]);
        let (tb, tn) = (point.iter().map(|v| v[0]).sum::<f64>() / ng as f64, point.iter().map(|v| v[1]).sum::<f64>() / ng as f64);
        lines.push(format!("    {:<22} net {:+6.2} -> {:+6.2}  change {:+6.2} (95% CI {:+6.2} .. {:+6.2})", format!("TOTAL (mean of {ng})"), tb, tn, tn - tb, tl, thi));
    }
    lines
}

/// v31 dev tools on the dev pairs only (see the header): `part` is probe (all three), halves, mirror or
/// attrib; base and new are two of her readings at 10M words; `tsv`, a path for every word of every pair
fn probe(pairs: &[Pair], uids: &[String], part: &str, base: Reader, new: Reader, tsv: Option<&str>) {
    let mut out = String::new();
    say!(out, "source sha256 (compiled in): {}", sha256(SOURCE.as_bytes()));
    // the open reading is v30's frozen dev choice, whatever the environment says
    if std::env::var("AVA_OPEN_MODE").map_or(false, |v| v != FROZEN_OPEN_MODE) || std::env::var("AVA_OPEN_PICK").map_or(false, |v| v != FROZEN_OPEN_PICK) {
        die("the dev tools read open as v30's frozen choice: unset AVA_OPEN_MODE / AVA_OPEN_PICK (or set them to last / rate)");
    }
    std::env::set_var("AVA_OPEN_MODE", FROZEN_OPEN_MODE);
    std::env::set_var("AVA_OPEN_PICK", FROZEN_OPEN_PICK);
    let dev: Vec<&Pair> = pairs.iter().filter(|p| p.id % 5 == 0).collect();
    let nb = dev.iter().filter(|p| half_b(p)).count() as u64;
    say!(out, "v31 dev tools ({part}): base = her {} reading, new = her {} reading, both at 10M words", base.name(), new.name());
    say!(out, "DEV: {} pairs (pairID % 5 == 0); test pairs are not read. Half A ((pairID / 5) even): {}; half B (odd): {}",
         th(dev.len() as u64), th(dev.len() as u64 - nb), th(nb));
    let r = reading(&mut out);
    let trig = if base != Reader::Plain || new != Reader::Plain { learn_openers(&r, &mut out) } else { Vec::new() };
    // v31 B2: her kinds, learned only for a kinds reading (and then only the blocks it reads)
    let (nc, ne) = (base.needs_class() || new.needs_class(), base.needs_endings() || new.needs_endings());
    let extra: Vec<String> = if ne { dev.iter().flat_map(|p| words(&p.good, FROZEN_LOWER).into_iter().chain(words(&p.bad, FROZEN_LOWER))).collect() } else { Vec::new() };
    let kinds = Kinds::learn(&r, nc, ne, &extra, &mut out);
    let lb = read_main(&r, &dev, base, &trig, &kinds, &mut out);
    let ln = read_main(&r, &dev, new, &trig, &kinds, &mut out);
    drop(r);
    let total = |v: &[f64]| v.iter().fold(0.0, |a, &x| a + x);
    let mb: Vec<f64> = (0..dev.len()).map(|k| total(&lb.w[2 * k]) - total(&lb.w[2 * k + 1])).collect();
    let mn: Vec<f64> = (0..dev.len()).map(|k| total(&ln.w[2 * k]) - total(&ln.w[2 * k + 1])).collect();
    let (rb, rn): (Vec<bool>, Vec<bool>) = (mb.iter().map(|&m| m > TIE).collect(), mn.iter().map(|&m| m > TIE).collect());
    let count = |r: &[bool], ks: &[usize]| ks.iter().filter(|&&k| r[k]).count() as u64;
    let part_idx: Vec<Vec<usize>> = (0..3).map(|q| (0..dev.len()).filter(|&k| in_part(q, dev[k])).collect()).collect();
    let set_idx = |u: usize, q: usize| -> Vec<usize> { (0..dev.len()).filter(|&k| dev[k].uid == u && in_part(q, dev[k])).collect() };
    let (bn, nn) = (base.name(), new.name());

    // sets sorted as open_dev sorts them: by the dev change, most helped first (ties in file order)
    let mut order: Vec<(f64, usize)> = (0..uids.len()).map(|u| {
        let ks = set_idx(u, 0);
        (pct(count(&rn, &ks), ks.len() as u64) - pct(count(&rb, &ks), ks.len() as u64), u)
    }).collect();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let acc3 = |u: usize| -> String {
        (0..3).map(|q| { let ks = set_idx(u, q); format!("{:>5.1} -> {:>5.1}", pct(count(&rb, &ks), ks.len() as u64), pct(count(&rn, &ks), ks.len() as u64)) })
            .collect::<Vec<_>>().join(" | ")
    };

    // (a) dev and its halves
    if part == "probe" || part == "halves" {
        say!(out, "\nDEV AND ITS FIXED HALVES ({bn} -> {nn}, at 10M words)");
        for (q, ks) in part_idx.iter().enumerate() {
            let (a, b) = (count(&rb, ks), count(&rn, ks));
            let d: Vec<f64> = ks.iter().map(|&k| rn[k] as u8 as f64 - rb[k] as u8 as f64).collect();
            let (lo, hi) = boot_mean(&d);
            say!(out, "  {:<7} {:>6} pairs: {bn} {:.2}%  {nn} {:.2}%  ({a} vs {b})  difference {:+.2} points, 95% CI {lo:+.2} .. {hi:+.2}",
                 PARTS[q], th(ks.len() as u64), pct(a, ks.len() as u64), pct(b, ks.len() as u64), pct(b, ks.len() as u64) - pct(a, ks.len() as u64));
        }
        say!(out, "set by set ({bn} -> {nn}), most helped on dev first:  {:>14} | {:>14} | {:>14}", "dev", "half A", "half B");
        for &(_, u) in &order { say!(out, "  {:<52} {}", uids[u], acc3(u)); }
    }

    // (c) mirror groups
    if part == "probe" || part == "mirror" {
        say!(out, "\nMIRROR GROUPS ({bn} -> {nn}). a1, a2 = side means (side 1: the word in the good sentence; side 2: in the bad one)");
        say!(out, "  net discrimination = (a1 + a2 - 100) / 2; lean toward the side-1 word = (a1 - a2) / 2. A pure bias moves lean, not net.");
        say!(out, "  the sets (accuracy {bn} -> {nn}):  {:>14} | {:>14} | {:>14}", "dev", "half A", "half B");
        for grp in MIRRORS.iter().chain(MIRROR_WIDE.iter()) {
            for (side, list) in [(1, grp.1), (2, grp.2)] {
                for s in list.iter() {
                    let u = uids.iter().position(|x| x == s).unwrap_or_else(|| die(&format!("no BLiMP set {s}")));
                    say!(out, "  {:<22} side {side}  {:<44} {}", grp.0, s, acc3(u));
                }
            }
        }
        for q in 0..3 {
            say!(out, "  {}:", PARTS[q]);
            for l in mirror_part(&MIRRORS, true, uids, &dev, &rb, &rn, q) { say!(out, "{l}"); }
            for l in mirror_part(&MIRROR_WIDE, false, uids, &dev, &rb, &rn, q) { say!(out, "{l}  (apart)"); }
        }
        // v31 B2 tool fix (the lead's ruling on B0/B1): the word-matched nets, judged from now on
        let labs = opener_labels(&dev, &trig);
        for l in matched_report(&labs, uids, &dev, &rb, &rn, bn, nn) { say!(out, "{l}"); }
    }

    // (b) per-word attribution
    if part == "probe" || part == "attrib" {
        let wds: Vec<(Vec<String>, Vec<String>)> = dev.iter().map(|p| (words(&p.good, FROZEN_LOWER), words(&p.bad, FROZEN_LOWER))).collect();
        // the opener states come from whichever reading has them (new first; with neither, every place is 0)
        let st = if ln.st.iter().any(|s| !s.is_empty()) { &ln.st } else { &lb.st };
        let state = |si: usize, i: usize| -> usize { if st[si].is_empty() { 0 } else { st[si][i] as usize } };
        // an opener's name, from the dev sentences' words (an opener never met in them is never a state here)
        let names: MixMap<u64, String> = wds.iter().flat_map(|(g, b)| g.iter().chain(b.iter())).filter(|w| trig.contains(&wkey(w))).map(|w| (wkey(w), w.clone())).collect();
        let sname = |s: usize| -> String { if s == 0 { "-".into() } else { names.get(&trig[s - 1]).cloned().unwrap_or_else(|| "?".into()) } };
        // per pair: the cut, and per region the change (new - base) in summed ln P on the good and the bad side.
        // `parts` splits the margin change by her opener state (added in the fix, after review): the swap at
        // places with no state and with a held one; next 3 where the two sentences' states match and where
        // they differ; later; the end. Shared places sit at the same distance from both ends, so they pair
        // up; at later places and the end, matching states add exactly 0 (`same0` checks it).
        // `late`: the margin change at later places and the end, split into the last 2 words with the end,
        // and the later words before them (a descriptive split added after the first B1 run).
        // `sw`: her openers among the swapped words, good / bad.
        struct At { pre: usize, suf: usize, dg: [f64; 5], db: [f64; 5], parts: [f64; 6], sdiff: bool, late: [f64; 2], same0: f64, sw: String }
        let ats: Vec<At> = (0..dev.len()).map(|k| {
            let (g, b) = (&wds[k].0, &wds[k].1);
            let (pre, suf) = shared(g, b);
            let dw = |side: usize, i: usize| ln.w[2 * k + side][i] - lb.w[2 * k + side][i];
            let (mut dg, mut db, mut parts) = ([0.0; 5], [0.0; 5], [0.0; 6]);
            for i in 0..=g.len() {
                let r = region(i, g.len(), pre, suf);
                dg[r] += dw(0, i);
                if r == 1 { parts[(state(2 * k, i) != 0) as usize] += dw(0, i); }
            }
            for i in 0..=b.len() {
                let r = region(i, b.len(), pre, suf);
                db[r] += dw(1, i);
                if r == 1 { parts[(state(2 * k + 1, i) != 0) as usize] -= dw(1, i); }
            }
            // the shared places after the swap, the end included (region 2, 3 or 4, the same in both)
            let (mut late, mut same0, mut sdiff) = ([0.0; 2], 0.0f64, false);
            for j in 0..=suf {
                let (i, h) = (g.len() - j, b.len() - j);
                let x = dw(0, i) - dw(1, h);
                let differ = state(2 * k, i) != state(2 * k + 1, h);
                sdiff |= differ;
                match region(i, g.len(), pre, suf) {
                    2 => parts[2 + differ as usize] += x,
                    r => {
                        parts[r + 1] += x;
                        late[(j > 2) as usize] += x;
                        if !differ { same0 = same0.max(x.abs()); }
                    }
                }
            }
            let ops = |w: &[String]| -> String {
                let v: Vec<&str> = w[pre..w.len() - suf].iter().filter(|x| trig.contains(&wkey(x))).map(|x| x.as_str()).collect();
                if v.is_empty() { "-".into() } else { v.join("+") }
            };
            At { pre, suf, dg, db, parts, sdiff, late, same0, sw: format!("{} / {}", ops(g), ops(b)) }
        }).collect();
        let dm = |a: &At, r: usize| a.dg[r] - a.db[r];
        let worst_before = ats.iter().map(|a| dm(a, 0).abs()).fold(0.0, f64::max);
        let worst_sum = (0..dev.len()).map(|k| ((0..5).map(|r| dm(&ats[k], r)).sum::<f64>() - (mn[k] - mb[k])).abs()).fold(0.0, f64::max);
        let worst_parts = (0..dev.len()).map(|k| (ats[k].parts.iter().sum::<f64>() - (mn[k] - mb[k])).abs()).fold(0.0, f64::max);
        let worst_same = ats.iter().map(|a| a.same0).fold(0.0, f64::max);
        say!(out, "\nPER-WORD ATTRIBUTION ({bn} -> {nn}): the change in margin, new - base, in nats, by where the words sit");
        say!(out, "  regions: before (shared start), swap (the words that differ), next 3 (shared places whose 3-word context still");
        say!(out, "  sees the swap; an end fewer than 3 places after the swap is one of them), later (shared words further on: only an");
        say!(out, "  opener state can move them), end (the sentence end, 3 or more places after the swap)");
        say!(out, "  checks: largest |change| from the shared start {worst_before:.1e} (must be 0); largest |regions - margin change| {worst_sum:.1e};");
        say!(out, "  largest |state parts - margin change| {worst_parts:.1e}; largest |change| at a later place or the end where the two");
        say!(out, "  sentences' states match {worst_same:.1e} (must be 0)");
        // every set: the mean change per pair by region, and how many pairs were lost and won; dev, then each half
        let mean_r = |ks: &[usize], r: usize| if ks.is_empty() { 0.0 } else { ks.iter().map(|&k| dm(&ats[k], r)).sum::<f64>() / ks.len() as f64 };
        for q in 0..3 {
            say!(out, "  {:<7} set (most helped on dev first)                    acc change     swap  next 3   later     end   lost  won", PARTS[q]);
            for &(_, u) in &order {
                let ks = set_idx(u, q);
                let lost = ks.iter().filter(|&&k| rb[k] && !rn[k]).count();
                let won = ks.iter().filter(|&&k| !rb[k] && rn[k]).count();
                say!(out, "  {:<7} {:<52} {:+6.1}  {:+7.3} {:+7.3} {:+7.3} {:+7.3} {:>6} {:>4}", PARTS[q], uids[u],
                     pct(count(&rn, &ks), ks.len() as u64) - pct(count(&rb, &ks), ks.len() as u64), mean_r(&ks, 1), mean_r(&ks, 2), mean_r(&ks, 3), mean_r(&ks, 4), lost, won);
            }
        }
        // focus: the 6 sets the new reading hurt most and the 6 it helped most on dev (found here, not by
        // hand), and the 7 filler-gap sets
        const NFOCUS: usize = 6;
        let helped: Vec<usize> = order.iter().take(NFOCUS).map(|x| x.1).collect();
        let hurt: Vec<usize> = order.iter().rev().take(NFOCUS).map(|x| x.1).collect();
        let fg: Vec<usize> = MIRROR_WIDE[0].1.iter().chain(MIRROR_WIDE[0].2.iter()).map(|s| uids.iter().position(|x| x == s).unwrap()).collect();
        say!(out, "\n  focus: hurt most: {}", hurt.iter().map(|&u| uids[u].as_str()).collect::<Vec<_>>().join(", "));
        say!(out, "         helped most: {}", helped.iter().map(|&u| uids[u].as_str()).collect::<Vec<_>>().join(", "));
        let fmt_r = |v: [f64; 5]| format!("{:+8.3} {:+8.3} {:+8.3} {:+8.3}  {:+8.3}", v[1], v[2], v[3], v[4], v[1] + v[2] + v[3] + v[4]);
        let word_line = |k: usize, side: usize| -> String {
            let (w, (pre, suf)) = (if side == 0 { &wds[k].0 } else { &wds[k].1 }, (ats[k].pre, ats[k].suf));
            let mut s = String::new();
            let mut last = 0;
            for i in 0..=w.len() {
                let rg = region(i, w.len(), pre, suf);
                if i > 0 && rg != last { s.push_str("| "); }
                last = rg;
                s.push_str(&format!("{}({:+.2}) ", if i < w.len() { w[i].as_str() } else { "</s>" }, ln.w[2 * k + side][i] - lb.w[2 * k + side][i]));
            }
            s
        };
        let mv = |ks: &[usize], f: &dyn Fn(&At, usize) -> f64| -> [f64; 5] {
            let mut v = [0.0; 5];
            if !ks.is_empty() { for r in 0..5 { v[r] = ks.iter().map(|&k| f(&ats[k], r)).sum::<f64>() / ks.len() as f64; } }
            v
        };
        for &(_, u) in order.iter().filter(|x| hurt.contains(&x.1) || helped.contains(&x.1) || fg.contains(&x.1)) {
            let ks = set_idx(u, 0);
            let lost: Vec<usize> = ks.iter().copied().filter(|&k| rb[k] && !rn[k]).collect();
            let won: Vec<usize> = ks.iter().copied().filter(|&k| !rb[k] && rn[k]).collect();
            let nsd = ks.iter().filter(|&&k| ats[k].sdiff).count();
            say!(out, "\n  {} (dev {}; {} pairs, {} lost, {} won; opener state differs at a shared place after the swap in {} pairs)",
                 uids[u], acc3(u).split(" | ").next().unwrap(), ks.len(), lost.len(), won.len(), nsd);
            say!(out, "    mean per pair                                swap   next 3    later      end     total");
            for q in 0..3 {
                let kq = set_idx(u, q);
                let lq: Vec<usize> = kq.iter().copied().filter(|&k| rb[k] && !rn[k]).collect();
                let wq: Vec<usize> = kq.iter().copied().filter(|&k| !rb[k] && rn[k]).collect();
                say!(out, "    {:<7} all pairs, margin change     {}", PARTS[q], fmt_r(mv(&kq, &|a, r| a.dg[r] - a.db[r])));
                say!(out, "    {:<7}   ln P change, good side     {}", PARTS[q], fmt_r(mv(&kq, &|a, r| a.dg[r])));
                say!(out, "    {:<7}   ln P change, bad side      {}", PARTS[q], fmt_r(mv(&kq, &|a, r| a.db[r])));
                say!(out, "    {:<7} lost pairs ({:>3}), margin    {}", PARTS[q], lq.len(), fmt_r(mv(&lq, &|a, r| a.dg[r] - a.db[r])));
                say!(out, "    {:<7} won pairs ({:>3}), margin     {}", PARTS[q], wq.len(), fmt_r(mv(&wq, &|a, r| a.dg[r] - a.db[r])));
            }
            // two examples, in dev order: lost pairs where the reading hurt, won pairs where it helped
            let (what, ex) = if hurt.contains(&u) || (!helped.contains(&u) && lost.len() >= won.len()) { ("lost", &lost) } else { ("won", &won) };
            for &k in ex.iter().take(2) {
                say!(out, "    {what} pair {} (half {}): margin {:+.3} -> {:+.3}; each word's ln P change, regions split by |",
                     dev[k].id, if half_b(dev[k]) { "B" } else { "A" }, mb[k], mn[k]);
                say!(out, "      good: {}", word_line(k, 0));
                say!(out, "      bad:  {}", word_line(k, 1));
            }
        }
        // B1: where the losses sit. The rule was fixed in the lane log before the run. The split by her opener
        // state was added in the fix, after review: next 3 also holds places where the two states differ, so
        // "early" there is not only the swap's own window.
        say!(out, "\nWHERE THE LOSSES SIT. Rule fixed before the run: closers could help only if MORE than half of the summed margin");
        say!(out, "drop of the lost pairs of the hurt filler-gap focus sets sits at later + end.");
        say!(out, "Split by her opener state (added in the fix, after review): the swap at places with no state (none) and with a");
        say!(out, "held one (held); next 3 where the two sentences' states match (same) and where they differ (differ). Later places and");
        say!(out, "the end add 0 where the states match, so all of later + end sits where they differ. Shares of the total:");
        say!(out, "early = swap + next 3, late = later + end (the rule's split), st.differ = next 3 differ + later + end.");
        let sum_p = |ks: &[usize]| -> [f64; 6] { let mut v = [0.0; 6]; for &k in ks { for r in 0..6 { v[r] += ats[k].parts[r]; } } v };
        let shares = |v: &[f64; 6]| -> (f64, f64, f64, f64) {
            let t: f64 = v.iter().sum();
            let sh = |x: f64| if t != 0.0 { 100.0 * x / t } else { 0.0 };
            (t, sh(v[0] + v[1] + v[2] + v[3]), sh(v[4] + v[5]), sh(v[3] + v[4] + v[5]))
        };
        let hurt_fg: Vec<usize> = hurt.iter().copied().filter(|u| fg.contains(u)).collect();
        let help_fg: Vec<usize> = helped.iter().copied().filter(|u| fg.contains(u)).collect();
        let pick = |sets: &[usize], q: usize, lostp: bool| -> Vec<usize> {
            (0..dev.len()).filter(|&k| sets.contains(&dev[k].uid) && in_part(q, dev[k]) && if lostp { rb[k] && !rn[k] } else { !rb[k] && rn[k] }).collect()
        };
        say!(out, "  summed over pairs (nats)               pairs  swap:none      held  next3:same    differ     later       end     total   early   late  st.differ");
        let mut verdict = None;
        for (label, sets, lostp) in [("lost, hurt filler-gap sets", &hurt_fg, true), ("lost, all 6 hurt sets", &hurt, true),
                                     ("won, helped filler-gap sets", &help_fg, false), ("won, all 6 helped sets", &helped, false)] {
            for q in 0..3 {
                let ks = pick(sets, q, lostp);
                let v = sum_p(&ks);
                let (t, early, late, differ) = shares(&v);
                say!(out, "  {:<30} {:<7} {:>5} {:+9.2} {:+9.2} {:+11.2} {:+9.2} {:+9.2} {:+9.2} {:+9.2}  {:>5.1}% {:>5.1}%     {:>5.1}%",
                     label, PARTS[q], ks.len(), v[0], v[1], v[2], v[3], v[4], v[5], t, early, late, differ);
                if q == 0 && label.starts_with("lost, hurt filler-gap") { verdict = Some((late, differ)); }
            }
        }
        if let Some((late, differ)) = verdict {
            if late > 50.0 {
                say!(out, "  -> by the rule (dev): MORE than half of the drop sits at later + end ({late:.1}%): closers could help");
            } else if differ > 50.0 {
                say!(out, "  -> by the rule (dev): later + end hold {late:.1}% of the drop, not more than half, so by its letter: no closers.");
                say!(out, "     But {differ:.1}% sits where the two sentences' opener states differ (next 3 included): a held expectation acts on");
                say!(out, "     between {late:.1}% and {differ:.1}% of the drop. The 50% bar falls inside that range, so the position split does not");
                say!(out, "     decide where the loss comes from. See the opener contrast and the state contrast below.");
            } else {
                say!(out, "  -> by the rule (dev): later + end hold {late:.1}%, and the places where the states differ {differ:.1}%: the loss is at the");
                say!(out, "     swapped words and their window, where no held expectation differs; no closers");
            }
        }
        // descriptive, added after the first B1 run (it does not change the rule above): set by set, and
        // split by whether her opener state differs between the two sentences at a shared place after the swap
        say!(out, "  set by set (descriptive; the late split was added after the first B1 run):    pairs     total   early   late  st.differ | late: last 2 words + end, before them");
        let late_line = |label: &str, ks: &[usize]| -> String {
            let v = sum_p(ks);
            let (t, early, late, differ) = shares(&v);
            let (l0, l1) = (ks.iter().map(|&k| ats[k].late[0]).sum::<f64>(), ks.iter().map(|&k| ats[k].late[1]).sum::<f64>());
            format!("  {:<78} {:>4} {:+9.2}  {:>5.1}% {:>5.1}%     {:>5.1}% | {:+9.2} {:+9.2}", label, ks.len(), t, early, late, differ, l0, l1)
        };
        for q in 0..3 {
            for &u in hurt.iter().chain(help_fg.iter()) {
                let lostp = hurt.contains(&u);
                let ks = pick(&[u], q, lostp);
                say!(out, "{}", late_line(&format!("{:<7} {} {}", PARTS[q], if lostp { "lost," } else { "won," }, uids[u]), &ks));
            }
            for (label, sets, lostp) in [("lost, hurt filler-gap sets", &hurt_fg, true), ("won, helped filler-gap sets", &help_fg, false)] {
                let ks = pick(sets, q, lostp);
                for (what, sd) in [("states differ after the swap", true), ("same states after the swap (late 0 by definition)", false)] {
                    let kk: Vec<usize> = ks.iter().copied().filter(|&k| ats[k].sdiff == sd).collect();
                    say!(out, "{}", late_line(&format!("{:<7} {label}: {what}", PARTS[q]), &kk));
                }
            }
        }
        // the opener contrast (added in the fix, after review): the filler-gap focus sets' pairs, grouped by
        // her openers among their swapped words. Counts on every pair, so no selection by lost or won.
        say!(out, "\nTHE OPENER CONTRAST IN THE SWAP (filler-gap focus sets; added in the fix, after review). Every pair, grouped by her");
        say!(out, "openers among its swapped words (good / bad; - none). Counts: dev (half A / half B). Margin change summed over the");
        say!(out, "group's pairs: dev (half A / half B), then dev split as in the rule table (swap none, held, next 3 same, differ, later, end).");
        for &u in hurt_fg.iter().chain(help_fg.iter()) {
            let ks = set_idx(u, 0);
            let mut labels: Vec<(usize, String)> = Vec::new();
            for &k in &ks {
                match labels.iter_mut().find(|x| x.1 == ats[k].sw) { Some(x) => x.0 += 1, None => labels.push((1, ats[k].sw.clone())) }
            }
            labels.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            say!(out, "  {} ({})", uids[u], if hurt_fg.contains(&u) { "hurt" } else { "helped" });
            for (_, l) in &labels {
                let grp = |q: usize| -> Vec<usize> { set_idx(u, q).into_iter().filter(|&k| ats[k].sw == *l).collect() };
                let c3 = |f: &dyn Fn(usize) -> bool| {
                    let c: Vec<usize> = (0..3).map(|q| grp(q).into_iter().filter(|&k| f(k)).count()).collect();
                    format!("{:>3} ({:>3}/{:>3})", c[0], c[1], c[2])
                };
                let tot = |q: usize| grp(q).iter().map(|&k| mn[k] - mb[k]).sum::<f64>();
                let v = sum_p(&grp(0));
                say!(out, "    {:<22} pairs {}  lost {}  won {}  margin change {:+8.2} ({:+8.2} / {:+8.2}); dev {:+8.2} {:+8.2} {:+8.2} {:+8.2} {:+8.2} {:+8.2}",
                     l, c3(&|_| true), c3(&|k| rb[k] && !rn[k]), c3(&|k| !rb[k] && rn[k]), tot(0), tot(1), tot(2), v[0], v[1], v[2], v[3], v[4], v[5]);
            }
        }
        // all the pairs of the filler-gap sets, with no selection by lost or won (added in the fix, after review)
        say!(out, "\nALL PAIRS OF THE 7 FILLER-GAP SETS (no selection by lost or won; added in the fix, after review). Summed margin change, nats:");
        say!(out, "  set                                      part         swap    next 3  later before the last 2 | last 2 words + end     total");
        for &u in &fg {
            for q in 0..3 {
                let ks = set_idx(u, q);
                let s = |f: &dyn Fn(&At) -> f64| ks.iter().map(|&k| f(&ats[k])).sum::<f64>();
                say!(out, "  {:<40} {:<7} {:+9.2} {:+9.2} {:+22.2} | {:+18.2} {:+9.2}", uids[u], PARTS[q], s(&|a| dm(a, 1)), s(&|a| dm(a, 2)),
                     s(&|a| a.late[1]), s(&|a| a.late[0]), s(&|a| (0..5).map(|r| dm(a, r)).sum::<f64>()));
            }
        }
        // her opener state and the change in ln P, place by place (added in the fix, after review): every place
        // 2 or more words into a dev sentence, where an opener can be held, by her state there
        if !trig.is_empty() {
            say!(out, "\nTHE STATE CONTRAST (added in the fix, after review): mean ln P change ({nn} - {bn}) per place, over every place 2 or");
            say!(out, "more words into a good or bad dev sentence (where an opener can be held), by her opener state there:");
            let mut acc = vec![[(0.0f64, 0u64); 3]; 256];
            for k in 0..dev.len() {
                for side in 0..2 {
                    for i in 2..ln.w[2 * k + side].len() {
                        let (s, x) = (state(2 * k + side, i), ln.w[2 * k + side][i] - lb.w[2 * k + side][i]);
                        for q in 0..3 { if in_part(q, dev[k]) { acc[s][q].0 += x; acc[s][q].1 += 1; } }
                    }
                }
            }
            say!(out, "  state          dev: places      mean   half A: places      mean   half B: places      mean");
            for s in (0..256).filter(|&s| acc[s][0].1 > 0) {
                let m = |q: usize| if acc[s][q].1 > 0 { acc[s][q].0 / acc[s][q].1 as f64 } else { 0.0 };
                say!(out, "  {:<14} {:>12} {:+9.4} {:>15} {:+9.4} {:>15} {:+9.4}", if s == 0 { "none".to_string() } else { sname(s) },
                     th(acc[s][0].1), m(0), th(acc[s][1].1), m(1), th(acc[s][2].1), m(2));
            }
        }
        // every word of every pair, for a closer look
        if let Some(path) = tsv {
            let f = std::fs::File::create(path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
            let mut fw = BufWriter::new(f);
            let mut put = |s: String| fw.write_all(s.as_bytes()).unwrap_or_else(|e| die(&format!("write {path}: {e}")));
            put(format!("set\tpairID\thalf\tside\tpos\tword\tregion\topener_state\tlnp_{bn}\tlnp_{nn}\n"));
            for k in 0..dev.len() {
                for side in 0..2 {
                    let w = if side == 0 { &wds[k].0 } else { &wds[k].1 };
                    for i in 0..=w.len() {
                        put(format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\n", uids[dev[k].uid], dev[k].id, if half_b(dev[k]) { "B" } else { "A" },
                                    if side == 0 { "good" } else { "bad" }, i, if i < w.len() { w[i].as_str() } else { "</s>" },
                                    REGIONS[region(i, w.len(), ats[k].pre, ats[k].suf)], sname(state(2 * k + side, i)), lb.w[2 * k + side][i], ln.w[2 * k + side][i]));
                    }
                }
            }
            drop(put);
            fw.flush().unwrap_or_else(|e| die(&format!("write {path}: {e}")));
            say!(out, "  every word of every dev pair written to {path}");
        }
    }
}

// ---------------------------------------------------------------- v31 B2 tool fix: word-matched mirror nets
/// her openers among the swapped words of every dev pair, good / bad ("-" none): the `sw` label of the attribution
fn opener_labels(dev: &[&Pair], trig: &[u64]) -> Vec<(String, String)> {
    dev.iter().map(|p| {
        let (g, b) = (words(&p.good, FROZEN_LOWER), words(&p.bad, FROZEN_LOWER));
        let (pre, suf) = shared(&g, &b);
        let ops = |w: &[String]| -> String {
            let v: Vec<&str> = w[pre..w.len() - suf].iter().filter(|x| trig.contains(&wkey(x))).map(|x| x.as_str()).collect();
            if v.is_empty() { "-".into() } else { v.join("+") }
        };
        (ops(&g), ops(&b))
    }).collect()
}

/// one cell of a mirror group: a label found on both sides, and its pairs on side 1 and side 2
struct Cell { label: String, s: [Vec<usize>; 2] }

/// v31 B2 tool fix (the lead's ruling on B0/B1): word-matched mirror nets on one part of dev. Each pair is labelled by
/// her openers among its swapped words, read from the side of the contrasted word: "openers in the sentence that holds
/// it / openers in the other sentence" (side 1: good / bad; side 2: bad / good). A cell is one label found on both sides
/// of a group in this part. In a cell, a1 and a2 are the share right of its side-1 and side-2 pairs (pairs pooled over a
/// side's sets); net = (a1 + a2 - 100) / 2 and lean = (a1 - a2) / 2. A group's matched net (and lean) is the mean of its
/// cells' nets (leans), each cell weighted by its pairs on both sides; a pair whose label is on one side only is left out
/// and counted. A pure lean toward an opener moves a1 and a2 of its cell in opposite directions, so the matched net is
/// about neutral to it, however unequal the counts. CI: paired bootstrap, pairs resampled within each (side, cell),
/// 20,000 resamples, splitmix64 seed 1. With `total`, the mean of the groups' matched nets too. Returns the lines.
fn mirror_matched(groups: &[(&str, &[&str], &[&str])], total: bool, uids: &[String], dev: &[&Pair], lab: &[(String, String)],
                  rb: &[bool], rn: &[bool], part: usize) -> Vec<String> {
    let uid = |name: &str| uids.iter().position(|u| u == name).unwrap_or_else(|| die(&format!("no BLiMP set {name}")));
    // per group: its cells, and how many pairs were left out
    let mut gc: Vec<(Vec<Cell>, usize)> = Vec::new();
    for g in groups {
        let side = |list: &[&str], one: bool| -> Vec<(String, usize)> {
            let us: Vec<usize> = list.iter().map(|s| uid(s)).collect();
            (0..dev.len()).filter(|&k| us.contains(&dev[k].uid) && in_part(part, dev[k]))
                .map(|k| (if one { format!("{} / {}", lab[k].0, lab[k].1) } else { format!("{} / {}", lab[k].1, lab[k].0) }, k)).collect()
        };
        let (a, b) = (side(g.1, true), side(g.2, false));
        let mut labels: Vec<String> = a.iter().filter(|x| b.iter().any(|y| y.0 == x.0)).map(|x| x.0.clone()).collect();
        labels.sort();
        labels.dedup();
        let cells: Vec<Cell> = labels.into_iter().map(|l| {
            let s1: Vec<usize> = a.iter().filter(|x| x.0 == l).map(|x| x.1).collect();
            let s2: Vec<usize> = b.iter().filter(|x| x.0 == l).map(|x| x.1).collect();
            Cell { label: l, s: [s1, s2] }
        }).collect();
        let kept: usize = cells.iter().map(|c| c.s[0].len() + c.s[1].len()).sum();
        gc.push((cells, a.len() + b.len() - kept));
    }
    // a group's [net base, net new, lean base, lean new], from the share right (base, new) of each cell's two sides
    let measure = |cells: &[Cell], acc: &[[(f64, f64); 2]]| -> [f64; 4] {
        let (mut v, mut wt) = ([0.0f64; 4], 0.0f64);
        for (j, c) in cells.iter().enumerate() {
            let w = (c.s[0].len() + c.s[1].len()) as f64;
            let ((b1, n1), (b2, n2)) = (acc[j][0], acc[j][1]);
            v[0] += w * (b1 + b2 - 100.0) / 2.0;
            v[1] += w * (n1 + n2 - 100.0) / 2.0;
            v[2] += w * (b1 - b2) / 2.0;
            v[3] += w * (n1 - n2) / 2.0;
            wt += w;
        }
        if wt > 0.0 { for x in v.iter_mut() { *x /= wt; } }
        v
    };
    let share = |ks: &[usize]| (pct(ks.iter().filter(|&&k| rb[k]).count() as u64, ks.len() as u64), pct(ks.iter().filter(|&&k| rn[k]).count() as u64, ks.len() as u64));
    let point: Vec<[f64; 4]> = gc.iter().map(|(cells, _)| measure(cells, &cells.iter().map(|c| [share(&c.s[0]), share(&c.s[1])]).collect::<Vec<_>>())).collect();
    let ng = groups.len();
    let (mut bnet, mut blean): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (vec![Vec::with_capacity(20_000); ng + 1], vec![Vec::with_capacity(20_000); ng]);
    let mut x = 1u64;
    for _ in 0..20_000 {
        let mut tot = 0.0;
        for (g, (cells, _)) in gc.iter().enumerate() {
            let mut acc: Vec<[(f64, f64); 2]> = Vec::with_capacity(cells.len());
            for c in cells {
                let mut r2 = [(0.0, 0.0); 2];
                for sd in 0..2 {
                    let ks = &c.s[sd];
                    let (mut cb, mut cn) = (0u64, 0u64);
                    for _ in 0..ks.len() { let k = ks[(splitmix(&mut x) % ks.len() as u64) as usize]; cb += rb[k] as u64; cn += rn[k] as u64; }
                    r2[sd] = (pct(cb, ks.len() as u64), pct(cn, ks.len() as u64));
                }
                acc.push(r2);
            }
            let m = measure(cells, &acc);
            bnet[g].push(m[1] - m[0]);
            blean[g].push(m[3] - m[2]);
            tot += m[1] - m[0];
        }
        bnet[ng].push(tot / ng as f64);
    }
    let ci = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); (v[499], v[19_499]) };
    let mut lines = Vec::new();
    for (g, grp) in groups.iter().enumerate() {
        let ((nl, nh), (ll, lh)) = (ci(&mut bnet[g]), ci(&mut blean[g]));
        let p = point[g];
        let cl: Vec<String> = gc[g].0.iter().map(|c| {
            let (s1, s2) = (share(&c.s[0]), share(&c.s[1]));
            format!("[{}] {}+{} pairs, net {:+.2} -> {:+.2}", c.label, c.s[0].len(), c.s[1].len(), (s1.0 + s2.0 - 100.0) / 2.0, (s1.1 + s2.1 - 100.0) / 2.0)
        }).collect();
        lines.push(format!("    {:<22} matched net {:+6.2} -> {:+6.2}  change {:+6.2} (95% CI {:+6.2} .. {:+6.2})   lean {:+6.2} -> {:+6.2}  change {:+6.2} ({:+6.2} .. {:+6.2})",
                           grp.0, p[0], p[1], p[1] - p[0], nl, nh, p[2], p[3], p[3] - p[2], ll, lh));
        lines.push(format!("    {:<22}   cells: {}; left out (label on one side only): {} pairs", "", cl.join("; "), gc[g].1));
    }
    if total {
        let (tl, thi) = ci(&mut bnet[ng]);
        let (tb, tn) = (point.iter().map(|v| v[0]).sum::<f64>() / ng as f64, point.iter().map(|v| v[1]).sum::<f64>() / ng as f64);
        lines.push(format!("    {:<22} matched net {:+6.2} -> {:+6.2}  change {:+6.2} (95% CI {:+6.2} .. {:+6.2})", format!("TOTAL (mean of {ng})"), tb, tn, tn - tb, tl, thi));
    }
    lines
}

/// the word-matched mirror section: dev, half A and half B; the 5 exact groups with their total, then W apart
fn matched_report(labs: &[(String, String)], uids: &[String], dev: &[&Pair], rb: &[bool], rn: &[bool], bn: &str, nn: &str) -> Vec<String> {
    let mut v = vec![
        format!("\nWORD-MATCHED MIRROR NETS ({bn} -> {nn}; v31 B2 tool fix, the mirror numbers judged from now on). Each pair is"),
        "labelled by her openers among its swapped words, read from the contrasted word's side: [openers in the sentence that holds".to_string(),
        "it / openers in the other] (side 1: good / bad; side 2: bad / good). A cell is a label found on both sides of a group. Net".to_string(),
        "and lean are taken within each cell (pairs pooled over a side's sets) and averaged over the cells, weighted by their pairs.".to_string(),
        "CI: pairs resampled within each (side, cell), 20,000 resamples, splitmix64 seed 1.".to_string(),
    ];
    for q in 0..3 {
        v.push(format!("  {}:", PARTS[q]));
        v.extend(mirror_matched(&MIRRORS, true, uids, dev, labs, rb, rn, q));
        v.extend(mirror_matched(&MIRROR_WIDE, false, uids, dev, labs, rb, rn, q).into_iter().map(|l| format!("{l}  (apart)")));
    }
    v
}

// ---------------------------------------------------------------- v31 B2: word kinds (AVA_CLASS) and word endings (AVA_ENDINGS)
/// AVA_CLASS=1: her word kinds as context. Off by default.
fn class_on() -> bool { std::env::var("AVA_CLASS").map_or(false, |v| v == "1") }
/// AVA_ENDINGS=1: her word endings as context. Off by default.
fn endings_on() -> bool { std::env::var("AVA_ENDINGS").map_or(false, |v| v == "1") }

/// K, her learned word kinds. Fixed before any run (shortlist row 2): enough kinds to keep number and role apart, few
/// enough that a pattern of kinds is read often
const NKIND: usize = 128;
/// a word read at least this often in her first 9M words is clustered; read fewer times, its neighbours are too few to
/// place it, and it shares the one rare kind
const KIND_MIN: u64 = 3;
/// a kind pattern holds her previous 1 to 5 kinds (shortlist row 2)
const KIND_PAT: usize = 5;
/// the exchange stops after this many passes, or sooner, once a pass moves fewer than 1 in 1,000 of her clustered words
/// (later passes then change little)
const KIND_PASSES: usize = 30;
/// endings of 1 to 5 letters, as morph_read reads a word's endings (its ENDS)
const END_LEN: usize = 5;
/// a stem keeps at least 3 letters: shorter remainders are mostly whole little words (is, as, us, it)
const END_STEM: usize = 3;
/// an ending is hers once at least 100 different stems take it (stem and stem + ending both read): a few accidents of
/// spelling do not make an ending
const END_MIN: usize = 100;
/// kind 0 is the sentence start in both blocks; kind 1 is CLASS's one rare kind and ENDINGS' "no learned ending"
const K_BOS: u8 = 0;
const K_REST: u8 = 1;
/// the bits that mark a kind key (word keys of up to 3 words use bits 0..48; opener states bits 48..56 and 63)
const MARK_CLASS: u64 = 1 << 62;
const MARK_END: u64 = 1 << 61;
/// the agreement sets B2 reports one by one (its brief; a list used only to measure)
const AGREE: [&str; 4] = ["determiner_noun_agreement", "distractor_agreement", "regular_plural_subject_verb_agreement", "irregular_plural_subject_verb_agreement"];

/// one block of kinds: each word's kind (by its key; a word not listed has kind `rest`), the mark of the block's keys,
/// and each kind's name for the reports (short, and long: CLASS its most-read words, ENDINGS how many stems take it)
struct KindMap { of: MixMap<u64, u8>, rest: u8, mark: u64, short: Vec<String>, long: Vec<String> }

impl KindMap {
    fn kind(&self, k: u64) -> u8 { self.of.get(&k).copied().unwrap_or(self.rest) }
}

/// the keys of the kind patterns before position i, shortest first: the previous 1..=5 kinds (the start mark, kind 0,
/// before word 0), kind + 1 in 8-bit slots, the nearest lowest, under the block's mark
fn kind_keys(kinds: &[u8], i: usize, mark: u64, out: &mut Vec<u64>) {
    out.clear();
    let mut key = mark;
    for n in 1..=KIND_PAT.min(i + 1) {
        let k = if n <= i { kinds[i - n] } else { K_BOS };
        key |= (k as u64 + 1) << (8 * (n - 1));
        out.push(key);
    }
}

/// her reading to the 10M-word mark (every sentence before the first that would take her past 10M words) and the end of
/// her first 9M words, split as learn_openers splits it (the sentence that takes her past 9M words is held out).
/// Returns (sentences to 10M words, the first held-out sentence).
fn fit_split(r: &Reading) -> (usize, usize) {
    let (mut n, mut n10, mut split) = (0u64, 0usize, None);
    for s in &r.sents {
        let w = words(s, false).len() as u64;
        if n + w > MAIN { break; }
        n += w;
        if split.is_none() && n > MAIN * 9 / 10 { split = Some(n10); }
        n10 += 1;
    }
    (n10, split.unwrap_or(n10))
}

/// her word kinds (AVA_CLASS): exchange clustering (Kneser & Ney 1993; Martin, Liermann & Ney 1998) of the words she read
/// KIND_MIN or more times in her first 9M words into NKIND kinds. The kind model reads each word through its kind,
/// P(w | v) = P(kind of w | kind of v) P(w | kind of w); its ln-likelihood of her first 9M words is
/// sum f(N(g, h)) - sum f(N_left(g)) - sum f(N_right(h)) + sum_w f(N(w)), with f(x) = x ln x. Each word in turn moves to
/// the kind that most raises it (it stays on a tie), pass after pass. The sentence edge (its start and its end) and the
/// rare words are two fixed kinds that never move. Start: her words, most read first, are dealt round the kinds in turn.
fn learn_class(r: &Reading, split: usize, out: &mut String) -> KindMap {
    let t0 = Instant::now();
    let fit = &r.sents[..split];
    let mut freq: MixMap<u64, u64> = MixMap::default();
    for s in fit { for w in words(s, false) { *freq.entry(wkey(&w)).or_insert(0) += 1; } }
    let ntypes = freq.len();
    let mut order: Vec<(u64, u64)> = freq.iter().filter(|x| *x.1 >= KIND_MIN).map(|(&k, &c)| (c, k)).collect();
    drop(freq);
    order.sort_unstable_by(|a, b| b.cmp(a));
    let nw = order.len();
    let nt = nw + 2;
    // index 0 the sentence edge, 1 a rare word, 2.. her clustered words, most read first
    let idx: MixMap<u64, u32> = order.iter().enumerate().map(|(i, x)| (x.1, i as u32 + 2)).collect();
    let mut name: Vec<String> = vec![String::new(); nt];
    let mut big: MixMap<u64, u32> = MixMap::default();
    let mut ntok = 0u64;
    for s in fit {
        let mut prev = 0u32;
        for w in words(s, false) {
            let cur = idx.get(&wkey(&w)).copied().unwrap_or(1);
            if cur > 1 && name[cur as usize].is_empty() { name[cur as usize] = w; }
            *big.entry((prev as u64) << 32 | cur as u64).or_insert(0) += 1;
            prev = cur;
            ntok += 1;
        }
        *big.entry((prev as u64) << 32).or_insert(0) += 1;
    }
    drop(idx);
    let mut bl: Vec<(u64, u32)> = big.into_iter().collect();
    bl.sort_unstable();
    let nbig = bl.len();
    let nbigram: u64 = bl.iter().map(|x| x.1 as u64).sum();
    ram_guard("kinds");
    // each word's neighbours both ways, as compressed rows: successors (sv, rows from so) and predecessors (pv, from po)
    let (mut so, mut po) = (vec![0usize; nt + 1], vec![0usize; nt + 1]);
    for &(k, _) in &bl { so[(k >> 32) as usize + 1] += 1; po[(k & 0xffff_ffff) as usize + 1] += 1; }
    for i in 0..nt { so[i + 1] += so[i]; po[i + 1] += po[i]; }
    let (mut sv, mut pv) = (vec![(0u32, 0u32); nbig], vec![(0u32, 0u32); nbig]);
    let (mut sf, mut pf) = (so.clone(), po.clone());
    for &(k, c) in &bl {
        let (a, b) = ((k >> 32) as usize, (k & 0xffff_ffff) as usize);
        sv[sf[a]] = (b as u32, c);
        sf[a] += 1;
        pv[pf[b]] = (a as u32, c);
        pf[b] += 1;
    }
    drop(bl);
    drop(sf);
    drop(pf);
    // a word's count as the left and as the right of a bigram (a pair with itself included)
    let lw: Vec<i64> = (0..nt).map(|w| sv[so[w]..so[w + 1]].iter().map(|x| x.1 as i64).sum()).collect();
    let rw: Vec<i64> = (0..nt).map(|w| pv[po[w]..po[w + 1]].iter().map(|x| x.1 as i64).sum()).collect();
    const NK2: usize = NKIND + 2;
    let mut cls: Vec<u8> = (0..nt).map(|w| if w < 2 { w as u8 } else { (2 + (w - 2) % NKIND) as u8 }).collect();
    let mut n2 = vec![0i64; NK2 * NK2];
    let (mut nl, mut nr) = (vec![0i64; NK2], vec![0i64; NK2]);
    for a in 0..nt {
        for &(b, c) in &sv[so[a]..so[a + 1]] { n2[cls[a] as usize * NK2 + cls[b as usize] as usize] += c as i64; }
        nl[cls[a] as usize] += lw[a];
        nr[cls[a] as usize] += rw[a];
    }
    let table: Vec<f64> = (0..1usize << 20).map(|x| if x == 0 { 0.0 } else { x as f64 * (x as f64).ln() }).collect();
    let f = |x: i64| -> f64 { if (x as usize) < table.len() { table[x as usize] } else { x as f64 * (x as f64).ln() } };
    let fw: f64 = rw.iter().map(|&x| f(x)).sum();
    let ll = |n2: &[i64], nl: &[i64], nr: &[i64]| -> f64 {
        (n2.iter().map(|&x| f(x)).sum::<f64>() - nl.iter().map(|&x| f(x)).sum::<f64>() - nr.iter().map(|&x| f(x)).sum::<f64>() + fw) / nbigram as f64
    };
    let t1 = Instant::now();
    say!(out, "her word kinds (AVA_CLASS): {} of her {} word types in her first 9M words ({} sentences, {} words, {} bigrams, {} different) are read {}+ times and clustered",
         th(nw as u64), th(ntypes as u64), th(split as u64), th(ntok), th(nbigram), th(nbig as u64), KIND_MIN);
    say!(out, "  start (dealt in turn): ln-likelihood {:.4} nats per bigram; counted in {:.1}s", ll(&n2, &nl, &nr), t0.elapsed().as_secs_f64());
    let (mut sc, mut pc) = (vec![0i64; NK2], vec![0i64; NK2]);
    let (mut st, mut pt): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    let mut passes = 0;
    for pass in 1..=KIND_PASSES {
        let tp = Instant::now();
        let mut moved = 0usize;
        for w in 2..nt {
            // the kinds of its neighbours (a pair with itself apart)
            let mut selfc = 0i64;
            st.clear();
            pt.clear();
            for &(v, c) in &sv[so[w]..so[w + 1]] {
                if v as usize == w { selfc = c as i64; continue; }
                let k = cls[v as usize] as usize;
                if sc[k] == 0 { st.push(k); }
                sc[k] += c as i64;
            }
            for &(u, c) in &pv[po[w]..po[w + 1]] {
                if u as usize == w { continue; }
                let k = cls[u as usize] as usize;
                if pc[k] == 0 { pt.push(k); }
                pc[k] += c as i64;
            }
            // take it out of its kind
            let a = cls[w] as usize;
            for &k in &st { n2[a * NK2 + k] -= sc[k]; }
            for &k in &pt { n2[k * NK2 + a] -= pc[k]; }
            n2[a * NK2 + a] -= selfc;
            nl[a] -= lw[w];
            nr[a] -= rw[w];
            // what putting it into kind b adds to the ln-likelihood
            let gain = |b: usize, n2: &[i64], nl: &[i64], nr: &[i64]| -> f64 {
                let mut d = 0.0;
                for &k in &st { if k != b { let x = n2[b * NK2 + k]; d += f(x + sc[k]) - f(x); } }
                for &k in &pt { if k != b { let x = n2[k * NK2 + b]; d += f(x + pc[k]) - f(x); } }
                let x = n2[b * NK2 + b];
                d += f(x + sc[b] + pc[b] + selfc) - f(x);
                d - (f(nl[b] + lw[w]) - f(nl[b])) - (f(nr[b] + rw[w]) - f(nr[b]))
            };
            let (mut best, mut bd) = (a, gain(a, &n2, &nl, &nr));
            for b in 2..NK2 {
                if b == a { continue; }
                let d = gain(b, &n2, &nl, &nr);
                if d > bd + 1e-9 { best = b; bd = d; }
            }
            // put it into the best kind
            for &k in &st { n2[best * NK2 + k] += sc[k]; }
            for &k in &pt { n2[k * NK2 + best] += pc[k]; }
            n2[best * NK2 + best] += selfc;
            nl[best] += lw[w];
            nr[best] += rw[w];
            if best != a { cls[w] = best as u8; moved += 1; }
            for &k in &st { sc[k] = 0; }
            for &k in &pt { pc[k] = 0; }
        }
        passes = pass;
        say!(out, "  pass {:>2}: {:>6} words moved; ln-likelihood {:.4} nats per bigram ({:.1}s)", pass, th(moved as u64), ll(&n2, &nl, &nr), tp.elapsed().as_secs_f64());
        if moved * 1000 < nw { break; }
        ram_guard("kinds");
    }
    // each kind's most-read words (the words are in order of how often she read them)
    let mut members: Vec<Vec<&str>> = vec![Vec::new(); NK2];
    let mut size = vec![0usize; NK2];
    for w in 2..nt { let k = cls[w] as usize; size[k] += 1; if members[k].len() < 8 { members[k].push(&name[w]); } }
    let short: Vec<String> = (0..NK2).map(|k| match k { 0 => "<s>".to_string(), 1 => "(rare)".to_string(), _ => format!("k{k}") }).collect();
    let long: Vec<String> = (0..NK2).map(|k| match k {
        0 => "<s>".to_string(),
        1 => "(rare: read under 3 times)".to_string(),
        _ => format!("k{k} ({} words: {})", size[k], members[k].join(" ")),
    }).collect();
    let used = (2..NK2).filter(|&k| size[k] > 0).count();
    let of: MixMap<u64, u8> = order.iter().enumerate().map(|(i, x)| (x.1, cls[i + 2])).collect();
    say!(out, "  {} kinds in use after {} passes; clustering took {:.1}s ({:.1}s with counting); peak RAM {:.2} GB",
         used, passes, t1.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64(), peak_gb());
    KindMap { of, rest: K_REST, mark: MARK_CLASS, short, long }
}

/// her word endings (AVA_ENDINGS), learned from her own word forms, as a child notices dog / dogs and walk / walked: an
/// ending of 1 to END_LEN letters is hers once at least END_MIN different stems of END_STEM or more letters take it, the
/// stem and the stem + ending both being words she read in her first 9M words. A word's kind is the ending it carries
/// over a stem she read (if several, the one most stems take; then the longer); a word with none has kind 1. No ending
/// is listed by hand. The kinds are given to every word of her reading to 10M words and to the `extra` words.
fn learn_endings(r: &Reading, split: usize, n10: usize, extra: &[String], out: &mut String) -> KindMap {
    let t0 = Instant::now();
    let mut seen: MixSet<u64> = MixSet::default();
    let mut typ: Vec<String> = Vec::new();
    for s in &r.sents[..split] { for w in words(s, false) { if seen.insert(wkey(&w)) { typ.push(w); } } }
    // (stem, ending) of a word, the ending n letters long, when the stem keeps END_STEM letters
    let cut = |w: &str, n: usize| -> Option<(String, String)> {
        let ch: Vec<char> = w.chars().collect();
        (ch.len() >= n + END_STEM).then(|| (ch[..ch.len() - n].iter().collect(), ch[ch.len() - n..].iter().collect()))
    };
    let mut support: HashMap<String, usize> = HashMap::new();
    for w in &typ {
        for n in 1..=END_LEN { if let Some((stem, e)) = cut(w, n) { if seen.contains(&wkey(&stem)) { *support.entry(e).or_insert(0) += 1; } } }
    }
    let ncand = support.len();
    let mut learned: Vec<(usize, String)> = support.into_iter().filter(|x| x.1 >= END_MIN).map(|(e, c)| (c, e)).collect();
    // most stems first; then the longer; then by spelling (so the order is fixed)
    learned.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.chars().count().cmp(&a.1.chars().count())).then(a.1.cmp(&b.1)));
    learned.truncate(253); // the kinds fit 8 bits (kind + 1 <= 255)
    let rank: HashMap<String, usize> = learned.iter().enumerate().map(|(i, x)| (x.1.clone(), i)).collect();
    let kind = |w: &str| -> u8 {
        let mut best: Option<usize> = None;
        for n in 1..=END_LEN {
            if let Some((stem, e)) = cut(w, n) {
                if let Some(&k) = rank.get(&e) { if seen.contains(&wkey(&stem)) && best.map_or(true, |b| k < b) { best = Some(k); } }
            }
        }
        best.map_or(K_REST, |k| (k + 2) as u8)
    };
    let mut of: MixMap<u64, u8> = MixMap::default();
    let mut done: MixSet<u64> = MixSet::default();
    let (mut tok, mut marked) = (0u64, 0u64);
    for s in &r.sents[..n10] {
        for w in words(s, false) {
            let k = wkey(&w);
            if done.insert(k) { let c = kind(&w); if c != K_REST { of.insert(k, c); } }
            tok += 1;
            if of.contains_key(&k) { marked += 1; }
        }
    }
    for w in extra { let k = wkey(w); if done.insert(k) { let c = kind(w); if c != K_REST { of.insert(k, c); } } }
    let mut short = vec!["<s>".to_string(), "(none)".to_string()];
    let mut long = vec!["<s>".to_string(), "(no learned ending)".to_string()];
    for (c, e) in &learned { short.push(format!("-{e}")); long.push(format!("-{e} ({c} stems)")); }
    say!(out, "her word endings (AVA_ENDINGS): from her {} word types in her first 9M words, {} endings of 1-{} letters met over a stem of {}+ letters;",
         th(typ.len() as u64), th(ncand as u64), END_LEN, END_STEM);
    say!(out, "  {} taken by {}+ stems are hers ({:.1}s); {:.1}% of the words of her reading to 10M carry one. Hers, most stems first:",
         learned.len(), END_MIN, t0.elapsed().as_secs_f64(), pct(marked, tok));
    for chunk in long[2..].chunks(8) { say!(out, "    {}", chunk.join(", ")); }
    KindMap { of, rest: K_REST, mark: MARK_END, short, long }
}

/// her kind blocks, learned once per run, each only when asked for
struct Kinds { class: Option<Rc<KindMap>>, end: Option<Rc<KindMap>> }

impl Kinds {
    /// learn the blocks asked for (none asked: nothing runs and nothing is printed). `extra`: words to give an ending
    /// kind beyond her reading (the scored words)
    fn learn(r: &Reading, class: bool, end: bool, extra: &[String], out: &mut String) -> Kinds {
        if !class && !end { return Kinds { class: None, end: None }; }
        let (n10, split) = fit_split(r);
        say!(out, "her kinds are learned on her first 9M words: sentences 0 .. {} (sentences {} .. {}, to 10M words, are held out of them)",
             th(split as u64), th(split as u64), th(n10 as u64));
        let end = if end { Some(Rc::new(learn_endings(r, split, n10, extra, out))) } else { None };
        let class = if class { Some(Rc::new(learn_class(r, split, out))) } else { None };
        Kinds { class, end }
    }
}

/// v31 B2: the pre-registered check, on her own reading only (no BLiMP pair is loaded). It was written in the lane log
/// before its first run. Do number-marking words land in different kinds?
///   (C1) this / these, is / are, was / were, has / have: each pair in two different CLASS kinds.
///   (C2) singular and plural forms, found by her own ending "-s": every word x of 3+ letters with x and x + "s" both read
///        20 or more times in her first 9M words (classes.rs's floor for a word's profile). At least 90% of these pairs
///        are in two different CLASS kinds.
///   (C3) the CLASS kinds carry number: purity = (sum over kinds of the larger of its x forms and its x + "s" forms) /
///        all forms, above 990 of 1,000 draws of its null (each pair's two forms swapped at random; splitmix64 seed 1).
///   (E1) "-s" is one of her learned endings; (E2) at least 90% of the pairs are in two different ENDINGS kinds.
/// PASS needs all five. Reported, not judged: the same for every clustered pair (both read 3+ times), and, on her held-out
/// reading (the words after 9M, to 10M), whether a reading fitted on her first 9M words prefers the form she actually read
/// over its partner, right after this / these / This / These and one word after them: plain, plain + each block.
fn kinds_check() {
    let mut out = String::new();
    say!(out, "source sha256 (compiled in): {}", sha256(SOURCE.as_bytes()));
    if !(class_on() && endings_on()) { die("kinds-check reads both blocks: set AVA_CLASS=1 and AVA_ENDINGS=1"); }
    say!(out, "v31 B2 pre-registered check on her own reading (no BLiMP pair is loaded)");
    let r = reading(&mut out);
    let (n10, split) = fit_split(&r);
    let kinds = Kinds::learn(&r, true, true, &[], &mut out);
    let (cm, em) = (kinds.class.clone().unwrap(), kinds.end.clone().unwrap());
    // how often she read each word in her first 9M words, with its spelling
    let mut freq: MixMap<u64, (u64, String)> = MixMap::default();
    for s in &r.sents[..split] { for w in words(s, false) { freq.entry(wkey(&w)).or_insert_with(|| (0, w.clone())).0 += 1; } }
    let ck = |w: &str| cm.kind(wkey(w)) as usize;
    let ek = |w: &str| em.kind(wkey(w)) as usize;
    let cnt = |w: &str| freq.get(&wkey(w)).map_or(0, |x| x.0);

    // (C1) the four function-word pairs
    say!(out, "\n(C1) NUMBER-MARKING FUNCTION WORDS (times read in her first 9M words; CLASS kind with its most-read words; ENDINGS kind)");
    let mut c1 = true;
    for (a, b) in [("this", "these"), ("is", "are"), ("was", "were"), ("has", "have")] {
        let differ = ck(a) != ck(b);
        c1 &= differ;
        for w in [a, b] { say!(out, "  {:<6} {:>9}  CLASS {}  | ENDINGS {}", w, th(cnt(w)), cm.long[ck(w)], em.short[ek(w)]); }
        say!(out, "    -> {} / {}: {}", a, b, if differ { "different CLASS kinds" } else { "SAME CLASS kind" });
    }
    say!(out, "  (C1) {}", if c1 { "pass: all 4 pairs are in different CLASS kinds" } else { "FAIL: a pair shares a CLASS kind" });

    // (C2), (C3), (E1), (E2): x and x + "s"
    let e1 = em.short.iter().any(|s| s == "-s");
    let mut all: Vec<(String, String, u64, u64)> = Vec::new();
    for (c, w) in freq.values() {
        if w.chars().count() < END_STEM { continue; }
        let ws = format!("{w}s");
        if let Some((cs, _)) = freq.get(&wkey(&ws)) { all.push((w.clone(), ws, *c, *cs)); }
    }
    all.sort_by(|a, b| b.2.min(b.3).cmp(&a.2.min(a.3)).then(a.0.cmp(&b.0)));
    let judged: Vec<&(String, String, u64, u64)> = all.iter().filter(|p| p.2 >= 20 && p.3 >= 20).collect();
    let clustered: Vec<&(String, String, u64, u64)> = all.iter().filter(|p| p.2 >= KIND_MIN && p.3 >= KIND_MIN).collect();
    let purity = |ps: &[&(String, String, u64, u64)], flip: &[bool]| -> f64 {
        let (mut nx, mut ns) = (vec![0usize; 256], vec![0usize; 256]);
        for (j, p) in ps.iter().enumerate() {
            let (a, b) = (ck(&p.0), ck(&p.1));
            if flip[j] { nx[b] += 1; ns[a] += 1; } else { nx[a] += 1; ns[b] += 1; }
        }
        (0..256).map(|k| nx[k].max(ns[k])).sum::<usize>() as f64 / (2 * ps.len()).max(1) as f64
    };
    // (share in different CLASS kinds, in different ENDINGS kinds, purity, the null's 990th of 1,000, its median)
    let judge = |ps: &[&(String, String, u64, u64)]| -> (f64, f64, f64, f64, f64) {
        let n = ps.len() as u64;
        let sc = pct(ps.iter().filter(|p| ck(&p.0) != ck(&p.1)).count() as u64, n);
        let se = pct(ps.iter().filter(|p| ek(&p.0) != ek(&p.1)).count() as u64, n);
        let obs = purity(ps, &vec![false; ps.len()]);
        let mut x = 1u64;
        let mut null: Vec<f64> = (0..1000).map(|_| { let fl: Vec<bool> = ps.iter().map(|_| splitmix(&mut x) & 1 == 1).collect(); purity(ps, &fl) }).collect();
        null.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (sc, se, obs, null[989], null[499])
    };
    say!(out, "\n(C2, C3, E1, E2) SINGULAR AND PLURAL FORMS BY HER ENDING \"-s\": word x (3+ letters) and x + \"s\", both read in her first 9M words");
    say!(out, "  (E1) \"-s\" {} one of her learned endings", if e1 { "is" } else { "is NOT" });
    let (jc, je, jp, jn, jm) = judge(&judged);
    let (ac, ae, ap, an, am) = judge(&clustered);
    say!(out, "  judged: both forms read 20+ times: {} pairs. In different CLASS kinds {:.1}%; in different ENDINGS kinds {:.1}%;", judged.len(), jc, je);
    say!(out, "          number purity of the CLASS kinds {:.3} (null: median {:.3}, 990th of 1,000 {:.3})", jp, jm, jn);
    say!(out, "  reported only: both forms read 3+ times (all clustered): {} pairs. Different CLASS kinds {:.1}%; ENDINGS {:.1}%; purity {:.3} (null median {:.3}, 990th {:.3})",
         clustered.len(), ac, ae, ap, am, an);
    let (c2, c3, e2) = (jc >= 90.0, jp > jn, je >= 90.0);
    say!(out, "  (C2) {}  (C3) {}  (E2) {}", if c2 { "pass" } else { "FAIL" }, if c3 { "pass" } else { "FAIL" }, if e2 { "pass" } else { "FAIL" });
    say!(out, "  the 40 judged pairs she read most (by the rarer form): x / x+s  (times read)  CLASS kinds  ENDINGS kinds");
    for p in judged.iter().take(40) {
        say!(out, "    {:<14} {:<15} ({:>6} / {:>6})  {:>6} / {:<6} {}  {} / {}", p.0, p.1, p.2, p.3, cm.short[ck(&p.0)], cm.short[ck(&p.1)],
             if ck(&p.0) == ck(&p.1) { "SAME" } else { "    " }, em.short[ek(&p.0)], em.short[ek(&p.1)]);
    }
    // the kinds that hold the judged forms: how many x and x + "s" forms each holds
    let (mut nx, mut ns) = (vec![0usize; 256], vec![0usize; 256]);
    for p in &judged { nx[ck(&p.0)] += 1; ns[ck(&p.1)] += 1; }
    let mut held: Vec<usize> = (0..256).filter(|&k| nx[k] + ns[k] > 0).collect();
    held.sort_by(|&a, &b| (nx[b] + ns[b]).cmp(&(nx[a] + ns[a])).then(a.cmp(&b)));
    say!(out, "  CLASS kinds holding the judged forms, most first (x forms / x+s forms): {}",
         held.iter().take(30).map(|&k| format!("{} {}/{}", cm.short[k], nx[k], ns[k])).collect::<Vec<_>>().join(", "));
    for &k in held.iter().take(12) { say!(out, "    {}", cm.long[k]); }
    let pass = c1 && c2 && c3 && e1 && e2;
    say!(out, "\nPRE-REGISTERED CHECK: {}  (C1 {}, C2 {}, C3 {}, E1 {}, E2 {})", if pass { "PASS" } else { "FAIL" },
         c1 as u8, c2 as u8, c3 as u8, e1 as u8, e2 as u8);

    // reported, not judged: her held-out reading. Right after this / these / This / These, and one word after them, where
    // she read a form of a judged pair, does a reading fitted on her first 9M words prefer that form over its partner?
    let det: Vec<u64> = ["this", "these", "This", "These"].iter().map(|w| wkey(w)).collect();
    let mut partner: MixMap<u64, (String, u32)> = MixMap::default();
    for p in &judged {
        for (a, b) in [(&p.0, &p.1), (&p.1, &p.0)] { partner.entry(wkey(a)).and_modify(|e| e.1 += 1).or_insert((b.clone(), 1)); }
    }
    let (mut pp, mut at): (Vec<Pair>, Vec<usize>) = (Vec::new(), Vec::new());
    for s in &r.sents[split..n10] {
        let w = words(s, false);
        for i in 1..w.len() {
            // a word in two judged pairs has no single partner and is left out
            let Some((q, 1)) = partner.get(&wkey(&w[i])).cloned() else { continue };
            for (gap, uid) in [(1usize, 0usize), (2, 1)] {
                if i >= gap && det.contains(&wkey(&w[i - gap])) {
                    let mut b = w[..=i].to_vec();
                    b[i] = q.clone();
                    pp.push(Pair { uid, id: pp.len() as u32, good: w[..=i].join(" "), bad: b.join(" ") });
                    at.push(i);
                }
            }
        }
    }
    let refs: Vec<&Pair> = pp.iter().collect();
    say!(out, "\nHELD-OUT READING (reported, not judged): places in her held-out reading (sentences {} .. {}) right after this / these /",
         th(split as u64), th(n10 as u64));
    say!(out, "This / These ({} places) and one word after them ({} places), where she read a form of a judged pair. Each reading is fitted",
         pp.iter().filter(|p| p.uid == 0).count(), pp.iter().filter(|p| p.uid == 1).count());
    say!(out, "on her first 9M words only; right = the form she read is more expected there than its partner. Plain = 3 words, cased.");
    let cfgs: [(&str, Vec<Rc<KindMap>>); 4] = [("plain", vec![]), ("plain+class", vec![cm.clone()]), ("plain+endings", vec![em.clone()]),
                                                ("plain+both", vec![em.clone(), cm.clone()])];
    let mut res: Vec<(Vec<bool>, Vec<f64>)> = Vec::new();
    for (nm, bl) in cfgs.iter() {
        let t0 = Instant::now();
        let mut her = Her::new_kinds(FROZEN_N, FROZEN_LOWER, &refs, Vec::new(), bl.clone());
        let (mut ids, mut keys, mut buf) = (Vec::new(), Vec::new(), Vec::new());
        for (k, s) in r.sents[..split].iter().enumerate() {
            her.read(&words(s, FROZEN_LOWER), &mut ids, &mut keys, &mut buf);
            if (k + 1) % 500_000 == 0 { ram_guard("held-out check"); }
        }
        let (mut right, mut lp) = (Vec::with_capacity(pp.len()), Vec::with_capacity(pp.len()));
        for k in 0..pp.len() {
            let (g, b) = (her.word_lnp(2 * k, &mut buf), her.word_lnp(2 * k + 1, &mut buf));
            right.push(g[at[k]] - b[at[k]] > TIE);
            lp.push(g[at[k]]);
        }
        say!(out, "  [{}] {} contexts to count; read {} sentences, {} words in {:.1}s; peak RAM {:.2} GB", nm, th(her.ctx.len() as u64),
             th(her.nsent), th(her.words), t0.elapsed().as_secs_f64(), peak_gb());
        res.push((right, lp));
    }
    for (uid, what) in [(0usize, "right after this / these / This / These"), (1, "one word after this / these / This / These")] {
        let ks: Vec<usize> = (0..pp.len()).filter(|&k| pp[k].uid == uid).collect();
        if ks.is_empty() { continue; }
        say!(out, "  {what} ({} places):", ks.len());
        for (j, (nm, _)) in cfgs.iter().enumerate() {
            let acc = pct(ks.iter().filter(|&&k| res[j].0[k]).count() as u64, ks.len() as u64);
            let mlp = ks.iter().map(|&k| res[j].1[k]).sum::<f64>() / ks.len() as f64;
            if j == 0 {
                say!(out, "    {:<14} right {:>6.2}%   mean ln P of the form she read {:+.4}", nm, acc, mlp);
            } else {
                let d: Vec<f64> = ks.iter().map(|&k| res[j].0[k] as u8 as f64 - res[0].0[k] as u8 as f64).collect();
                let (lo, hi) = boot_mean(&d);
                let dl = ks.iter().map(|&k| res[j].1[k] - res[0].1[k]).sum::<f64>() / ks.len() as f64;
                say!(out, "    {:<14} right {:>6.2}% ({:+.2} points vs plain, 95% CI {:+.2} .. {:+.2})   mean ln P {:+.4} ({:+.4} vs plain)",
                     nm, acc, 100.0 * d.iter().sum::<f64>() / d.len() as f64, lo, hi, mlp, dl);
            }
        }
    }
}

/// v31 B2 dev: plain, open (v30's frozen choice) and open + each kind block that is on (with both on: CLASS alone,
/// ENDINGS alone and both), at 10M words on the dev pairs. Every table gives dev, half A and half B: overall, the
/// agreement sets one by one, the mirror groups (pooled, and word-matched), and every set.
fn kinds_dev(pairs: &[Pair], uids: &[String]) {
    let mut out = String::new();
    say!(out, "source sha256 (compiled in): {}", sha256(SOURCE.as_bytes()));
    let (c_on, e_on) = (class_on(), endings_on());
    if !c_on && !e_on { die("kinds-dev reads open + her kinds: set AVA_CLASS=1 and/or AVA_ENDINGS=1"); }
    // the open reading is v30's frozen dev choice, whatever the environment says
    if std::env::var("AVA_OPEN_MODE").map_or(false, |v| v != FROZEN_OPEN_MODE) || std::env::var("AVA_OPEN_PICK").map_or(false, |v| v != FROZEN_OPEN_PICK) {
        die("kinds-dev reads open as v30's frozen choice: unset AVA_OPEN_MODE / AVA_OPEN_PICK (or set them to last / rate)");
    }
    std::env::set_var("AVA_OPEN_MODE", FROZEN_OPEN_MODE);
    std::env::set_var("AVA_OPEN_PICK", FROZEN_OPEN_PICK);
    let dev: Vec<&Pair> = pairs.iter().filter(|p| p.id % 5 == 0).collect();
    let mut rd = vec![Reader::Plain, Reader::Open];
    if c_on { rd.push(Reader::Class); }
    if e_on { rd.push(Reader::Endings); }
    if c_on && e_on { rd.push(Reader::Both); }
    let names: Vec<&str> = rd.iter().map(|x| x.name()).collect();
    say!(out, "v31 B2 kinds-dev: AVA_CLASS={} AVA_ENDINGS={}; her readings at 10M words: {}", c_on as u8, e_on as u8, names.join(", "));
    let nb = dev.iter().filter(|p| half_b(p)).count() as u64;
    say!(out, "DEV: {} pairs (pairID % 5 == 0); test pairs are not read. Half A ((pairID / 5) even): {}; half B (odd): {}",
         th(dev.len() as u64), th(dev.len() as u64 - nb), th(nb));
    let r = reading(&mut out);
    let trig = learn_openers(&r, &mut out);
    let extra: Vec<String> = if e_on { dev.iter().flat_map(|p| words(&p.good, FROZEN_LOWER).into_iter().chain(words(&p.bad, FROZEN_LOWER))).collect() } else { Vec::new() };
    let kinds = Kinds::learn(&r, c_on, e_on, &extra, &mut out);
    let ls: Vec<Lnp> = rd.iter().map(|&x| read_main(&r, &dev, x, &trig, &kinds, &mut out)).collect();
    drop(r);
    let total = |v: &[f64]| v.iter().fold(0.0, |a, &x| a + x);
    let right: Vec<Vec<bool>> = ls.iter().map(|l| (0..dev.len()).map(|k| total(&l.w[2 * k]) - total(&l.w[2 * k + 1]) > TIE).collect()).collect();
    let count = |j: usize, ks: &[usize]| ks.iter().filter(|&&k| right[j][k]).count() as u64;
    let acc = |j: usize, ks: &[usize]| pct(count(j, ks), ks.len() as u64);
    // reading b against reading a on these pairs: (difference in points, 95% CI low, high)
    let diff = |a: usize, b: usize, ks: &[usize]| -> (f64, f64, f64) {
        let d: Vec<f64> = ks.iter().map(|&k| right[b][k] as u8 as f64 - right[a][k] as u8 as f64).collect();
        let (lo, hi) = boot_mean(&d);
        (100.0 * d.iter().sum::<f64>() / d.len() as f64, lo, hi)
    };
    let part_idx: Vec<Vec<usize>> = (0..3).map(|q| (0..dev.len()).filter(|&k| in_part(q, dev[k])).collect()).collect();

    // (1) dev and its halves
    say!(out, "\nDEV AND ITS FIXED HALVES (at 10M words). Open against plain; each kinds reading against open. 95% CI: paired bootstrap, 20,000, seed 1.");
    say!(out, "  Mean ln P per word: over all dev sentences, good and bad, each word and the end.");
    for q in 0..3 {
        let ks = &part_idx[q];
        for j in 0..rd.len() {
            let nw: usize = ks.iter().map(|&k| ls[j].w[2 * k].len() + ls[j].w[2 * k + 1].len()).sum();
            let lp: f64 = ks.iter().map(|&k| total(&ls[j].w[2 * k]) + total(&ls[j].w[2 * k + 1])).sum::<f64>() / nw as f64;
            let vs = if j == 0 { String::new() } else {
                let base = if j == 1 { 0 } else { 1 };
                let (d, lo, hi) = diff(base, j, ks);
                format!("  {:+.2} points vs {} (95% CI {:+.2} .. {:+.2})", d, names[base], lo, hi)
            };
            say!(out, "  {:<7} {:<13} {:>6.2}% ({:>5} of {:>6}); mean ln P per word {:+.4}{}", PARTS[q], names[j], acc(j, ks), count(j, ks), ks.len(), lp, vs);
        }
    }

    // (2) the agreement sets, one by one
    let agree: Vec<usize> = (0..uids.len()).filter(|&u| AGREE.iter().any(|a| uids[u].starts_with(a))).collect();
    say!(out, "\nAGREEMENT SETS ({} sets, file order): accuracy % of {}; then each kinds reading minus open, points (95% CI)", agree.len(), names.join(" / "));
    let mut rows: Vec<(String, Vec<usize>)> = agree.iter().map(|&u| (uids[u].clone(), vec![u])).collect();
    rows.push(("ALL AGREEMENT SETS (pooled)".to_string(), agree.clone()));
    for (label, us) in &rows {
        for q in 0..3 {
            let ks: Vec<usize> = (0..dev.len()).filter(|&k| us.contains(&dev[k].uid) && in_part(q, dev[k])).collect();
            let a: Vec<String> = (0..rd.len()).map(|j| format!("{:>5.1}", acc(j, &ks))).collect();
            let d: Vec<String> = (2..rd.len()).map(|j| { let (x, lo, hi) = diff(1, j, &ks); format!("{}: {:+5.1} ({:+5.1} .. {:+5.1})", names[j], x, lo, hi) }).collect();
            say!(out, "  {:<46} {:<7} {} | {}", label, PARTS[q], a.join(" "), d.join(" | "));
        }
    }

    // (3) the mirror groups: pooled nets (B0's way) and the word-matched nets (judged from now on)
    let labs = opener_labels(&dev, &trig);
    let mut comps = vec![(0usize, 1usize)];
    for j in 2..rd.len() { comps.push((1, j)); }
    for &(a, b) in &comps {
        say!(out, "\nMIRROR GROUPS ({} -> {}): pooled nets, as B0 computes them (net and lean change, 95% CI)", names[a], names[b]);
        for q in 0..3 {
            say!(out, "  {}:", PARTS[q]);
            for l in mirror_part(&MIRRORS, true, uids, &dev, &right[a], &right[b], q) { say!(out, "{l}"); }
            for l in mirror_part(&MIRROR_WIDE, false, uids, &dev, &right[a], &right[b], q) { say!(out, "{l}  (apart)"); }
        }
        for l in matched_report(&labs, uids, &dev, &right[a], &right[b], names[a], names[b]) { say!(out, "{l}"); }
    }

    // (4) every set
    say!(out, "\nSET BY SET (accuracy % of {}): dev | half A | half B", names.join(" / "));
    for u in 0..uids.len() {
        let cols: Vec<String> = (0..3).map(|q| {
            let ks: Vec<usize> = (0..dev.len()).filter(|&k| dev[k].uid == u && in_part(q, dev[k])).collect();
            (0..rd.len()).map(|j| format!("{:>5.1}", acc(j, &ks))).collect::<Vec<_>>().join(" ")
        }).collect();
        say!(out, "  {:<52} {}", uids[u], cols.join(" | "));
    }
}
