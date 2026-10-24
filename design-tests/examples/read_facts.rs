// TESTING / DESIGN ONLY. (v29 WP4) Ava reads a science textbook, then answers science
// multiple-choice questions (AI2 ARC) by recognition.
//
//   cargo run --release --example read_facts -- dev             Train + Dev of ARC-Easy and ARC-Challenge
//   cargo run --release --example read_facts -- test <out.tsv>  Test of both, frozen settings (run ONCE)
//   cargo run --release --example read_facts -- dryrun <out.tsv>  the same frozen routine on Easy-Dev + Chal-Dev
//   cargo run --release --example read_facts -- leak <out.tsv> <dev|test>  verbatim stems in the corpus
//   AVA_THREADS (default 4, at most 4)
//   dev only: AVA_STEM=0 turns word endings off; AVA_CUE_MIN=<spec> moves the calling-word floor;
//             AVA_DEV_TSV=<file> writes each question's scores under the frozen setting
//
// What she does, the way a reader would:
//   reading   she streams the AI2 ARC Corpus (14.6M science sentences, 1.48 GB); one sentence is
//             one memory. A first pass counts how many sentences each word is in (df), which
//             gives each word's specificity ln(N/df): her readout weight. A word met everywhere
//             says little; a rare word points at a few memories.
//   recall    a question plus one choice is a cue. A remembered sentence that holds a calling
//             word of the choice is reactivated with strength = the summed specificity of every
//             word it shares with the cue (question words and choice words), damped when the
//             sentence is long (a long sentence is a diffuse trace; BM25's length damping):
//             hippocampal pattern completion.
//   answer    the choice whose cue brings back the strongest memories (sum of the k strongest
//             recall strengths) wins. Her confidence is the share by which it beats the
//             runner-up; the abstaining reading answers only above a margin chosen on Dev.
// Streaming is an implementation shortcut only: the strength of (cue, sentence) depends on the
// sentence and on corpus-wide counts alone, so scoring each sentence as it streams past gives the
// same answer as storing all 14.6M memories and recalling them afterwards.
// No answer rules: nothing looks at the answer keys except the few settings chosen on Train/Dev.
// The downloaded files are read as data only.
use std::collections::HashMap;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::time::Instant;

const DATA: &str = "../data/ai2-arc/ARC-V1-Feb2018-2";
const CORPUS: &str = "ARC_Corpus.txt";
const DEV_FILES: [&str; 4] = [
    "ARC-Easy/ARC-Easy-Train.jsonl", "ARC-Easy/ARC-Easy-Dev.jsonl",
    "ARC-Challenge/ARC-Challenge-Train.jsonl", "ARC-Challenge/ARC-Challenge-Dev.jsonl",
];
const DEV_NAMES: [&str; 4] = ["Easy-Train", "Easy-Dev", "Chal-Train", "Chal-Dev"];
const TEST_FILES: [&str; 2] = ["ARC-Easy/ARC-Easy-Test.jsonl", "ARC-Challenge/ARC-Challenge-Test.jsonl"];
const TEST_NAMES: [&str; 2] = ["Easy-Test", "Chal-Test"];

/// BM25's saturation constant for the length damping (its usual value)
const K1: f32 = 1.2;
/// the length dampings b tried on Train/Dev (0 = none, 1 = full)
const BS: [f32; 3] = [0.1, 0.25, 0.4];
/// how many of the strongest memories each cue keeps
const KEEP: usize = 10;
/// how many of them are summed into the choice's score (tried on Train/Dev)
const KS: [usize; 4] = [1, 3, 5, 10];

// ---------------------------------------------------------------- frozen settings (test mode)
// Chosen on ARC-Easy Train/Dev before any Test file was opened (dev runs: scratchpad v29/read_dev*.txt).
// Rule: best ARC-Easy-Train accuracy; a tie goes to the better pooled Easy Train+Dev accuracy.
const FROZEN_STEM: bool = true;
/// a word in more than 1 of every 100 sentences ("the", "of", "water") cannot call up a memory
/// by itself (it still adds its weight to a recall); a choice's most specific word always can
const FROZEN_CUE_MIN: f32 = 4.605_17; // ln(100)
const FROZEN_B: usize = 1; // BS[1] = 0.25
const FROZEN_K: usize = 10;
/// chosen on ARC-Easy-Dev only: the margin in 0.00..0.50 with the best right - wrong
const FROZEN_THETA: f64 = 0.01;

// ---------------------------------------------------------------- small helpers
#[derive(Default, Clone, Copy)]
struct Fx(u64);
impl Hasher for Fx {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, b: &[u8]) {
        const M: u64 = 0x517c_c1b7_2722_0a95;
        let mut h = self.0;
        let mut ch = b.chunks_exact(8);
        for c in &mut ch { h = (h.rotate_left(5) ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(M); }
        for &x in ch.remainder() { h = (h.rotate_left(5) ^ x as u64).wrapping_mul(M); }
        self.0 = h;
    }
}
type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

fn die(m: &str) -> ! { eprintln!("{m}"); std::process::exit(2) }

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
fn fnv(s: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in s { h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3); }
    h
}

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

// ---------------------------------------------------------------- a small JSON reader (with escapes)
enum J { Arr(Vec<J>), Obj(Vec<(String, J)>), Num, Str(String), Lit }

struct Rd<'a> { b: &'a [u8], i: usize }

impl Rd<'_> {
    fn ws(&mut self) { while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() { self.i += 1; } }
    fn at(&self) -> Result<u8, String> { self.b.get(self.i).copied().ok_or_else(|| "unexpected end".to_string()) }
    fn hex4(&mut self) -> Result<u32, String> {
        let s = self.b.get(self.i..self.i + 4).ok_or("short \\u")?;
        self.i += 4;
        u32::from_str_radix(std::str::from_utf8(s).map_err(|_| "bad \\u")?, 16).map_err(|_| "bad \\u".to_string())
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = self.at()?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.at()?;
                    self.i += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'b' => out.push(8), b'f' => out.push(12), b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'), b't' => out.push(b'\t'),
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) {
                                // a high surrogate: take the low one if it follows
                                if self.b.get(self.i..self.i + 2) == Some(b"\\u".as_slice()) {
                                    let save = self.i;
                                    self.i += 2;
                                    let lo = self.hex4()?;
                                    if (0xDC00..0xE000).contains(&lo) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); }
                                    else { self.i = save; cp = 0xFFFD; }
                                } else { cp = 0xFFFD; }
                            } else if (0xDC00..0xE000).contains(&cp) { cp = 0xFFFD; }
                            let ch = char::from_u32(cp).unwrap_or('\u{FFFD}');
                            let mut tmp = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                        }
                        _ => return Err(format!("bad escape \\{}", e as char)),
                    }
                }
                _ => out.push(c),
            }
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }
    fn val(&mut self) -> Result<J, String> {
        self.ws();
        match self.at()? {
            b'[' => {
                self.i += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.at()? == b']' { self.i += 1; break; }
                    v.push(self.val()?);
                    self.ws();
                    if self.at()? == b',' { self.i += 1; }
                }
                Ok(J::Arr(v))
            }
            b'{' => {
                self.i += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.at()? == b'}' { self.i += 1; break; }
                    if self.at()? != b'"' { return Err("object key".into()); }
                    let k = self.string()?;
                    self.ws();
                    if self.at()? != b':' { return Err("expected ':'".into()); }
                    self.i += 1;
                    let x = self.val()?;
                    v.push((k, x));
                    self.ws();
                    if self.at()? == b',' { self.i += 1; }
                }
                Ok(J::Obj(v))
            }
            b'"' => Ok(J::Str(self.string()?)),
            b't' | b'f' | b'n' => {
                while self.i < self.b.len() && self.b[self.i].is_ascii_alphabetic() { self.i += 1; }
                Ok(J::Lit)
            }
            _ => {
                let s = self.i;
                while self.i < self.b.len() && b"+-.0123456789eE".contains(&self.b[self.i]) { self.i += 1; }
                if self.i == s { return Err("unexpected byte".into()); }
                Ok(J::Num)
            }
        }
    }
}

fn get<'a>(o: &'a J, k: &str) -> Option<&'a J> {
    if let J::Obj(v) = o { v.iter().find(|(a, _)| a == k).map(|(_, b)| b) } else { None }
}
fn text(o: Option<&J>) -> String { if let Some(J::Str(s)) = o { s.clone() } else { String::new() } }

struct Q { id: String, stem: String, choices: Vec<(String, String)>, key: usize, set: usize }

fn load(file: &str, set: usize) -> Vec<Q> {
    let p = format!("{DATA}/{file}");
    let raw = std::fs::read(&p).unwrap_or_else(|e| die(&format!("cannot read {p}: {e}")));
    let mut out = Vec::new();
    for (n, line) in raw.split(|&c| c == b'\n').enumerate() {
        if line.iter().all(|c| c.is_ascii_whitespace()) { continue; }
        let j = Rd { b: line, i: 0 }.val().unwrap_or_else(|e| die(&format!("{p} line {}: {e}", n + 1)));
        let q = get(&j, "question");
        let choices: Vec<(String, String)> = match q.and_then(|q| get(q, "choices")) {
            Some(J::Arr(v)) => v.iter().map(|c| (text(get(c, "label")), text(get(c, "text")))).collect(),
            _ => Vec::new(),
        };
        let ak = text(get(&j, "answerKey"));
        let key = choices.iter().position(|(l, _)| *l == ak).unwrap_or(usize::MAX);
        out.push(Q { id: text(get(&j, "id")), stem: text(q.and_then(|q| get(q, "stem"))), choices, key, set });
    }
    out
}

// ---------------------------------------------------------------- her eyes: words
const MAXTOK: usize = 32;

/// word endings: plural -s/-es/-ies, -ing, -ed, a final -e (so "heats", "heated", "heating"
/// meet at "heat" and "move", "moved", "moving" at "mov"). Letters only; short words untouched.
fn stem(w: &mut [u8], mut n: usize) -> usize {
    if n < 4 || !w[..n].iter().all(|c| c.is_ascii_lowercase()) { return n; }
    let ends = |w: &[u8], n: usize, s: &[u8]| n >= s.len() && &w[n - s.len()..n] == s;
    let vowel = |c: u8| matches!(c, b'a' | b'e' | b'i' | b'o' | b'u' | b'y');
    if ends(w, n, b"sses") { n -= 2; }
    else if n > 4 && ends(w, n, b"ies") { n -= 3; w[n] = b'y'; n += 1; }
    else if n > 4 && ends(w, n, b"es") && (ends(w, n - 2, b"sh") || ends(w, n - 2, b"ch") || matches!(w[n - 3], b'x' | b'z' | b's')) { n -= 2; }
    else if w[n - 1] == b's' && !matches!(w[n - 2], b's' | b'u' | b'i') { n -= 1; }
    let mut cut = false;
    if n >= 6 && ends(w, n, b"ing") && w[..n - 3].iter().any(|&c| vowel(c)) { n -= 3; cut = true; }
    else if n >= 5 && ends(w, n, b"ed") && w[n - 3] != b'e' && w[..n - 2].iter().any(|&c| vowel(c)) { n -= 2; cut = true; }
    if cut && n >= 3 && w[n - 1] == w[n - 2] && !vowel(w[n - 1]) && !matches!(w[n - 1], b'l' | b's' | b'z') { n -= 1; }
    if n >= 4 && w[n - 1] == b'e' { n -= 1; }
    n
}

/// lowercase ASCII letter/digit runs; every other byte (spaces, punctuation, non-ASCII) is a gap.
/// Calls f on each word (runs longer than MAXTOK are counted but not read). Returns the word count.
fn words(line: &[u8], stem_on: bool, mut f: impl FnMut(&[u8])) -> u32 {
    let mut buf = [0u8; MAXTOK + 2];
    let (mut n, mut long, mut count) = (0usize, false, 0u32);
    for &c in line.iter().chain(std::iter::once(&b' ')) {
        let l = c.to_ascii_lowercase();
        if l.is_ascii_lowercase() || l.is_ascii_digit() {
            if n < MAXTOK { buf[n] = l; n += 1; } else { long = true; }
        } else if n > 0 {
            count += 1;
            if !long { let m = if stem_on { stem(&mut buf, n) } else { n }; f(&buf[..m]); }
            n = 0;
            long = false;
        }
    }
    count
}

// ---------------------------------------------------------------- the cues
struct Cues {
    stem_on: bool,
    vocab: FxMap<Box<[u8]>, u32>,
    names: Vec<String>,
    qw: Vec<Vec<u32>>,  // question -> its distinct words, sorted
    cw: Vec<Vec<u32>>,  // choice -> its distinct words that are not in its question, sorted
    cq: Vec<u32>,       // choice -> question
    qc: Vec<(usize, usize)>, // question -> its choices [a, b)
}

fn build_cues(qs: &[Q], stem_on: bool) -> Cues {
    let mut c = Cues { stem_on, vocab: FxMap::default(), names: Vec::new(), qw: Vec::new(), cw: Vec::new(), cq: Vec::new(), qc: Vec::new() };
    let ids = |s: &str, c: &mut Cues| -> Vec<u32> {
        let mut v = Vec::new();
        words(s.as_bytes(), stem_on, |t| {
            let next = c.vocab.len() as u32;
            let id = *c.vocab.entry(t.into()).or_insert_with(|| { c.names.push(String::from_utf8_lossy(t).into_owned()); next });
            v.push(id);
        });
        v.sort_unstable();
        v.dedup();
        v
    };
    for (qi, q) in qs.iter().enumerate() {
        let qw = ids(&q.stem, &mut c);
        let a = c.cw.len();
        for (_, t) in &q.choices {
            let w: Vec<u32> = ids(t, &mut c).into_iter().filter(|x| qw.binary_search(x).is_err()).collect();
            c.cw.push(w);
            c.cq.push(qi as u32);
        }
        c.qc.push((a, c.cw.len()));
        c.qw.push(qw);
    }
    c
}

// ---------------------------------------------------------------- streaming the corpus
/// split the corpus into n byte ranges and run f on each range's lines in its own thread
fn stream<S: Send>(n: usize, f: impl Fn(usize, &mut dyn FnMut(&mut dyn FnMut(&[u8]))) -> S + Sync) -> Vec<S> {
    let path = format!("{DATA}/{CORPUS}");
    let len = std::fs::metadata(&path).unwrap_or_else(|e| die(&format!("cannot open {path}: {e}"))).len();
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..n).map(|t| {
            let (path, f) = (&path, &f);
            sc.spawn(move || {
                let (start, end) = (len * t as u64 / n as u64, len * (t as u64 + 1) / n as u64);
                let mut each = |g: &mut dyn FnMut(&[u8])| {
                    let mut fh = File::open(path).expect("open corpus");
                    let mut pos = start.saturating_sub(1);
                    fh.seek(SeekFrom::Start(pos)).expect("seek");
                    let mut rd = BufReader::with_capacity(1 << 20, fh);
                    let mut buf = Vec::with_capacity(4096);
                    if start > 0 { pos += rd.read_until(b'\n', &mut buf).expect("read") as u64; }
                    let mut k = 0u64;
                    while pos < end {
                        buf.clear();
                        let got = rd.read_until(b'\n', &mut buf).expect("read");
                        if got == 0 { break; }
                        pos += got as u64;
                        while matches!(buf.last(), Some(b'\n' | b'\r')) { buf.pop(); }
                        g(&buf);
                        k += 1;
                        if k % 1_000_000 == 0 { ram_guard("corpus"); }
                    }
                };
                f(t, &mut each)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().expect("thread")).collect()
    })
}


struct Spec { n: f64, avg_len: f32, df: Vec<u32>, w1: Vec<f32> }

/// first reading: how many sentences each cue word is in
fn read_df(c: &Cues, threads: usize) -> Spec {
    let parts = stream(threads, |_, each| {
        let mut df = vec![0u32; c.vocab.len()];
        let (mut n, mut toks) = (0u64, 0u64);
        let mut ids: Vec<u32> = Vec::new();
        each(&mut |line: &[u8]| {
            ids.clear();
            let l = words(line, c.stem_on, |t| if let Some(&id) = c.vocab.get(t) { ids.push(id) });
            if l == 0 { return; }
            n += 1;
            toks += l as u64;
            ids.sort_unstable();
            ids.dedup();
            for &i in &ids { df[i as usize] += 1; }
        });
        (df, n, toks)
    });
    let mut df = vec![0u32; c.vocab.len()];
    let (mut n, mut toks) = (0u64, 0u64);
    for (d, a, b) in parts { for (x, y) in df.iter_mut().zip(d) { *x += y; } n += a; toks += b; }
    let nf = n as f64;
    let w1: Vec<f32> = df.iter().map(|&d| (nf / d.max(1) as f64).ln() as f32).collect();
    Spec { n: nf, avg_len: (toks as f64 / nf.max(1.0)) as f32, df, w1 }
}

#[derive(Clone, Copy)]
struct Hit { s: f32, h: u64 }
const NO: Hit = Hit { s: 0.0, h: 0 };

/// keep the KEEP strongest distinct memories (the same sentence text read twice is one memory)
fn keep(top: &mut [Hit], s: f32, h: u64) {
    if s <= top[KEEP - 1].s || top.iter().any(|x| x.s > 0.0 && x.h == h) { return; }
    let mut i = KEEP - 1;
    while i > 0 && top[i - 1].s < s { top[i] = top[i - 1]; i -= 1; }
    top[i] = Hit { s, h };
}

/// which words of a choice can call up a memory on their own: its most specific word always,
/// and any other word with specificity >= cue_min
fn cue_posts(c: &Cues, sp: &Spec, cue_min: f32) -> Vec<Vec<u32>> {
    let mut post: Vec<Vec<u32>> = vec![Vec::new(); c.vocab.len()];
    for (ch, w) in c.cw.iter().enumerate() {
        let top = w.iter().filter(|&&i| sp.df[i as usize] > 0).map(|&i| sp.w1[i as usize]).fold(f32::MIN, f32::max);
        for &i in w {
            let s = sp.w1[i as usize];
            if sp.df[i as usize] > 0 && (s >= cue_min || s == top) { post[i as usize].push(ch as u32); }
        }
    }
    post
}

/// second reading: every sentence reactivates, for each cue it shares a calling word with, its
/// recall strength (summed specificity of the shared words, damped for long sentences, one value
/// per damping b in BS); each cue keeps its KEEP strongest distinct memories
fn recall(c: &Cues, sp: &Spec, threads: usize, cue_min: f32) -> Vec<Hit> {
    let nv = c.vocab.len();
    let post = cue_posts(c, sp, cue_min);
    let work: f64 = (0..nv).map(|i| sp.df[i] as f64 * post[i].len() as f64).sum();
    let mut heavy: Vec<(f64, usize)> = (0..nv).map(|i| (sp.df[i] as f64 * post[i].len() as f64, i)).collect();
    heavy.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("cue-memory contacts: {:.2e}   heaviest calling words: {}", work,
        heavy.iter().take(12).map(|&(w, i)| format!("{} ({:.0}%, spec {:.2}, {} cues)", c.names[i], w / work * 100.0, sp.w1[i], post[i].len())).collect::<Vec<_>>().join(", "));
    let nc = c.cw.len();
    let nb = BS.len();
    let parts = stream(threads, |_, each| {
        let mut top = vec![NO; nc * nb * KEEP];
        let mut seen = vec![0u32; nc];
        let mut qseen = vec![0u32; c.qw.len()];
        let mut qval = vec![0f32; c.qw.len()];
        let mut sw = vec![0f32; nv];
        let mut stamp = 0u32;
        let mut ids: Vec<u32> = Vec::new();
        let mut touched: Vec<u32> = Vec::new();
        let mut fl = [0f32; BS.len()];
        each(&mut |line: &[u8]| {
            ids.clear();
            let l = words(line, c.stem_on, |t| if let Some(&id) = c.vocab.get(t) { ids.push(id) });
            if l == 0 || ids.is_empty() { return; }
            stamp += 1;
            touched.clear();
            for &i in &ids {
                if sw[i as usize] != 0.0 { continue; } // a repeated word counts once
                sw[i as usize] = sp.w1[i as usize].max(1e-6);
                for &ch in &post[i as usize] { if seen[ch as usize] != stamp { seen[ch as usize] = stamp; touched.push(ch); } }
            }
            if !touched.is_empty() {
                let h = fnv(line);
                let rel = l as f32 / sp.avg_len;
                for (k, &b) in BS.iter().enumerate() { fl[k] = (K1 + 1.0) / (1.0 + K1 * (1.0 - b + b * rel)); }
                for &ch in &touched {
                    let ch = ch as usize;
                    let q = c.cq[ch] as usize;
                    if qseen[q] != stamp { qseen[q] = stamp; qval[q] = c.qw[q].iter().map(|&i| sw[i as usize]).sum(); }
                    let raw = qval[q] + c.cw[ch].iter().map(|&i| sw[i as usize]).sum::<f32>();
                    for k in 0..nb {
                        let at = (ch * nb + k) * KEEP;
                        let s = raw * fl[k];
                        if s > top[at + KEEP - 1].s { keep(&mut top[at..at + KEEP], s, h); }
                    }
                }
            }
            for &i in &ids { sw[i as usize] = 0.0; }
        });
        top
    });
    let mut top = vec![NO; nc * nb * KEEP];
    for part in parts {
        for (slot, chunk) in top.chunks_mut(KEEP).zip(part.chunks(KEEP)) { for x in chunk { if x.s > 0.0 { keep(slot, x.s, x.h); } } }
    }
    top
}

// ---------------------------------------------------------------- answering
/// each choice's score: the sum of its k strongest memories under damping BS[b]
fn scores(c: &Cues, top: &[Hit], qi: usize, b: usize, k: usize) -> Vec<f32> {
    let (x, y) = c.qc[qi];
    (x..y).map(|ch| { let at = (ch * BS.len() + b) * KEEP; top[at..at + k].iter().map(|h| h.s).sum() }).collect()
}

/// her choice (ties broken by a fixed hash of the question id, not by position) and her margin:
/// the share by which the winner beats the runner-up (0 when nothing was recalled or on a tie)
fn pick(id: &str, s: &[f32]) -> (usize, f64) {
    if s.is_empty() { return (usize::MAX, 0.0); }
    let best = s.iter().cloned().fold(f32::MIN, f32::max);
    let tied: Vec<usize> = (0..s.len()).filter(|&i| s[i] == best).collect();
    let mut seed = fnv(id.as_bytes());
    let choice = tied[(splitmix(&mut seed) % tied.len() as u64) as usize];
    let second = (0..s.len()).filter(|&i| i != choice).map(|i| s[i]).fold(0f32, f32::max);
    let margin = if best > 0.0 { ((best - second) / best) as f64 } else { 0.0 };
    (choice, margin)
}

/// answered at margin theta: theta 0 answers everything; otherwise margin >= theta
fn answers(m: f64, theta: f64) -> bool { theta == 0.0 || m >= theta }

struct Tally { n: usize, right: usize, ans: usize, ans_right: usize }
fn tally(qs: &[Q], res: &[(usize, f64)], set: usize, theta: f64) -> Tally {
    let mut t = Tally { n: 0, right: 0, ans: 0, ans_right: 0 };
    for (q, &(ch, m)) in qs.iter().zip(res) {
        if q.set != set { continue; }
        t.n += 1;
        let ok = ch == q.key;
        t.right += ok as usize;
        if answers(m, theta) { t.ans += 1; t.ans_right += ok as usize; }
    }
    t
}
fn show(name: &str, t: &Tally) {
    let wrong = t.ans - t.ans_right;
    println!("   {:<10} answered {:>4}/{:<4} right {:>4} wrong {:>4} right-wrong {:+5}  share right {:.1}%",
        name, t.ans, t.n, t.ans_right, wrong, t.ans_right as i64 - wrong as i64, t.ans_right as f64 / t.ans.max(1) as f64 * 100.0);
}

/// the margin chosen on a dev set: the one in 0.00..0.50 that maximises right - wrong among the
/// answered questions (ties: the smaller margin, answering more)
fn choose_theta(qs: &[Q], res: &[(usize, f64)], set: usize) -> (f64, i64) {
    let mut best = (0.0, i64::MIN);
    for k in 0..=50 {
        let th = k as f64 / 100.0;
        let t = tally(qs, res, set, th);
        let net = t.ans_right as i64 - (t.ans - t.ans_right) as i64;
        if net > best.1 { best = (th, net); }
    }
    best
}

fn threads() -> usize { std::env::var("AVA_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(4).clamp(1, 4) }

fn read_all(files: &[&str], stem_on: bool, cue_min: f32) -> (Vec<Q>, Cues, Vec<Hit>, f64) {
    let qs: Vec<Q> = files.iter().enumerate().flat_map(|(i, f)| load(f, i)).collect();
    let c = build_cues(&qs, stem_on);
    println!("questions {}  choices {}  cue words {}  word endings {}  calling-word floor {:.3}", qs.len(), c.cw.len(), c.vocab.len(), stem_on, cue_min);
    let th = threads();
    let t0 = Instant::now();
    let sp = read_df(&c, th);
    let t1 = t0.elapsed().as_secs_f64();
    println!("reading 1 (specificity): {:.1} s   sentences {}   mean words/sentence {:.2}   free RAM {:.1} GB", t1, sp.n, sp.avg_len, free_gb());
    let t0 = Instant::now();
    let top = recall(&c, &sp, th, cue_min);
    let t2 = t0.elapsed().as_secs_f64();
    println!("reading 2 (recall):      {:.1} s   threads {th}   peak RAM {:.3} GB   free RAM {:.1} GB", t2, peak_gb(), free_gb());
    (qs, c, top, t1 + t2)
}

fn dev() {
    let stem_on = std::env::var("AVA_STEM").map(|v| v != "0").unwrap_or(FROZEN_STEM);
    let cue_min = std::env::var("AVA_CUE_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(FROZEN_CUE_MIN);
    let (qs, c, top, _) = read_all(&DEV_FILES, stem_on, cue_min);
    let missing = qs.iter().filter(|q| q.key == usize::MAX).count();
    if missing > 0 { println!("questions without a usable answer key: {missing}"); }
    println!("\naccuracy (%), answering every question");
    println!("{:<16} {:>10} {:>10} {:>10} {:>10}", "damping / k", DEV_NAMES[0], DEV_NAMES[1], DEV_NAMES[2], DEV_NAMES[3]);
    let mut all = Vec::new();
    for b in 0..BS.len() {
        for &k in &KS {
            let res: Vec<(usize, f64)> = (0..qs.len()).map(|qi| pick(&qs[qi].id, &scores(&c, &top, qi, b, k))).collect();
            let acc: Vec<f64> = (0..4).map(|s| { let t = tally(&qs, &res, s, 0.0); t.right as f64 / t.n.max(1) as f64 * 100.0 }).collect();
            println!("b={:<4} k={:<2}      {:>10.2} {:>10.2} {:>10.2} {:>10.2}", BS[b], k, acc[0], acc[1], acc[2], acc[3]);
            let (r0, r1) = (tally(&qs, &res, 0, 0.0).right, tally(&qs, &res, 1, 0.0).right);
            all.push((b, k, acc, res, (r0, r0 + r1)));
        }
    }
    // best Easy-Train accuracy; a tie goes to the better pooled Easy Train+Dev accuracy
    let best = all.iter().max_by_key(|x| x.4).unwrap();
    println!("\nbest on Easy-Train (tie -> pooled Easy Train+Dev): b={} k={}  (Easy-Dev {:.2}%)   frozen: b={} k={} floor {:.3} endings {}",
        BS[best.0], best.1, best.2[1], BS[FROZEN_B], FROZEN_K, FROZEN_CUE_MIN, FROZEN_STEM);
    for (b, k, _, res, _) in all.iter().filter(|x| (x.0, x.1) == (best.0, best.1) || (x.0, x.1) == (FROZEN_B, FROZEN_K)) {
        let (th, net) = choose_theta(&qs, res, 1);
        println!("b={} k={}: margin chosen on Easy-Dev {:.2} (right - wrong there {:+}); frozen margin {:.2}", BS[*b], k, th, net, FROZEN_THETA);
        for s in 0..4 { show(DEV_NAMES[s], &tally(&qs, res, s, th)); }
    }
    if let Ok(p) = std::env::var("AVA_DEV_TSV") {
        let mut f = std::io::BufWriter::new(File::create(&p).expect("create tsv"));
        for (qi, q) in qs.iter().enumerate() {
            let s = scores(&c, &top, qi, FROZEN_B, FROZEN_K);
            let (ch, m) = pick(&q.id, &s);
            let _ = writeln!(f, "{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{}", DEV_NAMES[q.set], q.id, q.key, ch, m,
                s.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(","), q.stem.replace('\t', " "),
                q.choices.iter().map(|x| x.1.replace('\t', " ")).collect::<Vec<_>>().join(" | "));
        }
        println!("wrote {p}");
    }
}

// ---------------------------------------------------------------- the one Test run
fn test(out: &str, files: [&str; 2], names: [&str; 2]) {
    if std::path::Path::new(out).exists() { die(&format!("refusing: {out} exists; the Test run happens once")); }
    let mut f = std::io::BufWriter::new(File::create(out).unwrap_or_else(|e| die(&format!("cannot create {out}: {e}"))));
    let t0 = Instant::now();
    let (qs, c, top, corpus_s) = read_all(&files, FROZEN_STEM, FROZEN_CUE_MIN);
    println!("frozen: word endings {}  calling-word floor {:.3}  damping b={}  k={}  margin {:.2}",
        FROZEN_STEM, FROZEN_CUE_MIN, BS[FROZEN_B], FROZEN_K, FROZEN_THETA);
    let res: Vec<(usize, f64)> = (0..qs.len()).map(|qi| pick(&qs[qi].id, &scores(&c, &top, qi, FROZEN_B, FROZEN_K))).collect();
    let _ = writeln!(f, "set\tid\tkey\tchoice\tright\tmargin\tanswered\tn_choices\tscores");
    for (qi, q) in qs.iter().enumerate() {
        let s = scores(&c, &top, qi, FROZEN_B, FROZEN_K);
        let (ch, m) = res[qi];
        let _ = writeln!(f, "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{}\t{}", names[q.set], q.id, q.key, ch, (ch == q.key) as u8, m,
            answers(m, FROZEN_THETA) as u8, q.choices.len(), s.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(","));
    }
    drop(f);
    let missing = qs.iter().filter(|q| q.key == usize::MAX).count();
    println!("questions without a usable answer key: {missing}");
    println!("\nanswering every question:");
    for s in 0..2 {
        let t = tally(&qs, &res, s, 0.0);
        println!("   {:<10} {} / {} right = {:.2}%", names[s], t.right, t.n, t.right as f64 / t.n.max(1) as f64 * 100.0);
    }
    println!("abstaining reading (margin {:.2}, chosen on Easy-Dev):", FROZEN_THETA);
    for s in 0..2 { show(names[s], &tally(&qs, &res, s, FROZEN_THETA)); }
    // paired bootstrap over the Easy questions: her accuracy, and her accuracy minus chance
    // (1 / number of choices) on the same resampled questions
    let easy: Vec<(f64, f64)> = qs.iter().zip(&res).filter(|(q, _)| q.set == 0)
        .map(|(q, r)| ((r.0 == q.key) as u8 as f64, 1.0 / q.choices.len().max(1) as f64)).collect();
    let n = easy.len();
    let mut rng = 1u64;
    let (mut acc, mut gain): (Vec<f64>, Vec<f64>) = (Vec::with_capacity(20_000), Vec::with_capacity(20_000));
    for _ in 0..20_000 {
        let (mut a, mut ch) = (0.0, 0.0);
        for _ in 0..n { let x = easy[(splitmix(&mut rng) % n as u64) as usize]; a += x.0; ch += x.1; }
        acc.push(a / n as f64 * 100.0);
        gain.push((a - ch) / n as f64 * 100.0);
    }
    acc.sort_by(|a, b| a.partial_cmp(b).unwrap());
    gain.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let a_all: f64 = easy.iter().map(|x| x.0).sum::<f64>() / n as f64 * 100.0;
    let chance: f64 = easy.iter().map(|x| x.1).sum::<f64>() / n as f64 * 100.0;
    println!("\n{} accuracy {:.2}%  95% CI {:.2} .. {:.2}  (paired bootstrap, splitmix64 seed 1, 20,000 resamples)", names[0], a_all, acc[500], acc[19_499]);
    println!("{} chance {:.2}%; accuracy minus chance {:+.2} points, 95% CI {:+.2} .. {:+.2}", names[0], chance, a_all - chance, gain[500], gain[19_499]);
    println!("corpus reading time {:.1} s (both passes), whole run {:.1} s, peak RAM {:.3} GB", corpus_s, t0.elapsed().as_secs_f64(), peak_gb());
    let pass = a_all >= 50.0 && missing == 0 && res.iter().zip(&qs).filter(|(_, q)| q.set == 0).all(|(r, _)| r.0 != usize::MAX);
    println!("\nVERDICT ({} accuracy >= 50%, answering every question): {}", names[0], if pass { "PASS" } else { "FAIL" });
    println!("wrote {out}");
}

// ---------------------------------------------------------------- leakage: verbatim stems in the corpus
/// For each question: (1) its whole stem appears word for word inside one corpus sentence;
/// (2) such a sentence also holds the right answer's words in a row; (3) the stem is followed
/// directly by the right answer; (4) some sentence of the stem (>= 6 words) appears word for word
/// in a corpus sentence that also holds the right answer. Words as her eyes see them, no endings cut.
fn leak(out: &str, split: &str) {
    let files: Vec<&str> = match split { "dev" => DEV_FILES.to_vec(), "test" => TEST_FILES.to_vec(), _ => die("leak split: dev|test") };
    let names: Vec<&str> = if split == "dev" { DEV_NAMES.to_vec() } else { TEST_NAMES.to_vec() };
    let qs: Vec<Q> = files.iter().enumerate().flat_map(|(i, f)| load(f, i)).collect();
    let toks = |s: &str| -> Vec<u64> { let mut v = Vec::new(); words(s.as_bytes(), false, |t| v.push(fnv(t))); v };
    const A: usize = 3; // anchor: a pattern's first A words
    let anchor = |t: &[u64]| -> u64 { t.iter().take(A).fold(0x9e37_79b9_7f4a_7c15u64, |h, &x| (h ^ x).wrapping_mul(0x100_0000_01b3)) };
    // patterns: (question, whole stem?, words)
    let mut pats: Vec<(usize, bool, Vec<u64>)> = Vec::new();
    let mut ans: Vec<Vec<u64>> = Vec::new();
    for (qi, q) in qs.iter().enumerate() {
        ans.push(q.choices.get(q.key).map(|c| toks(&c.1)).unwrap_or_default());
        let whole = toks(&q.stem);
        if whole.len() >= A { pats.push((qi, true, whole.clone())); }
        let mut piece = String::new();
        for ch in q.stem.chars().chain(std::iter::once(' ')) {
            piece.push(ch);
            if matches!(ch, '.' | '?' | '!') {
                let t = toks(&piece);
                if t.len() >= 6 && t.len() < whole.len() { pats.push((qi, false, t)); }
                piece.clear();
            }
        }
    }
    let mut index: FxMap<u64, Vec<u32>> = FxMap::default();
    for (pi, p) in pats.iter().enumerate() { index.entry(anchor(&p.2)).or_default().push(pi as u32); }
    let has = |line: &[u64], a: &[u64]| !a.is_empty() && line.windows(a.len()).any(|w| w == a);
    let t0 = Instant::now();
    let parts = stream(threads(), |_, each| {
        let mut flags = vec![0u8; qs.len()];
        let mut t: Vec<u64> = Vec::new();
        each(&mut |line: &[u8]| {
            t.clear();
            words(line, false, |w| t.push(fnv(w)));
            if t.len() < A { return; }
            for i in 0..=t.len() - A {
                let Some(list) = index.get(&anchor(&t[i..])) else { continue };
                for &pi in list {
                    let (qi, whole, p) = &pats[pi as usize];
                    if t.len() - i < p.len() || t[i..i + p.len()] != p[..] { continue; }
                    let a = &ans[*qi];
                    let with = has(&t, a);
                    if *whole {
                        flags[*qi] |= 1;
                        if with { flags[*qi] |= 2; }
                        if !a.is_empty() && t.len() >= i + p.len() + a.len() && t[i + p.len()..i + p.len() + a.len()] == a[..] { flags[*qi] |= 4; }
                    } else if with { flags[*qi] |= 8; }
                }
            }
        });
        flags
    });
    let mut flags = vec![0u8; qs.len()];
    for p in parts { for (x, y) in flags.iter_mut().zip(p) { *x |= y; } }
    let mut f = std::io::BufWriter::new(File::create(out).unwrap_or_else(|e| die(&format!("cannot create {out}: {e}"))));
    let _ = writeln!(f, "set\tid\tstem_verbatim\tstem_and_answer_in_line\tstem_then_answer\tstem_sentence_and_answer_in_line");
    for (q, &x) in qs.iter().zip(&flags) {
        let _ = writeln!(f, "{}\t{}\t{}\t{}\t{}\t{}", names[q.set], q.id, x & 1, (x >> 1) & 1, (x >> 2) & 1, (x >> 3) & 1);
    }
    println!("leak check over the corpus in {:.1} s ({} patterns)", t0.elapsed().as_secs_f64(), pats.len());
    for s in 0..files.len() {
        let of: Vec<u8> = qs.iter().zip(&flags).filter(|(q, _)| q.set == s).map(|(_, &x)| x).collect();
        let n = |b: u8| of.iter().filter(|&&x| x & b != 0).count();
        println!("   {:<10} questions {:>4}: whole stem verbatim {:>4}; + right answer in that sentence {:>4}; stem then answer {:>4}; a stem sentence (>= 6 words) + right answer in one sentence {:>4}; any {:>4}",
            names[s], of.len(), n(1), n(2), n(4), n(8), of.iter().filter(|&&x| x != 0).count());
    }
    println!("wrote {out}   peak RAM {:.3} GB", peak_gb());
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match (a.get(1).map(String::as_str), a.get(2), a.get(3)) {
        (Some("dev"), _, _) => dev(),
        (Some("test"), Some(out), _) => test(out, TEST_FILES, TEST_NAMES),
        (Some("dryrun"), Some(out), _) => test(out, [DEV_FILES[1], DEV_FILES[3]], [DEV_NAMES[1], DEV_NAMES[3]]),
        (Some("leak"), Some(out), Some(split)) => leak(out, split),
        _ => die("usage: read_facts dev | test <out.tsv> | dryrun <out.tsv> | leak <out.tsv> <dev|test>"),
    }
}
