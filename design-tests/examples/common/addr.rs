// TESTING / DESIGN ONLY.
// v29 addressing groundwork: what address_bench.rs and frames_build.rs share.
//   - the 6 USED blocks (day1.txt lessons, day2.txt questions), read from their block dirs
//   - compose.rs's talk tokenizer (seq::tokenize, or code_tokenize for code-looking lines)
//   - compose.rs's spec weight: ln(N / (1 + times w was a reply word)), N = her cortex writes
//   - the UniMorph lexeme table exactly as compose.rs's lemmas() builds it
//   - splitmix64, FNV-1a, and the process's own RAM readings (Windows API, no crates)
// Nothing here reads the fact files or the fresh confirmation blocks.
#![allow(dead_code)]
use design_tests::seq;
use std::collections::HashMap;

/// where experiment outputs go; set AVA_SCRATCH at build time to move it (default: ../scratch)
pub const SCRATCH: &str = match option_env!("AVA_SCRATCH") { Some(p) => p, None => "../scratch" };
/// the ava folder (the one holding data/); set AVA_ROOT at build time (default: .., from design-tests)
pub const AVA: &str = match option_env!("AVA_ROOT") { Some(p) => p, None => ".." };
/// (dir under the scratchpad, tag, domain)
pub const BLOCKS: [(&str, &str, &str); 6] = [
    ("v27/s2/RD", "RD", "rustdoc"), ("v27/s2/RN", "RN", "refnomicon"), ("v27/s2/AI", "AI", "ai"),
    ("v27/s3/RD2", "RD2", "rustdoc"), ("v27/s3/RN2", "RN2", "refnomicon"), ("v27/s3/AI2", "AI2", "ai"),
];
pub const DOMAINS: [&str; 3] = ["rustdoc", "refnomicon", "ai"];

pub fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
pub fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3); }
    h
}

// ---------------------------------------------------------------- tokenizers (compose.rs)

/// compose.rs code_tokenize, copied unchanged
pub fn code_tokenize(s: &str) -> Vec<String> {
    const OPS3: [&str; 3] = ["<<=", ">>=", "..="];
    const OPS2: [&str; 20] = ["::", "->", "=>", "==", "!=", "<=", ">=", "&&", "||",
        "+=", "-=", "*=", "/=", "%=", "^=", "&=", "|=", "<<", ">>", ".."];
    let c: Vec<char> = s.chars().collect();
    let ident = |ch: char| ch.is_ascii_alphanumeric() || ch == '_';
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() { i += 1; continue; }
        let end = if ident(ch) {
            (i..c.len()).find(|&j| !ident(c[j])).unwrap_or(c.len())
        } else if ch == '"' {
            let mut j = i + 1;
            while j < c.len() && c[j] != '"' { j += if c[j] == '\\' { 2 } else { 1 }; }
            (j + 1).min(c.len())
        } else if ch == '\'' {
            let close = if c.get(i + 1) == Some(&'\\') {
                (i + 3..c.len().min(i + 12)).find(|&j| c[j] == '\'')
            } else if c.get(i + 2) == Some(&'\'') { Some(i + 2) } else { None };
            match close {
                Some(j) => j + 1,
                None if c.get(i + 1).map_or(false, |&x| x.is_ascii_alphabetic() || x == '_') =>
                    (i + 1..c.len()).find(|&j| !ident(c[j])).unwrap_or(c.len()),
                None => i + 1,
            }
        } else {
            let at = |k: usize| -> String { c[i..(i + k).min(c.len())].iter().collect() };
            if OPS3.contains(&at(3).as_str()) { i + 3 }
            else if OPS2.contains(&at(2).as_str()) { i + 2 }
            else { i + 1 }
        };
        out.push(c[i..end].iter().collect());
        i = end;
    }
    out
}
/// compose.rs talk_tokenize: a line that looks like code is read the way code was learned
pub fn talk_tokenize(s: &str) -> Vec<String> {
    if s.chars().any(|ch| "{}();:=<>&_".contains(ch)) { code_tokenize(s) } else { seq::tokenize(s) }
}
/// teach_test.rs norm: how the grader compares answers
pub fn norm(s: &str) -> String {
    let s = s.to_lowercase().replace('\u{2019}', "'");
    let mut toks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '\'' { cur.push(c); continue; }
        if !cur.is_empty() { toks.push(std::mem::take(&mut cur)); }
        if !(c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)) { toks.push(c.to_string()); }
    }
    if !cur.is_empty() { toks.push(cur); }
    toks.join(" ")
}
/// meaning_build.rs words(): tokens holding a letter or digit
pub fn words(s: &str) -> Vec<String> {
    seq::tokenize(s).into_iter().filter(|w| w.chars().any(|c| c.is_alphanumeric())).collect()
}

// ---------------------------------------------------------------- the blocks

pub struct Lesson { pub q: Vec<String>, pub a: Vec<String>, pub q_raw: String, pub a_raw: String }
/// kind: 0 = as taught (q0), 1 = paraphrase (q1, q2), 2 = untaught
pub struct Question { pub toks: Vec<String>, pub raw: String, pub kind: u8, pub fact: Option<usize> }
pub struct Block { pub dir: String, pub tag: String, pub dom: usize, pub lessons: Vec<Lesson>, pub qs: Vec<Question> }

fn read(p: &str) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}"))
}
fn lines(p: &str) -> Vec<String> {
    read(p).replace("\r\n", "\n").split('\n').map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
}

pub fn load_block(rel: &str, tag: &str, dom: &str) -> Block {
    let dir = format!("{SCRATCH}/{rel}");
    let d1 = lines(&format!("{dir}/day1.txt"));
    let mut lessons: Vec<Lesson> = Vec::new();
    let mut last: Option<String> = None;
    for l in &d1 {
        if l == ":q" { break; }
        if let Some(t) = l.strip_prefix('=') {
            let q = last.clone().unwrap_or_else(|| panic!("{dir}/day1.txt: answer before any question"));
            let (qt, at) = (talk_tokenize(&q), talk_tokenize(t));
            // compose keeps an engram once: an identical (q, a) is not stored twice
            if !lessons.iter().any(|x| x.q == qt && x.a == at) {
                lessons.push(Lesson { q: qt, a: at, q_raw: q, a_raw: t.trim().to_string() });
            }
        } else { last = Some(l.clone()); }
    }
    let d2 = lines(&format!("{dir}/day2.txt"));
    let mut qs = Vec::new();
    for (i, l) in d2.iter().enumerate() {
        if l == ":q" { break; }
        let n = lessons.len();
        let (kind, fact) = if i < 3 * n { (if i % 3 == 0 { 0 } else { 1 }, Some(i / 3)) } else { (2, None) };
        qs.push(Question { toks: talk_tokenize(l), raw: l.clone(), kind, fact });
    }
    let dom = DOMAINS.iter().position(|d| *d == dom).expect("domain");
    Block { dir, tag: tag.to_string(), dom, lessons, qs }
}

// ---------------------------------------------------------------- spec weights (compose.rs)

/// her life's reply-side writes, as compose.rs reads data/triples.tsv with
/// ARGS "15000 0 0 0 5 2 5 0 128": the first 1,200 short exchanges (and the long ones
/// among them) held out, then onward to the 16,200th short exchange. every reply token and
/// one end mark per exchange is a write (reply::turns). returns (N, count per target word).
pub fn life_writes(n_short: usize, k_test: usize) -> (u64, HashMap<String, u32>, usize) {
    let raw = read(&format!("{AVA}/data/triples.tsv"));
    let mut pairs: Vec<Vec<String>> = Vec::new();
    let (mut seen_short, mut n_region) = (0usize, 0usize);
    for line in raw.lines() {
        let mut p = line.split('\t');
        let (Some(f0), Some(q), Some(r)) = (p.next(), p.next(), p.next()) else { continue };
        let tag = f0.strip_prefix('@').map(str::trim);
        let code = matches!(tag, Some("rust") | Some("ava_code")) || tag.map_or(false, |t| t.starts_with("code_"));
        let tok = |s: &str| if code { code_tokenize(s) } else { seq::tokenize(s) };
        let (qt, rt) = (tok(q), tok(r));
        let short = rt.len() <= 16 && qt.len() <= 20;
        if !qt.is_empty() && !rt.is_empty() {
            let held = seen_short < k_test;
            if held { n_region += 1; }
            if short { seen_short += 1; }
            pairs.push(rt);
            if seen_short >= k_test + n_short { break; }
        }
    }
    let n_test = n_region.min(pairs.len().saturating_sub(1));
    let (mut n, mut c): (u64, HashMap<String, u32>) = (0, HashMap::new());
    for r in &pairs[n_test..] {
        n += r.len() as u64 + 1;
        for w in r { *c.entry(w.clone()).or_default() += 1; }
    }
    (n, c, pairs.len() - n_test)
}

/// spec weights for one block: her life plus that block's sleep (each lesson's answer replayed
/// once: its tokens and an end mark)
pub struct Spec { pub n: f64, pub c: HashMap<String, u32>, pub sleep_words: usize }
impl Spec {
    pub fn new(life_n: u64, life_c: &HashMap<String, u32>, b: &Block) -> Spec {
        let mut c = life_c.clone();
        let mut n = life_n;
        let mut sw = 0usize;
        for l in &b.lessons {
            n += l.a.len() as u64 + 1;
            sw += l.a.len() + 1;
            for w in &l.a { *c.entry(w.clone()).or_default() += 1; }
        }
        Spec { n: n as f64, c, sleep_words: sw }
    }
    pub fn of(&self, w: &str) -> f32 {
        let c = self.c.get(w).copied().unwrap_or(0) as f32;
        (self.n as f32 / (1.0 + c)).ln().max(0.0)
    }
}

// ---------------------------------------------------------------- lexemes (compose.rs lemmas())

pub struct Lex { pub of: HashMap<String, Vec<u32>>, pub name: Vec<String> }
impl Lex {
    pub fn load(path: &str) -> Lex {
        let text = read(path);
        let (mut id, mut of): (HashMap<String, u32>, HashMap<String, Vec<u32>>) = Default::default();
        let mut name: Vec<String> = Vec::new();
        for l in text.lines() {
            let mut p = l.split('\t');
            let (Some(lemma), Some(form)) = (p.next(), p.next()) else { continue };
            if lemma.contains(' ') || form.contains(' ') { continue; }
            let n = id.len() as u32;
            let lw = lemma.to_lowercase();
            let i = *id.entry(lw.clone()).or_insert(n);
            if i == n { name.push(lw); }
            for w in [lemma.to_lowercase(), form.to_lowercase()] {
                let v = of.entry(w).or_default();
                if !v.contains(&i) { v.push(i); }
            }
        }
        Lex { of, name }
    }
    pub fn same(&self, a: &str, b: &str) -> bool {
        match (self.of.get(a), self.of.get(b)) { (Some(x), Some(y)) => x.iter().any(|i| y.contains(i)), _ => false }
    }
}

// ---------------------------------------------------------------- RAM (Windows API)

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
pub fn free_gb() -> f64 {
    #[cfg(windows)]
    unsafe {
        let mut m: win::MemStatusEx = std::mem::zeroed();
        m.len = std::mem::size_of::<win::MemStatusEx>() as u32;
        if win::GlobalMemoryStatusEx(&mut m) != 0 { return m.avail_phys as f64 / 1e9; }
    }
    f64::INFINITY
}
/// this process's peak working set so far, GB
pub fn peak_gb() -> f64 {
    #[cfg(windows)]
    unsafe {
        let mut p: win::Pmc = std::mem::zeroed();
        p.cb = std::mem::size_of::<win::Pmc>() as u32;
        if win::K32GetProcessMemoryInfo(win::GetCurrentProcess(), &mut p, p.cb) != 0 { return p.peak_ws as f64 / 1e9; }
    }
    0.0
}
/// stop before the machine runs short (the guard's floor is 3 GB; stop a little above it)
pub fn ram_guard(where_: &str) {
    let f = free_gb();
    if f < 3.5 {
        eprintln!("!! stopping at {where_}: free RAM {f:.2} GB is under 3.5 GB");
        std::process::exit(3);
    }
}

// ---------------------------------------------------------------- frames (C2), the file frames_build writes

/// one profiled word: its frame contexts on each side, (context key, PMI), sorted by key
pub struct Frames {
    pub names: Vec<String>,                      // corpus token id -> word (to read a context back)
    pub ix: HashMap<String, usize>,              // profiled word -> row
    pub words: Vec<String>,
    pub events: Vec<u64>,
    pub prof: Vec<[Vec<(u64, f32)>; 2]>,
    pub sum: Vec<[f32; 2]>,
}
pub const CTX_CAP: u32 = (1 << 20) - 2;
/// a context: side (0 = left, 1 = right), distance 1..3, the tokens between (id + 1, 0 = none), the filler
pub fn ctx_key(side: u64, dist: u64, t1: u64, t2: u64, f: u64) -> u64 { side << 63 | dist << 61 | t1 << 40 | t2 << 20 | f }
impl Frames {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        // a text first line, so even this data file says what it is; load() skips it
        let mut b: Vec<u8> = b"// TESTING / DESIGN ONLY.\nAVAFRM01".to_vec();
        let s = |b: &mut Vec<u8>, w: &str| { b.extend((w.len() as u16).to_le_bytes()); b.extend(w.as_bytes()); };
        b.extend((self.names.len() as u32).to_le_bytes());
        for w in &self.names { s(&mut b, w); }
        b.extend((self.words.len() as u32).to_le_bytes());
        for (i, w) in self.words.iter().enumerate() {
            s(&mut b, w);
            b.extend(self.events[i].to_le_bytes());
            for side in 0..2 {
                b.extend((self.prof[i][side].len() as u32).to_le_bytes());
                for (k, v) in &self.prof[i][side] { b.extend(k.to_le_bytes()); b.extend(v.to_le_bytes()); }
            }
        }
        std::fs::write(path, b)
    }
    pub fn load(path: &str) -> std::io::Result<Frames> {
        let raw = std::fs::read(path)?;
        let skip = if raw.starts_with(b"//") { raw.iter().position(|&c| c == b'\n').map_or(0, |p| p + 1) } else { 0 };
        let b = raw[skip..].to_vec();
        if b.len() < 12 || &b[..8] != b"AVAFRM01" { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not AVAFRM01")); }
        let mut p = 8usize;
        let u32_ = |p: &mut usize| { let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap()); *p += 4; v };
        let u64_ = |p: &mut usize| { let v = u64::from_le_bytes(b[*p..*p + 8].try_into().unwrap()); *p += 8; v };
        let f32_ = |p: &mut usize| { let v = f32::from_le_bytes(b[*p..*p + 4].try_into().unwrap()); *p += 4; v };
        let st = |p: &mut usize| { let l = u16::from_le_bytes([b[*p], b[*p + 1]]) as usize; *p += 2;
            let w = String::from_utf8_lossy(&b[*p..*p + l]).into_owned(); *p += l; w };
        let nn = u32_(&mut p) as usize;
        let names: Vec<String> = (0..nn).map(|_| st(&mut p)).collect();
        let nw = u32_(&mut p) as usize;
        let (mut words, mut events, mut prof, mut sum) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..nw {
            words.push(st(&mut p));
            events.push(u64_(&mut p));
            let mut pr: [Vec<(u64, f32)>; 2] = [Vec::new(), Vec::new()];
            let mut sm = [0f32; 2];
            for side in 0..2 {
                let n = u32_(&mut p) as usize;
                for _ in 0..n { let k = u64_(&mut p); let v = f32_(&mut p); pr[side].push((k, v)); sm[side] += v; }
            }
            prof.push(pr);
            sum.push(sm);
        }
        let ix = words.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect();
        Ok(Frames { names, ix, words, events, prof, sum })
    }
    fn lin(a: &[(u64, f32)], b: &[(u64, f32)], sa: f32, sb: f32) -> f32 {
        if sa + sb <= 0.0 { return 0.0; }
        let (mut i, mut j, mut num) = (0usize, 0usize, 0f32);
        while i < a.len() && j < b.len() {
            if a[i].0 == b[j].0 { num += a[i].1 + b[j].1; i += 1; j += 1; }
            else if a[i].0 < b[j].0 { i += 1 } else { j += 1 }
        }
        num / (sa + sb)
    }
    /// DIRT: both slots must agree, sqrt(Lin_left * Lin_right); 1 for the same word, 0 if unread
    pub fn sim_ix(&self, i: usize, j: usize) -> f32 {
        if i == j { return 1.0; }
        let l = Self::lin(&self.prof[i][0], &self.prof[j][0], self.sum[i][0], self.sum[j][0]);
        let r = Self::lin(&self.prof[i][1], &self.prof[j][1], self.sum[i][1], self.sum[j][1]);
        (l * r).sqrt()
    }
    pub fn sim(&self, a: &str, b: &str) -> f32 {
        if a == b { return 1.0; }
        match (self.ix.get(a), self.ix.get(b)) { (Some(&i), Some(&j)) => self.sim_ix(i, j), _ => 0.0 }
    }
    /// a context read back as words, X standing for the profiled word
    pub fn show(&self, k: u64) -> String {
        let (side, dist) = (k >> 63, (k >> 61) & 3);
        let t1 = ((k >> 40) & 0xF_FFFF) as usize;
        let t2 = ((k >> 20) & 0xF_FFFF) as usize;
        let f = (k & 0xF_FFFF) as usize;
        let nm = |i: usize| self.names.get(i).cloned().unwrap_or_else(|| "?".into());
        let mut mid: Vec<String> = Vec::new();
        if dist >= 2 && t1 > 0 { mid.push(nm(t1 - 1)); }
        if dist >= 3 && t2 > 0 { mid.push(nm(t2 - 1)); }
        if side == 0 { format!("{} {} X", nm(f), mid.join(" ")).replace("  ", " ") }
        else { format!("X {} {}", mid.join(" "), nm(f)).replace("  ", " ") }
    }
}
