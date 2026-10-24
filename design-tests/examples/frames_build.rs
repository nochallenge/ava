// TESTING / DESIGN ONLY.
// v29 C2: FRAMES LEARNED FROM READING (DIRT, Lin & Pantel 2001; Lin 1998), as Hebbian
// slot-filler counting. No gradient, no training loop: one reading, counts, and overlap.
//
//   cargo build --release --example frames_build
//   frames_build <out-file>          (default: scratchpad/v29/frames.bin)
//
// A word is known by the FRAMES it fills. Reading a sentence, every word w at position i is
// seen with the words up to 3 places away: a context c = (side, the tokens strictly between,
// the filler). "x was designed by y" gives designed the frames  [x was X]  and  [X by y].
// Two words are the same relation only if they fill the same frames with the SAME fillers
// on BOTH sides (DIRT's geometric mean): sim = sqrt(Lin_left * Lin_right), each side Lin's
// measure over PMI-weighted contexts. Words that merely occur near each other (input /
// output) share the topic, not necessarily the fillers.
//
// Brain reading: a first exposure only primes a synapse (a Bloom-filter "seen once" bit); a
// repeat lays down a count. So contexts heard once leave no lasting trace (they could not be
// shared reliably anyway) and memory stays small.
//
// What she reads: exactly what meaning_build.rs reads (eng_sentences col 3, cornell lines,
// triples.tsv minus the 1,200 dialogue-exam exchanges with each turn once, Rust reference +
// nomicon HTML <main> text, Rust by Example markdown). Never the fact files.
// Profiles are kept for the bench vocabulary only (every word of the six USED blocks' lessons
// and questions, the word list alone, plus a fixed probe list for display), to keep RAM small.
#[path = "common/addr.rs"]
mod addr;
use addr::*;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::{Path, PathBuf};

const WINDOW: usize = 3;
const TOP: usize = 2000;
const PROBES: [&str; 40] = ["car", "happy", "begin", "big", "mother", "year", "variable", "mutable", "function", "network",
    "designed", "invented", "memory", "neuron", "gate", "start", "large", "automobile", "created", "inspired", "modelled",
    "glad", "small", "when", "person", "who", "input", "output", "define", "declare", "changeable", "coined", "introduced",
    "wrote", "built", "proposed", "keyword", "word", "constant", "immutable"];

#[derive(Default)]
struct Mix(u64);
impl Hasher for Mix {
    fn finish(&self) -> u64 { let mut z = self.0; z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb); z ^ (z >> 31) }
    fn write(&mut self, b: &[u8]) { for &x in b { self.0 = (self.0 ^ x as u64).wrapping_mul(0x100_0000_01b3); } }
    fn write_u64(&mut self, x: u64) { self.0 = self.0.wrapping_add(x).wrapping_add(0x9e37_79b9_7f4a_7c15); }
    fn write_u32(&mut self, x: u32) { self.write_u64(x as u64) }
}
type Map<K, V> = HashMap<K, V, BuildHasherDefault<Mix>>;

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

struct Corpus { id: HashMap<String, u32>, count: Vec<u64>, toks: Vec<u32>, starts: Vec<u32> }
impl Corpus {
    fn push(&mut self, ws: Vec<String>) {
        if ws.is_empty() { return; }
        self.starts.push(self.toks.len() as u32);
        for w in ws {
            let n = self.id.len() as u32;
            let i = *self.id.entry(w).or_insert(n);
            if i == n { self.count.push(0); }
            self.count[i as usize] += 1;
            self.toks.push(i);
        }
    }
    fn sent(&self, k: usize) -> &[u32] {
        let a = self.starts[k] as usize;
        let b = self.starts.get(k + 1).map_or(self.toks.len(), |&x| x as usize);
        &self.toks[a..b]
    }
}

struct Bloom { bits: Vec<u64>, mask: u64 }
impl Bloom {
    /// true if every probe bit was already set (seen before); sets them
    fn seen(&mut self, h: u64) -> bool {
        let (h1, h2) = (h, (h >> 32) | 1);
        let mut all = true;
        for k in 0..4u64 {
            let i = h1.wrapping_add(k.wrapping_mul(h2)) & self.mask;
            let (w, b) = ((i >> 6) as usize, i & 63);
            if self.bits[w] >> b & 1 == 0 { all = false; self.bits[w] |= 1 << b; }
        }
        all
    }
}
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    // FRAMES_VARIANT=x (EXPLORATORY, added after the pre-registered C2 ran): a frame that only this
    // word ever fills can never be shared, so it is dropped; the 2,000 kept per side are the ones
    // heard MOST often (count), not the rarest (PMI). weights stay PMI.
    let variant_x = std::env::var("FRAMES_VARIANT").map_or(false, |v| v == "x");
    let out = a.get(1).cloned().unwrap_or_else(|| format!("{SCRATCH}/v29/{}", if variant_x { "frames_x.bin" } else { "frames.bin" }));
    let t0 = std::time::Instant::now();
    println!("// TESTING / DESIGN ONLY.");
    let data = PathBuf::from(AVA).join("data");
    ram_guard("start");

    // ---------------------------------------------------------------- what she reads (as meaning_build.rs)
    let mut c = Corpus { id: HashMap::new(), count: Vec::new(), toks: Vec::new(), starts: Vec::new() };
    let mut n_src: Vec<(&str, usize)> = Vec::new();
    let tat = String::from_utf8_lossy(&std::fs::read(data.join("eng_sentences.tsv")).expect("eng_sentences.tsv")).into_owned();
    let before = c.starts.len();
    for l in tat.lines() { if let Some(s) = l.split('\t').nth(2) { c.push(words(s)); } }
    n_src.push(("english sentences", c.starts.len() - before));
    drop(tat);
    ram_guard("after english sentences");
    let corn = String::from_utf8_lossy(&std::fs::read(data.join("cornell_lines.txt")).expect("cornell_lines.txt")).into_owned();
    let before = c.starts.len();
    for l in corn.lines() { if let Some(s) = l.rsplit(" +++$+++ ").next() { c.push(words(s)); } }
    n_src.push(("movie lines", c.starts.len() - before));
    drop(corn);
    let exam: HashSet<(String, String)> = std::fs::read_to_string(data.join("school/dialogue.tsv")).unwrap_or_default()
        .lines().take(1200).filter_map(|l| { let p: Vec<&str> = l.split('\t').collect(); (p.len() >= 3).then(|| (p[1].to_string(), p[2].to_string())) }).collect();
    let trip = std::fs::read_to_string(data.join("triples.tsv")).expect("triples.tsv");
    let (before, mut skipped) = (c.starts.len(), 0);
    {
        let mut seen_turn: HashSet<&str> = HashSet::new();
        for l in trip.lines() {
            let p: Vec<&str> = l.split('\t').collect();
            if p.len() < 3 { continue; }
            if exam.contains(&(p[1].to_string(), p[2].to_string())) { skipped += 1; continue; }
            for t in &p[..3] { if !t.is_empty() && seen_turn.insert(t) { c.push(words(t)); } }
        }
    }
    n_src.push(("dialogue turns", c.starts.len() - before));
    drop(trip);
    let dl = data.join("school/downloads");
    let before = c.starts.len();
    let mut pages = Vec::new();
    files(&dl.join("reference"), "html", &mut pages);
    files(&dl.join("nomicon"), "html", &mut pages);
    pages.sort();
    for p in &pages {
        let t = html_text(&String::from_utf8_lossy(&std::fs::read(p).expect("page")));
        for s in t.split(['.', '\n', '?', '!']) { let w = words(s); if w.len() > 1 { c.push(w); } }
    }
    let mut md = Vec::new();
    files(&dl.join("rust-by-example"), "md", &mut md);
    md.sort();
    for p in &md {
        let t = String::from_utf8_lossy(&std::fs::read(p).expect("md")).into_owned();
        for s in t.split(['.', '\n', '?', '!']) { let w = words(s); if w.len() > 1 { c.push(w); } }
    }
    n_src.push(("rust book sentences", c.starts.len() - before));
    for (k, n) in &n_src { println!("  read {k:<20} {n:>9}"); }
    println!("  {} sentences, {} tokens, {} distinct ({} dialogue exam exchanges skipped) in {:.1}s, peak {:.2} GB",
        c.starts.len(), c.toks.len(), c.id.len(), skipped, t0.elapsed().as_secs_f32(), peak_gb());
    ram_guard("after reading");

    // ids by frequency, so the rarest fall past the 20-bit cap (they share one id: "<rare>")
    let mut order: Vec<u32> = (0..c.id.len() as u32).collect();
    let mut name_of: Vec<&str> = vec![""; c.id.len()];
    for (w, &i) in &c.id { name_of[i as usize] = w.as_str(); }
    order.sort_by(|&x, &y| c.count[y as usize].cmp(&c.count[x as usize]).then(name_of[x as usize].cmp(name_of[y as usize])));
    let mut rank = vec![0u32; c.id.len()];
    for (r, &i) in order.iter().enumerate() { rank[i as usize] = (r as u32).min(CTX_CAP); }
    let n_names = (c.id.len() as u32).min(CTX_CAP + 1) as usize;
    let mut names: Vec<String> = order.iter().take(n_names).map(|&i| name_of[i as usize].to_string()).collect();
    if c.id.len() > CTX_CAP as usize { names[CTX_CAP as usize] = "<rare>".into(); }
    for t in c.toks.iter_mut() { *t = rank[*t as usize]; }
    let gid: HashMap<String, u32> = names.iter().enumerate().map(|(i, w)| (w.clone(), i as u32)).collect();
    drop(order); drop(rank); drop(name_of);
    let id_len = c.id.len();
    c.id = HashMap::new();
    println!("  vocabulary {} words ({} past the 20-bit cap share one id)", id_len, id_len.saturating_sub(CTX_CAP as usize));

    // ---------------------------------------------------------------- the bench vocabulary (word list only)
    let mut want: Vec<String> = Vec::new();
    for (rel, tag, dom) in BLOCKS {
        let b = load_block(rel, tag, dom);
        let mut add = |s: &str| {
            for w in words(s) { want.push(w); }
            for w in talk_tokenize(s) { let w = w.to_lowercase(); if w.chars().any(|c| c.is_alphanumeric()) { want.push(w); } }
        };
        for l in &b.lessons { add(&l.q_raw); add(&l.a_raw); }
        for q in &b.qs { add(&q.raw); }
    }
    for p in PROBES { want.push(p.to_string()); }
    want.sort(); want.dedup();
    let profiled: Vec<(String, u32)> = want.iter().filter_map(|w| gid.get(w).filter(|&&g| g < CTX_CAP).map(|&g| (w.clone(), g))).collect();
    println!("  bench vocabulary {} words, {} of them read at least once", want.len(), profiled.len());
    let mut bix = vec![u32::MAX; n_names];
    for (k, (_, g)) in profiled.iter().enumerate() { bix[*g as usize] = k as u32; }

    // ---------------------------------------------------------------- pass A: frames of the bench words
    let ns = c.starts.len();
    // (the window is the free function `window` below)
    let mut n_ev_bench = 0u64;
    for k in 0..ns {
        let s = c.sent(k);
        for i in 0..s.len() {
            if bix[s[i] as usize] != u32::MAX { n_ev_bench += (i.min(WINDOW) + (s.len() - 1 - i).min(WINDOW)) as u64; }
        }
    }
    let m_bits = (n_ev_bench * 16).next_power_of_two().clamp(1 << 20, 1 << 32);
    let mut bloom = Bloom { bits: vec![0u64; (m_bits / 64) as usize], mask: m_bits - 1 };
    println!("  {} frame events of bench words; first-exposure filter {} MB", n_ev_bench, m_bits / 8 / 1_000_000);
    ram_guard("after bloom");
    let mut maps: Vec<Map<u64, u32>> = (0..profiled.len()).map(|_| Map::default()).collect();
    let mut ev_w = vec![0u64; profiled.len()];
    for k in 0..ns {
        if k % 500_000 == 0 && k > 0 { ram_guard("pass A"); }
        let s = c.sent(k);
        for i in 0..s.len() {
            let b = bix[s[i] as usize];
            if b == u32::MAX { continue; }
            let (mp, ev, bl) = (&mut maps[b as usize], &mut ev_w[b as usize], &mut bloom);
            window(s, i, |key| {
                *ev += 1;
                if bl.seen(mix(key ^ (b as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))) { *mp.entry(key).or_default() += 1; }
            });
        }
    }
    drop(bloom);
    let kept: usize = maps.iter().map(|m| m.len()).sum();
    println!("  pass A: {} (word, frame) pairs heard twice or more, at {:.1}s, peak {:.2} GB", kept, t0.elapsed().as_secs_f32(), peak_gb());
    ram_guard("after pass A");

    // ---------------------------------------------------------------- pass B: how often each such frame occurs at all
    let mut cc: Map<u64, u32> = Map::default();
    for m in &maps { for k in m.keys() { cc.insert(*k, 0); } }
    let mut t_all = 0u64;
    for k in 0..ns {
        if k % 500_000 == 0 && k > 0 { ram_guard("pass B"); }
        let s = c.sent(k);
        for i in 0..s.len() {
            let (cc, t) = (&mut cc, &mut t_all);
            window(s, i, |key| { *t += 1; if let Some(v) = cc.get_mut(&key) { *v += 1; } });
        }
    }
    println!("  pass B: {} distinct frames counted over {} events at {:.1}s, peak {:.2} GB", cc.len(), t_all, t0.elapsed().as_secs_f32(), peak_gb());

    // ---------------------------------------------------------------- PMI, top 2,000 per side
    let tf = t_all as f64;
    let mut prof: Vec<[Vec<(u64, f32)>; 2]> = Vec::with_capacity(profiled.len());
    let mut sum: Vec<[f32; 2]> = Vec::with_capacity(profiled.len());
    for (b, m) in maps.iter().enumerate() {
        // (key, PMI, count): the pre-registered C2 keeps the 2,000 highest PMI per side; the
        // exploratory x keeps the 2,000 heard most often, among frames some other word fills too
        let mut cand: [Vec<(u64, f32, u32)>; 2] = [Vec::new(), Vec::new()];
        for (&k, &v) in m {
            let n = v as f64 + 1.0;
            let i = (n * tf / (ev_w[b] as f64 * cc[&k].max(1) as f64)).ln();
            if variant_x && cc[&k] as f64 <= n { continue; }
            if i > 0.0 { cand[(k >> 63) as usize].push((k, i as f32, v + 1)); }
        }
        let mut sm = [0f32; 2];
        let mut sides: [Vec<(u64, f32)>; 2] = [Vec::new(), Vec::new()];
        for (s, side) in cand.iter_mut().enumerate() {
            if variant_x { side.sort_by(|x, y| y.2.cmp(&x.2).then(y.1.total_cmp(&x.1)).then(x.0.cmp(&y.0))); }
            else { side.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0))); }
            side.truncate(TOP);
            sides[s] = side.iter().map(|x| (x.0, x.1)).collect();
            sides[s].sort_by_key(|x| x.0);
            sm[s] = sides[s].iter().map(|x| x.1).sum();
        }
        prof.push(sides);
        sum.push(sm);
    }
    drop(maps); drop(cc);
    let fr = Frames {
        ix: profiled.iter().enumerate().map(|(i, (w, _))| (w.clone(), i)).collect(),
        words: profiled.iter().map(|x| x.0.clone()).collect(),
        events: ev_w, names, prof, sum,
    };
    fr.save(&out).expect("write frames");
    println!("  saved {} profiles to {out}", fr.words.len());

    // ---------------------------------------------------------------- a look
    let pairs = [("begin", "start"), ("big", "large"), ("big", "small"), ("designed", "created"), ("designed", "invented"),
        ("input", "output"), ("mutable", "changeable"), ("mutable", "immutable"), ("define", "declare"), ("coined", "introduced"),
        ("wrote", "built"), ("keyword", "word"), ("happy", "glad"), ("car", "happy"), ("year", "when")];
    println!("  probe pairs (sim = sqrt(Lin_left x Lin_right)):");
    for (x, y) in pairs { println!("    {x:<10} ~ {y:<11} {:.3}", fr.sim(x, y)); }
    for w in ["designed", "mutable", "declare"] {
        if let Some(&i) = fr.ix.get(w) {
            for side in 0..2 {
                let mut v = fr.prof[i][side].clone();
                v.sort_by(|a, b| b.1.total_cmp(&a.1));
                println!("    {w} {} frames: {}", if side == 0 { "left " } else { "right" },
                    v.iter().take(6).map(|(k, x)| format!("[{}] {x:.1}", fr.show(*k))).collect::<Vec<_>>().join("  "));
            }
        }
    }
    println!("  frames_build done in {:.1}s, peak working set {:.2} GB", t0.elapsed().as_secs_f32(), peak_gb());
}

/// every frame of the word at position i: up to WINDOW places on each side
fn window<F: FnMut(u64)>(s: &[u32], i: usize, mut f: F) {
    for d in 1..=WINDOW {
        if i >= d {
            let j = i - d;
            let t1 = if d >= 2 { s[j + 1] as u64 + 1 } else { 0 };
            let t2 = if d >= 3 { s[j + 2] as u64 + 1 } else { 0 };
            f(ctx_key(0, d as u64, t1, t2, s[j] as u64));
        }
        if i + d < s.len() {
            let j = i + d;
            let t1 = if d >= 2 { s[i + 1] as u64 + 1 } else { 0 };
            let t2 = if d >= 3 { s[i + 2] as u64 + 1 } else { 0 };
            f(ctx_key(1, d as u64, t1, t2, s[j] as u64));
        }
    }
}
