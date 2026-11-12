// TESTING / DESIGN ONLY
//! arc_pack: ARC_Corpus's unrelated science sentences packed into topic documents, with dedup (V1 + A2).
//!
//! WHY. ARC_Corpus is 94% of her old reading, as 14.6M single sentences in no useful order, so an
//! earlier sentence almost never helps predict a later one and her memory gets no learning signal.
//! Here sentences that share a topic word sit next to each other in documents of a few hundred tokens.
//!
//! WORDS. `all_words`: runs of ASCII letters or digits, lowercased, hashed. `content_words`: the letter
//! runs of 3+ letters that are not in the small stop list below, as sorted unique hashes.
//!
//! PASS 1 (streaming). For every content word, the number of sentences that hold it (`df`). Also the
//! "before" coherence: for a fixed 1% sample of line numbers i (by `mix64(SAMPLE_SEED ^ i)`), whether
//! line i shares a content word with line i - 1.
//!
//! PASS 2 (streaming). Each sentence gets a topic key: its content word with the smallest df among those
//! with df >= `min_df` (ties: the smaller hash); the secondary key is the next one. When none reaches
//! `min_df`, the most common of its content words is the key (so it still joins a group); with no
//! content words at all the key is 0. The sentence goes to bucket `mix64(key) % buckets` (a binary
//! temp file), so every sentence of one key, and every exact duplicate, lands in the same bucket.
//!
//! PASS 3 (per bucket, two threads). In line order: an exact duplicate (same `all_words` sequence) is
//! dropped; a near duplicate is dropped when MinHash LSH (15 hashes of word pairs, 5 bands of 3) names
//! an earlier kept sentence of the bucket whose word-pair Jaccard similarity, computed exactly, is at
//! least `near_jaccard` (0.7). Sentences with fewer than 3 word pairs get only the exact check. So no
//! pair is ever compared unless LSH proposes it, and a false proposal is never dropped. Then the kept
//! sentences are sorted by (key, secondary key, line) and packed in that order (`Packer`): a document
//! is closed before a sentence when it would pass `max_tokens`; or when the key changes and the
//! document has `min_tokens`; or when it has `target_tokens` and the (key, secondary) pair changes.
//! Sentences are joined with one space. The "after" coherence uses the same 1% sample of line numbers:
//! a sampled sentence that is not first in its document, against the sentence before it.
//!
//! LIMITS. Near duplicates are found only inside a bucket; two near copies whose rarest topic word
//! differs (for example a typo in that very word) are not compared. The output order is by bucket
//! (topics are spread at random across the file); the mixer shuffles documents anyway.

use crate::bpe::Bpe;
use crate::docs::DocsWriter;
use crate::hash::{new_map, FastMap, FastSet};
use crate::input::InputSpec;
use crate::numbers::insert_marks;
use crate::rng::mix64;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

/// Common English function words (only words of 3+ letters matter: shorter ones are never content).
pub const STOPWORDS: [&str; 120] = [
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was", "one", "our", "out", "has", "him",
    "his", "how", "its", "may", "new", "now", "old", "see", "two", "way", "who", "did", "get", "let", "say", "she", "too", "use",
    "used", "uses", "using", "that", "with", "have", "this", "will", "your", "from", "they", "been", "more", "when", "were", "what",
    "which", "their", "there", "these", "those", "them", "then", "than", "also", "into", "only", "other", "some", "such", "each",
    "very", "most", "many", "much", "over", "about", "after", "before", "because", "while", "where", "would", "could", "should",
    "between", "through", "during", "under", "being", "both", "same", "make", "made", "like", "just", "well", "even", "does",
    "done", "here", "upon", "within", "without", "another", "every", "often", "called", "known", "include", "includes",
    "including", "example", "different", "usually", "first", "second", "number", "part", "form", "time", "help", "helps", "able",
];

pub const SAMPLE_SEED: u64 = 0x6172_635f_636f_6865;
pub const SAMPLE_PER_MILLE: u64 = 10;
const MH_K: usize = 15;
const MH_BANDS: usize = 5;
const MH_ROWS: usize = 3;
/// Most earlier sentences remembered per LSH band key.
const BAND_KEEP: usize = 4;

#[inline]
pub fn word_hash(w: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in w {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    mix64(h)
}

pub fn stop_set() -> FastSet<u64> {
    STOPWORDS.iter().map(|w| word_hash(w.as_bytes())).collect()
}

/// Content words: lowercase ASCII-letter runs of 3+ letters, not stop words; sorted, unique.
pub fn content_words(text: &str, stop: &FastSet<u64>, out: &mut Vec<u64>) {
    out.clear();
    let mut w = [0u8; 64];
    let mut n = 0usize;
    let mut long = false;
    let flush = |w: &[u8], n: usize, long: bool, out: &mut Vec<u64>| {
        if n >= 3 && !long {
            let h = word_hash(&w[..n]);
            if !stop.contains(&h) {
                out.push(h);
            }
        }
    };
    for &b in text.as_bytes() {
        if b.is_ascii_alphabetic() {
            if n < w.len() {
                w[n] = b.to_ascii_lowercase();
                n += 1;
            } else {
                long = true;
            }
        } else {
            flush(&w, n, long, out);
            n = 0;
            long = false;
        }
    }
    flush(&w, n, long, out);
    out.sort_unstable();
    out.dedup();
}

/// Every word: lowercase runs of ASCII letters or digits, hashed, in order.
pub fn all_words(text: &str, out: &mut Vec<u64>) {
    out.clear();
    let mut w: Vec<u8> = Vec::with_capacity(32);
    for &b in text.as_bytes() {
        if b.is_ascii_alphanumeric() {
            w.push(b.to_ascii_lowercase());
        } else if !w.is_empty() {
            out.push(word_hash(&w));
            w.clear();
        }
    }
    if !w.is_empty() {
        out.push(word_hash(&w));
    }
}

/// The exact-duplicate key: the word sequence (case, spacing and punctuation ignored).
pub fn exact_key(words: &[u64]) -> u64 {
    let mut h = 0x5eed_u64 ^ words.len() as u64;
    for &w in words {
        h = mix64(h ^ w);
    }
    h
}

/// Word pairs (2-shingles), sorted unique.
pub fn shingles(words: &[u64], out: &mut Vec<u64>) {
    out.clear();
    for p in words.windows(2) {
        out.push(mix64(p[0].rotate_left(17) ^ p[1]));
    }
    out.sort_unstable();
    out.dedup();
}

pub fn minhash(sh: &[u64]) -> [u64; MH_K] {
    let mut sig = [u64::MAX; MH_K];
    for &s in sh {
        for (k, m) in sig.iter_mut().enumerate() {
            let v = mix64(s ^ (0x9e37_79b9_7f4a_7c15u64.wrapping_mul(k as u64 + 1)));
            if v < *m {
                *m = v;
            }
        }
    }
    sig
}

pub fn band_keys(sig: &[u64; MH_K]) -> [u64; MH_BANDS] {
    let mut out = [0u64; MH_BANDS];
    for (b, o) in out.iter_mut().enumerate() {
        let mut h = b as u64 + 1;
        for r in 0..MH_ROWS {
            h = mix64(h ^ sig[b * MH_ROWS + r]);
        }
        *o = h;
    }
    out
}

/// Jaccard similarity of two sorted unique lists.
pub fn jaccard(a: &[u64], b: &[u64]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let (mut i, mut j, mut inter) = (0, 0, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
        }
    }
    inter as f64 / (a.len() + b.len() - inter) as f64
}

/// Whether two sorted lists share a value.
pub fn shares(a: &[u64], b: &[u64]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

/// (key, secondary key) from a sentence's content words and the df table (see the module docs).
pub fn topic_keys(cw: &[u64], df: &FastMap<u64, u32>, min_df: u32) -> (u64, u64) {
    let mut best: [(u32, u64); 2] = [(u32::MAX, 0); 2];
    let mut most: (u32, u64) = (0, 0);
    for &w in cw {
        let d = df.get(&w).copied().unwrap_or(0);
        if (d, std::cmp::Reverse(w)) > (most.0, std::cmp::Reverse(most.1)) || most.1 == 0 {
            most = (d, w);
        }
        if d >= min_df {
            let c = (d, w);
            if c < best[0] {
                best[1] = best[0];
                best[0] = c;
            } else if c < best[1] {
                best[1] = c;
            }
        }
    }
    if best[0].0 == u32::MAX {
        return (most.1, 0);
    }
    (best[0].1, if best[1].0 == u32::MAX { 0 } else { best[1].1 })
}

#[inline]
pub fn sampled(line: u64) -> bool {
    mix64(SAMPLE_SEED ^ line) % 1000 < SAMPLE_PER_MILLE
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackParams {
    pub min_tokens: usize,
    pub target_tokens: usize,
    pub max_tokens: usize,
}

impl Default for PackParams {
    fn default() -> Self {
        PackParams { min_tokens: 256, target_tokens: 512, max_tokens: 1024 }
    }
}

/// Packs sentences, fed in (key, secondary, line) order, into documents (lists of sentence ids).
pub struct Packer {
    p: PackParams,
    cur: Vec<usize>,
    cur_tokens: usize,
    last: (u64, u64),
}

impl Packer {
    pub fn new(p: PackParams) -> Packer {
        Packer { p, cur: Vec::new(), cur_tokens: 0, last: (0, 0) }
    }

    /// Add a sentence; returns the document closed before it, if any.
    pub fn push(&mut self, key: u64, sec: u64, tokens: usize, id: usize) -> Option<Vec<usize>> {
        let mut out = None;
        if !self.cur.is_empty() {
            let close = self.cur_tokens + tokens > self.p.max_tokens
                || (key != self.last.0 && self.cur_tokens >= self.p.min_tokens)
                || (self.cur_tokens >= self.p.target_tokens && (key, sec) != self.last);
            if close {
                out = Some(std::mem::take(&mut self.cur));
                self.cur_tokens = 0;
            }
        }
        self.cur.push(id);
        self.cur_tokens += tokens;
        self.last = (key, sec);
        out
    }

    pub fn finish(&mut self) -> Option<Vec<usize>> {
        self.cur_tokens = 0;
        if self.cur.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.cur))
        }
    }
}

#[derive(Clone, Debug)]
pub struct ArcParams {
    pub input: PathBuf,
    pub out_docs: PathBuf,
    pub tmp_dir: PathBuf,
    pub limit: Option<u64>,
    pub min_df: u32,
    pub buckets: usize,
    pub near_jaccard: f64,
    pub pack: PackParams,
    pub threads: usize,
}

#[derive(Clone, Debug, Default)]
pub struct ArcReport {
    pub lines: u64,
    pub bytes: u64,
    pub unique_words: usize,
    pub keyless: u64,
    pub fallback_keys: u64,
    pub before_pairs: u64,
    pub before_shared: u64,
    pub exact_drops: u64,
    pub near_drops: u64,
    pub near_proposals: u64,
    pub kept: u64,
    pub docs: u64,
    pub tokens: u64,
    pub after_pairs: u64,
    pub after_shared: u64,
    pub after_same_key: u64,
    /// Documents by token count: <256, 256-511, 512-767, 768-1023, 1024+.
    pub doc_sizes: [u64; 5],
    pub pass1_secs: f64,
    pub pass2_secs: f64,
    pub pass3_secs: f64,
}

impl ArcReport {
    pub fn text(&self) -> String {
        let pct = |a: u64, b: u64| 100.0 * a as f64 / b.max(1) as f64;
        let mut s = String::new();
        s.push_str(&format!("lines {}  text {:.1} MB  unique content words {}\n", self.lines, self.bytes as f64 / 1e6, self.unique_words));
        s.push_str(&format!("keys: {} sentences with no content word, {} with no word reaching min_df (keyed by their most common word)\n", self.keyless, self.fallback_keys));
        s.push_str(&format!(
            "dedup: exact {} ({:.2}%)  near {} ({:.2}%; {} LSH proposals checked)  kept {} ({:.2}%)\n",
            self.exact_drops,
            pct(self.exact_drops, self.lines),
            self.near_drops,
            pct(self.near_drops, self.lines),
            self.near_proposals,
            self.kept,
            pct(self.kept, self.lines)
        ));
        s.push_str(&format!(
            "docs {}  tokens {} (marks counted, no EOS)  mean {:.0} tokens / {:.1} sentences per doc\n",
            self.docs,
            self.tokens,
            self.tokens as f64 / self.docs.max(1) as f64,
            self.kept as f64 / self.docs.max(1) as f64
        ));
        s.push_str(&format!("doc sizes <256 / 256-511 / 512-767 / 768-1023 / 1024+: {:?}\n", self.doc_sizes));
        s.push_str(&format!(
            "coherence (adjacent sentences share a content word, 1% fixed sample): before {:.2}% of {}  after {:.2}% of {}  (after, same topic key: {:.2}%)\n",
            pct(self.before_shared, self.before_pairs),
            self.before_pairs,
            pct(self.after_shared, self.after_pairs),
            self.after_pairs,
            pct(self.after_same_key, self.after_pairs)
        ));
        let total = self.pass1_secs + self.pass2_secs + self.pass3_secs;
        s.push_str(&format!(
            "time: pass1 {:.1}s  pass2 {:.1}s  pass3 {:.1}s  total {:.1}s  ({:.0}k lines/s, {:.1} MB/s, {:.2}M tokens/s)\n",
            self.pass1_secs,
            self.pass2_secs,
            self.pass3_secs,
            total,
            self.lines as f64 / 1e3 / total.max(1e-9),
            self.bytes as f64 / 1e6 / total.max(1e-9),
            self.tokens as f64 / 1e6 / total.max(1e-9)
        ));
        s
    }
}

fn open_lines(path: &Path) -> io::Result<crate::input::DocReader<io::BufReader<File>>> {
    let spec = InputSpec::parse(&path.to_string_lossy()).map_err(io::Error::other)?;
    spec.open()
}

/// Pass 1: df table and the "before" coherence.
pub fn pass1(p: &ArcParams, rep: &mut ArcReport) -> io::Result<FastMap<u64, u32>> {
    let t0 = Instant::now();
    let stop = stop_set();
    let mut df: FastMap<u64, u32> = new_map();
    let mut r = open_lines(&p.input)?;
    let (mut cw, mut prev) = (Vec::new(), Vec::new());
    let mut line = 0u64;
    while let Some(doc) = r.next_doc()? {
        if p.limit.is_some_and(|l| line >= l) {
            break;
        }
        content_words(&doc, &stop, &mut cw);
        for &w in &cw {
            *df.entry(w).or_insert(0) += 1;
        }
        if line > 0 && sampled(line) {
            rep.before_pairs += 1;
            rep.before_shared += shares(&prev, &cw) as u64;
        }
        std::mem::swap(&mut prev, &mut cw);
        rep.bytes += doc.len() as u64;
        line += 1;
    }
    rep.lines = line;
    rep.unique_words = df.len();
    rep.pass1_secs = t0.elapsed().as_secs_f64();
    Ok(df)
}

fn bucket_path(dir: &Path, b: usize) -> PathBuf {
    dir.join(format!("bucket_{b:04}.bin"))
}

/// Pass 2: every sentence to its bucket file (u64 key, u64 secondary, u32 line, u32 length, text).
pub fn pass2(p: &ArcParams, df: &FastMap<u64, u32>, rep: &mut ArcReport) -> io::Result<()> {
    let t0 = Instant::now();
    fs::create_dir_all(&p.tmp_dir)?;
    let stop = stop_set();
    let mut ws: Vec<BufWriter<File>> = Vec::with_capacity(p.buckets);
    for b in 0..p.buckets {
        ws.push(BufWriter::with_capacity(1 << 16, File::create(bucket_path(&p.tmp_dir, b))?));
    }
    let mut r = open_lines(&p.input)?;
    let mut cw = Vec::new();
    let mut line = 0u64;
    while let Some(doc) = r.next_doc()? {
        if p.limit.is_some_and(|l| line >= l) {
            break;
        }
        content_words(&doc, &stop, &mut cw);
        let (key, sec) = topic_keys(&cw, df, p.min_df);
        if cw.is_empty() {
            rep.keyless += 1;
        } else if !cw.iter().any(|w| df.get(w).copied().unwrap_or(0) >= p.min_df) {
            rep.fallback_keys += 1;
        }
        let b = (mix64(key) % p.buckets as u64) as usize;
        let w = &mut ws[b];
        w.write_all(&key.to_le_bytes())?;
        w.write_all(&sec.to_le_bytes())?;
        w.write_all(&(line as u32).to_le_bytes())?;
        w.write_all(&(doc.len() as u32).to_le_bytes())?;
        w.write_all(doc.as_bytes())?;
        line += 1;
    }
    for mut w in ws {
        w.flush()?;
    }
    rep.pass2_secs = t0.elapsed().as_secs_f64();
    Ok(())
}

struct Rec {
    key: u64,
    sec: u64,
    line: u32,
    text: String,
}

fn read_bucket(path: &Path) -> io::Result<Vec<Rec>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let mut out = Vec::new();
    let mut i = 0;
    while i + 24 <= bytes.len() {
        let key = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        let sec = u64::from_le_bytes(bytes[i + 8..i + 16].try_into().unwrap());
        let line = u32::from_le_bytes(bytes[i + 16..i + 20].try_into().unwrap());
        let len = u32::from_le_bytes(bytes[i + 20..i + 24].try_into().unwrap()) as usize;
        let text = String::from_utf8_lossy(&bytes[i + 24..i + 24 + len]).into_owned();
        out.push(Rec { key, sec, line, text });
        i += 24 + len;
    }
    Ok(out)
}

#[derive(Default)]
struct BucketOut {
    docs: Vec<String>,
    exact: u64,
    near: u64,
    proposals: u64,
    kept: u64,
    tokens: u64,
    sizes: [u64; 5],
    after_pairs: u64,
    after_shared: u64,
    after_same_key: u64,
}

fn size_bin(t: usize) -> usize {
    match t {
        0..=255 => 0,
        256..=511 => 1,
        512..=767 => 2,
        768..=1023 => 3,
        _ => 4,
    }
}

/// Dedup, sort and pack one bucket.
fn do_bucket(recs: Vec<Rec>, bpe: &Bpe, p: &ArcParams, stop: &FastSet<u64>) -> BucketOut {
    let mut o = BucketOut::default();
    let mut exact: FastSet<u64> = FastSet::default();
    let mut bands: FastMap<u64, Vec<u32>> = new_map();
    let mut kept: Vec<Rec> = Vec::with_capacity(recs.len());
    let mut kept_sh: Vec<Vec<u64>> = Vec::with_capacity(recs.len());
    let (mut words, mut sh) = (Vec::new(), Vec::new());
    for r in recs {
        all_words(&r.text, &mut words);
        if !exact.insert(exact_key(&words)) {
            o.exact += 1;
            continue;
        }
        shingles(&words, &mut sh);
        if sh.len() >= 3 {
            let keys = band_keys(&minhash(&sh));
            let mut dup = false;
            'bands: for k in keys {
                if let Some(c) = bands.get(&k) {
                    for &j in c {
                        o.proposals += 1;
                        if jaccard(&sh, &kept_sh[j as usize]) >= p.near_jaccard {
                            dup = true;
                            break 'bands;
                        }
                    }
                }
            }
            if dup {
                o.near += 1;
                continue;
            }
            let id = kept.len() as u32;
            for k in keys {
                let c = bands.entry(k).or_default();
                if c.len() < BAND_KEEP {
                    c.push(id);
                }
            }
        }
        kept_sh.push(std::mem::take(&mut sh));
        kept.push(r);
    }
    drop(kept_sh);
    drop(bands);
    o.kept = kept.len() as u64;
    kept.sort_by_key(|r| (r.key, r.sec, r.line));
    let toks: Vec<usize> = kept.iter().map(|r| insert_marks(&bpe.encode(&r.text)).len()).collect();
    let mut packer = Packer::new(p.pack);
    let mut cw_prev = Vec::new();
    let mut cw = Vec::new();
    let mut emit = |ids: Vec<usize>, o: &mut BucketOut| {
        let t: usize = ids.iter().map(|&i| toks[i]).sum();
        o.tokens += t as u64;
        o.sizes[size_bin(t)] += 1;
        let mut text = String::new();
        for (n, &i) in ids.iter().enumerate() {
            if n > 0 {
                text.push(' ');
            }
            text.push_str(&kept[i].text);
            // after-coherence on the fixed sample
            if n > 0 && sampled(kept[i].line as u64) {
                content_words(&kept[ids[n - 1]].text, stop, &mut cw_prev);
                content_words(&kept[i].text, stop, &mut cw);
                o.after_pairs += 1;
                o.after_shared += shares(&cw_prev, &cw) as u64;
                o.after_same_key += (kept[i].key == kept[ids[n - 1]].key) as u64;
            }
        }
        o.docs.push(text);
    };
    for i in 0..kept.len() {
        if let Some(ids) = packer.push(kept[i].key, kept[i].sec, toks[i], i) {
            emit(ids, &mut o);
        }
    }
    if let Some(ids) = packer.finish() {
        emit(ids, &mut o);
    }
    o
}

/// Pass 3: buckets on `threads` workers, written in bucket order.
pub fn pass3(p: &ArcParams, bpe: &Bpe, rep: &mut ArcReport) -> io::Result<()> {
    let t0 = Instant::now();
    let mut w = DocsWriter::create(&p.out_docs)?;
    let threads = p.threads.max(1);
    let (tx, rx) = mpsc::sync_channel::<(usize, io::Result<BucketOut>)>(threads * 2);
    std::thread::scope(|s| -> io::Result<()> {
        for t in 0..threads {
            let tx = tx.clone();
            let bpe = bpe.clone();
            s.spawn(move || {
                let stop = stop_set();
                let mut b = t;
                while b < p.buckets {
                    let res = read_bucket(&bucket_path(&p.tmp_dir, b)).map(|recs| do_bucket(recs, &bpe, p, &stop));
                    if tx.send((b, res)).is_err() {
                        return;
                    }
                    b += threads;
                }
            });
        }
        drop(tx);
        let mut pending: BTreeMap<usize, BucketOut> = BTreeMap::new();
        let mut next = 0usize;
        for (b, res) in rx {
            pending.insert(b, res?);
            while let Some(o) = pending.remove(&next) {
                for d in &o.docs {
                    w.write_doc(d)?;
                }
                rep.exact_drops += o.exact;
                rep.near_drops += o.near;
                rep.near_proposals += o.proposals;
                rep.kept += o.kept;
                rep.docs += o.docs.len() as u64;
                rep.tokens += o.tokens;
                for k in 0..5 {
                    rep.doc_sizes[k] += o.sizes[k];
                }
                rep.after_pairs += o.after_pairs;
                rep.after_shared += o.after_shared;
                rep.after_same_key += o.after_same_key;
                next += 1;
            }
        }
        Ok(())
    })?;
    w.finish()?;
    rep.pass3_secs = t0.elapsed().as_secs_f64();
    Ok(())
}

/// All three passes; the bucket files are removed at the end.
pub fn pack_corpus(p: &ArcParams, bpe: &Bpe) -> io::Result<ArcReport> {
    let mut rep = ArcReport::default();
    let df = pass1(p, &mut rep)?;
    pass2(p, &df, &mut rep)?;
    drop(df);
    pass3(p, bpe, &mut rep)?;
    for b in 0..p.buckets {
        let _ = fs::remove_file(bucket_path(&p.tmp_dir, b));
    }
    let _ = fs::remove_dir(&p.tmp_dir);
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_words_skip_stop_and_short_words() {
        let stop = stop_set();
        let mut a = Vec::new();
        content_words("The Sun is a STAR; the sun, which is hot.", &stop, &mut a);
        let mut want: Vec<u64> = ["sun", "star", "hot"].iter().map(|w| word_hash(w.as_bytes())).collect();
        want.sort_unstable();
        assert_eq!(a, want);
    }

    #[test]
    fn exact_key_ignores_case_and_punctuation() {
        let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        all_words("Water boils at 100 C.", &mut a);
        all_words("water  boils at 100 c", &mut b);
        all_words("Water boils at 90 C.", &mut c);
        assert_eq!(exact_key(&a), exact_key(&b));
        assert_ne!(exact_key(&a), exact_key(&c));
    }

    #[test]
    fn jaccard_and_shares() {
        assert_eq!(jaccard(&[1, 2, 3], &[2, 3, 4]), 0.5);
        assert_eq!(jaccard(&[1], &[1]), 1.0);
        assert!(shares(&[1, 5, 9], &[2, 9]));
        assert!(!shares(&[1, 5], &[2, 9]));
    }

    #[test]
    fn minhash_agrees_with_similarity() {
        let mk = |s: &str| {
            let (mut w, mut sh) = (Vec::new(), Vec::new());
            all_words(s, &mut w);
            shingles(&w, &mut sh);
            sh
        };
        let a = mk("the mitochondria is the powerhouse of the cell and makes energy for the cell");
        let b = mk("the mitochondria is the powerhouse of the cell and makes energy for each cell");
        let c = mk("a volcano erupts when magma rises through cracks in the crust of the earth");
        let (ka, kb, kc) = (band_keys(&minhash(&a)), band_keys(&minhash(&b)), band_keys(&minhash(&c)));
        assert!(jaccard(&a, &b) >= 0.7);
        assert!(ka.iter().zip(kb.iter()).any(|(x, y)| x == y), "near copies share a band");
        assert!(ka.iter().zip(kc.iter()).all(|(x, y)| x != y), "unrelated sentences share none");
    }

    #[test]
    fn topic_keys_pick_the_rarest_common_enough_word() {
        let df: FastMap<u64, u32> = [(1u64, 100u32), (2, 5), (3, 1), (4, 40)].into_iter().collect();
        assert_eq!(topic_keys(&[1, 2, 3, 4], &df, 2), (2, 4));
        assert_eq!(topic_keys(&[1, 2, 3, 4], &df, 10), (4, 1));
        // none reaches min_df: the most common word
        assert_eq!(topic_keys(&[2, 3], &df, 50), (2, 0));
        assert_eq!(topic_keys(&[], &df, 2), (0, 0));
    }

    #[test]
    fn packer_respects_bounds_and_topic_changes() {
        let p = PackParams { min_tokens: 20, target_tokens: 40, max_tokens: 60 };
        let mut k = Packer::new(p);
        let mut docs = Vec::new();
        // key 1: five sentences of 10 tokens; key 2: one of 10; key 3: three of 25
        let feed = [(1, 0, 10), (1, 0, 10), (1, 0, 10), (1, 0, 10), (1, 1, 10), (2, 0, 10), (3, 0, 25), (3, 0, 25), (3, 0, 25)];
        for (i, &(key, sec, t)) in feed.iter().enumerate() {
            if let Some(d) = k.push(key, sec, t, i) {
                docs.push(d);
            }
        }
        docs.extend(k.finish());
        // key 1 reaches the target (40) then the secondary key changes -> close; the 10-token tail
        // is under min_tokens, so key 2 joins it; at key 3 the doc has 20 = min -> close; key 3's
        // third sentence would pass 60 -> close
        assert_eq!(docs, vec![vec![0, 1, 2, 3], vec![4, 5], vec![6, 7], vec![8]]);
        for d in &docs {
            assert!(d.iter().map(|&i| feed[i].2).sum::<usize>() <= 60);
        }
    }

    #[test]
    fn end_to_end_on_a_small_corpus() {
        let dir = std::env::temp_dir().join(format!("ava_arcpack_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let input = dir.join("corpus.txt");
        let mut body = String::new();
        let topics = ["volcano magma lava", "photosynthesis chlorophyll leaves", "planet orbit gravity"];
        for i in 0..60 {
            let t = topics[i % 3];
            body.push_str(&format!("Sentence {i} about {t} number {}.\n", i * 7));
        }
        body.push_str("Sentence 0 about volcano magma lava number 0.\n"); // exact copy of line 0
        body.push_str("sentence 3 about volcano magma lava number 21\n"); // exact up to case/punctuation
        fs::write(&input, body).unwrap();
        let bpe = Bpe::from_merges(Vec::new()).unwrap();
        let p = ArcParams {
            input,
            out_docs: dir.join("out.docs"),
            tmp_dir: dir.join("tmp"),
            limit: None,
            min_df: 2,
            buckets: 4,
            near_jaccard: 0.7,
            pack: PackParams { min_tokens: 50, target_tokens: 200, max_tokens: 400 },
            threads: 2,
        };
        let rep = pack_corpus(&p, &bpe).unwrap();
        assert_eq!(rep.lines, 62);
        assert_eq!(rep.exact_drops, 2);
        assert_eq!(rep.kept + rep.exact_drops + rep.near_drops, 62);
        let docs = crate::docs::read_all_docs(&p.out_docs).unwrap();
        assert_eq!(docs.len() as u64, rep.docs);
        // every kept sentence appears exactly once, and each document holds one topic
        let joined = docs.join(" ");
        assert_eq!(joined.matches("Sentence").count() as u64 + joined.matches("sentence").count() as u64, rep.kept);
        for d in &docs {
            let n = topics.iter().filter(|t| d.contains(*t)).count();
            assert_eq!(n, 1, "one topic per document: {d}");
        }
        assert!(rep.after_pairs == 0 || rep.after_shared == rep.after_pairs);
        assert!(!p.tmp_dir.exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
