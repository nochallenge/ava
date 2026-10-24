//! TESTING / DESIGN ONLY — byte-level BPE: pre-tokenizer, trainer, encoder, decoder, save/load.
//!
//! VOCABULARY
//!   ids 0..=255  the raw bytes, so every byte string is encodable and nothing is ever unknown
//!   id  256      EOS, the document separator (`encode_doc` appends it)
//!   id  257      PAD (never produced by the encoder; pads the last sequential eval window)
//!   ids 258..    learned merges in the order they were learned: id = 258 + rank
//!   `vocab_size()` = 258 + number of merges.
//!
//! PRE-TOKENIZER (generic character classes only, no word lists). Every char is one of
//!   L  letter      `char::is_alphabetic()`   (the apostrophe ' is NOT a letter)
//!   D  digit       `char::is_ascii_digit()`  (ASCII 0-9 only)
//!   W  whitespace  `char::is_whitespace()`
//!   P  other       everything else: punctuation, symbols, emoji, combining marks,
//!                  non-ASCII digits
//! The text is cut left to right into chunks; at each position the first rule that applies:
//!   1. ' ' L+   one ASCII space (U+0020) directly followed by a maximal run of letters
//!   2. ' ' P+   one ASCII space directly followed by a maximal run of other chars
//!   3. D        one ASCII digit, always alone (digits never merge with each other or with
//!               anything else, so numbers are spelt digit by digit: arithmetic stays learnable)
//!   4. L+       a maximal run of letters
//!   5. P+       a maximal run of other chars
//!   6. W+       a maximal run of whitespace, except that when the run is longer than one char,
//!               ends in ' ' and is followed by L or P, its last ' ' is left over to start rule
//!               1 or 2 for the next chunk ("a   b" -> "a", "  ", " b")
//! Only U+0020 can be a leading space (tab, newline, NBSP never attach), digits never take a
//! leading space (" 7" -> " ", "7"), and merges never cross a chunk boundary.
//!
//! TRAINING works on a chunk -> frequency table (counted while streaming a corpus sample).
//! Pair counts are kept incrementally: a merge only revisits the words that contain the pair
//! (found through a per-pair list of word indices, lazily validated), and the best pair comes
//! from a lazy max-heap whose entries are re-checked against the live count when popped.
//! Ties: higher count first, then the smaller (left_id, right_id). Overlapping occurrences
//! ("aaa" holds (a,a) twice) are all counted; applying a merge replaces non-overlapping
//! occurrences left to right. Fully deterministic (the table is sorted first).
//!
//! ENCODING pre-tokenizes, then for each chunk repeatedly merges the adjacent pair with the
//! lowest merge rank (all its occurrences, left to right) until no learned pair is left. This
//! gives exactly the tokens that applying every merge in learned order would give. A bounded
//! chunk -> ids cache (default 1M entries, cleared when full) makes repeated words cheap.
//!
//! DECODING concatenates token bytes and converts with `from_utf8_lossy`. EOS and PAD render
//! as NOTHING in `decode` (so `decode(encode_doc(s)) == s`); `decode_marked` renders them as
//! `<|eos|>` / `<|pad|>` for display. Ids outside the vocabulary render as U+FFFD.
//!
//! FILE FORMAT (plain text): first line `ava-bpe v1 <vocab_size>`, then one merge per line,
//! `left right` (decimal ids), in rank order.

use crate::hash::{new_map, FastMap};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

pub const EOS: u32 = 256;
pub const PAD: u32 = 257;
pub const FIRST_MERGE_ID: u32 = 258;
/// Default bound on the encoder's chunk cache (entries); the cache is cleared when full.
pub const DEFAULT_CACHE_CAP: usize = 1 << 20;
/// Chunks longer than this many bytes are encoded but never cached.
const CACHE_MAX_CHUNK_BYTES: usize = 256;
const FILE_MAGIC: &str = "ava-bpe";
const FILE_VERSION: &str = "v1";

// ---------------------------------------------------------------- pre-tokenizer

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Letter,
    Digit,
    Space,
    Other,
}

#[inline]
fn class_of(c: char) -> Class {
    if c.is_ascii() {
        let b = c as u8;
        if b.is_ascii_alphabetic() {
            Class::Letter
        } else if b.is_ascii_digit() {
            Class::Digit
        } else if c.is_whitespace() {
            Class::Space
        } else {
            Class::Other
        }
    } else if c.is_alphabetic() {
        Class::Letter
    } else if c.is_whitespace() {
        Class::Space
    } else {
        Class::Other
    }
}

/// The char starting at byte `i` (which must be a char boundary) and its byte length.
#[inline]
fn char_at(text: &str, i: usize) -> Option<(char, usize)> {
    let b = *text.as_bytes().get(i)?;
    if b < 0x80 {
        return Some((b as char, 1));
    }
    let c = text[i..].chars().next()?;
    Some((c, c.len_utf8()))
}

/// End (byte index) of the maximal run of class `cl` starting at byte `i`.
#[inline]
fn run_end(text: &str, mut i: usize, cl: Class) -> usize {
    while let Some((c, n)) = char_at(text, i) {
        if class_of(c) != cl {
            break;
        }
        i += n;
    }
    i
}

#[inline]
fn is_word_like(cl: Class) -> bool {
    matches!(cl, Class::Letter | Class::Other)
}

/// Call `f` on every chunk of `text`, in order (rules in the module docs). The chunks
/// concatenate back to `text` exactly.
pub fn for_each_chunk<'a>(text: &'a str, mut f: impl FnMut(&'a str)) {
    let n = text.len();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < n {
        let (c, clen) = char_at(text, i).expect("char boundary");
        let start = i;
        let cl = class_of(c);
        i = match cl {
            Class::Digit => i + 1,
            Class::Letter | Class::Other => run_end(text, i + clen, cl),
            Class::Space => {
                // rules 1 and 2: a single ' ' leading a letter run or an other-run
                let lead = if c == ' ' {
                    char_at(text, i + 1).map(|(c2, _)| class_of(c2)).filter(|&k| is_word_like(k))
                } else {
                    None
                };
                match lead {
                    Some(k) => run_end(text, i + 1, k),
                    None => {
                        // rule 6: whitespace run, giving back a final ' ' that leads a word
                        let mut j = run_end(text, i + clen, Class::Space);
                        if j < n && j - 1 > start && bytes[j - 1] == b' ' {
                            let (c3, _) = char_at(text, j).expect("char boundary");
                            if is_word_like(class_of(c3)) {
                                j -= 1;
                            }
                        }
                        j
                    }
                }
            }
        };
        f(&text[start..i]);
    }
}

/// The chunks of `text` as a vector (see `for_each_chunk`).
pub fn pre_tokenize(text: &str) -> Vec<&str> {
    let mut v = Vec::new();
    for_each_chunk(text, |c| v.push(c));
    v
}

// ---------------------------------------------------------------- chunk counting

/// Chunk -> frequency table, filled while streaming a corpus. Memory grows with the number of
/// unique chunks, so callers bound it by bounding the sample (see the `bpe_train` binary).
#[derive(Default)]
pub struct ChunkCounter {
    map: FastMap<Box<[u8]>, u64>,
    pub total_bytes: u64,
    pub total_chunks: u64,
}

impl ChunkCounter {
    pub fn new() -> ChunkCounter {
        ChunkCounter::default()
    }

    pub fn add_text(&mut self, text: &str) {
        self.total_bytes += text.len() as u64;
        let map = &mut self.map;
        let mut n = 0u64;
        for_each_chunk(text, |chunk| {
            n += 1;
            let b = chunk.as_bytes();
            if let Some(v) = map.get_mut(b) {
                *v += 1;
            } else {
                map.insert(Box::from(b), 1);
            }
        });
        self.total_chunks += n;
    }

    /// Number of distinct chunks seen.
    pub fn unique(&self) -> usize {
        self.map.len()
    }

    pub fn into_counts(self) -> impl Iterator<Item = (Vec<u8>, u64)> {
        self.map.into_iter().map(|(k, v)| (k.into_vec(), v))
    }
}

// ---------------------------------------------------------------- the model

#[inline]
fn pair_key(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

#[inline]
fn unkey(k: u64) -> (u32, u32) {
    ((k >> 32) as u32, k as u32)
}

/// Replace every non-overlapping occurrence of (a, b), left to right, by `new_id`.
#[inline]
fn merge_in_place(ids: &mut Vec<u32>, a: u32, b: u32, new_id: u32) {
    let n = ids.len();
    let (mut r, mut w) = (0usize, 0usize);
    while r < n {
        if r + 1 < n && ids[r] == a && ids[r + 1] == b {
            ids[w] = new_id;
            r += 2;
        } else {
            ids[w] = ids[r];
            r += 1;
        }
        w += 1;
    }
    ids.truncate(w);
}

pub struct Bpe {
    merges: Vec<(u32, u32)>,
    /// pair key -> the id it merges into (id order == rank order)
    ranks: FastMap<u64, u32>,
    /// id -> bytes (EOS and PAD are empty)
    token_bytes: Vec<Vec<u8>>,
    cache: Mutex<FastMap<Box<[u8]>, Box<[u32]>>>,
    cache_cap: usize,
}

impl Clone for Bpe {
    /// Clones the vocabulary; the clone starts with an empty cache.
    fn clone(&self) -> Bpe {
        Bpe {
            merges: self.merges.clone(),
            ranks: self.ranks.clone(),
            token_bytes: self.token_bytes.clone(),
            cache: Mutex::new(new_map()),
            cache_cap: self.cache_cap,
        }
    }
}

impl Bpe {
    /// Build a tokenizer from merges in rank order. Merge `i` makes id `258 + i` and may only
    /// use byte ids or earlier merge ids (never EOS/PAD); a pair may appear only once.
    pub fn from_merges(merges: Vec<(u32, u32)>) -> Result<Bpe, String> {
        let mut token_bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        token_bytes.push(Vec::new()); // EOS
        token_bytes.push(Vec::new()); // PAD
        let mut ranks: FastMap<u64, u32> = new_map();
        for (i, &(l, r)) in merges.iter().enumerate() {
            let id = FIRST_MERGE_ID + i as u32;
            for x in [l, r] {
                if x >= id || x == EOS || x == PAD {
                    return Err(format!("merge {i} ({l} {r}) uses id {x}, which is special or not yet defined"));
                }
            }
            if ranks.insert(pair_key(l, r), id).is_some() {
                return Err(format!("merge {i} ({l} {r}) is a duplicate"));
            }
            let mut b = token_bytes[l as usize].clone();
            b.extend_from_slice(&token_bytes[r as usize]);
            token_bytes.push(b);
        }
        Ok(Bpe { merges, ranks, token_bytes, cache: Mutex::new(new_map()), cache_cap: DEFAULT_CACHE_CAP })
    }

    /// Learn up to `vocab_size - 258` merges from a chunk-frequency table (duplicate chunks
    /// are summed). Stops early when no pair is left, so the result can be smaller.
    pub fn train(chunk_counts: impl Iterator<Item = (Vec<u8>, u64)>, vocab_size: usize) -> Bpe {
        let target = vocab_size.saturating_sub(FIRST_MERGE_ID as usize);

        // 1. words that can hold a merge, sorted (deterministic indices), duplicates summed
        let mut items: Vec<(Vec<u8>, u64)> = chunk_counts.filter(|(c, n)| c.len() >= 2 && *n > 0).collect();
        items.sort_unstable_by(|x, y| x.0.cmp(&y.0));
        let mut words: Vec<Vec<u32>> = Vec::with_capacity(items.len());
        let mut counts: Vec<i64> = Vec::with_capacity(items.len());
        let mut last: Option<Vec<u8>> = None;
        for (c, n) in items {
            if last.as_deref() == Some(c.as_slice()) {
                *counts.last_mut().unwrap() += n as i64;
            } else {
                words.push(c.iter().map(|&b| b as u32).collect());
                counts.push(n as i64);
                last = Some(c);
            }
        }
        drop(last);

        // 2. initial pair counts and where each pair occurs
        let mut pair_counts: FastMap<u64, i64> = new_map();
        let mut where_: FastMap<u64, Vec<u32>> = new_map();
        for (wi, w) in words.iter().enumerate() {
            let c = counts[wi];
            for p in w.windows(2) {
                let k = pair_key(p[0], p[1]);
                *pair_counts.entry(k).or_insert(0) += c;
                let v = where_.entry(k).or_default();
                if v.last() != Some(&(wi as u32)) {
                    v.push(wi as u32);
                }
            }
        }
        let mut heap: BinaryHeap<(i64, Reverse<u64>)> = pair_counts.iter().map(|(&k, &c)| (c, Reverse(k))).collect();

        // 3. merge loop
        let mut stamp: Vec<u32> = vec![u32::MAX; words.len()];
        let mut merges: Vec<(u32, u32)> = Vec::with_capacity(target);
        let mut touched: Vec<u64> = Vec::new();
        while merges.len() < target {
            let Some((c, Reverse(k))) = heap.pop() else { break };
            let live = pair_counts.get(&k).copied().unwrap_or(0);
            if live != c {
                // stale entry (the pair lost occurrences since it was pushed)
                if live > 0 {
                    heap.push((live, Reverse(k)));
                }
                continue;
            }
            if c <= 0 {
                continue;
            }
            let (a, b) = unkey(k);
            let m = merges.len() as u32;
            let new_id = FIRST_MERGE_ID + m;
            merges.push((a, b));
            let list = where_.remove(&k).unwrap_or_default();
            touched.clear();
            for &wi in &list {
                let wi = wi as usize;
                if stamp[wi] == m {
                    continue;
                }
                stamp[wi] = m;
                let w = &mut words[wi];
                if !w.windows(2).any(|p| p[0] == a && p[1] == b) {
                    continue; // stale index entry
                }
                let cnt = counts[wi];
                for p in w.windows(2) {
                    *pair_counts.get_mut(&pair_key(p[0], p[1])).expect("counted pair") -= cnt;
                }
                merge_in_place(w, a, b, new_id);
                for i in 0..w.len().saturating_sub(1) {
                    let (x, y) = (w[i], w[i + 1]);
                    let kk = pair_key(x, y);
                    *pair_counts.entry(kk).or_insert(0) += cnt;
                    if x == new_id || y == new_id {
                        // only pairs with the new token can be new or grow
                        let v = where_.entry(kk).or_default();
                        if v.last() != Some(&(wi as u32)) {
                            v.push(wi as u32);
                        }
                        touched.push(kk);
                    }
                }
            }
            debug_assert_eq!(pair_counts.get(&k).copied().unwrap_or(0), 0);
            pair_counts.remove(&k);
            touched.sort_unstable();
            touched.dedup();
            for &kk in &touched {
                let c = pair_counts[&kk];
                if c > 0 {
                    heap.push((c, Reverse(kk)));
                }
            }
        }
        Bpe::from_merges(merges).expect("trainer produced invalid merges")
    }

    /// 258 + number of merges (bytes, EOS, PAD and merges).
    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }

    pub fn merges(&self) -> &[(u32, u32)] {
        &self.merges
    }

    /// The bytes of token `id` (empty for EOS/PAD). Panics if `id >= vocab_size()`.
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        &self.token_bytes[id as usize]
    }

    /// Bound the chunk cache (entries). 0 disables caching.
    pub fn set_cache_capacity(&mut self, cap: usize) {
        self.cache_cap = cap;
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        c.clear();
        c.shrink_to_fit();
    }

    /// Apply merges by rank to one chunk; result in `ids`.
    fn merge_chunk(&self, chunk: &[u8], ids: &mut Vec<u32>) {
        ids.clear();
        ids.extend(chunk.iter().map(|&b| b as u32));
        if self.merges.is_empty() {
            return;
        }
        while ids.len() >= 2 {
            let mut best = u32::MAX;
            for p in ids.windows(2) {
                if let Some(&id) = self.ranks.get(&pair_key(p[0], p[1])) {
                    if id < best {
                        best = id;
                    }
                }
            }
            if best == u32::MAX {
                break;
            }
            let (a, b) = self.merges[(best - FIRST_MERGE_ID) as usize];
            merge_in_place(ids, a, b, best);
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        self.encode_into(text, &mut out);
        out
    }

    /// Append the tokens of `text` to `out`. The chunk cache is locked for the whole call, so
    /// threads sharing one `Bpe` take turns; for parallel encoding give each thread a clone
    /// (a clone has its own cache).
    pub fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        let mut guard = if self.cache_cap > 0 { Some(self.cache.lock().unwrap_or_else(|e| e.into_inner())) } else { None };
        let cap = self.cache_cap;
        let mut scratch: Vec<u32> = Vec::new();
        for_each_chunk(text, |chunk| {
            let b = chunk.as_bytes();
            if b.len() == 1 {
                out.push(b[0] as u32);
                return;
            }
            if let Some(cache) = guard.as_mut() {
                if let Some(ids) = cache.get(b) {
                    out.extend_from_slice(ids);
                    return;
                }
            }
            self.merge_chunk(b, &mut scratch);
            out.extend_from_slice(&scratch);
            if let Some(cache) = guard.as_mut() {
                if b.len() <= CACHE_MAX_CHUNK_BYTES {
                    if cache.len() >= cap {
                        cache.clear();
                    }
                    cache.insert(Box::from(b), Box::from(scratch.as_slice()));
                }
            }
        });
    }

    /// `encode(text)` followed by EOS.
    pub fn encode_doc(&self, text: &str) -> Vec<u32> {
        let mut v = self.encode(text);
        v.push(EOS);
        v
    }

    /// Raw bytes of `ids` (EOS/PAD contribute nothing, unknown ids contribute U+FFFD).
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            match self.token_bytes.get(id as usize) {
                Some(b) => out.extend_from_slice(b),
                None => out.extend_from_slice("\u{fffd}".as_bytes()),
            }
        }
        out
    }

    /// Text of `ids`: bytes concatenated, `from_utf8_lossy`; EOS and PAD render as nothing.
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Like `decode` but EOS renders as `<|eos|>` and PAD as `<|pad|>` (for display).
    pub fn decode_marked(&self, ids: &[u32]) -> String {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            match id {
                EOS => out.extend_from_slice(b"<|eos|>"),
                PAD => out.extend_from_slice(b"<|pad|>"),
                _ => out.extend(self.decode_bytes(&[id])),
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(w, "{FILE_MAGIC} {FILE_VERSION} {}", self.vocab_size())?;
        for &(l, r) in &self.merges {
            writeln!(w, "{l} {r}")?;
        }
        w.flush()
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Bpe> {
        let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
        let path = path.as_ref();
        let mut lines = BufReader::new(File::open(path)?).lines();
        let header = lines.next().transpose()?.unwrap_or_default();
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() != 3 || parts[0] != FILE_MAGIC || parts[1] != FILE_VERSION {
            return Err(bad(format!("{}: not an `{FILE_MAGIC} {FILE_VERSION}` file (header {header:?})", path.display())));
        }
        let vocab: usize = parts[2].parse().map_err(|_| bad(format!("bad vocab size {:?}", parts[2])))?;
        let mut merges = Vec::new();
        for (n, line) in lines.enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let mut it = line.split_whitespace();
            let parse = |s: Option<&str>| -> Option<u32> { s?.parse().ok() };
            let (l, r) = match (parse(it.next()), parse(it.next()), it.next()) {
                (Some(l), Some(r), None) => (l, r),
                _ => return Err(bad(format!("{}: line {}: bad merge {line:?}", path.display(), n + 2))),
            };
            merges.push((l, r));
        }
        if FIRST_MERGE_ID as usize + merges.len() != vocab {
            return Err(bad(format!("{}: header says vocab {vocab} but file has {} merges", path.display(), merges.len())));
        }
        Bpe::from_merges(merges).map_err(|e| bad(format!("{}: {e}", path.display())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;
    use std::path::PathBuf;

    fn tmp_dir(name: &str) -> PathBuf {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-tmp")
            .join(format!("bpe-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    const CORPUS: &[&str] = &[
        "The cat sat on the mat. The cat is fat.",
        "I have 3 cats and 12 dogs; 3 + 12 = 15.",
        "Don't stop -- it's 10:30 now!!! (Really?)",
        "Café crème, naïve résumé, Ελληνικά, русский, 日本語 テキスト.",
        "Emoji: 😀 🚀 ❤️ 👩‍👩‍👧 done.",
        "  leading spaces and trailing spaces  ",
        "tabs\tand\nnewlines\r\nand   runs   of   spaces",
        "the theory of the thermal theme then thereafter",
        "Large international companies are involved in bauxite, iron ore, diamond, and gold mining operations.",
        "Paleoceanography, 8(2): 193-208.",
        "What do you mean ? It will help us to relax .",
    ];

    fn counts_of(texts: &[&str], repeat: usize) -> Vec<(Vec<u8>, u64)> {
        let mut c = ChunkCounter::new();
        for _ in 0..repeat {
            for t in texts {
                c.add_text(t);
            }
        }
        let mut v: Vec<(Vec<u8>, u64)> = c.into_counts().collect();
        v.sort();
        v
    }

    fn trained(vocab: usize) -> Bpe {
        Bpe::train(counts_of(CORPUS, 5).into_iter(), vocab)
    }

    fn random_text(r: &mut SplitMix64, len: usize) -> String {
        const ALPHA: &[&str] = &[
            "a", "b", "e", "t", "h", "Z", " ", " ", " ", "  ", "\t", "\n", "\r\n", "0", "1", "7", "9",
            ".", ",", "'", "!", "-", "(", ")", "é", "ß", "日", "😀", "\u{301}", "\u{a0}", "\u{3000}", "٣",
            "²", "❤️",
        ];
        let mut s = String::new();
        for _ in 0..len {
            s.push_str(ALPHA[r.below(ALPHA.len() as u64) as usize]);
        }
        s
    }

    // ---------- pre-tokenizer ----------

    #[test]
    fn pretok_examples() {
        let cases: &[(&str, &[&str])] = &[
            ("", &[]),
            ("Hello world", &["Hello", " world"]),
            ("a  b", &["a", " ", " b"]),
            ("a   b", &["a", "  ", " b"]),
            ("x 12", &["x", " ", "1", "2"]),
            ("3.14", &["3", ".", "1", "4"]),
            ("don't", &["don", "'", "t"]),
            (" don't", &[" don", "'", "t"]),
            (" (hi)", &[" (", "hi", ")"]),
            ("hello, world", &["hello", ",", " world"]),
            ("a\n b", &["a", "\n", " b"]),
            ("a \nb", &["a", " \n", "b"]),
            ("a\tb", &["a", "\t", "b"]),
            ("  ", &["  "]),
            (" ", &[" "]),
            ("end. ", &["end", ".", " "]),
            ("café crème", &["café", " crème"]),
            ("I ❤️ u", &["I", " ❤️", " u"]),
            ("abc123def", &["abc", "1", "2", "3", "def"]),
            (" 7", &[" ", "7"]),
            ("...\"", &["...\""]),
            ("hi!? ok", &["hi", "!?", " ok"]),
            ("\u{a0}x", &["\u{a0}", "x"]),
            ("e\u{301}", &["e", "\u{301}"]),
            ("日本語 テキスト", &["日本語", " テキスト"]),
        ];
        for (text, want) in cases {
            assert_eq!(pre_tokenize(text), want.to_vec(), "pre_tokenize({text:?})");
        }
    }

    #[derive(PartialEq, Clone, Copy, Debug)]
    enum Cl {
        L,
        D,
        W,
        P,
    }
    fn cl(c: char) -> Cl {
        if c.is_ascii_digit() {
            Cl::D
        } else if c.is_alphabetic() {
            Cl::L
        } else if c.is_whitespace() {
            Cl::W
        } else {
            Cl::P
        }
    }

    /// Check the grammar of every chunk and the rules between neighbours.
    fn check_chunks(text: &str, chunks: &[&str]) {
        assert_eq!(chunks.concat(), text, "chunks must concatenate to the text");
        for (k, ch) in chunks.iter().enumerate() {
            assert!(!ch.is_empty());
            let cs: Vec<char> = ch.chars().collect();
            let body: &[char] = if cs.len() > 1 && cs[0] == ' ' && cl(cs[1]) != Cl::W { &cs[1..] } else { &cs };
            let c0 = cl(body[0]);
            match c0 {
                Cl::D => assert!(cs.len() == 1, "digit chunk must be one digit: {ch:?}"),
                Cl::L | Cl::P => assert!(body.iter().all(|&c| cl(c) == c0), "mixed chunk {ch:?}"),
                Cl::W => assert!(cs.iter().all(|&c| cl(c) == Cl::W), "mixed whitespace chunk {ch:?}"),
            }
            if let Some(next) = chunks.get(k + 1) {
                let n0 = next.chars().next().unwrap();
                // a whitespace chunk ending in ' ' must not be followed by an unspaced L/P chunk
                if c0 == Cl::W && ch.ends_with(' ') {
                    assert!(!(cl(n0) == Cl::L || cl(n0) == Cl::P), "space not attached: {ch:?} {next:?}");
                }
                // two L runs (or two P runs) are never adjacent without a leading space
                if (c0 == Cl::L || c0 == Cl::P) && cl(n0) == c0 {
                    panic!("run split in two: {ch:?} {next:?}");
                }
            }
        }
    }

    #[test]
    fn pretok_grammar_property() {
        let mut r = SplitMix64::new(99);
        for _ in 0..3000 {
            let len = r.below(30) as usize;
            let t = random_text(&mut r, len);
            check_chunks(&t, &pre_tokenize(&t));
        }
        for t in CORPUS {
            check_chunks(t, &pre_tokenize(t));
        }
    }

    // ---------- training ----------

    #[test]
    fn hand_checked_training_example() {
        // "aaab aaab aaab aaab" -> chunks "aaab" x1, " aaab" x3
        // pair counts: (a,a)=8 (overlapping counted), (a,b)=4, (' ',a)=3
        // 1st: a+a -> 258 "aa";  words [aa,a,b], [' ',aa,a,b]
        //      pairs (aa,a)=4, (a,b)=4, (' ',aa)=3 -> tie at 4, smaller pair (97,98) wins
        // 2nd: a+b -> 259 "ab";  pairs (aa,ab)=4, (' ',aa)=3
        // 3rd: aa+ab -> 260 "aaab"; pairs (' ',aaab)=3
        // 4th: ' '+aaab -> 261 " aaab"; then no pairs left
        let bpe = Bpe::train(counts_of(&["aaab aaab aaab aaab"], 1).into_iter(), 1000);
        assert_eq!(bpe.merges(), &[(97, 97), (97, 98), (258, 259), (32, 260)]);
        assert_eq!(bpe.vocab_size(), 262);
        assert_eq!(bpe.token_bytes(261), b" aaab");
        assert_eq!(bpe.encode("aaab aaab"), vec![260, 261]);
        // vocab cap respected
        let small = Bpe::train(counts_of(&["aaab aaab aaab aaab"], 1).into_iter(), 260);
        assert_eq!(small.merges(), &[(97, 97), (97, 98)]);
        assert_eq!(small.vocab_size(), 260);
    }

    #[test]
    fn vocab_below_base_means_no_merges() {
        let b = Bpe::train(counts_of(CORPUS, 1).into_iter(), 100);
        assert_eq!(b.merges().len(), 0);
        assert_eq!(b.vocab_size(), 258);
        let t = "hello 12";
        assert_eq!(b.encode(t), t.bytes().map(|x| x as u32).collect::<Vec<_>>());
    }

    #[test]
    fn training_is_deterministic() {
        let a = trained(400);
        let b = trained(400);
        assert_eq!(a.merges(), b.merges());
        // input order must not matter either
        let mut counts = counts_of(CORPUS, 5);
        SplitMix64::new(5).shuffle(&mut counts);
        let c = Bpe::train(counts.into_iter(), 400);
        assert_eq!(a.merges(), c.merges());
        // duplicate keys in the input are summed: counts ++ counts == counts x 2
        let once = counts_of(CORPUS, 5);
        let doubled: Vec<(Vec<u8>, u64)> = once.iter().cloned().chain(once.iter().cloned()).collect();
        let d = Bpe::train(doubled.into_iter(), 400);
        let e = Bpe::train(counts_of(CORPUS, 10).into_iter(), 400);
        assert_eq!(d.merges(), e.merges());
    }

    #[test]
    fn learns_up_to_vocab_size() {
        let b = trained(300);
        assert_eq!(b.vocab_size(), 300);
        assert_eq!(b.merges().len(), 42);
        for (i, &(l, r)) in b.merges().iter().enumerate() {
            let id = FIRST_MERGE_ID + i as u32;
            assert!(l < id && r < id && l != EOS && l != PAD && r != EOS && r != PAD);
            let mut want = b.token_bytes(l).to_vec();
            want.extend_from_slice(b.token_bytes(r));
            assert_eq!(b.token_bytes(id), want.as_slice());
        }
    }

    // ---------- encode / decode ----------

    #[test]
    fn roundtrip() {
        let b = trained(500);
        let mut cases: Vec<String> = vec![
            "".into(),
            "hello world".into(),
            "The cat sat on the mat.".into(),
            "Café crème, naïve résumé, Ελληνικά, русский, 日本語".into(),
            "😀🚀 ❤️ 👩‍👩‍👧".into(),
            "0123456789 3.14159 -42 1e10".into(),
            "line1\nline2\r\n\ttabbed\t\t".into(),
            "   leading and trailing   ".into(),
            " ".into(),
            "\n".into(),
            "e\u{301}\u{a0}\u{3000}x".into(),
        ];
        let mut r = SplitMix64::new(1);
        for _ in 0..500 {
            let len = r.below(40) as usize;
            cases.push(random_text(&mut r, len));
        }
        for t in &cases {
            let ids = b.encode(t);
            assert_eq!(&b.decode(&ids), t, "roundtrip of {t:?}");
            assert!(ids.iter().all(|&i| (i as usize) < b.vocab_size() && i != EOS && i != PAD));
        }
        assert!(b.encode("").is_empty());
    }

    #[test]
    fn digits_are_always_single_tokens() {
        // a digit-heavy corpus: frequent "12", "2024", " 3" pairs must still never merge
        let digits = ["12 12 12 2024 2024 x1 a2b 3.14 100% $5 +7 -8 (9)"];
        let b = Bpe::train(counts_of(&digits, 50).into_iter(), 400);
        for id in FIRST_MERGE_ID..b.vocab_size() as u32 {
            let tb = b.token_bytes(id);
            assert!(!tb.iter().any(|c| c.is_ascii_digit()), "token {id} {:?} contains a digit", String::from_utf8_lossy(tb));
        }
        let text = "In 2024 there were 1234567 x9y and 3.14";
        let ids = b.encode(text);
        for &id in &ids {
            let tb = b.token_bytes(id);
            if tb.iter().any(|c| c.is_ascii_digit()) {
                assert_eq!(tb.len(), 1);
            }
        }
        let n_digits = text.bytes().filter(|c| c.is_ascii_digit()).count();
        let n_digit_tokens = ids.iter().filter(|&&i| i < 256 && (i as u8).is_ascii_digit()).count();
        assert_eq!(n_digits, n_digit_tokens);
    }

    #[test]
    fn merges_never_cross_chunk_boundaries() {
        let b = trained(600);
        // 1) every learned token that is valid UTF-8 pre-tokenizes to exactly one chunk
        for id in FIRST_MERGE_ID..b.vocab_size() as u32 {
            if let Ok(s) = std::str::from_utf8(b.token_bytes(id)) {
                assert_eq!(pre_tokenize(s).len(), 1, "token {id} {s:?} spans chunks");
            }
        }
        // 2) when encoding, every chunk boundary is also a token boundary
        let mut r = SplitMix64::new(2);
        let mut texts: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        for _ in 0..300 {
            let len = r.below(40) as usize;
            texts.push(random_text(&mut r, len));
        }
        for t in &texts {
            let mut chunk_bounds = vec![0usize];
            for c in pre_tokenize(t) {
                chunk_bounds.push(chunk_bounds.last().unwrap() + c.len());
            }
            let mut tok_bounds = std::collections::HashSet::new();
            let mut pos = 0usize;
            tok_bounds.insert(0);
            for id in b.encode(t) {
                pos += b.token_bytes(id).len();
                tok_bounds.insert(pos);
            }
            for cb in chunk_bounds {
                assert!(tok_bounds.contains(&cb), "chunk boundary {cb} not a token boundary in {t:?}");
            }
        }
    }

    /// Reference encoder: apply every merge in rank order to each chunk (what training did).
    fn reference_encode(b: &Bpe, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for chunk in pre_tokenize(text) {
            let mut ids: Vec<u32> = chunk.bytes().map(|x| x as u32).collect();
            for (i, &(l, r)) in b.merges().iter().enumerate() {
                let new = FIRST_MERGE_ID + i as u32;
                let mut w = Vec::with_capacity(ids.len());
                let mut j = 0;
                while j < ids.len() {
                    if j + 1 < ids.len() && ids[j] == l && ids[j + 1] == r {
                        w.push(new);
                        j += 2;
                    } else {
                        w.push(ids[j]);
                        j += 1;
                    }
                }
                ids = w;
            }
            out.extend(ids);
        }
        out
    }

    #[test]
    fn encoder_matches_sequential_merge_reference() {
        let b = trained(600);
        let mut r = SplitMix64::new(3);
        let mut texts: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        texts.push("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into());
        texts.push("thethethethe the the theory".into());
        for _ in 0..300 {
            let len = r.below(40) as usize;
            texts.push(random_text(&mut r, len));
        }
        for t in &texts {
            assert_eq!(b.encode(t), reference_encode(&b, t), "encode({t:?})");
        }
    }

    #[test]
    fn encode_doc_appends_eos_and_decode_drops_specials() {
        let b = trained(400);
        let d = b.encode_doc("The cat.");
        assert_eq!(*d.last().unwrap(), EOS);
        assert_eq!(&d[..d.len() - 1], b.encode("The cat.").as_slice());
        assert_eq!(b.decode(&d), "The cat.");
        let mut two = b.encode_doc("a");
        two.extend(b.encode_doc("b"));
        two.push(PAD);
        assert_eq!(b.decode(&two), "ab");
        assert_eq!(b.decode_marked(&two), "a<|eos|>b<|eos|><|pad|>");
        assert_eq!(b.encode_doc(""), vec![EOS]);
    }

    #[test]
    fn cache_does_not_change_results() {
        let mut b = trained(500);
        let t = CORPUS.join(" ");
        let first = b.encode(&t);
        let second = b.encode(&t);
        assert_eq!(first, second);
        b.set_cache_capacity(2); // forces frequent clearing
        assert_eq!(b.encode(&t), first);
        b.set_cache_capacity(0); // no caching at all
        assert_eq!(b.encode(&t), first);
    }

    #[test]
    fn decode_handles_unknown_ids_and_partial_utf8() {
        let b = trained(300);
        // unknown id renders as U+FFFD; a lone UTF-8 lead byte renders lossily
        assert_eq!(b.decode(&[104, 999_999, 105]), "h\u{fffd}i");
        assert_eq!(b.decode(&[0xC3]), "\u{fffd}");
    }

    // ---------- save / load ----------

    #[test]
    fn save_load_roundtrip() {
        let b = trained(500);
        let dir = tmp_dir("saveload");
        let p = dir.join("bpe.txt");
        b.save(&p).unwrap();
        assert!(b.merges().len() > 100, "test corpus too small: {} merges", b.merges().len());
        let text = std::fs::read_to_string(&p).unwrap(); // a tiny file written by this test
        let header = format!("ava-bpe v1 {}\n", b.vocab_size());
        assert!(text.starts_with(&header), "header: {:?}", &text[..20.min(text.len())]);
        assert_eq!(text.lines().count(), 1 + b.merges().len());
        let c = Bpe::load(&p).unwrap();
        assert_eq!(c.merges(), b.merges());
        assert_eq!(c.vocab_size(), b.vocab_size());
        let t = CORPUS.join("\n");
        assert_eq!(c.encode(&t), b.encode(&t));
    }

    #[test]
    fn load_rejects_bad_files() {
        let dir = tmp_dir("badload");
        let cases: &[(&str, &str)] = &[
            ("nohdr.txt", "97 97\n"),
            ("badver.txt", "ava-bpe v2 259\n97 97\n"),
            ("count.txt", "ava-bpe v1 260\n97 97\n"),
            ("future.txt", "ava-bpe v1 259\n97 258\n"),
            ("special.txt", "ava-bpe v1 259\n256 97\n"),
            ("garbage.txt", "ava-bpe v1 259\n97 x\n"),
        ];
        for (name, body) in cases {
            let p = dir.join(name);
            std::fs::write(&p, body).unwrap();
            assert!(Bpe::load(&p).is_err(), "{name} should be rejected");
        }
        let ok = dir.join("ok.txt");
        std::fs::write(&ok, "ava-bpe v1 259\n97 97\n").unwrap();
        assert_eq!(Bpe::load(&ok).unwrap().encode("aa"), vec![258]);
    }

    #[test]
    fn chunk_counter_counts() {
        let mut c = ChunkCounter::new();
        c.add_text("the cat the");
        c.add_text("the");
        assert_eq!(c.total_bytes, 14);
        assert_eq!(c.total_chunks, 4);
        let mut v: Vec<(Vec<u8>, u64)> = c.into_counts().collect();
        v.sort();
        assert_eq!(
            v,
            vec![(b" cat".to_vec(), 1), (b" the".to_vec(), 1), (b"the".to_vec(), 2)]
        );
    }
}
