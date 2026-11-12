// TESTING / DESIGN ONLY — contam (v42): does an eval set's text appear in the training shards? (13-gram overlap)
//
// The check of GPT-3 (Brown et al. 2020, appendix C), on words: text is lowercased and split into words (a word is a
// run of ASCII letters and digits, or of non-ASCII bytes; everything else separates), and an item counts as
// contaminated when any run of 13 consecutive words of it also appears in training text. Words are hashed (64-bit
// FNV-1a) and 13 word hashes into one 64-bit gram hash, so a false match needs a 64-bit collision.
//
// Only the eval side is held in memory: a hash map from each of its 13-grams to the items that have it. The training
// shards are streamed one shard at a time, token by token, through the tokenizer's byte table (so a word split over
// two tokens is one word); EOS ends a document and no 13-gram spans it. A document can run on from one shard file into
// the next, so the stream carries over shard files in order; with several threads, each thread takes a run of shards
// and first re-reads the last OVERLAP tokens of the shard before its run, so no gram across a split is missed (grams
// found twice are one entry of a set).

use ava_data::bpe::{EOS, PAD};
use ava_data::rng::mix64;
use ava_data::shards::{ShardSet, ShardTokens};
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// Words per gram.
pub const GRAM: usize = 13;
/// Tokens of the previous shard a thread re-reads before its run (far more than 13 words).
pub const OVERLAP: usize = 1 << 16;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The keys are already mixed 64-bit hashes, so the table uses them as they are (no second hash).
#[derive(Default)]
pub struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8 | b as u64).rotate_left(5);
        }
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = x;
    }
}

pub type GramSet = HashSet<u64, BuildHasherDefault<KeyHasher>>;
type GramMap<V> = HashMap<u64, V, BuildHasherDefault<KeyHasher>>;

/// Turns a byte stream into the hashes of its 13-word grams.
#[derive(Clone, Debug)]
pub struct GramStream {
    word: u64,
    in_word: bool,
    ring: [u64; GRAM],
    /// words since the last document boundary
    words: usize,
}

impl Default for GramStream {
    fn default() -> Self {
        GramStream { word: FNV_OFFSET, in_word: false, ring: [0; GRAM], words: 0 }
    }
}

impl GramStream {
    pub fn new() -> GramStream {
        GramStream::default()
    }

    /// Feed one byte; `f` gets the hash of every 13-gram this byte completes.
    #[inline]
    pub fn byte(&mut self, b: u8, f: &mut impl FnMut(u64)) {
        if b.is_ascii_alphanumeric() || b >= 0x80 {
            if !self.in_word {
                self.word = FNV_OFFSET;
                self.in_word = true;
            }
            self.word = (self.word ^ b.to_ascii_lowercase() as u64).wrapping_mul(FNV_PRIME);
        } else if self.in_word {
            self.end_word(f);
        }
    }

    #[inline]
    fn end_word(&mut self, f: &mut impl FnMut(u64)) {
        self.in_word = false;
        self.ring[self.words % GRAM] = self.word;
        self.words += 1;
        if self.words >= GRAM {
            // the oldest of the last 13 words sits where the next one will go
            let start = self.words % GRAM;
            let mut h = GRAM as u64;
            for j in 0..GRAM {
                h = mix64(h ^ self.ring[(start + j) % GRAM]);
            }
            f(h);
        }
    }

    /// A document ends: its last word is finished and no gram crosses this point.
    pub fn boundary(&mut self, f: &mut impl FnMut(u64)) {
        if self.in_word {
            self.end_word(f);
        }
        self.words = 0;
    }
}

/// Every 13-gram hash of a text (one document).
pub fn text_grams(text: &str) -> Vec<u64> {
    let mut out = Vec::new();
    let mut g = GramStream::new();
    let mut push = |h: u64| out.push(h);
    for &b in text.as_bytes() {
        g.byte(b, &mut push);
    }
    g.boundary(&mut push);
    out
}

/// The eval side: each item's distinct 13-grams, and which items have each gram.
#[derive(Clone, Debug, Default)]
pub struct EvalGrams {
    pub ids: Vec<String>,
    /// distinct grams per item (an item shorter than 13 words has none and can never match)
    pub grams: Vec<Vec<u64>>,
    owners: GramMap<Vec<u32>>,
}

impl EvalGrams {
    /// From (id, text) items.
    pub fn new(items: &[(String, String)]) -> EvalGrams {
        let mut e = EvalGrams::default();
        for (i, (id, text)) in items.iter().enumerate() {
            let mut g = text_grams(text);
            g.sort_unstable();
            g.dedup();
            for &h in &g {
                e.owners.entry(h).or_default().push(i as u32);
            }
            e.ids.push(id.clone());
            e.grams.push(g);
        }
        e
    }

    /// Distinct grams over all items.
    pub fn distinct(&self) -> usize {
        self.owners.len()
    }

    #[inline]
    pub fn contains(&self, h: u64) -> bool {
        self.owners.contains_key(&h)
    }

    /// Per item: how many of its grams are in `found` (grams seen in training).
    pub fn hits(&self, found: &GramSet) -> Vec<usize> {
        self.grams.iter().map(|g| g.iter().filter(|h| found.contains(h)).count()).collect()
    }
}

/// A training-side stream: the gram state carried from token to token, and the eval grams seen so far.
pub struct Scanner<'a> {
    eval: &'a EvalGrams,
    /// the tokenizer's bytes per id; ids past the table (the glance's marks) add nothing
    bytes: &'a [Vec<u8>],
    g: GramStream,
    pub found: GramSet,
    pub tokens: u64,
}

impl<'a> Scanner<'a> {
    pub fn new(eval: &'a EvalGrams, bytes: &'a [Vec<u8>]) -> Scanner<'a> {
        Scanner { eval, bytes, g: GramStream::new(), found: GramSet::default(), tokens: 0 }
    }

    /// Feed tokens (EOS ends a document; PAD is skipped).
    pub fn feed(&mut self, tokens: impl Iterator<Item = u32>) {
        let (eval, bytes) = (self.eval, self.bytes);
        let found = &mut self.found;
        let mut hit = |h: u64| {
            if eval.contains(h) {
                found.insert(h);
            }
        };
        for t in tokens {
            self.tokens += 1;
            if t == EOS {
                self.g.boundary(&mut hit);
            } else if t != PAD {
                if let Some(bs) = bytes.get(t as usize) {
                    for &b in bs {
                        self.g.byte(b, &mut hit);
                    }
                }
            }
        }
    }

    /// Feed a loaded shard's tokens from `from` on.
    pub fn feed_shard(&mut self, sh: &ShardTokens, from: usize) {
        match sh {
            ShardTokens::U16(v) => self.feed(v[from.min(v.len())..].iter().map(|&t| t as u32)),
            ShardTokens::U32(v) => self.feed(v[from.min(v.len())..].iter().copied()),
        }
    }

    /// The end of the stream (a last document without EOS still counts).
    pub fn finish(mut self) -> (GramSet, u64) {
        let (eval, found) = (self.eval, &mut self.found);
        self.g.boundary(&mut |h: u64| {
            if eval.contains(h) {
                found.insert(h);
            }
        });
        (self.found, self.tokens)
    }
}

/// Scan the first `max_shards` shards of `set` (all if None) with `threads` threads, each over a run of shards that
/// first re-reads the last `overlap` tokens before its run (OVERLAP in the bin; see the top of the file). Returns the
/// eval grams seen and the tokens read (re-reads not counted).
pub fn scan_shards(eval: &EvalGrams, bytes: &[Vec<u8>], set: &ShardSet, threads: usize, max_shards: Option<usize>, overlap: usize) -> Result<(GramSet, u64), String> {
    let n = max_shards.map_or(set.len(), |m| m.min(set.len()));
    let threads = threads.clamp(1, n.max(1));
    let runs: Vec<(usize, usize)> = (0..threads).map(|t| (t * n / threads, (t + 1) * n / threads)).filter(|r| r.0 < r.1).collect();
    let results: Vec<Result<(GramSet, u64), String>> = std::thread::scope(|sc| {
        let hs: Vec<_> = runs
            .iter()
            .map(|&(a, b)| {
                sc.spawn(move || -> Result<(GramSet, u64), String> {
                    let load = |i: usize| set.load_shard(i).map_err(|e| format!("shard {i}: {e}"));
                    let mut s = Scanner::new(eval, bytes);
                    // the shards before the run that hold its last `overlap` tokens (usually just the one before)
                    let (mut first, mut have) = (a, 0usize);
                    while first > 0 && have < overlap {
                        first -= 1;
                        have += set.shards[first].header.n_tokens as usize;
                    }
                    let mut skip = have.saturating_sub(overlap);
                    let mut reread = 0u64;
                    for i in first..a {
                        let sh = load(i)?;
                        let from = skip.min(sh.len());
                        skip -= from;
                        s.feed_shard(&sh, from);
                        reread += (sh.len() - from) as u64;
                    }
                    for i in a..b {
                        let sh = load(i)?;
                        s.feed_shard(&sh, 0);
                    }
                    let (f, t) = s.finish();
                    Ok((f, t - reread))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("a scan thread panicked".into()))).collect()
    });
    let (mut found, mut tokens) = (GramSet::default(), 0u64);
    for r in results {
        let (f, t) = r?;
        found.extend(f);
        tokens += t;
    }
    Ok((found, tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ITEM: &str = "The mitochondria is the powerhouse of the cell, and it makes most of the energy a cell uses.";

    #[test]
    fn grams_follow_words_not_case_or_punctuation() {
        let a = text_grams(ITEM);
        assert_eq!(a.len(), 18 - GRAM + 1); // 18 words
        let b = text_grams("the MITOCHONDRIA is the powerhouse -- of the cell; and it makes most of the energy a cell uses");
        assert_eq!(a, b);
        assert!(text_grams("only twelve words here one two three four five six seven eight").is_empty());
        assert_eq!(text_grams("one two three four five six seven eight nine ten eleven twelve thirteen").len(), 1);
        // word order matters
        assert_ne!(text_grams("b a c d e f g h i j k l m"), text_grams("a b c d e f g h i j k l m"));
    }

    fn toy_bytes() -> Vec<Vec<u8>> {
        // byte tokens 0..255, then EOS and PAD (empty), then two merged tokens
        let mut t: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        t.push(Vec::new());
        t.push(Vec::new());
        t.push(b"mito".to_vec()); // 258
        t.push(b"chondria is".to_vec()); // 259: a word boundary inside one token
        t
    }

    fn encode(s: &str) -> Vec<u32> {
        // greedy: the two merged tokens where they fit, else bytes (so words cross token boundaries)
        let b = s.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < b.len() {
            if b[i..].starts_with(b"mito") {
                out.push(258);
                i += 4;
            } else if b[i..].starts_with(b"chondria is") {
                out.push(259);
                i += 11;
            } else {
                out.push(b[i] as u32);
                i += 1;
            }
        }
        out
    }

    #[test]
    fn a_toy_shard_set_finds_the_copied_item_only() {
        let items = vec![
            ("copied".to_string(), ITEM.to_string()),
            ("fresh".to_string(), "Which planet has the largest moon in the solar system, and what is that moon called today?".to_string()),
            ("split".to_string(), "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi".to_string()),
            ("short".to_string(), "too short to match".to_string()),
        ];
        let eval = EvalGrams::new(&items);
        assert_eq!(eval.grams.iter().map(Vec::len).collect::<Vec<_>>(), vec![6, 5, 2, 0]);
        // training: the copied item inside a longer document, in other case and punctuation, and split over tokens;
        // the "split" item cut by a document boundary after 7 words; only 12 words in a row of the "fresh" item
        let mut toks = encode("Biology notes. The mitochondria is THE POWERHOUSE OF THE CELL -- and it makes most of the energy a cell uses! More text.");
        assert!(toks.contains(&258) && toks.contains(&259));
        toks.push(EOS);
        toks.extend(encode("alpha beta gamma delta epsilon zeta eta"));
        toks.push(EOS);
        toks.extend(encode("theta iota kappa lambda mu nu xi"));
        toks.push(PAD);
        toks.push(EOS);
        toks.extend(encode("which planet has the largest moon in the solar system and what"));
        toks.push(EOS);
        let tb = toy_bytes();
        // one stream over all tokens
        let mut s = Scanner::new(&eval, &tb);
        s.feed(toks.iter().copied());
        let (found, n) = s.finish();
        assert_eq!(n, toks.len() as u64);
        assert_eq!(eval.hits(&found), vec![6, 0, 0, 0]);
        // the same tokens as a shard set of 40-token shards (documents and grams run across shard files), scanned
        // with 1, 2 and 3 threads; an overlap of 120 tokens covers any 13 words here (a gram spans < 100 tokens)
        let dir = std::env::temp_dir().join(format!("ava-contam-toy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = ava_data::shards::ShardWriter::new(&dir, 260, 40).unwrap();
        w.push_doc(&toks).unwrap();
        w.finish().unwrap();
        let set = ShardSet::open(&dir).unwrap();
        assert!(set.len() >= 5, "{} shards", set.len());
        for threads in [1, 2, 3] {
            let (f, t) = scan_shards(&eval, &tb, &set, threads, None, 120).unwrap();
            assert_eq!((t, &f), (toks.len() as u64, &found), "{threads} threads");
        }
        // without the overlap, a thread starting mid-document would miss grams across its first shard boundary
        let (f, _) = scan_shards(&eval, &tb, &set, 3, None, 0).unwrap();
        assert!(f.len() <= found.len());
        // a quick trial over the first shards only
        let (_, t) = scan_shards(&eval, &tb, &set, 2, Some(2), 120).unwrap();
        assert_eq!(t, 80);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
