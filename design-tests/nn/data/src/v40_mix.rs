//! TESTING / DESIGN ONLY — v40_mix: v40's reading mix (long books + usual text + practice v2), shuffled by document.
//!
//! SOURCES (all vocab 4112, length marks already in):
//!   0 books       every book given (one document each: `books::encode_book`)
//!   1 usual       documents read in order from an existing marked shard directory (v39's
//!                 train shards), split at EOS, taken whole until the target is reached
//!   2 practice_v2 practice documents (already encoded and marked), taken in order until the
//!                 target is reached
//!
//! SHARES. All books are taken (B tokens). The targets keep the shares 50 / 47 / 3: usual
//! `round(B * 0.47 / 0.50)` and practice `round(B * 0.03 / 0.50)`, so the total is about
//! 2B (v40 wanted about 120M: books of about 60M tokens). A source overshoots its target by
//! less than one document.
//!
//! SHUFFLE. Every document goes to a list; the list is shuffled once (`SplitMix64`, seeded by
//! `derive_seed(seed, MIX_LABEL, 0)`) and written in that order through a `ShardWriter` (shards
//! of `shard_tokens` tokens; a document may continue in the next shard, as in every shard set).
//!
//! The documents are held in one u16 pool (ids < 4112), about 2 bytes per token.
//!
//! `known` optionally names documents to count inside the usual text (by `token_hash`): v40
//! uses it to report how much of v39's practice text the usual text brings along.

use crate::bpe::EOS;
use crate::numbers::VOCAB_GLANCE;
use crate::rng::{derive_seed, SplitMix64};
use crate::shards::{ShardSet, ShardTokens, ShardWriter};
use std::collections::HashSet;
use std::io;
use std::path::Path;

pub const SOURCES: [&str; 3] = ["books", "usual", "practice_v2"];
/// Target shares of books, usual text and practice v2.
pub const SHARES: [f64; 3] = [0.50, 0.47, 0.03];
pub const DEFAULT_SEED: u64 = 40;
const MIX_LABEL: u64 = 0x7634_305F_6D69_78;

/// [books, usual, practice] token targets for `book_tokens` book tokens (see the module docs).
pub fn targets(book_tokens: u64) -> [u64; 3] {
    targets_with(book_tokens, SHARES)
}

/// `targets` for other [books, usual, practice] shares (v41: 25 / 70 / 5).
pub fn targets_with(book_tokens: u64, shares: [f64; 3]) -> [u64; 3] {
    let b = book_tokens as f64;
    [book_tokens, (b * shares[1] / shares[0]).round() as u64, (b * shares[2] / shares[0]).round() as u64]
}

/// FNV-1a over token values (a document's identity for `known`).
pub fn token_hash(toks: impl IntoIterator<Item = u32>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for t in toks {
        h ^= t as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MixReport {
    pub docs: [u64; 3],
    pub tokens: [u64; 3],
    pub targets: [u64; 3],
    pub total: u64,
    pub shards: usize,
    /// Documents and tokens of the usual text found in `known`.
    pub known_docs: u64,
    pub known_tokens: u64,
}

impl MixReport {
    pub fn share(&self, src: usize) -> f64 {
        self.tokens[src] as f64 / self.total.max(1) as f64
    }
}

struct Pool {
    toks: Vec<u16>,
    docs: Vec<(u8, usize, usize)>,
}

impl Pool {
    fn push(&mut self, src: u8, doc: impl ExactSizeIterator<Item = u32>) -> io::Result<()> {
        let start = self.toks.len();
        let len = doc.len();
        for t in doc {
            if t as usize >= VOCAB_GLANCE {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("token {t} >= {VOCAB_GLANCE}")));
            }
            self.toks.push(t as u16);
        }
        self.docs.push((src, start, len));
        Ok(())
    }
}

/// Push whole documents of the shard directory `dir`, in order, until `target` tokens.
fn take_usual(dir: &Path, target: u64, pool: &mut Pool, known: Option<&HashSet<u64>>, rep: &mut MixReport) -> io::Result<()> {
    let set = ShardSet::open(dir)?;
    if set.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound, format!("{}: no shards", dir.display())));
    }
    if set.vocab_size as usize != VOCAB_GLANCE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: vocab {} (want {VOCAB_GLANCE}, marked shards)", dir.display(), set.vocab_size),
        ));
    }
    let mut cur: Vec<u32> = Vec::new();
    for i in 0..set.len() {
        if rep.tokens[1] >= target {
            break;
        }
        let shard = set.load_shard(i)?;
        let n = shard.len();
        for j in 0..n {
            let t = match &shard {
                ShardTokens::U16(v) => v[j] as u32,
                ShardTokens::U32(v) => v[j],
            };
            cur.push(t);
            if t == EOS {
                if let Some(k) = known {
                    if k.contains(&token_hash(cur.iter().copied())) {
                        rep.known_docs += 1;
                        rep.known_tokens += cur.len() as u64;
                    }
                }
                pool.push(1, cur.iter().copied())?;
                rep.docs[1] += 1;
                rep.tokens[1] += cur.len() as u64;
                cur.clear();
                if rep.tokens[1] >= target {
                    break;
                }
            }
        }
    }
    if rep.tokens[1] < target {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("{}: only {} whole documents' tokens, target {target}", dir.display(), rep.tokens[1]),
        ));
    }
    Ok(())
}

/// Build the mix into `out` (which must hold no shards). `books` are moved into the pool one by
/// one (their memory is freed as they go); `practice` is read lazily, only as far as needed.
pub fn build_mix(
    books: Vec<Vec<u32>>,
    usual_dir: &Path,
    practice: impl Iterator<Item = Vec<u32>>,
    out: &Path,
    seed: u64,
    shard_tokens: usize,
    known: Option<&HashSet<u64>>,
) -> io::Result<MixReport> {
    build_mix_with(books, usual_dir, practice, out, seed, shard_tokens, known, SHARES)
}

/// `build_mix` with other [books, usual, practice] shares (v41: 25 / 70 / 5); the same rule otherwise.
#[allow(clippy::too_many_arguments)]
pub fn build_mix_with(
    books: Vec<Vec<u32>>,
    usual_dir: &Path,
    practice: impl Iterator<Item = Vec<u32>>,
    out: &Path,
    seed: u64,
    shard_tokens: usize,
    known: Option<&HashSet<u64>>,
    shares: [f64; 3],
) -> io::Result<MixReport> {
    let book_tokens: u64 = books.iter().map(|b| b.len() as u64).sum();
    let mut rep = MixReport { targets: targets_with(book_tokens, shares), ..Default::default() };
    let cap = (rep.targets.iter().sum::<u64>() as f64 * 1.001) as usize + (1 << 16);
    let mut pool = Pool { toks: Vec::with_capacity(cap), docs: Vec::new() };
    for b in books {
        if b.is_empty() {
            continue;
        }
        rep.docs[0] += 1;
        rep.tokens[0] += b.len() as u64;
        pool.push(0, b.into_iter())?;
    }
    take_usual(usual_dir, rep.targets[1], &mut pool, known, &mut rep)?;
    for d in practice {
        if rep.tokens[2] >= rep.targets[2] {
            break;
        }
        if d.is_empty() {
            continue;
        }
        rep.docs[2] += 1;
        rep.tokens[2] += d.len() as u64;
        pool.push(2, d.into_iter())?;
    }
    if rep.tokens[2] < rep.targets[2] {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("practice ran out at {} of {} tokens", rep.tokens[2], rep.targets[2]),
        ));
    }
    rep.total = rep.tokens.iter().sum();

    let mut order: Vec<u32> = (0..pool.docs.len() as u32).collect();
    SplitMix64::new(derive_seed(seed, MIX_LABEL, 0)).shuffle(&mut order);
    let mut w = ShardWriter::new(out, VOCAB_GLANCE, shard_tokens)?;
    let mut buf: Vec<u32> = Vec::new();
    for &i in &order {
        let (_, start, len) = pool.docs[i as usize];
        buf.clear();
        buf.extend(pool.toks[start..start + len].iter().map(|&t| t as u32));
        w.push_doc(&buf)?;
    }
    let st = w.finish()?;
    debug_assert_eq!(st.tokens, rep.total);
    rep.shards = st.shards;
    Ok(rep)
}

/// Copy every shard file of `src` into `dst` (created; must hold no shards) byte for byte.
/// Returns the number of files copied.
pub fn copy_shards(src: &Path, dst: &Path) -> io::Result<usize> {
    let set = ShardSet::open(src)?;
    std::fs::create_dir_all(dst)?;
    if !ShardSet::open(dst)?.is_empty() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} already holds shards", dst.display())));
    }
    for s in &set.shards {
        std::fs::copy(&s.path, dst.join(s.path.file_name().unwrap()))?;
    }
    Ok(set.len())
}

/// Write `docs` (marked, EOS-terminated) in the given order as one shard directory.
pub fn write_docs(dir: &Path, docs: &[&[u32]], shard_tokens: usize) -> io::Result<crate::shards::WriterStats> {
    let mut w = ShardWriter::new(dir, VOCAB_GLANCE, shard_tokens)?;
    for d in docs {
        w.push_doc(d)?;
    }
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::PAD;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ava_v40_mix-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Docs whose first token tags the source (books 4000, usual 4001, practice 4002) and whose
    /// second token numbers the doc, so each output doc can be traced back.
    fn docs(tag: u32, n: usize, min: u64, spread: u64, seed: u64) -> Vec<Vec<u32>> {
        let mut r = SplitMix64::new(seed);
        (0..n)
            .map(|i| {
                let len = min + r.below(spread);
                let mut d = vec![tag, (i % 3000) as u32 + 300];
                while (d.len() as u64) < len {
                    let t = r.below(4112) as u32;
                    d.push(if t == EOS || t == PAD || (4000..=4002).contains(&t) { 32 } else { t });
                }
                d.push(EOS);
                d
            })
            .collect()
    }

    fn read_all(dir: &Path) -> Vec<u32> {
        let set = ShardSet::open(dir).unwrap();
        let mut all = Vec::new();
        for i in 0..set.len() {
            all.extend(set.load_shard(i).unwrap().to_vec());
        }
        all
    }

    fn split(stream: &[u32]) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        let mut cur = Vec::new();
        for &t in stream {
            cur.push(t);
            if t == EOS {
                out.push(std::mem::take(&mut cur));
            }
        }
        assert!(cur.is_empty());
        out
    }

    #[test]
    fn targets_keep_the_shares() {
        let t = targets(60_000_000);
        assert_eq!(t, [60_000_000, 56_400_000, 3_600_000]);
        assert_eq!(t.iter().sum::<u64>(), 120_000_000);
        let t = targets(1001);
        assert_eq!(t, [1001, 941, 60]);
    }

    fn build(root: &Path, name: &str, seed: u64, known: Option<&HashSet<u64>>) -> (MixReport, Vec<Vec<u32>>) {
        let books = docs(4000, 12, 2000, 3000, 1);
        let practice = docs(4002, 5000, 20, 60, 3);
        let out = root.join(name);
        let rep = build_mix(books, &root.join("usual"), practice.into_iter(), &out, seed, 7919, known).unwrap();
        (rep, split(&read_all(&out)))
    }

    #[test]
    fn mix_shares_docs_intact_shuffled_and_deterministic() {
        let root = tmp_dir("mix");
        let usual = docs(4001, 3000, 50, 400, 2);
        let mut w = ShardWriter::new(root.join("usual"), VOCAB_GLANCE, 10_007).unwrap();
        for d in &usual {
            w.push_doc(d).unwrap();
        }
        w.finish().unwrap();
        // two usual docs are "known"
        let known: HashSet<u64> = [token_hash(usual[0].iter().copied()), token_hash(usual[5].iter().copied())].into();
        let (rep, out) = build(&root, "a", 7, Some(&known));
        let books = docs(4000, 12, 2000, 3000, 1);
        let practice = docs(4002, 5000, 20, 60, 3);
        // every book once, usual docs in order from the start, practice docs in order from the start
        let mut by_tag: HashMap<u32, Vec<Vec<u32>>> = HashMap::new();
        for d in &out {
            by_tag.entry(d[0]).or_default().push(d.clone());
        }
        let mut got_books = by_tag[&4000].clone();
        got_books.sort();
        let mut want_books = books.clone();
        want_books.sort();
        assert_eq!(got_books, want_books);
        let mut got_u = by_tag[&4001].clone();
        got_u.sort();
        let mut want_u = usual[..rep.docs[1] as usize].to_vec();
        want_u.sort();
        assert_eq!(got_u, want_u);
        let mut got_p = by_tag[&4002].clone();
        got_p.sort();
        let mut want_p = practice[..rep.docs[2] as usize].to_vec();
        want_p.sort();
        assert_eq!(got_p, want_p);
        // token counts and shares within 1% (here: within one document)
        let bt: u64 = books.iter().map(|d| d.len() as u64).sum();
        assert_eq!(rep.tokens[0], bt);
        assert_eq!(rep.total, out.iter().map(|d| d.len() as u64).sum::<u64>());
        for s in 0..3 {
            assert!((rep.share(s) - SHARES[s]).abs() < 0.01, "source {s}: share {}", rep.share(s));
            assert!(rep.tokens[s] >= rep.targets[s]);
        }
        assert!(rep.tokens[1] - rep.targets[1] < 460 && rep.tokens[2] - rep.targets[2] < 81);
        assert_eq!((rep.known_docs, rep.known_tokens), (2, (usual[0].len() + usual[5].len()) as u64));
        // shuffled: the sources are interleaved, not in blocks
        let tags: Vec<u32> = out.iter().map(|d| d[0]).collect();
        let first_book = tags.iter().position(|&t| t == 4000).unwrap();
        assert!(first_book < tags.len() / 4, "books should be spread out");
        // about 94 source changes are expected for these 222 docs; written in blocks there would be 2
        assert!(tags.windows(2).filter(|w| w[0] != w[1]).count() > 50);
        // deterministic per seed; another seed gives another order with the same docs
        let (rep_b, out_b) = build(&root, "b", 7, Some(&known));
        assert_eq!((rep_b, &out_b), (rep.clone(), &out));
        let (rep_c, out_c) = build(&root, "c", 8, Some(&known));
        assert_ne!(out_c, out);
        let (mut s1, mut s2) = (out.clone(), out_c);
        s1.sort();
        s2.sort();
        assert_eq!(s1, s2);
        assert_eq!(rep_c.tokens, rep.tokens);
        // an output directory that already holds shards is refused
        assert!(build_mix(vec![], &root.join("usual"), std::iter::empty(), &root.join("a"), 7, 100, None).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn copy_shards_is_byte_identical() {
        let root = tmp_dir("copy");
        let d = docs(4001, 200, 10, 100, 4);
        let refs: Vec<&[u32]> = d.iter().map(|x| x.as_slice()).collect();
        let st = write_docs(&root.join("src"), &refs, 1000).unwrap();
        assert!(st.shards > 3);
        assert_eq!(copy_shards(&root.join("src"), &root.join("dst")).unwrap(), st.shards);
        for i in 0..st.shards {
            let n = crate::shards::shard_name(i);
            assert_eq!(std::fs::read(root.join("src").join(&n)).unwrap(), std::fs::read(root.join("dst").join(&n)).unwrap());
        }
        assert!(copy_shards(&root.join("src"), &root.join("dst")).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }
}
