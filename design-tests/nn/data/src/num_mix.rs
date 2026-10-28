//! TESTING / DESIGN ONLY — num_mix: rewrite one shard directory with length marks and/or practice docs.
//!
//! `rewrite(src, out, opts)` reads the source shards in file order (one shard in memory at a
//! time), reassembles the documents by splitting at EOS (a document may span a shard boundary;
//! the partial document is carried over), and writes them to `out` through a `ShardWriter`:
//!
//! - with `marks`, every document (source and practice) goes through `insert_marks`, and the
//!   output vocab is `VOCAB_GLANCE` (4112); without, it keeps the source's vocab (4096);
//! - after each source document, while `practice_tokens < practice_frac x tokens written so
//!   far` (practice included), the next practice document is written, cycling through the
//!   list when it runs out (`practice_cycles` counts the restarts), so the practice share of
//!   the output ends in `[frac, frac + one practice doc / total)`;
//! - a trailing partial document without EOS at the very end is written as is (marked when
//!   `marks`, no EOS added, no practice after it);
//! - the output's shards hold at most as many tokens as the source's first shard, so a rewrite
//!   with neither marks nor practice reproduces the source byte for byte, shard sizes included.

use crate::bpe::EOS;
use crate::numbers::{insert_marks_into, MARK_BASE, VOCAB_GLANCE};
use crate::shards::{ShardSet, ShardTokens, ShardWriter};
use std::io;
use std::path::Path;

/// What to do to the stream. `practice` docs are already encoded (`Bpe::encode_doc`, EOS
/// included) and unmarked.
#[derive(Clone, Debug, Default)]
pub struct MixOpts {
    pub marks: bool,
    pub practice: Option<Vec<Vec<u32>>>,
    pub practice_frac: f64,
}

/// `src_tokens`: tokens read from the source. `practice_tokens`: practice tokens written
/// (marks included). `marks`: marks inserted in all (source and practice). `practice_cycles`:
/// how often the practice list ran out and restarted. `out_tokens`: tokens written in all.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MixStats {
    pub src_tokens: u64,
    pub practice_tokens: u64,
    pub marks: u64,
    pub practice_cycles: u64,
    pub out_tokens: u64,
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

struct Out<'a> {
    w: ShardWriter,
    marks: bool,
    buf: Vec<u32>,
    practice: &'a [Vec<u32>],
    frac: f64,
    next: usize,
    stats: MixStats,
}

impl Out<'_> {
    /// Write one document (marked if asked); returns the tokens written.
    fn push(&mut self, doc: &[u32]) -> io::Result<u64> {
        if self.marks {
            self.buf.clear();
            self.stats.marks += insert_marks_into(doc, &mut self.buf);
            self.w.push_doc(&self.buf)?;
            Ok(self.buf.len() as u64)
        } else {
            self.w.push_doc(doc)?;
            Ok(doc.len() as u64)
        }
    }

    fn source_doc(&mut self, doc: &[u32], complete: bool) -> io::Result<()> {
        let n = self.push(doc)?;
        self.stats.src_tokens += doc.len() as u64;
        self.stats.out_tokens += n;
        if complete {
            self.top_up()?;
        }
        Ok(())
    }

    fn top_up(&mut self) -> io::Result<()> {
        let practice = self.practice;
        if practice.is_empty() || self.frac <= 0.0 {
            return Ok(());
        }
        while (self.stats.practice_tokens as f64) < self.frac * self.stats.out_tokens as f64 {
            if self.next == practice.len() {
                self.next = 0;
                self.stats.practice_cycles += 1;
            }
            let n = self.push(&practice[self.next])?;
            self.next += 1;
            self.stats.practice_tokens += n;
            self.stats.out_tokens += n;
        }
        Ok(())
    }
}

fn scan(tokens: impl Iterator<Item = u32>, doc: &mut Vec<u32>, out: &mut Out) -> io::Result<()> {
    for t in tokens {
        doc.push(t);
        if t == EOS {
            out.source_doc(doc, true)?;
            doc.clear();
        }
    }
    Ok(())
}

/// Rewrite the shard directory `src` into `out` (which must hold no shards yet). See the
/// module docs.
pub fn rewrite(src: &Path, out: &Path, opts: &MixOpts) -> io::Result<MixStats> {
    let set = ShardSet::open(src)?;
    if set.is_empty() {
        return Err(invalid(format!("{}: no shards", src.display())));
    }
    if opts.marks && set.vocab_size > MARK_BASE {
        return Err(invalid(format!(
            "{}: vocab {} overlaps the marks (ids from {MARK_BASE})",
            src.display(),
            set.vocab_size
        )));
    }
    let practice: &[Vec<u32>] = opts.practice.as_deref().unwrap_or(&[]);
    if opts.practice.is_some() {
        if !(0.0..1.0).contains(&opts.practice_frac) {
            return Err(invalid(format!("practice_frac {} must be in [0, 1)", opts.practice_frac)));
        }
        if practice.is_empty() && opts.practice_frac > 0.0 {
            return Err(invalid("practice_frac > 0 but no practice docs".into()));
        }
        if practice.iter().any(|d| d.is_empty()) {
            return Err(invalid("an empty practice doc".into()));
        }
    }
    let vocab = if opts.marks { VOCAB_GLANCE } else { set.vocab_size as usize };
    let max_tokens = set.shards[0].header.n_tokens as usize;
    let mut o = Out {
        w: ShardWriter::new(out, vocab, max_tokens)?,
        marks: opts.marks,
        buf: Vec::new(),
        practice,
        frac: opts.practice_frac,
        next: 0,
        stats: MixStats::default(),
    };
    let mut doc: Vec<u32> = Vec::new();
    for i in 0..set.len() {
        let tokens = set.load_shard(i)?;
        match &tokens {
            ShardTokens::U16(v) => scan(v.iter().map(|&t| t as u32), &mut doc, &mut o)?,
            ShardTokens::U32(v) => scan(v.iter().copied(), &mut doc, &mut o)?,
        }
    }
    if !doc.is_empty() {
        o.source_doc(&doc, false)?;
    }
    let stats = o.stats;
    let written = o.w.finish()?;
    debug_assert_eq!(written.tokens, stats.out_tokens);
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numbers::strip_marks;
    use crate::rng::SplitMix64;
    use std::path::PathBuf;

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ava_num_mix-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Random docs over ids 0..4096 (never EOS or PAD inside), digit runs included, each
    /// ending in EOS.
    fn random_docs(seed: u64, n: usize) -> Vec<Vec<u32>> {
        let mut r = SplitMix64::new(seed);
        (0..n)
            .map(|_| {
                let len = r.below(40) as usize;
                let mut d: Vec<u32> = Vec::with_capacity(len + 1);
                while d.len() < len {
                    if r.below(3) == 0 {
                        for _ in 0..1 + r.below(5) {
                            d.push(48 + r.below(10) as u32);
                        }
                    } else {
                        let t = r.below(4000) as u32;
                        d.push(if t == EOS || t == crate::bpe::PAD { 32 } else { t });
                    }
                }
                d.push(EOS);
                d
            })
            .collect()
    }

    fn write_src(dir: &Path, docs: &[Vec<u32>], tail: &[u32], shard_tokens: usize) {
        let mut w = ShardWriter::new(dir, 4096, shard_tokens).unwrap();
        for d in docs {
            w.push_doc(d).unwrap();
        }
        if !tail.is_empty() {
            w.push_doc(tail).unwrap();
        }
        w.finish().unwrap();
    }

    fn read_all(dir: &Path) -> (Vec<u32>, Vec<u64>, u32) {
        let set = ShardSet::open(dir).unwrap();
        let mut all = Vec::new();
        for i in 0..set.len() {
            all.extend(set.load_shard(i).unwrap().to_vec());
        }
        (all, set.shards.iter().map(|s| s.header.n_tokens).collect(), set.vocab_size)
    }

    fn split_docs(stream: &[u32]) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        let mut cur = Vec::new();
        for &t in stream {
            cur.push(t);
            if t == EOS {
                out.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    #[test]
    fn noop_rewrite_reproduces_the_source() {
        let root = tmp_dir("noop");
        let (src, out) = (root.join("src"), root.join("out"));
        let docs = random_docs(1, 300);
        write_src(&src, &docs, &[], 97);
        let st = rewrite(&src, &out, &MixOpts::default()).unwrap();
        let (a, sa, va) = read_all(&src);
        let (b, sb, vb) = read_all(&out);
        assert!(sa.len() > 10, "docs should span many shards");
        assert_eq!(a, b);
        assert_eq!(sa, sb);
        assert_eq!(va, vb);
        assert_eq!(st.src_tokens, a.len() as u64);
        assert_eq!(st.out_tokens, a.len() as u64);
        assert_eq!((st.marks, st.practice_tokens, st.practice_cycles), (0, 0, 0));
        // the shard files are byte-identical too
        for i in 0..sa.len() {
            let name = crate::shards::shard_name(i);
            assert_eq!(std::fs::read(src.join(&name)).unwrap(), std::fs::read(out.join(&name)).unwrap());
        }
        // out must be empty: a second rewrite into it fails
        assert!(rewrite(&src, &out, &MixOpts::default()).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn marks_strip_back_to_the_source() {
        let root = tmp_dir("marks");
        let (src, out) = (root.join("src"), root.join("out"));
        let docs = random_docs(2, 300);
        let tail = vec![49u32, 50, 32, 51]; // trailing partial doc, no EOS
        write_src(&src, &docs, &tail, 64);
        let st = rewrite(&src, &out, &MixOpts { marks: true, ..Default::default() }).unwrap();
        let (a, _, _) = read_all(&src);
        let (b, sb, vb) = read_all(&out);
        assert_eq!(vb, VOCAB_GLANCE as u32);
        assert_eq!(strip_marks(&b), a);
        assert_eq!(b, crate::numbers::insert_marks(&a));
        assert_eq!(st.marks, (b.len() - a.len()) as u64);
        assert!(st.marks > 100);
        assert_eq!(st.out_tokens, b.len() as u64);
        assert!(sb[..sb.len() - 1].iter().all(|&n| n == 64));
        assert_eq!(&b[b.len() - 6..], &[4097, 49, 50, 32, 4096, 51]);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Practice docs start with id 4000 (never at the start of a source doc here).
    fn practice_list(n: usize) -> Vec<Vec<u32>> {
        random_docs(9, n)
            .into_iter()
            .enumerate()
            .map(|(i, mut d)| {
                d.insert(0, 4000);
                d.insert(1, 4001 + (i as u32 % 50));
                d
            })
            .collect()
    }

    fn check_practice(marks: bool) {
        let root = tmp_dir(if marks { "practice-marks" } else { "practice" });
        let (src, out) = (root.join("src"), root.join("out"));
        let docs = random_docs(3, 400); // ids < 4000, so 4000 marks a practice doc
        write_src(&src, &docs, &[], 128);
        let practice = practice_list(17);
        let frac = 0.25;
        let opts = MixOpts { marks, practice: Some(practice.clone()), practice_frac: frac };
        let st = rewrite(&src, &out, &opts).unwrap();
        let (b, _, vb) = read_all(&out);
        assert_eq!(vb as usize, if marks { VOCAB_GLANCE } else { 4096 });
        assert_eq!(st.out_tokens, b.len() as u64);
        let plain = if marks { strip_marks(&b) } else { b.clone() };
        let out_docs = split_docs(&plain);
        let (mut got_src, mut got_pr) = (Vec::new(), Vec::new());
        let mut pr_tokens = 0u64;
        for d in split_docs(&b) {
            let s = if marks { strip_marks(&d) } else { d.clone() };
            if s[0] == 4000 {
                pr_tokens += d.len() as u64;
                got_pr.push(s);
            } else {
                got_src.push(s);
            }
        }
        assert_eq!(out_docs.len(), got_src.len() + got_pr.len());
        // source docs intact and in order
        assert_eq!(got_src, docs);
        // practice docs intact, in list order, cycling
        assert!(got_pr.len() > practice.len(), "the list should cycle in this test");
        for (i, d) in got_pr.iter().enumerate() {
            assert_eq!(d, &practice[i % practice.len()]);
        }
        assert_eq!(st.practice_cycles, ((got_pr.len() - 1) / practice.len()) as u64);
        assert_eq!(pr_tokens, st.practice_tokens);
        // share in [frac, frac + one doc / total)
        let share = st.practice_tokens as f64 / st.out_tokens as f64;
        let max_doc = practice.iter().map(|d| crate::numbers::insert_marks(d).len()).max().unwrap() as f64;
        assert!(share >= frac - 1e-12, "share {share}");
        assert!(share < frac + max_doc / st.out_tokens as f64, "share {share}");
        // out = source + source marks + practice (its marks included)
        let all_marks = b.iter().filter(|&&t| crate::numbers::is_mark(t)).count() as u64;
        let practice_marks = pr_tokens - got_pr.iter().map(|d| d.len() as u64).sum::<u64>();
        assert_eq!(st.marks, all_marks);
        assert_eq!(st.src_tokens + (st.marks - practice_marks) + st.practice_tokens, st.out_tokens);
        assert_eq!(st.src_tokens, docs.iter().map(|d| d.len() as u64).sum::<u64>());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn practice_quarter_share_docs_intact() {
        check_practice(false);
    }

    #[test]
    fn practice_with_marks() {
        check_practice(true);
    }

    #[test]
    fn bad_options_are_refused() {
        let root = tmp_dir("bad");
        let src = root.join("src");
        write_src(&src, &random_docs(4, 20), &[], 50);
        let bad = |frac: f64, p: Vec<Vec<u32>>| MixOpts { marks: false, practice: Some(p), practice_frac: frac };
        assert!(rewrite(&src, &root.join("o1"), &bad(1.0, practice_list(3))).is_err());
        assert!(rewrite(&src, &root.join("o2"), &bad(0.1, vec![])).is_err());
        assert!(rewrite(&src, &root.join("o3"), &bad(0.1, vec![vec![]])).is_err());
        assert!(rewrite(&root.join("nothing"), &root.join("o4"), &MixOpts::default()).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }
}
