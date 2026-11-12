// TESTING / DESIGN ONLY
//! babylm: the BabyLM 2024 strict (100M words) text files packed into documents of about 512-1024 tokens.
//!
//! INPUT. One plain-text file per source and split: `train_100M/<source>.train` and `dev/<source>.dev`.
//! Each line is an utterance or a paragraph, IN ORDER, so consecutive lines are connected text. A line
//! `= = = X = = =` starts a new part (a CHILDES transcript, a Gutenberg book, a Wikipedia article).
//!
//! PACKING. Lines keep their order and are joined with newlines. A document is closed before a line when
//! adding that line would take it past `max_tokens` (1024), and always at a part header (the header line
//! itself is not kept: it is a file path or a book number). A single line longer than `max_tokens` is a
//! document by itself. So most documents hold 900-1024 tokens; the last one of a part may be shorter.
//! Tokens are counted with the BPE: a line's tokens plus one for its newline. Clean-up of a line: `\r`
//! removed, each tab becomes one space, the end trimmed. Empty lines are kept only between two non-empty
//! lines of one document, and a run of them becomes one.
//!
//! EXCLUDED BOOKS. A Gutenberg part whose header is `PG<number>` with a number in the exclusion set is
//! skipped whole. v42 excludes every PG-19 book on disk (`pg19_ids` reads the book numbers of PG-19's
//! MANIFEST.sha256, every split, without ever printing them): those books are her training books, her
//! held-out long-text books, or PG-19's own dev and sealed books, and none may come in by another door.
//!
//! SIMPLE_WIKI. BabyLM's simple_wiki part is an older copy of the Simple English Wikipedia, which she
//! now reads in full from `simplewiki.docs`; so it is dropped (train and dev), not deduplicated.

use crate::bpe::Bpe;
use crate::docs::DocsWriter;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// BabyLM's six parts; the first five are packed.
pub const SOURCES: [&str; 6] = ["childes", "bnc_spoken", "switchboard", "open_subtitles", "gutenberg", "simple_wiki"];
/// The parts v42 packs (simple_wiki is dropped, see the module notes).
pub const PACKED: [&str; 5] = ["childes", "bnc_spoken", "switchboard", "open_subtitles", "gutenberg"];
pub const DEFAULT_MAX_TOKENS: usize = 1024;

/// The title of a part header line `= = = X = = =`.
pub fn header(line: &str) -> Option<&str> {
    line.trim().strip_prefix("= = = ")?.strip_suffix(" = = =").map(str::trim)
}

/// The book number of a Gutenberg header title `PG12345`.
pub fn pg_id(title: &str) -> Option<u64> {
    title.strip_prefix("PG")?.parse().ok()
}

/// The book numbers of every `<dir>/<number>.txt` named in a sha256sum-format manifest.
pub fn pg19_ids(manifest: &str) -> HashSet<u64> {
    let mut s = HashSet::new();
    for line in manifest.lines() {
        let Some(path) = line.split_whitespace().nth(1) else { continue };
        let name = path.trim_start_matches('*').rsplit('/').next().unwrap_or("");
        if let Some(n) = name.strip_suffix(".txt").and_then(|n| n.parse::<u64>().ok()) {
            s.insert(n);
        }
    }
    s
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PackStats {
    pub lines: u64,
    pub parts: u64,
    pub skipped_parts: u64,
    pub skipped_lines: u64,
    pub docs: u64,
    /// Counted tokens (each line + 1); the shard builder's exact count differs slightly.
    pub tokens: u64,
    pub long_lines: u64,
    /// Documents of 512..=max tokens.
    pub in_range: u64,
    pub min_doc: u64,
    pub max_doc: u64,
}

/// Packs lines into documents (see PACKING).
pub struct LinePacker<'a> {
    bpe: &'a Bpe,
    exclude: &'a HashSet<u64>,
    max_tokens: usize,
    cur: String,
    cur_tokens: usize,
    blank: bool,
    skipping: bool,
    ids: Vec<u32>,
    pub stats: PackStats,
}

impl<'a> LinePacker<'a> {
    pub fn new(bpe: &'a Bpe, exclude: &'a HashSet<u64>, max_tokens: usize) -> LinePacker<'a> {
        LinePacker { bpe, exclude, max_tokens: max_tokens.max(1), cur: String::new(), cur_tokens: 0, blank: false, skipping: false, ids: Vec::new(), stats: PackStats::default() }
    }

    fn close(&mut self, out: &mut dyn FnMut(&str) -> io::Result<()>) -> io::Result<()> {
        if !self.cur.is_empty() {
            out(&self.cur)?;
            let t = self.cur_tokens as u64;
            let s = &mut self.stats;
            s.docs += 1;
            s.tokens += t;
            s.in_range += (512..=self.max_tokens as u64).contains(&t) as u64;
            s.min_doc = if s.docs == 1 { t } else { s.min_doc.min(t) };
            s.max_doc = s.max_doc.max(t);
        }
        self.cur.clear();
        self.cur_tokens = 0;
        self.blank = false;
        Ok(())
    }

    /// Take one raw line (with or without its line end).
    pub fn push_line(&mut self, raw: &str, out: &mut dyn FnMut(&str) -> io::Result<()>) -> io::Result<()> {
        self.stats.lines += 1;
        let line = raw.trim_end_matches(['\n', '\r']).replace('\t', " ");
        let line = line.trim_end();
        if let Some(title) = header(line) {
            self.close(out)?;
            self.stats.parts += 1;
            self.skipping = pg_id(title).is_some_and(|n| self.exclude.contains(&n));
            self.stats.skipped_parts += self.skipping as u64;
            return Ok(());
        }
        if self.skipping {
            self.stats.skipped_lines += 1;
            return Ok(());
        }
        if line.trim().is_empty() {
            self.blank = !self.cur.is_empty();
            return Ok(());
        }
        self.ids.clear();
        self.bpe.encode_into(line, &mut self.ids);
        let t = self.ids.len() + 1;
        if t > self.max_tokens {
            self.stats.long_lines += 1;
        }
        if self.cur_tokens > 0 && self.cur_tokens + t > self.max_tokens {
            self.close(out)?;
        }
        if !self.cur.is_empty() {
            self.cur.push('\n');
            if self.blank {
                self.cur.push('\n');
                self.cur_tokens += 1;
            }
        }
        self.blank = false;
        self.cur.push_str(line);
        self.cur_tokens += t;
        Ok(())
    }

    pub fn finish(&mut self, out: &mut dyn FnMut(&str) -> io::Result<()>) -> io::Result<()> {
        self.close(out)
    }
}

/// Pack one BabyLM file into a docs file (written to `.tmp`, moved into place at the end).
pub fn pack_file(input: &Path, output: &Path, bpe: &Bpe, exclude: &HashSet<u64>, max_tokens: usize) -> io::Result<PackStats> {
    let f = File::open(input).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", input.display())))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut w = DocsWriter::create(output)?;
    let mut p = LinePacker::new(bpe, exclude, max_tokens);
    let mut buf = Vec::new();
    {
        let mut out = |d: &str| w.write_doc(d);
        loop {
            buf.clear();
            if r.read_until(b'\n', &mut buf)? == 0 {
                break;
            }
            let line = String::from_utf8_lossy(&buf);
            p.push_line(&line, &mut out)?;
        }
        p.finish(&mut out)?;
    }
    w.finish()?;
    Ok(p.stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(lines: &[&str], max: usize, exclude: &HashSet<u64>) -> (Vec<String>, PackStats) {
        let bpe = Bpe::from_merges(Vec::new()).unwrap();
        let mut docs = Vec::new();
        let mut p = LinePacker::new(&bpe, exclude, max);
        {
            let mut out = |d: &str| {
                docs.push(d.to_string());
                Ok(())
            };
            for l in lines {
                p.push_line(l, &mut out).unwrap();
            }
            p.finish(&mut out).unwrap();
        }
        (docs, p.stats)
    }

    #[test]
    fn headers_and_ids() {
        assert_eq!(header("= = = PG21374 = = =\r"), Some("PG21374"));
        assert_eq!(header("= = = childes/CHILDES_NA/Brown/Eve/010800.cha = = ="), Some("childes/CHILDES_NA/Brown/Eve/010800.cha"));
        assert_eq!(header("= = no"), None);
        assert_eq!(pg_id("PG21374"), Some(21374));
        assert_eq!(pg_id("Champmotteux"), None);
        let ids = pg19_ids("abc  *train/10520.txt\ndef  *validation/77.txt\nxyz  *LICENSE\n");
        assert_eq!(ids, [10520u64, 77].into_iter().collect());
    }

    #[test]
    fn packs_in_order_up_to_the_budget_and_splits_at_headers() {
        // with no merges every byte is a token: "ab" = 2 tokens + 1 for the newline
        let none = HashSet::new();
        let (docs, _) = pack(&["ab", "cd\r\n", "ef"], 6, &none);
        assert_eq!(docs, vec!["ab\ncd", "ef"]);
        let (docs, st) = pack(&["x", "= = = part two = = =", "gh\tij", "", "", "kl"], 100, &none);
        assert_eq!(docs, vec!["x", "gh ij\n\nkl"]);
        assert_eq!((st.parts, st.docs, st.lines), (1, 2, 6));
        // order is kept: joining the documents gives the lines back
        let (docs, _) = pack(&["one", "two", "three", "four", "five"], 9, &none);
        assert_eq!(docs.join("\n").split('\n').collect::<Vec<_>>(), vec!["one", "two", "three", "four", "five"]);
        assert!(docs.len() > 1);
        // a too-long line is a document by itself; a leading empty line is not kept
        let (docs, st) = pack(&["", "x", "abcdefghijk", "y"], 6, &none);
        assert_eq!(docs, vec!["x", "abcdefghijk", "y"]);
        assert_eq!(st.long_lines, 1);
    }

    #[test]
    fn excluded_books_are_skipped_whole() {
        let ex: HashSet<u64> = [7u64].into_iter().collect();
        let (docs, st) = pack(&["= = = PG5 = = =", "keep me", "= = = PG7 = = =", "drop me", "and me", "= = = PG9 = = =", "keep too"], 100, &ex);
        assert_eq!(docs, vec!["keep me", "keep too"]);
        assert_eq!((st.skipped_parts, st.skipped_lines), (1, 2));
    }
}
