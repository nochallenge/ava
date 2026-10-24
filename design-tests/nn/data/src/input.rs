//! TESTING / DESIGN ONLY — input specs and a streaming document reader (shared by the binaries).
//!
//! An input spec names a text file and which part of each line is the document:
//!   `path`              the whole line
//!   `path@tab:N`        0-based tab-separated field N
//!   `path@sep:STR:N`    fields separated by the literal string STR, field N; a negative N
//!                       counts from the end (-1 = last field). STR may contain spaces and
//!                       colons: the number after the LAST ':' is N.
//!
//! Each line is one document. Lines are read as bytes, one at a time (never the whole file),
//! the line terminator (`\n` or `\r\n`) is stripped, the bytes are converted with
//! `from_utf8_lossy`, the field is extracted and trimmed of surrounding whitespace. Lines whose
//! field is missing or empty after trimming are skipped (counted in `skipped`). A line longer
//! than `MAX_LINE_BYTES` keeps only its first `MAX_LINE_BYTES` bytes (counted in `truncated`),
//! so memory stays bounded even if a file has no newlines at all.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;

/// Longest document line kept, in bytes (longer lines are truncated to this).
pub const MAX_LINE_BYTES: usize = 1 << 20;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Field {
    Whole,
    Tab(usize),
    Sep(String, i64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputSpec {
    /// The spec as given.
    pub spec: String,
    pub path: PathBuf,
    pub field: Field,
}

impl InputSpec {
    pub fn parse(s: &str) -> Result<InputSpec, String> {
        let err = |m: &str| Err(format!("bad input spec {s:?}: {m}"));
        if let Some(pos) = s.rfind("@tab:") {
            let (path, n) = (&s[..pos], &s[pos + 5..]);
            let Ok(n) = n.parse::<usize>() else { return err("@tab:N needs a non-negative integer N") };
            if path.is_empty() {
                return err("empty path");
            }
            return Ok(InputSpec { spec: s.to_string(), path: PathBuf::from(path), field: Field::Tab(n) });
        }
        if let Some(pos) = s.find("@sep:") {
            let (path, rest) = (&s[..pos], &s[pos + 5..]);
            let Some(colon) = rest.rfind(':') else { return err("@sep:STR:N needs ':N' at the end") };
            let (sep, n) = (&rest[..colon], &rest[colon + 1..]);
            let Ok(n) = n.parse::<i64>() else { return err("@sep:STR:N needs an integer N") };
            if sep.is_empty() {
                return err("empty separator");
            }
            if path.is_empty() {
                return err("empty path");
            }
            return Ok(InputSpec { spec: s.to_string(), path: PathBuf::from(path), field: Field::Sep(sep.to_string(), n) });
        }
        if s.is_empty() {
            return err("empty path");
        }
        Ok(InputSpec { spec: s.to_string(), path: PathBuf::from(s), field: Field::Whole })
    }

    /// The document text in `line` (terminator already stripped), trimmed; `None` when the
    /// field is missing or empty.
    pub fn extract<'a>(&self, line: &'a str) -> Option<&'a str> {
        let field = match &self.field {
            Field::Whole => Some(line),
            Field::Tab(n) => line.split('\t').nth(*n),
            Field::Sep(sep, n) => {
                if *n >= 0 {
                    line.split(sep.as_str()).nth(*n as usize)
                } else {
                    line.rsplit(sep.as_str()).nth((-(*n + 1)) as usize)
                }
            }
        }?;
        let t = field.trim();
        if t.is_empty() {
            None
        } else {
            Some(t)
        }
    }

    /// Open the file for streaming (1 MiB read buffer).
    pub fn open(&self) -> io::Result<DocReader<BufReader<File>>> {
        let f = File::open(&self.path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", self.path.display())))?;
        Ok(DocReader::new(self.clone(), BufReader::with_capacity(1 << 20, f)))
    }
}

/// Read one line (through `\n`) into `buf`, keeping at most `cap` bytes; the rest of an
/// over-long line is consumed and dropped. Returns (bytes consumed, whether bytes were dropped).
fn read_line_capped<R: BufRead>(r: &mut R, buf: &mut Vec<u8>, cap: usize) -> io::Result<(u64, bool)> {
    buf.clear();
    let mut consumed = 0u64;
    let mut dropped = false;
    loop {
        let avail = match r.fill_buf() {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if avail.is_empty() {
            break;
        }
        let (take, done) = match avail.iter().position(|&b| b == b'\n') {
            Some(i) => (i + 1, true),
            None => (avail.len(), false),
        };
        let room = cap.saturating_sub(buf.len());
        if take > room {
            dropped = true;
        }
        buf.extend_from_slice(&avail[..take.min(room)]);
        r.consume(take);
        consumed += take as u64;
        if done {
            break;
        }
    }
    Ok((consumed, dropped))
}

/// Streams documents (one per line) out of any `BufRead`.
pub struct DocReader<R: BufRead> {
    spec: InputSpec,
    reader: R,
    buf: Vec<u8>,
    /// Bytes consumed from the file, terminators included.
    pub raw_bytes: u64,
    /// Lines read.
    pub lines: u64,
    /// Documents returned.
    pub docs: u64,
    /// Bytes of the returned document texts (UTF-8, after lossy conversion and trimming).
    pub text_bytes: u64,
    /// Lines skipped (field missing or empty).
    pub skipped: u64,
    /// Lines cut to `MAX_LINE_BYTES`.
    pub truncated: u64,
    /// End of input reached.
    pub done: bool,
}

impl<R: BufRead> DocReader<R> {
    pub fn new(spec: InputSpec, reader: R) -> DocReader<R> {
        DocReader { spec, reader, buf: Vec::new(), raw_bytes: 0, lines: 0, docs: 0, text_bytes: 0, skipped: 0, truncated: 0, done: false }
    }

    pub fn spec(&self) -> &InputSpec {
        &self.spec
    }

    /// The next non-empty document, or `None` at end of input.
    pub fn next_doc(&mut self) -> io::Result<Option<String>> {
        loop {
            if self.done {
                return Ok(None);
            }
            // room for the content plus a "\r\n" terminator
            let (consumed, dropped) = read_line_capped(&mut self.reader, &mut self.buf, MAX_LINE_BYTES + 2)?;
            if consumed == 0 {
                self.done = true;
                return Ok(None);
            }
            self.raw_bytes += consumed;
            self.lines += 1;
            let mut end = self.buf.len();
            if !dropped {
                if end > 0 && self.buf[end - 1] == b'\n' {
                    end -= 1;
                }
                if end > 0 && self.buf[end - 1] == b'\r' {
                    end -= 1;
                }
            }
            if end > MAX_LINE_BYTES || dropped {
                end = end.min(MAX_LINE_BYTES);
                self.truncated += 1;
            }
            let line = String::from_utf8_lossy(&self.buf[..end]);
            match self.spec.extract(&line) {
                Some(t) => {
                    self.docs += 1;
                    self.text_bytes += t.len() as u64;
                    return Ok(Some(t.to_string()));
                }
                None => self.skipped += 1,
            }
        }
    }
}

/// Parse a count: digits with optional `_` separators and an optional decimal suffix
/// k = 10^3, M = 10^6, G = 10^9, T = 10^12 (so "200M" = 200_000_000).
pub fn parse_count(s: &str) -> Result<u64, String> {
    let t: String = s.trim().chars().filter(|&c| c != '_').collect();
    let (num, mult) = match t.chars().last() {
        Some('k') | Some('K') => (&t[..t.len() - 1], 1_000u64),
        Some('M') | Some('m') => (&t[..t.len() - 1], 1_000_000),
        Some('G') | Some('g') => (&t[..t.len() - 1], 1_000_000_000),
        Some('T') | Some('t') => (&t[..t.len() - 1], 1_000_000_000_000),
        _ => (&t[..], 1),
    };
    let n: u64 = num.parse().map_err(|_| format!("bad count {s:?}"))?;
    n.checked_mul(mult).ok_or_else(|| format!("count {s:?} overflows"))
}

/// How `Mixer` picks the source of the next document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixPolicy {
    /// The source with the fewest text bytes read so far (a byte-fair sample: every source
    /// contributes equally until it runs out). Used for tokenizer training samples.
    EqualBytes,
    /// The source with the smallest fraction read, where the fraction is
    /// `max(raw_bytes / file_size, docs / max_docs)`. All sources then progress at the same
    /// relative rate and finish together, so every stretch of the output (every shard) holds
    /// roughly the global mixture. Used for writing shards.
    Proportional,
}

/// Several inputs read as one interleaved stream of documents. Ties go to the first source.
pub struct Mixer {
    readers: Vec<DocReader<BufReader<File>>>,
    file_sizes: Vec<u64>,
    policy: MixPolicy,
    max_docs: Option<u64>,
}

impl Mixer {
    pub fn open(specs: &[InputSpec], policy: MixPolicy, max_docs_per_input: Option<u64>) -> io::Result<Mixer> {
        let mut readers = Vec::new();
        let mut file_sizes = Vec::new();
        for s in specs {
            readers.push(s.open()?);
            file_sizes.push(std::fs::metadata(&s.path)?.len());
        }
        Ok(Mixer { readers, file_sizes, policy, max_docs: max_docs_per_input })
    }

    pub fn readers(&self) -> &[DocReader<BufReader<File>>] {
        &self.readers
    }

    pub fn file_size(&self, i: usize) -> u64 {
        self.file_sizes[i]
    }

    /// Source `i` is at end of file or has reached `max_docs_per_input`.
    pub fn exhausted(&self, i: usize) -> bool {
        let r = &self.readers[i];
        r.done || self.max_docs.is_some_and(|m| r.docs >= m)
    }

    fn progress(&self, i: usize) -> f64 {
        let r = &self.readers[i];
        match self.policy {
            MixPolicy::EqualBytes => r.text_bytes as f64,
            MixPolicy::Proportional => {
                let by_bytes = r.raw_bytes as f64 / self.file_sizes[i].max(1) as f64;
                let by_docs = self.max_docs.map_or(0.0, |m| r.docs as f64 / m.max(1) as f64);
                by_bytes.max(by_docs)
            }
        }
    }

    /// The next document and the index of its source, or `None` when every source is
    /// exhausted.
    pub fn next_doc(&mut self) -> io::Result<Option<(usize, String)>> {
        loop {
            let mut best: Option<(usize, f64)> = None;
            for i in 0..self.readers.len() {
                if self.exhausted(i) {
                    continue;
                }
                let p = self.progress(i);
                if best.is_none_or(|(_, bp)| p < bp) {
                    best = Some((i, p));
                }
            }
            let Some((i, _)) = best else { return Ok(None) };
            if let Some(doc) = self.readers[i].next_doc()? {
                return Ok(Some((i, doc)));
            }
            // source i hit end of file: pick again
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parse_specs() {
        let a = InputSpec::parse("a.txt").unwrap();
        assert_eq!(a.path, PathBuf::from("a.txt"));
        assert_eq!(a.field, Field::Whole);
        let b = InputSpec::parse(r"C:\x\y.tsv@tab:2").unwrap();
        assert_eq!(b.path, PathBuf::from(r"C:\x\y.tsv"));
        assert_eq!(b.field, Field::Tab(2));
        let c = InputSpec::parse("c.txt@sep: +++$+++ :-1").unwrap();
        assert_eq!(c.path, PathBuf::from("c.txt"));
        assert_eq!(c.field, Field::Sep(" +++$+++ ".into(), -1));
        let d = InputSpec::parse("d@x.txt@sep:a:b:1").unwrap();
        assert_eq!(d.path, PathBuf::from("d@x.txt"));
        assert_eq!(d.field, Field::Sep("a:b".into(), 1));
        assert_eq!(d.spec, "d@x.txt@sep:a:b:1");
        for bad in ["", "x@tab:", "x@tab:abc", "x@tab:-1", "x@sep::2", "x@sep:ab", "x@sep:a:b", "@tab:1"] {
            assert!(InputSpec::parse(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn extract_fields() {
        let t = InputSpec::parse("f@tab:2").unwrap();
        assert_eq!(t.extract("1\teng\tHello there."), Some("Hello there."));
        assert_eq!(t.extract("1\teng\t  padded  "), Some("padded"));
        assert_eq!(t.extract("1\teng"), None);
        assert_eq!(t.extract("1\teng\t   "), None);
        let w = InputSpec::parse("f").unwrap();
        assert_eq!(w.extract(" whole line "), Some("whole line"));
        assert_eq!(w.extract(""), None);
        let s = InputSpec::parse("f@sep: +++$+++ :-1").unwrap();
        let line = "L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!";
        assert_eq!(s.extract(line), Some("They do not!"));
        let s2 = InputSpec::parse("f@sep: +++$+++ :-2").unwrap();
        assert_eq!(s2.extract(line), Some("BIANCA"));
        let s0 = InputSpec::parse("f@sep: +++$+++ :0").unwrap();
        assert_eq!(s0.extract(line), Some("L1045"));
        let s9 = InputSpec::parse("f@sep: +++$+++ :9").unwrap();
        assert_eq!(s9.extract(line), None);
        let sm = InputSpec::parse("f@sep: +++$+++ :-9").unwrap();
        assert_eq!(sm.extract(line), None);
    }

    #[test]
    fn doc_reader_streams_lines() {
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(b"1\teng\tFirst.\r\n");
        data.extend_from_slice(b"2\teng\t\n"); // empty text: skipped
        data.extend_from_slice(b"\n"); // blank line: skipped
        data.extend_from_slice(b"3\teng\tBad \xff byte\n"); // lossy
        data.extend_from_slice(b"4\teng\n"); // missing field: skipped
        data.extend_from_slice(b"5\teng\tLast, no newline");
        let spec = InputSpec::parse("mem@tab:2").unwrap();
        let n = data.len() as u64;
        let mut r = DocReader::new(spec, Cursor::new(data));
        let mut docs = vec![];
        while let Some(d) = r.next_doc().unwrap() {
            docs.push(d);
        }
        assert_eq!(docs, vec!["First.", "Bad \u{fffd} byte", "Last, no newline"]);
        assert_eq!(r.lines, 6);
        assert_eq!(r.docs, 3);
        assert_eq!(r.skipped, 3);
        assert_eq!(r.raw_bytes, n);
        assert_eq!(r.text_bytes, (6 + 12 + 16) as u64);
        assert!(r.done);
        assert!(r.next_doc().unwrap().is_none());
    }

    #[test]
    fn doc_reader_caps_long_lines() {
        let mut data = vec![b'a'; MAX_LINE_BYTES + 100];
        data.push(b'\n');
        data.extend_from_slice(b"next\n");
        let mut r = DocReader::new(InputSpec::parse("mem").unwrap(), Cursor::new(data));
        let first = r.next_doc().unwrap().unwrap();
        assert_eq!(first.len(), MAX_LINE_BYTES);
        assert_eq!(r.truncated, 1);
        assert_eq!(r.next_doc().unwrap().unwrap(), "next");
        assert!(r.next_doc().unwrap().is_none());
        assert_eq!(r.raw_bytes, (MAX_LINE_BYTES + 100 + 1 + 5) as u64);
    }

    #[test]
    fn parse_counts() {
        assert_eq!(parse_count("12").unwrap(), 12);
        assert_eq!(parse_count("50_000_000").unwrap(), 50_000_000);
        assert_eq!(parse_count("1k").unwrap(), 1_000);
        assert_eq!(parse_count("200M").unwrap(), 200_000_000);
        assert_eq!(parse_count("3G").unwrap(), 3_000_000_000);
        assert!(parse_count("").is_err());
        assert!(parse_count("x").is_err());
        assert!(parse_count("-5").is_err());
    }

    fn mixer_files(name: &str) -> (InputSpec, InputSpec) {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-tmp")
            .join(format!("input-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let a = d.join("a.txt");
        let b = d.join("b.txt");
        std::fs::write(&a, "aaaa\n".repeat(10)).unwrap(); // 10 docs x 4 text bytes, 50 raw bytes
        std::fs::write(&b, "bbbbbbbbbbbbbbbbbbbb\n".repeat(3)).unwrap(); // 3 docs x 20, 63 raw
        let sa = InputSpec::parse(a.to_str().unwrap()).unwrap();
        let sb = InputSpec::parse(b.to_str().unwrap()).unwrap();
        (sa, sb)
    }

    fn drain(m: &mut Mixer) -> Vec<usize> {
        let mut v = vec![];
        while let Some((i, _)) = m.next_doc().unwrap() {
            v.push(i);
        }
        v
    }

    #[test]
    fn mixer_equal_bytes_interleaves_by_text_bytes() {
        let (a, b) = mixer_files("eq");
        let mut m = Mixer::open(&[a, b], MixPolicy::EqualBytes, None).unwrap();
        // always read from the source with the fewest text bytes so far (ties: first source)
        assert_eq!(drain(&mut m), vec![0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        assert_eq!(m.readers()[0].docs, 10);
        assert_eq!(m.readers()[1].docs, 3);
        assert!(m.exhausted(0) && m.exhausted(1));
    }

    #[test]
    fn mixer_proportional_spreads_sources_over_the_stream() {
        let (a, b) = mixer_files("prop");
        let mut m = Mixer::open(&[a.clone(), b.clone()], MixPolicy::Proportional, None).unwrap();
        // the source with the smallest fraction of its file read goes next
        assert_eq!(drain(&mut m), vec![0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(m.file_size(0), 50);
        // with a doc cap, progress is docs / cap when that is larger
        let mut m = Mixer::open(&[a, b], MixPolicy::Proportional, Some(2)).unwrap();
        assert_eq!(drain(&mut m), vec![0, 1, 0, 1]);
        assert_eq!(m.readers()[0].docs, 2);
        assert!(m.exhausted(0) && m.exhausted(1));
    }

    #[test]
    fn mixer_missing_file_is_an_error() {
        let spec = InputSpec::parse("definitely-missing-file.txt").unwrap();
        assert!(Mixer::open(&[spec], MixPolicy::EqualBytes, None).is_err());
    }
}
