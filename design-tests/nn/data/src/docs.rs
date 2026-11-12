// TESTING / DESIGN ONLY
//! docs: a plain-text file of many multi-line documents, and one streaming reader for every input kind.
//!
//! DOCS FILE (`*.docs`). UTF-8 text. Each document is written as its text, then a newline, then a
//! line holding only the record separator U+001E. So one document may hold many lines (a whole
//! conversation, a packed page of sentences). A U+001E inside a text is replaced by a space when
//! writing; nothing else changes, so a document reads back exactly as written. The writer writes
//! to `<path>.tmp` and renames it to `<path>` only in `finish`, so a docs file that exists is complete
//! (this is what makes the builders resumable: a finished output is simply skipped).
//!
//! SOURCES. A source spec is either a docs file (its path ends in `.docs`) or a line spec as in
//! `input` (`path`, `path@tab:N`, `path@sep:STR:N`; one document per line). `SourceReader` streams
//! either kind with the same counters.
//!
//! LEGACY TEXT. `decode_legacy` turns bytes into text: valid UTF-8 stays as it is, otherwise each byte
//! is read as Windows-1252 (the Cornell movie lines are stored that way), never as U+FFFD.

use crate::input::{DocReader, InputSpec};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// The record separator line that ends every document in a docs file.
pub const RS: char = '\u{1e}';

/// Writes a docs file (to `<path>.tmp`, renamed into place by `finish`).
pub struct DocsWriter {
    path: PathBuf,
    tmp: PathBuf,
    w: BufWriter<File>,
    /// Documents written.
    pub docs: u64,
    /// Text bytes written (separators not counted).
    pub bytes: u64,
}

impl DocsWriter {
    pub fn create(path: impl AsRef<Path>) -> io::Result<DocsWriter> {
        let path = path.as_ref().to_path_buf();
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                fs::create_dir_all(dir)?;
            }
        }
        let mut tmp = path.clone().into_os_string();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        let w = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
        Ok(DocsWriter { path, tmp, w, docs: 0, bytes: 0 })
    }

    /// Append one document (a U+001E inside it becomes a space).
    pub fn write_doc(&mut self, text: &str) -> io::Result<()> {
        if text.contains(RS) {
            let clean = text.replace(RS, " ");
            self.w.write_all(clean.as_bytes())?;
        } else {
            self.w.write_all(text.as_bytes())?;
        }
        self.w.write_all("\n\u{1e}\n".as_bytes())?;
        self.docs += 1;
        self.bytes += text.len() as u64;
        Ok(())
    }

    /// Flush and move the finished file into place.
    pub fn finish(mut self) -> io::Result<(u64, u64)> {
        self.w.flush()?;
        let f = self.w.into_inner().map_err(|e| e.into_error())?;
        f.sync_all()?;
        drop(f);
        fs::rename(&self.tmp, &self.path)?;
        Ok((self.docs, self.bytes))
    }
}

/// Streams the documents of a docs file out of any `BufRead`.
pub struct DocsReader<R: BufRead> {
    reader: R,
    line: Vec<u8>,
    doc: Vec<u8>,
    pub raw_bytes: u64,
    pub docs: u64,
    pub text_bytes: u64,
    /// A last document with no separator after it (still returned).
    pub unterminated: u64,
    pub done: bool,
}

impl<R: BufRead> DocsReader<R> {
    pub fn new(reader: R) -> DocsReader<R> {
        DocsReader { reader, line: Vec::new(), doc: Vec::new(), raw_bytes: 0, docs: 0, text_bytes: 0, unterminated: 0, done: false }
    }

    /// The next document, or `None` at the end.
    pub fn next_doc(&mut self) -> io::Result<Option<String>> {
        if self.done {
            return Ok(None);
        }
        self.doc.clear();
        loop {
            self.line.clear();
            let n = self.reader.read_until(b'\n', &mut self.line)?;
            if n == 0 {
                self.done = true;
                if self.doc.is_empty() {
                    return Ok(None);
                }
                // a document cut short: drop the newline we added before it
                if self.doc.last() == Some(&b'\n') {
                    self.doc.pop();
                }
                self.unterminated += 1;
                return Ok(Some(self.emit()));
            }
            self.raw_bytes += n as u64;
            if self.line == b"\x1e\n" || self.line == b"\x1e" {
                // the text was followed by one '\n' before the separator line
                if self.doc.last() == Some(&b'\n') {
                    self.doc.pop();
                }
                return Ok(Some(self.emit()));
            }
            self.doc.extend_from_slice(&self.line);
        }
    }

    fn emit(&mut self) -> String {
        self.docs += 1;
        let s = String::from_utf8_lossy(&self.doc).into_owned();
        self.text_bytes += s.len() as u64;
        s
    }
}

/// Open a docs file for streaming.
pub fn open_docs(path: impl AsRef<Path>) -> io::Result<DocsReader<BufReader<File>>> {
    let p = path.as_ref();
    let f = File::open(p).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", p.display())))?;
    Ok(DocsReader::new(BufReader::with_capacity(1 << 20, f)))
}

/// Read every document of a small docs file into memory.
pub fn read_all_docs(path: impl AsRef<Path>) -> io::Result<Vec<String>> {
    let mut r = open_docs(path)?;
    let mut v = Vec::new();
    while let Some(d) = r.next_doc()? {
        v.push(d);
    }
    Ok(v)
}

/// True when a source spec names a docs file.
pub fn is_docs_spec(spec: &str) -> bool {
    spec.ends_with(".docs")
}

/// `DIR@list:FILE` names a book list: every file named in the first column of FILE (a header row
/// starting with "name" is skipped) is read from DIR as ONE document (`\r\n` -> `\n`, legacy decoding).
/// Only the listed files are ever opened.
pub fn parse_book_list(spec: &str) -> Option<(PathBuf, PathBuf)> {
    let pos = spec.find("@list:")?;
    Some((PathBuf::from(&spec[..pos]), PathBuf::from(&spec[pos + 6..])))
}

/// Streams the listed book files of a directory, one document each.
pub struct BookListReader {
    dir: PathBuf,
    names: Vec<String>,
    next: usize,
    pub raw_bytes: u64,
    pub docs: u64,
    pub total_bytes: u64,
}

impl BookListReader {
    pub fn open(dir: &Path, list: &Path) -> io::Result<BookListReader> {
        let text = fs::read_to_string(list)?;
        let mut names = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let name = line.split('\t').next().unwrap_or("").trim();
            if name.is_empty() || (i == 0 && name == "name") {
                continue;
            }
            names.push(name.to_string());
        }
        let mut total_bytes = 0;
        for n in &names {
            total_bytes += fs::metadata(dir.join(n))?.len();
        }
        Ok(BookListReader { dir: dir.to_path_buf(), names, next: 0, raw_bytes: 0, docs: 0, total_bytes })
    }

    pub fn next_doc(&mut self) -> io::Result<Option<String>> {
        let Some(name) = self.names.get(self.next) else { return Ok(None) };
        self.next += 1;
        let bytes = fs::read(self.dir.join(name))?;
        self.raw_bytes += bytes.len() as u64;
        self.docs += 1;
        Ok(Some(decode_legacy(&bytes).replace("\r\n", "\n")))
    }
}

/// One input of any kind.
pub enum SourceReader {
    Lines(DocReader<BufReader<File>>),
    Docs(DocsReader<BufReader<File>>),
    Books(BookListReader),
}

impl SourceReader {
    pub fn open(spec: &str) -> Result<SourceReader, String> {
        if let Some((dir, list)) = parse_book_list(spec) {
            BookListReader::open(&dir, &list).map(SourceReader::Books).map_err(|e| e.to_string())
        } else if is_docs_spec(spec) {
            open_docs(spec).map(SourceReader::Docs).map_err(|e| e.to_string())
        } else {
            let s = InputSpec::parse(spec)?;
            s.open().map(SourceReader::Lines).map_err(|e| e.to_string())
        }
    }

    pub fn next_doc(&mut self) -> io::Result<Option<String>> {
        match self {
            SourceReader::Lines(r) => r.next_doc(),
            SourceReader::Docs(r) => r.next_doc(),
            SourceReader::Books(r) => r.next_doc(),
        }
    }

    pub fn raw_bytes(&self) -> u64 {
        match self {
            SourceReader::Lines(r) => r.raw_bytes,
            SourceReader::Docs(r) => r.raw_bytes,
            SourceReader::Books(r) => r.raw_bytes,
        }
    }

    pub fn docs(&self) -> u64 {
        match self {
            SourceReader::Lines(r) => r.docs,
            SourceReader::Docs(r) => r.docs,
            SourceReader::Books(r) => r.docs,
        }
    }

    /// Total bytes this source will read (for proportional interleaving).
    pub fn total_bytes(&self, spec: &str) -> u64 {
        match self {
            SourceReader::Books(r) => r.total_bytes,
            _ => fs::metadata(spec_path(spec)).map(|m| m.len()).unwrap_or(1),
        }
    }
}

/// The file path inside a source spec (without any `@tab:` / `@sep:` part).
pub fn spec_path(spec: &str) -> PathBuf {
    if let Some((dir, _)) = parse_book_list(spec) {
        return dir;
    }
    if is_docs_spec(spec) {
        return PathBuf::from(spec);
    }
    match InputSpec::parse(spec) {
        Ok(s) => s.path,
        Err(_) => PathBuf::from(spec),
    }
}

/// Windows-1252 for the bytes 0x80..=0x9F (0 = unused there; read as a space).
const CP1252_HIGH: [u16; 32] = [
    0x20AC, 0, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0, 0x017D, 0, 0, 0x2018,
    0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0, 0x017E, 0x0178,
];

/// Bytes to text: UTF-8 when valid, else every byte as Windows-1252 (Latin-1 above 0x9F).
pub fn decode_legacy(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => bytes
            .iter()
            .map(|&b| match b {
                0x80..=0x9F => match CP1252_HIGH[(b - 0x80) as usize] {
                    0 => ' ',
                    c => char::from_u32(c as u32).unwrap_or(' '),
                },
                _ => b as char,
            })
            .collect(),
    }
}

/// Every run of whitespace becomes one space; the ends are trimmed.
pub fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for w in s.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(w);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn docs_round_trip_with_newlines_and_empty_lines() {
        let dir = std::env::temp_dir().join(format!("ava_docs_rt_{}", std::process::id()));
        let path = dir.join("x.docs");
        let docs = ["A: hi\nB: hello", "", "one line", "ends in newline\n", "\n\nblank lines\n\nin it", "has \u{1e} inside"];
        let mut w = DocsWriter::create(&path).unwrap();
        for d in docs {
            w.write_doc(d).unwrap();
        }
        assert!(!path.exists(), "not in place before finish");
        let (n, _) = w.finish().unwrap();
        assert_eq!(n, docs.len() as u64);
        let back = read_all_docs(&path).unwrap();
        let mut want: Vec<String> = docs.iter().map(|s| s.to_string()).collect();
        want[5] = "has   inside".to_string();
        assert_eq!(back, want);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reader_keeps_an_unterminated_last_doc() {
        let mut r = DocsReader::new(Cursor::new(b"first\n\x1e\nsecond\nline".to_vec()));
        assert_eq!(r.next_doc().unwrap().as_deref(), Some("first"));
        assert_eq!(r.next_doc().unwrap().as_deref(), Some("second\nline"));
        assert_eq!(r.next_doc().unwrap(), None);
        assert_eq!(r.unterminated, 1);
        assert_eq!(r.docs, 2);
    }

    #[test]
    fn legacy_decoding() {
        assert_eq!(decode_legacy("café".as_bytes()), "café");
        assert_eq!(decode_legacy(b"don\x92t \x97 caf\xe9"), "don\u{2019}t \u{2014} caf\u{e9}");
        assert_eq!(collapse_ws("  a  b\t c \n"), "a b c");
    }

    #[test]
    fn spec_kinds() {
        assert!(is_docs_spec("C:/x/y.docs"));
        assert!(!is_docs_spec("C:/x/y.tsv@tab:2"));
        assert_eq!(spec_path("C:/x/y.tsv@tab:2"), PathBuf::from("C:/x/y.tsv"));
        assert_eq!(parse_book_list("C:/b@list:C:/l.tsv"), Some((PathBuf::from("C:/b"), PathBuf::from("C:/l.tsv"))));
    }

    #[test]
    fn book_list_reads_only_listed_files() {
        let dir = std::env::temp_dir().join(format!("ava_books_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("1.txt"), "line one\r\nline two").unwrap();
        fs::write(dir.join("2.txt"), "not listed").unwrap();
        fs::write(dir.join("list.tsv"), "name\ttokens\n1.txt\t5\n").unwrap();
        let spec = format!("{}@list:{}", dir.display(), dir.join("list.tsv").display());
        let mut r = SourceReader::open(&spec).unwrap();
        assert_eq!(r.next_doc().unwrap().as_deref(), Some("line one\nline two"));
        assert_eq!(r.next_doc().unwrap(), None);
        assert_eq!(r.docs(), 1);
        let _ = fs::remove_dir_all(&dir);
    }
}
