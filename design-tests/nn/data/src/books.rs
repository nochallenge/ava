//! TESTING / DESIGN ONLY — books: PG-19 books as single documents for v40's long reading.
//!
//! A book is read as bytes and decoded as UTF-8 (lossily: an invalid byte becomes U+FFFD, and
//! the book is counted as `lossy`). Its line endings are normalised (`\r\n` -> `\n`); nothing
//! else changes. It is then ONE document: `Bpe::encode_doc` (EOS at the end) followed by
//! `numbers::insert_marks`, so its ids are those of the v39 shards (vocab 4112).
//!
//! `list_books` lists `*.txt` in a directory sorted by file name (byte order, which is also the
//! order of the PG-19 bucket listing). `encode_books` encodes many books on several threads,
//! each with its own `Bpe` clone (a clone has its own chunk cache), and keeps the input order.

use crate::bpe::Bpe;
use crate::numbers::insert_marks;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// One encoded book.
#[derive(Clone, Debug)]
pub struct Book {
    /// File name (e.g. `10146.txt`).
    pub name: String,
    /// Marked token ids, EOS last.
    pub tokens: Vec<u32>,
    /// Bytes of the file as read.
    pub bytes: u64,
    /// The file was not valid UTF-8 (some bytes became U+FFFD).
    pub lossy: bool,
}

/// `*.txt` files of `dir`, sorted by file name (byte order).
pub fn list_books(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "txt"))
        .collect();
    v.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    Ok(v)
}

/// `\r\n` -> `\n`; nothing else changes.
pub fn normalise(text: &str) -> String {
    text.replace("\r\n", "\n")
}

/// A book's text (normalised) and whether its bytes were valid UTF-8.
pub fn read_book_text(path: &Path) -> io::Result<(String, bool, u64)> {
    let bytes = std::fs::read(path)?;
    let n = bytes.len() as u64;
    match String::from_utf8(bytes) {
        Ok(s) => Ok((normalise(&s), false, n)),
        Err(e) => Ok((normalise(&String::from_utf8_lossy(e.as_bytes())), true, n)),
    }
}

/// One book text as one marked document: `insert_marks(encode_doc(normalise(text)))`.
pub fn encode_book(bpe: &Bpe, text: &str) -> Vec<u32> {
    insert_marks(&bpe.encode_doc(&normalise(text)))
}

/// Read and encode `paths` on `threads` threads; the result is in `paths` order.
pub fn encode_books(bpe: &Bpe, paths: &[PathBuf], threads: usize) -> io::Result<Vec<Book>> {
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<Book>>> = Mutex::new(vec![None; paths.len()]);
    let first_err: Mutex<Option<io::Error>> = Mutex::new(None);
    std::thread::scope(|s| {
        for _ in 0..threads.max(1).min(paths.len().max(1)) {
            s.spawn(|| {
                let bpe = bpe.clone();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= paths.len() {
                        break;
                    }
                    let p = &paths[i];
                    match read_book_text(p) {
                        Ok((text, lossy, bytes)) => {
                            let tokens = insert_marks(&bpe.encode_doc(&text));
                            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                            slots.lock().unwrap()[i] = Some(Book { name, tokens, bytes, lossy });
                        }
                        Err(e) => {
                            let e = io::Error::new(e.kind(), format!("{}: {e}", p.display()));
                            first_err.lock().unwrap().get_or_insert(e);
                            next.store(paths.len(), Ordering::Relaxed);
                            break;
                        }
                    }
                }
            });
        }
    });
    if let Some(e) = first_err.into_inner().unwrap() {
        return Err(e);
    }
    Ok(slots.into_inner().unwrap().into_iter().map(|b| b.expect("every book encoded")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::EOS;
    use crate::numbers::strip_marks;

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ava_books-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn normalise_only_crlf() {
        assert_eq!(normalise("a\r\nb\rc\n\r\n"), "a\nb\rc\n\n");
    }

    #[test]
    fn books_listed_in_name_order_and_encoded_in_order() {
        let d = tmp_dir("enc");
        let texts = ["Chapter 1\r\nIt was 1850.\r\n", "Plain text, no numbers.\n", "Two\nlines 12 and 345\n"];
        for (i, t) in texts.iter().enumerate() {
            std::fs::write(d.join(format!("{}.txt", [100, 10, 2][i])), t).unwrap();
        }
        std::fs::write(d.join("README.md"), "not a book").unwrap();
        std::fs::write(d.join("bad.txt"), [b'a', 0xff, b'b']).unwrap();
        let paths = list_books(&d).unwrap();
        let names: Vec<String> = paths.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, ["10.txt", "100.txt", "2.txt", "bad.txt"]);
        let bpe = Bpe::from_merges(vec![]).unwrap();
        for threads in [1, 3, 8] {
            let books = encode_books(&bpe, &paths, threads).unwrap();
            assert_eq!(books.len(), 4);
            for (b, p) in books.iter().zip(&paths) {
                let (text, lossy, bytes) = read_book_text(p).unwrap();
                assert_eq!(b.tokens, encode_book(&bpe, &text));
                assert_eq!(*b.tokens.last().unwrap(), EOS);
                assert_eq!(bpe.decode(&strip_marks(&b.tokens)), text);
                assert!(!text.contains("\r\n"));
                assert_eq!((b.lossy, b.bytes), (lossy, bytes));
            }
            assert!(books[3].lossy && !books[0].lossy);
            // "It was 1850." gets a 4-digit mark
            assert!(books[1].tokens.contains(&(crate::numbers::MARK_BASE + 3)));
        }
        let _ = std::fs::remove_dir_all(&d);
    }
}
