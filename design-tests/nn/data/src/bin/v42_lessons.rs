// TESTING / DESIGN ONLY
//! v42_lessons: the owner's early lessons into a docs file, one lesson per document, in her taught QA format
//! (see the `lessons` module).
//!
//!   v42_lessons [--dir data/teacher] [--out data/nn/docs/lessons.docs]
//!
//! Reads every `lessons_*.tsv` in --dir (sorted by name) and writes --out (paths relative to the repo root).
//! The output is rewritten on every run (it is small and deterministic) and its sha256 is printed, so a
//! rerun that changes nothing prints the same hash. The lessons are private: never copied anywhere else.
//! The shard builder reads the file only with `v42_shards --lessons`.

use ava_data::docs::DocsWriter;
use ava_data::lessons::{lesson_docs, lesson_files};
use ava_data::sha256::sha256_file;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

fn run() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(arg(&args, "--dir").unwrap_or("data/teacher".into()));
    let out = PathBuf::from(arg(&args, "--out").unwrap_or("data/nn/docs/lessons.docs".into()));
    let t = Instant::now();
    let files = lesson_files(&dir)?;
    if files.is_empty() {
        return Err(io::Error::other(format!("no lessons_*.tsv in {}", dir.display())));
    }
    let docs = lesson_docs(&files)?;
    let mut w = DocsWriter::create(&out)?;
    for d in &docs {
        w.write_doc(d)?;
    }
    let (n, bytes) = w.finish()?;
    let hex = sha256_file(&out)?;
    println!(
        "v42_lessons: {} files, {n} lessons, {bytes} bytes -> {} in {:.3}s\n  sha256 {hex}",
        files.len(),
        out.display(),
        t.elapsed().as_secs_f64()
    );
    Ok(())
}
