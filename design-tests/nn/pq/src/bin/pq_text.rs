// TESTING / DESIGN ONLY
//! pq_text: pulls a text column out of Parquet files into a `.docs` file for Ava's reading.
//!
//! Output format (the same as the data crate's docs.rs): each document's full text (line breaks
//! kept, `\r\n` -> `\n`), then a newline, then a line holding only U+001E. A text that itself holds
//! U+001E has it replaced by a space, so the separator is never ambiguous.
//!
//! Usage:
//!   pq_text --out FILE.docs [--col text] [--title title] [--min-chars 200] [--limit N]
//!           [--info] FILE.parquet...
//! `--title` puts that column's value as the first line of each document (Wikipedia titles).
//! `--info` only prints the schema, row groups, rows and codecs.

use ava_pq::ParquetFile;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("pq_text: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut out: Option<PathBuf> = None;
    let (mut col, mut title) = ("text".to_string(), None::<String>);
    let (mut min_chars, mut limit, mut info) = (0usize, usize::MAX, false);
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(val()?)),
            "--col" => col = val()?,
            "--title" => title = Some(val()?),
            "--min-chars" => min_chars = val()?.parse().map_err(|_| "--min-chars needs a number")?,
            "--limit" => limit = val()?.parse().map_err(|_| "--limit needs a number")?,
            "--info" => info = true,
            _ if a.starts_with("--") => return Err(format!("unknown option {a}")),
            _ => files.push(PathBuf::from(a)),
        }
    }
    if files.is_empty() {
        return Err("give one or more .parquet files".into());
    }
    let mut w = match (&out, info) {
        (_, true) => None,
        (Some(p), false) => Some(BufWriter::with_capacity(1 << 22, std::fs::File::create(p).map_err(|e| e.to_string())?)),
        (None, false) => return Err("--out is required (or --info)".into()),
    };
    let t0 = Instant::now();
    let (mut docs, mut kept, mut short, mut nulls, mut bytes) = (0usize, 0usize, 0usize, 0usize, 0u64);
    'files: for f in &files {
        let mut pf = ParquetFile::open(f)?;
        eprintln!(
            "{}: {} rows, {} row groups, codecs {:?}, by {:?}",
            f.display(),
            pf.num_rows,
            pf.row_groups(),
            pf.codecs(),
            pf.created_by
        );
        if info {
            for c in &pf.columns {
                eprintln!("  column {:?} type {} optional {}", c.name, c.physical, c.optional);
            }
            continue;
        }
        let w = w.as_mut().unwrap();
        for rg in 0..pf.row_groups() {
            let texts = pf.read_bytes_column(rg, &col)?;
            let titles = match &title {
                Some(t) => Some(pf.read_bytes_column(rg, t)?),
                None => None,
            };
            for (i, t) in texts.into_iter().enumerate() {
                docs += 1;
                let Some(t) = t else {
                    nulls += 1;
                    continue;
                };
                let mut s = String::from_utf8_lossy(&t).replace("\r\n", "\n").replace('\u{1e}', " ");
                if let Some(Some(tt)) = titles.as_ref().map(|v| v[i].as_ref()) {
                    let head = String::from_utf8_lossy(tt).replace('\u{1e}', " ");
                    s = format!("{}\n\n{}", head.trim(), s);
                }
                if s.trim().chars().count() < min_chars {
                    short += 1;
                    continue;
                }
                let s = s.trim();
                w.write_all(s.as_bytes()).map_err(|e| e.to_string())?;
                w.write_all(b"\n\x1e\n").map_err(|e| e.to_string())?;
                bytes += s.len() as u64 + 3;
                kept += 1;
                if kept >= limit {
                    break 'files;
                }
            }
        }
    }
    if let Some(mut w) = w {
        w.flush().map_err(|e| e.to_string())?;
    }
    let secs = t0.elapsed().as_secs_f64();
    if !info {
        eprintln!(
            "docs {docs}, kept {kept}, too short {short}, null {nulls}, {:.1} MB written, {:.1} s ({:.1} MB/s)",
            bytes as f64 / 1e6,
            secs,
            bytes as f64 / 1e6 / secs.max(1e-9)
        );
    }
    Ok(())
}
