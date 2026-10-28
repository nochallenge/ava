//! TESTING / DESIGN ONLY — num_shards: rewrite a dataset's train/val shards with length marks and/or practice docs.
//!
//!   num_shards --src DIR --out DIR --bpe FILE [--marks] [--practice FILE [--practice-frac 0.02]]
//!
//! DIR are dataset roots holding `train/` and `val/` (as written by make_shards). Train gets
//! the marks and the practice docs; val gets the marks only, never practice. The practice file
//! holds one document per line (empty lines skipped), each encoded with `encode_doc`. With
//! `--marks` the output vocab is 4112 (ids 4096..=4111 are the marks; the BPE file, which knows
//! only 4096 ids, is copied unchanged). Writes OUT/bpe.txt and OUT/manifest.txt. See
//! `ava_data::num_mix` for the exact mixing rule.

use ava_data::bpe::Bpe;
use ava_data::num_mix::{rewrite, MixOpts, MixStats};
use ava_data::shards::ShardSet;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str =
    "usage: num_shards --src DIR --out DIR --bpe FILE [--marks] [--practice FILE [--practice-frac 0.02]]";

struct Opts {
    src: PathBuf,
    out: PathBuf,
    bpe: PathBuf,
    marks: bool,
    practice: Option<PathBuf>,
    frac: f64,
}

fn parse_args() -> Result<Opts, String> {
    let (mut src, mut out, mut bpe, mut practice, mut frac) = (None, None, None, None, None);
    let mut marks = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--src" => src = Some(PathBuf::from(val("--src")?)),
            "--out" => out = Some(PathBuf::from(val("--out")?)),
            "--bpe" => bpe = Some(PathBuf::from(val("--bpe")?)),
            "--marks" => marks = true,
            "--practice" => practice = Some(PathBuf::from(val("--practice")?)),
            "--practice-frac" => {
                let v = val("--practice-frac")?;
                frac = Some(v.parse::<f64>().map_err(|_| format!("bad --practice-frac {v:?}"))?);
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s => return Err(format!("unknown argument {s}")),
        }
    }
    if frac.is_some() && practice.is_none() {
        return Err("--practice-frac needs --practice".into());
    }
    let frac = frac.unwrap_or(0.02);
    if !(0.0..1.0).contains(&frac) {
        return Err(format!("--practice-frac {frac} must be in [0, 1)"));
    }
    Ok(Opts {
        src: src.ok_or("--src is required")?,
        out: out.ok_or("--out is required")?,
        bpe: bpe.ok_or("--bpe is required")?,
        marks,
        practice,
        frac,
    })
}

fn main() {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(&opts) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn err(msg: String) -> std::io::Error {
    std::io::Error::other(msg)
}

/// The practice file, one encoded doc (EOS included) per non-empty line.
fn load_practice(bpe: &Bpe, path: &Path) -> std::io::Result<(Vec<Vec<u32>>, u64)> {
    let mut docs = Vec::new();
    let mut skipped = 0u64;
    for line in BufReader::with_capacity(1 << 20, std::fs::File::open(path)?).lines() {
        let line = line?;
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            skipped += 1;
            continue;
        }
        docs.push(bpe.encode_doc(line));
    }
    Ok((docs, skipped))
}

fn run(o: &Opts) -> std::io::Result<()> {
    let t0 = Instant::now();
    let bpe = Bpe::load(&o.bpe)?;
    for split in ["train", "val"] {
        let set = ShardSet::open(o.src.join(split))?;
        if set.is_empty() {
            return Err(err(format!("{}: no shards", o.src.join(split).display())));
        }
        if set.vocab_size as usize != bpe.vocab_size() {
            return Err(err(format!(
                "{split}: shard vocab {} != tokenizer vocab {}",
                set.vocab_size,
                bpe.vocab_size()
            )));
        }
        let dst = o.out.join(split);
        if dst.exists() && !ShardSet::open(&dst)?.is_empty() {
            return Err(err(format!("{} already holds shards; use an empty directory", dst.display())));
        }
    }

    let mut report = String::new();
    report.push_str(&format!(
        "num_shards: src {}  bpe {}  marks {}  practice {}  practice_frac {}\n",
        o.src.display(),
        o.bpe.display(),
        o.marks,
        o.practice.as_ref().map_or("none".into(), |p| p.display().to_string()),
        if o.practice.is_some() { o.frac } else { 0.0 }
    ));
    let mut practice = match &o.practice {
        Some(p) => {
            let t = Instant::now();
            let (docs, skipped) = load_practice(&bpe, p)?;
            let toks: u64 = docs.iter().map(|d| d.len() as u64).sum();
            report.push_str(&format!(
                "practice file: {} docs, {toks} tokens (EOS included), {skipped} empty lines skipped; encoded in {:.1}s\n",
                docs.len(),
                t.elapsed().as_secs_f64()
            ));
            Some(docs)
        }
        None => None,
    };
    print!("{report}");

    for split in ["train", "val"] {
        let t = Instant::now();
        let opts = MixOpts {
            marks: o.marks,
            practice: if split == "train" { practice.take() } else { None },
            practice_frac: if split == "train" { o.frac } else { 0.0 },
        };
        let st: MixStats = rewrite(&o.src.join(split), &o.out.join(split), &opts)?;
        let secs = t.elapsed().as_secs_f64();
        let out_set = ShardSet::open(o.out.join(split))?;
        let line = format!(
            "{split}: {} src tokens -> {} out tokens in {} shards (vocab {}); practice {} tokens = {:.3}% (cycles {}); marks {}; {secs:.1}s, {:.1}M tok/s\n",
            st.src_tokens,
            st.out_tokens,
            out_set.len(),
            out_set.vocab_size,
            st.practice_tokens,
            100.0 * st.practice_tokens as f64 / st.out_tokens.max(1) as f64,
            st.practice_cycles,
            st.marks,
            st.src_tokens as f64 / 1e6 / secs.max(1e-9)
        );
        print!("{line}");
        report.push_str(&line);
    }
    bpe.save(o.out.join("bpe.txt"))?;
    let total = format!("time: {:.1}s\n", t0.elapsed().as_secs_f64());
    print!("{total}");
    report.push_str(&total);
    std::fs::File::create(o.out.join("manifest.txt"))?.write_all(report.as_bytes())?;
    Ok(())
}
