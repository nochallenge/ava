// TESTING / DESIGN ONLY
//! v42_dyck: nested-bracket (Dyck-k) documents as content-free pre-pretraining text (see `dyck`).
//!
//!   v42_dyck [--out FILE.docs] [--docs N] [--k K] [--alphabet letters|brackets] [--min-pairs A]
//!            [--max-pairs B] [--max-depth D] [--p-open P] [--doc-symbols S] [--seed X] [--bpe FILE]
//!
//! Defaults: data/nn/v42/dyck_k16.docs, 20000 docs, k 16, letters, 8..128 pairs, depth <= 16,
//! p-open 0.5, 512+ symbols a document, seed 42. Prints sizes, tokens (encode + EOS), tokens per
//! symbol and speed, and writes the same next to the output. A finished output is skipped.

use ava_data::bpe::Bpe;
use ava_data::docs::DocsWriter;
use ava_data::dyck::{document, Alphabet, DyckParams};
use ava_data::input::parse_count;
use ava_data::numbers::insert_marks;
use ava_data::rng::SplitMix64;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

const DATA: &str = ava_data::DATA_DIR;

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
    let num = |f: &str, d: u64| -> io::Result<u64> { arg(&args, f).map(|v| parse_count(&v).map_err(io::Error::other)).unwrap_or(Ok(d)) };
    let mut p = DyckParams::default();
    p.k = num("--k", p.k as u64)? as usize;
    p.min_pairs = num("--min-pairs", p.min_pairs as u64)? as usize;
    p.max_pairs = num("--max-pairs", p.max_pairs as u64)? as usize;
    p.max_depth = num("--max-depth", p.max_depth as u64)? as usize;
    p.doc_symbols = num("--doc-symbols", p.doc_symbols as u64)? as usize;
    if let Some(v) = arg(&args, "--p-open") {
        p.p_open = v.parse().map_err(io::Error::other)?;
    }
    p.alphabet = match arg(&args, "--alphabet").as_deref() {
        None | Some("letters") => Alphabet::Letters,
        Some("brackets") => Alphabet::Brackets,
        Some(o) => return Err(io::Error::other(format!("unknown alphabet {o}"))),
    };
    if p.k == 0 || p.k > p.alphabet.max_k() || p.min_pairs > p.max_pairs || p.max_depth == 0 {
        return Err(io::Error::other(format!("bad parameters {p:?}")));
    }
    let n_docs = num("--docs", 20_000)?;
    let seed = num("--seed", 42)?;
    let out = arg(&args, "--out").map(PathBuf::from).unwrap_or(PathBuf::from(format!("{DATA}/nn/v42/dyck_k{}.docs", p.k)));
    if out.exists() {
        println!("{} exists (finished); skipping", out.display());
        return Ok(());
    }
    let bpe = Bpe::load(arg(&args, "--bpe").unwrap_or(format!("{DATA}/nn/bpe-4096.txt")))?;
    let t0 = Instant::now();
    let mut r = SplitMix64::new(seed);
    let mut w = DocsWriter::create(&out)?;
    let (mut tokens, mut symbols) = (0u64, 0u64);
    let mut first = String::new();
    for i in 0..n_docs {
        let d = document(&mut r, &p);
        symbols += d.split_whitespace().count() as u64;
        tokens += insert_marks(&bpe.encode_doc(&d)).len() as u64;
        if i == 0 {
            first = d.lines().next().unwrap_or("").to_string();
        }
        w.write_doc(&d)?;
    }
    let (docs, bytes) = w.finish()?;
    let secs = t0.elapsed().as_secs_f64();
    let report = format!(
        "v42_dyck: {p:?}  seed {seed}\ndocs {docs} ({:.1} MB), symbols {symbols}, tokens {tokens} (EOS included), {:.3} tokens per symbol\ntime {secs:.1}s ({:.2}M tokens/s)\nfirst line: {first}\n",
        bytes as f64 / 1e6,
        tokens as f64 / symbols.max(1) as f64,
        tokens as f64 / 1e6 / secs
    );
    print!("{report}");
    std::fs::write(out.with_extension("report.txt"), report)?;
    Ok(())
}
