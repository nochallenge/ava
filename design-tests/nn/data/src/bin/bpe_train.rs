//! TESTING / DESIGN ONLY — bpe_train: learn a byte-level BPE tokenizer from a bounded sample.
//!
//!   bpe_train --vocab N --out FILE [--max-bytes B] INPUT...
//!
//! INPUT is `path`, `path@tab:N` or `path@sep:STR:N` (see ava_data::input). The inputs are
//! streamed line by line and interleaved byte-fairly (always read from the source with the
//! fewest text bytes so far), so the sample holds every source equally until a source runs
//! out. Reading stops once `--max-bytes` (default 200M = 200_000_000) bytes of document text
//! have been counted; chunk counting therefore needs memory for the unique chunks of the
//! sample only. Prints the sources sampled, unique chunks, times, the first 30 learned
//! tokens and tokens per word on a few sample sentences.

use ava_data::bpe::{Bpe, ChunkCounter, FIRST_MERGE_ID};
use ava_data::input::{parse_count, InputSpec, MixPolicy, Mixer};
use std::time::Instant;

const USAGE: &str = "usage: bpe_train --vocab N --out FILE [--max-bytes B] INPUT...";

struct Opts {
    vocab: usize,
    out: String,
    max_bytes: u64,
    inputs: Vec<InputSpec>,
}

fn parse_args() -> Result<Opts, String> {
    let mut vocab = None;
    let mut out = None;
    let mut max_bytes = 200_000_000u64;
    let mut inputs = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--vocab" => vocab = Some(parse_count(&val("--vocab")?)? as usize),
            "--out" => out = Some(val("--out")?),
            "--max-bytes" => max_bytes = parse_count(&val("--max-bytes")?)?,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s if s.starts_with("--") => return Err(format!("unknown flag {s}")),
            _ => inputs.push(InputSpec::parse(&a)?),
        }
    }
    let vocab = vocab.ok_or("--vocab is required")?;
    let out = out.ok_or("--out is required")?;
    if inputs.is_empty() {
        return Err("at least one INPUT is required".into());
    }
    Ok(Opts { vocab, out, max_bytes, inputs })
}

/// Tokens of `text` shown with `|` between them.
fn show_split(bpe: &Bpe, text: &str) -> String {
    bpe.encode(text).iter().map(|&id| String::from_utf8_lossy(bpe.token_bytes(id)).into_owned()).collect::<Vec<_>>().join("|")
}

fn main() {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(opts) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(opts: Opts) -> std::io::Result<()> {
    let t0 = Instant::now();
    let mut mixer = Mixer::open(&opts.inputs, MixPolicy::EqualBytes, None)?;
    let mut counter = ChunkCounter::new();
    let mut next_report = 20_000_000u64;
    while counter.total_bytes < opts.max_bytes {
        let Some((_, doc)) = mixer.next_doc()? else { break };
        counter.add_text(&doc);
        if counter.total_bytes >= next_report {
            eprintln!(
                "  counted {:.0} MB, {} unique chunks, {:.1}s",
                counter.total_bytes as f64 / 1e6,
                counter.unique(),
                t0.elapsed().as_secs_f64()
            );
            next_report += 20_000_000;
        }
    }
    let count_secs = t0.elapsed().as_secs_f64();

    println!("== sample (byte-fair, limit {} bytes of text)", opts.max_bytes);
    println!("{:>10} {:>12} {:>9} {:>7}  source", "docs", "text bytes", "skipped", "status");
    for (i, r) in mixer.readers().iter().enumerate() {
        let status = if mixer.exhausted(i) { "all" } else { "part" };
        println!("{:>10} {:>12} {:>9} {:>7}  {}", r.docs, r.text_bytes, r.skipped, status, r.spec().spec);
    }
    println!(
        "total: {} bytes of text, {} chunks, {} unique chunks; counting took {:.1}s",
        counter.total_bytes,
        counter.total_chunks,
        counter.unique(),
        count_secs
    );

    let t1 = Instant::now();
    let unique = counter.unique();
    let bpe = Bpe::train(counter.into_counts(), opts.vocab);
    let train_secs = t1.elapsed().as_secs_f64();
    println!(
        "trained: vocab {} ({} merges) from {} unique chunks in {:.1}s",
        bpe.vocab_size(),
        bpe.merges().len(),
        unique,
        train_secs
    );
    bpe.save(&opts.out)?;
    println!("saved {}", opts.out);

    println!("== first 30 learned tokens");
    let n = bpe.merges().len().min(30);
    let shown: Vec<String> = (0..n)
        .map(|i| {
            let id = FIRST_MERGE_ID + i as u32;
            format!("{id}:{:?}", String::from_utf8_lossy(bpe.token_bytes(id)))
        })
        .collect();
    for row in shown.chunks(6) {
        println!("  {}", row.join("  "));
    }
    println!("== last 10 learned tokens");
    let total = bpe.merges().len();
    let last: Vec<String> = (total.saturating_sub(10)..total)
        .map(|i| {
            let id = FIRST_MERGE_ID + i as u32;
            format!("{id}:{:?}", String::from_utf8_lossy(bpe.token_bytes(id)))
        })
        .collect();
    println!("  {}", last.join("  "));

    println!("== tokens per word on sample sentences");
    let samples = [
        "The quick brown fox jumps over the lazy dog.",
        "I have to go to sleep now, see you tomorrow!",
        "Photosynthesis converts light energy into chemical energy stored in glucose.",
        "In 2024, about 8.1 billion people lived on Earth.",
        "What do you mean? It will help us to relax.",
        "Don't worry, she'll be there at 10:30.",
    ];
    let (mut tt, mut tw) = (0usize, 0usize);
    for s in samples {
        let toks = bpe.encode(s).len();
        let words = s.split_whitespace().count();
        tt += toks;
        tw += words;
        println!("  {:>3} tok / {:>2} words = {:.2}  {}", toks, words, toks as f64 / words as f64, show_split(&bpe, s));
    }
    println!("  overall {:.2} tokens per word", tt as f64 / tw as f64);
    println!("total time {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}
