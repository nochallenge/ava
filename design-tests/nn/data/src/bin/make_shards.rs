//! TESTING / DESIGN ONLY — make_shards: stream text inputs through a BPE into train/val shards.
//!
//!   make_shards --bpe FILE --out DIR [--shard-tokens N] [--val-per-mille V]
//!               [--max-docs-per-input M] [--seed S] [--verify-every K] INPUT...
//!
//! Defaults: --shard-tokens 50M, --val-per-mille 5 (0.5%), --seed ava_data's split seed.
//! The inputs are interleaved proportionally (the source with the smallest fraction read goes
//! next, the fraction being max(bytes read / file size, docs / M)), so all sources finish
//! together and every shard holds roughly the global mixture. Each line is one document,
//! encoded with EOS appended and routed to DIR/train or DIR/val by the deterministic split
//! rule. Writes DIR/bpe.txt (a copy of the tokenizer) and DIR/manifest.txt (this run's
//! numbers). --verify-every K checks decode(encode(doc)) == doc on every K-th document and
//! stops on the first mismatch (default 0 = off). Memory: the tokenizer, its chunk cache, and
//! 1 MiB buffers; nothing grows with the corpus.

use ava_data::bpe::Bpe;
use ava_data::input::{parse_count, InputSpec, MixPolicy, Mixer};
use ava_data::shards::{TrainValWriter, DEFAULT_SHARD_TOKENS, DEFAULT_SPLIT_SEED, DEFAULT_VAL_PER_MILLE};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: make_shards --bpe FILE --out DIR [--shard-tokens N] [--val-per-mille V] [--max-docs-per-input M] [--seed S] [--verify-every K] INPUT...";

struct Opts {
    bpe: String,
    out: PathBuf,
    shard_tokens: usize,
    val_per_mille: u32,
    max_docs: Option<u64>,
    verify_every: u64,
    seed: u64,
    inputs: Vec<InputSpec>,
}

fn parse_args() -> Result<Opts, String> {
    let (mut bpe, mut out, mut max_docs) = (None, None, None);
    let mut shard_tokens = DEFAULT_SHARD_TOKENS;
    let mut val_per_mille = DEFAULT_VAL_PER_MILLE;
    let mut seed = DEFAULT_SPLIT_SEED;
    let mut verify_every = 0u64;
    let mut inputs = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--bpe" => bpe = Some(val("--bpe")?),
            "--out" => out = Some(PathBuf::from(val("--out")?)),
            "--shard-tokens" => shard_tokens = parse_count(&val("--shard-tokens")?)? as usize,
            "--val-per-mille" => val_per_mille = parse_count(&val("--val-per-mille")?)? as u32,
            "--max-docs-per-input" => max_docs = Some(parse_count(&val("--max-docs-per-input")?)?),
            "--seed" => seed = parse_count(&val("--seed")?)?,
            "--verify-every" => verify_every = parse_count(&val("--verify-every")?)?,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s if s.starts_with("--") => return Err(format!("unknown flag {s}")),
            _ => inputs.push(InputSpec::parse(&a)?),
        }
    }
    let bpe = bpe.ok_or("--bpe is required")?;
    let out = out.ok_or("--out is required")?;
    if inputs.is_empty() {
        return Err("at least one INPUT is required".into());
    }
    Ok(Opts { bpe, out, shard_tokens, val_per_mille, max_docs, verify_every, seed, inputs })
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

#[derive(Default, Clone, Copy)]
struct PerInput {
    tokens: u64,
    words: u64,
}

fn run(opts: Opts) -> std::io::Result<()> {
    let t0 = Instant::now();
    let bpe = Bpe::load(&opts.bpe)?;
    let vocab = bpe.vocab_size();
    let mut writer = TrainValWriter::new(&opts.out, vocab, opts.shard_tokens, opts.val_per_mille, opts.seed)?;
    let mut mixer = Mixer::open(&opts.inputs, MixPolicy::Proportional, opts.max_docs)?;
    let mut per = vec![PerInput::default(); opts.inputs.len()];
    let mut ids: Vec<u32> = Vec::new();
    let (mut docs, mut tokens, mut bytes, mut words) = (0u64, 0u64, 0u64, 0u64);
    let mut verified = 0u64;
    let mut last_report = Instant::now();
    eprintln!("tokenizer {} (vocab {vocab}); writing {}", opts.bpe, opts.out.display());
    while let Some((src, doc)) = mixer.next_doc()? {
        ids.clear();
        bpe.encode_into(&doc, &mut ids);
        if opts.verify_every > 0 && docs % opts.verify_every == 0 {
            let back = bpe.decode(&ids);
            if back != doc {
                return Err(std::io::Error::other(format!("round trip failed on doc {docs}: {doc:?} -> {back:?}")));
            }
            verified += 1;
        }
        ids.push(ava_data::bpe::EOS);
        writer.push_doc(&ids)?;
        let w = doc.split_whitespace().count() as u64;
        docs += 1;
        tokens += ids.len() as u64;
        bytes += doc.len() as u64;
        words += w;
        per[src].tokens += ids.len() as u64;
        per[src].words += w;
        if last_report.elapsed().as_secs_f64() >= 15.0 {
            last_report = Instant::now();
            let s = t0.elapsed().as_secs_f64();
            eprintln!(
                "  {:>6.0}s  {} docs  {:.1}M tokens  {:.0} MB text  {:.2} MB/s  {:.0}k tok/s",
                s,
                docs,
                tokens as f64 / 1e6,
                bytes as f64 / 1e6,
                bytes as f64 / 1e6 / s,
                tokens as f64 / 1e3 / s
            );
        }
    }
    let stats = writer.finish()?;
    let secs = t0.elapsed().as_secs_f64();
    bpe.save(opts.out.join("bpe.txt"))?;

    let mut report = String::new();
    report.push_str(&format!("tokenizer: {} (vocab {vocab})\n", opts.bpe));
    report.push_str(&format!(
        "shard_tokens: {}  val_per_mille: {}  seed: {}  max_docs_per_input: {}\n",
        opts.shard_tokens,
        opts.val_per_mille,
        opts.seed,
        opts.max_docs.map_or("none".to_string(), |m| m.to_string())
    ));
    report.push_str("inputs (docs, text bytes, % of file read, skipped lines, truncated lines, tokens, tokens/word):\n");
    for (i, r) in mixer.readers().iter().enumerate() {
        let size = mixer.file_size(i);
        report.push_str(&format!(
            "  {:>9} docs  {:>12} B  {:>6.2}% of file  skipped {:>6}  trunc {}  {:>11} tok  {:.3} tok/word  {}\n",
            r.docs,
            r.text_bytes,
            100.0 * r.raw_bytes as f64 / size.max(1) as f64,
            r.skipped,
            r.truncated,
            per[i].tokens,
            per[i].tokens as f64 / per[i].words.max(1) as f64,
            r.spec().spec
        ));
    }
    report.push_str(&format!(
        "docs: {docs}  (train {}, val {})\n",
        stats.train.docs, stats.val.docs
    ));
    report.push_str(&format!(
        "tokens: {tokens}  (train {} in {} shards, val {} in {} shards; EOS included)\n",
        stats.train.tokens, stats.train.shards, stats.val.tokens, stats.val.shards
    ));
    report.push_str(&format!(
        "text: {bytes} bytes, {words} words; {:.3} tokens/word (EOS included), {:.3} bytes/token\n",
        tokens as f64 / words.max(1) as f64,
        bytes as f64 / tokens.max(1) as f64
    ));
    report.push_str(&format!(
        "time: {:.1}s  ({:.0}k tokens/s, {:.2} MB/s of text)\n",
        secs,
        tokens as f64 / 1e3 / secs,
        bytes as f64 / 1e6 / secs
    ));
    if opts.verify_every > 0 {
        report.push_str(&format!("verified decode(encode(doc)) == doc on {verified} docs (every {})\n", opts.verify_every));
    }
    print!("{report}");
    let mut f = std::fs::File::create(opts.out.join("manifest.txt"))?;
    f.write_all(report.as_bytes())?;
    Ok(())
}
