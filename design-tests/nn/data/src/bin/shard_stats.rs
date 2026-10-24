//! TESTING / DESIGN ONLY — shard_stats: describe a shard directory and show decoded samples.
//!
//!   shard_stats DIR [--bpe FILE] [--ctx N] [--seed S] [--count-docs]
//!
//! DIR is either a make_shards output (with train/ and val/) or a single shard directory.
//! Prints shard count, tokens, sizes, and one sampled window from each set decoded with the
//! tokenizer (DIR/bpe.txt unless --bpe is given; EOS shows as <|eos|>). --count-docs reads
//! every shard once to count EOS tokens (documents).

use ava_data::bpe::{Bpe, EOS};
use ava_data::input::parse_count;
use ava_data::shards::{Sampler, ShardSet, ShardTokens};
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: shard_stats DIR [--bpe FILE] [--ctx N] [--seed S] [--count-docs]";

fn main() {
    let mut dir = None;
    let mut bpe_path = None;
    let mut ctx = 64usize;
    let mut seed = 1u64;
    let mut count_docs = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |flag: &str| {
            args.next().unwrap_or_else(|| {
                eprintln!("error: {flag} needs a value\n{USAGE}");
                std::process::exit(2)
            })
        };
        let num = |s: String| {
            parse_count(&s).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(2)
            })
        };
        match a.as_str() {
            "--bpe" => bpe_path = Some(PathBuf::from(val("--bpe"))),
            "--ctx" => ctx = num(val("--ctx")) as usize,
            "--seed" => seed = num(val("--seed")),
            "--count-docs" => count_docs = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return;
            }
            s if s.starts_with("--") => {
                eprintln!("error: unknown flag {s}\n{USAGE}");
                std::process::exit(2);
            }
            _ => dir = Some(PathBuf::from(a)),
        }
    }
    let Some(dir) = dir else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    if let Err(e) = run(&dir, bpe_path, ctx, seed, count_docs) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(dir: &Path, bpe_path: Option<PathBuf>, ctx: usize, seed: u64, count_docs: bool) -> std::io::Result<()> {
    let bpe_path = bpe_path.unwrap_or_else(|| dir.join("bpe.txt"));
    let bpe = if bpe_path.exists() { Some(Bpe::load(&bpe_path)?) } else { None };
    match &bpe {
        Some(b) => println!("tokenizer: {} (vocab {})", bpe_path.display(), b.vocab_size()),
        None => println!("tokenizer: none found at {} (samples shown as ids)", bpe_path.display()),
    }
    let subsets: Vec<(String, PathBuf)> = if dir.join("train").is_dir() {
        vec![("train".into(), dir.join("train")), ("val".into(), dir.join("val"))]
    } else {
        vec![("shards".into(), dir.to_path_buf())]
    };
    let mut grand = 0u64;
    for (name, path) in subsets {
        let set = ShardSet::open(&path)?;
        grand += set.total_tokens;
        let bytes: u64 = set.shards.iter().map(|s| std::fs::metadata(&s.path).map(|m| m.len()).unwrap_or(0)).sum();
        let sizes: Vec<u64> = set.shards.iter().map(|s| s.header.n_tokens).collect();
        println!("== {name}: {}", path.display());
        println!(
            "  shards {}  tokens {}  vocab {}  width {}  on disk {:.1} MB",
            set.len(),
            set.total_tokens,
            set.vocab_size,
            set.shards.first().map_or(0, |s| s.header.width),
            bytes as f64 / 1e6
        );
        if let (Some(min), Some(max)) = (sizes.iter().min(), sizes.iter().max()) {
            println!("  tokens per shard: min {min}  max {max}");
        }
        if count_docs {
            let mut eos = 0u64;
            for i in 0..set.len() {
                eos += match set.load_shard(i)? {
                    ShardTokens::U16(v) => v.iter().filter(|&&t| t as u32 == EOS).count() as u64,
                    ShardTokens::U32(v) => v.iter().filter(|&&t| t == EOS).count() as u64,
                };
            }
            println!("  documents (EOS count) {eos}  mean tokens/doc {:.1}", set.total_tokens as f64 / eos.max(1) as f64);
        }
        match Sampler::new(&set, ctx, seed) {
            Ok(mut s) => {
                println!("  windows per epoch at ctx {ctx}: {}", s.windows_per_epoch());
                let w = s.next_window();
                let (shard, off) = s.last_position().unwrap();
                println!("  sample window (shard {shard}, offset {off}, {} tokens):", w.len());
                match &bpe {
                    Some(b) => println!("    {:?}", b.decode_marked(&w)),
                    None => println!("    {w:?}"),
                }
                println!("  cursor after it: {}", s.cursor().to_line());
            }
            Err(e) => println!("  (no sample: {e})"),
        }
    }
    println!("total tokens {grand}");
    Ok(())
}
