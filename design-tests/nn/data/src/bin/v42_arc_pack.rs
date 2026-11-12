// TESTING / DESIGN ONLY
//! v42_arc_pack: ARC_Corpus packed into topic documents with exact and near dedup (see `arc_pack`).
//!
//!   v42_arc_pack [--input FILE] [--out FILE.docs] [--bpe FILE] [--limit N] [--min-df N]
//!                [--buckets N] [--near-jaccard X] [--min T] [--target T] [--max T] [--threads N] [--force]
//!
//! Defaults: the ARC_Corpus of data/ai2-arc, --out data/nn/v42/arc_packed.docs (arc_packed_trial.docs
//! with --limit), --bpe data/nn/bpe-4096.txt, --min-df 20, --buckets 256, --near-jaccard 0.7,
//! --min 256 --target 512 --max 1024 tokens, --threads 2. Temp buckets go to <out dir>/tmp-arc-pack
//! and are removed at the end. A finished output is skipped unless --force. The report is printed
//! and written next to the output (`<out stem>_report.txt`).

use ava_data::arc_pack::{pack_corpus, ArcParams, PackParams};
use ava_data::bpe::Bpe;
use ava_data::input::parse_count;
use std::io;
use std::path::PathBuf;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

fn num(args: &[String], flag: &str, default: u64) -> io::Result<u64> {
    match arg(args, flag) {
        Some(v) => parse_count(&v).map_err(io::Error::other),
        None => Ok(default),
    }
}

fn run() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(ava_data::DATA_DIR);
    let input = arg(&args, "--input").map(PathBuf::from).unwrap_or(data.join("ai2-arc/ARC-V1-Feb2018-2/ARC_Corpus.txt"));
    let limit = arg(&args, "--limit").map(|v| parse_count(&v)).transpose().map_err(io::Error::other)?;
    let default_out = if limit.is_some() { "arc_packed_trial.docs" } else { "arc_packed.docs" };
    let out_docs = arg(&args, "--out").map(PathBuf::from).unwrap_or(data.join("nn/v42").join(default_out));
    let bpe_path = arg(&args, "--bpe").map(PathBuf::from).unwrap_or(data.join("nn/bpe-4096.txt"));
    let near_jaccard: f64 = arg(&args, "--near-jaccard").map(|v| v.parse::<f64>().map_err(io::Error::other)).transpose()?.unwrap_or(0.7);
    let p = ArcParams {
        input,
        tmp_dir: out_docs.parent().unwrap().join("tmp-arc-pack"),
        out_docs: out_docs.clone(),
        limit,
        min_df: num(&args, "--min-df", 20)? as u32,
        buckets: num(&args, "--buckets", 256)? as usize,
        near_jaccard,
        pack: PackParams {
            min_tokens: num(&args, "--min", 256)? as usize,
            target_tokens: num(&args, "--target", 512)? as usize,
            max_tokens: num(&args, "--max", 1024)? as usize,
        },
        threads: num(&args, "--threads", 2)? as usize,
    };
    let stem = out_docs.file_stem().unwrap().to_string_lossy().into_owned();
    let report_path = out_docs.with_file_name(format!("{stem}_report.txt"));
    if out_docs.exists() && !args.iter().any(|a| a == "--force") {
        println!("{} already exists (finished); skipping. Report: {}", out_docs.display(), report_path.display());
        return Ok(());
    }
    let bpe = Bpe::load(&bpe_path)?;
    let rep = pack_corpus(&p, &bpe)?;
    let mut text = format!(
        "v42_arc_pack: input {}  out {}  limit {:?}  min_df {}  buckets {}  near_jaccard {}  pack {:?}  threads {}\n",
        p.input.display(),
        p.out_docs.display(),
        p.limit,
        p.min_df,
        p.buckets,
        p.near_jaccard,
        p.pack,
        p.threads
    );
    text.push_str(&rep.text());
    print!("{text}");
    std::fs::write(&report_path, text)?;
    Ok(())
}
