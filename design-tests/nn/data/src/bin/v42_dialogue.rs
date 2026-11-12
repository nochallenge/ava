// TESTING / DESIGN ONLY
//! v42_dialogue: whole conversations as documents (V5). Cornell movie conversations and DailyDialog.
//!
//!   v42_dialogue [--data DIR] [--out DIR] [--bpe FILE] [--limit N] [--force]
//!
//! Defaults: --data <the repo's data folder>, --out <data>/nn/v42, --bpe <data>/nn/bpe-4096.txt.
//! Writes <out>/dialogue_cornell.docs, <out>/dialogue_dd.docs (see `docs` for the format) and
//! <out>/dialogue_report.txt. A finished output is skipped unless --force (outputs appear only when
//! complete). Tokens are counted as the shards will hold them: `encode_doc` + length marks, EOS included.
//! --limit N reads only the first N conversation rows / triple rows (a trial).
//! (`dd_train.zip` is a deflate zip; std has no inflater, so it is not read: `triples.tsv` holds the
//! same dialogues as sliding triples and is rebuilt instead. Its rows after 48,507 are the Cornell
//! corpus again; dialogues made mostly of Cornell lines are dropped and counted as `known`.)

use ava_data::bpe::Bpe;
use ava_data::dialogue::{cornell, cornell_line_set, dailydialog_triples, histogram, ConvStats};
use ava_data::docs::{read_all_docs, DocsWriter};
use ava_data::input::parse_count;
use ava_data::numbers::insert_marks;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

fn tokens_of(bpe: &Bpe, doc: &str) -> usize {
    insert_marks(&bpe.encode_doc(doc)).len()
}

fn summarize(name: &str, path: &Path, st: Option<&ConvStats>, bpe: &Bpe, secs: f64) -> io::Result<String> {
    let docs = read_all_docs(path)?;
    let toks: Vec<usize> = docs.iter().map(|d| tokens_of(bpe, d)).collect();
    let turns: Vec<usize> = docs.iter().map(|d| d.lines().count()).collect();
    let total: usize = toks.iter().sum();
    let mut sorted = toks.clone();
    sorted.sort_unstable();
    let pct = |p: f64| sorted.get(((sorted.len() as f64 - 1.0) * p).round() as usize).copied().unwrap_or(0);
    let mut s = format!(
        "{name}: {} docs, {} turns, {} tokens (EOS + marks included); tokens/doc mean {:.1} median {} p90 {} max {}\n",
        docs.len(),
        turns.iter().sum::<usize>(),
        total,
        total as f64 / docs.len().max(1) as f64,
        pct(0.5),
        pct(0.9),
        sorted.last().copied().unwrap_or(0)
    );
    s.push_str(&format!("  turns per doc: {:?}\n", histogram(&turns, &[3, 5, 9, 17])));
    s.push_str(&format!("  tokens per doc: {:?}\n", histogram(&toks, &[64, 128, 256, 512, 1024])));
    if let Some(st) = st {
        s.push_str(&format!("  build: {st:?}; {secs:.2}s\n"));
    } else {
        s.push_str("  build: skipped (output already there)\n");
    }
    Ok(s)
}

fn run() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(arg_value(&args, "--data").unwrap_or(ava_data::DATA_DIR.into()));
    let out = arg_value(&args, "--out").map(PathBuf::from).unwrap_or(data.join("nn").join("v42"));
    let bpe_path = arg_value(&args, "--bpe").map(PathBuf::from).unwrap_or(data.join("nn").join("bpe-4096.txt"));
    let limit = match arg_value(&args, "--limit") {
        Some(v) => Some(parse_count(&v).map_err(io::Error::other)?),
        None => None,
    };
    let force = args.iter().any(|a| a == "--force");
    std::fs::create_dir_all(&out)?;
    let bpe = Bpe::load(&bpe_path)?;
    let suffix = if limit.is_some() { "_trial" } else { "" };
    let mut report = format!("v42_dialogue: data {}  out {}  bpe {}  limit {:?}\n", data.display(), out.display(), bpe_path.display(), limit);

    // speaker marks cost: how the tokenizer splits them
    let probe = "A: Hello there.\nB: Hi.";
    let ids = bpe.encode(probe);
    let pieces: Vec<String> = ids.iter().map(|&i| format!("{:?}", String::from_utf8_lossy(bpe.token_bytes(i)))).collect();
    report.push_str(&format!("mark probe {probe:?} -> {} tokens: {}\n", ids.len(), pieces.join(" ")));
    report.push_str("  (every turn costs its mark: \"A\"/\"B\" and \":\" as existing byte tokens, plus the newline; no new tokens)\n");

    // Cornell
    let cpath = out.join(format!("dialogue_cornell{suffix}.docs"));
    let t0 = Instant::now();
    let cst = if force || !cpath.exists() {
        let mut w = DocsWriter::create(&cpath)?;
        let st = cornell(&data.join("cornell_lines.txt"), &data.join("cornell_convs.txt"), limit, |d| w.write_doc(d))?;
        w.finish()?;
        Some(st)
    } else {
        None
    };
    let csecs = t0.elapsed().as_secs_f64();
    report.push_str(&summarize("cornell", &cpath, cst.as_ref(), &bpe, csecs)?);

    // DailyDialog (from the sliding triples)
    let dpath = out.join(format!("dialogue_dd{suffix}.docs"));
    let t0 = Instant::now();
    let dst = if force || !dpath.exists() {
        let mut w = DocsWriter::create(&dpath)?;
        let known = cornell_line_set(&data.join("cornell_lines.txt"))?;
        let st = dailydialog_triples(&data.join("triples.tsv"), limit, &known, |d| w.write_doc(d))?;
        w.finish()?;
        Some(st)
    } else {
        None
    };
    let dsecs = t0.elapsed().as_secs_f64();
    report.push_str(&summarize("dailydialog", &dpath, dst.as_ref(), &bpe, dsecs)?);
    report.push_str("dd_train.zip: not read (deflate-compressed; std has no zip reader); triples.tsv rebuilt instead\n");

    // first conversation of each, for a look
    for p in [&cpath, &dpath] {
        if let Some(d) = read_all_docs(p)?.first() {
            report.push_str(&format!("first doc of {}:\n{}\n", p.file_name().unwrap().to_string_lossy(), d));
        }
    }
    print!("{report}");
    std::fs::write(out.join(format!("dialogue_report{suffix}.txt")), report)?;
    Ok(())
}
