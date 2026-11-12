// TESTING / DESIGN ONLY
//! v42_babylm: BabyLM 2024 strict (100M words) packed into docs files, one per source and split.
//!
//!   v42_babylm [--in DIR] [--out DIR] [--bpe FILE] [--max-tokens 1024] [--sources a,b,...]
//!              [--pg19-manifest FILE] [--threads 2]
//!
//! Defaults: --in data/downloads/babylm-eng (train_100M/train_100M/*.train, dev/dev/*.dev), --out
//! data/nn/docs/babylm, --bpe data/nn/bpe-4096.txt, the five packed sources (childes, bnc_spoken,
//! switchboard, open_subtitles, gutenberg; BabyLM's simple_wiki is dropped, see `babylm`), and
//! --pg19-manifest data/pg19/MANIFEST.sha256 (every PG-19 book on disk is left out of the Gutenberg part;
//! the book numbers are never printed). Writes OUT/<source>_train.docs, OUT/<source>_dev.docs and
//! OUT/babylm_report.txt. A docs file that exists is finished and is skipped, so a re-run resumes.

use ava_data::babylm::{pack_file, pg19_ids, PackStats, DEFAULT_MAX_TOKENS, PACKED};
use ava_data::bpe::Bpe;
use ava_data::input::parse_count;
use std::collections::HashSet;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
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
    let input = PathBuf::from(arg(&args, "--in").unwrap_or(format!("{DATA}/downloads/babylm-eng")));
    let out = PathBuf::from(arg(&args, "--out").unwrap_or(format!("{DATA}/nn/docs/babylm")));
    let bpe_path = PathBuf::from(arg(&args, "--bpe").unwrap_or(format!("{DATA}/nn/bpe-4096.txt")));
    let max_tokens = arg(&args, "--max-tokens").map(|v| parse_count(&v).map_err(io::Error::other)).transpose()?.unwrap_or(DEFAULT_MAX_TOKENS as u64) as usize;
    let threads = arg(&args, "--threads").map(|v| parse_count(&v).map_err(io::Error::other)).transpose()?.unwrap_or(2).max(1) as usize;
    let sources: Vec<String> = match arg(&args, "--sources") {
        Some(s) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
        None => PACKED.iter().map(|s| s.to_string()).collect(),
    };
    let manifest = PathBuf::from(arg(&args, "--pg19-manifest").unwrap_or(format!("{DATA}/pg19/MANIFEST.sha256")));
    let exclude: HashSet<u64> = pg19_ids(&std::fs::read_to_string(&manifest).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", manifest.display())))?);
    std::fs::create_dir_all(&out)?;
    let bpe = Bpe::load(&bpe_path)?;

    // one job per (source, split)
    let mut jobs: Vec<(String, &str, PathBuf, PathBuf)> = Vec::new();
    for s in &sources {
        jobs.push((s.clone(), "train", input.join("train_100M/train_100M").join(format!("{s}.train")), out.join(format!("{s}_train.docs"))));
        jobs.push((s.clone(), "dev", input.join("dev/dev").join(format!("{s}.dev")), out.join(format!("{s}_dev.docs"))));
    }
    let t0 = Instant::now();
    let next = AtomicUsize::new(0);
    type Res = io::Result<Option<(PackStats, u64, f64)>>;
    let slots: Vec<Mutex<Option<Res>>> = jobs.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..threads.min(jobs.len()) {
            let bpe = bpe.clone();
            let (jobs, next, slots, exclude) = (&jobs, &next, &slots, &exclude);
            sc.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= jobs.len() {
                    break;
                }
                let (_, _, src, dst) = &jobs[i];
                let res: Res = if dst.exists() {
                    Ok(None)
                } else {
                    let t = Instant::now();
                    let bytes = std::fs::metadata(src).map(|m| m.len()).unwrap_or(0);
                    pack_file(src, dst, &bpe, exclude, max_tokens).map(|st| Some((st, bytes, t.elapsed().as_secs_f64())))
                };
                *slots[i].lock().unwrap() = Some(res);
            });
        }
    });
    let secs = t0.elapsed().as_secs_f64();
    let mut report = format!(
        "v42_babylm: in {}  out {}  max_tokens {max_tokens}  threads {threads}\n\
         dropped: BabyLM simple_wiki (an older copy of the Simple English Wikipedia, read in full from simplewiki.docs)\n\
         PG-19 books on disk (any split) left out of gutenberg: {} book numbers from {}\n\
         source         split    lines      parts  skipped  docs     tokens(counted)  in 512..{max_tokens}  min..max   MB     secs\n",
        input.display(),
        out.display(),
        exclude.len(),
        manifest.display()
    );
    let (mut tot_tok, mut tot_bytes, mut busy) = (0u64, 0u64, 0f64);
    for ((name, split, _, dst), slot) in jobs.iter().zip(slots) {
        match slot.into_inner().unwrap().unwrap()? {
            None => report.push_str(&format!("{name:<14} {split:<5}  exists, skipped ({})\n", dst.display())),
            Some((st, bytes, s)) => {
                report.push_str(&format!(
                    "{name:<14} {split:<5} {:>9} {:>8} {:>6} {:>8} {:>14} {:>9.1}%  {:>4}..{:<5} {:>6.1} {:>6.1}\n",
                    st.lines,
                    st.parts,
                    st.skipped_parts,
                    st.docs,
                    st.tokens,
                    100.0 * st.in_range as f64 / st.docs.max(1) as f64,
                    st.min_doc,
                    st.max_doc,
                    bytes as f64 / 1e6,
                    s
                ));
                tot_tok += st.tokens;
                tot_bytes += bytes;
                busy += s;
            }
        }
    }
    report.push_str(&format!(
        "total {tot_tok} tokens from {:.1} MB in {secs:.1}s wall ({:.2}M tokens/s per thread, {:.1} MB/s wall)\n",
        tot_bytes as f64 / 1e6,
        tot_tok as f64 / 1e6 / busy.max(1e-9),
        tot_bytes as f64 / 1e6 / secs.max(1e-9)
    ));
    print!("{report}");
    let name = if report.contains("exists, skipped") { "babylm_report_rerun.txt" } else { "babylm_report.txt" };
    std::fs::write(out.join(name), report)?;
    Ok(())
}
