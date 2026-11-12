// TESTING / DESIGN ONLY
//! v42_reason: the Wave 5 reasoning curricula into docs files (see the `reason` module and its families).
//!
//!   v42_reason [FAMILY ...|all] [--docs N] [--seed S] [--out-dir DIR] [--bpe FILE] [--threads 2]
//!              [--templates FILE] [--wikidata DIR] [--force] [--avoid-test-words]
//!
//! FAMILY: search, causal, bayes, rules, state, analogy, library (default: all). Each one writes
//! `DIR/reason_<family>.docs` (default DIR: the repo's data/nn/docs) and a `.report.txt` next to it, from a
//! fixed seed (default 42), so a rerun gives the same bytes (the sha256 is printed). A finished output is
//! skipped (resumable); --force remakes it. --docs N overrides the family's default count (a trial).
//!
//! CHECKS on every document, before it is written: the frame guard (`reason_guard`: the reasoning, social
//! and other v42 test frames and made-up names), the wording guard (the fact test templates of --templates,
//! filled with every Wikidata and hand-written fact name, as the shard builder does), the 0/1 bit-run limit
//! (12) and a number-run limit (12). A document that fails any of them is dropped and counted; every count
//! is expected to be 0. Tokens are counted as the shards count them (encode + EOS, then the length marks).

use ava_data::bpe::Bpe;
use ava_data::docs::DocsWriter;
use ava_data::fact_ledger::{all_facts, guard_names, KEYS_DIR, TEMPLATES_FILE, WIKIDATA_DIR};
use ava_data::input::parse_count;
use ava_data::numbers::insert_marks;
use ava_data::reason::{doc, Family, FAMILIES};
use ava_data::reason_guard::{longest_number_run, FrameGuard};
use ava_data::sha256::sha256_file;
use ava_data::wikidata::Wikidata;
use ava_data::wording_guard::{longest_bit_run, WordingGuard};
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DATA: &str = ava_data::DATA_DIR;
const BATCH: u64 = 1024;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// What one document came to.
struct Made {
    text: String,
    tokens: u64,
    frame_hits: Vec<String>,
    wording_hits: u32,
    bit_run: usize,
    number_run: usize,
}

impl Made {
    fn clean(&self) -> bool {
        self.frame_hits.is_empty() && self.wording_hits == 0 && self.bit_run <= 12 && self.number_run <= 12
    }
}

struct Ctx<'a> {
    seed: u64,
    frames: &'a FrameGuard,
    wording: &'a WordingGuard,
}

fn make(ctx: &Ctx, bpe: &Bpe, f: Family, i: u64) -> Made {
    let text = doc(f, ctx.seed, i);
    let tokens = insert_marks(&bpe.encode_doc(&text)).len() as u64;
    let frame_hits = ctx.frames.hits(&text);
    let wording_hits = ctx.wording.count(&text).0;
    let (bit_run, number_run) = (longest_bit_run(&text), longest_number_run(&text));
    Made { text, tokens, frame_hits, wording_hits, bit_run, number_run }
}

fn one_family(ctx: &Ctx, bpe: &Bpe, f: Family, n: u64, out: &Path, threads: usize) -> io::Result<String> {
    let t0 = Instant::now();
    let bpes: Vec<Bpe> = (0..threads).map(|_| bpe.clone()).collect();
    let mut w = DocsWriter::create(out)?;
    let (mut tokens, mut kept, mut dropped) = (0u64, 0u64, 0u64);
    let (mut frame_docs, mut wording_docs, mut bit_docs, mut num_docs) = (0u64, 0u64, 0u64, 0u64);
    let (mut max_bits, mut max_nums, mut max_tokens) = (0usize, 0usize, 0u64);
    let mut examples: Vec<String> = Vec::new();
    let mut start = 0;
    while start < n {
        let end = (start + BATCH).min(n);
        // each thread makes every threads-th document of the batch; results are put back in order
        let mut parts: Vec<Vec<(u64, Made)>> = Vec::new();
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|k| {
                    let b = &bpes[k];
                    s.spawn(move || (start + k as u64..end).step_by(threads).map(|i| (i, make(ctx, b, f, i))).collect::<Vec<_>>())
                })
                .collect();
            for h in hs {
                parts.push(h.join().expect("a worker panicked"));
            }
        });
        let mut all: Vec<(u64, Made)> = parts.into_iter().flatten().collect();
        all.sort_by_key(|(i, _)| *i);
        for (i, m) in all {
            max_bits = max_bits.max(m.bit_run);
            max_nums = max_nums.max(m.number_run);
            frame_docs += !m.frame_hits.is_empty() as u64;
            wording_docs += (m.wording_hits > 0) as u64;
            bit_docs += (m.bit_run > 12) as u64;
            num_docs += (m.number_run > 12) as u64;
            if !m.clean() {
                dropped += 1;
                if examples.len() < 5 {
                    examples.push(format!("  doc {i}: frames {:?}, wording {}, bit run {}, number run {}", m.frame_hits, m.wording_hits, m.bit_run, m.number_run));
                }
                continue;
            }
            tokens += m.tokens;
            max_tokens = max_tokens.max(m.tokens);
            kept += 1;
            w.write_doc(&m.text)?;
        }
        start = end;
    }
    let (_, bytes) = w.finish()?;
    let secs = t0.elapsed().as_secs_f64();
    let hex = sha256_file(out)?;
    let mut rep = format!(
        "v42_reason {}: seed {}, {n} docs made, {kept} written, {dropped} dropped\n\
         tokens {tokens} (EOS and length marks included), {:.0} per doc, longest {max_tokens}; {:.1} MB\n\
         guard docs: frame {frame_docs}, wording {wording_docs}, bit run > 12 {bit_docs}, number run > 12 {num_docs}; longest bit run {max_bits}, longest number run {max_nums}\n\
         time {secs:.1}s on {threads} threads: {:.0} docs/s, {:.2}M tokens/s\n\
         out {}\n  sha256 {hex}\n",
        f.name(),
        ctx.seed,
        tokens as f64 / kept.max(1) as f64,
        bytes as f64 / 1e6,
        n as f64 / secs,
        tokens as f64 / 1e6 / secs,
        out.display()
    );
    for e in examples {
        rep.push_str(&e);
        rep.push('\n');
    }
    std::fs::write(out.with_extension("report.txt"), &rep)?;
    Ok(rep)
}

fn run() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidInput, m);
    let mut fams: Vec<Family> = Vec::new();
    let (mut docs, mut seed, mut threads, mut force) = (None::<u64>, 42u64, 2usize, false);
    let mut out_dir = PathBuf::from(format!("{DATA}/nn/docs"));
    let mut bpe_path = PathBuf::from(format!("{DATA}/nn/bpe-4096.txt"));
    let mut templates = PathBuf::from(KEYS_DIR).join(TEMPLATES_FILE);
    let mut wd_dir = PathBuf::from(WIKIDATA_DIR);
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--force" => force = true,
            // the analogy family redraws a puzzle that spells a test's made-up word (off: the v42 bytes)
            "--avoid-test-words" => ava_data::reason_analogy::AVOID_TEST_WORDS.store(true, std::sync::atomic::Ordering::Relaxed),
            "all" => fams.extend(FAMILIES),
            _ if !a.starts_with("--") => fams.push(Family::parse(a).ok_or_else(|| bad(format!("unknown family {a}")))?),
            _ => {
                let v = args.get(i + 1).cloned().ok_or_else(|| bad(format!("{a} needs a value")))?;
                match a {
                    "--docs" => docs = Some(parse_count(&v).map_err(bad)?),
                    "--seed" => seed = parse_count(&v).map_err(bad)?,
                    "--threads" => threads = (parse_count(&v).map_err(bad)? as usize).max(1),
                    "--out-dir" => out_dir = PathBuf::from(v),
                    "--bpe" => bpe_path = PathBuf::from(v),
                    "--templates" => templates = PathBuf::from(v),
                    "--wikidata" => wd_dir = PathBuf::from(v),
                    _ => return Err(bad(format!("unknown argument {a}"))),
                }
                i += 1;
            }
        }
        i += 1;
    }
    if fams.is_empty() {
        fams.extend(FAMILIES);
    }
    let t = Instant::now();
    let wd = Wikidata::load(&wd_dir)?;
    let wording = WordingGuard::load(&templates, guard_names(&all_facts(), &wd))?;
    let frames = FrameGuard::new();
    let bpe = Bpe::load(&bpe_path)?;
    println!(
        "guards: {} frames, wording guard {} templates x {} names; tokenizer {} (setup {:.1}s)",
        frames.frames.len(),
        wording.templates.len(),
        wording.n_names(),
        bpe_path.display(),
        t.elapsed().as_secs_f64()
    );
    let ctx = Ctx { seed, frames: &frames, wording: &wording };
    for f in fams {
        let out = out_dir.join(format!("{}.docs", f.source_name()));
        if out.exists() && !force {
            println!("{} exists (finished); skipping", out.display());
            continue;
        }
        let n = docs.unwrap_or(f.default_docs());
        let rep = one_family(&ctx, &bpe, f, n, &out, threads)?;
        print!("{rep}");
    }
    Ok(())
}
