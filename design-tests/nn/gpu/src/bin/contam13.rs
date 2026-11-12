// TESTING / DESIGN ONLY — contam13 (v42): train/test contamination check by 13-gram overlap (GPT-3's rule, on words).
// CPU only.
//
// contam13 (--items TSV | --lines FILE) [--train DIR] [--bpe FILE] [--all-choices] [--threads 2] [--max-shards N]
//          [--out FILE] [--allow-sealed]
//
// * --items: a multiple-choice item TSV (mc::read_items; `id kind bin right prompt c0 c1 ...`); each item's text is
//   its prompt and its right answer (--all-choices: every choice). --lines: one item per non-empty line.
// * --train: a shard folder (default data/nn/shards-4096/train), streamed one shard at a time per thread through the
//   tokenizer's byte table (--bpe). Only the eval side's 13-grams are held in memory (src/contam.rs).
// * Prints the items with any 13-gram found in training (ids only, never their text), the share of grams found, and
//   the speed. --out writes one line per item: id, grams, grams found. --max-shards N scans only the first N shards
//   (a quick trial to time the full run).
// * Sealed sets: a path containing "sealed" is refused unless --allow-sealed is given (a decision for the owner, made
//   before the check is pre-registered).

use ava_data::bpe::Bpe;
use ava_data::shards::ShardSet;
use ava_gpu::contam::{self, EvalGrams};
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: contam13 (--items TSV | --lines FILE) [--train DIR] [--bpe FILE] [--all-choices] [--threads 2]\n\
    \x20               [--max-shards N] [--out FILE] [--allow-sealed]";

fn main() {
    if let Err(e) = run() {
        eprintln!("contam13: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let (mut items_path, mut lines_path, mut out) = (None::<PathBuf>, None::<PathBuf>, None::<PathBuf>);
    let (mut train, mut bpe_path) = (PathBuf::from(ava_gpu::DEFAULT_TRAIN), PathBuf::from(ava_gpu::DEFAULT_BPE));
    let (mut all_choices, mut allow_sealed, mut threads, mut max_shards) = (false, false, 2usize, None::<usize>);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        match key {
            "-h" | "--help" => return Err(USAGE.into()),
            "--all-choices" => all_choices = true,
            "--allow-sealed" => allow_sealed = true,
            _ => {
                let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?;
                let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
                match key {
                    "--items" => items_path = Some(v.into()),
                    "--lines" => lines_path = Some(v.into()),
                    "--train" => train = v.into(),
                    "--bpe" => bpe_path = v.into(),
                    "--out" => out = Some(v.into()),
                    "--threads" => threads = num(v)?.max(1),
                    "--max-shards" => max_shards = Some(num(v)?),
                    _ => return Err(format!("unknown option {key}\n{USAGE}")),
                }
                i += 1;
            }
        }
        i += 1;
    }
    let src = match (&items_path, &lines_path) {
        (Some(p), None) | (None, Some(p)) => p.clone(),
        _ => return Err(format!("give exactly one of --items and --lines\n{USAGE}")),
    };
    if !allow_sealed && src.to_string_lossy().to_lowercase().contains("sealed") {
        return Err(format!("{} looks like a sealed set; refused (--allow-sealed is the owner's decision)", src.display()));
    }
    let t0 = Instant::now();
    let items: Vec<(String, String)> = if let Some(p) = &items_path {
        ava_gpu::mc::read_items(p)?
            .into_iter()
            .map(|it| {
                let ans = if all_choices { it.choices.join(" ") } else { it.choices[it.right].clone() };
                (it.id, format!("{} {ans}", it.prompt))
            })
            .collect()
    } else {
        let text = std::fs::read_to_string(&src).map_err(|e| format!("read {}: {e}", src.display()))?;
        text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()).map(|(n, l)| (format!("line{}", n + 1), l.to_string())).collect()
    };
    let eval = EvalGrams::new(&items);
    let with_grams = eval.grams.iter().filter(|g| !g.is_empty()).count();
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let bytes: Vec<Vec<u8>> = (0..bpe.vocab_size() as u32).map(|t| bpe.token_bytes(t).to_vec()).collect();
    let set = ShardSet::open(&train).map_err(|e| format!("shards {}: {e}", train.display()))?;
    let n_shards = max_shards.map_or(set.len(), |m| m.min(set.len()));
    println!(
        "{} items from {} ({} have 13+ words; {} distinct 13-grams); training {} ({} of {} shards, {} tokens in all); {} threads",
        items.len(),
        src.display(),
        with_grams,
        eval.distinct(),
        train.display(),
        n_shards,
        set.len(),
        set.total_tokens,
        threads
    );
    let t1 = Instant::now();
    let (found, tokens) = contam::scan_shards(&eval, &bytes, &set, threads, max_shards, contam::OVERLAP)?;
    let secs = t1.elapsed().as_secs_f64();
    let hits = eval.hits(&found);
    let dirty: Vec<usize> = (0..hits.len()).filter(|&k| hits[k] > 0).collect();
    let total_grams: usize = eval.grams.iter().map(Vec::len).sum();
    println!(
        "scanned {tokens} tokens in {secs:.1}s ({:.1}M tokens/s); setup {:.1}s",
        tokens as f64 / secs.max(1e-9) / 1e6,
        (t1 - t0).as_secs_f64()
    );
    println!(
        "contaminated items (any 13-gram in training): {} of {} ({:.2}%); item grams found {} of {}",
        dirty.len(),
        items.len(),
        100.0 * dirty.len() as f64 / items.len().max(1) as f64,
        hits.iter().sum::<usize>(),
        total_grams
    );
    for &k in dirty.iter().take(20) {
        println!("  {}  {} of {} grams", eval.ids[k], hits[k], eval.grams[k].len());
    }
    if dirty.len() > 20 {
        println!("  ... and {} more (see --out)", dirty.len() - 20);
    }
    if let Some(o) = out {
        let mut s = String::from("id\tgrams\tfound\n");
        for k in 0..hits.len() {
            s.push_str(&format!("{}\t{}\t{}\n", eval.ids[k], eval.grams[k].len(), hits[k]));
        }
        std::fs::write(&o, s).map_err(|e| format!("write {}: {e}", o.display()))?;
        println!("per-item counts: {}", o.display());
    }
    Ok(())
}
