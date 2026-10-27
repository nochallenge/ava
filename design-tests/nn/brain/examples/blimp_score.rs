// TESTING / DESIGN ONLY — blimp_score: BLiMP minimal pairs for a trained checkpoint.
//
// blimp_score [DIR data/blimp] [--model PATH] [--bpe PATH] [--threads N] [--limit N per phenomenon]
//             [--seqs N sentences per forward pass, default 128]
//
// For every pair, each sentence is scored as [EOS] sentence [EOS] (the way documents sit in the
// shards) and the model is right when the good sentence gets the higher total log-probability
// (BLiMP's simple LM method, no length normalisation). Prints accuracy per phenomenon, per field and
// overall. Chance is 50%.
//
// Sentences are scored in batches: sorted by length, right-padded with PAD (the models are causal, so
// padding after a sentence cannot change its scores; PAD targets are not scored). Same quantity as
// scoring each sentence alone through generate::score_tokens, about 50x faster.

use ava_brain::generate::{load_bpe, load_checkpoint, parse_cli, Cli};
use ava_brain::{row_nll, Model};
use ava_data::bpe::{EOS, PAD};
use ava_grad::{Graph, Params, IGNORE};
use std::collections::BTreeMap;

/// The string value of `"key": "..."` in one JSON line (handles \" and \\ escapes).
fn field(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\": \"");
    let start = line.find(&pat)? + pat.len();
    let mut out = String::new();
    let mut chars = line[start..].chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                o => out.push(o),
            },
            '"' => return Some(out),
            c => out.push(c),
        }
    }
    None
}

/// Total NLL in nats of every sequence (each starts with EOS, which is context and not scored).
fn seq_nll(model: &Model, params: &Params, seqs: &[Vec<u32>], batch: usize, threads: usize) -> Vec<f64> {
    let vocab = model.cfg.vocab;
    let mut out = vec![0.0f64; seqs.len()];
    let mut order: Vec<usize> = (0..seqs.len()).collect();
    order.sort_by_key(|&i| seqs[i].len());
    for chunk in order.chunks(batch.max(1)) {
        let t = chunk.iter().map(|&i| seqs[i].len()).max().unwrap_or(1).saturating_sub(1).max(1);
        assert!(t <= model.cfg.ctx, "a sentence of {t} tokens exceeds the model's context {}", model.cfg.ctx);
        let b = chunk.len();
        let (mut ids, mut tg) = (Vec::with_capacity(b * t), Vec::with_capacity(b * t));
        for &i in chunk {
            let s = &seqs[i];
            for p in 0..t {
                ids.push(if p + 1 < s.len() { s[p] } else { PAD });
                tg.push(if p + 1 < s.len() { s[p + 1] } else { IGNORE });
            }
        }
        let mut g = Graph::new(params);
        g.set_threads(threads);
        let f = model.forward(&mut g, &ids, b, t);
        let logits = &g.value(f.logits).data;
        for (r, &i) in chunk.iter().enumerate() {
            out[i] = (0..t)
                .filter(|&p| tg[r * t + p] != IGNORE)
                .map(|p| row_nll(&logits[(r * t + p) * vocab..(r * t + p + 1) * vocab], tg[r * t + p] as usize))
                .sum();
        }
    }
    out
}

fn take_flag(argv: &mut Vec<String>, name: &str) -> Option<usize> {
    let i = argv.iter().position(|a| a == name)?;
    let v = argv[i + 1].parse().unwrap_or_else(|_| panic!("{name} N"));
    argv.drain(i..i + 2);
    Some(v)
}

fn main() {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    let mut dir = String::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/blimp"));
    let limit = take_flag(&mut argv, "--limit").unwrap_or(usize::MAX);
    let seqs_per_pass = take_flag(&mut argv, "--seqs").unwrap_or(128);
    if argv.first().is_some_and(|a| !a.starts_with("--")) {
        dir = argv.remove(0);
    }
    let cli = parse_cli(&argv, Cli::default()).expect("flags").unwrap_or_default();
    let (model, params, cfg) = load_checkpoint(&cli.model, &cli.overrides).expect("checkpoint");
    let bpe = load_bpe(&cli.bpe, cfg.vocab).expect("bpe");
    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("blimp dir").filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "jsonl")).collect();
    files.sort();
    let (mut right, mut total) = (0usize, 0usize);
    let mut by_field: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let t0 = std::time::Instant::now();
    for f in &files {
        let text = std::fs::read_to_string(f).expect("read");
        let mut fld = String::new();
        let mut seqs = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()).take(limit) {
            let (Some(g), Some(b)) = (field(line, "sentence_good"), field(line, "sentence_bad")) else { continue };
            if fld.is_empty() {
                fld = field(line, "field").unwrap_or_default();
            }
            for s in [g, b] {
                let mut t = vec![EOS];
                t.extend(bpe.encode_doc(&s));
                seqs.push(t);
            }
        }
        let nll = seq_nll(&model, &params, &seqs, seqs_per_pass, cli.threads);
        let n = nll.len() / 2;
        let r = (0..n).filter(|&k| nll[2 * k] < nll[2 * k + 1]).count(); // lower NLL = higher probability
        let name = f.file_stem().unwrap().to_string_lossy();
        println!("{:<55} {:>5.1}%  ({r}/{n})", name, 100.0 * r as f64 / n.max(1) as f64);
        let e = by_field.entry(fld).or_default();
        e.0 += r;
        e.1 += n;
        right += r;
        total += n;
    }
    println!();
    for (k, (r, n)) in &by_field {
        println!("field {:<20} {:>5.1}%  ({r}/{n})", k, 100.0 * *r as f64 / (*n).max(1) as f64);
    }
    println!("\nBLiMP overall: {:.1}% ({right}/{total}) in {:.0}s; chance 50%", 100.0 * right as f64 / total.max(1) as f64, t0.elapsed().as_secs_f64());
}
