// TESTING / DESIGN ONLY — val_bpb (v42 Wave 3): a saved model's validation BPB on any val shard set.
//
// val_bpb --model FILE --val DIR [--arch D] [--ctx 256] [--eval-tokens 1000000] [--eval-batch 16] [--bpe FILE]
//         [--out FILE] [--cpu] [--threads 2] [--fused]
//
// The same scoring as the end of train_gpu (eval::evaluate on ShardSet::iter_sequential(ctx), the first --eval-tokens
// targets, PAD skipped, EOS = 1 byte, the glance's length marks 0 bytes), so a model scored on its own val set gives
// its result.tsv val_bpb again. Used to score one model on another run's val set (Wave 3 Stages 2 and 5). The model's
// shape comes from its file (ckpt::load_auto, with arch.txt beside it); the val shards must have the model's vocab.
// --out appends one line: model, val, ctx, eval_tokens, targets, bytes, nats_per_token, bpb, seconds.
// v52 --fused: the logits from the fused forward (eval::score_logits; runs/v52/fasteval.md); a model it does not take
// (or the CPU) is scored on the reference forward, and the first line says why.

use ava_data::bpe::Bpe;
use ava_data::shards::ShardSet;
use ava_gpu::eval::{byte_lengths_n, evaluate};
use ava_gpu::model::Kind;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "val_bpb --model FILE --val DIR [--arch D] [--ctx 256] [--eval-tokens 1000000] [--eval-batch 16] [--bpe FILE] [--out FILE] [--cpu] [--threads 2] [--fused]";

fn main() {
    if let Err(e) = run() {
        eprintln!("val_bpb: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let (mut model, mut val, mut out): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None);
    let (mut kind, mut ctx, mut eval_tokens, mut eval_batch, mut cpu, mut threads) = (Kind::D, 256usize, 1_000_000u64, 16usize, false, 2usize);
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        if key == "--cpu" {
            cpu = true;
            i += 1;
            continue;
        }
        if key == "--fused" {
            ava_gpu::eval::set_fused(true);
            i += 1;
            continue;
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
        match key {
            "--model" => model = Some(v.into()),
            "--val" => val = Some(v.into()),
            "--arch" => kind = Kind::parse(v)?,
            "--ctx" => ctx = num(v)?.max(1),
            "--eval-tokens" => eval_tokens = num(v)? as u64,
            "--eval-batch" => eval_batch = num(v)?.max(1),
            "--bpe" => bpe_path = v.into(),
            "--out" => out = Some(v.into()),
            "--threads" => threads = num(v)?.max(1),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let model_path = model.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    let val = val.ok_or_else(|| format!("--val is required\n{USAGE}"))?;
    ava_gpu::probe::check_path(&val, "--val")?;
    if cpu {
        ava_gpu::diary::set_threads(threads); // before the first tensor op
    }
    let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
    // on the CPU the lean model (ckpt::lean: the same logits bit for bit, without the autograd graph kept)
    let load = if cpu { ava_gpu::ckpt::load_auto_lean } else { ava_gpu::ckpt::load_auto };
    let (cfg, _varmap, m) = load(&model_path, kind, ctx, &dev)?;
    if ava_gpu::eval::fused() {
        println!("{}", ava_gpu::eval::fused_line(&m, &dev));
    }
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let byte_len = byte_lengths_n(&bpe, cfg.vocab);
    let set = ShardSet::open(&val).map_err(|e| format!("open val shards {}: {e}", val.display()))?;
    if set.vocab_size as usize != cfg.vocab {
        return Err(format!("val shards {} have vocab {}, the model has {}", val.display(), set.vocab_size, cfg.vocab));
    }
    let t0 = Instant::now();
    let (acc, windows) = evaluate(&m, set.iter_sequential(ctx), eval_batch, eval_tokens, &byte_len, &dev).map_err(|e| format!("eval: {e}"))?;
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "val_bpb {} on {}: {} targets ({} bytes, {windows} windows, ctx {ctx}): {:.4} nats/token, {:.6} bits/byte; {secs:.1}s ({:.0} tok/s)",
        model_path.display(),
        val.display(),
        acc.tokens,
        acc.bytes,
        acc.nats_per_token(),
        acc.bpb(),
        acc.tokens as f64 / secs.max(1e-9)
    );
    if acc.tokens < eval_tokens {
        println!("note: the val set ran out after {} scored targets (asked for {eval_tokens})", acc.tokens);
    }
    if let Some(o) = out {
        let new = std::fs::metadata(&o).map(|m| m.len() == 0).unwrap_or(true);
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&o).map_err(|e| format!("{}: {e}", o.display()))?;
        if new {
            writeln!(f, "model\tval\tctx\teval_tokens\ttargets\tbytes\tnats_per_token\tbpb\tseconds").map_err(|e| e.to_string())?;
        }
        writeln!(
            f,
            "{}\t{}\t{ctx}\t{eval_tokens}\t{}\t{}\t{:.6}\t{:.6}\t{secs:.2}",
            model_path.display(),
            val.display(),
            acc.tokens,
            acc.bytes,
            acc.nats_per_token(),
            acc.bpb()
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}
