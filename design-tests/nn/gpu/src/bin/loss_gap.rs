// TESTING / DESIGN ONLY — loss_gap (v42 T4): where she loses to the transformer yardstick, token type by token type.
//
// loss_gap --ava FILE --t FILE --val DIR [--ava-arch D] [--ctx 512] [--tokens 200000] [--t-ctx N]
//          [--batch-tokens 32768] [--out FILE] [--bpe FILE] [--cpu] [--threads 2]
//
// Both models score the same validation windows (ctx + 1 tokens from ShardSet::iter_sequential(ctx); a window with
// padding is left out) until --tokens targets; each target gets its NLL under each model (probe::token_nll, batched,
// inference mode), one model loaded at a time. The gap Ava − T is then split four ways (src/tokgap.rs): recall hits /
// repeats / new tokens, common / mid / rare types, kinds (EOS, numbers, punctuation, word starts, word insides), and
// position in the window. The table goes to stdout and --out. T's position table fixes its context: --t-ctx (default:
// read from the file) must be >= --ctx. Both models must read the same token stream: a vocabulary smaller than the
// stream's largest id is refused (a glance model's length marks are not in a 4096-word T's vocabulary). The device is
// CUDA unless --cpu (then --threads, default 2; --batch-tokens 256 by default there).
// v42 Wave 3: --t-arch KIND (default T) makes the yardstick any kind, e.g. a second Ava (D) trained on other data; a
// yardstick that is not T has no position table, so --t-ctx then defaults to --ctx.

use ava_data::bpe::{Bpe, PAD};
use ava_data::shards::ShardSet;
use ava_gpu::model::Kind;
use ava_gpu::{probe, tokgap};
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: loss_gap --ava FILE --t FILE --val DIR [--ava-arch D] [--t-arch T] [--ctx 512] [--tokens 200000] [--t-ctx N]\n\
    \x20               [--batch-tokens 32768] [--out FILE] [--bpe FILE] [--cpu] [--threads 2]";

fn main() {
    if let Err(e) = run() {
        eprintln!("loss_gap: {e}");
        std::process::exit(1);
    }
}

/// Per-target NLL of every window under the model in `path` (and the seconds it took).
fn score(path: &PathBuf, kind: Kind, ctx: usize, windows: &[Vec<u32>], max_id: u32, batch: usize, dev: &candle_core::Device) -> Result<(Vec<Vec<f32>>, f64), String> {
    let (cfg, _varmap, model) = ava_gpu::ckpt::load_auto(path, kind, ctx, dev)?;
    if (max_id as usize) >= cfg.vocab {
        return Err(format!("{}: vocabulary {} cannot read token {max_id} of the stream", path.display(), cfg.vocab));
    }
    model.set_track_activity(false);
    model.set_inference(true);
    let t0 = Instant::now();
    let from = vec![1usize; windows.len()];
    let nll = probe::token_nll(&model, dev, windows, &from, batch).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((nll, t0.elapsed().as_secs_f64()))
}

fn run() -> Result<(), String> {
    let (mut ava, mut yard, mut val, mut out): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None, None);
    let (mut kind, mut ctx, mut tokens, mut t_ctx, mut batch, mut cpu, mut threads) = (Kind::D, 512usize, 200_000u64, None::<usize>, None::<usize>, false, 2usize);
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let mut t_kind = Kind::T;
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
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
        match key {
            "--ava" => ava = Some(v.into()),
            "--t" => yard = Some(v.into()),
            "--val" => val = Some(v.into()),
            "--ava-arch" => kind = Kind::parse(v)?,
            "--t-arch" => t_kind = Kind::parse(v)?,
            "--ctx" => ctx = num(v)?.max(2),
            "--tokens" => tokens = num(v)? as u64,
            "--t-ctx" => t_ctx = Some(num(v)?),
            "--batch-tokens" => batch = Some(num(v)?.max(1)),
            "--out" => out = Some(v.into()),
            "--bpe" => bpe_path = v.into(),
            "--threads" => threads = num(v)?.max(1),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let batch = batch.unwrap_or(if cpu { probe::CPU_BATCH_TOKENS } else { 32768 });
    let (ava, yard, val) = match (ava, yard, val) {
        (Some(a), Some(t), Some(v)) => (a, t, v),
        _ => return Err(format!("--ava, --t and --val are required\n{USAGE}")),
    };
    probe::check_path(&val, "--val")?;
    if kind == Kind::T {
        return Err("--ava-arch: Ava's kind, not T".into());
    }
    let t_ctx = match t_ctx {
        Some(c) => c,
        None if t_kind != Kind::T => ctx,
        None => {
            let t = ava_gpu::ckpt::read(&yard)?;
            let pos = t.get(ava_gpu::model::POS_EMB).ok_or_else(|| format!("{}: no position table; is it a T model?", yard.display()))?;
            pos.dims2().map_err(|e| e.to_string())?.0
        }
    };
    if ctx > t_ctx {
        return Err(format!("--ctx {ctx} is longer than T's position table ({t_ctx})"));
    }
    if cpu {
        ava_gpu::diary::set_threads(threads); // before the first tensor op
    }
    let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let set = ShardSet::open(&val).map_err(|e| format!("open {}: {e}", val.display()))?;
    let mut windows: Vec<Vec<u32>> = Vec::new();
    let mut have = 0u64;
    for w in set.iter_sequential(ctx) {
        if have >= tokens {
            break;
        }
        if w.contains(&PAD) {
            continue;
        }
        have += (w.len() - 1) as u64;
        windows.push(w);
    }
    if windows.is_empty() {
        return Err(format!("{}: no full window of {} tokens", val.display(), ctx + 1));
    }
    let max_id = windows.iter().flatten().copied().max().unwrap_or(0);
    println!("loss_gap: {} windows of {ctx} targets ({have} targets) from {}", windows.len(), val.display());
    let (a, sa) = score(&ava, kind, ctx, &windows, max_id, batch, &dev)?;
    let (t, st) = score(&yard, t_kind, t_ctx, &windows, max_id, batch, &dev)?;
    let scored = tokgap::score_windows(&windows, &a, &t)?;
    let kinds = tokgap::kind_table(&bpe, max_id as usize + 1);
    let table = tokgap::report(&tokgap::tabulate(&scored, &kinds));
    print!("{table}");
    println!("speed: Ava {:.0} tok/s ({sa:.1}s), T {:.0} tok/s ({st:.1}s)", have as f64 / sa.max(1e-9), have as f64 / st.max(1e-9));
    if let Some(p) = &out {
        std::fs::write(p, &table).map_err(|e| format!("write {}: {e}", p.display()))?;
    }
    Ok(())
}
