// TESTING / DESIGN ONLY — pieces_check: does a saved model read text in pieces (state carried) as it reads it in one pass?
//
// pieces_check --model FILE --val DIR [--ctx 256] [--rows 4] [--batches 2] [--pieces 64,64,64,64] [--threads 8] [--arch D]
//
// For rows·batches val windows: logits of one forward over ctx tokens, and of the same tokens fed piece by piece through
// forward_with_state (--pieces must sum to ctx). Prints the largest |logit difference|, relative to the largest |logit|,
// and both readings' mean loss on the next token (nats). CPU only; nothing is written. AVA_TEM_BETA (read once per
// process) changes B10's read sharpness, to see how much of a gap is its rounding.

use ava_data::shards::ShardSet;
use ava_gpu::model::Kind;
use candle_core::{Device, Tensor, D};
use std::path::PathBuf;

fn main() {
    if let Err(e) = run() {
        eprintln!("pieces_check: {e}");
        std::process::exit(1);
    }
}

fn mean_nll(logits: &Tensor, targets: &Tensor) -> Result<f64, String> {
    let e = |e: candle_core::Error| e.to_string();
    let lp = candle_nn::ops::log_softmax(logits, D::Minus1).map_err(e)?;
    let picked = lp.gather(&targets.unsqueeze(1).map_err(e)?, 1).map_err(e)?;
    Ok(-(picked.mean_all().map_err(e)?.to_scalar::<f32>().map_err(e)? as f64))
}

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (mut model, mut val) = (None::<PathBuf>, None::<PathBuf>);
    let (mut ctx, mut rows, mut batches, mut threads, mut kind) = (256usize, 4usize, 2usize, 8usize, Kind::D);
    let mut pieces: Option<Vec<usize>> = None;
    let mut i = 0;
    while i < a.len() {
        let v = a.get(i + 1).ok_or_else(|| format!("{} needs a value", a[i]))?;
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
        match a[i].as_str() {
            "--model" => model = Some(v.into()),
            "--val" => val = Some(v.into()),
            "--ctx" => ctx = num(v)?,
            "--rows" => rows = num(v)?.max(1),
            "--batches" => batches = num(v)?.max(1),
            "--threads" => threads = num(v)?.max(1),
            "--arch" => kind = Kind::parse(v)?,
            "--pieces" => pieces = Some(v.split(',').map(num).collect::<Result<_, _>>()?),
            k => return Err(format!("unknown option {k}")),
        }
        i += 2;
    }
    let model = model.ok_or("--model FILE is required")?;
    let val = val.ok_or("--val DIR is required")?;
    let pieces = pieces.unwrap_or_else(|| vec![ctx / 4; 4]);
    if pieces.iter().sum::<usize>() != ctx || pieces.contains(&0) {
        return Err(format!("--pieces {pieces:?} must be positive and sum to --ctx {ctx}"));
    }
    ava_gpu::diary::set_threads(threads);
    let dev = Device::Cpu;
    let (cfg, _vm, m) = ava_gpu::ckpt::load_auto(&model, kind, ctx, &dev)?;
    let set = ShardSet::open(&val).map_err(|e| format!("open {}: {e}", val.display()))?;
    if set.vocab_size as usize != cfg.vocab {
        return Err(format!("val vocab {} but the model has {}", set.vocab_size, cfg.vocab));
    }
    let e = |e: candle_core::Error| e.to_string();
    let mut it = set.iter_sequential(ctx);
    let t0 = std::time::Instant::now();
    let (mut worst, mut top, mut nll_one, mut nll_pc) = (0f32, 0f32, 0f64, 0f64);
    for _ in 0..batches {
        let (mut ids, mut tg) = (Vec::with_capacity(rows * ctx), Vec::with_capacity(rows * ctx));
        for _ in 0..rows {
            let w = it.next().ok_or("the val set ran out")?;
            ids.extend_from_slice(&w[..ctx]);
            tg.extend_from_slice(&w[1..ctx + 1]);
        }
        let x = Tensor::from_vec(ids.clone(), (rows, ctx), &dev).map_err(e)?;
        let one = m.forward(&x).map_err(e)?.reshape((rows, ctx, ())).map_err(e)?;
        let mut st = m.state(rows).map_err(e)?;
        let mut outs = Vec::new();
        let mut p0 = 0;
        for &p in &pieces {
            let chunk: Vec<u32> = (0..rows).flat_map(|r| ids[r * ctx + p0..r * ctx + p0 + p].to_vec()).collect();
            let l = m.forward_with_state(&Tensor::from_vec(chunk, (rows, p), &dev).map_err(e)?, &mut st).map_err(e)?;
            outs.push(l.reshape((rows, p, ())).map_err(e)?);
            p0 += p;
        }
        let pc = Tensor::cat(&outs, 1).map_err(e)?;
        let d = one.sub(&pc).map_err(e)?.abs().map_err(e)?.flatten_all().map_err(e)?.max(0).map_err(e)?.to_scalar::<f32>().map_err(e)?;
        let t = one.abs().map_err(e)?.flatten_all().map_err(e)?.max(0).map_err(e)?.to_scalar::<f32>().map_err(e)?;
        (worst, top) = (worst.max(d), top.max(t));
        let tgt = Tensor::from_vec(tg, rows * ctx, &dev).map_err(e)?;
        let v = cfg.vocab;
        nll_one += mean_nll(&one.reshape((rows * ctx, v)).map_err(e)?, &tgt)? / batches as f64;
        nll_pc += mean_nll(&pc.reshape((rows * ctx, v)).map_err(e)?, &tgt)? / batches as f64;
    }
    println!(
        "{} ({} windows of {ctx}, pieces {pieces:?}; {:.1} s): max |logit diff| {worst:.3e}, relative {:.3e} (max |logit| {top:.2}); loss one pass {nll_one:.6} vs pieces {nll_pc:.6} nats/token (diff {:+.2e})",
        model.display(),
        rows * batches,
        t0.elapsed().as_secs_f64(),
        worst / top,
        nll_pc - nll_one
    );
    Ok(())
}
