// TESTING / DESIGN ONLY — dormant_probe: how many of a saved model's LIF channels and MLP units are silent, per block.
//
// dormant_probe --model FILE --val DIR [--ctx 256] [--rows 8] [--batches 4] [--threads 8] [--arch D]
//
// Reads the first rows·batches windows of the val shards and measures each unit's activity the way ReDo does
// (upkeep::unit_activity: a LIF channel's mean |spike| reaching wout, an MLP unit's mean |output| after kWTA and
// the activation). Prints per block: units at exactly 0, at most 0.1 and 0.025 of the layer mean, and the layer
// mean. CPU only; nothing is written.

use ava_data::shards::ShardSet;
use ava_gpu::model::Kind;
use candle_core::{Device, Tensor};
use std::path::PathBuf;

fn main() {
    if let Err(e) = run() {
        eprintln!("dormant_probe: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (mut model, mut val) = (None::<PathBuf>, None::<PathBuf>);
    let (mut ctx, mut rows, mut batches, mut threads, mut kind) = (256usize, 8usize, 4usize, 8usize, Kind::D);
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
            k => return Err(format!("unknown option {k}")),
        }
        i += 2;
    }
    let model = model.ok_or("--model FILE is required")?;
    let val = val.ok_or("--val DIR is required")?;
    ava_gpu::diary::set_threads(threads);
    let dev = Device::Cpu;
    let (cfg, _vm, m) = ava_gpu::ckpt::load_auto(&model, kind, ctx, &dev)?;
    let set = ShardSet::open(&val).map_err(|e| format!("open {}: {e}", val.display()))?;
    if set.vocab_size as usize != cfg.vocab {
        return Err(format!("val vocab {} but the model has {}", set.vocab_size, cfg.vocab));
    }
    let mut it = set.iter_sequential(ctx);
    let mut bs = Vec::with_capacity(batches);
    for _ in 0..batches {
        let mut toks = Vec::with_capacity(rows * ctx);
        for _ in 0..rows {
            let w = it.next().ok_or("the val set ran out")?;
            toks.extend_from_slice(&w[..ctx]);
        }
        bs.push(Tensor::from_vec(toks, (rows, ctx), &dev).map_err(|e| e.to_string())?);
    }
    let t0 = std::time::Instant::now();
    let act = ava_gpu::upkeep::unit_activity(&m, &bs).map_err(|e| format!("activity: {e}"))?;
    println!("{} ({} tokens, ctx {ctx}; {:.1} s)", model.display(), rows * ctx * batches, t0.elapsed().as_secs_f64());
    let line = |name: &str, v: &[f32]| {
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len().max(1) as f64;
        let zero = v.iter().filter(|&&x| x == 0.0).count();
        println!(
            "  {name}: {} units, silent {zero} ({:.0}%), <=0.025 of mean {}, <=0.1 of mean {}, mean {mean:.4}",
            v.len(),
            100.0 * zero as f64 / v.len().max(1) as f64,
            ava_gpu::upkeep::dormant(v, 0.025).len(),
            ava_gpu::upkeep::dormant(v, 0.1).len()
        );
    };
    for (l, (li, ml)) in act.lif.iter().zip(&act.mlp).enumerate() {
        println!("block {l}");
        line("LIF channels", li);
        line("MLP units", ml);
    }
    Ok(())
}
