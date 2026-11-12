// TESTING / DESIGN ONLY — syn_ops (v42 C14): synaptic operations per token (spikes × fan-out), her energy bill.
//
// syn_ops --model FILE (--val DIR [--ctx 256] [--tokens 32768] | --episodes FILE) [--arch D] [--batch 8]
//         [--mem chunk16] [--out FILE] [--bpe FILE] [--cpu] [--threads 2]
//
// Reads validation windows (ctx tokens of input each, in order, up to --tokens positions) or the episodes of a file
// ([EOS] text [EOS] each) through Model::forward_trace, --batch rows per pass, and counts every block's LIF spikes and
// active hidden units at the real positions (PAD inputs are not counted). The counting rule (src/synops.rs, the one of
// parts/src/event.rs): a non-zero input costs one synaptic op per output it feeds, so Wout [d, d] and W2 [4d, d] cost
// events × d; every other matrix reads a dense vector and stays dense (flops.rs; --mem picks how D's memory core is
// counted: chunk16 by default, or token). Printed (and --out): per block spikes and active units per token, then the
// synaptic ops per token, what those two matrices cost dense, and the whole forward pass dense vs event-driven. The kinds
// with spikes only (not T). v49: models with block options (Ava 0.4) too, by the rules in src/synops.rs (int2 spikes,
// kWTA 1/4, relu², channel gates, value residual, dense mix, the engram's real look-back, the magnitude code), with
// more lines after the usual five; every other model is counted and printed exactly as before. The model is the lean
// inference copy (ckpt::load_auto_lean: the same values, bit for bit, without autograd's kept intermediates). The device
// is CUDA unless --cpu (then --threads, default 2; --batch 1 by default there, for memory).

use ava_data::bpe::{Bpe, PAD};
use ava_data::shards::ShardSet;
use ava_gpu::flops::{MemForm, DEFAULT_MEM};
use ava_gpu::model::Kind;
use ava_gpu::probe;
use ava_gpu::synops::SynCount;
use candle_core::Tensor;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: syn_ops --model FILE (--val DIR [--ctx 256] [--tokens 32768] | --episodes FILE) [--arch D] [--batch 8]\n\
    \x20              [--mem chunk16] [--out FILE] [--bpe FILE] [--cpu] [--threads 2]";

fn main() {
    if let Err(e) = run() {
        eprintln!("syn_ops: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let (mut model, mut val, mut eps, mut out): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None, None);
    let (mut kind, mut ctx, mut tokens, mut batch, mut cpu, mut threads, mut mem) = (Kind::D, 256usize, 32768u64, None::<usize>, false, 2usize, DEFAULT_MEM);
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
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
        match key {
            "--model" => model = Some(v.into()),
            "--val" => val = Some(v.into()),
            "--episodes" => eps = Some(v.into()),
            "--arch" => kind = Kind::parse(v)?,
            "--ctx" => ctx = num(v)?.max(1),
            "--tokens" => tokens = num(v)? as u64,
            "--batch" => batch = Some(num(v)?.max(1)),
            "--mem" => mem = MemForm::parse(v)?,
            "--out" => out = Some(v.into()),
            "--bpe" => bpe_path = v.into(),
            "--threads" => threads = num(v)?.max(1),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let model = model.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    let batch = batch.unwrap_or(if cpu { 1 } else { 8 }); // the CPU forward keeps about 2 MB per batch token
    if kind == Kind::T {
        return Err("T has no spikes: its cost is all dense (flops_table)".into());
    }
    if cpu {
        ava_gpu::diary::set_threads(threads); // before the first tensor op
    }
    let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let (cfg, _varmap, m) = ava_gpu::ckpt::load_auto_lean(&model, kind, ctx.max(4096), &dev)?;
    ava_gpu::synops::check(&cfg)?;
    m.set_track_activity(false);
    m.set_inference(true);
    // input rows: the tokens read (a window's ctx inputs; an episode's tokens but the last)
    let rows: Vec<Vec<u32>> = match (&val, &eps) {
        (Some(v), None) => {
            probe::check_path(v, "--val")?;
            let set = ShardSet::open(v).map_err(|e| format!("open {}: {e}", v.display()))?;
            let mut rows = Vec::new();
            let mut have = 0u64;
            for w in set.iter_sequential(ctx) {
                if have >= tokens {
                    break;
                }
                let r = w[..ctx].to_vec();
                have += r.iter().filter(|&&t| t != PAD).count() as u64;
                rows.push(r);
            }
            rows
        }
        (None, Some(e)) => {
            let texts = probe::read_episodes(e)?;
            probe::episode_seqs(&bpe, cfg.glance, &texts).0.into_iter().map(|mut s| {
                s.pop();
                s
            }).collect()
        }
        _ => return Err(format!("give exactly one of --val and --episodes\n{USAGE}")),
    };
    if rows.is_empty() {
        return Err("no text to read".into());
    }
    let mut count = SynCount::new(&cfg);
    let t0 = Instant::now();
    let e = |e: candle_core::Error| e.to_string();
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&r| rows[r].len());
    for chunk in order.chunks(batch) {
        let t = chunk.iter().map(|&r| rows[r].len()).max().unwrap_or(1);
        let (mut ids, mut keep) = (Vec::with_capacity(chunk.len() * t), Vec::with_capacity(chunk.len() * t));
        for &r in chunk {
            for p in 0..t {
                let tok = rows[r].get(p).copied().unwrap_or(PAD);
                ids.push(tok);
                keep.push(if tok == PAD { 0f32 } else { 1f32 });
            }
        }
        let (_, tr) = m.forward_trace(&Tensor::from_vec(ids, (chunk.len(), t), &dev).map_err(e)?).map_err(|x| format!("trace: {x}"))?;
        count.add(&tr, &Tensor::from_vec(keep, chunk.len() * t, &dev).map_err(e)?).map_err(e)?;
    }
    let secs = t0.elapsed().as_secs_f64();
    let report = count.report(&cfg, mem);
    print!("{report}");
    println!("speed: {:.0} tokens/s ({secs:.1}s)", count.tokens as f64 / secs.max(1e-9));
    if let Some(p) = &out {
        std::fs::write(p, &report).map_err(|e| format!("write {}: {e}", p.display()))?;
    }
    Ok(())
}
