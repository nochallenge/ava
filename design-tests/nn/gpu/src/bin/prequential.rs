// TESTING / DESIGN ONLY — prequential (v42 I1): the bits she pays on new text before she learns it.
//
// prequential --model FILE (--val DIR [--ctx 256] [--tokens 100000] | --episodes FILE) [--arch D] [--batch 8]
//             [--lr 1e-4] [--warmup 0] [--wd 0] [--clip 1.0] [--no-static] [--batch-tokens 32768] [--out FILE]
//             [--bpe FILE] [--cpu] [--threads 2]
//
// The stream comes in order, --batch rows at a time: validation windows (ctx + 1 tokens, scored from the second; a
// window with padding is left out) up to --tokens targets, or the episodes of a file (blank-line separated, each read
// as [EOS] text [EOS] and scored after the first EOS, as sleep reads them). For each batch she is scored, then takes one
// learning step on it (src/preq.rs). A frozen copy of the same file scores the same batches (the static score) unless
// --no-static. Printed: the prequential and the static bits per byte, the gain, and the speed; --out gets one line per
// batch (step, targets, bytes, prequential nats, static nats, loss, cumulative bits per byte of both). Nothing is saved
// back: the learning here is only to measure it. The device is CUDA unless --cpu (then --threads, default 2;
// there --batch defaults to 1 and --batch-tokens to 256, for memory: give --batch to compare with a GPU run).

use ava_data::bpe::{Bpe, PAD};
use ava_data::shards::ShardSet;
use ava_gpu::model::Kind;
use ava_gpu::preq::{self, PreqOpts, PreqStep};
use ava_gpu::probe;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: prequential --model FILE (--val DIR [--ctx 256] [--tokens 100000] | --episodes FILE) [--arch D] [--batch 8]\n\
    \x20                  [--lr 1e-4] [--warmup 0] [--wd 0] [--clip 1.0] [--no-static] [--batch-tokens 32768] [--out FILE]\n\
    \x20                  [--bpe FILE] [--cpu] [--threads 2]";

fn main() {
    if let Err(e) = run() {
        eprintln!("prequential: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let (mut model, mut val, mut eps, mut out): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None, None);
    let (mut kind, mut ctx, mut tokens, mut batch, mut cpu, mut threads, mut stat) = (Kind::D, 256usize, 100_000u64, None::<usize>, false, 2usize, true);
    let mut o = PreqOpts { lr: 1e-4, warmup: 0, wd: 0.0, clip: 1.0, batch_tokens: 32768 };
    let mut batch_given = false;
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        match key {
            "-h" | "--help" => return Err(USAGE.into()),
            "--cpu" | "--no-static" => {
                if key == "--cpu" {
                    cpu = true;
                } else {
                    stat = false;
                }
                i += 1;
                continue;
            }
            _ => {}
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
        let fl = |v: &str| v.parse::<f64>().map_err(|_| format!("bad number {v:?} for {key}"));
        match key {
            "--model" => model = Some(v.into()),
            "--val" => val = Some(v.into()),
            "--episodes" => eps = Some(v.into()),
            "--arch" => kind = Kind::parse(v)?,
            "--ctx" => ctx = num(v)?.max(1),
            "--tokens" => tokens = num(v)? as u64,
            "--batch" => batch = Some(num(v)?.max(1)),
            "--lr" => o.lr = fl(v)?,
            "--warmup" => o.warmup = num(v)?,
            "--wd" => o.wd = fl(v)?,
            "--clip" => o.clip = fl(v)?,
            "--batch-tokens" => {
                o.batch_tokens = num(v)?.max(1);
                batch_given = true;
            }
            "--out" => out = Some(v.into()),
            "--bpe" => bpe_path = v.into(),
            "--threads" => threads = num(v)?.max(1),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let model = model.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    // the CPU forward keeps about 2 MB per batch token (probe::CPU_BATCH_TOKENS), and a learning step more
    let batch = batch.unwrap_or(if cpu { 1 } else { 8 });
    if cpu && !batch_given {
        o.batch_tokens = probe::CPU_BATCH_TOKENS;
    }
    if !(o.lr >= 0.0 && o.lr.is_finite()) {
        return Err(format!("--lr {}: must be 0 or more", o.lr));
    }
    if cpu {
        ava_gpu::diary::set_threads(threads); // before the first tensor op
    }
    let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    // T's position table fixes its context (read from the file); the other kinds have none
    let load_ctx = if kind == Kind::T {
        let t = ava_gpu::ckpt::read(&model)?;
        let rows = t.get(ava_gpu::model::POS_EMB).ok_or("a T model needs its position table")?.dims2().map_err(|e| e.to_string())?.0;
        if ctx > rows {
            return Err(format!("--ctx {ctx} is longer than T's position table ({rows})"));
        }
        rows
    } else {
        ctx.max(4096)
    };
    let (cfg, varmap, m) = ava_gpu::ckpt::load_auto(&model, kind, load_ctx, &dev)?;
    m.set_track_activity(false);
    let rows: Vec<(Vec<u32>, usize)> = match (&val, &eps) {
        (Some(v), None) => {
            probe::check_path(v, "--val")?;
            let set = ShardSet::open(v).map_err(|e| format!("open {}: {e}", v.display()))?;
            let mut rows = Vec::new();
            let mut have = 0u64;
            for w in set.iter_sequential(ctx) {
                if have >= tokens {
                    break;
                }
                if w.contains(&PAD) {
                    continue;
                }
                have += (w.len() - 1) as u64;
                rows.push((w, 1));
            }
            rows
        }
        (None, Some(e)) => {
            let texts = probe::read_episodes(e)?;
            let (seqs, from) = probe::episode_seqs(&bpe, cfg.glance, &texts);
            seqs.into_iter().zip(from).collect()
        }
        _ => return Err(format!("give exactly one of --val and --episodes\n{USAGE}")),
    };
    if rows.is_empty() {
        return Err("no text to read".into());
    }
    let max_id = rows.iter().flat_map(|r| r.0.iter()).copied().max().unwrap_or(0);
    if max_id as usize >= cfg.vocab {
        return Err(format!("the text holds token {max_id}, the model's vocabulary is {}", cfg.vocab));
    }
    let batches: Vec<Vec<(Vec<u32>, usize)>> = rows.chunks(batch).map(|c| c.to_vec()).collect();
    let frozen = if stat { Some(ava_gpu::ckpt::load_auto(&model, kind, load_ctx, &dev)?) } else { None };
    if let Some((_, _, f)) = &frozen {
        f.set_track_activity(false);
    }
    let byte_len = ava_gpu::eval::byte_lengths_n(&bpe, cfg.vocab);
    let mut log = match &out {
        Some(p) => {
            let mut f = std::fs::File::create(p).map_err(|e| format!("create {}: {e}", p.display()))?;
            writeln!(f, "step\ttargets\tbytes\tpreq_nats\tstatic_nats\tloss\tpreq_bpb_cum\tstatic_bpb_cum").map_err(|e| e.to_string())?;
            Some(f)
        }
        None => None,
    };
    println!("prequential: {} rows in {} batches of {batch}; lr {} (warm-up {}), wd {}, clip {}; static score {}", rows.len(), batches.len(), o.lr, o.warmup, o.wd, o.clip, if stat { "on" } else { "off" });
    let t0 = Instant::now();
    let (mut cp, mut cs, mut cb) = (0f64, 0f64, 0u64);
    let mut io_err: Option<String> = None;
    let mut on_step = |s: &PreqStep| {
        cp += s.preq;
        cs += s.stat;
        cb += s.bytes;
        if let Some(f) = log.as_mut() {
            if let Err(e) = writeln!(f, "{}\t{}\t{}\t{:.4}\t{:.4}\t{:.5}\t{:.5}\t{:.5}", s.step, s.tokens, s.bytes, s.preq, s.stat, s.loss, preq::bpb(cp, cb), preq::bpb(cs, cb)) {
                io_err.get_or_insert(e.to_string());
            }
        }
    };
    let steps = preq::prequential(&m, &varmap, frozen.as_ref().map(|f| &f.2), &dev, &batches, &byte_len, &o, &mut on_step)?;
    if let Some(e) = io_err {
        return Err(format!("writing the curve: {e}"));
    }
    let secs = t0.elapsed().as_secs_f64();
    let (p, s, b, n): (f64, f64, u64, u64) = steps.iter().fold((0.0, 0.0, 0, 0), |a, x| (a.0 + x.preq, a.1 + x.stat, a.2 + x.bytes, a.3 + x.tokens));
    println!("prequential: {:.5} bits/byte ({:.4} nats/token) over {n} targets, {b} bytes", preq::bpb(p, b), p / n.max(1) as f64);
    if stat {
        let (bp, bs) = (preq::bpb(p, b), preq::bpb(s, b));
        println!("static:      {bs:.5} bits/byte; learning on the fly saved {:+.5} bits/byte ({:+.2}%)", bs - bp, 100.0 * (bs - bp) / bs);
    }
    println!("speed: {:.0} targets/s ({secs:.1}s; each batch scored{} and learned once)", n as f64 / secs.max(1e-9), if stat { " twice" } else { "" });
    Ok(())
}
