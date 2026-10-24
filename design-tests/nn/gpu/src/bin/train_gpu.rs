// TESTING / DESIGN ONLY — train_gpu: train Ava-S on candle (CUDA by default, --cpu for the CPU backend).
//
// train_gpu --d D --layers L --ctx 256 --batch B --tokens N --lr LR --warmup W --min-lr-frac 0.1
//           --wd 0.1 --clip 1.0 --seed 1 --data-seed 1 --eval-tokens 500000 --log-every 50 --out DIR
//           [--bench-steps N] [--cpu] [--max-steps N] [--eval-batch 16] [--train DIR] [--val DIR] [--bpe FILE]
//
// * steps = tokens / (batch · ctx); step s uses lr = cosine_lr(s, warmup, steps, lr, lr · min_lr_frac).
// * Logs: stdout every --log-every steps (plus first and last); DIR/log.tsv gets every step
//   (step, loss, lr, grad_norm before clipping, tokens/s, step seconds, spike rates, kWTA post-relu).
// * DIR/model.safetensors is written every 500 steps and at the end (candle VarMap::save).
// * At the end: validation BPB on iter_sequential(ctx) of the val shards (first --eval-tokens targets,
//   PAD skipped, EOS = 1 byte), one line appended to DIR/result.tsv.
// * --bench-steps N: run N steps, print tokens/s (first step excluded), write nothing, no evaluation.

use ava_data::bpe::Bpe;
use ava_data::shards::{Sampler, ShardSet};
use ava_gpu::eval::{byte_lengths, evaluate, train_io};
use ava_gpu::model::{Activity, Config, Model, VOCAB};
use ava_gpu::optim::{clip_scale, cosine_lr, AdamW, Param};
use candle_core::{DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    d: usize,
    layers: usize,
    ctx: usize,
    batch: usize,
    tokens: u64,
    lr: f32,
    warmup: usize,
    min_lr_frac: f32,
    wd: f32,
    clip: f32,
    seed: u64,
    data_seed: u64,
    eval_tokens: u64,
    eval_batch: usize,
    log_every: usize,
    out: Option<PathBuf>,
    max_steps: Option<usize>,
    bench_steps: Option<usize>,
    cpu: bool,
    train: PathBuf,
    val: PathBuf,
    bpe: PathBuf,
}

const USAGE: &str = "usage: train_gpu --d D --layers L --lr LR --out DIR [--ctx 256] [--batch 64] [--tokens 20M]\n\
    \x20            [--warmup 100] [--min-lr-frac 0.1] [--wd 0.1] [--clip 1.0] [--seed 1] [--data-seed 1]\n\
    \x20            [--eval-tokens 500000] [--eval-batch 16] [--log-every 50] [--max-steps N] [--bench-steps N]\n\
    \x20            [--cpu] [--train DIR] [--val DIR] [--bpe FILE]";

/// "4M", "500k", "1.5B" or a plain integer.
fn parse_count(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let (num, mult) = match s.chars().last() {
        Some('k') | Some('K') => (&s[..s.len() - 1], 1_000f64),
        Some('m') | Some('M') => (&s[..s.len() - 1], 1_000_000f64),
        Some('b') | Some('B') | Some('g') | Some('G') => (&s[..s.len() - 1], 1_000_000_000f64),
        _ => (s, 1f64),
    };
    let v: f64 = num.parse().map_err(|_| format!("bad count {s:?}"))?;
    Ok((v * mult).round() as u64)
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        d: 0,
        layers: 0,
        ctx: 256,
        batch: 64,
        tokens: 20_000_000,
        lr: 0.0,
        warmup: 100,
        min_lr_frac: 0.1,
        wd: 0.1,
        clip: 1.0,
        seed: 1,
        data_seed: 1,
        eval_tokens: 500_000,
        eval_batch: 16,
        log_every: 50,
        out: None,
        max_steps: None,
        bench_steps: None,
        cpu: false,
        train: PathBuf::from(ava_gpu::DEFAULT_TRAIN),
        val: PathBuf::from(ava_gpu::DEFAULT_VAL),
        bpe: PathBuf::from(ava_gpu::DEFAULT_BPE),
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
    let flt = |v: &str| v.parse::<f32>().map_err(|_| format!("bad number {v:?}"));
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.to_string());
        }
        if key == "--cpu" {
            a.cpu = true;
            i += 1;
            continue;
        }
        let val = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        match key {
            "--d" => a.d = num(val)?,
            "--layers" => a.layers = num(val)?,
            "--ctx" => a.ctx = num(val)?,
            "--batch" => a.batch = num(val)?,
            "--tokens" => a.tokens = parse_count(val)?,
            "--lr" => a.lr = flt(val)?,
            "--warmup" => a.warmup = num(val)?,
            "--min-lr-frac" => a.min_lr_frac = flt(val)?,
            "--wd" => a.wd = flt(val)?,
            "--clip" => a.clip = flt(val)?,
            "--seed" => a.seed = parse_count(val)?,
            "--data-seed" => a.data_seed = parse_count(val)?,
            "--eval-tokens" => a.eval_tokens = parse_count(val)?,
            "--eval-batch" => a.eval_batch = num(val)?.max(1),
            "--log-every" => a.log_every = num(val)?.max(1),
            "--out" => a.out = Some(PathBuf::from(val)),
            "--max-steps" => a.max_steps = Some(num(val)?),
            "--bench-steps" => a.bench_steps = Some(num(val)?),
            "--train" => a.train = PathBuf::from(val),
            "--val" => a.val = PathBuf::from(val),
            "--bpe" => a.bpe = PathBuf::from(val),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    if a.d == 0 || a.layers == 0 {
        return Err("--d and --layers are required".into());
    }
    if !(a.lr > 0.0) {
        return Err("--lr is required (positive)".into());
    }
    if a.out.is_none() && a.bench_steps.is_none() {
        return Err("--out is required (except with --bench-steps)".into());
    }
    if a.batch == 0 || a.ctx == 0 {
        return Err("--batch and --ctx must be positive".into());
    }
    Ok(a)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("train_gpu: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let a = parse_args()?;
    let dev = ava_gpu::device(a.cpu).map_err(|e| format!("device: {e}"))?;
    if let Err(e) = dev.set_seed(a.seed) {
        eprintln!("warning: could not seed the device RNG ({e}); the init is not reproducible on this backend");
    }
    println!("device: {}", if dev.is_cuda() { "cuda:0" } else { "cpu" });

    let cfg = Config::new(VOCAB, a.d, a.layers, a.ctx);
    let tok_per_step = (a.batch * a.ctx) as u64;
    let steps = (a.tokens / tok_per_step) as usize;
    if steps == 0 && a.bench_steps.is_none() {
        return Err(format!("--tokens {} gives 0 steps of {tok_per_step} tokens", a.tokens));
    }
    let steps_run = match (a.bench_steps, a.max_steps) {
        (Some(n), _) => n,
        (None, Some(m)) => m.min(steps),
        (None, None) => steps,
    };

    // ---- data
    let train_set = ShardSet::open(&a.train).map_err(|e| format!("open train shards {}: {e}", a.train.display()))?;
    let mut sampler = Sampler::new(&train_set, a.ctx, a.data_seed).map_err(|e| format!("sampler: {e}"))?;

    // ---- model + optimiser
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Model::new(cfg, vb).map_err(|e| format!("model: {e}"))?;
    let mut names: Vec<(String, candle_core::Var)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    names.sort_by(|x, y| x.0.cmp(&y.0));
    let (w_total, w_nonembed) = cfg.weight_counts();
    let counted: usize = names.iter().map(|(_, v)| v.elem_count()).sum();
    assert_eq!(counted, w_total, "weight count mismatch");
    let params: Vec<Param> = names.into_iter().map(|(name, var)| Param { decay: Model::decays(&name), name, var }).collect();
    let n_decay = params.iter().filter(|p| p.decay).count();
    let mut opt = AdamW::new(params, a.wd as f64).map_err(|e| format!("adamw: {e}"))?;
    println!("model S d={} layers={} ctx={} vocab={VOCAB}: weights {w_total} total, {w_nonembed} non-embedding; {n_decay} decayed tensors of {}", a.d, a.layers, a.ctx, opt.params().len());
    println!(
        "batch {} x ctx {} = {tok_per_step} tokens/step; schedule {steps} steps ({} tokens); running {steps_run} steps; lr {} warmup {} min-lr-frac {} wd {} clip {} seed {} data-seed {}",
        a.batch,
        a.ctx,
        steps as u64 * tok_per_step,
        a.lr,
        a.warmup,
        a.min_lr_frac,
        a.wd,
        a.clip,
        a.seed,
        a.data_seed
    );

    let bench = a.bench_steps.is_some();
    let mut log = None;
    if !bench {
        let out = a.out.as_ref().unwrap();
        fs::create_dir_all(out).map_err(|e| format!("create {}: {e}", out.display()))?;
        let mut f = fs::File::create(out.join("log.tsv")).map_err(|e| format!("log.tsv: {e}"))?;
        writeln!(f, "step\tloss\tlr\tgrad_norm\ttokens_per_sec\tstep_s\tspike_rates\tkwta_post").map_err(|e| e.to_string())?;
        log = Some(f);
    }

    // ---- training loop
    let mut losses: Vec<f32> = Vec::with_capacity(steps_run);
    let mut timed_s = 0.0f64;
    let mut timed_steps = 0u64;
    let t_train = Instant::now();
    for step in 0..steps_run {
        let t0 = Instant::now();
        let windows = sampler.batch(a.batch);
        let (ids, tg) = train_io(&windows);
        let ids_t = Tensor::from_vec(ids, (a.batch, a.ctx), &dev).map_err(|e| format!("ids: {e}"))?;
        let tg_t = Tensor::from_vec(tg, (a.batch, a.ctx), &dev).map_err(|e| format!("targets: {e}"))?;
        let lr = cosine_lr(step, a.warmup, steps, a.lr, a.lr * a.min_lr_frac);
        let (loss_t, act) = model.loss(&ids_t, &tg_t).map_err(|e| format!("step {step} forward: {e}"))?;
        let grads = loss_t.backward().map_err(|e| format!("step {step} backward: {e}"))?;
        let gnorm = opt.grad_norm(&grads).map_err(|e| format!("step {step} grad norm: {e}"))?;
        let scale = clip_scale(gnorm, a.clip as f64);
        opt.step(&grads, lr as f64, scale).map_err(|e| format!("step {step} adamw: {e}"))?;
        drop(grads);
        let loss = loss_t.to_scalar::<f32>().map_err(|e| format!("step {step} loss: {e}"))?;
        drop(loss_t);
        if !loss.is_finite() {
            return Err(format!("step {step}: loss is {loss}"));
        }
        let wall = t0.elapsed().as_secs_f64();
        losses.push(loss);
        if step > 0 || steps_run == 1 {
            timed_s += wall;
            timed_steps += 1;
        }
        let tps = tok_per_step as f64 / wall;
        let show = step % a.log_every == 0 || step + 1 == steps_run;
        let (sr, kp) = if show || log.is_some() { (Activity::fmt(&act.spike_rate), Activity::fmt(&act.kwta_post)) } else { (String::new(), String::new()) };
        if let Some(f) = log.as_mut() {
            writeln!(f, "{step}\t{loss:.6}\t{lr:.6e}\t{gnorm:.6}\t{tps:.1}\t{wall:.4}\t{sr}\t{kp}").map_err(|e| e.to_string())?;
        }
        if show {
            println!("step {step}/{steps_run} loss {loss:.4} lr {lr:.3e} gnorm {gnorm:.3} {tps:.0} tok/s ({wall:.3}s) spikes [{sr}] kwta+ [{kp}]");
        }
        if !bench && (step + 1) % 500 == 0 && step + 1 != steps_run {
            let p = a.out.as_ref().unwrap().join("model.safetensors");
            varmap.save(&p).map_err(|e| format!("save {}: {e}", p.display()))?;
            if let Some(f) = log.as_mut() {
                f.flush().ok();
            }
        }
    }
    let train_s = t_train.elapsed().as_secs_f64();
    let tps_timed = if timed_steps > 0 { timed_steps as f64 * tok_per_step as f64 / timed_s } else { f64::NAN };
    let n50 = losses.len().min(50);
    let last50 = losses[losses.len() - n50..].iter().map(|&x| x as f64).sum::<f64>() / n50.max(1) as f64;
    println!("trained {steps_run} steps in {train_s:.1}s: {tps_timed:.0} tok/s (first step excluded), train_loss_last50 {last50:.4}");
    if bench {
        println!("bench: d={} layers={} ctx={} batch={} tokens_per_sec={tps_timed:.1} step_s={:.4}", a.d, a.layers, a.ctx, a.batch, timed_s / timed_steps.max(1) as f64);
        return Ok(());
    }

    // ---- save
    let out = a.out.as_ref().unwrap();
    let model_path = out.join("model.safetensors");
    varmap.save(&model_path).map_err(|e| format!("save {}: {e}", model_path.display()))?;
    if let Some(f) = log.as_mut() {
        f.flush().ok();
    }

    // ---- validation
    let bpe = Bpe::load(&a.bpe).map_err(|e| format!("bpe {}: {e}", a.bpe.display()))?;
    let byte_len = byte_lengths(&bpe);
    let val_set = ShardSet::open(&a.val).map_err(|e| format!("open val shards {}: {e}", a.val.display()))?;
    let t_eval = Instant::now();
    let (acc, windows) = evaluate(&model, val_set.iter_sequential(a.ctx), a.eval_batch, a.eval_tokens, &byte_len, &dev).map_err(|e| format!("eval: {e}"))?;
    let eval_s = t_eval.elapsed().as_secs_f64();
    println!("val: {} targets ({} bytes, {windows} windows): {:.4} nats/token, {:.4} bits/byte; eval {eval_s:.1}s", acc.tokens, acc.bytes, acc.nats_per_token(), acc.bpb());
    if acc.tokens < a.eval_tokens {
        println!("note: the val set ran out after {} scored targets (asked for {})", acc.tokens, a.eval_tokens);
    }

    // ---- result line
    let res_path = out.join("result.tsv");
    let new_file = fs::metadata(&res_path).map(|m| m.len() == 0).unwrap_or(true);
    let mut f = OpenOptions::new().create(true).append(true).open(&res_path).map_err(|e| format!("result.tsv: {e}"))?;
    if new_file {
        writeln!(
            f,
            "model\tdevice\td\tlayers\tctx\tbatch\ttokens\tsteps\tsteps_run\tlr\twarmup\tmin_lr_frac\twd\tclip\tseed\tdata_seed\teval_tokens\t\
             weights_total\tweights_nonembed\ttrain_loss_last50\tval_loss_nats_per_token\tval_bpb\tval_tokens_scored\tval_bytes\ttrain_seconds\teval_seconds\ttokens_per_sec"
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(
        f,
        "S\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{:.2}\t{:.2}\t{:.1}",
        if dev.is_cuda() { "cuda" } else { "cpu" },
        a.d,
        a.layers,
        a.ctx,
        a.batch,
        a.tokens,
        steps,
        steps_run,
        a.lr,
        a.warmup,
        a.min_lr_frac,
        a.wd,
        a.clip,
        a.seed,
        a.data_seed,
        a.eval_tokens,
        w_total,
        w_nonembed,
        last50,
        acc.nats_per_token(),
        acc.bpb(),
        acc.tokens,
        acc.bytes,
        train_s,
        eval_s,
        tps_timed
    )
    .map_err(|e| e.to_string())?;
    println!("wrote {} and {}", res_path.display(), model_path.display());
    Ok(())
}
