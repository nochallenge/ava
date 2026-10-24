// TESTING / DESIGN ONLY — train: train one of the pre-registered models (T, S, H) on the BPE-4096
// shards, evaluate validation bits per byte, append one result line, save the checkpoint.
//
// train --model T|S|H --d D --layers L --ctx 128 --batch 32 --tokens 4M --lr LR --warmup 50
//       --min-lr-frac 0.1 --wd 0.1 --clip 1.0 --seed 1 --data-seed 1 --threads 4
//       --eval-tokens 500000 --log-every 50 --out DIR [--max-steps N] [--bench-steps N]
//       [--train DIR] [--val DIR] [--bpe FILE]
//
// * steps = tokens / (batch · ctx), rounded down; the LR schedule is cosine_lr over `steps`.
//   --max-steps N stops after min(N, steps) steps WITHOUT changing the schedule, so a truncated run is
//   exactly the first N steps of the full run.
// * Data: Sampler::new(train, ctx, data_seed), `batch` windows of ctx + 1 tokens per step.
// * Logs: stdout every --log-every steps (plus the first and last step); DIR/log.tsv gets every step:
//   step, loss, lr, grad_norm, tokens_per_sec (+ fwd/bwd/opt seconds).
// * At the end: validation on iter_sequential(ctx) of the val shards until --eval-tokens targets are
//   scored (PAD targets skipped, EOS = 1 byte), one line appended to DIR/result.tsv (header written
//   when the file is new), checkpoint DIR/model.awts (Params::save).
// * --bench-steps N: run N steps, print throughput and the time split, write nothing, no evaluation.
// Deterministic for a given seed, data seed and thread count.

use ava_brain::eval::{evaluate, fmt_list};
use ava_brain::model::OP_CATS;
use ava_brain::sys::peak_rss_mib;
use ava_brain::{byte_lengths, train_io, Config, Hyper, Kind, Model, OpTimes, Trainer};
use ava_data::bpe::Bpe;
use ava_data::input::parse_count;
use ava_data::shards::{Sampler, ShardSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

// defaults: the ava folder's data/nn, found from this crate (design-tests/nn/brain)
const DEFAULT_TRAIN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/nn/shards-4096/train");
const DEFAULT_VAL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/nn/shards-4096/val");
const DEFAULT_BPE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/nn/bpe-4096.txt");
const VOCAB: usize = 4096;

#[derive(Clone, Debug)]
struct Args {
    model: Kind,
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
    threads: usize,
    eval_tokens: u64,
    log_every: usize,
    out: Option<PathBuf>,
    max_steps: Option<usize>,
    bench_steps: Option<usize>,
    train: PathBuf,
    val: PathBuf,
    bpe: PathBuf,
}

fn usage() -> ! {
    eprintln!(
        "usage: train --model T|S|H --d D --layers L --lr LR --out DIR [--ctx 128] [--batch 32] [--tokens 4M]\n\
         \x20            [--warmup 50] [--min-lr-frac 0.1] [--wd 0.1] [--clip 1.0] [--seed 1] [--data-seed 1]\n\
         \x20            [--threads 4] [--eval-tokens 500000] [--log-every 50] [--max-steps N] [--bench-steps N]\n\
         \x20            [--train DIR] [--val DIR] [--bpe FILE]"
    );
    std::process::exit(2)
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        model: Kind::T,
        d: 0,
        layers: 0,
        ctx: 128,
        batch: 32,
        tokens: 4_000_000,
        lr: f32::NAN,
        warmup: 50,
        min_lr_frac: 0.1,
        wd: 0.1,
        clip: 1.0,
        seed: 1,
        data_seed: 1,
        threads: 4,
        eval_tokens: 500_000,
        log_every: 50,
        out: None,
        max_steps: None,
        bench_steps: None,
        train: PathBuf::from(DEFAULT_TRAIN),
        val: PathBuf::from(DEFAULT_VAL),
        bpe: PathBuf::from(DEFAULT_BPE),
    };
    let mut have_model = false;
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        if key == "-h" || key == "--help" {
            usage();
        }
        let val = argv.get(i + 1).ok_or_else(|| format!("{key} needs a value"))?.as_str();
        let num = |v: &str| -> Result<usize, String> { parse_count(v).map(|x| x as usize).map_err(|e| format!("{key}: {e}")) };
        let flt = |v: &str| -> Result<f32, String> { v.parse::<f32>().map_err(|_| format!("{key}: bad number {v:?}")) };
        match key {
            "--model" => {
                a.model = Kind::parse(val)?;
                have_model = true;
            }
            "--d" => a.d = num(val)?,
            "--layers" => a.layers = num(val)?,
            "--ctx" => a.ctx = num(val)?,
            "--batch" => a.batch = num(val)?,
            "--tokens" => a.tokens = parse_count(val).map_err(|e| format!("--tokens: {e}"))?,
            "--lr" => a.lr = flt(val)?,
            "--warmup" => a.warmup = num(val)?,
            "--min-lr-frac" => a.min_lr_frac = flt(val)?,
            "--wd" => a.wd = flt(val)?,
            "--clip" => a.clip = flt(val)?,
            "--seed" => a.seed = parse_count(val).map_err(|e| format!("--seed: {e}"))?,
            "--data-seed" => a.data_seed = parse_count(val).map_err(|e| format!("--data-seed: {e}"))?,
            "--threads" => a.threads = num(val)?,
            "--eval-tokens" => a.eval_tokens = parse_count(val).map_err(|e| format!("--eval-tokens: {e}"))?,
            "--log-every" => a.log_every = num(val)?.max(1),
            "--out" => a.out = Some(PathBuf::from(val)),
            "--max-steps" => a.max_steps = Some(num(val)?),
            "--bench-steps" => a.bench_steps = Some(num(val)?),
            "--train" => a.train = PathBuf::from(val),
            "--val" => a.val = PathBuf::from(val),
            "--bpe" => a.bpe = PathBuf::from(val),
            _ => return Err(format!("unknown option {key:?}")),
        }
        i += 2;
    }
    if !have_model {
        return Err("--model is required".into());
    }
    if a.d == 0 || a.layers == 0 {
        return Err("--d and --layers are required".into());
    }
    if !a.lr.is_finite() || a.lr <= 0.0 {
        return Err("--lr is required (positive)".into());
    }
    if a.bench_steps.is_none() && a.out.is_none() {
        return Err("--out is required (except with --bench-steps)".into());
    }
    if a.threads == 0 || a.batch == 0 || a.ctx == 0 {
        return Err("--threads, --batch and --ctx must be positive".into());
    }
    Ok(a)
}

fn fmt_ops(t: &OpTimes) -> String {
    let tot = t.total().max(1e-12);
    OP_CATS.iter().zip(&t.secs).map(|(n, s)| format!("{n} {:.0}%", 100.0 * s / tot)).collect::<Vec<_>>().join(", ")
}

fn main() {
    let args = parse_args().unwrap_or_else(|e| {
        eprintln!("train: {e}");
        usage()
    });
    if let Err(e) = run(&args) {
        eprintln!("train: {e}");
        std::process::exit(1);
    }
}

fn run(a: &Args) -> Result<(), String> {
    let t_start = Instant::now();
    let cfg = Config { kind: a.model, vocab: VOCAB, d: a.d, layers: a.layers, ctx: a.ctx };
    cfg.validate()?;
    let tok_per_step = (a.batch * a.ctx) as u64;
    let steps = (a.tokens / tok_per_step) as usize;
    if steps == 0 && a.bench_steps.is_none() {
        return Err(format!("--tokens {} gives 0 steps of {} tokens", a.tokens, tok_per_step));
    }
    let steps_run = match (a.bench_steps, a.max_steps) {
        (Some(n), _) => n,
        (None, Some(m)) => m.min(steps),
        (None, None) => steps,
    };
    let hyper = Hyper { lr: a.lr, warmup: a.warmup, min_lr_frac: a.min_lr_frac, wd: a.wd, clip: a.clip, total_steps: steps.max(1), threads: a.threads };

    let train_set = ShardSet::open(&a.train).map_err(|e| format!("open {}: {e}", a.train.display()))?;
    if train_set.vocab_size as usize != VOCAB {
        return Err(format!("train shards have vocab {}, expected {VOCAB}", train_set.vocab_size));
    }
    let mut sampler = Sampler::new(&train_set, a.ctx, a.data_seed).map_err(|e| format!("sampler: {e}"))?;

    let mut tr = Trainer::new(cfg, a.seed, hyper);
    let (w_total, w_nonembed) = Model::weight_counts(&tr.params);
    println!(
        "model {} d={} layers={} ctx={} vocab={VOCAB}: weights {w_total} total, {w_nonembed} non-embedding",
        a.model.name(),
        a.d,
        a.layers,
        a.ctx
    );
    println!(
        "batch {} x ctx {} = {tok_per_step} tokens/step; schedule {steps} steps ({} tokens); running {steps_run} steps; lr {} warmup {} min-lr-frac {} wd {} clip {} seed {} data-seed {} threads {}",
        a.batch,
        a.ctx,
        steps as u64 * tok_per_step,
        a.lr,
        a.warmup,
        a.min_lr_frac,
        a.wd,
        a.clip,
        a.seed,
        a.data_seed,
        a.threads
    );

    let bench = a.bench_steps.is_some();
    let mut log = None;
    if !bench {
        let out = a.out.as_ref().unwrap();
        fs::create_dir_all(out).map_err(|e| format!("create {}: {e}", out.display()))?;
        let mut f = fs::File::create(out.join("log.tsv")).map_err(|e| format!("log.tsv: {e}"))?;
        writeln!(f, "step\tloss\tlr\tgrad_norm\ttokens_per_sec\tfwd_s\tbwd_s\topt_s\tdata_s").map_err(|e| e.to_string())?;
        log = Some(f);
    }

    let mut losses: Vec<f32> = Vec::with_capacity(steps_run);
    let mut ops_sum = OpTimes::default();
    let (mut fwd_sum, mut bwd_sum, mut opt_sum, mut data_sum, mut wall_sum) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut win_loss = 0.0f64;
    let mut win_n = 0usize;
    let mut win_t = Instant::now();
    let t_train = Instant::now();
    for step in 0..steps_run {
        let t0 = Instant::now();
        let windows = sampler.batch(a.batch);
        let (ids, tg) = train_io(&windows);
        let data_s = t0.elapsed().as_secs_f64();
        let s = tr.train_step(&ids, &tg, a.batch, a.ctx);
        let wall = t0.elapsed().as_secs_f64();
        if !s.loss.is_finite() {
            return Err(format!("step {step}: loss is {}", s.loss));
        }
        losses.push(s.loss);
        // the first step is excluded from the timing averages (allocator / page-fault warm-up)
        if step > 0 || steps_run == 1 {
            ops_sum.add(&s.fwd_ops);
            fwd_sum += s.fwd_s;
            bwd_sum += s.bwd_s;
            opt_sum += s.opt_s;
            data_sum += data_s;
            wall_sum += wall;
        }
        let tps = tok_per_step as f64 / wall;
        if let Some(f) = log.as_mut() {
            writeln!(f, "{step}\t{:.6}\t{:.6e}\t{:.6}\t{:.1}\t{:.4}\t{:.4}\t{:.4}\t{:.4}", s.loss, s.lr, s.grad_norm, tps, s.fwd_s, s.bwd_s, s.opt_s, data_s)
                .map_err(|e| e.to_string())?;
        }
        win_loss += s.loss as f64;
        win_n += 1;
        if step == 0 || (step + 1) % a.log_every == 0 || step + 1 == steps_run || bench {
            let el = win_t.elapsed().as_secs_f64();
            println!(
                "step {:>5}/{steps_run}  loss {:.4}  mean{} {:.4}  lr {:.3e}  gnorm {:.3}  tok/s {:.0}  (fwd {:.3}s bwd {:.3}s opt {:.3}s data {:.3}s)",
                step + 1,
                s.loss,
                win_n,
                win_loss / win_n as f64,
                s.lr,
                s.grad_norm,
                (win_n as u64 * tok_per_step) as f64 / el,
                s.fwd_s,
                s.bwd_s,
                s.opt_s,
                data_s
            );
            win_loss = 0.0;
            win_n = 0;
            win_t = Instant::now();
        }
    }
    let train_s = t_train.elapsed().as_secs_f64();
    let timed_steps = if steps_run > 1 { steps_run - 1 } else { steps_run };
    let tps_steady = (timed_steps as u64 * tok_per_step) as f64 / wall_sum.max(1e-12);
    let tps_all = (steps_run as u64 * tok_per_step) as f64 / train_s.max(1e-12);
    let per = |x: f64| x / timed_steps.max(1) as f64;
    println!(
        "timing ({} steps after the first): {:.3} s/step = fwd {:.3} + bwd {:.3} + opt {:.3} + data {:.3}; tokens/s {:.0} (all steps incl. first: {:.0}); 4M tokens -> {:.1} min",
        timed_steps,
        per(wall_sum),
        per(fwd_sum),
        per(bwd_sum),
        per(opt_sum),
        per(data_sum),
        tps_steady,
        tps_all,
        4.0e6 / tps_steady / 60.0
    );
    println!("forward op split: {}", fmt_ops(&ops_sum));
    if let Some(mb) = peak_rss_mib() {
        println!("peak resident memory {mb:.0} MiB");
    }
    if bench {
        return Ok(());
    }

    let out = a.out.as_ref().unwrap();
    let n_last = losses.len().min(50);
    let last50 = losses[losses.len() - n_last..].iter().map(|&x| x as f64).sum::<f64>() / n_last.max(1) as f64;

    // ---- checkpoint
    tr.params.save(out.join("model.awts")).map_err(|e| format!("save checkpoint: {e}"))?;

    // ---- validation
    let t_eval = Instant::now();
    let bpe = Bpe::load(&a.bpe).map_err(|e| format!("load {}: {e}", a.bpe.display()))?;
    if bpe.vocab_size() != VOCAB {
        return Err(format!("tokenizer vocab {} != {VOCAB}", bpe.vocab_size()));
    }
    let byte_len = byte_lengths(&bpe);
    let val_set = ShardSet::open(&a.val).map_err(|e| format!("open {}: {e}", a.val.display()))?;
    let ev = evaluate(&tr.model, &tr.params, val_set.iter_sequential(a.ctx), a.batch, a.eval_tokens, &byte_len, a.threads);
    let eval_s = t_eval.elapsed().as_secs_f64();
    let seconds = t_start.elapsed().as_secs_f64();
    println!(
        "val: {} targets ({} bytes, {} windows): {:.4} nats/token, {:.4} bits/byte; spike rates [{}]; kWTA active pre-relu [{}] post-relu [{}]; eval {:.1}s",
        ev.acc.tokens,
        ev.acc.bytes,
        ev.windows,
        ev.acc.nats_per_token(),
        ev.acc.bpb(),
        fmt_list(&ev.spike_rates),
        fmt_list(&ev.kwta_pre),
        fmt_list(&ev.kwta_post),
        eval_s
    );
    if ev.acc.tokens < a.eval_tokens {
        println!("note: the val set ran out after {} scored targets (asked for {})", ev.acc.tokens, a.eval_tokens);
    }

    // ---- result line
    let res_path = out.join("result.tsv");
    let new_file = fs::metadata(&res_path).map(|m| m.len() == 0).unwrap_or(true);
    let mut f = OpenOptions::new().create(true).append(true).open(&res_path).map_err(|e| format!("result.tsv: {e}"))?;
    if new_file {
        writeln!(
            f,
            "model\td\tlayers\tctx\tbatch\ttokens\tsteps\tsteps_run\tlr\twarmup\tmin_lr_frac\twd\tclip\tseed\tdata_seed\tthreads\teval_tokens\t\
             weights_total\tweights_nonembed\ttrain_loss_last50\tval_loss_nats_per_token\tval_bpb\tval_tokens_scored\tval_bytes\t\
             spike_rates\tkwta_active\tkwta_active_post_relu\ttrain_seconds\teval_seconds\tseconds\ttokens_per_sec\tpeak_rss_mib"
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(
        f,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.1}\t{:.0}",
        a.model.name(),
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
        a.threads,
        a.eval_tokens,
        w_total,
        w_nonembed,
        last50,
        ev.acc.nats_per_token(),
        ev.acc.bpb(),
        ev.acc.tokens,
        ev.acc.bytes,
        fmt_list(&ev.spike_rates),
        fmt_list(&ev.kwta_pre),
        fmt_list(&ev.kwta_post),
        train_s,
        eval_s,
        seconds,
        tps_all,
        peak_rss_mib().unwrap_or(f64::NAN)
    )
    .map_err(|e| e.to_string())?;
    println!("train_loss_last50 {last50:.4}; wrote {} and {}; total {seconds:.1}s", res_path.display(), out.join("model.awts").display());
    Ok(())
}
