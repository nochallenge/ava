// TESTING / DESIGN ONLY — train_gpu: train Ava on candle (CUDA by default, --cpu for the CPU backend).
//
// train_gpu --d D --layers L --ctx 256 --batch B --tokens N --lr LR --warmup W --min-lr-frac 0.1
//           --wd 0.1 --clip 1.0 --seed 1 --data-seed 1 --eval-tokens 500000 --log-every 50 --out DIR
//           [--bench-steps N] [--cpu] [--max-steps N] [--eval-batch 16] [--train DIR] [--val DIR] [--bpe FILE]
//
// * steps = tokens / (batch · ctx); step s uses lr = cosine_lr(s, warmup, steps, lr, lr · min_lr_frac).
// * Logs: stdout every --log-every steps (plus first and last); DIR/log.tsv gets every step
//   (step, loss, lr, grad_norm before clipping, tokens/s, step seconds, spike rates, kWTA post-relu).
// * DIR/model.safetensors is written every --ckpt-every steps (500) and at the end (candle VarMap::save).
// * Every --ckpt-every steps a full checkpoint goes to DIR/ckpt/: weights, the AdamW moments and step
//   count, the data cursor and the run-defining settings (state.txt, written last). --resume continues
//   from it exactly (same batches, same updates) if the settings match; ckpt/ is removed once the whole schedule
//   has run (a --max-steps run keeps a checkpoint at its last step, so it can be continued).
//   After a resume, train_seconds and tokens_per_sec cover only the steps run by that process.
// * At the end: validation BPB on iter_sequential(ctx) of the val shards (first --eval-tokens targets,
//   PAD skipped, EOS = 1 byte), one line appended to DIR/result.tsv.
// * --bench-steps N: run N steps, print tokens/s (first step excluded), write nothing, no evaluation.
// * v44 --gpu-share F: after each step wait for the GPU, then rest elapsed·(1/F − 1) (at most 10 s), so the GPU is
//   busy about F of the time while someone else uses it. Same numbers; not in the resume signature; the rests count
//   in train_seconds but not in tokens/s.
// * v39 --glance (model D only): the shards carry a length mark before every digit run (vocab 4112, see
//   ava_data::numbers) and each digit gets a learned place embedding. The train and val shards must say vocab
//   4112 (and 4096 without --glance). Marks are 0 bytes, so their loss counts against her BPB. result.tsv gains
//   a last column `glance` (only on glance runs); the resume signature gains glance=1.
// * v42: --beta2 B sets AdamW's beta2 (default 0.95); --decay-gates also decays the R/D/A memory's output gate .wg
//   (a bug fix, off by default so past runs stay reproducible; v42 runs should turn it on). Each joins the resume
//   signature only when set.
// * v42 Wave 2B. Every option is off by default, so an older command line computes exactly what it did. Each one
//   that is set joins the resume signature and a last result.tsv column `v42` (only on such runs).
//   Recipe:
//   - --schedule cosine|wsd [--decay-frac 0.2] [--decay-shape 1-sqrt|linear] [--lr-min X] (A1): WSD holds the peak
//     LR and anneals over the last decay-frac of the steps to the floor (--lr-min, default lr · min-lr-frac).
//   - --opt adamw|muon [--muon-lr 0.02] (A1): Muon (Newton–Schulz, 5 quintic steps, Nesterov 0.95) for the decayed
//     2-D hidden matrices, at its own peak LR on the same schedule; AdamW keeps the embedding, gains, biases and the
//     small gate projections. A later item: muP-style width transfer of these LRs.
//   - --ctx-warmup TOKENS [--ctx-start 256] (I2): the context starts at ctx-start and doubles in equal stages to
//     --ctx over the first TOKENS. Each window is cut into pieces of the stage's length, so the targets and the
//     tokens per step stay the same and the batch grows.
//   - --grokfast-alpha A --grokfast-lambda L (I4): Grokfast-EMA on every gradient before clipping (on when L > 0;
//     its averages are kept in the checkpoint).
//   Block options (model D; src/arch.rs; candle's autograd, except Stage 3's winners, the slow five and --neuromod 3:
//   the hand-written WIN step, src/model/fwin.rs; the first lines say which):
//     --head-size, --timescales, --gates, --value-residual, --kwta, --mlp-act, --mem-momentum, --feedback,
//     --resid-fix, --residual (plain|dense|attn), --lif-init, --lif-scale, --surrogate-slope, --spikes, --alif, --homeostasis,
//     v44 --gate silu|sigmoid and --gate-rank R (src/arch.rs K1, K2), and
//     v50 --dale FRAC, --loop N and --neuromod 3 (src/v50.rs; --dale's weights decay through Model::dale_decay), and
//     --top-weight W (R2: token-order prediction; the logged loss stays the next-token loss). A run with block
//     options writes DIR/arch.txt (the flags; Arch::from_flags reads it back).
//   v42 Wave 2C memory and parts flags (src/memory.rs: --eng-topk, --eng-temp, --eng-ssmax, --btsp, --mem-reset,
//     --recall-loss, --pk-mem/--pk-layer/--pk-topk, --dendrites, --stp, --delays, --wm; src/wave5.rs: --resonate,
//     --modulate, --tem, --halting, --cerebellum, --jepa, --diffusion, --critic, --confidence) are block options too.
//     --grad-shares (O2) prints each part's share of the squared gradient norm at the logged steps.
// * v42 plasticity (src/plastic.rs; off by default, each joins the resume signature and the v42 column when set):
//   - --critical-periods FIRST,GAP,TAU,FLOOR (H7): block l's LR (and its decay) times 1 until step FIRST + l·GAP,
//     then FLOOR + (1 − FLOOR)·exp(−(step − close)/TAU). Embeddings, final norm and aux heads keep the full LR.
//   - --rewire DENSITY,FRAC,EVERY (H6): each block's mlp.w1 and mlp.w2 keep a share DENSITY of live weights (a random
//     mask at step 0; dead weights 0 after every step). After every EVERY-th update the FRAC weakest live weights are
//     pruned and as many grow at random dead places, at 0, with zero moments. The masks are kept in the checkpoint.
//   - --redo TAU,EVERY,BATCHES[,reset] (I3, src/upkeep.rs): after every EVERY-th update, each unit's activity is
//     measured on BATCHES batches of training windows (a sampler of its own, seeded from --data-seed, its cursor kept
//     in the checkpoint, so the training batches stay the same); the LIF channels and MLP units at or below TAU of
//     their layer's mean get fresh incoming weights and zero outgoing ones (under --rewire, only at live places), and
//     the optimizer's moments are zeroed where weights were written. Any Ava model, with or without block options.
//     With reset, a revived LIF channel's per-channel tensors (leak, threshold, ...) go back to their init too (off
//     by default: see src/upkeep.rs).
//   - --lp-mix EPS[,EVERY[,CAP[,PROBE[,END]]]] (B6, ava_data::lp_mix; END: B6b, the tilt fades to the natural mix at that fraction): the batches come from a learning-progress mixture
//     over the sources of train/tags.tsv instead of the shard sampler (a window belongs to the source holding more
//     than half of it, else to "mixed"). After every EVERY-th step (default 250) her loss is measured on PROBE
//     (default 8) held-out windows of each class; the classes whose loss moves fastest get more of the batch, within
//     (1 - EPS) of their natural share and at most CAP (default 4) times it. EPS 0 is the natural mix drawn this way
//     (the control arm). The mixture's cursors and probe losses are kept in the checkpoint.

use ava_data::bpe::Bpe;
use ava_data::lp_mix::{read_tags, LpCfg, LpMix, SourceIndex, WindowReader};
use ava_data::numbers::VOCAB_GLANCE;
use ava_data::shards::{Cursor, Sampler, ShardSet};
use ava_gpu::arch::Arch;
use ava_gpu::eval::{byte_lengths_n, evaluate, train_io};
use ava_gpu::model::{Activity, Config, Kind, Model, VOCAB};
use ava_gpu::optim::{clip_scale, clip_scale_t, AdamW, MuonCfg, Param, Schedule};
use ava_gpu::plastic::{critical_signature, parse_critical, CriticalPeriods, Event, RewireCfg, Rewiring};
use ava_gpu::trainkit::{ctx_at, ctx_warmup_check, split_windows, Grokfast};
use ava_gpu::upkeep::{ChannelInit, RedoCfg};
use ava_data::rng::derive_seed;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    kind: Kind,
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
    /// v44: --gpu-share F: after each step wait for the GPU, then rest so it is busy about F of the time (1: off).
    /// Same numbers, only slower; not in any signature; the rests are left out of the timed tok/s.
    gpu_share: f64,
    cpu: bool,
    /// K4 (many GPUs, src/dist.rs): this process's rank, the number of processes, and rank 0's address
    rank: usize,
    world: usize,
    dist_addr: String,
    ckpt_every: usize,
    resume: bool,
    train: PathBuf,
    val: PathBuf,
    bpe: PathBuf,
    glance: bool,
    /// v42 (Q1): AdamW's beta2 (0.95, as before)
    beta2: f64,
    /// v42 (W1, a bug kept off by default): the memory's output gate .wg gets weight decay too (Model::decays_with)
    decay_gates: bool,
    /// v42 (A1): the LR schedule (cosine, as before) and its floor (--lr-min; default lr · min_lr_frac)
    schedule: Schedule,
    lr_min: Option<f32>,
    /// v42 (A1): Muon's peak LR when --opt muon
    muon: Option<f64>,
    /// v42 (I2): context warm-up over this many tokens (0: off), from this length
    ctx_warmup: u64,
    ctx_start: usize,
    /// v42 (I4): Grokfast-EMA (off unless lambda > 0)
    gf_alpha: f64,
    gf_lambda: f64,
    /// v42 (Wave 2B): the block options (today's D by default) and the token-order loss weight (R2)
    arch: Arch,
    top_weight: f64,
    /// v42 Wave 2C (O2): print each part's share of the gradient at logged steps
    grad_shares: bool,
    /// v42 plasticity: H7 critical periods and H6 rewiring (None: off)
    crit: Option<CriticalPeriods>,
    rewire: Option<RewireCfg>,
    /// v42 (I3): ReDo during training (None: off)
    redo: Option<RedoCfg>,
    /// v42 (B6): learning-progress data choice (None: the shard sampler)
    lp_mix: Option<LpCfg>,
    /// v45: the engram read in block L (model D; grow_gpu's flag), for --eng-gate's comparison (None: no engram)
    engram: Option<usize>,
}

const USAGE: &str = "usage: train_gpu [--model S|T] --d D --layers L --lr LR --out DIR [--ctx 256] [--batch 64] [--tokens 20M]\n\
    \x20            [--engram L]\n\
    \x20            [--warmup 100] [--min-lr-frac 0.1] [--wd 0.1] [--clip 1.0] [--seed 1] [--data-seed 1]\n\
    \x20            [--eval-tokens 500000] [--eval-batch 16] [--log-every 50] [--max-steps N] [--bench-steps N] [--gpu-share 1]\n\
    \x20            [--cpu] [--train DIR] [--val DIR] [--bpe FILE] [--ckpt-every 500] [--resume] [--glance]\n\
    \x20            [--beta2 0.95] [--decay-gates]\n\
    \x20            [--schedule cosine|wsd] [--decay-frac 0.2] [--decay-shape 1-sqrt|linear] [--lr-min X]\n\
    \x20            [--opt adamw|muon] [--muon-lr 0.02] [--ctx-warmup TOKENS] [--ctx-start 256]\n\
    \x20            [--grokfast-alpha 0.98] [--grokfast-lambda 0] [--top-weight 0] [--grad-shares]\n\
    \x20            [--critical-periods FIRST,GAP,TAU,FLOOR] [--rewire DENSITY,FRAC,EVERY] [--redo TAU,EVERY,BATCHES[,reset]]\n\
    \x20            [--lp-mix EPS[,EVERY[,CAP[,PROBE[,END]]]]]\n\
    \x20            [--rank 0 --world 1 --dist-addr 127.0.0.1:29541]  (K4: one process per GPU; see src/dist.rs)\n\
    \x20            block options (model D): see src/arch.rs and src/memory.rs";

/// The v42 settings that are set, as " key=value ..." (empty when none is): for the resume signature and result.tsv.
fn v42_sig(a: &Args) -> String {
    let mut s = a.schedule.signature();
    if let Some(x) = a.lr_min {
        s += &format!(" lr_min={x}");
    }
    if let Some(x) = a.muon {
        s += &format!(" opt=muon muon_lr={x}");
    }
    if a.ctx_warmup > 0 {
        s += &format!(" ctx_warmup={} ctx_start={}", a.ctx_warmup, a.ctx_start);
    }
    if a.gf_lambda > 0.0 {
        s += &format!(" grokfast={}/{}", a.gf_alpha, a.gf_lambda);
    }
    s += &a.arch.signature();
    if a.top_weight > 0.0 {
        s += &format!(" top_weight={}", a.top_weight);
    }
    if let Some(c) = &a.crit {
        s += &critical_signature(c);
    }
    if let Some(r) = &a.rewire {
        s += &r.signature();
    }
    if let Some(r) = &a.redo {
        s += &r.signature();
    }
    if let Some(c) = &a.lp_mix {
        s += &format!(" lp_mix={},{},{},{},{}", c.eps, c.every, c.cap, c.probe, c.end);
    }
    s
}

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
        kind: Kind::S,
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
        gpu_share: 1.0,
        cpu: false,
        rank: 0,
        world: 1,
        dist_addr: "127.0.0.1:29541".into(),
        ckpt_every: 500,
        resume: false,
        train: PathBuf::from(ava_gpu::DEFAULT_TRAIN),
        val: PathBuf::from(ava_gpu::DEFAULT_VAL),
        bpe: PathBuf::from(ava_gpu::DEFAULT_BPE),
        glance: false,
        beta2: 0.95,
        decay_gates: false,
        schedule: Schedule::Cosine,
        lr_min: None,
        muon: None,
        ctx_warmup: 0,
        ctx_start: 256,
        gf_alpha: 0.98,
        gf_lambda: 0.0,
        arch: Arch::default(),
        top_weight: 0.0,
        grad_shares: false,
        crit: None,
        rewire: None,
        redo: None,
        lp_mix: None,
        engram: None,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
    let flt = |v: &str| v.parse::<f32>().map_err(|_| format!("bad number {v:?}"));
    let f64v = |v: &str| v.parse::<f64>().map_err(|_| format!("bad number {v:?}"));
    let (mut sched_kind, mut decay_frac, mut decay_shape) = ("cosine".to_string(), 0.2f64, "1-sqrt".to_string());
    let (mut opt_kind, mut muon_lr) = ("adamw".to_string(), 0.02f64);
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.to_string());
        }
        // v42 Wave 2B block options (src/arch.rs)
        if let Some(n) = a.arch.take_flag(&args, i)? {
            i += n;
            continue;
        }
        if key == "--grad-shares" {
            a.grad_shares = true;
            i += 1;
            continue;
        }
        if key == "--cpu" || key == "--resume" || key == "--glance" || key == "--decay-gates" {
            match key {
                "--cpu" => a.cpu = true,
                "--resume" => a.resume = true,
                "--decay-gates" => a.decay_gates = true,
                _ => a.glance = true,
            }
            i += 1;
            continue;
        }
        let val = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        match key {
            "--model" => a.kind = Kind::parse(val)?,
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
            "--rank" => a.rank = num(val)?,
            "--world" => a.world = num(val)?.max(1),
            "--dist-addr" => a.dist_addr = val.into(),
            "--data-seed" => a.data_seed = parse_count(val)?,
            "--eval-tokens" => a.eval_tokens = parse_count(val)?,
            "--eval-batch" => a.eval_batch = num(val)?.max(1),
            "--log-every" => a.log_every = num(val)?.max(1),
            "--out" => a.out = Some(PathBuf::from(val)),
            "--max-steps" => a.max_steps = Some(num(val)?),
            "--bench-steps" => a.bench_steps = Some(num(val)?),
            "--gpu-share" => a.gpu_share = f64v(val)?.clamp(0.05, 1.0),
            "--ckpt-every" => a.ckpt_every = num(val)?.max(1),
            "--train" => a.train = PathBuf::from(val),
            "--val" => a.val = PathBuf::from(val),
            "--bpe" => a.bpe = PathBuf::from(val),
            "--beta2" => a.beta2 = val.parse::<f64>().map_err(|_| format!("bad number {val:?}"))?,
            "--schedule" => sched_kind = val.to_string(),
            "--decay-frac" => decay_frac = f64v(val)?,
            "--decay-shape" => decay_shape = val.to_string(),
            "--lr-min" => a.lr_min = Some(flt(val)?),
            "--opt" => opt_kind = val.to_string(),
            "--muon-lr" => muon_lr = f64v(val)?,
            "--ctx-warmup" => a.ctx_warmup = parse_count(val)?,
            "--ctx-start" => a.ctx_start = num(val)?,
            "--grokfast-alpha" => a.gf_alpha = f64v(val)?,
            "--grokfast-lambda" => a.gf_lambda = f64v(val)?,
            "--top-weight" => a.top_weight = f64v(val)?,
            "--critical-periods" => a.crit = Some(parse_critical(val)?),
            "--rewire" => a.rewire = Some(RewireCfg::parse(val)?),
            "--redo" => a.redo = Some(RedoCfg::parse(val)?),
            "--lp-mix" => a.lp_mix = Some(LpCfg::parse(val)?),
            "--engram" => a.engram = Some(num(val)?),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    if !(a.beta2 > 0.0 && a.beta2 < 1.0) {
        return Err(format!("--beta2 {}: must be in (0, 1)", a.beta2));
    }
    // v42 Wave 2B checks
    a.schedule = Schedule::parse(&sched_kind, decay_frac, &decay_shape)?;
    if let Some(x) = a.lr_min {
        if !(x >= 0.0) {
            return Err(format!("--lr-min {x}: must be 0 or more"));
        }
    }
    a.muon = match opt_kind.as_str() {
        "adamw" => None,
        "muon" if muon_lr > 0.0 => Some(muon_lr),
        "muon" => return Err(format!("--muon-lr {muon_lr}: must be positive")),
        o => return Err(format!("--opt {o:?}: adamw or muon")),
    };
    if a.ctx_warmup > 0 {
        ctx_warmup_check(a.ctx_start, a.ctx)?;
    }
    if !(0.0..1.0).contains(&a.gf_alpha) || !(a.gf_lambda >= 0.0) {
        return Err(format!("--grokfast-alpha {} must be in [0, 1) and --grokfast-lambda {} 0 or more", a.gf_alpha, a.gf_lambda));
    }
    if !(a.top_weight >= 0.0 && a.top_weight.is_finite()) {
        return Err(format!("--top-weight {}: must be 0 or more", a.top_weight));
    }
    a.arch.top = a.top_weight > 0.0;
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
    if a.world > 1 {
        // K4: every rank reads its share of every batch, so the share must be whole rows
        if a.batch % a.world != 0 || a.rank >= a.world {
            return Err(format!("--batch {} must split evenly over --world {}, and --rank {} must be below it", a.batch, a.world, a.rank));
        }
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

    let vocab = if a.glance { VOCAB_GLANCE } else { VOCAB };
    let mut cfg = Config::with_kind(a.kind, vocab, a.d, a.layers, a.ctx);
    if a.glance {
        cfg = cfg.with_glance().map_err(|e| format!("--glance: {e}"))?;
    }
    if let Some(l) = a.engram {
        // v45: before the block options (--eng-gate needs the engram)
        cfg = cfg.with_engram(l).map_err(|e| format!("--engram: {e}"))?;
        println!("engram read in block {l}");
    }
    if !a.arch.is_default() {
        cfg = cfg.with_arch(a.arch).map_err(|e| format!("block options: {e}"))?;
        println!("v42 block options:{}", a.arch.signature());
        // v51 fused N: which training step they take (the hand-written WIN step, src/model/fwin.rs, or candle's autograd)
        let fused = ava_gpu::model::fwin::would_take_on(&cfg, &dev); // v51 cbfix: the device's own CB bound too
        println!("training step: {}", if fused { "the fused WIN step" } else { "candle's autograd" });
    }
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
    if train_set.vocab_size as usize != vocab {
        return Err(format!("train shards {} have vocab {}, this run needs {vocab} (--glance needs marked shards)", a.train.display(), train_set.vocab_size));
    }
    let mut sampler = Sampler::new(&train_set, a.ctx, a.data_seed).map_err(|e| format!("sampler: {e}"))?;

    // ---- model + optimiser
    if a.arch.v51.letters {
        ava_gpu::v51::set_letters_from_bpe(&a.bpe, vocab)?; // v51 R: every token's letters, from the BPE file
    }
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Model::new(cfg, vb).map_err(|e| format!("model: {e}"))?;
    let mut names: Vec<(String, candle_core::Var)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    names.sort_by(|x, y| x.0.cmp(&y.0));
    let (w_total, w_nonembed) = cfg.weight_counts();
    let counted: usize = names.iter().map(|(_, v)| v.elem_count()).sum();
    assert_eq!(counted, w_total, "weight count mismatch");
    let params: Vec<Param> = names.into_iter().map(|(name, var)| Param { decay: Model::decays_with(&name, a.decay_gates), name, var }).collect();
    let n_decay = params.iter().filter(|p| p.decay).count();
    let mut opt = AdamW::new(params, a.wd as f64).map_err(|e| format!("adamw: {e}"))?;
    opt.beta2 = a.beta2; // v42 (Q1); 0.95 unless --beta2
    if a.beta2 != 0.95 || a.decay_gates {
        println!("v42: AdamW beta2 {}; decay gates {}", a.beta2, a.decay_gates);
    }
    if let Some(m) = a.muon {
        let n = opt.enable_muon(MuonCfg::standard(m, a.lr as f64));
        println!("v42 --opt muon: {n} hidden matrices step by Muon (peak LR {m}), the rest by AdamW (peak LR {})", a.lr);
    }
    if let Some(cp) = a.crit {
        let n = opt.enable_critical_periods(cp);
        println!(
            "v42 H7 critical periods: {n} block tensors; layer l's LR falls after step {} + l·{} toward {}x (tau {}); the rest keep the full LR",
            cp.first, cp.gap, cp.floor, cp.tau
        );
    }
    // v42 (H6): the first masks now (dead weights to 0); a resume replaces them with the checkpoint's
    let mut rw = match a.rewire {
        None => None,
        Some(rc) => {
            let r = Rewiring::fresh(rc, &opt, a.seed).map_err(|e| format!("rewire: {e}"))?;
            let (live, n) = r.counts().map_err(|e| format!("rewire: {e}"))?;
            println!(
                "v42 H6 rewiring: {} MLP tensors, {live} of {n} weights live (density {}); every {} updates the {} weakest live ones are pruned and regrown",
                r.len(),
                rc.density,
                rc.every,
                rc.frac
            );
            Some(r)
        }
    };
    let (mut rw_events, mut rw_ms, mut rw_last) = (0u64, 0.0f64, None::<Event>);
    // v42 (I3): ReDo's own sampler (a resume moves it to the checkpoint's cursor); with reset, the init values
    let mut redo_st = match a.redo {
        None => None,
        Some(rc) => {
            if !cfg.kind.is_ava() {
                return Err("--redo: Ava's kinds only (T has no spiking layer)".into());
            }
            let s = Sampler::new(&train_set, a.ctx, derive_seed(a.data_seed, REDO_TAG, 0)).map_err(|e| format!("redo sampler: {e}"))?;
            let inits = if rc.reset { Some(ChannelInit::of(&model).map_err(|e| format!("redo: {e}"))?) } else { None };
            println!(
                "v42 I3 ReDo: every {} updates, units at or below {} of their layer's mean activity on {} batches are revived ({})",
                rc.every,
                rc.tau,
                rc.batches,
                if inits.as_ref().is_some_and(|i| !i.is_empty()) { "win/wout, w1/w2 and the per-channel LIF tensors back to init" } else { "win/wout and w1/w2" }
            );
            Some((rc, s, inits))
        }
    };
    let (mut rd_events, mut rd_ms) = (0u64, 0.0f64);
    // v42 (B6): the learning-progress mixture and its window reader (a resume restores the mixture's state)
    let mut lp_st = match a.lp_mix {
        None => None,
        Some(lc) => {
            let t = Instant::now();
            let (names, docs) = read_tags(&a.train.join("tags.tsv")).map_err(|e| format!("--lp-mix: {e}"))?;
            let ix = SourceIndex::build(&train_set, &names, &docs, a.ctx);
            let m = LpMix::new(&ix, lc, derive_seed(a.data_seed, LP_TAG, 0));
            let rd = WindowReader::open(&train_set).map_err(|e| format!("--lp-mix reader: {e}"))?;
            let probed = m.probes.iter().filter(|p| !p.is_empty()).count();
            println!(
                "v42 B6 lp-mix: {} classes ({probed} probed, {} windows each), {} windows; EPS {} every {} steps, cap {}x; index {:.2}s",
                ix.names.len(),
                lc.probe,
                ix.total(),
                lc.eps,
                lc.every,
                lc.cap,
                t.elapsed().as_secs_f64()
            );
            let drawn = vec![0u64; ix.names.len()];
            Some((m, rd, ix.names, drawn))
        }
    };
    let (mut lp_events, mut lp_ms) = (0u64, 0.0f64);
    let lr_floor = a.lr_min.unwrap_or(a.lr * a.min_lr_frac);
    let mut gf = (a.gf_lambda > 0.0).then(|| Grokfast::new(a.gf_alpha, a.gf_lambda));
    let v42 = v42_sig(&a);
    if !v42.is_empty() {
        println!("v42 settings:{v42}");
    }
    println!(
        "model {} d={} layers={} ctx={} vocab={vocab}{}: weights {w_total} total, {w_nonembed} non-embedding; {n_decay} decayed tensors of {}",
        a.kind.name(),
        a.d,
        a.layers,
        a.ctx,
        if a.glance { " glance" } else { "" },
        opt.params().len()
    );
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
    let sig = settings_line(&a, steps);
    let ckdir = a.out.as_ref().map(|o| o.join("ckpt"));
    let mut start = 0usize;
    if a.resume && !bench {
        let dir = ckdir.as_ref().ok_or("--resume needs --out")?;
        match fs::read_to_string(dir.join("state.txt")) {
            Err(_) => println!("--resume: no checkpoint in {}, starting at step 0", dir.display()),
            Ok(text) => {
                let get = |k: &str| text.lines().find_map(|l| l.strip_prefix(&format!("{k}="))).map(str::to_string).ok_or(format!("state.txt lacks {k}"));
                if get("args")? != sig {
                    return Err(format!("--resume: the checkpoint was made with other settings\n  checkpoint: {}\n  this run:   {sig}", get("args")?));
                }
                start = get("step")?.parse().map_err(|_| "bad step in state.txt")?;
                let t: u64 = get("t")?.parse().map_err(|_| "bad t in state.txt")?;
                varmap.load(dir.join(format!("model-{start}.safetensors"))).map_err(|e| format!("load checkpoint weights: {e}"))?;
                let st = candle_core::safetensors::load(dir.join(format!("optim-{start}.safetensors")), &dev).map_err(|e| format!("load checkpoint moments: {e}"))?;
                opt.load_state(&st, t).map_err(|e| format!("restore moments: {e}"))?;
                if let Some(g) = gf.as_mut() {
                    let n = g.load_state(&st, &dev).map_err(|e| format!("restore grokfast: {e}"))?;
                    println!("grokfast: {n} averages restored");
                }
                if let Some(r) = rw.as_mut() {
                    r.load_state(&st).map_err(|e| format!("restore rewiring masks: {e}"))?;
                    println!("rewiring: {} masks restored", r.len());
                }
                if let Some((_, s, _)) = redo_st.as_mut() {
                    *s = Sampler::resume(&train_set, Cursor::parse(&get("redo_cursor")?)?).map_err(|e| format!("resume redo sampler: {e}"))?;
                }
                if let Some((m, ..)) = lp_st.as_mut() {
                    m.restore(&get("lpmix")?).map_err(|e| format!("resume lp-mix: {e}"))?;
                }
                sampler = Sampler::resume(&train_set, Cursor::parse(&get("cursor")?)?).map_err(|e| format!("resume sampler: {e}"))?;
                println!("resumed from the checkpoint at step {start}");
            }
        }
    }
    model.v51_schedule(start, steps); // v51 Pr: the pruning schedule (nothing without --prune)
    // K4, many GPUs (src/dist.rs): every rank reads every batch and keeps its share of the rows; the gradients are
    // averaged over the ranks before the step, so every rank takes the same step. Only rank 0 writes files.
    // --world 1 (the default) runs the single-process step unchanged.
    let lead = a.rank == 0;
    let mut dist = ava_gpu::dist::Dist::connect(a.rank, a.world, &a.dist_addr)?;
    let vars: Vec<candle_core::Var> = opt.params().iter().map(|p| p.var.clone()).collect();
    if a.world > 1 {
        dist.broadcast_vars(&vars)?;
        // every rank must resume from the same checkpoint: round 1 sums (step, t, cursor hash), round 2 counts ranks
        // that differ from the mean; all stop together on a mismatch
        let h = sampler.cursor().to_line().bytes().fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32)) & 0xF_FFFF;
        let mine = [start as f32, opt.t as f32, h as f32];
        let mut sum = mine;
        dist.all_reduce_sum(&mut sum)?;
        let mut bad = [mine.iter().zip(&sum).any(|(m, s)| m * a.world as f32 != *s) as u8 as f32];
        dist.all_reduce_sum(&mut bad)?;
        if bad[0] > 0.0 {
            return Err(format!("rank {}: the ranks resumed from different checkpoints (this rank: step {start}); give every rank the same --out", a.rank));
        }
        println!("rank {} of {}: {} of the {} rows of every batch", a.rank, a.world, a.batch / a.world, a.batch);
    }
    let mut losses: Vec<f32> = Vec::with_capacity(steps_run);
    let mut log = None;
    if !bench && lead {
        let out = a.out.as_ref().unwrap();
        fs::create_dir_all(out).map_err(|e| format!("create {}: {e}", out.display()))?;
        if !a.arch.flags().is_empty() {
            // v42: the block options, so a tool that loads model.safetensors can build the same block
            fs::write(out.join("arch.txt"), a.arch.flags() + "\n").map_err(|e| format!("arch.txt: {e}"))?;
        }
        let header = "step\tloss\tlr\tgrad_norm\ttokens_per_sec\tstep_s\tspike_rates\tkwta_post";
        // On resume keep the log's lines before the checkpoint (and their losses); drop any after it.
        let mut kept = String::new();
        if start > 0 {
            let old = fs::read_to_string(out.join("log.tsv")).unwrap_or_default();
            for l in old.lines().skip(1) {
                let mut cols = l.split('\t');
                let (Some(s), Some(loss)) = (cols.next().and_then(|s| s.parse::<usize>().ok()), cols.next().and_then(|x| x.parse::<f32>().ok())) else { continue };
                if s < start {
                    losses.push(loss);
                    kept.push_str(l);
                    kept.push('\n');
                }
            }
        }
        let mut f = fs::File::create(out.join("log.tsv")).map_err(|e| format!("log.tsv: {e}"))?;
        write!(f, "{header}\n{kept}").map_err(|e| e.to_string())?;
        log = Some(f);
    }

    // ---- training loop
    let mut timed_s = 0.0f64;
    let mut timed_steps = 0u64;
    let t_train = Instant::now();
    // Speed round 13b (CUDA fast path): a step never waits for the GPU. The clipping factor is worked out on the
    // device; each step's loss and gradient norm stay there until the next shown step, checkpoint or the last
    // step, when the block is copied over at once (and the steps' wall times logged as the block's average). The
    // CPU prepares the next batch while the GPU still works. Same numbers as the waiting loop.
    // (K4: the all-reduce copies to the host every step anyway, so many ranks use the waiting loop)
    let fast_loop = a.world == 1 && dev.is_cuda() && ava_gpu::fast_path() && std::env::var("AVA_SYNC_LOOP").map_or(true, |v| v != "1");
    let mut pend: Vec<(usize, f32, Tensor, Tensor)> = Vec::new(); // (step, lr, loss, grad norm) on the device
    let mut t_block = Instant::now();
    let mut rested = 0.0f64; // v44 --gpu-share
    for step in start..steps_run {
        let t0 = Instant::now();
        let mut windows = match lp_st.as_mut() {
            None => sampler.batch(a.batch),
            Some((m, rd, _, drawn)) => {
                // v42 (B6): each row's class from the mixture, then that class's next window
                let mut w = Vec::with_capacity(a.batch);
                for (c, s) in m.draw(step as u64, a.batch) {
                    drawn[c] += 1;
                    w.push(rd.read(s, a.ctx + 1).map_err(|e| format!("step {step} lp-mix window: {e}"))?);
                }
                w
            }
        };
        // v42 (I2): during the context warm-up each window is cut into pieces of this step's length (the same
        // targets, the same tokens per step); off, c is --ctx and nothing changes
        let c = ctx_at(step as u64 * tok_per_step, a.ctx_warmup, a.ctx_start, a.ctx);
        let mut rows = if c < a.ctx {
            windows = split_windows(&windows, c);
            a.batch * (a.ctx / c)
        } else {
            a.batch
        };
        if a.world > 1 {
            // K4: this rank's rows (every world-th, from its rank); the whole batch's targets go to the parts whose
            // random draws depend on the whole batch (wave5's diffusion masks), so they draw what one process draws
            let mine: Vec<usize> = (a.rank..rows).step_by(a.world).collect();
            let full: Vec<Vec<u32>> = windows.iter().map(|w| w[1..].to_vec()).collect();
            windows = mine.iter().map(|&j| windows[j].clone()).collect();
            rows = mine.len();
            ava_gpu::wave5::set_row_split(Some(ava_gpu::wave5::RowSplit { full, rows: mine }));
        }
        let (ids, tg) = train_io(&windows);
        let ids_t = Tensor::from_vec(ids, (rows, c), &dev).map_err(|e| format!("ids: {e}"))?;
        let tg_t = Tensor::from_vec(tg, (rows, c), &dev).map_err(|e| format!("targets: {e}"))?;
        // v42 (A1): Schedule::Cosine with the old floor is cosine_lr exactly
        let lr = a.schedule.lr(step, a.warmup, steps, a.lr, lr_floor);
        // the activity means are logged (stdout) every log_every steps; on other steps they are not computed
        let show = step % a.log_every == 0 || step + 1 == steps_run;
        model.set_track_activity(show);
        // S on CUDA: the hand-written backward (speed round 3); otherwise candle's autograd
        let mut top_t: Option<Tensor> = None;
        let o2 = a.arch.mem.mem_reset > 0 || a.arch.mem.recall_loss > 0.0 || a.arch.mem.aux_any();
        let (loss_t, act, mut grads) = if o2 {
            // v42 Wave 2C (O2): --mem-reset / --recall-loss act inside the training loss (and the order loss, if on,
            // joins it there too; it is not logged on its own then); so do src/wave5.rs's auxiliary losses
            let (total, ce, act) = model.loss_aux(&ids_t, &tg_t, a.top_weight).map_err(|e| format!("step {step} forward: {e}"))?;
            let mut grads = total.backward().map_err(|e| format!("step {step} backward: {e}"))?;
            ava_gpu::arch::ckpt::merge(&mut grads).map_err(|e| format!("step {step} merge: {e}"))?;
            (ce.detach(), act, grads)
        } else if a.arch.top {
            // v42 (R2): next-token loss + weight · token-order loss; the next-token loss is the one logged
            let (total, ce, top, act) = model.loss_top(&ids_t, &tg_t, a.top_weight).map_err(|e| format!("step {step} forward: {e}"))?;
            let mut grads = total.backward().map_err(|e| format!("step {step} backward: {e}"))?;
            ava_gpu::arch::ckpt::merge(&mut grads).map_err(|e| format!("step {step} merge: {e}"))?;
            top_t = Some(top.detach());
            (ce.detach(), act, grads)
        } else {
            match model.loss_and_grads(&ids_t, &tg_t).map_err(|e| format!("step {step} fast step: {e}"))? {
                Some(r) => r,
                None => {
                    let (loss_t, act) = model.loss(&ids_t, &tg_t).map_err(|e| format!("step {step} forward: {e}"))?;
                    let mut grads = loss_t.backward().map_err(|e| format!("step {step} backward: {e}"))?;
                    ava_gpu::arch::ckpt::merge(&mut grads).map_err(|e| format!("step {step} merge: {e}"))?;
                    // detached: the loss waits in `pend` until the next log line, and must not keep this step's graph
                    (loss_t.detach(), act, grads)
                }
            }
        };
        // K4: the whole batch's gradient and loss: every rank's mean over its rows, averaged over the ranks (equal to
        // the mean over all rows when every row scores the same number of targets, as full windows do)
        let loss_t = if a.world > 1 {
            dist.all_reduce_grads(&vars, &mut grads)?;
            let inv = 1.0 / a.world as f64;
            for v in &vars {
                let scaled = match grads.get(v.as_tensor()) {
                    Some(g) => Some(g.affine(inv, 0.0).map_err(|e| format!("step {step} grad mean: {e}"))?),
                    None => None,
                };
                if let Some(s) = scaled {
                    grads.insert(v.as_tensor(), s);
                }
            }
            let mut l = [loss_t.to_scalar::<f32>().map_err(|e| format!("step {step} loss: {e}"))?];
            dist.all_reduce_sum(&mut l)?;
            Tensor::new(l[0] / a.world as f32, &dev).map_err(|e| format!("step {step} loss: {e}"))?
        } else {
            loss_t
        };
        if a.grad_shares && show && lead {
            // v42 Wave 2C (O2): each part's share of the squared gradient norm (is a new part getting any?)
            let params: Vec<(String, Tensor)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.as_tensor().clone())).collect();
            let sh = ava_gpu::memory::grad_shares(&params, &grads).map_err(|e| format!("step {step} grad shares: {e}"))?;
            println!("step {step} grad shares: {}", sh.iter().map(|(k, v)| format!("{k} {v:.4}")).collect::<Vec<_>>().join(", "));
        }
        if let Some(g) = gf.as_mut() {
            // v42 (I4): the slow average joins the gradient before the norm and clipping
            g.apply(opt.params(), &mut grads).map_err(|e| format!("step {step} grokfast: {e}"))?;
        }
        if std::env::var("AVA_GRAD_TOP").ok().and_then(|v| v.parse::<usize>().ok()) == Some(step) {
            // v42 debug: the weights with the largest gradients at this step
            let mut v: Vec<(f32, String)> = opt
                .params()
                .iter()
                .filter_map(|p| grads.get(p.var.as_tensor()).map(|g| (g.sqr().and_then(|s| s.sum_all()).and_then(|s| s.to_scalar::<f32>()).map_or(f32::NAN, f32::sqrt), p.name.clone())))
                .collect();
            v.sort_by(|x, y| y.0.total_cmp(&x.0));
            for (n, name) in v.iter().take(12) {
                eprintln!("grad top step {step}: {name} {n:.4}");
            }
        }
        let ckpt_now = !bench && lead && (step + 1) % a.ckpt_every == 0 && step + 1 != steps_run;
        if fast_loop {
            // Speed round 13b: no copy to the host inside a step (see `pend`)
            let gn = opt.grad_norm_t(&grads).map_err(|e| format!("step {step} grad norm: {e}"))?.ok_or_else(|| format!("step {step}: no gradients"))?;
            let sc = clip_scale_t(&gn, a.clip as f64).map_err(|e| format!("step {step} clip: {e}"))?;
            opt.step_t(&grads, lr as f64, &sc).map_err(|e| format!("step {step} adamw: {e}"))?;
            model.homeostasis_step().map_err(|e| format!("step {step} homeostasis: {e}"))?; // v42 (E8); nothing when off
            model.dale_decay(lr as f64 * a.wd as f64).map_err(|e| format!("step {step} dale decay: {e}"))?; // v50 (--dale); nothing when off
            drop(grads);
            if let Some(r) = rw.as_mut() {
                // v42 (H6): the masks, and an event when one is due
                if let Some(ev) = r.after_step(&opt).map_err(|e| format!("step {step} rewire: {e}"))? {
                    (rw_events, rw_ms, rw_last) = (rw_events + 1, rw_ms + ev.ms, Some(ev));
                }
            }
            if let Some((rc, s, inits)) = redo_st.as_mut().filter(|r| opt.t % r.0.every == 0) {
                // v42 (I3): a ReDo event
                rd_ms += redo_event(rc, s, inits.as_ref(), &varmap, &model, &opt, rw.as_ref(), &a, step, &dev)?;
                rd_events += 1;
            }
            if let Some((m, rd, names, _)) = lp_st.as_mut().filter(|l| (step + 1) % l.0.cfg.every == 0) {
                // v42 (B6): the probes, and the mixture moves
                m.set_frac((step + 1) as f64 / steps.max(1) as f64); // B6b: with END the tilt fades
                lp_ms += lp_event(m, rd, names, &model, a.ctx, step, &dev)?;
                lp_events += 1;
            }
            // detached: a norm from the candle fallback (not grad_norm_multi) would be linked to the step's graph. Same values.
            pend.push((step, lr, loss_t, gn.detach()));
            // also every 32 steps: on the autograd path with the one-launch norm and optimizer kernels, values waiting in
            // `pend` hold ~25 MB of GPU memory a step until read (runs/v46/leakfix: 6.2 GB after 120 steps at --log-every
            // 500, 3.2 GB flat at 40). One read per 32 steps keeps the speed; same values.
            if show || ckpt_now || step == start || step + 1 == steps_run || pend.len() >= 32 {
                // one wait for the GPU: this block's losses and norms come over together
                let ls: Vec<f32> = Tensor::stack(&pend.iter().map(|p| p.2.clone()).collect::<Vec<_>>(), 0).and_then(|t| t.to_vec1()).map_err(|e| format!("step {step} losses: {e}"))?;
                let gs: Vec<f64> = Tensor::stack(&pend.iter().map(|p| p.3.clone()).collect::<Vec<_>>(), 0).and_then(|t| t.to_vec1()).map_err(|e| format!("step {step} norms: {e}"))?;
                let block_s = t_block.elapsed().as_secs_f64();
                t_block = Instant::now();
                let wall = block_s / pend.len() as f64; // steps' wall time: the block's average
                let tps = tok_per_step as f64 / wall;
                if step > start || steps_run == start + 1 {
                    timed_s += block_s;
                    timed_steps += pend.len() as u64;
                }
                let (sr, kp) = (Activity::fmt(&act.spike_rate), Activity::fmt(&act.kwta_post));
                for (j, (s, lr_s, _, _)) in pend.iter().enumerate() {
                    let (loss, gnorm) = (ls[j], gs[j]);
                    if !loss.is_finite() {
                        return Err(format!("step {s}: loss is {loss}"));
                    }
                    losses.push(loss);
                    let (sr, kp) = if *s == step { (sr.as_str(), kp.as_str()) } else { ("", "") };
                    if let Some(f) = log.as_mut() {
                        writeln!(f, "{s}\t{loss:.6}\t{lr_s:.6e}\t{gnorm:.6}\t{tps:.1}\t{wall:.4}\t{sr}\t{kp}").map_err(|e| e.to_string())?;
                    }
                }
                if show {
                    let (loss, gnorm) = (ls[ls.len() - 1], gs[gs.len() - 1]);
                    println!("step {step}/{steps_run} loss {loss:.4} lr {lr:.3e} gnorm {gnorm:.3} {tps:.0} tok/s ({wall:.3}s) spikes [{sr}] kwta+ [{kp}]");
                }
                pend.clear();
            }
        } else {
            let gnorm = opt.grad_norm(&grads).map_err(|e| format!("step {step} grad norm: {e}"))?;
            let scale = clip_scale(gnorm, a.clip as f64);
            opt.step(&grads, lr as f64, scale).map_err(|e| format!("step {step} adamw: {e}"))?;
            if a.world > 1 && a.arch.homeostasis > 0.0 {
                // K4: E8 moves the thresholds by the rates of all ranks' rows (each rank measured only its own)
                let mut r = model.homeostasis_rates().map_err(|e| format!("step {step} rates: {e}"))?;
                dist.all_reduce_sum(&mut r)?;
                r.iter_mut().for_each(|x| *x /= a.world as f32);
                model.set_homeostasis_rates(&r).map_err(|e| format!("step {step} rates: {e}"))?;
            }
            model.homeostasis_step().map_err(|e| format!("step {step} homeostasis: {e}"))?; // v42 (E8); nothing when off
            model.dale_decay(lr as f64 * a.wd as f64).map_err(|e| format!("step {step} dale decay: {e}"))?; // v50 (--dale); nothing when off
            drop(grads);
            if let Some(r) = rw.as_mut() {
                // v42 (H6): the masks, and an event when one is due
                if let Some(ev) = r.after_step(&opt).map_err(|e| format!("step {step} rewire: {e}"))? {
                    (rw_events, rw_ms, rw_last) = (rw_events + 1, rw_ms + ev.ms, Some(ev));
                }
            }
            if let Some((rc, s, inits)) = redo_st.as_mut().filter(|r| opt.t % r.0.every == 0) {
                // v42 (I3): a ReDo event
                rd_ms += redo_event(rc, s, inits.as_ref(), &varmap, &model, &opt, rw.as_ref(), &a, step, &dev)?;
                rd_events += 1;
            }
            if let Some((m, rd, names, _)) = lp_st.as_mut().filter(|l| (step + 1) % l.0.cfg.every == 0) {
                // v42 (B6): the probes, and the mixture moves
                m.set_frac((step + 1) as f64 / steps.max(1) as f64); // B6b: with END the tilt fades
                lp_ms += lp_event(m, rd, names, &model, a.ctx, step, &dev)?;
                lp_events += 1;
            }
            let loss = loss_t.to_scalar::<f32>().map_err(|e| format!("step {step} loss: {e}"))?;
            drop(loss_t);
            if !loss.is_finite() {
                return Err(format!("step {step}: loss is {loss}"));
            }
            let wall = t0.elapsed().as_secs_f64();
            losses.push(loss);
            if step > start || steps_run == start + 1 {
                timed_s += wall;
                timed_steps += 1;
            }
            let tps = tok_per_step as f64 / wall;
            let (sr, kp) = if show || log.is_some() { (Activity::fmt(&act.spike_rate), Activity::fmt(&act.kwta_post)) } else { (String::new(), String::new()) };
            if let Some(f) = log.as_mut() {
                writeln!(f, "{step}\t{loss:.6}\t{lr:.6e}\t{gnorm:.6}\t{tps:.1}\t{wall:.4}\t{sr}\t{kp}").map_err(|e| e.to_string())?;
            }
            if show && lead {
                println!("step {step}/{steps_run} loss {loss:.4} lr {lr:.3e} gnorm {gnorm:.3} {tps:.0} tok/s ({wall:.3}s) spikes [{sr}] kwta+ [{kp}]");
            }
        }
        if show && (top_t.is_some() || c < a.ctx) {
            // v42: the warm-up's context and the token-order loss (R2), on shown steps only
            let top = match &top_t {
                Some(t) => format!(" token-order loss {:.4}", t.to_scalar::<f32>().map_err(|e| format!("step {step} order loss: {e}"))?),
                None => String::new(),
            };
            println!("  v42: ctx {c} x {rows} rows{top}");
        }
        if show && (a.crit.is_some() || rw.is_some()) {
            // v42 plasticity: the first and last layers' LR multipliers; the rewiring events so far
            let mut s = String::new();
            if let Some(cp) = &a.crit {
                s += &format!(" H7 lr x{:.4} (layer 0) x{:.4} (layer {})", cp.multiplier(0, step as f64), cp.multiplier(a.layers - 1, step as f64), a.layers - 1);
            }
            if rw.is_some() {
                s += &match &rw_last {
                    None => " H6 no event yet".to_string(),
                    Some(e) => format!(" H6 {rw_events} events, last after update {}: {} pruned, {} grown, {:.1} ms (mean {:.1} ms)", e.t, e.pruned, e.grown, e.ms, rw_ms / rw_events as f64),
                };
            }
            println!("  v42 plasticity:{s}");
        }
        if ckpt_now {
            if let Some(f) = log.as_mut() {
                f.flush().ok();
            }
            let p = a.out.as_ref().unwrap().join("model.safetensors");
            varmap.save(&p).map_err(|e| format!("save {}: {e}", p.display()))?;
            save_checkpoint(ckdir.as_ref().unwrap(), step + 1, &varmap, &opt, gf.as_ref(), rw.as_ref(), redo_st.as_ref().map(|r| r.1.cursor().to_line()), lp_st.as_ref().map(|l| l.0.state_line()), &sampler, &sig)?;
            // the save is not training time (as in the waiting loop, whose step timer stopped before it)
            t_block = Instant::now();
        }
        // v44 --gpu-share: wait for the GPU, then rest so it is busy about that share of the time (same numbers, only
        // slower). The rest is not training time: the block timer of the fast loop skips it (the waiting loop's step
        // timer stopped before it).
        if a.gpu_share < 1.0 {
            dev.synchronize().map_err(|e| format!("step {step} synchronize: {e}"))?;
            let rest = (t0.elapsed().as_secs_f64() * (1.0 / a.gpu_share - 1.0)).min(10.0);
            std::thread::sleep(std::time::Duration::from_secs_f64(rest));
            rested += rest;
            if fast_loop {
                t_block += std::time::Duration::from_secs_f64(rest);
            }
        }
    }
    if a.gpu_share < 1.0 {
        println!("--gpu-share {}: rested {rested:.1}s (in train_seconds, not in tok/s)", a.gpu_share);
    }
    if !bench && lead && steps_run < steps {
        // stopped early by --max-steps: keep a checkpoint so --resume can continue the schedule
        save_checkpoint(ckdir.as_ref().unwrap(), steps_run, &varmap, &opt, gf.as_ref(), rw.as_ref(), redo_st.as_ref().map(|r| r.1.cursor().to_line()), lp_st.as_ref().map(|l| l.0.state_line()), &sampler, &sig)?;
    }
    let train_s = t_train.elapsed().as_secs_f64();
    let tps_timed = if timed_steps > 0 { timed_steps as f64 * tok_per_step as f64 / timed_s } else { f64::NAN };
    let n50 = losses.len().min(50);
    let last50 = losses[losses.len() - n50..].iter().map(|&x| x as f64).sum::<f64>() / n50.max(1) as f64;
    println!("trained {steps_run} steps in {train_s:.1}s: {tps_timed:.0} tok/s (first step excluded), train_loss_last50 {last50:.4}");
    if rw_events > 0 {
        println!("v42 H6 rewiring: {rw_events} events, {:.1} ms each on average ({:.2}s in all)", rw_ms / rw_events as f64, rw_ms / 1e3);
    }
    if rd_events > 0 {
        println!("v42 I3 ReDo: {rd_events} events, {:.1} ms each on average ({:.2}s in all)", rd_ms / rd_events as f64, rd_ms / 1e3);
    }
    if let Some((m, _, names, drawn)) = lp_st.as_ref() {
        let total = drawn.iter().sum::<u64>().max(1) as f64;
        let share: Vec<String> = names.iter().zip(drawn).zip(&m.base).filter(|(_, b)| **b > 0.0).map(|((n, d), b)| format!("{n} {:.2}% ({:.2}%)", 100.0 * *d as f64 / total, 100.0 * b)).collect();
        println!("v42 B6 lp-mix: {lp_events} probe events, {:.1} ms each on average ({:.2}s in all); drawn (natural): {}", lp_ms / lp_events.max(1) as f64, lp_ms / 1e3, share.join(", "));
    }
    if bench {
        println!("bench: d={} layers={} ctx={} batch={} tokens_per_sec={tps_timed:.1} step_s={:.4}", a.d, a.layers, a.ctx, a.batch, timed_s / timed_steps.max(1) as f64);
        return Ok(());
    }
    if !lead {
        // K4: rank 0 saves and evaluates (every rank holds the same weights)
        println!("rank {} of {}: done", a.rank, a.world);
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
    let byte_len = byte_lengths_n(&bpe, vocab);
    let val_set = ShardSet::open(&a.val).map_err(|e| format!("open val shards {}: {e}", a.val.display()))?;
    if val_set.vocab_size as usize != vocab {
        return Err(format!("val shards {} have vocab {}, this run needs {vocab}", a.val.display(), val_set.vocab_size));
    }
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
        write!(
            f,
            "model\tdevice\td\tlayers\tctx\tbatch\ttokens\tsteps\tsteps_run\tlr\twarmup\tmin_lr_frac\twd\tclip\tseed\tdata_seed\teval_tokens\t\
             weights_total\tweights_nonembed\ttrain_loss_last50\tval_loss_nats_per_token\tval_bpb\tval_tokens_scored\tval_bytes\ttrain_seconds\teval_seconds\ttokens_per_sec"
        )
        .map_err(|e| e.to_string())?;
        writeln!(f, "{}{}", if a.glance { "\tglance" } else { "" }, if v42.is_empty() { "" } else { "\tv42" }).map_err(|e| e.to_string())?;
    }
    write!(
        f,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{:.2}\t{:.2}\t{:.1}",
        a.kind.name(),
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
    let v42_col = if v42.is_empty() { String::new() } else { format!("\t{}", v42.trim()) };
    writeln!(f, "{}{v42_col}", if a.glance { "\t1" } else { "" }).map_err(|e| e.to_string())?;
    if let (Some(d), true) = (&ckdir, steps_run == steps) {
        let _ = fs::remove_dir_all(d); // the whole schedule ran and the result is written
    }
    println!("wrote {} and {}", res_path.display(), model_path.display());
    Ok(())
}

/// The settings that fix a run's trajectory; --resume only continues a checkpoint with the same line.
fn settings_line(a: &Args, steps: usize) -> String {
    format!(
        "model={} d={} layers={} ctx={} batch={} tokens={} steps={steps} lr={} warmup={} min_lr_frac={} wd={} clip={} seed={} data_seed={}",
        a.kind.name(),
        a.d,
        a.layers,
        a.ctx,
        a.batch,
        a.tokens,
        a.lr,
        a.warmup,
        a.min_lr_frac,
        a.wd,
        a.clip,
        a.seed,
        a.data_seed
    ) + if a.glance { " glance=1" } else { "" }
        // v42: only when set, so older checkpoints still resume
        + &if a.beta2 != 0.95 { format!(" beta2={}", a.beta2) } else { String::new() }
        + if a.decay_gates { " decay_gates=1" } else { "" }
        // v42 Wave 2B: only the options that are set
        + &v42_sig(a)
}

/// Write a checkpoint into `dir`: model-STEP and optim-STEP first, then state.txt (renamed into place, so
/// it always names files that are complete), then delete older checkpoint files. With Grokfast (v42 I4) its averages
/// go into the optim file too (gf.<param>), and so do the rewiring masks (v42 H6, mask.<param>, u8). With ReDo (v42 I3)
/// state.txt also holds its sampler's cursor (redo_cursor); with --lp-mix (v42 B6) the mixture's state (lpmix).
#[allow(clippy::too_many_arguments)]
fn save_checkpoint(dir: &Path, step: usize, varmap: &VarMap, opt: &AdamW, gf: Option<&Grokfast>, rw: Option<&Rewiring>, redo_cursor: Option<String>, lp_line: Option<String>, sampler: &Sampler, sig: &str) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    varmap.save(dir.join(format!("model-{step}.safetensors"))).map_err(|e| format!("checkpoint weights: {e}"))?;
    let mut st = opt.state();
    if let Some(g) = gf {
        st.extend(g.state());
    }
    if let Some(r) = rw {
        st.extend(r.state().map_err(|e| format!("checkpoint masks: {e}"))?);
    }
    candle_core::safetensors::save(&st, dir.join(format!("optim-{step}.safetensors"))).map_err(|e| format!("checkpoint moments: {e}"))?;
    let tmp = dir.join("state.tmp");
    let redo = redo_cursor.map_or(String::new(), |c| format!("redo_cursor={c}\n")) + &lp_line.map_or(String::new(), |l| format!("lpmix={l}\n"));
    fs::write(&tmp, format!("step={step}\nt={}\ncursor={}\nargs={sig}\n{redo}", opt.t, sampler.cursor().to_line())).map_err(|e| format!("state.tmp: {e}"))?;
    fs::rename(&tmp, dir.join("state.txt")).map_err(|e| format!("state.txt: {e}"))?;
    let keep = [format!("model-{step}.safetensors"), format!("optim-{step}.safetensors")];
    for e in fs::read_dir(dir).map_err(|e| e.to_string())?.flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        if (n.starts_with("model-") || n.starts_with("optim-")) && !keep.contains(&n) {
            let _ = fs::remove_file(e.path());
        }
    }
    Ok(())
}

/// v42 (I3): the tag ReDo's seeds are derived with ("redo").
const REDO_TAG: u64 = 0x7265_646f;
/// v42 (B6): the tag the mixture's seeds are derived with ("lpmx").
const LP_TAG: u64 = 0x6c70_6d78;

/// v42 (B6): her loss on every probed class's held-out windows (no gradient), fed to the mixture; prints one line with
/// the classes whose share moved most. Returns its ms.
fn lp_event(m: &mut LpMix, rd: &mut WindowReader, names: &[String], model: &Model, ctx: usize, step: usize, dev: &Device) -> Result<f64, String> {
    dev.synchronize().map_err(|e| format!("step {step} lp-mix: {e}"))?; // time the event alone, not the queued step
    let t0 = Instant::now();
    model.set_track_activity(false);
    let mut losses = Vec::new();
    for (c, probe) in m.probes.iter().enumerate() {
        if probe.is_empty() {
            continue;
        }
        let w: Vec<Vec<u32>> = probe.iter().map(|&s| rd.read(s, ctx + 1)).collect::<std::io::Result<_>>().map_err(|e| format!("step {step} lp-mix probe: {e}"))?;
        let (ids, tg) = train_io(&w);
        let ids = Tensor::from_vec(ids, (w.len(), ctx), dev).map_err(|e| format!("step {step} lp-mix ids: {e}"))?;
        let tg = Tensor::from_vec(tg, (w.len(), ctx), dev).map_err(|e| format!("step {step} lp-mix targets: {e}"))?;
        let (l, _) = model.loss(&ids, &tg).map_err(|e| format!("step {step} lp-mix probe forward: {e}"))?;
        losses.push((c, l.to_scalar::<f32>().map_err(|e| format!("step {step} lp-mix loss: {e}"))? as f64));
    }
    m.observe(&losses);
    let mut moved: Vec<(f64, String)> = names.iter().enumerate().filter(|&(c, _)| m.base[c] > 0.0).map(|(c, n)| (m.p[c] / m.base[c], format!("{n} {:.2}x", m.p[c] / m.base[c]))).collect();
    moved.sort_by(|x, y| y.0.total_cmp(&x.0));
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let mean = losses.iter().map(|x| x.1).sum::<f64>() / losses.len().max(1) as f64;
    let top: Vec<&str> = moved.iter().take(3).chain(moved.iter().rev().take(2)).map(|x| x.1.as_str()).collect();
    println!("  v42 B6 lp-mix after step {}: probe loss mean {mean:.4} over {} classes; share vs natural: {} ({ms:.0} ms)", step + 1, losses.len(), top.join(", "));
    Ok(ms)
}

/// v42 (I3): one ReDo event after update opt.t: measure on rc.batches batches from ReDo's sampler, revive (inside the
/// H6 masks when --rewire is on), zero the moments where weights were written, and print one line. Returns its ms.
#[allow(clippy::too_many_arguments)]
fn redo_event(rc: &RedoCfg, sampler: &mut Sampler, inits: Option<&ChannelInit>, varmap: &VarMap, model: &Model, opt: &AdamW, rw: Option<&Rewiring>, a: &Args, step: usize, dev: &Device) -> Result<f64, String> {
    dev.synchronize().map_err(|e| format!("step {step} redo: {e}"))?; // time the event alone, not the queued step
    let t0 = Instant::now();
    let mut batches = Vec::with_capacity(rc.batches);
    for _ in 0..rc.batches {
        let (ids, _) = train_io(&sampler.batch(a.batch));
        batches.push(Tensor::from_vec(ids, (a.batch, a.ctx), dev).map_err(|e| format!("step {step} redo ids: {e}"))?);
    }
    let act = ava_gpu::upkeep::unit_activity(model, &batches).map_err(|e| format!("step {step} redo activity: {e}"))?;
    drop(batches);
    let t1 = Instant::now();
    let masks = rw.map(|r| r.host_masks()).transpose().map_err(|e| format!("step {step} redo masks: {e}"))?;
    let rev = ava_gpu::upkeep::revive(varmap, model, &act, rc.tau, derive_seed(a.seed, REDO_TAG, opt.t), masks.as_ref(), inits).map_err(|e| format!("step {step} redo: {e}"))?;
    let zeroed = ava_gpu::upkeep::zero_moments(opt, &rev).map_err(|e| format!("step {step} redo moments: {e}"))?;
    dev.synchronize().map_err(|e| format!("step {step} redo: {e}"))?;
    let (m_ms, r_ms) = ((t1 - t0).as_secs_f64() * 1e3, t1.elapsed().as_secs_f64() * 1e3);
    let r = &rev.report;
    println!(
        "  v42 I3 ReDo after update {}: dormant LIF {:?} of {}, MLP {:?} of {} per block; {zeroed} moment places zeroed; measure {m_ms:.0} ms + revive {r_ms:.0} ms",
        opt.t, r.lif, r.d, r.mlp, r.hidden
    );
    Ok(m_ms + r_ms)
}
