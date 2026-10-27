// TESTING / DESIGN ONLY — sft_gpu: teach a trained Ava to answer questions (v37).
//
// sft_gpu --init MODEL --arch D --d 384 --layers 8 --data FILE --epochs 2 --lr 3e-4 --out DIR
//         [--warmup 50] [--min-lr-frac 0.1] [--wd 0.1] [--clip 1.0] [--seed 1] [--batch-tokens 16384]
//         [--eval-tokens 1000000] [--eval-ctx 256] [--log-every 50] [--max-steps N] [--show N] [--bpe FILE] [--val DIR]
//
// * Starts from the weights in MODEL and trains on the AVASFT01 examples of FILE (qa_prep). A fresh AdamW, a
//   linear warm-up, then cosine decay to lr · min-lr-frac over epochs · batches steps.
// * Only the answer is scored: the targets that predict the answer tokens and the closing EOS have weight 1;
//   the question, the passage and the padding have weight 0 (Model::loss_and_grads_a_w; D and A on CUDA).
// * Batches: the examples are sorted by length and cut into batches of at most --batch-tokens tokens (rows ·
//   longest row), right-padded with PAD. Every model here is causal, so padding after an example cannot change it.
//   Each epoch visits the batches in a new seeded order.
// * Writes DIR/log.tsv (every step), DIR/model.safetensors and one line in DIR/result.tsv, with the validation
//   bits per byte on her ordinary reading (the first --eval-tokens targets of the val shards, read at --eval-ctx,
//   as train_gpu does), which shows how much of it the teaching cost her.
// * Replay (v38, --replay-rows R > 0): every step also re-reads R random windows of --replay-ctx tokens from
//   her ordinary reading (shard --replay-shard of --train), scored on every token as in pre-training. The
//   step's gradient is grad(answer loss) + --replay-weight · grad(reading loss). With R = 0 nothing changes:
//   the replay draws use their own random stream, so the batch order and the weights match v37 exactly.

use ava_data::bpe::{Bpe, PAD};
use ava_data::rng::{derive_seed, SplitMix64};
use ava_data::shards::ShardSet;
use ava_gpu::eval::{byte_lengths, evaluate};
use ava_gpu::model::{Config, Kind, Mm, Model, VOCAB};
use ava_gpu::optim::{clip_scale, cosine_lr, AdamW, Param};
use ava_gpu::qa::{read_sft, Example};
use candle_core::{DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: sft_gpu --init MODEL --arch D --d D --layers L --data FILE --epochs E --lr LR --out DIR\n\
    \x20             [--warmup 50] [--min-lr-frac 0.1] [--wd 0.1] [--clip 1.0] [--seed 1] [--batch-tokens 16384]\n\
    \x20             [--eval-tokens 1000000] [--eval-ctx 256] [--log-every 50] [--max-steps N] [--show N] [--bpe FILE] [--val DIR]\n\
    \x20             [--replay-rows 0] [--replay-weight 1.0] [--replay-ctx 256] [--replay-shard 0] [--train DIR]";

fn main() {
    if let Err(e) = run() {
        eprintln!("sft_gpu: {e}");
        std::process::exit(1);
    }
}

/// Batches of example indices: sorted by length, at most `budget` tokens (rows · longest row) each.
fn make_batches(ex: &[Example], budget: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..ex.len()).collect();
    order.sort_by_key(|&i| (ex[i].toks.len(), i));
    let mut out = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for i in order {
        let t = ex[i].toks.len() - 1;
        if !cur.is_empty() && (cur.len() + 1) * t > budget {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(i);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// (ids, targets, weights) for one batch, each [rows · T], T = longest example − 1.
fn batch_io(ex: &[Example], rows: &[usize]) -> (Vec<u32>, Vec<u32>, Vec<f32>, usize, usize) {
    let t = rows.iter().map(|&i| ex[i].toks.len() - 1).max().unwrap_or(1);
    let (mut ids, mut tg, mut w) = (Vec::with_capacity(rows.len() * t), Vec::with_capacity(rows.len() * t), Vec::with_capacity(rows.len() * t));
    let mut scored = 0usize;
    for &i in rows {
        let (s, np) = (&ex[i].toks, ex[i].np);
        for p in 0..t {
            if p + 1 < s.len() {
                ids.push(s[p]);
                tg.push(s[p + 1]);
                let keep = p + 1 >= np; // the target is an answer token or the closing EOS
                w.push(if keep { 1.0 } else { 0.0 });
                scored += keep as usize;
            } else {
                ids.push(PAD);
                tg.push(PAD);
                w.push(0.0);
            }
        }
    }
    (ids, tg, w, t, scored)
}

fn run() -> Result<(), String> {
    let (mut init, mut data, mut out): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None);
    let (mut d, mut layers, mut epochs, mut warmup, mut log_every) = (0usize, 0usize, 0usize, 50usize, 50usize);
    let (mut batch_tokens, mut eval_ctx, mut eval_tokens, mut show) = (16384usize, 256usize, 1_000_000u64, 0usize);
    let (mut lr, mut min_lr_frac, mut wd, mut clip) = (0f32, 0.1f32, 0.1f32, 1.0f32);
    let mut seed = 1u64;
    let mut max_steps: Option<usize> = None;
    let mut kind = Kind::D;
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let mut val = PathBuf::from(ava_gpu::DEFAULT_VAL);
    let mut train = PathBuf::from(ava_gpu::DEFAULT_TRAIN);
    let (mut replay_rows, mut replay_ctx, mut replay_shard, mut replay_weight) = (0usize, 256usize, 0usize, 1.0f32);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
        let fl = |v: &str| v.parse::<f32>().map_err(|_| format!("bad number {v:?}"));
        match key {
            "--init" => init = Some(PathBuf::from(v)),
            "--arch" => kind = Kind::parse(v)?,
            "--d" => d = num(v)?,
            "--layers" => layers = num(v)?,
            "--data" => data = Some(PathBuf::from(v)),
            "--epochs" => epochs = num(v)?,
            "--lr" => lr = fl(v)?,
            "--warmup" => warmup = num(v)?,
            "--min-lr-frac" => min_lr_frac = fl(v)?,
            "--wd" => wd = fl(v)?,
            "--clip" => clip = fl(v)?,
            "--seed" => seed = num(v)? as u64,
            "--batch-tokens" => batch_tokens = num(v)?,
            "--eval-tokens" => eval_tokens = num(v)? as u64,
            "--eval-ctx" => eval_ctx = num(v)?,
            "--log-every" => log_every = num(v)?.max(1),
            "--max-steps" => max_steps = Some(num(v)?),
            "--show" => show = num(v)?,
            "--bpe" => bpe_path = PathBuf::from(v),
            "--val" => val = PathBuf::from(v),
            "--train" => train = PathBuf::from(v),
            "--replay-rows" => replay_rows = num(v)?,
            "--replay-ctx" => replay_ctx = num(v)?.max(1),
            "--replay-shard" => replay_shard = num(v)?,
            "--replay-weight" => replay_weight = fl(v)?,
            "--out" => out = Some(PathBuf::from(v)),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let (init, data, out) = (init.ok_or("--init is required")?, data.ok_or("--data is required")?, out.ok_or("--out is required")?);
    if d == 0 || layers == 0 || epochs == 0 || lr <= 0.0 {
        return Err(format!("--d, --layers, --epochs and --lr are required\n{USAGE}"));
    }
    if !matches!(kind, Kind::D | Kind::A) {
        return Err("sft_gpu teaches D or A (the masked loss is in their hand-written step)".into());
    }
    let dev = ava_gpu::device(false).map_err(|e| format!("device: {e}"))?;
    if !dev.is_cuda() {
        return Err("sft_gpu needs CUDA (the masked loss is in the CUDA step)".into());
    }
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let ex = read_sft(&data)?;
    for e in ex.iter().take(show) {
        println!("--- example (source {}, prompt {} of {} tokens)\n{}|{}", e.src, e.np, e.toks.len(), bpe.decode(&e.toks[..e.np]), bpe.decode(&e.toks[e.np..]));
    }
    let batches = make_batches(&ex, batch_tokens);
    // her ordinary reading, for replay: one train shard in memory (100 MB for a 50M-token u16 shard)
    let replay_src = if replay_rows > 0 {
        let set = ShardSet::open(&train).map_err(|e| format!("open train shards {}: {e}", train.display()))?;
        if replay_shard >= set.len() {
            return Err(format!("--replay-shard {replay_shard}: the train set has {} shards", set.len()));
        }
        let toks = set.load_shard(replay_shard).map_err(|e| format!("load train shard {replay_shard}: {e}"))?;
        if toks.len() <= replay_ctx + 1 {
            return Err("the replay shard is shorter than one window".into());
        }
        println!("replay: {replay_rows} windows of {replay_ctx} tokens per step from train shard {replay_shard} ({} tokens), weight {replay_weight}", toks.len());
        Some(toks)
    } else {
        None
    };
    let mut replay_rng = SplitMix64::new(derive_seed(seed, 0x7265_706c_6179, 0));
    let steps = epochs * batches.len();
    let steps_run = max_steps.map_or(steps, |m| m.min(steps));

    let cfg = Config::with_kind(kind, VOCAB, d, layers, eval_ctx);
    let mut varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &dev)).map_err(|e| format!("model: {e}"))?;
    varmap.load(&init).map_err(|e| format!("load {}: {e}", init.display()))?;
    let mut names: Vec<(String, candle_core::Var)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    names.sort_by(|x, y| x.0.cmp(&y.0));
    let params: Vec<Param> = names.into_iter().map(|(name, var)| Param { decay: Model::decays(&name), name, var }).collect();
    let mut opt = AdamW::new(params, wd as f64).map_err(|e| format!("adamw: {e}"))?;
    let mm = Mm { bf16: ava_gpu::bf16_on() };
    let total_tok: usize = ex.iter().map(|e| e.toks.len()).sum();
    println!(
        "teaching {} d={d} layers={layers} from {}: {} examples ({total_tok} tokens), {} batches of <= {batch_tokens} tokens, {epochs} epochs = {steps} steps (running {steps_run}); lr {lr} warmup {warmup} min-lr-frac {min_lr_frac} wd {wd} clip {clip} seed {seed} bf16 {}",
        kind.name(),
        init.display(),
        ex.len(),
        batches.len(),
        mm.bf16
    );

    fs::create_dir_all(&out).map_err(|e| format!("create {}: {e}", out.display()))?;
    let mut log = fs::File::create(out.join("log.tsv")).map_err(|e| format!("log.tsv: {e}"))?;
    writeln!(log, "step\tepoch\tloss\tlr\tgrad_norm\trows\tT\tscored\ttokens_per_sec\treplay_loss").map_err(|e| e.to_string())?;
    let mut rng = SplitMix64::new(seed);
    let mut visit: Vec<usize> = Vec::new();
    let t_train = Instant::now();
    let (mut seen_tok, mut last_losses) = (0u64, Vec::new());
    for step in 0..steps_run {
        let epoch = step / batches.len();
        if step % batches.len() == 0 {
            visit = (0..batches.len()).collect();
            rng.shuffle(&mut visit);
        }
        let rows = &batches[visit[step % batches.len()]];
        let t0 = Instant::now();
        let (ids, tg, w, t, scored) = batch_io(&ex, rows);
        let b = rows.len();
        let ids_t = Tensor::from_vec(ids, (b, t), &dev).map_err(|e| e.to_string())?;
        let tg_t = Tensor::from_vec(tg, (b, t), &dev).map_err(|e| e.to_string())?;
        let w_t = Tensor::from_vec(w, b * t, &dev).map_err(|e| e.to_string())?;
        let lr_s = cosine_lr(step, warmup, steps, lr, lr * min_lr_frac);
        let (loss_t, _, mut grads) = model.loss_and_grads_a_w(&ids_t, &tg_t, Some(&w_t), &mm).map_err(|e| format!("step {step}: {e}"))?;
        let mut replay_loss = f32::NAN;
        let mut replay_tok = 0usize;
        if let Some(src) = &replay_src {
            let (mut rid, mut rtg) = (Vec::with_capacity(replay_rows * replay_ctx), Vec::with_capacity(replay_rows * replay_ctx));
            for _ in 0..replay_rows {
                let start = replay_rng.below((src.len() - replay_ctx - 1) as u64) as usize;
                let win = src.window(start, replay_ctx + 1);
                rid.extend_from_slice(&win[..replay_ctx]);
                rtg.extend_from_slice(&win[1..]);
            }
            let rid_t = Tensor::from_vec(rid, (replay_rows, replay_ctx), &dev).map_err(|e| e.to_string())?;
            let rtg_t = Tensor::from_vec(rtg, (replay_rows, replay_ctx), &dev).map_err(|e| e.to_string())?;
            let (rl, _, rg) = model.loss_and_grads_a_w(&rid_t, &rtg_t, None, &mm).map_err(|e| format!("step {step} replay: {e}"))?;
            for p in opt.params() {
                let key = p.var.as_tensor();
                let Some(gr) = rg.get(key) else { continue };
                let gr = gr.affine(replay_weight as f64, 0.0).map_err(|e| e.to_string())?;
                let merged = match grads.remove(key) {
                    Some(ga) => ga.add(&gr).map_err(|e| e.to_string())?,
                    None => gr,
                };
                grads.insert(key, merged);
            }
            replay_loss = rl.to_scalar::<f32>().map_err(|e| e.to_string())?;
            replay_tok = replay_rows * replay_ctx;
        }
        let gn = opt.grad_norm(&grads).map_err(|e| format!("step {step} grad norm: {e}"))?;
        opt.step(&grads, lr_s as f64, clip_scale(gn, clip as f64)).map_err(|e| format!("step {step} adamw: {e}"))?;
        let loss = loss_t.to_scalar::<f32>().map_err(|e| e.to_string())?;
        if !loss.is_finite() {
            return Err(format!("step {step}: loss is {loss}"));
        }
        let secs = t0.elapsed().as_secs_f64();
        seen_tok += (b * t + replay_tok) as u64;
        last_losses.push(loss);
        writeln!(log, "{step}\t{epoch}\t{loss:.6}\t{lr_s:.6e}\t{gn:.6}\t{b}\t{t}\t{scored}\t{:.1}\t{replay_loss:.6}", (b * t + replay_tok) as f64 / secs).map_err(|e| e.to_string())?;
        if step % log_every == 0 || step + 1 == steps_run {
            println!("step {step}/{steps_run} epoch {epoch} loss {loss:.4} lr {lr_s:.3e} gnorm {gn:.3} [{b} x {t}, {scored} scored] replay {replay_loss:.4} {:.0} tok/s", (b * t + replay_tok) as f64 / secs);
        }
    }
    let train_s = t_train.elapsed().as_secs_f64();
    let model_path = out.join("model.safetensors");
    varmap.save(&model_path).map_err(|e| format!("save {}: {e}", model_path.display()))?;
    let tail: Vec<f32> = last_losses.iter().rev().take(batches.len().min(200)).copied().collect();
    let tail_mean = tail.iter().sum::<f32>() / tail.len().max(1) as f32;
    println!("taught {steps_run} steps in {train_s:.1}s ({:.0} tokens/s incl. padding); answer loss, last {} steps: {tail_mean:.4}", seen_tok as f64 / train_s, tail.len());

    // her ordinary reading after the teaching: the same 1M validation tokens as train_gpu
    let t_eval = Instant::now();
    let val_set = ShardSet::open(&val).map_err(|e| format!("open val shards {}: {e}", val.display()))?;
    let (acc, _) = evaluate(&model, val_set.iter_sequential(eval_ctx), 16, eval_tokens, &byte_lengths(&bpe), &dev).map_err(|e| format!("eval: {e}"))?;
    let eval_s = t_eval.elapsed().as_secs_f64();
    println!("val: {:.4} nats/token, {:.4} bits/byte ({eval_s:.1}s)", acc.nats_per_token(), acc.bpb());
    let res = out.join("result.tsv");
    let fresh = !res.exists();
    let mut f = OpenOptions::new().create(true).append(true).open(&res).map_err(|e| e.to_string())?;
    if fresh {
        writeln!(f, "model\tinit\tdata\td\tlayers\tepochs\tsteps\tlr\twarmup\tmin_lr_frac\twd\tclip\tseed\tbatch_tokens\texamples\tanswer_loss_tail\tval_bpb\ttrain_seconds\ttokens_per_sec\treplay_rows\treplay_ctx\treplay_weight").map_err(|e| e.to_string())?;
    }
    writeln!(
        f,
        "{}\t{}\t{}\t{d}\t{layers}\t{epochs}\t{steps_run}\t{lr}\t{warmup}\t{min_lr_frac}\t{wd}\t{clip}\t{seed}\t{batch_tokens}\t{}\t{tail_mean:.6}\t{:.6}\t{train_s:.2}\t{:.1}\t{replay_rows}\t{replay_ctx}\t{replay_weight}",
        kind.name(),
        init.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
        data.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
        ex.len(),
        acc.bpb(),
        seen_tok as f64 / train_s
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
