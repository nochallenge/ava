// TESTING / DESIGN ONLY — grow_gpu: v40/v41, one model grown from v39's ava-num in three phases.
//
// grow_gpu --phase grow   --init FILE [--mag] [--engram L] --train DIR --tokens N [--qa FILE] --out DIR [...]
//                         [--recall-every K --recall-rows R] [--long-ctx 8192 --long-from 0.75]
// grow_gpu --phase sleep  --init FILE --facts FILE --train DIR --out DIR [--anchor A --fisher-batches 16] [...]
// grow_gpu --phase reward --init FILE (--sets sciq-train,arc-train,obqa-train | --reward-items TSV) --train DIR
//                         --out DIR [--norm total|byte] [--anchor A --fisher-batches 16] [...]
// common: [--arch D] [--lr 3e-4] [--warmup 200] [--min-lr-frac 0.1] [--wd 0.1] [--clip 1.0] [--seed 1]
//         [--val DIR] [--eval-ctx 256] [--eval-tokens 1000000] [--log-every 50] [--bpe FILE]
//         [--replay-rows 8] [--replay-ctx 256] [--replay-shard 0] [--replay-weight 1.0]
//
// v41 additions (all off by default, so a v40 command line computes exactly what it did):
// - RECALL PRACTICE (grow): every --recall-every steps the last --recall-rows windows are replaced by recall rows
//   (ava_data::recall: a fact placed early in her own reading, asked about at the end), so the engram has
//   something only a look back can answer.
// - LONG PHASE (grow): from --long-from of the steps on, windows are --long-ctx tokens (same tokens per step,
//   fewer rows), in the same shard order from the first shard the short windows never reached (its cursor is
//   checkpointed as cursor_long).
// - ANCHOR (sleep, reward): synaptic consolidation; see `Anchor`.
// - NORM (reward): --norm byte scores each choice by NLL per byte, as mc_eval --norm byte grades.
// - REWARD ITEMS (reward): questions from a TSV (mc::read_items) instead of --sets; a file named *test* or under a
//   sealed/ folder is refused.
//
// GROW. The model is v39's (read from --init: vocab, width, layers, glance) plus --mag (the magnitude code) and
// --engram L (the engram read in block L). Every tensor of --init is copied (ckpt::init_from); the new read-outs
// (mag_w, the engram's Wo) start at 0, so step 0 computes exactly v39's model. She reads --batch windows of --ctx
// tokens per step from --train (books + usual text + practice v2), --tokens in all. Every --qa-every steps (set
// so the answers are seen --qa-epochs times) the step also takes one batch of answers from --qa (AVASFT01; length
// marks inserted on prompt and answer; only the answer scored), added with weight --qa-weight. A fresh AdamW,
// warm-up, cosine decay. Checkpoints every --ckpt-every steps; --resume continues from the last one with the same
// windows and answer batches.
//
// SLEEP. The day: --facts (episodes separated by blank lines, or one per line; made-up facts that no data can
// hold). Sleep: with the engram switched off (what is kept must go into her own weights), --sleep-steps steps
// replay episodes (--sleep-rows per step, only the episode scored) mixed with her usual reading (replay). Surprise
// priority: at the start of each of --cycles cycles every episode's loss per token is measured (engram off) and
// episodes are drawn with probability proportional to loss^--surprise-alpha (0 = uniform): the brain replays
// what surprised it more often.
//
// REWARD. Questions with a checkable answer: the train splits named by --sets (a name ending in "test" is
// refused). Per step --items questions are scored as qa_eval scores them. She answers only when her margin >=
// --tau; otherwise she says "I don't know" (reward 0, no update). Her answer is drawn from softmax(-NLL / --temp)
// over the choices. Reward +1 right, -1 wrong. Advantage = reward - running mean (--baseline-decay). Update:
// REINFORCE on the choice policy: each choice j of an answered question is weighted
// advantage · (1[j drawn] − π_j) / (temp · answered) on its tokens (signed raw weights, Model::loss_and_grads_a_wr),
// the exact gradient of −advantage · log π(drawn), plus reading replay. The right answer is never shown, only the outcome.
//
// Every phase ends with DIR/model.safetensors, the validation BPB at --eval-ctx on --val (every part switched
// on), and one line in DIR/result.tsv, written last: its presence marks the phase done, so a chain skips it.

use ava_data::bpe::{Bpe, EOS, PAD};
use ava_data::numbers::insert_marks;
use ava_data::rng::{derive_seed, SplitMix64};
use ava_data::shards::{Cursor, Sampler, ShardSet, ShardTokens};
use ava_gpu::eval::{byte_lengths_n, evaluate, train_io};
use ava_gpu::anchor::{Anchor, Importance};
use ava_gpu::mc;
use ava_gpu::model::{Kind, Mm, Model};
use ava_gpu::optim::{clip_scale, cosine_lr, AdamW, Param};
use ava_gpu::qa::{read_sft, Example};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: grow_gpu --phase grow|sleep|reward --init FILE --out DIR [options; see the top of grow_gpu.rs]";

fn main() {
    if let Err(e) = run() {
        eprintln!("grow_gpu: {e}");
        std::process::exit(1);
    }
}

#[derive(Debug, Clone)]
struct Args {
    phase: String,
    init: PathBuf,
    out: PathBuf,
    kind: Kind,
    lr: f32,
    warmup: usize,
    min_lr_frac: f32,
    wd: f32,
    clip: f32,
    seed: u64,
    val: Option<PathBuf>,
    eval_ctx: usize,
    eval_tokens: u64,
    log_every: usize,
    bpe: PathBuf,
    train: Option<PathBuf>,
    replay_rows: usize,
    replay_ctx: usize,
    replay_shard: usize,
    replay_weight: f32,
    max_steps: Option<usize>,
    // grow
    mag: bool,
    engram: Option<usize>,
    ctx: usize,
    batch: usize,
    tokens: usize,
    data_seed: u64,
    qa: Option<PathBuf>,
    qa_batch_tokens: usize,
    qa_epochs: usize,
    qa_weight: f32,
    ckpt_every: usize,
    resume: bool,
    rank: usize,
    world: usize,
    dist_addr: String,
    // sleep
    facts: Option<PathBuf>,
    sleep_steps: usize,
    sleep_rows: usize,
    cycles: usize,
    surprise_alpha: f64,
    // reward
    sets: String,
    passage: bool,
    reward_steps: usize,
    items: usize,
    tau: f64,
    temp: f64,
    baseline_decay: f64,
    score_batch: usize,
    // v41
    recall_every: usize,
    recall_rows: usize,
    long_ctx: usize,
    long_from: f64,
    anchor: f64,
    fisher_batches: usize,
    norm: mc::Norm,
    reward_items: Option<PathBuf>,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        phase: String::new(),
        init: PathBuf::new(),
        out: PathBuf::new(),
        kind: Kind::D,
        lr: 3e-4,
        warmup: 200,
        min_lr_frac: 0.1,
        wd: 0.1,
        clip: 1.0,
        seed: 1,
        val: None,
        eval_ctx: 256,
        eval_tokens: 1_000_000,
        log_every: 50,
        bpe: PathBuf::from(ava_gpu::DEFAULT_BPE),
        train: None,
        replay_rows: 8,
        replay_ctx: 256,
        replay_shard: 0,
        replay_weight: 1.0,
        max_steps: None,
        mag: false,
        engram: None,
        ctx: 2048,
        batch: 4,
        tokens: 0,
        data_seed: 1,
        qa: None,
        qa_batch_tokens: 4096,
        qa_epochs: 2,
        qa_weight: 1.0,
        ckpt_every: 500,
        resume: false,
        rank: 0,
        world: 1,
        dist_addr: "127.0.0.1:29540".into(),
        facts: None,
        sleep_steps: 300,
        sleep_rows: 16,
        cycles: 4,
        surprise_alpha: 1.0,
        sets: "sciq-train,arc-train,obqa-train".into(),
        passage: false,
        reward_steps: 300,
        items: 16,
        tau: 0.0,
        temp: 1.0,
        baseline_decay: 0.9,
        score_batch: 32768,
        recall_every: 0,
        recall_rows: 1,
        long_ctx: 0,
        long_from: 0.75,
        anchor: 0.0,
        fisher_batches: 16,
        norm: mc::Norm::Total,
        reward_items: None,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        match key {
            "-h" | "--help" => return Err(USAGE.into()),
            "--mag" => {
                a.mag = true;
                i += 1;
                continue;
            }
            "--resume" => {
                a.resume = true;
                i += 1;
                continue;
            }
            "--passage" => {
                a.passage = true;
                i += 1;
                continue;
            }
            _ => {}
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
        let fl = |v: &str| v.parse::<f64>().map_err(|_| format!("bad number {v:?} for {key}"));
        match key {
            "--phase" => a.phase = v.into(),
            "--init" => a.init = v.into(),
            "--out" => a.out = v.into(),
            "--arch" => a.kind = Kind::parse(v)?,
            "--lr" => a.lr = fl(v)? as f32,
            "--warmup" => a.warmup = num(v)?,
            "--min-lr-frac" => a.min_lr_frac = fl(v)? as f32,
            "--wd" => a.wd = fl(v)? as f32,
            "--clip" => a.clip = fl(v)? as f32,
            "--seed" => a.seed = num(v)? as u64,
            "--val" => a.val = Some(v.into()),
            "--eval-ctx" => a.eval_ctx = num(v)?.max(1),
            "--eval-tokens" => a.eval_tokens = num(v)? as u64,
            "--log-every" => a.log_every = num(v)?.max(1),
            "--bpe" => a.bpe = v.into(),
            "--train" => a.train = Some(v.into()),
            "--replay-rows" => a.replay_rows = num(v)?,
            "--replay-ctx" => a.replay_ctx = num(v)?.max(1),
            "--replay-shard" => a.replay_shard = num(v)?,
            "--replay-weight" => a.replay_weight = fl(v)? as f32,
            "--max-steps" => a.max_steps = Some(num(v)?),
            "--engram" => a.engram = Some(num(v)?),
            "--ctx" => a.ctx = num(v)?.max(1),
            "--batch" => a.batch = num(v)?.max(1),
            "--tokens" => a.tokens = num(v)?,
            "--data-seed" => a.data_seed = num(v)? as u64,
            "--qa" => a.qa = Some(v.into()),
            "--qa-batch-tokens" => a.qa_batch_tokens = num(v)?.max(1),
            "--qa-epochs" => a.qa_epochs = num(v)?,
            "--qa-weight" => a.qa_weight = fl(v)? as f32,
            "--ckpt-every" => a.ckpt_every = num(v)?,
            "--rank" => a.rank = num(v)?,
            "--world" => a.world = num(v)?.max(1),
            "--dist-addr" => a.dist_addr = v.into(),
            "--facts" => a.facts = Some(v.into()),
            "--sleep-steps" => a.sleep_steps = num(v)?,
            "--sleep-rows" => a.sleep_rows = num(v)?.max(1),
            "--cycles" => a.cycles = num(v)?.max(1),
            "--surprise-alpha" => a.surprise_alpha = fl(v)?,
            "--sets" => a.sets = v.into(),
            "--reward-steps" => a.reward_steps = num(v)?,
            "--items" => a.items = num(v)?.max(1),
            "--tau" => a.tau = fl(v)?,
            "--temp" => a.temp = fl(v)?.max(1e-6),
            "--baseline-decay" => a.baseline_decay = fl(v)?,
            "--score-batch" => a.score_batch = num(v)?.max(1),
            "--recall-every" => a.recall_every = num(v)?,
            "--recall-rows" => a.recall_rows = num(v)?.max(1),
            "--long-ctx" => a.long_ctx = num(v)?,
            "--long-from" => a.long_from = fl(v)?.clamp(0.0, 1.0),
            "--anchor" => a.anchor = fl(v)?.max(0.0),
            "--fisher-batches" => a.fisher_batches = num(v)?.max(1),
            "--norm" => a.norm = mc::Norm::parse(v)?,
            "--reward-items" => a.reward_items = Some(v.into()),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    if a.init.as_os_str().is_empty() || a.out.as_os_str().is_empty() {
        return Err(format!("--init and --out are required\n{USAGE}"));
    }
    if !matches!(a.kind, Kind::D | Kind::A) {
        return Err("grow_gpu trains D or A (their hand-written step)".into());
    }
    Ok(a)
}

fn run() -> Result<(), String> {
    let a = parse()?;
    let dev = ava_gpu::device(false).map_err(|e| format!("device: {e}"))?;
    if !dev.is_cuda() {
        return Err("grow_gpu needs CUDA (the weighted and signed losses are in the CUDA step)".into());
    }
    if a.out.join("result.tsv").exists() {
        println!("{}: result.tsv exists, phase {} is done (skipped)", a.out.display(), a.phase);
        return Ok(());
    }
    let bpe = Bpe::load(&a.bpe).map_err(|e| format!("bpe {}: {e}", a.bpe.display()))?;
    fs::create_dir_all(&a.out).map_err(|e| format!("create {}: {e}", a.out.display()))?;
    match a.phase.as_str() {
        "grow" => phase_grow(&a, &dev, &bpe),
        "sleep" => phase_sleep(&a, &dev, &bpe),
        "reward" => phase_reward(&a, &dev, &bpe),
        p => Err(format!("unknown --phase {p:?} (grow, sleep, reward)\n{USAGE}")),
    }
}

// ------------------------------------------------------------------------------------------------ shared parts

fn params_of(varmap: &VarMap) -> Vec<Param> {
    let mut names: Vec<(String, candle_core::Var)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    names.sort_by(|x, y| x.0.cmp(&y.0));
    names.into_iter().map(|(name, var)| Param { decay: Model::decays(&name), name, var }).collect()
}

/// grads += w · other, parameter by parameter.
fn merge(opt: &AdamW, grads: &mut GradStore, other: &GradStore, w: f32) -> Result<(), String> {
    for p in opt.params() {
        let key = p.var.as_tensor();
        let Some(g) = other.get(key) else { continue };
        let g = if w == 1.0 { g.clone() } else { g.affine(w as f64, 0.0).map_err(|e| e.to_string())? };
        let merged = match grads.remove(key) {
            Some(x) => x.add(&g).map_err(|e| e.to_string())?,
            None => g,
        };
        grads.insert(key, merged);
    }
    Ok(())
}

/// Her usual reading for replay: windows of `ctx` tokens drawn at random from one train shard.
struct Replay {
    toks: ShardTokens,
    ctx: usize,
    rows: usize,
    rng: SplitMix64,
}

impl Replay {
    /// `stream`: a different seed per phase (1 sleep, 2 reward), so the phases replay different windows.
    fn open(a: &Args, stream: u64) -> Result<Option<Replay>, String> {
        Replay::open_rows(a, stream, a.replay_rows)
    }

    fn open_rows(a: &Args, stream: u64, rows: usize) -> Result<Option<Replay>, String> {
        if rows == 0 {
            return Ok(None);
        }
        let dir = a.train.as_ref().ok_or("replay needs --train")?;
        let set = ShardSet::open(dir).map_err(|e| format!("open train shards {}: {e}", dir.display()))?;
        if a.replay_shard >= set.len() {
            return Err(format!("--replay-shard {}: the train set has {} shards", a.replay_shard, set.len()));
        }
        let toks = set.load_shard(a.replay_shard).map_err(|e| format!("load train shard {}: {e}", a.replay_shard))?;
        if toks.len() <= a.replay_ctx + 1 {
            return Err("the replay shard is shorter than one window".into());
        }
        Ok(Some(Replay { toks, ctx: a.replay_ctx, rows, rng: SplitMix64::new(derive_seed(a.seed, 0x7265_706c_6179, stream)) }))
    }

    /// The reading loss and its gradient on fresh random windows.
    fn grads(&mut self, model: &Model, mm: &Mm, dev: &Device) -> Result<(f32, GradStore, usize), String> {
        let (mut rid, mut rtg) = (Vec::with_capacity(self.rows * self.ctx), Vec::with_capacity(self.rows * self.ctx));
        for _ in 0..self.rows {
            let start = self.rng.below((self.toks.len() - self.ctx - 1) as u64) as usize;
            let win = self.toks.window(start, self.ctx + 1);
            rid.extend_from_slice(&win[..self.ctx]);
            rtg.extend_from_slice(&win[1..]);
        }
        let rid = Tensor::from_vec(rid, (self.rows, self.ctx), dev).map_err(|e| e.to_string())?;
        let rtg = Tensor::from_vec(rtg, (self.rows, self.ctx), dev).map_err(|e| e.to_string())?;
        let (l, _, g) = model.loss_and_grads_a_w(&rid, &rtg, None, mm).map_err(|e| format!("replay: {e}"))?;
        Ok((l.to_scalar::<f32>().map_err(|e| e.to_string())?, g, self.rows * self.ctx))
    }
}

/// v41 synaptic consolidation (ava_gpu::anchor): before sleep or reward, measure each weight's importance to her usual
/// reading on --fisher-batches fresh replay batches (`stream`: a replay stream of its own, 10 + the phase's, so these
/// are not the replay batches); after every optimizer step the anchor pulls weights back by that importance.
/// `--anchor 0` switches it off.
fn measure_anchor(a: &Args, model: &Model, opt: &AdamW, mm: &Mm, dev: &Device, stream: u64) -> Result<Option<Anchor>, String> {
    if a.anchor <= 0.0 {
        return Ok(None);
    }
    if a.fisher_batches == 0 {
        return Err("--anchor needs --fisher-batches > 0".into());
    }
    let mut r = Replay::open_rows(a, stream, a.replay_rows.max(4))?.ok_or("--anchor needs replay data")?;
    let t0 = Instant::now();
    let mut imp = Importance::new(opt.params());
    for _ in 0..a.fisher_batches {
        let (_, g, _) = r.grads(model, mm, dev)?;
        imp.add(&g).map_err(|e| format!("anchor importance: {e}"))?;
    }
    let an = imp.anchor(a.anchor).map_err(|e| format!("anchor: {e}"))?.ok_or("anchor: her reading gave no gradient")?;
    println!("anchor {}: importance from {} replay batches over {} weights, {:.1}s", a.anchor, a.fisher_batches, an.weights, t0.elapsed().as_secs_f64());
    Ok(Some(an))
}

/// Rows of tokens, right-padded with PAD, as (ids, targets, weights, T): a target is scored (weight 1) when it
/// is real and its index is >= from[r].
fn padded(rows: &[&[u32]], from: &[usize]) -> (Vec<u32>, Vec<u32>, Vec<f32>, usize) {
    let t = rows.iter().map(|s| s.len()).max().unwrap_or(2).saturating_sub(1).max(1);
    let (mut ids, mut tg, mut w) = (Vec::with_capacity(rows.len() * t), Vec::with_capacity(rows.len() * t), Vec::with_capacity(rows.len() * t));
    for (r, s) in rows.iter().enumerate() {
        for p in 0..t {
            let real = p + 1 < s.len();
            ids.push(if real { s[p] } else { PAD });
            tg.push(if real { s[p + 1] } else { PAD });
            w.push(if real && p + 1 >= from[r] { 1.0 } else { 0.0 });
        }
    }
    (ids, tg, w, t)
}

/// A uniform draw in [0, 1).
fn unit(rng: &mut SplitMix64) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// An index drawn with probability proportional to `p` (all >= 0, not all 0).
fn draw(p: &[f64], total: f64, rng: &mut SplitMix64) -> usize {
    let mut x = unit(rng) * total;
    for (i, &v) in p.iter().enumerate() {
        if x < v {
            return i;
        }
        x -= v;
    }
    p.len() - 1
}

/// Save the model, measure validation BPB (every part on), and write result.tsv last.
fn finish(model: &Model, varmap: &VarMap, a: &Args, bpe: &Bpe, dev: &Device, cols: Vec<(&str, String)>) -> Result<(), String> {
    let model_path = a.out.join("model.safetensors");
    varmap.save(&model_path).map_err(|e| format!("save {}: {e}", model_path.display()))?;
    model.set_lesion(false, false, false);
    let mut cols = cols;
    if let Some(val) = &a.val {
        let t0 = Instant::now();
        model.set_inference(true);
        let set = ShardSet::open(val).map_err(|e| format!("open val shards {}: {e}", val.display()))?;
        let byte_len = byte_lengths_n(bpe, set.vocab_size as usize);
        let (acc, _) = evaluate(model, set.iter_sequential(a.eval_ctx), 16, a.eval_tokens, &byte_len, dev).map_err(|e| format!("eval: {e}"))?;
        model.set_inference(false);
        println!("val: {:.4} nats/token, {:.4} bits/byte ({:.1}s)", acc.nats_per_token(), acc.bpb(), t0.elapsed().as_secs_f64());
        cols.push(("val_bpb", format!("{:.6}", acc.bpb())));
    }
    let res = a.out.join("result.tsv");
    let mut f = OpenOptions::new().create(true).write(true).truncate(true).open(&res).map_err(|e| e.to_string())?;
    let head: Vec<&str> = cols.iter().map(|c| c.0).collect();
    let vals: Vec<String> = cols.iter().map(|c| c.1.clone()).collect();
    writeln!(f, "{}\n{}", head.join("\t"), vals.join("\t")).map_err(|e| e.to_string())?;
    Ok(())
}

// ------------------------------------------------------------------------------------------------ grow

/// Teaching examples with length marks on the prompt and on the answer (separately, as mc::item_seqs does).
fn marked(ex: Vec<Example>, glance: bool) -> Vec<Example> {
    if !glance {
        return ex;
    }
    ex.into_iter()
        .map(|e| {
            let mut toks = insert_marks(&e.toks[..e.np]);
            let np = toks.len();
            toks.extend(insert_marks(&e.toks[e.np..]));
            Example { src: e.src, np, toks }
        })
        .collect()
}

/// Batches of example indices: sorted by length, at most `budget` tokens (rows · longest row) each (sft_gpu's rule).
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

fn grow_signature(a: &Args, steps: usize) -> String {
    format!(
        "init={} mag={} engram={:?} train={} ctx={} batch={} tokens={} steps={steps} lr={} warmup={} min_lr_frac={} wd={} clip={} seed={} data_seed={} qa={:?} qa_batch_tokens={} qa_epochs={} qa_weight={} recall_every={} recall_rows={} long_ctx={} long_from={}",
        a.init.display(),
        a.mag,
        a.engram,
        a.train.as_ref().map(|p| p.display().to_string()).unwrap_or_default(),
        a.ctx,
        a.batch,
        a.tokens,
        a.lr,
        a.warmup,
        a.min_lr_frac,
        a.wd,
        a.clip,
        a.seed,
        a.data_seed,
        a.qa.as_ref().map(|p| p.display().to_string()),
        a.qa_batch_tokens,
        a.qa_epochs,
        a.qa_weight,
        a.recall_every,
        a.recall_rows,
        a.long_ctx,
        a.long_from
    )
}

/// `secs` and `tok`: training seconds and tokens up to this step (all sessions), so a resumed run reports the whole run.
#[allow(clippy::too_many_arguments)]
/// `long`: the long phase's sampler (v41), whose cursor is saved too.
/// The long phase's sampler: windows of `long_ctx` tokens in the short sampler's own shard order (same seed and
/// epoch), starting at the shard after the one the short windows are in, so it reads text they never reached (unless
/// the long phase outruns the remaining shards and wraps into the next epoch).
fn long_sampler_after(set: &ShardSet, short: &Sampler, long_ctx: usize) -> Result<Sampler, String> {
    let c = short.cursor();
    let (epoch, shard_pos) = if c.shard_pos + 1 >= set.len() { (c.epoch + 1, 0) } else { (c.epoch, c.shard_pos + 1) };
    Sampler::resume(set, Cursor { seed: c.seed, ctx_len: long_ctx, epoch, shard_pos, slot_pos: 0 }).map_err(|e| format!("long sampler: {e}"))
}

fn save_checkpoint(dir: &Path, step: usize, varmap: &VarMap, opt: &AdamW, sampler: &Sampler, long: Option<&Sampler>, sig: &str, secs: f64, tok: u64) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    varmap.save(dir.join(format!("model-{step}.safetensors"))).map_err(|e| format!("checkpoint weights: {e}"))?;
    candle_core::safetensors::save(&opt.state(), dir.join(format!("optim-{step}.safetensors"))).map_err(|e| format!("checkpoint moments: {e}"))?;
    let tmp = dir.join("state.tmp");
    let long_line = long.map(|s| format!("cursor_long={}\n", s.cursor().to_line())).unwrap_or_default();
    fs::write(&tmp, format!("step={step}\nt={}\ncursor={}\n{long_line}secs={secs:.3}\ntok={tok}\nargs={sig}\n", opt.t, sampler.cursor().to_line())).map_err(|e| format!("state.tmp: {e}"))?;
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

fn phase_grow(a: &Args, dev: &Device, bpe: &Bpe) -> Result<(), String> {
    let train = a.train.as_ref().ok_or("--phase grow needs --train")?;
    if a.tokens == 0 {
        return Err("--phase grow needs --tokens".into());
    }
    // v39's shape from the file, plus the new parts
    let t0 = ava_gpu::ckpt::read(&a.init)?;
    let mut cfg = ava_gpu::ckpt::detect(&t0, a.kind, a.ctx)?;
    drop(t0);
    if a.mag {
        cfg = cfg.with_mag()?;
    }
    if let Some(l) = a.engram {
        cfg = cfg.with_engram(l)?;
    }
    let mut varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, dev)).map_err(|e| format!("model: {e}"))?;
    let (copied, kept) = ava_gpu::ckpt::init_from(&varmap, &a.init)?;
    println!("init: {} tensors from {}; new parts (their init): {kept:?}", copied.len(), a.init.display());
    model.set_track_activity(false);
    let train_set = ShardSet::open(train).map_err(|e| format!("open train shards {}: {e}", train.display()))?;
    if train_set.vocab_size as usize != cfg.vocab {
        return Err(format!("train shards {} have vocab {}, the model {}", train.display(), train_set.vocab_size, cfg.vocab));
    }
    let steps = a.tokens / (a.batch * a.ctx);
    let steps_run = a.max_steps.map_or(steps, |m| m.min(steps));
    let mut sampler = Sampler::new(&train_set, a.ctx, a.data_seed).map_err(|e| format!("sampler: {e}"))?;
    if a.recall_every > 0 && !cfg.glance {
        return Err("--recall-every: recall rows carry glance marks, so the model needs --glance (v39's ava-num has it)".into());
    }
    // v41, the long phase: from step `long_start` on, every step reads windows of --long-ctx tokens (as many rows as
    // keep its tokens: batch · ctx / long_ctx), so the engram meets gaps longer than --ctx. Its sampler is made at
    // long_start (`long_sampler_after`): the same shard order as the short windows, from the first shard they never
    // reached, so the long phase reads new text.
    let long_start = if a.long_ctx > 0 { ((steps as f64) * a.long_from).round() as usize } else { usize::MAX };
    let long_rows = if a.long_ctx > 0 { (a.batch * a.ctx) / a.long_ctx } else { 0 };
    if a.long_ctx > 0 && (long_rows == 0 || long_rows * a.long_ctx != a.batch * a.ctx || long_rows % a.world != 0) {
        return Err(format!("--long-ctx {} must split --batch · --ctx = {} tokens into whole rows that split over --world {}", a.long_ctx, a.batch * a.ctx, a.world));
    }
    let mut sampler_long: Option<Sampler> = None;
    let qa_ex = match &a.qa {
        Some(p) => marked(read_sft(p)?, cfg.glance),
        None => Vec::new(),
    };
    let qa_batches = make_batches(&qa_ex, a.qa_batch_tokens);
    let nb = qa_batches.len();
    // the qa_epochs · nb answer batches are spread evenly over all the steps (at most one per step), so answers
    // run to the end of training rather than stopping early when steps / batches is not a whole number
    let qa_total = if steps == 0 { 0 } else { (a.qa_epochs * nb).min(steps) };
    let qa_at = |s: usize| -> Option<usize> {
        let (lo, hi) = (s * qa_total / steps.max(1), (s + 1) * qa_total / steps.max(1));
        (hi > lo).then_some(lo)
    };
    let qa_every = if qa_total == 0 { 0.0 } else { steps as f64 / qa_total as f64 };
    if a.qa_epochs * nb > steps {
        println!("note: {} answer batches but only {steps} steps; one per step, {} are left out", a.qa_epochs * nb, a.qa_epochs * nb - steps);
    }
    let mut opt = AdamW::new(params_of(&varmap), a.wd as f64).map_err(|e| format!("adamw: {e}"))?;
    let mm = Mm { bf16: ava_gpu::bf16_on() };
    println!(
        "grow {} d={} layers={} glance={} mag={} engram={:?}: {steps} steps of {} x {} tokens (running {steps_run}); answers: {} examples in {nb} batches, one every {qa_every:.2} steps ({} epochs, weight {}); lr {} warmup {} bf16 {}",
        a.kind.name(),
        cfg.d,
        cfg.layers,
        cfg.glance,
        cfg.mag,
        cfg.engram,
        a.batch,
        a.ctx,
        qa_ex.len(),
        a.qa_epochs,
        a.qa_weight,
        a.lr,
        a.warmup,
        mm.bf16
    );
    if a.recall_every > 0 {
        println!("recall practice: {} row(s) every {} steps", a.recall_rows, a.recall_every);
    }
    if a.long_ctx > 0 {
        println!("long phase: from step {long_start} on, {long_rows} x {} tokens per step", a.long_ctx);
    }
    let sig = grow_signature(a, steps);
    let ckdir = a.out.join("ckpt");
    let mut start = 0usize;
    let (mut prior_s, mut prior_tok) = (0.0f64, 0u64);
    if a.resume {
        match fs::read_to_string(ckdir.join("state.txt")) {
            Err(_) => println!("--resume: no checkpoint in {}, starting at step 0", ckdir.display()),
            Ok(text) => {
                let get = |k: &str| text.lines().find_map(|l| l.strip_prefix(&format!("{k}="))).map(str::to_string).ok_or(format!("state.txt lacks {k}"));
                if get("args")? != sig {
                    return Err(format!("--resume: the checkpoint was made with other settings\n  checkpoint: {}\n  this run:   {sig}", get("args")?));
                }
                start = get("step")?.parse().map_err(|_| "bad step in state.txt")?;
                if start > steps_run {
                    return Err(format!("--resume: the checkpoint is at step {start}, past this run's last step {steps_run} (--max-steps)"));
                }
                let t: u64 = get("t")?.parse().map_err(|_| "bad t in state.txt")?;
                prior_s = get("secs")?.parse().map_err(|_| "bad secs in state.txt")?;
                prior_tok = get("tok")?.parse().map_err(|_| "bad tok in state.txt")?;
                varmap.load(ckdir.join(format!("model-{start}.safetensors"))).map_err(|e| format!("load checkpoint weights: {e}"))?;
                let st = candle_core::safetensors::load(ckdir.join(format!("optim-{start}.safetensors")), dev).map_err(|e| format!("load checkpoint moments: {e}"))?;
                opt.load_state(&st, t).map_err(|e| format!("restore moments: {e}"))?;
                sampler = Sampler::resume(&train_set, Cursor::parse(&get("cursor")?)?).map_err(|e| format!("resume sampler: {e}"))?;
                // the long sampler exists once the long phase has begun (a checkpoint at long_start or before has none)
                if start > long_start {
                    sampler_long = Some(Sampler::resume(&train_set, Cursor::parse(&get("cursor_long")?)?).map_err(|e| format!("resume long sampler: {e}"))?);
                }
                println!("resumed from the checkpoint at step {start}");
            }
        }
    }
    // many GPUs (src/dist.rs): every rank reads its share of each batch; rank 0's weights go to all; only rank 0
    // writes files. One process (--world 1) runs the exact single-process step.
    if a.batch % a.world != 0 {
        return Err(format!("--batch {} must split evenly over --world {}", a.batch, a.world));
    }
    let lead = a.rank == 0;
    let mut dist = ava_gpu::dist::Dist::connect(a.rank, a.world, &a.dist_addr)?;
    let vars: Vec<candle_core::Var> = opt.params().iter().map(|p| p.var.clone()).collect();
    dist.broadcast_vars(&vars)?;
    if a.world > 1 {
        // every rank must resume from the same checkpoint (step, AdamW count and data cursor), or the ranks' step
        // counts and moments would differ: round 1 sums each value, round 2 counts the ranks that differ from the
        // mean (none differ only when all are equal); then all ranks stop together on a mismatch
        let cur = sampler.cursor().to_line() + &sampler_long.as_ref().map(|s| s.cursor().to_line()).unwrap_or_default();
        let h = cur.bytes().fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32)) & 0xF_FFFF;
        let mine = [start as f32, opt.t as f32, h as f32];
        let mut sum = mine;
        dist.all_reduce_sum(&mut sum)?;
        let off = mine.iter().zip(&sum).any(|(m, s)| m * a.world as f32 != *s);
        let mut bad = [off as u8 as f32];
        dist.all_reduce_sum(&mut bad)?;
        if bad[0] > 0.0 {
            return Err(format!("rank {}: the ranks resumed from different checkpoints (this rank: step {start}); give every rank the same --out/ckpt", a.rank));
        }
        println!("rank {} of {}: {} of the {} rows of every batch", a.rank, a.world, a.batch / a.world, a.batch);
    }
    // the log keeps its lines before the checkpoint
    let header = "step\tloss\tqa_loss\tlr\tgrad_norm\ttokens_per_sec\tstep_s\tctx\trecall_rows";
    let mut kept_lines = String::new();
    if start > 0 && lead {
        let old = fs::read_to_string(a.out.join("log.tsv")).unwrap_or_default();
        for l in old.lines().skip(1) {
            if l.split('\t').next().and_then(|s| s.parse::<usize>().ok()).is_some_and(|s| s < start) {
                kept_lines.push_str(l);
                kept_lines.push('\n');
            }
        }
    }
    let mut log = if lead {
        let mut f = fs::File::create(a.out.join("log.tsv")).map_err(|e| format!("log.tsv: {e}"))?;
        write!(f, "{header}\n{kept_lines}").map_err(|e| e.to_string())?;
        Some(f)
    } else {
        None
    };
    let mut order_cache: Option<(usize, Vec<usize>)> = None;
    let t_train = Instant::now();
    let (mut seen_tok, mut tail) = (0u64, Vec::new());
    // after a resume, the loss tail continues from the log lines kept from before the checkpoint
    for l in kept_lines.lines() {
        if let Some(x) = l.split('\t').nth(1).and_then(|s| s.parse::<f32>().ok()) {
            tail.push(x);
        }
    }
    for step in start..steps_run {
        let t0 = Instant::now();
        let long = step >= long_start;
        let (ctx_s, rows_s) = if long { (a.long_ctx, long_rows) } else { (a.ctx, a.batch) };
        if long && sampler_long.is_none() {
            let sl = long_sampler_after(&train_set, &sampler, a.long_ctx)?;
            println!("long phase begins at step {step}: {}", sl.cursor().to_line());
            sampler_long = Some(sl);
        }
        let mut windows = match (long, sampler_long.as_mut()) {
            (true, Some(sl)) => sl.batch(rows_s),
            _ => sampler.batch(rows_s),
        };
        // v41 recall practice: on every --recall-every-th step the --recall-rows windows with the fewest end-of-text
        // marks (the most book-like: one text, not many short ones; ties go to the later row) become recall rows (the
        // window's own text with made-up notes planted in it and asked about later; ava_data::recall). Every rank
        // builds the same rows (the seed is the step), then takes its share.
        let mut n_recall = 0usize;
        if a.recall_every > 0 && step % a.recall_every == 0 {
            let mut rng = SplitMix64::new(derive_seed(a.seed, 0x7265_6361_6c6c, step as u64));
            let mut idx: Vec<usize> = (0..windows.len()).collect();
            idx.sort_by_key(|&i| (windows[i].iter().filter(|&&t| t == EOS).count(), std::cmp::Reverse(i)));
            for &i in idx.iter().take(a.recall_rows.min(rows_s)) {
                windows[i] = ava_data::recall::recall_row(&mut rng, bpe, &windows[i], ctx_s);
                n_recall += 1;
            }
        }
        let rows_here = rows_s / a.world;
        let (ids, tg) = train_io(&windows[a.rank * rows_here..(a.rank + 1) * rows_here]);
        let ids = Tensor::from_vec(ids, (rows_here, ctx_s), dev).map_err(|e| e.to_string())?;
        let tg = Tensor::from_vec(tg, (rows_here, ctx_s), dev).map_err(|e| e.to_string())?;
        let (loss_t, _, mut grads) = if a.world == 1 {
            model.loss_and_grads_a_w(&ids, &tg, None, &mm)
        } else {
            // this rank's part of the whole batch's mean: weight 1 / (rows · ctx) per target, summed over ranks
            let w = Tensor::full(1.0f32 / (rows_s * ctx_s) as f32, rows_here * ctx_s, dev).map_err(|e| e.to_string())?;
            model.loss_and_grads_a_wr(&ids, &tg, Some(&w), true, &mm)
        }
        .map_err(|e| format!("step {step}: {e}"))?;
        let mut step_tok = rows_s * ctx_s;
        let mut qa_loss = f32::NAN;
        if let Some(q) = qa_at(step) {
            let (ep, k) = (q / nb, q % nb);
            if ep < a.qa_epochs {
                if order_cache.as_ref().map(|c| c.0) != Some(ep) {
                    let mut o: Vec<usize> = (0..nb).collect();
                    SplitMix64::new(derive_seed(a.seed, 0x7161, ep as u64)).shuffle(&mut o);
                    order_cache = Some((ep, o));
                }
                let all_rows = &qa_batches[order_cache.as_ref().unwrap().1[k]];
                // with several ranks each takes every world-th example, weighted by the whole batch's answer count
                let rows: Vec<usize> = all_rows.iter().enumerate().filter(|(j, _)| j % a.world == a.rank).map(|(_, &i)| i).collect();
                let total_w: usize = all_rows.iter().map(|&i| qa_ex[i].toks.len() - qa_ex[i].np).sum();
                if !rows.is_empty() {
                    let seqs: Vec<&[u32]> = rows.iter().map(|&i| qa_ex[i].toks.as_slice()).collect();
                    let from: Vec<usize> = rows.iter().map(|&i| qa_ex[i].np).collect();
                    let (qi, qt, qw, t) = padded(&seqs, &from);
                    let b = rows.len();
                    let qi = Tensor::from_vec(qi, (b, t), dev).map_err(|e| e.to_string())?;
                    let qt = Tensor::from_vec(qt, (b, t), dev).map_err(|e| e.to_string())?;
                    let (ql, _, qg) = if a.world == 1 {
                        let qw = Tensor::from_vec(qw, b * t, dev).map_err(|e| e.to_string())?;
                        model.loss_and_grads_a_w(&qi, &qt, Some(&qw), &mm)
                    } else {
                        let qw: Vec<f32> = qw.iter().map(|x| x / total_w.max(1) as f32).collect();
                        let qw = Tensor::from_vec(qw, b * t, dev).map_err(|e| e.to_string())?;
                        model.loss_and_grads_a_wr(&qi, &qt, Some(&qw), true, &mm)
                    }
                    .map_err(|e| format!("step {step} answers: {e}"))?;
                    merge(&opt, &mut grads, &qg, a.qa_weight)?;
                    qa_loss = ql.to_scalar::<f32>().map_err(|e| e.to_string())?;
                } else {
                    qa_loss = 0.0;
                }
                step_tok += all_rows.iter().map(|&i| qa_ex[i].toks.len() - 1).sum::<usize>();
            }
        }
        let mut loss = loss_t.to_scalar::<f32>().map_err(|e| e.to_string())?;
        if a.world > 1 {
            dist.all_reduce_grads(&vars, &mut grads)?;
            let mut l = [loss, if qa_loss.is_nan() { 0.0 } else { qa_loss }];
            dist.all_reduce_sum(&mut l)?;
            loss = l[0];
            if !qa_loss.is_nan() {
                qa_loss = l[1];
            }
        }
        let lr = cosine_lr(step, a.warmup, steps, a.lr, a.lr * a.min_lr_frac);
        let gn = opt.grad_norm(&grads).map_err(|e| format!("step {step} grad norm: {e}"))?;
        opt.step(&grads, lr as f64, clip_scale(gn, a.clip as f64)).map_err(|e| format!("step {step} adamw: {e}"))?;
        if !loss.is_finite() {
            return Err(format!("step {step}: loss is {loss}"));
        }
        let secs = t0.elapsed().as_secs_f64();
        seen_tok += step_tok as u64;
        tail.push(loss);
        if let Some(log) = log.as_mut() {
            writeln!(log, "{step}\t{loss:.6}\t{qa_loss:.6}\t{lr:.6e}\t{gn:.6}\t{:.1}\t{secs:.4}\t{ctx_s}\t{n_recall}", step_tok as f64 / secs).map_err(|e| e.to_string())?;
        }
        if lead && (step % a.log_every == 0 || step + 1 == steps_run) {
            let el = t_train.elapsed().as_secs_f64();
            println!("step {step}/{steps_run} loss {loss:.4} answers {qa_loss:.4} lr {lr:.3e} gnorm {gn:.3} {:.0} tok/s (mean {:.0})", step_tok as f64 / secs, seen_tok as f64 / el);
        }
        if lead && a.ckpt_every > 0 && (step + 1) % a.ckpt_every == 0 && step + 1 < steps_run {
            save_checkpoint(&ckdir, step + 1, &varmap, &opt, &sampler, sampler_long.as_ref(), &sig, prior_s + t_train.elapsed().as_secs_f64(), prior_tok + seen_tok)?;
        }
    }
    if !lead {
        println!("rank {}: done ({} steps); rank 0 saves and scores the model", a.rank, steps_run - start);
        return Ok(());
    }
    let here_s = t_train.elapsed().as_secs_f64();
    println!("this session: {} steps in {here_s:.1}s ({:.0} tokens/s)", steps_run - start, seen_tok as f64 / here_s);
    // the whole run, every session (a resumed run adds the seconds and tokens stored in its checkpoint)
    let (train_s, seen_tok) = (prior_s + here_s, prior_tok + seen_tok);
    let n_tail = tail.len().min(200);
    let tail_mean = tail.iter().rev().take(n_tail).sum::<f32>() / n_tail.max(1) as f32;
    println!("grew {steps_run} steps in {train_s:.1}s ({:.0} tokens/s); reading loss, last {n_tail} steps: {tail_mean:.4}", seen_tok as f64 / train_s);
    finish(
        &model,
        &varmap,
        a,
        bpe,
        dev,
        vec![
            ("phase", "grow".into()),
            ("init", a.init.display().to_string()),
            ("mag", a.mag.to_string()),
            ("engram", format!("{:?}", a.engram)),
            ("ctx", a.ctx.to_string()),
            ("batch", a.batch.to_string()),
            ("tokens", a.tokens.to_string()),
            ("steps", steps_run.to_string()),
            ("lr", a.lr.to_string()),
            ("qa_examples", qa_ex.len().to_string()),
            ("qa_every", format!("{qa_every:.3}")),
            ("recall_every", a.recall_every.to_string()),
            // from the schedule, so a resumed run counts the rows of its earlier sessions too
            ("recall_rows_total", (0..steps_run).filter(|s| a.recall_every > 0 && s % a.recall_every == 0).map(|s| a.recall_rows.min(if s >= long_start { long_rows } else { a.batch })).sum::<usize>().to_string()),
            ("long_ctx", a.long_ctx.to_string()),
            ("long_from_step", if a.long_ctx > 0 { long_start.to_string() } else { "-".into() }),
            ("loss_tail", format!("{tail_mean:.6}")),
            ("train_seconds", format!("{train_s:.2}")),
            ("tokens_per_sec", format!("{:.1}", seen_tok as f64 / train_s)),
        ],
    )
}

// ------------------------------------------------------------------------------------------------ sleep

/// Episodes: blocks separated by blank lines; a file without blank lines has one episode per line.
fn episodes(text: &str) -> Vec<String> {
    let text = text.replace("\r\n", "\n");
    let blocks: Vec<String> = text.split("\n\n").map(|b| b.trim().to_string()).filter(|b| !b.is_empty()).collect();
    if blocks.len() > 1 {
        return blocks;
    }
    text.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
}

fn phase_sleep(a: &Args, dev: &Device, bpe: &Bpe) -> Result<(), String> {
    let facts = a.facts.as_ref().ok_or("--phase sleep needs --facts")?;
    let (cfg, varmap, model) = ava_gpu::ckpt::load_auto(&a.init, a.kind, a.eval_ctx, dev)?;
    model.set_track_activity(false);
    let text = fs::read_to_string(facts).map_err(|e| format!("read {}: {e}", facts.display()))?;
    let eps: Vec<Vec<u32>> = episodes(&text)
        .iter()
        .map(|e| {
            let t = bpe.encode(e);
            let mut s = vec![EOS];
            s.extend(if cfg.glance { insert_marks(&t) } else { t });
            s.push(EOS);
            s
        })
        .collect();
    if eps.is_empty() {
        return Err(format!("{}: no episodes", facts.display()));
    }
    let mut replay = Replay::open(a, 1)?;
    let mut opt = AdamW::new(params_of(&varmap), a.wd as f64).map_err(|e| format!("adamw: {e}"))?;
    let mm = Mm { bf16: ava_gpu::bf16_on() };
    let steps = a.max_steps.map_or(a.sleep_steps, |m| m.min(a.sleep_steps));
    let per_cycle = a.sleep_steps.div_ceil(a.cycles).max(1);
    println!(
        "sleep: {} episodes ({} tokens) from {}; {steps} steps of {} episodes, {} cycles, surprise alpha {}; replay {} x {}; engram off while asleep",
        eps.len(),
        eps.iter().map(Vec::len).sum::<usize>(),
        facts.display(),
        a.sleep_rows,
        a.cycles,
        a.surprise_alpha,
        a.replay_rows,
        a.replay_ctx
    );
    // what she knows of the episodes without the engram: loss per token of each (all tokens after the first EOS)
    let ep_loss = |model: &Model| -> Result<Vec<f64>, String> {
        model.set_inference(true);
        let from = vec![1usize; eps.len()];
        let r = mc::scored_nll(model, dev, &eps, &from, a.score_batch).map_err(|e| format!("episode loss: {e}"));
        model.set_inference(false);
        Ok(r?.iter().map(|(s, n)| s / (*n).max(1) as f64).collect())
    };
    model.set_lesion(false, false, true); // the engram is off while she sleeps (and for the surprise measure)
    let anchor = measure_anchor(a, &model, &opt, &mm, dev, 11)?;
    let mut log = fs::File::create(a.out.join("log.tsv")).map_err(|e| format!("log.tsv: {e}"))?;
    writeln!(log, "step\tcycle\tepisode_loss\treplay_loss\tlr\tgrad_norm\ttokens_per_sec").map_err(|e| e.to_string())?;
    let mut cyc = fs::File::create(a.out.join("cycles.tsv")).map_err(|e| format!("cycles.tsv: {e}"))?;
    writeln!(cyc, "cycle\tstep\tmean_episode_loss\tmax_episode_loss\tmin_episode_loss").map_err(|e| e.to_string())?;
    let mut rng = SplitMix64::new(derive_seed(a.seed, 0x736c_6565_70, 0));
    let (mut prio, mut total) = (Vec::new(), 0.0);
    let loss_before = ep_loss(&model)?;
    let t_train = Instant::now();
    let mut seen_tok = 0u64;
    for step in 0..steps {
        let t0 = Instant::now();
        if step % per_cycle == 0 {
            let l = if step == 0 { loss_before.clone() } else { ep_loss(&model)? };
            let (mean, mx, mn) = (l.iter().sum::<f64>() / l.len() as f64, l.iter().cloned().fold(f64::MIN, f64::max), l.iter().cloned().fold(f64::MAX, f64::min));
            writeln!(cyc, "{}\t{step}\t{mean:.6}\t{mx:.6}\t{mn:.6}", step / per_cycle).map_err(|e| e.to_string())?;
            println!("cycle {} at step {step}: episode loss mean {mean:.4} (min {mn:.4}, max {mx:.4})", step / per_cycle);
            prio = l.iter().map(|&x| x.max(1e-6).powf(a.surprise_alpha)).collect();
            total = prio.iter().sum();
        }
        let rows: Vec<usize> = (0..a.sleep_rows).map(|_| draw(&prio, total, &mut rng)).collect();
        let seqs: Vec<&[u32]> = rows.iter().map(|&i| eps[i].as_slice()).collect();
        let (ids, tg, w, t) = padded(&seqs, &vec![1; rows.len()]);
        let b = rows.len();
        let ids = Tensor::from_vec(ids, (b, t), dev).map_err(|e| e.to_string())?;
        let tg = Tensor::from_vec(tg, (b, t), dev).map_err(|e| e.to_string())?;
        let w = Tensor::from_vec(w, b * t, dev).map_err(|e| e.to_string())?;
        let (l, _, mut grads) = model.loss_and_grads_a_w(&ids, &tg, Some(&w), &mm).map_err(|e| format!("sleep step {step}: {e}"))?;
        let mut step_tok = b * t;
        let mut rl = f32::NAN;
        if let Some(r) = replay.as_mut() {
            let (x, rg, n) = r.grads(&model, &mm, dev)?;
            merge(&opt, &mut grads, &rg, a.replay_weight)?;
            rl = x;
            step_tok += n;
        }
        let lr = cosine_lr(step, a.warmup.min(steps / 10), steps, a.lr, a.lr * a.min_lr_frac);
        let gn = opt.grad_norm(&grads).map_err(|e| format!("sleep step {step} grad norm: {e}"))?;
        opt.step(&grads, lr as f64, clip_scale(gn, a.clip as f64)).map_err(|e| format!("sleep step {step} adamw: {e}"))?;
        if let Some(an) = &anchor {
            an.pull(lr as f64).map_err(|e| format!("sleep step {step} anchor: {e}"))?;
        }
        let loss = l.to_scalar::<f32>().map_err(|e| e.to_string())?;
        if !loss.is_finite() {
            return Err(format!("sleep step {step}: loss is {loss}"));
        }
        let secs = t0.elapsed().as_secs_f64();
        seen_tok += step_tok as u64;
        writeln!(log, "{step}\t{}\t{loss:.6}\t{rl:.6}\t{lr:.6e}\t{gn:.6}\t{:.1}", step / per_cycle, step_tok as f64 / secs).map_err(|e| e.to_string())?;
        if step % a.log_every == 0 || step + 1 == steps {
            println!("sleep step {step}/{steps} episode loss {loss:.4} replay {rl:.4} lr {lr:.3e} gnorm {gn:.3}");
        }
    }
    let train_s = t_train.elapsed().as_secs_f64();
    let after = ep_loss(&model)?;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!("slept {steps} steps in {train_s:.1}s; episode loss per token (engram off) {:.4} -> {:.4}", mean(&loss_before), mean(&after));
    let drift = match &anchor {
        Some(an) => format!("{:.6e}", an.drift().map_err(|e| format!("anchor drift: {e}"))?),
        None => "nan".into(),
    };
    finish(
        &model,
        &varmap,
        a,
        bpe,
        dev,
        vec![
            ("phase", "sleep".into()),
            ("init", a.init.display().to_string()),
            ("facts", facts.display().to_string()),
            ("episodes", eps.len().to_string()),
            ("steps", steps.to_string()),
            ("sleep_rows", a.sleep_rows.to_string()),
            ("cycles", a.cycles.to_string()),
            ("surprise_alpha", a.surprise_alpha.to_string()),
            ("lr", a.lr.to_string()),
            ("episode_loss_before", format!("{:.6}", mean(&loss_before))),
            ("episode_loss_after", format!("{:.6}", mean(&after))),
            ("anchor", a.anchor.to_string()),
            ("fisher_batches", a.fisher_batches.to_string()),
            ("anchor_drift", drift),
            ("train_seconds", format!("{train_s:.2}")),
            ("tokens_per_sec", format!("{:.1}", seen_tok as f64 / train_s)),
        ],
    )
}

// ------------------------------------------------------------------------------------------------ reward

fn phase_reward(a: &Args, dev: &Device, bpe: &Bpe) -> Result<(), String> {
    let (cfg, varmap, model) = ava_gpu::ckpt::load_auto(&a.init, a.kind, a.eval_ctx, dev)?;
    model.set_track_activity(false);
    let mut items = Vec::new();
    let source = match &a.reward_items {
        // v41: a question file (qa_split's reward_train.tsv: train questions, none of the dev or sealed stems)
        Some(p) => {
            let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if name.contains("test") || p.components().any(|c| c.as_os_str() == "sealed") {
                return Err(format!("--reward-items {}: reward learning never sees a test set", p.display()));
            }
            items = mc::read_items(p)?;
            println!("reward pool: {}, {} questions", p.display(), items.len());
            p.display().to_string()
        }
        None => {
            for name in a.sets.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                if name.ends_with("test") {
                    return Err(format!("--sets {name}: reward learning never sees a test set"));
                }
                let qs = ava_gpu::qa::eval_set(name)?;
                println!("reward pool: {name}, {} questions", qs.len());
                items.extend(mc::from_qa(name, &qs, a.passage));
            }
            a.sets.clone()
        }
    };
    if items.is_empty() {
        return Err("no reward questions (--sets or --reward-items)".into());
    }
    let seqs: Vec<(Vec<Vec<u32>>, Vec<usize>)> = items.iter().map(|it| mc::item_seqs(bpe, it, cfg.glance)).collect();
    // each choice's bytes (for --norm byte: her policy scores NLL per byte, and each choice's gradient is divided by
    // its bytes to match)
    let byte_len = ava_gpu::eval::byte_lengths_n(bpe, cfg.vocab);
    let choice_bytes: Vec<Vec<f64>> = seqs.iter().map(|(s, f)| mc::scored_bytes(&byte_len, s, f)).collect();
    // the policy's temperature in the score's units: per byte, --temp is divided by the pool's mean choice bytes, so
    // π is as sharp as with total NLL on an answer of average length (and the per-byte gradient as strong)
    let mean_bytes = {
        let all: Vec<f64> = choice_bytes.iter().flatten().map(|b| b.max(1.0)).collect();
        all.iter().sum::<f64>() / all.len().max(1) as f64
    };
    let temp = match a.norm {
        mc::Norm::Total => a.temp,
        mc::Norm::Byte => a.temp / mean_bytes,
    };
    let mut replay = Replay::open(a, 2)?;
    let mut opt = AdamW::new(params_of(&varmap), a.wd as f64).map_err(|e| format!("adamw: {e}"))?;
    let mm = Mm { bf16: ava_gpu::bf16_on() };
    let steps = a.max_steps.map_or(a.reward_steps, |m| m.min(a.reward_steps));
    println!(
        "reward: {} questions; {steps} steps of {} questions; scoring {} (mean choice bytes {mean_bytes:.2}); tau {} temp {} (in score units {temp:.6}) baseline decay {}; replay {} x {}",
        items.len(),
        a.items,
        a.norm.name(),
        a.tau,
        a.temp,
        a.baseline_decay,
        a.replay_rows,
        a.replay_ctx
    );
    let anchor = measure_anchor(a, &model, &opt, &mm, dev, 12)?;
    let mut log = fs::File::create(a.out.join("log.tsv")).map_err(|e| format!("log.tsv: {e}"))?;
    writeln!(log, "step\tanswered\tright\twrong\tabstained\tmean_reward\tbaseline\tpolicy_loss\treplay_loss\tlr\tgrad_norm\tstep_s").map_err(|e| e.to_string())?;
    let mut rng = SplitMix64::new(derive_seed(a.seed, 0x7265_7761_7264, 0));
    let mut baseline = 0.0f64;
    let (mut tot_r, mut tot_w, mut tot_a) = (0usize, 0usize, 0usize);
    let t_train = Instant::now();
    for step in 0..steps {
        let t0 = Instant::now();
        // her answers (no gradient)
        let pick: Vec<usize> = (0..a.items).map(|_| rng.below(items.len() as u64) as usize).collect();
        let (mut all, mut from, mut span) = (Vec::new(), Vec::new(), Vec::new());
        for &q in &pick {
            span.push(all.len());
            all.extend(seqs[q].0.iter().cloned());
            from.extend(seqs[q].1.iter().cloned());
        }
        model.set_inference(true);
        let nll = mc::scored_nll(&model, dev, &all, &from, a.score_batch).map_err(|e| format!("reward step {step} scoring: {e}"));
        model.set_inference(false);
        let nll = nll?;
        // (first row in `all`, choice drawn, reward, π, each choice's gradient scale: 1, or 1 / its bytes)
        let mut chosen: Vec<(usize, usize, f64, Vec<f64>, Vec<f64>)> = Vec::new();
        let (mut r_n, mut w_n, mut a_n) = (0usize, 0usize, 0usize);
        for (k, &q) in pick.iter().enumerate() {
            let nc = items[q].choices.len();
            let tot = mc::normed(&nll[span[k]..span[k] + nc], &choice_bytes[q], a.norm);
            let scale: Vec<f64> = match a.norm {
                mc::Norm::Total => vec![1.0; nc],
                mc::Norm::Byte => choice_bytes[q].iter().map(|b| 1.0 / b.max(1.0)).collect(),
            };
            let g = mc::grade_one(&tot, items[q].right);
            if g.margin < a.tau {
                a_n += 1; // "I don't know": no reward, no update
                continue;
            }
            // her answer: drawn from softmax(-NLL / temp)
            let m = tot.iter().cloned().fold(f64::MAX, f64::min);
            let p: Vec<f64> = tot.iter().map(|&x| (-(x - m) / temp).exp()).collect();
            let z: f64 = p.iter().sum();
            let c = draw(&p, z, &mut rng);
            let right = c == items[q].right;
            let r = if right { 1.0 } else { -1.0 };
            if right {
                r_n += 1;
            } else {
                w_n += 1;
            }
            chosen.push((span[k], c, r, p.iter().map(|x| x / z).collect(), scale));
        }
        tot_r += r_n;
        tot_w += w_n;
        tot_a += a_n;
        let answered = chosen.len();
        let mean_r = if answered > 0 { chosen.iter().map(|x| x.2).sum::<f64>() / answered as f64 } else { 0.0 };
        let mut pl = f32::NAN;
        let mut grads: Option<GradStore> = None;
        if answered > 0 {
            // REINFORCE on her choice policy π = softmax(−S / temp) over the choices, S_j = NLL_j (· 1/bytes_j with
            // --norm byte):
            //   −(r − b) ∇log π_c = (r − b) / temp · Σ_j (1[j = c] − π_j) ∇S_j,
            // so every choice of an answered question is a row, weighted (r − b)(1[j = c] − π_j) / (temp · answered)
            // (· 1/bytes_j) on each of its scored tokens (π held fixed)
            let (mut rows, mut fr, mut rw): (Vec<&[u32]>, Vec<usize>, Vec<f32>) = (Vec::new(), Vec::new(), Vec::new());
            for (s, c, r, pi, scale) in &chosen {
                for (j, &pj) in pi.iter().enumerate() {
                    rows.push(all[s + j].as_slice());
                    fr.push(from[s + j]);
                    let hit = if j == *c { 1.0 } else { 0.0 };
                    rw.push(((r - baseline) * (hit - pj) * scale[j] / (temp * answered as f64)) as f32);
                }
            }
            let (ids, tg, w01, t) = padded(&rows, &fr);
            let w: Vec<f32> = w01.iter().enumerate().map(|(j, &x)| x * rw[j / t]).collect();
            let b = rows.len();
            let ids = Tensor::from_vec(ids, (b, t), dev).map_err(|e| e.to_string())?;
            let tg = Tensor::from_vec(tg, (b, t), dev).map_err(|e| e.to_string())?;
            let w = Tensor::from_vec(w, b * t, dev).map_err(|e| e.to_string())?;
            let (l, _, g) = model.loss_and_grads_a_wr(&ids, &tg, Some(&w), true, &mm).map_err(|e| format!("reward step {step}: {e}"))?;
            pl = l.to_scalar::<f32>().map_err(|e| e.to_string())?;
            grads = Some(g);
            baseline = a.baseline_decay * baseline + (1.0 - a.baseline_decay) * mean_r;
        }
        let mut rl = f32::NAN;
        if let Some(r) = replay.as_mut() {
            let (x, mut rg, _) = r.grads(&model, &mm, dev)?;
            rl = x;
            match grads.as_mut() {
                Some(g) => merge(&opt, g, &rg, a.replay_weight)?,
                None => {
                    // no answer this step: the reading gradient alone, scaled in place
                    if a.replay_weight != 1.0 {
                        for p in opt.params() {
                            let key = p.var.as_tensor();
                            if let Some(x) = rg.remove(key) {
                                rg.insert(key, x.affine(a.replay_weight as f64, 0.0).map_err(|e| e.to_string())?);
                            }
                        }
                    }
                    grads = Some(rg);
                }
            }
        }
        let lr = cosine_lr(step, a.warmup.min(steps / 10), steps, a.lr, a.lr * a.min_lr_frac);
        let mut gn = f64::NAN;
        if let Some(g) = &grads {
            gn = opt.grad_norm(g).map_err(|e| format!("reward step {step} grad norm: {e}"))?;
            opt.step(g, lr as f64, clip_scale(gn, a.clip as f64)).map_err(|e| format!("reward step {step} adamw: {e}"))?;
            if let Some(an) = &anchor {
                an.pull(lr as f64).map_err(|e| format!("reward step {step} anchor: {e}"))?;
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        writeln!(log, "{step}\t{answered}\t{r_n}\t{w_n}\t{a_n}\t{mean_r:.4}\t{baseline:.4}\t{pl:.6}\t{rl:.6}\t{lr:.6e}\t{gn:.6}\t{secs:.3}").map_err(|e| e.to_string())?;
        if step % a.log_every == 0 || step + 1 == steps {
            println!("reward step {step}/{steps}: right {r_n} wrong {w_n} abstained {a_n}, baseline {baseline:.3}, replay {rl:.4}, {secs:.2}s");
        }
    }
    let train_s = t_train.elapsed().as_secs_f64();
    println!("reward learning: {steps} steps in {train_s:.1}s; right {tot_r}, wrong {tot_w}, abstained {tot_a} (her own answers while learning)");
    let drift = match &anchor {
        Some(an) => format!("{:.6e}", an.drift().map_err(|e| format!("anchor drift: {e}"))?),
        None => "nan".into(),
    };
    finish(
        &model,
        &varmap,
        a,
        bpe,
        dev,
        vec![
            ("phase", "reward".into()),
            ("init", a.init.display().to_string()),
            ("sets", source),
            ("steps", steps.to_string()),
            ("items", a.items.to_string()),
            ("norm", a.norm.name().into()),
            ("anchor", a.anchor.to_string()),
            ("fisher_batches", a.fisher_batches.to_string()),
            ("anchor_drift", drift),
            ("tau", a.tau.to_string()),
            ("temp", a.temp.to_string()),
            ("temp_score_units", format!("{temp:.6}")),
            ("mean_choice_bytes", format!("{mean_bytes:.3}")),
            ("lr", a.lr.to_string()),
            ("right", tot_r.to_string()),
            ("wrong", tot_w.to_string()),
            ("abstained", tot_a.to_string()),
            ("train_seconds", format!("{train_s:.2}")),
        ],
    )
}
