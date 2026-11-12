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
// windows and answer batches. v52 --keep-ckpts: every model-STEP.safetensors stays (for scoring afterwards; the
// optim-STEP and EMA files are still pruned to the last). v52 --fused-eval: the end-of-run val BPB on the fused forward
// (eval::score_logits; runs/v52/fasteval.md). Both off by default: exactly the behaviour before them.
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
//
// v42 (Wave 2A) additions. Every one is off by default, so a v41 command line computes exactly what it did; the
// ones marked BUG fix a real fault and v42 runs should turn them on.
// - --decay-gates (W1, BUG): the memory's output gate .wg gets weight decay too (Model::decays_with; .wa/.wb stay
//   undecayed on purpose, see there).
// - --fast-step (Q6): the fused optimizer step (AdamW::step_t, one launch for all parameters) and, in the grow loop,
//   no wait for the GPU inside a step: losses and norms come over in blocks at shown steps and checkpoints (as
//   train_gpu does). Meant to give the same weights bit for bit.
// - --beta2 B (Q1): AdamW's beta2 (0.95).
// - --sleep-lesion episodes|all (X1, BUG): `all` (old) switches the engram off for the whole sleep, so her usual
//   reading (replay) and the anchor's measure run without it too; `episodes` switches it off only around the episode
//   loss and the episode-loss measure.
// - --replay-cursor FILE (X2, BUG): every night used to replay the same windows (the replay seed never changed). The
//   file holds `night=N`; night N draws its replay (and anchor) windows from its own seed (night 0 = the old one), and
//   the file moves on by one when the phase ends.
// - --anchor-save FILE / --anchor-load FILE [--anchor-gamma 1.0] [--anchor-point now|saved] (U3): the running anchor.
//   The anchor (w0 and importance F) is saved at the end of the phase; a later phase reads it and uses
//   F = γ·F_old + F_new. w0 = the weights the phase starts from (now: online EWC) or the saved w0 (saved).
//   --anchor-episodes FILE[,FILE...]: importance is also measured on these (episode text, blank-line separated, or
//   AVASFT01 answers: slept episodes, QA, practice) with the engram off, so facts already learned are protected.
// - --load-opt FILE / --save-opt (U1): start from saved Adam moments (DIR/optim.safetensors, or a grow checkpoint's
//   optim-STEP file, whose step count is read from state.txt beside it); save them as DIR/optim.safetensors.
//   --wd-target w0|zero (U1): weight decay toward w0 (L2-Init; w0 = the anchor's w0 if there is an anchor, else the
//   weights the phase starts from) instead of toward 0. --sleep-lr X and --sleep-warmup N: the sleep's own peak LR and
//   warm-up (default --lr and min(--warmup, steps / 10), the old rule), so a small LR such as 3e-5 can be chosen.
// - --kl-ref CKPT --kl-weight X (U2): a frozen reference model (the serving copy); on the replay windows the step
//   adds X · KL(reference || current) over next-token distributions (exact gradient, candle's autograd). With a
//   reference, the KL on fixed windows is logged each cycle (DIR/kl.tsv: the forgetting gate), even at weight 0.
// - --reward-group K (U4): K answers drawn per question, the group's mean reward as the baseline (GRPO style; K = 1 is
//   the old running baseline). right/wrong/abstained then count draws. --idk-action: "I don't know" becomes a scored
//   choice of its own, reward 0 (right +1, wrong −1: TruthRL's three-way reward); picking it counts as abstained.
// - --sleep-stop-loss X (U5): the sleep stops at the first cycle whose mean episode loss is below X. --sleep-dev FILE
//   (an item TSV, never a test file): right − wrong and the abstain rate (greedy answers, --tau, and "I don't know"
//   with --idk-action) at every cycle and at the end, in DIR/dev.tsv (the reward phase checks at --cycles points).
// - --skip-bad-batch [--bad-norm-mult 10] [--bad-norm-window 100] (J3): before the optimizer step, a step whose loss or
//   gradient norm is not finite, or whose norm is over K × the running median, is skipped and counted
//   (DIR/skipped.tsv); 20 in a row stop the run.
// - --late-qa-frac F (Q3): the answer batches go in the last F of the grow steps (0: spread over all, the old rule).
// - --gpu-share F (v43): after each grow step, wait for the GPU, then rest so it is busy about F of the time (a video
//   or game sharing the GPU stays smooth). Only the speed changes: the numbers are the same.
//   (Not in the run signature, so a run can resume with a different share.)
//   --ema D (Q4): an EMA copy of the weights (e ← D·e + (1 − D)·w after every step), saved as model.ema.safetensors
//   (and beside each grow checkpoint, so --resume keeps it).
//
// v42 (Wave 2B) recipe, all off by default (the same as train_gpu's; each joins the resume signature and a `recipe`
// result column only when set):
// - --schedule cosine|wsd [--decay-frac 0.2] [--decay-shape 1-sqrt|linear] [--lr-min X] (A1): every phase's LR; WSD
//   holds the peak and anneals over the last decay-frac of the phase's steps to --lr-min (default peak · min-lr-frac).
// - --opt adamw|muon [--muon-lr 0.02] (A1): Muon for the decayed 2-D hidden matrices (in every phase), AdamW for the
//   rest. Muon's momentum sits in the m buffers, so --load-opt from an AdamW run gives Muon a stale momentum.
// - --ctx-warmup TOKENS [--ctx-start 256] (I2, grow phase): the short phase's context doubles from ctx-start to --ctx
//   over the first TOKENS (each window cut into pieces: the same targets and tokens per step). No recall rows while
//   it warms up. The long phase is never cut.
// - --grokfast-alpha A --grokfast-lambda L (I4): Grokfast-EMA on every update's gradient (on when L > 0); with
//   --skip-bad-batch the guard judges the raw gradient first. The averages ride in grow checkpoints; each phase
//   starts them afresh.
// v42 Wave 2C, block options (src/arch.rs, src/memory.rs; all off by default): --phase grow takes them as flags
// (the init's tensors copy over, the new parts start as no-ops; without flags it keeps the init's own arch.txt);
// sleep and reward keep the --init model's own (read from its arch.txt). Every phase writes DIR/arch.txt when the
// model has any, and --out-model FILE gets FILE's NAME.arch.txt. Their steps go through autograd (the hand-written D
// step does not know them); homeostasis (--homeostasis) is applied after every update.
// - sleep --redo TAU [--redo-batches 8] (A8): ReDo before the sleep: LIF channels and MLP units whose activity is at
//   most TAU x their layer's mean get fresh incoming and zero outgoing weights (src/upkeep.rs; DIR/upkeep.tsv).
// - sleep --shrink-perturb L,S (A8): every hidden matrix becomes L·W + S·rms(W)·noise before the sleep.
// - sleep --sparse-ft T [--sparse-ft-batches 8] (A9): only the T product-key value rows the episodes read most
//   (TF-IDF against her usual reading) learn; every other weight stays frozen. Needs --pk-mem and replay.
//
// v42 (Wave 4, K6) the learner's train command. --out-model FILE (sleep and reward only): the phase's model is also
// copied to FILE when the phase ends (through FILE.tmp and a rename, so FILE is whole or absent; a failed phase leaves
// no FILE). --out may then be left out (the run folder is FILE + ".run"), and a result.tsv left in the run folder by
// an earlier run is renamed result.prev.tsv first, so every call trains (the learner calls it night after night with
// the same paths). This is the always-online learner's --train-cmd contract: {model} -> --init, {episodes} ->
// --facts, {out} -> --out-model. Without --out-model nothing changes.

use ava_data::bpe::{Bpe, EOS, PAD};
use ava_data::numbers::insert_marks;
use ava_data::rng::{derive_seed, SplitMix64};
use ava_data::shards::{Cursor, Sampler, ShardSet, ShardTokens};
use ava_gpu::eval::{byte_lengths_n, evaluate, train_io};
use ava_gpu::anchor::{Anchor, Importance};
use ava_gpu::mc;
use ava_gpu::model::{Kind, Mm, Model};
use ava_gpu::optim::{clip_scale, clip_scale_t, AdamW, DecayToward, MuonCfg, Param, Schedule};
use ava_gpu::qa::{read_sft, Example};
use ava_gpu::trainkit::{self, ctx_at, ctx_warmup_check, split_windows, BadBatch, Ema, Grokfast, Outcome, QaPlan, Tally, Verdict};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: grow_gpu --phase grow|sleep|reward --init FILE (--out DIR | --out-model FILE) [options; see the top of grow_gpu.rs]";

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
    // v42 (see the top of the file)
    decay_gates: bool,
    fast_step: bool,
    beta2: f64,
    sleep_lesion: SleepLesion,
    replay_cursor: Option<PathBuf>,
    /// not a flag: the night read from --replay-cursor at the start of a sleep or reward phase
    replay_night: Option<u64>,
    anchor_save: Option<PathBuf>,
    anchor_load: Option<PathBuf>,
    anchor_gamma: f64,
    anchor_saved_point: bool,
    anchor_episodes: Vec<PathBuf>,
    load_opt: Option<PathBuf>,
    save_opt: bool,
    wd_target_w0: bool,
    sleep_lr: Option<f32>,
    sleep_warmup: Option<usize>,
    kl_ref: Option<PathBuf>,
    kl_weight: f64,
    reward_group: usize,
    idk_action: bool,
    sleep_stop_loss: Option<f64>,
    sleep_dev: Option<PathBuf>,
    skip_bad_batch: bool,
    /// v52 fasteval: the end-of-run val BPB on the fused forward (eval::set_fused around finish's evaluate)
    fused_eval: bool,
    /// v52: keep every model-STEP.safetensors checkpoint (optim and EMA files still pruned to the last)
    keep_ckpts: bool,
    bad_norm_mult: f64,
    bad_norm_window: usize,
    late_qa_frac: f64,
    /// v43: the share of time the GPU may be busy in the grow loop (1: no rests)
    gpu_share: f64,
    ema: Option<f64>,
    // v42 Wave 2B recipe (see the top of the file)
    schedule: Schedule,
    lr_min: Option<f32>,
    muon: Option<f64>,
    ctx_warmup: u64,
    ctx_start: usize,
    gf_alpha: f64,
    gf_lambda: f64,
    /// v42 (K6): the learner's train command: the model is copied here at the end too
    out_model: Option<PathBuf>,
    /// v42 Wave 2C: block options (src/arch.rs, src/memory.rs) for --phase grow; sleep and reward take the --init
    /// model's own (its arch.txt)
    arch: ava_gpu::arch::Arch,
    /// v42 (A8), sleep: ReDo at tau, measured on this many replay batches; shrink-and-perturb (lambda, sigma)
    redo: Option<f32>,
    redo_batches: usize,
    shrink_perturb: Option<(f64, f64)>,
    /// v42 (A9), sleep: train only the top-T product-key value rows by TF-IDF (episodes vs replay batches)
    sparse_ft: Option<usize>,
    sparse_ft_batches: usize,
}

/// X1: which parts of the sleep run with the engram switched off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SleepLesion {
    /// v41: the whole sleep (episodes, replay, the anchor's measure)
    All,
    /// v42: only the episode loss and the episode-loss measure
    Episodes,
}

/// A file a learning phase reads must not be a test set (the name has no "test"; nothing under a sealed/ folder).
fn not_a_test(p: &Path, flag: &str) -> Result<(), String> {
    let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    let bad = ["test", "sealed", "spent", "heldout"];
    if bad.iter().any(|b| name.contains(b)) || p.components().any(|c| bad.iter().any(|b| c.as_os_str().to_string_lossy().to_lowercase() == *b)) {
        return Err(format!("{flag} {}: learning never sees a test set", p.display()));
    }
    Ok(())
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
        decay_gates: false,
        fast_step: false,
        beta2: 0.95,
        sleep_lesion: SleepLesion::All,
        replay_cursor: None,
        replay_night: None,
        anchor_save: None,
        anchor_load: None,
        anchor_gamma: 1.0,
        anchor_saved_point: false,
        anchor_episodes: Vec::new(),
        load_opt: None,
        save_opt: false,
        wd_target_w0: false,
        sleep_lr: None,
        sleep_warmup: None,
        kl_ref: None,
        kl_weight: 0.0,
        reward_group: 1,
        idk_action: false,
        sleep_stop_loss: None,
        sleep_dev: None,
        skip_bad_batch: false,
        fused_eval: false,
        keep_ckpts: false,
        bad_norm_mult: 10.0,
        bad_norm_window: 100,
        late_qa_frac: 0.0,
        gpu_share: 1.0,
        ema: None,
        schedule: Schedule::Cosine,
        lr_min: None,
        muon: None,
        ctx_warmup: 0,
        ctx_start: 256,
        gf_alpha: 0.98,
        gf_lambda: 0.0,
        out_model: None,
        arch: ava_gpu::arch::Arch::default(),
        redo: None,
        redo_batches: 8,
        shrink_perturb: None,
        sparse_ft: None,
        sparse_ft_batches: 8,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    // v42 Wave 2B: --schedule, --decay-frac, --decay-shape and --opt, --muon-lr (checked after the loop)
    let mut sched = ("cosine".to_string(), 0.2f64, "1-sqrt".to_string());
    let mut opt_kind = ("adamw".to_string(), 0.02f64);
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        // v42 Wave 2C: the block options (src/arch.rs, src/memory.rs; no name is shared with grow_gpu's own flags)
        if let Some(n) = a.arch.take_flag(&args, i)? {
            i += n;
            continue;
        }
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
            "--decay-gates" | "--fast-step" | "--save-opt" | "--idk-action" | "--skip-bad-batch" | "--fused-eval" | "--keep-ckpts" => {
                match key {
                    "--decay-gates" => a.decay_gates = true,
                    "--fast-step" => a.fast_step = true,
                    "--save-opt" => a.save_opt = true,
                    "--idk-action" => a.idk_action = true,
                    "--fused-eval" => a.fused_eval = true,
                    "--keep-ckpts" => a.keep_ckpts = true,
                    _ => a.skip_bad_batch = true,
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
            "--phase" => a.phase = v.into(),
            "--init" => a.init = v.into(),
            "--out-model" => a.out_model = Some(v.into()),
            "--redo" => a.redo = Some(fl(v)? as f32),
            "--redo-batches" => a.redo_batches = num(v)?.max(1),
            "--shrink-perturb" => {
                let (l, s) = v.split_once(',').ok_or_else(|| format!("--shrink-perturb {v:?}: give LAMBDA,SIGMA (e.g. 0.9,0.01)"))?;
                a.shrink_perturb = Some((fl(l)?, fl(s)?));
            }
            "--sparse-ft" => a.sparse_ft = Some(num(v)?),
            "--sparse-ft-batches" => a.sparse_ft_batches = num(v)?.max(1),
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
            "--norm" => {
                a.norm = mc::Norm::parse(v)?;
                if a.norm == mc::Norm::Pmi {
                    return Err("--norm pmi is for scoring (mc_eval); reward learning takes total or byte".into());
                }
            }
            "--reward-items" => a.reward_items = Some(v.into()),
            "--beta2" => a.beta2 = fl(v)?,
            "--sleep-lesion" => {
                a.sleep_lesion = match v {
                    "all" => SleepLesion::All,
                    "episodes" => SleepLesion::Episodes,
                    _ => return Err(format!("--sleep-lesion {v:?}: all or episodes")),
                }
            }
            "--replay-cursor" => a.replay_cursor = Some(v.into()),
            "--anchor-save" => a.anchor_save = Some(v.into()),
            "--anchor-load" => a.anchor_load = Some(v.into()),
            "--anchor-gamma" => a.anchor_gamma = fl(v)?,
            "--anchor-point" => {
                a.anchor_saved_point = match v {
                    "now" => false,
                    "saved" => true,
                    _ => return Err(format!("--anchor-point {v:?}: now or saved")),
                }
            }
            "--anchor-episodes" => a.anchor_episodes = v.split(',').map(str::trim).filter(|s| !s.is_empty()).map(PathBuf::from).collect(),
            "--load-opt" => a.load_opt = Some(v.into()),
            "--wd-target" => {
                a.wd_target_w0 = match v {
                    "zero" => false,
                    "w0" => true,
                    _ => return Err(format!("--wd-target {v:?}: zero or w0")),
                }
            }
            "--sleep-lr" => a.sleep_lr = Some(fl(v)? as f32),
            "--sleep-warmup" => a.sleep_warmup = Some(num(v)?),
            "--kl-ref" => a.kl_ref = Some(v.into()),
            "--kl-weight" => a.kl_weight = fl(v)?,
            "--reward-group" => a.reward_group = num(v)?,
            "--sleep-stop-loss" => a.sleep_stop_loss = Some(fl(v)?),
            "--sleep-dev" => a.sleep_dev = Some(v.into()),
            "--bad-norm-mult" => a.bad_norm_mult = fl(v)?,
            "--bad-norm-window" => a.bad_norm_window = num(v)?,
            "--late-qa-frac" => a.late_qa_frac = fl(v)?,
            "--gpu-share" => a.gpu_share = fl(v)?.clamp(0.05, 1.0),
            "--ema" => a.ema = Some(fl(v)?),
            // v42 Wave 2B recipe
            "--schedule" => sched.0 = v.to_string(),
            "--decay-frac" => sched.1 = fl(v)?,
            "--decay-shape" => sched.2 = v.to_string(),
            "--lr-min" => a.lr_min = Some(fl(v)? as f32),
            "--opt" => opt_kind.0 = v.to_string(),
            "--muon-lr" => opt_kind.1 = fl(v)?,
            "--ctx-warmup" => a.ctx_warmup = num(v)? as u64,
            "--ctx-start" => a.ctx_start = num(v)?,
            "--grokfast-alpha" => a.gf_alpha = fl(v)?,
            "--grokfast-lambda" => a.gf_lambda = fl(v)?,
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    if let Some(m) = &a.out_model {
        if a.phase == "grow" {
            return Err("--out-model is for the sleep and reward phases (grow resumes from its run folder)".into());
        }
        if a.out.as_os_str().is_empty() {
            a.out = run_dir_for(m);
        }
    }
    if a.init.as_os_str().is_empty() || a.out.as_os_str().is_empty() {
        return Err(format!("--init and --out (or --out-model) are required\n{USAGE}"));
    }
    if !matches!(a.kind, Kind::D | Kind::A) {
        return Err("grow_gpu trains D or A (their hand-written step)".into());
    }
    // v42 checks
    if !(a.beta2 > 0.0 && a.beta2 < 1.0) {
        return Err(format!("--beta2 {}: must be in (0, 1)", a.beta2));
    }
    if a.anchor <= 0.0 && (a.anchor_save.is_some() || a.anchor_load.is_some() || !a.anchor_episodes.is_empty()) {
        return Err("--anchor-save, --anchor-load and --anchor-episodes need --anchor > 0".into());
    }
    if a.anchor_saved_point && a.anchor_load.is_none() {
        return Err("--anchor-point saved needs --anchor-load".into());
    }
    if !(a.anchor_gamma >= 0.0 && a.anchor_gamma.is_finite()) {
        return Err(format!("--anchor-gamma {}: must be 0 or more", a.anchor_gamma));
    }
    for p in &a.anchor_episodes {
        not_a_test(p, "--anchor-episodes")?;
    }
    if let Some(p) = &a.sleep_dev {
        not_a_test(p, "--sleep-dev")?;
    }
    if a.kl_weight < 0.0 || !a.kl_weight.is_finite() {
        return Err(format!("--kl-weight {}: must be 0 or more", a.kl_weight));
    }
    if a.kl_weight > 0.0 && (a.kl_ref.is_none() || a.replay_rows == 0) {
        return Err("--kl-weight > 0 needs --kl-ref and replay (--replay-rows > 0): the KL is taken on the replay windows".into());
    }
    if a.reward_group == 0 {
        return Err("--reward-group must be 1 or more".into());
    }
    if a.skip_bad_batch && !(a.bad_norm_mult > 1.0 && a.bad_norm_window > 0) {
        return Err("--skip-bad-batch needs --bad-norm-mult > 1 and --bad-norm-window > 0".into());
    }
    if !(0.0..=1.0).contains(&a.late_qa_frac) {
        return Err(format!("--late-qa-frac {}: must be in [0, 1]", a.late_qa_frac));
    }
    if let Some(d) = a.ema {
        if !(d > 0.0 && d < 1.0) {
            return Err(format!("--ema {d}: the decay must be in (0, 1)"));
        }
    }
    if a.arch.dale > 0.0 {
        // v50 (src/v50.rs): --loop and --neuromod grow as any block option; --dale's weight decay (Model::dale_decay
        // after each optimizer step, as train_gpu does) is not wired into grow_gpu's guarded updater yet
        return Err("--dale: train with train_gpu (grow_gpu does not apply its weight decay yet)".into());
    }
    if a.wd_target_w0 && a.world > 1 {
        // each rank would take its own w0 before rank 0's weights are sent out, and the new parts' random init may
        // differ between ranks
        return Err("--wd-target w0 runs on one GPU (--world 1) for now".into());
    }
    if let Some(x) = a.sleep_lr {
        if !(x > 0.0) {
            return Err("--sleep-lr must be positive".into());
        }
    }
    // v42 Wave 2B recipe checks
    a.schedule = Schedule::parse(&sched.0, sched.1, &sched.2)?;
    if let Some(x) = a.lr_min {
        if !(x >= 0.0) {
            return Err(format!("--lr-min {x}: must be 0 or more"));
        }
    }
    a.muon = match opt_kind.0.as_str() {
        "adamw" => None,
        "muon" if opt_kind.1 > 0.0 => Some(opt_kind.1),
        "muon" => return Err(format!("--muon-lr {}: must be positive", opt_kind.1)),
        o => return Err(format!("--opt {o:?}: adamw or muon")),
    };
    if a.ctx_warmup > 0 {
        ctx_warmup_check(a.ctx_start, a.ctx)?;
    }
    if !(0.0..1.0).contains(&a.gf_alpha) || !(a.gf_lambda >= 0.0) {
        return Err(format!("--grokfast-alpha {} must be in [0, 1) and --grokfast-lambda {} 0 or more", a.gf_alpha, a.gf_lambda));
    }
    // v42 Wave 2C checks
    if !a.arch.is_default() && a.phase != "grow" {
        return Err(format!("block options{}: --phase {} keeps the --init model's own (its arch.txt); give them to --phase grow", a.arch.signature(), a.phase));
    }
    if (a.redo.is_some() || a.shrink_perturb.is_some() || a.sparse_ft.is_some()) && a.phase != "sleep" {
        return Err("--redo, --shrink-perturb and --sparse-ft are sleep options (--phase sleep)".into());
    }
    if let Some(t) = a.redo {
        if !(0.0..1.0).contains(&t) {
            return Err(format!("--redo {t}: tau must be in [0, 1) (0.1 is ReDo's usual)"));
        }
    }
    if let Some((l, s)) = a.shrink_perturb {
        if !(l > 0.0 && l <= 1.0) || !(s >= 0.0) {
            return Err(format!("--shrink-perturb {l},{s}: lambda in (0, 1], sigma 0 or more"));
        }
    }
    if a.sparse_ft == Some(0) {
        return Err("--sparse-ft 0: give how many value rows to train".into());
    }
    if a.sparse_ft.is_some() && (a.muon.is_some() || a.load_opt.is_some()) {
        return Err("--sparse-ft trains the chosen value rows with a fresh AdamW of their own (no --opt muon, no --load-opt)".into());
    }
    Ok(a)
}

/// v42 (A1): the LR at `step` of `total` for a phase whose peak is `peak`: the chosen schedule (cosine by default, the
/// old cosine_lr exactly) down to --lr-min, or peak · --min-lr-frac as before.
fn lr_at(a: &Args, step: usize, warmup: usize, total: usize, peak: f32) -> f32 {
    a.schedule.lr(step, warmup, total, peak, a.lr_min.unwrap_or(peak * a.min_lr_frac))
}

/// v42: the settings that change a grow run's trajectory, added to its resume signature only when set (so a v41
/// checkpoint still resumes with a v41 command line). --fast-step is left out: it gives the same weights.
fn v42_signature(a: &Args) -> String {
    let mut s = String::new();
    if a.decay_gates {
        s += " decay_gates=1";
    }
    if a.beta2 != 0.95 {
        s += &format!(" beta2={}", a.beta2);
    }
    if a.late_qa_frac > 0.0 {
        s += &format!(" late_qa_frac={}", a.late_qa_frac);
    }
    if let Some(d) = a.ema {
        s += &format!(" ema={d}");
    }
    if a.skip_bad_batch {
        s += &format!(" skip_bad={}/{}", a.bad_norm_mult, a.bad_norm_window);
    }
    if a.wd_target_w0 {
        s += " wd_target=w0";
    }
    if let Some(p) = &a.load_opt {
        s += &format!(" load_opt={}", p.display());
    }
    s + &recipe_signature(a)
}

/// v42 Wave 2B: the recipe settings that are set, as " key=value ..." (empty when none is).
fn recipe_signature(a: &Args) -> String {
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
    s
}

fn run() -> Result<(), String> {
    let a = parse()?;
    let dev = ava_gpu::device(false).map_err(|e| format!("device: {e}"))?;
    if !dev.is_cuda() {
        return Err("grow_gpu needs CUDA (the weighted and signed losses are in the CUDA step)".into());
    }
    if a.out_model.is_some() {
        set_aside_result(&a.out)?; // K6: every learner call trains
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
    }?;
    if let Some(m) = &a.out_model {
        deliver_model(&a.out.join("model.safetensors"), m)?;
        deliver_arch(&a.out, m)?;
    }
    Ok(())
}

/// K6 --out-model: the run folder used when --out is not given (FILE + ".run").
fn run_dir_for(model: &Path) -> PathBuf {
    let mut s = model.as_os_str().to_owned();
    s.push(".run");
    PathBuf::from(s)
}

/// K6 --out-model: a result.tsv an earlier run left in the run folder is renamed result.prev.tsv, so this run trains.
fn set_aside_result(out: &Path) -> Result<(), String> {
    let res = out.join("result.tsv");
    if res.exists() {
        let prev = out.join("result.prev.tsv");
        fs::rename(&res, &prev).map_err(|e| format!("rename {} to {}: {e}", res.display(), prev.display()))?;
        println!("--out-model: an earlier run's result.tsv is now {}; this run trains", prev.display());
    }
    Ok(())
}

/// K6 --out-model: the phase's model copied to `to` through `to`.tmp and a rename (so `to` is whole or absent).
fn deliver_model(from: &Path, to: &Path) -> Result<(), String> {
    if let Some(d) = to.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(d).map_err(|e| format!("create {}: {e}", d.display()))?;
    }
    let mut tmp = to.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    fs::copy(from, &tmp).map_err(|e| format!("copy {} to {}: {e}", from.display(), tmp.display()))?;
    fs::rename(&tmp, to).map_err(|e| format!("rename {} to {}: {e}", tmp.display(), to.display()))?;
    println!("--out-model: the model is at {}", to.display());
    Ok(())
}

/// v42 Wave 2C: DIR/arch.txt holds the run's block options, so the model and the checkpoints in DIR load with them
/// (ckpt::load_auto reads it); a run with today's block removes a stale one.
fn write_run_arch(out: &Path, ar: &ava_gpu::arch::Arch) -> Result<(), String> {
    fs::create_dir_all(out).map_err(|e| format!("create {}: {e}", out.display()))?;
    let p = out.join("arch.txt");
    let f = ar.flags();
    if f.is_empty() {
        if p.exists() {
            fs::remove_file(&p).map_err(|e| format!("remove stale {}: {e}", p.display()))?;
        }
        return Ok(());
    }
    fs::write(&p, f + "\n").map_err(|e| format!("{}: {e}", p.display()))
}

/// v42 Wave 2C, --out-model FILE: the run's block options go beside FILE as FILE's NAME.arch.txt (a stale one goes).
fn deliver_arch(out: &Path, to: &Path) -> Result<(), String> {
    let own = ava_gpu::ckpt::arch_name(to);
    let run = out.join("arch.txt");
    if run.is_file() {
        fs::copy(&run, &own).map_err(|e| format!("copy {} to {}: {e}", run.display(), own.display()))?;
        println!("--out-model: its block options are in {}", own.display());
    } else if own.exists() {
        fs::remove_file(&own).map_err(|e| format!("remove stale {}: {e}", own.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod out_model_tests {
    use super::{deliver_model, run_dir_for, set_aside_result};
    use std::path::{Path, PathBuf};

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("grow_gpu_scratch_{}", std::process::id())).join(name);
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_run_folder_sits_beside_the_model() {
        assert_eq!(run_dir_for(Path::new("work/candidate.safetensors")), PathBuf::from("work/candidate.safetensors.run"));
    }

    #[test]
    fn an_old_result_is_set_aside_and_the_model_delivered_whole() {
        let d = scratch("deliver");
        let run = d.join("candidate.safetensors.run");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(run.join("result.tsv"), "old\n").unwrap();
        set_aside_result(&run).unwrap();
        assert!(!run.join("result.tsv").exists());
        assert_eq!(std::fs::read_to_string(run.join("result.prev.tsv")).unwrap(), "old\n");
        set_aside_result(&run).unwrap(); // nothing to do: fine
        std::fs::write(run.join("model.safetensors"), b"weights").unwrap();
        let out = d.join("work").join("candidate.safetensors");
        deliver_model(&run.join("model.safetensors"), &out).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"weights");
        assert!(!d.join("work").join("candidate.safetensors.tmp").exists());
        // a phase that wrote no model delivers nothing
        let gone = d.join("work").join("other.safetensors");
        assert!(deliver_model(&run.join("missing.safetensors"), &gone).is_err());
        assert!(!gone.exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// v42 Wave 2C: the run's block options go to DIR/arch.txt and beside the delivered model; today's block removes
    /// stale ones from both places.
    #[test]
    fn block_options_travel_with_the_model() {
        use super::{deliver_arch, write_run_arch};
        let d = scratch("arch");
        let run = d.join("m.safetensors.run");
        let to = d.join("m.safetensors");
        let ar = ava_gpu::arch::Arch::from_flags("--pk-mem 16 --dendrites 2").unwrap();
        write_run_arch(&run, &ar).unwrap();
        deliver_arch(&run, &to).unwrap();
        let back = ava_gpu::arch::Arch::from_flags(std::fs::read_to_string(d.join("m.arch.txt")).unwrap().trim()).unwrap();
        assert_eq!(back.signature(), ar.signature());
        write_run_arch(&run, &ava_gpu::arch::Arch::default()).unwrap();
        assert!(!run.join("arch.txt").exists());
        deliver_arch(&run, &to).unwrap();
        assert!(!d.join("m.arch.txt").exists());
        let _ = std::fs::remove_dir_all(&d);
    }
}

// ------------------------------------------------------------------------------------------------ shared parts

/// `gates`: --decay-gates (W1): the memory's output gate .wg decays too.
fn params_of(varmap: &VarMap, gates: bool) -> Vec<Param> {
    let mut names: Vec<(String, candle_core::Var)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    names.sort_by(|x, y| x.0.cmp(&y.0));
    names.into_iter().map(|(name, var)| Param { decay: Model::decays_with(&name, gates), name, var }).collect()
}

/// A fresh AdamW over every weight, with --beta2 (Q1), --decay-gates (W1) and --load-opt (U1).
fn new_opt(a: &Args, varmap: &VarMap, dev: &Device) -> Result<AdamW, String> {
    let mut opt = AdamW::new(params_of(varmap, a.decay_gates), a.wd as f64).map_err(|e| format!("adamw: {e}"))?;
    opt.beta2 = a.beta2;
    if a.decay_gates {
        let n = opt.params().iter().filter(|p| p.decay && p.name.ends_with(".wg")).count();
        println!("--decay-gates: weight decay on {n} output-gate matrices too");
    }
    if a.beta2 != 0.95 {
        println!("AdamW beta2 {}", a.beta2);
    }
    if let Some(m) = a.muon {
        // v42 (A1): Muon for the decayed hidden matrices; its momentum lives in their m buffers
        let n = opt.enable_muon(MuonCfg::standard(m, a.lr as f64));
        println!("--opt muon: {n} hidden matrices step by Muon (peak LR {m}), the rest by AdamW");
    }
    if let Some(p) = &a.load_opt {
        let map = candle_core::safetensors::load(p, dev).map_err(|e| format!("--load-opt {}: {e}", p.display()))?;
        let t = match map.get("meta.t") {
            Some(x) => x.to_dtype(DType::F64).and_then(|x| x.flatten_all()?.to_vec1::<f64>()).map_err(|e| e.to_string())?.first().copied().unwrap_or(0.0) as u64,
            None => {
                // a grow checkpoint's optim-STEP file: its step count is in state.txt beside it
                let st = p.with_file_name("state.txt");
                let text = fs::read_to_string(&st).map_err(|_| format!("--load-opt {}: no meta.t in the file and no {} beside it", p.display(), st.display()))?;
                text.lines().find_map(|l| l.strip_prefix("t=")).and_then(|x| x.trim().parse().ok()).ok_or(format!("{}: no t= line", st.display()))?
            }
        };
        let (got, miss) = opt.load_state_partial(&map, t).map_err(|e| format!("--load-opt {}: {e}", p.display()))?;
        println!("--load-opt {}: moments of {got} parameters restored ({miss} start at 0), step count {t}", p.display());
    }
    Ok(opt)
}

/// --save-opt (U1): the moments and the step count as DIR/optim.safetensors (what --load-opt reads).
fn save_opt(a: &Args, opt: &AdamW, dev: &Device) -> Result<(), String> {
    if !a.save_opt {
        return Ok(());
    }
    let mut st = opt.state();
    st.insert("meta.t".into(), Tensor::new(&[opt.t as f64], dev).map_err(|e| e.to_string())?);
    let p = a.out.join("optim.safetensors");
    candle_core::safetensors::save(&st, &p).map_err(|e| format!("save {}: {e}", p.display()))?;
    println!("saved the optimizer moments (step count {}) to {}", opt.t, p.display());
    Ok(())
}

/// X2: a copy of the settings with the night read from --replay-cursor (without a cursor: None, the old windows).
fn with_night(a: &Args) -> Result<Args, String> {
    let mut b = a.clone();
    if let Some(p) = &a.replay_cursor {
        let n = trainkit::read_night(p)?;
        println!("replay cursor {}: night {n}", p.display());
        b.replay_night = Some(n);
    }
    Ok(b)
}

/// X2: after the phase is done (result.tsv written), the replay cursor moves on to the next night. A phase that fails
/// before then leaves it where it was, so its rerun replays the same windows.
fn advance_cursor(a: &Args) -> Result<(), String> {
    if let (Some(p), Some(n)) = (&a.replay_cursor, a.replay_night) {
        trainkit::write_night(p, n + 1)?;
        println!("replay cursor {}: next night {}", p.display(), n + 1);
    }
    Ok(())
}

/// v42: what a phase writes besides the model, before result.tsv: the moments (--save-opt), the EMA copy (--ema) and
/// the anchor (--anchor-save).
fn save_extras(a: &Args, opt: &AdamW, upd: &Updater, anchor: Option<&Anchor>, dev: &Device) -> Result<(), String> {
    save_opt(a, opt, dev)?;
    if let Some(e) = &upd.ema {
        let p = trainkit::ema_path(&a.out.join("model.safetensors"));
        e.save(&p).map_err(|e| format!("save {}: {e}", p.display()))?;
        println!("saved the EMA copy (decay {}) to {}", e.decay, p.display());
    }
    if let (Some(p), Some(an)) = (&a.anchor_save, anchor) {
        an.save(p).map_err(|e| format!("--anchor-save {}: {e}", p.display()))?;
        println!("saved the anchor ({} weights) to {}", an.weights, p.display());
    }
    Ok(())
}

/// U1 (--wd-target w0): decay toward w0, per decayed parameter: the anchor's w0 when there is an anchor (so
/// --anchor-point saved makes it the saved weights), else the weights now.
fn decay_target(a: &Args, opt: &AdamW, anchor: Option<&Anchor>) -> Result<Option<DecayToward>, String> {
    if !a.wd_target_w0 {
        return Ok(None);
    }
    let mut pairs = Vec::new();
    for p in opt.params().iter().filter(|p| p.decay) {
        let w0 = match anchor.and_then(|an| an.w0_of(&p.name)) {
            Some(w) => w.clone(),
            None => p.var.as_tensor().copy().map_err(|e| e.to_string())?,
        };
        pairs.push((p.var.clone(), w0));
    }
    println!("--wd-target w0: weight decay pulls {} matrices toward w0 instead of 0", pairs.len());
    Ok(Some(DecayToward::new(pairs)))
}

/// v42: everything around one optimizer step, the same in every phase: the bad-batch guard (J3, before the step),
/// the step (per parameter as in v41, or with --fast-step the fused step, Q6), then the decay toward w0 (U1), the
/// anchor's pull and the EMA copy (Q4). With every v42 flag off this is v41's grad_norm, step and pull exactly.
struct Updater {
    clip: f64,
    wd: f64,
    fast: bool,
    guard: Option<BadBatch>,
    skip_log: Option<fs::File>,
    decay_to: Option<DecayToward>,
    ema: Option<Ema>,
    /// v42 (I4): Grokfast-EMA on the gradients (off unless --grokfast-lambda > 0)
    gf: Option<Grokfast>,
}

/// After this many skipped steps in a row the run stops: something is broken, not one bad batch.
const MAX_BAD_IN_A_ROW: usize = 20;

impl Updater {
    fn new(a: &Args, varmap: &VarMap, decay_to: Option<DecayToward>) -> Result<Updater, String> {
        let (guard, skip_log) = if a.skip_bad_batch {
            let mut f = fs::File::create(a.out.join("skipped.tsv")).map_err(|e| format!("skipped.tsv: {e}"))?;
            writeln!(f, "step\treason\tgrad_norm\tmedian\tlosses").map_err(|e| e.to_string())?;
            println!("--skip-bad-batch: a step is skipped when a loss or the grad norm is not finite or the norm is over {} x the median of the last {}", a.bad_norm_mult, a.bad_norm_window);
            (Some(BadBatch::new(a.bad_norm_mult, a.bad_norm_window)), Some(f))
        } else {
            (None, None)
        };
        let ema = match a.ema {
            Some(d) => {
                println!("--ema {d}: an EMA copy of the weights is kept");
                Some(Ema::new(varmap, d).map_err(|e| format!("ema: {e}"))?)
            }
            None => None,
        };
        let gf = (a.gf_lambda > 0.0).then(|| {
            println!("--grokfast-lambda {}: Grokfast-EMA (alpha {}) on every gradient before clipping", a.gf_lambda, a.gf_alpha);
            Grokfast::new(a.gf_alpha, a.gf_lambda)
        });
        Ok(Updater { clip: a.clip as f64, wd: a.wd as f64, fast: a.fast_step, guard, skip_log, decay_to, ema, gf })
    }

    /// v42 (I4): add the slow average to every gradient (nothing when Grokfast is off).
    fn grokfast(&mut self, opt: &AdamW, grads: &mut GradStore, step: usize) -> Result<(), String> {
        match self.gf.as_mut() {
            Some(g) => g.apply(opt.params(), grads).map_err(|e| format!("step {step} grokfast: {e}")),
            None => Ok(()),
        }
    }

    /// The step: returns the gradient norm (before clipping) and whether the update was made. `losses`: the step's
    /// losses on the host (the guard checks them; without the guard they are not used). With Grokfast the norm is the
    /// filtered gradient's; with the guard too, the guard judges the raw gradient first, so a bad batch never enters
    /// the averages.
    fn step(&mut self, opt: &mut AdamW, grads: &mut GradStore, lr: f64, losses: &[f32], anchor: Option<&Anchor>, step: usize) -> Result<(f64, bool), String> {
        if self.guard.is_none() {
            self.grokfast(opt, grads, step)?;
        }
        let norm_t = if self.fast { opt.grad_norm_t(grads).map_err(|e| format!("step {step} grad norm: {e}"))? } else { None };
        let gn = match &norm_t {
            Some(t) => t.to_scalar::<f64>().map_err(|e| format!("step {step} grad norm: {e}"))?,
            None => opt.grad_norm(grads).map_err(|e| format!("step {step} grad norm: {e}"))?,
        };
        if let Some(g) = self.guard.as_mut() {
            let v = g.judge(losses, gn);
            if v != Verdict::Good {
                let (why, med) = match v {
                    Verdict::NonFinite => ("not_finite", f64::NAN),
                    Verdict::Spike(m) => ("norm_spike", m),
                    Verdict::Good => unreachable!(),
                };
                let ls: Vec<String> = losses.iter().map(|l| format!("{l:.6}")).collect();
                if let Some(f) = self.skip_log.as_mut() {
                    writeln!(f, "{step}\t{why}\t{gn:.6}\t{med:.6}\t{}", ls.join(",")).map_err(|e| e.to_string())?;
                }
                println!("step {step}: skipped ({why}: grad norm {gn:.4}, median {med:.4}, losses {})", ls.join(","));
                if g.in_a_row >= MAX_BAD_IN_A_ROW {
                    return Err(format!("step {step}: {MAX_BAD_IN_A_ROW} bad batches in a row; stopping"));
                }
                return Ok((gn, false));
            }
        }
        if !gn.is_finite() {
            // v52 fix_S (a defense, not the fix): a non-finite gradient never reaches the weights or Adam's moments
            println!("step {step}: update skipped (gradient norm {gn})");
            return Ok((gn, false));
        }
        let (norm_t, gn) = if self.guard.is_some() && self.gf.is_some() {
            // the batch passed the guard: now the filter, and the clip on the filtered gradient's norm
            self.grokfast(opt, grads, step)?;
            let nt = if self.fast { opt.grad_norm_t(grads).map_err(|e| format!("step {step} grad norm: {e}"))? } else { None };
            let g2 = match &nt {
                Some(t) => t.to_scalar::<f64>().map_err(|e| format!("step {step} grad norm: {e}"))?,
                None => opt.grad_norm(grads).map_err(|e| format!("step {step} grad norm: {e}"))?,
            };
            (nt, g2)
        } else {
            (norm_t, gn)
        };
        match &norm_t {
            Some(t) => {
                let sc = clip_scale_t(t, self.clip).map_err(|e| format!("step {step} clip: {e}"))?;
                opt.step_t(grads, lr, &sc).map_err(|e| format!("step {step} adamw: {e}"))?;
            }
            None => opt.step(grads, lr, clip_scale(gn, self.clip)).map_err(|e| format!("step {step} adamw: {e}"))?,
        }
        self.after(lr, anchor, step)?;
        Ok((gn, true))
    }

    /// What follows every update: decay toward w0, the anchor's pull, the EMA.
    fn after(&self, lr: f64, anchor: Option<&Anchor>, step: usize) -> Result<(), String> {
        if let Some(d) = &self.decay_to {
            d.apply(lr, self.wd).map_err(|e| format!("step {step} decay toward w0: {e}"))?;
        }
        if let Some(an) = anchor {
            an.pull(lr).map_err(|e| format!("step {step} anchor: {e}"))?;
        }
        if let Some(e) = &self.ema {
            e.update().map_err(|e| format!("step {step} ema: {e}"))?;
        }
        Ok(())
    }

    fn skipped(&self) -> usize {
        self.guard.as_ref().map_or(0, |g| g.skipped)
    }
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
        // v42 (X2): with --replay-cursor each night has a stream of its own (night 0 and no cursor: the v41 seed)
        Ok(Some(Replay { toks, ctx: a.replay_ctx, rows, rng: SplitMix64::new(trainkit::replay_seed(a.seed, stream, a.replay_night)) }))
    }

    /// Fresh random windows as (ids, targets) [rows, ctx].
    fn batch(&mut self, dev: &Device) -> Result<(Tensor, Tensor), String> {
        let (mut rid, mut rtg) = (Vec::with_capacity(self.rows * self.ctx), Vec::with_capacity(self.rows * self.ctx));
        for start in trainkit::window_starts(&mut self.rng, self.rows, self.toks.len(), self.ctx) {
            let win = self.toks.window(start, self.ctx + 1);
            rid.extend_from_slice(&win[..self.ctx]);
            rtg.extend_from_slice(&win[1..]);
        }
        let rid = Tensor::from_vec(rid, (self.rows, self.ctx), dev).map_err(|e| e.to_string())?;
        let rtg = Tensor::from_vec(rtg, (self.rows, self.ctx), dev).map_err(|e| e.to_string())?;
        Ok((rid, rtg))
    }

    /// The reading loss and its gradient on given windows.
    fn grads_on(&self, model: &Model, mm: &Mm, rid: &Tensor, rtg: &Tensor) -> Result<(f32, GradStore, usize), String> {
        let (l, _, g) = model.loss_and_grads_a_w(rid, rtg, None, mm).map_err(|e| format!("replay: {e}"))?;
        Ok((l.to_scalar::<f32>().map_err(|e| e.to_string())?, g, self.rows * self.ctx))
    }

    /// The reading loss and its gradient on fresh random windows.
    fn grads(&mut self, model: &Model, mm: &Mm, dev: &Device) -> Result<(f32, GradStore, usize), String> {
        let (rid, rtg) = self.batch(dev)?;
        self.grads_on(model, mm, &rid, &rtg)
    }
}

/// U2: the frozen reference model (--kl-ref, the serving copy), its fixed windows for the forgetting gate, and the
/// KL log (DIR/kl.tsv).
struct KlRef {
    model: Model,
    _varmap: VarMap,
    eval_ids: Tensor,
    log: fs::File,
    /// the per-step KL on the replay windows since the last measure (sum, count)
    sum: f64,
    n: usize,
    last: f64,
}

impl KlRef {
    /// `stream`: the phase's replay stream; the fixed windows are stream 20 + it. `engram_off`: the lesion the
    /// current model reads her usual text with in this phase (the reference reads with the same).
    fn open(a: &Args, dev: &Device, stream: u64, vocab: usize, engram_off: bool) -> Result<Option<KlRef>, String> {
        let Some(p) = &a.kl_ref else { return Ok(None) };
        let (cfg, varmap, model) = ava_gpu::ckpt::load_auto(p, a.kind, a.eval_ctx, dev)?;
        if cfg.vocab != vocab {
            return Err(format!("--kl-ref {}: vocab {}, the model {vocab}", p.display(), cfg.vocab));
        }
        model.set_track_activity(false);
        model.set_inference(true);
        model.set_lesion(false, false, engram_off);
        let mut r = Replay::open_rows(a, 20 + stream, a.replay_rows.max(4))?.ok_or("--kl-ref needs replay data (--train)")?;
        let (eval_ids, _) = r.batch(dev)?;
        let mut log = fs::File::create(a.out.join("kl.tsv")).map_err(|e| format!("kl.tsv: {e}"))?;
        writeln!(log, "step\tcycle\tkl_fixed_windows\tkl_replay_mean").map_err(|e| e.to_string())?;
        println!("--kl-ref {}: weight {} on the replay windows; the forgetting gate on {} fixed windows of {}", p.display(), a.kl_weight, r.rows, r.ctx);
        Ok(Some(KlRef { model, _varmap: varmap, eval_ids, log, sum: 0.0, n: 0, last: f64::NAN }))
    }

    /// KL(reference || current) on `ids` and its gradient for the current model.
    fn grads(&mut self, model: &Model, ids: &Tensor) -> Result<(f32, GradStore), String> {
        let r = self.model.forward(ids).map_err(|e| format!("kl reference: {e}"))?.detach();
        let (k, g) = trainkit::kl_grads(model, &r, ids).map_err(|e| format!("kl: {e}"))?;
        self.sum += k as f64;
        self.n += 1;
        Ok((k, g))
    }

    /// The forgetting gate: KL on the fixed windows now (both models in inference mode), logged with the mean KL of
    /// the replay windows since the last measure.
    fn measure(&mut self, model: &Model, step: usize, cycle: usize) -> Result<f64, String> {
        model.set_inference(true);
        let cur = model.forward(&self.eval_ids).map(|t| t.detach());
        model.set_inference(false);
        let cur = cur.map_err(|e| format!("kl measure: {e}"))?;
        let r = self.model.forward(&self.eval_ids).map_err(|e| format!("kl measure (reference): {e}"))?.detach();
        let kl = trainkit::kl_ref_cur(&r, &cur).and_then(|t| t.to_scalar::<f32>()).map_err(|e| format!("kl measure: {e}"))? as f64;
        let mean = if self.n > 0 { self.sum / self.n as f64 } else { f64::NAN };
        writeln!(self.log, "{step}\t{cycle}\t{kl:.6e}\t{mean:.6e}").map_err(|e| e.to_string())?;
        println!("forgetting gate at step {step}: KL(reference || now) {kl:.4e} on the fixed windows (replay windows since the last check: {mean:.4e})");
        (self.sum, self.n, self.last) = (0.0, 0, kl);
        Ok(kl)
    }
}

/// U5: a dev question file (--sleep-dev) and its log (DIR/dev.tsv): right − wrong and the abstain rate of her greedy
/// answers, with everything switched on (as she would answer when serving).
struct DevCheck {
    seqs: Vec<Vec<u32>>,
    from: Vec<usize>,
    bytes: Vec<f64>,
    /// per item: (first row, real choices, right, has "I don't know")
    items: Vec<(usize, usize, usize, bool)>,
    log: fs::File,
    last: Tally,
}

impl DevCheck {
    fn open(a: &Args, bpe: &Bpe, glance: bool, vocab: usize) -> Result<Option<DevCheck>, String> {
        let Some(p) = &a.sleep_dev else { return Ok(None) };
        let items = mc::read_items(p)?;
        if items.is_empty() {
            return Err(format!("--sleep-dev {}: no items", p.display()));
        }
        let byte_len = byte_lengths_n(bpe, vocab);
        let (mut seqs, mut from, mut meta) = (Vec::new(), Vec::new(), Vec::new());
        for it in &items {
            let x = if a.idk_action { trainkit::with_idk(it) } else { it.clone() };
            let (s, f) = mc::item_seqs(bpe, &x, glance);
            meta.push((seqs.len(), it.choices.len(), it.right, a.idk_action));
            seqs.extend(s);
            from.extend(f);
        }
        let bytes = mc::scored_bytes(&byte_len, &seqs, &from);
        let mut log = fs::File::create(a.out.join("dev.tsv")).map_err(|e| format!("dev.tsv: {e}"))?;
        writeln!(log, "step\tcycle\titems\tright\twrong\tabstained\tright_minus_wrong\tabstain_rate\tshare_right").map_err(|e| e.to_string())?;
        println!("--sleep-dev {}: {} items (tau {}, \"I don't know\" {})", p.display(), items.len(), a.tau, a.idk_action);
        Ok(Some(DevCheck { seqs, from, bytes, items: meta, log, last: Tally::default() }))
    }

    /// Score every item now (everything on; the caller restores its lesion) and log one line.
    fn check(&mut self, a: &Args, model: &Model, dev: &Device, step: usize, cycle: usize) -> Result<Tally, String> {
        model.set_lesion(false, false, false);
        model.set_inference(true);
        let nll = mc::scored_nll(model, dev, &self.seqs, &self.from, a.score_batch).map_err(|e| format!("dev scoring: {e}"));
        model.set_inference(false);
        let nll = nll?;
        let mut t = Tally::default();
        for &(s, nc, right, idk) in &self.items {
            let na = nc + idk as usize;
            let score = mc::normed(&nll[s..s + na], &self.bytes[s..s + na], a.norm);
            t.add(trainkit::judge(&score, right, a.tau, idk.then_some(nc)));
        }
        writeln!(
            self.log,
            "{step}\t{cycle}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}",
            t.n(),
            t.right,
            t.wrong,
            t.abstain,
            t.right_minus_wrong(),
            t.abstain_rate(),
            t.share_right()
        )
        .map_err(|e| e.to_string())?;
        println!(
            "dev at step {step}: right {} wrong {} abstained {} of {}; right - wrong {:+.3}, abstain rate {:.3}",
            t.right,
            t.wrong,
            t.abstain,
            t.n(),
            t.right_minus_wrong(),
            t.abstain_rate()
        );
        self.last = t;
        Ok(t)
    }
}

/// U3 (--anchor-episodes): the sequences to protect, as (tokens, first scored target): episode text (blocks
/// separated by blank lines, each [EOS] text [EOS], all scored) or AVASFT01 answers (only the answer scored).
fn study_seqs(a: &Args, bpe: &Bpe, glance: bool) -> Result<Vec<(Vec<u32>, usize)>, String> {
    let mut out = Vec::new();
    for p in &a.anchor_episodes {
        let bytes = fs::read(p).map_err(|e| format!("--anchor-episodes {}: {e}", p.display()))?;
        if bytes.starts_with(b"AVASFT01") {
            out.extend(marked(read_sft(p)?, glance).into_iter().map(|e| (e.toks, e.np)));
        } else {
            for e in episodes(&String::from_utf8_lossy(&bytes)) {
                let t = bpe.encode(&e);
                let mut s = vec![EOS];
                s.extend(if glance { insert_marks(&t) } else { t });
                s.push(EOS);
                out.push((s, 1));
            }
        }
    }
    Ok(out)
}

/// v41 synaptic consolidation (ava_gpu::anchor): before sleep or reward, measure each weight's importance to her usual
/// reading on --fisher-batches fresh replay batches (`stream`: a replay stream of its own, 10 + the phase's, so these
/// are not the replay batches); after every optimizer step the anchor pulls weights back by that importance.
/// `--anchor 0` switches it off.
/// v42 (U3): with --anchor-episodes the importance also counts up to --fisher-batches batches of those sequences
/// (engram off: the facts must be held by her weights; `engram_off` is the caller's lesion, restored after), and
/// with --anchor-load it is combined with a saved anchor (F = γ·F_old + F_new).
#[allow(clippy::too_many_arguments)]
fn measure_anchor(a: &Args, model: &Model, opt: &AdamW, mm: &Mm, dev: &Device, stream: u64, bpe: &Bpe, glance: bool, engram_off: bool) -> Result<Option<Anchor>, String> {
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
    if !a.anchor_episodes.is_empty() {
        let seqs = study_seqs(a, bpe, glance)?;
        if seqs.is_empty() {
            return Err("--anchor-episodes: no sequences in those files".into());
        }
        // a seeded shuffle, then batches of at most replay rows · replay ctx tokens (rows · longest row)
        let mut order: Vec<usize> = (0..seqs.len()).collect();
        SplitMix64::new(derive_seed(a.seed, 0x7374_7564_79, stream)).shuffle(&mut order);
        let budget = a.replay_rows.max(4) * a.replay_ctx;
        let (mut batches, mut cur, mut longest) = (Vec::new(), Vec::new(), 0usize);
        for i in order {
            let l = seqs[i].0.len();
            if !cur.is_empty() && (cur.len() + 1) * longest.max(l) > budget {
                batches.push(std::mem::take(&mut cur));
                longest = 0;
            }
            longest = longest.max(l);
            cur.push(i);
        }
        if !cur.is_empty() {
            batches.push(cur);
        }
        model.set_lesion(false, false, true);
        let before = imp.batches();
        for batch in batches.iter().take(a.fisher_batches) {
            let rows: Vec<&[u32]> = batch.iter().map(|&i| seqs[i].0.as_slice()).collect();
            let from: Vec<usize> = batch.iter().map(|&i| seqs[i].1).collect();
            let (ids, tg, w, t) = padded(&rows, &from);
            let b = rows.len();
            let ids = Tensor::from_vec(ids, (b, t), dev).map_err(|e| e.to_string())?;
            let tg = Tensor::from_vec(tg, (b, t), dev).map_err(|e| e.to_string())?;
            let w = Tensor::from_vec(w, b * t, dev).map_err(|e| e.to_string())?;
            let (_, _, g) = model.loss_and_grads_a_w(&ids, &tg, Some(&w), mm).map_err(|e| format!("anchor (episodes): {e}"))?;
            imp.add(&g).map_err(|e| format!("anchor importance: {e}"))?;
        }
        model.set_lesion(false, false, engram_off);
        println!("anchor: + importance from {} batches of {} studied sequences ({} files, engram off)", imp.batches() - before, seqs.len(), a.anchor_episodes.len());
    }
    let mut an = imp.anchor(a.anchor).map_err(|e| format!("anchor: {e}"))?.ok_or("anchor: her reading gave no gradient")?;
    println!("anchor {}: importance from {} replay batches over {} weights, {:.1}s", a.anchor, a.fisher_batches, an.weights, t0.elapsed().as_secs_f64());
    if let Some(p) = &a.anchor_load {
        let old = Anchor::read(p, dev).map_err(|e| format!("--anchor-load {}: {e}", p.display()))?;
        an = an.combine(&old, a.anchor_gamma, a.anchor_saved_point, opt.params()).map_err(|e| format!("--anchor-load {}: {e}", p.display()))?;
        println!(
            "anchor: combined with {} (gamma {}, its strength {}): mean importance {:.4} over {} weights; pulled back to the {} weights",
            p.display(),
            a.anchor_gamma,
            old.strength,
            an.mean_importance().map_err(|e| e.to_string())?,
            an.weights,
            if a.anchor_saved_point { "saved" } else { "current" }
        );
    }
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
        if a.fused_eval {
            // v52 fasteval: this eval only (the sleep and reward phases' own scoring keeps the reference forward)
            println!("val: {}", ava_gpu::eval::fused_line(model, dev));
            ava_gpu::eval::set_fused(true);
        }
        let r = evaluate(model, set.iter_sequential(a.eval_ctx), 16, a.eval_tokens, &byte_len, dev).map_err(|e| format!("eval: {e}"));
        ava_gpu::eval::set_fused(false);
        let (acc, _) = r?;
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
    ) + &v42_signature(a)
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

/// `ema`: v42 (--ema), its copy saved beside the weights as model-STEP.ema.safetensors.
#[allow(clippy::too_many_arguments)]
fn save_checkpoint(dir: &Path, step: usize, varmap: &VarMap, opt: &AdamW, sampler: &Sampler, long: Option<&Sampler>, sig: &str, secs: f64, tok: u64, ema: Option<&Ema>, gf: Option<&Grokfast>, keep_models: bool) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    varmap.save(dir.join(format!("model-{step}.safetensors"))).map_err(|e| format!("checkpoint weights: {e}"))?;
    // v42 (I4): Grokfast's averages ride in the optim file (gf.<param>)
    let mut st = opt.state();
    if let Some(g) = gf {
        st.extend(g.state());
    }
    candle_core::safetensors::save(&st, dir.join(format!("optim-{step}.safetensors"))).map_err(|e| format!("checkpoint moments: {e}"))?;
    if let Some(e) = ema {
        e.save(&dir.join(format!("model-{step}.ema.safetensors"))).map_err(|e| format!("checkpoint ema: {e}"))?;
    }
    let tmp = dir.join("state.tmp");
    let long_line = long.map(|s| format!("cursor_long={}\n", s.cursor().to_line())).unwrap_or_default();
    fs::write(&tmp, format!("step={step}\nt={}\ncursor={}\n{long_line}secs={secs:.3}\ntok={tok}\nargs={sig}\n", opt.t, sampler.cursor().to_line())).map_err(|e| format!("state.tmp: {e}"))?;
    fs::rename(&tmp, dir.join("state.txt")).map_err(|e| format!("state.txt: {e}"))?;
    let keep = [format!("model-{step}.safetensors"), format!("optim-{step}.safetensors"), format!("model-{step}.ema.safetensors")];
    for e in fs::read_dir(dir).map_err(|e| e.to_string())?.flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        // v52 --keep-ckpts (`keep_models`): the older weights model-STEP.safetensors stay; optim and EMA files still go
        let old_weights = n.starts_with("model-") && n.ends_with(".safetensors") && !n.ends_with(".ema.safetensors");
        if (n.starts_with("model-") || n.starts_with("optim-")) && !keep.contains(&n) && !(keep_models && old_weights) {
            let _ = fs::remove_file(e.path());
        }
    }
    Ok(())
}

/// Q6 (--fast-step): one grow step whose loss, answer loss and gradient norm are still on the GPU.
struct Pending {
    step: usize,
    lr: f32,
    loss: Tensor,
    /// the answer loss: on the GPU, or already on the host (NaN: no answers this step; 0: this rank had none)
    qa: Option<Tensor>,
    qa_host: f32,
    gn: Tensor,
    tok: usize,
    ctx: usize,
    recall: usize,
}

fn phase_grow(a: &Args, dev: &Device, bpe: &Bpe) -> Result<(), String> {
    let train = a.train.as_ref().ok_or("--phase grow needs --train")?;
    if a.tokens == 0 {
        return Err("--phase grow needs --tokens".into());
    }
    // v39's shape from the file, plus the new parts
    let t0 = ava_gpu::ckpt::read(&a.init)?;
    let mut cfg = ava_gpu::ckpt::detect(&t0, a.kind, a.ctx)?;
    // v42 Wave 2C: the block options: the flags' when given (the init's parts copy over, the new ones start as
    // no-ops), else the init's own (its arch.txt); an order head in the file is kept
    let init_arch = ava_gpu::ckpt::arch_beside(&a.init)?;
    let mut ar = if !a.arch.is_default() { a.arch } else { init_arch.as_ref().map(|x| x.0).unwrap_or_default() };
    ar.top = t0.contains_key("top_emb");
    drop(t0);
    if a.mag {
        cfg = cfg.with_mag()?;
    }
    if let Some(l) = a.engram {
        cfg = cfg.with_engram(l)?;
    }
    if !ar.is_default() {
        cfg = cfg.with_arch(ar).map_err(|e| format!("block options: {e}"))?;
        let from = match (&init_arch, a.arch.is_default()) {
            (Some((_, p)), true) => format!(" (from {})", p.display()),
            _ => String::new(),
        };
        // v51 fused N: which training step the block options take (the hand-written WIN step or candle's autograd)
        let path = if ava_gpu::model::fwin::would_take_on(&cfg, dev) { "the fused WIN step" } else { "candle's autograd" };
        println!("v42 block options:{}{from}; their steps run through {path}", ar.signature());
    }
    write_run_arch(&a.out, &cfg.arch)?;
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
    // run to the end of training rather than stopping early when steps / batches is not a whole number.
    // v42 (Q3, --late-qa-frac F): over the last F of the steps instead (trainkit::QaPlan; F = 0 is this rule exactly).
    let plan = QaPlan::new(steps, a.qa_epochs * nb, a.late_qa_frac);
    let qa_at = |s: usize| -> Option<usize> { plan.at(s) };
    let qa_every = plan.every();
    let qa_span = steps - plan.start;
    if a.qa_epochs * nb > qa_span {
        println!("note: {} answer batches but only {qa_span} steps; one per step, {} are left out", a.qa_epochs * nb, a.qa_epochs * nb - qa_span);
    }
    if a.late_qa_frac > 0.0 {
        println!("--late-qa-frac {}: the {} answer batches go in steps {}..{steps}", a.late_qa_frac, plan.total, plan.start);
    }
    let mut opt = new_opt(a, &varmap, dev)?;
    // U1 (--wd-target w0): the weights grow starts from (v39's, the new parts at their init), taken before a resume
    let decay_to = decay_target(a, &opt, None)?;
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
    // v52 fix_Pr: the pruning schedule over the full run (`steps`, the cosine schedule's own length, not --max-steps),
    // from the resumed step, as train_gpu does; the cut mask and the use averages came back with the checkpoint's
    // weights. Without it a grow run with --prune never cut a unit (v51::after_step returned at total 0).
    if let Some(plan) = model.v51_begin_schedule(start, steps) {
        println!("{plan}");
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
    // v42: the step's helpers (guard, fused step, decay toward w0, EMA); a resumed run takes its EMA copy back
    let mut upd = Updater::new(a, &varmap, decay_to)?;
    if let (Some(e), true) = (&upd.ema, start > 0) {
        let p = ckdir.join(format!("model-{start}.ema.safetensors"));
        if p.exists() {
            e.load(&p, dev).map_err(|e| format!("resume ema {}: {e}", p.display()))?;
        } else {
            println!("note: no EMA copy in the checkpoint; the EMA starts from the resumed weights");
        }
    }
    if let (Some(g), true) = (upd.gf.as_mut(), start > 0) {
        // v42 (I4): the Grokfast averages saved with the moments
        let st = candle_core::safetensors::load(ckdir.join(format!("optim-{start}.safetensors")), dev).map_err(|e| format!("resume grokfast: {e}"))?;
        let n = g.load_state(&st, dev).map_err(|e| format!("resume grokfast: {e}"))?;
        println!("grokfast: {n} averages restored");
    }
    // Q6 (--fast-step): one process and no guard: nothing waits for the GPU inside a step (see `Pending`)
    let deferred = a.fast_step && a.world == 1 && upd.guard.is_none();
    if a.fast_step {
        println!("--fast-step: fused optimizer step{}", if deferred { ", host copies only at shown steps and checkpoints" } else { "" });
    }
    let mut pend: Vec<Pending> = Vec::new();
    let mut t_block = Instant::now();
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
    let mut rested = 0.0f64;
    let mut sat_seen = 0u64; // v52 fix_S: mem_gen::SAT_EVENTS already reported
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
        // v42 (I2, --ctx-warmup): this step's context in the short phase (the long phase is never cut); off, it is --ctx
        let warm_c = if long { ctx_s } else { ctx_at(step as u64 * (a.batch * a.ctx) as u64, a.ctx_warmup, a.ctx_start, a.ctx) };
        // v41 recall practice: on every --recall-every-th step the --recall-rows windows with the fewest end-of-text
        // marks (the most book-like: one text, not many short ones; ties go to the later row) become recall rows (the
        // window's own text with made-up notes planted in it and asked about later; ava_data::recall). Every rank
        // builds the same rows (the seed is the step), then takes its share. v42: none while the context warms up (a
        // cut window would lose the note or the question).
        let mut n_recall = 0usize;
        if a.recall_every > 0 && step % a.recall_every == 0 && warm_c == ctx_s {
            let mut rng = SplitMix64::new(derive_seed(a.seed, 0x7265_6361_6c6c, step as u64));
            let mut idx: Vec<usize> = (0..windows.len()).collect();
            idx.sort_by_key(|&i| (windows[i].iter().filter(|&&t| t == EOS).count(), std::cmp::Reverse(i)));
            for &i in idx.iter().take(a.recall_rows.min(rows_s)) {
                windows[i] = ava_data::recall::recall_row(&mut rng, bpe, &windows[i], ctx_s);
                n_recall += 1;
            }
        }
        // v42 (I2): cut each window into pieces of the warm-up length (the same targets and tokens per step)
        let (ctx_s, rows_s) = if warm_c < ctx_s {
            windows = split_windows(&windows, warm_c);
            (warm_c, rows_s * (ctx_s / warm_c))
        } else {
            (ctx_s, rows_s)
        };
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
        let mut qa_loss_t: Option<Tensor> = None;
        let mut qa_done = false;
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
                    if deferred {
                        qa_loss_t = Some(ql); // Q6: read at the next block copy
                    } else {
                        qa_loss = ql.to_scalar::<f32>().map_err(|e| e.to_string())?;
                    }
                } else {
                    qa_loss = 0.0;
                }
                qa_done = true;
                step_tok += all_rows.iter().map(|&i| qa_ex[i].toks.len() - 1).sum::<usize>();
            }
        }
        if std::env::var("AVA_GRAD_TOP").is_ok_and(|v| v.split(',').any(|s| s.trim().parse::<usize>().ok() == Some(step))) {
            // v49 debug (as train_gpu's): the weights with the largest gradients at these steps (reading + lessons)
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
        {
            // v52 fix_S: say when the memory state bound held (a runaway head; rare: twice in 2300 steps of B-s52)
            let n = ava_gpu::mem_gen::SAT_EVENTS.load(std::sync::atomic::Ordering::Relaxed);
            if n != sat_seen {
                println!("step {step}: the memory state bound held in {} scan(s)", n - sat_seen);
                sat_seen = n;
            }
        }
        let lr = lr_at(a, step, a.warmup, steps, a.lr);
        let show = lead && (step % a.log_every == 0 || step + 1 == steps_run);
        let ckpt_now = lead && a.ckpt_every > 0 && (step + 1) % a.ckpt_every == 0 && step + 1 < steps_run;
        if deferred {
            // Q6: norm, clipping factor and the fused step stay on the GPU; this step's numbers wait in `pend`
            upd.grokfast(&opt, &mut grads, step)?; // v42 (I4); nothing when off
            let gn_t = opt.grad_norm_t(&grads).map_err(|e| format!("step {step} grad norm: {e}"))?.ok_or_else(|| format!("step {step}: no gradients"))?;
            let sc = clip_scale_t(&gn_t, a.clip as f64).map_err(|e| format!("step {step} clip: {e}"))?;
            opt.step_t(&grads, lr as f64, &sc).map_err(|e| format!("step {step} adamw: {e}"))?;
            upd.after(lr as f64, None, step)?;
            model.homeostasis_step().map_err(|e| format!("step {step} homeostasis: {e}"))?; // v42; nothing when off
            drop(grads);
            pend.push(Pending { step, lr, loss: loss_t, qa: qa_loss_t, qa_host: qa_loss, gn: gn_t, tok: step_tok, ctx: ctx_s, recall: n_recall });
            // v49: also drain at 32 waiting steps, as train_gpu does (v47 leak fix): on the autograd path the waiting
            // values held about 30 MB a step until the next log line (v49 D1b ran out of memory 26 steps in)
            if show || ckpt_now || step == start || step + 1 == steps_run || pend.len() >= 32 {
                // one wait for the GPU: the block's losses and norms come over together
                let stack = |ts: Vec<Tensor>| Tensor::stack(&ts, 0);
                let ls: Vec<f32> = stack(pend.iter().map(|p| p.loss.clone()).collect()).and_then(|t| t.to_vec1()).map_err(|e| format!("step {step} losses: {e}"))?;
                let gs: Vec<f64> = stack(pend.iter().map(|p| p.gn.clone()).collect()).and_then(|t| t.to_vec1()).map_err(|e| format!("step {step} norms: {e}"))?;
                let qts: Vec<Tensor> = pend.iter().filter_map(|p| p.qa.clone()).collect();
                let mut qs: Vec<f32> = if qts.is_empty() { Vec::new() } else { stack(qts).and_then(|t| t.to_vec1()).map_err(|e| format!("step {step} answer losses: {e}"))? };
                qs.reverse(); // popped in step order
                let block_s = t_block.elapsed().as_secs_f64();
                t_block = Instant::now();
                let wall = block_s / pend.len() as f64; // the steps' wall time: the block's average
                for (j, p) in pend.iter().enumerate() {
                    let (loss, gn) = (ls[j], gs[j]);
                    let qa = if p.qa.is_some() { qs.pop().unwrap_or(f32::NAN) } else { p.qa_host };
                    if !loss.is_finite() {
                        return Err(format!("step {}: loss is {loss}", p.step));
                    }
                    seen_tok += p.tok as u64;
                    tail.push(loss);
                    if let Some(log) = log.as_mut() {
                        writeln!(log, "{}\t{loss:.6}\t{qa:.6}\t{:.6e}\t{gn:.6}\t{:.1}\t{wall:.4}\t{}\t{}", p.step, p.lr, p.tok as f64 / wall, p.ctx, p.recall).map_err(|e| e.to_string())?;
                    }
                    if show && p.step == step {
                        let el = t_train.elapsed().as_secs_f64();
                        println!("step {step}/{steps_run} loss {loss:.4} answers {qa:.4} lr {lr:.3e} gnorm {gn:.3} {:.0} tok/s (mean {:.0})", p.tok as f64 / wall, seen_tok as f64 / el);
                    }
                }
                pend.clear();
            }
        } else {
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
            // with --skip-bad-batch the guard sees the losses before the step (J3); without it this is v41's
            // grad_norm and step, and the loss check right after
            let losses: Vec<f32> = if qa_done { vec![loss, qa_loss] } else { vec![loss] };
            let (gn, _stepped) = upd.step(&mut opt, &mut grads, lr as f64, &losses, None, step)?;
            model.homeostasis_step().map_err(|e| format!("step {step} homeostasis: {e}"))?; // v42; nothing when off
            if upd.guard.is_none() && !loss.is_finite() {
                return Err(format!("step {step}: loss is {loss}"));
            }
            let secs = t0.elapsed().as_secs_f64();
            seen_tok += step_tok as u64;
            if loss.is_finite() {
                tail.push(loss);
            }
            if let Some(log) = log.as_mut() {
                writeln!(log, "{step}\t{loss:.6}\t{qa_loss:.6}\t{lr:.6e}\t{gn:.6}\t{:.1}\t{secs:.4}\t{ctx_s}\t{n_recall}", step_tok as f64 / secs).map_err(|e| e.to_string())?;
            }
            if show {
                let el = t_train.elapsed().as_secs_f64();
                println!("step {step}/{steps_run} loss {loss:.4} answers {qa_loss:.4} lr {lr:.3e} gnorm {gn:.3} {:.0} tok/s (mean {:.0})", step_tok as f64 / secs, seen_tok as f64 / el);
            }
        }
        if ckpt_now {
            save_checkpoint(&ckdir, step + 1, &varmap, &opt, &sampler, sampler_long.as_ref(), &sig, prior_s + t_train.elapsed().as_secs_f64(), prior_tok + seen_tok, upd.ema.as_ref(), upd.gf.as_ref(), a.keep_ckpts)?;
            t_block = Instant::now(); // the save is not training time
        }
        // --gpu-share: rest so the GPU is busy about that share of the time (same numbers, only slower)
        if a.gpu_share < 1.0 {
            dev.synchronize().map_err(|e| format!("step {step} synchronize: {e}"))?;
            let rest = t0.elapsed().as_secs_f64() * (1.0 / a.gpu_share - 1.0);
            std::thread::sleep(std::time::Duration::from_secs_f64(rest.min(10.0)));
            rested += rest.min(10.0);
        }
    }
    if a.gpu_share < 1.0 {
        println!("--gpu-share {}: rested {rested:.1}s in this session (included in the times below)", a.gpu_share);
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
    save_extras(a, &opt, &upd, None, dev)?;
    let mut extra = v42_cols(a, &upd);
    if a.late_qa_frac > 0.0 {
        extra.push(("late_qa_frac", a.late_qa_frac.to_string()));
        extra.push(("qa_from_step", plan.start.to_string()));
    }
    finish(
        &model,
        &varmap,
        a,
        bpe,
        dev,
        [
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
        ]
        .into_iter()
        .chain(extra)
        .collect(),
    )
}

/// v42: result.tsv columns for the v42 settings that are on (none when all are off, so v41's file is unchanged).
fn v42_cols(a: &Args, upd: &Updater) -> Vec<(&'static str, String)> {
    let mut c = Vec::new();
    if a.decay_gates {
        c.push(("decay_gates", "1".to_string()));
    }
    if a.beta2 != 0.95 {
        c.push(("beta2", a.beta2.to_string()));
    }
    if a.fast_step {
        c.push(("fast_step", "1".to_string()));
    }
    if a.skip_bad_batch {
        c.push(("skipped_steps", upd.skipped().to_string()));
    }
    if let Some(d) = a.ema {
        c.push(("ema", d.to_string()));
    }
    if a.wd_target_w0 {
        c.push(("wd_target", "w0".to_string()));
    }
    if let Some(p) = &a.load_opt {
        c.push(("load_opt", p.display().to_string()));
    }
    if let Some(n) = a.replay_night {
        c.push(("replay_night", n.to_string()));
    }
    let r = recipe_signature(a);
    if !r.is_empty() {
        c.push(("recipe", r.trim().to_string()));
    }
    c
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

/// v42 (A8, A9): the episodes packed end to end into rows of `ctx` tokens, `rows` rows a batch (to measure, not learn).
fn packed(eps: &[Vec<u32>], ctx: usize, rows: usize, dev: &Device) -> Result<Vec<Tensor>, String> {
    let all: Vec<u32> = eps.concat();
    let mut out = Vec::new();
    for chunk in all.chunks(ctx * rows) {
        let r = chunk.len() / ctx;
        if r > 0 {
            out.push(Tensor::from_vec(chunk[..r * ctx].to_vec(), (r, ctx), dev).map_err(|e| e.to_string())?);
        }
        let rest = &chunk[r * ctx..];
        if !rest.is_empty() {
            out.push(Tensor::from_vec(rest.to_vec(), (1, rest.len()), dev).map_err(|e| e.to_string())?);
        }
    }
    Ok(out)
}

/// v42 (A8, A9): `n` batches of her usual reading from the replay shard, on a stream of their own (30), so the sleep's
/// replay windows are the same with or without them. None without replay (--replay-rows 0).
fn background(a: &Args, n: usize, dev: &Device) -> Result<Option<Vec<Tensor>>, String> {
    let Some(mut r) = Replay::open(a, 30)? else { return Ok(None) };
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(r.batch(dev)?.0);
    }
    Ok(Some(out))
}

/// v42 (A8): plasticity upkeep at the start of the sleep, before the optimizer and the anchor (which then hold the
/// upkept weights): ReDo at --redo TAU on --redo-batches batches of her usual reading (the episodes without replay),
/// then shrink-and-perturb. Nothing without --redo / --shrink-perturb. ReDo writes DIR/upkeep.tsv.
fn sleep_upkeep(a: &Args, varmap: &VarMap, model: &Model, eps: &[Vec<u32>], dev: &Device) -> Result<(), String> {
    if let Some(tau) = a.redo {
        let t0 = Instant::now();
        let batches = match background(a, a.redo_batches, dev)? {
            Some(b) => b,
            None => packed(eps, a.replay_ctx, 8, dev)?,
        };
        let act = ava_gpu::upkeep::unit_activity(model, &batches).map_err(|e| format!("--redo activity: {e}"))?;
        let rep = ava_gpu::upkeep::redo(varmap, model, &act, tau, derive_seed(a.seed, 0x7265_646f, 0)).map_err(|e| format!("--redo: {e}"))?;
        let mut f = fs::File::create(a.out.join("upkeep.tsv")).map_err(|e| format!("upkeep.tsv: {e}"))?;
        writeln!(f, "block\tlif_dormant\tlif_units\tmlp_dormant\tmlp_units").map_err(|e| e.to_string())?;
        for (l, (dl, dm)) in rep.lif.iter().zip(&rep.mlp).enumerate() {
            writeln!(f, "{l}\t{dl}\t{}\t{dm}\t{}", rep.d, rep.hidden).map_err(|e| e.to_string())?;
        }
        println!(
            "--redo {tau}: {} dormant units recycled (LIF {:?} of {}, MLP {:?} of {} per block), measured on {} batches, {:.1}s",
            rep.total(),
            rep.lif,
            rep.d,
            rep.mlp,
            rep.hidden,
            batches.len(),
            t0.elapsed().as_secs_f64()
        );
    }
    if let Some((l, s)) = a.shrink_perturb {
        let n = ava_gpu::upkeep::shrink_perturb(varmap, l, s, derive_seed(a.seed, 0x7370, 0)).map_err(|e| format!("--shrink-perturb: {e}"))?;
        println!("--shrink-perturb {l},{s}: {n} hidden matrices shrunk and perturbed");
    }
    Ok(())
}

/// v42 (A9) --sparse-ft T: count each product-key slot's reads on the episodes and on her usual reading, and train only
/// the top-T value rows by TF-IDF; every other weight stays as it is. Returns a fresh AdamW over the values alone and
/// (the values, the [slots, 1] row mask) for `mask_rows`.
fn sparse_ft(a: &Args, varmap: &VarMap, model: &Model, eps: &[Vec<u32>], top: usize, dev: &Device) -> Result<(AdamW, (Tensor, Tensor)), String> {
    use ava_gpu::{memory, upkeep};
    let name = upkeep::pk_values_name(model).ok_or("--sparse-ft needs a model with a product-key memory (--pk-mem)")?;
    let bg = background(a, a.sparse_ft_batches, dev)?.ok_or("--sparse-ft needs replay (--replay-rows) for its background reading")?;
    let new = upkeep::slot_counts(model, &packed(eps, a.replay_ctx, 8, dev)?).map_err(|e| format!("--sparse-ft: {e}"))?;
    let back = upkeep::slot_counts(model, &bg).map_err(|e| format!("--sparse-ft: {e}"))?;
    let rows = memory::sparse_ft_rows(&new, &back, top);
    if rows.is_empty() {
        return Err("--sparse-ft: the episodes read no slot".into());
    }
    let var = varmap.data().lock().unwrap().get(&name).cloned().ok_or_else(|| format!("--sparse-ft: no weight {name}"))?;
    let slots = var.dim(0).map_err(|e| e.to_string())?;
    let mask = memory::row_mask(&rows, slots, dev).map_err(|e| e.to_string())?;
    let mut opt = AdamW::new(vec![Param { name: name.clone(), var: var.clone(), decay: false }], a.wd as f64).map_err(|e| format!("adamw: {e}"))?;
    opt.beta2 = a.beta2;
    println!("--sparse-ft {top}: {} of {slots} rows of {name} learn (TF-IDF of the episodes' slot reads against {} background batches); every other weight is frozen", rows.len(), bg.len());
    Ok((opt, (var.as_tensor().clone(), mask)))
}

/// v42 (A9): only the chosen rows of the values keep their gradient.
fn mask_rows(grads: &mut GradStore, sparse: &Option<(Tensor, Tensor)>, step: usize) -> Result<(), String> {
    let Some((key, mask)) = sparse else { return Ok(()) };
    let g = grads.remove(key).ok_or_else(|| format!("sleep step {step}: --sparse-ft: the values got no gradient"))?;
    grads.insert(key, g.broadcast_mul(mask).map_err(|e| format!("sleep step {step}: --sparse-ft mask: {e}"))?);
    Ok(())
}

fn phase_sleep(a: &Args, dev: &Device, bpe: &Bpe) -> Result<(), String> {
    let facts = a.facts.as_ref().ok_or("--phase sleep needs --facts")?;
    let (cfg, varmap, model) = ava_gpu::ckpt::load_auto(&a.init, a.kind, a.eval_ctx, dev)?;
    write_run_arch(&a.out, &cfg.arch)?; // v42: the init's block options (none: today's block)
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
    let a = &with_night(a)?; // X2: the replay cursor's night (None without a cursor: the v41 windows)
    sleep_upkeep(a, &varmap, &model, &eps, dev)?; // v42 (A8); nothing when off
    let mut replay = Replay::open(a, 1)?;
    // v42 (A9): with --sparse-ft only the chosen rows of the product-key values learn
    let (mut opt, sparse) = match a.sparse_ft {
        Some(top) => {
            let (o, s) = sparse_ft(a, &varmap, &model, &eps, top, dev)?;
            (o, Some(s))
        }
        None => (new_opt(a, &varmap, dev)?, None),
    };
    let mm = Mm { bf16: ava_gpu::bf16_on() };
    let steps = a.max_steps.map_or(a.sleep_steps, |m| m.min(a.sleep_steps));
    let per_cycle = a.sleep_steps.div_ceil(a.cycles).max(1);
    // U1: the sleep's own peak LR and warm-up (by default --lr and v41's min(--warmup, steps / 10))
    let lr_peak = a.sleep_lr.unwrap_or(a.lr);
    let warm = a.sleep_warmup.unwrap_or(a.warmup.min(steps / 10));
    // X1: v41 (all) reads everything with the engram off; `episodes` switches it off only around the episodes
    let ep_only = a.sleep_lesion == SleepLesion::Episodes;
    let base_off = !ep_only;
    println!(
        "sleep: {} episodes ({} tokens) from {}; {steps} steps of {} episodes, {} cycles, surprise alpha {}; replay {} x {}; {}",
        eps.len(),
        eps.iter().map(Vec::len).sum::<usize>(),
        facts.display(),
        a.sleep_rows,
        a.cycles,
        a.surprise_alpha,
        a.replay_rows,
        a.replay_ctx,
        if ep_only { "engram off for the episodes only (replay and the anchor read with it on)" } else { "engram off while asleep" }
    );
    if a.sleep_lr.is_some() || a.sleep_warmup.is_some() {
        println!("sleep lr: warm-up {warm} steps to {lr_peak}, then cosine down to {}", lr_peak * a.min_lr_frac);
    }
    // what she knows of the episodes without the engram: loss per token of each (all tokens after the first EOS)
    let ep_loss = |model: &Model| -> Result<Vec<f64>, String> {
        model.set_lesion(false, false, true);
        model.set_inference(true);
        let from = vec![1usize; eps.len()];
        let r = mc::scored_nll(model, dev, &eps, &from, a.score_batch).map_err(|e| format!("episode loss: {e}"));
        model.set_inference(false);
        model.set_lesion(false, false, base_off);
        Ok(r?.iter().map(|(s, n)| s / (*n).max(1) as f64).collect())
    };
    model.set_lesion(false, false, base_off); // v41: the engram is off while she sleeps (and for the surprise measure)
    let anchor = measure_anchor(a, &model, &opt, &mm, dev, 11, bpe, cfg.glance, base_off)?;
    let mut upd = Updater::new(a, &varmap, decay_target(a, &opt, anchor.as_ref())?)?;
    let mut kl = KlRef::open(a, dev, 1, cfg.vocab, base_off)?;
    let mut devq = DevCheck::open(a, bpe, cfg.glance, cfg.vocab)?;
    let mut log = fs::File::create(a.out.join("log.tsv")).map_err(|e| format!("log.tsv: {e}"))?;
    writeln!(log, "step\tcycle\tepisode_loss\treplay_loss\tlr\tgrad_norm\ttokens_per_sec").map_err(|e| e.to_string())?;
    let mut cyc = fs::File::create(a.out.join("cycles.tsv")).map_err(|e| format!("cycles.tsv: {e}"))?;
    writeln!(cyc, "cycle\tstep\tmean_episode_loss\tmax_episode_loss\tmin_episode_loss").map_err(|e| e.to_string())?;
    let mut rng = SplitMix64::new(derive_seed(a.seed, 0x736c_6565_70, 0));
    let (mut prio, mut total) = (Vec::new(), 0.0);
    let loss_before = ep_loss(&model)?;
    let t_train = Instant::now();
    let mut seen_tok = 0u64;
    let (mut steps_done, mut stopped) = (steps, false);
    for step in 0..steps {
        let t0 = Instant::now();
        if step % per_cycle == 0 {
            let cycle = step / per_cycle;
            let l = if step == 0 { loss_before.clone() } else { ep_loss(&model)? };
            let (mean, mx, mn) = (l.iter().sum::<f64>() / l.len() as f64, l.iter().cloned().fold(f64::MIN, f64::max), l.iter().cloned().fold(f64::MAX, f64::min));
            writeln!(cyc, "{cycle}\t{step}\t{mean:.6}\t{mx:.6}\t{mn:.6}").map_err(|e| e.to_string())?;
            println!("cycle {cycle} at step {step}: episode loss mean {mean:.4} (min {mn:.4}, max {mx:.4})");
            // v42: the forgetting gate (U2) and the dev questions (U5) at every cycle
            if let Some(k) = kl.as_mut() {
                k.measure(&model, step, cycle)?;
            }
            if let Some(d) = devq.as_mut() {
                d.check(a, &model, dev, step, cycle)?;
                model.set_lesion(false, false, base_off);
            }
            // U5: stop once the episodes are learned well enough
            if let Some(x) = a.sleep_stop_loss {
                if mean < x {
                    println!("--sleep-stop-loss {x}: the mean episode loss {mean:.4} is below it; the sleep stops at step {step}");
                    (steps_done, stopped) = (step, true);
                    break;
                }
            }
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
        if ep_only {
            model.set_lesion(false, false, true);
        }
        let res = model.loss_and_grads_a_w(&ids, &tg, Some(&w), &mm);
        if ep_only {
            model.set_lesion(false, false, false);
        }
        let (l, _, mut grads) = res.map_err(|e| format!("sleep step {step}: {e}"))?;
        let mut step_tok = b * t;
        let mut rl = f32::NAN;
        if let Some(r) = replay.as_mut() {
            let (rid, rtg) = r.batch(dev)?;
            let (x, rg, n) = r.grads_on(&model, &mm, &rid, &rtg)?;
            merge(&opt, &mut grads, &rg, a.replay_weight)?;
            rl = x;
            step_tok += n;
            if let (Some(k), true) = (kl.as_mut(), a.kl_weight > 0.0) {
                // U2: + weight · KL(reference || current) on the same windows
                let (_, kg) = k.grads(&model, &rid)?;
                merge(&opt, &mut grads, &kg, a.kl_weight as f32)?;
            }
        }
        let lr = lr_at(a, step, warm, steps, lr_peak);
        let loss = l.to_scalar::<f32>().map_err(|e| e.to_string())?;
        let losses: Vec<f32> = if replay.is_some() { vec![loss, rl] } else { vec![loss] };
        mask_rows(&mut grads, &sparse, step)?; // v42 (A9); nothing when off
        let (gn, _stepped) = upd.step(&mut opt, &mut grads, lr as f64, &losses, anchor.as_ref(), step)?;
        model.homeostasis_step().map_err(|e| format!("sleep step {step} homeostasis: {e}"))?; // v42; nothing when off
        if upd.guard.is_none() && !loss.is_finite() {
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
    if !stopped {
        // the forgetting gate and the dev questions once more at the end (a stopped sleep was just checked)
        if let Some(k) = kl.as_mut() {
            k.measure(&model, steps_done, steps_done.div_ceil(per_cycle))?;
        }
        if let Some(d) = devq.as_mut() {
            d.check(a, &model, dev, steps_done, steps_done.div_ceil(per_cycle))?;
            model.set_lesion(false, false, base_off);
        }
    }
    let after = ep_loss(&model)?;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!("slept {steps_done} steps in {train_s:.1}s; episode loss per token (engram off) {:.4} -> {:.4}", mean(&loss_before), mean(&after));
    let drift = match &anchor {
        Some(an) => format!("{:.6e}", an.drift().map_err(|e| format!("anchor drift: {e}"))?),
        None => "nan".into(),
    };
    save_extras(a, &opt, &upd, anchor.as_ref(), dev)?;
    let mut extra = v42_cols(a, &upd);
    if ep_only {
        extra.push(("sleep_lesion", "episodes".into()));
    }
    if let Some(x) = a.sleep_lr {
        extra.push(("sleep_lr", x.to_string()));
    }
    if let Some(x) = a.sleep_warmup {
        extra.push(("sleep_warmup", x.to_string()));
    }
    if let Some(x) = a.sleep_stop_loss {
        extra.push(("sleep_stop_loss", x.to_string()));
        extra.push(("steps_run", steps_done.to_string()));
    }
    extra.extend(anchor_cols(a));
    extra.extend(gate_cols(a, kl.as_ref(), devq.as_ref()));
    finish(
        &model,
        &varmap,
        a,
        bpe,
        dev,
        [
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
        ]
        .into_iter()
        .chain(extra)
        .collect(),
    )?;
    advance_cursor(a)
}

/// v42: result.tsv columns of the running anchor (U3), when used.
fn anchor_cols(a: &Args) -> Vec<(&'static str, String)> {
    let mut c = Vec::new();
    if let Some(p) = &a.anchor_load {
        c.push(("anchor_load", p.display().to_string()));
        c.push(("anchor_gamma", a.anchor_gamma.to_string()));
        c.push(("anchor_point", if a.anchor_saved_point { "saved" } else { "now" }.to_string()));
    }
    if !a.anchor_episodes.is_empty() {
        c.push(("anchor_episodes", a.anchor_episodes.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(",")));
    }
    c
}

/// v42: result.tsv columns of the forgetting gate (U2) and the dev questions (U5), when used.
fn gate_cols(a: &Args, kl: Option<&KlRef>, devq: Option<&DevCheck>) -> Vec<(&'static str, String)> {
    let mut c = Vec::new();
    if let Some(k) = kl {
        c.push(("kl_weight", a.kl_weight.to_string()));
        c.push(("kl_ref_final", format!("{:.6e}", k.last)));
    }
    if let Some(d) = devq {
        c.push(("dev_items", d.last.n().to_string()));
        c.push(("dev_right_minus_wrong", format!("{:.6}", d.last.right_minus_wrong())));
        c.push(("dev_abstain_rate", format!("{:.6}", d.last.abstain_rate())));
    }
    c
}

// ------------------------------------------------------------------------------------------------ reward

fn phase_reward(a: &Args, dev: &Device, bpe: &Bpe) -> Result<(), String> {
    let (cfg, varmap, model) = ava_gpu::ckpt::load_auto(&a.init, a.kind, a.eval_ctx, dev)?;
    write_run_arch(&a.out, &cfg.arch)?; // v42: the init's block options (none: today's block)
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
    let a = &with_night(a)?; // X2: the replay cursor's night (None without a cursor: the v41 windows)
    // U4 (--idk-action): "I don't know" is scored as one more choice, the last, of every question
    let seqs: Vec<(Vec<Vec<u32>>, Vec<usize>)> =
        items.iter().map(|it| if a.idk_action { mc::item_seqs(bpe, &trainkit::with_idk(it), cfg.glance) } else { mc::item_seqs(bpe, it, cfg.glance) }).collect();
    // each choice's bytes (for --norm byte: her policy scores NLL per byte, and each choice's gradient is divided by
    // its bytes to match)
    let byte_len = ava_gpu::eval::byte_lengths_n(bpe, cfg.vocab);
    let choice_bytes: Vec<Vec<f64>> = seqs.iter().map(|(s, f)| mc::scored_bytes(&byte_len, s, f)).collect();
    // the policy's temperature in the score's units: per byte, --temp is divided by the pool's mean choice bytes, so
    // π is as sharp as with total NLL on an answer of average length (and the per-byte gradient as strong); the mean
    // is over the real choices only
    let mean_bytes = {
        let all: Vec<f64> = choice_bytes.iter().zip(&items).flat_map(|(b, it)| b[..it.choices.len()].iter()).map(|b| b.max(1.0)).collect();
        all.iter().sum::<f64>() / all.len().max(1) as f64
    };
    let temp = match a.norm {
        mc::Norm::Total => a.temp,
        mc::Norm::Byte => a.temp / mean_bytes,
        mc::Norm::Pmi => unreachable!("--norm pmi is refused when parsed"),
    };
    let mut replay = Replay::open(a, 2)?;
    let mut opt = new_opt(a, &varmap, dev)?;
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
    if a.reward_group > 1 || a.idk_action {
        println!(
            "reward: {} draw(s) per question{}{}",
            a.reward_group,
            if a.reward_group > 1 { ", the group's mean reward as the baseline" } else { "" },
            if a.idk_action { "; \"I don't know\" is an action worth 0 (right +1, wrong -1)" } else { "" }
        );
    }
    let anchor = measure_anchor(a, &model, &opt, &mm, dev, 12, bpe, cfg.glance, false)?;
    let mut upd = Updater::new(a, &varmap, decay_target(a, &opt, anchor.as_ref())?)?;
    let mut kl = KlRef::open(a, dev, 2, cfg.vocab, false)?;
    let mut devq = DevCheck::open(a, bpe, cfg.glance, cfg.vocab)?;
    // the forgetting gate and the dev questions are checked --cycles times, and at the end
    let check_every = (steps / a.cycles).max(1);
    let mut log = fs::File::create(a.out.join("log.tsv")).map_err(|e| format!("log.tsv: {e}"))?;
    writeln!(log, "step\tanswered\tright\twrong\tabstained\tmean_reward\tbaseline\tpolicy_loss\treplay_loss\tlr\tgrad_norm\tstep_s").map_err(|e| e.to_string())?;
    let mut rng = SplitMix64::new(derive_seed(a.seed, 0x7265_7761_7264, 0));
    let mut baseline = 0.0f64;
    let (mut tot_r, mut tot_w, mut tot_a) = (0usize, 0usize, 0usize);
    let t_train = Instant::now();
    for step in 0..steps {
        let t0 = Instant::now();
        if step % check_every == 0 {
            if let Some(k) = kl.as_mut() {
                k.measure(&model, step, step / check_every)?;
            }
            if let Some(d) = devq.as_mut() {
                d.check(a, &model, dev, step, step / check_every)?;
            }
        }
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
        // (first row in `all`, the choices drawn, their rewards, π, each choice's gradient scale: 1, or 1 / its bytes)
        let mut chosen: Vec<(usize, Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::new();
        let (mut r_n, mut w_n, mut a_n) = (0usize, 0usize, 0usize);
        for (k, &q) in pick.iter().enumerate() {
            let nc = items[q].choices.len();
            let na = nc + a.idk_action as usize;
            let tot = mc::normed(&nll[span[k]..span[k] + na], &choice_bytes[q], a.norm);
            let scale: Vec<f64> = match a.norm {
                mc::Norm::Total => vec![1.0; na],
                mc::Norm::Byte => choice_bytes[q].iter().map(|b| 1.0 / b.max(1.0)).collect(),
                mc::Norm::Pmi => unreachable!("--norm pmi is refused when parsed"),
            };
            let g = mc::grade_one(&tot[..nc], items[q].right);
            if g.margin < a.tau {
                a_n += 1; // "I don't know" by the margin: no reward, no update
                continue;
            }
            // her answer(s): drawn from softmax(-NLL / temp) (with --idk-action "I don't know" is one of the actions)
            let m = tot.iter().cloned().fold(f64::MAX, f64::min);
            let p: Vec<f64> = tot.iter().map(|&x| (-(x - m) / temp).exp()).collect();
            let z: f64 = p.iter().sum();
            let (mut draws, mut rewards) = (Vec::with_capacity(a.reward_group), Vec::with_capacity(a.reward_group));
            for _ in 0..a.reward_group {
                let c = draw(&p, z, &mut rng);
                let o = if a.idk_action && c == nc {
                    Outcome::Abstain
                } else if c == items[q].right {
                    Outcome::Right
                } else {
                    Outcome::Wrong
                };
                match o {
                    Outcome::Right => r_n += 1,
                    Outcome::Wrong => w_n += 1,
                    Outcome::Abstain => a_n += 1,
                }
                draws.push(c);
                rewards.push(o.reward());
            }
            chosen.push((span[k], draws, rewards, p.iter().map(|x| x / z).collect(), scale));
        }
        tot_r += r_n;
        tot_w += w_n;
        tot_a += a_n;
        let answered = chosen.len();
        let n_draws: usize = chosen.iter().map(|x| x.1.len()).sum();
        let mean_r = if answered > 0 { chosen.iter().flat_map(|x| x.2.iter()).sum::<f64>() / n_draws as f64 } else { 0.0 };
        let mut pl = f32::NAN;
        let mut grads: Option<GradStore> = None;
        let mut has_policy = false;
        if answered > 0 {
            // REINFORCE on her choice policy π = softmax(−S / temp) over the choices, S_j = NLL_j (· 1/bytes_j with
            // --norm byte):
            //   −(r − b) ∇log π_c = (r − b) / temp · Σ_j (1[j = c] − π_j) ∇S_j,
            // so every choice of an answered question is a row, weighted (r − b)(1[j = c] − π_j) / (temp · answered)
            // (· 1/bytes_j) on each of its scored tokens (π held fixed). v42 (U4): with --reward-group K the K draws
            // share the group's mean reward as b (trainkit::weights_group); a question whose draws all earned the
            // same gives no signal and is left out.
            let (mut rows, mut fr, mut rw): (Vec<&[u32]>, Vec<usize>, Vec<f32>) = (Vec::new(), Vec::new(), Vec::new());
            for (s, draws, rewards, pi, scale) in &chosen {
                let wq = if a.reward_group == 1 {
                    trainkit::weights_single(rewards[0], baseline, draws[0], pi, scale, temp, answered)
                } else {
                    trainkit::weights_group(draws, rewards, pi, scale, temp, answered)
                };
                if a.reward_group > 1 && wq.iter().all(|&x| x == 0.0) {
                    continue;
                }
                for (j, &wj) in wq.iter().enumerate() {
                    rows.push(all[s + j].as_slice());
                    fr.push(from[s + j]);
                    rw.push(wj);
                }
            }
            if !rows.is_empty() {
                let (ids, tg, w01, t) = padded(&rows, &fr);
                let w: Vec<f32> = w01.iter().enumerate().map(|(j, &x)| x * rw[j / t]).collect();
                let b = rows.len();
                let ids = Tensor::from_vec(ids, (b, t), dev).map_err(|e| e.to_string())?;
                let tg = Tensor::from_vec(tg, (b, t), dev).map_err(|e| e.to_string())?;
                let w = Tensor::from_vec(w, b * t, dev).map_err(|e| e.to_string())?;
                let (l, _, g) = model.loss_and_grads_a_wr(&ids, &tg, Some(&w), true, &mm).map_err(|e| format!("reward step {step}: {e}"))?;
                pl = l.to_scalar::<f32>().map_err(|e| e.to_string())?;
                grads = Some(g);
                has_policy = true;
            }
            baseline = a.baseline_decay * baseline + (1.0 - a.baseline_decay) * mean_r;
        }
        let mut rl = f32::NAN;
        if let Some(r) = replay.as_mut() {
            let (rid, rtg) = r.batch(dev)?;
            let (x, mut rg, _) = r.grads_on(&model, &mm, &rid, &rtg)?;
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
            if let (Some(k), Some(g), true) = (kl.as_mut(), grads.as_mut(), a.kl_weight > 0.0) {
                // U2: + weight · KL(reference || current) on the same windows
                let (_, kg) = k.grads(&model, &rid)?;
                merge(&opt, g, &kg, a.kl_weight as f32)?;
            }
        }
        let lr = lr_at(a, step, a.warmup.min(steps / 10), steps, a.lr);
        let mut gn = f64::NAN;
        if let Some(g) = grads.as_mut() {
            let mut losses = Vec::new();
            if has_policy {
                losses.push(pl);
            }
            if replay.is_some() {
                losses.push(rl);
            }
            gn = upd.step(&mut opt, g, lr as f64, &losses, anchor.as_ref(), step)?.0;
            model.homeostasis_step().map_err(|e| format!("reward step {step} homeostasis: {e}"))?; // v42; nothing when off
        }
        let secs = t0.elapsed().as_secs_f64();
        writeln!(log, "{step}\t{answered}\t{r_n}\t{w_n}\t{a_n}\t{mean_r:.4}\t{baseline:.4}\t{pl:.6}\t{rl:.6}\t{lr:.6e}\t{gn:.6}\t{secs:.3}").map_err(|e| e.to_string())?;
        if step % a.log_every == 0 || step + 1 == steps {
            println!("reward step {step}/{steps}: right {r_n} wrong {w_n} abstained {a_n}, baseline {baseline:.3}, replay {rl:.4}, {secs:.2}s");
        }
    }
    let train_s = t_train.elapsed().as_secs_f64();
    println!("reward learning: {steps} steps in {train_s:.1}s; right {tot_r}, wrong {tot_w}, abstained {tot_a} (her own answers while learning)");
    if let Some(k) = kl.as_mut() {
        k.measure(&model, steps, steps.div_ceil(check_every))?;
    }
    if let Some(d) = devq.as_mut() {
        d.check(a, &model, dev, steps, steps.div_ceil(check_every))?;
    }
    let drift = match &anchor {
        Some(an) => format!("{:.6e}", an.drift().map_err(|e| format!("anchor drift: {e}"))?),
        None => "nan".into(),
    };
    save_extras(a, &opt, &upd, anchor.as_ref(), dev)?;
    let mut extra = v42_cols(a, &upd);
    if a.reward_group > 1 {
        extra.push(("reward_group", a.reward_group.to_string()));
    }
    if a.idk_action {
        extra.push(("idk_action", "1".into()));
    }
    extra.extend(anchor_cols(a));
    extra.extend(gate_cols(a, kl.as_ref(), devq.as_ref()));
    finish(
        &model,
        &varmap,
        a,
        bpe,
        dev,
        [
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
        ]
        .into_iter()
        .chain(extra)
        .collect(),
    )?;
    advance_cursor(a)
}
