// TESTING / DESIGN ONLY — ava_gpu::sae::run: the sae_train and probe_states tools (A13). The bins only call these, so
// the unit tests run both tools end to end on a tiny saved model.
//
// Both read val windows once (ShardSet::iter_sequential: ctx + 1 tokens, the first ctx read; windows holding PAD are
// skipped), run them through lean copies of the model (ckpt::load_auto_lean: no autograd graph) on the CPU, one per
// --threads worker (collect_par: the forward is mostly serial, so the calls are shared out), in calls of at most
// 2048 tokens with the recording hook on (Model::record_states), and keep the states they need. No path holding
// "sealed" is ever opened.
//
// sae_train --model FILE --val DIR [--block L|final[,..]] [--tokens 200000] [--latents 8d] [--k 32] [--steps N]
//           [--batch 1024] [--ctx 256] [--threads 8] [--out DIR] [--bpe FILE] [--by-source DIR] [--sources a,b]
//           [--source-tokens 16384] [--lr 0.001] [--k-aux d/2] [--aux-coef 0.03125] [--dead-after N] [--holdout 0.1]
//           [--seed 1] [--call-tokens 2048]
//   Trains one SAE per block named (one forward pass serves them all): the last `holdout` share of the windows is
//   held out for the FVU. Writes sae.safetensors, report.md and latents.tsv into --out (into --out/<block> when
//   several blocks are named). --steps defaults to 16 passes over the training tokens; --dead-after to one pass.
//   With --by-source, each latent's mean activation on each source slice (--source-tokens tokens of each).
//   For the final state, the report also gives the next-token loss with x̂ in place of xf.
// probe_states --model FILE --val DIR [--tokens 100000] [--ctx 256] [--threads 8] [--by-source DIR] [--sources a,b]
//           [--per-source 16384] [--epochs 20] [--out DIR] [--bpe FILE] [--seed 1]
//           [--call-tokens 2048]
//   Linear probes (sae::probe) on the token's own embedding row (a baseline: what the token alone tells), every
//   block's state and xf, for the four text labels and, with --by-source, the source (the same number of windows
//   from each source; sources with fewer are left out). Trained on the first half of the windows, scored on the
//   second half. Writes report.md and probes.tsv into --out.

use super::probe::{fit_and_score, text_labels, ProbeCfg, Score, TEXT_TASKS};
use super::{par_ranges, Sae, TrainCfg};
use crate::model::{Kind, Model, TOK_EMB};
use ava_data::bpe::{Bpe, EOS, PAD};
use ava_data::numbers::{is_mark, MARK_BASE};
use ava_data::shards::ShardSet;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// The most tokens one forward call holds (a lean forward frees each intermediate as it goes).
pub const MAX_CALL_TOKENS: usize = 2048;

pub const DEFAULT_BPE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../release/ava/bpe.txt");

/// Refuse any path that names a sealed set.
pub fn refuse_sealed(p: &Path) -> Result<(), String> {
    if p.to_string_lossy().to_lowercase().contains("sealed") {
        return Err(format!("refused: {} names a sealed set", p.display()));
    }
    Ok(())
}

/// Up to ceil(tokens / ctx) windows (ctx + 1 tokens each) from the start of the shards in `dir`.
pub fn read_windows(dir: &Path, ctx: usize, tokens: usize) -> Result<Vec<Vec<u32>>, String> {
    refuse_sealed(dir)?;
    let set = ShardSet::open(dir).map_err(|e| format!("open {}: {e}", dir.display()))?;
    let want = tokens.div_ceil(ctx).max(1);
    let mut out = Vec::with_capacity(want);
    for w in set.iter_sequential(ctx) {
        if out.len() == want {
            break;
        }
        if w.len() == ctx + 1 && !w.contains(&PAD) {
            out.push(w);
        }
    }
    Ok(out)
}

/// The states of `wins` (the first ctx tokens of each) at the layers `keep` (0..blocks: a block's output; blocks: xf),
/// each [windows·ctx × d] row by row, and the forward's tokens per second.
pub fn collect(model: &Model, wins: &[Vec<u32>], ctx: usize, keep: &[usize], call_tokens: usize) -> Result<(Vec<Vec<f32>>, f64), String> {
    let e = |e: candle_core::Error| e.to_string();
    let rows = (call_tokens / ctx).max(1);
    let d = model.cfg.d;
    let mut out: Vec<Vec<f32>> = keep.iter().map(|_| Vec::with_capacity(wins.len() * ctx * d)).collect();
    let t0 = std::time::Instant::now();
    model.record_states(true);
    let r: Result<(), String> = (|| {
        for g in wins.chunks(rows) {
            let ids: Vec<u32> = g.iter().flat_map(|w| w[..ctx].iter().copied()).collect();
            let ids = Tensor::from_vec(ids, (g.len(), ctx), &Device::Cpu).map_err(e)?;
            drop(model.forward(&ids).map_err(e)?);
            let st = model.take_states().ok_or("no states were recorded (does the model have block options?)")?;
            for (o, &l) in out.iter_mut().zip(keep) {
                let s = st.get(l).ok_or(format!("layer {l}: the forward recorded {} states", st.len()))?;
                o.extend(s.flatten_all().map_err(e)?.to_vec1::<f32>().map_err(e)?);
            }
        }
        Ok(())
    })();
    model.record_states(false);
    r?;
    Ok((out, (wins.len() * ctx) as f64 / t0.elapsed().as_secs_f64().max(1e-9)))
}

/// A model loaded for reading her states: the lean copy (no autograd graph), activity not tracked, inference on.
pub fn load_states_model(path: &Path, ctx: usize) -> Result<(crate::model::Config, candle_nn::VarMap, Model), String> {
    refuse_sealed(path)?;
    let (cfg, varmap, m) = crate::ckpt::load_auto_lean(path, Kind::D, ctx, &Device::Cpu)?;
    m.set_track_activity(false);
    m.set_inference(true);
    Ok((cfg, varmap, m))
}

/// `collect` with the calls shared out over `workers` threads, each running its own lean copy of the model in
/// `model_p` with candle on one thread (RAYON_NUM_THREADS is 1 while they run, then put back). The everything-on CPU
/// forward is mostly serial (one model on 8 candle threads keeps about 1.3 cores busy), so this is what uses the
/// cores. A window's rows never depend on the other rows of its call, and the calls are put back in order, so the
/// states are the ones `collect` gives.
pub fn collect_par(model_p: &Path, ctx: usize, wins: &[Vec<u32>], keep: &[usize], call_tokens: usize, workers: usize) -> Result<(Vec<Vec<f32>>, f64), String> {
    let rows = (call_tokens / ctx).max(1);
    let groups: Vec<&[Vec<u32>]> = wins.chunks(rows).collect();
    let workers = workers.max(1).min(groups.len().max(1));
    let t0 = std::time::Instant::now();
    // candle on one thread per worker while they run, then put back (crate::thread_env: also on a panic, and no other
    // test of the suite sees the change)
    let one = crate::thread_env::scoped(&[("RAYON_NUM_THREADS", "1".to_string())]);
    let next = std::sync::atomic::AtomicUsize::new(0);
    type Part = (usize, Vec<Vec<f32>>);
    let res: Vec<Result<Vec<Part>, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| -> Result<Vec<Part>, String> {
                    let (_, _, m) = load_states_model(model_p, ctx)?;
                    let mut mine = Vec::new();
                    loop {
                        let g = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(ws) = groups.get(g) else { break };
                        mine.push((g, collect(&m, ws, ctx, keep, call_tokens)?.0));
                    }
                    Ok(mine)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("a states worker panicked".into()))).collect()
    });
    drop(one);
    let mut parts: Vec<Part> = Vec::with_capacity(groups.len());
    for r in res {
        parts.extend(r?);
    }
    parts.sort_by_key(|p| p.0);
    let mut out: Vec<Vec<f32>> = (0..keep.len()).map(|l| Vec::with_capacity(parts.iter().map(|p| p.1[l].len()).sum())).collect();
    for (_, st) in parts {
        for (o, s) in out.iter_mut().zip(st) {
            o.extend_from_slice(&s);
        }
    }
    Ok((out, (wins.len() * ctx) as f64 / t0.elapsed().as_secs_f64().max(1e-9)))
}

/// A layer's name: block<i> or final.
pub fn layer_name(l: usize, blocks: usize) -> String {
    if l == blocks {
        "final".into()
    } else {
        format!("block{l}")
    }
}

fn parse_layer(s: &str, blocks: usize) -> Result<usize, String> {
    match s.trim() {
        "final" | "xf" => Ok(blocks),
        v => {
            let l = v.trim_start_matches("block").parse::<usize>().map_err(|_| format!("bad --block {v:?} (a block number or final)"))?;
            if l >= blocks {
                return Err(format!("--block {l}: the model has blocks 0..{}", blocks - 1));
            }
            Ok(l)
        }
    }
}

/// Text for display: length marks as [#L], ends of document as ⟨eos⟩, line breaks as ⏎.
pub fn show(bpe: &Bpe, ids: &[u32]) -> String {
    let mut out = String::new();
    let mut run: Vec<u32> = Vec::new();
    let flush = |run: &mut Vec<u32>, out: &mut String| {
        if !run.is_empty() {
            out.push_str(&bpe.decode(run));
            run.clear();
        }
    };
    for &t in ids {
        if is_mark(t) {
            flush(&mut run, &mut out);
            let _ = write!(out, "[#{}]", t - MARK_BASE + 1);
        } else if t == EOS || t == PAD || t as usize >= bpe.vocab_size() {
            flush(&mut run, &mut out);
            out.push_str(if t == EOS { "⟨eos⟩" } else { "⟨?⟩" });
        } else {
            run.push(t);
        }
    }
    flush(&mut run, &mut out);
    out.replace('\r', "").replace('\n', "⏎").replace('\t', "⇥").replace('`', "'")
}

/// About 12 tokens around position p of window `w` (8 before, 3 after), the token marked «so».
pub fn context(bpe: &Bpe, w: &[u32], p: usize) -> String {
    let a = p.saturating_sub(8);
    let e = (p + 4).min(w.len());
    format!("{}«{}»{}", show(bpe, &w[a..p]), show(bpe, &w[p..p + 1]), show(bpe, &w[p + 1..e]))
}

struct Args {
    kv: HashMap<String, String>,
}

impl Args {
    fn parse(a: &[String], known: &[&str]) -> Result<Args, String> {
        let mut kv = HashMap::new();
        let mut i = 0;
        while i < a.len() {
            let k = a[i].as_str();
            if !known.contains(&k) {
                return Err(format!("unknown option {k} (options: {})", known.join(" ")));
            }
            let v = a.get(i + 1).ok_or(format!("{k} needs a value"))?;
            kv.insert(k.to_string(), v.clone());
            i += 2;
        }
        Ok(Args { kv })
    }
    fn str(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(|s| s.as_str())
    }
    fn path(&self, k: &str) -> Result<Option<PathBuf>, String> {
        match self.str(k) {
            None => Ok(None),
            Some(v) => {
                let p = PathBuf::from(v);
                refuse_sealed(&p)?;
                Ok(Some(p))
            }
        }
    }
    fn num<T: std::str::FromStr>(&self, k: &str, def: T) -> Result<T, String> {
        match self.str(k) {
            None => Ok(def),
            Some(v) => v.parse::<T>().map_err(|_| format!("{k}: bad number {v:?}")),
        }
    }
}

/// The source slices to read: --sources a,b (in that order) or every folder in `dir` (sorted).
fn source_list(dir: &Path, names: Option<&str>) -> Result<Vec<(String, PathBuf)>, String> {
    refuse_sealed(dir)?;
    let mut v: Vec<String> = match names {
        Some(n) => n.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        None => {
            let mut v = Vec::new();
            for e in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
                let e = e.map_err(|e| e.to_string())?;
                let name = e.file_name().to_string_lossy().into_owned();
                if !name.to_lowercase().contains("sealed") && e.path().is_dir() {
                    v.push(name);
                }
            }
            v.sort();
            v
        }
    };
    v.dedup();
    v.into_iter()
        .map(|n| {
            let p = dir.join(&n);
            refuse_sealed(&p)?;
            Ok((n, p))
        })
        .collect()
}

/// The model's shape and weights (the forwards run in collect_par's workers, each on its own copy) and the BPE.
struct Loaded {
    cfg: crate::model::Config,
    varmap: candle_nn::VarMap,
    bpe: Bpe,
}

fn load(model: &Path, ctx: usize, bpe: Option<PathBuf>) -> Result<Loaded, String> {
    let (cfg, varmap, m) = load_states_model(model, ctx)?;
    if !m.arch_path() {
        return Err(format!("{}: no block options, so no block-options forward to record (A13 reads Ava 0.4 models)", model.display()));
    }
    let bp = bpe.unwrap_or_else(|| PathBuf::from(DEFAULT_BPE));
    refuse_sealed(&bp)?;
    let bpe = Bpe::load(&bp).map_err(|e| format!("bpe {}: {e}", bp.display()))?;
    Ok(Loaded { cfg, varmap, bpe })
}

fn check_vocab(set_dir: &Path, vocab: usize) -> Result<(), String> {
    let set = ShardSet::open(set_dir).map_err(|e| format!("open {}: {e}", set_dir.display()))?;
    if set.vocab_size as usize != vocab {
        return Err(format!("{}: vocab {} but the model has {vocab}", set_dir.display(), set.vocab_size));
    }
    Ok(())
}

fn pct(x: f64) -> String {
    format!("{:.1}%", 100.0 * x)
}

// ------------------------------------------------------------------------------------------- sae_train

const SAE_OPTS: [&str; 22] = [
    "--model", "--val", "--block", "--tokens", "--latents", "--k", "--steps", "--batch", "--ctx", "--threads", "--out", "--bpe", "--by-source",
    "--sources", "--source-tokens", "--lr", "--k-aux", "--aux-coef", "--dead-after", "--holdout", "--seed", "--call-tokens",
];

/// Per-latent statistics over a set of codes.
struct LatentStats {
    fires: Vec<u64>,
    sum: Vec<f64>,
    max: Vec<f32>,
    /// per latent: the strongest activations (value, row), most first, a few tokens apart
    top: Vec<Vec<(f32, u32)>>,
    /// per latent: (most common token among its firings, its share of them)
    top_tok: Vec<(u32, f64)>,
}

fn latent_stats(c: &super::Codes, m: usize, toks: &[u32], ctx: usize, keep: usize) -> LatentStats {
    let mut fires = vec![0u64; m];
    let mut sum = vec![0f64; m];
    let mut max = vec![0f32; m];
    let mut cand: Vec<Vec<(f32, u32)>> = vec![Vec::new(); m];
    let mut tok: Vec<HashMap<u32, u32>> = vec![HashMap::new(); m];
    let cap = keep * 6;
    for (r, (ii, vv)) in c.idx.chunks_exact(c.k).zip(c.val.chunks_exact(c.k)).enumerate() {
        for (&j, &v) in ii.iter().zip(vv) {
            if v <= 0.0 {
                continue;
            }
            let j = j as usize;
            fires[j] += 1;
            sum[j] += v as f64;
            max[j] = max[j].max(v);
            *tok[j].entry(toks[r]).or_insert(0) += 1;
            let cj = &mut cand[j];
            cj.push((v, r as u32));
            if cj.len() >= cap * 2 {
                cj.sort_by(|a, b| b.0.total_cmp(&a.0));
                cj.truncate(cap);
            }
        }
    }
    let top = cand
        .into_iter()
        .map(|mut cj| {
            cj.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut pick: Vec<(f32, u32)> = Vec::with_capacity(keep);
            for (v, r) in cj {
                // a few tokens apart within a window, so the contexts differ
                let near = pick.iter().any(|&(_, q)| q as usize / ctx == r as usize / ctx && (q as i64 - r as i64).abs() < 6);
                if !near {
                    pick.push((v, r));
                }
                if pick.len() == keep {
                    break;
                }
            }
            pick
        })
        .collect();
    let top_tok = tok
        .iter()
        .zip(&fires)
        .map(|(h, &f)| h.iter().max_by_key(|(&t, &n)| (n, std::cmp::Reverse(t))).map_or((0, 0.0), |(&t, &n)| (t, n as f64 / f.max(1) as f64)))
        .collect();
    LatentStats { fires, sum, max, top, top_tok }
}

/// Next-token cross-entropy (nats/token) of states `x` [n × d] through the tied head, targets `tg`.
fn head_ce(x: &[f32], tg: &[u32], d: usize, emb: &Tensor) -> Result<f64, String> {
    let e = |e: candle_core::Error| e.to_string();
    let mut tot = 0f64;
    for (xs, ts) in x.chunks(MAX_CALL_TOKENS * d).zip(tg.chunks(MAX_CALL_TOKENS)) {
        let n = ts.len();
        let xt = Tensor::from_slice(xs, (n, d), &Device::Cpu).map_err(e)?;
        let lg = xt.matmul(&emb.t().map_err(e)?).map_err(e)?;
        let lp = candle_nn::ops::log_softmax(&lg, 1).map_err(e)?;
        let idx = Tensor::from_slice(ts, (n, 1), &Device::Cpu).map_err(e)?;
        tot -= lp.gather(&idx, 1).map_err(e)?.sum_all().map_err(e)?.to_scalar::<f32>().map_err(e)? as f64;
    }
    Ok(tot / tg.len().max(1) as f64)
}

/// sae_train (see the top of the file). Returns the lines it printed.
pub fn sae_train(argv: &[String]) -> Result<String, String> {
    let a = Args::parse(argv, &SAE_OPTS)?;
    let model_p = a.path("--model")?.ok_or("--model FILE is required")?;
    let val = a.path("--val")?.ok_or("--val DIR is required")?;
    let by_source = a.path("--by-source")?;
    let out = a.path("--out")?;
    let bpe_p = a.path("--bpe")?;
    let ctx: usize = a.num("--ctx", 256)?;
    let threads: usize = a.num("--threads", 8usize)?.max(1);
    let tokens: usize = a.num("--tokens", 200_000)?;
    let k: usize = a.num("--k", 32)?;
    let batch: usize = a.num("--batch", 1024)?;
    let src_tokens: usize = a.num("--source-tokens", 16384)?;
    let call: usize = a.num("--call-tokens", MAX_CALL_TOKENS)?.max(1);
    let holdout: f64 = a.num("--holdout", 0.1)?;
    let seed: u64 = a.num("--seed", 1)?;
    let _threads = crate::thread_env::threads(threads); // --threads while this tool runs, put back after (crate::thread_env)
    let mut said = String::new();
    let mut say = |s: &str| {
        println!("{s}");
        said.push_str(s);
        said.push('\n');
    };
    let t_all = std::time::Instant::now();
    let l = load(&model_p, ctx, bpe_p)?;
    let (d, nb) = (l.cfg.d, l.cfg.layers);
    let latents = match a.str("--latents") {
        None => 8 * d,
        Some(v) if v.ends_with('d') => v.trim_end_matches('d').parse::<usize>().map_err(|_| format!("--latents {v:?}"))? * d,
        Some(v) => v.parse::<usize>().map_err(|_| format!("--latents {v:?}"))?,
    };
    let layers: Vec<usize> = a.str("--block").unwrap_or("final").split(',').map(|s| parse_layer(s, nb)).collect::<Result<_, _>>()?;
    check_vocab(&val, l.cfg.vocab)?;
    let wins = read_windows(&val, ctx, tokens)?;
    let nw = wins.len();
    if nw < 2 {
        return Err(format!("{}: only {nw} full windows of {ctx}", val.display()));
    }
    let n_hold = ((nw as f64 * holdout).round() as usize).clamp(1, nw - 1);
    let n_train_w = nw - n_hold;
    say(&format!(
        "sae_train: {} (d {d}, {nb} blocks); {} tokens ({nw} windows of {ctx}) from {}; held out: the last {n_hold} windows",
        model_p.display(),
        nw * ctx,
        val.display()
    ));
    let (states, fwd_tps) = collect_par(&model_p, ctx, &wins, &layers, call, threads)?;
    say(&format!("  states collected: {:.0} tokens/s (forward, {threads} workers of one thread)", fwd_tps));
    let toks: Vec<u32> = wins.iter().flat_map(|w| w[..ctx].iter().copied()).collect();
    // the source slices, read once for every layer
    let mut src: Vec<(String, usize, Vec<Vec<f32>>)> = Vec::new();
    if let Some(dir) = &by_source {
        for (name, p) in source_list(dir, a.str("--sources"))? {
            check_vocab(&p, l.cfg.vocab)?;
            let w = read_windows(&p, ctx, src_tokens)?;
            if w.is_empty() {
                say(&format!("  source {name}: no full window, left out"));
                continue;
            }
            let (s, tps) = collect_par(&model_p, ctx, &w, &layers, call, threads)?;
            say(&format!("  source {name}: {} tokens ({tps:.0} tokens/s)", w.len() * ctx));
            src.push((name, w.len() * ctx, s));
        }
    }
    let emb = l.varmap.data().lock().unwrap().get(TOK_EMB).ok_or("no tok_emb")?.as_tensor().clone();
    let base = out.unwrap_or_else(|| {
        let names: Vec<String> = layers.iter().map(|&x| layer_name(x, nb)).collect();
        model_p.parent().unwrap_or(Path::new(".")).join(format!("a13-sae-{}", names.join("+")))
    });
    for (li, &layer) in layers.iter().enumerate() {
        let name = layer_name(layer, nb);
        let dir = if layers.len() == 1 { base.clone() } else { base.join(&name) };
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let x = &states[li];
        let (xtr, xho) = x.split_at(n_train_w * ctx * d);
        let n_train = n_train_w * ctx;
        let steps: usize = a.num("--steps", (16 * n_train).div_ceil(batch).max(1))?;
        let mut tc = TrainCfg::new(d, latents, k, steps);
        tc.batch = batch;
        tc.threads = threads;
        tc.seed = seed;
        tc.lr = a.num("--lr", tc.lr)?;
        tc.k_aux = a.num("--k-aux", tc.k_aux)?;
        tc.aux_coef = a.num("--aux-coef", tc.aux_coef)?;
        tc.dead_after = a.num("--dead-after", n_train.min(1_000_000) as u64)?;
        tc.log_every = (steps / 10).max(1);
        say(&format!("{name}: {latents} latents, k {k}, k_aux {}, {steps} steps of {batch}, lr {}, dead after {} tokens", tc.k_aux, tc.lr, tc.dead_after));
        let (sae, log) = super::train(xtr, d, &tc, &mut say)?;
        let t_ev = std::time::Instant::now();
        let (fvu_ho, codes_ho) = sae.fvu(xho, threads)?;
        let (fvu_tr, codes_tr) = sae.fvu(xtr, threads)?;
        let mut all = codes_tr.clone();
        all.idx.extend_from_slice(&codes_ho.idx);
        all.val.extend_from_slice(&codes_ho.val);
        let st = latent_stats(&all, sae.m, &toks, ctx, 8);
        let n_all = all.len();
        let dead = st.fires.iter().filter(|&&f| f == 0).count();
        let l0 = all.mean_l0();
        let enc_tps = n_all as f64 / t_ev.elapsed().as_secs_f64().max(1e-9);
        say(&format!(
            "{name}: FVU held out {fvu_ho:.4} (train {fvu_tr:.4}); L0 {l0:.2}; dead {dead} of {} ({}) on all {n_all} tokens; training {:.1} s, {:.0} tokens/s",
            sae.m,
            pct(dead as f64 / sae.m as f64),
            log.seconds,
            log.tokens_per_sec
        ));
        // the final state: the next-token loss with x̂ in place of xf (held-out tokens)
        let ce = if layer == nb {
            let tg: Vec<u32> = wins[n_train_w..].iter().flat_map(|w| w[1..=ctx].iter().copied()).collect();
            let xh = sae.decode(&codes_ho);
            let mean: Vec<f32> = (0..xho.len() / d).flat_map(|_| sae.b_dec.iter().map(|&v| v / sae.scale)).collect();
            let (c0, c1, cm) = (head_ce(xho, &tg, d, &emb)?, head_ce(&xh, &tg, d, &emb)?, head_ce(&mean, &tg, d, &emb)?);
            say(&format!("{name}: next-token loss on held-out tokens: xf {c0:.4}, SAE x̂ {c1:.4}, the mean {cm:.4} nats/token"));
            Some((c0, c1, cm))
        } else {
            None
        };
        // per source: each latent's mean activation and firing share
        let mut src_act: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
        for (sname, n, s) in &src {
            let c = sae.encode(&s[li], threads)?;
            let mut act = vec![0f64; sae.m];
            let mut fire = vec![0f64; sae.m];
            for (&j, &v) in c.idx.iter().zip(&c.val) {
                if v > 0.0 {
                    act[j as usize] += v as f64;
                    fire[j as usize] += 1.0;
                }
            }
            act.iter_mut().for_each(|x| *x /= *n as f64);
            fire.iter_mut().for_each(|x| *x /= *n as f64);
            src_act.push((sname.clone(), act, fire));
        }
        sae.save(&dir.join("sae.safetensors"))?;
        write_latents_tsv(&dir.join("latents.tsv"), &l.bpe, &st, n_all, &src_act)?;
        let rep = SaeReport {
            model: &model_p,
            val: &val,
            name: &name,
            sae: &sae,
            tc: &tc,
            log: &log,
            n_all,
            n_ho: xho.len() / d,
            fvu: (fvu_ho, fvu_tr),
            l0,
            dead,
            fwd_tps,
            enc_tps,
            ce,
            st: &st,
            src_act: &src_act,
            wins: &wins,
            ctx,
        };
        std::fs::write(dir.join("report.md"), rep.render(&l.bpe)).map_err(|e| format!("{}: {e}", dir.display()))?;
        say(&format!("{name}: wrote {}", dir.display()));
    }
    say(&format!("sae_train: done in {:.1} s", t_all.elapsed().as_secs_f64()));
    Ok(said)
}

fn tsv_text(s: &str) -> String {
    s.replace('\t', "⇥").replace('\n', "⏎")
}

fn write_latents_tsv(path: &Path, bpe: &Bpe, st: &LatentStats, n: usize, src: &[(String, Vec<f64>, Vec<f64>)]) -> Result<(), String> {
    let mut s = String::from("latent\tfire_share\tmean_act\tmean_when_fired\tmax_act\tdead\ttop_token\ttop_token_share");
    for (name, _, _) in src {
        let _ = write!(s, "\tact_{name}");
    }
    s.push('\n');
    for j in 0..st.fires.len() {
        let f = st.fires[j];
        let _ = write!(
            s,
            "{j}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{:.4}",
            f as f64 / n.max(1) as f64,
            st.sum[j] / n.max(1) as f64,
            st.sum[j] / f.max(1) as f64,
            st.max[j],
            (f == 0) as u8,
            if f == 0 { String::new() } else { tsv_text(&show(bpe, &[st.top_tok[j].0])) },
            st.top_tok[j].1
        );
        for (_, act, _) in src {
            let _ = write!(s, "\t{:.6}", act[j]);
        }
        s.push('\n');
    }
    std::fs::write(path, s).map_err(|e| format!("{}: {e}", path.display()))
}

struct SaeReport<'a> {
    model: &'a Path,
    val: &'a Path,
    name: &'a str,
    sae: &'a Sae,
    tc: &'a TrainCfg,
    log: &'a super::TrainLog,
    n_all: usize,
    n_ho: usize,
    fvu: (f64, f64),
    l0: f64,
    dead: usize,
    fwd_tps: f64,
    enc_tps: f64,
    ce: Option<(f64, f64, f64)>,
    st: &'a LatentStats,
    src_act: &'a [(String, Vec<f64>, Vec<f64>)],
    wins: &'a [Vec<u32>],
    ctx: usize,
}

impl SaeReport<'_> {
    /// The source where latent j is most active, and how many times the mean over the sources that is.
    fn best_source(&self, j: usize) -> Option<(&str, f64)> {
        if self.src_act.is_empty() {
            return None;
        }
        let mean = self.src_act.iter().map(|s| s.1[j]).sum::<f64>() / self.src_act.len() as f64;
        let (n, v) = self.src_act.iter().map(|s| (s.0.as_str(), s.1[j])).max_by(|a, b| a.1.total_cmp(&b.1))?;
        Some((n, if mean > 0.0 { v / mean } else { 0.0 }))
    }

    fn latent_block(&self, s: &mut String, bpe: &Bpe, j: usize, contexts: usize) {
        let st = self.st;
        let f = st.fires[j];
        let _ = write!(
            s,
            "### latent {j}: fires on {} of tokens, mean {:.3} when it fires (max {:.3}); top token «{}» ({} of its firings)",
            pct(f as f64 / self.n_all.max(1) as f64),
            st.sum[j] / f.max(1) as f64,
            st.max[j],
            show(bpe, &[st.top_tok[j].0]),
            pct(st.top_tok[j].1)
        );
        if let Some((n, r)) = self.best_source(j) {
            let _ = write!(s, "; most active on {n} ({r:.1}× the mean over sources)");
        }
        s.push_str("\n\n");
        for (i, &(v, r)) in st.top[j].iter().take(contexts).enumerate() {
            let (w, p) = (r as usize / self.ctx, r as usize % self.ctx);
            let _ = writeln!(s, "{}. `{}` ({v:.3})", i + 1, context(bpe, &self.wins[w], p));
        }
        s.push('\n');
    }

    fn render(&self, bpe: &Bpe) -> String {
        let (sae, tc, log, st) = (self.sae, self.tc, self.log, self.st);
        let mut s = String::new();
        let _ = writeln!(s, "# A13 sparse autoencoder: {} of {}\n", self.name, self.model.display());
        let _ = writeln!(s, "Dev val shards only ({}); no test set was read.\n", self.val.display());
        let _ = writeln!(s, "## Numbers\n");
        let _ = writeln!(s, "- data: {} tokens ({} windows of {}); the last {} tokens held out for the FVU", self.n_all, self.wins.len(), self.ctx, self.n_ho);
        let _ = writeln!(
            s,
            "- SAE: TopK, {} latents ({}× d {}), k {}, AuxK k_aux {} × {:.4}, dead after {} tokens; Adam lr {}, {} steps of {} ({:.1} passes over the training tokens)",
            sae.m,
            sae.m / sae.d,
            sae.d,
            sae.k,
            tc.k_aux,
            tc.aux_coef,
            tc.dead_after,
            tc.lr,
            tc.steps,
            tc.batch,
            (tc.steps * tc.batch) as f64 / (self.n_all - self.n_ho).max(1) as f64
        );
        let _ = writeln!(s, "- **FVU held out {:.4}** (train {:.4}); 1 − FVU = {} of the variance explained", self.fvu.0, self.fvu.1, pct(1.0 - self.fvu.0));
        let _ = writeln!(s, "- mean L0 {:.2} of k {} (latents firing per token)", self.l0, sae.k);
        let _ = writeln!(
            s,
            "- dead: {} of {} latents ({}) never fired on all {} tokens; {} were dead (not fired in {} training tokens) at the end of training, {} at some point",
            self.dead,
            sae.m,
            pct(self.dead as f64 / sae.m as f64),
            self.n_all,
            log.dead_end,
            tc.dead_after,
            log.ever_dead
        );
        if let Some((c0, c1, cm)) = self.ce {
            let rec = if cm > c0 { (cm - c1) / (cm - c0) } else { f64::NAN };
            let _ = writeln!(
                s,
                "- next-token loss on the held-out tokens: xf {c0:.4}, SAE x̂ {c1:.4} (+{:.4}), the mean state {cm:.4} nats/token: {} of the loss gap recovered",
                c1 - c0,
                pct(rec)
            );
        }
        let _ = writeln!(
            s,
            "- speed: forward {:.0} tokens/s (states collected once); training {:.1} s, {:.0} tokens/s; encoding {:.0} tokens/s",
            self.fwd_tps, log.seconds, log.tokens_per_sec, self.enc_tps
        );
        // how many alive latents are mostly one token
        let alive: Vec<usize> = (0..sae.m).filter(|&j| st.fires[j] > 0).collect();
        let one_tok = alive.iter().filter(|&&j| st.top_tok[j].1 >= 0.5).count();
        let _ = writeln!(
            s,
            "- of the {} alive latents, {} ({}) fire on one token for at least half of their firings (token-like); the rest spread over many tokens\n",
            alive.len(),
            one_tok,
            pct(one_tok as f64 / alive.len().max(1) as f64)
        );
        let mut order: Vec<usize> = alive.clone();
        order.sort_by(|&a, &b| st.fires[b].cmp(&st.fires[a]).then(a.cmp(&b)));
        let _ = writeln!(s, "## The 40 latents that fire most often, with the 8 contexts that excite each most\n");
        for &j in order.iter().take(40) {
            self.latent_block(&mut s, bpe, j, 8);
        }
        if order.len() > 52 {
            let _ = writeln!(s, "## A spread sample: 12 alive latents evenly spaced in firing rank (4 contexts each)\n");
            for q in 0..12 {
                let j = order[40 + q * (order.len() - 41) / 11];
                self.latent_block(&mut s, bpe, j, 4);
            }
        }
        if !self.src_act.is_empty() {
            let _ = writeln!(s, "## Latents most tied to one source\n");
            let _ = writeln!(s, "Mean activation per token on each source slice; a latent's ratio is its mean on the source over its mean across all the sources (latents firing on at least 0.1% of the val tokens).\n");
            for (sname, act, _) in self.src_act {
                let mut cand: Vec<(usize, f64)> = alive
                    .iter()
                    .filter(|&&j| st.fires[j] as f64 >= 0.001 * self.n_all as f64)
                    .map(|&j| {
                        let mean = self.src_act.iter().map(|x| x.1[j]).sum::<f64>() / self.src_act.len() as f64;
                        (j, if mean > 0.0 { act[j] / mean } else { 0.0 })
                    })
                    .collect();
                cand.sort_by(|a, b| b.1.total_cmp(&a.1));
                let list: Vec<String> = cand
                    .iter()
                    .take(4)
                    .map(|&(j, r)| format!("{j} ({r:.1}×, top token «{}»)", show(bpe, &[st.top_tok[j].0])))
                    .collect();
                let _ = writeln!(s, "- **{sname}**: {}", list.join(", "));
            }
            s.push('\n');
        }
        s
    }
}

// ------------------------------------------------------------------------------------------- probe_states

const PROBE_OPTS: [&str; 13] = ["--model", "--val", "--tokens", "--ctx", "--threads", "--by-source", "--sources", "--per-source", "--epochs", "--out", "--bpe", "--seed", "--call-tokens"];

/// The rows of one probe data set: per feature set, the training rows and the test rows.
struct Rows {
    a: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
}

/// One probe task: its labels for the training and the test rows of data set `rows`.
struct Task {
    name: String,
    classes: usize,
    rows: usize,
    ya: Vec<u32>,
    yb: Vec<u32>,
}

/// probe_states (see the top of the file). Returns the lines it printed.
pub fn probe_states(argv: &[String]) -> Result<String, String> {
    let a = Args::parse(argv, &PROBE_OPTS)?;
    let model_p = a.path("--model")?.ok_or("--model FILE is required")?;
    let val = a.path("--val")?.ok_or("--val DIR is required")?;
    let by_source = a.path("--by-source")?;
    let out = a.path("--out")?;
    let bpe_p = a.path("--bpe")?;
    let ctx: usize = a.num("--ctx", 256)?;
    let threads: usize = a.num("--threads", 8usize)?.max(1);
    let tokens: usize = a.num("--tokens", 100_000)?;
    let per_source: usize = a.num("--per-source", 16384)?;
    let call: usize = a.num("--call-tokens", MAX_CALL_TOKENS)?.max(1);
    let pc = ProbeCfg { epochs: a.num("--epochs", 20)?, seed: a.num("--seed", 1)?, ..ProbeCfg::default() };
    let _threads = crate::thread_env::threads(threads); // --threads while this tool runs, put back after (crate::thread_env)
    let mut said = String::new();
    let mut say = |s: &str| {
        println!("{s}");
        said.push_str(s);
        said.push('\n');
    };
    let t_all = std::time::Instant::now();
    let l = load(&model_p, ctx, bpe_p)?;
    let (d, nb) = (l.cfg.d, l.cfg.layers);
    let layers: Vec<usize> = (0..=nb).collect();
    let mut feat_names = vec!["tok".to_string()];
    feat_names.extend(layers.iter().map(|&x| layer_name(x, nb)));
    let emb: Vec<f32> = {
        let data = l.varmap.data().lock().unwrap();
        let t = data.get(TOK_EMB).ok_or("no tok_emb")?.as_tensor().clone();
        t.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())?
    };
    let tok_rows = |ids: &[u32]| -> Vec<f32> { ids.iter().flat_map(|&t| emb[t as usize * d..(t as usize + 1) * d].iter().copied()).collect() };
    // the four text tasks on the val windows
    check_vocab(&val, l.cfg.vocab)?;
    let wins = read_windows(&val, ctx, tokens)?;
    if wins.len() < 2 {
        return Err(format!("{}: only {} full windows of {ctx}", val.display(), wins.len()));
    }
    let half = wins.len() / 2;
    let (states, fwd_tps) = collect_par(&model_p, ctx, &wins, &layers, call, threads)?;
    say(&format!(
        "probe_states: {} (d {d}, {nb} blocks); {} val tokens ({} windows; the first {half} train the probes); forward {fwd_tps:.0} tokens/s",
        model_p.display(),
        wins.len() * ctx,
        wins.len()
    ));
    let toks: Vec<u32> = wins.iter().flat_map(|w| w[..ctx].iter().copied()).collect();
    let cut = half * ctx;
    let mut sets: Vec<Rows> = Vec::new();
    {
        let mut feats: Vec<Vec<f32>> = vec![tok_rows(&toks)];
        feats.extend(states);
        let b = feats.iter().map(|f| f[cut * d..].to_vec()).collect();
        feats.iter_mut().for_each(|f| f.truncate(cut * d));
        sets.push(Rows { a: feats, b });
    }
    let mut labels: [Vec<u32>; 4] = Default::default();
    for w in &wins {
        for (o, v) in labels.iter_mut().zip(text_labels(&l.bpe, w, ctx)) {
            o.extend(v);
        }
    }
    let mut tasks: Vec<Task> = TEXT_TASKS
        .iter()
        .zip(&labels)
        .map(|(name, y)| Task { name: name.to_string(), classes: 2, rows: 0, ya: y[..cut].to_vec(), yb: y[cut..].to_vec() })
        .collect();
    // the source task: the same number of windows from each source
    let mut src_note = String::new();
    if let Some(dir) = &by_source {
        let want = (per_source / ctx).max(2);
        let mut kept: Vec<(String, Vec<Vec<u32>>)> = Vec::new();
        let mut left: Vec<String> = Vec::new();
        for (name, p) in source_list(dir, a.str("--sources"))? {
            check_vocab(&p, l.cfg.vocab)?;
            let w = read_windows(&p, ctx, want * ctx)?;
            if w.len() < want {
                left.push(format!("{name} ({} windows)", w.len()));
                continue;
            }
            kept.push((name, w));
        }
        let left = if left.is_empty() { String::new() } else { format!(" Left out (fewer windows): {}.", left.join(", ")) };
        if kept.len() >= 2 {
            let hw = want / 2;
            let mut rows = Rows { a: vec![Vec::new(); feat_names.len()], b: vec![Vec::new(); feat_names.len()] };
            let mut task = Task { name: format!("source ({} classes)", kept.len()), classes: kept.len(), rows: 1, ya: Vec::new(), yb: Vec::new() };
            for (c, (name, w)) in kept.iter().enumerate() {
                let (st, tps) = collect_par(&model_p, ctx, w, &layers, call, threads)?;
                say(&format!("  source {name}: {} tokens ({tps:.0} tokens/s)", w.len() * ctx));
                let ids: Vec<u32> = w.iter().flat_map(|x| x[..ctx].iter().copied()).collect();
                let mut fs = vec![tok_rows(&ids)];
                fs.extend(st);
                for ((ra, rb), f) in rows.a.iter_mut().zip(rows.b.iter_mut()).zip(&fs) {
                    ra.extend_from_slice(&f[..hw * ctx * d]);
                    rb.extend_from_slice(&f[hw * ctx * d..]);
                }
                task.ya.extend(std::iter::repeat(c as u32).take(hw * ctx));
                task.yb.extend(std::iter::repeat(c as u32).take((w.len() - hw) * ctx));
            }
            let names: Vec<&str> = kept.iter().map(|k| k.0.as_str()).collect();
            src_note = format!("Source classes: {} ({want} windows of {ctx} from each, the first {hw} to train).{left}", names.join(", "));
            sets.push(rows);
            tasks.push(task);
        } else {
            src_note = format!("Source task skipped: fewer than 2 sources with {want} windows.{left}");
        }
    }
    // every (task, feature set) probe, side by side on the threads
    let jobs: Vec<(usize, usize)> = (0..tasks.len()).flat_map(|t| (0..feat_names.len()).map(move |f| (t, f))).collect();
    let t_fit = std::time::Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<Vec<(usize, Score)>> = par_ranges(threads, threads, |_| {
        let mut mine = Vec::new();
        loop {
            let q = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(&(t, f)) = jobs.get(q) else { break };
            let tk = &tasks[t];
            let rs = &sets[tk.rows];
            mine.push((q, fit_and_score(&rs.a[f], &tk.ya, &rs.b[f], &tk.yb, d, tk.classes, &pc)));
        }
        mine
    });
    let mut score: Vec<Option<Score>> = vec![None; jobs.len()];
    for (q, s) in results.into_iter().flatten() {
        score[q] = Some(s);
    }
    let fit_s = t_fit.elapsed().as_secs_f64();
    // the table
    let mut md = String::new();
    let _ = writeln!(md, "# A13 linear probes: {}\n", model_p.display());
    let _ = writeln!(md, "Dev val shards only ({}); no test set was read.\n", val.display());
    let _ = writeln!(
        md,
        "Multinomial logistic regression on standardized states ({} epochs of Adam, batch {}, lr {} cosine, L2 {}), trained on the first half of the windows and scored on the second half. `tok` is the token's own embedding row: what the token alone tells. Cells: test accuracy (binary tasks: and ROC AUC). Majority: the training half's most common class guessed for every test token.\n",
        pc.epochs, pc.batch, pc.lr, pc.l2
    );
    let mut head = String::from("| probe | positives | majority |");
    let mut rule = String::from("|---|---|---|");
    for f in &feat_names {
        let _ = write!(head, " {f} |");
        rule.push_str("---|");
    }
    let _ = writeln!(md, "{head}\n{rule}");
    let mut tsv = String::from("probe\tfeatures\tacc\tmajority\tauc\tbalanced_acc\ttrain_rows\ttest_rows\n");
    for (t, tk) in tasks.iter().enumerate() {
        let pos = if tk.classes == 2 { pct(tk.yb.iter().filter(|&&y| y == 1).count() as f64 / tk.yb.len().max(1) as f64) } else { format!("{} × {}", tk.classes, pct(1.0 / tk.classes as f64)) };
        let maj = score[t * feat_names.len()].as_ref().map_or(0.0, |s| s.majority);
        let mut row = format!("| {} | {pos} | {} |", tk.name, pct(maj));
        for (f, fname) in feat_names.iter().enumerate() {
            let s = score[t * feat_names.len() + f].as_ref().expect("every probe ran");
            match s.auc {
                Some(au) => {
                    let _ = write!(row, " {} ({:.3}) |", pct(s.acc), au);
                }
                None => {
                    let _ = write!(row, " {} |", pct(s.acc));
                }
            }
            let _ = writeln!(
                tsv,
                "{}\t{fname}\t{:.6}\t{:.6}\t{}\t{:.6}\t{}\t{}",
                tk.name,
                s.acc,
                s.majority,
                s.auc.map_or("".into(), |x| format!("{x:.6}")),
                s.balanced,
                tk.ya.len(),
                tk.yb.len()
            );
        }
        let _ = writeln!(md, "{row}");
    }
    let _ = writeln!(md);
    if !src_note.is_empty() {
        let _ = writeln!(md, "{src_note}\n");
    }
    let _ = writeln!(md, "Labels (from the text alone): {}.\n", LABEL_NOTE);
    let _ = writeln!(md, "Speed: forward {fwd_tps:.0} tokens/s; {} probes fitted in {fit_s:.1} s on {threads} threads; {:.1} s in all.", jobs.len(), t_all.elapsed().as_secs_f64());
    let dir = out.unwrap_or_else(|| model_p.parent().unwrap_or(Path::new(".")).join("a13-probes"));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    std::fs::write(dir.join("report.md"), &md).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("probes.tsv"), &tsv).map_err(|e| e.to_string())?;
    say(&md);
    say(&format!("probe_states: wrote {}", dir.display()));
    Ok(said)
}

const LABEL_NOTE: &str = "next_is_digit = the next token is a digit or a length mark; in_number = the token is a length mark, all digits, or a . or , between digits; sentence_end = the token ends in . ! or ? (after closing quotes), is not in a number, and what follows past any closing quotes or brackets is an end of document or white space; in_quotes = after this token the text is inside double quotes (straight quotes toggle, curly ones open and close; counted from the window's start and each end of document)";
