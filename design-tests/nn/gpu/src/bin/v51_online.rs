// TESTING / DESIGN ONLY — v51_online: v51 Part B's online test (runs/v51/prereg_v51.md: E, e-prop, and CB's extra test)
// and its talking-speed numbers. CPU only (--threads, default 8).
//
// v51_online docs    --val DIR [--len 512] [--out FILE]
//     The held-out documents: the rows of DIR/tags.tsv of source 0 (fineweb), in written order, whose text before its
//     EOS holds at least --len tokens; each is read as [EOS] + its first --len tokens. Prints (or writes) one line per
//     document: list index, tags row, start, length.
// v51_online run     --model FILE --val DIR --cond frozen|eprop|dyn|cb --from A --n N [--len 512] [--rows 20] [--lr X]
//                    [--seg 16] [--blocks 6,7] [--cb-units 12288] [--threads 8] --out DIR
//     Documents [A, A+N) of the list, --rows side by side, each from the saved weights (every change reset per
//     document); each document's second half is scored (its targets at positions len/2 .. len-1, each scored before
//     she learns from it). Conditions: frozen (no learning); eprop (src/eprop.rs, one token at a time); dyn (online
//     backprop = dynamic evaluation: plain SGD on the same weights, segments of --seg tokens, each scored and then
//     learned); cb (the model plus a zero cerebellum head of --cb-units units, its read-out learning online from her
//     error at each word; --lr is then eta and the rate is eta / mean |E_v|^2 over her word embeddings E).
//     Writes DIR/cNNNNN.tsv per chunk (list index, tags row, start, bytes, nats) and for the learning conditions
//     DIR/cNNNNN.safetensors (the chunk's changes summed over its rows: the forgetting test keeps them all). A chunk
//     already written is skipped (resumable).
// v51_online boot    --frozen DIR --runs NAME=DIR[,NAME=DIR..] [--resamples 10000] [--seed 4848]
//     BPB of every condition over the documents all of them hold, and each minus frozen with a paired bootstrap over
//     documents (95% interval; BPB is a ratio of sums, recomputed per resample); for dyn present, each condition's
//     share of dyn's gain.
// v51_online forget  --model FILE --base-val DIR --runs NAME=DIR[,..] [--tokens 65536] [--ctx 256] [--batch 32]
//                    [--cb-units 12288]
//     Forgetting: the model with every document's changes of a run kept (summed over all its chunk files) scored on
//     the first --tokens targets of DIR's windows (ctx + 1 tokens, windows with PAD left out), against the frozen
//     model on the same windows (paired bootstrap over windows).
// v51_online speed   --model FILE --val DIR [--tokens 200] [--len 512] [--blocks 6,7] [--seg 16] [--cb-units 12288]
//     Talking speed of one stream (B = 1) on the first document's tokens: frozen (the chat app's fused step and the
//     State path), e-prop, cb online and dyn (per token, a segment's cost spread over its tokens).
// v51_online talk128 [--tokens 200] [--repeats 3]
//     CPU talking speed (one-token stream, f32, the State path as v50_talk's default) of untrained d128 models of Ava
//     0.4's recipe: B, --thalamus, --cerebellum-head 4096 (and B with the fused step, for reference).
// v51_online plus    --model FILE --out FILE [--cb-units 12288]
//     The model plus a zero cerebellum head (the online test's CB start: the same logits), saved with its arch.txt
//     beside it, for the chat app (ava_live --cb-online ETA).
//
// v51 Part B's one redesign (runs/v51/prereg_v51.md, Amendment B-4): E2, dyn2, CB2.
// v51_online stream  --model FILE --val DIR --cond eprop|dyn|cb --from A --n N --streams S [--len 512] [--lr X]
//                    [--half-lives H[,H..]] [--seg 16] [--blocks 6,7] [--cb-units 12288] [--threads 8]
//                    [--frozen DIR] --out DIR
//     Documents [A, A+N) as S streams of K = N / S documents: stream s reads A + s, A + s + S, .., A + s + (K-1)S in
//     turn, so round k holds documents A + kS .. A + kS + S - 1, side by side. Each document is read as in `run` (a
//     fresh state, [EOS] + its first --len tokens, its second half scored, each word before she learns from it), but
//     the changes are not reset: they carry over to the stream's next document, fading back to the saved weights with
//     the half-life H (tokens; inf: they never fade): a learning step over n tokens multiplies them by 2^(-n / H)
//     first. The eligibility traces restart with each document (they belong to the fresh state). Several half-lives
//     (a pick grid): every stream runs once per half-life, side by side; --frozen DIR (the frozen `run` of the same
//     documents) is then required and pick.txt holds Amendment B-4's rule: the shortest half-life whose gain over
//     frozen is at least half of the grid's largest gain.
//     Writes DIR/rKK.tsv per round (list index, tags row, start, bytes, nats, lr, half-life, stream) and, after each
//     round, the session (DIR/state.safetensors + state.txt): a run that stops resumes after its last finished round,
//     bit for bit. With one half-life, at the end, DIR/kept.safetensors: each stream's changes as they act at its end
//     ([S, in, out] per plastic weight for eprop / dyn; cbh.wo [S, G, d] for cb).
// v51_online forget2 --model FILE --base-val DIR --runs NAME=DIR[,..] --out DIR [--tokens 65536] [--ctx 256]
//                    [--batch 32] [--cb-units 12288] [--blocks 6,7] [--check N] [--part I/N]
//     The redesigned forgetting test: every stream's brain at the end of its documents (the saved weights plus its
//     row of DIR/kept.safetensors) on the windows of `forget` (the same first --tokens targets), against the frozen
//     model on the same windows; per run, the mean over streams of (brain BPB - frozen BPB) with a bootstrap over
//     streams (10,000 resamples, SplitMix64 seed 4848, 95% percentile interval); bar: the mean worse by at most 0.005.
//     The blocks below the first plastic one (all of them for cb) are computed once, for every brain
//     (src/v51e.rs's trunk: the same numbers bit for bit). --check N: the first N brains of each run are also scored
//     the full way and any difference stops the command. Writes OUT/frozen.tsv and OUT/NAME/sSS.tsv (per window:
//     nats, bytes); a brain already scored is skipped (resumable). --part I/N: only the brains with (s + r) % N == I
//     (stream s of the r-th run of --runs) are scored (parts run side by side as separate processes, each with its own
//     --threads; each part holds only its own brains' changes; each part's frozen run
//     must equal OUT/frozen.tsv bit for bit). A run's summary is printed once all its brains are scored; a call
//     with every brain already scored and OUT/frozen.tsv present runs no forward at all.
// `speed` also takes --half-eprop H, --half-dyn H and --half-cb H: those learners then fade with that half-life.
// `boot` also takes --group S: then each run's minus-frozen also gets a paired bootstrap over S streams (document list
// index modulo S: one unit = a stream's documents together), reported beside the per-document one.
// v51 fast4: `stream` also takes --fused: with one row (one stream, one half-life) each token runs the fused session
//     step (eprop: `Model::fast4_eprop_step`; cb: `fast4_state_step` + `fast4_cb_learn`), the same bits as without it.
// v51_online speed4 --model FILE --val DIR --arm A [--tokens 200] [--repeats 3] [--blocks 6,7] [--half-eprop 128]
//     [--lr 0.001] [--cb-units 12288] [--half-cb 256] [--eta 0.3]: one talking-speed arm per process (A: frozen-state,
//     frozen-fused, frozen-int8, eprop-old, eprop-new, cb-old, cb-new; see `cmd_speed4`).

use ava_data::bpe::{EOS, PAD};
use ava_data::rng::SplitMix64;
use ava_data::shards::ShardSet;
use ava_gpu::arch::Arch;
use ava_gpu::eprop::{Rule, Site};
use ava_gpu::model::{Config, Kind, Model};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: v51_online docs|run|boot|forget|speed|speed4|talk128|plus|stream|forget2 [options] (see the top of src/bin/v51_online.rs)";

fn main() {
    if let Err(e) = run() {
        eprintln!("v51_online: {e}");
        std::process::exit(1);
    }
}

struct Args(Vec<String>);

impl Args {
    fn get(&self, k: &str) -> Option<String> {
        self.0.iter().position(|a| a == k).and_then(|i| self.0.get(i + 1)).cloned()
    }
    fn num(&self, k: &str, d: usize) -> Result<usize, String> {
        self.get(k).map(|v| v.parse().map_err(|_| format!("bad {k} {v:?}"))).transpose().map(|v| v.unwrap_or(d))
    }
    fn fl(&self, k: &str, d: f64) -> Result<f64, String> {
        self.get(k).map(|v| v.parse().map_err(|_| format!("bad {k} {v:?}"))).transpose().map(|v| v.unwrap_or(d))
    }
    fn path(&self, k: &str) -> Result<PathBuf, String> {
        self.get(k).map(PathBuf::from).ok_or_else(|| format!("{k} is required\n{USAGE}"))
    }
}

fn ce(e: candle_core::Error) -> String {
    e.to_string()
}

/// candle's CPU matmul (the gemm crate) sizes its blocks once per process, from the cache sizes CPUID reports on the
/// core that runs its first matmul. On a CPU with two kinds of cores (here 8 P-cores and 16 E-cores) two processes
/// can so add up in a different order, and their numbers differ in the last digits (the trials found 0.04 nats in a
/// window of 256 targets). So this thread runs one small matmul on logical CPU AVA_GEMM_CPU (default 0) first, and
/// every process uses the same blocks; then it may run anywhere again. AVA_GEMM_CPU=off leaves it to chance.
fn gemm_blocks_from_one_cpu() -> Result<(), String> {
    let cpu = match std::env::var("AVA_GEMM_CPU") {
        Ok(v) if v == "off" => return Ok(()),
        Ok(v) => v.parse::<u32>().map_err(|_| format!("AVA_GEMM_CPU={v:?}: a logical CPU number or off"))?,
        Err(_) => 0,
    };
    let warm = || -> Result<(), String> {
        let x = Tensor::ones((128, 256), DType::F32, &Device::Cpu).map_err(ce)?;
        x.matmul(&x.t().map_err(ce)?).map_err(ce).map(|_| ())
    };
    #[cfg(windows)]
    {
        extern "system" {
            fn GetCurrentThread() -> *mut std::ffi::c_void;
            fn SetThreadAffinityMask(thread: *mut std::ffi::c_void, mask: usize) -> usize;
        }
        // SAFETY: plain Win32 calls on this thread's own pseudo-handle
        let old = unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu) };
        if old == 0 {
            return Err(format!("AVA_GEMM_CPU={cpu}: this thread cannot run on that CPU"));
        }
        std::thread::sleep(std::time::Duration::from_millis(1)); // now on that CPU
        let r = warm();
        unsafe { SetThreadAffinityMask(GetCurrentThread(), old) };
        r
    }
    #[cfg(not(windows))]
    {
        let _ = cpu;
        warm()
    }
}

fn run() -> Result<(), String> {
    let a = Args(std::env::args().skip(1).collect());
    let mode = a.0.first().cloned().ok_or(USAGE)?;
    ava_gpu::diary::set_threads(a.num("--threads", 8)?.clamp(1, 8)); // before the first tensor op
    gemm_blocks_from_one_cpu()?;
    let dev = Device::Cpu;
    match mode.as_str() {
        "docs" => cmd_docs(&a),
        "run" => cmd_run(&a, &dev),
        "boot" => cmd_boot(&a),
        "forget" => cmd_forget(&a, &dev),
        "speed" => cmd_speed(&a, &dev),
        "talk128" => cmd_talk128(&a, &dev),
        "speed4" => cmd_speed4(&a, &dev),
        "plus" => cmd_plus(&a, &dev),
        "stream" => cmd_stream(&a, &dev),
        "forget2" => cmd_forget2(&a, &dev),
        _ => Err(USAGE.into()),
    }
}

// ------------------------------------------------------------------------------------------------ the documents

struct Doc {
    li: usize,
    row: usize,
    start: usize,
    len: usize,
    /// [EOS] + the first `l` tokens of the text
    toks: Vec<u32>,
}

fn load_docs(val: &Path, l: usize) -> Result<Vec<Doc>, String> {
    ava_gpu::probe::check_path(val, "--val")?;
    let set = ShardSet::open(val).map_err(|e| format!("open {}: {e}", val.display()))?;
    if set.len() != 1 {
        return Err(format!("{}: one shard expected, {} found", val.display(), set.len()));
    }
    let sh = set.load_shard(0).map_err(|e| e.to_string())?;
    let tags = std::fs::read_to_string(val.join("tags.tsv")).map_err(|e| format!("{}/tags.tsv: {e}", val.display()))?;
    let (mut out, mut row) = (Vec::new(), 0usize);
    for line in tags.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let p = |i: usize| f.get(i).and_then(|v| v.parse::<usize>().ok()).ok_or_else(|| format!("tags.tsv: bad line {line:?}"));
        let (start, len, src) = (p(0)?, p(1)?, p(2)?);
        let this = row;
        row += 1;
        if src != 0 {
            continue;
        }
        if start + len > sh.len() {
            return Err(format!("tags.tsv row {this}: {start}+{len} is past the shard's {} tokens", sh.len()));
        }
        let text = if len > 0 && sh.get(start + len - 1) == EOS { len - 1 } else { len };
        if text < l {
            continue;
        }
        let mut toks = Vec::with_capacity(l + 1);
        toks.push(EOS);
        toks.extend(sh.window(start, l));
        if toks[1..].iter().any(|&t| t == EOS || t == PAD) {
            return Err(format!("tags.tsv row {this}: EOS or PAD inside the text"));
        }
        out.push(Doc { li: out.len(), row: this, start, len, toks });
    }
    Ok(out)
}

fn cmd_docs(a: &Args) -> Result<(), String> {
    let val = a.path("--val")?;
    let l = a.num("--len", 512)?;
    let docs = load_docs(&val, l)?;
    let mut s = format!("# v51 online documents: {} source 0 (fineweb) rows with >= {l} text tokens, written order\n# list\ttags_row\tstart\tlen\n", val.display());
    for d in &docs {
        s += &format!("{}\t{}\t{}\t{}\n", d.li, d.row, d.start, d.len);
    }
    match a.get("--out") {
        Some(p) => std::fs::write(&p, &s).map_err(|e| format!("{p}: {e}"))?,
        None => print!("{s}"),
    }
    eprintln!("v51_online docs: {} documents with >= {l} text tokens", docs.len());
    Ok(())
}

// ------------------------------------------------------------------------------------------------ models

/// 0.4b-style model from FILE (its arch.txt beside it) plus `extra` block flags; the new weights keep their init (a
/// zero cerebellum read-out). Returns the lean model and its VarMap.
fn load_plus(path: &Path, extra: &str, dev: &Device) -> Result<(Config, VarMap, Model), String> {
    if extra.trim().is_empty() {
        let (cfg, vm, m) = ava_gpu::ckpt::load_auto_lean(path, Kind::D, 4096, dev)?;
        m.set_track_activity(false);
        return Ok((cfg, vm, m));
    }
    let t = ava_gpu::ckpt::read(path)?;
    let cfg0 = ava_gpu::ckpt::detect(&t, Kind::D, 4096)?;
    let (ar0, _) = ava_gpu::ckpt::arch_beside(path)?.ok_or("the model needs its arch.txt beside it")?;
    let mut ar = Arch::from_flags(&format!("{} {extra}", ar0.flags()))?;
    ar.top = t.contains_key("top_emb");
    let cfg = cfg0.with_arch(ar)?;
    let vm = VarMap::new();
    let _build = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, dev)).map_err(ce)?;
    let (_copied, kept) = ava_gpu::ckpt::init_from(&vm, path)?;
    if kept != ["cbh.wo"] {
        return Err(format!("expected only the cerebellum read-out to be new, kept {kept:?}"));
    }
    let m = ava_gpu::ckpt::lean(cfg, &vm, dev)?;
    m.set_track_activity(false);
    Ok((cfg, vm, m))
}

/// The lean model over `vm`'s weights with `add` added (names of the VarMap).
fn model_plus(cfg: Config, vm: &VarMap, add: &HashMap<String, Tensor>, dev: &Device) -> Result<Model, String> {
    let mut ts: HashMap<String, Tensor> = vm.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.as_tensor().detach())).collect();
    for (k, d) in add {
        let w = ts.get(k).ok_or_else(|| format!("{k}: no such weight"))?;
        let nw = w.add(d).map_err(ce)?;
        ts.insert(k.clone(), nw);
    }
    let m = Model::new(cfg, VarBuilder::from_tensors(ts, DType::F32, dev)).map_err(ce)?;
    m.set_track_activity(false);
    Ok(m)
}

/// `plus`: the model plus a zero cerebellum head, saved with its arch.txt beside it.
fn cmd_plus(a: &Args, dev: &Device) -> Result<(), String> {
    let (model, out) = (a.path("--model")?, a.path("--out")?);
    let (cfg, vm, _) = load_plus(&model, &format!("--cerebellum-head {}", a.num("--cb-units", 12288)?), dev)?;
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    vm.save(&out).map_err(ce)?;
    ava_gpu::ckpt::write_arch_beside(&out, &cfg.arch)?;
    println!("v51_online plus: {} + a zero cerebellum head -> {} (arch: {})", model.display(), out.display(), cfg.arch.flags());
    Ok(())
}

fn blocks_arg(a: &Args, layers: usize) -> Result<Vec<usize>, String> {
    let s = a.get("--blocks").unwrap_or_else(|| format!("{},{}", layers - 2, layers - 1));
    s.split(',').map(|x| x.parse::<usize>().map_err(|_| format!("bad --blocks {s:?}"))).collect()
}

/// mean |E_v|^2 over the word embeddings (the cerebellum rate's scale)
fn emb_sq(vm: &VarMap) -> Result<f64, String> {
    let e = vm.data().lock().unwrap().get("tok_emb").ok_or("no tok_emb")?.as_tensor().clone();
    let v = e.dim(0).map_err(ce)?;
    Ok(e.sqr().and_then(|x| x.sum_all()).and_then(|x| x.to_scalar::<f32>()).map_err(ce)? as f64 / v as f64)
}

// ------------------------------------------------------------------------------------------------ the conditions

#[derive(Clone, Copy, PartialEq, Debug)]
enum Cond {
    Frozen,
    EProp,
    Dyn,
    Cb,
}

impl Cond {
    fn parse(s: &str) -> Result<Cond, String> {
        Ok(match s {
            "frozen" => Cond::Frozen,
            "eprop" => Cond::EProp,
            "dyn" => Cond::Dyn,
            "cb" => Cond::Cb,
            _ => return Err(format!("--cond {s:?}: frozen, eprop, dyn or cb")),
        })
    }
}

/// -log p(target) in f64 for every row of logits [n, V].
fn nll(logits: &Tensor, tg: &[u32]) -> candle_core::Result<Vec<f64>> {
    let t = Tensor::from_vec(tg.to_vec(), (tg.len(), 1), logits.device())?;
    candle_nn::ops::log_softmax(&logits.detach().to_dtype(DType::F64)?, D::Minus1)?.gather(&t, D::Minus1)?.neg()?.flatten_all()?.to_vec1::<f64>()
}

struct Opts {
    cond: Cond,
    l: usize,
    seg: usize,
    blocks: Vec<usize>,
    /// v51 fast4: stream with one row runs the fused session step (`fast4_eprop_step`; `fast4_state_step` + `fast4_cb_learn`)
    fused: bool,
}

/// One chunk of rows side by side, each a document and its own learning rate: each row's second-half nats, and the
/// changes summed over the rows.
fn run_chunk(m: &Model, rows: &[(&Doc, f64)], o: &Opts, dev: &Device) -> candle_core::Result<(Vec<f64>, HashMap<String, Tensor>)> {
    let (r, l, half) = (rows.len(), o.l, o.l / 2);
    let lrs: Vec<f64> = rows.iter().map(|x| x.1).collect();
    let mut nats = vec![0f64; r];
    let ids_at = |p0: usize, n: usize| -> candle_core::Result<Tensor> {
        Tensor::from_vec(rows.iter().flat_map(|d| d.0.toks[p0..p0 + n].iter().copied()).collect::<Vec<u32>>(), (r, n), dev)
    };
    let tg_at = |p0: usize, n: usize| -> Vec<u32> { rows.iter().flat_map(|d| d.0.toks[p0 + 1..p0 + n + 1].iter().copied()).collect() };
    let mut score = |v: &[f64], p0: usize, n: usize| {
        for (k, x) in v.iter().enumerate() {
            if p0 + k % n >= half {
                nats[k / n] += x;
            }
        }
    };
    let mut sums = HashMap::new();
    let mut st = m.state(r)?;
    match o.cond {
        Cond::Frozen => {
            let piece = 64;
            for p0 in (0..l).step_by(piece) {
                let n = piece.min(l - p0);
                let lg = m.forward_with_state(&ids_at(p0, n)?, &mut st)?;
                score(&nll(&lg, &tg_at(p0, n))?, p0, n);
            }
        }
        Cond::EProp | Cond::Dyn => {
            let (seg, rule) = if o.cond == Cond::EProp { (1, Rule::EProp) } else { (o.seg, Rule::Backprop) };
            m.plastic_begin(r, &o.blocks, &Site::ALL)?;
            m.plastic_probes(true);
            for p0 in (0..l).step_by(seg) {
                let n = seg.min(l - p0);
                let lg = m.forward_with_state(&ids_at(p0, n)?, &mut st)?;
                let tg = tg_at(p0, n);
                score(&nll(&lg, &tg)?, p0, n);
                let loss = candle_nn::loss::cross_entropy(&lg, &Tensor::from_vec(tg, r * n, dev)?)?.affine((r * n) as f64, 0.0)?;
                m.plastic_learn_rows(&loss.backward()?, &lrs, rule)?;
                st.detach_all();
            }
            for &i in &o.blocks {
                for s in Site::ALL {
                    if let Some(t) = m.plastic_delta_sum(i, s)? {
                        sums.insert(s.var_name(i), t);
                    }
                }
            }
            m.plastic_end();
        }
        Cond::Cb => {
            m.cb_online_begin(r, lrs[0] as f32)?;
            for (j, &x) in lrs.iter().enumerate() {
                m.cb_online_set_lr(j, x as f32);
            }
            for p in 0..l {
                let lg = m.forward_with_state(&ids_at(p, 1)?, &mut st)?;
                let tg = tg_at(p, 1);
                score(&nll(&lg, &tg)?, p, 1);
                m.cb_learn(&lg, &tg)?;
            }
            if let (Some(s), Some(w)) = (m.cb_online_sum(), m.cb_weight()) {
                sums.insert("cbh.wo".to_string(), Tensor::from_vec(s, w.dims(), dev)?);
            }
            m.cb_online_end();
        }
    }
    Ok((nats, sums))
}

fn cmd_run(a: &Args, dev: &Device) -> Result<(), String> {
    let model = a.path("--model")?;
    let val = a.path("--val")?;
    let out = a.path("--out")?;
    let cond = Cond::parse(&a.get("--cond").ok_or("--cond is required")?)?;
    let l = a.num("--len", 512)?;
    let (from, n, nrows) = (a.num("--from", 0)?, a.num("--n", 200)?, a.num("--rows", 20)?.max(1));
    // --lrs a,b,..: a rate grid (every document once per rate, side by side); else --lr
    let grid: Vec<f64> = match a.get("--lrs") {
        Some(s) => s.split(',').map(|x| x.parse::<f64>().map_err(|_| format!("bad --lrs {s:?}"))).collect::<Result<_, _>>()?,
        None => vec![a.fl("--lr", 0.0)?],
    };
    let units = a.num("--cb-units", 12288)?;
    let docs = load_docs(&val, l)?;
    if from + n > docs.len() {
        return Err(format!("documents {from}..{} asked, the list holds {}", from + n, docs.len()));
    }
    let bl = ava_gpu::eval::byte_lengths_n(&ava_data::bpe::Bpe::load(Path::new(ava_gpu::DEFAULT_BPE)).map_err(|e| e.to_string())?, 4112);
    let (cfg, vm, m) = load_plus(&model, if cond == Cond::Cb { format!("--cerebellum-head {units}") } else { String::new() }.as_str(), dev)?;
    if cfg.vocab != bl.len() {
        return Err(format!("vocab {} vs {} byte lengths", cfg.vocab, bl.len()));
    }
    let scale = if cond == Cond::Cb { 1.0 / emb_sq(&vm)? } else { 1.0 };
    let o = Opts { cond, l, seg: a.num("--seg", 16)?.max(1), blocks: blocks_arg(a, cfg.layers)?, fused: false };
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    println!(
        "v51_online run: {cond:?}, documents {from}..{} ({nrows} rows side by side), len {l} (second half scored), lr {grid:?}{}, seg {}, blocks {:?}, {} threads -> {}",
        from + n,
        if cond == Cond::Cb { format!(" (eta; rate = eta x {scale:.4e})") } else { String::new() },
        o.seg,
        o.blocks,
        candle_core::utils::get_num_threads(),
        out.display()
    );
    let t_all = Instant::now();
    let mut done_rows = 0usize;
    let mut secs_run = 0f64;
    let list: Vec<(&Doc, f64)> = grid.iter().flat_map(|&lr| docs[from..from + n].iter().map(move |d| (d, lr))).collect();
    for (ci, ch) in list.chunks(nrows).enumerate() {
        let tag = format!("c{:05}", if grid.len() == 1 { ch[0].0.li } else { ci * nrows });
        let tsv = out.join(format!("{tag}.tsv"));
        if tsv.is_file() {
            continue;
        }
        let t0 = Instant::now();
        let rows: Vec<(&Doc, f64)> = ch.iter().map(|&(d, lr)| (d, lr * scale)).collect();
        let (nats, sums) = run_chunk(&m, &rows, &o, dev).map_err(ce)?;
        let secs = t0.elapsed().as_secs_f64();
        if !sums.is_empty() && grid.len() == 1 {
            let p = out.join(format!("{tag}.safetensors"));
            candle_core::safetensors::save(&sums, &p).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        let mut s = format!("# {cond:?} rows {} len {l} seconds {secs:.2} row_tokens_per_sec {:.1}\n", ch.len(), (ch.len() * l) as f64 / secs);
        let mut bnat = (0f64, 0u64);
        for (&(d, lr), x) in ch.iter().zip(&nats) {
            let bytes: u64 = d.toks[l / 2 + 1..l + 1].iter().map(|&t| bl[t as usize] as u64).sum();
            s += &format!("{}\t{}\t{}\t{bytes}\t{x:.6}\t{lr}\n", d.li, d.row, d.start);
            bnat.0 += x;
            bnat.1 += bytes;
        }
        let tmp = out.join(format!("{tag}.tsv.tmp"));
        std::fs::write(&tmp, s).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &tsv).map_err(|e| e.to_string())?;
        done_rows += ch.len();
        secs_run += secs;
        println!(
            "  chunk {tag}: {} rows in {secs:.1} s ({:.1} row-tokens/s), chunk BPB {:.5}",
            ch.len(),
            (ch.len() * l) as f64 / secs,
            bnat.0 / std::f64::consts::LN_2 / bnat.1 as f64
        );
    }
    println!(
        "v51_online run: done ({done_rows} rows run now in {secs_run:.1} s, {:.1} row-tokens/s; wall {:.1} s)",
        (done_rows * l) as f64 / secs_run.max(1e-9),
        t_all.elapsed().as_secs_f64()
    );
    if grid.len() > 1 {
        // the grid's result: BPB per rate over its documents, from every chunk file of the folder
        let mut per: Vec<(f64, f64, u64)> = grid.iter().map(|&g| (g, 0.0, 0)).collect();
        for e in std::fs::read_dir(&out).map_err(|e| e.to_string())?.flatten() {
            if e.path().extension().is_some_and(|x| x == "tsv") {
                for line in std::fs::read_to_string(e.path()).map_err(|e| e.to_string())?.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
                    let f: Vec<&str> = line.split('\t').collect();
                    let (Some(b), Some(x), Some(lr)) = (f.get(3).and_then(|v| v.parse::<u64>().ok()), f.get(4).and_then(|v| v.parse::<f64>().ok()), f.get(5).and_then(|v| v.parse::<f64>().ok())) else { continue };
                    if let Some(p) = per.iter_mut().find(|p| p.0 == lr) {
                        p.1 += x;
                        p.2 += b;
                    }
                }
            }
        }
        let mut best = (f64::NAN, f64::INFINITY);
        let mut txt = String::new();
        for (lr, x, b) in &per {
            let v = x / std::f64::consts::LN_2 / *b as f64;
            txt += &format!("lr {lr}: BPB {v:.5} over {b} bytes\n");
            if v < best.1 {
                best = (*lr, v);
            }
        }
        txt += &format!("best_lr={}\n", best.0);
        print!("{txt}");
        std::fs::write(out.join("pick.txt"), txt).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ results

/// The chunk files of a run folder: list index -> (bytes, nats).
fn read_run(dir: &Path) -> Result<HashMap<usize, (u64, f64)>, String> {
    let mut out = HashMap::new();
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "tsv") {
            for line in std::fs::read_to_string(&p).map_err(|e| e.to_string())?.lines() {
                if line.starts_with('#') || line.is_empty() {
                    continue;
                }
                let f: Vec<&str> = line.split('\t').collect();
                let bad = || format!("{}: bad line {line:?}", p.display());
                let li: usize = f.first().and_then(|v| v.parse().ok()).ok_or_else(bad)?;
                let bytes: u64 = f.get(3).and_then(|v| v.parse().ok()).ok_or_else(bad)?;
                let nats: f64 = f.get(4).and_then(|v| v.parse().ok()).ok_or_else(bad)?;
                out.insert(li, (bytes, nats));
            }
        }
    }
    Ok(out)
}

fn runs_arg(a: &Args) -> Result<Vec<(String, PathBuf)>, String> {
    let s = a.get("--runs").ok_or("--runs NAME=DIR[,NAME=DIR..] is required")?;
    s.split(',').map(|kv| kv.split_once('=').map(|(k, v)| (k.to_string(), PathBuf::from(v))).ok_or_else(|| format!("bad --runs item {kv:?}"))).collect()
}

/// Paired bootstrap of the BPB difference (a - b) over units (nats a, nats b, bytes): (point, lo, hi) of 95%.
fn boot_diff(units: &[(f64, f64, u64)], resamples: usize, seed: u64) -> (f64, f64, f64) {
    let bpb = |idx: &mut dyn Iterator<Item = usize>| {
        let (mut x, mut y, mut b) = (0f64, 0f64, 0u64);
        for i in idx {
            x += units[i].0;
            y += units[i].1;
            b += units[i].2;
        }
        (x - y) / std::f64::consts::LN_2 / b.max(1) as f64
    };
    let point = bpb(&mut (0..units.len()));
    let mut rng = SplitMix64::new(seed);
    let n = units.len() as u64;
    let mut v: Vec<f64> = (0..resamples).map(|_| bpb(&mut (0..units.len()).map(|_| rng.below(n) as usize))).collect();
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let q = |p: f64| v[((p * resamples as f64) as usize).min(resamples - 1)];
    (point, q(0.025), q(0.975))
}

fn cmd_boot(a: &Args) -> Result<(), String> {
    let frozen = read_run(&a.path("--frozen")?)?;
    let runs: Vec<(String, HashMap<usize, (u64, f64)>)> = runs_arg(a)?.into_iter().map(|(k, d)| read_run(&d).map(|r| (k, r))).collect::<Result<_, _>>()?;
    let (res, seed) = (a.num("--resamples", 10000)?, a.num("--seed", 4848)? as u64);
    let mut keys: Vec<usize> = frozen.keys().copied().filter(|k| runs.iter().all(|(_, r)| r.contains_key(k))).collect();
    keys.sort();
    let ln2 = std::f64::consts::LN_2;
    let bytes: u64 = keys.iter().map(|k| frozen[k].0).sum();
    let fb = keys.iter().map(|k| frozen[k].1).sum::<f64>() / ln2 / bytes as f64;
    println!("documents {} ({} bytes scored); frozen BPB {fb:.5}", keys.len(), bytes);
    let gain_of = |r: &HashMap<usize, (u64, f64)>| fb - keys.iter().map(|k| r[k].1).sum::<f64>() / ln2 / bytes as f64;
    let dyn_gain = runs.iter().find(|(k, _)| k == "dyn").map(|(_, r)| gain_of(r));
    for (name, r) in &runs {
        let units: Vec<(f64, f64, u64)> = keys.iter().map(|k| (r[k].1, frozen[k].1, frozen[k].0)).collect();
        if keys.iter().any(|k| r[k].0 != frozen[k].0) {
            return Err(format!("{name}: bytes differ from frozen's (not the same documents)"));
        }
        let (d, lo, hi) = boot_diff(&units, res, seed);
        let better = keys.iter().filter(|k| r[k].1 < frozen[k].1).count();
        let share = dyn_gain.filter(|g| *g != 0.0 && name != "dyn").map(|g| format!(", {:.1}% of dyn's gain", 100.0 * gain_of(r) / g)).unwrap_or_default();
        println!(
            "{name}: BPB {:.5}; minus frozen {d:+.5} [95% {lo:+.5}, {hi:+.5}] ({res} resamples, seed {seed}); better on {better} of {} documents{share}; bar (hi < 0): {}",
            fb + d,
            keys.len(),
            if hi < 0.0 { "PASS" } else { "FAIL" }
        );
        if let Some(g) = a.get("--group") {
            // v51 B-4: one unit per stream (list index modulo S), its documents' nats and bytes summed
            let s: usize = g.parse().map_err(|_| format!("bad --group {g:?}"))?;
            let mut su = vec![(0f64, 0f64, 0u64); s.max(1)];
            for (k, u) in keys.iter().zip(&units) {
                let x = &mut su[k % s.max(1)];
                x.0 += u.0;
                x.1 += u.1;
                x.2 += u.2;
            }
            su.retain(|x| x.2 > 0);
            let (d2, lo2, hi2) = boot_diff(&su, res, seed);
            let better2 = su.iter().filter(|x| x.0 < x.1).count();
            println!("{name} by stream: minus frozen {d2:+.5} [95% {lo2:+.5}, {hi2:+.5}] over {} streams ({res} resamples, seed {seed}); better on {better2} of {} streams", su.len(), su.len());
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ forgetting

/// Per window: (nats, bytes) of every target; `batch` windows side by side, read with the State path in pieces of 64
/// tokens (the online test's own path; faster on the CPU than one window-long forward).
fn score_windows(m: &Model, wins: &[Vec<u32>], batch: usize, bl: &[u32], dev: &Device) -> candle_core::Result<Vec<(f64, u64)>> {
    let mut out = Vec::with_capacity(wins.len());
    for ch in wins.chunks(batch) {
        let (r, t) = (ch.len(), ch[0].len() - 1);
        let mut nats = vec![0f64; r];
        let mut st = m.state(r)?;
        for p0 in (0..t).step_by(64) {
            let n = 64.min(t - p0);
            let ids: Vec<u32> = ch.iter().flat_map(|w| w[p0..p0 + n].iter().copied()).collect();
            let tg: Vec<u32> = ch.iter().flat_map(|w| w[p0 + 1..p0 + n + 1].iter().copied()).collect();
            let lg = m.forward_with_state(&Tensor::from_vec(ids, (r, n), dev)?, &mut st)?;
            for (k, x) in nll(&lg, &tg)?.iter().enumerate() {
                nats[k / n] += x;
            }
        }
        for (w, x) in ch.iter().zip(nats) {
            out.push((x, w[1..].iter().map(|&y| bl[y as usize] as u64).sum()));
        }
    }
    Ok(out)
}

fn sum_changes(dir: &Path, dev: &Device) -> Result<(HashMap<String, Tensor>, usize), String> {
    let mut acc: HashMap<String, Tensor> = HashMap::new();
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "safetensors")).collect();
    files.sort();
    for p in &files {
        for (k, t) in candle_core::safetensors::load(p, dev).map_err(|e| format!("{}: {e}", p.display()))? {
            let nt = match acc.get(&k) {
                Some(s) => s.add(&t).map_err(ce)?,
                None => t,
            };
            acc.insert(k, nt);
        }
    }
    Ok((acc, files.len()))
}

fn cmd_forget(a: &Args, dev: &Device) -> Result<(), String> {
    let model = a.path("--model")?;
    let base = a.path("--base-val")?;
    ava_gpu::probe::check_path(&base, "--base-val")?;
    let (tokens, ctx, batch, units) = (a.num("--tokens", 65536)?, a.num("--ctx", 256)?, a.num("--batch", 32)?, a.num("--cb-units", 12288)?);
    let bl = ava_gpu::eval::byte_lengths_n(&ava_data::bpe::Bpe::load(Path::new(ava_gpu::DEFAULT_BPE)).map_err(|e| e.to_string())?, 4112);
    let set = ShardSet::open(&base).map_err(|e| format!("open {}: {e}", base.display()))?;
    let mut wins = Vec::new();
    let mut have = 0usize;
    for w in set.iter_sequential(ctx) {
        if have >= tokens {
            break;
        }
        if w.contains(&PAD) || w.len() != ctx + 1 {
            continue;
        }
        have += ctx;
        wins.push(w);
    }
    let (cfg, vm, frozen) = load_plus(&model, "", dev)?;
    let t0 = Instant::now();
    let fz = score_windows(&frozen, &wins, batch, &bl, dev).map_err(ce)?;
    let secs = t0.elapsed().as_secs_f64();
    let ln2 = std::f64::consts::LN_2;
    let bytes: u64 = fz.iter().map(|x| x.1).sum();
    let fb = fz.iter().map(|x| x.0).sum::<f64>() / ln2 / bytes as f64;
    println!("forgetting: {} windows of {ctx} ({have} targets, {bytes} bytes) of {}; frozen BPB {fb:.5} ({:.0} tok/s)", wins.len(), base.display(), have as f64 / secs);
    for (name, dir) in runs_arg(a)? {
        let (add, files) = sum_changes(&dir, dev)?;
        let m = if add.contains_key("cbh.wo") {
            let (ccfg, cvm, _) = load_plus(&model, &format!("--cerebellum-head {units}"), dev)?;
            model_plus(ccfg, &cvm, &add, dev)?
        } else {
            model_plus(cfg, &vm, &add, dev)?
        };
        let rms: Vec<String> = {
            let mut k: Vec<&String> = add.keys().collect();
            k.sort();
            k.iter().map(|k| format!("{k} {:.2e}", add[*k].sqr().and_then(|x| x.mean_all()).and_then(|x| x.to_scalar::<f32>()).map(|x| x.sqrt()).unwrap_or(f32::NAN))).collect()
        };
        let sc = score_windows(&m, &wins, batch, &bl, dev).map_err(ce)?;
        let units_v: Vec<(f64, f64, u64)> = sc.iter().zip(&fz).map(|(x, f)| (x.0, f.0, f.1)).collect();
        let (d, lo, hi) = boot_diff(&units_v, 10000, 4848);
        println!(
            "{name}: every change kept ({files} chunk files summed; rms {}): BPB {:.5}, minus frozen {d:+.5} [95% {lo:+.5}, {hi:+.5}]; bar (worse by at most 0.005): {}",
            rms.join(", "),
            fb + d,
            if d <= 0.005 { "PASS" } else { "FAIL" }
        );
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ speed

fn cmd_speed(a: &Args, dev: &Device) -> Result<(), String> {
    let model = a.path("--model")?;
    let val = a.path("--val")?;
    let (n, l, units) = (a.num("--tokens", 200)?, a.num("--len", 512)?, a.num("--cb-units", 12288)?);
    let docs = load_docs(&val, l.max(n + 1))?;
    let doc = docs.last().ok_or("no document")?; // the list's last: never one of the scored documents
    let toks = &doc.toks[..n + 1];
    let (cfg, vm, m) = load_plus(&model, "", dev)?;
    let blocks = blocks_arg(a, cfg.layers)?;
    let seg = a.num("--seg", 16)?;
    // v51 B-4: the learners' half-lives (None: they do not fade, as before)
    let half_of = |k: &str| -> Result<Option<f64>, String> { a.get(k).map(|v| parse_half(&v)).transpose() };
    let (he, hd, hc) = (half_of("--half-eprop")?, half_of("--half-dyn")?, half_of("--half-cb")?);
    let fades = |h: Option<f64>| h.map(|h| format!(", fading, half-life {} tokens", half_str(h))).unwrap_or_default();
    let e = |e: candle_core::Error| e.to_string();
    let per_tok = |f: &mut dyn FnMut() -> Result<(), String>| -> Result<f64, String> {
        f()?; // warm-up
        let t0 = Instant::now();
        f()?;
        Ok(t0.elapsed().as_secs_f64() / n as f64)
    };
    let mut lines = Vec::new();
    for fused in [true, false] {
        m.set_fused(fused);
        let s = per_tok(&mut || {
            let mut st = m.stream().map_err(e)?;
            for &t in &toks[..n] {
                m.step(&mut st, t).map_err(e)?;
            }
            Ok(())
        })?;
        lines.push((if fused { "frozen, fused step (chat app)" } else { "frozen, State path" }.to_string(), s));
    }
    m.set_fused(false);
    let one = |t: u32| Tensor::new(&[[t]], dev);
    let s = per_tok(&mut || {
        m.plastic_begin(1, &blocks, &Site::ALL).map_err(e)?;
        if let Some(h) = he {
            m.plastic_set_half_lives(&[h]).map_err(e)?;
        }
        m.plastic_probes(true);
        let mut st = m.state(1).map_err(e)?;
        for j in 0..n {
            let lg = m.forward_with_state(&one(toks[j]).map_err(e)?, &mut st).map_err(e)?;
            let loss = candle_nn::loss::cross_entropy(&lg, &Tensor::new(&[toks[j + 1]], dev).map_err(e)?).map_err(e)?;
            m.plastic_learn(&loss.backward().map_err(e)?, 1e-4, Rule::EProp).map_err(e)?;
            st.detach_all();
        }
        m.plastic_end();
        Ok(())
    })?;
    lines.push((format!("e-prop (blocks {blocks:?}, 4 sites{})", fades(he)), s));
    let s = per_tok(&mut || {
        m.plastic_begin(1, &blocks, &Site::ALL).map_err(e)?;
        if let Some(h) = hd {
            m.plastic_set_half_lives(&[h]).map_err(e)?;
        }
        m.plastic_probes(true);
        let mut st = m.state(1).map_err(e)?;
        for p0 in (0..n).step_by(seg) {
            let k = seg.min(n - p0);
            let lg = m.forward_with_state(&Tensor::from_vec(toks[p0..p0 + k].to_vec(), (1, k), dev).map_err(e)?, &mut st).map_err(e)?;
            let loss = candle_nn::loss::cross_entropy(&lg, &Tensor::from_vec(toks[p0 + 1..p0 + k + 1].to_vec(), k, dev).map_err(e)?).map_err(e)?.affine(k as f64, 0.0).map_err(e)?;
            m.plastic_learn(&loss.backward().map_err(e)?, 1e-4, Rule::Backprop).map_err(e)?;
            st.detach_all();
        }
        m.plastic_end();
        Ok(())
    })?;
    lines.push((format!("online backprop (segments of {seg}{})", fades(hd)), s));
    drop(m);
    let (_c, cvm, mc) = load_plus(&model, &format!("--cerebellum-head {units}"), dev)?;
    let rate = 0.3 / emb_sq(&cvm)?;
    let s = per_tok(&mut || {
        mc.cb_online_begin(1, rate as f32).map_err(e)?;
        if let Some(h) = hc {
            mc.cb_online_set_half_life(0, h).map_err(e)?;
        }
        let mut st = mc.state(1).map_err(e)?;
        for j in 0..n {
            let lg = mc.forward_with_state(&one(toks[j]).map_err(e)?, &mut st).map_err(e)?;
            mc.cb_learn(&lg, &[toks[j + 1]]).map_err(e)?;
        }
        mc.cb_online_end();
        Ok(())
    })?;
    lines.push((format!("cerebellum online ({units} units{})", fades(hc)), s));
    let s = per_tok(&mut || {
        let mut st = mc.state(1).map_err(e)?;
        for &t in &toks[..n] {
            mc.forward_with_state(&one(t).map_err(e)?, &mut st).map_err(e)?;
        }
        Ok(())
    })?;
    lines.push((format!("cerebellum head, not learning ({units} units), State path"), s));
    println!("talking speed, one stream, {n} tokens, {} threads, model {} (d {}, {} layers):", candle_core::utils::get_num_threads(), model.display(), cfg.d, cfg.layers);
    for (k, s) in lines {
        println!("  {k}: {:.1} tokens/s ({:.2} ms/token)", 1.0 / s, s * 1e3);
    }
    let _ = vm;
    Ok(())
}

/// v51 fast4: one talking-speed arm per process (`speed4 --arm A`), one row, --tokens N of the list's last document:
/// frozen-state (`forward_with_state`), frozen-fused (the f32 fused step, `fast4_state_step`), frozen-int8 (the chat
/// app's int8 fused step, `step`); eprop-old / eprop-new (E2: --blocks, 4 sites, --half-eprop 128, --lr 0.001; the
/// State path's token as `stream` runs it, or `fast4_eprop_step`); cb-old / cb-new (CB2: --cb-units 12288, --half-cb
/// 256, --eta 0.3; `forward_with_state` + `cb_learn`, or `fast4_state_step` + `fast4_cb_learn`). The scoring (nll) is
/// left out of every arm. A warm-up pass, then --repeats timed passes (their median; the session restarts each pass).
fn cmd_speed4(a: &Args, dev: &Device) -> Result<(), String> {
    let model = a.path("--model")?;
    let val = a.path("--val")?;
    let arm = a.get("--arm").ok_or("--arm frozen-state|frozen-fused|frozen-int8|eprop-old|eprop-new|cb-old|cb-new")?;
    let (n, l, units, reps) = (a.num("--tokens", 200)?, a.num("--len", 512)?, a.num("--cb-units", 12288)?, a.num("--repeats", 3)?.max(1));
    let docs = load_docs(&val, l.max(n + 1))?;
    let doc = docs.last().ok_or("no document")?; // the list's last: never one of the scored documents
    let toks = &doc.toks[..n + 1];
    let cb = arm.starts_with("cb");
    let (cfg, vm, m) = load_plus(&model, &if cb { format!("--cerebellum-head {units}") } else { String::new() }, dev)?;
    let blocks = blocks_arg(a, cfg.layers)?;
    let he = parse_half(&a.get("--half-eprop").unwrap_or_else(|| "128".into()))?;
    let hc = parse_half(&a.get("--half-cb").unwrap_or_else(|| "256".into()))?;
    let (lr, eta) = (a.fl("--lr", 0.001)?, a.fl("--eta", 0.3)?);
    let rate = if cb { eta / emb_sq(&vm)? } else { 0.0 };
    let one = |t: u32| Tensor::new(&[[t]], dev);
    let new = arm.ends_with("-new") || arm == "frozen-fused";
    m.set_fast4(new);
    if arm == "frozen-int8" {
        m.set_int8(true).map_err(ce)?;
        m.set_fused(true);
    }
    // one pass: (seconds of its N tokens, tokens the fused step ran)
    let pass = || -> candle_core::Result<(f64, usize)> {
        let mut nf = 0usize;
        match arm.as_str() {
            "frozen-state" | "frozen-fused" => {
                let mut st = m.state(1)?;
                let t0 = Instant::now();
                for &t in &toks[..n] {
                    if new {
                        if m.fast4_state_step(&mut st, t)?.is_some() {
                            nf += 1;
                            continue;
                        }
                    }
                    m.forward_with_state(&one(t)?, &mut st)?;
                }
                Ok((t0.elapsed().as_secs_f64(), nf))
            }
            "frozen-int8" => {
                let mut st = m.stream()?;
                let t0 = Instant::now();
                for &t in &toks[..n] {
                    m.step(&mut st, t)?;
                }
                Ok((t0.elapsed().as_secs_f64(), if m.fused_blocker().is_none() { n } else { 0 }))
            }
            "eprop-old" | "eprop-new" => {
                m.plastic_begin(1, &blocks, &Site::ALL)?;
                m.plastic_set_half_lives(&[he])?;
                m.plastic_new_documents()?;
                m.plastic_probes(true);
                let mut st = m.state(1)?;
                let t0 = Instant::now();
                for j in 0..n {
                    if new {
                        if m.fast4_eprop_step(&mut st, toks[j], toks[j + 1], lr)?.is_some() {
                            nf += 1;
                            continue;
                        }
                    }
                    let lg = m.forward_with_state(&one(toks[j])?, &mut st)?;
                    let loss = candle_nn::loss::cross_entropy(&lg, &Tensor::new(&[toks[j + 1]], dev)?)?.affine(1.0, 0.0)?;
                    m.plastic_learn_rows(&loss.backward()?, &[lr], Rule::EProp)?;
                    st.detach_all();
                }
                let s = t0.elapsed().as_secs_f64();
                m.plastic_end();
                Ok((s, nf))
            }
            "cb-old" | "cb-new" => {
                m.cb_online_begin(1, rate as f32)?;
                m.cb_online_set_lr(0, rate as f32);
                m.cb_online_set_half_life(0, hc)?;
                let mut st = m.state(1)?;
                let t0 = Instant::now();
                for j in 0..n {
                    if new {
                        if let Some(lg) = m.fast4_state_step(&mut st, toks[j])? {
                            m.fast4_cb_learn(&lg.flatten_all()?.to_vec1::<f32>()?, toks[j + 1])?;
                            nf += 1;
                            continue;
                        }
                    }
                    let lg = m.forward_with_state(&one(toks[j])?, &mut st)?;
                    m.cb_learn(&lg, &[toks[j + 1]])?;
                }
                let s = t0.elapsed().as_secs_f64();
                m.cb_online_end();
                Ok((s, nf))
            }
            _ => candle_core::bail!("--arm {arm:?}: frozen-state, frozen-fused, frozen-int8, eprop-old, eprop-new, cb-old or cb-new"),
        }
    };
    pass().map_err(ce)?; // warm-up
    let mut v = Vec::with_capacity(reps);
    let mut nf = 0;
    for _ in 0..reps {
        let (s, f) = pass().map_err(ce)?;
        v.push(s);
        nf = f;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let s = v[v.len() / 2];
    println!(
        "v51_online speed4 {arm}: {:.1} tokens/s ({:.2} ms/token), median of {reps} passes of {n} tokens, fused steps {nf} of {n}, {} threads, d {} x {} layers{}",
        n as f64 / s,
        1e3 * s / n as f64,
        candle_core::utils::get_num_threads(),
        cfg.d,
        cfg.layers,
        match arm.as_str() {
            "eprop-old" | "eprop-new" => format!(", blocks {blocks:?}, half-life {}, lr {lr}", half_str(he)),
            "cb-old" | "cb-new" => format!(", {units} units, half-life {}, eta {eta}", half_str(hc)),
            _ => String::new(),
        }
    );
    Ok(())
}

fn cmd_talk128(a: &Args, dev: &Device) -> Result<(), String> {
    let (n, reps) = (a.num("--tokens", 200)?, a.num("--repeats", 3)?.max(1));
    const RECIPE: &str = "--head-size 64 --timescales fixed --gates channel --value-residual --kwta 1/4 --mlp-act relu2 --mem-momentum 0.5 --resid-fix --residual dense --lif-init logspaced --lif-scale --spikes int2 --alif --homeostasis 0.1";
    let mut rng = SplitMix64::new(51);
    let toks: Vec<u32> = (0..n).map(|_| 258 + rng.below(3800) as u32).collect();
    println!("CPU talking speed, untrained d128 x 4 layers (vocab 4112), Ava 0.4's recipe, one-token stream, {n} tokens, median of {reps}, {} threads:", candle_core::utils::get_num_threads());
    for (name, extra, fused) in [("B (fused step, reference)", "", true), ("B", "", false), ("TH --thalamus", "--thalamus", false), ("CB --cerebellum-head 4096", "--cerebellum-head 4096", false), ("TH+CB", "--thalamus --cerebellum-head 4096", false)] {
        let ar = Arch::from_flags(&format!("{RECIPE} {extra}"))?;
        let cfg = Config::with_kind(Kind::D, 4112, 128, 4, 4096).with_arch(ar)?;
        let vm = VarMap::new();
        let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, dev)).map_err(ce)?;
        m.set_track_activity(false);
        m.set_fused(fused);
        let pass = || -> Result<f64, String> {
            let mut st = m.stream().map_err(ce)?;
            let t0 = Instant::now();
            for &t in &toks {
                m.step(&mut st, t).map_err(ce)?;
            }
            Ok(t0.elapsed().as_secs_f64())
        };
        pass()?;
        let mut v: Vec<f64> = (0..reps).map(|_| pass()).collect::<Result<_, _>>()?;
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let s = v[v.len() / 2];
        println!("  {name}: {:.1} tokens/s ({:.0} us/token){}", n as f64 / s, 1e6 * s / n as f64, if fused { format!(" [{}]", m.fused_blocker().unwrap_or("fused step applies")) } else { String::new() });
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ v51 B-4: streams

/// A half-life argument: "inf" (never fades) or tokens > 0.
fn parse_half(s: &str) -> Result<f64, String> {
    if s == "inf" {
        return Ok(f64::INFINITY);
    }
    match s.parse::<f64>() {
        Ok(h) if h > 0.0 && h.is_finite() => Ok(h),
        _ => Err(format!("bad half-life {s:?} (tokens > 0, or inf)")),
    }
}

fn half_str(h: f64) -> String {
    if h.is_infinite() {
        "inf".into()
    } else {
        format!("{h}")
    }
}

/// One round of a stream run: every row reads its document from a fresh state with the session's changes as they
/// are (carried from the stream's last document), each word scored before she learns from it; the same calls as
/// `run_chunk`'s, without the session's begin and end. Each row's second-half nats, and how many tokens the fused session step ran (v51 fast4, --fused).
fn read_round(m: &Model, rows: &[(&Doc, f64)], o: &Opts, dev: &Device) -> candle_core::Result<(Vec<f64>, usize)> {
    let (r, l, half) = (rows.len(), o.l, o.l / 2);
    let lrs: Vec<f64> = rows.iter().map(|x| x.1).collect();
    let mut nats = vec![0f64; r];
    let ids_at = |p0: usize, n: usize| -> candle_core::Result<Tensor> {
        Tensor::from_vec(rows.iter().flat_map(|d| d.0.toks[p0..p0 + n].iter().copied()).collect::<Vec<u32>>(), (r, n), dev)
    };
    let tg_at = |p0: usize, n: usize| -> Vec<u32> { rows.iter().flat_map(|d| d.0.toks[p0 + 1..p0 + n + 1].iter().copied()).collect() };
    let mut score = |v: &[f64], p0: usize, n: usize| {
        for (k, x) in v.iter().enumerate() {
            if p0 + k % n >= half {
                nats[k / n] += x;
            }
        }
    };
    let mut st = m.state(r)?;
    let mut nf = 0usize; // v51 fast4: tokens the fused session step ran
    let one = o.fused && r == 1;
    match o.cond {
        Cond::Frozen => candle_core::bail!("stream: frozen has no session (use run)"),
        Cond::EProp | Cond::Dyn => {
            let (seg, rule) = if o.cond == Cond::EProp { (1, Rule::EProp) } else { (o.seg, Rule::Backprop) };
            m.plastic_new_documents()?; // the traces restart with the fresh state; the changes carry on
            m.plastic_probes(true);
            for p0 in (0..l).step_by(seg) {
                if one && seg == 1 {
                    let (tok, next) = (rows[0].0.toks[p0], rows[0].0.toks[p0 + 1]);
                    if let Some((lg, _rms)) = m.fast4_eprop_step(&mut st, tok, next, lrs[0])? {
                        score(&nll(&lg, &[next])?, p0, 1);
                        nf += 1;
                        continue;
                    }
                }
                let n = seg.min(l - p0);
                let lg = m.forward_with_state(&ids_at(p0, n)?, &mut st)?;
                let tg = tg_at(p0, n);
                score(&nll(&lg, &tg)?, p0, n);
                let loss = candle_nn::loss::cross_entropy(&lg, &Tensor::from_vec(tg, r * n, dev)?)?.affine((r * n) as f64, 0.0)?;
                m.plastic_learn_rows(&loss.backward()?, &lrs, rule)?;
                st.detach_all();
            }
        }
        Cond::Cb => {
            for p in 0..l {
                let tg = tg_at(p, 1);
                if one {
                    if let Some(lg) = m.fast4_state_step(&mut st, rows[0].0.toks[p])? {
                        score(&nll(&lg, &tg)?, p, 1);
                        m.fast4_cb_learn(&lg.flatten_all()?.to_vec1::<f32>()?, tg[0])?;
                        nf += 1;
                        continue;
                    }
                }
                let lg = m.forward_with_state(&ids_at(p, 1)?, &mut st)?;
                score(&nll(&lg, &tg)?, p, 1);
                m.cb_learn(&lg, &tg)?;
            }
        }
    }
    Ok((nats, nf))
}

/// The session as saved after a round: eprop / dyn, every plastic weight's changes [R, in, out] by name; cb, the
/// online table's stored rows [R, G·d] and scales [R] (f64), so a resumed run goes on bit for bit.
fn session_tensors(m: &Model, cond: Cond, rows: usize, gd: usize, dev: &Device) -> Result<HashMap<String, Tensor>, String> {
    let mut out = HashMap::new();
    if cond == Cond::Cb {
        let (tab, sc) = m
            .cb_online_with(|o| {
                let mut tab = Vec::with_capacity(rows * gd);
                let mut sc = Vec::with_capacity(rows);
                for r in 0..rows {
                    let (t, s) = o.raw(r);
                    if t.is_empty() {
                        tab.extend(std::iter::repeat(0f32).take(gd));
                    } else {
                        tab.extend_from_slice(t);
                    }
                    sc.push(s);
                }
                (tab, sc)
            })
            .ok_or("stream: no online table")?;
        out.insert("cb.delta".to_string(), Tensor::from_vec(tab, (rows, gd), dev).map_err(ce)?);
        out.insert("cb.scale".to_string(), Tensor::from_vec(sc, rows, dev).map_err(ce)?);
    } else {
        for ((i, s), d) in m.plastic_deltas().map_err(ce)? {
            out.insert(s.var_name(i), d);
        }
    }
    Ok(out)
}

fn session_restore(m: &Model, cond: Cond, rows: usize, gd: usize, ts: &HashMap<String, Tensor>, blocks: &[usize]) -> Result<(), String> {
    if cond == Cond::Cb {
        let tab: Vec<f32> = ts.get("cb.delta").ok_or("state: no cb.delta")?.flatten_all().and_then(|t| t.to_vec1()).map_err(ce)?;
        let sc: Vec<f64> = ts.get("cb.scale").ok_or("state: no cb.scale")?.to_vec1().map_err(ce)?;
        if tab.len() != rows * gd || sc.len() != rows {
            return Err(format!("state: cb table of {} values and {} scales for {rows} rows of {gd}", tab.len(), sc.len()));
        }
        m.cb_online_with(|o| {
            for r in 0..rows {
                o.set_raw(r, tab[r * gd..(r + 1) * gd].to_vec(), sc[r]);
            }
        })
        .ok_or("stream: no online table")?;
    } else {
        for &i in blocks {
            for s in Site::ALL {
                let t = ts.get(&s.var_name(i)).ok_or_else(|| format!("state: no {}", s.var_name(i)))?;
                m.plastic_set_delta(i, s, t).map_err(ce)?;
            }
        }
    }
    Ok(())
}

fn cmd_stream(a: &Args, dev: &Device) -> Result<(), String> {
    let model = a.path("--model")?;
    let val = a.path("--val")?;
    let out = a.path("--out")?;
    let cond = Cond::parse(&a.get("--cond").ok_or("--cond is required")?)?;
    if cond == Cond::Frozen {
        return Err("stream: frozen learns nothing; score it with run".into());
    }
    let l = a.num("--len", 512)?;
    let (from, n, ns) = (a.num("--from", 0)?, a.num("--n", 200)?, a.num("--streams", 40)?.max(1));
    if n % ns != 0 {
        return Err(format!("stream: {n} documents do not split into {ns} streams"));
    }
    let kr = n / ns;
    let lr = a.fl("--lr", 0.0)?;
    let halves: Vec<f64> = a.get("--half-lives").unwrap_or_else(|| "inf".into()).split(',').map(parse_half).collect::<Result<_, _>>()?;
    let units = a.num("--cb-units", 12288)?;
    let docs = load_docs(&val, l)?;
    if from + n > docs.len() {
        return Err(format!("documents {from}..{} asked, the list holds {}", from + n, docs.len()));
    }
    if halves.len() > 1 && a.get("--frozen").is_none() {
        return Err("stream: a half-life grid needs --frozen DIR (the frozen run of the same documents)".into());
    }
    let bl = ava_gpu::eval::byte_lengths_n(&ava_data::bpe::Bpe::load(Path::new(ava_gpu::DEFAULT_BPE)).map_err(|e| e.to_string())?, 4112);
    let (cfg, vm, m) = load_plus(&model, if cond == Cond::Cb { format!("--cerebellum-head {units}") } else { String::new() }.as_str(), dev)?;
    if cfg.vocab != bl.len() {
        return Err(format!("vocab {} vs {} byte lengths", cfg.vocab, bl.len()));
    }
    let scale = if cond == Cond::Cb { 1.0 / emb_sq(&vm)? } else { 1.0 };
    let o = Opts { cond, l, seg: a.num("--seg", 16)?.max(1), blocks: blocks_arg(a, cfg.layers)?, fused: a.0.iter().any(|x| x == "--fused") };
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    // row j: half-life halves[j / S], stream j % S
    let nrows = halves.len() * ns;
    let row_half: Vec<f64> = (0..nrows).map(|j| halves[j / ns]).collect();
    let rate = lr * scale;
    let gd = units * cfg.d;
    println!(
        "v51_online stream: {cond:?}, documents {from}..{} as {ns} streams of {kr} ({nrows} rows side by side), len {l} (second half scored), lr {lr}{}, half-lives {:?} tokens, seg {}, blocks {:?}, {} threads -> {}",
        from + n,
        if cond == Cond::Cb { format!(" (eta; rate = eta x {scale:.4e})") } else { String::new() },
        halves.iter().map(|&h| half_str(h)).collect::<Vec<_>>(),
        o.seg,
        o.blocks,
        candle_core::utils::get_num_threads(),
        out.display()
    );
    match cond {
        Cond::EProp | Cond::Dyn => {
            m.plastic_begin(nrows, &o.blocks, &Site::ALL).map_err(ce)?;
            m.plastic_set_half_lives(&row_half).map_err(ce)?;
        }
        _ => {
            // as run_chunk: the table's rate, then each row's (the same f32 values)
            m.cb_online_begin(nrows, rate as f32).map_err(ce)?;
            for j in 0..nrows {
                m.cb_online_set_lr(j, rate as f32);
                m.cb_online_set_half_life(j, row_half[j]).map_err(ce)?;
            }
        }
    }
    if o.fused {
        // v51 fast4: the fused session step (one row; f32, int8 off as here)
        m.set_fast4(true);
        let why = if cond == Cond::Cb {
            m.fused_blocker()
        } else {
            m.plastic_probes(true);
            m.fast4_eprop_blocker()
        };
        println!("  --fused: {}", why.unwrap_or("the fused session step applies"));
    }
    let rpath = |k: usize| out.join(format!("r{k:02}.tsv"));
    let r0 = (0..kr).take_while(|&k| rpath(k).is_file()).count();
    let (stp, stt) = (out.join("state.safetensors"), out.join("state.txt"));
    let finished = if halves.len() == 1 { out.join("kept.safetensors").is_file() } else { out.join("pick.txt").is_file() };
    if r0 > 0 && !finished && (r0 < kr || halves.len() == 1) {
        let said = std::fs::read_to_string(&stt).map_err(|e| format!("{}: {e} (rounds written but no session to resume)", stt.display()))?;
        if said.trim() != format!("rounds {r0}") {
            return Err(format!("{}: {:?}, but {r0} rounds are written", stt.display(), said.trim()));
        }
        let ts = candle_core::safetensors::load(&stp, dev).map_err(|e| format!("{}: {e}", stp.display()))?;
        session_restore(&m, cond, nrows, gd, &ts, &o.blocks)?;
        println!("  resumed after round {r0} ({})", stp.display());
    }
    let t_all = Instant::now();
    let mut secs_run = 0f64;
    for k in r0..kr {
        let t0 = Instant::now();
        let rows: Vec<(&Doc, f64)> = (0..nrows).map(|j| (&docs[from + j % ns + ns * k], rate)).collect();
        let (nats, nf) = read_round(&m, &rows, &o, dev).map_err(ce)?;
        let secs = t0.elapsed().as_secs_f64();
        secs_run += secs;
        let mut s = format!("# {cond:?} stream round {k} of {kr}, rows {nrows} len {l} seconds {secs:.2} row_tokens_per_sec {:.1}\n", (nrows * l) as f64 / secs);
        let mut bnat = (0f64, 0u64);
        for (j, (&(d, _), x)) in rows.iter().zip(&nats).enumerate() {
            let bytes: u64 = d.toks[l / 2 + 1..l + 1].iter().map(|&t| bl[t as usize] as u64).sum();
            s += &format!("{}\t{}\t{}\t{bytes}\t{x:.6}\t{lr}\t{}\t{}\n", d.li, d.row, d.start, half_str(row_half[j]), j % ns);
            bnat.0 += x;
            bnat.1 += bytes;
        }
        // the session first, then the round's file: a round counts as done only with its session saved
        if k + 1 < kr || halves.len() == 1 {
            let tmp = out.join("state.safetensors.tmp");
            candle_core::safetensors::save(&session_tensors(&m, cond, nrows, gd, dev)?, &tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
            std::fs::rename(&tmp, &stp).map_err(|e| e.to_string())?;
            std::fs::write(&stt, format!("rounds {}\n", k + 1)).map_err(|e| e.to_string())?;
        }
        let tt = out.join(format!("r{k:02}.tsv.tmp"));
        std::fs::write(&tt, s).map_err(|e| e.to_string())?;
        std::fs::rename(&tt, rpath(k)).map_err(|e| e.to_string())?;
        println!("  round {k}: {nrows} rows in {secs:.1} s ({:.1} row-tokens/s), round BPB {:.5}{}", (nrows * l) as f64 / secs, bnat.0 / std::f64::consts::LN_2 / bnat.1 as f64, if o.fused { format!(", fused steps {nf} of {}", nrows * l) } else { String::new() });
    }
    let done_rows = (kr - r0) * nrows;
    println!(
        "v51_online stream: done ({done_rows} rows run now in {secs_run:.1} s, {:.1} row-tokens/s; wall {:.1} s)",
        (done_rows * l) as f64 / secs_run.max(1e-9),
        t_all.elapsed().as_secs_f64()
    );
    if halves.len() == 1 {
        let kp = out.join("kept.safetensors");
        if !kp.is_file() {
            // each stream's changes as they act at its end
            let mut kept = HashMap::new();
            if cond == Cond::Cb {
                let tabs = m.cb_online_with(|o| (0..nrows).map(|r| o.table(r).unwrap_or_else(|| vec![0.0; gd])).collect::<Vec<_>>()).ok_or("no online table")?;
                kept.insert("cbh.wo".to_string(), Tensor::from_vec(tabs.concat(), (nrows, units, cfg.d), dev).map_err(ce)?);
            } else {
                for ((i, s), d) in m.plastic_deltas().map_err(ce)? {
                    kept.insert(s.var_name(i), d);
                }
            }
            let tmp = out.join("kept.safetensors.tmp");
            candle_core::safetensors::save(&kept, &tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
            std::fs::rename(&tmp, &kp).map_err(|e| e.to_string())?;
            println!("  kept: {} ({nrows} streams)", kp.display());
        }
    } else {
        // the pick (Amendment B-4): per half-life, its BPB and its gain over frozen on the same documents
        let frozen = read_run(&a.path("--frozen")?)?;
        let mut per: Vec<(f64, f64, f64, u64)> = halves.iter().map(|&h| (h, 0.0, 0.0, 0)).collect(); // (h, nats, frozen nats, bytes)
        for k in 0..kr {
            for line in std::fs::read_to_string(rpath(k)).map_err(|e| e.to_string())?.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
                let f: Vec<&str> = line.split('\t').collect();
                let bad = || format!("{}: bad line {line:?}", rpath(k).display());
                let li: usize = f.first().and_then(|v| v.parse().ok()).ok_or_else(bad)?;
                let b: u64 = f.get(3).and_then(|v| v.parse().ok()).ok_or_else(bad)?;
                let x: f64 = f.get(4).and_then(|v| v.parse().ok()).ok_or_else(bad)?;
                let h = parse_half(f.get(6).ok_or_else(bad)?)?;
                let (fb, fx) = *frozen.get(&li).ok_or_else(|| format!("--frozen has no document {li}"))?;
                if fb != b {
                    return Err(format!("document {li}: {b} bytes here, {fb} in --frozen"));
                }
                let p = per.iter_mut().find(|p| p.0 == h).ok_or_else(bad)?;
                p.1 += x;
                p.2 += fx;
                p.3 += b;
            }
        }
        let ln2 = std::f64::consts::LN_2;
        let gains: Vec<f64> = per.iter().map(|p| (p.2 - p.1) / ln2 / p.3 as f64).collect();
        let best = gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut order: Vec<usize> = (0..per.len()).collect();
        order.sort_by(|&x, &y| per[x].0.partial_cmp(&per[y].0).expect("half-lives compare"));
        let pick = if best > 0.0 {
            *order.iter().find(|&&j| gains[j] >= best / 2.0).expect("the best one qualifies")
        } else {
            *order.iter().find(|&&j| gains[j] == best).expect("the best one")
        };
        let mut txt = String::new();
        for &j in &order {
            let p = &per[j];
            txt += &format!("half-life {}: BPB {:.5} over {} bytes (frozen {:.5}), gain {:+.5}\n", half_str(p.0), p.1 / ln2 / p.3 as f64, p.3, p.2 / ln2 / p.3 as f64, gains[j]);
        }
        txt += &format!("largest gain {best:+.5}; rule: the shortest half-life with a gain of at least half of it ({:+.5})\n", best / 2.0);
        txt += &format!("picked_half={}\n", half_str(per[pick].0));
        print!("{txt}");
        std::fs::write(out.join("pick.txt"), txt).map_err(|e| e.to_string())?;
    }
    match cond {
        Cond::Cb => m.cb_online_end(),
        _ => m.plastic_end(),
    }
    // the session is in kept.safetensors (main) or no longer needed (grid): its last copy goes
    if (halves.len() > 1 && out.join("pick.txt").is_file()) || out.join("kept.safetensors").is_file() {
        let _ = std::fs::remove_file(&stp);
        std::fs::write(&stt, format!("rounds {kr} (done; session removed)\n")).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ v51 B-4: forgetting

fn write_scores(p: &Path, sc: &[(f64, u64)]) -> Result<(), String> {
    let mut s = String::from("# window\tnats\tbytes\n");
    for (w, (x, b)) in sc.iter().enumerate() {
        s += &format!("{w}\t{x:e}\t{b}\n");
    }
    let tmp = p.with_extension("tsv.tmp");
    std::fs::write(&tmp, s).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, p).map_err(|e| e.to_string())
}

fn read_scores(p: &Path) -> Result<Vec<(f64, u64)>, String> {
    let mut v = Vec::new();
    for line in std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        let bad = || format!("{}: bad line {line:?}", p.display());
        v.push((f.get(1).and_then(|x| x.parse().ok()).ok_or_else(bad)?, f.get(2).and_then(|x| x.parse().ok()).ok_or_else(bad)?));
    }
    Ok(v)
}

/// How two score lists differ: windows, the largest difference in nats.
fn score_diff(x: &[(f64, u64)], y: &[(f64, u64)]) -> String {
    if x.len() != y.len() {
        return format!("{} windows against {}", x.len(), y.len());
    }
    let (mut n, mut m) = (0usize, 0f64);
    for (p, q) in x.iter().zip(y) {
        if p.0.to_bits() != q.0.to_bits() || p.1 != q.1 {
            n += 1;
            m = m.max((p.0 - q.0).abs());
        }
    }
    format!("{n} of {} windows differ, by at most {m:.3e} nats", x.len())
}

fn same_scores(x: &[(f64, u64)], y: &[(f64, u64)]) -> bool {
    x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.0.to_bits() == q.0.to_bits() && p.1 == q.1)
}

fn cmd_forget2(a: &Args, dev: &Device) -> Result<(), String> {
    let model = a.path("--model")?;
    let base = a.path("--base-val")?;
    let out = a.path("--out")?;
    ava_gpu::probe::check_path(&base, "--base-val")?;
    let (tokens, ctx, batch, units, check) = (a.num("--tokens", 65536)?, a.num("--ctx", 256)?, a.num("--batch", 32)?, a.num("--cb-units", 12288)?, a.num("--check", 0)?);
    // --part I/N: this process scores the streams s with s % N == I (parts run side by side, each with its own threads)
    let (pi, pn) = match a.get("--part") {
        None => (0usize, 1usize),
        Some(p) => {
            let bad = || format!("--part takes I/N with I < N, not {p:?}");
            let (x, y) = p.split_once('/').ok_or_else(bad)?;
            let (x, y): (usize, usize) = (x.parse().map_err(|_| bad())?, y.parse().map_err(|_| bad())?);
            if y == 0 || x >= y {
                return Err(bad());
            }
            (x, y)
        }
    };
    let part = if pn > 1 { format!(" part {pi}/{pn}") } else { String::new() };
    let bl = ava_gpu::eval::byte_lengths_n(&ava_data::bpe::Bpe::load(Path::new(ava_gpu::DEFAULT_BPE)).map_err(|e| e.to_string())?, 4112);
    let set = ShardSet::open(&base).map_err(|e| format!("open {}: {e}", base.display()))?;
    let mut wins = Vec::new();
    let mut have = 0usize;
    for w in set.iter_sequential(ctx) {
        if have >= tokens {
            break;
        }
        if w.contains(&PAD) || w.len() != ctx + 1 {
            continue;
        }
        have += ctx;
        wins.push(w);
    }
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let (cfg, vm, _) = load_plus(&model, "", dev)?;
    let blocks = blocks_arg(a, cfg.layers)?;
    let k_mid = *blocks.iter().min().ok_or("--blocks is empty")?;
    let brain_file = |name: &str, s: usize| out.join(name).join(format!("s{s:02}.tsv"));
    // the runs: cb or not, how many streams, and the brains this part has still to score, with their changes (each
    // its own copy: the rest of the file is not held); run r's stream s is part (s + r) % N's
    let mine_of = |r: usize, s: usize| (s + r) % pn == pi;
    let mut runs = Vec::new();
    for (r, (name, dir)) in runs_arg(a)?.into_iter().enumerate() {
        let kp = dir.join("kept.safetensors");
        let kept = candle_core::safetensors::load(&kp, dev).map_err(|e| format!("{}: {e}", kp.display()))?;
        let is_cb = kept.contains_key("cbh.wo");
        let ns = kept.values().next().ok_or_else(|| format!("{}: empty", kp.display()))?.dim(0).map_err(ce)?;
        let mut todo: Vec<(usize, HashMap<String, Tensor>)> = Vec::new();
        for s in (0..ns).filter(|&s| mine_of(r, s) && !brain_file(&name, s).is_file()) {
            let add = kept.iter().map(|(k, t)| t.get(s).and_then(|x| x.force_contiguous()).map(|x| (k.clone(), x))).collect::<candle_core::Result<HashMap<_, _>>>().map_err(ce)?;
            todo.push((s, add));
        }
        runs.push((r, name, dir, is_cb, ns, todo));
    }
    let need_mid = runs.iter().any(|r| !r.3 && !r.5.is_empty());
    let need_top = runs.iter().any(|r| r.3 && !r.5.is_empty());
    let fpath = out.join("frozen.tsv");
    let (mut rec_mid, mut rec_top) = (None, None);
    let fz = if need_mid || need_top || !fpath.is_file() {
        // the frozen model (the brains' common base, built as theirs are), recording the trunk at the cuts this part
        // needs. The first part to finish its frozen run writes frozen.tsv; every part's frozen run must equal it bit
        // for bit. A run that does not is printed and run again (at most 3 times): in the trials one frozen run in
        // about 25 came out different in some windows, a cause in the forward this redesign does not change.
        let mut tries = 0;
        loop {
            tries += 1;
            let fzm = model_plus(cfg, &vm, &HashMap::new(), dev)?;
            let mut cuts = Vec::new();
            if need_mid {
                cuts.push(k_mid);
            }
            if need_top {
                cuts.push(cfg.layers);
            }
            if !cuts.is_empty() {
                fzm.trunk_record(&cuts).map_err(ce)?;
            }
            let t0 = Instant::now();
            let fz = score_windows(&fzm, &wins, batch, &bl, dev).map_err(ce)?;
            let secs = t0.elapsed().as_secs_f64();
            let mut recs = fzm.trunk_take_records();
            drop(fzm);
            if !fpath.is_file() {
                let tmp = out.join(format!("frozen.{}.tsv", std::process::id()));
                write_scores(&tmp, &fz)?;
                if std::fs::rename(&tmp, &fpath).is_err() {
                    let _ = std::fs::remove_file(&tmp);
                }
            }
            let old = read_scores(&fpath)?;
            if !same_scores(&old, &fz) {
                println!("  frozen run{part}, try {tries}: differs from {} ({}); run again", fpath.display(), score_diff(&old, &fz));
                if tries == 3 {
                    return Err(format!("{}: 3 frozen runs differ from it", fpath.display()));
                }
                continue;
            }
            if need_top {
                rec_top = Some(recs.pop().ok_or("no record")?);
            }
            if need_mid {
                rec_mid = Some(recs.pop().ok_or("no record")?);
            }
            println!(
                "forgetting (v51 B-4){part}: {} windows of {ctx} ({have} targets) of {}; frozen run {:.0} tok/s, the same numbers as {}{} (trunk records: blocks 0..{} {:.0} MB, all {} blocks {:.0} MB)",
                wins.len(),
                base.display(),
                have as f64 / secs,
                fpath.display(),
                if tries > 1 { format!(" at try {tries}") } else { String::new() },
                k_mid - 1,
                rec_mid.as_ref().map_or(0, |r| r.bytes()) as f64 / 1e6,
                cfg.layers,
                rec_top.as_ref().map_or(0, |r| r.bytes()) as f64 / 1e6
            );
            break fz;
        }
    } else {
        read_scores(&fpath)?
    };
    if fz.len() != wins.len() {
        return Err(format!("{}: {} windows, not {}", fpath.display(), fz.len(), wins.len()));
    }
    let ln2 = std::f64::consts::LN_2;
    let bytes: u64 = fz.iter().map(|x| x.1).sum();
    let fnats: f64 = fz.iter().map(|x| x.0).sum();
    let fb = fnats / ln2 / bytes as f64;
    if pn == 1 {
        println!("forgetting (v51 B-4): {} windows of {ctx} ({have} targets, {bytes} bytes) of {}; frozen BPB {fb:.5}", wins.len(), base.display());
    }
    for (r, name, dir, is_cb, ns, todo) in runs {
        let rd = out.join(&name);
        std::fs::create_dir_all(&rd).map_err(|e| format!("{}: {e}", rd.display()))?;
        let others = (0..ns).filter(|&s| !mine_of(r, s) && !brain_file(&name, s).is_file()).count();
        if todo.is_empty() && others > 0 {
            println!("{name}{part}: nothing to score; {} of {ns} brains present", ns - others);
            continue;
        }
        let kp = dir.join("kept.safetensors");
        let n_now = todo.len();
        let (bcfg, bvm) = if is_cb {
            let (c, v, _) = load_plus(&model, &format!("--cerebellum-head {units}"), dev)?;
            (c, v)
        } else {
            (cfg, vm.clone())
        };
        let tr = Instant::now();
        for (s, add) in todo {
            let brain = model_plus(bcfg, &bvm, &add, dev)?;
            let slot = if is_cb { &mut rec_top } else { &mut rec_mid };
            let mut replay = || -> Result<Vec<(f64, u64)>, String> {
                brain.trunk_replay(slot.take().expect("the record is back")).map_err(ce)?;
                let res = score_windows(&brain, &wins, batch, &bl, dev);
                *slot = Some(brain.trunk_take().ok_or("the record went missing")?);
                res.map_err(ce)
            };
            let mut sc = replay()?;
            if s < check {
                // the full way, every block computed by the brain itself: the same numbers bit for bit (a check that
                // differs is printed and both are run again, at most 3 times, as the frozen run)
                let mut tries = 0;
                loop {
                    tries += 1;
                    let full = score_windows(&brain, &wins, batch, &bl, dev).map_err(ce)?;
                    if same_scores(&sc, &full) {
                        println!("  check {name} stream {s}: trunk replay vs the full forward: 0 of {} windows differ{}", sc.len(), if tries > 1 { format!(" (try {tries})") } else { String::new() });
                        break;
                    }
                    println!("  check {name} stream {s}, try {tries}: trunk replay vs the full forward: {}", score_diff(&sc, &full));
                    if tries == 3 {
                        return Err(format!("{name} stream {s}: the trunk replay differs from the full forward 3 times"));
                    }
                    sc = replay()?;
                }
            }
            write_scores(&brain_file(&name, s), &sc)?;
        }
        let el = tr.elapsed().as_secs_f64();
        let now = if n_now == 0 { String::new() } else { format!("; {n_now} brains scored now in {el:.0} s, {:.1} s per brain", el / n_now as f64) };
        if others > 0 {
            println!("{name}{part}: {} of {ns} brains present{now}", ns - others);
            continue;
        }
        // every brain is scored: the summary over streams
        let kept = candle_core::safetensors::load(&kp, dev).map_err(|e| format!("{}: {e}", kp.display()))?;
        let mut diffs = Vec::with_capacity(ns);
        let mut rms_sum: HashMap<String, f64> = HashMap::new();
        for s in 0..ns {
            for (k, t) in &kept {
                let t = t.get(s).map_err(ce)?;
                *rms_sum.entry(k.clone()).or_default() += t.sqr().and_then(|x| x.mean_all()).and_then(|x| x.to_scalar::<f32>()).map_err(ce)? as f64;
            }
            let p = brain_file(&name, s);
            let sc = read_scores(&p)?;
            if sc.len() != fz.len() || sc.iter().zip(&fz).any(|(x, f)| x.1 != f.1) {
                return Err(format!("{}: not the frozen run's windows", p.display()));
            }
            diffs.push((sc.iter().map(|x| x.0).sum::<f64>() - fnats) / ln2 / bytes as f64);
        }
        let mean = diffs.iter().sum::<f64>() / ns as f64;
        let mut rng = SplitMix64::new(4848);
        let res = 10000usize;
        let mut v: Vec<f64> = (0..res).map(|_| (0..ns).map(|_| diffs[rng.below(ns as u64) as usize]).sum::<f64>() / ns as f64).collect();
        v.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
        let q = |p: f64| v[((p * res as f64) as usize).min(res - 1)];
        let (lo, hi) = (q(0.025), q(0.975));
        let (mn, mx) = diffs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
        let mut rk: Vec<&String> = rms_sum.keys().collect();
        rk.sort();
        let rms: Vec<String> = rk.iter().map(|k| format!("{k} {:.2e}", (rms_sum[*k] / ns as f64).sqrt())).collect();
        println!(
            "{name}: {ns} stream brains, each with the changes it kept at its end (rms over streams: {}): mean BPB {:.5}, minus frozen {mean:+.5} [95% {lo:+.5}, {hi:+.5}] (bootstrap over streams, {res} resamples, seed 4848); per stream {mn:+.5} .. {mx:+.5}; bar (mean worse by at most 0.005): {}{now}",
            rms.join(", "),
            fb + mean,
            if mean <= 0.005 { "PASS" } else { "FAIL" },
        );
    }
    Ok(())
}
