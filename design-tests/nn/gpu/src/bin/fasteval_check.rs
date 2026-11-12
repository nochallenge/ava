// TESTING / DESIGN ONLY — fasteval_check (v52, runs/v52/fasteval.md): the scoring tools' plain and fused forwards on
// the same items, in one process, one model load: every number the adoption rule reads, and the seconds of each.
//
// fasteval_check --model FILE [--tasks val,blimp,mc,num,long] [--bpe FILE] [--val DIR].. [--eval-tokens 1000000]
//                [--blimp-dir DIR] [--blimp-limit N] [--mc TSV:NORM].. [--batch-tokens 16384] [--num TSV] [--long-shards DIR]
//                [--books 100] [--len 16384] [--cut 256] [--buckets 256,1024,4096,16384] [--out FILE] [--paths both|plain2|plain-shape|fused2|plain1]
//                [--cpu [--threads 8]] [--trunc N: the --len books cut to their first N tokens]
//
// Each task runs the tool's own scoring code (eval::evaluate, eval::seq_nll, mc::grade) or a copy of the tool's few lines
// around it (BLiMP's and the number test's sentences, long_eval's books and windows; marked "as TOOL" below), with the
// model's switches as that tool sets them (val_bpb and blimp_gpu: activity tracked, inference off; mc_eval, num_eval
// and long_eval: tracking off, inference on), first on the reference forward, then with eval::set_fused(true):
//   val    each --val set (default the v41-mix and v42-mix val sets), ctx 256, batch 16, --eval-tokens targets: BPB
//          to 9 decimals, the difference;
//   blimp  every pair of --blimp-dir (blimp_gpu's batching: one seq_nll per phenomenon, batch 256): the pairs whose
//          decision (good NLL < bad NLL) differs, with both margins (bad - good, nats);
//   mc     each --mc set (default qa_dev with norm byte and v42's dev set with norm total; mc::grade, --batch-tokens):
//          the items whose pick differs, with both margins (second - best score) and right/wrong;
//   num    the number test (num_eval's sentences and rule, batch 512): items whose pick or verdict differs, margins;
//   long   long_eval's books (--long-shards, --books, --len) in its three conditions (full: one pass; cut: windows of
//          --cut; no-eng: the engram off), BPB per bucket to 9 decimals, the difference.
// The near-tie line: decisions that differ while |margin| < 1e-4 under the reference forward. Seconds: the scoring
// calls only (the model is loaded once); a warm-up pass on both forwards runs before the first task.
// --out appends one TSV line per number: model, task, set, what, plain, fused, diff.

use ava_data::bpe::{Bpe, EOS};
use ava_data::numbers::insert_marks;
use ava_data::shards::ShardSet;
use ava_gpu::eval::{byte_lengths_n, evaluate, seq_nll, set_fused};
use ava_gpu::mc;
use ava_gpu::model::{Kind, Model};
use candle_core::{Device, Tensor, D};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: fasteval_check --model FILE [--tasks val,blimp,mc,num,long] [--bpe FILE] [--val DIR].. [--eval-tokens N]\n\
    \x20                     [--blimp-dir DIR] [--blimp-limit N] [--mc TSV:NORM].. [--batch-tokens N] [--num TSV] [--long-shards DIR] [--books N]\n\
    \x20                     [--len N] [--cut N] [--buckets A,B,..] [--out FILE] [--paths both|plain2|plain-shape|fused2|plain1] [--trunc N] [--cpu [--threads 8]]";

const TIE: f64 = 1e-4;

fn main() {
    if let Err(e) = run() {
        eprintln!("fasteval_check: {e}");
        std::process::exit(1);
    }
}

struct Out {
    model: String,
    file: Option<std::fs::File>,
}

impl Out {
    fn put(&mut self, task: &str, set: &str, what: &str, plain: f64, fused: f64) {
        if let Some(f) = self.file.as_mut() {
            let _ = writeln!(f, "{}\t{task}\t{set}\t{what}\t{plain:.9}\t{fused:.9}\t{:+.3e}", self.model, fused - plain);
        }
    }
}

/// --paths: the two runs each task compares, (first, second) with true = fused; default (plain, fused). "plain2" runs
/// the reference forward twice (its own run-to-run reproducibility), "fused2" the fused one twice; "plain1" runs the
/// reference forward once (the second result is a copy: for the reference's own numbers only).
static PATHS: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// --paths plain-shape: the reference forward twice, the second run with other batch shapes (`bs`: 3/8 of each batch
/// size), to measure how much the reference itself moves when only the shapes its matmuls see change.
static SHAPE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static ALT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A task's batch size, or 3/8 of it on the second run of --paths plain-shape.
fn bs(n: usize) -> usize {
    if ALT.load(std::sync::atomic::Ordering::Relaxed) { (n * 3 / 8).max(1) } else { n }
}

/// Run `f` on the reference forward, then on the fused one (or as --paths says): ((first, seconds), (second, seconds)).
fn both<T: Clone>(mut f: impl FnMut() -> Result<T, String>) -> Result<((T, f64), (T, f64)), String> {
    let (a, b) = *PATHS.get_or_init(|| (false, true));
    set_fused(a);
    let t0 = Instant::now();
    let p = f();
    set_fused(false);
    let p = p?;
    let tp = t0.elapsed().as_secs_f64();
    if ONCE.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(((p.clone(), tp), (p, tp)));
    }
    set_fused(b);
    ALT.store(SHAPE.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
    let t1 = Instant::now();
    let r = f();
    ALT.store(false, std::sync::atomic::Ordering::Relaxed);
    set_fused(false);
    Ok(((p, tp), (r?, t1.elapsed().as_secs_f64())))
}

/// Switches as each tool sets them: (track activity, inference).
fn tool_switches(m: &Model, tool: &str) {
    let (track, infer) = if matches!(tool, "val_bpb" | "blimp_gpu") { (true, false) } else { (false, std::env::var("AVA_FAST_INFER").map_or(true, |v| v != "0")) };
    m.set_track_activity(track);
    m.set_inference(infer);
}

/// Decision comparison lines: (index, plain margin, fused margin, plain right, fused right) of the items that differ.
fn report_flips(task: &str, set: &str, n: usize, flips: &[(String, f64, f64, bool, bool)], max_dm: f64, unit: &str, out: &mut Out) {
    let ties = flips.iter().filter(|f| f.1.abs() < TIE).count();
    println!("{task} {set}: {n} items, {} decisions differ ({ties} of them near-ties, |plain margin| < {TIE:.0e} {unit}); max |margin diff| {max_dm:.3e} {unit}", flips.len());
    for (id, mp, mf, rp, rf) in flips.iter().take(20) {
        println!("  differs: {id}: plain margin {mp:+.6e} ({}), fused margin {mf:+.6e} ({})", if *rp { "right" } else { "wrong" }, if *rf { "right" } else { "wrong" });
    }
    out.put(task, set, "decisions_differ", 0.0, flips.len() as f64);
    out.put(task, set, "differ_not_near_tie", 0.0, (flips.len() - ties) as f64);
    out.put(task, set, "max_margin_diff", 0.0, max_dm);
}

fn speed(task: &str, set: &str, tp: f64, tf: f64, units: f64, what: &str, out: &mut Out) {
    println!("{task} {set} speed: plain {tp:.1}s, fused {tf:.1}s ({:.2}x); {:.0} vs {:.0} {what}/s", tp / tf.max(1e-9), units / tp.max(1e-9), units / tf.max(1e-9));
    out.put(task, set, "seconds", tp, tf);
}

/// The string value of `"key": "..."` in one JSON line (as blimp_gpu).
fn field(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\": \"");
    let start = line.find(&pat)? + pat.len();
    let mut out = String::new();
    let mut chars = line[start..].chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                o => out.push(o),
            },
            '"' => return Some(out),
            c => out.push(c),
        }
    }
    None
}

/// [EOS] sentence [EOS] (+ the length marks under the glance), as blimp_gpu and num_eval.
fn sentence(bpe: &Bpe, s: &str, glance: bool) -> Vec<u32> {
    let mut t = vec![EOS];
    t.extend(bpe.encode_doc(s));
    if glance {
        insert_marks(&t)
    } else {
        t
    }
}

/// Books from token shards, as long_eval's books_from_shards.
fn books_from_shards(dir: &Path, len: usize, n: usize) -> Result<Vec<Vec<u32>>, String> {
    let set = ShardSet::open(dir).map_err(|e| format!("open shards {}: {e}", dir.display()))?;
    let (mut out, mut cur) = (Vec::new(), Vec::new());
    'outer: for s in 0..set.len() {
        let toks = set.load_shard(s).map_err(|e| format!("load shard {s}: {e}"))?;
        for i in 0..toks.len() {
            let t = toks.get(i);
            if t == EOS {
                if cur.len() >= len {
                    cur.truncate(len);
                    out.push(std::mem::take(&mut cur));
                    if out.len() == n {
                        break 'outer;
                    }
                }
                cur.clear();
            } else {
                cur.push(t);
            }
        }
    }
    Ok(out)
}

/// NLL of every target of each book read in windows of `win`, as long_eval's token_nll (its scoring call is
/// eval::score_logits there too).
fn token_nll(model: &Model, dev: &Device, books: &[Vec<u32>], win: usize) -> Result<Vec<Vec<f32>>, String> {
    let mut out = Vec::with_capacity(books.len());
    for b in books {
        let mut seq = Vec::with_capacity(b.len() + 1);
        seq.push(EOS);
        seq.extend_from_slice(b);
        let len = b.len();
        let w = win.min(len);
        let rows = len.div_ceil(w);
        let (mut ids, mut tg) = (Vec::with_capacity(rows * w), Vec::with_capacity(rows * w));
        for r in 0..rows {
            for p in 0..w {
                let k = (r * w + p).min(len - 1);
                ids.push(seq[k]);
                tg.push(seq[k + 1]);
            }
        }
        let per = (32768 / w).max(1);
        let mut nll = Vec::with_capacity(rows * w);
        for r0 in (0..rows).step_by(per) {
            let rn = per.min(rows - r0);
            let idt = Tensor::from_vec(ids[r0 * w..(r0 + rn) * w].to_vec(), (rn, w), dev).map_err(|e| e.to_string())?;
            let logits = ava_gpu::eval::score_logits(model, &idt).map_err(|e| format!("forward: {e}"))?;
            let lp = candle_nn::ops::log_softmax(&logits, D::Minus1).map_err(|e| e.to_string())?;
            drop(logits);
            let tgt = Tensor::from_vec(tg[r0 * w..(r0 + rn) * w].to_vec(), (rn * w, 1), dev).map_err(|e| e.to_string())?;
            let v: Vec<f32> = lp.gather(&tgt, 1).and_then(|x| x.squeeze(1)).and_then(|x| x.neg()).and_then(|x| x.to_vec1()).map_err(|e| e.to_string())?;
            nll.extend(v);
        }
        nll.truncate(len);
        out.push(nll);
    }
    Ok(out)
}

/// BPB of each bucket over all books (as long_eval's table: nats over bytes, summed over books).
fn bucket_bpb(nll: &[Vec<f32>], books: &[Vec<u32>], bounds: &[usize], bytes: &[u32]) -> Vec<f64> {
    let mut acc = vec![(0.0f64, 0.0f64); bounds.len()];
    for (n, b) in nll.iter().zip(books) {
        for (p, (&x, &t)) in n.iter().zip(b).enumerate() {
            if let Some(k) = bounds.iter().position(|&hi| p < hi) {
                acc[k].0 += x as f64;
                acc[k].1 += bytes[t as usize] as f64;
            }
        }
    }
    acc.iter().map(|(n, by)| n / std::f64::consts::LN_2 / by.max(1e-300)).collect()
}

fn run() -> Result<(), String> {
    let mut model_path: Option<PathBuf> = None;
    let mut tasks: Vec<String> = ["val", "blimp", "mc", "num", "long"].iter().map(|s| s.to_string()).collect();
    let mut bpe_path = PathBuf::from("release/ava/bpe.txt");
    let mut vals: Vec<PathBuf> = Vec::new();
    let mut eval_tokens = 1_000_000u64;
    let mut blimp_dir = PathBuf::from("data/blimp");
    let mut blimp_limit = usize::MAX;
    let mut mcs: Vec<(PathBuf, mc::Norm)> = Vec::new();
    let mut batch_tokens = 16384usize;
    let mut num_items = PathBuf::from("data/num/dev.tsv");
    let mut long_shards = PathBuf::from("data/nn/shards-v40-books-dev");
    let (mut n_books, mut len, mut cut) = (100usize, 16384usize, 256usize);
    let mut trunc: Option<usize> = None;
    let mut bounds = vec![256usize, 1024, 4096, 16384];
    let mut out_path: Option<PathBuf> = None;
    let (mut cpu, mut threads) = (false, 8usize);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "--cpu" {
            cpu = true;
            i += 1;
            continue;
        }
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
        match key {
            "--model" => model_path = Some(v.into()),
            "--tasks" => tasks = v.split(',').map(|s| s.trim().to_string()).collect(),
            "--bpe" => bpe_path = v.into(),
            "--val" => vals.push(v.into()),
            "--eval-tokens" => eval_tokens = num(v)? as u64,
            "--blimp-dir" => blimp_dir = v.into(),
            "--blimp-limit" => blimp_limit = num(v)?.max(1),
            "--mc" => {
                let (p, n) = v.rsplit_once(':').ok_or_else(|| format!("--mc {v:?}: give TSV:NORM"))?;
                mcs.push((p.into(), mc::Norm::parse(n)?));
            }
            "--batch-tokens" => batch_tokens = num(v)?.max(1),
            "--num" => num_items = v.into(),
            "--long-shards" => long_shards = v.into(),
            "--books" => n_books = num(v)?.max(1),
            "--len" => len = num(v)?.max(2),
            "--cut" => cut = num(v)?.max(1),
            "--trunc" => trunc = Some(num(v)?.max(2)),
            "--buckets" => bounds = v.split(',').map(|x| num(x.trim())).collect::<Result<_, _>>()?,
            "--out" => out_path = Some(v.into()),
            "--threads" => threads = num(v)?.max(1),
            "--paths" => {
                let p = match v {
                    "both" => (false, true),
                    "plain2" => (false, false),
                    "plain-shape" => {
                        SHAPE.store(true, std::sync::atomic::Ordering::Relaxed);
                        (false, false)
                    }
                    "fused2" => (true, true),
                    "plain1" => {
                        ONCE.store(true, std::sync::atomic::Ordering::Relaxed);
                        (false, false)
                    }
                    _ => return Err(format!("--paths {v:?}: both, plain2, plain-shape, fused2 or plain1")),
                };
                PATHS.set(p).map_err(|_| "--paths given twice".to_string())?;
            }
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    if vals.is_empty() {
        vals = vec!["data/nn/shards-v41-mix/val".into(), "data/nn/shards-v42-mix/val".into()];
    }
    if mcs.is_empty() {
        mcs = vec![("data/v41/qa_dev.tsv".into(), mc::Norm::Byte), ("design-tests/nn/gpu/runs/v42/ava04/dev/v42_dev_all.tsv".into(), mc::Norm::Total)];
    }
    let model_path = model_path.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    if cpu {
        ava_gpu::diary::set_threads(threads); // before the first tensor op
    }
    let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    // on the CPU the lean model (ckpt::lean: the same logits bit for bit, no autograd graph kept), as val_bpb and mc_eval
    let load = if cpu { ava_gpu::ckpt::load_auto_lean } else { ava_gpu::ckpt::load_auto };
    let (cfg, _varmap, model) = load(&model_path, Kind::D, 256, &dev)?;
    println!("{}: glance {}, mag {}, engram {:?}, d {}, {} blocks, arch [{}]", model_path.display(), cfg.glance, cfg.mag, cfg.engram, cfg.d, cfg.layers, cfg.arch.signature());
    println!("{}", ava_gpu::eval::fused_line(&model, &dev));
    let (pa, pb) = *PATHS.get_or_init(|| (false, true));
    let nm = |f: bool| if f { "fused" } else { "plain" };
    println!("compared: \"plain\" = the {} forward, \"fused\" = the {} forward (--paths)", nm(pa), nm(pb));
    let mut out = Out {
        model: model_path.display().to_string(),
        file: match &out_path {
            Some(p) => Some(std::fs::OpenOptions::new().create(true).append(true).open(p).map_err(|e| format!("{}: {e}", p.display()))?),
            None => None,
        },
    };
    let byte_len = byte_lengths_n(&bpe, cfg.vocab);
    {
        // warm-up: both forwards once (kernels loaded, the pool grown) before anything is timed
        let ids = Tensor::from_vec(vec![EOS; 4 * 256], (4, 256), &dev).map_err(|e| e.to_string())?;
        both(|| ava_gpu::eval::score_logits(&model, &ids).and_then(|l| l.sum_all()?.to_scalar::<f32>()).map_err(|e| e.to_string()))?;
    }
    let t_all = Instant::now();
    for task in &tasks {
        match task.as_str() {
            "val" => {
                tool_switches(&model, "val_bpb");
                for v in &vals {
                    let set = ShardSet::open(v).map_err(|e| format!("open val shards {}: {e}", v.display()))?;
                    let name = v.parent().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| v.display().to_string());
                    let ((p, tp), (f, tf)) = both(|| evaluate(&model, set.iter_sequential(256), bs(16), eval_tokens, &byte_len, &dev).map(|r| r.0).map_err(|e| e.to_string()))?;
                    println!(
                        "val {name}: {} targets ({} bytes); BPB plain {:.9}, fused {:.9}, diff {:+.3e}; nats/token plain {:.9}, fused {:.9}",
                        p.tokens,
                        p.bytes,
                        p.bpb(),
                        f.bpb(),
                        f.bpb() - p.bpb(),
                        p.nats_per_token(),
                        f.nats_per_token()
                    );
                    if p.tokens != f.tokens || p.bytes != f.bytes {
                        return Err(format!("val {name}: different targets scored ({} / {} vs {} / {})", p.tokens, p.bytes, f.tokens, f.bytes));
                    }
                    out.put("val", &name, "bpb", p.bpb(), f.bpb());
                    speed("val", &name, tp, tf, p.tokens as f64, "tok", &mut out);
                }
            }
            "blimp" => {
                tool_switches(&model, "blimp_gpu");
                let mut files: Vec<PathBuf> = std::fs::read_dir(&blimp_dir)
                    .map_err(|e| format!("blimp dir {}: {e}", blimp_dir.display()))?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
                    .collect();
                files.sort();
                let (mut flips, mut n, mut max_dm, mut tp, mut tf, mut sents) = (Vec::new(), 0usize, 0.0f64, 0.0, 0.0, 0usize);
                let (mut right_p, mut right_f) = (0usize, 0usize);
                for f in &files {
                    let text = std::fs::read_to_string(f).map_err(|e| format!("read {}: {e}", f.display()))?;
                    let mut seqs = Vec::new();
                    for line in text.lines().filter(|l| !l.trim().is_empty()).take(blimp_limit) {
                        let (Some(g), Some(b)) = (field(line, "sentence_good"), field(line, "sentence_bad")) else { continue };
                        seqs.push(sentence(&bpe, &g, cfg.glance));
                        seqs.push(sentence(&bpe, &b, cfg.glance));
                    }
                    let ((p, a), (q, c)) = both(|| seq_nll(&model, &dev, &seqs, bs(256)).map_err(|e| e.to_string()))?;
                    tp += a;
                    tf += c;
                    sents += seqs.len();
                    let stem = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                    for k in 0..seqs.len() / 2 {
                        let (mp, mf) = (p[2 * k + 1] - p[2 * k], q[2 * k + 1] - q[2 * k]);
                        let (rp, rf) = (mp > 0.0, mf > 0.0);
                        right_p += rp as usize;
                        right_f += rf as usize;
                        max_dm = max_dm.max((mf - mp).abs());
                        if rp != rf {
                            flips.push((format!("{stem} #{k}"), mp, mf, rp, rf));
                        }
                        n += 1;
                    }
                }
                println!("blimp: accuracy plain {:.4}% ({right_p}/{n}), fused {:.4}% ({right_f}/{n})", 100.0 * right_p as f64 / n.max(1) as f64, 100.0 * right_f as f64 / n.max(1) as f64);
                out.put("blimp", "all", "right", right_p as f64, right_f as f64);
                report_flips("blimp", "all", n, &flips, max_dm, "nats", &mut out);
                speed("blimp", "all", tp, tf, sents as f64, "sentences", &mut out);
            }
            "mc" => {
                tool_switches(&model, "mc_eval");
                for (p, norm) in &mcs {
                    let items = mc::read_items(p)?;
                    let name = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                    let ((a, tp), (b, tf)) = both(|| mc::grade(&model, &dev, &bpe, &items, cfg.glance, bs(batch_tokens), *norm))?;
                    let (mut flips, mut max_dm) = (Vec::new(), 0.0f64);
                    for ((it, x), y) in items.iter().zip(&a).zip(&b) {
                        max_dm = max_dm.max((x.margin - y.margin).abs());
                        if x.pick != y.pick || x.right != y.right {
                            flips.push((it.id.clone(), x.margin, y.margin, x.right, y.right));
                        }
                    }
                    let (ra, rb) = (a.iter().filter(|g| g.right).count(), b.iter().filter(|g| g.right).count());
                    println!("mc {name} ({norm:?}): accuracy plain {ra}/{}, fused {rb}/{}", items.len(), items.len());
                    out.put("mc", &name, "right", ra as f64, rb as f64);
                    report_flips("mc", &name, items.len(), &flips, max_dm, "score units", &mut out);
                    let choices: usize = items.iter().map(|it| it.choices.len()).sum();
                    speed("mc", &name, tp, tf, choices as f64, "choices", &mut out);
                }
            }
            "num" => {
                tool_switches(&model, "num_eval");
                // num_eval's items (id, kind, bin, right, choices..) and rule: pick = the lowest NLL; right only if the
                // right choice is strictly lower than every other
                let text = std::fs::read_to_string(&num_items).map_err(|e| format!("read {}: {e}", num_items.display()))?;
                let mut items: Vec<(String, usize, Vec<String>)> = Vec::new();
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    let f: Vec<&str> = line.split('\t').collect();
                    if f.len() < 6 {
                        return Err(format!("{}: a line with fewer than 6 fields", num_items.display()));
                    }
                    let right: usize = f[3].parse().map_err(|_| format!("bad right index {:?}", f[3]))?;
                    items.push((f[0].to_string(), right, f[4..].iter().map(|s| s.to_string()).collect()));
                }
                let seqs: Vec<Vec<u32>> = items.iter().flat_map(|(_, _, c)| c.iter().map(|s| sentence(&bpe, s, cfg.glance))).collect();
                let ((p, tp), (q, tf)) = both(|| seq_nll(&model, &dev, &seqs, bs(512)).map_err(|e| e.to_string()))?;
                let verdict = |s: &[f64], right: usize| -> (usize, bool, f64) {
                    let mut order: Vec<usize> = (0..s.len()).collect();
                    order.sort_by(|&a, &b| s[a].total_cmp(&s[b]));
                    let correct = (0..s.len()).all(|j| j == right || s[right] < s[j]);
                    (order[0], correct, s[order[1]] - s[order[0]])
                };
                let (mut flips, mut max_dm, mut k, mut ra, mut rb) = (Vec::new(), 0.0f64, 0usize, 0usize, 0usize);
                for (id, right, ch) in &items {
                    let (x, y) = (verdict(&p[k..k + ch.len()], *right), verdict(&q[k..k + ch.len()], *right));
                    k += ch.len();
                    ra += x.1 as usize;
                    rb += y.1 as usize;
                    max_dm = max_dm.max((x.2 - y.2).abs());
                    if x.0 != y.0 || x.1 != y.1 {
                        flips.push((id.clone(), x.2, y.2, x.1, y.1));
                    }
                }
                println!("num: right plain {ra}/{}, fused {rb}/{}", items.len(), items.len());
                out.put("num", "dev", "right", ra as f64, rb as f64);
                report_flips("num", "dev", items.len(), &flips, max_dm, "nats", &mut out);
                speed("num", "dev", tp, tf, seqs.len() as f64, "sentences", &mut out);
            }
            "long" => {
                tool_switches(&model, "long_eval");
                let mut books = books_from_shards(&long_shards, len, n_books)?;
                if let Some(tr) = trunc {
                    // the same books, cut to their first --trunc tokens (the reference forward's own spread across lengths)
                    for b in books.iter_mut() {
                        b.truncate(tr);
                    }
                }
                println!("long: {} books x {len} tokens from {}; buckets {bounds:?}; cut {cut}", books.len(), long_shards.display());
                let ln = books.first().map_or(len, |b| b.len());
                let conds: Vec<(&str, usize, bool)> = if cfg.engram.is_some() { vec![("full", ln, false), ("cut", cut, false), ("no-eng", ln, true)] } else { vec![("full", ln, false), ("cut", cut, false)] };
                for (name, win, no_eng) in conds {
                    model.set_lesion(false, false, no_eng);
                    let ((p, tp), (q, tf)) = both(|| token_nll(&model, &dev, &books, win))?;
                    model.set_lesion(false, false, false);
                    let (bp, bq) = (bucket_bpb(&p, &books, &bounds, &byte_len), bucket_bpb(&q, &books, &bounds, &byte_len));
                    let mut lo = 0usize;
                    for (k, &hi) in bounds.iter().enumerate() {
                        let b = format!("{lo}-{hi}");
                        // per token: mean and max |NLL difference| in this bucket, and the books' own BPB differences
                        let (mut sa, mut mx, mut cnt) = (0.0f64, 0.0f64, 0usize);
                        let mut per_book = Vec::new();
                        for (bk, (x, y)) in p.iter().zip(&q).enumerate() {
                            let r = lo.min(x.len())..hi.min(x.len());
                            for j in r.clone() {
                                let d = (y[j] as f64 - x[j] as f64).abs();
                                sa += d;
                                mx = mx.max(d);
                                cnt += 1;
                            }
                            let one = |v: &[f32]| bucket_bpb(&[v.to_vec()], &books[bk..bk + 1], &bounds, &byte_len)[k];
                            per_book.push(one(y) - one(x));
                        }
                        let sd = (per_book.iter().map(|d| d * d).sum::<f64>() / per_book.len().max(1) as f64).sqrt();
                        println!(
                            "long {name} {b}: BPB plain {:.9}, fused {:.9}, diff {:+.3e}; per token |dNLL| mean {:.2e} max {:.2e} nats; per book diff rms {sd:.2e}",
                            bp[k],
                            bq[k],
                            bq[k] - bp[k],
                            sa / cnt.max(1) as f64,
                            mx
                        );
                        out.put("long", name, &b, bp[k], bq[k]);
                        lo = hi;
                    }
                    let tokens = (books.len() * books.first().map_or(0, |b| b.len())) as f64;
                    speed("long", name, tp, tf, tokens, "tok", &mut out);
                }
            }
            t => return Err(format!("unknown task {t:?} (val, blimp, mc, num, long)")),
        }
    }
    println!("done in {:.1}s", t_all.elapsed().as_secs_f64());
    Ok(())
}
