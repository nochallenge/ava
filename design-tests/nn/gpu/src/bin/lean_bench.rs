// TESTING / DESIGN ONLY — lean_bench: heap bytes per batched token and tokens/s of one CPU forward, trained (Var) weights against the lean model.
//
// lean_bench (--model FILE [--arch D] | --flags "BLOCK FLAGS" [--d 128] [--layers 4] [--vocab 4112]) [--rows 16] [--ctx 256]
//            [--val DIR] [--threads 8] [--mode var|lean|both] [--reps 1] [--windows 0]
//
// One forward over rows·ctx tokens (val windows with --val, else fixed pseudo-random ids), as mc_eval and val_bpb run it
// (inference on, activity off). The heap is counted by a counting allocator around the system one: the peak above what
// was live just before the forward, per batched token, with the logits [rows·ctx, V] (the output) given apart. Also the
// seconds (median of --reps) and tokens/s, and the process's peak working set. --flags builds a model with candle's
// random init (the memory does not depend on the weights' values); --model loads a saved one (arch.txt beside it).
// --mode both runs the trained-weights model and the lean one (ckpt::lean) on the same ids and prints max |dlogit|
// (bit-identical is 0). --windows N (with --model and --val): N more batches of val windows compared the same way.
// CPU only.

use ava_data::shards::ShardSet;
use ava_gpu::arch::Arch;
use ava_gpu::model::{Config, Kind, Model};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(n: usize) {
    let now = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(now, Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        LIVE.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = System.realloc(p, l, new);
        if !q.is_null() {
            if new >= l.size() {
                grew(new - l.size());
            } else {
                LIVE.fetch_sub(l.size() - new, Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

#[repr(C)]
#[derive(Default)]
struct MemCounters {
    cb: u32,
    page_faults: u32,
    peak_working_set: usize,
    working_set: usize,
    quota_peak_paged: usize,
    quota_paged: usize,
    quota_peak_nonpaged: usize,
    quota_nonpaged: usize,
    pagefile: usize,
    peak_pagefile: usize,
}

#[cfg(windows)]
extern "system" {
    fn GetCurrentProcess() -> isize;
    fn K32GetProcessMemoryInfo(process: isize, counters: *mut MemCounters, cb: u32) -> i32;
}

/// The process's peak working set in bytes (0 where unknown).
fn peak_ws() -> usize {
    #[cfg(windows)]
    unsafe {
        let mut c = MemCounters { cb: std::mem::size_of::<MemCounters>() as u32, ..Default::default() };
        if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) != 0 {
            return c.peak_working_set;
        }
    }
    0
}

const USAGE: &str = "lean_bench (--model FILE [--arch D] | --flags \"BLOCK FLAGS\" [--d 128] [--layers 4] [--vocab 4112]) [--rows 16] [--ctx 256]\n\
    \x20          [--val DIR] [--threads 8] [--mode var|lean|both] [--reps 1] [--windows 0]";

fn main() {
    if let Err(e) = run() {
        eprintln!("lean_bench: {e}");
        std::process::exit(1);
    }
}

/// One forward: (logits, peak heap above the start, seconds).
fn measure(m: &Model, ids: &Tensor) -> candle_core::Result<(Tensor, usize, f64)> {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let t0 = Instant::now();
    let l = m.forward(ids)?;
    let s = t0.elapsed().as_secs_f64();
    Ok((l, PEAK.load(Relaxed).saturating_sub(base), s))
}

fn max_diff(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
    a.sub(b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut model, mut flags, mut val) = (None::<PathBuf>, None::<String>, None::<PathBuf>);
    let (mut d, mut layers, mut vocab, mut rows, mut ctx, mut threads, mut reps, mut windows) = (128usize, 4usize, 4112usize, 16usize, 256usize, 8usize, 1usize, 0usize);
    let (mut kind, mut mode) = (Kind::D, "both".to_string());
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.clone();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
        match key {
            "--model" => model = Some(v.into()),
            "--flags" => flags = Some(v),
            "--val" => val = Some(v.into()),
            "--arch" => kind = Kind::parse(&v)?,
            "--d" => d = num(&v)?,
            "--layers" => layers = num(&v)?,
            "--vocab" => vocab = num(&v)?,
            "--rows" => rows = num(&v)?.max(1),
            "--ctx" => ctx = num(&v)?.max(2),
            "--threads" => threads = num(&v)?.clamp(1, 8),
            "--reps" => reps = num(&v)?.max(1),
            "--windows" => windows = num(&v)?,
            "--mode" => mode = v,
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    if !["var", "lean", "both"].contains(&mode.as_str()) {
        return Err(format!("--mode var|lean|both, not {mode}"));
    }
    ava_gpu::diary::set_threads(threads); // before the first tensor op
    let dev = Device::Cpu;
    let e = |x: candle_core::Error| x.to_string();
    let (cfg, vm, var_model) = match (&model, &flags) {
        (Some(p), None) => ava_gpu::ckpt::load_auto(p, kind, ctx, &dev)?,
        (None, Some(f)) => {
            let cfg = Config::with_kind(Kind::D, vocab, d, layers, ctx).with_arch(Arch::from_flags(f)?)?;
            let vm = VarMap::new();
            let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev)).map_err(e)?;
            (cfg, vm, m)
        }
        _ => return Err(format!("give exactly one of --model and --flags\n{USAGE}")),
    };
    let lean_model = if mode != "var" { Some(ava_gpu::ckpt::lean(cfg, &vm, &dev)?) } else { None };
    let var_model = if mode != "lean" { Some(var_model) } else { None };
    for m in var_model.iter().chain(lean_model.iter()) {
        m.set_track_activity(false);
        m.set_inference(true);
    }
    let n = rows * ctx;
    // the batches: val windows, else fixed pseudo-random ids
    let mut batches: Vec<Vec<u32>> = Vec::new();
    match &val {
        Some(dir) => {
            let set = ShardSet::open(dir).map_err(|x| format!("open {}: {x}", dir.display()))?;
            if set.vocab_size as usize != cfg.vocab {
                return Err(format!("val shards have vocab {}, the model {}", set.vocab_size, cfg.vocab));
            }
            let mut it = set.iter_sequential(ctx);
            for _ in 0..=windows {
                let mut ids = Vec::with_capacity(n);
                for _ in 0..rows {
                    let w = it.next().ok_or("the val set ran out")?;
                    ids.extend_from_slice(&w[..ctx]);
                }
                batches.push(ids);
            }
        }
        None => {
            let mut s = 0x9E3779B97F4A7C15u64;
            batches.push(
                (0..n)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        (s % cfg.vocab.min(4096) as u64) as u32
                    })
                    .collect(),
            );
        }
    }
    let logit_bytes = n * cfg.vocab * 4;
    println!(
        "{} (d {}, {} layers, vocab {}{}), {rows} rows x {ctx} = {n} tokens, {} threads; logits {:.1} MB ({} B/token)",
        model.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "random init".into()),
        cfg.d,
        cfg.layers,
        cfg.vocab,
        if cfg.arch.needs_path() { format!(", block options {}", cfg.arch.flags()) } else { String::new() },
        candle_core::utils::get_num_threads(),
        logit_bytes as f64 / 1e6,
        logit_bytes / n
    );
    let ids0 = Tensor::from_vec(batches[0].clone(), (rows, ctx), &dev).map_err(e)?;
    let mut outs: Vec<Tensor> = Vec::new();
    for (name, m) in [("var", var_model.as_ref()), ("lean", lean_model.as_ref())] {
        let Some(m) = m else { continue };
        let (mut secs, mut peak, mut last) = (Vec::new(), 0usize, None);
        for _ in 0..reps {
            let (l, p, s) = measure(m, &ids0).map_err(e)?;
            peak = peak.max(p);
            secs.push(s);
            last = Some(l);
        }
        secs.sort_by(f64::total_cmp);
        let s = secs[secs.len() / 2];
        println!(
            "{name:<5} peak heap {:>9.1} MB = {:>8} B/token ({:>8} B/token besides the logits); {s:.3} s, {:.0} tokens/s; process peak working set {:.2} GB",
            peak as f64 / 1e6,
            peak / n,
            peak.saturating_sub(logit_bytes) / n,
            n as f64 / s,
            peak_ws() as f64 / 1e9
        );
        outs.push(last.expect("reps >= 1"));
    }
    if outs.len() == 2 {
        let mut worst = max_diff(&outs[0], &outs[1]).map_err(e)?;
        drop(outs);
        let (vmod, lmod) = (var_model.as_ref().expect("both"), lean_model.as_ref().expect("both"));
        for ids in batches.iter().skip(1) {
            let x = Tensor::from_vec(ids.clone(), (rows, ctx), &dev).map_err(e)?;
            let a = vmod.forward(&x).map_err(e)?;
            let b = lmod.forward(&x).map_err(e)?;
            worst = worst.max(max_diff(&a, &b).map_err(e)?);
        }
        println!("var vs lean over {} batch(es) of {n} tokens: max |dlogit| {worst:e}", batches.len());
    }
    Ok(())
}
