// TESTING / DESIGN ONLY — int8_bench: E10 int8 talking path and E2 event sums against f32, on the CPU only.
//
// int8_bench --micro [--threads 2] [--ms 300]
//     One-row products at Ava's sizes (d128 and d384: d×d, d×4d, 4d×d and the head d×4112): candle's f32 matmul,
//     candle's quantized Q8_0 (QMatMul), the int8 kernel (src/int8.rs) dense and as an event sum (E2) at the spike
//     and kWTA densities she runs at, and the f32 event sum. Microseconds per product (median of 5 timed runs).
//
// int8_bench --model FILE [--val DIR] [--eval-tokens 100000] [--ctx 256] [--eval-batch 16] [--steps 256]
//            [--threads 2] [--no-eval] [--out FILE] [--arch D]
//     One saved model (ckpt::load_auto_lean: no autograd graph; arch.txt beside it): talking speed (tokens/s of `step` on one stream, after a
//     16-token warm-up) in f32, in int8 with E2, and in int8 without E2; int8 read token by token against int8 in one
//     pass on one window (max |dlogit|); then, unless --no-eval, the first --eval-tokens val targets (windows of
//     --ctx, as val_bpb scores them) in f32 and in int8 on the same windows: BPB of each, top-1 next-token agreement,
//     max and mean |dlogit|, and the batched eval speed. --out appends one tab-separated line.
//
// The device is always the CPU; --threads sets candle's and the int8 kernel's thread count before the first op.

use ava_data::bpe::{Bpe, PAD};
use ava_data::shards::ShardSet;
use ava_gpu::eval::{byte_lengths_n, window_io};
use ava_gpu::int8::{event_f32, set_event_max, QW};
use ava_gpu::model::{Kind, Model};
use ava_gpu::IGNORE;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor, D};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "int8_bench --micro [--threads 2] [--ms 300]\n\
    int8_bench --micro-batch ROWS [--threads 2] [--ms 300]\n\
    int8_bench --model FILE [--val DIR] [--eval-tokens 100000] [--ctx 256] [--eval-batch 16] [--steps 256] [--threads 2] [--no-eval] [--out FILE] [--arch D] [--bpe FILE]";

fn main() {
    if let Err(e) = run() {
        eprintln!("int8_bench: {e}");
        std::process::exit(1);
    }
}

struct Opts {
    micro: bool,
    micro_batch: usize,
    model: Option<PathBuf>,
    val: PathBuf,
    bpe: PathBuf,
    kind: Kind,
    eval_tokens: u64,
    ctx: usize,
    batch: usize,
    steps: usize,
    threads: usize,
    ms: u64,
    eval: bool,
    out: Option<PathBuf>,
}

fn run() -> Result<(), String> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let mut o = Opts {
        micro: false,
        micro_batch: 0,
        model: None,
        val: PathBuf::from(format!("{root}/data/nn/shards-v42-base/val")),
        bpe: PathBuf::from(format!("{root}/release/ava/bpe.txt")),
        kind: Kind::D,
        eval_tokens: 100_000,
        ctx: 256,
        batch: 16,
        steps: 256,
        threads: 2,
        ms: 300,
        eval: true,
        out: None,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        match key {
            "-h" | "--help" => return Err(USAGE.into()),
            "--micro" => o.micro = true,
            "--no-eval" => o.eval = false,
            _ => {
                let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
                let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
                match key {
                    "--model" => o.model = Some(v.into()),
                    "--val" => o.val = v.into(),
                    "--bpe" => o.bpe = v.into(),
                    "--arch" => o.kind = Kind::parse(v)?,
                    "--eval-tokens" => o.eval_tokens = num(v)? as u64,
                    "--ctx" => o.ctx = num(v)?.max(2),
                    "--eval-batch" => o.batch = num(v)?.max(1),
                    "--steps" => o.steps = num(v)?.max(1),
                    "--threads" => o.threads = num(v)?.max(1),
                    "--ms" => o.ms = num(v)?.max(10) as u64,
                    "--micro-batch" => o.micro_batch = num(v)?.max(1),
                    "--out" => o.out = Some(v.into()),
                    _ => return Err(format!("unknown option {key}\n{USAGE}")),
                }
                i += 1;
            }
        }
        i += 1;
    }
    ava_gpu::diary::set_threads(o.threads); // before the first tensor op
    if o.micro_batch > 0 {
        return micro_batch(&o).map_err(|e| e.to_string());
    }
    if o.micro {
        return micro(&o).map_err(|e| e.to_string());
    }
    let path = o.model.clone().ok_or_else(|| format!("--model or --micro is required\n{USAGE}"))?;
    model_bench(&o, &path)
}

/// Median microseconds per call of `f` (5 timed runs of about ms/5 each, after a warm-up).
fn time_us(ms: u64, mut f: impl FnMut()) -> f64 {
    let t0 = Instant::now();
    let mut n = 0u64;
    while t0.elapsed().as_millis() < (ms / 10).max(5) as u128 {
        f();
        n += 1;
    }
    let per = (ms as f64 / 5.0) / (t0.elapsed().as_secs_f64() * 1e3 / n as f64);
    let reps = (per as u64).max(3);
    let mut v: Vec<f64> = (0..5)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            t.elapsed().as_secs_f64() * 1e6 / reps as f64
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[2]
}

fn noise(n: usize, seed: u64) -> Vec<f32> {
    // SplitMix64 -> Box-Muller
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (a, b) = (next().max(1e-12), next());
            ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
        })
        .collect()
}

/// A row with `rate` of its inputs at 1 (spikes) or, with `values`, at a positive value (kWTA'd units).
fn sparse(n: usize, rate: f64, values: bool, seed: u64) -> Vec<f32> {
    let z = noise(n, seed);
    let k = ((n as f64 * rate).round() as usize).max(1);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| z[b].total_cmp(&z[a]));
    let mut x = vec![0f32; n];
    for &j in &idx[..k] {
        x[j] = if values { z[j].abs() + 0.1 } else { 1.0 };
    }
    x
}

fn micro(o: &Opts) -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let th = candle_core::utils::get_num_threads();
    println!("micro: one-row products, {th} threads, microseconds per product (median of 5 runs); AVX2+FMA int8 kernel: {}", ava_gpu::int8::simd_ok());
    println!("{:<6} {:<12} {:>9} {:>9} {:>9} {:>9}   {:<40}", "d", "shape", "f32", "q8_0", "int8", "int8 tens", "event sums: density int8-E2 / f32-E2");
    for d in [128usize, 384] {
        for (name, din, dout, dens, vals) in [
            ("d x d", d, d, vec![0.01, 0.04, 0.08], false),
            ("d x 4d", d, 4 * d, vec![], false),
            ("4d x d", 4 * d, d, vec![0.125, 0.25], true),
            ("d x 4112", d, 4112, vec![], false),
        ] {
            let w = noise(din * dout, 1 + din as u64 * 7 + dout as u64);
            let w: Vec<f32> = w.iter().map(|v| v * 0.05).collect();
            let wt = Tensor::from_vec(w.clone(), (din, dout), &dev)?;
            let x = noise(din, 9);
            let xt = Tensor::from_vec(x.clone(), (1, din), &dev)?;
            let f32_us = time_us(o.ms, || {
                let _ = xt.matmul(&wt).unwrap();
            });
            let q8 = if din % 32 == 0 {
                let qt = QTensor::quantize(&wt.t()?.contiguous()?, GgmlDType::Q8_0)?;
                let qm = QMatMul::from_qtensor(qt)?;
                format!("{:.1}", time_us(o.ms, || {
                    let _ = qm.forward(&xt).unwrap();
                }))
            } else {
                "-".into()
            };
            let q = QW::from_rows(&w, din, dout);
            let mut y = vec![0f32; dout];
            set_event_max(-1.0);
            let i8_us = time_us(o.ms, || q.matmul_rows(&x, 1, &mut y, -1.0, th));
            // through a candle tensor (as the model calls it): the int8 set, scoped
            let mut set = ava_gpu::int8::Q8Set::new(th);
            set.add(&wt)?;
            let set = std::sync::Arc::new(set);
            let tens_us = {
                let _s = ava_gpu::int8::scope(Some(set));
                time_us(o.ms, || {
                    let _ = ava_gpu::int8::mm(&xt, &wt).unwrap();
                })
            };
            let mut ev = String::new();
            for &r in &dens {
                let xs = sparse(din, r, vals, 11);
                let e8 = time_us(o.ms, || q.matmul_rows(&xs, 1, &mut y, 1.0, th));
                let ef = time_us(o.ms, || event_f32(&xs, &w, dout, &mut y));
                let xst = Tensor::from_vec(xs.clone(), (1, din), &dev)?;
                let fd = time_us(o.ms, || {
                    let _ = xst.matmul(&wt).unwrap();
                });
                ev += &format!("{:.0}%: {e8:.1} / {ef:.1} (f32 dense {fd:.1})  ", r * 100.0);
            }
            println!("{:<6} {:<12} {:>9.1} {:>9} {:>9.1} {:>9.1}   {ev}", d, name, f32_us, q8, i8_us, tens_us);
        }
    }
    set_event_max(ava_gpu::int8::EVENT_MAX);
    Ok(())
}

/// Batched products of `o.micro_batch` rows at Ava's sizes: candle's f32 matmul, the int8 kernel per row (row blocking
/// off) and row-blocked, for dense rows and for the sparse rows she feeds them (spikes at 5%, kWTA'd units at 25%);
/// milliseconds per product (median of 5 runs) and the int8 outputs checked bit for bit between the two kernels.
fn micro_batch(o: &Opts) -> candle_core::Result<()> {
    use ava_gpu::int8::set_block;
    let dev = Device::Cpu;
    let th = candle_core::utils::get_num_threads();
    let m = o.micro_batch;
    println!("micro-batch: {m} rows, {th} threads, ms per product (median of 5 runs); AVX2+FMA int8 kernel: {}", ava_gpu::int8::simd_ok());
    println!("{:<5} {:<9} {:<11} {:>9} {:>11} {:>11} {:>8}  same bits", "d", "shape", "rows", "f32", "int8 row", "int8 block", "x vs f32");
    for d in [128usize, 384] {
        for (name, din, dout, kinds) in [
            ("d x d", d, d, vec![("dense", 0.0, false), ("spikes 5%", 0.05, false)]),
            ("d x 4d", d, 4 * d, vec![("dense", 0.0, false)]),
            ("4d x d", 4 * d, d, vec![("dense", 0.0, false), ("kwta 25%", 0.25, true)]),
            ("d x 4112", d, 4112, vec![("dense", 0.0, false)]),
        ] {
            let w: Vec<f32> = noise(din * dout, 1 + din as u64 * 7 + dout as u64).iter().map(|v| v * 0.05).collect();
            let wt = Tensor::from_vec(w.clone(), (din, dout), &dev)?;
            let q = QW::from_rows(&w, din, dout);
            for (kind, rate, vals) in kinds {
                let x: Vec<f32> = if rate == 0.0 { noise(m * din, 9) } else { (0..m).flat_map(|r| sparse(din, rate, vals, 11 + r as u64)).collect() };
                let xt = Tensor::from_vec(x.clone(), (m, din), &dev)?;
                let f32_ms = time_us(o.ms, || {
                    let _ = xt.matmul(&wt).unwrap();
                }) / 1e3;
                let (mut y0, mut y1) = (vec![0f32; m * dout], vec![0f32; m * dout]);
                set_block(false);
                let row_ms = time_us(o.ms, || q.matmul_rows(&x, m, &mut y0, ava_gpu::int8::EVENT_MAX, th)) / 1e3;
                set_block(true);
                let blk_ms = time_us(o.ms, || q.matmul_rows(&x, m, &mut y1, ava_gpu::int8::EVENT_MAX, th)) / 1e3;
                let mut same = y0.iter().zip(&y1).all(|(a, b)| a.to_bits() == b.to_bits());
                // sparse rows: blocked with every sparse row as an event sum, and with every row through the 6-row kernel
                let mut extra = String::new();
                if rate > 0.0 {
                    for (label, share) in [("all events", 1.0), ("all 6-row", 0.0)] {
                        ava_gpu::int8::set_block_event_max(share);
                        let t = time_us(o.ms, || q.matmul_rows(&x, m, &mut y1, ava_gpu::int8::EVENT_MAX, th)) / 1e3;
                        same &= y0.iter().zip(&y1).all(|(a, b)| a.to_bits() == b.to_bits());
                        extra += &format!("  block {label} {t:.3}");
                    }
                    ava_gpu::int8::set_block_event_max(ava_gpu::int8::BLOCK_EVENT_MAX);
                }
                println!("{:<5} {:<9} {:<11} {:>9.3} {:>11.3} {:>11.3} {:>8.2}  {same}{extra}", d, name, kind, f32_ms, row_ms, blk_ms, f32_ms / blk_ms);
            }
        }
    }
    Ok(())
}

/// Tokens/s of `step` on one stream: a 16-token warm-up, then `toks` timed.
fn step_speed(m: &Model, toks: &[u32]) -> candle_core::Result<(f64, Tensor)> {
    let mut st = m.stream()?;
    for &t in &toks[..16] {
        m.step(&mut st, t)?;
    }
    let t0 = Instant::now();
    let mut last = None;
    for &t in &toks[16..] {
        last = Some(m.step(&mut st, t)?);
    }
    Ok(((toks.len() - 16) as f64 / t0.elapsed().as_secs_f64(), last.expect("tokens")))
}

/// Max |dlogit| between reading `w` token by token (step) and in one pass (forward), with the model as it is set.
fn stream_vs_pass(m: &Model, w: &[u32]) -> candle_core::Result<f32> {
    let dev = Device::Cpu;
    let one = m.forward(&Tensor::from_vec(w.to_vec(), (1, w.len()), &dev)?)?;
    let mut st = m.stream()?;
    let mut worst = 0f32;
    for (i, &t) in w.iter().enumerate() {
        let l = m.step(&mut st, t)?;
        worst = worst.max(l.sub(&one.get(i)?)?.abs()?.max(0)?.to_scalar::<f32>()?);
    }
    Ok(worst)
}

#[derive(Default)]
struct Acc {
    nats_f: f64,
    nats_q: f64,
    bytes: u64,
    tokens: u64,
    agree: u64,
    max_d: f32,
    sum_d: f64,
    n_d: u64,
    secs_f: f64,
    secs_q: f64,
}

fn model_bench(o: &Opts, path: &PathBuf) -> Result<(), String> {
    let dev = Device::Cpu;
    let e = |x: candle_core::Error| x.to_string();
    let (cfg, _vm, m) = ava_gpu::ckpt::load_auto_lean(path, o.kind, o.ctx, &dev)?;
    m.set_track_activity(false);
    m.set_inference(true);
    let th = candle_core::utils::get_num_threads();
    let set = ShardSet::open(&o.val).map_err(|x| format!("open {}: {x}", o.val.display()))?;
    if set.vocab_size as usize != cfg.vocab {
        return Err(format!("val shards have vocab {}, the model {}", set.vocab_size, cfg.vocab));
    }
    let mut it = set.iter_sequential(o.ctx);
    let first = it.next().ok_or("no val windows")?;
    let mut talk: Vec<u32> = first.iter().copied().filter(|&t| t != PAD).collect();
    while talk.len() < 16 + o.steps {
        let w = it.next().ok_or("the val set is shorter than --steps")?;
        talk.extend(w.iter().copied().filter(|&t| t != PAD));
    }
    talk.truncate(16 + o.steps);
    println!(
        "model {} ({}, d {}, {} layers, vocab {}{}), {th} threads",
        path.display(),
        cfg.kind.name(),
        cfg.d,
        cfg.layers,
        cfg.vocab,
        if m.arch_path() { format!(", block options {}", cfg.arch.signature()) } else { String::new() }
    );
    // talking speed: f32, int8 with E2, int8 without E2
    let (sp_f, last_f) = step_speed(&m, &talk).map_err(e)?;
    let t0 = Instant::now();
    m.set_int8(true).map_err(e)?;
    let build_s = t0.elapsed().as_secs_f64();
    let (nq, bytes) = m.int8_size();
    let (sp_q, last_q) = step_speed(&m, &talk).map_err(e)?;
    set_event_max(-1.0);
    let (sp_qd, last_qd) = step_speed(&m, &talk).map_err(e)?;
    set_event_max(ava_gpu::int8::EVENT_MAX);
    let same_e2 = last_q.sub(&last_qd).map_err(e)?.abs().map_err(e)?.max(0).map_err(e)?.to_scalar::<f32>().map_err(e)?;
    let dl = last_q.sub(&last_f).map_err(e)?.abs().map_err(e)?.max(0).map_err(e)?.to_scalar::<f32>().map_err(e)?;
    println!(
        "int8 set: {nq} matrices, {:.1} MB, built in {build_s:.2}s\nstep tokens/s ({} tokens): f32 {sp_f:.1}  int8+E2 {sp_q:.1}  int8 dense {sp_qd:.1}  (x{:.2}, x{:.2}); E2 vs dense last logits max |d| {same_e2:e}; int8 vs f32 last logits max |d| {dl:.4}",
        bytes as f64 / 1e6,
        talk.len() - 16,
        sp_q / sp_f,
        sp_qd / sp_f
    );
    let sw = 64.min(first.len() - 1);
    let svp_q = stream_vs_pass(&m, &first[..sw]).map_err(e)?;
    m.set_int8(false).map_err(e)?;
    let svp_f = stream_vs_pass(&m, &first[..sw]).map_err(e)?;
    println!("token by token vs one pass ({sw} tokens), max |dlogit|: int8 {svp_q:.2e}, f32 {svp_f:.2e}");
    let mut acc = Acc::default();
    if o.eval {
        let bpe = Bpe::load(&o.bpe).map_err(|x| format!("bpe {}: {x}", o.bpe.display()))?;
        let byte_len = byte_lengths_n(&bpe, cfg.vocab);
        let mut it = set.iter_sequential(o.ctx);
        let mut remaining = o.eval_tokens;
        while remaining > 0 {
            let (mut ids, mut tg, mut b) = (Vec::new(), Vec::new(), 0usize);
            while b < o.batch && remaining > 0 {
                let Some(w) = it.next() else { break };
                let (i, t) = window_io(&w, &mut remaining);
                ids.extend(i);
                tg.extend(t);
                b += 1;
            }
            if b == 0 {
                break;
            }
            let t = ids.len() / b;
            let x = Tensor::from_vec(ids, (b, t), &dev).map_err(e)?;
            let t0 = Instant::now();
            let lf = m.forward(&x).map_err(e)?;
            acc.secs_f += t0.elapsed().as_secs_f64();
            m.set_int8(true).map_err(e)?;
            let t0 = Instant::now();
            let lq = m.forward(&x).map_err(e)?;
            acc.secs_q += t0.elapsed().as_secs_f64();
            m.set_int8(false).map_err(e)?;
            score(&lf, &lq, &tg, &byte_len, &mut acc).map_err(e)?;
            eprint!("\r  scored {} targets", acc.tokens);
        }
        eprintln!();
        let bpb = |n: f64| n / std::f64::consts::LN_2 / acc.bytes as f64;
        let (bf, bq) = (bpb(acc.nats_f), bpb(acc.nats_q));
        println!(
            "val {} targets ({} bytes, ctx {}): BPB f32 {bf:.6}  int8 {bq:.6}  (int8 - f32 {:+.6}); top-1 agreement {:.4}; |dlogit| max {:.4} mean {:.5}; batched eval tokens/s f32 {:.0} int8 {:.0}",
            acc.tokens,
            acc.bytes,
            o.ctx,
            bq - bf,
            acc.agree as f64 / acc.tokens.max(1) as f64,
            acc.max_d,
            acc.sum_d / acc.n_d.max(1) as f64,
            acc.tokens as f64 / acc.secs_f,
            acc.tokens as f64 / acc.secs_q
        );
    }
    if let Some(out) = &o.out {
        let new = std::fs::metadata(out).map(|x| x.len() == 0).unwrap_or(true);
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out).map_err(|x| format!("{}: {x}", out.display()))?;
        if new {
            writeln!(f, "model\tthreads\tstep_f32\tstep_int8_e2\tstep_int8_dense\tstream_vs_pass_int8\tstream_vs_pass_f32\ttargets\tbytes\tbpb_f32\tbpb_int8\tagree\tmax_dlogit\tmean_dlogit\teval_tps_f32\teval_tps_int8").map_err(|x| x.to_string())?;
        }
        let bpb = |n: f64| if acc.bytes > 0 { n / std::f64::consts::LN_2 / acc.bytes as f64 } else { f64::NAN };
        writeln!(
            f,
            "{}\t{th}\t{sp_f:.1}\t{sp_q:.1}\t{sp_qd:.1}\t{svp_q:.3e}\t{svp_f:.3e}\t{}\t{}\t{:.6}\t{:.6}\t{:.5}\t{:.5}\t{:.6}\t{:.0}\t{:.0}",
            path.display(),
            acc.tokens,
            acc.bytes,
            bpb(acc.nats_f),
            bpb(acc.nats_q),
            acc.agree as f64 / acc.tokens.max(1) as f64,
            acc.max_d,
            acc.sum_d / acc.n_d.max(1) as f64,
            acc.tokens as f64 / acc.secs_f.max(1e-9),
            acc.tokens as f64 / acc.secs_q.max(1e-9)
        )
        .map_err(|x| x.to_string())?;
    }
    Ok(())
}

/// Both passes' NLL (f64 log-softmax, as eval::evaluate scores), top-1 agreement and |dlogit| over the scored rows.
fn score(lf: &Tensor, lq: &Tensor, tg: &[u32], byte_len: &[u32], acc: &mut Acc) -> candle_core::Result<()> {
    let dev = lf.device();
    let safe: Vec<u32> = tg.iter().map(|&x| if x == IGNORE { 0 } else { x }).collect();
    let tg_t = Tensor::from_slice(&safe, (tg.len(), 1), dev)?;
    let nll = |l: &Tensor| -> candle_core::Result<Vec<f64>> {
        candle_nn::ops::log_softmax(&l.to_dtype(DType::F64)?, D::Minus1)?.gather(&tg_t, D::Minus1)?.neg()?.flatten_all()?.to_vec1::<f64>()
    };
    let (nf, nq) = (nll(lf)?, nll(lq)?);
    let (af, aq) = (lf.argmax(D::Minus1)?.to_vec1::<u32>()?, lq.argmax(D::Minus1)?.to_vec1::<u32>()?);
    let dmax: Vec<f32> = lf.sub(lq)?.abs()?.max(D::Minus1)?.to_vec1()?;
    let dmean: Vec<f32> = lf.sub(lq)?.abs()?.mean(D::Minus1)?.to_vec1()?;
    for (r, &target) in tg.iter().enumerate() {
        if target == IGNORE || target == PAD {
            continue;
        }
        acc.nats_f += nf[r];
        acc.nats_q += nq[r];
        acc.bytes += byte_len[target as usize] as u64;
        acc.tokens += 1;
        acc.agree += u64::from(af[r] == aq[r]);
        acc.max_d = acc.max_d.max(dmax[r]);
        acc.sum_d += dmean[r] as f64;
        acc.n_d += 1;
    }
    Ok(())
}
