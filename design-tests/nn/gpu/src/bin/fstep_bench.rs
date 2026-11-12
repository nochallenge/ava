// TESTING / DESIGN ONLY — fstep_bench: the fused one-token CPU step (src/model/fstep.rs) against the plain step, and where one step spends its time; CPU only.
//
// fstep_bench --model FILE [--val DIR] [--arch D] [--ctx 256] [--steps 256] [--reps 3] [--check 200] [--prof 128]
//             [--threads 8] [--q8-threads 0 (the int8 products' threads; 0: --threads)] [--modes f32,int8,fused]
//     One saved model (ckpt::load_auto_lean), the talk taken from the val shards (PAD dropped).
//     Speed: tokens/s of `step` on one fresh stream after a 16-token warm-up, for f32, int8 (E2 on) and int8 with the
//     fused step, the modes in turn --reps times (so slow drift from other jobs hits them alike); each rep and the
//     median.
//     Exactness: --check tokens with the state carried, the plain int8 step and the fused one on two streams in
//     lockstep: every step's logits and the final state bit for bit (max |dlogit| printed).
//     Profile: one step by op class (`Model::step_profiled`, today's block only): candle tensors created per step
//     (every op makes one) and microseconds per step, f32 and int8, with every profiled step's logits checked against
//     the plain step's bits. For a block-options model (the State-backed stream; the fused step and the class
//     profile do not apply): tensors per step and the time inside `int8::mm` / `int8::head`.

use ava_data::bpe::PAD;
use ava_data::shards::ShardSet;
use ava_gpu::model::fstep::prof::{next_tensor_id, StepProf, CLASSES};
use ava_gpu::model::{Kind, Model};
use candle_core::{DType, Device, Result, Tensor};
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "fstep_bench --model FILE [--val DIR] [--arch D] [--ctx 256] [--steps 256] [--reps 3] [--check 200] [--prof 128] [--threads 8] [--q8-threads 0] [--modes f32,int8,fused]";

struct Opts {
    model: Option<PathBuf>,
    val: PathBuf,
    kind: Kind,
    ctx: usize,
    steps: usize,
    reps: usize,
    check: usize,
    prof: usize,
    threads: usize,
    modes: Vec<Mode>,
    q8_threads: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    F32,
    Int8,
    Fused,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::F32 => "f32",
            Mode::Int8 => "int8",
            Mode::Fused => "int8 fused",
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("fstep_bench: {e}");
        std::process::exit(1);
    }
}

fn run() -> std::result::Result<(), String> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let mut o = Opts {
        model: None,
        val: PathBuf::from(format!("{root}/data/nn/shards-v42-base/val")),
        kind: Kind::D,
        ctx: 256,
        steps: 256,
        reps: 3,
        check: 200,
        prof: 128,
        threads: 8,
        modes: vec![Mode::F32, Mode::Int8, Mode::Fused],
        q8_threads: 0,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |v: &str| v.parse::<usize>().map_err(|_| format!("not a number: {v}"));
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        match key {
            "--model" => o.model = Some(v.into()),
            "--val" => o.val = v.into(),
            "--arch" => o.kind = Kind::parse(v)?,
            "--ctx" => o.ctx = num(v)?.max(2),
            "--steps" => o.steps = num(v)?.max(1),
            "--reps" => o.reps = num(v)?.max(1),
            "--check" => o.check = num(v)?,
            "--prof" => o.prof = num(v)?,
            "--threads" => o.threads = num(v)?.max(1),
            "--q8-threads" => o.q8_threads = num(v)?,
            "--modes" => {
                o.modes = v
                    .split(',')
                    .map(|s| match s {
                        "f32" => Ok(Mode::F32),
                        "int8" => Ok(Mode::Int8),
                        "fused" => Ok(Mode::Fused),
                        _ => Err(format!("unknown mode {s}\n{USAGE}")),
                    })
                    .collect::<std::result::Result<_, _>>()?
            }
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    ava_gpu::diary::set_threads(o.threads); // before the first tensor op
    Q8T.store(o.q8_threads, std::sync::atomic::Ordering::Relaxed);
    let path = o.model.clone().ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    bench(&o, &path).map_err(|e| e.to_string())
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.iter().map(|x| x.to_bits()).collect())
}

fn max_abs(a: &Tensor, b: &Tensor) -> Result<f32> {
    a.sub(b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
}

/// the int8 products' thread count (--q8-threads; 0: candle's)
static Q8T: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn set_mode(m: &Model, md: Mode) -> Result<()> {
    m.set_fused(false);
    m.set_int8_threads(md != Mode::F32, Q8T.load(std::sync::atomic::Ordering::Relaxed))?;
    m.set_fused(md == Mode::Fused);
    Ok(())
}

/// Tokens/s of `step` on a fresh stream over toks[16..] after toks[..16].
fn speed(m: &Model, toks: &[u32]) -> Result<f64> {
    let mut st = m.stream()?;
    for &t in &toks[..16] {
        m.step(&mut st, t)?;
    }
    let t0 = Instant::now();
    for &t in &toks[16..] {
        m.step(&mut st, t)?;
    }
    Ok((toks.len() - 16) as f64 / t0.elapsed().as_secs_f64())
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn bench(o: &Opts, path: &PathBuf) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (cfg, _vm, m) = ava_gpu::ckpt::load_auto_lean(path, o.kind, o.ctx, &dev)?;
    m.set_track_activity(false);
    m.set_inference(true);
    let set = ShardSet::open(&o.val).map_err(|x| format!("open {}: {x}", o.val.display()))?;
    if set.vocab_size as usize != cfg.vocab {
        return Err(format!("val shards have vocab {}, the model {}", set.vocab_size, cfg.vocab).into());
    }
    let need = 16 + o.steps.max(o.check).max(o.prof);
    let mut talk: Vec<u32> = Vec::with_capacity(need);
    for w in set.iter_sequential(o.ctx) {
        talk.extend(w.iter().copied().filter(|&t| t != PAD));
        if talk.len() >= need {
            break;
        }
    }
    if talk.len() < need {
        return Err("the val set is shorter than the steps asked for".into());
    }
    talk.truncate(need);
    m.set_int8(true)?;
    let blocker = m.fused_blocker();
    m.set_int8(false)?;
    println!(
        "model {} ({}, d {}, {} layers, vocab {}{}), {} threads, AVX2+FMA {}; fused step: {}",
        path.display(),
        cfg.kind.name(),
        cfg.d,
        cfg.layers,
        cfg.vocab,
        if m.arch_path() { format!(", block options {}", cfg.arch.signature()) } else { String::new() },
        candle_core::utils::get_num_threads(),
        ava_gpu::int8::simd_ok(),
        blocker.map_or("applies".to_string(), |b| format!("does not apply ({b})"))
    );

    // 1. speed, the modes in turn
    let modes: Vec<Mode> = o.modes.iter().copied().filter(|&md| md != Mode::Fused || blocker.is_none()).collect();
    let toks = &talk[..16 + o.steps];
    let mut sp: Vec<Vec<f64>> = vec![Vec::new(); modes.len()];
    for _ in 0..o.reps {
        for (k, &md) in modes.iter().enumerate() {
            set_mode(&m, md)?;
            sp[k].push(speed(&m, toks)?);
        }
    }
    for (k, &md) in modes.iter().enumerate() {
        let reps: Vec<String> = sp[k].iter().map(|x| format!("{x:.1}")).collect();
        println!("step tokens/s {:<10} median {:>7.1}  reps {}  ({} tokens after 16)", md.name(), median(&sp[k]), reps.join(" "), o.steps);
    }

    // 2. exactness: plain int8 vs fused, two streams in lockstep
    if blocker.is_none() && o.check > 0 {
        set_mode(&m, Mode::Int8)?;
        let (mut a, mut b) = (m.stream()?, m.stream()?);
        let (mut worst, mut differ) = (0f32, 0usize);
        for &t in &talk[..o.check] {
            m.set_fused(false);
            let la = m.step(&mut a, t)?;
            m.set_fused(true);
            let lb = m.step(&mut b, t)?;
            worst = worst.max(max_abs(&la, &lb)?);
            differ += (bits(&la)? != bits(&lb)?) as usize;
        }
        m.set_fused(false);
        let (sa, sb) = (a.state_tensors(), b.state_tensors());
        let mut state_same = sa.len() == sb.len() && a.state_scalars() == b.state_scalars();
        for (x, y) in sa.iter().zip(&sb) {
            state_same &= x.dims() == y.dims() && bits(x)? == bits(y)?;
        }
        println!(
            "fused vs plain int8 over {} tokens, state carried: max |dlogit| {worst:e}, steps with any logit bit different {differ}, final state bit for bit {state_same}",
            o.check
        );
    }

    // 3. profile
    if o.prof == 0 {
        return Ok(());
    }
    let ptoks = &talk[..16 + o.prof];
    for md in [Mode::F32, Mode::Int8] {
        set_mode(&m, md)?;
        m.set_fused(false);
        // tensors per plain step
        let mut st = m.stream()?;
        for &t in &ptoks[..16] {
            m.step(&mut st, t)?;
        }
        let n = 16.min(o.prof);
        let a = next_tensor_id();
        for &t in &ptoks[16..16 + n] {
            m.step(&mut st, t)?;
        }
        let per_step = (next_tensor_id() - a - 1) as f64 / n as f64;
        if m.arch_path() {
            let mut st = m.stream()?;
            for &t in &ptoks[..16] {
                m.step(&mut st, t)?;
            }
            ava_gpu::int8::set_mm_prof(true);
            let t0 = Instant::now();
            for &t in &ptoks[16..] {
                m.step(&mut st, t)?;
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / o.prof as f64;
            let (calls, ns) = ava_gpu::int8::mm_prof();
            ava_gpu::int8::set_mm_prof(false);
            let mm_us = ns as f64 / 1e3 / o.prof as f64;
            println!(
                "profile {:<4} (block options, State-backed step): {per_step:.0} tensors/step, {us:.0} us/step ({:.1} tok/s); int8::mm/head {:.1} calls/step {mm_us:.0} us/step ({:.1}%), the rest {:.0} us/step ({:.2} us per other tensor)",
                md.name(),
                1e6 / us,
                calls as f64 / o.prof as f64,
                100.0 * mm_us / us,
                us - mm_us,
                (us - mm_us) / per_step
            );
            continue;
        }
        // by class: a counting pass and a timing pass, each in lockstep with the plain step (logits checked)
        let mut cnt = StepProf { count: true, ..Default::default() };
        let mut tim = StepProf::default();
        let (mut sa, mut sb) = (m.stream()?, m.stream()?);
        let mut throw = StepProf { count: true, ..Default::default() };
        let mut same = true;
        let mut wall = 0f64;
        for (i, &t) in ptoks.iter().enumerate() {
            let la = m.step(&mut sa, t)?;
            let lb = if i < 16 {
                m.step_profiled(&mut sb, t, &mut throw)?
            } else if i < 16 + n {
                m.step_profiled(&mut sb, t, &mut cnt)?
            } else {
                let t0 = Instant::now();
                let l = m.step_profiled(&mut sb, t, &mut tim)?;
                wall += t0.elapsed().as_secs_f64();
                l
            };
            same &= bits(&la)? == bits(&lb)?;
        }
        let steps = tim.steps.max(1) as f64;
        let total: f64 = tim.secs.iter().sum();
        let counted: u64 = cnt.tensors.iter().sum();
        println!(
            "profile {:<4}: {per_step:.0} tensors/step ({:.0} in the classes), {:.0} us/step in the classes ({:.0} us/step wall, {:.1} tok/s); profiled logits = plain step's bits: {same}",
            md.name(),
            counted as f64 / cnt.steps.max(1) as f64,
            total * 1e6 / steps,
            wall * 1e6 / steps,
            steps / wall
        );
        for (c, name) in CLASSES.iter().enumerate() {
            let tn = cnt.tensors[c] as f64 / cnt.steps.max(1) as f64;
            let us = tim.secs[c] * 1e6 / steps;
            if tn == 0.0 && us < 0.5 {
                continue;
            }
            println!("    {name:<28} {tn:>6.0} tensors {us:>8.1} us {:>5.1}%  ({:.2} us/tensor)", 100.0 * tim.secs[c] / total, if tn > 0.0 { us / tn } else { 0.0 });
        }
    }
    if blocker.is_none() {
        // the fused step: its int8 products against the rest
        set_mode(&m, Mode::Fused)?;
        let mut st = m.stream()?;
        for &t in &ptoks[..16] {
            m.step(&mut st, t)?;
        }
        let a = next_tensor_id();
        ava_gpu::int8::set_mm_prof(true);
        ava_gpu::model::fstep::take_secs();
        let t0 = Instant::now();
        for &t in &ptoks[16..] {
            m.step(&mut st, t)?;
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / o.prof as f64;
        let (calls, ns) = ava_gpu::int8::mm_prof();
        ava_gpu::int8::set_mm_prof(false);
        let tens = (next_tensor_id() - a - 1) as f64 / o.prof as f64;
        let mm_us = ns as f64 / 1e3 / o.prof as f64;
        println!(
            "profile fused: {tens:.0} tensors/step, {us:.0} us/step ({:.1} tok/s); int8 products {:.0} calls/step {mm_us:.0} us/step ({:.1}%), the rest {:.0} us/step",
            1e6 / us,
            calls as f64 / o.prof as f64,
            100.0 * mm_us / us,
            us - mm_us
        );
        let secs = ava_gpu::model::fstep::take_secs();
        for (name, ns) in ava_gpu::model::fstep::SECTIONS.iter().zip(secs) {
            let s = ns as f64 / 1e3 / o.prof as f64;
            println!("    {name:<30} {s:>7.1} us {:>5.1}%", 100.0 * s / us);
        }
        set_mode(&m, Mode::F32)?;
    }
    if m.arch_path() {
        ablate(o, &cfg, ptoks)?;
    }
    Ok(())
}

/// Tensors and microseconds per step (f32, after 16 warm-up tokens) of `m`.
fn per_step(m: &Model, toks: &[u32]) -> Result<(f64, f64)> {
    let mut st = m.stream()?;
    for &t in &toks[..16] {
        m.step(&mut st, t)?;
    }
    let n = (toks.len() - 16) as f64;
    let a = next_tensor_id();
    let t0 = Instant::now();
    for &t in &toks[16..] {
        m.step(&mut st, t)?;
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / n;
    Ok(((next_tensor_id() - a - 1) as f64 / n, us))
}

/// What each block option adds to one step of the block-options path: the model's own config with fresh default
/// weights, all options on and then each one left out in turn (f32; the counts hardly depend on the weights, the
/// halting loop aside).
fn ablate(o: &Opts, cfg: &ava_gpu::model::Config, toks: &[u32]) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let flags = cfg.arch.flags();
    let words: Vec<&str> = flags.split_whitespace().collect();
    let mut groups: Vec<Vec<&str>> = Vec::new();
    for w in words {
        if w.starts_with("--") || groups.is_empty() {
            groups.push(vec![w]);
        } else {
            groups.last_mut().expect("a group").push(w);
        }
    }
    let build = |skip: Option<usize>| -> std::result::Result<Model, String> {
        let line: Vec<&str> = groups.iter().enumerate().filter(|(g, _)| Some(*g) != skip).flat_map(|(_, w)| w.iter().copied()).collect();
        let ar = ava_gpu::arch::Arch::from_flags(&line.join(" "))?;
        let c = (*cfg).with_arch(ar)?;
        let vm = candle_nn::VarMap::new();
        let m = Model::new(c, candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev)).map_err(|e| e.to_string())?;
        m.set_track_activity(false);
        m.set_inference(true);
        Ok(m)
    };
    let full = build(None)?;
    let reps = o.reps.max(1);
    let mut base_us = Vec::new();
    let mut base_t = 0.0;
    for _ in 0..reps {
        let (t, us) = per_step(&full, toks)?;
        base_t = t;
        base_us.push(us);
    }
    let base = median(&base_us);
    drop(full);
    println!("block options, fresh weights, f32: all on {base_t:.0} tensors/step {base:.0} us/step; each option left out (tensors and us it adds):");
    let mut rows = Vec::new();
    for (g, w) in groups.iter().enumerate() {
        match build(Some(g)) {
            Ok(m) => {
                let mut v = Vec::new();
                let mut t = 0.0;
                for _ in 0..reps {
                    let (tt, us) = per_step(&m, toks)?;
                    t = tt;
                    v.push(us);
                }
                rows.push((w.join(" "), base_t - t, base - median(&v)));
            }
            Err(e) => println!("    {:<26} (cannot leave out: {e})", w.join(" ")),
        }
    }
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (name, t, us) in rows {
        println!("    {name:<26} {t:>6.0} tensors {us:>7.0} us");
    }
    Ok(())
}
