// TESTING / DESIGN ONLY — talk_bench: where answer_eval's time goes (the library, reading the prompt, writing the answer), for the plain step and the fused one, and the two compared bit for bit on the real questions; CPU only.
//
// talk_bench --model FILE --questions TSV [--library FILE] [--k 2] [--tokens 40] [--threads 8] [--limit N] [--lean] [--no-eco] [--spin S]
//            [--modes plain,fused] [--check] [--arch D] [--bpe FILE]
// * The prompts are answer_eval's ([EOS] + "Context: ...\nQuestion: q\nAnswer:"), the reply its greedy one (diary::reply's
//   loop: until EOS, a new question or --tokens), int8 on (the chat app's way). Per mode: the library's time, then the
//   prompt tokens read (Model::step, one token at a time, as diary::reply reads them) and the tokens written, each in
//   tokens/s, with the share inside the int8 products; for the fused mode its sections (fstep::arch::SECTIONS).
// * Both modes run: their replies compared question by question.
// * --check: per question the plain and the fused step on two streams in lockstep over the prompt and the plain
//   step's greedy reply: every logit bit for bit, and the final State (fstep::arch::state_tensors).

use ava_data::bpe::{Bpe, EOS};
use ava_data::numbers::strip_marks;
use ava_data::rng::SplitMix64;
use ava_gpu::diary;
use ava_gpu::lookup::Library;
use ava_gpu::model::{Kind, Model, Stream};
use candle_core::{DType, Tensor};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("talk_bench: {e}");
        std::process::exit(1);
    }
}

fn bits(t: &Tensor) -> Result<Vec<u32>, String> {
    Ok(t.to_dtype(DType::F32).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())?.iter().map(|x| x.to_bits()).collect())
}

fn max_abs(a: &Tensor, b: &Tensor) -> Result<f32, String> {
    a.sub(b).and_then(|d| d.abs()).and_then(|d| d.flatten_all()).and_then(|d| d.max(0)).and_then(|d| d.to_scalar::<f32>()).map_err(|e| e.to_string())
}

/// Totals of one mode.
#[derive(Default)]
struct Tally {
    read: usize,
    written: usize,
    read_s: f64,
    write_s: f64,
    mm_s: f64,
    replies: Vec<String>,
}

/// diary::reply's loop on a fresh stream, the reading and the writing timed apart.
fn ask(model: &Model, bpe: &Bpe, ids: &[u32], max_tokens: usize, t: &mut Tally) -> Result<(), String> {
    let e = |e: candle_core::Error| e.to_string();
    let mut rng = SplitMix64::new(1);
    let mut st: Stream = model.stream().map_err(e)?;
    let t0 = Instant::now();
    let mut next = None;
    for &tok in ids {
        next = Some(model.step(&mut st, tok).map_err(e)?);
    }
    let mut next = next.ok_or("empty prompt")?;
    t.read_s += t0.elapsed().as_secs_f64();
    t.read += ids.len();
    let t0 = Instant::now();
    let mut out = Vec::new();
    for n in 0..max_tokens {
        let tok = diary::pick(&next, 0.0, 1, &mut rng).map_err(e)?;
        if tok == EOS {
            break;
        }
        out.push(tok);
        if diary::cut_reply(&bpe.decode(&strip_marks(&out))).1 {
            break;
        }
        if n + 1 < max_tokens {
            next = model.step(&mut st, tok).map_err(e)?;
            t.written += 1;
        }
    }
    t.write_s += t0.elapsed().as_secs_f64();
    t.replies.push(diary::cut_reply(&bpe.decode(&strip_marks(&out))).0);
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let num = |k: &str, d: usize| -> Result<usize, String> { get(k).map(|v| v.parse().map_err(|_| format!("bad {k}"))).transpose().map(|v| v.unwrap_or(d)) };
    let model_path = PathBuf::from(get("--model").ok_or("--model FILE")?);
    let qpath = PathBuf::from(get("--questions").ok_or("--questions TSV")?);
    let (k, tokens, limit) = (num("--k", 2)?, num("--tokens", 40)?, num("--limit", usize::MAX)?);
    let threads = num("--threads", 8)?;
    diary::set_threads(threads);
    let kind = Kind::parse(&get("--arch").unwrap_or_else(|| "D".into()))?;
    let bpe_path = PathBuf::from(get("--bpe").unwrap_or_else(|| ava_gpu::DEFAULT_BPE.into()));
    let modes: Vec<String> = get("--modes").unwrap_or_else(|| "plain,fused".into()).split(',').map(|s| s.to_string()).collect();
    if modes.iter().any(|m| m != "plain" && m != "fused") {
        return Err("--modes: plain and/or fused".into());
    }
    let check = args.iter().any(|a| a == "--check");
    if args.iter().any(|a| a == "--no-eco") {
        println!("power throttling opt-out: {}", no_eco());
    }

    let dev = ava_gpu::device(true).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    // --lean: the weights as plain tensors (ckpt::load_auto_lean); answer_eval loads them as variables (ckpt::load_auto)
    let load = if args.iter().any(|a| a == "--lean") { ava_gpu::ckpt::load_auto_lean } else { ava_gpu::ckpt::load_auto };
    let (cfg, _varmap, model) = load(&model_path, kind, 4096, &dev)?;
    model.set_track_activity(false);
    model.set_inference(true);
    model.set_int8(true).map_err(|e| format!("int8: {e}"))?;
    println!(
        "model {} (d {}, {} layers, vocab {}), {} threads; fused step: {}",
        model_path.display(),
        cfg.d,
        cfg.layers,
        cfg.vocab,
        candle_core::utils::get_num_threads(),
        model.fused_blocker().map_or("applies".to_string(), |b| format!("does not apply ({b})"))
    );

    // the prompts, as answer_eval makes them (the library searched twice: the context, and the titles it writes)
    // --spin S: S seconds of busy single-thread work first (to tell a scheduling effect from the library's memory)
    if let Some(s) = get("--spin").and_then(|v| v.parse::<f64>().ok()) {
        let t0 = Instant::now();
        let mut acc = 0u64;
        while t0.elapsed().as_secs_f64() < s {
            acc = acc.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        }
        println!("spun {s}s ({acc})");
    }
    let lib = match get("--library") {
        Some(p) => Some(Library::load(&PathBuf::from(&p))?),
        None => None,
    };
    let text = std::fs::read_to_string(&qpath).map_err(|e| format!("{}: {e}", qpath.display()))?;
    let rows: Vec<Vec<&str>> = text.lines().skip(1).filter(|l| !l.trim().is_empty()).map(|l| l.split('\t').collect()).take(limit).collect();
    let t0 = Instant::now();
    let mut prompts = Vec::with_capacity(rows.len());
    for r in &rows {
        let q = r.get(2).ok_or("a row without a question")?;
        let ctx = lib.as_ref().map(|l| l.context(q, k)).unwrap_or_default();
        if let Some(l) = lib.as_ref() {
            let _titles = l.search(q, k);
        }
        let mut ids = vec![EOS];
        ids.extend(diary::encode(&bpe, cfg.glance, &ava_gpu::qa::prompt_text(&ctx, &diary::one_line(q))));
        prompts.push(ids);
    }
    let lib_s = t0.elapsed().as_secs_f64();
    if args.iter().any(|a| a == "--drop-lib") {
        drop(lib);
    }
    let n_read: usize = prompts.iter().map(|p| p.len()).sum();
    println!("{} questions: library {lib_s:.2}s ({:.1} ms per question), prompts {} tokens ({:.0} per question)", prompts.len(), 1e3 * lib_s / prompts.len().max(1) as f64, n_read, n_read as f64 / prompts.len().max(1) as f64);

    let mut tallies: Vec<(String, Tally)> = Vec::new();
    for md in &modes {
        model.set_fused(md == "fused");
        let mut t = Tally::default();
        ava_gpu::int8::set_mm_prof(true);
        ava_gpu::model::fstep::arch::take_secs();
        for ids in &prompts {
            ask(&model, &bpe, ids, tokens, &mut t)?;
        }
        t.mm_s = ava_gpu::int8::mm_prof().1 as f64 / 1e9;
        ava_gpu::int8::set_mm_prof(false);
        let secs = ava_gpu::model::fstep::arch::take_secs();
        let all = t.read_s + t.write_s;
        println!(
            "{md:<5}: read {} tokens in {:.2}s ({:.0} tokens/s), wrote {} in {:.2}s ({:.0} tokens/s); all {:.0} tokens/s ({:.0} us/token), int8 products {:.0}%",
            t.read,
            t.read_s,
            t.read as f64 / t.read_s,
            t.written,
            t.write_s,
            t.written as f64 / t.write_s.max(1e-9),
            (t.read + t.written) as f64 / all,
            1e6 * all / (t.read + t.written) as f64,
            100.0 * t.mm_s / all
        );
        if md == "fused" && secs.iter().any(|&s| s > 0) {
            let n = (t.read + t.written) as f64;
            for (name, ns) in ava_gpu::model::fstep::arch::SECTIONS.iter().zip(secs) {
                println!("    {name:<40} {:>7.1} us/token", ns as f64 / 1e3 / n);
            }
            println!("    {:<40} {:>7.1} us/token", "int8 products", 1e6 * t.mm_s / n);
        }
        tallies.push((md.clone(), t));
    }
    if tallies.len() == 2 {
        let (a, b) = (&tallies[0].1.replies, &tallies[1].1.replies);
        let differ: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
        println!("replies {} vs {}: {} of {} differ {:?}", tallies[0].0, tallies[1].0, differ.len(), a.len(), differ);
    }

    if check {
        let e = |e: candle_core::Error| e.to_string();
        let (mut steps, mut differ, mut worst, mut state_differ) = (0usize, 0usize, 0f32, 0usize);
        for ids in &prompts {
            let (mut sa, mut sb) = (model.stream().map_err(e)?, model.stream().map_err(e)?);
            let mut rng = SplitMix64::new(1);
            let mut toks: Vec<u32> = ids.clone();
            let mut i = 0;
            let mut out = Vec::new();
            while i < toks.len() {
                model.set_fused(false);
                let la = model.step(&mut sa, toks[i]).map_err(e)?;
                model.set_fused(true);
                let lb = model.step(&mut sb, toks[i]).map_err(e)?;
                steps += 1;
                worst = worst.max(max_abs(&la, &lb)?);
                differ += (bits(&la)? != bits(&lb)?) as usize;
                if i + 1 == toks.len() && out.len() + 1 < tokens {
                    // the plain step's greedy reply goes on (diary::reply's stop rule)
                    let tok = diary::pick(&la, 0.0, 1, &mut rng).map_err(e)?;
                    if tok != EOS {
                        out.push(tok);
                        if !diary::cut_reply(&bpe.decode(&strip_marks(&out))).1 {
                            toks.push(tok);
                        }
                    }
                }
                i += 1;
            }
            let (ta, tb) = (Model::stream_state(&sa).ok_or("not State-backed")?, Model::stream_state(&sb).ok_or("not State-backed")?);
            let ((xa, ca, ha), (xb, cb, hb)) = (ava_gpu::model::fstep::arch::state_tensors(ta), ava_gpu::model::fstep::arch::state_tensors(tb));
            let mut same = xa.len() == xb.len() && ca == cb && ha == hb;
            for (x, y) in xa.iter().zip(&xb) {
                same &= x.dims() == y.dims() && bits(x)? == bits(y)?;
            }
            state_differ += (!same) as usize;
        }
        model.set_fused(false);
        println!(
            "check, plain vs fused in lockstep over {} questions, {steps} steps: max |dlogit| {worst:e}, steps with any logit bit different {differ}, questions whose final State differs {state_differ}",
            prompts.len()
        );
    }
    Ok(())
}

/// --no-eco (Windows): opt this process out of execution-speed power throttling (EcoQoS), which Windows applies to a
/// windowless background process after a few seconds (on a hybrid CPU it then runs at about half speed); a diagnostic
/// for benchmarking only, nothing system-wide changes.
#[cfg(windows)]
fn no_eco() -> bool {
    #[repr(C)]
    struct State {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn SetProcessInformation(h: isize, class: i32, info: *const std::ffi::c_void, size: u32) -> i32;
    }
    // ProcessPowerThrottling = 4; PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 1, state 0: never throttled
    let s = State { version: 1, control_mask: 1, state_mask: 0 };
    unsafe { SetProcessInformation(GetCurrentProcess(), 4, &s as *const State as *const std::ffi::c_void, std::mem::size_of::<State>() as u32) != 0 }
}

#[cfg(not(windows))]
fn no_eco() -> bool {
    false
}
