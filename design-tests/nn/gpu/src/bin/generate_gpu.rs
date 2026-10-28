// TESTING / DESIGN ONLY — generate_gpu: sample text from a train_gpu checkpoint (model.safetensors).
//
// generate_gpu --model FILE [--arch D|S|A|T|F|G|R] --d D --layers L [--ctx 256] [--prompt TEXT] [--tokens 100]
//              [--temperature 1.0] [--top-k 40] [--seed 1] [--cpu] [--bpe FILE] [--window] [--check] [--qa] [--chat] [--glance]
//
// Ava (every kind but T) keeps her LIF membranes and fast weights from token to token (`Model::step`;
// A also keeps a key/value cache for its attention blocks): each new token costs one
// position, and her memory is not cut at `ctx`. --window restores the old way (every token re-runs the
// last `ctx` tokens from a zero state), which T always uses. --check compares the carried-state logits
// with one full forward over the first `ctx` tokens. Sampling: logits / temperature, keep the top-k,
// softmax, draw with ava_data's SplitMix64. Generates --tokens tokens; EOS renders as a newline. Speed
// goes to stderr.
//
// v37: --qa asks her --prompt as a question in the format she was taught ("Question: ...\nAnswer:", see
// ava_gpu::qa) and prints her answer, which ends at her EOS (or after --tokens tokens). --chat does the same for
// every line typed on stdin; each question starts from a fresh memory.
//
// v39 --glance: a model D trained with the glance (vocab 4112). The prompt gets its length marks
// (ava_data::numbers::insert_marks); she writes her own marks, which are not printed.

use ava_data::bpe::{Bpe, EOS};
use ava_data::numbers::{insert_marks, strip_marks};
use ava_data::rng::SplitMix64;
use ava_gpu::model::{Kind, Model};
use candle_core::{Device, Tensor, D};
use std::io::Write;
use std::path::PathBuf;

const USAGE: &str = "usage: generate_gpu --model FILE [--arch D|S|A|T|F|G|R] --d D --layers L [--ctx 256] [--prompt TEXT] [--tokens 100]\n\
    \x20               [--temperature 1.0] [--top-k 40] [--seed 1] [--cpu] [--bpe FILE] [--window] [--check] [--qa] [--chat] [--glance]";

fn main() {
    if let Err(e) = run() {
        eprintln!("generate_gpu: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut model_path: Option<PathBuf> = None;
    let (mut d, mut layers, mut ctx) = (0usize, 0usize, 256usize);
    let mut kind = Kind::S;
    let mut prompt = String::new();
    let mut max_tokens = 100usize;
    let mut temperature = 1.0f32;
    let mut top_k = 40usize;
    let mut seed = 1u64;
    let (mut cpu, mut window, mut check, mut qa, mut chat, mut glance) = (false, false, false, false, false, false);
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        if matches!(key, "--cpu" | "--window" | "--check" | "--qa" | "--chat" | "--glance") {
            match key {
                "--cpu" => cpu = true,
                "--window" => window = true,
                "--qa" => qa = true,
                "--chat" => chat = true,
                "--glance" => glance = true,
                _ => check = true,
            }
            i += 1;
            continue;
        }
        let val = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
        match key {
            "--model" => model_path = Some(PathBuf::from(val)),
            "--arch" => kind = Kind::parse(val)?,
            "--d" => d = num(val)?,
            "--layers" => layers = num(val)?,
            "--ctx" => ctx = num(val)?,
            "--prompt" => prompt = val.to_string(),
            "--tokens" => max_tokens = num(val)?,
            "--temperature" => temperature = val.parse().map_err(|_| format!("bad number {val:?}"))?,
            "--top-k" => top_k = num(val)?,
            "--seed" => seed = val.parse().map_err(|_| format!("bad seed {val:?}"))?,
            "--bpe" => bpe_path = PathBuf::from(val),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let model_path = model_path.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    if d == 0 || layers == 0 {
        return Err("--d and --layers are required".into());
    }

    let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let (cfg, _varmap, model) = ava_gpu::ckpt::load_expect(&model_path, kind, ctx, d, layers, glance, &dev)?;

    if qa || chat {
        // v37: ask her a question in the format she was taught (qa::prompt_text); she answers until her EOS.
        if !kind.is_ava() || window {
            return Err("--qa and --chat use the carried state (an Ava kind, without --window)".into());
        }
        let mut rng = SplitMix64::new(seed);
        let top_k = top_k.max(1).min(cfg.vocab);
        let mut ask = |q: &str| -> Result<String, String> {
            let e = |e: candle_core::Error| e.to_string();
            let mut ids = vec![EOS];
            ids.extend(bpe.encode(&ava_gpu::qa::prompt_text("", q)));
            if glance {
                ids = insert_marks(&ids);
            }
            let mut st = model.stream().map_err(e)?;
            let mut next = None;
            for &t in &ids {
                next = Some(model.step(&mut st, t).map_err(e)?);
            }
            let mut next = next.expect("prompt has tokens");
            let mut out = Vec::new();
            for _ in 0..max_tokens {
                let pick = sample(&next, temperature, top_k, &mut rng).map_err(e)?;
                if pick == EOS {
                    break;
                }
                out.push(pick);
                next = model.step(&mut st, pick).map_err(e)?;
            }
            Ok(bpe.decode(&strip_marks(&out)).trim().to_string())
        };
        if qa {
            println!("Question: {}\nAnswer: {}", prompt.trim(), ask(&prompt)?);
            return Ok(());
        }
        eprintln!("Ask Ava a question (an empty line or quit ends). Each question starts from a fresh memory.");
        let stdin = std::io::stdin();
        loop {
            eprint!("You: ");
            std::io::stderr().flush().ok();
            let mut line = String::new();
            if stdin.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                break;
            }
            let q = line.trim();
            if q.is_empty() || q == "quit" || q == "exit" {
                break;
            }
            println!("Ava: {}", ask(q)?);
        }
        return Ok(());
    }

    let mut ids: Vec<u32> = if prompt.is_empty() { vec![EOS] } else { bpe.encode(&prompt) };
    if glance {
        ids = insert_marks(&ids);
    }
    if ids.is_empty() {
        ids.push(EOS);
    }
    let prompt_len = ids.len();
    let mut rng = SplitMix64::new(seed);
    let top_k = top_k.max(1).min(cfg.vocab);
    let e = |e: candle_core::Error| e.to_string();
    // S carries its LIF state from token to token; T (or --window) re-runs the last `ctx` tokens.
    let mut stream = if kind.is_ava() && !window { Some(model.stream().map_err(e)?) } else { None };
    let t0 = std::time::Instant::now();
    let mut next = match &mut stream {
        Some(st) => {
            let mut last = None;
            for &t in &ids {
                last = Some(model.step(st, t).map_err(e)?);
            }
            last.expect("prompt has at least one token")
        }
        None => window_logits(&model, &ids, ctx, &dev).map_err(e)?,
    };
    for n in 0..max_tokens {
        let pick = sample(&next, temperature, top_k, &mut rng).map_err(e)?;
        ids.push(pick); // EOS is kept and rendered as a newline
        if n + 1 < max_tokens {
            next = match &mut stream {
                Some(st) => model.step(st, pick).map_err(e)?,
                None => window_logits(&model, &ids, ctx, &dev).map_err(e)?,
            };
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let text = ids[prompt_len..].split(|&t| t == EOS).map(|seg| bpe.decode(&strip_marks(seg))).collect::<Vec<_>>().join("
");
    println!("{prompt}{text}");
    eprintln!(
        "[{} tokens ({} prompt + {} new) in {secs:.2}s, {:.0} new tokens/s, {}]",
        ids.len(),
        prompt_len,
        ids.len() - prompt_len,
        (ids.len() - prompt_len) as f64 / secs.max(1e-9),
        if stream.is_some() { "carried LIF state" } else { "window re-run" }
    );
    if check {
        check_stream(&model, &ids, ctx, &dev).map_err(e)?;
    }
    std::io::stdout().flush().ok();
    Ok(())
}

/// Logits [V] after the last `ctx` tokens of `ids`, by a full forward over that window.
fn window_logits(model: &Model, ids: &[u32], ctx: usize, dev: &Device) -> candle_core::Result<Tensor> {
    let window = &ids[ids.len().saturating_sub(ctx)..];
    let t = window.len();
    let logits = model.forward(&Tensor::from_slice(window, (1, t), dev)?)?; // [t, V]
    logits.narrow(0, t - 1, 1)?.squeeze(0)
}

/// logits / temperature, keep the top-k, softmax, one draw.
fn sample(logits: &Tensor, temperature: f32, top_k: usize, rng: &mut SplitMix64) -> candle_core::Result<u32> {
    let last = logits.affine(1.0 / temperature.max(1e-6) as f64, 0.0)?;
    let probs: Vec<f32> = candle_nn::ops::softmax(&last, D::Minus1)?.to_vec1()?;
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    order.truncate(top_k);
    let total: f64 = order.iter().map(|&j| probs[j] as f64).sum();
    let mut r = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
    let mut pick = order[order.len() - 1];
    for &j in &order {
        r -= probs[j] as f64;
        if r <= 0.0 {
            pick = j;
            break;
        }
    }
    Ok(pick as u32)
}

/// --check: the carried-state logits against one full forward over the first min(len, ctx) tokens,
/// position by position (max |difference|, positions whose top token differs).
fn check_stream(model: &Model, ids: &[u32], ctx: usize, dev: &Device) -> candle_core::Result<()> {
    let n = ids.len().min(ctx);
    let full = model.forward(&Tensor::from_slice(&ids[..n], (1, n), dev)?)?; // [n, V]
    let mut st = model.stream()?;
    let (mut max_diff, mut argmax_diff) = (0f32, 0usize);
    for (p, &t) in ids[..n].iter().enumerate() {
        let a = model.step(&mut st, t)?;
        let b = full.narrow(0, p, 1)?.squeeze(0)?;
        max_diff = max_diff.max(a.sub(&b)?.abs()?.max_all()?.to_scalar::<f32>()?);
        if a.argmax(0)?.to_scalar::<u32>()? != b.argmax(0)?.to_scalar::<u32>()? {
            argmax_diff += 1;
        }
    }
    eprintln!("[check over {n} positions: max |logit difference| {max_diff:.3e}, top-token disagreements {argmax_diff}]");
    Ok(())
}
