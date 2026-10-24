// TESTING / DESIGN ONLY — generate_gpu: sample text from a train_gpu checkpoint (model.safetensors).
//
// generate_gpu --model FILE --d D --layers L [--ctx 256] [--prompt TEXT] [--tokens 100]
//              [--temperature 1.0] [--top-k 40] [--seed 1] [--cpu] [--bpe FILE]
//
// No KV cache: every new token re-runs the whole context (the LIF scan is recurrent anyway). The
// context is the last `ctx` tokens of prompt + generated. Sampling: logits / temperature, keep the
// top-k, softmax, draw with ava_data's SplitMix64. Generates --tokens tokens; EOS renders as a newline.

use ava_data::bpe::{Bpe, EOS};
use ava_data::rng::SplitMix64;
use ava_gpu::model::{Config, Model, VOCAB};
use candle_core::{DType, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use std::io::Write;
use std::path::PathBuf;

const USAGE: &str = "usage: generate_gpu --model FILE --d D --layers L [--ctx 256] [--prompt TEXT] [--tokens 100]\n\
    \x20               [--temperature 1.0] [--top-k 40] [--seed 1] [--cpu] [--bpe FILE]";

fn main() {
    if let Err(e) = run() {
        eprintln!("generate_gpu: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut model_path: Option<PathBuf> = None;
    let (mut d, mut layers, mut ctx) = (0usize, 0usize, 256usize);
    let mut prompt = String::new();
    let mut max_tokens = 100usize;
    let mut temperature = 1.0f32;
    let mut top_k = 40usize;
    let mut seed = 1u64;
    let mut cpu = false;
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        if key == "--cpu" {
            cpu = true;
            i += 1;
            continue;
        }
        let val = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
        match key {
            "--model" => model_path = Some(PathBuf::from(val)),
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
    let cfg = Config::new(VOCAB, d, layers, ctx);
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Model::new(cfg, vb).map_err(|e| format!("model: {e}"))?;
    varmap.load(&model_path).map_err(|e| format!("load {}: {e}", model_path.display()))?;

    let mut ids: Vec<u32> = if prompt.is_empty() { vec![EOS] } else { bpe.encode(&prompt) };
    if ids.is_empty() {
        ids.push(EOS);
    }
    let prompt_len = ids.len();
    let mut rng = SplitMix64::new(seed);
    let top_k = top_k.max(1).min(VOCAB);
    for _ in 0..max_tokens {
        let start = ids.len().saturating_sub(ctx);
        let window = &ids[start..];
        let t = window.len();
        let x = Tensor::from_slice(window, (1, t), &dev).map_err(|e| e.to_string())?;
        let logits = model.forward(&x).map_err(|e| format!("forward: {e}"))?; // [t, V]
        let last = logits.narrow(0, t - 1, 1).map_err(|e| e.to_string())?.squeeze(0).map_err(|e| e.to_string())?;
        let last = last.affine(1.0 / temperature.max(1e-6) as f64, 0.0).map_err(|e| e.to_string())?;
        let last = candle_nn::ops::softmax(&last, D::Minus1).map_err(|e| e.to_string())?;
        let probs: Vec<f32> = last.to_vec1().map_err(|e| e.to_string())?;
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
        let pick = pick as u32;
        ids.push(pick); // EOS is kept and rendered as a newline
    }
    let text = ids[prompt_len..].split(|&t| t == EOS).map(|seg| bpe.decode(seg)).collect::<Vec<_>>().join("
");
    println!("{prompt}{text}");
    std::io::stdout().flush().ok();
    Ok(())
}
