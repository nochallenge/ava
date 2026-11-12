// TESTING / DESIGN ONLY — v50_talk: talking speed for runs/v50/prereg_v50.md: CPU, one-token stream, greedy.
//
// v50_talk --model FILE [--tokens 200] [--threads 8] [--repeats 3] [--prompt TEXT] [--bpe FILE] [--arch D] [--int8] [--fused]
// * Reads the prompt one token at a time (Model::step), then writes --tokens tokens greedily, one step each; the
//   time of the written tokens is the talking speed (tokens/s). --repeats times on fresh streams; the median is the
//   result (one warm-up pass first, not counted).
// * Default path: f32, no int8, no fused step, the same for every v50 arm (--dale, --loop and --neuromod have no
//   int8 or fused step yet). --dale's sign ⊙ softplus(v) weights are computed once (Model::set_dale_cache), as a chat
//   app would; nothing changes for a model without --dale.
// * Prints one line `talk: ... tokens_per_sec=X us_per_token=Y` for the runner to read.

use ava_data::bpe::Bpe;
use ava_gpu::diary;
use ava_gpu::model::Kind;
use ava_data::rng::SplitMix64;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("v50_talk: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let num = |k: &str, d: usize| -> Result<usize, String> { get(k).map(|v| v.parse().map_err(|_| format!("bad {k}"))).transpose().map(|v| v.unwrap_or(d)) };
    let model_path = PathBuf::from(get("--model").ok_or("--model FILE")?);
    let (tokens, threads, repeats) = (num("--tokens", 200)?, num("--threads", 8)?, num("--repeats", 3)?.max(1));
    diary::set_threads(threads);
    let kind = Kind::parse(&get("--arch").unwrap_or_else(|| "D".into()))?;
    let bpe_path = PathBuf::from(get("--bpe").unwrap_or_else(|| ava_gpu::DEFAULT_BPE.into()));
    let prompt = get("--prompt").unwrap_or_else(|| "Once upon a time, a little girl named Lily found a box in the garden. She".into());
    let (int8, fused) = (args.iter().any(|a| a == "--int8"), args.iter().any(|a| a == "--fused"));

    let dev = ava_gpu::device(true).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let (cfg, _varmap, model) = ava_gpu::ckpt::load_auto(&model_path, kind, 4096, &dev)?;
    model.set_track_activity(false);
    model.set_inference(true);
    model.set_int8(int8).map_err(|e| format!("int8: {e}"))?;
    model.set_fused(fused);
    model.set_dale_cache(true);
    let ids = bpe.encode(&prompt);
    if ids.is_empty() {
        return Err("empty prompt".into());
    }
    let e = |e: candle_core::Error| e.to_string();
    // one pass: read the prompt, then write `tokens` greedily; returns (seconds writing, the text written)
    let pass = |n: usize| -> Result<(f64, Vec<u32>), String> {
        let mut st = model.stream().map_err(e)?;
        let mut logits = None;
        for &t in &ids {
            logits = Some(model.step(&mut st, t).map_err(e)?);
        }
        let mut rng = SplitMix64::new(1);
        let mut out = Vec::with_capacity(n);
        let t0 = Instant::now();
        for _ in 0..n {
            let tok = diary::pick(logits.as_ref().expect("prompt read"), 0.0, 1, &mut rng).map_err(e)?;
            out.push(tok);
            logits = Some(model.step(&mut st, tok).map_err(e)?);
        }
        Ok((t0.elapsed().as_secs_f64(), out))
    };
    pass(tokens.min(20))?; // warm-up
    let mut secs = Vec::new();
    let mut text = Vec::new();
    for _ in 0..repeats {
        let (s, out) = pass(tokens)?;
        secs.push(s);
        text = out;
    }
    secs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let med = secs[secs.len() / 2];
    let reply: String = bpe.decode(&text).chars().take(120).collect::<String>().replace(['\n', '\t'], " ");
    println!(
        "model {} (d {}, {} layers), {} threads, path {}{}, prompt {} tokens, wrote {tokens} x {repeats}: {:?} s",
        model_path.display(),
        cfg.d,
        cfg.layers,
        candle_core::utils::get_num_threads(),
        if int8 { "int8" } else { "f32" },
        if fused { format!(" fused ({})", model.fused_blocker().unwrap_or("applies")) } else { String::new() },
        ids.len(),
        secs.iter().map(|s| (s * 1e3).round() / 1e3).collect::<Vec<_>>()
    );
    println!("reply: {reply}");
    println!("talk: tokens={tokens} repeats={repeats} tokens_per_sec={:.1} us_per_token={:.1}", tokens as f64 / med, 1e6 * med / tokens as f64);
    Ok(())
}
