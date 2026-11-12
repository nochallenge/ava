// TESTING / DESIGN ONLY — v51_talk: CPU talking speed of a trained model (runs/v51/prereg_v51.md, Amendment A2): one token at a time.
//
// v51_talk --model FILE [--bpe FILE] [--tokens 200] [--repeats 5] [--threads 4] [--compact] [--fused]
// * The model is loaded with its arch.txt (ckpt::load_auto_lean), CPU, the weights as plain tensors. A fixed English
//   text (with the glance's marks when the model has them) is read one token at a time on a fresh stream
//   (Model::step): 20 tokens to warm up, then --tokens timed; one untimed pass, then --repeats runs, the median tokens/s.
// * Modes, each its own line: "f32" (int8 off, fused off: the same State step for every arm, the primary measure),
//   "int8" (int8 on, fused off), and with --fused "int8_fused" (int8 on, fused on where the fused step applies; the
//   line says whether it applied).
// * --compact (a --prune model): the cut units removed first (v51::compact_pruned, a plain model with hidden 2d); the
//   logits of the first 40 tokens are checked against the masked model's (max |diff| printed).
// Output: "RESULT mode=M tokens=N tok_per_s=X ms_per_tok=Y fused=yes|no" per mode.

use ava_data::bpe::Bpe;
use ava_gpu::model::{Kind, Model};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use std::time::Instant;

const TEXT: &str = "The old lighthouse stood at the edge of the cliff, and every night its keeper climbed the narrow stairs to light the lamp. \
Ships passing in the dark could see the beam from twenty miles away. One winter a storm broke the great window, and the keeper \
spent three days repairing it with glass carried up from the village. When the children asked him why he stayed alone in such a \
cold place, he said that someone had to keep the light burning, because a single ship saved was worth a lifetime of quiet work. \
In spring the fishermen brought him bread and news, and he told them which nights the fog would come.";

fn main() {
    if let Err(e) = run() {
        eprintln!("v51_talk: {e}");
        std::process::exit(1);
    }
}

fn speed(model: &Model, toks: &[u32], warm: usize, n: usize, repeats: usize) -> Result<f64, String> {
    let e = |e: candle_core::Error| e.to_string();
    let mut rates = Vec::with_capacity(repeats + 1);
    // one untimed pass first (caches, thread pool), then the timed repeats
    for _ in 0..repeats + 1 {
        let mut st = model.stream().map_err(e)?;
        for &t in &toks[..warm] {
            model.step(&mut st, t).map_err(e)?;
        }
        let t0 = Instant::now();
        for &t in &toks[warm..warm + n] {
            model.step(&mut st, t).map_err(e)?;
        }
        rates.push(n as f64 / t0.elapsed().as_secs_f64());
    }
    rates.remove(0);
    rates.sort_by(|a, b| a.total_cmp(b));
    Ok(rates[rates.len() / 2])
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let num = |k: &str, d: usize| -> Result<usize, String> { get(k).map(|v| v.parse().map_err(|_| format!("bad {k}"))).transpose().map(|v| v.unwrap_or(d)) };
    let flag = |k: &str| args.iter().any(|a| a == k);
    let model_path = PathBuf::from(get("--model").ok_or("--model FILE")?);
    let (n, repeats, threads) = (num("--tokens", 200)?, num("--repeats", 5)?, num("--threads", 4)?);
    if threads > 4 {
        return Err("--threads: at most 4 (other jobs share the machine)".into());
    }
    ava_gpu::diary::set_threads(threads);
    let bpe_path = PathBuf::from(get("--bpe").unwrap_or_else(|| ava_gpu::DEFAULT_BPE.into()));
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let dev = Device::Cpu;
    // the letters (--letters) from the same BPE file, before the model is built
    let t = ava_gpu::ckpt::read(&model_path)?;
    let vocab = t.get("tok_emb").ok_or("no tok_emb")?.dims()[0];
    drop(t);
    ava_gpu::v51::set_letters_from_bpe(&bpe_path, vocab)?;
    let (cfg, varmap, mut model) = ava_gpu::ckpt::load_auto_lean(&model_path, Kind::D, 4096, &dev)?;
    let warm = 20;
    let mut toks = ava_gpu::diary::encode(&bpe, cfg.glance, TEXT);
    while toks.len() < warm + n {
        let again = toks.clone();
        toks.extend(again);
    }
    let mut note = String::new();
    if flag("--compact") {
        let w: std::collections::HashMap<String, Tensor> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.as_tensor().detach())).collect();
        let (c2, w2) = ava_gpu::v51::compact_pruned(&cfg, &w)?;
        let m2 = Model::new(c2, VarBuilder::from_tensors(w2, DType::F32, &dev)).map_err(|e| format!("compacted model: {e}"))?;
        // the same logits on the first 40 tokens
        let ids = Tensor::new(&toks[..40], &dev).and_then(|x| x.unsqueeze(0)).map_err(|e| e.to_string())?;
        let (a, b) = (model.forward(&ids).map_err(|e| e.to_string())?, m2.forward(&ids).map_err(|e| e.to_string())?);
        let d = a.sub(&b).and_then(|x| x.abs()).and_then(|x| x.flatten_all()).and_then(|x| x.max(0)).and_then(|x| x.to_scalar::<f32>()).map_err(|e| e.to_string())?;
        note = format!(" compacted (hidden {} -> {}, max |logit diff| {d:.2e})", cfg.hidden(), c2.hidden());
        model = m2;
    }
    model.set_track_activity(false);
    model.set_inference(true);
    println!(
        "model {} (d {}, {} layers, vocab {}{}){note}, {} threads",
        model_path.display(),
        cfg.d,
        cfg.layers,
        cfg.vocab,
        model.cfg.arch.signature(),
        candle_core::utils::get_num_threads()
    );
    let mut modes = vec![("f32", false, false), ("int8", true, false)];
    if flag("--fused") {
        modes.push(("int8_fused", true, true));
    }
    for (name, q8, fused) in modes {
        model.set_int8(q8).map_err(|e| format!("int8: {e}"))?;
        model.set_fused(fused);
        let applied = fused && model.fused_blocker().is_none();
        if fused && !applied {
            println!("fused step does not apply: {}", model.fused_blocker().unwrap_or("?"));
        }
        let r = speed(&model, &toks, warm, n, repeats)?;
        println!("RESULT mode={name} tokens={n} tok_per_s={r:.1} ms_per_tok={:.3} fused={}", 1000.0 / r, if applied { "yes" } else { "no" });
    }
    Ok(())
}
