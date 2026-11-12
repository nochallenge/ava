// TESTING / DESIGN ONLY — trace_check: on a real brain, the chat app's traced step changes nothing.
//
// trace_check --model FILE [--bpe FILE] [--threads 4] [--tokens 120] [--int8] [--fused] [--question TEXT]
//
// Two streams of the same brain read her taught prompt ("Question: ...\nAnswer:") and then answer greedily: stream A
// with `Model::step` (what she computes when nobody watches), stream B with `Model::step_trace` (what ava_live runs, the
// glow's record on). Pre-set bar: every token's logits are equal bit for bit (the answer is then the same word for
// word). Prints the time per token of each, and the first spikes/units of the trace as a sanity line. CPU only.

use ava_data::bpe::{Bpe, EOS};
use ava_gpu::diary;
use ava_gpu::model::{Kind, Model};
use candle_core::{DType, Device, Tensor};
use std::path::PathBuf;
use std::time::Instant;

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
}

fn main() {
    if let Err(e) = run() {
        eprintln!("trace_check: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut model, mut bpe, mut threads, mut tokens, mut int8, mut fused) = (None, PathBuf::from(ava_gpu::DEFAULT_BPE), 4usize, 120usize, false, false);
    let mut question = "What is the sun made of?".to_string();
    let mut i = 0;
    while i < args.len() {
        let v = || args.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", args[i]));
        match args[i].as_str() {
            "--model" => model = Some(PathBuf::from(v()?)),
            "--bpe" => bpe = PathBuf::from(v()?),
            "--threads" => threads = v()?.parse().map_err(|_| "bad --threads")?,
            "--tokens" => tokens = v()?.parse().map_err(|_| "bad --tokens")?,
            "--question" => question = v()?,
            "--int8" | "--fused" => {
                if args[i] == "--int8" { int8 = true } else { fused = true }
                i += 1;
                continue;
            }
            o => return Err(format!("unknown option {o}")),
        }
        i += 2;
    }
    let model = model.ok_or("--model FILE")?;
    diary::set_threads(threads);
    let dev = Device::Cpu;
    let bpe = Bpe::load(&bpe).map_err(|e| format!("bpe: {e}"))?;
    let (cfg, _vm, m) = ava_gpu::ckpt::load_auto(&model, Kind::D, 4096, &dev)?;
    m.set_track_activity(false);
    if int8 {
        m.set_int8(true).map_err(|e| e.to_string())?;
    }
    m.set_fused(fused);
    let (ids, _) = diary::prompt_ids_ctx(&bpe, cfg.glance, &[], &question, "", 0);
    let e = |e: candle_core::Error| e.to_string();
    let (mut a, mut b) = (m.stream().map_err(e)?, m.stream().map_err(e)?);
    let state_backed = Model::stream_state(&a).is_some();
    let (mut ta, mut tb, mut n, mut same) = (0f64, 0f64, 0usize, 0usize);
    let mut feed = |tok: u32, n: &mut usize| -> Result<Tensor, String> {
        let t = Instant::now();
        let la = m.step(&mut a, tok).map_err(e)?;
        ta += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (lb, tr) = m.step_trace(&mut b, tok).map_err(e)?;
        tb += t.elapsed().as_secs_f64();
        if *n == 0 {
            let l0 = &tr.layers[0];
            let s: Vec<f32> = l0.spikes.flatten_all().map_err(e)?.to_vec1().map_err(e)?;
            let w: Vec<f32> = l0.winners.flatten_all().map_err(e)?.to_vec1().map_err(e)?;
            println!(
                "first token's trace: {} blocks; block 0: {} of {} neurons spiking, {} of {} MLP winners; memory sizes {}; input size {:.4}",
                tr.layers.len(),
                s.iter().filter(|v| v.abs() > 0.5).count(),
                s.len(),
                w.iter().filter(|v| **v > 0.5).count(),
                w.len(),
                if l0.mem_write.is_some() && l0.mem_read.is_some() { "write + read" } else if l0.mem_write.is_some() { "write" } else { "none" },
                tr.input_rms
            );
        }
        *n += 1;
        if bits(&la) == bits(&lb) {
            same += 1;
        }
        Ok(la)
    };
    let mut last = None;
    for &t in &ids {
        last = Some(feed(t, &mut n)?);
    }
    let mut out = Vec::new();
    for _ in 0..tokens {
        let l = last.take().ok_or("no logits")?;
        let tok = l.argmax(0).map_err(e)?.to_scalar::<u32>().map_err(e)?;
        if tok == EOS {
            break;
        }
        out.push(tok);
        last = Some(feed(tok, &mut n)?);
    }
    let said = bpe.decode(&ava_data::numbers::strip_marks(&out));
    println!(
        "{}: {} {}, d {}, {} layers{}; int8 {int8}, fused {fused}, {threads} threads",
        model.display(),
        cfg.kind.name(),
        if state_backed { "(block options, State-backed)" } else { "(plain stream)" },
        cfg.d,
        cfg.layers,
        if cfg.engram.is_some() { ", engram" } else { "" }
    );
    println!("read {} tokens, said {}: {:?}", ids.len(), out.len(), said.trim());
    println!("logits identical bit for bit on {same} of {n} tokens: {}", if same == n { "PASS" } else { "FAIL" });
    println!("step {:.2} ms/token ({:.0} tok/s); step_trace {:.2} ms/token ({:.0} tok/s)", 1e3 * ta / n as f64, n as f64 / ta, 1e3 * tb / n as f64, n as f64 / tb);
    if same != n {
        return Err("tracing changed her logits".into());
    }
    Ok(())
}
