// TESTING / DESIGN ONLY — blimp_gpu: BLiMP minimal pairs for a train_gpu checkpoint (model.safetensors).
//
// blimp_gpu --model FILE --arch S|F|G|T --d D --layers L [--ctx 256] [--dir data/blimp] [--batch 256]
//           [--limit N per phenomenon] [--cpu] [--bpe FILE]
//
// Same method as brain/examples/blimp_score.rs: every sentence is scored as [EOS] sentence [EOS]
// (`Bpe::encode_doc` appends the closing EOS; the first EOS is context, not scored), and the model is
// right when the good sentence gets the higher total log-probability (BLiMP's simple LM method, no length
// normalisation). Chance is 50%. Sentences are scored in batches sorted by length and right-padded with
// PAD; every model here is causal, so padding after a sentence cannot change its scores, and PAD
// targets are not scored.

use ava_data::bpe::{Bpe, EOS, PAD};
use ava_gpu::model::{Config, Kind, Model, VOCAB};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use std::collections::BTreeMap;
use std::path::PathBuf;

const USAGE: &str = "usage: blimp_gpu --model FILE --arch S|F|G|T --d D --layers L [--ctx 256] [--dir DIR] [--batch 256]\n\
    \x20                [--limit N] [--cpu] [--bpe FILE]";

fn main() {
    if let Err(e) = run() {
        eprintln!("blimp_gpu: {e}");
        std::process::exit(1);
    }
}

/// The string value of `"key": "..."` in one JSON line (handles \" and \\ escapes).
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

/// Total NLL in nats of every sequence (the first token is context and not scored).
fn seq_nll(model: &Model, dev: &Device, seqs: &[Vec<u32>], batch: usize) -> candle_core::Result<Vec<f64>> {
    let mut out = vec![0.0f64; seqs.len()];
    let mut order: Vec<usize> = (0..seqs.len()).collect();
    order.sort_by_key(|&i| seqs[i].len());
    for chunk in order.chunks(batch.max(1)) {
        let t = chunk.iter().map(|&i| seqs[i].len()).max().unwrap_or(2).saturating_sub(1).max(1);
        let b = chunk.len();
        let (mut ids, mut tg, mut keep) = (Vec::with_capacity(b * t), Vec::with_capacity(b * t), Vec::with_capacity(b * t));
        for &i in chunk {
            let s = &seqs[i];
            for p in 0..t {
                let real = p + 1 < s.len();
                ids.push(if real { s[p] } else { PAD });
                tg.push(if real { s[p + 1] } else { 0 });
                keep.push(if real { 1f32 } else { 0f32 });
            }
        }
        let logits = model.forward(&Tensor::from_vec(ids, (b, t), dev)?)?; // [b·t, V]
        let lp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let picked = lp.gather(&Tensor::from_vec(tg, (b * t, 1), dev)?, 1)?.squeeze(1)?;
        let nll: Vec<f32> = picked.neg()?.mul(&Tensor::from_vec(keep, b * t, dev)?)?.reshape((b, t))?.sum(1)?.to_vec1()?;
        for (r, &i) in chunk.iter().enumerate() {
            out[i] = nll[r] as f64;
        }
    }
    Ok(out)
}

fn run() -> Result<(), String> {
    let mut model_path: Option<PathBuf> = None;
    let (mut d, mut layers, mut ctx, mut batch, mut limit) = (0usize, 0usize, 256usize, 256usize, usize::MAX);
    let mut kind = Kind::S;
    let mut cpu = false;
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let mut dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/blimp"));
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
            "--arch" => kind = Kind::parse(val)?,
            "--d" => d = num(val)?,
            "--layers" => layers = num(val)?,
            "--ctx" => ctx = num(val)?,
            "--batch" => batch = num(val)?,
            "--limit" => limit = num(val)?,
            "--dir" => dir = PathBuf::from(val),
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
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Model::new(Config::with_kind(kind, VOCAB, d, layers, ctx), vb).map_err(|e| format!("model: {e}"))?;
    varmap.load(&model_path).map_err(|e| format!("load {}: {e}", model_path.display()))?;

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("blimp dir {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    let (mut right, mut total) = (0usize, 0usize);
    let mut by_field: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let t0 = std::time::Instant::now();
    for f in &files {
        let text = std::fs::read_to_string(f).map_err(|e| format!("read {}: {e}", f.display()))?;
        let mut fld = String::new();
        let mut seqs = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()).take(limit) {
            let (Some(g), Some(b)) = (field(line, "sentence_good"), field(line, "sentence_bad")) else { continue };
            if fld.is_empty() {
                fld = field(line, "field").unwrap_or_default();
            }
            for s in [g, b] {
                let mut t = vec![EOS];
                t.extend(bpe.encode_doc(&s));
                seqs.push(t);
            }
        }
        let nll = seq_nll(&model, &dev, &seqs, batch).map_err(|e| format!("score: {e}"))?;
        let n = nll.len() / 2;
        let r = (0..n).filter(|&k| nll[2 * k] < nll[2 * k + 1]).count(); // lower NLL = higher probability
        println!("{:<55} {:>5.1}%  ({r}/{n})", f.file_stem().unwrap().to_string_lossy(), 100.0 * r as f64 / n.max(1) as f64);
        let e = by_field.entry(fld).or_default();
        e.0 += r;
        e.1 += n;
        right += r;
        total += n;
    }
    println!();
    for (k, (r, n)) in &by_field {
        println!("field {:<20} {:>5.1}%  ({r}/{n})", k, 100.0 * *r as f64 / (*n).max(1) as f64);
    }
    println!(
        "\nBLiMP overall: {:.1}% ({right}/{total}) in {:.0}s; chance 50%",
        100.0 * right as f64 / total.max(1) as f64,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
