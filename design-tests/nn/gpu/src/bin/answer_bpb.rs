// TESTING / DESIGN ONLY — answer_bpb: v40, how well she predicts held-out long answers (bits per byte of the answers).
//
// answer_bpb --model FILE [--baseline FILE]... --data FILE.sft [--arch D] [--batch-tokens 32768] [--resamples 10000]
//            [--bpe FILE]
//
// * --data: AVASFT01 examples ([EOS] prompt answer [EOS]; qa_prep --heldout writes the long answers that teaching
//   never saw). Glance models get length marks on the prompt and on the answer separately (grow_gpu's rule).
// * Scored: the answer's tokens and the closing EOS; bits per byte = total NLL / ln 2 / the answer's bytes (from
//   the tokenizer; marks are 0 bytes). The models' shapes come from their files (ckpt::load_auto).
// * Each baseline: model − baseline BPB with a 95% bootstrap over examples (a ratio of sums, recomputed per
//   resample; SplitMix64 seed 1). Last line: RESULT.

use ava_data::bpe::Bpe;
use ava_data::numbers::insert_marks;
use ava_data::rng::SplitMix64;
use ava_gpu::eval::byte_lengths_n;
use ava_gpu::model::Kind;
use ava_gpu::qa::read_sft;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: answer_bpb --model FILE [--baseline FILE]... --data FILE.sft [--arch D] [--batch-tokens 32768] [--resamples 10000] [--bpe FILE]";

fn main() {
    if let Err(e) = run() {
        eprintln!("answer_bpb: {e}");
        std::process::exit(1);
    }
}

/// Per example (nats, bytes) of the answer.
fn score(path: &PathBuf, kind: Kind, data: &PathBuf, bpe: &Bpe, batch: usize, dev: &candle_core::Device) -> Result<Vec<(f64, f64)>, String> {
    let (cfg, _vm, model) = ava_gpu::ckpt::load_auto(path, kind, 256, dev)?;
    model.set_track_activity(false);
    model.set_inference(std::env::var("AVA_FAST_INFER").map_or(true, |v| v != "0"));
    let ex = read_sft(data)?;
    let (mut seqs, mut from) = (Vec::with_capacity(ex.len()), Vec::with_capacity(ex.len()));
    for e in &ex {
        if cfg.glance {
            let mut s = insert_marks(&e.toks[..e.np]);
            from.push(s.len());
            s.extend(insert_marks(&e.toks[e.np..]));
            seqs.push(s);
        } else {
            from.push(e.np);
            seqs.push(e.toks.clone());
        }
    }
    let bytes = byte_lengths_n(bpe, cfg.vocab);
    let t0 = Instant::now();
    let nll = ava_gpu::mc::scored_nll(&model, dev, &seqs, &from, batch).map_err(|e| format!("score: {e}"))?;
    let ntok: usize = seqs.iter().map(Vec::len).sum();
    println!("  {}: {} examples, {ntok} tokens read in {:.1}s", path.display(), ex.len(), t0.elapsed().as_secs_f64());
    Ok(seqs
        .iter()
        .zip(&from)
        .zip(&nll)
        .map(|((s, &f), &(n, _))| (n, s[f..].iter().map(|&t| bytes.get(t as usize).copied().unwrap_or(0) as f64).sum()))
        .collect())
}

fn bpb(v: &[(f64, f64)], pick: &mut dyn Iterator<Item = usize>) -> f64 {
    let (n, b) = pick.fold((0.0, 0.0), |a, i| (a.0 + v[i].0, a.1 + v[i].1));
    n / std::f64::consts::LN_2 / b.max(1.0)
}

fn run() -> Result<(), String> {
    let (mut model, mut baselines, mut data) = (None::<PathBuf>, Vec::<PathBuf>::new(), None::<PathBuf>);
    let (mut kind, mut batch, mut resamples) = (Kind::D, 32768usize, 10000usize);
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
        match key {
            "--model" => model = Some(v.into()),
            "--baseline" => baselines.push(v.into()),
            "--data" => data = Some(v.into()),
            "--arch" => kind = Kind::parse(v)?,
            "--batch-tokens" => batch = num(v)?.max(1),
            "--resamples" => resamples = num(v)?.max(1),
            "--bpe" => bpe_path = v.into(),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let (model, data) = (model.ok_or(USAGE)?, data.ok_or(USAGE)?);
    let dev = ava_gpu::device(false).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let m = score(&model, kind, &data, &bpe, batch, &dev)?;
    let n = m.len();
    let full = bpb(&m, &mut (0..n));
    println!("model {}: answers {:.4} bits/byte", model.display(), full);
    let mut result = format!("RESULT n={n} bpb={full:.6}");
    for (j, bp) in baselines.iter().enumerate() {
        let b = score(bp, kind, &data, &bpe, batch, &dev)?;
        let bb = bpb(&b, &mut (0..n));
        let mut rng = SplitMix64::new(1);
        let mut ds: Vec<f64> = (0..resamples)
            .map(|_| {
                let pick: Vec<usize> = (0..n).map(|_| rng.below(n as u64) as usize).collect();
                bpb(&m, &mut pick.iter().copied()) - bpb(&b, &mut pick.iter().copied())
            })
            .collect();
        ds.sort_by(f64::total_cmp);
        let q = |p: f64| ds[((p * (resamples - 1) as f64).round() as usize).min(resamples - 1)];
        println!("baseline {}: {bb:.4} bits/byte; model - baseline {:+.4} [95%: {:+.4}, {:+.4}]", bp.display(), full - bb, q(0.025), q(0.975));
        let tag = if j == 0 { String::new() } else { (j + 1).to_string() };
        result += &format!(" base{tag}={bb:.6} diff{tag}={:.6} lo{tag}={:.6} hi{tag}={:.6}", full - bb, q(0.025), q(0.975));
    }
    println!("{result}");
    Ok(())
}
