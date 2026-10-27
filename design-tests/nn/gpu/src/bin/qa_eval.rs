// TESTING / DESIGN ONLY — qa_eval: multiple-choice questions for a train_gpu / sft_gpu checkpoint (v37).
//
// qa_eval --model FILE [--baseline FILE] --arch D --d 384 --layers 8 --set sciq-dev|sciq-test|arc-easy-dev|arc-challenge-dev
//         [--passage] [--tau X] [--batch-tokens 32768] [--limit N] [--out FILE] [--cpu] [--bpe FILE]
//
// Every choice is scored as an answer:  [EOS] prompt " choice" [EOS], with prompt = qa::prompt_text (the passage
// too with --passage), and only the choice's tokens and the closing EOS are scored (total log-probability, no
// length normalisation; the mean per token is reported beside it). Her pick is the choice with the lowest
// total NLL; she is right only if the right choice is strictly lower than every other (a tie counts as wrong).
//
// She may also say "I don't know": her margin is (second-lowest NLL − lowest NLL), and with a threshold tau she
// answers only when margin >= tau. Without --tau, the tau that gives the most right − wrong on this set is found
// and printed (for a dev set; a test set takes the tau fixed on dev).
//
// --baseline FILE scores a second model on the same questions and prints the paired bootstrap 95% interval of
// the difference in accuracy (model − baseline; 20,000 resamples, SplitMix64 seed 1). --out writes one line per
// question and model: index, right choice, pick, right (0/1), margin.

use ava_data::bpe::{Bpe, EOS, PAD};
use ava_data::rng::SplitMix64;
use ava_gpu::model::{Config, Kind, Model, VOCAB};
use ava_gpu::qa::{eval_set, prompt_text, Mc};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use std::io::Write;
use std::path::PathBuf;

const USAGE: &str = "usage: qa_eval --model FILE [--baseline FILE] --arch D --d D --layers L --set NAME [--passage] [--tau X]\n\
    \x20              [--batch-tokens 32768] [--limit N] [--out FILE] [--cpu] [--bpe FILE]";

fn main() {
    if let Err(e) = run() {
        eprintln!("qa_eval: {e}");
        std::process::exit(1);
    }
}

/// (total NLL in nats, scored tokens) of every sequence; positions whose target index is >= from[i] are scored.
fn scored_nll(model: &Model, dev: &Device, seqs: &[Vec<u32>], from: &[usize], batch: usize) -> candle_core::Result<Vec<(f64, usize)>> {
    let mut out = vec![(0.0f64, 0usize); seqs.len()];
    let mut order: Vec<usize> = (0..seqs.len()).collect();
    order.sort_by_key(|&i| seqs[i].len());
    // batches of at most `batch` tokens (rows · longest row), so long passages do not run the card out of memory
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for i in order {
        if !cur.is_empty() && (cur.len() + 1) * seqs[i].len() > batch.max(1) {
            chunks.push(std::mem::take(&mut cur));
        }
        cur.push(i);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    for chunk in &chunks {
        let chunk = chunk.as_slice();
        let t = chunk.iter().map(|&i| seqs[i].len()).max().unwrap_or(2).saturating_sub(1).max(1);
        let b = chunk.len();
        let (mut ids, mut tg, mut keep) = (Vec::with_capacity(b * t), Vec::with_capacity(b * t), Vec::with_capacity(b * t));
        for &i in chunk {
            let s = &seqs[i];
            for p in 0..t {
                let real = p + 1 < s.len();
                ids.push(if real { s[p] } else { PAD });
                tg.push(if real { s[p + 1] } else { 0 });
                keep.push(if real && p + 1 >= from[i] { 1f32 } else { 0f32 });
            }
        }
        let n_kept: Vec<usize> = chunk.iter().map(|&i| seqs[i].len() - from[i]).collect();
        let logits = model.forward(&Tensor::from_vec(ids, (b, t), dev)?)?; // [b·t, V]
        let lp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let picked = lp.gather(&Tensor::from_vec(tg, (b * t, 1), dev)?, 1)?.squeeze(1)?;
        let nll: Vec<f32> = picked.neg()?.mul(&Tensor::from_vec(keep, b * t, dev)?)?.reshape((b, t))?.sum(1)?.to_vec1()?;
        for (r, &i) in chunk.iter().enumerate() {
            out[i] = (nll[r] as f64, n_kept[r]);
        }
    }
    Ok(out)
}

/// Per question: (right, margin, pick, mean-per-token right).
struct Graded {
    right: bool,
    margin: f64,
    pick: usize,
    right_norm: bool,
}

fn grade(model: &Model, dev: &Device, bpe: &Bpe, qs: &[Mc], passage: bool, batch: usize) -> Result<Vec<Graded>, String> {
    let (mut seqs, mut from, mut owner) = (Vec::new(), Vec::new(), Vec::new());
    for (qi, q) in qs.iter().enumerate() {
        let mut p = vec![EOS];
        p.extend(bpe.encode(&prompt_text(if passage { &q.support } else { "" }, &q.question)));
        for c in &q.choices {
            let mut s = p.clone();
            s.extend(bpe.encode(&format!(" {}", c.trim())));
            s.push(EOS);
            from.push(p.len());
            seqs.push(s);
            owner.push(qi);
        }
    }
    let nll = scored_nll(model, dev, &seqs, &from, batch).map_err(|e| format!("score: {e}"))?;
    let mut out = Vec::with_capacity(qs.len());
    let mut k = 0;
    for q in qs {
        let sc: Vec<(f64, usize)> = nll[k..k + q.choices.len()].to_vec();
        k += q.choices.len();
        let tot: Vec<f64> = sc.iter().map(|x| x.0).collect();
        let norm: Vec<f64> = sc.iter().map(|x| x.0 / x.1.max(1) as f64).collect();
        let strict_best = |v: &[f64]| (0..v.len()).filter(|&j| j != q.right).all(|j| v[q.right] < v[j]);
        let mut idx: Vec<usize> = (0..tot.len()).collect();
        idx.sort_by(|&a, &b| tot[a].total_cmp(&tot[b]));
        let margin = if idx.len() > 1 { tot[idx[1]] - tot[idx[0]] } else { f64::INFINITY };
        out.push(Graded { right: strict_best(&tot), margin, pick: idx[0], right_norm: strict_best(&norm) });
    }
    debug_assert_eq!(owner.len(), seqs.len());
    Ok(out)
}

/// (right, wrong, abstained) when she answers only with margin >= tau.
fn with_tau(g: &[Graded], tau: f64) -> (usize, usize, usize) {
    let (mut r, mut w, mut a) = (0, 0, 0);
    for x in g {
        if x.margin < tau {
            a += 1;
        } else if x.right {
            r += 1;
        } else {
            w += 1;
        }
    }
    (r, w, a)
}

/// The tau with the most right − wrong (ties: the smaller tau); candidates are 0 and every margin.
fn best_tau(g: &[Graded]) -> f64 {
    let mut cands: Vec<f64> = g.iter().map(|x| x.margin).filter(|m| m.is_finite()).collect();
    cands.push(0.0);
    cands.push(f64::INFINITY); // abstain on every question
    cands.sort_by(f64::total_cmp);
    let mut best = (i64::MIN, 0.0);
    for &t in &cands {
        let (r, w, _) = with_tau(g, t);
        let s = r as i64 - w as i64;
        if s > best.0 {
            best = (s, t);
        }
    }
    best.1
}

fn report(name: &str, g: &[Graded], tau: Option<f64>) -> f64 {
    let n = g.len();
    let right = g.iter().filter(|x| x.right).count();
    let right_norm = g.iter().filter(|x| x.right_norm).count();
    let (tau, how) = match tau {
        Some(t) => (t, "given"),
        None => (best_tau(g), "best on this set"),
    };
    let (r, w, a) = with_tau(g, tau);
    println!(
        "{name}: accuracy {:.1}% ({right}/{n}); per-token-mean scoring {:.1}%; with \"I don't know\" (tau {tau:.4}, {how}): right {r}, wrong {w}, abstained {a}, right - wrong {}, share right among answers {:.1}%",
        100.0 * right as f64 / n.max(1) as f64,
        100.0 * right_norm as f64 / n.max(1) as f64,
        r as i64 - w as i64,
        100.0 * r as f64 / (r + w).max(1) as f64
    );
    tau
}

fn load(path: &PathBuf, kind: Kind, d: usize, layers: usize, dev: &Device) -> Result<Model, String> {
    let mut varmap = VarMap::new();
    let model = Model::new(Config::with_kind(kind, VOCAB, d, layers, 256), VarBuilder::from_varmap(&varmap, DType::F32, dev)).map_err(|e| format!("model: {e}"))?;
    varmap.load(path).map_err(|e| format!("load {}: {e}", path.display()))?;
    model.set_track_activity(false);
    // the fused inference forward (AVA_FAST_INFER=0: the autograd forward, for checking)
    model.set_inference(std::env::var("AVA_FAST_INFER").map_or(true, |v| v != "0"));
    Ok(model)
}

fn run() -> Result<(), String> {
    let (mut model_path, mut base_path, mut out_path): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None);
    let (mut d, mut layers, mut batch, mut limit) = (0usize, 0usize, 32768usize, usize::MAX);
    let (mut kind, mut set, mut passage, mut cpu, mut tau) = (Kind::D, String::new(), false, false, None::<f64>);
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        if key == "--cpu" || key == "--passage" {
            if key == "--cpu" {
                cpu = true;
            } else {
                passage = true;
            }
            i += 1;
            continue;
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
        match key {
            "--model" => model_path = Some(PathBuf::from(v)),
            "--baseline" => base_path = Some(PathBuf::from(v)),
            "--arch" => kind = Kind::parse(v)?,
            "--d" => d = num(v)?,
            "--layers" => layers = num(v)?,
            "--set" => set = v.to_string(),
            "--tau" => tau = Some(v.parse::<f64>().map_err(|_| format!("bad number {v:?}"))?),
            "--batch-tokens" => batch = num(v)?,
            "--limit" => limit = num(v)?,
            "--out" => out_path = Some(PathBuf::from(v)),
            "--bpe" => bpe_path = PathBuf::from(v),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let model_path = model_path.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    if d == 0 || layers == 0 || set.is_empty() {
        return Err(format!("--d, --layers and --set are required\n{USAGE}"));
    }
    let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let mut qs = eval_set(&set)?;
    qs.truncate(limit);
    let with_passage = qs.iter().filter(|q| !q.support.trim().is_empty()).count();
    println!("set {set}: {} questions ({} with a passage), {} choices each on average; passage {}", qs.len(), with_passage, qs.iter().map(|q| q.choices.len()).sum::<usize>() as f64 / qs.len().max(1) as f64, if passage { "shown" } else { "not shown (closed book)" });

    let t0 = std::time::Instant::now();
    let model = load(&model_path, kind, d, layers, &dev)?;
    let g = grade(&model, &dev, &bpe, &qs, passage, batch)?;
    drop(model);
    report("model", &g, tau);
    let gb = match &base_path {
        Some(p) => {
            let base = load(p, kind, d, layers, &dev)?;
            let gb = grade(&base, &dev, &bpe, &qs, passage, batch)?;
            report("baseline", &gb, tau);
            // paired bootstrap of the accuracy difference
            let n = qs.len();
            let diff: Vec<f64> = (0..n).map(|k| g[k].right as i32 as f64 - gb[k].right as i32 as f64).collect();
            let mean = diff.iter().sum::<f64>() / n as f64;
            let mut rng = SplitMix64::new(1);
            let mut bs: Vec<f64> = (0..20_000)
                .map(|_| {
                    let mut s = 0.0;
                    for _ in 0..n {
                        s += diff[rng.below(n as u64) as usize];
                    }
                    s / n as f64
                })
                .collect();
            bs.sort_by(f64::total_cmp);
            println!(
                "model - baseline accuracy: {:+.2} points, paired bootstrap 95% interval [{:+.2}, {:+.2}] (20,000 resamples, seed 1)",
                100.0 * mean,
                100.0 * bs[500],
                100.0 * bs[19_499]
            );
            Some(gb)
        }
        None => None,
    };
    println!("scored in {:.1}s", t0.elapsed().as_secs_f64());
    if let Some(p) = out_path {
        let mut f = std::fs::File::create(&p).map_err(|e| format!("create {}: {e}", p.display()))?;
        writeln!(f, "model\tindex\tright_choice\tpick\tright\tmargin").map_err(|e| e.to_string())?;
        for (name, gs) in [("model", Some(&g)), ("baseline", gb.as_ref())] {
            let Some(gs) = gs else { continue };
            for (k, x) in gs.iter().enumerate() {
                writeln!(f, "{name}\t{k}\t{}\t{}\t{}\t{:.6}", qs[k].right, x.pick, x.right as u8, x.margin).map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(())
}
