// TESTING / DESIGN ONLY — num_eval: the v39 number test for a train_gpu checkpoint (model.safetensors).
//
// num_eval --model FILE --arch D --d 384 --layers 8 --items FILE [--glance] [--baseline FILE [--baseline-glance]]
//          [--out FILE] [--batch 512] [--cpu] [--bpe FILE] [--min-kind-items N]
//
// Items come from ava_data's num_items (one TSV line each: id, kind, bin, right, choice0, choice1[, choice2, choice3];
// every choice is a whole sentence). Each choice is scored as [EOS] sentence [EOS] (Bpe::encode_doc appends the
// closing EOS; the first EOS is context) by its total NLL over every predicted token, batched (eval::seq_nll).
// With --glance the sentence gets its length marks first (ava_data::numbers::insert_marks), and their NLL counts
// too: knowing how long the answer is belongs to knowing the answer. Her pick is the lowest NLL; she is right only
// if the right choice is strictly lower than every other (a tie counts as wrong).
//
// Prints accuracy overall (the kinds weighted equally, as pre-registered; the item-weighted share is shown too),
// per kind (with chance) and per kind × bin, the seconds and
// sentences per second. With --baseline it scores a second model on the same items and prints the paired
// bootstrap 95% interval of the accuracy difference (model − baseline; 20,000 resamples, SplitMix64 seed 1; overall
// it resamples inside each kind, eval::stratified_paired_bootstrap),
// overall and per kind. The last line, "RESULT overall=... n=...", is for scripts. --out writes one line per item:
// id, kind, bin, right, pick, correct, margin (second-lowest − lowest NLL) [, the baseline's pick, correct, margin].
//
// v40 --lesion-mag / --lesion-engram: the magnitude code / the engram read add 0 (Model::set_lesion).
// --lesion-place (with --glance): the place embedding is set to 0 after loading, as a lesion study switches off
// one part of a brain: how much of her number sense rests on the place code. The length marks stay.
//
// v41 --min-kind-items N: the overall accuracy and its interval use only the kinds with at least N items (a kind
// with 2 items would otherwise weigh as much as one with 400); every kind is still scored and printed. Default 0.

use ava_data::bpe::{Bpe, EOS};
use ava_data::numbers::insert_marks;
use ava_gpu::eval::{paired_bootstrap, seq_nll, stratified_paired_bootstrap};
use ava_gpu::model::{Kind, Model, PLACE_EMB};
use candle_core::Device;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

const USAGE: &str = "usage: num_eval --model FILE --arch D --d 384 --layers 8 --items FILE [--glance [--lesion-place] [--lesion-mag] [--lesion-engram]]\n\
    \x20               [--baseline FILE [--baseline-glance]] [--out FILE] [--batch 512] [--cpu] [--bpe FILE] [--min-kind-items N]";

struct Item {
    id: String,
    kind: String,
    bin: String,
    right: usize,
    choices: Vec<String>,
}

/// One model's verdict on one item.
#[derive(Clone, Copy)]
struct Verdict {
    pick: usize,
    correct: bool,
    margin: f64,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("num_eval: {e}");
        std::process::exit(1);
    }
}

fn read_items(path: &PathBuf) -> Result<Vec<Item>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut items = Vec::new();
    for (ln, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 6 {
            return Err(format!("{}:{}: expected id, kind, bin, right and at least 2 choices", path.display(), ln + 1));
        }
        let right: usize = f[3].parse().map_err(|_| format!("{}:{}: bad right index {:?}", path.display(), ln + 1, f[3]))?;
        let choices: Vec<String> = f[4..].iter().map(|s| s.to_string()).collect();
        if right >= choices.len() {
            return Err(format!("{}:{}: right index {right} out of range", path.display(), ln + 1));
        }
        items.push(Item { id: f[0].into(), kind: f[1].into(), bin: f[2].into(), right, choices });
    }
    Ok(items)
}

fn load(path: &PathBuf, kind: Kind, d: usize, layers: usize, glance: bool, lesion: bool, dev: &Device) -> Result<Model, String> {
    let (_, varmap, model) = ava_gpu::ckpt::load_expect(path, kind, 256, d, layers, glance, dev)?;
    if lesion {
        let data = varmap.data().lock().unwrap();
        let pv = data.get(PLACE_EMB).ok_or("--lesion-place: the model has no place embedding")?;
        pv.set(&pv.as_tensor().zeros_like().map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    }
    model.set_track_activity(false);
    model.set_inference(std::env::var("AVA_FAST_INFER").map_or(true, |v| v != "0"));
    Ok(model)
}

/// Score every choice of every item; returns one verdict per item and the seconds taken.
fn grade(model: &Model, dev: &Device, bpe: &Bpe, items: &[Item], glance: bool, batch: usize) -> Result<(Vec<Verdict>, f64, usize), String> {
    let mut seqs = Vec::new();
    for it in items {
        for c in &it.choices {
            let mut t = vec![EOS];
            t.extend(bpe.encode_doc(c));
            seqs.push(if glance { insert_marks(&t) } else { t });
        }
    }
    let t0 = std::time::Instant::now();
    let nll = seq_nll(model, dev, &seqs, batch).map_err(|e| format!("score: {e}"))?;
    let secs = t0.elapsed().as_secs_f64();
    let mut out = Vec::with_capacity(items.len());
    let mut k = 0;
    for it in items {
        let s = &nll[k..k + it.choices.len()];
        k += it.choices.len();
        let mut order: Vec<usize> = (0..s.len()).collect();
        order.sort_by(|&a, &b| s[a].total_cmp(&s[b]));
        let (best, second) = (order[0], order[1]);
        let correct = (0..s.len()).all(|j| j == it.right || s[it.right] < s[j]);
        out.push(Verdict { pick: best, correct, margin: s[second] - s[best] });
    }
    Ok((out, secs, seqs.len()))
}

fn pct(r: usize, n: usize) -> f64 {
    100.0 * r as f64 / n.max(1) as f64
}

fn report(name: &str, items: &[Item], v: &[Verdict], secs: f64, sentences: usize, min_kind: usize) -> f64 {
    let mut by_kind: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new(); // right, n, choices
    let mut by_bin: BTreeMap<(&str, &str), (usize, usize)> = BTreeMap::new();
    for (it, v) in items.iter().zip(v) {
        let e = by_kind.entry(&it.kind).or_insert((0, 0, it.choices.len()));
        e.0 += v.correct as usize;
        e.1 += 1;
        let b = by_bin.entry((&it.kind, &it.bin)).or_default();
        b.0 += v.correct as usize;
        b.1 += 1;
    }
    let right = v.iter().filter(|v| v.correct).count();
    // overall: the kinds weighted equally (pre-registered); the item-weighted share is printed beside it
    let main: Vec<&(usize, usize, usize)> = by_kind.values().filter(|(_, n, _)| *n >= min_kind).collect();
    let overall = main.iter().map(|(r, n, _)| *r as f64 / (*n).max(1) as f64).sum::<f64>() / main.len().max(1) as f64;
    println!(
        "== {name}: {:.1}% overall (kinds weighted equally; item-weighted {:.1}% = {right}/{}), {sentences} sentences in {secs:.1}s ({:.0}/s)",
        100.0 * overall,
        pct(right, v.len()),
        v.len(),
        sentences as f64 / secs.max(1e-9)
    );
    for (k, (r, n, c)) in &by_kind {
        let bins: Vec<String> = by_bin.iter().filter(|((kk, _), _)| kk == k).map(|((_, b), (r, n))| format!("{b} {:.1}% ({r}/{n})", pct(*r, *n))).collect();
        let note = if *n < min_kind { "   [reported only: fewer than --min-kind-items]" } else { "" };
        println!("  {k:<14} {:>5.1}% ({r}/{n}; chance {:.0}%)   {}{note}", pct(*r, *n), 100.0 / *c as f64, bins.join(", "));
    }
    overall
}

fn run() -> Result<(), String> {
    let (mut model_path, mut base_path, mut out_path, mut items_path): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None, None);
    let (mut d, mut layers, mut batch, mut min_kind) = (0usize, 0usize, 512usize, 0usize);
    let (mut kind, mut cpu, mut glance, mut base_glance, mut lesion) = (Kind::D, false, false, false, false);
    let (mut lesion_mag, mut lesion_engram) = (false, false);
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        if matches!(key, "--cpu" | "--glance" | "--baseline-glance" | "--lesion-place" | "--lesion-mag" | "--lesion-engram") {
            match key {
                "--cpu" => cpu = true,
                "--glance" => glance = true,
                "--lesion-place" => lesion = true,
                "--lesion-mag" => lesion_mag = true,
                "--lesion-engram" => lesion_engram = true,
                _ => base_glance = true,
            }
            i += 1;
            continue;
        }
        let val = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
        match key {
            "--model" => model_path = Some(PathBuf::from(val)),
            "--baseline" => base_path = Some(PathBuf::from(val)),
            "--items" => items_path = Some(PathBuf::from(val)),
            "--out" => out_path = Some(PathBuf::from(val)),
            "--arch" => kind = Kind::parse(val)?,
            "--d" => d = num(val)?,
            "--layers" => layers = num(val)?,
            "--batch" => batch = num(val)?.max(1),
            "--bpe" => bpe_path = PathBuf::from(val),
            "--min-kind-items" => min_kind = num(val)?,
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let model_path = model_path.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    let items_path = items_path.ok_or_else(|| format!("--items is required\n{USAGE}"))?;
    if d == 0 || layers == 0 {
        return Err("--d and --layers are required".into());
    }
    let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let items = read_items(&items_path)?;
    println!("items {}: {} questions", items_path.display(), items.len());
    if min_kind > 0 && !items.iter().any(|it| items.iter().filter(|x| x.kind == it.kind).count() >= min_kind) {
        return Err(format!("--min-kind-items {min_kind}: no kind has that many items"));
    }

    if lesion && !glance {
        return Err("--lesion-place needs --glance".into());
    }
    let model = load(&model_path, kind, d, layers, glance, lesion, &dev)?;
    // v40: switch off the magnitude code and/or the engram (each then adds 0)
    model.set_lesion(false, lesion_mag, lesion_engram);
    let (v, secs, ns) = grade(&model, &dev, &bpe, &items, glance, batch)?;
    drop(model);
    let acc = report(&format!("model{}{}{}{}", if glance { " (glance)" } else { "" }, if lesion { " with the place code off" } else { "" }, if lesion_mag { " with the magnitude code off" } else { "" }, if lesion_engram { " with the engram off" } else { "" }), &items, &v, secs, ns, min_kind);

    let base = match &base_path {
        Some(p) => {
            let bm = load(p, kind, d, layers, base_glance, false, &dev)?;
            let (bv, bsecs, bns) = grade(&bm, &dev, &bpe, &items, base_glance, batch)?;
            drop(bm);
            report(&format!("baseline{}", if base_glance { " (glance)" } else { "" }), &items, &bv, bsecs, bns, min_kind);
            let a: Vec<bool> = v.iter().map(|x| x.correct).collect();
            let b: Vec<bool> = bv.iter().map(|x| x.correct).collect();
            let mut kinds: Vec<&str> = items.iter().map(|it| it.kind.as_str()).collect();
            kinds.sort();
            kinds.dedup();
            let groups: Vec<(Vec<bool>, Vec<bool>)> = kinds
                .iter()
                .filter(|k| items.iter().filter(|it| it.kind == **k).count() >= min_kind)
                .map(|k| {
                    let idx: Vec<usize> = (0..items.len()).filter(|&j| items[j].kind == *k).collect();
                    (idx.iter().map(|&j| a[j]).collect(), idx.iter().map(|&j| b[j]).collect())
                })
                .collect();
            let (m, lo, hi) = stratified_paired_bootstrap(&groups, 20_000, 1);
            println!(
                "model - baseline, overall (kinds weighted equally): {:+.2} points, paired bootstrap 95% interval [{:+.2}, {:+.2}] (20,000 resamples inside each kind, seed 1)",
                100.0 * m,
                100.0 * lo,
                100.0 * hi
            );
            let (m, lo, hi) = paired_bootstrap(&a, &b, 20_000, 1);
            println!("model - baseline, item-weighted: {:+.2} points [{:+.2}, {:+.2}]", 100.0 * m, 100.0 * lo, 100.0 * hi);
            for k in kinds {
                let idx: Vec<usize> = (0..items.len()).filter(|&j| items[j].kind == k).collect();
                let ak: Vec<bool> = idx.iter().map(|&j| a[j]).collect();
                let bk: Vec<bool> = idx.iter().map(|&j| b[j]).collect();
                let (m, lo, hi) = paired_bootstrap(&ak, &bk, 20_000, 1);
                println!("  {k:<14} {:+.2} points [{:+.2}, {:+.2}]", 100.0 * m, 100.0 * lo, 100.0 * hi);
            }
            Some(bv)
        }
        None => None,
    };

    if let Some(p) = &out_path {
        let mut f = std::io::BufWriter::new(std::fs::File::create(p).map_err(|e| format!("{}: {e}", p.display()))?);
        for (j, it) in items.iter().enumerate() {
            write!(f, "{}\t{}\t{}\t{}\t{}\t{}\t{:.4}", it.id, it.kind, it.bin, it.right, v[j].pick, v[j].correct as u8, v[j].margin).map_err(|e| e.to_string())?;
            if let Some(bv) = &base {
                write!(f, "\t{}\t{}\t{:.4}", bv[j].pick, bv[j].correct as u8, bv[j].margin).map_err(|e| e.to_string())?;
            }
            writeln!(f).map_err(|e| e.to_string())?;
        }
    }
    println!("RESULT overall={acc:.6} n={}", items.len());
    Ok(())
}
