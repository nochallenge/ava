// TESTING / DESIGN ONLY — mc_eval: v40's multiple-choice tests (find-the-fact, sleep questions, OpenBookQA, ...).
//
// mc_eval --model FILE (--items TSV | --set NAME [--passage]) [--arch D] [--lesion place,mag,engram]
//         [--context FILE] [--baseline FILE [--baseline-lesion LIST] [--baseline-context FILE]]
//         [--norm total|byte] [--tau X] [--batch-tokens 32768] [--limit N] [--out FILE] [--resamples 20000] [--bpe FILE]
//
// * The model's shape is read from its file (ckpt::load_auto): v39 and v40 files both load, with their glance,
//   magnitude code and engram; glance models get their length marks (mc::item_seqs).
// * Items: a TSV of the data crate (`id kind bin right prompt c0 c1 ...`, escaped) or a question set by name
//   (qa::eval_set; closed book unless --passage). Scoring is v37's: total NLL of the choice's tokens and the
//   closing EOS; right only when the right choice is strictly lowest (a tie is wrong). v41: --norm byte divides
//   each choice's NLL by its bytes (default total, v37's rule); margins and tau are then in nats per byte.
// * --context FILE: its text goes before every prompt (a blank line between), e.g. the day's facts for the
//   sleep test's "awake, facts in view" condition.
// * --lesion: switch off the place code, the magnitude code and/or the engram read (Model::set_lesion).
// * "I don't know": with --tau she answers only when margin >= tau; without it, the tau with the most
//   right − wrong on this set is printed too (labelled: a dev choice, never for a test).
// * --baseline FILE, repeatable (each may be the same file). --baseline-lesion and --baseline-context apply to the
//   --baseline just before them, so one pass scores every condition and a sealed set is opened once. The paired
//   bootstrap 95% interval of accuracy (model − baseline) over items (--resamples, SplitMix64 seed 1), overall
//   and per kind and per bin.
// * Last line: RESULT n=.. acc=.. [base=.. diff=.. lo=.. hi=..] for chain scripts.

use ava_data::bpe::Bpe;
use ava_gpu::eval::paired_bootstrap;
use ava_gpu::mc::{self, Graded, Item};
use ava_gpu::model::{Kind, Model};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: mc_eval --model FILE (--items TSV | --set NAME [--passage]) [--arch D] [--lesion place,mag,engram]\n\
    \x20              [--context FILE] [--baseline FILE [--baseline-lesion LIST] [--baseline-context FILE]]\n\
    \x20              [--norm total|byte] [--tau X] [--batch-tokens 32768] [--limit N] [--out FILE] [--resamples 20000] [--bpe FILE]";

fn main() {
    if let Err(e) = run() {
        eprintln!("mc_eval: {e}");
        std::process::exit(1);
    }
}

fn lesions(s: &str) -> Result<(bool, bool, bool), String> {
    let (mut p, mut m, mut e) = (false, false, false);
    for w in s.split(',').map(str::trim).filter(|w| !w.is_empty()) {
        match w {
            "place" => p = true,
            "mag" => m = true,
            "engram" => e = true,
            "none" => {}
            _ => return Err(format!("unknown lesion {w:?} (place, mag, engram, none)")),
        }
    }
    Ok((p, m, e))
}

fn with_context(items: &[Item], ctx: &Option<String>) -> Vec<Item> {
    match ctx {
        None => items.to_vec(),
        Some(c) => items.iter().map(|it| Item { prompt: format!("{}\n\n{}", c.trim_end(), it.prompt), ..it.clone() }).collect(),
    }
}

fn read_context(p: &Option<PathBuf>) -> Result<Option<String>, String> {
    p.as_ref().map(|p| std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))).transpose()
}

/// Load, switch off what `lesion` names, and grade every item.
fn grade_with(path: &PathBuf, kind: Kind, lesion: (bool, bool, bool), items: &[Item], bpe: &Bpe, batch: usize, norm: mc::Norm, dev: &candle_core::Device) -> Result<(Vec<Graded>, f64, String), String> {
    let (cfg, _varmap, model): (_, _, Model) = ava_gpu::ckpt::load_auto(path, kind, 256, dev)?;
    if (lesion.0 && !cfg.glance) || (lesion.1 && !cfg.mag) || (lesion.2 && cfg.engram.is_none()) {
        return Err(format!("{}: a lesion names a part the model does not have (glance {}, mag {}, engram {:?})", path.display(), cfg.glance, cfg.mag, cfg.engram));
    }
    model.set_track_activity(false);
    model.set_inference(std::env::var("AVA_FAST_INFER").map_or(true, |v| v != "0"));
    model.set_lesion(lesion.0, lesion.1, lesion.2);
    let t0 = Instant::now();
    let g = mc::grade(&model, dev, bpe, items, cfg.glance, batch, norm)?;
    let desc = format!("{} (glance {}, mag {}, engram {:?}; off: place {} mag {} engram {})", path.display(), cfg.glance, cfg.mag, cfg.engram, lesion.0, lesion.1, lesion.2);
    Ok((g, t0.elapsed().as_secs_f64(), desc))
}

fn report(name: &str, items: &[Item], g: &[Graded], tau: Option<f64>, secs: f64) {
    let n = g.len();
    let right = g.iter().filter(|x| x.right).count();
    let chance = items.iter().map(|it| 1.0 / it.choices.len() as f64).sum::<f64>() / n.max(1) as f64;
    println!("{name}: accuracy {:.1}% ({right}/{n}; chance {:.1}%) in {secs:.1}s", 100.0 * right as f64 / n.max(1) as f64, 100.0 * chance);
    let (t, how) = match tau {
        Some(t) => (t, "fixed"),
        None => (mc::best_tau(g), "best on this set: a dev choice, not for a test"),
    };
    let (r, w, a) = mc::with_tau(g, t);
    println!(
        "  with \"I don't know\" (tau {t}, {how}): right {r}, wrong {w}, abstained {a}; right - wrong {}; share right among answers {:.1}%",
        r as i64 - w as i64,
        100.0 * r as f64 / (r + w).max(1) as f64
    );
    for (label, key) in [("kind", 0usize), ("bin", 1)] {
        let mut by: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        for (it, x) in items.iter().zip(g) {
            let k = if key == 0 { it.kind.as_str() } else { it.bin.as_str() };
            let e = by.entry(k).or_default();
            e.0 += x.right as usize;
            e.1 += 1;
        }
        if by.len() > 1 || (by.len() == 1 && !by.keys().next().unwrap().is_empty()) {
            let parts: Vec<String> = by.iter().map(|(k, (r, n))| format!("{k} {:.1}% ({r}/{n})", 100.0 * *r as f64 / *n as f64)).collect();
            println!("  by {label}: {}", parts.join(", "));
        }
    }
}

fn run() -> Result<(), String> {
    let (mut model_path, mut baseline, mut items_path, mut set, mut out): (Option<PathBuf>, Vec<(PathBuf, (bool, bool, bool), Option<PathBuf>)>, Option<PathBuf>, Option<String>, Option<PathBuf>) = (None, Vec::new(), None, None, None);
    let mut context: Option<PathBuf> = None;
    let mut lesion = (false, false, false);
    let mut norm = mc::Norm::Total;
    let (mut kind, mut passage, mut tau, mut batch, mut limit, mut resamples) = (Kind::D, false, None::<f64>, 32768usize, None::<usize>, 20000usize);
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        if key == "--passage" {
            passage = true;
            i += 1;
            continue;
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
        match key {
            "--model" => model_path = Some(v.into()),
            "--baseline" => baseline.push((v.into(), (false, false, false), None)),
            "--items" => items_path = Some(v.into()),
            "--set" => set = Some(v.into()),
            "--arch" => kind = Kind::parse(v)?,
            "--lesion" => lesion = lesions(v)?,
            "--baseline-lesion" => baseline.last_mut().ok_or("--baseline-lesion goes after its --baseline")?.1 = lesions(v)?,
            "--context" => context = Some(v.into()),
            "--baseline-context" => baseline.last_mut().ok_or("--baseline-context goes after its --baseline")?.2 = Some(v.into()),
            "--norm" => norm = mc::Norm::parse(v)?,
            "--tau" => tau = Some(v.parse::<f64>().map_err(|_| format!("bad number {v:?}"))?),
            "--batch-tokens" => batch = num(v)?.max(1),
            "--limit" => limit = Some(num(v)?),
            "--out" => out = Some(v.into()),
            "--resamples" => resamples = num(v)?.max(1),
            "--bpe" => bpe_path = v.into(),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let model_path = model_path.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    if !model_path.is_file() {
        return Err(format!("model {} is not a file", model_path.display()));
    }
    let mut items = match (&items_path, &set) {
        (Some(p), None) => mc::read_items(p)?,
        (None, Some(s)) => mc::from_qa(s, &ava_gpu::qa::eval_set(s)?, passage),
        _ => return Err(format!("give exactly one of --items and --set\n{USAGE}")),
    };
    if let Some(l) = limit {
        items.truncate(l);
    }
    if items.is_empty() {
        return Err("no items".into());
    }
    let dev = ava_gpu::device(false).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let src = items_path.as_ref().map(|p| p.display().to_string()).or(set.clone()).unwrap_or_default();
    println!("{} items from {src}{}, scoring {}", items.len(), if context.is_some() { " (with --context)" } else { "" }, norm.name());

    let its = with_context(&items, &read_context(&context)?);
    for (bp, _, bc) in &baseline {
        // fail before any scoring if a baseline file or context is missing (a sealed set is read once)
        if !bp.is_file() {
            return Err(format!("baseline {} is not a file", bp.display()));
        }
        read_context(bc)?;
    }
    let (g, secs, desc) = grade_with(&model_path, kind, lesion, &its, &bpe, batch, norm, &dev)?;
    report(&format!("model {desc}"), &items, &g, tau, secs);
    let acc = g.iter().filter(|x| x.right).count() as f64 / g.len() as f64;
    let mut result = format!("RESULT n={} acc={acc:.6}", g.len());
    let mut gbs: Vec<Vec<Graded>> = Vec::new();
    for (j, (bp, base_lesion, base_context)) in baseline.iter().enumerate() {
        let tag = if j == 0 { String::new() } else { (j + 1).to_string() };
        let its_b = with_context(&items, &read_context(base_context)?);
        let (gb, secs_b, desc_b) = grade_with(bp, kind, *base_lesion, &its_b, &bpe, batch, norm, &dev)?;
        let cdesc = if base_context.is_some() { " (with --baseline-context)" } else { "" };
        report(&format!("baseline{tag} {desc_b}{cdesc}"), &items, &gb, tau, secs_b);
        let a: Vec<bool> = g.iter().map(|x| x.right).collect();
        let b: Vec<bool> = gb.iter().map(|x| x.right).collect();
        let (m, lo, hi) = paired_bootstrap(&a, &b, resamples, 1);
        println!("model - baseline{tag}: {:+.2} points [95%: {:+.2}, {:+.2}] ({resamples} paired resamples over items)", 100.0 * m, 100.0 * lo, 100.0 * hi);
        for (label, key) in [("kind", 0usize), ("bin", 1)] {
            let mut by: BTreeMap<String, (Vec<bool>, Vec<bool>)> = BTreeMap::new();
            for (k, it) in items.iter().enumerate() {
                let name = if key == 0 { it.kind.clone() } else { it.bin.clone() };
                let e = by.entry(name).or_default();
                e.0.push(a[k]);
                e.1.push(b[k]);
            }
            if by.len() > 1 {
                for (name, (x, y)) in &by {
                    let (m, lo, hi) = paired_bootstrap(x, y, resamples, 1);
                    println!("  {label} {name}: {:+.2} [{:+.2}, {:+.2}] (n {})", 100.0 * m, 100.0 * lo, 100.0 * hi, x.len());
                }
            }
        }
        let accb = b.iter().filter(|&&x| x).count() as f64 / b.len() as f64;
        result += &format!(" base{tag}={accb:.6} diff{tag}={m:.6} lo{tag}={lo:.6} hi{tag}={hi:.6}");
        if let Some(t) = tau {
            // right − wrong per item (+1, −1, 0 for "I don't know") at the fixed tau, model − baseline
            let score = |g: &[Graded]| -> Vec<f64> { g.iter().map(|x| if x.margin < t { 0.0 } else if x.right { 1.0 } else { -1.0 }).collect() };
            let (m, lo, hi) = paired_bootstrap_f(&score(&g), &score(&gb), resamples, 1);
            println!("right - wrong per question at tau {t}, model - baseline{tag}: {m:+.4} [95%: {lo:+.4}, {hi:+.4}]");
            result += &format!(" rw_diff{tag}={m:.6} rw_lo{tag}={lo:.6} rw_hi{tag}={hi:.6}");
        }
        gbs.push(gb);
    }
    if let Some(o) = &out {
        let mut f = std::fs::File::create(o).map_err(|e| format!("create {}: {e}", o.display()))?;
        let cols: String = (0..gbs.len()).map(|j| format!("\tbase{j}_pick\tbase{j}_correct\tbase{j}_margin")).collect();
        writeln!(f, "id\tkind\tbin\tright\tpick\tcorrect\tmargin{cols}").map_err(|e| e.to_string())?;
        for (k, it) in items.iter().enumerate() {
            write!(f, "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}", it.id, it.kind, it.bin, it.right, g[k].pick, g[k].right as u8, g[k].margin).map_err(|e| e.to_string())?;
            for gb in &gbs {
                write!(f, "\t{}\t{}\t{:.6}", gb[k].pick, gb[k].right as u8, gb[k].margin).map_err(|e| e.to_string())?;
            }
            writeln!(f).map_err(|e| e.to_string())?;
        }
    }
    println!("{result}");
    Ok(())
}

/// Paired bootstrap of mean(a) − mean(b) over items for real-valued per-item scores: (mean, lo, hi), 95%.
fn paired_bootstrap_f(a: &[f64], b: &[f64], resamples: usize, seed: u64) -> (f64, f64, f64) {
    let n = a.len();
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let mean = d.iter().sum::<f64>() / n.max(1) as f64;
    let mut rng = ava_data::rng::SplitMix64::new(seed);
    let mut ms: Vec<f64> = (0..resamples).map(|_| (0..n).map(|_| d[rng.below(n as u64) as usize]).sum::<f64>() / n.max(1) as f64).collect();
    ms.sort_by(f64::total_cmp);
    let q = |p: f64| ms[((p * (resamples - 1) as f64).round() as usize).min(resamples - 1)];
    (mean, q(0.025), q(0.975))
}
