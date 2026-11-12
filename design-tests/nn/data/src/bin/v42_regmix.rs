// TESTING / DESIGN ONLY
//! v42_regmix: pick v42's data mixture by regression over small proxy runs (see the `regmix` module).
//! CPU only. It writes the proxy plan and fits the results; the proxy runs themselves are GPU work that
//! is queued here and never run by this tool.
//!
//!   v42_regmix propose [--sources FILE] [--out DIR] [--n 96] [--seed 42] [--proxy-tokens 50M]
//!                      [--center MIX|sources|none] [--concentration 20] [--min-weight 0] [--force]
//!                      [--train-cmd TEMPLATE]
//!   v42_regmix fit     [--results FILE] [--sources FILE] [--train-tokens 1G] [--model logfeat]
//!                      [--search best|topK] [--lambda 1e-4] [--target NAME=W,...] [--max-weight NAME=W,...]
//!                      [--compare MIX|sources|none] [--samples 20000] [--seed 7]
//!   v42_regmix selftest [--out DIR]
//!
//! PROPOSE reads the full build's sources.tsv (default data/nn/shards-v42-full/sources.tsv): the source
//! names and how many train tokens each can give (all of it when the source was read to its end, else the
//! estimate for the whole file). A source's upper bound is that divided by --proxy-tokens (at most 1), so
//! every proxy mixture can really be built. Runs alternate between broad draws (Dirichlet, alpha 1 for every
//! source) and draws near the center mixture (alpha = concentration * center share; the center is the
//! full build's own weights by default). It writes into --out (default data/nn/v42/regmix):
//!   regmix_configs.tsv   run, kind, w_NAME (share), tok_NAME (train tokens) per run
//!   regmix_build.txt     one v42_shards line per run (--no-val: proxies are scored on the full build's
//!                        val slices, and val documents are never trained on, so no proxy sees them)
//!   regmix_queue.txt     the GPU jobs, QUEUED, NOT RUN: train each proxy (--train-cmd, with {run}, {train},
//!                        {val}, {tokens} filled in), then score it on every val_by_source/NAME slice
//!   regmix_results.tsv   run, w_NAME, empty loss_NAME columns to fill in (never overwritten once there)
//! A finished plan (regmix_configs.tsv exists) is skipped unless --force.
//!
//! FIT reads the results (rows with an empty loss are runs not done yet and are skipped), fits one
//! regression per source, prints its leave-one-out RMSE and R^2, and searches for the mixture with the
//! lowest weighted loss inside the bounds of the big run (what each source holds / --train-tokens,
//! lowered by --max-weight). --target sets the weight of each source's loss (default all equal; a source
//! not named gets 0). It prints the best mixture as a --mix string for v42_shards, its predicted losses
//! next to the --compare mixture (default the full build's own weights), and the best proxy run seen.
//!
//! SELFTEST runs the whole path on a made-up world with a known best mixture (5 sources): propose, write
//! a results file, read it back, fit, search, and compare with the true best on a grid. It fails when the
//! found mixture is more than 0.005 worse than the true best.

use ava_data::input::parse_count;
use ava_data::regmix::{dirichlet, fit_bounds, mix_string, quality, read_results, search, Kind, Model, Row, Search};
use ava_data::rng::SplitMix64;
use ava_data::v42_shards::parse_mix;
use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};

const DATA: &str = ava_data::DATA_DIR;
const DEFAULT_TRAIN_CMD: &str = "train_gpu --d 128 --layers 4 --ctx 256 --batch 32 --tokens {tokens} --lr 3e-3 --warmup 200 \
--min-lr-frac 0.1 --wd 0.1 --clip 1.0 --seed 1 --data-seed 1 --eval-tokens 500000 --decay-gates --train {train} --val {val} --out {run}/model";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn bad(m: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, m.into())
}

/// One row of a build's sources.tsv.
struct Src {
    name: String,
    /// Train tokens the source can give in all.
    avail: u64,
    /// Its weight in that build (0 when none).
    weight: f64,
}

fn read_sources(path: &Path) -> io::Result<Vec<Src>> {
    let text = std::fs::read_to_string(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let head: Vec<&str> = lines.next().ok_or_else(|| bad("empty sources.tsv"))?.split('\t').collect();
    let col = |n: &str| head.iter().position(|h| *h == n).ok_or_else(|| bad(format!("sources.tsv lacks column {n}")));
    let (cn, cw, ct, cx, ce) = (col("name")?, col("weight")?, col("train_tokens")?, col("exhausted")?, col("est_total_tokens")?);
    let mut out = Vec::new();
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        let num = |c: usize| f.get(c).and_then(|x| x.parse::<u64>().ok()).unwrap_or(0);
        let train = num(ct);
        let avail = if f.get(cx) == Some(&"true") { train } else { train.max(num(ce)) };
        out.push(Src { name: f[cn].to_string(), avail, weight: f.get(cw).and_then(|x| x.parse().ok()).unwrap_or(0.0) });
    }
    Ok(out)
}

/// NAME=W pairs as one value per name (names not given get `default`); unknown names are an error.
fn per_name(names: &[String], given: &[(String, f64)], default: f64) -> io::Result<Vec<f64>> {
    let mut v = vec![default; names.len()];
    for (n, w) in given {
        let k = names.iter().position(|x| x == n).ok_or_else(|| bad(format!("{n:?} is not a source ({})", names.join(", "))))?;
        v[k] = *w;
    }
    Ok(v)
}

/// A mixture named on the command line: "sources" (the build's weights), "none", or NAME=W,...
fn named_mix(arg: &str, names: &[String], srcs: Option<&[Src]>) -> io::Result<Option<Vec<f64>>> {
    let w = match arg {
        "none" => return Ok(None),
        "sources" => match srcs {
            Some(s) if s.iter().any(|x| x.weight > 0.0) => names.iter().map(|n| s.iter().find(|x| &x.name == n).map_or(0.0, |x| x.weight)).collect(),
            _ => return Ok(None),
        },
        m => per_name(names, &parse_mix(m).map_err(bad)?, 0.0)?,
    };
    let s: f64 = w.iter().sum();
    if s <= 0.0 {
        return Err(bad(format!("mixture {arg:?} has no weight")));
    }
    Ok(Some(w.iter().map(|x| x / s).collect()))
}

struct Args {
    cmd: String,
    kv: Vec<(String, String)>,
    flags: Vec<String>,
}

impl Args {
    fn parse() -> io::Result<Args> {
        let a: Vec<String> = std::env::args().skip(1).collect();
        let cmd = a.first().cloned().ok_or_else(|| bad("usage: v42_regmix propose|fit|selftest [options] (see the file's header)"))?;
        let (mut kv, mut flags, mut i) = (Vec::new(), Vec::new(), 1);
        while i < a.len() {
            if a[i] == "--force" {
                flags.push(a[i].clone());
            } else {
                let v = a.get(i + 1).ok_or_else(|| bad(format!("{} needs a value", a[i])))?;
                kv.push((a[i].clone(), v.clone()));
                i += 1;
            }
            i += 1;
        }
        Ok(Args { cmd, kv, flags })
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.kv.iter().rev().find(|(x, _)| x == k).map(|(_, v)| v.as_str())
    }

    fn check(&self, known: &[&str]) -> io::Result<()> {
        match self.kv.iter().find(|(k, _)| !known.contains(&k.as_str())) {
            Some((k, _)) => Err(bad(format!("unknown option {k} for {}", self.cmd))),
            None => Ok(()),
        }
    }

    fn count(&self, k: &str, d: u64) -> io::Result<u64> {
        self.get(k).map_or(Ok(d), |v| parse_count(v).map_err(bad))
    }

    fn float(&self, k: &str, d: f64) -> io::Result<f64> {
        self.get(k).map_or(Ok(d), |v| v.parse().map_err(|_| bad(format!("{k} wants a number, got {v:?}"))))
    }
}

fn run() -> io::Result<()> {
    let a = Args::parse()?;
    match a.cmd.as_str() {
        "propose" => propose(&a),
        "fit" => fit(&a),
        "selftest" => selftest(&a),
        c => Err(bad(format!("unknown command {c} (propose, fit, selftest)"))),
    }
}

/// Draw `n` mixtures: even runs broad (alpha 1), odd runs near `center` (when there is one).
fn draw(n: usize, center: Option<&[f64]>, conc: f64, lo: &[f64], hi: &[f64], seed: u64) -> io::Result<Vec<(&'static str, Vec<f64>)>> {
    let mut r = SplitMix64::new(seed);
    let k = lo.len();
    (0..n)
        .map(|i| {
            let (kind, alpha) = match center {
                Some(c) if i % 2 == 1 => ("center", c.iter().map(|x| conc * x).collect()),
                _ => ("broad", vec![1.0; k]),
            };
            let w = fit_bounds(&dirichlet(&mut r, &alpha), lo, hi).ok_or_else(|| bad("the bounds hold no mixture (the proxy is too big for the data?)"))?;
            Ok((kind, w))
        })
        .collect()
}

fn propose(a: &Args) -> io::Result<()> {
    a.check(&["--sources", "--out", "--n", "--seed", "--proxy-tokens", "--center", "--concentration", "--min-weight", "--train-cmd"])?;
    let sources_path = PathBuf::from(a.get("--sources").map_or(format!("{DATA}/nn/shards-v42-full/sources.tsv"), String::from));
    let full_dir = sources_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let out = PathBuf::from(a.get("--out").map_or(format!("{DATA}/nn/v42/regmix"), String::from));
    if out.join("regmix_configs.tsv").exists() && !a.flags.iter().any(|f| f == "--force") {
        println!("{} already holds a plan (regmix_configs.tsv); skipping (--force to redo)", out.display());
        return Ok(());
    }
    let n = a.count("--n", 96)? as usize;
    let seed = a.count("--seed", 42)?;
    let proxy = a.count("--proxy-tokens", 50_000_000)?;
    let conc = a.float("--concentration", 20.0)?;
    let min_w = a.float("--min-weight", 0.0)?;
    let train_cmd = a.get("--train-cmd").unwrap_or(DEFAULT_TRAIN_CMD);
    let srcs = read_sources(&sources_path)?;
    let names: Vec<String> = srcs.iter().map(|s| s.name.clone()).collect();
    let lo = vec![min_w; names.len()];
    let hi: Vec<f64> = srcs.iter().map(|s| (s.avail as f64 / proxy as f64).clamp(min_w, 1.0)).collect();
    let center = named_mix(a.get("--center").unwrap_or("sources"), &names, Some(&srcs))?;
    let t0 = std::time::Instant::now();
    let runs = draw(n, center.as_deref(), conc, &lo, &hi, seed)?;
    std::fs::create_dir_all(&out)?;

    let mut cfg = String::from("run\tkind");
    let mut res = String::from("run");
    for nm in &names {
        write!(cfg, "\tw_{nm}").unwrap();
        write!(res, "\tw_{nm}").unwrap();
    }
    for nm in &names {
        write!(cfg, "\ttok_{nm}").unwrap();
        write!(res, "\tloss_{nm}").unwrap();
    }
    cfg.push('\n');
    res.push('\n');
    let mut build = format!(
        "# v42_regmix proxy builds ({n} runs, {proxy} train tokens each, seed {seed}); run from design-tests/nn/data.\n\
         # --no-val: each proxy is scored on the full build's val slices ({}/val_by_source).\n",
        full_dir.display()
    );
    let mut queue = format!(
        "# v42_regmix GPU queue: QUEUED, NOT RUN. For each run, after its v42_shards line (regmix_build.txt):\n\
         # 1. train it (below); 2. score its model on every {}/val_by_source/NAME slice (BPB or loss per source)\n\
         # and write those into regmix_results.tsv as loss_NAME; 3. run `v42_regmix fit`.\n",
        full_dir.display()
    );
    for (i, (kind, w)) in runs.iter().enumerate() {
        let run_dir = format!("{DATA}/nn/regmix/run_{i:03}");
        write!(cfg, "{i}\t{kind}").unwrap();
        write!(res, "{i}").unwrap();
        for x in w {
            write!(cfg, "\t{x:.6}").unwrap();
            write!(res, "\t{x:.6}").unwrap();
        }
        for x in w {
            write!(cfg, "\t{}", (x * proxy as f64).round() as u64).unwrap();
        }
        for _ in w {
            res.push('\t');
        }
        cfg.push('\n');
        res.push('\n');
        writeln!(build, "cargo run --release -j 2 --bin v42_shards -- --full --mix {} --train-tokens {proxy} --no-val --out {run_dir}", mix_string(&names, w)).unwrap();
        let cmd = train_cmd
            .replace("{run}", &run_dir)
            .replace("{train}", &format!("{run_dir}/train"))
            .replace("{val}", &format!("{}/val", full_dir.display()))
            .replace("{tokens}", &proxy.to_string());
        writeln!(queue, "{cmd}").unwrap();
    }
    std::fs::write(out.join("regmix_configs.tsv"), cfg)?;
    std::fs::write(out.join("regmix_build.txt"), build)?;
    std::fs::write(out.join("regmix_queue.txt"), queue)?;
    let rp = out.join("regmix_results.tsv");
    let wrote_res = !rp.exists();
    if wrote_res {
        std::fs::write(&rp, res)?;
    }
    println!("{n} proxy mixtures over {} sources, {proxy} tokens each, in {:.3} s -> {}", names.len(), t0.elapsed().as_secs_f64(), out.display());
    println!("upper bounds (what each source holds / proxy tokens):");
    for (s, h) in srcs.iter().zip(&hi) {
        println!("  {:<15} holds {:>11} tok  max share {:.4}", s.name, s.avail, h);
    }
    let mean: Vec<f64> = (0..names.len()).map(|k| runs.iter().map(|(_, w)| w[k]).sum::<f64>() / n.max(1) as f64).collect();
    println!("mean proposal: {}", mix_string(&names, &mean));
    if !wrote_res {
        println!("{} exists and was kept", rp.display());
    }
    println!("proxy runs are QUEUED in regmix_queue.txt, not run (GPU work)");
    Ok(())
}

/// The weighted target of observed losses.
fn observed(loss: &[f64], t: &[f64]) -> f64 {
    let s: f64 = t.iter().sum();
    loss.iter().zip(t).map(|(l, x)| l * x / s).sum()
}

struct FitOut {
    best: Vec<f64>,
    pred: f64,
}

#[allow(clippy::too_many_arguments)]
fn fit_and_search(names: &[String], rows: &[Row], kind: Kind, how: Search, lambda: f64, target: &[f64], lo: &[f64], hi: &[f64], samples: usize, seed: u64, compare: Option<&[f64]>, say: bool) -> io::Result<FitOut> {
    let t0 = std::time::Instant::now();
    let model = Model::fit(rows, kind, lambda);
    let q = quality(rows, kind, lambda);
    let (best, pred) = search(&model, target, lo, hi, seed, samples, how).ok_or_else(|| bad("the bounds hold no mixture"))?;
    if say {
        println!("{} runs, model {}, lambda {lambda}, fit + search in {:.2} s", rows.len(), kind.name(), t0.elapsed().as_secs_f64());
        println!("fit quality per source (leave-one-out RMSE of the loss, in-sample R^2, target weight):");
        for ((n, (rmse, r2)), t) in names.iter().zip(&q).zip(target) {
            println!("  {n:<15} rmse {rmse:.4}  r2 {r2:.3}  target {t:.3}");
        }
        let (bi, bo) = rows.iter().enumerate().map(|(i, r)| (i, observed(&r.loss, target))).min_by(|x, y| x.1.total_cmp(&y.1)).unwrap();
        println!("best proxy run seen: row {bi}, target {bo:.4} (predicted {:.4})", model.target(&rows[bi].mix, target));
        println!("best mixture ({how:?}): predicted target {pred:.4}");
        println!("  --mix {}", mix_string(names, &best));
        let pb = model.predict(&best);
        let pc = compare.map(|c| model.predict(c));
        println!("  {:<15} {:>8} {:>8}   {:>9} {:>9}", "source", "share", "max", "pred loss", "compare");
        for (k, n) in names.iter().enumerate() {
            let c = pc.as_ref().map_or(String::from("-"), |p| format!("{:.4}", p[k]));
            println!("  {n:<15} {:>8.4} {:>8.4}   {:>9.4} {:>9}  {}", best[k], hi[k], pb[k], c, compare.map_or(String::new(), |c| format!("(compare share {:.4})", c[k])));
        }
        if let Some(c) = compare {
            println!("compare mixture: predicted target {:.4}", model.target(c, target));
        }
    }
    Ok(FitOut { best, pred })
}

fn fit(a: &Args) -> io::Result<()> {
    a.check(&["--results", "--sources", "--train-tokens", "--model", "--search", "--lambda", "--target", "--max-weight", "--compare", "--samples", "--seed"])?;
    let rp = PathBuf::from(a.get("--results").map_or(format!("{DATA}/nn/v42/regmix/regmix_results.tsv"), String::from));
    let text = std::fs::read_to_string(&rp).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", rp.display())))?;
    let names: Vec<String> = text
        .lines()
        .find(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .unwrap_or("")
        .split('\t')
        .filter_map(|h| h.trim().strip_prefix("w_").map(String::from))
        .collect();
    if names.is_empty() {
        return Err(bad(format!("{}: no w_NAME columns", rp.display())));
    }
    let (rows, skipped) = read_results(&text, &names).map_err(bad)?;
    println!("{}: {} runs with results, {skipped} without (skipped)", rp.display(), rows.len());
    if rows.len() < 3 {
        return Err(bad("need at least 3 finished proxy runs to fit"));
    }
    let sp = PathBuf::from(a.get("--sources").map_or(format!("{DATA}/nn/shards-v42-full/sources.tsv"), String::from));
    let srcs = if sp.exists() { Some(read_sources(&sp)?) } else { None };
    let tokens = a.count("--train-tokens", 1_000_000_000)?;
    let mut hi: Vec<f64> = names
        .iter()
        .map(|n| srcs.as_ref().and_then(|s| s.iter().find(|x| &x.name == n)).map_or(1.0, |x| (x.avail as f64 / tokens as f64).min(1.0)))
        .collect();
    if let Some(m) = a.get("--max-weight") {
        for (h, m) in hi.iter_mut().zip(per_name(&names, &parse_mix(m).map_err(bad)?, f64::INFINITY)?) {
            *h = h.min(m);
        }
    }
    let lo = vec![0.0; names.len()];
    let target = match a.get("--target") {
        Some(t) => per_name(&names, &parse_mix(t).map_err(bad)?, 0.0)?,
        None => vec![1.0; names.len()],
    };
    let kind = Kind::parse(a.get("--model").unwrap_or("logfeat")).map_err(bad)?;
    let how = Search::parse(a.get("--search").unwrap_or("best")).map_err(bad)?;
    let compare = named_mix(a.get("--compare").unwrap_or("sources"), &names, srcs.as_deref())?;
    fit_and_search(
        &names,
        &rows,
        kind,
        how,
        a.float("--lambda", 1e-4)?,
        &target,
        &lo,
        &hi,
        a.count("--samples", 20_000)? as usize,
        a.count("--seed", 7)?,
        compare.as_deref(),
        true,
    )?;
    Ok(())
}

/// A made-up world: loss on source t falls with its own share towards a floor, plus a small pull from the
/// next source.
fn world(w: &[f64]) -> Vec<f64> {
    let b = [2.0, 2.5, 3.0, 1.5, 2.2];
    let k = [0.4, 0.6, 0.3, 0.8, 0.5];
    let p = [0.3, 0.5, 0.2, 0.4, 0.35];
    (0..5).map(|t| b[t] + k[t] * (w[t] + 0.02).powf(-p[t]) + 0.05 * w[(t + 1) % 5]).collect()
}

fn selftest(a: &Args) -> io::Result<()> {
    a.check(&["--out"])?;
    let out = PathBuf::from(a.get("--out").map_or(std::env::temp_dir().join("v42_regmix_selftest").display().to_string(), String::from));
    std::fs::create_dir_all(&out)?;
    let names: Vec<String> = ["web", "wiki", "speech", "books", "facts"].iter().map(|s| s.to_string()).collect();
    let (lo, hi) = (vec![0.0; 5], vec![1.0, 1.0, 1.0, 0.5, 0.05]);
    let target = vec![1.0, 1.0, 2.0, 1.0, 0.5];
    let center = [0.5, 0.2, 0.15, 0.1, 0.05];
    let runs = draw(64, Some(&center), 20.0, &lo, &hi, 42)?;
    // write and read back a results file, with noise and two runs not done yet
    let mut r = SplitMix64::new(5);
    let mut text = format!("run\t{}\t{}\n", names.iter().map(|n| format!("w_{n}")).collect::<Vec<_>>().join("\t"), names.iter().map(|n| format!("loss_{n}")).collect::<Vec<_>>().join("\t"));
    for (i, (_, w)) in runs.iter().enumerate() {
        let ws: Vec<String> = w.iter().map(|x| format!("{x:.6}")).collect();
        let ls: Vec<String> = if i >= 62 { vec![String::new(); 5] } else { world(w).iter().map(|l| format!("{:.5}", l + 0.005 * ava_data::regmix::normal(&mut r))).collect() };
        writeln!(text, "{i}\t{}\t{}", ws.join("\t"), ls.join("\t")).unwrap();
    }
    let rp = out.join("selftest_results.tsv");
    std::fs::write(&rp, &text)?;
    let (rows, skipped) = read_results(&std::fs::read_to_string(&rp)?, &names).map_err(bad)?;
    assert_eq!((rows.len(), skipped), (62, 2));
    // the true best on a 0.01 grid
    let tv = |w: &[f64]| observed(&world(w), &target);
    let (mut truth, mut tbest) = (vec![], f64::INFINITY);
    let g = 100usize;
    for i in 0..=g {
        for j in 0..=(g - i) {
            for k in 0..=(g - i - j) {
                let rest = g - i - j - k;
                for l in rest.saturating_sub(5)..=rest.min(50) {
                    let m = g - i - j - k - l;
                    if m > 5 {
                        continue;
                    }
                    let w = [i, j, k, l, m].map(|x| x as f64 / g as f64);
                    let v = tv(&w);
                    if v < tbest {
                        (truth, tbest) = (w.to_vec(), v);
                    }
                }
            }
        }
    }
    let f = fit_and_search(&names, &rows, Kind::LogFeat, Search::Best, 1e-4, &target, &lo, &hi, 20_000, 7, Some(&center), true)?;
    let got = tv(&f.best);
    let best_proxy = rows.iter().map(|r| tv(&r.mix)).fold(f64::INFINITY, f64::min);
    println!("true best {} (target {tbest:.4})", mix_string(&names, &truth));
    println!("found     {} (true target {got:.4}, predicted {:.4}); gap {:.4}, best proxy gap {:.4}", mix_string(&names, &f.best), f.pred, got - tbest, best_proxy - tbest);
    if got - tbest > 0.005 {
        return Err(bad(format!("selftest FAILED: gap {:.4} > 0.005", got - tbest)));
    }
    println!("selftest passed ({})", rp.display());
    Ok(())
}
