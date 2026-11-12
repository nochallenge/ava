// TESTING / DESIGN ONLY — es_tune's library half: argument parsing, the tune/check split, mc fitness, the run loop.
//
// The binary (src/bin/es_tune.rs) only parses, sets the thread count and calls `run`, so the lib tests drive the
// whole tool through the same flags. See the binary's header for the flags and for H1.

use super::{h1_patterns, select, Es, EsOpts, GenReport};
use crate::mc::{self, Graded, Item, Norm};
use crate::model::{Kind, Model};
use ava_data::bpe::Bpe;
use ava_data::numbers::insert_marks;
use ava_data::rng::{derive_seed, mix64};
use candle_core::Device;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const USAGE: &str = "usage: es_tune --model FILE --items TSV[,TSV...] --params PATTERNS|h1 --out FILE\n\
    \x20              [--fitness rmw|acc|nll] [--pop 8] [--sigma 0.01] [--lr 0.01] [--rank 1] [--gens 10] [--seed 1]\n\
    \x20              [--tune-share 0.5] [--split-seed 1] [--tau X] [--norm total|byte|pmi] [--arch D] [--limit N]\n\
    \x20              [--batch-tokens 32768] [--cpu] [--threads 8] [--bpe release/ava/bpe.txt]";

/// The default tokenizer: the released Ava's.
pub const DEFAULT_BPE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../release/ava/bpe.txt");

/// The label of the tune/check split's hash.
pub const SPLIT_TAG: u64 = 0x5_7117;

/// What the search maximises over the tune items.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fitness {
    /// right − wrong per item: +1 right, −1 wrong, 0 "I don't know" (margin < tau)
    Rmw,
    /// the share right
    Acc,
    /// −(mean score of the right choice): its NLL under --norm
    Nll,
}

impl Fitness {
    pub fn parse(s: &str) -> Result<Fitness, String> {
        match s {
            "rmw" => Ok(Fitness::Rmw),
            "acc" => Ok(Fitness::Acc),
            "nll" => Ok(Fitness::Nll),
            _ => Err(format!("unknown fitness {s:?} (rmw, acc, nll)")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Fitness::Rmw => "rmw",
            Fitness::Acc => "acc",
            Fitness::Nll => "nll",
        }
    }
}

/// es_tune's settings.
#[derive(Clone, Debug)]
pub struct Args {
    pub model: PathBuf,
    pub items: Vec<PathBuf>,
    pub params: String,
    pub out: PathBuf,
    pub fitness: Fitness,
    pub es: EsOpts,
    pub gens: usize,
    pub tune_share: f64,
    pub split_seed: u64,
    pub tau: f64,
    pub norm: Norm,
    pub kind: Kind,
    pub limit: Option<usize>,
    pub batch: usize,
    pub cpu: bool,
    pub threads: usize,
    pub bpe: PathBuf,
}

/// Refuse any path that names a sealed test set: the words are checked first (nothing is touched then), and an existing
/// path's real location too.
pub fn refuse_sealed(p: &Path) -> Result<(), String> {
    let bad = |s: &str| s.to_lowercase().contains("sealed");
    if bad(&p.to_string_lossy()) {
        return Err(format!("refused: {} names a sealed test set", p.display()));
    }
    if let Ok(c) = std::fs::canonicalize(p) {
        if bad(&c.to_string_lossy()) {
            return Err(format!("refused: {} leads to a sealed test set", p.display()));
        }
    }
    Ok(())
}

fn default_threads() -> usize {
    std::env::var("AVA_THREADS").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(8)
}

/// Parse es_tune's flags (`args` without the program name). `-h` gives Err(USAGE).
pub fn parse(args: &[String]) -> Result<Args, String> {
    let (mut model, mut items, mut params, mut out) = (None::<PathBuf>, Vec::<PathBuf>::new(), None::<String>, None::<PathBuf>);
    let mut a = Args {
        model: PathBuf::new(),
        items: Vec::new(),
        params: String::new(),
        out: PathBuf::new(),
        fitness: Fitness::Rmw,
        es: EsOpts::default(),
        gens: 10,
        tune_share: 0.5,
        split_seed: 1,
        tau: 0.0,
        norm: Norm::Total,
        kind: Kind::D,
        limit: None,
        batch: 32768,
        cpu: false,
        threads: default_threads(),
        bpe: PathBuf::from(DEFAULT_BPE),
    };
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        if key == "--cpu" {
            a.cpu = true;
            i += 1;
            continue;
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let int = |v: &str| v.parse::<usize>().map_err(|_| format!("{key}: bad integer {v:?}"));
        let num = |v: &str| v.parse::<f64>().map_err(|_| format!("{key}: bad number {v:?}"));
        match key {
            "--model" => model = Some(v.into()),
            "--items" => items.extend(v.split(',').map(str::trim).filter(|s| !s.is_empty()).map(PathBuf::from)),
            "--params" => params = Some(v.into()),
            "--out" => out = Some(v.into()),
            "--fitness" => a.fitness = Fitness::parse(v)?,
            "--pop" => a.es.pop = int(v)?,
            "--sigma" => a.es.sigma = num(v)?,
            "--lr" => a.es.lr = num(v)?,
            "--rank" => a.es.rank = int(v)?,
            "--gens" => a.gens = int(v)?,
            "--seed" => a.es.seed = v.parse().map_err(|_| format!("--seed: bad integer {v:?}"))?,
            "--tune-share" => a.tune_share = num(v)?,
            "--split-seed" => a.split_seed = v.parse().map_err(|_| format!("--split-seed: bad integer {v:?}"))?,
            "--tau" => a.tau = num(v)?,
            "--norm" => a.norm = Norm::parse(v)?,
            "--arch" => a.kind = Kind::parse(v)?,
            "--limit" => a.limit = Some(int(v)?),
            "--batch-tokens" => a.batch = int(v)?.max(1),
            "--threads" => a.threads = int(v)?,
            "--bpe" => a.bpe = v.into(),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    a.model = model.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    a.out = out.ok_or_else(|| format!("--out is required\n{USAGE}"))?;
    a.params = params.ok_or_else(|| format!("--params is required (patterns, or h1 for the write-rule weights)\n{USAGE}"))?;
    if items.is_empty() {
        return Err(format!("--items is required\n{USAGE}"));
    }
    a.items = items;
    check(&a)?;
    Ok(a)
}

/// The settings' own checks (also run by `run`, for Args built in code).
fn check(a: &Args) -> Result<(), String> {
    for p in std::iter::once(&a.model).chain(&a.items).chain([&a.out, &a.bpe]) {
        refuse_sealed(p)?;
    }
    if a.es.pop == 0 || a.es.rank == 0 {
        return Err("--pop and --rank must be at least 1".into());
    }
    if !(a.es.sigma > 0.0 && a.es.sigma.is_finite()) || !(a.es.lr >= 0.0 && a.es.lr.is_finite()) {
        return Err(format!("--sigma must be positive and --lr at least 0 (got {} and {})", a.es.sigma, a.es.lr));
    }
    if !(a.tune_share > 0.0 && a.tune_share <= 1.0) {
        return Err(format!("--tune-share must be in (0, 1], not {}", a.tune_share));
    }
    if !a.tau.is_finite() || a.tau < 0.0 {
        return Err(format!("--tau must be at least 0, not {}", a.tau));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ the split

/// FNV-1a over the fields, each followed by 0xFF (a byte no UTF-8 text holds).
fn fnv(fields: &[&str]) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for f in fields {
        for &b in f.as_bytes().iter().chain(&[0xFF]) {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01B3);
        }
    }
    h
}

/// The tune and check items (indices): an item is in tune when the seeded hash of its id, kind and prompt falls below
/// `share`. It depends on the item only (not on its place or on the other items), so it is the same in every run.
pub fn split(items: &[Item], share: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let salt = derive_seed(seed, SPLIT_TAG, 0);
    let (mut tune, mut check) = (Vec::new(), Vec::new());
    for (i, it) in items.iter().enumerate() {
        let u = (mix64(fnv(&[&it.id, &it.kind, &it.prompt]) ^ salt) >> 11) as f64 / (1u64 << 53) as f64;
        if u < share {
            tune.push(i);
        } else {
            check.push(i);
        }
    }
    (tune, check)
}

// ------------------------------------------------------------------------------------------------ scoring

/// A set of items tokenized once, graded exactly as mc::grade grades them (the same sequences, the same batched
/// mc::scored_nll, the same norm), without tokenizing again for every evaluation.
pub struct Prepared {
    seqs: Vec<Vec<u32>>,
    from: Vec<usize>,
    nchoice: Vec<usize>,
    pub rights: Vec<usize>,
    norm: Norm,
    bytes: Vec<f64>,
    base: Option<(Vec<Vec<u32>>, Vec<usize>, Vec<usize>)>,
}

impl Prepared {
    pub fn new(bpe: &Bpe, items: &[Item], glance: bool, vocab: usize, norm: Norm) -> Prepared {
        let (mut seqs, mut from, mut nchoice) = (Vec::new(), Vec::new(), Vec::new());
        for it in items {
            let (s, f) = mc::item_seqs(bpe, it, glance);
            nchoice.push(s.len());
            seqs.extend(s);
            from.extend(f);
        }
        let (bytes, base) = if norm == Norm::Pmi {
            let pt = bpe.encode(mc::PMI_PROMPT);
            let pt = if glance { insert_marks(&pt) } else { pt };
            (Vec::new(), Some(mc::base_seqs(&pt, &seqs, &from)))
        } else {
            (mc::scored_bytes(&crate::eval::byte_lengths_n(bpe, vocab), &seqs, &from), None)
        };
        Prepared { seqs, from, nchoice, rights: items.iter().map(|it| it.right).collect(), norm, bytes, base }
    }

    pub fn len(&self) -> usize {
        self.rights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rights.is_empty()
    }

    /// Tokens through the model per grading (inputs, padding not counted).
    pub fn tokens(&self) -> usize {
        let n = |s: &[Vec<u32>]| s.iter().map(|x| x.len().saturating_sub(1)).sum::<usize>();
        n(&self.seqs) + self.base.as_ref().map_or(0, |(b, _, _)| n(b))
    }

    pub fn grade(&self, model: &Model, dev: &Device, batch: usize) -> Result<Vec<Graded>, String> {
        let nll = mc::scored_nll(model, dev, &self.seqs, &self.from, batch).map_err(|e| format!("score: {e}"))?;
        let score = match &self.base {
            Some((bs, bf, map)) => {
                let b = mc::scored_nll(model, dev, bs, bf, batch).map_err(|e| format!("score (pmi base): {e}"))?;
                mc::pmi_scores(&nll, &b, map)
            }
            None => mc::normed(&nll, &self.bytes, self.norm),
        };
        let mut out = Vec::with_capacity(self.nchoice.len());
        let mut k = 0;
        for (i, &n) in self.nchoice.iter().enumerate() {
            out.push(mc::grade_one(&score[k..k + n], self.rights[i]));
            k += n;
        }
        Ok(out)
    }

    /// The fitness of a grading of these items.
    pub fn fitness(&self, g: &[Graded], f: Fitness, tau: f64) -> f64 {
        fitness(g, &self.rights, f, tau)
    }
}

/// rmw: (right − wrong)/n with "I don't know" when margin < tau (mc::with_tau); acc: right/n; nll: −mean score of the
/// right choice.
pub fn fitness(g: &[Graded], rights: &[usize], f: Fitness, tau: f64) -> f64 {
    let n = g.len().max(1) as f64;
    match f {
        Fitness::Rmw => {
            let (r, w, _) = mc::with_tau(g, tau);
            (r as f64 - w as f64) / n
        }
        Fitness::Acc => g.iter().filter(|x| x.right).count() as f64 / n,
        Fitness::Nll => -g.iter().zip(rights).map(|(x, &r)| x.nll[r]).sum::<f64>() / n,
    }
}

// ------------------------------------------------------------------------------------------------ the run

/// One row of the log: generation 0 is the start (no members).
#[derive(Clone, Debug)]
pub struct Row {
    pub gen: usize,
    pub members: Option<GenReport>,
    pub tune: f64,
    pub check: f64,
    pub secs: f64,
    pub tok_per_s: f64,
}

/// What a run did.
#[derive(Clone, Debug)]
pub struct Summary {
    pub n_tune: usize,
    pub n_check: usize,
    pub tuned: Vec<String>,
    pub weights: usize,
    pub rows: Vec<Row>,
    pub log: PathBuf,
    pub secs: f64,
}

impl Summary {
    pub fn tune0(&self) -> f64 {
        self.rows[0].tune
    }
    pub fn check0(&self) -> f64 {
        self.rows[0].check
    }
    pub fn tune(&self) -> f64 {
        self.rows.last().unwrap().tune
    }
    pub fn check(&self) -> f64 {
        self.rows.last().unwrap().check
    }
}

/// The log TSV beside the output model: NAME.es_log.tsv.
pub fn log_path(out: &Path) -> PathBuf {
    out.with_extension("es_log.tsv")
}

fn fmt(x: f64) -> String {
    if x.is_nan() {
        "-".into()
    } else {
        format!("{x:.6}")
    }
}

fn write_log(path: &Path, a: &Args, rows: &[Row]) -> Result<(), String> {
    let mut s = format!(
        "# es_tune fitness={} pop={} sigma={} lr={} rank={} seed={} tune_share={} split_seed={} tau={} norm={} params={}\n",
        a.fitness.name(),
        a.es.pop,
        a.es.sigma,
        a.es.lr,
        a.es.rank,
        a.es.seed,
        a.tune_share,
        a.split_seed,
        a.tau,
        a.norm.name(),
        a.params
    );
    s += "gen\tmembers_mean\tmembers_best\tmembers_worst\ttune\tcheck\tstep_rel\tsecs\tevals_per_s\ttok_per_s\n";
    for r in rows {
        let (mean, best, worst, step, eps) = match &r.members {
            Some(m) => (m.mean(), m.best(), m.worst(), m.step / m.norm.max(1e-30), m.evals_per_sec()),
            None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        };
        s += &format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\n", r.gen, fmt(mean), fmt(best), fmt(worst), fmt(r.tune), fmt(r.check), fmt(step), r.secs, fmt(eps), fmt(r.tok_per_s));
    }
    std::fs::write(path, s).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Right, wrong and "I don't know" at tau, as text.
fn rwa(g: &[Graded], tau: f64) -> String {
    let (r, w, a) = mc::with_tau(g, tau);
    format!("right {r} wrong {w} idk {a} (right - wrong {:+}, acc {:.1}%)", r as i64 - w as i64, 100.0 * g.iter().filter(|x| x.right).count() as f64 / g.len().max(1) as f64)
}

/// Run es_tune: load, split, search, write the tuned model, its arch file and the log.
pub fn run(a: &Args) -> Result<Summary, String> {
    check(a)?;
    let t_run = Instant::now();
    if !a.model.is_file() {
        return Err(format!("model {} is not a file", a.model.display()));
    }
    if let (Ok(m), Ok(o)) = (std::fs::canonicalize(&a.model), std::fs::canonicalize(&a.out)) {
        if m == o {
            return Err(format!("--out {} is the input model; write the tuned model elsewhere", a.out.display()));
        }
    }
    let mut items: Vec<Item> = Vec::new();
    for p in &a.items {
        items.extend(mc::read_items(p)?);
    }
    if let Some(l) = a.limit {
        items.truncate(l);
    }
    if items.is_empty() {
        return Err("no items".into());
    }
    let (ti, ci) = split(&items, a.tune_share, a.split_seed);
    if ti.is_empty() {
        return Err(format!("the split put no item in tune (--tune-share {}, {} items)", a.tune_share, items.len()));
    }
    if ci.is_empty() && a.tune_share < 1.0 {
        return Err(format!("the split put no item in check ({} items); give more items or a smaller --tune-share", items.len()));
    }
    let tune_items: Vec<Item> = ti.iter().map(|&i| items[i].clone()).collect();
    let check_items: Vec<Item> = ci.iter().map(|&i| items[i].clone()).collect();

    let dev = crate::device(a.cpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&a.bpe).map_err(|e| format!("bpe {}: {e}", a.bpe.display()))?;
    let (cfg, varmap, model) = crate::ckpt::load_auto(&a.model, a.kind, 256, &dev)?;
    model.set_track_activity(false);
    model.set_inference(std::env::var("AVA_FAST_INFER").map_or(true, |v| v != "0"));
    let tune = Prepared::new(&bpe, &tune_items, cfg.glance, cfg.vocab, a.norm);
    let chk = Prepared::new(&bpe, &check_items, cfg.glance, cfg.vocab, a.norm);

    let patterns = if a.params.trim() == "h1" {
        let data = varmap.data().lock().unwrap();
        h1_patterns(data.keys().map(String::as_str))?
    } else {
        a.params.clone()
    };
    let sel = select(&varmap, &patterns)?;
    let tuned: Vec<String> = sel.iter().map(|(n, _)| n.clone()).collect();
    let mut es = Es::new(sel, a.es)?;
    let weights = es.count();
    let threads = candle_core::utils::get_num_threads();
    println!(
        "es_tune: {} ({:?} d {} layers {}{}) on {}; {} tensors, {} weights tuned ({patterns})",
        a.model.display(),
        cfg.kind,
        cfg.d,
        cfg.layers,
        if cfg.arch.flags().is_empty() { String::new() } else { format!(", {}", cfg.arch.flags()) },
        if dev.is_cpu() { format!("cpu, {threads} threads") } else { "cuda".to_string() },
        tuned.len(),
        weights
    );
    println!(
        "items: {} ({} tune, {} check; split seed {}, share {}); fitness {} (tau {}, norm {}); pop {} sigma {} lr {} rank {} seed {}, {} generations of {} evaluations",
        items.len(),
        tune.len(),
        chk.len(),
        a.split_seed,
        a.tune_share,
        a.fitness.name(),
        a.tau,
        a.norm.name(),
        a.es.pop,
        a.es.sigma,
        a.es.lr,
        a.es.rank,
        a.es.seed,
        a.gens,
        2 * a.es.pop
    );

    // the centre's tune and check scores (check is reported only; nothing is chosen by it)
    let centre = |m: &Model| -> Result<(f64, f64, Vec<Graded>, Vec<Graded>), String> {
        let gt = tune.grade(m, &dev, a.batch)?;
        let gc = if chk.is_empty() { Vec::new() } else { chk.grade(m, &dev, a.batch)? };
        let fc = if chk.is_empty() { f64::NAN } else { chk.fitness(&gc, a.fitness, a.tau) };
        Ok((tune.fitness(&gt, a.fitness, a.tau), fc, gt, gc))
    };
    let log = log_path(&a.out);
    if let Some(dir) = a.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    let t0 = Instant::now();
    let (f_tune, f_check, gt0, gc0) = centre(&model)?;
    let s0 = t0.elapsed().as_secs_f64();
    println!("gen 0: centre tune {f_tune:+.4} check {f_check:+.4} ({s0:.2} s, {:.0} tok/s)", (tune.tokens() + chk.tokens()) as f64 / s0.max(1e-9));
    let mut rows = vec![Row { gen: 0, members: None, tune: f_tune, check: f_check, secs: s0, tok_per_s: (tune.tokens() + chk.tokens()) as f64 / s0.max(1e-9) }];
    write_log(&log, a, &rows)?;
    let mut fit = |m: &Model| -> Result<f64, String> {
        let g = tune.grade(m, &dev, a.batch)?;
        Ok(tune.fitness(&g, a.fitness, a.tau))
    };
    let mut last = (gt0.clone(), gc0.clone());
    for gen in 1..=a.gens {
        let t = Instant::now();
        let rep = es.generation(&model, &mut fit)?;
        let (ft, fc, gt, gc) = centre(&model)?;
        let secs = t.elapsed().as_secs_f64();
        let tok = rep.evals as f64 * tune.tokens() as f64 / rep.eval_secs.max(1e-9);
        let rel = rep.step / rep.norm.max(1e-30);
        println!(
            "gen {gen}/{}: members mean {:+.4} best {:+.4} | centre tune {ft:+.4} check {fc:+.4} | step {:.2e} ({:.3}% of |θ|{}) | {secs:.2} s/gen, {:.2} evals/s, {tok:.0} tok/s",
            a.gens,
            rep.mean(),
            rep.best(),
            rep.step,
            100.0 * rel,
            if rel > 0.1 { "; large: try a smaller --lr" } else { "" },
            rep.evals_per_sec()
        );
        rows.push(Row { gen, members: Some(rep), tune: ft, check: fc, secs, tok_per_s: tok });
        write_log(&log, a, &rows)?;
        last = (gt, gc);
    }
    es.restore()?;
    varmap.save(&a.out).map_err(|e| format!("write {}: {e}", a.out.display()))?;
    if let Some((ar, _)) = crate::ckpt::arch_beside(&a.model)? {
        crate::ckpt::write_arch_beside(&a.out, &ar)?;
    }
    let secs = t_run.elapsed().as_secs_f64();
    println!("tune  start: {}", rwa(&gt0, a.tau));
    println!("tune  end:   {}", rwa(&last.0, a.tau));
    if !chk.is_empty() {
        println!("check start: {}", rwa(&gc0, a.tau));
        println!("check end:   {}", rwa(&last.1, a.tau));
    }
    let gen_secs: Vec<f64> = rows[1..].iter().map(|r| r.secs).collect();
    let eps: Vec<f64> = rows[1..].iter().filter_map(|r| r.members.as_ref().map(GenReport::evals_per_sec)).collect();
    let mean = |v: &[f64]| if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 };
    println!("wrote {} (and {}) in {secs:.1} s", a.out.display(), log.display());
    let s = Summary { n_tune: tune.len(), n_check: chk.len(), tuned, weights, rows, log, secs };
    println!(
        "RESULT fitness={} n_tune={} n_check={} tune0={:.6} tune={:.6} check0={:.6} check={:.6} gens={} s_per_gen={:.3} evals_per_s={:.3}",
        a.fitness.name(),
        s.n_tune,
        s.n_check,
        s.tune0(),
        s.tune(),
        s.check0(),
        s.check(),
        a.gens,
        mean(&gen_secs),
        mean(&eps)
    );
    Ok(s)
}
