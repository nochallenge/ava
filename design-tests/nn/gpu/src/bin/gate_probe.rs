// TESTING / DESIGN ONLY — gate_probe (v42 Wave 4): the always-online learner's gate command (ava_learner --gate-cmd).
//
// gate_probe --old FILE --new FILE --out FILE [--probe-root DIR] [--per-skill 32] [--skills a,b,...] [--items TSV ...]
//            [--self-questions FILE] [--self-tokens 32] [--score acc|prob] [--norm total|byte|pmi] [--arch D]
//            [--batch-tokens 32768] [--gen-batch-tokens 8192] [--bpe FILE] [--cpu] [--threads 2]
//
// The learner calls it with the live brain (old) and a candidate (new) and reads OUT:
//   skill<TAB>name<TAB>old<TAB>new          one row per probe item (--score acc: 1 or 0; prob: the right choice's share)
//   self<TAB>index<TAB>old answer<TAB>new   one row per self-question (greedy answers, escaped as the learner unescapes)
// Probe items (src/probe.rs): the dev halves under --probe-root (default data/nn): every NAME_dev.tsv in an
// evalsets_v42* folder; a name with sealed, test, spent or heldout is refused and the retired sets are left out. Each
// skill uses its --per-skill items with the lowest hash, the same items at every gate. --skills keeps the named ones;
// --items FILE (repeatable) uses these files instead (skill = the file's stem without _dev; the same name check).
// Each brain is loaded in turn (one in memory at a time) and graded in one batched pass over every skill (mc::grade,
// --norm, total by default), then answers the self-questions in one batched greedy pass (probe::answers;
// --self-questions FILE, else probe::SELF_QUESTIONS). The learner keys its identity anchor by index, so a questions file
// must only ever grow at the end. OUT is written whole and then renamed into place. A summary per skill and the speed
// go to stdout. The device is CUDA unless --cpu (then --threads, default 2; with --cpu the batch budgets default to
// probe::CPU_BATCH_TOKENS, 256, since the CPU forward keeps about 2 MB per batch token).

use ava_data::bpe::Bpe;
use ava_data::rng::SplitMix64;
use ava_gpu::mc::{self, Graded, Item};
use ava_gpu::model::Kind;
use ava_gpu::probe::{self, GenOpts, Score};
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: gate_probe --old FILE --new FILE --out FILE [--probe-root DIR] [--per-skill 32] [--skills a,b]\n\
    \x20                 [--items TSV ...] [--self-questions FILE] [--self-tokens 32] [--score acc|prob]\n\
    \x20                 [--norm total|byte|pmi] [--arch D] [--batch-tokens 32768] [--gen-batch-tokens 8192] [--bpe FILE]\n\
    \x20                 [--cpu] [--threads 2]";
const PROBE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/nn");

struct Opts {
    old: PathBuf,
    new: PathBuf,
    out: PathBuf,
    root: PathBuf,
    per_skill: usize,
    skills: Option<Vec<String>>,
    items: Vec<PathBuf>,
    self_q: Option<PathBuf>,
    self_tokens: usize,
    score: Score,
    norm: mc::Norm,
    kind: Kind,
    batch: usize,
    gen_batch: usize,
    bpe: PathBuf,
    cpu: bool,
    threads: usize,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("gate_probe: {e}");
        std::process::exit(1);
    }
}

fn parse() -> Result<Opts, String> {
    let mut o = Opts {
        old: PathBuf::new(),
        new: PathBuf::new(),
        out: PathBuf::new(),
        root: PathBuf::from(PROBE_ROOT),
        per_skill: 32,
        skills: None,
        items: Vec::new(),
        self_q: None,
        self_tokens: 32,
        score: Score::Acc,
        norm: mc::Norm::Total,
        kind: Kind::D,
        batch: 32768,
        gen_batch: 8192,
        bpe: PathBuf::from(ava_gpu::DEFAULT_BPE),
        cpu: false,
        threads: 2,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut batch_given, mut gen_given) = (false, false);
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        match key {
            "-h" | "--help" => return Err(USAGE.into()),
            "--cpu" => {
                o.cpu = true;
                i += 1;
                continue;
            }
            _ => {}
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
        match key {
            "--old" => o.old = v.into(),
            "--new" => o.new = v.into(),
            "--out" => o.out = v.into(),
            "--probe-root" => o.root = v.into(),
            "--per-skill" => o.per_skill = num(v)?.max(1),
            "--skills" => o.skills = Some(v.split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect()),
            "--items" => o.items.push(v.into()),
            "--self-questions" => o.self_q = Some(v.into()),
            "--self-tokens" => o.self_tokens = num(v)?,
            "--score" => o.score = Score::parse(v)?,
            "--norm" => o.norm = mc::Norm::parse(v)?,
            "--arch" => o.kind = Kind::parse(v)?,
            "--batch-tokens" => {
                o.batch = num(v)?.max(1);
                batch_given = true;
            }
            "--gen-batch-tokens" => {
                o.gen_batch = num(v)?.max(1);
                gen_given = true;
            }
            "--bpe" => o.bpe = v.into(),
            "--threads" => o.threads = num(v)?.max(1),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    if o.old.as_os_str().is_empty() || o.new.as_os_str().is_empty() || o.out.as_os_str().is_empty() {
        return Err(format!("--old, --new and --out are required\n{USAGE}"));
    }
    // the CPU forward needs far more memory per token than the GPU kernels (probe::CPU_BATCH_TOKENS)
    if o.cpu && !batch_given {
        o.batch = probe::CPU_BATCH_TOKENS;
    }
    if o.cpu && !gen_given {
        o.gen_batch = probe::CPU_BATCH_TOKENS;
    }
    Ok(o)
}

/// The probe sets: (skill, its fixed sample).
fn probe_sets(o: &Opts) -> Result<Vec<(String, Vec<Item>)>, String> {
    let files: Vec<(String, PathBuf)> = if o.items.is_empty() {
        probe::dev_probe_files(&o.root)?
    } else {
        o.items
            .iter()
            .map(|p| {
                probe::check_path(p, "--items")?;
                let stem = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                Ok((stem.strip_suffix("_dev").unwrap_or(&stem).to_string(), p.clone()))
            })
            .collect::<Result<_, String>>()?
    };
    let mut sets = Vec::new();
    for (skill, path) in files {
        if o.skills.as_ref().is_some_and(|k| !k.contains(&skill)) {
            continue;
        }
        if sets.iter().any(|(s, _): &(String, Vec<Item>)| *s == skill) {
            return Err(format!("skill {skill:?} is given twice"));
        }
        probe::check_path(&path, "probe")?;
        let items = probe::pick_fixed(mc::read_items(&path)?, &skill, o.per_skill);
        if !items.is_empty() {
            sets.push((skill, items));
        }
    }
    if let Some(k) = &o.skills {
        for s in k {
            if !sets.iter().any(|(x, _)| x == s) {
                return Err(format!("--skills: no probe set {s:?}"));
            }
        }
    }
    if sets.is_empty() {
        return Err("no probe items".into());
    }
    Ok(sets)
}

/// One brain: every item graded (in item order) and the self-answers; and the seconds it took.
fn brain(o: &Opts, path: &PathBuf, items: &[Item], qs: &[String], bpe: &Bpe, dev: &candle_core::Device) -> Result<(Vec<Graded>, Vec<String>, f64), String> {
    let t0 = Instant::now();
    let (cfg, _varmap, model) = ava_gpu::ckpt::load_auto(path, o.kind, 4096, dev)?;
    model.set_track_activity(false);
    model.set_inference(true);
    let g = mc::grade(&model, dev, bpe, items, cfg.glance, o.batch, o.norm)?;
    let gen = GenOpts { max_new: o.self_tokens, temperature: 0.0, top_k: 1, batch_tokens: o.gen_batch };
    let mut rngs: Vec<SplitMix64> = (0..qs.len()).map(|i| SplitMix64::new(i as u64)).collect();
    let answers = if qs.is_empty() || o.self_tokens == 0 { Vec::new() } else { probe::answers(&model, dev, bpe, qs, &gen, &mut rngs)? };
    Ok((g, answers, t0.elapsed().as_secs_f64()))
}

fn run() -> Result<(), String> {
    let o = parse()?;
    for p in [&o.old, &o.new] {
        if !p.is_file() {
            return Err(format!("{} is not a file", p.display()));
        }
    }
    let sets = probe_sets(&o)?;
    let qs: Vec<String> = match &o.self_q {
        Some(p) => probe::read_questions(p)?,
        None => probe::SELF_QUESTIONS.iter().map(|s| s.to_string()).collect(),
    };
    if o.cpu {
        ava_gpu::diary::set_threads(o.threads); // before the first tensor op
    }
    let dev = ava_gpu::device(o.cpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&o.bpe).map_err(|e| format!("bpe {}: {e}", o.bpe.display()))?;
    let items: Vec<Item> = sets.iter().flat_map(|(_, its)| its.iter().cloned()).collect();
    // tokens one brain reads for the probes (each choice is its own sequence), for the speed line
    let toks: usize = items.iter().map(|it| mc::item_seqs(&bpe, it, false).0.iter().map(Vec::len).sum::<usize>()).sum();
    println!("gate_probe: {} skills, {} items ({} tokens per brain), {} self-questions; scoring {}", sets.len(), items.len(), toks, qs.len(), o.norm.name());
    let (g_old, a_old, s_old) = brain(&o, &o.old, &items, &qs, &bpe, &dev)?;
    let (g_new, a_new, s_new) = brain(&o, &o.new, &items, &qs, &bpe, &dev)?;
    let mut rows: Vec<String> = Vec::with_capacity(items.len() + qs.len());
    let mut k = 0;
    for (skill, its) in &sets {
        let (go, gn) = (&g_old[k..k + its.len()], &g_new[k..k + its.len()]);
        k += its.len();
        let r = probe::skill_rows(skill, its, go, gn, o.score);
        let mean = |g: &[Graded]| g.iter().zip(its).map(|(x, it)| probe::item_score(x, it.right, o.score)).sum::<f64>() / its.len() as f64;
        println!("  {skill:<12} n={:<3} old {:.3}  new {:.3}  ({:+.3})", its.len(), mean(go), mean(gn), mean(gn) - mean(go));
        rows.extend(r);
    }
    rows.extend(probe::self_rows(&a_old, &a_new));
    let changed = a_old.iter().zip(&a_new).filter(|(a, b)| a != b).count();
    println!("  self-answers: {changed} of {} changed", a_old.len());
    let tmp = o.out.with_extension("tmp");
    std::fs::write(&tmp, rows.join("\n") + "\n").map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &o.out).map_err(|e| format!("rename to {}: {e}", o.out.display()))?;
    println!(
        "wrote {} rows to {}; old brain {s_old:.2}s, new {s_new:.2}s ({:.0} probe tokens/s, both brains)",
        rows.len(),
        o.out.display(),
        2.0 * toks as f64 / (s_old + s_new).max(1e-9)
    );
    Ok(())
}
