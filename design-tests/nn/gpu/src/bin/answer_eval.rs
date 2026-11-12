// TESTING / DESIGN ONLY — answer_eval (v43): ask her a question set, the way the chat app does, and grade it.
//
// answer_eval --model FILE --questions TSV [--library FILE] [--k 2] [--tokens 40] [--int8] [--threads 8]
//             [--limit N] [--out FILE] [--arch D] [--bpe FILE]
//
// * TSV columns: id, cat, question, accept (header line first). Pre-registration: runs/v43/prereg_answers.md.
// * Each question is asked fresh: [EOS] + "Context: <her library's best --k passages>\nQuestion: q\nAnswer:" (no
//   Context line without --library or when nothing is cued), then greedy decoding (diary::reply, temperature 0)
//   until EOS, a new "Question:" or --tokens. CPU; --int8 is what the chat app runs.
// * Grades (lookup::grade): right, abstain ("I don't know"), wrong. Prints per category and in all: right − wrong,
//   precision = right / (right + wrong), coverage = (right + wrong) / n; and the speed. --out writes every item.

use ava_data::bpe::{Bpe, EOS};
use ava_data::rng::SplitMix64;
use ava_gpu::diary;
use ava_gpu::lookup::{grade, Grade, Library};
use ava_gpu::model::Kind;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("answer_eval: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let num = |k: &str, d: usize| -> Result<usize, String> { get(k).map(|v| v.parse().map_err(|_| format!("bad {k}"))).transpose().map(|v| v.unwrap_or(d)) };
    let model_path = PathBuf::from(get("--model").ok_or("--model FILE")?);
    let qpath = PathBuf::from(get("--questions").ok_or("--questions TSV")?);
    let k = num("--k", 2)?;
    let tokens = num("--tokens", 40)?;
    let limit = num("--limit", usize::MAX)?;
    let int8 = args.iter().any(|a| a == "--int8");
    diary::set_threads(num("--threads", 8)?);
    let kind = Kind::parse(&get("--arch").unwrap_or_else(|| "D".into()))?;
    let bpe_path = PathBuf::from(get("--bpe").unwrap_or_else(|| ava_gpu::DEFAULT_BPE.into()));

    let dev = ava_gpu::device(true).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let (cfg, _varmap, model) = ava_gpu::ckpt::load_auto(&model_path, kind, 4096, &dev)?;
    model.set_track_activity(false);
    model.set_inference(true);
    if int8 {
        model.set_int8(true).map_err(|e| format!("--int8: {e}"))?;
        model.set_fused(true);
    }
    let lib = match get("--library") {
        Some(p) => {
            let t = Instant::now();
            let l = Library::load(&PathBuf::from(&p))?;
            eprintln!("library: {} passages, loaded in {:.1}s", l.len(), t.elapsed().as_secs_f64());
            Some(l)
        }
        None => None,
    };

    let text = std::fs::read_to_string(&qpath).map_err(|e| format!("{}: {e}", qpath.display()))?;
    let rows: Vec<Vec<&str>> = text.lines().skip(1).filter(|l| !l.trim().is_empty()).map(|l| l.split('\t').collect()).take(limit).collect();
    let mut out = match get("--out") {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(&p).map_err(|e| format!("{p}: {e}"))?)),
        None => None,
    };
    if let Some(w) = out.as_mut() {
        writeln!(w, "id\tcat\tgrade\tanswer\tcontext").map_err(|e| e.to_string())?;
    }
    let mut by: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    let mut rng = SplitMix64::new(1);
    let (mut steps, t0) = (0usize, Instant::now());
    for r in &rows {
        if r.len() < 4 {
            return Err(format!("{}: a row without 4 columns: {:?}", qpath.display(), r));
        }
        let (id, cat, q, accept) = (r[0], r[1], r[2], r[3]);
        let ctx = lib.as_ref().map(|l| l.context(q, k)).unwrap_or_default();
        let mut ids = vec![EOS];
        ids.extend(diary::encode(&bpe, cfg.glance, &ava_gpu::qa::prompt_text(&ctx, &diary::one_line(q))));
        let (ans, n) = diary::reply(&model, &bpe, &ids, tokens, 0.0, 1, &mut rng)?;
        steps += ids.len() + n;
        let g = grade(&ans, accept);
        let slot = match g {
            Grade::Right => 0,
            Grade::Abstain => 1,
            Grade::Wrong => 2,
        };
        by.entry(cat.to_string()).or_insert([0; 3])[slot] += 1;
        by.entry("ALL".to_string()).or_insert([0; 3])[slot] += 1;
        if let Some(w) = out.as_mut() {
            let ctx_titles: Vec<String> = lib.as_ref().map(|l| l.search(q, k).iter().map(|h| h.1.to_string()).collect()).unwrap_or_default();
            writeln!(w, "{id}\t{cat}\t{g:?}\t{}\t{}", ans.replace(['\t', '\n'], " "), ctx_titles.join(" | ")).map_err(|e| e.to_string())?;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    println!("cat\tn\tright\tabstain\twrong\tright-wrong\tprecision\tcoverage");
    for (cat, [r, a, w]) in &by {
        let n = r + a + w;
        let prec = if r + w > 0 { *r as f64 / (r + w) as f64 } else { f64::NAN };
        println!("{cat}\t{n}\t{r}\t{a}\t{w}\t{}\t{prec:.3}\t{:.3}", *r as i64 - *w as i64, (r + w) as f64 / n.max(1) as f64);
    }
    println!("{} questions in {secs:.1}s ({:.0} tokens/s, {} tokens read+written)", rows.len(), steps as f64 / secs.max(1e-9), steps);
    Ok(())
}
