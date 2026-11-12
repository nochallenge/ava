// TESTING / DESIGN ONLY — episode_surprise (v42 Wave 4, L1): how much each episode surprised the live brain.
//
// episode_surprise --model FILE --episodes FILE [--ids FILE] [--out FILE] [--per-token] [--arch D]
//                  [--batch-tokens 32768] [--bpe FILE] [--cpu] [--threads 2]
//
// Episodes are blank-line separated (one per line in a file without blank lines): grow_gpu's rule, the form of the
// learner's work/batch.txt. Each is read as [EOS] text [EOS] (sleep's form) and every token after the first EOS is
// scored, the closing EOS too (probe::surprise, one batched pass). OUT (default stdout) is a TSV:
//   id  nats_per_token  tokens  [per_token]
// id is the episode's index, or with --ids FILE (one id per line, e.g. the learner's work/batch_ids.txt) its id; the
// learner writes a lone episode twice, so one id for two identical episodes gives one row. --per-token adds each token's
// surprise (nats, comma separated), the input of the learner's surprise cuts (episodes::surprise_cuts). The brain is
// used as it is (every part on). The speed goes to stderr. The device is CUDA unless --cpu (then --threads, default 2; --batch-tokens 256 by default there).

use ava_data::bpe::Bpe;
use ava_gpu::model::Kind;
use ava_gpu::probe;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: episode_surprise --model FILE --episodes FILE [--ids FILE] [--out FILE] [--per-token] [--arch D]\n\
    \x20                       [--batch-tokens 32768] [--bpe FILE] [--cpu] [--threads 2]";

fn main() {
    if let Err(e) = run() {
        eprintln!("episode_surprise: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let (mut model, mut eps_path, mut ids_path, mut out): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None, None);
    let (mut per_token, mut cpu, mut threads, mut batch, mut kind) = (false, false, 2usize, None::<usize>, Kind::D);
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        match key {
            "-h" | "--help" => return Err(USAGE.into()),
            "--per-token" | "--cpu" => {
                if key == "--cpu" {
                    cpu = true;
                } else {
                    per_token = true;
                }
                i += 1;
                continue;
            }
            _ => {}
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
        match key {
            "--model" => model = Some(v.into()),
            "--episodes" => eps_path = Some(v.into()),
            "--ids" => ids_path = Some(v.into()),
            "--out" => out = Some(v.into()),
            "--arch" => kind = Kind::parse(v)?,
            "--batch-tokens" => batch = Some(num(v)?.max(1)),
            "--bpe" => bpe_path = v.into(),
            "--threads" => threads = num(v)?.max(1),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let batch = batch.unwrap_or(if cpu { probe::CPU_BATCH_TOKENS } else { 32768 });
    let model = model.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    let eps_path = eps_path.ok_or_else(|| format!("--episodes is required\n{USAGE}"))?;
    let mut eps = probe::read_episodes(&eps_path)?;
    if eps.is_empty() {
        return Err(format!("{}: no episodes", eps_path.display()));
    }
    let ids: Vec<String> = match &ids_path {
        None => (0..eps.len()).map(|k| k.to_string()).collect(),
        Some(p) => {
            probe::check_path(p, "--ids")?;
            let ids: Vec<String> = std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))?.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect();
            if ids.len() == 1 && eps.len() == 2 && eps[0] == eps[1] {
                eps.truncate(1); // the learner's lone episode, written twice
            }
            if ids.len() != eps.len() {
                return Err(format!("{} ids for {} episodes", ids.len(), eps.len()));
            }
            ids
        }
    };
    if cpu {
        ava_gpu::diary::set_threads(threads); // before the first tensor op
    }
    let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let (cfg, _varmap, m) = ava_gpu::ckpt::load_auto(&model, kind, 4096, &dev)?;
    m.set_track_activity(false);
    m.set_inference(true);
    let t0 = Instant::now();
    let (seqs, from) = probe::episode_seqs(&bpe, cfg.glance, &eps);
    let per = probe::token_nll(&m, &dev, &seqs, &from, batch).map_err(|e| format!("score: {e}"))?;
    let secs = t0.elapsed().as_secs_f64();
    let mut s = String::from(if per_token { "id\tnats_per_token\ttokens\tper_token\n" } else { "id\tnats_per_token\ttokens\n" });
    for (id, v) in ids.iter().zip(&per) {
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len().max(1) as f64;
        s += &format!("{}\t{mean:.4}\t{}", ava_gpu::diary::escape(id), v.len());
        if per_token {
            s += "\t";
            s += &v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(",");
        }
        s.push('\n');
    }
    match &out {
        Some(p) => {
            let tmp = p.with_extension("tmp");
            std::fs::write(&tmp, &s).map_err(|e| format!("write {}: {e}", tmp.display()))?;
            std::fs::rename(&tmp, p).map_err(|e| format!("rename to {}: {e}", p.display()))?;
        }
        None => std::io::stdout().write_all(s.as_bytes()).map_err(|e| e.to_string())?,
    }
    let toks: usize = seqs.iter().map(Vec::len).sum();
    eprintln!("episode_surprise: {} episodes, {toks} tokens in {secs:.2}s ({:.0} tok/s)", eps.len(), toks as f64 / secs.max(1e-9));
    Ok(())
}
