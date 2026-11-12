// TESTING / DESIGN ONLY — think (v42 Wave 4, B11 + F1): the always-online learner's dream command (--dream-cmd).
//
// think --model FILE --out FILE [--dreams 8] [--dream-tokens 64] [--about FILE] [--about-max 16] [--self N]
//       [--self-questions FILE] [--answer-tokens 40] [--temperature 0.8] [--top-k 40] [--seed S] [--arch D]
//       [--gen-batch-tokens 8192] [--bpe FILE] [--cpu] [--threads 2] [--int8]
//
// --int8: the int8 path on the CPU (Model::set_int8, src/int8.rs; implies --cpu).
//
// Samples text from the live brain and writes it to OUT, blank-line separated, the form the learner reads. The learner
// keeps every text as dreamt (never recalled or learned as fact, at most its small share of a rehearsal batch).
// - Dreams (--dreams N): free text from a fresh state, as if starting a new document after EOS, up to --dream-tokens.
// - Thinking back (--about FILE: episodes, blank-line separated, e.g. the learner's work/batch.txt; at most --about-max):
//   an episode "Question: q\nAnswer: a" is asked again and written with her answer now ("Question: q\nAnswer: ...");
//   other text gives the first half of its first sentence (at most 24 words) as a prompt and she goes on from it
//   (a row where she adds nothing is dropped, so no heard text comes back as a dream).
// - Self-questions (--self N): the first N of --self-questions FILE (else probe::SELF_QUESTIONS), answered.
// Every row is sampled with --temperature and --top-k from its own rng (seed, kind, row), so the batch packing never
// changes what a row says; --temperature 0 is greedy. The default seed comes from the clock and is printed, so a run can
// be repeated with --seed. The rows of each kind (dreams, questions, going on from a text) go through one batched
// generation pass (probe::generate). A text is put in one block (blank lines inside it become one newline); empty texts
// and empty answers are dropped. OUT is written whole, then renamed into place. The device is CUDA unless --cpu (then
// --threads, default 2; with --cpu --gen-batch-tokens defaults to probe::CPU_BATCH_TOKENS).

use ava_data::bpe::{Bpe, EOS};
use ava_data::numbers::strip_marks;
use ava_data::rng::{derive_seed, SplitMix64};
use ava_gpu::diary;
use ava_gpu::model::Kind;
use ava_gpu::probe::{self, GenOpts};
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: think --model FILE --out FILE [--dreams 8] [--dream-tokens 64] [--about FILE] [--about-max 16] [--self N]\n\
    \x20            [--self-questions FILE] [--answer-tokens 40] [--temperature 0.8] [--top-k 40] [--seed S] [--arch D]\n\
    \x20            [--gen-batch-tokens 8192] [--bpe FILE] [--cpu] [--threads 2]";

struct Opts {
    model: PathBuf,
    out: PathBuf,
    dreams: usize,
    dream_tokens: usize,
    about: Option<PathBuf>,
    about_max: usize,
    self_n: usize,
    self_q: Option<PathBuf>,
    answer_tokens: usize,
    temperature: f32,
    top_k: usize,
    seed: Option<u64>,
    kind: Kind,
    /// None: 8192, or probe::CPU_BATCH_TOKENS with --cpu
    gen_batch: Option<usize>,
    bpe: PathBuf,
    cpu: bool,
    threads: usize,
    /// E10: int8 weights (CPU only)
    int8: bool,
}

/// What a row is: its prompt tokens, the text written before what she says, whether it is a question (her reply is cut
/// as diary::cut_reply cuts it), and its token limit.
struct Row {
    prompt: Vec<u32>,
    lead: String,
    question: bool,
    max: usize,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("think: {e}");
        std::process::exit(1);
    }
}

fn parse() -> Result<Opts, String> {
    let mut o = Opts {
        model: PathBuf::new(),
        out: PathBuf::new(),
        dreams: 8,
        dream_tokens: 64,
        about: None,
        about_max: 16,
        self_n: 0,
        self_q: None,
        answer_tokens: 40,
        temperature: 0.8,
        top_k: 40,
        seed: None,
        kind: Kind::D,
        gen_batch: None,
        bpe: PathBuf::from(ava_gpu::DEFAULT_BPE),
        cpu: false,
        threads: 2,
        int8: false,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
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
            "--int8" => {
                (o.int8, o.cpu) = (true, true);
                i += 1;
                continue;
            }
            _ => {}
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
        match key {
            "--model" => o.model = v.into(),
            "--out" => o.out = v.into(),
            "--dreams" => o.dreams = num(v)?,
            "--dream-tokens" => o.dream_tokens = num(v)?.max(1),
            "--about" => o.about = Some(v.into()),
            "--about-max" => o.about_max = num(v)?,
            "--self" => o.self_n = num(v)?,
            "--self-questions" => o.self_q = Some(v.into()),
            "--answer-tokens" => o.answer_tokens = num(v)?.max(1),
            "--temperature" => o.temperature = v.parse().map_err(|_| format!("bad number {v:?} for {key}"))?,
            "--top-k" => o.top_k = num(v)?.max(1),
            "--seed" => o.seed = Some(v.parse().map_err(|_| format!("bad integer {v:?} for {key}"))?),
            "--arch" => o.kind = Kind::parse(v)?,
            "--gen-batch-tokens" => o.gen_batch = Some(num(v)?.max(1)),
            "--bpe" => o.bpe = v.into(),
            "--threads" => o.threads = num(v)?.max(1),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    if o.model.as_os_str().is_empty() || o.out.as_os_str().is_empty() {
        return Err(format!("--model and --out are required\n{USAGE}"));
    }
    Ok(o)
}

fn run() -> Result<(), String> {
    let o = parse()?;
    if !o.model.is_file() {
        return Err(format!("{} is not a file", o.model.display()));
    }
    let seed = o.seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64));
    let about = match &o.about {
        Some(p) => probe::read_episodes(p)?,
        None => Vec::new(),
    };
    let selfq: Vec<String> = match (&o.self_q, o.self_n) {
        (_, 0) => Vec::new(),
        (Some(p), n) => probe::read_questions(p)?.into_iter().take(n).collect(),
        (None, n) => probe::SELF_QUESTIONS.iter().take(n).map(|s| s.to_string()).collect(),
    };
    if o.cpu {
        diary::set_threads(o.threads); // before the first tensor op
    }
    let dev = ava_gpu::device(o.cpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&o.bpe).map_err(|e| format!("bpe {}: {e}", o.bpe.display()))?;
    let (cfg, _varmap, model) = ava_gpu::ckpt::load_auto(&o.model, o.kind, 4096, &dev)?;
    model.set_track_activity(false);
    model.set_inference(true);
    if o.int8 {
        model.set_int8(true).map_err(|e| format!("--int8: {e}"))?;
    }
    let g = cfg.glance;

    let mut rows: Vec<Row> = Vec::new();
    for _ in 0..o.dreams {
        rows.push(Row { prompt: vec![EOS], lead: String::new(), question: false, max: o.dream_tokens });
    }
    let mut seen = std::collections::HashSet::new();
    for ep in about.iter().filter(|e| seen.insert(e.to_string())).take(o.about_max) {
        let qa = ep.strip_prefix("Question: ").and_then(|r| r.split_once("\nAnswer:"));
        match qa {
            Some((q, _)) => rows.push(Row { prompt: diary::prompt_ids(&bpe, g, &[], q, 0).0, lead: format!("Question: {}\nAnswer:", diary::one_line(q)), question: true, max: o.answer_tokens }),
            None => {
                let lead = probe::opening(ep, 24);
                if lead.is_empty() {
                    continue;
                }
                let mut p = vec![EOS];
                p.extend(diary::encode(&bpe, g, &lead));
                rows.push(Row { prompt: p, lead, question: false, max: o.answer_tokens });
            }
        }
    }
    for q in &selfq {
        rows.push(Row { prompt: diary::prompt_ids(&bpe, g, &[], q, 0).0, lead: format!("Question: {}\nAnswer:", diary::one_line(q)), question: true, max: o.answer_tokens });
    }
    if rows.is_empty() {
        return Err("nothing to think about: give --dreams, --about or --self".into());
    }
    let t0 = Instant::now();
    let mut rngs: Vec<SplitMix64> = (0..rows.len()).map(|r| SplitMix64::new(derive_seed(seed, 0x7468_696e_6b, r as u64))).collect();
    let prompts: Vec<Vec<u32>> = rows.iter().map(|r| r.prompt.clone()).collect();
    let limits: Vec<(usize, bool)> = rows.iter().map(|r| (r.max, r.question)).collect();
    let gen = GenOpts { max_new: 1, temperature: o.temperature, top_k: o.top_k, batch_tokens: o.gen_batch.unwrap_or(if o.cpu { probe::CPU_BATCH_TOKENS } else { 8192 }) };
    let outs = generate_rows(&model, &dev, &bpe, &prompts, &limits, &gen, &mut rngs)?;
    let secs = t0.elapsed().as_secs_f64();
    let written: usize = outs.iter().map(Vec::len).sum();
    let mut texts: Vec<String> = Vec::new();
    for (r, toks) in rows.iter().zip(&outs) {
        let said = bpe.decode(&strip_marks(toks));
        let body = if r.question { diary::cut_reply(&said).0 } else { said };
        let text = if r.lead.is_empty() {
            probe::one_block(&body)
        } else if r.question {
            probe::one_block(&format!("{} {}", r.lead, diary::one_line(&body)))
        } else {
            probe::one_block(&format!("{}{}", r.lead, body))
        };
        // a row where she added nothing is dropped (a bare lead would only copy the heard text)
        if !text.is_empty() && !diary::one_line(&body).is_empty() {
            texts.push(text);
        }
    }
    let tmp = o.out.with_extension("tmp");
    std::fs::write(&tmp, texts.join("\n\n") + "\n").map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &o.out).map_err(|e| format!("rename to {}: {e}", o.out.display()))?;
    println!(
        "think: {} texts ({} dreams, {} thoughts on episodes, {} self-answers asked) to {}; seed {seed}; {written} tokens in {secs:.2}s ({:.0} tok/s)",
        texts.len(),
        o.dreams,
        rows.len() - o.dreams - selfq.len(),
        selfq.len(),
        o.out.display(),
        written as f64 / secs.max(1e-9)
    );
    Ok(())
}

/// probe::generate with a limit per row: rows are grouped by (limit, question) and each group is one batched pass.
fn generate_rows(model: &ava_gpu::model::Model, dev: &candle_core::Device, bpe: &Bpe, prompts: &[Vec<u32>], limits: &[(usize, bool)], gen: &GenOpts, rngs: &mut [SplitMix64]) -> Result<Vec<Vec<u32>>, String> {
    let mut out = vec![Vec::new(); prompts.len()];
    let mut groups: Vec<(usize, bool)> = limits.to_vec();
    groups.sort();
    groups.dedup();
    for (max, question) in groups {
        let idx: Vec<usize> = (0..prompts.len()).filter(|&i| limits[i] == (max, question)).collect();
        let ps: Vec<Vec<u32>> = idx.iter().map(|&i| prompts[i].clone()).collect();
        let mut rs: Vec<SplitMix64> = idx.iter().map(|&i| SplitMix64(rngs[i].0)).collect();
        let stop_q = |toks: &[u32]| diary::cut_reply(&bpe.decode(&strip_marks(toks))).1;
        let never = |_: &[u32]| false;
        let stop: &dyn Fn(&[u32]) -> bool = if question { &stop_q } else { &never };
        let got = probe::generate(model, dev, &ps, &GenOpts { max_new: max, ..*gen }, &mut rs, stop)?;
        for (k, &i) in idx.iter().enumerate() {
            out[i] = got[k].clone();
            rngs[i] = SplitMix64(rs[k].0);
        }
    }
    Ok(out)
}
