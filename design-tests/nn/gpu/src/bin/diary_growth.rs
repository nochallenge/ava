// TESTING / DESIGN ONLY — diary_growth: how much Ava has learned from your corrections (her diary, src/diary.rs).
//
// diary_growth --model FILE [--diary data/diary] [--out DIARY/growth.tsv] [--items-out DIARY/growth-items.tsv] [--cpu]
//              [--arch D] [--bpe FILE] [--batch 8192] [--tokens 40] [--threads N]
//
// For every corrected turn in the whole diary (every DIARY/YYYY-MM-DD.tsv, the last /correct of each turn):
//   - her NLL per byte of your right answer given the question, and of her original reply, both scored in one batched
//     pass as mc_eval --norm byte scores a choice ("Question: q\nAnswer:" then " answer" [EOS]; the answer's tokens
//     and the closing EOS scored, divided by their bytes; glance marks as she was taught), from a fresh memory;
//   - her fresh reply now (the most likely token each step, up to --tokens, cut at a new question), and whether it
//     says the right answer (equal after lowercasing and dropping punctuation, or starting with it as whole words).
// One summary line per run is appended to --out (run time, model, days, corrections, mean NLL/byte of the right
// answers, share where the right answer beats her original reply, share whose fresh reply says it, mean NLL/byte of
// her original replies, seconds); per-item lines go to --items-out first. Re-run after each sleep: the lines of
// growth.tsv are the curve of her learning from you. Both files hold diary text and must stay private: their paths
// must contain "diary". The diary itself is only read.

use ava_data::bpe::Bpe;
use ava_data::rng::SplitMix64;
use ava_gpu::diary;
use ava_gpu::mc::{self, Item, Norm};
use ava_gpu::model::Kind;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: diary_growth --model FILE [--diary data/diary] [--out DIARY/growth.tsv] [--items-out DIARY/growth-items.tsv] [--cpu]\n\
    \x20                   [--arch D] [--bpe FILE] [--batch 8192] [--tokens 40] [--threads N]";

const SUMMARY_HEADER: &str = "run\tmodel\tdays\tcorrections\tmean_right_nll_per_byte\tright_beats_original\tfresh_says_right\tmean_original_nll_per_byte\tseconds";
const ITEMS_HEADER: &str = "run\tmodel\tdate\tturn\tquestion\tright\toriginal\tright_nll_per_byte\toriginal_nll_per_byte\tright_beats_original\tfresh\tfresh_says_right";

fn main() {
    if let Err(e) = run() {
        eprintln!("diary_growth: {e}");
        std::process::exit(1);
    }
}

/// Append one line (the header first if the file is new), synced; nothing already there is touched.
fn append_tsv(path: &Path, header: &str, lines: &[String]) -> Result<(), String> {
    let ctx = |e: std::io::Error| format!("{}: {e}", path.display());
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(ctx)?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(ctx)?;
    let mut buf = String::new();
    if f.metadata().map_err(ctx)?.len() == 0 {
        buf.push_str(header);
        buf.push('\n');
    }
    for l in lines {
        buf.push_str(l);
        buf.push('\n');
    }
    f.write_all(buf.as_bytes()).map_err(ctx)?;
    f.sync_data().map_err(ctx)
}

fn run() -> Result<(), String> {
    let mut model_path: Option<PathBuf> = None;
    let mut dir = PathBuf::from(diary::DIARY_DIR);
    let (mut out, mut items_out): (Option<PathBuf>, Option<PathBuf>) = (None, None);
    let mut kind = Kind::D;
    let mut cpu = false;
    let (mut batch, mut max_tokens) = (8192usize, 40usize);
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        match key {
            "-h" | "--help" => return Err(USAGE.into()),
            "--cpu" => {
                cpu = true;
                i += 1;
                continue;
            }
            _ => {}
        }
        let val = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
        match key {
            "--model" => model_path = Some(PathBuf::from(val)),
            "--diary" => dir = PathBuf::from(val),
            "--out" => out = Some(PathBuf::from(val)),
            "--items-out" => items_out = Some(PathBuf::from(val)),
            "--arch" => kind = Kind::parse(val)?,
            "--bpe" => bpe_path = PathBuf::from(val),
            "--batch" => batch = num(val)?,
            "--tokens" => max_tokens = num(val)?,
            "--threads" => diary::set_threads(num(val)?),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let model_path = model_path.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    let out = out.unwrap_or_else(|| dir.join("growth.tsv"));
    let items_out = items_out.unwrap_or_else(|| dir.join("growth-items.tsv"));
    for p in [&out, &items_out] {
        if !p.to_string_lossy().to_lowercase().contains("diary") || diary::day_of(p).is_some() {
            return Err(format!("{}: growth files hold diary text: the path must contain \"diary\" and not be a day file", p.display()));
        }
    }

    // every corrected turn of the whole diary
    let days = diary::day_files(&dir)?;
    let mut turns = Vec::new();
    for p in &days {
        let date = diary::day_of(p).expect("day_files gives day files");
        let (lines, _) = diary::read_day(p)?;
        turns.extend(diary::resolve(&date, &lines).0.into_iter().filter(|t| t.correction.as_deref().is_some_and(|c| !diary::one_line(c).is_empty())));
    }
    let name = diary::model_name(&model_path);
    let (_, run_time) = diary::local_now();
    println!("{} diary days, {} corrected turns; model {name}", days.len(), turns.len());

    let t_all = Instant::now();
    let (mut right_nll, mut orig_nll, mut fresh) = (Vec::new(), Vec::new(), Vec::new());
    if !turns.is_empty() {
        let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
        let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
        let (cfg, _varmap, model) = ava_gpu::ckpt::load_auto(&model_path, kind, 4096, &dev)?;
        model.set_track_activity(false);
        model.set_inference(true);
        let items: Vec<Item> = turns
            .iter()
            .map(|t| Item {
                id: format!("{}#{}", t.date, t.turn),
                kind: "diary".into(),
                bin: String::new(),
                prompt: ava_gpu::qa::prompt_text("", &diary::one_line(&t.user)),
                choices: vec![diary::one_line(t.correction.as_deref().unwrap_or("")), diary::one_line(&t.ava)],
                right: 0,
            })
            .collect();
        // one batched pass: the right answer and her original reply of every item
        let (mut seqs, mut from) = (Vec::new(), Vec::new());
        for it in &items {
            let (s, f) = mc::item_seqs(&bpe, it, cfg.glance);
            seqs.extend(s);
            from.extend(f);
        }
        let t0 = Instant::now();
        let nll = mc::scored_nll(&model, &dev, &seqs, &from, batch).map_err(|e| format!("score: {e}"))?;
        let bytes = mc::scored_bytes(&ava_gpu::eval::byte_lengths_n(&bpe, cfg.vocab), &seqs, &from);
        let per_byte = mc::normed(&nll, &bytes, Norm::Byte);
        let s_secs = t0.elapsed().as_secs_f64();
        let s_tok: usize = seqs.iter().map(Vec::len).sum();
        println!("scored {} answers ({s_tok} tokens) in {s_secs:.2}s, {:.0} tokens/s", seqs.len(), s_tok as f64 / s_secs.max(1e-9));
        for k in 0..items.len() {
            right_nll.push(per_byte[2 * k]);
            orig_nll.push(per_byte[2 * k + 1]);
        }
        // what she says now, from a fresh memory, the most likely token each step
        let t0 = Instant::now();
        let (mut rng, mut g_tok, mut p_tok) = (SplitMix64::new(1), 0usize, 0usize);
        for t in &turns {
            if max_tokens == 0 {
                fresh.push(String::new());
                continue;
            }
            let (ids, _) = diary::prompt_ids(&bpe, cfg.glance, &[], &t.user, 0);
            let (r, n) = diary::reply(&model, &bpe, &ids, max_tokens, 0.0, 1, &mut rng)?;
            p_tok += ids.len();
            g_tok += n;
            fresh.push(r);
        }
        let g_secs = t0.elapsed().as_secs_f64();
        if max_tokens > 0 {
            println!(
                "fresh replies: {} prompts ({p_tok} tokens) + {g_tok} written in {g_secs:.2}s, {:.0} tokens/s (one token at a time)",
                turns.len(),
                (p_tok + g_tok) as f64 / g_secs.max(1e-9)
            );
        }
    }
    let secs = t_all.elapsed().as_secs_f64();

    let n = turns.len();
    let mean = |v: &[f64]| if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 };
    let beats: Vec<bool> = (0..n).map(|k| right_nll[k] < orig_nll[k]).collect();
    let says: Vec<bool> = (0..n).map(|k| max_tokens > 0 && diary::says(&fresh[k], turns[k].correction.as_deref().unwrap_or(""))).collect();
    let share = |v: &[bool]| if v.is_empty() { f64::NAN } else { v.iter().filter(|&&b| b).count() as f64 / v.len() as f64 };
    let item_lines: Vec<String> = (0..n)
        .map(|k| {
            let t = &turns[k];
            let e = diary::escape;
            format!(
                "{run_time}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{}",
                e(&name),
                t.date,
                t.turn,
                e(&t.user),
                e(t.correction.as_deref().unwrap_or("")),
                e(&t.ava),
                right_nll[k],
                orig_nll[k],
                beats[k] as u8,
                e(&fresh[k]),
                says[k] as u8
            )
        })
        .collect();
    if !item_lines.is_empty() {
        append_tsv(&items_out, ITEMS_HEADER, &item_lines)?;
    }
    let summary = format!(
        "{run_time}\t{}\t{}\t{n}\t{:.6}\t{:.4}\t{:.4}\t{:.6}\t{secs:.2}",
        diary::escape(&name),
        days.len(),
        mean(&right_nll),
        share(&beats),
        share(&says),
        mean(&orig_nll)
    );
    append_tsv(&out, SUMMARY_HEADER, &[summary])?;
    for k in 0..n {
        let t = &turns[k];
        println!(
            "  {}#{}: right {:.3} vs her old reply {:.3} nats/byte{}; now says {:?}{}",
            t.date,
            t.turn,
            right_nll[k],
            orig_nll[k],
            if beats[k] { " (right is likelier)" } else { "" },
            fresh[k],
            if says[k] { " = right" } else { "" }
        );
    }
    println!(
        "growth: {n} corrections; mean NLL/byte of the right answers {:.4} (her old replies {:.4}); right beats her old reply in {:.1}%; fresh reply says it in {:.1}%; {secs:.1}s",
        mean(&right_nll),
        mean(&orig_nll),
        100.0 * share(&beats),
        100.0 * share(&says)
    );
    println!("appended to {} (and {} item lines to {})", out.display(), item_lines.len(), items_out.display());
    Ok(())
}
