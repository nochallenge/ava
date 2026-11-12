// TESTING / DESIGN ONLY — surprisal_gpu (v42 M5): Ava's token surprisals over texts, for the reading-time yardstick.
//
// surprisal_gpu --model FILE --texts FILE.tsv --out FILE.tsv [--arch D] [--ctx N] [--batch-tokens 32768] [--bpe FILE]
//               [--cpu] [--threads 2]
//
// Writes the SURPRISAL FILE of ava_pythia::readtime (the format `pythia_eval surprisal` writes, so readtime_score reads
// both): header `id pos start end surprisal_bits`, one row per token of each text in order, with the token's byte span in
// the text's UTF-8. Each text is read as [EOS] text, as her training text starts after an EOS, and every token of the
// text is scored (probe::token_nll, batched by length). A glance model gets its length marks (diary::encode); a mark is
// a zero-width token, and readtime carries it to the next word. The spans must tile the text, or the text is refused.
// --ctx N: texts longer than N tokens are read in windows of N with half overlap, each window scoring only its second
// half (pythia_eval's rule); without it each text is one sequence. --texts is the `id text` file of readtime_score prep.

use ava_data::bpe::{Bpe, EOS};
use ava_gpu::eval::byte_lengths_n;
use ava_gpu::model::Kind;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "surprisal_gpu --model FILE --texts FILE.tsv --out FILE.tsv [--arch D] [--ctx N] [--batch-tokens 32768] [--bpe FILE] [--cpu] [--threads 2]";
const TEXTS_HEADER: &str = "id\ttext";
const SURPRISAL_HEADER: &str = "id\tpos\tstart\tend\tsurprisal_bits";

fn main() {
    if let Err(e) = run() {
        eprintln!("surprisal_gpu: {e}");
        std::process::exit(1);
    }
}

/// readtime's escape rule, reversed: \\ \t \n (anything else after a backslash is kept as it is).
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('\\') => out.push('\\'),
            Some(o) => {
                out.push('\\');
                out.push(o);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn read_texts(path: &Path) -> Result<Vec<(String, String)>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = s.lines();
    if lines.next().map(|l| l.trim_end_matches('\r')) != Some(TEXTS_HEADER) {
        return Err(format!("{}: header must be {TEXTS_HEADER:?}", path.display()));
    }
    let mut out = Vec::new();
    for (i, l) in lines.enumerate() {
        let l = l.trim_end_matches('\r');
        if l.is_empty() {
            continue;
        }
        let (id, t) = l.split_once('\t').ok_or_else(|| format!("{} line {}: want id<TAB>text", path.display(), i + 2))?;
        out.push((id.to_string(), unescape(t)));
    }
    Ok(out)
}

/// Windows of `ids` ([EOS] then the text's tokens): (start, end, first scored index in the window).
fn windows(len: usize, ctx: Option<usize>) -> Vec<(usize, usize, usize)> {
    let Some(ctx) = ctx else { return vec![(0, len, 1)] };
    let half = ctx / 2;
    let mut out = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + ctx).min(len);
        out.push((start, end, if start == 0 { 1 } else { ctx - half }));
        if end == len {
            break;
        }
        start += half;
    }
    out
}

fn run() -> Result<(), String> {
    let (mut model, mut texts, mut out): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None);
    let (mut kind, mut ctx, mut batch_tokens, mut cpu, mut threads) = (Kind::D, None::<usize>, 32768usize, false, 2usize);
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        if key == "--cpu" {
            cpu = true;
            i += 1;
            continue;
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
        match key {
            "--model" => model = Some(v.into()),
            "--texts" => texts = Some(v.into()),
            "--out" => out = Some(v.into()),
            "--arch" => kind = Kind::parse(v)?,
            "--ctx" => ctx = Some(num(v)?.max(4)),
            "--batch-tokens" => batch_tokens = num(v)?.max(16),
            "--bpe" => bpe_path = v.into(),
            "--threads" => threads = num(v)?.max(1),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let model_path = model.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    let texts_path = texts.ok_or_else(|| format!("--texts is required\n{USAGE}"))?;
    let out = out.ok_or_else(|| format!("--out is required\n{USAGE}"))?;
    ava_gpu::probe::check_path(&texts_path, "--texts")?;
    ava_gpu::probe::check_path(&out, "--out")?;
    let texts = read_texts(&texts_path)?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    if cpu {
        ava_gpu::diary::set_threads(threads); // before the first tensor op
    }
    let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
    // the model is built once its longest sequence is known; the glance flag comes from the file, so encode after loading
    let probe_cfg_ctx = ctx.unwrap_or(4096);
    let (cfg, _varmap, m) = ava_gpu::ckpt::load_auto(&model_path, kind, probe_cfg_ctx, &dev)?;
    let byte_len = byte_lengths_n(&bpe, cfg.vocab);
    let t0 = Instant::now();
    let (mut seqs, mut from, mut owner, mut spans) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (ti, (id, text)) in texts.iter().enumerate() {
        let enc = ava_gpu::diary::encode(&bpe, cfg.glance, text);
        let mut sp = Vec::with_capacity(enc.len());
        let mut p = 0usize;
        for &tok in &enc {
            let n = *byte_len.get(tok as usize).ok_or_else(|| format!("text {id}: token {tok} outside the vocabulary"))? as usize;
            sp.push((p, p + n));
            p += n;
        }
        if p != text.len() {
            return Err(format!("text {id}: its tokens hold {p} bytes, the text {}", text.len()));
        }
        let mut ids = vec![EOS];
        ids.extend(enc);
        if ctx.is_none() && ids.len() > probe_cfg_ctx && kind == Kind::T {
            return Err(format!("text {id}: {} tokens is longer than T's {probe_cfg_ctx} positions; give --ctx", ids.len()));
        }
        for (s, e, f) in windows(ids.len(), ctx) {
            seqs.push(ids[s..e].to_vec());
            from.push(f);
            owner.push(ti);
        }
        spans.push(sp);
    }
    let n_tok: usize = spans.iter().map(|s| s.len()).sum();
    let t1 = Instant::now();
    let nll = ava_gpu::probe::token_nll(&m, &dev, &seqs, &from, batch_tokens).map_err(|e| format!("score: {e}"))?;
    let secs = t1.elapsed().as_secs_f64();
    let mut per: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
    for (w, v) in nll.iter().enumerate() {
        per[owner[w]].extend(v);
    }
    let f = std::fs::File::create(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut w = std::io::BufWriter::new(f);
    writeln!(w, "{SURPRISAL_HEADER}").map_err(|e| e.to_string())?;
    let mut total_bits = 0.0f64;
    for (ti, (id, _)) in texts.iter().enumerate() {
        if per[ti].len() != spans[ti].len() {
            return Err(format!("text {id}: {} tokens scored of {}", per[ti].len(), spans[ti].len()));
        }
        for (k, (&(s, e), &x)) in spans[ti].iter().zip(&per[ti]).enumerate() {
            let bits = x as f64 / std::f64::consts::LN_2;
            total_bits += bits;
            writeln!(w, "{id}\t{k}\t{s}\t{e}\t{bits:.6}").map_err(|e| e.to_string())?;
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    let bytes: usize = texts.iter().map(|(_, t)| t.len()).sum();
    println!(
        "surprisal_gpu: {} texts, {n_tok} tokens ({} sequences{}), {:.4} bits/byte; scored in {secs:.1}s ({:.0} tok/s), total {:.1}s -> {}",
        texts.len(),
        seqs.len(),
        ctx.map(|c| format!(", windows of {c}")).unwrap_or_default(),
        total_bits / bytes.max(1) as f64,
        n_tok as f64 / secs.max(1e-9),
        t0.elapsed().as_secs_f64(),
        out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_cover_every_target_once() {
        for len in [2usize, 5, 256, 257, 300, 1000] {
            for ctx in [None, Some(4), Some(64), Some(256)] {
                let mut next = 1; // the first target is ids[1]
                for (s, e, f) in windows(len, ctx) {
                    assert!(e - s >= 1 && f >= 1);
                    for p in f..(e - s) {
                        assert_eq!(s + p, next, "len {len} ctx {ctx:?}");
                        next += 1;
                    }
                }
                assert_eq!(next, len, "len {len} ctx {ctx:?}");
            }
        }
    }

    #[test]
    fn unescape_reverses_readtime_escape() {
        let esc = |s: &str| s.replace('\\', "\\\\").replace('\t', "\\t").replace('\n', "\\n");
        for s in ["plain", "a\tb", "line\nnext", "back\\slash", "\\t literal"] {
            assert_eq!(unescape(&esc(s)), s);
        }
    }
}
