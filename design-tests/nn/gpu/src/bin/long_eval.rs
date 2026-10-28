// TESTING / DESIGN ONLY — long_eval: v40's long-text test. Does her prediction keep improving deep into a book?
//
// long_eval --model FILE (--shards DIR | --text DIR) [--arch D] [--books 100] [--len 16384] [--cut 256]
//           [--buckets 256,1024,4096,16384] [--baseline FILE] [--resamples 10000] [--out FILE] [--bpe FILE]
//
// * Books: --shards DIR (token shards; the stream is split at EOS and every document of at least --len tokens is
//   a book) or --text DIR (one .txt file per book, sorted by name; the start of each is tokenised, with length
//   marks for a glance model). The first --books books, the first --len tokens of each, read from the book's start
//   (an EOS before the first token, as in training).
// * Conditions (one model, as v40's one-model rule asks):
//     full    every part on, the whole --len tokens in one pass (D has no position table and no window);
//     cut     the same tokens in independent windows of --cut tokens (her memory is wiped every --cut tokens);
//     no-eng  the engram switched off (only for a model with one);
//     base    --baseline FILE, full (e.g. v39's ava-num on the same books).
// * Bits per byte of every position bucket [0, b1), [b1, b2), ... (the bucket bounds are --buckets), with bytes
//   from the tokenizer (length marks are 0 bytes). Differences between conditions get a 95% interval from a
//   bootstrap over books (--resamples, SplitMix64 seed 1): BPB is a ratio of sums, recomputed per resample.
// * Last line: RESULT with every condition's BPB of the last bucket.

use ava_data::bpe::{Bpe, EOS};
use ava_data::numbers::insert_marks;
use ava_data::rng::SplitMix64;
use ava_data::shards::ShardSet;
use ava_gpu::eval::byte_lengths_n;
use ava_gpu::model::{Kind, Model};
use candle_core::{Device, Tensor, D};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: long_eval --model FILE (--shards DIR | --text DIR) [--arch D] [--books 100] [--len 16384] [--cut 256]\n\
    \x20                [--buckets 256,1024,4096,16384] [--baseline FILE] [--resamples 10000] [--out FILE] [--bpe FILE]";

fn main() {
    if let Err(e) = run() {
        eprintln!("long_eval: {e}");
        std::process::exit(1);
    }
}

/// Books from token shards: the stream split at EOS, documents of at least `len` tokens.
fn books_from_shards(dir: &Path, len: usize, n: usize) -> Result<Vec<Vec<u32>>, String> {
    let set = ShardSet::open(dir).map_err(|e| format!("open shards {}: {e}", dir.display()))?;
    let (mut out, mut cur) = (Vec::new(), Vec::new());
    'outer: for s in 0..set.len() {
        let toks = set.load_shard(s).map_err(|e| format!("load shard {s}: {e}"))?;
        for i in 0..toks.len() {
            let t = toks.get(i);
            if t == EOS {
                if cur.len() >= len {
                    cur.truncate(len);
                    out.push(std::mem::take(&mut cur));
                    if out.len() == n {
                        break 'outer;
                    }
                }
                cur.clear();
            } else {
                cur.push(t);
            }
        }
    }
    Ok(out)
}

/// Books from text files: the start of each (enough characters for `len` tokens), tokenised.
fn books_from_text(dir: &Path, len: usize, n: usize, bpe: &Bpe, glance: bool) -> Result<Vec<Vec<u32>>, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    files.sort();
    let mut out = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).map_err(|e| format!("read {}: {e}", f.display()))?;
        let mut take = (len * 8).min(text.len());
        while !text.is_char_boundary(take) {
            take -= 1;
        }
        let mut toks = bpe.encode(&text[..take]);
        if glance {
            toks = insert_marks(&toks);
        }
        if toks.len() >= len {
            toks.truncate(len);
            out.push(toks);
            if out.len() == n {
                break;
            }
        }
    }
    Ok(out)
}

/// NLL (nats) of every target of `rows` (each [EOS] + book), read in windows of `win` tokens; [books][len].
fn token_nll(model: &Model, dev: &Device, books: &[Vec<u32>], win: usize) -> Result<Vec<Vec<f32>>, String> {
    let mut out = Vec::with_capacity(books.len());
    for b in books {
        let mut seq = Vec::with_capacity(b.len() + 1);
        seq.push(EOS);
        seq.extend_from_slice(b);
        let len = b.len();
        let w = win.min(len);
        let rows = len.div_ceil(w);
        // rows of w inputs (the last row padded by repeating the final token; its extra targets are dropped)
        let (mut ids, mut tg) = (Vec::with_capacity(rows * w), Vec::with_capacity(rows * w));
        for r in 0..rows {
            for p in 0..w {
                let k = (r * w + p).min(len - 1);
                ids.push(seq[k]);
                tg.push(seq[k + 1]);
            }
        }
        // a few rows at a time keeps the logits (rows · w · vocab floats) small
        let per = (32768 / w).max(1);
        let mut nll = Vec::with_capacity(rows * w);
        for r0 in (0..rows).step_by(per) {
            let rn = per.min(rows - r0);
            let idt = Tensor::from_vec(ids[r0 * w..(r0 + rn) * w].to_vec(), (rn, w), dev).map_err(|e| e.to_string())?;
            let logits = model.forward(&idt).map_err(|e| format!("forward: {e}"))?;
            let lp = candle_nn::ops::log_softmax(&logits, D::Minus1).map_err(|e| e.to_string())?;
            drop(logits);
            let tgt = Tensor::from_vec(tg[r0 * w..(r0 + rn) * w].to_vec(), (rn * w, 1), dev).map_err(|e| e.to_string())?;
            let v: Vec<f32> = lp.gather(&tgt, 1).and_then(|x| x.squeeze(1)).and_then(|x| x.neg()).and_then(|x| x.to_vec1()).map_err(|e| e.to_string())?;
            nll.extend(v);
        }
        nll.truncate(len);
        out.push(nll);
    }
    Ok(out)
}

/// Per book and bucket: (nats, bytes).
fn bucketed(nll: &[Vec<f32>], books: &[Vec<u32>], bounds: &[usize], bytes: &[u32]) -> Vec<Vec<(f64, f64)>> {
    nll.iter()
        .zip(books)
        .map(|(n, b)| {
            let mut v = vec![(0.0, 0.0); bounds.len()];
            for (p, (&x, &t)) in n.iter().zip(b).enumerate() {
                let Some(k) = bounds.iter().position(|&e| p < e) else { continue };
                v[k].0 += x as f64;
                v[k].1 += bytes.get(t as usize).copied().unwrap_or(0) as f64;
            }
            v
        })
        .collect()
}

fn bpb(per: &[Vec<(f64, f64)>], k: usize, pick: &[usize]) -> f64 {
    let (n, b) = pick.iter().fold((0.0, 0.0), |acc, &i| (acc.0 + per[i][k].0, acc.1 + per[i][k].1));
    n / std::f64::consts::LN_2 / b.max(1.0)
}

/// Bootstrap over books of BPB(a) − BPB(b) in bucket k: (mean, lo, hi).
fn diff_ci(a: &[Vec<(f64, f64)>], b: &[Vec<(f64, f64)>], k: usize, resamples: usize) -> (f64, f64, f64) {
    let n = a.len();
    let all: Vec<usize> = (0..n).collect();
    let point = bpb(a, k, &all) - bpb(b, k, &all);
    let mut rng = SplitMix64::new(1);
    let mut ds: Vec<f64> = (0..resamples)
        .map(|_| {
            let pick: Vec<usize> = (0..n).map(|_| rng.below(n as u64) as usize).collect();
            bpb(a, k, &pick) - bpb(b, k, &pick)
        })
        .collect();
    ds.sort_by(f64::total_cmp);
    let q = |p: f64| ds[((p * (resamples - 1) as f64).round() as usize).min(resamples - 1)];
    (point, q(0.025), q(0.975))
}

fn run() -> Result<(), String> {
    let (mut model_path, mut baseline, mut shards, mut text, mut out): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None, None, None);
    let (mut kind, mut n_books, mut len, mut cut, mut resamples) = (Kind::D, 100usize, 16384usize, 256usize, 10000usize);
    let mut bounds = vec![256usize, 1024, 4096, 16384];
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
        match key {
            "--model" => model_path = Some(v.into()),
            "--baseline" => baseline = Some(v.into()),
            "--shards" => shards = Some(v.into()),
            "--text" => text = Some(v.into()),
            "--arch" => kind = Kind::parse(v)?,
            "--books" => n_books = num(v)?.max(1),
            "--len" => len = num(v)?.max(2),
            "--cut" => cut = num(v)?.max(1),
            "--buckets" => bounds = v.split(',').map(|x| num(x.trim())).collect::<Result<_, _>>()?,
            "--resamples" => resamples = num(v)?.max(1),
            "--out" => out = Some(v.into()),
            "--bpe" => bpe_path = v.into(),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let model_path = model_path.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    if bounds.windows(2).any(|w| w[0] >= w[1]) || bounds.last() != Some(&len) {
        return Err(format!("--buckets must rise and end at --len ({len})"));
    }
    let dev = ava_gpu::device(false).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let (cfg, _varmap, model) = ava_gpu::ckpt::load_auto(&model_path, kind, 256, &dev)?;
    model.set_track_activity(false);
    model.set_inference(std::env::var("AVA_FAST_INFER").map_or(true, |v| v != "0"));
    let books = match (&shards, &text) {
        (Some(d), None) => books_from_shards(d, len, n_books)?,
        (None, Some(d)) => books_from_text(d, len, n_books, &bpe, cfg.glance)?,
        _ => return Err(format!("give exactly one of --shards and --text\n{USAGE}")),
    };
    if books.is_empty() {
        return Err(format!("no book of {len}+ tokens found"));
    }
    if books.iter().flatten().any(|&t| t as usize >= cfg.vocab) {
        return Err(format!("the books hold token ids >= the model's vocab {}", cfg.vocab));
    }
    let bytes = byte_lengths_n(&bpe, cfg.vocab);
    println!("{} books x {len} tokens; model {} (glance {}, mag {}, engram {:?}); buckets {bounds:?}; cut {cut}", books.len(), model_path.display(), cfg.glance, cfg.mag, cfg.engram);

    let mut conds: Vec<(String, Vec<Vec<(f64, f64)>>)> = Vec::new();
    let mut run_cond = |name: &str, m: &Model, win: usize| -> Result<(), String> {
        let t0 = Instant::now();
        let nll = token_nll(m, &dev, &books, win)?;
        let secs = t0.elapsed().as_secs_f64();
        println!("  {name}: {:.1}s ({:.0} tokens/s)", secs, (books.len() * len) as f64 / secs);
        conds.push((name.to_string(), bucketed(&nll, &books, &bounds, &bytes)));
        Ok(())
    };
    run_cond("full", &model, len)?;
    run_cond("cut", &model, cut)?;
    if cfg.engram.is_some() {
        model.set_lesion(false, false, true);
        run_cond("no-eng", &model, len)?;
        model.set_lesion(false, false, false);
    }
    drop(model);
    if let Some(bp) = &baseline {
        let (_, _vb, bm) = ava_gpu::ckpt::load_auto(bp, kind, 256, &dev)?;
        bm.set_track_activity(false);
        bm.set_inference(std::env::var("AVA_FAST_INFER").map_or(true, |v| v != "0"));
        run_cond("base", &bm, len)?;
    }
    let all: Vec<usize> = (0..books.len()).collect();
    let mut lo_b = 0usize;
    let mut table = String::from("bucket");
    for (name, _) in &conds {
        table += &format!("\t{name}");
    }
    table.push('\n');
    for (k, &hi_b) in bounds.iter().enumerate() {
        table += &format!("{lo_b}-{hi_b}");
        for (_, per) in &conds {
            table += &format!("\t{:.4}", bpb(per, k, &all));
        }
        table.push('\n');
        lo_b = hi_b;
    }
    println!("bits per byte by position:\n{table}");
    let full = &conds[0].1;
    for (name, per) in conds.iter().skip(1) {
        let parts: Vec<String> = (0..bounds.len())
            .map(|k| {
                let (m, lo, hi) = diff_ci(full, per, k, resamples);
                format!("{:+.4} [{:+.4}, {:+.4}]", m, lo, hi)
            })
            .collect();
        println!("full - {name} by bucket: {}", parts.join("; "));
    }
    if let Some(o) = &out {
        let mut f = std::fs::File::create(o).map_err(|e| format!("create {}: {e}", o.display()))?;
        f.write_all(table.as_bytes()).map_err(|e| e.to_string())?;
    }
    let last = bounds.len() - 1;
    let res: Vec<String> = conds.iter().map(|(n, per)| format!("{n}={:.6}", bpb(per, last, &all))).collect();
    println!("RESULT books={} last_bucket {}", books.len(), res.join(" "));
    Ok(())
}
