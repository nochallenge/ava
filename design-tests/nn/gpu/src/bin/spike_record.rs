// TESTING / DESIGN ONLY — spike_record: record Ava's real neurons while she reads, for the brain viewer.
//
// spike_record --model FILE --text "..." [--text "..."]... [--texts FILE] --out FILE.json [--html OUT.html]
//              [--cpu] [--bpe FILE] [--top 5] [--etop 3] [--arch D] [--name LABEL] [--threads 6] [--no-bos] [--check]
//
// Every text is read in one forward pass (`Model::forward_trace`, the same arithmetic as `forward`; texts are
// batched, padded at the end with PAD, which a causal model never reads back). Each starts with EOS (her "a new
// text begins"; --no-bos drops it) and gets its glance length marks when the model has them. Recorded per token
// and per block, as lists of unit indices (never dense floats):
//   s   the LIF memory's spikes: which of the d neurons fired (lif_scan's 0/1 output)
//   w   the kWTA winners of the MLP's 4d hidden units (the k largest of h·W1, kwta_mask)
//   a   one digit per winner, 0..9: its hidden activity after ReLU over the block's largest at that token
//       (0 = a winner whose value was not positive, so ReLU passed nothing)
//   m   the fast-weight (Hebbian) memory's read: RMS of what it added to the residual stream
//   mw  its write strength per head (beta in (0, 2));  mk  its keep fraction per head (alpha in (0, 1))
// and per token: e = the engram block's strongest reads of earlier tokens ([position, weight], weights averaged
// over its heads; es = its weight on the token itself; er = RMS of the engram read), p = her top predictions for
// the next token, nx = the probability and rank she gave the token that really came next.
// The JSON holds only the model's file name (no paths). --html writes the viewer (design-tests/nn/viz/ava_brain.html,
// built in) with the recording embedded: one file that opens by double-click. --check also runs `forward` and
// reports the largest logit difference (0 expected). Threads: RAYON_NUM_THREADS, else --threads (default 6).

use ava_data::bpe::{Bpe, EOS, PAD};
use ava_data::numbers::{insert_marks, is_mark, MARK_BASE};
use ava_gpu::model::Kind;
use candle_core::{DType, Tensor};
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: spike_record --model FILE --text \"...\" [--text \"...\"]... [--texts FILE] --out FILE.json [--html OUT.html]\n\
    \x20                   [--cpu] [--bpe FILE] [--top 5] [--etop 3] [--arch D] [--name LABEL] [--threads 6] [--no-bos] [--check]";

/// The viewer, built into the binary; `DATA_SLOT` is where the recording goes.
const VIEWER: &str = include_str!("../../../viz/ava_brain.html");
const DATA_SLOT: &str = "<script id=\"ava-data\" type=\"application/json\">null</script>";

fn main() {
    if let Err(e) = run() {
        eprintln!("spike_record: {e}");
        std::process::exit(1);
    }
}

fn r4(x: f32) -> f64 {
    ((x as f64) * 1e4).round() / 1e4
}

fn r3(x: f32) -> f64 {
    ((x as f64) * 1e3).round() / 1e3
}

/// A token as text for the viewer: EOS as a pilcrow, a glance mark as its length.
fn tok_str(bpe: &Bpe, id: u32) -> String {
    if id == EOS {
        "\u{b6}".into()
    } else if is_mark(id) {
        let l = id - MARK_BASE + 1;
        if l >= 16 { "\u{27e8}16+ digits\u{27e9}".into() } else { format!("\u{27e8}{l} digit{}\u{27e9}", if l == 1 { "" } else { "s" }) }
    } else {
        bpe.decode(&[id])
    }
}

/// 0 = a word piece, 1 = EOS, 2 = a glance length mark.
fn tok_kind(id: u32) -> u8 {
    if id == EOS {
        1
    } else if is_mark(id) {
        2
    } else {
        0
    }
}

fn rms(row: &[f32]) -> f32 {
    (row.iter().map(|v| v * v).sum::<f32>() / row.len().max(1) as f32).sqrt()
}

fn rows(t: &Tensor) -> Result<Vec<Vec<f32>>, String> {
    t.to_dtype(DType::F32).and_then(|t| t.to_vec2::<f32>()).map_err(|e| format!("to host: {e}"))
}

fn run() -> Result<(), String> {
    let (mut model_path, mut out, mut html, mut texts_file): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None, None);
    let mut texts: Vec<String> = Vec::new();
    let (mut cpu, mut bos, mut check) = (false, true, false);
    let (mut top, mut etop, mut threads) = (5usize, 3usize, 6usize);
    let mut kind = Kind::D;
    let mut name: Option<String> = None;
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        match key {
            "-h" | "--help" => return Err(USAGE.into()),
            "--cpu" | "--no-bos" | "--check" => {
                match key {
                    "--cpu" => cpu = true,
                    "--no-bos" => bos = false,
                    _ => check = true,
                }
                i += 1;
                continue;
            }
            _ => {}
        }
        let val = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.clone();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
        match key {
            "--model" => model_path = Some(PathBuf::from(val)),
            "--text" => texts.push(val),
            "--texts" => texts_file = Some(PathBuf::from(val)),
            "--out" => out = Some(PathBuf::from(val)),
            "--html" => html = Some(PathBuf::from(val)),
            "--bpe" => bpe_path = PathBuf::from(val),
            "--top" => top = num(&val)?.max(1),
            "--etop" => etop = num(&val)?,
            "--threads" => threads = num(&val)?.max(1),
            "--arch" => kind = Kind::parse(&val)?,
            "--name" => name = Some(val),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    if let Some(p) = &texts_file {
        let s = std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))?;
        texts.extend(s.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from));
    }
    let model_path = model_path.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    if texts.is_empty() {
        return Err(format!("give at least one --text (or --texts FILE)\n{USAGE}"));
    }
    if out.is_none() && html.is_none() {
        return Err(format!("give --out FILE.json and/or --html FILE.html\n{USAGE}"));
    }
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }

    let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let t_load = Instant::now();
    let (cfg, _varmap, model) = ava_gpu::ckpt::load_auto(&model_path, kind, 4096, &dev)?;
    model.set_track_activity(false);
    let load_s = t_load.elapsed().as_secs_f64();
    let file_name = model_path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    let name = name.unwrap_or_else(|| file_name.clone());

    // tokenize: [EOS] text, then the glance marks
    let seqs: Vec<Vec<u32>> = texts
        .iter()
        .map(|t| {
            let mut ids = if bos { vec![EOS] } else { Vec::new() };
            ids.extend(bpe.encode(t));
            if cfg.glance {
                insert_marks(&ids)
            } else {
                ids
            }
        })
        .collect();
    if seqs.iter().any(|s| s.is_empty()) {
        return Err("a text has no tokens".into());
    }
    let (b, tmax) = (seqs.len(), seqs.iter().map(Vec::len).max().unwrap_or(1));
    let flat: Vec<u32> = seqs.iter().flat_map(|s| s.iter().copied().chain(std::iter::repeat(PAD).take(tmax - s.len()))).collect();
    let ids = Tensor::from_vec(flat, (b, tmax), &dev).map_err(|e| format!("ids: {e}"))?;
    let real: usize = seqs.iter().map(Vec::len).sum();

    let t0 = Instant::now();
    let (logits, trace) = model.forward_trace(&ids).map_err(|e| format!("forward_trace: {e}"))?;
    let lg = rows(&logits)?; // also waits for the device
    let fwd_s = t0.elapsed().as_secs_f64();
    if check {
        let want = rows(&model.forward(&ids).map_err(|e| format!("forward: {e}"))?)?;
        let diff = lg.iter().flatten().zip(want.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("check: largest |forward_trace - forward| logit difference = {diff:e}");
        if diff != 0.0 {
            return Err(format!("forward_trace differs from forward by {diff:e}"));
        }
    }

    // to the host
    let t1 = Instant::now();
    let (d, hid, k, nh) = (cfg.d, cfg.hidden(), cfg.kwta_k(), cfg.heads());
    struct L {
        s: Vec<Vec<f32>>,
        w: Vec<Vec<f32>>,
        a: Vec<Vec<f32>>,
        m: Option<Vec<Vec<f32>>>,
        mw: Option<Vec<Vec<f32>>>,
        mk: Option<Vec<Vec<f32>>>,
    }
    let mut ls = Vec::with_capacity(cfg.layers);
    let mut eng: Option<(usize, Vec<f32>, Vec<Vec<f32>>)> = None; // (block, attn [b, nh, t, t] flat, read rows)
    for (li, l) in trace.layers.iter().enumerate() {
        let opt = |t: &Option<Tensor>| t.as_ref().map(rows).transpose();
        ls.push(L { s: rows(&l.spikes)?, w: rows(&l.winners)?, a: rows(&l.hidden)?, m: opt(&l.hebb_read)?, mw: opt(&l.hebb_write)?, mk: opt(&l.hebb_keep)? });
        if let (Some(at), Some(rd)) = (&l.engram_attn, &l.engram_read) {
            let flat = at.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| format!("engram: {e}"))?;
            eng = Some((li, flat, rows(rd)?));
        }
    }

    let mut recs = Vec::with_capacity(b);
    let mut rate_sum = vec![0f64; cfg.layers];
    let mut win_counts: Vec<usize> = Vec::new();
    for (bi, seq) in seqs.iter().enumerate() {
        let t = seq.len();
        let mut steps = Vec::with_capacity(t);
        for pos in 0..t {
            let r = bi * tmax + pos;
            let mut layers = Vec::with_capacity(cfg.layers);
            for (li, l) in ls.iter().enumerate() {
                let s: Vec<u32> = l.s[r].iter().enumerate().filter(|(_, &v)| v > 0.5).map(|(j, _)| j as u32).collect();
                rate_sum[li] += s.len() as f64 / d as f64;
                let w: Vec<u32> = l.w[r].iter().enumerate().filter(|(_, &v)| v > 0.5).map(|(j, _)| j as u32).collect();
                win_counts.push(w.len());
                let amax = w.iter().map(|&j| l.a[r][j as usize]).fold(0f32, f32::max);
                let a: String = w
                    .iter()
                    .map(|&j| {
                        let v = l.a[r][j as usize];
                        let q = if v > 0.0 && amax > 0.0 { (1.0 + (8.0 * v / amax).round()).min(9.0) as u8 } else { 0 };
                        (b'0' + q) as char
                    })
                    .collect();
                let mut o = json!({ "s": s, "w": w, "a": a });
                if let Some(m) = &l.m {
                    o["m"] = json!(r4(rms(&m[r])));
                }
                if let Some(mw) = &l.mw {
                    o["mw"] = json!(mw[r].iter().map(|&x| r3(x)).collect::<Vec<_>>());
                }
                if let Some(mk) = &l.mk {
                    o["mk"] = json!(mk[r].iter().map(|&x| r4(x)).collect::<Vec<_>>());
                }
                layers.push(o);
            }
            let mut step = json!({ "L": layers });
            if let Some((_, at, rd)) = &eng {
                // mean over heads of row `pos` of [bi, h, pos, ..]
                let mut wsum = vec![0f32; pos + 1];
                for h in 0..nh {
                    let base = ((bi * nh + h) * tmax + pos) * tmax;
                    for (j, ws) in wsum.iter_mut().enumerate() {
                        *ws += at[base + j] / nh as f32;
                    }
                }
                let mut earlier: Vec<(usize, f32)> = wsum[..pos].iter().copied().enumerate().collect();
                earlier.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
                earlier.truncate(etop);
                step["e"] = json!(earlier.iter().map(|&(j, w)| json!([j, r4(w)])).collect::<Vec<_>>());
                step["es"] = json!(r4(wsum[pos]));
                step["er"] = json!(r4(rms(&rd[r])));
            }
            // her prediction for the next token
            let row = &lg[r];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let z: f64 = row.iter().map(|&x| ((x - mx) as f64).exp()).sum();
            let p = |id: usize| (((row[id] - mx) as f64).exp() / z) as f32;
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.select_nth_unstable_by(top.min(row.len()) - 1, |&x, &y| row[y].partial_cmp(&row[x]).unwrap_or(std::cmp::Ordering::Equal));
            idx.truncate(top);
            idx.sort_by(|&x, &y| row[y].partial_cmp(&row[x]).unwrap_or(std::cmp::Ordering::Equal));
            step["p"] = json!(idx.iter().map(|&id| json!([tok_str(&bpe, id as u32), r4(p(id)), id])).collect::<Vec<_>>());
            if pos + 1 < t {
                let nx = seq[pos + 1] as usize;
                let rank = 1 + row.iter().filter(|&&x| x > row[nx]).count();
                step["nx"] = json!([r4(p(nx)), rank]);
            }
            steps.push(step);
        }
        recs.push(json!({
            "text": texts[bi],
            "ids": seq,
            "tokens": seq.iter().map(|&id| tok_str(&bpe, id)).collect::<Vec<_>>(),
            "kinds": seq.iter().map(|&id| tok_kind(id)).collect::<Vec<_>>(),
            "steps": steps,
        }));
    }
    let host_s = t1.elapsed().as_secs_f64();

    let doc = json!({
        "format": "ava-spikes/1",
        "model": {
            "name": name, "file": file_name, "kind": cfg.kind.name(), "d": d, "layers": cfg.layers, "hidden": hid, "k": k,
            "heads": nh, "engram": cfg.engram, "glance": cfg.glance, "mag": cfg.mag, "params": cfg.weight_counts().0,
        },
        "units": { "spike": d, "mlp": hid },
        "fields": {
            "s": "LIF spikes: indices of the neurons that fired",
            "w": "kWTA winners: indices of the k MLP hidden units that won",
            "a": "per winner, its ReLU activity 0..9 over the block's largest at that token",
            "m": "Hebbian fast-weight memory: RMS of its read added to the stream",
            "mw": "memory write strength per head (0..2)", "mk": "memory keep fraction per head (0..1)",
            "e": "engram: strongest earlier tokens read [position, weight averaged over heads]",
            "p": "top next-token predictions [text, probability, id]", "nx": "the real next token: [probability, rank]",
        },
        "device": if dev.is_cuda() { "cuda" } else { "cpu" },
        "recordings": recs,
    });
    let mut js = serde_json::to_string(&doc).map_err(|e| format!("json: {e}"))?;
    js = js.replace("</", "<\\/"); // safe inside a <script> element; still the same JSON
    if let Some(p) = &out {
        std::fs::write(p, &js).map_err(|e| format!("write {}: {e}", p.display()))?;
        println!("wrote {} ({:.1} KB)", p.display(), js.len() as f64 / 1024.0);
    }
    if let Some(p) = &html {
        if !VIEWER.contains(DATA_SLOT) {
            return Err("the built-in viewer has no data slot".into());
        }
        let page = VIEWER.replacen(DATA_SLOT, &format!("<script id=\"ava-data\" type=\"application/json\">{js}</script>"), 1);
        std::fs::write(p, page).map_err(|e| format!("write {}: {e}", p.display()))?;
        println!("wrote {}", p.display());
    }

    let rates: Vec<String> = rate_sum.iter().map(|s| format!("{:.3}", s / real as f64)).collect();
    let (wmin, wmax) = (win_counts.iter().min().copied().unwrap_or(0), win_counts.iter().max().copied().unwrap_or(0));
    println!("{} text(s), {real} tokens ({} x {tmax} padded); model {} ({}, d {d}, {} blocks, k {k} of {hid}, engram {:?})", b, b, name, cfg.kind.name(), cfg.layers, cfg.engram);
    println!("spike rate per block: {}", rates.join(" "));
    println!("kWTA winners per block-token: min {wmin}, max {wmax} (k = {k})");
    println!(
        "speed: load {load_s:.2}s, recording pass {fwd_s:.3}s ({:.0} tok/s), extraction {host_s:.3}s; {} threads, {}",
        real as f64 / fwd_s,
        std::env::var("RAYON_NUM_THREADS").unwrap_or_default(),
        if dev.is_cuda() { "cuda" } else { "cpu" }
    );
    Ok(())
}
