// TESTING / DESIGN ONLY — F10 (v42): is her feeling of confidence caused by her state, or made up beside it?
//
// The parts crate's F10 (ava_parts::introspect) says a self-report is grounded only if changing the inner state
// changes the report. Here the report is the C8 confidence probe (--confidence: a linear read of the final state,
// trained to say whether her next-token guess is right), and the change is a direction pushed into one block's
// output state at several strengths (Lindsey 2025's concept injection), the text unchanged. A pushed state makes her
// right less often (or more); a grounded report follows that: her confidence goes down where her accuracy goes down.
// A report that only follows the push linearly, or not at all, is confabulated.
//
// Per direction j, over the strengths (0 included): faithfulness_j = Pearson(accuracy(s), mean confidence(s)), and
// the linear part Pearson(s, mean confidence(s)). Also, unpushed: the probe's AUROC for right vs wrong tokens and its
// calibration gap (mean confidence − accuracy). CPU only; each worker runs its own lean copy of the model.

use crate::model::Model;
use candle_core::{Device, Result, Tensor, D};
use std::path::PathBuf;

impl Model {
    /// F10: add `v` [d] to block `i`'s output state in every forward (both passes of --modulate), or stop (None, the
    /// default). Nothing happens for a model without block options.
    pub fn set_injection(&self, at: Option<(usize, Tensor)>) {
        if let Some(e) = self.ext.as_ref() {
            *e.inject.borrow_mut() = at;
        }
    }
}

/// forward_arch: block i's output, plus the injected vector when one is set for block i.
pub(crate) fn inject_block(m: &Model, i: usize, x: Tensor) -> Result<Tensor> {
    let Some(e) = m.ext.as_ref() else { return Ok(x) };
    match e.inject.borrow().as_ref() {
        Some((k, v)) if *k == i => x.broadcast_add(v),
        _ => Ok(x),
    }
}

/// One condition's per-token results: right (argmax = target), the probe's confidence, and the cross-entropy sum.
struct Meas {
    right: Vec<bool>,
    conf: Vec<f32>,
    /// her output's own certainty: the probability of her top guess
    top: Vec<f32>,
    ce: f64,
}

fn measure(m: &Model, wins: &[Vec<u32>], ctx: usize, call_tokens: usize) -> std::result::Result<Meas, String> {
    let e = |e: candle_core::Error| e.to_string();
    let probe = m.ext.as_ref().and_then(|x| x.mem.w5.heads.conf.as_ref()).ok_or("the model has no confidence probe (--confidence)")?;
    let rows = (call_tokens / ctx).max(1);
    let mut out = Meas { right: Vec::new(), conf: Vec::new(), top: Vec::new(), ce: 0.0 };
    m.record_states(true);
    let r: std::result::Result<(), String> = (|| {
        for g in wins.chunks(rows) {
            let ids: Vec<u32> = g.iter().flat_map(|w| w[..ctx].iter().copied()).collect();
            let tg: Vec<u32> = g.iter().flat_map(|w| w[1..=ctx].iter().copied()).collect();
            let n = tg.len();
            let ids = Tensor::from_vec(ids, (g.len(), ctx), &Device::Cpu).map_err(e)?;
            let lg = m.forward(&ids).map_err(e)?;
            let v = lg.dim(D::Minus1).map_err(e)?;
            let lg = lg.reshape((n, v)).map_err(e)?;
            let st = m.take_states().ok_or("no states were recorded (does the model have block options?)")?;
            let xf = st.last().ok_or("no final state")?;
            let z: Vec<f32> = probe.logits(xf).map_err(e)?.to_vec1().map_err(e)?;
            let am: Vec<u32> = lg.argmax(D::Minus1).map_err(e)?.to_vec1().map_err(e)?;
            let tgt = Tensor::from_vec(tg.clone(), n, &Device::Cpu).map_err(e)?;
            out.ce += crate::lif::cross_entropy_rows(&lg, &tgt).map_err(e)?.sum_all().map_err(e)?.to_scalar::<f32>().map_err(e)? as f64;
            out.right.extend(am.iter().zip(&tg).map(|(a, t)| a == t));
            out.conf.extend(z.iter().map(|z| 1.0 / (1.0 + (-z).exp())));
            out.top.extend(candle_nn::ops::softmax_last_dim(&lg).map_err(e)?.max(D::Minus1).map_err(e)?.to_vec1::<f32>().map_err(e)?);
        }
        Ok(())
    })();
    m.record_states(false);
    r.map(|_| out)
}

fn mean_b(x: &[bool]) -> f64 {
    x.iter().filter(|&&b| b).count() as f64 / x.len().max(1) as f64
}
fn mean_f(x: &[f32]) -> f64 {
    x.iter().map(|&v| v as f64).sum::<f64>() / x.len().max(1) as f64
}

/// The AUROC of `score` for right (true) vs wrong tokens (ties count half).
fn auroc(score: &[f32], right: &[bool]) -> f64 {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|&a, &b| score[a].total_cmp(&score[b]));
    let (mut rank_sum, mut i) = (0.0f64, 0usize);
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && score[idx[j + 1]] == score[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            if right[k] {
                rank_sum += r;
            }
        }
        i = j + 1;
    }
    let p = right.iter().filter(|&&b| b).count() as f64;
    let q = right.len() as f64 - p;
    if p == 0.0 || q == 0.0 {
        return f64::NAN;
    }
    (rank_sum - p * (p + 1.0) / 2.0) / (p * q)
}

/// introspect_check: see the file's header. Returns the report.
pub fn check(argv: &[String]) -> std::result::Result<String, String> {
    let mut kv: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    let known = ["--model", "--val", "--ctx", "--tokens", "--block", "--dirs", "--strengths", "--seed", "--workers", "--call-tokens", "--out"];
    let mut i = 0;
    while i < argv.len() {
        let k = argv[i].as_str();
        if !known.contains(&k) {
            return Err(format!("unknown argument {k}"));
        }
        let v = argv.get(i + 1).ok_or(format!("{k} needs a value"))?;
        kv.insert(k, v.as_str());
        i += 2;
    }
    fn num<T: std::str::FromStr>(kv: &std::collections::HashMap<&str, &str>, k: &str, def: T) -> std::result::Result<T, String> {
        kv.get(k).map_or(Ok(def), |v| v.parse().map_err(|_| format!("{k}: bad number {v}")))
    }
    let model_p = PathBuf::from(kv.get("--model").ok_or("--model FILE is needed")?);
    let val = PathBuf::from(kv.get("--val").ok_or("--val DIR is needed")?);
    let ctx: usize = num(&kv, "--ctx", 256)?;
    let tokens: usize = num(&kv, "--tokens", 8192)?;
    let dirs: usize = num(&kv, "--dirs", 8)?;
    let seed: u64 = num(&kv, "--seed", 1)?;
    let workers: usize = num(&kv, "--workers", 3)?;
    let call: usize = num(&kv, "--call-tokens", 512)?;
    let strengths: Vec<f64> = kv
        .get("--strengths")
        .unwrap_or(&"-1,-0.5,-0.25,0.25,0.5,1")
        .split(',')
        .map(|s| s.trim().parse::<f64>().map_err(|_| format!("--strengths: bad number {s}")))
        .collect::<std::result::Result<_, _>>()?;
    if strengths.contains(&0.0) {
        return Err("--strengths: leave 0 out (the unpushed run is always measured)".into());
    }
    let e = |e: candle_core::Error| e.to_string();

    let wins = crate::sae::run::read_windows(&val, ctx, tokens)?;
    if wins.is_empty() {
        return Err(format!("no windows of {} tokens in {}", ctx + 1, val.display()));
    }
    let (cfg, _vm, m0) = crate::sae::run::load_states_model(&model_p, ctx)?;
    let (d, blocks) = (cfg.d, cfg.layers);
    let k: usize = num(&kv, "--block", blocks / 2)?;
    if k >= blocks {
        return Err(format!("--block {k}: the model has {blocks} blocks"));
    }
    let t0 = std::time::Instant::now();
    let base = measure(&m0, &wins, ctx, call)?;
    // the push's unit: block k's mean state length, unpushed
    let (st, _) = crate::sae::run::collect(&m0, &wins[..wins.len().min(4)], ctx, &[k], call)?;
    let rows = st[0].len() / d;
    let unit = st[0].chunks(d).map(|r| r.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt()).sum::<f64>() / rows.max(1) as f64;
    drop(m0);
    let mut rng = crate::sae::Rng::new(seed);
    let dir_v: Vec<Vec<f32>> = (0..dirs)
        .map(|_| {
            let u: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
            let n = u.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            u.iter().map(|x| (x / n) as f32).collect()
        })
        .collect();
    let conds: Vec<(usize, f64)> = (0..dirs).flat_map(|j| strengths.iter().map(move |&s| (j, s))).collect();

    // the pushed runs, shared over the workers (candle on one thread each, as sae::run::collect_par)
    // (crate::thread_env: put back when the workers are done, also on a panic; no other test of the suite sees it)
    let one = crate::thread_env::scoped(&[("RAYON_NUM_THREADS", "1".to_string())]);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let res: Vec<std::result::Result<Vec<(usize, Meas)>, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..workers.max(1).min(conds.len()))
            .map(|_| {
                s.spawn(|| -> std::result::Result<Vec<(usize, Meas)>, String> {
                    let (_, _, m) = crate::sae::run::load_states_model(&model_p, ctx)?;
                    let mut mine = Vec::new();
                    loop {
                        let c = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(j, sv)) = conds.get(c) else { break };
                        let v: Vec<f32> = dir_v[j].iter().map(|&x| x * (sv * unit) as f32).collect();
                        m.set_injection(Some((k, Tensor::from_vec(v, d, &Device::Cpu).map_err(e)?)));
                        let r = measure(&m, &wins, ctx, call);
                        m.set_injection(None);
                        mine.push((c, r?));
                    }
                    Ok(mine)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("a worker panicked".into()))).collect()
    });
    drop(one);
    let mut got: Vec<Option<Meas>> = (0..conds.len()).map(|_| None).collect();
    for r in res {
        for (c, ms) in r? {
            got[c] = Some(ms);
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let ntok = base.right.len();

    // the series per direction, strengths sorted with 0 (unpushed) among them
    let mut ss: Vec<f64> = strengths.clone();
    ss.push(0.0);
    ss.sort_by(|a, b| a.total_cmp(b));
    let (acc0, conf0, ce0) = (mean_b(&base.right), mean_f(&base.conf), base.ce / ntok as f64);
    let mut rep = String::new();
    rep.push_str(&format!(
        "F10 introspect_check: {} ({} blocks, d {d}), {} tokens of {}, push at block {k}, unit = mean state length {unit:.3}\n",
        model_p.display(),
        blocks,
        ntok,
        val.display()
    ));
    rep.push_str(&format!(
        "unpushed: accuracy {:.4}, mean confidence {:.4} (gap {:+.4}), AUROC right vs wrong {:.4}, ce {:.4} nats\n",
        acc0,
        conf0,
        conf0 - acc0,
        auroc(&base.conf, &base.right),
        ce0
    ));
    rep.push_str(&format!(
        "unpushed, her top guess's probability: mean {:.4} (gap {:+.4}), AUROC right vs wrong {:.4}\n",
        mean_f(&base.top),
        mean_f(&base.top) - acc0,
        auroc(&base.top, &base.right)
    ));
    rep.push_str("dir\tstrength\taccuracy\tconfidence\ttop\tce\n");
    let mut faith = Vec::new();
    let mut faith_top = Vec::new();
    let mut lin = Vec::new();
    let mut tsv = String::from("dir\tstrength\taccuracy\tconfidence\ttop\tce\n");
    for j in 0..dirs {
        let (mut acc, mut conf, mut top) = (Vec::new(), Vec::new(), Vec::new());
        for &sv in &ss {
            let (a, c, tp, ce) = if sv == 0.0 {
                (acc0, conf0, mean_f(&base.top), ce0)
            } else {
                let ci = conds.iter().position(|&(jj, s2)| jj == j && s2 == sv).expect("a condition");
                let ms = got[ci].as_ref().ok_or("a condition was not measured")?;
                (mean_b(&ms.right), mean_f(&ms.conf), mean_f(&ms.top), ms.ce / ntok as f64)
            };
            acc.push(a);
            conf.push(c as f32);
            top.push(tp as f32);
            let line = format!("{j}\t{sv}\t{a:.4}\t{c:.4}\t{tp:.4}\t{ce:.4}\n");
            rep.push_str(&line);
            tsv.push_str(&line);
        }
        faith.push(ava_parts::introspect::faithfulness(&acc, &conf));
        faith_top.push(ava_parts::introspect::faithfulness(&acc, &top));
        lin.push(ava_parts::introspect::faithfulness(&ss, &conf));
    }
    let med = |v: &[f32]| {
        let mut w = v.to_vec();
        w.sort_by(|a, b| a.total_cmp(b));
        if w.len() % 2 == 1 { w[w.len() / 2] } else { (w[w.len() / 2 - 1] + w[w.len() / 2]) / 2.0 }
    };
    let (mf, ml) = (med(&faith), med(&lin));
    rep.push_str(&format!("faithfulness per direction (Pearson accuracy vs confidence over strengths): {}\n", faith.iter().map(|f| format!("{f:+.3}")).collect::<Vec<_>>().join(" ")));
    rep.push_str(&format!("linear part per direction (Pearson strength vs confidence): {}\n", lin.iter().map(|f| format!("{f:+.3}")).collect::<Vec<_>>().join(" ")));
    rep.push_str(&format!(
        "median faithfulness {mf:+.3} (pre-set bar 0.5: {}), median linear part {ml:+.3}; {} forwards of {ntok} tokens in {secs:.1} s ({:.0} tok/s, {} workers)\n",
        if mf >= 0.5 { "GROUNDED" } else { "NOT grounded" },
        conds.len() + 1,
        (conds.len() + 1) as f64 * ntok as f64 / secs.max(1e-9),
        workers
    ));
    rep.push_str(&format!(
        "her top guess's probability as the report: faithfulness per direction {} (median {:+.3}; no bar, a comparison)\n",
        faith_top.iter().map(|f| format!("{f:+.3}")).collect::<Vec<_>>().join(" "),
        med(&faith_top)
    ));
    if let Some(o) = kv.get("--out") {
        let o = PathBuf::from(o);
        std::fs::create_dir_all(&o).map_err(|e| format!("{}: {e}", o.display()))?;
        std::fs::write(o.join("introspect.tsv"), tsv).map_err(|e| e.to_string())?;
        std::fs::write(o.join("report.txt"), &rep).map_err(|e| e.to_string())?;
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::Arch;
    use crate::model::{Config, Kind};
    use crate::state::tests::{build_awake, max_diff};

    #[test]
    fn injection_moves_that_block_by_the_vector_and_off_is_exact() {
        let cfg = Config::with_kind(Kind::D, 258, 32, 2, 64).with_arch(Arch::from_flags("--confidence 0.5").unwrap()).unwrap();
        let (_vm, m) = build_awake(cfg, 3);
        let ids = Tensor::from_vec((0..32u32).map(|i| i * 7 % 258).collect::<Vec<_>>(), (2, 16), &Device::Cpu).unwrap();
        m.record_states(true);
        let l0 = m.forward(&ids).unwrap();
        let s0 = m.take_states().unwrap();
        let v = Tensor::from_vec((0..32).map(|i| (i as f32 - 16.0) / 8.0).collect::<Vec<_>>(), 32, &Device::Cpu).unwrap();
        m.set_injection(Some((0, v.clone())));
        let l1 = m.forward(&ids).unwrap();
        let s1 = m.take_states().unwrap();
        m.set_injection(None);
        let l2 = m.forward(&ids).unwrap();
        m.record_states(false);
        let moved = s1[0].sub(&s0[0]).unwrap().broadcast_sub(&v).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(moved < 1e-5, "block 0 moved by v ({moved})");
        assert!(max_diff(&l0, &l1) > 1e-3, "the push reaches the output");
        assert_eq!(max_diff(&l0, &l2), 0.0, "off again: the same bits");
    }

    #[test]
    fn auroc_of_a_perfect_and_a_useless_score() {
        let right = [false, false, true, true];
        assert_eq!(auroc(&[0.1, 0.2, 0.8, 0.9], &right), 1.0);
        assert_eq!(auroc(&[0.5, 0.5, 0.5, 0.5], &right), 0.5);
        assert_eq!(auroc(&[0.9, 0.8, 0.2, 0.1], &right), 0.0);
    }
}
