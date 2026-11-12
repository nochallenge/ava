// TESTING / DESIGN ONLY — v47 gradient check (runs/v47/prereg_fast_winners.md, part G): D with Stage 3's winning block
// options on the hand-written step (src/model/fwin.rs) against loss + candle's autograd, same model and batch.
// CUDA only (AVA_GPU_TEST_CUDA=1); tiny models, a few seconds of GPU.
//
// * G1: D d128 L3 ctx 64 batch 4, all nine winners + the glance (head size 64), weights that start at 0 moved off it
// * G2: D d64 L2 head size 32, --kwta 0.25 --mlp-act relu2 --lif-scale --value-residual
// * G3: G1's model with a 0/1 weight per target (the weighted mean), and signed raw weights (sum(w·loss))
// f32: loss within 1e-4, every parameter cos > 0.9999 and |g_fused - g_auto| / |g_auto| < 1e-2.
// bf16: loss within 1e-2; gradient agreement printed only (rounding flips spikes).

use ava_gpu::arch::Arch;
use ava_gpu::model::{fwin, Config, Kind, Mm, Model, VOCAB};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};

fn cuda() -> Option<Device> {
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v == "1").unwrap_or(false) {
        Some(Device::new_cuda(0).expect("CUDA device 0"))
    } else {
        None
    }
}

fn arch_of(flags: &str) -> Arch {
    let args: Vec<String> = flags.split_whitespace().map(String::from).collect();
    let mut a = Arch::default();
    let mut i = 0;
    while i < args.len() {
        i += a.take_flag(&args, i).unwrap().unwrap_or_else(|| panic!("not a block flag: {}", args[i]));
    }
    a
}

const WINNERS: &str =
    "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix --lif-init logspaced --lif-scale";

fn noise(dev: &Device, n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let _ = dev;
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * amp
        })
        .collect()
}

fn vars(varmap: &VarMap) -> Vec<(String, Var)> {
    let mut v: Vec<(String, Var)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    v.sort_by(|x, y| x.0.cmp(&y.0));
    v
}

/// Moves the weights that start at 0 (the dense weights, p0/p1, the place table, the gates' wa/wb/ba/bb) off it.
fn perturb(varmap: &VarMap, dev: &Device) {
    for (k, (name, var)) in vars(varmap).into_iter().enumerate() {
        let leaf = name.rsplit('.').next().unwrap_or(&name);
        // v47 Amendment 5: the engram's output projection starts at 0 (the magnitude code's mag_w too)
        let leaf = if leaf == "wo" && name.contains(".eng.") { "eng_wo" } else { leaf };
        let amp = match leaf {
            "dense" => 0.1,
            "vres" => 0.3,
            "place_emb" => 0.05,
            "wa" | "wb" | "wa2" | "wb2" | "we" | "eng_wo" | "mag_w" => 0.05,
            // SLOW: per-neuron thresholds and ALIF gains moved around their start (1)
            "theta" | "again" => {
                let t = var.as_tensor();
                let v = Tensor::from_vec(noise(dev, t.elem_count(), 1000 + k as u64, 0.3), t.shape(), dev).unwrap();
                var.set(&t.add(&v).unwrap()).unwrap();
                continue;
            }
            "ba" | "bb" => 0.5,
            _ => continue,
        };
        let t = var.as_tensor();
        let v = Tensor::from_vec(noise(dev, t.elem_count(), 1000 + k as u64, amp), t.shape(), dev).unwrap();
        var.set(&v).unwrap();
    }
}

fn model(dev: &Device, cfg: Config, seed: u64) -> (VarMap, Model) {
    dev.set_seed(seed).unwrap(); // the same init whatever ran before
    let varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, dev)).unwrap();
    perturb(&varmap, dev);
    (varmap, model)
}

fn glance_stream(len: usize, seed: u64) -> Vec<u32> {
    use ava_data::numbers::insert_marks;
    let mut rng = ava_data::rng::SplitMix64::new(seed);
    let mut out = Vec::with_capacity(len + 32);
    while out.len() < len {
        match rng.below(10) {
            0..=4 => out.push(300 + rng.below(256) as u32),
            5..=7 => {
                let l = 1 + rng.below(6) as usize;
                let digits: Vec<u32> = (0..l).map(|_| 48 + rng.below(10) as u32).collect();
                out.extend(insert_marks(&digits));
            }
            8 => out.push(48 + rng.below(10) as u32),
            _ => {
                out.push(4096 + rng.below(16) as u32);
                out.extend((0..1 + rng.below(4)).map(|_| 48 + rng.below(10) as u32));
            }
        }
    }
    out.truncate(len);
    out
}

fn batch(dev: &Device, b: usize, t: usize, seed: u64) -> (Tensor, Tensor) {
    let (mut ids, mut tg) = (Vec::with_capacity(b * t), Vec::with_capacity(b * t));
    for r in 0..b {
        let s = glance_stream(t + 1, seed + r as u64);
        ids.extend_from_slice(&s[..t]);
        tg.extend_from_slice(&s[1..]);
    }
    (Tensor::from_vec(ids, (b, t), dev).unwrap(), Tensor::from_vec(tg, (b, t), dev).unwrap())
}


/// A batch of plain tokens (300..556: no glance marks), for a model without the glance.
fn plain_batch(dev: &Device, b: usize, t: usize, seed: u64) -> (Tensor, Tensor) {
    let mut rng = ava_data::rng::SplitMix64::new(seed);
    let toks: Vec<u32> = (0..b * (t + 1)).map(|_| 300 + rng.below(256) as u32).collect();
    let (mut ids, mut tg) = (Vec::with_capacity(b * t), Vec::with_capacity(b * t));
    for row in toks.chunks(t + 1) {
        ids.extend_from_slice(&row[..t]);
        tg.extend_from_slice(&row[1..]);
    }
    (Tensor::from_vec(ids, (b, t), dev).unwrap(), Tensor::from_vec(tg, (b, t), dev).unwrap())
}
fn s(t: &Tensor) -> f32 {
    t.to_scalar::<f32>().unwrap()
}

/// Every parameter's agreement; f32: the failures. Prints one line per parameter (f32) and the mean cos.
fn compare(tag: &str, varmap: &VarMap, g_ref: &GradStore, g_new: &GradStore, hold: bool) -> Vec<String> {
    let mut bad = Vec::new();
    let mut coss = Vec::new();
    let mut worst = (1.0f32, 0.0f32, String::new());
    for (name, var) in vars(varmap) {
        let Some(r) = g_ref.get(var.as_tensor()) else {
            if g_new.get(var.as_tensor()).is_some() {
                bad.push(format!("{tag} {name}: fused grad for a parameter autograd leaves out"));
            }
            continue;
        };
        let Some(m) = g_new.get(var.as_tensor()) else {
            bad.push(format!("{tag} {name}: no fused grad"));
            continue;
        };
        if r.dims() != m.dims() {
            bad.push(format!("{tag} {name}: shape {:?} vs {:?}", r.dims(), m.dims()));
            continue;
        }
        let (r, m) = (r.to_dtype(DType::F32).unwrap().flatten_all().unwrap(), m.to_dtype(DType::F32).unwrap().flatten_all().unwrap());
        let nr = s(&r.sqr().unwrap().sum_all().unwrap()).sqrt();
        let nm = s(&m.sqr().unwrap().sum_all().unwrap()).sqrt();
        if nr == 0.0 && nm == 0.0 {
            eprintln!("{tag} {name:<28} zero in both");
            continue;
        }
        let cos = s(&r.mul(&m).unwrap().sum_all().unwrap()) / (nr * nm).max(1e-30);
        let diff = r.sub(&m).unwrap();
        let rel = s(&diff.sqr().unwrap().sum_all().unwrap()).sqrt() / nr.max(1e-30);
        let maxrel = s(&diff.abs().unwrap().max(0).unwrap()) / s(&r.abs().unwrap().max(0).unwrap()).max(1e-30);
        coss.push(cos);
        if cos < worst.0 {
            worst = (cos, rel, name.clone());
        }
        if hold {
            eprintln!("{tag} {name:<28} cos {cos:.7} rel-norm-diff {rel:.2e} max-rel {maxrel:.2e}");
            if !(cos > 0.9999 && rel < 1e-2) {
                bad.push(format!("{tag} {name}: cos {cos} rel-norm-diff {rel}"));
            }
        }
    }
    let mean = coss.iter().sum::<f32>() / coss.len().max(1) as f32;
    eprintln!("{tag}: {} parameters, mean cos {mean:.6}, lowest {:.6} ({}, rel {:.2e})", coss.len(), worst.0, worst.2, worst.1);
    bad
}

/// One model and batch: the fused step (f32 and bf16) against autograd, weights `wt` / `raw` as in loss_and_grads_a_wr.
fn check(tag: &str, varmap: &VarMap, model: &Model, ids: &Tensor, tg: &Tensor, wt: Option<&Tensor>, raw: bool) -> Vec<String> {
    let (l_ref, _, g_ref) = model.loss_and_grads_autograd(ids, tg, wt, raw).unwrap();
    let rates_ref = model.homeostasis_rates().unwrap(); // --homeostasis: the rates the forward leaves (else empty)
    let x = s(&l_ref);
    let mut bad = Vec::new();
    for bf16 in [false, true] {
        model.set_homeostasis_rates(&vec![-1.0; rates_ref.len()]).unwrap(); // so stale rates cannot pass
        let (l_new, _, g_new) = model.loss_and_grads_win(ids, tg, wt, raw, &Mm { bf16 }).unwrap();
        let y = s(&l_new);
        let tol = if bf16 { 1e-2 } else { 1e-4 };
        eprintln!("{tag} bf16={bf16}: loss autograd {x:.6} fused {y:.6} (|diff| {:.2e}, bar {tol:.0e})", (x - y).abs());
        if !((x - y).abs() < tol) {
            bad.push(format!("{tag} bf16={bf16}: loss {x} vs fused {y}"));
        }
        bad.extend(compare(&format!("{tag} bf16={bf16}"), varmap, &g_ref, &g_new, !bf16));
        let rates = model.homeostasis_rates().unwrap();
        if !rates_ref.is_empty() || !rates.is_empty() {
            let dmax = rates.iter().zip(&rates_ref).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            eprintln!("{tag} bf16={bf16}: homeostasis rates {} vs {}, max |diff| {dmax:.2e}", rates.len(), rates_ref.len());
            if rates.len() != rates_ref.len() || (!bf16 && dmax > 1e-6) {
                bad.push(format!("{tag} bf16={bf16}: homeostasis rates off ({} vs {}, max diff {dmax})", rates.len(), rates_ref.len()));
            }
        }
    }
    bad
}

#[test]
fn winners_step_matches_autograd() {
    let Some(dev) = cuda() else {
        eprintln!("winners step: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let mut bad: Vec<String> = Vec::new();
    // G1 and G3: d128 L3, all nine winners and the glance
    let cfg = Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 128, 3, 64).with_glance().unwrap().with_arch(arch_of(WINNERS)).unwrap();
    assert!(fwin::step_ok(&cfg), "G1's options must take the fused step");
    let (varmap, m1) = model(&dev, cfg, 1);
    let (ids, tg) = batch(&dev, 4, cfg.ctx, 31);
    // the training entry point takes the fused step for this model
    assert!(m1.win_step(&ids) || std::env::var("AVA_WIN_FUSED").is_ok_and(|v| v == "0"));
    let (l_with, _, _) = m1.loss_and_grads_with(&ids, &tg, false).unwrap().expect("loss_and_grads_with: a step");
    bad.extend(check("G1", &varmap, &m1, &ids, &tg, None, false));
    let (l_g1, _, _) = m1.loss_and_grads_win(&ids, &tg, None, false, &Mm { bf16: false }).unwrap();
    assert_eq!(s(&l_with), s(&l_g1), "loss_and_grads_with routes to the fused step");
    let n = 4 * cfg.ctx;
    let mask: Vec<f32> = (0..n).map(|i| if (i % cfg.ctx) >= cfg.ctx / 2 || i % 7 == 0 { 1.0 } else { 0.0 }).collect();
    let w = Tensor::from_vec(mask, n, &dev).unwrap();
    bad.extend(check("G3 mask", &varmap, &m1, &ids, &tg, Some(&w), false));
    {
        // printed only: plain D (no block options, round 12's step) on the same shape, weights and batch, for the bf16
        // gradients' usual spread against autograd (bf16 rounding flips spikes; small vectors like bb move most)
        let cfg0 = Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 128, 3, 64).with_glance().unwrap();
        let (vm0, m0) = model(&dev, cfg0, 1);
        let (_, _, g_ref) = m0.loss_and_grads_autograd(&ids, &tg, Some(&w), false).unwrap();
        let (_, _, g_new) = m0.loss_and_grads_a_w(&ids, &tg, Some(&w), &Mm { bf16: true }).unwrap();
        compare("calibration: plain D, G3 mask bf16=true", &vm0, &g_ref, &g_new, false);
        let (_, _, g_ref) = m0.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
        let (_, _, g_new) = m0.loss_and_grads_a_w(&ids, &tg, None, &Mm { bf16: true }).unwrap();
        compare("calibration: plain D, G1 bf16=true", &vm0, &g_ref, &g_new, false);
        let (_, _, g_new) = m0.loss_and_grads_a_w(&ids, &tg, None, &Mm { bf16: false }).unwrap();
        compare("calibration: plain D, G1 bf16=false", &vm0, &g_ref, &g_new, false);
    }
    let signed: Vec<f32> = (0..n).map(|i| [0.0, 1.0, -0.5, 0.25][i % 4] / n as f32).collect();
    let w = Tensor::from_vec(signed, n, &dev).unwrap();
    bad.extend(check("G3 raw", &varmap, &m1, &ids, &tg, Some(&w), true));
    // G2: d64 L2, head size 32
    let cfg2 = Config::with_kind(Kind::D, VOCAB, 64, 2, 64).with_arch(arch_of("--head-size 32 --kwta 0.25 --mlp-act relu2 --lif-scale --value-residual")).unwrap();
    assert!(fwin::step_ok(&cfg2), "G2's options must take the fused step");
    let (varmap2, m2) = model(&dev, cfg2, 2);
    for (name, var) in vars(&varmap2) {
        if name.ends_with(".mem.win") {
            // at d64 with --lif-scale the input currents are too small to fire: larger ones, so the spikes carry signal
            var.set(&var.as_tensor().affine(4.0, 0.0).unwrap()).unwrap();
        }
    }
    let (ids2, tg2) = plain_batch(&dev, 4, cfg2.ctx, 37);
    bad.extend(check("G2", &varmap2, &m2, &ids2, &tg2, None, false));
    // any other option keeps autograd
    for extra in ["--gates kda", "--token-shift", "--bonus", "--feedback", "--surrogate-slope 2"] {
        if let Ok(c) = Config::with_kind(Kind::D, VOCAB, 64, 2, 64).with_arch(arch_of(&format!("{WINNERS} {extra}"))) {
            assert!(!fwin::step_ok(&c), "{extra} must keep the autograd path");
        }
    }
    assert!(bad.is_empty(), "winners step off: {bad:#?}");
}

const SLOW: &str = "--gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";

#[test]
fn slow_options_step_matches_autograd() {
    // v47 SLOW (runs/v47/prereg_fast_winners.md, Amendment 1): G4 = G1's model with the slow five too (the 0/1 weights
    // as well); G5 = the slow five alone at head size 32 (d64 L2). Same bars as G1-G3.
    let Some(dev) = cuda() else {
        eprintln!("slow step: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let mut bad: Vec<String> = Vec::new();
    let all = format!("{WINNERS} {SLOW}");
    let cfg = Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 128, 3, 64).with_glance().unwrap().with_arch(arch_of(&all)).unwrap();
    assert!(fwin::step_ok(&cfg), "G4's options must take the fused step");
    let (varmap, m4) = model(&dev, cfg, 4);
    let (ids, tg) = batch(&dev, 4, cfg.ctx, 41);
    assert!(m4.win_step(&ids) || std::env::var("AVA_WIN_FUSED").is_ok_and(|v| v == "0"));
    bad.extend(check("G4", &varmap, &m4, &ids, &tg, None, false));
    let n = 4 * cfg.ctx;
    let mask: Vec<f32> = (0..n).map(|i| if (i % cfg.ctx) >= cfg.ctx / 2 || i % 7 == 0 { 1.0 } else { 0.0 }).collect();
    let w = Tensor::from_vec(mask, n, &dev).unwrap();
    bad.extend(check("G4 mask", &varmap, &m4, &ids, &tg, Some(&w), false));
    let cfg5 = Config::with_kind(Kind::D, VOCAB, 64, 2, 64).with_arch(arch_of(&format!("--head-size 32 {SLOW}"))).unwrap();
    assert!(fwin::step_ok(&cfg5), "G5's options must take the fused step");
    let (varmap5, m5) = model(&dev, cfg5, 5);
    let (ids5, tg5) = plain_batch(&dev, 4, cfg5.ctx, 43);
    bad.extend(check("G5", &varmap5, &m5, &ids5, &tg5, None, false));
    assert!(bad.is_empty(), "slow step off: {bad:#?}");
}

/// v52 spike fix (--lif-reset-grad): G4's model (the nine winners, the glance and the slow five) with the LIF reset
/// carried in the backward: the fused step against autograd at G1-G3's bars. The autograd side runs lif_gen's op, itself
/// checked against lif_ref's autograd with the reset carried (lif_gen's tests, slope -4).
#[test]
fn reset_grad_step_matches_autograd() {
    let Some(dev) = cuda() else {
        eprintln!("reset-grad step: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let all = format!("{WINNERS} {SLOW} --lif-reset-grad");
    let cfg = Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 128, 3, 64).with_glance().unwrap().with_arch(arch_of(&all)).unwrap();
    assert!(cfg.arch.lif_reset_grad, "the flag must parse");
    assert!(fwin::step_ok(&cfg), "the reset-grad options must take the fused step");
    let (varmap, m) = model(&dev, cfg, 4);
    let (ids, tg) = batch(&dev, 4, cfg.ctx, 41);
    assert!(m.win_step(&ids) || std::env::var("AVA_WIN_FUSED").is_ok_and(|v| v == "0"));
    let bad = check("G4 reset-grad", &varmap, &m, &ids, &tg, None, false);
    assert!(bad.is_empty(), "reset-grad step off: {bad:#?}");
}

/// v52 fix_S, G-cap: G4's model (the nine winners, the glance and the slow five: channel gates, momentum 0.5) with
/// a memory-state bound small enough to hold part of the state (mem_gen::with_cap): the fused step against autograd at
/// G1-G3's bars. Both zero the gradient where the bound holds (the fused step in fwin_ops' scan kernels, autograd in
/// mem_gen's op, itself checked against delta_rec's clamp in mem_gen's tests). The bound must move the loss (so it was
/// in force). With AVA_MGK=1 the fused step also takes memgk's fall-back to memg_bwd4.
#[test]
fn bounded_memory_step_matches_autograd() {
    let Some(dev) = cuda() else {
        eprintln!("bounded memory step: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let mut bad: Vec<String> = Vec::new();
    let all = format!("{WINNERS} {SLOW}");
    let cfg = Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 128, 3, 64).with_glance().unwrap().with_arch(arch_of(&all)).unwrap();
    assert!(fwin::step_ok(&cfg), "G-cap's options must take the fused step");
    let (varmap, m4) = model(&dev, cfg, 4);
    let (ids, tg) = batch(&dev, 4, cfg.ctx, 41);
    let (l0, _, _) = m4.loss_and_grads_win(&ids, &tg, None, false, &Mm { bf16: false }).unwrap();
    for cap in [0.02f32, 0.2] {
        let (l1, _, _) = ava_gpu::mem_gen::with_cap(cap, || m4.loss_and_grads_win(&ids, &tg, None, false, &Mm { bf16: false })).unwrap();
        eprintln!("G-cap {cap}: fused loss {:.6} without the bound, {:.6} with it", s(&l0), s(&l1));
        if !((s(&l0) - s(&l1)).abs() > 1e-4) {
            bad.push(format!("G-cap {cap}: the bound did not move the loss ({} vs {})", s(&l0), s(&l1)));
        }
        bad.extend(ava_gpu::mem_gen::with_cap(cap, || check(&format!("G-cap {cap}"), &varmap, &m4, &ids, &tg, None, false)));
    }
    assert!(bad.is_empty(), "bounded memory step off: {bad:#?}");
}

#[test]
fn engram_step_matches_autograd() {
    // v47 Amendment 5 (runs/v47/prereg_fast_winners.md): G6 = WIN + SLOW + glance + mag + --engram 1 (d128 L3 ctx 64),
    // mean, 0/1 and signed raw weights; G7 = WIN + --engram 0 (d64 L2, plain tokens); G8 = G6's options at d64 L2,
    // ctx 2048, batch 1. Same bars as G1-G5.
    let Some(dev) = cuda() else {
        eprintln!("engram step: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let mut bad: Vec<String> = Vec::new();
    let all = format!("{WINNERS} {SLOW}");
    let g = ava_data::numbers::VOCAB_GLANCE;
    let cfg = Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
    let cfg = cfg.with_arch(arch_of(&all)).unwrap();
    assert!(fwin::step_ok(&cfg), "G6's options must take the fused step");
    let (varmap, m6) = model(&dev, cfg, 6);
    let (ids, tg) = batch(&dev, 4, cfg.ctx, 61);
    assert!(m6.win_step(&ids) || std::env::var("AVA_WIN_FUSED").is_ok_and(|v| v == "0"));
    bad.extend(check("G6", &varmap, &m6, &ids, &tg, None, false));
    {
        // printed only: plain D + glance + mag + the same engram on loss_and_grads_a (f32), for the engram's own spread
        // against autograd (its attention backward hands dqkv over as bf16 even in f32)
        let cfg0 = Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
        let (vm0, m0) = model(&dev, cfg0, 6);
        let (_, _, g_ref) = m0.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
        let (_, _, g_new) = m0.loss_and_grads_a_w(&ids, &tg, None, &Mm { bf16: false }).unwrap();
        compare("calibration: plain D + engram, bf16=false", &vm0, &g_ref, &g_new, false);
    }
    let n = 4 * cfg.ctx;
    let mask: Vec<f32> = (0..n).map(|i| if (i % cfg.ctx) >= cfg.ctx / 2 || i % 7 == 0 { 1.0 } else { 0.0 }).collect();
    let w = Tensor::from_vec(mask, n, &dev).unwrap();
    bad.extend(check("G6 mask", &varmap, &m6, &ids, &tg, Some(&w), false));
    let signed: Vec<f32> = (0..n).map(|i| [0.0, 1.0, -0.5, 0.25][i % 4] / n as f32).collect();
    let w = Tensor::from_vec(signed, n, &dev).unwrap();
    bad.extend(check("G6 raw", &varmap, &m6, &ids, &tg, Some(&w), true));
    let cfg7 = Config::with_kind(Kind::D, VOCAB, 64, 2, 64).with_engram(0).unwrap().with_arch(arch_of(WINNERS)).unwrap();
    assert!(fwin::step_ok(&cfg7), "G7's options must take the fused step");
    let (varmap7, m7) = model(&dev, cfg7, 7);
    let (ids7, tg7) = plain_batch(&dev, 4, cfg7.ctx, 71);
    bad.extend(check("G7", &varmap7, &m7, &ids7, &tg7, None, false));
    let cfg8 = Config::with_kind(Kind::D, g, 64, 2, 2048).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
    let cfg8 = cfg8.with_arch(arch_of(&all)).unwrap();
    assert!(fwin::step_ok(&cfg8), "G8's options must take the fused step");
    let (varmap8, m8) = model(&dev, cfg8, 8);
    let (ids8, tg8) = batch(&dev, 1, cfg8.ctx, 81);
    bad.extend(check("G8", &varmap8, &m8, &ids8, &tg8, None, false));
    assert!(bad.is_empty(), "engram step off: {bad:#?}");
}

// ---------------------------------------------------------------- v51 fused N: --neuromod 3 on the fused step

/// v51 fused N: the nucleus's weights moved off their start (where every signal is exactly 1): its norm gain around 1,
/// its read-out and bias noise; `ach_bias` is added to the ACh signal's bias.
fn perturb_nm(varmap: &VarMap, dev: &Device, ach_bias: f32) {
    let mut seen = 0;
    for (k, (name, var)) in vars(varmap).into_iter().enumerate() {
        let t = var.as_tensor();
        let (amp, off) = match name.as_str() {
            "nm.norm" => (0.3, 1.0),
            "nm.wn" => (0.1, 0.0),
            "nm.bn" => (0.5, 0.0),
            _ => continue,
        };
        seen += 1;
        let mut v: Vec<f32> = noise(dev, t.elem_count(), 2000 + k as u64, amp).iter().map(|x| x + off).collect();
        if name == "nm.bn" {
            v[1] += ach_bias; // ACh is column 1 (v50::ACH)
        }
        var.set(&Tensor::from_vec(v, t.shape(), dev).unwrap()).unwrap();
    }
    assert_eq!(seen, 3, "the nucleus's three tensors");
}

/// Adds `by` to every block's write bias bb (more write strengths near 2, so ACh's clamp acts more often).
fn shift_bb(varmap: &VarMap, by: f32) {
    for (name, var) in vars(varmap) {
        if name.ends_with(".bb") {
            var.set(&var.as_tensor().affine(1.0, by as f64).unwrap()).unwrap();
        }
    }
}

/// The fused step's logits (fwin::capture_logits) against the reference forward's, same model and batch. f32:
/// ||fused - autograd|| / ||autograd|| < `bar`; bf16: printed only. `hold` false: printed only (calibration).
fn check_logits(tag: &str, model: &Model, ids: &Tensor, tg: &Tensor, bar: f32, hold: bool) -> Vec<String> {
    let lr = model.forward(ids).unwrap();
    let v = lr.dim(lr.rank() - 1).unwrap();
    let lr = lr.reshape(((), v)).unwrap();
    let nr = s(&lr.sqr().unwrap().sum_all().unwrap()).sqrt();
    let mut bad = Vec::new();
    for bf16 in [false, true] {
        fwin::capture_logits(true);
        model.loss_and_grads_win(ids, tg, None, false, &Mm { bf16 }).unwrap();
        let lf = fwin::take_logits().expect("the fused step's logits");
        fwin::capture_logits(false);
        let diff = lf.sub(&lr).unwrap();
        let rel = s(&diff.sqr().unwrap().sum_all().unwrap()).sqrt() / nr.max(1e-30);
        let maxd = s(&diff.abs().unwrap().flatten_all().unwrap().max(0).unwrap());
        let held = hold && !bf16;
        let note = if held { format!(" (bar {bar:.0e})") } else { " (printed only)".to_string() };
        eprintln!("{tag} bf16={bf16}: logits rel-norm-diff {rel:.2e}, max |diff| {maxd:.2e}{note}");
        if held && !(rel < bar) {
            bad.push(format!("{tag} bf16={bf16}: logits rel-norm-diff {rel}"));
        }
    }
    bad
}

const LOGITS_BAR: f32 = 1e-3;

#[test]
fn neuromod_step_matches_autograd() {
    // v51 fused N (runs/v51/fused_n_check.sh): --neuromod 3 on the fused step against loss + candle's autograd, G1-G8's
    // bars (f32: loss within 1e-4, every parameter cos > 0.9999 and rel-norm-diff < 1e-2; bf16: loss within 1e-2, the
    // gradients printed), and the logits (f32: rel-norm-diff < 1e-3; bf16 printed), the nucleus's weights off their start.
    // * N1: Ava 0.4b's recipe (WIN + SLOW, glance, mag, --engram 1) + --neuromod 3, d128 L3 ctx 64 batch 4: mean, 0/1
    //   and signed raw weights
    // * N2: N1's model with the ACh bias +1.5 and every write bias bb +1: most write strengths clamp at 2
    // * N3: WIN + --neuromod 3 at d64 L2 head size 32 (per-head gates, the plain binary LIF, no momentum), plain tokens
    // * N4: N1's options at d64 L2, ctx 2048, batch 1 (as G8)
    // * routing: these take the fused step; --dale, --loop and round 2's R and W keep autograd
    // fused_n_check.sh runs this test three times: as built, with AVA_WIN_KEEP_PREP=0 (the backward reruns the memory's
    // prep and scales it again) and with AVA_NM_K=0 (the candle-op twins instead of the kernels).
    // Printed only: the logits of N1's recipe without --neuromod (the step's own spread, for calibration).
    let Some(dev) = cuda() else {
        eprintln!("neuromod step: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let mut bad: Vec<String> = Vec::new();
    let all = format!("{WINNERS} {SLOW}");
    let g = ava_data::numbers::VOCAB_GLANCE;
    let n1 = |nm: bool| {
        let c = Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
        let flags = if nm { format!("{all} --neuromod 3") } else { all.clone() };
        c.with_arch(arch_of(&flags)).unwrap()
    };
    let cfg = n1(true);
    assert!(fwin::step_ok(&cfg), "N1's options must take the fused step");
    let (varmap, m1) = model(&dev, cfg, 9);
    perturb_nm(&varmap, &dev, 0.0);
    let (ids, tg) = batch(&dev, 4, cfg.ctx, 91);
    assert!(m1.win_step(&ids) || std::env::var("AVA_WIN_FUSED").is_ok_and(|v| v == "0"));
    let (l_with, _, _) = m1.loss_and_grads_with(&ids, &tg, false).unwrap().expect("loss_and_grads_with: a step");
    let (l_n1, _, _) = m1.loss_and_grads_win(&ids, &tg, None, false, &Mm { bf16: false }).unwrap();
    assert_eq!(s(&l_with), s(&l_n1), "loss_and_grads_with routes --neuromod to the fused step");
    bad.extend(check("N1", &varmap, &m1, &ids, &tg, None, false));
    bad.extend(check_logits("N1", &m1, &ids, &tg, LOGITS_BAR, true));
    let n = 4 * cfg.ctx;
    let mask: Vec<f32> = (0..n).map(|i| if (i % cfg.ctx) >= cfg.ctx / 2 || i % 7 == 0 { 1.0 } else { 0.0 }).collect();
    let w = Tensor::from_vec(mask, n, &dev).unwrap();
    bad.extend(check("N1 mask", &varmap, &m1, &ids, &tg, Some(&w), false));
    let signed: Vec<f32> = (0..n).map(|i| [0.0, 1.0, -0.5, 0.25][i % 4] / n as f32).collect();
    let w = Tensor::from_vec(signed, n, &dev).unwrap();
    bad.extend(check("N1 raw", &varmap, &m1, &ids, &tg, Some(&w), true));
    {
        // printed only: the same recipe without --neuromod, same seed and batch
        let (_, m0) = model(&dev, n1(false), 9);
        check_logits("calibration: N1 without --neuromod", &m0, &ids, &tg, LOGITS_BAR, false);
    }
    // N2: ACh's clamp at work
    let (varmap2, m2) = model(&dev, cfg, 9);
    perturb_nm(&varmap2, &dev, 1.5);
    shift_bb(&varmap2, 1.0);
    bad.extend(check("N2", &varmap2, &m2, &ids, &tg, None, false));
    bad.extend(check_logits("N2", &m2, &ids, &tg, LOGITS_BAR, true));
    // N3: head size 32, per-head gates, the plain LIF
    let cfg3 = Config::with_kind(Kind::D, VOCAB, 64, 2, 64).with_arch(arch_of("--head-size 32 --kwta 0.25 --mlp-act relu2 --lif-scale --value-residual --neuromod 3")).unwrap();
    assert!(fwin::step_ok(&cfg3), "N3's options must take the fused step");
    let (varmap3, m3) = model(&dev, cfg3, 10);
    perturb_nm(&varmap3, &dev, 0.0);
    for (name, var) in vars(&varmap3) {
        if name.ends_with(".mem.win") {
            // as G2: at d64 with --lif-scale the input currents are too small to fire
            var.set(&var.as_tensor().affine(4.0, 0.0).unwrap()).unwrap();
        }
    }
    let (ids3, tg3) = plain_batch(&dev, 4, cfg3.ctx, 101);
    bad.extend(check("N3", &varmap3, &m3, &ids3, &tg3, None, false));
    bad.extend(check_logits("N3", &m3, &ids3, &tg3, LOGITS_BAR, true));
    // N4: long windows
    let cfg4 = Config::with_kind(Kind::D, g, 64, 2, 2048).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
    let cfg4 = cfg4.with_arch(arch_of(&format!("{all} --neuromod 3"))).unwrap();
    assert!(fwin::step_ok(&cfg4), "N4's options must take the fused step");
    let (varmap4, m4) = model(&dev, cfg4, 11);
    perturb_nm(&varmap4, &dev, 0.0);
    let (ids4, tg4) = batch(&dev, 1, cfg4.ctx, 111);
    bad.extend(check("N4", &varmap4, &m4, &ids4, &tg4, None, false));
    bad.extend(check_logits("N4", &m4, &ids4, &tg4, LOGITS_BAR, true));
    // routing: the other v50 options keep autograd (with or without --neuromod); round 2's ten parts take the fused
    // step now (v51 fast1: P, DA, M, Y, Pr and S, round2_step_matches_autograd; v51 fast2: R, W, TH and CB,
    // tests/fast2_parts.rs; merged in runs/v51/merge_f.md)
    for extra in ["--dale 0.8", "--loop 2"] {
        for nm in ["", " --neuromod 3"] {
            let flags = format!("{all} {extra}{nm}");
            if let Ok(c) = Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_arch(arch_of(&flags)) {
                assert!(!fwin::step_ok(&c), "{extra}{nm} must keep the autograd path");
            }
        }
    }
    assert!(bad.is_empty(), "neuromod step off: {bad:#?}");
}

// ---- v51 fast1, Amendment F-1 (runs/v51/prereg_v51.md): the fused f32 step and the f32 autograd against an f64
// reference: the same model, weights and batch on the CPU in f64 through candle's autograd (AVA_LIF_REF=1: the
// plain-op LIF; lif_gen has no f64 path).

fn cpu64(t: &Tensor) -> Tensor {
    t.to_device(&Device::Cpu).unwrap().to_dtype(DType::F64).unwrap()
}

/// `cfg`'s model on the CPU in f64 with `src`'s weights (by name): (loss, logits [N, V], gradients by name).
fn f64_ref(cfg: Config, src: &VarMap, ids: &Tensor, tg: &Tensor, wt: Option<&Tensor>, raw: bool) -> (f64, Tensor, Vec<(String, Tensor)>) {
    let cpu = Device::Cpu;
    let vm = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F64, &cpu)).unwrap();
    m.set_track_activity(false);
    let theirs = vars(src);
    let mine = vars(&vm);
    assert_eq!(theirs.iter().map(|x| &x.0).collect::<Vec<_>>(), mine.iter().map(|x| &x.0).collect::<Vec<_>>(), "the f64 model's weights");
    for ((_, a), (_, b)) in theirs.iter().zip(&mine) {
        b.set(&cpu64(a.as_tensor())).unwrap();
    }
    let (ids, tg) = (ids.to_device(&cpu).unwrap(), tg.to_device(&cpu).unwrap());
    let wt = wt.map(cpu64);
    std::env::set_var("AVA_LIF_REF", "1");
    let r = m.loss_and_grads_autograd(&ids, &tg, wt.as_ref(), raw).map(|(l, _, g)| (l, g));
    let lg = m.forward(&ids);
    std::env::remove_var("AVA_LIF_REF");
    let (l, g) = r.unwrap();
    let lg = lg.unwrap();
    let v = lg.dim(lg.rank() - 1).unwrap();
    let gr = mine.into_iter().filter_map(|(n, x)| g.get(x.as_tensor()).map(|t| (n, t.clone()))).collect();
    (l.to_scalar::<f64>().unwrap(), lg.reshape(((), v)).unwrap(), gr)
}

/// |a - r| / |r| over the whole tensor, in f64.
fn rel64(a: &Tensor, r: &Tensor) -> f64 {
    let (a, r) = (cpu64(a).flatten_all().unwrap(), r.to_dtype(DType::F64).unwrap().flatten_all().unwrap());
    let nr = r.sqr().unwrap().sum_all().unwrap().to_scalar::<f64>().unwrap().sqrt();
    let nd = a.sub(&r).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f64>().unwrap().sqrt();
    nd / nr.max(1e-300)
}

/// F-1 for one case: the GPU's f32 autograd and fused step against the f64 reference, per parameter. Returns the
/// parameters that break e_f <= 2 e_a + 1e-4 (and any missing gradient); prints every parameter and the worst ratio.
fn f1_case(tag: &str, cfg: Config, varmap: &VarMap, model: &Model, ids: &Tensor, tg: &Tensor, wt: Option<&Tensor>, raw: bool) -> Vec<String> {
    let (l_a, _, g_a) = model.loss_and_grads_autograd(ids, tg, wt, raw).unwrap();
    let (l_f, _, g_f) = model.loss_and_grads_win(ids, tg, wt, raw, &Mm { bf16: false }).unwrap();
    let lg32 = model.forward(ids).unwrap();
    let t0 = std::time::Instant::now();
    let (l64, lg64, g64) = f64_ref(cfg, varmap, ids, tg, wt, raw);
    let v = lg32.dim(lg32.rank() - 1).unwrap();
    let lrel = rel64(&lg32.reshape(((), v)).unwrap(), &lg64);
    eprintln!(
        "F-1 {tag}: loss f64 {l64:.9}, autograd {:.9}, fused {:.9}; logits f32 vs f64 rel {lrel:.2e}; f64 reference {:.1}s",
        s(&l_a),
        s(&l_f),
        t0.elapsed().as_secs_f64()
    );
    let by: std::collections::HashMap<String, Var> = vars(varmap).into_iter().collect();
    let mut bad = Vec::new();
    let mut worst = (0.0f64, String::new());
    for (name, r) in &g64 {
        let var = &by[name];
        let (Some(ga), Some(gf)) = (g_a.get(var.as_tensor()), g_f.get(var.as_tensor())) else {
            bad.push(format!("{tag} {name}: a gradient is missing (autograd {}, fused {})", g_a.get(var.as_tensor()).is_some(), g_f.get(var.as_tensor()).is_some()));
            continue;
        };
        let (ea, ef) = (rel64(ga, r), rel64(gf, r));
        let ok = ef <= 2.0 * ea + 1e-4;
        let ratio = ef / ea.max(1e-12);
        if ratio > worst.0 {
            worst = (ratio, name.clone());
        }
        eprintln!("F-1 {tag} {name:<28} e_auto {ea:.2e} e_fused {ef:.2e} ratio {ratio:>8.2}{}", if ok { "" } else { "  BREAKS F-1" });
        if !ok {
            bad.push(format!("{tag} {name}: e_fused {ef:.3e} > 2 x e_auto {ea:.3e} + 1e-4"));
        }
    }
    eprintln!("F-1 {tag}: {} parameters, worst e_fused/e_auto {:.2} ({}), {} break the rule", g64.len(), worst.0, worst.1, bad.len());
    bad
}

/// F-1's validity check (CPU only): N1's model in f32 (the usual CPU path) and in f64 (the reference) give the same
/// loss within 1e-5; the gradients' agreement is printed.
#[test]
fn f1_reference_is_the_reference_in_f64() {
    let cpu = Device::Cpu;
    let all = format!("{WINNERS} {SLOW} --neuromod 3");
    let c = Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 128, 3, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
    let cfg = c.with_arch(arch_of(&all)).unwrap();
    // the CPU has no seeded generator: a fresh random model (the check holds for any weights)
    let varmap = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &cpu)).unwrap();
    perturb(&varmap, &cpu);
    perturb_nm(&varmap, &cpu, 0.0);
    let (ids, tg) = batch(&cpu, 2, cfg.ctx, 91);
    let (l32, _, g32) = m.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
    let (l64, _, g64) = f64_ref(cfg, &varmap, &ids, &tg, None, false);
    let by: std::collections::HashMap<String, Var> = vars(&varmap).into_iter().collect();
    let mut worst = (0.0f64, String::new());
    for (name, r) in &g64 {
        let e = rel64(g32.get(by[name].as_tensor()).unwrap_or_else(|| panic!("no f32 gradient for {name}")), r);
        if e > worst.0 {
            worst = (e, name.clone());
        }
    }
    eprintln!("F-1 validity: loss f32 {:.9} f64 {l64:.9}; {} gradients, worst rel {:.2e} ({})", s(&l32), g64.len(), worst.0, worst.1);
    assert!((s(&l32) as f64 - l64).abs() < 1e-5, "f32 {} vs f64 {l64}", s(&l32));
    let n32 = vars(&varmap).iter().filter(|(_, v)| g32.get(v.as_tensor()).is_some()).count();
    assert_eq!(g64.len(), n32, "the f64 reference has a gradient for every weight the f32 one has");
}

/// F-1 (CUDA): C3's cases (N1, N1 mask, N1 raw, N2, N3, N4: the same models, weights and batches as
/// neuromod_step_matches_autograd) under F-1's rule; G6, G6 raw and G8 (no --neuromod) reported for calibration.
#[test]
fn f1_neuromod_against_f64() {
    let Some(dev) = cuda() else {
        eprintln!("F-1: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let mut bad: Vec<String> = Vec::new();
    let all = format!("{WINNERS} {SLOW}");
    let g = ava_data::numbers::VOCAB_GLANCE;
    let c1 = Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
    let cfg = c1.with_arch(arch_of(&format!("{all} --neuromod 3"))).unwrap();
    let (varmap, m1) = model(&dev, cfg, 9);
    perturb_nm(&varmap, &dev, 0.0);
    let (ids, tg) = batch(&dev, 4, cfg.ctx, 91);
    bad.extend(f1_case("N1", cfg, &varmap, &m1, &ids, &tg, None, false));
    let n = 4 * cfg.ctx;
    let mask: Vec<f32> = (0..n).map(|i| if (i % cfg.ctx) >= cfg.ctx / 2 || i % 7 == 0 { 1.0 } else { 0.0 }).collect();
    let wm = Tensor::from_vec(mask, n, &dev).unwrap();
    bad.extend(f1_case("N1 mask", cfg, &varmap, &m1, &ids, &tg, Some(&wm), false));
    let signed: Vec<f32> = (0..n).map(|i| [0.0, 1.0, -0.5, 0.25][i % 4] / n as f32).collect();
    let ws = Tensor::from_vec(signed, n, &dev).unwrap();
    bad.extend(f1_case("N1 raw", cfg, &varmap, &m1, &ids, &tg, Some(&ws), true));
    let (varmap2, m2) = model(&dev, cfg, 9);
    perturb_nm(&varmap2, &dev, 1.5);
    shift_bb(&varmap2, 1.0);
    bad.extend(f1_case("N2", cfg, &varmap2, &m2, &ids, &tg, None, false));
    let cfg3 = Config::with_kind(Kind::D, VOCAB, 64, 2, 64).with_arch(arch_of("--head-size 32 --kwta 0.25 --mlp-act relu2 --lif-scale --value-residual --neuromod 3")).unwrap();
    let (varmap3, m3) = model(&dev, cfg3, 10);
    perturb_nm(&varmap3, &dev, 0.0);
    for (name, var) in vars(&varmap3) {
        if name.ends_with(".mem.win") {
            var.set(&var.as_tensor().affine(4.0, 0.0).unwrap()).unwrap();
        }
    }
    let (ids3, tg3) = plain_batch(&dev, 4, cfg3.ctx, 101);
    bad.extend(f1_case("N3", cfg3, &varmap3, &m3, &ids3, &tg3, None, false));
    let c4 = Config::with_kind(Kind::D, g, 64, 2, 2048).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
    let cfg4 = c4.with_arch(arch_of(&format!("{all} --neuromod 3"))).unwrap();
    let (varmap4, m4) = model(&dev, cfg4, 11);
    perturb_nm(&varmap4, &dev, 0.0);
    let (ids4, tg4) = batch(&dev, 1, cfg4.ctx, 111);
    bad.extend(f1_case("N4", cfg4, &varmap4, &m4, &ids4, &tg4, None, false));
    // calibration (not part of the decision): the engram cases without --neuromod
    let cfg6 = c1.with_arch(arch_of(&all)).unwrap();
    let (varmap6, m6) = model(&dev, cfg6, 6);
    let (ids6, tg6) = batch(&dev, 4, cfg6.ctx, 61);
    let mut cal = f1_case("G6", cfg6, &varmap6, &m6, &ids6, &tg6, None, false);
    let n6 = 4 * cfg6.ctx;
    let signed6: Vec<f32> = (0..n6).map(|i| [0.0, 1.0, -0.5, 0.25][i % 4] / n6 as f32).collect();
    let ws6 = Tensor::from_vec(signed6, n6, &dev).unwrap();
    cal.extend(f1_case("G6 raw", cfg6, &varmap6, &m6, &ids6, &tg6, Some(&ws6), true));
    let cfg8 = c4.with_arch(arch_of(&all)).unwrap();
    let (varmap8, m8) = model(&dev, cfg8, 8);
    let (ids8, tg8) = batch(&dev, 1, cfg8.ctx, 81);
    cal.extend(f1_case("G8", cfg8, &varmap8, &m8, &ids8, &tg8, None, false));
    eprintln!("F-1 calibration (G6, G6 raw, G8; not part of the decision): {} parameters break the rule: {cal:#?}", cal.len());
    eprintln!("F-1 DECISION: {}", if bad.is_empty() { "ROUNDING (every parameter of every N case within 2 e_auto + 1e-4)" } else { "BUG (see the list)" });
    assert!(bad.is_empty(), "F-1: {bad:#?}");
}

// ---- v51 fast1 (runs/v51/prereg_v51.md Amendment F-2, runs/v51/fast1_check.sh): round 2's P, DA, M, Y, Pr and S on
// the fused step (src/fwin_r2.rs) against loss + candle's autograd, at the N bars.

const SIX: &str = "--dendrite --dopamine --mood --rhythm --prune --stp";

/// v51 fast1: round 2's weights moved off their start (where each part changes nothing): every b{i}.v51 weight but Pr's
/// alive and use, and the mood's three vectors, with noise 0.3 added; the rhythms' reset bias set near -1 (resets every
/// few tokens) and ln ω moved by 0.1; the synapse's mix noise 0.5; Pr: every third unit of each block cut (the cached
/// counts then forgotten).
fn perturb_r2(varmap: &VarMap, model: &Model, dev: &Device) {
    for (k, (name, var)) in vars(varmap).into_iter().enumerate() {
        let t = var.as_tensor();
        let nz = |amp: f32| Tensor::from_vec(noise(dev, t.elem_count(), 3000 + k as u64, amp), t.shape(), dev).unwrap();
        let new = if name.ends_with(".v51.alive") {
            Tensor::from_vec((0..t.elem_count()).map(|j| if j % 3 == 0 { 0f32 } else { 1.0 }).collect::<Vec<f32>>(), t.shape(), dev).unwrap()
        } else if name.ends_with(".v51.use") {
            continue;
        } else if name == "v51.rhythm.br" {
            nz(0.5).affine(1.0, -1.0).unwrap()
        } else if name == "v51.rhythm.lw" {
            t.add(&nz(0.1)).unwrap()
        } else if name.ends_with("stp.mix") {
            t.add(&nz(0.5)).unwrap()
        } else if name.contains("v51") {
            t.add(&nz(0.3)).unwrap()
        } else {
            continue;
        };
        var.set(&new).unwrap();
    }
    model.v51_schedule(0, 1000);
}

/// Pr: the units' statistics (v51::after_step's input) the fused f32 step leaves, against the reference's: printed, and
/// the failures (max |diff| / max |ref| > 1e-4).
fn check_prune_stats(tag: &str, model: &Model, ids: &Tensor, tg: &Tensor) -> Vec<String> {
    let Some(alive) = model.v51_alive().unwrap() else { return Vec::new() };
    let mut bad = Vec::new();
    model.loss_and_grads_autograd(ids, tg, None, false).unwrap();
    let r = ava_gpu::fwin_r2::prune_stats_now(model);
    model.loss_and_grads_win(ids, tg, None, false, &Mm { bf16: false }).unwrap();
    let f = ava_gpu::fwin_r2::prune_stats_now(model);
    for (i, (a, b)) in r.iter().zip(&f).enumerate() {
        let (Some(a), Some(b)) = (a, b) else {
            bad.push(format!("{tag} block {i}: Pr statistics missing (reference {}, fused {})", a.is_some(), b.is_some()));
            continue;
        };
        let e = s(&a.sub(b).unwrap().abs().unwrap().max(0).unwrap()) / s(&a.abs().unwrap().max(0).unwrap()).max(1e-30);
        eprintln!("{tag} block {i}: Pr statistics max rel diff {e:.2e} ({} live units)", alive[i]);
        if !(e <= 1e-4) {
            bad.push(format!("{tag} block {i}: Pr statistics off ({e})"));
        }
    }
    bad
}

#[test]
fn round2_step_matches_autograd() {
    // v51 fast1 (prereg Amendment F-2): P --dendrite, DA --dopamine, M --mood, Y --rhythm, Pr --prune, S --stp on the
    // fused step against loss + candle's autograd, the N bars (f32: loss within 1e-4, every parameter cos > 0.9999 and
    // rel-norm-diff < 1e-2, logits rel-norm-diff < 1e-3; bf16: loss within 1e-2, the rest printed), the parts' weights off
    // their start (perturb_r2), Pr's cut units at work:
    // * R1-R6: N1's model (Ava 0.4b's recipe: WIN + SLOW, glance, mag, --engram 1; d128 L3 ctx 64 batch 4) + each part
    // * R7: N1's model + all six + --neuromod 3 (the nucleus off its start too): mean, 0/1 and signed raw weights; the
    //   training entry point routes it to the fused step; Pr's statistics (after_step's input) within 1e-4
    // * R8: R7's options at d64 L2, ctx 2048, batch 1 (32 scan chunks; the mood's slowest unit 1024 tokens)
    // * R9: WIN + all six at d64 L2 head size 32 (per-head gates, the plain binary LIF, the synapse on 0/1 spikes), plain
    //   tokens
    // * routing: each part alone and all six, with or without --neuromod 3, take the fused step; since the merge with
    //   v51 fast2 (runs/v51/merge_f.md) --letters and --columns take it too (alone or with the six)
    // fast1_check.sh also runs this test with AVA_WIN_KEEP_PREP=0 (Y's write gate redone on the rerun prep) and with
    // AVA_NM_K=0.
    let Some(dev) = cuda() else {
        eprintln!("round 2 step: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let mut bad: Vec<String> = Vec::new();
    let all = format!("{WINNERS} {SLOW}");
    let g = ava_data::numbers::VOCAB_GLANCE;
    let c1 = Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
    let (ids, tg) = batch(&dev, 4, 64, 91);
    for (tag, extra) in [("R1 P", "--dendrite"), ("R2 DA", "--dopamine"), ("R3 M", "--mood"), ("R4 Y", "--rhythm"), ("R5 Pr", "--prune"), ("R6 S", "--stp")] {
        let cfg = c1.with_arch(arch_of(&format!("{all} {extra}"))).unwrap();
        assert!(fwin::step_ok(&cfg), "{tag}: must take the fused step");
        let (varmap, m) = model(&dev, cfg, 9);
        perturb_r2(&varmap, &m, &dev);
        bad.extend(check(tag, &varmap, &m, &ids, &tg, None, false));
        bad.extend(check_logits(tag, &m, &ids, &tg, LOGITS_BAR, true));
    }
    // R7: everything at once
    let cfg7 = c1.with_arch(arch_of(&format!("{all} {SIX} --neuromod 3"))).unwrap();
    assert!(fwin::step_ok(&cfg7), "R7's options must take the fused step");
    let (varmap7, m7) = model(&dev, cfg7, 9);
    perturb_nm(&varmap7, &dev, 0.0);
    perturb_r2(&varmap7, &m7, &dev);
    assert!(m7.win_step(&ids) || std::env::var("AVA_WIN_FUSED").is_ok_and(|v| v == "0"));
    let (l_with, _, _) = m7.loss_and_grads_with(&ids, &tg, false).unwrap().expect("loss_and_grads_with: a step");
    let (l_r7, _, _) = m7.loss_and_grads_win(&ids, &tg, None, false, &Mm { bf16: false }).unwrap();
    assert_eq!(s(&l_with), s(&l_r7), "loss_and_grads_with routes the six parts to the fused step");
    bad.extend(check("R7", &varmap7, &m7, &ids, &tg, None, false));
    bad.extend(check_logits("R7", &m7, &ids, &tg, LOGITS_BAR, true));
    let n = 4 * cfg7.ctx;
    let mask: Vec<f32> = (0..n).map(|i| if (i % cfg7.ctx) >= cfg7.ctx / 2 || i % 7 == 0 { 1.0 } else { 0.0 }).collect();
    let w = Tensor::from_vec(mask, n, &dev).unwrap();
    bad.extend(check("R7 mask", &varmap7, &m7, &ids, &tg, Some(&w), false));
    let signed: Vec<f32> = (0..n).map(|i| [0.0, 1.0, -0.5, 0.25][i % 4] / n as f32).collect();
    let w = Tensor::from_vec(signed, n, &dev).unwrap();
    bad.extend(check("R7 raw", &varmap7, &m7, &ids, &tg, Some(&w), true));
    bad.extend(check_prune_stats("R7", &m7, &ids, &tg));
    {
        // printed only: N1's recipe + --neuromod 3 without the six, same seed and batch (the step's own spread)
        let (vm0, m0) = model(&dev, c1.with_arch(arch_of(&format!("{all} --neuromod 3"))).unwrap(), 9);
        perturb_nm(&vm0, &dev, 0.0);
        check_logits("calibration: R7 without the six", &m0, &ids, &tg, LOGITS_BAR, false);
    }
    // R8: long windows
    let c8 = Config::with_kind(Kind::D, g, 64, 2, 2048).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
    let cfg8 = c8.with_arch(arch_of(&format!("{all} {SIX} --neuromod 3"))).unwrap();
    assert!(fwin::step_ok(&cfg8), "R8's options must take the fused step");
    let (varmap8, m8) = model(&dev, cfg8, 11);
    perturb_nm(&varmap8, &dev, 0.0);
    perturb_r2(&varmap8, &m8, &dev);
    let (ids8, tg8) = batch(&dev, 1, cfg8.ctx, 111);
    bad.extend(check("R8", &varmap8, &m8, &ids8, &tg8, None, false));
    bad.extend(check_logits("R8", &m8, &ids8, &tg8, LOGITS_BAR, true));
    // R9: per-head gates, the plain LIF
    let cfg9 = Config::with_kind(Kind::D, VOCAB, 64, 2, 64).with_arch(arch_of(&format!("--head-size 32 --kwta 0.25 --mlp-act relu2 --lif-scale --value-residual {SIX}"))).unwrap();
    assert!(fwin::step_ok(&cfg9), "R9's options must take the fused step");
    let (varmap9, m9) = model(&dev, cfg9, 10);
    perturb_r2(&varmap9, &m9, &dev);
    for (name, var) in vars(&varmap9) {
        if name.ends_with(".mem.win") {
            // as G2: at d64 with --lif-scale the input currents are too small to fire
            var.set(&var.as_tensor().affine(4.0, 0.0).unwrap()).unwrap();
        }
    }
    let (ids9, tg9) = plain_batch(&dev, 4, cfg9.ctx, 101);
    bad.extend(check("R9", &varmap9, &m9, &ids9, &tg9, None, false));
    bad.extend(check_logits("R9", &m9, &ids9, &tg9, LOGITS_BAR, true));
    bad.extend(check_prune_stats("R9", &m9, &ids9, &tg9));
    // routing
    for nm in ["", " --neuromod 3"] {
        for extra in ["--dendrite", "--dopamine", "--mood", "--rhythm", "--prune", "--stp", SIX] {
            let c = c1.with_arch(arch_of(&format!("{all} {extra}{nm}"))).unwrap();
            assert!(fwin::step_ok(&c), "{extra}{nm} must take the fused step");
        }
        // merge_f: R and W too, alone and with the six (never W with Pr: Arch::validate refuses --prune with --columns)
        for extra in ["--letters", "--columns 4", &format!("{SIX} --letters"), "--dendrite --dopamine --mood --rhythm --stp --columns 4"] {
            let c = c1.with_arch(arch_of(&format!("{all} {extra}{nm}"))).unwrap();
            assert!(fwin::step_ok(&c), "{extra}{nm} must take the fused step (merged with v51 fast2)");
        }
    }
    assert!(bad.is_empty(), "round 2 step off: {bad:#?}");
}

/// F-1's reference for round 2 (CPU only, as f1_reference_is_the_reference_in_f64): R7's options but S (the synapse's
/// scan has no f64 path) in f32 (the usual CPU path) and in f64 give the same loss within 1e-5; the gradients' worst
/// agreement is printed (F-1's e_auto on the CPU).
#[test]
fn f1_round2_reference_in_f64() {
    let cpu = Device::Cpu;
    let flags = format!("{WINNERS} {SLOW} --dendrite --dopamine --mood --rhythm --prune --neuromod 3");
    let c = Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 128, 3, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
    let cfg = c.with_arch(arch_of(&flags)).unwrap();
    let varmap = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &cpu)).unwrap();
    perturb(&varmap, &cpu);
    perturb_nm(&varmap, &cpu, 0.0);
    perturb_r2(&varmap, &m, &cpu);
    let (ids, tg) = batch(&cpu, 2, cfg.ctx, 91);
    let (l32, _, g32) = m.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
    let (l64, _, g64) = f64_ref(cfg, &varmap, &ids, &tg, None, false);
    let by: std::collections::HashMap<String, Var> = vars(&varmap).into_iter().collect();
    let mut worst = (0.0f64, String::new());
    let mut r2 = 0;
    for (name, r) in &g64 {
        let e = rel64(g32.get(by[name].as_tensor()).unwrap_or_else(|| panic!("no f32 gradient for {name}")), r);
        if name.contains("v51") {
            r2 += 1;
            eprintln!("F-1 round 2 reference {name:<24} f32 vs f64 rel {e:.2e}");
        }
        if e > worst.0 {
            worst = (e, name.clone());
        }
    }
    eprintln!("F-1 round 2 validity: loss f32 {:.9} f64 {l64:.9}; {} gradients ({r2} of round 2), worst rel {:.2e} ({})", s(&l32), g64.len(), worst.0, worst.1);
    assert!(r2 > 0, "round 2's weights have gradients");
    assert!((s(&l32) as f64 - l64).abs() < 1e-5, "f32 {} vs f64 {l64}", s(&l32));
    let n32 = vars(&varmap).iter().filter(|(_, v)| g32.get(v.as_tensor()).is_some()).count();
    assert_eq!(g64.len(), n32, "the f64 reference has a gradient for every weight the f32 one has");
}

/// F-1 (CUDA) for round 2, printed (the fallback bar if R1-R9 miss the N bars on rounding): R7's options but S, the
/// same model and batch as R7, under F-1's rule (e_fused <= 2 e_auto + 1e-4 per parameter).
#[test]
fn f1_round2_against_f64() {
    let Some(dev) = cuda() else {
        eprintln!("F-1 round 2: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let flags = format!("{WINNERS} {SLOW} --dendrite --dopamine --mood --rhythm --prune --neuromod 3");
    let c = Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 128, 3, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
    let cfg = c.with_arch(arch_of(&flags)).unwrap();
    let (varmap, m) = model(&dev, cfg, 9);
    perturb_nm(&varmap, &dev, 0.0);
    perturb_r2(&varmap, &m, &dev);
    let (ids, tg) = batch(&dev, 4, cfg.ctx, 91);
    let bad = f1_case("R7-S", cfg, &varmap, &m, &ids, &tg, None, false);
    eprintln!("F-1 round 2 (R7 without S): {}", if bad.is_empty() { "every parameter within 2 e_auto + 1e-4" } else { "some parameters break the rule (listed)" });
    assert!(bad.is_empty(), "F-1 round 2: {bad:#?}");
}
