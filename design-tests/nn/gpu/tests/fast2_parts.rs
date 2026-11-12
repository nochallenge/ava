// TESTING / DESIGN ONLY — v51 fast2 gradient check (runs/v51/fast2_check.sh): round 2's R --letters, W --columns 4,
// TH --thalamus and CB --cerebellum-head on the hand-written WIN step (src/model/fwin.rs + src/model/fwin_f2.rs) against
// loss + candle's autograd, same model and batch. CUDA only (AVA_GPU_TEST_CUDA=1); tiny models, seconds of GPU.
//
// The bars are fast_winners.rs's (G1-G8, N1-N4), written before any run:
//   f32: loss within 1e-4, every parameter cos > 0.9999 and rel-norm-diff < 1e-2; logits rel-norm-diff < 1e-3
//   bf16: loss within 1e-2 (the gradients and logits printed only: rounding flips spikes)
// Cases (Ava 0.4b's block recipe, WIN + SLOW, with the glance; d128 L3 ctx 64 batch 4 unless said), the new parts'
// weights that start at 0 (TH's gates, CB's read-out, R's Wo) moved off it:
//   P1 R    --letters (a synthetic letter table of the glance vocabulary: 0..20 bytes per token)
//   P2 W    --columns 4
//   P3 TH   --thalamus
//   P4 CB   --cerebellum-head 4096 (32 d)
//   P5 ALL  the four + --neuromod 3 (the nucleus off its start too); also 0/1 and signed raw target weights
//   P6 ALL at d64 L2, head size 32 (per-head gates, the plain binary LIF), plain tokens, CB 2048
//   P7 ALL at d64 L2, ctx 2048, batch 1
//   routing: each takes the fused step (step_ok, win_step, loss_and_grads_with's loss = the fused step's); --loop with
//   --thalamus keeps autograd (merge_f: the other round-2 options take the fused step since gpu-fast1 was merged)
//   P8 ALL + --engram 1 + the magnitude code: printed only in gpu-fast2 (which lacked gpu-fast1's engram backward fix,
//   F-1); held at the same bars in the merged crate (runs/v51/merge_f.md)
// cb_d384_step_matches_autograd (below, v51 cbfix): P9, CB at d384 G 12288 (the shared-memory opt-in).
// merged_ten_parts_step_matches_autograd (below, merge_f): round 2's ten parts together with --neuromod 3.
// fast2_check.sh runs this test as built, with AVA_WIN_KEEP_PREP=0 and with AVA_F2_K=0 (the candle-op twins in the step
// instead of fwin_f2's kernels).

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
const SLOW: &str = "--gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";

fn noise(n: usize, seed: u64, amp: f32) -> Vec<f32> {
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

/// fast_winners.rs's `perturb` (the weights that start at 0 moved off it), plus the new parts': TH's gates and biases,
/// CB's read-out, R's output projection, W's low-rank paths, and the nucleus's (as perturb_nm).
fn perturb(varmap: &VarMap, dev: &Device) {
    for (k, (name, var)) in vars(varmap).into_iter().enumerate() {
        let leaf = name.rsplit('.').next().unwrap_or(&name);
        let leaf = if leaf == "wo" && name.contains(".eng.") { "eng_wo" } else { leaf };
        let t = var.as_tensor();
        let (amp, off, add) = match (name.as_str(), leaf) {
            ("th.gate", _) => (0.8, 0.0, false),
            ("th.bias", _) => (0.5, 0.0, false),
            ("cbh.wo", _) => (0.05, 0.0, false),
            ("v51.letters.wo", _) => (0.1, 0.0, false),
            ("nm.norm", _) => (0.3, 1.0, false),
            ("nm.wn", _) => (0.1, 0.0, false),
            ("nm.bn", _) => (0.5, 0.0, false),
            (_, "col_b1" | "col_b2") => (0.05, 0.0, true),
            (_, "dense") => (0.1, 0.0, false),
            (_, "vres") => (0.3, 0.0, false),
            (_, "place_emb") => (0.05, 0.0, false),
            (_, "wa" | "wb" | "wa2" | "wb2" | "we" | "eng_wo" | "mag_w") => (0.05, 0.0, false),
            (_, "theta" | "again") => (0.3, 0.0, true),
            (_, "ba" | "bb") => (0.5, 0.0, false),
            _ => continue,
        };
        let v: Vec<f32> = noise(t.elem_count(), 1000 + k as u64, amp).iter().map(|x| x + off).collect();
        let v = Tensor::from_vec(v, t.shape(), dev).unwrap();
        var.set(&if add { t.add(&v).unwrap() } else { v }).unwrap();
    }
}

/// A letter table of `vocab` tokens for --letters (0..20 bytes each, some empty), set before the model is built.
fn letters(vocab: usize, seed: u64) {
    let r = noise(vocab * 22, seed, 1.0);
    let t: Vec<Vec<u8>> = (0..vocab)
        .map(|v| {
            let len = ((r[v * 22] + 1.0) * 10.5) as usize % 21;
            (0..len).map(|j| (((r[v * 22 + 1 + j] + 1.0) * 128.0) as usize).min(255) as u8).collect()
        })
        .collect();
    ava_gpu::v51::set_letters(t);
}

fn model(dev: &Device, cfg: Config, seed: u64) -> (VarMap, Model) {
    dev.set_seed(seed).unwrap();
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

/// Every parameter's agreement (fast_winners.rs's compare); `hold`: the failures against the f32 bars.
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
        eprintln!("{tag} {name:<28} cos {cos:.7} rel-norm-diff {rel:.2e} max-rel {maxrel:.2e}");
        if hold && !(cos > 0.9999 && rel < 1e-2) {
            bad.push(format!("{tag} {name}: cos {cos} rel-norm-diff {rel}"));
        }
    }
    let mean = coss.iter().sum::<f32>() / coss.len().max(1) as f32;
    eprintln!("{tag}: {} parameters, mean cos {mean:.6}, lowest {:.6} ({}, rel {:.2e})", coss.len(), worst.0, worst.2, worst.1);
    bad
}

/// The fused step (f32 and bf16) against autograd; `hold` false: printed only.
fn check(tag: &str, varmap: &VarMap, model: &Model, ids: &Tensor, tg: &Tensor, wt: Option<&Tensor>, raw: bool, hold: bool) -> Vec<String> {
    let (l_ref, _, g_ref) = model.loss_and_grads_autograd(ids, tg, wt, raw).unwrap();
    let rates_ref = model.homeostasis_rates().unwrap();
    let x = s(&l_ref);
    let mut bad = Vec::new();
    for bf16 in [false, true] {
        model.set_homeostasis_rates(&vec![-1.0; rates_ref.len()]).unwrap();
        let (l_new, _, g_new) = model.loss_and_grads_win(ids, tg, wt, raw, &Mm { bf16 }).unwrap();
        let y = s(&l_new);
        let tol = if bf16 { 1e-2 } else { 1e-4 };
        eprintln!("{tag} bf16={bf16}: loss autograd {x:.6} fused {y:.6} (|diff| {:.2e}, bar {tol:.0e})", (x - y).abs());
        if hold && !((x - y).abs() < tol) {
            bad.push(format!("{tag} bf16={bf16}: loss {x} vs fused {y}"));
        }
        bad.extend(compare(&format!("{tag} bf16={bf16}"), varmap, &g_ref, &g_new, hold && !bf16));
        let rates = model.homeostasis_rates().unwrap();
        if !rates_ref.is_empty() || !rates.is_empty() {
            let dmax = rates.iter().zip(&rates_ref).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            eprintln!("{tag} bf16={bf16}: homeostasis rates {} vs {}, max |diff| {dmax:.2e}", rates.len(), rates_ref.len());
            if hold && (rates.len() != rates_ref.len() || (!bf16 && dmax > 1e-6)) {
                bad.push(format!("{tag} bf16={bf16}: homeostasis rates off ({} vs {}, max diff {dmax})", rates.len(), rates_ref.len()));
            }
        }
    }
    bad
}

const LOGITS_BAR: f32 = 1e-3;

fn check_logits(tag: &str, model: &Model, ids: &Tensor, tg: &Tensor, hold: bool) -> Vec<String> {
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
        let note = if held { format!(" (bar {LOGITS_BAR:.0e})") } else { " (printed only)".to_string() };
        eprintln!("{tag} bf16={bf16}: logits rel-norm-diff {rel:.2e}, max |diff| {maxd:.2e}{note}");
        if held && !(rel < LOGITS_BAR) {
            bad.push(format!("{tag} bf16={bf16}: logits rel-norm-diff {rel}"));
        }
    }
    bad
}

/// One case: the routing, the step against autograd, the logits.
fn case(tag: &str, dev: &Device, cfg: Config, seed: u64, ids: &Tensor, tg: &Tensor, hold: bool) -> Vec<String> {
    let mut bad = Vec::new();
    if !fwin::step_ok(&cfg) {
        return vec![format!("{tag}: step_ok refuses {}", cfg.arch.signature())];
    }
    let (varmap, m) = model(dev, cfg, seed);
    let fused_on = !std::env::var("AVA_WIN_FUSED").is_ok_and(|v| v == "0");
    if fused_on && !m.win_step(ids) {
        bad.push(format!("{tag}: win_step says no"));
    }
    let (l_with, _, _) = m.loss_and_grads_with(ids, tg, false).unwrap().expect("loss_and_grads_with: a step");
    let (l_win, _, _) = m.loss_and_grads_win(ids, tg, None, false, &Mm { bf16: false }).unwrap();
    if fused_on && s(&l_with) != s(&l_win) {
        bad.push(format!("{tag}: loss_and_grads_with did not route to the fused step ({} vs {})", s(&l_with), s(&l_win)));
    }
    eprintln!("{tag}: {} ({}); routing: fused step {}", cfg.arch.flags(), if hold { "held" } else { "printed only" }, fused_on && bad.is_empty());
    bad.extend(check(tag, &varmap, &m, ids, tg, None, false, hold));
    bad.extend(check_logits(tag, &m, ids, tg, hold));
    bad
}

#[test]
fn round2_parts_step_matches_autograd() {
    let Some(dev) = cuda() else {
        eprintln!("fast2 parts step: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let g = ava_data::numbers::VOCAB_GLANCE;
    letters(g, 7);
    letters(VOCAB, 8);
    let all = format!("{WINNERS} {SLOW}");
    let c128 = |extra: &str| Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_arch(arch_of(&format!("{all} {extra}"))).unwrap();
    let (ids, tg) = batch(&dev, 4, 64, 91);
    let mut bad: Vec<String> = Vec::new();
    let four = "--letters --columns 4 --thalamus --cerebellum-head 4096";
    for (tag, extra) in [("P1 R", "--letters"), ("P2 W", "--columns 4"), ("P3 TH", "--thalamus"), ("P4 CB", "--cerebellum-head 4096")] {
        bad.extend(case(tag, &dev, c128(extra), 9, &ids, &tg, true));
    }
    let cfg5 = c128(&format!("{four} --neuromod 3"));
    bad.extend(case("P5 ALL", &dev, cfg5, 9, &ids, &tg, true));
    {
        let (varmap, m) = model(&dev, cfg5, 9);
        let n = 4 * 64;
        let mask: Vec<f32> = (0..n).map(|i| if (i % 64) >= 32 || i % 7 == 0 { 1.0 } else { 0.0 }).collect();
        let w = Tensor::from_vec(mask, n, &dev).unwrap();
        bad.extend(check("P5 ALL mask", &varmap, &m, &ids, &tg, Some(&w), false, true));
        let signed: Vec<f32> = (0..n).map(|i| [0.0, 1.0, -0.5, 0.25][i % 4] / n as f32).collect();
        let w = Tensor::from_vec(signed, n, &dev).unwrap();
        bad.extend(check("P5 ALL raw", &varmap, &m, &ids, &tg, Some(&w), true, true));
    }
    // P6: head size 32, per-head gates, the plain LIF (as N3; its inputs scaled up so it fires)
    let cfg6 = Config::with_kind(Kind::D, VOCAB, 64, 2, 64)
        .with_arch(arch_of("--head-size 32 --kwta 0.25 --mlp-act relu2 --lif-scale --value-residual --letters --columns 4 --thalamus --cerebellum-head 2048 --neuromod 3"))
        .unwrap();
    {
        let (ids6, tg6) = plain_batch(&dev, 4, 64, 101);
        if !fwin::step_ok(&cfg6) {
            bad.push("P6: step_ok refuses it".into());
        } else {
            let (varmap6, m6) = model(&dev, cfg6, 10);
            for (name, var) in vars(&varmap6) {
                if name.ends_with(".mem.win") {
                    var.set(&var.as_tensor().affine(4.0, 0.0).unwrap()).unwrap();
                }
            }
            bad.extend(check("P6 ALL hs32", &varmap6, &m6, &ids6, &tg6, None, false, true));
            bad.extend(check_logits("P6 ALL hs32", &m6, &ids6, &tg6, true));
        }
    }
    // P7: long windows
    let cfg7 = Config::with_kind(Kind::D, g, 64, 2, 2048).with_glance().unwrap().with_arch(arch_of(&format!("{all} --letters --columns 4 --thalamus --cerebellum-head 2048 --neuromod 3"))).unwrap();
    let (ids7, tg7) = batch(&dev, 1, 2048, 111);
    bad.extend(case("P7 ALL ctx2048", &dev, cfg7, 11, &ids7, &tg7, true));
    // P8: with the engram read and the magnitude code. merge_f (runs/v51/merge_f.md): the merged crate has gpu-fast1's
    // F-1 fix (the f32 step's engram backward stores f32, attn_bwd_sf), so P8 is held at the same bars as P1-P7 now
    let cfg8 = Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap().with_arch(arch_of(&format!("{all} {four} --neuromod 3"))).unwrap();
    bad.extend(case("P8 ALL engram", &dev, cfg8, 9, &ids, &tg, true));
    // routing. merge_f: with v51 fast1 merged, the four with fast1's parts take the fused step too; --loop keeps autograd
    // (--columns with --prune is not a model: Arch::validate refuses it)
    for extra in ["--letters --dendrite", "--columns 4 --dopamine", "--cerebellum-head 4096 --mood", "--thalamus --rhythm"] {
        let c = Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_arch(arch_of(&format!("{all} {extra}"))).unwrap();
        if !fwin::step_ok(&c) {
            bad.push(format!("{extra}: must take the fused step (merged)"));
        }
    }
    for extra in ["--thalamus --loop 2", "--columns 4 --prune"] {
        if let Ok(c) = Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_arch(arch_of(&format!("{all} {extra}"))) {
            if fwin::step_ok(&c) {
                bad.push(format!("{extra}: must keep the autograd path"));
            }
        }
    }
    assert!(bad.is_empty(), "fast2 parts step off: {bad:#?}");
}

// v51 cbfix (runs/v51/cbfix.md, runs/v51/cbfix_check.sh): CB at the final size's head, d384 G 12288 (32 d), whose rows
// need the shared-memory opt-in both in the step's kernel (f2_cb_code) and in autograd's kWTA (kwta_mask_rows, which
// failed at the launch with "invalid argument" before the fix). Bars: the N bars of P1-P8, written before any run.
//   P9 CB d384 L2 ctx 64 batch 2, --cerebellum-head 12288, on P1-P8's recipe and glance vocabulary; routed to the
//   fused step (step_ok, win_step on this device, loss_and_grads_with's loss = the fused step's)
#[test]
fn cb_d384_step_matches_autograd() {
    let Some(dev) = cuda() else {
        eprintln!("fast2 CB d384: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let g = ava_data::numbers::VOCAB_GLANCE;
    let cfg = Config::with_kind(Kind::D, g, 384, 2, 64).with_glance().unwrap().with_arch(arch_of(&format!("{WINNERS} {SLOW} --cerebellum-head 12288"))).unwrap();
    let (ids, tg) = batch(&dev, 2, 64, 131);
    let bad = case("P9 CB d384 G12288", &dev, cfg, 12, &ids, &tg, true);
    assert!(bad.is_empty(), "fast2 CB d384 step off: {bad:#?}");
}

// ---- merge_f (runs/v51/merge_f.md, runs/v51/fast_merged_check.sh): round 2's ten parts together with --neuromod 3.
// All ten at once is not a model (Arch::validate refuses --prune with --columns), so two sets, each + --neuromod 3:
//   A = all but W: --dendrite --stp --letters --dopamine --mood --prune --rhythm --thalamus --cerebellum-head G
//   B = all but Pr: --dendrite --stp --letters --dopamine --mood --columns 4 --rhythm --thalamus --cerebellum-head G
// Bars, written before any run (the N bars, as P1-P8 here and R1-R9 in fast_winners.rs): f32 loss within 1e-4, every
// parameter cos > 0.9999 and rel-norm-diff < 1e-2, logits rel-norm-diff < 1e-3; bf16 loss within 1e-2 (the rest
// printed); each case routed to the fused step (step_ok, win_step, loss_and_grads_with's loss = the fused step's); Pr's
// statistics within 1e-4 (set A). S and fast2's parts have no f64 path, so F-1's fallback cannot cover these cases.
// Every part's weights off its start: fast2's `perturb` (in `model`), then fast_winners.rs's perturb_r2 for fast1's
// parts (the letter tables and W's paths left to `perturb`), every third MLP unit cut (Pr).
//   M1 A and M2 B on N1's / P8's model (Ava 0.4b's recipe, glance, mag, --engram 1; d128 L3 ctx 64 batch 4, CB 4096),
//      mean, 0/1 and signed raw target weights
//   M3 A and M4 B at d64 L2, ctx 2048, batch 1 (glance, CB 2048)
//   M5 B at d64 L2, head size 32 (per-head gates, the plain binary LIF), plain tokens, CB 2048

const SET_A: &str = "--dendrite --stp --letters --dopamine --mood --prune --rhythm --thalamus";
const SET_B: &str = "--dendrite --stp --letters --dopamine --mood --columns 4 --rhythm --thalamus";

/// fast_winners.rs's perturb_r2 for fast1's parts: v51's weights off their start (the letter tables and W's paths are
/// fast2's, moved by `perturb`), every third unit cut (Pr), the synapse's mix and the rhythms' resets moved.
fn perturb_r2(varmap: &VarMap, model: &Model, dev: &Device) {
    for (k, (name, var)) in vars(varmap).into_iter().enumerate() {
        let t = var.as_tensor();
        let nz = |amp: f32| Tensor::from_vec(noise(t.elem_count(), 3000 + k as u64, amp), t.shape(), dev).unwrap();
        let new = if name.ends_with(".v51.alive") {
            Tensor::from_vec((0..t.elem_count()).map(|j| if j % 3 == 0 { 0f32 } else { 1.0 }).collect::<Vec<f32>>(), t.shape(), dev).unwrap()
        } else if name.ends_with(".v51.use") || name.contains(".letters.") || name.contains(".col_") {
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

/// fast_winners.rs's check_prune_stats: Pr's statistics (v51::after_step's input) of the fused f32 step against the
/// reference's, max |diff| / max |ref| <= 1e-4.
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

/// One merged case: the routing, the step against autograd (mean weights; with `weights` also 0/1 and signed raw), the
/// logits, Pr's statistics.
fn case10(tag: &str, dev: &Device, cfg: Config, seed: u64, ids: &Tensor, tg: &Tensor, weights: bool) -> Vec<String> {
    let mut bad = Vec::new();
    if !fwin::step_ok(&cfg) {
        return vec![format!("{tag}: step_ok refuses {}", cfg.arch.signature())];
    }
    let (varmap, m) = model(dev, cfg, seed);
    perturb_r2(&varmap, &m, dev);
    let fused_on = !std::env::var("AVA_WIN_FUSED").is_ok_and(|v| v == "0");
    if fused_on && !m.win_step(ids) {
        bad.push(format!("{tag}: win_step says no"));
    }
    let (l_with, _, _) = m.loss_and_grads_with(ids, tg, false).unwrap().expect("loss_and_grads_with: a step");
    let (l_win, _, _) = m.loss_and_grads_win(ids, tg, None, false, &Mm { bf16: false }).unwrap();
    if fused_on && s(&l_with) != s(&l_win) {
        bad.push(format!("{tag}: loss_and_grads_with did not route to the fused step ({} vs {})", s(&l_with), s(&l_win)));
    }
    eprintln!("{tag}: {} (held); routing: fused step {}", cfg.arch.flags(), fused_on && bad.is_empty());
    bad.extend(check(tag, &varmap, &m, ids, tg, None, false, true));
    bad.extend(check_logits(tag, &m, ids, tg, true));
    if weights {
        let n = ids.elem_count();
        let t = ids.dim(1).unwrap();
        let mask: Vec<f32> = (0..n).map(|i| if (i % t) >= t / 2 || i % 7 == 0 { 1.0 } else { 0.0 }).collect();
        let w = Tensor::from_vec(mask, n, dev).unwrap();
        bad.extend(check(&format!("{tag} mask"), &varmap, &m, ids, tg, Some(&w), false, true));
        let signed: Vec<f32> = (0..n).map(|i| [0.0, 1.0, -0.5, 0.25][i % 4] / n as f32).collect();
        let w = Tensor::from_vec(signed, n, dev).unwrap();
        bad.extend(check(&format!("{tag} raw"), &varmap, &m, ids, tg, Some(&w), true, true));
    }
    bad.extend(check_prune_stats(tag, &m, ids, tg));
    bad
}

#[test]
fn merged_ten_parts_step_matches_autograd() {
    let Some(dev) = cuda() else {
        eprintln!("merged ten parts step: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let g = ava_data::numbers::VOCAB_GLANCE;
    letters(g, 7);
    letters(VOCAB, 8);
    let all = format!("{WINNERS} {SLOW}");
    let mut bad: Vec<String> = Vec::new();
    let c1 = Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
    let (ids, tg) = batch(&dev, 4, 64, 91);
    for (tag, set) in [("M1 A (all but W)", SET_A), ("M2 B (all but Pr)", SET_B)] {
        let cfg = c1.with_arch(arch_of(&format!("{all} {set} --cerebellum-head 4096 --neuromod 3"))).unwrap();
        bad.extend(case10(tag, &dev, cfg, 9, &ids, &tg, true));
    }
    let c7 = Config::with_kind(Kind::D, g, 64, 2, 2048).with_glance().unwrap();
    let (ids7, tg7) = batch(&dev, 1, 2048, 111);
    for (tag, set) in [("M3 A ctx2048", SET_A), ("M4 B ctx2048", SET_B)] {
        let cfg = c7.with_arch(arch_of(&format!("{all} {set} --cerebellum-head 2048 --neuromod 3"))).unwrap();
        bad.extend(case10(tag, &dev, cfg, 11, &ids7, &tg7, false));
    }
    {
        // M5: head size 32, per-head gates, the plain LIF (as P6 and R9; its inputs scaled up so it fires)
        let cfg = Config::with_kind(Kind::D, VOCAB, 64, 2, 64)
            .with_arch(arch_of(&format!("--head-size 32 --kwta 0.25 --mlp-act relu2 --lif-scale --value-residual {SET_B} --cerebellum-head 2048 --neuromod 3")))
            .unwrap();
        let (ids5, tg5) = plain_batch(&dev, 4, 64, 101);
        if !fwin::step_ok(&cfg) {
            bad.push("M5: step_ok refuses it".into());
        } else {
            let (varmap5, m5) = model(&dev, cfg, 10);
            perturb_r2(&varmap5, &m5, &dev);
            for (name, var) in vars(&varmap5) {
                if name.ends_with(".mem.win") {
                    var.set(&var.as_tensor().affine(4.0, 0.0).unwrap()).unwrap();
                }
            }
            bad.extend(check("M5 B hs32", &varmap5, &m5, &ids5, &tg5, None, false, true));
            bad.extend(check_logits("M5 B hs32", &m5, &ids5, &tg5, true));
        }
    }
    // routing: what neither builder fused keeps autograd
    for extra in [format!("{SET_A} --cerebellum-head 4096 --neuromod 3 --loop 2"), format!("{SET_B} --cerebellum-head 4096 --dale 0.8"), format!("{SET_A} --cerebellum-head 20512 --neuromod 3")] {
        if let Ok(c) = c1.with_arch(arch_of(&format!("{all} {extra}"))) {
            if fwin::step_ok(&c) {
                bad.push(format!("{extra}: must keep the autograd path"));
            }
        }
    }
    assert!(bad.is_empty(), "merged ten parts step off: {bad:#?}");
}
