// TESTING / DESIGN ONLY — v52 fix_Pr (runs/v52/fix_Pr.md): Pr --prune on the fused WIN step at v52 Stage 1's shape
// (s5-v42's block options, the magnitude code, the engram; d384, 8 blocks, ctx 2048), CUDA only (AVA_GPU_TEST_CUDA=1).
//
// pr_with_nothing_cut_is_b_bit_for_bit: Pr with every unit alive (as every v52 grow run was: grow_gpu never set the
//   schedule) against B with the same weights and batch: the same loss and every shared gradient bit for bit, in f32 and
//   in bf16; Pr's own variables (the cut mask, the use averages) get no gradient. Before the fix Pr made its MLP units
//   in f32 (fwin_r2::mlp_f32) where B makes them in bf16, so its bf16 gradients differed from B's from the first step
//   (f32 bit for bit). Also the cost of one step of each (ms, median of 5 after 2 warm-ups).
// pr_step_matches_autograd_at_d384: B, Pr with none cut and Pr with half of every block's units cut, at v52's width and
//   depth (d384, 8 blocks, engram in block 4, the magnitude code) on a short window (ctx 64, batch 4), the fused step
//   against candle's autograd on the same device and weights (one captured step, the reference inside a test): f32 at
//   fast2_parts.rs's bars (loss within 1e-4, every parameter cos > 0.9999 and rel-norm-diff < 1e-2; Pr's statistics
//   within 1e-4 after up to 4 kWTA boundary swaps per block, each within 1e-2: amended 11-07, see stats_diff), bf16 the loss within 1e-2 and the gradients printed (rounding flips spikes), as every check before.
// v52_window_fused_vs_autograd_sensitivity (printed only): the same at the full window (ctx 2048, batch 1) and by
//   window length, with the yardstick of how far autograd itself (and the fused step itself) moves when every weight is
//   nudged by 1e-6 of itself. A first run held the f32 bars at ctx 2048 and B itself failed them (lowest cos 0.56,
//   b7.hebb.bb; Pr the same numbers): runs/v52/fix_Pr.md.

use ava_gpu::arch::Arch;
use ava_gpu::model::{Config, Kind, Mm, Model};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use std::time::Instant;

/// runs/v42/wave3/s5-v42/arch.txt, the block options of every v52 Stage 1 arm
const S5: &str = "--head-size 64 --timescales fixed --gates channel --value-residual --kwta 1/4 --mlp-act relu2 --mem-momentum 0.5 \
                  --resid-fix --residual dense --lif-init logspaced --lif-scale --spikes int2 --alif --homeostasis 0.1";

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

/// fast2_parts.rs's perturb for the weights of this recipe that start at 0 (so every path carries a gradient).
fn perturb(varmap: &VarMap, dev: &Device) {
    for (k, (name, var)) in vars(varmap).into_iter().enumerate() {
        let leaf = name.rsplit('.').next().unwrap_or(&name);
        let leaf = if leaf == "wo" && name.contains(".eng.") { "eng_wo" } else { leaf };
        let t = var.as_tensor();
        let (amp, off, add) = match leaf {
            "dense" => (0.1, 0.0, false),
            "vres" => (0.3, 0.0, false),
            "wa" | "wb" | "wa2" | "wb2" | "we" | "eng_wo" | "mag_w" => (0.05, 0.0, false),
            "theta" | "again" => (0.3, 0.0, true),
            "ba" | "bb" => (0.5, 0.0, false),
            _ => continue,
        };
        let v: Vec<f32> = noise(t.elem_count(), 1000 + k as u64, amp).iter().map(|x| x + off).collect();
        let v = Tensor::from_vec(v, t.shape(), dev).unwrap();
        var.set(&if add { t.add(&v).unwrap() } else { v }).unwrap();
    }
}

fn cfg(d: usize, layers: usize, ctx: usize, engram: usize, extra: &str) -> Config {
    Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, d, layers, ctx)
        .with_glance()
        .unwrap()
        .with_mag()
        .unwrap()
        .with_engram(engram)
        .unwrap()
        .with_arch(arch_of(&format!("{S5} {extra}")))
        .unwrap()
}

fn model(dev: &Device, cfg: Config, seed: u64) -> (VarMap, Model) {
    dev.set_seed(seed).unwrap();
    let varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, dev)).unwrap();
    (varmap, model)
}

/// B and Pr with the same weights: B's (perturbed) copied into Pr by name; Pr's own are only its cut mask (all alive)
/// and its use averages (0).
fn pair(dev: &Device, d: usize, layers: usize, ctx: usize, engram: usize) -> ((VarMap, Model), (VarMap, Model)) {
    let (vb, mb) = model(dev, cfg(d, layers, ctx, engram, ""), 9);
    perturb(&vb, dev);
    let (vp, mp) = model(dev, cfg(d, layers, ctx, engram, "--prune"), 9);
    let pm = vp.data().lock().unwrap();
    for (name, var) in vars(&vb) {
        pm.get(&name).unwrap_or_else(|| panic!("Pr lacks {name}")).set(var.as_tensor()).unwrap();
    }
    let extra: Vec<&String> = pm.keys().filter(|k| !vb.data().lock().unwrap().contains_key(*k)).collect();
    assert!(extra.iter().all(|k| k.ends_with(".v51.alive") || k.ends_with(".v51.use")), "Pr's own variables: {extra:?}");
    drop(pm);
    mp.v51_schedule(0, 0);
    ((vb, mb), (vp, mp))
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

fn s(t: &Tensor) -> f32 {
    t.to_dtype(DType::F32).unwrap().to_scalar::<f32>().unwrap()
}

fn grad_f32(g: &GradStore, v: &Var) -> Option<Tensor> {
    g.get(v.as_tensor()).map(|t| t.to_dtype(DType::F32).unwrap().flatten_all().unwrap())
}

/// B's and Pr's gradients of every shared weight: (how many differ, the worst max |diff| / max |B's| and its name).
fn diff_grads(vb: &VarMap, gb: &GradStore, vp: &VarMap, gp: &GradStore) -> (usize, usize, f32, String) {
    let pm = vp.data().lock().unwrap();
    let (mut n, mut differ, mut worst, mut wname) = (0, 0, 0f32, String::new());
    for (name, var) in vars(vb) {
        let (a, b) = (grad_f32(gb, &var), grad_f32(gp, &pm[&name]));
        match (a, b) {
            (None, None) => continue,
            (Some(a), Some(b)) => {
                n += 1;
                let d = s(&a.sub(&b).unwrap().abs().unwrap().max(0).unwrap());
                if d != 0.0 {
                    differ += 1;
                    let r = d / s(&a.abs().unwrap().max(0).unwrap()).max(1e-30);
                    if r > worst {
                        (worst, wname) = (r, name.clone());
                    }
                }
            }
            (a, b) => panic!("{name}: a gradient in one model only (B {}, Pr {})", a.is_some(), b.is_some()),
        }
    }
    (n, differ, worst, wname)
}

/// ms per fused step (median of 5 after 2 warm-ups); reading the loss waits for every kernel queued before it.
fn time_step(m: &Model, ids: &Tensor, tg: &Tensor, mm: &Mm) -> f64 {
    let mut ts = Vec::new();
    for i in 0..7 {
        let t0 = Instant::now();
        let (l, _, g) = m.loss_and_grads_win(ids, tg, None, false, mm).unwrap();
        s(&l);
        drop(g);
        if i >= 2 {
            ts.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    ts.sort_by(f64::total_cmp);
    ts[ts.len() / 2]
}

#[test]
fn pr_with_nothing_cut_is_b_bit_for_bit() {
    let Some(dev) = cuda() else {
        eprintln!("pr nothing cut: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let mut bad = Vec::new();
    for (tag, d, layers, ctx, rows, engram) in [("small d128 L3 ctx64 b4", 128, 3, 64, 4, 1), ("v52 d384 L8 ctx2048 b4", 384, 8, 2048, 4, 4)] {
        let ((vb, mb), (vp, mp)) = pair(&dev, d, layers, ctx, engram);
        assert!(mb.win_step(&batch(&dev, 1, 8, 1).0) && mp.win_step(&batch(&dev, 1, 8, 1).0), "{tag}: both take the fused step");
        let (ids, tg) = batch(&dev, rows, ctx, 91);
        for bf16 in [false, true] {
            let mm = Mm { bf16 };
            let (lb, _, gb) = mb.loss_and_grads_win(&ids, &tg, None, false, &mm).unwrap();
            let (lp, _, gp) = mp.loss_and_grads_win(&ids, &tg, None, false, &mm).unwrap();
            let (lb, lp) = (s(&lb), s(&lp));
            let (n, differ, worst, wname) = diff_grads(&vb, &gb, &vp, &gp);
            let own = vars(&vp).into_iter().filter(|(k, _)| k.contains(".v51.")).filter(|(_, v)| gp.get(v.as_tensor()).is_some()).count();
            eprintln!(
                "{tag} bf16={bf16}: loss B {lb:.7} Pr {lp:.7} ({}); {n} gradients, {differ} differ (worst max-rel {worst:.2e} {wname}); Pr's own with a gradient: {own}",
                if lb.to_bits() == lp.to_bits() { "same bits" } else { "DIFFER" }
            );
            if lb.to_bits() != lp.to_bits() || differ > 0 || own > 0 {
                bad.push(format!("{tag} bf16={bf16}: loss {lb} vs {lp}, {differ} of {n} gradients differ (worst {worst:.2e} {wname}), {own} own"));
            }
        }
        let mm = Mm { bf16: true };
        let (tb, tp) = (time_step(&mb, &ids, &tg, &mm), time_step(&mp, &ids, &tg, &mm));
        eprintln!("{tag} bf16: one fused step B {tb:.1} ms, Pr {tp:.1} ms ({:+.1}%), {:.0} vs {:.0} tokens/s", (tp / tb - 1.0) * 100.0, (rows * ctx) as f64 / tb * 1e3, (rows * ctx) as f64 / tp * 1e3);
    }
    assert!(bad.is_empty(), "Pr with nothing cut is not B: {bad:#?}");
}

/// Every parameter's agreement with the reference (fast2_parts.rs's compare): the failures against the f32 bars when
/// `hold`, and the lowest cosine.
fn compare(tag: &str, varmap: &VarMap, g_ref: &GradStore, g_new: &GradStore, hold: bool) -> (Vec<String>, f32) {
    let mut bad = Vec::new();
    let mut worst = (1.0f32, 0.0f32, String::new());
    for (name, var) in vars(varmap) {
        let (r, m) = (grad_f32(g_ref, &var), grad_f32(g_new, &var));
        let (r, m) = match (r, m) {
            (None, None) => continue,
            (Some(r), Some(m)) => (r, m),
            (r, m) => {
                bad.push(format!("{tag} {name}: gradient in autograd {} fused {}", r.is_some(), m.is_some()));
                continue;
            }
        };
        let nr = s(&r.sqr().unwrap().sum_all().unwrap()).sqrt();
        let nm = s(&m.sqr().unwrap().sum_all().unwrap()).sqrt();
        if nr == 0.0 && nm == 0.0 {
            continue;
        }
        let cos = s(&r.mul(&m).unwrap().sum_all().unwrap()) / (nr * nm).max(1e-30);
        let rel = s(&r.sub(&m).unwrap().sqr().unwrap().sum_all().unwrap()).sqrt() / nr.max(1e-30);
        if cos < worst.0 {
            worst = (cos, rel, name.clone());
        }
        if hold && !(cos > 0.9999 && rel < 1e-2) {
            bad.push(format!("{tag} {name}: cos {cos} rel-norm-diff {rel}"));
        }
    }
    eprintln!("{tag}: lowest cos {:.6} ({}, rel-norm-diff {:.2e})", worst.0, worst.2, worst.1);
    (bad, worst.0)
}

/// Pr's statistics of the fused step against the reference's, per block as |diff| / max |ref|: (the largest over every
/// unit, the largest once each block's SWAPS largest are set aside). Merge check 11-07 (runs/v52/merge_tests_try2.log):
/// in f32 with none cut, block 7 had exactly two units off, by the same amount in opposite directions (unit 318
/// -2.49e-4, unit 681 +2.49e-4 of the max; every other unit within 1e-5): one row's kWTA boundary, where two units of
/// equal value swapped places on rounding (a unit's mean |a| moves by its value / rows). A few such swaps per block are
/// allowed (each still within 1e-2); every other unit is held at 1e-4.
const SWAPS: usize = 4;
fn stats_diff(m: &Model, ids: &Tensor, tg: &Tensor, mm: &Mm) -> (f32, f32) {
    m.loss_and_grads_autograd(ids, tg, None, false).unwrap();
    let r = ava_gpu::fwin_r2::prune_stats_now(m);
    m.loss_and_grads_win(ids, tg, None, false, mm).unwrap();
    let f = ava_gpu::fwin_r2::prune_stats_now(m);
    let (mut worst, mut rest) = (0f32, 0f32);
    for (i, (a, b)) in r.iter().zip(&f).enumerate() {
        let (Some(a), Some(b)) = (a, b) else { panic!("block {i}: Pr statistics missing (reference {}, fused {})", a.is_some(), b.is_some()) };
        let (av, bv) = (a.to_vec1::<f32>().unwrap(), b.to_vec1::<f32>().unwrap());
        let mx = av.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
        let mut d: Vec<(f32, usize)> = av.iter().zip(&bv).enumerate().map(|(j, (x, y))| ((x - y).abs() / mx, j)).collect();
        d.sort_by(|p, q| q.0.total_cmp(&p.0));
        if d[0].0 > 1e-5 {
            // which units differ, and by how much
            let over = d.iter().filter(|p| p.0 > 1e-5).count();
            let top: Vec<String> = d.iter().take(3).map(|&(r, j)| format!("unit {j}: ref {:.6e} fused {:.6e} (rel {r:.2e})", av[j], bv[j])).collect();
            eprintln!("  block {i}: {over} of {} units off by > 1e-5 of the max; {}", av.len(), top.join("; "));
        }
        worst = worst.max(d[0].0);
        rest = rest.max(d.get(SWAPS).map_or(0.0, |p| p.0));
    }
    (worst, rest)
}

/// One model's fused step (f32 and bf16) against candle's autograd on the same device and weights (one captured step):
/// the failures against the bars when `hold` (f32: loss within 1e-4, every parameter cos > 0.9999 and rel-norm-diff
/// < 1e-2, Pr's statistics within 1e-4; bf16: the loss within 1e-2, the gradients printed only).
fn vs_autograd(tag: &str, vm: &VarMap, m: &Model, ids: &Tensor, tg: &Tensor, pr: bool, hold: bool) -> Vec<String> {
    let mut bad = Vec::new();
    if pr {
        m.v51_schedule(0, 1_000_000); // the statistics are kept while a schedule runs (nothing due in this one step)
    }
    let (l_ref, _, g_ref) = m.loss_and_grads_autograd(ids, tg, None, false).unwrap();
    let x = s(&l_ref);
    for bf16 in [false, true] {
        let mm = Mm { bf16 };
        let (l, _, g) = m.loss_and_grads_win(ids, tg, None, false, &mm).unwrap();
        let y = s(&l);
        let tol = if bf16 { 1e-2 } else { 1e-4 };
        let t = format!("{tag} bf16={bf16}");
        eprintln!("{t}: loss autograd {x:.6} fused {y:.6} (|diff| {:.2e}, bar {tol:.0e}{})", (x - y).abs(), if hold { "" } else { ", printed only" });
        if hold && !((x - y).abs() < tol) {
            bad.push(format!("{t}: loss {x} vs {y}"));
        }
        bad.extend(compare(&t, vm, &g_ref, &g, hold && !bf16).0);
        if pr {
            let (e, rest) = stats_diff(m, ids, tg, &mm);
            eprintln!(
                "{t}: Pr statistics max rel diff {e:.2e}, {rest:.2e} without each block's {SWAPS} largest{}",
                if hold && !bf16 { " (bars 1e-2 and 1e-4)" } else { " (printed only)" }
            );
            if hold && !bf16 && !(e <= 1e-2 && rest <= 1e-4) {
                bad.push(format!("{t}: Pr statistics off ({e}, {rest} without the {SWAPS} largest per block)"));
            }
        }
    }
    bad
}

/// Half of every block's units cut (a different half per block), as at the schedule's end.
fn cut_half(vp: &VarMap, mp: &Model, dev: &Device) {
    let (h, layers) = (mp.cfg.hidden(), mp.cfg.layers);
    for i in 0..layers {
        let al: Vec<f32> = (0..h).map(|j| if (j + i) % 2 == 0 { 1.0 } else { 0.0 }).collect();
        vp.data().lock().unwrap()[&format!("b{i}.v51.alive")].set(&Tensor::from_vec(al, h, dev).unwrap()).unwrap();
    }
    mp.v51_schedule(0, 0);
    assert_eq!(mp.v51_alive().unwrap().unwrap(), vec![h / 2; layers]);
}

/// Held: Pr (none cut, half cut) and B at v52's width and depth (d384, 8 blocks, the engram in block 4, the magnitude
/// code), a short window (ctx 64, batch 4), the fused step against autograd at the bars above.
#[test]
fn pr_step_matches_autograd_at_d384() {
    let Some(dev) = cuda() else {
        eprintln!("pr vs autograd at d384: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let ((vb, mb), (vp, mp)) = pair(&dev, 384, 8, 64, 4);
    let (ids, tg) = batch(&dev, 4, 64, 93);
    let mut bad = vs_autograd("d384 ctx64 B", &vb, &mb, &ids, &tg, false, true);
    bad.extend(vs_autograd("d384 ctx64 Pr none cut", &vp, &mp, &ids, &tg, true, true));
    cut_half(&vp, &mp, &dev);
    bad.extend(vs_autograd("d384 ctx64 Pr half cut", &vp, &mp, &ids, &tg, true, true));
    assert!(bad.is_empty(), "Pr's fused step off the reference at d384: {bad:#?}");
}

/// Every weight but Pr's own times (1 + rel · noise) (returns copies of the old values), or back to `keep`'s values.
fn nudge(vm: &VarMap, rel: f32, keep: Option<&[(String, Tensor)]>) -> Vec<(String, Tensor)> {
    let mut old = Vec::new();
    for (k, (name, var)) in vars(vm).into_iter().enumerate() {
        if let Some(keep) = keep {
            if let Some((_, t0)) = keep.iter().find(|(n, _)| *n == name) {
                var.set(t0).unwrap();
            }
            continue;
        }
        if name.contains(".v51.") {
            continue;
        }
        let t = var.as_tensor().copy().unwrap(); // a copy: Var::set writes into the variable's own storage
        let z = Tensor::from_vec(noise(t.elem_count(), 7000 + k as u64, rel), t.shape(), t.device()).unwrap();
        var.set(&t.add(&t.mul(&z).unwrap()).unwrap()).unwrap();
        old.push((name, t));
    }
    old
}

/// Printed only (never fails): one window of v52's length (d384 L8 ctx 2048, batch 1). The fused step against autograd
/// for B, Pr none cut and Pr half cut, by window length (64, 256, 1024, 2048: the same weights), and how far autograd
/// itself moves when every weight is nudged by 1e-6 of itself (and the fused step the same way): the gradients'
/// sensitivity to rounding at this length, the yardstick for the fused-vs-autograd numbers.
#[test]
fn v52_window_fused_vs_autograd_sensitivity() {
    let Some(dev) = cuda() else {
        eprintln!("v52 window sensitivity: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let ((vb, mb), (vp, mp)) = pair(&dev, 384, 8, 2048, 4);
    for t in [64usize, 256, 1024, 2048] {
        let (ids, tg) = batch(&dev, 1, t, 93);
        vs_autograd(&format!("ctx{t} B"), &vb, &mb, &ids, &tg, false, false);
    }
    let (ids, tg) = batch(&dev, 1, 2048, 93);
    vs_autograd("ctx2048 Pr none cut", &vp, &mp, &ids, &tg, true, false);
    // the yardstick: autograd against autograd (and fused against fused) after a 1e-6 relative nudge of every weight
    let (l0, _, g0) = mb.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
    let (f0, _, h0) = mb.loss_and_grads_win(&ids, &tg, None, false, &Mm { bf16: false }).unwrap();
    let keep = nudge(&vb, 1e-6, None);
    let (l1, _, g1) = mb.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
    let (f1, _, h1) = mb.loss_and_grads_win(&ids, &tg, None, false, &Mm { bf16: false }).unwrap();
    nudge(&vb, 0.0, Some(&keep));
    eprintln!("ctx2048 B nudged 1e-6: autograd loss {:.6} -> {:.6}, fused f32 loss {:.6} -> {:.6}", s(&l0), s(&l1), s(&f0), s(&f1));
    compare("ctx2048 B autograd vs autograd nudged 1e-6", &vb, &g0, &g1, false);
    compare("ctx2048 B fused f32 vs fused f32 nudged 1e-6", &vb, &h0, &h1, false);
    cut_half(&vp, &mp, &dev);
    vs_autograd("ctx2048 Pr half cut", &vp, &mp, &ids, &tg, true, false);
}
