// TESTING / DESIGN ONLY — tests of src/v50.rs (runs/v50/prereg_v50.md: --dale, --loop, --neuromod), CPU only.
//
// Gradient checks. (1) Every new op alone in f64 (the functions the model calls): central finite differences against
// autograd, |fd - autograd| <= 1e-6 (1 + |fd|). (2) The whole model, in f32 (the model builds its states in f32): a
// tiny model with Ava 0.4's recipe (kWTA 1/1 so the MLP is smooth, relu2), every LIF input weight at 0 (the spikes are
// thresholds: a finite difference cannot see a surrogate gradient, so the spiking path is silenced; with zero input
// current it carries no gradient either), and the new parts at small noise. Per tensor, the directional derivative
// along the gradient itself and two fixed random directions, each with unit spread per entry, by Ridders'
// extrapolation of central differences (steps from 3e-2 of the tensor's spread down by 1.4 per step, raised where the
// loss would move less than 1e-3 along the gradient): |fd - autograd| <= 1e-2 of the full gradient (the gradient
// direction's derivative, the largest a unit-spread direction can have), and along the gradient also <= 2e-2 of its
// own value (a wiring error is off by O(1)). Measured (16 tensors, 48 directions): along the gradient <= 3.6e-3 for
// every tensor but nm.bn, random directions <= 1.4e-3 of the full gradient but nm.bn's 6.6e-3; nm.bn's derivative
// (6.2e-3, 3 entries) is close to the forward's f32 noise: its central differences scatter +-1.5% around autograd with
// no trend over steps 1e-3 .. 0.5 (1.1e-2 along its gradient; the f64 op check holds it to 4e-10). A step sweep
// (3e-2 .. 3e-4 of the spread) showed why one fixed step fails: the difference converges as h^2 along the gradient
// (curvature) while f32 rounding of a loss near 5.5 dominates small derivatives at small steps.
// Chat checks: Ava 0.4's recipe with an engram plus each option (and all three): pieces, the one-token stream and the
// batched feed give the full forward's logits within 1e-4 (as state::tests::ava04_recipe_streams_match_forward).

use super::*;
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, in_pieces, max_diff, one_pass, tokens};
use crate::state::State;
use crate::testkit::{scratch, unit_noise};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};

const CPU: Device = Device::Cpu;
const RECIPE: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense \
                      --resid-fix --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif \
                      --homeostasis 0.1";

fn arch(extra: &str) -> Arch {
    Arch::from_flags(&format!("{RECIPE} {extra}")).unwrap()
}

fn set(vm: &VarMap, name: &str, t: &Tensor) {
    vm.data().lock().unwrap()[name].set(t).unwrap();
}

fn get(vm: &VarMap, name: &str) -> Tensor {
    vm.data().lock().unwrap()[name].as_tensor().copy().unwrap()
}

/// --dale's v back to softplus^-1(|noise · resid_std|) (fix_weights and build_awake give it zero-mean noise).
fn dale_fix(vm: &VarMap, cfg: &Config, seed: u64) {
    let names: Vec<String> = vm.data().lock().unwrap().keys().filter(|k| k.ends_with(".mlp.w2s")).cloned().collect();
    for (k, n) in names.iter().enumerate() {
        let v = get(vm, n);
        let noise = unit_noise(v.elem_count(), seed * 31 + k as u64);
        let w: Vec<f32> = if cfg.arch.dale_proj {
            // D2: v = |noise · resid_std| · the E/I start scale of its row (as mlp_w2_param)
            let d = v.dims()[1];
            let (ne, (se, si)) = proj_balance(cfg.arch.dale, v.dims()[0]);
            noise.iter().enumerate().map(|(j, z)| (z.abs() as f64 * cfg.resid_std()) as f32 * if j / d < ne { se } else { si }).collect()
        } else {
            noise.iter().map(|z| softplus_inv((z.abs() as f64 * cfg.resid_std()).max(1e-9)) as f32).collect()
        };
        set(vm, n, &Tensor::from_vec(w, v.dims(), &CPU).unwrap());
    }
}

/// D2 (--dale-form proj, v50 Amendment 3): flags round-trip; the start is E/I-balanced with today's mean square;
/// w = sign ⊙ v exactly; the step after each optimizer step is AdamW's decay then the clip v >= 0 (also at c = 0);
/// the op's gradient matches finite differences (f64); the talking cache changes nothing.
#[test]
fn dale_proj_balance_projection_and_gradient() {
    let a = Arch::from_flags("--dale 0.8 --dale-form proj").unwrap();
    assert!(a.dale_proj && a.dale == 0.8);
    assert_eq!(a.flags(), "--dale 0.8 --dale-form proj");
    assert_eq!(a.signature(), " dale=0.8 dale_form=proj");
    assert_eq!(Arch::from_flags(&a.flags()).unwrap(), a);
    assert!(Arch::from_flags("--dale 0.8 --dale-form other").is_err());
    assert!(Config::with_kind(Kind::D, 258, 64, 2, 16).with_arch(Arch::from_flags("--dale-form proj").unwrap()).is_err(), "needs --dale");
    assert_eq!(proj_balance(0.8, 1000), (800, (0.5, 2.0)));
    let cfg = Config::with_kind(Kind::D, 258, 128, 2, 16).with_arch(arch("--dale 0.8 --dale-form proj")).unwrap();
    let vm = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
    let (hid, d) = (cfg.hidden(), cfg.d);
    let ne = dale_e(0.8, hid);
    let v0 = get(&vm, "b0.mlp.w2s");
    assert!(v0.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|&x| x >= 0.0), "v starts >= 0");
    let w = m.mlp_w2(0).unwrap();
    let rows: Vec<Vec<f32>> = w.to_vec2().unwrap();
    assert!(rows[..ne].iter().all(|r| r.iter().all(|&x| x >= 0.0)) && rows[ne..].iter().all(|r| r.iter().all(|&x| x <= 0.0)));
    let all: Vec<f64> = rows.iter().flatten().map(|&x| x as f64).collect();
    let rms = (all.iter().map(|x| x * x).sum::<f64>() / all.len() as f64).sqrt();
    assert!((rms / cfg.resid_std() - 1.0).abs() < 0.03, "spread {rms} vs today's {}", cfg.resid_std());
    // balance: the summed E and I magnitudes over all outputs agree within 3%
    let se: f64 = rows[..ne].iter().flatten().map(|&x| x as f64).sum();
    let si: f64 = -rows[ne..].iter().flatten().map(|&x| x as f64).sum::<f64>();
    assert!((se / si - 1.0).abs() < 0.03, "E drive {se} vs I drive {si}");
    // w = sign ⊙ v exactly
    let sign = Tensor::from_vec((0..hid).map(|i| if i < ne { 1f32 } else { -1.0 }).collect::<Vec<f32>>(), (hid, 1), &CPU).unwrap();
    assert_eq!(max_diff(&w, &v0.broadcast_mul(&sign).unwrap()), 0.0);
    // the step: v <- max(v (1 - c), 0), and the clip also at c = 0
    let vs: Vec<f32> = (0..hid * d).map(|i| -1.0 + 2.0 * i as f32 / (hid * d) as f32).collect();
    for c in [2e-4, 0.0] {
        set(&vm, "b0.mlp.w2s", &Tensor::from_vec(vs.clone(), (hid, d), &CPU).unwrap());
        m.dale_decay(c).unwrap();
        let got: Vec<f32> = get(&vm, "b0.mlp.w2s").flatten_all().unwrap().to_vec1().unwrap();
        let worst = vs.iter().zip(&got).map(|(&x, &y)| ((x * (1.0 - c as f32)).max(0.0) - y).abs()).fold(0f32, f32::max);
        assert!(worst <= 1e-7, "c {c}: worst {worst}");
    }
    // the cache (talking) gives the same logits
    let ids = Tensor::from_vec(tokens(1, 16, 258, 3), (1, 16), &CPU).unwrap();
    let plain = m.forward(&ids).unwrap();
    m.set_dale_cache(true);
    assert_eq!(max_diff(&m.forward(&ids).unwrap(), &plain), 0.0);
    m.set_dale_cache(false);
    // the op's gradient, f64: loss = sum(R ⊙ (a · (sign ⊙ v)))
    let (n, dd, h) = (6usize, 8usize, 10usize);
    let f64t = |shape: &[usize], seed: u64| Tensor::from_vec(unit_noise(shape.iter().product(), seed).iter().map(|&z| z as f64).collect::<Vec<f64>>(), shape, &CPU).unwrap();
    let a = f64t(&[n, h], 11);
    let r = f64t(&[n, dd], 13);
    let v = Var::from_tensor(&f64t(&[h, dd], 12).abs().unwrap()).unwrap();
    let sg = Tensor::from_vec((0..h).map(|i| if i < 8 { 1.0 } else { -1.0 }).collect::<Vec<f64>>(), (h, 1), &CPU).unwrap();
    let f = || a.matmul(&dale_w2_proj(v.as_tensor(), &sg).unwrap()).unwrap().mul(&r).unwrap().sum_all().unwrap();
    let ga: Vec<f64> = f().backward().unwrap().get(v.as_tensor()).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let base = v.as_tensor().copy().unwrap();
    let flat: Vec<f64> = base.flatten_all().unwrap().to_vec1().unwrap();
    for j in 0..flat.len() {
        let mut p = flat.clone();
        p[j] += 1e-6;
        v.set(&Tensor::from_vec(p.clone(), base.dims(), &CPU).unwrap()).unwrap();
        let up = f().to_scalar::<f64>().unwrap();
        p[j] -= 2e-6;
        v.set(&Tensor::from_vec(p, base.dims(), &CPU).unwrap()).unwrap();
        let down = f().to_scalar::<f64>().unwrap();
        v.set(&base).unwrap();
        let fd = (up - down) / 2e-6;
        assert!((fd - ga[j]).abs() <= 1e-6 * (1.0 + fd.abs()), "dale proj v[{j}]: fd {fd} vs autograd {}", ga[j]);
    }
}

fn awake(cfg: Config, seed: u64) -> (VarMap, Model) {
    let (vm, m) = build_awake(cfg, seed);
    dale_fix(&vm, &cfg, seed);
    (vm, m)
}

#[test]
fn options_parse_count_and_refuse() {
    let a = Arch::from_flags("--dale 0.8 --loop 2 --neuromod 3").unwrap();
    assert_eq!((a.dale, a.loops, a.neuromod), (0.8, 2, 3));
    assert_eq!(a.flags(), "--dale 0.8 --loop 2 --neuromod 3");
    assert_eq!(a.signature(), " dale=0.8 loop=2 neuromod=3");
    assert_eq!(Arch::from_flags(&a.flags()).unwrap(), a);
    assert!(Arch::default().flags().is_empty() && !any(&Arch::default()));
    let c = Config::with_kind(Kind::D, 258, 64, 2, 16);
    for bad in ["--dale 1.5", "--dale -0.1", "--loop 0", "--loop 5", "--neuromod 2", "--loop 2 --feedback", "--loop 2 --modulate"] {
        assert!(c.with_arch(Arch::from_flags(bad).unwrap()).is_err(), "{bad} should be refused");
    }
    assert!(Config::with_kind(Kind::D, 258, 64, 1, 16).with_arch(Arch::from_flags("--neuromod 3").unwrap()).is_err(), "one block");
    assert!(Config::with_kind(Kind::S, 258, 64, 2, 16).with_arch(Arch::from_flags("--dale 0.8").unwrap()).is_err(), "model D only");
    // the weight count matches the tensors made, for each option on the recipe
    for extra in ["", "--dale 0.8", "--loop 2", "--neuromod 3", "--dale 0.8 --loop 2 --neuromod 3"] {
        let cfg = Config::with_kind(Kind::D, 258, 64, 2, 16).with_engram(1).unwrap().with_arch(arch(extra)).unwrap();
        let vm = VarMap::new();
        let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
        let made: usize = vm.data().lock().unwrap().values().map(|v| v.elem_count()).sum();
        assert_eq!(made, cfg.weight_counts().0, "{extra}");
        let names: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
        assert_eq!(names.iter().any(|n| n.ends_with(".mlp.w2s")), extra.contains("dale"), "{extra}");
        assert_eq!(names.iter().any(|n| n.ends_with(".mlp.w2")), !extra.contains("dale"), "{extra}");
        assert_eq!(names.iter().any(|n| n == "loop.wo"), extra.contains("loop"), "{extra}");
        assert_eq!(names.iter().any(|n| n == "nm.wn"), extra.contains("neuromod"), "{extra}");
        // AdamW decays loop.wo (a .wo) but not --dale's v (its own decay) nor the nucleus (gate-like, starts at 0)
        for n in &names {
            if n.ends_with(".w2s") || n.starts_with("nm.") {
                assert!(!Model::decays_with(n, true), "{n}");
            }
        }
        if extra.contains("loop") {
            assert!(Model::decays_with("loop.wo", true));
        }
        if any(&cfg.arch) {
            assert!(crate::model::fstep::arch::blocker(&m).is_some(), "{extra}: the fused talking step must refuse it");
        }
    }
}

/// --dale's init: the effective W2 has today's spread and the E/I signs by index; softplus^-1 inverts softplus; the
/// decay step shrinks every effective weight by (1 - c) (to first order) at any v; the talking cache changes nothing.
#[test]
fn dale_init_signs_decay_and_cache() {
    for y in [1e-9, 1e-5, 0.01, 0.3, 2.0, 40.0] {
        let v = softplus_inv(y);
        let back = softplus(&Tensor::new(&[v], &CPU).unwrap()).unwrap().to_vec1::<f64>().unwrap()[0];
        assert!((back - y).abs() <= 1e-12 + 1e-9 * y, "softplus(softplus_inv({y})) = {back}");
    }
    let cfg = Config::with_kind(Kind::D, 258, 128, 2, 16).with_arch(arch("--dale 0.8")).unwrap();
    let vm = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
    let hid = cfg.hidden();
    let ne = dale_e(0.8, hid);
    assert_eq!(ne, 410);
    let w = m.mlp_w2(0).unwrap();
    let rows: Vec<Vec<f32>> = w.to_vec2().unwrap();
    assert!(rows[..ne].iter().all(|r| r.iter().all(|&x| x >= 0.0)) && rows[ne..].iter().all(|r| r.iter().all(|&x| x <= 0.0)));
    let all: Vec<f64> = rows.iter().flatten().map(|&x| x as f64).collect();
    let rms = (all.iter().map(|x| x * x).sum::<f64>() / all.len() as f64).sqrt();
    assert!((rms / cfg.resid_std() - 1.0).abs() < 0.02, "the effective W2's spread {rms} vs today's {}", cfg.resid_std());
    // the decay at v spread over [-12, 6] (both branches)
    let n = hid * cfg.d;
    let vs: Vec<f32> = (0..n).map(|i| -12.0 + 18.0 * i as f32 / n as f32).collect();
    set(&vm, "b0.mlp.w2s", &Tensor::from_vec(vs, (hid, cfg.d), &CPU).unwrap());
    let before = softplus(&get(&vm, "b0.mlp.w2s").to_dtype(DType::F64).unwrap()).unwrap();
    let c = 2e-4;
    m.dale_decay(c).unwrap();
    let after = softplus(&get(&vm, "b0.mlp.w2s").to_dtype(DType::F64).unwrap()).unwrap();
    let ratio: Vec<f64> = after.div(&before).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    // f32 storage: v moves by ~2e-4, held to ~1e-7 absolute, so the shrink is exact to ~1e-6 of the weight
    let worst = ratio.iter().map(|r| (r - (1.0 - c)).abs()).fold(0.0, f64::max);
    assert!(worst < 2e-6, "decay: worst |w'/w - (1 - c)| = {worst}");
    // the cache (talking) gives the same logits and is dropped by a decay step
    let ids = Tensor::from_vec(tokens(1, 16, 258, 3), (1, 16), &CPU).unwrap();
    let plain = m.forward(&ids).unwrap();
    m.set_dale_cache(true);
    assert_eq!(max_diff(&m.forward(&ids).unwrap(), &plain), 0.0);
    m.dale_decay(c).unwrap();
    assert!(m.ext.as_ref().unwrap().v50.dale_cache.borrow().is_none());
    m.set_dale_cache(false);
}

/// (1) The new ops alone, f64: central finite differences against autograd.
#[test]
fn v50_ops_gradients_match_finite_differences_f64() {
    let f64t = |shape: &[usize], seed: u64, scale: f64, shift: f64| {
        let n: usize = shape.iter().product();
        Tensor::from_vec(unit_noise(n, seed).iter().map(|&z| z as f64 * scale + shift).collect::<Vec<f64>>(), shape, &CPU).unwrap()
    };
    let eps = 1e-6;
    let check = |what: &str, var: &Var, f: &dyn Fn() -> Tensor| {
        let g = f().backward().unwrap();
        let ga: Vec<f64> = g.get(var.as_tensor()).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let base = var.as_tensor().copy().unwrap();
        let flat: Vec<f64> = base.flatten_all().unwrap().to_vec1().unwrap();
        let mut worst = 0f64;
        for j in 0..flat.len() {
            let mut p = flat.clone();
            p[j] += eps;
            var.set(&Tensor::from_vec(p.clone(), base.dims(), &CPU).unwrap()).unwrap();
            let up = f().to_scalar::<f64>().unwrap();
            p[j] -= 2.0 * eps;
            var.set(&Tensor::from_vec(p, base.dims(), &CPU).unwrap()).unwrap();
            let down = f().to_scalar::<f64>().unwrap();
            var.set(&base).unwrap();
            let fd = (up - down) / (2.0 * eps);
            let e = (fd - ga[j]).abs() / (1.0 + fd.abs());
            worst = worst.max(e);
            assert!(e <= 1e-6, "{what}[{j}]: finite difference {fd} vs autograd {}", ga[j]);
        }
        eprintln!("v50 f64 gradient check {what}: {} entries, worst |fd - autograd| / (1 + |fd|) = {worst:.2e}", flat.len());
    };
    let (n, d, hid, k) = (6usize, 8usize, 10usize, 3usize);
    // --dale: loss = sum(R ⊙ (a · (sign ⊙ softplus(v)))), v on both sides of 0
    let a = f64t(&[n, hid], 1, 1.0, 0.5);
    let v = Var::from_tensor(&f64t(&[hid, d], 2, 3.0, -1.0)).unwrap();
    let sign = Tensor::from_vec((0..hid).map(|i| if i < 8 { 1.0 } else { -1.0 }).collect::<Vec<f64>>(), (hid, 1), &CPU).unwrap();
    let r = f64t(&[n, d], 3, 1.0, 0.0);
    check("dale v", &v, &|| a.matmul(&dale_w2(v.as_tensor(), &sign).unwrap()).unwrap().mul(&r).unwrap().sum_all().unwrap());
    // --loop: loss = sum(R ⊙ (x + norm(top, g) · W)), by top, g and W
    let x = f64t(&[n, d], 4, 1.0, 0.0);
    let top = Var::from_tensor(&f64t(&[n, d], 5, 2.0, 0.3)).unwrap();
    let g = Var::from_tensor(&f64t(&[d], 6, 0.3, 1.0)).unwrap();
    let w = Var::from_tensor(&f64t(&[d, d], 7, 0.4, 0.0)).unwrap();
    let f_loop = || loop_add(&x, top.as_tensor(), g.as_tensor(), w.as_tensor()).unwrap().mul(&r).unwrap().sum_all().unwrap();
    check("loop top", &top, &f_loop);
    check("loop gain", &g, &f_loop);
    check("loop W", &w, &f_loop);
    // --neuromod: loss = sum(R3 ⊙ 2 sigmoid(norm(x1, g) · W + b)) and the three uses
    let x1 = Var::from_tensor(&f64t(&[n, d], 8, 1.5, 0.2)).unwrap();
    let gn = Var::from_tensor(&f64t(&[d], 9, 0.3, 1.0)).unwrap();
    let wn = Var::from_tensor(&f64t(&[d, k], 10, 0.5, 0.0)).unwrap();
    let bn = Var::from_tensor(&f64t(&[k], 11, 0.5, 0.0)).unwrap();
    let r3 = f64t(&[n, k], 12, 1.0, 0.0);
    let f_nm = || nucleus_signals(x1.as_tensor(), gn.as_tensor(), wn.as_tensor(), bn.as_tensor()).unwrap().mul(&r3).unwrap().sum_all().unwrap();
    check("nucleus input", &x1, &f_nm);
    check("nucleus gain", &gn, &f_nm);
    check("nucleus W", &wn, &f_nm);
    check("nucleus b", &bn, &f_nm);
    // ACh: loss = sum(R4 ⊙ clamp(beta ⊙ s, 0, 2)), beta in (0, 2), s in (0, 2) (away from the clamp's corners)
    let (bb, nh, t, gg) = (2usize, 2usize, 3usize, 4usize);
    let beta = Var::from_tensor(&f64t(&[bb, nh, t, gg], 13, 0.25, 0.6)).unwrap();
    let s = Var::from_tensor(&f64t(&[bb * t, 1], 14, 0.2, 1.0)).unwrap();
    let r4 = f64t(&[bb, nh, t, gg], 15, 1.0, 0.0);
    let f_ach = || ach_scale(beta.as_tensor(), s.as_tensor()).unwrap().mul(&r4).unwrap().sum_all().unwrap();
    check("ACh write strength", &beta, &f_ach);
    check("ACh signal", &s, &f_ach);
}

/// (2) The whole model, f32 (see the top of the file): autograd's directional derivatives against finite differences.
#[test]
fn v50_model_gradients_match_finite_differences_f32() {
    let flags = "--head-size 32 --timescales fixed --value-residual --kwta 1 --mlp-act relu2 --residual dense --resid-fix \
                 --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";
    let (b, t) = (2, 16);
    let toks = tokens(b, t + 1, 258, 91);
    let ids: Vec<u32> = (0..b).flat_map(|r| toks[r * (t + 1)..r * (t + 1) + t].to_vec()).collect();
    let tg: Vec<u32> = (0..b).flat_map(|r| toks[r * (t + 1) + 1..(r + 1) * (t + 1)].to_vec()).collect();
    let ids = Tensor::from_vec(ids, (b, t), &CPU).unwrap();
    let tg = Tensor::from_vec(tg, (b, t), &CPU).unwrap();
    let cases: [(&str, &[&str]); 5] = [
        ("--dale 0.8", &["b0.mlp.w2s", "b1.mlp.w2s", "b0.mlp.w1"]),
        ("--dale 0.8 --dale-form proj", &["b0.mlp.w2s", "b1.mlp.w2s", "b0.mlp.w1"]),
        ("--loop 2", &["loop.wo", "b0.mlp.w1", "b1.hebb.wv", "final.norm"]),
        ("--neuromod 3", &["nm.wn", "nm.bn", "nm.norm", "b1.mlp.w2", "b1.hebb.wa1"]),
        ("--dale 0.8 --loop 2 --neuromod 3", &["b1.mlp.w2s", "loop.wo", "nm.wn", "b0.hebb.wq"]),
    ];
    for (extra, names) in cases {
        let ar = Arch::from_flags(&format!("{flags} {extra}")).unwrap();
        let cfg = Config::with_kind(Kind::D, 258, 64, 2, 32).with_arch(ar).unwrap();
        let (vm, m) = awake(cfg, 52);
        for i in 0..cfg.layers {
            // the spiking path silenced (see the top of the file)
            set(&vm, &format!("b{i}.mem.win"), &Tensor::zeros((64, 64), DType::F32, &CPU).unwrap());
        }
        let loss = || m.loss_aux(&ids, &tg, 0.0).unwrap().0;
        let grads = loss().backward().unwrap();
        for (j, name) in names.iter().enumerate() {
            let base = get(&vm, name);
            let ga = grads.get(vm.data().lock().unwrap()[*name].as_tensor()).unwrap_or_else(|| panic!("{extra}: no gradient for {name}")).copy().unwrap();
            let spread = base.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt().max(1e-3) as f64;
            // direction 0: the gradient itself (scaled to unit spread per entry: the largest directional derivative a
            // unit-spread direction can have); directions 1 and 2: fixed random ones
            let g_rms = ga.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            assert!(g_rms > 0.0, "{extra}: zero gradient for {name}");
            let full = ga.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() as f64 / g_rms as f64;
            for dir in 0..3u64 {
                let u = if dir == 0 {
                    ga.affine(1.0 / g_rms as f64, 0.0).unwrap()
                } else {
                    Tensor::from_vec(unit_noise(base.elem_count(), 700 + 10 * j as u64 + dir), base.dims(), &CPU).unwrap()
                };
                // Ridders' extrapolation of central differences over steps h0, h0/1.4, ... (12 steps, a factor 40):
                // it keeps the estimate whose error estimate is smallest, so each tensor finds its own step between
                // curvature (large steps) and f32 rounding of the loss (small ones); h0 = 3e-2 of the tensor's
                // spread, raised where the loss would move less than 1e-3 along the gradient
                let h0 = (3e-2 * spread).max(1e-3 / full);
                let ad = ga.mul(&u).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() as f64;
                let fd_at = |h: f64| {
                    set(&vm, name, &base.add(&u.affine(h, 0.0).unwrap()).unwrap());
                    let up = loss().to_scalar::<f32>().unwrap() as f64;
                    set(&vm, name, &base.sub(&u.affine(h, 0.0).unwrap()).unwrap());
                    let down = loss().to_scalar::<f32>().unwrap() as f64;
                    set(&vm, name, &base);
                    (up - down) / (2.0 * h)
                };
                let (fd, fd_err) = ridders(&fd_at, h0);
                // the error as a share of the full gradient (direction 0's derivative), and of this direction's own
                let (of_full, of_own) = ((fd - ad).abs() / full, (fd - ad).abs() / ad.abs().max(1e-12));
                eprintln!("v50 f32 model gradient check {extra:<34} {name:<12} dir {dir}: autograd {ad:+.6e} fd {fd:+.6e} (+-{fd_err:.1e}) err/full {of_full:.2e} err/own {of_own:.2e}");
                assert!(if dir == 0 { of_own <= 2e-2 } else { of_full <= 1e-2 }, "{extra} {name} dir {dir}: autograd {ad} vs finite difference {fd}");
            }
        }
    }
}

/// --loop and --neuromod start neutral: with today's weights copied in and their new parts at init (W = 0, signals 1),
/// they give the base recipe's logits bit for bit (--dale changes the signs, so it does not).
#[test]
fn loop_and_neuromod_start_as_the_base() {
    let base_cfg = Config::with_kind(Kind::D, 258, 64, 2, 64).with_engram(1).unwrap().with_arch(arch("")).unwrap();
    let (bvm, bm) = build_awake(base_cfg, 61);
    let ids = Tensor::from_vec(tokens(2, 32, 258, 5), (2, 32), &CPU).unwrap();
    let want = bm.forward(&ids).unwrap();
    for extra in ["--loop 2", "--neuromod 3", "--loop 2 --neuromod 3"] {
        let cfg = Config::with_kind(Kind::D, 258, 64, 2, 64).with_engram(1).unwrap().with_arch(arch(extra)).unwrap();
        let vm = VarMap::new();
        let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
        m.set_track_activity(false);
        for (k, v) in bvm.data().lock().unwrap().iter() {
            set(&vm, k, v.as_tensor());
        }
        assert_eq!(max_diff(&m.forward(&ids).unwrap(), &want), 0.0, "{extra}");
    }
}

/// The chat check: Ava 0.4's recipe with an engram plus each option: pieces, the one-token stream and the batched feed
/// give the full forward's logits (within 1e-4), and a saved and loaded state continues as the unsaved one.
#[test]
fn v50_options_stream_match_forward() {
    let dir = scratch("v50_stream");
    for extra in ["--dale 0.8", "--dale 0.8 --dale-form proj", "--loop 2", "--neuromod 3", "--dale 0.8 --loop 2 --neuromod 3"] {
        let cfg = Config::with_kind(Kind::D, 258, 64, 2, 64).with_engram(1).unwrap().with_arch(arch(extra)).unwrap();
        let (_, m) = awake(cfg, 48);
        let (b, t) = (2, 64);
        let toks = tokens(b, t, 258, 72);
        let want = one_pass(&m, &toks, b, t);
        let mut st = m.state(b).unwrap();
        assert_eq!(st.loop2.is_some(), extra.contains("loop"));
        let mut worst = [0f32; 4];
        worst[0] = max_diff(&in_pieces(&m, &mut st, &toks, b, t, &[25, 39]), &want);
        let last = want.get(0).unwrap().get(t - 1).unwrap();
        let mut s = m.stream().unwrap();
        let mut out = None;
        let mut step_worst = 0f32;
        for (j, &tok) in toks[..t].iter().enumerate() {
            let l = m.step(&mut s, tok).unwrap();
            step_worst = step_worst.max(max_diff(&l, &want.get(0).unwrap().get(j).unwrap()));
            out = Some(l);
        }
        worst[1] = step_worst.max(max_diff(&out.unwrap(), &last));
        let mut s = m.stream().unwrap();
        worst[2] = max_diff(&m.feed(&mut s, &toks[..t]).unwrap(), &last);
        // saved after 40 tokens, loaded, the rest read: the same logits as reading on
        let mut s1 = m.state(b).unwrap();
        let first = in_pieces(&m, &mut s1, &toks.iter().enumerate().filter(|(i, _)| i % t < 40).map(|(_, &x)| x).collect::<Vec<_>>(), b, 40, &[40]);
        drop(first);
        let p = dir.join(format!("st{}.safetensors", extra.len()));
        s1.save(&m, &p).unwrap();
        let mut s2 = State::load(&m, &p, &CPU).unwrap();
        let rest: Vec<u32> = toks.iter().enumerate().filter(|(i, _)| i % t >= 40).map(|(_, &x)| x).collect();
        let a1 = in_pieces(&m, &mut s1, &rest, b, t - 40, &[t - 40]);
        let a2 = in_pieces(&m, &mut s2, &rest, b, t - 40, &[t - 40]);
        worst[3] = max_diff(&a1, &a2).max(max_diff(&a1, &want.narrow(1, 40, t - 40).unwrap()));
        eprintln!("v50 chat check {extra:<34}: pieces {:.2e}, stream (every token) {:.2e}, feed {:.2e}, saved state {:.2e}", worst[0], worst[1], worst[2], worst[3]);
        for (what, w) in ["pieces", "stream", "feed", "saved state"].iter().zip(worst) {
            assert!(w < 1e-4, "{extra} {what}: {w}");
        }
        // two rows selected and joined again read on as the whole state
        let joined = State::cat(&[st.select(&[0]).unwrap(), st.select(&[1]).unwrap()]).unwrap();
        assert_eq!(joined.loop2.is_some(), extra.contains("loop"));
    }
}

/// Ridders' method (Numerical Recipes 5.7): central differences `cd(h)` at h0, h0/1.4, ..., extrapolated to h = 0 in a
/// Neville tableau; returns the estimate with the smallest error estimate, and that estimate.
fn ridders(cd: &dyn Fn(f64) -> f64, h0: f64) -> (f64, f64) {
    const CON: f64 = 1.4;
    const N: usize = 12;
    let mut a = vec![vec![0.0f64; N]; N];
    let (mut h, mut best, mut err) = (h0, 0.0, f64::INFINITY);
    a[0][0] = cd(h);
    for i in 1..N {
        h /= CON;
        a[0][i] = cd(h);
        let mut fac = CON * CON;
        for j in 1..=i {
            a[j][i] = (a[j - 1][i] * fac - a[j - 1][i - 1]) / (fac - 1.0);
            fac *= CON * CON;
            let e = (a[j][i] - a[j - 1][i]).abs().max((a[j][i] - a[j - 1][i - 1]).abs());
            if e <= err {
                err = e;
                best = a[j][i];
            }
        }
        if (a[i][i] - a[i - 1][i - 1]).abs() >= 2.0 * err {
            break;
        }
    }
    (best, err)
}
