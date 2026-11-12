// TESTING / DESIGN ONLY — tests of src/v51b.rs (runs/v51/prereg_v51.md Part B2: --thalamus, --cerebellum-head), CPU only.
//
// Gradient checks, as v50's (src/v50/tests.rs). (1) Every new op alone in f64: central finite differences against
// autograd, |fd - autograd| <= 1e-6 (1 + |fd|). (2) The whole model in f32: a tiny model with Ava 0.4's recipe (kWTA
// 1/1, relu2), every LIF input weight at 0 (a finite difference cannot see a surrogate gradient), the new parts awake
// (small noise); directional derivatives along the gradient and two random directions by Ridders' extrapolation;
// bars as v50's: along the gradient |fd - ad| <= 2e-2 of its own value, random directions <= 1e-2 of the full gradient.
// The cerebellum's code reads her state detached by design (it learns to correct her, she does not bend to suit it),
// so a finite difference of a weight below it also moves its code while autograd does not: under --cerebellum-head
// only its read-out W is checked against finite differences; the thalamus is checked with every tensor of its own and
// model weights around it.
// Chat checks: Ava 0.4's recipe with an engram plus each option: pieces, the one-token stream and the batched feed
// give the full forward's logits within 1e-4, a saved and loaded state continues as the unsaved one. Checkpoint
// round trip: saved and loaded through ckpt::load_auto (arch.txt beside), the same logits bit for bit.
// The cerebellum's online rule: its step equals -lr times autograd's gradient of that word's cross-entropy with respect
// to W, the table's correction equals the folded W's, and on a repeated text it lowers the loss.

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

#[test]
fn options_parse_count_and_refuse() {
    let a = Arch::from_flags("--thalamus --cerebellum-head 256").unwrap();
    assert_eq!(a.v51b, Opts { thalamus: true, cb_units: 256 });
    assert_eq!(a.flags(), "--thalamus --cerebellum-head 256");
    assert_eq!(a.signature(), " thalamus=1 cerebellum_head=256");
    assert_eq!(Arch::from_flags(&a.flags()).unwrap(), a);
    assert!(!any(&Arch::default()));
    // wave5's older --cerebellum G is another option
    let w = Arch::from_flags("--cerebellum 64").unwrap();
    assert_eq!(w.v51b, Opts::default());
    let c = Config::with_kind(Kind::D, 258, 64, 2, 16);
    for bad in ["--cerebellum-head 20", "--cerebellum-head 100", "--thalamus --halting 2"] {
        assert!(c.with_arch(Arch::from_flags(bad).unwrap()).is_err(), "{bad} should be refused");
    }
    assert!(Arch::from_flags("--cerebellum-head").is_err());
    assert!(Config::with_kind(Kind::S, 258, 64, 2, 16).with_arch(Arch::from_flags("--thalamus").unwrap()).is_err(), "model D only");
    for extra in ["", "--thalamus", "--cerebellum-head 512", "--thalamus --cerebellum-head 512", "--thalamus --loop 2"] {
        let cfg = Config::with_kind(Kind::D, 258, 64, 2, 16).with_engram(1).unwrap().with_arch(arch(extra)).unwrap();
        let vm = VarMap::new();
        let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
        let made: usize = vm.data().lock().unwrap().values().map(|v| v.elem_count()).sum();
        assert_eq!(made, cfg.weight_counts().0, "{extra}");
        let names: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
        assert_eq!(names.iter().any(|n| n == "th.read"), extra.contains("thalamus"), "{extra}");
        assert_eq!(names.iter().any(|n| n == "th.tdg"), extra.contains("loop"), "{extra}");
        assert_eq!(names.iter().any(|n| n == "cbh.wo"), extra.contains("cerebellum"), "{extra}");
        for n in &names {
            if n.starts_with("th.") {
                assert!(!Model::decays_with(n, true), "{n}");
            }
        }
        if extra.contains("cerebellum") {
            assert!(Model::decays_with("cbh.wo", true));
        }
        if any(&cfg.arch) {
            assert!(crate::model::fstep::arch::blocker(&m).is_some(), "{extra}: the fused talking step must refuse it");
            // v51 fast2 (src/model/fwin_f2.rs): the fused training step takes TH and CB, not --loop
            assert_eq!(crate::model::fwin::step_ok(&cfg), !extra.contains("loop"), "{extra}: the fused training step");
        }
    }
    // the relay's init: seeded, unit spread per relay unit; the cerebellum's wiring: 4 distinct inputs per unit
    let cfg = Config::with_kind(Kind::D, 258, 128, 2, 16).with_arch(arch("--thalamus")).unwrap();
    let vm = VarMap::new();
    let _m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
    let r = get(&vm, "th.read");
    let rms = r.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt() as f64;
    assert!((rms * (128f64).sqrt() - 1.0).abs() < 0.03, "relay read spread {rms}");
    let vm2 = VarMap::new();
    let _m2 = Model::new(cfg, VarBuilder::from_varmap(&vm2, DType::F32, &CPU)).unwrap();
    assert_eq!(max_diff(&r, &get(&vm2, "th.read")), 0.0, "seeded: the same every build");
    let (idx, w) = cb_wiring(256, 1024, CB_FAN_IN, CB_SEED);
    assert_eq!(idx.len(), 1024 * 4);
    for u in 0..1024 {
        let mut s: Vec<u32> = idx[u * 4..u * 4 + 4].to_vec();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 4);
    }
    assert!(w.iter().all(|&x| x == 0.5 || x == -0.5));
}

/// Both start as the base: with the base's weights copied in and the new parts at init (gates 1, read-out 0), the
/// logits are the base recipe's bit for bit, on the parallel forward and in pieces.
#[test]
fn thalamus_and_cerebellum_start_as_the_base() {
    let toks = tokens(2, 32, 258, 5);
    let ids = Tensor::from_vec(toks.clone(), (2, 32), &CPU).unwrap();
    for (extra, base) in [("--thalamus", ""), ("--cerebellum-head 512", ""), ("--thalamus --cerebellum-head 512", ""), ("--thalamus --loop 2", "--loop 2")] {
        let cfg_of = |e: &str| Config::with_kind(Kind::D, 258, 64, 2, 64).with_engram(1).unwrap().with_arch(arch(e)).unwrap();
        let (bvm, bm) = build_awake(cfg_of(base), 61);
        let want = bm.forward(&ids).unwrap();
        let mut bst = bm.state(2).unwrap();
        let want_p = in_pieces(&bm, &mut bst, &toks, 2, 32, &[11, 21]);
        let vm = VarMap::new();
        let m = Model::new(cfg_of(extra), VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
        m.set_track_activity(false);
        for (k, v) in bvm.data().lock().unwrap().iter() {
            set(&vm, k, v.as_tensor());
        }
        assert_eq!(max_diff(&m.forward(&ids).unwrap(), &want), 0.0, "{extra}: forward");
        let mut st = m.state(2).unwrap();
        assert_eq!(max_diff(&in_pieces(&m, &mut st, &toks, 2, 32, &[11, 21]), &want_p), 0.0, "{extra}: pieces");
    }
}

/// (1) The new ops alone, f64: central finite differences against autograd.
#[test]
fn v51b_ops_gradients_match_finite_differences_f64() {
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
        eprintln!("v51b f64 gradient check {what}: {} entries, worst |fd - autograd| / (1 + |fd|) = {worst:.2e}", flat.len());
    };
    let (n, d, k) = (5usize, 8usize, 4usize);
    let ones = Tensor::ones(d, DType::F64, &CPU).unwrap();
    // TH read + gates: loss = sum(R2 ⊙ th_gates(th_read(x, R) + acc0, reads, G, c))
    let x = Var::from_tensor(&f64t(&[n, d], 1, 1.5, 0.2)).unwrap();
    let r = Var::from_tensor(&f64t(&[d, k], 2, 0.4, 0.0)).unwrap();
    let acc0 = f64t(&[n, k], 3, 0.8, 0.0);
    let g = Var::from_tensor(&f64t(&[k, 2], 4, 0.6, 0.0)).unwrap();
    let c = Var::from_tensor(&f64t(&[2], 5, 0.5, 0.0)).unwrap();
    let r2 = f64t(&[n, 2], 6, 1.0, 0.0);
    let f_th = || th_gates(&acc0.add(&th_read(x.as_tensor(), &ones, r.as_tensor()).unwrap()).unwrap(), 2, g.as_tensor(), c.as_tensor()).unwrap().mul(&r2).unwrap().sum_all().unwrap();
    check("relay input", &x, &f_th);
    check("relay read R", &r, &f_th);
    check("gate G", &g, &f_th);
    check("gate bias c", &c, &f_th);
    // TH residual rule: loss = sum(R3 ⊙ th_res(x_in, x_out, g))
    let xi = Var::from_tensor(&f64t(&[n, d], 7, 1.0, 0.0)).unwrap();
    let xo = Var::from_tensor(&f64t(&[n, d], 8, 1.0, 0.1)).unwrap();
    let gg = Var::from_tensor(&f64t(&[n, 1], 9, 0.3, 1.0)).unwrap();
    let r3 = f64t(&[n, d], 10, 1.0, 0.0);
    let f_res = || th_res(xi.as_tensor(), xo.as_tensor(), gg.as_tensor()).unwrap().mul(&r3).unwrap().sum_all().unwrap();
    check("residual x_in", &xi, &f_res);
    check("residual x_out", &xo, &f_res);
    check("residual gate", &gg, &f_res);
    // CB read-out (CPU gather) by W: loss = sum(R4 ⊙ cb_readout(c, W)), c a sparse code; and it equals the dense matmul
    let gu = 64usize;
    let code: Vec<f64> = unit_noise(n * gu, 11).iter().enumerate().map(|(j, &z)| if j % 7 == 0 { (z as f64).abs() + 0.1 } else { 0.0 }).collect();
    let code = Tensor::from_vec(code, (n, gu), &CPU).unwrap();
    let w = Var::from_tensor(&f64t(&[gu, d], 12, 0.3, 0.0)).unwrap();
    let r4 = f64t(&[n, d], 13, 1.0, 0.0);
    let dense = code.matmul(w.as_tensor()).unwrap();
    let gathered = cb_readout(&code, w.as_tensor()).unwrap();
    let dd = dense.sub(&gathered).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f64>().unwrap();
    assert!(dd < 1e-12, "gather read-out vs dense: {dd}");
    check("read-out W", &w, &|| cb_readout(&code, w.as_tensor()).unwrap().mul(&r4).unwrap().sum_all().unwrap());
}

/// (2) The whole model, f32: autograd's directional derivatives against finite differences (see the top of the file).
#[test]
fn v51b_model_gradients_match_finite_differences_f32() {
    let flags = "--head-size 32 --timescales fixed --value-residual --kwta 1 --mlp-act relu2 --residual dense --resid-fix \
                 --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";
    let (b, t) = (2, 16);
    let toks = tokens(b, t + 1, 258, 91);
    let ids: Vec<u32> = (0..b).flat_map(|r| toks[r * (t + 1)..r * (t + 1) + t].to_vec()).collect();
    let tg: Vec<u32> = (0..b).flat_map(|r| toks[r * (t + 1) + 1..(r + 1) * (t + 1)].to_vec()).collect();
    let ids = Tensor::from_vec(ids, (b, t), &CPU).unwrap();
    let tg = Tensor::from_vec(tg, (b, t), &CPU).unwrap();
    let cases: [(&str, &[&str]); 4] = [
        ("--thalamus", &["th.read", "th.gate", "th.bias", "b0.mlp.w1", "b1.mlp.w2", "b1.hebb.wv", "final.norm"]),
        // th.tdb (one entry) has a derivative near the f32 forward's noise here (3.5e-5): the f64 op check holds the
        // gate bias, as v50 found for nm.bn
        ("--thalamus --loop 2", &["th.tdg", "th.read", "th.gate", "loop.wo"]),
        ("--cerebellum-head 256", &["cbh.wo"]),
        ("--thalamus --cerebellum-head 256", &["cbh.wo"]),
    ];
    for (extra, names) in cases {
        let ar = Arch::from_flags(&format!("{flags} {extra}")).unwrap();
        let cfg = Config::with_kind(Kind::D, 258, 64, 2, 32).with_arch(ar).unwrap();
        let (vm, m) = build_awake(cfg, 52);
        for i in 0..cfg.layers {
            set(&vm, &format!("b{i}.mem.win"), &Tensor::zeros((64, 64), DType::F32, &CPU).unwrap());
        }
        let loss = || m.loss_aux(&ids, &tg, 0.0).unwrap().0;
        let grads = loss().backward().unwrap();
        for (j, name) in names.iter().enumerate() {
            let base = get(&vm, name);
            let ga = grads.get(vm.data().lock().unwrap()[*name].as_tensor()).unwrap_or_else(|| panic!("{extra}: no gradient for {name}")).copy().unwrap();
            let spread = base.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt().max(1e-3) as f64;
            let g_rms = ga.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            assert!(g_rms > 0.0, "{extra}: zero gradient for {name}");
            let full = ga.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() as f64 / g_rms as f64;
            for dir in 0..3u64 {
                let u = if dir == 0 {
                    ga.affine(1.0 / g_rms as f64, 0.0).unwrap()
                } else {
                    Tensor::from_vec(unit_noise(base.elem_count(), 700 + 10 * j as u64 + dir), base.dims(), &CPU).unwrap()
                };
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
                let (of_full, of_own) = ((fd - ad).abs() / full, (fd - ad).abs() / ad.abs().max(1e-12));
                eprintln!("v51b f32 model gradient check {extra:<34} {name:<12} dir {dir}: autograd {ad:+.6e} fd {fd:+.6e} (+-{fd_err:.1e}) err/full {of_full:.2e} err/own {of_own:.2e}");
                assert!(if dir == 0 { of_own <= 2e-2 } else { of_full <= 1e-2 }, "{extra} {name} dir {dir}: autograd {ad} vs finite difference {fd}");
            }
        }
    }
}

/// The chat check: Ava 0.4's recipe with an engram plus each option: pieces, the one-token stream and the batched feed
/// give the full forward's logits (within 1e-4), and a saved and loaded state continues as the unsaved one.
#[test]
fn v51b_options_stream_match_forward() {
    let dir = scratch("v51b_stream");
    for extra in ["--thalamus", "--cerebellum-head 512", "--thalamus --cerebellum-head 512", "--thalamus --loop 2 --neuromod 3"] {
        let cfg = Config::with_kind(Kind::D, 258, 64, 2, 64).with_engram(1).unwrap().with_arch(arch(extra)).unwrap();
        let (_, m) = build_awake(cfg, 48);
        let (b, t) = (2, 64);
        let toks = tokens(b, t, 258, 72);
        let want = one_pass(&m, &toks, b, t);
        let mut st = m.state(b).unwrap();
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
        let mut s1 = m.state(b).unwrap();
        let first: Vec<u32> = toks.iter().enumerate().filter(|(i, _)| i % t < 40).map(|(_, &x)| x).collect();
        drop(in_pieces(&m, &mut s1, &first, b, 40, &[40]));
        let p = dir.join(format!("st{}.safetensors", extra.len()));
        s1.save(&m, &p).unwrap();
        let mut s2 = State::load(&m, &p, &CPU).unwrap();
        let rest: Vec<u32> = toks.iter().enumerate().filter(|(i, _)| i % t >= 40).map(|(_, &x)| x).collect();
        let a1 = in_pieces(&m, &mut s1, &rest, b, t - 40, &[t - 40]);
        let a2 = in_pieces(&m, &mut s2, &rest, b, t - 40, &[t - 40]);
        worst[3] = max_diff(&a1, &a2).max(max_diff(&a1, &want.narrow(1, 40, t - 40).unwrap()));
        eprintln!("v51b chat check {extra:<34}: pieces {:.2e}, stream (every token) {:.2e}, feed {:.2e}, saved state {:.2e}", worst[0], worst[1], worst[2], worst[3]);
        for (what, w) in ["pieces", "stream", "feed", "saved state"].iter().zip(worst) {
            assert!(w < 1e-4, "{extra} {what}: {w}");
        }
    }
}

/// Checkpoint round trip: the varmap saved with its arch.txt and loaded by ckpt::load_auto gives the same logits bit
/// for bit (the relay's learned reads load; the cerebellum's wiring is rebuilt from its seed).
#[test]
fn v51b_checkpoint_round_trip() {
    let dir = scratch("v51b_ckpt");
    for (k, extra) in ["--thalamus", "--cerebellum-head 512", "--thalamus --cerebellum-head 512"].iter().enumerate() {
        let cfg = Config::with_kind(Kind::D, 258, 64, 2, 64).with_engram(1).unwrap().with_arch(arch(extra)).unwrap();
        let (vm, m) = build_awake(cfg, 70 + k as u64);
        let sub = dir.join(format!("m{k}"));
        std::fs::create_dir_all(&sub).unwrap();
        let p = sub.join("model.safetensors");
        vm.save(&p).unwrap();
        std::fs::write(sub.join("arch.txt"), cfg.arch.flags() + "\n").unwrap();
        let (cfg2, _vm2, m2) = crate::ckpt::load_auto(&p, Kind::D, 64, &CPU).unwrap();
        assert_eq!(cfg2.arch, cfg.arch, "{extra}");
        m2.set_track_activity(false);
        let ids = Tensor::from_vec(tokens(2, 32, 258, 9), (2, 32), &CPU).unwrap();
        let d = max_diff(&m.forward(&ids).unwrap(), &m2.forward(&ids).unwrap());
        eprintln!("v51b checkpoint round trip {extra}: max |logit difference| {d:e}");
        assert_eq!(d, 0.0, "{extra}");
    }
}

/// The cerebellum's online rule: one step equals -lr times autograd's gradient of that word's cross-entropy with
/// respect to W; the table's correction equals the folded read-out's; on a text read three times the loss of the
/// later readings goes down.
#[test]
fn cerebellum_online_rule_is_the_gradient_and_learns() {
    let cfg = Config::with_kind(Kind::D, 258, 64, 2, 64).with_engram(1).unwrap().with_arch(arch("--cerebellum-head 512")).unwrap();
    let (vm, m) = build_awake(cfg, 33);
    set(&vm, "cbh.wo", &Tensor::zeros((512, 64), DType::F32, &CPU).unwrap());
    let toks = tokens(1, 40, 258, 4);
    let lr = 0.5f32;
    // (a) the gradient: the logits of 20 tokens, then token 20's cross-entropy against token 21, by autograd
    let mut st = m.state(1).unwrap();
    drop(m.forward_with_state(&Tensor::from_vec(toks[..20].to_vec(), (1, 20), &CPU).unwrap(), &mut st).unwrap());
    let mut st2 = st.clone();
    let l = m.forward_with_state(&Tensor::new(&[[toks[20]]], &CPU).unwrap(), &mut st).unwrap();
    let ce = candle_nn::loss::cross_entropy(&l, &Tensor::new(&[toks[21]], &CPU).unwrap()).unwrap();
    let g = ce.backward().unwrap();
    let gw = g.get(vm.data().lock().unwrap()["cbh.wo"].as_tensor()).unwrap().copy().unwrap();
    // the same step through the online table
    m.cb_online_begin(1, lr).unwrap();
    let l2 = m.forward_with_state(&Tensor::new(&[[toks[20]]], &CPU).unwrap(), &mut st2).unwrap();
    assert_eq!(max_diff(&l, &l2), 0.0, "an empty table adds nothing");
    m.cb_learn(&l2, &[toks[21]]).unwrap();
    m.cb_consolidate(0).unwrap();
    let step = get(&vm, "cbh.wo");
    let want = gw.affine(-(lr as f64), 0.0).unwrap();
    let err = max_diff(&step, &want);
    let size = want.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    eprintln!("v51b cerebellum online rule vs -lr x autograd gradient: max |difference| {err:.2e} (largest entry {size:.2e})");
    assert!(size > 0.0 && err <= 1e-5 * size.max(1.0), "online step {err} vs gradient");
    // (b) the table's correction = the folded read-out's: learn on a few words, then compare logits both ways
    set(&vm, "cbh.wo", &Tensor::zeros((512, 64), DType::F32, &CPU).unwrap());
    m.cb_online_begin(1, lr).unwrap();
    let mut s = m.state(1).unwrap();
    for j in 0..10 {
        let l = m.forward_with_state(&Tensor::new(&[[toks[j]]], &CPU).unwrap(), &mut s).unwrap();
        m.cb_learn(&l, &[toks[j + 1]]).unwrap();
    }
    let mut sa = s.clone();
    let mut sb = s.clone();
    let la = m.forward_with_state(&Tensor::new(&[[toks[10]]], &CPU).unwrap(), &mut sa).unwrap();
    m.cb_consolidate(0).unwrap();
    let lb = m.forward_with_state(&Tensor::new(&[[toks[10]]], &CPU).unwrap(), &mut sb).unwrap();
    let d = max_diff(&la, &lb);
    eprintln!("v51b cerebellum table vs folded read-out: max |logit difference| {d:.2e}");
    assert!(d < 1e-5, "table vs folded: {d}");
    m.cb_online_end();
    // (c) it learns: a 40-token text read three times in one stream, the online rule on every word; the rate is
    // scaled by the mean squared norm of her word vectors (a step of eta moves the right word's logit by about eta (1 - p)
    // for the same code), eta 0.5
    set(&vm, "cbh.wo", &Tensor::zeros((512, 64), DType::F32, &CPU).unwrap());
    let e2 = get(&vm, "tok_emb").sqr().unwrap().sum(1).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
    m.cb_online_begin(1, 0.5 / e2).unwrap();
    let mut s = m.state(1).unwrap();
    let mut per = Vec::new();
    for _ in 0..3 {
        let mut tot = 0f64;
        for j in 0..39 {
            let l = m.forward_with_state(&Tensor::new(&[[toks[j]]], &CPU).unwrap(), &mut s).unwrap();
            tot += m.cb_learn(&l, &[toks[j + 1]]).unwrap();
        }
        per.push(tot / 39.0);
    }
    m.cb_online_end();
    eprintln!("v51b cerebellum online, a text read 3 times: mean cross-entropy per reading {per:?}");
    assert!(per[2] < per[0] - 0.05, "the online rule should lower the loss on a repeated text: {per:?}");
}

/// The parts' signals for the chat app: the relay's gates per block and the cerebellum's correction, and the
/// signals change nothing in the logits.
#[test]
fn part_signals_are_read_only() {
    let cfg = Config::with_kind(Kind::D, 258, 64, 2, 64).with_engram(1).unwrap().with_arch(arch("--thalamus --cerebellum-head 512 --neuromod 3")).unwrap();
    let (_, m) = build_awake(cfg, 21);
    let toks = tokens(1, 12, 258, 8);
    let mut s1 = m.stream().unwrap();
    let mut s2 = m.stream().unwrap();
    let mut s3 = m.stream().unwrap();
    m.set_part_signals(false);
    let plain: Vec<Tensor> = toks.iter().map(|&t| m.step(&mut s1, t).unwrap()).collect();
    m.set_part_signals(true);
    for (j, &t) in toks.iter().enumerate() {
        let l = m.step(&mut s2, t).unwrap();
        assert_eq!(max_diff(&l, &plain[j]), 0.0);
        let sg = m.part_signals();
        assert_eq!(sg.th.len(), 2);
        assert!(sg.cb.is_some_and(|x| x > 0.0) && sg.nm.is_some());
        // the chat app's traced step, with the signals on: the same logits, the same signals
        let (lt, _) = m.step_trace(&mut s3, t).unwrap();
        assert_eq!(max_diff(&lt, &plain[j]), 0.0);
        let st = m.part_signals();
        assert_eq!((st.th, st.cb, st.nm), (sg.th, sg.cb, sg.nm));
    }
    assert!(m.cb_rate(0.3).unwrap() > 0.0 && !m.cb_online_on());
    m.set_part_signals(false);
    assert_eq!(m.part_flags(), [true, true, true, true, true]);
}

/// Merge (runs/v51/merge_b.md): the fused steps refuse what they do not compute. The fused WIN training step takes Ava
/// 0.4b's recipe with or without --neuromod 3 (v51 fused N), and TH and CB since v51 fast2 (never with --loop); the fused
/// talking step takes TH and --neuromod since merge_f (runs/v51/merge_f.md), refuses CB, and a plastic session while it
/// is on.
#[test]
fn fused_steps_refuse_v51b_and_sessions() {
    let c = |extra: &str| Config::with_kind(Kind::D, 258, 64, 2, 16).with_arch(arch(extra)).unwrap();
    assert!(crate::model::fwin::step_ok(&c("")), "0.4b's recipe trains on the fused WIN step");
    assert!(crate::model::fwin::step_ok(&c("--neuromod 3")), "and with --neuromod 3 (fused N)");
    // merge_f: this tiny model (d64, one memory head) is refused by the talking step's own checks anyway; the base
    // model's reason is the yardstick: TH and --neuromod must add no reason of their own, CB must give its own
    let base = {
        let (_, m) = build_awake(c(""), 3);
        m.set_int8(true).unwrap();
        m.set_fused(true);
        crate::model::fstep::arch::blocker(&m)
    };
    for extra in ["--thalamus", "--cerebellum-head 512", "--thalamus --cerebellum-head 512", "--thalamus --cerebellum-head 512 --neuromod 3", "--thalamus --neuromod 3", "--cerebellum-head 512 --neuromod 3"] {
        let cfg = c(extra);
        // v51 fast2 (src/model/fwin_f2.rs): the fused WIN step now trains TH and CB too
        assert!(crate::model::fwin::step_ok(&cfg), "{extra}: the fused WIN step takes it (fast2)");
        assert!(!crate::model::fwin::step_ok(&c(&format!("{extra} --loop 2"))), "{extra} --loop 2: the fused WIN step must refuse it");
        let (_, m) = build_awake(cfg, 3);
        m.set_int8(true).unwrap();
        m.set_fused(true);
        // merge_f (runs/v51/merge_f.md): the talking step now takes TH (v51 fast2) and --neuromod (v51 fast1); v51 fast3:
        // and CB, but not while the cerebellum's online table is on (its own reason)
        let why = crate::model::fstep::arch::blocker(&m);
        assert_eq!(why, base, "{extra}: TH, CB and --neuromod add no reason to refuse the fused talking step");
        if extra.contains("cerebellum") {
            m.cb_online_begin(1, 0.1).unwrap();
            let on = crate::model::fstep::arch::blocker(&m);
            assert!(on.is_some() && on != base, "{extra}: the online table keeps the plain step for its own reason ({on:?})");
            assert!(m.fused_blocker().is_some(), "{extra}: the fused talking step must refuse an online table");
            m.cb_online_end();
            assert_eq!(crate::model::fstep::arch::blocker(&m), base, "{extra}: after the online table");
        }
        m.set_fused(false);
        m.set_int8(false).unwrap();
    }
    let (_, m) = build_awake(c(""), 3);
    m.set_int8(true).unwrap();
    m.set_fused(true);
    let before = m.fused_blocker();
    m.plastic_begin(1, &[1], &[crate::eprop::Site::Win]).unwrap();
    assert!(m.fused_blocker().is_some(), "a plastic session runs on the State path");
    m.plastic_end();
    assert_eq!(m.fused_blocker(), before);
    // W1 / W2 sessions under v51's MLP options would never learn (their MLP has no plastic hook): refused
    let (_, m) = build_awake(c("--dendrite"), 3);
    assert!(m.plastic_begin(1, &[1], &[crate::eprop::Site::W1]).is_err());
    assert!(m.plastic_begin(1, &[1], &[crate::eprop::Site::Win, crate::eprop::Site::Wout]).is_ok());
    m.plastic_end();
}

/// Merge (runs/v51/merge_b.md): the nucleus' signals (the chat app's midbrain) reach the part signals on every path a
/// --neuromod brain reads or talks on: the window, the batched feed, the one-token State step and the chat app's traced
/// step, in f32 and with int8 and the fused step on (as ava_live runs her); with and without TH and CB.
/// merge_f (runs/v51/merge_f.md): since v51 fast1 the fused talking step takes --neuromod (alone, without CB or --loop),
/// so it must keep the signals itself: they are cleared before every step here (a path that keeps none fails), and the
/// fused step must have run for "--neuromod 3".
#[test]
fn nucleus_signals_reach_every_talking_path() {
    for extra in ["--neuromod 3", "--thalamus --cerebellum-head 512 --neuromod 3", "--neuromod 3 --loop 2"] {
        let cfg = Config::with_kind(Kind::D, 258, 64, 2, 64).with_arch(arch(extra)).unwrap();
        let (_, m) = build_awake(cfg, 23);
        assert!(m.part_flags()[2], "{extra}");
        let toks = tokens(1, 12, 258, 9);
        m.set_part_signals(true);
        m.forward(&Tensor::from_vec(toks.clone(), (1, 12), &CPU).unwrap()).unwrap();
        let win = m.part_signals().nm.expect("the window: nm");
        let mut s = m.stream_batched().unwrap();
        m.feed(&mut s, &toks).unwrap();
        let fed = m.part_signals().nm.expect("the batched feed: nm");
        for (int8, fused) in [(false, false), (true, false), (true, true)] {
            m.set_int8(int8).unwrap();
            m.set_fused(fused);
            let (mut s1, mut s2) = (m.stream().unwrap(), m.stream().unwrap());
            let mut last = [0f32; 3];
            let ran = crate::model::fstep::arch::RAN.with(|r| r.get());
            for &t in &toks {
                m.set_part_signals(true); // merge_f: cleared before each step
                let l1 = m.step(&mut s1, t).unwrap();
                let a = m.part_signals().nm.expect("the one-token step: nm");
                m.set_part_signals(true);
                let (l2, _) = m.step_trace(&mut s2, t).unwrap();
                let b = m.part_signals().nm.expect("the traced step: nm");
                assert_eq!(max_diff(&l1, &l2), 0.0, "{extra}");
                assert_eq!(a, b, "{extra}: the traced step's signals are the step's");
                last = a;
            }
            let fused_steps = crate::model::fstep::arch::RAN.with(|r| r.get()) - ran;
            let want = if int8 && fused && extra == "--neuromod 3" && m.fused_blocker().is_none() { 2 * toks.len() as u64 } else { 0 };
            if int8 && fused && extra == "--neuromod 3" {
                eprintln!("{extra}: fused talking steps {fused_steps} ({:?})", m.fused_blocker());
            }
            assert_eq!(fused_steps, want, "{extra} int8 {int8} fused {fused}: the fused talking steps");
            // the last token's signals: the same token on every path (int8 rounds the weights: a looser bar)
            let tol = if int8 { 2e-2 } else { 1e-4 };
            for (k, (&x, (&w, &f))) in last.iter().zip(win.iter().zip(&fed)).enumerate() {
                assert!((x - w).abs() <= tol && (x - f).abs() <= tol, "{extra} int8 {int8}: signal {k}: step {x}, window {w}, feed {f}");
            }
        }
        m.set_fused(false);
        m.set_int8(false).unwrap();
        m.set_part_signals(false);
    }
}

/// Ridders' method (Numerical Recipes 5.7), as src/v50/tests.rs.
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
