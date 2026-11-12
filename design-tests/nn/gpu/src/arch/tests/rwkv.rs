// TESTING / DESIGN ONLY — tests of v45's four ideas from RWKV-7 and Qwen3-Next (src/arch.rs W1-W4), on the CPU
// (two ignored tests on the GPU, one ignored CPU speed test).

use super::*;
use crate::ckpt::{load_auto, write_arch_beside};
use crate::testkit::scratch;
use candle_core::Var;

/// The v42 "healthy" block options of experiment 17p (the v44 and v45 baseline).
const HEALTHY: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
                       --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";

fn build_e(arch: Arch, d: usize, layers: usize, t: usize, engram: Option<usize>) -> (Config, VarMap, Model) {
    let mut cfg = Config::with_kind(Kind::D, V, d, layers, t);
    if let Some(l) = engram {
        cfg = cfg.with_engram(l).unwrap();
    }
    let cfg = cfg.with_arch(arch).unwrap();
    let varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu)).unwrap();
    (cfg, varmap, model)
}

fn rel(a: &Tensor, b: &Tensor) -> f32 {
    let s = b.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap().max(1e-6);
    max_diff(a, b) / s
}

/// Give the new parts values that are not their start (so round trips and the talking path test something).
fn wake(vm: &VarMap, seed: u64) {
    let names: Vec<(String, Vec<usize>)> = vm.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.dims().to_vec())).collect();
    for (j, (k, dims)) in names.iter().enumerate() {
        let scale = if k.ends_with(".wa2") || k.ends_with(".bonus") || k.ends_with(".shift") || k.ends_with(".wth") {
            0.8
        } else if k.ends_with(".eng.wo") {
            0.05
        } else {
            continue;
        };
        set_var(vm, k, &noise(dims, seed + j as u64).affine(scale, 0.0).unwrap());
    }
}

/// Off by default; the flags read back; bad combinations are refused; --gates head is today's block; the baseline's
/// signature is unchanged.
#[test]
fn rwkv_options_parse_and_are_off_by_default() {
    let z = Arch::default();
    assert!(!z.bonus && !z.token_shift && z.eng_gate == EngGate::Off && z.gates == Gating::Head);
    assert_eq!(Arch::from_flags("--gates head --eng-gate off").unwrap(), z, "--gates head is today's per-head block");
    let a = Arch::from_flags("--gates kda --bonus --token-shift --eng-gate elem").unwrap();
    assert_eq!((a.gates, a.bonus, a.token_shift, a.eng_gate), (Gating::Kda, true, true, EngGate::Elem));
    assert_eq!(a.signature(), " gates=kda bonus=1 token_shift=1 eng_gate=elem");
    assert_eq!(Arch::from_flags(&a.flags()).unwrap(), a);
    assert!(a.needs_path());
    for one in ["--gates kda", "--bonus", "--token-shift", "--eng-gate head"] {
        let x = Arch::from_flags(one).unwrap();
        assert!(x.needs_path() && !x.is_default(), "{one}");
        assert_eq!(Arch::from_flags(&x.flags()).unwrap(), x, "{one}");
    }
    let h = Arch::from_flags(HEALTHY).unwrap();
    assert!(h.signature().contains(" gates=channel ") && !h.signature().contains("kda"), "the 17p signature is unchanged");
    let kda = Arch::from_flags(&format!("{HEALTHY} --gates kda")).unwrap();
    assert_eq!(kda.gates, Gating::Kda, "the last --gates wins");
    assert!(Arch::from_flags("--gates diag").is_err() && Arch::from_flags("--eng-gate both").is_err());
    let c = Config::with_kind(Kind::D, V, 64, 2, 16);
    assert!(c.with_arch(Arch { eng_gate: EngGate::Head, ..z }).is_err(), "--eng-gate needs --engram");
    assert!(c.with_engram(1).unwrap().with_arch(Arch { eng_gate: EngGate::Head, ..z }).is_ok());
    assert!(Config::with_kind(Kind::S, V, 64, 2, 16).with_arch(Arch { bonus: true, ..z }).is_err(), "model D only");
}

/// W1 at its start: with the per-channel part at 0 (alpha the same over a head's channels), KDA = P2 = today's per-head
/// rule, with and without momentum. With b = 1, reading k right after writing (k, v) gives v exactly under KDA for any
/// per-channel fades, and not under P2's order. The chunked form equals the token loop (values, end state, gradients).
#[test]
fn kda_rule_start_recall_and_chunks() {
    let unit = |x: Tensor| crate::model::l2_norm(&x).unwrap();
    let (b, nh, t, hs) = (2usize, 2usize, 24usize, 8usize);
    let q = unit(noise(&[b, nh, t, hs], 1));
    let k = unit(noise(&[b, nh, t, hs], 2));
    let v = noise(&[b, nh, t, hs], 3);
    let lah = log_sigmoid(&noise(&[b, nh, t, 1], 4).affine(2.0, 2.5).unwrap()).unwrap();
    let bh = sigmoid(&noise(&[b, nh, t, 1], 5)).unwrap().affine(2.0, 0.0).unwrap();
    let lac = lah.broadcast_as((b, nh, t, hs)).unwrap().contiguous().unwrap();
    let bc = bh.broadcast_as((b, nh, t, hs)).unwrap().contiguous().unwrap();
    for m in [0.0, 0.5] {
        let (yh, _) = delta_rec(&q, &k, &k, &v, &lah, &bh, m, None).unwrap();
        let (yc, _) = delta_rec(&q, &k, &k, &v, &lac, &bc, m, None).unwrap();
        let (yk, _) = kda_rec(&q, &k, &v, &lac, &bh, m, None).unwrap();
        assert!(rel(&yk, &yh) <= 1e-5 && rel(&yc, &yh) <= 1e-5, "M {m}: kda {} channel {}", rel(&yk, &yh), rel(&yc, &yh));
    }
    // exact recall at b = 1 with fades that differ per channel
    let la = log_sigmoid(&noise(&[b, nh, t, hs], 6).affine(4.0, 1.0).unwrap()).unwrap();
    let ones = Tensor::ones((b, nh, t, 1), DType::F32, &Device::Cpu).unwrap();
    let (yk, _) = kda_rec(&k, &k, &v, &la, &ones, 0.0, None).unwrap();
    assert!(max_diff(&yk, &v) <= 1e-5, "KDA recalls v exactly: {}", max_diff(&yk, &v));
    let (yp, _) = delta_rec(&k, &k, &k, &v, &la, &ones.broadcast_as((b, nh, t, hs)).unwrap().contiguous().unwrap(), 0.0, None).unwrap();
    assert!(max_diff(&yp, &v) > 1e-3, "P2's order does not: {}", max_diff(&yp, &v));
    // the chunks against the loop, from zero and from a carried state, with and without momentum (t not a multiple of 16)
    for (m, t2) in [(0.0, 24usize), (0.5, 40), (0.7, 16)] {
        let q = unit(noise(&[b, nh, t2, hs], 11));
        let k = unit(noise(&[b, nh, t2, hs], 12));
        let v = noise(&[b, nh, t2, hs], 13);
        let la = log_sigmoid(&noise(&[b, nh, t2, hs], 14).affine(3.0, 1.0).unwrap()).unwrap();
        let bw = sigmoid(&noise(&[b, nh, t2, 1], 15)).unwrap().affine(2.0, 0.0).unwrap();
        let init = MemState {
            f: noise(&[b * nh, hs, hs], 16).affine(0.3, 0.0).unwrap(),
            s: if m > 0.0 { Some(noise(&[b * nh, hs, hs], 17).affine(0.3, 0.0).unwrap()) } else { None },
            prev: None,
        };
        for st in [None, Some(&init)] {
            let (y1, s1) = kda_rec(&q, &k, &v, &la, &bw, m, st).unwrap();
            let (y2, s2) = kda_chunk(&q, &k, &v, &la, &bw, m, st).unwrap();
            let tag = format!("M {m} t {t2} carried {}", st.is_some());
            assert!(rel(&y2, &y1) < 1e-4 && rel(&s2.f, &s1.f) < 1e-4, "{tag}: y {} F {}", rel(&y2, &y1), rel(&s2.f, &s1.f));
            if let (Some(a), Some(c)) = (&s2.s, &s1.s) {
                assert!(rel(a, c) < 1e-4, "{tag}: S {}", rel(a, c));
            }
        }
        let vars: Vec<Var> = [&q, &k, &v, &la, &bw].iter().map(|x| Var::from_tensor(x).unwrap()).collect();
        let (ry, rf) = (noise(&[b, nh, t2, hs], 18), noise(&[b * nh, hs, hs], 19));
        let loss = |f: fn(&Tensor, &Tensor, &Tensor, &Tensor, &Tensor, f64, Option<&MemState>) -> candle_core::Result<(Tensor, MemState)>| {
            let x: Vec<Tensor> = vars.iter().map(|v| v.as_tensor().clone()).collect();
            let (y, st) = f(&x[0], &x[1], &x[2], &x[3], &x[4], m, Some(&init)).unwrap();
            y.mul(&ry).unwrap().sum_all().unwrap().add(&st.f.mul(&rf).unwrap().sum_all().unwrap()).unwrap()
        };
        let (g1, g2) = (loss(kda_rec).backward().unwrap(), loss(kda_chunk).backward().unwrap());
        for (i, var) in vars.iter().enumerate() {
            let (a, c) = (g2.get(var.as_tensor()).unwrap(), g1.get(var.as_tensor()).unwrap());
            assert!(rel(a, c) < 2e-4, "M {m}: grad {i} {}", rel(a, c));
        }
    }
}

/// Each option at its start is today's block (the same weights copied in): logits within 1e-5, on today's D and on the
/// 17p healthy set (W1 there replaces --gates channel, as in the comparison). W4 is checked with the engram's Wo not 0
/// (it starts at 0, which would hide any difference). Then the new parts, once not 0, change the output.
#[test]
fn rwkv_options_start_as_todays_block() {
    let h = Arch::from_flags(HEALTHY).unwrap();
    let z = Arch::default();
    let cases: Vec<(&str, Arch, Arch, usize, Option<usize>)> = vec![
        ("kda on D", z, Arch { gates: Gating::Kda, ..z }, 64, None),
        ("kda momentum on D", Arch { mem_momentum: 0.5, ..z }, Arch { gates: Gating::Kda, mem_momentum: 0.5, ..z }, 64, None),
        ("bonus on D", z, Arch { bonus: true, ..z }, 64, None),
        ("token_shift on D", z, Arch { token_shift: true, ..z }, 64, None),
        ("kda for channel on healthy", h, Arch { gates: Gating::Kda, ..h }, 128, None),
        ("bonus on healthy", h, Arch { bonus: true, ..h }, 128, None),
        ("token_shift on healthy", h, Arch { token_shift: true, ..h }, 128, None),
        ("eng_gate head on healthy", h, Arch { eng_gate: EngGate::Head, ..h }, 128, Some(1)),
        ("eng_gate elem on healthy", h, Arch { eng_gate: EngGate::Elem, ..h }, 128, Some(1)),
    ];
    for (name, base, arch, d, eng) in cases {
        let (_, vm0, m0) = build_e(base, d, 2, 16, eng);
        let (_, vm1, m1) = build_e(arch, d, 2, 16, eng);
        if eng.is_some() {
            set_var(&vm0, "b1.eng.wo", &noise(&[d, d], 77).affine(0.05, 0.0).unwrap());
        }
        // today's tensors into the option's model (KDA on healthy: from the option's model, as P2's wb1 has no twin)
        let n = if arch.gates == Gating::Kda && base.gates == Gating::Channel { copy_shared(&vm1, &vm0) } else { copy_shared(&vm0, &vm1) };
        assert!(n > 10, "{name}: {n} shared");
        let (ids, _) = batch(2, 16, 5);
        let (a, b) = (m0.forward(&ids).unwrap(), m1.forward(&ids).unwrap());
        assert!(max_diff(&a, &b) <= 1e-5, "{name}: {}", max_diff(&a, &b));
        wake(&vm1, 90);
        let c = m1.forward(&ids).unwrap();
        assert!(max_diff(&a, &c) > 1e-4, "{name}: the new parts change the output");
    }
}

/// Central finite differences (f64) against autograd: W1's rule (kda_rec, by q, k, v, log alpha, b, with momentum),
/// W2's bonus (by rho, q, v), W3's mix (by mu, h and the carried token), W4's gate (by W_th per head and per number,
/// and the input).
#[test]
fn rwkv_parts_gradients_match_finite_differences() {
    let f64t = |shape: &[usize], seed: u64, scale: f64| noise(shape, seed).to_dtype(DType::F64).unwrap().affine(scale, 0.0).unwrap();
    let eps = 1e-6;
    let check = |what: &str, var: &Var, ga: &Tensor, f: &dyn Fn() -> f64| {
        let base = var.as_tensor().copy().unwrap();
        let flat: Vec<f64> = base.flatten_all().unwrap().to_vec1().unwrap();
        let gv: Vec<f64> = ga.flatten_all().unwrap().to_vec1().unwrap();
        for j in (0..flat.len()).step_by(3) {
            let mut p = flat.clone();
            p[j] += eps;
            var.set(&Tensor::from_vec(p.clone(), base.dims(), &Device::Cpu).unwrap()).unwrap();
            let up = f();
            p[j] -= 2.0 * eps;
            var.set(&Tensor::from_vec(p, base.dims(), &Device::Cpu).unwrap()).unwrap();
            let down = f();
            var.set(&base).unwrap();
            let fd = (up - down) / (2.0 * eps);
            assert!((fd - gv[j]).abs() <= 1e-6 * (1.0 + fd.abs()), "{what}[{j}]: finite difference {fd} vs autograd {}", gv[j]);
        }
    };
    // W1: loss = sum(R ⊙ y) of kda_rec, momentum 0.5 (q and k need not be unit length for the rule itself)
    let (b, nh, t, hs) = (1usize, 2usize, 5usize, 4usize);
    let q = Var::from_tensor(&f64t(&[b, nh, t, hs], 1, 0.6)).unwrap();
    let k = Var::from_tensor(&f64t(&[b, nh, t, hs], 2, 0.6)).unwrap();
    let v = Var::from_tensor(&f64t(&[b, nh, t, hs], 3, 1.0)).unwrap();
    let la = Var::from_tensor(&f64t(&[b, nh, t, hs], 4, 0.3).affine(1.0, -0.4).unwrap()).unwrap();
    let bw = Var::from_tensor(&f64t(&[b, nh, t, 1], 5, 0.3).affine(1.0, 1.0).unwrap()).unwrap();
    let r = f64t(&[b, nh, t, hs], 6, 1.0);
    let w1 = || kda_rec(q.as_tensor(), k.as_tensor(), v.as_tensor(), la.as_tensor(), bw.as_tensor(), 0.5, None).unwrap().0.mul(&r).unwrap().sum_all().unwrap();
    let g = w1().backward().unwrap();
    let f1 = || w1().to_scalar::<f64>().unwrap();
    for (name, x) in [("W1 q", &q), ("W1 k", &k), ("W1 v", &v), ("W1 log alpha", &la), ("W1 b", &bw)] {
        check(name, x, g.get(x.as_tensor()).unwrap(), &f1);
    }
    // W2: loss = sum(R ⊙ bonus_add(y, q, k, v, rho))
    let y0 = f64t(&[b, nh, t, hs], 7, 1.0);
    let rho = Var::from_tensor(&f64t(&[nh * hs], 8, 0.9)).unwrap();
    let w2 = || bonus_add(&y0, q.as_tensor(), k.as_tensor(), v.as_tensor(), rho.as_tensor()).unwrap().mul(&r).unwrap().sum_all().unwrap();
    let g2 = w2().backward().unwrap();
    let f2 = || w2().to_scalar::<f64>().unwrap();
    for (name, x) in [("W2 rho", &rho), ("W2 q", &q), ("W2 v", &v)] {
        check(name, x, g2.get(x.as_tensor()).unwrap(), &f2);
    }
    // W3: loss = sum_j sum(R_j ⊙ x_j) over the four mixes, with a carried token
    let (rows, tt, d) = (2usize, 3usize, 4usize);
    let hv = Var::from_tensor(&f64t(&[rows * tt, d], 9, 1.0)).unwrap();
    let mu = Var::from_tensor(&f64t(&[4, d], 10, 0.7)).unwrap();
    let prev = Var::from_tensor(&f64t(&[rows, d], 11, 1.0)).unwrap();
    let rr: Vec<Tensor> = (0..4).map(|j| f64t(&[rows * tt, d], 12 + j, 1.0)).collect();
    let w3 = || {
        let (xs, last) = token_shift(hv.as_tensor(), mu.as_tensor(), rows, tt, Some(prev.as_tensor())).unwrap();
        let mut s = last.sum_all().unwrap();
        for (x, rj) in xs.iter().zip(&rr) {
            s = s.add(&x.mul(rj).unwrap().sum_all().unwrap()).unwrap();
        }
        s
    };
    let g3 = w3().backward().unwrap();
    let f3 = || w3().to_scalar::<f64>().unwrap();
    for (name, x) in [("W3 mu", &mu), ("W3 h", &hv), ("W3 carried token", &prev)] {
        check(name, x, g3.get(x.as_tensor()).unwrap(), &f3);
    }
    // W4: loss = sum(R ⊙ (o ⊙ 2 sigmoid(h·W_th)) · Wo), per head (2 heads of 4) and per number
    let (n, dd) = (5usize, 8usize);
    let o = f64t(&[n, dd], 20, 1.0);
    let wo = f64t(&[dd, dd], 21, 0.5);
    let r4 = f64t(&[n, dd], 22, 1.0);
    let he = Var::from_tensor(&f64t(&[n, dd], 23, 1.0)).unwrap();
    for (tag, w) in [("head", 2usize), ("elem", dd)] {
        let wth = Var::from_tensor(&f64t(&[dd, w], 24, 0.8)).unwrap();
        let w4 = || eng_gate_apply(&o, &eng_gate_of(he.as_tensor(), wth.as_tensor()).unwrap()).unwrap().matmul(&wo).unwrap().mul(&r4).unwrap().sum_all().unwrap();
        let g4 = w4().backward().unwrap();
        let f4 = || w4().to_scalar::<f64>().unwrap();
        check(&format!("W4 {tag} W_th"), &wth, g4.get(wth.as_tensor()).unwrap(), &f4);
        check(&format!("W4 {tag} input"), &he, g4.get(he.as_tensor()).unwrap(), &f4);
    }
    // W4 per head is the per-number gate repeated over each head's channels
    let g = f64t(&[n, 2], 30, 1.0);
    let rep = Tensor::cat(&[g.narrow(1, 0, 1).unwrap().repeat((1, 4)).unwrap(), g.narrow(1, 1, 1).unwrap().repeat((1, 4)).unwrap()], 1).unwrap();
    let diff = eng_gate_apply(&o, &g).unwrap().sub(&eng_gate_apply(&o, &rep).unwrap()).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f64>().unwrap();
    assert!(diff == 0.0, "per-head gate layout {diff}");
}

/// Each option on top of the 17p healthy set trains on the CPU (a few AdamW steps on one batch lower the loss), the
/// weights made match the count, and --decay-gates covers W_th.
#[test]
fn rwkv_options_train_on_top_of_the_healthy_set() {
    let cases: [(&str, Option<usize>); 7] = [
        ("--gates kda", None),
        ("--gates head", None),
        ("--bonus", None),
        ("--token-shift", None),
        ("--gates kda --bonus --token-shift", None),
        ("--eng-gate head", Some(1)),
        ("--eng-gate elem", Some(1)),
    ];
    for (extra, eng) in cases {
        let ar = Arch::from_flags(&format!("{HEALTHY} {extra}")).unwrap();
        let (cfg, varmap, model) = build_e(ar, 128, 2, 16, eng);
        let made: usize = varmap.data().lock().unwrap().values().map(|v| v.elem_count()).sum();
        assert_eq!(made, cfg.weight_counts().0, "{extra}: weight count");
        let (ids, tg) = batch(2, 16, 9);
        let mut params: Vec<crate::optim::Param> =
            varmap.data().lock().unwrap().iter().map(|(k, v)| crate::optim::Param { name: k.clone(), var: v.clone(), decay: Model::decays_with(k, true) }).collect();
        params.sort_by(|a, b| a.name.cmp(&b.name));
        if eng.is_some() {
            assert!(params.iter().any(|p| p.name.ends_with(".wth") && p.decay), "{extra}: --decay-gates covers W_th");
        }
        let mut opt = crate::optim::AdamW::new(params, 0.0).unwrap();
        let mut losses = Vec::new();
        for _ in 0..6 {
            let (l, _) = model.loss(&ids, &tg).unwrap();
            losses.push(l.to_scalar::<f32>().unwrap());
            let g = l.backward().unwrap();
            opt.step(&g, 3e-3, 1.0).unwrap();
            model.homeostasis_step().unwrap();
        }
        assert!(losses.iter().all(|l| l.is_finite()) && losses[5] < losses[0], "{extra}: {losses:?}");
    }
    // the weights each option adds at d128, 4 blocks (the comparison's shape), against the note's numbers
    let base = Config::with_kind(Kind::D, V, 128, 4, 16).with_arch(Arch::from_flags(HEALTHY).unwrap()).unwrap().weight_counts().0;
    let count = |extra: &str, eng: Option<usize>| {
        let mut c = Config::with_kind(Kind::D, V, 128, 4, 16);
        if let Some(l) = eng {
            c = c.with_engram(l).unwrap();
        }
        c.with_arch(Arch::from_flags(&format!("{HEALTHY} {extra}")).unwrap()).unwrap().weight_counts().0 as i64
    };
    assert_eq!(count("--gates kda", None) - base as i64, -4 * (2 * 128 * 32 + 128 * 128), "kda: no W_b1, W_b2, W_e");
    assert_eq!(count("--bonus", None) - base as i64, 4 * 128);
    assert_eq!(count("--token-shift", None) - base as i64, 4 * 4 * 128);
    assert_eq!(count("--eng-gate head", Some(2)) - count("", Some(2)), 128 * 4);
    assert_eq!(count("--eng-gate elem", Some(2)) - count("", Some(2)), 128 * 128);
}

/// A model with each option (its new parts not at 0) saved with its arch.txt loads back (load_auto) with the same
/// logits, bit for bit. The talking path (forward_with_state, in two pieces, with the state saved to a file and read
/// back between them) gives forward's logits: W3 carries the last token, W1 its memory (with momentum).
#[test]
fn rwkv_checkpoint_round_trip_and_talking() {
    let _steady = crate::thread_env::steady(); // forward vs the talking path: one thread count from start to end
    let dir = scratch("rwkv_ckpt");
    let cases: [(&str, &str, Option<usize>); 6] = [
        ("kda", "--gates kda", None),
        ("kda-mom-64", "--gates kda --mem-momentum 0.5 --head-size 64", None),
        ("bonus", "--bonus", None),
        ("shift", "--token-shift", None),
        ("all3", "--gates kda --bonus --token-shift --mem-momentum 0.5", None),
        ("enghead", "--eng-gate head --token-shift", Some(1)),
    ];
    let mut cases: Vec<(&str, &str, Option<usize>)> = cases.to_vec();
    cases.push(("engelem", "--eng-gate elem", Some(1)));
    for (name, flags, eng) in cases {
        let ar = Arch::from_flags(flags).unwrap();
        let (cfg, vm, m) = build_e(ar, 64, 2, 32, eng);
        wake(&vm, 300);
        let (ids, _) = batch(2, 32, 13);
        let want = m.forward(&ids).unwrap();
        let path = dir.join(format!("{name}.safetensors"));
        vm.save(&path).unwrap();
        write_arch_beside(&path, &ar).unwrap();
        let (c2, _, m2) = load_auto(&path, Kind::D, 32, &Device::Cpu).unwrap();
        assert_eq!((c2.arch, c2.engram), (ar, eng), "{name}");
        assert_eq!(max_diff(&want, &m2.forward(&ids).unwrap()), 0.0, "{name}: logits after the round trip");
        // the talking path in two pieces, the state through a file in between
        let mut st = m.state(2).unwrap();
        let p1 = m.forward_with_state(&ids.narrow(1, 0, 7).unwrap(), &mut st).unwrap();
        let sp = dir.join(format!("{name}.state.safetensors"));
        st.save(&m, &sp).unwrap();
        let mut st2 = crate::state::State::load(&m, &sp, &Device::Cpu).unwrap();
        let p2 = m.forward_with_state(&ids.narrow(1, 7, 25).unwrap(), &mut st2).unwrap();
        let v = cfg.vocab;
        let joined = Tensor::cat(&[p1.reshape((2, 7, v)).unwrap(), p2.reshape((2, 25, v)).unwrap()], 1).unwrap().reshape((64, v)).unwrap();
        assert!(max_diff(&joined, &want) <= 1e-4, "{name}: state path {}", max_diff(&joined, &want));
        // a fresh state that skips the carry would differ (the carried token / memory matter)
        if ar.token_shift || ar.gates == Gating::Kda {
            let mut fresh = m.state(2).unwrap();
            let p2f = m.forward_with_state(&ids.narrow(1, 7, 25).unwrap(), &mut fresh).unwrap();
            assert!(max_diff(&p2f, &p2) > 1e-4, "{name}: the carried state matters");
        }
    }
}

/// GPU (run with --ignored): the healthy set with W1 (CUDA: the fused KDA kernels), W2, W3 and W4 (per head) gives the
/// CPU's loss for the same weights, and every new part gets a gradient on CUDA.
#[test]
#[ignore]
fn rwkv_gpu_matches_cpu() {
    let Ok(dev) = Device::new_cuda(0) else { return };
    for extra in ["--gates kda --bonus --token-shift --eng-gate head", "--gates kda --mem-momentum 0 --eng-gate elem"] {
        let ar = Arch::from_flags(&format!("{HEALTHY} {extra}")).unwrap();
        let (cfg, vc, mc) = build_e(ar, 128, 2, 64, Some(1));
        wake(&vc, 500);
        let vg = VarMap::new();
        let mg = Model::new(cfg, VarBuilder::from_varmap(&vg, DType::F32, &dev)).unwrap();
        for (k, v) in vc.data().lock().unwrap().iter() {
            vg.data().lock().unwrap()[k].set(&v.as_tensor().to_device(&dev).unwrap()).unwrap();
        }
        let (ids, tg) = batch(4, 64, 17);
        let (lc, _) = mc.loss(&ids, &tg).unwrap();
        let (lg, _) = mg.loss(&ids.to_device(&dev).unwrap(), &tg.to_device(&dev).unwrap()).unwrap();
        let gg = lg.backward().unwrap();
        let (lc, lgs) = (lc.to_scalar::<f32>().unwrap(), lg.to_scalar::<f32>().unwrap());
        assert!((lc - lgs).abs() <= 1e-3 * lc.abs(), "{extra}: loss CPU {lc} vs CUDA {lgs}");
        for (k, v) in vg.data().lock().unwrap().iter() {
            if k.ends_with(".wa2") || k.ends_with(".bonus") || k.ends_with(".shift") || k.ends_with(".wth") {
                let s = gg.get(v.as_tensor()).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
                assert!(s > 0.0 && s.is_finite(), "{extra}: {k} gradient on CUDA {s}");
            }
        }
        println!("rwkv gpu {extra}: loss CPU {lc} CUDA {lgs}");
    }
}

/// GPU memory and speed of one arm (run with --ignored --nocapture, AVA_V45_ARM=base|kda|head|bonus|shift|eng|enghead|
/// engelem; one arm per process, so no arm sees another's freed pool): d128 L4 at the comparison's shape but 4 rows
/// (1/8 of its batch of 32) x 256 tokens. Prints the GPU memory the forward's graph holds, the memory with the
/// gradients too, and the tok/s of 5 forward + backward steps (after one warm-up).
#[test]
#[ignore]
fn rwkv_gpu_memory_and_speed() {
    let (Ok(dev), Ok(arm)) = (Device::new_cuda(0), std::env::var("AVA_V45_ARM")) else { return };
    let (extra, eng) = match arm.as_str() {
        "base" => ("", None),
        "kda" => ("--gates kda", None),
        "head" => ("--gates head", None),
        "bonus" => ("--bonus", None),
        "shift" => ("--token-shift", None),
        "eng" => ("", Some(2)),
        "enghead" => ("--eng-gate head", Some(2)),
        "engelem" => ("--eng-gate elem", Some(2)),
        a => panic!("unknown arm {a}"),
    };
    let used = || -> u64 {
        let candle_core::Device::Cuda(c) = &dev else { unreachable!() };
        let s = c.cuda_stream();
        s.synchronize().unwrap();
        let (free, total) = s.context().mem_get_info().unwrap();
        ((total - free) >> 20) as u64
    };
    let mut cfg = Config::with_kind(Kind::D, 4112, 128, 4, 256);
    if let Some(l) = eng {
        cfg = cfg.with_engram(l).unwrap();
    }
    let cfg = cfg.with_arch(Arch::from_flags(&format!("{HEALTHY} {extra}")).unwrap()).unwrap();
    let vm = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev)).unwrap();
    let (rows, t) = (4usize, 256usize);
    let n = rows * (t + 1);
    let s: Vec<u32> = (0..n as u32).map(|i| (i.wrapping_mul(2654435761) >> 7) % 4112).collect();
    let ids = Tensor::from_vec(s.iter().enumerate().filter(|(i, _)| i % (t + 1) != t).map(|(_, x)| *x).collect::<Vec<u32>>(), (rows, t), &dev).unwrap();
    let tg = Tensor::from_vec(s.iter().enumerate().filter(|(i, _)| i % (t + 1) != 0).map(|(_, x)| *x).collect::<Vec<u32>>(), (rows, t), &dev).unwrap();
    m.set_track_activity(false);
    let step = || {
        let (l, _) = m.loss(&ids, &tg).unwrap();
        drop(l.backward().unwrap());
    };
    step();
    let m0 = used();
    let (l, _) = m.loss(&ids, &tg).unwrap();
    let m1 = used();
    let g = l.backward().unwrap();
    let m2 = used();
    drop((g, l));
    let t0 = std::time::Instant::now();
    for _ in 0..5 {
        step();
    }
    let _ = used();
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "v45 arm {arm:>8}: weights {}, graph {} MiB, graph + grads {} MiB (4 x 256 tokens), {:.0} tok/s",
        cfg.weight_counts().0,
        m1.saturating_sub(m0),
        m2.saturating_sub(m0),
        (5 * rows * t) as f64 / secs
    );
}

/// CPU speed (run with --ignored --nocapture): ms per forward + backward at d128 L4, 4 x 128 tokens, for the healthy
/// set (P2) and each v45 arm.
#[test]
#[ignore]
fn rwkv_cpu_speed() {
    let cases: [(&str, &str, Option<usize>); 8] = [
        ("healthy (channel)", "", None),
        ("kda for channel", "--gates kda", None),
        ("head for channel", "--gates head", None),
        ("+bonus", "--bonus", None),
        ("+token-shift", "--token-shift", None),
        ("+engram (block 2)", "", Some(2)),
        ("+engram +eng-gate head", "--eng-gate head", Some(2)),
        ("+engram +eng-gate elem", "--eng-gate elem", Some(2)),
    ];
    for (name, extra, eng) in cases {
        let (_, _, model) = build_e(Arch::from_flags(&format!("{HEALTHY} {extra}")).unwrap(), 128, 4, 128, eng);
        model.set_track_activity(false);
        let (ids, tg) = batch(4, 128, 1);
        let run = || {
            let (l, _) = model.loss(&ids, &tg).unwrap();
            l.backward().unwrap();
        };
        run();
        let t0 = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;
        println!("v45 cpu speed {name:>24}: {ms:8.1} ms per step ({:.0} tok/s)", 512.0 / (ms / 1000.0));
    }
}
