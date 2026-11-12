// TESTING / DESIGN ONLY — tests of v44's two ideas from Moonshot's open Kimi models (src/arch.rs K1, K2), on the CPU
// (one ignored test on the GPU, one ignored CPU speed test).

use super::*;
use crate::ckpt::{load_auto, write_arch_beside};
use crate::testkit::scratch;
use candle_core::Var;

/// The v42 "healthy" block options of experiment 17p (WIN + SLOW), the baseline of the v44 comparison.
const HEALTHY: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
                       --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";

fn build_l(arch: Arch, d: usize, layers: usize, t: usize) -> (Config, VarMap, Model) {
    let cfg = Config::with_kind(Kind::D, V, d, layers, t).with_arch(arch).unwrap();
    let varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu)).unwrap();
    (cfg, varmap, model)
}

/// Off by default; the flags read back; bad combinations are refused; the 17p baseline's options (and so its resume
/// signature and arch.txt) are what they were.
#[test]
fn kimi_options_parse_and_are_off_by_default() {
    let z = Arch::default();
    assert!(!z.attn_res && z.gate == OutGate::Silu && z.gate_rank == OUT_GATE_RANK && z.is_default());
    assert_eq!((z.signature(), z.flags()), (String::new(), String::new()));
    assert_eq!(Arch::from_flags("--residual plain --gate silu --gate-rank 32").unwrap(), z);
    let a = Arch::from_flags("--residual attn").unwrap();
    assert!(a.attn_res && !a.dense && a.needs_path());
    assert_eq!((a.signature().as_str(), a.flags().as_str()), (" residual=attn", "--residual attn"));
    let g = Arch::from_flags("--gate sigmoid --gate-rank 16").unwrap();
    assert_eq!((g.gate, g.gate_rank), (OutGate::Sigmoid, 16));
    assert_eq!(g.signature(), " gate=sigmoid gate_rank=16");
    assert_eq!(Arch::from_flags(&g.flags()).unwrap(), g);
    let both = Arch::from_flags(&format!("{HEALTHY} --residual attn --gate sigmoid")).unwrap();
    assert!(both.attn_res && !both.dense, "the last --residual wins");
    assert_eq!(Arch::from_flags(&both.flags()).unwrap(), both);
    let h = Arch::from_flags(HEALTHY).unwrap();
    assert_eq!(
        h.signature(),
        " head_size=64 timescales=fixed gates=channel value_residual=1 kwta=1/4 mlp_act=relu2 mem_momentum=0.5 resid_fix=1 \
         residual=dense lif_init=logspaced lif_scale=1 spikes=int2 alif=1 homeostasis=0.1",
        "the 17p runs' signature (their result.tsv v42 column) is unchanged"
    );
    let c = Config::with_kind(Kind::D, V, 64, 2, 16);
    assert!(c.with_arch(Arch { dense: true, attn_res: true, ..z }).is_err());
    assert!(c.with_arch(Arch { gate_rank: 16, ..z }).is_err(), "a rank without --gate sigmoid");
    assert!(c.with_arch(Arch { gate: OutGate::Sigmoid, gate_rank: 0, ..z }).is_err());
    assert!(c.with_arch(Arch { gate: OutGate::Sigmoid, gate_rank: 65, ..z }).is_err());
    assert!(c.with_arch(Arch { gate: OutGate::Sigmoid, gate_rank: 64, ..z }).is_ok());
    assert!(Config::with_kind(Kind::S, V, 64, 2, 16).with_arch(a).is_err(), "model D only");
}

/// K1 at its start (every query 0) is today's plain residual, up to rounding; the queries get a gradient at 0, and a
/// query that is not 0 changes the output. The talking path (forward_with_state, in two pieces) gives forward's logits.
#[test]
fn attn_res_starts_as_the_plain_residual() {
    let z = Arch::default();
    for layers in [2, 4] {
        let (_, vm0, m0) = build_l(z, 64, layers, 16);
        let (cfg, vm1, m1) = build_l(Arch { attn_res: true, ..z }, 64, layers, 16);
        assert_eq!(copy_shared(&vm0, &vm1), vm0.data().lock().unwrap().len());
        let (ids, tg) = batch(2, 16, 5);
        let (a, b) = (m0.forward(&ids).unwrap(), m1.forward(&ids).unwrap());
        assert!(max_diff(&a, &b) <= 1e-5, "L{layers}: {}", max_diff(&a, &b));
        let (loss, _) = m1.loss(&ids, &tg).unwrap();
        let g = loss.backward().unwrap();
        for i in 0..layers {
            let q = vm1.data().lock().unwrap()[&format!("b{i}.ares")].clone();
            let gs = g.get(q.as_tensor()).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(gs > 0.0 && gs.is_finite(), "L{layers} b{i}.ares gradient {gs}");
            set_var(&vm1, &format!("b{i}.ares"), &noise(&[64], 10 + i as u64).affine(3.0, 0.0).unwrap());
        }
        let c = m1.forward(&ids).unwrap();
        assert!(max_diff(&a, &c) > 1e-4, "L{layers}: the queries change the output");
        // the talking path, in two pieces
        let mut st = m1.state(2).unwrap();
        let p1 = m1.forward_with_state(&ids.narrow(1, 0, 7).unwrap(), &mut st).unwrap();
        let p2 = m1.forward_with_state(&ids.narrow(1, 7, 9).unwrap(), &mut st).unwrap();
        let v = cfg.vocab;
        let joined = Tensor::cat(&[p1.reshape((2, 7, v)).unwrap(), p2.reshape((2, 9, v)).unwrap()], 1).unwrap().reshape((32, v)).unwrap();
        assert!(max_diff(&joined, &c) <= 1e-4, "L{layers}: state path {}", max_diff(&joined, &c));
    }
}

/// K2 applies y·sigmoid(h·Wd·Wu) to the memory's read-out: checked on mem_arch with Wo = I (Wu = 0 gives the gate
/// 1/2 exactly, so the read-out y is twice that output). The model has no Wg and the low-rank weights instead.
#[test]
fn sigmoid_gate_is_the_formula() {
    let ar = Arch { gate: OutGate::Sigmoid, gate_rank: 8, ..Arch::default() };
    let (cfg, vm, m) = build_l(ar, 64, 2, 16);
    {
        let data = vm.data().lock().unwrap();
        assert!(!data.keys().any(|k| k.ends_with(".wg")), "no d x d gate");
        assert_eq!(data["b0.hebb.wgd"].dims(), &[64, 8]);
        assert_eq!(data["b1.hebb.wgu"].dims(), &[8, 64]);
    }
    let made: usize = vm.data().lock().unwrap().values().map(|v| v.elem_count()).sum();
    assert_eq!(made, cfg.weight_counts().0);
    let base = Config::with_kind(Kind::D, V, 64, 2, 16).weight_counts().0;
    assert_eq!(base - made, 2 * (64 * 64 - 2 * 64 * 8), "Wg out, Wd and Wu in");
    set_var(&vm, "b0.hebb.wo", &Tensor::eye(64, DType::F32, &Device::Cpu).unwrap());
    let hb = m.blocks[0].hebb.as_ref().unwrap();
    assert!(m.ext.is_none(), "the gate alone keeps the default (fused) path");
    let (b, t) = (2, 16);
    let h = noise(&[b * t, 64], 3);
    let run = || m.mem_arch(&h, hb, None, b, t, &mut None, &mut None, false, None).unwrap();
    let out = run();
    let (wd, wu) = hb.wgl.as_ref().map(|(a, c)| (a.copy().unwrap(), c.copy().unwrap())).unwrap();
    set_var(&vm, "b0.hebb.wgu", &wu.zeros_like().unwrap());
    let y = run().affine(2.0, 0.0).unwrap();
    let want = y.mul(&sigmoid(&h.matmul(&wd).unwrap().matmul(&wu).unwrap()).unwrap()).unwrap();
    assert!(max_diff(&out, &want) <= 1e-6, "{}", max_diff(&out, &want));
}

/// Central finite differences (f64) against autograd: K1's mix (by the query and by a source) and K2's gate (by Wd,
/// Wu and the input).
#[test]
fn kimi_parts_gradients_match_finite_differences() {
    let f64t = |shape: &[usize], seed: u64, scale: f64| noise(shape, seed).to_dtype(DType::F64).unwrap().affine(scale, 0.0).unwrap();
    let (n, d, s) = (5usize, 8usize, 3usize);
    // K1: loss = sum(R ⊙ mix(sources, q))
    let srcs: Vec<Var> = (0..s).map(|i| Var::from_tensor(&f64t(&[n, d], 20 + i as u64, 1.0 + i as f64)).unwrap()).collect();
    let q = Var::from_tensor(&f64t(&[d], 30, 0.7)).unwrap();
    let r = f64t(&[n, d], 31, 1.0);
    let k1 = |srcs: &[Tensor], q: &Tensor| attn_res_mix(srcs, q).unwrap().mul(&r).unwrap().sum_all().unwrap();
    let ts: Vec<Tensor> = srcs.iter().map(|v| v.as_tensor().clone()).collect();
    let g = k1(&ts, q.as_tensor()).backward().unwrap();
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
    let f_k1 = || k1(&srcs.iter().map(|v| v.as_tensor().clone()).collect::<Vec<_>>(), q.as_tensor()).to_scalar::<f64>().unwrap();
    check("K1 query", &q, g.get(q.as_tensor()).unwrap(), &f_k1);
    check("K1 source 1", &srcs[1], g.get(srcs[1].as_tensor()).unwrap(), &f_k1);
    // K2: loss = sum(R ⊙ y ⊙ sigmoid(h·Wd·Wu))
    let (rank, dd) = (3usize, 8usize);
    let y = f64t(&[n, dd], 40, 1.0);
    let h = Var::from_tensor(&f64t(&[n, dd], 41, 1.0)).unwrap();
    let wd = Var::from_tensor(&f64t(&[dd, rank], 42, 0.8)).unwrap();
    let wu = Var::from_tensor(&f64t(&[rank, dd], 43, 0.8)).unwrap();
    let r2 = f64t(&[n, dd], 44, 1.0);
    let k2 = || y.mul(&sigmoid(&h.as_tensor().matmul(wd.as_tensor()).unwrap().matmul(wu.as_tensor()).unwrap()).unwrap()).unwrap().mul(&r2).unwrap().sum_all().unwrap();
    let g2 = k2().backward().unwrap();
    let f_k2 = || k2().to_scalar::<f64>().unwrap();
    check("K2 Wd", &wd, g2.get(wd.as_tensor()).unwrap(), &f_k2);
    check("K2 Wu", &wu, g2.get(wu.as_tensor()).unwrap(), &f_k2);
    check("K2 input", &h, g2.get(h.as_tensor()).unwrap(), &f_k2);
}

/// Each option on top of the 17p baseline's options (and both together) trains on the CPU: a few AdamW steps on one
/// batch lower the loss, and the weights made match the count.
#[test]
fn kimi_options_train_on_top_of_the_healthy_set() {
    for extra in ["--residual attn", "--gate sigmoid", "--residual attn --gate sigmoid --gate-rank 16"] {
        let ar = Arch::from_flags(&format!("{HEALTHY} {extra}")).unwrap();
        let (cfg, varmap, model) = build_l(ar, 128, 2, 16);
        let made: usize = varmap.data().lock().unwrap().values().map(|v| v.elem_count()).sum();
        assert_eq!(made, cfg.weight_counts().0, "{extra}: weight count");
        let (ids, tg) = batch(2, 16, 9);
        let mut params: Vec<crate::optim::Param> =
            varmap.data().lock().unwrap().iter().map(|(k, v)| crate::optim::Param { name: k.clone(), var: v.clone(), decay: Model::decays_with(k, true) }).collect();
        params.sort_by(|a, b| a.name.cmp(&b.name));
        assert!(params.iter().filter(|p| p.name.ends_with(".wgd") || p.name.ends_with(".wgu")).all(|p| p.decay), "{extra}: --decay-gates covers Wd, Wu");
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
        assert!(!Model::decays_with("b0.hebb.wgd", false), "no decay without --decay-gates, as .wg");
    }
}

/// A model with each option saved with its arch.txt loads back (load_auto) with the same logits, bit for bit; a model
/// of today's block saved without one still loads as before. With int8 on (the CPU talking path), the fused one-token
/// step steps aside (block options) and the state path gives forward's logits.
#[test]
fn kimi_checkpoint_round_trip_and_int8_talking() {
    let dir = scratch("kimi_ckpt");
    let cases = [("plain", String::new()), ("attn", "--residual attn".to_string()), ("sigmoid", "--gate sigmoid --gate-rank 8".to_string()), ("both", "--residual attn --gate sigmoid".to_string())];
    for (name, flags) in cases {
        let ar = Arch::from_flags(&flags).unwrap();
        let (_, vm, m) = build_l(ar, 64, 2, 32);
        if ar.attn_res {
            for i in 0..2 {
                set_var(&vm, &format!("b{i}.ares"), &noise(&[64], 50 + i as u64).affine(2.0, 0.0).unwrap());
            }
        }
        let (ids, _) = batch(2, 32, 13);
        let want = m.forward(&ids).unwrap();
        let path = dir.join(format!("{name}.safetensors"));
        vm.save(&path).unwrap();
        write_arch_beside(&path, &ar).unwrap();
        assert_eq!(crate::ckpt::arch_name(&path).exists(), !ar.is_default(), "{name}: arch.txt only with options");
        let (c2, _, m2) = load_auto(&path, Kind::D, 32, &Device::Cpu).unwrap();
        assert_eq!(c2.arch, ar, "{name}");
        let got = m2.forward(&ids).unwrap();
        assert_eq!(max_diff(&want, &got), 0.0, "{name}: logits after the round trip");
        if !ar.is_default() {
            m2.set_int8(true).unwrap();
            m2.set_fused(true);
            // the gate alone keeps the fused one-token step (fstep tests: bit for bit); attn residuals step aside
            assert_eq!(m2.fused_blocker().is_some(), ar.attn_res, "{name}: the fused one-token step");
            let l8 = m2.forward(&ids).unwrap();
            let mut st = m2.state(2).unwrap();
            let s8 = m2.forward_with_state(&ids, &mut st).unwrap();
            assert!(max_diff(&l8, &s8) <= 1e-4, "{name}: int8 state path {}", max_diff(&l8, &s8));
            assert!(l8.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|x| x.is_finite()), "{name}: int8 logits");
        }
    }
}

/// GPU (run with --ignored): K2 draws Wg's random numbers and drops them, so every other weight gets the init it gets
/// without the option (the same seed); the K1 + K2 model runs its forward and backward on CUDA with the CPU's loss.
#[test]
#[ignore]
fn kimi_gpu_same_init_and_forward() {
    let Ok(dev) = Device::new_cuda(0) else { return };
    let mk = |ar: Arch| {
        dev.set_seed(7).unwrap();
        let cfg = Config::with_kind(Kind::D, V, 64, 2, 32).with_arch(ar).unwrap();
        let vm = VarMap::new();
        let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev)).unwrap();
        (vm, m)
    };
    let (v0, _) = mk(Arch::default());
    let ar = Arch::from_flags("--residual attn --gate sigmoid").unwrap();
    let (v1, m1) = mk(ar);
    let (d0, d1) = (v0.data().lock().unwrap(), v1.data().lock().unwrap());
    let mut same = 0;
    for (k, a) in d0.iter() {
        match d1.get(k) {
            Some(b) => {
                assert_eq!(max_diff(&a.as_tensor().to_device(&Device::Cpu).unwrap(), &b.as_tensor().to_device(&Device::Cpu).unwrap()), 0.0, "{k}");
                same += 1;
            }
            None => assert!(k.ends_with(".wg"), "{k}"),
        }
    }
    assert_eq!(same, d0.len() - 2, "every tensor but the two Wg");
    // the same weights on the CPU
    let (_, vc, mc) = build_l(ar, 64, 2, 32);
    for (k, v) in d1.iter() {
        vc.data().lock().unwrap()[k].set(&v.as_tensor().to_device(&Device::Cpu).unwrap()).unwrap();
    }
    drop((d0, d1));
    let (ids, tg) = batch(2, 32, 17);
    let (lc, _) = mc.loss(&ids, &tg).unwrap();
    let (lg, _) = m1.loss(&ids.to_device(&dev).unwrap(), &tg.to_device(&dev).unwrap()).unwrap();
    let gg = lg.backward().unwrap();
    let (lc, lg) = (lc.to_scalar::<f32>().unwrap(), lg.to_scalar::<f32>().unwrap());
    assert!((lc - lg).abs() <= 1e-3 * lc.abs(), "loss CPU {lc} vs CUDA {lg}");
    let wd = v1.data().lock().unwrap()["b1.hebb.wgd"].clone();
    let gs = gg.get(wd.as_tensor()).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(gs > 0.0 && gs.is_finite(), "Wd gradient on CUDA {gs}");
    println!("kimi gpu: loss CPU {lc} CUDA {lg}");
}

/// GPU (run with --ignored --nocapture): plain D with --gate sigmoid keeps the fused step. loss_and_grads_a (the fused
/// memory kernels with the sigmoid gate; Wd·Wu formed as one gate matrix, dWd = dG·Wuᵀ, dWu = Wdᵀ·dG) against candle's
/// autograd at the bar of the fused D test (smoke.rs hand_written_da_step_matches_autograd): f32 loss to 1e-4 and every
/// gradient cos > 0.9999 with norm error < 1%; bf16 loss to 1e-2. Batch 4 and 96 take the chunk and the per-token form
/// (by shape, on a GPU of 24 to 47 SMs). The fused inference forward (set_inference) gives the plain loss to 1e-4.
#[test]
#[ignore]
fn kimi_gate_fused_step_matches_autograd() {
    let Ok(dev) = Device::new_cuda(0) else { return };
    let mut bad: Vec<String> = Vec::new();
    let mut forms = Vec::new();
    // today's silu gate first, under the same check, as the yardstick for the bf16 numbers
    for (gate, flags) in [("silu", ""), ("sigmoid", "--gate sigmoid --gate-rank 8")] {
    dev.set_seed(11).unwrap();
    let cfg = Config::with_kind(Kind::D, V, 64, 2, 64).with_arch(Arch::from_flags(flags).unwrap()).unwrap();
    let vm = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev)).unwrap();
    assert!(!m.arch_path(), "the gate alone keeps the fused step");
    for b in [4, 96] {
        forms.push(crate::fused::hebb_use_chunk(&dev, b, 2).unwrap());
        let (ids, tg) = batch(b, 64, 17 + b as u32);
        let (ids, tg) = (ids.to_device(&dev).unwrap(), tg.to_device(&dev).unwrap());
        let (l_ref, _) = m.loss(&ids, &tg).unwrap();
        let g_ref = l_ref.backward().unwrap();
        let x = l_ref.to_scalar::<f32>().unwrap();
        for bf16 in [false, true] {
            let (l_new, _, g_new) = m.loss_and_grads_a(&ids, &tg, &crate::model::Mm { bf16 }).unwrap();
            let y = l_new.to_scalar::<f32>().unwrap();
            let tol = if bf16 { 1e-2 } else { 1e-4 };
            if (x - y).abs() >= tol {
                bad.push(format!("{gate} b {b} bf16 {bf16}: loss {x} vs fused {y}"));
            }
            let mut worst = (1.0f32, 0.0f32, String::new());
            let mut coss = Vec::new();
            for (name, var) in vm.data().lock().unwrap().iter() {
                let r = g_ref.get(var.as_tensor()).unwrap_or_else(|| panic!("no autograd grad for {name}"));
                let g = g_new.get(var.as_tensor()).unwrap_or_else(|| panic!("b {b}: no fused grad for {name}"));
                assert_eq!(r.dims(), g.dims(), "{name}");
                let (rf, gf) = (r.flatten_all().unwrap(), g.to_dtype(DType::F32).unwrap().flatten_all().unwrap());
                let s = |t: &Tensor| t.sum_all().unwrap().to_scalar::<f32>().unwrap();
                let (nr, ng) = (s(&rf.sqr().unwrap()).sqrt(), s(&gf.sqr().unwrap()).sqrt());
                let cos = s(&rf.mul(&gf).unwrap()) / (nr * ng).max(1e-30);
                let rel = s(&rf.sub(&gf).unwrap().sqr().unwrap()).sqrt() / nr.max(1e-30);
                coss.push(cos);
                if cos < worst.0 {
                    worst = (cos, rel, name.clone());
                }
                if !bf16 && !(cos > 0.9999 && rel < 1e-2) {
                    bad.push(format!("{gate} b {b} f32 {name}: cos {cos} rel-norm-diff {rel}"));
                }
            }
            let mean = coss.iter().sum::<f32>() / coss.len() as f32;
            println!("{gate} b {b} bf16 {bf16}: loss {x:.6} vs fused {y:.6}; mean cos {mean:.5}, lowest {:.6} ({}, rel {:.2e})", worst.0, worst.2, worst.1);
        }
        m.set_inference(true);
        let (li, _) = m.loss(&ids, &tg).unwrap();
        m.set_inference(false);
        let li = li.to_scalar::<f32>().unwrap();
        if (x - li).abs() >= 1e-4 {
            bad.push(format!("{gate} b {b}: inference loss {li} vs {x}"));
        }
    }
    }
    println!("kernel forms (chunk?) {forms:?}");
    assert!(bad.is_empty(), "{bad:#?}");
}

/// GPU speed (run with --ignored --nocapture; seconds): tok/s of the fused D training step (loss_and_grads_a, forward
/// + backward, f32 and bf16) for plain D without and with --gate sigmoid (rank 32), at d128 and d384, L4, 8 x 256 tokens.
/// The two models take turns and each keeps its fastest of 11 rounds (3 steps each), so a GPU shared with another job
/// disturbs them as little as it can.
#[test]
#[ignore]
fn kimi_gate_fused_speed() {
    let Ok(dev) = Device::new_cuda(0) else { return };
    for d in [128, 384] {
        let mk = |flags: &str| {
            let cfg = Config::with_kind(Kind::D, 258, d, 4, 256).with_arch(Arch::from_flags(flags).unwrap()).unwrap();
            let vm = VarMap::new();
            let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev)).unwrap();
            (vm, m)
        };
        let ms = [("silu (today)", mk("")), ("sigmoid r32", mk("--gate sigmoid"))];
        let (ids, tg) = batch(8, 256, 5);
        let (ids, tg) = (ids.to_device(&dev).unwrap(), tg.to_device(&dev).unwrap());
        for bf16 in [false, true] {
            let mm = crate::model::Mm { bf16 };
            let mut secs = [f64::MAX; 2];
            for round in 0..12 {
                for k in 0..2 {
                    let j = (k + round) % 2; // ABBA: neither model always goes first
                    let m = &ms[j].1 .1;
                    let t0 = std::time::Instant::now();
                    for _ in 0..3 {
                        drop(m.loss_and_grads_a(&ids, &tg, &mm).unwrap());
                    }
                    dev.synchronize().unwrap();
                    if round > 0 {
                        secs[j] = secs[j].min(t0.elapsed().as_secs_f64()); // the fastest round: least disturbed by a shared GPU
                    }
                }
            }
            let toks = (8 * 256 * 3) as f64;
            let r: Vec<String> = ms.iter().zip(secs).map(|((n, _), s)| format!("{n} {:.0}k tok/s", toks / s / 1e3)).collect();
            println!("d{d} L4 bf16 {bf16}: {} (gate costs {:+.1}%)", r.join(", "), (secs[1] / secs[0] - 1.0) * 100.0);
        }
    }
}

/// CPU speed (run with --ignored --nocapture): ms per forward + backward at d128 L4, 4 x 128 tokens, for today's block
/// (its own path and forced onto the block-options path), each option alone, and the 17p baseline with each option.
#[test]
#[ignore]
fn kimi_cpu_speed() {
    let healthy = Arch::from_flags(HEALTHY).unwrap();
    let cases: Vec<(&str, Arch, bool)> = vec![
        ("default", Arch::default(), false),
        ("default, options path", Arch::default(), true),
        ("+gate sigmoid", Arch::from_flags("--gate sigmoid").unwrap(), false),
        ("+residual attn", Arch::from_flags("--residual attn").unwrap(), false),
        ("healthy (dense)", healthy, false),
        ("healthy, attn for dense", Arch::from_flags(&format!("{HEALTHY} --residual attn")).unwrap(), false),
        ("healthy +gate sigmoid", Arch::from_flags(&format!("{HEALTHY} --gate sigmoid")).unwrap(), false),
    ];
    for (name, arch, force) in cases {
        let (_, _, model) = build_l(arch, 128, 4, 128);
        model.set_force_arch(force);
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
        println!("kimi cpu speed {name:>26}: {ms:8.1} ms per step ({:.0} tok/s)", 512.0 / (ms / 1000.0));
    }
}

/// GPU (run with --ignored, AVA_GOLDEN=FILE): FNV-1a hashes of the bits of every fused D-memory kernel's outputs
/// (per-token and chunk forms, f32 and bf16, both backward launch shapes) on fixed inputs. With AVA_GOLDEN_WRITE=1 the
/// hashes are written to FILE; otherwise they must equal FILE's: today's silu kernels give the same bits after a change.
#[test]
#[ignore]
fn fused_memory_kernels_golden_bits() {
    let (Ok(dev), Ok(path)) = (Device::new_cuda(0), std::env::var("AVA_GOLDEN")) else { return };
    use crate::fused as f;
    let hash = |t: &Tensor| -> String {
        let v: Vec<f32> = t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for x in v {
            for b in x.to_bits().to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
            }
        }
        format!("{h:016x}")
    };
    let mut lines = Vec::new();
    for (b, nh, t) in [(2usize, 4usize, 40usize), (48, 4, 20)] {
        let d = nh * 32;
        let ld = 4 * d + 2 * nh;
        let p = noise(&[b * t, ld], 7).affine(1.5, 0.0).unwrap().to_device(&dev).unwrap();
        let cst = noise(&[3 * nh], 8).affine(0.5, 0.0).unwrap().add(&Tensor::cat(&[Tensor::zeros(2 * nh, DType::F32, &Device::Cpu).unwrap(), Tensor::full(3f32, nh, &Device::Cpu).unwrap()], 0).unwrap()).unwrap().to_device(&dev).unwrap();
        let d_o = noise(&[b * t, d], 9).to_device(&dev).unwrap();
        let eps = crate::model::NORM_EPS;
        let tag = format!("b{b}h{nh}t{t}");
        let (o, ck) = f::hebb_fwd_per_token(&p, &cst, b, t, nh, eps).unwrap();
        lines.push(format!("{tag} fwd_pt {} {}", hash(&o), hash(&ck)));
        let (o, ck2) = f::hebc_fwd_raw(&p, &cst, b, t, nh, eps).unwrap();
        lines.push(format!("{tag} fwd_chunk {} {}", hash(&o), hash(&ck2)));
        let (o, _) = f::hebb_fwd_raw_b(&p, &cst, b, t, nh, eps).unwrap();
        lines.push(format!("{tag} fwd_b {}", hash(&o)));
        for w1 in [false, true] {
            lines.push(format!("{tag} bwd_pt w1={w1} {}", hash(&f::hebb_bwd_per_token_k(&p, &cst, &ck, &d_o, b, t, nh, eps, w1).unwrap())));
            let (dp, dz) = f::hebb_bwd_per_token_b(&p, &cst, &ck, &d_o, b, t, nh, eps, w1).unwrap();
            lines.push(format!("{tag} bwd_pt_b w1={w1} {} {}", hash(&dp), hash(&dz)));
        }
        lines.push(format!("{tag} bwd_chunk {}", hash(&f::hebc_bwd_raw(&p, &cst, &ck2, &d_o, b, t, nh, eps).unwrap())));
        let (dp, dz) = f::hebb_bwd_raw_b(&p, &cst, &ck2, &d_o, b, t, nh, eps).unwrap();
        lines.push(format!("{tag} bwd_b {} {}", hash(&dp), hash(&dz)));
    }
    let text = lines.join("\n");
    if std::env::var("AVA_GOLDEN_WRITE").is_ok_and(|v| v == "1") {
        std::fs::write(&path, &text).unwrap();
        println!("golden: {} lines written", lines.len());
    } else {
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text, "the silu kernels' bits changed");
        println!("golden: {} lines equal", lines.len());
    }
}
