// TESTING / DESIGN ONLY — tests of src/arch.rs (v42 Wave 2B block options), on the CPU only.

use super::*;
use crate::model::{Config, Kind, Model, TOK_EMB};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarBuilder, VarMap};

const V: usize = 64;

fn build(arch: Arch, d: usize, t: usize) -> (Config, VarMap, Model) {
    let cfg = Config::with_kind(Kind::D, V, d, 2, t).with_arch(arch).unwrap();
    let varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu)).unwrap();
    (cfg, varmap, model)
}

/// ids and targets [b, t] from a small deterministic stream.
fn batch(b: usize, t: usize, seed: u32) -> (Tensor, Tensor) {
    let n = b * (t + 1);
    let s: Vec<u32> = (0..n as u32).map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(97)) >> 7) % V as u32).collect();
    let (mut ids, mut tg) = (Vec::new(), Vec::new());
    for r in 0..b {
        let w = &s[r * (t + 1)..(r + 1) * (t + 1)];
        ids.extend_from_slice(&w[..t]);
        tg.extend_from_slice(&w[1..]);
    }
    (Tensor::from_vec(ids, (b, t), &Device::Cpu).unwrap(), Tensor::from_vec(tg, (b, t), &Device::Cpu).unwrap())
}

/// Copy every tensor of `from` into the same name in `to` (both models' shared weights become equal).
fn copy_shared(from: &VarMap, to: &VarMap) -> usize {
    let src = from.data().lock().unwrap();
    let dst = to.data().lock().unwrap();
    let mut n = 0;
    for (k, v) in src.iter() {
        if let Some(w) = dst.get(k) {
            if w.dims() == v.dims() {
                w.set(&v.as_tensor().copy().unwrap()).unwrap();
                n += 1;
            }
        }
    }
    n
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.sub(b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
}

fn set_var(varmap: &VarMap, name: &str, t: &Tensor) {
    varmap.data().lock().unwrap()[name].set(t).unwrap();
}

fn noise(shape: &[usize], seed: u64) -> Tensor {
    let n: usize = shape.iter().product();
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let v: Vec<f32> = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        })
        .collect();
    Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
}

/// Each option alone (with the sizes a tiny d64 model allows).
fn each_option() -> Vec<(&'static str, Arch)> {
    let z = Arch::default();
    vec![
        ("head_size=64", Arch { head_size: 64, ..z }),
        ("timescales=fixed", Arch { timescales: Timescales::Fixed, ..z }),
        ("gates=channel", Arch { gates: Gating::Channel, ..z }),
        ("value_residual", Arch { value_residual: true, ..z }),
        ("kwta=1/4", Arch { kwta_div: 4, ..z }),
        ("relu2", Arch { relu2: true, ..z }),
        ("mem_momentum=0.5", Arch { mem_momentum: 0.5, ..z }),
        ("feedback", Arch { feedback: true, ..z }),
        ("resid_fix", Arch { resid_fix: true, ..z }),
        ("dense", Arch { dense: true, ..z }),
        ("lif_logspaced", Arch { lif_logspaced: true, ..z }),
        ("lif_scale", Arch { lif_scale: true, ..z }),
        ("slope=2", Arch { slope: 2.0, ..z }),
        ("spikes=ternary", Arch { spikes: Spikes::Ternary, ..z }),
        ("spikes=int2", Arch { spikes: Spikes::Int2, ..z }),
        ("alif", Arch { alif: true, ..z }),
        ("homeostasis=0.1", Arch { homeostasis: 0.1, ..z }),
        ("top", Arch { top: true, ..z }),
        ("residual=attn", Arch { attn_res: true, ..z }),
        ("gate=sigmoid", Arch { gate: OutGate::Sigmoid, ..z }),
        ("gate=sigmoid rank 8", Arch { gate: OutGate::Sigmoid, gate_rank: 8, ..z }),
        ("gates=kda", Arch { gates: Gating::Kda, ..z }),
        ("bonus", Arch { bonus: true, ..z }),
        ("token_shift", Arch { token_shift: true, ..z }),
    ]
}

/// Every option at once (those that go together).
fn all_on() -> Arch {
    Arch {
        head_size: 64,
        timescales: Timescales::Fixed,
        gates: Gating::Channel,
        value_residual: true,
        kwta_div: 4,
        relu2: true,
        mem_momentum: 0.3,
        feedback: true,
        resid_fix: true,
        dense: true,
        lif_logspaced: true,
        lif_scale: true,
        slope: 3.0,
        spikes: Spikes::Int2,
        alif: true,
        lif_reset_grad: false,
        homeostasis: 0.1,
        top: true,
        mem: Default::default(),
        attn_res: false,
        gate: OutGate::Silu,
        gate_rank: OUT_GATE_RANK,
        bonus: false,
        token_shift: false,
        eng_gate: EngGate::Off,
        // v50 (src/v50.rs): off
        dale: 0.0,
        dale_proj: false,
        loops: 1,
        neuromod: 0,
        v51: Default::default(),
        v51b: Default::default(),
    }
}

/// The default is today's block: no extra parts, the old path, and the new forward (forced) gives the same logits and
/// gradients as the old one. --resid-fix alone also keeps the old path.
#[test]
fn default_arch_is_todays_block() {
    let (cfg, varmap, model) = build(Arch::default(), 64, 16);
    assert!(cfg.arch.is_default() && !cfg.arch.needs_path());
    assert!(model.ext.is_none() && !model.arch_path());
    assert_eq!(cfg.arch.signature(), "");
    let old = Config::with_kind(Kind::D, V, 64, 2, 16);
    assert_eq!(cfg, old, "with_kind builds the default arch");
    assert_eq!(cfg.weight_counts(), old.weight_counts());
    for name in varmap.data().lock().unwrap().keys() {
        let new_part = ["fb.", "top_emb", ".dense", ".vres", ".theta", ".arho", ".again", ".wa1", ".wa2", ".wb1", ".wb2", ".we", ".off"];
        assert!(!new_part.iter().any(|p| name.contains(p)), "today's tensors only: {name}");
    }
    let (ids, tg) = batch(2, 16, 1);
    let l_old = model.forward(&ids).unwrap();
    let (loss_old, _) = model.loss(&ids, &tg).unwrap();
    let g_old = loss_old.backward().unwrap();
    model.set_force_arch(true);
    let l_new = model.forward(&ids).unwrap();
    assert!(max_diff(&l_old, &l_new) < 1e-5, "forced new forward: {}", max_diff(&l_old, &l_new));
    let (loss_new, _) = model.loss(&ids, &tg).unwrap();
    let g_new = loss_new.backward().unwrap();
    for (name, var) in varmap.data().lock().unwrap().iter() {
        let (a, b) = (g_old.get(var.as_tensor()).unwrap(), g_new.get(var.as_tensor()).unwrap());
        let scale = a.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap().max(1e-6);
        assert!(max_diff(a, b) / scale < 1e-3, "{name}: {} of {scale}", max_diff(a, b));
    }
    let rf = Arch { resid_fix: true, ..Arch::default() };
    let (_, _, m2) = build(rf, 64, 16);
    assert!(m2.ext.is_none() && !m2.arch_path(), "--resid-fix keeps the fast path");
}

/// Every option alone and all together: the weight count matches the tensors made, the loss is finite, every
/// trained parameter gets a finite gradient, and a few AdamW steps on one batch lower the loss.
#[test]
fn every_option_counts_its_weights_and_learns() {
    let mut cases = each_option();
    cases.push(("all", all_on()));
    for (name, arch) in cases {
        let t = if arch.feedback { 80 } else { 16 };
        let (cfg, varmap, model) = build(arch, 64, t);
        let made: usize = varmap.data().lock().unwrap().values().map(|v| v.elem_count()).sum();
        assert_eq!(made, cfg.weight_counts().0, "{name}: weight count");
        assert_eq!(model.ext.is_some(), arch.needs_path(), "{name}");
        let (ids, tg) = batch(2, t, 3);
        let mut params: Vec<crate::optim::Param> = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| crate::optim::Param { name: k.clone(), var: v.clone(), decay: Model::decays_with(k, true) })
            .collect();
        params.sort_by(|a, b| a.name.cmp(&b.name));
        let mut opt = crate::optim::AdamW::new(params, 0.0).unwrap();
        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for step in 0..6 {
            let (loss, act) = if arch.top {
                let (total, ce, _, act) = model.loss_top(&ids, &tg, 0.5).unwrap();
                last = ce.to_scalar::<f32>().unwrap();
                (total, act)
            } else {
                let (l, act) = model.loss(&ids, &tg).unwrap();
                last = l.to_scalar::<f32>().unwrap();
                (l, act)
            };
            assert!(last.is_finite(), "{name}: loss {last}");
            assert_eq!(act.spike_rate.len(), 2, "{name}");
            let g = loss.backward().unwrap();
            if step == 0 {
                first = last;
                for p in opt.params() {
                    let theta = p.name.ends_with(".theta");
                    match g.get(p.var.as_tensor()) {
                        Some(x) => {
                            assert!(!theta, "{name}: the homeostatic threshold must get no gradient");
                            let s = x.abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
                            assert!(s.is_finite(), "{name}: {} gradient {s}", p.name);
                        }
                        None => assert!(theta, "{name}: {} has no gradient", p.name),
                    }
                }
            }
            opt.step(&g, 3e-3, 1.0).unwrap();
            model.homeostasis_step().unwrap();
        }
        assert!(last < first, "{name}: loss {first} -> {last}");
    }
}

/// The options whose new parts start at 0 start exactly as today's D (the same weights copied in): value residual,
/// dense residual, feedback (over two chunks), per-channel gates (alpha and b equal over a head's channels, e = k).
#[test]
fn zero_init_options_start_as_todays_d() {
    let z = Arch::default();
    for (name, arch, t) in [
        ("value_residual", Arch { value_residual: true, ..z }, 16),
        ("dense", Arch { dense: true, ..z }, 16),
        ("feedback", Arch { feedback: true, ..z }, 128),
        ("gates=channel", Arch { gates: Gating::Channel, ..z }, 16),
        ("momentum=0 head 32", Arch { timescales: Timescales::Index, ..z }, 16),
    ] {
        let (_, vm0, m0) = build(z, 64, t);
        let (_, vm1, m1) = build(arch, 64, t);
        let n = copy_shared(&vm0, &vm1);
        assert_eq!(n, vm0.data().lock().unwrap().len(), "{name}: every one of today's tensors is shared");
        let (ids, _) = batch(2, t, 5);
        let (a, b) = (m0.forward(&ids).unwrap(), m1.forward(&ids).unwrap());
        let diff = max_diff(&a, &b);
        let tol = if arch.feedback || arch.gates == Gating::Channel { 2e-4 } else { 1e-6 };
        assert!(diff <= tol, "{name}: {diff}");
    }
}

/// O1: the feedback reaches only later chunks (chunk 0 is unchanged by it), changes them once its projection is not
/// 0, and its projection gets a gradient even at 0.
#[test]
fn feedback_reaches_only_later_chunks() {
    let arch = Arch { feedback: true, ..Arch::default() };
    let (_, varmap, model) = build(arch, 64, 128);
    let (ids, tg) = batch(1, 128, 7);
    let (loss, _) = model.loss(&ids, &tg).unwrap();
    let g = loss.backward().unwrap();
    let fbv = varmap.data().lock().unwrap()["fb.wo"].clone();
    let gs = g.get(fbv.as_tensor()).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(gs > 0.0, "the feedback projection gets a gradient at 0");
    let before = model.forward(&ids).unwrap();
    set_var(&varmap, "fb.wo", &noise(&[64, 64], 3).affine(0.5, 0.0).unwrap());
    let after = model.forward(&ids).unwrap();
    let first = max_diff(&before.narrow(0, 0, 64).unwrap(), &after.narrow(0, 0, 64).unwrap());
    let second = max_diff(&before.narrow(0, 64, 64).unwrap(), &after.narrow(0, 64, 64).unwrap());
    assert_eq!(first, 0.0, "chunk 0 never sees feedback");
    assert!(second > 1e-3, "chunk 1 does: {second}");
}

/// P1: head sizes 64 and 128 at d128; the chunked delta rule (any head size, the CPU path) and the recurrence agree;
/// the fixed timescales are log-spaced between 2^-5 and 2^-8 at every width, and `index` is today's rule.
#[test]
fn head_sizes_and_timescales() {
    for hs in [64usize, 128] {
        let arch = Arch { head_size: hs, timescales: Timescales::Fixed, ..Arch::default() };
        let (cfg, _, model) = build(arch, 128, 32);
        assert_eq!(cfg.heads(), 128 / hs);
        let (ids, _) = batch(2, 32, 9);
        let l = model.forward(&ids).unwrap();
        assert_eq!(l.dims(), &[64, V]);
        assert!(l.to_dtype(DType::F32).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().is_finite());
    }
    assert_eq!(gate_gammas(12, Timescales::Index), crate::model::hebb_gammas(12));
    for nh in [1usize, 2, 4, 12] {
        let g = gate_gammas(nh, Timescales::Fixed);
        assert!((1.0 - g[0] - 2f64.powi(-5)).abs() < 1e-12);
        if nh > 1 {
            assert!((1.0 - g[nh - 1] - 2f64.powi(-8)).abs() < 1e-12, "{nh}: {g:?}");
            for w in g.windows(2) {
                assert!(w[1] > w[0]);
            }
        }
    }
    // the chunked form and the recurrence agree at head size 64
    let (b, nh, t, hs) = (2usize, 2usize, 64usize, 64usize);
    let unit = |x: Tensor| crate::model::l2_norm(&x).unwrap();
    let q = unit(noise(&[b, nh, t, hs], 1));
    let k = unit(noise(&[b, nh, t, hs], 2));
    let v = noise(&[b, nh, t, hs], 3);
    let la = log_sigmoid(&noise(&[b, nh, t], 4).affine(2.0, 3.0).unwrap()).unwrap();
    let beta = sigmoid(&noise(&[b, nh, t], 5)).unwrap().affine(2.0, 0.0).unwrap();
    let y1 = crate::model::delta_rule(&q, &k, &v, &la, &beta).unwrap();
    let (y2, _) = delta_rec(&q, &k, &k, &v, &la.unsqueeze(3).unwrap(), &beta.unsqueeze(3).unwrap(), 0.0, None).unwrap();
    assert!(max_diff(&y1, &y2) < 2e-4, "{}", max_diff(&y1, &y2));
    // and in two halves with the state carried (O1's chunks)
    let half = |x: &Tensor, i: usize| x.narrow(2, i * 32, 32).unwrap().contiguous().unwrap();
    let (ya, fa) = crate::model::delta_rule_from(&half(&q, 0), &half(&k, 0), &half(&v, 0), &half(&la, 0), &half(&beta, 0), None).unwrap();
    let (yb, _) = crate::model::delta_rule_from(&half(&q, 1), &half(&k, 1), &half(&v, 1), &half(&la, 1), &half(&beta, 1), Some(&fa)).unwrap();
    let y3 = Tensor::cat(&[&ya, &yb], 2).unwrap();
    assert!(max_diff(&y1, &y3) < 2e-4, "{}", max_diff(&y1, &y3));
}

/// P2 and P5 in the recurrence: per-channel rates that are equal over a head equal the per-head form; with momentum M
/// the first two steps are F1 = U1 and F2 = alpha2·F1 + M·U1 + U2 (by hand).
#[test]
fn channel_gates_and_momentum_recurrence() {
    let (b, nh, t, hs) = (1usize, 2usize, 6usize, 8usize);
    let unit = |x: Tensor| crate::model::l2_norm(&x).unwrap();
    let q = unit(noise(&[b, nh, t, hs], 11));
    let k = unit(noise(&[b, nh, t, hs], 12));
    let e = unit(noise(&[b, nh, t, hs], 13));
    let v = noise(&[b, nh, t, hs], 14);
    let la = log_sigmoid(&noise(&[b, nh, t, 1], 15).affine(1.0, 2.0).unwrap()).unwrap();
    let bw = sigmoid(&noise(&[b, nh, t, 1], 16)).unwrap().affine(2.0, 0.0).unwrap();
    let (y1, _) = delta_rec(&q, &k, &e, &v, &la, &bw, 0.0, None).unwrap();
    let rep = |x: &Tensor| x.broadcast_as((b, nh, t, hs)).unwrap().contiguous().unwrap();
    let (y2, _) = delta_rec(&q, &k, &e, &v, &rep(&la), &rep(&bw), 0.0, None).unwrap();
    assert!(max_diff(&y1, &y2) < 1e-6);
    // momentum by hand for head 0 (scalar gates)
    let m = 0.7;
    let (_, st1) = delta_rec(&q.narrow(2, 0, 1).unwrap(), &k.narrow(2, 0, 1).unwrap(), &e.narrow(2, 0, 1).unwrap(), &v.narrow(2, 0, 1).unwrap(), &la.narrow(2, 0, 1).unwrap(), &bw.narrow(2, 0, 1).unwrap(), m, None).unwrap();
    let (_, st2) = delta_rec(&q.narrow(2, 0, 2).unwrap(), &k.narrow(2, 0, 2).unwrap(), &e.narrow(2, 0, 2).unwrap(), &v.narrow(2, 0, 2).unwrap(), &la.narrow(2, 0, 2).unwrap(), &bw.narrow(2, 0, 2).unwrap(), m, None).unwrap();
    let get = |x: &Tensor, i: usize| x.narrow(2, i, 1).unwrap().reshape((nh, hs)).unwrap(); // [nh, hs]
    let gs = |x: &Tensor, i: usize| x.narrow(2, i, 1).unwrap().reshape(nh).unwrap().to_vec1::<f32>().unwrap();
    // U_i = b k v^T - alpha b e (e^T F_prev)
    let u = |i: usize, f: &Tensor| -> Tensor {
        let (ki, ei, vi) = (get(&k, i), get(&e, i), get(&v, i));
        let (a, bb) = (gs(&la, i), gs(&bw, i));
        let mut out = Vec::new();
        for h in 0..nh {
            let kh = ki.get(h).unwrap().reshape((hs, 1)).unwrap();
            let eh = ei.get(h).unwrap().reshape((hs, 1)).unwrap();
            let vh = vi.get(h).unwrap().reshape((1, hs)).unwrap();
            let fh = f.get(h).unwrap();
            let pred = eh.t().unwrap().matmul(&fh).unwrap();
            let w = kh.matmul(&vh).unwrap().affine(bb[h] as f64, 0.0).unwrap();
            let er = eh.matmul(&pred).unwrap().affine((a[h].exp() * bb[h]) as f64, 0.0).unwrap();
            out.push(w.sub(&er).unwrap());
        }
        Tensor::stack(&out, 0).unwrap()
    };
    let f0 = Tensor::zeros((nh, hs, hs), DType::F32, &Device::Cpu).unwrap();
    let u1 = u(0, &f0);
    assert!(max_diff(&st1.f, &u1) < 1e-6, "F1 = U1");
    let u2 = u(1, &u1);
    let a2 = gs(&la, 1);
    let a2 = Tensor::from_vec(a2.iter().map(|x| x.exp()).collect::<Vec<f32>>(), (nh, 1, 1), &Device::Cpu).unwrap();
    let f2 = u1.broadcast_mul(&a2).unwrap().add(&u1.affine(m, 0.0).unwrap()).unwrap().add(&u2).unwrap();
    assert!(max_diff(&st2.f, &f2) < 1e-5, "F2 = a2 F1 + M U1 + U2: {}", max_diff(&st2.f, &f2));
}

/// R1: --resid-fix counts 3 writes per D block (4 in the engram block); the default is today's 2 per block.
#[test]
fn resid_fix_init_std() {
    let c = Config::with_kind(Kind::D, V, 64, 8, 16);
    assert_eq!(c.resid_std(), crate::model::INIT_STD / 16f64.sqrt());
    let f = c.with_arch(Arch { resid_fix: true, ..Arch::default() }).unwrap();
    assert_eq!(f.resid_writes(), 24);
    assert_eq!(f.resid_std(), crate::model::INIT_STD / 24f64.sqrt());
    let s = Config::with_kind(Kind::S, V, 64, 8, 16).with_arch(Arch { resid_fix: true, ..Arch::default() }).unwrap();
    assert_eq!(s.resid_writes(), 16);
    let e = f.with_engram(3).unwrap();
    assert_eq!(e.resid_writes(), 25);
}

/// R3: the plain-op LIF with today's settings gives the fused scan's spikes exactly and its gradients up to rounding;
/// with another surrogate slope the two still agree; log-spaced leaks span 2..512 tokens and only a fresh variable
/// gets them.
#[test]
fn lif_reference_matches_the_fused_scan() {
    let (b, t, d) = (3usize, 40usize, 16usize);
    for slope in [crate::model::LIF_SLOPE as f32, 2.0] {
        let cur = candle_core::Var::from_tensor(&noise(&[b, t, d], 21).affine(1.2, 0.4).unwrap()).unwrap();
        let bp = candle_core::Var::from_tensor(&noise(&[d], 22).affine(1.0, 2.0).unwrap()).unwrap();
        let beta = sigmoid(bp.as_tensor()).unwrap();
        let s1 = crate::lif::lif_slope(cur.as_tensor(), &beta, slope).unwrap();
        let (s2, _) = lif_ref(cur.as_tensor(), &beta, None, None, Spikes::Binary, slope as f64, None).unwrap();
        assert_eq!(max_diff(&s1, &s2), 0.0, "the same spikes");
        let w = noise(&[b, t, d], 23);
        let g1 = s1.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        let g2 = s2.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        for x in [cur.as_tensor(), bp.as_tensor()] {
            let (a, c) = (g1.get(x).unwrap(), g2.get(x).unwrap());
            assert!(max_diff(a, c) < 1e-4, "slope {slope}: {}", max_diff(a, c));
        }
    }
    let p = logspaced_beta_params(64);
    let tau = |x: f32| 1.0 / (1.0 - sigmoid(&Tensor::new(&[x], &Device::Cpu).unwrap()).unwrap().to_vec1::<f32>().unwrap()[0]);
    assert!((tau(p[0]) - 2.0).abs() < 1e-3 && (tau(p[63]) - 512.0).abs() / 512.0 < 1e-3, "{} {}", tau(p[0]), tau(p[63]));
    let arch = Arch { lif_logspaced: true, ..Arch::default() };
    let (_, varmap, _) = build(arch, 64, 16);
    let got = varmap.data().lock().unwrap()["b1.mem.beta"].as_tensor().to_vec1::<f32>().unwrap();
    assert_eq!(got, p);
    // a variable that was already there (a loaded model) keeps its values
    let vm2 = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm2, DType::F32, &Device::Cpu);
    vb.get_with_hints(64, "b0.mem.beta", candle_nn::Init::Const(0.5)).unwrap();
    let cfg = Config::with_kind(Kind::D, V, 64, 2, 16).with_arch(arch).unwrap();
    Model::new(cfg, vb).unwrap();
    assert_eq!(vm2.data().lock().unwrap()["b0.mem.beta"].as_tensor().to_vec1::<f32>().unwrap(), vec![0.5; 64]);
    assert_eq!(vm2.data().lock().unwrap()["b1.mem.beta"].as_tensor().to_vec1::<f32>().unwrap(), p);
}

/// R3 --lif-scale: the input is scaled by (1 - beta), so the same model fires less (beta 0.9: a tenth of the input).
#[test]
fn lif_scale_quiets_the_input() {
    let (_, vm0, m0) = build(Arch { feedback: false, relu2: false, ..Arch { slope: 4.0, ..Arch::default() } }, 64, 32);
    let (_, vm1, m1) = build(Arch { lif_scale: true, ..Arch::default() }, 64, 32);
    copy_shared(&vm0, &vm1);
    let (ids, _) = batch(2, 32, 4);
    let (_, a0) = m0.forward_with_activity(&ids).unwrap();
    let (_, a1) = m1.forward_with_activity(&ids).unwrap();
    let r = |a: &crate::model::Activity| a.spike_rate[0].to_scalar::<f32>().unwrap();
    assert!(r(&a1) < r(&a0), "{} vs {}", r(&a1), r(&a0));
}

/// E1: signed integer spikes by hand (beta 0, theta 1): v_t = I_t - s_{t-1}; s = sign(v)·min(floor(|v|), L).
#[test]
fn integer_spikes_by_hand() {
    let beta = Tensor::new(&[0f32], &Device::Cpu).unwrap();
    let cur = Tensor::from_vec(vec![2.5f32, -1.2, 0.3, -3.0, 1.0, 0.99], (1, 6, 1), &Device::Cpu).unwrap();
    // ternary: v = 2.5 -> 1; v = -1.2 - 1 = -2.2 -> -1; v = 0.3 + 1 = 1.3 -> 1; v = -3 - 1 = -4 -> -1; v = 1 + 1 = 2 -> 1;
    // v = 0.99 - 1 = -0.01 -> 0
    let (s, _) = lif_ref(&cur, &beta, None, None, Spikes::Ternary, 4.0, None).unwrap();
    assert_eq!(s.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![1.0, -1.0, 1.0, -1.0, 1.0, 0.0]);
    // int2: 2.5 -> 2; -1.2 - 2 = -3.2 -> -2; 0.3 + 2 = 2.3 -> 2; -3 - 2 = -5 -> -2; 1 + 2 = 3 -> 2; 0.99 - 2 = -1.01 -> -1
    let (s, _) = lif_ref(&cur, &beta, None, None, Spikes::Int2, 4.0, None).unwrap();
    assert_eq!(s.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![2.0, -2.0, 2.0, -2.0, 2.0, -1.0]);
    // binary as before: 2.5 -> 1; -1.2 - 1 -> 0; 0.3 -> 0; -3 -> 0; 1 -> 1; 0.99 - 1 -> 0
    let (s, _) = lif_ref(&cur, &beta, None, None, Spikes::Binary, 4.0, None).unwrap();
    assert_eq!(s.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    // the surrogate has a bump at each edge: a gradient at v near -1 for signed spikes, none for binary
    let c = candle_core::Var::from_tensor(&Tensor::from_vec(vec![-1.05f32], (1, 1, 1), &Device::Cpu).unwrap()).unwrap();
    for (sp, want) in [(Spikes::Ternary, true), (Spikes::Binary, false)] {
        let (s, _) = lif_ref(c.as_tensor(), &beta, None, None, sp, 4.0, None).unwrap();
        let g = s.sum_all().unwrap().backward().unwrap().get(c.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert_eq!(g > 0.5, want, "{sp:?}: {g}");
    }
}

/// E4: with the adaptive threshold a neuron under constant drive fires less, and rho and the gain get gradients.
#[test]
fn alif_fires_less_under_constant_drive() {
    let (t, d) = (60usize, 4usize);
    let beta = Tensor::new(&[0.5f32; 4], &Device::Cpu).unwrap();
    let cur = Tensor::full(0.8f32, (1, t, d), &Device::Cpu).unwrap();
    let (s0, _) = lif_ref(&cur, &beta, None, None, Spikes::Binary, 4.0, None).unwrap();
    let rho = candle_core::Var::from_tensor(&Tensor::new(&[0.9f32; 4], &Device::Cpu).unwrap()).unwrap();
    let gain = candle_core::Var::from_tensor(&Tensor::new(&[1.0f32; 4], &Device::Cpu).unwrap()).unwrap();
    let (s1, st) = lif_ref(&cur, &beta, None, Some((rho.as_tensor(), gain.as_tensor())), Spikes::Binary, 4.0, None).unwrap();
    let n = |s: &Tensor| s.sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(n(&s1) < n(&s0) && n(&s1) > 0.0, "{} vs {}", n(&s1), n(&s0));
    assert!(st.a.is_some());
    let g = s1.sum_all().unwrap().backward().unwrap();
    for x in [rho.as_tensor(), gain.as_tensor()] {
        let s = g.get(x).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(s.is_finite() && s > 0.0, "{s}");
    }
}

/// E8: after a forward, each threshold moves by 0.01·(its neuron's rate - target): busy neurons up, quiet ones down.
#[test]
fn homeostasis_moves_thresholds_toward_the_rate() {
    let arch = Arch { homeostasis: 0.1, ..Arch::default() };
    let (_, varmap, model) = build(arch, 64, 32);
    let (ids, _) = batch(2, 32, 8);
    model.forward(&ids).unwrap();
    let rates: Vec<f32> = model.ext.as_ref().unwrap().rates.borrow()[0].as_ref().unwrap().to_vec1().unwrap();
    model.homeostasis_step().unwrap();
    let th: Vec<f32> = varmap.data().lock().unwrap()["b0.mem.theta"].as_tensor().to_vec1().unwrap();
    let (mut up, mut down) = (0, 0);
    for (r, t) in rates.iter().zip(&th) {
        let want = (1.0 + 0.01 * (r - 0.1)).clamp(HOMEO_RANGE.0 as f32, HOMEO_RANGE.1 as f32);
        assert!((t - want).abs() < 1e-6, "{t} vs {want}");
        if *r > 0.1 {
            up += 1;
        } else if *r < 0.1 {
            down += 1;
        }
    }
    assert!(up + down > 0);
}

/// P4: kWTA 1/4 keeps a quarter of the hidden units (at most), and relu2 squares the winners' activity.
#[test]
fn kwta_quarter_and_relu2() {
    let (cfg, _, model) = build(Arch { kwta_div: 4, ..Arch::default() }, 64, 16);
    assert_eq!(cfg.kwta_k(), 64);
    let (ids, _) = batch(2, 16, 2);
    let (_, act) = model.forward_with_activity(&ids).unwrap();
    for k in &act.kwta_post {
        let x = k.to_scalar::<f32>().unwrap();
        assert!(x > 0.0 && x <= 0.25 + 1e-6, "{x}");
    }
    // relu2: the same model with relu2 gives a different read-out (the squares), and the same winners
    let (_, vm0, m0) = build(Arch::default(), 64, 16);
    let (_, vm1, m1) = build(Arch { relu2: true, ..Arch::default() }, 64, 16);
    copy_shared(&vm0, &vm1);
    let (_, a0) = m0.forward_with_activity(&ids).unwrap();
    let (l1, _) = m1.forward_with_activity(&ids).unwrap();
    assert!(max_diff(&m0.forward(&ids).unwrap(), &l1) > 1e-6);
    assert!(a0.kwta_post[0].to_scalar::<f32>().unwrap() <= 0.125 + 1e-6);
}

/// R2: the order targets by hand, the loss's value at weight 0, and gradients into the order head.
#[test]
fn token_order_targets_and_loss() {
    // targets of one row: [5, 7, 5, IGNORE, 9]; window 3
    let ig = crate::IGNORE;
    let (idx, p) = top_targets(&[5, 7, 5, ig, 9], 1, 5, 64, 3);
    // position 0 sees 5 (dist 1, score 3), 7 (dist 2, score 2), 5 again (skipped)
    let e = |s: &[f64]| -> Vec<f32> {
        let z: f64 = s.iter().map(|x| x.exp()).sum();
        s.iter().map(|x| (x.exp() / z) as f32).collect()
    };
    assert_eq!(&idx[0..3], &[5, 7, 0]);
    let pe = e(&[3.0, 2.0]);
    assert!((p[0] - pe[0]).abs() < 1e-6 && (p[1] - pe[1]).abs() < 1e-6 && p[2] == 0.0);
    // position 1: 7 (3), 5 (2), IGNORE skipped
    assert_eq!(&idx[3..5], &[7, 5]);
    // position 3: IGNORE skipped, 9 at distance 2 (score 2): one token, weight 1
    assert_eq!((idx[9], p[9], p[10]), (9, 1.0, 0.0));
    // position 4: only 9
    assert_eq!((idx[12], p[12]), (9, 1.0));
    let arch = Arch { top: true, ..Arch::default() };
    let (_, varmap, model) = build(arch, 64, 16);
    let (ids, tg) = batch(2, 16, 6);
    let (total, ce, top, _) = model.loss_top(&ids, &tg, 0.0).unwrap();
    assert_eq!(total.to_scalar::<f32>().unwrap(), ce.to_scalar::<f32>().unwrap());
    let tv = top.to_scalar::<f32>().unwrap();
    assert!(tv > 0.0 && tv.is_finite());
    let (total, _, _, _) = model.loss_top(&ids, &tg, 0.5).unwrap();
    let g = total.backward().unwrap();
    let head = varmap.data().lock().unwrap()["top_emb"].clone();
    assert!(g.get(head.as_tensor()).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() > 0.0);
    // the main embedding gets the CE and the order loss's gradient through the trunk
    let tok = varmap.data().lock().unwrap()[TOK_EMB].clone();
    assert!(g.get(tok.as_tensor()).is_some());
}

/// The weighted training step (grow_gpu's) runs through autograd for a model with options: its loss is the
/// weighted mean of the per-target losses (and the signed sum with `raw`).
#[test]
fn weighted_step_through_autograd() {
    let (_, _, model) = build(Arch { value_residual: true, ..Arch::default() }, 64, 16);
    let (ids, tg) = batch(2, 16, 12);
    let w = noise(&[32], 30).abs().unwrap();
    let mm = crate::model::Mm { bf16: false };
    let (loss, _, g) = model.loss_and_grads_a_wr(&ids, &tg, Some(&w), false, &mm).unwrap();
    let logits = model.forward(&ids).unwrap();
    let rows = crate::lif::cross_entropy_rows(&logits, &tg.flatten_all().unwrap()).unwrap();
    let want = rows.mul(&w).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() / w.sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert!((loss.to_scalar::<f32>().unwrap() - want).abs() < 1e-5);
    assert!(!g.get_ids().collect::<Vec<_>>().is_empty());
    let (raw, _, _) = model.loss_and_grads_a_wr(&ids, &tg, Some(&w), true, &mm).unwrap();
    let want_raw = rows.mul(&w).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert!((raw.to_scalar::<f32>().unwrap() - want_raw).abs() < 1e-4);
    let (mean, _, _) = model.loss_and_grads_a_wr(&ids, &tg, None, false, &mm).unwrap();
    assert!((mean.to_scalar::<f32>().unwrap() - rows.mean_all().unwrap().to_scalar::<f32>().unwrap()).abs() < 1e-6);
}

/// v42 Wave 2C: token-by-token generation with the options goes through forward_with_state (a one-row State; the
/// match with forward is checked in src/state/tests.rs); the step trace still refuses them.
#[test]
fn stream_steps_block_options_through_state() {
    let (_, _, model) = build(Arch { dense: true, ..Arch::default() }, 64, 16);
    let mut s = model.stream().unwrap();
    assert!(crate::model::Model::stream_state(&s).is_some());
    assert_eq!(model.step(&mut s, 3).unwrap().dims(), &[V]);
    // v49: forward_trace knows the options now (src/synops/arch_tests.rs checks its logits), and so does the step trace
    // (src/model/fstep/trace_tests.rs checks it bit for bit)
    assert!(model.forward_trace(&batch(1, 16, 1).0).is_ok());
    assert!(model.step_trace(&mut model.stream().unwrap(), 3).unwrap().0.dims() == [V]);
    let (_, _, plain) = build(Arch::default(), 64, 16);
    assert!(plain.stream().is_ok());
}

/// The flags parse into the options, the signature names exactly the changed ones, and bad values and bad mixes are
/// refused.
#[test]
fn flags_parse_validate_and_sign() {
    let args: Vec<String> = "--head-size 64 --timescales fixed --gates channel --value-residual --kwta 0.25 --mlp-act relu2 --mem-momentum 0.5 \
                             --feedback --resid-fix --residual dense --lif-init logspaced --lif-scale --surrogate-slope 2 --spikes int2 --alif \
                             --homeostasis 0.05 --other"
        .split_whitespace()
        .map(String::from)
        .collect();
    let mut a = Arch::default();
    let mut i = 0;
    while i < args.len() {
        match a.take_flag(&args, i).unwrap() {
            Some(n) => i += n,
            None => {
                assert_eq!(args[i], "--other");
                i += 1;
            }
        }
    }
    let want = Arch {
        head_size: 64,
        timescales: Timescales::Fixed,
        gates: Gating::Channel,
        value_residual: true,
        kwta_div: 4,
        relu2: true,
        mem_momentum: 0.5,
        feedback: true,
        resid_fix: true,
        dense: true,
        lif_logspaced: true,
        lif_scale: true,
        slope: 2.0,
        spikes: Spikes::Int2,
        alif: true,
        lif_reset_grad: false,
        homeostasis: 0.05,
        top: false,
        mem: Default::default(),
        attn_res: false,
        gate: OutGate::Silu,
        gate_rank: OUT_GATE_RANK,
        bonus: false,
        token_shift: false,
        eng_gate: EngGate::Off,
        // v50 (src/v50.rs): off
        dale: 0.0,
        dale_proj: false,
        loops: 1,
        neuromod: 0,
        v51: Default::default(),
        v51b: Default::default(),
    };
    assert_eq!(a, want);
    assert_eq!(a.signature().split_whitespace().count(), 16);
    // the arch.txt line reads back to the same options; the default writes nothing
    assert_eq!(Arch::from_flags(&a.flags()).unwrap(), a);
    assert_eq!(Arch::from_flags(&all_on().flags()).unwrap(), Arch { top: false, ..all_on() });
    assert_eq!(Arch::default().flags(), "");
    assert_eq!(Arch::from_flags("").unwrap(), Arch::default());
    assert!(Arch::from_flags("--feedback --glance").is_err());
    let one = |s: &str| -> std::result::Result<Arch, String> {
        let v: Vec<String> = s.split_whitespace().map(String::from).collect();
        let mut a = Arch::default();
        a.take_flag(&v, 0)?;
        Ok(a)
    };
    assert_eq!(one("--kwta 1/8").unwrap(), Arch::default());
    assert!(one("--kwta 0.3").is_err());
    assert!(one("--spikes four").is_err());
    assert!(one("--head-size").is_err());
    let s = Config::with_kind(Kind::S, V, 64, 2, 16);
    assert!(s.with_arch(Arch { dense: true, ..Arch::default() }).is_err(), "the block options are for D");
    let d = Config::with_kind(Kind::D, V, 64, 2, 16);
    assert!(d.with_arch(Arch { head_size: 48, ..Arch::default() }).is_err());
    assert!(d.with_engram(1).unwrap().with_arch(Arch { feedback: true, ..Arch::default() }).is_err());
    assert!(d.with_arch(Arch { mem_momentum: 1.0, ..Arch::default() }).is_err());
}

/// Speed on the CPU (run with --ignored --nocapture): milliseconds per forward + backward at d128 L2, 4 x 128 tokens.
#[test]
#[ignore]
fn cpu_speed_of_each_option() {
    let mut cases = vec![("default", Arch::default())];
    cases.extend(each_option());
    for (name, arch) in cases {
        let cfg = Config::with_kind(Kind::D, V, 128, 2, 128).with_arch(arch).unwrap();
        let varmap = VarMap::new();
        let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu)).unwrap();
        model.set_track_activity(false);
        let (ids, tg) = batch(4, 128, 1);
        let run = || {
            let loss = if arch.top { model.loss_top(&ids, &tg, 0.5).unwrap().0 } else { model.loss(&ids, &tg).unwrap().0 };
            loss.backward().unwrap();
        };
        run();
        let t0 = std::time::Instant::now();
        for _ in 0..3 {
            run();
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;
        println!("cpu speed {name:>18}: {ms:8.1} ms per step ({:.0} tok/s)", 512.0 / (ms / 1000.0));
    }
    let _ = D::Minus1;
}

/// O3: `delta_chunk` gives what the token loop `delta_rec` gives: outputs, carried states and gradients, for
/// per-head and per-channel gates, with and without momentum, from zero and from a carried state, at lengths that are
/// not multiples of 16, and under forgetting strong enough that the products of rates underflow.
#[test]
fn chunked_memory_matches_the_token_loop() {
    let unit = |x: Tensor| crate::model::l2_norm(&x).unwrap();
    let rel = |a: &Tensor, b: &Tensor| -> f32 {
        let scale = b.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap().max(1.0);
        max_diff(a, b) / scale
    };
    let cases = [(false, 0.0, 32, 2.0), (true, 0.0, 32, 2.0), (false, 0.5, 40, 2.0), (true, 0.5, 24, 2.0), (true, 0.0, 32, -6.0), (true, 0.7, 16, -6.0), (false, 0.0, 128, 1.0)];
    for (n, &(chan, m, t, shift)) in cases.iter().enumerate() {
        let (b, nh, hs) = (2usize, 2usize, 8usize);
        let g = if chan { hs } else { 1 };
        let s = 100 * n as u64;
        let q = unit(noise(&[b, nh, t, hs], s + 1));
        let k = unit(noise(&[b, nh, t, hs], s + 2));
        let e = if chan { unit(noise(&[b, nh, t, hs], s + 3)) } else { k.clone() };
        let v = noise(&[b, nh, t, hs], s + 4);
        let la = log_sigmoid(&noise(&[b, nh, t, g], s + 5).affine(3.0, shift).unwrap()).unwrap();
        let bw = sigmoid(&noise(&[b, nh, t, g], s + 6)).unwrap().affine(2.0, 0.0).unwrap();
        let init = MemState {
            f: noise(&[b * nh, hs, hs], s + 7).affine(0.3, 0.0).unwrap(),
            s: if m > 0.0 { Some(noise(&[b * nh, hs, hs], s + 8).affine(0.3, 0.0).unwrap()) } else { None },
            prev: None,
        };
        for st in [None, Some(&init)] {
            let (y1, s1) = delta_rec(&q, &k, &e, &v, &la, &bw, m, st).unwrap();
            let (y2, s2) = delta_chunk(&q, &k, &e, &v, &la, &bw, m, st).unwrap();
            let tag = format!("case {n} (chan {chan}, M {m}, t {t}, shift {shift}, carried {})", st.is_some());
            assert!(y2.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|x| x.is_finite()), "{tag}: not finite");
            assert!(rel(&y2, &y1) < 1e-4, "{tag}: y {}", rel(&y2, &y1));
            assert!(rel(&s2.f, &s1.f) < 1e-4, "{tag}: F {}", rel(&s2.f, &s1.f));
            if let (Some(a), Some(c)) = (&s2.s, &s1.s) {
                assert!(rel(a, c) < 1e-4, "{tag}: S {}", rel(a, c));
            }
            assert_eq!(s1.s.is_some(), s2.s.is_some());
        }
        // gradients through both, for a fixed random read-out of y and of the final state
        let vars: Vec<candle_core::Var> = [&q, &k, &e, &v, &la, &bw].iter().map(|x| candle_core::Var::from_tensor(x).unwrap()).collect();
        let ry = noise(&[b, nh, t, hs], s + 9);
        let rf = noise(&[b * nh, hs, hs], s + 10);
        let loss = |f: fn(&Tensor, &Tensor, &Tensor, &Tensor, &Tensor, &Tensor, f64, Option<&MemState>) -> candle_core::Result<(Tensor, MemState)>| {
            let x: Vec<Tensor> = vars.iter().map(|v| v.as_tensor().clone()).collect();
            let (y, st) = f(&x[0], &x[1], &x[2], &x[3], &x[4], &x[5], m, Some(&init)).unwrap();
            y.mul(&ry).unwrap().sum_all().unwrap().add(&st.f.mul(&rf).unwrap().sum_all().unwrap()).unwrap()
        };
        let (g1, g2) = (loss(delta_rec).backward().unwrap(), loss(delta_chunk).backward().unwrap());
        for (i, var) in vars.iter().enumerate() {
            let (a, c) = (g2.get(var.as_tensor()).unwrap(), g1.get(var.as_tensor()).unwrap());
            assert!(rel(a, c) < 2e-4, "case {n}: grad {i} {}", rel(a, c));
        }
    }
}

/// O3 diagnostic (ignored; run by hand): the chunked memory against the loop at Stage 3's real shape (head size 32,
/// t 256) with init-like gates (rates near 1, write strength near 1, e = k), printing the errors.
#[test]
#[ignore]
fn o3_real_shape_errors() {
    let unit = |x: Tensor| crate::model::l2_norm(&x).unwrap();
    let (b, nh, t, hs) = (2usize, 4usize, 256usize, 32usize);
    let q = unit(noise(&[b, nh, t, hs], 1));
    let k = unit(noise(&[b, nh, t, hs], 2));
    let v = noise(&[b, nh, t, hs], 4);
    let off: Vec<f32> = gate_gammas(nh, Timescales::Index).into_iter().map(|g| logit(g) as f32).collect();
    let off = Tensor::from_vec(off, (1, nh, 1, 1), &Device::Cpu).unwrap();
    let la1 = log_sigmoid(&noise(&[b, nh, t, 1], 5).affine(0.5, 0.0).unwrap().broadcast_add(&off).unwrap()).unwrap();
    let bw1 = sigmoid(&noise(&[b, nh, t, 1], 6).affine(0.5, 0.0).unwrap()).unwrap().affine(2.0, 0.0).unwrap();
    let rep = |x: &Tensor| x.broadcast_as((b, nh, t, hs)).unwrap().contiguous().unwrap();
    let rel = |a: &Tensor, c: &Tensor| max_diff(a, c) / c.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
    for (name, la, bw) in [("per-head", la1.clone(), bw1.clone()), ("per-channel", rep(&la1), rep(&bw1))] {
        let (y1, s1) = delta_rec(&q, &k, &k, &v, &la, &bw, 0.0, None).unwrap();
        let (y2, s2) = delta_chunk(&q, &k, &k, &v, &la, &bw, 0.0, None).unwrap();
        let y3 = crate::model::delta_rule(&q, &k, &v, &la1.squeeze(3).unwrap(), &bw1.squeeze(3).unwrap()).unwrap();
        // the error at each position (max over the rest), to see whether it grows along the window
        let pos: Vec<f32> = (0..t).step_by(32).map(|i| max_diff(&y2.narrow(2, i, 1).unwrap(), &y1.narrow(2, i, 1).unwrap())).collect();
        println!("{name}: chunk vs loop y {:.2e} F {:.2e}; scalar chunk vs loop y {:.2e}; by position {pos:?}", rel(&y2, &y1), rel(&s2.f, &s1.f), rel(&y3, &y1));
    }
}

/// O3 diagnostic on the GPU (ignored; run by hand): the same inputs on CUDA, chunk and loop, against the CPU loop.
#[test]
#[ignore]
fn o3_gpu_errors() {
    let Ok(dev) = Device::new_cuda(0) else { return };
    let unit = |x: Tensor| crate::model::l2_norm(&x).unwrap();
    let (b, nh, t, hs) = (2usize, 4usize, 256usize, 32usize);
    let q = unit(noise(&[b, nh, t, hs], 1));
    let k = unit(noise(&[b, nh, t, hs], 2));
    let e = unit(noise(&[b, nh, t, hs], 3));
    let v = noise(&[b, nh, t, hs], 4);
    let la = log_sigmoid(&noise(&[b, nh, t, hs], 5).affine(2.0, 3.0).unwrap()).unwrap();
    let bw = sigmoid(&noise(&[b, nh, t, hs], 6)).unwrap().affine(2.0, 0.0).unwrap();
    let g = |x: &Tensor| x.to_device(&dev).unwrap();
    let c = |x: &Tensor| x.to_device(&Device::Cpu).unwrap();
    let rel = |a: &Tensor, r: &Tensor| max_diff(a, r) / r.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
    for m in [0.0, 0.5] {
        let (y0, _) = delta_rec(&q, &k, &e, &v, &la, &bw, m, None).unwrap();
        let (y1, _) = delta_rec(&g(&q), &g(&k), &g(&e), &g(&v), &g(&la), &g(&bw), m, None).unwrap();
        let (y2, _) = delta_chunk(&g(&q), &g(&k), &g(&e), &g(&v), &g(&la), &g(&bw), m, None).unwrap();
        let (y3, _) = delta_chunk(&q, &k, &e, &v, &la, &bw, m, None).unwrap();
        println!("M {m}: gpu loop {:.2e}, gpu chunk {:.2e}, cpu chunk {:.2e} (each vs the cpu loop)", rel(&c(&y1), &y0), rel(&c(&y2), &y0), rel(&y3, &y0));
    }
    // the pieces: cumsum and a narrowed-batch matmul
    let x = noise(&[3, 5, 16, 4], 7);
    println!("cumsum gpu vs cpu {:.2e}", max_diff(&c(&g(&x).cumsum(2).unwrap()), &x.cumsum(2).unwrap()));
    let a = noise(&[6, 5, 16, 8], 8);
    let bm = noise(&[6, 8, 8], 9);
    let at = |x: &Tensor| x.narrow(1, 2, 1).unwrap().squeeze(1).unwrap();
    println!("narrowed matmul gpu vs cpu {:.2e}", max_diff(&c(&at(&g(&a)).matmul(&g(&bm)).unwrap()), &at(&a).matmul(&bm).unwrap()));
}

/// Ava 0.4 with every part: the 13 Wave 5 parts and the 5 Stage 3 winners switch on together, build, give a finite
/// loss and finite gradients, and train a few steps without blowing up (the loss falls). Ignored; run by hand.
#[test]
#[ignore]
fn all_parts_together_build_and_train() {
    let flags = "--dendrites 4 --stp --delays 4 --wm 8 --resonate 64 --modulate --tem 16 --halting 3 --cerebellum 512 \
                 --jepa 0.1 --diffusion 0.1 --critic 0.1 --confidence 0.1";
    let winners = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense \
                   --resid-fix --lif-init logspaced --lif-scale";
    for (name, s) in [("13 parts", flags.to_string()), ("13 parts + 5 winners", format!("{flags} {winners}"))] {
        let args: Vec<String> = s.split_whitespace().map(String::from).collect();
        let mut a = Arch::default();
        let mut i = 0;
        while i < args.len() {
            i += a.take_flag(&args, i).unwrap_or_else(|e| panic!("{name}: {e}")).unwrap_or_else(|| panic!("{name}: {} not taken", args[i]));
        }
        let t = 64;
        let cfg = Config::with_kind(Kind::D, V, 128, 2, t).with_arch(a).unwrap_or_else(|e| panic!("{name}: {e}"));
        let varmap = VarMap::new();
        let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu)).unwrap();
        let mut opt = candle_nn::AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW { lr: 3e-3, ..Default::default() }).unwrap();
        use candle_nn::Optimizer;
        let (ids, tg) = batch(4, t, 3);
        let mut losses = Vec::new();
        for _ in 0..12 {
            let (l, _) = model.loss(&ids, &tg).unwrap();
            let v = l.to_dtype(DType::F32).unwrap().to_scalar::<f32>().unwrap();
            assert!(v.is_finite(), "{name}: loss {v}");
            losses.push(v);
            let g = l.backward().unwrap();
            for var in varmap.all_vars() {
                if let Some(gr) = g.get(var.as_tensor()) {
                    assert!(gr.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().is_finite(), "{name}: a gradient is not finite");
                }
            }
            opt.step(&g).unwrap();
        }
        println!("{name}: {} weights; loss {:.4} -> {:.4}", varmap.all_vars().iter().map(|v| v.elem_count()).sum::<usize>(), losses[0], losses[11]);
        assert!(losses[11] < losses[0], "{name}: the loss did not fall");
    }
}

// v44: the two ideas from Kimi (src/arch.rs K1, K2)
mod kimi;

// v45: four ideas from RWKV-7 and Qwen3-Next (src/arch.rs W1-W4)
mod rwkv;
