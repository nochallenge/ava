// TESTING / DESIGN ONLY — tests of src/memory.rs (v42 Wave 2C: R4, L3, O2, A9 and the in-block parts), CPU only.

use super::*;
use crate::arch::Arch;
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, max_diff, tokens};
use crate::testkit::fix_weights;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

const CPU: Device = Device::Cpu;

fn cfg_with(flags: &str, engram: Option<usize>) -> Config {
    let mut c = Config::with_kind(Kind::D, 258, 32, 2, 64);
    if let Some(l) = engram {
        c = c.with_engram(l).unwrap();
    }
    c.with_arch(Arch::from_flags(flags).unwrap()).unwrap()
}

/// A model of `cfg` whose shared weights are `base`'s (the new ones keep their init).
fn twin(cfg: Config, base: &VarMap) -> (VarMap, Model) {
    let vm = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
    fix_weights(&vm, &cfg, 99);
    let src = base.data().lock().unwrap();
    let dst = vm.data().lock().unwrap();
    for (k, v) in src.iter() {
        let w = dst.get(k).unwrap_or_else(|| panic!("{k} missing in the twin"));
        w.set(&v.as_tensor().copy().unwrap()).unwrap();
    }
    drop(dst);
    m.set_track_activity(false);
    (vm, m)
}

fn logits(m: &Model, toks: &[u32], b: usize, t: usize) -> Tensor {
    m.forward(&Tensor::from_vec(toks.to_vec(), (b, t), &CPU).unwrap()).unwrap()
}

#[test]
fn flags_round_trip_and_validate() {
    let line = "--eng-topk 8 --eng-temp --eng-ssmax --btsp 4 --mem-reset 32 --recall-loss 0.5 --pk-mem 64 --pk-layer 0 --pk-topk 4 --dendrites 2 --stp --delays 3 --wm 2";
    let a = Arch::from_flags(line).unwrap();
    assert_eq!(a.flags(), line);
    assert_eq!(Arch::from_flags(&a.flags()).unwrap(), a);
    assert!(a.signature().contains("eng_topk=8") && a.signature().contains(" stp"));
    assert!(Arch::default().mem.is_default() && Arch::default().flags().is_empty());
    let base = Config::with_kind(Kind::D, 258, 32, 2, 64);
    assert!(base.with_engram(1).unwrap().with_arch(a).is_ok());
    let bad = |f: &str, eng: Option<usize>| {
        let mut c = base;
        if let Some(l) = eng {
            c = c.with_engram(l).unwrap();
        }
        c.with_arch(Arch::from_flags(f).unwrap()).is_err()
    };
    assert!(bad("--eng-temp", None), "engram options need the engram");
    assert!(bad("--recall-loss 1", None));
    assert!(bad("--pk-mem 60", None), "not a square");
    assert!(bad("--pk-mem 16 --pk-topk 5", None), "top-k above sqrt(slots)");
    assert!(bad("--pk-layer 1", None));
    assert!(bad("--stp --feedback", None));
    assert!(bad("--mem-reset 8 --gates channel", None));
    assert!(bad("--delays 1", None));
    assert!(Config::with_kind(Kind::S, 258, 32, 2, 64).with_arch(Arch::from_flags("--stp").unwrap()).is_err(), "model D only");
    // the weight count matches the variables made
    let c = base.with_engram(1).unwrap().with_arch(a).unwrap();
    let vm = VarMap::new();
    let _m = Model::new(c, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
    let have: usize = vm.all_vars().iter().map(|v| v.elem_count()).sum();
    assert_eq!(have, c.weight_counts().0);
}

/// Each option starts as today's model: the same logits as the model without it (shared weights equal).
#[test]
fn every_option_starts_as_a_no_op() {
    let (base_vm, base) = build_awake(Config::with_kind(Kind::D, 258, 32, 2, 64).with_engram(1).unwrap(), 3);
    let (b, t) = (2, 48);
    let toks = tokens(b, t, 258, 5);
    let want = logits(&base, &toks, b, t);
    for flags in ["--eng-temp", "--eng-ssmax", "--btsp 4", "--eng-topk 64", "--mem-reset 8", "--recall-loss 0.5", "--pk-mem 16", "--dendrites 2", "--stp", "--delays 3", "--wm 2"] {
        let (_, m) = twin(cfg_with(flags, Some(1)), &base_vm);
        let d = max_diff(&logits(&m, &toks, b, t), &want);
        assert!(d < 1e-6, "{flags}: {d}");
    }
}

#[test]
fn every_option_learns() {
    for (flags, names) in [
        ("--eng-temp --eng-ssmax --btsp 3", vec!["b1.eng.tau", "b1.eng.ss", "b1.eng.btsp_g", "b1.eng.btsp_mix"]),
        ("--pk-mem 16 --pk-topk 2", vec!["b1.pk.values"]),
        ("--dendrites 2 --stp --delays 3 --wm 2", vec!["b0.parts.dend.a", "b0.parts.stp.mix", "b0.parts.delays.gain", "b0.parts.wm.wo"]),
    ] {
        let (vm, m) = build_awake(Config::with_kind(Kind::D, 258, 32, 2, 64).with_engram(1).unwrap(), 4);
        let (cvm, cm) = twin(cfg_with(flags, Some(1)), &vm);
        let toks = tokens(2, 33, 258, 6);
        let ids = Tensor::from_vec(toks.iter().enumerate().filter(|(i, _)| i % 33 != 32).map(|(_, &x)| x).collect::<Vec<u32>>(), (2, 32), &CPU).unwrap();
        let tg = Tensor::from_vec(toks.iter().enumerate().filter(|(i, _)| i % 33 != 0).map(|(_, &x)| x).collect::<Vec<u32>>(), (2, 32), &CPU).unwrap();
        let (_, _, g) = cm.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
        let data = cvm.data().lock().unwrap();
        for n in names {
            let gn = g.get(data[n].as_tensor()).unwrap_or_else(|| panic!("{flags}: no gradient for {n}"));
            let s = gn.abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(s > 0.0 && s.is_finite(), "{flags}: {n} gradient {s}");
        }
        let _ = m;
    }
}

#[test]
fn top_k_read_keeps_the_best_keys() {
    let (_, m) = build_awake(cfg_with("--eng-topk 1", Some(0)), 5);
    let e = m.blocks[0].eng.as_ref().unwrap();
    let ex = m.ext.as_ref().unwrap().mem.eng.as_ref();
    let (b, t, d) = (1, 12, 32);
    let h = Tensor::from_vec(crate::testkit::unit_noise(b * t * d, 9), (b * t, d), &CPU).unwrap();
    let o = MemOpts { eng_topk: 1, ..Default::default() };
    let got = read(&h, e, ex, &o, b, t, None, StoreUse::default(), false, None).unwrap().unwrap();
    // by hand: each query takes the value of its highest-scoring visible key, per head
    let nh = d / HEAD_SIZE;
    let heads = |w: &Tensor| h.matmul(w).unwrap().reshape((t, nh, HEAD_SIZE)).unwrap().transpose(0, 1).unwrap().contiguous().unwrap();
    let (q, k, v) = (heads(&e.wq), heads(&e.wk), heads(&e.wv));
    let sc: Vec<Vec<Vec<f32>>> = q.matmul(&k.t().unwrap()).unwrap().to_vec3().unwrap();
    let vv: Vec<Vec<Vec<f32>>> = v.to_vec3().unwrap();
    let mut y = vec![0f32; t * d];
    for hh in 0..nh {
        for i in 0..t {
            let j = (0..=i).max_by(|&a, &c| sc[hh][i][a].total_cmp(&sc[hh][i][c])).unwrap();
            for c in 0..HEAD_SIZE {
                y[i * d + hh * HEAD_SIZE + c] = vv[hh][j][c];
            }
        }
    }
    let want = Tensor::from_vec(y, (t, d), &CPU).unwrap().matmul(&e.wo).unwrap();
    assert!(max_diff(&got, &want) < 1e-5);
}

#[test]
fn btsp_keys_by_hand_and_in_pieces() {
    let (b, nh, t, hs) = (1, 2, 7, 3);
    let k = Tensor::from_vec(crate::testkit::unit_noise(b * nh * t * hs, 2), (b, nh, t, hs), &CPU).unwrap();
    let mix = Tensor::ones(nh, DType::F32, &CPU).unwrap();
    let (kw, tail) = btsp_keys(&k, None, &mix, 3).unwrap();
    // mix 1: the mean of the last (up to) 3 raw keys
    let kv: Vec<Vec<Vec<f32>>> = k.get(0).unwrap().to_vec3().unwrap();
    let got: Vec<Vec<Vec<f32>>> = kw.get(0).unwrap().to_vec3().unwrap();
    for hh in 0..nh {
        for j in 0..t {
            let lo = j.saturating_sub(2);
            for c in 0..hs {
                let m: f32 = (lo..=j).map(|x| kv[hh][x][c]).sum::<f32>() / (j - lo + 1) as f32;
                assert!((got[hh][j][c] - m).abs() < 1e-6);
            }
        }
    }
    assert_eq!(tail.unwrap().dim(2).unwrap(), 2);
    // in two pieces, with the tail carried, the same keys bit for bit
    let (a, ta) = btsp_keys(&k.narrow(2, 0, 3).unwrap(), None, &mix, 3).unwrap();
    let (c, _) = btsp_keys(&k.narrow(2, 3, 4).unwrap().contiguous().unwrap(), ta.as_ref(), &mix, 3).unwrap();
    assert_eq!(max_diff(&Tensor::cat(&[&a, &c], 2).unwrap(), &kw), 0.0);
}

#[test]
fn mem_reset_and_recall_loss_act_in_training_only() {
    let (vm, base) = build_awake(Config::with_kind(Kind::D, 258, 32, 2, 64).with_engram(1).unwrap(), 7);
    let toks = tokens(2, 33, 258, 8);
    let ids = Tensor::from_vec(toks.iter().enumerate().filter(|(i, _)| i % 33 != 32).map(|(_, &x)| x).collect::<Vec<u32>>(), (2, 32), &CPU).unwrap();
    let tg = Tensor::from_vec(toks.iter().enumerate().filter(|(i, _)| i % 33 != 0).map(|(_, &x)| x).collect::<Vec<u32>>(), (2, 32), &CPU).unwrap();
    let (l0, _, g0) = base.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
    let l0 = l0.to_scalar::<f32>().unwrap();
    // --mem-reset: scoring unchanged, training loss changed
    let (_, mr) = twin(cfg_with("--mem-reset 8", Some(1)), &vm);
    assert!(max_diff(&mr.forward(&ids).unwrap(), &base.forward(&ids).unwrap()) < 1e-6);
    let (l1, _, _) = mr.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
    assert!((l1.to_scalar::<f32>().unwrap() - l0).abs() > 1e-6);
    // the reset read is the delta rule restarted every 8 tokens
    let d = delta_reset_check();
    assert!(d < 1e-5, "{d}");
    // --recall-loss: the logged loss is the plain one; the engram's gradient changes, the rest of the gradient path too
    let (rvm, rl) = twin(cfg_with("--recall-loss 0.5", Some(1)), &vm);
    let (l2, _, g2) = rl.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
    assert!((l2.to_scalar::<f32>().unwrap() - l0).abs() < 1e-6);
    let wq0 = g0.get(vm.data().lock().unwrap()["b1.eng.wq"].as_tensor()).unwrap().clone();
    let wq2 = g2.get(rvm.data().lock().unwrap()["b1.eng.wq"].as_tensor()).unwrap().clone();
    assert!(max_diff(&wq0, &wq2) > 0.0);
    let (total, ce, _) = rl.loss_aux(&ids, &tg, 0.0).unwrap();
    assert!(total.to_scalar::<f32>().unwrap() > ce.to_scalar::<f32>().unwrap());
    assert!(rl.ext.as_ref().unwrap().mem.recall.borrow().is_none(), "the read is used once");
    // O2 gradient shares: every part present, summing to 1
    let params: Vec<(String, Tensor)> = rvm.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.as_tensor().clone())).collect();
    let sh = grad_shares(&params, &g2).unwrap();
    let total: f64 = sh.iter().map(|x| x.1).sum();
    assert!((total - 1.0).abs() < 1e-9);
    for p in ["embedding", "engram", "fast_memory", "mlp", "spiking"] {
        assert!(sh.iter().any(|(k, v)| k == p && *v > 0.0), "{p} in {sh:?}");
    }
    assert_eq!(part_of("b3.parts.stp.mix"), "parts");
    assert_eq!(part_of("final.norm"), "other");
}

/// delta_reset over 24 tokens with N = 8 equals three separate reads from F = 0.
fn delta_reset_check() -> f32 {
    let (b, nh, t, hs) = (1, 2, 24, 4);
    let n = |s| Tensor::from_vec(crate::testkit::unit_noise(b * nh * t * hs, s), (b, nh, t, hs), &CPU).unwrap();
    let unit = |x: Tensor| crate::model::l2_norm(&x).unwrap();
    let (q, k, v) = (unit(n(1)), unit(n(2)), n(3));
    let la = Tensor::full(-0.05f32, (b, nh, t), &CPU).unwrap();
    let be = Tensor::full(0.7f32, (b, nh, t), &CPU).unwrap();
    let got = delta_reset(&q, &k, &v, &la, &be, 8).unwrap();
    let mut parts = Vec::new();
    for c in 0..3 {
        let s = |x: &Tensor| x.narrow(2, c * 8, 8).unwrap().contiguous().unwrap();
        parts.push(crate::model::delta_rule(&s(&q), &s(&k), &s(&v), &s(&la), &s(&be)).unwrap());
    }
    max_diff(&got, &Tensor::cat(&parts, 2).unwrap())
}

#[test]
fn product_keys_and_sparse_finetuning() {
    let (vm, m) = build_awake(cfg_with("--pk-mem 64 --pk-topk 3", None), 9);
    let pk = m.ext.as_ref().unwrap().mem.pkm.as_ref().unwrap();
    let n = 10;
    let x = Tensor::from_vec(crate::testkit::unit_noise(n * 32, 3), (n, 32), &CPU).unwrap();
    *pk.counts.borrow_mut() = Some(vec![0; 64]);
    let y = pk.forward(&x).unwrap();
    assert_eq!(y.dims(), &[n, 32]);
    let counts = pk.counts.borrow_mut().take().unwrap();
    assert_eq!(counts.iter().sum::<u64>(), (n * 3) as u64);
    // by hand for row 0: the best 3 of all 64 slot scores (s1[a] + s2[b]) when both halves' top-3 hold them
    let h = crate::model::rmsnorm(&x, &pk.norm).unwrap();
    let q = h.matmul(&pk.wq).unwrap();
    let s1: Vec<Vec<f32>> = q.narrow(1, 0, PK_HALF).unwrap().matmul(&pk.k1.t().unwrap()).unwrap().to_vec2().unwrap();
    let s2: Vec<Vec<f32>> = q.narrow(1, PK_HALF, PK_HALF).unwrap().matmul(&pk.k2.t().unwrap()).unwrap().to_vec2().unwrap();
    let mut all: Vec<(f32, usize)> = (0..64).map(|s| (s1[0][s / 8] + s2[0][s % 8], s)).collect();
    all.sort_by(|a, b| b.0.total_cmp(&a.0));
    let top: Vec<(f32, usize)> = all[..3].to_vec();
    let z: f32 = top.iter().map(|x| (x.0 - top[0].0).exp()).sum();
    let vals: Vec<Vec<f32>> = pk.values.to_vec2().unwrap();
    let want: Vec<f32> = (0..32).map(|c| top.iter().map(|(s, i)| (s - top[0].0).exp() / z * vals[*i][c]).sum()).collect();
    let got: Vec<f32> = y.get(0).unwrap().to_vec1().unwrap();
    for c in 0..32 {
        assert!((got[c] - want[c]).abs() < 1e-5);
    }
    // the values' gradient touches only the slots read
    let ids = Tensor::from_vec(tokens(1, 16, 258, 4), (1, 16), &CPU).unwrap();
    let (_, _, g) = m.loss_and_grads_autograd(&ids, &ids, None, false).unwrap();
    let gv: Vec<Vec<f32>> = g.get(vm.data().lock().unwrap()["b1.pk.values"].as_tensor()).unwrap().to_vec2().unwrap();
    let touched = gv.iter().filter(|r| r.iter().any(|&x| x != 0.0)).count();
    assert!(touched > 0 && touched <= 16 * 3);
    // TF-IDF: a slot the new data reads often and the background rarely comes first
    let rows = sparse_ft_rows(&[10, 0, 5, 5], &[1000, 10, 1, 1000], 2);
    assert_eq!(rows, vec![0, 2]);
    assert_eq!(sparse_ft_rows(&[0, 0], &[1, 1], 3), Vec::<usize>::new());
    let mk = row_mask(&[2], 4, &CPU).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(mk, vec![0.0, 0.0, 1.0, 0.0]);
}
