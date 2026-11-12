// TESTING / DESIGN ONLY — tests of src/eprop.rs (runs/v51/prereg_v51.md Part B, E), CPU only, on tiny models with Ava
// 0.4's block recipe (int2 spikes, ALIF, dense residual, channel gates, momentum), spikes on.
//
// (1) Where the theory says e-prop is exact: the plastic block is the last one and its fast-memory read is silent (its
//     Wo = 0), so the only path back in time is each unit's own membrane. Then the e-prop changes summed over a text
//     (lr tiny, so the forward does not move) equal backprop through time over the whole text (autograd on the full
//     forward, the reference path) for all four plastic matmuls, W_in through its eligibility traces.
// (2) The same summed changes with the memory on and two plastic blocks: e-prop drops the paths back in time through
//     the memory and the other block's membranes; how close it stays (cosine, relative error) is printed.
// (3) The online-backprop rule over a whole text in one forward equals autograd's gradient (the probes' machinery).
// (4) A session's per-row changes act as each row's own weights, and a row's reset clears only that row.

use super::*;
use crate::arch::Arch;
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, max_diff, tokens};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use std::collections::HashMap;

const CPU: Device = Device::Cpu;
const RECIPE: &str = "--head-size 32 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense \
                      --resid-fix --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif \
                      --homeostasis 0.1";

fn set(vm: &VarMap, name: &str, t: &Tensor) {
    vm.data().lock().unwrap()[name].set(t).unwrap();
}

fn cfg() -> Config {
    Config::with_kind(Kind::D, 258, 64, 2, 64).with_arch(Arch::from_flags(RECIPE).unwrap()).unwrap()
}

/// e-prop over `toks` (one row), lr tiny: the summed changes per (block, site) = -sum of the steps / lr.
fn eprop_sums(m: &Model, toks: &[u32], blocks: &[usize]) -> Vec<((usize, Site), Tensor)> {
    let lr = 1e-20;
    m.plastic_begin(1, blocks, &Site::ALL).unwrap();
    m.plastic_probes(true);
    let mut st = m.state(1).unwrap();
    for j in 0..toks.len() - 1 {
        let l = m.forward_with_state(&Tensor::new(&[[toks[j]]], &CPU).unwrap(), &mut st).unwrap();
        let ce = candle_nn::loss::cross_entropy(&l, &Tensor::new(&[toks[j + 1]], &CPU).unwrap()).unwrap();
        let g = ce.backward().unwrap();
        m.plastic_learn(&g, lr, Rule::EProp).unwrap();
        st.detach_all();
    }
    let mut out = Vec::new();
    for &i in blocks {
        for s in Site::ALL {
            out.push(((i, s), m.plastic_delta(i, s, 0).unwrap().unwrap().affine(-1.0 / lr, 0.0).unwrap()));
        }
    }
    m.plastic_end();
    out
}

/// Backprop through time over the whole text: autograd's gradient of the summed cross-entropy (the full forward).
fn bptt(vm: &VarMap, m: &Model, toks: &[u32], i: usize, s: Site) -> Tensor {
    let t = toks.len() - 1;
    let ids = Tensor::from_vec(toks[..t].to_vec(), (1, t), &CPU).unwrap();
    let tg = Tensor::from_vec(toks[1..].to_vec(), t, &CPU).unwrap();
    let l = m.forward(&ids).unwrap();
    let ce = candle_nn::loss::cross_entropy(&l, &tg).unwrap().affine(t as f64, 0.0).unwrap();
    let g = ce.backward().unwrap();
    g.get(vm.data().lock().unwrap()[&s.var_name(i)].as_tensor()).unwrap().copy().unwrap()
}

fn cos_rel(a: &Tensor, b: &Tensor) -> (f64, f64) {
    let dot = a.mul(b).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() as f64;
    let na = a.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt() as f64;
    let nb = b.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt() as f64;
    let diff = a.sub(b).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt() as f64;
    (dot / (na * nb).max(1e-30), diff / nb.max(1e-30))
}

#[test]
fn eprop_equals_bptt_where_the_theory_says_so() {
    let c = cfg();
    let (vm, m) = build_awake(c, 81);
    // the last block's fast-memory read silent: no path back in time but the membranes
    set(&vm, "b1.hebb.wo", &Tensor::zeros((64, 64), DType::F32, &CPU).unwrap());
    let toks = tokens(1, 33, 258, 17);
    let sums = eprop_sums(&m, &toks, &[1]);
    for ((i, s), g) in &sums {
        let want = bptt(&vm, &m, &toks, *i, *s);
        let (cos, rel) = cos_rel(g, &want);
        eprintln!("v51 e-prop vs BPTT (exact case: last block, memory read silent) b{i} {s:?}: cosine {cos:.7}, relative error {rel:.2e}, |grad| {:.3e}", want.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt());
        assert!(rel < 1e-3, "b{i} {s:?}: e-prop differs from BPTT by {rel}");
    }
    // the trace matters: W_in's present-input-only change (no trace) is further from BPTT
    let (vm2, m2) = build_awake(c, 81);
    set(&vm2, "b1.hebb.wo", &Tensor::zeros((64, 64), DType::F32, &CPU).unwrap());
    let one_step = {
        let lr = 1e-20;
        m2.plastic_begin(1, &[1], &[Site::Win]).unwrap();
        m2.plastic_probes(true);
        let mut st = m2.state(1).unwrap();
        for j in 0..toks.len() - 1 {
            let l = m2.forward_with_state(&Tensor::new(&[[toks[j]]], &CPU).unwrap(), &mut st).unwrap();
            let ce = candle_nn::loss::cross_entropy(&l, &Tensor::new(&[toks[j + 1]], &CPU).unwrap()).unwrap();
            m2.plastic_learn(&ce.backward().unwrap(), lr, Rule::Backprop).unwrap();
            st.detach_all();
        }
        let d = m2.plastic_delta(1, Site::Win, 0).unwrap().unwrap().affine(-1.0 / lr, 0.0).unwrap();
        m2.plastic_end();
        d
    };
    let want = bptt(&vm2, &m2, &toks, 1, Site::Win);
    let (cos1, rel1) = cos_rel(&one_step, &want);
    eprintln!("v51 W_in without the trace (one-step backprop) vs BPTT: cosine {cos1:.5}, relative error {rel1:.2e}");
    assert!(rel1 > 1e-2, "the membrane trace should matter here ({rel1})");
}

#[test]
fn eprop_with_memory_on_is_an_approximation() {
    let c = cfg();
    let (vm, m) = build_awake(c, 82);
    let toks = tokens(1, 33, 258, 18);
    let sums = eprop_sums(&m, &toks, &[0, 1]);
    for ((i, s), g) in &sums {
        let want = bptt(&vm, &m, &toks, *i, *s);
        let (cos, rel) = cos_rel(g, &want);
        eprintln!("v51 e-prop vs BPTT (memory on, blocks 0 and 1 plastic) b{i} {s:?}: cosine {cos:.4}, relative error {rel:.3}");
        assert!(cos.is_finite() && rel.is_finite());
    }
}

#[test]
fn online_backprop_over_a_text_equals_autograd() {
    let c = cfg();
    let (vm, m) = build_awake(c, 83);
    let toks = tokens(1, 25, 258, 19);
    let t = toks.len() - 1;
    let lr = 1e-20;
    m.plastic_begin(1, &[0, 1], &Site::ALL).unwrap();
    m.plastic_probes(true);
    let mut st = m.state(1).unwrap();
    let l = m.forward_with_state(&Tensor::from_vec(toks[..t].to_vec(), (1, t), &CPU).unwrap(), &mut st).unwrap();
    let ce = candle_nn::loss::cross_entropy(&l, &Tensor::from_vec(toks[1..].to_vec(), t, &CPU).unwrap()).unwrap().affine(t as f64, 0.0).unwrap();
    m.plastic_learn(&ce.backward().unwrap(), lr, Rule::Backprop).unwrap();
    let got: Vec<((usize, Site), Tensor)> = [0, 1].iter().flat_map(|&i| Site::ALL.map(|s| ((i, s), m.plastic_delta(i, s, 0).unwrap().unwrap().affine(-1.0 / lr, 0.0).unwrap()))).collect();
    m.plastic_end();
    for ((i, s), g) in got {
        let want = bptt(&vm, &m, &toks, i, s);
        let (cos, rel) = cos_rel(&g, &want);
        eprintln!("v51 online backprop over the text vs autograd b{i} {s:?}: cosine {cos:.7}, relative error {rel:.2e}");
        assert!(rel < 1e-3, "b{i} {s:?}: {rel}");
    }
}

#[test]
fn per_row_changes_act_as_own_weights_and_reset_one_row() {
    let c = cfg();
    let (vm, m) = build_awake(c, 84);
    let toks = tokens(2, 12, 258, 20);
    m.plastic_begin(2, &[1], &[Site::W1, Site::Win]).unwrap();
    {
        let e = m.v51b().unwrap();
        let mut g = e.plastic.borrow_mut();
        let p = g.as_mut().unwrap();
        for (k, v) in p.delta.iter_mut() {
            let n = v.elem_count();
            let noise: Vec<f32> = crate::testkit::unit_noise(n, 5 + k.1 as u64).iter().map(|z| z * 0.05).collect();
            *v = Tensor::from_vec(noise, v.dims(), &CPU).unwrap();
        }
    }
    let ids = Tensor::from_vec(toks.clone(), (2, 12), &CPU).unwrap();
    let mut st = m.state(2).unwrap();
    let both = m.forward_with_state(&ids, &mut st).unwrap().reshape((2, 12, ())).unwrap();
    // row r alone: the weights W + dW_r in a plain model
    for r in 0..2 {
        let (vmr, mr) = build_awake(c, 84);
        for s in [Site::W1, Site::Win] {
            let n = s.var_name(1);
            let w = vm.data().lock().unwrap()[&n].as_tensor().add(&m.plastic_delta(1, s, r).unwrap().unwrap()).unwrap();
            set(&vmr, &n, &w);
        }
        let mut sr = mr.state(1).unwrap();
        let one = mr.forward_with_state(&Tensor::from_vec(toks[r * 12..(r + 1) * 12].to_vec(), (1, 12), &CPU).unwrap(), &mut sr).unwrap();
        let d = max_diff(&one, &both.get(r).unwrap());
        eprintln!("v51 plastic row {r} vs a model with its own weights: max |logit difference| {d:.2e}");
        assert!(d < 1e-4, "row {r}: {d}");
    }
    m.plastic_reset_row(0).unwrap();
    let z0 = m.plastic_delta(1, Site::W1, 0).unwrap().unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    let z1 = m.plastic_delta(1, Site::W1, 1).unwrap().unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    assert_eq!(z0, 0.0);
    assert!(z1 > 0.0);
    m.plastic_end();
}

#[test]
fn per_row_rates_scale_each_rows_change() {
    let c = cfg();
    let (_vm, m) = build_awake(c, 85);
    let toks = tokens(1, 9, 258, 21);
    m.plastic_begin(2, &[0, 1], &Site::ALL).unwrap();
    m.plastic_probes(true);
    let mut st = m.state(2).unwrap();
    for j in 0..toks.len() - 1 {
        let ids = Tensor::new(&[[toks[j]], [toks[j]]], &CPU).unwrap();
        let l = m.forward_with_state(&ids, &mut st).unwrap();
        let ce = candle_nn::loss::cross_entropy(&l, &Tensor::new(&[toks[j + 1], toks[j + 1]], &CPU).unwrap()).unwrap().affine(2.0, 0.0).unwrap();
        m.plastic_learn_rows(&ce.backward().unwrap(), &[1e-20, 3e-20], Rule::EProp).unwrap();
        st.detach_all();
    }
    for i in [0, 1] {
        for s in Site::ALL {
            let a = m.plastic_delta(i, s, 0).unwrap().unwrap().affine(3e20, 0.0).unwrap(); // rescaled: (1e-20)^2 underflows f32
            let b = m.plastic_delta(i, s, 1).unwrap().unwrap().affine(1e20, 0.0).unwrap();
            let (cos, rel) = cos_rel(&b, &a);
            assert!(rel < 1e-5 && cos > 0.99999, "b{i} {s:?}: row 1 (3x rate) vs 3 x row 0: cosine {cos}, relative error {rel}");
        }
    }
    m.plastic_end();
}

#[test]
fn learning_signals_flow_back_through_the_rows_changes() {
    // block 1's changes non-zero (as after reading): block 0's learning signal must come back through W + dW of block
    // 1 (the row matmul's backward), as autograd gives it for a model whose block 1 weights are W + dW
    let c = cfg();
    let (_vm, m) = build_awake(c, 86);
    let toks = tokens(1, 17, 258, 22);
    let t = toks.len() - 1;
    let lr = 1e-20;
    m.plastic_begin(1, &[0, 1], &Site::ALL).unwrap();
    let mut noise = HashMap::new();
    {
        let e = m.v51b().unwrap();
        let mut g = e.plastic.borrow_mut();
        let p = g.as_mut().unwrap();
        for (k, v) in p.delta.iter_mut().filter(|(k, _)| k.0 == 1) {
            let z: Vec<f32> = crate::testkit::unit_noise(v.elem_count(), 40 + k.1 as u64).iter().map(|z| z * 0.05).collect();
            *v = Tensor::from_vec(z, v.dims(), &CPU).unwrap();
            noise.insert(k.1, v.get(0).unwrap());
        }
    }
    m.plastic_probes(true);
    let mut st = m.state(1).unwrap();
    let l = m.forward_with_state(&Tensor::from_vec(toks[..t].to_vec(), (1, t), &CPU).unwrap(), &mut st).unwrap();
    let ce = candle_nn::loss::cross_entropy(&l, &Tensor::from_vec(toks[1..].to_vec(), t, &CPU).unwrap()).unwrap().affine(t as f64, 0.0).unwrap();
    m.plastic_learn(&ce.backward().unwrap(), lr, Rule::Backprop).unwrap();
    let got: Vec<(Site, Tensor)> = Site::ALL.iter().map(|&s| (s, m.plastic_delta(0, s, 0).unwrap().unwrap().affine(-1.0 / lr, 0.0).unwrap())).collect();
    m.plastic_end();
    let (vm2, m2) = build_awake(c, 86);
    for (s, d) in &noise {
        let n = s.var_name(1);
        let w = vm2.data().lock().unwrap()[&n].as_tensor().add(d).unwrap();
        set(&vm2, &n, &w);
    }
    for (s, g) in got {
        let want = bptt(&vm2, &m2, &toks, 0, s);
        let (cos, rel) = cos_rel(&g, &want);
        eprintln!("v51 learning signal through block 1's changes, b0 {s:?}: cosine {cos:.7}, relative error {rel:.2e}");
        assert!(rel < 1e-3, "b0 {s:?}: {rel}");
    }
}
