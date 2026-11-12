// TESTING / DESIGN ONLY — tests of v51 Part B's one redesign (runs/v51/prereg_v51.md, Amendment B-4), CPU only, on tiny
// models with Ava 0.4's block recipe:
// (1) the forgetting test's shared trunk (src/v51e.rs): a replay gives the full forward bit for bit, for a brain that
//     differs in its later blocks and for a cerebellum head after the last block;
// (2) with nothing fading, one document of the redesigned stream (E2 / dyn2 / CB2) is the old per-document run bit for
//     bit (logits and changes);
// (3) the carry: a stream's next document starts from the changes the last one left, its traces from 0;
// (4) the fading is exact: with steps too small to move the forward, the kept change equals
//     sum_k 2^(-(tokens after step k) / h) · step_k (e-prop, online backprop in segments, the cerebellum's table, its
//     fold included).

use crate::arch::Arch;
use crate::eprop::{Rule, Site};
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, max_diff, tokens};
use crate::testkit::unit_noise;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

const CPU: Device = Device::Cpu;
const RECIPE: &str = "--head-size 32 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense \
                      --resid-fix --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif \
                      --homeostasis 0.1";

fn cfg(extra: &str) -> Config {
    Config::with_kind(Kind::D, 258, 64, 3, 64).with_arch(Arch::from_flags(&format!("{RECIPE} {extra}")).unwrap()).unwrap()
}

fn set(vm: &VarMap, name: &str, t: &Tensor) {
    vm.data().lock().unwrap()[name].set(t).unwrap();
}

fn get(vm: &VarMap, name: &str) -> Tensor {
    vm.data().lock().unwrap()[name].as_tensor().copy().unwrap()
}

/// A model of `c` with every weight of `vm` (its own new weights keep their init).
fn copy_of(c: Config, vm: &VarMap) -> (VarMap, Model) {
    let v2 = VarMap::new();
    let m = Model::new(c, VarBuilder::from_varmap(&v2, DType::F32, &CPU)).unwrap();
    m.set_track_activity(false);
    for (k, v) in vm.data().lock().unwrap().iter() {
        set(&v2, k, v.as_tensor());
    }
    (v2, m)
}

fn noise_like(t: &Tensor, seed: u64, s: f32) -> Tensor {
    let v: Vec<f32> = unit_noise(t.elem_count(), seed).iter().map(|z| z * s).collect();
    Tensor::from_vec(v, t.dims(), &CPU).unwrap()
}

/// Logits [b·T, V] of reading toks [b, T] in `pieces`, from a fresh state.
fn read(m: &Model, toks: &[u32], b: usize, t: usize, pieces: &[usize]) -> Tensor {
    let mut st = m.state(b).unwrap();
    let mut outs = Vec::new();
    let mut p0 = 0;
    for &p in pieces {
        let chunk: Vec<u32> = (0..b).flat_map(|r| toks[r * t + p0..r * t + p0 + p].to_vec()).collect();
        let l = m.forward_with_state(&Tensor::from_vec(chunk, (b, p), &CPU).unwrap(), &mut st).unwrap();
        outs.push(l.reshape((b, p, ())).unwrap());
        p0 += p;
    }
    Tensor::cat(&outs, 1).unwrap()
}

#[test]
fn trunk_replay_is_the_full_forward_bit_for_bit() {
    let c = cfg("");
    let (vm, a) = build_awake(c, 91);
    a.set_track_activity(false);
    let (b_, t, pieces) = (2usize, 24usize, [8usize, 16]);
    let toks = tokens(b_, t, 258, 31);
    // brain B: blocks 1 and 2 changed (their four plastic weights)
    let (vb, bm) = copy_of(c, &vm);
    for (j, i) in [1usize, 2].iter().enumerate() {
        for (k, s) in Site::ALL.iter().enumerate() {
            let n = s.var_name(*i);
            let w = get(&vb, &n);
            set(&vb, &n, &w.add(&noise_like(&w, 100 + 10 * j as u64 + k as u64, 0.05)).unwrap());
        }
    }
    let want_b = read(&bm, &toks, b_, t, &pieces);
    // brain C: a cerebellum head after the last block, its read-out not 0
    let (vc, cm) = copy_of(cfg("--cerebellum-head 512"), &vm);
    let w = get(&vc, "cbh.wo");
    set(&vc, "cbh.wo", &noise_like(&w, 7, 0.3));
    let want_c = read(&cm, &toks, b_, t, &pieces);
    // the trunk from the base model, once for both cuts
    a.trunk_record(&[1, 3]).unwrap();
    let base = read(&a, &toks, b_, t, &pieces);
    let mut recs = a.trunk_take_records();
    assert_eq!(max_diff(&base, &read(&a, &toks, b_, t, &pieces)), 0.0, "recording changes nothing");
    let rc = recs.pop().unwrap();
    let rb = recs.pop().unwrap();
    assert_eq!((rb.k, rb.pieces.len(), rc.k, rc.pieces.len()), (1, 2, 3, 2));
    eprintln!("v51e trunk: records of {} and {} bytes", rb.bytes(), rc.bytes());
    bm.trunk_replay(rb).unwrap();
    let got_b = read(&bm, &toks, b_, t, &pieces);
    let rb = bm.trunk_take().unwrap();
    cm.trunk_replay(rc).unwrap();
    let got_c = read(&cm, &toks, b_, t, &pieces);
    cm.trunk_take().unwrap();
    assert!(max_diff(&want_b, &base) > 1e-3 && max_diff(&want_c, &base) > 1e-3, "the brains differ from the base");
    assert_eq!(max_diff(&got_b, &want_b), 0.0, "brain B: replay vs full forward");
    assert_eq!(max_diff(&got_c, &want_c), 0.0, "brain C: replay vs full forward");
    // a replay past its record, or of another shape, is refused
    bm.trunk_replay(rb).unwrap();
    drop(read(&bm, &toks, b_, t, &pieces));
    let mut st = bm.state(b_).unwrap();
    assert!(bm.forward_with_state(&Tensor::from_vec(toks[..b_ * 4].to_vec(), (b_, 4), &CPU).unwrap(), &mut st).is_err());
    // tiled: 2 x the rows, each copy reading the record's rows
    let mut rb = bm.trunk_take().unwrap();
    rb.next = 0;
    let toks2: Vec<u32> = toks.iter().chain(&toks).copied().collect();
    let want2 = read(&bm, &toks2, 2 * b_, t, &pieces);
    bm.trunk_replay(rb).unwrap();
    let got2 = read(&bm, &toks2, 2 * b_, t, &pieces);
    bm.trunk_take().unwrap();
    let d2 = max_diff(&got2, &want2);
    eprintln!("v51e trunk tiled x2: max |logit difference| vs the full forward {d2:.2e}");
    assert!(d2 < 1e-5, "tiled replay {d2}");
    // refused where blocks have side effects
    let (_vt, mt) = copy_of(cfg("--thalamus"), &vm);
    assert!(mt.trunk_record(&[1]).is_err());
}

/// One document, one row, e-prop (seg 1) or online backprop (seg > 1), at lr; with `half` set (the stream's way) or
/// not (E's way): the logits of every token and the changes at the end.
fn one_doc(m: &Model, toks: &[u32], seg: usize, lr: f64, half: Option<f64>) -> (Vec<Tensor>, Vec<((usize, Site), Tensor)>) {
    let rule = if seg == 1 { Rule::EProp } else { Rule::Backprop };
    m.plastic_begin(1, &[1, 2], &Site::ALL).unwrap();
    if let Some(h) = half {
        m.plastic_set_half_lives(&[h]).unwrap();
        m.plastic_new_documents().unwrap();
    }
    m.plastic_probes(true);
    let mut st = m.state(1).unwrap();
    let n = toks.len() - 1;
    let mut logits = Vec::new();
    for p0 in (0..n).step_by(seg) {
        let k = seg.min(n - p0);
        let l = m.forward_with_state(&Tensor::from_vec(toks[p0..p0 + k].to_vec(), (1, k), &CPU).unwrap(), &mut st).unwrap();
        let loss = candle_nn::loss::cross_entropy(&l, &Tensor::from_vec(toks[p0 + 1..p0 + k + 1].to_vec(), k, &CPU).unwrap()).unwrap().affine(k as f64, 0.0).unwrap();
        m.plastic_learn(&loss.backward().unwrap(), lr, rule).unwrap();
        st.detach_all();
        logits.push(l.detach());
    }
    let d = m.plastic_deltas().unwrap();
    m.plastic_end();
    (logits, d)
}

#[test]
fn without_fading_one_document_is_the_old_run_bit_for_bit() {
    let (vm, m) = build_awake(cfg(""), 92);
    m.set_track_activity(false);
    let toks = tokens(1, 33, 258, 32);
    for seg in [1usize, 4] {
        let (la, da) = one_doc(&m, &toks, seg, 1e-3, None);
        let (lb, db) = one_doc(&m, &toks, seg, 1e-3, Some(f64::INFINITY));
        for (x, y) in la.iter().zip(&lb) {
            assert_eq!(max_diff(x, y), 0.0, "seg {seg}: logits");
        }
        for ((ka, a), (kb, b)) in da.iter().zip(&db) {
            assert_eq!(ka, kb);
            assert_eq!(max_diff(a, b), 0.0, "seg {seg}: {ka:?}");
            assert!(a.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap() > 0.0, "seg {seg}: {ka:?} learned");
        }
    }
    // the cerebellum: an infinite half-life is CB online itself
    let (vc, cm) = copy_of(cfg("--cerebellum-head 512"), &vm);
    set(&vc, "cbh.wo", &Tensor::zeros((512, 64), DType::F32, &CPU).unwrap());
    let run = |half: Option<f64>| -> (Vec<Tensor>, Vec<f32>) {
        cm.cb_online_begin(1, 0.01).unwrap();
        if let Some(h) = half {
            cm.cb_online_set_half_life(0, h).unwrap();
        }
        let mut st = cm.state(1).unwrap();
        let mut ls = Vec::new();
        for j in 0..32 {
            let l = cm.forward_with_state(&Tensor::new(&[[toks[j]]], &CPU).unwrap(), &mut st).unwrap();
            cm.cb_learn(&l, &[toks[j + 1]]).unwrap();
            ls.push(l);
        }
        let s = cm.cb_online_sum().unwrap();
        cm.cb_online_end();
        (ls, s)
    };
    let (la, sa) = run(None);
    let (lb, sb) = run(Some(f64::INFINITY));
    for (x, y) in la.iter().zip(&lb) {
        assert_eq!(max_diff(x, y), 0.0, "cb: logits");
    }
    assert!(sa == sb && sa.iter().any(|&v| v != 0.0), "cb: tables");
}

#[test]
fn a_stream_carries_its_changes_to_the_next_document() {
    let (vm, m) = build_awake(cfg(""), 93);
    m.set_track_activity(false);
    let d1 = tokens(1, 25, 258, 33);
    let d2 = tokens(1, 25, 258, 34);
    m.plastic_begin(1, &[1, 2], &Site::ALL).unwrap();
    m.plastic_set_half_lives(&[64.0]).unwrap();
    m.plastic_probes(true);
    let mut st = m.state(1).unwrap();
    for j in 0..24 {
        let l = m.forward_with_state(&Tensor::new(&[[d1[j]]], &CPU).unwrap(), &mut st).unwrap();
        let loss = candle_nn::loss::cross_entropy(&l, &Tensor::new(&[d1[j + 1]], &CPU).unwrap()).unwrap();
        m.plastic_learn(&loss.backward().unwrap(), 1e-2, Rule::EProp).unwrap();
        st.detach_all();
    }
    let before = m.plastic_deltas().unwrap();
    let tr_max = |m: &Model| -> f32 {
        let e = m.v51b().unwrap();
        let g = e.plastic.borrow();
        g.as_ref().unwrap().trace.values().map(|t| t.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()).fold(0.0, f32::max)
    };
    assert!(tr_max(&m) > 0.0);
    m.plastic_new_documents().unwrap();
    assert_eq!(tr_max(&m), 0.0, "the traces restart with the document");
    for ((ka, a), (kb, b)) in before.iter().zip(&m.plastic_deltas().unwrap()) {
        assert_eq!(ka, kb);
        assert_eq!(max_diff(a, b), 0.0, "the changes are kept: {ka:?}");
    }
    // document 2's first word, from a fresh state, reads with the kept changes: a plain model with W + dW
    let mut st2 = m.state(1).unwrap();
    let carried = m.forward_with_state(&Tensor::new(&[[d2[0]]], &CPU).unwrap(), &mut st2).unwrap();
    m.plastic_end();
    let (vp, pm) = copy_of(cfg(""), &vm);
    for ((i, s), d) in &before {
        let n = s.var_name(*i);
        set(&vp, &n, &get(&vm, &n).add(&d.get(0).unwrap()).unwrap());
    }
    let mut sp = pm.state(1).unwrap();
    let plain = pm.forward_with_state(&Tensor::new(&[[d2[0]]], &CPU).unwrap(), &mut sp).unwrap();
    let mut sf = m.state(1).unwrap();
    let fresh = m.forward_with_state(&Tensor::new(&[[d2[0]]], &CPU).unwrap(), &mut sf).unwrap();
    let (dp, df) = (max_diff(&carried, &plain), max_diff(&carried, &fresh));
    eprintln!("v51e carry: document 2's first logits vs W + dW {dp:.2e}, vs the saved weights {df:.2e}");
    assert!(dp < 1e-4 && df > 1e-4, "carry: {dp} {df}");
}

/// The closed form sum_k f^(K-k) step_k in f64, from steps [K][elements].
fn closed(steps: &[Vec<f32>], f: f64) -> Vec<f64> {
    let k = steps.len();
    let mut out = vec![0f64; steps[0].len()];
    for (j, s) in steps.iter().enumerate() {
        let w = f.powi((k - 1 - j) as i32);
        for (o, &v) in out.iter_mut().zip(s) {
            *o += w * v as f64;
        }
    }
    out
}

fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    let top = want.iter().fold(0f64, |a, &b| a.max(b.abs()));
    got.iter().zip(want).fold(0f64, |a, (&g, &w)| a.max((g as f64 - w).abs())) / top.max(1e-300)
}

#[test]
fn fading_is_exact() {
    let (vm, m) = build_awake(cfg(""), 94);
    m.set_track_activity(false);
    let toks = tokens(1, 33, 258, 35);
    let lr = 1e-20; // steps far too small to move the forward: both runs see the same signals
    for (seg, h) in [(1usize, 3.0f64), (4, 5.0)] {
        let rule = if seg == 1 { Rule::EProp } else { Rule::Backprop };
        let n = toks.len() - 1;
        // run 1: nothing fades, each step alone (the changes set to 0 before it); run 2: half-life h
        let mut steps: Vec<Vec<((usize, Site), Vec<f32>)>> = Vec::new();
        let mut logits: [Vec<Tensor>; 2] = [Vec::new(), Vec::new()];
        let mut kept = Vec::new();
        for run in 0..2 {
            m.plastic_begin(1, &[1, 2], &Site::ALL).unwrap();
            if run == 1 {
                m.plastic_set_half_lives(&[h]).unwrap();
            }
            m.plastic_probes(true);
            let mut st = m.state(1).unwrap();
            for p0 in (0..n).step_by(seg) {
                let k = seg.min(n - p0);
                if run == 0 {
                    for ((i, s), d) in m.plastic_deltas().unwrap() {
                        m.plastic_set_delta(i, s, &d.zeros_like().unwrap()).unwrap();
                    }
                }
                let l = m.forward_with_state(&Tensor::from_vec(toks[p0..p0 + k].to_vec(), (1, k), &CPU).unwrap(), &mut st).unwrap();
                let loss = candle_nn::loss::cross_entropy(&l, &Tensor::from_vec(toks[p0 + 1..p0 + k + 1].to_vec(), k, &CPU).unwrap()).unwrap().affine(k as f64, 0.0).unwrap();
                m.plastic_learn(&loss.backward().unwrap(), lr, rule).unwrap();
                st.detach_all();
                logits[run].push(l.detach());
                if run == 0 {
                    steps.push(m.plastic_deltas().unwrap().into_iter().map(|(k, d)| (k, d.flatten_all().unwrap().to_vec1::<f32>().unwrap())).collect());
                }
            }
            if run == 1 {
                kept = m.plastic_deltas().unwrap();
            }
            m.plastic_end();
        }
        for (x, y) in logits[0].iter().zip(&logits[1]) {
            assert_eq!(max_diff(x, y), 0.0, "seg {seg}: the forward did not move");
        }
        let f = (-(seg as f64) / h).exp2();
        for (j, (key, d)) in kept.iter().enumerate() {
            let per: Vec<Vec<f32>> = steps.iter().map(|s| s[j].1.clone()).collect();
            assert_eq!(steps[0][j].0, *key);
            let want = closed(&per, f);
            let got: Vec<f32> = d.flatten_all().unwrap().to_vec1().unwrap();
            let e = rel_err(&got, &want);
            eprintln!("v51e fading (seg {seg}, half-life {h} tokens) {key:?}: relative error vs the closed form {e:.2e}");
            assert!(e < 1e-5, "seg {seg} {key:?}: {e}");
            // and it is not the unfaded sum
            let flat = closed(&per, 1.0);
            assert!(rel_err(&got, &flat) > 1e-2, "seg {seg} {key:?}: faded");
        }
    }
    // the cerebellum's table, the fold included (h 0.4: 2^-75 over 30 words, past the fold at 1e-18)
    let (vc, cm) = copy_of(cfg("--cerebellum-head 512"), &vm);
    set(&vc, "cbh.wo", &Tensor::zeros((512, 64), DType::F32, &CPU).unwrap());
    for h in [3.0f64, 0.4] {
        let mut steps: Vec<Vec<f32>> = Vec::new();
        let mut kept = Vec::new();
        for run in 0..2 {
            cm.cb_online_begin(1, 1e-20).unwrap();
            if run == 1 {
                cm.cb_online_set_half_life(0, h).unwrap();
            }
            let mut st = cm.state(1).unwrap();
            for j in 0..30 {
                if run == 0 && j > 0 {
                    cm.cb_online_with(|o| o.set_raw(0, vec![0.0; 512 * 64], 1.0)).unwrap();
                }
                let l = cm.forward_with_state(&Tensor::new(&[[toks[j]]], &CPU).unwrap(), &mut st).unwrap();
                cm.cb_learn(&l, &[toks[j + 1]]).unwrap();
                if run == 0 {
                    steps.push(cm.cb_online_with(|o| o.table(0)).unwrap().unwrap());
                }
            }
            if run == 1 {
                kept = cm.cb_online_with(|o| o.table(0)).unwrap().unwrap();
            }
            cm.cb_online_end();
        }
        let want = closed(&steps, (-1.0 / h).exp2());
        let e = rel_err(&kept, &want);
        eprintln!("v51e cerebellum fading (half-life {h} tokens): relative error vs the closed form {e:.2e}");
        assert!(e < 1e-5, "cb h {h}: {e}");
    }
}
