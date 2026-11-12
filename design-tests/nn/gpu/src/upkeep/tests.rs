// TESTING / DESIGN ONLY — tests of src/upkeep.rs (v42 A8 ReDo and shrink-and-perturb, A9 slot counts, I3 ReDo for block options), CPU; the _gpu twins are ignored.

use super::*;
use crate::arch::Arch;
use crate::model::{Config, Kind};
use crate::state::tests::{build_awake, max_diff, one_pass, tokens};
use candle_core::{Device, Tensor};

const CPU: Device = Device::Cpu;

fn d_cfg() -> Config {
    Config::with_kind(Kind::D, 256, 32, 2, 64)
}

fn get(vm: &VarMap, name: &str) -> Vec<f32> {
    vm.data().lock().unwrap()[name].as_tensor().flatten_all().unwrap().to_vec1().unwrap()
}

fn set_col(vm: &VarMap, name: &str, c: usize, x: f32) {
    let var = vm.data().lock().unwrap()[name].clone();
    let (r, cols) = var.dims2().unwrap();
    let mut w: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    for i in 0..r {
        w[i * cols + c] = x;
    }
    var.set(&Tensor::from_vec(w, (r, cols), &CPU).unwrap()).unwrap();
}

fn col(w: &[f32], cols: usize, c: usize) -> Vec<f32> {
    w.iter().skip(c).step_by(cols).copied().collect()
}

fn batches(vocab: usize) -> Vec<Tensor> {
    (0..2).map(|s| Tensor::from_vec(tokens(2, 32, vocab, 40 + s), (2, 32), &CPU).unwrap()).collect()
}

#[test]
fn dormant_units_by_their_share_of_the_mean() {
    assert_eq!(dormant(&[1.0, 0.0, 2.0, 0.05], 0.1), vec![1, 3]);
    assert_eq!(dormant(&[0.0, 0.0], 0.1), vec![0, 1]); // a silent layer: every unit
    assert_eq!(dormant(&[1.0, 1.0], 0.0), Vec::<usize>::new());
    let z = normals(20001, 9);
    let mean = z.iter().map(|&x| x as f64).sum::<f64>() / z.len() as f64;
    let var = z.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / z.len() as f64;
    assert!(mean.abs() < 0.03 && (var - 1.0).abs() < 0.05, "mean {mean} var {var}");
}

/// Dead units (an all-zero incoming column never fires) are found and recycled: a fresh incoming column at the live
/// columns' scale, a zero outgoing row; every other weight is untouched; recycling LIF channels keeps every output.
#[test]
fn redo_recycles_dead_units_and_nothing_else() {
    let cfg = d_cfg();
    let (vm, m) = build_awake(cfg, 3);
    set_col(&vm, "b0.mem.win", 3, 0.0);
    set_col(&vm, "b1.mlp.w1", 5, 0.0);
    let act = unit_activity(&m, &batches(cfg.vocab)).unwrap();
    assert_eq!(act.lif[0][3], 0.0);
    assert_eq!(act.mlp[1][5], 0.0);
    let toks = tokens(2, 24, cfg.vocab, 8);
    let before = one_pass(&m, &toks, 2, 24);
    let snap: Vec<(String, Vec<f32>)> = ["b0.mem.win", "b0.mem.wout", "b1.mlp.w1", "b1.mlp.w2", "b0.mlp.w1", "b1.hebb.wq"].iter().map(|n| (n.to_string(), get(&vm, n))).collect();

    // LIF channels only (the MLP's activity set to all live): the output stays exactly the same
    let mut lif_only = act.clone();
    lif_only.mlp.iter_mut().for_each(|v| v.iter_mut().for_each(|x| *x = 1.0));
    let tau = 0.0;
    let rep = redo(&vm, &m, &lif_only, tau, 77).unwrap();
    assert!(rep.lif[0] >= 1 && rep.mlp.iter().sum::<usize>() == 0, "{rep:?}");
    assert_eq!(max_diff(&before, &one_pass(&m, &toks, 2, 24)), 0.0, "recycled LIF channels change nothing");
    let (d, h) = (cfg.d, cfg.hidden());
    let win = get(&vm, "b0.mem.win");
    let fresh = col(&win, d, 3);
    let live: Vec<f32> = (0..d).filter(|&c| c != 3).flat_map(|c| col(&snap[0].1, d, c)).collect();
    let rms = |v: &[f32]| (v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    assert!(rms(&fresh) > 0.3 * rms(&live) && rms(&fresh) < 3.0 * rms(&live), "fresh {} live {}", rms(&fresh), rms(&live));
    assert!(get(&vm, "b0.mem.wout")[3 * d..4 * d].iter().all(|&x| x == 0.0));
    for c in (0..d).filter(|&c| c != 3) {
        assert_eq!(col(&win, d, c), col(&snap[0].1, d, c));
    }
    assert_eq!(get(&vm, "b1.hebb.wq"), snap[5].1);
    // the revived channel fires now
    let act2 = unit_activity(&m, &batches(cfg.vocab)).unwrap();
    assert!(act2.lif[0][3] > 0.0, "revived channel rate {}", act2.lif[0][3]);

    // the MLP unit: fresh column of w1, zero row of w2, the rest of both as before
    let rep = redo(&vm, &m, &act2, tau, 78).unwrap();
    assert!(rep.mlp[1] >= 1, "{rep:?}");
    let w1 = get(&vm, "b1.mlp.w1");
    assert!(col(&w1, h, 5).iter().any(|&x| x != 0.0));
    assert!(get(&vm, "b1.mlp.w2")[5 * d..6 * d].iter().all(|&x| x == 0.0));
    for c in (0..h).filter(|&c| c != 5 && act2.mlp[1][c] > 0.0) {
        assert_eq!(col(&w1, h, c), col(&snap[2].1, h, c));
    }
    assert!(one_pass(&m, &toks, 2, 24).flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|x| x.is_finite()));

    // no dormant unit: nothing written
    let all_live = UnitActivity { lif: vec![vec![1.0; d]; 2], mlp: vec![vec![1.0; h]; 2] };
    let w_before = get(&vm, "b0.mlp.w1");
    let rep = redo(&vm, &m, &all_live, 0.1, 1).unwrap();
    assert_eq!(rep.total(), 0);
    assert_eq!(get(&vm, "b0.mlp.w1"), w_before);
}

#[test]
fn shrink_and_perturb_scales_the_hidden_matrices_only() {
    let cfg = Config::with_kind(Kind::D, 256, 32, 2, 64).with_engram(1).unwrap();
    let vm = VarMap::new();
    let _m = Model::new(cfg, candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
    crate::testkit::fix_weights(&vm, &cfg, 5);
    let names: Vec<String> = { vm.data().lock().unwrap().keys().cloned().collect() };
    let snap: Vec<(String, Vec<f32>)> = names.iter().map(|n| (n.clone(), get(&vm, n))).collect();
    // lambda 1, sigma 0: nothing written
    assert_eq!(shrink_perturb(&vm, 1.0, 0.0, 1).unwrap(), 0);
    for (n, v) in &snap {
        assert_eq!(&get(&vm, n), v, "{n}");
    }
    // shrink only: every decayed non-zero matrix exactly halves; the rest (norms, embeddings, the engram's Wo at 0) stay
    let changed = shrink_perturb(&vm, 0.5, 0.0, 1).unwrap();
    assert!(changed > 0);
    for (n, v) in &snap {
        let now = get(&vm, n);
        if Model::decays(n) && v.iter().any(|&x| x != 0.0) {
            assert!(now.iter().zip(v).all(|(a, b)| *a == 0.5 * *b), "{n}");
        } else {
            assert_eq!(&now, v, "{n}");
        }
    }
    assert!(get(&vm, "b1.eng.wo").iter().all(|&x| x == 0.0));
    // perturb: seeded, so twice from the same weights gives the same weights; the noise is sigma · rms
    let base = get(&vm, "b0.mlp.w1");
    let rms = (base.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / base.len() as f64).sqrt();
    shrink_perturb(&vm, 1.0, 0.1, 9).unwrap();
    let a = get(&vm, "b0.mlp.w1");
    let noise = (a.iter().zip(&base).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / base.len() as f64).sqrt();
    assert!((noise / (0.1 * rms) - 1.0).abs() < 0.1, "noise {noise} want {}", 0.1 * rms);
}

/// I3: block options are traced and revived now (the tests below); T, with no spiking layer, is still refused.
#[test]
fn redo_refuses_t() {
    let act = UnitActivity { lif: vec![vec![0.0; 32]; 2], mlp: vec![vec![0.0; 128]; 2] };
    let (vm, m) = build_awake(Config::with_kind(Kind::T, 256, 32, 2, 64), 2);
    assert!(redo(&vm, &m, &act, 0.1, 1).is_err());
}

// ---------------------------------------------------------------- I3: ReDo for block options and in training

/// Ava 0.4's everything-on set (Stage 3's winners, the five slow options, Wave 2C's and Wave 5's parts) at d 64.
pub(crate) const ALL_ON: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
     --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1 \
     --dendrites 4 --stp --delays 4 --wm 8 --resonate 16 --modulate --tem 4 --halting 3 --cerebellum 64 --workspace 4 \
     --holo 64 --sync 64 --jepa 0.5 --diffusion 0.5 --critic 0.5 --confidence 0.5 --topo 0.5 --schema 0.5";

pub(crate) fn all_on_cfg() -> Config {
    Config::with_kind(Kind::D, 258, 64, 2, 64).with_arch(Arch::from_flags(ALL_ON).unwrap()).unwrap()
}

/// build_awake's model of `cfg`, on `dev` (the same weights: built on the CPU, then copied).
fn awake_on(cfg: Config, seed: u64, dev: &Device) -> (VarMap, Model) {
    let (vm, m) = build_awake(cfg, seed);
    if dev.is_cpu() {
        return (vm, m);
    }
    let vg = VarMap::new();
    let mg = Model::new(cfg, candle_nn::VarBuilder::from_varmap(&vg, DType::F32, dev)).unwrap();
    {
        let (src, dst) = (vm.data().lock().unwrap(), vg.data().lock().unwrap());
        for (k, v) in src.iter() {
            dst[k].set(&v.as_tensor().to_device(dev).unwrap()).unwrap();
        }
    }
    mg.set_track_activity(false);
    (vg, mg)
}

fn snapshot(vm: &VarMap) -> Vec<(String, Vec<f32>)> {
    let mut names: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
    names.sort();
    names.into_iter().map(|n| (n.clone(), get(vm, &n))).collect()
}

/// Set element i of a weight (on its own device).
fn set_at(vm: &VarMap, name: &str, i: usize, x: f32) {
    let var = vm.data().lock().unwrap()[name].clone();
    let mut w: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    w[i] = x;
    var.set(&Tensor::from_vec(w, var.shape(), var.device()).unwrap()).unwrap();
}

/// Set column c of a matrix to x (on its own device).
fn set_col_on(vm: &VarMap, name: &str, c: usize, x: f32) {
    let var = vm.data().lock().unwrap()[name].clone();
    let (r, cols) = var.dims2().unwrap();
    let mut w: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    (0..r).for_each(|i| w[i * cols + c] = x);
    var.set(&Tensor::from_vec(w, (r, cols), var.device()).unwrap()).unwrap();
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
}

/// Fixed (ids, targets) [2, 32] on `dev`.
fn io(vocab: usize, dev: &Device) -> (Tensor, Tensor) {
    let toks = tokens(2, 33, vocab, 8);
    let ids: Vec<u32> = (0..2).flat_map(|r| toks[r * 33..r * 33 + 32].to_vec()).collect();
    let tg: Vec<u32> = (0..2).flat_map(|r| toks[r * 33 + 1..r * 33 + 33].to_vec()).collect();
    (Tensor::from_vec(ids, (2, 32), dev).unwrap(), Tensor::from_vec(tg, (2, 32), dev).unwrap())
}

fn batches_on(vocab: usize, dev: &Device) -> Vec<Tensor> {
    batches(vocab).into_iter().map(|b| b.to_device(dev).unwrap()).collect()
}

/// Logits, and the training loss (total with every auxiliary term, and the next-token part), of fixed tokens.
fn outputs(m: &Model, vocab: usize, dev: &Device) -> (Tensor, Tensor, Tensor) {
    let (ids, tg) = io(vocab, dev);
    let logits = m.forward(&ids).unwrap();
    let (total, ce, _) = m.loss_aux(&ids, &tg, 0.0).unwrap();
    (logits, total, ce)
}

/// I3: the record is off outside a measurement, a measurement changes no value (the logits, the training loss and
/// every gradient are the same bits before and after one), and the record is off again afterwards.
fn measurement_case(dev: &Device) {
    let cfg = all_on_cfg();
    let (vm, m) = awake_on(cfg, 3, dev);
    let ext = m.ext.as_ref().unwrap();
    assert!(ext.redo_rec.borrow().is_none());
    let grads_of = |m: &Model| {
        let (ids, tg) = io(cfg.vocab, dev);
        let (total, _, _) = m.loss_aux(&ids, &tg, 0.0).unwrap();
        let mut g = total.backward().unwrap();
        crate::arch::ckpt::merge(&mut g).unwrap();
        let data = vm.data().lock().unwrap();
        let mut names: Vec<&String> = data.keys().collect();
        names.sort();
        names.into_iter().map(|n| g.get(data[n].as_tensor()).map(bits)).collect::<Vec<_>>()
    };
    let (l0, t0, c0) = outputs(&m, cfg.vocab, dev);
    let g0 = grads_of(&m);
    let act = unit_activity(&m, &batches_on(cfg.vocab, dev)).unwrap();
    assert!(ext.redo_rec.borrow().is_none());
    assert_eq!((act.lif.len(), act.mlp.len()), (2, 2));
    assert_eq!((act.lif[0].len(), act.mlp[0].len()), (cfg.d, cfg.hidden()));
    assert!(act.lif.iter().chain(&act.mlp).all(|v| v.iter().all(|x| x.is_finite() && *x >= 0.0)));
    assert!(act.lif[0].iter().any(|&x| x > 0.0) && act.mlp[1].iter().any(|&x| x > 0.0));
    let (l1, t1, c1) = outputs(&m, cfg.vocab, dev);
    assert_eq!(bits(&l0), bits(&l1));
    assert_eq!((bits(&t0), bits(&c0)), (bits(&t1), bits(&c1)));
    assert_eq!(g0, grads_of(&m), "the gradients after a measurement");
}

#[test]
fn arch_measurement_leaves_no_trace() {
    measurement_case(&CPU);
}

#[test]
#[ignore]
fn arch_measurement_leaves_no_trace_gpu() {
    measurement_case(&Device::new_cuda(0).expect("no CUDA device"));
}

/// I3, everything on: a silent LIF channel (all-zero input column) is found and revived; its per-channel tensors go
/// back to their init (and no other channel's); every output (the logits, the training loss with every auxiliary
/// term) stays exactly the same; the revived channel fires afterwards. Every [d] tensor on the LIF path is either
/// reset or kept on purpose. Then channels that still fire a little (tau 0.5): the size of the change is printed.
fn lif_case(dev: &Device) {
    let cfg = all_on_cfg();
    let (vm, m) = awake_on(cfg, 3, dev);
    let d = cfg.d;
    for (name, v) in snapshot(&vm) {
        let Some(rest) = name.strip_prefix("b0.") else { continue };
        let on_path = rest.starts_with("mem.") || ["parts.dend.", "parts.delays.", "parts.stp.", "parts.res."].iter().any(|p| rest.starts_with(p));
        if on_path && v.len() == d && rest != "mem.norm" {
            assert!(CHANNEL_RESETS.contains(&rest) || CHANNEL_KEEPS.contains(&rest), "{name}: reset or keep?");
        }
    }
    let inits = ChannelInit::of(&m).unwrap();
    assert_eq!(inits.len(), 2 * CHANNEL_RESETS.len(), "every reset tensor exists with everything on");
    assert_eq!(inits.get("b0.mem.beta").unwrap(), crate::arch::logspaced_beta_params(d).as_slice());
    assert!(inits.get("b1.mem.theta").unwrap().iter().all(|&x| x == crate::model::LIF_THRESHOLD as f32));
    // channel 3 of block 0: silent, and every per-channel value moved off its init
    set_col_on(&vm, "b0.mem.win", 3, 0.0);
    for s in CHANNEL_RESETS {
        set_at(&vm, &format!("b0.{s}"), 3, 0.37);
    }
    let act = unit_activity(&m, &batches_on(cfg.vocab, dev)).unwrap();
    assert_eq!(act.lif[0][3], 0.0);
    let (l0, t0, c0) = outputs(&m, cfg.vocab, dev);
    let snap = snapshot(&vm);
    let mut lif_only = act.clone();
    lif_only.mlp.iter_mut().for_each(|v| v.iter_mut().for_each(|x| *x = 1.0));
    let rev = revive(&vm, &m, &lif_only, 0.0, 77, None, Some(&inits)).unwrap();
    let dead: Vec<usize> = dormant(&act.lif[0], 0.0);
    assert!(dead.contains(&3) && rev.report.lif[0] == dead.len() && rev.report.mlp.iter().sum::<usize>() == 0, "{:?}", rev.report);
    let (l1, t1, c1) = outputs(&m, cfg.vocab, dev);
    assert_eq!(bits(&l0), bits(&l1), "logits");
    assert_eq!((bits(&t0), bits(&c0)), (bits(&t1), bits(&c1)), "training loss");
    // what changed: win columns, wout rows and the reset tensors at the revived channels; nothing else
    let changed: std::collections::HashMap<&str, &Vec<bool>> = rev.places.iter().map(|(n, p)| (n.as_str(), p)).collect();
    for (name, old) in &snap {
        let now = get(&vm, name);
        match changed.get(name.as_str()) {
            None => assert_eq!(&now, old, "{name} untouched"),
            Some(p) => {
                for i in 0..now.len() {
                    if !p[i] {
                        assert_eq!(now[i].to_bits(), old[i].to_bits(), "{name}[{i}] outside the revived places");
                    }
                }
            }
        }
    }
    assert!(get(&vm, "b0.mem.wout")[3 * d..4 * d].iter().all(|&x| x == 0.0));
    for s in CHANNEL_RESETS {
        let name = format!("b0.{s}");
        assert_eq!(get(&vm, &name)[3], inits.get(&name).unwrap()[3], "{name} back to init");
    }
    assert!(col(&get(&vm, "b0.mem.win"), d, 3).iter().any(|&x| x != 0.0));
    let act2 = unit_activity(&m, &batches_on(cfg.vocab, dev)).unwrap();
    assert!(act2.lif[0][3] > 0.0, "the revived channel fires: {}", act2.lif[0][3]);
    // channels that still fire a little (at or below half the mean) lose that little through wout: printed
    let mut few = act2;
    few.mlp.iter_mut().for_each(|v| v.iter_mut().for_each(|x| *x = 1.0));
    let rev = revive(&vm, &m, &few, 0.5, 80, None, Some(&inits)).unwrap();
    let (l2, _, c2) = outputs(&m, cfg.vocab, dev);
    let (dl, dc) = (max_diff(&l1, &l2), (c2.to_scalar::<f32>().unwrap() - c1.to_scalar::<f32>().unwrap()).abs());
    let scale = l1.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    eprintln!("I3 LIF revival at tau 0.5 ({:?} of {d} channels): max |d logit| {dl:.3e} of {scale:.3e}, |d next-token loss| {dc:.3e}", rev.report.lif);
    assert!(dl.is_finite() && dc.is_finite());
}

#[test]
fn arch_redo_lif_keeps_every_output() {
    lif_case(&CPU);
}

#[test]
#[ignore]
fn arch_redo_lif_keeps_every_output_gpu() {
    lif_case(&Device::new_cuda(0).expect("no CUDA device"));
}

/// I3, everything on: a dead MLP unit (all-zero w1 column) is revived: a fresh column, a zero w2 row. kWTA lets the
/// revived unit push another out, so the outputs move a little; the size is printed and bounded. Then many units at
/// once (tau 0.5): the size is printed.
fn mlp_case(dev: &Device) {
    let cfg = all_on_cfg();
    let (vm, m) = awake_on(cfg, 4, dev);
    let (d, h) = (cfg.d, cfg.hidden());
    set_col_on(&vm, "b1.mlp.w1", 5, 0.0);
    let act = unit_activity(&m, &batches_on(cfg.vocab, dev)).unwrap();
    assert_eq!(act.mlp[1][5], 0.0);
    let (l0, _, _) = outputs(&m, cfg.vocab, dev);
    let mut mlp_only = act.clone();
    mlp_only.lif.iter_mut().for_each(|v| v.iter_mut().for_each(|x| *x = 1.0));
    let rev = revive(&vm, &m, &mlp_only, 0.0, 78, None, None).unwrap();
    assert!(rev.report.mlp[1] >= 1 && rev.report.lif.iter().sum::<usize>() == 0, "{:?}", rev.report);
    assert!(col(&get(&vm, "b1.mlp.w1"), h, 5).iter().any(|&x| x != 0.0));
    assert!(get(&vm, "b1.mlp.w2")[5 * d..6 * d].iter().all(|&x| x == 0.0));
    let (l1, _, _) = outputs(&m, cfg.vocab, dev);
    let diff = max_diff(&l0, &l1);
    let scale = l0.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    eprintln!("I3 MLP revival ({} units): max |d logit| {diff:.3e} of max |logit| {scale:.3e}", rev.report.mlp.iter().sum::<usize>());
    assert!(diff < 0.05 * scale, "{diff} vs {scale}");
    // many at once (every unit at or below half its layer's mean): printed, so the size is on record
    let (_, _, c1) = outputs(&m, cfg.vocab, dev);
    let mut many = unit_activity(&m, &batches_on(cfg.vocab, dev)).unwrap();
    many.lif.iter_mut().for_each(|v| v.iter_mut().for_each(|x| *x = 1.0));
    let rev = revive(&vm, &m, &many, 0.5, 79, None, None).unwrap();
    let (l2, _, c2) = outputs(&m, cfg.vocab, dev);
    let (dl, dc) = (max_diff(&l1, &l2), (c2.to_scalar::<f32>().unwrap() - c1.to_scalar::<f32>().unwrap()).abs());
    eprintln!("I3 MLP revival at tau 0.5 ({:?} of {h} units): max |d logit| {dl:.3e}, |d next-token loss| {dc:.3e}", rev.report.mlp);
    assert!(dl.is_finite() && dc.is_finite());
}

#[test]
fn arch_redo_mlp_moves_outputs_a_little() {
    mlp_case(&CPU);
}

#[test]
#[ignore]
fn arch_redo_mlp_moves_outputs_a_little_gpu() {
    mlp_case(&Device::new_cuda(0).expect("no CUDA device"));
}

/// I3: the optimizer's moments (m, v) are 0 at every revived place and the same bits elsewhere, in every tensor; under
/// H6 a revived MLP column gets values only at live places, and the masks' live counts do not change.
fn moments_case(dev: &Device) {
    use crate::optim::{AdamW, Param};
    use crate::plastic::{RewireCfg, Rewiring};
    let cfg = all_on_cfg();
    let (vm, m) = awake_on(cfg, 5, dev);
    let (d, h) = (cfg.d, cfg.hidden());
    let mut names: Vec<(String, candle_core::Var)> = vm.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    names.sort_by(|a, b| a.0.cmp(&b.0));
    let params: Vec<Param> = names.into_iter().map(|(name, var)| Param { decay: Model::decays(&name), name, var }).collect();
    let mut opt = AdamW::new(params, 0.1).unwrap();
    let rw = Rewiring::fresh(RewireCfg { density: 0.5, frac: 0.1, every: 100 }, &opt, 3).unwrap();
    let (ids, tg) = io(cfg.vocab, dev);
    let one = Tensor::new(1.0f64, dev).unwrap();
    for _ in 0..2 {
        let (total, _, _) = m.loss_aux(&ids, &tg, 0.0).unwrap();
        let mut g = total.backward().unwrap();
        crate::arch::ckpt::merge(&mut g).unwrap();
        opt.step_t(&g, 1e-3, &one).unwrap();
        rw.apply(&opt).unwrap();
    }
    set_col_on(&vm, "b0.mem.win", 3, 0.0);
    set_col_on(&vm, "b1.mlp.w1", 5, 0.0);
    let masks = rw.host_masks().unwrap();
    let live0: Vec<usize> = rw.names().iter().map(|n| masks[n].iter().filter(|&&x| x).count()).collect();
    let before: Vec<(Vec<u32>, Vec<u32>)> = (0..opt.params().len()).map(|i| (bits(opt.moments(i).0), bits(opt.moments(i).1))).collect();
    let act = unit_activity(&m, &batches_on(cfg.vocab, dev)).unwrap();
    let rev = revive(&vm, &m, &act, 0.0, 9, Some(&masks), None).unwrap();
    assert!(rev.report.lif[0] >= 1 && rev.report.mlp[1] >= 1, "{:?}", rev.report);
    assert!(zero_moments(&opt, &rev).unwrap() > 0);
    let places: std::collections::HashMap<&str, &Vec<bool>> = rev.places.iter().map(|(n, p)| (n.as_str(), p)).collect();
    for (i, p) in opt.params().iter().enumerate() {
        let (m1, v1) = (bits(opt.moments(i).0), bits(opt.moments(i).1));
        for j in 0..m1.len() {
            match places.get(p.name.as_str()) {
                Some(pl) if pl[j] => assert!(f32::from_bits(m1[j]) == 0.0 && f32::from_bits(v1[j]) == 0.0, "{} [{j}] revived", p.name),
                _ => assert!(m1[j] == before[i].0[j] && v1[j] == before[i].1[j], "{} [{j}] untouched", p.name),
            }
        }
    }
    // H6: values only at live places of the revived column; dead places still 0; masks and live counts unchanged
    let w1 = get(&vm, "b1.mlp.w1");
    let mk = &masks["b1.mlp.w1"];
    let c5: Vec<usize> = (0..d).map(|r| r * h + 5).collect();
    assert!(c5.iter().any(|&i| mk[i] && w1[i] != 0.0));
    assert!(c5.iter().all(|&i| mk[i] || w1[i] == 0.0));
    assert!(w1.iter().zip(mk).all(|(&x, &live)| live || x == 0.0));
    let after = rw.host_masks().unwrap();
    assert_eq!(rw.names().iter().map(|n| after[n].iter().filter(|&&x| x).count()).collect::<Vec<_>>(), live0);
}

#[test]
fn redo_zeroes_moments_and_respects_masks() {
    moments_case(&CPU);
}

#[test]
#[ignore]
fn redo_zeroes_moments_and_respects_masks_gpu() {
    moments_case(&Device::new_cuda(0).expect("no CUDA device"));
}

#[test]
fn redo_flag() {
    assert_eq!(RedoCfg::parse("0.1,500,4").unwrap(), RedoCfg { tau: 0.1, every: 500, batches: 4, reset: false });
    assert_eq!(RedoCfg::parse("0.1,500,4,reset").unwrap(), RedoCfg { tau: 0.1, every: 500, batches: 4, reset: true });
    assert_eq!(RedoCfg::parse("0.1,500,4,reset").unwrap().signature(), " redo=0.1,500,4,reset");
    assert_eq!(RedoCfg::parse("0,1,1").unwrap().signature(), " redo=0,1,1");
    for bad in ["1,10,2", "-0.1,10,2", "0.1,0,2", "0.1,10,0", "0.1,10", "0.1,10,2,3", "0.1,10,2,reset,1", "a,b,c"] {
        assert!(RedoCfg::parse(bad).is_err(), "{bad}");
    }
}

/// A9: every token reads k slots; the counts are off again afterwards; no product keys: an error.
#[test]
fn slot_counts_count_every_read() {
    let ar = Arch::from_flags("--pk-mem 16 --pk-topk 2").unwrap();
    let (_, m) = build_awake(d_cfg().with_arch(ar).unwrap(), 4);
    assert_eq!(pk_values_name(&m).as_deref(), Some("b1.pk.values"));
    let c = slot_counts(&m, &batches(256)).unwrap();
    assert_eq!(c.len(), 16);
    assert_eq!(c.iter().sum::<u64>(), 2 * 2 * 32 * 2);
    let pk = m.ext.as_ref().unwrap().mem.pkm.as_ref().unwrap();
    assert!(pk.counts.borrow().is_none());
    let (_, plain) = build_awake(d_cfg(), 4);
    assert!(pk_values_name(&plain).is_none());
    assert!(slot_counts(&plain, &batches(256)).is_err());
}
