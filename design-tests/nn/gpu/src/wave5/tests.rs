// TESTING / DESIGN ONLY — tests of src/wave5.rs (v42 Wave 5: the model-level brain parts as block flags), CPU only.

use super::*;
use crate::arch::Arch;
use crate::ckpt::{load_auto, write_arch_beside};
use crate::model::{Config, Kind, Model};
use crate::optim::{AdamW, Param};
use crate::state::tests::{build_awake, in_pieces, max_diff, one_pass, tokens};
use crate::state::State;
use crate::testkit::{fix_weights, scratch, unit_noise};
use candle_core::Device;
use candle_nn::{VarBuilder, VarMap};

const CPU: Device = Device::Cpu;

/// Every flag alone, at the sizes a tiny d32 model allows.
const EACH: [&str; 15] = [
    "--resonate 16",
    "--modulate",
    "--tem 4",
    "--halting 3",
    "--cerebellum 64",
    "--workspace 4",
    "--holo 8",
    "--sync 4",
    "--jepa 0.5",
    "--diffusion 0.5",
    "--critic 0.5",
    "--confidence 0.5",
    "--topo 0.5",
    "--workspace 4 --schema 0.5",
    "--halting 3 --reflex 0.5",
];
/// Every flag at once (the order `flags()` writes them in).
const ALL: &str = "--resonate 16 --modulate --tem 4 --halting 3 --cerebellum 64 --workspace 4 --holo 8 --sync 4 --jepa 0.5 --diffusion 0.5 --critic 0.5 --confidence 0.5 --topo 0.5 --schema 0.5 --reflex 0.5 --fok-push 0.5 --fok-qa";
/// The flags that change the forward pass (the parts with a block state, and A7 and H11 in the block).
const IN_BLOCK: &str = "--resonate 16 --modulate --tem 4 --halting 3 --cerebellum 64 --workspace 4 --holo 8 --sync 4";

fn cfg_with(flags: &str) -> Config {
    Config::with_kind(Kind::D, 258, 32, 2, 64).with_arch(Arch::from_flags(flags).unwrap()).unwrap()
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

/// ids and targets [b, t] from one deterministic stream per row.
fn ids_tg(b: usize, t: usize, seed: u64) -> (Tensor, Tensor) {
    let toks = tokens(b, t + 1, 258, seed);
    let w = t + 1;
    let ids: Vec<u32> = (0..b).flat_map(|r| toks[r * w..r * w + t].to_vec()).collect();
    let tg: Vec<u32> = (0..b).flat_map(|r| toks[r * w + 1..(r + 1) * w].to_vec()).collect();
    (Tensor::from_vec(ids, (b, t), &CPU).unwrap(), Tensor::from_vec(tg, (b, t), &CPU).unwrap())
}

fn gsum(g: &candle_core::backprop::GradStore, v: &candle_core::Var) -> Option<f32> {
    g.get(v.as_tensor()).map(|x| x.abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap())
}

fn stash_empty(m: &Model) -> bool {
    let st = m.ext.as_ref().unwrap().mem.w5.stash.borrow();
    st.xf.is_none() && st.logits.is_none() && st.halt.is_empty() && st.extra.is_empty() && st.ids.is_none() && st.scored.is_none()
}

#[test]
fn flags_round_trip_validate_and_count() {
    let a = Arch::from_flags(ALL).unwrap();
    assert_eq!(a.flags(), ALL);
    assert_eq!(Arch::from_flags(&a.flags()).unwrap(), a);
    assert!(a.needs_path() && a.mem.aux_any());
    for f in EACH {
        let x = Arch::from_flags(f).unwrap();
        assert_eq!(x.flags(), f);
        assert!(!x.mem.is_default() && x.needs_path(), "{f}");
    }
    assert!(!Arch::from_flags("--resonate 16 --modulate --tem 4 --cerebellum 64").unwrap().mem.aux_any());
    assert!(Arch::from_flags("--halting 2").unwrap().mem.aux_any(), "A7 brings its KL");
    let base = Config::with_kind(Kind::D, 258, 32, 2, 64);
    let bad = |f: &str| Arch::from_flags(f).map_or(true, |a| base.with_arch(a).is_err());
    for f in [
        "--resonate 1",
        "--halting 1",
        "--cerebellum 8",
        "--jepa -0.5",
        "--critic inf",
        "--confidence NaN",
        "--fok-qa",
        "--fok-push 0.5",
        "--confidence 0.5 --fok-push -1",
        "--modulate --feedback",
        "--resonate 8 --feedback",
        "--tem 2 --feedback",
        "--halting 2 --feedback",
        "--workspace 2",
        "--topo -1",
        "--topo inf",
        "--workspace 4 --feedback",
        "--holo 1",
        "--holo 8 --feedback",
        "--sync 4 --feedback",
        "--schema 0.5",
    ] {
        assert!(bad(f), "{f} must be refused");
    }
    assert!(Config::with_kind(Kind::S, 258, 32, 2, 64).with_arch(Arch::from_flags("--modulate").unwrap()).is_err(), "model D only");
    // the weight count matches the variables made; flags off make none of the new names
    for f in EACH.iter().copied().chain([ALL, ""]) {
        let c = cfg_with(f);
        let vm = VarMap::new();
        let m = Model::new(c, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
        let have: usize = vm.all_vars().iter().map(|v| v.elem_count()).sum();
        assert_eq!(have, c.weight_counts().0, "{f}: weight count");
        if f.is_empty() {
            assert!(m.ext.is_none());
            assert!(vm.data().lock().unwrap().keys().all(|k| !k.contains("parts.") && !k.starts_with("aux.")));
        }
    }
    // N1's vocabulary tables count as embeddings: the non-embedding count grows by the head's other weights only
    let (d, v) = (32usize, 258usize);
    let (t0, n0) = cfg_with("").weight_counts();
    let (t1, n1) = cfg_with("--diffusion 0.5").weight_counts();
    assert_eq!(t1 - t0, (v + 1) * d + d * v + DIFF_K * d + 12 * d * d);
    assert_eq!(n1 - n0, DIFF_K * d + 12 * d * d);
}

/// Every flag starts as today's D (the same weights copied in): the same logits and the same logged loss, within 1e-6.
/// A7 is the exception, measured: the part's halting bias 4 makes the first pass's share sigmoid(4) = 0.982, not 1.
#[test]
fn every_flag_starts_as_plain_d() {
    let (base_vm, base) = build_awake(Config::with_kind(Kind::D, 258, 32, 2, 64), 3);
    let (b, t) = (2, 48);
    let toks = tokens(b, t, 258, 5);
    let want = one_pass(&base, &toks, b, t);
    let (ids, tg) = ids_tg(2, 40, 9);
    let l0 = base.loss_and_grads_autograd(&ids, &tg, None, false).unwrap().0.to_scalar::<f32>().unwrap();
    let all_exact = "--resonate 16 --modulate --tem 4 --cerebellum 64 --workspace 4 --holo 8 --sync 4 --jepa 0.5 --diffusion 0.5 --critic 0.5 --confidence 0.5 --topo 0.5 --schema 0.5";
    for f in EACH.iter().copied().chain([all_exact]) {
        let (_, m) = twin(cfg_with(f), &base_vm);
        let d = max_diff(&one_pass(&m, &toks, b, t), &want);
        let dl = (m.loss_and_grads_autograd(&ids, &tg, None, false).unwrap().0.to_scalar::<f32>().unwrap() - l0).abs();
        if f.starts_with("--halting") {
            eprintln!("A7 at init vs plain D: logits max diff {d:.3e}, loss diff {dl:.3e} (loss {l0})");
            // measured (d32 tiny model): logits 0.129, loss 4.8e-4 on 5.58
            assert!(d > 0.0 && d < 0.5 && dl < 0.01, "{f}: logits {d}, loss {dl}");
        } else {
            assert!(d < 1e-6 && dl < 1e-6, "{f}: logits {d}, loss {dl}");
        }
    }
}

/// Every flag: finite gradients for every weight at step 0, the new parts' weights moved by the loss, the logged CE is
/// the plain one (total > CE only with an auxiliary loss), the stash is taken by every step and never filled by an
/// evaluation forward, and the CE drops over six AdamW steps.
#[test]
fn every_flag_learns_with_finite_gradients() {
    let (ids, tg) = ids_tg(2, 40, 6);
    let moved = |f: &str| -> Vec<&'static str> {
        let mut v = Vec::new();
        for (flag, names) in [
            ("--resonate", vec!["b0.parts.res.mix", "b1.parts.res.mix"]),
            ("--modulate", vec!["b0.parts.mod.gains", "b0.parts.mood.w"]),
            ("--tem", vec!["b0.parts.tem.wo"]),
            ("--halting", vec!["b0.parts.halt.w", "b0.parts.halt.b"]),
            ("--cerebellum", vec!["b0.parts.cb.readout"]),
            ("--workspace", vec!["b1.parts.ws.wo"]),
            ("--schema", vec!["b1.parts.schema.w"]),
            ("--holo", vec!["b0.parts.holo.wo"]),
            ("--sync", vec!["b0.parts.sync.wo"]),
            ("--jepa", vec!["aux.jepa"]),
            ("--diffusion", vec!["aux.diff"]),
            ("--critic", vec!["aux.critic"]),
            ("--confidence", vec!["aux.conf"]),
        ] {
            if f.split_whitespace().any(|w| w == flag) {
                v.extend(names);
            }
        }
        v
    };
    for f in EACH.iter().copied().chain([ALL]) {
        let cfg = cfg_with(f);
        let vm = VarMap::new();
        let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
        fix_weights(&vm, &cfg, 21);
        m.set_track_activity(false);
        let mut params: Vec<Param> =
            vm.data().lock().unwrap().iter().map(|(k, v)| Param { name: k.clone(), var: v.clone(), decay: Model::decays_with(k, true) }).collect();
        params.sort_by(|a, b| a.name.cmp(&b.name));
        let mut opt = AdamW::new(params, 0.0).unwrap();
        let aux = cfg.arch.mem.aux_any();
        let (mut first, mut last) = (f32::NAN, f32::NAN);
        for step in 0..6 {
            let (total, ce, _) = m.loss_aux(&ids, &tg, 0.0).unwrap();
            let (tv, cv) = (total.to_scalar::<f32>().unwrap(), ce.to_scalar::<f32>().unwrap());
            assert!(tv.is_finite() && cv.is_finite(), "{f}: total {tv}, ce {cv}");
            if aux {
                assert!(tv > cv, "{f}: total {tv} <= ce {cv}");
            } else {
                assert_eq!(tv, cv, "{f}");
            }
            assert!(stash_empty(&m), "{f}: the stash is taken by the step");
            let g = total.backward().unwrap();
            if step == 0 {
                first = cv;
                for p in opt.params() {
                    let s = gsum(&g, &p.var).unwrap_or_else(|| panic!("{f}: {} has no gradient", p.name));
                    assert!(s.is_finite(), "{f}: {} gradient {s}", p.name);
                }
                let data = vm.data().lock().unwrap();
                for pre in moved(f) {
                    let s: f32 = data.iter().filter(|(k, _)| k.starts_with(pre)).map(|(_, v)| gsum(&g, v).unwrap()).sum();
                    assert!(s > 0.0, "{f}: {pre} gets no gradient at init");
                }
            }
            last = cv;
            opt.step(&g, 3e-3, 1.0).unwrap();
        }
        assert!(last < first, "{f}: ce {first} -> {last}");
        let _ = m.forward(&ids).unwrap();
        assert!(stash_empty(&m), "{f}: an evaluation forward keeps nothing");
    }
}

/// The logged loss of a step with an auxiliary loss is the step's plain CE; C8 never reaches the trunk (its input is
/// detached); B1, N1, B2 and A7's KL do. A signed reward step (`raw`) gets none of them.
#[test]
fn auxiliary_losses_leave_the_logged_loss_and_reach_where_they_should() {
    let (ids, tg) = ids_tg(2, 40, 12);
    for (f, reaches) in [("--confidence 0.5", false), ("--jepa 0.5", true), ("--diffusion 0.5", true), ("--critic 0.5", true), ("--halting 3", true), ("--halting 3 --reflex 0.5", true), ("--confidence 0.5 --fok-push 0.5", false)] {
        let (vm, m) = build_awake(cfg_with(f), 5);
        let (l_on, _, g_on) = m.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
        assert!(stash_empty(&m), "{f}");
        let (l_off, _, g_off) = m.loss_and_grads_autograd(&ids, &tg, None, true).unwrap();
        assert!(stash_empty(&m), "{f}");
        assert_eq!(l_on.to_scalar::<f32>().unwrap(), l_off.to_scalar::<f32>().unwrap(), "{f}: the logged loss is the plain CE");
        let (plain, _) = m.loss(&ids, &tg).unwrap();
        assert!((plain.to_scalar::<f32>().unwrap() - l_on.to_scalar::<f32>().unwrap()).abs() < 1e-5, "{f}");
        let data = vm.data().lock().unwrap();
        let mut trunk_diff = 0f32;
        for (k, v) in data.iter() {
            if k.starts_with("aux.") || k.contains("parts.halt") {
                assert!(gsum(&g_on, v).is_some_and(|s| s > 0.0), "{f}: {k} learns from its loss");
                continue;
            }
            let (a, b) = (g_on.get(v.as_tensor()).unwrap(), g_off.get(v.as_tensor()).unwrap());
            trunk_diff = trunk_diff.max(max_diff(a, b));
        }
        if reaches {
            assert!(trunk_diff > 1e-6, "{f}: its loss shapes the stream ({trunk_diff})");
        } else {
            assert!(trunk_diff < 1e-7, "{f}: the probe's loss must not reach the trunk ({trunk_diff})");
        }
    }
}

/// How far a read in pieces may differ from one pass. Every part but B10 agrees within 2e-6 (measured on this d32
/// model: L5 1.9e-6, D2 4.0e-7, the rest 1.8e-7 or less). B10 reads by a softmax of 30·g·g', which magnifies f32
/// rounding: alone it differed by up to 5.1e-5, and with the other in-block parts by up to 1.5e-4. With AVA_TEM_BETA=2
/// those fell to 2.8e-7 and 6.1e-7, so the gap is the sharp read's rounding, not a state that fails to carry.
fn pieces_tol(f: &str) -> f32 {
    if f.contains("--tem") {
        5e-4
    } else {
        1e-5
    }
}

/// The stateful flags (and A7, H11 in the block) read in pieces as in one pass: a fresh state exactly, pieces within
/// pieces_tol; the token-by-token stream and a batched feed too.
#[test]
fn stateful_flags_read_in_pieces_and_stream() {
    for f in ["--resonate 16", "--modulate", "--tem 4", "--halting 3", "--cerebellum 64", "--workspace 4", "--holo 8", "--sync 4", "--resonate 16 --modulate --tem 4 --halting 3 --cerebellum 64", IN_BLOCK, ALL] {
        let (_, m) = build_awake(cfg_with(f), 14);
        let (b, t) = (2, 64);
        let toks = tokens(b, t, 258, 71);
        let want = one_pass(&m, &toks, b, t);
        let mut st = m.state(b).unwrap();
        assert_eq!(max_diff(&in_pieces(&m, &mut st, &toks, b, t, &[t]), &want), 0.0, "{f} fresh");
        for pieces in [vec![20, 44], vec![9, 40, 15], vec![1, 1, 62]] {
            let mut st = m.state(b).unwrap();
            let d = max_diff(&in_pieces(&m, &mut st, &toks, b, t, &pieces), &want);
            assert!(d < pieces_tol(f), "{f} {pieces:?}: {d}");
        }
        let mut s = m.stream().unwrap();
        let mut last = None;
        for &tok in &toks[..t] {
            last = Some(m.step(&mut s, tok).unwrap());
        }
        let d = max_diff(&last.unwrap(), &want.get(0).unwrap().get(t - 1).unwrap());
        assert!(d < pieces_tol(f), "{f} stream: {d}");
        let mut s = m.stream().unwrap();
        let fed = m.feed(&mut s, &toks[..t]).unwrap();
        let d = max_diff(&fed, &want.get(0).unwrap().get(t - 1).unwrap());
        assert!(d < pieces_tol(f), "{f} feed: {d}");
    }
}

/// The new state (E7's resonators, B10's store and phase, B8/F13's statistics and shadow state) copies rows, joins
/// rows of different lengths, and saves and loads.
#[test]
fn stateful_rows_copy_join_save_and_load() {
    let (_, m) = build_awake(cfg_with(IN_BLOCK), 8);
    let (b, t) = (3, 30);
    let toks = tokens(b, t, 258, 31);
    // rows are independent
    let mut st = m.state(b).unwrap();
    let all = in_pieces(&m, &mut st, &toks, b, t, &[20, 10]);
    for r in 0..b {
        let one = one_pass(&m, &toks[r * t..(r + 1) * t], 1, t);
        let d = max_diff(&all.narrow(0, r, 1).unwrap(), &one);
        assert!(d < pieces_tol(IN_BLOCK), "row {r}: {d}");
    }
    // one prompt's state copied to two rows
    let (p, c) = (18, 12);
    let prompt = &toks[..p];
    let mut st = m.state(1).unwrap();
    in_pieces(&m, &mut st, prompt, 1, p, &[p]);
    assert!(st.modc.is_some() && st.shadow.is_some());
    let mut two = st.select(&[0, 0]).unwrap();
    let conts: Vec<u32> = [&toks[t..t + c], &toks[2 * t..2 * t + c]].concat();
    let got = in_pieces(&m, &mut two, &conts, 2, c, &[c]);
    for r in 0..2 {
        let whole: Vec<u32> = prompt.iter().chain(&conts[r * c..(r + 1) * c]).copied().collect();
        let want = one_pass(&m, &whole, 1, p + c).narrow(1, p, c).unwrap();
        assert!(max_diff(&got.narrow(0, r, 1).unwrap(), &want) < 1e-4, "copy {r}");
    }
    // rows of different lengths join
    let (la, lb, more) = (10, 21, 6);
    let ta = tokens(1, la + more, 258, 41);
    let tb = tokens(1, lb + more, 258, 42);
    let mut sa = m.state(1).unwrap();
    in_pieces(&m, &mut sa, &ta[..la], 1, la, &[la]);
    let mut sb = m.state(1).unwrap();
    in_pieces(&m, &mut sb, &tb[..lb], 1, lb, &[lb]);
    let mut both = State::cat(&[sa, sb]).unwrap();
    let next: Vec<u32> = ta[la..].iter().chain(&tb[lb..]).copied().collect();
    let got = in_pieces(&m, &mut both, &next, 2, more, &[more]);
    let wa = one_pass(&m, &ta, 1, la + more).narrow(1, la, more).unwrap();
    let wb = one_pass(&m, &tb, 1, lb + more).narrow(1, lb, more).unwrap();
    assert!(max_diff(&got.narrow(0, 0, 1).unwrap(), &wa) < 1e-4, "short row");
    assert!(max_diff(&got.narrow(0, 1, 1).unwrap(), &wb) < 1e-4, "long row");
    // saved and loaded, it reads on exactly as the state in memory
    let toks2 = tokens(2, 30, 258, 51);
    let mut st = m.state(2).unwrap();
    let first: Vec<u32> = (0..2).flat_map(|r| toks2[r * 30..r * 30 + 20].to_vec()).collect();
    in_pieces(&m, &mut st, &first, 2, 20, &[20]);
    let dir = scratch("wave5_state");
    let path = dir.join("s.safetensors");
    st.save(&m, &path).unwrap();
    let mut back = State::load(&m, &path, &CPU).unwrap();
    assert_eq!(back.seen, st.seen);
    assert_eq!(back.modc.as_ref().unwrap().rows, st.modc.as_ref().unwrap().rows);
    let rest: Vec<u32> = (0..2).flat_map(|r| toks2[r * 30 + 20..r * 30 + 30].to_vec()).collect();
    let a = in_pieces(&m, &mut st, &rest, 2, 10, &[10]);
    let b2 = in_pieces(&m, &mut back, &rest, 2, 10, &[10]);
    assert_eq!(max_diff(&a, &b2), 0.0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every flag rebuilds from the arch.txt beside the model (the fixed tensors, E7's base frequencies and H11's
/// expansion, are rebuilt from their sizes and seeds): the same logits and the weight count.
#[test]
fn flags_load_from_the_arch_file_beside_the_model() {
    let ar = Arch::from_flags(ALL).unwrap();
    let cfg = Config::with_kind(Kind::D, 258, 32, 2, 64).with_arch(ar).unwrap();
    let (vm, m) = build_awake(cfg, 11);
    let toks = tokens(2, 24, 258, 5);
    let want = one_pass(&m, &toks, 2, 24);
    let dir = scratch("wave5_ckpt");
    let path = dir.join("brain.safetensors");
    vm.save(&path).unwrap();
    assert!(load_auto(&path, Kind::D, 64, &CPU).is_err(), "no arch.txt: refused");
    write_arch_beside(&path, &ar).unwrap();
    let (c2, vm2, m2) = load_auto(&path, Kind::D, 64, &CPU).unwrap();
    assert_eq!(c2.arch.signature(), ar.signature());
    assert_eq!(c2.arch.flags(), ALL);
    let have: usize = vm2.all_vars().iter().map(|v| v.elem_count()).sum();
    assert_eq!(have, c2.weight_counts().0);
    assert_eq!(max_diff(&want, &one_pass(&m2, &toks, 2, 24)), 0.0);
    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------------------------------------- the pieces against the parts

/// B8/F13: a row's running statistics are the part's zscores and mood exactly (valence -z, |z|); a text's first token
/// (no surprise) scores 0 and moves nothing; the 7-number form round-trips; zero gains leave the gates exactly as given.
#[test]
fn mod_rows_are_the_parts_zscores_and_mood() {
    let s: Vec<f32> = unit_noise(25, 3).iter().map(|z| 3.0 + 2.0 * z).collect();
    let (z, zbar) = ava_parts::modulate::zscores(std::slice::from_ref(&s), MOD_RHO);
    let absz: Vec<f32> = z[0].iter().map(|x| x.abs()).collect();
    let neg: Vec<f32> = z[0].iter().map(|x| -x).collect();
    let (mood, arousal) = ava_parts::modulate::mood(&[neg], &[absz], MOOD_RHO);
    let mut row = ModRow::default();
    assert_eq!(row.step(None), (0.0, 0.0, 0.0, 0.0));
    assert_eq!(row, ModRow::default());
    for i in 0..s.len() {
        let got = row.step(Some(s[i]));
        assert_eq!(got, (z[0][i], zbar[0][i], mood[0][i], arousal[0][i]), "token {i}");
    }
    let back = ModRow::from_slice(&row.to_vec());
    assert_eq!(back, ModRow { has_prev: false, ..row });
    let mut on = row;
    on.has_prev = true;
    assert_eq!(ModRow::from_slice(&on.to_vec()), on);
    // gains 0: write 1, reset 0 (D's write strength is 2·sigmoid, inside the clamp's [0, 2])
    let la = Tensor::from_vec(unit_noise(2 * 2 * 5 * 3, 4), (2, 2, 5, 3), &CPU).unwrap().affine(0.5, -1.0).unwrap();
    let beta = candle_nn::ops::sigmoid(&Tensor::from_vec(unit_noise(2 * 2 * 5 * 3, 5), (2, 2, 5, 3), &CPU).unwrap().affine(3.0, 0.0).unwrap()).unwrap().affine(2.0, 0.0).unwrap();
    let (la2, b2) = mod_gates(la.clone(), beta.clone(), &Tensor::ones((2, 5), DType::F32, &CPU).unwrap(), &Tensor::zeros((2, 5), DType::F32, &CPU).unwrap()).unwrap();
    assert_eq!(max_diff(&la, &la2), 0.0);
    assert_eq!(max_diff(&beta, &b2), 0.0);
}

/// B8/F13 signals read in pieces with the carried statistics and last logits equal one read.
#[test]
fn mod_signals_carry_across_pieces() {
    let (b, t, v) = (2, 12, 10);
    let logits = Tensor::from_vec(unit_noise(b * t * v, 7), (b * t, v), &CPU).unwrap().affine(2.0, 0.0).unwrap();
    let ids: Vec<u32> = unit_noise(b * t, 8).iter().map(|z| ((z + 1.0) * 4.99) as u32 % v as u32).collect();
    let ids = Tensor::from_vec(ids, (b, t), &CPU).unwrap();
    let whole = mod_signals(&logits, &ids, None).unwrap();
    let l3 = logits.reshape((b, t, v)).unwrap();
    let mut carry = ModCarry { rows: vec![ModRow::default(); b], prev: Tensor::zeros((b, v), DType::F32, &CPU).unwrap() };
    let mut parts: Vec<ModSig> = Vec::new();
    let mut p0 = 0;
    for p in [5usize, 1, 6] {
        let lp = l3.narrow(1, p0, p).unwrap().reshape((b * p, v)).unwrap();
        parts.push(mod_signals(&lp, &ids.narrow(1, p0, p).unwrap(), Some(&mut carry)).unwrap());
        p0 += p;
    }
    let cat = |f: fn(&ModSig) -> &Tensor| Tensor::cat(&parts.iter().map(f).collect::<Vec<_>>(), 1).unwrap();
    assert!(max_diff(&cat(|s| &s.z), &whole.z) < 1e-6);
    assert!(max_diff(&cat(|s| &s.zbar), &whole.zbar) < 1e-6);
    assert!(max_diff(&cat(|s| &s.mood), &whole.mood) < 1e-6);
    assert!(max_diff(&cat(|s| &s.arousal), &whole.arousal) < 1e-6);
    // the first token scores 0; the second token's surprise is row 0's cross-entropy against it
    let z: Vec<Vec<f32>> = whole.z.to_vec2().unwrap();
    assert!(z.iter().all(|r| r[0] == 0.0 && r[1] == 0.0), "the z-score starts at its first surprise");
    let m: Vec<Vec<f32>> = whole.mood.to_vec2().unwrap();
    // v42 (Amendment 15): valence is −z, and the first surprise's z is 0, so mood moves only from the second one on
    assert_eq!(m[0][1], 0.0);
    assert!((m[0][2] - (1.0 - MOOD_RHO) * -z[0][2]).abs() < 1e-6 && z[0][2] != 0.0);
}

/// B2: the λ-return targets are the part's lambda_returns with v[T] = v[T − 1]; the matrix is kept per length.
#[test]
fn lambda_targets_are_the_parts_lambda_returns() {
    let (b, t) = (2, 9);
    let r = Tensor::from_vec(unit_noise(b * t, 6), (b, t), &CPU).unwrap();
    let v = Tensor::from_vec(unit_noise(b * t, 7), (b, t), &CPU).unwrap();
    let cache = RefCell::new(None);
    let got = lambda_targets(&r, &v, CRITIC_GAMMA, CRITIC_LAMBDA, Some(&cache)).unwrap();
    let (gv, rv, vv): (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) = (got.to_vec2().unwrap(), r.to_vec2().unwrap(), v.to_vec2().unwrap());
    for i in 0..b {
        let mut ve = vv[i].clone();
        ve.push(vv[i][t - 1]);
        let want = ava_parts::critic::lambda_returns(&rv[i], &ve, CRITIC_GAMMA as f32, CRITIC_LAMBDA as f32);
        for j in 0..t {
            assert!((gv[i][j] - want[j]).abs() < 1e-5, "row {i} token {j}: {} vs {}", gv[i][j], want[j]);
        }
    }
    assert_eq!(cache.borrow().as_ref().unwrap().0, t);
    assert_eq!(max_diff(&lambda_targets(&r, &v, CRITIC_GAMMA, CRITIC_LAMBDA, Some(&cache)).unwrap(), &got), 0.0);
    // one token: G = r + γ v
    let one = lambda_targets(&r.narrow(1, 0, 1).unwrap(), &v.narrow(1, 0, 1).unwrap(), CRITIC_GAMMA, CRITIC_LAMBDA, None).unwrap();
    let want = r.narrow(1, 0, 1).unwrap().add(&v.narrow(1, 0, 1).unwrap().affine(CRITIC_GAMMA, 0.0).unwrap()).unwrap();
    assert!(max_diff(&one, &want) < 1e-6);
}

/// B10: with no store the structure read is the part's tem::recall on the codes of the path-integrated moves (the
/// first token reads nothing); with the store and phase carried, pieces read as one.
#[test]
fn tem_read_is_the_parts_recall_and_reads_in_pieces() {
    let (d, k, t) = (8, 3, 10);
    let wth = Tensor::from_vec(unit_noise(d * k, 1), (d, k), &CPU).unwrap();
    let tp = Tem { norm: Tensor::ones(d, DType::F32, &CPU).unwrap(), wth: wth.clone(), wo: Tensor::eye(d, DType::F32, &CPU).unwrap(), k };
    let h = Tensor::from_vec(unit_noise(t * d, 2), (t, d), &CPU).unwrap();
    let got = tem_read(&h, &tp, 1, t, None, None).unwrap();
    let th: Vec<Vec<f32>> = h.matmul(&wth).unwrap().to_vec2().unwrap();
    let (mut phi, mut g) = (vec![0f32; k], Vec::new());
    let sk = (k as f32).sqrt();
    for row in th.iter() {
        g.extend(phi.iter().map(|p| p.cos() / sk));
        g.extend(phi.iter().map(|p| p.sin() / sk));
        for j in 0..k {
            phi[j] += row[j];
        }
    }
    let g = Tensor::from_vec(g, (t, 2 * k), &CPU).unwrap();
    let want = ava_parts::tem::recall(&g, &h, TEM_BETA).unwrap();
    let dd = max_diff(&got, &want);
    assert!(dd < 1e-4, "recall: {dd}");
    assert_eq!(max_diff(&got.narrow(0, 0, 1).unwrap(), &Tensor::zeros((1, d), DType::F32, &CPU).unwrap()), 0.0);
    // pieces with the store and phase carried
    let b = 2;
    let h2 = Tensor::from_vec(unit_noise(b * t * d, 3), (b * t, d), &CPU).unwrap();
    let whole = tem_read(&h2, &tp, b, t, None, None).unwrap();
    let h3 = h2.reshape((b, t, d)).unwrap();
    let mut store: Option<Kv> = None;
    let mut phase = Tensor::zeros((b, k), DType::F32, &CPU).unwrap();
    let (mut outs, mut p0) = (Vec::new(), 0);
    for p in [4usize, 1, 5] {
        let hp = h3.narrow(1, p0, p).unwrap().contiguous().unwrap().reshape((b * p, d)).unwrap();
        outs.push(tem_read(&hp, &tp, b, p, Some(&mut store), Some(&mut phase)).unwrap().reshape((b, p, d)).unwrap());
        p0 += p;
    }
    let got = Tensor::cat(&outs, 1).unwrap().reshape((b * t, d)).unwrap();
    let dd = max_diff(&got, &whole);
    assert!(dd < 1e-5, "pieces: {dd}");
    assert_eq!(store.unwrap().valid, vec![t; b]);
}

/// H11: the block's correction is the part's sparse code (the model's kWTA mask in place of the part's sort) through
/// the read-out, on the normed stream; k units per row.
#[test]
fn cerebellum_code_is_the_parts_code() {
    let (d, g) = (16, 64);
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &CPU);
    let cb = Cerebellum::new(vb.pp("cb"), d, g, CB_FAN_IN, g / CB_SPARSITY, CB_SEED).unwrap();
    let ro = Tensor::from_vec(unit_noise(g * d, 4), (g, d), &CPU).unwrap();
    vm.data().lock().unwrap()["cb.readout"].set(&ro).unwrap();
    let ones = Tensor::ones(d, DType::F32, &CPU).unwrap();
    let x = Tensor::from_vec(unit_noise(5 * d, 5), (5, d), &CPU).unwrap().affine(3.0, 0.0).unwrap();
    let got = cb_add(&cb, &ones, &x).unwrap().sub(&x).unwrap();
    let code = cb.code(&rmsnorm(&x, &ones).unwrap()).unwrap();
    let want = code.matmul(&ro).unwrap();
    assert!(max_diff(&got, &want) < 1e-5);
    let nz: Vec<Vec<f32>> = code.to_vec2().unwrap();
    assert!(nz.iter().all(|r| r.iter().filter(|&&x| x > 0.0).count() <= g / CB_SPARSITY));
}

/// N1: the blocks sit at K − 1, K − 1 + stride, .. in every row; a block with an unscored target is left out.
#[test]
fn diffusion_blocks_layout() {
    let t = 40;
    let mut tg: Vec<Vec<u32>> = (0..2).map(|r| (0..t as u32).map(|i| (i * 7 + r) % 250).collect()).collect();
    let (at, blocks) = diffusion_blocks(&tg, t, 258);
    let p1 = (DIFF_K - 1) as u32;
    let p2 = p1 + DIFF_STRIDE as u32;
    assert_eq!(at, vec![p1, p2, t as u32 + p1, t as u32 + p2]);
    assert_eq!(blocks[1], tg[0][p2 as usize..p2 as usize + DIFF_K].to_vec());
    tg[1][p2 as usize + 3] = crate::IGNORE;
    let (at, blocks) = diffusion_blocks(&tg, t, 258);
    assert_eq!(at, vec![p1, p2, t as u32 + p1]);
    assert_eq!(blocks.len(), 3);
    assert!(diffusion_blocks(&tg, DIFF_K - 1, 258).0.is_empty(), "a window shorter than a block has none");
}

/// v42: B10's fused training read (tem_read_fast) = tem_read's own (reached here with a zero phase, which adds 0 to the
/// codes): the output and the gradients to h and both weights, within 1e-4 of the largest. CUDA only (--ignored).
#[test]
#[ignore]
fn tem_fast_read_matches_the_plain_read_gpu() {
    let dev = Device::new_cuda(0).expect("no CUDA device");
    let (b, t, d, k) = (2usize, 100usize, 64usize, 16usize);
    let wave = |n: usize, s: f32, salt: u64| {
        let v: Vec<f32> = (0..n)
            .map(|j| {
                let x = ((j as u64 * 2654435761 + salt * 7919 + 13) % 10007) as f32 / 10007.0;
                ((x * 6.2831853).sin() * 1.7 + (x * 51.0).cos()) * s
            })
            .collect();
        Tensor::from_vec(v, n, &dev).unwrap()
    };
    let h = candle_core::Var::from_tensor(&wave(b * t * d, 1.0, 1).reshape((b * t, d)).unwrap()).unwrap();
    let wth = candle_core::Var::from_tensor(&wave(d * k, 0.3, 2).reshape((d, k)).unwrap()).unwrap();
    let wo = candle_core::Var::from_tensor(&wave(d * d, 0.2, 3).reshape((d, d)).unwrap()).unwrap();
    let w = wave(b * t * d, 1.0, 4).reshape((b * t, d)).unwrap();
    let run = |fast: bool| {
        let tp = Tem { norm: Tensor::ones(d, DType::F32, &dev).unwrap(), wth: wth.as_tensor().clone(), wo: wo.as_tensor().clone(), k };
        let y = if fast {
            tem_read_fast(h.as_tensor(), &tp, b, t).unwrap()
        } else {
            let mut ph = Tensor::zeros((b, k), DType::F32, &dev).unwrap();
            tem_read(h.as_tensor(), &tp, b, t, None, Some(&mut ph)).unwrap()
        };
        let g = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        (y, [h.as_tensor(), wth.as_tensor(), wo.as_tensor()].map(|v| g.get(v).unwrap().clone()))
    };
    let (y1, g1) = run(true);
    let (y2, g2) = run(false);
    let rel = |a: &Tensor, b: &Tensor| {
        let m = |x: &Tensor| x.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        m(&a.sub(b).unwrap()) / m(b).max(1e-6)
    };
    assert!(rel(&y1, &y2) < 1e-4, "y rel err {:.2e}", rel(&y1, &y2));
    for (j, name) in ["h", "wth", "wo"].iter().enumerate() {
        let e = rel(&g1[j], &g2[j]);
        assert!(e < 1e-4, "grad {name} rel err {e:.2e}");
    }
    eprintln!("tem_read_fast = tem_read (y {:.1e})", rel(&y1, &y2));
}

/// D2 and L5 on CUDA give the CPU's output and gradients (to h and to every weight) within 1e-4 of the largest, over a
/// text longer than two of D2's chunks. CUDA only (--ignored).
#[test]
#[ignore]
fn holo_and_sync_on_cuda_match_the_cpu_gpu() {
    let cuda = Device::new_cuda(0).expect("no CUDA device");
    let (b, t, d) = (2usize, ava_parts::holo::HOLO_CHUNK * 2 + 40, 32usize);
    let rel = |a: &Tensor, b: &Tensor| {
        let a = a.to_device(&CPU).unwrap();
        let b = b.to_device(&CPU).unwrap();
        let m = |x: &Tensor| x.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        m(&a.sub(&b).unwrap()) / m(&b).max(1e-6)
    };
    let h0 = Tensor::from_vec(unit_noise(b * t * d, 21), (b, t, d), &CPU).unwrap();
    let w0 = Tensor::from_vec(unit_noise(b * t * d, 22), (b, t, d), &CPU).unwrap();
    for part in ["holo", "sync"] {
        // the same weights on both devices: made on the CPU (read-outs set to noise), then copied
        let cpu_vm = VarMap::new();
        let make = |vm: &VarMap, dev: &Device| -> Box<dyn Fn(&Tensor) -> Tensor> {
            let vb = VarBuilder::from_varmap(vm, DType::F32, dev);
            if part == "holo" {
                let hm = ava_parts::holo::HoloMem::new(vb, d, 16).unwrap();
                Box::new(move |h: &Tensor| hm.forward(h).unwrap())
            } else {
                let s = ava_parts::sync::Oscillators::new(vb, d, 8).unwrap();
                Box::new(move |h: &Tensor| s.forward(h).unwrap())
            }
        };
        let f_cpu = make(&cpu_vm, &CPU);
        let names: Vec<String> = cpu_vm.data().lock().unwrap().keys().cloned().collect();
        for (k, n) in names.iter().enumerate() {
            let v = cpu_vm.data().lock().unwrap()[n].clone();
            v.set(&(Tensor::from_vec(unit_noise(v.elem_count(), 30 + k as u64), v.dims(), &CPU).unwrap() * 0.3).unwrap()).unwrap();
        }
        let gpu_vm = VarMap::new();
        let f_gpu = make(&gpu_vm, &cuda);
        for n in &names {
            let src = cpu_vm.data().lock().unwrap()[n].as_tensor().to_device(&cuda).unwrap();
            gpu_vm.data().lock().unwrap()[n].set(&src).unwrap();
        }
        let run = |f: &dyn Fn(&Tensor) -> Tensor, vm: &VarMap, dev: &Device| {
            let h = candle_core::Var::from_tensor(&h0.to_device(dev).unwrap()).unwrap();
            let y = f(h.as_tensor());
            let g = y.mul(&w0.to_device(dev).unwrap()).unwrap().sum_all().unwrap().backward().unwrap();
            let mut gs = vec![g.get(h.as_tensor()).unwrap().clone()];
            for n in &names {
                gs.push(g.get(vm.data().lock().unwrap()[n].as_tensor()).unwrap().clone());
            }
            (y, gs)
        };
        let (y1, g1) = run(&*f_gpu, &gpu_vm, &cuda);
        let (y2, g2) = run(&*f_cpu, &cpu_vm, &CPU);
        assert!(rel(&y1, &y2) < 1e-4, "{part}: y rel err {:.2e}", rel(&y1, &y2));
        for (j, name) in std::iter::once(&"h".to_string()).chain(&names).enumerate() {
            let e = rel(&g1[j], &g2[j]);
            assert!(e < 1e-4, "{part}: grad {name} rel err {e:.2e}");
        }
        eprintln!("{part} on CUDA = CPU (y {:.1e})", rel(&y1, &y2));
    }
}

/// v42 (src/arch/ckpt.rs): the block recompute gives the step's loss and every weight's gradient as the plain backward
/// does, with every part on (one recompute per block); a backward whose block gradients nobody merged stops the next
/// training forward.
#[test]
fn block_recompute_matches_the_plain_backward() {
    use crate::arch::ckpt;
    let every = format!(
        "{ALL} --dendrites 4 --stp --delays 4 --wm 8 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1"
    );
    let (ids, tg) = ids_tg(2, 40, 7);
    for f in [ALL.to_string(), every] {
        let cfg = cfg_with(&f);
        let vm = VarMap::new();
        let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
        fix_weights(&vm, &cfg, 5);
        m.set_track_activity(false);
        let run = |on: bool| {
            ckpt::force(Some(on));
            let r0 = ckpt::runs();
            let (total, _, _) = m.loss_aux(&ids, &tg, 0.0).unwrap();
            let mut g = total.backward().unwrap();
            ckpt::merge(&mut g).unwrap();
            (total.to_scalar::<f32>().unwrap(), g, ckpt::runs() - r0)
        };
        let (l0, g0, u0) = run(false);
        let (l1, g1, u1) = run(true);
        assert_eq!(u0, 0, "{f}");
        // H3b: --reflex runs the text once more in one pass, so its blocks are recomputed once more too
        let fwd = if cfg.arch.mem.reflex > 0.0 { 2 } else { 1 };
        assert_eq!(u1, fwd * cfg.layers, "{f}: one recompute per block per forward");
        assert_eq!(l0, l1, "{f}: the loss");
        let data = vm.data().lock().unwrap();
        let mut worst = (0.0f32, String::new());
        for (k, v) in data.iter() {
            match (g0.get(v.as_tensor()), g1.get(v.as_tensor())) {
                (Some(a), Some(b)) => {
                    let scale = a.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap().max(1e-6);
                    let diff = a.sub(b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap() / scale;
                    if diff > worst.0 {
                        worst = (diff, k.clone());
                    }
                }
                (None, None) => {}
                _ => panic!("{f}: {k} has a gradient in one backward only"),
            }
        }
        drop(data);
        eprintln!("{f}: loss {l0}, worst relative gradient difference {:.2e} ({})", worst.0, worst.1);
        assert!(worst.0 < 1e-4, "{f}: {} differs by {:.2e}", worst.1, worst.0);
        // the autograd step merges by itself
        ckpt::force(Some(true));
        let (la, _, ga) = m.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
        ckpt::force(Some(false));
        let (lb, _, gb) = m.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
        assert_eq!(la.to_scalar::<f32>().unwrap(), lb.to_scalar::<f32>().unwrap(), "{f}");
        for v in vm.data().lock().unwrap().values() {
            assert_eq!(ga.get(v.as_tensor()).is_some(), gb.get(v.as_tensor()).is_some(), "{f}");
        }
        // a backward left unmerged stops the next forward
        ckpt::force(Some(true));
        let (total, _, _) = m.loss_aux(&ids, &tg, 0.0).unwrap();
        let _ = total.backward().unwrap();
        assert!(m.loss_aux(&ids, &tg, 0.0).is_err(), "{f}: an unmerged backward must stop the next forward");
        let mut sink = Tensor::zeros((), DType::F32, &CPU).unwrap().backward().unwrap();
        ckpt::merge(&mut sink).unwrap();
        ckpt::force(None);
    }
}

/// H12: topo_loss is the part's spatial_loss: on all rows when they fit, and on every s-th row when they do not (both
/// strides, the reshape and the index form); the cached pair weights are made once per unit count.
#[test]
fn topo_loss_is_the_parts_spatial_loss() {
    let cfg = cfg_with("--topo 0.5");
    let vm = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
    let w5 = &m.ext.as_ref().unwrap().mem.w5;
    for (n, u) in [(300usize, 50usize), (2048, 40), (2100, 36)] {
        let a = Tensor::from_vec(unit_noise(n * u, 31 + n as u64), (n, u), &CPU).unwrap().relu().unwrap();
        let got = topo_loss(w5, &a).unwrap().to_scalar::<f32>().unwrap();
        let s = n.div_ceil(TOPO_ROWS);
        let idx: Vec<u32> = (0..n).step_by(s).map(|i| i as u32).collect();
        let k = idx.len();
        let r = a.index_select(&Tensor::from_vec(idx, k, &CPU).unwrap(), 0).unwrap();
        let want = ava_parts::topo::spatial_loss(&r, topo_width(u)).unwrap().to_scalar::<f32>().unwrap();
        assert!((got - want).abs() < 1e-4, "n {n}, u {u}: {got} vs the part's {want}");
        assert_eq!(w5.topo.borrow().as_ref().unwrap().u, u);
    }
}

/// B3: the workspace is the part's competing SlotMemory (top WS_TOPK writers), in ws_block only, and its read-out
/// reaches the stream once it is not 0.
#[test]
fn workspace_is_the_parts_competing_slots() {
    let cfg = cfg_with("--workspace 4");
    let vm = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
    let blocks = &m.ext.as_ref().unwrap().mem.w5.blocks;
    for (i, b) in blocks.iter().enumerate() {
        assert_eq!(b.ws.is_some(), i == ws_block(cfg.layers), "block {i}");
    }
    let (_, ws) = blocks[ws_block(cfg.layers)].ws.as_ref().unwrap();
    assert_eq!(ws.gating, ava_parts::wm::Gating::Compete { k: WS_TOPK });
    let wo = Tensor::from_vec(unit_noise(ws.dk * cfg.d, 4), (ws.dk, cfg.d), &CPU).unwrap();
    vm.data().lock().unwrap().get(&format!("b{}.parts.ws.wo", ws_block(cfg.layers))).unwrap().set(&wo).unwrap();
    let h = Tensor::from_vec(unit_noise(2 * 12 * cfg.d, 5), (2, 12, cfg.d), &CPU).unwrap();
    let fast = crate::scan_fast::wm_forward(ws, &h).unwrap();
    let want = ws.forward(&h).unwrap();
    assert!(max_diff(&fast, &want) < 1e-5);
    // each token writes at most WS_TOPK slots
    let g: Vec<f32> = ws.gates(&h).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    for row in g.chunks(ws.slots) {
        assert!(row.iter().filter(|&&x| x > 0.0).count() <= WS_TOPK, "{row:?}");
    }
}

/// D2 and L5: in every block at their sizes, and each read-out reaches the model's output once it is not 0.
#[test]
fn holo_and_sync_sit_in_every_block_and_reach_the_output() {
    let (base_vm, base) = build_awake(cfg_with(""), 3);
    let (b, t) = (2, 24);
    let toks = tokens(b, t, 258, 8);
    let want = one_pass(&base, &toks, b, t);
    for (f, name, rows) in [("--holo 8", "holo", 8), ("--sync 4", "sync", 8)] {
        let (vm, m) = twin(cfg_with(f), &base_vm);
        let blocks = &m.ext.as_ref().unwrap().mem.w5.blocks;
        for (i, bl) in blocks.iter().enumerate() {
            let ok = match name {
                "holo" => bl.holo.as_ref().is_some_and(|(_, h)| h.db == 8),
                _ => bl.sync.as_ref().is_some_and(|(_, s)| s.n == 4),
            };
            assert!(ok, "{f}: block {i}");
        }
        assert!(max_diff(&one_pass(&m, &toks, b, t), &want) < 1e-6, "{f} starts as no change");
        let wo = Tensor::from_vec(unit_noise(rows * 32, 6), (rows, 32), &CPU).unwrap();
        vm.data().lock().unwrap().get(&format!("b0.parts.{name}.wo")).unwrap().set(&wo).unwrap();
        let d = max_diff(&one_pass(&m, &toks, b, t), &want);
        assert!(d > 1e-4, "{f}: the read-out reaches the output ({d})");
    }
}

/// F12: the schema only watches. With the same weights, every other weight's gradient is the same with and without it,
/// its own weight learns, and its loss is the part's surprise of the workspace's shares against its guess.
#[test]
fn schema_watches_without_changing_the_model() {
    let (base_vm, _) = build_awake(cfg_with("--workspace 4"), 5);
    let (pvm, plain) = twin(cfg_with("--workspace 4"), &base_vm);
    let (vm, with) = twin(cfg_with("--workspace 4 --schema 0.5"), &base_vm);
    let (ids, tg) = ids_tg(2, 40, 12);
    let g0 = plain.loss_aux(&ids, &tg, 0.0).unwrap().0.backward().unwrap();
    let (total, ce, _) = with.loss_aux(&ids, &tg, 0.0).unwrap();
    assert!(total.to_scalar::<f32>().unwrap() > ce.to_scalar::<f32>().unwrap());
    let g1 = total.backward().unwrap();
    let data = vm.data().lock().unwrap();
    let pdata = pvm.data().lock().unwrap();
    let mut checked = 0;
    for (k, v) in data.iter() {
        if k.contains("parts.schema") {
            assert!(gsum(&g1, v).unwrap() > 0.0, "{k} learns");
            continue;
        }
        let p = pdata.get(k).unwrap_or_else(|| panic!("{k} only with the schema"));
        let (a, b) = (g1.get(v.as_tensor()), g0.get(p.as_tensor()));
        match (a, b) {
            (Some(a), Some(b)) => assert!(max_diff(a, b) < 1e-6, "{k}: the schema changed its gradient"),
            (None, None) => {}
            _ => panic!("{k}: a gradient on one side only"),
        }
        checked += 1;
    }
    assert_eq!(checked, pdata.len());
}

/// Diagnostic (ignored): which flags make Ava 0.4's set (d64 L4, glance vocab) read in pieces differently from one
/// pass. Prints the relative gap (max |diff| / max |logit|) for the full set and with one group left out. Run once per
/// AVA_TEM_BETA value (it is read once per process).
#[test]
#[ignore]
fn pieces_gap_by_flag_diag() {
    const WIN: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix --lif-init logspaced --lif-scale";
    let slow = ["--gates channel", "--mem-momentum 0.5", "--spikes int2", "--alif", "--homeostasis 0.1"];
    let all: Vec<&str> = ALL.split(" --").collect();
    let join = |parts: &[&str]| parts.join(" ");
    let mut sets: Vec<(String, String)> = vec![("full".into(), format!("{WIN} {} {ALL}", join(&slow)))];
    for (i, s) in slow.iter().enumerate() {
        let rest: Vec<&str> = slow.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, s)| *s).collect();
        sets.push((format!("no {s}"), format!("{WIN} {} {ALL}", join(&rest))));
    }
    for i in 0..all.len() {
        let rest: Vec<String> = all.iter().enumerate().filter(|&(j, _)| j != i).map(|(j, s)| if j == 0 { s.to_string() } else { format!("--{s}") }).collect();
        let name = if i == 0 { all[0].to_string() } else { format!("--{}", all[i]) };
        sets.push((format!("no {name}"), format!("{WIN} {} {}", join(&slow), rest.join(" "))));
    }
    sets.push(("WIN + SLOW only".into(), format!("{WIN} {}", join(&slow))));
    sets.push(("WIN + SLOW + --tem 4".into(), format!("{WIN} {} --tem 4", join(&slow))));
    sets.push(("WIN only".into(), WIN.into()));
    for (name, flags) in sets {
        // a part that needs the one left out (the schema reads the workspace) is skipped, with the reason
        let built = Arch::from_flags(&flags).and_then(|ar| Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 64, 4, 64).with_glance().and_then(|c| c.with_arch(ar)).map_err(|e| e.to_string()));
        let cfg = match built {
            Ok(c) => c,
            Err(e) => {
                println!("PIECESDIAG {name:28} skipped: {e}");
                continue;
            }
        };
        let (_, m) = build_awake(cfg, 3);
        let (b, t) = (2, 48);
        let toks = tokens(b, t, ava_data::numbers::VOCAB_GLANCE, 9);
        let want = one_pass(&m, &toks, b, t);
        let top = want.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        let mut worst = 0f32;
        for pieces in [vec![17, 14, 17], vec![1; 48], vec![20, 28]] {
            let mut st = m.state(b).unwrap();
            worst = worst.max(max_diff(&in_pieces(&m, &mut st, &toks, b, t, &pieces), &want));
        }
        println!("PIECESDIAG {name:28} abs {worst:.3e} rel {:.3e} (max logit {top:.2})", worst / top);
    }
}

/// H3b (prereg 17n): with --reflex the one-pass answer learns the pondered one where she is right. Two twins train the
/// same steps on the same batch; the control has a vanishing weight (1e-9, the same code path). Bar set before the
/// run: |CE(one pass) − CE(pondered)| ends at most 0.8x the control's, and the pondered CE is at most 0.01 worse.
/// (H3's first design, pulling the first MLP pass toward the pondered block output, failed: prereg 17j.)
#[test]
fn reflex_brings_the_one_pass_answer_to_the_pondered_one() {
    let (ids, tg) = ids_tg(2, 24, 31);
    let train = |f: &str| -> (f32, f32) {
        let cfg = cfg_with(f);
        let vm = VarMap::new();
        let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
        fix_weights(&vm, &cfg, 21);
        m.set_track_activity(false);
        let mut params: Vec<Param> =
            vm.data().lock().unwrap().iter().map(|(k, v)| Param { name: k.clone(), var: v.clone(), decay: Model::decays_with(k, true) }).collect();
        params.sort_by(|a, b| a.name.cmp(&b.name));
        let mut opt = AdamW::new(params, 0.0).unwrap();
        for _ in 0..150 {
            let (total, _, _) = m.loss_aux(&ids, &tg, 0.0).unwrap();
            opt.step(&total.backward().unwrap(), 3e-3, 1.0).unwrap();
            assert!(stash_empty(&m), "{f}");
        }
        let w5 = &m.ext.as_ref().unwrap().mem.w5;
        let ponder = m.loss(&ids, &tg).unwrap().0.to_scalar::<f32>().unwrap();
        w5.fast.set(true);
        let quick = m.loss(&ids, &tg).unwrap().0.to_scalar::<f32>().unwrap();
        w5.fast.set(false);
        (ponder, quick)
    };
    let (p_off, q_off) = train("--halting 3 --reflex 1e-9");
    let (p_on, q_on) = train("--halting 3 --reflex 1");
    let (g_off, g_on) = (q_off - p_off, q_on - p_on);
    eprintln!("H3b after 150 steps: pondered ce {p_off:.4} -> {p_on:.4}, one-pass ce {q_off:.4} -> {q_on:.4}; gap {g_off:.4e} -> {g_on:.4e} (x{:.3})", g_on.abs() / g_off.abs());
    assert!(g_on.abs() <= 0.8 * g_off.abs(), "gap {g_off} -> {g_on}");
    assert!(p_on <= p_off + 0.01, "pondered ce {p_off} -> {p_on}");
}

/// F9: with --fok-qa an answer-only step teaches the confidence probe only where the answer is scored. Targets at the
/// unscored (question) positions are flipped between right and wrong: the probe's gradient must not move with them,
/// and without --fok-qa it must (the control, so the test can fail).
#[test]
fn fok_qa_teaches_the_probe_only_where_the_answer_is_scored() {
    let (ids, tg) = ids_tg(2, 24, 41);
    let (b, t) = (2usize, 24usize);
    let n = b * t;
    let wt: Vec<f32> = (0..n).map(|i| if i % t >= t / 2 { 1.0 } else { 0.0 }).collect();
    let wt = Tensor::from_vec(wt, (b, t), &CPU).unwrap();
    for (f, masked) in [("--confidence 0.5 --fok-qa", true), ("--confidence 0.5", false)] {
        let (vm, m) = build_awake(cfg_with(f), 7);
        let (lg, _) = m.forward_with_activity(&ids).unwrap();
        let v = lg.dim(D::Minus1).unwrap();
        let am: Vec<u32> = lg.reshape((n, v)).unwrap().argmax(D::Minus1).unwrap().to_vec1().unwrap();
        let tv: Vec<u32> = tg.flatten_all().unwrap().to_vec1().unwrap();
        // at the question positions: a = always wrong, b = always right; the answer positions keep their targets
        let pick = |right: bool| -> Tensor {
            let x: Vec<u32> = (0..n).map(|i| if i % t >= t / 2 { tv[i] } else if right { am[i] } else { (am[i] + 1) % v as u32 }).collect();
            Tensor::from_vec(x, (b, t), &CPU).unwrap()
        };
        let (la, _, ga) = m.loss_and_grads_autograd(&ids, &pick(false), Some(&wt), false).unwrap();
        assert!(stash_empty(&m), "{f}");
        let (lb, _, gb) = m.loss_and_grads_autograd(&ids, &pick(true), Some(&wt), false).unwrap();
        assert!(stash_empty(&m), "{f}");
        assert_eq!(la.to_scalar::<f32>().unwrap(), lb.to_scalar::<f32>().unwrap(), "{f}: the answer-only loss ignores the question");
        let data = vm.data().lock().unwrap();
        let mut probe = 0f32;
        for (k, var) in data.iter().filter(|(k, _)| k.starts_with("aux.conf")) {
            assert!(gsum(&ga, var).is_some_and(|s| s > 0.0), "{f}: {k} learns");
            probe = probe.max(max_diff(ga.get(var.as_tensor()).unwrap(), gb.get(var.as_tensor()).unwrap()));
        }
        if masked {
            assert!(probe < 1e-7, "{f}: the question positions must not teach the probe ({probe})");
        } else {
            assert!(probe > 1e-4, "{f}: control, every position teaches the probe ({probe})");
        }
    }
}

/// F10 follow-up: --fok-push gives the probe a second lesson on a pushed state. Same weights, same batch: the probe's
/// gradient changes, and every other gradient is the same bits (the lesson is detached from her).
#[test]
fn fok_push_teaches_only_the_probe() {
    let (ids, tg) = ids_tg(2, 24, 43);
    let grads = |f: &str| {
        let (vm, m) = build_awake(cfg_with(f), 9);
        let (l, _, g) = m.loss_and_grads_autograd(&ids, &tg, None, false).unwrap();
        assert!(stash_empty(&m), "{f}");
        let data = vm.data().lock().unwrap();
        let mut v: Vec<(String, Tensor)> = data.iter().map(|(k, var)| (k.clone(), g.get(var.as_tensor()).unwrap().clone())).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        (l.to_scalar::<f32>().unwrap(), v)
    };
    let (l0, g0) = grads("--confidence 0.5");
    let (l1, g1) = grads("--confidence 0.5 --fok-push 0.5");
    assert_eq!(l0, l1, "the logged loss is the plain CE");
    let mut probe = 0f32;
    for ((k, a), (_, b)) in g0.iter().zip(&g1) {
        if k.starts_with("aux.conf") {
            probe = probe.max(max_diff(a, b));
        } else {
            assert_eq!(max_diff(a, b), 0.0, "{k}: the pushed lesson must not reach her");
        }
    }
    assert!(probe > 1e-6, "the probe learns from the pushed state ({probe})");
}
