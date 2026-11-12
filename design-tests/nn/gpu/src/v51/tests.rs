// TESTING / DESIGN ONLY — v51 tests: every option's gradient check, chat check, checkpoint round trip, flags and counts.
//
// The spiking neurons train through a surrogate gradient (not the true derivative), so a finite-difference check of a
// whole model is only valid for weights that act after the last block's spikes. So each part is checked twice: its
// maths alone in f64 against central finite differences (all of R, M, Y, W and S's synapse), and, where the part has
// weights after the last block's spikes (P, DA, W, Y, S, Pr), the whole model's f32 loss along random directions.

use super::*;
use crate::arch::Arch;
use crate::ckpt::{load_auto, write_arch_beside};
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, in_pieces, max_diff, one_pass, tokens};
use crate::testkit::{scratch, unit_noise};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};

const CPU: Device = Device::Cpu;
const V: usize = 258;
use super::golden::AVA04;

/// The test vocabulary's letters: token v < 256 has 1 to 7 bytes, EOS and PAD none.
fn test_letters() {
    let t: Vec<Vec<u8>> = (0..V).map(|v| if v < 256 { (0..1 + v % 7).map(|j| ((v * 31 + j * 17) % 256) as u8).collect() } else { Vec::new() }).collect();
    set_letters(t);
}

/// The 0.4 recipe at d128 with 3 blocks (2 memory heads of 64), the engram in block 1, plus `extra`.
fn cfg(extra: &str) -> Config {
    let arch = Arch::from_flags(&format!("{AVA04} {extra}")).unwrap();
    Config::with_kind(Kind::D, V, 128, 3, 256).with_engram(1).unwrap().with_arch(arch).unwrap()
}

fn noise(shape: &[usize], seed: u64) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec(unit_noise(n, seed), shape, &CPU).unwrap()
}

fn set_var(vm: &VarMap, name: &str, t: &Tensor) {
    let data = vm.data().lock().unwrap();
    data.get(name).unwrap_or_else(|| panic!("no variable {name}")).set(t).unwrap();
}

fn var(vm: &VarMap, name: &str) -> Var {
    vm.data().lock().unwrap().get(name).unwrap_or_else(|| panic!("no variable {name}")).clone()
}

/// A model with the option on, awake weights (every zero-init weight of v51 non-zero, so the parts act), resets
/// active for Y, and half the MLP units cut for Pr.
fn build(extra: &str, seed: u64) -> (Config, VarMap, Model) {
    test_letters();
    let c = cfg(extra);
    let (vm, m) = build_awake(c, seed);
    tune(&c, &vm, &m);
    (c, vm, m)
}

fn tune(c: &Config, vm: &VarMap, m: &Model) {
    let o = c.arch.v51;
    if o.rhythm {
        set_var(vm, "v51.rhythm.br", &Tensor::new(&[0.0f32, -1.0], &CPU).unwrap());
    }
    if o.prune > 0.0 {
        let h = c.hidden();
        for i in 0..c.layers {
            // half the units, a different half per block
            let al: Vec<f32> = (0..h).map(|j| if (j + i) % 2 == 0 { 1.0 } else { 0.0 }).collect();
            set_var(vm, &format!("b{i}.v51.alive"), &Tensor::from_vec(al, h, &CPU).unwrap());
        }
        m.v51_schedule(0, 0);
    }
}

/// Every option alone, and all of them together (W and Pr are not combined; the together set takes Pr).
const OPTIONS: [(&str, &str); 9] = [
    ("P", "--dendrite"),
    ("S", "--stp"),
    ("R", "--letters"),
    ("DA", "--dopamine"),
    ("M", "--mood"),
    ("W", "--columns 4"),
    ("Pr", "--prune"),
    ("Y", "--rhythm"),
    ("all", "--dendrite --stp --letters --dopamine --mood --prune --rhythm"),
];

fn batch(b: usize, t: usize, seed: u64) -> (Tensor, Tensor) {
    let v = tokens(b, t + 1, V, seed);
    let x: Vec<u32> = (0..b).flat_map(|r| v[r * (t + 1)..r * (t + 1) + t].to_vec()).collect();
    let y: Vec<u32> = (0..b).flat_map(|r| v[r * (t + 1) + 1..(r + 1) * (t + 1)].to_vec()).collect();
    (Tensor::from_vec(x, (b, t), &CPU).unwrap(), Tensor::from_vec(y, (b, t), &CPU).unwrap())
}

// ------------------------------------------------------------------------------------------- flags, counts, golden

#[test]
fn v51_flags_round_trip_and_off_by_default() {
    assert!(!Arch::default().v51.any());
    for (name, extra) in OPTIONS.iter().copied().chain([("W8", "--columns 8"), ("Pr3", "--prune-to 0.25"), ("M2", "--mood-units 2")]) {
        let a = cfg(extra).arch;
        assert_eq!(Arch::from_flags(&a.flags()).unwrap(), a, "{name}: flags {}", a.flags());
        assert!(!a.signature().is_empty(), "{name}");
        if name != "S" {
            assert!(a.v51.any(), "{name}");
        }
    }
    // combinations it refuses
    let bad = |extra: &str| Arch::from_flags(&format!("{AVA04} {extra}")).ok().and_then(|a| Config::with_kind(Kind::D, V, 64, 3, 256).with_arch(a).ok()).is_none();
    assert!(bad("--columns 3"), "3 does not divide d 64");
    assert!(bad("--prune --columns 4"));
    assert!(bad("--prune-to 1.5"));
    assert!(bad("--dopamine --feedback"));
    assert!(Config::with_kind(Kind::D, V, 64, 1, 256).with_arch(Arch::from_flags("--mood").unwrap()).is_err(), "M needs 2 blocks");
    assert!(Config::with_kind(Kind::T, V, 64, 2, 256).with_arch(Arch::from_flags("--rhythm").unwrap()).is_err(), "D only");
}

#[test]
fn v51_weight_counts_match_the_variables() {
    test_letters();
    for (name, extra) in OPTIONS {
        let c = cfg(extra);
        let vm = VarMap::new();
        let _m = Model::new(c, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
        let n: usize = vm.data().lock().unwrap().values().map(|v| v.elem_count()).sum();
        assert_eq!(c.weight_counts().0, n, "{name}");
    }
}

/// Each new part changes nothing at its init (the zero gains): the model's logits equal the same model's without the
/// option, its other weights copied (W rewires the MLP, so it is left out).
#[test]
fn v51_parts_start_as_no_change() {
    test_letters();
    let (x, _) = batch(2, 70, 3);
    for (name, extra) in OPTIONS {
        if name == "W" {
            continue; // W rewires the MLP
        }
        let c0 = cfg("");
        let (vm0, m0) = build_awake(c0, 11);
        let c1 = cfg(extra);
        let vm1 = VarMap::new();
        let m1 = Model::new(c1, VarBuilder::from_varmap(&vm1, DType::F32, &CPU)).unwrap();
        // the shared weights from the plain model; the new ones keep their init
        let d0 = vm0.data().lock().unwrap();
        for (k, v) in vm1.data().lock().unwrap().iter() {
            if let Some(w) = d0.get(k) {
                v.set(w.as_tensor()).unwrap();
            }
        }
        drop(d0);
        m1.v51_schedule(0, 0);
        let a = m0.forward(&x).unwrap();
        let b = m1.forward(&x).unwrap();
        let d = max_diff(&a, &b);
        // R's and Y's start values are exactly neutral; DA, M, P too (gains 1 + tanh(0)); Pr's mask adds 0
        assert!(d == 0.0, "{name}: {d}");
        let mut st = m1.state(2).unwrap();
        let d = max_diff(&m1.forward_with_state(&x, &mut st).unwrap(), &a);
        assert!(d < 1e-4, "{name} state path: {d}");
    }
}

// ------------------------------------------------------------------------------------------- f64 maths against finite differences

fn f64n(shape: &[usize], seed: u64, scale: f64) -> Tensor {
    noise(shape, seed).to_dtype(DType::F64).unwrap().affine(scale, 0.0).unwrap()
}

/// Central differences of f against the autograd gradient ga of `var`, every `stride`-th element.
fn fd_check(what: &str, var: &Var, ga: &Tensor, f: &dyn Fn() -> f64, stride: usize) {
    let eps = 1e-6;
    let base = var.as_tensor().copy().unwrap();
    let flat: Vec<f64> = base.flatten_all().unwrap().to_vec1().unwrap();
    let gv: Vec<f64> = ga.flatten_all().unwrap().to_vec1().unwrap();
    let mut worst = 0f64;
    for j in (0..flat.len()).step_by(stride) {
        let mut p = flat.clone();
        p[j] += eps;
        var.set(&Tensor::from_vec(p.clone(), base.dims(), &CPU).unwrap()).unwrap();
        let up = f();
        p[j] -= 2.0 * eps;
        var.set(&Tensor::from_vec(p, base.dims(), &CPU).unwrap()).unwrap();
        let down = f();
        var.set(&base).unwrap();
        let fd = (up - down) / (2.0 * eps);
        let err = (fd - gv[j]).abs() / (1.0 + fd.abs());
        worst = worst.max(err);
        assert!(err <= 1e-6, "{what}[{j}]: finite difference {fd} vs autograd {}", gv[j]);
    }
    eprintln!("v51 fd {what}: worst relative error {worst:.2e} over {} entries", flat.len().div_ceil(stride));
}

#[test]
fn v51_letter_encoder_gradients_match_finite_differences() {
    let (m, l, dl) = (4usize, 5usize, 6usize);
    let byte = Var::from_tensor(&f64n(&[256, dl], 1, 0.6)).unwrap();
    let pos = Var::from_tensor(&f64n(&[l, dl], 2, 0.6)).unwrap();
    let end = Var::from_tensor(&f64n(&[l, dl], 3, 0.6)).unwrap();
    let wo = Var::from_tensor(&f64n(&[dl, 7], 4, 0.5)).unwrap();
    let lens = [1usize, 3, 5, 0];
    let (mut ids, mut ei, mut mk, mut inv) = (vec![0u32; m * l], vec![0u32; m * l], vec![0f64; m * l], vec![0f64; m]);
    for (t, &len) in lens.iter().enumerate() {
        for j in 0..len {
            ids[t * l + j] = ((t * 37 + j * 11) % 256) as u32;
            ei[t * l + j] = (len - 1 - j) as u32;
            mk[t * l + j] = 1.0;
        }
        if len > 0 {
            inv[t] = 1.0 / (len as f64).sqrt();
        }
    }
    let ids = Tensor::from_vec(ids, (m, l), &CPU).unwrap();
    let ei = Tensor::from_vec(ei, (m, l), &CPU).unwrap();
    let mk = Tensor::from_vec(mk, (m, l), &CPU).unwrap();
    let inv = Tensor::from_vec(inv, (m, 1), &CPU).unwrap();
    let r = f64n(&[m, 7], 5, 1.0);
    let f = || letter_core(byte.as_tensor(), pos.as_tensor(), end.as_tensor(), &ids, &ei, &mk, &inv).unwrap().matmul(wo.as_tensor()).unwrap().mul(&r).unwrap().sum_all().unwrap();
    let out = letter_core(byte.as_tensor(), pos.as_tensor(), end.as_tensor(), &ids, &ei, &mk, &inv).unwrap();
    // the token without bytes gets no vector
    assert_eq!(out.get(3).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f64>().unwrap(), 0.0);
    let g = f().backward().unwrap();
    let fs = || f().to_scalar::<f64>().unwrap();
    fd_check("R byte table", &byte, g.get(byte.as_tensor()).unwrap(), &fs, 7);
    fd_check("R place", &pos, g.get(pos.as_tensor()).unwrap(), &fs, 1);
    fd_check("R place from the end", &end, g.get(end.as_tensor()).unwrap(), &fs, 1);
    fd_check("R out", &wo, g.get(wo.as_tensor()).unwrap(), &fs, 1);
}

/// M's scan: the chunked closed form (3 chunks) equals the serial recurrence, and its gradients (by λ, the drive and
/// the start) match finite differences; then the mood's gain on a current.
#[test]
fn v51_mood_scan_is_the_recurrence_and_its_gradients_match() {
    let (b, t, k) = (2usize, 150usize, 3usize);
    let u = Var::from_tensor(&f64n(&[b, t, k], 10, 1.0)).unwrap();
    let lam = Var::from_tensor(&Tensor::new(&[0.5f64, 2.0, 4.0], &CPU).unwrap()).unwrap();
    let m0 = Var::from_tensor(&f64n(&[b, k], 11, 0.5)).unwrap();
    let got = mood_scan(u.as_tensor(), lam.as_tensor(), m0.as_tensor()).unwrap();
    // serial
    let lv: Vec<f64> = lam.as_tensor().to_vec1().unwrap();
    let uv: Vec<f64> = u.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    let mut mv: Vec<f64> = m0.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    let gv: Vec<f64> = got.flatten_all().unwrap().to_vec1().unwrap();
    let mut worst = 0f64;
    for r in 0..b {
        for s in 0..t {
            for j in 0..k {
                let l = 1.0 / (1.0 + (-lv[j]).exp());
                mv[r * k + j] = l * mv[r * k + j] + (1.0 - l) * uv[(r * t + s) * k + j];
                worst = worst.max((mv[r * k + j] - gv[(r * t + s) * k + j]).abs());
            }
        }
    }
    assert!(worst < 1e-12, "closed form vs serial: {worst}");
    let rr = f64n(&[b, t, k], 12, 1.0);
    let f = || mood_scan(u.as_tensor(), lam.as_tensor(), m0.as_tensor()).unwrap().mul(&rr).unwrap().sum_all().unwrap();
    let g = f().backward().unwrap();
    let fs = || f().to_scalar::<f64>().unwrap();
    fd_check("M lambda", &lam, g.get(lam.as_tensor()).unwrap(), &fs, 1);
    fd_check("M drive", &u, g.get(u.as_tensor()).unwrap(), &fs, 37);
    fd_check("M start", &m0, g.get(m0.as_tensor()).unwrap(), &fs, 1);
    // the gain on a block's current: cur ⊙ (1 + tanh(m·G))
    let gm = Var::from_tensor(&f64n(&[k, 5], 13, 0.7)).unwrap();
    let mm = f64n(&[6, k], 14, 1.0);
    let cur = f64n(&[6, 5], 15, 1.0);
    let r2 = f64n(&[6, 5], 16, 1.0);
    let f2 = || cur.mul(&gain(&mm.matmul(gm.as_tensor()).unwrap()).unwrap()).unwrap().mul(&r2).unwrap().sum_all().unwrap();
    let g2 = f2().backward().unwrap();
    fd_check("M gain", &gm, g2.get(gm.as_tensor()).unwrap(), &|| f2().to_scalar::<f64>().unwrap(), 1);
}

/// Y's scan: the chunked closed form equals the serial a_t = (1 - r_t)(a_{t-1} + 1), and the gradients (by the
/// resets, the start and the frequencies through the phases) match finite differences.
#[test]
fn v51_rhythm_scan_is_the_recurrence_and_its_gradients_match() {
    let (b, t, o) = (2usize, 140usize, 2usize);
    let pre = Var::from_tensor(&f64n(&[b, t, o], 20, 2.0).affine(1.0, -1.5).unwrap()).unwrap();
    let a0 = Var::from_tensor(&f64n(&[b, o], 21, 3.0).abs().unwrap()).unwrap();
    let lw = Var::from_tensor(&Tensor::new(&[(2.0 * std::f64::consts::PI / 16.0).ln(), (2.0 * std::f64::consts::PI / 3.2).ln()], &CPU).unwrap()).unwrap();
    let got = rhythm_scan(pre.as_tensor(), a0.as_tensor()).unwrap();
    let pv: Vec<f64> = pre.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    let mut av: Vec<f64> = a0.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    let gv: Vec<f64> = got.flatten_all().unwrap().to_vec1().unwrap();
    let mut worst = 0f64;
    for r in 0..b {
        for s in 0..t {
            for j in 0..o {
                let rr = 1.0 / (1.0 + (-pv[(r * t + s) * o + j]).exp());
                av[r * o + j] = (1.0 - rr) * (av[r * o + j] + 1.0);
                worst = worst.max((av[r * o + j] - gv[(r * t + s) * o + j]).abs());
            }
        }
    }
    assert!(worst < 1e-9, "closed form vs serial: {worst}");
    let rr = f64n(&[b * t, 2 * o], 22, 1.0);
    let f = || rhythm_feats(&rhythm_scan(pre.as_tensor(), a0.as_tensor()).unwrap(), lw.as_tensor()).unwrap().mul(&rr).unwrap().sum_all().unwrap();
    let g = f().backward().unwrap();
    let fs = || f().to_scalar::<f64>().unwrap();
    fd_check("Y resets", &pre, g.get(pre.as_tensor()).unwrap(), &fs, 13);
    fd_check("Y start", &a0, g.get(a0.as_tensor()).unwrap(), &fs, 1);
    fd_check("Y frequencies", &lw, g.get(lw.as_tensor()).unwrap(), &fs, 1);
}

/// W's wiring: the column matmul equals the dense matmul with every off-column block zeroed, plus the long-range path;
/// gradients match finite differences.
#[test]
fn v51_columns_are_the_masked_matmul_and_their_gradients_match() {
    let (n, di, dout, k, r) = (5usize, 8usize, 12usize, 4usize, 2usize);
    let h = Var::from_tensor(&f64n(&[n, di], 30, 1.0)).unwrap();
    let w = Var::from_tensor(&f64n(&[di, dout], 31, 0.5)).unwrap();
    let a = Var::from_tensor(&f64n(&[di, r], 32, 0.5)).unwrap();
    let c = Var::from_tensor(&f64n(&[r, dout], 33, 0.5)).unwrap();
    let mask: Vec<f64> = (0..di * dout).map(|e| if (e / dout) / (di / k) == (e % dout) / (dout / k) { 1.0 } else { 0.0 }).collect();
    let mask = Tensor::from_vec(mask, (di, dout), &CPU).unwrap();
    let got = col_mm(h.as_tensor(), w.as_tensor(), a.as_tensor(), c.as_tensor(), k).unwrap();
    let want = h.as_tensor().matmul(&w.as_tensor().mul(&mask).unwrap()).unwrap().add(&h.as_tensor().matmul(a.as_tensor()).unwrap().matmul(c.as_tensor()).unwrap()).unwrap();
    let d = got.sub(&want).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f64>().unwrap();
    assert!(d < 1e-12, "column matmul vs masked dense: {d}");
    let rr = f64n(&[n, dout], 34, 1.0);
    let f = || col_mm(h.as_tensor(), w.as_tensor(), a.as_tensor(), c.as_tensor(), k).unwrap().mul(&rr).unwrap().sum_all().unwrap();
    let g = f().backward().unwrap();
    let fs = || f().to_scalar::<f64>().unwrap();
    let gw = g.get(w.as_tensor()).unwrap();
    // an off-column weight gets no gradient
    let off = gw.mul(&mask.affine(-1.0, 1.0).unwrap()).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f64>().unwrap();
    assert_eq!(off, 0.0);
    fd_check("W blocks", &w, gw, &fs, 1);
    fd_check("W long-range in", &a, g.get(a.as_tensor()).unwrap(), &fs, 1);
    fd_check("W long-range out", &c, g.get(c.as_tensor()).unwrap(), &fs, 1);
    fd_check("W input", &h, g.get(h.as_tensor()).unwrap(), &fs, 1);
}

/// S (E9, ava_parts::stp): the synapse's own autograd ops against finite differences in f64 (U, τf, τd and the mix;
/// int2-sized events), and the fused scan the model runs equal to them in value and gradient (f32).
#[test]
fn v51_stp_gradients_match_finite_differences_and_the_fused_scan() {
    let (b, t, c) = (2usize, 30usize, 4usize);
    let ev = |seed: u64, dt: DType| -> Tensor {
        // int2 spikes: -2..2, mostly 0
        let v: Vec<f32> = unit_noise(b * t * c, seed).iter().map(|z| if z.abs() < 0.8 { 0.0 } else { (z * 1.3).round().clamp(-2.0, 2.0) }).collect();
        Tensor::from_vec(v, (b, t, c), &CPU).unwrap().to_dtype(dt).unwrap()
    };
    let mk = |dt: DType| -> (ava_parts::stp::Stp, [Var; 4]) {
        let vs = [
            Var::from_tensor(&noise(&[c], 40).affine(0.3, -1.0).unwrap().to_dtype(dt).unwrap()).unwrap(),
            Var::from_tensor(&noise(&[c], 41).affine(0.3, 2.5).unwrap().to_dtype(dt).unwrap()).unwrap(),
            Var::from_tensor(&noise(&[c], 42).affine(0.3, 1.5).unwrap().to_dtype(dt).unwrap()).unwrap(),
            Var::from_tensor(&noise(&[c], 43).affine(0.5, 0.0).unwrap().to_dtype(dt).unwrap()).unwrap(),
        ];
        let p = ava_parts::stp::Stp { u_param: vs[0].as_tensor().clone(), f_param: vs[1].as_tensor().clone(), d_param: vs[2].as_tensor().clone(), mix: vs[3].as_tensor().clone() };
        (p, vs)
    };
    let s = ev(44, DType::F64);
    let rr = f64n(&[b, t, c], 45, 1.0);
    let (_, vs) = mk(DType::F64);
    let f = || {
        let p = ava_parts::stp::Stp { u_param: vs[0].as_tensor().clone(), f_param: vs[1].as_tensor().clone(), d_param: vs[2].as_tensor().clone(), mix: vs[3].as_tensor().clone() };
        p.forward(&s).unwrap().out.mul(&rr).unwrap().sum_all().unwrap()
    };
    let g = f().backward().unwrap();
    let fs = || f().to_scalar::<f64>().unwrap();
    for (name, v) in ["S U", "S tau_f", "S tau_d", "S mix"].iter().zip(&vs) {
        fd_check(name, v, g.get(v.as_tensor()).unwrap(), &fs, 1);
    }
    // the fused scan (f32) vs the part's ops (f32): values and gradients
    let s32 = ev(44, DType::F32);
    let r32 = rr.to_dtype(DType::F32).unwrap();
    let (p, vs) = mk(DType::F32);
    let a = p.forward(&s32).unwrap().out;
    let bq = crate::scan_fast::stp_forward(&p, &s32).unwrap();
    assert!(max_diff(&a, &bq) < 1e-6, "fused stp vs the part: {}", max_diff(&a, &bq));
    let ga = a.mul(&r32).unwrap().sum_all().unwrap().backward().unwrap();
    let gb = bq.mul(&r32).unwrap().sum_all().unwrap().backward().unwrap();
    for v in &vs {
        let d = max_diff(ga.get(v.as_tensor()).unwrap(), gb.get(v.as_tensor()).unwrap());
        assert!(d < 1e-4, "fused stp gradient vs the part's: {d}");
    }
}

// ------------------------------------------------------------------------------------------- whole-model gradient checks

fn sum_ce(m: &Model, x: &Tensor, y: &Tensor) -> Tensor {
    let l = m.forward(x).unwrap();
    crate::lif::cross_entropy_rows(&l, &y.flatten_all().unwrap()).unwrap().sum_all().unwrap()
}

/// The summed loss with the rows added in f64 (finer than an f32 sum, for the differences).
fn sum_ce64(m: &Model, x: &Tensor, y: &Tensor) -> f64 {
    let l = m.forward(x).unwrap();
    crate::lif::cross_entropy_rows(&l, &y.flatten_all().unwrap()).unwrap().to_vec1::<f32>().unwrap().iter().map(|&v| v as f64).sum()
}

/// The whole model's summed loss along a random direction of each weight (scaled to the weight's size), by central
/// differences, against the autograd gradient. Valid for weights that act after the last block's spikes in a model
/// whose kWTA keeps every unit (--kwta 1): then the loss is smooth in them and the gradient is the true one.
fn model_fd(tag: &str, vm: &VarMap, m: &Model, names: &[String]) {
    let (x, y) = batch(4, 64, 9);
    let g = sum_ce(m, &x, &y).backward().unwrap();
    for (k, name) in names.iter().enumerate() {
        let v = var(vm, name);
        let ga = g.get(v.as_tensor()).unwrap_or_else(|| panic!("{tag}: no gradient for {name}"));
        let base = v.as_tensor().copy().unwrap();
        let rms = (base.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() as f64).sqrt().max(1e-2);
        let dir = noise(base.dims(), 900 + k as u64).affine(rms, 0.0).unwrap();
        let an = ga.mul(&dir).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() as f64;
        let eps = 5e-2;
        v.set(&base.add(&dir.affine(eps, 0.0).unwrap()).unwrap()).unwrap();
        let up = sum_ce64(m, &x, &y);
        v.set(&base.sub(&dir.affine(eps, 0.0).unwrap()).unwrap()).unwrap();
        let down = sum_ce64(m, &x, &y);
        v.set(&base).unwrap();
        let fd = (up - down) / (2.0 * eps);
        eprintln!("v51 model fd {tag} {name}: autograd {an:.6}, finite difference {fd:.6}");
        assert!(an.abs() > 2e-3, "{tag} {name}: the check is empty (directional derivative {an})");
        assert!((fd - an).abs() <= 0.02 * an.abs() + 5e-4, "{tag} {name}: finite difference {fd} vs autograd {an}");
    }
}

#[test]
fn v51_model_gradients_match_finite_differences() {
    let last = 2;
    let cases: [(&str, &str, Vec<String>); 6] = [
        ("P", "--dendrite", vec![format!("b{last}.v51.dend_a"), format!("b{last}.v51.dend_b")]),
        ("DA", "--dopamine", vec![format!("b{last}.v51.da_w"), format!("b{last}.v51.da_c")]),
        ("W", "--columns 4", vec![format!("b{last}.v51.col_a1"), format!("b{last}.v51.col_b1"), format!("b{last}.v51.col_a2"), format!("b{last}.v51.col_b2"), format!("b{last}.mlp.w2")]),
        ("Y", "--rhythm", vec![format!("b{last}.v51.rh_w"), format!("b{last}.v51.rh_r")]),
        ("S", "--stp", vec![format!("b{last}.parts.stp.mix"), format!("b{last}.parts.stp.u_param")]),
        ("Pr", "--prune", vec![format!("b{last}.mlp.w1"), format!("b{last}.mlp.w2")]),
    ];
    for (tag, extra, names) in cases {
        // kWTA keeps every (live) unit here, so the loss is smooth in these weights (the bake-off keeps 1/4)
        let (_, vm, m) = build(&format!("{extra} --kwta 1"), 21);
        model_fd(tag, &vm, &m, &names);
    }
    // Pr: a cut unit's weights get no gradient (W1's column and W2's row)
    let (c, vm, m) = build("--prune", 21);
    let (x, y) = batch(2, 40, 9);
    let g = sum_ce(&m, &x, &y).backward().unwrap();
    let al: Vec<f32> = var(&vm, "b2.v51.alive").as_tensor().to_vec1().unwrap();
    let g1: Vec<Vec<f32>> = g.get(var(&vm, "b2.mlp.w1").as_tensor()).unwrap().t().unwrap().to_vec2().unwrap();
    let g2: Vec<Vec<f32>> = g.get(var(&vm, "b2.mlp.w2").as_tensor()).unwrap().to_vec2().unwrap();
    for j in 0..c.hidden() {
        let s1: f32 = g1[j].iter().map(|x| x.abs()).sum();
        let s2: f32 = g2[j].iter().map(|x| x.abs()).sum();
        if al[j] == 0.0 {
            assert!(s1 == 0.0 && s2 == 0.0, "cut unit {j}: gradients {s1} {s2}");
        }
    }
}

/// Every trainable v51 weight gets a gradient at the real init (the zero-init gains get a non-zero one).
#[test]
fn v51_every_new_weight_learns_from_the_start() {
    test_letters();
    let c = cfg("--dendrite --letters --dopamine --mood --rhythm");
    let vm = VarMap::new();
    let m = Model::new(c, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
    crate::testkit::fix_weights(&vm, &c, 5); // today's weights awake; the v51 zero inits stay 0
    for z in ["b1.v51.dend_b", "b1.v51.da_w", "v51.letters.wo", "b1.v51.rh_w", "b1.v51.mood_g"] {
        assert_eq!(var(&vm, z).as_tensor().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap(), 0.0, "{z} starts at 0");
    }
    let (x, y) = batch(2, 40, 9);
    let (_, _, g) = m.loss_and_grads_autograd(&x, &y, None, false).unwrap();
    let data = vm.data().lock().unwrap();
    // the zero-init gains learn at once; the weights behind them (A, the letter tables, the mood's and the rhythms'
    // own weights) get 0 until their gain moves
    let zero_init_nonzero = ["dend_b", "da_w", "da_c", "mood_g", "letters.wo", "rh_w", "rh_r"];
    for (k, v) in data.iter().filter(|(k, _)| k.contains("v51.")) {
        let gr = g.get(v.as_tensor());
        let s = gr.map(|x| x.abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap());
        if zero_init_nonzero.iter().any(|z| k.ends_with(z)) && !k.starts_with("b0.v51.mood") {
            assert!(s.is_some_and(|s| s > 0.0 && s.is_finite()), "{k}: gradient {s:?} at the init");
        } else {
            assert!(s.is_some_and(|s| s.is_finite()), "{k}: gradient {s:?}");
        }
    }
}

// ------------------------------------------------------------------------------------------- chat checks

/// The largest |diff| at each position of two logits [t, V].
fn per_pos(a: &Tensor, b: &Tensor) -> Vec<f32> {
    a.sub(b).unwrap().abs().unwrap().max(1).unwrap().to_vec1().unwrap()
}

/// Whether one row's logits got [t, V] agree with want [t, V]: within 1e-4 at every position but at most one
/// isolated flip (one position off, both neighbours within 1e-5: a winner of the last block's kWTA sitting on its
/// threshold, tipped the other way by rounding, since the paths add in other orders; nothing carries it on).
fn agree(tag: &str, what: &str, got: &Tensor, want: &Tensor) {
    let dp = per_pos(got, want);
    let t = dp.len();
    let bad: Vec<(usize, f32)> = dp.iter().enumerate().filter(|(_, &x)| x > 1e-4).map(|(i, &x)| (i, x)).collect();
    for &(p, x) in &bad {
        let iso = (p == 0 || dp[p - 1] < 1e-5) && (p + 1 == t || dp[p + 1] < 1e-5);
        assert!(iso, "{tag} {what}: position {p} off by {x}, and not alone");
    }
    assert!(bad.len() <= 1, "{tag} {what}: {} positions off: {bad:?}", bad.len());
    if !bad.is_empty() {
        eprintln!("v51 chat {tag} {what}: one isolated rounding flip {bad:?}");
    }
}

/// The chat check. Strict, as state::tests::ava04_recipe_streams_match_forward (64 tokens, 2 rows): pieces, every
/// token of a one-token stream, a batched feed, a State saved, loaded and read on, and State::select / cat, all
/// within 1e-4 of the full forward (`agree`: but one isolated rounding flip per row). Long (150 tokens, across the
/// scans' 64-token chunks): pieces and the stream agree within 1e-5 at every position (isolated flips aside) up to the
/// first lasting divergence, if there is one (a spike on its threshold tipped by rounding moves every later logit),
/// and that must come after the first chunk boundary, so the boundary itself is checked.
fn chat_check(tag: &str, m: &Model) {
    let (b, t) = (2, 64);
    let toks = tokens(b, t, V, 72);
    let want = one_pass(m, &toks, b, t); // [b, t, V]
    let mut st = m.state(b).unwrap();
    let got = in_pieces(m, &mut st, &toks, b, t, &[1, 30, 33]);
    for r in 0..b {
        agree(tag, &format!("pieces row {r}"), &got.get(r).unwrap(), &want.get(r).unwrap());
    }
    // one token at a time, every position of row 1
    let mut s = m.stream().unwrap();
    let outs: Vec<Tensor> = toks[t..2 * t].iter().map(|&tk| m.step(&mut s, tk).unwrap()).collect();
    agree(tag, "stream", &Tensor::stack(&outs, 0).unwrap(), &want.get(1).unwrap());
    let mut s = m.stream().unwrap();
    let d = max_diff(&m.feed(&mut s, &toks[..t]).unwrap(), &want.get(0).unwrap().get(t - 1).unwrap());
    assert!(d < 1e-4, "{tag} feed: {d}");
    // a State saved after 30 tokens, loaded and read on
    let cut = 30;
    let mut st = m.state(b).unwrap();
    let first = in_pieces(m, &mut st, &(0..b).flat_map(|r| toks[r * t..r * t + cut].to_vec()).collect::<Vec<_>>(), b, cut, &[cut]);
    let p = scratch(&format!("v51_chat_{tag}")).join("state.safetensors");
    st.save(m, &p).unwrap();
    let mut st2 = crate::state::State::load(m, &p, &CPU).unwrap();
    let rest: Vec<u32> = (0..b).flat_map(|r| toks[r * t + cut..(r + 1) * t].to_vec()).collect();
    let second = in_pieces(m, &mut st2, &rest, b, t - cut, &[t - cut]);
    let joined = Tensor::cat(&[first, second.clone()], 1).unwrap();
    for r in 0..b {
        agree(tag, &format!("saved state row {r}"), &joined.get(r).unwrap(), &want.get(r).unwrap());
    }
    // rows swapped by select, then put back together by cat: the same as the loaded state
    let sw = st.select(&[1, 0]).unwrap();
    let mut both = crate::state::State::cat(&[sw.select(&[1]).unwrap(), sw.select(&[0]).unwrap()]).unwrap();
    let again = in_pieces(m, &mut both, &rest, b, t - cut, &[t - cut]);
    let d = max_diff(&again, &second);
    assert!(d < 1e-4, "{tag} select/cat: {d}");
    // long: across the chunks, on two texts (a tie tipped by rounding can end one comparison early; a fault at the
    // boundary would end every one of them there)
    let t = 150;
    let mut clean = [0usize; 3];
    for seed in [73u64, 74] {
        let toks = tokens(b, t, V, seed);
        let want = one_pass(m, &toks, b, t);
        let mut st = m.state(b).unwrap();
        let pieces = in_pieces(m, &mut st, &toks, b, t, &[1, 63, 2, 84]);
        let mut s = m.stream().unwrap();
        let stream = Tensor::stack(&toks[t..2 * t].iter().map(|&tk| m.step(&mut s, tk).unwrap()).collect::<Vec<_>>(), 0).unwrap();
        for (k, (what, got, row)) in [("pieces row 0", pieces.get(0).unwrap(), 0), ("pieces row 1", pieces.get(1).unwrap(), 1), ("stream row 1", stream, 1)].into_iter().enumerate() {
            let dp = per_pos(&got, &want.get(row).unwrap());
            let iso = |p: usize| (p == 0 || dp[p - 1] < 1e-5) && (p + 1 == t || dp[p + 1] < 1e-5);
            let lasting = (0..t).find(|&p| dp[p] > 1e-5 && !(dp[p] > 1e-4 && iso(p)));
            let flips: Vec<usize> = (0..lasting.unwrap_or(t)).filter(|&p| dp[p] > 1e-4).collect();
            let before = (0..lasting.unwrap_or(t)).filter(|&p| dp[p] <= 1e-4).map(|p| dp[p]).fold(0f32, f32::max);
            eprintln!("v51 chat {tag} long (text {seed}) {what}: max |diff| {before:.1e} up to the first lasting divergence {lasting:?}; isolated flips {flips:?}");
            assert!(before < 1e-5, "{tag} long {what}: {before}");
            if flips.len() <= 1 && lasting.map_or(true, |p| p > SCAN_CHUNK) {
                clean[k] += 1;
            }
        }
    }
    assert!(clean.iter().all(|&c| c >= 1), "{tag} long: no text crossed the chunk boundary cleanly {clean:?}");
}

#[test]
fn v51_streams_match_forward() {
    for (tag, extra) in OPTIONS {
        let (_, _, m) = build(extra, 31);
        chat_check(tag, &m);
        // the step trace shows the slow units as the mood signal, the surprise and the rhythms
        let o = m.cfg.arch.v51;
        let mut s = m.stream().unwrap();
        let mut tr = None;
        for tk in [5u32, 60, 61, 7] {
            tr = Some(m.step_trace(&mut s, tk).unwrap().1);
        }
        let tr = tr.unwrap();
        match tr.v51 {
            Some(v) => {
                assert!(o.surprise() || o.rhythm, "{tag}");
                assert_eq!(v.mood.len(), o.mood, "{tag}");
                assert_eq!(v.surprise.is_some(), o.surprise(), "{tag}");
                assert_eq!(v.rhythm.len(), if o.rhythm { 2 } else { 0 }, "{tag}");
                assert!(v.mood.iter().chain(&v.rhythm).chain(v.surprise.iter()).all(|x| x.is_finite()), "{tag}");
                if let Some(sp) = v.surprise {
                    assert!(sp > 0.0, "{tag}: surprise {sp}");
                }
            }
            None => assert!(!(o.surprise() || o.rhythm), "{tag}: no trace"),
        }
        assert!(m.fused_blocker().is_some() || !o.any(), "{tag}: the fused step says why not");
    }
}

/// The surprise is her own: s_t = -ln q_{t-1}(x_t) from block 0's stream through the final norm and the head, 0 at
/// the start (checked against a hand computation on a fresh stream).
#[test]
fn v51_surprise_is_her_early_guess() {
    let (_, vm, m) = build("--dopamine", 41);
    let toks = [9u32, 100, 33, 250, 4];
    let mut s = m.stream().unwrap();
    let mut sur = Vec::new();
    for &tk in &toks {
        sur.push(m.step_trace(&mut s, tk).unwrap().1.v51.unwrap().surprise.unwrap());
    }
    assert_eq!(sur[0], 0.0, "no guess before the first token");
    // by hand: a one-block model with block 0's weights (and the same embedding and final norm) is her early guess
    let c1 = Config::with_kind(Kind::D, V, 128, 1, 256).with_arch(Arch::from_flags(AVA04).unwrap()).unwrap();
    let vm1 = VarMap::new();
    let m1 = Model::new(c1, VarBuilder::from_varmap(&vm1, DType::F32, &CPU)).unwrap();
    m1.set_track_activity(false);
    let src = vm.data().lock().unwrap();
    for (k, v) in vm1.data().lock().unwrap().iter() {
        let w = src.get(k).unwrap_or_else(|| panic!("{k}"));
        v.set(w.as_tensor()).unwrap();
    }
    drop(src);
    let l = one_pass(&m1, &toks, 1, 5).get(0).unwrap(); // [5, V]
    let ce = crate::lif::cross_entropy_rows(&l.narrow(0, 0, 4).unwrap(), &Tensor::new(&toks[1..], &CPU).unwrap()).unwrap().to_vec1::<f32>().unwrap();
    for t in 1..5 {
        assert!((ce[t - 1] - sur[t]).abs() < 1e-4, "token {t}: by hand {} vs the trace {}", ce[t - 1], sur[t]);
    }
    // the batched state holds the same last surprise
    let ids = Tensor::new(&toks[..], &CPU).unwrap().unsqueeze(0).unwrap();
    let mut st = m.state(1).unwrap();
    m.forward_with_state(&ids, &mut st).unwrap();
    let last = st.v51.last_s.as_ref().unwrap().to_vec1::<f32>().unwrap()[0];
    assert!((last - sur[4]).abs() < 1e-4, "{last} vs {}", sur[4]);
}

// ------------------------------------------------------------------------------------------- checkpoints

#[test]
fn v51_checkpoint_round_trip() {
    let dir = scratch("v51_ckpt");
    let (x, _) = batch(2, 50, 13);
    for (tag, extra) in OPTIONS {
        let (c, vm, m) = build(extra, 51);
        let want = m.forward(&x).unwrap();
        let path = dir.join(format!("{tag}.safetensors"));
        vm.save(&path).unwrap();
        write_arch_beside(&path, &c.arch).unwrap();
        let (c2, vm2, m2) = load_auto(&path, Kind::D, 256, &CPU).unwrap();
        assert_eq!(c2.arch, c.arch, "{tag}");
        let got = m2.forward(&x).unwrap();
        assert_eq!(max_diff(&want, &got), 0.0, "{tag}: logits after the round trip");
        // every weight, the cut mask and the use averages included
        let (a, b) = (vm.data().lock().unwrap(), vm2.data().lock().unwrap());
        for (k, v) in a.iter().filter(|(k, _)| k.contains("v51.")) {
            assert_eq!(max_diff(v.as_tensor(), b[k].as_tensor()), 0.0, "{tag}: {k}");
        }
        drop((a, b));
        // the lean (inference) model over the same weights: the same logits
        let lean = crate::ckpt::lean(c2, &vm2, &CPU).unwrap();
        assert_eq!(max_diff(&want, &lean.forward(&x).unwrap()), 0.0, "{tag}: lean");
    }
}

// ------------------------------------------------------------------------------------------- Pr: the schedule and the compaction

#[test]
fn v51_prune_schedule_is_cubic_and_monotone() {
    assert_eq!(prune_share(0.5, 99, 1000), 0.0);
    assert!((prune_share(0.5, 425, 1000) - 0.5 * (1.0 - 0.5f64.powi(3))).abs() < 1e-12);
    assert_eq!(prune_share(0.5, 750, 1000), 0.5);
    assert_eq!(prune_share(0.5, 999, 1000), 0.5);
    let mut last = 0.0;
    for s in 0..1000 {
        let p = prune_share(0.5, s, 1000);
        assert!(p >= last);
        last = p;
    }
    // a short run: cuts at 10%, then every 50 steps, and at 75%; the weakest live units go and never come back
    test_letters();
    let c = cfg("--prune");
    let (vm, m) = build_awake(c, 61);
    for i in 0..c.layers {
        // the use averages start at 0, as in training (build_awake gave them noise)
        set_var(&vm, &format!("b{i}.v51.use"), &Tensor::zeros(c.hidden(), DType::F32, &CPU).unwrap());
    }
    let total = 120;
    m.v51_schedule(0, total);
    let h = c.hidden();
    let mut alive_before = vec![1f32; h];
    let (x, y) = batch(1, 16, 5);
    for s in 1..=total {
        let (_, _, _g) = m.loss_and_grads_autograd(&x, &y, None, false).unwrap();
        m.homeostasis_step().unwrap();
        let al: Vec<f32> = var(&vm, "b1.v51.alive").as_tensor().to_vec1().unwrap();
        assert!(al.iter().zip(&alive_before).all(|(a, b)| *a <= *b), "step {s}: a unit came back");
        let live = al.iter().filter(|&&a| a > 0.5).count();
        let s0 = (PRUNE_FROM * total as f64).round() as usize;
        let s1 = (PRUNE_TO * total as f64).round() as usize;
        let due = s >= s0 && ((s - s0) % PRUNE_EVERY == 0 || s == s1);
        if due {
            let want = h - (prune_share(0.5, s, total) * h as f64).round() as usize;
            assert_eq!(live, want, "step {s}");
            // the units cut now are the weakest by |W1 col|·|W2 row|·use
            let n1: Vec<f32> = var(&vm, "b1.mlp.w1").as_tensor().sqr().unwrap().sum(0).unwrap().sqrt().unwrap().to_vec1().unwrap();
            let n2: Vec<f32> = var(&vm, "b1.mlp.w2").as_tensor().sqr().unwrap().sum(1).unwrap().sqrt().unwrap().to_vec1().unwrap();
            let us: Vec<f32> = var(&vm, "b1.v51.use").as_tensor().to_vec1().unwrap();
            let sc = |j: usize| n1[j] * n2[j] * (us[j] + 1e-8);
            let cut_max = (0..h).filter(|&j| alive_before[j] > 0.5 && al[j] < 0.5).map(sc).fold(f32::NEG_INFINITY, f32::max);
            let kept_min = (0..h).filter(|&j| al[j] > 0.5).map(sc).fold(f32::INFINITY, f32::min);
            assert!(cut_max <= kept_min, "step {s}: cut {cut_max} > kept {kept_min}");
        }
        alive_before = al;
    }
    assert_eq!(m.v51_alive().unwrap().unwrap(), vec![h / 2; c.layers]);
    let u: Vec<f32> = var(&vm, "b1.v51.use").as_tensor().to_vec1().unwrap();
    assert!(u.iter().all(|x| x.is_finite() && *x >= 0.0) && u.iter().any(|&x| x > 0.0), "use averages");
}

/// v52 fix_Pr: a grow run (`v51_begin_schedule`, as grow_gpu now calls it) cuts on the schedule of all its steps, not
/// of --max-steps, and a run stopped and resumed from its checkpoint (weights, cut mask and use averages saved, the
/// schedule set again at the resumed step) cuts exactly the units the uninterrupted run cuts, with the same use
/// averages, step by step. Without the call (grow_gpu before the fix) nothing is ever cut.
#[test]
fn v51_prune_schedule_follows_the_full_run_and_resumes() {
    test_letters();
    let c = cfg("--prune");
    let h = c.hidden();
    let (total, stop, end) = (200usize, 80usize, 160usize); // the schedule, the first session's --max-steps, the second's
    let fresh = |seed: u64| {
        let (vm, m) = build_awake(c, seed);
        for i in 0..c.layers {
            set_var(&vm, &format!("b{i}.v51.use"), &Tensor::zeros(h, DType::F32, &CPU).unwrap());
        }
        (vm, m)
    };
    let step = |m: &Model, s: usize| {
        let (x, y) = batch(1, 16, 1000 + s as u64); // a different batch per step: the use averages carry history
        let _ = m.loss_and_grads_autograd(&x, &y, None, false).unwrap();
        m.homeostasis_step().unwrap();
    };
    let snap = |vm: &VarMap| -> Vec<Vec<f32>> {
        (0..c.layers).flat_map(|i| ["alive", "use"].map(|w| var(vm, &format!("b{i}.v51.{w}")).as_tensor().to_vec1::<f32>().unwrap())).collect()
    };
    let (s0, s1) = ((PRUNE_FROM * total as f64).round() as usize, (PRUNE_TO * total as f64).round() as usize);
    // the uninterrupted run
    let (vm, m) = fresh(81);
    let plan = m.v51_begin_schedule(0, total).expect("a plan with --prune");
    assert!(plan.contains(&format!("all {total} steps")) && plan.contains(&format!("from step {s0}")), "{plan}");
    let mut want = Vec::new();
    for s in 0..end {
        step(&m, s);
        want.push(snap(&vm));
        let done = s + 1;
        let live = m.v51_alive().unwrap().unwrap();
        // the counts of the full schedule: cut at s0, every PRUNE_EVERY steps, and at s1 (none before s0)
        let last_due = (s0..=done).rev().find(|&q| (q - s0) % PRUNE_EVERY == 0 || q == s1);
        let n = last_due.map_or(h, |q| h - (prune_share(0.5, q, total) * h as f64).round() as usize);
        assert_eq!(live, vec![n; c.layers], "after {done} steps");
    }
    assert!(m.v51_alive().unwrap().unwrap()[0] < h, "units were cut");
    // the same run in two sessions: stop after `stop` steps, save, a new model loads the checkpoint, the schedule is set
    // again at the resumed step, and it goes on to `end`
    let dir = scratch("v51_prune_resume");
    let path = dir.join("model.safetensors");
    let (vm1, m1) = fresh(81);
    m1.v51_begin_schedule(0, total);
    for s in 0..stop {
        step(&m1, s);
        assert_eq!(snap(&vm1), want[s], "session 1, after {} steps", s + 1);
    }
    vm1.save(&path).unwrap();
    drop((vm1, m1));
    let (mut vm2, m2) = fresh(5); // other weights, replaced by the checkpoint's
    vm2.load(&path).unwrap();
    m2.v51_begin_schedule(stop, total).expect("a plan");
    for s in stop..end {
        step(&m2, s);
        assert_eq!(snap(&vm2), want[s], "session 2 (resumed at {stop}), after {} steps", s + 1);
    }
    // before the fix: no schedule set, so after_step returns at once: the use averages stay 0 and nothing is cut
    let (vm3, m3) = fresh(81);
    for s in 0..s0 + PRUNE_EVERY + 1 {
        step(&m3, s);
    }
    assert_eq!(m3.v51_alive().unwrap().unwrap(), vec![h; c.layers], "no schedule: nothing cut");
    assert!(var(&vm3, "b1.v51.use").as_tensor().to_vec1::<f32>().unwrap().iter().all(|&u| u == 0.0), "no schedule: no use averages");
}

/// v52 fix_Pr (diagnostic, CPU, by hand: `--ignored v52_window_autograd_sensitivity`; ~2 GB, minutes): the yardstick
/// for the fused-vs-autograd numbers at v52's shape (tests/prune_nocut.rs). B with s5-v42's block options at d384, 8
/// blocks, the engram in block 4 and the magnitude code (fixed weights, every zero-start weight awake), one window of
/// 64 and of 2048 tokens (or `AVA_SENS_CTX`, e.g. 256,1024; ctx 2048 needs over 19 GB of RAM on the CPU): how far candle's autograd itself moves (loss, every gradient's cos and rel-norm-diff) when
/// every weight is nudged by 1e-6 of itself. Printed only.
#[test]
#[ignore]
fn v52_window_autograd_sensitivity() {
    const S5: &str = "--head-size 64 --timescales fixed --gates channel --value-residual --kwta 1/4 --mlp-act relu2 --mem-momentum 0.5 \
                      --resid-fix --residual dense --lif-init logspaced --lif-scale --spikes int2 --alif --homeostasis 0.1";
    let g = ava_data::numbers::VOCAB_GLANCE;
    let c = Config::with_kind(Kind::D, g, 384, 8, 2048).with_glance().unwrap().with_mag().unwrap().with_engram(4).unwrap().with_arch(Arch::from_flags(S5).unwrap()).unwrap();
    let (vm, m) = build_awake(c, 52);
    let names: Vec<(String, Var)> = {
        let d = vm.data().lock().unwrap();
        let mut v: Vec<(String, Var)> = d.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    };
    let f = |t: &Tensor| t.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let sc = |t: Tensor| t.to_scalar::<f32>().unwrap();
    let ctxs: Vec<usize> = std::env::var("AVA_SENS_CTX").unwrap_or_else(|_| "64,2048".into()).split(',').map(|s| s.trim().parse().unwrap()).collect();
    for t in ctxs {
        let v = tokens(1, t + 1, g, 93);
        let x = Tensor::from_vec(v[..t].to_vec(), (1, t), &CPU).unwrap();
        let y = Tensor::from_vec(v[1..].to_vec(), (1, t), &CPU).unwrap();
        let t0 = std::time::Instant::now();
        let (l0, _, g0) = m.loss_and_grads_autograd(&x, &y, None, false).unwrap();
        let keep: Vec<Tensor> = names.iter().map(|(_, var)| var.as_tensor().copy().unwrap()).collect();
        for (k, ((_, var), w)) in names.iter().zip(&keep).enumerate() {
            let z = Tensor::from_vec(unit_noise(w.elem_count(), 9000 + k as u64), w.shape(), &CPU).unwrap().affine(1e-6, 0.0).unwrap();
            var.set(&w.add(&w.mul(&z).unwrap()).unwrap()).unwrap();
        }
        let (l1, _, g1) = m.loss_and_grads_autograd(&x, &y, None, false).unwrap();
        for ((_, var), w) in names.iter().zip(&keep) {
            var.set(w).unwrap();
        }
        let mut rows: Vec<(f32, f32, String)> = Vec::new();
        for (name, var) in &names {
            let (Some(a), Some(b)) = (g0.get(var.as_tensor()), g1.get(var.as_tensor())) else { continue };
            let (a, b) = (f(a), f(b));
            let na = sc(a.sqr().unwrap().sum_all().unwrap()).sqrt();
            let nb = sc(b.sqr().unwrap().sum_all().unwrap()).sqrt();
            if na == 0.0 && nb == 0.0 {
                continue;
            }
            let cos = sc(a.mul(&b).unwrap().sum_all().unwrap()) / (na * nb).max(1e-30);
            let rel = sc(a.sub(&b).unwrap().sqr().unwrap().sum_all().unwrap()).sqrt() / na.max(1e-30);
            rows.push((cos, rel, name.clone()));
        }
        rows.sort_by(|p, q| p.0.total_cmp(&q.0));
        let below = rows.iter().filter(|r| !(r.0 > 0.9999 && r.1 < 1e-2)).count();
        eprintln!(
            "ctx {t}: autograd loss {:.7} -> {:.7} after a 1e-6 nudge ({:.1} s for both); {} gradients, {below} outside cos > 0.9999 and rel < 1e-2",
            sc(l0),
            sc(l1),
            t0.elapsed().as_secs_f64(),
            rows.len()
        );
        for (cos, rel, name) in rows.iter().take(12) {
            eprintln!("ctx {t}:   {name:<24} cos {cos:.6} rel-norm-diff {rel:.3e}");
        }
    }
}

/// The cut units removed for real (`compact_pruned`): a plain model with hidden 2d gives the same logits, in a forward
/// and in a one-token stream, and it is a model the fused step may take.
#[test]
fn v51_compacted_model_is_the_pruned_model() {
    let (c, vm, m) = build("--prune", 71);
    let (x, _) = batch(2, 60, 17);
    let want = m.forward(&x).unwrap();
    let w: std::collections::HashMap<String, Tensor> = vm.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.as_tensor().copy().unwrap())).collect();
    let (c2, w2) = compact_pruned(&c, &w).unwrap();
    assert_eq!(c2.hidden(), 2 * c.d);
    assert!(!c2.arch.v51.any());
    let m2 = Model::new(c2, VarBuilder::from_tensors(w2, DType::F32, &CPU)).unwrap();
    let got = m2.forward(&x).unwrap();
    let d = max_diff(&want, &got);
    assert!(d < 1e-5, "compacted vs masked: {d}");
    // per block: half of W1 and of W2 (d·hidden in all) and the mask and use averages (2·hidden) are gone
    assert_eq!(c2.weight_counts().0 + c.layers * (c.d * c.hidden() + 2 * c.hidden()), c.weight_counts().0);
    let toks: Vec<u32> = x.get(0).unwrap().to_vec1().unwrap();
    let mut s = m2.stream().unwrap();
    let outs: Vec<Tensor> = toks.iter().map(|&tk| m2.step(&mut s, tk).unwrap()).collect();
    let d = max_diff(&Tensor::stack(&outs, 0).unwrap(), &want.reshape((2, 60, ())).unwrap().get(0).unwrap());
    assert!(d < 1e-4, "compacted stream: {d}");
    eprintln!("v51 compacted model: fused step {:?}", m2.fused_blocker());
}
