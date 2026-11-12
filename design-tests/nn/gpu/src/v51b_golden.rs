// TESTING / DESIGN ONLY — v51 Part B2 (runs/v51/prereg_v51.md), CPU unit test: with the new options off, every number is what it was.
//
// One run on the code before the v51b edits writes a golden file (V51B_GOLDEN_WRITE=FILE); the same test on the edited
// code compares against it (V51B_GOLDEN_READ=FILE), bit for bit: for Ava 0.4's block recipe (with an engram), the v50
// base recipe (WIN + SLOW + glance, no engram) and today's plain D, the forward logits, the training loss and every
// parameter's gradient (candle's autograd, the reference path), the one-token stream's logits, and the int8 + fused
// talking step's logits. CPU only; without either variable the test only checks that everything runs.

use crate::arch::Arch;
use crate::model::{Config, Kind, Model};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::fmt::Write as _;

const RECIPE: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense \
                      --resid-fix --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif \
                      --homeostasis 0.1";

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let u = (x >> 11) as f64 / (1u64 << 53) as f64;
            ((u * 2.0 - 1.0) * 3f64.sqrt()) as f32
        })
        .collect()
}

/// Fixed weights: every random weight gets noise of its own spread, every all-zero weight small noise (so every part
/// is exercised), constant non-zero weights (gains, leaks, thresholds) stay.
fn build(cfg: Config, seed: u64) -> (VarMap, Model) {
    let vm = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu)).unwrap();
    {
        let data = vm.data().lock().unwrap();
        let mut names: Vec<&String> = data.keys().collect();
        names.sort();
        for (k, name) in names.into_iter().enumerate() {
            let var = &data[name];
            let v: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
            let n = v.len() as f64;
            let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
            let sd = (v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
            let zero = v.iter().all(|&x| x == 0.0);
            if sd == 0.0 && !zero {
                continue;
            }
            // the measured spread snapped to the nearest init spread (candle's CPU init is not seeded), as testkit does
            let spreads = [crate::model::INIT_STD, 1.0 / (cfg.d as f64).sqrt(), cfg.resid_std()];
            let snap = spreads.iter().copied().min_by(|a, b| (a.ln() - sd.ln()).abs().total_cmp(&(b.ln() - sd.ln()).abs())).unwrap();
            let s = if zero { 0.1 } else { snap };
            let w: Vec<f32> = noise(v.len(), seed.wrapping_mul(1000).wrapping_add(k as u64)).iter().map(|&z| (s * z as f64) as f32).collect();
            var.set(&Tensor::from_vec(w, var.shape(), &Device::Cpu).unwrap()).unwrap();
        }
    }
    m.set_track_activity(false);
    (vm, m)
}

fn fnv(t: &Tensor) -> (u64, usize) {
    let v: Vec<f32> = t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in &v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    (h, v.len())
}

fn line(out: &mut String, tag: &str, t: &Tensor) {
    let (h, n) = fnv(t);
    writeln!(out, "{tag}\t{h:016x}\t{n}").unwrap();
}

fn tokens(b: usize, t: usize, seed: u64) -> Vec<u32> {
    noise(b * t, seed).iter().map(|z| (((z + 1.8) / 3.6).clamp(0.0, 0.999) * 250.0) as u32).collect()
}

fn golden(name: &str, cfg: Config, out: &mut String) {
    let (vm, m) = build(cfg, 50);
    let (b, t) = (2, 32);
    let toks = tokens(b, t + 1, 7);
    let ids: Vec<u32> = (0..b).flat_map(|r| toks[r * (t + 1)..r * (t + 1) + t].to_vec()).collect();
    let tg: Vec<u32> = (0..b).flat_map(|r| toks[r * (t + 1) + 1..(r + 1) * (t + 1)].to_vec()).collect();
    let ids = Tensor::from_vec(ids, (b, t), &Device::Cpu).unwrap();
    let tg = Tensor::from_vec(tg, (b, t), &Device::Cpu).unwrap();
    line(out, &format!("{name} forward"), &m.forward(&ids).unwrap());
    let (total, _, _) = m.loss_aux(&ids, &tg, 0.0).unwrap();
    line(out, &format!("{name} loss"), &total);
    let grads = total.backward().unwrap();
    let data = vm.data().lock().unwrap();
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    for n in names {
        match grads.get(data[n].as_tensor()) {
            Some(g) => line(out, &format!("{name} grad {n}"), g),
            None => writeln!(out, "{name} grad {n}\tnone").unwrap(),
        }
    }
    drop(data);
    let row: Vec<u32> = toks[..24].to_vec();
    let mut s = m.stream().unwrap();
    for (j, &tok) in row.iter().enumerate() {
        let l = m.step(&mut s, tok).unwrap();
        if j % 4 == 3 {
            line(out, &format!("{name} stream {j}"), &l);
        }
    }
    let mut s = m.stream_batched().unwrap();
    line(out, &format!("{name} feed"), &m.feed(&mut s, &row).unwrap());
    // v51b: two rows read in pieces with carried state (forward_with_state, the path the online learning hooks sit on)
    let mut st = m.state(b).unwrap();
    let mut p0 = 0;
    for (k, len) in [13usize, 19].into_iter().enumerate() {
        let piece: Vec<u32> = (0..b).flat_map(|r| toks[r * (t + 1) + p0..r * (t + 1) + p0 + len].to_vec()).collect();
        let l = m.forward_with_state(&Tensor::from_vec(piece, (b, len), &Device::Cpu).unwrap(), &mut st).unwrap();
        line(out, &format!("{name} pieces {k}"), &l);
        p0 += len;
    }
    m.set_int8(true).unwrap();
    m.set_fused(true);
    let mut s = m.stream().unwrap();
    for (j, &tok) in row.iter().enumerate() {
        let l = m.step(&mut s, tok).unwrap();
        if j % 4 == 3 {
            line(out, &format!("{name} int8+fused {j}"), &l);
        }
    }
    m.set_fused(false);
    m.set_int8(false).unwrap();
}

#[test]
fn option_off_numbers_unchanged() {
    let recipe = Arch::from_flags(RECIPE).unwrap();
    let mut out = String::new();
    let ava04 = Config::with_kind(Kind::D, 258, 64, 2, 64).with_engram(1).unwrap().with_arch(recipe).unwrap();
    golden("ava04", ava04, &mut out);
    let v50 = Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 64, 2, 64).with_glance().unwrap().with_arch(recipe).unwrap();
    golden("v50base", v50, &mut out);
    golden("plainD", Config::with_kind(Kind::D, 258, 64, 2, 64), &mut out);
    // v51b: round 1's options on (their code paths share the hooks)
    let r1 = Arch::from_flags(&format!("{RECIPE} --dale 0.8 --loop 2 --neuromod 3")).unwrap();
    golden("round1", Config::with_kind(Kind::D, 258, 64, 2, 64).with_engram(1).unwrap().with_arch(r1).unwrap(), &mut out);
    if let Ok(p) = std::env::var("V51B_GOLDEN_WRITE") {
        std::fs::write(&p, &out).unwrap();
        eprintln!("v51b golden: {} lines written to {p}", out.lines().count());
    }
    if let Ok(p) = std::env::var("V51B_GOLDEN_READ") {
        let want = std::fs::read_to_string(&p).unwrap();
        let (mut same, mut diff) = (0, Vec::new());
        for (a, b) in want.lines().zip(out.lines()) {
            if a == b {
                same += 1;
            } else {
                diff.push(format!("was {a} / now {b}"));
            }
        }
        assert_eq!(want.lines().count(), out.lines().count(), "line count");
        assert!(diff.is_empty(), "{} of {} lines differ:\n{}", diff.len(), same + diff.len(), diff.join("\n"));
        eprintln!("v51b golden: all {same} lines bit-identical to {p}");
    }
}
