// TESTING / DESIGN ONLY — tests of src/fwin_r2.rs (v51 fast1: round 2's P, DA, M, Y, Pr and S on the fused WIN step).
//
// CPU only (the fused step itself needs CUDA: tests/fast_winners.rs round2_step_matches_autograd, run by
// runs/v51/fast1_check.sh). Here every backward piece of fwin_r2 against candle's autograd of the reference path's own
// ops (src/v51.rs, scan_fast::stp_forward), and the forward pieces against the reference ops bit for bit:
// * the pure pieces (scans, gains, the norm, the gates) in f64 (within 1e-10 relative) and f32 (within 2e-5; 1e-4 for
//   the long scans' sums);
// * each part's hooks on a small CPU model with all six parts on (f32, within 2e-5), the zero-start weights moved off 0;
// * the reverse scans against plain loops.

use super::*;
use crate::model::{Config, Kind, VOCAB};
use crate::testkit::unit_noise;
use crate::v51::gain;
use candle_core::Var;
use candle_nn::{VarBuilder, VarMap};

fn noise(dev: &Device, shape: &[usize], seed: u64, scale: f64, offset: f64, dt: DType) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec(unit_noise(n, seed), shape, dev).unwrap().to_dtype(dt).unwrap().affine(scale, offset).unwrap()
}

/// max |a - b| / max(max |a|, floor), in f64.
fn rel(a: &Tensor, b: &Tensor, floor: f64) -> f64 {
    let a = a.to_dtype(DType::F64).unwrap().flatten_all().unwrap();
    let b = b.to_dtype(DType::F64).unwrap().flatten_all().unwrap();
    assert_eq!(a.dims(), b.dims(), "shapes");
    let d = a.sub(&b).unwrap().abs().unwrap().max(0).unwrap().to_scalar::<f64>().unwrap();
    let m = a.abs().unwrap().max(0).unwrap().to_scalar::<f64>().unwrap();
    d / m.max(floor)
}

fn same(a: &Tensor, b: &Tensor) -> bool {
    rel(a, b, 1e-30) == 0.0
}

fn tol(dt: DType) -> f64 {
    if dt == DType::F64 {
        1e-10
    } else {
        2e-5
    }
}

fn check(tag: &str, want: &Tensor, got: &Tensor, bar: f64) {
    let e = rel(want, got, 1e-30);
    eprintln!("{tag}: rel {e:.2e} (bar {bar:.0e})");
    assert!(e <= bar, "{tag}: rel {e} over {bar}");
}

fn empty_grads(dev: &Device) -> GradStore {
    Tensor::zeros((), DType::F32, dev).unwrap().backward().unwrap()
}

// ------------------------------------------------------------------------------------------- the pure pieces

#[test]
fn scans_match_the_recurrences() {
    let cpu = Device::Cpu;
    for t in [1usize, 5, 150] {
        let (b, o) = (2usize, 3usize);
        let x = noise(&cpu, &[b, t, o], 1, 1.0, 0.0, DType::F64);
        let lc = noise(&cpu, &[b, t, o], 2, 1.0, 0.0, DType::F64).abs().unwrap().affine(-0.5, 0.0).unwrap(); // ln c <= 0
        let (xv, cv) = (x.to_vec3::<f64>().unwrap(), lc.to_vec3::<f64>().unwrap());
        let (mut fw, mut bw) = (vec![vec![vec![0f64; o]; t]; b], vec![vec![vec![0f64; o]; t]; b]);
        for bi in 0..b {
            for c in 0..o {
                let mut y = 0.0;
                for ti in 0..t {
                    y = xv[bi][ti][c] + cv[bi][ti][c].exp() * y;
                    fw[bi][ti][c] = y;
                }
                let mut g = 0.0;
                for ti in (0..t).rev() {
                    let next = if ti + 1 < t { cv[bi][ti + 1][c].exp() * g } else { 0.0 };
                    g = xv[bi][ti][c] + next;
                    bw[bi][ti][c] = g;
                }
            }
        }
        let flat = |v: Vec<Vec<Vec<f64>>>| Tensor::from_vec(v.into_iter().flatten().flatten().collect::<Vec<f64>>(), (b, t, o), &cpu).unwrap();
        check(&format!("lin_scan t {t}"), &flat(fw), &lin_scan(&x, &lc).unwrap(), 1e-12);
        check(&format!("rev_scan t {t}"), &flat(bw), &rev_scan(&x, &lc).unwrap(), 1e-12);
    }
}

#[test]
fn mood_scan_backward_matches_autograd() {
    let cpu = Device::Cpu;
    for dt in [DType::F64, DType::F32] {
        let (b, t, k) = (2usize, 150usize, 4usize);
        let u = Var::from_tensor(&noise(&cpu, &[b, t, k], 3, 1.0, 0.2, dt)).unwrap();
        let l0: Vec<f32> = (0..k).map(|j| crate::model::logit(1.0 - 1.0 / (16.0 * 4f64.powi(j as i32))) as f32).collect();
        let lam = Var::from_tensor(&Tensor::from_vec(l0, k, &cpu).unwrap().to_dtype(dt).unwrap().add(&noise(&cpu, &[k], 4, 0.3, 0.0, dt)).unwrap()).unwrap();
        let r = noise(&cpu, &[b, t, k], 5, 1.0, 0.0, dt);
        let m = crate::v51::mood_scan(u.as_tensor(), lam.as_tensor(), &Tensor::zeros((b, k), dt, &cpu).unwrap()).unwrap();
        let gr = m.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
        let (du, dlam) = mood_scan_bwd(&r, u.as_tensor(), &m, lam.as_tensor()).unwrap();
        let bar = if dt == DType::F64 { 1e-10 } else { 1e-4 };
        check(&format!("mood scan {dt:?} du"), gr.get(u.as_tensor()).unwrap(), &du, bar);
        check(&format!("mood scan {dt:?} dlam"), gr.get(lam.as_tensor()).unwrap(), &dlam, bar);
    }
}

#[test]
fn rhythm_backward_matches_autograd() {
    let cpu = Device::Cpu;
    for dt in [DType::F64, DType::F32] {
        let (b, t) = (2usize, 150usize);
        // the resets' logits around -3 with a wide spread: r from ~0 to ~1
        let pre = Var::from_tensor(&noise(&cpu, &[b, t, 2], 6, 4.0, -3.0, dt)).unwrap();
        let w: Vec<f32> = crate::v51::RHYTHM_PERIODS.iter().map(|p| (2.0 * std::f64::consts::PI / p).ln() as f32).collect();
        let lw = Var::from_tensor(&Tensor::from_vec(w, 2, &cpu).unwrap().to_dtype(dt).unwrap()).unwrap();
        let r = noise(&cpu, &[b * t, 4], 7, 1.0, 0.0, dt);
        let a = crate::v51::rhythm_scan(pre.as_tensor(), &Tensor::zeros((b, 2), dt, &cpu).unwrap()).unwrap();
        let f = crate::v51::rhythm_feats(&a, lw.as_tensor()).unwrap();
        let gr = f.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
        let (dpre, dlw) = rhythm_bwd(&r, pre.as_tensor(), &a, lw.as_tensor()).unwrap();
        let bar = if dt == DType::F64 { 1e-9 } else { 2e-4 };
        check(&format!("rhythm {dt:?} dpre"), gr.get(pre.as_tensor()).unwrap(), &dpre, bar);
        check(&format!("rhythm {dt:?} dlw"), gr.get(lw.as_tensor()).unwrap(), &dlw, bar);
    }
}

#[test]
fn gains_norm_and_gates_match_autograd() {
    let cpu = Device::Cpu;
    for dt in [DType::F64, DType::F32] {
        // y = x (1 + tanh p)
        let (n, m) = (13usize, 24usize);
        let x = Var::from_tensor(&noise(&cpu, &[n, m], 8, 1.5, 0.1, dt)).unwrap();
        let p = Var::from_tensor(&noise(&cpu, &[n, m], 9, 2.0, 0.0, dt)).unwrap();
        let r = noise(&cpu, &[n, m], 10, 1.0, 0.0, dt);
        let y = x.as_tensor().mul(&gain(p.as_tensor()).unwrap()).unwrap();
        let (g2, th) = gain_th(p.as_tensor()).unwrap();
        assert!(same(&y, &x.as_tensor().mul(&g2).unwrap()), "gain_th is v51::gain bit for bit");
        let gr = y.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
        let (dx, dp) = gain_bwd(&r, x.as_tensor(), &th).unwrap();
        check(&format!("gain {dt:?} dx"), gr.get(x.as_tensor()).unwrap(), &dx, tol(dt));
        check(&format!("gain {dt:?} dp"), gr.get(p.as_tensor()).unwrap(), &dp, tol(dt));
        // hn: rms norm without gain
        let hn = head_norm(x.as_tensor()).unwrap();
        let gr = hn.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
        check(&format!("hn {dt:?} dx"), gr.get(x.as_tensor()).unwrap(), &hn_bwd(x.as_tensor(), &r).unwrap(), tol(dt));
        // Y's write gate on a scan input vg [B·H, T, hs + 2G] (G = 1 per head and G = hs per channel)
        let (b, nh, t, hs) = (2usize, 3usize, 7usize, 4usize);
        for g in [1usize, hs] {
            let w = hs + 2 * g;
            let vg0 = noise(&cpu, &[b * nh, t, w], 11, 1.0, 0.5, dt);
            let beta = Var::from_tensor(&vg0.narrow(2, w - g, g).unwrap().contiguous().unwrap().reshape((b, nh, t, g)).unwrap()).unwrap();
            let gate = Var::from_tensor(&noise(&cpu, &[b, nh, t, 1], 12, 0.5, 1.0, dt)).unwrap();
            let r = noise(&cpu, &[b, nh, t, g], 13, 1.0, 0.0, dt);
            let out = beta.as_tensor().broadcast_mul(gate.as_tensor()).unwrap();
            let gr = out.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
            let vg = vg0.copy().unwrap();
            let lay = gate.as_tensor().reshape((b * nh, t, 1)).unwrap();
            let bu = wgate_fwd(&vg, &lay, g).unwrap();
            assert!(same(&bu, beta.as_tensor()), "the ungated strengths");
            assert!(same(&vg.narrow(2, w - g, g).unwrap(), &out.reshape((b * nh, t, g)).unwrap()), "the gated strengths in place");
            assert!(same(&vg.narrow(2, 0, w - g).unwrap(), &vg0.narrow(2, 0, w - g).unwrap()), "the other columns untouched");
            let dvg = Tensor::zeros((b * nh, t, w), dt, &cpu).unwrap();
            dvg.slice_set(&r.reshape((b * nh, t, g)).unwrap(), 2, w - g).unwrap();
            let dgate = wgate_bwd(&dvg, &bu, &lay, g).unwrap();
            check(&format!("write gate {dt:?} G {g} dbeta"), &gr.get(beta.as_tensor()).unwrap().reshape((b * nh, t, g)).unwrap(), &dvg.narrow(2, w - g, g).unwrap(), tol(dt));
            check(&format!("write gate {dt:?} G {g} dgate"), &gr.get(gate.as_tensor()).unwrap().reshape((b * nh, t, 1)).unwrap(), &dgate, tol(dt));
        }
        // Y's read gate: per head on [n, d] (head-major columns)
        let o = noise(&cpu, &[b * t, nh * hs], 14, 1.0, 0.0, dt);
        let gr_ = noise(&cpu, &[b * t, nh], 15, 0.5, 1.0, dt);
        let want = o.reshape((b * t, nh, hs)).unwrap().broadcast_mul(&gr_.unsqueeze(2).unwrap()).unwrap().reshape((b * t, nh * hs)).unwrap();
        assert!(same(&want, &rgate_fwd(&o, &gr_).unwrap()), "the read gate per head");
    }
}

// ------------------------------------------------------------------------------------------- the hooks, on a CPU model

const ALL: &str = "--head-size 64 --kwta 0.25 --mlp-act relu2 --dendrite --dopamine --mood --rhythm --prune --stp";

fn arch_of(flags: &str) -> crate::arch::Arch {
    let args: Vec<String> = flags.split_whitespace().map(String::from).collect();
    let mut a = crate::arch::Arch::default();
    let mut i = 0;
    while i < args.len() {
        i += a.take_flag(&args, i).unwrap().unwrap_or_else(|| panic!("not a block flag: {}", args[i]));
    }
    a
}

/// A d128 L3 model (2 heads) with all six parts, every v51 weight and the synapse's mix moved off its start.
fn cpu_model() -> (VarMap, Model) {
    let cpu = Device::Cpu;
    let cfg = Config::with_kind(Kind::D, VOCAB, 128, 3, 16).with_arch(arch_of(ALL)).unwrap();
    let varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &cpu)).unwrap();
    let mut names: Vec<(String, Var)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    names.sort_by(|a, b| a.0.cmp(&b.0));
    for (k, (name, var)) in names.into_iter().enumerate() {
        let t = var.as_tensor();
        let moved = name.contains("v51") && !name.ends_with(".alive") && !name.ends_with(".use") || name.ends_with("stp.mix");
        if moved {
            let amp = if name.ends_with(".br") { 1.0 } else { 0.4 };
            var.set(&t.add(&noise(&cpu, t.dims(), 100 + k as u64, amp, 0.0, DType::F32)).unwrap()).unwrap();
        }
    }
    (varmap, model)
}

fn v(m: &Model) -> &crate::v51::V51Ext {
    &m.ext.as_ref().unwrap().v51
}

#[test]
fn signals_are_the_reference_forwards() {
    // Y's phases (v51::begin), DA's surprise and M's mood (v51::after0), bit for bit
    let cpu = Device::Cpu;
    let (_, m) = cpu_model();
    let (b, t, d) = (2usize, 70usize, 128usize);
    let x = noise(&cpu, &[b * t, d], 20, 1.0, 0.0, DType::F32);
    let ids = Tensor::from_vec((0..(b * t) as u32).map(|i| (i * 37 + 5) % VOCAB as u32).collect::<Vec<u32>>(), (b, t), &cpu).unwrap();
    crate::v51::begin(&m, &x, b, t, None).unwrap();
    let rh_ref = v(&m).sig.borrow().rh.clone().unwrap();
    let mut f = begin(&m, &x, b, t).unwrap();
    assert!(same(&rh_ref, &f.rh.as_ref().unwrap().f), "Y's phases are v51::begin's");
    let x0 = noise(&cpu, &[b * t, d], 21, 1.0, 0.0, DType::F32);
    crate::v51::after0(&m, &x0, &ids, b, t, None).unwrap();
    let (z_ref, m_ref) = {
        let s = v(&m).sig.borrow();
        (s.z.clone().unwrap(), s.mood.clone().unwrap())
    };
    crate::v51::end(&m);
    after0(&m, &mut f, &x0, &ids, &m.tok.detach()).unwrap();
    assert!(same(&z_ref, f.z.as_ref().unwrap()), "the surprise is v51::after0's");
    assert!(same(&m_ref, &f.mood.as_ref().unwrap().m2), "the mood is v51::after0's");
    let z = f.z.as_ref().unwrap().to_vec2::<f32>().unwrap();
    assert!(z[0][0] == 0.0 && z[t][0] == 0.0 && z[1][0] > 0.0, "0 at each stream's start, a surprise after");
}

#[test]
fn part_hooks_match_autograd() {
    let cpu = Device::Cpu;
    let (_, m) = cpu_model();
    let (b, t, d) = (2usize, 70usize, 128usize);
    let n = b * t;
    let nh = d / 64;
    let hid = m.cfg.hidden();
    let i = 1; // a block after 0 (DA and M act there)
    let bx = &v(&m).blocks[i];
    let bar = tol(DType::F32);
    // the signals, as the step makes them
    let x = noise(&cpu, &[n, d], 30, 1.0, 0.0, DType::F32);
    let ids = Tensor::from_vec((0..n as u32).map(|i| (i * 53 + 11) % VOCAB as u32).collect::<Vec<u32>>(), (b, t), &cpu).unwrap();
    let mut f = begin(&m, &x, b, t).unwrap();
    after0(&m, &mut f, &noise(&cpu, &[n, d], 31, 1.0, 0.0, DType::F32), &ids, &m.tok.detach()).unwrap();
    // ---- DA: y (1 + tanh(z w + c))
    {
        let y = Var::from_tensor(&noise(&cpu, &[n, d], 32, 1.0, 0.0, DType::F32)).unwrap();
        let (w, c) = bx.da.as_ref().unwrap();
        let z = f.z.as_ref().unwrap();
        let want = y.as_tensor().mul(&gain(&z.broadcast_mul(&w.unsqueeze(0).unwrap()).unwrap().broadcast_add(c).unwrap()).unwrap()).unwrap();
        let mut sv = Blk::default();
        let got = da_fwd(&m, i, &f, y.as_tensor().detach(), &mut sv).unwrap();
        assert!(same(&want, &got), "DA's forward");
        let r = noise(&cpu, &[n, d], 33, 1.0, 0.0, DType::F32);
        let gr = want.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
        let mut gs = empty_grads(&cpu);
        let dy = da_bwd(&m, i, &f, &sv, &r, &mut gs).unwrap();
        check("DA dy", gr.get(y.as_tensor()).unwrap(), &dy, bar);
        check("DA dw", gr.get(w).unwrap(), gs.get(w).unwrap(), bar);
        check("DA dc", gr.get(c).unwrap(), gs.get(c).unwrap(), bar);
    }
    // ---- M: cur (1 + tanh(m G)) on two blocks, then the scan (lam, win, bin) through `finish`
    {
        let mp = v(&m).mood.as_ref().unwrap();
        let k = mp.lam.dim(0).unwrap();
        let z = f.z.as_ref().unwrap();
        let u = z.reshape((b, t)).unwrap().unsqueeze(2).unwrap().broadcast_mul(&mp.win.reshape((1, 1, k)).unwrap()).unwrap().broadcast_add(&mp.bin.reshape((1, 1, k)).unwrap()).unwrap();
        let mood = crate::v51::mood_scan(&u, &mp.lam, &Tensor::zeros((b, k), DType::F32, &cpu).unwrap()).unwrap().reshape((n, k)).unwrap();
        let mut loss = Tensor::zeros((), DType::F32, &cpu).unwrap();
        let mut gs = empty_grads(&cpu);
        let mut bw = Bwd::default();
        let mut curs = Vec::new();
        for j in [1usize, 2] {
            let g = v(&m).blocks[j].mood_g.as_ref().unwrap();
            let cur = Var::from_tensor(&noise(&cpu, &[b, t, d], 40 + j as u64, 1.0, 0.0, DType::F32)).unwrap();
            let want = cur.as_tensor().mul(&gain(&mood.matmul(g).unwrap()).unwrap().reshape((b, t, d)).unwrap()).unwrap();
            let mut sv = Blk::default();
            let got = mood_fwd(&m, j, &f, cur.as_tensor().detach(), &mut sv).unwrap();
            assert!(same(&want, &got), "M's forward, block {j}");
            let r = noise(&cpu, &[b, t, d], 50 + j as u64, 1.0, 0.0, DType::F32);
            loss = loss.add(&want.mul(&r).unwrap().sum_all().unwrap()).unwrap();
            let dcur = mood_bwd(&m, j, &f, &sv, &r.reshape((n, d)).unwrap(), &mut bw, &mut gs).unwrap();
            curs.push((cur, dcur, g.clone()));
        }
        assert!(mood_fwd(&m, 0, &f, Tensor::zeros((b, t, d), DType::F32, &cpu).unwrap(), &mut Blk::default()).is_ok(), "block 0: no gain");
        let gr = loss.backward().unwrap();
        for (j, (cur, dcur, g)) in curs.iter().enumerate() {
            check(&format!("M block {} dcur", j + 1), &gr.get(cur.as_tensor()).unwrap().reshape((n, d)).unwrap(), dcur, bar);
            check(&format!("M block {} dG", j + 1), gr.get(g).unwrap(), gs.get(g).unwrap(), bar);
        }
        let dx = finish(&m, &f, Bwd { df: None, dm: bw.dm }, &x, &mut gs).unwrap();
        assert!(dx.is_none(), "no df: no rhythm gradient");
        for (name, var) in [("lam", &mp.lam), ("win", &mp.win), ("bin", &mp.bin)] {
            check(&format!("M d{name}"), gr.get(var).unwrap(), gs.get(var).unwrap(), 1e-4);
        }
    }
    // ---- Y: the write and read gates of every block, through the phases back to the embedding (finish)
    {
        let rp = v(&m).rhythm.as_ref().unwrap();
        let xv = Var::from_tensor(&x).unwrap();
        let pre = head_norm(xv.as_tensor()).unwrap().matmul(&rp.wr).unwrap().broadcast_add(&rp.br).unwrap().reshape((b, t, 2)).unwrap();
        let a = crate::v51::rhythm_scan(&pre, &Tensor::zeros((b, 2), DType::F32, &cpu).unwrap()).unwrap();
        let fr = crate::v51::rhythm_feats(&a, &rp.lw).unwrap();
        assert!(same(&fr, &f.rh.as_ref().unwrap().f), "Y's phases");
        let (g, hs) = (1usize, 64usize);
        let w = hs + 2 * g;
        let mut loss = Tensor::zeros((), DType::F32, &cpu).unwrap();
        let mut gs = empty_grads(&cpu);
        let mut bw = Bwd::default();
        let mut wants = Vec::new();
        for j in 0..3usize {
            let (gw, gr_) = v(&m).blocks[j].rh.as_ref().unwrap();
            // the reference: v51::rh_gate's ops on beta [b, nh, t, g] and on the read [b, nh, t, hs] (as [n, d] here)
            let gate_w = gain(&fr.matmul(gw).unwrap()).unwrap().reshape((b, t, nh)).unwrap().permute((0, 2, 1)).unwrap().unsqueeze(3).unwrap();
            let gate_r = gain(&fr.matmul(gr_).unwrap()).unwrap(); // [n, nh]
            let vg0 = noise(&cpu, &[b * nh, t, w], 60 + j as u64, 1.0, 0.5, DType::F32);
            let beta = vg0.narrow(2, w - g, g).unwrap().contiguous().unwrap().reshape((b, nh, t, g)).unwrap();
            let o = noise(&cpu, &[n, d], 70 + j as u64, 1.0, 0.0, DType::F32);
            let r1 = noise(&cpu, &[b, nh, t, g], 80 + j as u64, 1.0, 0.0, DType::F32);
            let r2 = noise(&cpu, &[n, d], 90 + j as u64, 1.0, 0.0, DType::F32);
            let og = o.reshape((n, nh, hs)).unwrap().broadcast_mul(&gate_r.unsqueeze(2).unwrap()).unwrap().reshape((n, d)).unwrap();
            loss = loss.add(&beta.broadcast_mul(&gate_w).unwrap().mul(&r1).unwrap().sum_all().unwrap()).unwrap();
            loss = loss.add(&og.mul(&r2).unwrap().sum_all().unwrap()).unwrap();
            // the hooks
            let mut sv = Blk::default();
            let lay = wgate(&m, j, &f, nh, &mut sv).unwrap().unwrap();
            assert!(same(&lay, &gate_w.reshape((b * nh, t, 1)).unwrap()), "the write gate's layout, block {j}");
            let vg = vg0.copy().unwrap();
            let bu = wgate_fwd(&vg, &lay, g).unwrap();
            let (gr2, th) = rgate(&m, j, &f).unwrap().unwrap();
            assert!(same(&gr2, &gate_r), "the read gate, block {j}");
            sv.rg = Some((gr2, th, o.clone()));
            let d_o = rgate_bwd(&m, j, &f, &sv, &r2, &mut bw, &mut gs).unwrap();
            check(&format!("Y block {j} d(read)"), &r2.reshape((n, nh, hs)).unwrap().broadcast_mul(&gate_r.unsqueeze(2).unwrap()).unwrap().reshape((n, d)).unwrap(), &d_o, bar);
            let dvg = Tensor::zeros((b * nh, t, w), DType::F32, &cpu).unwrap();
            dvg.slice_set(&r1.reshape((b * nh, t, g)).unwrap(), 2, w - g).unwrap();
            let dgate = wgate_bwd(&dvg, &bu, &lay, g).unwrap();
            wgate_bwd_params(&m, j, &f, &sv, &dgate, &mut bw, &mut gs).unwrap();
            wants.push((gw.clone(), gr_.clone()));
        }
        let gr = loss.backward().unwrap();
        for (j, (gw, gr_)) in wants.iter().enumerate() {
            check(&format!("Y block {j} d rh_w"), gr.get(gw).unwrap(), gs.get(gw).unwrap(), bar);
            check(&format!("Y block {j} d rh_r"), gr.get(gr_).unwrap(), gs.get(gr_).unwrap(), bar);
        }
        let dx = finish(&m, &f, Bwd { df: bw.df, dm: None }, &x, &mut gs).unwrap().expect("Y's embedding gradient");
        check("Y dx", gr.get(xv.as_tensor()).unwrap(), &dx, 1e-4);
        for (name, var) in [("lw", &rp.lw), ("wr", &rp.wr), ("br", &rp.br)] {
            check(&format!("Y d{name}"), gr.get(var).unwrap(), gs.get(var).unwrap(), 1e-4);
        }
    }
    // ---- P: a (1 + tanh(hn(c) A B)), c the memory read
    {
        let (wa, wb) = bx.dend.as_ref().unwrap();
        let a = Var::from_tensor(&noise(&cpu, &[n, hid], 100, 1.0, 0.0, DType::F32).relu().unwrap()).unwrap();
        let ctx = Var::from_tensor(&noise(&cpu, &[n, d], 101, 1.0, 0.0, DType::F32)).unwrap();
        let want = a.as_tensor().mul(&gain(&head_norm(ctx.as_tensor()).unwrap().matmul(wa).unwrap().matmul(wb).unwrap()).unwrap()).unwrap();
        let mut sv = Blk { ctx: Some(ctx.as_tensor().detach()), ..Default::default() };
        let got = dend_fwd(&m, i, a.as_tensor().detach(), &mut sv).unwrap();
        assert!(same(&want, &got), "P's forward");
        let r = noise(&cpu, &[n, hid], 102, 1.0, 0.0, DType::F32);
        let gr = want.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
        let mut gs = empty_grads(&cpu);
        let (da, dctx) = dend_bwd(&m, i, &sv, &r, &mut gs).unwrap().unwrap();
        check("P da", gr.get(a.as_tensor()).unwrap(), &da, bar);
        check("P dctx", gr.get(ctx.as_tensor()).unwrap(), &dctx, bar);
        check("P dA", gr.get(wa).unwrap(), gs.get(wa).unwrap(), bar);
        check("P dB", gr.get(wb).unwrap(), gs.get(wb).unwrap(), bar);
        assert!(same(dend_units(&sv).unwrap(), a.as_tensor()), "P keeps the units before its gain");
    }
    // ---- S: scan_fast::stp_forward on 0/1 spikes and on int2's graded ones (-2..2: the events' slope at work)
    for (kind, vals) in [("binary", [0f32, 1.0, 0.0, 1.0, 0.0]), ("int2", [-2f32, -1.0, 0.0, 1.0, 2.0])] {
        let p = stp_of(&m, i).unwrap();
        let sv0: Vec<f32> = unit_noise(b * t * d, 110).iter().map(|u| vals[(((u + 1.0) * 2.5) as usize).min(4)]).collect();
        let s = Var::from_tensor(&Tensor::from_vec(sv0, (b, t, d), &cpu).unwrap()).unwrap();
        let want = crate::scan_fast::stp_forward(p, s.as_tensor()).unwrap();
        let mut sv = Blk::default();
        let got = stp_fwd(&m, i, s.as_tensor().detach(), &mut sv).unwrap();
        assert!(same(&want, &got), "S's forward ({kind})");
        let r = noise(&cpu, &[b, t, d], 111, 1.0, 0.0, DType::F32);
        let gr = want.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
        let mut gs = empty_grads(&cpu);
        let ds = stp_bwd(&m, i, &sv, &r, &mut gs).unwrap();
        check(&format!("S {kind} ds"), gr.get(s.as_tensor()).unwrap(), &ds, bar);
        for (name, var) in [("U", &p.u_param), ("f", &p.f_param), ("d", &p.d_param), ("mix", &p.mix)] {
            check(&format!("S {kind} d{name}"), gr.get(var).unwrap(), gs.get(var).unwrap(), 1e-4);
        }
    }
}

#[test]
fn prune_cuts_and_counts() {
    let cpu = Device::Cpu;
    let (_, m) = cpu_model();
    let hid = m.cfg.hidden();
    let (alive, _) = v(&m).blocks[1].prune.as_ref().unwrap();
    let al: Vec<f32> = (0..hid).map(|j| if j % 3 == 0 { 0.0 } else { 1.0 }).collect();
    let live = al.iter().filter(|&&a| a > 0.5).count();
    Var::from_tensor(alive).unwrap().set(&Tensor::from_vec(al.clone(), hid, &cpu).unwrap()).unwrap();
    m.v51_schedule(0, 100); // forget the cached counts
    let n = 9;
    let u = noise(&cpu, &[n, hid], 120, 1.0, 0.0, DType::F32);
    let (u2, k) = prune_u(&m, 1, u.clone(), m.cfg.kwta_k()).unwrap();
    assert_eq!(k, (live / m.cfg.arch.kwta_div).max(1), "k from the live units");
    let want = u.broadcast_add(&alive.affine(1e9, -1e9).unwrap()).unwrap();
    assert!(same(&want, &u2), "the cut units at -1e9, the live ones as they were");
    // the stats: the units' mean |a| over the rows
    let a = noise(&cpu, &[n, hid], 121, 1.0, 0.0, DType::F32);
    prune_stats(&m, 1, &a).unwrap();
    let st = v(&m).prune.stats.borrow()[1].clone().unwrap();
    assert!(same(&st, &a.abs().unwrap().mean(0).unwrap()), "the stats");
}
