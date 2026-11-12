// TESTING / DESIGN ONLY — tests of src/fwin_nm.rs (v51 fused N: --neuromod 3 on the fused WIN step).
//
// CPU: the candle-op twins of the step's --neuromod pieces against candle's autograd of the reference path's own ops
// (src/v50.rs: nucleus_signals, ach_scale; the per-row scale of block_arch), f64 and f32. The twins' backward ops
// differ from autograd's only in summation order, so: f64 within 1e-12 relative, f32 within 2e-6 (5e-6 for sums);
// the forward and the elementwise backward parts are the reference's own ops and must be bit for bit.
// CUDA (AVA_GPU_TEST_CUDA=1, a fraction of a second): the kernels against the twins, the same bars (elementwise parts
// bit for bit, the per-token sums within 1e-5).

use super::*;
use crate::testkit::unit_noise;
use candle_core::{Device, Var};

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

fn tol(dt: DType, sum: bool) -> f64 {
    match (dt, sum) {
        (DType::F64, _) => 1e-12,
        (_, false) => 2e-6,
        (_, true) => 5e-6,
    }
}

#[test]
fn nucleus_backward_matches_autograd() {
    let cpu = Device::Cpu;
    for dt in [DType::F64, DType::F32] {
        let (n, d, k) = (13usize, 24usize, 3usize);
        let x0 = Var::from_tensor(&noise(&cpu, &[n, d], 1, 1.5, 0.2, dt)).unwrap();
        let g = Var::from_tensor(&noise(&cpu, &[d], 2, 0.3, 1.0, dt)).unwrap();
        let w = Var::from_tensor(&noise(&cpu, &[d, k], 3, 0.5, 0.0, dt)).unwrap();
        let b = Var::from_tensor(&noise(&cpu, &[k], 4, 0.5, 0.0, dt)).unwrap();
        let r = noise(&cpu, &[n, k], 5, 1.0, 0.0, dt);
        let sig = crate::v50::nucleus_signals(x0.as_tensor(), g.as_tensor(), w.as_tensor(), b.as_tensor()).unwrap();
        let gr = sig.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
        let nu = nucleus_fwd(x0.as_tensor(), g.as_tensor(), w.as_tensor(), b.as_tensor()).unwrap();
        assert!(same(&sig, &nu.sig), "{dt:?}: the signals are v50::nucleus_signals's, bit for bit");
        for (k, c) in nu.cols.iter().enumerate() {
            assert!(c.is_contiguous() && c.dims() == [n, 1] && same(c, &sig.narrow(1, k, 1).unwrap()), "column {k}");
        }
        let (dx, dg, dw, db) = nucleus_bwd(&nu, x0.as_tensor(), g.as_tensor(), w.as_tensor(), &r).unwrap();
        for (name, want, got) in [("x0", &x0, &dx), ("gain", &g, &dg), ("W", &w, &dw), ("b", &b, &db)] {
            let e = rel(gr.get(want.as_tensor()).unwrap(), got, 1e-30);
            eprintln!("nucleus {dt:?} d{name}: rel {e:.2e}");
            assert!(e <= tol(dt, true), "nucleus {dt:?} d{name}: {e}");
        }
    }
}

#[test]
fn rms_backward_matches_autograd() {
    let cpu = Device::Cpu;
    for dt in [DType::F64, DType::F32] {
        let x = Var::from_tensor(&noise(&cpu, &[9, 40], 11, 2.0, -0.3, dt)).unwrap();
        let g = Var::from_tensor(&noise(&cpu, &[40], 12, 0.4, 1.0, dt)).unwrap();
        let dy = noise(&cpu, &[9, 40], 13, 1.0, 0.0, dt);
        let gr = crate::model::rmsnorm(x.as_tensor(), g.as_tensor()).unwrap().mul(&dy).unwrap().sum_all().unwrap().backward().unwrap();
        let (dx, dg) = rms_bwd_ops(x.as_tensor(), g.as_tensor(), &dy, NORM_EPS).unwrap();
        let (ex, eg) = (rel(gr.get(x.as_tensor()).unwrap(), &dx, 1e-30), rel(gr.get(g.as_tensor()).unwrap(), &dg, 1e-30));
        eprintln!("rmsnorm {dt:?}: dx rel {ex:.2e}, dg rel {eg:.2e}");
        assert!(ex <= tol(dt, true) && eg <= tol(dt, true), "rmsnorm {dt:?}: {ex} {eg}");
        let (dx2, dg2) = rms_bwd(x.as_tensor(), g.as_tensor(), &dy).unwrap();
        assert!(same(&dx, &dx2) && same(&dg, &dg2), "rms_bwd on the CPU is rms_bwd_ops");
    }
}

#[test]
fn row_scale_backward_matches_autograd() {
    let cpu = Device::Cpu;
    for dt in [DType::F64, DType::F32] {
        let (n, m) = (11usize, 40usize);
        let x = Var::from_tensor(&noise(&cpu, &[n, m], 21, 1.0, 0.1, dt)).unwrap();
        let s = Var::from_tensor(&noise(&cpu, &[n, 1], 22, 0.5, 1.0, dt)).unwrap();
        let dy = noise(&cpu, &[n, m], 23, 1.0, 0.0, dt);
        // block_arch's ops: NE cur.broadcast_mul(s reshaped [b, t, 1]), DA y.broadcast_mul(s)
        let y = x.as_tensor().broadcast_mul(s.as_tensor()).unwrap();
        let gr = y.mul(&dy).unwrap().sum_all().unwrap().backward().unwrap();
        let (dx, ds) = rows_bwd_ops(&dy, x.as_tensor(), s.as_tensor()).unwrap();
        assert!(same(gr.get(x.as_tensor()).unwrap(), &dx), "{dt:?}: dx = dy s, autograd's op");
        let e = rel(gr.get(s.as_tensor()).unwrap(), &ds, 1e-30);
        eprintln!("row scale {dt:?}: ds rel {e:.2e}");
        assert!(e <= tol(dt, true), "row scale {dt:?} ds: {e}");
        if dt == DType::F32 {
            // the dispatching form (the step's call): x in another shape of the same numbers, f32 operand on the CPU
            let (b, t) = (1usize, n);
            let (dx2, ds2) = rows_bwd(&dy, &x.as_tensor().reshape((b, t, m)).unwrap(), s.as_tensor(), false).unwrap();
            assert!(same(&dx, &dx2) && same(&ds, &ds2) && dx2.dtype() == DType::F32, "rows_bwd on the CPU is rows_bwd_ops");
        }
    }
}

#[test]
fn clamp_mask_is_candles_clamp_gradient() {
    let cpu = Device::Cpu;
    let y = Var::from_tensor(&Tensor::new(&[-1.0f32, 0.0, 0.5, 2.0, 3.0, 1.999, 2.001], &cpu).unwrap()).unwrap();
    let gr = y.as_tensor().clamp(0f32, 2f32).unwrap().sum_all().unwrap().backward().unwrap();
    let m = clamp_mask(y.as_tensor(), 0.0, 2.0).unwrap();
    assert_eq!(m.to_vec1::<f32>().unwrap(), vec![0.0, 0.5, 1.0, 0.5, 0.0, 1.0, 0.0]);
    assert!(same(gr.get(y.as_tensor()).unwrap(), &m), "autograd's clamp gradient, ties included");
}

/// The memory's write strengths beta [b, nh, t, g] in (0, 2) and the ACh signal s [b t, 1] in (0, 2), with both clamp
/// ties placed: beta 0 (y = 0) and beta 1 at a token whose signal is exactly 2 (y = 2); some products above 2.
fn ach_inputs(dev: &Device, sh: MemShape, seed: u64) -> (Tensor, Tensor) {
    let g = sh.g();
    let (b, nh, t) = (sh.b, sh.nh, sh.t);
    let mut be: Vec<f32> = unit_noise(b * nh * t * g, seed).iter().map(|&z| 2.0 / (1.0 + (-1.5 * z).exp())).collect();
    let mut s: Vec<f32> = unit_noise(b * t, seed + 1).iter().map(|&z| 2.0 / (1.0 + (-1.5 * z).exp())).collect();
    be[0] = 0.0; // (b 0, h 0, t 0, c 0): the lower tie
    s[1] = 2.0;
    be[g] = 1.0; // (b 0, h 0, t 1, c 0): the upper tie
    (Tensor::from_vec(be, (b, nh, t, g), dev).unwrap(), Tensor::from_vec(s, (b * t, 1), dev).unwrap())
}

/// vg [b nh, t, hs + 2g] (v, log alpha, beta) with the given strengths in fwin_ops's layout.
fn vg_with(dev: &Device, beta: &Tensor, sh: MemShape, seed: u64) -> Tensor {
    let (bh, g) = (sh.b * sh.nh, sh.g());
    let v = noise(dev, &[bh, sh.t, sh.hs], seed, 1.0, 0.0, DType::F32);
    let la = noise(dev, &[bh, sh.t, g], seed + 1, 0.1, -0.26, DType::F32);
    Tensor::cat(&[&v, &la, &beta.reshape((bh, sh.t, g)).unwrap()], 2).unwrap().contiguous().unwrap()
}

fn shapes() -> Vec<MemShape> {
    vec![
        MemShape { b: 2, t: 5, nh: 3, hs: 32, chan: false, mom: 0.0 },
        MemShape { b: 2, t: 4, nh: 2, hs: 64, chan: true, mom: 0.5 },
        MemShape { b: 1, t: 7, nh: 3, hs: 32, chan: true, mom: 0.0 },
    ]
}

#[test]
fn ach_matches_the_reference_op_and_autograd() {
    let cpu = Device::Cpu;
    for (j, sh) in shapes().into_iter().enumerate() {
        let (g, w) = (sh.g(), sh.hs + 2 * sh.g());
        let (bh, n) = (sh.b * sh.nh, sh.b * sh.t);
        let (beta, s) = ach_inputs(&cpu, sh, 100 + 10 * j as u64);
        let vg = vg_with(&cpu, &beta, sh, 200 + 10 * j as u64);
        let vg0 = vg.copy().unwrap();
        let braw = ach_fwd_ops(&vg, &s, sh).unwrap();
        assert!(same(&braw, &beta), "{sh:?}: the unscaled strengths handed back");
        let want = crate::v50::ach_scale(&beta, &s).unwrap();
        let got = vg.narrow(2, w - g, g).unwrap().contiguous().unwrap().reshape((sh.b, sh.nh, sh.t, g)).unwrap();
        assert!(same(&want, &got), "{sh:?}: vg's strengths are v50::ach_scale's, bit for bit");
        assert!(same(&vg0.narrow(2, 0, w - g).unwrap(), &vg.narrow(2, 0, w - g).unwrap()), "{sh:?}: v and log alpha untouched");
        let over = beta.broadcast_mul(&s.reshape((sh.b, 1, sh.t, 1)).unwrap()).unwrap().gt(2f32).unwrap().to_dtype(DType::F32).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(over > 0.0, "{sh:?}: the clamp acts somewhere");
        let vg2 = vg0.copy().unwrap();
        assert!(same(&ach_fwd(&vg2, &s, sh).unwrap(), &braw) && same(&vg2, &vg), "ach_fwd on the CPU is ach_fwd_ops");
        // backward: loss = sum(R ⊙ ach_scale(beta, s)) by autograd; R in vg's strength columns of dvg
        let bv = Var::from_tensor(&beta).unwrap();
        let sv = Var::from_tensor(&s).unwrap();
        let r = noise(&cpu, &[sh.b, sh.nh, sh.t, g], 300 + j as u64, 1.0, 0.0, DType::F32);
        let gr = crate::v50::ach_scale(bv.as_tensor(), sv.as_tensor()).unwrap().mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
        let rest = noise(&cpu, &[bh, sh.t, sh.hs + g], 400 + j as u64, 1.0, 0.0, DType::F32);
        let dvg = Tensor::cat(&[&rest, &r.reshape((bh, sh.t, g)).unwrap()], 2).unwrap().contiguous().unwrap();
        let ds = ach_bwd_ops(&dvg, &beta, &s, sh).unwrap();
        let dbeta = dvg.narrow(2, w - g, g).unwrap().contiguous().unwrap().reshape((sh.b, sh.nh, sh.t, g)).unwrap();
        assert!(same(gr.get(bv.as_tensor()).unwrap(), &dbeta), "{sh:?}: dL/dbeta is autograd's, bit for bit (ties included)");
        assert!(same(&rest, &dvg.narrow(2, 0, w - g).unwrap()), "{sh:?}: dv and d log alpha untouched");
        assert_eq!(ds.dims(), [n, 1]);
        let e = rel(gr.get(sv.as_tensor()).unwrap(), &ds, 1e-30);
        eprintln!("ACh {sh:?}: {over} of {} products clamped; ds rel {e:.2e}", beta.elem_count());
        assert!(e <= tol(DType::F32, true), "{sh:?}: ds {e}");
    }
}

#[test]
fn signal_sums_cat_in_order() {
    let cpu = Device::Cpu;
    let mut parts: [Option<Tensor>; 3] = [None, None, None];
    acc(&mut parts[ACH], Tensor::new(&[[1f32], [2.0]], &cpu).unwrap()).unwrap();
    acc(&mut parts[ACH], Tensor::new(&[[0.5f32], [0.25]], &cpu).unwrap()).unwrap();
    acc(&mut parts[DA], Tensor::new(&[[3f32], [4.0]], &cpu).unwrap()).unwrap();
    let d = dsig_cat(&parts, 2, &Tensor::zeros(1, DType::F32, &cpu).unwrap()).unwrap();
    assert_eq!(d.to_vec2::<f32>().unwrap(), vec![vec![0.0, 1.5, 3.0], vec![0.0, 2.25, 4.0]]);
    assert_eq!((NE, ACH, DA), (0, 1, 2), "v50's columns");
}

// ---------------------------------------------------------------- CUDA: the kernels against the twins

fn cuda() -> Option<Device> {
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v == "1").unwrap_or(false) {
        Some(Device::new_cuda(0).expect("CUDA device 0"))
    } else {
        eprintln!("fwin_nm kernels: skipped (set AVA_GPU_TEST_CUDA=1)");
        None
    }
}

#[test]
fn kernels_match_the_twins() {
    let Some(dev) = cuda() else { return };
    if !kernels_on() {
        eprintln!("fwin_nm kernels: AVA_NM_K=0, the twins on both sides (nothing to compare)");
    }
    let mut shs = shapes();
    shs.push(MemShape { b: 2, t: 300, nh: 6, hs: 64, chan: true, mom: 0.5 });
    shs.push(MemShape { b: 3, t: 257, nh: 4, hs: 32, chan: false, mom: 0.0 });
    let mut worst = 0.0f64;
    for (j, sh) in shs.into_iter().enumerate() {
        let (beta, s) = ach_inputs(&dev, sh, 500 + 10 * j as u64);
        let vg = vg_with(&dev, &beta, sh, 600 + 10 * j as u64);
        let (vk, vo) = (vg.copy().unwrap(), vg.copy().unwrap());
        let (bk, bo) = (ach_fwd(&vk, &s, sh).unwrap(), ach_fwd_ops(&vo, &s, sh).unwrap());
        assert!(same(&vk, &vo) && same(&bk, &bo), "nm_ach_fwd {sh:?}: not the twin's bits");
        let dvg = noise(&dev, &[sh.b * sh.nh, sh.t, sh.hs + 2 * sh.g()], 700 + j as u64, 1.0, 0.0, DType::F32);
        let (dk, dop) = (dvg.copy().unwrap(), dvg.copy().unwrap());
        let (sk, so) = (ach_bwd(&dk, &bk, &s, sh).unwrap(), ach_bwd_ops(&dop, &bo, &s, sh).unwrap());
        assert!(same(&dk, &dop), "nm_ach_bwd {sh:?}: dvg not the twin's bits");
        let e = rel(&so, &sk, 1e-30);
        eprintln!("nm_ach {sh:?}: forward and dvg bit for bit, ds rel {e:.2e}");
        worst = worst.max(e);
        assert!(e <= 1e-5, "nm_ach_bwd {sh:?}: ds {e}");
    }
    for (n, m, seed) in [(37usize, 384usize, 800u64), (300, 100, 810), (5, 31, 820)] {
        let dy = noise(&dev, &[n, m], seed, 1.0, 0.0, DType::F32);
        let x = noise(&dev, &[n, m], seed + 1, 1.0, 0.0, DType::F32);
        let s = noise(&dev, &[n, 1], seed + 2, 0.5, 1.0, DType::F32);
        let (dxo, dso) = rows_bwd_ops(&dy, &x, &s).unwrap();
        let (dxk, dsk) = rows_bwd(&dy, &x, &s, false).unwrap();
        assert!(same(&dxo, &dxk), "nm_rows_bwd [{n}, {m}]: dx not the twin's bits");
        let (dxb, dsb) = rows_bwd(&dy, &x, &s, true).unwrap();
        assert!(dxb.dtype() == DType::BF16 && same(&crate::fused::to_bf16(&dxo).unwrap(), &dxb), "nm_rows_bwd_b [{n}, {m}]: dx not to_bf16 of the twin's");
        let (e, eb) = (rel(&dso, &dsk, 1e-30), rel(&dso, &dsb, 1e-30));
        eprintln!("nm_rows [{n}, {m}]: dx bit for bit (f32 and bf16), ds rel {e:.2e} / {eb:.2e}");
        worst = worst.max(e).max(eb);
        assert!(e <= 1e-5 && eb <= 1e-5, "nm_rows_bwd [{n}, {m}]: ds {e} {eb}");
        // bf16 dy (the plain LIF's bf16 backward) goes through f32 first
        let (dxh, _) = rows_bwd(&crate::fused::to_bf16(&dy).unwrap(), &x, &s, false).unwrap();
        let want = crate::fused::to_bf16(&dy).unwrap().to_dtype(DType::F32).unwrap().broadcast_mul(&s).unwrap();
        assert!(same(&want, &dxh), "nm_rows_bwd: a bf16 dy");
    }
    eprintln!("fwin_nm kernels: worst per-token sum rel {worst:.2e} (bar 1e-5)");
}
