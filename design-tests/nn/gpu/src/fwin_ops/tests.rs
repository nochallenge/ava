// TESTING / DESIGN ONLY — v47: fwin_ops's kernels against the plain candle ops (autograd for the gradients), f32,
// CUDA only (AVA_GPU_TEST_CUDA=1; each a fraction of a second on tiny shapes).

use super::*;
use crate::model::{head_norm, l2_norm, log_sigmoid, sigmoid};
use crate::testkit::unit_noise;
use candle_core::Var;

fn cuda() -> Option<Device> {
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v == "1").unwrap_or(false) {
        Some(Device::new_cuda(0).expect("CUDA device 0"))
    } else {
        eprintln!("fwin_ops: skipped (set AVA_GPU_TEST_CUDA=1)");
        None
    }
}

fn noise(dev: &Device, shape: &[usize], seed: u64, scale: f64) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec(unit_noise(n, seed), shape, dev).unwrap().affine(scale, 0.0).unwrap()
}

/// max |a - b| / max(max |a|, floor)
fn rel(a: &Tensor, b: &Tensor, floor: f32) -> f32 {
    let a = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let b = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let d = a.sub(&b).unwrap().abs().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
    let m = a.abs().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
    d / m.max(floor)
}

#[test]
fn relu2_matches_kwta_relu_squared() {
    let Some(dev) = cuda() else { return };
    let u = noise(&dev, &[37, 96], 1, 1.0);
    let da = noise(&dev, &[37, 96], 2, 1.0);
    let a_ref = crate::fused::kwta_relu_raw(&u, 24).unwrap();
    let (a, a2) = kwta_relu2(&u, 24, false).unwrap();
    assert_eq!(rel(&a_ref, &a, 1.0), 0.0, "a is kwta_relu's");
    assert_eq!(rel(&a_ref.sqr().unwrap(), &a2, 1.0), 0.0, "a2 = a^2");
    let du = relu2_bwd(&a, &da).unwrap();
    let want = a_ref.mul(&da).unwrap().affine(2.0, 0.0).unwrap().mul(&a_ref.gt(0.0).unwrap().to_dtype(DType::F32).unwrap()).unwrap();
    assert_eq!(rel(&want, &du, 1.0), 0.0, "du = 2 a da");
    let (ab, a2b) = kwta_relu2(&u, 24, true).unwrap();
    assert!(rel(&a_ref, &ab, 1.0) < 1e-2 && rel(&a_ref.sqr().unwrap(), &a2b, 1.0) < 1e-2);
    assert!(rel(&want, &relu2_bwd(&ab, &da).unwrap(), 1.0) < 2e-2);
}

/// The memory chain of arch.rs mem_arch (D, per-head gates) in plain ops on p [N, ld], cst [3 nh]; `vr`: (p0 [2], the
/// first block's p).
fn mem_ref(p: &Tensor, cst: &Tensor, vr: Option<(&Tensor, &Tensor)>, s: MemShape) -> Tensor {
    mem_ref_nm(p, cst, vr, s, None)
}

/// mem_ref with v51 fused N's ACh: the write strengths times the signal `ach` [N, 1], kept in [0, 2] (v50::ach_scale).
fn mem_ref_nm(p: &Tensor, cst: &Tensor, vr: Option<(&Tensor, &Tensor)>, s: MemShape, ach: Option<&Tensor>) -> Tensor {
    let (b, t, nh, hs) = (s.b, s.t, s.nh, s.hs);
    let (n, d) = (b * t, nh * hs);
    let heads = |x: &Tensor, c: usize| x.narrow(1, c * d, d).unwrap().reshape((b, t, nh, hs)).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
    let q = l2_norm(&heads(p, 0)).unwrap();
    let k = l2_norm(&heads(p, 1)).unwrap();
    let mut v = heads(p, 2);
    if let Some((w, p0)) = vr {
        let v0 = heads(p0, 2);
        v = v.broadcast_mul(&w.narrow(0, 0, 1).unwrap().affine(1.0, 1.0).unwrap()).unwrap().add(&v0.broadcast_mul(&w.narrow(0, 1, 1).unwrap()).unwrap()).unwrap();
    }
    let per_head = |x: Tensor| x.reshape((b, t, nh, 1)).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
    let za = p.narrow(1, 4 * d, nh).unwrap().broadcast_add(&cst.narrow(0, 0, nh).unwrap()).unwrap();
    let la = per_head(log_sigmoid(&za.broadcast_add(&cst.narrow(0, 2 * nh, nh).unwrap()).unwrap()).unwrap());
    let zb = p.narrow(1, 4 * d + nh, nh).unwrap().broadcast_add(&cst.narrow(0, nh, nh).unwrap()).unwrap();
    let be = per_head(sigmoid(&zb).unwrap().affine(2.0, 0.0).unwrap());
    let be = match ach {
        Some(a) => crate::v50::ach_scale(&be, a).unwrap(),
        None => be,
    };
    let (y, _) = crate::arch::delta_rec(&q, &k, &k, &v, &la, &be, 0.0, None).unwrap();
    let y = head_norm(&y).unwrap().transpose(1, 2).unwrap().contiguous().unwrap().reshape((n, d)).unwrap();
    y.mul(&candle_nn::ops::silu(&p.narrow(1, 3 * d, d).unwrap()).unwrap()).unwrap()
}

#[test]
fn memory_matches_the_plain_chain() {
    let Some(dev) = cuda() else { return };
    for (s, seed) in [(MemShape { b: 2, t: 37, nh: 2, hs: 64, chan: false, mom: 0.0 }, 10u64), (MemShape { b: 3, t: 20, nh: 3, hs: 32, chan: false, mom: 0.0 }, 20)] {
        let (n, d) = (s.b * s.t, s.nh * s.hs);
        let ld = 4 * d + 2 * s.nh;
        let p = Var::from_tensor(&noise(&dev, &[n, ld], seed, 1.0)).unwrap();
        let p0 = Var::from_tensor(&noise(&dev, &[n, ld], seed + 1, 1.0)).unwrap();
        let cst = Var::from_tensor(&noise(&dev, &[3 * s.nh], seed + 2, 1.0)).unwrap();
        let w = Var::from_tensor(&noise(&dev, &[2], seed + 3, 0.3)).unwrap();
        let go = noise(&dev, &[n, d], seed + 4, 1.0);
        for mix in [false, true] {
            let vr_ref = if mix { Some((w.as_tensor(), p0.as_tensor())) } else { None };
            let o_ref = mem_ref(p.as_tensor(), cst.as_tensor(), vr_ref, s);
            let vr = if mix { Vres::Mix { p: w.as_tensor(), p0: p0.as_tensor() } } else { Vres::Off };
            let (o, y) = wmem_fwd(p.as_tensor(), cst.as_tensor(), None, vr, s, 1e-5, false).unwrap();
            let e = rel(&o_ref, &o, 1.0);
            eprintln!("wmem hs {} mix {mix}: O rel {e:.2e}", s.hs);
            assert!(e < 1e-4, "O hs {} mix {mix}: {e}", s.hs);
            let gr = o_ref.mul(&go).unwrap().sum_all().unwrap().backward().unwrap();
            let acc0 = noise(&dev, &[n, d], seed + 5, 0.5);
            let g = wmem_bwd(p.as_tensor(), cst.as_tensor(), None, vr, &y, &go, Some(&acc0), s, 1e-5, false).unwrap();
            let dp_ref = gr.get(p.as_tensor()).unwrap();
            for (name, c0, wd) in [("dq", 0, d), ("dk", d, d), ("dv", 2 * d, d), ("dg", 3 * d, d), ("dza", 4 * d, s.nh), ("dzb", 4 * d + s.nh, s.nh)] {
                let e = rel(&dp_ref.narrow(1, c0, wd).unwrap(), &g.dp.narrow(1, c0, wd).unwrap(), 1e-3);
                eprintln!("wmem hs {} mix {mix}: {name} rel {e:.2e}", s.hs);
                assert!(e < 1e-3, "{name} hs {} mix {mix}: {e}", s.hs);
            }
            let dz_ref = Tensor::cat(&[dp_ref.narrow(1, 4 * d, s.nh).unwrap(), dp_ref.narrow(1, 4 * d + s.nh, s.nh).unwrap()], 1).unwrap();
            assert!(rel(&dz_ref, &g.dz, 1e-3) < 1e-3);
            let dc = gr.get(cst.as_tensor()).unwrap();
            let (dba, dbb) = (g.dz.narrow(1, 0, s.nh).unwrap().sum(0).unwrap(), g.dz.narrow(1, s.nh, s.nh).unwrap().sum(0).unwrap());
            assert!(rel(&dc.narrow(0, 0, s.nh).unwrap(), &dba, 1e-3) < 1e-3 && rel(&dc.narrow(0, s.nh, s.nh).unwrap(), &dbb, 1e-3) < 1e-3);
            if mix {
                // the first block's v gradient: acc0 + p1 dv; the weights' gradient (dp0, dp1)
                let dv0_ref = gr.get(p0.as_tensor()).unwrap().narrow(1, 2 * d, d).unwrap().add(&acc0).unwrap();
                let e = rel(&dv0_ref, g.acc.as_ref().unwrap(), 1e-3);
                eprintln!("wmem hs {}: acc rel {e:.2e}", s.hs);
                assert!(e < 1e-3, "acc: {e}");
                let e = rel(gr.get(w.as_tensor()).unwrap(), g.dvres.as_ref().unwrap(), 1e-3);
                eprintln!("wmem hs {}: dvres rel {e:.2e}", s.hs);
                assert!(e < 1e-3, "dvres: {e}");
            } else {
                // Vres::First adds acc to dv
                let gf = wmem_bwd(p.as_tensor(), cst.as_tensor(), None, Vres::First, &y, &go, Some(&acc0), s, 1e-5, false).unwrap();
                let want = dp_ref.narrow(1, 2 * d, d).unwrap().add(&acc0).unwrap();
                assert!(rel(&want, &gf.dp.narrow(1, 2 * d, d).unwrap(), 1e-3) < 1e-3, "first block's dv + acc");
            }
            // bf16 forms: the same values to bf16 rounding
            let (ob, _) = wmem_fwd(p.as_tensor(), cst.as_tensor(), None, vr, s, 1e-5, true).unwrap();
            assert!(rel(&o, &ob, 1.0) < 1e-2);
            let gb = wmem_bwd(p.as_tensor(), cst.as_tensor(), None, vr, &y, &go, Some(&acc0), s, 1e-5, true).unwrap();
            assert!(rel(&g.dp, &gb.dp, 1e-3) < 1e-2 && rel(&g.dz, &gb.dz, 1e-3) == 0.0);
        }
    }
}

/// v51 fused N: the memory with --neuromod's ACh (wmem_fwd_nm / wmem_bwd_nm; scan inputs kept, and the prep rerun)
/// against mem_ref_nm by autograd, per-head gates, head size 64 and 32: O within 1e-4, dP's columns and dL/d(signal)
/// within 1e-3 (memory_matches_the_plain_chain's bars), with a share of the strengths clamped at 2; a signal of ones is
/// wmem_fwd / wmem_bwd bit for bit.
#[test]
fn memory_with_ach_matches_the_plain_chain() {
    let Some(dev) = cuda() else { return };
    for (s, seed) in [(MemShape { b: 2, t: 37, nh: 2, hs: 64, chan: false, mom: 0.0 }, 30u64), (MemShape { b: 3, t: 20, nh: 3, hs: 32, chan: false, mom: 0.0 }, 40)] {
        let (n, d, nh) = (s.b * s.t, s.nh * s.hs, s.nh);
        let ld = 4 * d + 2 * nh;
        let p = Var::from_tensor(&noise(&dev, &[n, ld], seed, 1.0)).unwrap();
        // bb near 1: 2 sigmoid(zb + bb) times a signal up to 2 passes 2 for a good share of the tokens
        let cst = Tensor::cat(&[noise(&dev, &[nh], seed + 1, 0.3), noise(&dev, &[nh], seed + 2, 0.3).affine(1.0, 1.0).unwrap(), noise(&dev, &[nh], seed + 3, 0.3)], 0).unwrap();
        let a = Var::from_tensor(&sigmoid(&noise(&dev, &[n, 1], seed + 4, 1.5)).unwrap().affine(2.0, 0.0).unwrap()).unwrap();
        let go = noise(&dev, &[n, d], seed + 5, 1.0);
        let o_ref = mem_ref_nm(p.as_tensor(), &cst, None, s, Some(a.as_tensor()));
        let gr = o_ref.mul(&go).unwrap().sum_all().unwrap().backward().unwrap();
        let dp_ref = gr.get(p.as_tensor()).unwrap();
        for keep in [true, false] {
            let (o, y, pre, braw) = wmem_fwd_nm(p.as_tensor(), &cst, None, Vres::Off, s, 1e-5, false, keep, Some(a.as_tensor())).unwrap();
            assert_eq!(braw.is_some(), keep, "the unscaled strengths are kept with the scan inputs");
            let e = rel(&o_ref, &o, 1.0);
            let g = wmem_bwd_nm(p.as_tensor(), &cst, None, Vres::Off, &y, &go, None, s, 1e-5, false, pre.as_ref().map(|(x, z)| (x, z)), Some((a.as_tensor(), braw.as_ref()))).unwrap();
            let mut line = format!("wmem ACh hs {} keep {keep}: O rel {e:.2e}", s.hs);
            assert!(e < 1e-4, "{line}");
            for (name, c0, wd) in [("dq", 0, d), ("dk", d, d), ("dv", 2 * d, d), ("dg", 3 * d, d), ("dza", 4 * d, nh), ("dzb", 4 * d + nh, nh)] {
                let e = rel(&dp_ref.narrow(1, c0, wd).unwrap(), &g.dp.narrow(1, c0, wd).unwrap(), 1e-3);
                line += &format!(", {name} {e:.2e}");
                assert!(e < 1e-3, "{line}");
            }
            let e = rel(gr.get(a.as_tensor()).unwrap(), g.dach.as_ref().expect("dL/d(signal)"), 1e-3);
            eprintln!("{line}, d signal {e:.2e}");
            assert!(e < 1e-3, "{line}, d signal {e}");
        }
        let ones = Tensor::ones((n, 1), DType::F32, &dev).unwrap();
        let (o1, y1) = wmem_fwd(p.as_tensor(), &cst, None, Vres::Off, s, 1e-5, false).unwrap();
        let (o2, y2, _, _) = wmem_fwd_nm(p.as_tensor(), &cst, None, Vres::Off, s, 1e-5, false, false, Some(&ones)).unwrap();
        assert!(rel(&o1, &o2, 1e-30) == 0.0 && rel(&y1, &y2, 1e-30) == 0.0, "a unit signal: the plain memory's forward");
        let g1 = wmem_bwd(p.as_tensor(), &cst, None, Vres::Off, &y1, &go, None, s, 1e-5, false).unwrap();
        let g2 = wmem_bwd_nm(p.as_tensor(), &cst, None, Vres::Off, &y2, &go, None, s, 1e-5, false, None, Some((&ones, None))).unwrap();
        assert!(rel(&g1.dp, &g2.dp, 1e-30) == 0.0 && rel(&g1.dz, &g2.dz, 1e-30) == 0.0, "a unit signal: the plain memory's backward");
    }
}

#[test]
fn dense_matches_the_plain_chain() {
    let Some(dev) = cuda() else { return };
    let (l, n, d) = (4usize, 24usize, 16usize);
    let m = n * d;
    // forward: the chain of arch.rs forward_arch on raw streams x_0..x_L, with autograd for the reference backward
    let x0 = Var::from_tensor(&noise(&dev, &[n, d], 1, 1.0)).unwrap();
    let raws: Vec<(Tensor, Tensor)> = (0..l).map(|i| (noise(&dev, &[n, d], 10 + i as u64, 1.0), noise(&dev, &[n, d], 20 + i as u64, 1.0))).collect();
    let ws: Vec<Var> = (0..l).map(|i| Var::from_tensor(&noise(&dev, &[i + 2], 30 + i as u64, 0.3)).unwrap()).collect();
    let wf = Tensor::cat(&ws.iter().map(|w| w.as_tensor().clone()).collect::<Vec<_>>(), 0).unwrap();
    // the "block": raw_{i+1} = (in_i + xa_i) + xb_i, so the gradient flows through the stream as in a real block
    let mut streams = vec![x0.as_tensor().clone()];
    let mut x = x0.as_tensor().clone();
    let s = empty(&x, (l + 1, m), DType::F32).unwrap();
    s.slice_set(&x0.as_tensor().reshape((1, m)).unwrap(), 0, 0).unwrap();
    let mut xk = x0.as_tensor().clone();
    for (i, (xa, xb)) in raws.iter().enumerate() {
        let raw = x.add(xa).unwrap().add(xb).unwrap();
        streams.push(raw.clone());
        let mut mix = raw.clone();
        for (j, sj) in streams.iter().enumerate() {
            mix = mix.add(&sj.broadcast_mul(&ws[i].as_tensor().narrow(0, j, 1).unwrap()).unwrap()).unwrap();
        }
        x = mix;
        let got = dense_mix(&xk.add(xa).unwrap(), xb, &wf, &s, i).unwrap();
        let e = rel(&x, &got, 1.0);
        assert!(e < 1e-6, "dense_mix block {i}: {e}");
        xk = got;
    }
    let gfin = noise(&dev, &[n, d], 50, 1.0);
    let gr = x.mul(&gfin).unwrap().sum_all().unwrap().backward().unwrap();
    // the hand-written backward: per block, the stream gradient of raw_{i+1}; the in-block identity passes it to in_i
    let g = empty(&x, (l, m), DType::F32).unwrap();
    let mut dx = gfin.clone();
    for i in (0..l).rev() {
        g.slice_set(&dx.reshape((1, m)).unwrap(), 0, i).unwrap();
        let (out, _, dw) = dense_back(&s, &g, &wf, &g.narrow(0, i, 1).unwrap(), i + 1, i, i + 2, false).unwrap();
        let e = rel(gr.get(ws[i].as_tensor()).unwrap(), dw.as_ref().unwrap(), 1e-3);
        eprintln!("dense block {i}: dw rel {e:.2e}");
        assert!(e < 1e-4, "dw block {i}: {e}");
        dx = out.reshape((n, d)).unwrap(); // raw_{i+1} = in_i + xa + xb: dL/d in_i = dL/d raw_{i+1}
    }
    let (out, _, _) = dense_back(&s, &g, &wf, &dx, 0, 0, 0, false).unwrap();
    let e = rel(gr.get(x0.as_tensor()).unwrap(), &out.reshape((n, d)).unwrap(), 1e-3);
    eprintln!("dense: dx0 rel {e:.2e}");
    assert!(e < 1e-5, "dx0: {e}");
    let (_, ob, _) = dense_back(&s, &g, &wf, &dx, 0, 0, 0, true).unwrap();
    assert!(rel(&out, ob.as_ref().unwrap(), 1.0) < 1e-2);
}

/// v47 Amendment 3, E2: memg_bwd4 (rows split over 4 threads) against the old scan backward on random inputs, head size
/// 32 and 64, per-head and per-channel gates, momentum 0 and 0.5, T = 37 and 256: max |new - old| / max |old| <= 1e-4.
#[test]
fn memg_bwd4_matches_the_old_kernel() {
    let Some(dev) = cuda() else { return };
    let mut worst = 0.0f32;
    let mut seed = 900;
    for hs in [32usize, 64] {
        for chan in [false, true] {
            for mom in [0.0f32, 0.5] {
                for t in [37usize, 256] {
                    let s = MemShape { b: 2, t, nh: 3, hs, chan, mom };
                    let bn = s.b * s.nh;
                    let mut nz = |w: usize, scale: f64| {
                        seed += 1;
                        noise(&dev, &[bn, t, w], seed, scale)
                    };
                    let (q, k, e) = (l2_norm(&nz(hs, 1.0)).unwrap(), l2_norm(&nz(hs, 1.0)).unwrap(), l2_norm(&nz(hs, 1.0)).unwrap());
                    let qke = Tensor::cat(&[&q, &k, &e], 2).unwrap().contiguous().unwrap();
                    let v = nz(hs, 1.0);
                    let la = nz(s.g(), 0.1).affine(1.0, -0.26).unwrap();
                    let b = sigmoid(&nz(s.g(), 1.0)).unwrap().affine(2.0, 0.0).unwrap();
                    let vg = Tensor::cat(&[&v, &la, &b], 2).unwrap().contiguous().unwrap();
                    let dy = nz(hs, 1.0);
                    let (q1, g1) = memg_bwd_with(&qke, &vg, &dy, s, true).unwrap();
                    let (q0, g0) = memg_bwd_with(&qke, &vg, &dy, s, false).unwrap();
                    let (eq, eg) = (rel(&q1, &q0, 1e-12), rel(&g1, &g0, 1e-12));
                    eprintln!("E2 hs {hs} chan {chan} mom {mom} T {t}: dqke {eq:.2e}, dvg {eg:.2e}");
                    worst = worst.max(eq).max(eg);
                    assert!(eq <= 1e-4 && eg <= 1e-4, "memg_bwd4 differs: hs {hs} chan {chan} mom {mom} T {t}");
                }
            }
        }
    }
    eprintln!("E2 worst {worst:.2e} (bar 1e-4)");
}

/// and 64, per-head and per-channel gates, momentum 0 and 0.5, T = 37, 256, 300, items of C and 4 C steps, 1, 2 and 4
/// blocks per head in the walks: memgk_scan +
/// memgk_bwd against memg_bwd4 and memgk_fwd against memg_fwd, max |new - old| / max |old| <= 1e-6 (expected 0: the same
/// ops in the same order); against the old memg_bwd at E2's bar (1e-4).
#[test]
fn memgk_matches_the_v47_kernels() {
    let Some(dev) = cuda() else { return };
    let (mut worst, mut worst_old, mut exact, mut cases) = (0.0f32, 0.0f32, 0usize, 0usize);
    let mut seed = 4900;
    for hs in [32usize, 64] {
        let c = crate::mem_gen::ck_len(hs);
        for chan in [false, true] {
            for mom in [0.0f32, 0.5] {
                for t in [37usize, 256, 300] {
                    let s = MemShape { b: 2, t, nh: 3, hs, chan, mom };
                    let bn = s.b * s.nh;
                    let mut nz = |w: usize, scale: f64| {
                        seed += 1;
                        noise(&dev, &[bn, t, w], seed, scale)
                    };
                    let (q, k, e) = (l2_norm(&nz(hs, 1.0)).unwrap(), l2_norm(&nz(hs, 1.0)).unwrap(), l2_norm(&nz(hs, 1.0)).unwrap());
                    let qke = Tensor::cat(&[&q, &k, &e], 2).unwrap().contiguous().unwrap();
                    let v = nz(hs, 1.0);
                    let la = nz(s.g(), 0.1).affine(1.0, -0.26).unwrap();
                    let b = sigmoid(&nz(s.g(), 1.0)).unwrap().affine(2.0, 0.0).unwrap();
                    let vg = Tensor::cat(&[&v, &la, &b], 2).unwrap().contiguous().unwrap();
                    let dy = nz(hs, 1.0);
                    let y0 = memg_fwd_with(&qke, &vg, s, false, 1).unwrap();
                    let ys: Vec<Tensor> = [1usize, 2, 4].iter().map(|&g| memg_fwd_with(&qke, &vg, s, true, g).unwrap()).collect();
                    let (q4, g4) = memg_bwd_with(&qke, &vg, &dy, s, true).unwrap();
                    let (q0, g0) = memg_bwd_with(&qke, &vg, &dy, s, false).unwrap();
                    let maxd = |a: &Tensor, b: &Tensor| a.sub(b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
                    let ey = ys.iter().map(|y1| rel(y1, &y0, 1e-12)).fold(0.0f32, f32::max);
                    let mut zero = ys.iter().all(|y1| maxd(y1, &y0) == 0.0);
                    let mut line = format!("K1 hs {hs} chan {chan} mom {mom} T {t}: y {ey:.2e}");
                    let mut w = ey;
                    for (l, g) in [(c, 4usize), (4 * c, 1), (4 * c, 2)] {
                        let (qk, gk) = memg_bwd_k(&qke, &vg, &dy, s, l, g).unwrap();
                        let (eq, eg) = (rel(&qk, &q4, 1e-12), rel(&gk, &g4, 1e-12));
                        let (oq, og) = (rel(&qk, &q0, 1e-12), rel(&gk, &g0, 1e-12));
                        zero &= maxd(&qk, &q4) == 0.0 && maxd(&gk, &g4) == 0.0;
                        line += &format!(", L {l} G {g}: dqke {eq:.2e} dvg {eg:.2e} (old kernel {oq:.2e} {og:.2e})");
                        w = w.max(eq).max(eg);
                        worst_old = worst_old.max(oq).max(og);
                        assert!(oq <= 1e-4 && og <= 1e-4, "memgk differs from memg_bwd: {line}");
                    }
                    eprintln!("{line}{}", if zero { " [bit-identical]" } else { "" });
                    cases += 1;
                    exact += usize::from(zero);
                    worst = worst.max(w);
                    assert!(w <= 1e-6, "memgk differs from memg_bwd4 / memg_fwd: {line}");
                }
            }
        }
    }
    eprintln!("K1 worst {worst:.2e} (bar 1e-6), {exact}/{cases} cases bit-identical; against memg_bwd worst {worst_old:.2e} (bar 1e-4)");
}

// ------------------------------------------------------------------------------------------------- v52 fix_S

/// v52 fix_S: fused::to_bf16 (lif.cu cast_bf16's f2bf, which every bf16 matmul operand of the fused steps goes
/// through) keeps a NaN a NaN and every other value as candle's to_dtype(BF16). Before the fix the rounding add carried
/// a NaN's mantissa into its exponent and sign: CUDA's NaN 0x7FFFFFFF became -0.0, so a NaN in the forward silently
/// turned into 0 (runs/v52/fix_S.md). Fails on the old kernel (the first value comes back -0).
#[test]
fn to_bf16_keeps_nan() {
    let Some(dev) = cuda() else { return };
    let mut vals: Vec<f32> = vec![
        f32::from_bits(0x7FFF_FFFF), // CUDA's NaN (what 0/0, inf - inf give on the GPU)
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0xFFC0_0001),
        f32::from_bits(0x7F80_0001), // a signalling NaN with one mantissa bit
        f32::from_bits(0x7FBF_FFFF),
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        1.0,
        -1.5,
        3.38e38,
        -3.38e38,
        1e-40,
        0.1,
        65504.0,
    ];
    vals.extend(unit_noise(4096, 77).iter().enumerate().map(|(i, x)| x * 10f32.powi((i % 60) as i32 - 30)));
    let x = Tensor::from_vec(vals.clone(), vals.len(), &dev).unwrap();
    let a = crate::fused::to_bf16(&x).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let b = x.to_dtype(DType::BF16).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    for (i, v) in vals.iter().enumerate() {
        if v.is_nan() {
            assert!(a[i].is_nan() && b[i].is_nan(), "value {i} ({:#010x}): to_bf16 gives {} (candle {})", v.to_bits(), a[i], b[i]);
        } else {
            assert_eq!(a[i].to_bits(), b[i].to_bits(), "value {i} ({v:e}): to_bf16 {} vs candle {}", a[i], b[i]);
        }
    }
}

/// Random scan inputs as the memgk test makes them (q, k, e unit; log alpha ~ -0.26; b = 2 sigmoid; dy).
fn scan_inputs(dev: &Device, s: MemShape, seed: u64) -> (Tensor, Tensor, Tensor) {
    let bn = s.b * s.nh;
    let (t, hs) = (s.t, s.hs);
    let mut sd = seed;
    let mut nz = |w: usize, scale: f64| {
        sd += 1;
        noise(dev, &[bn, t, w], sd, scale)
    };
    let (q, k, e) = (l2_norm(&nz(hs, 1.0)).unwrap(), l2_norm(&nz(hs, 1.0)).unwrap(), l2_norm(&nz(hs, 1.0)).unwrap());
    let qke = Tensor::cat(&[&q, &k, &e], 2).unwrap().contiguous().unwrap();
    let v = nz(hs, 1.0);
    let la = nz(s.g(), 0.1).affine(1.0, -0.26).unwrap();
    let b = sigmoid(&nz(s.g(), 1.0)).unwrap().affine(2.0, 0.0).unwrap();
    let vg = Tensor::cat(&[&v, &la, &b], 2).unwrap().contiguous().unwrap();
    let dy = nz(hs, 1.0);
    (qke, vg, dy)
}

/// Every scan kernel's (y, dqke, dvg): memg_fwd, memgk_fwd (G = 4), memg_bwd, memg_bwd4, memg_bwd_k (L = C, G = 4).
fn all_scans(qke: &Tensor, vg: &Tensor, dy: &Tensor, s: MemShape) -> Vec<(&'static str, Tensor)> {
    let c = crate::mem_gen::ck_len(s.hs);
    let mut out = vec![("memg_fwd y", memg_fwd_with(qke, vg, s, false, 1).unwrap()), ("memgk_fwd y", memg_fwd_with(qke, vg, s, true, 4).unwrap())];
    for (name, (dq, dg)) in [
        ("memg_bwd", memg_bwd_with(qke, vg, dy, s, false).unwrap()),
        ("memg_bwd4", memg_bwd_with(qke, vg, dy, s, true).unwrap()),
        ("memgk", memg_bwd_k(qke, vg, dy, s, c, 4).unwrap()),
    ] {
        out.push((name, Tensor::cat(&[dq.flatten_all().unwrap(), dg.flatten_all().unwrap()], 0).unwrap()));
    }
    out
}

/// The CPU reference for the same inputs: mem_gen's loops (their hand backward checked against delta_rec's autograd
/// in mem_gen's tests), (y, [dqke | dvg]) as the kernels lay them out.
fn cpu_scan(qke: &Tensor, vg: &Tensor, dy: &Tensor, s: MemShape) -> (Tensor, Tensor) {
    let cpu = Device::Cpu;
    let (bn, t, hs, g) = (s.b * s.nh, s.t, s.hs, s.g());
    let to4 = |x: Tensor, w: usize| x.to_device(&cpu).unwrap().reshape((1, bn, t, w)).unwrap();
    let part = |x: &Tensor, a: usize, w: usize| -> Var { Var::from_tensor(&to4(x.narrow(2, a, w).unwrap(), w)).unwrap() };
    let (q, k, e) = (part(qke, 0, hs), part(qke, hs, hs), part(qke, 2 * hs, hs));
    let (v, la, b) = (part(vg, 0, hs), part(vg, hs, g), part(vg, hs + g, g));
    let y = crate::mem_gen::mem_gen(q.as_tensor(), k.as_tensor(), e.as_tensor(), v.as_tensor(), la.as_tensor(), b.as_tensor(), s.mom as f64).unwrap();
    let w = to4(dy.clone(), hs);
    let gr = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
    let gq = |x: &Var| gr.get(x.as_tensor()).unwrap().reshape((bn, t, x.dim(3).unwrap())).unwrap();
    let dqke = Tensor::cat(&[gq(&q), gq(&k), gq(&e)], 2).unwrap();
    let dvg = Tensor::cat(&[gq(&v), gq(&la), gq(&b)], 2).unwrap();
    (y.reshape((bn, t, hs)).unwrap(), Tensor::cat(&[dqke.flatten_all().unwrap(), dvg.flatten_all().unwrap()], 0).unwrap())
}

/// v52 fix_S, B1: below the bound the kernels give the old bits. On ordinary inputs (no element near MEM_CAP) every
/// scan kernel's outputs with the bound equal those with cap = inf (fminf/fmaxf with inf and the |fn| < inf test are
/// the identity: the old kernels' maths) exactly. B2: with a bound that holds part of the state (0.6), every kernel
/// matches the CPU loops (max |kernel - cpu| / max |cpu| <= 1e-4), memgk included (its scan sees the bound held and the
/// host runs memg_bwd4 instead; mem_gen::SAT_EVENTS counts it), and the bound moved y (it was in force).
#[test]
fn memory_bound_kernels() {
    let Some(dev) = cuda() else { return };
    let mut seed = 52_000;
    for hs in [32usize, 64] {
        for chan in [false, true] {
            for mom in [0.0f32, 0.5] {
                let s = MemShape { b: 2, t: 300, nh: 3, hs, chan, mom };
                seed += 10;
                let (qke, vg, dy) = scan_inputs(&dev, s, seed);
                let old = crate::mem_gen::with_cap(f32::INFINITY, || all_scans(&qke, &vg, &dy, s));
                let new = all_scans(&qke, &vg, &dy, s);
                for ((name, a), (_, b)) in old.iter().zip(&new) {
                    let d = a.sub(b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
                    assert!(d == 0.0, "B1 hs {hs} chan {chan} mom {mom}: {name} moved by {d} below the bound");
                }
                let cap = 0.6f32;
                let ev0 = crate::mem_gen::SAT_EVENTS.load(std::sync::atomic::Ordering::Relaxed);
                let capped = crate::mem_gen::with_cap(cap, || all_scans(&qke, &vg, &dy, s));
                let ev = crate::mem_gen::SAT_EVENTS.load(std::sync::atomic::Ordering::Relaxed) - ev0;
                let (y_ref, g_ref) = crate::mem_gen::with_cap(cap, || cpu_scan(&qke, &vg, &dy, s));
                let moved = rel(&new[0].1, &capped[0].1, 1e-12);
                let mut line = format!("B2 hs {hs} chan {chan} mom {mom}: y moved {moved:.2e} by the bound {cap}, memgk fell back {ev}x");
                for (name, x) in &capped {
                    let e = if name.ends_with(" y") { rel(&y_ref, &x.to_device(&Device::Cpu).unwrap(), 1e-12) } else { rel(&g_ref, &x.to_device(&Device::Cpu).unwrap(), 1e-12) };
                    line += &format!(", {name} {e:.1e}");
                    assert!(e <= 1e-4, "{line}: {name} differs from the CPU loops");
                }
                eprintln!("{line}");
                assert!(moved > 1e-2, "{line}: the bound should move y");
                assert!(ev >= 1, "{line}: memgk should see the bound held");
            }
        }
    }
}

/// v52 fix_S, the failing condition in small on the fused step's kernels: mem_gen's runaway head (momentum 0.5, write
/// strength 1.9, keys cycling through 3 directions). Without the bound (cap = inf, the old maths) y overflows f32; with
/// it every kernel's y and gradients are finite.
#[test]
fn memory_bound_keeps_the_kernels_finite() {
    let Some(dev) = cuda() else { return };
    let (t, hs) = (1024usize, 64usize);
    let [q, k, e, v, la, b] = crate::mem_gen::tests::runaway(&dev, t, hs);
    let s = MemShape { b: 1, t, nh: 2, hs, chan: true, mom: 0.5 };
    let r3 = |x: &Tensor| x.reshape((2, t, x.dim(3).unwrap())).unwrap();
    let qke = Tensor::cat(&[r3(&q), r3(&k), r3(&e)], 2).unwrap().contiguous().unwrap();
    let vg = Tensor::cat(&[r3(&v), r3(&la), r3(&b)], 2).unwrap().contiguous().unwrap();
    let dy = noise(&dev, &[2, t, hs], 5, 1e-3);
    let fin = |x: &Tensor| x.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|v| v.is_finite());
    let old = crate::mem_gen::with_cap(f32::INFINITY, || memg_fwd_with(&qke, &vg, s, true, 4).unwrap());
    assert!(!fin(&old), "without the bound the runaway head should overflow f32");
    for (name, x) in all_scans(&qke, &vg, &dy, s) {
        assert!(fin(&x), "{name}: not finite with the bound");
    }
    eprintln!("runaway head: no bound -> y not finite; bound {:e} -> every scan kernel finite", crate::mem_gen::MEM_CAP);
}
