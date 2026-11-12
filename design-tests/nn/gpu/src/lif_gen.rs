// TESTING / DESIGN ONLY — ava_gpu::lif_gen: the general LIF scan as one fused op.
//
// arch::lif_ref runs the LIF with every option (E1 graded spikes, E4 adaptive thresholds, E8 per-neuron thresholds,
// O1's carried state) as ~15 small candle ops per time step, so a block with any of them trained 15-40x slower than
// the fused binary scan of lif.rs. `lif_gen` is the same scan in one kernel launch per layer and direction:
// - forward: the same f32 operations in the same order as lif_ref, so the spikes and the final state are the same;
// - backward: written by hand (BPTT through the leak and the trace, the reset and the trace's spikes detached, the
//   surrogate 1/(1 + S|x|)^2 per edge), the same gradients as lif_ref's autograd up to rounding.
// The parameters and the initial state travel as one flat tensor `ps` (candle custom ops take at most three inputs):
//     ps = [beta | theta | rho | gain] (each [d]), then [v0 | s0 | a0] (each [B·d]).
// Off switch: AVA_LIF_REF=1 runs lif_ref instead. CUDA kernels: lifg_* in src/lif.cu; the CPU path runs the same loops.

use candle_core::{CpuStorage, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Tensor};

use crate::arch::{LifState, Spikes};
use crate::lif::{cuda_slice, launch_cfg};
use crate::model::LIF_THRESHOLD;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/lif.ptx"));

/// The fused general scan is on unless AVA_LIF_REF=1.
pub fn on() -> bool {
    std::env::var("AVA_LIF_REF").map_or(true, |v| v != "1")
}

#[derive(Clone, Copy, Debug)]
struct Kind {
    /// spike levels each side of 0 (1 or 2)
    levels: usize,
    /// spikes can be negative (ternary, int2)
    signed: bool,
    /// E4's trace
    alif: bool,
    /// the surrogate slope (positive)
    slope: f32,
    /// v52 spike fix: the backward carries the gradient through the reset (asked for by a negative slope)
    rg: bool,
}

impl Kind {
    fn new(spikes: Spikes, alif: bool, slope: f32) -> Kind {
        Kind { levels: spikes.levels(), signed: spikes != Spikes::Binary, alif, slope: slope.abs(), rg: slope < 0.0 }
    }

    fn kernel(&self, dir: &str) -> String {
        let s = match (self.signed, self.levels) {
            (false, _) => "b",
            (true, 1) => "t",
            _ => "i",
        };
        format!("lifg_{dir}_{s}{}", self.alif as u8)
    }

    fn spike(&self, v: f32, th: f32) -> f32 {
        if !self.signed {
            return if v - th >= 0.0 { 1.0 } else { 0.0 };
        }
        let mut h = 0f32;
        for k in 1..=self.levels {
            let kth = th * k as f32;
            h += (if v - kth >= 0.0 { 1.0 } else { 0.0 }) - (if -v - kth >= 0.0 { 1.0 } else { 0.0 });
        }
        h
    }

    /// (ds/dv, ds/dth) of the surrogate at (v, th)
    fn dspike(&self, v: f32, th: f32) -> (f32, f32) {
        let sur = |x: f32| {
            let a = 1.0 + self.slope * x.abs();
            1.0 / (a * a)
        };
        if !self.signed {
            let g = sur(v - th);
            return (g, -g);
        }
        let (mut dv, mut dth) = (0f32, 0f32);
        for k in 1..=self.levels {
            let kth = th * k as f32;
            let (gu, gd) = (sur(v - kth), sur(-v - kth));
            dv += gu + gd;
            dth += k as f32 * (gd - gu);
        }
        (dv, dth)
    }
}

/// The LIF scan of arch::lif_ref (same arguments, same results), fused.
pub fn lif_gen(
    cur: &Tensor,
    beta: &Tensor,
    theta: Option<&Tensor>,
    alif: Option<(&Tensor, &Tensor)>,
    spikes: Spikes,
    slope: f64,
    init: Option<&LifState>,
) -> Result<(Tensor, LifState)> {
    let (b, t, d) = cur.dims3()?;
    let dev = cur.device();
    let th = match theta {
        Some(x) => x.detach(),
        None => Tensor::full(LIF_THRESHOLD as f32, d, dev)?,
    };
    let zd = Tensor::zeros(d, DType::F32, dev)?;
    let zbd = Tensor::zeros(b * d, DType::F32, dev)?;
    let (rho, gain) = match alif {
        Some((r, g)) => (r.clone(), g.clone()),
        None => (zd.clone(), zd),
    };
    let v0 = match init {
        Some(s) => s.v.flatten_all()?,
        None => zbd.clone(),
    };
    let s0 = match init {
        Some(s) => s.s.detach().flatten_all()?,
        None => zbd.clone(),
    };
    let a0 = match (alif, init.and_then(|s| s.a.as_ref())) {
        (Some(_), Some(a)) => a.flatten_all()?,
        _ => zbd,
    };
    let ps = Tensor::cat(&[beta, &th, &rho, &gain, &v0, &s0, &a0], 0)?.contiguous()?;
    let kind = Kind::new(spikes, alif.is_some(), slope as f32);
    let out = cur.contiguous()?.apply_op2(&ps, LifGen { kind })?;
    let (btd, bd) = (b * t * d, b * d);
    let s = out.narrow(0, 0, btd)?.reshape((b, t, d))?;
    let v = out.narrow(0, btd, bd)?.reshape((b, d))?;
    let a = if alif.is_some() { Some(out.narrow(0, btd + bd, bd)?.reshape((b, d))?) } else { None };
    let sp = s.narrow(1, t - 1, 1)?.squeeze(1)?.detach();
    Ok((s, LifState { v, s: sp, a }))
}

fn dims(l1: &Layout, l2: &Layout) -> Result<(usize, usize, usize)> {
    let (b, t, d) = l1.shape().dims3()?;
    if l2.shape().dims() != [4 * d + 3 * b * d] {
        candle_core::bail!("lif_gen: ps has shape {:?}, expected [{}]", l2.shape().dims(), 4 * d + 3 * b * d);
    }
    if t == 0 {
        candle_core::bail!("lif_gen: no time steps");
    }
    Ok((b, t, d))
}

fn cpu_slice<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let v = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(&v[a..b]),
        None => candle_core::bail!("lif_gen: input must be contiguous"),
    }
}

/// v52 spike fix (lif.cu's lifg_te): the threshold the reset path carries in the backward, th, or 2 beta / (ds/dv) where
/// th ds/dv > 2 beta, so the per-step factor beta - th ds/dv stays in [-beta, beta] (the cap passes no gradient).
fn reset_te(th: f32, dv: f32, bt: f32) -> f32 {
    let (q, cap) = (th * dv, 2.0 * bt);
    if q > cap {
        cap / dv
    } else {
        th
    }
}

/// The per-channel parameters of `ps` for channel c: (beta, theta, rho, 1 - rho, gain).
fn params(ps: &[f32], d: usize, c: usize) -> (f32, f32, f32, f32, f32) {
    let rho = ps[2 * d + c];
    (ps[c], ps[d + c], rho, 1.0 - rho, ps[3 * d + c])
}

/// One step (the kernel's lifg_step): updates v, sp, a and the reset threshold thp.
#[allow(clippy::too_many_arguments)]
#[inline]
fn step(k: &Kind, x: f32, bt: f32, th0: f32, rho: f32, omr: f32, gain: f32, v: &mut f32, sp: &mut f32, a: &mut f32, thp: &mut f32) {
    let mut tht = th0;
    if k.alif {
        *a = *a * rho + sp.abs() * omr;
        tht = th0 + *a * gain;
    }
    *v = (*v * bt + x) - *sp * *thp;
    *sp = k.spike(*v, tht);
    *thp = tht;
}

struct LifGen {
    kind: Kind,
}

impl CustomOp2 for LifGen {
    fn name(&self) -> &'static str {
        "lif-gen"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, t, d) = dims(l1, l2)?;
        let (cur, ps) = (cpu_slice(s1, l1)?, cpu_slice(s2, l2)?);
        let k = &self.kind;
        let (bd, btd) = (b * d, b * t * d);
        let st = &ps[4 * d..];
        let mut out = vec![0f32; btd + 2 * bd];
        for bi in 0..b {
            for c in 0..d {
                let i = bi * d + c;
                let (bt, th0, rho, omr, gain) = params(ps, d, c);
                let (mut v, mut sp, mut a) = (st[i], st[bd + i], st[2 * bd + i]);
                let mut thp = if k.alif { th0 + a * gain } else { th0 };
                for ti in 0..t {
                    let j = (bi * t + ti) * d + c;
                    step(k, cur[j], bt, th0, rho, omr, gain, &mut v, &mut sp, &mut a, &mut thp);
                    out[j] = sp;
                }
                out[btd + i] = v;
                out[btd + bd + i] = a;
            }
        }
        Ok((CpuStorage::F32(out), Shape::from(btd + 2 * bd)))
    }

    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
        use candle_core::cuda_backend::WrapErr;
        let (b, t, d) = dims(l1, l2)?;
        let dev = s1.device().clone();
        let (cur, ps) = (cuda_slice(s1, l1)?, cuda_slice(s2, l2)?);
        let n = b * t * d + 2 * b * d;
        let out = unsafe { dev.alloc::<f32>(n)? };
        let func = dev.get_or_load_custom_func(&self.kind.kernel("fwd"), "ava_lif", PTX)?;
        let (bi, ti, di) = (b as i32, t as i32, d as i32);
        let mut lb = func.builder();
        lb.arg(&cur).arg(&ps).arg(&out).arg(&bi).arg(&ti).arg(&di);
        unsafe { lb.launch(launch_cfg(b * d)) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev), Shape::from(n)))
    }

    fn bwd(&self, cur: &Tensor, ps: &Tensor, _out: &Tensor, g: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (b, t, d) = cur.dims3()?;
        let dev = cur.device();
        let packed = cur.detach().apply_op3_no_bwd(&ps.detach(), &g.detach().contiguous()?, &LifGenBwd { kind: self.kind })?;
        let (btd, bd) = (b * t * d, b * d);
        let g_cur = packed.narrow(0, 0, btd)?.reshape((b, t, d))?;
        let part = |j: usize| packed.narrow(0, btd + (2 + j) * bd, bd)?.reshape((b, d))?.sum(0);
        let zd = Tensor::zeros(d, DType::F32, dev)?;
        let zbd = Tensor::zeros(bd, DType::F32, dev)?;
        let g_ps = Tensor::cat(
            &[&part(0)?, &zd, &part(1)?, &part(2)?, &packed.narrow(0, btd, bd)?, &zbd, &packed.narrow(0, btd + bd, bd)?],
            0,
        )?;
        Ok((Some(g_cur), Some(g_ps)))
    }
}

/// (cur, ps, grad of the forward's flat output) -> flat [dL/dI (B·T·d) | dL/dv0 | dL/da0 | per-row dL/dbeta, dL/drho,
/// dL/dgain (each B·d)].
struct LifGenBwd {
    kind: Kind,
}

impl CustomOp3 for LifGenBwd {
    fn name(&self) -> &'static str {
        "lif-gen-bwd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b, t, d) = dims(l1, l2)?;
        let (cur, ps, g) = (cpu_slice(s1, l1)?, cpu_slice(s2, l2)?, cpu_slice(s3, l3)?);
        let k = &self.kind;
        let (bd, btd) = (b * d, b * t * d);
        let st = &ps[4 * d..];
        let mut out = vec![0f32; btd + 5 * bd];
        let (mut vh, mut ah) = (vec![0f32; t], vec![0f32; t]);
        for bi in 0..b {
            for c in 0..d {
                let i = bi * d + c;
                let (bt, th0, rho, omr, gain) = params(ps, d, c);
                let (v0, s0, a0) = (st[i], st[bd + i], st[2 * bd + i]);
                let (mut v, mut sp, mut a) = (v0, s0, a0);
                let mut thp = if k.alif { th0 + a0 * gain } else { th0 };
                for ti in 0..t {
                    step(k, cur[(bi * t + ti) * d + c], bt, th0, rho, omr, gain, &mut v, &mut sp, &mut a, &mut thp);
                    vh[ti] = v;
                    ah[ti] = a;
                }
                let (mut cv, mut ca) = (g[btd + i], if k.alif { g[btd + bd + i] } else { 0.0 });
                let (mut gb, mut gr, mut gg) = (0f32, 0f32, 0f32);
                let mut gvn = 0f32; // dL/dv_{t+1}
                for ti in (0..t).rev() {
                    let (vt, at) = (vh[ti], if k.alif { ah[ti] } else { 0.0 });
                    let tht = if k.alif { th0 + at * gain } else { th0 };
                    let (dv, dth) = k.dspike(vt, tht);
                    let gst = if k.rg { g[(bi * t + ti) * d + c] - reset_te(tht, dv, bt) * gvn } else { g[(bi * t + ti) * d + c] };
                    let gv = gst * dv + cv;
                    let vp = if ti > 0 { vh[ti - 1] } else { v0 };
                    gb += gv * vp;
                    if k.alif {
                        let gth = if k.rg { gst * dth - k.spike(vt, tht) * gvn } else { gst * dth };
                        let ga = gth * gain + ca;
                        gg += gth * at;
                        let ap = if ti > 0 { ah[ti - 1] } else { a0 };
                        let spv = if ti > 0 { k.spike(vp, th0 + ap * gain) } else { s0 };
                        gr += ga * (ap - spv.abs());
                        ca = rho * ga;
                    }
                    out[(bi * t + ti) * d + c] = gv;
                    gvn = gv;
                    cv = bt * gv;
                }
                if k.alif && k.rg {
                    // the first step's reset used the initial threshold th0 + a0 gain
                    let gth0 = -s0 * gvn;
                    gg += gth0 * a0;
                    ca += gth0 * gain;
                }
                out[btd + i] = cv;
                out[btd + bd + i] = ca;
                out[btd + 2 * bd + i] = gb;
                out[btd + 3 * bd + i] = gr;
                out[btd + 4 * bd + i] = gg;
            }
        }
        Ok((CpuStorage::F32(out), Shape::from(btd + 5 * bd)))
    }

    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
        use candle_core::cuda_backend::WrapErr;
        let (b, t, d) = dims(l1, l2)?;
        let dev = s1.device().clone();
        let (cur, ps, g) = (cuda_slice(s1, l1)?, cuda_slice(s2, l2)?, cuda_slice(s3, l3)?);
        let n = b * t * d + 5 * b * d;
        let out = unsafe { dev.alloc::<f32>(n)? };
        // the trace's history (E4 only; a one-element placeholder otherwise)
        let ah = unsafe { dev.alloc::<f32>(if self.kind.alif { b * t * d } else { 1 })? };
        let func = dev.get_or_load_custom_func(&self.kind.kernel("bwd"), "ava_lif", PTX)?;
        let (bi, ti, di, slope, rgi) = (b as i32, t as i32, d as i32, self.kind.slope, self.kind.rg as i32);
        let mut lb = func.builder();
        lb.arg(&cur).arg(&ps).arg(&g).arg(&out).arg(&ah).arg(&bi).arg(&ti).arg(&di).arg(&slope).arg(&rgi);
        unsafe { lb.launch(launch_cfg(b * d)) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev), Shape::from(n)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::lif_ref;
    use candle_core::{Device, Var};

    fn maxabs(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
    }

    fn scale(a: &Tensor) -> f32 {
        a.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap().max(1e-6)
    }

    /// Every option and its combinations: lif_gen gives lif_ref's spikes and final state exactly, and its gradients
    /// (to the input current, beta, rho, gain, v0 and a0) within 1e-4 of the largest.
    fn compare(dev: &Device, tol: f32) {
        let (b, t, d) = (3, 37, 8);
        let mut cases = 0;
        let mut flips = 0usize;
        for spikes in [Spikes::Binary, Spikes::Ternary, Spikes::Int2] {
            for alif in [false, true] {
                for theta_on in [false, true] {
                    for init_on in [false, true] {
                        for slope in [4.0f64, 2.0, -4.0] { // -4: the reset path carried (--lif-reset-grad)
                            cases += 1;
                            let seed = cases as u64;
                            let r = |shape: (usize, usize, usize), s: f64| {
                                let n = shape.0 * shape.1 * shape.2;
                                let v: Vec<f32> = (0..n)
                                    .map(|j| {
                                        let x = ((j as u64 * 2654435761 + seed * 97 + 13) % 10007) as f32 / 10007.0;
                                        ((x * 6.2831853).sin() * 1.7 + (x * 51.0).cos()) * s as f32
                                    })
                                    .collect();
                                Tensor::from_vec(v, n, dev).unwrap().reshape(shape).unwrap()
                            };
                            let cur = Var::from_tensor(&r((b, t, d), 1.3)).unwrap();
                            let braw = Var::from_tensor(&r((1, 1, d), 0.8).flatten_all().unwrap()).unwrap();
                            let rraw = Var::from_tensor(&r((1, 1, d), 1.0).affine(1.0, 2.5).unwrap().flatten_all().unwrap()).unwrap();
                            let gain = Var::from_tensor(&r((1, 1, d), 0.5).affine(1.0, 1.0).unwrap().flatten_all().unwrap()).unwrap();
                            let theta = r((1, 1, d), 0.3).affine(1.0, 1.0).unwrap().flatten_all().unwrap();
                            let v0 = Var::from_tensor(&r((1, b, d), 0.7).reshape((b, d)).unwrap()).unwrap();
                            let a0 = Var::from_tensor(&r((1, b, d), 0.3).abs().unwrap().reshape((b, d)).unwrap()).unwrap();
                            let s0 = r((1, b, d), 1.0).ge(0.5).unwrap().to_dtype(DType::F32).unwrap().reshape((b, d)).unwrap();
                            let w1 = r((b, t, d), 1.0);
                            let w2 = r((1, b, d), 0.9).reshape((b, d)).unwrap();
                            let w3 = r((1, b, d), 1.1).reshape((b, d)).unwrap();
                            let run = |gen: bool| {
                                let beta = candle_nn::ops::sigmoid(braw.as_tensor()).unwrap();
                                let rho = candle_nn::ops::sigmoid(rraw.as_tensor()).unwrap();
                                let init = LifState { v: v0.as_tensor().clone(), s: s0.clone(), a: Some(a0.as_tensor().clone()) };
                                let th = if theta_on { Some(&theta) } else { None };
                                let al = if alif { Some((&rho, gain.as_tensor())) } else { None };
                                let ini = if init_on { Some(&init) } else { None };
                                let f = if gen { lif_gen } else { lif_ref };
                                let (s, st) = f(cur.as_tensor(), &beta, th, al, spikes, slope, ini).unwrap();
                                let mut loss = s.mul(&w1).unwrap().sum_all().unwrap().add(&st.v.mul(&w2).unwrap().sum_all().unwrap()).unwrap();
                                if let Some(a) = &st.a {
                                    loss = loss.add(&a.mul(&w3).unwrap().sum_all().unwrap()).unwrap();
                                }
                                let gs = loss.backward().unwrap();
                                let gr = |x: &Var| gs.get(x.as_tensor()).cloned().unwrap_or_else(|| x.as_tensor().zeros_like().unwrap());
                                (s, st, [gr(&cur), gr(&braw), gr(&rraw), gr(&gain), gr(&v0), gr(&a0)])
                            };
                            let (s1, st1, g1) = run(true);
                            let (s2, st2, g2) = run(false);
                            let tag = format!("{spikes:?} alif={alif} theta={theta_on} init={init_on} slope={slope}");
                            let ds = maxabs(&s1, &s2);
                            flips += s1.ne(&s2).unwrap().to_dtype(DType::F32).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() as usize;
                            assert!(ds == 0.0, "{tag}: spikes differ by {ds}");
                            assert!(maxabs(&st1.v, &st2.v) == 0.0, "{tag}: v_T differs");
                            assert!(maxabs(&st1.s, &st2.s) == 0.0, "{tag}: s_T differs");
                            if alif {
                                assert!(maxabs(st1.a.as_ref().unwrap(), st2.a.as_ref().unwrap()) == 0.0, "{tag}: a_T differs");
                            }
                            for (j, name) in ["cur", "beta", "rho", "gain", "v0", "a0"].iter().enumerate() {
                                let e = maxabs(&g1[j], &g2[j]) / scale(&g2[j]);
                                assert!(e < tol, "{tag}: grad {name} rel err {e:.2e}");
                            }
                            // the share of steps that spike (any level, either sign)
                            let rate = s2.ne(0f32).unwrap().to_dtype(DType::F32).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
                            assert!(rate > 0.02 && rate < 0.98, "{tag}: spike rate {rate} too extreme for a useful test");
                        }
                    }
                }
            }
        }
        assert_eq!(flips, 0);
        eprintln!("lif_gen = lif_ref over {cases} cases on {dev:?}");
    }

    /// v52 spike fix: the reset path's cap (lif.cu's lifg_te) in force. Neurons held just under or at the threshold for
    /// many steps with a fast leak (beta 0.27, so th ds/dv > 2 beta near the threshold): lif_gen's backward (slope -4,
    /// the reset carried) against lif_ref's autograd, with the cap active on most steps.
    fn cap_case(dev: &Device, tol: f32) {
        let (b, t, d) = (2, 40, 6);
        let braw_v = -1.0f32;
        let beta_f = 1.0 / (1.0 + (-braw_v).exp());
        let th = LIF_THRESHOLD as f32;
        let m = [0.995f32, 0.999, 1.0005, 1.002, 1.01, 1.05];
        let curv: Vec<f32> = (0..b * t * d)
            .map(|j| {
                let (c, ti, bi) = (j % d, (j / d) % t, j / (d * t));
                (1.0 - beta_f) * th * m[c] + 1e-3 * (((ti * 7 + bi * 3 + c) % 5) as f32 - 2.0)
            })
            .collect();
        // the cap is in force on most steps (binary, no ALIF, worked out here)
        let mut active = 0usize;
        for bi in 0..b {
            for c in 0..d {
                let (mut v, mut sp) = (0f32, 0f32);
                for ti in 0..t {
                    v = v * beta_f + curv[(bi * t + ti) * d + c] - sp * th;
                    sp = if v >= th { 1.0 } else { 0.0 };
                    let dv = 1.0 / (1.0 + 4.0 * (v - th).abs()).powi(2);
                    active += (th * dv > 2.0 * beta_f) as usize;
                }
            }
        }
        let share = active as f32 / (b * t * d) as f32;
        assert!(share > 0.3, "the cap must be in force on many steps (share {share})");
        for spikes in [Spikes::Binary, Spikes::Int2] {
            for alif in [false, true] {
                let cur = Var::from_tensor(&Tensor::from_vec(curv.clone(), (b, t, d), dev).unwrap()).unwrap();
                let braw = Var::from_tensor(&Tensor::full(braw_v, d, dev).unwrap()).unwrap();
                let rraw = Var::from_tensor(&Tensor::full(2.0f32, d, dev).unwrap()).unwrap();
                let gain = Var::from_tensor(&Tensor::full(0.1f32, d, dev).unwrap()).unwrap();
                let w1: Vec<f32> = (0..b * t * d).map(|j| ((j * 37 % 11) as f32 - 5.0) / 5.0).collect();
                let w1 = Tensor::from_vec(w1, (b, t, d), dev).unwrap();
                let w2: Vec<f32> = (0..b * d).map(|j| ((j * 13 % 7) as f32 - 3.0) / 3.0).collect();
                let w2 = Tensor::from_vec(w2, (b, d), dev).unwrap();
                let run = |gen: bool| {
                    let beta = candle_nn::ops::sigmoid(braw.as_tensor()).unwrap();
                    let rho = candle_nn::ops::sigmoid(rraw.as_tensor()).unwrap();
                    let al = if alif { Some((&rho, gain.as_tensor())) } else { None };
                    let f = if gen { lif_gen } else { lif_ref };
                    let (s, st) = f(cur.as_tensor(), &beta, None, al, spikes, -4.0, None).unwrap();
                    let loss = s.mul(&w1).unwrap().sum_all().unwrap().add(&st.v.mul(&w2).unwrap().sum_all().unwrap()).unwrap();
                    let gs = loss.backward().unwrap();
                    let gr = |x: &Var| gs.get(x.as_tensor()).cloned().unwrap_or_else(|| x.as_tensor().zeros_like().unwrap());
                    (s, [gr(&cur), gr(&braw), gr(&gain)])
                };
                let (s1, g1) = run(true);
                let (s2, g2) = run(false);
                let tag = format!("cap case {spikes:?} alif={alif}");
                assert!(maxabs(&s1, &s2) == 0.0, "{tag}: spikes differ");
                for (j, name) in ["cur", "beta", "gain"].iter().enumerate() {
                    if alif || j < 2 {
                        let e = maxabs(&g1[j], &g2[j]) / scale(&g2[j]);
                        assert!(e < tol, "{tag}: grad {name} rel err {e:.2e}");
                    }
                }
            }
        }
    }

    #[test]
    fn lif_gen_reset_cap_matches_lif_ref_cpu() {
        cap_case(&Device::Cpu, 1e-4);
    }

    /// The reset cap on the GPU (run only when the GPU is free: --ignored).
    #[test]
    #[ignore]
    fn lif_gen_reset_cap_matches_lif_ref_gpu() {
        cap_case(&Device::new_cuda(0).expect("no CUDA device"), 1e-4);
    }

    #[test]
    fn lif_gen_matches_lif_ref_cpu() {
        compare(&Device::Cpu, 1e-4);
    }

    /// The CUDA kernels against lif_ref on the GPU (run only when the GPU is free: --ignored).
    #[test]
    #[ignore]
    fn lif_gen_matches_lif_ref_gpu() {
        let dev = Device::new_cuda(0).expect("no CUDA device");
        compare(&dev, 1e-4);
    }
}
