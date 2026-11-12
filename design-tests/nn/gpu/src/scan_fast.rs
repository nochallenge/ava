// TESTING / DESIGN ONLY — ava_gpu::scan_fast: three Wave 5 parts' time scans as fused ops.
//
// E7 (resonators), E9 (short-term plasticity) and B9 (working memory) each run a per-channel recurrence over time,
// which their ava_parts code does as a few candle ops per step: 12-18x slower than the rest of the model. Here each
// scan is one kernel per direction (res_*, stp_*, wm_* in src/lif.cu), one thread per channel:
// - forward: the parts' f32 operations in the same order, so the same values;
// - backward: written by hand, the same gradients as the parts' autograd up to rounding.
// The rest of each part (gates, read-out, spike, mix) stays the parts' own candle ops. The CPU path runs the same
// loops (for the tests). Off switch: AVA_SCAN_REF=1 runs the parts' own code.

use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor, D};

use crate::lif::{cuda_slice, launch_cfg};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/lif.ptx"));

/// The fused scans are on unless AVA_SCAN_REF=1.
pub fn on() -> bool {
    std::env::var("AVA_SCAN_REF").map_or(true, |v| v != "1")
}

fn cpu_slice<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let v = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(&v[a..b]),
        None => candle_core::bail!("scan_fast: input must be contiguous"),
    }
}

/// [B, T, C] and a per-channel parameter vector of `np` blocks of C.
fn btc(l1: &Layout, l2: &Layout, np: usize, what: &str) -> Result<(usize, usize, usize)> {
    let (b, t, c) = l1.shape().dims3()?;
    if l2.shape().dims() != [np * c] || t == 0 {
        candle_core::bail!("{what}: shapes {:?} and {:?} do not fit", l1.shape(), l2.shape());
    }
    Ok((b, t, c))
}

// ------------------------------------------------------------------------------------------------------ E7

/// E7: the part's spikes for `cur` [B, T, C] (ava_parts::resonate::Resonate::forward with the fused membrane).
pub fn res_forward(p: &ava_parts::resonate::Resonate, cur: &Tensor, theta: f64, slope: f64) -> Result<Tensor> {
    if !on() {
        return p.forward(cur, theta, slope);
    }
    ava_parts::spike(&(res_membrane(p, cur)? - theta)?, slope)
}

/// E7: the membrane Re z [B, T, C], as Resonate::membrane.
pub fn res_membrane(p: &ava_parts::resonate::Resonate, cur: &Tensor) -> Result<Tensor> {
    let r = candle_nn::ops::sigmoid(&p.r_param)?;
    let om = p.omega()?;
    let (rc, rs) = ((&r * om.cos()?)?, (&r * om.sin()?)?);
    let ps = Tensor::cat(&[&rc, &rs], 0)?.contiguous()?;
    cur.contiguous()?.apply_op2(&ps, ResScan)
}

fn res_cpu(cur: &[f32], ps: &[f32], b: usize, t: usize, c: usize, mut each: impl FnMut(usize, f32, f32)) {
    for bi in 0..b {
        for ci in 0..c {
            let (rc, rs) = (ps[ci], ps[c + ci]);
            let (mut re, mut im) = (0f32, 0f32);
            for ti in 0..t {
                let j = (bi * t + ti) * c + ci;
                let re2 = (re * rc - im * rs) + cur[j];
                im = re * rs + im * rc;
                re = re2;
                each(j, re, im);
            }
        }
    }
}

struct ResScan;

impl CustomOp2 for ResScan {
    fn name(&self) -> &'static str {
        "res-scan"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, t, c) = btc(l1, l2, 2, "res")?;
        let mut out = vec![0f32; b * t * c];
        res_cpu(cpu_slice(s1, l1)?, cpu_slice(s2, l2)?, b, t, c, |j, re, _| out[j] = re);
        Ok((CpuStorage::F32(out), l1.shape().clone()))
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
        let (b, t, c) = btc(l1, l2, 2, "res")?;
        let dev = s1.device().clone();
        let (cur, ps) = (cuda_slice(s1, l1)?, cuda_slice(s2, l2)?);
        let out = unsafe { dev.alloc::<f32>(b * t * c)? };
        let func = dev.get_or_load_custom_func("res_fwd", "ava_lif", PTX)?;
        let (bi, ti, ci) = (b as i32, t as i32, c as i32);
        let mut lb = func.builder();
        lb.arg(&cur).arg(&ps).arg(&out).arg(&bi).arg(&ti).arg(&ci);
        unsafe { lb.launch(launch_cfg(b * c)) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev), l1.shape().clone()))
    }

    fn bwd(&self, cur: &Tensor, ps: &Tensor, _out: &Tensor, g: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (b, t, c) = cur.dims3()?;
        let packed = cur.detach().apply_op3_no_bwd(&ps.detach(), &g.detach().contiguous()?, &ResScanBwd)?;
        let btc = b * t * c;
        let g_cur = packed.narrow(0, 0, btc)?.reshape((b, t, c))?;
        let g_ps = packed.narrow(0, btc, 2 * b * c)?.reshape((2, b, c))?.sum(1)?.flatten_all()?;
        Ok((Some(g_cur), Some(g_ps)))
    }
}

/// (cur, ps, dL/dRe z) -> flat [dL/dI (B·T·C) | per-row dL/drc (B·C) | per-row dL/drs (B·C)].
struct ResScanBwd;

impl CustomOp3 for ResScanBwd {
    fn name(&self) -> &'static str {
        "res-scan-bwd"
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
        let (b, t, c) = btc(l1, l2, 2, "res")?;
        let (cur, ps, g) = (cpu_slice(s1, l1)?, cpu_slice(s2, l2)?, cpu_slice(s3, l3)?);
        let (n, bc) = (b * t * c, b * c);
        let (mut reh, mut imh) = (vec![0f32; n], vec![0f32; n]);
        res_cpu(cur, ps, b, t, c, |j, re, im| {
            reh[j] = re;
            imh[j] = im;
        });
        let mut out = vec![0f32; n + 2 * bc];
        for bi in 0..b {
            for ci in 0..c {
                let (rc, rs) = (ps[ci], ps[c + ci]);
                let (mut gr, mut gi, mut grc, mut grs) = (0f32, 0f32, 0f32, 0f32);
                for ti in (0..t).rev() {
                    let j = (bi * t + ti) * c + ci;
                    grc += gr * reh[j] + gi * imh[j];
                    grs += gi * reh[j] - gr * imh[j];
                    let nr = g[j] + rc * gr + rs * gi;
                    gi = rc * gi - rs * gr;
                    gr = nr;
                    out[j] = gr;
                }
                out[n + bi * c + ci] = grc;
                out[n + bc + bi * c + ci] = grs;
            }
        }
        Ok((CpuStorage::F32(out), Shape::from(n + 2 * bc)))
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
        let (b, t, c) = btc(l1, l2, 2, "res")?;
        let dev = s1.device().clone();
        let (cur, ps, g) = (cuda_slice(s1, l1)?, cuda_slice(s2, l2)?, cuda_slice(s3, l3)?);
        let n = b * t * c + 2 * b * c;
        let out = unsafe { dev.alloc::<f32>(n)? };
        let ih = unsafe { dev.alloc::<f32>(b * t * c)? };
        let func = dev.get_or_load_custom_func("res_bwd", "ava_lif", PTX)?;
        let (bi, ti, ci) = (b as i32, t as i32, c as i32);
        let mut lb = func.builder();
        lb.arg(&cur).arg(&ps).arg(&g).arg(&out).arg(&ih).arg(&bi).arg(&ti).arg(&ci);
        unsafe { lb.launch(launch_cfg(b * c)) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev), Shape::from(n)))
    }
}

// ------------------------------------------------------------------------------------------------------ E9

/// E9: the part's output for spikes `s` [B, T, C] (ava_parts::stp::Stp::forward(..).out with the fused scan).
pub fn stp_forward(p: &ava_parts::stp::Stp, s: &Tensor) -> Result<Tensor> {
    if !on() {
        return Ok(p.forward(s)?.out);
    }
    // the synapse is driven by the event size (ava_parts::stp::events; = s for 0/1 spikes)
    let efficacy = stp_efficacy(p, &ava_parts::stp::events(s)?)?;
    s + ((s * &efficacy)? - s)?.broadcast_mul(&p.mix)?
}

/// E9: e/U at each step [B, T, C], as Stp::forward's efficacy.
pub fn stp_efficacy(p: &ava_parts::stp::Stp, s: &Tensor) -> Result<Tensor> {
    use candle_nn::ops::sigmoid;
    let ps = Tensor::cat(&[&sigmoid(&p.u_param)?, &sigmoid(&p.f_param)?, &sigmoid(&p.d_param)?], 0)?.contiguous()?;
    s.contiguous()?.apply_op2(&ps, StpScan)
}

/// v51 fast1 (src/fwin_r2.rs, the fused step's --stp): the scan's efficacy for events `s` [B, T, C] and the packed
/// parameters `ps` [3 C] (sigmoid of U, df, dd), outside autograd.
pub(crate) fn stp_scan_nb(s: &Tensor, ps: &Tensor) -> Result<Tensor> {
    s.contiguous()?.apply_op2_no_bwd(ps, &StpScan)
}

/// v51 fast1: StpScan's backward outside autograd (the same kernel): (dL/ds [B, T, C], dL/dps [3 C]) for dL/deff `g`.
pub(crate) fn stp_scan_bwd(s: &Tensor, ps: &Tensor, g: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, t, c) = s.dims3()?;
    let packed = s.contiguous()?.apply_op3_no_bwd(ps, &g.contiguous()?, &StpScanBwd)?;
    let btc = b * t * c;
    let g_s = packed.narrow(0, 0, btc)?.reshape((b, t, c))?;
    let g_ps = packed.narrow(0, btc, 3 * b * c)?.reshape((3, b, c))?.sum(1)?.flatten_all()?;
    Ok((g_s, g_ps))
}

/// One step from (u, x): returns (up, e, eff, xu) and moves (u, x) on.
#[inline]
fn stp_step(s: f32, uu: f32, df: f32, dd: f32, u: &mut f32, x: &mut f32) -> (f32, f32, f32, f32) {
    let up = *u + ((1.0 - *u) * uu) * s;
    let e = up * *x;
    let eff = e / uu;
    let xu = *x - e * s;
    let st = (up, e, eff, xu);
    *u = up * df;
    *x = 1.0 - (1.0 - xu) * dd;
    st
}

struct StpScan;

impl CustomOp2 for StpScan {
    fn name(&self) -> &'static str {
        "stp-scan"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, t, c) = btc(l1, l2, 3, "stp")?;
        let (s, ps) = (cpu_slice(s1, l1)?, cpu_slice(s2, l2)?);
        let mut out = vec![0f32; b * t * c];
        for bi in 0..b {
            for ci in 0..c {
                let (uu, df, dd) = (ps[ci], ps[c + ci], ps[2 * c + ci]);
                let (mut u, mut x) = (0f32, 1f32);
                for ti in 0..t {
                    let j = (bi * t + ti) * c + ci;
                    out[j] = stp_step(s[j], uu, df, dd, &mut u, &mut x).2;
                }
            }
        }
        Ok((CpuStorage::F32(out), l1.shape().clone()))
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
        let (b, t, c) = btc(l1, l2, 3, "stp")?;
        let dev = s1.device().clone();
        let (s, ps) = (cuda_slice(s1, l1)?, cuda_slice(s2, l2)?);
        let out = unsafe { dev.alloc::<f32>(b * t * c)? };
        let func = dev.get_or_load_custom_func("stp_fwd", "ava_lif", PTX)?;
        let (bi, ti, ci) = (b as i32, t as i32, c as i32);
        let mut lb = func.builder();
        lb.arg(&s).arg(&ps).arg(&out).arg(&bi).arg(&ti).arg(&ci);
        unsafe { lb.launch(launch_cfg(b * c)) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev), l1.shape().clone()))
    }

    fn bwd(&self, s: &Tensor, ps: &Tensor, _out: &Tensor, g: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (b, t, c) = s.dims3()?;
        let packed = s.detach().apply_op3_no_bwd(&ps.detach(), &g.detach().contiguous()?, &StpScanBwd)?;
        let btc = b * t * c;
        let g_s = packed.narrow(0, 0, btc)?.reshape((b, t, c))?;
        let g_ps = packed.narrow(0, btc, 3 * b * c)?.reshape((3, b, c))?.sum(1)?.flatten_all()?;
        Ok((Some(g_s), Some(g_ps)))
    }
}

/// (s, ps, dL/deff) -> flat [dL/ds (B·T·C) | per-row dL/dU, dL/ddf, dL/ddd (B·C each)].
struct StpScanBwd;

impl CustomOp3 for StpScanBwd {
    fn name(&self) -> &'static str {
        "stp-scan-bwd"
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
        let (b, t, c) = btc(l1, l2, 3, "stp")?;
        let (s, ps, g) = (cpu_slice(s1, l1)?, cpu_slice(s2, l2)?, cpu_slice(s3, l3)?);
        let (n, bc) = (b * t * c, b * c);
        let mut out = vec![0f32; n + 3 * bc];
        let (mut uh, mut xh) = (vec![0f32; t], vec![0f32; t]);
        for bi in 0..b {
            for ci in 0..c {
                let (uu, df, dd) = (ps[ci], ps[c + ci], ps[2 * c + ci]);
                let (mut u, mut x) = (0f32, 1f32);
                for ti in 0..t {
                    uh[ti] = u;
                    xh[ti] = x;
                    stp_step(s[(bi * t + ti) * c + ci], uu, df, dd, &mut u, &mut x);
                }
                let (mut gu, mut gx, mut g_u, mut gdf, mut gdd) = (0f32, 0f32, 0f32, 0f32, 0f32);
                for ti in (0..t).rev() {
                    let j = (bi * t + ti) * c + ci;
                    let (ut, xt, st, ge) = (uh[ti], xh[ti], s[j], g[j]);
                    let a = 1.0 - ut;
                    let up = ut + a * uu * st;
                    let e = up * xt;
                    let xu = xt - e * st;
                    let gxu = gx * dd;
                    gdd -= gx * (1.0 - xu);
                    let mut gup = gu * df;
                    gdf += gu * up;
                    let mut gxt = gxu;
                    let gee = -gxu * st + ge / uu;
                    let mut gst = -gxu * e;
                    g_u -= ge * e / (uu * uu);
                    gup += gee * xt;
                    gxt += gee * up;
                    g_u += gup * a * st;
                    gst += gup * a * uu;
                    gu = gup - gup * uu * st;
                    gx = gxt;
                    out[j] = gst;
                }
                out[n + bi * c + ci] = g_u;
                out[n + bc + bi * c + ci] = gdf;
                out[n + 2 * bc + bi * c + ci] = gdd;
            }
        }
        Ok((CpuStorage::F32(out), Shape::from(n + 3 * bc)))
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
        let (b, t, c) = btc(l1, l2, 3, "stp")?;
        let dev = s1.device().clone();
        let (s, ps, g) = (cuda_slice(s1, l1)?, cuda_slice(s2, l2)?, cuda_slice(s3, l3)?);
        let n = b * t * c + 3 * b * c;
        let out = unsafe { dev.alloc::<f32>(n)? };
        let xh = unsafe { dev.alloc::<f32>(b * t * c)? };
        let func = dev.get_or_load_custom_func("stp_bwd", "ava_lif", PTX)?;
        let (bi, ti, ci) = (b as i32, t as i32, c as i32);
        let mut lb = func.builder();
        lb.arg(&s).arg(&ps).arg(&g).arg(&out).arg(&xh).arg(&bi).arg(&ti).arg(&ci);
        unsafe { lb.launch(launch_cfg(b * c)) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev), Shape::from(n)))
    }
}

// ------------------------------------------------------------------------------------------------------ B9

/// B9: what each token adds to its stream [B, T, d] (ava_parts::wm::SlotMemory::forward with the fused scan).
pub fn wm_forward(p: &ava_parts::wm::SlotMemory, h: &Tensor) -> Result<Tensor> {
    if !on() {
        return p.forward(h);
    }
    let (b, t, _) = h.dims3()?;
    let g = p.gates(h)?;
    let v = ava_parts::linear(h, &p.wv)?.reshape((b, t, p.slots, p.dk))?;
    let m = wm_scan(&g, &v)?;
    let q = ava_parts::linear(h, &p.wq)?.unsqueeze(2)?;
    let att = (m.broadcast_mul(&q)?.sum(D::Minus1)? / (p.dk as f64).sqrt())?;
    let att = candle_nn::ops::softmax(&att, D::Minus1)?.unsqueeze(3)?;
    let r = m.broadcast_mul(&att)?.sum(2)?;
    ava_parts::linear(&r, &p.wo)
}

/// B9: the slots after each token [B, T, K, dk] for gates [B, T, K] and values [B, T, K, dk], as SlotMemory::scan.
pub fn wm_scan(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    v.contiguous()?.apply_op2(&g.contiguous()?, WmScan)
}

fn wm_dims(l1: &Layout, l2: &Layout) -> Result<(usize, usize, usize, usize)> {
    let (b, t, k, dk) = l1.shape().dims4()?;
    if l2.shape().dims() != [b, t, k] || t == 0 {
        candle_core::bail!("wm: shapes {:?} and {:?} do not fit", l1.shape(), l2.shape());
    }
    Ok((b, t, k, dk))
}

struct WmScan;

impl CustomOp2 for WmScan {
    fn name(&self) -> &'static str {
        "wm-scan"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, t, k, dk) = wm_dims(l1, l2)?;
        let (v, g) = (cpu_slice(s1, l1)?, cpu_slice(s2, l2)?);
        let mut out = vec![0f32; b * t * k * dk];
        for bi in 0..b {
            for ki in 0..k {
                for c in 0..dk {
                    let mut m = 0f32;
                    for ti in 0..t {
                        let gt = g[(bi * t + ti) * k + ki];
                        let j = ((bi * t + ti) * k + ki) * dk + c;
                        m = m * (1.0 - gt) + v[j] * gt;
                        out[j] = m;
                    }
                }
            }
        }
        Ok((CpuStorage::F32(out), l1.shape().clone()))
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
        let (b, t, k, dk) = wm_dims(l1, l2)?;
        let dev = s1.device().clone();
        let (v, g) = (cuda_slice(s1, l1)?, cuda_slice(s2, l2)?);
        let out = unsafe { dev.alloc::<f32>(b * t * k * dk)? };
        let func = dev.get_or_load_custom_func("wm_fwd", "ava_lif", PTX)?;
        let (bi, ti, ki, di) = (b as i32, t as i32, k as i32, dk as i32);
        let mut lb = func.builder();
        lb.arg(&v).arg(&g).arg(&out).arg(&bi).arg(&ti).arg(&ki).arg(&di);
        unsafe { lb.launch(launch_cfg(b * k * dk)) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev), l1.shape().clone()))
    }

    fn bwd(&self, v: &Tensor, g: &Tensor, _m: &Tensor, gm: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (b, t, k, dk) = v.dims4()?;
        let packed = v.detach().apply_op3_no_bwd(&g.detach(), &gm.detach().contiguous()?, &WmScanBwd)?;
        let n = b * t * k * dk;
        let dv = packed.narrow(0, 0, n)?.reshape((b, t, k, dk))?;
        let dg = packed.narrow(0, n, n)?.reshape((b, t, k, dk))?.sum(3)?;
        Ok((Some(dv), Some(dg)))
    }
}

/// (v, g, dL/dm) -> flat [dL/dv | each element's share of dL/dg], both [B, T, K, dk].
struct WmScanBwd;

impl CustomOp3 for WmScanBwd {
    fn name(&self) -> &'static str {
        "wm-scan-bwd"
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
        let (b, t, k, dk) = wm_dims(l1, l2)?;
        let (v, g, gm) = (cpu_slice(s1, l1)?, cpu_slice(s2, l2)?, cpu_slice(s3, l3)?);
        let n = b * t * k * dk;
        let mut out = vec![0f32; 2 * n];
        let mut mh = vec![0f32; t];
        for bi in 0..b {
            for ki in 0..k {
                for c in 0..dk {
                    let at = |ti: usize| ((bi * t + ti) * k + ki) * dk + c;
                    let mut m = 0f32;
                    for ti in 0..t {
                        let gt = g[(bi * t + ti) * k + ki];
                        m = m * (1.0 - gt) + v[at(ti)] * gt;
                        mh[ti] = m;
                    }
                    let mut carry = 0f32;
                    for ti in (0..t).rev() {
                        let gt = g[(bi * t + ti) * k + ki];
                        let gg = gm[at(ti)] + carry;
                        let mp = if ti > 0 { mh[ti - 1] } else { 0.0 };
                        out[n + at(ti)] = gg * (v[at(ti)] - mp);
                        out[at(ti)] = gg * gt;
                        carry = gg * (1.0 - gt);
                    }
                }
            }
        }
        Ok((CpuStorage::F32(out), Shape::from(2 * n)))
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
        let (b, t, k, dk) = wm_dims(l1, l2)?;
        let dev = s1.device().clone();
        let (v, g, gm) = (cuda_slice(s1, l1)?, cuda_slice(s2, l2)?, cuda_slice(s3, l3)?);
        let n = b * t * k * dk;
        let out = unsafe { dev.alloc::<f32>(2 * n)? };
        let func = dev.get_or_load_custom_func("wm_bwd", "ava_lif", PTX)?;
        let (bi, ti, ki, di) = (b as i32, t as i32, k as i32, dk as i32);
        let mut lb = func.builder();
        lb.arg(&v).arg(&g).arg(&gm).arg(&out).arg(&bi).arg(&ti).arg(&ki).arg(&di);
        unsafe { lb.launch(launch_cfg(b * k * dk)) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev), Shape::from(2 * n)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Var};
    use candle_nn::{VarBuilder, VarMap};

    fn maxabs(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
    }

    fn scale(a: &Tensor) -> f32 {
        a.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap().max(1e-6)
    }

    fn wave(n: usize, s: f32, salt: u64, dev: &Device) -> Tensor {
        let v: Vec<f32> = (0..n)
            .map(|j| {
                let x = ((j as u64 * 2654435761 + salt * 7919 + 13) % 10007) as f32 / 10007.0;
                ((x * 6.2831853).sin() * 1.7 + (x * 51.0).cos()) * s
            })
            .collect();
        Tensor::from_vec(v, n, dev).unwrap()
    }

    /// Set every variable of `vm` to a varied value (so the parts are not at their no-change start).
    fn perturb(vm: &VarMap, dev: &Device) {
        for (salt, v) in vm.all_vars().iter().enumerate() {
            let n = v.elem_count();
            let x = v.as_tensor().add(&wave(n, 0.4, salt as u64 + 100, dev).reshape(v.shape()).unwrap()).unwrap();
            v.set(&x).unwrap();
        }
    }

    fn grads_of(vm: &VarMap, inputs: &[&Var], loss: &Tensor) -> Vec<Tensor> {
        let gs = loss.backward().unwrap();
        let mut vars: Vec<Var> = inputs.iter().map(|v| (*v).clone()).collect();
        let mut named: Vec<(String, Var)> = vm.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));
        vars.extend(named.into_iter().map(|(_, v)| v));
        vars.iter().map(|v| gs.get(v.as_tensor()).cloned().unwrap_or_else(|| v.as_tensor().zeros_like().unwrap())).collect()
    }

    fn check(tag: &str, a: &(Tensor, Vec<Tensor>), b: &(Tensor, Vec<Tensor>), exact: bool, tol: f32) {
        let d = maxabs(&a.0, &b.0);
        if exact {
            assert!(d == 0.0, "{tag}: forward differs by {d}");
        } else {
            assert!(d / scale(&b.0) < tol, "{tag}: forward rel err {:.2e}", d / scale(&b.0));
        }
        for (j, (x, y)) in a.1.iter().zip(b.1.iter()).enumerate() {
            let e = maxabs(x, y) / scale(y);
            assert!(e < tol, "{tag}: grad {j} rel err {e:.2e}");
        }
    }

    /// The three fused scans give the parts' own forward (exactly: the same ops in the same order) and their
    /// gradients (to the inputs and every weight) within 1e-4 of the largest.
    fn compare(dev: &Device) {
        let (b, t, c) = (3, 37, 16);
        let tol = 1e-4;
        // E7
        {
            let vm = VarMap::new();
            let p = ava_parts::resonate::Resonate::new(VarBuilder::from_varmap(&vm, DType::F32, dev), c, 64.0).unwrap();
            perturb(&vm, dev);
            let cur = Var::from_tensor(&wave(b * t * c, 1.0, 1, dev).reshape((b, t, c)).unwrap()).unwrap();
            let w = wave(b * t * c, 1.0, 2, dev).reshape((b, t, c)).unwrap();
            let run = |fast: bool| {
                let m = if fast { res_membrane(&p, cur.as_tensor()).unwrap() } else { p.membrane(cur.as_tensor()).unwrap() };
                let loss = m.mul(&w).unwrap().sum_all().unwrap();
                (m, grads_of(&vm, &[&cur], &loss))
            };
            check("E7 membrane", &run(true), &run(false), true, tol);
            let sp = |fast: bool| {
                let s = if fast { res_forward(&p, cur.as_tensor(), 1.0, 4.0).unwrap() } else { p.forward(cur.as_tensor(), 1.0, 4.0).unwrap() };
                let loss = s.mul(&w).unwrap().sum_all().unwrap();
                (s, grads_of(&vm, &[&cur], &loss))
            };
            check("E7 spikes", &sp(true), &sp(false), true, tol);
        }
        // E9
        {
            let vm = VarMap::new();
            let p = ava_parts::stp::Stp::new(VarBuilder::from_varmap(&vm, DType::F32, dev), c).unwrap();
            perturb(&vm, dev);
            // graded spikes in [-2, 2] with zeros, as int2 gives
            let s0 = wave(b * t * c, 1.4, 3, dev).round().unwrap().reshape((b, t, c)).unwrap();
            let s = Var::from_tensor(&s0).unwrap();
            let w = wave(b * t * c, 1.0, 4, dev).reshape((b, t, c)).unwrap();
            let run = |fast: bool| {
                let o = if fast { stp_forward(&p, s.as_tensor()).unwrap() } else { p.forward(s.as_tensor()).unwrap().out };
                let loss = o.mul(&w).unwrap().sum_all().unwrap();
                (o, grads_of(&vm, &[&s], &loss))
            };
            check("E9", &run(true), &run(false), true, tol);
        }
        // B9, both gatings
        for gating in [ava_parts::wm::Gating::Independent, ava_parts::wm::Gating::Compete { k: 2 }] {
            let (d, slots, dk) = (24, 4, 8);
            let vm = VarMap::new();
            let p = ava_parts::wm::SlotMemory::new(VarBuilder::from_varmap(&vm, DType::F32, dev), d, slots, dk, gating).unwrap();
            perturb(&vm, dev);
            let h = Var::from_tensor(&wave(b * t * d, 1.0, 5, dev).reshape((b, t, d)).unwrap()).unwrap();
            let w = wave(b * t * d, 1.0, 6, dev).reshape((b, t, d)).unwrap();
            let run = |fast: bool| {
                let o = if fast { wm_forward(&p, h.as_tensor()).unwrap() } else { p.forward(h.as_tensor()).unwrap() };
                let loss = o.mul(&w).unwrap().sum_all().unwrap();
                (o, grads_of(&vm, &[&h], &loss))
            };
            check(&format!("B9 {gating:?}"), &run(true), &run(false), true, tol);
        }
        eprintln!("scan_fast = the parts on {dev:?}");
    }

    #[test]
    fn fused_scans_match_the_parts_cpu() {
        compare(&Device::Cpu);
    }

    /// The CUDA kernels against the parts on the GPU (run only when the GPU is free: --ignored).
    #[test]
    #[ignore]
    fn fused_scans_match_the_parts_gpu() {
        compare(&Device::new_cuda(0).expect("no CUDA device"));
    }
}
