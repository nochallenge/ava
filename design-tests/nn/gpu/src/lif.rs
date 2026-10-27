// TESTING / DESIGN ONLY — ava_gpu::lif: the fused LIF scan as a candle custom op.
//
// `lif(cur [B,T,D], beta [D])` returns the spikes [B,T,D] (exactly 0/1). Its backward is written by
// hand: the fast-sigmoid surrogate 1/(1 + 4|v - theta|)^2, the reset detached, full BPTT through the
// leak, and dL/dbeta. Same maths as model::lif_scan_unrolled (the straight-through graph), one kernel
// launch per layer and direction instead of ~10 per time step. CUDA kernels in src/lif.cu; the CPU
// path runs the same loops in Rust.

use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};

use crate::model::{LIF_SLOPE, LIF_THRESHOLD};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/lif.ptx"));

/// Spikes of the LIF scan over the time axis of `cur` [B, T, D] with leak `beta` [D] (already in (0,1)).
pub fn lif(cur: &Tensor, beta: &Tensor) -> Result<Tensor> {
    cur.contiguous()?.apply_op2(&beta.contiguous()?, LifScan)
}

fn dims(l1: &Layout, l2: &Layout) -> Result<(usize, usize, usize)> {
    let (b, t, d) = l1.shape().dims3()?;
    if l2.shape().dims() != [d] {
        candle_core::bail!("lif: beta has shape {:?}, expected [{d}]", l2.shape().dims());
    }
    Ok((b, t, d))
}

fn cpu_slice<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let v = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(&v[a..b]),
        None => candle_core::bail!("lif: input must be contiguous"),
    }
}

struct LifScan;

impl CustomOp2 for LifScan {
    fn name(&self) -> &'static str {
        "lif-scan"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, t, d) = dims(l1, l2)?;
        let (cur, beta) = (cpu_slice(s1, l1)?, cpu_slice(s2, l2)?);
        let theta = LIF_THRESHOLD as f32;
        let mut s = vec![0f32; b * t * d];
        for bi in 0..b {
            for c in 0..d {
                let (mut v, mut sp) = (0f32, 0f32);
                for ti in 0..t {
                    let k = (bi * t + ti) * d + c;
                    v = beta[c] * v + cur[k] - theta * sp;
                    sp = if v - theta >= 0.0 { 1.0 } else { 0.0 };
                    s[k] = sp;
                }
            }
        }
        Ok((CpuStorage::F32(s), l1.shape().clone()))
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
        let cur = cuda_slice(s1, l1)?;
        let beta = cuda_slice(s2, l2)?;
        let out = unsafe { dev.alloc::<f32>(b * t * d)? };
        let func = dev.get_or_load_custom_func("lif_fwd", "ava_lif", PTX)?;
        let (bi, ti, di, theta) = (b as i32, t as i32, d as i32, LIF_THRESHOLD as f32);
        let mut lb = func.builder();
        lb.arg(&cur).arg(&beta).arg(&out).arg(&bi).arg(&ti).arg(&di).arg(&theta);
        unsafe { lb.launch(launch_cfg(b * d)) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev), l1.shape().clone()))
    }

    fn bwd(&self, cur: &Tensor, beta: &Tensor, _s: &Tensor, grad_s: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (b, t, d) = cur.dims3()?;
        let packed = cur.detach().apply_op3_no_bwd(&beta.detach(), &grad_s.detach().contiguous()?, &LifScanBwd)?;
        let g_cur = packed.narrow(1, 0, t)?;
        let g_beta = packed.narrow(1, t, 1)?.reshape((b, d))?.sum(0)?;
        Ok((Some(g_cur), Some(g_beta)))
    }
}

/// (cur, beta, grad of spikes) -> [B, T+1, D]: dL/dI in rows 0..T, per-batch-row dL/dbeta in row T.
struct LifScanBwd;

impl CustomOp3 for LifScanBwd {
    fn name(&self) -> &'static str {
        "lif-scan-bwd"
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
        let (cur, beta, gs) = (cpu_slice(s1, l1)?, cpu_slice(s2, l2)?, cpu_slice(s3, l3)?);
        let (theta, slope) = (LIF_THRESHOLD as f32, LIF_SLOPE as f32);
        let mut out = vec![0f32; b * (t + 1) * d];
        let mut v_hist = vec![0f32; t];
        for bi in 0..b {
            for c in 0..d {
                let (mut v, mut sp) = (0f32, 0f32);
                for ti in 0..t {
                    v = beta[c] * v + cur[(bi * t + ti) * d + c] - theta * sp;
                    sp = if v - theta >= 0.0 { 1.0 } else { 0.0 };
                    v_hist[ti] = v;
                }
                let (mut gv_next, mut gb) = (0f32, 0f32);
                for ti in (0..t).rev() {
                    let a = 1.0 + slope * (v_hist[ti] - theta).abs();
                    let gv = gs[(bi * t + ti) * d + c] / (a * a) + beta[c] * gv_next;
                    if ti > 0 {
                        gb += gv * v_hist[ti - 1];
                    }
                    out[(bi * (t + 1) + ti) * d + c] = gv;
                    gv_next = gv;
                }
                out[(bi * (t + 1) + t) * d + c] = gb;
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((b, t + 1, d))))
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
        let (cur, beta, gs) = (cuda_slice(s1, l1)?, cuda_slice(s2, l2)?, cuda_slice(s3, l3)?);
        let out = unsafe { dev.alloc::<f32>(b * (t + 1) * d)? };
        let func = dev.get_or_load_custom_func("lif_bwd", "ava_lif", PTX)?;
        let (bi, ti, di) = (b as i32, t as i32, d as i32);
        let (theta, slope) = (LIF_THRESHOLD as f32, LIF_SLOPE as f32);
        let mut lb = func.builder();
        lb.arg(&cur).arg(&beta).arg(&gs).arg(&out).arg(&bi).arg(&ti).arg(&di).arg(&theta).arg(&slope);
        unsafe { lb.launch(launch_cfg(b * d)) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev), Shape::from((b, t + 1, d))))
    }
}

fn cuda_slice<'a>(
    s: &'a candle_core::CudaStorage,
    l: &Layout,
) -> Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>> {
    let v = s.as_cuda_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(v.slice(a..b)),
        None => candle_core::bail!("lif: input must be contiguous"),
    }
}

fn launch_cfg(threads: usize) -> candle_core::cuda_backend::cudarc::driver::LaunchConfig {
    const BLOCK: u32 = 128;
    candle_core::cuda_backend::cudarc::driver::LaunchConfig {
        grid_dim: ((threads as u32).div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// 0/1 kWTA mask of `u` [N, H] (no gradient): 1 where u >= the k-th largest value of its row. One CUDA
/// block per row finds that value by radix select (`kwta_mask_rows` in lif.cu), instead of sorting every
/// row; the mask is the same as model::kwta_mask_sorted gives.
pub fn kwta_mask_fast(u: &Tensor, k: usize) -> Result<Tensor> {
    u.detach().contiguous()?.apply_op1_no_bwd(&KwtaMask { k })
}

struct KwtaMask {
    k: usize,
}

impl candle_core::CustomOp1 for KwtaMask {
    fn name(&self) -> &'static str {
        "kwta-mask"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let (n, h) = l.shape().dims2()?;
        let u = cpu_slice(s, l)?;
        let mut m = vec![0f32; n * h];
        let mut row = vec![0f32; h];
        for r in 0..n {
            let x = &u[r * h..(r + 1) * h];
            row.copy_from_slice(x);
            row.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let kth = row[self.k.min(h) - 1];
            for (o, &v) in m[r * h..(r + 1) * h].iter_mut().zip(x) {
                *o = if v - kth >= 0.0 { 1.0 } else { 0.0 };
            }
        }
        Ok((CpuStorage::F32(m), l.shape().clone()))
    }

    fn cuda_fwd(&self, s: &candle_core::CudaStorage, l: &Layout) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        let (n, h) = l.shape().dims2()?;
        if h > 12288 {
            candle_core::bail!("kwta-mask: rows of {h} exceed the shared-memory limit");
        }
        let dev = s.device().clone();
        let u = cuda_slice(s, l)?;
        let out = unsafe { dev.alloc::<f32>(n * h)? };
        let func = dev.get_or_load_custom_func("kwta_mask_rows", "ava_lif", PTX)?;
        let (ni, hi, ki) = (n as i32, h as i32, self.k.min(h) as i32);
        let mut lb = func.builder();
        lb.arg(&u).arg(&out).arg(&ni).arg(&hi).arg(&ki);
        let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (h * 4) as u32 };
        unsafe { lb.launch(cfg) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev), l.shape().clone()))
    }
}

/// Mean cross-entropy (nats) of `logits` [N, V] against `targets` [N] (u32; a target >= V counts as 0
/// and gets no gradient, but still counts in the mean's N, as with candle's cross_entropy only if no
/// such targets occur). One kernel per row forward (log-sum-exp and the target's logit) and one backward,
/// instead of candle's chain of full-size [N, V] ops. Rounding differs from candle's in the last bits.
pub fn cross_entropy_fused(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let out = logits.contiguous()?.apply_op2(&targets.contiguous()?, CeFwd)?; // [N, 2] = (loss, lse)
    out.narrow(1, 0, 1)?.mean_all()
}

fn u32_cpu<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [u32]> {
    let v = s.as_slice::<u32>()?;
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(&v[a..b]),
        None => candle_core::bail!("cross-entropy: targets must be contiguous"),
    }
}

struct CeFwd;

impl CustomOp2 for CeFwd {
    fn name(&self) -> &'static str {
        "ce-fwd"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, Shape)> {
        let (n, v) = l1.shape().dims2()?;
        let (x, tg) = (cpu_slice(s1, l1)?, u32_cpu(s2, l2)?);
        let mut out = vec![0f32; 2 * n];
        for r in 0..n {
            let row = &x[r * v..(r + 1) * v];
            let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let lse = m + row.iter().map(|&z| (z - m).exp()).sum::<f32>().ln();
            let t = tg[r] as usize;
            out[2 * r] = if t < v { lse - row[t] } else { 0.0 };
            out[2 * r + 1] = lse;
        }
        Ok((CpuStorage::F32(out), Shape::from((n, 2))))
    }

    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        let (n, v) = l1.shape().dims2()?;
        let dev = s1.device().clone();
        let x = cuda_slice(s1, l1)?;
        let tg = match l2.contiguous_offsets() {
            Some((a, b)) => s2.as_cuda_slice::<u32>()?.slice(a..b),
            None => candle_core::bail!("cross-entropy: targets must be contiguous"),
        };
        let out = unsafe { dev.alloc::<f32>(2 * n)? };
        let func = dev.get_or_load_custom_func("ce_fwd", "ava_lif", PTX)?;
        let (ni, vi) = (n as i32, v as i32);
        let mut lb = func.builder();
        lb.arg(&x).arg(&tg).arg(&out).arg(&ni).arg(&vi);
        let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { lb.launch(cfg) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev), Shape::from((n, 2))))
    }

    fn bwd(&self, logits: &Tensor, targets: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let aux = Tensor::cat(&[&grad_res.narrow(1, 0, 1)?, &res.narrow(1, 1, 1)?], 1)?.contiguous()?;
        let gx = logits.contiguous()?.apply_op3_no_bwd(&targets.contiguous()?, &aux, &CeBwd)?;
        Ok((Some(gx), None))
    }
}

struct CeBwd;

impl CustomOp3 for CeBwd {
    fn name(&self) -> &'static str {
        "ce-bwd"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout, s3: &CpuStorage, l3: &Layout) -> Result<(CpuStorage, Shape)> {
        let (n, v) = l1.shape().dims2()?;
        let (x, tg, aux) = (cpu_slice(s1, l1)?, u32_cpu(s2, l2)?, cpu_slice(s3, l3)?);
        let mut gx = vec![0f32; n * v];
        for r in 0..n {
            let t = tg[r] as usize;
            let (g, lse) = (if t < v { aux[2 * r] } else { 0.0 }, aux[2 * r + 1]);
            for i in 0..v {
                gx[r * v + i] = ((x[r * v + i] - lse).exp() - if i == t { 1.0 } else { 0.0 }) * g;
            }
        }
        Ok((CpuStorage::F32(gx), l1.shape().clone()))
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
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        let (n, v) = l1.shape().dims2()?;
        let dev = s1.device().clone();
        let x = cuda_slice(s1, l1)?;
        let tg = match l2.contiguous_offsets() {
            Some((a, b)) => s2.as_cuda_slice::<u32>()?.slice(a..b),
            None => candle_core::bail!("cross-entropy: targets must be contiguous"),
        };
        let aux = cuda_slice(s3, l3)?;
        let gx = unsafe { dev.alloc::<f32>(n * v)? };
        let func = dev.get_or_load_custom_func("ce_bwd", "ava_lif", PTX)?;
        let (ni, vi) = (n as i32, v as i32);
        let mut lb = func.builder();
        lb.arg(&x).arg(&tg).arg(&aux).arg(&gx).arg(&ni).arg(&vi);
        let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { lb.launch(cfg) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(gx, dev), l1.shape().clone()))
    }
}

// ---- raw entry points for the hand-written S backward (model::Model::loss_and_grads): no autograd

/// Spikes of the fused LIF scan, untracked. `beta` is the leak in (0, 1) (already through the sigmoid).
pub fn lif_fwd_raw(cur: &Tensor, beta: &Tensor) -> Result<Tensor> {
    cur.contiguous()?.apply_op2_no_bwd(&beta.contiguous()?, &LifScan)
}

/// (dL/dcur [B, T, D], dL/dbeta [D]) of the fused LIF scan for the spike gradient `grad_s`.
pub fn lif_bwd_raw(cur: &Tensor, beta: &Tensor, grad_s: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, t, d) = cur.dims3()?;
    let packed = cur.contiguous()?.apply_op3_no_bwd(&beta.contiguous()?, &grad_s.contiguous()?, &LifScanBwd)?;
    Ok((packed.narrow(1, 0, t)?, packed.narrow(1, t, 1)?.reshape((b, d))?.sum(0)?))
}

/// Per row (loss, lse) of the fused cross-entropy, untracked: [N, 2].
pub fn ce_fwd_raw(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    logits.contiguous()?.apply_op2_no_bwd(&targets.contiguous()?, &CeFwd)
}

/// dL/dlogits of the fused cross-entropy; aux [N, 2] = (dL/dloss_row, lse).
pub fn ce_bwd_raw(logits: &Tensor, targets: &Tensor, aux: &Tensor) -> Result<Tensor> {
    logits.contiguous()?.apply_op3_no_bwd(&targets.contiguous()?, &aux.contiguous()?, &CeBwd)
}
