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
