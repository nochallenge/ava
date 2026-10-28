// TESTING / DESIGN ONLY — ava_gpu::fused: speed round 2, fused CUDA ops for the training step.
//
// Each op replaces a chain of candle primitives (every one a full pass over memory, plus a zeros + add
// per gradient in candle's autograd) by one kernel forward and one or two backward. Same maths; the
// rounding differs from the primitive chains in the last bits, so a run with these ops is checked
// against the primitive path on a trial before it is used (never inside a running pre-registered test).
// CUDA only: the model keeps the primitive path on the CPU. Kernels in src/lif.cu.
//
//   rms_norm(x [N, D], g [D])   y = x / sqrt(mean(x^2) + eps) * g; backward dx and dg (fixed-order sums)
//   kwta_relu(u [N, H], k)      relu(u) on the k largest of each row, 0 elsewhere; backward ga where a > 0
//   adamw(w, g, m, v, ...)      one AdamW update per parameter tensor in one pass: new (w, m, v)

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::CudaDevice;
use candle_core::cuda_backend::WrapErr;
use candle_core::op::BackpropOp;
use candle_core::{CpuStorage, CudaStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Storage, Tensor, D};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/lif.ptx"));
const MODULE: &str = "ava_lif";
/// Rows per partial sum of the rmsnorm gain gradient.
const GAIN_ROWS: usize = 64;

fn view<'a>(s: &'a CudaStorage, l: &Layout, what: &str) -> Result<CudaView<'a, f32>> {
    let v = s.as_cuda_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(v.slice(a..b)),
        None => candle_core::bail!("{what}: input must be contiguous"),
    }
}

fn no_cpu(what: &str) -> Result<(CpuStorage, Shape)> {
    candle_core::bail!("{what}: CUDA only (the model uses the primitive path on the CPU)")
}

fn rows_cfg(rows: usize, threads: u32, shared: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: shared }
}

fn elems_cfg(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

/// Threads per row block for a row of `d` values (a multiple of 32, at most 1024).
fn row_threads(d: usize) -> u32 {
    (d.div_ceil(32) * 32).clamp(32, 256) as u32
}

// ---------------------------------------------------------------- rmsnorm

pub fn rms_norm(x: &Tensor, g: &Tensor, eps: f64) -> Result<Tensor> {
    x.contiguous()?.apply_op2(&g.contiguous()?, RmsNorm { eps: eps as f32 })
}

struct RmsNorm {
    eps: f32,
}

impl CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "rms-norm"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu("rms-norm")
    }

    fn cuda_fwd(&self, s1: &CudaStorage, l1: &Layout, s2: &CudaStorage, l2: &Layout) -> Result<(CudaStorage, Shape)> {
        let (n, d) = l1.shape().dims2()?;
        let dev = s1.device().clone();
        let (x, g) = (view(s1, l1, "rms-norm")?, view(s2, l2, "rms-norm")?);
        let y = unsafe { dev.alloc::<f32>(n * d)? };
        let func = dev.get_or_load_custom_func("rms_fwd", MODULE, PTX)?;
        let (ni, di) = (n as i32, d as i32);
        let mut lb = func.builder();
        lb.arg(&x).arg(&g).arg(&y).arg(&ni).arg(&di).arg(&self.eps);
        unsafe { lb.launch(rows_cfg(n, row_threads(d), 0)) }.w()?;
        Ok((CudaStorage::wrap_cuda_slice(y, dev), l1.shape().clone()))
    }

    fn bwd(&self, x: &Tensor, g: &Tensor, _res: &Tensor, gy: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let n = x.dim(0)?;
        let both = x.contiguous()?.apply_op3_no_bwd(&g.contiguous()?, &gy.contiguous()?, &RmsNormBwd { eps: self.eps })?;
        let dx = both.narrow(0, 0, n)?;
        let dg = both.narrow(0, n, both.dim(0)? - n)?.sum(0)?;
        Ok((Some(dx), Some(dg)))
    }
}

/// Returns [N + P, D]: rows 0..N are dx, rows N.. are the P partial sums of dg (P = ceil(N / GAIN_ROWS)).
struct RmsNormBwd {
    eps: f32,
}

impl CustomOp3 for RmsNormBwd {
    fn name(&self) -> &'static str {
        "rms-norm-bwd"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu("rms-norm-bwd")
    }

    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
        s3: &CudaStorage,
        l3: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (n, d) = l1.shape().dims2()?;
        let p = n.div_ceil(GAIN_ROWS);
        let dev = s1.device().clone();
        let (x, g, gy) = (view(s1, l1, "rms-norm-bwd")?, view(s2, l2, "rms-norm-bwd")?, view(s3, l3, "rms-norm-bwd")?);
        let out = unsafe { dev.alloc::<f32>((n + p) * d)? };
        let invs = unsafe { dev.alloc::<f32>(n)? };
        let (ni, di, ri) = (n as i32, d as i32, GAIN_ROWS as i32);
        {
            let dx = out.slice(0..n * d);
            let func = dev.get_or_load_custom_func("rms_bwd_rows", MODULE, PTX)?;
            let mut lb = func.builder();
            lb.arg(&x).arg(&g).arg(&gy).arg(&dx).arg(&invs).arg(&ni).arg(&di).arg(&self.eps);
            unsafe { lb.launch(rows_cfg(n, row_threads(d), 0)) }.w()?;
        }
        {
            let part = out.slice(n * d..(n + p) * d);
            let func = dev.get_or_load_custom_func("rms_bwd_gain", MODULE, PTX)?;
            let mut lb = func.builder();
            lb.arg(&x).arg(&gy).arg(&invs).arg(&part).arg(&ni).arg(&di).arg(&ri);
            unsafe { lb.launch(rows_cfg(p, (d.div_ceil(32) * 32).min(1024) as u32, 0)) }.w()?;
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from((n + p, d))))
    }
}

// ---------------------------------------------------------------- kWTA + relu

pub fn kwta_relu(u: &Tensor, k: usize) -> Result<Tensor> {
    if k >= u.dim(D::Minus1)? {
        return u.relu();
    }
    u.contiguous()?.apply_op1(KwtaRelu { k })
}

struct KwtaRelu {
    k: usize,
}

impl CustomOp1 for KwtaRelu {
    fn name(&self) -> &'static str {
        "kwta-relu"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu("kwta-relu")
    }

    fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
        let (n, h) = l.shape().dims2()?;
        if h > 12288 {
            candle_core::bail!("kwta-relu: rows of {h} exceed the shared-memory limit");
        }
        let dev = s.device().clone();
        let u = view(s, l, "kwta-relu")?;
        let out = unsafe { dev.alloc::<f32>(n * h)? };
        let func = dev.get_or_load_custom_func("kwta_relu_rows", MODULE, PTX)?;
        let (ni, hi, ki) = (n as i32, h as i32, self.k.min(h) as i32);
        let mut lb = func.builder();
        lb.arg(&u).arg(&out).arg(&ni).arg(&hi).arg(&ki);
        unsafe { lb.launch(rows_cfg(n, 256, (h * 4) as u32)) }.w()?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), l.shape().clone()))
    }

    fn bwd(&self, _u: &Tensor, res: &Tensor, ga: &Tensor) -> Result<Option<Tensor>> {
        Ok(Some(res.contiguous()?.apply_op2_no_bwd(&ga.contiguous()?, &ReluMaskBwd)?))
    }
}

struct ReluMaskBwd;

impl CustomOp2 for ReluMaskBwd {
    fn name(&self) -> &'static str {
        "relu-mask-bwd"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu("relu-mask-bwd")
    }

    fn cuda_fwd(&self, s1: &CudaStorage, l1: &Layout, s2: &CudaStorage, l2: &Layout) -> Result<(CudaStorage, Shape)> {
        let n = l1.shape().elem_count();
        let dev = s1.device().clone();
        let (a, ga) = (view(s1, l1, "relu-mask-bwd")?, view(s2, l2, "relu-mask-bwd")?);
        let gu = unsafe { dev.alloc::<f32>(n)? };
        let func = dev.get_or_load_custom_func("relu_mask_bwd", MODULE, PTX)?;
        let ni = n as i32;
        let mut lb = func.builder();
        lb.arg(&a).arg(&ga).arg(&gu).arg(&ni);
        unsafe { lb.launch(elems_cfg(n)) }.w()?;
        Ok((CudaStorage::wrap_cuda_slice(gu, dev), l1.shape().clone()))
    }
}

// ---------------------------------------------------------------- AdamW

/// The scalars of one AdamW step (all f32 on the device, computed in f64 on the host).
pub struct AdamWScalars {
    pub grad_scale: f64,
    pub b1: f64,
    pub b2: f64,
    pub eps: f64,
    pub step_size: f64,
    pub inv_sqrt_bc2: f64,
    pub shrink: f64,
}

/// One AdamW update of a parameter tensor in one pass. Returns [3, numel]: new w, m, v (flattened).
pub fn adamw(w: &Tensor, g: &Tensor, m: &Tensor, v: &Tensor, sc: &AdamWScalars) -> Result<Tensor> {
    let n = w.elem_count();
    let (w, g, m, v) = (w.contiguous()?, g.contiguous()?, m.contiguous()?, v.contiguous()?);
    let (ws, wl) = w.storage_and_layout();
    let (gs, gl) = g.storage_and_layout();
    let (ms, ml) = m.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let (Storage::Cuda(ws), Storage::Cuda(gs), Storage::Cuda(ms), Storage::Cuda(vs)) = (&*ws, &*gs, &*ms, &*vs) else {
        candle_core::bail!("adamw: CUDA only");
    };
    let dev = ws.device().clone();
    let (wv, gv, mv, vv) = (view(ws, wl, "adamw")?, view(gs, gl, "adamw")?, view(ms, ml, "adamw")?, view(vs, vl, "adamw")?);
    let out = unsafe { dev.alloc::<f32>(3 * n)? };
    let func = dev.get_or_load_custom_func("adamw_step", MODULE, PTX)?;
    let ni = n as i32;
    let f = |x: f64| x as f32;
    let (s, b1, c1, b2, c2) = (f(sc.grad_scale), f(sc.b1), f(1.0 - sc.b1), f(sc.b2), f(1.0 - sc.b2));
    let (eps, step, ibc2, shrink) = (f(sc.eps), f(sc.step_size), f(sc.inv_sqrt_bc2), f(sc.shrink));
    let mut lb = func.builder();
    lb.arg(&wv).arg(&gv).arg(&mv).arg(&vv).arg(&out).arg(&ni);
    lb.arg(&s).arg(&b1).arg(&c1).arg(&b2).arg(&c2).arg(&eps).arg(&step).arg(&ibc2).arg(&shrink);
    unsafe { lb.launch(elems_cfg(n)) }.w()?;
    let storage = Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev));
    Ok(Tensor::from_storage(storage, (3, n), BackpropOp::none(), false))
}

// ---------------------------------------------------------------- raw entry points (no autograd)

pub fn rms_fwd_raw(x: &Tensor, g: &Tensor, eps: f64) -> Result<Tensor> {
    x.contiguous()?.apply_op2_no_bwd(&g.contiguous()?, &RmsNorm { eps: eps as f32 })
}

/// (dx, dg) of rmsnorm for the output gradient `gy`.
pub fn rms_bwd_raw(x: &Tensor, g: &Tensor, gy: &Tensor, eps: f64) -> Result<(Tensor, Tensor)> {
    let n = x.dim(0)?;
    let both = x.contiguous()?.apply_op3_no_bwd(&g.contiguous()?, &gy.contiguous()?, &RmsNormBwd { eps: eps as f32 })?;
    Ok((both.narrow(0, 0, n)?, both.narrow(0, n, both.dim(0)? - n)?.sum(0)?))
}

pub fn kwta_relu_raw(u: &Tensor, k: usize) -> Result<Tensor> {
    if k >= u.dim(D::Minus1)? {
        return u.relu();
    }
    u.contiguous()?.apply_op1_no_bwd(&KwtaRelu { k })
}

/// The gradient of kwta_relu: `ga` where `a` > 0, else 0.
pub fn relu_mask_bwd_raw(a: &Tensor, ga: &Tensor) -> Result<Tensor> {
    a.contiguous()?.apply_op2_no_bwd(&ga.contiguous()?, &ReluMaskBwd)
}

// ---------------------------------------------------------------- causal attention (speed round 4)

/// Head size the attention kernels are written for (one warp lane per output dim).
pub const ATTN_HS: usize = 32;
/// Query (or key) rows per block: 4 warps x 8 rows (must match AT in lif.cu).
const ATTN_ROWS: u32 = 32;

/// softmax(q kᵀ · scale + causal mask) v for q, k, v [B, H, T, 32], without storing the [T, T] scores.
/// Same maths as the candle chain; the backward recomputes the scores from the saved log-sum-exp.
pub fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let both = q.contiguous()?.apply_op3(&k.contiguous()?, &v.contiguous()?, Attn { scale: scale as f32 })?;
    both.narrow(3, 0, ATTN_HS)
}

/// Output [B, H, T, 33]: the attention output (32) then the row's log-sum-exp.
struct Attn {
    scale: f32,
}

fn attn_dims(l: &Layout) -> Result<(usize, usize)> {
    let d = l.shape().dims();
    if d.len() != 4 || d[3] != ATTN_HS {
        candle_core::bail!("causal_attention: want [B, H, T, {ATTN_HS}], got {:?}", d);
    }
    Ok((d[0] * d[1], d[2]))
}

fn attn_cfg(bh: usize, t: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((t as u32).div_ceil(ATTN_ROWS), bh as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }
}

impl CustomOp3 for Attn {
    fn name(&self) -> &'static str {
        "ava-causal-attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu("causal_attention")
    }

    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
        s3: &CudaStorage,
        l3: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (bh, t) = attn_dims(l1)?;
        let dev = s1.device().clone();
        let (q, k, v) = (view(s1, l1, "attn q")?, view(s2, l2, "attn k")?, view(s3, l3, "attn v")?);
        let out = unsafe { dev.alloc::<f32>(bh * t * (ATTN_HS + 1))? };
        let func = dev.get_or_load_custom_func("attn_fwd", MODULE, PTX)?;
        let (bhi, ti) = (bh as i32, t as i32);
        let mut lb = func.builder();
        lb.arg(&q).arg(&k).arg(&v).arg(&out).arg(&bhi).arg(&ti).arg(&self.scale);
        unsafe { lb.launch(attn_cfg(bh, t)) }.w()?;
        let d = l1.shape().dims();
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from((d[0], d[1], t, ATTN_HS + 1))))
    }

    fn bwd(&self, q: &Tensor, k: &Tensor, v: &Tensor, res: &Tensor, gres: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let (dq, dk, dv) = attn_bwd(q, k, v, res, &gres.narrow(3, 0, ATTN_HS)?, self.scale)?;
        Ok((Some(dq), Some(dk), Some(dv)))
    }
}

/// (dq, dk, dv) of causal_attention for the output gradient `dout` [B, H, T, 32]. Deterministic (no atomics).
fn attn_bwd(q: &Tensor, k: &Tensor, v: &Tensor, res: &Tensor, dout: &Tensor, scale: f32) -> Result<(Tensor, Tensor, Tensor)> {
    let (q, k, v, res, dout) = (q.contiguous()?, k.contiguous()?, v.contiguous()?, res.contiguous()?, dout.contiguous()?);
    let shape = q.shape().clone();
    let (bh, t) = attn_dims(q.layout())?;
    let n = bh * t * ATTN_HS;
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let (rs, rl) = res.storage_and_layout();
    let (ds, dl) = dout.storage_and_layout();
    let (Storage::Cuda(qs), Storage::Cuda(ks), Storage::Cuda(vs), Storage::Cuda(rs), Storage::Cuda(ds)) = (&*qs, &*ks, &*vs, &*rs, &*ds) else {
        candle_core::bail!("causal_attention backward: CUDA only");
    };
    let dev = qs.device().clone();
    let (qv, kv, vv) = (view(qs, ql, "attn q")?, view(ks, kl, "attn k")?, view(vs, vl, "attn v")?);
    let (rv, dov) = (view(rs, rl, "attn out")?, view(ds, dl, "attn dout")?);
    let dq = unsafe { dev.alloc::<f32>(n)? };
    let dk = unsafe { dev.alloc::<f32>(n)? };
    let dv = unsafe { dev.alloc::<f32>(n)? };
    let dvec = unsafe { dev.alloc::<f32>(bh * t)? };
    let (bhi, ti) = (bh as i32, t as i32);
    let fq = dev.get_or_load_custom_func("attn_bwd_q", MODULE, PTX)?;
    let mut lb = fq.builder();
    lb.arg(&qv).arg(&kv).arg(&vv).arg(&rv).arg(&dov).arg(&dq).arg(&dvec).arg(&bhi).arg(&ti).arg(&scale);
    unsafe { lb.launch(attn_cfg(bh, t)) }.w()?;
    let fkv = dev.get_or_load_custom_func("attn_bwd_kv", MODULE, PTX)?;
    let mut lb = fkv.builder();
    lb.arg(&qv).arg(&kv).arg(&vv).arg(&rv).arg(&dov).arg(&dvec).arg(&dk).arg(&dv).arg(&bhi).arg(&ti).arg(&scale);
    unsafe { lb.launch(attn_cfg(bh, t)) }.w()?;
    let wrap = |s| Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(s, dev.clone())), shape.clone(), BackpropOp::none(), false);
    Ok((wrap(dq), wrap(dk), wrap(dv)))
}

// ---------------------------------------------------------------- embedding backward and GELU (speed round 5)

/// Largest row count embed_bwd lists in shared memory (48 KB of i32); beyond it candle's index_add is used.
const EMBED_MAX_ROWS: usize = 12 * 1024;

/// `base` [V, d] plus dx [N, d] scattered to rows ids [N] (u32): candle's index_add(ids, dx, 0) on `base`, same
/// summation order (bit-identical), one block per vocabulary row instead of one thread per column.
pub fn embed_bwd_add(base: &Tensor, ids: &Tensor, dx: &Tensor) -> Result<Tensor> {
    let n = ids.elem_count();
    if !base.device().is_cuda() || n > EMBED_MAX_ROWS {
        return base.index_add(ids, dx, 0);
    }
    let (v, d) = base.dims2()?;
    let (base, ids, dx) = (base.contiguous()?, ids.contiguous()?, dx.contiguous()?);
    let (bs, bl) = base.storage_and_layout();
    let (is, il) = ids.storage_and_layout();
    let (xs, xl) = dx.storage_and_layout();
    let (Storage::Cuda(bs), Storage::Cuda(is), Storage::Cuda(xs)) = (&*bs, &*is, &*xs) else {
        candle_core::bail!("embed_bwd_add: CUDA only");
    };
    let dev = bs.device().clone();
    let ids_v = match il.contiguous_offsets() {
        Some((a, b)) => is.as_cuda_slice::<u32>()?.slice(a..b),
        None => candle_core::bail!("embed_bwd_add: ids must be contiguous"),
    };
    let (bv, xv) = (view(bs, bl, "embed base")?, view(xs, xl, "embed dx")?);
    let out = unsafe { dev.alloc::<f32>(v * d)? };
    let func = dev.get_or_load_custom_func("embed_bwd", MODULE, PTX)?;
    let (ni, di) = (n as i32, d as i32);
    let mut lb = func.builder();
    lb.arg(&bv).arg(&ids_v).arg(&xv).arg(&out).arg(&ni).arg(&di);
    unsafe { lb.launch(rows_cfg(v, 256, (n * 4) as u32)) }.w()?;
    Ok(Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)), (v, d), BackpropOp::none(), false))
}

/// v39 glance, backward of the place embedding: d place[p] = sum of dx over the tokens whose place is p, for the
/// `places` real rows (p < places); tokens of place `places` ("none", most of them) are dropped. On CUDA the
/// embed_bwd kernel runs with only the real rows, so no block sums the "none" tokens (one block used to add up
/// ~95% of dx for a row that was thrown away); each real row's sum is in the same order as before, so the result
/// is bit-identical to `embed_bwd_add` on places + 1 rows narrowed to the first `places`.
pub fn place_bwd(pid: &Tensor, dx: &Tensor, places: usize) -> Result<Tensor> {
    let d = dx.dim(D::Minus1)?;
    let n = pid.elem_count();
    if dx.device().is_cuda() && n <= EMBED_MAX_ROWS {
        return embed_bwd_add(&Tensor::zeros((places, d), DType::F32, dx.device())?, pid, dx);
    }
    embed_bwd_add(&Tensor::zeros((places + 1, d), DType::F32, dx.device())?, pid, dx)?.narrow(0, 0, places)?.contiguous()
}

/// v39 glance: the place of every token of `ids` [B, T] (u32), as [B·T] u32: 0 = units, 1 = tens, ...,
/// 16 = none (the rule of ava_data::numbers::place_ids, applied per row; one CUDA thread per token, or that
/// function itself on the CPU).
pub fn place_ids(ids: &Tensor) -> Result<Tensor> {
    let (b, t) = ids.dims2()?;
    let n = b * t;
    if n == 0 || !ids.device().is_cuda() {
        let v: Vec<u32> = ids.flatten_all()?.to_vec1()?;
        let mut out = Vec::with_capacity(n);
        for row in v.chunks(t.max(1)) {
            out.extend(ava_data::numbers::place_ids(row));
        }
        return Tensor::from_vec(out, n, ids.device());
    }
    let ids = ids.contiguous()?;
    let (is, il) = ids.storage_and_layout();
    let Storage::Cuda(is) = &*is else {
        candle_core::bail!("place_ids: CUDA only");
    };
    let dev = is.device().clone();
    let ids_v = match il.contiguous_offsets() {
        Some((a, b)) => is.as_cuda_slice::<u32>()?.slice(a..b),
        None => candle_core::bail!("place_ids: ids must be contiguous"),
    };
    let out = unsafe { dev.alloc::<u32>(n.max(1))? };
    let func = dev.get_or_load_custom_func("place_ids", MODULE, PTX)?;
    let (ni, ti) = (n as i32, t as i32);
    let mut lb = func.builder();
    lb.arg(&ids_v).arg(&out).arg(&ni).arg(&ti);
    unsafe { lb.launch(elems_cfg(n.max(1))) }.w()?;
    Ok(Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)), n, BackpropOp::none(), false))
}

/// tok[ids] with a backward through embed_bwd_add (the T path's embedding; autograd).
pub fn embedding(tok: &Tensor, ids: &Tensor) -> Result<Tensor> {
    tok.contiguous()?.apply_op2(&ids.flatten_all()?.contiguous()?, Embed)
}

struct Embed;

impl CustomOp2 for Embed {
    fn name(&self) -> &'static str {
        "ava-embedding"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu("embedding")
    }

    fn cuda_fwd(&self, s1: &CudaStorage, l1: &Layout, s2: &CudaStorage, l2: &Layout) -> Result<(CudaStorage, Shape)> {
        let out = s1.index_select(s2, l1, l2, 0)?;
        Ok((out, Shape::from((l2.shape().elem_count(), l1.shape().dims()[1]))))
    }

    fn bwd(&self, tok: &Tensor, ids: &Tensor, _res: &Tensor, gy: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        Ok((Some(embed_bwd_add(&tok.zeros_like()?, ids, gy)?), None))
    }
}

/// candle's tanh GELU in one kernel each way (same formulas).
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    x.contiguous()?.apply_op1(Gelu)
}

struct Gelu;

fn unary_launch(name: &str, ins: &[&CudaView<'_, f32>], dev: &candle_core::CudaDevice, n: usize) -> Result<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>> {
    let out = unsafe { dev.alloc::<f32>(n)? };
    let func = dev.get_or_load_custom_func(name, MODULE, PTX)?;
    let ni = n as i32;
    let mut lb = func.builder();
    for v in ins {
        lb.arg(*v);
    }
    lb.arg(&out).arg(&ni);
    unsafe { lb.launch(elems_cfg(n)) }.w()?;
    Ok(out)
}

impl CustomOp1 for Gelu {
    fn name(&self) -> &'static str {
        "ava-gelu"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu("gelu")
    }

    fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
        let dev = s.device().clone();
        let x = view(s, l, "gelu")?;
        let out = unary_launch("gelu_fwd", &[&x], &dev, l.shape().elem_count())?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), l.shape().clone()))
    }

    fn bwd(&self, x: &Tensor, _res: &Tensor, gy: &Tensor) -> Result<Option<Tensor>> {
        Ok(Some(gelu_bwd_raw(x, gy)?))
    }
}

/// gy * gelu'(x), candle's formula, one kernel.
pub fn gelu_bwd_raw(x: &Tensor, gy: &Tensor) -> Result<Tensor> {
    x.contiguous()?.apply_op2_no_bwd(&gy.contiguous()?, &GeluBwd)
}

struct GeluBwd;

impl CustomOp2 for GeluBwd {
    fn name(&self) -> &'static str {
        "ava-gelu-bwd"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu("gelu backward")
    }

    fn cuda_fwd(&self, s1: &CudaStorage, l1: &Layout, s2: &CudaStorage, l2: &Layout) -> Result<(CudaStorage, Shape)> {
        let dev = s1.device().clone();
        let (x, gy) = (view(s1, l1, "gelu bwd x")?, view(s2, l2, "gelu bwd gy")?);
        let out = unary_launch("gelu_bwd", &[&x, &gy], &dev, l1.shape().elem_count())?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), l1.shape().clone()))
    }
}

/// causal_attention without autograd: [B, H, T, 33] (output, then the row log-sum-exp), for attn_bwd_raw.
pub fn causal_attention_raw(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    q.contiguous()?.apply_op3_no_bwd(&k.contiguous()?, &v.contiguous()?, &Attn { scale: scale as f32 })
}

/// (dq, dk, dv) for the output gradient `dout` [B, H, T, 32]; `res` from causal_attention_raw.
pub fn attn_bwd_raw(q: &Tensor, k: &Tensor, v: &Tensor, res: &Tensor, dout: &Tensor, scale: f64) -> Result<(Tensor, Tensor, Tensor)> {
    attn_bwd(q, k, v, res, dout, scale as f32)
}

// ---------------------------------------------------------------- bf16 matmul (speed round 7)

/// C [M, N] (f32) = op(A) · op(B) for row-major bf16 A and B (op = transpose when the flag is set), tensor cores with
/// f32 accumulation and an f32 result (cublasGemmEx: no bf16 rounding of the output, no cast afterwards).
pub fn mm_bf16(a: &Tensor, ta: bool, b: &Tensor, tb: bool) -> Result<Tensor> {
    use candle_core::cuda_backend::cudarc::cublas::sys as bsys;
    let (out, dev, m, n) = gemm_bf16::<f32>(a, ta, b, tb, bsys::cudaDataType_t::CUDA_R_32F)?;
    Ok(Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)), (m, n), BackpropOp::none(), false))
}

/// mm_bf16 with a bf16 result (rounded once from the f32 accumulator): for operands that only feed bf16 kernels.
pub fn mm_bf16_b(a: &Tensor, ta: bool, b: &Tensor, tb: bool) -> Result<Tensor> {
    use candle_core::cuda_backend::cudarc::cublas::sys as bsys;
    let (out, dev, m, n) = gemm_bf16::<half::bf16>(a, ta, b, tb, bsys::cudaDataType_t::CUDA_R_16BF)?;
    Ok(wrap_bf16(out, dev, &Shape::from((m, n))))
}

fn gemm_bf16<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    a: &Tensor,
    ta: bool,
    b: &Tensor,
    tb: bool,
    ctype: candle_core::cuda_backend::cudarc::cublas::sys::cudaDataType_t,
) -> Result<(candle_core::cuda_backend::cudarc::driver::CudaSlice<T>, candle_core::CudaDevice, usize, usize)> {
    use candle_core::cuda_backend::cudarc::cublas::{result as blas, sys as bsys};
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;
    let (ar, ac) = a.dims2()?;
    let (br, bc) = b.dims2()?;
    let (m, k) = if ta { (ac, ar) } else { (ar, ac) };
    let (k2, n) = if tb { (bc, br) } else { (br, bc) };
    if k != k2 || a.dtype() != DType::BF16 || b.dtype() != DType::BF16 {
        candle_core::bail!("mm_bf16: bad operands {:?}{} x {:?}{} ({:?}, {:?})", a.dims(), if ta { "T" } else { "" }, b.dims(), if tb { "T" } else { "" }, a.dtype(), b.dtype());
    }
    let (a, b) = (a.contiguous()?, b.contiguous()?);
    let (as_, al) = a.storage_and_layout();
    let (bs, bl) = b.storage_and_layout();
    let (Storage::Cuda(as_), Storage::Cuda(bs)) = (&*as_, &*bs) else {
        candle_core::bail!("mm_bf16: CUDA only");
    };
    let (CudaStorageSlice::BF16(asl), CudaStorageSlice::BF16(bsl)) = (&as_.slice, &bs.slice) else {
        candle_core::bail!("mm_bf16: bf16 storage expected");
    };
    let (Some((a0, a1)), Some((b0, b1))) = (al.contiguous_offsets(), bl.contiguous_offsets()) else {
        candle_core::bail!("mm_bf16: contiguous operands expected");
    };
    let (av, bv) = (asl.slice(a0..a1), bsl.slice(b0..b1));
    let dev = as_.device().clone();
    let mut out = unsafe { dev.alloc::<T>(m * n)? };
    let blas_h = dev.cublas_handle();
    let stream = dev.cuda_stream();
    let (alpha, beta) = (1.0f32, 0.0f32);
    // Row-major C = op(A) op(B) is column-major C^T = op(B)^T op(A)^T: cuBLAS gets B first.
    let (opb, ldb_) = if tb { (bsys::cublasOperation_t::CUBLAS_OP_T, k) } else { (bsys::cublasOperation_t::CUBLAS_OP_N, n) };
    let (opa, lda_) = if ta { (bsys::cublasOperation_t::CUBLAS_OP_T, m) } else { (bsys::cublasOperation_t::CUBLAS_OP_N, k) };
    {
        let (bp, _gb) = bv.device_ptr(&stream);
        let (ap, _ga) = av.device_ptr(&stream);
        let (cp, _gc) = out.device_ptr_mut(&stream);
        unsafe {
            blas::gemm_ex(
                *blas_h.handle(),
                opb,
                opa,
                n as i32,
                m as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                bp as *const _,
                bsys::cudaDataType_t::CUDA_R_16BF,
                ldb_ as i32,
                ap as *const _,
                bsys::cudaDataType_t::CUDA_R_16BF,
                lda_ as i32,
                (&beta) as *const f32 as *const _,
                cp as *mut _,
                ctype,
                n as i32,
                bsys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bsys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .map_err(|e| candle_core::Error::Msg(format!("mm_bf16 cublas: {e:?}")))?;
    }
    Ok((out, dev, m, n))
}

fn wrap_bf16(s: candle_core::cuda_backend::cudarc::driver::CudaSlice<half::bf16>, dev: candle_core::CudaDevice, shape: &Shape) -> Tensor {
    Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(s, dev)), shape.clone(), BackpropOp::none(), false)
}

/// f32 -> bf16 (round to nearest even; same values as to_dtype(BF16)), about 4x faster than candle's cast kernel.
pub fn to_bf16(x: &Tensor) -> Result<Tensor> {
    if x.dtype() == DType::BF16 {
        return Ok(x.clone());
    }
    let x = x.contiguous()?;
    let (xs, xl) = x.storage_and_layout();
    let Storage::Cuda(xs) = &*xs else {
        return x.to_dtype(DType::BF16);
    };
    let dev = xs.device().clone();
    let xv = view(xs, xl, "to_bf16")?;
    let n = xl.shape().elem_count();
    let out = unsafe { dev.alloc::<half::bf16>(n)? };
    let func = dev.get_or_load_custom_func("cast_bf16", MODULE, PTX)?;
    let ni = n as i32;
    let mut lb = func.builder();
    lb.arg(&xv).arg(&out).arg(&ni);
    unsafe { lb.launch(elems_cfg(n.div_ceil(4))) }.w()?;
    Ok(wrap_bf16(out, dev, xl.shape()))
}

/// (a + b, its bf16 copy) in one pass; a, b f32 of the same shape.
pub fn add_cast_bf16(a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
    let (a, b) = (a.contiguous()?, b.contiguous()?);
    let (as_, al) = a.storage_and_layout();
    let (bs, bl) = b.storage_and_layout();
    let (Storage::Cuda(as_), Storage::Cuda(bs)) = (&*as_, &*bs) else {
        candle_core::bail!("add_cast_bf16: CUDA only");
    };
    let dev = as_.device().clone();
    let (av, bv) = (view(as_, al, "add_cast a")?, view(bs, bl, "add_cast b")?);
    let n = al.shape().elem_count();
    let y = unsafe { dev.alloc::<f32>(n)? };
    let yb = unsafe { dev.alloc::<half::bf16>(n)? };
    let func = dev.get_or_load_custom_func("add_cast_bf16", MODULE, PTX)?;
    let ni = n as i32;
    let mut lb = func.builder();
    lb.arg(&av).arg(&bv).arg(&y).arg(&yb).arg(&ni);
    unsafe { lb.launch(elems_cfg(n)) }.w()?;
    let shape = al.shape().clone();
    let yt = Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(y, dev.clone())), shape.clone(), BackpropOp::none(), false);
    Ok((yt, wrap_bf16(yb, dev, &shape)))
}

// ---------------------------------------------------------------- kernels writing bf16 operands (speed round 7b)

/// A kernel input: f32, bf16 or u32 tensor (contiguous on the GPU).
pub enum In<'a> {
    F(&'a Tensor),
    B(&'a Tensor),
    U(&'a Tensor),
}

/// A scalar kernel argument.
pub enum Sc {
    I(i32),
    F(f32),
}

enum View<'v> {
    F(CudaView<'v, f32>),
    B(CudaView<'v, half::bf16>),
    U(CudaView<'v, u32>),
}

/// Launches `name(inputs..., out_bf16, scalars...)` and returns the bf16 output of `shape`.
fn launch_to_bf16(name: &str, ins: &[In], sc: &[Sc], shape: Shape, cfg: LaunchConfig) -> Result<Tensor> {
    let ts: Vec<Tensor> = ins
        .iter()
        .map(|i| match i {
            In::F(t) | In::B(t) | In::U(t) => t.contiguous(),
        })
        .collect::<Result<_>>()?;
    let guards: Vec<_> = ts.iter().map(|t| t.storage_and_layout()).collect();
    let mut dev = None;
    let mut views = Vec::with_capacity(ins.len());
    for ((g, l), kind) in guards.iter().zip(ins.iter()) {
        let Storage::Cuda(s) = &**g else {
            candle_core::bail!("{name}: CUDA only");
        };
        if dev.is_none() {
            dev = Some(s.device().clone());
        }
        let Some((a, b)) = l.contiguous_offsets() else {
            candle_core::bail!("{name}: contiguous input expected");
        };
        views.push(match kind {
            In::F(_) => View::F(s.as_cuda_slice::<f32>()?.slice(a..b)),
            In::B(_) => View::B(s.as_cuda_slice::<half::bf16>()?.slice(a..b)),
            In::U(_) => View::U(s.as_cuda_slice::<u32>()?.slice(a..b)),
        });
    }
    let dev = dev.expect("at least one input");
    let out = unsafe { dev.alloc::<half::bf16>(shape.elem_count())? };
    let func = dev.get_or_load_custom_func(name, MODULE, PTX)?;
    let mut lb = func.builder();
    for v in &views {
        match v {
            View::F(x) => lb.arg(x),
            View::B(x) => lb.arg(x),
            View::U(x) => lb.arg(x),
        };
    }
    lb.arg(&out);
    for s in sc {
        match s {
            Sc::I(x) => lb.arg(x),
            Sc::F(x) => lb.arg(x),
        };
    }
    unsafe { lb.launch(cfg) }.w()?;
    Ok(wrap_bf16(out, dev, &shape))
}

/// One thread per (batch row, channel), 128 per block: the LIF kernels' launch.
fn lif_cfg(b: usize, d: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (((b * d) as u32).div_ceil(128), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }
}

/// rmsnorm forward with a bf16 result (= to_bf16(rms_fwd_raw(x, g, eps))).
pub fn rms_fwd_b(x: &Tensor, g: &Tensor, eps: f64) -> Result<Tensor> {
    let (n, d) = x.dims2()?;
    let sc = [Sc::I(n as i32), Sc::I(d as i32), Sc::F(eps as f32)];
    launch_to_bf16("rms_fwd_b", &[In::F(x), In::F(g)], &sc, x.shape().clone(), rows_cfg(n, row_threads(d), 0))
}

/// LIF spikes as bf16 (exactly 0 or 1), [B, T, D].
pub fn lif_fwd_b(cur: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let (b, t, d) = cur.dims3()?;
    let theta = crate::model::LIF_THRESHOLD as f32;
    let sc = [Sc::I(b as i32), Sc::I(t as i32), Sc::I(d as i32), Sc::F(theta)];
    launch_to_bf16("lif_fwd_b", &[In::F(cur), In::F(beta)], &sc, cur.shape().clone(), lif_cfg(b, d))
}

/// kWTA + relu with a bf16 result (= to_bf16(kwta_relu_raw(u, k))).
pub fn kwta_relu_b(u: &Tensor, k: usize) -> Result<Tensor> {
    let (n, h) = u.dims2()?;
    if h > 12288 {
        candle_core::bail!("kwta-relu: rows of {h} exceed the shared-memory limit");
    }
    let sc = [Sc::I(n as i32), Sc::I(h as i32), Sc::I(k.min(h) as i32)];
    launch_to_bf16("kwta_relu_rows_b", &[In::F(u)], &sc, u.shape().clone(), rows_cfg(n, 256, (h * 4) as u32))
}

/// The kwta-relu gradient from the bf16 activation, as bf16: ga where a > 0, else 0.
pub fn relu_mask_bwd_b(a: &Tensor, ga: &Tensor) -> Result<Tensor> {
    let n = a.elem_count();
    launch_to_bf16("relu_mask_bwd_b", &[In::B(a), In::F(ga)], &[Sc::I(n as i32)], a.shape().clone(), elems_cfg(n))
}

/// dL/dlogits of the fused cross-entropy as bf16 (= to_bf16(ce_bwd_raw(..))).
pub fn ce_bwd_b(logits: &Tensor, targets: &Tensor, aux: &Tensor) -> Result<Tensor> {
    let (n, v) = logits.dims2()?;
    let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let ins = [In::F(logits), In::U(targets), In::F(aux)];
    launch_to_bf16("ce_bwd_b", &ins, &[Sc::I(n as i32), Sc::I(v as i32)], logits.shape().clone(), cfg)
}

/// GELU with a bf16 result.
pub fn gelu_b(u: &Tensor) -> Result<Tensor> {
    let n = u.elem_count();
    launch_to_bf16("gelu_fwd_b", &[In::F(u)], &[Sc::I(n as i32)], u.shape().clone(), elems_cfg(n))
}

/// gy * gelu'(x) as bf16.
pub fn gelu_bwd_b(x: &Tensor, gy: &Tensor) -> Result<Tensor> {
    let n = x.elem_count();
    launch_to_bf16("gelu_bwd_b", &[In::F(x), In::F(gy)], &[Sc::I(n as i32)], x.shape().clone(), elems_cfg(n))
}

/// (dL/dcur as bf16 [B, T, D], dL/dbeta [D]) of the fused LIF scan (= lif_bwd_raw with dcur rounded to bf16).
pub fn lif_bwd_b(cur: &Tensor, beta: &Tensor, grad_s: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, t, d) = cur.dims3()?;
    let (cur, beta, gs) = (cur.contiguous()?, beta.contiguous()?, grad_s.contiguous()?);
    let (cs, cl) = cur.storage_and_layout();
    let (bs, bl) = beta.storage_and_layout();
    let (gs_, gl) = gs.storage_and_layout();
    let (Storage::Cuda(cs), Storage::Cuda(bs), Storage::Cuda(gs_)) = (&*cs, &*bs, &*gs_) else {
        candle_core::bail!("lif_bwd_b: CUDA only");
    };
    let dev = cs.device().clone();
    let (cv, bv, gv) = (view(cs, cl, "lif_bwd_b cur")?, view(bs, bl, "lif_bwd_b beta")?, view(gs_, gl, "lif_bwd_b gs")?);
    let scratch = unsafe { dev.alloc::<f32>(b * (t + 1) * d)? };
    let gb = unsafe { dev.alloc::<half::bf16>(b * t * d)? };
    let func = dev.get_or_load_custom_func("lif_bwd_b", MODULE, PTX)?;
    let (bi, ti, di) = (b as i32, t as i32, d as i32);
    let (theta, slope) = (crate::model::LIF_THRESHOLD as f32, crate::model::LIF_SLOPE as f32);
    let mut lb = func.builder();
    lb.arg(&cv).arg(&bv).arg(&gv).arg(&scratch).arg(&gb).arg(&bi).arg(&ti).arg(&di).arg(&theta).arg(&slope);
    unsafe { lb.launch(lif_cfg(b, d)) }.w()?;
    let scratch = Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(scratch, dev.clone())), (b, t + 1, d), BackpropOp::none(), false);
    let dbeta = scratch.narrow(1, t, 1)?.reshape((b, d))?.sum(0)?;
    Ok((wrap_bf16(gb, dev, &Shape::from((b, t, d))), dbeta))
}

/// Causal attention read in place from the fused projection qkv [B*T, 3D] (heads of 32): returns (o f32 [B*T, D],
/// o bf16, lse [B*H, T]). Same arithmetic as causal_attention.
pub fn attn_fwd_s(qkv: &Tensor, b: usize, t: usize, scale: f64) -> Result<(Tensor, Tensor, Tensor)> {
    let d = qkv.dim(1)? / 3;
    let h = d / ATTN_HS;
    let qkv = qkv.contiguous()?;
    let (qs, ql) = qkv.storage_and_layout();
    let Storage::Cuda(qs) = &*qs else {
        candle_core::bail!("attn_fwd_s: CUDA only");
    };
    let dev = qs.device().clone();
    let qv = view(qs, ql, "attn_fwd_s qkv")?;
    let o = unsafe { dev.alloc::<f32>(b * t * d)? };
    let ob = unsafe { dev.alloc::<half::bf16>(b * t * d)? };
    let lse = unsafe { dev.alloc::<f32>(b * h * t)? };
    let func = dev.get_or_load_custom_func("attn_fwd_s", MODULE, PTX)?;
    let (hi, ti, di, sc) = (h as i32, t as i32, d as i32, scale as f32);
    let mut lb = func.builder();
    lb.arg(&qv).arg(&o).arg(&ob).arg(&lse).arg(&hi).arg(&ti).arg(&di).arg(&sc);
    unsafe { lb.launch(attn_cfg(b * h, t)) }.w()?;
    let ot = Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(o, dev.clone())), (b * t, d), BackpropOp::none(), false);
    let lt = Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(lse, dev.clone())), (b * h, t), BackpropOp::none(), false);
    Ok((ot, wrap_bf16(ob, dev, &Shape::from((b * t, d))), lt))
}

/// The gradient of the fused projection, dqkv [B*T, 3D] as bf16, for the attention-output gradient dout [B*T, D].
pub fn attn_bwd_s(qkv: &Tensor, o: &Tensor, lse: &Tensor, dout: &Tensor, b: usize, t: usize, scale: f64) -> Result<Tensor> {
    let d = qkv.dim(1)? / 3;
    let h = d / ATTN_HS;
    let (qkv, o, lse, dout) = (qkv.contiguous()?, o.contiguous()?, lse.contiguous()?, dout.contiguous()?);
    let (qs, ql) = qkv.storage_and_layout();
    let (os, ol) = o.storage_and_layout();
    let (ls, ll) = lse.storage_and_layout();
    let (ds, dl) = dout.storage_and_layout();
    let (Storage::Cuda(qs), Storage::Cuda(os), Storage::Cuda(ls), Storage::Cuda(ds)) = (&*qs, &*os, &*ls, &*ds) else {
        candle_core::bail!("attn_bwd_s: CUDA only");
    };
    let dev = qs.device().clone();
    let (qv, ov) = (view(qs, ql, "attn_bwd_s qkv")?, view(os, ol, "attn_bwd_s o")?);
    let (lv, dv) = (view(ls, ll, "attn_bwd_s lse")?, view(ds, dl, "attn_bwd_s dout")?);
    let dqkv = unsafe { dev.alloc::<half::bf16>(b * t * 3 * d)? };
    let dvec = unsafe { dev.alloc::<f32>(b * h * t)? };
    let (hi, ti, di, sc) = (h as i32, t as i32, d as i32, scale as f32);
    let fq = dev.get_or_load_custom_func("attn_bwd_q_s", MODULE, PTX)?;
    let mut lb = fq.builder();
    lb.arg(&qv).arg(&ov).arg(&lv).arg(&dv).arg(&dqkv).arg(&dvec).arg(&hi).arg(&ti).arg(&di).arg(&sc);
    unsafe { lb.launch(attn_cfg(b * h, t)) }.w()?;
    let fkv = dev.get_or_load_custom_func("attn_bwd_kv_s", MODULE, PTX)?;
    let mut lb = fkv.builder();
    lb.arg(&qv).arg(&lv).arg(&dv).arg(&dvec).arg(&dqkv).arg(&hi).arg(&ti).arg(&di).arg(&sc);
    unsafe { lb.launch(attn_cfg(b * h, t)) }.w()?;
    Ok(wrap_bf16(dqkv, dev, &Shape::from((b * t, 3 * d))))
}

// ---------------------------------------------------------------- tensor-core attention (speed round 9)

/// A contiguous bf16 input starting on a 16-byte boundary (the tensor-core kernels read 8 values at a time).
fn bview<'a>(s: &'a CudaStorage, l: &Layout, what: &str) -> Result<CudaView<'a, half::bf16>> {
    let v = s.as_cuda_slice::<half::bf16>()?;
    match l.contiguous_offsets() {
        Some((a, b)) if a % 8 == 0 => Ok(v.slice(a..b)),
        _ => candle_core::bail!("{what}: contiguous bf16 input on a 16-byte boundary expected"),
    }
}

fn tc_cfg(bh: usize, t: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (t.div_ceil(64) as u32, bh as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }
}

/// Causal attention on the tensor cores, read in place from the fused bf16 projection qkv [B*T, 3D] (heads of 32):
/// returns (out bf16 [B*T, D], lse [B*H, T]). bf16 inputs and probabilities, f32 accumulation (FlashAttention-2 style).
pub fn attn_fwd_tc(qkv: &Tensor, b: usize, t: usize, scale: f64) -> Result<(Tensor, Tensor)> {
    let d = qkv.dim(1)? / 3;
    let h = d / ATTN_HS;
    if d % ATTN_HS != 0 || qkv.dim(0)? != b * t {
        candle_core::bail!("attn_fwd_tc: qkv {:?} does not fit b {b}, t {t}", qkv.dims());
    }
    let qkv = qkv.contiguous()?;
    let (qs, ql) = qkv.storage_and_layout();
    let Storage::Cuda(qs) = &*qs else {
        candle_core::bail!("attn_fwd_tc: CUDA only");
    };
    let dev = qs.device().clone();
    let qv = bview(qs, ql, "attn_fwd_tc qkv")?;
    let ob = unsafe { dev.alloc::<half::bf16>(b * t * d)? };
    let lse = unsafe { dev.alloc::<f32>(b * h * t)? };
    let func = dev.get_or_load_custom_func("attn_fwd_tc", MODULE, PTX)?;
    let (hi, ti, di, sc) = (h as i32, t as i32, d as i32, scale as f32);
    let mut lb = func.builder();
    lb.arg(&qv).arg(&ob).arg(&lse).arg(&hi).arg(&ti).arg(&di).arg(&sc);
    unsafe { lb.launch(tc_cfg(b * h, t)) }.w()?;
    let lt = Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(lse, dev.clone())), (b * h, t), BackpropOp::none(), false);
    Ok((wrap_bf16(ob, dev, &Shape::from((b * t, d))), lt))
}

/// The gradient of the fused projection, dqkv [B*T, 3D] (bf16), for dout = dL/d(out) [B*T, D] (bf16), given the
/// forward's qkv, out and lse (attn_fwd_tc).
pub fn attn_bwd_tc(qkv: &Tensor, out: &Tensor, lse: &Tensor, dout: &Tensor, b: usize, t: usize, scale: f64) -> Result<Tensor> {
    let d = qkv.dim(1)? / 3;
    let h = d / ATTN_HS;
    let (qkv, out, lse, dout) = (qkv.contiguous()?, out.contiguous()?, lse.contiguous()?, dout.contiguous()?);
    let (qs, ql) = qkv.storage_and_layout();
    let (os, ol) = out.storage_and_layout();
    let (ls, ll) = lse.storage_and_layout();
    let (ds, dl) = dout.storage_and_layout();
    let (Storage::Cuda(qs), Storage::Cuda(os), Storage::Cuda(ls), Storage::Cuda(ds)) = (&*qs, &*os, &*ls, &*ds) else {
        candle_core::bail!("attn_bwd_tc: CUDA only");
    };
    let dev = qs.device().clone();
    let (qv, ov) = (bview(qs, ql, "attn_bwd_tc qkv")?, bview(os, ol, "attn_bwd_tc out")?);
    let (lv, dv) = (view(ls, ll, "attn_bwd_tc lse")?, bview(ds, dl, "attn_bwd_tc dout")?);
    let dqkv = unsafe { dev.alloc::<half::bf16>(b * t * 3 * d)? };
    let dvec = unsafe { dev.alloc::<f32>(b * h * t)? };
    let (hi, ti, di, sc) = (h as i32, t as i32, d as i32, scale as f32);
    let fq = dev.get_or_load_custom_func("attn_bwd_q_tc", MODULE, PTX)?;
    let mut lb = fq.builder();
    lb.arg(&qv).arg(&ov).arg(&lv).arg(&dv).arg(&dqkv).arg(&dvec).arg(&hi).arg(&ti).arg(&di).arg(&sc);
    unsafe { lb.launch(tc_cfg(b * h, t)) }.w()?;
    let fkv = dev.get_or_load_custom_func("attn_bwd_kv_tc", MODULE, PTX)?;
    let mut lb = fkv.builder();
    lb.arg(&qv).arg(&lv).arg(&dv).arg(&dvec).arg(&dqkv).arg(&hi).arg(&ti).arg(&di).arg(&sc);
    unsafe { lb.launch(tc_cfg(b * h, t)) }.w()?;
    Ok(wrap_bf16(dqkv, dev, &Shape::from((b * t, 3 * d))))
}

// ---------------------------------------------------------------- R's fast-weight read (speed round 10)

/// R's (and G's) fast-weight read in one op: y_i = sum_{j<=i} scale (q_i . k_j) exp(c_i - c_j) v_j, for q, k
/// [B, H, T, 32] (k already times the write strength) and vc [B, H, T, 33] = v then c (the running sum of log alpha).
/// The same values as the [T, T] chain (scores * decay matrix, then times v) up to float rounding, without storing
/// any [T, T] tensor; autograd through gla_bwd (dq, dk, and dv with dc in vc's last column).
pub fn gla(q: &Tensor, k: &Tensor, vc: &Tensor, scale: f64) -> Result<Tensor> {
    q.contiguous()?.apply_op3(&k.contiguous()?, &vc.contiguous()?, Gla { scale: scale as f32 })
}

struct Gla {
    scale: f32,
}

impl CustomOp3 for Gla {
    fn name(&self) -> &'static str {
        "ava-gated-linear-read"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu("gla")
    }

    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
        s3: &CudaStorage,
        l3: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (bh, t) = attn_dims(l1)?;
        if l3.shape().dims().last() != Some(&(ATTN_HS + 1)) {
            candle_core::bail!("gla: vc must be [B, H, T, {}]", ATTN_HS + 1);
        }
        let dev = s1.device().clone();
        let (q, k, vc) = (view(s1, l1, "gla q")?, view(s2, l2, "gla k")?, view(s3, l3, "gla vc")?);
        let y = unsafe { dev.alloc::<f32>(bh * t * ATTN_HS)? };
        let func = dev.get_or_load_custom_func("gla_fwd", MODULE, PTX)?;
        let (bhi, ti) = (bh as i32, t as i32);
        let mut lb = func.builder();
        lb.arg(&q).arg(&k).arg(&vc).arg(&y).arg(&bhi).arg(&ti).arg(&self.scale);
        unsafe { lb.launch(attn_cfg(bh, t)) }.w()?;
        Ok((CudaStorage::wrap_cuda_slice(y, dev), l1.shape().clone()))
    }

    fn bwd(&self, q: &Tensor, k: &Tensor, vc: &Tensor, _y: &Tensor, gy: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let (dq, dk, dvc) = gla_bwd(q, k, vc, gy, self.scale)?;
        Ok((Some(dq), Some(dk), Some(dvc)))
    }
}

/// (dq, dk, dvc) of gla for the output gradient dy [B, H, T, 32]: dvc = (dv, dc) [B, H, T, 33].
pub fn gla_bwd(q: &Tensor, k: &Tensor, vc: &Tensor, dy: &Tensor, scale: f32) -> Result<(Tensor, Tensor, Tensor)> {
    let (q, k, vc, dy) = (q.contiguous()?, k.contiguous()?, vc.contiguous()?, dy.contiguous()?);
    let (shape, shape_c) = (q.shape().clone(), vc.shape().clone());
    let (bh, t) = attn_dims(q.layout())?;
    let n = bh * t * ATTN_HS;
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (cs, cl) = vc.storage_and_layout();
    let (ds, dl) = dy.storage_and_layout();
    let (Storage::Cuda(qs), Storage::Cuda(ks), Storage::Cuda(cs), Storage::Cuda(ds)) = (&*qs, &*ks, &*cs, &*ds) else {
        candle_core::bail!("gla backward: CUDA only");
    };
    let dev = qs.device().clone();
    let (qv, kv, cv, dyv) = (view(qs, ql, "gla q")?, view(ks, kl, "gla k")?, view(cs, cl, "gla vc")?, view(ds, dl, "gla dy")?);
    let dq = unsafe { dev.alloc::<f32>(n)? };
    let dk = unsafe { dev.alloc::<f32>(n)? };
    let dvc = unsafe { dev.alloc::<f32>(bh * t * (ATTN_HS + 1))? };
    let rowp = unsafe { dev.alloc::<f32>(bh * t)? };
    let (bhi, ti) = (bh as i32, t as i32);
    let fq = dev.get_or_load_custom_func("gla_bwd_q", MODULE, PTX)?;
    let mut lb = fq.builder();
    lb.arg(&qv).arg(&kv).arg(&cv).arg(&dyv).arg(&dq).arg(&rowp).arg(&bhi).arg(&ti).arg(&scale);
    unsafe { lb.launch(attn_cfg(bh, t)) }.w()?;
    let fkv = dev.get_or_load_custom_func("gla_bwd_kv", MODULE, PTX)?;
    let mut lb = fkv.builder();
    lb.arg(&qv).arg(&kv).arg(&cv).arg(&dyv).arg(&rowp).arg(&dk).arg(&dvc).arg(&bhi).arg(&ti).arg(&scale);
    unsafe { lb.launch(attn_cfg(bh, t)) }.w()?;
    let wrap = |s, sh: &Shape| Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(s, dev.clone())), sh.clone(), BackpropOp::none(), false);
    Ok((wrap(dq, &shape), wrap(dk, &shape), wrap(dvc, &shape_c)))
}

/// D's gated delta rule in one op (speed round 11): per head, F <- alpha (I - beta k k^T) F + beta k v^T and
/// y_t = q_t F_t, run as a direct recurrence (one warp per head, F in registers), for q, k [B, H, T, 32] (unit length)
/// and vg [B, H, T, 34] = v, then log alpha, then beta. The same values as the chunked WY form (model.rs delta_rule)
/// up to float rounding; autograd through delta_bwd (dq, dk, and dv, d log alpha, d beta in vg's layout).
pub fn delta(q: &Tensor, k: &Tensor, vg: &Tensor) -> Result<Tensor> {
    q.contiguous()?.apply_op3(&k.contiguous()?, &vg.contiguous()?, Delta)
}

const DELTA_VG: usize = ATTN_HS + 2;

fn delta_cfg(bh: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (bh.div_ceil(4) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }
}

struct Delta;

impl CustomOp3 for Delta {
    fn name(&self) -> &'static str {
        "ava-delta-rule"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu("delta")
    }

    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
        s3: &CudaStorage,
        l3: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (bh, t) = attn_dims(l1)?;
        if l3.shape().dims().last() != Some(&DELTA_VG) {
            candle_core::bail!("delta: vg must be [B, H, T, {DELTA_VG}]");
        }
        let dev = s1.device().clone();
        let (q, k, vg) = (view(s1, l1, "delta q")?, view(s2, l2, "delta k")?, view(s3, l3, "delta vg")?);
        let y = unsafe { dev.alloc::<f32>(bh * t * ATTN_HS)? };
        let func = dev.get_or_load_custom_func("delta_fwd", MODULE, PTX)?;
        let (bhi, ti) = (bh as i32, t as i32);
        let mut lb = func.builder();
        lb.arg(&q).arg(&k).arg(&vg).arg(&y).arg(&bhi).arg(&ti);
        unsafe { lb.launch(delta_cfg(bh)) }.w()?;
        Ok((CudaStorage::wrap_cuda_slice(y, dev), l1.shape().clone()))
    }

    fn bwd(&self, q: &Tensor, k: &Tensor, vg: &Tensor, _y: &Tensor, gy: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let (dq, dk, dvg) = delta_bwd(q, k, vg, gy)?;
        Ok((Some(dq), Some(dk), Some(dvg)))
    }
}

/// (dq, dk, dvg) of delta for the output gradient dy [B, H, T, 32]. Recomputes the state before every step
/// ([B*H, T, 32, 32] f32, freed on return), then walks back through time.
pub fn delta_bwd(q: &Tensor, k: &Tensor, vg: &Tensor, dy: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let (q, k, vg, dy) = (q.contiguous()?, k.contiguous()?, vg.contiguous()?, dy.contiguous()?);
    let (shape, shape_g) = (q.shape().clone(), vg.shape().clone());
    let (bh, t) = attn_dims(q.layout())?;
    let n = bh * t * ATTN_HS;
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (gs, gl) = vg.storage_and_layout();
    let (ds, dl) = dy.storage_and_layout();
    let (Storage::Cuda(qs), Storage::Cuda(ks), Storage::Cuda(gs), Storage::Cuda(ds)) = (&*qs, &*ks, &*gs, &*ds) else {
        candle_core::bail!("delta backward: CUDA only");
    };
    let dev = qs.device().clone();
    let (qv, kv, gv, dyv) = (view(qs, ql, "delta q")?, view(ks, kl, "delta k")?, view(gs, gl, "delta vg")?, view(ds, dl, "delta dy")?);
    let states = unsafe { dev.alloc::<f32>(bh * t * ATTN_HS * ATTN_HS)? };
    let dq = unsafe { dev.alloc::<f32>(n)? };
    let dk = unsafe { dev.alloc::<f32>(n)? };
    let dvg = unsafe { dev.alloc::<f32>(bh * t * DELTA_VG)? };
    let (bhi, ti) = (bh as i32, t as i32);
    let fs = dev.get_or_load_custom_func("delta_states", MODULE, PTX)?;
    let mut lb = fs.builder();
    lb.arg(&qv).arg(&kv).arg(&gv).arg(&states).arg(&bhi).arg(&ti);
    unsafe { lb.launch(delta_cfg(bh)) }.w()?;
    let fb = dev.get_or_load_custom_func("delta_bwd", MODULE, PTX)?;
    let mut lb = fb.builder();
    lb.arg(&qv).arg(&kv).arg(&gv).arg(&states).arg(&dyv).arg(&dq).arg(&dk).arg(&dvg).arg(&bhi).arg(&ti);
    unsafe { lb.launch(delta_cfg(bh)) }.w()?;
    let wrap = |s, sh: &Shape| Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(s, dev.clone())), sh.clone(), BackpropOp::none(), false);
    Ok((wrap(dq, &shape), wrap(dk, &shape), wrap(dvg, &shape_g)))
}

/// Steps between the saved fast-weight states of hebb_fwd_raw (HB_C in lif.cu).
pub const HEBB_CK: usize = 16;

/// Speed round 12: D's fast-weight block after its input projection, in one kernel. `p` [N, 4d + 2H] f32 = h times
/// [Wq | Wk | Wv | Wg | Wa | Wb] (rows n = b·T + t), `cst` [3H] = (ba, bb, logit γ). Returns (O [N, d] f32, ck): per
/// head the gated delta rule over unit-length q, k (α = sigmoid(za + ba + off), β = 2·sigmoid(zb + bb)), the read-out
/// head-normed and times silu(g); ck holds the state every HEBB_CK steps for hebb_bwd_raw. The same maths as model.rs
/// hebb() for D, without autograd.
pub fn hebb_fwd_per_token(p: &Tensor, cst: &Tensor, b: usize, t: usize, nh: usize, eps: f64) -> Result<(Tensor, Tensor)> {
    hebb_fwd_pt(p, cst, b, t, nh, eps, false)
}

/// v40: hebb_fwd_per_token with O as f32, or as bf16 if `bf` (hebb_fwd_b: f2bf of the same values).
fn hebb_fwd_pt(p: &Tensor, cst: &Tensor, b: usize, t: usize, nh: usize, eps: f64, bf: bool) -> Result<(Tensor, Tensor)> {
    let (p, cst) = (p.contiguous()?, cst.contiguous()?);
    let d = nh * ATTN_HS;
    let ld = 4 * d + 2 * nh;
    if p.dims() != [b * t, ld] || cst.dims() != [3 * nh] {
        candle_core::bail!("hebb_fwd: want p [{}, {ld}] and cst [{}], got {:?} {:?}", b * t, 3 * nh, p.dims(), cst.dims());
    }
    let (ps, pl) = p.storage_and_layout();
    let (cs, cl) = cst.storage_and_layout();
    let (Storage::Cuda(ps), Storage::Cuda(cs)) = (&*ps, &*cs) else {
        candle_core::bail!("hebb_fwd: CUDA only");
    };
    let dev = ps.device().clone();
    let (pv, cv) = (view(ps, pl, "hebb p")?, view(cs, cl, "hebb cst")?);
    let nck = t.div_ceil(HEBB_CK);
    let o = OutBuf::alloc(&dev, b * t * d, bf)?;
    let ck = unsafe { dev.alloc::<f32>(b * nh * nck * ATTN_HS * ATTN_HS)? };
    let func = dev.get_or_load_custom_func(if bf { "hebb_fwd_b" } else { "hebb_fwd" }, MODULE, PTX)?;
    let (bi, hi, ti, ldi, e) = (b as i32, nh as i32, t as i32, ld as i32, eps as f32);
    let mut lb = func.builder();
    lb.arg(&pv).arg(&cv);
    o.push(&mut lb);
    lb.arg(&ck).arg(&bi).arg(&hi).arg(&ti).arg(&ldi).arg(&e);
    unsafe { lb.launch(delta_cfg(b * nh)) }.w()?;
    let ot = o.tensor(dev.clone(), (b * t, d));
    let ckt = Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(ck, dev)), (b * nh * nck, ATTN_HS * ATTN_HS), BackpropOp::none(), false);
    Ok((ot, ckt))
}

/// dP [N, 4d + 2H] of hebb_fwd_raw for dO [N, d], from its saved states `ck`: each chunk of HEBB_CK steps is
/// recomputed from its saved state into a per-head scratch ([B·H, HEBB_CK, 32, 32] f32, freed on return), then walked
/// back through. Gradients of ba and bb are the column sums of dP's za and zb parts.
pub fn hebb_bwd_per_token(p: &Tensor, cst: &Tensor, ck: &Tensor, d_o: &Tensor, b: usize, t: usize, nh: usize, eps: f64) -> Result<Tensor> {
    hebb_bwd_per_token_k(p, cst, ck, d_o, b, t, nh, eps, hebb_bwd_w1_on())
}

/// v40: hebb_bwd_per_token with the kernel chosen: hebb_bwd_w1 (one head per block) if `w1`, else hebb_bwd.
#[allow(clippy::too_many_arguments)]
pub fn hebb_bwd_per_token_k(p: &Tensor, cst: &Tensor, ck: &Tensor, d_o: &Tensor, b: usize, t: usize, nh: usize, eps: f64, w1: bool) -> Result<Tensor> {
    Ok(hebb_bwd_pt(p, cst, ck, d_o, b, t, nh, eps, w1, false)?.0)
}

/// v40: hebb_bwd_per_token_k with dP as f32 (dz None), or if `bf` as bf16 (f2bf of the same values) with dza, dzb
/// in f32 apart (dz [N, 2H]).
#[allow(clippy::too_many_arguments)]
fn hebb_bwd_pt(p: &Tensor, cst: &Tensor, ck: &Tensor, d_o: &Tensor, b: usize, t: usize, nh: usize, eps: f64, w1: bool, bf: bool) -> Result<(Tensor, Option<Tensor>)> {
    let (p, cst, ck, d_o) = (p.contiguous()?, cst.contiguous()?, ck.contiguous()?, d_o.contiguous()?);
    let d = nh * ATTN_HS;
    let ld = 4 * d + 2 * nh;
    if p.dims() != [b * t, ld] || d_o.dims() != [b * t, d] || ck.dims() != [b * nh * t.div_ceil(HEBB_CK), ATTN_HS * ATTN_HS] {
        candle_core::bail!("hebb_bwd: want p [{}, {ld}], dO [{}, {d}] and ck from hebb_fwd_raw, got {:?} {:?} {:?}", b * t, b * t, p.dims(), d_o.dims(), ck.dims());
    }
    let (ps, pl) = p.storage_and_layout();
    let (cs, cl) = cst.storage_and_layout();
    let (ks, kl) = ck.storage_and_layout();
    let (ds, dl) = d_o.storage_and_layout();
    let (Storage::Cuda(ps), Storage::Cuda(cs), Storage::Cuda(ks), Storage::Cuda(ds)) = (&*ps, &*cs, &*ks, &*ds) else {
        candle_core::bail!("hebb_bwd: CUDA only");
    };
    let dev = ps.device().clone();
    let (pv, cv, kv, dv) = (view(ps, pl, "hebb p")?, view(cs, cl, "hebb cst")?, view(ks, kl, "hebb ck")?, view(ds, dl, "hebb dO")?);
    let scratch = unsafe { dev.alloc::<f32>(b * nh * HEBB_CK * ATTN_HS * ATTN_HS)? };
    let dp = OutBuf::alloc(&dev, b * t * ld, bf)?;
    let dz = if bf { Some(unsafe { dev.alloc::<f32>(b * t * 2 * nh)? }) } else { None };
    let (bi, hi, ti, ldi, e) = (b as i32, nh as i32, t as i32, ld as i32, eps as f32);
    // v40: one head per block (hebb_bwd_w1, the same code per head): no second wave of 4 heads at b·nh = 384
    let name = match (w1, bf) {
        (true, false) => "hebb_bwd_w1",
        (false, false) => "hebb_bwd",
        (true, true) => "hebb_bwd_w1_bf",
        (false, true) => "hebb_bwd_bf",
    };
    let fb = dev.get_or_load_custom_func(name, MODULE, PTX)?;
    let mut lb = fb.builder();
    lb.arg(&pv).arg(&cv).arg(&kv).arg(&scratch).arg(&dv);
    dp.push(&mut lb);
    if let Some(z) = &dz {
        lb.arg(z);
    }
    lb.arg(&bi).arg(&hi).arg(&ti).arg(&ldi).arg(&e);
    unsafe { lb.launch(if w1 { one_warp(b * nh) } else { delta_cfg(b * nh) }) }.w()?;
    let dzt = dz.map(|z| Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(z, dev.clone())), (b * t, 2 * nh), BackpropOp::none(), false));
    Ok((dp.tensor(dev, (b * t, ld)), dzt))
}

/// v40: hebb_bwd_per_token launches hebb_bwd_w1 (one head per block of 32 threads) unless AVA_HEBB_W1=0 (read once).
pub fn hebb_bwd_w1_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("AVA_HEBB_W1").is_ok_and(|v| v == "0"))
}

/// D's fast-weight block forward (see hebb_fwd_per_token): the chunk form or the per-token kernel, by hebb_use_chunk.
pub fn hebb_fwd_raw(p: &Tensor, cst: &Tensor, b: usize, t: usize, nh: usize, eps: f64) -> Result<(Tensor, Tensor)> {
    if hebb_use_chunk(p.device(), b, nh)? {
        hebc_fwd_raw(p, cst, b, t, nh, eps)
    } else {
        hebb_fwd_per_token(p, cst, b, t, nh, eps)
    }
}

/// D's fast-weight block backward (see hebb_bwd_per_token), chosen as hebb_fwd_raw is.
pub fn hebb_bwd_raw(p: &Tensor, cst: &Tensor, ck: &Tensor, d_o: &Tensor, b: usize, t: usize, nh: usize, eps: f64) -> Result<Tensor> {
    if hebb_use_chunk(p.device(), b, nh)? {
        hebc_bwd_raw(p, cst, ck, d_o, b, t, nh, eps)
    } else {
        hebb_bwd_per_token(p, cst, ck, d_o, b, t, nh, eps)
    }
}

/// v40: hebb_fwd_raw with O written as bf16 (the matmul operand of the bf16 training step): f2bf of the values
/// hebb_fwd_raw gives, without the f32 O and its cast. Chunk form or per-token kernel as hebb_fwd_raw.
pub fn hebb_fwd_raw_b(p: &Tensor, cst: &Tensor, b: usize, t: usize, nh: usize, eps: f64) -> Result<(Tensor, Tensor)> {
    if hebb_use_chunk(p.device(), b, nh)? {
        hebc_fwd(p, cst, b, t, nh, eps, true)
    } else {
        hebb_fwd_pt(p, cst, b, t, nh, eps, true)
    }
}

/// v40: hebb_bwd_raw with dP written as bf16 (the matmul operand; f2bf of the values hebb_bwd_raw gives) and its
/// za, zb columns also as f32, dz [N, 2H] (dz[:, ..H] = dP[:, 4d..4d+H], dz[:, H..] = dP[:, 4d+H..]), for the bias
/// gradients. Returns (dP bf16, dz).
pub fn hebb_bwd_raw_b(p: &Tensor, cst: &Tensor, ck: &Tensor, d_o: &Tensor, b: usize, t: usize, nh: usize, eps: f64) -> Result<(Tensor, Tensor)> {
    let (dp, dz) = if hebb_use_chunk(p.device(), b, nh)? {
        hebc_bwd(p, cst, ck, d_o, b, t, nh, eps, true)?
    } else {
        hebb_bwd_pt(p, cst, ck, d_o, b, t, nh, eps, hebb_bwd_w1_on(), true)?
    };
    Ok((dp, dz.expect("bf16 dP comes with dz")))
}

/// v40: the per-token kernel of hebb_bwd_raw_b, with the kernel chosen as in hebb_bwd_per_token_k (for tests).
#[allow(clippy::too_many_arguments)]
pub fn hebb_bwd_per_token_b(p: &Tensor, cst: &Tensor, ck: &Tensor, d_o: &Tensor, b: usize, t: usize, nh: usize, eps: f64, w1: bool) -> Result<(Tensor, Tensor)> {
    let (dp, dz) = hebb_bwd_pt(p, cst, ck, d_o, b, t, nh, eps, w1, true)?;
    Ok((dp, dz.expect("bf16 dP comes with dz")))
}

/// v40: an output buffer a kernel writes as f32 or as bf16.
enum OutBuf {
    F(candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>),
    B(candle_core::cuda_backend::cudarc::driver::CudaSlice<half::bf16>),
}

impl OutBuf {
    fn alloc(dev: &CudaDevice, n: usize, bf: bool) -> Result<OutBuf> {
        Ok(if bf { OutBuf::B(unsafe { dev.alloc::<half::bf16>(n)? }) } else { OutBuf::F(unsafe { dev.alloc::<f32>(n)? }) })
    }

    fn push<'a>(&'a self, lb: &mut candle_core::cuda_backend::cudarc::driver::LaunchArgs<'a>) {
        match self {
            OutBuf::F(s) => {
                lb.arg(s);
            }
            OutBuf::B(s) => {
                lb.arg(s);
            }
        }
    }

    fn tensor<S: Into<Shape>>(self, dev: CudaDevice, shape: S) -> Tensor {
        match self {
            OutBuf::F(s) => Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(s, dev)), shape, BackpropOp::none(), false),
            OutBuf::B(s) => wrap_bf16(s, dev, &shape.into()),
        }
    }
}

/// v38: whether hebb_fwd_raw / hebb_bwd_raw use the chunk form (hebc_*) for b sequences of nh heads.
/// AVA_HEBB_CHUNK=0 keeps the per-token kernels and AVA_HEBB_CHUNK=1 the chunk form (both read once). Unset, the chunk
/// form runs when b·nh < 4 × the GPU's SM count, that is when the per-token kernel (4 heads per block of 128 threads)
/// would leave SMs idle. The choice depends only on the shape and the GPU, so a run is repeatable. Measured on the RTX
/// 4070 (46 SMs, d 384, 8,192 rows, forward + backward): b·nh 384 and 192 per-token 1.0x and 1.2x faster; 96, 48, 24
/// the chunk form 1.2x, 2.2x and 4.1x faster.
pub fn hebb_use_chunk(dev: &candle_core::Device, b: usize, nh: usize) -> Result<bool> {
    static MODE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    static SMS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    if let Some(m) = *MODE.get_or_init(|| std::env::var("AVA_HEBB_CHUNK").ok().map(|v| v != "0")) {
        return Ok(m);
    }
    let candle_core::Device::Cuda(cd) = dev else {
        return Ok(false);
    };
    let sms = match SMS.get() {
        Some(s) => *s,
        None => {
            use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
            let s = cd.cuda_stream().context().attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).w()? as usize;
            *SMS.get_or_init(|| s)
        }
    };
    Ok(b * nh < 4 * sms)
}

/// AVA_HC_PROF=1: synchronise after every chunk-form kernel and print its time since the last one (profiling only).
fn hc_prof(dev: &CudaDevice, name: &str) -> Result<()> {
    use candle_core::backend::BackendDevice;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    thread_local!(static LAST: std::cell::Cell<Option<std::time::Instant>> = const { std::cell::Cell::new(None) });
    if !*ON.get_or_init(|| std::env::var("AVA_HC_PROF").map(|v| v == "1").unwrap_or(false)) {
        return Ok(());
    }
    dev.synchronize()?;
    let now = std::time::Instant::now();
    LAST.with(|l| {
        if let (Some(t), false) = (l.get(), name == "start") {
            eprintln!("  {name} {:.3} ms", (now - t).as_secs_f64() * 1000.0);
        }
        l.set(Some(now));
    });
    Ok(())
}

fn one_warp(blocks: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 }
}

/// The two scans: one block of 128 threads per head.
fn per_head(heads: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (heads as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }
}

/// v38: hebb_fwd_raw in the chunk form of the gated delta rule (lif.cu hc_prep, hc_scan, hc_out): the same inputs and
/// outputs, the same maths to rounding, and ck is the same tensor (the state entering every 16-token chunk).
pub fn hebc_fwd_raw(p: &Tensor, cst: &Tensor, b: usize, t: usize, nh: usize, eps: f64) -> Result<(Tensor, Tensor)> {
    hebc_fwd(p, cst, b, t, nh, eps, false)
}

/// v40: hebc_fwd_raw with O as f32, or as bf16 if `bf` (hc_out_b: f2bf of the same values).
fn hebc_fwd(p: &Tensor, cst: &Tensor, b: usize, t: usize, nh: usize, eps: f64, bf: bool) -> Result<(Tensor, Tensor)> {
    let (p, cst) = (p.contiguous()?, cst.contiguous()?);
    let d = nh * ATTN_HS;
    let ld = 4 * d + 2 * nh;
    if p.dims() != [b * t, ld] || cst.dims() != [3 * nh] {
        candle_core::bail!("hebc_fwd: want p [{}, {ld}] and cst [{}], got {:?} {:?}", b * t, 3 * nh, p.dims(), cst.dims());
    }
    let (ps, pl) = p.storage_and_layout();
    let (cs, cl) = cst.storage_and_layout();
    let (Storage::Cuda(ps), Storage::Cuda(cs)) = (&*ps, &*cs) else {
        candle_core::bail!("hebc_fwd: CUDA only");
    };
    let dev = ps.device().clone();
    let (pv, cv) = (view(ps, pl, "hebc p")?, view(cs, cl, "hebc cst")?);
    let nck = t.div_ceil(HEBB_CK);
    let nchunks = b * nh * nck;
    let tile = nchunks * HEBB_CK * ATTN_HS;
    let (bi, hi, ti, ldi, nci, e) = (b as i32, nh as i32, t as i32, ld as i32, nck as i32, eps as f32);
    let u = unsafe { dev.alloc::<f32>(tile)? };
    let w = unsafe { dev.alloc::<f32>(tile)? };
    let kd = unsafe { dev.alloc::<f32>(tile)? };
    let gc = unsafe { dev.alloc::<f32>(nchunks)? };
    let f0 = dev.get_or_load_custom_func("hc_prep", MODULE, PTX)?;
    let mut lb = f0.builder();
    lb.arg(&pv).arg(&cv).arg(&u).arg(&w).arg(&kd).arg(&gc).arg(&bi).arg(&hi).arg(&ti).arg(&ldi);
    hc_prof(&dev, "start")?;
    unsafe { lb.launch(one_warp(nchunks)) }.w()?;
    hc_prof(&dev, "hc_prep")?;
    let ck = unsafe { dev.alloc::<f32>(nchunks * ATTN_HS * ATTN_HS)? };
    let dl = unsafe { dev.alloc::<f32>(tile)? };
    let fs = dev.get_or_load_custom_func("hc_scan", MODULE, PTX)?;
    let mut lb = fs.builder();
    lb.arg(&u).arg(&w).arg(&kd).arg(&gc).arg(&ck).arg(&dl).arg(&nci);
    unsafe { lb.launch(per_head(b * nh)) }.w()?;
    hc_prof(&dev, "hc_scan")?;
    drop((u, w, kd, gc));
    let o = OutBuf::alloc(&dev, b * t * d, bf)?;
    let fo = dev.get_or_load_custom_func(if bf { "hc_out_b" } else { "hc_out" }, MODULE, PTX)?;
    let mut lb = fo.builder();
    lb.arg(&pv).arg(&cv).arg(&dl).arg(&ck);
    o.push(&mut lb);
    lb.arg(&bi).arg(&hi).arg(&ti).arg(&ldi).arg(&e);
    unsafe { lb.launch(one_warp(nchunks)) }.w()?;
    hc_prof(&dev, "hc_out")?;
    let ot = o.tensor(dev.clone(), (b * t, d));
    let ckt = Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(ck, dev)), (nchunks, ATTN_HS * ATTN_HS), BackpropOp::none(), false);
    Ok((ot, ckt))
}

/// v38: hebb_bwd_raw in the chunk form (lif.cu hc_bout, hc_bscan, hc_bmain). Takes ck from either forward.
pub fn hebc_bwd_raw(p: &Tensor, cst: &Tensor, ck: &Tensor, d_o: &Tensor, b: usize, t: usize, nh: usize, eps: f64) -> Result<Tensor> {
    Ok(hebc_bwd(p, cst, ck, d_o, b, t, nh, eps, false)?.0)
}

/// v40: hebc_bwd_raw with dP as f32 (dz None), or if `bf` as bf16 (hc_bout_b, hc_bmain_b: f2bf of the same values)
/// with dza, dzb in f32 apart (dz [N, 2H]).
#[allow(clippy::too_many_arguments)]
fn hebc_bwd(p: &Tensor, cst: &Tensor, ck: &Tensor, d_o: &Tensor, b: usize, t: usize, nh: usize, eps: f64, bf: bool) -> Result<(Tensor, Option<Tensor>)> {
    let (p, cst, ck, d_o) = (p.contiguous()?, cst.contiguous()?, ck.contiguous()?, d_o.contiguous()?);
    let d = nh * ATTN_HS;
    let ld = 4 * d + 2 * nh;
    let nck = t.div_ceil(HEBB_CK);
    if p.dims() != [b * t, ld] || d_o.dims() != [b * t, d] || ck.dims() != [b * nh * nck, ATTN_HS * ATTN_HS] {
        candle_core::bail!("hebc_bwd: want p [{}, {ld}], dO [{}, {d}] and ck from hebb_fwd_raw, got {:?} {:?} {:?}", b * t, b * t, p.dims(), d_o.dims(), ck.dims());
    }
    let (ps, pl) = p.storage_and_layout();
    let (cs, cl) = cst.storage_and_layout();
    let (ks, kl) = ck.storage_and_layout();
    let (ds, dl) = d_o.storage_and_layout();
    let (Storage::Cuda(ps), Storage::Cuda(cs), Storage::Cuda(ks), Storage::Cuda(ds)) = (&*ps, &*cs, &*ks, &*ds) else {
        candle_core::bail!("hebc_bwd: CUDA only");
    };
    let dev = ps.device().clone();
    let (pv, cv, kv, dv) = (view(ps, pl, "hebc p")?, view(cs, cl, "hebc cst")?, view(ks, kl, "hebc ck")?, view(ds, dl, "hebc dO")?);
    let nchunks = b * nh * nck;
    let tile = nchunks * HEBB_CK * ATTN_HS;
    let (bi, hi, ti, ldi, nci, e) = (b as i32, nh as i32, t as i32, ld as i32, nck as i32, eps as f32);
    let w = unsafe { dev.alloc::<f32>(tile)? };
    let kd = unsafe { dev.alloc::<f32>(tile)? };
    let gc = unsafe { dev.alloc::<f32>(nchunks)? };
    let dy = unsafe { dev.alloc::<f32>(tile)? };
    let x1 = unsafe { dev.alloc::<f32>(tile)? };
    let x2 = unsafe { dev.alloc::<f32>(nchunks * ATTN_HS * ATTN_HS)? };
    let dp = OutBuf::alloc(&dev, b * t * ld, bf)?;
    let dz = if bf { Some(unsafe { dev.alloc::<f32>(b * t * 2 * nh)? }) } else { None };
    let f1 = dev.get_or_load_custom_func(if bf { "hc_bout_b" } else { "hc_bout" }, MODULE, PTX)?;
    let mut lb = f1.builder();
    lb.arg(&pv).arg(&cv).arg(&kv).arg(&dv).arg(&w).arg(&kd).arg(&gc);
    dp.push(&mut lb);
    lb.arg(&dy).arg(&x1).arg(&x2).arg(&bi).arg(&hi).arg(&ti).arg(&ldi).arg(&e);
    hc_prof(&dev, "start")?;
    unsafe { lb.launch(one_warp(nchunks)) }.w()?;
    hc_prof(&dev, "hc_bout")?;
    let dfn = unsafe { dev.alloc::<f32>(nchunks * ATTN_HS * ATTN_HS)? };
    let dd = unsafe { dev.alloc::<f32>(tile)? };
    let f2 = dev.get_or_load_custom_func("hc_bscan", MODULE, PTX)?;
    let mut lb = f2.builder();
    lb.arg(&w).arg(&kd).arg(&gc).arg(&x1).arg(&x2).arg(&dfn).arg(&dd).arg(&nci);
    unsafe { lb.launch(per_head(b * nh)) }.w()?;
    hc_prof(&dev, "hc_bscan")?;
    drop((w, kd, gc, x1, x2));
    let f3 = dev.get_or_load_custom_func(if bf { "hc_bmain_b" } else { "hc_bmain" }, MODULE, PTX)?;
    let mut lb = f3.builder();
    lb.arg(&pv).arg(&cv).arg(&kv).arg(&dfn).arg(&dd).arg(&dy);
    dp.push(&mut lb);
    if let Some(z) = &dz {
        lb.arg(z);
    }
    lb.arg(&bi).arg(&hi).arg(&ti).arg(&ldi);
    unsafe { lb.launch(one_warp(nchunks)) }.w()?;
    hc_prof(&dev, "hc_bmain")?;
    let dzt = dz.map(|z| Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(z, dev.clone())), (b * t, 2 * nh), BackpropOp::none(), false));
    Ok((dp.tensor(dev, (b * t, ld)), dzt))
}

/// v40: raw device pointer of a contiguous f32 tensor's data (the tensor must outlive every use of the pointer).
fn f32_ptr(t: &Tensor, what: &str) -> Result<u64> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    if t.dtype() != DType::F32 {
        candle_core::bail!("{what}: f32 expected, got {:?}", t.dtype());
    }
    let (s, l) = t.storage_and_layout();
    let Storage::Cuda(s) = &*s else {
        candle_core::bail!("{what}: CUDA only");
    };
    let v = view(s, l, what)?;
    let stream = s.device().cuda_stream();
    let (p, _g) = v.device_ptr(&stream);
    Ok(p)
}

/// v40: sqrt(sum over `gs` of sum(g^2)) as an f64 scalar on the device, in three launches: the same value, bit for
/// bit, as AdamW::grad_norm_t's per-tensor candle ops (sqr, sum_keepdim(-1), sum_all, the f32 sums added in order,
/// to f64, sqrt): lif.cu gn_rows / gn_final reproduce candle's reduction grouping exactly. f32 CUDA tensors.
pub fn grad_norm_multi(gs: &[Tensor]) -> Result<Tensor> {
    if gs.is_empty() {
        candle_core::bail!("grad_norm_multi: no tensors");
    }
    let gs: Vec<Tensor> = gs.iter().map(|g| g.contiguous()).collect::<Result<_>>()?;
    let candle_core::Device::Cuda(dev) = gs[0].device().clone() else {
        candle_core::bail!("grad_norm_multi: CUDA only");
    };
    let mut tab: Vec<u64> = Vec::with_capacity(4 * gs.len());
    let mut nrows = 0usize;
    for g in &gs {
        let n = g.elem_count();
        let cols = if g.rank() >= 2 { g.dim(D::Minus1)? } else { n };
        let rows = if cols == 0 { 0 } else { n / cols };
        if rows == 0 {
            candle_core::bail!("grad_norm_multi: empty tensor {:?}", g.dims());
        }
        tab.extend_from_slice(&[f32_ptr(g, "grad_norm_multi")?, rows as u64, cols as u64, nrows as u64]);
        nrows += rows;
    }
    let nt = gs.len();
    let tab_d = dev.clone_htod(&tab)?;
    let rs = unsafe { dev.alloc::<f32>(nrows + nt)? };
    let out = unsafe { dev.alloc::<f64>(1)? };
    let (nti, nri) = (nt as i32, nrows as i32);
    {
        let rv = rs.slice(0..nrows);
        let f = dev.get_or_load_custom_func("gn_rows", MODULE, PTX)?;
        let mut lb = f.builder();
        lb.arg(&tab_d).arg(&nti).arg(&nri).arg(&rv);
        let cfg = LaunchConfig { grid_dim: ((nrows as u32).div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { lb.launch(cfg) }.w()?;
    }
    {
        let (rv, tv) = (rs.slice(0..nrows), rs.slice(nrows..nrows + nt));
        let f = dev.get_or_load_custom_func("gn_final", MODULE, PTX)?;
        let mut lb = f.builder();
        lb.arg(&tab_d).arg(&nti).arg(&rv).arg(&tv).arg(&out);
        unsafe { lb.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 }) }.w()?;
    }
    drop(gs);
    Ok(Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)), (), BackpropOp::none(), false))
}

/// v40: adamw_in_place_s for many parameters in one launch (lif.cu adamw_multi): element for element the same
/// arithmetic, so the same bits. `shrink[i]` is parameter i's decay factor; sc.shrink and sc.grad_scale are unused.
/// Each w, m, v must be a parameter's own contiguous storage (as for adamw_in_place).
pub fn adamw_multi(w: &[Tensor], g: &[Tensor], m: &[Tensor], v: &[Tensor], shrink: &[f64], scale: &Tensor, sc: &AdamWScalars) -> Result<()> {
    let nt = w.len();
    if nt == 0 {
        return Ok(());
    }
    if g.len() != nt || m.len() != nt || v.len() != nt || shrink.len() != nt {
        candle_core::bail!("adamw_multi: lists of different lengths");
    }
    if scale.dtype() != DType::F64 || scale.elem_count() != 1 {
        candle_core::bail!("adamw_multi: scale must be one f64");
    }
    const THREADS: usize = 256;
    let gc: Vec<Tensor> = g.iter().map(|x| x.contiguous()).collect::<Result<_>>()?;
    let mut tab: Vec<u64> = Vec::with_capacity(6 * nt);
    let mut blocks = 0usize;
    for i in 0..nt {
        let n = w[i].elem_count();
        if !(w[i].is_contiguous() && m[i].is_contiguous() && v[i].is_contiguous()) || m[i].elem_count() != n || v[i].elem_count() != n || gc[i].elem_count() != n {
            candle_core::bail!("adamw_multi: w, m, v must be contiguous and of one size");
        }
        let sh = (shrink[i] as f32).to_bits() as u64;
        tab.extend_from_slice(&[
            f32_ptr(&w[i], "adamw_multi w")?,
            f32_ptr(&gc[i], "adamw_multi g")?,
            f32_ptr(&m[i], "adamw_multi m")?,
            f32_ptr(&v[i], "adamw_multi v")?,
            n as u64,
            (blocks as u64) | (sh << 32),
        ]);
        blocks += n.div_ceil(THREADS);
    }
    if blocks >= u32::MAX as usize {
        candle_core::bail!("adamw_multi: too many elements");
    }
    let scale = scale.contiguous()?;
    let (ss, sl) = scale.storage_and_layout();
    let Storage::Cuda(ss) = &*ss else {
        candle_core::bail!("adamw_multi: CUDA only");
    };
    let dev = ss.device().clone();
    let sv = ss.as_cuda_slice::<f64>()?.slice(sl.start_offset()..sl.start_offset() + 1);
    let tab_d = dev.clone_htod(&tab)?;
    let f = dev.get_or_load_custom_func("adamw_multi", MODULE, PTX)?;
    let nti = nt as i32;
    let fl = |x: f64| x as f32;
    let (b1, c1, b2, c2) = (fl(sc.b1), fl(1.0 - sc.b1), fl(sc.b2), fl(1.0 - sc.b2));
    let (eps, step, ibc2) = (fl(sc.eps), fl(sc.step_size), fl(sc.inv_sqrt_bc2));
    let mut lb = f.builder();
    lb.arg(&tab_d).arg(&nti).arg(&sv).arg(&b1).arg(&c1).arg(&b2).arg(&c2).arg(&eps).arg(&step).arg(&ibc2);
    unsafe { lb.launch(LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (THREADS as u32, 1, 1), shared_mem_bytes: 0 }) }.w()?;
    drop(gc);
    Ok(())
}

/// Speed round 13: adamw without the [3, n] result and the three copies back: the kernel overwrites w, m and v where
/// they are (each element is read, then written, by one thread; the same arithmetic as adamw). w, m and v must be
/// the contiguous storage of the parameter and its moments (a Var's own tensor), and nothing else may hold a view
/// that expects the old values. g is only read.
pub fn adamw_in_place(w: &Tensor, g: &Tensor, m: &Tensor, v: &Tensor, sc: &AdamWScalars) -> Result<()> {
    let n = w.elem_count();
    if !(w.is_contiguous() && m.is_contiguous() && v.is_contiguous()) || m.elem_count() != n || v.elem_count() != n || g.elem_count() != n {
        candle_core::bail!("adamw_in_place: w, m, v must be contiguous and of one size");
    }
    let g = g.contiguous()?;
    let (ws, wl) = w.storage_and_layout();
    let (gs, gl) = g.storage_and_layout();
    let (ms, ml) = m.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let (Storage::Cuda(ws), Storage::Cuda(gs), Storage::Cuda(ms), Storage::Cuda(vs)) = (&*ws, &*gs, &*ms, &*vs) else {
        candle_core::bail!("adamw_in_place: CUDA only");
    };
    let dev = ws.device().clone();
    let (wv, gv, mv, vv) = (view(ws, wl, "adamw")?, view(gs, gl, "adamw")?, view(ms, ml, "adamw")?, view(vs, vl, "adamw")?);
    let func = dev.get_or_load_custom_func("adamw_inplace", MODULE, PTX)?;
    let ni = n as i32;
    let f = |x: f64| x as f32;
    let (s, b1, c1, b2, c2) = (f(sc.grad_scale), f(sc.b1), f(1.0 - sc.b1), f(sc.b2), f(1.0 - sc.b2));
    let (eps, step, ibc2, shrink) = (f(sc.eps), f(sc.step_size), f(sc.inv_sqrt_bc2), f(sc.shrink));
    let mut lb = func.builder();
    lb.arg(&wv).arg(&gv).arg(&mv).arg(&vv).arg(&ni);
    lb.arg(&s).arg(&b1).arg(&c1).arg(&b2).arg(&c2).arg(&eps).arg(&step).arg(&ibc2).arg(&shrink);
    unsafe { lb.launch(elems_cfg(n)) }.w()?;
    Ok(())
}

/// Speed round 13b: adamw_in_place with the clipping factor taken from `scale` (an f64 scalar on the device, from
/// optim::clip_scale_t) instead of sc.grad_scale, so the step needs no copy to the host. Same arithmetic.
pub fn adamw_in_place_s(w: &Tensor, g: &Tensor, m: &Tensor, v: &Tensor, scale: &Tensor, sc: &AdamWScalars) -> Result<()> {
    let n = w.elem_count();
    if !(w.is_contiguous() && m.is_contiguous() && v.is_contiguous()) || m.elem_count() != n || v.elem_count() != n || g.elem_count() != n {
        candle_core::bail!("adamw_in_place_s: w, m, v must be contiguous and of one size");
    }
    if scale.dtype() != DType::F64 || scale.elem_count() != 1 {
        candle_core::bail!("adamw_in_place_s: scale must be one f64");
    }
    let (g, scale) = (g.contiguous()?, scale.contiguous()?);
    let (ws, wl) = w.storage_and_layout();
    let (gs, gl) = g.storage_and_layout();
    let (ms, ml) = m.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let (ss, sl) = scale.storage_and_layout();
    let (Storage::Cuda(ws), Storage::Cuda(gs), Storage::Cuda(ms), Storage::Cuda(vs), Storage::Cuda(ss)) = (&*ws, &*gs, &*ms, &*vs, &*ss) else {
        candle_core::bail!("adamw_in_place_s: CUDA only");
    };
    let dev = ws.device().clone();
    let (wv, gv, mv, vv) = (view(ws, wl, "adamw")?, view(gs, gl, "adamw")?, view(ms, ml, "adamw")?, view(vs, vl, "adamw")?);
    let sv = ss.as_cuda_slice::<f64>()?.slice(sl.start_offset()..sl.start_offset() + 1);
    let func = dev.get_or_load_custom_func("adamw_inplace_s", MODULE, PTX)?;
    let ni = n as i32;
    let f = |x: f64| x as f32;
    let (b1, c1, b2, c2) = (f(sc.b1), f(1.0 - sc.b1), f(sc.b2), f(1.0 - sc.b2));
    let (eps, step, ibc2, shrink) = (f(sc.eps), f(sc.step_size), f(sc.inv_sqrt_bc2), f(sc.shrink));
    let mut lb = func.builder();
    lb.arg(&wv).arg(&gv).arg(&mv).arg(&vv).arg(&sv).arg(&ni);
    lb.arg(&b1).arg(&c1).arg(&b2).arg(&c2).arg(&eps).arg(&step).arg(&ibc2).arg(&shrink);
    unsafe { lb.launch(elems_cfg(n)) }.w()?;
    Ok(())
}

/// v40: the magnitude code of every token of `ids` [B, T] (u32), as [B·T, MAG_F] f32 (src/mag.rs; one CUDA thread
/// per token, or that rule itself on the CPU).
pub fn mag_feats(ids: &Tensor) -> Result<Tensor> {
    use crate::mag::{feats_row, MAG_F};
    let (b, t) = ids.dims2()?;
    let n = b * t;
    if n == 0 || !ids.device().is_cuda() {
        let v: Vec<u32> = ids.flatten_all()?.to_vec1()?;
        let mut out = Vec::with_capacity(n * MAG_F);
        for row in v.chunks(t.max(1)) {
            out.extend(feats_row(row));
        }
        return Tensor::from_vec(out, (n, MAG_F), ids.device());
    }
    let ids = ids.contiguous()?;
    let (is, il) = ids.storage_and_layout();
    let Storage::Cuda(is) = &*is else {
        candle_core::bail!("mag_feats: CUDA only");
    };
    let dev = is.device().clone();
    let ids_v = match il.contiguous_offsets() {
        Some((a, b)) => is.as_cuda_slice::<u32>()?.slice(a..b),
        None => candle_core::bail!("mag_feats: ids must be contiguous"),
    };
    let out = unsafe { dev.alloc::<f32>(n * MAG_F)? };
    let func = dev.get_or_load_custom_func("mag_feats", MODULE, PTX)?;
    let (ni, ti) = (n as i32, t as i32);
    let mut lb = func.builder();
    lb.arg(&ids_v).arg(&out).arg(&ni).arg(&ti);
    unsafe { lb.launch(elems_cfg(n)) }.w()?;
    Ok(Tensor::from_storage(Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)), (n, MAG_F), BackpropOp::none(), false))
}
