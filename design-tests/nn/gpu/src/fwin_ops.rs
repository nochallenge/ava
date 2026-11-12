// TESTING / DESIGN ONLY — ava_gpu::fwin_ops: v47, the kernels of model D's hand-written training step with Stage 3's
// winning block options (src/model/fwin.rs). Kernels in src/lif.cu (section "v47"); CUDA only, no autograd.
//
//   kwta_relu2 / relu2_bwd     --mlp-act relu2: the kWTA winners' relu squared (a and a^2 in one pass), and du = 2 a da
//   wmem_fwd / wmem_bwd        the fast-weight memory at head size 32 or 64 (--head-size 64), with --value-residual:
//                              prep (unit q, k; gates; v mix) -> the memg scan (src/mem_gen.rs's kernels) -> post (head
//                              norm, silu gate); the backward is post_bwd -> memg_bwd -> prep_bwd
//   dense_mix / dense_back     --residual dense (DenseFormer): one block's mix, and its backward (the stream's gradient
//                              and the mixing weights' dot products)
//   wmem_fwd_nm / wmem_bwd_nm  v51 fused N: wmem_* with --neuromod's ACh on the write strengths between the prep and
//                              the scan (src/fwin_nm.rs); without a signal exactly wmem_fwd_keep / wmem_bwd_pre

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::WrapErr;
use candle_core::op::BackpropOp;
use candle_core::{CudaStorage, DType, Device, Result, Shape, Storage, Tensor};

use crate::arch::Spikes;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/lif.ptx"));
const MODULE: &str = "ava_lif";
/// The most mixing weights one dense block has (DENSE_MAXD in lif.cu): layers + 1 <= this.
pub const DENSE_MAXD: usize = 32;

/// A kernel argument: a contiguous CUDA tensor (f32 or bf16; read, or written in place when the kernel's output), or a
/// scalar.
pub enum A<'a> {
    T(&'a Tensor),
    I(i32),
    L(i64),
    F(f32),
}

enum V<'v> {
    F(CudaView<'v, f32>),
    B(CudaView<'v, half::bf16>),
}

/// Launches `name` with `args` in order. Output tensors are passed like inputs (made by `empty`); the kernel writes them.
pub(crate) fn launch(name: &str, args: &[A], cfg: LaunchConfig) -> Result<()> {
    let tens: Vec<&Tensor> = args.iter().filter_map(|a| if let A::T(t) = a { Some(*t) } else { None }).collect();
    let guards: Vec<_> = tens.iter().map(|t| t.storage_and_layout()).collect();
    let mut dev = None;
    let mut views = Vec::with_capacity(guards.len());
    for (g, l) in guards.iter() {
        let Storage::Cuda(s) = &**g else {
            candle_core::bail!("{name}: CUDA only");
        };
        if dev.is_none() {
            dev = Some(s.device().clone());
        }
        let Some((a, b)) = l.contiguous_offsets() else {
            candle_core::bail!("{name}: contiguous tensors only");
        };
        views.push(match s.dtype() {
            DType::F32 => V::F(s.as_cuda_slice::<f32>()?.slice(a..b)),
            DType::BF16 => V::B(s.as_cuda_slice::<half::bf16>()?.slice(a..b)),
            d => candle_core::bail!("{name}: f32 or bf16 tensors only, got {d:?}"),
        });
    }
    let dev = dev.ok_or_else(|| candle_core::Error::Msg(format!("{name}: no tensor argument")))?;
    let func = dev.get_or_load_custom_func(name, MODULE, PTX)?;
    let mut lb = func.builder();
    let mut vi = views.iter();
    for a in args {
        match a {
            A::T(_) => match vi.next().expect("one view per tensor") {
                V::F(x) => {
                    lb.arg(x);
                }
                V::B(x) => {
                    lb.arg(x);
                }
            },
            A::I(x) => {
                lb.arg(x);
            }
            A::L(x) => {
                lb.arg(x);
            }
            A::F(x) => {
                lb.arg(x);
            }
        }
    }
    unsafe { lb.launch(cfg) }.w()?;
    Ok(())
}

/// An uninitialised tensor of `shape` and `dt` on `like`'s CUDA device (for a kernel to write in full).
pub fn empty<S: Into<Shape>>(like: &Tensor, shape: S, dt: DType) -> Result<Tensor> {
    let Device::Cuda(dev) = like.device() else {
        candle_core::bail!("fwin_ops: CUDA only");
    };
    let shape = shape.into();
    let n = shape.elem_count().max(1);
    let st = match dt {
        DType::F32 => CudaStorage::wrap_cuda_slice(unsafe { dev.alloc::<f32>(n)? }, dev.clone()),
        DType::BF16 => CudaStorage::wrap_cuda_slice(unsafe { dev.alloc::<half::bf16>(n)? }, dev.clone()),
        d => candle_core::bail!("fwin_ops: f32 or bf16 only, got {d:?}"),
    };
    Ok(Tensor::from_storage(Storage::Cuda(st), shape, BackpropOp::none(), false))
}

pub(crate) fn elems_cfg(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

/// One warp per (row, head), 4 warps per block.
pub(crate) fn warps_cfg(warps: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((warps as u32).div_ceil(4), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }
}

fn dt(bf: bool) -> DType {
    if bf {
        DType::BF16
    } else {
        DType::F32
    }
}

// ---------------------------------------------------------------- --mlp-act relu2

/// kWTA (k per row) then relu squared, for u [N, H] f32: (a, a^2), a = kwta_relu(u) (bit for bit), both bf16 if `bf`.
pub fn kwta_relu2(u: &Tensor, k: usize, bf: bool) -> Result<(Tensor, Tensor)> {
    let (n, h) = u.dims2()?;
    if h > 12288 {
        candle_core::bail!("kwta-relu2: rows of {h} exceed the shared-memory limit");
    }
    let u = u.contiguous()?;
    let (a, a2) = (empty(&u, (n, h), dt(bf))?, empty(&u, (n, h), dt(bf))?);
    let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (h * 4) as u32 };
    let name = if bf { "kwta_relu2_rows_b" } else { "kwta_relu2_rows" };
    launch(name, &[A::T(&u), A::T(&a), A::T(&a2), A::I(n as i32), A::I(h as i32), A::I(k.min(h) as i32)], cfg)?;
    Ok((a, a2))
}

/// The gradient of u for kwta_relu2's a2: (a da2) 2 where a > 0, else 0, in a's dtype (`a` from kwta_relu2, da2 f32).
pub fn relu2_bwd(a: &Tensor, da2: &Tensor) -> Result<Tensor> {
    let n = a.elem_count();
    let da2 = da2.contiguous()?;
    let gu = empty(a, a.shape().clone(), a.dtype())?;
    let name = if a.dtype() == DType::BF16 { "relu2_bwd_b" } else { "relu2_bwd" };
    launch(name, &[A::T(a), A::T(&da2), A::T(&gu), A::I(n as i32)], elems_cfg(n))?;
    Ok(gu)
}

// ---------------------------------------------------------------- v47 SLOW: the general LIF (--spikes, --alif, --homeostasis)

/// lif_gen.rs's kernel for the spike kind and ALIF (its Kind::kernel).
fn lifg_name(dir: &str, spikes: Spikes, alif: bool) -> String {
    let s = match spikes {
        Spikes::Binary => "b",
        Spikes::Ternary => "t",
        Spikes::Int2 => "i",
    };
    format!("lifg_{dir}_{s}{}", u8::from(alif))
}

fn lifg_check(cur: &Tensor, ps: &Tensor) -> Result<(usize, usize, usize)> {
    let (b, t, d) = cur.dims3()?;
    if ps.dims() != [4 * d + 3 * b * d] || ps.dtype() != DType::F32 || cur.dtype() != DType::F32 {
        candle_core::bail!("lifg: cur [B, T, d] f32 and ps [4d + 3Bd] f32 wanted, got {:?} {:?}", cur.dims(), ps.dims());
    }
    Ok((b, t, d))
}

/// lif_gen's forward (the same kernel, so the same spikes as the autograd path): cur [B, T, d] f32 and ps = [beta |
/// theta | rho | gain] (d each) then [v0 | s0 | a0] (B·d each) -> [the spikes (B·T·d) | final v | final a (B·d each)].
pub fn lifg_fwd(cur: &Tensor, ps: &Tensor, spikes: Spikes, alif: bool) -> Result<Tensor> {
    let (b, t, d) = lifg_check(cur, ps)?;
    let (cur, ps) = (cur.contiguous()?, ps.contiguous()?);
    let out = empty(&cur, b * t * d + 2 * b * d, DType::F32)?;
    let args = [A::T(&cur), A::T(&ps), A::T(&out), A::I(b as i32), A::I(t as i32), A::I(d as i32)];
    launch(&lifg_name("fwd", spikes, alif), &args, crate::lif::launch_cfg(b * d))?;
    Ok(out)
}

/// lif_gen's backward kernel: `g` the gradient of lifg_fwd's flat output -> [dL/dcur (B·T·d) | dL/dv0 | dL/da0 | per-row
/// dL/dbeta, dL/drho, dL/dgain (B·d each)]. `reset_grad` (--lif-reset-grad): the gradient also flows through the
/// reset, its carry capped (lif.cu's lifg_te).
pub fn lifg_bwd(cur: &Tensor, ps: &Tensor, g: &Tensor, spikes: Spikes, alif: bool, slope: f32, reset_grad: bool) -> Result<Tensor> {
    let (b, t, d) = lifg_check(cur, ps)?;
    let (cur, ps, g) = (cur.contiguous()?, ps.contiguous()?, g.contiguous()?);
    if g.elem_count() != b * t * d + 2 * b * d || g.dtype() != DType::F32 {
        candle_core::bail!("lifg_bwd: g [{}] f32 wanted, got {:?}", b * t * d + 2 * b * d, g.dims());
    }
    let out = empty(&cur, b * t * d + 5 * b * d, DType::F32)?;
    let ah = empty(&cur, if alif { b * t * d } else { 1 }, DType::F32)?;
    let args = [A::T(&cur), A::T(&ps), A::T(&g), A::T(&out), A::T(&ah), A::I(b as i32), A::I(t as i32), A::I(d as i32), A::F(slope), A::I(reset_grad as i32)];
    launch(&lifg_name("bwd", spikes, alif), &args, crate::lif::launch_cfg(b * d))?;
    Ok(out)
}

// ---------------------------------------------------------------- the fast-weight memory (head size 32 or 64)

/// --value-residual for one block's memory: off; `Mix` (blocks after the first: v = v (1 + p0) + v0 p1, `p` [2] the
/// block's weights, `p0` the first block's projection P); `First` (the first block, whose v the later ones read).
#[derive(Clone, Copy)]
pub enum Vres<'a> {
    Off,
    Mix { p: &'a Tensor, p0: &'a Tensor },
    First,
}

/// The memory's shape: b rows of t tokens, nh heads of hs channels; v47 SLOW: `chan` (--gates channel: P2's gates and
/// erase key per channel) and `mom` (--mem-momentum, 0 = none).
#[derive(Clone, Copy, Debug)]
pub struct MemShape {
    pub b: usize,
    pub t: usize,
    pub nh: usize,
    pub hs: usize,
    pub chan: bool,
    pub mom: f32,
}

/// The rank of the channel gates' low-rank parts (arch::GATE_RANK).
pub const GATE_RANK: usize = crate::arch::GATE_RANK;

impl MemShape {
    /// The projection's width: q, k, v, g (d each), za, zb (nh each); with channel gates also e (d) and the gates'
    /// low-rank inputs Wa1, Wb1 (GATE_RANK each).
    pub fn ld(&self) -> usize {
        let d = self.nh * self.hs;
        if self.chan {
            5 * d + 2 * self.nh + 2 * GATE_RANK
        } else {
            4 * d + 2 * self.nh
        }
    }

    /// The constants' length: ba, bb (nh each), then the forget offsets (per head, or per channel).
    pub fn ncst(&self) -> usize {
        2 * self.nh + if self.chan { self.nh * self.hs } else { self.nh }
    }

    /// The gate columns per head in vg (1, or hs per channel).
    pub fn g(&self) -> usize {
        if self.chan {
            self.hs
        } else {
            1
        }
    }

    fn check(&self, p: &Tensor, cst: &Tensor, z: Option<&Tensor>) -> Result<(usize, usize, usize)> {
        let (n, d, ld) = (self.b * self.t, self.nh * self.hs, self.ld());
        let z_ok = match z {
            Some(z) => self.chan && z.dims() == [n, 2 * d],
            None => !self.chan,
        };
        if !crate::mem_gen::supports(self.hs) || p.dims() != [n, ld] || cst.dims() != [self.ncst()] || !z_ok || !self.mom.is_finite() {
            candle_core::bail!(
                "wmem: head size {} (32 or 64), p [{n}, {ld}], cst [{}] and Z [{n}, {}] iff channel gates wanted, got {:?} {:?} {:?}",
                self.hs,
                self.ncst(),
                2 * d,
                p.dims(),
                cst.dims(),
                z.map(|z| z.dims().to_vec())
            );
        }
        Ok((n, d, ld))
    }

    /// The memg kernels' suffix: head size, h (per-head gates) or c (per channel), momentum 0/1.
    fn memg(&self) -> String {
        format!("{}{}{}", self.hs, if self.chan { 'c' } else { 'h' }, u8::from(self.mom != 0.0))
    }
}

/// The scan's inputs (qke [B H, T, 3 hs], vg [B H, T, hs + 2 G]) from the projection p [N, ld] (f32), cst and (channel
/// gates) Z [N, 2d] f32.
fn wprep(p: &Tensor, cst: &Tensor, z: Option<&Tensor>, vr: Vres, s: MemShape) -> Result<(Tensor, Tensor)> {
    let (n, _, ld) = s.check(p, cst, z)?;
    let bn = s.b * s.nh;
    let qke = empty(p, (bn, s.t, 3 * s.hs), DType::F32)?;
    let vg = empty(p, (bn, s.t, s.hs + 2 * s.g()), DType::F32)?;
    let (mode, vp, p0) = match vr {
        Vres::Mix { p: vp, p0 } => (1, vp, p0),
        _ => (0, cst, p),
    };
    let mut args = vec![A::T(p), A::T(cst), A::T(vp), A::T(p0)];
    if let Some(z) = z {
        args.push(A::T(z));
    }
    args.extend([A::T(&qke), A::T(&vg), A::I(n as i32), A::I(s.t as i32), A::I(s.nh as i32), A::I(ld as i32), A::I(mode)]);
    launch(&format!("{}_{}", if s.chan { "wprepc" } else { "wprep" }, s.hs), &args, warps_cfg(n * s.nh))?;
    Ok((qke, vg))
}

/// The memg scan forward: y [B H, T, hs]. v49 (Amendment 6): memgk_fwd (4 hs threads per head, memg_fwd's bits) when
/// mem_gen::mgk is on.
fn memg_fwd(qke: &Tensor, vg: &Tensor, s: MemShape) -> Result<Tensor> {
    memg_fwd_with(qke, vg, s, crate::mem_gen::mgk(s.hs, s.t, s.chan), crate::mem_gen::mgk_groups())
}

/// `k`: memgk_fwd with `g` blocks per head (1, 2 or 4), else memg_fwd.
fn memg_fwd_with(qke: &Tensor, vg: &Tensor, s: MemShape, k: bool, g: usize) -> Result<Tensor> {
    let bn = s.b * s.nh;
    let y = empty(qke, (bn, s.t, s.hs), DType::F32)?;
    if k {
        // v49: g blocks per head of hs / g columns, 4 threads per column
        let cfg = LaunchConfig { grid_dim: ((g * bn) as u32, 1, 1), block_dim: ((4 * s.hs / g) as u32, 1, 1), shared_mem_bytes: 0 };
        let args = [A::T(qke), A::T(vg), A::T(&y), A::I(bn as i32), A::I(s.t as i32), A::F(s.mom), A::I(g as i32), A::F(crate::mem_gen::cap())];
        launch(&format!("memgk_fwd_{}", s.memg()), &args, cfg)?;
    } else {
        let args = [A::T(qke), A::T(vg), A::T(&y), A::I(bn as i32), A::I(s.t as i32), A::F(s.mom), A::F(crate::mem_gen::cap())];
        let cfg = LaunchConfig { grid_dim: (bn as u32, 1, 1), block_dim: (s.hs as u32, 1, 1), shared_mem_bytes: 0 };
        launch(&format!("memg_fwd_{}", s.memg()), &args, cfg)?;
    }
    Ok(y)
}

/// The memg scan backward: (dqke, dvg) for dy. v49: the scan split in time when mem_gen::mgk is on.
fn memg_bwd(qke: &Tensor, vg: &Tensor, dy: &Tensor, s: MemShape) -> Result<(Tensor, Tensor)> {
    if crate::mem_gen::mgk(s.hs, s.t, s.chan) {
        return memg_bwd_k(qke, vg, dy, s, crate::mem_gen::mgk_len(s.hs), crate::mem_gen::mgk_groups());
    }
    memg_bwd_with(qke, vg, dy, s, crate::mem_gen::bwd4(s.hs, s.mom != 0.0))
}

/// v49 (runs/v47 Amendment 6): memg_bwd4 split in time, the same ops in the same order. memgk_scan walks the forward
/// (saving (F, S) at the start of every item of `l` steps) and the carried gradients (saving (dF, dS) as they enter every
/// item) at once, 2 B H blocks; then memgk_bwd's persistent blocks take the (head, item) pairs, each from its exact
/// boundary values, with memg_bwd4's per-step code. `l` a multiple of ck_len.
fn memg_bwd_k(qke: &Tensor, vg: &Tensor, dy: &Tensor, s: MemShape, l: usize, g: usize) -> Result<(Tensor, Tensor)> {
    let (bn, hs) = (s.b * s.nh, s.hs);
    let c = crate::mem_gen::ck_len(hs);
    if l == 0 || l % c != 0 || !crate::mem_gen::supports(hs) {
        candle_core::bail!("memg_bwd_k: item length {l} must be a multiple of the chunk {c}, head size 32 or 64 (got {hs})");
    }
    let dy = dy.contiguous()?;
    let ns = if s.mom != 0.0 { 2 } else { 1 };
    let sq = ns * hs * hs;
    let nit = s.t.div_ceil(l);
    let dqke = empty(qke, (bn, s.t, 3 * hs), DType::F32)?;
    let dvg = empty(qke, (bn, s.t, hs + 2 * s.g()), DType::F32)?;
    let ck = empty(qke, bn * nit * sq, DType::F32)?;
    let ca = empty(qke, bn * nit * sq, DType::F32)?;
    let name = s.memg();
    let nb = mgk_blocks(qke, &format!("memgk_bwd_{name}"), 4 * hs, bn * nit)?;
    let scr = empty(qke, nb * (l / c + c) * sq, DType::F32)?;
    let blk = (4 * hs) as u32;
    let (bi, ti, m) = (bn as i32, s.t as i32, s.mom);
    let cap = crate::mem_gen::cap();
    // v52 fix_S: the forward walk's blocks each say whether the bound held any element of their columns
    let sat = empty(qke, g * bn, DType::F32)?;
    let cfg = LaunchConfig { grid_dim: ((2 * g * bn) as u32, 1, 1), block_dim: ((4 * hs / g) as u32, 1, 1), shared_mem_bytes: 0 };
    let args = [A::T(qke), A::T(vg), A::T(&dy), A::T(&ck), A::T(&ca), A::I(bi), A::I(ti), A::F(m), A::I(l as i32), A::I(g as i32), A::F(cap), A::T(&sat)];
    launch(&format!("memgk_scan_{name}"), &args, cfg)?;
    if sat.to_vec1::<f32>()?.iter().any(|&x| x != 0.0) {
        // the carried gradients at the item starts came from the adjoint walk, which ran without the states and so could
        // not zero the gradient where the bound held: walk the whole sequence with memg_bwd4 instead (its per-step code
        // is memgk_bwd's, so a head the bound never held gets the same bits either way). Rare: a runaway head only.
        crate::mem_gen::SAT_EVENTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return memg_bwd_with(qke, vg, &dy, s, hs == 64);
    }
    let cfg = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (blk, 1, 1), shared_mem_bytes: 0 };
    let args = [A::T(qke), A::T(vg), A::T(&dy), A::T(&dqke), A::T(&dvg), A::T(&ck), A::T(&ca), A::T(&scr), A::I(bi), A::I(ti), A::F(m), A::I(l as i32), A::I(c as i32), A::F(cap)];
    launch(&format!("memgk_bwd_{name}"), &args, cfg)?;
    Ok((dqke, dvg))
}

/// v49: the persistent grid for `name` (blocks of `threads`): the GPU's SMs times the blocks one SM holds (the driver's
/// occupancy), at most `items`. AVA_MGK_P overrides the total. Read once per kernel.
fn mgk_blocks(like: &Tensor, name: &str, threads: usize, items: usize) -> Result<usize> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static ENV: OnceLock<Option<usize>> = OnceLock::new();
    static SEEN: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();
    if let Some(p) = *ENV.get_or_init(|| std::env::var("AVA_MGK_P").ok().and_then(|v| v.parse().ok()).filter(|&p| p >= 1)) {
        return Ok(p.min(items).max(1));
    }
    let seen = SEEN.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&p) = seen.lock().unwrap().get(name) {
        return Ok(p.min(items).max(1));
    }
    let Device::Cuda(dev) = like.device() else {
        candle_core::bail!("{name}: CUDA only");
    };
    let func = dev.get_or_load_custom_func(name, MODULE, PTX)?;
    let per = func.occupancy_max_active_blocks_per_multiprocessor(threads as u32, 0, None).w()? as usize;
    let sms = dev.cuda_stream().context().attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).w()? as usize;
    let p = (per.max(1) * sms).max(1);
    seen.lock().unwrap().insert(name.to_string(), p);
    Ok(p.min(items).max(1))
}

/// memg_bwd with the kernel chosen: `b4` = memg_bwd4 (head size 64 only), else the old one.
fn memg_bwd_with(qke: &Tensor, vg: &Tensor, dy: &Tensor, s: MemShape, b4: bool) -> Result<(Tensor, Tensor)> {
    let (bn, hs) = (s.b * s.nh, s.hs);
    let ns = if s.mom != 0.0 { 2 } else { 1 };
    let dqke = empty(qke, (bn, s.t, 3 * hs), DType::F32)?;
    let dvg = empty(qke, (bn, s.t, hs + 2 * s.g()), DType::F32)?;
    let ckl = crate::mem_gen::ck_len(hs);
    let ck = empty(qke, bn * s.t.div_ceil(ckl) * ns * hs * hs, DType::F32)?;
    let scr = empty(qke, bn * ckl * ns * hs * hs, DType::F32)?;
    let cfg = LaunchConfig { grid_dim: (bn as u32, 1, 1), block_dim: (if b4 { 4 * hs } else { hs } as u32, 1, 1), shared_mem_bytes: 0 };
    let args = [A::T(qke), A::T(vg), A::T(dy), A::T(&dqke), A::T(&dvg), A::T(&ck), A::T(&scr), A::I(bn as i32), A::I(s.t as i32), A::F(s.mom), A::I(ckl as i32), A::F(crate::mem_gen::cap())];
    launch(&format!("memg_bwd{}_{}", if b4 { "4" } else { "" }, s.memg()), &args, cfg)?;
    Ok((dqke, dvg))
}

/// The memory's read-out O [N, d] (bf16 if `bf`, else f32) for the projection p = h·[Wq|Wk|Wv|Wg|Wa|Wb] [N, 4d + 2nh]
/// (f32) and cst = (ba, bb, off) [3 nh]; also the scan's y (for the backward). arch.rs mem_arch's maths for D.
/// v47 SLOW: with `s.chan`, p = h·[Wq|Wk|Wv|Wg|Wa|Wb|We|Wa1|Wb1] (`s.ld()`), cst = (ba, bb, the channel offsets)
/// [2 nh + d] and `z` = (h Wa1 Wa2 | h Wb1 Wb2) [N, 2d] f32; `s.mom`: the memory's momentum.
pub fn wmem_fwd(p: &Tensor, cst: &Tensor, z: Option<&Tensor>, vr: Vres, s: MemShape, eps: f64, bf: bool) -> Result<(Tensor, Tensor)> {
    let (o, y, _) = wmem_fwd_keep(p, cst, z, vr, s, eps, bf, false)?;
    Ok((o, y))
}

/// wmem_fwd, also handing back the scan inputs (qke, vg) when `keep`, so wmem_bwd_pre need not rerun the prep (v47:
/// the same kernel on the same inputs, so the same bits; costs N 6d f32 per layer at per-channel gates).
#[allow(clippy::too_many_arguments)]
pub fn wmem_fwd_keep(p: &Tensor, cst: &Tensor, z: Option<&Tensor>, vr: Vres, s: MemShape, eps: f64, bf: bool, keep: bool) -> Result<(Tensor, Tensor, Option<(Tensor, Tensor)>)> {
    let (o, y, pre, _) = wmem_fwd_nm(p, cst, z, vr, s, eps, bf, keep, None)?;
    Ok((o, y, pre))
}

/// v51 fused N: wmem_fwd_keep with --neuromod's ACh signal `ach` [N, 1] (src/fwin_nm.rs): the scan's write strengths
/// (vg's beta columns) times the token's signal, kept in [0, 2], between the prep and the scan; also hands back the
/// unscaled strengths [B, H, T, G] when `keep` (wmem_bwd_nm needs them). None: wmem_fwd_keep exactly (no op added).
#[allow(clippy::too_many_arguments)]
pub fn wmem_fwd_nm(
    p: &Tensor,
    cst: &Tensor,
    z: Option<&Tensor>,
    vr: Vres,
    s: MemShape,
    eps: f64,
    bf: bool,
    keep: bool,
    ach: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Option<(Tensor, Tensor)>, Option<Tensor>)> {
    let (o, y, pre, braw, _) = wmem_fwd_g(p, cst, z, vr, s, eps, bf, keep, ach, None)?;
    Ok((o, y, pre, braw))
}

/// v51 fast1 (src/fwin_r2.rs, --rhythm): wmem_fwd_nm with Y's write gate `wgate` [B·H, T, 1] on the write strengths
/// right after the prep (before ACh, as mem_arch: v51's gate, then v50's clamp); also hands back the ungated strengths
/// [B·H, T, G] when `keep`. None: wmem_fwd_nm exactly (no op added).
#[allow(clippy::too_many_arguments)]
pub fn wmem_fwd_g(
    p: &Tensor,
    cst: &Tensor,
    z: Option<&Tensor>,
    vr: Vres,
    s: MemShape,
    eps: f64,
    bf: bool,
    keep: bool,
    ach: Option<&Tensor>,
    wgate: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Option<(Tensor, Tensor)>, Option<Tensor>, Option<Tensor>)> {
    let (n, d, ld) = s.check(p, cst, z)?;
    let (qke, vg) = wprep(p, cst, z, vr, s)?;
    let bung = match wgate {
        Some(g) => Some(crate::fwin_r2::wgate_fwd(&vg, g, s.g())?), // vg's beta columns gated in place
        None => None,
    };
    let bung = if keep { bung } else { None };
    let braw = match ach {
        Some(a) => Some(crate::fwin_nm::ach_fwd(&vg, a, s)?), // vg's beta columns scaled in place
        None => None,
    };
    let y = memg_fwd(&qke, &vg, s)?;
    let pre = if keep { Some((qke, vg)) } else { None };
    let braw = if keep { braw } else { None };
    let o = empty(p, (n, d), dt(bf))?;
    let args = [A::T(p), A::T(&y), A::T(&o), A::I(n as i32), A::I(s.t as i32), A::I(s.nh as i32), A::I(ld as i32), A::F(eps as f32)];
    launch(&format!("{}_{}", if bf { "wpost_b" } else { "wpost" }, s.hs), &args, warps_cfg(n * s.nh))?;
    Ok((o, y, pre, braw, bung))
}

/// The backward of wmem_fwd: what one block's memory sends back.
pub struct WmemGrads {
    /// dP [N, ld] (bf16 if bf, else f32): the matmul operand for the projection's weight and input gradients
    pub dp: Tensor,
    /// dza, dzb [N, 2 nh] f32 (the bias gradients are their column sums)
    pub dz: Tensor,
    /// Vres::Mix: the first block's v gradient so far (acc_in plus this block's share), [N, d]
    pub acc: Option<Tensor>,
    /// Vres::Mix: (dp0, dp1) [2]
    pub dvres: Option<Tensor>,
    /// channel gates: dZ [N, 2d] (bf16 if bf, else f32). dP's Wa1 and Wb1 columns are NOT written: the caller fills them
    /// with dZ times the low-rank weights' transposes before using dP.
    pub dzc: Option<Tensor>,
    /// v51 fused N (wmem_bwd_nm with ACh): dL/d(the ACh signal) [N, 1] f32
    pub dach: Option<Tensor>,
    /// v51 fast1 (wmem_bwd_g with Y's write gate): dL/d(the gate) [B·H, T, 1] f32
    pub dwgate: Option<Tensor>,
}

/// dP etc. for dO [N, d] (f32), from the forward's p, cst, z and y. `acc_in`: Vres::Mix, the first block's v gradient
/// from the later blocks so far (None: none yet); Vres::First, all of it (added to its own dv).
#[allow(clippy::too_many_arguments)]
pub fn wmem_bwd(p: &Tensor, cst: &Tensor, z: Option<&Tensor>, vr: Vres, y: &Tensor, d_o: &Tensor, acc_in: Option<&Tensor>, s: MemShape, eps: f64, bf: bool) -> Result<WmemGrads> {
    wmem_bwd_pre(p, cst, z, vr, y, d_o, acc_in, s, eps, bf, None)
}

/// wmem_bwd with the forward's scan inputs (qke, vg) when it kept them (wmem_fwd_keep); else the prep is rerun.
#[allow(clippy::too_many_arguments)]
pub fn wmem_bwd_pre(p: &Tensor, cst: &Tensor, z: Option<&Tensor>, vr: Vres, y: &Tensor, d_o: &Tensor, acc_in: Option<&Tensor>, s: MemShape, eps: f64, bf: bool, pre: Option<(&Tensor, &Tensor)>) -> Result<WmemGrads> {
    wmem_bwd_nm(p, cst, z, vr, y, d_o, acc_in, s, eps, bf, pre, None)
}

/// v51 fused N: wmem_bwd_pre for wmem_fwd_nm with --neuromod's ACh: `ach` = (the signal [N, 1], the unscaled write
/// strengths wmem_fwd_nm kept, or None when the forward kept no scan inputs: the prep is then rerun and scaled again,
/// the same ops on the same inputs). Between the scan's backward and the prep's, dL/d(scaled strengths) becomes
/// dL/d(strengths) in place, and WmemGrads::dach = dL/d(signal) [N, 1]. None: wmem_bwd_pre exactly (no op added).
#[allow(clippy::too_many_arguments)]
pub fn wmem_bwd_nm(
    p: &Tensor,
    cst: &Tensor,
    z: Option<&Tensor>,
    vr: Vres,
    y: &Tensor,
    d_o: &Tensor,
    acc_in: Option<&Tensor>,
    s: MemShape,
    eps: f64,
    bf: bool,
    pre: Option<(&Tensor, &Tensor)>,
    ach: Option<(&Tensor, Option<&Tensor>)>,
) -> Result<WmemGrads> {
    wmem_bwd_g(p, cst, z, vr, y, d_o, acc_in, s, eps, bf, pre, ach, None)
}

/// v51 fast1 (src/fwin_r2.rs, --rhythm): wmem_bwd_nm for wmem_fwd_g with Y's write gate: `wgate` = (the gate [B·H, T, 1],
/// the ungated strengths wmem_fwd_g kept, or None when the forward kept no scan inputs: the rerun prep is gated again).
/// After ACh's backward, dL/d(gated strengths) becomes dL/d(strengths) in place, and WmemGrads::dwgate = dL/d(gate).
/// None: wmem_bwd_nm exactly (no op added).
#[allow(clippy::too_many_arguments)]
pub fn wmem_bwd_g(
    p: &Tensor,
    cst: &Tensor,
    z: Option<&Tensor>,
    vr: Vres,
    y: &Tensor,
    d_o: &Tensor,
    acc_in: Option<&Tensor>,
    s: MemShape,
    eps: f64,
    bf: bool,
    pre: Option<(&Tensor, &Tensor)>,
    ach: Option<(&Tensor, Option<&Tensor>)>,
    wgate: Option<(&Tensor, Option<&Tensor>)>,
) -> Result<WmemGrads> {
    let (n, d, ld) = s.check(p, cst, z)?;
    let d_o = d_o.contiguous()?;
    if d_o.dims() != [n, d] || d_o.dtype() != DType::F32 {
        candle_core::bail!("wmem_bwd: dO [{n}, {d}] f32 wanted, got {:?} {:?}", d_o.dims(), d_o.dtype());
    }
    let dp = empty(p, (n, ld), dt(bf))?;
    let dy = empty(p, (s.b * s.nh, s.t, s.hs), DType::F32)?;
    let args = [A::T(p), A::T(y), A::T(&d_o), A::T(&dy), A::T(&dp), A::I(n as i32), A::I(s.t as i32), A::I(s.nh as i32), A::I(ld as i32), A::F(eps as f32)];
    launch(&format!("{}_{}", if bf { "wpost_bwd_b" } else { "wpost_bwd" }, s.hs), &args, warps_cfg(n * s.nh))?;
    let (qke, vg) = match pre {
        Some((a, b)) => (a.clone(), b.clone()),
        None => wprep(p, cst, z, vr, s)?,
    };
    // v51 fast1 (Y): the ungated strengths; a rerun prep is gated again first (ACh and the scan saw the gated ones)
    let bung = match (wgate, pre) {
        (None, _) => None,
        (Some((_, Some(bu))), Some(_)) => Some(bu.clone()),
        (Some((g, _)), None) => Some(crate::fwin_r2::wgate_fwd(&vg, g, s.g())?),
        (Some((_, None)), Some(_)) => candle_core::bail!("wmem_bwd_g: kept scan inputs need the kept ungated strengths"),
    };
    // v51 fused N (ACh): the unscaled strengths; a rerun prep is scaled again first (the scan saw the scaled ones)
    let braw = match (ach, pre) {
        (None, _) => None,
        (Some((_, Some(br))), Some(_)) => Some(br.clone()),
        (Some((a, _)), None) => Some(crate::fwin_nm::ach_fwd(&vg, a, s)?),
        (Some((_, None)), Some(_)) => candle_core::bail!("wmem_bwd_nm: kept scan inputs need the kept unscaled strengths"),
    };
    let (dqke, dvg) = memg_bwd(&qke, &vg, &dy, s)?;
    drop((qke, vg, dy));
    let dach = match (ach, &braw) {
        (Some((a, _)), Some(br)) => Some(crate::fwin_nm::ach_bwd(&dvg, br, a, s)?), // dvg's beta columns in place
        _ => None,
    };
    drop(braw);
    let dwgate = match (wgate, &bung) {
        (Some((g, _)), Some(bu)) => Some(crate::fwin_r2::wgate_bwd(&dvg, bu, g, s.g())?), // dvg's beta columns in place
        _ => None,
    };
    drop(bung);
    let dz = empty(p, (n, 2 * s.nh), DType::F32)?;
    let (mode, vp, p0, acc_out, part) = match vr {
        Vres::Mix { p: vp, p0 } => (1, vp, p0, Some(empty(p, (n, d), DType::F32)?), Some(empty(p, (n * s.nh, 2), DType::F32)?)),
        Vres::First => (2, cst, p, None, None),
        Vres::Off => (0, cst, p, None, None),
    };
    let acc_t = acc_in.map(|a| a.contiguous()).transpose()?;
    if let Some(a) = &acc_t {
        if a.dims() != [n, d] {
            candle_core::bail!("wmem_bwd: acc [{n}, {d}] wanted, got {:?}", a.dims());
        }
    }
    let has_acc = i32::from(acc_t.is_some());
    // absent buffers: any valid pointer (the kernel does not touch them in that mode)
    let accin = acc_t.as_ref().unwrap_or(&dz);
    let (ao, pt) = (acc_out.as_ref().unwrap_or(&dz), part.as_ref().unwrap_or(&dz));
    let dzc = match z {
        Some(_) => Some(empty(p, (n, 2 * d), dt(bf))?),
        None => None,
    };
    let mut args = vec![A::T(p), A::T(cst), A::T(vp), A::T(p0)];
    if let Some(z) = z {
        args.push(A::T(z));
    }
    args.extend([A::T(accin), A::T(&dqke), A::T(&dvg), A::T(&dp), A::T(&dz)]);
    if let Some(dzc) = &dzc {
        args.push(A::T(dzc));
    }
    args.extend([A::T(ao), A::T(pt), A::I(n as i32), A::I(s.t as i32), A::I(s.nh as i32), A::I(ld as i32), A::I(mode), A::I(has_acc)]);
    let name = format!("{}{}{}_{}", if s.chan { "wprepc" } else { "wprep" }, "_bwd", if bf { "_b" } else { "" }, s.hs);
    launch(&name, &args, warps_cfg(n * s.nh))?;
    let dvres = part.map(|pt| pt.sum(0)).transpose()?;
    Ok(WmemGrads { dp, dz, acc: acc_out, dvres, dzc, dach, dwgate })
}

// ---------------------------------------------------------------- --residual dense (DenseFormer)

/// Block i's DenseFormer step: raw = xa + xb (its last residual add) is written to row i + 1 of the streams `s` [L + 1,
/// M] (M = N·d), and the mix raw + sum_{j <= i+1} w_ij s_j returned [N, d]; `wf` holds every block's weights in one
/// vector (block i's from i (i + 3) / 2).
pub fn dense_mix(xa: &Tensor, xb: &Tensor, wf: &Tensor, s: &Tensor, i: usize) -> Result<Tensor> {
    let (xa, xb) = (xa.contiguous()?, xb.contiguous()?);
    let m = xa.elem_count();
    if xb.elem_count() != m || s.dim(1)? != m || s.dim(0)? < i + 2 {
        candle_core::bail!("dense_mix: shapes {:?} {:?} {:?} at block {i}", xa.dims(), xb.dims(), s.dims());
    }
    let out = empty(&xa, xa.shape().clone(), DType::F32)?;
    launch("dense_mix", &[A::T(&xa), A::T(&xb), A::T(wf), A::T(s), A::T(&out), A::I(i as i32), A::L(m as i64)], elems_cfg(m))?;
    Ok(out)
}

/// out = base + sum_{k = i0..L-1} w_k[col] g_k (g [L, M]: row k = dL/d(the mix after block k)), with its bf16 copy if
/// `bf`, and (nd > 0) the dot products sum_e base_e s_{j, e}, j < nd [nd]. Block i: base = g_i, col = i + 1, i0 = i,
/// nd = i + 2 (dL/d its raw output, and dL/dw_i). The embedding: base = dL/dx0 from block 0, col 0, i0 0, nd 0.
#[allow(clippy::too_many_arguments)]
pub fn dense_back(s: &Tensor, g: &Tensor, wf: &Tensor, base: &Tensor, col: usize, i0: usize, nd: usize, bf: bool) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
    let (l, m) = g.dims2()?;
    let base = base.contiguous()?;
    if base.elem_count() != m || s.dim(1)? != m || nd > DENSE_MAXD || nd > s.dim(0)? || i0 > l {
        candle_core::bail!("dense_back: shapes {:?} {:?} {:?}, nd {nd}", s.dims(), g.dims(), base.dims());
    }
    let out = empty(&base, m, DType::F32)?;
    let blocks = m.div_ceil(256).clamp(1, 512);
    let part = empty(&base, (blocks, nd.max(1)), DType::F32)?;
    let cfg = LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let sc = [A::I(col as i32), A::I(i0 as i32), A::I(l as i32), A::I(nd as i32), A::L(m as i64)];
    let outb = if bf {
        let ob = empty(&base, m, DType::BF16)?;
        let mut args = vec![A::T(s), A::T(g), A::T(wf), A::T(&base), A::T(&out), A::T(&ob), A::T(&part)];
        args.extend(sc);
        launch("dense_back_b", &args, cfg)?;
        Some(ob)
    } else {
        let mut args = vec![A::T(s), A::T(g), A::T(wf), A::T(&base), A::T(&out), A::T(&part)];
        args.extend(sc);
        launch("dense_back", &args, cfg)?;
        None
    };
    let dw = if nd > 0 { Some(part.sum(0)?) } else { None };
    Ok((out, outb, dw))
}

#[cfg(test)]
mod tests;
