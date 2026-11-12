// TESTING / DESIGN ONLY — ava_gpu::model::fwin::f2: v51 fast2, four of round 2's brain parts on the hand-written WIN
// training step (src/model/fwin.rs), so they train at the fused step's speed instead of candle's autograd
// (runs/v51/fast2_check.sh checks them on the GPU; runs/v51/fast2.md says what was done):
//   R  --letters             the letter vectors of the whole vocabulary (one kernel, f2_let_fwd), through Wo, gathered
//                            into the input; backward: the rows' gradients scattered (embed_bwd), dWo, and the three
//                            tables' gradients by a chunked row list (f2_let_bwd_part, f2_let_bwd_red; no atomics)
//   W  --columns K           the MLP's two column-wired matrices folded into one dense matrix each, W' = W ⊙ M + A B (M
//                            the K diagonal blocks): the same product as v51::col_mm in one matmul; kWTA per column on
//                            the rows reshaped [N K, hidden / K]; backward dW = dW' ⊙ M, dA = dW' B^T, dB = A^T dW'
//   TH --thalamus            the relay's reads, gates and the two rules in candle ops (all small); backward by hand:
//                            the gates' sigmoid and tanh, the reads' rmsnorm (fused::rms_bwd_raw), and the two per-row
//                            scales through fwin_nm::rows_bwd (the memory read times g_mem; the block's update times
//                            g_res - 1)
//   CB --cerebellum-head G   the granule code in one kernel per row (f2_cb_code: the fixed fan-in-4 sums, kWTA's radix
//                            select, relu, unit length), the read-out c W as one matmul added to her final normed
//                            state; backward: dW = c^T dxf (the code is detached, so xf's gradient passes unchanged)
// The maths are the reference path's (arch.rs forward_arch / block_arch, v51::embed / v51::mlp / v51::col_mm,
// v51b's th_* and cb_* functions), so a model computes the same numbers on either path up to rounding. Every kernel has
// a candle-op twin here (`*_ops`), used on the CPU (where the tests check the twins against autograd) and on CUDA when
// AVA_F2_K=0; the GPU tests check the kernels against the twins.
// Off (none of the four options): nothing here runs (F2::new holds only Nones and every hook hands its input back).
// Not here: TH's --loop gate (the fused step refuses --loop), CB's online table and plastic sessions (talk-time state:
// `model_ok` sends such a model to autograd).

use super::super::{Mm, Model, NORM_EPS};
use crate::arch::Arch;
use crate::fused::{embed_bwd_add, rms_bwd_raw, rms_fwd_raw};
use crate::model::{norm, rmsnorm, sigmoid, Config};
use candle_core::backend::BackendStorage;
use candle_core::backprop::GradStore;
use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Device, Result, Storage, Tensor};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/lif.ptx"));
const MODULE: &str = "ava_lif";
/// R: the letter tables' rows (bytes, then places from the start, then places from the end) and the most entries one
/// backward chunk sums.
const LET_ROWS: usize = 256 + 2 * crate::v51::LETTER_MAX;
const LET_CHUNK: usize = 32;
/// CB: the largest G one kernel row holds (its keys in shared memory, 4 G bytes plus 5 KB of its own; past 48 KB the
/// launch asks for the opt-in size, at most 99 KB on this card). A larger G trains through autograd. v51 cbfix: a
/// device with a smaller opt-in limit holds less (cb_max_g); Model::win_step checks it (device_ok).
pub const CB_MAX_G: usize = 20 * 1024;

/// The kernels (CUDA) unless AVA_F2_K=0; the candle-op twins otherwise (and always on the CPU). Read once.
pub fn kernels_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("AVA_F2_K").is_ok_and(|v| v == "0"))
}

fn use_k(t: &Tensor) -> bool {
    t.device().is_cuda() && kernels_on()
}

// ---------------------------------------------------------------- routing (fwin::step_ok, Model::win_step)

/// fwin::step_ok's hook: the Arch with the four parts this step takes cleared (so the rest is checked as before).
pub fn strip(a: Arch) -> Arch {
    let mut a = a;
    a.v51.letters = false;
    a.v51.columns = 0;
    a.v51b.thalamus = false;
    a.v51b.cb_units = 0;
    a
}

/// The shapes the step's kernels need for these parts (Arch::validate has checked the rest).
pub fn shapes_ok(cfg: &Config) -> bool {
    let a = cfg.arch;
    let k = a.v51.columns;
    let cols = k == 0 || (k >= 2 && cfg.d % k == 0 && cfg.hidden() % k == 0 && cfg.kwta_k() % k == 0);
    let g = a.v51b.cb_units;
    let cb = g == 0 || (g <= CB_MAX_G && g % crate::v51b::CB_SPARSITY == 0 && crate::v51b::CB_FAN_IN == 4);
    let letters = !a.v51.letters || cfg.vocab >= 256;
    cols && cb && letters
}

/// v51 cbfix (runs/v51/cbfix.md): the largest G f2_cb_code holds on `dev`: CB_MAX_G, or less where the device's opt-in
/// shared memory per block cannot hold 4 G bytes plus the kernel's static bytes. CB_MAX_G on the CPU; 0 if the device
/// cannot be asked (CB then keeps autograd).
pub fn cb_max_g(dev: &Device) -> usize {
    let Device::Cuda(cd) = dev else { return CB_MAX_G };
    let fit = || -> Result<usize> {
        let func = cd.get_or_load_custom_func("f2_cb_code", MODULE, PTX)?;
        let stat = func.shared_size_bytes().w()?.max(0) as usize;
        let (_, optin) = crate::lif::smem_limits(cd)?;
        Ok(optin.saturating_sub(stat) / 4)
    };
    fit().map(|g| g.min(CB_MAX_G)).unwrap_or(0)
}

/// v51 cbfix: shapes_ok's CB bound on the device the step runs on (Model::win_step, fwin::would_take_on): G within
/// cb_max_g(dev), so a card with less shared memory falls back to autograd instead of failing at the launch.
pub fn device_ok(cfg: &Config, dev: &Device) -> bool {
    let g = cfg.arch.v51b.cb_units;
    g == 0 || g <= cb_max_g(dev)
}

/// The model's talk-time state the step does not carry: CB's online table, a plastic session or the chat app's part
/// signals (autograd then).
pub fn model_ok(m: &Model) -> bool {
    !m.cb_online_on() && !crate::eprop::active(m) && !m.v51b().is_some_and(|e| e.sig_on.get())
}

// ---------------------------------------------------------------- kernels: launch (f32, bf16 and u32 tensors)

pub(crate) enum K<'a> {
    T(&'a Tensor),
    I(i32),
}

enum V<'v> {
    F(CudaView<'v, f32>),
    B(CudaView<'v, half::bf16>),
    U(CudaView<'v, u32>),
}

/// Launches `name` with `args` in order (fwin_ops::launch with u32 tensors too); `smem` dynamic shared bytes, past
/// 48 KB (with the kernel's static bytes) asked for by the function's opt-in attribute (lif::smem_ready).
pub(crate) fn launch(name: &str, args: &[K], grid: usize, block: usize, smem: usize) -> Result<()> {
    let tens: Vec<&Tensor> = args.iter().filter_map(|a| if let K::T(t) = a { Some(*t) } else { None }).collect();
    let guards: Vec<_> = tens.iter().map(|t| t.storage_and_layout()).collect();
    let mut dev = None;
    let mut views = Vec::with_capacity(guards.len());
    for (g, l) in guards.iter() {
        let Storage::Cuda(s) = &**g else { candle_core::bail!("{name}: CUDA only") };
        if dev.is_none() {
            dev = Some(s.device().clone());
        }
        let Some((a, b)) = l.contiguous_offsets() else { candle_core::bail!("{name}: contiguous tensors only") };
        views.push(match s.dtype() {
            DType::F32 => V::F(s.as_cuda_slice::<f32>()?.slice(a..b)),
            DType::BF16 => V::B(s.as_cuda_slice::<half::bf16>()?.slice(a..b)),
            DType::U32 => V::U(s.as_cuda_slice::<u32>()?.slice(a..b)),
            d => candle_core::bail!("{name}: f32, bf16 or u32 tensors only, got {d:?}"),
        });
    }
    let dev = dev.ok_or_else(|| candle_core::Error::Msg(format!("{name}: no tensor argument")))?;
    let func = dev.get_or_load_custom_func(name, MODULE, PTX)?;
    // v51 cbfix: the opt-in as before (past 48 KB with the kernel's static bytes), and past the device's opt-in limit a
    // plain error instead of the driver's "invalid argument"
    crate::lif::smem_ready(&dev, &func, name, smem)?;
    let mut lb = func.builder();
    let mut vi = views.iter();
    for a in args {
        match a {
            K::T(_) => match vi.next().expect("one view per tensor") {
                V::F(x) => {
                    lb.arg(x);
                }
                V::B(x) => {
                    lb.arg(x);
                }
                V::U(x) => {
                    lb.arg(x);
                }
            },
            K::I(x) => {
                lb.arg(x);
            }
        }
    }
    let cfg = LaunchConfig { grid_dim: (grid.max(1) as u32, 1, 1), block_dim: (block as u32, 1, 1), shared_mem_bytes: smem as u32 };
    unsafe { lb.launch(cfg) }.w()?;
    Ok(())
}

fn empty(like: &Tensor, shape: impl Into<candle_core::Shape>) -> Result<Tensor> {
    crate::fwin_ops::empty(like, shape, DType::F32)
}

// ---------------------------------------------------------------- R: the letter vectors

/// R's tables (v51's Letters): the byte, place and end embeddings E [256, 64], P [16, 64], Q [16, 64] and the
/// vocabulary's rows (byte ids, end places [V, 16] u32, mask [V, 16], 1/sqrt(len) [V, 1]).
pub struct LetTabs<'a> {
    pub byte: &'a Tensor,
    pub pos: &'a Tensor,
    pub end: &'a Tensor,
    pub ids: &'a Tensor,
    pub eidx: &'a Tensor,
    pub mask: &'a Tensor,
    pub inv: &'a Tensor,
}

fn let_check(t: &LetTabs) -> Result<usize> {
    let (v, l) = t.ids.dims2()?;
    let dl = crate::v51::LETTER_DIM;
    if l != crate::v51::LETTER_MAX || t.byte.dims() != [256, dl] || t.pos.dims() != [l, dl] || t.end.dims() != [l, dl] || t.eidx.dims() != [v, l] || t.mask.dims() != [v, l] || t.inv.elem_count() != v {
        candle_core::bail!("letters: tables of {:?} {:?} {:?} rows {:?} {:?} {:?} {:?}", t.byte.dims(), t.pos.dims(), t.end.dims(), t.ids.dims(), t.eidx.dims(), t.mask.dims(), t.inv.dims());
    }
    Ok(v)
}

/// The letter vectors [V, 64] of the whole vocabulary: v51::letter_core's numbers (f2_let_fwd on CUDA, letter_core
/// itself otherwise).
pub fn let_fwd(t: &LetTabs) -> Result<Tensor> {
    if use_k(t.byte) && t.byte.dtype() == DType::F32 { let_fwd_k(t) } else { let_fwd_ops(t) }
}

/// let_fwd's kernel (CUDA, f32), whatever AVA_F2_K says.
pub fn let_fwd_k(t: &LetTabs) -> Result<Tensor> {
    let v = let_check(t)?;
    if !t.byte.device().is_cuda() || t.byte.dtype() != DType::F32 {
        candle_core::bail!("f2_let_fwd: CUDA f32 tables only");
    }
    let out = empty(t.byte, (v, crate::v51::LETTER_DIM))?;
    let (byte, pos, end, ids, eidx, mask, inv) = (t.byte.contiguous()?, t.pos.contiguous()?, t.end.contiguous()?, t.ids.contiguous()?, t.eidx.contiguous()?, t.mask.contiguous()?, t.inv.contiguous()?);
    let n = v * crate::v51::LETTER_DIM;
    let args = [K::T(&byte), K::T(&pos), K::T(&end), K::T(&ids), K::T(&eidx), K::T(&mask), K::T(&inv), K::T(&out), K::I(v as i32)];
    launch("f2_let_fwd", &args, n.div_ceil(256), 256, 0)?;
    Ok(out)
}

/// let_fwd in candle ops: v51::letter_core.
pub fn let_fwd_ops(t: &LetTabs) -> Result<Tensor> {
    crate::v51::letter_core(t.byte, t.pos, t.end, t.ids, t.eidx, t.mask, t.inv)
}

/// The backward of let_fwd for dL [V, 64]: (dE [256, 64], dP [16, 64], dQ [16, 64]). Kernels on CUDA (the row lists of
/// `LetCsr`, sums in a fixed order), let_bwd_ops otherwise.
pub fn let_bwd(t: &LetTabs, dl: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    if use_k(dl) && dl.dtype() == DType::F32 { let_bwd_k(t, dl) } else { let_bwd_ops(t, dl) }
}

/// let_bwd's kernels (CUDA, f32), whatever AVA_F2_K says.
pub fn let_bwd_k(t: &LetTabs, dl: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let v = let_check(t)?;
    if dl.dims() != [v, crate::v51::LETTER_DIM] {
        candle_core::bail!("let_bwd: dL [{v}, 64] wanted, got {:?}", dl.dims());
    }
    if !dl.device().is_cuda() || dl.dtype() != DType::F32 {
        candle_core::bail!("f2_let_bwd: CUDA f32 only");
    }
    let csr = let_csr(t)?;
    let (byte, pos, end, ids, eidx, inv, dl) = (t.byte.contiguous()?, t.pos.contiguous()?, t.end.contiguous()?, t.ids.contiguous()?, t.eidx.contiguous()?, t.inv.contiguous()?, dl.contiguous()?);
    let dl_ = crate::v51::LETTER_DIM;
    let part = empty(&dl, (csr.nch.max(1), dl_))?;
    let args = [K::T(&byte), K::T(&pos), K::T(&end), K::T(&ids), K::T(&eidx), K::T(&inv), K::T(&dl), K::T(&csr.pos), K::T(&csr.cbeg), K::T(&part), K::I(csr.nch as i32)];
    if csr.nch > 0 {
        launch("f2_let_bwd_part", &args, csr.nch, dl_, 0)?;
    }
    let out = empty(&dl, (LET_ROWS, dl_))?;
    launch("f2_let_bwd_red", &[K::T(&part), K::T(&csr.rbeg), K::T(&out), K::I(LET_ROWS as i32)], LET_ROWS, dl_, 0)?;
    let lm = crate::v51::LETTER_MAX;
    Ok((out.narrow(0, 0, 256)?, out.narrow(0, 256, lm)?, out.narrow(0, 256 + lm, lm)?))
}

/// let_bwd in candle ops (autograd's maths: dz = ((dL inv) mask)(1 - s^2), then the embeddings' index_add and the
/// places' sum over the vocabulary). Any float dtype.
pub fn let_bwd_ops(t: &LetTabs, dl: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let (v, l) = t.ids.dims2()?;
    let dt = t.byte.dtype();
    let dd = t.byte.dim(1)?;
    let e = t.byte.embedding(&t.ids.flatten_all()?)?.reshape((v, l, dd))?;
    let q = t.end.embedding(&t.eidx.flatten_all()?)?.reshape((v, l, dd))?;
    let s = e.add(&q)?.broadcast_add(&t.pos.unsqueeze(0)?)?.tanh()?;
    let g = dl.broadcast_mul(&t.inv.to_dtype(dt)?)?.unsqueeze(1)?.broadcast_mul(&t.mask.to_dtype(dt)?.unsqueeze(2)?)?;
    let dz = g.mul(&s.sqr()?.affine(-1.0, 1.0)?)?;
    let flat = dz.reshape((v * l, dd))?;
    let de = Tensor::zeros((256, dd), dt, dl.device())?.index_add(&t.ids.flatten_all()?, &flat, 0)?;
    let dq = Tensor::zeros((l, dd), dt, dl.device())?.index_add(&t.eidx.flatten_all()?, &flat, 0)?;
    let dp = dz.sum(0)?;
    Ok((de, dp, dq))
}

/// R's backward row lists: for each table row (byte b; place j; place from the end e), the flat entries p = v L + j
/// that use it (mask 1), in order, cut into chunks of at most LET_CHUNK entries (each chunk inside one row).
pub(crate) struct LetCsr {
    pos: Tensor,
    cbeg: Tensor,
    rbeg: Tensor,
    nch: usize,
}

thread_local! {
    static CSR: RefCell<HashMap<candle_core::TensorId, Arc<LetCsr>>> = RefCell::new(HashMap::new());
}

fn let_csr(t: &LetTabs) -> Result<Arc<LetCsr>> {
    if let Some(c) = CSR.with(|m| m.borrow().get(&t.ids.id()).cloned()) {
        return Ok(c);
    }
    let lm = crate::v51::LETTER_MAX;
    let ids: Vec<u32> = t.ids.flatten_all()?.to_vec1()?;
    let eidx: Vec<u32> = t.eidx.flatten_all()?.to_vec1()?;
    let mask: Vec<f32> = t.mask.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); LET_ROWS];
    for p in 0..ids.len() {
        if mask[p] != 0.0 {
            rows[ids[p] as usize].push(p as u32);
            rows[256 + p % lm].push(p as u32);
            rows[256 + lm + eidx[p] as usize].push(p as u32);
        }
    }
    let (mut pos, mut cbeg, mut rbeg) = (Vec::new(), vec![0u32], vec![0u32]);
    for r in &rows {
        for ch in r.chunks(LET_CHUNK) {
            pos.extend_from_slice(ch);
            cbeg.push(pos.len() as u32);
        }
        rbeg.push((cbeg.len() - 1) as u32);
    }
    let dev = t.ids.device();
    let nch = cbeg.len() - 1;
    let np = pos.len().max(1);
    pos.resize(np, 0);
    let c = Arc::new(LetCsr { pos: Tensor::from_vec(pos, np, dev)?, cbeg: Tensor::from_vec(cbeg, nch + 1, dev)?, rbeg: Tensor::from_vec(rbeg, LET_ROWS + 1, dev)?, nch });
    CSR.with(|m| m.borrow_mut().insert(t.ids.id(), c.clone()));
    Ok(c)
}

// ---------------------------------------------------------------- W: the column-wired MLP matrices

thread_local! {
    static MASKS: RefCell<HashMap<(usize, usize, usize, String), Tensor>> = RefCell::new(HashMap::new());
}

/// The K diagonal blocks of a [rows, cols] matrix as a 0/1 mask (made once per shape and device).
pub fn col_mask(rows: usize, cols: usize, k: usize, dt: DType, dev: &Device) -> Result<Tensor> {
    let key = (rows, cols, k, format!("{:?} {dt:?}", dev.location()));
    if let Some(m) = MASKS.with(|c| c.borrow().get(&key).cloned()) {
        return Ok(m);
    }
    let (bi, bo) = (rows / k, cols / k);
    let v: Vec<f32> = (0..rows * cols).map(|e| if (e / cols) / bi == (e % cols) / bo { 1.0 } else { 0.0 }).collect();
    let m = Tensor::from_vec(v, (rows, cols), dev)?.to_dtype(dt)?;
    MASKS.with(|c| c.borrow_mut().insert(key, m.clone()));
    Ok(m)
}

/// W' = w ⊙ M + a c: v51::col_mm(h, w, a, c, k) = h W' (w [din, dout] whose K diagonal blocks are used, a [din, r],
/// c [r, dout]).
pub fn col_fold(w: &Tensor, a: &Tensor, c: &Tensor, k: usize) -> Result<Tensor> {
    let (di, dout) = w.dims2()?;
    if di % k != 0 || dout % k != 0 {
        candle_core::bail!("col_fold: [{di}, {dout}] into {k} columns");
    }
    w.mul(&col_mask(di, dout, k, w.dtype(), w.device())?)?.add(&a.matmul(c)?)
}

/// col_fold's backward for dL/dW' `dwe`: (dL/dw = dwe ⊙ M, dL/da = dwe c^T, dL/dc = a^T dwe).
pub fn col_unfold(dwe: &Tensor, a: &Tensor, c: &Tensor, k: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let (di, dout) = dwe.dims2()?;
    let dw = dwe.mul(&col_mask(di, dout, k, dwe.dtype(), dwe.device())?)?;
    Ok((dw, dwe.matmul(&c.t()?)?, a.t()?.matmul(dwe)?))
}

// ---------------------------------------------------------------- TH: the relay

/// v51b::th_gates with the relay's state f = tanh(acc / sqrt(reads)) kept: (f [n, K], gates [n, m]), the same ops.
pub fn th_gates_keep(acc: &Tensor, reads: usize, g: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
    let f = acc.affine(1.0 / (reads as f64).sqrt(), 0.0)?.tanh()?;
    let gates = sigmoid(&f.matmul(g)?.broadcast_add(c)?)?.affine(2.0, 0.0)?;
    Ok((f, gates))
}

/// The backward of th_gates_keep for dL/d(gates) `dg` [n, m]: (dL/dacc [n, K], dL/dG [K, m], dL/dc [m]).
pub fn th_gates_bwd(f: &Tensor, gates: &Tensor, g: &Tensor, reads: usize, dg: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    // d(2 sigmoid(z))/dz = s (1 - s/2), s the gate
    let dz = dg.mul(gates)?.mul(&gates.affine(-0.5, 1.0)?)?;
    let dgw = f.t()?.matmul(&dz)?;
    let dc = dz.sum(0)?;
    let df = dz.matmul(&g.t()?)?;
    let dacc = df.mul(&f.sqr()?.affine(-1.0, 1.0)?)?.affine(1.0 / (reads as f64).sqrt(), 0.0)?;
    Ok((dacc, dgw, dc))
}

/// rmsnorm's backward (no gain gradient): fused::rms_bwd_raw on CUDA, fwin_nm::rms_bwd_ops otherwise.
fn rms_dx(x: &Tensor, ones: &Tensor, dy: &Tensor) -> Result<Tensor> {
    if x.device().is_cuda() {
        Ok(rms_bwd_raw(x, ones, dy, NORM_EPS)?.0)
    } else {
        Ok(crate::fwin_nm::rms_bwd_ops(x, ones, dy, NORM_EPS)?.0)
    }
}

/// One relay read's backward: x [n, d] read as rms(x) R (xn = rms(x), R [d, K]), the summed gradient `da` [n, K] of the
/// relay sums that hold it: (dL/dx [n, d], dL/dR [d, K]).
pub fn th_read_bwd(x: &Tensor, xn: &Tensor, r: &Tensor, da: &Tensor, ones: &Tensor) -> Result<(Tensor, Tensor)> {
    let dr = xn.t()?.matmul(da)?;
    let dx = rms_dx(x, ones, &da.matmul(&r.t()?)?)?;
    Ok((dx, dr))
}

// ---------------------------------------------------------------- CB: the granule code

/// CB's fixed wiring for 2d inputs and g units (v51b::cb_wiring, rebuilt from the seed as v51b::build does): (idx [g·4]
/// u32, sgn [g, 4]) on `dev`, made once per shape and device.
pub fn cb_wiring(two_d: usize, g: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    thread_local! {
        static W: RefCell<HashMap<(usize, usize, String), (Tensor, Tensor)>> = RefCell::new(HashMap::new());
    }
    let key = (two_d, g, format!("{:?}", dev.location()));
    if let Some(w) = W.with(|c| c.borrow().get(&key).cloned()) {
        return Ok(w);
    }
    let f = crate::v51b::CB_FAN_IN;
    let (idx, sgn) = crate::v51b::cb_wiring(two_d, g, f, crate::v51b::CB_SEED);
    let w = (Tensor::from_vec(idx, g * f, dev)?, Tensor::from_vec(sgn, (g, f), dev)?);
    W.with(|c| c.borrow_mut().insert(key, w.clone()));
    Ok(w)
}

/// CB's mossy-fibre input [n, 2d]: her final normed state and the words' rms-normed embeddings (v51b::cb_code's first
/// line, the same ops).
pub fn cb_input(xf: &Tensor, emb: &Tensor, ones: &Tensor) -> Result<Tensor> {
    Tensor::cat(&[xf.detach(), rmsnorm(&emb.detach(), ones)?], 1)
}

/// The granule code [n, G] of the input u [n, 2d] (v51b::cb_code from its input on): f2_cb_code on CUDA, cb_code_ops
/// otherwise.
pub fn cb_code(u: &Tensor, idx: &Tensor, sgn: &Tensor, k: usize) -> Result<Tensor> {
    if use_k(u) && u.dtype() == DType::F32 { cb_code_k(u, idx, sgn, k) } else { cb_code_ops(u, idx, sgn, k) }
}

/// cb_code's kernel (CUDA, f32), whatever AVA_F2_K says.
pub fn cb_code_k(u: &Tensor, idx: &Tensor, sgn: &Tensor, k: usize) -> Result<Tensor> {
    let (n, ud) = u.dims2()?;
    let (g, f) = sgn.dims2()?;
    if f != 4 || idx.elem_count() != g * f || g > CB_MAX_G || k == 0 || k >= g {
        candle_core::bail!("cb_code: {g} units of {f} inputs, k {k}");
    }
    if !u.device().is_cuda() || u.dtype() != DType::F32 {
        candle_core::bail!("f2_cb_code: CUDA f32 only");
    }
    let (u, idx, sgn) = (u.contiguous()?, idx.contiguous()?, sgn.contiguous()?);
    let out = empty(&u, (n, g))?;
    let block = g.min(1024).next_power_of_two();
    launch("f2_cb_code", &[K::T(&u), K::T(&idx), K::T(&sgn), K::T(&out), K::I(n as i32), K::I(ud as i32), K::I(g as i32), K::I(k as i32)], n, block, g * 4)?;
    Ok(out)
}

/// cb_code in candle ops: v51b::cb_code's ops after its input.
pub fn cb_code_ops(u: &Tensor, idx: &Tensor, sgn: &Tensor, k: usize) -> Result<Tensor> {
    let n = u.dim(0)?;
    let (g, f) = sgn.dims2()?;
    let h = u.index_select(idx, 1)?.reshape((n, g, f))?.broadcast_mul(sgn)?.sum(2)?;
    let a = h.mul(&crate::model::kwta_mask(&h, k)?)?.relu()?;
    let len = a.sqr()?.sum_keepdim(1)?.sqrt()?.affine(1.0, 1e-6)?;
    a.broadcast_div(&len)
}

// ---------------------------------------------------------------- the step's hooks (src/model/fwin.rs)

struct LetRun {
    /// the letter vectors [V, 64] and Wo (detached)
    lv: Tensor,
    wo: Tensor,
}

struct ThBlock {
    /// the relay's state, the gates [n, 2] (residual, memory), its reads so far
    f: Tensor,
    gates: Tensor,
    reads: usize,
    g_res: Tensor,
    g_mem: Tensor,
    /// the memory read before its gate [n, d], the block's update x_raw - x_in [n, d] and g_res - 1 [n, 1]
    y: Option<Tensor>,
    delta: Option<Tensor>,
    gm1: Option<Tensor>,
    /// backward: dL/d(memory gate) [n, 1], dL/d(residual gate) [n, 1] and -dL/d(update) for the block's input
    dgm: Option<Tensor>,
    dgr: Option<Tensor>,
    neg: Option<Tensor>,
}

struct ThRun {
    ones: Tensor,
    /// R [reads, d, K], G [L, K, 2], c [L, 2] (detached)
    read: Tensor,
    gate: Tensor,
    bias: Tensor,
    acc: Option<Tensor>,
    reads: usize,
    /// rms(x_j) of every read j
    xn: Vec<Tensor>,
    blocks: Vec<ThBlock>,
    /// backward: the relay sums' gradient over the blocks walked back so far, and the weights' gradients
    dacc: Option<Tensor>,
    dread: Vec<Option<Tensor>>,
    dgate: Vec<Option<Tensor>>,
    dbias: Vec<Option<Tensor>>,
}

struct CbRun {
    /// the code (as a matmul operand) and W (detached, as an operand)
    c: Tensor,
}

/// One fused step's state for R, W, TH and CB (all None without them).
pub(crate) struct F2<'a> {
    m: &'a Model,
    bf: bool,
    idf: Tensor,
    letters: Option<LetRun>,
    cols: usize,
    th: Option<ThRun>,
    cb: Option<CbRun>,
}

fn opt_add(slot: &mut Option<Tensor>, t: Tensor) -> Result<()> {
    crate::fwin_nm::acc(slot, t)
}

impl<'a> F2<'a> {
    pub(crate) fn new(m: &'a Model, idf: &Tensor, mm: &Mm) -> Result<F2<'a>> {
        let ar = m.cfg.arch;
        let th = match (ar.v51b.thalamus, m.v51b().and_then(|e| e.th.as_ref())) {
            (false, _) => None,
            (true, None) => candle_core::bail!("fused step: --thalamus without the relay"),
            (true, Some(t)) => {
                let nr = t.read.dim(0)?;
                Some(ThRun {
                    ones: Tensor::ones(m.cfg.d, DType::F32, m.tok.device())?,
                    read: t.read.detach(),
                    gate: t.gate.detach(),
                    bias: t.bias.detach(),
                    acc: None,
                    reads: 0,
                    xn: Vec::with_capacity(nr),
                    blocks: Vec::with_capacity(m.blocks.len()),
                    dacc: None,
                    dread: vec![None; nr],
                    dgate: vec![None; m.blocks.len()],
                    dbias: vec![None; m.blocks.len()],
                })
            }
        };
        if ar.v51.letters && v51x(m).and_then(|v| v.letters.as_ref()).is_none() {
            candle_core::bail!("fused step: --letters without its tables");
        }
        if ar.v51b.cb_units > 0 && m.v51b().and_then(|e| e.cb.as_ref()).is_none() {
            candle_core::bail!("fused step: --cerebellum-head without its read-out");
        }
        Ok(F2 { m, bf: mm.bf16, idf: idf.clone(), letters: None, cols: ar.v51.columns, th, cb: None })
    }

    fn tabs(&self) -> Option<LetTabs<'a>> {
        let lt = v51x(self.m)?.letters.as_ref()?;
        Some(LetTabs { byte: &lt.byte, pos: &lt.pos, end: &lt.end, ids: &lt.ids, eidx: &lt.eidx, mask: &lt.mask, inv: &lt.inv })
    }

    // ---- forward

    /// R: the input x [N, d] plus the letter vectors of the tokens (v51::embed's sum; x itself without --letters).
    pub(crate) fn embed(&mut self, x: Tensor) -> Result<Tensor> {
        let Some(t) = self.tabs() else { return Ok(x) };
        let t = LetTabs { byte: &t.byte.detach(), pos: &t.pos.detach(), end: &t.end.detach(), ..t };
        let lv = let_fwd(&t)?;
        let wo = v51x(self.m).and_then(|v| v.letters.as_ref()).expect("checked").wo.detach();
        let add = lv.matmul(&wo)?.embedding(&self.idf)?;
        self.letters = Some(LetRun { lv, wo });
        x.add(&add)
    }

    /// TH: the relay reads the blocks' input x [N, d].
    pub(crate) fn relay_begin(&mut self, x: &Tensor) -> Result<()> {
        let Some(th) = self.th.as_mut() else { return Ok(()) };
        let xn = norm(x, &th.ones)?; // v51b::th_read's norm
        th.acc = Some(xn.matmul(&th.read.get(0)?)?);
        th.reads = 1;
        th.xn.push(xn);
        Ok(())
    }

    /// TH: block i starts: its two gates from the relay's reads so far.
    pub(crate) fn block_begin(&mut self, i: usize) -> Result<()> {
        let Some(th) = self.th.as_mut() else { return Ok(()) };
        let acc = th.acc.as_ref().expect("relay_begin first");
        let (f, gates) = th_gates_keep(acc, th.reads, &th.gate.get(i)?, &th.bias.get(i)?)?;
        let (g_res, g_mem) = (gates.narrow(1, 0, 1)?.contiguous()?, gates.narrow(1, 1, 1)?.contiguous()?);
        th.blocks.push(ThBlock { f, gates, reads: th.reads, g_res, g_mem, y: None, delta: None, gm1: None, dgm: None, dgr: None, neg: None });
        Ok(())
    }

    /// W: block i's first MLP matrix as used (folded with --columns; w1 itself, detached, otherwise).
    pub(crate) fn w1(&self, i: usize, w1: &Tensor) -> Result<Tensor> {
        match self.cols_of(i) {
            Some([a1, b1, _, _]) => col_fold(&w1.detach(), &a1.detach(), &b1.detach(), self.cols),
            None => Ok(w1.detach()),
        }
    }

    /// W: the second.
    pub(crate) fn w2(&self, i: usize, w2: &Tensor) -> Result<Tensor> {
        match self.cols_of(i) {
            Some([_, _, a2, b2]) => col_fold(&w2.detach(), &a2.detach(), &b2.detach(), self.cols),
            None => Ok(w2.detach()),
        }
    }

    fn cols_of(&self, i: usize) -> Option<&'a [Tensor; 4]> {
        if self.cols == 0 {
            return None;
        }
        v51x(self.m)?.blocks.get(i)?.cols.as_ref()
    }

    /// TH: block i's memory read y [N, d] (after its output projection), times the memory gate.
    pub(crate) fn mem_out(&mut self, i: usize, y: Tensor) -> Result<Tensor> {
        let Some(th) = self.th.as_mut() else { return Ok(y) };
        let tb = &mut th.blocks[i];
        let out = y.broadcast_mul(&tb.g_mem)?;
        tb.y = Some(y);
        Ok(out)
    }

    /// W: the MLP's drive u [N, hidden] and kWTA's k, per column with --columns (u as [N K, hidden / K], k / K).
    pub(crate) fn cols_in(&self, u: Tensor, k: usize) -> Result<(Tensor, usize)> {
        if self.cols == 0 {
            return Ok((u, k));
        }
        let (n, h) = u.dims2()?;
        Ok((u.reshape((n * self.cols, h / self.cols))?, k / self.cols))
    }

    /// W: kWTA's outputs back as rows of the hidden width.
    pub(crate) fn cols_out(&self, a_r: Option<Tensor>, a_m: Tensor, n: usize) -> Result<(Option<Tensor>, Tensor)> {
        if self.cols == 0 {
            return Ok((a_r, a_m));
        }
        let back = |t: Tensor| -> Result<Tensor> {
            let h = t.elem_count() / n;
            t.reshape((n, h))
        };
        Ok((a_r.map(back).transpose()?, back(a_m)?))
    }

    /// TH: block i's last residual add x_out = xa + xb: (x_m2, mlp) itself without --thalamus; with it (x_raw, (x_raw -
    /// x_in)(g_res - 1)), x_raw = x_m2 + mlp, so xa + xb is v51b::th_res(x_in, x_raw, g_res).
    pub(crate) fn res_in(&mut self, i: usize, x_in: &Tensor, x_m2: Tensor, mlp: Tensor) -> Result<(Tensor, Tensor)> {
        let Some(th) = self.th.as_mut() else { return Ok((x_m2, mlp)) };
        let tb = &mut th.blocks[i];
        let x_raw = x_m2.add(&mlp)?;
        let delta = x_raw.sub(x_in)?;
        let gm1 = tb.g_res.affine(1.0, -1.0)?;
        let xb = delta.broadcast_mul(&gm1)?;
        tb.delta = Some(delta);
        tb.gm1 = Some(gm1);
        Ok((x_raw, xb))
    }

    /// TH: block i's output x [N, d] is final (after its dense mix): the relay reads it if a later gate needs it.
    pub(crate) fn relay_after(&mut self, i: usize, x: &Tensor) -> Result<()> {
        let Some(th) = self.th.as_mut() else { return Ok(()) };
        let nr = th.read.dim(0)?;
        if i + 1 < nr {
            let xn = norm(x, &th.ones)?;
            let r = xn.matmul(&th.read.get(i + 1)?)?;
            th.acc = Some(th.acc.take().expect("relay_begin first").add(&r)?);
            th.reads += 1;
            th.xn.push(xn);
        }
        Ok(())
    }

    /// CB: the head's input: her final normed state `hf` (the operand mm.rms gave) plus the cerebellum's correction,
    /// as a matmul operand (hf itself without --cerebellum-head). `x` the stream, `gf` the final norm's gain.
    pub(crate) fn head_in(&mut self, x: &Tensor, gf: &Tensor, hf: Tensor) -> Result<Tensor> {
        let Some(cbw) = self.m.v51b().and_then(|e| e.cb.as_ref()) else { return Ok(hf) };
        let d = self.m.cfg.d;
        let w = cbw.w.detach();
        let g = w.dim(0)?;
        let xf = if self.bf { rms_fwd_raw(x, gf, NORM_EPS)? } else { hf };
        let ones = Tensor::ones(d, DType::F32, x.device())?;
        let emb = self.m.tok.detach().embedding(&self.idf)?;
        let (idx, sgn) = cb_wiring(2 * d, g, x.device())?;
        let c = cb_code(&cb_input(&xf, &emb, &ones)?, &idx, &sgn, g / crate::v51b::CB_SPARSITY)?;
        let mm = Mm { bf16: self.bf };
        let c = mm.op(&c)?;
        let corr = mm.mm(&c, false, &mm.op(&w)?, false)?;
        self.cb = Some(CbRun { c });
        mm.op(&xf.add(&corr)?)
    }

    // ---- backward

    /// CB: dL/dW = c^T dL/dxf for the head input's gradient `dhf` [N, d] (which passes to xf unchanged).
    pub(crate) fn head_bwd(&self, dhf: &Tensor, grads: &mut GradStore) -> Result<()> {
        let Some(cb) = &self.cb else { return Ok(()) };
        let w = &self.m.v51b().and_then(|e| e.cb.as_ref()).expect("checked").w;
        let mm = Mm { bf16: self.bf };
        grads.insert(w, mm.mm(&cb.c, true, &mm.op(dhf)?, false)?);
        Ok(())
    }

    /// TH: the relay's gradient for its read j of the stream x_j (block j's input; j > 0: block j - 1's final output),
    /// called once blocks j..L-1 have been walked back (block_back(j) included: every gate that holds read j), to be
    /// added to dL/dx_j (None without --thalamus or when j is not read).
    pub(crate) fn read_grad(&mut self, j: usize, x_j: &Tensor) -> Result<Option<Tensor>> {
        let Some(th) = self.th.as_mut() else { return Ok(None) };
        if j >= th.xn.len() {
            return Ok(None);
        }
        let Some(da) = th.dacc.as_ref() else { return Ok(None) };
        let (dx, dr) = th_read_bwd(x_j, &th.xn[j], &th.read.get(j)?, da, &th.ones)?;
        th.dread[j] = Some(dr);
        Ok(Some(dx))
    }

    /// TH: block i's residual rule, backward, for dL/d(the gated output) `dx`: dL/d(the update) (g_res - 1) dx, to be
    /// added to dx (giving dL/dx_raw); its negative is kept for the block's input and the gate's gradient for the relay.
    pub(crate) fn res_back(&mut self, i: usize, dx: &Tensor) -> Result<Option<Tensor>> {
        let Some(th) = self.th.as_mut() else { return Ok(None) };
        let tb = &mut th.blocks[i];
        let (delta, gm1) = (tb.delta.as_ref().expect("res_in ran"), tb.gm1.as_ref().expect("res_in ran"));
        let (dd, dgr) = crate::fwin_nm::rows_bwd(dx, delta, gm1, false)?;
        tb.neg = Some(dd.neg()?);
        tb.dgr = Some(dgr);
        Ok(Some(dd))
    }

    /// TH: block i's memory gate, backward: dL/d(the read before the gate) as the matmul operand for dL/dx_m2 `dx`
    /// (f32); `dx_m` (dx as the operand) itself without --thalamus.
    pub(crate) fn mem_back(&mut self, i: usize, dx: &Tensor, dx_m: &Tensor) -> Result<Tensor> {
        let bf = self.bf;
        self.mem_back_as(i, dx, dx_m, bf)
    }

    /// mem_back with the result's type chosen: the bf16 operand when `bf`, else f32 (runs/v51/merge_f.md: with v51
    /// fast1's P the dendrites' share of the read's gradient is added in f32 before the operand is made).
    pub(crate) fn mem_back_as(&mut self, i: usize, dx: &Tensor, dx_m: &Tensor, bf: bool) -> Result<Tensor> {
        let Some(th) = self.th.as_mut() else { return Ok(dx_m.clone()) };
        let tb = &mut th.blocks[i];
        let (dy, dgm) = crate::fwin_nm::rows_bwd(dx, tb.y.as_ref().expect("mem_out ran"), &tb.g_mem, bf)?;
        tb.dgm = Some(dgm);
        Ok(dy)
    }

    /// TH: block i is walked back: its gates' gradients go into the relay; returns what joins the block input's
    /// gradient (the residual rule's direct share).
    pub(crate) fn block_back(&mut self, i: usize) -> Result<Option<Tensor>> {
        let Some(th) = self.th.as_mut() else { return Ok(None) };
        let tb = &mut th.blocks[i];
        let dg = Tensor::cat(&[tb.dgr.take().expect("res_back ran"), tb.dgm.take().expect("mem_back ran")], 1)?;
        let (dacc, dgw, dc) = th_gates_bwd(&tb.f, &tb.gates, &th.gate.get(i)?, tb.reads, &dg)?;
        opt_add(&mut th.dacc, dacc)?;
        th.dgate[i] = Some(dgw);
        th.dbias[i] = Some(dc);
        Ok(tb.neg.take())
    }

    /// W: block i's MLP matrix gradients from dL/d(the matrix as used) (unfolded with --columns).
    pub(crate) fn put_w(&self, i: usize, second: bool, w: &Tensor, dwe: Tensor, grads: &mut GradStore) -> Result<()> {
        let Some(c) = self.cols_of(i) else {
            grads.insert(w, dwe);
            return Ok(());
        };
        let (a, b) = if second { (&c[2], &c[3]) } else { (&c[0], &c[1]) };
        let (dw, da, db) = col_unfold(&dwe, &a.detach(), &b.detach(), self.cols)?;
        grads.insert(w, dw);
        grads.insert(a, da);
        grads.insert(b, db);
        Ok(())
    }

    /// The end of the backward, `dx` = dL/d(the blocks' input) (the relay's read 0 included): R's and TH's weights.
    pub(crate) fn finish(&mut self, dx: &Tensor, grads: &mut GradStore) -> Result<()> {
        if let (Some(lr), Some(t)) = (self.letters.as_ref(), self.tabs()) {
            let lt = v51x(self.m).and_then(|v| v.letters.as_ref()).expect("checked");
            let (v, d) = (lr.lv.dim(0)?, self.m.cfg.d);
            let dlw = embed_bwd_add(&Tensor::zeros((v, d), DType::F32, dx.device())?, &self.idf, dx)?; // [V, d]
            grads.insert(&lt.wo, lr.lv.t()?.matmul(&dlw)?);
            let dl = dlw.matmul(&lr.wo.t()?)?;
            let t = LetTabs { byte: &t.byte.detach(), pos: &t.pos.detach(), end: &t.end.detach(), ..t };
            let (de, dp, dq) = let_bwd(&t, &dl)?;
            grads.insert(&lt.byte, de);
            grads.insert(&lt.pos, dp);
            grads.insert(&lt.end, dq);
        }
        if let Some(th) = self.th.as_mut() {
            let e = self.m.v51b().and_then(|e| e.th.as_ref()).expect("checked");
            let z = |t: &Tensor| t.get(0).and_then(|x| x.zeros_like());
            let reads: Vec<Tensor> = th.dread.iter_mut().map(|r| r.take().map_or_else(|| z(&th.read), Ok)).collect::<Result<_>>()?;
            let gates: Vec<Tensor> = th.dgate.iter_mut().map(|r| r.take().map_or_else(|| z(&th.gate), Ok)).collect::<Result<_>>()?;
            let bias: Vec<Tensor> = th.dbias.iter_mut().map(|r| r.take().map_or_else(|| z(&th.bias), Ok)).collect::<Result<_>>()?;
            grads.insert(&e.read, Tensor::stack(&reads, 0)?);
            grads.insert(&e.gate, Tensor::stack(&gates, 0)?);
            grads.insert(&e.bias, Tensor::stack(&bias, 0)?);
        }
        Ok(())
    }
}

/// The model's v51 parts when any v51 option is on.
fn v51x(m: &Model) -> Option<&crate::v51::V51Ext> {
    m.ext.as_ref().map(|e| &e.v51).filter(|v| v.on)
}

#[cfg(test)]
#[path = "fwin_f2_tests.rs"]
mod tests;
