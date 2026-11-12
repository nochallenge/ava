// TESTING / DESIGN ONLY — ava_gpu::fwin_nm: v51 fused N, --neuromod 3 (src/v50.rs) on the hand-written WIN training
// step (src/model/fwin.rs), so the 0.4b recipe with the brain chemicals trains at the fused step's speed instead of
// candle's autograd (runs/v51/fused_n_check.sh checks it on the GPU).
//
// The maths are the reference path's (arch.rs forward_arch / block_arch / mem_arch with src/v50.rs's hooks):
//   signals  s = 2 sigmoid(rms(x0, g) W + b) [N, 3], x0 = block 0's output (after its DenseFormer mix), per token
//   NE       the LIF's input current of blocks 1..L-1 times s_NE (after --lif-scale)
//   ACh      the memory's write strengths beta [B, H, T, G] times s_ACh, kept in [0, 2]
//   DA       the MLP's output times s_DA
// Forward: the same candle ops as the reference (so the same numbers for the same inputs), plus one kernel for ACh
// (the strengths live in the memory scan's input vg, between fwin_ops's prep and scan). Backward, no autograd:
//   NE, DA   y = x s per row: dx = dy s, ds = rowsum(dy x)                        kernel nm_rows_bwd(_b)
//   ACh      dbeta = dy m s, ds = sum over heads and columns of dy m beta, m the clamp's gradient (candle's: 1 inside
//            (0, 2), 1/2 on either bound, 0 outside)                              kernel nm_ach_bwd
//   nucleus  dz = ds s (1 - s/2); dW = rms(x0)^T dz, db = colsum(dz), d rms(x0) = dz W^T, then rmsnorm's backward
//            (fused::rms_bwd_raw on CUDA), added to the gradient of block 0's output
// Every kernel has a candle-op twin here (`*_ops`), used on the CPU (where the tests check them against autograd) and on
// CUDA when AVA_NM_K=0; the GPU tests check the kernels against the twins.
// Kept for the backward, per modulated block: the current before NE, the unscaled strengths, the MLP output before DA
// (3 N·d f32 at per-channel gates).
//
// Round 2's options (src/v51.rs) are not here: a keeper would join the same way (its forward op in loss_and_grads_win,
// its backward next to the part it scales), and step_ok lets it in only then.

use crate::fwin_ops::{self as wo, MemShape, A};
use crate::model::{norm, sigmoid, NORM_EPS};
use candle_core::{DType, Result, Tensor, D};

pub use crate::v50::{ACH, DA, NE};

/// The kernels (CUDA) unless AVA_NM_K=0; the candle-op twins otherwise (and always on the CPU). Read once.
pub fn kernels_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("AVA_NM_K").is_ok_and(|v| v == "0"))
}

fn use_k(t: &Tensor) -> bool {
    t.device().is_cuda() && kernels_on()
}

/// The nucleus's forward on block 0's output: the signals and what its backward needs.
pub struct Nucleus {
    /// 2·sigmoid(rms(x0, g)·W + b) [N, 3] f32: v50::nucleus_signals's ops, so the reference's numbers for the same x0
    pub sig: Tensor,
    /// each signal's column as its own contiguous [N, 1] (NE, ACH, DA)
    pub cols: Vec<Tensor>,
    /// rms(x0, g) [N, d]
    hn: Tensor,
}

/// The signals for block 0's output `x0` [N, d], with the nucleus's gain `g` [d], read-out `w` [d, k] and bias `b` [k].
pub fn nucleus_fwd(x0: &Tensor, g: &Tensor, w: &Tensor, b: &Tensor) -> Result<Nucleus> {
    let hn = norm(x0, g)?;
    let sig = sigmoid(&hn.matmul(w)?.broadcast_add(b)?)?.affine(2.0, 0.0)?;
    let cols = (0..sig.dim(1)?).map(|k| sig.narrow(1, k, 1)?.contiguous()).collect::<Result<Vec<_>>>()?;
    Ok(Nucleus { sig, cols, hn })
}

/// The nucleus's backward for dL/d(signals) `dsig` [N, k]: (dL/dx0 [N, d], dL/dg [d], dL/dW [d, k], dL/db [k]).
pub fn nucleus_bwd(nu: &Nucleus, x0: &Tensor, g: &Tensor, w: &Tensor, dsig: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    // d(2 sigmoid(z))/dz = 2 sigmoid (1 - sigmoid) = s (1 - s/2)
    let dz = dsig.mul(&nu.sig)?.mul(&nu.sig.affine(-0.5, 1.0)?)?;
    let dw = nu.hn.t()?.matmul(&dz)?;
    let db = dz.sum(0)?;
    let dhn = dz.matmul(&w.t()?)?;
    let (dx, dg) = rms_bwd(x0, g, &dhn)?;
    Ok((dx, dg, dw, db))
}

/// rmsnorm's backward (dx [N, d], dg [d]) for dy: fused::rms_bwd_raw on CUDA, rms_bwd_ops on the CPU.
pub fn rms_bwd(x: &Tensor, g: &Tensor, dy: &Tensor) -> Result<(Tensor, Tensor)> {
    if x.device().is_cuda() {
        crate::fused::rms_bwd_raw(x, g, dy, NORM_EPS)
    } else {
        rms_bwd_ops(x, g, dy, NORM_EPS)
    }
}

/// rmsnorm's backward in candle ops: y = x r g, r = (mean(x^2) + eps)^-1/2; with u = g dy,
/// dx = r (u - x r · mean(u x r)), dg = sum over rows of dy x r.
pub fn rms_bwd_ops(x: &Tensor, g: &Tensor, dy: &Tensor, eps: f64) -> Result<(Tensor, Tensor)> {
    let d = x.dim(D::Minus1)? as f64;
    let r = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.powf(-0.5)?;
    let xr = x.broadcast_mul(&r)?;
    let u = dy.broadcast_mul(g)?;
    let dot = u.mul(&xr)?.sum_keepdim(D::Minus1)?.affine(1.0 / d, 0.0)?;
    let dx = u.sub(&xr.broadcast_mul(&dot)?)?.broadcast_mul(&r)?;
    let dg = dy.mul(&xr)?.sum(0)?;
    Ok((dx, dg))
}

/// `slot` += `t` (the first one is taken as it is).
pub fn acc(slot: &mut Option<Tensor>, t: Tensor) -> Result<()> {
    *slot = Some(match slot.take() {
        Some(a) => a.add(&t)?,
        None => t,
    });
    Ok(())
}

/// dL/d(signals) [n, k] from the per-signal sums (a missing one is 0).
pub fn dsig_cat(parts: &[Option<Tensor>], n: usize, like: &Tensor) -> Result<Tensor> {
    let cols: Vec<Tensor> = parts
        .iter()
        .map(|p| match p {
            Some(t) => t.reshape((n, 1)),
            None => Tensor::zeros((n, 1), DType::F32, like.device()),
        })
        .collect::<Result<_>>()?;
    Tensor::cat(&cols, 1)
}

// ---------------------------------------------------------------- NE and DA: y = x s, one s per row

/// The backward of y = x ⊙ s with s [N, 1] one number per row (NE on the LIF's input current, DA on the MLP's
/// output), for dy [N, m] (f32 or bf16) and the unscaled x (any shape of N·m numbers, f32): (dL/dx as a matmul
/// operand, bf16 when `bf` else f32; dL/ds [N, 1] f32 = rowsum(dy ⊙ x)).
pub fn rows_bwd(dy: &Tensor, x: &Tensor, s: &Tensor, bf: bool) -> Result<(Tensor, Tensor)> {
    let dy = if dy.dtype() == DType::F32 { dy.clone() } else { dy.to_dtype(DType::F32)? };
    let (n, m) = dy.dims2()?;
    let x = x.reshape((n, m))?;
    if s.elem_count() != n || x.dtype() != DType::F32 {
        candle_core::bail!("rows_bwd: dy [{n}, {m}], x f32 of the same size and s [{n}, 1] wanted, got {:?} {:?} {:?}", x.dims(), x.dtype(), s.dims());
    }
    if use_k(&dy) {
        let (dy, x, s) = (dy.contiguous()?, x.contiguous()?, s.contiguous()?);
        let dx = wo::empty(&dy, (n, m), if bf { DType::BF16 } else { DType::F32 })?;
        let ds = wo::empty(&dy, (n, 1), DType::F32)?;
        let name = if bf { "nm_rows_bwd_b" } else { "nm_rows_bwd" };
        wo::launch(name, &[A::T(&dy), A::T(&x), A::T(&s), A::T(&dx), A::T(&ds), A::I(n as i32), A::I(m as i32)], wo::warps_cfg(n))?;
        return Ok((dx, ds));
    }
    let (dx, ds) = rows_bwd_ops(&dy, &x, &s.reshape((n, 1))?)?;
    Ok((if bf { crate::fused::to_bf16(&dx)? } else { dx }, ds))
}

/// rows_bwd in candle ops (dy, x [N, m], s [N, 1], one dtype): (dy ⊙ s, rowsum(dy ⊙ x) [N, 1]).
pub fn rows_bwd_ops(dy: &Tensor, x: &Tensor, s: &Tensor) -> Result<(Tensor, Tensor)> {
    Ok((dy.broadcast_mul(s)?, dy.mul(x)?.sum_keepdim(1)?))
}

// ---------------------------------------------------------------- ACh: the memory's write strengths

fn ach_check(vg: &Tensor, s: &Tensor, sh: MemShape) -> Result<(usize, usize)> {
    let (g, w) = (sh.g(), sh.hs + 2 * sh.g());
    if vg.dims() != [sh.b * sh.nh, sh.t, w] || s.elem_count() != sh.b * sh.t || !vg.is_contiguous() {
        candle_core::bail!("ach: vg [{}, {}, {w}] contiguous and s [{}, 1] wanted, got {:?} {:?}", sh.b * sh.nh, sh.t, sh.b * sh.t, vg.dims(), s.dims());
    }
    Ok((g, w))
}

/// --neuromod ACh, forward, on the memory scan's inputs vg [B·H, T, hs + 2G] (fwin_ops's layout: v, log alpha, beta;
/// row (b H + h) T + t): the beta columns (the write strengths) times the token's signal s [N, 1] (N = B·T row b T + t),
/// kept in [0, 2] (v50::ach_scale, the reference's op), written in place. Returns the unscaled strengths [B, H, T, G].
pub fn ach_fwd(vg: &Tensor, s: &Tensor, sh: MemShape) -> Result<Tensor> {
    let (g, w) = ach_check(vg, s, sh)?;
    if !use_k(vg) {
        return ach_fwd_ops(vg, s, sh);
    }
    let e = sh.b * sh.nh * sh.t * g;
    let s = s.contiguous()?;
    let braw = wo::empty(vg, (sh.b, sh.nh, sh.t, g), DType::F32)?;
    let args = [A::T(vg), A::T(&s), A::T(&braw), A::L(e as i64), A::I(sh.t as i32), A::I(sh.nh as i32), A::I(w as i32), A::I(g as i32)];
    wo::launch("nm_ach_fwd", &args, wo::elems_cfg(e))?;
    Ok(braw)
}

/// ach_fwd in candle ops.
pub fn ach_fwd_ops(vg: &Tensor, s: &Tensor, sh: MemShape) -> Result<Tensor> {
    let (g, w) = ach_check(vg, s, sh)?;
    let beta = vg.narrow(2, w - g, g)?.contiguous()?.reshape((sh.b, sh.nh, sh.t, g))?;
    let y = crate::v50::ach_scale(&beta, &s.reshape((sh.b * sh.t, 1))?)?;
    vg.slice_set(&y.reshape((sh.b * sh.nh, sh.t, g))?.contiguous()?, 2, w - g)?;
    Ok(beta)
}

/// --neuromod ACh, backward: dvg [B·H, T, hs + 2G] (the scan's gradient) holds dL/d(scaled strengths) in its beta
/// columns; they become dL/d(strengths) = dL/d(scaled)·m·s in place, m = clamp_mask(beta s); returns dL/ds [N, 1] = sum
/// over heads and columns of dL/d(scaled)·m·beta. `braw` [B, H, T, G]: ach_fwd's unscaled strengths.
pub fn ach_bwd(dvg: &Tensor, braw: &Tensor, s: &Tensor, sh: MemShape) -> Result<Tensor> {
    let (g, w) = ach_check(dvg, s, sh)?;
    if braw.dims() != [sh.b, sh.nh, sh.t, g] {
        candle_core::bail!("ach_bwd: strengths [{}, {}, {}, {g}] wanted, got {:?}", sh.b, sh.nh, sh.t, braw.dims());
    }
    if !use_k(dvg) {
        return ach_bwd_ops(dvg, braw, s, sh);
    }
    let n = sh.b * sh.t;
    let (braw, s) = (braw.contiguous()?, s.contiguous()?);
    let part = wo::empty(dvg, (n, sh.nh), DType::F32)?;
    let args = [A::T(dvg), A::T(&braw), A::T(&s), A::T(&part), A::I(n as i32), A::I(sh.t as i32), A::I(sh.nh as i32), A::I(w as i32), A::I(g as i32)];
    wo::launch("nm_ach_bwd", &args, wo::warps_cfg(n * sh.nh))?;
    part.sum_keepdim(1)
}

/// ach_bwd in candle ops.
pub fn ach_bwd_ops(dvg: &Tensor, braw: &Tensor, s: &Tensor, sh: MemShape) -> Result<Tensor> {
    let (g, w) = ach_check(dvg, s, sh)?;
    let (b, nh, t) = (sh.b, sh.nh, sh.t);
    let dsc = dvg.narrow(2, w - g, g)?.contiguous()?.reshape((b, nh, t, g))?;
    let s4 = s.reshape((b, 1, t, 1))?;
    let gd = dsc.mul(&clamp_mask(&braw.broadcast_mul(&s4)?, 0.0, 2.0)?)?;
    dvg.slice_set(&gd.broadcast_mul(&s4)?.reshape((b * nh, t, g))?.contiguous()?, 2, w - g)?;
    gd.mul(braw)?.sum_keepdim(3)?.sum_keepdim(1)?.reshape((b * t, 1))
}

/// The gradient factor of candle's y.clamp(lo, hi) = minimum(maximum(y, lo), hi): 1 inside (lo, hi), 1/2 on either
/// bound (candle splits a tie between both sides), 0 outside.
pub fn clamp_mask(y: &Tensor, lo: f64, hi: f64) -> Result<Tensor> {
    let f = |t: Tensor| t.to_dtype(y.dtype());
    let lo_side = f(y.ge(lo)?)?.add(&f(y.gt(lo)?)?)?;
    let hi_side = f(y.le(hi)?)?.add(&f(y.lt(hi)?)?)?;
    lo_side.mul(&hi_side)?.affine(0.25, 0.0)
}

#[cfg(test)]
mod tests;
