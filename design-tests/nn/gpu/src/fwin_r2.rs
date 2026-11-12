// TESTING / DESIGN ONLY — ava_gpu::fwin_r2: v51 fast1, round 2's first six parts on the hand-written WIN training step
// (src/model/fwin.rs), so Ava 0.4b's recipe with any of them trains without candle's autograd:
//   P  --dendrite   (src/v51.rs)  the MLP units a times 1 + tanh(hn(c) A B), c the block's memory read
//   DA --dopamine   (src/v51.rs)  blocks 1.. : the MLP output times 1 + tanh(z w + c), z the surprise (no gradient)
//   M  --mood       (src/v51.rs)  blocks 1.. : the LIF's input current times 1 + tanh(m G), m the leaky mood scan of z
//   Y  --rhythm     (src/v51.rs)  every block: the memory's write strengths and its read, per head, times 1 + tanh(f G),
//                                 f = (cos, sin) of the rhythms' phases (a scan over the embedding's resets)
//   Pr --prune      (src/v51.rs)  -1e9 on a cut unit before kWTA (none while no unit is cut), k from the live units; the
//                                 units in the step's own precision, their mean |a| (f32) for after_step (v52 fix_Pr)
//   S  --stp        (src/memory.rs, ava_parts::stp) the spikes through the short-term plasticity scan
// (hn: rms norm without gain; every gain starts at exactly 1.)
//
// Forward: the reference path's own candle ops (v51::rhythm_scan, mood_scan, rhythm_feats, gain, head_norm; the
// surprise as v51::after0; scan_fast's stp kernel), so the same numbers for the same inputs. Backward, no autograd,
// written out here in candle ops (the same ops on the CPU, where the tests check them against autograd, and on CUDA):
//   gains   y = x (1 + tanh p): dx = dy (1 + tanh p), dp = dy x (1 - tanh² p)
//   P       p = hn(c) A B: dB = (hn A)^T dp, dA = hn^T dp B^T, dc = hn's backward of dp B^T A^T (joins the memory
//           read's gradient, before the memory's own backward)
//   DA      p = z w + c: dw = colsum(dp z), dc = colsum(dp)
//   M       p = m G: dG = m^T dp, dm = dp G^T summed over the blocks; then the scan m_t = λ m_{t-1} + (1-λ) u_t
//           backward: g_t = dm_t + λ g_{t+1} (a reverse linear scan), du = (1-λ) g, dλ = Σ g_t (m_{t-1} - u_t);
//           u = z win + bin
//   Y       the write gate on the memory scan's input vg (between fwin_ops's prep and --neuromod's ACh), the read gate
//           on the memory's read-out per head (out_gate is elementwise, so the same product); df summed over the
//           blocks; f = (cos φ, sin φ), φ = a ω; a_t = q_t (a_{t-1} + 1), q = sigmoid(-pre): g_t = da_t + q_{t+1}
//           g_{t+1}, dpre_t = -g_t (a_{t-1} + 1) q_t (1 - q_t); pre = hn(x) wr + br, x the embedding
//   Pr      nothing (a constant on u; the cut units never win, so their gradient is 0 as in autograd)
//   S       out = s + mix (s e - s), e the scan's efficacy of events(s): scan_fast's stp_bwd kernel (its autograd op's
//           own), the events' slope (1 on [0, 1], -1 on [-1, 0), 0 beyond)
// The reverse scans: chunks of v51::SCAN_CHUNK in closed form, as the forward scans.

use crate::model::{head_norm, log_sigmoid, norm, sigmoid, Model};
use crate::v51::SCAN_CHUNK;
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result, Tensor};

// ------------------------------------------------------------------------------------------- which parts are on

/// The options this step takes from round 2 cleared (the rest, R --letters and W --columns, stay).
pub fn off(o: crate::v51::V51Opts) -> crate::v51::V51Opts {
    crate::v51::V51Opts { dendrite: false, dopamine: false, mood: 0, rhythm: false, prune: 0.0, ..o }
}

fn v51(m: &Model) -> Option<&crate::v51::V51Ext> {
    m.ext.as_ref().map(|e| &e.v51).filter(|v| v.on)
}

fn blk(m: &Model, i: usize) -> Option<&crate::v51::V51Block> {
    v51(m).and_then(|v| v.blocks.get(i))
}

/// S: block i's synapse (None without --stp).
pub fn stp_of(m: &Model, i: usize) -> Option<&ava_parts::stp::Stp> {
    m.ext.as_ref().and_then(|e| e.mem.parts.get(i)).and_then(|p| p.stp.as_ref())
}

/// P on in block i: the MLP's units are made in f32 (the dendrites scale them before W2). v52 fix_Pr: Pr's units are
/// no longer made in f32: they stay in the step's own precision, as B's (a cut unit is 0 either way, and its statistics
/// are read in f32 from the units as they are), so Pr with no unit cut takes B's step bit for bit. Pr's f32 units
/// made its bf16 steps differ from B's from the first step on (a different rounding of the relu² backward), the
/// trajectory that blew up in v52 Stage 1. AVA_UNITS_F32=1 (diagnostic only) makes every block's units f32, the old Pr
/// path, in any model (runs/v52/fix_Pr.md).
pub fn mlp_f32(m: &Model, i: usize) -> bool {
    blk(m, i).is_some_and(|b| b.dend.is_some()) || units_f32()
}

/// AVA_UNITS_F32=1: every block's MLP units in f32 (diagnostic; read once).
pub fn units_f32() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AVA_UNITS_F32").is_ok_and(|v| v == "1"))
}

// ------------------------------------------------------------------------------------------- saved values

/// Y's forward: the embedding's norm, the resets' logits [b, t, 2], the times since the resets a [b, t, 2], the
/// phases' features f [n, 4].
pub struct Rh {
    hn: Tensor,
    pre: Tensor,
    a: Tensor,
    pub f: Tensor,
}

/// M's forward: the drive u [b, t, k] and the mood m [b, t, k] (`m2` the same as [n, k]).
pub struct Mood {
    u: Tensor,
    m: Tensor,
    pub m2: Tensor,
}

/// The signals of one forward (made before the blocks and after block 0).
#[derive(Default)]
pub struct Fwd {
    pub rh: Option<Rh>,
    /// DA, M: the surprise / ln V [n, 1] (a constant)
    pub z: Option<Tensor>,
    pub mood: Option<Mood>,
    b: usize,
    t: usize,
}

/// P's saved values: the units before the gain (f32), hn(c) A [n, R], tanh(p) [n, hidden], the memory read c [n, d].
pub struct Dend {
    a: Tensor,
    ha: Tensor,
    th: Tensor,
    ctx: Tensor,
}

/// One block's saved values.
#[derive(Default)]
pub struct Blk {
    /// M: (the current before the mood's gain [b, t, d], tanh(p) [n, d])
    pub mood: Option<(Tensor, Tensor)>,
    /// S: (the spikes [b, t, d] f32, their events, the efficacy)
    pub stp: Option<(Tensor, Tensor, Tensor)>,
    /// Y write: (the gate as the scan's layout [B·H, T, 1], tanh(p) [n, H]) and the ungated strengths fwin_ops kept
    pub wg: Option<(Tensor, Tensor)>,
    pub bung: Option<Tensor>,
    /// Y read: (the gate [n, H], tanh(p) [n, H], the read-out before it [n, d] f32)
    pub rg: Option<(Tensor, Tensor, Tensor)>,
    /// P: the memory read (set by `put_ctx`), then the dendrite's values
    pub ctx: Option<Tensor>,
    pub dend: Option<Dend>,
    /// DA: (the MLP output before the gain [n, d], tanh(p) [n, d])
    pub da: Option<(Tensor, Tensor)>,
}

/// The gradients the blocks hand on: df [n, 4] (Y) and dm [n, k] (M), each summed over the blocks.
#[derive(Default)]
pub struct Bwd {
    df: Option<Tensor>,
    dm: Option<Tensor>,
}

fn acc(slot: &mut Option<Tensor>, t: Tensor) -> Result<()> {
    crate::fwin_nm::acc(slot, t)
}

/// 1 + tanh(p), and tanh(p) (`gain`'s bits: tanh, then + 1).
fn gain_th(p: &Tensor) -> Result<(Tensor, Tensor)> {
    let th = p.tanh()?;
    Ok(((th.clone() + 1.0)?, th))
}

/// The backward of y = x (1 + th): (dx, dp) for dy, x and th = tanh(p).
pub fn gain_bwd(dy: &Tensor, x: &Tensor, th: &Tensor) -> Result<(Tensor, Tensor)> {
    let dx = dy.mul(&(th.clone() + 1.0)?)?;
    let dp = dy.mul(x)?.mul(&th.sqr()?.affine(-1.0, 1.0)?)?;
    Ok((dx, dp))
}

/// hn's backward (rms norm without gain): dx for dy (fwin_nm::rms_bwd with a unit gain: the fused kernel on CUDA).
pub fn hn_bwd(x: &Tensor, dy: &Tensor) -> Result<Tensor> {
    let ones = Tensor::ones(x.dim(1)?, x.dtype(), x.device())?;
    Ok(crate::fwin_nm::rms_bwd(x, &ones, dy)?.0)
}

// ------------------------------------------------------------------------------------------- the scans' backward

fn tri(c: usize, dt: DType, dev: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; c * c];
    for t in 0..c {
        for j in 0..=t {
            m[t * c + j] = 1.0;
        }
    }
    Tensor::from_vec(m, (c, c), dev)?.to_dtype(dt)
}

/// y_t = x_t + c_t y_{t-1} (y_{-1} = 0) along axis 1 of x [b, t, o], from ln c [b, t, o] (<= 0): chunks of SCAN_CHUNK
/// in closed form, y_t = Σ_{j<=t} exp(L_t - L_j) x_j + exp(L_t) y_init, L the chunk's running sum of ln c.
pub fn lin_scan(x: &Tensor, lc: &Tensor) -> Result<Tensor> {
    let (b, t, o) = x.dims3()?;
    let (dt, dev) = (x.dtype(), x.device());
    let xs = x.permute((0, 2, 1))?.contiguous()?; // [b, o, t]
    let ls = lc.permute((0, 2, 1))?.contiguous()?;
    let mut init = Tensor::zeros((b, o), dt, dev)?;
    let mut outs = Vec::new();
    let mut c0 = 0;
    while c0 < t {
        let c = SCAN_CHUNK.min(t - c0);
        let tr = tri(c, dt, dev)?;
        let l = ls.narrow(2, c0, c)?.contiguous()?;
        let big = l.broadcast_matmul(&tr.t()?.contiguous()?)?; // L_t [b, o, c]
        let diff = big.unsqueeze(3)?.broadcast_sub(&big.unsqueeze(2)?)?; // L_t - L_j [b, o, c(t), c(j)]
        let w = diff.minimum(0.0)?.exp()?.broadcast_mul(&tr.reshape((1, 1, c, c))?)?;
        let xc = xs.narrow(2, c0, c)?.contiguous()?.unsqueeze(3)?; // [b, o, c, 1]
        let y = w.matmul(&xc)?.squeeze(3)?.add(&big.exp()?.broadcast_mul(&init.unsqueeze(2)?)?)?;
        init = y.narrow(2, c - 1, 1)?.squeeze(2)?;
        outs.push(y);
        c0 += c;
    }
    let y = if outs.len() == 1 { outs.pop().expect("one chunk") } else { Tensor::cat(&outs, 2)? };
    y.permute((0, 2, 1))?.contiguous()
}

/// g_t = x_t + c_{t+1} g_{t+1} (g_T = 0), the reverse of lin_scan's recurrence (a scan's backward), from ln c [b, t, o].
pub fn rev_scan(x: &Tensor, lc: &Tensor) -> Result<Tensor> {
    let (b, t, o) = x.dims3()?;
    let idx = Tensor::from_vec((0..t as u32).rev().collect::<Vec<u32>>(), t, x.device())?;
    let xr = x.index_select(&idx, 1)?;
    // reversed (s = T-1-t): G_s = X_s + C_s G_{s-1}, C_s = c_{T-s}: the reversed ln c moved on by one (C_0 unused)
    let lr = lc.index_select(&idx, 1)?;
    let z = Tensor::zeros((b, 1, o), x.dtype(), x.device())?;
    let lsh = if t > 1 { Tensor::cat(&[&z, &lr.narrow(1, 0, t - 1)?], 1)? } else { z };
    lin_scan(&xr, &lsh)?.index_select(&idx, 1)
}

/// x [b, t, o] moved one step later along t (x_{t-1}, 0 at t = 0).
fn prev(x: &Tensor) -> Result<Tensor> {
    let (b, t, o) = x.dims3()?;
    let z = Tensor::zeros((b, 1, o), x.dtype(), x.device())?;
    if t == 1 {
        return Ok(z);
    }
    Tensor::cat(&[&z, &x.narrow(1, 0, t - 1)?], 1)
}

/// M: v51::mood_scan's backward (from m0 = 0): (du [b, t, k], dlam [k]) for dm [b, t, k].
pub fn mood_scan_bwd(dm: &Tensor, u: &Tensor, m: &Tensor, lam: &Tensor) -> Result<(Tensor, Tensor)> {
    let k = lam.dim(0)?;
    let ll = log_sigmoid(lam)?.reshape((1, 1, k))?; // ln λ
    let g = rev_scan(dm, &ll.broadcast_as(dm.shape())?.contiguous()?)?;
    let lv = sigmoid(lam)?;
    let oml = sigmoid(&lam.neg()?)?; // 1 - λ
    let dl = g.mul(&prev(m)?.sub(u)?)?.sum(0)?.sum(0)?; // dL/dλ [k]
    let du = g.broadcast_mul(&oml.reshape((1, 1, k))?)?;
    Ok((du, dl.mul(&lv)?.mul(&oml)?))
}

/// Y: the backward of pre -> a (v51::rhythm_scan from a0 = 0) -> f (v51::rhythm_feats): (dpre [b, t, 2], dlw [2]) for
/// df [n, 4].
pub fn rhythm_bwd(df: &Tensor, pre: &Tensor, a: &Tensor, lw: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, t, o) = a.dims3()?;
    let w = lw.exp()?.reshape((1, 1, o))?;
    let ph = a.broadcast_mul(&w)?;
    let d4 = df.reshape((b, t, o, 2))?;
    let (dc, ds) = (d4.narrow(3, 0, 1)?.squeeze(3)?, d4.narrow(3, 1, 1)?.squeeze(3)?);
    let dph = ds.mul(&ph.cos()?)?.sub(&dc.mul(&ph.sin()?)?)?;
    let dlw = dph.mul(&ph)?.sum(0)?.sum(0)?;
    let da = dph.broadcast_mul(&w)?;
    let g = rev_scan(&da, &log_sigmoid(&pre.neg()?)?)?; // q_t = 1 - r_t = sigmoid(-pre)
    let qr = sigmoid(&pre.neg()?)?.mul(&sigmoid(pre)?)?;
    let dpre = g.mul(&prev(a)?.affine(1.0, 1.0)?)?.mul(&qr)?.neg()?;
    Ok((dpre, dlw))
}

// ------------------------------------------------------------------------------------------- forward hooks

/// Before the blocks: Y's phases for the embedding x [n, d] (v51::begin's ops), and the forward's shape.
pub fn begin(m: &Model, x: &Tensor, b: usize, t: usize) -> Result<Fwd> {
    let mut f = Fwd { b, t, ..Default::default() };
    if let Some(rp) = v51(m).and_then(|v| v.rhythm.as_ref()) {
        let hn = head_norm(x)?;
        let pre = hn.matmul(&rp.wr.detach())?.broadcast_add(&rp.br.detach())?.reshape((b, t, 2))?;
        let a = crate::v51::rhythm_scan(&pre, &Tensor::zeros((b, 2), x.dtype(), x.device())?)?;
        let ft = crate::v51::rhythm_feats(&a, &rp.lw.detach())?;
        f.rh = Some(Rh { hn, pre, a, f: ft });
    }
    Ok(f)
}

/// After block 0 (its output x0 [n, d], after its mix): DA's and M's surprise and the mood (v51::after0's ops, no
/// carried state: the first token's surprise is 0). `tok` the tied embedding (detached).
pub fn after0(m: &Model, f: &mut Fwd, x0: &Tensor, ids: &Tensor, tok: &Tensor) -> Result<()> {
    let Some(v) = v51(m) else { return Ok(()) };
    let o = m.cfg.arch.v51;
    if !o.surprise() {
        return Ok(());
    }
    let (b, t, d, dev) = (f.b, f.t, m.cfg.d, x0.device());
    let xf = norm(&x0.detach(), &m.final_norm.detach())?;
    let mut parts = vec![Tensor::zeros(b, x0.dtype(), dev)?.reshape((b, 1))?];
    if t > 1 {
        let rows = xf.reshape((b, t, d))?.narrow(1, 0, t - 1)?.contiguous()?.reshape((b * (t - 1), d))?;
        let l = crate::int8::head(&rows, tok)?;
        let tg = ids.narrow(1, 1, t - 1)?.contiguous()?.flatten_all()?;
        parts.push(crate::lif::cross_entropy_rows(&l, &tg)?.reshape((b, t - 1))?);
    }
    let z = Tensor::cat(&parts, 1)?.affine(1.0 / (m.cfg.vocab as f64).ln(), 0.0)?; // [b, t]
    if let Some(mp) = v.mood.as_ref() {
        let k = o.mood;
        let u = z.unsqueeze(2)?.broadcast_mul(&mp.win.detach().reshape((1, 1, k))?)?.broadcast_add(&mp.bin.detach().reshape((1, 1, k))?)?;
        let mm = crate::v51::mood_scan(&u, &mp.lam.detach(), &Tensor::zeros((b, k), x0.dtype(), dev)?)?;
        let m2 = mm.reshape((b * t, k))?;
        f.mood = Some(Mood { u, m: mm, m2 });
    }
    f.z = Some(z.reshape((b * t, 1))?);
    Ok(())
}

/// M: block i's LIF input current cur [b, t, d] times the mood's gain (cur itself in block 0 or without --mood).
pub fn mood_fwd(m: &Model, i: usize, f: &Fwd, cur: Tensor, sv: &mut Blk) -> Result<Tensor> {
    let (Some(g), Some(md)) = (blk(m, i).and_then(|b| b.mood_g.as_ref()), f.mood.as_ref()) else { return Ok(cur) };
    let (gn, th) = gain_th(&md.m2.matmul(&g.detach())?)?;
    let out = cur.mul(&gn.reshape(cur.dims())?)?;
    sv.mood = Some((cur, th));
    Ok(out)
}

/// S: block i's spikes s [b, t, d] f32 through the synapse (scan_fast::stp_forward's ops; s itself without --stp).
pub fn stp_fwd(m: &Model, i: usize, s: Tensor, sv: &mut Blk) -> Result<Tensor> {
    let Some(p) = stp_of(m, i) else { return Ok(s) };
    let ps = stp_ps(p)?;
    let ev = ava_parts::stp::events(&s)?;
    let eff = crate::scan_fast::stp_scan_nb(&ev, &ps)?;
    let out = (&s + ((&s * &eff)? - &s)?.broadcast_mul(&p.mix.detach())?)?;
    sv.stp = Some((s, ev, eff));
    Ok(out)
}

fn stp_ps(p: &ava_parts::stp::Stp) -> Result<Tensor> {
    use candle_nn::ops::sigmoid as sg;
    Tensor::cat(&[&sg(&p.u_param.detach())?, &sg(&p.f_param.detach())?, &sg(&p.d_param.detach())?], 0)?.contiguous()
}

/// Y: block i's write gate as the memory scan's layout [B·H, T, 1] (None without --rhythm), kept in `sv`.
pub fn wgate(m: &Model, i: usize, f: &Fwd, nh: usize, sv: &mut Blk) -> Result<Option<Tensor>> {
    let (Some(rh), Some((gw, _))) = (f.rh.as_ref(), blk(m, i).and_then(|b| b.rh.as_ref())) else { return Ok(None) };
    let (g, th) = gain_th(&rh.f.matmul(&gw.detach())?)?; // [n, H]
    let lay = g.reshape((f.b, f.t, nh))?.permute((0, 2, 1))?.contiguous()?.reshape((f.b * nh, f.t, 1))?;
    sv.wg = Some((lay.clone(), th));
    Ok(Some(lay))
}

/// Y: block i's read gate [n, H] (None without --rhythm).
pub fn rgate(m: &Model, i: usize, f: &Fwd) -> Result<Option<(Tensor, Tensor)>> {
    let (Some(rh), Some((_, gr))) = (f.rh.as_ref(), blk(m, i).and_then(|b| b.rh.as_ref())) else { return Ok(None) };
    gain_th(&rh.f.matmul(&gr.detach())?).map(Some)
}

/// Y: the write gate on the memory scan's inputs vg [B·H, T, hs + 2G] (fwin_ops's layout; the strengths are the last G
/// columns), in place; returns the ungated strengths [B·H, T, G].
pub fn wgate_fwd(vg: &Tensor, gate: &Tensor, g: usize) -> Result<Tensor> {
    let w = vg.dim(2)?;
    let beta = vg.narrow(2, w - g, g)?.contiguous()?;
    vg.slice_set(&beta.broadcast_mul(gate)?.contiguous()?, 2, w - g)?;
    Ok(beta)
}

/// Y: wgate_fwd's backward: dvg's strength columns hold dL/d(gated strengths); they become dL/d(strengths) in place;
/// returns dL/d(gate) [B·H, T, 1].
pub fn wgate_bwd(dvg: &Tensor, beta: &Tensor, gate: &Tensor, g: usize) -> Result<Tensor> {
    let w = dvg.dim(2)?;
    let dsc = dvg.narrow(2, w - g, g)?.contiguous()?;
    let dgate = dsc.mul(beta)?.sum_keepdim(2)?;
    dvg.slice_set(&dsc.broadcast_mul(gate)?.contiguous()?, 2, w - g)?;
    Ok(dgate)
}

/// Y: the read gate on the memory's read-out o [n, d] f32 (head-major columns), per head.
pub fn rgate_fwd(o: &Tensor, g: &Tensor) -> Result<Tensor> {
    let (n, d) = o.dims2()?;
    let nh = g.dim(1)?;
    o.reshape((n, nh, d / nh))?.broadcast_mul(&g.unsqueeze(2)?)?.reshape((n, d))
}

/// P: block i's units a [n, hidden] f32 times the dendrite's gain from the memory read in `sv.ctx` (a itself without
/// --dendrite).
pub fn dend_fwd(m: &Model, i: usize, a: Tensor, sv: &mut Blk) -> Result<Tensor> {
    let Some((da, db)) = blk(m, i).and_then(|b| b.dend.as_ref()) else { return Ok(a) };
    let ctx = sv.ctx.take().ok_or_else(|| candle_core::Error::Msg(format!("fwin_r2: block {i}'s dendrites without the memory read")))?;
    let ha = head_norm(&ctx)?.matmul(&da.detach())?;
    let (g, th) = gain_th(&ha.matmul(&db.detach())?)?;
    let out = a.mul(&g)?;
    sv.dend = Some(Dend { a, ha, th, ctx });
    Ok(out)
}

/// P: whether block i keeps its memory read for the dendrites.
pub fn wants_ctx(m: &Model, i: usize) -> bool {
    blk(m, i).is_some_and(|b| b.dend.is_some())
}

/// DA: block i's MLP output y [n, d] times the surprise's gain (y itself in block 0 or without --dopamine).
pub fn da_fwd(m: &Model, i: usize, f: &Fwd, y: Tensor, sv: &mut Blk) -> Result<Tensor> {
    let (Some((w, c)), Some(z)) = (blk(m, i).and_then(|b| b.da.as_ref()), f.z.as_ref()) else { return Ok(y) };
    let (g, th) = gain_th(&z.broadcast_mul(&w.detach().unsqueeze(0)?)?.broadcast_add(&c.detach())?)?;
    let out = y.mul(&g)?;
    sv.da = Some((y, th));
    Ok(out)
}

/// Pr: u [n, hidden] with the cut units at -1e9 and block i's k (u and `k` as they are without --prune, and while no
/// unit of block i is cut: the mask would add exactly 0, one pass over u for nothing).
pub fn prune_u(m: &Model, i: usize, u: Tensor, k: usize) -> Result<(Tensor, usize)> {
    let (Some(v), Some((alive, _))) = (v51(m), blk(m, i).and_then(|b| b.prune.as_ref())) else { return Ok((u, k)) };
    let kb = crate::v51::kwta_k(m, v, i)?;
    if crate::v51::alive_count(m, v, i)? == m.cfg.hidden() {
        return Ok((u, kb));
    }
    let u = u.broadcast_add(&alive.detach().affine(1e9, -1e9)?)?;
    Ok((u, kb))
}

/// Pr (tests): the units' statistics the last training forward left, per block (v51::after_step's input; empty
/// without round 2's parts).
pub fn prune_stats_now(m: &Model) -> Vec<Option<Tensor>> {
    v51(m).map(|v| v.prune.stats.borrow().clone()).unwrap_or_default()
}

/// Pr: the units' mean |a| over the rows [hidden], in f32 whatever a's dtype (bf16 units in a bf16 step), for
/// v51::after_step; kept only while a schedule runs (`v51_schedule`'s total > 0; without one nothing reads them, and the
/// slot is cleared so no older value stays); nothing without --prune.
pub fn prune_stats(m: &Model, i: usize, a: &Tensor) -> Result<()> {
    if let (Some(v), true) = (v51(m), blk(m, i).is_some_and(|b| b.prune.is_some())) {
        let st = if v.prune.total.get() > 0 { Some(a.detach().to_dtype(DType::F32)?.abs()?.mean(0)?) } else { None };
        v.prune.stats.borrow_mut()[i] = st;
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------- backward hooks

fn ins(grads: &mut GradStore, var: &Tensor, g: Tensor) -> Result<()> {
    match grads.remove(var) {
        Some(old) => grads.insert(var, old.add(&g)?),
        None => grads.insert(var, g),
    };
    Ok(())
}

/// DA: dL/d(the MLP output before the gain) [n, d] for dy (f32) = dL/d(after it); the gain's weights' gradients.
pub fn da_bwd(m: &Model, i: usize, f: &Fwd, sv: &Blk, dy: &Tensor, grads: &mut GradStore) -> Result<Tensor> {
    let (Some((y, th)), Some((w, c)), Some(z)) = (sv.da.as_ref(), blk(m, i).and_then(|b| b.da.as_ref()), f.z.as_ref()) else { return Ok(dy.clone()) };
    let (dx, dp) = gain_bwd(dy, y, th)?;
    ins(grads, w, dp.broadcast_mul(z)?.sum(0)?)?;
    ins(grads, c, dp.sum(0)?)?;
    Ok(dx)
}

/// P: (dL/d(the units before the gain), dL/d(the memory read)) for da (f32) = dL/d(after it); A's and B's gradients.
pub fn dend_bwd(m: &Model, i: usize, sv: &Blk, da: &Tensor, grads: &mut GradStore) -> Result<Option<(Tensor, Tensor)>> {
    let (Some(dv), Some((wa, wb))) = (sv.dend.as_ref(), blk(m, i).and_then(|b| b.dend.as_ref())) else { return Ok(None) };
    let (dx, dp) = gain_bwd(da, &dv.a, &dv.th)?;
    ins(grads, wb, dv.ha.t()?.matmul(&dp)?)?;
    let dha = dp.matmul(&wb.detach().t()?)?;
    let hn = head_norm(&dv.ctx)?;
    ins(grads, wa, hn.t()?.matmul(&dha)?)?;
    let dctx = hn_bwd(&dv.ctx, &dha.matmul(&wa.detach().t()?)?)?;
    Ok(Some((dx, dctx)))
}

/// P: the units before the gain (f32), for the activation's backward.
pub fn dend_units(sv: &Blk) -> Option<&Tensor> {
    sv.dend.as_ref().map(|d| &d.a)
}

/// Y read: dL/d(the read-out before the gate) for d_o [n, d] (f32); df and the gate's weight gradient.
pub fn rgate_bwd(m: &Model, i: usize, f: &Fwd, sv: &Blk, d_o: &Tensor, bw: &mut Bwd, grads: &mut GradStore) -> Result<Tensor> {
    let (Some((g, th, o)), Some((_, gr)), Some(rh)) = (sv.rg.as_ref(), blk(m, i).and_then(|b| b.rh.as_ref()), f.rh.as_ref()) else { return Ok(d_o.clone()) };
    let (n, d) = d_o.dims2()?;
    let nh = g.dim(1)?;
    let d3 = d_o.reshape((n, nh, d / nh))?;
    let dx = d3.broadcast_mul(&g.unsqueeze(2)?)?.reshape((n, d))?;
    let dg = d3.mul(&o.reshape((n, nh, d / nh))?)?.sum(2)?; // [n, H]
    gate_params(&dg, th, &rh.f, gr, bw, grads)?;
    Ok(dx)
}

/// Y write: fwin_ops's dL/d(gate) [B·H, T, 1] into df and the gate's weight gradient.
pub fn wgate_bwd_params(m: &Model, i: usize, f: &Fwd, sv: &Blk, dgate: &Tensor, bw: &mut Bwd, grads: &mut GradStore) -> Result<()> {
    let (Some((_, th)), Some((gw, _)), Some(rh)) = (sv.wg.as_ref(), blk(m, i).and_then(|b| b.rh.as_ref()), f.rh.as_ref()) else { return Ok(()) };
    let nh = th.dim(1)?;
    let dg = dgate.reshape((f.b, nh, f.t))?.permute((0, 2, 1))?.contiguous()?.reshape((f.b * f.t, nh))?;
    gate_params(&dg, th, &rh.f, gw, bw, grads)
}

/// g = 1 + tanh(f W) [n, H]: dW = f^T dp, df += dp W^T, for dg.
fn gate_params(dg: &Tensor, th: &Tensor, f: &Tensor, w: &Tensor, bw: &mut Bwd, grads: &mut GradStore) -> Result<()> {
    let dp = dg.mul(&th.sqr()?.affine(-1.0, 1.0)?)?;
    ins(grads, w, f.t()?.matmul(&dp)?)?;
    acc(&mut bw.df, dp.matmul(&w.detach().t()?)?)
}

/// S: dL/d(the LIF's spikes) [b, t, d] for ds (f32) = dL/d(the synapse's output); the synapse's gradients.
pub fn stp_bwd(m: &Model, i: usize, sv: &Blk, ds: &Tensor, grads: &mut GradStore) -> Result<Tensor> {
    let (Some((s, ev, eff)), Some(p)) = (sv.stp.as_ref(), stp_of(m, i)) else { return Ok(ds.clone()) };
    let c = s.dim(2)?;
    let ds = ds.reshape(s.dims())?;
    let mix = p.mix.detach();
    // out = s + mix (s e - s): direct ds (1 + mix (e - 1)), de = ds mix s, dmix = Σ ds (s e - s)
    let de = ds.broadcast_mul(&mix)?.mul(s)?;
    grads.insert(&p.mix, ds.mul(&s.mul(eff)?.sub(s)?)?.sum(0)?.sum(0)?);
    let direct = ds.add(&ds.broadcast_mul(&mix)?.mul(&eff.affine(1.0, -1.0)?)?)?;
    let ps = stp_ps(p)?;
    let (g_ev, g_ps) = crate::scan_fast::stp_scan_bwd(ev, &ps, &de)?;
    // the events' slope: 1 on [0, 1], -1 on [-1, 0), 0 beyond (ava_parts::stp::events)
    let f = |t: Tensor| t.to_dtype(s.dtype());
    let k = f(s.ge(0.0)?)?.mul(&f(s.le(1.0)?)?)?.sub(&f(s.lt(0.0)?)?.mul(&f(s.ge(-1.0)?)?)?)?;
    let dsp = direct.add(&g_ev.mul(&k)?)?;
    for (j, var) in [&p.u_param, &p.f_param, &p.d_param].into_iter().enumerate() {
        let sg = ps.narrow(0, j * c, c)?;
        grads.insert(var, g_ps.narrow(0, j * c, c)?.mul(&sg)?.mul(&sg.affine(-1.0, 1.0)?)?);
    }
    Ok(dsp)
}

/// M: dL/d(the current before the mood's gain) [n, d] for dcur (f32) = dL/d(after it); dm and G's gradient.
pub fn mood_bwd(m: &Model, i: usize, f: &Fwd, sv: &Blk, dcur: &Tensor, bw: &mut Bwd, grads: &mut GradStore) -> Result<Tensor> {
    let (Some((cur, th)), Some(g), Some(md)) = (sv.mood.as_ref(), blk(m, i).and_then(|b| b.mood_g.as_ref()), f.mood.as_ref()) else { return Ok(dcur.clone()) };
    let dcur = dcur.reshape(th.dims())?;
    let (dx, dp) = gain_bwd(&dcur, &cur.reshape(th.dims())?, th)?;
    ins(grads, g, md.m2.t()?.matmul(&dp)?)?;
    acc(&mut bw.dm, dp.matmul(&g.detach().t()?)?)?;
    Ok(dx)
}

/// After the blocks' backward: M's scan and Y's rhythms into their weights' gradients; returns Y's dL/d(embedding)
/// [n, d] (None without --rhythm). `x` the embedding (Y's input).
pub fn finish(m: &Model, f: &Fwd, bw: Bwd, x: &Tensor, grads: &mut GradStore) -> Result<Option<Tensor>> {
    let Some(v) = v51(m) else { return Ok(None) };
    if let (Some(md), Some(mp), Some(dm), Some(z)) = (f.mood.as_ref(), v.mood.as_ref(), bw.dm, f.z.as_ref()) {
        let k = mp.lam.dim(0)?;
        let (du, dlam) = mood_scan_bwd(&dm.reshape(md.m.dims())?, &md.u, &md.m, &mp.lam.detach())?;
        let du = du.reshape((f.b * f.t, k))?;
        grads.insert(&mp.lam, dlam);
        grads.insert(&mp.win, du.broadcast_mul(z)?.sum(0)?);
        grads.insert(&mp.bin, du.sum(0)?);
    }
    let (Some(rh), Some(rp), Some(df)) = (f.rh.as_ref(), v.rhythm.as_ref(), bw.df) else { return Ok(None) };
    let (dpre, dlw) = rhythm_bwd(&df, &rh.pre, &rh.a, &rp.lw.detach())?;
    let dp2 = dpre.reshape((f.b * f.t, 2))?;
    grads.insert(&rp.lw, dlw);
    grads.insert(&rp.wr, rh.hn.t()?.matmul(&dp2)?);
    grads.insert(&rp.br, dp2.sum(0)?);
    Ok(Some(hn_bwd(x, &dp2.matmul(&rp.wr.detach().t()?)?)?))
}

#[cfg(test)]
mod tests;
