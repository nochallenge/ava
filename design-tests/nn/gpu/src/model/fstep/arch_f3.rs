// TESTING / DESIGN ONLY — v51 fast3 in the fused one-token talking step (src/model/fstep/arch.rs): S --stp, M --mood,
// Pr --prune and CB --cerebellum-head, with the same bits as `forward_with_state` with int8 on (every float operation
// the one candle performs on the CPU, in its order; the small products in gemm's one-row order, fstep.rs's `gemv`):
//   S   the spikes s through the synapse (state.rs `stp_from` at t = 1, from the State's (u, x)): the event size
//       ev = s·k + (min(|s|, 1) - s·k) (ava_parts::stp::events: k = 1 on [0, 1], -1 on [-1, 0), 0 beyond; the clamp
//       maximum then minimum), u+ = u + ((u·-1 + 1) U) ev, e = u+ x, eff = e / U, x' = ((x - e ev)·-1 + 1) D·-1 + 1,
//       u' = u+ F, out = s + ((s eff) - s) mix (U, F, D candle_nn's sigmoid, the same bits as fstep.rs's `sigmoid`);
//       what reaches W_out is out, the LIF keeps s
//   M   after block 0 (v51::after0 at t = 1): xf = rmsnorm(x, final norm), the surprise s = CE(head(prev), tok)·valid
//       (prev: the State's last xf, its head product the int8 head's, the CE lif.rs's CPU loop), z = s·(1/ln V) + 0,
//       u = z win + bin, the mood scan at c = 1 (v51::mood_scan): m = (exp(0·ll)·1)(u oml) + m0 exp(1·ll) (ll =
//       log_sigmoid(lam), oml = sigmoid(-lam); gemm's K = 1 product is the plain product); in every later block the
//       LIF's input current times tanh(m G)·1 + 1 (G [k, d], gemv), after --neuromod's NE (block_state's order)
//   Pr  the cut units at -1e9 before kWTA (u + (alive·1e9 + -1e9)) and k from the live units (v51::kwta_k)
//   CB  after the final norm (v51b_cb on one row): u = [xf, rmsnorm(emb, 1)], h_j = 0 + u[i0] s0 + ... (fan-in 4, candle's
//       sum from 0), kWTA by the k-th largest (lif.rs sorts; the same mask), relu, c = a / (sqrt(0 + a0² + ...)·1 + 1e-6),
//       the CPU read-out's gather: corr = 0 + W[j0] c_j0 + W[j1] c_j1 + ... over the non-zero codes in unit order (candle's
//       sum over a middle dim), xf + corr; the chat app's part signal sqrt(sum(corr²)·(1/d) + 0). An online table keeps the
//       plain step (arch_f2.rs `v51b_why`).
// The State moves on only when the whole token is computed (arch.rs `step_rec`): `commit`.
// f4ext (runs/v52/f4ext/report.md): M's surprise also runs for DA alone (an empty mood; `Mood::z` feeds DA's gates in
// arch_f4.rs), and `commit` also keeps Y's times.
// Tests: src/model/fstep/arch_f3_tests.rs (every logit and the whole State bit for bit over 300 tokens).

use super::super::{gemv, log_sigmoid, rmsnorm, ro, sigmoid, vals};
use crate::model::Model;
use crate::state::State;
use crate::v51::{V51Ext, V51Opts};
use candle_core::{CpuStorage, DType, Device, Result, Storage, Tensor};

/// The v51 options this step takes off the blocker's list (M, Pr); with arch_f2's R. The others stay.
pub(super) fn strip(o: V51Opts) -> V51Opts {
    V51Opts { mood: 0, prune: 0.0, ..o }
}

fn v51(m: &Model) -> Option<&V51Ext> {
    m.ext.as_ref().map(|e| &e.v51).filter(|v| v.on)
}

fn stp_of(m: &Model, i: usize) -> Option<&ava_parts::stp::Stp> {
    m.ext.as_ref().and_then(|e| e.mem.parts.get(i)).and_then(|p| p.stp.as_ref())
}

/// The State carries what S and M need, in the one-row shapes this step reads and writes.
pub(super) fn state_ok(m: &Model, st: &State) -> bool {
    let d = m.cfg.d;
    let n = |t: &Tensor, k: usize| t.elem_count() == k;
    for i in 0..m.blocks.len() {
        if stp_of(m, i).is_some() && !st.parts.get(i).and_then(|p| p.stp.as_ref()).is_some_and(|(u, x)| n(u, d) && n(x, d)) {
            return false;
        }
    }
    // f4ext: the surprise for DA too (M's stream and valid flag), and Y's carried times
    let o = m.cfg.arch.v51;
    let v = &st.v51;
    let ok = |t: &Option<Tensor>, k: usize| t.as_ref().is_none_or(|t| n(t, k));
    if o.surprise() && !(ok(&v.prev, d) && ok(&v.pvalid, 1)) {
        return false;
    }
    if o.mood > 0 && !ok(&v.mood, o.mood) {
        return false;
    }
    if if o.rhythm { !ok(&v.rh, 2) } else { v.rh.is_some() } {
        return false;
    }
    true
}

/// S: block i's spikes s [d] through the synapse from the State's (u, x): (what reaches W_out, the new (u, x)); None
/// without --stp.
#[allow(clippy::type_complexity)]
pub(super) fn stp(m: &Model, st: &State, i: usize, s: &[f32]) -> Result<Option<(Vec<f32>, (Vec<f32>, Vec<f32>))>> {
    let Some(sp) = stp_of(m, i) else { return Ok(None) };
    let Some((u0, x0)) = st.parts.get(i).and_then(|p| p.stp.as_ref()) else { return Ok(None) };
    let (u0, x0) = (vals(u0)?, vals(x0)?);
    let (up, fp, dp, mx) = (ro(&sp.u_param)?, ro(&sp.f_param)?, ro(&sp.d_param)?, ro(&sp.mix)?);
    let (up, fp, dp, mx) = (up.s(), fp.s(), dp.s(), mx.s());
    let d = s.len();
    if u0.len() != d || x0.len() != d || up.len() != d || fp.len() != d || dp.len() != d || mx.len() != d {
        candle_core::bail!("fused step: the synapse's sizes");
    }
    let (mut out, mut un, mut xn) = (vec![0f32; d], vec![0f32; d], vec![0f32; d]);
    for c in 0..d {
        let (uu, df, dd) = (sigmoid(up[c]), sigmoid(fp[c]), sigmoid(dp[c]));
        let sv = s[c];
        // ava_parts::stp::events: k = (s >= 0)(s <= 1) - (s < 0)(s >= -1), s·k + (clamp(|s|, 0, 1) - s·k)
        let f = |b: bool| if b { 1f32 } else { 0f32 };
        let k = f(sv >= 0.0) * f(sv <= 1.0) - f(sv < 0.0) * f(sv >= -1.0);
        let sk = sv * k;
        let a = sv.abs();
        let a = if a < 0.0 { 0.0 } else { a };
        let a = if a > 1.0 { 1.0 } else { a };
        let ev = sk + (a - sk);
        // stp_from's one step
        let (u, x) = (u0[c], x0[c]);
        let u_plus = u + ((u * -1.0 + 1.0) * uu) * ev;
        let e = u_plus * x;
        let eff = e / uu;
        let x_used = x - e * ev;
        un[c] = u_plus * df;
        xn[c] = ((x_used * -1.0 + 1.0) * dd) * -1.0 + 1.0;
        out[c] = sv + ((sv * eff) - sv) * mx[c];
    }
    Ok(Some((out, (un, xn))))
}

/// M: what block 0 leaves for the State (block 0's normed stream, the surprise) and the mood of this token.
/// f4ext: also for DA alone (--dopamine without --mood: the mood is then empty), with z = s / ln V for DA's gates.
pub(super) struct Mood {
    prev: Vec<f32>,
    s: f32,
    z: f32,
    mood: Vec<f32>,
}

impl Mood {
    /// f4ext DA: this token's surprise / ln V (v51::after0's z).
    pub(super) fn z(this: Option<&Mood>) -> Option<f32> {
        this.map(|md| md.z)
    }

    /// After block 0 (x its output as block 1 reads it): the surprise at `tok` and the mood (None without --mood and
    /// --dopamine; f4ext: an empty mood with --dopamine alone).
    pub(super) fn after0(m: &Model, st: &State, x: &[f32], tok: u32, head: &dyn Fn(&[f32]) -> Result<Vec<f32>>) -> Result<Option<Mood>> {
        let o = m.cfg.arch.v51;
        if v51(m).is_none() || !o.surprise() {
            return Ok(None);
        }
        let prev = rmsnorm(x, ro(&m.final_norm)?.s());
        let first = match (st.v51.prev.as_ref(), st.v51.pvalid.as_ref()) {
            (Some(p), Some(valid)) => {
                // lif.rs cross_entropy_rows (its CPU loop) on the head's row for the last xf, times valid
                let row = head(&vals(p)?)?;
                let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let lse = mx + row.iter().map(|&z| (z - mx).exp()).sum::<f32>().ln();
                let t = tok as usize;
                let ce = if t < row.len() { lse - row[t] } else { 0.0 };
                ce * vals(valid)?[0]
            }
            _ => 0.0,
        };
        let z = first * ((1.0 / (m.cfg.vocab as f64).ln()) as f32) + 0.0;
        let k = o.mood;
        let mp = match v51(m).and_then(|v| v.mood.as_ref()) {
            Some(mp) if k > 0 => mp,
            _ => return Ok(Some(Mood { prev, s: first, z, mood: Vec::new() })), // f4ext DA: the surprise alone
        };
        let (lam, win, bin) = (ro(&mp.lam)?, ro(&mp.win)?, ro(&mp.bin)?);
        let (lam, win, bin) = (lam.s(), win.s(), bin.s());
        let m0 = match st.v51.mood.as_ref() {
            Some(t) => vals(t)?,
            None => vec![0f32; k],
        };
        if lam.len() != k || win.len() != k || bin.len() != k || m0.len() != k {
            candle_core::bail!("fused step: the mood's sizes");
        }
        let mood = (0..k)
            .map(|j| {
                let u = z * win[j] + bin[j];
                let ll = log_sigmoid(lam[j]);
                let oml = sigmoid(-lam[j]);
                let p = (0.0 * ll).exp() * 1.0; // λ^(t-j) at t = j = 0, times the triangle's 1
                let mc = p * (u * oml);
                mc + m0[j] * (1.0 * ll).exp()
            })
            .collect();
        Ok(Some(Mood { prev, s: first, z, mood }))
    }

    /// Block i's LIF input current cur [d] times the mood's gain tanh(m G)·1 + 1 (nothing in block 0).
    pub(super) fn gain(this: Option<&Mood>, m: &Model, i: usize, cur: &mut [f32]) -> Result<()> {
        let (Some(md), Some(g)) = (this, v51(m).and_then(|v| v.blocks.get(i)).and_then(|b| b.mood_g.as_ref())) else { return Ok(()) };
        let d = cur.len();
        let mut z = vec![0f32; d];
        gemv(&md.mood, ro(g)?.s(), d, &mut z);
        for (c, zv) in cur.iter_mut().zip(z) {
            *c = *c * (zv.tanh() * 1.0 + 1.0);
        }
        Ok(())
    }
}

/// Pr: block i's MLP units u [hidden] with the cut units at -1e9, and its k (None without --prune: u unchanged).
pub(super) fn prune(m: &Model, i: usize, u: &mut [f32]) -> Result<Option<usize>> {
    let Some(v) = v51(m) else { return Ok(None) };
    let Some((alive, _)) = v.blocks.get(i).and_then(|b| b.prune.as_ref()) else { return Ok(None) };
    {
        let al = ro(alive)?;
        if al.s().len() != u.len() {
            candle_core::bail!("fused step: the pruning mask's size");
        }
        for (uv, &a) in u.iter_mut().zip(al.s()) {
            *uv = *uv + (a * 1e9f64 as f32 + -1e9f64 as f32);
        }
    }
    Ok(Some(crate::v51::kwta_k(m, v, i)?))
}

/// u32 values of a contiguous CPU tensor, read in place.
fn with_u32<R>(t: &Tensor, f: impl FnOnce(&[u32]) -> R) -> Result<R> {
    let (g, l) = t.storage_and_layout();
    let Some((a, b)) = l.contiguous_offsets() else { candle_core::bail!("fused step: a non-contiguous u32 tensor") };
    match &*g {
        Storage::Cpu(CpuStorage::U32(v)) => Ok(f(&v[a..b])),
        _ => candle_core::bail!("fused step: CPU u32 tensors only"),
    }
}

/// CB: her final normed state xf [d] plus the cerebellum's correction for token `tok` (v51b_cb on one row, no online
/// table: the blocker refuses one); xf itself without --cerebellum-head. The chat app's part signal as v51b_cb keeps it.
pub(super) fn cb(m: &Model, xf: Vec<f32>, tok: u32) -> Result<Vec<f32>> {
    let parts = {
        let emb = ro(&m.tok)?;
        cb_parts(m, &xf, tok, emb.s())?
    };
    let Some((corr, online)) = parts else { return Ok(xf) };
    let mut out: Vec<f32> = xf.iter().zip(&corr).map(|(&x, &c)| x + c).collect();
    if let Some(add) = online {
        out.iter_mut().zip(&add).for_each(|(o, &a)| *o = *o + a);
    }
    Ok(out)
}

/// v52 (the chat app's live learning; split out of `cb`, the same bits): the cerebellum's two additions to xf [d] for
/// token `tok`, the read-out's correction and (with an online table, one row) the table's share, with their side
/// effects (the code kept for `cb_learn`, the chat app's part signals); None without --cerebellum-head. `emb`: the
/// tied embedding's values [V, d], passed in so a caller already reading them takes no second read lock. xf + corr,
/// then + the table's share, is `cb`'s value; src/fast4/eprop.rs adds the two on its tape the same way.
pub(crate) fn cb_parts(m: &Model, xf: &[f32], tok: u32, emb: &[f32]) -> Result<Option<(Vec<f32>, Option<Vec<f32>>)>> {
    let Some(e) = m.v51b() else { return Ok(None) };
    let Some(cb) = &e.cb else { return Ok(None) };
    let d = m.cfg.d;
    let t = tok as usize;
    // the granule input u = [xf, rmsnorm(the token's embedding, 1)] [2d]
    let mut u = xf.to_vec();
    u.extend(rmsnorm(&emb[t * d..(t + 1) * d], &vec![1f32; d]));
    let sg = ro(&cb.sgn)?;
    let sg = sg.s();
    let g = sg.len() / crate::v51b::CB_FAN_IN;
    // h_j = sum_q u[idx_jq] sgn_jq (cb_code: the gather, the product, candle's sum over the fan-in from 0)
    let h: Vec<f32> = with_u32(&cb.idx, |idx| {
        if idx.len() != sg.len() || idx.iter().any(|&i| i as usize >= u.len()) {
            return None;
        }
        Some(
            idx.chunks_exact(crate::v51b::CB_FAN_IN)
                .zip(sg.chunks_exact(crate::v51b::CB_FAN_IN))
                .map(|(ix, sv)| {
                    let mut s = 0f32;
                    for (&i, &w) in ix.iter().zip(sv) {
                        s += u[i as usize] * w;
                    }
                    s
                })
                .collect(),
        )
    })?
    .ok_or_else(|| candle_core::Error::Msg("fused step: the cerebellum's wiring".into()))?;
    // the code: kWTA (the k-th largest, as lif.rs's sort), relu, over its length (sum from 0, sqrt, + 1e-6)
    let k = cb.k;
    let kth = if k >= g { None } else { Some(super::super::kth_largest(&h, k)) };
    let a: Vec<f32> = match kth {
        None => h.iter().map(|&v| (v * 1.0).max(0f32)).collect(),
        Some(kth) => h.iter().map(|&v| (v * if v - kth >= 0.0 { 1.0 } else { 0.0 }).max(0f32)).collect(),
    };
    let mut s2 = 0f32;
    for &v in &a {
        s2 += v * v;
    }
    let len = s2.sqrt() * 1.0 + 1e-6f64 as f32;
    // the read-out on the CPU (cb_readout): the winners' rows of W in unit order, each times its code, summed from 0
    let w = ro(&cb.w)?;
    let w = w.s();
    if w.len() != g * d {
        candle_core::bail!("fused step: the cerebellum's read-out size");
    }
    let mut act: Vec<(usize, f32)> = a.iter().enumerate().map(|(j, &v)| (j, v / len)).filter(|&(_, c)| c != 0.0).collect();
    if act.is_empty() {
        act.push((0, 0.0));
    }
    let mut corr = vec![0f32; d];
    for &(j, c) in &act {
        for (o, &wv) in corr.iter_mut().zip(&w[j * d..(j + 1) * d]) {
            *o += wv * c;
        }
    }
    // v51 fast4: the online table (v51b_cb with `cb_online_begin`, one row): + the row's change for this token's code
    // (`CbOnline::add`, its scale), its size for the signals; the code kept for `cb_learn` (one position)
    let mut online_rms = None;
    let mut online = None;
    if let Some(on) = e.cb_online.borrow_mut().as_mut() {
        if on.rows != 1 {
            candle_core::bail!("cerebellum online table for {} rows, a forward of 1", on.rows);
        }
        let a1: Vec<(u32, f32)> = act.iter().filter(|&&(_, c)| c != 0.0).map(|&(j, c)| (j as u32, c)).collect();
        let a1 = vec![a1];
        let add = on.add(&a1, 1, d);
        if e.sig_on.get() {
            online_rms = Some((add.iter().map(|x| x * x).sum::<f32>() / d as f32).sqrt());
        }
        online = Some(add);
        on.last = a1;
        on.last_t = 1;
    }
    if e.sig_on.get() {
        // the size of the correction: sqrt(mean(corr²)) (candle's sum from 0, the mean's affine)
        let mut s = 0f32;
        for &v in &corr {
            s += v * v;
        }
        let r = (s * ((1.0 / d as f64) as f32) + 0.0).sqrt();
        let mut ps = e.sig.borrow_mut();
        ps.cb = Some(r);
        ps.cb_online = online_rms;
    }
    Ok(Some((corr, online)))
}

/// The State moves on: each block's synapse state, then block 0's stream, the surprise and the mood.
/// f4ext: with DA alone (an empty mood) the mood stays as it is (None); `rh`: Y's times after this token (arch_f4.rs).
pub(super) fn commit(st: &mut State, stps: Vec<Option<(Vec<f32>, Vec<f32>)>>, mood: Option<Mood>, rh: Option<&[f32]>, d: usize, dev: &Device) -> Result<()> {
    for (i, n) in stps.into_iter().enumerate() {
        if let Some((u, x)) = n {
            st.parts[i].stp = Some((Tensor::from_vec(u, (1, d), dev)?, Tensor::from_vec(x, (1, d), dev)?));
        }
    }
    if let Some(md) = mood {
        let k = md.mood.len();
        let v = &mut st.v51;
        v.prev = Some(Tensor::from_vec(md.prev, (1, d), dev)?);
        v.pvalid = Some(Tensor::ones(1, DType::F32, dev)?);
        v.last_s = Some(Tensor::from_vec(vec![md.s], 1, dev)?);
        if k > 0 {
            v.mood = Some(Tensor::from_vec(md.mood, (1, k), dev)?);
        }
    }
    if let Some(a) = rh {
        st.v51.rh = Some(Tensor::from_vec(a.to_vec(), (1, a.len()), dev)?);
    }
    Ok(())
}

#[cfg(test)]
#[path = "arch_f3_tests.rs"]
mod tests;
