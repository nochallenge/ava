// TESTING / DESIGN ONLY — ava_grad::brain
// The two brain-like primitives as plain forward/backward kernels (Graph wraps them as tape nodes):
//
// k-winners-take-all (kwta): per row of the last axis keep the k largest entries, zero the rest.
//   Ties are broken deterministically by the lower index (sort key: value descending, then index
//   ascending, using f32::total_cmp). Backward passes the gradient through the kept entries only.
//
// Leaky integrate-and-fire scan (lif): input current I [B, T, D], per-channel leak parameter p [D].
//   β_d = σ(p_d) ∈ (0,1);  v_0 = 0, s_0 = 0;
//   v_t = β·v_{t−1} + I_t − θ·s_{t−1}     (soft reset by subtraction)
//   s_t = f(v_t − θ)
//   f = Heaviside step (s = 1 iff v ≥ θ) for SpikeFn::Surrogate, with backward surrogate
//       f'(x) = 1/(1 + slope·|x|)²  (fast sigmoid), or f = σ(slope·x) for SpikeFn::Sigmoid with its exact
//       derivative (smooth, so the whole BPTT path can be finite-difference checked).
//   Backward is full backprop-through-time inside one op. With δv_{T+1} = 0, going t = T..1:
//       δs_t = g_t − θ·δv_{t+1}   (the −θ term is dropped when detach_reset is true)
//       δv_t = δs_t·f'(v_t − θ) + β·δv_{t+1}
//       dI_t = δv_t,   dβ = Σ_t δv_t·v_{t−1},   dp = dβ·β(1−β)
//   detach_reset (default true, as in Zenke & Vogels 2021 / SpyTorch) treats the reset spike as a
//   constant in the backward pass; set it false for the exact gradient of the forward function.

use crate::matmul::par_rows2;
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpikeFn {
    /// Hard step forward, fast-sigmoid surrogate derivative 1/(1 + slope·|v − θ|)² backward.
    Surrogate { slope: f32 },
    /// Smooth σ(slope·(v − θ)) forward and its exact derivative backward.
    Sigmoid { slope: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LifOpts {
    pub threshold: f32,
    pub spike_fn: SpikeFn,
    /// Treat the reset term θ·s_{t−1} as a constant in backward (default true).
    pub detach_reset: bool,
}

impl LifOpts {
    pub fn new(threshold: f32, spike_fn: SpikeFn) -> LifOpts {
        LifOpts { threshold, spike_fn, detach_reset: true }
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Spike value for membrane-minus-threshold `x`.
#[inline]
pub fn spike(x: f32, f: SpikeFn) -> f32 {
    match f {
        SpikeFn::Surrogate { .. } => {
            if x >= 0.0 {
                1.0
            } else {
                0.0
            }
        }
        SpikeFn::Sigmoid { slope } => sigmoid(slope * x),
    }
}

/// Derivative used in backward for membrane-minus-threshold `x`.
#[inline]
pub fn spike_grad(x: f32, f: SpikeFn) -> f32 {
    match f {
        SpikeFn::Surrogate { slope } => {
            let a = 1.0 + slope * x.abs();
            1.0 / (a * a)
        }
        SpikeFn::Sigmoid { slope } => {
            let s = sigmoid(slope * x);
            slope * s * (1.0 - s)
        }
    }
}

/// LIF forward. Returns (spikes, membrane v), both [B, T, D]. Single-threaded; see `lif_forward_par`.
pub fn lif_forward(cur: &[f32], beta_p: &[f32], b: usize, t: usize, d: usize, opts: &LifOpts) -> (Vec<f32>, Vec<f32>) {
    lif_forward_par(cur, beta_p, b, t, d, opts, 1)
}

/// LIF backward (BPTT). `g` is dL/dspikes [B, T, D]; `v` the saved membrane. Returns (dI, d beta_param).
/// Single-threaded; see `lif_backward_par`.
pub fn lif_backward(g: &[f32], v: &[f32], beta_p: &[f32], b: usize, t: usize, d: usize, opts: &LifOpts) -> (Vec<f32>, Vec<f32>) {
    lif_backward_par(g, v, beta_p, b, t, d, opts, 1)
}

/// Minimum (batch × time × channel) cells per thread before the LIF scan spawns another thread.
const LIF_MIN_PER_THREAD: usize = 1 << 16;
/// Channels per LIF task at least (keeps the per-step inner loop wide enough to vectorise).
const LIF_MIN_CHANNELS: usize = 16;

/// How a LIF scan over [B, T, D] is split across threads: `tb` batch groups of `bper` batches × `tc`
/// channel blocks of `cw` channels (task index = group · tc + block). Every (batch, channel) pair's time
/// scan runs whole inside one task, in the same order as on one thread, so the split never changes a bit.
struct LifSplit {
    tb: usize,
    bper: usize,
    tc: usize,
    cw: usize,
}

impl LifSplit {
    fn new(b: usize, t: usize, d: usize, threads: usize) -> LifSplit {
        let nt = threads.max(1).min((b * t * d / LIF_MIN_PER_THREAD).max(1));
        let tb = nt.min(b).max(1);
        let tc = (nt / tb).min(d.div_ceil(LIF_MIN_CHANNELS)).max(1);
        LifSplit { tb, bper: b.div_ceil(tb).max(1), tc, cw: d.div_ceil(tc).max(1) }
    }

    /// Width of channel block `c` (the last block may be narrower, or empty).
    fn width(&self, c: usize, d: usize) -> usize {
        self.cw.min(d.saturating_sub(c * self.cw))
    }
}

/// Split `buf` ([B, T, D] row-major) into one list of row pieces per task: the task's channel block of
/// every (batch, time) row in its batch group, in row order. Safe disjoint borrows, no copying.
fn lif_pieces<'a>(buf: &'a mut [f32], t: usize, d: usize, sp: &LifSplit) -> Vec<Vec<&'a mut [f32]>> {
    let mut tasks: Vec<Vec<&'a mut [f32]>> = (0..sp.tb * sp.tc).map(|_| Vec::with_capacity(sp.bper * t)).collect();
    for (row, mut rest) in buf.chunks_mut(d).enumerate() {
        let bg = row / t / sp.bper;
        for c in 0..sp.tc {
            let w = sp.cw.min(rest.len());
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(w);
            rest = tail;
            tasks[bg * sp.tc + c].push(head);
        }
    }
    tasks
}

/// `lif_forward` with up to `threads` worker threads over (batch, channel) blocks; bit-identical for any
/// thread count.
#[allow(clippy::too_many_arguments)]
pub fn lif_forward_par(cur: &[f32], beta_p: &[f32], b: usize, t: usize, d: usize, opts: &LifOpts, threads: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(cur.len(), b * t * d);
    assert_eq!(beta_p.len(), d);
    let beta: Vec<f32> = beta_p.iter().map(|&p| sigmoid(p)).collect();
    let mut spikes = vec![0.0f32; cur.len()];
    let mut v = vec![0.0f32; cur.len()];
    if cur.is_empty() {
        return (spikes, v);
    }
    let sp = LifSplit::new(b, t, d, threads);
    let (beta, th, f) = (&beta[..], opts.threshold, opts.spike_fn);
    // one task: rows = this task's pieces (nb batches × t steps, in order), each `w` channels from d0
    let scan = |task: usize, srows: &mut [&mut [f32]], vrows: &mut [&mut [f32]]| {
        let (bg, c) = (task / sp.tc, task % sp.tc);
        let d0 = c * sp.cw;
        let row0 = bg * sp.bper * t;
        let w = srows.first().map_or(0, |r| r.len());
        let mut vprev = vec![0.0f32; w];
        let mut sprev = vec![0.0f32; w];
        for (i, (srow, vrow)) in srows.iter_mut().zip(vrows.iter_mut()).enumerate() {
            if i % t == 0 {
                vprev.fill(0.0);
                sprev.fill(0.0);
            }
            let base = (row0 + i) * d + d0;
            let crow = &cur[base..base + w];
            for j in 0..w {
                let vt = beta[d0 + j] * vprev[j] + crow[j] - th * sprev[j];
                let st = spike(vt - th, f);
                vrow[j] = vt;
                srow[j] = st;
                vprev[j] = vt;
                sprev[j] = st;
            }
        }
    };
    let mut st = lif_pieces(&mut spikes, t, d, &sp);
    let mut vt = lif_pieces(&mut v, t, d, &sp);
    if st.len() == 1 {
        scan(0, &mut st[0], &mut vt[0]);
    } else {
        let scan = &scan;
        std::thread::scope(|s| {
            for (task, (sr, vr)) in st.iter_mut().zip(vt.iter_mut()).enumerate() {
                s.spawn(move || scan(task, sr, vr));
            }
        });
    }
    (spikes, v)
}

/// `lif_backward` with up to `threads` worker threads. dβ is summed per (batch, channel) inside the
/// task and the per-batch partials are then added in batch order, so the result is independent of the
/// split (and of the thread count).
#[allow(clippy::too_many_arguments)]
pub fn lif_backward_par(g: &[f32], v: &[f32], beta_p: &[f32], b: usize, t: usize, d: usize, opts: &LifOpts, threads: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(g.len(), b * t * d);
    assert_eq!(v.len(), b * t * d);
    assert_eq!(beta_p.len(), d);
    let beta: Vec<f32> = beta_p.iter().map(|&p| sigmoid(p)).collect();
    let mut dcur = vec![0.0f32; g.len()];
    let mut dbeta = vec![0.0f64; d];
    if !g.is_empty() {
        let sp = LifSplit::new(b, t, d, threads);
        let (th, f, detach) = (opts.threshold, opts.spike_fn, opts.detach_reset);
        let beta = &beta[..];
        // one task: writes its dcur pieces, returns dβ partials laid out [local batch][channel in block]
        let scan = |task: usize, rows: &mut [&mut [f32]]| -> Vec<f64> {
            let (bg, c) = (task / sp.tc, task % sp.tc);
            let d0 = c * sp.cw;
            let row0 = bg * sp.bper * t;
            let w = rows.first().map_or(0, |r| r.len());
            let nb = rows.len() / t;
            let mut part = vec![0.0f64; nb * w];
            let mut dv_next = vec![0.0f32; w];
            for lb in 0..nb {
                dv_next.fill(0.0);
                for ti in (0..t).rev() {
                    let i = lb * t + ti;
                    let base = (row0 + i) * d + d0;
                    let drow = &mut rows[i];
                    for j in 0..w {
                        let vt = v[base + j];
                        let fp = spike_grad(vt - th, f);
                        let mut ds = g[base + j];
                        if !detach {
                            ds -= th * dv_next[j];
                        }
                        let dv = ds * fp + beta[d0 + j] * dv_next[j];
                        drow[j] = dv;
                        if ti > 0 {
                            part[lb * w + j] += dv as f64 * v[base - d + j] as f64;
                        }
                        dv_next[j] = dv;
                    }
                }
            }
            part
        };
        let mut tasks = lif_pieces(&mut dcur, t, d, &sp);
        let parts: Vec<Vec<f64>> = if tasks.len() == 1 {
            vec![scan(0, &mut tasks[0])]
        } else {
            let scan = &scan;
            std::thread::scope(|s| {
                let handles: Vec<_> = tasks.iter_mut().enumerate().map(|(task, rows)| s.spawn(move || scan(task, rows))).collect();
                handles.into_iter().map(|h| h.join().expect("lif_backward_par: task panicked")).collect()
            })
        };
        // combine the per-batch partials in batch order, whatever the split
        for bi in 0..b {
            let (bg, lb) = (bi / sp.bper, bi % sp.bper);
            for c in 0..sp.tc {
                let (d0, w) = (c * sp.cw, sp.width(c, d));
                let part = &parts[bg * sp.tc + c];
                for j in 0..w {
                    dbeta[d0 + j] += part[lb * w + j];
                }
            }
        }
    }
    let dbeta_p = dbeta.iter().zip(&beta).map(|(&db, &bt)| (db * bt as f64 * (1.0 - bt as f64)) as f32).collect();
    (dcur, dbeta_p)
}

/// kWTA forward over rows of length `d`. Returns (output, keep-mask). Single-threaded; see `kwta_forward_par`.
pub fn kwta_forward(x: &[f32], d: usize, k: usize) -> (Vec<f32>, Vec<bool>) {
    kwta_forward_par(x, d, k, 1)
}

/// `kwta_forward` with up to `threads` worker threads, rows split between them (each row's top-k is
/// independent, so the result is bit-identical for any thread count).
pub fn kwta_forward_par(x: &[f32], d: usize, k: usize, threads: usize) -> (Vec<f32>, Vec<bool>) {
    let mut out = vec![0.0f32; x.len()];
    let mut mask = vec![false; x.len()];
    if d == 0 || x.is_empty() {
        return (out, mask);
    }
    par_rows2(&mut out, d, &mut mask, d, d * 8, threads, |r0, orows, mrows| {
        let mut idx: Vec<usize> = Vec::with_capacity(d);
        for (ri, (orow, mrow)) in orows.chunks_mut(d).zip(mrows.chunks_mut(d)).enumerate() {
            let r = r0 + ri;
            let row = &x[r * d..(r + 1) * d];
            if k >= d {
                orow.copy_from_slice(row);
                mrow.fill(true);
                continue;
            }
            if k == 0 {
                continue;
            }
            idx.clear();
            idx.extend(0..d);
            // total order: larger value first, then lower index first
            idx.select_nth_unstable_by(k - 1, |&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            for &j in &idx[..k] {
                orow[j] = row[j];
                mrow[j] = true;
            }
        }
    });
    (out, mask)
}

/// Fraction of entries that are non-zero (e.g. the activity level of a kWTA code or a spike train).
pub fn fraction_active(t: &Tensor) -> f32 {
    if t.data.is_empty() {
        return 0.0;
    }
    t.data.iter().filter(|&&v| v != 0.0).count() as f32 / t.data.len() as f32
}
