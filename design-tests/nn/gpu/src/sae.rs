// TESTING / DESIGN ONLY — ava_gpu::sae: A13, looking inside her states. A TopK sparse autoencoder (Gao et al. 2024)
// over one block's state, the recording hook that hands a forward's states out, and (sae::probe) linear probes.
//
// Recording: Model::record_states(true) makes every block-options forward (src/arch.rs, forward_arch) keep each
// block's output state [N, d] (the stream the next block reads, after the DenseFormer mix) and the final normed state
// xf, all detached; Model::take_states() hands the last finished forward's list out (blocks in order, xf last). With
// --feedback the window runs in chunks; the pieces are put back in token order. Off (the default) the two calls in
// forward_arch return at once and the forward is what it was, bit for bit. --modulate's shadow pass is not recorded.
//
// The SAE works on one block's states x, multiplied by one number `scale` so that each dimension has unit variance on
// average (E||x − mean||² = d):
//   z = TopK_k(ReLU(W_enc (x − b_dec) + b_enc)),   x̂ = W_decᵀ z + b_dec
// W_enc and W_dec are stored [latents, d]: row j is latent j's encoder vector and its decoder atom (the paper's decoder
// column). The atoms are kept at unit norm: the part of each atom's gradient along the atom is removed before the
// step and the atom is renormalized after it. b_dec starts at the data mean, W_enc at W_dec (the paper's W_decᵀ).
// Loss = Σ||x − x̂||² / (B·V) (V = the data's variance per token, so the loss is the batch's FVU)
//      + aux_coef · AuxK: the residual e = x − x̂ (detached) rebuilt from the k_aux largest latents among the dead ones
//        (not fired in the last `dead_after` training tokens), ||e − ê||² over the residual's own variance in the batch.
// Adam (0.9, 0.999, 1e-8); the learning rate warms up over the first 1% of steps and falls linearly to 0 over the last
// 20%. The dense encoder product runs on candle's CPU matmul; TopK, the sparse decoder and the sparse gradients run on
// `threads` threads of their own.

use crate::model::Model;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

pub mod probe;
pub mod run;
#[cfg(test)]
mod tests;

// ------------------------------------------------------------------------------------------- the recording hook

/// A13: what a recording forward keeps (ModelExt::state_rec, Some only while Model::record_states is on).
#[derive(Default)]
pub struct StateRec {
    /// per block, the pieces of the forward under way ([b·tc, d] each, one per chunk)
    pieces: Vec<Vec<Tensor>>,
    /// the last finished forward: each block's state [N, d], xf last
    done: Option<Vec<Tensor>>,
}

/// forward_arch: block i's output on this chunk (nothing unless a record is on).
pub(crate) fn rec_block(m: &Model, i: usize, x: &Tensor) -> candle_core::Result<()> {
    let Some(e) = m.ext.as_ref() else { return Ok(()) };
    let mut rec = e.state_rec.borrow_mut();
    let Some(r) = rec.as_mut() else { return Ok(()) };
    if r.pieces.len() <= i {
        r.pieces.resize(i + 1, Vec::new());
    }
    r.pieces[i].push(x.detach());
    Ok(())
}

/// forward_arch: the final normed state of a window of `b` rows; finishes the record of this forward.
pub(crate) fn rec_final(m: &Model, b: usize, xf: &Tensor) -> candle_core::Result<()> {
    let Some(e) = m.ext.as_ref() else { return Ok(()) };
    let mut rec = e.state_rec.borrow_mut();
    let Some(r) = rec.as_mut() else { return Ok(()) };
    let pieces = std::mem::take(&mut r.pieces);
    let mut out = Vec::with_capacity(pieces.len() + 1);
    for mut p in pieces {
        if p.len() == 1 {
            out.push(p.pop().expect("one piece"));
            continue;
        }
        // --feedback: chunks of [b·tc, d] in (row, token) order, back into [b, t, d] order
        let d = p[0].dim(1)?;
        let parts: Vec<Tensor> = p.iter().map(|o| o.reshape((b, o.dim(0)? / b, d))).collect::<candle_core::Result<_>>()?;
        let n: usize = p.iter().map(|o| o.dim(0)).sum::<candle_core::Result<usize>>()?;
        out.push(Tensor::cat(&parts, 1)?.reshape((n, d))?);
    }
    out.push(xf.detach());
    r.done = Some(out);
    Ok(())
}

impl Model {
    /// A13: keep (on) or stop keeping (off, the default) each block's output state and xf in every forward of a model
    /// with block options. Nothing happens for a model without them (it has no block-options forward).
    pub fn record_states(&self, on: bool) {
        if let Some(e) = self.ext.as_ref() {
            *e.state_rec.borrow_mut() = on.then(StateRec::default);
        }
    }

    /// A13: the states of the last forward since recording went on: each block's output [N, d] (detached), xf last.
    /// None when nothing was recorded (recording off, no forward yet, or no block options).
    pub fn take_states(&self) -> Option<Vec<Tensor>> {
        self.ext.as_ref()?.state_rec.borrow_mut().as_mut()?.done.take()
    }
}

// ------------------------------------------------------------------------------------------- small helpers

/// `f` over 0..n split into at most `threads` contiguous ranges, run in parallel; results in order.
pub fn par_ranges<T: Send>(n: usize, threads: usize, f: impl Fn(Range<usize>) -> T + Sync) -> Vec<T> {
    let t = threads.max(1).min(n.max(1));
    if t == 1 {
        return vec![f(0..n)];
    }
    let per = n.div_ceil(t);
    std::thread::scope(|s| {
        let f = &f;
        let hs: Vec<_> = (0..t).map(|i| s.spawn(move || f((i * per).min(n)..((i + 1) * per).min(n)))).collect();
        hs.into_iter().map(|h| h.join().expect("worker thread")).collect()
    })
}

/// Deterministic random numbers (xorshift64*).
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform on [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let u = self.uniform().max(1e-300);
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // eight partial sums, so the compiler can keep them in vector registers
    let mut s = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..8 {
            s[l] += x[l] * y[l];
        }
    }
    let mut t = s.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        t += x * y;
    }
    t
}

#[inline]
fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += a * x;
    }
}

/// Rows `r` [B, d] times the rows of `w` [m, d], transposed: [B, m] (candle's CPU matmul, which uses every thread).
fn mat_rows_t(r: &[f32], b: usize, d: usize, w: &[f32], m: usize) -> Result<Vec<f32>, String> {
    let e = |e: candle_core::Error| e.to_string();
    let x = Tensor::from_slice(r, (b, d), &Device::Cpu).map_err(e)?;
    let w = Tensor::from_slice(w, (m, d), &Device::Cpu).map_err(e)?;
    x.matmul(&w.t().map_err(e)?).map_err(e)?.flatten_all().map_err(e)?.to_vec1::<f32>().map_err(e)
}

/// The k largest of `pre` (indices into it, any order) among `cand` (all when None), written into `idx`.
fn top_k(pre: &[f32], k: usize, cand: Option<&[u32]>, buf: &mut Vec<u32>, idx: &mut [u32]) {
    buf.clear();
    match cand {
        Some(c) => buf.extend_from_slice(c),
        None => buf.extend(0..pre.len() as u32),
    }
    let k = k.min(buf.len());
    if k == 0 {
        return;
    }
    if k < buf.len() {
        buf.select_nth_unstable_by(k - 1, |&a, &b| pre[b as usize].total_cmp(&pre[a as usize]));
    }
    idx[..k].copy_from_slice(&buf[..k]);
}

// ------------------------------------------------------------------------------------------- the SAE

/// A TopK sparse autoencoder over d-dimensional states (see the top of the file).
#[derive(Clone, Debug)]
pub struct Sae {
    pub d: usize,
    pub m: usize,
    pub k: usize,
    /// inputs are multiplied by this before encoding; b_dec and x̂ are in the scaled units
    pub scale: f32,
    /// [m, d]: row j is latent j's encoder vector
    pub w_enc: Vec<f32>,
    pub b_enc: Vec<f32>,
    /// [m, d]: row j is latent j's decoder atom (unit norm)
    pub w_dec: Vec<f32>,
    pub b_dec: Vec<f32>,
}

/// The latents of n inputs: for each, the k chosen latents and their values (ReLU of the pre-activation, so a chosen
/// latent can be 0: it then has not fired).
#[derive(Clone, Debug)]
pub struct Codes {
    pub k: usize,
    pub idx: Vec<u32>,
    pub val: Vec<f32>,
}

impl Codes {
    pub fn len(&self) -> usize {
        self.idx.len() / self.k.max(1)
    }
    pub fn is_empty(&self) -> bool {
        self.idx.is_empty()
    }
    /// Mean number of latents that fired (value > 0) per input.
    pub fn mean_l0(&self) -> f64 {
        self.val.iter().filter(|&&v| v > 0.0).count() as f64 / self.len().max(1) as f64
    }
    /// How many inputs each latent fired on.
    pub fn fire_counts(&self, m: usize) -> Vec<u64> {
        let mut c = vec![0u64; m];
        for (&j, &v) in self.idx.iter().zip(&self.val) {
            if v > 0.0 {
                c[j as usize] += 1;
            }
        }
        c
    }
}

/// Mean and E||x − mean||² of n rows of d.
pub fn mean_and_var(x: &[f32], d: usize) -> (Vec<f32>, f64) {
    let n = x.len() / d;
    let mut mu = vec![0f64; d];
    for r in x.chunks_exact(d) {
        for (m, &v) in mu.iter_mut().zip(r) {
            *m += v as f64;
        }
    }
    mu.iter_mut().for_each(|m| *m /= n.max(1) as f64);
    let mut var = 0f64;
    for r in x.chunks_exact(d) {
        var += r.iter().zip(&mu).map(|(&v, &m)| (v as f64 - m).powi(2)).sum::<f64>();
    }
    (mu.iter().map(|&m| m as f32).collect(), var / n.max(1) as f64)
}

const ENC_BLOCK: usize = 4096;

impl Sae {
    /// A fresh SAE for data `x` (n rows of d): random unit atoms, W_enc = W_dec, b_enc = 0, b_dec = the data mean.
    pub fn init(x: &[f32], d: usize, m: usize, k: usize, seed: u64) -> Result<Sae, String> {
        if d == 0 || m == 0 || k == 0 || k > m || x.len() < d || x.len() % d != 0 {
            return Err(format!("sae init: d {d}, latents {m}, k {k}, {} values", x.len()));
        }
        let (mu, var) = mean_and_var(x, d);
        let scale = if var > 0.0 { (d as f64 / var).sqrt() as f32 } else { 1.0 };
        let mut rng = Rng::new(seed);
        let mut w_dec = vec![0f32; m * d];
        for row in w_dec.chunks_exact_mut(d) {
            for v in row.iter_mut() {
                *v = rng.normal() as f32;
            }
            let n = dot(row, row).sqrt().max(1e-12);
            row.iter_mut().for_each(|v| *v /= n);
        }
        Ok(Sae { d, m, k, scale, w_enc: w_dec.clone(), b_enc: vec![0.0; m], b_dec: mu.iter().map(|&v| v * scale).collect(), w_dec })
    }

    /// Scaled, centred inputs (x·scale − b_dec) of n rows.
    fn centre(&self, x: &[f32]) -> Vec<f32> {
        let mut c = Vec::with_capacity(x.len());
        for r in x.chunks_exact(self.d) {
            c.extend(r.iter().zip(&self.b_dec).map(|(&v, &b)| v * self.scale - b));
        }
        c
    }

    /// Pre-activations [n, m] of centred rows (b_enc added).
    fn pre(&self, xc: &[f32]) -> Result<Vec<f32>, String> {
        let n = xc.len() / self.d;
        let mut p = mat_rows_t(xc, n, self.d, &self.w_enc, self.m)?;
        for r in p.chunks_exact_mut(self.m) {
            r.iter_mut().zip(&self.b_enc).for_each(|(v, b)| *v += b);
        }
        Ok(p)
    }

    /// The codes of n rows of raw states (in blocks of 4096 rows).
    pub fn encode(&self, x: &[f32], threads: usize) -> Result<Codes, String> {
        let (d, m, k) = (self.d, self.m, self.k);
        let mut out = Codes { k, idx: Vec::with_capacity(x.len() / d * k), val: Vec::with_capacity(x.len() / d * k) };
        for blk in x.chunks(ENC_BLOCK * d) {
            let pre = self.pre(&self.centre(blk))?;
            let n = blk.len() / d;
            let parts = par_ranges(n, threads, |rows| {
                let mut buf = Vec::with_capacity(m);
                let mut idx = vec![0u32; rows.len() * k];
                let mut val = vec![0f32; rows.len() * k];
                for (q, r) in rows.enumerate() {
                    let p = &pre[r * m..(r + 1) * m];
                    top_k(p, k, None, &mut buf, &mut idx[q * k..(q + 1) * k]);
                    for s in 0..k {
                        val[q * k + s] = p[idx[q * k + s] as usize].max(0.0);
                    }
                }
                (idx, val)
            });
            for (i, v) in parts {
                out.idx.extend(i);
                out.val.extend(v);
            }
        }
        Ok(out)
    }

    /// x̂ of the codes, in raw units.
    pub fn decode(&self, c: &Codes) -> Vec<f32> {
        let d = self.d;
        let mut out = Vec::with_capacity(c.len() * d);
        let mut row = vec![0f32; d];
        for (ii, vv) in c.idx.chunks_exact(c.k).zip(c.val.chunks_exact(c.k)) {
            row.copy_from_slice(&self.b_dec);
            for (&j, &v) in ii.iter().zip(vv) {
                if v > 0.0 {
                    axpy(&mut row, v, &self.w_dec[j as usize * d..(j as usize + 1) * d]);
                }
            }
            out.extend(row.iter().map(|&v| v / self.scale));
        }
        out
    }

    /// Fraction of variance unexplained on rows `x`: Σ||x − x̂||² / Σ||x − mean(x)||², and the codes.
    pub fn fvu(&self, x: &[f32], threads: usize) -> Result<(f64, Codes), String> {
        let c = self.encode(x, threads)?;
        let xh = self.decode(&c);
        let err: f64 = x.iter().zip(&xh).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum();
        let (_, var) = mean_and_var(x, self.d);
        let tot = var * (x.len() / self.d) as f64;
        Ok((if tot > 0.0 { err / tot } else { 0.0 }, c))
    }

    /// Save as safetensors: w_enc, b_enc, w_dec, b_dec, and scale and k ([1] each).
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let e = |e: candle_core::Error| e.to_string();
        let cpu = Device::Cpu;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert("w_enc".into(), Tensor::from_slice(&self.w_enc, (self.m, self.d), &cpu).map_err(e)?);
        t.insert("b_enc".into(), Tensor::from_slice(&self.b_enc, self.m, &cpu).map_err(e)?);
        t.insert("w_dec".into(), Tensor::from_slice(&self.w_dec, (self.m, self.d), &cpu).map_err(e)?);
        t.insert("b_dec".into(), Tensor::from_slice(&self.b_dec, self.d, &cpu).map_err(e)?);
        t.insert("scale".into(), Tensor::from_slice(&[self.scale], 1, &cpu).map_err(e)?);
        t.insert("k".into(), Tensor::from_slice(&[self.k as f32], 1, &cpu).map_err(e)?);
        candle_core::safetensors::save(&t, path).map_err(|e| format!("save {}: {e}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Sae, String> {
        let t = candle_core::safetensors::load(path, &Device::Cpu).map_err(|e| format!("load {}: {e}", path.display()))?;
        let get = |n: &str| -> Result<Vec<f32>, String> {
            t.get(n).ok_or(format!("{}: no {n}", path.display()))?.flatten_all().and_then(|x| x.to_vec1::<f32>()).map_err(|e| e.to_string())
        };
        let (m, d) = t.get("w_enc").ok_or("no w_enc")?.dims2().map_err(|e| e.to_string())?;
        Ok(Sae { d, m, k: get("k")?[0] as usize, scale: get("scale")?[0], w_enc: get("w_enc")?, b_enc: get("b_enc")?, w_dec: get("w_dec")?, b_dec: get("b_dec")? })
    }
}

// ------------------------------------------------------------------------------------------- training

#[derive(Clone, Debug)]
pub struct TrainCfg {
    pub latents: usize,
    pub k: usize,
    /// AuxK's k (Gao et al.: about d/2); at most the number of dead latents
    pub k_aux: usize,
    pub aux_coef: f32,
    /// a latent is dead when it has not fired in this many training tokens
    pub dead_after: u64,
    pub lr: f32,
    pub batch: usize,
    pub steps: usize,
    pub seed: u64,
    pub threads: usize,
    /// print a line every this many steps (0: never)
    pub log_every: usize,
}

impl TrainCfg {
    pub fn new(d: usize, latents: usize, k: usize, steps: usize) -> TrainCfg {
        TrainCfg { latents, k, k_aux: (d / 2).max(1), aux_coef: 1.0 / 32.0, dead_after: 100_000, lr: 1e-3, batch: 1024, steps, seed: 1, threads: 8, log_every: 0 }
    }
}

/// What training measured.
#[derive(Clone, Debug, Default)]
pub struct TrainLog {
    pub seconds: f64,
    /// training tokens (steps × batch) per second
    pub tokens_per_sec: f64,
    /// the last step's batch FVU and AuxK loss
    pub last_fvu: f64,
    pub last_aux: f64,
    /// dead latents (window rule) at the end of training
    pub dead_end: usize,
    /// latents that were dead (window rule) at some step
    pub ever_dead: usize,
}

/// Adam state of one parameter group.
struct AdamBuf {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamBuf {
    fn new(n: usize) -> AdamBuf {
        AdamBuf { m: vec![0.0; n], v: vec![0.0; n] }
    }
    fn step(&mut self, p: &mut [f32], g: &[f32], lr: f32, t: usize) {
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let c1 = 1.0 - b1.powi(t as i32);
        let c2 = 1.0 - b2.powi(t as i32);
        for i in 0..p.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g[i] * g[i];
            p[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + eps);
        }
    }
}

/// One row range's forward pass of a training step.
struct RowPass {
    idx: Vec<u32>,
    val: Vec<f32>,
    aidx: Vec<u32>,
    aval: Vec<f32>,
    /// x̂ − x
    r: Vec<f32>,
    /// ê − e (e = x − x̂ = −r)
    ra: Vec<f32>,
    sq: f64,
    /// Σ e (for the residual's variance)
    e_sum: Vec<f64>,
}

/// The learning rate at step s of n: warm-up over the first 1%, flat, linear decay to 0 over the last 20%.
pub fn lr_at(lr: f32, s: usize, n: usize) -> f32 {
    let warm = (n / 100).max(1);
    let decay_from = n - n / 5;
    if s < warm {
        lr * (s + 1) as f32 / warm as f32
    } else if s >= decay_from && n > decay_from {
        lr * (n - s) as f32 / (n - decay_from) as f32
    } else {
        lr
    }
}

/// The trainer: the SAE, Adam, and each latent's last firing (in training tokens).
pub struct Trainer {
    pub sae: Sae,
    pub cfg: TrainCfg,
    opt: [AdamBuf; 4],
    t: usize,
    pub tokens_seen: u64,
    pub last_fired: Vec<u64>,
    pub ever_dead: Vec<bool>,
}

impl Trainer {
    pub fn new(sae: Sae, cfg: TrainCfg) -> Trainer {
        let (m, d) = (sae.m, sae.d);
        let opt = [AdamBuf::new(m * d), AdamBuf::new(m), AdamBuf::new(m * d), AdamBuf::new(d)];
        Trainer { sae, cfg, opt, t: 0, tokens_seen: 0, last_fired: vec![0; m], ever_dead: vec![false; m] }
    }

    /// The latents that have not fired in the last `dead_after` training tokens.
    pub fn dead(&self) -> Vec<u32> {
        let after = self.cfg.dead_after;
        (0..self.sae.m as u32).filter(|&j| self.tokens_seen.saturating_sub(self.last_fired[j as usize]) >= after && self.tokens_seen >= after).collect()
    }

    /// One Adam step on `xb` (B rows of raw states); `var` = the data's E||x − mean||² in scaled units (d after
    /// init's scaling). Returns (batch FVU, AuxK loss).
    pub fn step(&mut self, xb: &[f32], var: f64, lr: f32) -> Result<(f64, f64), String> {
        let (d, m, k) = (self.sae.d, self.sae.m, self.sae.k);
        let threads = self.cfg.threads;
        let b = xb.len() / d;
        let dead = self.dead();
        for &j in &dead {
            self.ever_dead[j as usize] = true;
        }
        let ka = self.cfg.k_aux.min(dead.len());
        let sae = &self.sae;
        let xc = sae.centre(xb);
        let pre = sae.pre(&xc)?;
        // pass 1 (rows): TopK, AuxK's TopK among the dead, x̂, ê
        let parts = par_ranges(b, threads, |rows| {
            let nr = rows.len();
            let mut o = RowPass {
                idx: vec![0; nr * k],
                val: vec![0.0; nr * k],
                aidx: vec![0; nr * ka],
                aval: vec![0.0; nr * ka],
                r: vec![0.0; nr * d],
                ra: vec![0.0; nr * d],
                sq: 0.0,
                e_sum: vec![0.0; d],
            };
            let mut buf = Vec::with_capacity(m);
            for (q, row) in rows.enumerate() {
                let p = &pre[row * m..(row + 1) * m];
                let (ii, vv) = (&mut o.idx[q * k..(q + 1) * k], &mut o.val[q * k..(q + 1) * k]);
                top_k(p, k, None, &mut buf, ii);
                let r = &mut o.r[q * d..(q + 1) * d];
                // x̂ − x = b_dec + Σ z W_dec − x·scale = Σ z W_dec − xc
                for (rv, &c) in r.iter_mut().zip(&xc[row * d..(row + 1) * d]) {
                    *rv = -c;
                }
                for s in 0..k {
                    let j = ii[s] as usize;
                    vv[s] = p[j].max(0.0);
                    if vv[s] > 0.0 {
                        axpy(r, vv[s], &sae.w_dec[j * d..(j + 1) * d]);
                    }
                }
                o.sq += r.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
                for (es, &rv) in o.e_sum.iter_mut().zip(r.iter()) {
                    *es -= rv as f64;
                }
                if ka > 0 {
                    let (ai, av) = (&mut o.aidx[q * ka..(q + 1) * ka], &mut o.aval[q * ka..(q + 1) * ka]);
                    top_k(p, ka, Some(&dead), &mut buf, ai);
                    let ra = &mut o.ra[q * d..(q + 1) * d];
                    ra.copy_from_slice(r); // ê − e = ê + r
                    for s in 0..ka {
                        let j = ai[s] as usize;
                        av[s] = p[j].max(0.0);
                        if av[s] > 0.0 {
                            axpy(ra, av[s], &sae.w_dec[j * d..(j + 1) * d]);
                        }
                    }
                }
            }
            o
        });
        let mut sq = 0f64;
        let mut e_sum = vec![0f64; d];
        let mut sqa = 0f64;
        for p in &parts {
            sq += p.sq;
            e_sum.iter_mut().zip(&p.e_sum).for_each(|(a, b)| *a += b);
            if ka > 0 {
                sqa += p.ra.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
            }
        }
        let fvu = sq / (b as f64 * var);
        // the residual's variance in this batch (AuxK's normalizer)
        let e_var = sq - e_sum.iter().map(|s| s * s).sum::<f64>() / b as f64;
        let aux_on = ka > 0 && e_var > 1e-12;
        let aux = if aux_on { sqa / e_var } else { 0.0 };
        let gs = (2.0 / (b as f64 * var)) as f32; // dL/dx̂ = gs · r
        let hs = if aux_on { (self.cfg.aux_coef as f64 * 2.0 / e_var) as f32 } else { 0.0 }; // dL/dê = hs · ra
        // pass 2 (rows): δ = dL/dpre of the chosen latents; b_dec's gradient
        struct Back {
            del: Vec<f32>,
            adel: Vec<f32>,
            gbd: Vec<f32>,
        }
        let mut row0 = 0usize;
        let starts: Vec<usize> = parts
            .iter()
            .map(|p| {
                let s = row0;
                row0 += p.r.len() / d;
                s
            })
            .collect();
        let backs: Vec<Back> = std::thread::scope(|s| {
            let hs_ = parts
                .iter()
                .map(|p| {
                    s.spawn(move || {
                        let nr = p.r.len() / d;
                        let mut o = Back { del: vec![0.0; nr * k], adel: vec![0.0; nr * ka], gbd: vec![0.0; d] };
                        for q in 0..nr {
                            let g = &p.r[q * d..(q + 1) * d];
                            for (bd, &gv) in o.gbd.iter_mut().zip(g) {
                                *bd += gs * gv;
                            }
                            for s in 0..k {
                                if p.val[q * k + s] > 0.0 {
                                    let j = p.idx[q * k + s] as usize;
                                    let dl = gs * dot(&sae.w_dec[j * d..(j + 1) * d], g);
                                    o.del[q * k + s] = dl;
                                    axpy(&mut o.gbd, -dl, &sae.w_enc[j * d..(j + 1) * d]);
                                }
                            }
                            if aux_on {
                                let h = &p.ra[q * d..(q + 1) * d];
                                for s in 0..ka {
                                    if p.aval[q * ka + s] > 0.0 {
                                        let j = p.aidx[q * ka + s] as usize;
                                        o.adel[q * ka + s] = hs * dot(&sae.w_dec[j * d..(j + 1) * d], h);
                                    }
                                }
                            }
                        }
                        o
                    })
                })
                .collect::<Vec<_>>();
            hs_.into_iter().map(|h| h.join().expect("worker thread")).collect()
        });
        // pass 3 (latent ranges): the weight gradients
        let mut g_we = vec![0f32; m * d];
        let mut g_be = vec![0f32; m];
        let mut g_wd = vec![0f32; m * d];
        let per = m.div_ceil(threads.max(1).min(m));
        std::thread::scope(|s| {
            for (ci, ((we, be), wd)) in g_we.chunks_mut(per * d).zip(g_be.chunks_mut(per)).zip(g_wd.chunks_mut(per * d)).enumerate() {
                let (parts, backs, starts, xc) = (&parts, &backs, &starts, &xc);
                s.spawn(move || {
                    let lo = ci * per;
                    let hi = lo + be.len();
                    for ((p, bk), &r0) in parts.iter().zip(backs).zip(starts) {
                        let nr = p.r.len() / d;
                        for q in 0..nr {
                            let x = &xc[(r0 + q) * d..(r0 + q + 1) * d];
                            for s in 0..k {
                                let j = p.idx[q * k + s] as usize;
                                let v = p.val[q * k + s];
                                if v > 0.0 && j >= lo && j < hi {
                                    let l = j - lo;
                                    axpy(&mut wd[l * d..(l + 1) * d], gs * v, &p.r[q * d..(q + 1) * d]);
                                    let dl = bk.del[q * k + s];
                                    axpy(&mut we[l * d..(l + 1) * d], dl, x);
                                    be[l] += dl;
                                }
                            }
                            if aux_on {
                                for s in 0..ka {
                                    let j = p.aidx[q * ka + s] as usize;
                                    let v = p.aval[q * ka + s];
                                    if v > 0.0 && j >= lo && j < hi {
                                        let l = j - lo;
                                        axpy(&mut wd[l * d..(l + 1) * d], hs * v, &p.ra[q * d..(q + 1) * d]);
                                        let dl = bk.adel[q * ka + s];
                                        axpy(&mut we[l * d..(l + 1) * d], dl, x);
                                        be[l] += dl;
                                    }
                                }
                            }
                        }
                    }
                });
            }
        });
        let mut g_bd = vec![0f32; d];
        for bk in &backs {
            g_bd.iter_mut().zip(&bk.gbd).for_each(|(a, b)| *a += b);
        }
        // the atoms stay unit: drop each atom's gradient along itself, step, renormalize
        let sae = &mut self.sae;
        for (g, w) in g_wd.chunks_exact_mut(d).zip(sae.w_dec.chunks_exact(d)) {
            let a = dot(g, w);
            axpy(g, -a, w);
        }
        self.t += 1;
        let t = self.t;
        let [o_we, o_be, o_wd, o_bd] = &mut self.opt;
        o_we.step(&mut sae.w_enc, &g_we, lr, t);
        o_be.step(&mut sae.b_enc, &g_be, lr, t);
        o_wd.step(&mut sae.w_dec, &g_wd, lr, t);
        o_bd.step(&mut sae.b_dec, &g_bd, lr, t);
        for w in sae.w_dec.chunks_exact_mut(d) {
            let n = dot(w, w).sqrt().max(1e-12);
            w.iter_mut().for_each(|v| *v /= n);
        }
        self.tokens_seen += b as u64;
        for p in &parts {
            for (&j, &v) in p.idx.iter().zip(&p.val) {
                if v > 0.0 {
                    self.last_fired[j as usize] = self.tokens_seen;
                }
            }
        }
        Ok((fvu, aux))
    }
}

/// Train an SAE on rows `x` (n × d raw states) for cfg.steps steps of cfg.batch rows (each epoch a new shuffle).
/// `say` gets a progress line every cfg.log_every steps.
pub fn train(x: &[f32], d: usize, cfg: &TrainCfg, mut say: impl FnMut(&str)) -> Result<(Sae, TrainLog), String> {
    let n = x.len() / d;
    if n == 0 {
        return Err("sae train: no data".into());
    }
    let sae = Sae::init(x, d, cfg.latents, cfg.k, cfg.seed)?;
    let var = {
        let (_, v) = mean_and_var(x, d);
        v * (sae.scale as f64).powi(2)
    };
    let mut tr = Trainer::new(sae, cfg.clone());
    let mut rng = Rng::new(cfg.seed ^ 0x5AE);
    let mut order: Vec<u32> = (0..n as u32).collect();
    rng.shuffle(&mut order);
    let mut at = 0usize;
    let bsz = cfg.batch.min(n).max(1);
    let mut xb = vec![0f32; bsz * d];
    let t0 = std::time::Instant::now();
    let (mut fvu, mut aux) = (0.0, 0.0);
    for s in 0..cfg.steps {
        for q in 0..bsz {
            if at == n {
                rng.shuffle(&mut order);
                at = 0;
            }
            let r = order[at] as usize;
            at += 1;
            xb[q * d..(q + 1) * d].copy_from_slice(&x[r * d..(r + 1) * d]);
        }
        (fvu, aux) = tr.step(&xb, var, lr_at(cfg.lr, s, cfg.steps))?;
        if cfg.log_every > 0 && (s % cfg.log_every == 0 || s + 1 == cfg.steps) {
            say(&format!(
                "  step {s}/{}: batch FVU {fvu:.4}, AuxK {aux:.4}, dead {} ({:.1} s)",
                cfg.steps,
                tr.dead().len(),
                t0.elapsed().as_secs_f64()
            ));
        }
    }
    let seconds = t0.elapsed().as_secs_f64();
    let log = TrainLog {
        seconds,
        tokens_per_sec: (cfg.steps * bsz) as f64 / seconds.max(1e-9),
        last_fvu: fvu,
        last_aux: aux,
        dead_end: tr.dead().len(),
        ever_dead: tr.ever_dead.iter().filter(|&&x| x).count(),
    };
    Ok((tr.sae, log))
}
