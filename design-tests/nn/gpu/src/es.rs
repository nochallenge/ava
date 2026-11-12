// TESTING / DESIGN ONLY — M7: evolution strategies with low-rank perturbations (EGGROLL-style), a gradient-free tuner.
//
// After training: take a trained model, search over a chosen subset of its weights with a fitness that needs no
// gradient (exact spikes, int8, the owner's right − wrong rule with "I don't know" at 0), and keep the better θ.
// H1 (a learned learning rule) is this pointed at the memory write-rule weights (`H1_PARAMS`, src/bin/es_tune.rs).
//
// * Selection by name (`select`): a comma list of patterns, each a prefix in dot-separated parts; `*` matches any run
//   of characters inside one part. `b*.parts.holo.lam` is every block's holographic trace rate, `b3.mlp` every weight
//   of block 3's MLP, `b1` block 1 only (not b10). A pattern that matches nothing is an error (catches typos);
//   `h1` (es_tune) expands to the H1 names the model has.
// * Noise for member k of generation g, from derive_seed(seed, g, k) (ava_data::rng, Box–Muller on SplitMix64), with
//   its own stream per selected tensor p, derive_seed(that, p, NOISE_TAG), so one tensor's noise is regenerated
//   without the others':
//     a 2-D tensor [m, n] gets E = A·Bᵀ/√r, A [m, r] and B [n, r] standard normal (rank r; every entry has variance 1),
//     any other shape gets full standard normal noise.
//   Antithetic pairs: θ + σE and θ − σE. Every evaluation sets the weights from the stored centre θ and puts them back
//   exactly afterwards (Var::set copies the stored bits), even when the fitness fails.
// * Fitness shaping: centred ranks over the 2·pop evaluations (ties share their mean rank; NaN counts as the worst),
//   u in [−0.5, 0.5]; a constant fitness gives every member u = 0 and so no update at all.
// * Update: θ ← θ + lr/(pop·σ) · Σ_k (u_k⁺ − u_k⁻)/2 · E_k. The noise is never stored: E_k is regenerated from its seed,
//   and for a 2-D tensor the sum is one product of the stacked factors, [c_1·A_1 | … | c_pop·A_pop] · [B_1 | … | B_pop]ᵀ/√r
//   (EGGROLL's trick: the update is a rank pop·r matrix). The noise and the update are made on the host in f64, so the
//   same seed gives the same E on the CPU and on CUDA.
// * Step size: σ is per entry. A rank-1 E on a d×d matrix has spectral norm about σ·d, so for big matrices use a much
//   smaller σ (e.g. 0.001) than for the gains, rates and biases of a write rule (0.01 is fine there). Each generation
//   reports the step's size relative to |θ|.
// * Plain update, no Adam (the rank-pop·r form would be lost); no weight decay.
//
// The fitness is any `FnMut(&Model) -> Result<f64, String>` (higher is better); src/es/tune.rs is es_tune's half
// (dev items split into tune and check, rmw / acc / nll fitness from mc::grade's scoring).

use crate::model::Model;
use ava_data::rng::{derive_seed, SplitMix64};
use candle_core::{DType, Tensor, Var};
use candle_nn::VarMap;
use std::time::Instant;

pub mod tune;

#[cfg(test)]
mod tests;

/// The label of the per-tensor noise streams.
pub const NOISE_TAG: u64 = 0xE5_6E66_0011;

/// H1: the memory write-rule weights of this codebase (src/model.rs, src/arch.rs, src/memory.rs, src/wave5.rs and the
/// parts crate), as selection patterns. A model has only the ones its kind and block options build; es_tune's
/// `--params h1` keeps those (see src/bin/es_tune.rs for what each one does).
pub const H1_PARAMS: &[&str] = &[
    // fast-weight memory (G, R, D, A): forget alpha = sigmoid(h·Wa + ba + logit γ_h), write beta = 2·sigmoid(h·Wb + bb)
    "b*.hebb.wa",
    "b*.hebb.ba",
    "b*.hebb.wb",
    "b*.hebb.bb",
    // P2 --gates channel: per-channel forget and write (low-rank) and the erase key
    "b*.hebb.wa1",
    "b*.hebb.wa2",
    "b*.hebb.wb1",
    "b*.hebb.wb2",
    "b*.hebb.we",
    // L3 --btsp: the engram's plateau-gated, window-bound write
    "b*.eng.wp",
    "b*.eng.bp",
    "b*.eng.btsp_g",
    "b*.eng.btsp_mix",
    // D2 --holo: the holographic trace's fade λ = sigmoid(lam)
    "b*.parts.holo.lam",
    // B10 --tem: the moves path-integrated into the structural code every token is written under
    "b*.parts.tem.wth",
    // B9 --wm and B3 --workspace: the slots' write gates (and the workspace's overall write strength)
    "b*.parts.wm.wg",
    "b*.parts.wm.bg",
    "b*.parts.ws.wg",
    "b*.parts.ws.bg",
    "b*.parts.ws.ws",
    // B8 --modulate: surprise's gains on the write strength, the forget push and the input
    "b*.parts.mod.gains",
    // E9 --stp: short-term facilitation and depression of the synapses
    "b*.parts.stp.u_param",
    "b*.parts.stp.f_param",
    "b*.parts.stp.d_param",
];

// ------------------------------------------------------------------------------------------------ selection

/// `*` matches any run of characters (possibly none); every other character matches itself.
fn glob(p: &[u8], s: &[u8]) -> bool {
    let (mut pi, mut si, mut star, mut mark) = (0usize, 0usize, None::<usize>, 0usize);
    while si < s.len() {
        if pi < p.len() && p[pi] == b'*' {
            star = Some(pi);
            mark = si;
            pi += 1;
        } else if pi < p.len() && p[pi] == s[si] {
            pi += 1;
            si += 1;
        } else if let Some(st) = star {
            pi = st + 1;
            mark += 1;
            si = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

/// Whether `name` starts with the dot-separated parts of `pattern` (each part matched whole, `*` inside one part).
pub fn name_matches(pattern: &str, name: &str) -> bool {
    let pp: Vec<&str> = pattern.split('.').collect();
    let np: Vec<&str> = name.split('.').collect();
    pp.len() <= np.len() && pp.iter().zip(&np).all(|(p, n)| glob(p.as_bytes(), n.as_bytes()))
}

/// The names matched by a comma list of patterns, sorted, each once. Every pattern must match something.
pub fn select_names<'a>(names: impl IntoIterator<Item = &'a str>, patterns: &str) -> Result<Vec<String>, String> {
    let pats: Vec<&str> = patterns.split(',').map(str::trim).filter(|p| !p.is_empty()).collect();
    if pats.is_empty() {
        return Err("no parameter patterns given".into());
    }
    let names: Vec<&str> = names.into_iter().collect();
    let mut out: Vec<String> = Vec::new();
    for p in &pats {
        let hit: Vec<&str> = names.iter().copied().filter(|n| name_matches(p, n)).collect();
        if hit.is_empty() {
            return Err(format!("parameter pattern {p:?} matches no weight of the model"));
        }
        out.extend(hit.into_iter().map(String::from));
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// The patterns of `H1_PARAMS` that match at least one of `names`, as a comma list (Err when none does).
pub fn h1_patterns<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<String, String> {
    let names: Vec<&str> = names.into_iter().collect();
    let have: Vec<&str> = H1_PARAMS.iter().copied().filter(|p| names.iter().any(|n| name_matches(p, n))).collect();
    if have.is_empty() {
        return Err("the model has none of the H1 write-rule weights (src/es.rs H1_PARAMS)".into());
    }
    Ok(have.join(","))
}

/// The variables of `varmap` that `patterns` selects, sorted by name.
pub fn select(varmap: &VarMap, patterns: &str) -> Result<Vec<(String, Var)>, String> {
    let data = varmap.data().lock().unwrap();
    let names = select_names(data.keys().map(String::as_str), patterns)?;
    Ok(names.into_iter().map(|n| (n.clone(), data[&n].clone())).collect())
}

// ------------------------------------------------------------------------------------------------ noise

/// A uniform number in (0, 1) (never 0, so its log is finite).
fn open01(rng: &mut SplitMix64) -> f64 {
    ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// `n` standard normal numbers (Box–Muller, both outputs of each pair used).
pub fn gaussian(rng: &mut SplitMix64, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u1, u2) = (open01(rng), open01(rng));
        let r = (-2.0 * u1.ln()).sqrt();
        let a = std::f64::consts::TAU * u2;
        out.push(r * a.cos());
        out.push(r * a.sin());
    }
    out.truncate(n);
    out
}

/// One tensor's perturbation direction for one member.
#[derive(Clone, Debug, PartialEq)]
pub enum Noise {
    /// E = A·Bᵀ/√r, A [m, r] and B [n, r] row-major
    LowRank { m: usize, n: usize, r: usize, a: Vec<f64>, b: Vec<f64> },
    /// full standard normal noise, as many entries as the tensor
    Full(Vec<f64>),
}

impl Noise {
    /// The noise of a tensor of shape `dims` from its stream seed (`stream_seed`).
    pub fn new(seed: u64, dims: &[usize], rank: usize) -> Noise {
        let mut rng = SplitMix64::new(seed);
        if dims.len() == 2 {
            let (m, n, r) = (dims[0], dims[1], rank.max(1));
            let a = gaussian(&mut rng, m * r);
            let b = gaussian(&mut rng, n * r);
            Noise::LowRank { m, n, r, a, b }
        } else {
            Noise::Full(gaussian(&mut rng, dims.iter().product()))
        }
    }

    /// E as a dense row-major array.
    pub fn dense(&self) -> Vec<f64> {
        match self {
            Noise::Full(e) => e.clone(),
            Noise::LowRank { m, n, r, a, b } => {
                let s = 1.0 / (*r as f64).sqrt();
                let mut e = vec![0f64; m * n];
                for i in 0..*m {
                    let ai = &a[i * r..(i + 1) * r];
                    for j in 0..*n {
                        let bj = &b[j * r..(j + 1) * r];
                        e[i * n + j] = ai.iter().zip(bj).map(|(x, y)| x * y).sum::<f64>() * s;
                    }
                }
                e
            }
        }
    }
}

/// The seed of member k of generation g.
pub fn member_seed(seed: u64, g: u64, k: usize) -> u64 {
    derive_seed(seed, g, k as u64)
}

/// The noise stream of selected tensor p within a member.
pub fn stream_seed(member: u64, p: usize) -> u64 {
    derive_seed(member, p as u64, NOISE_TAG)
}

// ------------------------------------------------------------------------------------------------ fitness shaping

/// Centred ranks in [−0.5, 0.5] (lowest fitness −0.5, highest +0.5); tied values share their mean rank, NaN is the
/// worst. One value gives 0.
pub fn centred_ranks(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let key = |x: f64| if x.is_nan() { f64::NEG_INFINITY } else { x };
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| key(f[a]).total_cmp(&key(f[b])));
    let mut rank = vec![0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && key(f[idx[j + 1]]) == key(f[idx[i]]) {
            j += 1;
        }
        let mean = (i + j) as f64 / 2.0;
        for &q in &idx[i..=j] {
            rank[q] = mean;
        }
        i = j + 1;
    }
    rank.iter().map(|r| r / (n - 1) as f64 - 0.5).collect()
}

/// (u_k⁺ − u_k⁻)/2 per pair, the ranks taken over all 2·pop evaluations.
pub fn pair_coeffs(plus: &[f64], minus: &[f64]) -> Vec<f64> {
    assert_eq!(plus.len(), minus.len(), "one minus per plus");
    let all: Vec<f64> = plus.iter().chain(minus).copied().collect();
    let u = centred_ranks(&all);
    let p = plus.len();
    (0..p).map(|k| (u[k] - u[p + k]) / 2.0).collect()
}

// ------------------------------------------------------------------------------------------------ the search

/// The search's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EsOpts {
    /// antithetic pairs per generation (2·pop evaluations)
    pub pop: usize,
    /// the noise's spread per entry
    pub sigma: f64,
    pub lr: f64,
    /// the rank of a 2-D tensor's noise
    pub rank: usize,
    pub seed: u64,
}

impl Default for EsOpts {
    fn default() -> EsOpts {
        EsOpts { pop: 8, sigma: 0.01, lr: 0.01, rank: 1, seed: 1 }
    }
}

/// What one generation did.
#[derive(Clone, Debug)]
pub struct GenReport {
    pub gen: u64,
    /// fitness of θ + σE_k and of θ − σE_k
    pub plus: Vec<f64>,
    pub minus: Vec<f64>,
    /// (u_k⁺ − u_k⁻)/2
    pub coeffs: Vec<f64>,
    /// |Δθ| and |θ| before the step (Frobenius over every selected weight)
    pub step: f64,
    pub norm: f64,
    pub evals: usize,
    /// seconds in the evaluations, and in the whole generation
    pub eval_secs: f64,
    pub secs: f64,
}

impl GenReport {
    fn all(&self) -> impl Iterator<Item = f64> + '_ {
        self.plus.iter().chain(&self.minus).copied()
    }
    pub fn mean(&self) -> f64 {
        self.all().sum::<f64>() / self.evals.max(1) as f64
    }
    pub fn best(&self) -> f64 {
        self.all().fold(f64::NEG_INFINITY, f64::max)
    }
    pub fn worst(&self) -> f64 {
        self.all().fold(f64::INFINITY, f64::min)
    }
    pub fn evals_per_sec(&self) -> f64 {
        self.evals as f64 / self.eval_secs.max(1e-9)
    }
}

/// ES over the selected variables of a model. The variables hold the centre θ between evaluations.
pub struct Es {
    pub opts: EsOpts,
    pub names: Vec<String>,
    vars: Vec<Var>,
    dims: Vec<Vec<usize>>,
    /// the centre θ on the host, and the same bits on the variables' device (what `restore` sets)
    theta: Vec<Vec<f32>>,
    theta_dev: Vec<Tensor>,
    /// the next generation's index
    pub gen: u64,
}

fn host(t: &Tensor) -> Result<Vec<f32>, String> {
    t.flatten_all().and_then(|x| x.to_vec1::<f32>()).map_err(|e| e.to_string())
}

impl Es {
    /// Start from the variables' current values. They must be f32.
    pub fn new(sel: Vec<(String, Var)>, opts: EsOpts) -> Result<Es, String> {
        if sel.is_empty() {
            return Err("no weights selected".into());
        }
        if opts.pop == 0 || !(opts.sigma > 0.0 && opts.sigma.is_finite()) || !(opts.lr >= 0.0 && opts.lr.is_finite()) || opts.rank == 0 {
            return Err(format!("ES needs pop >= 1, sigma > 0, lr >= 0 and rank >= 1 (got {opts:?})"));
        }
        let (mut names, mut vars, mut dims, mut theta, mut theta_dev) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (name, var) in sel {
            if var.dtype() != DType::F32 {
                return Err(format!("{name} is {:?}; ES tunes f32 weights only", var.dtype()));
            }
            let t = var.as_tensor();
            let h = host(t)?;
            let td = Tensor::from_vec(h.clone(), t.dims(), t.device()).map_err(|e| e.to_string())?;
            names.push(name);
            dims.push(t.dims().to_vec());
            theta.push(h);
            theta_dev.push(td);
            vars.push(var);
        }
        Ok(Es { opts, names, vars, dims, theta, theta_dev, gen: 0 })
    }

    /// How many weights are tuned.
    pub fn count(&self) -> usize {
        self.theta.iter().map(Vec::len).sum()
    }

    /// The selected tensors' shapes, in `names` order.
    pub fn dims(&self) -> &[Vec<usize>] {
        &self.dims
    }

    /// The centre θ (host copies, in `names` order).
    pub fn theta(&self) -> &[Vec<f32>] {
        &self.theta
    }

    /// The noise of tensor p for member k of generation g (regenerated from its seed every time).
    pub fn noise(&self, g: u64, k: usize, p: usize) -> Noise {
        Noise::new(stream_seed(member_seed(self.opts.seed, g, k), p), &self.dims[p], self.opts.rank)
    }

    /// Put the centre θ back into the variables, bit for bit.
    pub fn restore(&self) -> Result<(), String> {
        for (v, t) in self.vars.iter().zip(&self.theta_dev) {
            v.set(t).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Set the variables to θ + s·E (one dense E per selected tensor).
    fn set_shifted(&self, e: &[Vec<f64>], s: f64) -> Result<(), String> {
        for (p, v) in self.vars.iter().enumerate() {
            let w: Vec<f32> = self.theta[p].iter().zip(&e[p]).map(|(&t, &x)| (t as f64 + s * x) as f32).collect();
            let t = Tensor::from_vec(w, self.dims[p].as_slice(), v.device()).map_err(|e| e.to_string())?;
            v.set(&t).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// The fitness of θ + s·E, with θ put back afterwards whatever the fitness returned.
    pub fn eval_shifted<F>(&self, model: &Model, e: &[Vec<f64>], s: f64, f: &mut F) -> Result<f64, String>
    where
        F: FnMut(&Model) -> Result<f64, String>,
    {
        let set = self.set_shifted(e, s);
        let r = set.and_then(|_| f(model));
        self.restore()?;
        r
    }

    /// The fitness of member k's pair (θ + σE_k, θ − σE_k) in generation g.
    pub fn eval_pair<F>(&self, model: &Model, g: u64, k: usize, f: &mut F) -> Result<(f64, f64), String>
    where
        F: FnMut(&Model) -> Result<f64, String>,
    {
        let e: Vec<Vec<f64>> = (0..self.vars.len()).map(|p| self.noise(g, k, p).dense()).collect();
        let plus = self.eval_shifted(model, &e, self.opts.sigma, f)?;
        let minus = self.eval_shifted(model, &e, -self.opts.sigma, f)?;
        Ok((plus, minus))
    }

    /// Σ_k c_k E_k for every selected tensor, the noise regenerated from its seeds (low rank: one product of the
    /// stacked, scaled factors).
    pub fn direction(&self, g: u64, coeffs: &[f64]) -> Vec<Vec<f64>> {
        (0..self.vars.len()).map(|p| self.direction_one(g, coeffs, p)).collect()
    }

    fn direction_one(&self, g: u64, coeffs: &[f64], p: usize) -> Vec<f64> {
        let len = self.theta[p].len();
        let live: Vec<(usize, f64)> = coeffs.iter().copied().enumerate().filter(|&(_, c)| c != 0.0).collect();
        if live.is_empty() {
            return vec![0.0; len];
        }
        if self.dims[p].len() == 2 {
            let (m, n, r) = (self.dims[p][0], self.dims[p][1], self.opts.rank);
            let q = live.len() * r;
            // [c_1·A_1 | … ] [m, q] and [B_1 | … ] [n, q]
            let (mut ac, mut bc) = (vec![0f64; m * q], vec![0f64; n * q]);
            for (slot, &(k, c)) in live.iter().enumerate() {
                let Noise::LowRank { a, b, .. } = self.noise(g, k, p) else { unreachable!("2-D tensors get low-rank noise") };
                for i in 0..m {
                    for l in 0..r {
                        ac[i * q + slot * r + l] = c * a[i * r + l];
                    }
                }
                for j in 0..n {
                    for l in 0..r {
                        bc[j * q + slot * r + l] = b[j * r + l];
                    }
                }
            }
            let s = 1.0 / (r as f64).sqrt();
            let mut out = vec![0f64; m * n];
            for i in 0..m {
                let ai = &ac[i * q..(i + 1) * q];
                for j in 0..n {
                    out[i * n + j] = ai.iter().zip(&bc[j * q..(j + 1) * q]).map(|(x, y)| x * y).sum::<f64>() * s;
                }
            }
            out
        } else {
            let mut out = vec![0f64; len];
            for &(k, c) in &live {
                for (o, e) in out.iter_mut().zip(self.noise(g, k, p).dense()) {
                    *o += c * e;
                }
            }
            out
        }
    }

    /// θ ← θ + scale · dir, into the variables too. Returns (|Δθ|, |θ| before).
    fn apply(&mut self, dir: &[Vec<f64>], scale: f64) -> Result<(f64, f64), String> {
        let norm = self.theta.iter().flatten().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        if scale == 0.0 || dir.iter().flatten().all(|&x| x == 0.0) {
            return Ok((0.0, norm));
        }
        let mut step = 0f64;
        for p in 0..self.vars.len() {
            for (t, &d) in self.theta[p].iter_mut().zip(&dir[p]) {
                let new = (*t as f64 + scale * d) as f32;
                step += (new as f64 - *t as f64).powi(2);
                *t = new;
            }
            let v = &self.vars[p];
            self.theta_dev[p] = Tensor::from_vec(self.theta[p].clone(), self.dims[p].as_slice(), v.device()).map_err(|e| e.to_string())?;
        }
        self.restore()?;
        Ok((step.sqrt(), norm))
    }

    /// One generation: evaluate the pop antithetic pairs, shape by centred ranks, step θ.
    pub fn generation<F>(&mut self, model: &Model, f: &mut F) -> Result<GenReport, String>
    where
        F: FnMut(&Model) -> Result<f64, String>,
    {
        let t0 = Instant::now();
        let g = self.gen;
        let (mut plus, mut minus) = (Vec::with_capacity(self.opts.pop), Vec::with_capacity(self.opts.pop));
        for k in 0..self.opts.pop {
            let (a, b) = self.eval_pair(model, g, k, f)?;
            plus.push(a);
            minus.push(b);
        }
        let eval_secs = t0.elapsed().as_secs_f64();
        let coeffs = pair_coeffs(&plus, &minus);
        let dir = self.direction(g, &coeffs);
        let scale = self.opts.lr / (self.opts.pop as f64 * self.opts.sigma);
        let (step, norm) = self.apply(&dir, scale)?;
        self.gen += 1;
        Ok(GenReport { gen: g, plus, minus, coeffs, step, norm, evals: 2 * self.opts.pop, eval_secs, secs: t0.elapsed().as_secs_f64() })
    }
}
