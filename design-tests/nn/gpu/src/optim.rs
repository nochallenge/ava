// TESTING / DESIGN ONLY — ava_gpu::optim: AdamW with per-parameter weight decay, global-norm
// gradient clipping and the warmup + cosine LR schedule, all matching ava_grad::optim.
//
// AdamW step t (1-based): m = b1·m + (1-b1)·g; v = b2·v + (1-b2)·g^2;
// w = w·(1 - lr·wd·[decay]) - (lr / (1 - b1^t)) · m / (sqrt(v) / sqrt(1 - b2^t) + eps).
// Clipping: if |g| > clip, every gradient is scaled by clip / (|g| + 1e-6) before the step.
// v42 (H7, enable_critical_periods): an LR per block (lr · multiplier); off, every step runs as before.

use ava_parts::plasticity::CriticalPeriods;
use candle_core::backprop::GradStore;
use candle_core::{DType, Result, Tensor, Var};
use std::collections::HashMap;

pub struct Param {
    pub name: String,
    pub var: Var,
    pub decay: bool,
}

pub struct AdamW {
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub t: u64,
    params: Vec<Param>,
    // Vars so each step writes into the same buffers (Var::set copies in place). Fresh moment tensors every
    // step would live across the next step's transient allocations and pin the CUDA pool's chunks: the pool's
    // reserved memory grew ~17 MB/step until the card thrashed (v33 Ava-M run, step ~650).
    m: Vec<Var>,
    v: Vec<Var>,
    /// v42 (A1, --opt muon): Muon's settings and which parameters it steps (None: AdamW for all, as before)
    muon: Option<MuonCfg>,
    muon_on: Vec<bool>,
    /// v42 (H7, --critical-periods): the LR multiplier schedule and each parameter's block (None: off, as before)
    crit: Option<(CriticalPeriods, Vec<Option<usize>>)>,
}

impl AdamW {
    /// b1 = 0.9, b2 = 0.95, eps = 1e-8.
    pub fn new(params: Vec<Param>, weight_decay: f64) -> Result<AdamW> {
        let mut m = Vec::with_capacity(params.len());
        let mut v = Vec::with_capacity(params.len());
        for p in &params {
            m.push(Var::zeros(p.var.shape(), p.var.dtype(), p.var.device())?);
            v.push(Var::zeros(p.var.shape(), p.var.dtype(), p.var.device())?);
        }
        let muon_on = vec![false; params.len()];
        Ok(AdamW { beta1: 0.9, beta2: 0.95, eps: 1e-8, weight_decay, t: 0, params, m, v, muon: None, muon_on, crit: None })
    }

    /// v42 (H7, --critical-periods): from now on each parameter of block l (named b{l}.…) steps at
    /// lr · cp.multiplier(l, s), s the 0-based index of the update (t − 1 once the count has gone up: the loop's step,
    /// restored with t on a resume). Its decay shrink 1 − lr·wd uses the same LR, and so does Muon. Every other
    /// parameter keeps lr. Returns how many parameters are in blocks.
    pub fn enable_critical_periods(&mut self, cp: CriticalPeriods) -> usize {
        let layers: Vec<Option<usize>> = self.params.iter().map(|p| crate::plastic::block_of(&p.name)).collect();
        let n = layers.iter().filter(|l| l.is_some()).count();
        self.crit = Some((cp, layers));
        n
    }

    /// v42 (H7): each parameter's LR multiplier for the update with 0-based index `s` (None when off; 1 outside blocks).
    pub fn lr_mults(&self, s: u64) -> Option<Vec<f64>> {
        let (cp, layers) = self.crit.as_ref()?;
        Some(layers.iter().map(|l| l.map_or(1.0, |l| cp.multiplier(l, s as f64))).collect())
    }

    /// v42 (H6): the position of the parameter called `name`.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p.name == name)
    }

    /// v42 (H6): parameter i's moment buffers (m, v); on a Muon parameter m holds the momentum.
    pub fn moments(&self, i: usize) -> (&Tensor, &Tensor) {
        (self.m[i].as_tensor(), self.v[i].as_tensor())
    }

    /// v42 (H6): set both of parameter i's moments to 0 where `at` (u8, its shape) is 1; elsewhere they are untouched.
    pub fn zero_moments_at(&self, i: usize, at: &Tensor) -> Result<()> {
        for var in [&self.m[i], &self.v[i]] {
            let z = var.as_tensor().zeros_like()?;
            var.set(&at.where_cond(&z, var.as_tensor())?)?;
        }
        Ok(())
    }

    /// v42 (A1, --opt muon): from now on the hidden weight matrices (the 2-D parameters that get weight decay: not the
    /// embeddings, the unembedding (tied to them), the gains, biases or the small gate projections) step by Muon;
    /// the rest by AdamW as before. Muon's momentum lives in the m buffer (saved and restored as before). Returns how
    /// many parameters Muon steps.
    pub fn enable_muon(&mut self, cfg: MuonCfg) -> usize {
        self.muon_on = self.params.iter().map(|p| p.decay && p.var.as_tensor().rank() == 2).collect();
        self.muon = Some(cfg);
        self.muon_on.iter().filter(|&&x| x).count()
    }

    fn is_muon(&self, i: usize) -> bool {
        self.muon.is_some() && self.muon_on[i]
    }

    /// One Muon step of parameter `i` with its (clipped) gradient `g`, at Adam's scheduled LR `lr`:
    ///     buf = mu·buf + g;   u = g + mu·buf (Nesterov) or buf;   o = newton_schulz5(u)
    ///     w = w·(1 - lr·wd) - lr·ratio·sqrt(max(1, out/in))·o
    /// Weights are [in, out]; the decay shrink is AdamW's, so --wd means the same with either optimizer.
    fn muon_step(&self, i: usize, g: &Tensor, lr: f64) -> Result<()> {
        let mc = self.muon.expect("muon is on");
        let p = &self.params[i];
        let buf = self.m[i].as_tensor().affine(mc.momentum, 0.0)?.add(g)?;
        let u = if mc.nesterov { g.add(&buf.affine(mc.momentum, 0.0)?)? } else { buf.clone() };
        let o = newton_schulz5(&u, mc.ns_steps)?;
        let (r, c) = p.var.as_tensor().dims2()?;
        let scale = (c as f64 / r as f64).max(1.0).sqrt();
        let shrink = if p.decay { 1.0 - lr * self.weight_decay } else { 1.0 };
        let w = p.var.as_tensor().affine(shrink, 0.0)?.sub(&o.affine(lr * mc.lr_ratio * scale, 0.0)?)?;
        p.var.set(&w)?;
        self.m[i].set(&buf)?;
        Ok(())
    }

    pub fn params(&self) -> &[Param] {
        &self.params
    }

    /// Global L2 norm of the gradients of all parameters (one device-to-host copy).
    pub fn grad_norm(&self, grads: &GradStore) -> Result<f64> {
        let mut acc: Option<Tensor> = None;
        for p in &self.params {
            if let Some(g) = grads.get(p.var.as_tensor()) {
                // fast path: rows first (candle's sum_all of a big tensor runs on ONE GPU block)
                let s = if crate::fast_path() && g.rank() >= 2 { g.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sum_all()? } else { g.sqr()?.sum_all()? };
                acc = Some(match acc {
                    None => s,
                    Some(a) => a.add(&s)?,
                });
            }
        }
        match acc {
            None => Ok(0.0),
            Some(a) => Ok(a.to_dtype(DType::F64)?.to_scalar::<f64>()?.sqrt()),
        }
    }

    /// Speed round 13b: grad_norm without the copy to the host: the same sums, then the same f64 sqrt, as an f64
    /// scalar on the device. None when there are no gradients.
    pub fn grad_norm_t(&self, grads: &GradStore) -> Result<Option<Tensor>> {
        if crate::fast_path() && multi_opt_on() {
            // v40: the same sums in three launches (fused::grad_norm_multi, bit for bit the ops below)
            let gs: Vec<Tensor> = self.params.iter().filter_map(|p| grads.get(p.var.as_tensor()).cloned()).collect();
            if !gs.is_empty() && gs.iter().all(|g| g.device().is_cuda() && g.dtype() == DType::F32) {
                return Ok(Some(crate::fused::grad_norm_multi(&gs)?));
            }
        }
        let mut acc: Option<Tensor> = None;
        for p in &self.params {
            if let Some(g) = grads.get(p.var.as_tensor()) {
                let s = if crate::fast_path() && g.rank() >= 2 { g.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sum_all()? } else { g.sqr()?.sum_all()? };
                acc = Some(match acc {
                    None => s,
                    Some(a) => a.add(&s)?,
                });
            }
        }
        acc.map(|a| a.to_dtype(DType::F64)?.sqrt()).transpose()
    }

    /// One AdamW step with the clipping factor on the device (`scale`, from clip_scale_t): the same update as `step`
    /// on the CUDA fast path, without waiting for the host. Falls back to `step` (one copy to the host) elsewhere.
    pub fn step_t(&mut self, grads: &GradStore, lr: f64, scale: &Tensor) -> Result<()> {
        if !(crate::fast_path() && scale.device().is_cuda()) {
            let s = scale.to_scalar::<f64>()?;
            return self.step(grads, lr, s);
        }
        self.t += 1;
        let (b1, b2) = (self.beta1, self.beta2);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        let mults = self.lr_mults(self.t - 1); // v42 (H7); None when off
        if multi_opt_on() {
            if let Some(mu) = mults.as_deref() {
                return self.step_t_by_lr(grads, lr, scale, mu, bc1, bc2);
            }
            // v40: every parameter in one launch (fused::adamw_multi: the same arithmetic per element)
            let (mut ws, mut gs, mut ms, mut vs, mut sh) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (i, p) in self.params.iter().enumerate() {
                let Some(g) = grads.get(p.var.as_tensor()) else { continue };
                if self.is_muon(i) {
                    continue; // v42: stepped by Muon below
                }
                ws.push(p.var.as_tensor().clone());
                gs.push(g.detach());
                ms.push(self.m[i].as_tensor().clone());
                vs.push(self.v[i].as_tensor().clone());
                sh.push(if p.decay { 1.0 - lr * self.weight_decay } else { 1.0 });
            }
            let sc = crate::fused::AdamWScalars {
                grad_scale: f64::NAN,
                b1,
                b2,
                eps: self.eps,
                step_size: lr / bc1,
                inv_sqrt_bc2: 1.0 / bc2.sqrt(),
                shrink: f64::NAN, // per parameter, from `sh`
            };
            if self.muon.is_none() {
                return crate::fused::adamw_multi(&ws, &gs, &ms, &vs, &sh, scale, &sc);
            }
            if !ws.is_empty() {
                crate::fused::adamw_multi(&ws, &gs, &ms, &vs, &sh, scale, &sc)?;
            }
            return self.muon_steps_t(grads, lr, scale, None);
        }
        for (i, p) in self.params.iter().enumerate() {
            let Some(g) = grads.get(p.var.as_tensor()) else { continue };
            if self.is_muon(i) {
                continue; // v42: stepped by Muon below
            }
            // v42 (H7): this parameter's LR (lr itself when off)
            let lr = match &mults {
                None => lr,
                Some(mu) => lr * mu[i],
            };
            let sc = crate::fused::AdamWScalars {
                grad_scale: f64::NAN, // read from `scale` on the device
                b1,
                b2,
                eps: self.eps,
                step_size: lr / bc1,
                inv_sqrt_bc2: 1.0 / bc2.sqrt(),
                shrink: if p.decay { 1.0 - lr * self.weight_decay } else { 1.0 },
            };
            crate::fused::adamw_in_place_s(p.var.as_tensor(), &g.detach(), self.m[i].as_tensor(), self.v[i].as_tensor(), scale, &sc)?;
        }
        if self.muon.is_some() {
            return self.muon_steps_t(grads, lr, scale, mults.as_deref());
        }
        Ok(())
    }

    /// v42 (H7): `step_t`'s multi-tensor path with an LR per parameter (lr · mults[i]): one adamw_multi launch per
    /// distinct LR (at most one per layer plus one; a single one before any period closes), the same arithmetic per
    /// element as one launch with that LR.
    fn step_t_by_lr(&self, grads: &GradStore, lr: f64, scale: &Tensor, mults: &[f64], bc1: f64, bc2: f64) -> Result<()> {
        let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
        for (i, p) in self.params.iter().enumerate() {
            if grads.get(p.var.as_tensor()).is_none() || self.is_muon(i) {
                continue;
            }
            let lr_i = lr * mults[i];
            match groups.iter_mut().find(|g| g.0.to_bits() == lr_i.to_bits()) {
                Some(g) => g.1.push(i),
                None => groups.push((lr_i, vec![i])),
            }
        }
        for (lr_g, idx) in &groups {
            let (mut ws, mut gs, mut ms, mut vs, mut sh) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for &i in idx {
                let p = &self.params[i];
                let g = grads.get(p.var.as_tensor()).expect("grouped above");
                ws.push(p.var.as_tensor().clone());
                gs.push(g.detach());
                ms.push(self.m[i].as_tensor().clone());
                vs.push(self.v[i].as_tensor().clone());
                sh.push(if p.decay { 1.0 - lr_g * self.weight_decay } else { 1.0 });
            }
            let sc = crate::fused::AdamWScalars {
                grad_scale: f64::NAN,
                b1: self.beta1,
                b2: self.beta2,
                eps: self.eps,
                step_size: lr_g / bc1,
                inv_sqrt_bc2: 1.0 / bc2.sqrt(),
                shrink: f64::NAN, // per parameter, from `sh`
            };
            crate::fused::adamw_multi(&ws, &gs, &ms, &vs, &sh, scale, &sc)?;
        }
        if self.muon.is_some() {
            return self.muon_steps_t(grads, lr, scale, Some(mults));
        }
        Ok(())
    }

    /// v42: the Muon steps of `step_t`, the gradients times the clipping factor `scale` (an f64 scalar on the device).
    /// `mults`: v42 (H7) LR multipliers per parameter (None: lr for all).
    fn muon_steps_t(&self, grads: &GradStore, lr: f64, scale: &Tensor, mults: Option<&[f64]>) -> Result<()> {
        let s32 = scale.to_dtype(DType::F32)?;
        for (i, p) in self.params.iter().enumerate() {
            if !self.is_muon(i) {
                continue;
            }
            let Some(g) = grads.get(p.var.as_tensor()) else { continue };
            let lr = match mults {
                None => lr,
                Some(mu) => lr * mu[i],
            };
            self.muon_step(i, &g.detach().broadcast_mul(&s32)?, lr)?;
        }
        Ok(())
    }

    /// One AdamW step; `grad_scale` multiplies every gradient first (the clipping factor). v52 fix_S: a factor that is
    /// not finite (clip_scale of a non-finite norm) skips the step: no weight, moment or step count changes.
    pub fn step(&mut self, grads: &GradStore, lr: f64, grad_scale: f64) -> Result<()> {
        if !grad_scale.is_finite() {
            return Ok(());
        }
        self.t += 1;
        let (b1, b2) = (self.beta1, self.beta2);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        let step_size = lr / bc1;
        let inv_sqrt_bc2 = 1.0 / bc2.sqrt();
        let mults = self.lr_mults(self.t - 1); // v42 (H7); None when off
        for (i, p) in self.params.iter().enumerate() {
            let Some(g) = grads.get(p.var.as_tensor()) else { continue };
            // v42 (H7): this parameter's LR and step size (lr and step_size themselves when off)
            let (lr, step_size) = match &mults {
                None => (lr, step_size),
                Some(mu) => (lr * mu[i], lr * mu[i] / bc1),
            };
            if self.is_muon(i) {
                // v42 (A1): Muon for this matrix
                let g = if grad_scale != 1.0 { g.affine(grad_scale, 0.0)? } else { g.clone() };
                self.muon_step(i, &g.detach(), lr)?;
                continue;
            }
            if crate::fast_path() && g.device().is_cuda() {
                // speed round 2: one fused kernel for the whole update, then three copies into the buffers
                let sc = crate::fused::AdamWScalars {
                    grad_scale,
                    b1,
                    b2,
                    eps: self.eps,
                    step_size,
                    inv_sqrt_bc2,
                    shrink: if p.decay { 1.0 - lr * self.weight_decay } else { 1.0 },
                };
                // speed round 13: in place (no [3, n] result, no three copies back); same arithmetic
                crate::fused::adamw_in_place(p.var.as_tensor(), &g.detach(), self.m[i].as_tensor(), self.v[i].as_tensor(), &sc)?;
                continue;
            }
            let g = if grad_scale != 1.0 { g.affine(grad_scale, 0.0)? } else { g.clone() };
            // detach g: the moments must not keep an op chain into the step's graph
            let g = g.detach();
            let m = (self.m[i].affine(b1, 0.0)? + g.affine(1.0 - b1, 0.0)?)?;
            let v = (self.v[i].affine(b2, 0.0)? + g.sqr()?.affine(1.0 - b2, 0.0)?)?;
            let denom = (v.sqrt()?.affine(inv_sqrt_bc2, 0.0)? + self.eps)?;
            let shrink = if p.decay { 1.0 - lr * self.weight_decay } else { 1.0 };
            let w = (p.var.as_tensor().affine(shrink, 0.0)? - m.div(&denom)?.affine(step_size, 0.0)?)?;
            p.var.set(&w)?;
            self.m[i].set(&m)?;
            self.v[i].set(&v)?;
        }
        Ok(())
    }

    /// The moments as named tensors (m.<param>, v.<param>), for a checkpoint.
    pub fn state(&self) -> HashMap<String, Tensor> {
        let mut out = HashMap::new();
        for (i, p) in self.params.iter().enumerate() {
            out.insert(format!("m.{}", p.name), self.m[i].as_tensor().clone());
            out.insert(format!("v.{}", p.name), self.v[i].as_tensor().clone());
        }
        out
    }

    /// Restore the moments saved by `state` and the step count `t`, in place.
    pub fn load_state(&mut self, map: &HashMap<String, Tensor>, t: u64) -> Result<()> {
        for (i, p) in self.params.iter().enumerate() {
            for (pre, var) in [("m", &self.m[i]), ("v", &self.v[i])] {
                let k = format!("{pre}.{}", p.name);
                let x = map.get(&k).ok_or_else(|| candle_core::Error::Msg(format!("checkpoint lacks {k}")))?;
                var.set(&x.to_device(var.device())?)?;
            }
        }
        self.t = t;
        Ok(())
    }

    /// v42 (U1, --load-opt): restore the moments `map` holds for these parameters, by name, and the step count `t`.
    /// A parameter the file lacks keeps zero moments (a part added since the file was written). Returns (restored,
    /// missing) parameter counts; a moment whose shape differs is an error.
    pub fn load_state_partial(&mut self, map: &HashMap<String, Tensor>, t: u64) -> Result<(usize, usize)> {
        let (mut got, mut miss) = (0usize, 0usize);
        for (i, p) in self.params.iter().enumerate() {
            let (km, kv) = (format!("m.{}", p.name), format!("v.{}", p.name));
            match (map.get(&km), map.get(&kv)) {
                (Some(m), Some(v)) => {
                    for (x, var, k) in [(m, &self.m[i], &km), (v, &self.v[i], &kv)] {
                        if x.dims() != var.dims() {
                            candle_core::bail!("{k} has shape {:?}, the parameter {:?}", x.dims(), var.dims());
                        }
                        var.set(&x.to_device(var.device())?.to_dtype(var.dtype())?)?;
                    }
                    got += 1;
                }
                _ => miss += 1,
            }
        }
        self.t = t;
        Ok((got, miss))
    }
}

/// v42 (U1, --wd-target w0): decoupled weight decay toward w0 instead of toward 0 (L2-Init). AdamW has already
/// shrunk every decayed weight by (1 − lr·wd); adding lr·wd·w0 after the step makes the whole decay
///     w ← w − lr·wd·(w − w0),
/// so a model learning after release is held near the weights she started from rather than shrunk toward 0.
pub struct DecayToward {
    vars: Vec<Var>,
    w0: Vec<Tensor>,
}

impl DecayToward {
    /// One (parameter, its w0) pair per decayed parameter.
    pub fn new(pairs: Vec<(Var, Tensor)>) -> DecayToward {
        let (vars, w0) = pairs.into_iter().unzip();
        DecayToward { vars, w0 }
    }

    pub fn len(&self) -> usize {
        self.vars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    /// After an AdamW step at `lr` with weight decay `wd`.
    pub fn apply(&self, lr: f64, wd: f64) -> Result<()> {
        let c = lr * wd;
        if c == 0.0 {
            return Ok(());
        }
        for (v, w0) in self.vars.iter().zip(&self.w0) {
            v.set(&v.as_tensor().add(&w0.affine(c, 0.0)?)?)?;
        }
        Ok(())
    }
}

/// v40: grad_norm_t and step_t (CUDA fast path) in a few launches for all parameters (fused::grad_norm_multi,
/// fused::adamw_multi), the same bits as one launch chain per parameter. AVA_MULTI_OPT=0 keeps the per-parameter ops.
pub fn multi_opt_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("AVA_MULTI_OPT").is_ok_and(|v| v == "0"))
}

/// Clipping factor for a gradient norm: clip / (norm + 1e-6) when norm > clip, else 1. v52 fix_S (a defense, not the
/// fix): a norm that is not finite gives NaN, which AdamW::step (and step_t's kernels) take as "skip this update".
/// Before, NaN > clip was false, so the factor was 1 and the NaN gradients went into the weights and Adam's moments
/// (runs/v52/fix_S.md: b2.hebb.{ba, bb, vres} in B-s52).
pub fn clip_scale(norm: f64, clip: f64) -> f64 {
    if !norm.is_finite() {
        return f64::NAN;
    }
    if clip > 0.0 && norm > clip && norm > 0.0 {
        clip / (norm + 1e-6)
    } else {
        1.0
    }
}

/// clip_scale on the device: `norm` an f64 scalar (grad_norm_t), result an f64 scalar. The same IEEE operations
/// (compare, add 1e-6, divide), so the same value as clip_scale on the host; NaN when the norm is not finite (v52).
pub fn clip_scale_t(norm: &Tensor, clip: f64) -> Result<Tensor> {
    let one = Tensor::ones((), DType::F64, norm.device())?;
    let s = if clip <= 0.0 {
        one
    } else {
        let s = Tensor::full(clip, (), norm.device())?.div(&(norm + 1e-6)?)?;
        norm.gt(clip)?.where_cond(&s, &one)?
    };
    // norm - norm is 0 exactly when the norm is finite (inf - inf and NaN - NaN are NaN)
    let nan = Tensor::full(f64::NAN, (), norm.device())?;
    norm.sub(norm)?.eq(0.0)?.where_cond(&s, &nan)
}

/// Linear warmup over `warmup` steps (step 0 gets max_lr/warmup, step warmup-1 gets max_lr), then cosine
/// decay from max_lr at step `warmup` to min_lr at step `total`; min_lr afterwards. Same arithmetic as
/// ava_grad::optim::cosine_lr.
pub fn cosine_lr(step: usize, warmup: usize, total: usize, max_lr: f32, min_lr: f32) -> f32 {
    if step < warmup {
        return max_lr * (step + 1) as f32 / warmup as f32;
    }
    if step >= total || total <= warmup {
        return min_lr;
    }
    let progress = (step - warmup) as f64 / (total - warmup) as f64;
    let cos = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    (min_lr as f64 + (max_lr - min_lr) as f64 * cos) as f32
}

// ------------------------------------------------------------------------------------------- v42 (A1): WSD and Muon

/// v42 (A1, --schedule): the LR schedule. `Cosine` is `cosine_lr` exactly (the default). `Wsd` (warmup, stable,
/// decay): the same warmup, then the peak held, then over the last `decay_frac` of the steps an anneal down to the
/// floor, linear or 1 - sqrt (Hägele et al. 2024 found 1 - sqrt the better of the two). A WSD run can stop at the end
/// of its stable part and anneal from there, so one run serves several lengths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Schedule {
    Cosine,
    Wsd { decay_frac: f64, sqrt: bool },
}

impl Schedule {
    /// From the flags: --schedule cosine|wsd, --decay-frac F, --decay-shape linear|1-sqrt.
    pub fn parse(kind: &str, decay_frac: f64, shape: &str) -> std::result::Result<Schedule, String> {
        let sqrt = match shape {
            "linear" => false,
            "1-sqrt" | "sqrt" => true,
            o => return Err(format!("--decay-shape {o:?}: linear or 1-sqrt")),
        };
        match kind {
            "cosine" => Ok(Schedule::Cosine),
            "wsd" => {
                if !(decay_frac > 0.0 && decay_frac <= 1.0) {
                    return Err(format!("--decay-frac {decay_frac}: must be in (0, 1]"));
                }
                Ok(Schedule::Wsd { decay_frac, sqrt })
            }
            o => Err(format!("--schedule {o:?}: cosine or wsd")),
        }
    }

    /// The LR at `step` of `total` (0-based).
    pub fn lr(&self, step: usize, warmup: usize, total: usize, max_lr: f32, min_lr: f32) -> f32 {
        match *self {
            Schedule::Cosine => cosine_lr(step, warmup, total, max_lr, min_lr),
            Schedule::Wsd { decay_frac, sqrt } => wsd_lr(step, warmup, total, max_lr, min_lr, decay_frac, sqrt),
        }
    }

    /// " schedule=wsd decay_frac=F decay_shape=S" (empty for cosine): for resume signatures and logs.
    pub fn signature(&self) -> String {
        match *self {
            Schedule::Cosine => String::new(),
            Schedule::Wsd { decay_frac, sqrt } => format!(" schedule=wsd decay_frac={decay_frac} decay_shape={}", if sqrt { "1-sqrt" } else { "linear" }),
        }
    }
}

/// WSD: warmup as `cosine_lr`; then max_lr; over the last round(total · decay_frac) steps (never inside the warmup)
/// the LR falls to min_lr, reaching it at the last step: lr = min + (max - min) · f(p), p = (step - start + 1) / span,
/// f = 1 - p (linear) or 1 - sqrt(p). min_lr from `total` on.
pub fn wsd_lr(step: usize, warmup: usize, total: usize, max_lr: f32, min_lr: f32, decay_frac: f64, sqrt: bool) -> f32 {
    if step < warmup {
        return max_lr * (step + 1) as f32 / warmup as f32;
    }
    if step >= total {
        return min_lr;
    }
    let n_decay = ((total as f64 * decay_frac).round() as usize).clamp(1, total);
    let start = (total - n_decay).max(warmup);
    if step < start {
        return max_lr;
    }
    let span = (total - start).max(1);
    let p = (step - start + 1) as f64 / span as f64;
    let f = if sqrt { 1.0 - p.sqrt() } else { 1.0 - p };
    (min_lr as f64 + (max_lr - min_lr) as f64 * f) as f32
}

/// v42 (A1, --opt muon): Muon's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MuonCfg {
    /// Muon's LR over Adam's (--muon-lr / --lr); both follow the same schedule shape
    pub lr_ratio: f64,
    pub momentum: f64,
    pub nesterov: bool,
    pub ns_steps: usize,
}

impl MuonCfg {
    /// The usual Muon: momentum 0.95, Nesterov, 5 Newton–Schulz steps; its own peak LR `muon_lr` beside Adam's `lr`.
    pub fn standard(muon_lr: f64, lr: f64) -> MuonCfg {
        MuonCfg { lr_ratio: muon_lr / lr, momentum: 0.95, nesterov: true, ns_steps: 5 }
    }
}

/// The quintic Newton–Schulz coefficients of Muon (Keller Jordan, 2024): fast growth of small singular values.
pub const NS_COEFFS: (f64, f64, f64) = (3.4445, -4.7750, 2.0315);

/// An almost orthogonal matrix in the direction of `g` [r, c] (its polar factor U V^T, with every singular value pushed
/// near 1): X = g / |g|_F, then `steps` times X <- aX + (bA + cA²)X with A = X X^T, on the wide side (r <= c, so A is
/// the small Gram matrix). The quintic's coefficients put the singular values in about [0.7, 1.2], not exactly 1.
pub fn newton_schulz5(g: &Tensor, steps: usize) -> Result<Tensor> {
    let (r, c) = g.dims2()?;
    let tall = r > c;
    let x0 = if tall { g.t()?.contiguous()? } else { g.contiguous()? };
    let nrm = x0.sqr()?.sum_all()?.sqrt()?.affine(1.0, 1e-7)?;
    let mut x = x0.broadcast_div(&nrm)?;
    let (a, b, cc) = NS_COEFFS;
    for _ in 0..steps {
        let am = x.matmul(&x.t()?)?;
        let bm = am.affine(b, 0.0)?.add(&am.matmul(&am)?.affine(cc, 0.0)?)?;
        x = x.affine(a, 0.0)?.add(&bm.matmul(&x)?)?;
    }
    if tall {
        x.t()?.contiguous()
    } else {
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn param(name: &str, vals: Vec<f32>, decay: bool) -> Param {
        let n = vals.len();
        Param { name: name.into(), var: Var::from_vec(vals, n, &Device::Cpu).unwrap(), decay }
    }

    /// Zero gradients for every parameter (loss = Σ 0·x).
    fn zero_grads(ps: &[Param]) -> GradStore {
        let mut loss = Tensor::zeros((), DType::F32, &Device::Cpu).unwrap();
        for p in ps {
            // times a zero tensor (candle drops affine(0) from the graph, which would leave no gradient at all)
            let x = p.var.as_tensor();
            loss = loss.add(&x.mul(&x.zeros_like().unwrap()).unwrap().sum_all().unwrap()).unwrap();
        }
        loss.backward().unwrap()
    }

    /// U1: AdamW's shrink plus DecayToward is w − lr·wd·(w − w0).
    #[test]
    fn decay_toward_w0_is_l2_init() {
        let p = param("x.w1", vec![2.0, -1.0, 0.5], true);
        let w0 = Tensor::from_vec(vec![1.0f32, 1.0, 0.5], 3, &Device::Cpu).unwrap();
        let var = p.var.clone();
        let mut opt = AdamW::new(vec![p], 0.1).unwrap();
        let dt = DecayToward::new(vec![(var.clone(), w0)]);
        let g = zero_grads(opt.params());
        let lr = 0.5;
        opt.step(&g, lr, 1.0).unwrap();
        dt.apply(lr, 0.1).unwrap();
        let got = var.as_tensor().to_vec1::<f32>().unwrap();
        for (x, (w, w0)) in got.iter().zip([(2.0f32, 1.0f32), (-1.0, 1.0), (0.5, 0.5)]) {
            let want = w - 0.05 * (w - w0);
            assert!((x - want).abs() < 1e-6, "{x} vs {want}");
        }
    }

    /// U1: --load-opt restores the moments a file holds by name and leaves the others at 0; Q1: beta2 is a setting.
    #[test]
    fn partial_moment_load_and_beta2() {
        let ps = vec![param("a", vec![1.0, 2.0], true), param("b", vec![3.0], false)];
        let mut opt = AdamW::new(ps, 0.0).unwrap();
        opt.beta2 = 0.99;
        let g = {
            let l = opt.params()[0].var.as_tensor().sum_all().unwrap().add(&opt.params()[1].var.as_tensor().affine(2.0, 0.0).unwrap().sum_all().unwrap()).unwrap();
            l.backward().unwrap()
        };
        opt.step(&g, 0.0, 1.0).unwrap();
        // v = (1 − b2)·g²: with beta2 0.99, 0.01 · 4 for b
        let st = opt.state();
        let vb = st["v.b"].to_vec1::<f32>().unwrap()[0];
        assert!((vb - 0.04).abs() < 1e-7, "{vb}");
        let mut part = HashMap::new();
        part.insert("m.a".to_string(), st["m.a"].clone());
        part.insert("v.a".to_string(), st["v.a"].clone());
        let ps2 = vec![param("a", vec![1.0, 2.0], true), param("b", vec![3.0], false)];
        let mut opt2 = AdamW::new(ps2, 0.0).unwrap();
        assert_eq!(opt2.load_state_partial(&part, 7).unwrap(), (1, 1));
        assert_eq!(opt2.t, 7);
        let s2 = opt2.state();
        assert_eq!(s2["m.a"].to_vec1::<f32>().unwrap(), st["m.a"].to_vec1::<f32>().unwrap());
        assert_eq!(s2["v.b"].to_vec1::<f32>().unwrap(), vec![0.0]);
        // a wrong shape is refused
        let mut bad = HashMap::new();
        bad.insert("m.b".to_string(), Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap());
        bad.insert("v.b".to_string(), Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap());
        assert!(opt2.load_state_partial(&bad, 1).is_err());
    }

    /// A1: the default schedule is cosine_lr exactly; WSD warms up the same, holds the peak, then anneals over the
    /// last 20% to the floor (reached at the last step), linear or 1 - sqrt; a warmup longer than the stable part wins.
    #[test]
    fn wsd_schedule_shape() {
        let cos = Schedule::parse("cosine", 0.2, "1-sqrt").unwrap();
        for s in 0..120 {
            assert_eq!(cos.lr(s, 10, 100, 3e-3, 3e-4).to_bits(), cosine_lr(s, 10, 100, 3e-3, 3e-4).to_bits());
        }
        assert_eq!(cos.signature(), "");
        let (max, min) = (1.0f32, 0.1f32);
        for sqrt in [false, true] {
            let w = Schedule::Wsd { decay_frac: 0.2, sqrt };
            for s in 0..10 {
                assert_eq!(w.lr(s, 10, 100, max, min), cosine_lr(s, 10, 100, max, min), "warmup {s}");
            }
            for s in 10..80 {
                assert_eq!(w.lr(s, 10, 100, max, min), max, "stable {s}");
            }
            let mut prev = max;
            for s in 80..100 {
                let x = w.lr(s, 10, 100, max, min);
                assert!(x < prev && x >= min, "{sqrt} {s}: {x}");
                prev = x;
            }
            assert!((w.lr(99, 10, 100, max, min) - min).abs() < 1e-6);
            assert_eq!(w.lr(150, 10, 100, max, min), min);
        }
        // the middle of the anneal: linear is half way, 1 - sqrt lower
        let lin = wsd_lr(89, 10, 100, max, min, 0.2, false);
        let sq = wsd_lr(89, 10, 100, max, min, 0.2, true);
        assert!((lin - (min + (max - min) * 0.5)).abs() < 1e-6, "{lin}");
        assert!((sq - (min + (max - min) * (1.0 - 0.5f32.sqrt()))).abs() < 1e-6, "{sq}");
        // decay_frac 1 with a warmup: the anneal starts when the warmup ends
        assert!((wsd_lr(10, 10, 100, max, min, 1.0, false) - (max * (1.0 - 1.0 / 90.0) + min / 90.0)).abs() < 1e-6);
        assert!(Schedule::parse("wsd", 0.0, "linear").is_err());
        assert!(Schedule::parse("wsd", 0.2, "cubic").is_err());
        assert!(Schedule::parse("step", 0.2, "linear").is_err());
        assert_eq!(Schedule::parse("wsd", 0.2, "1-sqrt").unwrap().signature(), " schedule=wsd decay_frac=0.2 decay_shape=1-sqrt");
    }

    /// The eigenvalues of a symmetric matrix (host Jacobi sweeps; small sizes only).
    fn sym_eigen(mut a: Vec<f64>, n: usize) -> Vec<f64> {
        for _ in 0..60 {
            let mut off = 0.0;
            for p in 0..n {
                for q in p + 1..n {
                    off += a[p * n + q] * a[p * n + q];
                }
            }
            if off < 1e-22 {
                break;
            }
            for p in 0..n {
                for q in p + 1..n {
                    let apq = a[p * n + q];
                    if apq.abs() < 1e-30 {
                        continue;
                    }
                    let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let t = if theta == 0.0 { 1.0 } else { t };
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..n {
                        let (akp, akq) = (a[k * n + p], a[k * n + q]);
                        a[k * n + p] = c * akp - s * akq;
                        a[k * n + q] = s * akp + c * akq;
                    }
                    for k in 0..n {
                        let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                        a[p * n + k] = c * apk - s * aqk;
                        a[q * n + k] = s * apk + c * aqk;
                    }
                }
            }
        }
        (0..n).map(|i| a[i * n + i]).collect()
    }

    fn randn(r: usize, c: usize, seed: u64) -> Tensor {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut u = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let v: Vec<f32> = (0..r * c).map(|_| ((-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos()) as f32).collect();
        Tensor::from_vec(v, (r, c), &Device::Cpu).unwrap()
    }

    /// A1: five quintic Newton–Schulz steps turn a random 64 x 128 matrix (singular values spread over 6x) into a nearly
    /// orthogonal one: every singular value in [0.6, 1.25] (the quintic's band), so X X^T is near I. A tall matrix
    /// gives the transpose of its wide twin's result; the result does not depend on the gradient's size.
    #[test]
    fn newton_schulz_is_near_orthogonal() {
        let g = randn(64, 128, 7);
        let x = newton_schulz5(&g, 5).unwrap();
        assert_eq!(x.dims(), &[64, 128]);
        let gram: Vec<f64> = x.matmul(&x.t().unwrap()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|&v| v as f64).collect();
        let ev = sym_eigen(gram, 64);
        let sv: Vec<f64> = ev.iter().map(|e| e.max(0.0).sqrt()).collect();
        let (lo, hi) = sv.iter().fold((f64::MAX, 0f64), |(a, b), &s| (a.min(s), b.max(s)));
        assert!(lo > 0.6 && hi < 1.25, "singular values in [{lo}, {hi}]");
        // the input was far from orthogonal
        let g0: Vec<f64> = g.matmul(&g.t().unwrap()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|&v| v as f64).collect();
        let e0 = sym_eigen(g0, 64);
        let (l0, h0) = e0.iter().fold((f64::MAX, 0f64), |(a, b), &s| (a.min(s), b.max(s)));
        assert!((h0 / l0).sqrt() > 3.0, "input spread {}", (h0 / l0).sqrt());
        let xt = newton_schulz5(&g.t().unwrap().contiguous().unwrap(), 5).unwrap();
        assert_eq!(xt.dims(), &[128, 64]);
        let d = xt.t().unwrap().sub(&x).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(d < 1e-5, "{d}");
        let x9 = newton_schulz5(&g.affine(9.0, 0.0).unwrap(), 5).unwrap();
        let d = x9.sub(&x).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(d < 1e-4, "{d}");
    }

    /// A1: with Muon on, only decayed matrices take Muon steps; every other parameter steps exactly as plain AdamW.
    /// The first Muon step is w·(1 - lr·wd) - lr·ratio·sqrt(max(1, out/in))·NS((1 + mu)·g), and the momentum is in m.
    /// `step_t` on the CPU gives the same as `step`.
    #[test]
    fn muon_steps_matrices_and_leaves_adam_alone() {
        let mk = || {
            vec![
                Param { name: "b0.mlp.w1".into(), var: Var::from_tensor(&randn(16, 32, 1)).unwrap(), decay: true },
                Param { name: "tok_emb".into(), var: Var::from_tensor(&randn(24, 16, 2)).unwrap(), decay: false },
                Param { name: "b0.norm".into(), var: Var::from_tensor(&randn(1, 16, 3).flatten_all().unwrap()).unwrap(), decay: false },
            ]
        };
        let grads_of = |ps: &[Param]| {
            let mut l = Tensor::zeros((), DType::F32, &Device::Cpu).unwrap();
            for (j, p) in ps.iter().enumerate() {
                let w = p.var.as_tensor();
                let c = randn(1, w.elem_count(), 10 + j as u64).reshape(w.shape()).unwrap();
                l = l.add(&w.mul(&c).unwrap().sum_all().unwrap()).unwrap();
            }
            l.backward().unwrap()
        };
        let (lr, wd) = (0.01, 0.1);
        let mut plain = AdamW::new(mk(), wd).unwrap();
        let mut mu = AdamW::new(mk(), wd).unwrap();
        assert_eq!(mu.enable_muon(MuonCfg::standard(0.02, lr)), 1);
        let mut mu_t = AdamW::new(mk(), wd).unwrap();
        mu_t.enable_muon(MuonCfg::standard(0.02, lr));
        let w0 = mu.params()[0].var.as_tensor().copy().unwrap();
        let g0 = grads_of(plain.params()).get(plain.params()[0].var.as_tensor()).unwrap().copy().unwrap();
        plain.step(&grads_of(plain.params()), lr, 0.5).unwrap();
        mu.step(&grads_of(mu.params()), lr, 0.5).unwrap();
        let half = Tensor::new(0.5f64, &Device::Cpu).unwrap();
        mu_t.step_t(&grads_of(mu_t.params()), lr, &half).unwrap();
        for i in 1..3 {
            let (a, b) = (plain.params()[i].var.as_tensor(), mu.params()[i].var.as_tensor());
            assert_eq!(a.flatten_all().unwrap().to_vec1::<f32>().unwrap(), b.flatten_all().unwrap().to_vec1::<f32>().unwrap(), "{}", plain.params()[i].name);
        }
        let g = g0.affine(0.5, 0.0).unwrap();
        let o = newton_schulz5(&g.affine(1.95, 0.0).unwrap(), 5).unwrap();
        let want = w0.affine(1.0 - lr * wd, 0.0).unwrap().sub(&o.affine(lr * 2.0 * 2f64.sqrt(), 0.0).unwrap()).unwrap();
        let diff = |a: &Tensor, b: &Tensor| a.sub(b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(diff(mu.params()[0].var.as_tensor(), &want) < 1e-6);
        assert!(diff(&mu.state()["m.b0.mlp.w1"], &g) < 1e-7, "the momentum is the first gradient");
        for i in 0..3 {
            assert!(diff(mu.params()[i].var.as_tensor(), mu_t.params()[i].var.as_tensor()) < 1e-7, "step_t = step");
        }
    }

    // ---------------------------------------------------------------- v42 H7 (critical periods)

    /// `step` as it was before H7 (copied unchanged), for the flags-off check.
    fn step_before_h7(o: &mut AdamW, grads: &GradStore, lr: f64, grad_scale: f64) -> Result<()> {
        o.t += 1;
        let (b1, b2) = (o.beta1, o.beta2);
        let bc1 = 1.0 - b1.powi(o.t as i32);
        let bc2 = 1.0 - b2.powi(o.t as i32);
        let step_size = lr / bc1;
        let inv_sqrt_bc2 = 1.0 / bc2.sqrt();
        for (i, p) in o.params.iter().enumerate() {
            let Some(g) = grads.get(p.var.as_tensor()) else { continue };
            if o.is_muon(i) {
                let g = if grad_scale != 1.0 { g.affine(grad_scale, 0.0)? } else { g.clone() };
                o.muon_step(i, &g.detach(), lr)?;
                continue;
            }
            if crate::fast_path() && g.device().is_cuda() {
                let sc = crate::fused::AdamWScalars { grad_scale, b1, b2, eps: o.eps, step_size, inv_sqrt_bc2, shrink: if p.decay { 1.0 - lr * o.weight_decay } else { 1.0 } };
                crate::fused::adamw_in_place(p.var.as_tensor(), &g.detach(), o.m[i].as_tensor(), o.v[i].as_tensor(), &sc)?;
                continue;
            }
            let g = if grad_scale != 1.0 { g.affine(grad_scale, 0.0)? } else { g.clone() };
            let g = g.detach();
            let m = (o.m[i].affine(b1, 0.0)? + g.affine(1.0 - b1, 0.0)?)?;
            let v = (o.v[i].affine(b2, 0.0)? + g.sqr()?.affine(1.0 - b2, 0.0)?)?;
            let denom = (v.sqrt()?.affine(inv_sqrt_bc2, 0.0)? + o.eps)?;
            let shrink = if p.decay { 1.0 - lr * o.weight_decay } else { 1.0 };
            let w = (p.var.as_tensor().affine(shrink, 0.0)? - m.div(&denom)?.affine(step_size, 0.0)?)?;
            p.var.set(&w)?;
            o.m[i].set(&m)?;
            o.v[i].set(&v)?;
        }
        Ok(())
    }

    /// `step_t`'s CUDA fast path as it was before H7 (copied unchanged), for the flags-off check on the GPU.
    fn step_t_before_h7(o: &mut AdamW, grads: &GradStore, lr: f64, scale: &Tensor) -> Result<()> {
        o.t += 1;
        let (b1, b2) = (o.beta1, o.beta2);
        let bc1 = 1.0 - b1.powi(o.t as i32);
        let bc2 = 1.0 - b2.powi(o.t as i32);
        if multi_opt_on() {
            let (mut ws, mut gs, mut ms, mut vs, mut sh) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (i, p) in o.params.iter().enumerate() {
                let Some(g) = grads.get(p.var.as_tensor()) else { continue };
                if o.is_muon(i) {
                    continue;
                }
                ws.push(p.var.as_tensor().clone());
                gs.push(g.detach());
                ms.push(o.m[i].as_tensor().clone());
                vs.push(o.v[i].as_tensor().clone());
                sh.push(if p.decay { 1.0 - lr * o.weight_decay } else { 1.0 });
            }
            let sc = crate::fused::AdamWScalars { grad_scale: f64::NAN, b1, b2, eps: o.eps, step_size: lr / bc1, inv_sqrt_bc2: 1.0 / bc2.sqrt(), shrink: f64::NAN };
            if o.muon.is_none() {
                return crate::fused::adamw_multi(&ws, &gs, &ms, &vs, &sh, scale, &sc);
            }
            if !ws.is_empty() {
                crate::fused::adamw_multi(&ws, &gs, &ms, &vs, &sh, scale, &sc)?;
            }
            return o.muon_steps_t(grads, lr, scale, None);
        }
        for (i, p) in o.params.iter().enumerate() {
            let Some(g) = grads.get(p.var.as_tensor()) else { continue };
            if o.is_muon(i) {
                continue;
            }
            let sc = crate::fused::AdamWScalars { grad_scale: f64::NAN, b1, b2, eps: o.eps, step_size: lr / bc1, inv_sqrt_bc2: 1.0 / bc2.sqrt(), shrink: if p.decay { 1.0 - lr * o.weight_decay } else { 1.0 } };
            crate::fused::adamw_in_place_s(p.var.as_tensor(), &g.detach(), o.m[i].as_tensor(), o.v[i].as_tensor(), scale, &sc)?;
        }
        if o.muon.is_some() {
            return o.muon_steps_t(grads, lr, scale, None);
        }
        Ok(())
    }

    /// Block and non-block weights (decayed 2-D ones take Muon when it is on).
    fn h7_params(dev: &Device) -> Vec<Param> {
        let spec: [(&str, &[usize], bool); 6] = [
            ("aux.head.w", &[8, 6], true),
            ("b0.mlp.w1", &[8, 16], true),
            ("b1.mem.wq", &[8, 8], true),
            ("b2.parts.tem.wo", &[8, 8], true),
            ("b3.mlp.norm", &[8], false),
            ("tok_emb", &[12, 8], false),
        ];
        spec.iter()
            .enumerate()
            .map(|(k, (name, shape, decay))| {
                let n: usize = shape.iter().product();
                let w: Vec<f32> = crate::testkit::unit_noise(n, 300 + k as u64).iter().map(|x| 0.3 * x).collect();
                Param { name: name.to_string(), var: Var::from_tensor(&Tensor::from_vec(w, *shape, dev).unwrap()).unwrap(), decay: *decay }
            })
            .collect()
    }

    /// Gradients of Σ c·w + Σ w³/3, c fixed by the parameter's name and the step (so a parameter gets the same gradient
    /// in any optimizer, as long as its weights are the same).
    fn h7_grads(ps: &[Param], t: u64) -> GradStore {
        let dev = ps[0].var.device().clone();
        let mut l = Tensor::zeros((), DType::F32, &dev).unwrap();
        for p in ps {
            let w = p.var.as_tensor();
            let h = p.name.bytes().fold(17u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
            let c = Tensor::from_vec(crate::testkit::unit_noise(w.elem_count(), h ^ (t * 7919)), w.shape(), &dev).unwrap();
            let cube = w.sqr().unwrap().mul(w).unwrap().affine(1.0 / 3.0, 0.0).unwrap();
            l = l.add(&w.mul(&c).unwrap().add(&cube).unwrap().sum_all().unwrap()).unwrap();
        }
        l.backward().unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
    }

    /// Flags off: four steps give the same bits as the code before H7, with plain AdamW and with Muon; and critical
    /// periods that never close (every multiplier 1) give those bits too.
    fn flags_off_case(dev: &Device) {
        for muon in [false, true] {
            let mk = |crit: bool| {
                let mut o = AdamW::new(h7_params(dev), 0.1).unwrap();
                if muon {
                    o.enable_muon(MuonCfg::standard(0.02, 0.01));
                }
                if crit {
                    assert_eq!(o.enable_critical_periods(CriticalPeriods { first: 1e9, gap: 1.0, tau: 5.0, floor: 0.1 }), 4);
                }
                o
            };
            let (mut new, mut old, mut never) = (mk(false), mk(false), mk(true));
            for s in 0..4 {
                let lr = 0.01 * (1.0 + s as f64);
                let scale = Tensor::new(0.6f64, dev).unwrap();
                let g_old = h7_grads(old.params(), s);
                if dev.is_cuda() {
                    new.step_t(&h7_grads(new.params(), s), lr, &scale).unwrap();
                    step_t_before_h7(&mut old, &g_old, lr, &scale).unwrap();
                } else {
                    new.step(&h7_grads(new.params(), s), lr, 0.6).unwrap();
                    step_before_h7(&mut old, &g_old, lr, 0.6).unwrap();
                }
                never.step_t(&h7_grads(never.params(), s), lr, &scale).unwrap();
                for i in 0..new.params().len() {
                    let (a, b, c) = (bits(new.params()[i].var.as_tensor()), bits(old.params()[i].var.as_tensor()), bits(never.params()[i].var.as_tensor()));
                    assert_eq!(a, b, "muon {muon} step {s} {}: not today's step", new.params()[i].name);
                    assert_eq!(a, c, "muon {muon} step {s} {}: multiplier 1 differs", new.params()[i].name);
                    assert_eq!(bits(&new.state()[&format!("m.{}", new.params()[i].name)]), bits(&old.state()[&format!("m.{}", old.params()[i].name)]));
                }
            }
        }
    }

    #[test]
    fn flags_off_step_is_todays_step() {
        flags_off_case(&Device::Cpu);
    }

    #[test]
    #[ignore]
    fn flags_off_step_is_todays_step_gpu() {
        flags_off_case(&Device::new_cuda(0).expect("no CUDA device"));
    }

    /// H7: lr_mults is CriticalPeriods::multiplier(block, step) for block weights and 1 for the rest, and each
    /// parameter's weights after every step equal (bit for bit) those of an optimizer of that parameter alone stepped
    /// at lr · multiplier: the multiplier reaches both the update and the decay, with AdamW and with Muon. The steps run
    /// through closing periods (layer l closes at 1 + 2l), and a resumed count (t set to 20) continues the curve.
    fn critical_case(dev: &Device) {
        let cp = CriticalPeriods { first: 1.0, gap: 2.0, tau: 3.0, floor: 0.25 };
        let layer = [None, Some(0), Some(1), Some(2), Some(3), None];
        for muon in [false, true] {
            let setup = |o: &mut AdamW| {
                if muon {
                    o.enable_muon(MuonCfg::standard(0.02, 0.01));
                }
            };
            let mut all = AdamW::new(h7_params(dev), 0.1).unwrap();
            setup(&mut all);
            assert_eq!(all.enable_critical_periods(cp), 4);
            let mut solo: Vec<AdamW> = h7_params(dev)
                .into_iter()
                .map(|p| {
                    let mut o = AdamW::new(vec![p], 0.1).unwrap();
                    setup(&mut o);
                    o
                })
                .collect();
            let mut closed = 0;
            for s in 0..12u64 {
                if s == 8 {
                    // a resume: the count goes on from the checkpoint's t
                    all.t = 20;
                    solo.iter_mut().for_each(|o| o.t = 20);
                }
                let idx = all.t; // 0-based index of this update
                let mu = all.lr_mults(idx).unwrap();
                for (j, l) in layer.iter().enumerate() {
                    let want = l.map_or(1.0, |l| cp.multiplier(l, idx as f64));
                    assert_eq!(mu[j].to_bits(), want.to_bits(), "{}", all.params()[j].name);
                }
                closed += mu.iter().filter(|&&x| x < 1.0).count();
                let lr = 0.01;
                let scale = Tensor::new(0.6f64, dev).unwrap();
                all.step_t(&h7_grads(all.params(), s), lr, &scale).unwrap();
                for (j, o) in solo.iter_mut().enumerate() {
                    o.step_t(&h7_grads(o.params(), s), lr * mu[j], &scale).unwrap();
                    assert_eq!(bits(all.params()[j].var.as_tensor()), bits(o.params()[0].var.as_tensor()), "muon {muon} step {s} {}", o.params()[0].name);
                }
            }
            assert!(closed > 10, "periods closed {closed} times");
            assert!(all.lr_mults(10_000).unwrap()[1] - 0.25 < 1e-9);
        }
    }

    #[test]
    fn critical_periods_scale_each_layer() {
        critical_case(&Device::Cpu);
    }

    #[test]
    #[ignore]
    fn critical_periods_scale_each_layer_gpu() {
        critical_case(&Device::new_cuda(0).expect("no CUDA device"));
    }
}
