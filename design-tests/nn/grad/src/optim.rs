// TESTING / DESIGN ONLY — ava_grad::optim
// AdamW (decoupled weight decay, bias-corrected moments), global-norm gradient clipping, and the
// linear-warmup + cosine-decay learning-rate schedule.

use crate::params::Params;

/// AdamW with per-parameter first/second moment state.
/// Update per element (t = step count after increment):
///   m = β1·m + (1−β1)·g;  v = β2·v + (1−β2)·g²
///   if decay(param): w −= lr·weight_decay·w            (decoupled, as in PyTorch AdamW)
///   w −= lr · (m / (1−β1^t)) / (sqrt(v / (1−β2^t)) + eps)
/// The optimiser reads the gradients stored in Params; it does not zero them (call zero_grads()).
#[derive(Clone, Debug)]
pub struct AdamW {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    t: u64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamW {
    /// β1 = 0.9, β2 = 0.95, eps = 1e-8.
    pub fn new(weight_decay: f32) -> AdamW {
        AdamW { beta1: 0.9, beta2: 0.95, eps: 1e-8, weight_decay, t: 0, m: Vec::new(), v: Vec::new() }
    }

    /// Number of steps taken.
    pub fn steps(&self) -> u64 {
        self.t
    }

    pub fn step(&mut self, params: &mut Params, lr: f32) {
        self.t += 1;
        let (b1, b2) = (self.beta1, self.beta2);
        let bc1 = 1.0 - (b1 as f64).powi(self.t as i32);
        let bc2 = 1.0 - (b2 as f64).powi(self.t as i32);
        let step_size = (lr as f64 / bc1) as f32;
        let inv_sqrt_bc2 = (1.0 / bc2.sqrt()) as f32;
        let eps = self.eps;
        let wd = self.weight_decay;
        let ids: Vec<_> = params.ids().collect();
        while self.m.len() < ids.len() {
            let n = params.value(ids[self.m.len()]).numel();
            self.m.push(vec![0.0; n]);
            self.v.push(vec![0.0; n]);
        }
        for id in ids {
            let decay = params.decay(id);
            let (w, g) = params.value_grad_mut(id);
            let (m, v) = (&mut self.m[id.0], &mut self.v[id.0]);
            assert_eq!(m.len(), w.data.len(), "AdamW: parameter {} changed size", id.0);
            let shrink = if decay { 1.0 - lr * wd } else { 1.0 };
            for (((wi, &gi), mi), vi) in w.data.iter_mut().zip(&g.data).zip(m.iter_mut()).zip(v.iter_mut()) {
                *mi = b1 * *mi + (1.0 - b1) * gi;
                *vi = b2 * *vi + (1.0 - b2) * gi * gi;
                let denom = vi.sqrt() * inv_sqrt_bc2 + eps;
                *wi = *wi * shrink - step_size * *mi / denom;
            }
        }
    }
}

/// Scale all stored gradients so their global L2 norm is at most `max_norm`. Returns the norm before
/// clipping.
pub fn clip_grad_norm(params: &mut Params, max_norm: f32) -> f32 {
    let norm = params.grad_norm();
    if norm > max_norm && norm > 0.0 {
        let s = max_norm / (norm + 1e-6);
        let ids: Vec<_> = params.ids().collect();
        for id in ids {
            params.grad_mut(id).data.iter_mut().for_each(|g| *g *= s);
        }
    }
    norm
}

/// Linear warmup over `warmup` steps (step 0 gets max_lr/warmup, step warmup−1 gets max_lr), then cosine
/// decay from max_lr at step `warmup` to min_lr at step `total`; min_lr afterwards.
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
