// TESTING / DESIGN ONLY
//! E3: e-prop, learning forward in time with eligibility traces (Bellec et al. 2020). Backprop through
//! time keeps every past state and runs backward; no brain can. In e-prop each synapse keeps a fading
//! trace of its recent inputs (its eligibility), and a learning signal broadcast to each neuron turns the
//! trace into a weight change, like a third factor (dopamine, or a top-down error) in three-factor
//! plasticity. Memory is per synapse, not per time step, so she could learn while she reads and talks.
//!
//! The layer (the gpu crate's LIF, reset by subtraction with the reset path detached, as Bellec et al.):
//!   v_t = α·v_{t−1} + x_t·W_in + z_{t−1}·W_rec − θ·z_{t−1},   z_t = H(v_t − θ),   y_t = z_t·W_out
//!   E = ½ Σ_t ‖y_t − y*_t‖²
//! e-prop, with ψ_t = 1 / (1 + slope·|v_t − θ|)² (the same surrogate as `crate::spike`):
//!   x̄_t = α·x̄_{t−1} + x_t,   z̄_t = α·z̄_{t−1} + z_{t−1}          (the eligibility vectors)
//!   L_t = (y_t − y*_t)·Bᵀ                                         (the learning signal; B = W_out, or a
//!                                                                  fixed random B for "random e-prop")
//!   ΔW_in[i,j] ∝ Σ_t L_t[j]·ψ_t[j]·x̄_t[i],   ΔW_rec[k,j] ∝ Σ_t L_t[j]·ψ_t[j]·z̄_t[k]
//! With no recurrent weights this is exactly the BPTT gradient (the test checks it against candle's
//! autodiff); with them it drops the paths through other neurons' future spikes, and the test measures
//! how close it stays.

pub struct Lif {
    pub n_in: usize,
    pub n: usize,
    pub n_out: usize,
    /// [n_in][n], [n][n], [n][n_out], row-major.
    pub w_in: Vec<f32>,
    pub w_rec: Vec<f32>,
    pub w_out: Vec<f32>,
    pub alpha: f32,
    pub theta: f32,
    pub slope: f32,
}

/// The running state and the gradient accumulators of one sequence, updated one step at a time.
pub struct Eprop {
    v: Vec<f32>,
    z: Vec<f32>,
    xbar: Vec<f32>,
    zbar: Vec<f32>,
    pub g_in: Vec<f32>,
    pub g_rec: Vec<f32>,
    pub g_out: Vec<f32>,
    pub loss: f32,
    pub spikes: usize,
}

impl Eprop {
    pub fn new(l: &Lif) -> Eprop {
        Eprop {
            v: vec![0.0; l.n],
            z: vec![0.0; l.n],
            xbar: vec![0.0; l.n_in],
            zbar: vec![0.0; l.n],
            g_in: vec![0.0; l.n_in * l.n],
            g_rec: vec![0.0; l.n * l.n],
            g_out: vec![0.0; l.n * l.n_out],
            loss: 0.0,
            spikes: 0,
        }
    }

    /// One time step: input x [n_in], target [n_out], feedback b [n][n_out] (pass `&l.w_out` for
    /// symmetric e-prop). Returns the output y [n_out].
    pub fn step(&mut self, l: &Lif, x: &[f32], target: &[f32], b: &[f32]) -> Vec<f32> {
        let (n, n_in, n_out) = (l.n, l.n_in, l.n_out);
        for i in 0..n_in {
            self.xbar[i] = l.alpha * self.xbar[i] + x[i];
        }
        for k in 0..n {
            self.zbar[k] = l.alpha * self.zbar[k] + self.z[k];
        }
        let mut v = vec![0f32; n];
        for j in 0..n {
            let mut cur = 0.0;
            for i in 0..n_in {
                cur += x[i] * l.w_in[i * n + j];
            }
            for k in 0..n {
                cur += self.z[k] * l.w_rec[k * n + j];
            }
            v[j] = l.alpha * self.v[j] + cur - l.theta * self.z[j];
        }
        let z: Vec<f32> = v.iter().map(|&vj| if vj - l.theta >= 0.0 { 1.0 } else { 0.0 }).collect();
        self.spikes += z.iter().filter(|&&s| s > 0.0).count();
        let mut y = vec![0f32; n_out];
        for o in 0..n_out {
            for j in 0..n {
                y[o] += z[j] * l.w_out[j * n_out + o];
            }
        }
        let err: Vec<f32> = (0..n_out).map(|o| y[o] - target[o]).collect();
        self.loss += 0.5 * err.iter().map(|e| e * e).sum::<f32>();
        for j in 0..n {
            let mut sig = 0.0;
            for o in 0..n_out {
                sig += err[o] * b[j * n_out + o];
                self.g_out[j * n_out + o] += z[j] * err[o];
            }
            let psi = 1.0 / (1.0 + l.slope * (v[j] - l.theta).abs()).powi(2);
            let ls = sig * psi;
            for i in 0..n_in {
                self.g_in[i * n + j] += ls * self.xbar[i];
            }
            for k in 0..n {
                self.g_rec[k * n + j] += ls * self.zbar[k];
            }
        }
        self.v = v;
        self.z = z;
        y
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    d / (na * nb + 1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, Var};

    fn vals(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        crate::randn_seeded(n, seed, &Device::Cpu).unwrap().to_vec1::<f32>().unwrap().iter().map(|v| v * scale).collect()
    }

    /// BPTT through candle's autodiff, the same layer, with the reset path detached.
    fn bptt(l: &Lif, xs: &[Vec<f32>], ts: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        let dev = Device::Cpu;
        let w_in = Var::from_vec(l.w_in.clone(), (l.n_in, l.n), &dev).unwrap();
        let w_rec = Var::from_vec(l.w_rec.clone(), (l.n, l.n), &dev).unwrap();
        let w_out = Var::from_vec(l.w_out.clone(), (l.n, l.n_out), &dev).unwrap();
        let mut v = Tensor::zeros((1, l.n), candle_core::DType::F32, &dev).unwrap();
        let mut z = v.clone();
        let mut loss = Tensor::new(0f32, &dev).unwrap();
        for (x, t) in xs.iter().zip(ts) {
            let x = Tensor::from_vec(x.clone(), (1, l.n_in), &dev).unwrap();
            let t = Tensor::from_vec(t.clone(), (1, l.n_out), &dev).unwrap();
            let cur = (x.matmul(w_in.as_tensor()).unwrap() + z.matmul(w_rec.as_tensor()).unwrap()).unwrap();
            v = ((v * l.alpha as f64).unwrap() + cur).unwrap().sub(&(z.detach() * l.theta as f64).unwrap()).unwrap();
            z = crate::spike(&(&v - l.theta as f64).unwrap(), l.slope as f64).unwrap();
            let y = z.matmul(w_out.as_tensor()).unwrap();
            loss = (loss + ((y - t).unwrap().sqr().unwrap().sum_all().unwrap() * 0.5).unwrap()).unwrap();
        }
        let g = loss.backward().unwrap();
        let f = |w: &Var| g.get(w.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        (f(&w_in), f(&w_rec), f(&w_out), loss.to_scalar::<f32>().unwrap())
    }

    fn setup(rec_scale: f32) -> (Lif, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let (n_in, n, n_out, steps) = (5, 8, 3, 30);
        let l = Lif {
            n_in,
            n,
            n_out,
            w_in: vals(n_in * n, 1, 0.8),
            w_rec: vals(n * n, 2, rec_scale),
            w_out: vals(n * n_out, 3, 0.5),
            alpha: 0.9,
            theta: 1.0,
            slope: 5.0,
        };
        let xs: Vec<Vec<f32>> = (0..steps).map(|t| vals(n_in, 100 + t as u64, 0.7)).collect();
        let ts: Vec<Vec<f32>> = (0..steps).map(|t| vals(n_out, 200 + t as u64, 0.5)).collect();
        (l, xs, ts)
    }

    fn run(l: &Lif, xs: &[Vec<f32>], ts: &[Vec<f32>], b: &[f32]) -> Eprop {
        let mut e = Eprop::new(l);
        for (x, t) in xs.iter().zip(ts) {
            e.step(l, x, t, b);
        }
        e
    }

    fn rel(a: &[f32], b: &[f32]) -> f32 {
        let d: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
        d / b.iter().map(|y| y * y).sum::<f32>().sqrt().max(1e-12)
    }

    #[test]
    fn without_recurrence_eprop_is_exactly_bptt() {
        let (l, xs, ts) = setup(0.0);
        let e = run(&l, &xs, &ts, &l.w_out);
        let steps = xs.len() * l.n;
        assert!(e.spikes > steps / 20 && e.spikes < steps * 19 / 20, "spikes {} of {steps}", e.spikes);
        let (g_in, g_rec, g_out, loss) = bptt(&l, &xs, &ts);
        assert!((e.loss - loss).abs() < 1e-4 * loss.max(1.0), "{} {}", e.loss, loss);
        assert!(rel(&e.g_in, &g_in) < 1e-4, "{}", rel(&e.g_in, &g_in));
        assert!(rel(&e.g_rec, &g_rec) < 1e-4, "{}", rel(&e.g_rec, &g_rec));
        assert!(rel(&e.g_out, &g_out) < 1e-4);
    }

    #[test]
    fn with_recurrence_eprop_stays_close_to_bptt() {
        let (l, xs, ts) = setup(0.3);
        let e = run(&l, &xs, &ts, &l.w_out);
        let (g_in, g_rec, _, _) = bptt(&l, &xs, &ts);
        let (c_in, c_rec) = (cosine(&e.g_in, &g_in), cosine(&e.g_rec, &g_rec));
        eprintln!("symmetric e-prop vs BPTT, recurrent: cos W_in {c_in:.3}, W_rec {c_rec:.3}");
        assert!(c_in > 0.7 && c_rec > 0.5, "{c_in} {c_rec}");
        // Random feedback (a fixed random B instead of W_out) is only measured here: it is not BPTT's
        // gradient at the start, and it helps only once W_out has aligned with B during training.
        let b = vals(l.n * l.n_out, 9, 0.5);
        let r = run(&l, &xs, &ts, &b);
        eprintln!("random e-prop vs BPTT at the start: cos W_in {:.3}", cosine(&r.g_in, &g_in));
    }

    #[test]
    fn eprop_training_lowers_the_loss() {
        // Plain gradient steps with the online e-prop gradient, recurrent layer, symmetric feedback.
        let (mut l, xs, ts) = setup(0.3);
        let first = run(&l, &xs, &ts, &l.w_out.clone()).loss;
        for _ in 0..60 {
            let b = l.w_out.clone();
            let e = run(&l, &xs, &ts, &b);
            for (w, g) in l.w_in.iter_mut().zip(&e.g_in) {
                *w -= 0.02 * g;
            }
            for (w, g) in l.w_rec.iter_mut().zip(&e.g_rec) {
                *w -= 0.02 * g;
            }
            for (w, g) in l.w_out.iter_mut().zip(&e.g_out) {
                *w -= 0.02 * g;
            }
        }
        let last = run(&l, &xs, &ts, &l.w_out.clone()).loss;
        assert!(last < 0.8 * first, "{first} -> {last}");
    }
}
