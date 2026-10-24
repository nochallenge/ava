// TESTING / DESIGN ONLY — shared helpers for the ava_grad integration tests.
// A central finite-difference gradient checker (perturb every scalar of every listed parameter by ±1e-3,
// compare with the analytic gradient from Graph::backward under |a−n| ≤ 2e-2·max(|a|,|n|) + 1e-3), and a
// "random weighted sum" loss so every output element of an op receives its own distinct upstream gradient.
#![allow(dead_code, clippy::needless_range_loop)]

use ava_grad::{Graph, ParamId, Params, Rng, Tensor, Var};

pub const H: f32 = 1e-3;

/// loss = Σ y ⊙ w  (w is a fixed random tensor fed as a constant input).
pub fn weighted(g: &mut Graph, y: Var, w: &Tensor) -> Var {
    let wv = g.input(w.clone());
    let p = g.mul(y, wv);
    g.sum(p)
}

pub fn loss_at<F: Fn(&mut Graph) -> Var>(params: &Params, build: &F) -> f32 {
    let mut g = Graph::new(params);
    let l = build(&mut g);
    g.value(l).data[0]
}

/// Central finite differences for every scalar of every parameter in `ids`.
pub fn fd_check<F: Fn(&mut Graph) -> Var>(params: &mut Params, ids: &[ParamId], build: F) {
    let grads = {
        let mut g = Graph::new(params);
        let l = build(&mut g);
        assert_eq!(g.value(l).numel(), 1, "loss must be a scalar");
        g.backward(l)
    };
    for &id in ids {
        let n = params.value(id).numel();
        let analytic: Vec<f32> = match grads.get(id) {
            Some(s) => s.to_vec(),
            None => vec![0.0; n],
        };
        let mut max_abs = 0f32;
        for i in 0..n {
            let orig = params.value(id).data[i];
            let up = orig + H;
            let dn = orig - H;
            params.value_mut(id).data[i] = up;
            let lp = loss_at(params, &build);
            params.value_mut(id).data[i] = dn;
            let lm = loss_at(params, &build);
            params.value_mut(id).data[i] = orig;
            let num = ((lp as f64 - lm as f64) / (up as f64 - dn as f64)) as f32;
            let a = analytic[i];
            max_abs = max_abs.max(a.abs());
            let tol = 2e-2 * a.abs().max(num.abs()) + 1e-3;
            assert!(
                (a - num).abs() <= tol,
                "{}[{}]: analytic {} vs numeric {} (tol {})",
                params.name(id),
                i,
                a,
                num,
                tol
            );
        }
        assert!(max_abs > 1e-4, "{}: gradient is ~0 everywhere, the check would be vacuous", params.name(id));
    }
}

pub fn rt(shape: &[usize], rng: &mut Rng) -> Tensor {
    Tensor::randn(shape, 1.0, rng)
}

/// Random values bounded away from zero (for ops with a kink at 0).
pub fn away_from_zero(shape: &[usize], rng: &mut Rng) -> Tensor {
    let mut t = rt(shape, rng);
    for x in t.data.iter_mut() {
        *x = x.signum() * (0.05 + x.abs());
    }
    t
}
