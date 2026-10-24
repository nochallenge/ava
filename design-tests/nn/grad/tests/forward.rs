// TESTING / DESIGN ONLY — forward-value and bookkeeping tests for ava_grad::Graph: exact values of
// softmax / causal mask / cross-entropy / gelu / kWTA (incl. tie rule), shapes of matmul variants,
// transpose layout, gradient accumulation across reuse and across backward passes, and the helpers.
mod common;

use ava_grad::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn matmul_values_and_shapes() {
    let p = Params::new();
    let mut g = Graph::new(&p);
    // [2,2] x [2,3]
    let a = g.input(Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let b = g.input(Tensor::from_vec(&[2, 3], vec![1.0, 0.0, -1.0, 2.0, 1.0, 0.5]));
    let c = g.matmul(a, b);
    assert_eq!(g.shape(c), &[2, 3]);
    assert_eq!(g.value(c).data, vec![5.0, 2.0, 0.0, 11.0, 4.0, -1.0]);
    // a × aᵀ
    let at = g.matmul_t(a, a);
    assert_eq!(g.value(at).data, vec![5.0, 11.0, 11.0, 25.0]);
    // flat with 3-D activations: [1,2,2] x [2,3] -> [1,2,3]
    let a3 = g.reshape(a, &[1, 2, 2]);
    let c3 = g.matmul(a3, b);
    assert_eq!(g.shape(c3), &[1, 2, 3]);
    assert_eq!(g.value(c3).data, g.value(c).data);
    // batched [2,1,2] x [2,2,1]
    let x = g.input(Tensor::from_vec(&[2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let y = g.input(Tensor::from_vec(&[2, 2, 1], vec![1.0, 1.0, 2.0, -1.0]));
    let z = g.matmul(x, y);
    assert_eq!(g.shape(z), &[2, 1, 1]);
    assert_eq!(g.value(z).data, vec![3.0, 2.0]);
}

#[test]
fn transpose_layout() {
    let p = Params::new();
    let mut g = Graph::new(&p);
    let x = g.input(Tensor::from_vec(&[2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));
    let t = g.transpose(x, 0, 1);
    assert_eq!(g.shape(t), &[3, 2]);
    assert_eq!(g.value(t).data, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    // [B,T,H,D] -> [B,H,T,D]: element (b,t,h,d) must land at (b,h,t,d)
    let (b, tt, h, d) = (2, 3, 4, 2);
    let data: Vec<f32> = (0..b * tt * h * d).map(|i| i as f32).collect();
    let x = g.input(Tensor::from_vec(&[b, tt, h, d], data.clone()));
    let y = g.transpose(x, 1, 2);
    assert_eq!(g.shape(y), &[b, h, tt, d]);
    let yv = &g.value(y).data;
    for bi in 0..b {
        for ti in 0..tt {
            for hi in 0..h {
                for di in 0..d {
                    let src = ((bi * tt + ti) * h + hi) * d + di;
                    let dst = ((bi * h + hi) * tt + ti) * d + di;
                    assert_eq!(yv[dst], data[src]);
                }
            }
        }
    }
}

#[test]
fn softmax_rows_and_causal_mask() {
    let p = Params::new();
    let mut g = Graph::new(&p);
    let mut rng = Rng::new(5);
    let x = g.input(Tensor::randn(&[2, 3, 3], 3.0, &mut rng));
    let y = g.softmax_last(x, true);
    let yv = g.value(y).data.clone();
    for m in 0..2 {
        for i in 0..3 {
            let row = &yv[(m * 3 + i) * 3..(m * 3 + i + 1) * 3];
            let s: f32 = row.iter().sum();
            assert!(close(s, 1.0, 1e-6));
            for (j, &v) in row.iter().enumerate() {
                if j > i {
                    assert_eq!(v, 0.0, "masked entry j={j} > i={i} must be 0");
                } else {
                    assert!(v > 0.0);
                }
            }
        }
    }
    // numerically stable on huge logits
    let big = g.input(Tensor::from_vec(&[1, 3], vec![1000.0, 1001.0, 999.0]));
    let s = g.softmax_last(big, false);
    let sv = &g.value(s).data;
    assert!(sv.iter().all(|v| v.is_finite()));
    let e = [(-1.0f64).exp(), 1.0, (-2.0f64).exp()];
    let z: f64 = e.iter().sum();
    for j in 0..3 {
        assert!(close(sv[j], (e[j] / z) as f32, 1e-6));
    }
}

#[test]
fn cross_entropy_value_and_ignore() {
    let p = Params::new();
    let mut g = Graph::new(&p);
    let z = g.input(Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 0.0, 0.0, 1000.0]));
    let l = g.cross_entropy(z, &[0, 2]);
    let lse0 = (1f64.exp() + 2f64.exp() + 3f64.exp()).ln();
    let row0 = lse0 - 1.0;
    let row1 = 0.0; // lse ≈ 1000 exactly, target logit 1000
    let want = ((row0 + row1) / 2.0) as f32;
    assert!(close(g.scalar(l), want, 1e-5), "{} vs {}", g.scalar(l), want);
    let l2 = g.cross_entropy(z, &[0, IGNORE]);
    assert!(close(g.scalar(l2), row0 as f32, 1e-5));
    let l3 = g.cross_entropy(z, &[IGNORE, IGNORE]);
    assert_eq!(g.scalar(l3), 0.0);
    // accuracy helper (argmax), ignoring IGNORE rows
    let logits = g.value(z).clone();
    assert!(close(accuracy(&logits, &[2, 2]), 1.0, 0.0));
    assert!(close(accuracy(&logits, &[0, 2]), 0.5, 0.0));
    assert!(close(accuracy(&logits, &[0, IGNORE]), 0.0, 0.0));
}

#[test]
fn gelu_known_values() {
    let p = Params::new();
    let mut g = Graph::new(&p);
    let x = g.input(Tensor::from_vec(&[4], vec![0.0, 1.0, -1.0, 3.0]));
    let y = g.gelu(x);
    let yv = &g.value(y).data;
    let f = |x: f64| 0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
    for (i, &xi) in [0.0, 1.0, -1.0, 3.0].iter().enumerate() {
        assert!(close(yv[i], f(xi) as f32, 1e-6));
    }
    assert!(close(yv[1], 0.841192, 1e-5));
}

#[test]
fn kwta_keeps_k_largest_and_breaks_ties_by_lower_index() {
    let p = Params::new();
    let mut g = Graph::new(&p);
    let x = g.input(Tensor::from_vec(
        &[3, 5],
        vec![
            0.1, 0.9, -0.3, 0.5, 0.2, // keep 0.9, 0.5
            1.0, 1.0, 1.0, 0.0, 1.0, // ties: keep indices 0 and 1
            -5.0, -1.0, -2.0, -3.0, -4.0, // all negative: keep -1, -2
        ],
    ));
    let y = g.kwta(x, 2);
    assert_eq!(
        g.value(y).data,
        vec![0.0, 0.9, 0.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -2.0, 0.0, 0.0]
    );
    // k >= d keeps everything, k = 0 keeps nothing
    let all = g.kwta(x, 9);
    assert_eq!(g.value(all).data, g.value(x).data);
    let none = g.kwta(x, 0);
    assert!(g.value(none).data.iter().all(|&v| v == 0.0));
    // fraction-active helper on a k-of-d code
    let mut rng = Rng::new(3);
    let r = g.input(Tensor::randn(&[10, 20], 1.0, &mut rng));
    let s = g.kwta(r, 5);
    assert!(close(fraction_active(g.value(s)), 0.25, 1e-6));
}

#[test]
fn kwta_gradient_only_through_winners() {
    let mut p = Params::new();
    let x = p.add("x", Tensor::from_vec(&[1, 4], vec![0.3, -0.2, 0.8, 0.1]), true);
    let mut g = Graph::new(&p);
    let xv = g.param(x);
    let y = g.kwta(xv, 2);
    let l = g.sum(y);
    let grads = g.backward(l);
    assert_eq!(grads.get(x).unwrap(), &[1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn reused_var_and_repeated_param_accumulate() {
    // d/dx Σ (x·x + 3x) = 2x + 3
    let mut p = Params::new();
    let x = p.add("x", Tensor::from_vec(&[3], vec![1.0, -2.0, 0.5]), true);
    let grads = {
        let mut g = Graph::new(&p);
        let a = g.param(x);
        let b = g.param(x);
        let sq = g.mul(a, b);
        let t = g.scale(a, 3.0);
        let y = g.add(sq, t);
        let l = g.sum(y);
        g.backward(l)
    };
    assert_eq!(grads.get(x).unwrap(), &[5.0, -1.0, 4.0]);
    // accumulate twice into Params (gradient accumulation over micro-batches)
    p.accumulate(&grads);
    p.accumulate(&grads);
    assert_eq!(p.grad(x).data, vec![10.0, -2.0, 8.0]);
    p.zero_grads();
    assert_eq!(p.grad(x).data, vec![0.0, 0.0, 0.0]);
}

#[test]
fn inputs_get_no_gradient_and_unused_params_are_absent() {
    let mut p = Params::new();
    let used = p.add("used", Tensor::from_vec(&[2], vec![1.0, 2.0]), true);
    let unused = p.add("unused", Tensor::from_vec(&[2], vec![1.0, 2.0]), true);
    let mut g = Graph::new(&p);
    let i = g.input(Tensor::from_vec(&[2], vec![3.0, 4.0]));
    let u = g.param(used);
    let y = g.mul(i, u);
    let l = g.sum(y);
    assert_eq!(g.scalar(l), 11.0);
    let grads = g.backward(l);
    assert_eq!(grads.get(used).unwrap(), &[3.0, 4.0]);
    assert!(grads.get(unused).is_none());
    // a loss built only from inputs yields no gradients at all
    let mut g2 = Graph::new(&p);
    let a = g2.input(Tensor::from_vec(&[2], vec![1.0, 1.0]));
    let s = g2.sum(a);
    let none = g2.backward(s);
    assert!(none.get(used).is_none());
}

#[test]
fn backward_from_non_scalar_seeds_ones() {
    let mut p = Params::new();
    let x = p.add("x", Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]), true);
    let mut g = Graph::new(&p);
    let xv = g.param(x);
    let y = g.scale(xv, 2.0);
    let grads = g.backward(y); // same as backward(sum(y))
    assert_eq!(grads.get(x).unwrap(), &[2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn graph_matmul_is_thread_count_independent() {
    let mut rng = Rng::new(9);
    let mut p = Params::new();
    let a = p.add("a", Tensor::randn(&[96, 130], 1.0, &mut rng), true);
    let b = p.add("b", Tensor::randn(&[130, 70], 1.0, &mut rng), true);
    let run = |threads: usize| {
        let mut g = Graph::new(&p);
        g.set_threads(threads);
        let (av, bv) = (g.param(a), g.param(b));
        let y = g.matmul(av, bv);
        let y = g.tanh(y);
        let l = g.sum(y);
        let grads = g.backward(l);
        (g.value(y).data.clone(), grads.get(a).unwrap().to_vec(), grads.get(b).unwrap().to_vec())
    };
    let r1 = run(1);
    for t in [2, 3, 4, 8] {
        let rt = run(t);
        assert!(r1 == rt, "results differ between 1 and {t} threads");
    }
}

#[test]
fn optimiser_round_trip_through_graph() {
    // The canonical loop: forward, backward, accumulate, clip, step, zero.
    let mut rng = Rng::new(1);
    let mut p = Params::new();
    let w = p.add("w", Tensor::randn(&[3, 1], 1.0, &mut rng), true);
    let xs = Tensor::randn(&[16, 3], 1.0, &mut rng);
    let true_w = [0.5f32, -1.0, 2.0];
    let ys: Vec<f32> = xs.data.chunks(3).map(|r| r.iter().zip(&true_w).map(|(a, b)| a * b).sum()).collect();
    let ys = Tensor::from_vec(&[16, 1], ys);
    let mut opt = AdamW::new(0.0);
    let mut first = 0.0;
    let mut last = 0.0;
    for step in 0..300 {
        let (loss, grads) = {
            let mut g = Graph::new(&p);
            let x = g.input(xs.clone());
            let wv = g.param(w);
            let pred = g.matmul(x, wv);
            let y = g.input(ys.clone());
            let e = g.sub(pred, y);
            let e2 = g.mul(e, e);
            let l = g.mean(e2);
            (g.scalar(l), g.backward(l))
        };
        if step == 0 {
            first = loss;
        }
        last = loss;
        p.accumulate(&grads);
        clip_grad_norm(&mut p, 10.0);
        opt.step(&mut p, 0.05);
        p.zero_grads();
    }
    assert!(last < first * 1e-3, "loss {first} -> {last}");
}
