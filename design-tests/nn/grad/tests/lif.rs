// TESTING / DESIGN ONLY — hand-worked checks of the leaky integrate-and-fire scan (lif_scan).
// One channel, three time steps, hard-step forward with the fast-sigmoid surrogate backward, worked out
// on paper both with the reset term detached (the default) and with it kept in the gradient.
//
// Setup: θ = 1, beta_param = 0 → β = σ(0) = 0.5, currents I = [0.8, 0.9, 0.3], loss = 1·s1 + 2·s2 + 3·s3,
// surrogate slope = 2, f'(x) = 1/(1 + 2|x|)².
//   t1: v1 = 0.8                       s1 = H(−0.2) = 0
//   t2: v2 = 0.5·0.8 + 0.9 = 1.3        s2 = H(0.3)  = 1
//   t3: v3 = 0.5·1.3 + 0.3 − 1·1 = −0.05 s3 = H(−1.05) = 0
//   f'1 = 1/1.4² = 0.5102041, f'2 = 1/1.6² = 0.390625, f'3 = 1/3.1² = 0.1040583
// Detached reset:   δv3 = 3f'3; δv2 = 2f'2 + β δv3; δv1 = 1f'1 + β δv2
// Full reset grad:  δv3 = 3f'3; δv2 = (2 − θδv3) f'2 + β δv3; δv1 = (1 − θδv2) f'1 + β δv2
// Both: dI_t = δv_t, dβ = δv2·v1 + δv3·v2, d beta_param = dβ·β(1−β).

use ava_grad::*;

fn run(detach: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut p = Params::new();
    let cur = p.add("cur", Tensor::from_vec(&[1, 3, 1], vec![0.8, 0.9, 0.3]), false);
    let beta = p.add("beta", Tensor::from_vec(&[1], vec![0.0]), false);
    let mut g = Graph::new(&p);
    let (c, b) = (g.param(cur), g.param(beta));
    let opts = LifOpts { threshold: 1.0, spike_fn: SpikeFn::Surrogate { slope: 2.0 }, detach_reset: detach };
    let s = if detach {
        // the plain entry point must default to the detached reset
        g.lif_scan(c, b, 1.0, SpikeFn::Surrogate { slope: 2.0 })
    } else {
        g.lif_scan_opts(c, b, opts)
    };
    let w = g.input(Tensor::from_vec(&[1, 3, 1], vec![1.0, 2.0, 3.0]));
    let y = g.mul(s, w);
    let l = g.sum(y);
    let grads = g.backward(l);
    (g.value(s).data.clone(), grads.get(cur).unwrap().to_vec(), grads.get(beta).unwrap().to_vec())
}

fn fp(v: f64) -> f64 {
    let a = 1.0 + 2.0 * (v - 1.0).abs();
    1.0 / (a * a)
}

#[test]
fn lif_forward_hand_worked() {
    let (s, _, _) = run(true);
    assert_eq!(s, vec![0.0, 1.0, 0.0]);
}

#[test]
fn lif_surrogate_backward_detached_reset() {
    let (_, dcur, dbeta) = run(true);
    let (v1, v2, v3) = (0.8, 1.3, -0.05);
    let dv3 = 3.0 * fp(v3);
    let dv2 = 2.0 * fp(v2) + 0.5 * dv3;
    let dv1 = 1.0 * fp(v1) + 0.5 * dv2;
    let dp = (dv2 * v1 + dv3 * v2) * 0.25;
    // paper values
    assert!((dv1 - 0.978873).abs() < 1e-5 && (dv2 - 0.937337).abs() < 1e-5 && (dv3 - 0.312175).abs() < 1e-5);
    for (got, want) in dcur.iter().zip([dv1, dv2, dv3]) {
        assert!((*got as f64 - want).abs() < 1e-5, "dcur {got} vs {want}");
    }
    assert!((dbeta[0] as f64 - dp).abs() < 1e-5, "dbeta {} vs {}", dbeta[0], dp);
    assert!((dp - 0.288924).abs() < 1e-5);
}

#[test]
fn lif_surrogate_backward_full_reset_gradient() {
    let (_, dcur, dbeta) = run(false);
    let (v1, v2, v3) = (0.8, 1.3, -0.05);
    let th = 1.0;
    let dv3 = 3.0 * fp(v3);
    let dv2 = (2.0 - th * dv3) * fp(v2) + 0.5 * dv3;
    let dv1 = (1.0 - th * dv2) * fp(v1) + 0.5 * dv2;
    let dp = (dv2 * v1 + dv3 * v2) * 0.25;
    for (got, want) in dcur.iter().zip([dv1, dv2, dv3]) {
        assert!((*got as f64 - want).abs() < 1e-5, "dcur {got} vs {want}");
    }
    assert!((dbeta[0] as f64 - dp).abs() < 1e-5, "dbeta {} vs {}", dbeta[0], dp);
}

#[test]
fn lif_batches_and_channels_are_independent() {
    // Same per-channel trace placed in different (batch, channel) slots must give the same spikes.
    let mut p = Params::new();
    let (b, t, d) = (2, 3, 2);
    let mut data = vec![0.0; b * t * d];
    let trace = [0.8, 0.9, 0.3];
    for ti in 0..t {
        data[(t + ti) * d + 1] = trace[ti]; // batch 1, channel 1
        data[ti * d] = -trace[ti]; // batch 0, channel 0: never spikes
    }
    let cur = p.add("cur", Tensor::from_vec(&[b, t, d], data), false);
    let beta = p.add("beta", Tensor::from_vec(&[d], vec![0.0, 0.0]), false);
    let mut g = Graph::new(&p);
    let (c, bp) = (g.param(cur), g.param(beta));
    let s = g.lif_scan(c, bp, 1.0, SpikeFn::Surrogate { slope: 2.0 });
    let sv = &g.value(s).data;
    let ch = |bi: usize, di: usize| (0..t).map(|ti| sv[(bi * t + ti) * d + di]).collect::<Vec<_>>();
    assert_eq!(ch(1, 1), vec![0.0, 1.0, 0.0]);
    assert_eq!(ch(0, 0), vec![0.0, 0.0, 0.0]);
    assert_eq!(ch(0, 1), vec![0.0, 0.0, 0.0]);
}
