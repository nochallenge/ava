// TESTING / DESIGN ONLY — tests of AdamW (hand-computed two-step update, decoupled decay only on
// decay=true params), global-norm gradient clipping, and the warmup + cosine learning-rate schedule.

use ava_grad::*;

#[test]
fn adamw_two_steps_match_hand_computation() {
    let mut p = Params::new();
    let w = p.add("w", Tensor::from_vec(&[1], vec![1.0]), true);
    let nd = p.add("gain", Tensor::from_vec(&[1], vec![1.0]), false);
    let mut opt = AdamW::new(0.1);
    assert_eq!((opt.beta1, opt.beta2, opt.eps), (0.9, 0.95, 1e-8));
    let lr = 0.1f32;

    // step 1, g = 0.5
    p.grad_mut(w).data[0] = 0.5;
    p.grad_mut(nd).data[0] = 0.5;
    opt.step(&mut p, lr);
    p.zero_grads();
    // m = 0.05, v = 0.0125, m̂ = 0.5, v̂ = 0.25 → update ≈ 1; decay: w −= lr·wd·w
    let w1 = 1.0 - 0.1 * 0.1 * 1.0 - 0.1 * (0.5 / (0.5 + 1e-8));
    assert!((p.value(w).data[0] as f64 - w1).abs() < 1e-6, "{}", p.value(w).data[0]);
    assert!((p.value(w).data[0] - 0.89).abs() < 1e-6);
    assert!((p.value(nd).data[0] - 0.9).abs() < 1e-6, "no decay on decay=false params");

    // step 2, g = −0.25
    p.grad_mut(w).data[0] = -0.25;
    opt.step(&mut p, lr);
    let (b1, b2): (f64, f64) = (0.9, 0.95);
    let m = b1 * 0.05 + (1.0 - b1) * -0.25;
    let v = b2 * 0.0125 + (1.0 - b2) * 0.0625;
    let mhat = m / (1.0 - b1 * b1);
    let vhat = v / (1.0 - b2 * b2);
    let w2 = w1 - 0.1 * 0.1 * w1 - 0.1 * mhat / (vhat.sqrt() + 1e-8);
    assert!((p.value(w).data[0] as f64 - w2).abs() < 1e-6, "{} vs {}", p.value(w).data[0], w2);
    assert!((w2 - 0.854263).abs() < 1e-6);
    assert_eq!(opt.steps(), 2);
}

#[test]
fn clip_grad_norm_scales_only_when_needed() {
    let mut p = Params::new();
    let a = p.add("a", Tensor::zeros(&[1]), true);
    let b = p.add("b", Tensor::zeros(&[1]), true);
    p.grad_mut(a).data[0] = 3.0;
    p.grad_mut(b).data[0] = 4.0;
    let n = clip_grad_norm(&mut p, 1.0);
    assert!((n - 5.0).abs() < 1e-6);
    assert!((p.grad(a).data[0] - 0.6).abs() < 1e-5);
    assert!((p.grad(b).data[0] - 0.8).abs() < 1e-5);
    let n2 = clip_grad_norm(&mut p, 10.0);
    assert!((n2 - 1.0).abs() < 1e-4);
    assert!((p.grad(a).data[0] - 0.6).abs() < 1e-5, "under the limit: untouched");
}

#[test]
fn cosine_schedule_endpoints() {
    let (warm, total, max, min) = (10, 110, 1.0f32, 0.1f32);
    assert!((cosine_lr(0, warm, total, max, min) - 0.1).abs() < 1e-6); // (0+1)/10 of max
    assert!((cosine_lr(9, warm, total, max, min) - 1.0).abs() < 1e-6);
    assert!((cosine_lr(10, warm, total, max, min) - 1.0).abs() < 1e-6);
    assert!((cosine_lr(60, warm, total, max, min) - 0.55).abs() < 1e-6); // halfway down
    assert!((cosine_lr(110, warm, total, max, min) - 0.1).abs() < 1e-6);
    assert!((cosine_lr(500, warm, total, max, min) - 0.1).abs() < 1e-6);
    // monotone non-increasing after warmup
    let mut prev = f32::INFINITY;
    for s in warm..=total {
        let lr = cosine_lr(s, warm, total, max, min);
        assert!(lr <= prev + 1e-7);
        prev = lr;
    }
    // no warmup: starts at max
    assert!((cosine_lr(0, 0, 100, 2.0, 0.0) - 2.0).abs() < 1e-6);
}
