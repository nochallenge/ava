// TESTING / DESIGN ONLY — ava_gpu smoke tests on a tiny model (CPU backend by default; set
// AVA_GPU_TEST_CUDA=1 to run the same tests on CUDA device 0).
//
// * loss at init ≈ ln 4096 (±0.5)
// * one backward gives finite, non-zero gradients for every parameter
// * 30 AdamW steps on one fixed batch drive the loss down by > 2 nats
// * the LIF straight-through spike has forward value H(u) and backward 1/(1+4|u|)²
// * kWTA keeps exactly k entries per row; cosine_lr matches the reference arithmetic
// * the fused LIF op gives the unrolled reference's spikes exactly and its gradients to 1e-4
// * three AdamW steps match the formula in f64 (moments carried in place, decay on matrices only)

use ava_gpu::model::{kwta_mask, lif_scan, lif_scan_unrolled, Config, Model, VOCAB};
use ava_gpu::optim::{clip_scale, cosine_lr, AdamW, Param};
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::{VarBuilder, VarMap};

fn device() -> Device {
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v == "1").unwrap_or(false) {
        Device::new_cuda(0).expect("CUDA device 0")
    } else {
        Device::Cpu
    }
}

fn tiny(dev: &Device, seed: u64) -> (Config, VarMap, Model) {
    let _ = dev.set_seed(seed); // the CPU backend of candle 0.11 cannot be seeded
    let cfg = Config::new(VOCAB, 32, 2, 16);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let model = Model::new(cfg, vb).unwrap();
    (cfg, varmap, model)
}

/// Deterministic pseudo-random token batch [b, t] with next-token targets.
fn batch(dev: &Device, b: usize, t: usize, seed: u64) -> (Tensor, Tensor) {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut toks = Vec::with_capacity(b * (t + 1));
    for _ in 0..b * (t + 1) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        toks.push((x % 256) as u32 + 300); // a 256-token slice of the vocab
    }
    let mut ids = Vec::with_capacity(b * t);
    let mut tg = Vec::with_capacity(b * t);
    for row in toks.chunks(t + 1) {
        ids.extend_from_slice(&row[..t]);
        tg.extend_from_slice(&row[1..]);
    }
    (Tensor::from_vec(ids, (b, t), dev).unwrap(), Tensor::from_vec(tg, (b, t), dev).unwrap())
}

fn params_of(varmap: &VarMap) -> Vec<Param> {
    let mut names: Vec<(String, Var)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    names.sort_by(|x, y| x.0.cmp(&y.0));
    names.into_iter().map(|(name, var)| Param { decay: Model::decays(&name), name, var }).collect()
}

#[test]
fn loss_at_init_is_about_ln_vocab() {
    let dev = device();
    let (_cfg, _vm, model) = tiny(&dev, 1);
    let (ids, tg) = batch(&dev, 4, 16, 7);
    let (loss, act) = model.loss(&ids, &tg).unwrap();
    let loss = loss.to_scalar::<f32>().unwrap();
    let expect = (VOCAB as f32).ln();
    println!("init loss {loss:.4} (ln V = {expect:.4}); spikes {:?}", act.spike_rate.iter().map(|t| t.to_scalar::<f32>().unwrap()).collect::<Vec<_>>());
    assert!((loss - expect).abs() < 0.5, "init loss {loss} far from ln V = {expect}");
}

#[test]
fn backward_gives_finite_nonzero_grads_for_every_param() {
    let dev = device();
    let (cfg, varmap, model) = tiny(&dev, 2);
    let (ids, tg) = batch(&dev, 4, 16, 8);
    let (loss, _) = model.loss(&ids, &tg).unwrap();
    let grads = loss.backward().unwrap();
    let params = params_of(&varmap);
    assert_eq!(params.len(), 1 + cfg.layers * 7 + 1, "parameter count");
    let (total, nonembed) = cfg.weight_counts();
    let counted: usize = params.iter().map(|p| p.var.elem_count()).sum();
    assert_eq!(counted, total);
    assert_eq!(nonembed, total - VOCAB * cfg.d);
    for p in &params {
        let g = grads.get(p.var.as_tensor()).unwrap_or_else(|| panic!("no gradient for {}", p.name));
        assert_eq!(g.shape(), p.var.shape(), "{}: grad shape", p.name);
        let v: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "{}: non-finite gradient", p.name);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("{:<14} decay={} |g| = {norm:.3e}", p.name, p.decay);
        assert!(norm > 0.0, "{}: zero gradient", p.name);
    }
    let decayed: Vec<&str> = params.iter().filter(|p| p.decay).map(|p| p.name.as_str()).collect();
    assert_eq!(decayed.len(), cfg.layers * 4, "decayed tensors: {decayed:?}");
    assert!(decayed.iter().all(|n| n.ends_with(".win") || n.ends_with(".wout") || n.ends_with(".w1") || n.ends_with(".w2")));
}

#[test]
fn thirty_steps_overfit_one_batch() {
    let dev = device();
    let (_cfg, varmap, model) = tiny(&dev, 3);
    let (ids, tg) = batch(&dev, 4, 16, 9);
    let mut opt = AdamW::new(params_of(&varmap), 0.1).unwrap();
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for step in 0..30 {
        let (loss, _) = model.loss(&ids, &tg).unwrap();
        let grads = loss.backward().unwrap();
        let gnorm = opt.grad_norm(&grads).unwrap();
        assert!(gnorm.is_finite() && gnorm > 0.0, "step {step}: grad norm {gnorm}");
        opt.step(&grads, 1e-2, clip_scale(gnorm, 1.0)).unwrap();
        let l = loss.to_scalar::<f32>().unwrap();
        assert!(l.is_finite(), "step {step}: loss {l}");
        if step == 0 {
            first = l;
        }
        last = l;
        if step % 5 == 0 || step == 29 {
            println!("step {step} loss {l:.4} gnorm {gnorm:.3}");
        }
    }
    assert!(first - last > 2.0, "loss went {first:.3} -> {last:.3}, expected a drop > 2 nats");
}

#[test]
fn lif_straight_through_forward_and_surrogate_backward() {
    let dev = device();
    // T = 1: v = I, u = I - 1, s = H(u); ds/dI = 1 / (1 + 4|u|)^2.
    let inputs = [-1.0f32, 0.0, 0.5, 0.9, 1.0, 1.1, 1.5, 3.0];
    let cur = Var::from_tensor(&Tensor::new(&inputs, &dev).unwrap().reshape((1, 1, 8)).unwrap()).unwrap();
    let beta = Tensor::new(&[2.1972f32; 8], &dev).unwrap();
    let s = lif_scan(&cur, &beta).unwrap();
    let fwd: Vec<f32> = s.flatten_all().unwrap().to_vec1().unwrap();
    for (&i, &f) in inputs.iter().zip(&fwd) {
        let expect = if i - 1.0 >= 0.0 { 1.0 } else { 0.0 };
        assert_eq!(f, expect, "forward spike for I = {i}");
    }
    let grads = s.sum_all().unwrap().backward().unwrap();
    let g: Vec<f32> = grads.get(cur.as_tensor()).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    for (&i, &gi) in inputs.iter().zip(&g) {
        let u = i - 1.0;
        let expect = 1.0 / (1.0 + 4.0 * u.abs()).powi(2);
        assert!((gi - expect).abs() < 1e-5, "surrogate grad for I = {i}: {gi} vs {expect}");
    }
    // T = 3 recurrence with beta = sigmoid(2.1972) = 0.9: I = [1.0, 0.5, 0.95] in one channel:
    // v1 = 1.0 -> s1 = 1; v2 = 0.9·1.0 + 0.5 - 1 = 0.4 -> s2 = 0; v3 = 0.36 + 0.95 = 1.31 -> s3 = 1.
    let cur = Tensor::new(&[1.0f32, 0.5, 0.95], &dev).unwrap().reshape((1, 3, 1)).unwrap();
    let beta = Tensor::new(&[2.1972f32], &dev).unwrap();
    let s: Vec<f32> = lif_scan(&cur, &beta).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(s, vec![1.0, 0.0, 1.0]);
}

#[test]
fn kwta_mask_keeps_k_per_row() {
    let dev = device();
    let _ = dev.set_seed(5);
    let u = Tensor::randn(0f32, 1f32, (6, 128), &dev).unwrap();
    let mask = kwta_mask(&u, 16).unwrap();
    let per_row: Vec<f32> = mask.sum_keepdim(D::Minus1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(per_row, vec![16.0; 6]);
    // the kept entries are the largest ones
    let (sorted, _) = u.sort_last_dim(false).unwrap();
    let kth: Vec<f32> = sorted.narrow(D::Minus1, 15, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let uv: Vec<Vec<f32>> = u.to_vec2().unwrap();
    let mv: Vec<Vec<f32>> = mask.to_vec2().unwrap();
    for r in 0..6 {
        for c in 0..128 {
            assert_eq!(mv[r][c] == 1.0, uv[r][c] >= kth[r], "row {r} col {c}");
        }
    }
}

#[test]
fn cosine_schedule_matches_reference() {
    assert!((cosine_lr(0, 10, 100, 1.0, 0.1) - 0.1).abs() < 1e-7);
    assert!((cosine_lr(9, 10, 100, 1.0, 0.1) - 1.0).abs() < 1e-7);
    assert!((cosine_lr(10, 10, 100, 1.0, 0.1) - 1.0).abs() < 1e-7);
    assert!((cosine_lr(55, 10, 100, 1.0, 0.1) - 0.55).abs() < 1e-6);
    assert!((cosine_lr(100, 10, 100, 1.0, 0.1) - 0.1).abs() < 1e-7);
    assert!((cosine_lr(500, 10, 100, 1.0, 0.1) - 0.1).abs() < 1e-7);
    assert_eq!(clip_scale(0.5, 1.0), 1.0);
    assert!((clip_scale(4.0, 1.0) - 1.0 / (4.0 + 1e-6)).abs() < 1e-12);
}

#[test]
fn fused_lif_matches_unrolled_reference() {
    // Random currents around the threshold so many neurons spike and many sit near it; the loss weights
    // every spike differently so dL/ds is not uniform. Spikes must match exactly; gradients closely.
    let dev = device();
    let (b, t, d) = (3, 40, 24);
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut rnd = |n: usize, scale: f32, shift: f32| {
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 24) as f32 * scale + shift
            })
            .collect::<Vec<f32>>()
    };
    let cur = Var::from_vec(rnd(b * t * d, 1.6, -0.3), (b, t, d), &dev).unwrap();
    let beta = Var::from_vec(rnd(d, 4.0, -1.0), d, &dev).unwrap(); // leaks from sigmoid(-1) to sigmoid(3)
    let w = Tensor::from_vec(rnd(b * t * d, 2.0, -1.0), (b, t, d), &dev).unwrap();
    let run = |f: fn(&Tensor, &Tensor) -> candle_core::Result<Tensor>| {
        let s = f(cur.as_tensor(), beta.as_tensor()).unwrap();
        let g = s.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        (flat(&s), flat(g.get(cur.as_tensor()).unwrap()), flat(g.get(beta.as_tensor()).unwrap()))
    };
    let (s_f, gc_f, gb_f) = run(lif_scan);
    let (s_r, gc_r, gb_r) = run(lif_scan_unrolled);
    let rate = s_f.iter().sum::<f32>() / s_f.len() as f32;
    assert!(rate > 0.1 && rate < 0.9, "spike rate {rate} should exercise both branches");
    assert_eq!(s_f, s_r, "fused spikes differ from the unrolled reference");
    let close = |a: &[f32], r: &[f32], what: &str| {
        let scale = r.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (i, (x, y)) in a.iter().zip(r).enumerate() {
            assert!((x - y).abs() <= 1e-4 * scale, "{what}[{i}]: fused {x} vs unrolled {y} (scale {scale})");
        }
    };
    close(&gc_f, &gc_r, "dL/dcur");
    close(&gb_f, &gb_r, "dL/dbeta_param");
}

#[test]
fn adamw_matches_reference_over_three_steps() {
    // loss = sum(w · c) has the constant gradient c; compare three steps (the moments carried in place
    // across steps) with the AdamW formula in f64, one decayed and one undecayed parameter.
    let dev = device();
    let c = [3.0f64, -0.5];
    let w0 = [1.0f64, -2.0];
    let (lr, wd, scale) = (0.1f64, 0.1f64, 0.5f64);
    let wd_var = Var::from_vec(vec![w0[0] as f32, w0[1] as f32], 2, &dev).unwrap();
    let nd_var = Var::from_vec(vec![w0[0] as f32, w0[1] as f32], 2, &dev).unwrap();
    let ct = Tensor::from_vec(vec![c[0] as f32, c[1] as f32], 2, &dev).unwrap();
    let params = vec![
        Param { name: "a.w1".into(), var: wd_var.clone(), decay: true },
        Param { name: "a.norm".into(), var: nd_var.clone(), decay: false },
    ];
    let mut opt = AdamW::new(params, wd).unwrap();
    let mut want = [[w0, [0.0; 2], [0.0; 2]], [w0, [0.0; 2], [0.0; 2]]]; // [param][w, m, v]
    for t in 1..=3 {
        let loss = (wd_var.as_tensor().mul(&ct).unwrap().sum_all().unwrap() + nd_var.as_tensor().mul(&ct).unwrap().sum_all().unwrap()).unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads, lr, scale).unwrap();
        let (bc1, bc2) = (1.0 - 0.9f64.powi(t), 1.0 - 0.95f64.powi(t));
        for (p, decay) in [(0usize, true), (1usize, false)] {
            for i in 0..2 {
                let g = c[i] * scale;
                let [w, m, v] = &mut want[p];
                m[i] = 0.9 * m[i] + 0.1 * g;
                v[i] = 0.95 * v[i] + 0.05 * g * g;
                let shrink = if decay { 1.0 - lr * wd } else { 1.0 };
                w[i] = w[i] * shrink - (lr / bc1) * m[i] / (v[i].sqrt() / bc2.sqrt() + 1e-8);
            }
        }
        for (p, var) in [(0usize, &wd_var), (1usize, &nd_var)] {
            let got = var.as_tensor().to_vec1::<f32>().unwrap();
            for i in 0..2 {
                assert!((got[i] as f64 - want[p][0][i]).abs() < 1e-5, "step {t} param {p} [{i}]: got {} want {}", got[i], want[p][0][i]);
            }
        }
    }
}
