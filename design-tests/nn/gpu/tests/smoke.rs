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
// * feeding tokens one by one with the carried LIF state (Model::step) gives the full forward's logits
// * fast-weight models F, G and R: weight counts, learning, carried state = full forward, G at init = F

use ava_gpu::model::{kwta_mask, lif_scan, lif_scan_unrolled, Config, Kind, Model, VOCAB};
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

#[test]
fn carried_state_steps_match_the_full_forward() {
    let dev = device();
    let (cfg, _vm, model) = tiny(&dev, 11);
    let (ids, _) = batch(&dev, 1, cfg.ctx, 21);
    let full = model.forward(&ids).unwrap(); // [ctx, V]
    let mut st = model.stream().unwrap();
    let toks: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
    let mut worst = 0f32;
    for (p, &t) in toks.iter().enumerate() {
        let a = model.step(&mut st, t).unwrap();
        let b = full.narrow(0, p, 1).unwrap().squeeze(0).unwrap();
        worst = worst.max(a.sub(&b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap());
    }
    assert_eq!(st.seen, cfg.ctx);
    assert!(worst < 1e-4, "carried-state logits differ from the full forward by {worst}");
}

fn tiny_f(dev: &Device, kind: Kind) -> (Config, VarMap, Model) {
    // A gets 6 layers (attention in blocks 1 and 5, fast weights in the rest); the others 2.
    let cfg = Config::with_kind(kind, VOCAB, 64, if kind == Kind::A { 6 } else { 2 }, 24);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let model = Model::new(cfg, vb).unwrap();
    (cfg, varmap, model)
}

fn counts_and_learns(kind: Kind) {
    let dev = device();
    let (cfg, varmap, model) = tiny_f(&dev, kind);
    let counted: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    assert_eq!(counted, cfg.weight_counts().0, "weight_counts disagrees with the parameters created");
    let (ids, tg) = batch(&dev, 4, cfg.ctx, 7);
    let params = params_of(&varmap);
    let vars: Vec<Var> = params.iter().map(|p| p.var.clone()).collect();
    let mut opt = AdamW::new(params, 0.1).unwrap();
    let first = model.loss(&ids, &tg).unwrap().0.to_scalar::<f32>().unwrap();
    let mut last = first;
    for _ in 0..30 {
        let (loss, _) = model.loss(&ids, &tg).unwrap();
        last = loss.to_scalar::<f32>().unwrap();
        let grads = loss.backward().unwrap();
        for v in &vars {
            let g = grads.get(v).expect("every parameter gets a gradient");
            assert!(g.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap().is_finite());
        }
        let gnorm = opt.grad_norm(&grads).unwrap();
        opt.step(&grads, 1e-2, clip_scale(gnorm, 1.0)).unwrap();
    }
    assert!(first - last > 2.0, "{kind:?} did not overfit one batch: {first} -> {last}");
}

fn carried_state_matches(kind: Kind) {
    let dev = device();
    let (cfg, _vm, model) = tiny_f(&dev, kind);
    let (ids, _) = batch(&dev, 1, cfg.ctx, 23);
    let full = model.forward(&ids).unwrap();
    let mut st = model.stream().unwrap();
    let toks: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
    let mut worst = 0f32;
    for (p, &t) in toks.iter().enumerate() {
        let a = model.step(&mut st, t).unwrap();
        let b = full.narrow(0, p, 1).unwrap().squeeze(0).unwrap();
        worst = worst.max(a.sub(&b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap());
    }
    assert!(worst < 1e-3, "{kind:?} carried-state logits differ from the full forward by {worst}");
}

#[test]
fn fast_weight_models_count_their_weights_and_learn() {
    counts_and_learns(Kind::F);
    counts_and_learns(Kind::G);
    counts_and_learns(Kind::R);
    counts_and_learns(Kind::D);
    counts_and_learns(Kind::A);
}

#[test]
fn fast_weight_carried_state_matches_the_full_forward() {
    carried_state_matches(Kind::F);
    carried_state_matches(Kind::G);
    carried_state_matches(Kind::R);
    carried_state_matches(Kind::D);
    carried_state_matches(Kind::A);
}

#[test]
fn gated_fast_weights_start_exactly_as_fixed_ones() {
    // G's gates start at alpha = gamma_h, beta = 1: same weights in, same logits out as F.
    let dev = device();
    let (_cf, vf, mf) = tiny_f(&dev, Kind::F);
    let (_cg, vg, mg) = tiny_f(&dev, Kind::G);
    let src = vf.data().lock().unwrap().clone();
    for (name, var) in vg.data().lock().unwrap().iter() {
        if let Some(v) = src.get(name) {
            var.set(v.as_tensor()).unwrap();
        }
    }
    let (ids, _) = batch(&dev, 2, 24, 31);
    let diff = mf.forward(&ids).unwrap().sub(&mg.forward(&ids).unwrap()).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(diff < 1e-4, "G at init differs from F by {diff}");
}

#[test]
fn delta_rule_matches_its_recurrence_over_full_chunks_and_repeated_keys() {
    // ctx 128 = two chunks of 64 (all six doubling levels of the inverse); the second half is one token
    // repeated, so its keys are identical (where a power-series inverse would cancel catastrophically).
    let dev = device();
    let cfg = Config::with_kind(Kind::D, VOCAB, 64, 2, 128);
    let varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &dev)).unwrap();
    let toks: Vec<u32> = (0..128u32).map(|i| if i < 64 { (i * 7919 + 13) % VOCAB as u32 } else { 42 }).collect();
    let full = model.forward(&Tensor::from_vec(toks.clone(), (1, 128), &dev).unwrap()).unwrap();
    let mut st = model.stream().unwrap();
    let mut worst = 0f32;
    for (p, &t) in toks.iter().enumerate() {
        let a = model.step(&mut st, t).unwrap();
        let b = full.narrow(0, p, 1).unwrap().squeeze(0).unwrap();
        worst = worst.max(a.sub(&b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap());
    }
    assert!(worst < 1e-3, "D carried-state logits differ from the chunked forward by {worst}");
}

/// Random [rows, cols] values in about (-2, 2) from a xorshift seed.
fn noise(dev: &Device, rows: usize, cols: usize, seed: u64) -> Tensor {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let v: Vec<f32> = (0..rows * cols)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect();
    Tensor::from_vec(v, (rows, cols), dev).unwrap()
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.sub(b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
}

#[test]
fn fused_ops_match_the_primitive_chains() {
    // Speed round 2 kernels (CUDA only): values and gradients against candle's primitive chains.
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("fused ops: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    // rmsnorm forward and backward (dx and the gain gradient), N not a multiple of the 64-row partials
    let (n, d) = (1000, 384);
    let x = Var::from_tensor(&noise(&dev, n, d, 3)).unwrap();
    let g = Var::from_tensor(&noise(&dev, 1, d, 4).squeeze(0).unwrap()).unwrap();
    let w = noise(&dev, n, d, 5);
    let y_ref = ava_gpu::model::rmsnorm(x.as_tensor(), g.as_tensor()).unwrap();
    let y_new = ava_gpu::fused::rms_norm(x.as_tensor(), g.as_tensor(), ava_gpu::model::NORM_EPS).unwrap();
    assert!(max_diff(&y_ref, &y_new) < 1e-5, "rms fwd {}", max_diff(&y_ref, &y_new));
    let gr = y_ref.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
    let gn = y_new.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
    for (name, v) in [("dx", &x), ("dg", &g)] {
        let (a, b) = (gr.get(v.as_tensor()).unwrap(), gn.get(v.as_tensor()).unwrap());
        let scale = a.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(max_diff(a, b) < 1e-5 * scale.max(1.0), "rms {name}: {} (scale {scale})", max_diff(a, b));
    }
    // kWTA + relu: exactly relu(u * mask) forward, and the same gradient
    let (n, h, k) = (512, 1536, 192);
    let u = Var::from_tensor(&noise(&dev, n, h, 6)).unwrap();
    let w = noise(&dev, n, h, 7);
    let a_ref = u.as_tensor().mul(&ava_gpu::model::kwta_mask(u.as_tensor(), k).unwrap()).unwrap().relu().unwrap();
    let a_new = ava_gpu::fused::kwta_relu(u.as_tensor(), k).unwrap();
    assert_eq!(max_diff(&a_ref, &a_new), 0.0, "kwta-relu forward differs");
    let gr = a_ref.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
    let gn = a_new.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
    assert_eq!(max_diff(gr.get(u.as_tensor()).unwrap(), gn.get(u.as_tensor()).unwrap()), 0.0, "kwta-relu backward differs");
    // AdamW: one fused step against the primitive formula
    let (wt, gt, mt, vt) = (noise(&dev, 64, 96, 8), noise(&dev, 64, 96, 9), noise(&dev, 64, 96, 10), noise(&dev, 64, 96, 11).abs().unwrap());
    let (b1, b2, eps, lr, wd, s, t) = (0.9f64, 0.95f64, 1e-8f64, 3e-3f64, 0.1f64, 0.7f64, 5i32);
    let (bc1, bc2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
    let sc = ava_gpu::fused::AdamWScalars { grad_scale: s, b1, b2, eps, step_size: lr / bc1, inv_sqrt_bc2: 1.0 / bc2.sqrt(), shrink: 1.0 - lr * wd };
    let out = ava_gpu::fused::adamw(&wt, &gt, &mt, &vt, &sc).unwrap();
    let g = gt.affine(s, 0.0).unwrap();
    let m = (mt.affine(b1, 0.0).unwrap() + g.affine(1.0 - b1, 0.0).unwrap()).unwrap();
    let v = (vt.affine(b2, 0.0).unwrap() + g.sqr().unwrap().affine(1.0 - b2, 0.0).unwrap()).unwrap();
    let den = (v.sqrt().unwrap().affine(1.0 / bc2.sqrt(), 0.0).unwrap() + eps).unwrap();
    let w = (wt.affine(1.0 - lr * wd, 0.0).unwrap() - m.div(&den).unwrap().affine(lr / bc1, 0.0).unwrap()).unwrap();
    for (i, r) in [w, m, v].iter().enumerate() {
        let got = out.get(i).unwrap().reshape((64, 96)).unwrap();
        assert!(max_diff(&got, r) < 1e-6, "adamw part {i}: {}", max_diff(&got, r));
    }
}

#[test]
fn hand_written_s_backward_matches_autograd() {
    // Speed round 3 (CUDA only): Model::loss_and_grads against loss + candle's backward, every parameter.
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("manual backward: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    let cfg = Config::with_kind(Kind::S, VOCAB, 64, 2, 32);
    let varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &dev)).unwrap();
    let (ids, tg) = batch(&dev, 4, cfg.ctx, 11);
    let (l_ref, _) = model.loss(&ids, &tg).unwrap();
    let g_ref = l_ref.backward().unwrap();
    let (l_new, act, g_new) = model.loss_and_grads_with(&ids, &tg, false).unwrap().expect("S on CUDA takes the manual path");
    let (a, b) = (l_ref.to_scalar::<f32>().unwrap(), l_new.to_scalar::<f32>().unwrap());
    assert!((a - b).abs() < 1e-5, "loss {a} vs {b}");
    assert_eq!(act.spike_rate.len(), 2);
    let mut n = 0;
    for p in params_of(&varmap) {
        let (r, m) = (g_ref.get(p.var.as_tensor()).expect("autograd grad"), g_new.get(p.var.as_tensor()).expect("manual grad"));
        let scale = r.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        let diff = max_diff(r, m);
        assert!(diff <= 1e-4 * scale.max(1e-6), "{}: max |diff| {diff} (scale {scale})", p.name);
        n += 1;
    }
    assert_eq!(n, 16, "every S parameter checked");
}

#[test]
fn fused_attention_matches_the_candle_chain() {
    // Speed round 4 (CUDA only): causal attention values and q, k, v gradients against softmax(q kᵀ s + mask) v.
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("fused attention: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    for (b, nh, t, seed) in [(2usize, 3usize, 100usize, 20u64), (1, 2, 256, 30), (3, 1, 33, 40)] {
        let mk = |s: u64| Var::from_tensor(&noise(&dev, b * nh * t, 32, s).reshape((b, nh, t, 32)).unwrap()).unwrap();
        let (q, k, v) = (mk(seed), mk(seed + 1), mk(seed + 2));
        let w = noise(&dev, b * nh * t, 32, seed + 3).reshape((b, nh, t, 32)).unwrap();
        let scale = 1.0 / 32f64.sqrt();
        let sc = q.as_tensor().matmul(&k.as_tensor().t().unwrap().contiguous().unwrap()).unwrap().affine(scale, 0.0).unwrap();
        let mask = ava_gpu::model::causal_mask(t, &dev).unwrap();
        let o_ref = candle_nn::ops::softmax(&sc.broadcast_add(&mask).unwrap(), candle_core::D::Minus1).unwrap().matmul(v.as_tensor()).unwrap();
        let o_new = ava_gpu::fused::causal_attention(q.as_tensor(), k.as_tensor(), v.as_tensor(), scale).unwrap();
        assert!(max_diff(&o_ref, &o_new) < 1e-5, "attn fwd t={t}: {}", max_diff(&o_ref, &o_new));
        let gr = o_ref.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        let gn = o_new.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        for (name, x) in [("dq", &q), ("dk", &k), ("dv", &v)] {
            let (a, c) = (gr.get(x.as_tensor()).unwrap(), gn.get(x.as_tensor()).unwrap());
            let mag = a.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
            assert!(max_diff(a, c) < 1e-5 * mag.max(1.0), "attn {name} t={t}: {} (scale {mag})", max_diff(a, c));
        }
    }
}

#[test]
fn fused_fast_weight_read_matches_the_chain() {
    // Speed round 10 (CUDA only): R's read y_i = sum_{j<=i} scale (q_i.k_j) exp(c_i - c_j) beta_j v_j as one op
    // (beta folded into k, c packed after v) against the [T, T] chain R used: values and the gradients of q, k, v,
    // log alpha and beta through a random projection, within float rounding (f32 on both sides).
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("fused fast-weight read: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    for (b, nh, t, seed) in [(2usize, 3usize, 100usize, 110u64), (1, 2, 256, 120), (3, 1, 33, 130)] {
        let mk = |s: u64, cols: usize| noise(&dev, b * nh * t, cols, s);
        let unit = |x: Tensor| -> Tensor { x.broadcast_div(&x.sqr().unwrap().sum_keepdim(1).unwrap().sqrt().unwrap()).unwrap() };
        let q = Var::from_tensor(&unit(mk(seed, 32)).reshape((b, nh, t, 32)).unwrap()).unwrap();
        let k = Var::from_tensor(&unit(mk(seed + 1, 32)).reshape((b, nh, t, 32)).unwrap()).unwrap();
        let v = Var::from_tensor(&mk(seed + 2, 32).reshape((b, nh, t, 32)).unwrap()).unwrap();
        // log alpha in about [-0.2, 0] (slow to fast forgetting), beta in [0, 2]
        let la = Var::from_tensor(&mk(seed + 3, 1).abs().unwrap().affine(-0.1, 0.0).unwrap().reshape((b, nh, t)).unwrap()).unwrap();
        let beta = Var::from_tensor(&mk(seed + 4, 1).affine(0.5, 1.0).unwrap().reshape((b, nh, t)).unwrap()).unwrap();
        let wproj = mk(seed + 5, 32).reshape((b, nh, t, 32)).unwrap();
        let scale = 1.0 / 32f64.sqrt();
        let c = la.as_tensor().cumsum(2).unwrap();
        // the chain R used
        let sc = q.as_tensor().matmul(&k.as_tensor().t().unwrap().contiguous().unwrap()).unwrap().affine(scale, 0.0).unwrap();
        let mask = ava_gpu::model::causal_mask(t, &dev).unwrap();
        let dmat = c.unsqueeze(3).unwrap().broadcast_sub(&c.unsqueeze(2).unwrap()).unwrap().broadcast_add(&mask).unwrap().exp().unwrap();
        let w = sc.mul(&dmat).unwrap().broadcast_mul(&beta.as_tensor().unsqueeze(2).unwrap()).unwrap();
        let y_ref = w.matmul(v.as_tensor()).unwrap();
        // the fused op
        let kb = k.as_tensor().broadcast_mul(&beta.as_tensor().unsqueeze(3).unwrap()).unwrap();
        let vc = Tensor::cat(&[v.as_tensor(), &c.unsqueeze(3).unwrap()], 3).unwrap();
        let y_new = ava_gpu::fused::gla(q.as_tensor(), &kb, &vc, scale).unwrap();
        let mag = y_ref.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(max_diff(&y_ref, &y_new) < 1e-5 * mag.max(1.0), "gla fwd t={t}: {} (scale {mag})", max_diff(&y_ref, &y_new));
        let gr = y_ref.mul(&wproj).unwrap().sum_all().unwrap().backward().unwrap();
        let gn = y_new.mul(&wproj).unwrap().sum_all().unwrap().backward().unwrap();
        for (name, x) in [("dq", q.as_tensor()), ("dk", k.as_tensor()), ("dv", v.as_tensor()), ("dlog_alpha", la.as_tensor()), ("dbeta", beta.as_tensor())] {
            let (a, n) = (gr.get(x).unwrap(), gn.get(x).unwrap());
            let mag = a.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
            eprintln!("gla t={t} {name}: max diff {:.3e} (scale {mag:.3e})", max_diff(a, n));
            assert!(max_diff(a, n) < 1e-4 * mag.max(1e-3), "gla {name} t={t}: {} (scale {mag})", max_diff(a, n));
        }
    }
}

#[test]
fn fused_delta_rule_matches_the_chunked_form() {
    // Speed round 11 (CUDA only): D's gated delta rule as one recurrence op against the chunked WY form D used
    // (model::delta_rule): values and the gradients of q, k, v, log alpha and beta through a random projection,
    // f32 on both sides. t = 100, 256, 33 give chunks of 4, 64 and 1.
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("fused delta rule: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    for (b, nh, t, seed) in [(2usize, 3usize, 100usize, 210u64), (1, 2, 256, 220), (3, 1, 33, 230)] {
        let mk = |s: u64, cols: usize| noise(&dev, b * nh * t, cols, s);
        let unit = |x: Tensor| -> Tensor { x.broadcast_div(&x.sqr().unwrap().sum_keepdim(1).unwrap().sqrt().unwrap()).unwrap() };
        let q = Var::from_tensor(&unit(mk(seed, 32)).reshape((b, nh, t, 32)).unwrap()).unwrap();
        let k = Var::from_tensor(&unit(mk(seed + 1, 32)).reshape((b, nh, t, 32)).unwrap()).unwrap();
        let v = Var::from_tensor(&mk(seed + 2, 32).reshape((b, nh, t, 32)).unwrap()).unwrap();
        // log alpha in about [-0.2, 0], beta in [0, 2] (as D's gates give)
        let la = Var::from_tensor(&mk(seed + 3, 1).abs().unwrap().affine(-0.1, 0.0).unwrap().reshape((b, nh, t)).unwrap()).unwrap();
        let beta = Var::from_tensor(&mk(seed + 4, 1).affine(0.5, 1.0).unwrap().clamp(0.0, 2.0).unwrap().reshape((b, nh, t)).unwrap()).unwrap();
        let wproj = mk(seed + 5, 32).reshape((b, nh, t, 32)).unwrap();
        let y_ref = ava_gpu::model::delta_rule(q.as_tensor(), k.as_tensor(), v.as_tensor(), la.as_tensor(), beta.as_tensor()).unwrap();
        let vg = Tensor::cat(&[v.as_tensor(), &la.as_tensor().unsqueeze(3).unwrap(), &beta.as_tensor().unsqueeze(3).unwrap()], 3).unwrap();
        let y_new = ava_gpu::fused::delta(q.as_tensor(), k.as_tensor(), &vg).unwrap();
        let mag = y_ref.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        eprintln!("delta t={t} y: max diff {:.3e} (scale {mag:.3e})", max_diff(&y_ref, &y_new));
        assert!(max_diff(&y_ref, &y_new) < 1e-4 * mag.max(1.0), "delta fwd t={t}: {} (scale {mag})", max_diff(&y_ref, &y_new));
        let gr = y_ref.mul(&wproj).unwrap().sum_all().unwrap().backward().unwrap();
        let gn = y_new.mul(&wproj).unwrap().sum_all().unwrap().backward().unwrap();
        for (name, x) in [("dq", q.as_tensor()), ("dk", k.as_tensor()), ("dv", v.as_tensor()), ("dlog_alpha", la.as_tensor()), ("dbeta", beta.as_tensor())] {
            let (a, n) = (gr.get(x).unwrap(), gn.get(x).unwrap());
            let mag = a.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
            eprintln!("delta t={t} {name}: max diff {:.3e} (scale {mag:.3e})", max_diff(a, n));
            assert!(max_diff(a, n) < 1e-3 * mag.max(1e-3), "delta {name} t={t}: {} (scale {mag})", max_diff(a, n));
        }
    }
}

#[test]
fn fused_fast_weight_block_matches_the_chain() {
    // Speed round 12 (CUDA only): hebb_fwd_raw / hebb_bwd_raw (D's whole fast-weight block after its projection)
    // against the chain model.rs runs for D (unit q and k, gates, chunked delta rule, head norm, silu output gate),
    // f32 on both sides: O and dP (every column) and the bias gradients. t = 64 and 33: the chain's chunks of 64
    // and 1; the kernel's saved-state chunks of 16 (4 full, and 2 full + 1 of one step).
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("fused fast-weight block: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    let eps = ava_gpu::model::NORM_EPS;
    for (b, nh, t, seed) in [(2usize, 3usize, 64usize, 310u64), (3, 2, 33, 320)] {
        let (d, n) = (nh * 32, b * t);
        let ld = 4 * d + 2 * nh;
        let p = Var::from_tensor(&noise(&dev, n, ld, seed)).unwrap();
        let cst = Var::from_tensor(&noise(&dev, 1, 3 * nh, seed + 1).squeeze(0).unwrap()).unwrap();
        let w = noise(&dev, n, d, seed + 2);
        let (pt, ct) = (p.as_tensor(), cst.as_tensor());
        let heads = |c: usize| pt.narrow(1, c * d, d).unwrap().reshape((b, t, nh, 32)).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
        let unit = |x: Tensor| x.broadcast_mul(&(x.sqr().unwrap().sum_keepdim(D::Minus1).unwrap() + 1e-6).unwrap().powf(-0.5).unwrap()).unwrap();
        let (q, k, v) = (unit(heads(0)), unit(heads(1)), heads(2));
        let per_head = |x: Tensor| x.reshape((b, t, nh)).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
        let za = pt.narrow(1, 4 * d, nh).unwrap().broadcast_add(&ct.narrow(0, 0, nh).unwrap()).unwrap().broadcast_add(&ct.narrow(0, 2 * nh, nh).unwrap()).unwrap();
        let la = per_head(candle_nn::ops::sigmoid(&za).unwrap().log().unwrap());
        let zb = pt.narrow(1, 4 * d + nh, nh).unwrap().broadcast_add(&ct.narrow(0, nh, nh).unwrap()).unwrap();
        let beta = per_head(candle_nn::ops::sigmoid(&zb).unwrap().affine(2.0, 0.0).unwrap());
        let y = ava_gpu::model::delta_rule(&q, &k, &v, &la, &beta).unwrap();
        let y = y.broadcast_mul(&(y.sqr().unwrap().mean_keepdim(D::Minus1).unwrap() + eps).unwrap().powf(-0.5).unwrap()).unwrap();
        let g = pt.narrow(1, 3 * d, d).unwrap();
        let o_ref = y.transpose(1, 2).unwrap().contiguous().unwrap().reshape((n, d)).unwrap().mul(&candle_nn::ops::silu(&g).unwrap()).unwrap();
        let (o_new, ck) = ava_gpu::fused::hebb_fwd_raw(pt, ct, b, t, nh, eps).unwrap();
        let mag = o_ref.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        eprintln!("hebb t={t} O: max diff {:.3e} (scale {mag:.3e})", max_diff(&o_ref, &o_new));
        assert!(max_diff(&o_ref, &o_new) < 1e-4 * mag.max(1.0), "hebb fwd t={t}: {} (scale {mag})", max_diff(&o_ref, &o_new));
        let gr = o_ref.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        let dp_ref = gr.get(pt).unwrap();
        let dp_new = ava_gpu::fused::hebb_bwd_raw(pt, ct, &ck, &w, b, t, nh, eps).unwrap();
        for (name, c0, w_) in [("dq", 0, d), ("dk", d, d), ("dv", 2 * d, d), ("dg", 3 * d, d), ("dza", 4 * d, nh), ("dzb", 4 * d + nh, nh)] {
            let (a, m) = (dp_ref.narrow(1, c0, w_).unwrap(), dp_new.narrow(1, c0, w_).unwrap());
            let mag = a.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
            eprintln!("hebb t={t} {name}: max diff {:.3e} (scale {mag:.3e})", max_diff(&a, &m));
            assert!(max_diff(&a, &m) < 1e-3 * mag.max(1e-3), "hebb {name} t={t}: {} (scale {mag})", max_diff(&a, &m));
        }
        let dc = gr.get(ct).unwrap();
        let dba = dp_new.narrow(1, 4 * d, nh).unwrap().sum(0).unwrap();
        let dbb = dp_new.narrow(1, 4 * d + nh, nh).unwrap().sum(0).unwrap();
        for (name, a, m) in [("dba", dc.narrow(0, 0, nh).unwrap(), dba.clone()), ("dbb", dc.narrow(0, nh, nh).unwrap(), dbb), ("doff", dc.narrow(0, 2 * nh, nh).unwrap(), dba)] {
            let mag = a.abs().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
            assert!(max_diff(&a, &m) < 1e-3 * mag.max(1e-3), "hebb {name} t={t}: {} (scale {mag})", max_diff(&a, &m));
        }
    }
}

#[test]
fn hand_written_da_step_matches_autograd() {
    // Speed round 12 (CUDA only): Model::loss_and_grads_a against loss + candle's backward, every parameter, for D
    // and A (A: 6 layers, attention in blocks 1 and 5). f32 form: the same spikes as autograd, so it must match
    // (loss to 1e-4; every gradient cos > 0.9999 and norm error < 1%; A's attention backward hands back a bf16
    // dqkv, so not to 1e-4). bf16 form: rounding flips spikes, so only the loss is held (to 1e-2) and the gradient
    // agreement is printed; as for S, its real check is a full training run (prereg_speed_r2.md, round 12).
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("D/A manual step: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    let mut bad: Vec<String> = Vec::new();
    for kind in [Kind::D, Kind::A] {
        let cfg = Config::with_kind(kind, VOCAB, 64, if kind == Kind::A { 6 } else { 2 }, 64);
        let varmap = VarMap::new();
        let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &dev)).unwrap();
        let (ids, tg) = batch(&dev, 4, cfg.ctx, 17);
        let (l_ref, _) = model.loss(&ids, &tg).unwrap();
        let g_ref = l_ref.backward().unwrap();
        let x = l_ref.to_scalar::<f32>().unwrap();
        for bf16 in [false, true] {
            let (l_new, act, g_new) = model.loss_and_grads_a(&ids, &tg, &ava_gpu::model::Mm { bf16 }).unwrap();
            let y = l_new.to_scalar::<f32>().unwrap();
            let tol = if bf16 { 1e-2 } else { 1e-4 };
            assert!((x - y).abs() < tol, "{kind:?} bf16={bf16} loss {x} vs manual {y}");
            assert_eq!(act.spike_rate.len(), cfg.layers);
            let mut coss: Vec<f32> = Vec::new();
            for p in params_of(&varmap) {
                let r = g_ref.get(p.var.as_tensor()).expect("autograd grad");
                let m = g_new.get(p.var.as_tensor()).unwrap_or_else(|| panic!("{kind:?} no manual grad for {}", p.name));
                assert_eq!(r.dims(), m.dims(), "{kind:?} {} shape", p.name);
                let (rf, mf) = (r.flatten_all().unwrap(), m.flatten_all().unwrap());
                let dot = rf.mul(&mf).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
                let nr = rf.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
                let nm = mf.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
                let cos = dot / (nr * nm).max(1e-30);
                let relnorm = rf.sub(&mf).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt() / nr.max(1e-30);
                coss.push(cos);
                if !bf16 {
                    eprintln!("manual f32 {kind:?} {:<20} cos {cos:.6} rel-norm-diff {relnorm:.2e}", p.name);
                    if !(cos > 0.9999 && relnorm < 1e-2) {
                        bad.push(format!("{kind:?} f32 {}: cos {cos} rel-norm-diff {relnorm}", p.name));
                    }
                }
            }
            let mean = coss.iter().sum::<f32>() / coss.len() as f32;
            eprintln!("manual {kind:?} bf16={bf16}: loss {x:.6} vs {y:.6}, {} parameters, mean cos {mean:.5}", coss.len());
        }
    }
    assert!(bad.is_empty(), "D/A manual gradients off: {bad:#?}");
}

#[test]
fn masked_answer_loss_matches_autograd() {
    // v37 (CUDA only): loss_and_grads_a_w with a 0/1 weight per target (only answers scored) against candle's
    // autograd of sum(w · nll) / sum(w), f32, every parameter (cos > 0.9999, norm error < 1%). All-ones weights
    // must give the plain mean.
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("masked answer loss: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    let cfg = Config::with_kind(Kind::D, VOCAB, 64, 2, 64);
    let varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &dev)).unwrap();
    let (ids, tg) = batch(&dev, 4, cfg.ctx, 23);
    let n = 4 * cfg.ctx;
    let mask: Vec<f32> = (0..n).map(|i| if (i % cfg.ctx) >= cfg.ctx / 2 || i % 7 == 0 { 1.0 } else { 0.0 }).collect();
    let w = Tensor::from_vec(mask, n, &dev).unwrap();
    let mm = ava_gpu::model::Mm { bf16: false };
    // all ones = the plain mean
    let ones = Tensor::ones(n, DType::F32, &dev).unwrap();
    let (l_plain, _, _) = model.loss_and_grads_a(&ids, &tg, &mm).unwrap();
    let (l_ones, _, _) = model.loss_and_grads_a_w(&ids, &tg, Some(&ones), &mm).unwrap();
    let (a, b) = (l_plain.to_scalar::<f32>().unwrap(), l_ones.to_scalar::<f32>().unwrap());
    assert!((a - b).abs() < 1e-5, "all-ones weights {b} vs plain mean {a}");
    // masked: autograd reference
    let logits = model.forward(&ids).unwrap();
    let lp = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1).unwrap();
    let nll = lp.gather(&tg.flatten_all().unwrap().unsqueeze(1).unwrap(), 1).unwrap().squeeze(1).unwrap().neg().unwrap();
    let l_ref = nll.mul(&w).unwrap().sum_all().unwrap().broadcast_div(&w.sum_all().unwrap()).unwrap();
    let g_ref = l_ref.backward().unwrap();
    let (l_new, _, g_new) = model.loss_and_grads_a_w(&ids, &tg, Some(&w), &mm).unwrap();
    let (x, y) = (l_ref.to_scalar::<f32>().unwrap(), l_new.to_scalar::<f32>().unwrap());
    assert!((x - y).abs() < 1e-4, "masked loss {x} vs manual {y}");
    let mut bad = Vec::new();
    for p in params_of(&varmap) {
        let r = g_ref.get(p.var.as_tensor()).expect("autograd grad").flatten_all().unwrap();
        let m = g_new.get(p.var.as_tensor()).expect("manual grad").flatten_all().unwrap();
        let dot = r.mul(&m).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let nr = r.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        let nm = m.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        let cos = dot / (nr * nm).max(1e-30);
        let rel = r.sub(&m).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt() / nr.max(1e-30);
        if !(cos > 0.9999 && rel < 1e-2) {
            bad.push(format!("{}: cos {cos} rel {rel}", p.name));
        }
    }
    eprintln!("masked answer loss: {x:.6} vs {y:.6}; all-ones {b:.6} vs plain {a:.6}");
    assert!(bad.is_empty(), "masked gradients off: {bad:#?}");
}

#[test]
fn tensor_core_attention_tracks_the_f32_kernels() {
    // Speed round 9 (CUDA only): the tensor-core attention (bf16 q, k, v, probabilities and dout; f32 accumulate)
    // against the f32 kernels on the same bf16-rounded inputs. Scores are exact products summed in f32 on both, so
    // lse agrees to 1e-4; outputs within 1e-2; every gradient block cos > 0.999 and norm error < 2%. Odd lengths
    // and partial tiles included.
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("tensor-core attention: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    use ava_gpu::fused::{attn_bwd_s, attn_bwd_tc, attn_fwd_s, attn_fwd_tc, to_bf16};
    let dev = device();
    let cos_rel = |r: &Tensor, m: &Tensor| -> (f32, f32) {
        let (rf, mf) = (r.flatten_all().unwrap(), m.flatten_all().unwrap());
        let dot = rf.mul(&mf).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let nr = rf.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        let nm = mf.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        let rel = rf.sub(&mf).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt() / nr.max(1e-30);
        (dot / (nr * nm).max(1e-30), rel)
    };
    for (b, nh, t, seed) in [(2usize, 3usize, 100usize, 50u64), (1, 2, 256, 60), (3, 1, 33, 70), (2, 2, 64, 80), (1, 4, 2, 90)] {
        // (t = 1 is left out: its true dq is exactly 0, so both sides are rounding noise and the cosine means nothing)
        let d = nh * 32;
        let scale = 1.0 / 32f64.sqrt();
        let qkv_b = to_bf16(&noise(&dev, b * t, 3 * d, seed)).unwrap();
        let qkv_r = qkv_b.to_dtype(DType::F32).unwrap();
        let dout_b = to_bf16(&noise(&dev, b * t, d, seed + 1)).unwrap();
        let dout_r = dout_b.to_dtype(DType::F32).unwrap();
        let (o_ref, _, lse_ref) = attn_fwd_s(&qkv_r, b, t, scale).unwrap();
        let (o_tc, lse_tc) = attn_fwd_tc(&qkv_b, b, t, scale).unwrap();
        let o_tc32 = o_tc.to_dtype(DType::F32).unwrap();
        assert!(max_diff(&lse_ref, &lse_tc) < 1e-4, "lse t={t}: {}", max_diff(&lse_ref, &lse_tc));
        assert!(max_diff(&o_ref, &o_tc32) < 1e-2, "out t={t}: {}", max_diff(&o_ref, &o_tc32));
        let g_ref = attn_bwd_s(&qkv_r, &o_ref, &lse_ref, &dout_r, b, t, scale).unwrap().to_dtype(DType::F32).unwrap();
        let g_tc = attn_bwd_tc(&qkv_b, &o_tc, &lse_tc, &dout_b, b, t, scale).unwrap().to_dtype(DType::F32).unwrap();
        for (i, name) in ["dq", "dk", "dv"].iter().enumerate() {
            let (r, m) = (g_ref.narrow(1, i * d, d).unwrap(), g_tc.narrow(1, i * d, d).unwrap());
            let (cos, rel) = cos_rel(&r, &m);
            eprintln!("tc attention t={t} {name}: cos {cos:.6} rel-norm-diff {rel:.5}");
            assert!(cos > 0.999 && rel < 0.02, "tc attention t={t} {name}: cos {cos} rel {rel}");
        }
    }
}

#[test]
fn fused_embedding_and_gelu_match_candle() {
    // Speed round 5 (CUDA only): the embedding scatter is bit-identical to index_add (same summation order),
    // GELU forward and backward match candle's gelu and its autograd.
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("fused embedding/gelu: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    let (v, d, n) = (500usize, 96usize, 3000usize);
    // ids with heavy repeats (a few very common rows, like real text) and rows never hit
    let ids: Vec<u32> = (0..n as u64).map(|i| ((i * 2654435761 % 977) % 300) as u32 % if i % 3 == 0 { 7 } else { 300 }).collect();
    let ids = Tensor::from_vec(ids, n, &dev).unwrap();
    let base = noise(&dev, v, d, 50);
    let dx = noise(&dev, n, d, 51);
    let a = base.index_add(&ids, &dx, 0).unwrap();
    let b = ava_gpu::fused::embed_bwd_add(&base, &ids, &dx).unwrap();
    assert_eq!(max_diff(&a, &b), 0.0, "embed_bwd_add differs from index_add");
    let tok = Var::from_tensor(&noise(&dev, v, d, 52)).unwrap();
    let e_ref = tok.as_tensor().embedding(&ids).unwrap();
    let e_new = ava_gpu::fused::embedding(tok.as_tensor(), &ids).unwrap();
    assert_eq!(max_diff(&e_ref, &e_new), 0.0, "embedding forward differs");
    let gr = e_ref.mul(&dx).unwrap().sum_all().unwrap().backward().unwrap();
    let gn = e_new.mul(&dx).unwrap().sum_all().unwrap().backward().unwrap();
    assert_eq!(max_diff(gr.get(tok.as_tensor()).unwrap(), gn.get(tok.as_tensor()).unwrap()), 0.0, "embedding backward differs");
    let u = Var::from_tensor(&noise(&dev, 777, 130, 53)).unwrap();
    let w = noise(&dev, 777, 130, 54);
    let y_ref = u.as_tensor().gelu().unwrap();
    let y_new = ava_gpu::fused::gelu(u.as_tensor()).unwrap();
    assert!(max_diff(&y_ref, &y_new) < 1e-6, "gelu fwd {}", max_diff(&y_ref, &y_new));
    let gr = y_ref.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
    let gn = y_new.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
    let (a, b) = (gr.get(u.as_tensor()).unwrap(), gn.get(u.as_tensor()).unwrap());
    assert!(max_diff(a, b) < 1e-5, "gelu bwd {}", max_diff(a, b));
}

#[test]
fn hand_written_t_backward_matches_autograd() {
    // Speed round 6 (CUDA only): the T yardstick's Model::loss_and_grads against loss + candle's backward.
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("manual T backward: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    let cfg = Config::with_kind(Kind::T, VOCAB, 64, 2, 48);
    let varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &dev)).unwrap();
    for (bsz, t, seed) in [(4usize, 48usize, 12u64), (3, 37, 13)] {
        let (ids, tg) = batch(&dev, bsz, cfg.ctx, seed);
        let (ids, tg) = (ids.narrow(1, 0, t).unwrap(), tg.narrow(1, 0, t).unwrap());
        let (l_ref, _) = model.loss(&ids, &tg).unwrap();
        let g_ref = l_ref.backward().unwrap();
        let (l_new, _, g_new) = model.loss_and_grads_with(&ids, &tg, false).unwrap().expect("T on CUDA takes the manual path");
        let (a, b) = (l_ref.to_scalar::<f32>().unwrap(), l_new.to_scalar::<f32>().unwrap());
        assert!((a - b).abs() < 1e-5, "loss {a} vs {b}");
        let mut n = 0;
        for p in params_of(&varmap) {
            let (r, m) = (g_ref.get(p.var.as_tensor()).expect("autograd grad"), g_new.get(p.var.as_tensor()).expect("manual grad"));
            let scale = r.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
            let diff = max_diff(r, m);
            assert!(diff <= 1e-4 * scale.max(1e-6), "t={t} {}: max |diff| {diff} (scale {scale})", p.name);
            n += 1;
        }
        assert_eq!(n, 19, "every T parameter checked");
    }
}

#[test]
fn bf16_steps_track_the_f32_gradients() {
    // Speed round 7 (CUDA only): bf16 matmul operands in the hand-written S and T steps. Not exact. T (no spikes): every
    // gradient cos > 0.999 and norm error < 5%. S: bf16 rounding flips some spikes, and the LIF resets carry each flip
    // down the sequence, so an untrained S net's gradient moves a lot (cos 0.67 to 0.95 at init; TF32 alone gives about
    // 0.94). For S the test asks only that the loss agrees and every gradient points the same way (cos > 0.5, mean
    // > 0.7); the real S check is a full training run against the f32 result. Plus mm_bf16 itself against f32.
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("bf16 steps: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    let (a, b) = (noise(&dev, 300, 70, 60), noise(&dev, 70, 90, 61));
    let want = a.matmul(&b).unwrap();
    let scale = want.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    let ab = ava_gpu::fused::to_bf16(&a).unwrap();
    assert_eq!(max_diff(&ab.to_dtype(DType::F32).unwrap(), &a.to_dtype(DType::BF16).unwrap().to_dtype(DType::F32).unwrap()), 0.0, "to_bf16 differs from candle's cast");
    let bb = ava_gpu::fused::to_bf16(&b).unwrap();
    let at = ava_gpu::fused::to_bf16(&a.t().unwrap().contiguous().unwrap()).unwrap();
    let bt = ava_gpu::fused::to_bf16(&b.t().unwrap().contiguous().unwrap()).unwrap();
    for (name, got) in [
        ("NN", ava_gpu::fused::mm_bf16(&ab, false, &bb, false).unwrap()),
        ("TN", ava_gpu::fused::mm_bf16(&at, true, &bb, false).unwrap()),
        ("NT", ava_gpu::fused::mm_bf16(&ab, false, &bt, true).unwrap()),
        ("TT", ava_gpu::fused::mm_bf16(&at, true, &bt, true).unwrap()),
    ] {
        assert!(max_diff(&got, &want) < 1e-2 * scale, "mm_bf16 {name}: {} (scale {scale})", max_diff(&got, &want));
    }
    let mut bad: Vec<String> = Vec::new();
    for kind in [Kind::S, Kind::T] {
        let mut coss: Vec<f32> = Vec::new();
        let cfg = Config::with_kind(kind, VOCAB, 128, 2, 64);
        let varmap = VarMap::new();
        let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &dev)).unwrap();
        let (ids, tg) = batch(&dev, 4, cfg.ctx, 14);
        let (l32, _, g32) = model.loss_and_grads_with(&ids, &tg, false).unwrap().expect("manual path");
        let (l16, _, g16) = model.loss_and_grads_with(&ids, &tg, true).unwrap().expect("manual path");
        let (x, y) = (l32.to_scalar::<f32>().unwrap(), l16.to_scalar::<f32>().unwrap());
        assert!((x - y).abs() < 1e-2, "{kind:?} loss {x} vs bf16 {y}");
        for p in params_of(&varmap) {
            let (r, m) = (g32.get(p.var.as_tensor()).unwrap(), g16.get(p.var.as_tensor()).unwrap());
            let s = r.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
            let diff = max_diff(r, m);
            let (rf, mf) = (r.flatten_all().unwrap(), m.flatten_all().unwrap());
            let dot = rf.mul(&mf).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            let nr = rf.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            let nm = mf.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            let cos = dot / (nr * nm).max(1e-30);
            let relnorm = rf.sub(&mf).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt() / nr.max(1e-30);
            eprintln!("bf16 {kind:?} {:<16} cos {cos:.5} rel-norm-diff {relnorm:.4} max-diff/scale {:.4}", p.name, diff / s.max(1e-30));
            coss.push(cos);
            let ok = if kind == Kind::T { cos > 0.999 && relnorm < 0.05 } else { cos > 0.5 };
            if !ok {
                bad.push(format!("{kind:?} {}: cos {cos} rel-norm-diff {relnorm} (max diff {diff}, scale {s})", p.name));
            }
        }
        let mean = coss.iter().sum::<f32>() / coss.len() as f32;
        if kind != Kind::T && mean <= 0.7 {
            bad.push(format!("{kind:?}: mean cos {mean}"));
        }
    }
    assert!(bad.is_empty(), "bf16 gradients off: {bad:#?}");
}

// ---------------------------------------------------------------- v39: the glance

/// Rows of `t + 1` tokens with numbers in them: words (ids 300..556), length-marked digit runs of 1..6 digits,
/// bare digits, stray marks, and now and then a run of 18 digits or a mark of the wrong length.
fn glance_stream(len: usize, seed: u64) -> Vec<u32> {
    use ava_data::numbers::insert_marks;
    let mut rng = ava_data::rng::SplitMix64::new(seed);
    let mut out = Vec::with_capacity(len + 32);
    while out.len() < len {
        match rng.below(10) {
            0..=4 => out.push(300 + rng.below(256) as u32),
            5..=7 => {
                let l = 1 + rng.below(6) as usize;
                let digits: Vec<u32> = (0..l).map(|_| 48 + rng.below(10) as u32).collect();
                out.extend(insert_marks(&digits));
            }
            8 => out.push(48 + rng.below(10) as u32),
            _ => {
                if rng.below(2) == 0 {
                    out.extend(insert_marks(&(0..18).map(|_| 48 + rng.below(10) as u32).collect::<Vec<_>>()));
                } else {
                    out.push(4096 + rng.below(16) as u32);
                    out.extend((0..1 + rng.below(4)).map(|_| 48 + rng.below(10) as u32));
                }
            }
        }
    }
    out.truncate(len);
    out
}

fn glance_model(dev: &Device, seed: u64) -> (Config, VarMap, Model) {
    let cfg = Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 64, 2, 64).with_glance().unwrap();
    let varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, dev)).unwrap();
    // the place embedding starts at 0; give it values so the forward and the stream really use it
    let data = varmap.data().lock().unwrap();
    let pv = data.get(ava_gpu::model::PLACE_EMB).expect("place_emb");
    pv.set(&noise(dev, 16, 64, seed).affine(0.05, 0.0).unwrap()).unwrap();
    drop(data);
    (cfg, varmap, model)
}

#[test]
fn place_ids_kernel_matches_cpu() {
    // v39 (CUDA only): fused::place_ids against ava_data::numbers::place_ids, row by row, exactly.
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("place ids kernel: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    for (b, t, seed) in [(8usize, 256usize, 1u64), (3, 17, 2), (1, 1, 3), (5, 64, 4)] {
        let toks = glance_stream(b * t, seed);
        let gpu: Vec<u32> = ava_gpu::fused::place_ids(&Tensor::from_vec(toks.clone(), (b, t), &dev).unwrap()).unwrap().to_vec1().unwrap();
        let cpu: Vec<u32> = toks.chunks(t).flat_map(ava_data::numbers::place_ids).collect();
        assert_eq!(gpu, cpu, "place ids differ for b {b} t {t}");
        assert!(cpu.iter().any(|&p| p < 16) && cpu.iter().any(|&p| p == 16) || t < 4, "test rows have no places");
    }
}

#[test]
fn glance_grads_match_autograd() {
    // v39 (CUDA only): D with the glance, the hand-written step (f32) against candle's autograd of the same loss,
    // every parameter including place_emb (cos > 0.9999, norm error < 1%).
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("glance grads: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    let (cfg, varmap, model) = glance_model(&dev, 41);
    let (b, t) = (4usize, cfg.ctx);
    let mut ids = Vec::with_capacity(b * t);
    let mut tg = Vec::with_capacity(b * t);
    for r in 0..b {
        let s = glance_stream(t + 1, 100 + r as u64);
        ids.extend_from_slice(&s[..t]);
        tg.extend_from_slice(&s[1..]);
    }
    let ids = Tensor::from_vec(ids, (b, t), &dev).unwrap();
    let tg = Tensor::from_vec(tg, (b, t), &dev).unwrap();
    let (l_ref, _) = model.loss(&ids, &tg).unwrap();
    let g_ref = l_ref.backward().unwrap();
    let (l_new, _, g_new) = model.loss_and_grads_a(&ids, &tg, &ava_gpu::model::Mm { bf16: false }).unwrap();
    let (x, y) = (l_ref.to_scalar::<f32>().unwrap(), l_new.to_scalar::<f32>().unwrap());
    assert!((x - y).abs() < 1e-4, "glance loss {x} vs manual {y}");
    let mut bad = Vec::new();
    let mut saw_place = false;
    for p in params_of(&varmap) {
        let r = g_ref.get(p.var.as_tensor()).expect("autograd grad").flatten_all().unwrap();
        let m = g_new.get(p.var.as_tensor()).expect("manual grad").flatten_all().unwrap();
        let dot = r.mul(&m).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let nr = r.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        let nm = m.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        let cos = dot / (nr * nm).max(1e-30);
        let rel = r.sub(&m).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt() / nr.max(1e-30);
        if p.name == ava_gpu::model::PLACE_EMB {
            saw_place = nr > 0.0;
        }
        if !(cos > 0.9999 && rel < 1e-2) {
            bad.push(format!("{}: cos {cos} rel {rel}", p.name));
        }
    }
    eprintln!("glance grads: loss {x:.6} vs {y:.6}");
    assert!(saw_place, "place_emb got no gradient");
    assert!(bad.is_empty(), "glance gradients off: {bad:#?}");
}

#[test]
fn stream_glance_matches_forward() {
    // v39: the carried state (Model::step, which tracks the marks itself) gives the logits of one full forward
    // over a row full of numbers.
    let dev = device();
    let (cfg, _varmap, model) = glance_model(&dev, 43);
    let toks = glance_stream(cfg.ctx, 7);
    let full = model.forward(&Tensor::from_vec(toks.clone(), (1, cfg.ctx), &dev).unwrap()).unwrap();
    let mut st = model.stream().unwrap();
    let mut worst = 0f32;
    for (p, &t) in toks.iter().enumerate() {
        let a = model.step(&mut st, t).unwrap();
        let b = full.narrow(0, p, 1).unwrap().squeeze(0).unwrap();
        worst = worst.max(a.sub(&b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap());
    }
    assert!(worst < 1e-3, "glance carried-state logits differ from the forward by {worst}");
}

#[test]
fn stream_glance_overrun() {
    // v39: a mark of 2 followed by 3 digits, then a mark and a word, then a mark of 3 and one digit: the extra digit
    // has no place, the digit after the mark of 3 is in the hundreds place (2), nothing panics,
    // and the stream agrees with numbers::place_ids on the same row.
    let dev = device();
    let (_cfg, _varmap, model) = glance_model(&dev, 47);
    let row = [4097u32, 49, 50, 51, 4096, 300, 4098, 52];
    let mut st = model.stream().unwrap();
    let got: Vec<u32> = row.iter().map(|&t| st.next_place(t)).collect();
    assert_eq!(got, vec![16, 1, 0, 16, 16, 16, 16, 2]);
    assert_eq!(got, ava_data::numbers::place_ids(&row));
    let mut st = model.stream().unwrap();
    for &t in &row {
        model.step(&mut st, t).unwrap();
    }
}

/// v39: with one group the stratified bootstrap is the plain paired bootstrap (same draws, same order), and with
/// equal-size groups its mean is the item mean.
#[test]
fn stratified_bootstrap_matches_plain() {
    let a: Vec<bool> = (0..97).map(|k| k % 3 != 0).collect();
    let b: Vec<bool> = (0..97).map(|k| k % 5 == 0).collect();
    let plain = ava_gpu::eval::paired_bootstrap(&a, &b, 2_000, 7);
    let strat = ava_gpu::eval::stratified_paired_bootstrap(&[(a.clone(), b.clone())], 2_000, 7);
    assert_eq!(plain, strat);
    let g = vec![(a[..40].to_vec(), b[..40].to_vec()), (a[40..80].to_vec(), b[40..80].to_vec())];
    let (m, lo, hi) = ava_gpu::eval::stratified_paired_bootstrap(&g, 2_000, 7);
    let item = ava_gpu::eval::paired_bootstrap(&a[..80], &b[..80], 10, 7).0;
    assert!((m - item).abs() < 1e-12 && lo <= m && m <= hi, "{m} {item} {lo} {hi}");
}

// ---------------------------------------------------------------- v40: magnitude code, engram, lesions, reward weights

/// D with the glance, the magnitude code and the engram in block 1 of 2; place_emb, mag_w and the engram's Wo get
/// values (they start at 0) so every part really acts.
fn v40_model(dev: &Device, seed: u64) -> (Config, VarMap, Model) {
    let cfg = Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 64, 2, 64)
        .with_glance()
        .and_then(|c| c.with_mag())
        .and_then(|c| c.with_engram(1))
        .unwrap();
    let varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, dev)).unwrap();
    let data = varmap.data().lock().unwrap();
    data.get(ava_gpu::model::PLACE_EMB).unwrap().set(&noise(dev, 16, 64, seed).affine(0.05, 0.0).unwrap()).unwrap();
    data.get(ava_gpu::model::MAG_W).unwrap().set(&noise(dev, ava_gpu::mag::MAG_F, 64, seed + 1).affine(0.05, 0.0).unwrap()).unwrap();
    data.get("b1.eng.wo").expect("engram wo").set(&noise(dev, 64, 64, seed + 2).affine(0.05, 0.0).unwrap()).unwrap();
    drop(data);
    (cfg, varmap, model)
}

fn glance_batch(dev: &Device, b: usize, t: usize, seed: u64) -> (Tensor, Tensor) {
    let mut ids = Vec::with_capacity(b * t);
    let mut tg = Vec::with_capacity(b * t);
    for r in 0..b {
        let s = glance_stream(t + 1, seed + r as u64);
        ids.extend_from_slice(&s[..t]);
        tg.extend_from_slice(&s[1..]);
    }
    (Tensor::from_vec(ids, (b, t), dev).unwrap(), Tensor::from_vec(tg, (b, t), dev).unwrap())
}

/// Compare two gradient stores on every parameter: a parameter missing from the reference (unused under a
/// lesion) must be missing from the manual step too; otherwise cos > 0.9999 and norm error < 1%.
fn grads_agree(varmap: &VarMap, g_ref: &candle_core::backprop::GradStore, g_new: &candle_core::backprop::GradStore) -> (Vec<String>, Vec<String>) {
    let (mut bad, mut absent) = (Vec::new(), Vec::new());
    for p in params_of(varmap) {
        let Some(r) = g_ref.get(p.var.as_tensor()) else {
            if g_new.get(p.var.as_tensor()).is_some() {
                bad.push(format!("{}: manual grad for an unused parameter", p.name));
            }
            absent.push(p.name.clone());
            continue;
        };
        let r = r.flatten_all().unwrap();
        let m = g_new.get(p.var.as_tensor()).unwrap_or_else(|| panic!("manual grad missing: {}", p.name)).flatten_all().unwrap();
        let dot = r.mul(&m).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let nr = r.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        let nm = m.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        let cos = dot / (nr * nm).max(1e-30);
        let rel = r.sub(&m).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt() / nr.max(1e-30);
        if !(cos > 0.9999 && rel < 1e-2) {
            bad.push(format!("{}: cos {cos} rel {rel}", p.name));
        }
    }
    (bad, absent)
}

#[test]
fn mag_feats_kernel_matches_cpu() {
    // v40 (CUDA only): fused::mag_feats against mag::feats_row, row by row (both work in f64, then round to f32).
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("mag feats kernel: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    for (b, t, seed) in [(8usize, 256usize, 1u64), (3, 17, 2), (1, 1, 3), (5, 64, 4)] {
        let toks = glance_stream(b * t, seed);
        let gpu: Vec<f32> = ava_gpu::fused::mag_feats(&Tensor::from_vec(toks.clone(), (b, t), &dev).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let cpu: Vec<f32> = toks.chunks(t).flat_map(ava_gpu::mag::feats_row).collect();
        assert_eq!(gpu.len(), cpu.len());
        let worst = gpu.iter().zip(&cpu).map(|(a, c)| (a - c).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-6, "mag feats differ by {worst} for b {b} t {t}");
        assert!(cpu.iter().any(|&v| v != 0.0) || t < 4, "test rows have no codes");
    }
}

#[test]
fn v40_grads_match_autograd() {
    // v40 (CUDA only): D with glance + magnitude code + engram, the hand-written step (f32) against candle's
    // autograd of the same loss, every parameter; then the same with all three parts lesioned (their parameters
    // must get no gradient and the rest must still agree).
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("v40 grads: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    let (cfg, varmap, model) = v40_model(&dev, 51);
    let (ids, tg) = glance_batch(&dev, 4, cfg.ctx, 200);
    let mm = ava_gpu::model::Mm { bf16: false };
    for lesion in [false, true] {
        model.set_lesion(lesion, lesion, lesion);
        let (l_ref, _) = model.loss(&ids, &tg).unwrap();
        let g_ref = l_ref.backward().unwrap();
        let (l_new, _, g_new) = model.loss_and_grads_a(&ids, &tg, &mm).unwrap();
        let (x, y) = (l_ref.to_scalar::<f32>().unwrap(), l_new.to_scalar::<f32>().unwrap());
        assert!((x - y).abs() < 1e-4, "v40 loss {x} vs manual {y} (lesion {lesion})");
        let (bad, absent) = grads_agree(&varmap, &g_ref, &g_new);
        eprintln!("v40 grads (lesion {lesion}): loss {x:.6} vs {y:.6}; unused {absent:?}");
        assert!(bad.is_empty(), "v40 gradients off (lesion {lesion}): {bad:#?}");
        if lesion {
            for name in [ava_gpu::model::PLACE_EMB, ava_gpu::model::MAG_W, "b1.eng.wo", "b1.eng.wq"] {
                assert!(absent.iter().any(|a| a == name), "{name} should be unused under the lesion");
            }
        } else {
            assert!(absent.is_empty(), "every parameter should get a gradient: {absent:?}");
        }
    }
    model.set_lesion(false, false, false);
}

#[test]
fn raw_signed_weights_match_autograd() {
    // v40 (CUDA only): loss_and_grads_a_wr with raw signed weights (+, - and 0 per target) against candle's
    // autograd of sum(w * nll), every parameter.
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("raw weights: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    let (cfg, varmap, model) = v40_model(&dev, 53);
    let (ids, tg) = glance_batch(&dev, 4, cfg.ctx, 300);
    let n = 4 * cfg.ctx;
    let w: Vec<f32> = (0..n).map(|i| match i % 5 { 0 => 0.02, 1 => -0.015, 2 => 0.0, 3 => 0.01, _ => -0.03 }).collect();
    let w = Tensor::from_vec(w, n, &dev).unwrap();
    let logits = model.forward(&ids).unwrap();
    let lp = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1).unwrap();
    let nll = lp.gather(&tg.flatten_all().unwrap().unsqueeze(1).unwrap(), 1).unwrap().squeeze(1).unwrap().neg().unwrap();
    let l_ref = nll.mul(&w).unwrap().sum_all().unwrap();
    let g_ref = l_ref.backward().unwrap();
    let (l_new, _, g_new) = model.loss_and_grads_a_wr(&ids, &tg, Some(&w), true, &ava_gpu::model::Mm { bf16: false }).unwrap();
    let (x, y) = (l_ref.to_scalar::<f32>().unwrap(), l_new.to_scalar::<f32>().unwrap());
    assert!((x - y).abs() < 1e-4, "raw-weight loss {x} vs manual {y}");
    let (bad, _) = grads_agree(&varmap, &g_ref, &g_new);
    eprintln!("raw weights: {x:.6} vs {y:.6}");
    assert!(bad.is_empty(), "raw-weight gradients off: {bad:#?}");
}

#[test]
fn stream_v40_matches_forward() {
    // v40: the carried state (step: magnitude code from the token history, engram keys and values kept) gives the
    // logits of one full forward over a row full of numbers, with and without the lesions.
    let dev = device();
    let (cfg, _varmap, model) = v40_model(&dev, 57);
    let toks = glance_stream(cfg.ctx, 9);
    for lesion in [false, true] {
        model.set_lesion(lesion, lesion, lesion);
        let full = model.forward(&Tensor::from_vec(toks.clone(), (1, cfg.ctx), &dev).unwrap()).unwrap();
        let mut st = model.stream().unwrap();
        let mut worst = 0f32;
        for (p, &t) in toks.iter().enumerate() {
            let a = model.step(&mut st, t).unwrap();
            let b = full.narrow(0, p, 1).unwrap().squeeze(0).unwrap();
            worst = worst.max(a.sub(&b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap());
        }
        assert!(worst < 1e-3, "v40 carried-state logits differ from the forward by {worst} (lesion {lesion})");
    }
    model.set_lesion(false, false, false);
}

#[test]
fn each_lesion_changes_the_forward() {
    // v40: switching off the place code, the magnitude code or the engram each changes the output.
    let dev = device();
    let (cfg, _varmap, model) = v40_model(&dev, 59);
    let toks = glance_stream(cfg.ctx, 11);
    let ids = Tensor::from_vec(toks, (1, cfg.ctx), &dev).unwrap();
    let base = model.forward(&ids).unwrap();
    for (i, (p, m, e)) in [(true, false, false), (false, true, false), (false, false, true)].into_iter().enumerate() {
        model.set_lesion(p, m, e);
        let out = model.forward(&ids).unwrap();
        let diff = out.sub(&base).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff > 1e-4, "lesion {i} changed nothing");
    }
    model.set_lesion(false, false, false);
}

#[test]
fn grown_model_starts_as_the_old_one_and_files_say_their_shape() {
    // v40: a v39-shaped model (glance) saved to a file; a v40 model (glance + magnitude code + engram) started from
    // it with ckpt::init_from gives the same logits exactly (the new read-outs start at 0). Then the v40 model saved
    // and loaded with ckpt::load_auto comes back with the same config and the same logits; load_expect refuses a
    // file whose shape differs from the flags.
    let dev = device();
    let dir = std::env::temp_dir().join(format!("ava_v40_ckpt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (cfg39, vm39, m39) = glance_model(&dev, 61);
    let p39 = dir.join("v39.safetensors");
    vm39.save(&p39).unwrap();
    let cfg40 = cfg39.with_mag().and_then(|c| c.with_engram(1)).unwrap();
    let vm40 = VarMap::new();
    let m40 = Model::new(cfg40, VarBuilder::from_varmap(&vm40, DType::F32, &dev)).unwrap();
    let (copied, kept) = ava_gpu::ckpt::init_from(&vm40, &p39).unwrap();
    assert_eq!(copied.len(), vm39.data().lock().unwrap().len());
    assert!(kept.iter().any(|k| k == "mag_w") && kept.iter().any(|k| k.starts_with("b1.eng.")), "new parts: {kept:?}");
    let ids = Tensor::from_vec(glance_stream(cfg39.ctx, 13), (1, cfg39.ctx), &dev).unwrap();
    let a = m39.forward(&ids).unwrap();
    let b = m40.forward(&ids).unwrap();
    let diff = a.sub(&b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    assert_eq!(diff, 0.0, "the grown model at step 0 differs from the old one by {diff}");
    // shape from the file
    let (_, vm, _) = v40_model(&dev, 63);
    let p40 = dir.join("v40.safetensors");
    vm.save(&p40).unwrap();
    let (cfg_l, _vl, ml) = ava_gpu::ckpt::load_auto(&p40, Kind::D, cfg40.ctx, &dev).unwrap();
    assert!(cfg_l.glance && cfg_l.mag && cfg_l.engram == Some(1) && cfg_l.d == 64 && cfg_l.layers == 2, "{cfg_l:?}");
    let (_, _, m_ref) = {
        let vm2 = VarMap::new();
        let m = Model::new(cfg40, VarBuilder::from_varmap(&vm2, DType::F32, &dev)).unwrap();
        let mut vm2 = vm2;
        vm2.load(&p40).unwrap();
        (cfg40, vm2, m)
    };
    let d2 = ml.forward(&ids).unwrap().sub(&m_ref.forward(&ids).unwrap()).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    assert_eq!(d2, 0.0);
    assert!(ava_gpu::ckpt::load_expect(&p40, Kind::D, 64, 64, 2, false, &dev).is_err(), "glance mismatch must fail");
    assert!(ava_gpu::ckpt::load_expect(&p39, Kind::D, 64, 64, 2, true, &dev).is_ok());
    // a v40 file cannot be squeezed into a smaller model
    assert!(ava_gpu::ckpt::init_from(&vm39, &p40).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every parameter's values (or gradient), by name, on the CPU.
fn snapshot(varmap: &VarMap, grads: Option<&candle_core::backprop::GradStore>) -> Vec<(String, Vec<f32>)> {
    params_of(varmap)
        .into_iter()
        .map(|p| {
            let t = match grads {
                Some(g) => g.get(p.var.as_tensor()).expect("grad").clone(),
                None => p.var.as_tensor().clone(),
            };
            (p.name, t.flatten_all().unwrap().to_vec1::<f32>().unwrap())
        })
        .collect()
}

#[test]
fn dist_two_ranks_match_one_process() {
    // v40 (CUDA only): two ranks (threads, each with its own model and CUDA handle, linked by ava_gpu::dist over
    // localhost TCP; rank 1 starts from other weights and gets rank 0's by broadcast) each read half of a batch
    // with weights 1 / (whole batch). After the gradient all-reduce and one AdamW step the two ranks hold the same
    // bits, and they match one process that read the whole batch (gradients to rounding).
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("dist: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    let dir = std::env::temp_dir().join(format!("ava_v40_dist_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (cfg, vm, _m) = v40_model(&dev, 71);
    let p0 = dir.join("rank0.safetensors");
    vm.save(&p0).unwrap();
    let (_, vm_other, _m2) = v40_model(&dev, 72);
    let p1 = dir.join("rank1.safetensors");
    vm_other.save(&p1).unwrap();
    let (b, t) = (4usize, cfg.ctx);
    let (ids, tg) = glance_batch(&dev, b, t, 400);
    let ids_v: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
    let tg_v: Vec<u32> = tg.flatten_all().unwrap().to_vec1().unwrap();
    let mm = ava_gpu::model::Mm { bf16: false };
    // one process, the whole batch
    let (_, vm1, m1) = ava_gpu::ckpt::load_auto(&p0, Kind::D, t, &dev).unwrap();
    let (_, _, g1) = m1.loss_and_grads_a_w(&ids, &tg, None, &mm).unwrap();
    let g1s = snapshot(&vm1, Some(&g1));
    let mut opt1 = AdamW::new(params_of(&vm1), 0.1).unwrap();
    let gn = opt1.grad_norm(&g1).unwrap();
    opt1.step(&g1, 1e-3, clip_scale(gn, 1.0)).unwrap();
    let w1 = snapshot(&vm1, None);
    // two ranks, half each
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let addr = format!("127.0.0.1:{port}");
    let handles: Vec<_> = (0..2usize)
        .map(|r| {
            let (p, addr, ids_v, tg_v) = (if r == 0 { p0.clone() } else { p1.clone() }, addr.clone(), ids_v.clone(), tg_v.clone());
            std::thread::spawn(move || {
                let dev = Device::new_cuda(0).unwrap();
                let (_, vm, m) = ava_gpu::ckpt::load_auto(&p, Kind::D, t, &dev).unwrap();
                let mut d = ava_gpu::dist::Dist::connect(r, 2, &addr).unwrap();
                let vars: Vec<Var> = params_of(&vm).into_iter().map(|p| p.var).collect();
                d.broadcast_vars(&vars).unwrap();
                let rows = b / 2;
                let ids = Tensor::from_vec(ids_v[r * rows * t..(r + 1) * rows * t].to_vec(), (rows, t), &dev).unwrap();
                let tg = Tensor::from_vec(tg_v[r * rows * t..(r + 1) * rows * t].to_vec(), (rows, t), &dev).unwrap();
                let w = Tensor::full(1.0f32 / (b * t) as f32, rows * t, &dev).unwrap();
                let mm = ava_gpu::model::Mm { bf16: false };
                let (_, _, mut g) = m.loss_and_grads_a_wr(&ids, &tg, Some(&w), true, &mm).unwrap();
                d.all_reduce_grads(&vars, &mut g).unwrap();
                let gs = snapshot(&vm, Some(&g));
                let mut opt = AdamW::new(params_of(&vm), 0.1).unwrap();
                let gn = opt.grad_norm(&g).unwrap();
                opt.step(&g, 1e-3, clip_scale(gn, 1.0)).unwrap();
                (gs, snapshot(&vm, None))
            })
        })
        .collect();
    let res: Vec<_> = handles.into_iter().map(|h| h.join().expect("rank thread")).collect();
    assert!(res[0].1 == res[1].1, "the two ranks hold different weights after the step");
    assert!(res[0].0 == res[1].0, "the two ranks hold different summed gradients");
    let mut bad = Vec::new();
    for ((name, a), (_, c)) in g1s.iter().zip(&res[0].0) {
        let dot: f64 = a.iter().zip(c).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nc: f64 = c.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let rel: f64 = a.iter().zip(c).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>().sqrt() / na.max(1e-30);
        if !(dot / (na * nc).max(1e-30) > 0.99999 && rel < 1e-4) {
            bad.push(format!("{name}: rel {rel}"));
        }
    }
    assert!(bad.is_empty(), "summed gradients differ from one process: {bad:#?}");
    // weights after one AdamW step: a first step moves every weight by about lr * sign(grad), so only a grad at
    // rounding level can flip; allow at most 0.1% of the weights to differ by more than 1e-6
    let (mut n, mut off) = (0usize, 0usize);
    for ((_, a), (_, c)) in w1.iter().zip(&res[0].1) {
        n += a.len();
        off += a.iter().zip(c).filter(|(x, y)| (**x - **y).abs() > 1e-6).count();
    }
    eprintln!("dist: {off} of {n} weights differ from one process by more than 1e-6");
    assert!(off * 1000 <= n, "{off} of {n} weights differ from one process");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn v40_bf16_step_tracks_f32() {
    // v40 (CUDA only): the burst's step (bf16 matmul operands, tensor-core attention in the engram) against the f32
    // step on the same batch: the loss agrees to 1e-2 and every gradient, including mag_w and the engram's, points
    // the same way (cos > 0.5, mean > 0.7: the spiking blocks' rule in bf16_steps_track_the_f32_gradients).
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v != "1").unwrap_or(true) {
        eprintln!("v40 bf16: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    }
    let dev = device();
    let (cfg, varmap, model) = v40_model(&dev, 81);
    let (ids, tg) = glance_batch(&dev, 4, cfg.ctx, 500);
    let (l32, _, g32) = model.loss_and_grads_a(&ids, &tg, &ava_gpu::model::Mm { bf16: false }).unwrap();
    let (l16, _, g16) = model.loss_and_grads_a(&ids, &tg, &ava_gpu::model::Mm { bf16: true }).unwrap();
    let (x, y) = (l32.to_scalar::<f32>().unwrap(), l16.to_scalar::<f32>().unwrap());
    assert!((x - y).abs() < 1e-2, "v40 loss {x} vs bf16 {y}");
    let (mut bad, mut coss) = (Vec::new(), Vec::new());
    for p in params_of(&varmap) {
        let (r, m) = (g32.get(p.var.as_tensor()).unwrap().flatten_all().unwrap(), g16.get(p.var.as_tensor()).unwrap().flatten_all().unwrap());
        let dot = r.mul(&m).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let nr = r.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        let nm = m.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        let cos = dot / (nr * nm).max(1e-30);
        coss.push(cos);
        if p.name.contains("eng") || p.name == ava_gpu::model::MAG_W {
            eprintln!("v40 bf16 {:<12} cos {cos:.5}", p.name);
        }
        if !(cos > 0.5) {
            bad.push(format!("{}: cos {cos}", p.name));
        }
    }
    let mean = coss.iter().sum::<f32>() / coss.len() as f32;
    eprintln!("v40 bf16: loss {x:.5} vs {y:.5}; mean cos {mean:.4}");
    assert!(bad.is_empty() && mean > 0.7, "v40 bf16 gradients off (mean cos {mean}): {bad:#?}");
}

// ---------------------------------------------------------------- v40: a faster block (CUDA only)

fn cuda_tests_on(what: &str) -> bool {
    let on = std::env::var("AVA_GPU_TEST_CUDA").map(|v| v == "1").unwrap_or(false);
    if !on {
        eprintln!("{what}: skipped (set AVA_GPU_TEST_CUDA=1)");
    }
    on
}

fn bits_f32(t: &Tensor) -> Vec<u32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
}

/// v40: fused::grad_norm_multi (three launches) gives the same f64 bits as AdamW::grad_norm_t's per-tensor candle
/// ops (sqr, sum_keepdim(-1) for rank >= 2, sum_all, f32 sums in order, to f64, sqrt), over shapes that exercise
/// every block size of candle's reduction: rows of 1..16 columns, of 384, 1536, 4112 x 384, columns above 1024
/// that are not a power of two, rank-1 tensors shorter and longer than 1024, and one [n, 1] column.
#[test]
fn v40_grad_norm_multi_is_bit_identical() {
    if !cuda_tests_on("v40 grad norm") {
        return;
    }
    let dev = device();
    let shapes: Vec<Vec<usize>> = vec![
        vec![4112, 384],
        vec![384],
        vec![384, 384],
        vec![384, 12],
        vec![12],
        vec![384, 1536],
        vec![1536, 384],
        vec![16, 384],
        vec![1, 5],
        vec![3, 1],
        vec![2000],
        vec![7, 2500],
        vec![5000, 3],
        vec![1],
        vec![2, 3, 33],
        vec![1025],
    ];
    for round in 0..3u64 {
        let gs: Vec<Tensor> = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let n: usize = s.iter().product();
                // values of very different sizes, so the order of the additions matters in the last bits
                let x = noise(&dev, 1, n, 1000 * round + i as u64).affine(10f64.powi((i % 5) as i32 - 2), 0.0).unwrap();
                x.reshape(s.as_slice()).unwrap()
            })
            .collect();
        let mut acc: Option<Tensor> = None;
        for g in &gs {
            let s = if g.rank() >= 2 { g.sqr().unwrap().sum_keepdim(D::Minus1).unwrap().sum_all().unwrap() } else { g.sqr().unwrap().sum_all().unwrap() };
            acc = Some(match acc {
                None => s,
                Some(a) => a.add(&s).unwrap(),
            });
        }
        let old = acc.unwrap().to_dtype(DType::F64).unwrap().sqrt().unwrap().to_scalar::<f64>().unwrap();
        let new = ava_gpu::fused::grad_norm_multi(&gs).unwrap().to_scalar::<f64>().unwrap();
        eprintln!("v40 grad norm round {round}: old {old:.17e} new {new:.17e}");
        assert_eq!(old.to_bits(), new.to_bits(), "grad norm differs: {old} vs {new}");
        // and one tensor at a time
        for g in &gs {
            let s = if g.rank() >= 2 { g.sqr().unwrap().sum_keepdim(D::Minus1).unwrap().sum_all().unwrap() } else { g.sqr().unwrap().sum_all().unwrap() };
            let o = s.to_dtype(DType::F64).unwrap().sqrt().unwrap().to_scalar::<f64>().unwrap();
            let n = ava_gpu::fused::grad_norm_multi(std::slice::from_ref(g)).unwrap().to_scalar::<f64>().unwrap();
            assert_eq!(o.to_bits(), n.to_bits(), "grad norm of {:?} differs: {o} vs {n}", g.dims());
        }
    }
}

/// v40: fused::adamw_multi (one launch) leaves w, m and v with exactly the bits of adamw_in_place_s run tensor by
/// tensor, with and without decay, for a clipping scale below 1 and for 1, over three steps.
#[test]
fn v40_adamw_multi_is_bit_identical() {
    if !cuda_tests_on("v40 adamw") {
        return;
    }
    let dev = device();
    let shapes = [(4112usize, 384usize), (384, 1), (384, 1536), (12, 1), (1536, 384), (3, 77)];
    let mk = |seed: u64| -> Vec<Tensor> { shapes.iter().enumerate().map(|(i, &(r, c))| noise(&dev, r, c, seed + i as u64)).collect() };
    let copy = |v: &[Tensor]| -> Vec<Tensor> { v.iter().map(|t| t.copy().unwrap()).collect() };
    let (w0, m0) = (mk(10), mk(20));
    let v0: Vec<Tensor> = mk(30).iter().map(|t| t.abs().unwrap()).collect();
    let (wa, ma, va) = (copy(&w0), copy(&m0), copy(&v0));
    let (wb, mb, vb) = (copy(&w0), copy(&m0), copy(&v0));
    let shrink: Vec<f64> = (0..shapes.len()).map(|i| if i % 2 == 0 { 1.0 - 1e-3 * 0.1 } else { 1.0 }).collect();
    for (step, sc_val) in [(1u64, 0.37f64), (2, 1.0), (3, 0.9123456789)] {
        let gs = mk(100 * step);
        let scale = Tensor::new(sc_val, &dev).unwrap();
        let (b1, b2, lr) = (0.9f64, 0.95f64, 1e-3);
        let sc = ava_gpu::fused::AdamWScalars {
            grad_scale: f64::NAN,
            b1,
            b2,
            eps: 1e-8,
            step_size: lr / (1.0 - b1.powi(step as i32)),
            inv_sqrt_bc2: 1.0 / (1.0 - b2.powi(step as i32)).sqrt(),
            shrink: f64::NAN,
        };
        for i in 0..shapes.len() {
            let s = ava_gpu::fused::AdamWScalars { shrink: shrink[i], ..sc };
            ava_gpu::fused::adamw_in_place_s(&wa[i], &gs[i], &ma[i], &va[i], &scale, &s).unwrap();
        }
        ava_gpu::fused::adamw_multi(&wb, &gs, &mb, &vb, &shrink, &scale, &sc).unwrap();
        for i in 0..shapes.len() {
            for (name, a, b) in [("w", &wa[i], &wb[i]), ("m", &ma[i], &mb[i]), ("v", &va[i], &vb[i])] {
                assert_eq!(bits_f32(a), bits_f32(b), "adamw step {step} tensor {i} {name} differs");
            }
        }
    }
    assert!(max_diff(&wa[0], &w0[0]) > 0.0, "the update did nothing");
}

/// bf16 bits of f (round to nearest even), as lif.cu f2bf.
fn f2bf_bits(f: f32) -> u16 {
    let u = f.to_bits();
    (u.wrapping_add(0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

/// candle's sum over dim 0 of [b, d] (fast_sum: block of next_pow2(b) threads, thread i takes row i, then a tree).
fn candle_sum0(x: &[f32], b: usize, d: usize) -> Vec<f32> {
    let bd = b.next_power_of_two();
    (0..d)
        .map(|c| {
            let mut s: Vec<f32> = (0..bd).map(|i| if i < b { 0.0f32 + x[i * d + c] } else { 0.0 }).collect();
            let mut w = bd / 2;
            while w > 0 {
                for i in 0..w {
                    s[i] += s[i + w];
                }
                w /= 2;
            }
            s[0]
        })
        .collect()
}

/// The LIF kernels' arithmetic on the CPU, operation for operation (IEEE f32 on both sides, fma as mul_add), as
/// compiled from the v39 kernels: forward v = (beta v + x) - theta s, s = [v - theta >= 0]; backward
/// a = fma(|v - theta|, slope, 1), gv = fma(beta, gv', gs / (a a)), gbeta = fma(gv, v_{t-1}, gbeta).
/// Returns (spikes [B, T, D], dcur [B, T, D], per-row gbeta [B, D]).
fn lif_cpu(cur: &[f32], beta: &[f32], gs: &[f32], b: usize, t: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (theta, slope) = (ava_gpu::model::LIF_THRESHOLD as f32, ava_gpu::model::LIF_SLOPE as f32);
    let (mut s, mut dc, mut gb) = (vec![0f32; b * t * d], vec![0f32; b * t * d], vec![0f32; b * d]);
    for bi in 0..b {
        for c in 0..d {
            let bt = beta[c];
            let at = |k: usize| (bi * t + k) * d + c;
            let mut vs = vec![0f32; t];
            let (mut v, mut sp) = (0f32, 0f32);
            for k in 0..t {
                v = ((bt * v) + cur[at(k)]) - (theta * sp);
                sp = if v - theta >= 0.0 { 1.0 } else { 0.0 };
                s[at(k)] = sp;
                vs[k] = v;
            }
            let (mut gn, mut g) = (0f32, 0f32);
            for k in (0..t).rev() {
                let a = (vs[k] - theta).abs().mul_add(slope, 1.0);
                let gv = bt.mul_add(gn, gs[at(k)] / (a * a));
                if k > 0 {
                    g = gv.mul_add(vs[k - 1], g);
                }
                dc[at(k)] = gv;
                gn = gv;
            }
            gb[bi * d + c] = g;
        }
    }
    (s, dc, gb)
}

/// v40: the LIF kernels (lif_fwd_raw f32, lif_fwd_b, lif_bwd_b) give exactly the bits of lif_cpu, the operation-for-
/// operation model of the v39 kernels (checked against them before the v40 rewrite): spikes, dL/dcur (bf16) and
/// dL/dbeta (summed over rows as candle does), at lengths that are and are not multiples of the kernels' groups.
#[test]
fn v40_lif_kernels_match_their_exact_model() {
    if !cuda_tests_on("v40 lif") {
        return;
    }
    let dev = device();
    for (b, t, d, seed) in [(2usize, 300usize, 40usize, 1u64), (3, 64, 33, 2), (1, 1000, 8, 3), (4, 2048, 64, 4), (2, 7, 5, 5), (1, 1, 3, 6)] {
        let cur = noise(&dev, b * t, d, seed).affine(0.9, 0.35).unwrap().reshape((b, t, d)).unwrap();
        let beta = noise(&dev, 1, d, seed + 10).affine(0.04, 0.88).unwrap().squeeze(0).unwrap();
        let gs = noise(&dev, b * t, d, seed + 20).reshape((b, t, d)).unwrap();
        let (cv, bv, gv) = (cur.flatten_all().unwrap().to_vec1::<f32>().unwrap(), beta.to_vec1::<f32>().unwrap(), gs.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let (s_ref, dc_ref, gb_ref) = lif_cpu(&cv, &bv, &gv, b, t, d);
        let rate = s_ref.iter().sum::<f32>() / s_ref.len() as f32;
        let s32 = ava_gpu::lif::lif_fwd_raw(&cur, &beta).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(s32.iter().zip(&s_ref).all(|(x, y)| x.to_bits() == y.to_bits()), "lif_fwd_raw spikes b{b} t{t} d{d}");
        let sb = ava_gpu::fused::lif_fwd_b(&cur, &beta).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(sb.iter().zip(&s_ref).all(|(x, y)| x.to_bits() == y.to_bits()), "lif_fwd_b spikes b{b} t{t} d{d}");
        let (dcur, dbeta) = ava_gpu::fused::lif_bwd_b(&cur, &beta, &gs).unwrap();
        let dcv: Vec<u16> = dcur.flatten_all().unwrap().to_vec1::<half::bf16>().unwrap().iter().map(|x| x.to_bits()).collect();
        let bad = dcv.iter().zip(&dc_ref).filter(|(x, y)| **x != f2bf_bits(**y)).count();
        assert_eq!(bad, 0, "lif_bwd_b dcur b{b} t{t} d{d}: {bad} of {} differ", dcv.len());
        let db = dbeta.to_vec1::<f32>().unwrap();
        let db_ref = candle_sum0(&gb_ref, b, d);
        assert!(db.iter().zip(&db_ref).all(|(x, y)| x.to_bits() == y.to_bits()), "lif_bwd_b dbeta b{b} t{t} d{d}: {db:?} vs {db_ref:?}");
        eprintln!("v40 lif b{b} t{t} d{d}: spike rate {rate:.3}, all bits equal");
    }
}

/// v40: hebb_bwd_w1 (one head per block of 32 threads) gives exactly the bits of hebb_bwd (four heads per block of
/// 128): the same code per head, compiled under a lower register cap. dP, every column, at shapes whose head count is
/// and is not a multiple of 4 and whose length is and is not a multiple of the 16-step checkpoints.
#[test]
fn v40_hebb_bwd_one_head_blocks_are_bit_identical() {
    if !cuda_tests_on("v40 hebb_bwd_w1") {
        return;
    }
    let dev = device();
    let eps = ava_gpu::model::NORM_EPS;
    for (b, nh, t, seed) in [(2usize, 3usize, 64usize, 410u64), (3, 2, 33, 420), (32, 12, 256, 430), (1, 5, 129, 440)] {
        let (d, n) = (nh * 32, b * t);
        let ld = 4 * d + 2 * nh;
        let p = noise(&dev, n, ld, seed);
        let cst = noise(&dev, 1, 3 * nh, seed + 1).squeeze(0).unwrap();
        let d_o = noise(&dev, n, d, seed + 2);
        let (_, ck) = ava_gpu::fused::hebb_fwd_per_token(&p, &cst, b, t, nh, eps).unwrap();
        let old = ava_gpu::fused::hebb_bwd_per_token_k(&p, &cst, &ck, &d_o, b, t, nh, eps, false).unwrap();
        let new = ava_gpu::fused::hebb_bwd_per_token_k(&p, &cst, &ck, &d_o, b, t, nh, eps, true).unwrap();
        let (a, c) = (bits_f32(&old), bits_f32(&new));
        let bad = a.iter().zip(&c).filter(|(x, y)| x != y).count();
        assert_eq!(bad, 0, "hebb_bwd_w1 b{b} h{nh} t{t}: {bad} of {} differ", a.len());
        eprintln!("v40 hebb_bwd_w1 b{b} h{nh} t{t}: all {} bits equal", a.len());
    }
}

fn bits_bf16(t: &Tensor) -> Vec<u16> {
    t.flatten_all().unwrap().to_vec1::<half::bf16>().unwrap().iter().map(|x| x.to_bits()).collect()
}

/// v40: the fast-weight kernels that write their matmul operand as bf16 (hebb_fwd_raw_b: O; hebb_bwd_raw_b: dP and
/// dz) give exactly f2bf of what the f32 kernels give, the same saved states, and dz equal to dP's za, zb columns;
/// the bias gradients summed from dz have the same bits as from dP's columns (model.rs reads them from dz). Shapes
/// for both the chunk form (b·nh < 4 × SMs) and the per-token kernels (b·nh = 192, 384), both per-token layouts.
#[test]
fn v40_hebb_bf16_outputs_are_the_cast_of_the_f32_outputs() {
    if !cuda_tests_on("v40 hebb bf16 outputs") {
        return;
    }
    let dev = device();
    let eps = ava_gpu::model::NORM_EPS;
    for (b, nh, t, seed) in [(2usize, 3usize, 64usize, 510u64), (3, 2, 33, 520), (4, 12, 2048, 530), (16, 12, 37, 540), (32, 12, 256, 550)] {
        let (d, n) = (nh * 32, b * t);
        let ld = 4 * d + 2 * nh;
        let p = noise(&dev, n, ld, seed);
        let cst = noise(&dev, 1, 3 * nh, seed + 1).squeeze(0).unwrap();
        let d_o = noise(&dev, n, d, seed + 2);
        let chunk = ava_gpu::fused::hebb_use_chunk(&dev, b, nh).unwrap();
        let (o32, ck32) = ava_gpu::fused::hebb_fwd_raw(&p, &cst, b, t, nh, eps).unwrap();
        let (ob, ckb) = ava_gpu::fused::hebb_fwd_raw_b(&p, &cst, b, t, nh, eps).unwrap();
        assert_eq!(ob.dtype(), DType::BF16);
        let want: Vec<u16> = bits_f32(&o32).into_iter().map(|u| f2bf_bits(f32::from_bits(u))).collect();
        let bad = bits_bf16(&ob).iter().zip(&want).filter(|(x, y)| x != y).count();
        assert_eq!(bad, 0, "hebb O bf16 b{b} h{nh} t{t}: {bad} of {} differ", want.len());
        assert!(bits_f32(&ck32) == bits_f32(&ckb), "hebb ck b{b} h{nh} t{t}");
        let dp32 = ava_gpu::fused::hebb_bwd_raw(&p, &cst, &ck32, &d_o, b, t, nh, eps).unwrap();
        let want: Vec<u16> = bits_f32(&dp32).into_iter().map(|u| f2bf_bits(f32::from_bits(u))).collect();
        let mut runs = vec![ava_gpu::fused::hebb_bwd_raw_b(&p, &cst, &ck32, &d_o, b, t, nh, eps).unwrap()];
        if !chunk {
            for w1 in [false, true] {
                runs.push(ava_gpu::fused::hebb_bwd_per_token_b(&p, &cst, &ck32, &d_o, b, t, nh, eps, w1).unwrap());
            }
        }
        let zcols = dp32.narrow(1, 4 * d, 2 * nh).unwrap().contiguous().unwrap();
        for (i, (dpb, dz)) in runs.iter().enumerate() {
            let bad = bits_bf16(dpb).iter().zip(&want).filter(|(x, y)| x != y).count();
            assert_eq!(bad, 0, "hebb dP bf16 b{b} h{nh} t{t} run {i}: {bad} of {} differ", want.len());
            assert!(bits_f32(dz) == bits_f32(&zcols), "hebb dz b{b} h{nh} t{t} run {i}");
            for (c0, z0) in [(4 * d, 0), (4 * d + nh, nh)] {
                let s32 = dp32.narrow(1, c0, nh).unwrap().sum(0).unwrap();
                let sz = dz.narrow(1, z0, nh).unwrap().sum(0).unwrap();
                assert!(bits_f32(&s32) == bits_f32(&sz), "hebb bias sum b{b} h{nh} t{t} run {i} col {c0}");
            }
        }
        eprintln!("v40 hebb bf16 b{b} h{nh} t{t} ({}): O, ck, dP, dz and the bias sums all bit-equal ({} runs)", if chunk { "chunk" } else { "per-token" }, runs.len());
    }
}

// ------------------------------------------------------------------------------------------------ v41 anchor

fn vals(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// Importance = mean g² over batches normalised to mean 1; pull = w − min(lr·s·F, 1)·(w − w0), exactly; a weight
/// with no gradient is not anchored; drift = mean of F·(w − w0)².
#[test]
fn v41_anchor_importance_pull_and_drift() {
    use ava_gpu::anchor::Importance;
    let dev = device();
    let a = Var::from_vec(vec![1f32, 2., 3., 4.], 4, &dev).unwrap();
    let b = Var::from_vec(vec![0f32, 0.], 2, &dev).unwrap();
    let c = Var::from_vec(vec![5f32], 1, &dev).unwrap();
    let params = vec![
        Param { name: "a".into(), var: a.clone(), decay: true },
        Param { name: "b".into(), var: b.clone(), decay: false },
        Param { name: "c".into(), var: c.clone(), decay: false },
    ];
    assert!(Importance::new(&params).anchor(1.0).unwrap().is_none(), "no batches: no anchor");
    let mut imp = Importance::new(&params);
    // gradients: a [1,1,2,0] then [1,1,0,0]; b [2,0] then [0,0]; c none
    for (ga, gb) in [([1f32, 1., 2., 0.], [2f32, 0.]), ([1., 1., 0., 0.], [0., 0.])] {
        let ca = Tensor::new(&ga, &dev).unwrap();
        let cb = Tensor::new(&gb, &dev).unwrap();
        let loss = a.as_tensor().mul(&ca).unwrap().sum_all().unwrap().add(&b.as_tensor().mul(&cb).unwrap().sum_all().unwrap()).unwrap();
        imp.add(&loss.backward().unwrap()).unwrap();
    }
    // sums of g²: a [2,2,4,0], b [4,0]; mean 12/6 = 2 → F a [1,1,2,0], b [2,0]
    let an = imp.anchor(10.0).unwrap().expect("anchor");
    assert_eq!(an.weights, 6);
    assert_eq!(an.importance().len(), 2, "c got no gradient, so it is not anchored");
    assert_eq!(vals(&an.importance()[0]), vec![1., 1., 2., 0.]);
    assert_eq!(vals(&an.importance()[1]), vec![2., 0.]);
    // move away, then pull at lr 0.05 (s = 0.5): a c = [0.5,0.5,1,0], b c = [1,0]
    a.set(&Tensor::new(&[2f32, 2., 2., 2.], &dev).unwrap()).unwrap();
    b.set(&Tensor::new(&[1f32, 1.], &dev).unwrap()).unwrap();
    c.set(&Tensor::new(&[7f32], &dev).unwrap()).unwrap();
    an.pull(0.05).unwrap();
    assert_eq!(vals(a.as_tensor()), vec![1.5, 2., 3., 2.]);
    assert_eq!(vals(b.as_tensor()), vec![0., 1.]);
    assert_eq!(vals(c.as_tensor()), vec![7.], "an unanchored weight is free");
    // drift: F·(w−w0)²: a [1·0.25, 0, 0, 0·4], b [0, 0] → 0.25 / 6
    assert!((an.drift().unwrap() - 0.25 / 6.0).abs() < 1e-7);
    // a pull past 1 is clamped: lr 1 (s = 10) puts every weight with F > 0 back exactly
    an.pull(1.0).unwrap();
    assert_eq!(vals(a.as_tensor()), vec![1., 2., 3., 2.]);
    assert_eq!(vals(b.as_tensor()), vec![0., 1.]);
    // strength · lr = 0 leaves every weight bit-identical
    a.set(&Tensor::new(&[9f32, 8., 7., 6.], &dev).unwrap()).unwrap();
    an.pull(0.0).unwrap();
    assert_eq!(vals(a.as_tensor()), vec![9., 8., 7., 6.]);
}
