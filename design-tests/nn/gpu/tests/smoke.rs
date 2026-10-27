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
