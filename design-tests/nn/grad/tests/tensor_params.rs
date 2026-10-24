// TESTING / DESIGN ONLY — tests of Rng (splitmix64 + Box–Muller), Tensor constructors, the Params store
// (counts, decay flags, gradients) and the AVAWTS01 checkpoint format (exact byte layout + round trip).

use ava_grad::params::read_checkpoint;
use ava_grad::*;
use std::path::PathBuf;

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ava_grad_test_{}_{}", std::process::id(), name));
    p
}

#[test]
fn rng_is_deterministic_and_normal_has_unit_moments() {
    let mut a = Rng::new(123);
    let mut b = Rng::new(123);
    for _ in 0..100 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
    // splitmix64 reference value for seed 0 (first output)
    assert_eq!(Rng::new(0).next_u64(), 0xE220A8397B1DCDAF);
    let mut r = Rng::new(7);
    let n = 200_000;
    let xs: Vec<f64> = (0..n).map(|_| r.normal() as f64).collect();
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    assert!(mean.abs() < 0.01, "mean {mean}");
    assert!((var - 1.0).abs() < 0.01, "var {var}");
    for _ in 0..1000 {
        let u = r.next_f32();
        assert!((0.0..1.0).contains(&u));
        assert!(r.below(10) < 10);
    }
}

#[test]
fn tensor_constructors() {
    let z = Tensor::zeros(&[2, 3]);
    assert_eq!(z.numel(), 6);
    assert!(z.data.iter().all(|&v| v == 0.0));
    let f = Tensor::from_vec(&[3], vec![1.0, 2.0, 3.0]);
    assert_eq!(f.shape, vec![3]);
    let mut rng = Rng::new(1);
    let r = Tensor::randn(&[1000], 0.5, &mut rng);
    let sd = (r.data.iter().map(|x| x * x).sum::<f32>() / 1000.0).sqrt();
    assert!((sd - 0.5).abs() < 0.05);
    let r2 = Tensor::randn(&[1000], 0.5, &mut Rng::new(1));
    assert_eq!(r, r2);
    assert_eq!(Tensor::scalar(2.5).numel(), 1);
}

#[test]
#[should_panic]
fn tensor_from_vec_rejects_wrong_length() {
    let _ = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0]);
}

#[test]
fn params_counts_and_flags() {
    let mut p = Params::new();
    let w = p.add("blk0.w", Tensor::zeros(&[4, 8]), true);
    let g = p.add("blk0.gain", Tensor::zeros(&[8]), false);
    let e = p.add("emb", Tensor::zeros(&[10, 8]), false);
    assert_eq!(p.len(), 3);
    assert_eq!(p.count(), 32 + 8 + 80);
    assert_eq!(p.count_where(|n| n.starts_with("blk0.")), 40);
    assert!(p.decay(w) && !p.decay(g) && !p.decay(e));
    assert_eq!(p.name(g), "blk0.gain");
    assert_eq!(p.find("emb"), Some(e));
    assert_eq!(p.find("nope"), None);
    assert_eq!(p.grad(w).shape, vec![4, 8]);
    p.grad_mut(w).data[3] = 2.0;
    assert_eq!(p.grad_norm(), 2.0);
    p.zero_grads();
    assert_eq!(p.grad_norm(), 0.0);
}

#[test]
#[should_panic]
fn params_reject_duplicate_names() {
    let mut p = Params::new();
    p.add("w", Tensor::zeros(&[1]), true);
    p.add("w", Tensor::zeros(&[1]), true);
}

#[test]
fn checkpoint_exact_byte_layout() {
    let mut p = Params::new();
    p.add("ab", Tensor::from_vec(&[2], vec![1.0, -2.0]), true);
    let path = tmp_path("layout.bin");
    p.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let mut want = Vec::new();
    want.extend_from_slice(b"AVAWTS01");
    want.extend_from_slice(&1u32.to_le_bytes()); // n_tensors
    want.extend_from_slice(&2u32.to_le_bytes()); // name_len
    want.extend_from_slice(b"ab");
    want.extend_from_slice(&1u32.to_le_bytes()); // ndim
    want.extend_from_slice(&2u64.to_le_bytes()); // dim 0
    want.extend_from_slice(&1.0f32.to_le_bytes());
    want.extend_from_slice(&(-2.0f32).to_le_bytes());
    assert_eq!(bytes, want);
}

#[test]
fn checkpoint_round_trip() {
    let mut rng = Rng::new(5);
    let mut p = Params::new();
    let a = p.add("layer.0.w", Tensor::randn(&[3, 4, 5], 1.0, &mut rng), true);
    let b = p.add("gain", Tensor::randn(&[7], 1.0, &mut rng), false);
    let s = p.add("scalarish", Tensor::from_vec(&[1], vec![f32::MIN_POSITIVE]), false);
    let path = tmp_path("roundtrip.bin");
    p.save(&path).unwrap();

    // a fresh store with the same names/shapes but different values, then load
    let mut q = Params::new();
    let qa = q.add("layer.0.w", Tensor::zeros(&[3, 4, 5]), true);
    let qb = q.add("gain", Tensor::zeros(&[7]), false);
    let qs = q.add("scalarish", Tensor::zeros(&[1]), false);
    q.load(&path).unwrap();
    assert_eq!(q.value(qa), p.value(a));
    assert_eq!(q.value(qb), p.value(b));
    assert_eq!(q.value(qs), p.value(s));

    // raw reader
    let items = read_checkpoint(&path).unwrap();
    assert_eq!(items.len(), 3);
    assert_eq!(items[0].0, "layer.0.w");
    assert_eq!(&items[0].1, p.value(a));

    // shape mismatch and missing names are errors, not silent partial loads
    let mut bad = Params::new();
    bad.add("layer.0.w", Tensor::zeros(&[3, 4, 6]), true);
    bad.add("gain", Tensor::zeros(&[7]), false);
    bad.add("scalarish", Tensor::zeros(&[1]), false);
    assert!(bad.load(&path).is_err());
    let mut missing = Params::new();
    missing.add("layer.0.w", Tensor::zeros(&[3, 4, 5]), true);
    missing.add("other", Tensor::zeros(&[7]), false);
    assert!(missing.load(&path).is_err());

    // a truncated file is rejected
    let bytes = std::fs::read(&path).unwrap();
    let cut = tmp_path("truncated.bin");
    std::fs::write(&cut, &bytes[..bytes.len() - 3]).unwrap();
    assert!(read_checkpoint(&cut).is_err());
    let _ = std::fs::remove_file(&cut);
    let _ = std::fs::remove_file(&path);
}
