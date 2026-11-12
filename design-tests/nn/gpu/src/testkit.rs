// TESTING / DESIGN ONLY — testkit: small helpers for the CPU unit tests (tiny models with fixed weights, a byte-level
// tokenizer, scratch folders). Compiled for tests only.

use crate::model::{Config, Kind, Model};
use ava_data::bpe::Bpe;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::path::{Path, PathBuf};

/// The byte-level tokenizer (no merges): 256 bytes, EOS and PAD, 258 ids.
pub fn byte_bpe() -> Bpe {
    Bpe::from_merges(Vec::new()).expect("byte-level tokenizer")
}

/// A fresh, empty scratch folder for one test (under the system temp folder; the name never holds a refused word).
pub fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ava_gpu_scratch_{}", std::process::id())).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch folder");
    d
}

/// Deterministic noise with mean 0 and variance 1 (xorshift, uniform on ±√3).
pub fn unit_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let u = (x >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
            ((u * 2.0 - 1.0) * 3f64.sqrt()) as f32
        })
        .collect()
}

/// Replace every randomly initialised weight of `varmap` by fixed noise (candle's CPU init cannot be seeded), so a test
/// model is the same in every run. Model::new draws every random weight from a normal of mean 0 with one of three
/// spreads (INIT_STD, 1/√d, resid_std); the measured spread picks the nearest of them, so two builds with the same seed
/// are equal bit for bit. Constant weights (gains, offsets, zero read-outs) stay as they are.
pub fn fix_weights(varmap: &VarMap, cfg: &Config, seed: u64) {
    let spreads = [crate::model::INIT_STD, 1.0 / (cfg.d as f64).sqrt(), cfg.resid_std()];
    let data = varmap.data().lock().unwrap();
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    for (k, name) in names.into_iter().enumerate() {
        let var = &data[name];
        let v: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
        let n = v.len() as f64;
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
        let sd = (v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
        if sd == 0.0 {
            continue;
        }
        let sd = spreads.iter().copied().min_by(|a, b| (a.ln() - sd.ln()).abs().total_cmp(&(b.ln() - sd.ln()).abs())).unwrap();
        let w: Vec<f32> = unit_noise(v.len(), seed.wrapping_mul(1000).wrapping_add(k as u64)).iter().map(|&z| (sd * z as f64) as f32).collect();
        var.set(&Tensor::from_vec(w, var.shape(), &Device::Cpu).unwrap()).unwrap();
    }
}

/// A tiny model on the CPU (vocab 258 = the byte tokenizer, d 32, 2 blocks, ctx 64) with fixed weights from `seed`.
pub fn tiny(kind: Kind, seed: u64) -> (Config, VarMap, Model) {
    tiny_with(kind, 258, 32, 2, seed)
}

pub fn tiny_with(kind: Kind, vocab: usize, d: usize, layers: usize, seed: u64) -> (Config, VarMap, Model) {
    let cfg = Config::with_kind(kind, vocab, d, layers, 64);
    let varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu)).unwrap();
    fix_weights(&varmap, &cfg, seed);
    model.set_track_activity(false);
    (cfg, varmap, model)
}

/// Save a tiny model to `path` (safetensors).
pub fn save(varmap: &VarMap, path: &Path) {
    varmap.save(path).unwrap();
}

#[test]
fn the_same_seed_gives_the_same_weights() {
    for kind in [Kind::D, Kind::T] {
        let (_, a, _) = tiny(kind, 9);
        let (_, b, _) = tiny(kind, 9);
        let (a, b) = (a.data().lock().unwrap(), b.data().lock().unwrap());
        for (k, v) in a.iter() {
            let x: Vec<f32> = v.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
            let y: Vec<f32> = b[k].as_tensor().flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(x, y, "{k}");
        }
    }
}
