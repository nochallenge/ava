// TESTING / DESIGN ONLY — op_bench: GPU time of each op of one Ava-S block at a training shape.
//
// op_bench [--d 384] [--rows 8192] [--reps 20]
// Times forward (and forward+backward where it matters) of: the d→d and d→4d matmuls, rmsnorm, the
// fused LIF scan, the kWTA mask (sort-based), relu and the tied-output logits. Synchronises the device
// around every timed loop. Numbers are per call, in milliseconds.

use ava_gpu::model::{kwta_mask, kwta_mask_sorted, lif_scan, rmsnorm, KWTA_DIV, MLP_MULT, VOCAB};
use candle_core::{DType, Device, Tensor, Var};
use std::time::Instant;

fn arg(name: &str, default: usize) -> usize {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).map(|i| a[i + 1].parse().expect("number")).unwrap_or(default)
}

fn time<F: FnMut() -> candle_core::Result<()>>(dev: &Device, reps: usize, mut f: F) -> f64 {
    f().unwrap();
    dev.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..reps {
        f().unwrap();
    }
    dev.synchronize().unwrap();
    t.elapsed().as_secs_f64() * 1000.0 / reps as f64
}

fn main() -> candle_core::Result<()> {
    let (d, n, reps) = (arg("--d", 384), arg("--rows", 8192), arg("--reps", 20));
    let h = MLP_MULT * d;
    let k = h / KWTA_DIV;
    let dev = Device::new_cuda(0)?;
    let x = Var::randn(0f32, 1.0, (n, d), &dev)?;
    let w = Var::randn(0f32, 0.05, (d, d), &dev)?;
    let w1 = Var::randn(0f32, 0.05, (d, h), &dev)?;
    let emb = Var::randn(0f32, 0.02, (VOCAB, d), &dev)?;
    let g = Tensor::ones(d, DType::F32, &dev)?;
    let beta = Var::new(vec![2.0f32; d], &dev)?;
    let u = Tensor::randn(0f32, 1.0, (n, h), &dev)?;
    let b = 32usize;
    let t = n / b;

    println!("op_bench d={d} rows={n} (batch {b} x ctx {t}) hidden={h} k={k}; ms per call");
    let r = |name: &str, ms: f64| println!("{name:<34} {ms:>8.3}");
    r("matmul d->d fwd", time(&dev, reps, || x.matmul(&w).map(|_| ())));
    r("matmul d->4d fwd", time(&dev, reps, || x.matmul(&w1).map(|_| ())));
    r("matmul d->4d fwd+bwd", time(&dev, reps, || x.matmul(&w1)?.sum_all()?.backward().map(|_| ())));
    r("rmsnorm fwd", time(&dev, reps, || rmsnorm(&x, &g).map(|_| ())));
    r("rmsnorm fwd+bwd", time(&dev, reps, || rmsnorm(&x, &g)?.sum_all()?.backward().map(|_| ())));
    let cur = x.as_tensor().reshape((b, t, d))?;
    r("lif fused fwd", time(&dev, reps, || lif_scan(&cur, &beta).map(|_| ())));
    r("lif fused fwd+bwd", time(&dev, reps, || lif_scan(&cur, &beta)?.sum_all()?.backward().map(|_| ())));
    r("kwta mask (sort) fwd", time(&dev, reps, || kwta_mask_sorted(&u, k).map(|_| ())));
    r("kwta mask (radix kernel) fwd", time(&dev, reps, || kwta_mask(&u, k).map(|_| ())));
    for (name, uu) in [("randn", u.clone()), ("rounded (many ties)", (u.clone() * 4.0)?.round()?), ("relu-like (many zeros)", u.relu()?)] {
        let (a, b2) = (kwta_mask_sorted(&uu, k)?, kwta_mask(&uu, k)?);
        let diff = a.sub(&b2)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        let kept = b2.sum_all()?.to_scalar::<f32>()? / n as f32;
        println!("  mask check {name:<24} differing entries {diff}, kept per row {kept:.2} (k {k})");
    }
    r("kwta mask + mul + relu fwd", time(&dev, reps, || kwta_mask(&u, k).and_then(|m| u.mul(&m)?.relu()).map(|_| ())));
    r("relu fwd", time(&dev, reps, || u.relu().map(|_| ())));
    r("logits x.E^T fwd", time(&dev, reps, || x.matmul(&emb.t()?).map(|_| ())));
    {
        let l = x.matmul(&emb.t()?)?;
        let tg = Tensor::from_vec((0..n).map(|i| ((i * 2654435761usize) % VOCAB) as u32).collect::<Vec<u32>>(), n, &dev)?;
        let a = candle_nn::loss::cross_entropy(&l, &tg)?.to_scalar::<f32>()?;
        let b2 = ava_gpu::lif::cross_entropy_fused(&l, &tg)?.to_scalar::<f32>()?;
        let lv = Var::from_tensor(&l)?;
        let ga = candle_nn::loss::cross_entropy(lv.as_tensor(), &tg)?.backward()?.get(&lv).unwrap().clone();
        let gb = ava_gpu::lif::cross_entropy_fused(lv.as_tensor(), &tg)?.backward()?.get(&lv).unwrap().clone();
        let gd = ga.sub(&gb)?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("  cross-entropy check: candle {a:.6} fused {b2:.6}; max |grad diff| {gd:.3e}");
        r("fused cross-entropy fwd+bwd", time(&dev, reps, || ava_gpu::lif::cross_entropy_fused(lv.as_tensor(), &tg)?.backward().map(|_| ())));
    }
    r("logits + cross-entropy fwd+bwd", time(&dev, reps, || {
        let l = x.matmul(&emb.t()?)?;
        let tg = Tensor::zeros(n, DType::U32, &dev)?;
        candle_nn::loss::cross_entropy(&l, &tg)?.backward().map(|_| ())
    }));
    Ok(())
}
