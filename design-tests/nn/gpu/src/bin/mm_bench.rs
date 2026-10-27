// TESTING / DESIGN ONLY — mm_bench: matmul time at the training shapes, f32 (TF32 on) vs bf16 (f32 accumulate).
//
// mm_bench [--rows 8192] [--d 384] [--reps 20] [--rounds 8]
// For each shape [rows, k] x [k, m]: TF32 f32 matmul; candle's bf16 matmul (bf16 out); fused::mm_bf16 (cublasGemmEx,
// bf16 in, f32 out) on bf16 inputs, and with the activation's f32 -> bf16 cast included. Best round of each (least
// disturbed by other GPU jobs), ms per call; plus mm_bf16's max error relative to the largest f32 output.

use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn arg(name: &str, default: usize) -> usize {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).map(|i| a[i + 1].parse().expect("number")).unwrap_or(default)
}

fn time<F: FnMut() -> candle_core::Result<()>>(dev: &Device, reps: usize, mut f: F) -> f64 {
    dev.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..reps {
        f().unwrap();
    }
    dev.synchronize().unwrap();
    t.elapsed().as_secs_f64() * 1000.0 / reps as f64
}

fn main() -> candle_core::Result<()> {
    use ava_gpu::fused::mm_bf16;
    let (n, d, reps, rounds) = (arg("--rows", 8192), arg("--d", 384), arg("--reps", 20), arg("--rounds", 8));
    let dev = Device::new_cuda(0)?;
    candle_core::cuda::set_gemm_reduced_precision_f32(true);
    let shapes = [("x.W d->d", n, d, d), ("x.W1 d->4d", n, d, 4 * d), ("a.W2 4d->d", n, 4 * d, d), ("logits d->V", n, d, 4096), ("dW = aT.dx", 4 * d, n, d)];
    let (mut t32, mut tex, mut texc) = (0.0, 0.0, 0.0);
    for (name, r, k, m) in shapes {
        let a = Tensor::randn(0f32, 1.0, (r, k), &dev)?;
        let b = Tensor::randn(0f32, 1.0, (k, m), &dev)?;
        let (ah, bh) = (a.to_dtype(DType::BF16)?, b.to_dtype(DType::BF16)?);
        let (mut f32b, mut bf, mut ex, mut exc) = (f64::MAX, f64::MAX, f64::MAX, f64::MAX);
        for _ in 0..rounds {
            f32b = f32b.min(time(&dev, reps, || a.matmul(&b).map(|_| ())));
            bf = bf.min(time(&dev, reps, || ah.matmul(&bh).map(|_| ())));
            ex = ex.min(time(&dev, reps, || mm_bf16(&ah, false, &bh, false).map(|_| ())));
            exc = exc.min(time(&dev, reps, || mm_bf16(&a.to_dtype(DType::BF16)?, false, &bh, false).map(|_| ())));
        }
        let want = a.matmul(&b)?;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
        let rel = mm_bf16(&ah, false, &bh, false)?.sub(&want)?.abs()?.max_all()?.to_scalar::<f32>()? / scale;
        // transposed forms against f32
        let at = a.t()?.contiguous()?.to_dtype(DType::BF16)?;
        let bt = b.t()?.contiguous()?.to_dtype(DType::BF16)?;
        let rel_t = mm_bf16(&at, true, &bt, true)?.sub(&want)?.abs()?.max_all()?.to_scalar::<f32>()? / scale;
        let gf = 2.0 * (r * k * m) as f64 / 1e9;
        println!(
            "{name:<14} [{r},{k}]x[{k},{m}]: tf32 {f32b:.3} ms ({:.1} TFLOP/s), candle bf16 {bf:.3}, mm_bf16 {ex:.3} ({:.1} TFLOP/s), + act cast {exc:.3}; rel err {rel:.4} (transposed {rel_t:.4})",
            gf / f32b,
            gf / ex
        );
        t32 += f32b;
        tex += ex;
        texc += exc;
    }
    println!("sum: tf32 {t32:.3} ms, mm_bf16 {tex:.3} ms ({:.2}x), with activation cast {texc:.3} ms ({:.2}x)", t32 / tex, t32 / texc);
    Ok(())
}
