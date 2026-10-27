// TESTING / DESIGN ONLY — attn_bench: the transformer's attention core, candle chain vs the fused kernel.
//
// attn_bench [--b 32] [--heads 8] [--t 256] [--reps 10] [--rounds 15]
// Times forward + backward of softmax(q kᵀ s + causal mask) v both ways, alternating, and prints the best
// round of each (the minimum is the least disturbed by other jobs sharing the GPU). Milliseconds per call.

use candle_core::{Device, Tensor, Var, D};
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
    let (b, nh, t, reps, rounds) = (arg("--b", 32), arg("--heads", 8), arg("--t", 256), arg("--reps", 10), arg("--rounds", 15));
    let dev = Device::new_cuda(0)?;
    let mk = || Var::from_tensor(&Tensor::randn(0f32, 1.0, (b, nh, t, 32), &dev).unwrap()).unwrap();
    let (q, k, v) = (mk(), mk(), mk());
    let w = Tensor::randn(0f32, 1.0, (b, nh, t, 32), &dev)?;
    let mask = ava_gpu::model::causal_mask(t, &dev)?;
    let scale = 1.0 / 32f64.sqrt();
    let chain = || -> candle_core::Result<()> {
        let sc = q.as_tensor().matmul(&k.as_tensor().t()?.contiguous()?)?.affine(scale, 0.0)?;
        let o = candle_nn::ops::softmax(&sc.broadcast_add(&mask)?, D::Minus1)?.matmul(v.as_tensor())?;
        o.mul(&w)?.sum_all()?.backward()?;
        Ok(())
    };
    let fused = || -> candle_core::Result<()> {
        let o = ava_gpu::fused::causal_attention(q.as_tensor(), k.as_tensor(), v.as_tensor(), scale)?;
        o.mul(&w)?.sum_all()?.backward()?;
        Ok(())
    };
    let (mut best_c, mut best_f) = (f64::MAX, f64::MAX);
    for _ in 0..rounds {
        best_c = best_c.min(time(&dev, reps, chain));
        best_f = best_f.min(time(&dev, reps, fused));
    }
    println!("attn fwd+bwd b={b} heads={nh} t={t}: chain {best_c:.3} ms, fused {best_f:.3} ms, {:.2}x", best_c / best_f);
    Ok(())
}
