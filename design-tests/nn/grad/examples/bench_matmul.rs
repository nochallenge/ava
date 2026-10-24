// TESTING / DESIGN ONLY — matmul throughput benchmark for budgeting training runs.
// Measures GFLOPS (2·m·k·n / best wall time) of ava_grad::matmul::gemm for square 256/512/1024 and the
// [4096,256]×[256,1024] "activations × weight" shape, plus that shape's two backward products
// (dA = dC·Bᵀ via trans_b, dB = Aᵀ·dC via trans_a), with 1 thread and with AVA_THREADS (default 4).
// Run: CARGO_BUILD_JOBS=4 cargo run --release --example bench_matmul

use ava_grad::matmul::gemm;
use ava_grad::{threads, Rng, Tensor};
use std::time::{Duration, Instant};

#[allow(clippy::too_many_arguments)]
fn gflops(m: usize, k: usize, n: usize, ta: bool, tb: bool, threads: usize, rng: &mut Rng) -> f64 {
    let a = Tensor::randn(&[m * k], 1.0, rng).data;
    let b = Tensor::randn(&[k * n], 1.0, rng).data;
    let mut c = vec![0.0f32; m * n];
    gemm(m, k, n, &a, ta, &b, tb, &mut c, threads); // warm-up
    let mut best = Duration::MAX;
    let start = Instant::now();
    let mut reps = 0;
    while reps < 3 || (start.elapsed() < Duration::from_millis(600) && reps < 40) {
        c.iter_mut().for_each(|v| *v = 0.0);
        let t0 = Instant::now();
        gemm(m, k, n, &a, ta, &b, tb, &mut c, threads);
        best = best.min(t0.elapsed());
        reps += 1;
    }
    std::hint::black_box(&c);
    2.0 * (m * k * n) as f64 / best.as_secs_f64() / 1e9
}

fn main() {
    let nt = threads();
    let mut rng = Rng::new(1);
    let cases: [(&str, usize, usize, usize, bool, bool); 7] = [
        ("256 x 256 x 256", 256, 256, 256, false, false),
        ("512 x 512 x 512", 512, 512, 512, false, false),
        ("1024 x 1024 x 1024", 1024, 1024, 1024, false, false),
        ("[4096,256] x [256,1024]", 4096, 256, 1024, false, false),
        ("  bwd dA: [4096,1024] x [256,1024]^T", 4096, 1024, 256, false, true),
        ("  bwd dB: [4096,256]^T x [4096,1024]", 256, 4096, 1024, true, false),
        ("matmul_t [512,64] x [512,64]^T", 512, 64, 512, false, true),
    ];
    println!("ava_grad gemm benchmark (AVA_THREADS -> {nt} threads)");
    println!("{:<40} {:>12} {:>12} {:>9}", "shape (m x k x n)", "1 thr GF/s", format!("{nt} thr GF/s"), "speedup");
    for (name, m, k, n, ta, tb) in cases {
        let g1 = gflops(m, k, n, ta, tb, 1, &mut rng);
        let gn = gflops(m, k, n, ta, tb, nt, &mut rng);
        println!("{:<40} {:>12.1} {:>12.1} {:>8.2}x", name, g1, gn, gn / g1);
    }
}
