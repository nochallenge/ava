// TESTING / DESIGN ONLY — ava_gpu::fast4::fmm against candle's CPU matmul, bit for bit.

use super::fmm::{xet, xw};
use candle_core::{Device, Tensor};

fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect()
}

#[test]
fn fast4_fmm_matches_candle() {
    if !crate::int8::simd_ok() {
        return;
    }
    let dev = Device::Cpu;
    let mut checked = 0;
    // x · W, every gemm one-row path: k = 1 and 2 (gevv), k >= 3 (column-major gemv); n with every tail
    for &(k, n) in &[(1, 9), (2, 13), (3, 2), (5, 7), (17, 33), (64, 6), (64, 100), (384, 384), (384, 1536), (1536, 384), (100, 1029)] {
        let (x, w) = (rnd(k, k as u64), rnd(k * n, (k * n) as u64 + 7));
        let want: Vec<f32> = Tensor::from_vec(x.clone(), (1, k), &dev).unwrap().matmul(&Tensor::from_vec(w.clone(), (k, n), &dev).unwrap()).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        for th in [1, 8] {
            let got = xw(&x, &w, n, th);
            assert!(want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()), "x·W [{k}, {n}] threads {th}");
            checked += 1;
        }
    }
    // x · E^T (the head, gemm's row-major gemv): d with every tail; V > 256 (below, gemm takes its small kernel)
    for &(v, d) in &[(300, 3), (257, 9), (512, 64), (1000, 100), (4096, 384), (777, 71), (300, 130)] {
        let (x, e) = (rnd(d, d as u64 + 3), rnd(v * d, (v * d) as u64));
        let et = Tensor::from_vec(e.clone(), (v, d), &dev).unwrap();
        let want: Vec<f32> = Tensor::from_vec(x.clone(), (1, d), &dev).unwrap().matmul(&et.t().unwrap()).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        for th in [1, 8] {
            let got = xet(&x, &e, v, th);
            assert!(want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()), "x·E^T [{v}, {d}] threads {th}");
            checked += 1;
        }
    }
    println!("fast4 fmm: {checked} products bit for bit");
}
