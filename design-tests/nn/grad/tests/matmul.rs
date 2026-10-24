// TESTING / DESIGN ONLY — tests of the raw GEMM kernels in ava_grad::matmul: agreement with a naive
// f64 reference over odd shapes and all four transpose combinations, accumulation semantics (C += A·B),
// batched GEMM, and bit-exact determinism across thread counts.

use ava_grad::matmul::{gemm, gemm_batched};
use ava_grad::{Rng, Tensor};

/// Naive reference: C[m,n] = op(A)[m,k] · op(B)[k,n] in f64.
#[allow(clippy::too_many_arguments)]
fn naive(m: usize, k: usize, n: usize, a: &[f32], ta: bool, b: &[f32], tb: bool) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for kk in 0..k {
                let av = if ta { a[kk * m + i] } else { a[i * k + kk] };
                let bv = if tb { b[j * k + kk] } else { b[kk * n + j] };
                s += av as f64 * bv as f64;
            }
            c[i * n + j] = s;
        }
    }
    c
}

fn rand(n: usize, rng: &mut Rng) -> Vec<f32> {
    Tensor::randn(&[n], 1.0, rng).data
}

#[test]
fn gemm_matches_naive_over_odd_shapes_and_transposes() {
    let mut rng = Rng::new(42);
    let shapes = [
        (1, 1, 1),
        (1, 7, 1),
        (3, 5, 4),
        (4, 16, 16),
        (5, 17, 33),
        (17, 33, 19),
        (64, 300, 31),
        (33, 257, 65),
        (130, 70, 96),
    ];
    for &(m, k, n) in &shapes {
        for (ta, tb) in [(false, false), (false, true), (true, false), (true, true)] {
            let a = rand(m * k, &mut rng);
            let b = rand(k * n, &mut rng);
            let want = naive(m, k, n, &a, ta, &b, tb);
            for threads in [1, 3, 8] {
                let mut c = vec![0.0f32; m * n];
                gemm(m, k, n, &a, ta, &b, tb, &mut c, threads);
                for idx in 0..m * n {
                    let tol = 4e-5 * (k as f64).sqrt(); // typical f32 error is ~1e-6·sqrt(k)·sqrt(k)
                    assert!(
                        (c[idx] as f64 - want[idx]).abs() < tol,
                        "({m},{k},{n}) ta={ta} tb={tb} t={threads} idx {idx}: {} vs {}",
                        c[idx],
                        want[idx]
                    );
                }
            }
        }
    }
}

#[test]
fn gemm_accumulates_into_c() {
    let mut rng = Rng::new(7);
    let (m, k, n) = (9, 13, 21);
    let a = rand(m * k, &mut rng);
    let b = rand(k * n, &mut rng);
    let start = rand(m * n, &mut rng);
    let mut c = start.clone();
    gemm(m, k, n, &a, false, &b, false, &mut c, 2);
    let prod = naive(m, k, n, &a, false, &b, false);
    for i in 0..m * n {
        assert!((c[i] as f64 - (start[i] as f64 + prod[i])).abs() < 1e-4);
    }
}

#[test]
fn gemm_handles_degenerate_sizes() {
    let mut c = vec![1.0f32; 6];
    gemm(2, 0, 3, &[], false, &[], false, &mut c, 4); // k = 0 adds nothing
    assert_eq!(c, vec![1.0; 6]);
    let mut empty: Vec<f32> = vec![];
    gemm(0, 4, 3, &[], false, &[0.0; 12], false, &mut empty, 4);
}

#[test]
fn gemm_batched_matches_per_batch() {
    let mut rng = Rng::new(3);
    for &(batch, m, k, n) in &[(1, 5, 6, 7), (3, 17, 9, 20), (12, 8, 32, 8), (5, 64, 40, 33)] {
        for (ta, tb) in [(false, false), (false, true), (true, false)] {
            let a = rand(batch * m * k, &mut rng);
            let b = rand(batch * k * n, &mut rng);
            for threads in [1, 4, 8] {
                let mut c = vec![0.0f32; batch * m * n];
                gemm_batched(batch, m, k, n, &a, ta, &b, tb, &mut c, threads);
                for bi in 0..batch {
                    let want = naive(m, k, n, &a[bi * m * k..(bi + 1) * m * k], ta, &b[bi * k * n..(bi + 1) * k * n], tb);
                    for idx in 0..m * n {
                        assert!((c[bi * m * n + idx] as f64 - want[idx]).abs() < 1e-3);
                    }
                }
            }
        }
    }
}

#[test]
fn gemm_is_bit_identical_for_any_thread_count() {
    let mut rng = Rng::new(11);
    let (m, k, n) = (301, 517, 129); // big enough that every thread count really splits the rows
    let a = rand(m * k, &mut rng);
    let b = rand(k * n, &mut rng);
    for (ta, tb) in [(false, false), (false, true), (true, false)] {
        let mut base = vec![0.0f32; m * n];
        gemm(m, k, n, &a, ta, &b, tb, &mut base, 1);
        for threads in [2, 3, 4, 5, 7, 8] {
            let mut c = vec![0.0f32; m * n];
            gemm(m, k, n, &a, ta, &b, tb, &mut c, threads);
            assert!(c == base, "ta={ta} tb={tb}: {threads} threads differ from 1 thread");
        }
    }
    // batched, both splitting strategies (batch >= threads and batch < threads)
    for batch in [2usize, 16] {
        let (m, k, n) = (70, 90, 50);
        let a = rand(batch * m * k, &mut rng);
        let b = rand(batch * k * n, &mut rng);
        let mut base = vec![0.0f32; batch * m * n];
        gemm_batched(batch, m, k, n, &a, false, &b, true, &mut base, 1);
        for threads in [2, 3, 8] {
            let mut c = vec![0.0f32; batch * m * n];
            gemm_batched(batch, m, k, n, &a, false, &b, true, &mut c, threads);
            assert!(c == base, "batched {batch}: {threads} threads differ");
        }
    }
}

#[test]
fn threads_env_default_is_clamped() {
    let t = ava_grad::threads();
    assert!((1..=8).contains(&t));
}
