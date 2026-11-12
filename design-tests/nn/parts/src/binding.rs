// TESTING / DESIGN ONLY
//! Two ways to bind things together: "red" with "ball", or a role with its filler.
//!
//! D2 variable binding by holographic reduced representations (Plate 1995). A role and a filler (both
//! d-vectors) are bound by circular convolution, r ⊛ f, which gives another d-vector. Several bindings can
//! be added into one vector, and unbinding with the role's approximate inverse (its involution, r*[i] =
//! r[−i mod d]) gives back a noisy filler, which a clean-up memory (the nearest known filler) recognises.
//! Fixed-size vectors can thus hold "x = 3, y = 5" style structure; the fast-weight memory can store such
//! vectors.
//!
//! L5 binding by synchrony (the binding-by-synchrony idea of von der Malsburg and Singer; Kuramoto
//! oscillators; AKOrN, Miyato et al., ICLR 2025). Each unit has a phase. Units that should go together
//! are coupled and pull each other into step:
//!   dθ_i = ω_i + (K/N) Σ_j A_ij sin(θ_j − θ_i)
//! After some steps, the features of one object share a phase and those of another object a different
//! one. The phase-locking matrix cos(θ_i − θ_j) is then a grouping that downstream parts can read.

/// Circular convolution (d² operations; fine at these sizes).
pub fn bind(a: &[f32], b: &[f32]) -> Vec<f32> {
    let d = a.len();
    (0..d).map(|k| (0..d).map(|i| a[i] * b[(k + d - i) % d]).sum()).collect()
}

/// The approximate inverse: x*[i] = x[−i mod d].
pub fn involution(a: &[f32]) -> Vec<f32> {
    let d = a.len();
    (0..d).map(|i| a[(d - i) % d]).collect()
}

pub fn unbind(trace: &[f32], role: &[f32]) -> Vec<f32> {
    bind(trace, &involution(role))
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let n = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (n(a) * n(b)).max(1e-12)
}

/// Kuramoto steps (Euler). `theta` and `omega` [N]; `a` [N·N] coupling (row-major).
pub fn kuramoto(theta: &mut [f32], omega: &[f32], a: &[f32], k: f32, dt: f32, steps: usize) {
    let n = theta.len();
    for _ in 0..steps {
        let th = theta.to_vec();
        for i in 0..n {
            let pull: f32 = (0..n).map(|j| a[i * n + j] * (th[j] - th[i]).sin()).sum();
            theta[i] = th[i] + dt * (omega[i] + k / n as f32 * pull);
        }
    }
}

/// How in step a set of units is, 0 to 1 (the Kuramoto order parameter).
pub fn coherence(theta: &[f32], idx: &[usize]) -> f32 {
    let (c, s) = idx.iter().fold((0f32, 0f32), |(c, s), &i| (c + theta[i].cos(), s + theta[i].sin()));
    (c * c + s * s).sqrt() / idx.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vecs(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let all: Vec<f32> = crate::testkit::randn(&[n * d], seed).to_vec1().unwrap();
        all.chunks(d).map(|c| c.iter().map(|x| x / (d as f32).sqrt()).collect()).collect()
    }

    #[test]
    fn hrr_holds_three_bindings_in_one_vector() {
        let d = 512;
        let roles = vecs(3, d, 1);
        let fillers = vecs(10, d, 2);
        // x = filler 4, y = filler 7, z = filler 1, all added into one trace.
        let pairs = [(0, 4), (1, 7), (2, 1)];
        let mut trace = vec![0f32; d];
        for &(r, f) in &pairs {
            for (t, v) in trace.iter_mut().zip(bind(&roles[r], &fillers[f])) {
                *t += v;
            }
        }
        for &(r, f) in &pairs {
            let got = unbind(&trace, &roles[r]);
            let best = (0..10).max_by(|&a, &b| cosine(&got, &fillers[a]).total_cmp(&cosine(&got, &fillers[b]))).unwrap();
            assert_eq!(best, f);
            assert!(cosine(&got, &fillers[f]) > 0.35, "{}", cosine(&got, &fillers[f]));
        }
    }

    #[test]
    fn coupled_groups_fall_into_step_apart_from_each_other() {
        // 12 units, two objects of 6: strong coupling inside each object, none across.
        let n = 12;
        let mut a = vec![0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j && (i < 6) == (j < 6) {
                    a[i * n + j] = 1.0;
                }
            }
        }
        let mut theta: Vec<f32> = (0..n).map(|i| (i as f32 * 2.399) % 6.283).collect();
        let omega: Vec<f32> = (0..n).map(|i| if i < 6 { 1.0 } else { 1.3 }).collect();
        let g1: Vec<usize> = (0..6).collect();
        let g2: Vec<usize> = (6..12).collect();
        let before = (coherence(&theta, &g1), coherence(&theta, &g2));
        kuramoto(&mut theta, &omega, &a, 8.0, 0.05, 400);
        let after = (coherence(&theta, &g1), coherence(&theta, &g2));
        assert!(before.0 < 0.6 && before.1 < 0.6, "{before:?}");
        assert!(after.0 > 0.99 && after.1 > 0.99, "{after:?}");
        // The two objects keep separate rhythms: the gap between their phases keeps growing at the
        // difference of their frequencies (0.3 per time unit), so they are never locked together.
        let mean_phase = |th: &[f32], idx: &[usize]| {
            let (c, s) = idx.iter().fold((0f32, 0f32), |(c, s), &i| (c + th[i].cos(), s + th[i].sin()));
            s.atan2(c)
        };
        let gap0 = mean_phase(&theta, &g2) - mean_phase(&theta, &g1);
        kuramoto(&mut theta, &omega, &a, 8.0, 0.05, 100);
        let gap1 = mean_phase(&theta, &g2) - mean_phase(&theta, &g1);
        let drift = (gap1 - gap0).rem_euclid(std::f32::consts::TAU);
        assert!((drift - 1.5).abs() < 0.15, "{drift}");
    }
}
