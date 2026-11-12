// TESTING / DESIGN ONLY
//! H1: a learning rule that is itself learned. Evolution did not hand-write the brain's plasticity
//! rules; it tuned them. Here a synapse changes only from what it can see locally, with coefficients θ
//! that are found by an outer search (evolution strategies, `crate::es`), not set by hand:
//!   Δw_ij = A·pre_i·post_j + B·pre_i + C·post_j + D          (θ = [A, B, C, D])
//! This goes beyond B8 (which only scales an existing rule): the form of the rule is open, and the test
//! starts from "no learning at all" (θ = 0) and must discover a Hebbian rule (A > 0, the rest near 0) that
//! stores associations in one exposure each. Later the same outer search can tune a rule for her fast
//! weights or memory writes on real dev tasks.

use crate::es::Normal;

pub fn apply_rule(w: &mut [f32], pre: &[f32], post: &[f32], theta: &[f32]) {
    let n_out = post.len();
    for (i, &x) in pre.iter().enumerate() {
        for (j, &y) in post.iter().enumerate() {
            w[i * n_out + j] += theta[0] * x * y + theta[1] * x + theta[2] * y + theta[3];
        }
    }
}

/// One-shot associative memory: store `pairs` random ±1 key → value pairs with the rule, one exposure
/// each (the value clamped on the output, as a teacher would), then recall each value from its key.
/// Returns the mean cosine between recalled and true values over `tasks` fixed tasks (common random
/// numbers, so the outer search sees a smooth, repeatable score). 0 when nothing was stored.
pub fn association_score(theta: &[f32], n: usize, pairs: usize, tasks: usize, seed: u64) -> f32 {
    let mut total = 0.0;
    for task in 0..tasks {
        let mut g = Normal::new(seed + task as u64);
        let sign = |g: &mut Normal| if g.next() >= 0.0 { 1.0f32 } else { -1.0 };
        let keys: Vec<Vec<f32>> = (0..pairs).map(|_| (0..n).map(|_| sign(&mut g)).collect()).collect();
        let vals: Vec<Vec<f32>> = (0..pairs).map(|_| (0..n).map(|_| sign(&mut g)).collect()).collect();
        let mut w = vec![0f32; n * n];
        for (k, v) in keys.iter().zip(&vals) {
            apply_rule(&mut w, k, v, theta);
        }
        for (k, v) in keys.iter().zip(&vals) {
            let y: Vec<f32> = (0..n).map(|j| (0..n).map(|i| k[i] * w[i * n + j]).sum()).collect();
            let ny = y.iter().map(|a| a * a).sum::<f32>().sqrt();
            if ny > 1e-9 {
                total += y.iter().zip(v).map(|(a, b)| a * b).sum::<f32>() / (ny * (n as f32).sqrt());
            }
        }
    }
    total / (tasks * pairs) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::es::es_grad;

    #[test]
    fn evolution_discovers_a_hebbian_rule_from_nothing() {
        let f = |th: &[f32]| association_score(th, 16, 4, 4, 77);
        let mut theta = vec![0f32; 4];
        assert_eq!(f(&theta), 0.0);
        for it in 0..150 {
            let g = es_grad(&f, &theta, 0.1, 16, 1000 + it);
            theta.iter_mut().zip(&g).for_each(|(t, gi)| *t += 0.1 * gi);
        }
        let score = f(&theta);
        let others = theta[1].abs().max(theta[2].abs()).max(theta[3].abs());
        assert!(score > 0.85 && theta[0] > 3.0 * others, "score {score}, theta {theta:?}");
        // The hand-written Hebb rule scores the same: the search found it.
        assert!((association_score(&[1.0, 0.0, 0.0, 0.0], 16, 4, 4, 77) - score).abs() < 0.1);
    }
}
