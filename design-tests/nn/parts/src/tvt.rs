// TESTING / DESIGN ONLY
//! H9: credit across long gaps by recalling the episode that caused it (Temporal Value Transport, Hung
//! et al. 2019). Discounted rewards cannot reach back far: after 50 steps a reward is worth γ^50 of
//! itself and is lost in the noise of everything in between. People instead remember: "the door opened
//! because I took the key earlier". When a memory read at step t strongly recalls an earlier step τ, the
//! value at t is sent back to τ as extra reward:
//!   r'_τ = r_τ + α · Σ_{t : read strength_t ≥ threshold} w_t(τ) · V_{t+1}
//! In Ava the recalled step is a diary episode (the memory read is her retrieval), so a good outcome
//! credits the remembered choice directly, however long ago it was.

/// `reads[t]` lists (τ, weight) for what the memory read at step t recalled (τ < t). `v` has T + 1 values.
pub fn transport(r: &[f32], v: &[f32], reads: &[Vec<(usize, f32)>], alpha: f32, threshold: f32) -> Vec<f32> {
    let mut out = r.to_vec();
    for (t, read) in reads.iter().enumerate() {
        let strength: f32 = read.iter().map(|&(_, w)| w).sum();
        if strength < threshold {
            continue;
        }
        for &(tau, w) in read {
            if tau < t {
                out[tau] += alpha * w * v[t + 1];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::critic::gae;

    /// A key-and-door episode: the key is taken (or not) at step 3, the door at step 58 pays 1 only with
    /// the key, and steps in between hold unrelated rewards. The critic has learned values only near the
    /// door (0.9 once it sees the key in hand), as a real critic would early on.
    fn episode(key: bool, tvt: bool) -> f32 {
        let t_len = 60;
        let mut r: Vec<f32> = (0..t_len).map(|t| if (10..40).contains(&t) { 0.3 * (((t * 7919) % 13) as f32 / 6.0 - 1.0) } else { 0.0 }).collect();
        if key {
            r[58] = 1.0;
        }
        let v: Vec<f32> = (0..=t_len).map(|t| if key && (50..=58).contains(&t) { 0.9 } else { 0.0 }).collect();
        let mut reads = vec![Vec::new(); t_len];
        // At the door her memory read recalls the key step (and a little of another moment).
        reads[50] = vec![(3, 0.85), (20, 0.1)];
        let r = if tvt { transport(&r, &v, &reads, 1.0, 0.5) } else { r };
        gae(&r, &v, 0.96, 0.9)[3]
    }

    #[test]
    fn recalled_episodes_carry_credit_across_the_gap() {
        let plain = episode(true, false) - episode(false, false);
        let with = episode(true, true) - episode(false, true);
        eprintln!("credit to the key choice: without transport {plain:.3}, with {with:.3}");
        // Without transport the key choice gets almost nothing (0.96·0.9)^47 of the signal.
        assert!(plain.abs() < 0.02, "{plain}");
        assert!(with > 0.7, "{with}");
        // A weak read (below the threshold) transports nothing.
        let r = vec![0.0; 4];
        let v = vec![1.0; 5];
        assert_eq!(transport(&r, &v, &[vec![], vec![], vec![(0, 0.2)], vec![]], 1.0, 0.5), r);
    }
}
