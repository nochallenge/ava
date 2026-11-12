// TESTING / DESIGN ONLY
//! Structural plasticity and its timing.
//!
//! H6 rewiring (DEEP R, Bellec et al., ICLR 2018; like synapse turnover in cortex): a layer keeps a fixed
//! number of connections. After each update the weakest live ones are pruned, and the same number of new
//! ones grow at random empty places, starting at 0. The wiring itself is then learned, at a fixed cost.
//!
//! H7 critical periods (Hensch 2005): plasticity closes in a sequence. Early sensory layers settle first
//! and later layers stay plastic longer, so the low-level code stops shifting under the higher layers
//! once they depend on it. A learning-rate multiplier per layer:
//!   m(l, step) = 1 before close(l), then floor + (1 − floor)·exp(−(step − close(l)) / tau)
//!   close(l) = first + l·gap

/// Prunes the `frac` weakest live connections and grows as many at random dead places (value 0).
/// Returns (pruned, grown). `mask` marks live connections; dead weights are kept at exactly 0.
pub fn rewire(w: &mut [f32], mask: &mut [bool], frac: f32, seed: u64) -> (usize, usize) {
    let live: Vec<usize> = (0..w.len()).filter(|&i| mask[i]).collect();
    let n = ((live.len() as f32) * frac).round() as usize;
    let mut by_mag = live.clone();
    by_mag.sort_by(|&a, &b| w[a].abs().total_cmp(&w[b].abs()).then(a.cmp(&b)));
    let mut just_pruned = vec![false; w.len()];
    for &i in &by_mag[..n] {
        mask[i] = false;
        w[i] = 0.0;
        just_pruned[i] = true;
    }
    // New connections grow where there was none (not straight back where one was just pruned).
    let mut dead: Vec<usize> = (0..w.len()).filter(|&i| !mask[i] && !just_pruned[i]).collect();
    let mut s = seed | 1;
    let mut grown = 0;
    while grown < n && !dead.is_empty() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let i = dead.swap_remove((s % dead.len() as u64) as usize);
        mask[i] = true;
        w[i] = 0.0;
        grown += 1;
    }
    (n, grown)
}

#[derive(Clone, Copy, Debug)]
pub struct CriticalPeriods {
    pub first: f64,
    pub gap: f64,
    pub tau: f64,
    pub floor: f64,
}

impl CriticalPeriods {
    pub fn multiplier(&self, layer: usize, step: f64) -> f64 {
        let close = self.first + layer as f64 * self.gap;
        if step <= close {
            1.0
        } else {
            self.floor + (1.0 - self.floor) * (-(step - close) / self.tau).exp()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewiring_keeps_the_count_and_drops_the_weakest() {
        let mut w: Vec<f32> = (0..20).map(|i| if i < 10 { (i as f32 + 1.0) * 0.1 } else { 0.0 }).collect();
        let mut mask: Vec<bool> = (0..20).map(|i| i < 10).collect();
        let (p, g) = rewire(&mut w, &mut mask, 0.3, 7);
        assert_eq!((p, g), (3, 3));
        assert_eq!(mask.iter().filter(|&&m| m).count(), 10);
        // The three weakest (0.1, 0.2, 0.3) are gone; the strong ones stay; new ones start at 0.
        assert!(!mask[0] && !mask[1] && !mask[2] && mask[3..10].iter().all(|&m| m));
        assert!((10..20).filter(|&i| mask[i]).all(|i| w[i] == 0.0));
        assert!(w.iter().zip(&mask).all(|(x, m)| *m || *x == 0.0));
    }

    #[test]
    fn lower_layers_close_first() {
        let cp = CriticalPeriods { first: 1000.0, gap: 1000.0, tau: 500.0, floor: 0.1 };
        assert_eq!(cp.multiplier(0, 900.0), 1.0);
        let (l0, l3) = (cp.multiplier(0, 3000.0), cp.multiplier(3, 3000.0));
        assert!(l0 < 0.2 && l3 == 1.0, "{l0} {l3}");
        assert!((cp.multiplier(0, 1e9) - 0.1).abs() < 1e-9);
    }
}
