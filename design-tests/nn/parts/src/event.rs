// TESTING / DESIGN ONLY
//! E2: event-driven compute. In a brain, a neuron that does not fire costs its targets nothing. For spike
//! input S [n, d_in] (mostly zeros) the product S · W is the sum of W's rows for the inputs that spiked,
//! weighted by the spike's value (1, or ±1/±2 for graded or ternary spikes). The work is
//! (number of spikes) × d_out synaptic operations instead of n × d_in × d_out multiply-adds.
//! The result is exactly the dense product, and the counts give C14's synaptic-op numbers.

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EventStats {
    /// Inputs that spiked (non-zero).
    pub events: usize,
    /// Row additions: events × d_out.
    pub synops: usize,
    /// What the dense product costs: n × d_in × d_out.
    pub dense_macs: usize,
}

/// y [n, d_out] = s [n, d_in] · w [d_in, d_out], touching only the non-zero inputs.
pub fn event_matmul(s: &[f32], w: &[f32], n: usize, d_in: usize, d_out: usize) -> (Vec<f32>, EventStats) {
    assert_eq!(s.len(), n * d_in);
    assert_eq!(w.len(), d_in * d_out);
    let mut y = vec![0f32; n * d_out];
    let mut st = EventStats { dense_macs: n * d_in * d_out, ..Default::default() };
    for r in 0..n {
        let out = &mut y[r * d_out..(r + 1) * d_out];
        for (i, &v) in s[r * d_in..(r + 1) * d_in].iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            st.events += 1;
            let row = &w[i * d_out..(i + 1) * d_out];
            if v == 1.0 {
                out.iter_mut().zip(row).for_each(|(o, &x)| *o += x);
            } else {
                out.iter_mut().zip(row).for_each(|(o, &x)| *o += v * x);
            }
        }
    }
    st.synops = st.events * d_out;
    (y, st)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equals_the_dense_product_at_a_fraction_of_the_work() {
        let (n, d_in, d_out) = (16, 128, 32);
        let w: Vec<f32> = crate::testkit::randn(&[d_in * d_out], 21).to_vec1().unwrap();
        // About 10% of inputs spike, some with graded values (ternary / int2 spikes).
        let r: Vec<f32> = crate::testkit::randn(&[n * d_in], 22).to_vec1().unwrap();
        let s: Vec<f32> = r.iter().map(|&x| if x > 1.9 { 2.0 } else if x > 1.28 { 1.0 } else if x < -1.9 { -1.0 } else { 0.0 }).collect();
        let (y, st) = event_matmul(&s, &w, n, d_in, d_out);
        for row in 0..n {
            for c in 0..d_out {
                let want: f32 = (0..d_in).map(|i| s[row * d_in + i] * w[i * d_out + c]).sum();
                assert!((y[row * d_out + c] - want).abs() < 1e-4);
            }
        }
        let frac = st.events as f64 / (n * d_in) as f64;
        assert!(frac > 0.05 && frac < 0.2, "{frac}");
        assert_eq!(st.synops * d_in, st.events * st.dense_macs / n);
    }
}
