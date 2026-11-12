// TESTING / DESIGN ONLY
//! Growing her brain without losing what she knows, and giving dead neurons a new start.
//!
//! A15 growth (Net2Net, Chen, Goodfellow & Shlens, ICLR 2016): widen a block y = act(x·W1 + b1)·W2 from h to
//! h2 hidden units, so that it computes exactly the same function. Each new unit copies a random existing
//! unit's incoming weights. Every unit that now has c copies (itself included) has its outgoing row divided
//! by c. Training then breaks the symmetry. A new residual block whose output weights start at 0 is
//! already exact, so deepening needs nothing more.
//!
//! I3 rebirth (ReDo, Sokar et al., ICML 2023): a unit whose activity has fallen to almost nothing is
//! dormant: it no longer learns and wastes capacity, a known cause of lost plasticity in long training.
//! Dormant units get fresh random incoming weights and zero outgoing weights. The block's output does not
//! change (they were silent), but they can learn again.

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

/// (w1 [d, h2], b1 [h2], w2 [h2, out]) for w1 [d, h], b1 [h], w2 [h, out].
pub fn widen(w1: &[f32], b1: &[f32], w2: &[f32], d: usize, h: usize, out: usize, h2: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(h2 >= h);
    let mut next = rng(seed);
    let src: Vec<usize> = (0..h2).map(|j| if j < h { j } else { (next() % h as u64) as usize }).collect();
    let mut copies = vec![0usize; h];
    for &s in &src {
        copies[s] += 1;
    }
    let mut nw1 = vec![0f32; d * h2];
    for r in 0..d {
        for j in 0..h2 {
            nw1[r * h2 + j] = w1[r * h + src[j]];
        }
    }
    let nb1 = src.iter().map(|&s| b1[s]).collect();
    let mut nw2 = vec![0f32; h2 * out];
    for j in 0..h2 {
        for c in 0..out {
            nw2[j * out + c] = w2[src[j] * out + c] / copies[src[j]] as f32;
        }
    }
    (nw1, nb1, nw2)
}

/// Units whose mean |activation| is at most `tau` times the layer's mean (ReDo's score), for acts [n, h].
pub fn dormant(acts: &[f32], n: usize, h: usize, tau: f32) -> Vec<usize> {
    let mut m = vec![0f32; h];
    for r in 0..n {
        for j in 0..h {
            m[j] += acts[r * h + j].abs() / n as f32;
        }
    }
    let avg = m.iter().sum::<f32>() / h as f32;
    (0..h).filter(|&j| m[j] <= tau * avg).collect()
}

/// Re-initialises the given units: incoming weights random (std 1/sqrt(d)), bias 0, outgoing weights 0.
pub fn redo(w1: &mut [f32], b1: &mut [f32], w2: &mut [f32], d: usize, h: usize, out: usize, units: &[usize], seed: u64) {
    let mut next = rng(seed);
    let std = 1.0 / (d as f32).sqrt();
    for &j in units {
        for r in 0..d {
            let u1 = (next() >> 11) as f32 / (1u64 << 53) as f32 + 1e-12;
            let u2 = (next() >> 11) as f32 / (1u64 << 53) as f32;
            w1[r * h + j] = std * (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        }
        b1[j] = 0.0;
        for c in 0..out {
            w2[j * out + c] = 0.0;
        }
    }
}

/// y = relu(x·W1 + b1)·W2, and the hidden activity, for x [n, d].
pub fn block(x: &[f32], w1: &[f32], b1: &[f32], w2: &[f32], n: usize, d: usize, h: usize, out: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = vec![0f32; n * h];
    for r in 0..n {
        for j in 0..h {
            let s: f32 = (0..d).map(|k| x[r * d + k] * w1[k * h + j]).sum::<f32>() + b1[j];
            a[r * h + j] = s.max(0.0);
        }
    }
    let mut y = vec![0f32; n * out];
    for r in 0..n {
        for c in 0..out {
            y[r * out + c] = (0..h).map(|j| a[r * h + j] * w2[j * out + c]).sum();
        }
    }
    (y, a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        crate::testkit::randn(&[n], seed).to_vec1().unwrap()
    }

    #[test]
    fn widening_keeps_the_function_exactly() {
        let (n, d, h, out, h2) = (32, 6, 8, 5, 20);
        let (w1, b1, w2, x) = (rnd(d * h, 1), rnd(h, 2), rnd(h * out, 3), rnd(n * d, 4));
        let (y, _) = block(&x, &w1, &b1, &w2, n, d, h, out);
        let (nw1, nb1, nw2) = widen(&w1, &b1, &w2, d, h, out, h2, 5);
        let (y2, _) = block(&x, &nw1, &nb1, &nw2, n, d, h2, out);
        for (a, b) in y.iter().zip(&y2) {
            assert!((a - b).abs() < 1e-5, "{a} {b}");
        }
    }

    #[test]
    fn dormant_units_are_found_and_reborn_without_changing_the_output() {
        let (n, d, h, out) = (64, 6, 8, 4);
        let (mut w1, mut b1, mut w2, x) = (rnd(d * h, 11), rnd(h, 12), rnd(h * out, 13), rnd(n * d, 14));
        // Units 2 and 5 are dead: a big negative bias keeps them silent.
        b1[2] = -100.0;
        b1[5] = -100.0;
        let (y, a) = block(&x, &w1, &b1, &w2, n, d, h, out);
        let dead = dormant(&a, n, h, 0.025);
        assert_eq!(dead, vec![2, 5]);
        redo(&mut w1, &mut b1, &mut w2, d, h, out, &dead, 15);
        let (y2, a2) = block(&x, &w1, &b1, &w2, n, d, h, out);
        for (p, q) in y.iter().zip(&y2) {
            assert!((p - q).abs() < 1e-5);
        }
        // They are alive again.
        assert!(dormant(&a2, n, h, 0.025).is_empty());
    }
}
