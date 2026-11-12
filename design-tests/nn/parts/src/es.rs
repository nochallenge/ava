// TESTING / DESIGN ONLY
//! M7 evolution strategies (Salimans et al. 2017): learning without backpropagation. Try small random
//! changes, keep the direction of the ones that did better. It needs only the score, so it works for things
//! with no gradient: a whole conversation's outcome, a spike count, a hand-made reward. It runs in
//! parallel, and needs no memory for backward passes. The antithetic estimate:
//!   g ≈ 1/(2nσ) Σ_i (f(θ + σε_i) − f(θ − σε_i)) ε_i

/// A standard normal stream (Box-Muller on xorshift).
pub struct Normal(u64);

impl Normal {
    pub fn new(seed: u64) -> Normal {
        Normal(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn next(&mut self) -> f32 {
        ((-2.0 * self.u().ln()).sqrt() * (std::f64::consts::TAU * self.u()).cos()) as f32
    }
}

pub fn es_grad(f: &dyn Fn(&[f32]) -> f32, theta: &[f32], sigma: f32, pairs: usize, seed: u64) -> Vec<f32> {
    let mut g = vec![0f32; theta.len()];
    let mut nrm = Normal::new(seed);
    for _ in 0..pairs {
        let eps: Vec<f32> = (0..theta.len()).map(|_| nrm.next()).collect();
        let plus: Vec<f32> = theta.iter().zip(&eps).map(|(t, e)| t + sigma * e).collect();
        let minus: Vec<f32> = theta.iter().zip(&eps).map(|(t, e)| t - sigma * e).collect();
        let diff = f(&plus) - f(&minus);
        for (gi, e) in g.iter_mut().zip(&eps) {
            *gi += diff * e;
        }
    }
    let s = 1.0 / (2.0 * pairs as f32 * sigma);
    g.iter_mut().for_each(|x| *x *= s);
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn es_finds_the_gradient_and_descends() {
        let a: Vec<f32> = (0..10).map(|i| 1.0 + i as f32 * 0.3).collect();
        let f = |x: &[f32]| x.iter().zip(&a).map(|(x, a)| a * x * x).sum::<f32>();
        let mut th = vec![1.0f32; 10];
        let true_g: Vec<f32> = th.iter().zip(&a).map(|(x, a)| 2.0 * a * x).collect();
        let g = es_grad(&f, &th, 0.05, 400, 3);
        let cos = g.iter().zip(&true_g).map(|(p, q)| p * q).sum::<f32>() / (g.iter().map(|x| x * x).sum::<f32>().sqrt() * true_g.iter().map(|x| x * x).sum::<f32>().sqrt());
        assert!(cos > 0.9, "{cos}");
        let start = f(&th);
        for step in 0..100 {
            let g = es_grad(&f, &th, 0.05, 50, 100 + step);
            th.iter_mut().zip(&g).for_each(|(t, g)| *t -= 0.05 * g);
        }
        assert!(f(&th) < 0.1 * start, "{} -> {}", start, f(&th));
    }
}
