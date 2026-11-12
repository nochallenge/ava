// TESTING / DESIGN ONLY
//! A11 curiosity and B6 learning progress: choosing what to practise.
//!
//! Children practise what they are getting better at. They are bored by what they know, and give up on
//! noise they cannot learn (Oudeyer, Kaplan & Hafner 2007; Gottlieb et al. 2013). Seeking plain surprise
//! fails on "noisy TVs": random data stays surprising forever. Learning progress does not, because
//! noise never improves:
//!   LP(task) = |slow EMA of loss − fast EMA of loss|
//!
//! The WP17 lesson (v29, disclosed there): progress alone lost on the times table. It ignores how clearly
//! an example teaches. A 3-digit × 2-digit product splits one right/wrong over about 6 facts, while 1 × 1
//! teaches exactly one. So the pull divides by the credit spread (the mean number of pieces an example's
//! one score is shared by, e.g. facts or answer tokens, starting at 1), and an unseen task gets a novelty
//! bonus 1 / (1 + times practised):
//!   pull = (LP + 1 / (1 + n)) / spread
//!   P(task) = (1 − ε)·softmax(pull / temperature) + ε / tasks
//! Pre-registered here, before any run: on a dev curriculum, this rule must reach the same final accuracy
//! as a fixed order and need fewer examples on at least as many tasks as it needs more on. If not, it stays
//! an option and the fixed order stays the default (WP17's rule 1).

#[derive(Clone, Debug)]
pub struct Curriculum {
    pub fast: Vec<f64>,
    pub slow: Vec<f64>,
    pub spread: Vec<f64>,
    pub n: Vec<u64>,
    pub a_fast: f64,
    pub a_slow: f64,
}

impl Curriculum {
    pub fn new(tasks: usize) -> Curriculum {
        Curriculum { fast: vec![0.0; tasks], slow: vec![0.0; tasks], spread: vec![1.0; tasks], n: vec![0; tasks], a_fast: 0.1, a_slow: 0.01 }
    }

    /// One practised example of `task`: its loss, and how many pieces its score was shared by.
    /// Each average is a plain running mean until it has seen 1/a examples, then an EMA. Without that, the
    /// slow average sits near the first (noisy) loss for hundreds of examples, and a noisy task shows
    /// false progress.
    pub fn update(&mut self, task: usize, loss: f64, pieces: f64) {
        let first = 1.0 / (self.n[task] as f64 + 1.0);
        let (af, as_) = (self.a_fast.max(first), self.a_slow.max(first));
        self.fast[task] += af * (loss - self.fast[task]);
        self.slow[task] += as_ * (loss - self.slow[task]);
        self.spread[task] += as_ * (pieces.max(1.0) - self.spread[task]);
        self.n[task] += 1;
    }

    pub fn progress(&self, task: usize) -> f64 {
        (self.slow[task] - self.fast[task]).abs()
    }

    pub fn pull(&self, task: usize) -> f64 {
        (self.progress(task) + 1.0 / (1.0 + self.n[task] as f64)) / self.spread[task]
    }

    pub fn probs(&self, temperature: f64, eps: f64) -> Vec<f64> {
        let k = self.fast.len();
        let z: Vec<f64> = (0..k).map(|t| self.pull(t) / temperature).collect();
        let m = z.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|x| (1.0 - eps) * x / s + eps / k as f64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_what_is_improving_over_noise_and_what_is_known() {
        let mut c = Curriculum::new(3);
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut noise = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for t in 0..150 {
            c.update(0, 1.0 + 2.0 * (-(t as f64) / 100.0).exp(), 1.0); // learnable, still improving
            c.update(1, 0.5, 1.0); // already known
            c.update(2, 3.0 + noise(), 1.0); // noise: always surprising, never improving
        }
        assert!(c.progress(0) > 3.0 * c.progress(2) && c.progress(0) > 3.0 * c.progress(1), "{:?}", (c.progress(0), c.progress(1), c.progress(2)));
        let p = c.probs(0.1, 0.1);
        assert!(p[0] > p[1] && p[0] > p[2] && (p.iter().sum::<f64>() - 1.0).abs() < 1e-9, "{p:?}");
        // Plain surprise-seeking would pick the noise (the highest loss); this rule does not.
        assert!(c.fast[2] > c.fast[0]);
    }

    #[test]
    fn spread_credit_pulls_less_than_clear_credit() {
        // Two tasks improving equally fast; one shares each score over 6 facts, the other over 1.
        let mut c = Curriculum::new(2);
        for t in 0..150 {
            let loss = 1.0 + 2.0 * (-(t as f64) / 100.0).exp();
            c.update(0, loss, 6.0);
            c.update(1, loss, 1.0);
        }
        assert!((c.progress(0) - c.progress(1)).abs() < 1e-12);
        assert!(c.pull(1) > 5.0 * c.pull(0));
    }
}
