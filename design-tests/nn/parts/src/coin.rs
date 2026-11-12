// TESTING / DESIGN ONLY
//! L2: update an old memory or open a new one (COIN, Heald, Lengyel & Wolpert 2021). The brain does not
//! overwrite what it knows each time something new comes along. It first asks which situation (context)
//! it is in. If the cues fit a known context, that context's memory is updated; if they fit none, a new
//! memory is opened and the old ones are left alone. This explains savings, less forgetting, and
//! "spontaneous recovery" (an old habit comes back when the situation is unclear).
//!
//! Each context k keeps a cue prototype μ_k, a count n_k and a memory m_k. For a cue c:
//!   P(k | c) ∝ (n_k + κ·[k was the last context]) · N(c; μ_k, σ²)      (known contexts; κ = stickiness)
//!   P(new | c) ∝ γ · N(c; 0, σ0² + σ²)                                  (a new context, γ = openness)
//! A new context is opened when "new" is the likeliest. Every context then learns in proportion to its
//! responsibility, so nothing is ever hard-assigned or banned. Recall mixes the memories by the same
//! posterior; the "new" share recalls nothing (a zero memory), so strange cues give a humble answer.
//! In the online learner a cue is who said it, when, and the topic; the memory is what was said.

#[derive(Clone, Debug)]
pub struct Context {
    pub mu: Vec<f32>,
    pub n: f32,
    pub memory: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Coin {
    pub contexts: Vec<Context>,
    pub gamma: f32,
    pub kappa: f32,
    pub sigma: f32,
    pub sigma0: f32,
    pub lr: f32,
    pub last: Option<usize>,
    pub d_mem: usize,
}

fn log_gauss(c: &[f32], mu: Option<&[f32]>, s: f32) -> f32 {
    let d2: f32 = match mu {
        Some(m) => c.iter().zip(m).map(|(a, b)| (a - b).powi(2)).sum(),
        None => c.iter().map(|a| a * a).sum(),
    };
    -d2 / (2.0 * s * s) - c.len() as f32 * s.ln()
}

impl Coin {
    pub fn new(d_mem: usize, gamma: f32, kappa: f32, sigma: f32, sigma0: f32, lr: f32) -> Coin {
        Coin { contexts: Vec::new(), gamma, kappa, sigma, sigma0, lr, last: None, d_mem }
    }

    /// P(context | cue) for each known context, then P(new) last.
    pub fn posterior(&self, cue: &[f32]) -> Vec<f32> {
        let mut lw: Vec<f32> = self
            .contexts
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let prior = c.n + if self.last == Some(k) { self.kappa } else { 0.0 };
                prior.max(1e-12).ln() + log_gauss(cue, Some(&c.mu), self.sigma)
            })
            .collect();
        let s_new = (self.sigma0 * self.sigma0 + self.sigma * self.sigma).sqrt();
        lw.push(if self.gamma > 0.0 || self.contexts.is_empty() { self.gamma.max(1e-12).ln() + log_gauss(cue, None, s_new) } else { f32::NEG_INFINITY });
        let m = lw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = lw.iter().map(|x| (x - m).exp()).collect();
        let s: f32 = e.iter().sum();
        e.iter().map(|x| x / s).collect()
    }

    /// What she recalls for a cue: the posterior mix of the memories ("new" recalls zeros).
    pub fn recall(&self, cue: &[f32]) -> Vec<f32> {
        let p = self.posterior(cue);
        let mut out = vec![0f32; self.d_mem];
        for (k, c) in self.contexts.iter().enumerate() {
            for (o, m) in out.iter_mut().zip(&c.memory) {
                *o += p[k] * m;
            }
        }
        out
    }

    /// One experience: a cue and what was true. Returns the context it was mostly given to, and
    /// whether that context is new.
    pub fn observe(&mut self, cue: &[f32], target: &[f32]) -> (usize, bool) {
        let mut p = self.posterior(cue);
        let k_new = self.contexts.len();
        let best = (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
        let opened = best == k_new;
        if opened {
            self.contexts.push(Context { mu: cue.to_vec(), n: 0.0, memory: vec![0.0; self.d_mem] });
        } else {
            // "New" did not win: its share goes back to the known contexts.
            p.pop();
            let s: f32 = p.iter().sum();
            p.iter_mut().for_each(|x| *x /= s);
        }
        for (k, c) in self.contexts.iter_mut().enumerate() {
            let r = p[k];
            if r < 1e-6 {
                continue;
            }
            c.n += r;
            for (m, x) in c.mu.iter_mut().zip(cue) {
                *m += r * (x - *m) / c.n;
            }
            for (m, t) in c.memory.iter_mut().zip(target) {
                *m += self.lr * r * (t - *m);
            }
        }
        self.last = Some(best);
        (best, opened)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn train(gamma: f32) -> Coin {
        let mut c = Coin::new(1, gamma, 5.0, 0.2, 2.0, 0.2);
        for t in 0..60 {
            let j = 0.02 * ((t % 5) as f32 - 2.0);
            c.observe(&[1.0 + j, j], &[1.0]);
        }
        for t in 0..20 {
            let j = 0.02 * ((t % 5) as f32 - 2.0);
            c.observe(&[j, 1.0 + j], &[-1.0]);
        }
        c
    }

    #[test]
    fn a_new_situation_opens_a_new_memory_and_the_old_one_survives() {
        let c = train(1.0);
        assert_eq!(c.contexts.len(), 2);
        let (a, b) = (c.recall(&[1.0, 0.0])[0], c.recall(&[0.0, 1.0])[0]);
        assert!(a > 0.95 && b < -0.95, "{a} {b}");
        // One memory for everything (never opening a new one) is overwritten by B.
        let one = train(0.0);
        assert_eq!(one.contexts.len(), 1);
        assert!(one.recall(&[1.0, 0.0])[0] < -0.9);
    }

    #[test]
    fn unclear_cues_bring_back_the_old_habit_and_strange_cues_recall_little() {
        let c = train(1.0);
        // Halfway between: the more practised context A dominates (spontaneous recovery).
        let mid = c.recall(&[0.5, 0.5])[0];
        assert!(mid > 0.2 && mid < 0.9, "{mid}");
        // Far from anything known: mostly "new", so she recalls almost nothing.
        let p = c.posterior(&[5.0, 5.0]);
        assert!(p[2] > 0.99 && c.recall(&[5.0, 5.0])[0].abs() < 0.01, "{p:?}");
    }
}
