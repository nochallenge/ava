// TESTING / DESIGN ONLY — stats (v42): honest daily gates for an always-online Ava.
//
// v42's background learner asks the same dev gate every day whether the new copy is better. A gate that is asked
// hundreds of times and whose answers steer what the learner keeps stops being a fair test: the learner slowly fits
// the gate's own items (adaptive overfitting), and noise starts to pass. Three tools:
//
// THRESHOLDOUT (Dwork, Feldman, Hardt, Pitassi, Reingold, Roth, "The reusable holdout", Science 2015). Each query
// gives a value measured on the learner's own data (train) and on the gate's items (holdout). When the two agree
// within a noisy threshold, the gate answers with the train value and the holdout reveals nothing. When they disagree
// (the train value is overfitted), it answers with the holdout value plus Laplace noise and spends one unit of
// budget. When the budget is spent, the holdout is used up: retire it and draw fresh items.
//
// THE LEDGER (data/nn/gate_ledger.tsv, private, append-only): one line each time a set is used by a gate, keyed by
// the set file's sha256, so every report can say how many times its gate set had been used before.
//
// MINIMUM DETECTABLE EFFECT for paired right/wrong outcomes (McNemar's test in its normal form, Connor 1987): the
// smallest accuracy gain a gate on n items can find with the given power, to be computed and written down before a
// test is pre-registered. Only discordant items (one model right, the other wrong) carry information.

use ava_data::rng::SplitMix64;
use std::path::Path;

/// The gate ledger's default place (git-ignored data folder; never published).
pub const DEFAULT_LEDGER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/nn/gate_ledger.tsv");
pub const LEDGER_HEADER: &str = "time\tset\tsha256\tuse\twhat";

// ------------------------------------------------------------------------------------------------ normal law

/// The standard normal quantile (inverse CDF) for 0 < p < 1, by Acklam's rational approximation (relative error
/// below 1.2e-9).
pub fn normal_quantile(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "normal_quantile needs 0 < p < 1, got {p}");
    const A: [f64; 6] = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const B: [f64; 5] = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const C: [f64; 6] = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const D: [f64; 4] = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const P_LOW: f64 = 0.02425;
    let tail = |q: f64| (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    if p < P_LOW {
        tail((-2.0 * p.ln()).sqrt())
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    }
}

/// The standard normal CDF (erfc by the Numerical Recipes Chebyshev fit, fractional error below 1.2e-7).
pub fn normal_cdf(x: f64) -> f64 {
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.26551223
        + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let erfc = t * poly.exp();
    if x >= 0.0 {
        1.0 - 0.5 * erfc
    } else {
        0.5 * erfc
    }
}

// ------------------------------------------------------------------------------------------------ paired outcomes

/// McNemar's z for paired right/wrong outcomes: `gain` items only the candidate gets right, `loss` items only the
/// current model gets right. 0 when there are no discordant items.
pub fn mcnemar_z(gain: usize, loss: usize) -> f64 {
    let n = (gain + loss) as f64;
    if n == 0.0 {
        0.0
    } else {
        (gain as f64 - loss as f64) / n.sqrt()
    }
}

/// The minimum detectable effect: the smallest accuracy gain δ (candidate − current, as a share of all n items)
/// that a one-sided McNemar test at level `alpha` detects with probability `power`, when a share `p_disc` of the items
/// is discordant (from earlier paired runs). Solves δ√n = z(1−α)·√p_d + z(power)·√(p_d − δ²) (Connor 1987) by
/// bisection. None when even δ = p_disc (every discordant item a gain) cannot be detected. For a two-sided test pass
/// alpha / 2.
pub fn mde_paired(n: usize, p_disc: f64, alpha: f64, power: f64) -> Option<f64> {
    assert!(n > 0 && p_disc > 0.0 && p_disc <= 1.0, "mde_paired: n > 0 and 0 < p_disc <= 1");
    let (za, zb) = (normal_quantile(1.0 - alpha), normal_quantile(power));
    let rn = (n as f64).sqrt();
    let f = |d: f64| d * rn - za * p_disc.sqrt() - zb * (p_disc - d * d).max(0.0).sqrt();
    if f(p_disc) < 0.0 {
        return None;
    }
    let (mut lo, mut hi) = (0.0f64, p_disc);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// The items needed to detect a gain `delta` (same model as `mde_paired`): the smallest n whose MDE is <= delta.
pub fn n_for_effect(delta: f64, p_disc: f64, alpha: f64, power: f64) -> usize {
    assert!(delta > 0.0 && delta <= p_disc, "n_for_effect: 0 < delta <= p_disc");
    let (za, zb) = (normal_quantile(1.0 - alpha), normal_quantile(power));
    let n = ((za * p_disc.sqrt() + zb * (p_disc - delta * delta).sqrt()) / delta).powi(2);
    n.ceil().max(1.0) as usize
}

// ------------------------------------------------------------------------------------------------ Thresholdout

/// A Laplace(0, b) draw (b = scale).
pub fn laplace(rng: &mut SplitMix64, b: f64) -> f64 {
    // u uniform in (-1/2, 1/2), never exactly ±1/2
    let u = ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64 - 0.5;
    -b * u.signum() * (1.0 - 2.0 * u.abs()).ln()
}

/// The reusable holdout. `threshold` (T) is how far train and holdout may differ before the holdout is revealed,
/// `sigma` the noise scale, `budget` how many times it may be revealed in all.
#[derive(Clone, Debug)]
pub struct Thresholdout {
    pub threshold: f64,
    pub sigma: f64,
    /// reveals left
    pub budget: usize,
    /// queries answered so far
    pub queries: usize,
    t_hat: f64,
    rng: SplitMix64,
}

impl Thresholdout {
    pub fn new(threshold: f64, sigma: f64, budget: usize, seed: u64) -> Thresholdout {
        let mut rng = SplitMix64::new(seed);
        let t_hat = threshold + laplace(&mut rng, 2.0 * sigma);
        Thresholdout { threshold, sigma, budget, queries: 0, t_hat, rng }
    }

    /// Settings for statistics that are means over n items of per-item values in [−1, 1] (right − wrong of a paired
    /// comparison, a correlation), with n items on each side: T = 3·√(2/n), three standard deviations of a train −
    /// holdout difference, and σ = 1/(8√n). Measured in the tests (n 1000, 500 adaptive queries on pure noise, then a
    /// one-sided 5% gate): false accepts 1.00 with the plain reused holdout, 0.05–0.09 with T of 2.7–3.6 SD and this
    /// σ, but 0.37–0.41 when σ is near T/4 (the noisy threshold then reveals the holdout on ~100 of 500 queries).
    pub fn for_pm1_mean(n: usize, budget: usize, seed: u64) -> Thresholdout {
        let n = n.max(1) as f64;
        Thresholdout::new(3.0 * (2.0 / n).sqrt(), 1.0 / (8.0 * n.sqrt()), budget, seed)
    }

    /// Answer one query from its train-side value and its holdout value: the train value when they agree within the
    /// noisy threshold, else the holdout value plus Laplace(sigma) noise (one unit of budget spent). None once the
    /// budget is spent: the holdout is used up and must be retired.
    pub fn query(&mut self, train: f64, holdout: f64) -> Option<f64> {
        if self.budget == 0 {
            return None;
        }
        self.queries += 1;
        let eta = laplace(&mut self.rng, 4.0 * self.sigma);
        if (holdout - train).abs() > self.t_hat + eta {
            self.budget -= 1;
            self.t_hat = self.threshold + laplace(&mut self.rng, 2.0 * self.sigma);
            Some(holdout + laplace(&mut self.rng, self.sigma))
        } else {
            Some(train)
        }
    }
}

// ------------------------------------------------------------------------------------------------ the ledger

/// How many times the set with this sha256 has been used by a gate (0 if never; a missing ledger is empty).
pub fn ledger_uses(ledger: &Path, sha256: &str) -> Result<usize, String> {
    let (lines, _) = crate::diary::read_text_lines(ledger, LEDGER_HEADER)?;
    Ok(lines.iter().filter(|l| l.split('\t').nth(2) == Some(sha256)).count())
}

/// Record one use of a set (its name, its file's sha256, what used it) and return its use number (1 = first use).
pub fn ledger_record(ledger: &Path, set: &str, sha256: &str, what: &str) -> Result<usize, String> {
    let n = ledger_uses(ledger, sha256)? + 1;
    let (_, time) = crate::diary::local_now();
    let rec = format!("{}\t{}\t{sha256}\t{n}\t{}", crate::diary::escape(&time), crate::diary::escape(set), crate::diary::escape(what));
    crate::diary::append_record(ledger, LEDGER_HEADER, &rec)?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_law_matches_known_values() {
        for (p, z) in [(0.975, 1.959963984540054), (0.95, 1.6448536269514722), (0.8, 0.8416212335729143), (0.5, 0.0), (0.01, -2.3263478740408408), (1e-6, -4.753424308822899)] {
            assert!((normal_quantile(p) - z).abs() < 1e-8, "q({p}) = {} want {z}", normal_quantile(p));
            assert!((normal_cdf(z) - p).abs() < 1e-7 * p.max(1e-3) + 1e-9, "cdf({z}) = {} want {p}", normal_cdf(z));
        }
        assert!((normal_cdf(-1.0) + normal_cdf(1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_mde_has_the_power_it_claims() {
        // n 500 items, 20% discordant, one-sided 5%, power 80%
        let (n, pd, alpha, power) = (500usize, 0.2, 0.05, 0.8);
        let d = mde_paired(n, pd, alpha, power).unwrap();
        // the closed form for n agrees with the bisection
        assert!(n_for_effect(d * 1.000_01, pd, alpha, power) <= n && n_for_effect(d * 0.98, pd, alpha, power) > n);
        // simulate: gain with probability (pd + d)/2, loss with (pd - d)/2, else concordant
        let mut rng = SplitMix64::new(42);
        let za = normal_quantile(1.0 - alpha);
        let trials = 4000;
        let mut hits = 0;
        for _ in 0..trials {
            let (mut g, mut l) = (0, 0);
            for _ in 0..n {
                let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                if u < (pd + d) / 2.0 {
                    g += 1;
                } else if u < pd {
                    l += 1;
                }
            }
            hits += usize::from(mcnemar_z(g, l) > za);
        }
        let got = hits as f64 / trials as f64;
        assert!((got - power).abs() < 0.035, "power {got} at mde {d}");
        // and at no effect the false-accept rate is about alpha
        let mut fa = 0;
        for _ in 0..trials {
            let (mut g, mut l) = (0, 0);
            for _ in 0..n {
                let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                if u < pd / 2.0 {
                    g += 1;
                } else if u < pd {
                    l += 1;
                }
            }
            fa += usize::from(mcnemar_z(g, l) > za);
        }
        assert!((fa as f64 / trials as f64 - alpha).abs() < 0.02);
        // more items or more power move the MDE the right way; too few items detect nothing
        assert!(mde_paired(2000, pd, alpha, power).unwrap() < d && mde_paired(n, pd, alpha, 0.95).unwrap() > d);
        assert_eq!(mde_paired(2, 0.05, 0.05, 0.8), None);
    }

    #[test]
    fn laplace_draws_have_the_right_spread() {
        let mut rng = SplitMix64::new(9);
        let xs: Vec<f64> = (0..200_000).map(|_| laplace(&mut rng, 0.5)).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let mad = xs.iter().map(|x| x.abs()).sum::<f64>() / xs.len() as f64; // E|X| = b
        assert!(mean.abs() < 0.01 && (mad - 0.5).abs() < 0.01, "mean {mean} mean |x| {mad}");
    }

    /// One trial of Dwork et al.'s experiment, recast as a daily gate on pure noise. The learner has its own items
    /// (train) and the gate has its items (holdout); n of each, labels y = ±1. It proposes d changes, each a random ±1
    /// per item (a change "helps" item i when x_i = y_i), so no change truly helps: every acceptance is false. For each
    /// change it asks the gate for its effect (the mean of x·y) and keeps those that look good on its own items and on
    /// the gate's answer (same sign, both above 1/√n). It then combines the best k kept changes by a vote and asks the
    /// gate once more, on the same scale (the mean of right − wrong, 2·accuracy − 1): accept if it beats 0 by more than
    /// a fresh one-sided 5% test would need (1.645/√n). Returns whether the gate accepted.
    fn gate_trial(seed: u64, n: usize, d: usize, k: usize, mut gate: impl FnMut(f64, f64) -> Option<f64>) -> bool {
        let mut rng = SplitMix64::new(seed);
        let pm = |rng: &mut SplitMix64| if rng.next_u64() >> 63 == 0 { 1i8 } else { -1i8 };
        let yt: Vec<i8> = (0..n).map(|_| pm(&mut rng)).collect();
        let yh: Vec<i8> = (0..n).map(|_| pm(&mut rng)).collect();
        let xt: Vec<Vec<i8>> = (0..d).map(|_| (0..n).map(|_| pm(&mut rng)).collect()).collect();
        let xh: Vec<Vec<i8>> = (0..d).map(|_| (0..n).map(|_| pm(&mut rng)).collect()).collect();
        let corr = |x: &[i8], y: &[i8]| x.iter().zip(y).map(|(&a, &b)| (a * b) as f64).sum::<f64>() / n as f64;
        let tau = 1.0 / (n as f64).sqrt();
        let mut kept: Vec<(f64, usize, f64)> = Vec::new(); // (|train effect|, change, sign)
        for j in 0..d {
            let ct = corr(&xt[j], &yt);
            let Some(ans) = gate(ct, corr(&xh[j], &yh)) else { break };
            if ct.abs() > tau && ans.abs() > tau && ct.signum() == ans.signum() {
                kept.push((ct.abs(), j, ct.signum()));
            }
        }
        kept.sort_by(|a, b| b.0.total_cmp(&a.0));
        kept.truncate(k);
        // right − wrong per item, averaged: 2·accuracy − 1
        let adv = |x: &[Vec<i8>], y: &[i8]| {
            (0..n)
                .map(|i| {
                    let s: f64 = kept.iter().map(|&(_, j, sg)| sg * x[j][i] as f64).sum();
                    if (if s >= 0.0 { 1 } else { -1 }) == y[i] {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .sum::<f64>()
                / n as f64
        };
        let bar = normal_quantile(0.95) / (n as f64).sqrt();
        match gate(adv(&xt, &yt), adv(&xh, &yh)) {
            Some(a) => a > bar,
            None => false, // budget spent: the gate refuses to answer, so nothing passes
        }
    }

    #[test]
    fn thresholdout_cuts_false_accepts_when_a_gate_is_reused_on_noise() {
        let (n, d, k, trials) = (1000usize, 500usize, 40usize, 100u64);
        let budget = 1000;
        let mut naive = 0usize;
        for s in 0..trials {
            naive += usize::from(gate_trial(1000 + s, n, d, k, |_, h| Some(h)));
        }
        let fa_naive = naive as f64 / trials as f64;
        // (false accept rate, reveals per trial) with a given Thresholdout per trial
        let guarded = |make: &dyn Fn(u64) -> Thresholdout| {
            let (mut acc, mut rev) = (0usize, 0usize);
            for s in 0..trials {
                let mut to = make(77 + s);
                acc += usize::from(gate_trial(1000 + s, n, d, k, |tr, h| to.query(tr, h)));
                rev += budget - to.budget;
            }
            (acc as f64 / trials as f64, rev as f64 / trials as f64)
        };
        let (fa_to, rev) = guarded(&|seed| Thresholdout::for_pm1_mean(n, budget, seed));
        let (fa_noisy, rev_noisy) = guarded(&|seed| Thresholdout::new(0.09, 0.02, budget, seed));
        eprintln!(
            "false accepts on pure noise after {d} reuses: plain holdout {fa_naive:.2}; Thresholdout {fa_to:.2} ({rev:.1} reveals); \
             too noisy (T 0.09, sigma 0.02) {fa_noisy:.2} ({rev_noisy:.1} reveals); a fresh test: 0.05"
        );
        assert!(fa_naive >= 0.9, "the plain reused gate should pass most noise ({fa_naive})");
        assert!(fa_to <= 0.12, "Thresholdout should keep false accepts near 5% ({fa_to})");
    }

    #[test]
    fn thresholdout_answers_from_train_when_they_agree_and_stops_when_spent() {
        let mut to = Thresholdout::new(0.1, 1e-9, 2, 1);
        assert_eq!(to.query(0.50, 0.52), Some(0.50)); // agree: the holdout says nothing
        let a = to.query(0.90, 0.50).unwrap(); // overfitted train value: the holdout (plus tiny noise)
        assert!((a - 0.50).abs() < 1e-6);
        assert_eq!(to.budget, 1);
        assert!(to.query(0.10, 0.60).is_some());
        assert_eq!((to.budget, to.query(0.5, 0.5)), (0, None));
        assert_eq!(to.queries, 3);
    }

    #[test]
    fn the_ledger_counts_uses_per_set() {
        let p = std::env::temp_dir().join(format!("ava-gate-ledger-test-{}.tsv", std::process::id()));
        let _ = std::fs::remove_file(&p);
        let (a, b) = ("a".repeat(64), "b".repeat(64));
        assert_eq!(ledger_uses(&p, &a).unwrap(), 0);
        assert_eq!(ledger_record(&p, "dev-qa", &a, "day gate 2026-10-28").unwrap(), 1);
        assert_eq!(ledger_record(&p, "dev-qa", &a, "day gate 2026-10-29").unwrap(), 2);
        assert_eq!(ledger_record(&p, "dev-num", &b, "day gate\tx").unwrap(), 1);
        assert_eq!((ledger_uses(&p, &a).unwrap(), ledger_uses(&p, &b).unwrap()), (2, 1));
        let text = std::fs::read_to_string(&p).unwrap();
        assert_eq!(text.lines().next(), Some(LEDGER_HEADER));
        assert_eq!(text.lines().count(), 4);
        std::fs::remove_file(&p).unwrap();
    }
}
