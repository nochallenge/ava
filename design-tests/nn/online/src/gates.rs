// TESTING / DESIGN ONLY
//! The checks a candidate brain must pass before it is swapped in (Wave 4).
//!
//! - G1 regression gate: per skill, the candidate's and the live brain's scores on the same probe
//!   items (paired). A skill blocks the swap when it drops by more than its margin and the drop is
//!   unlikely to be noise (z <= -z_block), or when it drops clearly (z <= -3) by any amount. This is
//!   the guard against the "stability gap": a skill that dips while something new is learned.
//! - F3 identity gate: a fixed set of questions about herself, answered by both brains (greedy).
//!   Drift = 1 - word overlap, averaged. A swap may move her by at most `per_swap`, and she may move
//!   by at most `total` from her anchor (the first brain) without the owner's say. The changed
//!   answers are listed, so she (and the owner) can see what changed.
//! - L4 forgotten or unreachable: before calling a memory lost, relearn it briefly. If it comes
//!   back much faster than a matched new item is learned (savings), it was still there, only out of
//!   reach (Ebbinghaus savings; "spurious forgetting").

use crate::content_words;
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct SkillCheck {
    pub skill: String,
    /// Per item: (live brain's score, candidate's score). Same items, same order.
    pub pairs: Vec<(f64, f64)>,
    /// The largest mean drop allowed for this skill.
    pub margin: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SkillReport {
    pub skill: String,
    pub n: usize,
    pub old: f64,
    pub new: f64,
    pub diff: f64,
    pub se: f64,
    pub z: f64,
    pub blocks: bool,
}

pub fn regression_gate(checks: &[SkillCheck], z_block: f64) -> Vec<SkillReport> {
    checks
        .iter()
        .map(|c| {
            let n = c.pairs.len();
            let nf = n.max(1) as f64;
            let old = c.pairs.iter().map(|p| p.0).sum::<f64>() / nf;
            let new = c.pairs.iter().map(|p| p.1).sum::<f64>() / nf;
            let d: Vec<f64> = c.pairs.iter().map(|p| p.1 - p.0).collect();
            let diff = new - old;
            let var = if n > 1 { d.iter().map(|x| (x - diff) * (x - diff)).sum::<f64>() / (nf - 1.0) } else { 0.0 };
            let se = (var / nf).sqrt();
            let z = if se > 0.0 {
                diff / se
            } else if diff < 0.0 {
                f64::NEG_INFINITY
            } else {
                0.0
            };
            let blocks = n > 0 && ((diff < -c.margin && z <= -z_block) || z <= -3.0);
            SkillReport { skill: c.skill.clone(), n, old, new, diff, se, z, blocks }
        })
        .collect()
}

/// 1 - Jaccard overlap of content words (0 = same words, 1 = nothing shared).
pub fn answer_drift(a: &str, b: &str) -> f64 {
    let (x, y): (BTreeSet<String>, BTreeSet<String>) = (content_words(a).into_iter().collect(), content_words(b).into_iter().collect());
    if x.is_empty() && y.is_empty() {
        return 0.0;
    }
    1.0 - x.intersection(&y).count() as f64 / x.union(&y).count() as f64
}

pub fn mean_drift(a: &[String], b: &[String]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    (0..n).map(|i| answer_drift(&a[i], &b[i])).sum::<f64>() / n as f64
}

#[derive(Clone, Debug)]
pub struct IdentityBudget {
    pub per_swap: f64,
    pub total: f64,
}

impl Default for IdentityBudget {
    fn default() -> Self {
        IdentityBudget { per_swap: 0.25, total: 0.5 }
    }
}

#[derive(Clone, Debug)]
pub struct IdentityReport {
    pub swap_drift: f64,
    pub total_drift: f64,
    pub blocks: Option<String>,
    /// (question index, old answer, new answer) for answers that changed.
    pub changed: Vec<(usize, String, String)>,
}

/// F3: `anchor`, `live`, `cand` are the three brains' answers to the same self-questions.
pub fn identity_gate(anchor: &[String], live: &[String], cand: &[String], b: &IdentityBudget) -> IdentityReport {
    let swap_drift = mean_drift(live, cand);
    let total_drift = mean_drift(anchor, cand);
    let blocks = if swap_drift > b.per_swap {
        Some(format!("self-answers moved {swap_drift:.2} in one swap (limit {:.2})", b.per_swap))
    } else if total_drift > b.total {
        Some(format!("self-answers moved {total_drift:.2} from the anchor (limit {:.2}); the owner decides", b.total))
    } else {
        None
    };
    let changed = live
        .iter()
        .zip(cand)
        .enumerate()
        .filter(|(_, (a, c))| answer_drift(a, c) > 0.0)
        .map(|(i, (a, c))| (i, a.clone(), c.clone()))
        .collect();
    IdentityReport { swap_drift, total_drift, blocks, changed }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Forgetting {
    Kept,
    /// Still in the weights, only out of reach: it comes back fast.
    Unreachable,
    Lost,
}

/// Savings: 1 - (area under the relearning loss curve) / (area under a matched new item's learning
/// curve), both measured from the same number of short steps. Near 1: it was still there.
pub fn savings(relearn: &[f64], fresh: &[f64]) -> f64 {
    let (a, b): (f64, f64) = (relearn.iter().sum(), fresh.iter().sum());
    if b <= 0.0 {
        return 0.0;
    }
    1.0 - a / b
}

/// L4: was a memory kept, made unreachable, or lost? Losses are per token; `known` is the loss
/// below which it counts as known.
pub fn classify(loss_now: f64, known: f64, savings: f64, min_savings: f64) -> Forgetting {
    if loss_now <= known {
        Forgetting::Kept
    } else if savings >= min_savings {
        Forgetting::Unreachable
    } else {
        Forgetting::Lost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(skill: &str, pairs: Vec<(f64, f64)>) -> SkillCheck {
        SkillCheck { skill: skill.into(), pairs, margin: 0.02 }
    }

    #[test]
    fn a_clear_drop_blocks_and_noise_does_not() {
        // 100 items: 10 that were right are now wrong, none gained.
        let mut drop = vec![(1.0, 1.0); 90];
        drop.extend(vec![(1.0, 0.0); 10]);
        // 100 items: 6 lost, 5 gained (noise).
        let mut noise = vec![(1.0, 1.0); 89];
        noise.extend(vec![(1.0, 0.0); 6]);
        noise.extend(vec![(0.0, 1.0); 5]);
        let r = regression_gate(&[check("numbers", drop), check("facts", noise), check("empty", vec![])], 1.645);
        assert!(r[0].blocks, "{:?}", r[0]);
        assert!((r[0].diff + 0.1).abs() < 1e-12);
        assert!(!r[1].blocks, "{:?}", r[1]);
        assert!(!r[2].blocks);
        // A small but certain drop below the margin still blocks (z <= -3).
        let mut tiny = vec![(1.0, 1.0); 990];
        tiny.extend(vec![(1.0, 0.0); 10]);
        let r = regression_gate(&[check("x", tiny)], 1.645);
        assert!(r[0].blocks && r[0].z <= -3.0, "{:?}", r[0]);
    }

    #[test]
    fn identity_gate_limits_each_swap_and_the_total() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let anchor = s(&["My name is Ava.", "I like reading books."]);
        let small = s(&["My name is Ava.", "I like reading long books."]);
        let big = s(&["I am a teapot.", "Nothing at all."]);
        let r = identity_gate(&anchor, &anchor, &small, &IdentityBudget::default());
        assert!(r.blocks.is_none(), "{r:?}");
        assert_eq!(r.changed.len(), 1);
        assert_eq!(r.changed[0].0, 1);
        let r = identity_gate(&anchor, &anchor, &big, &IdentityBudget::default());
        assert!(r.blocks.unwrap().contains("one swap"));
        // Small steps that add up still stop at the total.
        let b = IdentityBudget { per_swap: 1.0, total: 0.3 };
        let r = identity_gate(&anchor, &small, &big, &b);
        assert!(r.blocks.unwrap().contains("anchor"));
    }

    #[test]
    fn savings_tell_unreachable_from_lost() {
        let fresh = [4.0, 3.5, 3.0, 2.6];
        let back_fast = [3.0, 1.0, 0.5, 0.4];
        let slow = [4.0, 3.4, 2.9, 2.5];
        let s1 = savings(&back_fast, &fresh);
        let s2 = savings(&slow, &fresh);
        assert!(s1 > 0.6 && s2 < 0.05, "{s1} {s2}");
        assert_eq!(classify(0.3, 0.5, 0.0, 0.3), Forgetting::Kept);
        assert_eq!(classify(2.0, 0.5, s1, 0.3), Forgetting::Unreachable);
        assert_eq!(classify(2.0, 0.5, s2, 0.3), Forgetting::Lost);
    }
}
