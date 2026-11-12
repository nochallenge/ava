// TESTING / DESIGN ONLY
//! What to rehearse, and when (Wave 4).
//!
//! - F7 priority = gain x need (Mattar & Daw 2018): gain = how much there is still to learn (her
//!   current loss on it, or its surprise when it happened); need = how likely it is to be needed
//!   soon, from how often and how recently it came up (ACT-R base-level activation).
//! - H10 spacing: each memory has a strength S (days); the chance she still has it after t days is
//!   exp(-t/S). It is due when that falls below the target (0.9). Each good rehearsal multiplies S
//!   by 2 to 4, so the gaps keep growing.
//! - B12 schema fit: a memory that fits what she already knows (low surprise in context) settles
//!   faster: S grows more per rehearsal.
//! - I9 repetition budget: the value of repeating the same text falls after a few passes (about 4
//!   epochs), so each memory gets at most `max_replays` rehearsals.
//! - C2 + B11 + M1: her own words (things she said, her dreams, her daydreams) are capped as a share
//!   of each batch, and dreams have their own smaller cap, so she does not feed on herself. Her own
//!   good answers (M1, on-policy) count under the same cap.

use crate::episodes::Origin;

/// ACT-R base-level activation, squashed to 0..1: need = 1 / (1 + exp(-B)),
/// B = ln(sum over past uses of max(age in days, 1 hour)^-0.5).
pub fn need(uses: &[i64], now: i64) -> f64 {
    let s: f64 = uses.iter().map(|&t| (((now - t) as f64 / 86_400.0).max(1.0 / 24.0)).powf(-0.5)).sum();
    if s <= 0.0 {
        return 0.0;
    }
    1.0 / (1.0 + (-s.ln()).exp())
}

/// The chance a memory of strength `s` days is still there after `t` days.
pub fn recall_prob(s: f64, t: f64) -> f64 {
    (-t.max(0.0) / s.max(1e-6)).exp()
}

/// The strength after one rehearsal. `fit` is how well it fits what she knows (0..1, B12);
/// `ok` is whether she got it right before the rehearsal.
pub fn strength_after(s: f64, fit: f64, ok: bool) -> f64 {
    if ok {
        s * (2.0 + 2.0 * fit.clamp(0.0, 1.0))
    } else {
        (s * 0.5).max(1.0)
    }
}

#[derive(Clone, Debug)]
pub struct Cand {
    pub id: String,
    pub origin: Origin,
    pub gain: f64,
    pub need: f64,
    pub strength_days: f64,
    /// Seconds since it was last rehearsed (None = never).
    pub since: Option<i64>,
    pub replays: u32,
}

#[derive(Clone, Debug)]
pub struct MixPolicy {
    pub batch: usize,
    pub max_replays: u32,
    pub target_recall: f64,
    /// Her own words (said, dreamt, imagined) at most this share of a batch.
    pub hers_cap: f64,
    /// Dreams at most this share of a batch (counted inside `hers_cap` too).
    pub dream_cap: f64,
}

impl Default for MixPolicy {
    fn default() -> Self {
        MixPolicy { batch: 64, max_replays: 4, target_recall: 0.9, hers_cap: 0.2, dream_cap: 0.1 }
    }
}

pub fn is_due(c: &Cand, target: f64) -> bool {
    match c.since {
        None => true,
        Some(t) => recall_prob(c.strength_days, t as f64 / 86_400.0) < target,
    }
}

/// Picks a batch: due memories with budget left, highest gain x need first, within the caps.
/// Returns indices into `cands`.
pub fn pick_batch(cands: &[Cand], p: &MixPolicy) -> Vec<usize> {
    let mut order: Vec<usize> = (0..cands.len()).filter(|&i| cands[i].replays < p.max_replays && is_due(&cands[i], p.target_recall)).collect();
    order.sort_by(|&a, &b| (cands[b].gain * cands[b].need).total_cmp(&(cands[a].gain * cands[a].need)).then(cands[a].id.cmp(&cands[b].id)));
    let hers_max = (p.hers_cap * p.batch as f64).floor() as usize;
    let dream_max = (p.dream_cap * p.batch as f64).floor() as usize;
    let (mut hers, mut dreams, mut out) = (0usize, 0usize, Vec::new());
    for i in order {
        if out.len() >= p.batch {
            break;
        }
        let o = cands[i].origin;
        if o.is_hers() {
            if hers >= hers_max || (o == Origin::Dreamt && dreams >= dream_max) {
                continue;
            }
            hers += 1;
            if o == Origin::Dreamt {
                dreams += 1;
            }
        }
        out.push(i);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(id: &str, o: Origin, gain: f64, since: Option<i64>, replays: u32) -> Cand {
        Cand { id: id.into(), origin: o, gain, need: 0.5, strength_days: 1.0, since, replays }
    }

    #[test]
    fn need_grows_with_use_and_fades_with_time() {
        let day = 86_400;
        let now = 1000 * day;
        let once_recent = need(&[now - day], now);
        let once_old = need(&[now - 100 * day], now);
        let often = need(&[now - day, now - 2 * day, now - 3 * day], now);
        assert!((once_recent - 0.5).abs() < 1e-9);
        assert!(once_old < once_recent && often > once_recent);
        assert_eq!(need(&[], now), 0.0);
    }

    #[test]
    fn spacing_grows_and_schema_fit_speeds_it() {
        let mut s = 1.0;
        let mut gaps = Vec::new();
        for _ in 0..3 {
            s = strength_after(s, 0.0, true);
            gaps.push(s);
        }
        assert_eq!(gaps, vec![2.0, 4.0, 8.0]);
        assert_eq!(strength_after(1.0, 1.0, true), 4.0);
        assert_eq!(strength_after(8.0, 1.0, false), 4.0);
        // Due when the chance of still having it falls below 0.9: for S = 1 day, after about 2.5 hours.
        let mut x = c("a", Origin::Heard, 1.0, Some(2 * 3600), 1);
        assert!(!is_due(&x, 0.9));
        x.since = Some(3 * 3600);
        assert!(is_due(&x, 0.9));
    }

    #[test]
    fn batch_respects_priority_budget_and_caps() {
        let mut v = vec![
            c("spent", Origin::Heard, 9.0, None, 4),
            c("fresh", Origin::Heard, 1.0, None, 0),
            c("said1", Origin::Said, 5.0, None, 0),
            c("said2", Origin::Said, 4.0, None, 0),
            c("dream1", Origin::Dreamt, 8.0, None, 0),
            c("dream2", Origin::Dreamt, 7.0, None, 0),
            c("taught", Origin::Taught, 3.0, None, 0),
        ];
        v.push(c("notdue", Origin::Heard, 9.0, Some(60), 1));
        let p = MixPolicy { batch: 10, max_replays: 4, target_recall: 0.9, hers_cap: 0.2, dream_cap: 0.1 };
        let ids: Vec<&str> = pick_batch(&v, &p).iter().map(|&i| v[i].id.as_str()).collect();
        // 2 of her own at most (20% of 10), 1 dream at most (10%).
        assert_eq!(ids, vec!["dream1", "said1", "taught", "fresh"]);
    }
}
