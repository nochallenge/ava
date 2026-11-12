// TESTING / DESIGN ONLY
//! C8 confidence and F9 feeling of knowing. People often know whether they know something before they
//! recall it (Hart 1965; Koriat), and a good mind says "I don't know" instead of guessing. The owner scores
//! answers by right minus wrong, so knowing when to stay quiet is worth as much as knowing more.
//!
//! - A probe head on her state before she answers: c = sigmoid(h·w + b), trained (BCE) to predict
//!   whether her answer will be right. That is her feeling of knowing.
//! - Calibration: expected calibration error (ECE, equal-width bins) and the Brier score.
//! - `best_threshold`: the confidence level to answer at, chosen on DEV items to maximise right − wrong
//!   (abstaining scores 0). The threshold is then fixed before any sealed test.

use crate::linear;
use candle_core::{Result, Tensor};
use candle_nn::{Init, VarBuilder};

pub struct Probe {
    pub w: Tensor,
    pub b: Tensor,
}

impl Probe {
    pub fn new(vb: VarBuilder, d: usize) -> Result<Probe> {
        Ok(Probe { w: vb.get_with_hints((d, 1), "w", Init::Const(0.0))?, b: vb.get_with_hints(1, "b", Init::Const(0.0))? })
    }
    /// Confidence logits [..] for h [.., d].
    pub fn logits(&self, h: &Tensor) -> Result<Tensor> {
        linear(&h.detach(), &self.w)?.broadcast_add(&self.b)?.squeeze(candle_core::D::Minus1)
    }
    /// BCE against whether the answer was right (1) or wrong (0), same shape as the logits.
    pub fn loss(&self, h: &Tensor, right: &Tensor) -> Result<Tensor> {
        let z = self.logits(h)?;
        // log(1 + e^z) − y·z, written stably.
        let softplus = (z.relu()? + (z.abs()?.neg()?.exp()? + 1.0)?.log()?)?;
        (softplus - (right * &z)?)?.mean_all()
    }
}

pub fn ece(conf: &[f32], right: &[bool], bins: usize) -> f32 {
    let mut s = vec![(0f32, 0f32, 0usize); bins];
    for (&c, &r) in conf.iter().zip(right) {
        let b = ((c * bins as f32) as usize).min(bins - 1);
        s[b].0 += c;
        s[b].1 += r as u8 as f32;
        s[b].2 += 1;
    }
    let n = conf.len() as f32;
    s.iter().filter(|b| b.2 > 0).map(|b| (b.0 - b.1).abs() / n).sum()
}

pub fn brier(conf: &[f32], right: &[bool]) -> f32 {
    conf.iter().zip(right).map(|(&c, &r)| (c - r as u8 as f32).powi(2)).sum::<f32>() / conf.len() as f32
}

/// (threshold, right, wrong): answer only when confidence >= threshold, chosen for the most right − wrong.
pub fn best_threshold(conf: &[f32], right: &[bool]) -> (f32, usize, usize) {
    let mut order: Vec<usize> = (0..conf.len()).collect();
    order.sort_by(|&a, &b| conf[b].total_cmp(&conf[a]));
    let (mut best, mut r, mut w) = ((f32::INFINITY, 0usize, 0usize), 0usize, 0usize);
    let mut best_score = 0i64;
    for (k, &i) in order.iter().enumerate() {
        if right[i] {
            r += 1;
        } else {
            w += 1;
        }
        // Only cut between different confidence values.
        if k + 1 < order.len() && conf[order[k + 1]] == conf[i] {
            continue;
        }
        if r as i64 - w as i64 > best_score {
            best_score = r as i64 - w as i64;
            best = (conf[i], r, w);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_and_the_answer_threshold() {
        // Calibrated by construction: in each group the share right equals the confidence.
        let mut conf = Vec::new();
        let mut right = Vec::new();
        for (c, n_right) in [(0.15f32, 3usize), (0.55, 11), (0.95, 19)] {
            for k in 0..20 {
                conf.push(c);
                right.push(k < n_right);
            }
        }
        assert!(ece(&conf, &right, 10) < 0.01);
        assert!(ece(&conf.iter().map(|_| 0.99).collect::<Vec<_>>(), &right, 10) > 0.4);
        assert!(brier(&conf, &right) < 0.2);
        // Answering at 0.55 and up gives 30 right, 10 wrong (20); at 0.95 only, 19 − 1 = 18; all, 33 − 27 = 6.
        assert_eq!(best_threshold(&conf, &right), (0.55, 30, 10));
    }

    #[test]
    fn the_probe_learns_to_predict_right_answers() {
        let (vm, vb) = crate::testkit::vm();
        let p = Probe::new(vb, 3).unwrap();
        let h = crate::testkit::randn(&[200, 3], 2);
        let y = h.narrow(1, 0, 1).unwrap().squeeze(1).unwrap().ge(0f64).unwrap().to_dtype(candle_core::DType::F32).unwrap();
        let w = crate::testkit::var(&vm, "w");
        let first = p.loss(&h, &y).unwrap().to_scalar::<f32>().unwrap();
        for _ in 0..200 {
            let g = p.loss(&h, &y).unwrap().backward().unwrap();
            w.set(&(w.as_tensor() - (g.get(w.as_tensor()).unwrap() * 2.0).unwrap()).unwrap()).unwrap();
        }
        let last = p.loss(&h, &y).unwrap().to_scalar::<f32>().unwrap();
        assert!((first - std::f32::consts::LN_2).abs() < 1e-4 && last < 0.3, "{first} {last}");
    }
}
