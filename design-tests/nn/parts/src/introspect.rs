// TESTING / DESIGN ONLY
//! F10: introspection grounded in real internal states, and checked by cause, not by plausibility
//! (concept injection, Lindsey 2025; the confabulation findings of Nisbett & Wilson 1977). People often
//! give reasons that sound right but have nothing to do with what happened inside them. A self-report
//! is only trusted if changing the inner state changes the report:
//! - a reporter reads a property from her state (a probe on detached activity, like `confidence`);
//! - `inject` adds a direction v to the state at strength s (h + s·v), the text unchanged;
//! - `causal_check` measures the report at each strength; `faithfulness` is the correlation between
//!   the strength and the report. A report that follows the injection is grounded; one that does not
//!   is confabulated, however fluent it is.

use candle_core::{Result, Tensor};

pub fn inject(h: &Tensor, v: &Tensor, strength: f64) -> Result<Tensor> {
    h.broadcast_add(&(v * strength)?)
}

/// The mean report at each strength.
pub fn causal_check(report: &dyn Fn(&Tensor) -> Result<Tensor>, h: &Tensor, v: &Tensor, strengths: &[f64]) -> Result<Vec<f32>> {
    strengths.iter().map(|&s| report(&inject(h, v, s)?)?.mean_all()?.to_scalar::<f32>()).collect()
}

/// Pearson correlation between the strengths and the reports (0 when either does not vary).
pub fn faithfulness(strengths: &[f64], reports: &[f32]) -> f32 {
    let n = strengths.len() as f64;
    let (ms, mr) = (strengths.iter().sum::<f64>() / n, reports.iter().map(|&r| r as f64).sum::<f64>() / n);
    let (mut c, mut vs, mut vr) = (0.0, 0.0, 0.0);
    for (&s, &r) in strengths.iter().zip(reports) {
        let (a, b) = (s - ms, r as f64 - mr);
        c += a * b;
        vs += a * a;
        vr += b * b;
    }
    if vs < 1e-12 || vr < 1e-12 {
        return 0.0;
    }
    (c / (vs * vr).sqrt()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_core::{Device, Var};
    use candle_nn::{AdamW, Optimizer, ParamsAdamW};

    #[test]
    fn grounded_reports_follow_the_injection_and_confabulated_ones_do_not() {
        let d = 8;
        let h = randn(&[256, d], 1);
        // The inner property: how much of direction u is in the state.
        let mut u = vec![0f32; d];
        u[0] = 1.0;
        let u = Tensor::from_vec(u, (d, 1), &Device::Cpu).unwrap();
        let y = h.matmul(&u).unwrap();
        let w = Var::from_tensor(&Tensor::zeros((d, 1), candle_core::DType::F32, &Device::Cpu).unwrap()).unwrap();
        let mut opt = AdamW::new(vec![w.clone()], ParamsAdamW { lr: 0.05, weight_decay: 0.0, ..Default::default() }).unwrap();
        for _ in 0..300 {
            let loss = (h.detach().matmul(w.as_tensor()).unwrap() - &y).unwrap().sqr().unwrap().mean_all().unwrap();
            opt.backward_step(&loss).unwrap();
        }
        let probe = |x: &Tensor| x.detach().matmul(w.as_tensor());
        let strengths = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let along = u.t().unwrap().squeeze(0).unwrap();
        let mut o = vec![0f32; d];
        o[3] = 1.0;
        let other = Tensor::from_vec(o, d, &Device::Cpu).unwrap();
        let r_along = causal_check(&probe, &h, &along, &strengths).unwrap();
        let r_other = causal_check(&probe, &h, &other, &strengths).unwrap();
        assert!(faithfulness(&strengths, &r_along) > 0.99);
        assert!((r_along[4] - r_along[0] - 4.0).abs() < 0.2, "{r_along:?}");
        assert!((r_other[4] - r_other[0]).abs() < 0.2, "{r_other:?}");
        // A confabulating reporter (it answers from the words, which injection does not change).
        let confab = |x: &Tensor| Tensor::ones((x.dim(0)?, 1), candle_core::DType::F32, x.device());
        assert_eq!(faithfulness(&strengths, &causal_check(&confab, &h, &along, &strengths).unwrap()), 0.0);
    }
}
