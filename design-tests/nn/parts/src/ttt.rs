// TESTING / DESIGN ONLY
//! A6: test-time training for rule induction (Akyürek et al. 2024 on ARC; the "few-shot learning in the
//! moment" of a person who studies the examples before answering). Before she answers a question that
//! comes with worked examples, she practises on the examples themselves for a few steps (each example in
//! turn held out and predicted from the others), answers, then lets the change go. Her lasting weights
//! never move, so nothing from one question leaks into the next.
//!
//! `adapt` takes the weights to adapt (in practice a small fast-weight or low-rank part), a loss on the
//! examples, and the number of steps; it returns the snapshot to restore after answering.

use candle_core::{Result, Tensor, Var};

/// Copies of the weights' current values (to restore after answering).
pub fn snapshot(vars: &[Var]) -> Result<Vec<Tensor>> {
    vars.iter().map(|v| v.as_tensor().copy()).collect()
}

pub fn restore(vars: &[Var], saved: &[Tensor]) -> Result<()> {
    for (v, s) in vars.iter().zip(saved) {
        v.set(s)?;
    }
    Ok(())
}

/// `steps` plain gradient steps of size `lr` on `loss` (recomputed each step). Returns the snapshot taken
/// before adapting.
pub fn adapt(vars: &[Var], loss: &dyn Fn() -> Result<Tensor>, steps: usize, lr: f64) -> Result<Vec<Tensor>> {
    let saved = snapshot(vars)?;
    for _ in 0..steps {
        let g = loss()?.backward()?;
        for v in vars {
            if let Some(gv) = g.get(v.as_tensor()) {
                v.set(&(v.as_tensor() - (gv * lr)?)?)?;
            }
        }
    }
    Ok(saved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;

    /// Each task is a hidden linear rule y = x·R_task on 4 numbers. The model's shared weights W were
    /// "pre-trained" on the average rule, so a new task's rule is unknown to it. Three examples come with
    /// the question; the query is a fourth input.
    #[test]
    fn practising_on_the_examples_helps_and_leaves_no_trace() {
        let base = randn(&[4, 4], 1);
        let w = Var::from_tensor(&base).unwrap();
        let (mut err_without, mut err_with) = (0f32, 0f32);
        for task in 0..20u64 {
            let rule = (&base + (randn(&[4, 4], 100 + task) * 0.5).unwrap()).unwrap();
            let xs = randn(&[3, 4], 200 + task);
            let ys = xs.matmul(&rule).unwrap();
            let q = randn(&[1, 4], 300 + task);
            let want = q.matmul(&rule).unwrap();
            let err = |w: &Var| (q.matmul(w.as_tensor()).unwrap() - &want).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            err_without += err(&w);
            let loss = || (xs.matmul(w.as_tensor())? - &ys)?.sqr()?.mean_all();
            let saved = adapt(&[w.clone()], &loss, 100, 0.3).unwrap();
            err_with += err(&w);
            restore(&[w.clone()], &saved).unwrap();
            assert_eq!(max_diff(w.as_tensor(), &base), 0.0, "the lasting weights are back exactly");
        }
        eprintln!("query error over 20 new rules: without practice {err_without:.2}, with {err_with:.2}");
        assert!(err_with < 0.5 * err_without, "{err_without} {err_with}");
    }
}
