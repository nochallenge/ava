// TESTING / DESIGN ONLY
//! I7: learning by teaching (the protégé effect; Chase et al. 2009), scored by what the student learns.
//! An explanation is not good because it sounds good; it is good if someone who reads it gets better.
//! She writes several explanations of the same thing; a student (a small or earlier copy of her) studies
//! each one for a few steps (`crate::ttt::adapt`), and is tested on questions the explanation did not
//! contain. The student's gain is the explanation's reward, and the group-relative advantages
//! (`crate::rlp::advantages`) train her to explain better. The student is restored after each one.
//!   gain = mean log-prob (or −error) on held-out questions after studying − before studying

/// The gain of each explanation from the student's held-out scores before and after (higher is better).
pub fn gains(before: f32, after: &[f32]) -> Vec<f32> {
    after.iter().map(|a| a - before).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rlp::advantages;
    use crate::testkit::*;
    use crate::ttt::{adapt, restore};
    use candle_core::{Tensor, Var};

    #[test]
    fn explanations_are_ranked_by_what_the_student_learns() {
        // The thing to teach: a rule y = x·R. The student starts knowing nothing (W = 0).
        let rule = randn(&[4, 2], 1);
        let w = Var::from_tensor(&Tensor::zeros((4, 2), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap()).unwrap();
        let test_x = randn(&[64, 4], 2);
        let test_y = test_x.matmul(&rule).unwrap();
        let score = |w: &Var| -(test_x.matmul(w.as_tensor()).unwrap() - &test_y).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        let before = score(&w);
        // Three explanations, each four worked examples: varied and right; the same example four times;
        // varied but with wrong answers.
        let varied = randn(&[4, 4], 3);
        let same = randn(&[1, 4], 4).broadcast_as((4, 4)).unwrap().contiguous().unwrap();
        let wrong_rule = (rule.clone() * -1.0).unwrap();
        let explanations = [(varied.clone(), varied.matmul(&rule).unwrap()), (same.clone(), same.matmul(&rule).unwrap()), (varied.clone(), varied.matmul(&wrong_rule).unwrap())];
        let mut after = Vec::new();
        for (x, y) in &explanations {
            let loss = || (x.matmul(w.as_tensor())? - y)?.sqr()?.mean_all();
            let saved = adapt(&[w.clone()], &loss, 50, 0.1).unwrap();
            after.push(score(&w));
            restore(&[w.clone()], &saved).unwrap();
        }
        let g = gains(before, &after);
        eprintln!("student gains: varied {:.3}, repeated {:.3}, wrong {:.3}", g[0], g[1], g[2]);
        assert!(g[0] > g[1] && g[1] > 0.0 && g[2] < 0.0, "{g:?}");
        let a = advantages(&g, 1e-6);
        assert!(a[0] > 0.0 && a[2] < 0.0);
    }
}
