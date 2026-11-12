// TESTING / DESIGN ONLY
//! C13 merging: combine brains that were tuned from the same base into one, without training. For example,
//! a copy that learned numbers and a copy that learned conversation become one that keeps both, which is
//! a cheap form of consolidation.
//! - average: plain weight averaging (model soups, Wortsman et al. 2022);
//! - task arithmetic (Ilharco et al., ICLR 2023): base + λ·Σ (tuned_i − base);
//! - TIES (Yadav et al., NeurIPS 2023): trim each task vector to its largest `keep` share, elect a sign
//!   per weight by the summed trimmed values, and average only the values that agree with the elected
//!   sign. This avoids interference between the tasks.
//! Each works on one flat tensor at a time, so a whole brain is merged name by name.

pub fn average(models: &[&[f32]]) -> Vec<f32> {
    let n = models.len() as f32;
    (0..models[0].len()).map(|i| models.iter().map(|m| m[i]).sum::<f32>() / n).collect()
}

pub fn task_arithmetic(base: &[f32], tuned: &[&[f32]], lambda: f32) -> Vec<f32> {
    (0..base.len()).map(|i| base[i] + lambda * tuned.iter().map(|t| t[i] - base[i]).sum::<f32>()).collect()
}

pub fn ties(base: &[f32], tuned: &[&[f32]], keep: f32, lambda: f32) -> Vec<f32> {
    let n = base.len();
    let trimmed: Vec<Vec<f32>> = tuned
        .iter()
        .map(|t| {
            let tau: Vec<f32> = (0..n).map(|i| t[i] - base[i]).collect();
            let k = ((n as f32) * keep).round() as usize;
            let mut mags: Vec<f32> = tau.iter().map(|x| x.abs()).collect();
            mags.sort_by(|a, b| b.total_cmp(a));
            let cut = if k == 0 { f32::INFINITY } else { mags[k.min(n) - 1] };
            tau.iter().map(|&x| if x.abs() >= cut && x != 0.0 { x } else { 0.0 }).collect()
        })
        .collect();
    (0..n)
        .map(|i| {
            let sign = trimmed.iter().map(|t| t[i]).sum::<f32>().signum();
            let agree: Vec<f32> = trimmed.iter().map(|t| t[i]).filter(|&x| x != 0.0 && x.signum() == sign).collect();
            let m = if agree.is_empty() { 0.0 } else { agree.iter().sum::<f32>() / agree.len() as f32 };
            base[i] + lambda * m
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_ways_to_merge() {
        assert_eq!(average(&[&[1.0, 2.0], &[3.0, 6.0]]), vec![2.0, 4.0]);
        assert_eq!(task_arithmetic(&[1.0, 1.0], &[&[2.0, 1.0], &[1.0, 4.0]], 0.5), vec![1.5, 2.5]);
        // TIES: task vectors [1, -2, 0.1] and [1, 3, -0.1], keep 2 of 3 each: [1, -2, 0] and [1, 3, 0].
        // Elected signs: + and + (2 and 1). Weight 1 keeps only the +3. Weight 2 has nothing left.
        let got = ties(&[0.0, 0.0, 0.0], &[&[1.0, -2.0, 0.1], &[1.0, 3.0, -0.1]], 2.0 / 3.0, 1.0);
        assert_eq!(got, vec![1.0, 3.0, 0.0]);
    }
}
