// TESTING / DESIGN ONLY
//! C19: several minds together (the "wisdom of crowds", Galton 1907; weighted voting, Nitzan & Paroush
//! 1982; multi-agent debate, Du et al. 2023). A group of independent minds, each right more often than
//! not, is right more often than any one of them; weighting each by how sure it is helps further, if
//! the sureness is honest (calibrated, `crate::confidence`).
//! - `vote`: the plain majority (ties go to the lowest answer, so it is deterministic);
//! - `weighted_vote`: each voter counts ln((K−1)·c/(1−c)) for K choices (the optimal weight for
//!   independent voters whose mistakes spread evenly; a mind right 30% of the time on 4 choices still
//!   counts for something, since chance is 25%);
//! - `debate_round`: each mind looks at the others' answers and sureness and keeps or changes its own
//!   (it moves only if the others' weighted support for another answer beats its own weight).
//! For Ava the minds are copies of her at different temperatures or with different thoughts, and in a
//! debate they read each other's reasons, not only the answers; that part is on the gpu side.

pub fn vote(answers: &[usize], choices: usize) -> usize {
    let mut n = vec![0usize; choices];
    answers.iter().for_each(|&a| n[a] += 1);
    (0..choices).max_by(|&a, &b| n[a].cmp(&n[b]).then(b.cmp(&a))).unwrap()
}

fn log_odds(c: f32, choices: usize) -> f32 {
    let c = c.clamp(1e-4, 1.0 - 1e-4);
    ((choices - 1) as f32 * c / (1.0 - c)).ln()
}

pub fn weighted_vote(answers: &[usize], conf: &[f32], choices: usize) -> usize {
    let mut s = vec![0f32; choices];
    answers.iter().zip(conf).for_each(|(&a, &c)| s[a] += log_odds(c, choices));
    (0..choices).max_by(|&a, &b| s[a].total_cmp(&s[b]).then(b.cmp(&a))).unwrap()
}

pub fn debate_round(answers: &[usize], conf: &[f32], choices: usize) -> Vec<usize> {
    (0..answers.len())
        .map(|i| {
            let mut s = vec![0f32; choices];
            for (j, (&a, &c)) in answers.iter().zip(conf).enumerate() {
                if j != i {
                    s[a] += log_odds(c, choices);
                }
            }
            let own = log_odds(conf[i], choices) + s[answers[i]];
            let (best, &sb) = s.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
            if best != answers[i] && sb > own {
                best
            } else {
                answers[i]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    #[test]
    fn groups_beat_one_and_honest_sureness_helps() {
        let (minds, choices, questions) = (5usize, 4usize, 4000usize);
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        let (mut one, mut plain, mut weighted, mut debated) = (0, 0, 0, 0);
        for _ in 0..questions {
            let truth = (r.u() * choices as f32) as usize % choices;
            let (mut ans, mut conf) = (Vec::new(), Vec::new());
            for _ in 0..minds {
                // Each mind has its own skill on this question, knows it (calibrated), and is right that often.
                let skill = 0.3 + 0.6 * r.u();
                let right = r.u() < skill;
                ans.push(if right { truth } else { (truth + 1 + (r.u() * (choices - 1) as f32) as usize % (choices - 1)) % choices });
                conf.push(skill);
            }
            one += (ans[0] == truth) as usize;
            plain += (vote(&ans, choices) == truth) as usize;
            weighted += (weighted_vote(&ans, &conf, choices) == truth) as usize;
            let after = debate_round(&ans, &conf, choices);
            debated += (vote(&after, choices) == truth) as usize;
        }
        let pct = |n: usize| 100.0 * n as f32 / questions as f32;
        eprintln!("one mind {:.1}%, majority {:.1}%, weighted {:.1}%, after a debate round {:.1}%", pct(one), pct(plain), pct(weighted), pct(debated));
        assert!(plain > one && weighted > plain && debated > plain, "{one} {plain} {weighted} {debated}");
        assert_eq!(vote(&[2, 1, 1, 2], 3), 1);
    }
}
