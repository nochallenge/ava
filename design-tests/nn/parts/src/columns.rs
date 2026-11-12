// TESTING / DESIGN ONLY
//! H8: many cortical columns voting on what they sense (the Thousand Brains theory, Hawkins et al. 2017,
//! 2019). Each column senses one small patch: a feature at a location on the object (the location comes
//! from path integration, as grid cells give it; `crate::tem` learns that structure). It learns objects
//! by one exposure, as associations "this feature at this location belongs to this object" (Hebbian
//! counts). A single touch is ambiguous, because many objects share a feature at a place. Alone, a column
//! has to move and touch again and again. Columns that touch different places at once send each other
//! their beliefs over long-range lateral links, and the object is known in one step:
//!   own_c(o)   = Σ_steps ln P(feature | o, location)          (smoothed counts; unknown = uniform)
//!   msg_c(o)   = max(own_c(o) − max_o own_c, floor)           (the objects it still allows; no lone veto)
//!   belief_c   = softmax(own_c + w · Σ_{c' ≠ c} msg_c')
//! With w = 0 the columns are independent. The floor keeps one broken column from overruling the rest,
//! and the vote corrects that column top-down.

/// The object memory of a set of columns: per column, counts[(loc · n_feat + feat) · n_obj + obj].
pub struct Columns {
    pub n_cols: usize,
    pub n_obj: usize,
    pub n_feat: usize,
    pub n_loc: usize,
    counts: Vec<Vec<f32>>,
    /// Laplace smoothing of P(feature | object, location).
    pub alpha: f32,
}

impl Columns {
    pub fn new(n_cols: usize, n_obj: usize, n_feat: usize, n_loc: usize) -> Self {
        Self { n_cols, n_obj, n_feat, n_loc, counts: vec![vec![0.0; n_loc * n_feat * n_obj]; n_cols], alpha: 0.01 }
    }

    /// One exposure: column `col` felt `feat` at `loc` on object `obj`.
    pub fn learn(&mut self, col: usize, obj: usize, loc: usize, feat: usize) {
        self.counts[col][(loc * self.n_feat + feat) * self.n_obj + obj] += 1.0;
    }

    /// ln P(feat | object, loc) for every object, as column `col` has learned it.
    pub fn evidence(&self, col: usize, loc: usize, feat: usize) -> Vec<f32> {
        let c = &self.counts[col];
        let a = self.alpha;
        (0..self.n_obj)
            .map(|o| {
                let total: f32 = (0..self.n_feat).map(|f| c[(loc * self.n_feat + f) * self.n_obj + o]).sum();
                ((c[(loc * self.n_feat + feat) * self.n_obj + o] + a) / (total + a * self.n_feat as f32)).ln()
            })
            .collect()
    }

    /// Adds one step of sensing to each column's own evidence: `touch[c]` = Some((loc, feat)) or None.
    pub fn sense(&self, own: &mut [Vec<f32>], touch: &[Option<(usize, usize)>]) {
        for (c, t) in touch.iter().enumerate() {
            if let Some((loc, feat)) = *t {
                for (o, e) in self.evidence(c, loc, feat).into_iter().enumerate() {
                    own[c][o] += e;
                }
            }
        }
    }
}

/// ln softmax of a vector.
pub fn log_softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let z = x.iter().map(|v| (v - m).exp()).sum::<f32>().ln() + m;
    x.iter().map(|v| v - z).collect()
}

/// One round of lateral voting. Returns each column's belief (ln probabilities) over the objects.
pub fn vote(own: &[Vec<f32>], weight: f32, floor: f32) -> Vec<Vec<f32>> {
    // Each column sends the set of objects its own evidence still allows (its best ones at 0, the ruled
    // out ones down to the floor), like the union of candidates a column sends in the Thousand Brains model.
    let msgs: Vec<Vec<f32>> = own
        .iter()
        .map(|o| {
            let m = o.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            o.iter().map(|v| (v - m).max(floor)).collect()
        })
        .collect();
    (0..own.len())
        .map(|c| {
            let x: Vec<f32> = (0..own[c].len())
                .map(|o| own[c][o] + weight * (0..own.len()).filter(|&k| k != c).map(|k| msgs[k][o]).sum::<f32>())
                .collect();
            log_softmax(&x)
        })
        .collect()
}

/// The object a belief points to, only if it is the single highest (a tie counts as not knowing).
pub fn decided(belief: &[f32]) -> Option<usize> {
    let mut best = 0;
    for o in 1..belief.len() {
        if belief[o] > belief[best] {
            best = o;
        }
    }
    let unique = belief.iter().enumerate().all(|(o, &v)| o == best || v < belief[best]);
    unique.then_some(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    const OBJ: usize = 30;
    const FEAT: usize = 5;
    const LOC: usize = 10;
    const COLS: usize = 7;

    /// Random objects: a feature at every location. Each column learns every object once.
    fn world(seed: u64, n_obj: usize) -> (Vec<Vec<usize>>, Columns) {
        let mut r = Rng(seed | 1);
        let objs: Vec<Vec<usize>> = (0..n_obj).map(|_| (0..LOC).map(|_| r.next(FEAT)).collect()).collect();
        let mut cols = Columns::new(COLS, n_obj, FEAT, LOC);
        for (o, obj) in objs.iter().enumerate() {
            for c in 0..COLS {
                for (l, &f) in obj.iter().enumerate() {
                    cols.learn(c, o, l, f);
                }
            }
        }
        (objs, cols)
    }

    /// Distinct random locations, one per column.
    fn spots(r: &mut Rng, n: usize) -> Vec<usize> {
        let mut l: Vec<usize> = (0..LOC).collect();
        for i in 0..n {
            let j = i + r.next(LOC - i);
            l.swap(i, j);
        }
        l[..n].to_vec()
    }

    #[test]
    fn one_touch_is_ambiguous_but_voting_columns_know_at_once() {
        let (objs, cols) = world(7, OBJ);
        let mut r = Rng(99);
        let (mut alone, mut voted, trials) = (0, 0, 300);
        for t in 0..trials {
            let truth = t % OBJ;
            let at = spots(&mut r, COLS);
            let mut own = vec![vec![0.0; OBJ]; COLS];
            let touch: Vec<_> = at.iter().map(|&l| Some((l, objs[truth][l]))).collect();
            cols.sense(&mut own, &touch);
            alone += (decided(&log_softmax(&own[0])) == Some(truth)) as usize;
            voted += vote(&own, 1.0, -3.0).iter().filter(|b| decided(b) == Some(truth)).count();
        }
        let alone = alone as f32 / trials as f32;
        let voted = voted as f32 / (trials * COLS) as f32;
        eprintln!("one touch: a column alone is right {:.1}%, with {COLS} columns voting {:.1}%", 100.0 * alone, 100.0 * voted);
        assert!(alone < 0.35, "{alone}");
        assert!(voted > 0.95, "{voted}");
        // With no vote each column stays on its own evidence.
        let own = vec![vec![0.0, -1.0, -2.0]; 3];
        let v = vote(&own, 0.0, -3.0);
        for b in &v {
            for (x, y) in b.iter().zip(log_softmax(&own[0])) {
                assert!((x - y).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn alone_a_column_must_touch_many_times() {
        let (objs, cols) = world(7, OBJ);
        let mut r = Rng(5);
        let trials = 300;
        let mut right_after = vec![0usize; LOC + 1];
        for t in 0..trials {
            let truth = t % OBJ;
            let path = spots(&mut r, LOC);
            let mut own = vec![vec![0.0; OBJ]; 1];
            for (k, &l) in path.iter().enumerate() {
                cols.sense(&mut own, &[Some((l, objs[truth][l]))]);
                right_after[k + 1] += (decided(&own[0]) == Some(truth)) as usize;
            }
        }
        let need = (1..=LOC).find(|&k| right_after[k] as f32 >= 0.95 * trials as f32).unwrap_or(LOC + 1);
        eprintln!("a lone column needs {need} touches to be right 95% of the time");
        assert!(need >= 4, "{need}");
    }

    #[test]
    fn a_broken_column_is_outvoted_and_corrected() {
        let (objs, cols) = world(11, OBJ);
        let mut r = Rng(3);
        let (mut rest, mut broken, trials) = (0, 0, 300);
        for t in 0..trials {
            let truth = t % OBJ;
            let at = spots(&mut r, COLS);
            let mut touch: Vec<_> = at.iter().map(|&l| Some((l, objs[truth][l]))).collect();
            // Column 0's sensor is broken: it reports a random feature (wrong 4 times in 5).
            touch[0] = Some((at[0], r.next(FEAT)));
            let mut own = vec![vec![0.0; OBJ]; COLS];
            cols.sense(&mut own, &touch);
            let b = vote(&own, 1.0, -3.0);
            broken += (decided(&b[0]) == Some(truth)) as usize;
            rest += b[1..].iter().filter(|x| decided(x) == Some(truth)).count();
        }
        let rest = rest as f32 / (trials * (COLS - 1)) as f32;
        let broken = broken as f32 / trials as f32;
        eprintln!("with a broken column: the others right {:.1}%, the broken one corrected to right {:.1}%", 100.0 * rest, 100.0 * broken);
        assert!(rest > 0.9, "{rest}");
        assert!(broken > 0.85, "{broken}");
    }

    #[test]
    fn new_objects_do_not_erase_old_ones() {
        // Learn 30 objects, then 10 more: recognition of the first 30 is the same as before.
        let (objs, cols_old) = world(21, OBJ);
        let (objs_all, cols_all) = world(21, OBJ + 10);
        assert_eq!(objs, objs_all[..OBJ].to_vec());
        let mut r = Rng(8);
        let (mut a, mut b) = (0, 0);
        for t in 0..300 {
            let truth = t % OBJ;
            let at = spots(&mut r, COLS);
            let touch: Vec<_> = at.iter().map(|&l| Some((l, objs[truth][l]))).collect();
            let (mut o1, mut o2) = (vec![vec![0.0; OBJ]; COLS], vec![vec![0.0; OBJ + 10]; COLS]);
            cols_old.sense(&mut o1, &touch);
            cols_all.sense(&mut o2, &touch);
            a += (decided(&vote(&o1, 1.0, -3.0)[0]) == Some(truth)) as usize;
            b += (decided(&vote(&o2, 1.0, -3.0)[0]) == Some(truth)) as usize;
        }
        eprintln!("old objects right: {a}/300 before, {b}/300 after learning 10 more");
        assert!(b + 6 >= a, "{a} -> {b}");
    }
}
