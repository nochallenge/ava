// TESTING / DESIGN ONLY
//! D10: intuitive physics, learned and tested the way infants are, by violation of expectation (Spelke;
//! Baillargeon; the PLATO model, Piloto et al. 2022). A small forward model (like the cerebellum's)
//! learns from watching a ball move under gravity and bounce off the walls and floor. It is never told
//! the laws. While the ball is hidden behind a screen she keeps imagining it, feeding her own prediction
//! back as the input (object permanence). Her surprise is how far the world departs from what she
//! expected:
//!   surprise = Σ_t visible_t · |seen_t − expected_t|² + κ · [she expected to see it and it is not there]
//! Impossible events must surprise her more than matched possible ones: a ball that jumps (continuity),
//! falls through the floor (solidity), turns in mid-air (inertia), vanishes behind the screen
//! (permanence), or comes out of it somewhere else.

use candle_core::{Device, Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

pub const DT: f32 = 0.05;
pub const G: f32 = 1.0;
pub const BOUNCE: f32 = 0.9;
/// The screen: the ball is hidden while 0.4 < x < 0.6 and y < 0.6.
pub const SCREEN: (f32, f32, f32) = (0.4, 0.6, 0.6);
/// Velocities are scaled by this for the network (a step moves about 0.02).
const VS: f32 = 1.0 / DT;

pub fn hidden_at(p: [f32; 2]) -> bool {
    p[0] > SCREEN.0 && p[0] < SCREEN.1 && p[1] < SCREEN.2
}

/// How far a point is inside (+) or outside (−) the screen.
fn depth_in_screen(p: [f32; 2]) -> f32 {
    (p[0] - SCREEN.0).min(SCREEN.1 - p[0]).min(SCREEN.2 - p[1])
}

/// One step of the true world: gravity, then bounces off x = 0, x = 1, the floor y = 0 and the top y = 1.5.
pub fn step(p: [f32; 2], v: [f32; 2]) -> ([f32; 2], [f32; 2]) {
    let mut v = [v[0], v[1] - G * DT];
    let mut p = [p[0] + v[0] * DT, p[1] + v[1] * DT];
    for (k, (lo, hi)) in [(0.0f32, 1.0f32), (0.0, 1.5)].into_iter().enumerate() {
        if p[k] < lo {
            p[k] = lo + (lo - p[k]) * BOUNCE;
            v[k] = -v[k] * BOUNCE;
        } else if p[k] > hi {
            p[k] = hi - (p[k] - hi) * BOUNCE;
            v[k] = -v[k] * BOUNCE;
        }
    }
    (p, v)
}

pub fn trajectory(p0: [f32; 2], v0: [f32; 2], frames: usize) -> Vec<[f32; 2]> {
    let (mut p, mut v) = (p0, v0);
    let mut out = vec![p];
    for _ in 1..frames {
        (p, v) = step(p, v);
        out.push(p);
    }
    out
}

/// The input the forward model sees from three positions: where it is, and its last two velocities.
fn features(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> [f32; 6] {
    [c[0], c[1], (c[0] - b[0]) * VS, (c[1] - b[1]) * VS, (b[0] - a[0]) * VS, (b[1] - a[1]) * VS]
}

/// A learned forward model: 6 → H → H → 2 (the next velocity), ReLU.
pub struct Forward {
    pub w: Vec<Var>,
}

impl Forward {
    pub fn new(hidden: usize, seed: u64) -> Result<Self> {
        let dev = Device::Cpu;
        let mk = |r: usize, c: usize, s: u64| -> Result<Var> {
            Var::from_tensor(&(crate::randn_seeded(r * c, s, &dev)?.reshape((r, c))? * (2.0 / r as f64).sqrt())?)
        };
        let z = |c: usize| Var::from_tensor(&Tensor::zeros(c, candle_core::DType::F32, &dev)?);
        Ok(Self { w: vec![mk(6, hidden, seed)?, z(hidden)?, mk(hidden, hidden, seed + 1)?, z(hidden)?, mk(hidden, 2, seed + 2)?, z(2)?] })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.matmul(self.w[0].as_tensor())?.broadcast_add(self.w[1].as_tensor())?.relu()?;
        let h = h.matmul(self.w[2].as_tensor())?.broadcast_add(self.w[3].as_tensor())?.relu()?;
        h.matmul(self.w[4].as_tensor())?.broadcast_add(self.w[5].as_tensor())
    }

    /// The expected next positions for a batch of three-position windows.
    pub fn predict(&self, wins: &[[[f32; 2]; 3]]) -> Result<Vec<[f32; 2]>> {
        let x: Vec<f32> = wins.iter().flat_map(|w| features(w[0], w[1], w[2])).collect();
        let y = self.forward(&Tensor::from_vec(x, (wins.len(), 6), &Device::Cpu)?)?.to_vec2::<f32>()?;
        Ok(wins.iter().zip(y).map(|(w, v)| [w[2][0] + v[0] / VS, w[2][1] + v[1] / VS]).collect())
    }

    /// Learns from watching `n_traj` seeded random throws (no laws given). Returns the last mean loss.
    pub fn learn(&mut self, n_traj: usize, steps: usize, batch: usize, lr: f64, seed: u64) -> Result<f32> {
        let mut r = Rng(seed | 1);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..n_traj {
            let t = trajectory(r.throw(), r.velocity(), 60);
            for i in 2..t.len() - 1 {
                x.extend_from_slice(&features(t[i - 2], t[i - 1], t[i]));
                y.extend_from_slice(&[(t[i + 1][0] - t[i][0]) * VS, (t[i + 1][1] - t[i][1]) * VS]);
            }
        }
        let n = y.len() / 2;
        let xs = Tensor::from_vec(x, (n, 6), &Device::Cpu)?;
        let ys = Tensor::from_vec(y, (n, 2), &Device::Cpu)?;
        let mut opt = AdamW::new(self.w.clone(), ParamsAdamW { lr, weight_decay: 0.0, ..Default::default() })?;
        let mut last = 0.0;
        for s in 0..steps {
            let idx: Vec<u32> = (0..batch).map(|_| r.below(n) as u32).collect();
            let idx = Tensor::from_vec(idx, batch, &Device::Cpu)?;
            let loss = (self.forward(&xs.index_select(&idx, 0)?)? - ys.index_select(&idx, 0)?)?.sqr()?.mean_all()?;
            // A cosine decay to a tenth of the rate.
            let f = 0.55 + 0.45 * (std::f64::consts::PI * s as f64 / steps as f64).cos();
            opt.set_learning_rate(lr * f);
            opt.backward_step(&loss)?;
            last = loss.to_scalar::<f32>()?;
        }
        Ok(last)
    }
}

/// What she saw: a position per frame, or None while the ball is not in sight.
pub type Seen = Vec<Option<[f32; 2]>>;

/// Watching a scene: she tracks the ball, imagines it while it is out of sight, and adds up her surprise.
/// `kappa` is the surprise of missing a ball she expected to see (counted only when her expectation is
/// clearly outside the screen, by more than `margin`).
pub fn surprise(model: &Forward, seen: &Seen, kappa: f32, margin: f32) -> Result<f32> {
    let mut belief: Vec<[f32; 2]> = Vec::new();
    let mut total = 0.0;
    for (t, s) in seen.iter().enumerate() {
        let expected = if t >= 3 { Some(model.predict(&[[belief[t - 3], belief[t - 2], belief[t - 1]]])?[0]) } else { None };
        let now = match (s, expected) {
            (Some(p), Some(e)) => {
                total += (p[0] - e[0]).powi(2) + (p[1] - e[1]).powi(2);
                *p
            }
            (Some(p), None) => *p,
            (None, Some(e)) => {
                if -depth_in_screen(e) > margin {
                    total += kappa;
                }
                e
            }
            (None, None) => belief.last().copied().unwrap_or([0.5, 0.5]),
        };
        belief.push(now);
    }
    Ok(total)
}

/// What an observer sees of a true path: hidden behind the screen.
pub fn look(path: &[[f32; 2]]) -> Seen {
    path.iter().map(|&p| (!hidden_at(p)).then_some(p)).collect()
}

pub struct Rng(pub u64);
impl Rng {
    pub fn u(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn below(&mut self, n: usize) -> usize {
        ((self.u() * n as f32) as usize).min(n - 1)
    }
    pub fn throw(&mut self) -> [f32; 2] {
        [0.1 + 0.8 * self.u(), 0.3 + 0.6 * self.u()]
    }
    pub fn velocity(&mut self) -> [f32; 2] {
        [-0.6 + 1.2 * self.u(), -0.3 + 0.6 * self.u()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAMES: usize = 60;
    const KAPPA: f32 = 0.02;
    const MARGIN: f32 = 0.05;
    /// Surprise is read in a short clip after the event, as looking time is in infant studies.
    const WINDOW: usize = 10;

    fn clip(mut a: Seen, mut b: Seen, end: usize) -> (Seen, Seen) {
        a.truncate(end.min(FRAMES));
        b.truncate(end.min(FRAMES));
        (a, b)
    }

    fn trained() -> Forward {
        let mut m = Forward::new(64, 7).unwrap();
        let t0 = std::time::Instant::now();
        let loss = m.learn(400, 2500, 512, 3e-3, 11).unwrap();
        eprintln!("learned physics from 400 throws in {:.1} s (last loss {loss:.5})", t0.elapsed().as_secs_f32());
        m
    }

    /// One-step error of the learned model and of "it keeps its velocity" on unseen throws.
    fn one_step_errors(m: &Forward) -> (f32, f32) {
        let mut r = Rng(12345);
        let (mut learned, mut inertia, mut n) = (0.0, 0.0, 0.0);
        for _ in 0..100 {
            let t = trajectory(r.throw(), r.velocity(), FRAMES);
            let wins: Vec<[[f32; 2]; 3]> = (2..FRAMES - 1).map(|i| [t[i - 2], t[i - 1], t[i]]).collect();
            for (k, p) in m.predict(&wins).unwrap().into_iter().enumerate() {
                let i = k + 2;
                let next = t[i + 1];
                learned += (p[0] - next[0]).powi(2) + (p[1] - next[1]).powi(2);
                let q = [2.0 * t[i][0] - t[i - 1][0], 2.0 * t[i][1] - t[i - 1][1]];
                inertia += (q[0] - next[0]).powi(2) + (q[1] - next[1]).powi(2);
                n += 1.0;
            }
        }
        (learned / n, inertia / n)
    }

    #[test]
    fn she_learns_gravity_and_bounces_from_watching() {
        let m = trained();
        let (learned, inertia) = one_step_errors(&m);
        eprintln!("one-step error: learned {learned:.2e}, keep-the-velocity {inertia:.2e}");
        assert!(learned < 0.25 * inertia, "{learned} vs {inertia}");
    }

    /// A throw whose path goes behind the screen and out again, and the frame it first hides at.
    fn throw_behind(r: &mut Rng) -> (Vec<[f32; 2]>, usize, usize) {
        loop {
            let p0 = [0.05 + 0.25 * r.u(), 0.3 + 0.5 * r.u()];
            let v0 = [0.3 + 0.4 * r.u(), -0.2 + 0.4 * r.u()];
            let t = trajectory(p0, v0, FRAMES);
            let hide = (0..FRAMES).find(|&i| hidden_at(t[i]));
            if let Some(h) = hide {
                if let Some(out) = (h..FRAMES).find(|&i| !hidden_at(t[i])) {
                    if h >= 5 && out + WINDOW < FRAMES {
                        return (t, h, out);
                    }
                }
            }
        }
    }

    /// Pairs of (possible, impossible) scenes of one kind.
    fn pairs(kind: &str, n: usize, seed: u64) -> Vec<(Seen, Seen)> {
        let mut r = Rng(seed);
        let mut out = Vec::new();
        while out.len() < n {
            match kind {
                // Continuity: the ball jumps 0.15–0.3 in the open.
                "jump" => {
                    let t = trajectory(r.throw(), r.velocity(), FRAMES);
                    let k = 15 + r.below(25);
                    let d = 0.15 + 0.15 * r.u();
                    let bad: Vec<[f32; 2]> = t.iter().enumerate().map(|(i, p)| if i >= k { [p[0], p[1] + d] } else { *p }).collect();
                    let (a, b) = (look(&t), look(&bad));
                    if (k..FRAMES).all(|i| a[i].is_some() && b[i].is_some()) {
                        out.push(clip(a, b, k + WINDOW));
                    }
                }
                // Solidity: at its first floor bounce, the ball goes on through the floor.
                "floor" => {
                    let t = trajectory(r.throw(), [r.velocity()[0], 0.0], FRAMES);
                    let Some(k) = (1..FRAMES - 8).find(|&i| t[i][1] > t[i - 1][1] && t[i - 1][1] < 0.1) else { continue };
                    let mut bad = t.clone();
                    let mut v = [t[k - 1][0] - t[k - 2][0], t[k - 1][1] - t[k - 2][1]];
                    for i in k..FRAMES {
                        v[1] -= G * DT * DT;
                        bad[i] = [bad[i - 1][0] + v[0], bad[i - 1][1] + v[1]];
                    }
                    // Keep only scenes seen in the open (no screen in the way, no side wall).
                    if t.iter().chain(&bad).all(|p| !hidden_at(*p) && p[0] > 0.02 && p[0] < 0.98) {
                        out.push(clip(look(&t), look(&bad), k + WINDOW));
                    }
                }
                // Inertia: in mid-air the ball turns back with no wall.
                "turn" => {
                    let t = trajectory(r.throw(), r.velocity(), FRAMES);
                    let k = 15 + r.below(25);
                    if (t[k][0] - t[k - 1][0]).abs() < 0.015 {
                        continue;
                    }
                    let mut bad = t[..k].to_vec();
                    let (mut p, mut v) = (t[k - 1], [-(t[k - 1][0] - t[k - 2][0]) * VS, (t[k - 1][1] - t[k - 2][1]) * VS]);
                    for _ in k..FRAMES {
                        (p, v) = step(p, v);
                        bad.push(p);
                    }
                    let (a, b) = (look(&t), look(&bad));
                    if (k..k + 8).all(|i| a[i].is_some() && b[i].is_some()) {
                        out.push(clip(a, b, k + WINDOW));
                    }
                }
                // Permanence: the ball goes behind the screen and never comes out.
                "vanish" => {
                    let (t, h, o) = throw_behind(&mut r);
                    let mut bad = look(&t);
                    for s in bad.iter_mut().skip(h) {
                        *s = None;
                    }
                    out.push(clip(look(&t), bad, o + WINDOW));
                }
                // Spatio-temporal continuity: it comes out of the screen 0.15–0.3 higher than it went in.
                "reappear" => {
                    let (t, _, o) = throw_behind(&mut r);
                    let d = 0.15 + 0.15 * r.u();
                    let mut bad = look(&t);
                    for s in bad.iter_mut().skip(o) {
                        if let Some(p) = s.as_mut() {
                            p[1] += d;
                        }
                    }
                    out.push(clip(look(&t), bad, o + WINDOW));
                }
                _ => unreachable!(),
            }
        }
        out
    }

    #[test]
    fn impossible_events_surprise_her_more_than_possible_ones() {
        let m = trained();
        let t0 = std::time::Instant::now();
        let mut scenes = 0;
        for (kind, seed) in [("jump", 1u64), ("floor", 2), ("turn", 3), ("vanish", 4), ("reappear", 5)] {
            let ps = pairs(kind, 100, seed * 7919 + 1);
            let mut right = 0;
            let (mut sp, mut si) = (0.0, 0.0);
            for (a, b) in &ps {
                let (x, y) = (surprise(&m, a, KAPPA, MARGIN).unwrap(), surprise(&m, b, KAPPA, MARGIN).unwrap());
                right += (y > x) as usize;
                sp += x;
                si += y;
                scenes += 2;
            }
            eprintln!("{kind:>8}: impossible more surprising in {right}/100 pairs (mean surprise {:.4} vs {:.4})", sp / 100.0, si / 100.0);
            assert!(right >= 95, "{kind}: {right}");
        }
        eprintln!("{scenes} scenes watched in {:.2} s", t0.elapsed().as_secs_f32());
    }

    #[test]
    fn behind_the_screen_she_keeps_imagining_the_ball() {
        let m = trained();
        let mut r = Rng(77);
        let (mut err, mut n) = (0.0, 0.0);
        for _ in 0..50 {
            let (t, h, o) = throw_behind(&mut r);
            let seen = look(&t);
            // Rebuild her belief through the hidden frames and compare with where it really was.
            let mut b: Vec<[f32; 2]> = Vec::new();
            for (i, s) in seen.iter().enumerate() {
                let p = match s {
                    Some(p) => *p,
                    None => m.predict(&[[b[i - 3], b[i - 2], b[i - 1]]]).unwrap()[0],
                };
                b.push(p);
            }
            for i in h..o {
                err += ((b[i][0] - t[i][0]).powi(2) + (b[i][1] - t[i][1]).powi(2)).sqrt();
                n += 1.0;
            }
        }
        let err = err / n;
        eprintln!("while hidden, her imagined ball is {err:.4} from the real one (the screen is 0.2 wide)");
        assert!(err < 0.03, "{err}");
    }
}
