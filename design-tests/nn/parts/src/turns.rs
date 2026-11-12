// TESTING / DESIGN ONLY
//! G11 live turn-taking, the deciding part. People answer about 200 ms after the other stops, in every
//! language studied (Stivers et al. 2009), much faster than it takes to plan a reply, so they predict the
//! end of the turn before it comes (Levinson & Torreira 2015). The cues: the pitch moves at the end (falls,
//! or rises for a question), the last word is drawn out, the voice softens, and the words make a whole
//! (her language model's chance that the sentence can end here). A pause inside a turn (thinking, "um")
//! has a level pitch and words that do not yet make a whole. A machine that only waits for silence must
//! choose between cutting in on pauses and long, awkward gaps.
//!
//! She learns P(the turn is over | the cues, the silence so far) from listening (a learned readout), and
//! speaks as soon as it passes θ. The sound itself comes from `crate::cochlea`; speaking aloud and the
//! microphone need audio crates, which are not added (they need the owner's OK to download).

use crate::readout::Readout;
use candle_core::Result;

/// The cues at the end of a word: [pitch movement, lengthening, softening, words make a whole].
pub type Cues = [f32; 4];

/// One pause after a word: its cues, whether the turn was over, and (if not) how long the pause lasted.
#[derive(Clone, Copy)]
pub struct Pause {
    pub cues: Cues,
    pub end: bool,
    pub hold_ms: f32,
}

pub const TICK_MS: f32 = 50.0;
pub const WAIT_MS: f32 = 2000.0;

pub fn input(c: &Cues, silence_ms: f32) -> Vec<f32> {
    vec![c[0], c[1], c[2], c[3], silence_ms / 1000.0]
}

/// Learns from heard pauses: every 50 ms of silence is an example (over, or not yet).
pub fn learn(pauses: &[Pause]) -> Result<Readout> {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for p in pauses {
        let until = if p.end { WAIT_MS } else { p.hold_ms };
        let mut t = 0.0;
        while t < until {
            x.push(input(&p.cues, t));
            y.push(p.end as usize);
            t += TICK_MS;
        }
    }
    Readout::train(&x, &y, 2, 600, 0.05)
}

/// When she starts to speak in this pause (ms of silence), if she does before the other goes on.
pub fn speaks_at(model: &Readout, p: &Pause, theta: f32) -> Result<Option<f32>> {
    let until = if p.end { WAIT_MS } else { p.hold_ms };
    let ticks: Vec<f32> = (0..=(until / TICK_MS) as usize).map(|k| k as f32 * TICK_MS).filter(|&t| t < until).collect();
    let probs = model.probs(&ticks.iter().map(|&t| input(&p.cues, t)).collect::<Vec<_>>())?;
    Ok(ticks.iter().zip(probs).find(|(_, pr)| pr[1] > theta).map(|(&t, _)| t))
}

/// Interruptions (share of within-turn pauses she spoke in) and the mean gap after turn ends (ms; a
/// missed end counts as the full wait).
pub fn score(pauses: &[Pause], at: impl Fn(&Pause) -> Option<f32>) -> (f32, f32) {
    let (mut cut, mut holds, mut gap, mut ends) = (0.0, 0.0, 0.0, 0.0);
    for p in pauses {
        let t = at(p);
        if p.end {
            gap += t.unwrap_or(WAIT_MS);
            ends += 1.0;
        } else {
            cut += t.is_some() as u8 as f32;
            holds += 1.0;
        }
    }
    (cut / holds, gap / ends)
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
        fn n(&mut self, m: f32, s: f32) -> f32 {
            m + s * ((self.u() + self.u() + self.u() + self.u() - 2.0) * 3f32.sqrt())
        }
    }

    /// A conversation: a turn ends after a word one time in five. A quarter of the pauses inside a turn come
    /// after words that already make a whole ("I went home. ... And then").
    fn talk(n: usize, seed: u64) -> Vec<Pause> {
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        (0..n)
            .map(|_| {
                if r.u() < 0.2 {
                    let whole = r.n(0.8, 0.15).clamp(0.0, 1.0);
                    Pause { cues: [r.n(0.8, 0.3).abs(), r.n(1.4, 0.2), r.n(0.7, 0.2), whole], end: true, hold_ms: 0.0 }
                } else {
                    let whole = if r.u() < 0.25 { r.n(0.75, 0.15) } else { r.n(0.3, 0.2) }.clamp(0.0, 1.0);
                    let hold = if r.u() < 0.1 { 600.0 + 600.0 * r.u() } else { (r.n(5.5, 0.5)).exp() };
                    Pause { cues: [r.n(0.0, 0.25).abs(), r.n(1.0, 0.2), r.n(0.3, 0.25), whole], end: false, hold_ms: hold }
                }
            })
            .collect()
    }

    #[test]
    fn she_answers_quickly_without_cutting_in() {
        let model = learn(&talk(3000, 1)).unwrap();
        let test = talk(3000, 2);
        let (cut, gap) = score(&test, |p| speaks_at(&model, p, 0.5).unwrap());
        eprintln!("learned cues: cuts in on {:.1}% of pauses, mean gap {gap:.0} ms", 100.0 * cut);
        // Every rule that waits for a fixed silence is worse on one count or the other.
        for wait in [200.0f32, 300.0, 500.0, 700.0, 1000.0] {
            let (c, g) = score(&test, |p| {
                let until = if p.end { WAIT_MS } else { p.hold_ms };
                (wait < until).then_some(wait)
            });
            eprintln!("  wait {wait:>4} ms of silence: cuts in on {:.1}%, mean gap {g:.0} ms", 100.0 * c);
            assert!(c > cut || g > gap, "{wait}: {c} {g}");
        }
        assert!(cut < 0.05, "{cut}");
        assert!(gap < 400.0, "{gap}");
    }
}
