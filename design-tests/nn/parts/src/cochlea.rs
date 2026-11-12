// TESTING / DESIGN ONLY
//! K5 hearing, a small version: a cochlea. The basilar membrane is a bank of resonators, each tuned to one
//! pitch, laid out from low to high on a log scale (tonotopy, von Békésy). Hair cells rectify and compress
//! the motion (loudness grows as about the 0.3 power of intensity). The olivocochlear system turns the gain
//! down when it is loud, so the pattern across pitches, not the loudness, carries what is said:
//!   band_k = cascade of 2 band-pass resonators at f_k (constant Q)
//!   rate_k(frame) = (mean |band_k| over 10 ms)^0.3 / mean over k        (the gain control)
//! Each 10 ms frame is one input token for Ava (a learned projection to her width), as a sound model reads
//! a spectrogram; silence gives no events at all.

pub const RATE: f32 = 8000.0;
pub const FRAME: usize = 80;

struct Biquad {
    b0: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl Biquad {
    /// A band-pass with a peak gain of 1 at `f` and quality `q` (the RBJ cookbook form).
    fn band(f: f32, q: f32) -> Self {
        let w = 2.0 * std::f32::consts::PI * f / RATE;
        let alpha = w.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self { b0: alpha / a0, b2: -alpha / a0, a1: -2.0 * w.cos() / a0, a2: (1.0 - alpha) / a0 }
    }
    fn run(&self, x: &[f32]) -> Vec<f32> {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        x.iter()
            .map(|&v| {
                let y = self.b0 * v + self.b2 * x2 - self.a1 * y1 - self.a2 * y2;
                (x2, x1, y2, y1) = (x1, v, y1, y);
                y
            })
            .collect()
    }
}

pub struct Cochlea {
    pub centres: Vec<f32>,
    filters: Vec<Biquad>,
    /// Below this mean |motion| in a frame, the frame is silent (no events).
    pub quiet: f32,
}

impl Cochlea {
    /// `bands` resonators from `lo` to `hi` Hz, log-spaced, quality `q`.
    pub fn new(bands: usize, lo: f32, hi: f32, q: f32) -> Self {
        let centres: Vec<f32> = (0..bands).map(|k| lo * (hi / lo).powf(k as f32 / (bands - 1) as f32)).collect();
        let filters = centres.iter().map(|&f| Biquad::band(f, q)).collect();
        Self { centres, filters, quiet: 1e-7 }
    }

    /// Hair-cell rates before the gain control: [frames][bands], (mean |motion|)^0.3.
    pub fn raw(&self, wave: &[f32]) -> Vec<Vec<f32>> {
        let n = wave.len() / FRAME;
        let bands: Vec<Vec<f32>> = self.filters.iter().map(|f| f.run(&f.run(wave))).collect();
        (0..n)
            .map(|t| bands.iter().map(|b| (b[t * FRAME..(t + 1) * FRAME].iter().map(|v| v.abs()).sum::<f32>() / FRAME as f32).powf(0.3)).collect())
            .collect()
    }

    /// The cochleagram after the gain control: one token per 10 ms frame; silent frames are all 0.
    pub fn frames(&self, wave: &[f32]) -> Vec<Vec<f32>> {
        self.raw(wave)
            .into_iter()
            .map(|f| {
                let m = f.iter().sum::<f32>() / f.len() as f32;
                if m < self.quiet.powf(0.3) {
                    vec![0.0; f.len()]
                } else {
                    f.iter().map(|v| v / m).collect()
                }
            })
            .collect()
    }

    /// The mean of the frames: what was heard over the whole sound.
    pub fn heard(&self, wave: &[f32]) -> Vec<f32> {
        let fr = self.frames(wave);
        let mut out = vec![0.0; self.centres.len()];
        for f in &fr {
            for (o, v) in out.iter_mut().zip(f) {
                *o += v / fr.len() as f32;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readout::Readout;
    use std::f32::consts::PI;

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    fn ear() -> Cochlea {
        Cochlea::new(24, 100.0, 3500.0, 4.0)
    }

    fn tone(f: f32, amp: f32, secs: f32) -> Vec<f32> {
        (0..(RATE * secs) as usize).map(|i| amp * (2.0 * PI * f * i as f32 / RATE).sin()).collect()
    }

    /// The five vowels' formants (Peterson & Barney 1952, adult male means).
    const VOWELS: [(&str, [f32; 3]); 5] =
        [("a", [730.0, 1090.0, 2440.0]), ("i", [270.0, 2290.0, 3010.0]), ("u", [300.0, 870.0, 2240.0]), ("e", [530.0, 1840.0, 2480.0]), ("o", [570.0, 840.0, 2410.0])];

    /// A spoken vowel: harmonics of the voice pitch f0, shaped by the formants, at loudness `amp`.
    fn vowel(v: usize, f0: f32, amp: f32, r: &mut Rng) -> Vec<f32> {
        let fm = VOWELS[v].1;
        let n = (0.2 * RATE) as usize;
        let mut w = vec![0.0; n];
        let mut h = 1.0;
        while h * f0 < 3800.0 {
            let f = h * f0;
            let g: f32 = fm.iter().enumerate().map(|(k, &c)| (1.0 - 0.3 * k as f32) / (1.0 + ((f - c) / 90.0).powi(2))).sum();
            let ph = 2.0 * PI * r.u();
            for (i, s) in w.iter_mut().enumerate() {
                *s += amp * g * (2.0 * PI * f * i as f32 / RATE + ph).sin() / h.sqrt();
            }
            h += 1.0;
        }
        w
    }

    #[test]
    fn each_pitch_has_its_own_place() {
        let e = ear();
        for f in [150.0f32, 300.0, 600.0, 1000.0, 1500.0, 2200.0, 3000.0] {
            let h = e.heard(&tone(f, 0.5, 0.1));
            let best = (0..h.len()).max_by(|&a, &b| h[a].total_cmp(&h[b])).unwrap();
            let nearest = (0..e.centres.len()).min_by(|&a, &b| (e.centres[a] / f).ln().abs().total_cmp(&(e.centres[b] / f).ln().abs())).unwrap();
            assert!(best.abs_diff(nearest) <= 1, "{f} Hz: band {best} ({:.0} Hz), nearest {nearest}", e.centres[best]);
        }
    }

    #[test]
    fn she_hears_vowels_in_new_voices_at_any_loudness() {
        let e = ear();
        let mut r = Rng(17);
        let t0 = std::time::Instant::now();
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for i in 0..400 {
            let (v, f0, amp) = (i % 5, 100.0 + 100.0 * r.u(), 0.3 + 2.7 * r.u());
            x.push(e.heard(&vowel(v, f0, amp, &mut r)));
            y.push(v);
        }
        let ro = Readout::train(&x, &y, 5, 500, 0.05).unwrap();
        // New voices (higher pitch than any heard in learning), whispered and shouted.
        for amp in [0.01f32, 1.0, 100.0] {
            let (mut tx, mut ty) = (Vec::new(), Vec::new());
            for i in 0..200 {
                let (v, f0) = (i % 5, 205.0 + 45.0 * r.u());
                tx.push(e.heard(&vowel(v, f0, amp, &mut r)));
                ty.push(v);
            }
            let acc = ro.accuracy(&tx, &ty).unwrap();
            eprintln!("vowels in new voices at loudness x{amp}: {:.1}% right (chance 20%)", 100.0 * acc);
            assert!(acc > 0.9, "{amp}: {acc}");
        }
        eprintln!("1000 sounds heard and learned in {:.2} s", t0.elapsed().as_secs_f32());
    }

    #[test]
    fn loudness_does_not_change_the_code_and_silence_is_silent() {
        let e = ear();
        let mut r = Rng(3);
        let w = vowel(1, 150.0, 1.0, &mut r);
        let base = e.frames(&w);
        for k in [0.01f32, 100.0] {
            let s = e.frames(&w.iter().map(|v| v * k).collect::<Vec<_>>());
            let d = base.iter().flatten().zip(s.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(d < 1e-3, "{k}: {d}");
        }
        assert!(e.frames(&vec![0.0; 800]).iter().flatten().all(|&v| v == 0.0));
        assert_eq!(base.len(), 20);
    }
}
