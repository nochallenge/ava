// TESTING / DESIGN ONLY
//! Low-precision numbers. Brains compute with noisy, low-resolution synapses, and so can Ava.
//! - E10 int8 weights: each output column keeps one f32 scale and its weights as int8 (symmetric,
//!   round to nearest). That is 4x smaller weights for serving her, and integer-friendly sums.
//! - Q5 FP8: exact emulation of the two 8-bit float formats used in training (E4M3 and E5M2, OCP FP8
//!   spec), with round-to-nearest-even, subnormals and saturation (no infinities). An FP8 training option
//!   can then be tested on the CPU for what it changes, before any kernel.

/// One weight matrix [rows = d_in, cols = d_out] in int8 with a scale per output column.
#[derive(Clone, Debug)]
pub struct Int8 {
    pub q: Vec<i8>,
    pub scale: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

pub fn quantize_int8(w: &[f32], rows: usize, cols: usize) -> Int8 {
    assert_eq!(w.len(), rows * cols);
    let mut scale = vec![0f32; cols];
    for r in 0..rows {
        for c in 0..cols {
            scale[c] = scale[c].max(w[r * cols + c].abs());
        }
    }
    for s in &mut scale {
        *s = if *s > 0.0 { *s / 127.0 } else { 1.0 };
    }
    let q = (0..rows * cols).map(|i| (w[i] / scale[i % cols]).round().clamp(-127.0, 127.0) as i8).collect();
    Int8 { q, scale, rows, cols }
}

impl Int8 {
    pub fn dequant(&self) -> Vec<f32> {
        self.q.iter().enumerate().map(|(i, &q)| q as f32 * self.scale[i % self.cols]).collect()
    }

    /// y = x · W for one input row, with integer-weight products summed in f32, then scaled once per column.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        let mut acc = vec![0f32; self.cols];
        for (r, &xr) in x.iter().enumerate() {
            if xr == 0.0 {
                continue;
            }
            let row = &self.q[r * self.cols..(r + 1) * self.cols];
            for (a, &q) in acc.iter_mut().zip(row) {
                *a += xr * q as f32;
            }
        }
        acc.iter().zip(&self.scale).map(|(a, s)| a * s).collect()
    }

    pub fn bytes(&self) -> usize {
        self.q.len() + 4 * self.scale.len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fp8 {
    E4M3,
    E5M2,
}

impl Fp8 {
    /// (mantissa bits, smallest normal exponent, largest finite value)
    fn spec(self) -> (i32, i32, f32) {
        match self {
            Fp8::E4M3 => (3, -6, 448.0),
            Fp8::E5M2 => (2, -14, 57344.0),
        }
    }

    pub fn max(self) -> f32 {
        self.spec().2
    }
}

/// x rounded to the nearest value of the format (ties to even), saturating at ± the largest finite value.
pub fn round_fp8(x: f32, f: Fp8) -> f32 {
    if x.is_nan() || x == 0.0 {
        return x;
    }
    let (man, emin, max) = f.spec();
    let a = x.abs();
    if a >= max {
        return max.copysign(x);
    }
    let e = (a.log2().floor() as i32).max(emin);
    let step = (2f32).powi(e - man);
    let q = (a / step).round_ties_even() * step;
    q.min(max).copysign(x)
}

/// Per-tensor scaled FP8 (as in FP8 training): scale so the largest |x| maps to the format's largest
/// value, round, and scale back. Returns the values and the scale.
pub fn scaled_fp8(x: &[f32], f: Fp8) -> (Vec<f32>, f32) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let s = if amax > 0.0 { f.max() / amax } else { 1.0 };
    (x.iter().map(|&v| round_fp8(v * s, f) / s).collect(), s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        crate::testkit::randn(&[n], seed).to_vec1().unwrap()
    }

    #[test]
    fn int8_is_close_four_times_smaller_and_matvec_matches() {
        let (r, c) = (64, 48);
        let w = rnd(r * c, 11);
        let q = quantize_int8(&w, r, c);
        let d = q.dequant();
        for i in 0..w.len() {
            assert!((w[i] - d[i]).abs() <= q.scale[i % c] * 0.5 + 1e-7);
        }
        assert!(q.bytes() * 3 < w.len() * 4);
        let x = rnd(r, 12);
        let y = q.matvec(&x);
        for col in 0..c {
            let want: f32 = (0..r).map(|k| x[k] * d[k * c + col]).sum();
            assert!((y[col] - want).abs() < 1e-3, "{} {}", y[col], want);
        }
    }

    #[test]
    fn fp8_rounds_like_the_spec() {
        use Fp8::*;
        let cases = [
            (E4M3, 1.0, 1.0),
            (E4M3, 1.0625, 1.0),  // a tie: to the even mantissa
            (E4M3, 1.1875, 1.25), // a tie: to the even mantissa, upward
            (E4M3, 448.0, 448.0),
            (E4M3, 500.0, 448.0), // saturates
            (E4M3, -3.3, -3.25),
            (E4M3, 2f32.powi(-9), 2f32.powi(-9)), // the smallest subnormal
            (E4M3, 2f32.powi(-11), 0.0),
            (E5M2, 1.125, 1.0),
            (E5M2, 1.0e6, 57344.0),
            (E5M2, 2f32.powi(-16), 2f32.powi(-16)), // the smallest subnormal
        ];
        for (f, x, want) in cases {
            assert_eq!(round_fp8(x, f), want, "{f:?} {x}");
        }
        // Every value is a fixed point of the rounding.
        for x in [0.015625f32, 0.875, 6.5, 240.0] {
            assert_eq!(round_fp8(round_fp8(x, E4M3), E4M3), round_fp8(x, E4M3));
        }
        // Scaled E4M3 keeps a normal vector to a few percent.
        let v = rnd(4096, 13);
        let (q, _) = scaled_fp8(&v, E4M3);
        let rel = (v.iter().zip(&q).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / v.iter().map(|a| a * a).sum::<f32>()).sqrt();
        assert!(rel < 0.04, "{rel}");
    }
}
