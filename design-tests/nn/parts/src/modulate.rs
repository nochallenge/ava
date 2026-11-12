// TESTING / DESIGN ONLY
//! B8 neuromodulation and F13 mood: slow, global signals that change how the whole network works rather
//! than what it says.
//!
//! B8: three modulators, from how surprising each token was (its loss in nats, known after the token,
//! detached; e.g. from the previous pass or a running prediction). Surprise is z-scored against its own
//! running mean and variance (EMA, rate `rho`):
//! - dopamine-like: `write`, how strongly memories are written now (multiplies the fast-weight write
//!   strength beta), 1 + a_w·tanh(z): learn more from what was unexpected;
//! - noradrenaline-like: `reset`, a push toward forgetting the current context, a_r·sigmoid(z − 2): a
//!   sudden surprise means the situation changed (unexpected uncertainty, Yu & Dayan 2005);
//! - acetylcholine-like: `input`, the weight of new input against memory, 1 + a_i·tanh(z̄), with z̄ the
//!   slow average of z: a run of surprises means the model of the world is unreliable, so rely on input.
//! The gains a_w, a_r and a_i start at 0, so the modulators start at (1, 0, 1): no change.
//!
//! F13 mood: a slow running average of valence (reward, or minus surprise), m_t = rho_m·m_{t-1} +
//! (1 − rho_m)·valence_t, and arousal, the running average of |z|. Both are added to the stream through
//! a vector that starts at 0. A bad patch (many errors) then colours everything a little, like a mood, and
//! fades on its own.

use candle_core::{Result, Tensor};
use candle_nn::{Init, VarBuilder};

/// Running z-scores of a [B, T] surprise sequence (EMA mean and variance; the first token scores 0), and
/// their slow average z̄.
pub fn zscores(surprise: &[Vec<f32>], rho: f32) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (mut z, mut zbar) = (Vec::new(), Vec::new());
    for row in surprise {
        let (mut mean, mut var, mut slow) = (row.first().copied().unwrap_or(0.0), 1.0f32, 0.0f32);
        let (mut zr, mut sr) = (Vec::with_capacity(row.len()), Vec::with_capacity(row.len()));
        for &s in row {
            let zt = (s - mean) / var.sqrt().max(1e-3);
            zr.push(zt);
            slow = rho * slow + (1.0 - rho) * zt;
            sr.push(slow);
            let d = s - mean;
            mean += (1.0 - rho) * d;
            var = rho * (var + (1.0 - rho) * d * d);
        }
        z.push(zr);
        zbar.push(sr);
    }
    (z, zbar)
}

pub struct Neuromod {
    /// [3]: a_w, a_r, a_i.
    pub gains: Tensor,
}

/// Per-token modulators, each [B, T].
pub struct Mods {
    pub write: Tensor,
    pub reset: Tensor,
    pub input: Tensor,
}

impl Neuromod {
    pub fn new(vb: VarBuilder) -> Result<Neuromod> {
        Ok(Neuromod { gains: vb.get_with_hints(3, "gains", Init::Const(0.0))? })
    }

    /// z and z̄ as [B, T] tensors (from `zscores`).
    pub fn forward(&self, z: &Tensor, zbar: &Tensor) -> Result<Mods> {
        let a = |i: usize| self.gains.narrow(0, i, 1);
        Ok(Mods {
            write: (z.tanh()?.broadcast_mul(&a(0)?)? + 1.0)?,
            reset: candle_nn::ops::sigmoid(&(z - 2.0)?)?.broadcast_mul(&a(1)?)?,
            input: (zbar.tanh()?.broadcast_mul(&a(2)?)? + 1.0)?,
        })
    }
}

/// F13: mood and arousal, [B, T] each, from valence and |z|.
pub fn mood(valence: &[Vec<f32>], absz: &[Vec<f32>], rho_m: f32) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let ema = |rows: &[Vec<f32>]| -> Vec<Vec<f32>> {
        rows.iter()
            .map(|r| {
                let mut m = 0.0;
                r.iter().map(|&x| {
                    m = rho_m * m + (1.0 - rho_m) * x;
                    m
                })
                .collect()
            })
            .collect()
    };
    (ema(valence), ema(absz))
}

pub struct Mood {
    /// [2, d]: how mood and arousal colour the stream.
    pub w: Tensor,
}

impl Mood {
    pub fn new(vb: VarBuilder, d: usize) -> Result<Mood> {
        Ok(Mood { w: vb.get_with_hints((2, d), "w", Init::Const(0.0))? })
    }

    /// What to add to the stream [B, T, d], from mood and arousal [B, T].
    pub fn forward(&self, mood: &Tensor, arousal: &Tensor) -> Result<Tensor> {
        let ma = Tensor::stack(&[mood, arousal], 2)?;
        crate::linear(&ma, &self.w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_core::Device;

    #[test]
    fn surprise_spikes_stand_out_and_modulators_start_neutral() {
        let mut row = vec![2.0f32; 40];
        row[30] = 8.0;
        let (z, zbar) = zscores(&[row], 0.9);
        assert!(z[0][30] > 3.0 && z[0][29].abs() < 0.5 && z[0][31] < z[0][30], "{:?}", &z[0][28..33]);
        assert!(zbar[0][30] > zbar[0][29]);

        let (vm, vb) = vm();
        let nm = Neuromod::new(vb).unwrap();
        let zt = Tensor::new(z.clone(), &Device::Cpu).unwrap();
        let zb = Tensor::new(zbar.clone(), &Device::Cpu).unwrap();
        let m = nm.forward(&zt, &zb).unwrap();
        assert!(max_diff(&m.write, &m.write.ones_like().unwrap()) < 1e-7);
        assert!(max_diff(&m.reset, &m.reset.zeros_like().unwrap()) < 1e-7);
        assert!(max_diff(&m.input, &m.input.ones_like().unwrap()) < 1e-7);

        // With gains on, the surprising token gets the strongest write and a reset push.
        set(&vm, "gains", &Tensor::new(&[0.5f32, 1.0, 0.5], &Device::Cpu).unwrap());
        let m = nm.forward(&zt, &zb).unwrap();
        let w = m.write.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let r = m.reset.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(w[30] > 1.45 && (w[20] - 1.0).abs() < 0.1 && r[30] > 0.7 && r[20] < 0.2, "{} {} {} {}", w[30], w[20], r[30], r[20]);
        let loss = (m.write.sum_all().unwrap() + m.reset.sum_all().unwrap()).unwrap();
        assert!(grad_norms(&vm, &loss, &["gains"])[0] > 1e-3);
    }

    #[test]
    fn mood_follows_a_bad_patch_and_fades() {
        let mut v = vec![0f32; 300];
        for x in &mut v[50..100] {
            *x = -1.0;
        }
        let (m, _) = mood(&[v.clone()], &[v.iter().map(|x| x.abs()).collect()], 0.98);
        let m = &m[0];
        assert!(m[99] < -0.5 && m[49] == 0.0 && m[299] > -0.02 && m[299] < 0.0, "{} {}", m[99], m[299]);

        let (vm, vb) = vm();
        let md = Mood::new(vb, 4).unwrap();
        let mt = Tensor::from_vec(m[..10].to_vec(), (1, 10), &Device::Cpu).unwrap();
        let y = md.forward(&mt, &mt.abs().unwrap()).unwrap();
        assert_eq!(y.dims(), &[1, 10, 4]);
        assert!(max_diff(&y, &y.zeros_like().unwrap()) < 1e-9);
        let _ = vm;
    }
}
