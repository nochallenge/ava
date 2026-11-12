// TESTING / DESIGN ONLY
//! ava_parts: Wave 5 brain parts for Ava (from "ava v42 - plan (draft)"), each built and tested here on
//! the CPU before it is switched on in the model (the gpu crate's `arch` options) by a flag.
//!
//! The conventions are the gpu crate's:
//! - weights are [d_in, d_out], applied as h · W;
//! - activity is [batch, time, channels];
//! - softmax is `candle_nn::ops::softmax`, never `softmax_last_dim`: the fused one has no backward pass,
//!   so gradients through it are silently dropped;
//! - every new weight starts at a value where the part changes nothing, so a model with the part
//!   switched on starts exactly as today's and only moves away from it if training finds it useful.
//!
//! Neurons and synapses:
//! - `delays` (E5): learned synaptic delays.
//! - `dendrites` (E6): active dendrites, where context gates the unit.
//! - `resonate` (E7): resonate-and-fire neurons.
//! - `stp` (E9): short-term plasticity (depression and facilitation).
//! - `quant` (E10 int8, Q5 FP8): low-precision weights and numbers.
//! - `event` (E2): event-driven compute, where only spikes cost work.
//!
//! Systems:
//! - `wm` (B9 gated working memory, B3 global workspace): slots with gates, or slots that compete.
//! - `modulate` (B8 neuromodulation, F13 mood): global signals from surprise and valence.
//! - `cerebellum` (H11): a fixed, wide, sparse expansion with a learned readout.
//! - `binding` (D2 role-filler binding, L5 binding by synchrony).
//! - `tem` (B10): structure apart from content, so one visit is enough in a new world.
//!
//! Learning and structure:
//! - `plasticity` (H6 rewiring, H7 critical periods).
//! - `topo` (H12): topographic maps from a spatial loss.
//! - `growth` (A15 growing without changing the function, I3 rebirth of dormant units).
//! - `merge` (C13): averaging, task arithmetic and TIES merging.
//! - `es` (M7): evolution strategies, learning from a score alone.
//! - `sae` (A13): TopK sparse autoencoders to read her features.
//! - `jepa` (B1): predicting her own future states.
//! - `metaplastic` (H1): a plasticity rule found by evolution, not written by hand.
//! - `hyper` (H4 genomic bottleneck, M6 one-pass weight writing): weights made by a smaller network.
//! - `chunking` (A14): learned chunking where the content shifts, instead of a fixed tokenizer.
//! - `diffusion` (N1): an any-order block denoising loss on top of her state, for replay.
//! - `eprop` (E3): learning forward in time with eligibility traces, no backprop through time.
//!
//! Reasoning and self:
//! - `halting` (A7): thinking longer on hard inputs, with a learned stop (PonderNet).
//! - `critic` (B2): a value head and TD errors, like dopamine.
//! - `confidence` (C8, F9): a feeling-of-knowing probe, calibration, and when to answer.
//! - `progress` (A11, B6): practising what she is getting better at (the WP17 rule).
//! - `rlp` (M2): thinking before predicting, rewarded by how much the thought helps.
//! - `reflex` (H3): practised slow answers distilled into the fast pass.
//! - `tvt` (H9): credit across long gaps through the recalled episode.
//! - `coin` (L2): update an old memory or open a new one, by inferred context.
//! - `schema` (F12): a model of her own attention, to notice distraction and steer.
//! - `introspect` (F10): self-reports checked by injecting into the state.
//! - `ttt` (A6): practising on a question's own examples before answering, then letting it go.
//! - `teach` (I7): explanations scored by what a student learns from them.
//! - `debate` (C19): several minds together: votes weighted by honest sureness, and a debate round.
//! - `evolve` (H2): evolving her design with an archive of variants and proxy runs.
//!
//! Senses and the world (Wave 6, small versions):
//! - `columns` (H8): many cortical columns voting on what they sense (Thousand Brains).
//! - `physics` (D10): intuitive physics, a learned forward model surprised by impossible events.
//! - `retina` (K5 vision): light-adapted ON/OFF cells and V1 angle cells; patch tokens for Ava.
//! - `cochlea` (K5 hearing): tonotopic resonators, hair-cell compression and gain control; frame tokens.
//! - `motor` (K5 acting): dopamine actor-critic in a small world, with a learned world model and replay.
//! - `turns` (G11): live turn-taking, knowing when the other has finished from the cues and the silence.
//! - `readout`: a learned linear readout, to test what a sense makes easy to learn.

pub mod binding;
pub mod cerebellum;
pub mod chunking;
pub mod cochlea;
pub mod coin;
pub mod columns;
pub mod confidence;
pub mod critic;
pub mod debate;
pub mod delays;
pub mod dendrites;
pub mod diffusion;
pub mod eprop;
pub mod es;
pub mod event;
pub mod evolve;
pub mod growth;
pub mod halting;
pub mod holo;
pub mod hyper;
pub mod introspect;
pub mod jepa;
pub mod merge;
pub mod metaplastic;
pub mod modulate;
pub mod motor;
pub mod physics;
pub mod plasticity;
pub mod progress;
pub mod quant;
pub mod readout;
pub mod reflex;
pub mod resonate;
pub mod retina;
pub mod rlp;
pub mod sae;
pub mod schema;
pub mod stp;
pub mod sync;
pub mod teach;
pub mod tem;
pub mod topo;
pub mod ttt;
pub mod turns;
pub mod tvt;
pub mod wm;

use candle_core::{Device, Result, Tensor};

/// `n` standard-normal-ish values from a seed (a sum of 4 uniforms; the same seed gives the same values).
pub fn randn_seeded(n: usize, seed: u64, dev: &Device) -> Result<Tensor> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let v: Vec<f32> = (0..n)
        .map(|_| {
            let mut u = 0.0;
            for _ in 0..4 {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                u += (s >> 11) as f64 / (1u64 << 53) as f64;
            }
            ((u - 2.0) * 3f64.sqrt()) as f32
        })
        .collect();
    Tensor::from_vec(v, n, dev)
}

/// x [.., d_in] · w [d_in, d_out] as one 2-D matmul.
pub fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let d_in = *dims.last().unwrap();
    let mut out = dims.clone();
    *out.last_mut().unwrap() = w.dim(1)?;
    x.reshape((x.elem_count() / d_in, d_in))?.matmul(w)?.reshape(out)
}

/// A spike with a surrogate gradient. Forward: 1 where `x` >= 0, else 0 (`x` = v - theta). Backward: the
/// fast sigmoid 1 / (1 + slope·|x|)^2, the same surrogate as the LIF kernel. It is built as
/// f(x) + (H(x) - f(x)) with the bracket detached, where f(x) = x / (1 + slope·|x|) has exactly that slope.
pub fn spike(x: &Tensor, slope: f64) -> Result<Tensor> {
    let f = x.broadcast_div(&x.abs()?.affine(slope, 1.0)?)?;
    let h = x.ge(0f64)?.to_dtype(x.dtype())?;
    (h - f.detach())? + f
}

/// logit(p) = ln(p / (1 - p)).
pub fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

#[cfg(test)]
pub(crate) mod testkit {
    use candle_core::{DType, Device, Tensor, Var};
    use candle_nn::{VarBuilder, VarMap};

    pub fn vm() -> (VarMap, VarBuilder<'static>) {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        (vm, vb)
    }

    /// Sets a weight by name.
    pub fn set(vm: &VarMap, name: &str, t: &Tensor) {
        let d = vm.data().lock().unwrap();
        d.get(name).unwrap_or_else(|| panic!("no var {name}")).set(t).unwrap();
    }

    pub fn var(vm: &VarMap, name: &str) -> Var {
        vm.data().lock().unwrap().get(name).unwrap().clone()
    }

    /// The gradient norm of each named weight after `loss.backward()`.
    pub fn grad_norms(vm: &VarMap, loss: &Tensor, names: &[&str]) -> Vec<f32> {
        let g = loss.backward().unwrap();
        names
            .iter()
            .map(|n| match g.get(var(vm, n).as_tensor()) {
                Some(t) => t.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar::<f32>().unwrap(),
                None => 0.0,
            })
            .collect()
    }

    pub fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
    }

    /// Replaces named weights with seeded noise of the given spread, so a test that trains gives the same
    /// numbers on every run (candle's own CPU random start cannot be seeded).
    pub fn seed(vm: &VarMap, spreads: &[(&str, f64)], seed: u64) {
        for (k, &(name, sd)) in spreads.iter().enumerate() {
            let v = var(vm, name);
            let t = (randn(v.dims(), seed + k as u64) * sd).unwrap();
            v.set(&t).unwrap();
        }
    }

    pub fn randn(shape: &[usize], seed: u64) -> Tensor {
        let n: usize = shape.iter().product();
        crate::randn_seeded(n, seed, &Device::Cpu).unwrap().reshape(shape).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn spike_is_a_step_forward_and_fast_sigmoid_backward() {
        let x = Var::new(&[-1.0f32, -0.1, 0.0, 0.5], &Device::Cpu).unwrap();
        let s = spike(x.as_tensor(), 4.0).unwrap();
        let v = s.to_vec1::<f32>().unwrap();
        for (a, b) in v.iter().zip([0.0, 0.0, 1.0, 1.0]) {
            assert!((a - b).abs() < 1e-6, "{v:?}");
        }
        let g = s.sum_all().unwrap().backward().unwrap();
        let g = g.get(x.as_tensor()).unwrap().to_vec1::<f32>().unwrap();
        for (gi, xi) in g.iter().zip([-1.0f32, -0.1, 0.0, 0.5]) {
            let want = 1.0 / (1.0 + 4.0 * xi.abs()).powi(2);
            assert!((gi - want).abs() < 1e-5, "{g:?}");
        }
    }

    #[test]
    fn linear_flattens_leading_dims() {
        let x = testkit::randn(&[2, 3, 4], 1);
        let w = testkit::randn(&[4, 5], 2);
        let y = linear(&x, &w).unwrap();
        assert_eq!(y.dims(), &[2, 3, 5]);
        let want = x.broadcast_matmul(&w).unwrap();
        assert!(testkit::max_diff(&y, &want) < 1e-5);
    }
}
