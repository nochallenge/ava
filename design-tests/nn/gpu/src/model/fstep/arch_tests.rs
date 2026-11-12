// TESTING / DESIGN ONLY — tests of src/model/fstep/arch.rs (the fused one-token step for block options), CPU only.
//
// Pre-set bar (written before the first run): bit for bit. Over 300 tokens with the State carried, every step's logits
// and the final State (every LIF membrane, spike and ALIF trace, every fast weight and momentum, the engram store and
// its counts, tokens read, the history) of the fused step equal `forward_with_state`'s (the plain State-backed step)
// with int8 on: for Ava 0.4's exact recipe with the glance, the magnitude code and an engram, for the other spike
// kinds, momentum off and the sigmoid gate, with the lesions, with the two steps taking turns, after a batched `feed`,
// and with an oldest-first engram cap, with the memory heads on one thread or several. (The task's fallback bar, max |dlogit| < 1e-4, is weaker; it is not used.) Where
// the fused step does not apply (another option, a least-read cap) it does not run and the step is the plain one.

use crate::arch::Arch;
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, tokens};
use crate::state::{EngramCap, Evict};
use candle_core::{DType, Tensor};

pub(super) const AVA04: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
                     --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";

pub(super) fn bits(t: &Tensor) -> Vec<u32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
}

/// d 128 (two memory heads of 64, four engram heads of 32), 3 blocks, the engram in block 1, glance + magnitude code.
pub(super) fn cfg(flags: &str) -> Config {
    let g = ava_data::numbers::VOCAB_GLANCE;
    Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap().with_arch(Arch::from_flags(flags).unwrap()).unwrap()
}

pub(super) struct Run {
    pub(super) logits: Vec<Tensor>,
    pub(super) state: (Vec<Tensor>, Vec<usize>, Vec<Vec<u32>>),
    pub(super) fused_steps: u64,
}

/// A fresh State-backed stream (with `cap`): the first `feed` tokens read in one batched `feed` (0: none), then every
/// other token stepped, step i fused when `fused(i)`.
pub(super) fn run(m: &Model, toks: &[u32], feed: usize, cap: Option<EngramCap>, fused: impl Fn(usize) -> bool) -> Run {
    let mut s = m.stream().unwrap();
    Model::stream_state_mut(&mut s).expect("State-backed").cap = cap;
    let ran = super::RAN.with(|r| r.get());
    let mut logits = Vec::with_capacity(toks.len());
    if feed > 0 {
        m.set_fused(false);
        logits.push(m.feed(&mut s, &toks[..feed]).unwrap());
    }
    for (i, &t) in toks.iter().enumerate().skip(feed) {
        m.set_fused(fused(i));
        logits.push(m.step(&mut s, t).unwrap());
    }
    m.set_fused(false);
    assert_eq!(s.seen, toks.len());
    Run { logits, state: super::state_tensors(Model::stream_state(&s).unwrap()), fused_steps: super::RAN.with(|r| r.get()) - ran }
}

pub(super) fn same(name: &str, want: &Run, got: &Run) {
    assert_eq!(want.logits.len(), got.logits.len(), "{name}: steps");
    for (i, (a, b)) in want.logits.iter().zip(&got.logits).enumerate() {
        assert_eq!(a.dims(), b.dims(), "{name}: logits shape at step {i}");
        assert!(bits(a) == bits(b), "{name}: logits differ at step {i}");
    }
    let ((wt, wc, wh), (gt, gc, gh)) = (&want.state, &got.state);
    assert_eq!(wt.len(), gt.len(), "{name}: state tensors");
    for (j, (a, b)) in wt.iter().zip(gt).enumerate() {
        assert_eq!(a.dims(), b.dims(), "{name}: state tensor {j} shape");
        assert!(bits(a) == bits(b), "{name}: state tensor {j} differs");
    }
    assert_eq!(wc, gc, "{name}: tokens read and engram slots");
    assert_eq!(wh, gh, "{name}: history");
}

/// The fused step gives forward_with_state's logits and State bit for bit over 300 tokens.
#[test]
fn the_fused_arch_step_is_forward_with_state_bit_for_bit() {
    let cases = [
        ("Ava 0.4's recipe", AVA04.to_string()),
        ("ternary spikes + ALIF, no momentum, head size 32, plain residual", "--head-size 32 --gates channel --value-residual --kwta 0.25 --lif-scale --spikes ternary --alif".to_string()),
        (
            "binary spikes + threshold (lif_infer), momentum 0.3, sigmoid gate rank 8, dense",
            "--head-size 64 --gates channel --mem-momentum 0.3 --spikes binary --homeostasis 0.1 --residual dense --gate sigmoid --gate-rank 8".to_string(),
        ),
        ("binary spikes + ALIF (lif_ref), relu2, slope 2", "--head-size 64 --gates channel --spikes binary --alif --mlp-act relu2 --surrogate-slope 2".to_string()),
    ];
    for (k, (name, flags)) in cases.iter().enumerate() {
        let (_, m) = build_awake(cfg(flags), 60 + k as u64);
        m.set_int8(true).unwrap();
        assert_eq!(m.fused_blocker(), None, "{name}");
        let toks = tokens(1, 300, ava_data::numbers::VOCAB_GLANCE, 110 + k as u64);
        let want = run(&m, &toks, 0, None, |_| false);
        assert_eq!(want.fused_steps, 0, "{name}: the plain run ran plain");
        let got = run(&m, &toks, 0, None, |_| true);
        assert_eq!(got.fused_steps, toks.len() as u64, "{name}: every step ran fused");
        same(name, &want, &got);
        if k == 1 {
            // four memory heads on one thread, and split 2 + 2 + 0 over three (the heads run in parallel)
            for th in [1, 3] {
                m.set_int8_threads(true, th).unwrap();
                let got = run(&m, &toks, 0, None, |_| true);
                assert_eq!(got.fused_steps, toks.len() as u64);
                same(&format!("{name}, {th} threads"), &want, &got);
            }
        }
        if k == 0 {
            // the two steps taking turns on one State
            let turns = run(&m, &toks, 0, None, |i| i % 3 != 1);
            assert_eq!(turns.fused_steps, 200, "{name}: taking turns");
            same(&format!("{name}, taking turns"), &want, &turns);
            // a prompt read in one batched feed (another rounding than one token at a time), then fused steps
            let fw = run(&m, &toks, 40, None, |_| false);
            let fg = run(&m, &toks, 40, None, |_| true);
            assert_eq!(fg.fused_steps, 260);
            same(&format!("{name}, after a batched feed"), &fw, &fg);
            // an oldest-first engram cap of 24 slots
            let cap = Some(EngramCap { max: 24, rule: Evict::Oldest });
            let cw = run(&m, &toks, 0, cap, |_| false);
            let cg = run(&m, &toks, 0, cap, |_| true);
            assert_eq!(cg.fused_steps, 300);
            assert_eq!(cg.state.1, vec![300, 24], "{name}: capped store");
            same(&format!("{name}, engram cap 24 oldest"), &cw, &cg);
            // a least-read cap needs the read counts: the fused step stands aside
            let cap = Some(EngramCap { max: 24, rule: Evict::LeastRead });
            let lw = run(&m, &toks[..60], 0, cap, |_| false);
            let lg = run(&m, &toks[..60], 0, cap, |_| true);
            assert_eq!(lg.fused_steps, 0);
            same(&format!("{name}, least-read cap (plain)"), &lw, &lg);
            // the lesions: the place code, the magnitude code and the engram read off (it keeps writing its store)
            m.set_lesion(true, true, true);
            let want = run(&m, &toks, 0, None, |_| false);
            let got = run(&m, &toks, 0, None, |_| true);
            assert_eq!(got.fused_steps, 300);
            same(&format!("{name}, lesioned"), &want, &got);
            m.set_lesion(false, false, false);
        }
    }
}

/// Options the fused step does not cover leave the step to forward_with_state (it does not run, nothing changes).
#[test]
fn other_options_run_the_plain_step() {
    for flags in ["--head-size 64 --gates head --mem-momentum 0.5", "--head-size 64 --gates channel --token-shift", "--head-size 64 --gates kda"] {
        let (_, m) = build_awake(cfg(flags), 7);
        m.set_int8(true).unwrap();
        assert!(m.fused_blocker().is_some(), "{flags}");
        let toks = tokens(1, 40, ava_data::numbers::VOCAB_GLANCE, 5);
        let want = run(&m, &toks, 0, None, |_| false);
        let got = run(&m, &toks, 0, None, |_| true);
        assert_eq!(got.fused_steps, 0, "{flags}");
        same(flags, &want, &got);
    }
    // int8 off: the fused step is the int8 talking path
    let (_, m) = build_awake(cfg(AVA04), 8);
    assert!(m.fused_blocker().is_some());
    let toks = tokens(1, 20, ava_data::numbers::VOCAB_GLANCE, 6);
    assert_eq!(run(&m, &toks, 0, None, |_| true).fused_steps, 0);
}

/// v51 fast1: --neuromod 3 on Ava 0.4's recipe (the nucleus after block 0; NE, ACh and DA in the later blocks) gives
/// forward_with_state's logits and State bit for bit over 300 tokens (the same bar as above): with the awake weights,
/// with ACh's bias raised by 1.5 (its signal near 1.6, so the [0, 2] clamp acts on most strong writes), the two steps
/// taking turns, and after a batched feed. ACh's raised bias must change the logits (the signals reach the blocks).
#[test]
fn the_fused_arch_step_has_neuromod_bit_for_bit() {
    let flags = format!("{AVA04} --neuromod 3");
    let toks = tokens(1, 300, ava_data::numbers::VOCAB_GLANCE, 131);
    let mut plain_logits: Vec<Vec<Tensor>> = Vec::new();
    for (k, ach) in [0.0f32, 1.5].into_iter().enumerate() {
        let (vm, m) = build_awake(cfg(&flags), 70);
        if ach != 0.0 {
            let data = vm.data().lock().unwrap();
            let bn = &data["nm.bn"];
            let mut v: Vec<f32> = bn.as_tensor().to_vec1().unwrap();
            v[crate::v50::ACH] += ach;
            bn.set(&Tensor::from_vec(v, 3, &candle_core::Device::Cpu).unwrap()).unwrap();
        }
        m.set_int8(true).unwrap();
        assert_eq!(m.fused_blocker(), None, "--neuromod 3 takes the fused step");
        let name = format!("Ava 0.4's recipe + --neuromod 3, ACh bias +{ach}");
        let want = run(&m, &toks, 0, None, |_| false);
        assert_eq!(want.fused_steps, 0);
        let got = run(&m, &toks, 0, None, |_| true);
        assert_eq!(got.fused_steps, 300, "{name}: every step ran fused");
        same(&name, &want, &got);
        if k == 1 {
            let turns = run(&m, &toks, 0, None, |i| i % 3 != 1);
            assert_eq!(turns.fused_steps, 200);
            same(&format!("{name}, taking turns"), &want, &turns);
            let fw = run(&m, &toks, 40, None, |_| false);
            let fg = run(&m, &toks, 40, None, |_| true);
            assert_eq!(fg.fused_steps, 260);
            same(&format!("{name}, after a batched feed"), &fw, &fg);
        }
        plain_logits.push(want.logits);
    }
    let differ = plain_logits[0].iter().zip(&plain_logits[1]).filter(|(a, b)| bits(a) != bits(b)).count();
    assert!(differ > 250, "ACh's bias changed only {differ} of 300 steps' logits");
    // --dale and --loop still run the plain step
    for extra in ["--dale 0.8", "--loop 2"] {
        let (_, m) = build_awake(cfg(&format!("{flags} {extra}")), 71);
        m.set_int8(true).unwrap();
        assert!(m.fused_blocker().is_some(), "{extra}");
    }
}
