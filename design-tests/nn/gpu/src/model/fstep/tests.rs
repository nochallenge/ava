// TESTING / DESIGN ONLY — tests of src/model/fstep.rs (the fused one-token CPU step), CPU only.
//
// Pre-set bar: bit for bit. Over 200 tokens with the state carried, every step's logits and the final carried state
// (membranes, spikes, fast weights, attention and engram stores, place and history) of the fused step equal the
// plain step's, for every kind of today's block, with the glance, magnitude code and engram on, with lesions, and with
// the two steps taking turns on one stream. Where the fused step does not apply it changes nothing.

use super::super::{Config, Kind, Model};
use crate::state::tests::{build_awake, tokens};
use candle_core::{DType, Tensor};

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
}

type Run = (Vec<Tensor>, Vec<Tensor>, (usize, u32, u32, Vec<u32>));

/// Step `toks` on a fresh stream, step i fused when `fused(i)`: every step's logits, the carried state's tensors and
/// its scalars.
fn run(m: &Model, toks: &[u32], fused: impl Fn(usize) -> bool) -> Run {
    let mut s = m.stream().unwrap();
    let mut logits = Vec::with_capacity(toks.len());
    for (i, &t) in toks.iter().enumerate() {
        m.set_fused(fused(i));
        logits.push(m.step(&mut s, t).unwrap());
    }
    m.set_fused(false);
    (logits, s.state_tensors(), s.state_scalars())
}

fn same(name: &str, want: &Run, got: &Run) {
    assert_eq!(want.0.len(), got.0.len(), "{name}: steps");
    for (i, (a, b)) in want.0.iter().zip(&got.0).enumerate() {
        assert_eq!(a.dims(), b.dims(), "{name}: logits shape at step {i}");
        assert!(bits(a) == bits(b), "{name}: logits differ at step {i}");
    }
    assert_eq!(want.1.len(), got.1.len(), "{name}: state tensors");
    for (j, (a, b)) in want.1.iter().zip(&got.1).enumerate() {
        assert_eq!(a.dims(), b.dims(), "{name}: state tensor {j} shape");
        assert!(bits(a) == bits(b), "{name}: state tensor {j} differs");
    }
    assert_eq!(want.2, got.2, "{name}: state scalars");
}

fn cases() -> Vec<(&'static str, Config, usize)> {
    let gate = |r: usize| crate::arch::Arch::from_flags(&format!("--gate sigmoid --gate-rank {r}")).unwrap();
    let g = ava_data::numbers::VOCAB_GLANCE;
    vec![
        ("D + glance + magnitude + engram, d64", Config::with_kind(Kind::D, g, 64, 2, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap(), g),
        ("D, d96 (3 heads)", Config::with_kind(Kind::D, 258, 96, 2, 64), 258),
        ("S", Config::with_kind(Kind::S, 258, 64, 2, 64), 258),
        ("F", Config::with_kind(Kind::F, 258, 64, 2, 64), 258),
        ("G", Config::with_kind(Kind::G, 258, 64, 2, 64), 258),
        ("R", Config::with_kind(Kind::R, 258, 64, 2, 64), 258),
        ("A, 4 blocks", Config::with_kind(Kind::A, 258, 64, 4, 64), 258),
        ("D + engram in block 0, 3 blocks", Config::with_kind(Kind::D, 258, 64, 3, 64).with_engram(0).unwrap(), 258),
        // v44 K2: the low-rank sigmoid output gate (its Wd, Wu stay f32), rank 8 and the default 32
        ("D + sigmoid gate rank 8, d64", Config::with_kind(Kind::D, 258, 64, 2, 64).with_arch(gate(8)).unwrap(), 258),
        ("D + sigmoid gate rank 32, d96", Config::with_kind(Kind::D, 258, 96, 2, 64).with_arch(gate(32)).unwrap(), 258),
    ]
}

/// The fused step gives the plain step's logits and carried state bit for bit over 200 tokens, for every kind.
#[test]
fn the_fused_step_is_the_step_bit_for_bit() {
    for (k, (name, cfg, vocab)) in cases().into_iter().enumerate() {
        let (_, m) = build_awake(cfg, 40 + k as u64);
        m.set_int8(true).unwrap();
        assert_eq!(m.fused_blocker(), None, "{name}");
        let toks = tokens(1, 200, vocab, 90 + k as u64);
        let want = run(&m, &toks, |_| false);
        let ran = super::RAN.with(|r| r.get());
        let got = run(&m, &toks, |_| true);
        assert_eq!(super::RAN.with(|r| r.get()) - ran, toks.len() as u64, "{name}: every step ran fused");
        same(name, &want, &got);
        // the two steps taking turns on one stream
        same(&format!("{name}, taking turns"), &want, &run(&m, &toks, |i| i % 3 != 1));
        // the lesions (place, magnitude code, engram read off)
        if cfg.glance || cfg.engram.is_some() {
            m.set_lesion(true, true, true);
            same(&format!("{name}, lesioned"), &run(&m, &toks[..80], |_| false), &run(&m, &toks[..80], |_| true));
            m.set_lesion(false, false, false);
        }
        // and the int8 thread count of the set (the kernels' bits do not depend on it)
        m.set_int8_threads(true, 3).unwrap();
        same(&format!("{name}, 3 threads"), &want, &run(&m, &toks, |_| true));
    }
}

/// Where the fused step does not apply (int8 off, one head, block options), switching it on changes nothing.
#[test]
fn the_fused_step_stays_out_where_it_does_not_apply() {
    let (_, m) = build_awake(Config::with_kind(Kind::D, 258, 64, 2, 64), 3);
    assert!(m.fused_blocker().is_some(), "int8 is off");
    let toks = tokens(1, 40, 258, 5);
    same("int8 off", &run(&m, &toks, |_| false), &run(&m, &toks, |_| true));
    let (_, one) = build_awake(Config::with_kind(Kind::D, 258, 32, 2, 64), 4);
    one.set_int8(true).unwrap();
    assert!(one.fused_blocker().is_some(), "one head");
    same("one head", &run(&one, &toks, |_| false), &run(&one, &toks, |_| true));
    let ar = crate::arch::Arch::from_flags("--gates channel --kwta 1/4").unwrap();
    let (_, a) = build_awake(Config::with_kind(Kind::D, 258, 64, 2, 64).with_arch(ar).unwrap(), 6);
    a.set_int8(true).unwrap();
    // Block options now have their own fused step (src/model/fstep/arch.rs), so this model is no longer blocked; it
    // must still equal the plain step bit for bit, and the plain fused step (super::RAN) must not run for it.
    #[cfg(target_arch = "x86_64")]
    if crate::int8::simd_ok() && !std::arch::is_x86_feature_detected!("avx512f") {
        assert!(a.fused_blocker().is_none(), "block options: {:?}", a.fused_blocker());
    }
    same("block options", &run(&a, &toks, |_| false), &run(&a, &toks, |_| true));
    assert_eq!(super::RAN.with(|r| r.get()), 0, "nothing ran on the plain fused step");
}

/// The fused step's small f32 product (every vector-width path: 64, 32 and 8 outputs at a time and the masked last
/// ones) is candle's one-row matmul bit for bit, for 2 outputs or more.
#[test]
#[cfg(target_arch = "x86_64")]
fn the_small_product_is_candles_one_row_matmul() {
    if !crate::int8::simd_ok() || std::arch::is_x86_feature_detected!("avx512f") {
        return; // the fused step never runs here (fused_blocker)
    }
    let dev = candle_core::Device::Cpu;
    for k in [1usize, 2, 3, 7, 64, 384] {
        for n in [2usize, 3, 6, 8, 9, 31, 32, 33, 40, 64, 65, 96, 130] {
            let x = crate::testkit::unit_noise(k, 7 + k as u64);
            let w = crate::testkit::unit_noise(k * n, 11 + (k * n) as u64);
            let want = Tensor::from_vec(x.clone(), (1, k), &dev).unwrap().matmul(&Tensor::from_vec(w.clone(), (k, n), &dev).unwrap()).unwrap();
            let mut got = vec![0f32; n];
            super::gemv(&x, &w, n, &mut got);
            let got = Tensor::from_vec(got, (1, n), &dev).unwrap();
            assert!(bits(&want) == bits(&got), "k {k}, n {n}");
        }
    }
}
