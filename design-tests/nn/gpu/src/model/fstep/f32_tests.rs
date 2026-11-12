// TESTING / DESIGN ONLY — v51 fast4: the fused block-options step with int8 off (`Model::set_fast4`, f32 products
// in gemm's one-row order, src/fast4/fmm.rs) against `forward_with_state` with int8 off, CPU only.
//
// Pre-set bar (written before the first run): bit for bit. Over 300 tokens with the State carried, every step's
// logits and the final State of the fused step equal the plain State-backed step's (arch_tests.rs's `same`), for Ava
// 0.4's recipe and two other option sets, fused every step and taking turns; and on Ava 0.4b itself (AVA_F4_MODEL).

use super::tests::{cfg, run, same, AVA04};
use crate::state::tests::{build_awake, tokens};

#[test]
fn fast4_the_f32_fused_step_is_forward_with_state_bit_for_bit() {
    let cases = [
        ("Ava 0.4's recipe", AVA04.to_string()),
        ("ternary spikes + ALIF, no momentum, head size 32, plain residual", "--head-size 32 --gates channel --value-residual --kwta 0.25 --lif-scale --spikes ternary --alif".to_string()),
        ("binary spikes, momentum 0.3, sigmoid gate rank 8, dense", "--head-size 64 --gates channel --mem-momentum 0.3 --spikes binary --homeostasis 0.1 --residual dense --gate sigmoid --gate-rank 8".to_string()),
    ];
    for (k, (name, flags)) in cases.iter().enumerate() {
        let (_, m) = build_awake(cfg(flags), 160 + k as u64);
        m.set_fast4(true);
        assert!(m.fused_blocker().is_none(), "{name}: {:?}", m.fused_blocker());
        let toks = tokens(1, 300, ava_data::numbers::VOCAB_GLANCE, 170 + k as u64);
        let want = run(&m, &toks, 0, None, |_| false);
        assert_eq!(want.fused_steps, 0);
        let got = run(&m, &toks, 0, None, |_| true);
        assert_eq!(got.fused_steps, 300, "{name}: every step fused");
        same(name, &want, &got);
        let turns = run(&m, &toks, 0, None, |i| i % 3 != 1);
        same(&format!("{name}, taking turns"), &want, &turns);
        // off again: the int8-off refusal as before
        m.set_fast4(false);
        assert!(m.fused_blocker().is_some_and(|w| w.starts_with("int8 is off")), "{name}");
    }
}

/// AVA_F4_MODEL=path (Ava 0.4b): the f32 fused step against forward_with_state over AVA_F4_TOKENS tokens (300).
#[test]
#[ignore]
fn fast4_the_f32_fused_step_on_a_saved_model() {
    let Ok(path) = std::env::var("AVA_F4_MODEL") else { return };
    let n: usize = std::env::var("AVA_F4_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    let dev = candle_core::Device::Cpu;
    let (cfg, _vm, m) = crate::ckpt::load_auto_lean(std::path::Path::new(&path), crate::model::Kind::D, 4096, &dev).unwrap();
    m.set_track_activity(false);
    m.set_fast4(true);
    assert!(m.fused_blocker().is_none(), "{:?}", m.fused_blocker());
    let toks = tokens(1, n, cfg.vocab, 99);
    let t0 = std::time::Instant::now();
    let want = run(&m, &toks, 0, None, |_| false);
    let t1 = std::time::Instant::now();
    let got = run(&m, &toks, 0, None, |_| true);
    let t2 = std::time::Instant::now();
    assert_eq!(got.fused_steps, n as u64);
    same("Ava 0.4b", &want, &got);
    println!(
        "fast4 f32 fused on {path}: {n} tokens bit for bit; State path {:.1} tok/s, fused f32 {:.1} tok/s ({} threads)",
        n as f64 / (t1 - t0).as_secs_f64(),
        n as f64 / (t2 - t1).as_secs_f64(),
        candle_core::utils::get_num_threads()
    );
}
