// TESTING / DESIGN ONLY — tests of src/model/fstep/arch_f2.rs (v51 fast2: R and TH in the fused talking step), CPU.
//
// Pre-set bar (arch_tests.rs's): bit for bit. Over 300 tokens with the State carried, every step's logits and the final
// State of the fused step equal forward_with_state's with int8 on, for Ava 0.4's recipe + --letters, + --thalamus and
// + both (d128, 3 blocks, the engram in block 1; vocabulary 293 with a letter table of this test's own), the parts'
// weights that start at 0 moved off it (build_awake); the two steps taking turns; and with the chat app's part signals
// on, the relay's gates they show the same as the plain step's. W still keeps the plain step (the blocker; CB: v51 fast3).

use crate::arch::Arch;
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, tokens};
use candle_core::{DType, Tensor};

const AVA04: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
                     --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";
const V: usize = 293;

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
}

fn cfg(extra: &str) -> Config {
    Config::with_kind(Kind::D, V, 128, 3, 64).with_engram(1).unwrap().with_arch(Arch::from_flags(&format!("{AVA04} {extra}")).unwrap()).unwrap()
}

fn letters() {
    let r = crate::testkit::unit_noise(V * 22, 5);
    let t: Vec<Vec<u8>> = (0..V)
        .map(|v| {
            let len = ((r[v * 22] + 1.0) * 10.5) as usize % 21;
            (0..len).map(|j| (((r[v * 22 + 1 + j] + 1.0) * 128.0) as usize).min(255) as u8).collect()
        })
        .collect();
    crate::v51::set_letters(t);
}

/// Logits of every step, the final State and the part signals' gates after each step; step i fused when `fused(i)`.
fn run(m: &Model, toks: &[u32], fused: impl Fn(usize) -> bool) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Vec<[f32; 2]>>, u64) {
    let mut s = m.stream().unwrap();
    let ran = super::super::RAN.with(|r| r.get());
    let (mut logits, mut sig) = (Vec::new(), Vec::new());
    for (i, &t) in toks.iter().enumerate() {
        m.set_fused(fused(i));
        logits.push(bits(&m.step(&mut s, t).unwrap()));
        sig.push(m.part_signals().th);
    }
    m.set_fused(false);
    let st = super::super::state_tensors(Model::stream_state(&s).unwrap());
    (logits, st.0.iter().map(bits).collect(), sig, super::super::RAN.with(|r| r.get()) - ran)
}

#[test]
fn letters_and_relay_in_the_fused_talking_step_bit_for_bit() {
    letters();
    for (k, extra) in ["--letters", "--thalamus", "--letters --thalamus"].iter().enumerate() {
        let (_, m) = build_awake(cfg(extra), 70 + k as u64);
        m.set_int8(true).unwrap();
        assert_eq!(m.fused_blocker(), None, "{extra}: the fused talking step takes it");
        let toks = tokens(1, 300, V, 120 + k as u64);
        for sig in [false, true] {
            m.set_part_signals(sig);
            let (wl, ws, wsig, wn) = run(&m, &toks, |_| false);
            assert_eq!(wn, 0, "{extra}: the plain run ran plain");
            let (gl, gs, gsig, gn) = run(&m, &toks, |_| true);
            assert_eq!(gn, toks.len() as u64, "{extra}: every step ran fused");
            assert!(wl == gl, "{extra} (signals {sig}): logits differ");
            assert!(ws == gs, "{extra} (signals {sig}): the State differs");
            if sig && extra.contains("thalamus") {
                assert!(wsig == gsig, "{extra}: the relay's gates in the part signals differ");
                assert_eq!(gsig.last().map(|g| g.len()), Some(3), "{extra}: one pair of gates per block");
                assert!(gsig.last().unwrap().iter().all(|g| g[0] != 1.0 && g[1] != 1.0), "{extra}: the gates moved off 1");
            }
            let (tl, ts, _, tn) = run(&m, &toks, |i| i % 3 != 1);
            assert_eq!(tn, 200, "{extra}: taking turns");
            assert!(wl == tl && ws == ts, "{extra} (signals {sig}): taking turns differs");
        }
        m.set_part_signals(false);
        m.set_int8(false).unwrap();
    }
    // v51 fast3 (arch_f3.rs): CB now takes the fused step too; W still keeps the plain step
    for extra in ["--columns 4", "--letters --thalamus --columns 4"] {
        let (_, m) = build_awake(cfg(extra), 80);
        m.set_int8(true).unwrap();
        assert!(m.fused_blocker().is_some(), "{extra}: still the plain step");
        m.set_int8(false).unwrap();
    }
}

/// merge_f (runs/v51/merge_f.md): v51 fast1's --neuromod 3 with v51 fast2's TH, and with R and TH, in the fused talking
/// step, at this file's bar: over 300 tokens every step's logits and the final State equal forward_with_state's with
/// int8 on, also with the two steps taking turns; with the chat app's part signals on (cleared before every step), the
/// relay's gates and the nucleus's three signals (the midbrain) after every token are the plain step's, bit for bit.
#[test]
fn neuromod_with_letters_and_relay_in_the_fused_talking_step_bit_for_bit() {
    letters();
    type Sig = (Vec<[f32; 2]>, Option<[f32; 3]>);
    let run_nm = |m: &Model, toks: &[u32], sig: bool, fused: &dyn Fn(usize) -> bool| -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Sig>, u64) {
        let mut s = m.stream().unwrap();
        let ran = super::super::RAN.with(|r| r.get());
        let (mut logits, mut sigs) = (Vec::new(), Vec::new());
        for (i, &t) in toks.iter().enumerate() {
            m.set_fused(fused(i));
            m.set_part_signals(sig);
            logits.push(bits(&m.step(&mut s, t).unwrap()));
            let p = m.part_signals();
            sigs.push((p.th, p.nm));
        }
        m.set_fused(false);
        m.set_part_signals(false);
        let st = super::super::state_tensors(Model::stream_state(&s).unwrap());
        (logits, st.0.iter().map(bits).collect(), sigs, super::super::RAN.with(|r| r.get()) - ran)
    };
    for (k, extra) in ["--thalamus --neuromod 3", "--letters --thalamus --neuromod 3"].iter().enumerate() {
        let (_, m) = build_awake(cfg(extra), 90 + k as u64);
        m.set_int8(true).unwrap();
        assert_eq!(m.fused_blocker(), None, "{extra}: the fused talking step takes it");
        let toks = tokens(1, 300, V, 140 + k as u64);
        for sig in [false, true] {
            let (wl, ws, wsig, wn) = run_nm(&m, &toks, sig, &|_| false);
            assert_eq!(wn, 0, "{extra}: the plain run ran plain");
            let (gl, gs, gsig, gn) = run_nm(&m, &toks, sig, &|_| true);
            assert_eq!(gn, toks.len() as u64, "{extra}: every step ran fused");
            assert!(wl == gl, "{extra} (signals {sig}): logits differ");
            assert!(ws == gs, "{extra} (signals {sig}): the State differs");
            if sig {
                let bad = wsig.iter().zip(&gsig).filter(|(a, b)| a != b).count();
                assert_eq!(bad, 0, "{extra}: the part signals differ after {bad} of {} tokens", toks.len());
                assert!(gsig.iter().all(|(th, nm)| th.len() == 3 && nm.is_some()), "{extra}: gates of every block and the midbrain after every token");
                let nm_last = gsig.last().unwrap().1.unwrap();
                assert!(nm_last.iter().all(|v| v.is_finite() && *v != 1.0), "{extra}: the midbrain's signals moved off 1: {nm_last:?}");
            }
            let (tl, ts, tsig, tn) = run_nm(&m, &toks, sig, &|i| i % 3 != 1);
            assert_eq!(tn, 200, "{extra}: taking turns");
            assert!(wl == tl && ws == ts && wsig == tsig, "{extra} (signals {sig}): taking turns differs");
        }
        m.set_int8(false).unwrap();
    }
}
