// TESTING / DESIGN ONLY — tests of `Model::step_trace` on a State-backed stream (block options; the chat app's glow).
//
// Pre-set bar (written before the first run): tracing changes nothing. Over 300 tokens with the State carried, every
// logit and the final State of `step_trace` equal `step`'s bit for bit, on the fused step (int8, `set_fused`) and on
// the plain State step (fused off), for Ava 0.4's recipe (glance, magnitude code, engram) and a binary-spike variant.
// The trace itself: one layer per block, spikes [1, d], the k winners of each MLP, units only where the winners are,
// per-head memory sizes that are finite and >= 0, an engram attention that sums to 1 over every token so far, and the
// two paths' copies agreeing (spikes, winners, units bit for bit; the memory's write and gates within 1e-4 relative).

use crate::arch::Arch;
use crate::model::{Config, Kind, Model, StepTrace};
use crate::state::tests::{build_awake, tokens};
use candle_core::{DType, Tensor};

const AVA04: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
                     --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";

fn cfg(flags: &str) -> Config {
    let g = ava_data::numbers::VOCAB_GLANCE;
    Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap().with_arch(Arch::from_flags(flags).unwrap()).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// Logits of every step and the final State: `step` (traced = false) or `step_trace` (with its traces).
fn run(m: &Model, toks: &[u32], fused: bool, traced: bool) -> (Vec<Tensor>, Vec<Tensor>, Vec<StepTrace>) {
    m.set_fused(fused);
    let mut s = m.stream().unwrap();
    let (mut logits, mut trs) = (Vec::new(), Vec::new());
    for &t in toks {
        if traced {
            let (l, tr) = m.step_trace(&mut s, t).unwrap();
            logits.push(l);
            trs.push(tr);
        } else {
            logits.push(m.step(&mut s, t).unwrap());
        }
    }
    m.set_fused(false);
    assert_eq!(s.seen, toks.len());
    (logits, super::state_tensors(Model::stream_state(&s).unwrap()).0, trs)
}

fn same(name: &str, a: &(Vec<Tensor>, Vec<Tensor>, Vec<StepTrace>), b: &(Vec<Tensor>, Vec<Tensor>, Vec<StepTrace>)) {
    assert_eq!(a.0.len(), b.0.len());
    for (i, (x, y)) in a.0.iter().zip(&b.0).enumerate() {
        assert!(bits(x) == bits(y), "{name}: logits differ at step {i}");
    }
    assert_eq!(a.1.len(), b.1.len());
    for (j, (x, y)) in a.1.iter().zip(&b.1).enumerate() {
        assert!(bits(x) == bits(y), "{name}: state tensor {j} differs");
    }
}

fn check_trace(name: &str, m: &Model, trs: &[StepTrace], fused: bool) {
    let (d, k, nb) = (m.cfg.d, m.cfg.kwta_k(), m.blocks.len());
    for (i, tr) in trs.iter().enumerate() {
        assert_eq!(tr.layers.len(), nb, "{name}: layers at {i}");
        assert!(tr.input_rms.is_finite() && tr.input_rms > 0.0, "{name}: input size at {i}");
        for (li, l) in tr.layers.iter().enumerate() {
            assert_eq!(l.spikes.dims(), &[1, d], "{name}: spikes");
            let (w, h) = (v(&l.winners), v(&l.hidden));
            assert_eq!(w.len(), h.len());
            let nw = w.iter().filter(|&&x| x > 0.5).count();
            assert!(nw >= k, "{name}: {nw} winners < k {k} at {i}/{li}");
            assert!(h.iter().zip(&w).all(|(&a, &b)| b > 0.5 || a == 0.0), "{name}: units outside the winners");
            for t in [&l.mem_write, &l.mem_keep, &l.mem_gate].into_iter().flatten() {
                assert!(v(t).iter().all(|x| x.is_finite() && *x >= 0.0), "{name}: memory sizes at {i}/{li}");
            }
            assert!(l.mem_write.is_some(), "{name}: the write size");
            assert_eq!(l.mem_read.is_some(), fused, "{name}: the read size (fused step only)");
            if li == 1 && fused {
                let at = l.engram_attn.as_ref().expect("the engram's attention");
                let (nh, seen) = at.dims2().unwrap();
                assert_eq!(seen, i + 1, "{name}: attention over every token so far");
                let a = v(at);
                for h in 0..nh {
                    let s: f32 = a[h * seen..(h + 1) * seen].iter().sum();
                    assert!((s - 1.0).abs() < 1e-4, "{name}: attention sums to {s}");
                }
            }
            if li == 1 {
                assert!(l.engram_read.is_some(), "{name}: the engram's read");
            }
        }
    }
}

#[test]
fn step_trace_on_block_options_changes_nothing() {
    for (k, flags) in [AVA04, "--head-size 64 --gates channel --spikes binary --mem-momentum 0.3 --kwta 0.25"].iter().enumerate() {
        let (_, m) = build_awake(cfg(flags), 80 + k as u64);
        m.set_int8(true).unwrap();
        assert_eq!(m.fused_blocker(), None, "{flags}");
        // numbers and words, so the glance and the magnitude code add something now and then
        let toks = tokens(1, 300, ava_data::numbers::VOCAB_GLANCE, 120 + k as u64);
        let ran = super::RAN.with(|r| r.get());
        let plain_step = run(&m, &toks, true, false);
        let fused_trace = run(&m, &toks, true, true);
        assert_eq!(super::RAN.with(|r| r.get()) - ran, 600, "{flags}: every step ran fused, traced or not");
        same(&format!("{flags}: fused, traced vs not"), &plain_step, &fused_trace);
        check_trace(flags, &m, &fused_trace.2, true);
        let ran = super::RAN.with(|r| r.get());
        let slow_step = run(&m, &toks, false, false);
        let slow_trace = run(&m, &toks, false, true);
        assert_eq!(super::RAN.with(|r| r.get()) - ran, 0, "{flags}: fused off");
        same(&format!("{flags}: plain State step, traced vs not"), &slow_step, &slow_trace);
        same(&format!("{flags}: fused traced vs plain"), &slow_step, &fused_trace);
        check_trace(flags, &m, &slow_trace.2, false);
        // the two paths hand over the same activity
        for (i, (a, b)) in fused_trace.2.iter().zip(&slow_trace.2).enumerate() {
            assert_eq!(a.place, b.place, "place at {i}");
            assert_eq!(a.mag, b.mag, "mag at {i}");
            assert!((a.input_rms - b.input_rms).abs() <= 1e-5 * a.input_rms.max(1.0), "input size at {i}");
            for (la, lb) in a.layers.iter().zip(&b.layers) {
                assert!(bits(&la.spikes) == bits(&lb.spikes), "spikes at {i}");
                assert!(bits(&la.winners) == bits(&lb.winners), "winners at {i}");
                assert!(bits(&la.hidden) == bits(&lb.hidden), "units at {i}");
                for (x, y) in [(&la.mem_write, &lb.mem_write), (&la.mem_keep, &lb.mem_keep), (&la.mem_gate, &lb.mem_gate)] {
                    let (x, y) = (v(x.as_ref().unwrap()), v(y.as_ref().unwrap()));
                    for (p, q) in x.iter().zip(&y) {
                        assert!((p - q).abs() <= 1e-4 * p.abs().max(q.abs()).max(1e-3), "memory size {p} vs {q} at {i}");
                    }
                }
                if let (Some(x), Some(y)) = (&la.engram_read, &lb.engram_read) {
                    assert!(bits(x) == bits(y), "engram read at {i}");
                }
            }
        }
    }
}
