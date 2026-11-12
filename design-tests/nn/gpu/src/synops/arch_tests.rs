// TESTING / DESIGN ONLY — v49 tests of forward_trace with block options and of synops' block-option rules, CPU only.

use super::*;
use crate::arch::Arch;
use crate::model::{Config, Kind, LayerTrace, Model, Trace};
use crate::state::tests::{build_awake, tokens};
use candle_core::{Device, Tensor};

const CPU: Device = Device::Cpu;
/// Ava 0.4's block options (runs/v42/wave3/s5-v42/arch.txt).
const AVA04: &str = "--head-size 64 --timescales fixed --gates channel --value-residual --kwta 1/4 --mlp-act relu2 --mem-momentum 0.5 \
                     --resid-fix --residual dense --lif-init logspaced --lif-scale --spikes int2 --alif --homeostasis 0.1";

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap().into_iter().map(f32::to_bits).collect()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// Ava 0.4's options with the glance, the magnitude code and the engram (block 1), d 64 (one memory head of 64).
fn ava04_cfg() -> Config {
    Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 64, 2, 64)
        .with_glance()
        .unwrap()
        .with_mag()
        .unwrap()
        .with_engram(1)
        .unwrap()
        .with_arch(Arch::from_flags(AVA04).unwrap())
        .unwrap()
}

fn ids(m: &Model, b: usize, t: usize, seed: u64) -> Tensor {
    Tensor::from_vec(tokens(b, t, m.cfg.vocab, seed), (b, t), &CPU).unwrap()
}

/// forward_trace runs forward's own code path, so its logits are forward's bit for bit; the trace holds what the
/// blocks computed (int2 spikes, the kWTA winners and the relu² units, the per-channel gates, the engram's weights,
/// the magnitude features), and the spikes and units agree with the I3 recorder (an independent tap of forward_arch).
#[test]
fn ava04_trace_logits_are_forwards_bit_for_bit() {
    let (vm, m) = build_awake(ava04_cfg(), 49);
    m.set_track_activity(false);
    // lower thresholds (homeostasis' per-neuron theta) so that the ±2 level fires too
    for (name, var) in vm.data().lock().unwrap().iter() {
        if name.ends_with(".mem.theta") {
            var.set(&var.as_tensor().affine(0.0, 0.2).unwrap()).unwrap();
        }
    }
    let (b, t) = (2, 64);
    let x = ids(&m, b, t, 7);
    let want = m.forward(&x).unwrap();
    let (got, tr) = m.forward_trace(&x).unwrap();
    assert_eq!(bits(&want), bits(&got), "forward_trace logits differ from forward");
    assert!(!m.tracing(), "the sink is closed after the trace");
    assert_eq!(bits(&m.forward(&x).unwrap()), bits(&want), "forward after a trace is unchanged");
    let (n, d, hid, k) = (b * t, m.cfg.d, m.cfg.hidden(), m.cfg.kwta_k());
    assert_eq!((tr.dims, tr.layers.len(), k), ((b, t), 2, hid / 4));
    let mag = host(tr.mag.as_ref().expect("the magnitude code is on"));
    assert_eq!(mag.len(), n * crate::mag::MAG_F);
    assert!(mag.iter().any(|&v| v != 0.0) && mag.iter().any(|&v| v == 0.0), "number tokens only");
    let (mut sizes, mut fired) = ([0usize; 5], 0usize);
    for (i, l) in tr.layers.iter().enumerate() {
        assert_eq!(l.spikes.dims(), &[n, d]);
        for v in host(&l.spikes) {
            assert!(v == v.round() && v.abs() <= 2.0, "int2 spike {v}");
            sizes[(v + 2.0) as usize] += 1;
        }
        let (w, a) = (host(&l.winners), host(&l.hidden));
        assert_eq!((w.len(), a.len()), (n * hid, n * hid));
        for r in 0..n {
            assert_eq!(w[r * hid..(r + 1) * hid].iter().filter(|&&v| v == 1.0).count(), k, "block {i} row {r}: k winners");
        }
        assert!(a.iter().zip(&w).all(|(&a, &w)| a >= 0.0 && (a == 0.0 || w == 1.0)), "units fire among the winners only");
        fired += a.iter().filter(|&&v| v != 0.0).count();
        // channel gates: one write strength and keep fraction per channel
        let (hw, hk) = (host(l.hebb_write.as_ref().unwrap()), host(l.hebb_keep.as_ref().unwrap()));
        assert_eq!((hw.len(), hk.len()), (n * d, n * d));
        assert!(hw.iter().all(|&v| v > 0.0 && v < 2.0) && hk.iter().all(|&v| v > 0.0 && v < 1.0));
        assert_eq!(l.hebb_read.as_ref().unwrap().dims(), &[n, d]);
        assert_eq!(l.engram_read.is_some(), i == 1);
        if let Some(at) = &l.engram_attn {
            assert_eq!((i, at.dims()), (1, &[b, d / crate::model::HEAD_SIZE, t, t][..]));
            let s = host(&at.sum(3).unwrap());
            assert!(s.iter().all(|&v| (v - 1.0).abs() < 1e-4));
        } else {
            assert_ne!(i, 1);
        }
    }
    assert!(sizes[1] + sizes[3] > 0 && sizes[0] + sizes[4] > 0, "both signs fire, ±2s among them: {sizes:?}");
    assert!(fired > 0);
    // the I3 recorder (forward_arch's ReDo tap): |spikes| per channel and |units| per unit, means over the rows
    let ua = crate::upkeep::unit_activity(&m, &[x.clone()]).unwrap();
    for (i, l) in tr.layers.iter().enumerate() {
        let s = host(&l.spikes.abs().unwrap().sum(0).unwrap());
        let a = host(&l.hidden.abs().unwrap().sum(0).unwrap());
        assert!(s.iter().zip(&ua.lif[i]).all(|(x, y)| (x / n as f32 - y).abs() < 1e-6), "block {i}: spikes vs I3");
        assert!(a.iter().zip(&ua.mlp[i]).all(|(x, y)| (x / n as f32 - y).abs() < 1e-5), "block {i}: units vs I3");
    }
}

/// O1 (--feedback) runs a window in chunks of 64: the trace joins each block's chunks back into batch-major rows, and
/// the first chunk's rows are what the first 64 tokens alone give.
#[test]
fn chunked_window_trace_joins_in_row_order() {
    let ar = Arch { feedback: true, spikes: crate::arch::Spikes::Int2, ..Arch::default() };
    let cfg = Config::with_kind(Kind::D, 258, 32, 2, 128).with_arch(ar).unwrap();
    let (_vm, m) = build_awake(cfg, 5);
    let (b, t) = (2, 96);
    let toks = tokens(b, t, 258, 11);
    let x = Tensor::from_vec(toks.clone(), (b, t), &CPU).unwrap();
    let (got, tr) = m.forward_trace(&x).unwrap();
    assert_eq!(bits(&m.forward(&x).unwrap()), bits(&got));
    let first: Vec<u32> = (0..b).flat_map(|r| toks[r * t..r * t + 64].to_vec()).collect();
    let (_, tr64) = m.forward_trace(&Tensor::from_vec(first, (b, 64), &CPU).unwrap()).unwrap();
    for (l, l64) in tr.layers.iter().zip(&tr64.layers) {
        assert_eq!(l.spikes.dims(), &[b * t, 32]);
        for r in 0..b {
            for (a, c) in [(&l.spikes, &l64.spikes), (&l.hidden, &l64.hidden), (l.hebb_write.as_ref().unwrap(), l64.hebb_write.as_ref().unwrap())] {
                assert_eq!(bits(&a.narrow(0, r * t, 64).unwrap()), bits(&c.narrow(0, r * 64, 64).unwrap()), "row {r}");
            }
        }
    }
}

fn layer(spikes: Vec<f32>, hidden: Vec<f32>, n: usize, d: usize, hid: usize) -> LayerTrace {
    LayerTrace {
        spikes: Tensor::from_vec(spikes, (n, d), &CPU).unwrap(),
        winners: Tensor::ones((n, hid), candle_core::DType::F32, &CPU).unwrap(),
        hidden: Tensor::from_vec(hidden, (n, hid), &CPU).unwrap(),
        hebb_read: None,
        hebb_write: None,
        hebb_keep: None,
        engram_attn: None,
        engram_read: None,
    }
}

/// A hand-checked count: one block of Ava 0.4 at d 64 with the engram, one row of 3 positions (the last one PAD).
#[test]
fn hand_count_of_a_tiny_ava04_block() {
    let cfg = Config::with_kind(Kind::D, 258, 64, 1, 64).with_engram(0).unwrap().with_arch(Arch::from_flags(AVA04).unwrap()).unwrap();
    assert!(check(&cfg).is_ok());
    let (d, hid) = (64, 256);
    let mut s = vec![0f32; 3 * d];
    s[0] = 2.0; // row 0: a 2 and a -1
    s[1] = -1.0;
    s[d] = -2.0; // row 1: a -2 and two 1s
    s[d + 2] = 1.0;
    s[d + 3] = 1.0;
    s[2 * d..].iter_mut().for_each(|v| *v = 1.0); // row 2 (PAD): not counted
    let mut h = vec![0f32; 3 * hid];
    h[..5].iter_mut().for_each(|v| *v = 0.25); // row 0: 5 units, row 1: none
    h[2 * hid..].iter_mut().for_each(|v| *v = 1.0);
    let tr = Trace { layers: vec![layer(s, h, 3, d, hid)], dims: (1, 3), mag: None };
    let mut c = SynCount::new(&cfg);
    c.add(&tr, &Tensor::from_vec(vec![1f32, 1.0, 0.0], 3, &CPU).unwrap()).unwrap();
    assert_eq!((c.tokens, c.spikes[0], c.spikes2[0], c.active[0], c.keys), (2, 5, 2, 5, 3));
    assert_eq!(c.synops_per_token(), (5.0 + 5.0) * 64.0 / 2.0);
    // dense multiply-adds per token, part by part (d 64, one head of 64, hidden 256, gate rank 32, vocab 258):
    // Win 4096, Wout 4096, Wq Wk Wv Wo 16384, Wg 4096, head gates 2·64·1 = 128, channel gates 4·64·32 + We 4096 = 12288,
    // core 6·64² = 24576, value residual 0 (one block), dense mix 2·64 = 128, W1 16384, W2 16384, engram 4·4096 = 16384,
    // look-back 2·64·(1 + 2)/2 = 192, unembedding 64·258 = 16512
    let dense: f64 = arch_parts(&cfg, crate::flops::DEFAULT_MEM, c.keys_per_token()).iter().map(|p| p.1).sum();
    assert_eq!(dense, 131_648.0);
    let r = c.report(&cfg, crate::flops::DEFAULT_MEM);
    let val = |key: &str| r.lines().find(|l| l.split('\t').next() == Some(key)).unwrap().split('\t').nth(1).unwrap().to_string();
    assert_eq!(val("synops_per_token"), "320");
    assert_eq!(val("those_two_dense_macs"), "20480"); // 64² + 256·64
    assert_eq!(val("forward_macs_dense"), "131648");
    assert_eq!(val("forward_macs_event"), "111488"); // 131648 − 20480 + 320
    assert_eq!(val("spikes_of_size_2_per_token"), "1.00");
    assert_eq!(val("engram_keys_per_token"), "1.5");
    // the first lines keep today's report's keys and order (scripts read them)
    let keys: Vec<&str> = r.lines().skip(2).take(5).map(|l| l.split('\t').next().unwrap()).collect();
    assert_eq!(keys, ["tokens", "synops_per_token", "those_two_dense_macs", "forward_macs_dense", "forward_macs_event"]);
}

/// With today's memory rule (per-head gates, no momentum, head size 32) and no engram, the block-option count is
/// flops.rs's in both memory forms; parts it cannot count are refused.
#[test]
fn today_rule_parts_match_flops_and_unknown_parts_are_refused() {
    let cfg = Config::with_kind(Kind::D, 258, 64, 3, 256).with_arch(Arch { kwta_div: 4, ..Arch::default() }).unwrap();
    for mem in [crate::flops::MemForm::PerToken, crate::flops::MemForm::Chunk(16)] {
        let ours: f64 = arch_parts(&cfg, mem, 0.0).iter().map(|p| p.1).sum();
        assert!((ours - per_token(&cfg, mem).forward() / 2.0).abs() < 1e-6, "{mem:?}");
    }
    let mut ar = Arch::from_flags(AVA04).unwrap();
    ar.mem.dendrites = 4;
    let bad = Config::with_kind(Kind::D, 258, 64, 2, 64).with_arch(ar);
    if let Ok(bad) = bad {
        assert!(check(&bad).is_err());
    }
    assert!(!SynCount::new(&crate::testkit::tiny(Kind::D, 1).0).arch, "today's block keeps today's count");
}
