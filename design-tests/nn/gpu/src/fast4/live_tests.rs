// TESTING / DESIGN ONLY — v52: the chat app's live session step (`Model::fast4_live_step`, src/fast4/live.rs) against
// the State path's session, CPU only.
//
// Pre-set bar (written before the first run): bit for bit. One row reads 80 tokens of a text (each learning the next
// word) and then speaks 40 (each learning the word drawn from its own logits, greedily), and a last token learns
// nothing (no target): every token's logits, e-prop's rms and the cerebellum's nats, the chat app's cerebellum signals,
// then the final State, every plastic weight's changes and traces and the online table (stored values, scale, update
// count) are the State path's (`forward_with_state`, `cross_entropy(..).affine(1)` + `backward` +
// `plastic_learn_rows(.., EProp)` + `detach_all`, then `cb_learn`, on the same tokens), for E2 alone, CB2 alone and
// both together, with the viewer's record and the part signals on; CB2 alone also with int8 on (the frozen weights
// int8, the table f32). Lean models with a cerebellum head: d 128 (3 blocks, blocks 1-2 plastic) and d 192 (2 blocks,
// both plastic: block 0's values on the tape). And the viewer's record of the plastic blocks (from the tape) is the
// fused step's own record of the same blocks when nothing is learned (rate 0): spikes, winners and units bit for bit,
// the memory's sizes to 1e-5.

use crate::arch::Arch;
use crate::eprop::{Rule, Site};
use crate::model::fstep::arch::state_tensors;
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, tokens};
use candle_core::{DType, Device, Tensor};

const AVA04: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
                     --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";
const V: usize = 293;
const CPU: Device = Device::Cpu;
const READ: usize = 80;
const SAY: usize = 40;

fn vals(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    vals(t).iter().map(|x| x.to_bits()).collect()
}

fn argmax(v: &[f32]) -> u32 {
    let mut b = 0;
    for (j, &x) in v.iter().enumerate() {
        if x > v[b] {
            b = j;
        }
    }
    b as u32
}

#[derive(PartialEq, Debug)]
struct Out {
    /// the tokens read and the targets learned (None: none)
    steps: Vec<(u32, Option<u32>)>,
    logits: Vec<Vec<u32>>,
    eprop: Vec<u64>,
    cb: Vec<u64>,
    sig: Vec<(Option<u32>, Option<u32>)>,
    state: Vec<Vec<u32>>,
    counts: (Vec<usize>, Vec<Vec<u32>>),
    deltas: Vec<((usize, Site), Vec<u32>)>,
    traces: Vec<(usize, Vec<u32>)>,
    table: Option<(Vec<u32>, u64, usize)>,
}

#[derive(Clone, Copy)]
struct Learn {
    eprop: bool,
    cb: bool,
    blocks: &'static [usize],
}

fn begin(m: &Model, l: Learn) {
    if l.eprop {
        m.plastic_begin(1, l.blocks, &Site::ALL).unwrap();
        m.plastic_set_half_lives(&[128.0]).unwrap();
        m.plastic_new_documents().unwrap();
        m.plastic_probes(true);
    }
    if l.cb {
        m.cb_online_begin(1, 0.05).unwrap();
        m.cb_online_set_lr(0, 0.05);
        m.cb_online_set_half_life(0, 256.0).unwrap();
    }
}

fn end(m: &Model, st: &crate::state::State, mut o: Out) -> Out {
    let (ts, seen, hist) = state_tensors(st);
    o.state = ts.iter().map(bits).collect();
    o.counts = (seen, hist);
    o.deltas = m.plastic_deltas().unwrap().into_iter().map(|(k, t)| (k, bits(&t))).collect();
    o.traces = match m.v51b().unwrap().plastic.borrow().as_ref() {
        Some(p) => {
            let mut ks: Vec<usize> = p.trace.keys().copied().collect();
            ks.sort();
            ks.into_iter().map(|i| (i, bits(&p.trace[&i]))).collect()
        }
        None => Vec::new(),
    };
    o.table = m.cb_online_with(|t| (t.raw(0).0.iter().map(|x| x.to_bits()).collect(), t.raw(0).1.to_bits(), t.updates[0]));
    m.plastic_end();
    m.cb_online_end();
    o
}

fn empty() -> Out {
    Out { steps: Vec::new(), logits: Vec::new(), eprop: Vec::new(), cb: Vec::new(), sig: Vec::new(), state: Vec::new(), counts: (Vec::new(), Vec::new()), deltas: Vec::new(), traces: Vec::new(), table: None }
}

fn sig(m: &Model) -> (Option<u32>, Option<u32>) {
    let s = m.part_signals();
    (s.cb.map(f32::to_bits), s.cb_online.map(f32::to_bits))
}

/// The live session: READ tokens of `text` (each learning the next), SAY tokens drawn greedily from each token's own
/// logits (each learning the word drawn), then one token with no target.
fn live(m: &Model, l: Learn, text: &[u32], lr: f64) -> Out {
    begin(m, l);
    let mut st = m.state(1).unwrap();
    let mut o = empty();
    let mut tok = text[0];
    for p in 0..READ + SAY + 1 {
        let mut tgt = |lg: &[f32]| -> candle_core::Result<Option<u32>> {
            Ok(if p + 1 < READ {
                Some(text[p + 1])
            } else if p < READ + SAY {
                Some(argmax(lg))
            } else {
                None
            })
        };
        let s = m.fast4_live_step(&mut st, tok, lr, true, &mut tgt).unwrap().unwrap_or_else(|| panic!("the live step runs: {:?}", m.fast4_live_blocker()));
        assert!(s.trace.as_ref().is_some_and(|t| t.layers.len() == m.blocks.len()), "a record of every block");
        o.steps.push((tok, s.next));
        o.logits.push(bits(&s.logits));
        o.eprop.extend(s.eprop_rms.map(f64::to_bits));
        o.cb.extend(s.cb.map(|c| c.0.to_bits()));
        o.sig.push(sig(m));
        assert_eq!(s.eprop_rms.is_some(), l.eprop && s.next.is_some());
        assert_eq!(s.cb.is_some(), l.cb && s.next.is_some());
        if let Some(n) = s.next {
            tok = n;
        }
    }
    end(m, &st, o)
}

/// The State path on the live session's own tokens and targets.
fn reference(m: &Model, l: Learn, steps: &[(u32, Option<u32>)], lr: f64) -> Out {
    begin(m, l);
    let mut st = m.state(1).unwrap();
    let mut o = empty();
    for &(t, y) in steps {
        let lg = m.forward_with_state(&Tensor::new(&[[t]], &CPU).unwrap(), &mut st).unwrap();
        o.steps.push((t, y));
        o.logits.push(bits(&lg));
        o.sig.push(sig(m));
        if let Some(y) = y {
            if l.eprop {
                let loss = candle_nn::loss::cross_entropy(&lg, &Tensor::new(&[y], &CPU).unwrap()).unwrap().affine(1.0, 0.0).unwrap();
                o.eprop.push(m.plastic_learn_rows(&loss.backward().unwrap(), &[lr], Rule::EProp).unwrap().to_bits());
            }
            if l.cb {
                o.cb.push(m.cb_learn(&lg, &[y]).unwrap().to_bits());
            }
        }
        st.detach_all();
    }
    end(m, &st, o)
}

fn lean(cfg: Config, seed: u64) -> Model {
    let (vm, _) = build_awake(cfg, seed);
    let m = crate::ckpt::lean(cfg, &vm, &CPU).unwrap();
    m.set_track_activity(false);
    m.set_part_signals(true);
    m
}

#[test]
fn fast4_the_live_session_step_is_the_state_paths_bit_for_bit() {
    let _steady = crate::thread_env::steady();
    let cfg = |d: usize, layers: usize| Config::with_kind(Kind::D, V, d, layers, 64).with_arch(Arch::from_flags(&format!("{AVA04} --cerebellum-head 512")).unwrap()).unwrap();
    let text = tokens(1, READ + 1, V, 521);
    let mut n = 0;
    for (name, c, blocks) in [("d128", cfg(128, 3), &[1usize, 2][..]), ("d192 (k 0)", cfg(192, 2), &[0, 1][..])] {
        let m = lean(c, 523);
        m.set_fast4(true);
        let modes = [("E2", Learn { eprop: true, cb: false, blocks }, false), ("CB2", Learn { eprop: false, cb: true, blocks }, false), ("E2+CB2", Learn { eprop: true, cb: true, blocks }, false), ("CB2, int8", Learn { eprop: false, cb: true, blocks }, true)];
        for (mode, l, int8) in modes {
            let name = format!("{name}, {mode}");
            m.set_int8(int8).unwrap();
            let got = live(&m, l, &text, 0.001);
            let want = reference(&m, l, &got.steps, 0.001);
            m.set_int8(false).unwrap();
            assert_eq!(got.steps.len(), READ + SAY + 1);
            for (i, (a, b)) in want.logits.iter().zip(&got.logits).enumerate() {
                assert!(a == b, "{name}: logits differ at step {i}");
            }
            assert_eq!(want.sig, got.sig, "{name}: the cerebellum's signals");
            assert_eq!(want.cb, got.cb, "{name}: the cerebellum's nats");
            assert_eq!(want.state, got.state, "{name}: the State");
            assert_eq!(want.counts, got.counts, "{name}: counts, history");
            assert_eq!(want.deltas.len(), got.deltas.len());
            for ((ka, a), (kb, b)) in want.deltas.iter().zip(&got.deltas) {
                assert_eq!(ka, kb);
                assert!(a == b, "{name}: the changes of {ka:?}");
                assert!(a.iter().any(|&x| x != 0), "{name}: {ka:?} learned");
            }
            assert!(want.traces == got.traces, "{name}: the traces");
            assert_eq!(want.table, got.table, "{name}: the online table");
            if l.cb {
                let t = got.table.as_ref().expect("a table");
                assert_eq!(t.2, READ + SAY, "{name}: one update per target");
                assert!(t.0.iter().any(|&x| x != 0), "{name}: the table learned");
            }
            assert_eq!(want.eprop, got.eprop, "{name}: e-prop's rms");
            assert_eq!(got.eprop.len(), if l.eprop { READ + SAY } else { 0 });
            n += 1;
        }
    }
    println!("fast4 live: {n} sessions of {} tokens bit for bit", READ + SAY + 1);
}

#[test]
fn fast4_the_live_record_of_the_plastic_blocks_is_the_fused_steps() {
    let _steady = crate::thread_env::steady();
    let c = Config::with_kind(Kind::D, V, 128, 3, 64).with_arch(Arch::from_flags(AVA04).unwrap()).unwrap();
    let m = lean(c, 527);
    m.set_fast4(true);
    let text = tokens(1, 61, V, 529);
    // the fused step's own record, nothing learning
    let mut st = m.state(1).unwrap();
    let mut want = Vec::new();
    for &t in &text[..60] {
        let mut rec = crate::model::fstep::arch::TokTrace::default();
        let lg = crate::model::fstep::arch::step_traced(&m, &mut st, t, &mut rec).unwrap().expect("fused");
        want.push((bits(&lg), rec));
    }
    // a live e-prop session at rate 0 on blocks 1-2: its changes stay 0, so it computes the same token
    m.plastic_begin(1, &[1, 2], &Site::ALL).unwrap();
    m.plastic_probes(true);
    let mut st = m.state(1).unwrap();
    for (p, &t) in text[..60].iter().enumerate() {
        let s = m.fast4_live_step(&mut st, t, 0.0, true, &mut |_| Ok(Some(text[p + 1]))).unwrap().expect("live");
        let (wl, wr) = &want[p];
        assert!(&bits(&s.logits) == wl, "logits at {p}");
        let tr = s.trace.unwrap();
        assert_eq!((tr.place, tr.mag, tr.input_rms.to_bits()), (wr.place, wr.mag, wr.input_rms.to_bits()), "input at {p}");
        for (i, (a, b)) in tr.layers.iter().zip(&wr.layers).enumerate() {
            assert!(bits(&a.spikes) == b.spikes.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "spikes, block {i}, token {p}");
            assert!(bits(&a.winners) == b.winners.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "winners, block {i}, token {p}");
            assert!(bits(&a.hidden) == b.hidden.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "units, block {i}, token {p}");
            for (what, x, y) in [("write", &a.mem_write, &b.mem_write), ("read", &a.mem_read, &b.mem_read), ("keep", &a.mem_keep, &b.mem_keep), ("gate", &a.mem_gate, &b.mem_gate)] {
                let x = vals(x.as_ref().expect("a memory"));
                assert_eq!(x.len(), y.len());
                for (u, v) in x.iter().zip(y) {
                    assert!((u - v).abs() <= 1e-5 * (1.0 + v.abs()), "memory {what}, block {i}, token {p}: {u} vs {v}");
                }
            }
        }
    }
    m.plastic_end();
}
