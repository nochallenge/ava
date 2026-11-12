// TESTING / DESIGN ONLY — v51 fast4: the e-prop session step (`Model::fast4_eprop_step`) against the State path's token
// (`forward_with_state` + `cross_entropy(..).affine(1)` + `loss.backward()` + `plastic_learn_rows(.., EProp)` +
// `detach_all`, as v51_online's stream runs it), and fast4's tape products against candle's `matmul`. CPU only.
//
// Pre-set bar (written before the first run): bit for bit. Over 300 tokens of one row: every token's logits and the
// learning step's rms, then the final State, every plastic weight's changes and every block's trace; for half-lives
// 128 (E2's), inf and 2 tokens; on lean models (weights not variables, as the stream loads them) at d 128 (3 blocks,
// blocks 1-2 plastic), d 192 (2 blocks, both plastic: block 0's values on the tape) and d 384 (2 blocks, block 1).
// The tape's products: the same bits as candle's matmul for every shape and layout the step multiplies, on 1 and 8
// threads.

use super::tape::Tape;
use crate::arch::Arch;
use crate::eprop::{Rule, Site};
use crate::model::fstep::arch::state_tensors;
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, tokens};
use crate::testkit::unit_noise;
use candle_core::{DType, Device, Tensor};

const AVA04: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
                     --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";
const V: usize = 293;
const CPU: Device = Device::Cpu;

fn vals(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    vals(t).iter().map(|x| x.to_bits()).collect()
}

/// (lhs base dims, lhs transposed (last two), rhs base dims, rhs transposed)
type Case = (Vec<usize>, bool, Vec<usize>, bool);

#[test]
fn fast4_tape_products_are_candles() {
    let mut cases: Vec<Case> = Vec::new();
    for (d, nh) in [(128usize, 2usize), (192, 3), (384, 6)] {
        let hs = 64;
        // x · W and its backward g · W^T, the head x · E^T and its backward g · E
        for n in [nh, 16, 64, d, 4 * d, V] {
            cases.push((vec![1, d], false, vec![d, n], false));
            cases.push((vec![1, n], false, vec![d, n], true));
        }
        cases.push((vec![1, d], false, vec![V, d], true));
        cases.push((vec![1, V], false, vec![V, d], false));
        // the memory's batched products at one token and their backwards
        cases.push((vec![nh, 1, 1, 1], false, vec![nh, 1, 1, hs], false));
        cases.push((vec![nh, 1, 1, hs], false, vec![nh, 1, 1, hs], true));
        cases.push((vec![nh, 1, 1, 1], true, vec![nh, 1, 1, hs], false));
        cases.push((vec![nh, 1, 1, hs], true, vec![nh, 1, 1, hs], false));
        cases.push((vec![nh, 1, hs, hs], false, vec![nh, 1, 1, hs], true));
        cases.push((vec![nh, 1, 1, hs], false, vec![nh, 1, hs, hs], false));
        cases.push((vec![nh, 1, hs], false, vec![nh, hs, hs], false));
        cases.push((vec![nh, 1, hs], false, vec![nh, hs, hs], true));
        cases.push((vec![nh, 1, 1], false, vec![nh, 1, hs], false));
        cases.push((vec![nh, 1, hs], false, vec![nh, 1, hs], true));
        cases.push((vec![nh, 1, 1], true, vec![nh, 1, hs], false));
        cases.push((vec![nh, 1, hs], true, vec![nh, 1, hs], false));
        cases.push((vec![nh, hs, hs], false, vec![nh, 1, hs], true));
        cases.push((vec![nh, hs, 1], true, vec![nh, hs, hs], false));
        cases.push((vec![nh * hs, 1], false, vec![1, 1], false));
        cases.push((vec![nh * hs, 1], false, vec![1, 1], true));
    }
    let mut n = 0;
    for (j, (ab, at, bb, bt)) in cases.iter().enumerate() {
        let av = unit_noise(ab.iter().product(), 900 + j as u64);
        let bv = unit_noise(bb.iter().product(), 1900 + j as u64);
        let ta = Tensor::from_vec(av.clone(), ab.as_slice(), &CPU).unwrap();
        let tb = Tensor::from_vec(bv.clone(), bb.as_slice(), &CPU).unwrap();
        let ta = if *at { ta.t().unwrap() } else { ta };
        let tb = if *bt { tb.t().unwrap() } else { tb };
        let want = bits(&ta.matmul(&tb).unwrap());
        for th in [1, 8] {
            let mut tp = Tape::new(th);
            let a = tp.leaf(av.clone(), ab);
            let b = tp.leaf(bv.clone(), bb);
            let a = if *at { tp.t(a).unwrap() } else { a };
            let b = if *bt { tp.t(b).unwrap() } else { b };
            let c = tp.matmul(a, b).unwrap();
            let got: Vec<u32> = tp.vals(c).iter().map(|x| x.to_bits()).collect();
            assert!(want == got, "product {j}: {ab:?}{} x {bb:?}{} on {th} threads", if *at { "^T" } else { "" }, if *bt { "^T" } else { "" });
            n += 1;
        }
    }
    println!("fast4 tape: {n} products bit for bit");
}

struct Out {
    logits: Vec<Vec<u32>>,
    rms: Vec<u64>,
    state: (Vec<Tensor>, Vec<usize>, Vec<Vec<u32>>),
    deltas: Vec<((usize, Site), Vec<u32>)>,
    traces: Vec<(usize, Vec<u32>)>,
}

/// One row's e-prop session over `toks`: the State path (`fused` false) or the fused session step.
fn session(m: &Model, toks: &[u32], fused: bool, blocks: &[usize], half: f64, lr: f64) -> Out {
    m.plastic_begin(1, blocks, &Site::ALL).unwrap();
    m.plastic_set_half_lives(&[half]).unwrap();
    m.plastic_new_documents().unwrap();
    m.plastic_probes(true);
    let mut st = m.state(1).unwrap();
    let (mut logits, mut rms) = (Vec::new(), Vec::new());
    for w in toks.windows(2) {
        let (t, y) = (w[0], w[1]);
        if fused {
            let (lg, r) = m.fast4_eprop_step(&mut st, t, y, lr).unwrap().unwrap_or_else(|| panic!("the fused step runs: {:?}", m.fast4_eprop_blocker()));
            logits.push(bits(&lg));
            rms.push(r.to_bits());
        } else {
            let lg = m.forward_with_state(&Tensor::new(&[[t]], &CPU).unwrap(), &mut st).unwrap();
            let loss = candle_nn::loss::cross_entropy(&lg, &Tensor::new(&[y], &CPU).unwrap()).unwrap().affine(1.0, 0.0).unwrap();
            let r = m.plastic_learn_rows(&loss.backward().unwrap(), &[lr], Rule::EProp).unwrap();
            st.detach_all();
            logits.push(bits(&lg));
            rms.push(r.to_bits());
        }
    }
    let deltas = m.plastic_deltas().unwrap().into_iter().map(|(k, t)| (k, bits(&t))).collect();
    let traces = {
        let e = m.v51b().unwrap();
        let g = e.plastic.borrow();
        let p = g.as_ref().unwrap();
        let mut ks: Vec<usize> = p.trace.keys().copied().collect();
        ks.sort();
        ks.into_iter().map(|i| (i, bits(&p.trace[&i]))).collect()
    };
    m.plastic_end();
    Out { logits, rms, state: state_tensors(&st), deltas, traces }
}

/// The lean model (weights detached, as v51_online loads them) of `cfg` with build_awake's fixed weights.
fn lean(cfg: Config, seed: u64) -> Model {
    let (vm, _) = build_awake(cfg, seed);
    let m = crate::ckpt::lean(cfg, &vm, &CPU).unwrap();
    m.set_track_activity(false);
    m
}

#[test]
fn fast4_the_eprop_session_step_is_the_state_paths_bit_for_bit() {
    let _steady = crate::thread_env::steady(); // no other test changes the thread count while the sessions run
    let cfg = |d: usize, layers: usize| Config::with_kind(Kind::D, V, d, layers, 64).with_arch(Arch::from_flags(AVA04).unwrap()).unwrap();
    let models = [("d128", cfg(128, 3), vec![1usize, 2]), ("d192 (k 0)", cfg(192, 2), vec![0, 1]), ("d384", cfg(384, 2), vec![1])];
    let toks = tokens(1, 301, V, 303);
    let mut n = 0;
    for (name, c, blocks) in models.iter() {
        let m = lean(*c, 311);
        m.set_fast4(true);
        for (half, lr) in [(128.0, 0.001), (f64::INFINITY, 0.02), (2.0, 0.05)] {
            let name = format!("{name}, half-life {half}, lr {lr}");
            let th0 = candle_core::utils::get_num_threads();
            let want = session(&m, &toks, false, blocks, half, lr);
            let got = session(&m, &toks, true, blocks, half, lr);
            // both paths sum the rms's lines in as many chunks as RAYON_NUM_THREADS says at the time; thread_env::steady
            // (top of the test) keeps any other test of the suite from changing it meanwhile; the check below stays as a guard
            let th1 = candle_core::utils::get_num_threads();
            assert_eq!(want.logits.len(), 300);
            for (i, (a, b)) in want.logits.iter().zip(&got.logits).enumerate() {
                assert!(a == b, "{name}: logits differ at step {i}");
            }
            for (j, (a, b)) in want.state.0.iter().zip(&got.state.0).enumerate() {
                assert!(bits(a) == bits(b), "{name}: state tensor {j}");
            }
            assert_eq!((&want.state.1, &want.state.2), (&got.state.1, &got.state.2), "{name}: counts, history");
            assert_eq!(want.deltas.len(), got.deltas.len());
            for ((ka, a), (kb, b)) in want.deltas.iter().zip(&got.deltas) {
                assert_eq!(ka, kb);
                assert!(a == b, "{name}: the changes of {ka:?}");
                assert!(a.iter().any(|&x| x != 0), "{name}: {ka:?} learned");
            }
            for ((ia, a), (ib, b)) in want.traces.iter().zip(&got.traces) {
                assert_eq!(ia, ib);
                assert!(a == b, "{name}: block {ia}'s trace");
            }
            // the rms last: everything above does not depend on the thread count, the rms does
            assert_eq!(th0, th1, "{name}: logits, State, changes and traces bit for bit, but RAYON_NUM_THREADS changed while the two sessions ran (another test set it), so their rms are not comparable: run this test on its own");
            for (i, (a, b)) in want.rms.iter().zip(&got.rms).enumerate() {
                assert!(a == b, "{name}: the learning step's rms differs at step {i}");
            }
            n += 1;
        }
    }
    println!("fast4 e-prop: {n} sessions of 300 tokens bit for bit");
}
