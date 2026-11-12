// TESTING / DESIGN ONLY — v51 fast4: the cerebellum's online table in the fused step (`Model::fast4_state_step`) and
// its fused learning step (`Model::fast4_cb_learn`) against the State path's session (`forward_with_state` +
// `cb_learn`), CPU only.
//
// Pre-set bar (written before the first run): bit for bit. Over 300 tokens of one row, every step's logits and
// returned nats, the final State, the row's stored table, its scale and its update count are the same, with int8 off
// (f32) and on, for a half-life of 256 tokens (CB2's), no fading (inf) and a half-life of 2 tokens (the scale folds).

use crate::arch::Arch;
use crate::model::fstep::arch::state_tensors;
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, tokens};
use candle_core::{DType, Tensor};

const AVA04: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
                     --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";
const V: usize = 293;

fn vals(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    vals(t).iter().map(|x| x.to_bits()).collect()
}

struct Out {
    logits: Vec<Vec<u32>>,
    nats: Vec<u64>,
    state: (Vec<Tensor>, Vec<usize>, Vec<Vec<u32>>),
    table: Vec<u32>,
    scale: f64,
    updates: usize,
}

/// One row's session over `toks`: the State path (`fused` false) or the fused step + fused learning step.
fn session(m: &Model, toks: &[u32], fused: bool, half: f64, lr: f32) -> Out {
    let dev = candle_core::Device::Cpu;
    m.cb_online_begin(1, lr).unwrap();
    m.cb_online_set_half_life(0, half).unwrap();
    let mut st = m.state(1).unwrap();
    let (mut logits, mut nats) = (Vec::new(), Vec::new());
    for w in toks.windows(2) {
        let (t, y) = (w[0], w[1]);
        if fused {
            let lg = m.fast4_state_step(&mut st, t).unwrap().expect("the fused step runs");
            nats.push(m.fast4_cb_learn(&vals(&lg), y).unwrap().to_bits());
            logits.push(bits(&lg));
        } else {
            let lg = m.forward_with_state(&Tensor::new(&[[t]], &dev).unwrap(), &mut st).unwrap();
            nats.push(m.cb_learn(&lg, &[y]).unwrap().to_bits());
            logits.push(bits(&lg));
        }
    }
    let (table, scale, updates) = m.cb_online_with(|o| (o.raw(0).0.iter().map(|x| x.to_bits()).collect(), o.raw(0).1, o.updates[0])).unwrap();
    m.cb_online_end();
    Out { logits, nats, state: state_tensors(&st), table, scale, updates }
}

#[test]
fn fast4_the_cerebellums_online_table_is_the_state_paths_bit_for_bit() {
    let c = Config::with_kind(Kind::D, V, 128, 3, 64).with_arch(Arch::from_flags(&format!("{AVA04} --cerebellum-head 512")).unwrap()).unwrap();
    let (_, m) = build_awake(c, 301);
    m.set_fast4(true);
    let toks = tokens(1, 301, V, 302);
    let mut n = 0;
    for int8 in [false, true] {
        m.set_int8(int8).unwrap();
        for (half, lr) in [(256.0, 0.05f32), (f64::INFINITY, 0.05), (2.0, 0.2)] {
            let name = format!("int8 {int8}, half-life {half}");
            m.set_fused(false);
            let want = session(&m, &toks, false, half, lr);
            m.set_fused(true);
            assert!(m.fused_blocker().is_none(), "{name}: {:?}", m.fused_blocker());
            let got = session(&m, &toks, true, half, lr);
            m.set_fused(false);
            assert_eq!(want.logits.len(), 300);
            for (i, (a, b)) in want.logits.iter().zip(&got.logits).enumerate() {
                assert!(a == b, "{name}: logits differ at step {i}");
            }
            assert_eq!(want.nats, got.nats, "{name}: nats");
            for (j, (a, b)) in want.state.0.iter().zip(&got.state.0).enumerate() {
                assert!(bits(a) == bits(b), "{name}: state tensor {j}");
            }
            assert_eq!((&want.state.1, &want.state.2), (&got.state.1, &got.state.2), "{name}: counts, history");
            assert!(want.table == got.table, "{name}: the table");
            assert_eq!(want.scale.to_bits(), got.scale.to_bits(), "{name}: scale");
            assert_eq!((want.updates, got.updates), (300, 300), "{name}: updates");
            assert!(want.table.iter().any(|&x| x != 0), "{name}: the table learned");
            n += 1;
        }
    }
    m.set_int8(false).unwrap();
    // without set_fast4 an online table keeps the plain step
    m.set_fast4(false);
    m.cb_online_begin(1, 0.1).unwrap();
    m.set_fused(true);
    assert!(m.fused_blocker().is_some());
    m.cb_online_end();
    m.set_fused(false);
    println!("fast4 CB online: {n} sessions of 300 tokens bit for bit");
}
