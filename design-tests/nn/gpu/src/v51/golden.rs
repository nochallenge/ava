// TESTING / DESIGN ONLY — v51 golden test: today's outputs (written before any v51 code) pinned bit for bit.
//
// With every v51 option off, a model must compute exactly what it computed before v51: the 0.4 recipe (glance, magnitude
// code, engram) and today's plain D, through the full forward, the batched State in pieces, the token-by-token stream,
// the traced step, the int8 talking path (fused where it applies), a training gradient, the homeostasis nudge and a
// saved State. Each is hashed (FNV-1a over the f32 bits) and compared with src/v51/golden_04.txt, which the untouched
// source wrote (AVA_V51_GOLDEN=write). A mismatch names every part that differs.
// Merge into the main crate (runs/v51/merge_a.md): golden_04.txt was rewritten by the main crate's own source just
// before the merge (v50 and D2 in, v51 not yet). Against the file v51 was written with, only `ava04.state` and
// `plainD.state` changed: a saved State holds the model fingerprint, which hashes the config's Debug text, and v50's
// Arch fields (dale, dale_proj, loops, neuromod) are in that text. Every other line is the same.

use crate::arch::Arch;
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, tokens};
use crate::testkit::scratch;
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use std::collections::BTreeMap;

const CPU: Device = Device::Cpu;
const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/v51/golden_04.txt");
pub(crate) const AVA04: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense \
    --resid-fix --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";

fn fnv(h: &mut u64, t: &Tensor) {
    let v: Vec<f32> = t.flatten_all().unwrap().to_dtype(candle_core::DType::F32).unwrap().to_vec1().unwrap();
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn hash(ts: &[&Tensor]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for t in ts {
        fnv(&mut h, t);
    }
    format!("{h:016x}")
}

/// The 0.4 recipe on a d128 model of 3 blocks (2 memory heads of 64), the glance, the magnitude code, the engram in block 1.
pub(crate) fn cfg04() -> Config {
    Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 128, 3, 64)
        .with_glance()
        .unwrap()
        .with_mag()
        .unwrap()
        .with_engram(1)
        .unwrap()
        .with_arch(Arch::from_flags(AVA04).unwrap())
        .unwrap()
}

fn ids(b: usize, t: usize, vocab: usize, seed: u64) -> (Tensor, Tensor) {
    let v = tokens(b, t + 1, vocab, seed);
    let x: Vec<u32> = (0..b).flat_map(|r| v[r * (t + 1)..r * (t + 1) + t].to_vec()).collect();
    let y: Vec<u32> = (0..b).flat_map(|r| v[r * (t + 1) + 1..(r + 1) * (t + 1)].to_vec()).collect();
    (Tensor::from_vec(x, (b, t), &CPU).unwrap(), Tensor::from_vec(y, (b, t), &CPU).unwrap())
}

/// Every pinned part of one model: name -> hash.
fn pin(tag: &str, vm: &VarMap, m: &Model, out: &mut BTreeMap<String, String>) {
    let vocab = m.cfg.vocab;
    let (b, t) = (2, 40);
    let (x, y) = ids(b, t, vocab, 501);
    m.set_track_activity(false);
    out.insert(format!("{tag}.forward"), hash(&[&m.forward(&x).unwrap()]));
    let ar = m.arch_path();
    if ar || m.cfg.kind == Kind::D {
        // the batched State in two pieces, and the State saved after it
        let mut st = m.state(b).unwrap();
        let a = m.forward_with_state(&x.narrow(1, 0, 17).unwrap(), &mut st).unwrap();
        let c = m.forward_with_state(&x.narrow(1, 17, t - 17).unwrap(), &mut st).unwrap();
        out.insert(format!("{tag}.pieces"), hash(&[&a, &c]));
        let dir = scratch(&format!("v51_golden_{tag}"));
        let p = dir.join("state.safetensors");
        st.save(m, &p).unwrap();
        let map = candle_core::safetensors::load(&p, &CPU).unwrap();
        let mut keys: Vec<&String> = map.keys().collect();
        keys.sort();
        let ts: Vec<Tensor> = keys.iter().map(|k| map[*k].to_dtype(candle_core::DType::F32).unwrap()).collect();
        out.insert(format!("{tag}.state_names"), keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(","));
        out.insert(format!("{tag}.state"), hash(&ts.iter().collect::<Vec<_>>()));
    }
    // the token-by-token stream and the traced step
    let toks: Vec<u32> = x.get(0).unwrap().to_vec1().unwrap();
    let mut s = m.stream().unwrap();
    let mut ls = Vec::new();
    for &tk in &toks[..12] {
        ls.push(m.step(&mut s, tk).unwrap());
    }
    out.insert(format!("{tag}.stream"), hash(&ls.iter().collect::<Vec<_>>()));
    let mut s = m.stream().unwrap();
    let mut tr = Vec::new();
    for &tk in &toks[..6] {
        let (l, st) = m.step_trace(&mut s, tk).unwrap();
        tr.push(l);
        for ly in st.layers {
            tr.push(ly.spikes);
            tr.push(ly.hidden);
        }
    }
    out.insert(format!("{tag}.trace"), hash(&tr.iter().collect::<Vec<_>>()));
    // a training gradient (CPU autograd) and the homeostasis nudge after it
    let (loss, _, g) = m.loss_and_grads_autograd(&x, &y, None, false).unwrap();
    let data = vm.data().lock().unwrap();
    let mut names: Vec<String> = data.keys().cloned().collect();
    names.sort();
    let mut gs = vec![loss];
    for n in &names {
        if let Some(gr) = g.get(data[n].as_tensor()) {
            gs.push(gr.clone());
        }
    }
    out.insert(format!("{tag}.grads"), hash(&gs.iter().collect::<Vec<_>>()));
    out.insert(format!("{tag}.var_names"), names.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(","));
    drop(data);
    m.homeostasis_step().unwrap();
    let data = vm.data().lock().unwrap();
    let th: Vec<Tensor> = names.iter().filter(|n| n.ends_with("theta")).map(|n| data[n].as_tensor().clone()).collect();
    out.insert(format!("{tag}.homeo"), hash(&th.iter().collect::<Vec<_>>()));
    drop(data);
    out.insert(format!("{tag}.weights"), format!("{:?}", m.cfg.weight_counts()));
    out.insert(format!("{tag}.signature"), m.cfg.arch.signature());
    // the int8 talking path, fused where it applies
    m.set_int8(true).unwrap();
    m.set_fused(true);
    out.insert(format!("{tag}.fused_blocker"), format!("{:?}", m.fused_blocker()));
    let mut s = m.stream().unwrap();
    let mut ls = Vec::new();
    for &tk in &toks[..12] {
        ls.push(m.step(&mut s, tk).unwrap());
    }
    out.insert(format!("{tag}.int8_stream"), hash(&ls.iter().collect::<Vec<_>>()));
    m.set_fused(false);
    m.set_int8(false).unwrap();
}

fn compute() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let (vm, m) = build_awake(cfg04(), 51);
    pin("ava04", &vm, &m, &mut out);
    // today's plain D (no block options: the fused-kernel path's CPU twin)
    let (vm, m) = build_awake(Config::with_kind(Kind::D, 258, 64, 2, 64), 52);
    pin("plainD", &vm, &m, &mut out);
    out
}

#[test]
fn v51_off_is_bit_identical_to_before() {
    let got = compute();
    let text: String = got.iter().map(|(k, v)| format!("{k}\t{v}\n")).collect();
    if std::env::var("AVA_V51_GOLDEN").is_ok_and(|v| v == "write") {
        std::fs::write(GOLDEN, &text).unwrap();
        return;
    }
    let want = std::fs::read_to_string(GOLDEN).expect("src/v51/golden_04.txt (written by the untouched source)");
    let want: BTreeMap<String, String> = want.lines().filter_map(|l| l.split_once('\t')).map(|(k, v)| (k.to_string(), v.to_string())).collect();
    let bad: Vec<String> = want.iter().filter(|(k, v)| got.get(*k) != Some(*v)).map(|(k, v)| format!("{k}: want {v}, got {:?}", got.get(k))).collect();
    assert!(bad.is_empty() && got.len() == want.len(), "v51 options off changed today's outputs:\n{}", bad.join("\n"));
}
