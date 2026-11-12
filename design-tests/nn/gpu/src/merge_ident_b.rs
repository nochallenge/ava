// TESTING / DESIGN ONLY — merge check B (runs/v51/merge_b.md): v51 Part B2's options compute the same numbers in the
// merged main crate as in the crate they were built in (design-tests/nn/gpu-v51b), bit for bit.
//
// Small CPU models (f32, d 64, 3 blocks, the glance vocabulary) with Ava 0.4b's block recipe (the block flags of
// runs/v51/run_v51b2.sh's args_for) and --thalamus (TH), --cerebellum-head 2048 (CB, 32 d), both (THCB), and both with
// --neuromod 3 (THCBN). Every weight is set from a fixed seed (as src/v51b_golden.rs does), so no unseeded init is left.
// On one fixed input (2 rows x 32 tokens) each of these is hashed (FNV-1a over the f32 bits):
//   the window's logits; the training loss and every parameter's gradient (candle's autograd); two pieces read with a
//   carried State; the one-token stream's logits and greedy picks; the parts' signals after each streamed token (TH's
//   gates, CB's size, the nucleus' NE / ACh / DA); the chat app's traced step with int8 and the fused step switched on
//   (logits, spikes, hidden units, signals); the cerebellum's online rule over the stream (CB arms); one e-prop session
//   and one online-backprop session over the stream (blocks 1 and 2, all four sites); every weight after one
//   homeostasis nudge.
// The model fingerprint is not compared: the main crate's config text also holds D2's `dale_proj` field.
// AVA_MERGE_OUT=FILE writes this run's lines to FILE. AVA_MERGE_REF=write writes runs/v51/merge_b_ref.tsv; any other
// run compares against that file line by line (AVA_MERGE_DIR: the crate folder holding runs/, default this crate).
// Run at 1 thread (RAYON_NUM_THREADS=1, AVA_THREADS=1).

use crate::arch::Arch;
use crate::eprop::{Rule, Site};
use crate::model::{Config, Kind, Model};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::path::PathBuf;

const RECIPE: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense \
                      --resid-fix --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif \
                      --homeostasis 0.1";
const ARMS: [(&str, &str); 4] = [
    ("TH", "--thalamus"),
    ("CB", "--cerebellum-head 2048"),
    ("THCB", "--thalamus --cerebellum-head 2048"),
    ("THCBN", "--thalamus --cerebellum-head 2048 --neuromod 3"),
];
const B: usize = 2;
const T: usize = 32;
const S: usize = 24;

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let u = (x >> 11) as f64 / (1u64 << 53) as f64;
            ((u * 2.0 - 1.0) * 3f64.sqrt()) as f32
        })
        .collect()
}

/// Every weight from the seed (src/v51b_golden.rs's rule): random weights get noise of their snapped spread, all-zero
/// weights small noise (so every part is exercised), constant non-zero weights stay.
fn build(cfg: Config, seed: u64) -> (VarMap, Model) {
    let vm = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu)).unwrap();
    {
        let data = vm.data().lock().unwrap();
        let mut names: Vec<&String> = data.keys().collect();
        names.sort();
        for (k, name) in names.into_iter().enumerate() {
            let var = &data[name];
            let v: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
            let n = v.len() as f64;
            let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
            let sd = (v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
            let zero = v.iter().all(|&x| x == 0.0);
            if sd == 0.0 && !zero {
                continue;
            }
            let spreads = [crate::model::INIT_STD, 1.0 / (cfg.d as f64).sqrt(), cfg.resid_std()];
            let snap = spreads.iter().copied().min_by(|a, b| (a.ln() - sd.ln()).abs().total_cmp(&(b.ln() - sd.ln()).abs())).unwrap();
            let s = if zero { 0.05 } else { snap };
            let w: Vec<f32> = noise(v.len(), seed.wrapping_mul(1000).wrapping_add(k as u64)).iter().map(|&z| (s * z as f64) as f32).collect();
            var.set(&Tensor::from_vec(w, var.shape(), &Device::Cpu).unwrap()).unwrap();
        }
    }
    m.set_track_activity(false);
    (vm, m)
}

fn fnv(h: &mut u64, t: &Tensor) -> usize {
    let v: Vec<f32> = t.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap();
    for x in &v {
        for b in x.to_bits().to_le_bytes() {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    v.len()
}

fn hash(ts: &[&Tensor]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let n: usize = ts.iter().map(|t| fnv(&mut h, t)).sum();
    format!("{h:016x}\t{n}")
}

fn hash_f32(v: &[f32]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{h:016x}\t{}", v.len())
}

/// The parts' signals as f32 values (TH's gates per block, CB's size and online share, the nucleus' three; NaN marks
/// a missing one).
fn signals(m: &Model) -> Vec<f32> {
    let s = m.part_signals();
    let mut v: Vec<f32> = s.th.iter().flat_map(|g| g.iter().copied()).collect();
    v.push(s.cb.unwrap_or(f32::NAN));
    v.push(s.cb_online.unwrap_or(f32::NAN));
    v.extend(s.nm.map(|n| n.to_vec()).unwrap_or_else(|| vec![f32::NAN; 3]));
    v
}

/// B rows of T + 1 fixed token ids below the glance vocabulary (xorshift; no file needed).
fn tokens(vocab: usize) -> Vec<u32> {
    let mut x = 0x51B0_2026_1105_0002u64;
    (0..B * (T + 1))
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % vocab as u64) as u32
        })
        .collect()
}

fn names(vm: &VarMap) -> Vec<String> {
    let mut n: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
    n.sort();
    n
}

fn pin(arm: &str, flags: &str, out: &mut Vec<String>) {
    let cpu = Device::Cpu;
    let ar = Arch::from_flags(&format!("{RECIPE} {flags}")).unwrap();
    let cfg = Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 64, 3, 64).with_glance().unwrap().with_arch(ar).unwrap();
    let (vm, m) = build(cfg, 51);
    m.set_int8(false).unwrap();
    m.set_fused(false);
    let mut put = |part: &str, v: String| out.push(format!("{arm}\t{part}\t{v}"));
    put("arch", cfg.arch.signature());
    let tk = tokens(cfg.vocab);
    let ids: Vec<u32> = (0..B).flat_map(|r| tk[r * (T + 1)..r * (T + 1) + T].to_vec()).collect();
    let tg: Vec<u32> = (0..B).flat_map(|r| tk[r * (T + 1) + 1..(r + 1) * (T + 1)].to_vec()).collect();
    let x = Tensor::from_vec(ids, (B, T), &cpu).unwrap();
    let y = Tensor::from_vec(tg, (B, T), &cpu).unwrap();
    // the window (and its signals at the last row)
    m.set_part_signals(true);
    let lw = m.forward(&x).unwrap();
    put("forward", hash(&[&lw]));
    put("forward_signals", hash_f32(&signals(&m)));
    m.set_part_signals(false);
    // two pieces with a carried State
    let mut st = m.state(B).unwrap();
    let mut p0 = 0;
    for (k, len) in [13usize, 19].into_iter().enumerate() {
        let piece: Vec<u32> = (0..B).flat_map(|r| tk[r * (T + 1) + p0..r * (T + 1) + p0 + len].to_vec()).collect();
        let l = m.forward_with_state(&Tensor::from_vec(piece, (B, len), &cpu).unwrap(), &mut st).unwrap();
        put(&format!("pieces {k}"), hash(&[&l]));
        p0 += len;
    }
    // the one-token stream (as v50_talk talks), its greedy picks and the parts' signals of every token
    let row: Vec<u32> = tk[..S + 1].to_vec();
    m.set_inference(true);
    m.set_part_signals(true);
    let mut s = m.stream().unwrap();
    let (mut ls, mut sg) = (Vec::new(), Vec::new());
    for &t in &row[..S] {
        ls.push(m.step(&mut s, t).unwrap());
        sg.extend(signals(&m));
    }
    put("stream", hash(&ls.iter().collect::<Vec<_>>()));
    let greedy: Vec<u32> = ls.iter().map(|l| l.argmax(0).unwrap().to_scalar::<u32>().unwrap()).collect();
    put("stream_greedy", format!("{greedy:?}"));
    put("stream_signals", hash_f32(&sg));
    // the chat app's step: int8 and the fused step on, traced
    m.set_int8(true).unwrap();
    m.set_fused(true);
    let mut s = m.stream().unwrap();
    let (mut ls, mut tr, mut sg) = (Vec::new(), Vec::new(), Vec::new());
    for &t in &row[..S] {
        let (l, trace) = m.step_trace(&mut s, t).unwrap();
        ls.push(l);
        for ly in &trace.layers {
            tr.push(ly.spikes.clone());
            tr.push(ly.hidden.clone());
        }
        sg.extend(signals(&m));
    }
    put("int8_trace", hash(&ls.iter().collect::<Vec<_>>()));
    put("int8_trace_layers", hash(&tr.iter().collect::<Vec<_>>()));
    put("int8_trace_signals", hash_f32(&sg));
    m.set_fused(false);
    m.set_int8(false).unwrap();
    m.set_part_signals(false);
    m.set_inference(false); // the online sessions run as v51_online runs them (inference off)
    let one = |t: u32| Tensor::new(&[[t]], &cpu).unwrap();
    // the cerebellum's online rule (CB arms), as v51_online and ava_live --cb-online run it
    if cfg.arch.v51b.cb_units > 0 {
        let rate = m.cb_rate(0.3).unwrap();
        put("cb_rate", format!("{:08x}", rate.to_bits()));
        m.cb_online_begin(1, rate).unwrap();
        let mut st = m.state(1).unwrap();
        let mut ls = Vec::new();
        for j in 0..S {
            let lg = m.forward_with_state(&one(row[j]), &mut st).unwrap();
            m.cb_learn(&lg, &[row[j + 1]]).unwrap();
            ls.push(lg);
        }
        put("cb_online", hash(&ls.iter().collect::<Vec<_>>()));
        put("cb_online_sum", hash_f32(&m.cb_online_sum().unwrap_or_default()));
        m.cb_online_end();
    }
    // one e-prop session and one online-backprop session (segments of 8), blocks 1 and 2, all four sites
    let blocks = [1usize, 2];
    for (tag, rule, seg) in [("eprop", Rule::EProp, 1usize), ("dyn", Rule::Backprop, 8)] {
        m.plastic_begin(1, &blocks, &Site::ALL).unwrap();
        m.plastic_probes(true);
        let mut st = m.state(1).unwrap();
        let mut ls = Vec::new();
        for p0 in (0..S).step_by(seg) {
            let k = seg.min(S - p0);
            let lg = m.forward_with_state(&Tensor::from_vec(row[p0..p0 + k].to_vec(), (1, k), &cpu).unwrap(), &mut st).unwrap();
            let ce = candle_nn::loss::cross_entropy(&lg, &Tensor::from_vec(row[p0 + 1..p0 + k + 1].to_vec(), k, &cpu).unwrap()).unwrap();
            let loss = if seg > 1 { ce.affine(k as f64, 0.0).unwrap() } else { ce };
            m.plastic_learn(&loss.backward().unwrap(), 1e-3, rule).unwrap();
            st.detach_all();
            ls.push(lg.detach());
        }
        put(tag, hash(&ls.iter().collect::<Vec<_>>()));
        let mut ds = Vec::new();
        for &i in &blocks {
            for site in Site::ALL {
                ds.push(m.plastic_delta_sum(i, site).unwrap().unwrap());
            }
        }
        put(&format!("{tag}_changes"), hash(&ds.iter().collect::<Vec<_>>()));
        m.plastic_end();
    }
    // a training gradient (CPU autograd), then the homeostasis nudge
    let (loss, _, g) = m.loss_and_grads_autograd(&x, &y, None, false).unwrap();
    put("loss", format!("{:08x}\t{}", loss.to_scalar::<f32>().unwrap().to_bits(), loss.to_scalar::<f32>().unwrap()));
    let nm = names(&vm);
    let data = vm.data().lock().unwrap();
    for n in &nm {
        match g.get(data[n].as_tensor()) {
            Some(gr) => put(&format!("grad {n}"), hash(&[gr])),
            None => put(&format!("grad {n}"), "none".into()),
        }
    }
    drop(data);
    m.homeostasis_step().unwrap();
    let data = vm.data().lock().unwrap();
    let ws: Vec<Tensor> = nm.iter().map(|n| data[n].as_tensor().clone()).collect();
    put("after_step", hash(&ws.iter().collect::<Vec<_>>()));
}

#[test]
fn merge_keeps_v51b_options_bit_identical() {
    let dir = std::env::var("AVA_MERGE_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let refp = dir.join("runs/v51/merge_b_ref.tsv");
    let mut out = Vec::new();
    for (arm, flags) in ARMS {
        pin(arm, flags, &mut out);
    }
    let text: String = out.iter().map(|l| format!("{l}\n")).collect();
    if let Ok(p) = std::env::var("AVA_MERGE_OUT") {
        std::fs::write(&p, &text).unwrap();
        eprintln!("merge check B: {} lines written to {p}", out.len());
    }
    if std::env::var("AVA_MERGE_REF").is_ok_and(|v| v == "write") {
        std::fs::write(&refp, &text).unwrap();
        eprintln!("merge check B: {} lines written to {}", out.len(), refp.display());
        return;
    }
    let Ok(want) = std::fs::read_to_string(&refp) else {
        eprintln!("merge check B: no {} (nothing to compare)", refp.display());
        return;
    };
    let want: Vec<&str> = want.lines().collect();
    let bad: Vec<String> = want.iter().zip(&out).filter(|(a, b)| **a != b.as_str()).map(|(a, b)| format!("was {a}\n now {b}")).collect();
    assert_eq!(want.len(), out.len(), "line count");
    assert!(bad.is_empty(), "{} of {} lines differ:\n{}", bad.len(), out.len(), bad.join("\n"));
    eprintln!("merge check B: all {} lines bit-identical to {}", out.len(), refp.display());
}
