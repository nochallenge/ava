// TESTING / DESIGN ONLY — merge check (runs/v51/merge_a.md): round 1's trained models compute the same numbers
// before and after a source merge, bit for bit.
//
// For each of round 1's seed-1 models (runs/v50/{B,D,L,N}-s1/model.safetensors; CPU, f32) and one fixed 64-token
// input: the window's logits (Model::forward), the batched State read of the same tokens and the State saved after it,
// the one-token stream's logits for all 64 tokens (Model::step, as v50_talk talks: inference on, the --dale cache on),
// the model fingerprint, the training loss and every parameter's gradient (candle's autograd), and every weight after
// one homeostasis nudge and one --dale decay (what train_gpu does after an optimizer step). Each is hashed (FNV-1a over
// the f32 bits). AVA_MERGE_REF=write writes runs/v51/merge_a_ref.tsv (run once on the source before the merge); any
// other run compares against that file line by line. AVA_MERGE_DIR is the crate folder holding runs/ (default: this
// crate). Without the models or the file the test says so and passes.

use crate::model::Kind;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use std::path::PathBuf;

const ARMS: [&str; 4] = ["B", "D", "L", "N"];
const T: usize = 64;

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

/// T + 1 fixed token ids below 3800 (xorshift; no file needed).
fn tokens() -> Vec<u32> {
    let mut x = 0x5151_2026_1105_0001u64;
    (0..=T)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 3800) as u32
        })
        .collect()
}

fn names(vm: &VarMap) -> Vec<String> {
    let mut n: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
    n.sort();
    n
}

fn pin(arm: &str, path: &std::path::Path, out: &mut Vec<String>) {
    let cpu = Device::Cpu;
    let (cfg, vm, m) = crate::ckpt::load_auto(path, Kind::D, 4096, &cpu).unwrap();
    m.set_track_activity(false);
    m.set_int8(false).unwrap();
    m.set_fused(false);
    let mut put = |part: &str, v: String| out.push(format!("{arm}\t{part}\t{v}"));
    put("arch", cfg.arch.signature());
    let tk = tokens();
    let x = Tensor::from_vec(tk[..T].to_vec(), (1, T), &cpu).unwrap();
    let y = Tensor::from_vec(tk[1..].to_vec(), (1, T), &cpu).unwrap();
    // the window
    let lw = m.forward(&x).unwrap();
    put("forward", hash(&[&lw]));
    let probe: Vec<f32> = lw.get(T - 1).unwrap().narrow(0, 0, 4).unwrap().to_vec1().unwrap();
    put("forward_probe", format!("{probe:?}"));
    // the batched State, and the State saved after it
    let mut st = m.state(1).unwrap();
    let lf = m.forward_with_state(&x, &mut st).unwrap();
    put("state_read", hash(&[&lf]));
    let p = crate::testkit::scratch(&format!("merge_ident_{arm}")).join("state.safetensors");
    st.save(&m, &p).unwrap();
    let map = candle_core::safetensors::load(&p, &cpu).unwrap();
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    let ts: Vec<Tensor> = keys.iter().map(|k| map[*k].to_dtype(DType::F32).unwrap()).collect();
    put("state_saved", hash(&ts.iter().collect::<Vec<_>>()));
    put("state_names", keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(","));
    put("fingerprint", format!("{:016x}", m.fingerprint().unwrap()));
    // the one-token stream, as v50_talk talks; greedy picks of every step too
    m.set_inference(true);
    m.set_dale_cache(true);
    let mut s = m.stream().unwrap();
    let mut ls = Vec::new();
    for &t in &tk[..T] {
        ls.push(m.step(&mut s, t).unwrap());
    }
    put("stream", hash(&ls.iter().collect::<Vec<_>>()));
    let greedy: Vec<u32> = ls.iter().map(|l| l.argmax(0).unwrap().to_scalar::<u32>().unwrap()).collect();
    put("stream_greedy", format!("{greedy:?}"));
    m.set_dale_cache(false);
    m.set_inference(false);
    // a training gradient (CPU autograd), then the post-step updates
    let (loss, _, g) = m.loss_and_grads_autograd(&x, &y, None, false).unwrap();
    put("loss", format!("{:08x}\t{}", loss.to_scalar::<f32>().unwrap().to_bits(), loss.to_scalar::<f32>().unwrap()));
    let nm = names(&vm);
    let data = vm.data().lock().unwrap();
    let gs: Vec<Tensor> = nm.iter().filter_map(|n| g.get(data[n].as_tensor()).cloned()).collect();
    put("grads", hash(&gs.iter().collect::<Vec<_>>()));
    put("var_names", nm.join(","));
    if let Ok(dump) = std::env::var("AVA_MERGE_DUMP") {
        // diagnosis only: every parameter's gradient hash and its sum of |g|
        let mut s = String::new();
        for n in &nm {
            if let Some(gr) = g.get(data[n].as_tensor()) {
                let a: f64 = gr.abs().unwrap().sum_all().unwrap().to_dtype(DType::F64).unwrap().to_scalar().unwrap();
                s += &format!("{n}\t{}\t{a:.9e}\n", hash(&[gr]));
            }
        }
        std::fs::write(format!("{dump}_{arm}.tsv"), s).unwrap();
    }
    drop(data);
    m.homeostasis_step().unwrap();
    m.dale_decay(2e-3 * 0.1).unwrap();
    let data = vm.data().lock().unwrap();
    let ws: Vec<Tensor> = nm.iter().map(|n| data[n].as_tensor().clone()).collect();
    put("after_step", hash(&ws.iter().collect::<Vec<_>>()));
}

#[test]
fn merge_keeps_round1_models_bit_identical() {
    let dir = std::env::var("AVA_MERGE_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let refp = dir.join("runs/v51/merge_a_ref.tsv");
    let mut out = Vec::new();
    let mut seen = 0;
    for arm in ARMS {
        let p = dir.join(format!("runs/v50/{arm}-s1/model.safetensors"));
        if p.exists() {
            pin(arm, &p, &mut out);
            seen += 1;
        } else {
            eprintln!("merge check: no {} (skipped)", p.display());
        }
    }
    let text: String = out.iter().map(|l| format!("{l}\n")).collect();
    if std::env::var("AVA_MERGE_REF").is_ok_and(|v| v == "write") {
        std::fs::write(&refp, &text).unwrap();
        eprintln!("merge check: {} lines of {seen} models written to {}", out.len(), refp.display());
        return;
    }
    let Ok(want) = std::fs::read_to_string(&refp) else {
        eprintln!("merge check: no {} (nothing to compare; {seen} models ran)", refp.display());
        return;
    };
    let want: Vec<&str> = want.lines().collect();
    let bad: Vec<String> = want.iter().zip(&out).filter(|(a, b)| **a != b.as_str()).map(|(a, b)| format!("was {a}\n now {b}")).collect();
    assert_eq!(want.len(), out.len(), "line count (models present now: {seen})");
    assert!(bad.is_empty(), "{} of {} lines differ:\n{}", bad.len(), out.len(), bad.join("\n"));
    eprintln!("merge check: all {} lines of {seen} models bit-identical to {}", out.len(), refp.display());
}
