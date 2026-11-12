// TESTING / DESIGN ONLY — ava_gpu::fast4 profile of the State-path online session (where a token's time goes).

use crate::eprop::{Rule, Site};
use crate::model::Kind;
use candle_core::{Device, Tensor};
use std::time::Instant;

/// AVA_F4_MODEL=path: e-prop (blocks 6, 7; half-life 128) on the State path, per token: forward, backward, learn.
#[test]
#[ignore]
fn fast4_profile_state_path() {
    let Ok(path) = std::env::var("AVA_F4_MODEL") else { return };
    let dev = Device::Cpu;
    let (cfg, _vm, m) = crate::ckpt::load_auto_lean(std::path::Path::new(&path), Kind::D, 4096, &dev).unwrap();
    m.set_track_activity(false);
    let n = 60usize;
    let toks: Vec<u32> = (0..=n).map(|i| ((i * 37 + 11) % cfg.vocab) as u32).collect();
    let blocks = [cfg.layers - 2, cfg.layers - 1];
    m.plastic_begin(1, &blocks, &Site::ALL).unwrap();
    m.plastic_set_half_lives(&[128.0]).unwrap();
    m.plastic_probes(true);
    let mut st = m.state(1).unwrap();
    let (mut tf, mut tb, mut tl) = (0f64, 0f64, 0f64);
    for j in 0..n {
        let t0 = Instant::now();
        let lg = m.forward_with_state(&Tensor::new(&[[toks[j]]], &dev).unwrap(), &mut st).unwrap();
        let t1 = Instant::now();
        let loss = candle_nn::loss::cross_entropy(&lg, &Tensor::new(&[toks[j + 1]], &dev).unwrap()).unwrap();
        let g = loss.backward().unwrap();
        let t2 = Instant::now();
        m.plastic_learn(&g, 1e-3, Rule::EProp).unwrap();
        st.detach_all();
        let t3 = Instant::now();
        if j >= 10 {
            tf += (t1 - t0).as_secs_f64();
            tb += (t2 - t1).as_secs_f64();
            tl += (t3 - t2).as_secs_f64();
        }
    }
    let k = (n - 10) as f64;
    m.plastic_end();
    let mut st = m.state(1).unwrap();
    let t0 = Instant::now();
    for j in 0..n {
        m.forward_with_state(&Tensor::new(&[[toks[j]]], &dev).unwrap(), &mut st).unwrap();
    }
    let fz = t0.elapsed().as_secs_f64() / n as f64;
    println!(
        "fast4 profile ({} threads): e-prop forward {:.2} ms, backward {:.2} ms, learn {:.2} ms per token; frozen State path {:.2} ms",
        candle_core::utils::get_num_threads(),
        tf / k * 1e3,
        tb / k * 1e3,
        tl / k * 1e3,
        fz * 1e3
    );
}
