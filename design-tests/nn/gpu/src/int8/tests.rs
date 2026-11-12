// TESTING / DESIGN ONLY — tests of src/int8.rs (E10 int8 weights, E2 event sums) and of Model::set_int8, CPU only.
//
// Tolerances, set before the first run: the kernel equals its scalar twin bit for bit and the dense f32 product of the
// dequantized weights to 1e-5 relative; event sums equal dense int8 sums bit for bit; a tiny everything-on model's
// int8 logits are within 3% relative RMS of its f32 logits; int8 token by token equals int8 batched within 1e-4 (the
// --tem configs within the 5e-4 their f32 paths already need); int8 off gives f32 bit for bit.
// The 3% bound failed (tiny random spiking models are chaotic, see `int8_is_as_near_f32_as_its_own_weight_error`)
// and was replaced by the first-token and same-size-control bounds there; the real models are judged by val BPB.

use super::*;
use crate::arch::Arch;
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, in_pieces, max_diff, one_pass, tokens, v40_cfg};
use crate::testkit::unit_noise;
use candle_core::{DType, Device, Tensor};

const CPU: Device = Device::Cpu;

fn rnd(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    unit_noise(n, seed).iter().map(|z| z * scale).collect()
}

/// A row with about `rate` of its inputs non-zero: 0/1 spikes, or graded (-2..2) ones.
fn sparse(n: usize, rate: f32, graded: bool, seed: u64) -> Vec<f32> {
    let z = unit_noise(2 * n, seed);
    (0..n)
        .map(|i| {
            let on = (z[i] * 0.5 + 0.5).clamp(0.0, 1.0) < rate;
            match (on, graded) {
                (false, _) => 0.0,
                (true, false) => 1.0,
                (true, true) => [-2.0, -1.0, 1.0, 2.0][((z[n + i] * 0.5 + 0.5).clamp(0.0, 0.999) * 4.0) as usize],
            }
        })
        .collect()
}

fn dense_ref(x: &[f32], w: &[f32], m: usize, din: usize, dout: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * dout];
    for r in 0..m {
        for c in 0..dout {
            y[r * dout + c] = (0..din).map(|j| x[r * din + j] as f64 * w[j * dout + c] as f64).sum::<f64>() as f32;
        }
    }
    y
}

fn run(q: &QW, x: &[f32], m: usize, event: f64, threads: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * q.dout];
    q.matmul_rows(x, m, &mut y, event, threads);
    y
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The packed kernel (AVX2 when present) gives its scalar twin's bits, the dequantized product to 1e-5, and the
/// reference matvec of ava_parts::quant to float rounding; shapes include a partial last tile (4112 = 64·64 + 16).
#[test]
fn the_kernel_is_the_reference_int8_product() {
    for (din, dout, m) in [(32, 32, 3), (128, 128, 1), (96, 200, 5), (64, 4112, 2)] {
        let w = rnd(din * dout, 1 + din as u64, 0.05);
        let x = rnd(m * din, 2 + dout as u64, 1.0);
        let q = QW::from_rows(&w, din, dout);
        let got = run(&q, &x, m, -1.0, 1);
        // the scalar twin, tile by tile
        let mut twin = vec![0f32; m * dout];
        for r in 0..m {
            for t in 0..dout.div_ceil(TILE) {
                let c0 = t * TILE;
                let n = TILE.min(dout - c0);
                tile_scalar(&x[r * din..(r + 1) * din], None, &q.q[t * din * TILE..(t + 1) * din * TILE], &q.scale[c0..c0 + TILE], &mut twin[r * dout + c0..r * dout + c0 + n]);
            }
        }
        assert_eq!(bits(&got), bits(&twin), "{din}x{dout}: SIMD and scalar differ");
        let want = dense_ref(&x, &q.dequant(), m, din, dout);
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        let err = got.iter().zip(&want).fold(0f32, |a, (g, w)| a.max((g - w).abs()));
        assert!(err <= 1e-5 * scale.max(1.0), "{din}x{dout}: {err} vs dequantized");
        // ava_parts' own int8 matvec (one row)
        let p = ava_parts::quant::quantize_int8(&w, din, dout);
        let pv = p.matvec(&x[..din]);
        let err = got[..dout].iter().zip(&pv).fold(0f32, |a, (g, w)| a.max((g - w).abs()));
        assert!(err <= 1e-5 * scale.max(1.0), "{din}x{dout}: {err} vs ava_parts::quant");
        // and the int8 weights are the reference's
        assert_eq!(q.dequant(), p.dequant(), "{din}x{dout}: the quantized weights");
    }
}

/// E2: an event sum over the non-zero inputs gives the dense int8 sum bit for bit, for 0/1 and graded spikes and the
/// kWTA'd MLP's sparsity; a batch of rows, and any thread count, give each row's own bits.
#[test]
fn event_sums_and_threads_change_no_bit() {
    for (din, dout) in [(128, 128), (384, 384), (512, 128), (1536, 384), (384, 4112)] {
        let w = rnd(din * dout, 7 + din as u64, 0.05);
        let q = QW::from_rows(&w, din, dout);
        for (rate, graded) in [(0.01, false), (0.05, false), (0.08, true), (0.25, false), (0.6, true)] {
            let m = 4;
            let x: Vec<f32> = (0..m).flat_map(|r| sparse(din, rate, graded, 100 + r as u64)).collect();
            let dense = run(&q, &x, m, -1.0, 1);
            let event = run(&q, &x, m, 1.0, 1);
            assert_eq!(bits(&dense), bits(&event), "{din}x{dout} rate {rate}: event vs dense");
            for th in [2, 4, 8] {
                assert_eq!(bits(&run(&q, &x, m, 0.5, th)), bits(&dense), "{din}x{dout} rate {rate}: {th} threads");
            }
            for r in 0..m {
                let one = run(&q, &x[r * din..(r + 1) * din], 1, 0.5, 2);
                assert_eq!(bits(&one), bits(&dense[r * dout..(r + 1) * dout]), "{din}x{dout} rate {rate}: row {r} alone");
            }
            // the f32 event sum (for the timings) is the dense f32 product up to the order of the float sums
            let mut ye = vec![0f32; dout];
            event_f32(&x[..din], &w, dout, &mut ye);
            let want = dense_ref(&x[..din], &w, 1, din, dout);
            let err = ye.iter().zip(&want).fold(0f32, |a, (g, w)| a.max((g - w).abs()));
            assert!(err < 1e-5, "{din}x{dout} rate {rate}: f32 event {err}");
        }
    }
}

/// `mm` and `head` are x.matmul(w) and x.matmul(e^T) exactly with no set active, and the int8 copies inside a scope.
#[test]
fn mm_is_the_plain_matmul_outside_a_scope() {
    let w = Tensor::from_vec(rnd(64 * 96, 3, 0.1), (64, 96), &CPU).unwrap();
    let e = Tensor::from_vec(rnd(130 * 64, 4, 0.1), (130, 64), &CPU).unwrap();
    let x = Tensor::from_vec(rnd(5 * 64, 5, 1.0), (5, 64), &CPU).unwrap();
    assert!(!is_active());
    assert_eq!(max_diff(&mm(&x, &w).unwrap(), &x.matmul(&w).unwrap()), 0.0);
    assert_eq!(max_diff(&head(&x, &e).unwrap(), &x.matmul(&e.t().unwrap()).unwrap()), 0.0);
    let mut set = Q8Set::new(1);
    set.add(&w).unwrap();
    set.set_head(&e).unwrap();
    let set = Arc::new(set);
    {
        let _s = scope(Some(set.clone()));
        let qw = QW::from_tensor(&w).unwrap();
        let want = run(&qw, &x.flatten_all().unwrap().to_vec1().unwrap(), 5, -1.0, 1);
        let got: Vec<f32> = mm(&x, &w).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(bits(&got), bits(&want));
        let d = max_diff(&head(&x, &e).unwrap(), &x.matmul(&e.t().unwrap()).unwrap());
        assert!(d > 0.0 && d < 0.05, "head {d}");
        {
            // a nested scope of a model with int8 off is plain f32, and the outer set comes back after it
            let _n = scope(None);
            assert_eq!(max_diff(&mm(&x, &w).unwrap(), &x.matmul(&w).unwrap()), 0.0);
        }
        assert!(is_active());
        // a weight the set does not hold stays f32
        let w2 = w.affine(1.0, 0.0).unwrap();
        assert_eq!(max_diff(&mm(&x, &w2).unwrap(), &x.matmul(&w2).unwrap()), 0.0);
    }
    assert!(!is_active());
}

/// Every output of a model with int8 switched on and off again is the f32 output bit for bit (forward, step, the
/// trace twins and the carried state), and a second model with int8 off is untouched while the first is on.
#[test]
fn int8_off_is_f32_bit_for_bit() {
    let (_, m) = build_awake(v40_cfg(), 5);
    let (_, other) = build_awake(v40_cfg(), 6);
    let vocab = ava_data::numbers::VOCAB_GLANCE;
    let (b, t) = (2, 24);
    let toks = tokens(b, t, vocab, 13);
    let outs = |m: &Model| -> Vec<Tensor> {
        let mut v = vec![one_pass(m, &toks, b, t)];
        let mut s = m.stream().unwrap();
        for &tok in &toks[..t] {
            v.push(m.step(&mut s, tok).unwrap());
        }
        v.extend(s.state_tensors());
        let mut s = m.stream().unwrap();
        for &tok in &toks[..t] {
            v.push(m.step_trace(&mut s, tok).unwrap().0);
        }
        v.push(m.forward_trace(&Tensor::from_vec(toks.clone(), (b, t), &CPU).unwrap()).unwrap().0);
        let mut st = m.state(b).unwrap();
        v.push(in_pieces(m, &mut st, &toks, b, t, &[10, 1, 13]));
        v
    };
    let (before, other_before) = (outs(&m), outs(&other));
    m.set_int8(true).unwrap();
    assert!(m.int8());
    let on = outs(&m);
    assert!(before.iter().zip(&on).any(|(a, b)| max_diff(a, b) > 0.0), "int8 changed nothing");
    for (a, b) in other_before.iter().zip(&outs(&other)) {
        assert_eq!(max_diff(a, b), 0.0, "the other model (int8 off)");
    }
    m.set_int8(false).unwrap();
    assert!(!m.int8());
    for (i, (a, b)) in before.iter().zip(&outs(&m)).enumerate() {
        assert_eq!(a.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "output {i}");
    }
}

/// Everything on, from the block recompute test (src/wave5/tests.rs): every Wave 5 part in the block plus the v42 block
/// options (channel gates, momentum, int2 spikes, ALIF, dendrites, STP, delays, working memory, dense residual ...).
fn every() -> String {
    "--resonate 16 --modulate --tem 4 --halting 3 --cerebellum 64 --workspace 4 --holo 8 --sync 4 --jepa 0.5 --diffusion 0.5 --critic 0.5 --confidence 0.5 --topo 0.5 --schema 0.5 \
     --dendrites 4 --stp --delays 4 --wm 8 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1"
        .to_string()
}

fn cfg_with(flags: &str) -> Config {
    Config::with_kind(Kind::D, 258, 32, 2, 64).with_arch(Arch::from_flags(flags).unwrap()).unwrap()
}

fn rel_rms(a: &Tensor, b: &Tensor) -> f32 {
    let num = a.sub(b).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
    let den = b.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
    (num / den).sqrt()
}

/// Int8 read token by token (`step`, `forward_with_state` pieces, `feed`) equals
/// int8 read in one pass within 1e-4 (5e-4 with --tem, as in f32).
#[test]
fn int8_streams_as_it_batches() {
    let models: Vec<(String, Model, usize)> = vec![
        ("everything on".into(), build_awake(cfg_with(&every()), 14).1, 258),
        ("everything on, no tem".into(), build_awake(cfg_with(&every().replace("--tem 4 ", "")), 15).1, 258),
        ("D+glance+mag+engram".into(), build_awake(v40_cfg(), 5).1, ava_data::numbers::VOCAB_GLANCE),
        ("S".into(), crate::testkit::tiny(Kind::S, 3).2, 258),
    ];
    for (name, m, vocab) in models {
        let (b, t) = (2, 64);
        let toks = tokens(b, t, vocab, 71);
        let f32_pass = one_pass(&m, &toks, b, t);
        m.set_int8(true).unwrap();
        let (nq, bytes) = m.int8_size();
        assert!(nq > 0 && bytes > 0, "{name}");
        let want = one_pass(&m, &toks, b, t);
        eprintln!("{name}: int8 vs f32 relative RMS {:.4}, max |dlogit| {:.4}", rel_rms(&want, &f32_pass), max_diff(&want, &f32_pass));
        let tol = if name.contains("everything on") && !name.contains("no tem") { 5e-4 } else { 1e-4 };
        // a fresh state in one piece is the batched pass exactly (the same rows through the same kernel)
        if m.cfg.kind != Kind::S {
            let mut st = m.state(b).unwrap();
            assert_eq!(max_diff(&in_pieces(&m, &mut st, &toks, b, t, &[t]), &want), 0.0, "{name} fresh state");
            for pieces in [vec![20, 44], vec![1; 64]] {
                let mut st = m.state(b).unwrap();
                let d = max_diff(&in_pieces(&m, &mut st, &toks, b, t, &pieces), &want);
                eprintln!("{name} pieces {}: {d:.2e}", pieces.len());
                assert!(d < tol, "{name} {pieces:?}: {d}");
            }
        }
        // the talking stream, token by token, and a batched feed
        let mut s = m.stream().unwrap();
        let mut worst = 0f32;
        for (i, &tok) in toks[..t].iter().enumerate() {
            let l = m.step(&mut s, tok).unwrap();
            worst = worst.max(max_diff(&l, &want.get(0).unwrap().get(i).unwrap()));
        }
        eprintln!("{name} step: {worst:.2e}");
        assert!(worst < tol, "{name} step: {worst}");
        let mut s = m.stream().unwrap();
        let fed = m.feed(&mut s, &toks[..t]).unwrap();
        let d = max_diff(&fed, &want.get(0).unwrap().get(t - 1).unwrap());
        assert!(d < tol, "{name} feed: {d}");
        m.set_int8(false).unwrap();
    }
}

/// set_int8 refuses a model that is not on the CPU only by device; on the CPU it holds every listed matrix.
#[test]
fn the_set_holds_the_big_matrices() {
    let (_, m) = build_awake(v40_cfg(), 5);
    m.set_int8_threads(true, 2).unwrap();
    let (n, _) = m.int8_size();
    // per block: win, wout, q k v o g of the memory, w1, w2 (9); the engram block +4; the head
    assert_eq!(n, 2 * 9 + 4 + 1);
    let (_, e) = build_awake(cfg_with(&every()), 14);
    e.set_int8(true).unwrap();
    // + the channel gates' erase keys, per block
    assert_eq!(e.int8_size().0, 2 * 10 + 1);
}

/// The int8 path against f32 on a tiny everything-on model, over 8 seeds (each its own weights and text). The int8
/// error of each matmul is ~0.5% (`the_kernel_is_the_reference_int8_product`); tiny random spiking models are chaotic,
/// so any weight error of that size flips a spike or a kWTA winner now and then and the carried membranes then diverge.
/// The pre-set 3% bound over all 64 positions failed for that reason (0.2-0.4 relative RMS), and f32 with the same
/// error flipped in sign (2w - deq, a control of the same size) diverges as much. So the bounds here: the first
/// token (nothing carried yet) within 1% relative RMS in the median over seeds, and the later tokens no further from
/// f32 than 1.5x the control's median (int8 adds nothing beyond its own weight error).
#[test]
fn int8_is_as_near_f32_as_its_own_weight_error() {
    let (mut first, mut rest, mut ctl) = (Vec::new(), Vec::new(), Vec::new());
    for seed in 10..18u64 {
        let (vm, m) = build_awake(cfg_with(&every()), seed);
        let (b, t) = (2, 64);
        let toks = tokens(b, t, 258, 71 + seed);
        let f32p = one_pass(&m, &toks, b, t);
        m.set_int8(true).unwrap();
        let q = one_pass(&m, &toks, b, t);
        m.set_int8(false).unwrap();
        // the control: every matrix the int8 set holds moved by its int8 error with the sign flipped
        let orig: Vec<(String, Tensor)> = vm.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.as_tensor().copy().unwrap())).collect();
        {
            let data = vm.data().lock().unwrap();
            for (k, w) in &orig {
                let blockw = [".win", ".wout", ".wq", ".wk", ".wv", ".wo", ".wg", ".we", ".w1", ".w2"].iter().any(|s| k.ends_with(s))
                    && k.starts_with('b')
                    && !k.contains(".wm.")
                    && !k.contains(".ws.")
                    && w.rank() == 2;
                let deq = if k == "tok_emb" {
                    let qw = QW::from_tensor_t(w).unwrap();
                    Tensor::from_vec(qw.dequant(), (w.dim(1).unwrap(), w.dim(0).unwrap()), &CPU).unwrap().t().unwrap().contiguous().unwrap()
                } else if blockw {
                    Tensor::from_vec(QW::from_tensor(w).unwrap().dequant(), w.shape(), &CPU).unwrap()
                } else {
                    continue;
                };
                data[k].set(&w.affine(2.0, 0.0).unwrap().sub(&deq).unwrap()).unwrap();
            }
        }
        let c = one_pass(&m, &toks, b, t);
        let pos = |a: &Tensor, z: &Tensor, p0: usize, n: usize| rel_rms(&a.narrow(1, p0, n).unwrap(), &z.narrow(1, p0, n).unwrap());
        first.push(pos(&q, &f32p, 0, 1));
        rest.push(pos(&q, &f32p, 1, t - 1));
        ctl.push(pos(&c, &f32p, 1, t - 1));
    }
    let median = |v: &mut Vec<f32>| {
        v.sort_by(f32::total_cmp);
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    };
    let (f, r, c) = (median(&mut first), median(&mut rest), median(&mut ctl));
    eprintln!("everything on, 8 seeds: int8 first token {f:.4}, later tokens {r:.4}; control later tokens {c:.4}");
    assert!(f < 0.01, "first token: {f}");
    assert!(r < 1.5 * c, "later tokens: int8 {r}, control {c}");
}

/// Row blocking (src/int8/block.rs) gives the per-row kernel's bits: dense, sparse 0/1, graded sparse and all-zero rows
/// mixed in one batch (dense rows with exact zeros and -0 in them), row counts 1, 5, 6, 7, 13 and past one chunk, full
/// and partial last tiles, 1 to 8 threads; the blocked kernel is also forced where `matmul_rows` would not pick it.
#[test]
fn row_blocking_changes_no_bit() {
    if !simd_ok() {
        eprintln!("no AVX2 + FMA here: the blocked kernel never runs");
        return;
    }
    for (din, dout) in [(32, 32), (128, 128), (96, 200), (128, 512), (512, 128), (384, 1536), (1536, 384), (128, 4112)] {
        let q = QW::from_rows(&rnd(din * dout, 31 + din as u64 + dout as u64, 0.05), din, dout);
        for m in [1usize, 5, 6, 7, 13, 50, 101] {
            for mix in 0..3 {
                // mix 0: every kind of row; 1: all dense; 2: all sparse
                let x: Vec<f32> = (0..m)
                    .flat_map(|r| {
                        let kind = match mix {
                            0 => r % 4,
                            1 => 0,
                            _ => 1 + r % 2,
                        };
                        let seed = 500 + 7 * r as u64 + din as u64;
                        match kind {
                            0 => {
                                let mut v = rnd(din, seed, 1.0);
                                v[0] = 0.0;
                                v[din / 2] = -0.0;
                                v
                            }
                            1 => sparse(din, 0.05, false, seed),
                            2 => sparse(din, 0.2, true, seed),
                            _ => vec![0.0; din],
                        }
                    })
                    .collect();
                let mut want = vec![0f32; m * dout];
                q.matmul_rows_with(&x, m, &mut want, 0.5, 1, false);
                for th in [1, 3, 8] {
                    let mut got = vec![0f32; m * dout];
                    q.matmul_rows_with(&x, m, &mut got, 0.5, th, true);
                    assert_eq!(bits(&got), bits(&want), "{din}x{dout} m {m} mix {mix} {th} threads");
                    // forced: every product through the blocked kernel, sparse rows as events (share 1), through the
                    // 6-row kernel (share 0) or split at the default share
                    for share in [0.0, 1.0, block::BLOCK_EVENT_MAX, 0.5] {
                        let lists = q.event_lists(&x, m, share, th);
                        let mut got = vec![0f32; m * dout];
                        q.matmul_blocked(&x, m, &mut got, &lists, th);
                        assert_eq!(bits(&got), bits(&want), "{din}x{dout} m {m} mix {mix} {th} threads, forced, share {share}");
                    }
                    // E2 off: every row dense
                    let mut dense = vec![0f32; m * dout];
                    q.matmul_rows_with(&x, m, &mut dense, -1.0, 1, false);
                    let mut got = vec![0f32; m * dout];
                    q.matmul_rows_with(&x, m, &mut got, -1.0, th, true);
                    assert_eq!(bits(&got), bits(&dense), "{din}x{dout} m {m} mix {mix} {th} threads, E2 off");
                }
            }
        }
    }
}

/// The SIMD scan of the event lists (nnz, nonzeros) is `v != 0.0` exactly: -0 skipped, NaN and inf kept, every tail
/// length, and the lists are the same on 1 or 8 threads.
#[test]
fn event_lists_are_the_nonzero_inputs() {
    for n in [1usize, 7, 8, 9, 16, 33, 128, 517] {
        let mut row = sparse(n, 0.3, true, 900 + n as u64);
        for (j, v) in row.iter_mut().enumerate() {
            match j % 11 {
                3 => *v = -0.0,
                5 => *v = f32::NAN,
                7 => *v = f32::NEG_INFINITY,
                9 => *v = 1e-40,
                _ => {}
            }
        }
        let want: Vec<u32> = row.iter().enumerate().filter(|(_, &v)| v != 0.0).map(|(j, _)| j as u32).collect();
        assert_eq!(nnz(&row), want.len(), "n {n}");
        let mut got = vec![0u32; want.len()];
        nonzeros(&row, &mut got);
        assert_eq!(got, want, "n {n}");
    }
    let (din, m) = (96, 300);
    let q = QW::from_rows(&rnd(din * 64, 3, 0.05), din, 64);
    let x: Vec<f32> = (0..m).flat_map(|r| sparse(din, (r % 10) as f32 / 10.0, r % 2 == 0, 40 + r as u64)).collect();
    for share in [-1.0, 0.0, 0.25, 1.0] {
        let one = q.event_lists(&x, m, share, 1);
        assert_eq!(one, q.event_lists(&x, m, share, 8), "share {share}");
        for r in 0..m {
            let row = &x[r * din..(r + 1) * din];
            let nz: Vec<u32> = row.iter().enumerate().filter(|(_, &v)| v != 0.0).map(|(j, _)| j as u32).collect();
            let want = ((nz.len() as f64) <= share * din as f64).then_some(nz);
            assert_eq!(one.get(r), want.as_deref(), "share {share} row {r}");
        }
    }
}

/// Side by side (`QW::concat`): each part's columns of the merged product are that part's own product bit for bit,
/// dense and sparse rows, 1 and 8 threads, one row and a batch; parts with a partial tile do not concatenate.
#[test]
fn concatenated_matrices_give_each_part_its_bits() {
    let (din, outs) = (192, [128usize, 64, 256]);
    let qs: Vec<QW> = outs.iter().enumerate().map(|(k, &o)| QW::from_rows(&rnd(din * o, 70 + k as u64, 0.3), din, o)).collect();
    let refs: Vec<&QW> = qs.iter().collect();
    let all = QW::concat(&refs).expect("whole tiles concatenate");
    assert_eq!((all.din, all.dout), (din, outs.iter().sum()));
    for m in [1usize, 7] {
        let mut x = rnd(m * din, 80, 1.0);
        x[..din].copy_from_slice(&sparse(din, 0.05, false, 81));
        for th in [1usize, 8] {
            let got = run(&all, &x, m, EVENT_MAX, th);
            let mut c0 = 0;
            for (q, &o) in qs.iter().zip(&outs) {
                let want = run(q, &x, m, EVENT_MAX, th);
                for r in 0..m {
                    let g = &got[r * all.dout + c0..r * all.dout + c0 + o];
                    assert_eq!(bits(g), bits(&want[r * o..(r + 1) * o]), "rows {m}, threads {th}, part at column {c0}, row {r}");
                }
                c0 += o;
            }
        }
    }
    let odd = QW::from_rows(&rnd(din * 96, 90, 0.3), din, 96);
    assert!(QW::concat(&[&qs[0], &odd]).is_none(), "a partial tile");
    let other = QW::from_rows(&rnd(64 * 64, 91, 0.3), 64, 64);
    assert!(QW::concat(&[&qs[0], &other]).is_none(), "another din");
}
