// TESTING / DESIGN ONLY — ava_brain tests: shapes, loss at init, every parameter gets a gradient,
// exact causality, bits-per-byte on a hand-computed case, evaluation bookkeeping, determinism of
// training steps (on the real train shards), checkpoint round trip.
// Run: CARGO_BUILD_JOBS=4 cargo test --release -- --test-threads=4

use ava_brain::eval::{Activity, BpbAcc};
use ava_brain::*;
use ava_data::bpe::{Bpe, EOS, PAD};
use ava_data::shards::{Sampler, ShardSet};
use ava_grad::{Graph, Rng, Tensor, IGNORE};

const VOCAB: usize = 4096;
const BPE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/nn/bpe-4096.txt");
const TRAIN_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/nn/shards-4096/train");

const KINDS: [Kind; 3] = [Kind::T, Kind::S, Kind::H];

fn rand_ids(n: usize, vocab: usize, rng: &mut Rng) -> Vec<u32> {
    (0..n).map(|_| rng.below(vocab) as u32).collect()
}

fn cfg(kind: Kind, d: usize, layers: usize, ctx: usize) -> Config {
    Config::new(kind, VOCAB, d, layers, ctx)
}

#[test]
fn forward_shapes() {
    for kind in KINDS {
        let c = cfg(kind, 64, 2, 16);
        let (m, p) = Model::new(c, 1);
        let (b, t) = (3, 16);
        let ids = rand_ids(b * t, VOCAB, &mut Rng::new(7));
        let mut g = Graph::new(&p);
        let f = m.forward(&mut g, &ids, b, t);
        assert_eq!(g.shape(f.logits), &[b * t, VOCAB], "{kind:?} logits");
        let n_mem = if kind.has_spike_mem() { 2 } else { 0 };
        assert_eq!(f.spikes.len(), n_mem, "{kind:?} spike layers");
        for &s in &f.spikes {
            assert_eq!(g.shape(s), &[b, t, 64]);
            assert!(g.value(s).data.iter().all(|&v| v == 0.0 || v == 1.0), "spikes must be 0/1");
        }
        let n_kw = if kind.has_sparse_mlp() { 2 } else { 0 };
        assert_eq!(f.kwta_pre.len(), n_kw);
        assert_eq!(f.kwta_post.len(), n_kw);
        for (&a, &r) in f.kwta_pre.iter().zip(&f.kwta_post) {
            assert_eq!(g.shape(a), &[b * t, 256]);
            assert_eq!(g.shape(r), &[b * t, 256]);
            // exactly k = 4d/8 = 32 winners per row before relu (values are continuous, no exact zeros)
            for row in g.value(a).data.chunks(256) {
                assert_eq!(row.iter().filter(|&&v| v != 0.0).count(), 32);
            }
        }
        // shorter sequences than ctx work too
        let mut g = Graph::new(&p);
        let f = m.forward(&mut g, &ids[..2 * 5], 2, 5);
        assert_eq!(g.shape(f.logits), &[10, VOCAB]);
    }
}

#[test]
fn weight_counts_match_formula() {
    let (v, d, l, ctx) = (VOCAB, 64, 2, 128);
    for kind in KINDS {
        let (_, p) = Model::new(cfg(kind, d, l, ctx), 1);
        let (total, nonembed) = Model::weight_counts(&p);
        let attn = 4 * d * d + d;
        let mem = 2 * d * d + 2 * d;
        let mlp = 2 * d * 4 * d + d;
        let per_block = match kind {
            Kind::T => attn + mlp,
            Kind::S => mem + mlp,
            Kind::H => mem + attn + mlp,
        };
        let want_nonembed = l * per_block + d;
        let pos = if kind.has_pos() { ctx * d } else { 0 };
        assert_eq!(nonembed, want_nonembed, "{kind:?}");
        assert_eq!(total, want_nonembed + v * d + pos, "{kind:?}");
    }
}

#[test]
fn loss_at_init_is_ln_vocab() {
    let want = (VOCAB as f32).ln();
    for kind in KINDS {
        for &(d, l) in &[(64usize, 2usize), (128, 4)] {
            let (m, p) = Model::new(cfg(kind, d, l, 32), 3);
            let mut rng = Rng::new(11);
            let (b, t) = (4, 32);
            let ids = rand_ids(b * t, VOCAB, &mut rng);
            let tg = rand_ids(b * t, VOCAB, &mut rng);
            let mut g = Graph::new(&p);
            let (lv, _) = m.loss(&mut g, &ids, &tg, b, t);
            let loss = g.scalar(lv);
            println!("{kind:?} d={d} L={l}: init loss {loss:.4} (ln V = {want:.4})");
            assert!((loss - want).abs() < 0.5, "{kind:?} d={d} L={l}: init loss {loss} vs ln V {want}");
        }
    }
}

#[test]
fn every_parameter_gets_a_gradient() {
    for kind in KINDS {
        let ctx = 16;
        let (m, mut p) = Model::new(cfg(kind, 64, 2, ctx), 5);
        let mut rng = Rng::new(13);
        let (b, t) = (2, ctx); // t = ctx so every position-table row is used
        let ids = rand_ids(b * t, VOCAB, &mut rng);
        let tg = rand_ids(b * t, VOCAB, &mut rng);
        let grads = {
            let mut g = Graph::new(&p);
            let (lv, f) = m.loss(&mut g, &ids, &tg, b, t);
            for (i, &s) in f.spikes.iter().enumerate() {
                println!("{kind:?} layer {i}: init spike rate {:.4}", ava_grad::fraction_active(g.value(s)));
            }
            g.backward(lv)
        };
        p.accumulate(&grads);
        let mut bad = Vec::new();
        for id in p.ids() {
            let gd = &p.grad(id).data;
            let finite = gd.iter().all(|v| v.is_finite());
            let nonzero = gd.iter().any(|&v| v != 0.0);
            if !finite || !nonzero {
                bad.push(format!("{} (finite {finite}, nonzero {nonzero})", p.name(id)));
            }
        }
        assert!(bad.is_empty(), "{kind:?}: parameters with a bad gradient: {bad:?}");
    }
}

#[test]
fn causality_is_exact() {
    for kind in KINDS {
        let ctx = 24;
        let (m, p) = Model::new(cfg(kind, 64, 2, ctx), 9);
        let (b, t) = (2, ctx);
        let mut rng = Rng::new(21);
        let base = rand_ids(b * t, VOCAB, &mut rng);
        let logits_of = |ids: &[u32]| {
            let mut g = Graph::new(&p);
            let f = m.forward(&mut g, ids, b, t);
            g.value(f.logits).data.clone()
        };
        let l0 = logits_of(&base);
        for &cut in &[0usize, 1, 7, 15, 22] {
            // change every token at positions > cut (in both sequences)
            let mut alt = base.clone();
            for s in 0..b {
                for pos in cut + 1..t {
                    let i = s * t + pos;
                    alt[i] = ((alt[i] as usize + 1 + rng.below(VOCAB - 1)) % VOCAB) as u32;
                    assert_ne!(alt[i], base[i]);
                }
            }
            let l1 = logits_of(&alt);
            let mut changed_after = false;
            for s in 0..b {
                for pos in 0..t {
                    let r = (s * t + pos) * VOCAB;
                    let (a, c) = (&l0[r..r + VOCAB], &l1[r..r + VOCAB]);
                    let same = a.iter().zip(c).all(|(x, y)| x.to_bits() == y.to_bits());
                    if pos <= cut {
                        assert!(same, "{kind:?}: logits at position {pos} changed when tokens after {cut} changed");
                    } else if !same {
                        changed_after = true;
                    }
                }
            }
            assert!(changed_after, "{kind:?}: changing tokens after {cut} changed nothing (test is vacuous)");
        }
    }
}

#[test]
fn bpb_hand_computed() {
    // vocab 260: ids 0..255 bytes, 256 = EOS, 257 = PAD, 258 a 3-byte merge, 259 a 2-byte merge.
    let v = 260usize;
    assert_eq!((EOS, PAD), (256, 257));
    let mut byte_len = vec![1u32; v];
    byte_len[EOS as usize] = 1;
    byte_len[PAD as usize] = 0;
    byte_len[258] = 3;
    byte_len[259] = 2;
    let ln = |x: f64| x.ln();
    let mut logits = vec![0.0f32; 4 * v];
    // row 0: uniform, target 65 ('A', 1 byte)               → nll = ln 260
    // row 1: z[EOS] = ln 259, others 0, target EOS (1 byte)   → Σ = 518, nll = ln 518 − ln 259 = ln 2
    logits[v + EOS as usize] = ln(259.0) as f32;
    // row 2: target PAD                                      → skipped entirely
    logits[2 * v + 7] = 5.0;
    // row 3: z[258] = ln 777, others 0, target 258 (3 bytes)  → Σ = 1036, nll = ln 1036 − ln 777 = ln(4/3)
    logits[3 * v + 258] = ln(777.0) as f32;
    let targets = [65u32, EOS, PAD, 258];
    let mut acc = BpbAcc::default();
    acc.add_rows(&logits, v, &targets, &byte_len);
    let want_nats = ln(260.0) + ln(2.0) + ln(4.0 / 3.0);
    assert_eq!(acc.tokens, 3);
    assert_eq!(acc.bytes, 5);
    assert!((acc.nats - want_nats).abs() < 1e-5, "nats {} vs {}", acc.nats, want_nats);
    let want_bpb = want_nats / std::f64::consts::LN_2 / 5.0;
    assert!((acc.bpb() - want_bpb).abs() < 1e-6, "bpb {} vs {}", acc.bpb(), want_bpb);
    assert!((acc.nats_per_token() - want_nats / 3.0).abs() < 1e-6);
    // IGNORE behaves like PAD
    let mut acc2 = BpbAcc::default();
    acc2.add_rows(&logits, v, &[65, EOS, IGNORE, 258], &byte_len);
    assert_eq!(acc2, acc);
    // agrees with Graph::cross_entropy (a mean over non-IGNORE rows) × counted rows
    let p = ava_grad::Params::new();
    let mut g = Graph::new(&p);
    let z = g.input(Tensor::from_vec(&[4, v], logits.clone()));
    let ce = g.cross_entropy(z, &[65, EOS, IGNORE, 258]);
    assert!((g.scalar(ce) as f64 * 3.0 - want_nats).abs() < 1e-4);
}

#[test]
fn window_io_pads_and_limits() {
    let w = vec![10u32, 11, 12, EOS, 14, PAD, PAD];
    let mut rem = 100u64;
    let (i, t) = window_io(&w, &mut rem);
    assert_eq!(i, vec![10, 11, 12, EOS, 14, PAD]);
    assert_eq!(t, vec![11, 12, EOS, 14, IGNORE, IGNORE]);
    assert_eq!(rem, 96);
    let mut rem = 2u64;
    let (_, t) = window_io(&w, &mut rem);
    assert_eq!(t, vec![11, 12, IGNORE, IGNORE, IGNORE, IGNORE]);
    assert_eq!(rem, 0);
}

#[test]
fn byte_lengths_from_real_bpe() {
    let bpe = Bpe::load(BPE_PATH).expect("load bpe-4096");
    let bl = byte_lengths(&bpe);
    assert_eq!(bl.len(), VOCAB);
    assert_eq!(bl[EOS as usize], 1);
    assert_eq!(bl[PAD as usize], 0);
    assert!(bl[..256].iter().all(|&x| x == 1));
    for id in 258..VOCAB as u32 {
        assert_eq!(bl[id as usize] as usize, bpe.decode_bytes(&[id]).len());
        assert!(bl[id as usize] >= 2);
    }
}

#[test]
fn evaluate_counts_exactly_and_is_batch_invariant() {
    // synthetic windows of ctx + 1 tokens, the last one PAD-padded like iter_sequential
    let ctx = 16;
    let mut rng = Rng::new(5);
    let stream = rand_ids(5 * ctx + 6, 256, &mut rng); // byte ids only: no EOS/PAD inside the stream
    let mut windows = Vec::new();
    let mut off = 0;
    while off + ctx < stream.len() {
        windows.push(stream[off..off + ctx + 1].to_vec());
        off += ctx;
    }
    let mut last = stream[off..].to_vec();
    last.resize(ctx + 1, PAD);
    windows.push(last); // 6 tokens → 5 real targets, the rest PAD
    let real_targets = (stream.len() - 1) as u64; // every token after the first is a target once
    let byte_len = vec![1u32; VOCAB];
    for kind in KINDS {
        let (m, p) = Model::new(cfg(kind, 64, 2, ctx), 2);
        let r1 = evaluate(&m, &p, windows.clone(), 1, u64::MAX, &byte_len, 4);
        let r3 = evaluate(&m, &p, windows.clone(), 3, u64::MAX, &byte_len, 4);
        assert_eq!(r1.acc.tokens, real_targets, "{kind:?}: PAD targets must be skipped");
        assert_eq!(r1.acc, r3.acc, "{kind:?}: batch size changed the result");
        assert_eq!(r1.spike_rates, r3.spike_rates);
        let r = evaluate(&m, &p, windows.clone(), 2, 37, &byte_len, 4);
        assert_eq!(r.acc.tokens, 37);
        assert_eq!(r.windows, 3); // 16 + 16 + 5
        if kind.has_sparse_mlp() {
            for &f in &r1.kwta_pre {
                assert!((f - 0.125).abs() < 1e-9, "{kind:?}: kWTA pre-relu fraction {f}");
            }
            assert_eq!(r1.kwta_post.len(), 2);
        } else {
            assert!(r1.kwta_pre.is_empty() && r1.spike_rates.is_empty());
        }
    }
    // Activity counts only masked rows
    let mut a = Activity::new(1);
    a.add(0, &Tensor::from_vec(&[2, 2], vec![1.0, 0.0, 1.0, 1.0]), &[true, false]);
    assert_eq!(a.fractions(), vec![0.5]);
}

fn hyper(total_steps: usize) -> Hyper {
    Hyper { lr: 3e-3, warmup: 2, min_lr_frac: 0.1, wd: 0.1, clip: 1.0, total_steps, threads: 4 }
}

fn three_step_losses(kind: Kind, seed: u64, data_seed: u64, set: &ShardSet) -> Vec<f32> {
    three_step_losses_threads(kind, seed, data_seed, set, 4)
}

fn three_step_losses_threads(kind: Kind, seed: u64, data_seed: u64, set: &ShardSet, threads: usize) -> Vec<f32> {
    let ctx = 32;
    let batch = 4;
    let mut tr = Trainer::new(cfg(kind, 32, 2, ctx), seed, Hyper { threads, ..hyper(3) });
    let mut sampler = Sampler::new(set, ctx, data_seed).expect("sampler");
    (0..3)
        .map(|_| {
            let w = sampler.batch(batch);
            let (ids, tg) = train_io(&w);
            let s = tr.train_step(&ids, &tg, batch, ctx);
            assert!(s.loss.is_finite() && s.grad_norm.is_finite());
            s.loss
        })
        .collect()
}

#[test]
fn training_steps_are_deterministic() {
    let set = ShardSet::open(TRAIN_DIR).expect("open train shards");
    for kind in KINDS {
        let a = three_step_losses(kind, 1, 1, &set);
        let b = three_step_losses(kind, 1, 1, &set);
        println!("{kind:?}: {a:?}");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a), bits(&b), "{kind:?}: same seed, different losses");
        // ava_grad's kernels are bit-identical for any thread count, so 1 thread gives the same run
        let one = three_step_losses_threads(kind, 1, 1, &set, 1);
        assert_eq!(bits(&a), bits(&one), "{kind:?}: 1 thread vs 4 threads differ");
        let c = three_step_losses(kind, 2, 1, &set);
        assert_ne!(bits(&a), bits(&c), "{kind:?}: a different seed should change the losses");
    }
}

#[test]
fn checkpoint_roundtrip_gives_identical_logits() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ava_brain_ckpt_test");
    std::fs::create_dir_all(&dir).unwrap();
    for kind in KINDS {
        let ctx = 16;
        let c = cfg(kind, 32, 2, ctx);
        let mut tr = Trainer::new(c, 4, hyper(10));
        let mut rng = Rng::new(17);
        let (b, t) = (2, ctx);
        // move the weights away from their init first
        for _ in 0..2 {
            let ids = rand_ids(b * t, VOCAB, &mut rng);
            let tg = rand_ids(b * t, VOCAB, &mut rng);
            tr.train_step(&ids, &tg, b, t);
        }
        let path = dir.join(format!("model_{}.awts", kind.name()));
        tr.params.save(&path).unwrap();
        let ids = rand_ids(b * t, VOCAB, &mut rng);
        let before = {
            let mut g = Graph::new(&tr.params);
            let f = tr.model.forward(&mut g, &ids, b, t);
            g.value(f.logits).data.clone()
        };
        let (m2, mut p2) = Model::new(c, 999); // different init
        p2.load(&path).unwrap();
        let after = {
            let mut g = Graph::new(&p2);
            let f = m2.forward(&mut g, &ids, b, t);
            g.value(f.logits).data.clone()
        };
        assert!(before.iter().zip(&after).all(|(x, y)| x.to_bits() == y.to_bits()), "{kind:?}: logits differ after reload");
        std::fs::remove_file(&path).ok();
    }
}
