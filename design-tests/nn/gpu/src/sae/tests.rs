// TESTING / DESIGN ONLY
// Tests of src/sae.rs (A13): the TopK SAE on a planted dictionary, TopK, unit atoms, dead latents and AuxK, the
// recording hook (off = on = off, shapes, xf, --feedback's chunks, a lean-loaded model), the probes, the text labels,
// and both tools end to end on a tiny saved model. CPU only.

use super::probe::{fit_and_score, text_labels, ProbeCfg};
use super::run::{probe_states, refuse_sealed, sae_train};
use super::*;
use crate::arch::Arch;
use crate::model::{norm, Config, Kind};
use crate::state::tests::{build_awake, max_diff, tokens};
use crate::testkit::{byte_bpe, scratch};
use ava_data::bpe::EOS;
use ava_data::shards::ShardWriter;
use candle_core::{Device, Tensor};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const CPU: Device = Device::Cpu;

fn unit_rows(n: usize, d: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n * d).map(|_| rng.normal() as f32).collect();
    for r in v.chunks_exact_mut(d) {
        let s = dot(r, r).sqrt();
        r.iter_mut().for_each(|x| *x /= s);
    }
    v
}

/// n samples, each `per` distinct atoms of `atoms` [na, d] with weights in [1, 2).
fn planted(atoms: &[f32], d: usize, n: usize, per: usize, rng: &mut Rng) -> Vec<f32> {
    let na = atoms.len() / d;
    let mut x = vec![0f32; n * d];
    for r in x.chunks_exact_mut(d) {
        let mut pick: Vec<usize> = Vec::with_capacity(per);
        while pick.len() < per {
            let j = rng.below(na);
            if !pick.contains(&j) {
                pick.push(j);
            }
        }
        for j in pick {
            axpy(r, 1.0 + rng.uniform() as f32, &atoms[j * d..(j + 1) * d]);
        }
    }
    x
}

#[test]
fn the_sae_recovers_a_planted_dictionary() {
    let (d, na, per, m) = (32, 48, 3, 96);
    let mut rng = Rng::new(7);
    let atoms = unit_rows(na, d, &mut rng);
    let x = planted(&atoms, d, 16384, per, &mut rng);
    let xt = planted(&atoms, d, 2048, per, &mut rng);
    let mut cfg = TrainCfg::new(d, m, per, 3000);
    cfg.batch = 256;
    cfg.lr = 3e-3;
    cfg.threads = 2;
    cfg.dead_after = 8192;
    let (sae, log) = train(&x, d, &cfg, |_| {}).unwrap();
    let (fvu, codes) = sae.fvu(&xt, 2).unwrap();
    // every true atom: its best cosine with a decoder atom
    let mut matched = 0;
    let mut worst = 1f32;
    for a in atoms.chunks_exact(d) {
        let best = sae.w_dec.chunks_exact(d).map(|w| dot(a, w)).fold(f32::MIN, f32::max);
        worst = worst.min(best);
        matched += (best > 0.9) as usize;
    }
    eprintln!("planted: FVU {fvu:.4}, matched {matched}/{na} (worst cosine {worst:.3}), L0 {:.2}, dead {} ({:.1} s)", codes.mean_l0(), log.dead_end, log.seconds);
    assert!(fvu < 0.1, "FVU {fvu}");
    assert!(matched >= 40, "only {matched} of {na} atoms recovered");
    // a saved SAE loads back the same
    let dir = scratch("sae_planted");
    let p = dir.join("sae.safetensors");
    sae.save(&p).unwrap();
    let back = Sae::load(&p).unwrap();
    assert_eq!((back.d, back.m, back.k, back.scale), (sae.d, sae.m, sae.k, sae.scale));
    assert_eq!(back.w_dec, sae.w_dec);
    assert_eq!(back.encode(&xt, 1).unwrap().idx, codes.idx);
}

#[test]
fn topk_keeps_exactly_k_and_the_largest() {
    let (d, m, k) = (16, 40, 5);
    let mut rng = Rng::new(3);
    let x: Vec<f32> = (0..300 * d).map(|_| rng.normal() as f32).collect();
    let mut sae = Sae::init(&x, d, m, k, 9).unwrap();
    // every pre-activation positive: then exactly k fire on every row
    sae.b_enc.iter_mut().for_each(|b| *b = 50.0);
    let c = sae.encode(&x, 3).unwrap();
    assert_eq!(c.len(), 300);
    assert_eq!(c.mean_l0(), k as f64);
    let pre = sae.pre(&sae.centre(&x)).unwrap();
    for (r, ii) in c.idx.chunks_exact(k).enumerate() {
        let mut s: Vec<u32> = ii.to_vec();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), k, "row {r}: k distinct latents");
        let p = &pre[r * m..(r + 1) * m];
        let lo = ii.iter().map(|&j| p[j as usize]).fold(f32::MAX, f32::min);
        let hi = (0..m as u32).filter(|j| !ii.contains(j)).map(|j| p[j as usize]).fold(f32::MIN, f32::max);
        assert!(lo >= hi, "row {r}: chose {lo} but left {hi}");
    }
    // with the default biases, never more than k fire
    sae.b_enc.iter_mut().for_each(|b| *b = 0.0);
    let c = sae.encode(&x, 1).unwrap();
    assert!(c.mean_l0() <= k as f64 && c.mean_l0() > 0.0);
    assert!(c.idx.chunks_exact(k).zip(c.val.chunks_exact(k)).all(|(_, v)| v.iter().all(|&v| v >= 0.0)));
}

#[test]
fn decoder_rows_stay_unit_norm() {
    let (d, m, k) = (24, 64, 4);
    let mut rng = Rng::new(5);
    let x: Vec<f32> = (0..2048 * d).map(|_| rng.normal() as f32 * 3.0 + 1.0).collect();
    let mut cfg = TrainCfg::new(d, m, k, 40);
    cfg.batch = 128;
    cfg.lr = 1e-2;
    cfg.threads = 2;
    let (sae, _) = train(&x, d, &cfg, |_| {}).unwrap();
    let init = Sae::init(&x, d, m, k, cfg.seed).unwrap();
    assert!(sae.w_dec.iter().zip(&init.w_dec).any(|(a, b)| (a - b).abs() > 1e-3), "training moved the atoms");
    for (j, w) in sae.w_dec.chunks_exact(d).enumerate() {
        let n = dot(w, w).sqrt();
        assert!((n - 1.0).abs() < 1e-5, "atom {j}: norm {n}");
    }
}

/// A trainer whose latent 7 never makes the TopK (its bias is far below the others') but has a positive
/// pre-activation, so AuxK can reach it once it is dead.
fn dead_case(aux_coef: f32) -> (Trainer, Vec<f32>, f64) {
    let (d, m, k) = (8, 8, 2);
    let mut rng = Rng::new(11);
    let x: Vec<f32> = (0..32 * d).map(|_| rng.normal() as f32).collect();
    let mut sae = Sae::init(&x, d, m, k, 2).unwrap();
    sae.w_enc.iter_mut().for_each(|w| *w *= 0.01);
    sae.b_enc.iter_mut().for_each(|b| *b = 5.0);
    sae.b_enc[7] = 0.5;
    let var = mean_and_var(&x, d).1 * (sae.scale as f64).powi(2);
    let mut cfg = TrainCfg::new(d, m, k, 3);
    cfg.dead_after = 64;
    cfg.k_aux = 4;
    cfg.aux_coef = aux_coef;
    cfg.threads = 2;
    (Trainer::new(sae, cfg), x, var)
}

#[test]
fn dead_latents_are_counted_and_auxk_moves_only_them() {
    for aux_coef in [0.0f32, 0.5] {
        let (mut tr, x, var) = dead_case(aux_coef);
        assert!(tr.dead().is_empty(), "nothing is dead before dead_after tokens");
        tr.step(&x, var, 1e-3).unwrap(); // 32 tokens
        assert!(tr.dead().is_empty());
        tr.step(&x, var, 1e-3).unwrap(); // 64 tokens: latent 7 has not fired in 64
        assert_eq!(tr.dead(), vec![7]);
        assert!(!tr.ever_dead[7], "ever_dead is marked by the step that sees it dead");
        let before = tr.sae.w_dec[7 * 8..8 * 8].to_vec();
        let (_, aux) = tr.step(&x, var, 1e-3).unwrap();
        assert!(tr.ever_dead[7] && tr.ever_dead.iter().filter(|&&e| e).count() == 1);
        assert!(aux > 0.0, "AuxK loss measured");
        let moved = tr.sae.w_dec[7 * 8..8 * 8].iter().zip(&before).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        if aux_coef == 0.0 {
            assert!(moved < 1e-6, "aux_coef 0: the dead atom moved by {moved}");
        } else {
            assert!(moved > 1e-4, "aux_coef {aux_coef}: the dead atom did not move ({moved})");
        }
        // the live latents fired, so they are not dead
        assert!(tr.last_fired[..7].iter().all(|&t| t == tr.tokens_seen));
    }
}

/// Off = on = off (bit for bit), n_blocks + 1 states of [N, d], norm(last block) = xf, and xf · tok_embᵀ = the logits.
fn check_hook(cfg: Config, b: usize, t: usize, seed: u64) {
    let (_vm, m) = build_awake(cfg, seed);
    assert!(m.arch_path());
    let ids = Tensor::from_vec(tokens(b, t, cfg.vocab, seed), (b, t), &CPU).unwrap();
    let l0 = m.forward(&ids).unwrap();
    assert!(m.take_states().is_none(), "nothing kept with recording off");
    m.record_states(true);
    assert!(m.take_states().is_none(), "nothing before the first recorded forward");
    let l1 = m.forward(&ids).unwrap();
    let st = m.take_states().expect("states recorded");
    assert!(m.take_states().is_none(), "taken once");
    m.record_states(false);
    let l2 = m.forward(&ids).unwrap();
    assert!(m.take_states().is_none());
    assert_eq!(max_diff(&l0, &l1), 0.0, "recording changed the forward");
    assert_eq!(max_diff(&l0, &l2), 0.0, "the forward after recording differs");
    let nb = cfg.layers;
    assert_eq!(st.len(), nb + 1);
    for s in &st {
        assert_eq!(s.dims(), &[b * t, cfg.d]);
    }
    let xf = norm(&st[nb - 1], &m.final_norm).unwrap();
    assert_eq!(max_diff(&xf, &st[nb]), 0.0, "norm(the last block's state) is xf");
    let lg = st[nb].matmul(&m.tok.t().unwrap()).unwrap();
    assert!(max_diff(&lg, &l1) < 1e-5, "xf through the tied head gives the logits");
    // the blocks' states differ from each other (the hook does not hand one tensor out twice)
    assert!(max_diff(&st[0], &st[1]) > 1e-4);
}

#[test]
fn the_hook_is_bit_exact_on_ava_04s_everything_on_model() {
    check_hook(crate::upkeep::tests::all_on_cfg(), 2, 48, 21);
}

/// ALL_ON without the options memory.rs refuses together with --feedback (and --schema, which needs --workspace), and
/// with --feedback.
fn feedback_flags() -> String {
    let drop = ["--stp", "--delays", "--wm", "--workspace", "--holo", "--sync", "--resonate", "--modulate", "--tem", "--halting", "--schema"];
    let words: Vec<&str> = crate::upkeep::tests::ALL_ON.split_whitespace().collect();
    let mut out: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let w = words[i];
        if drop.contains(&w) {
            // a flag with a value drops its value too
            i += if words.get(i + 1).is_some_and(|v| !v.starts_with("--")) { 2 } else { 1 };
            continue;
        }
        out.push(w);
        i += 1;
    }
    out.push("--feedback");
    out.join(" ")
}

#[test]
fn the_hook_puts_feedbacks_chunks_back_in_token_order() {
    let fl = feedback_flags();
    let cfg = Config::with_kind(Kind::D, 258, 64, 2, 192).with_arch(Arch::from_flags(&fl).unwrap()).unwrap();
    assert!(cfg.arch.feedback);
    // 150 tokens: chunks of 64, 64 and 22
    check_hook(cfg, 2, 150, 23);
}

#[test]
fn the_hook_fires_on_a_lean_loaded_model() {
    let cfg = crate::upkeep::tests::all_on_cfg();
    let (vm, m) = build_awake(cfg, 25);
    let dir = scratch("sae_lean");
    let p = dir.join("model.safetensors");
    vm.save(&p).unwrap();
    std::fs::write(dir.join("arch.txt"), cfg.arch.flags() + "\n").unwrap();
    let (_, _vm2, lm) = crate::ckpt::load_auto_lean(&p, Kind::D, cfg.ctx, &CPU).unwrap();
    lm.set_track_activity(false);
    let ids = Tensor::from_vec(tokens(2, 40, cfg.vocab, 25), (2, 40), &CPU).unwrap();
    let mut got = Vec::new();
    for mm in [&m, &lm] {
        mm.record_states(true);
        let l = mm.forward(&ids).unwrap();
        got.push((l, mm.take_states().expect("states recorded")));
        mm.record_states(false);
    }
    assert_eq!(max_diff(&got[0].0, &got[1].0), 0.0, "the lean model's logits");
    assert_eq!(got[1].1.len(), cfg.layers + 1);
    for (a, b) in got[0].1.iter().zip(&got[1].1) {
        assert_eq!(max_diff(a, b), 0.0, "the lean model's states");
    }
}

#[test]
fn a_probe_separates_planted_classes() {
    let d = 16;
    let mut rng = Rng::new(13);
    for c in [2usize, 3] {
        let means: Vec<Vec<f32>> = (0..c).map(|_| (0..d).map(|_| rng.normal() as f32 * 1.5).collect()).collect();
        let make = |n: usize, rng: &mut Rng| {
            let (mut x, mut y) = (Vec::with_capacity(n * d), Vec::with_capacity(n));
            for i in 0..n {
                // half the rows are class 0 (with 3 classes, the majority)
                let k = if i % 2 == 0 { 0 } else { 1 + (i / 2) % (c - 1) };
                x.extend(means[k].iter().map(|&mu| mu + rng.normal() as f32));
                y.push(k as u32);
            }
            (x, y)
        };
        let (xa, ya) = make(3000, &mut rng);
        let (xb, yb) = make(1000, &mut rng);
        let s = fit_and_score(&xa, &ya, &xb, &yb, d, c, &ProbeCfg { epochs: 10, ..ProbeCfg::default() });
        eprintln!("{c} classes: acc {:.3}, majority {:.3}, auc {:?}, balanced {:.3}", s.acc, s.majority, s.auc, s.balanced);
        assert!((s.majority - 0.5).abs() < 1e-9, "the majority baseline is class 0's share");
        assert!(s.acc > 0.95 && s.balanced > 0.95, "{c} classes: acc {}", s.acc);
        assert_eq!(s.auc.is_some(), c == 2);
        if let Some(a) = s.auc {
            assert!(a > 0.98, "auc {a}");
        }
    }
    // shuffled labels: nothing to find, so the probe is near chance
    let mut rng = Rng::new(17);
    let x: Vec<f32> = (0..4000 * d).map(|_| rng.normal() as f32).collect();
    let y: Vec<u32> = (0..4000).map(|_| rng.below(2) as u32).collect();
    let s = fit_and_score(&x[..3000 * d], &y[..3000], &x[3000 * d..], &y[3000..], d, 2, &ProbeCfg::default());
    assert!(s.acc < 0.56 && s.auc.unwrap() < 0.56, "noise: acc {} auc {:?}", s.acc, s.auc);
}

/// The labels of every position but the last of `ids` (ctx = ids.len() − 1).
fn labels_of(ids: &[u32]) -> [Vec<u32>; 4] {
    text_labels(&byte_bpe(), ids, ids.len() - 1)
}

#[test]
fn the_text_labels_on_byte_tokens() {
    let bpe = byte_bpe();
    // the first document ends inside an open quote, so the second one shows the count starting afresh
    let text = "Ann said \"Stop.\" Then 3.5 kg! \u{201c}Hi.\u{201d} End.x Mr. \"open";
    let mut ids = bpe.encode(text);
    ids.push(EOS);
    ids.extend(bpe.encode("\"Q.\" ok"));
    ids.push(EOS);
    ids.extend(bpe.encode("z"));
    let [digit, num, end, quote] = labels_of(&ids);
    let at = |s: &str, nth: usize| -> usize {
        // the byte offset of the nth match of s in the first document
        text.match_indices(s).nth(nth).map(|(i, _)| i).unwrap_or_else(|| panic!("{s:?} #{nth}"))
    };
    let p3 = at("3", 0);
    // next_is_digit: the space before 3, and the . of 3.5
    assert_eq!((digit[p3 - 1], digit[p3], digit[p3 + 1]), (1, 0, 1));
    assert_eq!(digit.iter().filter(|&&v| v == 1).count(), 2);
    // in_number: 3 . 5, nothing else
    assert_eq!((num[p3 - 1], num[p3], num[p3 + 1], num[p3 + 2], num[p3 + 3]), (0, 1, 1, 1, 0));
    assert_eq!(num.iter().filter(|&&v| v == 1).count(), 3);
    // sentence_end: Stop." (past the closing quote), kg!, Hi.” (past the curly quote's three bytes), Mr. before the
    // space, and Q." before the space; not 3.5's point, not End.x
    let want = [at(".", 0), at("!", 0), text.find("Hi.").unwrap() + 2, text.find("Mr.").unwrap() + 2];
    let first2 = text.len() + 1; // the second document's first token
    let mut ends: Vec<usize> = end.iter().enumerate().filter(|(_, &v)| v == 1).map(|(i, _)| i).collect();
    ends.sort();
    let mut expect: Vec<usize> = want.to_vec();
    expect.push(first2 + 2);
    assert_eq!(ends, expect, "sentence ends");
    // in_quotes: inside "Stop." and “Hi.” (after the opening quote's last byte, until the closing quote's last byte)
    let (q0, q1) = (at("\"", 0), at("\"", 1));
    assert!((q0..q1).all(|p| quote[p] == 1) && quote[q1] == 0 && quote[q0 - 1] == 0);
    let (c0, c1) = (text.find('\u{201c}').unwrap(), text.find('\u{201d}').unwrap());
    assert_eq!((quote[c0], quote[c0 + 1], quote[c0 + 2]), (0, 0, 1), "“ opens on its last byte");
    assert!((c0 + 2..c1 + 2).all(|p| quote[p] == 1) && quote[c1 + 2] == 0, "” closes on its last byte");
    // "open is still inside at the document's end; the EOS resets the count, so the second document's quote opens
    // afresh (without the reset it would close)
    assert_eq!((quote[text.len() - 1], quote[text.len()], quote[first2]), (1, 0, 1));
    // with length marks: the mark is in the number, and the token before it is followed by a digit
    let marked = ava_data::numbers::insert_marks(&ids);
    let [digit_m, num_m, end_m, _] = labels_of(&marked);
    let pm = marked.iter().position(|&t| ava_data::numbers::is_mark(t)).unwrap();
    assert_eq!((digit_m[pm - 1], digit_m[pm], num_m[pm], num_m[pm + 1], num_m[pm + 2], num_m[pm + 3], num_m[pm + 4]), (1, 1, 1, 1, 1, 1, 1));
    assert_eq!(end_m.iter().filter(|&&v| v == 1).count(), 5, "the marks change no sentence end");
}

/// Deterministic text with sentences, quotes and numbers.
fn sample_text(seed: u64, sentences: usize) -> String {
    let mut r = Rng::new(seed);
    let names = ["Ann", "Bo", "Cy", "Di"];
    let mut s = String::new();
    for _ in 0..sentences {
        let nm = names[r.below(4)];
        let _ = match r.below(3) {
            0 => write!(s, "{nm} has {} apples. ", r.below(1000)),
            1 => write!(s, "{nm} said \"come here.\" "),
            _ => write!(s, "Is it {}.{}? Yes! ", r.below(10), r.below(100)),
        };
    }
    s
}

fn write_shards(dir: &Path, seed: u64, docs: usize, sentences: usize) {
    let bpe = byte_bpe();
    let mut w = ShardWriter::new(dir, bpe.vocab_size(), 1 << 20).unwrap();
    for k in 0..docs {
        let mut ids = bpe.encode(&sample_text(seed * 1000 + k as u64, sentences));
        ids.push(EOS);
        w.push_doc(&ids).unwrap();
    }
    w.finish().unwrap();
}

/// A tiny saved model with block options, val and source shards and the byte BPE, in a scratch folder.
fn tiny_setup(name: &str) -> (PathBuf, Config) {
    let dir = scratch(name);
    let cfg = Config::with_kind(Kind::D, 258, 32, 2, 64).with_arch(Arch::from_flags("--residual dense --value-residual --mlp-act relu2").unwrap()).unwrap();
    let (vm, _) = build_awake(cfg, 31);
    vm.save(dir.join("model.safetensors")).unwrap();
    std::fs::write(dir.join("arch.txt"), cfg.arch.flags() + "\n").unwrap();
    byte_bpe().save(dir.join("bpe.txt")).unwrap();
    write_shards(&dir.join("val"), 1, 8, 30);
    write_shards(&dir.join("src").join("alpha"), 2, 2, 20);
    write_shards(&dir.join("src").join("beta"), 3, 2, 20);
    (dir, cfg)
}

fn args(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn sae_train_runs_end_to_end_on_a_tiny_saved_model() {
    let (dir, _) = tiny_setup("sae_tool");
    let out = dir.join("out");
    let said = sae_train(&args(&[
        "--model", s(&dir.join("model.safetensors")), "--val", s(&dir.join("val")), "--block", "0,final", "--tokens", "2048", "--ctx", "32",
        "--latents", "4d", "--k", "4", "--steps", "60", "--batch", "64", "--threads", "2", "--out", s(&out), "--bpe", s(&dir.join("bpe.txt")),
        "--by-source", s(&dir.join("src")), "--source-tokens", "256", "--seed", "3",
    ]))
    .unwrap();
    assert!(said.contains("FVU held out") && said.contains("next-token loss"), "{said}");
    for layer in ["block0", "final"] {
        let ld = out.join(layer);
        let sae = Sae::load(&ld.join("sae.safetensors")).unwrap();
        assert_eq!((sae.d, sae.m, sae.k), (32, 128, 4));
        let rep = std::fs::read_to_string(ld.join("report.md")).unwrap();
        assert!(rep.contains("FVU held out") && rep.contains("alpha") && rep.contains("beta"), "{layer}: {rep}");
        let tsv = std::fs::read_to_string(ld.join("latents.tsv")).unwrap();
        assert_eq!(tsv.lines().count(), 129);
        assert!(tsv.lines().next().unwrap().ends_with("act_alpha\tact_beta"));
    }
    assert!(std::fs::read_to_string(out.join("final").join("report.md")).unwrap().contains("of the loss gap recovered"));
}

#[test]
fn probe_states_runs_end_to_end_on_a_tiny_saved_model() {
    let (dir, _) = tiny_setup("probe_tool");
    let out = dir.join("out");
    let said = probe_states(&args(&[
        "--model", s(&dir.join("model.safetensors")), "--val", s(&dir.join("val")), "--tokens", "2048", "--ctx", "32", "--threads", "2",
        "--by-source", s(&dir.join("src")), "--per-source", "256", "--epochs", "3", "--out", s(&out), "--bpe", s(&dir.join("bpe.txt")),
    ]))
    .unwrap();
    assert!(said.contains("| sentence_end |") && said.contains("source (2 classes)"), "{said}");
    let tsv = std::fs::read_to_string(out.join("probes.tsv")).unwrap();
    // 4 text tasks + the source task, each on tok, block0, block1 and final
    assert_eq!(tsv.lines().count(), 1 + 5 * 4, "{tsv}");
    let md = std::fs::read_to_string(out.join("report.md")).unwrap();
    assert!(md.contains("| probe | positives | majority | tok | block0 | block1 | final |"), "{md}");
}

#[test]
fn collect_par_gives_the_states_collect_gives() {
    let (dir, _) = tiny_setup("sae_par");
    let p = dir.join("model.safetensors");
    let wins = super::run::read_windows(&dir.join("val"), 32, 32 * 11).unwrap();
    assert_eq!(wins.len(), 11);
    let (_, _, m) = super::run::load_states_model(&p, 32).unwrap();
    let keep = [0, 1, 2];
    // calls of 2 windows (the last of 1), shared out over 3 workers
    let (a, _) = super::run::collect(&m, &wins, 32, &keep, 64).unwrap();
    let (b, _) = super::run::collect_par(&p, 32, &wins, &keep, 64, 3).unwrap();
    assert_eq!(a.len(), 3);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.len(), 11 * 32 * 32);
        assert!(x == y, "collect_par's states differ from collect's");
    }
    // one call holding every window gives the same states as calls of 2 (a window's rows never see the others')
    let (c, _) = super::run::collect(&m, &wins, 32, &keep, 32 * 11).unwrap();
    let md = a.iter().zip(&c).flat_map(|(x, y)| x.iter().zip(y).map(|(u, v)| (u - v).abs())).fold(0f32, f32::max);
    assert!(md < 1e-5, "batch size changed the states by {md}");
}

#[test]
fn both_tools_refuse_a_sealed_path() {
    let (dir, _) = tiny_setup("sae_refuse");
    let model = dir.join("model.safetensors");
    let val = dir.join("val");
    assert!(refuse_sealed(Path::new("a/b/c")).is_ok());
    for bad in ["x/sealed/m.safetensors", "x/Sealed-val", "SEALED"] {
        assert!(refuse_sealed(Path::new(bad)).unwrap_err().contains("refused"));
    }
    let e = sae_train(&args(&["--model", "nowhere/sealed/model.safetensors", "--val", s(&val)])).unwrap_err();
    assert!(e.contains("refused"), "{e}");
    let e = sae_train(&args(&["--model", s(&model), "--val", s(&dir.join("val_Sealed"))])).unwrap_err();
    assert!(e.contains("refused"), "{e}");
    let e = probe_states(&args(&["--model", s(&model), "--val", s(&val), "--by-source", "nowhere/SEALED"])).unwrap_err();
    assert!(e.contains("refused"), "{e}");
    let e = probe_states(&args(&["--model", s(&model), "--val", s(&val), "--bpe", "sealed.txt"])).unwrap_err();
    assert!(e.contains("refused"), "{e}");
}
