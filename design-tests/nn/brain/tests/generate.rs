// TESTING / DESIGN ONLY — ava_brain generation tests: architecture inference from a checkpoint,
// sampling helpers (greedy, top-k, top-p, repetition penalty, seed determinism), deterministic
// generation from the v32 S-small checkpoint (when present), and the `score` bits-per-byte round trip.
// Run: CARGO_BUILD_JOBS=4 cargo test --release -- --test-threads=4

use ava_brain::eval::{byte_lengths, window_io, BpbAcc};
use ava_brain::generate::{
    apply_repetition_penalty, infer_arch, load_checkpoint, sample_token, score_tokens, top_k_indices, top_p_truncate,
    windows_of, Generator, Overrides, Sampling, Utf8Stream, DEFAULT_CTX,
};
use ava_brain::{Config, Kind, Model};
use ava_data::bpe::{Bpe, EOS};
use ava_grad::params::read_checkpoint;
use ava_grad::{Graph, Rng};
use std::path::PathBuf;

const VOCAB: usize = 4096;
const BPE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/nn/bpe-4096.txt");
const S_SMALL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/runs/v32/S-small/model.awts");
const TMP_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/target/test-tmp");

fn tmp_dir() -> PathBuf {
    let d = PathBuf::from(TMP_DIR);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn arch_is_inferred_from_each_kind_of_checkpoint() {
    let dir = tmp_dir();
    for kind in [Kind::T, Kind::S, Kind::H] {
        let (d, layers, ctx) = (32usize, 3usize, 16usize);
        let cfg = Config::new(kind, VOCAB, d, layers, ctx);
        let (m, p) = Model::new(cfg, 4);
        let path = dir.join(format!("arch_{}.awts", kind.name()));
        p.save(&path).unwrap();

        let items = read_checkpoint(&path).unwrap();
        let inf = infer_arch(&items).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        assert_eq!(inf.kind, kind);
        assert_eq!(inf.vocab, VOCAB);
        assert_eq!(inf.d, d);
        assert_eq!(inf.layers, layers);
        // only T and H carry a position table, so only they fix ctx
        assert_eq!(inf.ctx, if kind.has_pos() { Some(ctx) } else { None }, "{kind:?} ctx");
        assert_eq!(inf.ctx_or_default(DEFAULT_CTX), if kind.has_pos() { ctx } else { DEFAULT_CTX });

        // the loader rebuilds the same model and the same logits
        let (m2, p2, cfg2) = load_checkpoint(&path, &Overrides::default()).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        assert_eq!(cfg2.kind, kind);
        assert_eq!((cfg2.d, cfg2.layers), (d, layers));
        assert_eq!(cfg2.ctx, if kind.has_pos() { ctx } else { DEFAULT_CTX });
        let ids: Vec<u32> = (0..16).map(|i| (i * 37 % VOCAB) as u32).collect();
        let a = {
            let mut g = Graph::new(&p);
            let f = m.forward(&mut g, &ids, 1, 16);
            g.value(f.logits).data.clone()
        };
        let b = {
            let mut g = Graph::new(&p2);
            let f = m2.forward(&mut g, &ids, 1, 16);
            g.value(f.logits).data.clone()
        };
        assert_eq!(bits(&a), bits(&b), "{kind:?}: reloaded logits differ");

        // flags that repeat the checkpoint are accepted
        let same = Overrides { kind: Some(kind), d: Some(d), layers: Some(layers), ctx: inf.ctx };
        assert_eq!(load_checkpoint(&path, &same).unwrap().2, cfg2, "{kind:?}: matching flags changed the config");
        // --ctx sets the window for S (no table) but may not contradict a T/H position table
        let ov = Overrides { ctx: Some(48), ..Overrides::default() };
        match load_checkpoint(&path, &ov) {
            Ok((_, _, cfg3)) => {
                assert!(!kind.has_pos(), "{kind:?}: --ctx 48 contradicts the {ctx}-row position table");
                assert_eq!(cfg3.ctx, 48);
            }
            Err(e) => assert!(kind.has_pos(), "{kind:?}: --ctx 48 should be accepted: {e}"),
        }
        // a wrong override is an error, not a panic
        let bad = Overrides { d: Some(64), ..Overrides::default() };
        assert!(load_checkpoint(&path, &bad).is_err(), "{kind:?}: wrong --d must fail");
        let bad = Overrides { kind: Some(if kind == Kind::T { Kind::S } else { Kind::T }), ..Overrides::default() };
        assert!(load_checkpoint(&path, &bad).is_err(), "{kind:?}: wrong --arch must fail");
        std::fs::remove_file(&path).ok();
    }
}

#[test]
fn greedy_when_temperature_zero_or_top_k_one() {
    let logits = [0.1f32, 3.0, -1.0, 2.9, 2.99];
    for seed in 1..20u64 {
        let mut rng = Rng::new(seed);
        let s = Sampling { temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0 };
        assert_eq!(sample_token(&logits, &s, &mut rng), 1, "temperature 0 is argmax");
        let s = Sampling { temperature: 1.0, top_k: 1, top_p: 1.0, repetition_penalty: 1.0 };
        assert_eq!(sample_token(&logits, &s, &mut rng), 1, "top-k 1 is argmax");
        let s = Sampling { temperature: 1.0, top_k: 0, top_p: 0.0, repetition_penalty: 1.0 };
        assert_eq!(sample_token(&logits, &s, &mut rng), 1, "top-p 0 keeps only the best token");
    }
    // ties go to the lowest index
    let mut rng = Rng::new(1);
    let s = Sampling { temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0 };
    assert_eq!(sample_token(&[1.0, 2.0, 2.0], &s, &mut rng), 1);
}

#[test]
fn top_k_and_top_p_filters() {
    let logits = [0.0f32, 2.0, 1.0, 3.0, -1.0];
    assert_eq!(top_k_indices(&logits, 0), vec![3, 1, 2, 0, 4], "k = 0 keeps all, sorted by logit");
    assert_eq!(top_k_indices(&logits, 2), vec![3, 1]);
    assert_eq!(top_k_indices(&logits, 99), vec![3, 1, 2, 0, 4], "k past the vocab keeps all");
    // p = 1 keeps everything
    let mut c = top_k_indices(&logits, 0);
    top_p_truncate(&mut c, &logits, 1.0);
    assert_eq!(c, vec![3, 1, 2, 0, 4]);
    let mut c = top_k_indices(&logits, 0);
    top_p_truncate(&mut c, &logits, 1.5);
    assert_eq!(c.len(), 5);
    // softmax of [3, 2, 1, 0, -1] ≈ [.636, .234, .086, .032, .012]: p = .5 keeps 1, p = .7 keeps 2, p = .9 keeps 3
    for (p, want) in [(0.5f32, 1usize), (0.7, 2), (0.9, 3), (0.99, 5)] {
        let mut c = top_k_indices(&logits, 0);
        top_p_truncate(&mut c, &logits, p);
        assert_eq!(c.len(), want, "top-p {p}");
        assert_eq!(c[0], 3);
    }
    // top-p applies to the mass of the candidates it is given (after top-k), never to tokens cut by top-k
    let mut c = top_k_indices(&logits, 2);
    top_p_truncate(&mut c, &logits, 0.75);
    assert_eq!(c, vec![3, 1], "two candidates of ≈73%/27%: p = .75 needs both");
}

#[test]
fn repetition_penalty_pushes_seen_tokens_down() {
    let mut logits = vec![2.0f32, -2.0, 1.0, 0.5];
    apply_repetition_penalty(&mut logits, &[0, 1, 1], 2.0);
    assert_eq!(logits, vec![1.0, -4.0, 1.0, 0.5], "positive logits divide, negative multiply, repeats count once");
    // penalty 1 (or below) is the identity
    let mut same = vec![2.0f32, -2.0, 1.0];
    apply_repetition_penalty(&mut same, &[0, 1, 2], 1.0);
    assert_eq!(same, vec![2.0, -2.0, 1.0]);
    // out-of-range ids are ignored
    let mut l = vec![1.0f32, 1.0];
    apply_repetition_penalty(&mut l, &[7], 3.0);
    assert_eq!(l, vec![1.0, 1.0]);
    // through the sampler: a heavily penalised favourite loses to the runner-up under greedy decoding
    let s = Sampling { temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0 };
    let mut rng = Rng::new(1);
    let mut l = vec![3.0f32, 2.5, 0.0];
    assert_eq!(sample_token(&l, &s, &mut rng), 0);
    apply_repetition_penalty(&mut l, &[0], 1.5);
    assert_eq!(sample_token(&l, &s, &mut rng), 1);
}

#[test]
fn sampling_is_deterministic_for_a_seed() {
    let mut rng = Rng::new(3);
    let logits: Vec<f32> = (0..200).map(|_| rng.normal() * 2.0).collect();
    let s = Sampling { temperature: 0.8, top_k: 40, top_p: 0.95, repetition_penalty: 1.1 };
    let draw = |seed: u64| -> Vec<u32> {
        let mut r = Rng::new(seed);
        (0..100).map(|_| sample_token(&logits, &s, &mut r)).collect()
    };
    let a = draw(7);
    assert_eq!(a, draw(7), "same seed, different draws");
    assert_ne!(a, draw(8), "different seeds should differ over 100 draws");
    // every draw is one of the top-40 candidates
    let allowed = top_k_indices(&logits, 40);
    assert!(a.iter().all(|&t| allowed.contains(&(t as usize))));
    // and not all the same token (temperature 0.8 over 200 N(0, 2) logits spreads the mass)
    assert!(a.iter().any(|&t| t != a[0]));
}

#[test]
fn utf8_stream_emits_only_complete_characters() {
    let mut st = Utf8Stream::default();
    let e = "é".as_bytes(); // 2 bytes
    assert_eq!(st.push(b"ab"), "ab");
    assert_eq!(st.push(&e[..1]), "", "half a character waits");
    assert_eq!(st.push(&e[1..]), "é");
    assert_eq!(st.push(&[0xff]), "\u{fffd}", "a byte that can start nothing is replaced at once");
    assert_eq!(st.push(&e[..1]), "", "a lead byte waits in case its continuation follows");
    assert_eq!(st.push(b"z"), "\u{fffd}z", "a lead byte followed by something else is replaced");
    assert_eq!(st.push(&e[..1]), "");
    assert_eq!(st.finish(), "\u{fffd}");
    assert_eq!(st.finish(), "", "finish drains the buffer");
}

#[test]
fn generation_from_v32_s_small_is_deterministic() {
    if !std::path::Path::new(S_SMALL).exists() {
        println!("note: {S_SMALL} not present, skipping");
        return;
    }
    let bpe = Bpe::load(BPE_PATH).expect("load bpe-4096");
    let (m, p, cfg) = load_checkpoint(S_SMALL, &Overrides::default()).expect("load S-small");
    assert_eq!(cfg.kind, Kind::S);
    assert_eq!((cfg.d, cfg.layers), (64, 2));
    let gen = Generator { model: &m, params: &p, threads: 4 };
    let s = Sampling { temperature: 0.8, top_k: 40, top_p: 0.95, repetition_penalty: 1.1 };
    let prompt = bpe.encode("The cat");
    let run = || gen.generate(&prompt, 20, &s, 1, |t| t == EOS);
    let a = run();
    let b = run();
    assert_eq!(a, b, "seed 1 twice gave different tokens");
    assert!(!a.is_empty());
    assert!(a.len() <= 20);
    assert!(a.iter().all(|&t| (t as usize) < VOCAB && t != EOS));
    let text = bpe.decode(&a);
    assert!(std::str::from_utf8(text.as_bytes()).is_ok());
    assert!(!text.is_empty());
    let lossy = std::str::from_utf8(&bpe.decode_bytes(&a)).is_err();
    let shown = format!("The cat{text}");
    println!("S-small sample (seed 1, 20 tokens{}): {shown:?}", if lossy { ", had invalid UTF-8 bytes" } else { "" });
    // a different seed changes the continuation
    let c = gen.generate(&prompt, 20, &s, 2, |t| t == EOS);
    assert_ne!(a, c, "seed 2 reproduced seed 1");
    // greedy decoding is seed-independent
    let g0 = Sampling { temperature: 0.0, ..s };
    assert_eq!(gen.generate(&prompt, 10, &g0, 1, |t| t == EOS), gen.generate(&prompt, 10, &g0, 99, |t| t == EOS));
}

/// Generation speed with a full 256-token window at two sizes (random init: speed does not depend on
/// the weights). Ignored by default (timing); run with `--ignored --nocapture`.
#[test]
#[ignore]
fn generation_speed_at_two_sizes() {
    let ctx = 256;
    for &(d, layers) in &[(128usize, 4usize), (192, 6)] {
        let (m, p) = Model::new(Config::new(Kind::S, VOCAB, d, layers, ctx), 1);
        let gen = Generator { model: &m, params: &p, threads: 4 };
        let window: Vec<u32> = (0..ctx).map(|i| (i * 131 % 256) as u32).collect();
        gen.next_logits(&window); // warm-up
        let steps = 8;
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            gen.next_logits(&window);
        }
        let secs = t0.elapsed().as_secs_f64();
        let (total, _) = Model::weight_counts(&p);
        println!("S d={d} L={layers} ctx={ctx} ({total} weights): {:.3} s/token = {:.1} tokens/s at a full window", secs / steps as f64, steps as f64 / secs);
    }
}

#[test]
fn score_round_trip_matches_eval() {
    let dir = tmp_dir();
    let path = dir.join("score_tiny.txt");
    let text = "The quick brown fox jumps over the lazy dog. Water is wet, and the cat sat on the mat.\nA second line, with numbers 123 and a tab\there.\n";
    std::fs::write(&path, text).unwrap();
    let bpe = Bpe::load(BPE_PATH).expect("load bpe-4096");
    let byte_len = byte_lengths(&bpe);
    let ctx = 16;
    let (m, p) = Model::new(Config::new(Kind::S, VOCAB, 32, 2, ctx), 1);

    let read = std::fs::read_to_string(&path).unwrap();
    let tokens = bpe.encode_doc(&read);
    assert_eq!(*tokens.last().unwrap(), EOS);
    let w = windows_of(&tokens, ctx);
    assert_eq!(w.len(), (tokens.len() - 1).div_ceil(ctx));
    assert!(w.iter().all(|x| x.len() == ctx + 1));
    for pair in w.windows(2) {
        assert_eq!(pair[0][ctx], pair[1][0], "windows must overlap by one token (stride ctx)");
    }

    let a = score_tokens(&m, &p, &tokens, ctx, &byte_len, 1, 4);
    let b = score_tokens(&m, &p, &tokens, ctx, &byte_len, 1, 4);
    assert_eq!(a, b, "scoring twice must agree exactly");
    assert!(a.bpb().is_finite() && a.bpb() > 0.0);
    // every token after the first is scored once; bytes = file bytes − first token's bytes + 1 for EOS
    assert_eq!(a.tokens, tokens.len() as u64 - 1);
    assert_eq!(a.bytes, text.len() as u64 - byte_len[tokens[0] as usize] as u64 + 1);

    // the same number from eval.rs's pieces by hand: one window at a time
    let mut acc = BpbAcc::default();
    let mut rem = u64::MAX;
    for win in &w {
        let (ids, tg) = window_io(win, &mut rem);
        let mut g = Graph::new(&p);
        g.set_threads(4);
        let f = m.forward(&mut g, &ids, 1, ctx);
        acc.add_rows(&g.value(f.logits).data, VOCAB, &tg, &byte_len);
    }
    assert_eq!(a, acc, "score_tokens disagrees with eval.rs on the same tokens");
    println!("tiny file: {} tokens, {} bytes, {:.4} nats/token, {:.4} bpb", a.tokens, a.bytes, a.nats_per_token(), a.bpb());
    // batching does not change the count, and the nats only within float rounding of the logits
    let c = score_tokens(&m, &p, &tokens, ctx, &byte_len, 4, 4);
    assert_eq!((c.tokens, c.bytes), (a.tokens, a.bytes));
    assert!((c.nats - a.nats).abs() < 1e-3 * a.nats.max(1.0), "batched {} vs {}", c.nats, a.nats);
    std::fs::remove_file(&path).ok();
}
