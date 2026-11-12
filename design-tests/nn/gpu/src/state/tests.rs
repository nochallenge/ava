// TESTING / DESIGN ONLY — tests of src/state.rs (v42 S1 forward_with_state, S2 saved state and engram cap), CPU only.

use super::*;
use crate::arch::Arch;
use crate::model::{Config, Kind, Model};
use crate::testkit::{fix_weights, scratch, tiny, unit_noise};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

const CPU: Device = Device::Cpu;

/// Build `cfg` with fixed weights, and give every all-zero weight (the read-outs that start at 0: the magnitude code, the
/// engram's Wo, the gates, the new parts' gains) small fixed noise too, so every part is exercised.
pub(crate) fn build_awake(cfg: Config, seed: u64) -> (VarMap, Model) {
    let vm = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
    fix_weights(&vm, &cfg, seed);
    {
        let data = vm.data().lock().unwrap();
        let mut names: Vec<&String> = data.keys().collect();
        names.sort();
        for (k, name) in names.into_iter().enumerate() {
            let var = &data[name];
            let v: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
            if v.iter().all(|&x| x == 0.0) {
                let w: Vec<f32> = unit_noise(v.len(), seed.wrapping_mul(7919).wrapping_add(k as u64)).iter().map(|z| z * 0.1).collect();
                var.set(&Tensor::from_vec(w, var.shape(), &CPU).unwrap()).unwrap();
            }
        }
    }
    m.set_track_activity(false);
    (vm, m)
}

/// A v40-style D: glance, magnitude code and the engram in block 1 (vocab 4112, d 32, 2 blocks).
pub(crate) fn v40_cfg() -> Config {
    Config::with_kind(Kind::D, ava_data::numbers::VOCAB_GLANCE, 32, 2, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap()
}

/// Deterministic tokens; for the glance vocabulary, numbers with their length marks among them.
pub(crate) fn tokens(b: usize, t: usize, vocab: usize, seed: u64) -> Vec<u32> {
    let noise = unit_noise(b * t, seed);
    let mut v: Vec<u32> = noise.iter().map(|z| (((z + 1.8) / 3.6).clamp(0.0, 0.999) * 250.0) as u32).collect();
    if vocab > 4096 {
        // every 9 tokens: a mark of length 3 and three digits
        let mark = ava_data::numbers::MARK_BASE + 2;
        for r in 0..b {
            for p in (2..t.saturating_sub(4)).step_by(9) {
                v[r * t + p] = mark;
                for k in 0..3 {
                    v[r * t + p + 1 + k] = 48 + ((p + k) % 10) as u32;
                }
            }
        }
    }
    v
}

pub(crate) fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.sub(b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
}

/// Logits [B, T, V] of one forward over toks [B, T].
pub(crate) fn one_pass(m: &Model, toks: &[u32], b: usize, t: usize) -> Tensor {
    m.forward(&Tensor::from_vec(toks.to_vec(), (b, t), &CPU).unwrap()).unwrap().reshape((b, t, ())).unwrap()
}

/// Logits [B, T, V] of reading toks [B, T] in pieces with carried state.
pub(crate) fn in_pieces(m: &Model, st: &mut State, toks: &[u32], b: usize, t: usize, pieces: &[usize]) -> Tensor {
    assert_eq!(pieces.iter().sum::<usize>(), t);
    let mut outs = Vec::new();
    let mut p0 = 0;
    for &p in pieces {
        let chunk: Vec<u32> = (0..b).flat_map(|r| toks[r * t + p0..r * t + p0 + p].to_vec()).collect();
        let l = m.forward_with_state(&Tensor::from_vec(chunk, (b, p), &CPU).unwrap(), st).unwrap();
        outs.push(l.reshape((b, p, ())).unwrap());
        p0 += p;
    }
    Tensor::cat(&outs, 1).unwrap()
}

fn models() -> Vec<(&'static str, Model, usize)> {
    let mut v = Vec::new();
    for (name, kind) in [("S", Kind::S), ("D", Kind::D), ("A", Kind::A), ("T", Kind::T)] {
        let (_, _, m) = tiny(kind, 3);
        v.push((name, m, 258));
    }
    let (_, m) = build_awake(v40_cfg(), 5);
    v.push(("D+glance+mag+engram", m, ava_data::numbers::VOCAB_GLANCE));
    v
}

#[test]
fn a_fresh_state_gives_forward_exactly() {
    for (name, m, vocab) in models() {
        let (b, t) = (3, 40);
        let toks = tokens(b, t, vocab, 11);
        let mut st = m.state(b).unwrap();
        let got = in_pieces(&m, &mut st, &toks, b, t, &[t]);
        assert_eq!(max_diff(&got, &one_pass(&m, &toks, b, t)), 0.0, "{name}");
        assert_eq!(st.seen, vec![t; b]);
    }
}

#[test]
fn pieces_give_one_pass() {
    let (b, t) = (2, 64);
    for (name, m, vocab) in models() {
        let toks = tokens(b, t, vocab, 21);
        let want = one_pass(&m, &toks, b, t);
        for pieces in [vec![32, 32], vec![17, 30, 17], vec![5; 12].into_iter().chain([4]).collect::<Vec<_>>()] {
            let mut st = m.state(b).unwrap();
            let got = in_pieces(&m, &mut st, &toks, b, t, &pieces);
            let diff = max_diff(&got, &want);
            assert!(diff < 1e-4, "{name} {pieces:?}: {diff}");
        }
        // token by token
        let mut st = m.state(b).unwrap();
        let got = in_pieces(&m, &mut st, &toks, b, t, &[1; 64]);
        assert!(max_diff(&got, &want) < 1e-4, "{name} token by token");
    }
}

#[test]
fn the_lif_loop_and_its_ops_agree() {
    let (b, t, d) = (2, 30, 16);
    let cur = Tensor::from_vec(unit_noise(b * t * d, 4).iter().map(|x| x * 1.5).collect::<Vec<f32>>(), (b, t, d), &CPU).unwrap();
    let beta = Tensor::from_vec(unit_noise(d, 5).iter().map(|x| 0.8 + 0.1 * x.clamp(-1.0, 1.0)).collect::<Vec<f32>>(), d, &CPU).unwrap();
    let theta = Tensor::from_vec(unit_noise(d, 6).iter().map(|x| 1.0 + 0.2 * x).collect::<Vec<f32>>(), d, &CPU).unwrap();
    for th in [None, Some(&theta)] {
        let (s1, l1) = lif_infer(&cur, &beta, th, None).unwrap();
        let (s2, l2) = lif_infer_ops(&cur, &beta, th, None).unwrap();
        assert_eq!(max_diff(&s1, &s2), 0.0);
        assert_eq!(max_diff(&l1.v, &l2.v), 0.0);
        // and from a carried state, in two pieces
        let (a, la) = lif_infer(&cur.narrow(1, 0, 11).unwrap(), &beta, th, None).unwrap();
        let (bb, _) = lif_infer_ops(&cur.narrow(1, 11, t - 11).unwrap(), &beta, th, Some(&la)).unwrap();
        assert_eq!(max_diff(&Tensor::cat(&[&a, &bb], 1).unwrap(), &s1), 0.0);
    }
    // the fused scan's CPU kernel gives the same spikes
    let fused = crate::lif::lif(&cur, &beta).unwrap();
    assert_eq!(max_diff(&fused, &lif_infer(&cur, &beta, None, None).unwrap().0), 0.0);
}

#[test]
fn rows_are_independent_and_copyable() {
    let (_, m) = build_awake(v40_cfg(), 8);
    let vocab = ava_data::numbers::VOCAB_GLANCE;
    let (b, t) = (3, 30);
    let toks = tokens(b, t, vocab, 31);
    let mut st = m.state(b).unwrap();
    let all = in_pieces(&m, &mut st, &toks, b, t, &[20, 10]);
    for r in 0..b {
        let one = one_pass(&m, &toks[r * t..(r + 1) * t], 1, t);
        assert!(max_diff(&all.narrow(0, r, 1).unwrap(), &one) < 1e-4, "row {r}");
    }
    // one prompt's state copied to two rows, each read on with its own continuation
    let (p, c) = (18, 12);
    let prompt = &toks[..p];
    let mut st = m.state(1).unwrap();
    in_pieces(&m, &mut st, prompt, 1, p, &[p]);
    let mut two = st.select(&[0, 0]).unwrap();
    let conts: Vec<u32> = [&toks[t..t + c], &toks[2 * t..2 * t + c]].concat();
    let got = in_pieces(&m, &mut two, &conts, 2, c, &[c]);
    for r in 0..2 {
        let whole: Vec<u32> = prompt.iter().chain(&conts[r * c..(r + 1) * c]).copied().collect();
        let want = one_pass(&m, &whole, 1, p + c).narrow(1, p, c).unwrap();
        assert!(max_diff(&got.narrow(0, r, 1).unwrap(), &want) < 1e-4, "copy {r}");
    }
    // a clone is a full copy: reading on with one leaves the other where it was
    let mut a = st.clone();
    in_pieces(&m, &mut a, &toks[..4], 1, 4, &[4]);
    assert_eq!(st.seen, vec![p]);
    assert_eq!(a.seen, vec![p + 4]);
}

#[test]
fn rows_of_different_lengths_join() {
    for (name, m, vocab) in models() {
        let (la, lb, more) = (10, 21, 6);
        let ta = tokens(1, la + more, vocab, 41);
        let tb = tokens(1, lb + more, vocab, 42);
        let mut sa = m.state(1).unwrap();
        in_pieces(&m, &mut sa, &ta[..la], 1, la, &[la]);
        let mut sb = m.state(1).unwrap();
        in_pieces(&m, &mut sb, &tb[..lb], 1, lb, &[lb]);
        let mut both = State::cat(&[sa, sb]).unwrap();
        assert_eq!(both.seen, vec![la, lb]);
        let next: Vec<u32> = ta[la..].iter().chain(&tb[lb..]).copied().collect();
        let got = in_pieces(&m, &mut both, &next, 2, more, &[more]);
        let wa = one_pass(&m, &ta, 1, la + more).narrow(1, la, more).unwrap();
        let wb = one_pass(&m, &tb, 1, lb + more).narrow(1, lb, more).unwrap();
        assert!(max_diff(&got.narrow(0, 0, 1).unwrap(), &wa) < 1e-4, "{name} short row");
        assert!(max_diff(&got.narrow(0, 1, 1).unwrap(), &wb) < 1e-4, "{name} long row");
    }
}

#[test]
fn saved_state_resumes_and_is_keyed_by_the_model() {
    let (_, m) = build_awake(v40_cfg(), 9);
    let vocab = ava_data::numbers::VOCAB_GLANCE;
    let toks = tokens(2, 30, vocab, 51);
    let mut st = m.state(2).unwrap();
    st.cap = Some(EngramCap { max: 12, rule: Evict::LeastRead });
    let first: Vec<u32> = (0..2).flat_map(|r| toks[r * 30..r * 30 + 20].to_vec()).collect();
    in_pieces(&m, &mut st, &first, 2, 20, &[20]);
    let dir = scratch("state_save");
    let path = dir.join("s.safetensors");
    st.save(&m, &path).unwrap();
    let mut back = State::load(&m, &path, &CPU).unwrap();
    assert_eq!(back.seen, st.seen);
    assert_eq!(back.hist, st.hist);
    assert_eq!(back.cap, st.cap);
    let rest: Vec<u32> = (0..2).flat_map(|r| toks[r * 30 + 20..r * 30 + 30].to_vec()).collect();
    let a = in_pieces(&m, &mut st, &rest, 2, 10, &[10]);
    let b = in_pieces(&m, &mut back, &rest, 2, 10, &[10]);
    assert_eq!(max_diff(&a, &b), 0.0);
    // another model's state is refused; per-conversation files
    let (_, other) = build_awake(v40_cfg(), 10);
    assert!(State::load(&other, &path, &CPU).err().unwrap().contains("another model"));
    let cs = ConvStore { dir: dir.join("conversations") };
    assert!(matches!(cs.load("chat-1", &m, &CPU), Loaded::Fresh));
    cs.save("chat-1", &m, &st.select(&[1]).unwrap()).unwrap();
    match cs.load("chat-1", &m, &CPU) {
        Loaded::Resumed(s) => assert_eq!(s.seen, vec![30]),
        _ => panic!("not resumed"),
    }
    assert!(matches!(cs.load("chat-1", &other, &CPU), Loaded::Stale(_)));
    assert!(cs.path("a/b c").file_name().unwrap().to_string_lossy().starts_with("state-h"));
    assert!(cs.path("chat-1").ends_with("state-chat-1.safetensors"));
    assert_eq!(ConvStore::next_to(Path::new("runs/x/model.safetensors")).dir, Path::new("runs/x/conversations"));
    // a one-row stream resumes from a saved state
    let mut s = m.stream_from(st.select(&[0]).unwrap()).unwrap();
    assert_eq!(s.seen, 30);
    let _ = m.step(&mut s, 65).unwrap();
}

#[test]
fn engram_cap_rules() {
    let (_, m) = build_awake(v40_cfg(), 12);
    let vocab = ava_data::numbers::VOCAB_GLANCE;
    let (t1, t2) = (20, 10);
    let toks = tokens(1, t1 + t2, vocab, 61);
    let want = one_pass(&m, &toks, 1, t1 + t2);
    // a cap the text never reaches changes nothing
    let mut st = m.state(1).unwrap();
    st.cap = Some(EngramCap { max: 100, rule: Evict::Oldest });
    assert_eq!(max_diff(&in_pieces(&m, &mut st, &toks, 1, t1 + t2, &[t1, t2]), &want), max_diff(&in_pieces(&m, &mut m.state(1).unwrap(), &toks, 1, t1 + t2, &[t1, t2]), &want));
    // oldest: after the first call the store is the last 8 slots of the uncapped store
    let mut free = m.state(1).unwrap();
    st = m.state(1).unwrap();
    st.cap = Some(EngramCap { max: 8, rule: Evict::Oldest });
    in_pieces(&m, &mut free, &toks[..t1], 1, t1, &[t1]);
    in_pieces(&m, &mut st, &toks[..t1], 1, t1, &[t1]);
    assert_eq!(st.engram_slots(), 8);
    assert_eq!(max_diff(&st.eng.as_ref().unwrap().k, &free.eng.as_ref().unwrap().k.narrow(2, t1 - 8, 8).unwrap()), 0.0);
    // the capped store still reads on, and differs from the uncapped one only through what it forgot
    let more = in_pieces(&m, &mut st, &toks[t1..], 1, t2, &[t2]);
    assert_eq!(st.engram_slots(), 8);
    assert!(max_diff(&more, &want.narrow(1, t1, t2).unwrap()) > 0.0);
    // least-read: the kept slots are those with the most reads per later query (counted by a cap not yet reached)
    let mut counted = m.state(1).unwrap();
    counted.cap = Some(EngramCap { max: 1000, rule: Evict::LeastRead });
    in_pieces(&m, &mut counted, &toks[..t1], 1, t1, &[t1]);
    let kv = counted.eng.as_ref().unwrap();
    let reads: Vec<f32> = kv.reads.as_ref().unwrap().get(0).unwrap().to_vec1().unwrap();
    let age: Vec<f32> = kv.age.as_ref().unwrap().get(0).unwrap().to_vec1().unwrap();
    assert_eq!(age[0], (t1 - 1) as f32);
    assert_eq!(age[t1 - 1], 0.0);
    let mut lr = m.state(1).unwrap();
    lr.cap = Some(EngramCap { max: 8, rule: Evict::LeastRead });
    in_pieces(&m, &mut lr, &toks[..t1], 1, t1, &[t1]);
    let mut rank: Vec<(u8, f32, usize)> = (0..t1).map(|j| if age[j] >= 1.0 { (1, reads[j] / age[j], j) } else { (2, 0.0, j) }).collect();
    rank.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut keep: Vec<u32> = rank[t1 - 8..].iter().map(|x| x.2 as u32).collect();
    keep.sort_unstable();
    let want_k = kv.k.index_select(&Tensor::from_vec(keep, 8, &CPU).unwrap(), 2).unwrap();
    assert_eq!(max_diff(&lr.eng.as_ref().unwrap().k, &want_k), 0.0);
    assert!(EngramCap::parse("none").unwrap().is_none());
    assert_eq!(EngramCap::parse("64:least-read").unwrap(), Some(EngramCap { max: 64, rule: Evict::LeastRead }));
    assert_eq!(EngramCap::parse("64").unwrap(), Some(EngramCap { max: 64, rule: Evict::Oldest }));
    assert!(EngramCap::parse("0").is_err() && EngramCap::parse("5:newest").is_err());
}

/// Block options with every carried state: pieces and the token-by-token stream give one pass.
#[test]
fn block_options_read_in_pieces_and_stream() {
    let mut many = Arch::from_flags("--value-residual --gates channel --mem-momentum 0.3 --alif --spikes ternary --lif-scale --residual dense --mlp-act relu2 --kwta 0.25 --homeostasis 0.1").unwrap();
    many.head_size = 16;
    let fb = Arch::from_flags("--feedback --lif-init logspaced").unwrap();
    let mem = Arch::from_flags("--dendrites 2 --stp --delays 3 --wm 2 --pk-mem 16 --pk-topk 2").unwrap();
    let eng = Arch::from_flags("--eng-temp --eng-ssmax --btsp 4 --eng-topk 6").unwrap();
    for (name, arch, engram, t, pieces) in [
        ("many", many, None, 64, vec![20, 44]),
        ("feedback", fb, None, 150, vec![50, 14, 70, 16]),
        ("parts", mem, Some(1), 64, vec![30, 34]),
        ("engram options", eng, Some(1), 64, vec![9, 40, 15]),
    ] {
        let mut cfg = Config::with_kind(Kind::D, 258, 32, 2, 64);
        if let Some(l) = engram {
            cfg = cfg.with_engram(l).unwrap();
        }
        let cfg = cfg.with_arch(arch).unwrap();
        let (_, m) = build_awake(cfg, 14);
        let b = 2;
        let toks = tokens(b, t, 258, 71);
        let want = one_pass(&m, &toks, b, t);
        let mut st = m.state(b).unwrap();
        assert_eq!(max_diff(&in_pieces(&m, &mut st, &toks, b, t, &[t]), &want), 0.0, "{name} fresh");
        let mut st = m.state(b).unwrap();
        let d = max_diff(&in_pieces(&m, &mut st, &toks, b, t, &pieces), &want);
        assert!(d < 1e-4, "{name} pieces: {d}");
        // the token-by-token stream (generation for block options) and a batched feed
        let mut s = m.stream().unwrap();
        let mut last = None;
        for &tok in &toks[..t] {
            last = Some(m.step(&mut s, tok).unwrap());
        }
        assert!(max_diff(&last.unwrap(), &want.get(0).unwrap().get(t - 1).unwrap()) < 1e-4, "{name} stream");
        let mut s = m.stream().unwrap();
        let fed = m.feed(&mut s, &toks[..t]).unwrap();
        assert!(max_diff(&fed, &want.get(0).unwrap().get(t - 1).unwrap()) < 1e-4, "{name} feed");
    }
}

/// v48: Ava 0.4's exact block recipe (WIN + SLOW, with int2 spikes, fixed timescales and head size 64, which the test above
/// does not cover), with an engram: chat's token-by-token stream and batched feed give the
/// full forward's logits. (The glance and magnitude code only change the input; stream_glance_* and stream_v40_* cover them.)
#[test]
fn ava04_recipe_streams_match_forward() {
    let flags = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense \
                 --resid-fix --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif \
                 --homeostasis 0.1";
    let arch = Arch::from_flags(flags).unwrap();
    let cfg = Config::with_kind(Kind::D, 258, 64, 2, 64).with_engram(1).unwrap().with_arch(arch).unwrap();
    let (_, m) = build_awake(cfg, 48);
    let (b, t) = (2, 64);
    let toks = tokens(b, t, 258, 72);
    let want = one_pass(&m, &toks, b, t);
    let mut st = m.state(b).unwrap();
    let d = max_diff(&in_pieces(&m, &mut st, &toks, b, t, &[25, 39]), &want);
    assert!(d < 1e-4, "pieces: {d}");
    let last = want.get(0).unwrap().get(t - 1).unwrap();
    let mut s = m.stream().unwrap();
    let mut out = None;
    for &tok in &toks[..t] {
        out = Some(m.step(&mut s, tok).unwrap());
    }
    let d = max_diff(&out.unwrap(), &last);
    assert!(d < 1e-4, "stream: {d}");
    let mut s = m.stream().unwrap();
    let d = max_diff(&m.feed(&mut s, &toks[..t]).unwrap(), &last);
    assert!(d < 1e-4, "feed: {d}");
}

#[test]
fn batched_stream_matches_todays_stream() {
    let (_, m) = build_awake(v40_cfg(), 15);
    let toks = tokens(1, 40, ava_data::numbers::VOCAB_GLANCE, 81);
    let mut old = m.stream().unwrap();
    let mut new = m.stream_batched().unwrap();
    let mut a = None;
    for &t in &toks {
        a = Some(m.step(&mut old, t).unwrap());
    }
    let b = m.feed(&mut new, &toks).unwrap();
    assert!(max_diff(&a.unwrap(), &b) < 1e-4);
    assert_eq!(new.seen, 40);
    assert!(Model::stream_state(&new).is_some() && Model::stream_state(&old).is_none());
}

#[test]
fn stateful_generation_matches_the_reforward() {
    let (_, _, m) = tiny(Kind::D, 7);
    let prompts: Vec<Vec<u32>> = vec![b"Animals:".iter().map(|&x| x as u32).collect(), b"Hi".iter().map(|&x| x as u32).collect(), b"Tools: saw,".iter().map(|&x| x as u32).collect(), b"Hi".iter().map(|&x| x as u32).collect()];
    let mut greedy = |_r: usize, l: &Tensor| -> candle_core::Result<u32> { l.argmax(0)?.to_scalar::<u32>() };
    let got = generate(&m, &CPU, &prompts, 16, 3, &mut greedy, &|_: &[u32]| false).unwrap();
    let o = crate::probe::GenOpts { max_new: 16, temperature: 0.0, top_k: 1, batch_tokens: 256 };
    let mut rngs: Vec<ava_data::rng::SplitMix64> = (0..prompts.len()).map(|i| ava_data::rng::SplitMix64::new(i as u64)).collect();
    let want = crate::probe::generate(&m, &CPU, &prompts, &o, &mut rngs, &|_: &[u32]| false).unwrap();
    assert_eq!(got, want);
    assert_eq!(got[1], got[3]);
}

#[test]
fn kinds_without_a_batched_state_are_refused() {
    for kind in [Kind::F, Kind::G, Kind::R] {
        let (_, _, m) = tiny(kind, 1);
        assert!(m.state(1).is_err());
    }
    let (_, _, m) = tiny(Kind::T, 1);
    let mut st = m.state(1).unwrap();
    let toks = tokens(1, 64, 258, 3);
    in_pieces(&m, &mut st, &toks, 1, 64, &[64]);
    assert!(m.forward_with_state(&Tensor::from_vec(vec![1u32], (1, 1), &CPU).unwrap(), &mut st).is_err(), "T past its position table");
}
