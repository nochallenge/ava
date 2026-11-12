// TESTING / DESIGN ONLY — tests of src/model/fstep/arch_f3.rs (v51 fast3: S --stp, M --mood, Pr --prune and CB
// --cerebellum-head in the fused talking step), CPU.
//
// Pre-set bar (arch_tests.rs's and arch_f2_tests.rs's): bit for bit. Over 300 tokens with the State carried, every step's
// logits and the final State (state_tensors: LIF, fast weights, engram, and now each block's synapse (u, x) and v51's
// stream, valid flag, surprise and mood) of the fused step equal forward_with_state's with int8 on, for Ava 0.4's recipe
// (d128, 3 blocks, the engram in block 1, vocabulary 293) + S, + M, + Pr, + S M, + N S M R Pr, + N S M R TH Pr, + CB 4096,
// + N S M R Pr CB (the kept set) and + N S M R TH Pr CB; the parts' weights that start at 0 moved off it (build_awake),
// half the units cut for Pr (a different half per block); with the two steps taking turns, after a batched feed, and with
// the chat app's part signals on (cleared before every step; TH's gates, CB's correction size, the midbrain: the same bits
// after every token) where TH, N or CB is in. Each part changes the logits (switched off one after another). The traced
// step (the chat app's glow) gives the same logits, spikes (after the synapse), winners, units and v51 signals (mood,
// surprise) on both paths. The other v51 options and CB's online table still keep the plain step.

use crate::arch::Arch;
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, tokens};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;

const AVA04: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
                     --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";
const V: usize = 293;

fn vals(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
}

fn cfg(extra: &str) -> Config {
    Config::with_kind(Kind::D, V, 128, 3, 64).with_engram(1).unwrap().with_arch(Arch::from_flags(&format!("{AVA04} {extra}")).unwrap()).unwrap()
}

fn letters() {
    let r = crate::testkit::unit_noise(V * 22, 5);
    let t: Vec<Vec<u8>> = (0..V)
        .map(|v| {
            let len = ((r[v * 22] + 1.0) * 10.5) as usize % 21;
            (0..len).map(|j| (((r[v * 22 + 1 + j] + 1.0) * 128.0) as usize).min(255) as u8).collect()
        })
        .collect();
    crate::v51::set_letters(t);
}

/// Pr: half the units of every block cut (a different half per block), as v51/tests.rs `tune` does.
fn cut_half(c: &Config, vm: &VarMap, m: &Model) {
    if c.arch.v51.prune <= 0.0 {
        return;
    }
    let h = c.hidden();
    let data = vm.data().lock().unwrap();
    for i in 0..c.layers {
        let al: Vec<f32> = (0..h).map(|j| if (j + i) % 2 == 0 { 1.0 } else { 0.0 }).collect();
        data[&format!("b{i}.v51.alive")].set(&Tensor::from_vec(al, h, &Device::Cpu).unwrap()).unwrap();
    }
    drop(data);
    m.v51_schedule(0, 0); // forget the cached unit counts
}

fn build(extra: &str, seed: u64) -> Model {
    build_vm(extra, seed).1
}

fn build_vm(extra: &str, seed: u64) -> (VarMap, Model) {
    letters();
    let c = cfg(extra);
    let (vm, m) = build_awake(c, seed);
    cut_half(&c, &vm, &m);
    (vm, m)
}

/// Sets every variable named `b{i}.{tail}` (i in `blocks`) to the constant `v`.
fn set_all(vm: &VarMap, blocks: std::ops::Range<usize>, tail: &str, v: f32) {
    let data = vm.data().lock().unwrap();
    for i in blocks {
        let var = &data[&format!("b{i}.{tail}")];
        let n = var.as_tensor().elem_count();
        var.set(&Tensor::from_vec(vec![v; n], var.shape(), &Device::Cpu).unwrap()).unwrap();
    }
}

type Run = (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<usize>, Vec<Vec<u32>>, u64, Vec<Vec<u32>>);

/// The chat app's part signals (TH's gates, CB's correction size and its online share, the midbrain) as bits.
fn sig_bits(p: &crate::v51b::PartSignals) -> Vec<u32> {
    let opt = |o: Option<f32>| o.map_or(vec![0], |v| vec![1, v.to_bits()]);
    let mut b: Vec<u32> = p.th.iter().flat_map(|g| [g[0].to_bits(), g[1].to_bits()]).collect();
    b.push(u32::MAX);
    b.extend(opt(p.cb));
    b.extend(opt(p.cb_online));
    b.extend(p.nm.map_or(vec![0], |n| vec![1, n[0].to_bits(), n[1].to_bits(), n[2].to_bits()]));
    b
}

/// A fresh stream: the first `feed` tokens in one batched feed (plain), then each token stepped, step i fused when
/// `fused(i)`; part signals on (cleared before every step) when `sig`. Returns every logit, the final State's tensors,
/// its counts and history, the fused steps run and the part signals after every step (empty without `sig`).
fn run(m: &Model, toks: &[u32], feed: usize, sig: bool, fused: &dyn Fn(usize) -> bool) -> Run {
    let mut s = m.stream().unwrap();
    let ran = super::super::RAN.with(|r| r.get());
    let (mut logits, mut sigs) = (Vec::new(), Vec::new());
    if feed > 0 {
        m.set_fused(false);
        logits.push(bits(&m.feed(&mut s, &toks[..feed]).unwrap()));
    }
    for (i, &t) in toks.iter().enumerate().skip(feed) {
        m.set_fused(fused(i));
        m.set_part_signals(sig);
        logits.push(bits(&m.step(&mut s, t).unwrap()));
        if sig {
            sigs.push(sig_bits(&m.part_signals()));
        }
    }
    m.set_fused(false);
    m.set_part_signals(false);
    let (st, counts, hist) = super::super::state_tensors(Model::stream_state(&s).unwrap());
    (logits, st.iter().map(bits).collect(), counts, hist, super::super::RAN.with(|r| r.get()) - ran, sigs)
}

fn same(name: &str, want: &Run, got: &Run) {
    assert_eq!(want.0.len(), got.0.len(), "{name}: steps");
    let bad = want.0.iter().zip(&got.0).position(|(a, b)| a != b);
    assert!(bad.is_none(), "{name}: logits differ first at step {bad:?}");
    assert_eq!(want.1.len(), got.1.len(), "{name}: State tensors");
    let bad = want.1.iter().zip(&got.1).position(|(a, b)| a != b);
    assert!(bad.is_none(), "{name}: State tensor {bad:?} differs");
    assert_eq!((&want.2, &want.3), (&got.2, &got.3), "{name}: tokens read, engram slots, history");
    let bad = want.5.iter().zip(&got.5).position(|(a, b)| a != b);
    assert!(want.5.len() == got.5.len() && bad.is_none(), "{name}: the part signals differ first after step {bad:?}");
}

const CASES: [&str; 9] = [
    "--stp",
    "--mood",
    "--prune",
    "--stp --mood",
    "--neuromod 3 --stp --mood --letters --prune",
    "--neuromod 3 --stp --mood --letters --thalamus --prune",
    "--cerebellum-head 4096",
    "--neuromod 3 --stp --mood --letters --prune --cerebellum-head 4096",
    "--neuromod 3 --stp --mood --letters --thalamus --prune --cerebellum-head 4096",
];

#[test]
fn stp_mood_prune_cb_in_the_fused_talking_step_bit_for_bit() {
    for (k, extra) in CASES.iter().enumerate() {
        let (vm, m) = build_vm(extra, 170 + k as u64);
        m.set_int8(true).unwrap();
        assert_eq!(m.fused_blocker(), None, "{extra}: the fused talking step takes it");
        let toks = tokens(1, 300, V, 220 + k as u64);
        let sigs: &[bool] = if extra.contains("thalamus") || extra.contains("neuromod") || extra.contains("cerebellum") { &[false, true] } else { &[false] };
        for &sig in sigs {
            let tag = format!("{extra} (signals {sig})");
            let want = run(&m, &toks, 0, sig, &|_| false);
            assert_eq!(want.4, 0, "{tag}: the plain run ran plain");
            let got = run(&m, &toks, 0, sig, &|_| true);
            assert_eq!(got.4, toks.len() as u64, "{tag}: every step ran fused");
            same(&tag, &want, &got);
            if sig && extra.contains("cerebellum") {
                // CB's correction size reaches the chat app after every token (cleared before each), the online share none
                let p = sig_bits(&{
                    let mut s = m.stream().unwrap();
                    m.set_fused(true);
                    m.set_part_signals(true);
                    m.step(&mut s, toks[0]).unwrap();
                    m.set_fused(false);
                    let p = m.part_signals();
                    m.set_part_signals(false);
                    assert!(p.cb.is_some_and(|r| r.is_finite() && r > 0.0) && p.cb_online.is_none(), "{tag}: CB's signal {:?}", p.cb);
                    p
                });
                assert_eq!(Some(&p), got.5.first(), "{tag}: the first token's signals");
            }
            let turns = run(&m, &toks, 0, sig, &|i| i % 3 != 1);
            assert_eq!(turns.4, 200, "{tag}: taking turns");
            same(&format!("{tag}, taking turns"), &want, &turns);
            // after a batched feed of 37 tokens (the State path's scan over them), each token stepped
            let fw = run(&m, &toks, 37, sig, &|_| false);
            let fg = run(&m, &toks, 37, sig, &|_| true);
            assert_eq!(fg.4, (toks.len() - 37) as u64, "{tag}: after a feed, every step fused");
            same(&format!("{tag}, after a batched feed"), &fw, &fg);
        }
        // the parts moved what the State carries: the synapses off (0, 1), the mood off 0
        let mut s = m.stream().unwrap();
        m.set_fused(true);
        for &t in &toks[..50] {
            m.step(&mut s, t).unwrap();
        }
        m.set_fused(false);
        let st = Model::stream_state(&s).unwrap();
        let o = m.cfg.arch.v51;
        if extra.contains("--stp") {
            let syn: Vec<(Vec<f32>, Vec<f32>)> = st.parts.iter().filter_map(|p| p.stp.as_ref()).map(|(u, x)| (vals(u), vals(x))).collect();
            assert_eq!(syn.len(), 3, "{extra}: a synapse per block");
            assert!(syn.iter().all(|(u, x)| u.iter().any(|&v| v != 0.0) && x.iter().any(|&v| v != 1.0)), "{extra}: the synapses moved");
        }
        if o.mood > 0 {
            let md = vals(st.v51.mood.as_ref().unwrap());
            assert!(md.len() == o.mood && md.iter().all(|v| v.is_finite() && *v != 0.0), "{extra}: the mood moved: {md:?}");
            let sp = vals(st.v51.last_s.as_ref().unwrap());
            assert!(sp[0] > 0.0 && sp[0].is_finite(), "{extra}: the surprise {sp:?}");
        }
        // each part changes the logits (so the match above is not the fused step leaving one out): switched off one
        // after another (S: mix 0; M: G 0; Pr: every unit alive; CB: W 0), the plain step's logits move at most of the 300
        // steps
        let mut last = run(&m, &toks, 0, false, &|_| false).0;
        let o = m.cfg.arch.v51;
        for (on, what) in [(extra.contains("--stp"), "S"), (o.mood > 0, "M"), (o.prune > 0.0, "Pr"), (extra.contains("cerebellum"), "CB")] {
            if !on {
                continue;
            }
            m.set_int8(false).unwrap();
            match what {
                "S" => set_all(&vm, 0..3, "parts.stp.mix", 0.0),
                "M" => set_all(&vm, 1..3, "v51.mood_g", 0.0),
                "CB" => {
                    let data = vm.data().lock().unwrap();
                    let w = &data["cbh.wo"];
                    w.set(&w.as_tensor().zeros_like().unwrap()).unwrap();
                }
                _ => {
                    set_all(&vm, 0..3, "v51.alive", 1.0);
                    m.v51_schedule(0, 0);
                }
            }
            m.set_int8(true).unwrap(); // the int8 copies of the weights made again
            let off = run(&m, &toks, 0, false, &|_| false).0;
            let moved = off.iter().zip(&last).filter(|(a, b)| a != b).count();
            assert!(moved > 250, "{extra}: {what} changed only {moved} of 300 steps' logits");
            last = off;
        }
        m.set_int8(false).unwrap();
    }
}

/// The traced step (step_trace, the chat app's glow) on the fused path: the same logits as the plain traced step, the
/// spikes after the synapse, the winners (never a cut unit) and units bit for bit, and the v51 signals (mood, surprise)
/// equal; the full kept set N S M R Pr CB.
#[test]
fn stp_mood_prune_cb_in_the_fused_traced_step() {
    let extra = "--neuromod 3 --stp --mood --letters --prune --cerebellum-head 4096";
    let m = build(extra, 191);
    m.set_int8(true).unwrap();
    assert_eq!(m.fused_blocker(), None);
    let toks = tokens(1, 80, V, 231);
    let h = m.cfg.hidden();
    let trace = |fused: bool| {
        let mut s = m.stream().unwrap();
        m.set_fused(fused);
        let ran = super::super::RAN.with(|r| r.get());
        let out: Vec<_> = toks.iter().map(|&t| m.step_trace(&mut s, t).unwrap()).collect();
        let n = super::super::RAN.with(|r| r.get()) - ran;
        m.set_fused(false);
        (out, n)
    };
    let (want, wn) = trace(false);
    let (got, gn) = trace(true);
    assert_eq!((wn, gn), (0, toks.len() as u64), "the traced steps ran plain, then fused");
    for (i, ((wl, wt), (gl, gt))) in want.iter().zip(&got).enumerate() {
        assert!(bits(wl) == bits(gl), "step {i}: logits differ");
        assert_eq!(wt.layers.len(), gt.layers.len());
        for (b, (a, c)) in wt.layers.iter().zip(&gt.layers).enumerate() {
            assert!(bits(&a.spikes) == bits(&c.spikes), "step {i} block {b}: spikes differ");
            let (wa, ca) = (bits(&a.winners), bits(&c.winners));
            assert!(wa == ca, "step {i} block {b}: winners differ");
            assert!(bits(&a.hidden) == bits(&c.hidden), "step {i} block {b}: units differ");
            let w = vals(&c.winners);
            assert!(w.iter().enumerate().all(|(j, &v)| v == 0.0 || (j + b) % 2 == 0), "step {i} block {b}: a cut unit won");
            assert!(w.len() == h && w.iter().filter(|&&v| v > 0.0).count() >= 1);
        }
        let (a, c) = (wt.v51.as_ref().expect("plain: v51 signals"), gt.v51.as_ref().expect("fused: v51 signals"));
        assert!(a.mood.iter().map(|v| v.to_bits()).eq(c.mood.iter().map(|v| v.to_bits())), "step {i}: the mood signal differs");
        assert_eq!(a.surprise.map(f32::to_bits), c.surprise.map(f32::to_bits), "step {i}: the surprise differs");
    }
    m.set_int8(false).unwrap();
}

/// The v51 options this step does not take still keep the plain step (the blocker says why).
#[test]
fn other_v51_options_keep_the_plain_step() {
    // f4ext: P, DA and Y run fused now (arch_f4_tests.rs); W still keeps the plain step, alone or beside them
    for extra in ["--columns 4 --stp", "--columns 4 --dendrite", "--columns 4 --mood --rhythm", "--columns 4 --dopamine --cerebellum-head 512", "--stp --delays 3"] {
        let m = build(extra, 201);
        m.set_int8(true).unwrap();
        assert!(m.fused_blocker().is_some(), "{extra}: still the plain step");
        m.set_int8(false).unwrap();
    }
    // CB's online table (the chat app's learning while she reads) keeps the plain step: nothing runs fused while it is on
    let m = build("--neuromod 3 --stp --mood --letters --prune --cerebellum-head 4096", 202);
    m.set_int8(true).unwrap();
    assert_eq!(m.fused_blocker(), None);
    m.cb_online_begin(1, 0.1).unwrap();
    assert!(m.fused_blocker().is_some(), "an online table: the plain step");
    let toks = tokens(1, 20, V, 7);
    assert_eq!(run(&m, &toks, 0, false, &|_| true).4, 0, "an online table: no fused step ran");
    m.cb_online_end();
    assert_eq!(m.fused_blocker(), None, "after the online table");
    m.set_int8(false).unwrap();
}

/// Talking speed on the CPU (not a check; `cargo test ... fast3_talk_speed -- --ignored --nocapture`): the State path
/// with int8 against the fused step, tokens/s, for B, S, M, S+M, N+S+M+R+Pr, CB 4096 and N+S+M+R+Pr+CB at the bake-off's
/// shape (d128, 4 blocks, Ava 0.4's recipe, vocabulary 4096; random weights moved off 0, half the units cut for Pr).
/// AVA_F3_TOKENS (200) and AVA_F3_REPS (5) set it; the thread count is the process's (the test build never changes it,
/// crate::thread_env): run it with RAYON_NUM_THREADS=N CANDLE_NUM_THREADS=N. Each mode's median over the reps, the modes interleaved.
#[test]
#[ignore]
fn fast3_talk_speed() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n, reps, th) = (env("AVA_F3_TOKENS", 200), env("AVA_F3_REPS", 5), candle_core::utils::get_num_threads());
    letters();
    let vocab = 4096;
    let full = "--neuromod 3 --stp --mood --letters --prune --cerebellum-head 4096";
    let arms = [("B", ""), ("S", "--stp"), ("M", "--mood"), ("S+M", "--stp --mood"), ("N+S+M+R+Pr", "--neuromod 3 --stp --mood --letters --prune"), ("CB", "--cerebellum-head 4096"), ("N+S+M+R+Pr+CB", full), ("B again", "")];
    let pick = std::env::var("AVA_F3_ARMS").ok();
    for (name, extra) in arms.into_iter().filter(|(n, _)| pick.as_ref().is_none_or(|p| p.split(',').any(|q| q == *n))) {
        let c = Config::with_kind(Kind::D, vocab, 128, 4, 256).with_arch(Arch::from_flags(&format!("{AVA04} {extra}")).unwrap()).unwrap();
        let (vm, m) = build_awake(c, 7);
        cut_half(&c, &vm, &m);
        m.set_int8(true).unwrap();
        assert_eq!(m.fused_blocker(), None, "{name}");
        let toks = tokens(1, n + 20, vocab, 9);
        let (mut plain, mut fused) = (Vec::new(), Vec::new());
        for r in 0..=reps {
            for f in [false, true] {
                m.set_fused(f);
                let mut s = m.stream().unwrap();
                for &t in &toks[..20] {
                    m.step(&mut s, t).unwrap();
                }
                let t0 = std::time::Instant::now();
                for &t in &toks[20..] {
                    m.step(&mut s, t).unwrap();
                }
                let tps = n as f64 / t0.elapsed().as_secs_f64();
                if r > 0 {
                    if f { fused.push(tps) } else { plain.push(tps) }
                }
            }
        }
        m.set_fused(false);
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let (p, f) = (med(&mut plain), med(&mut fused));
        println!("FAST3 {name}: State path int8 {p:.1} tok/s, fused int8 {f:.1} tok/s, x{:.2} ({th} threads, {n} tokens, {reps} reps)", f / p);
        if std::env::var("AVA_F3_PROF").is_ok_and(|v| v == "1") {
            // where a fused token's time goes: the int8 products and the step's sections (arch.rs SECTIONS), us per token
            m.set_fused(true);
            let mut s = m.stream().unwrap();
            for &t in &toks[..20] {
                m.step(&mut s, t).unwrap();
            }
            crate::int8::set_mm_prof(true);
            super::super::take_secs();
            for &t in &toks[20..] {
                m.step(&mut s, t).unwrap();
            }
            let (calls, ns) = crate::int8::mm_prof();
            let secs = super::super::take_secs();
            crate::int8::set_mm_prof(false);
            m.set_fused(false);
            let us = |x: u64| x as f64 / 1e3 / n as f64;
            let parts: Vec<String> = super::super::SECTIONS.iter().zip(secs).map(|(s, v)| format!("{s} {:.1}", us(v))).collect();
            println!("FAST3 {name} profile (us/token): int8 {:.1} ({} products/token); {}", us(ns), calls / n as u64, parts.join("; "));
        }
    }
}

/// Trained models (not a pre-set check; `... fast3_trained_bit_for_bit -- --ignored`, AVA_F3_MODELS = comma-separated
/// model.safetensors paths with their arch.txt): the fused step against forward_with_state with int8 on, over 300 tokens
/// of deterministic text (and the two taking turns): every logit and the final State bit for bit, and every step fused.
#[test]
#[ignore]
fn fast3_trained_bit_for_bit() {
    let Ok(list) = std::env::var("AVA_F3_MODELS") else { return };
    for path in list.split(',').filter(|p| !p.is_empty()) {
        let p = std::path::Path::new(path);
        let vocab = crate::ckpt::read(p).unwrap()["tok_emb"].dims()[0];
        crate::v51::set_letters_from_bpe(std::path::Path::new(crate::DEFAULT_BPE), vocab).unwrap();
        let (_, _, m) = crate::ckpt::load_auto_lean(p, Kind::D, 4096, &Device::Cpu).unwrap();
        m.set_int8(true).unwrap();
        assert_eq!(m.fused_blocker(), None, "{path}: the fused talking step takes it");
        let toks = tokens(1, 300, vocab, 77);
        let want = run(&m, &toks, 0, true, &|_| false);
        let got = run(&m, &toks, 0, true, &|_| true);
        assert_eq!((want.4, got.4), (0, toks.len() as u64), "{path}: plain, then every step fused");
        same(path, &want, &got);
        let turns = run(&m, &toks, 0, true, &|i| i % 3 != 1);
        same(&format!("{path}, taking turns"), &want, &turns);
        println!("FAST3 trained {path}: 300 tokens bit for bit ({} State tensors, the part signals too)", want.1.len());
        m.set_int8(false).unwrap();
    }
}
