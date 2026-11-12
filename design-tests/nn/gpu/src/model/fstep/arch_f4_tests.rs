// TESTING / DESIGN ONLY — tests of src/model/fstep/arch_f4.rs (f4ext: v51 P --dendrite, DA --dopamine and Y --rhythm in
// the fused one-token talking step, and so in the chat app's CB2 live learning), CPU only.
//
// Pre-set bar (written before the first run, runs/v52/f4ext/report.md): the owner's: every step's logits within a
// relative error of 1e-5 of the plain State step's (max |fused - plain| / max |plain| per step, f32), the CB2 table's
// learning the plain path's (`forward_with_state` + `cb_learn`: its values, scale and update count, and each step's
// nats), and every spike or winner flip counted and reported (asserted 0: a flip moves the State off the plain path's).
// The step aims at arch_f3_tests.rs's bit for bit; every comparison also counts the values whose bits differ, and each
// case prints one "F4EXT" line with all the counts (run with --nocapture).
// Cases: Ava 0.4's recipe as the v52 Stage 1c brains have it (B': + --lif-reset-grad; d128, 3 blocks, the engram in block
// 1, vocabulary 293) + P, + DA, + Y, + Y with resets (r near 1/2, so the time restarts), + TH (already fused, the control),
// + DA M, + Y M, + P DA Y, + P DA Y TH CB 4096, and the kept set N S M R Pr CB with P DA Y; the parts' weights that start at
// 0 moved off it (build_awake). Over 300 tokens with the State carried: fused every step, taking turns, after a batched
// feed, with the chat app's part signals on; int8 on (the chat app's start) and f32 (`set_fast4`, int8 off). Each part
// changes the logits (switched off one after another). The traced step (the viewer) on both paths: spikes, winners and
// units per block, and the v51 signals (surprise, rhythm phases). The CB2 live session (`fast4_live_step`, as ava_live
// --learn cb2 runs it): 80 tokens read, 40 spoken greedily, one with no target, against the State path on the same
// tokens, int8 on and off. Trained brains: `f4ext_trained` (ignored; AVA_F4EXT_MODELS).

use crate::arch::Arch;
use crate::model::{Config, Kind, Model};
use crate::state::tests::{build_awake, tokens};
use crate::state::State;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;

const AVA04: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
                     --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1 --lif-reset-grad";
const V: usize = 293;
const CPU: Device = Device::Cpu;
const READ: usize = 80;
const SAY: usize = 40;

fn vals(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn cfg(extra: &str) -> Config {
    Config::with_kind(Kind::D, V, 128, 3, 64).with_engram(1).unwrap().with_arch(Arch::from_flags(&format!("{AVA04} {extra}")).unwrap()).unwrap()
}

/// R's letter table (arch_f3_tests.rs's).
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

/// Every variable named `b{i}.{tail}` (i in `blocks`), or `name` itself when `blocks` is empty, times `f` plus `add`.
fn scale_var(vm: &VarMap, name: &str, f: f32, add: f32) {
    let data = vm.data().lock().unwrap();
    let var = &data[name];
    let v: Vec<f32> = vals(var.as_tensor()).iter().map(|x| x * f + add).collect();
    var.set(&Tensor::from_vec(v, var.shape(), &CPU).unwrap()).unwrap();
}

fn set_all(vm: &VarMap, blocks: std::ops::Range<usize>, tail: &str, v: f32) {
    for i in blocks {
        scale_var(vm, &format!("b{i}.{tail}"), 0.0, v);
    }
}

/// A case: its flags, and " (resets)" for Y with resets (the reset logit's bias near 0 and its weights 20 times larger,
/// so r is near 1/2 and depends on what she reads).
fn build_vm(case: &str, seed: u64) -> (VarMap, Model) {
    letters();
    let resets = case.ends_with(" (resets)");
    let extra = case.trim_end_matches(" (resets)");
    let c = cfg(extra);
    let (vm, m) = build_awake(c, seed);
    if resets {
        scale_var(&vm, "v51.rhythm.wr", 20.0, 0.0);
        scale_var(&vm, "v51.rhythm.br", 0.0, 0.25);
    }
    if c.arch.v51.prune > 0.0 {
        // Pr: half the units of every block cut (a different half per block), as arch_f3_tests.rs does
        let h = c.hidden();
        let data = vm.data().lock().unwrap();
        for i in 0..c.layers {
            let al: Vec<f32> = (0..h).map(|j| if (j + i) % 2 == 0 { 1.0 } else { 0.0 }).collect();
            data[&format!("b{i}.v51.alive")].set(&Tensor::from_vec(al, h, &CPU).unwrap()).unwrap();
        }
        drop(data);
        m.v51_schedule(0, 0);
    }
    (vm, m)
}

/// The chat app's part signals as bits (arch_f3_tests.rs's).
fn sig_bits(p: &crate::v51b::PartSignals) -> Vec<u32> {
    let opt = |o: Option<f32>| o.map_or(vec![0], |v| vec![1, v.to_bits()]);
    let mut b: Vec<u32> = p.th.iter().flat_map(|g| [g[0].to_bits(), g[1].to_bits()]).collect();
    b.push(u32::MAX);
    b.extend(opt(p.cb));
    b.extend(opt(p.cb_online));
    b.extend(p.nm.map_or(vec![0], |n| vec![1, n[0].to_bits(), n[1].to_bits(), n[2].to_bits()]));
    b
}

/// One token's traced activity per block: spikes, winners, units; and the v51 signals (surprise, rhythm phases).
type Traced = (Vec<[Vec<f32>; 3]>, Vec<f32>);

#[derive(Default)]
struct Run {
    logits: Vec<Vec<f32>>,
    state: Vec<Vec<f32>>,
    counts: Vec<usize>,
    hist: Vec<Vec<u32>>,
    fused: u64,
    sigs: Vec<Vec<u32>>,
    traced: Vec<Traced>,
}

/// A fresh stream: `feed` tokens in one batched feed (plain), then each token stepped (traced when `traced`), step i fused
/// when `fused(i)`; the part signals on (cleared before every step) when `sig`.
fn run(m: &Model, toks: &[u32], feed: usize, sig: bool, traced: bool, fused: &dyn Fn(usize) -> bool) -> Run {
    let mut s = m.stream().unwrap();
    let ran = super::super::RAN.with(|r| r.get());
    let mut r = Run::default();
    if feed > 0 {
        m.set_fused(false);
        r.logits.push(vals(&m.feed(&mut s, &toks[..feed]).unwrap()));
    }
    for (i, &t) in toks.iter().enumerate().skip(feed) {
        m.set_fused(fused(i));
        m.set_part_signals(sig);
        if traced {
            let (l, tr) = m.step_trace(&mut s, t).unwrap();
            r.logits.push(vals(&l));
            let layers = tr.layers.iter().map(|ly| [vals(&ly.spikes), vals(&ly.winners), vals(&ly.hidden)]).collect();
            let v = tr.v51.map(|v| v.mood.iter().chain(&v.rhythm).copied().chain(v.surprise).collect()).unwrap_or_default();
            r.traced.push((layers, v));
        } else {
            r.logits.push(vals(&m.step(&mut s, t).unwrap()));
        }
        if sig {
            r.sigs.push(sig_bits(&m.part_signals()));
        }
    }
    m.set_fused(false);
    m.set_part_signals(false);
    let (st, counts, hist) = super::super::state_tensors(Model::stream_state(&s).unwrap());
    r.state = st.iter().map(vals).collect();
    (r.counts, r.hist) = (counts, hist);
    r.fused = super::super::RAN.with(|x| x.get()) - ran;
    r
}

/// How far `got` is from `want`: values whose bits differ and the largest relative error (max |got - want| / max |want|
/// per row, the largest over the rows).
#[derive(Default, Clone, Copy, Debug)]
struct Gap {
    off: usize,
    rel: f64,
}

impl Gap {
    fn add(&mut self, want: &[f32], got: &[f32]) {
        assert_eq!(want.len(), got.len(), "sizes");
        let mut dmax = 0f64;
        let mut amax = 0f64;
        for (&a, &b) in want.iter().zip(got) {
            if a.to_bits() != b.to_bits() {
                self.off += 1;
            }
            dmax = dmax.max((a as f64 - b as f64).abs());
            amax = amax.max((a as f64).abs());
        }
        let rel = if dmax == 0.0 { 0.0 } else { dmax / amax.max(1e-30) };
        self.rel = self.rel.max(if rel.is_nan() { f64::INFINITY } else { rel });
    }
}

/// The comparison of two runs: logits, State, part signals, and (traced) the spike, winner and unit flips.
#[derive(Default, Debug)]
struct Cmp {
    logits: Gap,
    state: Gap,
    sigs_off: usize,
    spike_flips: usize,
    winner_flips: usize,
    units: Gap,
    v51: Gap,
}

fn compare(name: &str, want: &Run, got: &Run) -> Cmp {
    assert_eq!(want.logits.len(), got.logits.len(), "{name}: steps");
    let mut c = Cmp::default();
    for (a, b) in want.logits.iter().zip(&got.logits) {
        c.logits.add(a, b);
    }
    assert_eq!(want.state.len(), got.state.len(), "{name}: State tensors");
    for (a, b) in want.state.iter().zip(&got.state) {
        c.state.add(a, b);
    }
    assert_eq!((&want.counts, &want.hist), (&got.counts, &got.hist), "{name}: tokens read, engram slots, history");
    assert_eq!(want.sigs.len(), got.sigs.len(), "{name}: part signals");
    c.sigs_off = want.sigs.iter().zip(&got.sigs).filter(|(a, b)| a != b).count();
    assert_eq!(want.traced.len(), got.traced.len(), "{name}: traced steps");
    for ((wl, wv), (gl, gv)) in want.traced.iter().zip(&got.traced) {
        assert_eq!(wl.len(), gl.len(), "{name}: traced blocks");
        for (a, b) in wl.iter().zip(gl) {
            c.spike_flips += a[0].iter().zip(&b[0]).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            c.winner_flips += a[1].iter().zip(&b[1]).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            c.units.add(&a[2], &b[2]);
        }
        c.v51.add(wv, gv);
    }
    c
}

/// The owner's bar on a comparison (and its F4EXT line).
fn bar(name: &str, c: &Cmp) {
    println!(
        "F4EXT {name}: logits {} values off, max rel {:.1e}; State {} off, max rel {:.1e}; part signals {} steps off; spike flips {}, winner flips {}; units {} off; v51 signals {} off",
        c.logits.off, c.logits.rel, c.state.off, c.state.rel, c.sigs_off, c.spike_flips, c.winner_flips, c.units.off, c.v51.off
    );
    assert!(c.logits.rel < 1e-5, "{name}: logits' relative error {:.2e}", c.logits.rel);
    assert!(c.state.rel < 1e-5, "{name}: the State's relative error {:.2e}", c.state.rel);
    assert_eq!((c.spike_flips, c.winner_flips), (0, 0), "{name}: spike and winner flips");
    assert!(c.units.rel < 1e-5 && c.v51.rel < 1e-5, "{name}: units {:.2e}, v51 signals {:.2e}", c.units.rel, c.v51.rel);
    assert_eq!(c.sigs_off, 0, "{name}: the part signals");
}

const CASES: [&str; 11] = [
    "--dendrite",
    "--dopamine",
    "--rhythm",
    "--rhythm (resets)",
    "--thalamus",
    "--dopamine --mood",
    "--rhythm --mood",
    "--dendrite --dopamine --rhythm",
    "--dendrite --dopamine --rhythm --thalamus --cerebellum-head 4096",
    "--neuromod 3 --stp --mood --letters --prune --dendrite --dopamine --rhythm --cerebellum-head 4096",
    "--neuromod 3 --stp --mood --letters --thalamus --prune --dendrite --dopamine --rhythm --cerebellum-head 4096",
];

/// The fused step against the plain State step, int8 on and f32 (`set_fast4`), for every case.
#[test]
fn f4ext_dendrite_dopamine_rhythm_in_the_fused_talking_step() {
    let _steady = crate::thread_env::steady();
    for (k, case) in CASES.iter().enumerate() {
        let (vm, m) = build_vm(case, 270 + k as u64);
        let toks = tokens(1, 300, V, 320 + k as u64);
        for int8 in [true, false] {
            m.set_int8(int8).unwrap();
            m.set_fast4(!int8);
            assert_eq!(m.fused_blocker(), None, "{case} (int8 {int8}): the fused talking step takes it");
            let sigs: &[bool] = if case.contains("thalamus") || case.contains("neuromod") || case.contains("cerebellum") { &[false, true] } else { &[false] };
            for &sig in sigs {
                let tag = format!("{case} (int8 {int8}, signals {sig})");
                let want = run(&m, &toks, 0, sig, false, &|_| false);
                assert_eq!(want.fused, 0, "{tag}: the plain run ran plain");
                let got = run(&m, &toks, 0, sig, false, &|_| true);
                assert_eq!(got.fused, toks.len() as u64, "{tag}: every step ran fused");
                bar(&tag, &compare(&tag, &want, &got));
                let turns = run(&m, &toks, 0, sig, false, &|i| i % 3 != 1);
                assert_eq!(turns.fused, 200, "{tag}: taking turns");
                bar(&format!("{tag}, taking turns"), &compare(&tag, &want, &turns));
                let fw = run(&m, &toks, 37, sig, false, &|_| false);
                let fg = run(&m, &toks, 37, sig, false, &|_| true);
                assert_eq!(fg.fused, (toks.len() - 37) as u64, "{tag}: after a feed, every step fused");
                bar(&format!("{tag}, after a batched feed"), &compare(&tag, &fw, &fg));
            }
            // the traced step (the chat app's viewer): spikes, winners, units, the v51 signals
            let tag = format!("{case} (int8 {int8}, traced)");
            let want = run(&m, &toks[..120], 0, false, true, &|_| false);
            let got = run(&m, &toks[..120], 0, false, true, &|_| true);
            assert_eq!((want.fused, got.fused), (0, 120), "{tag}: plain, then fused");
            bar(&tag, &compare(&tag, &want, &got));
        }
        m.set_fast4(false);
        // the parts moved what the State carries: Y's times off 0 (and, with resets, restarted), DA's surprise
        m.set_int8(true).unwrap();
        let mut s = m.stream().unwrap();
        m.set_fused(true);
        for &t in &toks[..50] {
            m.step(&mut s, t).unwrap();
        }
        m.set_fused(false);
        let st: &State = Model::stream_state(&s).unwrap();
        let o = m.cfg.arch.v51;
        if o.rhythm {
            let a = vals(st.v51.rh.as_ref().expect("Y's times"));
            let full = a.iter().all(|&x| x > 45.0);
            assert!(a.len() == 2 && a.iter().all(|x| x.is_finite() && *x > 0.0), "{case}: Y's times {a:?}");
            assert_eq!(full, !case.ends_with("(resets)"), "{case}: Y's times {a:?} (50 tokens: about 50 without resets)");
        }
        if o.surprise() {
            let sp = vals(st.v51.last_s.as_ref().expect("the surprise"));
            assert!(sp[0] > 0.0 && sp[0].is_finite(), "{case}: the surprise {sp:?}");
        }
        // each new part changes the logits (so the match is not the fused step leaving it out): switched off one after
        // another (P: B 0; DA: w and c 0; Y: both gates' weights 0), the plain step's logits move at most of the 300 steps
        let mut last = run(&m, &toks, 0, false, false, &|_| false).logits;
        for (on, what) in [(o.dendrite, "P"), (o.dopamine, "DA"), (o.rhythm, "Y")] {
            if !on {
                continue;
            }
            m.set_int8(false).unwrap();
            match what {
                "P" => set_all(&vm, 0..3, "v51.dend_b", 0.0),
                "DA" => {
                    set_all(&vm, 1..3, "v51.da_w", 0.0);
                    set_all(&vm, 1..3, "v51.da_c", 0.0);
                }
                _ => {
                    set_all(&vm, 0..3, "v51.rh_w", 0.0);
                    set_all(&vm, 0..3, "v51.rh_r", 0.0);
                }
            }
            m.set_int8(true).unwrap();
            let off = run(&m, &toks, 0, false, false, &|_| false).logits;
            let moved = off.iter().zip(&last).filter(|(a, b)| a != b).count();
            println!("F4EXT {case}: {what} off moves {moved} of 300 steps' logits");
            assert!(moved > 250, "{case}: {what} changed only {moved} of 300 steps' logits");
            last = off;
        }
        m.set_int8(false).unwrap();
    }
}

// ------------------------------------------------------------------------------------------- the CB2 live session

/// One live session's record: the tokens and targets, each token's logits and nats, the spikes the State keeps after
/// each token (per block), the cerebellum's signals, then the final State and the online table.
#[derive(Default)]
struct Live {
    steps: Vec<(u32, Option<u32>)>,
    logits: Vec<Vec<f32>>,
    nats: Vec<f64>,
    spikes: Vec<Vec<Vec<f32>>>,
    sig: Vec<(Option<u32>, Option<u32>)>,
    state: Vec<Vec<f32>>,
    counts: (Vec<usize>, Vec<Vec<u32>>),
    table: (Vec<f32>, f64, usize),
}

fn argmax(v: &[f32]) -> u32 {
    let mut b = 0;
    for (j, &x) in v.iter().enumerate() {
        if x > v[b] {
            b = j;
        }
    }
    b as u32
}

fn spikes_of(st: &State) -> Vec<Vec<f32>> {
    st.lif.iter().map(|l| vals(&l.as_ref().expect("LIF state").s)).collect()
}

fn cb_sig(m: &Model) -> (Option<u32>, Option<u32>) {
    let s = m.part_signals();
    (s.cb.map(f32::to_bits), s.cb_online.map(f32::to_bits))
}

fn cb_begin(m: &Model, rate: f32) {
    m.cb_online_begin(1, rate).unwrap();
    m.cb_online_set_lr(0, rate);
    m.cb_online_set_half_life(0, 256.0).unwrap();
}

fn cb_end(m: &Model, st: &State, mut o: Live) -> Live {
    let (ts, seen, hist) = super::super::state_tensors(st);
    o.state = ts.iter().map(vals).collect();
    o.counts = (seen, hist);
    o.table = m.cb_online_with(|t| (t.raw(0).0.to_vec(), t.raw(0).1, t.updates[0])).expect("an online table");
    m.cb_online_end();
    o
}

/// The chat app's CB2 session (`fast4_live_step`, CB2 alone): READ tokens of `text`, SAY drawn greedily, one more with
/// no target.
fn live(m: &Model, text: &[u32], rate: f32) -> Live {
    cb_begin(m, rate);
    let mut st = m.state(1).unwrap();
    let mut o = Live::default();
    let mut tok = text[0];
    for p in 0..READ + SAY + 1 {
        let mut tgt = |lg: &[f32]| -> candle_core::Result<Option<u32>> {
            Ok(if p + 1 < READ {
                Some(text[p + 1])
            } else if p < READ + SAY {
                Some(argmax(lg))
            } else {
                None
            })
        };
        let s = m.fast4_live_step(&mut st, tok, 0.0, true, &mut tgt).unwrap().unwrap_or_else(|| panic!("the live step runs: {:?}", m.fast4_live_blocker()));
        assert!(s.trace.as_ref().is_some_and(|t| t.layers.len() == m.blocks.len()), "a record of every block");
        assert!(s.eprop_rms.is_none(), "CB2 alone");
        o.steps.push((tok, s.next));
        o.logits.push(vals(&s.logits));
        o.nats.extend(s.cb.map(|c| c.0));
        o.spikes.push(spikes_of(&st));
        o.sig.push(cb_sig(m));
        assert_eq!(s.cb.is_some(), s.next.is_some());
        if let Some(n) = s.next {
            tok = n;
        }
    }
    cb_end(m, &st, o)
}

/// The State path on the live session's own tokens and targets: `forward_with_state`, then `cb_learn`.
fn reference(m: &Model, steps: &[(u32, Option<u32>)], rate: f32) -> Live {
    cb_begin(m, rate);
    let mut st = m.state(1).unwrap();
    let mut o = Live::default();
    for &(t, y) in steps {
        let lg = m.forward_with_state(&Tensor::new(&[[t]], &CPU).unwrap(), &mut st).unwrap();
        o.steps.push((t, y));
        o.logits.push(vals(&lg));
        o.spikes.push(spikes_of(&st));
        o.sig.push(cb_sig(m));
        if let Some(y) = y {
            o.nats.push(m.cb_learn(&lg, &[y]).unwrap());
        }
        st.detach_all();
    }
    cb_end(m, &st, o)
}

/// The live session against the State path: the owner's bar, and its F4EXT line. Returns the table's values that differ.
fn live_check(name: &str, m: &Model, text: &[u32], rate: f32) -> usize {
    let got = live(m, text, rate);
    let want = reference(m, &got.steps, rate);
    assert_eq!(got.steps.len(), READ + SAY + 1, "{name}");
    let (mut lg, mut st, mut tab, mut nats) = (Gap::default(), Gap::default(), Gap::default(), Gap::default());
    for (a, b) in want.logits.iter().zip(&got.logits) {
        lg.add(a, b);
    }
    for (a, b) in want.state.iter().zip(&got.state) {
        st.add(a, b);
    }
    tab.add(&want.table.0, &got.table.0);
    let (wn, gn): (Vec<f32>, Vec<f32>) = (want.nats.iter().map(|&x| x as f32).collect(), got.nats.iter().map(|&x| x as f32).collect());
    nats.add(&wn, &gn);
    let nats_off = want.nats.iter().zip(&got.nats).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let mut flips = 0;
    for (a, b) in want.spikes.iter().zip(&got.spikes) {
        for (x, y) in a.iter().zip(b) {
            flips += x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        }
    }
    let sig_off = want.sig.iter().zip(&got.sig).filter(|(a, b)| a != b).count();
    println!(
        "F4EXT live CB2 {name}: logits {} off, max rel {:.1e}; nats {nats_off} off, max rel {:.1e}; table {} of {} off, max rel {:.1e}, scale {} / {}, updates {} / {}; State {} off; spike flips {flips}; cb signals {sig_off} steps off",
        lg.off, lg.rel, nats.rel, tab.off, got.table.0.len(), tab.rel, want.table.1, got.table.1, want.table.2, got.table.2, st.off
    );
    assert!(lg.rel < 1e-5, "{name}: logits' relative error {:.2e}", lg.rel);
    assert!(nats.rel < 1e-5, "{name}: the nats' relative error {:.2e}", nats.rel);
    assert!(tab.rel < 1e-5, "{name}: the table's relative error {:.2e}", tab.rel);
    assert_eq!((want.table.1.to_bits(), want.table.2), (got.table.1.to_bits(), got.table.2), "{name}: the table's scale and update count");
    assert_eq!(got.table.2, READ + SAY, "{name}: one update per target");
    assert!(got.table.0.iter().any(|&x| x != 0.0), "{name}: the table learned");
    assert!(st.rel < 1e-5, "{name}: the State's relative error {:.2e}", st.rel);
    assert_eq!(flips, 0, "{name}: spike flips");
    assert_eq!(sig_off, 0, "{name}: the cerebellum's signals");
    assert_eq!(want.counts, got.counts, "{name}: counts, history");
    tab.off
}

/// The chat app's CB2 live learning with P, DA, Y (and TH, and all of them) against the State path's learning, lean
/// models as ava_live loads them, int8 on (the double-click start) and off.
#[test]
fn f4ext_cb2_live_learning_with_dendrite_dopamine_rhythm() {
    let _steady = crate::thread_env::steady();
    let text = tokens(1, READ + 1, V, 541);
    let cases = ["--dendrite", "--dopamine", "--rhythm", "--rhythm (resets)", "--thalamus", "--dendrite --dopamine --rhythm --thalamus", "--neuromod 3 --stp --mood --letters --prune --dendrite --dopamine --rhythm"];
    for (k, case) in cases.iter().enumerate() {
        let (vm, _) = build_vm(&format!("{} --cerebellum-head 512{}", case.trim_end_matches(" (resets)"), if case.ends_with(" (resets)") { " (resets)" } else { "" }), 550 + k as u64);
        let c = cfg(&format!("{} --cerebellum-head 512", case.trim_end_matches(" (resets)")));
        let m = crate::ckpt::lean(c, &vm, &CPU).unwrap();
        m.set_track_activity(false);
        m.set_part_signals(true);
        m.set_fast4(true);
        for int8 in [true, false] {
            m.set_int8(int8).unwrap();
            assert_eq!(m.cb_online_with(|_| ()), None, "no table yet");
            cb_begin(&m, 0.05);
            assert_eq!(m.fast4_live_blocker(), None, "{case} (int8 {int8}): live learning takes it");
            m.cb_online_end();
            live_check(&format!("{case} (int8 {int8})"), &m, &text, 0.05);
        }
        m.set_int8(false).unwrap();
    }
}

// ------------------------------------------------------------------------------------------- trained brains

/// Trained brains (not a pre-set check of its own; `... f4ext_trained -- --ignored --nocapture`, AVA_F4EXT_MODELS =
/// comma-separated model.safetensors paths with their arch.txt beside them): with int8 on, the fused step against the
/// plain State step over 300 tokens (and traced over 120: spike and winner flips), then the chat app's CB2 session as
/// ava_live starts it (the brain plus a zero cerebellum head of 12,288 units, rate 0.3 / mean |E_v|², half-life 256)
/// against the State path. Prints counts only (no speed, no text).
#[test]
#[ignore]
fn f4ext_trained() {
    let Ok(list) = std::env::var("AVA_F4EXT_MODELS") else { return };
    let _steady = crate::thread_env::steady();
    for path in list.split(',').filter(|p| !p.is_empty()) {
        let p = std::path::Path::new(path);
        let vocab = crate::ckpt::read(p).unwrap()["tok_emb"].dims()[0];
        crate::v51::set_letters_from_bpe(std::path::Path::new(crate::DEFAULT_BPE), vocab).unwrap();
        {
            let (_, _, m) = crate::ckpt::load_auto_lean(p, Kind::D, 4096, &CPU).unwrap();
            m.set_track_activity(false);
            m.set_int8(true).unwrap();
            assert_eq!(m.fused_blocker(), None, "{path}: the fused talking step takes it");
            let toks = tokens(1, 300, vocab, 77);
            let want = run(&m, &toks, 0, true, false, &|_| false);
            let got = run(&m, &toks, 0, true, false, &|_| true);
            assert_eq!((want.fused, got.fused), (0, toks.len() as u64), "{path}: plain, then every step fused");
            bar(&format!("trained {path}"), &compare(path, &want, &got));
            let want = run(&m, &toks[..120], 0, false, true, &|_| false);
            let got = run(&m, &toks[..120], 0, false, true, &|_| true);
            bar(&format!("trained {path}, traced"), &compare(path, &want, &got));
            m.set_int8(false).unwrap();
        }
        // CB2 as ava_live loads and starts it (load_plus_cb, learn_setup)
        let t = crate::ckpt::read(p).unwrap();
        let cfg0 = crate::ckpt::detect(&t, Kind::D, 4096).unwrap();
        let (ar0, _) = crate::ckpt::arch_beside(p).unwrap().expect("arch.txt beside the brain");
        let mut ar = Arch::from_flags(&format!("{} --cerebellum-head 12288", ar0.flags())).unwrap();
        ar.top = t.contains_key("top_emb");
        let c = cfg0.with_arch(ar).unwrap();
        drop(t);
        let vm = VarMap::new();
        let _build = Model::new(c, candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
        let (_, kept) = crate::ckpt::init_from(&vm, p).unwrap();
        assert_eq!(kept, ["cbh.wo"], "{path}: only the cerebellum's read-out is new");
        let e = vm.data().lock().unwrap()["tok_emb"].as_tensor().clone();
        let emb_sq = e.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() as f64 / e.dim(0).unwrap() as f64;
        let rate = (0.3 * (1.0 / emb_sq)) as f32;
        let m = crate::ckpt::lean(c, &vm, &CPU).unwrap();
        m.set_track_activity(false);
        m.set_part_signals(true);
        m.set_fast4(true);
        m.set_int8(true).unwrap();
        let text = tokens(1, READ + 1, vocab, 79);
        live_check(&format!("trained {path} (int8)"), &m, &text, rate);
        m.set_int8(false).unwrap();
    }
}
