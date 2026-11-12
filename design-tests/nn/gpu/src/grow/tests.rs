// TESTING / DESIGN ONLY — tests of src/grow.rs (A15 exact growth, grow_exact), CPU only.

use super::*;
use crate::arch::Arch;
use crate::ckpt::{load_auto, write_arch_beside};
use crate::model::{kwta_mask, kwta_mask_sorted, rmsnorm};
use crate::optim::{AdamW, Param};
use crate::state::tests::{build_awake, in_pieces, max_diff, one_pass, tokens};
use crate::testkit::{scratch, unit_noise};
use ava_data::numbers::VOCAB_GLANCE;

const CPU: Device = Device::Cpu;

/// Ava 0.4's block flags (runs/v42/run_wave3.sh's winners, the slow set and Wave 5). ON_BASE adds --glance (the
/// vocabulary below) and V42 adds --decay-gates (a training flag: `Model::decays_with(_, true)` in the learning test).
const WIN: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix --lif-init logspaced --lif-scale";
const SLOW: &str = "--gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";
const WAVE5: &str = "--resonate 16 --modulate --tem 4 --halting 3 --cerebellum 64 --workspace 4 --holo 8 --sync 4 --jepa 0.5 --diffusion 0.5 --critic 0.5 --confidence 0.5 --topo 0.5 --schema 0.5";
/// The Wave 2C memory parts (not in Ava 0.4's set; their read-outs are paths into the stream too).
const MEMORY: &str = "--eng-topk 4 --eng-temp --eng-ssmax --btsp 4 --pk-mem 16 --pk-topk 2 --dendrites 2 --stp --delays 3 --wm 4";

fn d_with(flags: &str, vocab: usize, d: usize, layers: usize) -> Config {
    let c = Config::with_kind(Kind::D, vocab, d, layers, 64);
    let c = if vocab == VOCAB_GLANCE { c.with_glance().unwrap() } else { c };
    c.with_arch(Arch::from_flags(flags).unwrap()).unwrap()
}

/// Ava 0.4 at d64 (head size 64), 4 blocks, the glance's vocabulary.
fn everything() -> Config {
    d_with(&format!("{WIN} {SLOW} {WAVE5}"), VOCAB_GLANCE, 64, 4)
}

fn plan(mlp_times: usize, deeper: usize, at: Option<usize>) -> Plan {
    Plan { mlp_times, split: Split::First, deeper, at, seed: 5 }
}

fn ids(b: usize, t: usize, vocab: usize, seed: u64) -> Tensor {
    Tensor::from_vec(tokens(b, t, vocab, seed), (b, t), &CPU).unwrap()
}

/// Grow and check: the relative logit gap on two rows of `t` tokens (asserted under TOL), printed with `name`.
fn grown_exact(name: &str, cfg: Config, p: &Plan, t: usize) -> (Model, Grown, f32) {
    let (vm, m) = build_awake(cfg, 3);
    let g = grow(&cfg, &vm, p, &CPU).unwrap_or_else(|e| panic!("{name}: {e}"));
    let x = ids(2, t, cfg.vocab, 7);
    let (abs, rel) = logit_gap(&m, &g.model, &x).unwrap();
    println!("A15 {name}: max |logit diff| {abs:.3e}, relative {rel:.3e}");
    assert!(rel <= TOL, "{name}: relative gap {rel:.3e} (abs {abs:.3e})");
    (m, g, rel)
}

/// Every tensor of the new blocks except READOUTS and PINNED moved by noise (0.1 x unit noise added), so the identity
/// cannot hang on some other weight happening to be 0.
fn stir_new_blocks(g: &Grown) {
    let data = g.varmap.data().lock().unwrap();
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    for (k, name) in names.into_iter().enumerate() {
        let Some((j, rest)) = block_of(name) else { continue };
        if !g.layout.fresh.contains(&j) || READOUTS.contains(&rest) || PINNED.iter().any(|(p, _)| *p == rest) {
            continue;
        }
        let v = &data[name];
        let z = Tensor::from_vec(unit_noise(v.elem_count(), 1000 + k as u64), v.shape(), &CPU).unwrap();
        v.set(&v.as_tensor().add(&z.affine(0.1, 0.0).unwrap()).unwrap()).unwrap();
    }
}

/// Whether this build can record a widened MLP (Config::hidden() reads --mlp-mult); else the refusal it gives.
fn width_record() -> Result<(), String> {
    layout(&Config::with_kind(Kind::D, 258, 32, 2, 64), &plan(2, 0, None)).map(|_| ())
}

#[test]
fn deeper_plain_d_is_exact_anywhere() {
    let cfg = Config::with_kind(Kind::D, 258, 32, 3, 64);
    for at in [0, 1, 2] {
        let (m, g, _) = grown_exact(&format!("plain D L3 + 2 after block {at}"), cfg, &plan(1, 2, Some(at)), 40);
        assert_eq!(g.model.cfg.layers, 5);
        // the new blocks' other weights do not matter
        stir_new_blocks(&g);
        let (_, rel) = logit_gap(&m, &g.model, &ids(2, 40, 258, 7)).unwrap();
        assert!(rel <= TOL, "stirred, after {at}: {rel:.3e}");
    }
}

#[test]
fn deeper_everything_on_is_exact() {
    // the cerebellum allows the top only; the workspace (block 2 of 4) stays in block 2 while the middle of 6 is 3
    let (m, g, _) = grown_exact("Ava 0.4 (WIN+SLOW+Wave 5+glance) d64 L4 + 2 at the top", everything(), &plan(1, 2, None), 48);
    assert_eq!(g.model.cfg.arch.mem.ws_layer, Some(2));
    assert!(g.model.cfg.arch.flags().contains("--ws-layer 2"), "{}", g.model.cfg.arch.flags());
    stir_new_blocks(&g);
    let (abs, rel) = logit_gap(&m, &g.model, &ids(2, 48, VOCAB_GLANCE, 7)).unwrap();
    println!("A15 Ava 0.4 + 2, new blocks stirred: {abs:.3e}, relative {rel:.3e}");
    assert!(rel <= TOL);
    // without the cerebellum, blocks can go in the middle: dense re-indexed, the workspace moves 2 -> 4
    let flags = format!("{WIN} {SLOW} {}", WAVE5.replace("--cerebellum 64 ", ""));
    let cfg = d_with(&flags, VOCAB_GLANCE, 64, 4);
    let (m, g, _) = grown_exact("Ava 0.4 without the cerebellum + 2 after block 1", cfg, &plan(1, 2, Some(1)), 48);
    assert_eq!(g.model.cfg.arch.mem.ws_layer, Some(4));
    stir_new_blocks(&g);
    assert!(logit_gap(&m, &g.model, &ids(2, 48, VOCAB_GLANCE, 7)).unwrap().1 <= TOL);
}

#[test]
fn deeper_with_the_memory_parts_and_feedback_is_exact() {
    // the engram (block 1) and the product-key memory (the last block) move with their blocks
    let cfg = Config::with_kind(Kind::D, 258, 32, 3, 64).with_engram(1).unwrap().with_arch(Arch::from_flags(MEMORY).unwrap()).unwrap();
    let (m, g, _) = grown_exact("D + engram + Wave 2C parts L3 + 2 after block 0", cfg, &plan(1, 2, Some(0)), 40);
    assert_eq!((g.model.cfg.engram, g.model.cfg.arch.mem.pk_block(5)), (Some(3), Some(4)));
    assert_eq!(g.model.cfg.arch.mem.pk_layer, None, "the last block is still the default");
    stir_new_blocks(&g);
    assert!(logit_gap(&m, &g.model, &ids(2, 40, 258, 7)).unwrap().1 <= TOL);
    let (_, g, _) = grown_exact("D + engram + Wave 2C parts L3 + 1 at the top", cfg, &plan(1, 1, None), 40);
    assert_eq!(g.model.cfg.arch.mem.pk_layer, Some(2), "pinned below the new top block");
    // O1: the window in chunks of 64 with every block's state carried (150 tokens = 3 chunks)
    let cfg = d_with("--feedback --residual dense --value-residual --gates channel --mem-momentum 0.5", 258, 32, 3);
    let (m, g, _) = grown_exact("D + feedback + dense + vres L3 + 2 after block 0", cfg, &plan(1, 2, Some(0)), 150);
    stir_new_blocks(&g);
    assert!(logit_gap(&m, &g.model, &ids(2, 150, 258, 7)).unwrap().1 <= TOL);
}

#[test]
fn deeper_works_for_every_kind_and_keeps_a_s_attention_pattern() {
    for kind in [Kind::S, Kind::T, Kind::F, Kind::G, Kind::R] {
        let (cfg, vm, m) = crate::testkit::tiny(kind, 4);
        let g = grow(&cfg, &vm, &plan(1, 1, Some(0)), &CPU).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        let (abs, rel) = logit_gap(&m, &g.model, &ids(2, 40, 258, 7)).unwrap();
        println!("A15 {} L2 + 1 after block 0: {abs:.3e}, relative {rel:.3e}", kind.name());
        assert!(rel <= TOL, "{kind:?}");
    }
    // A: attention in every 4th block counting back from the last (block 3 of 4)
    let (cfg, vm, m) = crate::testkit::tiny_with(Kind::A, 258, 32, 4, 4);
    let err = grow(&cfg, &vm, &plan(1, 1, Some(0)), &CPU).err().unwrap();
    assert!(err.contains("multiple of 4"), "{err}");
    let g = grow(&cfg, &vm, &plan(1, 4, Some(1)), &CPU).unwrap();
    assert_eq!((0..8).filter(|&i| g.model.cfg.attn_layer(i)).collect::<Vec<_>>(), vec![3, 7]);
    let (abs, rel) = logit_gap(&m, &g.model, &ids(2, 40, 258, 7)).unwrap();
    println!("A15 A L4 + 4 after block 1: {abs:.3e}, relative {rel:.3e}");
    assert!(rel <= TOL);
}

#[test]
fn what_cannot_be_exact_is_refused_with_the_reason() {
    let cfg = everything();
    let refuse = |p: Plan| layout(&cfg, &p).err().unwrap_or_else(|| panic!("{p:?} was not refused"));
    let e = refuse(plan(1, 1, Some(1)));
    assert!(e.contains("--cerebellum") && e.contains("seed of its block number") && e.contains("--at 3"), "{e}");
    assert!(refuse(plan(1, 0, None)).contains("nothing to grow"));
    assert!(refuse(plan(1, 0, Some(1))).contains("needs --deeper"));
    assert!(refuse(plan(1, 1, Some(4))).contains("0 <= L < 4"));
    assert!(refuse(plan(0, 1, None)).contains("C >= 1"));
    let (t, _, _) = crate::testkit::tiny(Kind::T, 1);
    let e = layout(&t, &plan(2, 0, None)).err().unwrap();
    assert!(e.contains("for model D"), "{e}");
    // grow_exact itself
    let e = run(&["--init".into(), "x".into(), "--out".into(), "y".into(), "--wider-d".into(), "2".into()]).err().unwrap();
    assert!(e.contains("not offered"), "{e}");
}

#[test]
fn the_talking_state_works_on_the_grown_model() {
    let (m, g, _) = grown_exact("Ava 0.4 + 2 (for the talking state)", everything(), &plan(1, 2, None), 48);
    let (b, t) = (2, 48);
    let toks = tokens(b, t, VOCAB_GLANCE, 9);
    let want = one_pass(&g.model, &toks, b, t);
    let top = want.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
    for pieces in [vec![48], vec![17, 14, 17], vec![1; 48]] {
        // the grown model talks as the source talks, piece by piece
        let (mut sg, mut ss) = (g.model.state(b).unwrap(), m.state(b).unwrap());
        let got = in_pieces(&g.model, &mut sg, &toks, b, t, &pieces);
        let src = in_pieces(&m, &mut ss, &toks, b, t, &pieces);
        let d = max_diff(&got, &src);
        // the state path's own gap to one pass (the same in both: a property of the flags, not of growth; the crate allows
        // 5e-4 for --tem at d32, wave5::tests::pieces_tol)
        let (own, own_src) = (max_diff(&got, &want), max_diff(&src, &one_pass(&m, &toks, b, t)));
        println!("A15 talking state {pieces:?}: grown vs source in pieces {d:.3e} (relative {:.3e}); pieces vs one pass: grown {own:.3e}, source {own_src:.3e}", d / top);
        assert!(d / top <= TOL, "{pieces:?}: {d}");
        if pieces.len() == 1 {
            assert_eq!(own, 0.0, "a fresh state reads one piece as one pass");
        }
        assert_eq!(sg.seen, vec![t; b]);
    }
}

#[test]
fn the_grown_model_learns_with_finite_gradients() {
    let (_, g, _) = grown_exact("Ava 0.4 + 2 (for learning)", everything(), &plan(1, 2, None), 48);
    let m = &g.model;
    let toks = tokens(2, 41, VOCAB_GLANCE, 6);
    let x: Vec<u32> = (0..2).flat_map(|r| toks[r * 41..r * 41 + 40].to_vec()).collect();
    let y: Vec<u32> = (0..2).flat_map(|r| toks[r * 41 + 1..(r + 1) * 41].to_vec()).collect();
    let (x, y) = (Tensor::from_vec(x, (2, 40), &CPU).unwrap(), Tensor::from_vec(y, (2, 40), &CPU).unwrap());
    let mut params: Vec<Param> =
        g.varmap.data().lock().unwrap().iter().map(|(k, v)| Param { name: k.clone(), var: v.clone(), decay: Model::decays_with(k, true) }).collect();
    params.sort_by(|a, b| a.name.cmp(&b.name));
    let mut opt = AdamW::new(params, 0.1).unwrap();
    let (mut first, mut last) = (f32::NAN, f32::NAN);
    for step in 0..6 {
        let (total, ce, _) = m.loss_aux(&x, &y, 0.0).unwrap();
        let cv = ce.to_scalar::<f32>().unwrap();
        assert!(cv.is_finite() && total.to_scalar::<f32>().unwrap().is_finite());
        let gr = total.backward().unwrap();
        if step == 0 {
            first = cv;
            for p in opt.params() {
                match gr.get(p.var.as_tensor()) {
                    Some(t) => {
                        let s = t.abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
                        assert!(s.is_finite(), "{}: gradient {s}", p.name);
                        // the new blocks' zero read-outs can learn
                        if ["b4.mlp.w2", "b5.mlp.w2", "b4.hebb.wo", "b5.hebb.wo"].contains(&p.name.as_str()) {
                            assert!(s > 0.0, "{} gets no gradient", p.name);
                        }
                    }
                    None => assert!(p.name.ends_with(".theta"), "{} has no gradient", p.name),
                }
            }
        }
        last = cv;
        opt.step(&gr, 3e-3, 1.0).unwrap();
        m.homeostasis_step().unwrap();
    }
    println!("A15 grown Ava 0.4 learns: ce {first:.4} -> {last:.4} in 6 AdamW steps");
    assert!(last < first, "ce {first} -> {last}");
}

/// The MLP of a block by hand: kWTA (k) of h·w1, relu (squared with relu2), times w2.
fn mlp(h: &Tensor, w1: &Tensor, w2: &Tensor, k: usize, relu2: bool) -> Tensor {
    let u = h.matmul(w1).unwrap();
    let a = u.mul(&kwta_mask(&u, k).unwrap()).unwrap().relu().unwrap();
    let a = if relu2 { a.sqr().unwrap() } else { a };
    a.matmul(w2).unwrap()
}

/// A7 by the part: three passes of x + MLP(norm(x)) (wave5::mlp_step's ops), mixed by the stop weights.
fn ponder(hl: &ava_parts::halting::Halting, x: &Tensor, gain: &Tensor, w1: &Tensor, w2: &Tensor, k: usize, relu2: bool) -> ava_parts::halting::Pondered {
    let step = |x: &Tensor| -> candle_core::Result<Tensor> { x.add(&mlp(&rmsnorm(x, gain)?, w1, w2, k, relu2)) };
    hl.run(x, &step, 3).unwrap()
}

fn noise(dims: (usize, usize), seed: u64, scale: f32) -> Tensor {
    let n = dims.0 * dims.1;
    Tensor::from_vec(unit_noise(n, seed).iter().map(|z| z * scale).collect::<Vec<f32>>(), dims, &CPU).unwrap()
}

#[test]
fn copies_are_bit_identical_and_kwta_keeps_whole_groups() {
    let (n, d, h) = (64, 64, 256);
    let x = noise((n, d), 1, 1.0);
    let w1 = noise((d, h), 2, 0.02);
    for c in [2, 3, 4] {
        // the premise: a copied column gives bit-identical values (CPU matmul)
        let u = x.matmul(&w1).unwrap();
        let u2 = x.matmul(&carry("mlp.w1", &w1, 0, &[0], c, Split::First).unwrap()).unwrap();
        let tiled = Tensor::cat(&vec![&u; c], 1).unwrap();
        assert_eq!(max_diff(&u2, &tiled), 0.0, "C = {c}: copies differ");
        for k in [1, 32, 64, 255] {
            let m1 = Tensor::cat(&vec![&kwta_mask(&u, k).unwrap(); c], 1).unwrap();
            assert_eq!(max_diff(&kwta_mask(&u2, c * k).unwrap(), &m1), 0.0, "C = {c}, k = {k}");
            assert_eq!(max_diff(&kwta_mask_sorted(&u2, c * k).unwrap(), &m1), 0.0, "sorted, C = {c}, k = {k}");
        }
    }
    // exact ties at the boundary: [5, 3, 3, 1] keeps 5, 3, 3 at k = 2 (every value >= the 2nd largest); the copies keep
    // all six copies of them at k = 2C
    let u = Tensor::new(&[[5f32, 3., 3., 1.], [2., 2., 2., 2.], [0., -1., 7., 0.]], &CPU).unwrap();
    for c in [2, 3] {
        let uc = Tensor::cat(&vec![&u; c], 1).unwrap();
        for k in [1, 2, 3] {
            let want = Tensor::cat(&vec![&kwta_mask(&u, k).unwrap(); c], 1).unwrap();
            assert_eq!(max_diff(&kwta_mask(&uc, c * k).unwrap(), &want), 0.0, "ties, C = {c}, k = {k}");
        }
    }
}

#[test]
fn a_widened_mlp_and_its_halting_ponder_are_exact() {
    let cfg = everything();
    let (vm, _m) = build_awake(cfg, 3);
    let data = vm.data().lock().unwrap();
    let w1 = data["b1.mlp.w1"].as_tensor().clone();
    let w2 = data["b1.mlp.w2"].as_tensor().clone();
    let gain = data["b1.mlp.norm"].as_tensor().clone();
    let (k, relu2) = (cfg.kwta_k(), cfg.arch.relu2);
    assert_eq!((k, relu2, cfg.hidden()), (64, true, 256));
    let x = noise((96, 64), 4, 1.0);
    let h = rmsnorm(&x, &gain).unwrap();
    let want = mlp(&h, &w1, &w2, k, relu2);
    let hl = ava_parts::halting::Halting { w: noise((64, 1), 5, 0.3), b: Tensor::new(&[0.5f32], &CPU).unwrap() };
    let p0 = ponder(&hl, &x, &gain, &w1, &w2, k, relu2);
    for split in [Split::First, Split::Even] {
        for c in [2, 3, 4] {
            let (v1, v2) = (carry("mlp.w1", &w1, 0, &[0], c, split).unwrap(), carry("mlp.w2", &w2, 0, &[0], c, split).unwrap());
            let (abs, rel) = gap(&want, &mlp(&h, &v1, &v2, c * k, relu2)).unwrap();
            println!("A15 one MLP (kWTA 1/4, relu2) x{c} {split:?}: {abs:.3e}, relative {rel:.3e}");
            assert!(rel <= TOL, "C = {c} {split:?}: {rel}");
            // A7: three passes of x + MLP(norm(x)), mixed by the stop weights
            let p1 = ponder(&hl, &x, &gain, &v1, &v2, c * k, relu2);
            let (abs, rel) = gap(&p0.output, &p1.output).unwrap();
            println!("A15 the halting ponder (3 passes) x{c} {split:?}: {abs:.3e}, relative {rel:.3e}");
            assert!(rel <= TOL && max_diff(&p0.p, &p1.p) < 1e-6, "halting, C = {c} {split:?}");
        }
    }
    drop(data);
}

#[test]
fn without_kwta_the_widening_agrees_with_the_parts_widen() {
    // --kwta 1 keeps every unit: Ava's MLP is then the part's block relu(x·W1 + b1)·W2 with b1 = 0
    let (n, d, h, out) = (40, 16, 64, 16);
    let x = noise((n, d), 11, 1.0);
    let (w1, w2) = (noise((d, h), 12, 0.3), noise((h, out), 13, 0.3));
    let v = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (xs, w1s, w2s) = (v(&x), v(&w1), v(&w2));
    let (y, _) = ava_parts::growth::block(&xs, &w1s, &vec![0.0; h], &w2s, n, d, h, out);
    let want = Tensor::from_vec(y, (n, out), &CPU).unwrap();
    assert!(max_diff(&want, &mlp(&x, &w1, &w2, h, false)) < 1e-5, "the part's block is Ava's MLP without kWTA");
    for c in [2, 3] {
        let h2 = c * h;
        let (nw1, nb1, nw2) = ava_parts::growth::widen(&w1s, &vec![0.0; h], &w2s, d, h, out, h2, 7);
        let (yp, _) = ava_parts::growth::block(&xs, &nw1, &nb1, &nw2, n, d, h2, out);
        let part = Tensor::from_vec(yp, (n, out), &CPU).unwrap();
        for split in [Split::Even, Split::First] {
            let ours = mlp(&x, &carry("mlp.w1", &w1, 0, &[0], c, split).unwrap(), &carry("mlp.w2", &w2, 0, &[0], c, split).unwrap(), h2, false);
            let (dp, du) = (max_diff(&part, &want), max_diff(&ours, &want));
            println!("A15 no kWTA x{c}: the part's widen {dp:.3e}, ours ({split:?}) {du:.3e}, apart {:.3e}", max_diff(&ours, &part));
            assert!(dp < 1e-5 && du < 1e-5 && max_diff(&ours, &part) < 1e-5);
        }
    }
}

#[test]
fn mlp_times_is_exact_once_the_width_is_recorded() {
    if let Err(e) = width_record() {
        // this build cannot record the width: --mlp-times is refused, and says why
        assert!(e.contains("Config::hidden()") && e.contains("--mlp-mult"), "{e}");
        println!("A15 --mlp-times REFUSED in this build (src/model.rs's Config::hidden() does not read --mlp-mult): {e}");
        return;
    }
    let plain = Config::with_kind(Kind::D, 258, 32, 3, 64);
    // Net2Net's even split: exact in real arithmetic; measured against the model's own sensitivity to rounding (every
    // MLP's w2 moved by one part in 2^23, about the size of the even split's regrouping)
    for (name, cfg) in [("plain D L3", plain), ("Ava 0.4", everything())] {
        let (vm, m) = build_awake(cfg, 3);
        let x = ids(2, 48, cfg.vocab, 7);
        let even = grow(&cfg, &vm, &Plan { split: Split::Even, ..plan(2, 0, None) }, &CPU).unwrap();
        let (ea, er) = logit_gap(&m, &even.model, &x).unwrap();
        let (vm2, m2) = build_awake(cfg, 3);
        for (k, v) in vm2.data().lock().unwrap().iter() {
            if k.ends_with(".mlp.w2") {
                v.set(&v.as_tensor().affine(1.0 + f32::EPSILON as f64, 0.0).unwrap()).unwrap();
            }
        }
        let (sa, sr) = logit_gap(&m, &m2, &x).unwrap();
        println!("A15 {name} MLP x2 Even: {ea:.3e}, relative {er:.3e}; the source with w2 x (1 + 2^-23): {sa:.3e}, relative {sr:.3e}");
        drop(vm);
    }
    for (name, cfg, p) in [
        ("plain D L3 MLP x2", plain, plan(2, 0, None)),
        ("plain D L3 MLP x3 + 2 after block 0", plain, plan(3, 2, Some(0))),
        ("Ava 0.4 MLP x2", everything(), plan(2, 0, None)),
        ("Ava 0.4 MLP x2 + 2 at the top", everything(), plan(2, 2, None)),
    ] {
        let (m, g, _) = grown_exact(name, cfg, &p, 48);
        assert_eq!(g.model.cfg.hidden(), p.mlp_times * cfg.hidden());
        assert_eq!(g.model.cfg.kwta_k(), p.mlp_times * cfg.kwta_k());
        stir_new_blocks(&g);
        assert!(logit_gap(&m, &g.model, &ids(2, 48, cfg.vocab, 7)).unwrap().1 <= TOL, "{name} stirred");
        // grown twice: x2 then x2 again is x4
        let g2 = grow(&g.model.cfg, &g.varmap, &plan(2, 0, None), &CPU).unwrap();
        assert_eq!(g2.model.cfg.hidden(), 2 * g.model.cfg.hidden());
        assert!(logit_gap(&m, &g2.model, &ids(2, 48, cfg.vocab, 7)).unwrap().1 <= TOL, "{name} grown again");
    }
}

#[test]
fn growth_is_seeded() {
    let cfg = d_with(WIN, 258, 64, 2);
    let (vm, _) = build_awake(cfg, 3);
    let tensors = |p: &Plan| -> BTreeMap<String, Vec<f32>> {
        let g = grow(&cfg, &vm, p, &CPU).unwrap();
        let data = g.varmap.data().lock().unwrap();
        data.iter().map(|(k, v)| (k.clone(), v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())).collect()
    };
    let a = tensors(&Plan { seed: 7, ..plan(1, 1, Some(0)) });
    assert_eq!(a, tensors(&Plan { seed: 7, ..plan(1, 1, Some(0)) }), "the same seed, the same model");
    let b = tensors(&Plan { seed: 8, ..plan(1, 1, Some(0)) });
    for (k, v) in &a {
        let new = block_of(k).is_some_and(|(j, _)| j == 1);
        if !new {
            assert_eq!(v, &b[k], "{k}: an old tensor depends on the seed");
        }
    }
    assert_ne!(a["b1.mlp.w1"], b["b1.mlp.w1"], "a new block's init follows the seed");
}

#[test]
fn grow_exact_end_to_end_through_files() {
    let dir = scratch("grow_exact_e2e");
    let cfg = everything();
    let (vm, m) = build_awake(cfg, 3);
    let src = dir.join("src").join("model.safetensors");
    std::fs::create_dir_all(src.parent().unwrap()).unwrap();
    vm.save(&src).unwrap();
    write_arch_beside(&src, &cfg.arch).unwrap();
    let out = dir.join("grown");
    let args = |extra: &[&str]| -> Vec<String> {
        let mut a: Vec<String> = vec!["--init".into(), src.parent().unwrap().display().to_string(), "--out".into(), out.display().to_string()];
        a.extend(extra.iter().map(|s| s.to_string()));
        a
    };
    let rep = run(&args(&["--deeper", "2", "--seed", "3"])).unwrap();
    println!(
        "A15 grow_exact Ava 0.4 d64 L4 -> L6: gap in memory {:.3e} (rel {:.3e}), loaded back {:.3e} (rel {:.3e}); {} -> {} weights; secs {:?}",
        rep.gap_mem.0, rep.gap_mem.1, rep.gap_file.0, rep.gap_file.1, rep.params.0, rep.params.1, rep.secs
    );
    assert!(rep.gap_file.1 <= TOL);
    let arch = std::fs::read_to_string(out.join("model.arch.txt")).unwrap();
    assert!(arch.contains("--ws-layer 2") && arch.contains("--cerebellum 64"), "{arch}");
    assert!(std::fs::read_to_string(out.join("grow.txt")).unwrap().contains("not carried"));
    // the written model, loaded as any tool loads it, on other tokens
    let (c2, _, back) = load_auto(&out.join("model.safetensors"), Kind::D, 64, &CPU).unwrap();
    assert_eq!((c2.layers, c2.glance), (6, true));
    let x = ids(2, 40, VOCAB_GLANCE, 13);
    let (abs, rel) = logit_gap(&m, &back, &x).unwrap();
    println!("A15 grow_exact's file loaded back, other tokens: {abs:.3e}, relative {rel:.3e}");
    assert!(rel <= TOL);
    // a folder that already holds a model is refused
    assert!(run(&args(&["--deeper", "1"])).err().unwrap().contains("empty folder"));
    let refused = run(&[args(&[])[..2].to_vec(), vec!["--out".into(), dir.join("x").display().to_string(), "--deeper".into(), "1".into(), "--at".into(), "0".into()]].concat());
    assert!(refused.err().unwrap().contains("--cerebellum"));
    assert!(!dir.join("x").exists(), "nothing is written when growth is refused");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn each_part_of_ava04_alone_grows_exactly() {
    let parts: Vec<String> = WAVE5.split(" --").map(|s| if s.starts_with("--") { s.to_string() } else { format!("--{s}") }).collect();
    let mut sets: Vec<String> = vec![WIN.into(), SLOW.into(), format!("{WIN} {SLOW}")];
    sets.extend(parts.iter().map(|p| if p.contains("--schema") { format!("{WIN} {SLOW} --workspace 4 {p}") } else { format!("{WIN} {SLOW} {p}") }));
    sets.push(format!("{WIN} {SLOW} --modulate --halting 3"));
    for f in sets {
        let cfg = d_with(&f, VOCAB_GLANCE, 64, 4);
        let (vm, m) = build_awake(cfg, 3);
        let g = grow(&cfg, &vm, &plan(1, 2, None), &CPU).unwrap();
        stir_new_blocks(&g);
        let (abs, rel) = logit_gap(&m, &g.model, &ids(2, 48, VOCAB_GLANCE, 7)).unwrap();
        println!("A15 d64 L4 + 2 at the top, new blocks stirred, {}: {abs:.3e}, relative {rel:.3e}", f.replace(WIN, "WIN").replace(SLOW, "SLOW"));
        assert!(rel <= TOL, "{f}: {rel:.3e}");
    }
}

#[test]
fn identical_passes_mix_back_exactly_with_stop_bias_0_for_up_to_3_passes() {
    // what PINNED and HALT_EXACT_MAX rest on: A7 over passes that return their input (a new block), on values of every size
    let x = noise((512, 64), 77, 3.0);
    let x = Tensor::cat(&[&x, &x.affine(1e-3, 0.0).unwrap(), &x.affine(1e4, 0.0).unwrap(), &x.affine(-0.7, 0.0).unwrap()], 0).unwrap();
    let id = |h: &Tensor| -> candle_core::Result<Tensor> { h.add(&h.zeros_like()?) };
    let mix = |b: f32, n: usize| {
        let hl = ava_parts::halting::Halting { w: Tensor::zeros((64, 1), DType::F32, &CPU).unwrap(), b: Tensor::new(&[b], &CPU).unwrap() };
        max_diff(&hl.run(&x, &id, n).unwrap().output, &x)
    };
    for n in 2..=8 {
        let (d0, d4) = (mix(0.0, n), mix(4.0, n));
        println!("A15 A7 mix of {n} identical passes: stop bias 0 -> {d0:.3e}, the part's 4 -> {d4:.3e}");
        assert_eq!(d0 == 0.0, n <= HALT_EXACT_MAX, "{n} passes, bias 0: {d0}");
        assert!(d4 > 0.0);
    }
    // and --halting 4 is refused for --deeper
    let cfg = d_with("--halting 4", 258, 32, 2);
    let e = layout(&cfg, &plan(1, 1, None)).err().unwrap();
    assert!(e.contains("--halting 4") && e.contains("exactly 1"), "{e}");
    assert!(layout(&cfg, &plan(1, 0, None)).is_err(), "nothing to grow");
}

#[test]
fn the_cpu_matmul_keeps_copies_bit_identical_and_the_first_split_exact_up_to_512_units() {
    // the premises of Split: a copied column of w1 gives bit-identical values; with the first-copy split the extra terms
    // are exact zeros, so the sum is the old one as long as the matmul keeps the old units in one block of the inner
    // dimension (the CPU gemm does up to 512; at 1536 it regroups); the even split regroups always
    for (n, d, h) in [(96usize, 64usize, 256usize), (192, 128, 512), (192, 384, 1536), (37, 384, 1536)] {
        let x = noise((n, d), 1, 1.0);
        let w1 = noise((d, h), 2, 0.02);
        let w2 = noise((h, d), 3, 0.02);
        let a = x.matmul(&w1).unwrap().relu().unwrap();
        let y = a.matmul(&w2).unwrap();
        for c in [2usize, 3, 4] {
            let u2 = x.matmul(&carry("mlp.w1", &w1, 0, &[0], c, Split::First).unwrap()).unwrap();
            let cols = max_diff(&u2, &Tensor::cat(&vec![&x.matmul(&w1).unwrap(); c], 1).unwrap());
            let ac = Tensor::cat(&vec![&a; c], 1).unwrap();
            let split = |s: Split| gap(&y, &ac.matmul(&carry("mlp.w2", &w2, 0, &[0], c, s).unwrap()).unwrap()).unwrap().1;
            let (first, even) = (split(Split::First), split(Split::Even));
            println!("A15 matmul n{n} d{d} h{h} x{c}: copies {cols:.3e}, first-copy split {first:.3e}, even split {even:.3e} (relative)");
            assert_eq!(cols, 0.0, "h {h} x{c}: copies differ");
            assert!(first <= TOL && even <= TOL);
            if h <= 512 {
                assert_eq!(first, 0.0, "h {h} x{c}: the first-copy split moved the sum");
            }
        }
    }
}

/// The timing at Ava's next size: d384 L8 with the trained full-all-a16 run's block options, grown by one block at the
/// top (and, where the width can be recorded, its MLPs x2 as well) through grow_exact's whole path (files in, files
/// out, both checks). Ignored by default (about 5 seconds a growth): `-- grow:: --ignored`.
#[test]
#[ignore]
fn timing_d384_l8() {
    const A16: &str = "--head-size 64 --timescales fixed --gates channel --value-residual --kwta 1/4 --mlp-act relu2 --mem-momentum 0.5 --resid-fix --residual dense --lif-init logspaced --lif-scale --spikes int2 --alif --homeostasis 0.1 --dendrites 4 --stp --delays 4 --wm 8 --resonate 64 --modulate --tem 16 --halting 3 --cerebellum 512 --jepa 0.1 --diffusion 0.1 --critic 0.1 --confidence 0.1";
    let dir = scratch("grow_exact_d384");
    let t0 = std::time::Instant::now();
    let cfg = d_with(A16, VOCAB_GLANCE, 384, 8);
    let (vm, _m) = build_awake(cfg, 3);
    let src = dir.join("src").join("model.safetensors");
    std::fs::create_dir_all(src.parent().unwrap()).unwrap();
    vm.save(&src).unwrap();
    write_arch_beside(&src, &cfg.arch).unwrap();
    drop((vm, _m));
    let made = t0.elapsed().as_secs_f64();
    let args: Vec<String> = ["--init", &src.display().to_string(), "--out", &dir.join("grown").display().to_string(), "--deeper", "1"].iter().map(|s| s.to_string()).collect();
    let rep = run(&args).unwrap();
    println!(
        "A15 timing d384 L8 -> L9: {} -> {} weights; secs load {:.2} grow {:.2} check {:.2} save {:.2} reload+check {:.2}, total {:.2} (making the source {made:.1}); gap {:.3e} / {:.3e}",
        rep.params.0, rep.params.1, rep.secs[0], rep.secs[1], rep.secs[2], rep.secs[3], rep.secs[4], rep.secs.iter().sum::<f64>(), rep.gap_mem.1, rep.gap_file.1
    );
    assert!(rep.gap_file.1 <= TOL);
    if width_record().is_ok() {
        // at 1536 hidden units the CPU matmul regroups the widened sum (about 1e-7), and these noise-filled spiking
        // models amplify that past TOL (a spike flips), so grow_exact refuses: printed with its reason and the model's
        // own sensitivity, whichever way it goes
        let mut a = args.clone();
        a[3] = dir.join("wide").display().to_string();
        a.extend(["--mlp-times".to_string(), "2".to_string()]);
        let t = std::time::Instant::now();
        match run(&a) {
            Ok(rep) => println!("A15 timing d384 L8 a16 -> L9 + MLP x2: gap {:.3e} / {:.3e}, total {:.2} s", rep.gap_mem.1, rep.gap_file.1, rep.secs.iter().sum::<f64>()),
            Err(e) => println!("A15 timing d384 L8 a16 -> L9 + MLP x2 REFUSED after {:.2} s: {e}", t.elapsed().as_secs_f64()),
        }
        let plain = Config::with_kind(Kind::D, VOCAB_GLANCE, 384, 8, 64).with_glance().unwrap();
        let (vm, _m) = build_awake(plain, 3);
        let src = dir.join("plain").join("model.safetensors");
        std::fs::create_dir_all(src.parent().unwrap()).unwrap();
        vm.save(&src).unwrap();
        drop((vm, _m));
        let a: Vec<String> = ["--init", &src.display().to_string(), "--out", &dir.join("plain-wide").display().to_string(), "--deeper", "1", "--mlp-times", "2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let t = std::time::Instant::now();
        match run(&a) {
            Ok(rep) => println!(
                "A15 timing plain D d384 L8 -> L9 + MLP x2: {} -> {} weights; secs load {:.2} grow {:.2} check {:.2} save {:.2} reload+check {:.2}, total {:.2}; gap {:.3e} / {:.3e}",
                rep.params.0, rep.params.1, rep.secs[0], rep.secs[1], rep.secs[2], rep.secs[3], rep.secs[4], rep.secs.iter().sum::<f64>(), rep.gap_mem.1, rep.gap_file.1
            ),
            Err(e) => println!("A15 timing plain D d384 L8 -> L9 + MLP x2 REFUSED after {:.2} s: {e}", t.elapsed().as_secs_f64()),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
