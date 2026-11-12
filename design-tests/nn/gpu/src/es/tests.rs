// TESTING / DESIGN ONLY — tests of src/es.rs and src/es/tune.rs (M7 evolution strategies, H1, es_tune), CPU only.

use super::tune::{self, fitness, parse, run, split, Fitness, Prepared};
use super::*;
use crate::arch::Arch;
use crate::mc::{self, Item, Norm};
use crate::model::{Config, Kind, Model};
use crate::state::tests::build_awake;
use crate::testkit::{byte_bpe, scratch, tiny, unit_noise};
use candle_core::{Device, Tensor, Var};
use candle_nn::VarMap;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const CPU: Device = Device::Cpu;

/// A D model with the engram (block 1), its L3 write and the D2 holographic trace: the H1 weights to find.
fn h1_cfg() -> Config {
    Config::with_kind(Kind::D, 258, 32, 2, 64).with_engram(1).unwrap().with_arch(Arch::from_flags("--btsp 4 --holo 8").unwrap()).unwrap()
}

fn bits(vm: &VarMap) -> BTreeMap<String, Vec<u32>> {
    vm.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect())).collect()
}

fn host_of(v: &Var) -> Vec<f32> {
    v.as_tensor().flatten_all().unwrap().to_vec1().unwrap()
}

/// The numerical rank of a row-major m×n matrix (Gaussian elimination with partial pivoting).
fn rank_of(m: usize, n: usize, a: &[f64]) -> usize {
    let mut x = a.to_vec();
    let tol = 1e-9 * x.iter().fold(0f64, |s, v| s.max(v.abs())) * m.max(n) as f64;
    let (mut row, mut rank) = (0, 0);
    for col in 0..n {
        if row == m {
            break;
        }
        let (piv, val) = (row..m).map(|r| (r, x[r * n + col].abs())).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        if val <= tol {
            continue;
        }
        for c in 0..n {
            x.swap(piv * n + c, row * n + c);
        }
        for r in row + 1..m {
            let f = x[r * n + col] / x[row * n + col];
            for c in col..n {
                x[r * n + c] -= f * x[row * n + c];
            }
        }
        row += 1;
        rank += 1;
    }
    rank
}

fn toy_items(n: usize) -> Vec<Item> {
    let words = ["red", "blue", "green", "cat", "dog", "sun", "moon", "tree", "a long answer"];
    (0..n)
        .map(|i| Item {
            id: format!("toy-{i:04}"),
            kind: "toy".into(),
            bin: if i % 2 == 0 { "even".into() } else { "odd".into() },
            prompt: format!("Q: which word is number {i}?\nA:"),
            choices: (0..3).map(|j| words[(i * 3 + j) % words.len()].to_string()).collect(),
            right: i % 3,
        })
        .collect()
}

/// An item TSV of the data crate's format (`id kind bin right prompt c0 c1 c2`, \n escaped).
fn write_items(path: &Path, items: &[Item]) {
    let mut s = String::from("id\tkind\tbin\tright\tprompt\tc0\tc1\tc2\n");
    for it in items {
        s += &format!("{}\t{}\t{}\t{}\t{}\t{}\n", it.id, it.kind, it.bin, it.right, it.prompt.replace('\n', "\\n"), it.choices.join("\t"));
    }
    std::fs::write(path, s).unwrap();
}

// ------------------------------------------------------------------------------------------------ selection

#[test]
fn parameters_are_selected_by_prefix_parts_and_star() {
    let names = [
        "b0.hebb.wa",
        "b0.hebb.wa1",
        "b1.hebb.wa",
        "b10.hebb.wa",
        "b1.mlp.w1",
        "b1.mlp.norm",
        "b3.mlp.w1",
        "b0.parts.holo.lam",
        "b12.parts.holo.lam",
        "b12.parts.holo.wo",
        "tok_emb",
        "final.norm",
    ];
    let sel = |p: &str| select_names(names.iter().copied(), p);
    assert_eq!(sel("b*.parts.holo.lam").unwrap(), vec!["b0.parts.holo.lam", "b12.parts.holo.lam"]);
    assert_eq!(sel("b3.mlp").unwrap(), vec!["b3.mlp.w1"]);
    assert_eq!(sel("b1").unwrap(), vec!["b1.hebb.wa", "b1.mlp.norm", "b1.mlp.w1"], "b1 is not b10");
    assert_eq!(sel("b*.hebb.wa").unwrap(), vec!["b0.hebb.wa", "b1.hebb.wa", "b10.hebb.wa"], "a part is matched whole: not wa1");
    assert_eq!(sel("*.mlp.w1").unwrap(), vec!["b1.mlp.w1", "b3.mlp.w1"]);
    assert_eq!(sel("b1*.hebb.wa").unwrap(), vec!["b1.hebb.wa", "b10.hebb.wa"]);
    assert_eq!(sel("b*.parts.holo").unwrap(), vec!["b0.parts.holo.lam", "b12.parts.holo.lam", "b12.parts.holo.wo"]);
    assert_eq!(sel(" b3.mlp , b*.parts.holo.lam,b3.mlp.w1 ").unwrap(), vec!["b0.parts.holo.lam", "b12.parts.holo.lam", "b3.mlp.w1"], "a union, each once, sorted");
    assert_eq!(sel("tok_emb,final").unwrap(), vec!["final.norm", "tok_emb"]);
    assert!(sel("b3.mlp,b9.nothing").unwrap_err().contains("b9.nothing"), "a pattern that matches nothing is an error");
    assert!(sel(" , ").is_err());
    assert!(sel("b1.mlp.w1.x").is_err(), "longer than the name");
    assert!(name_matches("b*", "b0.anything") && name_matches("*", "tok_emb") && !name_matches("c*", "b0.x"));

    // on a real model, in name order, and H1's names where the model has them
    let (vm, _) = build_awake(h1_cfg(), 2);
    let got: Vec<String> = select(&vm, "b1.mlp").unwrap().into_iter().map(|(n, _)| n).collect();
    let mut want: Vec<String> = vm.data().lock().unwrap().keys().filter(|k| k.starts_with("b1.mlp.")).cloned().collect();
    want.sort();
    assert!(!want.is_empty());
    assert_eq!(got, want);
    let keys: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
    assert_eq!(
        h1_patterns(keys.iter().map(String::as_str)).unwrap(),
        "b*.hebb.wa,b*.hebb.ba,b*.hebb.wb,b*.hebb.bb,b*.eng.wp,b*.eng.bp,b*.eng.btsp_g,b*.eng.btsp_mix,b*.parts.holo.lam"
    );
    let h1: Vec<String> = select(&vm, &h1_patterns(keys.iter().map(String::as_str)).unwrap()).unwrap().into_iter().map(|(n, _)| n).collect();
    assert_eq!(h1.len(), 4 * 2 + 4 + 2, "four gates in each of 2 blocks, the engram's four, a trace rate per block: {h1:?}");
    let (_, plain, _) = tiny(Kind::D, 1);
    let keys: Vec<String> = plain.data().lock().unwrap().keys().cloned().collect();
    assert_eq!(h1_patterns(keys.iter().map(String::as_str)).unwrap(), "b*.hebb.wa,b*.hebb.ba,b*.hebb.wb,b*.hebb.bb");
    assert!(h1_patterns(["tok_emb", "b0.mlp.w1"]).is_err());
}

// ------------------------------------------------------------------------------------------------ noise

#[test]
fn low_rank_noise_has_rank_r_and_regenerates_from_its_seed() {
    for r in [1usize, 2, 3, 5] {
        let e = Noise::new(stream_seed(member_seed(4, 2, 1), 0), &[12, 9], r);
        assert!(matches!(e, Noise::LowRank { m: 12, n: 9, .. }));
        assert_eq!(rank_of(12, 9, &e.dense()), r, "rank {r}");
    }
    assert_eq!(rank_of(12, 9, &Noise::new(5, &[12, 9], 20).dense()), 9, "rank above min(m, n) is full");
    // the same seed gives the same noise, bit for bit; another member, generation or tensor does not
    let es_like = |g: u64, k: usize, p: usize| Noise::new(stream_seed(member_seed(7, g, k), p), &[16, 8], 2);
    assert_eq!(es_like(3, 1, 0), es_like(3, 1, 0));
    assert_eq!(es_like(3, 1, 0).dense(), es_like(3, 1, 0).dense());
    assert_ne!(es_like(3, 1, 0), es_like(3, 2, 0));
    assert_ne!(es_like(3, 1, 0), es_like(4, 1, 0));
    assert_ne!(es_like(3, 1, 0), es_like(3, 1, 1));
    // every entry has variance 1 (low rank and full), mean 0
    let stats = |v: &[f64]| {
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        (m, v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n)
    };
    let (m, var) = stats(&Noise::new(11, &[256, 256], 4).dense());
    assert!(m.abs() < 0.1 && (0.75..1.3).contains(&var), "low rank: mean {m}, variance {var}");
    let full = Noise::new(12, &[20000], 1);
    assert!(matches!(full, Noise::Full(_)));
    let (m, var) = stats(&full.dense());
    assert!(m.abs() < 0.03 && (var - 1.0).abs() < 0.05, "full: mean {m}, variance {var}");
    assert_eq!(Noise::new(1, &[2, 3, 4], 2).dense().len(), 24, "3-D gets full noise");
    assert_eq!(Noise::new(1, &[], 2).dense().len(), 1, "a scalar too");

    // the noise an Es draws is this noise, and E is what moves the weights: (θ+σE − θ)/σ ≈ E
    let (_, vm, model) = tiny(Kind::D, 3);
    let es = Es::new(select(&vm, "b0.hebb.wq").unwrap(), EsOpts { pop: 1, sigma: 0.01, lr: 0.0, rank: 2, seed: 9 }).unwrap();
    let e = es.noise(0, 0, 0).dense();
    assert_eq!(e, Noise::new(stream_seed(member_seed(9, 0, 0), 0), &[32, 32], 2).dense());
    let th = es.theta()[0].clone();
    let var = select(&vm, "b0.hebb.wq").unwrap().remove(0).1;
    let mut got = Vec::new();
    es.eval_shifted(&model, &[e.clone()], 0.01, &mut |_: &Model| {
        got = host_of(&var);
        Ok(0.0)
    })
    .unwrap();
    let moved: Vec<f64> = got.iter().zip(&th).map(|(a, b)| (*a as f64 - *b as f64) / 0.01).collect();
    assert!(moved.iter().zip(&e).all(|(a, b)| (a - b).abs() < 1e-3), "the weights moved by σE");
}

// ------------------------------------------------------------------------------------------------ shaping and update

#[test]
fn centred_ranks_and_a_constant_fitness_give_no_update() {
    assert_eq!(centred_ranks(&[3.0, 1.0, 2.0]), vec![0.5, -0.5, 0.0]);
    let t = centred_ranks(&[1.0, 1.0, 2.0, 2.0]);
    assert!((t[0] + 1.0 / 3.0).abs() < 1e-12 && t[0] == t[1] && (t[2] - 1.0 / 3.0).abs() < 1e-12 && t[2] == t[3], "ties share their rank: {t:?}");
    assert_eq!(centred_ranks(&[f64::NAN, 0.0, -5.0]), vec![-0.5, 0.5, 0.0], "NaN is the worst");
    assert_eq!(centred_ranks(&[4.0]), vec![0.0]);
    let c = pair_coeffs(&[2.0, 0.0], &[0.0, 2.0]);
    assert!((c[0] - 1.0 / 3.0).abs() < 1e-12 && (c[1] + 1.0 / 3.0).abs() < 1e-12, "{c:?}");
    assert_eq!(pair_coeffs(&[5.0; 4], &[5.0; 4]), vec![0.0; 4]);

    let (_, vm, model) = tiny(Kind::D, 4);
    let before = bits(&vm);
    let mut es = Es::new(select(&vm, "b0.hebb.wq,b0.mem.beta,b1.mlp.w2").unwrap(), EsOpts { pop: 5, sigma: 0.05, lr: 1.0, rank: 2, seed: 3 }).unwrap();
    let mut evals = 0;
    let rep = es
        .generation(&model, &mut |_: &Model| {
            evals += 1;
            Ok(0.25)
        })
        .unwrap();
    assert_eq!(evals, 10);
    assert!(rep.coeffs.iter().all(|&c| c == 0.0));
    assert!(es.direction(0, &rep.coeffs).iter().flatten().all(|&x| x == 0.0));
    assert_eq!(rep.step, 0.0);
    assert_eq!(bits(&vm), before, "a constant fitness moves nothing, bit for bit");
    assert_eq!((rep.mean(), rep.best(), rep.worst(), es.gen), (0.25, 0.25, 0.25, 1));
}

#[test]
fn weights_are_restored_bit_for_bit_after_every_evaluation() {
    let (vm, model) = build_awake(h1_cfg(), 4);
    let before = bits(&vm);
    let sel = select(&vm, "b0.hebb.wq,b*.parts.holo.lam,b1.mlp").unwrap();
    let chosen: BTreeSet<String> = sel.iter().map(|(n, _)| n.clone()).collect();
    let n = sel.len();
    let mut es = Es::new(sel, EsOpts { pop: 3, sigma: 0.05, lr: 0.0, rank: 2, seed: 5 }).unwrap();
    let ids = Tensor::from_vec((0..20u32).map(|i| i * 11 % 250 + 3).collect::<Vec<_>>(), (1, 20), &CPU).unwrap();
    let logit_sum = |m: &Model| -> Result<f64, String> { Ok(m.forward(&ids).map_err(|e| e.to_string())?.sum_all().and_then(|s| s.to_scalar::<f32>()).map_err(|e| e.to_string())? as f64) };
    let mut seen = 0;
    let mut fit = |m: &Model| -> Result<f64, String> {
        for (k, v) in &bits(&vm) {
            if chosen.contains(k) {
                assert_ne!(v, &before[k], "{k} is perturbed during the evaluation");
            } else {
                assert_eq!(v, &before[k], "{k} is not chosen and stays");
            }
        }
        seen += 1;
        logit_sum(m)
    };
    es.generation(&model, &mut fit).unwrap();
    assert_eq!(seen, 6);
    assert_eq!(bits(&vm), before, "lr 0: every weight back, bit for bit");
    let f0 = logit_sum(&model).unwrap();
    // a failing fitness still puts the weights back
    let e: Vec<Vec<f64>> = (0..n).map(|p| es.noise(9, 0, p).dense()).collect();
    assert!(es.eval_shifted(&model, &e, 0.1, &mut |_: &Model| Err("boom".to_string())).is_err());
    assert_eq!(bits(&vm), before);
    // the same perturbation twice gives the same fitness, and the centre's own fitness is unchanged after it
    let a = es.eval_shifted(&model, &e, 0.1, &mut |m: &Model| logit_sum(m)).unwrap();
    let b = es.eval_shifted(&model, &e, 0.1, &mut |m: &Model| logit_sum(m)).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, f0);
    assert_eq!(logit_sum(&model).unwrap(), f0);
    // with lr > 0 only the chosen weights move
    es.opts.lr = 0.01;
    let mut k = 0.0;
    es.generation(&model, &mut |m: &Model| {
        k += 1.0;
        Ok(logit_sum(m)? + k)
    })
    .unwrap();
    let after = bits(&vm);
    let mut moved = 0;
    for (name, v) in &after {
        if chosen.contains(name) {
            moved += (v != &before[name]) as usize;
        } else {
            assert_eq!(v, &before[name], "{name} is not chosen");
        }
    }
    assert!(moved > 0);
    assert_eq!(es.theta().iter().map(|t| t.iter().map(|x| x.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>(), es.names.iter().map(|n| after[n].clone()).collect::<Vec<_>>(), "the variables hold the centre");
}

#[test]
fn the_update_is_the_regenerated_sum_of_the_noise() {
    let (_, vm, model) = tiny(Kind::D, 6);
    let opts = EsOpts { pop: 4, sigma: 0.02, lr: 0.003, rank: 3, seed: 21 };
    let mut es = Es::new(select(&vm, "b0.hebb.wq,b0.hebb.ba,b1.mlp.w1,b1.mem.beta").unwrap(), opts).unwrap();
    let th0 = es.theta().to_vec();
    let mut calls = 0usize;
    let rep = es
        .generation(&model, &mut |_: &Model| {
            calls += 1;
            Ok((calls * 7 % 11) as f64)
        })
        .unwrap();
    assert!(rep.coeffs.iter().any(|&c| c != 0.0));
    let scale = opts.lr / (opts.pop as f64 * opts.sigma);
    let mut step = 0f64;
    for p in 0..th0.len() {
        let mut want: Vec<f64> = th0[p].iter().map(|&x| x as f64).collect();
        for (k, &c) in rep.coeffs.iter().enumerate() {
            for (w, e) in want.iter_mut().zip(es.noise(0, k, p).dense()) {
                *w += scale * c * e;
            }
        }
        for (i, (&got, &w)) in es.theta()[p].iter().zip(&want).enumerate() {
            assert!((got as f64 - w).abs() < 1e-6 * w.abs().max(1.0), "{} [{i}]: {got} vs {w}", es.names[p]);
            step += (got as f64 - th0[p][i] as f64).powi(2);
        }
    }
    assert!((rep.step - step.sqrt()).abs() < 1e-9 * step.sqrt().max(1.0));
    assert!(rep.step > 0.0 && rep.norm > 0.0);
}

#[test]
fn es_climbs_a_toy_fitness() {
    // pull one weight matrix (low-rank noise) and one vector (full noise) of a fixed tiny model toward a target
    let (_, vm, model) = tiny(Kind::D, 3);
    let sel = select(&vm, "b0.hebb.wq,b0.mem.beta").unwrap();
    let vars: Vec<Var> = sel.iter().map(|(_, v)| v.clone()).collect();
    let target: Vec<Vec<f32>> = vars.iter().enumerate().map(|(i, v)| host_of(v).iter().zip(unit_noise(v.elem_count(), 50 + i as u64)).map(|(a, z)| a + 0.05 * z).collect()).collect();
    let dist = |vars: &[Var]| -> f64 { vars.iter().zip(&target).map(|(v, t)| host_of(v).iter().zip(t).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>()).sum() };
    let d0 = dist(&vars);
    let mut es = Es::new(sel, EsOpts { pop: 24, sigma: 0.01, lr: 0.001, rank: 1, seed: 11 }).unwrap();
    let mut f = |_: &Model| -> Result<f64, String> { Ok(-dist(&vars)) };
    let mut first = None;
    for _ in 0..120 {
        let r = es.generation(&model, &mut f).unwrap();
        first.get_or_insert(r.mean());
    }
    let d1 = dist(&vars);
    assert!(d1 < 0.6 * d0, "ES should pull the weights toward the target: squared distance {d0:.4} -> {d1:.4}");
    assert!(-d1 > first.unwrap(), "the fitness climbed");
}

// ------------------------------------------------------------------------------------------------ es_tune

#[test]
fn the_split_is_deterministic_and_disjoint() {
    let items = toy_items(2000);
    let (t, c) = split(&items, 0.5, 1);
    assert_eq!((t.clone(), c.clone()), split(&items, 0.5, 1), "the same every time");
    let (ts, cs): (BTreeSet<usize>, BTreeSet<usize>) = (t.iter().copied().collect(), c.iter().copied().collect());
    assert!(ts.is_disjoint(&cs));
    assert_eq!(ts.union(&cs).copied().collect::<Vec<_>>(), (0..items.len()).collect::<Vec<_>>());
    assert!((t.len() as f64 / items.len() as f64 - 0.5).abs() < 0.05, "share {}", t.len());
    // an item's part is its own: reordering the items does not move any of them
    let rev: Vec<Item> = items.iter().rev().cloned().collect();
    let ids = |its: &[Item], idx: &[usize]| -> BTreeSet<String> { idx.iter().map(|&i| its[i].id.clone()).collect() };
    assert_eq!(ids(&rev, &split(&rev, 0.5, 1).0), ids(&items, &t));
    assert_ne!(split(&items, 0.5, 2).0, t, "another split seed, another split");
    assert!(split(&items, 0.3, 1).0.iter().all(|i| ts.contains(i)), "a smaller share takes a subset");
    assert!(split(&items, 1.0, 1).1.is_empty());
}

#[test]
fn prepared_grading_is_mc_grade_exactly() {
    let (cfg, _vm, model) = tiny(Kind::D, 8);
    model.set_inference(true);
    let bpe = byte_bpe();
    let items = toy_items(14);
    for norm in [Norm::Total, Norm::Byte, Norm::Pmi] {
        let want = mc::grade(&model, &CPU, &bpe, &items, false, 700, norm).unwrap();
        let p = Prepared::new(&bpe, &items, false, cfg.vocab, norm);
        assert!(p.tokens() > 0 && p.len() == items.len());
        let got = p.grade(&model, &CPU, 700).unwrap();
        for (a, b) in want.iter().zip(&got) {
            assert_eq!(a.nll, b.nll, "{norm:?}");
            assert_eq!((a.right, a.pick, a.margin.to_bits()), (b.right, b.pick, b.margin.to_bits()), "{norm:?}");
        }
        let rights: Vec<usize> = items.iter().map(|it| it.right).collect();
        let (r, w, _) = mc::with_tau(&got, 0.0);
        assert_eq!(fitness(&got, &rights, Fitness::Rmw, 0.0), (r as f64 - w as f64) / items.len() as f64);
        let tau = mc::best_tau(&got);
        let (r, w, _) = mc::with_tau(&got, tau);
        assert_eq!(p.fitness(&got, Fitness::Rmw, tau), (r as f64 - w as f64) / items.len() as f64);
        assert_eq!(p.fitness(&got, Fitness::Acc, 0.0), got.iter().filter(|g| g.right).count() as f64 / items.len() as f64);
        let nll = -got.iter().zip(&rights).map(|(g, &r)| g.nll[r]).sum::<f64>() / items.len() as f64;
        assert_eq!(p.fitness(&got, Fitness::Nll, 0.0), nll);
    }
}

#[test]
fn es_tune_runs_end_to_end_on_a_tiny_saved_model() {
    let dir = scratch("es_tune_e2e");
    let cfg = h1_cfg();
    let (vm, _) = build_awake(cfg, 5);
    let base = dir.join("base.safetensors");
    crate::testkit::save(&vm, &base);
    crate::ckpt::write_arch_beside(&base, &cfg.arch).unwrap();
    let bpe = dir.join("bpe.txt");
    byte_bpe().save(&bpe).unwrap();
    let items = dir.join("toy_dev.tsv");
    write_items(&items, &toy_items(30));
    let out = dir.join("out").join("tuned.safetensors");
    let s = |x: &Path| x.to_string_lossy().into_owned();
    let flags = |fit: &str, gens: &str, out: &Path| -> Vec<String> {
        [
            "--model", &s(&base), "--items", &s(&items), "--params", "h1", "--out", &s(out), "--fitness", fit, "--pop", "3", "--sigma", "0.05", "--lr", "0.05", "--rank", "2",
            "--gens", gens, "--seed", "7", "--tune-share", "0.5", "--batch-tokens", "2048", "--cpu", "--threads", "2", "--bpe", &s(&bpe),
        ]
        .iter()
        .map(|x| x.to_string())
        .collect()
    };
    let a = parse(&flags("nll", "2", &out)).unwrap();
    assert_eq!((a.fitness, a.es.pop, a.es.rank, a.gens, a.es.seed, a.threads, a.cpu, a.tau), (Fitness::Nll, 3, 2, 2, 7, 2, true, 0.0));
    let sum = run(&a).unwrap();
    assert_eq!(sum.n_tune + sum.n_check, 30);
    assert!(sum.n_tune > 0 && sum.n_check > 0);
    assert_eq!(sum.tuned.len(), 14, "the H1 weights of this model: {:?}", sum.tuned);
    assert_eq!(sum.rows.len(), 3);
    assert!(sum.rows.iter().all(|r| r.tune.is_finite() && r.check.is_finite()));
    assert!(sum.rows[1..].iter().all(|r| r.members.as_ref().is_some_and(|m| m.evals == 6 && m.step > 0.0)));

    // the tuned model loads with its arch file; only the tuned weights moved
    assert!(out.is_file() && crate::ckpt::arch_name(&out).is_file());
    let (c2, _, m2) = crate::ckpt::load_auto(&out, Kind::D, 64, &CPU).unwrap();
    assert_eq!(c2.arch.flags(), cfg.arch.flags());
    let (t0, t1) = (crate::ckpt::read(&base).unwrap(), crate::ckpt::read(&out).unwrap());
    assert_eq!(t0.len(), t1.len());
    let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut moved = 0;
    for (k, v) in &t0 {
        if sum.tuned.contains(k) {
            moved += (flat(v) != flat(&t1[k])) as usize;
        } else {
            assert_eq!(flat(v), flat(&t1[k]), "{k} was not tuned");
        }
    }
    assert!(moved > 0);
    // the log: a comment, a header and one row per generation (0 = the start)
    let log = std::fs::read_to_string(&sum.log).unwrap();
    assert_eq!(sum.log, tune::log_path(&out));
    assert_eq!(log.lines().count(), 2 + 3);
    assert!(log.lines().nth(1).unwrap().starts_with("gen\tmembers_mean"));

    // the saved θ scores what the run reported, on the same tune split
    let all = mc::read_items(&items).unwrap();
    let (ti, _) = split(&all, 0.5, 1);
    let tune_items: Vec<Item> = ti.iter().map(|&i| all[i].clone()).collect();
    m2.set_track_activity(false);
    m2.set_inference(true);
    let p = Prepared::new(&byte_bpe(), &tune_items, false, c2.vocab, Norm::Total);
    let g = p.grade(&m2, &CPU, 2048).unwrap();
    assert_eq!(p.fitness(&g, Fitness::Nll, 0.0), sum.tune(), "the saved model is the reported one");

    // the same flags give the same run
    let out2 = dir.join("out").join("again.safetensors");
    let again = run(&parse(&flags("nll", "2", &out2)).unwrap()).unwrap();
    assert_eq!((again.tune(), again.check()), (sum.tune(), sum.check()));
    let t2 = crate::ckpt::read(&out2).unwrap();
    assert!(t1.iter().all(|(k, v)| flat(v) == flat(&t2[k])), "same seed, same tuned model");

    // right − wrong with "I don't know", one generation, chosen weights by pattern
    let out3 = dir.join("out").join("rmw.safetensors");
    let mut f3 = flags("rmw", "1", &out3);
    let at = f3.iter().position(|x| x == "h1").unwrap();
    f3[at] = "b*.parts.holo.lam,b1.hebb".into();
    f3.extend(["--tau".to_string(), "0.05".to_string()]);
    let a3 = parse(&f3).unwrap();
    assert_eq!(a3.tau, 0.05);
    let s3 = run(&a3).unwrap();
    assert!(s3.tuned.contains(&"b0.parts.holo.lam".to_string()) && s3.tuned.iter().any(|n| n.starts_with("b1.hebb.")));
    assert!(s3.rows.iter().all(|r| (-1.0..=1.0).contains(&r.tune)));
    // writing over the input is refused
    assert!(run(&parse(&flags("acc", "1", &base)).unwrap()).unwrap_err().contains("input model"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn es_tune_refuses_sealed_paths_and_bad_flags() {
    let base: Vec<String> = ["--model", "m.safetensors", "--items", "x_dev.tsv", "--params", "h1", "--out", "o.safetensors", "--cpu"].iter().map(|x| x.to_string()).collect();
    let a = parse(&base).unwrap();
    assert_eq!((a.fitness, a.es, a.gens, a.tune_share, a.tau, a.batch), (Fitness::Rmw, EsOpts::default(), 10, 0.5, 0.0, 32768));
    let with = |flag: &str, v: &str| -> Result<tune::Args, String> {
        let mut f = base.clone();
        match f.iter().position(|x| x == flag) {
            Some(i) => f[i + 1] = v.to_string(),
            None => f.extend([flag.to_string(), v.to_string()]),
        }
        parse(&f)
    };
    for (flag, path) in [("--items", "data/nn/sealed/q.tsv"), ("--items", "a_dev.tsv,b_SEALED.tsv"), ("--model", "x/sealed_model.safetensors"), ("--out", "runs/sealed/o.safetensors"), ("--bpe", "Sealed/bpe.txt")] {
        let e = with(flag, path).unwrap_err();
        assert!(e.contains("sealed"), "{flag} {path}: {e}");
    }
    // run refuses too, for settings built in code, before reading anything
    let mut b = a.clone();
    b.items = vec!["nowhere/sealed_x.tsv".into()];
    assert!(run(&b).unwrap_err().contains("sealed"));
    for (flag, v) in [("--fitness", "best"), ("--pop", "0"), ("--sigma", "0"), ("--lr", "-1"), ("--rank", "0"), ("--tune-share", "0"), ("--tune-share", "1.5"), ("--norm", "bits"), ("--tau", "-0.1")] {
        assert!(with(flag, v).is_err(), "{flag} {v}");
    }
    assert!(parse(&["-h".to_string()]).unwrap_err().starts_with("usage"));
    assert!(parse(&base[..6]).is_err(), "--out is required");
    assert_eq!(with("--items", "a.tsv, b.tsv").unwrap().items.len(), 2);
}
