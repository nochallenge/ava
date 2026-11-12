// TESTING / DESIGN ONLY — v52 fasteval (runs/v52/fasteval.md): the scoring tools' fused forward.
//
// Three recipes at d128, 3 blocks, ctx 64, batch 4 (the weights that start at 0 moved off it, as tests/fast_winners.rs):
//   A03  Ava 0.3's: today's block D, glance, magnitude code, --engram 1          (Model::logits_a)
//   A04b Ava 0.4 / 0.4b's: + WIN + SLOW (runs/v49/ava04b/grow/arch.txt)         (Model::logits_win)
//   V51F v51's final brain: A04b + --letters                                     (Model::logits_win)
// * CPU (always): the fused kernels are CUDA only, so with the switch on every scorer falls back to the reference
//   forward and gives its numbers bit for bit (eval::score_logits, seq_nll, evaluate); the reason is given.
// * CUDA (AVA_GPU_TEST_CUDA=1):
//   - the fused forward IS the training step's forward: logits_win = the f32 step's logits (fwin::capture_logits),
//     and for A03 the step's f32 loss = the mean of ce_fwd_raw over logits_a; both bit for bit;
//   - against the reference forward (Model::forward, inference on and off): rel-norm difference of the logits, max
//     |difference|, the per-row NLL's max |difference| printed; bar: rel-norm < 1e-4 (f32 rounding in different
//     summation orders, plus the odd spike it flips);
//   - routing: models with options the fused WIN step does not take (--loop 2, --dale 0.8) are refused with a reason.

use ava_gpu::arch::Arch;
use ava_gpu::eval::{evaluate, score_logits, seq_nll, set_fused};
use ava_gpu::model::{fwin, Config, Kind, Mm, Model};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};

const WIN_SLOW: &str = "--head-size 64 --timescales fixed --gates channel --value-residual --kwta 0.25 --mlp-act relu2 \
                        --mem-momentum 0.5 --resid-fix --residual dense --lif-init logspaced --lif-scale --spikes int2 --alif \
                        --homeostasis 0.1";

fn cuda() -> Option<Device> {
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v == "1").unwrap_or(false) {
        Some(Device::new_cuda(0).expect("CUDA device 0"))
    } else {
        None
    }
}

fn arch_of(flags: &str) -> Arch {
    let args: Vec<String> = flags.split_whitespace().map(String::from).collect();
    let mut a = Arch::default();
    let mut i = 0;
    while i < args.len() {
        i += a.take_flag(&args, i).unwrap().unwrap_or_else(|| panic!("not a block flag: {}", args[i]));
    }
    a
}

fn noise(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * amp
        })
        .collect()
}

fn vars(varmap: &VarMap) -> Vec<(String, Var)> {
    let mut v: Vec<(String, Var)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    v.sort_by(|x, y| x.0.cmp(&y.0));
    v
}

/// tests/fast_winners.rs's perturb: the weights that start at 0 (and the thresholds, gains) moved off their start.
fn perturb(varmap: &VarMap, dev: &Device) {
    for (k, (name, var)) in vars(varmap).into_iter().enumerate() {
        let leaf = name.rsplit('.').next().unwrap_or(&name);
        let leaf = if leaf == "wo" && name.contains(".eng.") { "eng_wo" } else { leaf };
        let t = var.as_tensor();
        let amp = match leaf {
            "dense" => 0.1,
            "vres" => 0.3,
            "place_emb" => 0.05,
            "wa" | "wb" | "wa2" | "wb2" | "we" | "eng_wo" | "mag_w" => 0.05,
            "theta" | "again" => {
                let v = Tensor::from_vec(noise(t.elem_count(), 1000 + k as u64, 0.3), t.shape(), dev).unwrap();
                var.set(&t.add(&v).unwrap()).unwrap();
                continue;
            }
            "ba" | "bb" => 0.5,
            _ => continue,
        };
        var.set(&Tensor::from_vec(noise(t.elem_count(), 1000 + k as u64, amp), t.shape(), dev).unwrap()).unwrap();
    }
}

/// (tag, config) of the three recipes.
fn recipes() -> Vec<(&'static str, Config)> {
    let g = ava_data::numbers::VOCAB_GLANCE;
    let base = || Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_mag().unwrap().with_engram(1).unwrap();
    vec![
        ("A03", base()),
        ("A04b", base().with_arch(arch_of(WIN_SLOW)).unwrap()),
        ("V51F", base().with_arch(arch_of(&format!("{WIN_SLOW} --letters"))).unwrap()),
    ]
}

fn model(dev: &Device, cfg: Config, seed: u64) -> Model {
    if dev.is_cuda() {
        dev.set_seed(seed).unwrap();
    }
    let varmap = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, dev)).unwrap();
    perturb(&varmap, dev);
    std::mem::forget(varmap); // the model holds the same Vars; the map is not needed again
    m.set_track_activity(false);
    m
}

/// Glance-vocabulary streams (plain tokens, digits with their length marks), as tests/fast_winners.rs's batch.
fn stream(len: usize, seed: u64) -> Vec<u32> {
    use ava_data::numbers::insert_marks;
    let mut rng = ava_data::rng::SplitMix64::new(seed);
    let mut out = Vec::with_capacity(len + 32);
    while out.len() < len {
        match rng.below(10) {
            0..=4 => out.push(300 + rng.below(256) as u32),
            5..=7 => {
                let l = 1 + rng.below(6) as usize;
                let digits: Vec<u32> = (0..l).map(|_| 48 + rng.below(10) as u32).collect();
                out.extend(insert_marks(&digits));
            }
            8 => out.push(48 + rng.below(10) as u32),
            _ => out.push(4096 + rng.below(16) as u32),
        }
    }
    out.truncate(len);
    out
}

fn batch(dev: &Device, b: usize, t: usize, seed: u64) -> (Tensor, Tensor) {
    let (mut ids, mut tg) = (Vec::with_capacity(b * t), Vec::with_capacity(b * t));
    for r in 0..b {
        let s = stream(t + 1, seed + r as u64);
        ids.extend_from_slice(&s[..t]);
        tg.extend_from_slice(&s[1..]);
    }
    (Tensor::from_vec(ids, (b, t), dev).unwrap(), Tensor::from_vec(tg, (b, t), dev).unwrap())
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap()
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// (rel-norm difference, max |difference|) of a against the reference r, in f64.
fn diff(a: &[f32], r: &[f32]) -> (f64, f64) {
    let (mut nd, mut nr, mut mx) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(r) {
        let d = x as f64 - y as f64;
        nd += d * d;
        nr += y as f64 * y as f64;
        mx = mx.max(d.abs());
    }
    (nd.sqrt() / nr.sqrt().max(1e-300), mx)
}

#[test]
fn cpu_scoring_falls_back_bit_for_bit() {
    // the only test here that moves the process-wide switch (eval::set_fused): the tests of a file share it
    let cpu = Device::Cpu;
    for (tag, cfg) in recipes() {
        let m = model(&cpu, cfg, 5);
        let why = m.fused_eval_why(&cpu).unwrap_or_else(|| panic!("{tag}: the CPU must fall back"));
        assert!(why.contains("CUDA only"), "{tag}: {why}");
        assert!(m.logits_fused(&batch(&cpu, 1, 8, 1).0).is_err(), "{tag}: logits_fused must refuse the CPU");
        let (ids, _) = batch(&cpu, 4, cfg.ctx, 31);
        // sequences of uneven lengths (seq_nll pads), and ctx + 1 windows (evaluate)
        let seqs: Vec<Vec<u32>> = (0..9).map(|k| stream(5 + 7 * k, 100 + k as u64)).collect();
        let windows: Vec<Vec<u32>> = (0..6).map(|k| stream(cfg.ctx + 1, 200 + k as u64)).collect();
        let bl = vec![1u32; cfg.vocab];
        let mut got = Vec::new();
        for on in [false, true] {
            set_fused(on);
            let lg = flat(&score_logits(&m, &ids).unwrap());
            let nll = seq_nll(&m, &cpu, &seqs, 4).unwrap();
            let (acc, _) = evaluate(&m, windows.clone(), 4, 300, &bl, &cpu).unwrap();
            got.push((lg, nll, acc));
        }
        set_fused(false);
        let reference = flat(&m.forward(&ids).unwrap());
        assert!(bits_equal(&got[0].0, &reference), "{tag}: the switch off must be Model::forward");
        assert!(bits_equal(&got[1].0, &got[0].0), "{tag}: CPU logits differ with the switch on");
        assert!(got[1].1.iter().zip(&got[0].1).all(|(a, b)| a.to_bits() == b.to_bits()), "{tag}: seq_nll differs");
        assert_eq!(got[1].2, got[0].2, "{tag}: evaluate differs");
        eprintln!("CPU {tag}: falls back ({why}); logits, seq_nll and evaluate bit for bit (BPB {:.9})", got[0].2.bpb());
    }
}

#[test]
fn fused_forward_is_the_step_forward() {
    let Some(dev) = cuda() else {
        eprintln!("fused forward: skipped (set AVA_GPU_TEST_CUDA=1)");
        return;
    };
    let f32mm = Mm { bf16: false };
    let mut bad: Vec<String> = Vec::new();
    for (k, (tag, cfg)) in recipes().into_iter().enumerate() {
        let m = model(&dev, cfg, 7 + k as u64);
        assert_eq!(m.fused_eval_why(&dev), None, "{tag}: the fused forward must take this model");
        let (ids, tg) = batch(&dev, 4, cfg.ctx, 41 + k as u64);
        let lf = m.logits_fused(&ids).unwrap();
        let f = flat(&lf);
        // (a) the training step's own forward, f32
        if cfg.arch == Arch::default() {
            let (loss, _, _) = m.loss_and_grads_a(&ids, &tg, &f32mm).unwrap();
            let ce = ava_gpu::lif::ce_fwd_raw(&lf, &tg.flatten_all().unwrap()).unwrap();
            let mine = ce.narrow(1, 0, 1).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
            let step = loss.to_scalar::<f32>().unwrap();
            eprintln!("{tag}: step loss {step:.9}, mean ce of logits_a {mine:.9}");
            if mine.to_bits() != step.to_bits() {
                bad.push(format!("{tag}: logits_a's loss {mine} != the step's {step}"));
            }
        } else {
            fwin::capture_logits(true);
            m.loss_and_grads_win(&ids, &tg, None, false, &f32mm).unwrap();
            let ls = fwin::take_logits().expect("the step's logits");
            fwin::capture_logits(false);
            let same = bits_equal(&f, &flat(&ls));
            eprintln!("{tag}: logits_win vs the f32 step's logits: {}", if same { "bit for bit" } else { "DIFFER" });
            if !same {
                bad.push(format!("{tag}: logits_win differs from the step's logits"));
            }
        }
        // (b) against the reference forward
        for infer in [false, true] {
            m.set_inference(infer);
            let r = flat(&m.forward(&ids).unwrap());
            let (rel, mx) = diff(&f, &r);
            // the per-row NLL of the targets, as the scorers read it
            let nll = |lg: &[f32]| -> Vec<f64> {
                let v = cfg.vocab;
                let t: Vec<u32> = tg.flatten_all().unwrap().to_vec1().unwrap();
                (0..t.len())
                    .map(|row| {
                        let x = &lg[row * v..(row + 1) * v];
                        let mxx = x.iter().fold(f64::MIN, |a, &b| a.max(b as f64));
                        let lse = mxx + x.iter().map(|&y| (y as f64 - mxx).exp()).sum::<f64>().ln();
                        lse - x[t[row] as usize] as f64
                    })
                    .collect()
            };
            let (nf, nr) = (nll(&f), nll(&r));
            let dn = nf.iter().zip(&nr).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            let mean = nf.iter().zip(&nr).map(|(a, b)| a - b).sum::<f64>() / nf.len() as f64;
            eprintln!("{tag} vs Model::forward (inference {infer}): logits rel {rel:.2e}, max |d| {mx:.2e}; target NLL max |d| {dn:.2e}, mean d {mean:+.2e}");
            if rel >= 1e-4 {
                bad.push(format!("{tag} (inference {infer}): logits rel {rel:.3e} >= 1e-4"));
            }
        }
        m.set_inference(false);
    }
    // routing: options the fused WIN step does not take keep the reference forward, with the reason
    let g = ava_data::numbers::VOCAB_GLANCE;
    for extra in ["--loop 2", "--dale 0.8"] {
        if let Ok(c) = Config::with_kind(Kind::D, g, 128, 3, 64).with_glance().unwrap().with_arch(arch_of(&format!("{WIN_SLOW} {extra}"))) {
            let m = model(&dev, c, 3);
            let why = m.fused_eval_why(&dev);
            eprintln!("routing {extra}: {why:?}");
            assert!(why.is_some(), "{extra} must keep the reference forward");
        }
    }
    assert!(bad.is_empty(), "fused forward: {bad:#?}");
}
