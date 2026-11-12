// TESTING / DESIGN ONLY — merge_gpu: C13, merging brains tuned from the same base into one, without training.
//
// merge_gpu --base FILE --tuned A,B[,C..] --out FILE [--method ties|task|average] [--lambda 1.0] [--keep 0.2]
//           [--threads N]
//
// For example, a copy of Ava that practised numbers and a copy that practised conversation become one that keeps
// both, a cheap form of consolidation. Each saved tensor is merged name by name with the part's functions
// (ava_parts::merge):
// - average: the tuned models' mean (model soups, Wortsman et al. 2022); the base is only checked;
// - task: base + λ·Σ (tuned_i − base) (task arithmetic, Ilharco et al. 2023);
// - ties (the default): each task vector trimmed to its largest `keep` share, one sign elected per weight, and only
//   the agreeing values averaged (Yadav et al. 2023). The trim is per tensor, not across the whole model.
// Every model must have the same tensor names, shapes and block options (arch.txt), or it is refused. A tensor that
// is not f32 must be the same in every model and is copied. The block options go beside the output as NAME.arch.txt.
// Tensors are merged on every core (they are independent). Only f32 host code: no GPU is used.

use ava_gpu::ckpt;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Method {
    Average,
    Task,
    Ties,
}

#[derive(Clone, Copy, Debug)]
struct Opts {
    method: Method,
    lambda: f32,
    keep: f32,
}

fn usage() -> ! {
    eprintln!("merge_gpu --base FILE --tuned A,B[,C..] --out FILE [--method ties|task|average] [--lambda 1.0] [--keep 0.2] [--threads N]");
    std::process::exit(2)
}

/// One tensor merged: base and the tuned copies as flat f32.
fn merge_one(o: Opts, base: &[f32], tuned: &[&[f32]]) -> Vec<f32> {
    match o.method {
        Method::Average => ava_parts::merge::average(tuned),
        Method::Task => ava_parts::merge::task_arithmetic(base, tuned, o.lambda),
        Method::Ties => ava_parts::merge::ties(base, tuned, o.keep, o.lambda),
    }
}

/// Merges the tensors of `base` and `tuned` (same names, shapes and dtypes, else Err), on `threads` cores. Returns the
/// merged tensors and how many were merged (the rest are non-f32 copies).
fn merge_all(o: Opts, base: &HashMap<String, Tensor>, tuned: &[HashMap<String, Tensor>], threads: usize) -> Result<(HashMap<String, Tensor>, usize), String> {
    let mut names: Vec<&String> = base.keys().collect();
    names.sort();
    for (k, t) in tuned.iter().enumerate() {
        let mut tn: Vec<&String> = t.keys().collect();
        tn.sort();
        if tn != names {
            let only_base: Vec<_> = names.iter().filter(|n| !t.contains_key(n.as_str())).take(5).collect();
            let only_t: Vec<_> = tn.iter().filter(|n| !base.contains_key(n.as_str())).take(5).collect();
            return Err(format!("tuned model {k} has other tensors (only in the base: {only_base:?}; only in it: {only_t:?})"));
        }
        for n in &names {
            let (a, b) = (&base[n.as_str()], &t[n.as_str()]);
            if a.dims() != b.dims() || a.dtype() != b.dtype() {
                return Err(format!("{n}: base {:?} {:?}, tuned model {k} {:?} {:?}", a.dims(), a.dtype(), b.dims(), b.dtype()));
            }
        }
    }
    let flat = |t: &Tensor| t.flatten_all().and_then(|x| x.to_vec1::<f32>()).map_err(|e| e.to_string());
    let work: Vec<&String> = names.iter().copied().filter(|n| base[n.as_str()].dtype() == DType::F32).collect();
    let mut out: HashMap<String, Tensor> = HashMap::new();
    for n in names.iter().filter(|n| base[n.as_str()].dtype() != DType::F32) {
        for (k, t) in tuned.iter().enumerate() {
            let same = base[n.as_str()].to_dtype(DType::F32).and_then(|a| t[n.as_str()].to_dtype(DType::F32).map(|b| (a, b))).and_then(|(a, b)| (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()).map_err(|e| e.to_string())?;
            if same != 0.0 {
                return Err(format!("{n} is not f32 and differs in tuned model {k}: it cannot be merged"));
            }
        }
        out.insert((*n).clone(), base[n.as_str()].clone());
    }
    let threads = threads.max(1).min(work.len().max(1));
    let results: Vec<Result<Vec<(String, Tensor)>, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|w| {
                let work = &work;
                s.spawn(move || -> Result<Vec<(String, Tensor)>, String> {
                    let mut got = Vec::new();
                    for n in work.iter().skip(w).step_by(threads) {
                        let b = &base[n.as_str()];
                        let bf = flat(b)?;
                        let tf: Vec<Vec<f32>> = tuned.iter().map(|t| flat(&t[n.as_str()])).collect::<Result<_, _>>()?;
                        let refs: Vec<&[f32]> = tf.iter().map(Vec::as_slice).collect();
                        let m = merge_one(o, &bf, &refs);
                        got.push(((*n).clone(), Tensor::from_vec(m, b.dims(), &Device::Cpu).map_err(|e| e.to_string())?));
                    }
                    Ok(got)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("a merge thread panicked".into()))).collect()
    });
    for r in results {
        out.extend(r?);
    }
    Ok((out, work.len()))
}

/// The block options of every model must be the same (None = today's block).
fn same_arch(base: &Path, tuned: &[PathBuf]) -> Result<Option<ava_gpu::arch::Arch>, String> {
    let a = ckpt::arch_beside(base)?.map(|(a, _)| a);
    let fa = a.as_ref().map(|x| x.flags()).unwrap_or_default();
    for t in tuned {
        let ft = ckpt::arch_beside(t)?.map(|(x, _)| x.flags()).unwrap_or_default();
        if ft != fa {
            return Err(format!("{}: block options {ft:?}, the base's are {fa:?}", t.display()));
        }
    }
    Ok(a)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut base, mut tuned, mut out) = (None, Vec::new(), None);
    let mut o = Opts { method: Method::Ties, lambda: 1.0, keep: 0.2 };
    let mut threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    let mut i = 0;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_else(|| usage());
        match args[i].as_str() {
            "--base" => base = Some(PathBuf::from(v)),
            "--tuned" => tuned = v.split(',').filter(|s| !s.is_empty()).map(PathBuf::from).collect(),
            "--out" => out = Some(PathBuf::from(v)),
            "--method" => {
                o.method = match v.as_str() {
                    "ties" => Method::Ties,
                    "task" => Method::Task,
                    "average" => Method::Average,
                    _ => usage(),
                }
            }
            "--lambda" => o.lambda = v.parse().unwrap_or_else(|_| usage()),
            "--keep" => o.keep = v.parse().unwrap_or_else(|_| usage()),
            "--threads" => threads = v.parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
        i += 2;
    }
    let (Some(base), Some(out)) = (base, out) else { usage() };
    if tuned.is_empty() || !(o.keep > 0.0 && o.keep <= 1.0) || !o.lambda.is_finite() {
        eprintln!("merge_gpu: needs at least one --tuned model, --keep in (0, 1] and a finite --lambda");
        std::process::exit(2);
    }
    let run = || -> Result<(), String> {
        let t0 = std::time::Instant::now();
        let ar = same_arch(&base, &tuned)?;
        let b = ckpt::read(&base)?;
        let ts: Vec<HashMap<String, Tensor>> = tuned.iter().map(|p| ckpt::read(p)).collect::<Result<_, _>>()?;
        let weights: usize = b.values().map(Tensor::elem_count).sum();
        let (m, n) = merge_all(o, &b, &ts, threads)?;
        candle_core::safetensors::save(&m, &out).map_err(|e| format!("write {}: {e}", out.display()))?;
        if let Some(ar) = &ar {
            ckpt::write_arch_beside(&out, ar)?;
        }
        let s = t0.elapsed().as_secs_f64();
        println!("merged {n} tensors ({weights} weights) of {} tuned models by {:?} (lambda {}, keep {}) in {s:.2} s ({:.1}M weights/s, {threads} threads) -> {}", ts.len(), o.method, o.lambda, o.keep, weights as f64 * (ts.len() + 1) as f64 / s / 1e6, out.display());
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("merge_gpu: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ava_gpu::model::{Config, Kind, Model};
    use candle_nn::{VarBuilder, VarMap};

    fn tiny(flags: &str) -> (VarMap, Model) {
        let cfg = Config::with_kind(Kind::D, 258, 32, 2, 64).with_arch(ava_gpu::arch::Arch::from_flags(flags).unwrap()).unwrap();
        let vm = VarMap::new();
        let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu)).unwrap();
        (vm, m)
    }

    fn tensors(vm: &VarMap) -> HashMap<String, Tensor> {
        vm.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.as_tensor().copy().unwrap())).collect()
    }

    /// base + small noise on every tensor (a stand-in for a tuned copy).
    fn nudged(b: &HashMap<String, Tensor>, seed: u64) -> HashMap<String, Tensor> {
        b.iter()
            .map(|(k, t)| {
                let n = ava_parts::randn_seeded(t.elem_count(), seed + k.len() as u64, &Device::Cpu).unwrap().reshape(t.dims()).unwrap();
                (k.clone(), (t + (n * 0.01).unwrap()).unwrap())
            })
            .collect()
    }

    fn o(method: Method) -> Opts {
        Opts { method, lambda: 1.0, keep: 0.2 }
    }

    fn diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
    }

    /// Merging copies of one model gives that model, by every method; threads do not change the result.
    #[test]
    fn copies_merge_to_themselves_and_threads_agree() {
        let (vm, _) = tiny("");
        let b = tensors(&vm);
        for m in [Method::Average, Method::Task, Method::Ties] {
            let (got, n) = merge_all(o(m), &b, &[b.clone(), b.clone()], 3).unwrap();
            assert_eq!(n, b.len());
            for (k, t) in &b {
                assert_eq!(diff(&got[k], t), 0.0, "{m:?} {k}");
            }
        }
        let (x, y) = (nudged(&b, 1), nudged(&b, 2));
        let (one, _) = merge_all(o(Method::Ties), &b, &[x.clone(), y.clone()], 1).unwrap();
        let (many, _) = merge_all(o(Method::Ties), &b, &[x, y], 8).unwrap();
        for k in b.keys() {
            assert_eq!(diff(&one[k], &many[k]), 0.0, "{k}");
        }
    }

    /// Task arithmetic is base + Σ (tuned − base), and TIES equals the part's function on each tensor.
    #[test]
    fn task_and_ties_are_the_parts_rules() {
        let (vm, _) = tiny("");
        let b = tensors(&vm);
        let (x, y) = (nudged(&b, 3), nudged(&b, 4));
        let (task, _) = merge_all(o(Method::Task), &b, &[x.clone(), y.clone()], 4).unwrap();
        let (ties, _) = merge_all(o(Method::Ties), &b, &[x.clone(), y.clone()], 4).unwrap();
        let f = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for k in b.keys() {
            let want = ((&x[k] + &y[k]).unwrap() - &b[k]).unwrap();
            assert!(diff(&task[k], &want) < 1e-6, "task {k}");
            let w = ava_parts::merge::ties(&f(&b[k]), &[&f(&x[k]), &f(&y[k])], 0.2, 1.0);
            assert_eq!(f(&ties[k]), w, "ties {k}");
        }
    }

    /// A merged model with block options saves, loads back with its arch file and gives finite logits; mismatched
    /// models are refused.
    #[test]
    fn merged_model_loads_and_mismatches_are_refused() {
        let flags = "--kwta 0.25 --mlp-act relu2 --holo 8 --sync 4";
        let (vm, _) = tiny(flags);
        let b = tensors(&vm);
        let dir = std::env::temp_dir().join(format!("merge_gpu_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ar = ava_gpu::arch::Arch::from_flags(flags).unwrap();
        let save = |name: &str, t: &HashMap<String, Tensor>| -> PathBuf {
            let p = dir.join(name);
            candle_core::safetensors::save(t, &p).unwrap();
            ckpt::write_arch_beside(&p, &ar).unwrap();
            p
        };
        let pb = save("base.safetensors", &b);
        let px = save("x.safetensors", &nudged(&b, 5));
        let py = save("y.safetensors", &nudged(&b, 6));
        assert!(same_arch(&pb, &[px.clone(), py.clone()]).unwrap().is_some());
        let (m, _) = merge_all(o(Method::Ties), &ckpt::read(&pb).unwrap(), &[ckpt::read(&px).unwrap(), ckpt::read(&py).unwrap()], 4).unwrap();
        let out = save("merged.safetensors", &m);
        let (_, _, model) = ckpt::load_auto(&out, Kind::D, 64, &Device::Cpu).unwrap();
        let ids = Tensor::from_vec((0..40u32).map(|i| i * 5 % 250).collect::<Vec<_>>(), (1, 40), &Device::Cpu).unwrap();
        let lg = model.forward(&ids).unwrap();
        assert!(lg.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|x| x.is_finite()));
        // a different arch, a missing tensor and a different shape are refused
        let (vm2, _) = tiny("--kwta 0.25 --mlp-act relu2");
        let p2 = dir.join("other.safetensors");
        candle_core::safetensors::save(&tensors(&vm2), &p2).unwrap();
        assert!(same_arch(&pb, &[p2]).is_err());
        let mut miss = b.clone();
        let k0 = miss.keys().next().unwrap().clone();
        miss.remove(&k0);
        assert!(merge_all(o(Method::Ties), &b, &[miss], 2).is_err());
        let mut bad = b.clone();
        bad.insert(k0.clone(), Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap());
        assert!(merge_all(o(Method::Ties), &b, &[bad], 2).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
