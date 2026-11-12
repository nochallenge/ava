// TESTING / DESIGN ONLY — v40: read a saved model's shape from its file, and start a bigger model from a smaller one.
//
// A file says what it holds by its tensor names: vocab and width from tok_emb, the layer count from the highest
// b{i}, the glance from place_emb, the magnitude code from mag_w, and the engram's block from b{i}.eng.wq. The
// kind (D, A, ...) and the context length are not in the file (D has no position table) and are given.
// `init_from` copies every tensor of a smaller saved model into a bigger one (same names, same shapes); the new
// parts keep their init (place, magnitude and engram read-outs start at 0, so the bigger model starts as the old one).

use crate::model::{Config, Kind, Model, MAG_W, PLACE_EMB, TOK_EMB};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;

/// Every tensor of a .safetensors file, on the CPU.
pub fn read(path: &Path) -> Result<HashMap<String, Tensor>, String> {
    candle_core::safetensors::load(path, &Device::Cpu).map_err(|e| format!("read {}: {e}", path.display()))
}

/// The config a saved file was made with (see the top of this file).
pub fn detect(t: &HashMap<String, Tensor>, kind: Kind, ctx: usize) -> Result<Config, String> {
    let tok = t.get(TOK_EMB).ok_or("the file has no tok_emb")?;
    let (vocab, d) = tok.dims2().map_err(|e| format!("tok_emb: {e}"))?;
    let (mut layers, mut engram) = (0usize, None);
    for name in t.keys() {
        let Some((i, sub)) = name.strip_prefix('b').and_then(|r| r.split_once('.')) else { continue };
        let Ok(i) = i.parse::<usize>() else { continue };
        layers = layers.max(i + 1);
        if sub == "eng.wq" {
            engram = Some(i);
        }
    }
    if layers == 0 {
        return Err("the file has no blocks (b0.*)".into());
    }
    let mut cfg = Config::with_kind(kind, vocab, d, layers, ctx);
    if t.contains_key(PLACE_EMB) {
        cfg = cfg.with_glance()?;
    }
    if t.contains_key(MAG_W) {
        cfg = cfg.with_mag()?;
    }
    if let Some(l) = engram {
        cfg = cfg.with_engram(l)?;
    }
    Ok(cfg)
}

/// Build the model a file describes and load it. Every variable must be in the file and every tensor of the file
/// must be used (a wrong `kind` fails here instead of loading half a model).
pub fn load_auto(path: &Path, kind: Kind, ctx: usize, dev: &Device) -> Result<(Config, VarMap, Model), String> {
    let t = read(path)?;
    let mut cfg = detect(&t, kind, ctx)?;
    // v42: a model trained with block options (src/arch.rs, src/memory.rs) is rebuilt from the arch.txt beside it
    let arch_file = arch_beside(path)?;
    if let Some((mut ar, _)) = arch_file.clone() {
        ar.top = t.contains_key("top_emb"); // R2's order head is saved with the weights but not in the flags
        cfg = cfg.with_arch(ar).map_err(|e| format!("{}: block options: {e}", path.display()))?;
    }
    let mut varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, dev)).map_err(|e| format!("build: {e}"))?;
    let have = varmap.data().lock().unwrap().len();
    if have != t.len() {
        let hint = match &arch_file {
            Some((_, p)) => format!(" (block options from {})", p.display()),
            None => format!(" (a model trained with block options needs its arch.txt beside it: {} or arch.txt in its folder)", arch_name(path).display()),
        };
        return Err(format!("{} holds {} tensors, a {kind:?} model of that shape has {have}{hint}", path.display(), t.len()));
    }
    varmap.load(path).map_err(|e| format!("load {}: {e}", path.display()))?;
    Ok((cfg, varmap, model))
}

/// Lean inference: the same model over `varmap`'s weights as plain tensors (the same storage, nothing copied).
/// Candle keeps every op's inputs alive for a backward whenever one of them is a variable, so a forward of the
/// trained (Var) model holds every intermediate of every block until its logits are dropped (about 2 MB per batched
/// token with every block option on at d128). Over plain tensors each intermediate is freed as soon as it goes out of
/// scope. The same ops on the same values, so the logits are bit-identical. For scoring and generation only: no
/// gradient can reach these weights. Writes into the varmap's weights (Var::set copies in place) are seen; the
/// per-model switches (inference, activity, lesions, int8) are the new model's own and the caller sets them.
pub fn lean(cfg: Config, varmap: &VarMap, dev: &Device) -> Result<Model, String> {
    let ts: HashMap<String, Tensor> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.as_tensor().detach())).collect();
    Model::new(cfg, VarBuilder::from_tensors(ts, DType::F32, dev)).map_err(|e| format!("build lean: {e}"))
}

/// `load_auto`, then the lean inference model over the loaded weights (`lean`); the VarMap is returned too (it holds
/// the storage both models share).
pub fn load_auto_lean(path: &Path, kind: Kind, ctx: usize, dev: &Device) -> Result<(Config, VarMap, Model), String> {
    let (cfg, varmap, _trainable) = load_auto(path, kind, ctx, dev)?;
    let m = lean(cfg, &varmap, dev)?;
    Ok((cfg, varmap, m))
}

/// v42: `NAME.arch.txt` for the model file NAME.safetensors (the file's own block options).
pub fn arch_name(path: &Path) -> std::path::PathBuf {
    path.with_extension("arch.txt")
}

/// v42: the block options a saved model was made with, and the file they came from: `NAME.arch.txt` beside the model
/// file (grow_gpu --out-model writes one), else `arch.txt` in its folder (train_gpu and grow_gpu write that); None
/// when neither exists (today's block). The file holds the flags `Arch::flags` wrote (an empty file: today's block).
pub fn arch_beside(path: &Path) -> Result<Option<(crate::arch::Arch, std::path::PathBuf)>, String> {
    let own = arch_name(path);
    let folder = path.parent().map(|p| p.join("arch.txt"));
    for p in std::iter::once(own).chain(folder) {
        if p.is_file() {
            let line = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            let ar = crate::arch::Arch::from_flags(line.trim()).map_err(|e| format!("{}: {e}", p.display()))?;
            return Ok(Some((ar, p)));
        }
    }
    Ok(None)
}

/// v42: write the block options `ar` for the model file `path` as NAME.arch.txt (nothing for today's block), so a
/// model copied out of its run folder still loads with them.
pub fn write_arch_beside(path: &Path, ar: &crate::arch::Arch) -> Result<(), String> {
    let f = ar.flags();
    if f.is_empty() {
        return Ok(());
    }
    let p = arch_name(path);
    std::fs::write(&p, f + "\n").map_err(|e| format!("{}: {e}", p.display()))
}

/// Start `varmap` from a smaller saved model: each variable whose name is in the file takes its values (the shapes
/// must match); the others keep their init. Every tensor of the file must find its variable. Returns the copied and
/// the kept names, sorted.
pub fn init_from(varmap: &VarMap, path: &Path) -> Result<(Vec<String>, Vec<String>), String> {
    let t = read(path)?;
    let data = varmap.data().lock().unwrap();
    for name in t.keys() {
        if !data.contains_key(name) {
            return Err(format!("{} has {name}, which the new model lacks", path.display()));
        }
    }
    let (mut copied, mut kept) = (Vec::new(), Vec::new());
    for (name, var) in data.iter() {
        match t.get(name) {
            Some(src) => {
                if src.dims() != var.dims() {
                    return Err(format!("{name}: the file has {:?}, the new model {:?}", src.dims(), var.dims()));
                }
                let v = src.to_dtype(DType::F32).and_then(|s| s.to_device(var.device())).map_err(|e| format!("{name}: {e}"))?;
                var.set(&v).map_err(|e| format!("{name}: {e}"))?;
                copied.push(name.clone());
            }
            None => kept.push(name.clone()),
        }
    }
    copied.sort();
    kept.sort();
    Ok((copied, kept))
}

/// `load_auto`, then check the file is what the tool was told (width, layers, glance). An older file loads exactly
/// as before; a v40 file also brings its magnitude code and engram (a plain `varmap.load` would silently skip them).
pub fn load_expect(path: &Path, kind: Kind, ctx: usize, d: usize, layers: usize, glance: bool, dev: &Device) -> Result<(Config, VarMap, Model), String> {
    let (cfg, varmap, model) = load_auto(path, kind, ctx, dev)?;
    if cfg.d != d || cfg.layers != layers || cfg.glance != glance {
        return Err(format!(
            "{} holds d={} layers={} glance={}, but the flags say d={d} layers={layers} glance={glance}",
            path.display(),
            cfg.d,
            cfg.layers,
            cfg.glance
        ));
    }
    if cfg.mag || cfg.engram.is_some() {
        println!("{}: v40 model (magnitude code {}, engram {:?})", path.display(), cfg.mag, cfg.engram);
    }
    Ok((cfg, varmap, model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::Arch;
    use crate::state::tests::{build_awake, max_diff, one_pass, tokens, v40_cfg};
    use crate::testkit::scratch;

    /// v42: a model with block options saved with its arch.txt loads back as the same model (same logits); without
    /// the arch.txt the load fails and says where the file belongs; a folder arch.txt is read when NAME.arch.txt is absent.
    #[test]
    fn block_options_load_from_the_arch_file_beside_the_model() {
        let ar = Arch::from_flags("--dendrites 2 --pk-mem 16 --pk-topk 2 --eng-temp --btsp 4").unwrap();
        let cfg = v40_cfg().with_arch(ar).unwrap();
        let (vm, m) = build_awake(cfg, 11);
        let toks = tokens(2, 24, cfg.vocab, 5);
        let want = one_pass(&m, &toks, 2, 24);
        let dir = scratch("ckpt_arch");
        let path = dir.join("brain.safetensors");
        vm.save(&path).unwrap();
        // no arch.txt: refused, with the hint
        let err = load_auto(&path, Kind::D, 64, &Device::Cpu).err().unwrap();
        assert!(err.contains("brain.arch.txt"), "{err}");
        // NAME.arch.txt
        write_arch_beside(&path, &ar).unwrap();
        let (c2, _, m2) = load_auto(&path, Kind::D, 64, &Device::Cpu).unwrap();
        assert_eq!(c2.arch.signature(), ar.signature());
        assert_eq!(max_diff(&want, &one_pass(&m2, &toks, 2, 24)), 0.0);
        // the folder's arch.txt
        std::fs::rename(arch_name(&path), dir.join("arch.txt")).unwrap();
        let (_, _, m3) = load_auto(&path, Kind::D, 64, &Device::Cpu).unwrap();
        assert_eq!(max_diff(&want, &one_pass(&m3, &toks, 2, 24)), 0.0);
        // today's block writes nothing
        let plain = dir.join("plain.safetensors");
        write_arch_beside(&plain, &Arch::default()).unwrap();
        assert!(!arch_name(&plain).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every block option and every Wave 5 part at once (as src/int8/tests.rs `every`).
    const EVERY: &str = "--resonate 16 --modulate --tem 4 --halting 3 --cerebellum 64 --workspace 4 --holo 8 --sync 4 --jepa 0.5 --diffusion 0.5 --critic 0.5 \
        --confidence 0.5 --topo 0.5 --schema 0.5 --dendrites 4 --stp --delays 4 --wm 8 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 \
        --residual dense --resid-fix --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";

    /// Lean inference: over plain tensors the model gives its Var twin's logits bit for bit, in one pass (f32 and
    /// int8), in pieces with carried state and token by token; a later write into the VarMap reaches both; and the
    /// forward keeps no graph (its logits track no op).
    #[test]
    fn the_lean_model_gives_the_same_logits_bit_for_bit() {
        let all = Config::with_kind(Kind::D, 258, 32, 2, 64).with_arch(Arch::from_flags(EVERY).unwrap()).unwrap();
        let p64 = all.with_arch(Arch::from_flags(&EVERY.replace("--tem 4 ", "").replace("--gates channel", "--head-size 16")).unwrap()).unwrap();
        for (name, cfg, seed) in [("everything on", all, 14u64), ("everything on, no tem, head 16", p64, 15), ("D+glance+mag+engram", v40_cfg(), 5)] {
            let (vm, m) = build_awake(cfg, seed);
            let lm = lean(cfg, &vm, &Device::Cpu).unwrap();
            lm.set_track_activity(false);
            let (b, t) = (3, 48);
            let toks = tokens(b, t, cfg.vocab, 71);
            let want = one_pass(&m, &toks, b, t);
            assert!(want.track_op(), "{name}: the Var model's logits keep their graph");
            let got = one_pass(&lm, &toks, b, t);
            assert!(!got.track_op(), "{name}: the lean model's logits keep a graph");
            assert_eq!(max_diff(&want, &got), 0.0, "{name}: one pass");
            let mut sa = m.state(b).unwrap();
            let mut sb = lm.state(b).unwrap();
            let pa = crate::state::tests::in_pieces(&m, &mut sa, &toks, b, t, &[16, 32]);
            assert_eq!(max_diff(&pa, &crate::state::tests::in_pieces(&lm, &mut sb, &toks, b, t, &[16, 32])), 0.0, "{name}: pieces");
            let (mut st_a, mut st_b) = (m.stream().unwrap(), lm.stream().unwrap());
            for &tok in &toks[..12] {
                assert_eq!(max_diff(&m.step(&mut st_a, tok).unwrap(), &lm.step(&mut st_b, tok).unwrap()), 0.0, "{name}: step");
            }
            m.set_int8(true).unwrap();
            lm.set_int8(true).unwrap();
            assert_eq!(max_diff(&one_pass(&m, &toks, b, t), &one_pass(&lm, &toks, b, t)), 0.0, "{name}: int8");
            m.set_int8(false).unwrap();
            lm.set_int8(false).unwrap();
            // the storage is shared: a write into the VarMap is seen by both
            {
                let data = vm.data().lock().unwrap();
                let w = &data["tok_emb"];
                w.set(&w.as_tensor().affine(1.01, 0.0).unwrap()).unwrap();
            }
            let moved = one_pass(&m, &toks, b, t);
            assert!(max_diff(&moved, &want) > 0.0, "{name}: the write moved nothing");
            assert_eq!(max_diff(&moved, &one_pass(&lm, &toks, b, t)), 0.0, "{name}: after the write");
        }
    }
}
