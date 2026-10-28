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
    let cfg = detect(&t, kind, ctx)?;
    let mut varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, dev)).map_err(|e| format!("build: {e}"))?;
    let have = varmap.data().lock().unwrap().len();
    if have != t.len() {
        return Err(format!("{} holds {} tensors, a {kind:?} model of that shape has {have}", path.display(), t.len()));
    }
    varmap.load(path).map_err(|e| format!("load {}: {e}", path.display()))?;
    Ok((cfg, varmap, model))
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
