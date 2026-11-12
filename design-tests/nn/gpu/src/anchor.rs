// TESTING / DESIGN ONLY — ava_gpu::anchor: v41 synaptic consolidation (like EWC, and like the brain keeping the
// synapses an old skill leans on).
//
// Before sleep or reward, each weight's importance to her usual reading is measured: F = the mean of g² over a few
// reading batches (g = the batch's reading gradient), divided by the mean of F over every weight (so F averages 1).
// After every optimizer step each weight is pulled back toward where it started:
//     w ← w − min(lr · strength · F, 1) · (w − w0)
// Weights her reading leans on hardly move; the others learn freely. A decoupled pull, like AdamW's weight decay,
// so it does not touch the Adam moments.
//
// v42 (U3, the running anchor): an anchor can be saved (`save`: w0 and F by parameter name) and read back by the
// next phase (`read`), which combines it with its own fresh measure (`combine`): F = γ·F_old + F_new, the online form
// of EWC, so what earlier nights protected stays protected. The anchor point w0 is the weights the phase starts
// from (online EWC), or with `saved_point` the saved w0 (a fixed point such as the serving copy).

use crate::optim::Param;
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result, Tensor, Var};
use std::collections::HashMap;
use std::path::Path;

/// Running sums of g² per parameter (None: the parameter got no gradient yet).
pub struct Importance {
    names: Vec<String>,
    vars: Vec<Var>,
    acc: Vec<Option<Tensor>>,
    batches: usize,
}

impl Importance {
    pub fn new(params: &[Param]) -> Importance {
        Importance {
            names: params.iter().map(|p| p.name.clone()).collect(),
            vars: params.iter().map(|p| p.var.clone()).collect(),
            acc: vec![None; params.len()],
            batches: 0,
        }
    }

    /// Add one batch's gradient.
    pub fn add(&mut self, grads: &GradStore) -> Result<()> {
        for (k, v) in self.vars.iter().enumerate() {
            let Some(g) = grads.get(v.as_tensor()) else { continue };
            let sq = g.to_dtype(DType::F32)?.sqr()?;
            self.acc[k] = Some(match self.acc[k].take() {
                Some(s) => s.add(&sq)?,
                None => sq,
            });
        }
        self.batches += 1;
        Ok(())
    }

    /// v42: the number of batches added so far.
    pub fn batches(&self) -> usize {
        self.batches
    }

    /// The anchor at `strength`, holding every parameter that got a gradient at its current value. Returns None when
    /// no gradient was seen or the mean of g² is 0 or not finite.
    pub fn anchor(self, strength: f64) -> Result<Option<Anchor>> {
        let (mut sum, mut count) = (0f64, 0usize);
        for x in self.acc.iter().flatten() {
            sum += x.sum_all()?.to_scalar::<f32>()? as f64;
            count += x.elem_count();
        }
        let mean = sum / count.max(1) as f64;
        if count == 0 || !(mean.is_finite() && mean > 0.0) {
            return Ok(None);
        }
        let (mut names, mut vars, mut w0, mut imp) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for ((n, v), x) in self.names.into_iter().zip(self.vars).zip(self.acc) {
            let Some(x) = x else { continue }; // a weight her reading never touches is free
            imp.push(x.affine(1.0 / mean, 0.0)?.to_dtype(v.as_tensor().dtype())?);
            w0.push(v.as_tensor().copy()?);
            vars.push(v);
            names.push(n);
        }
        Ok(Some(Anchor { names, vars, w0, imp, strength, weights: count }))
    }
}

pub struct Anchor {
    names: Vec<String>,
    vars: Vec<Var>,
    w0: Vec<Tensor>,
    imp: Vec<Tensor>,
    pub strength: f64,
    /// the number of anchored weights
    pub weights: usize,
}

/// v42: an anchor read back from a file (`Anchor::save`), by parameter name.
pub struct SavedAnchor {
    pub w0: HashMap<String, Tensor>,
    pub imp: HashMap<String, Tensor>,
    pub strength: f64,
}

impl Anchor {
    /// The pull after an optimizer step at learning rate `lr`.
    pub fn pull(&self, lr: f64) -> Result<()> {
        let s = lr * self.strength;
        for ((v, w0), f) in self.vars.iter().zip(&self.w0).zip(&self.imp) {
            let w = v.as_tensor();
            let c = f.affine(s, 0.0)?.clamp(0f32, 1f32)?;
            let next = w.sub(&c.mul(&w.sub(w0)?)?)?;
            v.set(&next)?;
        }
        Ok(())
    }

    /// The importance-weighted mean squared distance from where she started (logged: how far she has moved).
    pub fn drift(&self) -> Result<f64> {
        let (mut s, mut n) = (0f64, 0usize);
        for ((v, w0), f) in self.vars.iter().zip(&self.w0).zip(&self.imp) {
            let d = v.as_tensor().sub(w0)?.sqr()?.mul(f)?.to_dtype(DType::F32)?;
            s += d.sum_all()?.to_scalar::<f32>()? as f64;
            n += d.elem_count();
        }
        Ok(s / n.max(1) as f64)
    }

    /// Each anchored parameter's importance (in the order of the parameters that got a gradient).
    pub fn importance(&self) -> &[Tensor] {
        &self.imp
    }

    /// v42: the anchored parameters' names (the order of `importance`).
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// v42: where the parameter called `name` is pulled back to (None: not anchored).
    pub fn w0_of(&self, name: &str) -> Option<&Tensor> {
        self.names.iter().position(|n| n == name).map(|i| &self.w0[i])
    }

    /// v42: the mean importance over every anchored weight (1 for a fresh anchor; γ + 1 after one `combine` with
    /// the same coverage).
    pub fn mean_importance(&self) -> Result<f64> {
        let (mut s, mut n) = (0f64, 0usize);
        for f in &self.imp {
            s += f.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as f64;
            n += f.elem_count();
        }
        Ok(s / n.max(1) as f64)
    }

    /// v42: write w0 and F as `w0.<name>` and `imp.<name>`, plus `meta.strength`, to a safetensors file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for ((n, w0), f) in self.names.iter().zip(&self.w0).zip(&self.imp) {
            map.insert(format!("w0.{n}"), w0.clone());
            map.insert(format!("imp.{n}"), f.clone());
        }
        let dev = self.w0.first().map(|t| t.device().clone()).unwrap_or(Device::Cpu);
        map.insert("meta.strength".into(), Tensor::new(&[self.strength], &dev)?);
        candle_core::safetensors::save(&map, path)
    }

    /// v42: read an anchor written by `save` onto `dev`.
    pub fn read(path: &Path, dev: &Device) -> Result<SavedAnchor> {
        let map = candle_core::safetensors::load(path, dev)?;
        let (mut w0, mut imp) = (HashMap::new(), HashMap::new());
        let mut strength = f64::NAN;
        for (k, t) in map {
            if let Some(n) = k.strip_prefix("w0.") {
                w0.insert(n.to_string(), t);
            } else if let Some(n) = k.strip_prefix("imp.") {
                imp.insert(n.to_string(), t);
            } else if k == "meta.strength" {
                strength = t.to_dtype(DType::F64)?.to_vec1::<f64>()?.first().copied().unwrap_or(f64::NAN);
            }
        }
        if imp.is_empty() {
            candle_core::bail!("{}: no importance (imp.*) in this file", path.display());
        }
        Ok(SavedAnchor { w0, imp, strength })
    }

    /// v42: this fresh anchor combined with a saved one, parameter by parameter over `params`:
    /// F = gamma · F_old + F_new (a parameter only one side holds takes that side's term alone). The anchor point is
    /// this anchor's w0 (the weights the phase starts from), or with `saved_point` the saved w0. Strength: this one's.
    pub fn combine(self, old: &SavedAnchor, gamma: f64, saved_point: bool, params: &[Param]) -> Result<Anchor> {
        let mut fresh: HashMap<String, (Tensor, Tensor)> = HashMap::new();
        for ((n, w0), f) in self.names.into_iter().zip(self.w0).zip(self.imp) {
            fresh.insert(n, (w0, f));
        }
        let (mut names, mut vars, mut w0s, mut imps) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut weights = 0usize;
        for p in params {
            let cur = p.var.as_tensor();
            let new = fresh.remove(&p.name);
            let f_old = match old.imp.get(&p.name) {
                Some(f) => {
                    if f.dims() != cur.dims() {
                        candle_core::bail!("saved anchor: {} has shape {:?}, the model {:?}", p.name, f.dims(), cur.dims());
                    }
                    Some(f.to_device(cur.device())?.to_dtype(cur.dtype())?.affine(gamma, 0.0)?)
                }
                None => None,
            };
            let f = match (f_old, new.as_ref().map(|x| &x.1)) {
                (Some(a), Some(b)) => a.add(b)?,
                (Some(a), None) => a,
                (None, Some(b)) => b.clone(),
                (None, None) => continue,
            };
            let w0 = if saved_point {
                let w = old.w0.get(&p.name).ok_or_else(|| candle_core::Error::Msg(format!("saved anchor lacks w0.{}", p.name)))?;
                if w.dims() != cur.dims() {
                    candle_core::bail!("saved anchor: w0.{} has shape {:?}, the model {:?}", p.name, w.dims(), cur.dims());
                }
                w.to_device(cur.device())?.to_dtype(cur.dtype())?
            } else {
                match new {
                    Some((w, _)) => w,
                    None => cur.copy()?,
                }
            };
            weights += f.elem_count();
            names.push(p.name.clone());
            vars.push(p.var.clone());
            w0s.push(w0);
            imps.push(f);
        }
        Ok(Anchor { names, vars, w0: w0s, imp: imps, strength: self.strength, weights })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CA: [f32; 4] = [1., 2., 3., 4.];
    const CB: [f32; 2] = [5., 6.];

    /// Two parameters on the CPU: a [2, 2] = (1, 2, 3, 4) and b [2] = (5, 6).
    fn params() -> Vec<Param> {
        let a = Var::from_vec(CA.to_vec(), (2, 2), &Device::Cpu).unwrap();
        let b = Var::from_vec(CB.to_vec(), 2, &Device::Cpu).unwrap();
        vec![Param { name: "a".into(), var: a, decay: true }, Param { name: "b".into(), var: b, decay: false }]
    }

    /// An Importance fed with one batch whose loss is Σ c·x (so the gradient is c, whatever the weights).
    fn measured(params: &[Param]) -> Importance {
        let mut imp = Importance::new(params);
        let ca = Tensor::from_vec(CA.to_vec(), (2, 2), &Device::Cpu).unwrap();
        let cb = Tensor::from_vec(CB.to_vec(), 2, &Device::Cpu).unwrap();
        let la = params[0].var.as_tensor().mul(&ca).unwrap().sum_all().unwrap();
        let lb = params[1].var.as_tensor().mul(&cb).unwrap().sum_all().unwrap();
        imp.add(&la.add(&lb).unwrap().backward().unwrap()).unwrap();
        imp
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn save_read_and_combine_is_gamma_old_plus_new() {
        let dir = std::env::temp_dir().join(format!("ava_anchor_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("anchor.safetensors");
        let ps = params();
        let an = measured(&ps).anchor(2.5).unwrap().unwrap();
        assert!((an.mean_importance().unwrap() - 1.0).abs() < 1e-5);
        let f_old: Vec<Vec<f32>> = an.importance().iter().map(flat).collect();
        an.save(&path).unwrap();
        let saved = Anchor::read(&path, &Device::Cpu).unwrap();
        assert_eq!(saved.strength, 2.5);
        assert_eq!(flat(&saved.imp["a"]), f_old[0]);
        assert_eq!(flat(&saved.w0["b"]), CB.to_vec());
        // she moves; the next phase measures again (the same gradient, so the same normalised F) and combines
        ps[0].var.set(&Tensor::from_vec(vec![10f32, 20., 30., 40.], (2, 2), &Device::Cpu).unwrap()).unwrap();
        let fresh = measured(&ps).anchor(2.5).unwrap().unwrap();
        let f_new: Vec<Vec<f32>> = fresh.importance().iter().map(flat).collect();
        let gamma = 0.5;
        let comb = fresh.combine(&saved, gamma, false, &ps).unwrap();
        assert_eq!(comb.names(), &["a".to_string(), "b".to_string()]);
        for k in 0..2 {
            for (j, g) in flat(&comb.importance()[k]).iter().enumerate() {
                let want = gamma as f32 * f_old[k][j] + f_new[k][j];
                assert!((g - want).abs() < 1e-5, "param {k} weight {j}: {g} vs {want}");
            }
        }
        assert!((comb.mean_importance().unwrap() - 1.5).abs() < 1e-5);
        // online EWC: the anchor point is where she is now
        assert_eq!(flat(comb.w0_of("a").unwrap()), vec![10., 20., 30., 40.]);
        // a fixed point: the saved w0
        let fresh2 = measured(&ps).anchor(2.5).unwrap().unwrap();
        let comb2 = fresh2.combine(&saved, 1.0, true, &ps).unwrap();
        assert_eq!(flat(comb2.w0_of("a").unwrap()), CA.to_vec());
        assert!((comb2.mean_importance().unwrap() - 2.0).abs() < 1e-5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pull_moves_back_toward_w0() {
        let ps = params();
        let an = measured(&ps).anchor(1.0).unwrap().unwrap();
        ps[1].var.set(&Tensor::from_vec(vec![7f32, 8.], 2, &Device::Cpu).unwrap()).unwrap();
        an.pull(1e9).unwrap(); // a huge pull clamps at 1: straight back to w0
        assert_eq!(flat(ps[1].var.as_tensor()), CB.to_vec());
    }
}
