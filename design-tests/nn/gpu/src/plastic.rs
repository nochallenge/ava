// TESTING / DESIGN ONLY — ava_gpu::plastic: H6 rewiring and H7 critical periods (ava_parts::plasticity) in train_gpu.
//
// H7 (--critical-periods FIRST,GAP,TAU,FLOOR): AdamW::enable_critical_periods. Block l's LR (its weights are named
//   b{l}.…) is lr · CriticalPeriods::multiplier(l, step); every other weight keeps lr.
// H6 (--rewire DENSITY,FRAC,EVERY): each block's b{l}.mlp.w1 and b{l}.mlp.w2 keep round(n · DENSITY) live weights,
//   marked by a mask on the device. After every optimizer step the weights are multiplied by their masks, so dead ones
//   stay 0. After every EVERY-th update each tensor goes to the host for ava_parts::plasticity::rewire (the FRAC
//   weakest live ones pruned, as many grown at random dead places, at 0) and comes back; both AdamW moments (Muon's
//   momentum lives in m) are set to 0 where weights grew. The masks go into the checkpoint as mask.<name> (u8).

use crate::optim::AdamW;
pub use ava_parts::plasticity::{rewire, CriticalPeriods};
use candle_core::{DType, Result, Tensor};
use std::collections::HashMap;
use std::time::Instant;

/// The block a weight belongs to: `b{l}.…` gives Some(l); every other name None.
pub fn block_of(name: &str) -> Option<usize> {
    let (i, _) = name.strip_prefix('b')?.split_once('.')?;
    if i.is_empty() || !i.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    i.parse().ok()
}

/// Whether H6 rewires this weight: b{l}.mlp.w1 and b{l}.mlp.w2.
pub fn is_rewired(name: &str) -> bool {
    block_of(name).is_some() && name.split_once('.').is_some_and(|(_, rest)| rest == "mlp.w1" || rest == "mlp.w2")
}

/// "FIRST,GAP,TAU,FLOOR" (--critical-periods).
pub fn parse_critical(s: &str) -> std::result::Result<CriticalPeriods, String> {
    let v: Vec<f64> = s.split(',').map(|x| x.trim().parse::<f64>()).collect::<std::result::Result<_, _>>().map_err(|_| format!("--critical-periods {s:?}: four numbers FIRST,GAP,TAU,FLOOR"))?;
    let [first, gap, tau, floor] = v[..] else {
        return Err(format!("--critical-periods {s:?}: four numbers FIRST,GAP,TAU,FLOOR"));
    };
    if !(first >= 0.0 && gap >= 0.0 && tau > 0.0 && (0.0..=1.0).contains(&floor)) || !(first + gap + tau).is_finite() {
        return Err(format!("--critical-periods {s:?}: FIRST and GAP 0 or more, TAU positive, FLOOR in [0, 1]"));
    }
    Ok(CriticalPeriods { first, gap, tau, floor })
}

/// " critical_periods=F,G,T,FL": for the resume signature and result.tsv.
pub fn critical_signature(c: &CriticalPeriods) -> String {
    format!(" critical_periods={},{},{},{}", c.first, c.gap, c.tau, c.floor)
}

/// H6's settings (--rewire DENSITY,FRAC,EVERY).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewireCfg {
    /// share of each tensor that is live
    pub density: f64,
    /// share of the live weights pruned (and regrown) at each event
    pub frac: f32,
    /// an event after every EVERY-th update
    pub every: u64,
}

impl RewireCfg {
    /// "DENSITY,FRAC,EVERY": density in (0, 1), frac in (0, 1), every 1 or more.
    pub fn parse(s: &str) -> std::result::Result<RewireCfg, String> {
        let bad = || format!("--rewire {s:?}: DENSITY,FRAC,EVERY (density and frac in (0, 1), every a whole number of steps)");
        let p: Vec<&str> = s.split(',').map(str::trim).collect();
        let [d, f, e] = p[..] else { return Err(bad()) };
        let density: f64 = d.parse().map_err(|_| bad())?;
        let frac: f32 = f.parse().map_err(|_| bad())?;
        let every: u64 = e.parse().map_err(|_| bad())?;
        if !(density > 0.0 && density < 1.0 && frac > 0.0 && frac < 1.0 && every >= 1) {
            return Err(bad());
        }
        Ok(RewireCfg { density, frac, every })
    }

    /// " rewire=D,F,E": for the resume signature and result.tsv.
    pub fn signature(&self) -> String {
        format!(" rewire={},{},{}", self.density, self.frac, self.every)
    }
}

/// splitmix64's finalizer.
fn mix(z: u64) -> u64 {
    let mut z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed of tensor k's first mask in a run with seed `seed`.
pub fn init_seed(seed: u64, k: usize) -> u64 {
    mix(mix(seed ^ 0x6D61_736B) ^ k as u64)
}

/// The seed `rewire` gets for tensor k at the event after update `t` (the run seed, the step, the tensor).
pub fn event_seed(seed: u64, t: u64, k: usize) -> u64 {
    mix(mix(mix(seed) ^ t) ^ k as u64)
}

/// `live` of `n` places, chosen at random (a partial Fisher–Yates shuffle).
pub fn random_mask(n: usize, live: usize, seed: u64) -> Vec<bool> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = seed | 1;
    for j in 0..live.min(n) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let r = j + (s % (n - j) as u64) as usize;
        idx.swap(j, r);
    }
    let mut m = vec![false; n];
    for &i in &idx[..live.min(n)] {
        m[i] = true;
    }
    m
}

/// One rewiring event's totals over all tensors, and its time on the host side (ms; the copies both ways included).
#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub t: u64,
    pub pruned: usize,
    pub grown: usize,
    pub ms: f64,
}

/// H6: the masks of the rewired weights.
pub struct Rewiring {
    pub cfg: RewireCfg,
    seed: u64,
    /// each rewired weight's position in the optimizer, and its name
    idx: Vec<usize>,
    names: Vec<String>,
    /// 1 = live, 0 = dead (f32, the weight's shape and device)
    masks: Vec<Tensor>,
}

impl Rewiring {
    /// Masks for every rewired weight of `opt`, each keeping round(n · density) places live at random; the dead
    /// weights are set to 0 now. Errors when a tensor would have too few dead places to regrow into.
    pub fn fresh(cfg: RewireCfg, opt: &AdamW, seed: u64) -> Result<Rewiring> {
        let (mut idx, mut names, mut masks) = (Vec::new(), Vec::new(), Vec::new());
        for (i, p) in opt.params().iter().enumerate() {
            if !is_rewired(&p.name) {
                continue;
            }
            let n = p.var.elem_count();
            let live = ((n as f64) * cfg.density).round() as usize;
            let moved = ((live as f32) * cfg.frac).round() as usize;
            if live == 0 || n - live < moved {
                candle_core::bail!("--rewire: {} has {n} weights, {live} live; {moved} must regrow into {} dead places", p.name, n - live);
            }
            let m = random_mask(n, live, init_seed(seed, idx.len()));
            masks.push(Tensor::from_vec(m.iter().map(|&b| b as u8 as f32).collect::<Vec<f32>>(), p.var.shape(), p.var.device())?);
            idx.push(i);
            names.push(p.name.clone());
        }
        if idx.is_empty() {
            candle_core::bail!("--rewire: the model has no b{{l}}.mlp.w1 or b{{l}}.mlp.w2");
        }
        let r = Rewiring { cfg, seed, idx, names, masks };
        r.apply(opt)?;
        Ok(r)
    }

    pub fn len(&self) -> usize {
        self.idx.len()
    }

    pub fn is_empty(&self) -> bool {
        self.idx.is_empty()
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn masks(&self) -> &[Tensor] {
        &self.masks
    }

    /// (live, total) weights over all rewired tensors.
    pub fn counts(&self) -> Result<(usize, usize)> {
        let mut live = 0usize;
        let mut n = 0usize;
        for m in &self.masks {
            live += m.to_dtype(DType::F64)?.sum_all()?.to_scalar::<f64>()? as usize;
            n += m.elem_count();
        }
        Ok((live, n))
    }

    /// w ← w · mask for every rewired weight (dead weights back to 0; live ones unchanged bit for bit).
    pub fn apply(&self, opt: &AdamW) -> Result<()> {
        for (k, &i) in self.idx.iter().enumerate() {
            let v = &opt.params()[i].var;
            v.set(&v.as_tensor().mul(&self.masks[k])?)?;
        }
        Ok(())
    }

    /// Whether an event follows update number `t` (opt.t after the step).
    pub fn due(&self, t: u64) -> bool {
        t % self.cfg.every == 0
    }

    /// One event after update `t`. For each tensor k: weight and mask to the host, rewire(w, mask, frac,
    /// event_seed(seed, t, k)), both back, and the moments zeroed where weights grew. Returns (pruned, grown).
    pub fn rewire_now(&mut self, opt: &AdamW, t: u64) -> Result<(usize, usize)> {
        let (mut pruned, mut grown) = (0usize, 0usize);
        for k in 0..self.idx.len() {
            let i = self.idx[k];
            let var = &opt.params()[i].var;
            let (shape, dev) = (var.shape().clone(), var.device().clone());
            let mut w: Vec<f32> = var.as_tensor().flatten_all()?.to_vec1()?;
            let old: Vec<bool> = self.masks[k].flatten_all()?.to_vec1::<f32>()?.iter().map(|&x| x != 0.0).collect();
            let mut mask = old.clone();
            let (p, g) = rewire(&mut w, &mut mask, self.cfg.frac, event_seed(self.seed, t, k));
            let grew: Vec<u8> = old.iter().zip(&mask).map(|(&o, &m)| (!o && m) as u8).collect();
            var.set(&Tensor::from_vec(w, &shape, &dev)?)?;
            self.masks[k] = Tensor::from_vec(mask.iter().map(|&b| b as u8 as f32).collect::<Vec<f32>>(), &shape, &dev)?;
            opt.zero_moments_at(i, &Tensor::from_vec(grew, &shape, &dev)?)?;
            pruned += p;
            grown += g;
        }
        Ok((pruned, grown))
    }

    /// After each optimizer step: the masks, then an event when one is due.
    pub fn after_step(&mut self, opt: &AdamW) -> Result<Option<Event>> {
        self.apply(opt)?;
        if !self.due(opt.t) {
            return Ok(None);
        }
        // time the event alone, not the step's work still queued on the GPU
        opt.params()[self.idx[0]].var.device().synchronize()?;
        let t0 = Instant::now();
        let (pruned, grown) = self.rewire_now(opt, opt.t)?;
        Ok(Some(Event { t: opt.t, pruned, grown, ms: t0.elapsed().as_secs_f64() * 1e3 }))
    }

    /// The masks on the host by weight name (true = live), for I3's ReDo (src/upkeep.rs).
    pub fn host_masks(&self) -> Result<HashMap<String, Vec<bool>>> {
        let mut out = HashMap::new();
        for (name, m) in self.names.iter().zip(&self.masks) {
            out.insert(name.clone(), m.flatten_all()?.to_vec1::<f32>()?.iter().map(|&x| x != 0.0).collect());
        }
        Ok(out)
    }

    /// The masks as mask.<name> (u8), for the checkpoint's optim file.
    pub fn state(&self) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        for (name, m) in self.names.iter().zip(&self.masks) {
            out.insert(format!("mask.{name}"), m.to_dtype(DType::U8)?);
        }
        Ok(out)
    }

    /// Restore the masks `state` saved (the weights come back with the model file, already masked).
    pub fn load_state(&mut self, map: &HashMap<String, Tensor>) -> Result<()> {
        for k in 0..self.masks.len() {
            let key = format!("mask.{}", self.names[k]);
            let x = map.get(&key).ok_or_else(|| candle_core::Error::Msg(format!("checkpoint lacks {key}")))?;
            if x.dims() != self.masks[k].dims() {
                candle_core::bail!("{key} has shape {:?}, the weight {:?}", x.dims(), self.masks[k].dims());
            }
            self.masks[k] = x.to_device(self.masks[k].device())?.to_dtype(DType::F32)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::{AdamW, MuonCfg, Param};
    use candle_core::backprop::GradStore;
    use candle_core::{Device, Var};

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        crate::testkit::unit_noise(n, seed)
    }

    /// Small block and non-block weights (the MLP ones rewired), the same values for the same seed.
    fn params(dev: &Device, seed: u64) -> Vec<Param> {
        let spec: [(&str, &[usize], bool); 7] = [
            ("aux.head.w", &[6, 5], true),
            ("b0.mem.wq", &[6, 6], true),
            ("b0.mlp.w1", &[6, 12], true),
            ("b0.mlp.w2", &[12, 6], true),
            ("b1.mlp.norm", &[6], false),
            ("b1.mlp.w1", &[6, 12], true),
            ("b1.mlp.w2", &[12, 6], true),
        ];
        spec.iter()
            .enumerate()
            .map(|(k, (name, shape, decay))| {
                let n: usize = shape.iter().product();
                let w: Vec<f32> = noise(n, seed * 100 + k as u64).iter().map(|x| 0.3 * x).collect();
                Param { name: name.to_string(), var: Var::from_tensor(&Tensor::from_vec(w, *shape, dev).unwrap()).unwrap(), decay: *decay }
            })
            .collect()
    }

    /// Gradients of Σ c·w + Σ w³/3 (they depend on the weights, so two runs agree only if their weights do).
    fn grads(opt: &AdamW, t: u64) -> GradStore {
        let dev = opt.params()[0].var.device().clone();
        let mut l = Tensor::zeros((), DType::F32, &dev).unwrap();
        for (k, p) in opt.params().iter().enumerate() {
            let w = p.var.as_tensor();
            let c = Tensor::from_vec(noise(w.elem_count(), 7000 + 31 * t + k as u64), w.shape(), &dev).unwrap();
            let cube = w.sqr().unwrap().mul(w).unwrap().affine(1.0 / 3.0, 0.0).unwrap();
            l = l.add(&w.mul(&c).unwrap().add(&cube).unwrap().sum_all().unwrap()).unwrap();
        }
        l.backward().unwrap()
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    fn step(opt: &mut AdamW, lr: f64, dev: &Device) {
        let g = grads(opt, opt.t);
        // step_t: the fast path on CUDA, `step` on the CPU
        opt.step_t(&g, lr, &Tensor::new(0.7f64, dev).unwrap()).unwrap();
    }

    #[test]
    fn names_and_flags() {
        assert_eq!(block_of("b0.mlp.w1"), Some(0));
        assert_eq!(block_of("b12.parts.tem.wo"), Some(12));
        for n in ["tok_emb", "final_norm", "aux.head.w", "bias.x", "b.x", "b+1.x", "b1x.y", "b3"] {
            assert_eq!(block_of(n), None, "{n}");
        }
        assert!(is_rewired("b0.mlp.w1") && is_rewired("b7.mlp.w2"));
        assert!(!is_rewired("b0.mlp.norm") && !is_rewired("b0.mem.w1") && !is_rewired("aux.mlp.w1") && !is_rewired("b0.parts.mlp.w1"));
        let c = parse_critical("1000,500,250,0.1").unwrap();
        assert_eq!((c.first, c.gap, c.tau, c.floor), (1000.0, 500.0, 250.0, 0.1));
        assert_eq!(critical_signature(&c), " critical_periods=1000,500,250,0.1");
        for bad in ["1,2,3", "1,2,0,0.1", "1,2,3,1.5", "-1,2,3,0.1", "a,b,c,d", "1,2,3,0.1,5"] {
            assert!(parse_critical(bad).is_err(), "{bad}");
        }
        let r = RewireCfg::parse("0.5,0.1,100").unwrap();
        assert_eq!(r, RewireCfg { density: 0.5, frac: 0.1, every: 100 });
        assert_eq!(r.signature(), " rewire=0.5,0.1,100");
        for bad in ["1,0.1,10", "0,0.1,10", "0.5,0,10", "0.5,0.1,0", "0.5,0.1", "0.5,0.1,2.5"] {
            assert!(RewireCfg::parse(bad).is_err(), "{bad}");
        }
        let m = random_mask(50, 17, 3);
        assert_eq!(m.iter().filter(|&&b| b).count(), 17);
        assert_eq!(m, random_mask(50, 17, 3));
        assert_ne!(m, random_mask(50, 17, 4));
    }

    /// H6: the live count of each tensor never changes, dead weights are exactly 0 after every step, each event is the
    /// part's `rewire` on the same weights, mask and seed, the moments are 0 where weights grew (and unchanged
    /// elsewhere), and the other weights are never masked.
    fn rewiring_case(dev: &Device, muon: bool) {
        let mut opt = AdamW::new(params(dev, 1), 0.1).unwrap();
        if muon {
            opt.enable_muon(MuonCfg::standard(0.02, 0.01));
        }
        let cfg = RewireCfg { density: 0.4, frac: 0.25, every: 3 };
        let mut rw = Rewiring::fresh(cfg, &opt, 5).unwrap();
        assert_eq!(rw.names(), ["b0.mlp.w1", "b0.mlp.w2", "b1.mlp.w1", "b1.mlp.w2"]);
        let want_live: Vec<usize> = rw.masks().iter().map(|m| (m.elem_count() as f64 * 0.4).round() as usize).collect();
        let check = |opt: &AdamW, rw: &Rewiring| {
            for (k, name) in rw.names().iter().enumerate() {
                let i = opt.index_of(name).unwrap();
                let (w, m) = (host(opt.params()[i].var.as_tensor()), host(&rw.masks()[k]));
                assert_eq!(m.iter().filter(|&&x| x == 1.0).count(), want_live[k], "{name} live count");
                assert!(m.iter().all(|&x| x == 0.0 || x == 1.0));
                assert!(w.iter().zip(&m).all(|(&x, &mk)| mk == 1.0 || x == 0.0), "{name}: a dead weight is not 0");
            }
        };
        check(&opt, &rw);
        let other0 = host(opt.params()[opt.index_of("b0.mem.wq").unwrap()].var.as_tensor());
        assert!(other0.iter().all(|&x| x != 0.0), "b0.mem.wq is not rewired");
        let mut events = 0;
        for _ in 0..10 {
            step(&mut opt, 0.01, dev);
            rw.apply(&opt).unwrap();
            check(&opt, &rw);
            if !rw.due(opt.t) {
                continue;
            }
            // the part on host copies, before the event
            let before: Vec<(Vec<f32>, Vec<bool>, Vec<f32>, Vec<f32>)> = rw
                .names()
                .iter()
                .enumerate()
                .map(|(k, n)| {
                    let i = opt.index_of(n).unwrap();
                    let (m, v) = opt.moments(i);
                    (host(opt.params()[i].var.as_tensor()), host(&rw.masks()[k]).iter().map(|&x| x != 0.0).collect(), host(m), host(v))
                })
                .collect();
            let t = opt.t;
            let (p, g) = rw.rewire_now(&opt, t).unwrap();
            events += 1;
            let (mut wp, mut wg) = (0, 0);
            for (k, (w0, m0, am, av)) in before.into_iter().enumerate() {
                let (mut w, mut m) = (w0.clone(), m0.clone());
                let (pp, gg) = rewire(&mut w, &mut m, cfg.frac, event_seed(5, t, k));
                assert!(pp > 0 && pp == gg, "{pp} {gg}");
                wp += pp;
                wg += gg;
                let i = opt.index_of(&rw.names()[k]).unwrap();
                let got_w = host(opt.params()[i].var.as_tensor());
                let got_m: Vec<bool> = host(&rw.masks()[k]).iter().map(|&x| x != 0.0).collect();
                assert_eq!(got_m, m, "mask = the part's");
                assert!(got_w.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()), "weights = the part's");
                let (bm, bv) = opt.moments(i);
                let (bm, bv) = (host(bm), host(bv));
                for j in 0..m.len() {
                    if !m0[j] && m[j] {
                        assert!(bm[j] == 0.0 && bv[j] == 0.0, "moments at a grown place");
                    } else {
                        assert!(bm[j].to_bits() == am[j].to_bits() && bv[j].to_bits() == av[j].to_bits(), "moments elsewhere untouched");
                    }
                }
            }
            assert_eq!((p, g), (wp, wg));
            check(&opt, &rw);
        }
        assert_eq!(events, 3);
        let other = host(opt.params()[opt.index_of("b0.mem.wq").unwrap()].var.as_tensor());
        assert!(other.iter().all(|&x| x != 0.0));
    }

    #[test]
    fn rewiring_keeps_counts_and_matches_the_part() {
        rewiring_case(&Device::Cpu, false);
        rewiring_case(&Device::Cpu, true);
    }

    #[test]
    #[ignore]
    fn rewiring_keeps_counts_and_matches_the_part_gpu() {
        let dev = Device::new_cuda(0).expect("no CUDA device");
        rewiring_case(&dev, false);
        rewiring_case(&dev, true);
    }

    /// The weights, moments and t of a run: what the checkpoint keeps.
    fn snapshot(opt: &AdamW) -> (HashMap<String, Tensor>, HashMap<String, Tensor>, u64) {
        let w = opt.params().iter().map(|p| (p.name.clone(), p.var.as_tensor().copy().unwrap())).collect();
        (w, opt.state(), opt.t)
    }

    /// H6 + H7 through a checkpoint: the masks go through a .safetensors file and back unchanged, and a run stopped
    /// after 7 updates and resumed from its files ends with the same weights, bit for bit, as an unbroken run of 14.
    fn resume_case(dev: &Device) {
        let cp = CriticalPeriods { first: 2.0, gap: 3.0, tau: 4.0, floor: 0.2 };
        let cfg = RewireCfg { density: 0.5, frac: 0.2, every: 3 };
        let start = |dev: &Device| {
            let mut opt = AdamW::new(params(dev, 2), 0.1).unwrap();
            opt.enable_critical_periods(cp);
            let rw = Rewiring::fresh(cfg, &opt, 9).unwrap();
            (opt, rw)
        };
        let run = |opt: &mut AdamW, rw: &mut Rewiring, n: usize| {
            for _ in 0..n {
                step(opt, 0.01, dev);
                rw.after_step(opt).unwrap();
            }
        };
        let (mut a, mut ra) = start(dev);
        run(&mut a, &mut ra, 14);
        let (want, _, _) = snapshot(&a);

        let (mut b, mut rb) = start(dev);
        run(&mut b, &mut rb, 7);
        let dir = crate::testkit::scratch("plastic_resume");
        let (w, mut st, t) = snapshot(&b);
        st.extend(rb.state().unwrap());
        candle_core::safetensors::save(&w, dir.join("model.safetensors")).unwrap();
        candle_core::safetensors::save(&st, dir.join("optim.safetensors")).unwrap();
        drop((b, rb));
        // a fresh process: new masks and weights first (as train_gpu builds them), then the files
        let (mut c, mut rc) = start(dev);
        let w = candle_core::safetensors::load(dir.join("model.safetensors"), dev).unwrap();
        for p in c.params() {
            p.var.set(&w[&p.name]).unwrap();
        }
        let st = candle_core::safetensors::load(dir.join("optim.safetensors"), dev).unwrap();
        assert_eq!(st["mask.b0.mlp.w1"].dtype(), DType::U8);
        c.load_state(&st, t).unwrap();
        rc.load_state(&st).unwrap();
        let saved: Vec<Vec<f32>> = rc.names().iter().map(|n| host(&st[&format!("mask.{n}")].to_dtype(DType::F32).unwrap())).collect();
        for (k, m) in rc.masks().iter().enumerate() {
            assert_eq!(host(m), saved[k], "mask round trip");
        }
        run(&mut c, &mut rc, 7);
        let (got, _, _) = snapshot(&c);
        for (name, x) in &want {
            let (x, y) = (host(x), host(&got[name]));
            assert!(x.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: resumed run differs");
        }
        // a checkpoint without masks is refused
        let mut lacking = st.clone();
        lacking.remove("mask.b1.mlp.w2");
        assert!(rc.load_state(&lacking).is_err());
    }

    #[test]
    fn masks_survive_a_checkpoint_and_resume_is_exact() {
        resume_case(&Device::Cpu);
    }

    #[test]
    #[ignore]
    fn masks_survive_a_checkpoint_and_resume_is_exact_gpu() {
        resume_case(&Device::new_cuda(0).expect("no CUDA device"));
    }
}
