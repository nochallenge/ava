// TESTING / DESIGN ONLY — v42 (A8): plasticity upkeep done at the start of a sleep: ReDo and shrink-and-perturb;
// and (A9) the slot counts behind sparse memory finetuning.
//
// ReDo (recycle dormant units, after Sokar et al. 2023): a unit is dormant when its activity divided by the mean
// activity of its layer is at most tau. The units are each block's LIF channels (activity: the spike rate) and each
// MLP hidden unit (activity: the mean |output| after kWTA and ReLU), measured by `unit_activity` on a few batches of
// her usual reading. A dormant unit gets fresh incoming weights (a random column of win / w1 with the RMS of the
// layer's live columns) and zero outgoing weights (its row of wout / w2), so it adds nothing to the output until
// learning finds it a use. The LIF channels reach the stream only through wout, so reviving a silent one leaves every
// output exactly as it was (one that still fired a little loses that little). For the MLP, kWTA keeps the k largest,
// so a revived unit that wins a place pushes another out, and the output can move a little.
//
// Shrink and perturb (after Ash and Adams 2020): every hidden matrix (the weights `Model::decays` names) becomes
// lambda · W + sigma · rms(W) · z with z ~ N(0, 1). A matrix at exactly 0 (a read-out that has not woken) stays 0.
//
// Both are off unless asked for (grow_gpu --redo TAU, --shrink-perturb L,S); lambda = 1 with sigma = 0, or ReDo with
// no dormant unit, changes nothing (no weight is written). Shrink-and-perturb works on any model.
//
// I3: ReDo for models with block options (src/arch.rs), and during training (train_gpu --redo TAU,EVERY,BATCHES).
// - Activity: a measurement forward turns on ModelExt::redo_rec, which adds each block's |spikes| per channel (what
//   reaches wout, after --resonate and --stp) and |MLP output| per unit (after kWTA and relu2; the first pass under
//   --halting). Training forwards never record.
// - A channel's incoming side is its column of b{l}.mem.win, its outgoing side its row of b{l}.mem.wout, under every
//   option: --residual dense mixes whole block outputs, --value-residual and --gates channel live in the fast-weight
//   memory, not on the LIF path, and --lif-scale only reads beta. By default that is all a revival writes, under
//   every block (grow_gpu's rule). With `inits` (train_gpu: --redo ...,reset) a revived channel also takes back the
//   init value of each per-channel tensor on its path (CHANNEL_RESETS: the leak beta, --homeostasis's threshold,
//   --alif's rho and gain, the dendrite and delay gains, the synapse dynamics of --stp, the resonator's). Those act
//   only on that channel before wout, whose row is now 0, so reviving a silent channel still leaves every output
//   exactly as it was. Dendrite segments and delay centres (random init) are kept; their gains at 0 switch them off.
//   Why the reset is not the default: with --lif-scale the input is scaled by 1 - beta, and with everything on about
//   70% of the channels do not spike at all at init (d64, ctx 64 to 2048; without --lif-scale, none), so the init
//   values put them back where they were silent; and it undoes --homeostasis, which lowers a silent channel's
//   threshold. A revived channel gets no gradient until it spikes (its wout row is 0), so that threshold is its way
//   back.
// - An MLP unit: its column of b{l}.mlp.w1 and its row of b{l}.mlp.w2. Under H6 (--rewire) only the live places of
//   the column get values, at the RMS of the live places of the live columns; the masks themselves are unchanged.
// - `Revived::places` names every place written, so the trainer zeroes the optimizer's moments there.

use crate::model::Model;
use ava_data::rng::{derive_seed, SplitMix64};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;

/// I3: the per-channel tensors on a block's LIF path (b{l}.<suffix>, each [d]) that a revived channel resets to init.
pub const CHANNEL_RESETS: [&str; 13] = [
    "mem.beta",
    "mem.theta",
    "mem.arho",
    "mem.again",
    "parts.dend.a",
    "parts.delays.gain",
    "parts.stp.u_param",
    "parts.stp.f_param",
    "parts.stp.d_param",
    "parts.stp.mix",
    "parts.res.r_param",
    "parts.res.omega_delta",
    "parts.res.mix",
];

/// I3: the per-channel tensors on the LIF path that keep their values (a random init; their gain is reset instead).
pub const CHANNEL_KEEPS: [&str; 2] = ["parts.dend.seg", "parts.delays.center"];

/// Activity per unit, per block: the LIF channels' spike rates [d] and the MLP units' mean |output| [hidden].
#[derive(Clone, Debug)]
pub struct UnitActivity {
    pub lif: Vec<Vec<f32>>,
    pub mlp: Vec<Vec<f32>>,
}

/// What ReDo did, per block: (dormant LIF channels, dormant MLP units), and the layer sizes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RedoReport {
    pub lif: Vec<usize>,
    pub mlp: Vec<usize>,
    pub d: usize,
    pub hidden: usize,
}

impl RedoReport {
    pub fn total(&self) -> usize {
        self.lif.iter().sum::<usize>() + self.mlp.iter().sum::<usize>()
    }
}

/// Every unit's activity over `batches` (ids [B, T] each): through `forward_trace` for today's block, through a
/// recording forward (I3) for a model with block options.
pub fn unit_activity(model: &Model, batches: &[Tensor]) -> Result<UnitActivity> {
    if model.ext.is_some() {
        return arch_activity(model, batches);
    }
    let layers = model.cfg.layers;
    let mut lif: Vec<Option<Tensor>> = vec![None; layers];
    let mut mlp: Vec<Option<Tensor>> = vec![None; layers];
    let mut n = 0usize;
    for ids in batches {
        let (_, tr) = model.forward_trace(ids)?;
        n += ids.elem_count();
        for (l, lt) in tr.layers.iter().enumerate() {
            let s = lt.spikes.to_dtype(DType::F32)?.sum(0)?;
            let h = lt.hidden.to_dtype(DType::F32)?.abs()?.sum(0)?;
            lif[l] = Some(match lif[l].take() {
                Some(x) => x.add(&s)?,
                None => s,
            });
            mlp[l] = Some(match mlp[l].take() {
                Some(x) => x.add(&h)?,
                None => h,
            });
        }
    }
    if n == 0 {
        candle_core::bail!("unit_activity: no tokens");
    }
    let host = |v: Vec<Option<Tensor>>| -> Result<Vec<Vec<f32>>> {
        v.into_iter().map(|t| t.map_or(Ok(Vec::new()), |t| Ok(t.to_vec1::<f32>()?.into_iter().map(|x| x / n as f32).collect()))).collect()
    };
    Ok(UnitActivity { lif: host(lif)?, mlp: host(mlp)? })
}

/// I3: unit_activity for a model with block options: recording forwards (forward_arch with ModelExt::redo_rec on; no
/// activity means, no training-only work), the record switched off again whatever happens.
fn arch_activity(model: &Model, batches: &[Tensor]) -> Result<UnitActivity> {
    let ext = model.ext.as_ref().expect("a model with block options");
    if !model.cfg.kind.is_ava() || model.blocks.iter().any(|b| b.attn.is_some()) {
        candle_core::bail!("--redo: block options run on model D only");
    }
    let layers = model.cfg.layers;
    let track = model.track.replace(false);
    *ext.redo_rec.borrow_mut() = Some(vec![[None, None]; layers]);
    let mut n = 0usize;
    let mut res = Ok(());
    for ids in batches {
        if let Err(e) = model.forward_arch(ids) {
            res = Err(e);
            break;
        }
        n += ids.elem_count();
    }
    let rec = ext.redo_rec.borrow_mut().take().unwrap_or_default();
    model.track.set(track);
    res?;
    if n == 0 {
        candle_core::bail!("unit_activity: no tokens");
    }
    let (mut lif, mut mlp) = (Vec::with_capacity(layers), Vec::with_capacity(layers));
    for (l, [s, a]) in rec.into_iter().enumerate() {
        let (Some(s), Some(a)) = (s, a) else { candle_core::bail!("unit_activity: block {l} recorded nothing") };
        lif.push(s.to_vec1::<f32>()?.into_iter().map(|x| x / n as f32).collect());
        mlp.push(a.to_vec1::<f32>()?.into_iter().map(|x| x / n as f32).collect());
    }
    Ok(UnitActivity { lif, mlp })
}

/// The dormant units of one layer: activity / the layer's mean activity <= tau (a whole silent layer: every unit).
pub fn dormant(act: &[f32], tau: f32) -> Vec<usize> {
    if act.is_empty() {
        return Vec::new();
    }
    let mean = act.iter().map(|&x| x as f64).sum::<f64>() / act.len() as f64;
    act.iter().enumerate().filter(|(_, &x)| mean <= 0.0 || (x as f64 / mean) <= tau as f64).map(|(i, _)| i).collect()
}

/// `n` draws of N(0, 1) (Box-Muller on SplitMix64).
pub fn normals(n: usize, seed: u64) -> Vec<f32> {
    let mut r = SplitMix64::new(seed);
    let mut u = || ((r.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64; // (0, 1)
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (a, b) = (u(), u());
        let rad = (-2.0 * a.ln()).sqrt();
        out.push((rad * (std::f64::consts::TAU * b).cos()) as f32);
        out.push((rad * (std::f64::consts::TAU * b).sin()) as f32);
    }
    out.truncate(n);
    out
}

fn var_of(varmap: &VarMap, name: &str) -> Result<candle_core::Var> {
    varmap.data().lock().unwrap().get(name).cloned().ok_or_else(|| candle_core::Error::Msg(format!("upkeep: no weight {name}")))
}

/// Recycle `units` of the layer whose incoming matrix is `w_in` [rows, cols] (unit = column) and outgoing matrix is
/// `w_out` [cols, out] (unit = row): fresh incoming columns at the RMS of the live columns, zero outgoing rows.
/// I3: `live` (H6's mask of w_in, true = live): only live places get values and count in the RMS (None: all).
fn recycle(w_in: &candle_core::Var, w_out: &candle_core::Var, units: &[usize], seed: u64, live: Option<&[bool]>) -> Result<()> {
    if units.is_empty() {
        return Ok(());
    }
    let dev = w_in.device().clone();
    let (rows, cols) = w_in.dims2()?;
    let mut wi: Vec<f32> = w_in.as_tensor().to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    if live.is_some_and(|m| m.len() != wi.len()) {
        candle_core::bail!("redo: a mask of {} places for a weight of {}", live.map_or(0, |m| m.len()), wi.len());
    }
    let on = |i: usize| live.map_or(true, |m| m[i]);
    let dead: std::collections::HashSet<usize> = units.iter().copied().collect();
    let (mut ss, mut cnt) = (0f64, 0usize);
    for r in 0..rows {
        for c in 0..cols {
            if !dead.contains(&c) && on(r * cols + c) {
                ss += (wi[r * cols + c] as f64).powi(2);
                cnt += 1;
            }
        }
    }
    let mut rms = if cnt > 0 { (ss / cnt as f64).sqrt() } else { 0.0 };
    if rms <= 0.0 {
        // every unit dormant: the matrix's own scale, else the usual 1/sqrt(fan-in)
        let all = wi.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / wi.len().max(1) as f64;
        rms = if all > 0.0 { all.sqrt() } else { 1.0 / (rows as f64).sqrt() };
    }
    let z = normals(rows * units.len(), seed);
    for (k, &c) in units.iter().enumerate() {
        for r in 0..rows {
            if on(r * cols + c) {
                wi[r * cols + c] = (z[k * rows + r] as f64 * rms) as f32;
            }
        }
    }
    w_in.set(&Tensor::from_vec(wi, (rows, cols), &dev)?.to_dtype(w_in.dtype())?)?;
    let (orows, ocols) = w_out.dims2()?;
    let mut wo: Vec<f32> = w_out.as_tensor().to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    for &c in units {
        if c < orows {
            wo[c * ocols..(c + 1) * ocols].iter_mut().for_each(|x| *x = 0.0);
        }
    }
    w_out.set(&Tensor::from_vec(wo, (orows, ocols), &dev)?.to_dtype(w_out.dtype())?)
}

/// ReDo: recycle every unit whose activity in `act` is dormant at `tau` (see the top of the file). Writes only the
/// layers that have a dormant unit.
pub fn redo(varmap: &VarMap, model: &Model, act: &UnitActivity, tau: f32, seed: u64) -> Result<RedoReport> {
    Ok(revive(varmap, model, act, tau, seed, None, None)?.report)
}

/// I3: the init values of a model's CHANNEL_RESETS tensors (those it has), from a fresh build on the CPU. Empty for
/// today's block, whose rule resets none.
pub struct ChannelInit(HashMap<String, Vec<f32>>);

impl ChannelInit {
    pub fn of(model: &Model) -> Result<ChannelInit> {
        if model.ext.is_none() {
            return Ok(ChannelInit(HashMap::new()));
        }
        let vm = VarMap::new();
        let _fresh = Model::new(model.cfg, VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu))?;
        let data = vm.data().lock().unwrap();
        let mut out = HashMap::new();
        for l in 0..model.cfg.layers {
            for s in CHANNEL_RESETS {
                let name = format!("b{l}.{s}");
                if let Some(v) = data.get(&name) {
                    out.insert(name, v.as_tensor().flatten_all()?.to_vec1::<f32>()?);
                }
            }
        }
        Ok(ChannelInit(out))
    }

    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.0.get(name).map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// I3: what one ReDo changed: the counts, and every weight it wrote with the places written (true), for the trainer to
/// zero the optimizer's moments there.
#[derive(Clone, Debug, Default)]
pub struct Revived {
    pub report: RedoReport,
    pub places: Vec<(String, Vec<bool>)>,
}

/// The places of `units` as columns of a [rows, cols] matrix, or (rows = false) as rows of a [units, cols] one.
fn unit_places(rows: usize, cols: usize, units: &[usize], as_cols: bool) -> Vec<bool> {
    let mut p = vec![false; rows * cols];
    for &u in units {
        if as_cols {
            (0..rows).for_each(|r| p[r * cols + u] = true);
        } else {
            p[u * cols..(u + 1) * cols].iter_mut().for_each(|x| *x = true);
        }
    }
    p
}

/// I3: ReDo for any Ava model (see the top of the file), today's block or block options. `masks`: H6's masks by
/// weight name (true = live; None: no rewiring). `inits`: the channel inits (None: built here, for a model with block
/// options). Writes only the layers that have a dormant unit.
pub fn revive(varmap: &VarMap, model: &Model, act: &UnitActivity, tau: f32, seed: u64, masks: Option<&HashMap<String, Vec<bool>>>, inits: Option<&ChannelInit>) -> Result<Revived> {
    if !model.cfg.kind.is_ava() {
        candle_core::bail!("--redo: Ava's kinds only (T has no spiking layer)");
    }
    let layers = model.cfg.layers;
    if act.lif.len() != layers || act.mlp.len() != layers {
        candle_core::bail!("--redo: activity for {} blocks, the model has {layers}", act.lif.len());
    }
    let built;
    let inits = match inits {
        Some(i) => i,
        None => {
            built = ChannelInit::of(model)?;
            &built
        }
    };
    let (d, hid) = (model.cfg.d, model.cfg.hidden());
    let mut out = Revived { report: RedoReport { d, hidden: hid, ..Default::default() }, places: Vec::new() };
    for l in 0..layers {
        let dl = dormant(&act.lif[l], tau);
        let dm = dormant(&act.mlp[l], tau);
        let (win, wout) = (format!("b{l}.mem.win"), format!("b{l}.mem.wout"));
        let (w1, w2) = (format!("b{l}.mlp.w1"), format!("b{l}.mlp.w2"));
        let mask = |n: &str| masks.and_then(|m| m.get(n)).map(|v| v.as_slice());
        recycle(&var_of(varmap, &win)?, &var_of(varmap, &wout)?, &dl, derive_seed(seed, l as u64, 1), mask(&win))?;
        recycle(&var_of(varmap, &w1)?, &var_of(varmap, &w2)?, &dm, derive_seed(seed, l as u64, 2), mask(&w1))?;
        if !dl.is_empty() {
            out.places.push((win, unit_places(d, d, &dl, true)));
            out.places.push((wout, unit_places(d, d, &dl, false)));
            // each per-channel tensor on the LIF path back to its init at the revived channels
            for s in CHANNEL_RESETS {
                let name = format!("b{l}.{s}");
                let Some(init) = inits.get(&name) else { continue };
                let var = var_of(varmap, &name)?;
                let mut v: Vec<f32> = var.as_tensor().to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                if v.len() != d || init.len() != d {
                    candle_core::bail!("redo: {name} has {} values, the init {}, d is {d}", v.len(), init.len());
                }
                for &c in &dl {
                    v[c] = init[c];
                }
                var.set(&Tensor::from_vec(v, d, var.device())?.to_dtype(var.dtype())?)?;
                out.places.push((name, unit_places(1, d, &dl, true)));
            }
        }
        if !dm.is_empty() {
            out.places.push((w1, unit_places(d, hid, &dm, true)));
            out.places.push((w2, unit_places(hid, d, &dm, false)));
        }
        out.report.lif.push(dl.len());
        out.report.mlp.push(dm.len());
    }
    Ok(out)
}

/// I3: zero the optimizer's moments (m and v; Muon's momentum lives in m) at every place `rev` wrote. Returns how
/// many places.
pub fn zero_moments(opt: &crate::optim::AdamW, rev: &Revived) -> Result<usize> {
    let mut n = 0;
    for (name, places) in &rev.places {
        let i = opt.index_of(name).ok_or_else(|| candle_core::Error::Msg(format!("redo: the optimizer has no {name}")))?;
        let (m, _) = opt.moments(i);
        let at: Vec<u8> = places.iter().map(|&p| p as u8).collect();
        opt.zero_moments_at(i, &Tensor::from_vec(at, m.shape(), m.device())?)?;
        n += places.iter().filter(|&&p| p).count();
    }
    Ok(n)
}

/// I3: train_gpu's --redo TAU,EVERY,BATCHES[,reset]: tau in [0, 1), an event after every EVERY-th update, measured on
/// BATCHES batches of training windows; with `reset`, a revived channel's per-channel LIF tensors go back to init too.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RedoCfg {
    pub tau: f32,
    pub every: u64,
    pub batches: usize,
    pub reset: bool,
}

impl RedoCfg {
    pub fn parse(s: &str) -> std::result::Result<RedoCfg, String> {
        let bad = || format!("--redo {s:?}: TAU,EVERY,BATCHES[,reset] (tau in [0, 1), 0.1 is ReDo's usual; every and batches whole numbers, 1 or more)");
        let p: Vec<&str> = s.split(',').map(str::trim).collect();
        let (t, e, b, reset) = match p[..] {
            [t, e, b] => (t, e, b, false),
            [t, e, b, "reset"] => (t, e, b, true),
            _ => return Err(bad()),
        };
        let tau: f32 = t.parse().map_err(|_| bad())?;
        let every: u64 = e.parse().map_err(|_| bad())?;
        let batches: usize = b.parse().map_err(|_| bad())?;
        if !((0.0..1.0).contains(&tau) && every >= 1 && batches >= 1) {
            return Err(bad());
        }
        Ok(RedoCfg { tau, every, batches, reset })
    }

    /// " redo=T,E,B" (",reset" when set): for the resume signature and result.tsv.
    pub fn signature(&self) -> String {
        format!(" redo={},{},{}{}", self.tau, self.every, self.batches, if self.reset { ",reset" } else { "" })
    }
}

/// Shrink and perturb every hidden matrix (`Model::decays`): W <- lambda · W + sigma · rms(W) · z. Returns how many
/// matrices changed. lambda = 1 with sigma = 0 writes nothing.
pub fn shrink_perturb(varmap: &VarMap, lambda: f64, sigma: f64, seed: u64) -> Result<usize> {
    if lambda == 1.0 && sigma == 0.0 {
        return Ok(0);
    }
    let data = varmap.data().lock().unwrap();
    let mut names: Vec<&String> = data.keys().filter(|n| Model::decays(n)).collect();
    names.sort();
    let mut changed = 0;
    for (k, name) in names.into_iter().enumerate() {
        let var = &data[name];
        let w = var.as_tensor().to_dtype(DType::F32)?;
        let rms = w.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt() as f64;
        if rms == 0.0 {
            continue;
        }
        let mut new = w.affine(lambda, 0.0)?;
        if sigma > 0.0 {
            let z = Tensor::from_vec(normals(w.elem_count(), derive_seed(seed, k as u64, 3)), w.shape(), w.device())?;
            new = new.add(&z.affine(sigma * rms, 0.0)?)?;
        }
        var.set(&new.to_dtype(var.dtype())?)?;
        changed += 1;
    }
    Ok(changed)
}

/// A9: the product-key memory's values (its name, e.g. `b1.pk.values`), None without one.
pub fn pk_values_name(model: &Model) -> Option<String> {
    let l = model.cfg.arch.mem.pk_block(model.cfg.layers)?;
    model.ext.as_ref().and_then(|e| e.mem.pkm.as_ref()).map(|_| format!("b{l}.pk.values"))
}

/// A9: how often each slot of the product-key memory was read over `batches` (ids [B, T] each; no learning).
pub fn slot_counts(model: &Model, batches: &[Tensor]) -> Result<Vec<u64>> {
    let Some(pk) = model.ext.as_ref().and_then(|e| e.mem.pkm.as_ref()) else {
        candle_core::bail!("slot counts: the model has no product-key memory (--pk-mem)");
    };
    let slots = pk.values.dim(0)?;
    *pk.counts.borrow_mut() = Some(vec![0; slots]);
    let mut res = Ok(());
    for ids in batches {
        if let Err(e) = model.forward(ids) {
            res = Err(e);
            break;
        }
    }
    let counts = pk.counts.borrow_mut().take().unwrap_or_default();
    res.map(|_| counts)
}

#[cfg(test)]
pub(crate) mod tests;
