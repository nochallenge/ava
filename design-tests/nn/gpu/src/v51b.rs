// TESTING / DESIGN ONLY — ava_gpu::v51b: v51 Part B2 (runs/v51/prereg_v51.md): a thalamus and a cerebellum, two options
// for model D, each off by default (off: every number is what it was; src/v51b_golden.rs checks it bit for bit).
//
// --thalamus           TH, the switchboard. A small relay of K = 16 units reads every block's output at each word (and
//                      first the word as it enters, the blocks' input) and sets, for each block, two graded gates on
//                      what that block sends on: its whole update of the stream (the residual path) and its fast-memory
//                      read (the memory path); with --loop also one on the top-down path. Per word and per pass, no
//                      state in time (so training stays parallel over the window):
//                        r_j = rms(x_j) · R_j          x_0 = the blocks' input, x_j = block j-1's output, R_j [d, K]
//                        f_i = tanh((r_0 + .. + r_i) / sqrt(i + 1))     the relay once it has read x_0 .. x_i
//                        [g_res, g_mem]_i = 2 sigmoid(f_i · G_i + c_i)  G_i [K, 2], c_i [2]: 0 at init, every gate 1
//                        block i: memory read y -> g_mem · y; output x_out -> x_out + (g_res - 1)(x_out - x_in)
//                      At init every gate is exactly 1 and both rules give today's numbers bit for bit. R starts at
//                      fixed seeded noise (unit spread per relay unit, no random draws: every other weight keeps today's
//                      init) and learns once G moves; R, G and c are not weight-decayed (gate-like, as v50's nucleus).
//                      --loop N: after a pass the relay reads the last block's output too; g_td = 2 sigmoid(f · G_td +
//                      c_td) [one per word] scales the top-down signal that pass adds to the next one.
// --cerebellum-head G  CB, fast error correction with G granule units. Mossy fibres: her final normed state xf and the
//                      current word's embedding (rms-normed), both detached (she does not bend to suit it): 2d inputs.
//                      Granule cells: each sums 4 of them with fixed seeded weights ±1/2 (a fixed sparse expansion, not
//                      learned and not saved: rebuilt from the seed); Golgi inhibition keeps the top G/32 (kWTA); the
//                      code c = relu(h) on the winners, scaled to unit length. Purkinje cells: a learned read-out W
//                      [G, d] (cbh.wo, 0 at init, decayed as every .wo) adds c · W to xf, so her next-word guess is
//                      softmax((xf + c·W) · E^T). Trained with her by backprop. While she reads or talks its read-out
//                      can also learn alone from her own error at each word (`Model::cb_learn`, the delta rule):
//                        W -= lr · c^T ((softmax(logits) - onehot(next)) · E)
//                      the gradient of that word's cross-entropy with respect to W, which touches only the G/32 rows
//                      that fired. Those online changes are kept apart, per stream row, in a fast table on the host
//                      (`Model::cb_online_begin`); `Model::cb_consolidate` folds them into W.
//                      The read-out runs as a dense matmul on CUDA and as a gather of the winning rows on the CPU (the
//                      same sum in another order). (`--cerebellum G` is wave5's older H11, a stream correction inside
//                      every block; this part is new and reads out into her guess.)
//
// Both run on the block-options path (Model::forward_arch, candle's autograd) and the State path (src/state.rs); the
// fused WIN training step (fwin::step_ok: the Arch differs from its set) and the fused CPU talking step
// (fstep/arch.rs::blocker) refuse them.

use crate::arch::Arch;
use crate::model::{kwta_mask, norm, sigmoid, Config, Model};
use candle_core::{DType, Result, Tensor, Var, D};
use candle_nn::{Init, VarBuilder};
use std::cell::RefCell;

/// TH: relay units.
pub const TH_K: usize = 16;
/// CB: inputs per granule unit, 1 in CB_SPARSITY units kept, the wiring seed.
pub const CB_FAN_IN: usize = 4;
pub const CB_SPARSITY: usize = 32;
pub const CB_SEED: u64 = 0x51B0_CE7E;
/// TH: the relay's init seed.
pub const TH_SEED: u64 = 0x51B0_7A1A;

/// The v51b options (a part of Arch).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Opts {
    /// --thalamus
    pub thalamus: bool,
    /// --cerebellum-head G (0 = off)
    pub cb_units: usize,
}

/// Whether any v51b option is on.
pub fn any(a: &Arch) -> bool {
    a.v51b.thalamus || a.v51b.cb_units > 0
}

/// One v51b flag at args[i] (Arch::take_flag): how many arguments it used, None when it is not one.
pub fn take_flag(o: &mut Opts, args: &[String], i: usize) -> std::result::Result<Option<usize>, String> {
    match args[i].as_str() {
        "--thalamus" => {
            o.thalamus = true;
            Ok(Some(1))
        }
        "--cerebellum-head" => {
            let v = args.get(i + 1).ok_or("--cerebellum-head needs a value")?;
            o.cb_units = v.parse().map_err(|_| format!("bad integer {v:?} for --cerebellum-head"))?;
            Ok(Some(2))
        }
        _ => Ok(None),
    }
}

/// " key=value ..." (Arch::signature).
pub fn signature(a: &Arch) -> String {
    let mut s = String::new();
    if a.v51b.thalamus {
        s += " thalamus=1";
    }
    if a.v51b.cb_units > 0 {
        s += &format!(" cerebellum_head={}", a.v51b.cb_units);
    }
    s
}

/// The flags (Arch::flags, arch.txt).
pub fn flags(a: &Arch) -> Vec<String> {
    let mut f = Vec::new();
    if a.v51b.thalamus {
        f.push("--thalamus".into());
    }
    if a.v51b.cb_units > 0 {
        f.push(format!("--cerebellum-head {}", a.v51b.cb_units));
    }
    f
}

/// Checked against the rest of the config (Arch::validate).
pub fn validate(a: &Arch, c: &Config) -> std::result::Result<(), String> {
    let g = a.v51b.cb_units;
    if g > 0 && (g < CB_SPARSITY || g % CB_SPARSITY != 0) {
        return Err(format!("--cerebellum-head {g}: a multiple of {CB_SPARSITY} (the top G/{CB_SPARSITY} granule units fire)"));
    }
    if any(a) && a.mem.halting > 0 {
        return Err("--thalamus / --cerebellum-head: not with --halting".into());
    }
    if any(a) && c.layers < 1 {
        return Err("--thalamus / --cerebellum-head: at least one block".into());
    }
    Ok(())
}

/// How many relay reads TH makes: the input and every block's output but the last, and with --loop the last too.
fn th_reads(cfg: &Config) -> usize {
    cfg.layers + usize::from(cfg.arch.loops > 1)
}

/// The extra weights (Config::weight_counts).
pub fn extra_weights(cfg: &Config) -> usize {
    let (a, d, l) = (cfg.arch, cfg.d, cfg.layers);
    let mut n = 0;
    if a.v51b.thalamus {
        n += th_reads(cfg) * d * TH_K + l * (TH_K * 2 + 2);
        if a.loops > 1 {
            n += TH_K + 1;
        }
    }
    n += a.v51b.cb_units * d;
    n
}

/// TH's weights.
pub(crate) struct Thal {
    /// R [reads, d, K], the relay's reads
    pub(crate) read: Tensor,
    /// G [L, K, 2] and c [L, 2], the gates (0 at init)
    pub(crate) gate: Tensor,
    pub(crate) bias: Tensor,
    /// --loop: G_td [K, 1], c_td [1]
    pub(crate) td: Option<(Tensor, Tensor)>,
}

/// CB's weights and fixed wiring.
pub(crate) struct Cereb {
    /// W [G, d], the Purkinje read-out (0 at init)
    pub(crate) w: Tensor,
    /// [G·F] u32: each granule unit's inputs in [0, 2d); [G, F] f32: their fixed weights ±1/2
    pub(crate) idx: Tensor,
    pub(crate) sgn: Tensor,
    pub(crate) g: usize,
    pub(crate) k: usize,
}

/// One forward's relay, for the blocks of the current pass.
struct ThRun {
    /// the relay's sum of reads [n, K] and how many reads it holds
    acc: Tensor,
    reads: usize,
    /// the gates [n, 2] of the block being computed (None outside a block)
    gates: Option<Tensor>,
    /// --loop: the top-down gate [n, 1] for the next pass
    td: Option<Tensor>,
}

/// What the chat app shows of these parts, kept for the last token while `Model::set_part_signals(true)`.
#[derive(Clone, Debug, Default)]
pub struct PartSignals {
    /// TH: per block, the mean residual and memory gates over the forward's last row
    pub th: Vec<[f32; 2]>,
    /// CB: the size (rms over d) of the correction it added to her state at the last token
    pub cb: Option<f32>,
    /// CB: the size of the online table's share of it (None without an online table)
    pub cb_online: Option<f32>,
    /// v50 --neuromod: the NE, ACh and DA signals at the last token (None without the option)
    pub nm: Option<[f32; 3]>,
}

/// The online table: per stream row, the read-out's changes learned while reading (host, [G·d] each, made on the
/// first write), and the last forward's active codes per position.
pub struct CbOnline {
    pub rows: usize,
    pub lr: f32,
    /// the rate of each row (`lr` unless `Model::cb_online_set_lr` set another)
    pub lrs: Vec<f32>,
    pub(crate) delta: Vec<Vec<f32>>,
    pub(crate) last: Vec<Vec<(u32, f32)>>,
    pub(crate) last_t: usize,
    /// how many updates each row took
    pub updates: Vec<usize>,
    /// v51 CB2 (runs/v51/prereg_v51.md, Amendment B-4): per row, the half-life of its learned change in tokens
    /// (f64::INFINITY, the default: it never fades, as in CB online)
    pub half: Vec<f64>,
    /// v51 CB2: per row, the table's scale: the row's change is scale · delta. Each token on which the rule runs
    /// multiplies it by 2^(-1 / half) (the whole change fades, in one multiply), and the token's own step is added to
    /// delta divided by it. With an infinite half-life it stays exactly 1 and the old arithmetic runs unchanged.
    pub(crate) scale: Vec<f64>,
}

/// v51 CB2: below this scale a row's table is folded (delta · scale, scale back to 1), so delta never grows past f32.
pub(crate) const CB_SCALE_FOLD: f64 = 1e-18;

/// The parts of a model (ModelExt::v51b).
pub(crate) struct V51bExt {
    pub(crate) th: Option<Thal>,
    pub(crate) cb: Option<Cereb>,
    th_run: RefCell<Option<ThRun>>,
    ones: Tensor,
    pub(crate) cb_online: RefCell<Option<CbOnline>>,
    pub(crate) sig_on: std::cell::Cell<bool>,
    pub(crate) sig: RefCell<PartSignals>,
    /// v51 E (src/eprop.rs): a plastic session (per-row weight changes and their records), None when off
    pub(crate) plastic: RefCell<Option<crate::eprop::Plastic>>,
    /// v51 B-4 (src/v51e.rs): the forgetting test's shared trunk: a replay (None when off) and records (empty when off)
    pub(crate) trunk: RefCell<Option<crate::v51e::Trunk>>,
    pub(crate) trunk_rec: RefCell<Vec<crate::v51e::Trunk>>,
}

/// Uniform noise of unit variance from a seed (xorshift; the same as testkit's, not candle's unseeded draws).
pub fn seeded_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let u = (x >> 11) as f64 / (1u64 << 53) as f64;
            ((u * 2.0 - 1.0) * 3f64.sqrt()) as f32
        })
        .collect()
}

/// CB's fixed wiring: for each of g units, `fan_in` distinct inputs of `inputs` and weights ±1/2 (from the seed).
pub fn cb_wiring(inputs: usize, g: usize, fan_in: usize, seed: u64) -> (Vec<u32>, Vec<f32>) {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let amp = 1.0 / (fan_in as f32).sqrt();
    let (mut idx, mut w) = (Vec::with_capacity(g * fan_in), Vec::with_capacity(g * fan_in));
    for _ in 0..g {
        let mut picked: Vec<u32> = Vec::with_capacity(fan_in);
        while picked.len() < fan_in.min(inputs) {
            let r = (next() % inputs as u64) as u32;
            if !picked.contains(&r) {
                picked.push(r);
            }
        }
        for r in picked {
            idx.push(r);
            w.push(if next() & 1 == 0 { amp } else { -amp });
        }
    }
    (idx, w)
}

/// Make the parts (called last in arch::build_ext: constants and seeded values only, no random draws).
pub(crate) fn build(cfg: &Config, vb: &VarBuilder) -> Result<V51bExt> {
    let (a, d, l) = (cfg.arch, cfg.d, cfg.layers);
    let dev = vb.device();
    let th = if a.v51b.thalamus {
        let vt = vb.pp("th");
        let nr = th_reads(cfg);
        let fresh = !vt.contains_tensor("read");
        let read = vt.get_with_hints((nr, d, TH_K), "read", Init::Const(0.0))?;
        if fresh {
            let s = 1.0 / (d as f32).sqrt();
            let v: Vec<f32> = seeded_noise(nr * d * TH_K, TH_SEED).iter().map(|z| z * s).collect();
            Var::from_tensor(&read)?.set(&Tensor::from_vec(v, (nr, d, TH_K), dev)?)?;
        }
        let gate = vt.get_with_hints((l, TH_K, 2), "gate", Init::Const(0.0))?;
        let bias = vt.get_with_hints((l, 2), "bias", Init::Const(0.0))?;
        let td = if a.loops > 1 { Some((vt.get_with_hints((TH_K, 1), "tdg", Init::Const(0.0))?, vt.get_with_hints(1, "tdb", Init::Const(0.0))?)) } else { None };
        Some(Thal { read, gate, bias, td })
    } else {
        None
    };
    let cb = if a.v51b.cb_units > 0 {
        let g = a.v51b.cb_units;
        let w = vb.get_with_hints((g, d), "cbh.wo", Init::Const(0.0))?;
        let (idx, sgn) = cb_wiring(2 * d, g, CB_FAN_IN, CB_SEED);
        Some(Cereb { w, idx: Tensor::from_vec(idx, g * CB_FAN_IN, dev)?, sgn: Tensor::from_vec(sgn, (g, CB_FAN_IN), dev)?, g, k: g / CB_SPARSITY })
    } else {
        None
    };
    Ok(V51bExt {
        th,
        cb,
        th_run: RefCell::new(None),
        ones: Tensor::ones(d, DType::F32, dev)?,
        cb_online: RefCell::new(None),
        sig_on: std::cell::Cell::new(false),
        sig: RefCell::new(PartSignals::default()),
        plastic: RefCell::new(None),
        trunk: RefCell::new(None),
        trunk_rec: RefCell::new(Vec::new()),
    })
}

/// TH: one read of the stream x [n, d] by relay read j: rms(x) · R_j [n, K].
pub fn th_read(x: &Tensor, ones: &Tensor, r: &Tensor) -> Result<Tensor> {
    norm(x, ones)?.matmul(r)
}

/// TH: the gates 2 sigmoid(tanh(acc / sqrt(reads)) · G + c) [n, m] from the relay's sum acc [n, K].
pub fn th_gates(acc: &Tensor, reads: usize, g: &Tensor, c: &Tensor) -> Result<Tensor> {
    let f = acc.affine(1.0 / (reads as f64).sqrt(), 0.0)?.tanh()?;
    Ok(sigmoid(&f.matmul(g)?.broadcast_add(c)?)?.affine(2.0, 0.0)?)
}

/// TH's residual rule: x_out + (g - 1)(x_out - x_in), g [n, 1] (exactly x_out at g = 1).
pub fn th_res(x_in: &Tensor, x_out: &Tensor, g: &Tensor) -> Result<Tensor> {
    x_out.add(&x_out.sub(x_in)?.broadcast_mul(&g.affine(1.0, -1.0)?)?)
}

/// CB: the granule code [n, G] (detached, unit length, G/32 non-zero per row) of her state xf [n, d] and the words'
/// embeddings emb [n, d].
pub fn cb_code(xf: &Tensor, emb: &Tensor, ones: &Tensor, idx: &Tensor, sgn: &Tensor, k: usize) -> Result<Tensor> {
    let u = Tensor::cat(&[xf.detach(), crate::model::rmsnorm(&emb.detach(), ones)?], 1)?; // [n, 2d]
    let n = u.dim(0)?;
    let (g, f) = sgn.dims2()?;
    let h = u.index_select(idx, 1)?.reshape((n, g, f))?.broadcast_mul(sgn)?.sum(2)?; // [n, G]
    let a = h.mul(&kwta_mask(&h, k)?)?.relu()?;
    let len = a.sqr()?.sum_keepdim(1)?.sqrt()?.affine(1.0, 1e-6)?;
    a.broadcast_div(&len)
}

/// CB's active codes per row of c [n, G] (host): (unit, value) for every non-zero.
pub fn active(c: &Tensor) -> Result<Vec<Vec<(u32, f32)>>> {
    Ok(active_f64(c)?.into_iter().map(|r| r.into_iter().map(|(j, x)| (j, x as f32)).collect()).collect())
}

/// `active` in f64 (the read-out's gather keeps the code's own precision).
pub fn active_f64(c: &Tensor) -> Result<Vec<Vec<(u32, f64)>>> {
    let v: Vec<Vec<f64>> = c.to_dtype(DType::F64)?.to_vec2()?;
    Ok(v.iter().map(|r| r.iter().enumerate().filter(|(_, &x)| x != 0.0).map(|(j, &x)| (j as u32, x)).collect()).collect())
}

/// CB: the read-out c · W [n, d]: a dense matmul on CUDA, a gather of the winning rows on the CPU.
pub fn cb_readout(c: &Tensor, w: &Tensor) -> Result<Tensor> {
    if c.device().is_cuda() {
        return c.matmul(w);
    }
    let act = active_f64(c)?;
    let (n, d) = (act.len(), w.dim(1)?);
    let m = act.iter().map(|r| r.len()).max().unwrap_or(0).max(1);
    let (mut idx, mut val) = (Vec::with_capacity(n * m), Vec::with_capacity(n * m));
    for r in &act {
        for j in 0..m {
            let (u, x) = r.get(j).copied().unwrap_or((0, 0.0));
            idx.push(u);
            val.push(x);
        }
    }
    let dev = c.device();
    let rows = w.index_select(&Tensor::from_vec(idx, n * m, dev)?, 0)?.reshape((n, m, d))?;
    rows.broadcast_mul(&Tensor::from_vec(val, (n, m, 1), dev)?.to_dtype(w.dtype())?)?.sum(1)
}

impl CbOnline {
    /// The online table's share of the correction [n, d] for the last forward's codes (rows of `t` positions each).
    pub(crate) fn add(&self, act: &[Vec<(u32, f32)>], t: usize, d: usize) -> Vec<f32> {
        let mut out = vec![0f32; act.len() * d];
        for (p, a) in act.iter().enumerate() {
            let Some(tab) = self.delta.get(p / t).filter(|v| !v.is_empty()) else { continue };
            let sc = self.scale.get(p / t).copied().unwrap_or(1.0);
            let o = &mut out[p * d..(p + 1) * d];
            for &(u, x) in a {
                let row = &tab[u as usize * d..(u as usize + 1) * d];
                // v51 CB2: the row's change is scale · delta (scale exactly 1 without fading: the old sum)
                let x = if sc == 1.0 { x } else { (x as f64 * sc) as f32 };
                for (oi, ri) in o.iter_mut().zip(row) {
                    *oi += x * ri;
                }
            }
        }
        out
    }

    /// v51 CB2: row r's change as it acts (scale · delta), or None if it has learned nothing.
    pub fn table(&self, r: usize) -> Option<Vec<f32>> {
        let tab = self.delta.get(r).filter(|v| !v.is_empty())?;
        let sc = self.scale[r];
        Some(if sc == 1.0 { tab.clone() } else { tab.iter().map(|&v| (v as f64 * sc) as f32).collect() })
    }

    /// v51 CB2: row r's stored table and scale (a stream's brain saved as it is, to resume bit for bit).
    pub fn raw(&self, r: usize) -> (&[f32], f64) {
        (&self.delta[r], self.scale[r])
    }

    /// v51 CB2: set row r's stored table and scale (from `raw`).
    pub fn set_raw(&mut self, r: usize, tab: Vec<f32>, scale: f64) {
        self.delta[r] = tab;
        self.scale[r] = scale;
    }
}

impl Model {
    pub(crate) fn v51b(&self) -> Option<&V51bExt> {
        self.ext.as_ref().map(|e| &e.v51b)
    }

    // ----------------------------------------------------------------------------------------- TH, in the block loop

    /// A pass starts: the relay reads the blocks' input x [n, d] (nothing without --thalamus).
    pub(crate) fn v51b_th_begin(&self, x: &Tensor) -> Result<()> {
        let Some(e) = self.v51b() else { return Ok(()) };
        let Some(th) = &e.th else { return Ok(()) };
        let acc = th_read(x, &e.ones, &th.read.get(0)?)?;
        *e.th_run.borrow_mut() = Some(ThRun { acc, reads: 1, gates: None, td: None });
        Ok(())
    }

    /// Block i starts: its gates from the relay (and, for the chat app, the signals of the parts before it).
    pub(crate) fn v51b_th_block(&self, i: usize) -> Result<()> {
        let Some(e) = self.v51b() else { return Ok(()) };
        if i == 1 && e.sig_on.get() && self.cfg.arch.neuromod > 0 {
            // v50 --neuromod's signals of this token (set after block 0), for the chat app's midbrain
            let nm = self.ext.as_ref().and_then(|x| x.v50.sig.borrow().clone());
            if let Some(s) = nm {
                let n = s.dim(0)?;
                let v: Vec<f32> = s.get(n - 1)?.to_vec1()?;
                e.sig.borrow_mut().nm = Some([v[0], v[1], v[2]]);
            }
        }
        let Some(th) = &e.th else { return Ok(()) };
        let mut run = e.th_run.borrow_mut();
        let Some(r) = run.as_mut() else { candle_core::bail!("--thalamus: block {i} before the pass began") };
        r.gates = Some(th_gates(&r.acc, r.reads, &th.gate.get(i)?, &th.bias.get(i)?)?);
        Ok(())
    }

    /// The memory path's gate on block i's fast-memory read y [n, d] (y itself without --thalamus).
    pub(crate) fn v51b_th_mem(&self, y: Tensor) -> Result<Tensor> {
        let Some(e) = self.v51b() else { return Ok(y) };
        if e.th.is_none() {
            return Ok(y);
        }
        match e.th_run.borrow().as_ref().and_then(|r| r.gates.as_ref()) {
            Some(g) => y.broadcast_mul(&g.narrow(1, 1, 1)?),
            None => Ok(y),
        }
    }

    /// Block i is done: the residual gate on its output (x_out itself without --thalamus); then the relay reads it.
    pub(crate) fn v51b_th_out(&self, i: usize, x_in: &Tensor, x_out: Tensor) -> Result<Tensor> {
        let Some(e) = self.v51b() else { return Ok(x_out) };
        if e.th.is_none() {
            return Ok(x_out);
        }
        let mut run = e.th_run.borrow_mut();
        let Some(r) = run.as_mut() else { candle_core::bail!("--thalamus: block {i} before the pass began") };
        let g = r.gates.take().ok_or_else(|| candle_core::Error::Msg(format!("--thalamus: block {i} has no gates")))?;
        let x = th_res(x_in, &x_out, &g.narrow(1, 0, 1)?)?;
        if e.sig_on.get() && !self.v50_quiet() {
            let n = g.dim(0)?;
            let last: Vec<f32> = g.get(n - 1)?.to_vec1()?;
            let mut s = e.sig.borrow_mut();
            if s.th.len() <= i {
                s.th.resize(i + 1, [1.0, 1.0]);
            }
            s.th[i] = [last[0], last[1]];
        }
        Ok(x)
    }

    /// After block i's output is final (its dense mix included): the relay reads it, if a later gate needs it.
    pub(crate) fn v51b_th_after(&self, i: usize, x: &Tensor) -> Result<()> {
        let Some(e) = self.v51b() else { return Ok(()) };
        let Some(th) = &e.th else { return Ok(()) };
        let nr = th.read.dim(0)?;
        let mut run = e.th_run.borrow_mut();
        let Some(r) = run.as_mut() else { return Ok(()) };
        if i + 1 < nr {
            r.acc = r.acc.add(&th_read(x, &e.ones, &th.read.get(i + 1)?)?)?;
            r.reads += 1;
        }
        if i + 1 == self.cfg.layers {
            if let Some((g, c)) = &th.td {
                r.td = Some(th_gates(&r.acc, r.reads, g, c)?);
            }
        }
        Ok(())
    }

    /// --loop: the next pass's input from `loop_in` (= x_in + the top-down signal), its top-down part scaled by the
    /// relay's gate of the pass before (loop_in itself without --thalamus).
    pub(crate) fn v51b_th_td(&self, x_in: &Tensor, loop_in: Tensor) -> Result<Tensor> {
        let Some(e) = self.v51b() else { return Ok(loop_in) };
        if e.th.is_none() {
            return Ok(loop_in);
        }
        match e.th_run.borrow().as_ref().and_then(|r| r.td.clone()) {
            Some(g) => th_res(x_in, &loop_in, &g),
            None => Ok(loop_in),
        }
    }

    /// The window's passes are done (nothing kept).
    pub(crate) fn v51b_th_end(&self) {
        if let Some(e) = self.v51b() {
            *e.th_run.borrow_mut() = None;
        }
    }

    // ----------------------------------------------------------------------------------------- CB, after the final norm

    /// CB: her final normed state xf [n, d] plus the cerebellum's correction for the words `ids` [b, t] (xf itself
    /// without --cerebellum-head). With an online table, its rows' learned changes are added too and the codes kept.
    pub(crate) fn v51b_cb(&self, xf: Tensor, ids: &Tensor) -> Result<Tensor> {
        let Some(e) = self.v51b() else { return Ok(xf) };
        let Some(cb) = &e.cb else { return Ok(xf) };
        let (b, t) = ids.dims2()?;
        let emb = self.tok.detach().embedding(&ids.flatten_all()?)?;
        let c = cb_code(&xf, &emb, &e.ones, &cb.idx, &cb.sgn, cb.k)?;
        let corr = cb_readout(&c, &cb.w)?;
        let mut out = xf.add(&corr)?;
        let mut online_rms = None;
        if let Some(on) = e.cb_online.borrow_mut().as_mut() {
            if on.rows != b {
                candle_core::bail!("cerebellum online table for {} rows, a forward of {b}", on.rows);
            }
            let act = active(&c)?;
            let d = self.cfg.d;
            let add = on.add(&act, t, d);
            if e.sig_on.get() {
                let last = &add[add.len() - d..];
                online_rms = Some((last.iter().map(|x| x * x).sum::<f32>() / d as f32).sqrt());
            }
            out = out.add(&Tensor::from_vec(add, (b * t, d), xf.device())?)?;
            on.last = act;
            on.last_t = t;
        }
        if e.sig_on.get() {
            let n = b * t;
            let r: f32 = corr.get(n - 1)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
            let mut s = e.sig.borrow_mut();
            s.cb = Some(r);
            s.cb_online = online_rms;
        }
        Ok(out)
    }

    /// The cerebellum's read-out W (None without --cerebellum-head).
    pub fn cb_weight(&self) -> Option<&Tensor> {
        self.v51b().and_then(|e| e.cb.as_ref()).map(|c| &c.w)
    }

    /// Start an online table for streams of `rows` rows with learning rate `lr` (replacing any table). From now on the
    /// cerebellum's read-out per row is W + that row's learned change, and `cb_learn` changes it.
    pub fn cb_online_begin(&self, rows: usize, lr: f32) -> Result<()> {
        let Some(e) = self.v51b().filter(|e| e.cb.is_some()) else { candle_core::bail!("cb_online_begin: the model has no --cerebellum-head") };
        *e.cb_online.borrow_mut() = Some(CbOnline {
            rows,
            lr,
            lrs: vec![lr; rows],
            delta: vec![Vec::new(); rows],
            last: Vec::new(),
            last_t: 1,
            updates: vec![0; rows],
            half: vec![f64::INFINITY; rows],
            scale: vec![1.0; rows],
        });
        Ok(())
    }

    /// v51 CB2: row r's half-life of its learned change, in tokens (f64::INFINITY: it never fades, the default).
    pub fn cb_online_set_half_life(&self, r: usize, h: f64) -> Result<()> {
        if !(h > 0.0) {
            candle_core::bail!("cb_online_set_half_life: half-life {h} (must be > 0)");
        }
        if let Some(e) = self.v51b() {
            if let Some(o) = e.cb_online.borrow_mut().as_mut() {
                if let Some(x) = o.half.get_mut(r) {
                    *x = h;
                }
            }
        }
        Ok(())
    }

    /// v51 CB2: run `f` on the online table (None without one), e.g. to save or restore a stream's brain.
    pub fn cb_online_with<R>(&self, f: impl FnOnce(&mut CbOnline) -> R) -> Option<R> {
        let e = self.v51b()?;
        let mut g = e.cb_online.borrow_mut();
        g.as_mut().map(f)
    }

    /// Whether an online table is on.
    pub fn cb_online_on(&self) -> bool {
        self.v51b().is_some_and(|e| e.cb_online.borrow().is_some())
    }

    /// The online rule's rate for `eta`: eta / mean |E_v|^2 over her word embeddings E (as the online test).
    pub fn cb_rate(&self, eta: f64) -> Result<f32> {
        let e = self.tok.to_dtype(DType::F32)?;
        let v = e.dim(0)? as f64;
        Ok((eta * v / e.sqr()?.sum_all()?.to_scalar::<f32>()? as f64) as f32)
    }

    /// Row r's own learning rate from now on (the online test's rate grid reads several rates side by side).
    pub fn cb_online_set_lr(&self, r: usize, lr: f32) {
        if let Some(e) = self.v51b() {
            if let Some(o) = e.cb_online.borrow_mut().as_mut() {
                if let Some(x) = o.lrs.get_mut(r) {
                    *x = lr;
                }
            }
        }
    }

    /// Clear row r's learned change (a new document starts there), keeping the table.
    pub fn cb_online_reset_row(&self, r: usize) {
        if let Some(e) = self.v51b() {
            if let Some(o) = e.cb_online.borrow_mut().as_mut() {
                if let Some(v) = o.delta.get_mut(r) {
                    v.clear();
                }
                if let Some(u) = o.updates.get_mut(r) {
                    *u = 0;
                }
                if let Some(s) = o.scale.get_mut(r) {
                    *s = 1.0;
                }
            }
        }
    }

    /// The sum over rows of the online table's changes [G·d] (None without a table or with nothing learned yet).
    pub fn cb_online_sum(&self) -> Option<Vec<f32>> {
        let e = self.v51b()?;
        let g = e.cb_online.borrow();
        let on = g.as_ref()?;
        let mut out: Option<Vec<f32>> = None;
        for r in 0..on.delta.len() {
            // v51 CB2: each row's change as it acts (scale · delta; the table itself while nothing fades)
            let Some(tab) = on.table(r) else { continue };
            match out.as_mut() {
                None => out = Some(tab),
                Some(o) => o.iter_mut().zip(&tab).for_each(|(a, b)| *a += b),
            }
        }
        out
    }

    /// Stop the online table (its changes are dropped unless `cb_consolidate` was called first).
    pub fn cb_online_end(&self) {
        if let Some(e) = self.v51b() {
            *e.cb_online.borrow_mut() = None;
        }
    }

    /// The online rule after a forward of one token per row (or t per row): `logits` [B·t, V] that forward gave and
    /// `next` [B·t] the words that came (u32::MAX: no update at that position). Each position's read-out rows that
    /// fired move by -lr · c_j · ((softmax - onehot(next)) · E). Returns the mean cross-entropy (nats) of the scored
    /// positions before the update.
    pub fn cb_learn(&self, logits: &Tensor, next: &[u32]) -> Result<f64> {
        let Some(e) = self.v51b() else { candle_core::bail!("cb_learn: no block options") };
        let mut guard = e.cb_online.borrow_mut();
        let Some(on) = guard.as_mut() else { candle_core::bail!("cb_learn: no online table (Model::cb_online_begin)") };
        let n = logits.dim(0)?;
        if n != next.len() || n != on.last.len() {
            candle_core::bail!("cb_learn: {n} logits rows, {} next words, {} kept codes", next.len(), on.last.len());
        }
        let d = self.cfg.d;
        let dev = logits.device();
        let p = candle_nn::ops::softmax(&logits.to_dtype(DType::F32)?, D::Minus1)?;
        let v = p.dim(1)?;
        let mut oh = vec![0f32; n * v];
        let mut ce = 0f64;
        let pv: Vec<Vec<f32>> = p.to_vec2()?;
        let mut scored = 0usize;
        for (r, &y) in next.iter().enumerate() {
            if y != u32::MAX {
                oh[r * v + y as usize] = 1.0;
                ce -= (pv[r][y as usize].max(1e-30) as f64).ln();
                scored += 1;
            }
        }
        let err = p.sub(&Tensor::from_vec(oh, (n, v), dev)?)?;
        let delta: Vec<Vec<f32>> = err.matmul(&self.tok.detach())?.to_vec2()?; // [n, d]
        let g = e.cb.as_ref().map(|c| c.g).unwrap_or(0);
        let t = on.last_t.max(1);
        for (pos, &y) in next.iter().enumerate() {
            if y == u32::MAX {
                continue;
            }
            let row = pos / t;
            let tab = &mut on.delta[row];
            if tab.is_empty() {
                tab.resize(g * d, 0.0);
            }
            // v51 CB2: the row's whole change fades by 2^(-1 / half) (its scale), then this token's step is added
            if on.half[row].is_finite() {
                on.scale[row] *= (-1.0 / on.half[row]).exp2();
                if on.scale[row] < CB_SCALE_FOLD {
                    let sc = on.scale[row];
                    tab.iter_mut().for_each(|v| *v = (*v as f64 * sc) as f32);
                    on.scale[row] = 1.0;
                }
            }
            let sc = on.scale[row];
            for &(u, x) in &on.last[pos] {
                let w = &mut tab[u as usize * d..(u as usize + 1) * d];
                let s = on.lrs[row] * x;
                let s = if sc == 1.0 { s } else { (s as f64 / sc) as f32 };
                for (wi, di) in w.iter_mut().zip(&delta[pos]) {
                    *wi -= s * di;
                }
            }
            on.updates[row] += 1;
        }
        Ok(if scored > 0 { ce / scored as f64 } else { f64::NAN })
    }

    /// Fold row r's online change into the read-out W itself (the change is kept in W from now on and the row's table
    /// is cleared). Nothing to fold: Ok.
    pub fn cb_consolidate(&self, r: usize) -> Result<()> {
        let Some(e) = self.v51b() else { return Ok(()) };
        let Some(cb) = &e.cb else { return Ok(()) };
        let mut guard = e.cb_online.borrow_mut();
        let Some(on) = guard.as_mut() else { return Ok(()) };
        let Some(tab) = on.table(r) else { return Ok(()) };
        on.delta[r].clear();
        on.scale[r] = 1.0; // v51 CB2: what was folded is the change as it acted (scale · delta)
        let add = Tensor::from_vec(tab, (cb.g, self.cfg.d), cb.w.device())?;
        let neww = cb.w.add(&add)?;
        Var::from_tensor(&cb.w)?.set(&neww)?;
        Ok(())
    }

    // ----------------------------------------------------------------------------------------- the chat app's signals

    /// Keep the parts' signals of the last token (TH gates, CB correction, v50's nucleus) for `part_signals`.
    pub fn set_part_signals(&self, on: bool) {
        if let Some(e) = self.v51b() {
            e.sig_on.set(on);
            *e.sig.borrow_mut() = PartSignals::default();
        }
    }

    /// The parts' signals of the last token computed (empty when off or without block options).
    pub fn part_signals(&self) -> PartSignals {
        self.v51b().map(|e| e.sig.borrow().clone()).unwrap_or_default()
    }

    /// Which of the brain parts this model has, for the chat app's anatomy map: (thalamus, cerebellum head, v50
    /// nucleus, fast memory, engram).
    pub fn part_flags(&self) -> [bool; 5] {
        let a = self.cfg.arch;
        [a.v51b.thalamus, a.v51b.cb_units > 0, a.neuromod > 0, self.blocks.iter().any(|b| b.hebb.is_some()), self.blocks.iter().any(|b| b.eng.is_some())]
    }
}

#[cfg(test)]
mod tests;
