// TESTING / DESIGN ONLY — ava_gpu::state: v42 S1 (forward_with_state: B independent row streams, state copy) and
// S2 (a conversation's state saved and loaded, keyed by the model; an optional cap on the engram store).
//
// S1. `Model::state(B)` is the state at the start of every window for B rows: every membrane, spike and fast weight
// at 0, empty key-value stores. `Model::forward_with_state(ids [B, t], &mut state)` reads t more tokens per row in one
// batched pass (the chunked delta rule, not token by token) and returns the logits [B·t, V]; the state then holds
// everything the next call needs: per block the LIF membranes and last spikes (and the ALIF trace), the fast-weight
// memory F (and the momentum), the attention caches (T, A), the engram store, the parts' states, the last 17 tokens
// (the glance's place and the magnitude code read back that far), T's position, and O1's feedback. So a text read in
// pieces gives the logits of one pass over the whole text, up to float rounding (the delta rule's chunks differ;
// the spikes and the stores' reads are the same ops). Kinds S, D, A and T, with every block option; F, G and R use
// Model::stream / step. A fresh state gives `forward`'s logits exactly (the same ops in the same order on the CPU).
// Rows are independent: `State::rows(&[..])` copies rows out (repeats allowed: one prompt's state for each choice),
// `State::cat` joins states row-wise (rows of different lengths: the shorter stores are left-padded with masked
// slots), `clone()` is a full copy (tensors are never changed in place).
//
// S2. `State::save` / `State::load` (safetensors): every tensor above plus the row lengths, the token history and a
// fingerprint of the model; `ConvStore` keeps one file per conversation id in a folder next to the brain
// (conversations/), written through a temporary file and a rename. A state saved by another model (another
// fingerprint) is not loaded: the conversation starts fresh. `EngramCap { max, rule }`: after each call the engram
// store keeps at most `max` slots per row (within one call the new tokens also see each other). Rules: `Oldest` drops
// the oldest slots (a sliding window); `LeastRead` drops the slots that got the least attention per later query since
// they were written (padding first; slots no later query has seen yet are dropped last, oldest first). Default: no cap
// (the store grows by one slot per token, as today).

use crate::arch::{lif_ref, LifState, MemState, Spikes, FB_CHUNK};
use crate::memory::{self, Kv, MemOpts, StoreUse, NEG};
use crate::model::{kwta_mask, norm, rmsnorm, sigmoid, Block, Kind, Model, LIF_THRESHOLD};
use crate::wave5::{ModCarry, ModRow, ModSig};
use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// How many past tokens the state keeps per row (the magnitude code reads back at most 17).
pub const HIST: usize = 17;

/// S2: which engram slots go when the store is over its cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evict {
    Oldest,
    LeastRead,
}

/// S2: the engram store's cap (see the top of the file).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngramCap {
    pub max: usize,
    pub rule: Evict,
}

impl EngramCap {
    /// "none" (no cap), "N" (oldest), "N:oldest" or "N:least-read".
    pub fn parse(s: &str) -> std::result::Result<Option<EngramCap>, String> {
        if s == "none" {
            return Ok(None);
        }
        let (n, rule) = s.split_once(':').unwrap_or((s, "oldest"));
        let max: usize = n.parse().map_err(|_| format!("engram cap {s:?}: none, N, N:oldest or N:least-read"))?;
        if max == 0 {
            return Err("engram cap: at least 1 slot (none = no cap)".into());
        }
        let rule = match rule {
            "oldest" => Evict::Oldest,
            "least-read" => Evict::LeastRead,
            _ => return Err(format!("engram cap rule {rule:?}: oldest or least-read")),
        };
        Ok(Some(EngramCap { max, rule }))
    }
}

/// The carried state of the parts of one block (src/memory.rs).
#[derive(Clone, Default)]
pub(crate) struct PartState {
    /// E9: the release fraction u and the resources x [B, d]
    pub(crate) stp: Option<(Tensor, Tensor)>,
    /// E5: the last K − 1 input currents [B, K − 1, d]
    delays: Option<Tensor>,
    /// B9: the slots [B, slots, dk]
    wm: Option<Tensor>,
    /// E7 (src/wave5.rs): the resonators' (Re z, Im z) [B, d]
    res: Option<(Tensor, Tensor)>,
    /// B10: the structure read's store (each token's code [B, 1, S, 2K] and normed input [B, 1, S, d]) and the phase
    /// reached [B, K]
    tem: Option<Kv>,
    tem_phase: Option<Tensor>,
    /// B3 (src/wave5.rs, its block only): the workspace slots [B, slots, dk]
    ws: Option<Tensor>,
    /// D2 (src/wave5.rs): the binding trace (Re T, Im T) [B, DB]
    holo: Option<(Tensor, Tensor)>,
}

/// The state of B row streams (see the top of the file).
#[derive(Clone)]
pub struct State {
    /// tokens read so far, per row
    pub seen: Vec<usize>,
    pub(crate) lif: Vec<Option<LifState>>,
    pub(crate) mem: Vec<Option<MemState>>,
    pub(crate) kv: Vec<Option<Kv>>,
    pub(crate) eng: Option<Kv>,
    pub(crate) hist: Vec<Vec<u32>>,
    /// O1: the feedback for the next chunk [B, d]
    pub(crate) fb: Option<Tensor>,
    pub(crate) parts: Vec<PartState>,
    /// S2: the engram store's cap (None: no cap)
    pub cap: Option<EngramCap>,
    /// B8/F13 (--modulate, src/wave5.rs): the modulators' running statistics, and the shadow pass's own state (the
    /// same model with the modulators off reads every token too)
    pub(crate) modc: Option<ModCarry>,
    pub(crate) shadow: Option<Box<State>>,
    /// v50 (--loop, src/v50.rs): the later passes' own state (each pass reads every token with its own memory, LIF and
    /// stores; only the per-block parts are used, the outer state keeps the rows' tokens)
    pub(crate) loop2: Option<Box<State>>,
    /// v51 (src/v51.rs): what the round 2 parts carry (empty without them)
    pub(crate) v51: crate::v51::V51State,
}

impl State {
    pub fn rows(&self) -> usize {
        self.seen.len()
    }

    /// Slots in the engram store (the longest row's, padding included).
    pub fn engram_slots(&self) -> usize {
        self.eng.as_ref().map_or(0, |k| k.k.dim(2).unwrap_or(0))
    }

    /// A new state of the rows `idx` of this one, in that order (a row may come more than once).
    pub fn select(&self, idx: &[usize]) -> Result<State> {
        if idx.is_empty() || idx.iter().any(|&r| r >= self.rows()) {
            candle_core::bail!("State::select: rows {idx:?} of {}", self.rows());
        }
        let dev = self.device();
        let it = Tensor::from_vec(idx.iter().map(|&r| r as u32).collect::<Vec<u32>>(), idx.len(), &dev)?;
        let sel = |x: &Tensor| x.index_select(&it, 0);
        let selo = |x: &Option<Tensor>| x.as_ref().map(sel).transpose();
        let b = self.rows();
        let lif = self.lif.iter().map(|l| l.as_ref().map(|l| Ok(LifState { v: sel(&l.v)?, s: sel(&l.s)?, a: selo(&l.a)? })).transpose()).collect::<Result<_>>()?;
        let per_row = |f: &Tensor| -> Result<Tensor> {
            // F [B·nh, hs, hs]: rows of nh heads
            let (bn, hs1, hs2) = f.dims3()?;
            let nh = bn / b;
            f.reshape((b, nh, hs1, hs2))?.index_select(&it, 0)?.reshape((idx.len() * nh, hs1, hs2))
        };
        let mem = self.mem.iter().map(|m| m.as_ref().map(|m| Ok(MemState { f: per_row(&m.f)?, s: m.s.as_ref().map(per_row).transpose()?, prev: selo(&m.prev)? })).transpose()).collect::<Result<_>>()?;
        let selkv = |kv: &Kv| -> Result<Kv> {
            Ok(Kv {
                k: sel(&kv.k)?,
                v: sel(&kv.v)?,
                pad: selo(&kv.pad)?,
                valid: idx.iter().map(|&r| kv.valid[r]).collect(),
                logp: selo(&kv.logp)?,
                ktail: selo(&kv.ktail)?,
                reads: selo(&kv.reads)?,
                age: selo(&kv.age)?,
            })
        };
        let kv = self.kv.iter().map(|k| k.as_ref().map(selkv).transpose()).collect::<Result<_>>()?;
        let eng = self.eng.as_ref().map(selkv).transpose()?;
        let parts = self
            .parts
            .iter()
            .map(|p| {
                Ok(PartState {
                    stp: p.stp.as_ref().map(|(u, x)| Ok::<_, candle_core::Error>((sel(u)?, sel(x)?))).transpose()?,
                    delays: selo(&p.delays)?,
                    wm: selo(&p.wm)?,
                    res: p.res.as_ref().map(|(a, c)| Ok::<_, candle_core::Error>((sel(a)?, sel(c)?))).transpose()?,
                    tem: p.tem.as_ref().map(selkv).transpose()?,
                    tem_phase: selo(&p.tem_phase)?,
                    ws: selo(&p.ws)?,
                    holo: p.holo.as_ref().map(|(a, c)| Ok::<_, candle_core::Error>((sel(a)?, sel(c)?))).transpose()?,
                })
            })
            .collect::<Result<_>>()?;
        let modc = match &self.modc {
            Some(c) => Some(ModCarry { rows: idx.iter().map(|&r| c.rows[r]).collect(), prev: sel(&c.prev)? }),
            None => None,
        };
        let shadow = match &self.shadow {
            Some(s) => Some(Box::new(s.select(idx)?)),
            None => None,
        };
        let loop2 = match &self.loop2 {
            Some(s) => Some(Box::new(s.select(idx)?)),
            None => None,
        };
        Ok(State {
            seen: idx.iter().map(|&r| self.seen[r]).collect(),
            lif,
            mem,
            kv,
            eng,
            hist: idx.iter().map(|&r| self.hist[r].clone()).collect(),
            fb: selo(&self.fb)?,
            parts,
            cap: self.cap,
            modc,
            shadow,
            loop2,
            v51: self.v51.select(&it)?,
        })
    }

    /// The rows of `parts` one after the other (states of the same model; stores of different lengths are left-padded
    /// with masked slots). The cap is the first part's.
    pub fn cat(parts: &[State]) -> Result<State> {
        let first = parts.first().ok_or_else(|| candle_core::Error::Msg("State::cat: no states".into()))?;
        if parts.iter().any(|p| p.lif.len() != first.lif.len()) {
            candle_core::bail!("State::cat: states of different models");
        }
        if parts.len() == 1 {
            return Ok(first.clone());
        }
        let catv = |xs: Vec<&Tensor>| Tensor::cat(&xs, 0);
        // an Option tensor in every part, or zeros shaped like another part's where one is missing
        let cat_opt = |xs: Vec<Option<&Tensor>>, rows: &[usize]| -> Result<Option<Tensor>> {
            let Some(model) = xs.iter().flatten().next() else { return Ok(None) };
            let mut out = Vec::with_capacity(xs.len());
            for (x, &r) in xs.iter().zip(rows) {
                out.push(match x {
                    Some(t) => (*t).clone(),
                    None => {
                        let mut dims = model.dims().to_vec();
                        dims[0] = r;
                        Tensor::zeros(dims, DType::F32, model.device())?
                    }
                });
            }
            Ok(Some(Tensor::cat(&out, 0)?))
        };
        let rows: Vec<usize> = parts.iter().map(State::rows).collect();
        let nl = first.lif.len();
        let mut lif = Vec::with_capacity(nl);
        let mut mem = Vec::with_capacity(nl);
        let mut kv = Vec::with_capacity(nl);
        let mut pstate = Vec::with_capacity(nl);
        for i in 0..nl {
            lif.push(match &first.lif[i] {
                None => None,
                Some(_) => {
                    let ls: Vec<&LifState> = parts.iter().map(|p| p.lif[i].as_ref().expect("same model")).collect();
                    Some(LifState {
                        v: catv(ls.iter().map(|l| &l.v).collect())?,
                        s: catv(ls.iter().map(|l| &l.s).collect())?,
                        a: cat_opt(ls.iter().map(|l| l.a.as_ref()).collect(), &rows)?,
                    })
                }
            });
            mem.push(match &first.mem[i] {
                None => None,
                Some(_) => {
                    let ms: Vec<&MemState> = parts.iter().map(|p| p.mem[i].as_ref().expect("same model")).collect();
                    Some(MemState { f: catv(ms.iter().map(|m| &m.f).collect())?, s: cat_opt(ms.iter().map(|m| m.s.as_ref()).collect(), &rows)?, prev: cat_opt(ms.iter().map(|m| m.prev.as_ref()).collect(), &rows)? })
                }
            });
            kv.push(cat_kv(parts.iter().map(|p| p.kv[i].as_ref()).collect(), &rows)?);
            let ps: Vec<&PartState> = parts.iter().map(|p| &p.parts[i]).collect();
            pstate.push(PartState {
                stp: match &ps[0].stp {
                    None => None,
                    Some(_) => Some((
                        catv(ps.iter().map(|p| &p.stp.as_ref().expect("same model").0).collect())?,
                        catv(ps.iter().map(|p| &p.stp.as_ref().expect("same model").1).collect())?,
                    )),
                },
                delays: cat_opt(ps.iter().map(|p| p.delays.as_ref()).collect(), &rows)?,
                wm: cat_opt(ps.iter().map(|p| p.wm.as_ref()).collect(), &rows)?,
                res: match &ps[0].res {
                    None => None,
                    Some(_) => Some((
                        catv(ps.iter().map(|p| &p.res.as_ref().expect("same model").0).collect())?,
                        catv(ps.iter().map(|p| &p.res.as_ref().expect("same model").1).collect())?,
                    )),
                },
                tem: cat_kv(ps.iter().map(|p| p.tem.as_ref()).collect(), &rows)?,
                tem_phase: cat_opt(ps.iter().map(|p| p.tem_phase.as_ref()).collect(), &rows)?,
                ws: cat_opt(ps.iter().map(|p| p.ws.as_ref()).collect(), &rows)?,
                holo: match &ps[0].holo {
                    None => None,
                    Some(_) => Some((
                        catv(ps.iter().map(|p| &p.holo.as_ref().expect("same model").0).collect())?,
                        catv(ps.iter().map(|p| &p.holo.as_ref().expect("same model").1).collect())?,
                    )),
                },
            });
        }
        let modc = match &first.modc {
            None => None,
            Some(_) => {
                let cs: Vec<&ModCarry> = parts.iter().map(|p| p.modc.as_ref().expect("same model")).collect();
                Some(ModCarry { rows: cs.iter().flat_map(|c| c.rows.iter().copied()).collect(), prev: catv(cs.iter().map(|c| &c.prev).collect())? })
            }
        };
        let shadow = match &first.shadow {
            None => None,
            Some(_) => {
                let ss: Vec<State> = parts.iter().map(|p| (**p.shadow.as_ref().expect("same model")).clone()).collect();
                Some(Box::new(State::cat(&ss)?))
            }
        };
        let loop2 = match &first.loop2 {
            None => None,
            Some(_) => {
                let ss: Vec<State> = parts.iter().map(|p| (**p.loop2.as_ref().expect("same model")).clone()).collect();
                Some(Box::new(State::cat(&ss)?))
            }
        };
        let fb_any = parts.iter().any(|p| p.fb.is_some());
        if fb_any && parts.iter().any(|p| p.seen.iter().any(|&s| s % FB_CHUNK != first.seen[0] % FB_CHUNK)) {
            candle_core::bail!("State::cat: --feedback rows must stand at the same place in their {FB_CHUNK}-token chunk");
        }
        Ok(State {
            seen: parts.iter().flat_map(|p| p.seen.iter().copied()).collect(),
            lif,
            mem,
            kv,
            eng: cat_kv(parts.iter().map(|p| p.eng.as_ref()).collect(), &rows)?,
            hist: parts.iter().flat_map(|p| p.hist.iter().cloned()).collect(),
            fb: cat_opt(parts.iter().map(|p| p.fb.as_ref()).collect(), &rows)?,
            parts: pstate,
            cap: first.cap,
            modc,
            shadow,
            loop2,
            v51: crate::v51::V51State::cat(&parts.iter().map(|p| &p.v51).collect::<Vec<_>>())?,
        })
    }

    fn device(&self) -> Device {
        for l in self.lif.iter().flatten() {
            return l.v.device().clone();
        }
        for k in self.kv.iter().flatten() {
            return k.k.device().clone();
        }
        Device::Cpu
    }
}

/// Stores of several states joined row-wise, each left-padded to the longest (masked slots).
fn cat_kv(xs: Vec<Option<&Kv>>, rows: &[usize]) -> Result<Option<Kv>> {
    let Some(model) = xs.iter().flatten().next() else { return Ok(None) };
    let smax = xs.iter().flatten().map(|k| k.k.dim(2).unwrap_or(0)).max().unwrap_or(0);
    let (_, nh, _, hs) = model.k.dims4()?;
    // the values' width (B10's store keeps codes of 2K and inputs of d; attention and the engram have both hs)
    let vhs = model.v.dim(3)?;
    let dev = model.k.device();
    let has = |f: &dyn Fn(&Kv) -> bool| xs.iter().flatten().any(|k| f(k));
    let (want_logp, want_reads) = (has(&|k| k.logp.is_some()), has(&|k| k.reads.is_some()));
    let tails: Vec<usize> = xs.iter().flatten().map(|k| k.ktail.as_ref().map_or(0, |t| t.dim(2).unwrap_or(0))).collect();
    if tails.iter().any(|&l| l != tails[0]) || (tails[0] > 0 && xs.iter().any(|k| k.is_none())) {
        candle_core::bail!("State::cat: --btsp rows whose eligibility windows differ in length cannot be joined yet");
    }
    let (mut ks, mut vs, mut pads, mut lps, mut rds, mut ages, mut tls, mut valid) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut any_pad = false;
    for (x, &r) in xs.iter().zip(rows) {
        let s = x.map_or(0, |k| k.k.dim(2).unwrap_or(0));
        let extra = smax - s;
        any_pad |= extra > 0 || x.is_some_and(|k| k.pad.is_some());
        let lead = |w: usize| Tensor::zeros((r, nh, extra, w), DType::F32, dev);
        let front = |t: Option<&Tensor>, fill: f32| -> Result<Tensor> {
            // no empty fill: candle's CUDA fill kernel refuses zero elements (CUDA_ERROR_INVALID_VALUE), which made
            // every engram model's batched generation fail on the GPU while the CPU ran
            if extra == 0 {
                return match t {
                    Some(t) => Ok(t.clone()),
                    None => Tensor::zeros((r, s), DType::F32, dev),
                };
            }
            let head = Tensor::full(fill, (r, extra), dev)?;
            match t {
                Some(t) => Tensor::cat(&[&head, t], 1),
                None if s > 0 => Tensor::cat(&[&head, &Tensor::zeros((r, s), DType::F32, dev)?], 1),
                None => Ok(head),
            }
        };
        match x {
            Some(k) => {
                ks.push(if extra > 0 { Tensor::cat(&[&lead(hs)?, &k.k], 2)? } else { k.k.clone() });
                vs.push(if extra > 0 { Tensor::cat(&[&lead(vhs)?, &k.v], 2)? } else { k.v.clone() });
                valid.extend(k.valid.iter().copied());
            }
            None => {
                ks.push(lead(hs)?);
                vs.push(lead(vhs)?);
                valid.extend(std::iter::repeat(0).take(r));
            }
        }
        pads.push(front(x.and_then(|k| k.pad.as_ref()), NEG)?);
        if want_logp {
            lps.push(front(x.and_then(|k| k.logp.as_ref()), 0.0)?);
        }
        if want_reads {
            rds.push(front(x.and_then(|k| k.reads.as_ref()), 0.0)?);
            ages.push(front(x.and_then(|k| k.age.as_ref()), 0.0)?);
        }
        if let Some(t) = x.and_then(|k| k.ktail.as_ref()) {
            tls.push(t.clone());
        }
    }
    let c = |v: Vec<Tensor>| Tensor::cat(&v, 0);
    Ok(Some(Kv {
        k: c(ks)?,
        v: c(vs)?,
        pad: if any_pad { Some(c(pads)?) } else { None },
        valid,
        logp: if want_logp { Some(c(lps)?) } else { None },
        ktail: if tls.is_empty() { None } else { Some(c(tls)?) },
        reads: if want_reads { Some(c(rds)?) } else { None },
        age: if want_reads { Some(c(ages)?) } else { None },
    }))
}

/// The LIF scan of `cur` [B, T, d] from `init` (None: at rest), for inference: binary spikes, the threshold `theta`
/// [d] (None: 1). On the CPU it is the fused scan's own loop (the same rounding, step for step); elsewhere the same
/// steps in candle ops.
pub(crate) fn lif_infer(cur: &Tensor, beta: &Tensor, theta: Option<&Tensor>, init: Option<&LifState>) -> Result<(Tensor, LifState)> {
    let (b, t, d) = cur.dims3()?;
    let dev = cur.device();
    if dev.is_cpu() {
        let c: Vec<f32> = cur.flatten_all()?.to_vec1()?;
        let be: Vec<f32> = beta.to_vec1()?;
        let th: Vec<f32> = match theta {
            Some(x) => x.to_vec1()?,
            None => vec![LIF_THRESHOLD as f32; d],
        };
        let (mut v, mut sp) = match init {
            Some(s) => (s.v.flatten_all()?.to_vec1::<f32>()?, s.s.flatten_all()?.to_vec1::<f32>()?),
            None => (vec![0f32; b * d], vec![0f32; b * d]),
        };
        let mut out = vec![0f32; b * t * d];
        for bi in 0..b {
            for ch in 0..d {
                let (mut vv, mut ss) = (v[bi * d + ch], sp[bi * d + ch]);
                for ti in 0..t {
                    let k = (bi * t + ti) * d + ch;
                    vv = be[ch] * vv + c[k] - th[ch] * ss;
                    ss = if vv - th[ch] >= 0.0 { 1.0 } else { 0.0 };
                    out[k] = ss;
                }
                v[bi * d + ch] = vv;
                sp[bi * d + ch] = ss;
            }
        }
        return Ok((Tensor::from_vec(out, (b, t, d), dev)?, LifState { v: Tensor::from_vec(v, (b, d), dev)?, s: Tensor::from_vec(sp, (b, d), dev)?, a: None }));
    }
    lif_infer_ops(cur, beta, theta, init)
}

/// `lif_infer` in candle ops (the path off the CPU; on the CPU the tests check it against the loop).
pub(crate) fn lif_infer_ops(cur: &Tensor, beta: &Tensor, theta: Option<&Tensor>, init: Option<&LifState>) -> Result<(Tensor, LifState)> {
    let (b, t, d) = cur.dims3()?;
    let dev = cur.device();
    let th = match theta {
        Some(x) => x.clone(),
        None => Tensor::full(LIF_THRESHOLD as f32, d, dev)?,
    };
    let (mut v, mut sp) = match init {
        Some(s) => (s.v.clone(), s.s.clone()),
        None => (Tensor::zeros((b, d), DType::F32, dev)?, Tensor::zeros((b, d), DType::F32, dev)?),
    };
    let mut out = Vec::with_capacity(t);
    for ti in 0..t {
        let i_t = cur.narrow(1, ti, 1)?.squeeze(1)?;
        v = v.broadcast_mul(beta)?.add(&i_t)?.sub(&sp.broadcast_mul(&th)?)?;
        sp = v.broadcast_sub(&th)?.ge(0.0)?.to_dtype(DType::F32)?;
        out.push(sp.clone());
    }
    Ok((Tensor::stack(&out, 1)?, LifState { v, s: sp, a: None }))
}

/// E9 short-term plasticity from a carried (u, x): the part's own forward (ava_parts::stp) with its start state given.
fn stp_from(p: &ava_parts::stp::Stp, s: &Tensor, init: &(Tensor, Tensor)) -> Result<(Tensor, (Tensor, Tensor))> {
    use candle_nn::ops::sigmoid as sg;
    let t = s.dim(1)?;
    let (uu, df, dd) = (sg(&p.u_param)?, sg(&p.f_param)?, sg(&p.d_param)?);
    let a = ava_parts::stp::events(s)?; // the event size (= s for 0/1 spikes)
    let (mut u, mut x) = (init.0.clone(), init.1.clone());
    let mut effs = Vec::with_capacity(t);
    for k in 0..t {
        let sk = a.narrow(1, k, 1)?.squeeze(1)?;
        let u_plus = (&u + (u.affine(-1.0, 1.0)?.broadcast_mul(&uu)? * &sk)?)?;
        let e = (&u_plus * &x)?;
        effs.push(e.broadcast_div(&uu)?);
        let x_used = (&x - (&e * &sk)?)?;
        u = u_plus.broadcast_mul(&df)?;
        x = x_used.affine(-1.0, 1.0)?.broadcast_mul(&dd)?.affine(-1.0, 1.0)?;
    }
    let efficacy = Tensor::stack(&effs, 1)?;
    let out = (s + ((s * &efficacy)? - s)?.broadcast_mul(&p.mix)?)?;
    Ok((out, (u, x)))
}

/// B9 working memory from carried slots m0 [B, K, dk]: the part's own forward (ava_parts::wm) with its start state.
fn wm_from(p: &ava_parts::wm::SlotMemory, h: &Tensor, m0: &Tensor) -> Result<(Tensor, Tensor)> {
    use candle_core::D;
    let (b, t, _) = h.dims3()?;
    let g = p.gates(h)?;
    let v = ava_parts::linear(h, &p.wv)?.reshape((b, t, p.slots, p.dk))?;
    let mut m = m0.clone();
    let mut ms = Vec::with_capacity(t);
    for i in 0..t {
        let gi = g.narrow(1, i, 1)?.squeeze(1)?.unsqueeze(2)?;
        let vi = v.narrow(1, i, 1)?.squeeze(1)?;
        m = (m.broadcast_mul(&gi.affine(-1.0, 1.0)?)? + vi.broadcast_mul(&gi)?)?;
        ms.push(m.clone());
    }
    let mm = Tensor::stack(&ms, 1)?;
    let q = ava_parts::linear(h, &p.wq)?.unsqueeze(2)?;
    let att = (mm.broadcast_mul(&q)?.sum(D::Minus1)? / (p.dk as f64).sqrt())?;
    let att = candle_nn::ops::softmax(&att, D::Minus1)?.unsqueeze(3)?;
    let r = mm.broadcast_mul(&att)?.sum(2)?;
    Ok((ava_parts::linear(&r, &p.wo)?, m))
}

/// S2: drop slots of `store` over the cap (see the top of the file).
pub(crate) fn evict(store: &mut Option<Kv>, cap: EngramCap) -> Result<()> {
    let Some(kv) = store.as_mut() else { return Ok(()) };
    let s = kv.k.dim(2)?;
    if s <= cap.max {
        return Ok(());
    }
    let b = kv.k.dim(0)?;
    let counted = kv.reads.is_some() && kv.age.is_some();
    if cap.rule == Evict::Oldest || !counted {
        let drop = s - cap.max;
        let cut = |x: &Tensor, dim: usize| x.narrow(dim, drop, cap.max);
        let cuto = |x: &Option<Tensor>| x.as_ref().map(|x| cut(x, 1)).transpose();
        kv.k = cut(&kv.k, 2)?;
        kv.v = cut(&kv.v, 2)?;
        kv.pad = cuto(&kv.pad)?;
        kv.logp = cuto(&kv.logp)?;
        kv.reads = cuto(&kv.reads)?;
        kv.age = cuto(&kv.age)?;
        for v in kv.valid.iter_mut() {
            *v = (*v).min(cap.max);
        }
    } else {
        let reads: Vec<Vec<f32>> = kv.reads.as_ref().expect("counted").to_vec2()?;
        let age: Vec<Vec<f32>> = kv.age.as_ref().expect("counted").to_vec2()?;
        let pad: Option<Vec<Vec<f32>>> = kv.pad.as_ref().map(|p| p.to_vec2()).transpose()?;
        let dev = kv.k.device().clone();
        let mut keep_rows: Vec<Tensor> = Vec::with_capacity(b);
        for r in 0..b {
            // eviction order: padding, then read slots by mean reads per later query (oldest first on ties), then
            // the slots no later query has seen yet (oldest first)
            let mut order: Vec<(u8, f32, usize)> = (0..s)
                .map(|j| {
                    if pad.as_ref().is_some_and(|p| p[r][j] < 0.0) {
                        (0, 0.0, j)
                    } else if age[r][j] >= 1.0 {
                        (1, reads[r][j] / age[r][j], j)
                    } else {
                        (2, 0.0, j)
                    }
                })
                .collect();
            order.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)));
            let mut keep: Vec<u32> = order[s - cap.max..].iter().map(|x| x.2 as u32).collect();
            keep.sort_unstable();
            kv.valid[r] = keep.iter().filter(|&&j| !pad.as_ref().is_some_and(|p| p[r][j as usize] < 0.0)).count();
            keep_rows.push(Tensor::from_vec(keep, cap.max, &dev)?);
        }
        let gather4 = |x: &Tensor| -> Result<Tensor> {
            let rows: Vec<Tensor> = (0..b).map(|r| x.get(r)?.index_select(&keep_rows[r], 1)).collect::<Result<_>>()?;
            Tensor::stack(&rows, 0)
        };
        let gather2 = |x: &Option<Tensor>| -> Result<Option<Tensor>> {
            match x {
                None => Ok(None),
                Some(x) => {
                    let rows: Vec<Tensor> = (0..b).map(|r| x.get(r)?.index_select(&keep_rows[r], 0)).collect::<Result<_>>()?;
                    Ok(Some(Tensor::stack(&rows, 0)?))
                }
            }
        };
        kv.k = gather4(&kv.k)?;
        kv.v = gather4(&kv.v)?;
        kv.pad = gather2(&kv.pad)?;
        kv.logp = gather2(&kv.logp)?;
        kv.reads = gather2(&kv.reads)?;
        kv.age = gather2(&kv.age)?;
    }
    if kv.pad.is_some() && kv.valid.iter().all(|&v| v == cap.max) {
        kv.pad = None;
    }
    Ok(())
}

impl Model {
    /// S1: the state at the start of a window for `rows` row streams (see the top of the file).
    pub fn state(&self, rows: usize) -> Result<State> {
        self.state_with(rows, true)
    }

    /// `state`; with `shadow`, --modulate's shadow state and statistics too (the shadow state itself has none).
    fn state_with(&self, rows: usize, shadow: bool) -> Result<State> {
        if matches!(self.cfg.kind, Kind::F | Kind::G | Kind::R) {
            candle_core::bail!("state: model {} reads with Model::stream / step (forward_with_state is for S, D, A and T)", self.cfg.kind.name());
        }
        if rows == 0 {
            candle_core::bail!("state: at least one row");
        }
        let dev = self.tok.device();
        let (d, ar) = (self.cfg.d, self.cfg.arch);
        let z = |dims: &[usize]| Tensor::zeros(dims, DType::F32, dev);
        let mut lif = Vec::with_capacity(self.blocks.len());
        let mut mem = Vec::with_capacity(self.blocks.len());
        let mut parts = Vec::with_capacity(self.blocks.len());
        for (i, blk) in self.blocks.iter().enumerate() {
            let alif = self.ext.as_ref().is_some_and(|e| e.blocks[i].alif.is_some());
            lif.push(match &blk.mem {
                Some(_) => Some(LifState { v: z(&[rows, d])?, s: z(&[rows, d])?, a: if alif { Some(z(&[rows, d])?) } else { None } }),
                None => None,
            });
            mem.push(match &blk.hebb {
                Some(_) => {
                    let hs = ar.head_size;
                    let f = z(&[rows * (d / hs), hs, hs])?;
                    Some(MemState { s: if ar.mem_momentum > 0.0 { Some(f.clone()) } else { None }, f, prev: None })
                }
                None => None,
            });
            let px = self.ext.as_ref().and_then(|e| e.mem.parts.get(i));
            parts.push(PartState {
                stp: match px.and_then(|p| p.stp.as_ref()) {
                    Some(_) => Some((z(&[rows, d])?, Tensor::ones((rows, d), DType::F32, dev)?)),
                    None => None,
                },
                delays: match px.and_then(|p| p.delays.as_ref()) {
                    Some(dl) => Some(z(&[rows, dl.k - 1, d])?),
                    None => None,
                },
                wm: match px.and_then(|p| p.wm.as_ref()) {
                    Some((_, w)) => Some(z(&[rows, w.slots, w.dk])?),
                    None => None,
                },
                res: match self.ext.as_ref().and_then(|e| e.mem.w5.blocks.get(i)).and_then(|w| w.res.as_ref()) {
                    Some(_) => Some((z(&[rows, d])?, z(&[rows, d])?)),
                    None => None,
                },
                tem: None,
                tem_phase: match self.ext.as_ref().and_then(|e| e.mem.w5.blocks.get(i)).and_then(|w| w.tem.as_ref()) {
                    Some(tp) => Some(z(&[rows, tp.k])?),
                    None => None,
                },
                ws: match self.ext.as_ref().and_then(|e| e.mem.w5.blocks.get(i)).and_then(|w| w.ws.as_ref()) {
                    Some((_, w)) => Some(z(&[rows, w.slots, w.dk])?),
                    None => None,
                },
                holo: match self.ext.as_ref().and_then(|e| e.mem.w5.blocks.get(i)).and_then(|w| w.holo.as_ref()) {
                    Some((_, hm)) => Some((z(&[rows, hm.db])?, z(&[rows, hm.db])?)),
                    None => None,
                },
            });
        }
        // v50 (--loop): the later passes' state (made with the top-level state only; --loop refuses --modulate)
        let loop2 = if ar.loops > 1 && shadow { Some(Box::new(self.state_with(rows, false)?)) } else { None };
        let (modc, shadow) = if ar.mem.modulate && shadow {
            let c = ModCarry { rows: vec![Default::default(); rows], prev: z(&[rows, self.cfg.vocab])? };
            (Some(c), Some(Box::new(self.state_with(rows, false)?)))
        } else {
            (None, None)
        };
        Ok(State {
            seen: vec![0; rows],
            lif,
            mem,
            kv: vec![None; self.blocks.len()],
            eng: None,
            hist: vec![Vec::new(); rows],
            fb: None,
            parts,
            cap: None,
            modc,
            shadow,
            loop2,
            v51: crate::v51::V51State::new(self, rows, dev)?,
        })
    }

    /// S1: read `ids` [B, t] (u32) after what `st` has read; the logits [B·t, V]. See the top of the file.
    pub fn forward_with_state(&self, ids: &Tensor, st: &mut State) -> Result<Tensor> {
        let _q8 = self.int8_scope(); // E10: the int8 copies while on (src/int8.rs)
        let (b, t) = ids.dims2()?;
        if b != st.rows() || t == 0 {
            candle_core::bail!("forward_with_state: ids [{b}, {t}] for a state of {} rows", st.rows());
        }
        if st.lif.len() != self.blocks.len() {
            candle_core::bail!("forward_with_state: the state was made for another model ({} blocks, this one {})", st.lif.len(), self.blocks.len());
        }
        // B8/F13 (--modulate): the shadow pass reads the same tokens with its own state, and its surprise drives the
        // modulators (as Model::forward does it, so a fresh state gives forward's logits exactly)
        let w5 = self.ext.as_ref().map(|e| &e.mem.w5);
        let msig = match w5 {
            Some(w) if self.cfg.arch.mem.modulate && !w.mod_off.get() => {
                let mut sh = st.shadow.take().ok_or_else(|| candle_core::Error::Msg("forward_with_state: --modulate needs the state's shadow (Model::state makes it)".into()))?;
                w.mod_off.set(true);
                let r = self.forward_with_state(ids, &mut sh);
                w.mod_off.set(false);
                st.shadow = Some(sh);
                let logits = r?.detach();
                let carry = st.modc.as_mut().ok_or_else(|| candle_core::Error::Msg("forward_with_state: --modulate needs the state's statistics".into()))?;
                Some(crate::wave5::mod_signals(&logits, ids, Some(carry))?)
            }
            _ => None,
        };
        let cfg = self.cfg;
        let (d, n) = (cfg.d, b * t);
        let dev = ids.device();
        let fast = crate::fast_path() && dev.is_cuda();
        let mut x = if fast { crate::fused::embedding(&self.tok, ids)? } else { self.tok.embedding(&ids.flatten_all()?)? };
        let (place_off, mag_off, _) = self.lesion.get();
        let rows: Vec<Vec<u32>> = ids.to_vec2()?;
        // the glance and the magnitude code read the new tokens after each row's history
        let full: Vec<Vec<u32>> = rows.iter().zip(&st.hist).map(|(r, h)| h.iter().chain(r.iter()).copied().collect()).collect();
        if let (Some(place), false) = (&self.place, place_off) {
            let mut pid = Vec::with_capacity(n);
            for (f, h) in full.iter().zip(&st.hist) {
                pid.extend_from_slice(&ava_data::numbers::place_ids(f)[h.len()..]);
            }
            x = x.add(&Model::place_table(place)?.embedding(&Tensor::from_vec(pid, n, dev)?)?)?;
        }
        if let (Some(mw), false) = (&self.mag_w, mag_off) {
            let mut fe = Vec::with_capacity(n * crate::mag::MAG_F);
            for (f, h) in full.iter().zip(&st.hist) {
                fe.extend_from_slice(&crate::mag::feats_row(f)[h.len() * crate::mag::MAG_F..]);
            }
            x = x.add(&Tensor::from_vec(fe, (n, crate::mag::MAG_F), dev)?.matmul(mw)?)?;
        }
        let x = crate::v51::embed(self, ids, x)?; // v51 R: the letters (x itself without --letters)
        crate::v51::begin(self, &x, b, t, Some(&mut st.v51))?; // v51 Y: the rhythms
        let mut x = x;
        if let Some(pos) = &self.pos {
            let top = st.seen.iter().max().copied().unwrap_or(0) + t;
            if top > cfg.ctx {
                candle_core::bail!("forward_with_state: position {top} is past the position table ({})", cfg.ctx);
            }
            if st.seen.iter().all(|&s| s == st.seen[0]) {
                x = x.reshape((b, t, d))?.broadcast_add(&pos.narrow(0, st.seen[0], t)?)?.reshape((n, d))?;
            } else {
                let idx: Vec<u32> = st.seen.iter().flat_map(|&s| (s..s + t).map(|p| p as u32)).collect();
                x = x.add(&pos.index_select(&Tensor::from_vec(idx, n, dev)?, 0)?)?;
            }
        }
        // O1: the window runs in the 64-token chunks of the whole text, so pieces end where a full pass's chunks end
        let ar = cfg.arch;
        let s0 = st.seen[0];
        let segs: Vec<(usize, usize)> = if ar.feedback {
            if st.seen.iter().any(|&s| s != s0) {
                candle_core::bail!("forward_with_state: --feedback rows must have read the same number of tokens");
            }
            let mut v = Vec::new();
            let mut p = 0;
            while p < t {
                let to_edge = FB_CHUNK - (s0 + p) % FB_CHUNK;
                let len = to_edge.min(t - p);
                v.push((p, len));
                p += len;
            }
            v
        } else {
            vec![(0, t)]
        };
        let x3 = if segs.len() > 1 { Some(x.reshape((b, t, d))?) } else { None };
        let mut outs: Vec<Tensor> = Vec::with_capacity(segs.len());
        // v50 (--loop N): the later passes read every token too, each with its own state (st.loop2)
        let np = self.v50_passes();
        let mut l2 = st.loop2.take();
        if let Some(s2) = l2.as_mut() {
            s2.cap = st.cap;
        }
        for &(t0, tc) in &segs {
            let mut xc = match &x3 {
                Some(x3) => x3.narrow(1, t0, tc)?.contiguous()?.reshape((b * tc, d))?,
                None => x.clone(),
            };
            if let Some(f) = &st.fb {
                xc = xc.reshape((b, tc, d))?.broadcast_add(&f.unsqueeze(1)?)?.reshape((b * tc, d))?;
            }
            let x_in = xc.clone();
            for pass in 0..np {
                if pass > 0 {
                    xc = self.v51b_th_td(&x_in, self.v50_loop_input(&x_in, &xc)?)?; // v51b --thalamus, as in arch.rs forward_arch
                }
                self.v50_begin_pass(pass + 1 == np);
                self.v51b_th_begin(&xc)?;
                let sp: &mut State = if pass == 0 { &mut *st } else { l2.as_deref_mut().ok_or_else(|| candle_core::Error::Msg("forward_with_state: --loop needs the state's loop2 (Model::state makes it)".into()))? };
                let mut streams: Vec<Tensor> = if ar.dense || ar.attn_res { vec![xc.clone()] } else { Vec::new() };
                let mut v0: Option<Tensor> = None;
                // v51 B-4 (src/v51e.rs): the forgetting test's shared trunk (None and false unless it is on)
                let (replay, recording) = (self.trunk_replay_k(), self.trunk_recording());
                for (i, blk) in self.blocks.iter().enumerate() {
                    if let Some(k) = replay {
                        if i < k {
                            continue;
                        }
                        if i == k {
                            (xc, streams, v0) = self.trunk_get(b, tc)?;
                        }
                    }
                    let xin = xc.clone(); // v44 K1: the block's input
                    self.v51b_th_block(i)?;
                    xc = self.block_state(i, blk, &xc, b, tc, sp, &mut v0, msig.as_ref())?;
                    xc = self.v51b_th_out(i, &xin, xc)?;
                    if let Some(w) = self.ext.as_ref().and_then(|e| e.blocks[i].dense.as_ref()) {
                        streams.push(xc.clone());
                        let mut mix = xc.clone();
                        for (j, sj) in streams.iter().enumerate() {
                            mix = mix.add(&sj.broadcast_mul(&w.narrow(0, j, 1)?)?)?;
                        }
                        xc = mix;
                    }
                    if let Some(q) = self.ext.as_ref().and_then(|e| e.blocks[i].attn_res.as_ref()) {
                        // v44 K1: Attention Residuals, as in arch.rs forward_arch
                        streams.push(xc.sub(&xin)?);
                        xc = crate::arch::attn_res_mix(&streams, q)?;
                    }
                    self.v51b_th_after(i, &xc)?;
                    if i == 0 {
                        self.v50_after_block0(&xc)?; // v50 --neuromod: the nucleus reads block 0's output
                        // v51 DA, M: the surprise and the mood, carried in this pass's state (st itself without --loop;
                        // under --loop each pass keeps its own, as each pass of forward_arch reads its own block 0)
                        crate::v51::after0(self, &xc, ids, b, tc, Some(&mut sp.v51))?;
                    }
                    if recording {
                        self.trunk_put(i, &xc, &streams, &v0, b, tc)?;
                    }
                }
                if replay == Some(self.blocks.len()) {
                    xc = self.trunk_get(b, tc)?.0;
                }
            }
            self.v50_end();
            self.v51b_th_end();
            if let Some(wf) = self.ext.as_ref().and_then(|e| e.fb.as_ref()) {
                if (s0 + t0 + tc) % FB_CHUNK == 0 {
                    let top = xc.reshape((b, tc, d))?.narrow(1, tc - 1, 1)?.squeeze(1)?;
                    st.fb = Some(rmsnorm(&top, &self.final_norm)?.matmul(wf)?);
                }
            }
            outs.push(xc);
        }
        crate::v51::end(self);
        let xo = if outs.len() == 1 {
            outs.pop().expect("one piece")
        } else {
            let parts: Vec<Tensor> = outs.iter().map(|o| o.reshape((b, o.dim(0)? / b, d))).collect::<Result<_>>()?;
            Tensor::cat(&parts, 1)?.reshape((n, d))?
        };
        for (r, f) in full.into_iter().enumerate() {
            st.seen[r] += t;
            st.hist[r] = f[f.len().saturating_sub(HIST)..].to_vec();
        }
        if let Some(s2) = l2.as_mut() {
            // v50 (--loop): the later passes have read the same tokens
            s2.seen.clone_from(&st.seen);
            s2.hist.clone_from(&st.hist);
        }
        st.loop2 = l2;
        let xo = norm(&xo, &self.final_norm)?;
        let xo = self.v51b_cb(xo, ids)?; // v51b --cerebellum-head (xo itself without it)
        crate::int8::head(&xo, &self.tok)
    }

    /// One block of `forward_with_state` on the rows of a piece (b rows of t tokens): the same ops as `forward` (and
    /// `forward_arch` for a model with block options), with every state taken from and put back into `st`.
    #[allow(clippy::too_many_arguments)]
    fn block_state(&self, i: usize, blk: &Block, x: &Tensor, b: usize, t: usize, st: &mut State, v0: &mut Option<Tensor>, msig: Option<&ModSig>) -> Result<Tensor> {
        let ar = self.cfg.arch;
        let (d, n) = (self.cfg.d, b * t);
        let bx = self.ext.as_ref().map(|e| &e.blocks[i]);
        let px = self.ext.as_ref().and_then(|e| e.mem.parts.get(i));
        // src/wave5.rs parts, as in arch.rs `block_arch`
        let w5 = self.ext.as_ref().map(|e| &e.mem.w5);
        let wb = w5.and_then(|w| w.blocks.get(i));
        let mods = match (wb.and_then(|w| w.modl.as_ref()), msig) {
            (Some(md), Some(sg)) => Some(crate::wave5::block_mods(md, sg)?),
            _ => None,
        };
        // v49: step_trace's sink (src/model.rs): copies of what this block computed, never fed back (not in the shadow pass)
        let tr = self.tracing() && w5.map_or(true, |w| !w.mod_off.get()) && !self.v50_quiet();
        if tr {
            self.trace_begin(i);
        }
        crate::v51::at_block(self, i);
        let mut x = match &mods {
            Some((_, add)) => x.add(add)?,
            None => x.clone(),
        };
        if let Some(m) = &blk.mem {
            let h = norm(&x, &m.norm)?;
            let mut cur = self.plastic_mm(i, crate::eprop::Site::Win, &h, &m.win, b, t)?.reshape((b, t, d))?; // v51 E (src/eprop.rs): + a plastic session's change (here and at W_out, W1, W2)
            let beta = sigmoid(&m.beta)?;
            if ar.lif_scale {
                cur = cur.broadcast_mul(&beta.affine(-1.0, 1.0)?)?;
            }
            cur = self.v50_ne(cur, b, t)?; // v50 --neuromod NE (blocks after 0), as in arch.rs block_arch
            if let Some((md, _)) = &mods {
                cur = cur.broadcast_mul(&md.input.unsqueeze(2)?)?;
            }
            cur = crate::v51::lif_gain(self, i, cur)?; // v51 M: the mood's gain
            if let Some(dn) = px.and_then(|p| p.dend.as_ref()) {
                cur = cur.mul(&dn.gate(&h.reshape((b, t, d))?)?)?;
            }
            if let Some(dl) = px.and_then(|p| p.delays.as_ref()) {
                let hist = st.parts[i].delays.take().expect("state has the delays' history");
                let k1 = hist.dim(1)?;
                let all = Tensor::cat(&[&hist, &cur], 1)?;
                cur = dl.forward(&all)?.narrow(1, k1, t)?;
                st.parts[i].delays = Some(all.narrow(1, t, k1)?.contiguous()?);
            }
            let theta = bx.and_then(|e| e.theta.as_ref());
            let alif = match bx.and_then(|e| e.alif.as_ref()) {
                Some((r, g)) => Some((sigmoid(r)?, g.clone())),
                None => None,
            };
            let (s, ls) = if ar.spikes == Spikes::Binary && alif.is_none() {
                lif_infer(&cur, &beta, theta, st.lif[i].as_ref())?
            } else {
                lif_ref(&cur, &beta, theta, alif.as_ref().map(|(r, g)| (r, g)), ar.spikes, ar.slope as f64, st.lif[i].as_ref())?
            };
            st.lif[i] = Some(ls);
            let s = match wb.and_then(|w| w.res.as_ref()) {
                Some(rs) => {
                    let init = st.parts[i].res.take().expect("state has the resonators");
                    let (mem, next) = crate::wave5::res_from(&rs.part, &cur, &init)?;
                    st.parts[i].res = Some(next);
                    // the part's forward: spike(membrane - theta, slope)
                    let sr = ava_parts::spike(&(mem - LIF_THRESHOLD)?, ar.slope as f64)?;
                    crate::wave5::res_mix(&s, &sr, &rs.mix)?
                }
                None => s,
            };
            let s = match px.and_then(|p| p.stp.as_ref()) {
                Some(sp) => {
                    let init = st.parts[i].stp.take().expect("state has the stp state");
                    let (out, next) = stp_from(sp, &s, &init)?;
                    st.parts[i].stp = Some(next);
                    out
                }
                None => s,
            };
            if tr {
                let sp = s.reshape((n, d))?.detach();
                self.trace_put(|p| p.spikes = Some(sp));
            }
            x = x.add(&self.plastic_mm(i, crate::eprop::Site::Wout, &s.reshape((n, d))?, &m.wout, b, t)?)?;
        }
        if let Some(hb) = &blk.hebb {
            let h = norm(&x, &hb.norm)?;
            let md = mods.as_ref().map(|(md, _)| (&md.write, &md.reset));
            let y = self.mem_arch(&h, hb, bx, b, t, &mut st.mem[i], v0, true, md)?;
            crate::v51::put_ctx(self, &y); // v51 P: the dendrites' context
            if tr {
                self.trace_put(|p| p.hebb_read = Some(y.detach()));
            }
            x = x.add(&self.v51b_th_mem(y)?)?; // v51b --thalamus: the memory path's gate
        }
        if let Some(a) = &blk.attn {
            let h = norm(&x, &a.norm)?;
            let r = memory::read(&h, a, None, &MemOpts::default(), b, t, Some(&mut st.kv[i]), StoreUse::default(), false, None)?.expect("not quiet");
            x = x.add(&r)?;
        }
        if let Some(e) = &blk.eng {
            // a lesioned engram keeps writing its store (as Model::step does) and adds nothing
            let (_, _, eng_off) = self.lesion.get();
            let h = norm(&x, &e.norm)?;
            let ex = self.ext.as_ref().and_then(|x| x.mem.eng.as_ref());
            let usage = StoreUse { count_reads: st.cap.is_some_and(|c| c.rule == Evict::LeastRead) };
            // v45 W4: the engram's output gate (None without --eng-gate)
            let eg = bx.and_then(|e| e.eng_gate.as_ref()).map(|w| crate::arch::eng_gate_of(&h, w)).transpose()?;
            if let Some(r) = memory::read(&h, e, ex, &ar.mem, b, t, Some(&mut st.eng), usage, eng_off, eg.as_ref())? {
                if tr {
                    self.trace_put(|p| p.engram_read = Some(r.detach()));
                }
                x = x.add(&r)?;
            }
            if let Some(cap) = st.cap {
                evict(&mut st.eng, cap)?;
            }
        }
        if let Some((nm, wm)) = px.and_then(|p| p.wm.as_ref()) {
            let h = norm(&x, nm)?;
            let m0 = st.parts[i].wm.take().expect("state has the slots");
            let (o, m) = wm_from(wm, &h.reshape((b, t, d))?, &m0)?;
            st.parts[i].wm = Some(m);
            x = x.add(&o.reshape((n, d))?)?;
        }
        if let Some((nm, ws)) = wb.and_then(|w| w.ws.as_ref()) {
            // B3: the shared workspace's slots carry on, as the working memory's do
            let h = norm(&x, nm)?;
            let m0 = st.parts[i].ws.take().expect("state has the workspace");
            let (o, m) = wm_from(ws, &h.reshape((b, t, d))?, &m0)?;
            st.parts[i].ws = Some(m);
            x = x.add(&o.reshape((n, d))?)?;
        }
        if let Some((nm, hm)) = wb.and_then(|w| w.holo.as_ref()) {
            // D2: the binding trace carries on
            let h = norm(&x, nm)?.reshape((b, t, d))?;
            let t0 = st.parts[i].holo.take().expect("state has the binding trace");
            let (o, t1) = hm.forward_from(&h, Some(&t0))?;
            st.parts[i].holo = Some(t1);
            x = x.add(&o.reshape((n, d))?)?;
        }
        if let Some(tp) = wb.and_then(|w| w.tem.as_ref()) {
            // B10: the structure read over every token so far
            let h = norm(&x, &tp.norm)?;
            let mut store = st.parts[i].tem.take();
            let mut phase = st.parts[i].tem_phase.take().expect("state has the phase");
            let y = crate::wave5::tem_read(&h, tp, b, t, Some(&mut store), Some(&mut phase))?;
            st.parts[i].tem = store;
            st.parts[i].tem_phase = Some(phase);
            x = x.add(&y)?;
        }
        if let Some((nm, sy)) = wb.and_then(|w| w.sync.as_ref()) {
            // L5: one token at a time, nothing to carry
            x = x.add(&sy.forward(&norm(&x, nm)?)?)?;
        }
        let l = &blk.mlp;
        let x = if let Some(hl) = wb.and_then(|w| w.halt.as_ref()) {
            crate::wave5::halt_mlp(self, l, hl, &x, ar.mem.halting)?.0
        } else if ar.v51.mlp() {
            // v51 P, DA, W, Pr (src/v51.rs)
            let (a, wm, y) = crate::v51::mlp(self, i, &norm(&x, &l.norm)?, l, tr, false)?.expect("a v51 MLP option is on");
            if tr {
                self.trace_put(|p| {
                    p.winners = wm;
                    p.hidden = Some(a.detach());
                });
            }
            x.add(&y)?
        } else {
            let h = norm(&x, &l.norm)?;
            let u = self.plastic_mm(i, crate::eprop::Site::W1, &h, &l.w1, b, t)?;
            let fast = crate::fast_path() && u.device().is_cuda();
            let a = if self.cfg.kind.is_ava() {
                let k = self.cfg.kwta_k();
                // v49: the kWTA mask kept for step_trace's sink (the same ops: the mask, then u * mask, relu)
                let (a, wm) = if fast {
                    (crate::fused::kwta_relu(&u, k)?, if tr { Some(kwta_mask(&u, k)?.detach()) } else { None })
                } else {
                    let wm = kwta_mask(&u, k)?;
                    (u.mul(&wm)?.relu()?, Some(wm))
                };
                let a = if ar.relu2 { a.sqr()? } else { a };
                if tr {
                    self.trace_put(|p| {
                        p.winners = wm.map(|w| w.detach());
                        p.hidden = Some(a.detach());
                    });
                }
                a
            } else if fast {
                crate::fused::gelu(&u)?
            } else {
                u.gelu()?
            };
            // v50: --dale's effective W2 and --neuromod DA, as in arch.rs block_arch
            x.add(&self.v50_da(self.plastic_mm(i, crate::eprop::Site::W2, &a, &self.mlp_w2(i)?, b, t)?)?)?
        };
        let x = match (wb.and_then(|w| w.cb.as_ref()), w5.and_then(|w| w.ones.as_ref())) {
            (Some(cb), Some(ones)) => crate::wave5::cb_add(cb, ones, &x)?,
            _ => x,
        };
        match self.ext.as_ref().and_then(|e| e.mem.pkm.as_ref()) {
            Some(pk) if ar.mem.pk_block(self.cfg.layers) == Some(i) => x.add(&pk.forward(&x)?),
            _ => Ok(x),
        }
    }

    /// S2: a fingerprint of this model (FNV-1a over its config and a sample of its weights), so a saved state is only
    /// ever loaded into the model that wrote it.
    pub fn fingerprint(&self) -> Result<u64> {
        let mut h = Fnv::new();
        h.bytes(crate::v51::cfg_debug(&self.cfg).as_bytes()); // v51: the same text as before while its options are off
        let mut add = |t: &Tensor, stride: usize| -> Result<()> {
            let v: Vec<f32> = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
            for x in v.iter().step_by(stride.max(1)) {
                h.bytes(&x.to_le_bytes());
            }
            Ok(())
        };
        add(&self.tok, 97)?;
        add(&self.final_norm, 1)?;
        for blk in &self.blocks {
            add(&blk.mlp.w2, 101)?;
        }
        Ok(h.0)
    }
}

struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

// ------------------------------------------------------------------------------------------- S2: saving

fn u32s(v: &[usize], dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(v.iter().map(|&x| x as u32).collect::<Vec<u32>>(), v.len(), dev)
}

impl State {
    /// Every tensor of the state, by name (see `save`).
    fn tensors(&self, fingerprint: u64) -> Result<HashMap<String, Tensor>> {
        let dev = Device::Cpu;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        let mut put = |k: String, t: &Tensor| -> Result<()> {
            m.insert(k, t.to_device(&Device::Cpu)?);
            Ok(())
        };
        put("meta.seen".into(), &u32s(&self.seen, &dev)?)?;
        put("meta.model".into(), &Tensor::from_vec(vec![(fingerprint >> 32) as u32, fingerprint as u32], 2, &dev)?)?;
        put("meta.blocks".into(), &u32s(&[self.lif.len()], &dev)?)?;
        let cap = match self.cap {
            None => vec![0, 0],
            Some(c) => vec![c.max, if c.rule == Evict::Oldest { 0 } else { 1 }],
        };
        put("meta.cap".into(), &u32s(&cap, &dev)?)?;
        let mut hist = vec![0u32; self.rows() * HIST];
        let mut hlen = Vec::with_capacity(self.rows());
        for (r, h) in self.hist.iter().enumerate() {
            hist[r * HIST..r * HIST + h.len()].copy_from_slice(h);
            hlen.push(h.len());
        }
        put("meta.hist".into(), &Tensor::from_vec(hist, (self.rows(), HIST), &dev)?)?;
        put("meta.hist_len".into(), &u32s(&hlen, &dev)?)?;
        let kv_put = |p: &str, kv: &Kv, put: &mut dyn FnMut(String, &Tensor) -> Result<()>| -> Result<()> {
            put(format!("{p}.k"), &kv.k)?;
            put(format!("{p}.v"), &kv.v)?;
            put(format!("{p}.valid"), &u32s(&kv.valid, &dev)?)?;
            for (name, t) in [("pad", &kv.pad), ("logp", &kv.logp), ("ktail", &kv.ktail), ("reads", &kv.reads), ("age", &kv.age)] {
                if let Some(t) = t {
                    put(format!("{p}.{name}"), t)?;
                }
            }
            Ok(())
        };
        for i in 0..self.lif.len() {
            if let Some(l) = &self.lif[i] {
                put(format!("lif.{i}.v"), &l.v)?;
                put(format!("lif.{i}.s"), &l.s)?;
                if let Some(a) = &l.a {
                    put(format!("lif.{i}.a"), a)?;
                }
            }
            if let Some(ms) = &self.mem[i] {
                put(format!("mem.{i}.f"), &ms.f)?;
                if let Some(s) = &ms.s {
                    put(format!("mem.{i}.s"), s)?;
                }
                if let Some(p) = &ms.prev {
                    // v45 W3 (--token-shift): the last token's input
                    put(format!("mem.{i}.p"), p)?;
                }
            }
            if let Some(kv) = &self.kv[i] {
                kv_put(&format!("kv.{i}"), kv, &mut put)?;
            }
            let p = &self.parts[i];
            if let Some((u, x)) = &p.stp {
                put(format!("parts.{i}.stp_u"), u)?;
                put(format!("parts.{i}.stp_x"), x)?;
            }
            if let Some(x) = &p.delays {
                put(format!("parts.{i}.delays"), x)?;
            }
            if let Some(x) = &p.wm {
                put(format!("parts.{i}.wm"), x)?;
            }
            if let Some((re, im)) = &p.res {
                put(format!("parts.{i}.res_re"), re)?;
                put(format!("parts.{i}.res_im"), im)?;
            }
            if let Some(kv) = &p.tem {
                kv_put(&format!("tem.{i}"), kv, &mut put)?;
            }
            if let Some(x) = &p.tem_phase {
                put(format!("parts.{i}.tem_phase"), x)?;
            }
            if let Some(x) = &p.ws {
                put(format!("parts.{i}.ws"), x)?;
            }
            if let Some((re, im)) = &p.holo {
                put(format!("parts.{i}.holo_re"), re)?;
                put(format!("parts.{i}.holo_im"), im)?;
            }
        }
        if let Some(kv) = &self.eng {
            kv_put("eng", kv, &mut put)?;
        }
        if let Some(f) = &self.fb {
            put("fb".into(), f)?;
        }
        self.v51.save(&mut put)?;
        if let Some(c) = &self.modc {
            let rows: Vec<f32> = c.rows.iter().flat_map(|r| r.to_vec()).collect();
            put("mod.rows".into(), &Tensor::from_vec(rows, (self.rows(), 7), &dev)?)?;
            put("mod.prev".into(), &c.prev)?;
        }
        if let Some(sh) = &self.shadow {
            for (k, v) in sh.tensors(fingerprint)? {
                put(format!("shadow.{k}"), &v)?;
            }
        }
        if let Some(l2) = &self.loop2 {
            // v50 (--loop): the later passes' state
            for (k, v) in l2.tensors(fingerprint)? {
                put(format!("loop2.{k}"), &v)?;
            }
        }
        Ok(m)
    }

    /// S2: write the state to `path` (safetensors), through `path`.tmp and a rename (the file is whole or absent).
    pub fn save(&self, model: &Model, path: &Path) -> std::result::Result<(), String> {
        let fp = model.fingerprint().map_err(|e| format!("fingerprint: {e}"))?;
        let m = self.tensors(fp).map_err(|e| format!("state: {e}"))?;
        let tmp = path.with_extension("tmp");
        candle_core::safetensors::save(&m, &tmp).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))
    }

    /// S2: read a state `save` wrote, for `model` on `dev`. Err when the file is not one, or was written by another
    /// model (another fingerprint).
    pub fn load(model: &Model, path: &Path, dev: &Device) -> std::result::Result<State, String> {
        let e = |x: candle_core::Error| format!("{}: {x}", path.display());
        let t = candle_core::safetensors::load(path, dev).map_err(e)?;
        let get = |k: &str| t.get(k).cloned().ok_or_else(|| format!("{}: no {k}", path.display()));
        let ints = |k: &str| -> std::result::Result<Vec<usize>, String> { Ok(get(k)?.to_vec1::<u32>().map_err(e)?.into_iter().map(|x| x as usize).collect()) };
        let fp = ints("meta.model")?;
        let have = ((fp[0] as u64) << 32) | fp[1] as u64;
        if have != model.fingerprint().map_err(e)? {
            return Err(format!("{} was saved by another model (fingerprint {have:016x})", path.display()));
        }
        State::from_map(model, &t, "", path, true)
    }

    /// The state in `t` under the names `pre`.. (`load`; --modulate's shadow state is under "shadow.").
    fn from_map(model: &Model, t: &HashMap<String, Tensor>, pre: &str, path: &Path, with_shadow: bool) -> std::result::Result<State, String> {
        let e = |x: candle_core::Error| format!("{}: {x}", path.display());
        let get = |k: &str| t.get(&format!("{pre}{k}")).cloned().ok_or_else(|| format!("{}: no {pre}{k}", path.display()));
        let ints = |k: &str| -> std::result::Result<Vec<usize>, String> { Ok(get(k)?.to_vec1::<u32>().map_err(e)?.into_iter().map(|x| x as usize).collect()) };
        let nb = ints("meta.blocks")?[0];
        if nb != model.blocks.len() {
            return Err(format!("{}: {nb} blocks, the model has {}", path.display(), model.blocks.len()));
        }
        let seen = ints("meta.seen")?;
        let rows = seen.len();
        let cap = ints("meta.cap")?;
        let cap = if cap[0] == 0 { None } else { Some(EngramCap { max: cap[0], rule: if cap[1] == 0 { Evict::Oldest } else { Evict::LeastRead } }) };
        let hist_t: Vec<Vec<u32>> = get("meta.hist")?.to_vec2().map_err(e)?;
        let hlen = ints("meta.hist_len")?;
        let hist: Vec<Vec<u32>> = hist_t.into_iter().zip(&hlen).map(|(h, &l)| h[..l].to_vec()).collect();
        let opt = |k: String| t.get(&format!("{pre}{k}")).cloned();
        let kv_get = |p: &str| -> std::result::Result<Option<Kv>, String> {
            let Some(k) = opt(format!("{p}.k")) else { return Ok(None) };
            Ok(Some(Kv {
                k,
                v: get(&format!("{p}.v"))?,
                valid: ints(&format!("{p}.valid"))?,
                pad: opt(format!("{p}.pad")),
                logp: opt(format!("{p}.logp")),
                ktail: opt(format!("{p}.ktail")),
                reads: opt(format!("{p}.reads")),
                age: opt(format!("{p}.age")),
            }))
        };
        let mut st = model.state_with(rows, with_shadow).map_err(e)?;
        for i in 0..nb {
            if st.lif[i].is_some() {
                st.lif[i] = Some(LifState { v: get(&format!("lif.{i}.v"))?, s: get(&format!("lif.{i}.s"))?, a: opt(format!("lif.{i}.a")) });
            }
            if st.mem[i].is_some() {
                st.mem[i] = Some(MemState { f: get(&format!("mem.{i}.f"))?, s: opt(format!("mem.{i}.s")), prev: opt(format!("mem.{i}.p")) });
            }
            st.kv[i] = kv_get(&format!("kv.{i}"))?;
            let p = &mut st.parts[i];
            if p.stp.is_some() {
                p.stp = Some((get(&format!("parts.{i}.stp_u"))?, get(&format!("parts.{i}.stp_x"))?));
            }
            if p.delays.is_some() {
                p.delays = Some(get(&format!("parts.{i}.delays"))?);
            }
            if p.wm.is_some() {
                p.wm = Some(get(&format!("parts.{i}.wm"))?);
            }
            if p.res.is_some() {
                p.res = Some((get(&format!("parts.{i}.res_re"))?, get(&format!("parts.{i}.res_im"))?));
            }
            if p.tem_phase.is_some() {
                p.tem = kv_get(&format!("tem.{i}"))?;
                p.tem_phase = Some(get(&format!("parts.{i}.tem_phase"))?);
            }
            if p.ws.is_some() {
                p.ws = Some(get(&format!("parts.{i}.ws"))?);
            }
            if p.holo.is_some() {
                p.holo = Some((get(&format!("parts.{i}.holo_re"))?, get(&format!("parts.{i}.holo_im"))?));
            }
        }
        st.eng = kv_get("eng")?;
        st.fb = opt("fb".into());
        st.v51.load(&|k: &str| opt(k.to_string())).map_err(|x| format!("{}: {x}", path.display()))?;
        st.seen = seen;
        st.hist = hist;
        st.cap = cap;
        if let Some(c) = st.modc.as_mut() {
            let rows: Vec<Vec<f32>> = get("mod.rows")?.to_vec2().map_err(e)?;
            c.rows = rows.iter().map(|r| ModRow::from_slice(r)).collect();
            c.prev = get("mod.prev")?;
        }
        if st.shadow.is_some() {
            st.shadow = Some(Box::new(State::from_map(model, t, &format!("{pre}shadow."), path, false)?));
        }
        if st.loop2.is_some() {
            st.loop2 = Some(Box::new(State::from_map(model, t, &format!("{pre}loop2."), path, false)?));
        }
        Ok(st)
    }
}

/// S2: one saved state per conversation, in a folder (by default `conversations/` next to the brain).
pub struct ConvStore {
    pub dir: PathBuf,
}

/// What `ConvStore::load` found.
pub enum Loaded {
    /// no saved state for this conversation: start fresh
    Fresh,
    Resumed(State),
    /// a state saved by another model (or unreadable): start fresh; the reason
    Stale(String),
}

impl ConvStore {
    /// The folder `conversations` beside the brain file.
    pub fn next_to(brain: &Path) -> ConvStore {
        let dir = brain.parent().map(Path::to_path_buf).unwrap_or_default().join("conversations");
        ConvStore { dir }
    }

    /// The file of conversation `id`: the id itself when it is short and plain (letters, digits, '-' and '_'), else
    /// its FNV-1a hash.
    pub fn path(&self, id: &str) -> PathBuf {
        let plain = !id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        let name = if plain {
            id.to_string()
        } else {
            let mut h = Fnv::new();
            h.bytes(id.as_bytes());
            format!("h{:016x}", h.0)
        };
        self.dir.join(format!("state-{name}.safetensors"))
    }

    pub fn save(&self, id: &str, model: &Model, st: &State) -> std::result::Result<(), String> {
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("{}: {e}", self.dir.display()))?;
        st.save(model, &self.path(id))
    }

    pub fn load(&self, id: &str, model: &Model, dev: &Device) -> Loaded {
        let p = self.path(id);
        if !p.exists() {
            return Loaded::Fresh;
        }
        match State::load(model, &p, dev) {
            Ok(s) => Loaded::Resumed(s),
            Err(e) => Loaded::Stale(e),
        }
    }
}

// ------------------------------------------------------------------------------------------- generation

/// Greedy (or sampled, with `pick`) continuations of every prompt with carried state: the prompts are read in batched
/// passes (rows of the same length together, at most `batch_rows` per pass), then all rows step together, one token
/// per pass, and a row leaves the batch at EOS or when `stop(tokens)` says so. Returns each row's tokens (EOS not
/// included). Works for every model `state` accepts, block options included.
pub fn generate(
    model: &Model,
    dev: &Device,
    prompts: &[Vec<u32>],
    max_new: usize,
    batch_rows: usize,
    pick: &mut dyn FnMut(usize, &Tensor) -> Result<u32>,
    stop: &dyn Fn(&[u32]) -> bool,
) -> Result<Vec<Vec<u32>>> {
    use ava_data::bpe::EOS;
    if prompts.iter().any(Vec::is_empty) {
        candle_core::bail!("generate: an empty prompt");
    }
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    let mut order: Vec<usize> = (0..prompts.len()).collect();
    order.sort_by_key(|&r| (prompts[r].len(), r));
    let br = batch_rows.max(1);
    let mut i = 0;
    while i < order.len() {
        // a group: up to br rows; each run of equal-length prompts is read in one pass
        let group: Vec<usize> = order[i..(i + br).min(order.len())].to_vec();
        i += group.len();
        let mut states: Vec<State> = Vec::new();
        let mut lasts: Vec<Tensor> = Vec::new();
        let mut j = 0;
        while j < group.len() {
            let len = prompts[group[j]].len();
            let run: Vec<usize> = group[j..].iter().copied().take_while(|&r| prompts[r].len() == len).collect();
            j += run.len();
            let ids: Vec<u32> = run.iter().flat_map(|&r| prompts[r].iter().copied()).collect();
            let mut st = model.state(run.len())?;
            let logits = model.forward_with_state(&Tensor::from_vec(ids, (run.len(), len), dev)?, &mut st)?;
            let at: Vec<u32> = (0..run.len()).map(|k| (k * len + len - 1) as u32).collect();
            lasts.push(logits.index_select(&Tensor::from_vec(at, run.len(), dev)?, 0)?);
            states.push(st);
        }
        let mut st = State::cat(&states)?;
        let mut last = Tensor::cat(&lasts, 0)?; // [rows, V]
        let mut live: Vec<usize> = group.clone(); // prompt index of each state row
        for step in 0..max_new {
            let mut keep: Vec<usize> = Vec::new();
            let mut next: Vec<u32> = Vec::new();
            for (k, &r) in live.iter().enumerate() {
                let tok = pick(r, &last.get(k)?)?;
                if tok == EOS {
                    continue;
                }
                out[r].push(tok);
                if stop(&out[r]) {
                    continue;
                }
                keep.push(k);
                next.push(tok);
            }
            if keep.is_empty() || step + 1 == max_new {
                break;
            }
            if keep.len() != live.len() {
                st = st.select(&keep)?;
                live = keep.iter().map(|&k| live[k]).collect();
            }
            let nb = next.len();
            last = model.forward_with_state(&Tensor::from_vec(next, (nb, 1), dev)?, &mut st)?;
        }
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests;
