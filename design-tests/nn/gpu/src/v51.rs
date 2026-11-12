// TESTING / DESIGN ONLY — ava_gpu::v51: round 2 brain parts (runs/v51/prereg_v51.md, Part A and Part D), model D.
//
// Every option is off by default (V51Opts::default()); then nothing here runs and a model computes exactly what it
// did before (src/v51/golden.rs pins that bit for bit). They are block options (`Arch::v51`), so a model with any of
// them on runs `Model::forward_arch` (training, candle's autograd) and `forward_with_state` (talking). Each new weight
// starts so that the part changes nothing, except W, which rewires the MLP.
//
// P  --dendrite      two-part MLP neurons: each hidden unit's output (after kWTA and relu²) is scaled by its apical
//                    dendrite, fed by the block's fast-memory read c (rms-normed, no gain): a ⊙ (1 + tanh(c·A·B)),
//                    A [d, 32] random, B [32, hidden] at 0. The context never adds to the input (kWTA sees the same u).
// S  --stp           (v42's E9 part, src/memory.rs + ava_parts::stp: Tsodyks-Markram depression and facilitation per
//                    spiking neuron, exact recurrence, a fused scan; nothing new here, see the prereg's Amendment A1)
// R  --letters       a letter encoder: token v's bytes b_j (j < 16) with their place from the start and from the end:
//                    l_v = (1/sqrt(len)) Σ_j tanh(E[b_j] + P[j] + Q[len-1-j]) [64]; x += l_v · Wo (Wo [64, d] at 0).
//                    The bytes come from the BPE file (`set_letters_from_bpe` before the model is built). Per step the
//                    vectors of the whole vocabulary are made once and gathered (or, for fewer tokens than the
//                    vocabulary, only the rows needed): the cost does not grow with the sequence.
// DA --dopamine      her surprise at each word, from an early guess: block 0's stream through the final norm and the
//                    tied output head (no gradient), s_t = -ln q_{t-1}(x_t) (0 at the start of a stream), z = s/ln V;
//                    every later block's MLP output is scaled per channel by 1 + tanh(z·w + c) (w, c [d] at 0). One
//                    parallel pass in training; talking carries block 0's last normed state (the guess needs one extra
//                    head row per token), so a stream computes what the window computes.
// M  --mood          4 slow units driven by the same surprise: m_t = λ m_{t-1} + (1-λ)(a z_t + b), λ learned (time
//                    constants 16, 64, 256, 1024 tokens at init), a = 1, b = 0; carried across the whole stream (part of
//                    the State, saved and loaded). Every later block's spiking input is scaled by 1 + tanh(m·G) (G at 0).
//                    The step trace shows m as the "mood signal" (`StepTrace::v51`).
// W  --columns K     cortical columns in the MLP: the stream's d channels and the hidden units fall into K columns; W1
//                    and W2 keep only their K diagonal blocks (inside a column everything connects), plus one shared
//                    long-range path of rank 8 for each (h·A1·B1, a·A2·B2); kWTA picks hidden/(4K) winners per column
//                    (local inhibition). Uses W1's and W2's own diagonal blocks (the same init); the rest of them is unused.
// Pr --prune         gradual structured pruning of MLP hidden units: score = |W1 col|·|W2 row|·use (use = running mean
//                    activity), the weakest units are cut on a cubic schedule (Zhu & Gupta 2017) from 10% to 75% of
//                    training until 50% of each block's units are gone; kWTA keeps a quarter of the units left. A cut
//                    unit never comes back. `compact_pruned` then removes the cut units for real: a plain model with
//                    hidden 2d (--mlp-mult 2), the same logits, half the MLP work.
// Y  --rhythm        two learned rhythms (theta-like, period 16 tokens; gamma-like, period 3.2), phases moving with each
//                    word: phase φ = ω·a, a_t = (1 - r_t)(a_{t-1} + 1) the time since the last reset (r_t = sigmoid(rms(x_t)·w
//                    + b), w at 0, b = -10 at init: a full reset to phase 0 when r = 1, by what she reads); (cos φ, sin φ)
//                    of both gate every block's memory write strength and read, per head: 1 + tanh(f·G) (G at 0). A linear
//                    scan (chunks of 64 in closed form); a, per rhythm, is carried in the State.
// The signals between blocks (rhythm phases, surprise, mood) and the dendrites' context pass through `V51Ext::sig`
// for the forward in flight (set before the blocks run, cleared after); so --feedback, --modulate and the activation
// checkpoint (AVA_CKPT) are not combined with them.

use crate::model::{head_norm, kwta_mask, log_sigmoid, norm, sigmoid, Config, Kind, Mlp, Model, INIT_STD};
use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{Init, VarBuilder};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub const DEND_RANK: usize = 32;
pub const MOOD_UNITS: usize = 4;
pub const LETTER_DIM: usize = 64;
pub const LETTER_MAX: usize = 16;
pub const COL_RANK: usize = 8;
pub const PRUNE_SHARE: f32 = 0.5;
pub const PRUNE_FROM: f64 = 0.10;
pub const PRUNE_TO: f64 = 0.75;
pub const PRUNE_EVERY: usize = 50;
pub const USE_DECAY: f64 = 0.99;
/// Y: the rhythms' periods at init (tokens): theta-like and gamma-like (5 gamma cycles in one theta cycle)
pub const RHYTHM_PERIODS: [f64; 2] = [16.0, 3.2];
pub const RESET_BIAS: f64 = -10.0;
/// M and Y: the closed-form scans run in chunks of this many tokens
pub const SCAN_CHUNK: usize = 64;

/// The v51 options (see the top of the file). Default: all off.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct V51Opts {
    pub dendrite: bool,
    pub letters: bool,
    pub dopamine: bool,
    /// M: the number of slow units (0 = off)
    pub mood: usize,
    /// W: the number of columns (0 = off)
    pub columns: usize,
    /// Pr: the share of hidden units cut by the end (0 = off)
    pub prune: f32,
    pub rhythm: bool,
}

impl V51Opts {
    pub fn any(&self) -> bool {
        *self != V51Opts::default()
    }
    /// The options with signals between the blocks (set by the forward, read by the blocks).
    pub fn signals(&self) -> bool {
        self.dopamine || self.mood > 0 || self.rhythm || self.dendrite
    }
    pub fn surprise(&self) -> bool {
        self.dopamine || self.mood > 0
    }
    /// The options that change the MLP.
    pub fn mlp(&self) -> bool {
        self.dendrite || self.dopamine || self.columns > 0 || self.prune > 0.0
    }

    /// One v51 flag at `args[i]`: how many arguments it used, or None when it is not one.
    pub fn take_flag(&mut self, args: &[String], i: usize) -> std::result::Result<Option<usize>, String> {
        let key = args[i].as_str();
        let val = || args.get(i + 1).map(|s| s.as_str()).ok_or_else(|| format!("{key} needs a value"));
        match key {
            "--dendrite" => self.dendrite = true,
            "--letters" => self.letters = true,
            "--dopamine" => self.dopamine = true,
            "--mood" => self.mood = MOOD_UNITS,
            "--rhythm" => self.rhythm = true,
            "--prune" => self.prune = PRUNE_SHARE,
            "--mood-units" => {
                self.mood = val()?.parse().map_err(|_| format!("bad integer for {key}"))?;
                return Ok(Some(2));
            }
            "--columns" => {
                self.columns = val()?.parse().map_err(|_| format!("bad integer for {key}"))?;
                return Ok(Some(2));
            }
            "--prune-to" => {
                self.prune = val()?.parse().map_err(|_| format!("bad number for {key}"))?;
                return Ok(Some(2));
            }
            _ => return Ok(None),
        }
        Ok(Some(1))
    }

    pub fn signature(&self) -> String {
        let mut s = String::new();
        if self.dendrite {
            s += " dendrite=1";
        }
        if self.letters {
            s += " letters=1";
        }
        if self.dopamine {
            s += " dopamine=1";
        }
        if self.mood > 0 {
            s += &format!(" mood={}", self.mood);
        }
        if self.columns > 0 {
            s += &format!(" columns={}", self.columns);
        }
        if self.prune > 0.0 {
            s += &format!(" prune={}", self.prune);
        }
        if self.rhythm {
            s += " rhythm=1";
        }
        s
    }

    pub fn flags(&self) -> Vec<String> {
        let mut f = Vec::new();
        if self.dendrite {
            f.push("--dendrite".into());
        }
        if self.letters {
            f.push("--letters".into());
        }
        if self.dopamine {
            f.push("--dopamine".into());
        }
        match self.mood {
            0 => {}
            MOOD_UNITS => f.push("--mood".into()),
            k => f.push(format!("--mood-units {k}")),
        }
        if self.columns > 0 {
            f.push(format!("--columns {}", self.columns));
        }
        if self.prune == PRUNE_SHARE {
            f.push("--prune".into());
        } else if self.prune > 0.0 {
            f.push(format!("--prune-to {}", self.prune));
        }
        if self.rhythm {
            f.push("--rhythm".into());
        }
        f
    }

    /// The options checked against the rest of the config (c.arch holds them).
    pub fn validate(&self, c: &Config) -> std::result::Result<(), String> {
        if !self.any() {
            return Ok(());
        }
        let ar = c.arch;
        if c.kind != Kind::D {
            return Err(format!("the v51 options{} are for model D", self.signature()));
        }
        if self.signals() && (ar.feedback || ar.mem.modulate) {
            return Err("v51: --dopamine, --mood, --rhythm and --dendrite are not combined with --feedback or --modulate".into());
        }
        if self.mlp() && (ar.mem.halting > 0 || ar.mem.topo > 0.0) {
            return Err("v51: the MLP options are not combined with --halting or --topo".into());
        }
        if self.any() && ar.feedback {
            return Err("v51: not combined with --feedback (its chunks)".into());
        }
        if self.surprise() && c.layers < 2 {
            return Err("v51: --dopamine and --mood act on the blocks after block 0: at least 2 blocks".into());
        }
        let h = c.hidden();
        if self.columns > 0 {
            let k = self.columns;
            if k < 2 || c.d % k != 0 || h % k != 0 || c.kwta_k() % k != 0 {
                return Err(format!("--columns {k}: at least 2, dividing d {}, the hidden width {h} and the kWTA winners {}", c.d, c.kwta_k()));
            }
        }
        if self.prune != 0.0 {
            if !(self.prune > 0.0 && self.prune < 1.0) {
                return Err(format!("--prune-to {}: a share in (0, 1)", self.prune));
            }
            if self.columns > 0 {
                return Err("v51: --prune and --columns are not combined".into());
            }
            if ar.dale > 0.0 {
                // merged with v50: Pr's scores and compact_pruned read the stored w2, which --dale replaces by its v (w2s)
                return Err("v51: --prune and v50's --dale are not combined (Pr reads the stored W2)".into());
            }
            let left = h - (self.prune as f64 * h as f64).round() as usize;
            if left / ar.kwta_div == 0 {
                return Err("--prune: too few units would be left for kWTA".into());
            }
        }
        if self.mood > 16 {
            return Err("--mood-units: at most 16".into());
        }
        if self.letters && c.vocab < 256 {
            return Err("--letters: the vocabulary has no bytes".into());
        }
        Ok(())
    }

    /// The number of weights `build` makes.
    pub fn extra_weights(&self, c: &Config) -> usize {
        if !self.any() {
            return 0;
        }
        let (d, l, h) = (c.d, c.layers, c.hidden());
        let nh = c.heads();
        let mut n = 0;
        if self.dendrite {
            n += l * (d * DEND_RANK + DEND_RANK * h);
        }
        if self.dopamine {
            n += (l - 1) * 2 * d;
        }
        if self.mood > 0 {
            n += 3 * self.mood + (l - 1) * self.mood * d;
        }
        if self.columns > 0 {
            n += l * COL_RANK * (2 * d + 2 * h);
        }
        if self.prune > 0.0 {
            n += l * 2 * h;
        }
        if self.rhythm {
            n += 2 + 2 * d + 2 + l * 2 * 4 * nh;
        }
        if self.letters {
            n += 256 * LETTER_DIM + 2 * LETTER_MAX * LETTER_DIM + LETTER_DIM * d;
        }
        n
    }
}

/// Why the fused one-token CPU step does not take these options (None: no v51 option is on). The step strips the options
/// it runs first (R, M, Pr; f4ext: P, DA, Y), so what is left is W.
pub fn fused_why(o: &V51Opts) -> Option<&'static str> {
    if o.any() {
        Some("v51 --columns is not in the fused step yet: it talks through the State step (prereg Amendment 0: fused steps only for kept parts)")
    } else {
        None
    }
}

// ------------------------------------------------------------------------------------------- the parts' weights

pub(crate) struct Letters {
    pub byte: Tensor,
    pub pos: Tensor,
    pub end: Tensor,
    pub wo: Tensor,
    /// per vocabulary row: the byte ids [V, L] (u32), the place-from-the-end ids [V, L] (u32), the mask [V, L] and
    /// 1/sqrt(len) [V, 1] (0 for a token without bytes)
    pub ids: Tensor,
    pub eidx: Tensor,
    pub mask: Tensor,
    pub inv: Tensor,
}

pub(crate) struct MoodPart {
    /// logit λ, the drive's gain and offset, each [k]
    pub lam: Tensor,
    pub win: Tensor,
    pub bin: Tensor,
}

pub(crate) struct RhythmPart {
    /// ln ω [2], the reset projection [d, 2] and its bias [2]
    pub lw: Tensor,
    pub wr: Tensor,
    pub br: Tensor,
}

#[derive(Default)]
pub(crate) struct V51Block {
    /// P: A [d, R], B [R, hidden]
    pub dend: Option<(Tensor, Tensor)>,
    /// DA: w, c [d] (blocks after the first)
    pub da: Option<(Tensor, Tensor)>,
    /// M: G [k, d] (blocks after the first)
    pub mood_g: Option<Tensor>,
    /// W: A1 [d, r], B1 [r, hidden], A2 [hidden, r], B2 [r, d]
    pub cols: Option<[Tensor; 4]>,
    /// Pr: alive [hidden] (0/1) and use [hidden], variables without gradient
    pub prune: Option<(Tensor, Tensor)>,
    /// Y: the write and read gates' weights, each [4, heads]
    pub rh: Option<(Tensor, Tensor)>,
}

/// The signals of the forward in flight.
#[derive(Default)]
pub(crate) struct Sig {
    pub block: usize,
    /// Y: [n, 4] the phases (cos, sin of theta, then of gamma)
    pub rh: Option<Tensor>,
    /// DA: [n, 1] the surprise / ln V
    pub z: Option<Tensor>,
    /// M: [n, k]
    pub mood: Option<Tensor>,
    /// P: this block's memory read [n, d]
    pub ctx: Option<Tensor>,
}

#[derive(Default)]
pub(crate) struct PruneCtl {
    pub step: Cell<usize>,
    pub total: Cell<usize>,
    pub alive_n: RefCell<Vec<Option<usize>>>,
    pub stats: RefCell<Vec<Option<Tensor>>>,
}

/// A model's v51 parts (`ModelExt::v51`; empty when every option is off).
#[derive(Default)]
pub(crate) struct V51Ext {
    pub on: bool,
    pub blocks: Vec<V51Block>,
    pub letters: Option<Letters>,
    pub mood: Option<MoodPart>,
    pub rhythm: Option<RhythmPart>,
    pub sig: RefCell<Sig>,
    pub prune: PruneCtl,
}

fn ext(m: &Model) -> Option<&V51Ext> {
    m.ext.as_ref().map(|e| &e.v51).filter(|v| v.on)
}

// The letter tables, per vocabulary size (set before a --letters model is built).
static LETTERS: Mutex<Option<HashMap<usize, Arc<Vec<Vec<u8>>>>>> = Mutex::new(None);

/// R: the bytes of every token of a vocabulary (row v = token v's bytes; empty rows for tokens without bytes).
pub fn set_letters(table: Vec<Vec<u8>>) {
    let mut g = LETTERS.lock().unwrap_or_else(|e| e.into_inner());
    g.get_or_insert_with(HashMap::new).insert(table.len(), Arc::new(table));
}

/// R: the letter table of `vocab` tokens from a BPE file (tokens past the file's own, the glance's length marks, and
/// EOS/PAD have no bytes).
pub fn set_letters_from_bpe(path: &std::path::Path, vocab: usize) -> std::result::Result<(), String> {
    let bpe = ava_data::bpe::Bpe::load(path).map_err(|e| format!("--letters: bpe {}: {e}", path.display()))?;
    let n = bpe.vocab_size();
    set_letters((0..vocab).map(|v| if v < n { bpe.token_bytes(v as u32).to_vec() } else { Vec::new() }).collect());
    Ok(())
}

fn letter_table(vocab: usize) -> Option<Arc<Vec<Vec<u8>>>> {
    LETTERS.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&vocab).cloned())
}

/// Make the v51 parts in `vb` (called last in arch::build_ext, so every earlier weight keeps its init).
pub(crate) fn build(cfg: &Config, vb: &VarBuilder) -> Result<V51Ext> {
    let o = cfg.arch.v51;
    if !o.any() {
        return Ok(V51Ext::default());
    }
    let (d, l, h, dev) = (cfg.d, cfg.layers, cfg.hidden(), vb.device().clone());
    let nh = cfg.heads();
    let randn = |std: f64| Init::Randn { mean: 0.0, stdev: std };
    let zero = Init::Const(0.0);
    let mut blocks = Vec::with_capacity(l);
    for i in 0..l {
        let vp = vb.pp(format!("b{i}.v51"));
        let mut b = V51Block::default();
        if o.dendrite {
            b.dend = Some((vp.get_with_hints((d, DEND_RANK), "dend_a", randn(1.0 / (d as f64).sqrt()))?, vp.get_with_hints((DEND_RANK, h), "dend_b", zero)?));
        }
        if o.dopamine && i > 0 {
            b.da = Some((vp.get_with_hints(d, "da_w", zero)?, vp.get_with_hints(d, "da_c", zero)?));
        }
        if o.mood > 0 && i > 0 {
            b.mood_g = Some(vp.get_with_hints((o.mood, d), "mood_g", zero)?);
        }
        if o.columns > 0 {
            let r = COL_RANK;
            b.cols = Some([
                vp.get_with_hints((d, r), "col_a1", randn(1.0 / (d as f64).sqrt()))?,
                vp.get_with_hints((r, h), "col_b1", randn(INIT_STD))?,
                vp.get_with_hints((h, r), "col_a2", randn(1.0 / (h as f64).sqrt()))?,
                vp.get_with_hints((r, d), "col_b2", randn(cfg.resid_std()))?,
            ]);
        }
        if o.prune > 0.0 {
            b.prune = Some((vp.get_with_hints(h, "alive", Init::Const(1.0))?, vp.get_with_hints(h, "use", zero)?));
        }
        if o.rhythm {
            b.rh = Some((vp.get_with_hints((4, nh), "rh_w", zero)?, vp.get_with_hints((4, nh), "rh_r", zero)?));
        }
        blocks.push(b);
    }
    let mood = if o.mood > 0 {
        let k = o.mood;
        // time constants 16·4^j tokens: λ = 1 - 1/τ
        let lam: Vec<f32> = (0..k).map(|j| crate::model::logit(1.0 - 1.0 / (16.0 * 4f64.powi(j as i32))) as f32).collect();
        let vm = vb.pp("v51.mood");
        let lam_t = vm.get_with_hints(k, "lam", zero)?;
        let fresh = lam_t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.iter().all(|&x| x == 0.0);
        if fresh {
            Var::from_tensor(&lam_t)?.set(&Tensor::from_vec(lam, k, &dev)?.to_dtype(lam_t.dtype())?)?; // v51 fast1: the var's dtype (F-1's f64 reference)
        }
        Some(MoodPart { lam: lam_t, win: vm.get_with_hints(k, "win", Init::Const(1.0))?, bin: vm.get_with_hints(k, "bin", zero)? })
    } else {
        None
    };
    let rhythm = if o.rhythm {
        let vr = vb.pp("v51.rhythm");
        let lw = vr.get_with_hints(2, "lw", zero)?;
        let fresh = lw.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.iter().all(|&x| x == 0.0);
        if fresh {
            let w: Vec<f32> = RHYTHM_PERIODS.iter().map(|p| (2.0 * std::f64::consts::PI / p).ln() as f32).collect();
            Var::from_tensor(&lw)?.set(&Tensor::from_vec(w, 2, &dev)?.to_dtype(lw.dtype())?)?;
        }
        Some(RhythmPart { lw, wr: vr.get_with_hints((d, 2), "wr", zero)?, br: vr.get_with_hints(2, "br", Init::Const(RESET_BIAS))? })
    } else {
        None
    };
    let letters = if o.letters {
        if letter_table(cfg.vocab).is_none() {
            // the release BPE (release/ava/bpe.txt is the same file) when no tool set a table
            set_letters_from_bpe(std::path::Path::new(crate::DEFAULT_BPE), cfg.vocab).map_err(candle_core::Error::Msg)?;
        }
        let table = letter_table(cfg.vocab).ok_or_else(|| candle_core::Error::Msg(format!("--letters: no letter table for vocab {}", cfg.vocab)))?;
        let (v, lm) = (cfg.vocab, LETTER_MAX);
        let (mut ids, mut eidx, mut mask, mut inv) = (vec![0u32; v * lm], vec![0u32; v * lm], vec![0f32; v * lm], vec![0f32; v]);
        for (t, bytes) in table.iter().enumerate() {
            let len = bytes.len();
            let used = len.min(lm);
            for j in 0..used {
                ids[t * lm + j] = bytes[j] as u32;
                eidx[t * lm + j] = (len - 1 - j).min(lm - 1) as u32;
                mask[t * lm + j] = 1.0;
            }
            if used > 0 {
                inv[t] = 1.0 / (used as f32).sqrt();
            }
        }
        let vl = vb.pp("v51.letters");
        let s = 1.0 / 3f64.sqrt();
        Some(Letters {
            byte: vl.get_with_hints((256, LETTER_DIM), "byte", randn(s))?,
            pos: vl.get_with_hints((lm, LETTER_DIM), "pos", randn(s))?,
            end: vl.get_with_hints((lm, LETTER_DIM), "end", randn(s))?,
            wo: vl.get_with_hints((LETTER_DIM, d), "wo", zero)?,
            ids: Tensor::from_vec(ids, (v, lm), &dev)?,
            eidx: Tensor::from_vec(eidx, (v, lm), &dev)?,
            mask: Tensor::from_vec(mask, (v, lm), &dev)?,
            inv: Tensor::from_vec(inv, (v, 1), &dev)?,
        })
    } else {
        None
    };
    let prune = PruneCtl { alive_n: RefCell::new(vec![None; l]), stats: RefCell::new(vec![None; l]), ..Default::default() };
    Ok(V51Ext { on: true, blocks, letters, mood, rhythm, sig: RefCell::new(Sig::default()), prune })
}

// ------------------------------------------------------------------------------------------- the parts' maths
// (plain functions of tensors, any float dtype, so the tests check them in f64 against finite differences)

/// R: the letter vectors [m, LETTER_DIM] of m tokens given their byte ids, end places (u32 [m, L]), mask [m, L] and
/// 1/sqrt(len) [m, 1].
pub fn letter_core(byte: &Tensor, pos: &Tensor, end: &Tensor, ids: &Tensor, eidx: &Tensor, mask: &Tensor, inv: &Tensor) -> Result<Tensor> {
    let (m, l) = ids.dims2()?;
    let dl = byte.dim(1)?;
    let e = byte.embedding(&ids.flatten_all()?)?.reshape((m, l, dl))?;
    let q = end.embedding(&eidx.flatten_all()?)?.reshape((m, l, dl))?;
    let s = e.add(&q)?.broadcast_add(&pos.unsqueeze(0)?)?.tanh()?;
    let mk = mask.to_dtype(byte.dtype())?.unsqueeze(2)?;
    s.broadcast_mul(&mk)?.sum(1)?.broadcast_mul(&inv.to_dtype(byte.dtype())?)
}

/// [c, c] (t, j): max(t - j, 0) and the lower-triangular mask (j <= t).
fn lag(c: usize, dt: DType, dev: &Device) -> Result<(Tensor, Tensor)> {
    let mut l = vec![0f32; c * c];
    let mut m = vec![0f32; c * c];
    for t in 0..c {
        for j in 0..=t {
            l[t * c + j] = (t - j) as f32;
            m[t * c + j] = 1.0;
        }
    }
    Ok((Tensor::from_vec(l, (c, c), dev)?.to_dtype(dt)?, Tensor::from_vec(m, (c, c), dev)?.to_dtype(dt)?))
}

/// M: the leaky scan m_t = λ m_{t-1} + (1 - λ) u_t over u [b, t, k] from m0 [b, k], λ = sigmoid(lam) [k]; returns
/// m [b, t, k]. Each chunk in closed form: m_t = Σ_{j<=t} λ^{t-j} (1-λ) u_j + λ^{t+1} m_init.
pub fn mood_scan(u: &Tensor, lam: &Tensor, m0: &Tensor) -> Result<Tensor> {
    let (_, t, k) = u.dims3()?;
    let (dt, dev) = (u.dtype(), u.device());
    let ll = log_sigmoid(lam)?; // ln λ [k]
    let oml = sigmoid(&lam.neg()?)?; // 1 - λ
    let mut init = m0.clone();
    let mut outs = Vec::new();
    let mut c0 = 0;
    while c0 < t {
        let c = SCAN_CHUNK.min(t - c0);
        let (lg, tri) = lag(c, dt, dev)?;
        // P [k, c, c] = λ^(t-j) on and below the diagonal
        let p = lg.unsqueeze(0)?.broadcast_mul(&ll.reshape((k, 1, 1))?)?.exp()?.broadcast_mul(&tri.unsqueeze(0)?)?;
        let uc = u.narrow(1, c0, c)?.broadcast_mul(&oml.reshape((1, 1, k))?)?; // [b, c, k]
        let uk = uc.permute((2, 1, 0))?.contiguous()?; // [k, c, b]
        let mc = p.matmul(&uk)?.permute((2, 1, 0))?; // [b, c, k]
        // λ^(t+1) m_init
        let steps: Vec<f32> = (1..=c).map(|s| s as f32).collect();
        let pw = Tensor::from_vec(steps, (c, 1), dev)?.to_dtype(dt)?.broadcast_mul(&ll.reshape((1, k))?)?.exp()?; // [c, k]
        let mc = mc.add(&init.unsqueeze(1)?.broadcast_mul(&pw.unsqueeze(0)?)?)?.contiguous()?;
        init = mc.narrow(1, c - 1, 1)?.squeeze(1)?;
        outs.push(mc);
        c0 += c;
    }
    if outs.len() == 1 {
        return Ok(outs.pop().expect("one chunk"));
    }
    Tensor::cat(&outs, 1)
}

/// Y: each rhythm's time since its last reset. pre [b, t, o] the resets' logits, a0 [b, o] the time carried in; per
/// rhythm a_t = (1 - r_t)(a_{t-1} + 1), r = sigmoid(pre) (r = 1: a full reset to phase 0); returns a [b, t, o].
/// Each chunk in closed form: a_t = Σ_{j<=t} Π_{j<=l<=t}(1 - r_l) + Π_{l<=t}(1 - r_l) a_init, the products as
/// exp(L_t - L_{j-1}) with L_t = Σ_{l<=t} ln(1 - r_l) (a triangular matmul).
pub fn rhythm_scan(pre: &Tensor, a0: &Tensor) -> Result<Tensor> {
    let (_, t, _) = pre.dims3()?;
    let (dt, dev) = (pre.dtype(), pre.device());
    let la = log_sigmoid(&pre.neg()?)?.permute((0, 2, 1))?.contiguous()?; // ln(1 - r) [b, o, t]
    let mut init = a0.clone(); // [b, o]
    let mut outs = Vec::new();
    let mut c0 = 0;
    while c0 < t {
        let c = SCAN_CHUNK.min(t - c0);
        let (_, tri) = lag(c, dt, dev)?;
        let lc = la.narrow(2, c0, c)?.contiguous()?; // [b, o, c]
        let big_l = lc.broadcast_matmul(&tri.t()?.contiguous()?)?; // L_t
        let before = big_l.sub(&lc)?; // L_{j-1}
        let diff = big_l.unsqueeze(3)?.broadcast_sub(&before.unsqueeze(2)?)?; // [b, o, c(t), c(j)]
        let w = diff.minimum(0.0)?.exp()?.broadcast_mul(&tri.reshape((1, 1, c, c))?)?;
        let a = w.sum(3)?.add(&big_l.exp()?.broadcast_mul(&init.unsqueeze(2)?)?)?; // [b, o, c]
        init = a.narrow(2, c - 1, 1)?.squeeze(2)?;
        outs.push(a.permute((0, 2, 1))?.contiguous()?); // [b, c, o]
        c0 += c;
    }
    if outs.len() == 1 {
        return Ok(outs.pop().expect("one chunk"));
    }
    Tensor::cat(&outs, 1)
}

/// Y: the rhythms' phases ω·a as (cos, sin) per rhythm, [b·t, 2·o], from a [b, t, o] and lw = ln ω [o].
pub fn rhythm_feats(a: &Tensor, lw: &Tensor) -> Result<Tensor> {
    let (b, t, o) = a.dims3()?;
    let ph = a.broadcast_mul(&lw.exp()?.reshape((1, 1, o))?)?;
    Tensor::stack(&[ph.cos()?, ph.sin()?], 3)?.reshape((b * t, o * 2))
}

/// The config's Debug text, as it was before v51 while every v51 option is off (State::fingerprint hashes it, so a
/// conversation saved before v51 still loads).
pub(crate) fn cfg_debug(c: &Config) -> String {
    let mut s = format!("{c:?}");
    // v51b (merge, runs/v51/merge_b.md): its field is left out too while its options are off, whatever v51's are (so
    // every fingerprint made before the merge, v51's own included, stays what it was)
    if !crate::v51b::any(&c.arch) {
        s = s.replacen(&format!(", v51b: {:?}", crate::v51b::Opts::default()), "", 1);
    }
    // v52 --lif-reset-grad (runs/v52/spike/report2.md): left out while off, so every earlier fingerprint stays
    if !c.arch.lif_reset_grad {
        s = s.replacen(", lif_reset_grad: false", "", 1);
    }
    if c.arch.v51.any() {
        return s;
    }
    s.replacen(&format!(", v51: {:?}", V51Opts::default()), "", 1)
}

/// 1 + tanh(x) (the gains of P, DA, M and Y; exactly 1 where x = 0).
pub fn gain(x: &Tensor) -> Result<Tensor> {
    x.tanh()? + 1.0
}

/// W: h [n, d] through a column-wired matrix: w's K diagonal blocks plus the long-range path h·a·c (w [d_in, d_out]).
pub fn col_mm(h: &Tensor, w: &Tensor, a: &Tensor, c: &Tensor, k: usize) -> Result<Tensor> {
    let n = h.dim(0)?;
    let (di, dout) = w.dims2()?;
    let (bi, bo) = (di / k, dout / k);
    let hb = h.reshape((n, k, bi))?.transpose(0, 1)?.contiguous()?; // [K, n, bi]
    let blocks: Vec<Tensor> = (0..k).map(|j| w.narrow(0, j * bi, bi)?.narrow(1, j * bo, bo)).collect::<Result<_>>()?;
    let wb = Tensor::stack(&blocks, 0)?; // [K, bi, bo]
    let loc = hb.matmul(&wb)?.transpose(0, 1)?.contiguous()?.reshape((n, dout))?;
    loc.add(&h.matmul(a)?.matmul(c)?)
}

// ------------------------------------------------------------------------------------------- the hooks

/// R: x [n, d] plus the letter vectors of `ids` (x itself without --letters).
pub(crate) fn embed(m: &Model, ids: &Tensor, x: Tensor) -> Result<Tensor> {
    let Some(lt) = ext(m).and_then(|v| v.letters.as_ref()) else { return Ok(x) };
    let flat = ids.flatten_all()?;
    let n = flat.dim(0)?;
    let add = if n < m.cfg.vocab {
        // only the rows of these tokens (talking, short reads)
        let sel = |t: &Tensor| t.index_select(&flat, 0);
        letter_core(&lt.byte, &lt.pos, &lt.end, &sel(&lt.ids)?, &sel(&lt.eidx)?, &sel(&lt.mask)?, &sel(&lt.inv)?)?.matmul(&lt.wo)?
    } else {
        // the whole vocabulary once, then gathered (training)
        letter_core(&lt.byte, &lt.pos, &lt.end, &lt.ids, &lt.eidx, &lt.mask, &lt.inv)?.matmul(&lt.wo)?.embedding(&flat)?
    };
    x.add(&add)
}

/// Y: before the blocks: the rhythms' phases for the input stream x [b·t, d] (carried by a State's `carry`).
pub(crate) fn begin(m: &Model, x: &Tensor, b: usize, t: usize, carry: Option<&mut V51State>) -> Result<()> {
    let Some(v) = ext(m) else { return Ok(()) };
    *v.sig.borrow_mut() = Sig::default();
    let Some(rp) = v.rhythm.as_ref() else { return Ok(()) };
    let pre = head_norm(x)?.matmul(&rp.wr)?.broadcast_add(&rp.br)?.reshape((b, t, 2))?;
    let a0 = match carry.as_ref().and_then(|c| c.rh.as_ref()) {
        Some(a) => a.clone(),
        None => Tensor::zeros((b, 2), x.dtype(), x.device())?,
    };
    let a = rhythm_scan(&pre, &a0)?;
    if let Some(c) = carry {
        c.rh = Some(a.narrow(1, t - 1, 1)?.squeeze(1)?.contiguous()?.detach());
    }
    v.sig.borrow_mut().rh = Some(rhythm_feats(&a, &rp.lw)?);
    Ok(())
}

/// DA and M: after block 0, x [b·t, d] the stream block 1 reads, `ids` [b, t]: the surprise and the mood.
pub(crate) fn after0(m: &Model, x: &Tensor, ids: &Tensor, b: usize, t: usize, mut carry: Option<&mut V51State>) -> Result<()> {
    let Some(v) = ext(m) else { return Ok(()) };
    let o = m.cfg.arch.v51;
    if !o.surprise() {
        return Ok(());
    }
    let (d, dev) = (m.cfg.d, x.device());
    let xf = norm(&x.detach(), &m.final_norm)?; // [n, d] (the surprise is detached below)
    let tok = &m.tok; // itself, so talking uses its int8 copy
    let first = match carry.as_ref().and_then(|c| c.prev.as_ref().zip(c.pvalid.as_ref())) {
        Some((prev, valid)) => {
            let l = crate::int8::head(prev, &tok)?;
            let tg = ids.narrow(1, 0, 1)?.contiguous()?.flatten_all()?;
            crate::lif::cross_entropy_rows(&l, &tg)?.mul(valid)?
        }
        None => Tensor::zeros(b, x.dtype(), dev)?,
    };
    let mut parts = vec![first.reshape((b, 1))?];
    if t > 1 {
        let rows = xf.reshape((b, t, d))?.narrow(1, 0, t - 1)?.contiguous()?.reshape((b * (t - 1), d))?;
        let l = crate::int8::head(&rows, &tok)?;
        let tg = ids.narrow(1, 1, t - 1)?.contiguous()?.flatten_all()?;
        parts.push(crate::lif::cross_entropy_rows(&l, &tg)?.reshape((b, t - 1))?);
    }
    let s = Tensor::cat(&parts, 1)?.detach(); // [b, t] nats
    let z = s.affine(1.0 / (m.cfg.vocab as f64).ln(), 0.0)?;
    let mood = match v.mood.as_ref() {
        Some(mp) => {
            let k = o.mood;
            let u = z.unsqueeze(2)?.broadcast_mul(&mp.win.reshape((1, 1, k))?)?.broadcast_add(&mp.bin.reshape((1, 1, k))?)?;
            let m0 = match carry.as_ref().and_then(|c| c.mood.as_ref()) {
                Some(x) => x.clone(),
                None => Tensor::zeros((b, k), x.dtype(), dev)?,
            };
            let mm = mood_scan(&u, &mp.lam, &m0)?; // [b, t, k]
            if let Some(c) = carry.as_mut() {
                c.mood = Some(mm.narrow(1, t - 1, 1)?.squeeze(1)?.contiguous()?.detach());
            }
            Some(mm.reshape((b * t, k))?)
        }
        None => None,
    };
    if let Some(c) = carry {
        c.prev = Some(xf.reshape((b, t, d))?.narrow(1, t - 1, 1)?.squeeze(1)?.contiguous()?);
        c.pvalid = Some(Tensor::ones(b, DType::F32, dev)?);
        c.last_s = Some(s.narrow(1, t - 1, 1)?.squeeze(1)?.contiguous()?);
    }
    let mut sg = v.sig.borrow_mut();
    sg.z = Some(z.reshape((b * t, 1))?);
    sg.mood = mood;
    Ok(())
}

/// After the blocks: the signals go.
pub(crate) fn end(m: &Model) {
    if let Some(v) = ext(m) {
        *v.sig.borrow_mut() = Sig::default();
    }
}

/// At the top of block i.
pub(crate) fn at_block(m: &Model, i: usize) {
    if let Some(v) = ext(m) {
        let mut s = v.sig.borrow_mut();
        s.block = i;
        s.ctx = None;
    }
}

/// M: the spiking input current cur [b, t, d] of block i scaled by the mood (cur itself elsewhere).
pub(crate) fn lif_gain(m: &Model, i: usize, cur: Tensor) -> Result<Tensor> {
    let Some(v) = ext(m) else { return Ok(cur) };
    let (Some(g), Some(md)) = (v.blocks[i].mood_g.as_ref(), v.sig.borrow().mood.clone()) else { return Ok(cur) };
    cur.mul(&gain(&md.matmul(g)?)?.reshape(cur.dims())?)
}

/// Y: the memory's write strength beta [b, nh, t, g] of the current block, gated by the rhythms.
pub(crate) fn write_gate(m: &Model, beta: Tensor) -> Result<Tensor> {
    rh_gate(m, beta, true)
}

/// Y: the memory's read y [b, nh, t, hs] (after the head norm), gated by the rhythms.
pub(crate) fn read_gate(m: &Model, y: Tensor) -> Result<Tensor> {
    rh_gate(m, y, false)
}

fn rh_gate(m: &Model, x: Tensor, write: bool) -> Result<Tensor> {
    let Some(v) = ext(m) else { return Ok(x) };
    let sg = v.sig.borrow();
    let (Some(f), Some((gw, gr))) = (sg.rh.as_ref(), v.blocks.get(sg.block).and_then(|b| b.rh.as_ref())) else { return Ok(x) };
    let (b, nh, t, _) = x.dims4()?;
    let g = gain(&f.matmul(if write { gw } else { gr })?)?.reshape((b, t, nh))?.permute((0, 2, 1))?.unsqueeze(3)?;
    x.broadcast_mul(&g)
}

/// P: block i's memory read y [n, d], the context of its MLP's dendrites.
pub(crate) fn put_ctx(m: &Model, y: &Tensor) {
    if let Some(v) = ext(m).filter(|_| m.cfg.arch.v51.dendrite) {
        v.sig.borrow_mut().ctx = Some(y.clone());
    }
}

/// Pr: block i's live units (its cut mask's count, cached until `v51_schedule` or a cut; the hidden width without
/// --prune).
pub(crate) fn alive_count(m: &Model, v: &V51Ext, i: usize) -> Result<usize> {
    let Some((alive, _)) = v.blocks[i].prune.as_ref() else { return Ok(m.cfg.hidden()) };
    let mut c = v.prune.alive_n.borrow_mut();
    Ok(match c[i] {
        Some(n) => n,
        None => {
            let n = alive.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?.round() as usize;
            c[i] = Some(n);
            n
        }
    })
}

/// The kWTA winners kept in block i (k, less as Pr cuts units).
pub(crate) fn kwta_k(m: &Model, v: &V51Ext, i: usize) -> Result<usize> {
    if v.blocks[i].prune.is_none() {
        return Ok(m.cfg.kwta_k());
    }
    Ok((alive_count(m, v, i)? / m.cfg.arch.kwta_div).max(1))
}

/// The MLP of block i with the v51 options (None: none of them is on): from the normed input h [n, d], the units
/// a [n, hidden] (what reaches W2), the kWTA winners (only when `tr`) and the output [n, d]. `collect`: keep the
/// units' mean activity for Pr (training forwards).
pub(crate) fn mlp(m: &Model, i: usize, h: &Tensor, l: &Mlp, tr: bool, collect: bool) -> Result<Option<(Tensor, Option<Tensor>, Tensor)>> {
    let Some(v) = ext(m).filter(|_| m.cfg.arch.v51.mlp()) else { return Ok(None) };
    let o = m.cfg.arch.v51;
    let bx = &v.blocks[i];
    let n = h.dim(0)?;
    let hid = m.cfg.hidden();
    let mut k = kwta_k(m, v, i)?;
    let mut u = match &bx.cols {
        Some([a1, b1, _, _]) => col_mm(h, &l.w1, a1, b1, o.columns)?,
        None => crate::int8::mm(h, &l.w1)?,
    };
    if let Some((alive, _)) = &bx.prune {
        // a cut unit never wins: -1e9 before the competition (0 added to a live one)
        u = u.broadcast_add(&alive.detach().affine(1e9, -1e9)?)?;
    }
    let kc = o.columns.max(1);
    if kc > 1 {
        // local inhibition: the winners per column
        u = u.reshape((n * kc, hid / kc))?;
        k /= kc;
    }
    let fast = crate::fast_path() && u.device().is_cuda();
    let (mut a, wm) = if fast {
        (crate::fused::kwta_relu(&u, k)?, if tr { Some(kwta_mask(&u, k)?.detach()) } else { None })
    } else {
        let wm = kwta_mask(&u, k)?;
        (u.mul(&wm)?.relu()?, Some(wm))
    };
    if kc > 1 {
        a = a.reshape((n, hid))?;
    }
    if m.cfg.arch.relu2 {
        a = a.sqr()?;
    }
    if let (Some((da_, db)), Some(ctx)) = (&bx.dend, v.sig.borrow().ctx.as_ref()) {
        // P: the apical dendrite scales the unit's firing
        a = a.mul(&gain(&head_norm(ctx)?.matmul(da_)?.matmul(db)?)?)?;
    }
    if collect && bx.prune.is_some() {
        v.prune.stats.borrow_mut()[i] = Some(a.detach().abs()?.mean(0)?);
    }
    // merged with v50: W2 as used (`Model::mlp_w2`: l.w2 itself without --dale, so the same tensor and the same int8
    // copy; --dale's effective sign ⊙ f(v) with it), and v50's --neuromod DA on the output after v51's DA (identity
    // without --neuromod). Both are exactly today's numbers while the v50 options are off.
    debug_assert!(std::ptr::eq(l, &m.blocks[i].mlp), "v51::mlp: l is block i's MLP");
    let w2 = m.mlp_w2(i)?;
    let mut y = match &bx.cols {
        Some([_, _, a2, b2]) => col_mm(&a, &w2, a2, b2, o.columns)?,
        None => crate::int8::mm(&a, &w2)?,
    };
    if let (Some((w, c)), Some(z)) = (&bx.da, v.sig.borrow().z.as_ref()) {
        // DA: the surprise scales this block's drive
        y = y.mul(&gain(&z.broadcast_mul(&w.unsqueeze(0)?)?.broadcast_add(c)?)?)?;
    }
    let y = m.v50_da(y)?;
    let wm = match wm {
        Some(w) if tr => Some(if kc > 1 { w.reshape((n, hid))? } else { w }),
        _ => None,
    };
    Ok(Some((a, wm, y)))
}

// ------------------------------------------------------------------------------------------- Pr: the schedule

impl Model {
    /// Pr: the training schedule (`start` steps done, `total` steps in the run); nothing without --prune. Also forgets
    /// the cached unit counts (call it after loading other weights into this model).
    pub fn v51_schedule(&self, start: usize, total: usize) {
        if let Some(v) = ext(self) {
            v.prune.step.set(start);
            v.prune.total.set(total);
            for c in v.prune.alive_n.borrow_mut().iter_mut() {
                *c = None;
            }
        }
    }

    /// Pr (v52 fix_Pr): `v51_schedule(start, total)` for a run of `total` steps (the full schedule, not --max-steps)
    /// resumed at `start`, and the plan for the run's log (None without --prune). Every training binary calls it once,
    /// after a resume and before the first step: grow_gpu never did, so a grow run with --prune cut nothing.
    pub fn v51_begin_schedule(&self, start: usize, total: usize) -> Option<String> {
        self.v51_schedule(start, total);
        let p = self.cfg.arch.v51.prune;
        if p <= 0.0 || ext(self).is_none() {
            return None;
        }
        let (s0, s1) = ((PRUNE_FROM * total as f64).round() as usize, (PRUNE_TO * total as f64).round() as usize);
        // the share cut by the last cut due at or before `start` (after_step cuts at s0, every PRUNE_EVERY steps, and s1)
        let due = if start < s0 { None } else if start >= s1 { Some(s1) } else { Some(s0 + (start - s0) / PRUNE_EVERY * PRUNE_EVERY) };
        Some(format!(
            "Pr: {:.0}% of every block's MLP units cut on the cubic schedule of all {total} steps: from step {s0}, every {PRUNE_EVERY} \
             steps, to step {s1}; this session starts at step {start} ({:.2}% cut by then)",
            p * 100.0,
            due.map_or(0.0, |q| prune_share(p as f64, q, total)) * 100.0
        ))
    }

    /// Pr: the live units per block (None without --prune).
    pub fn v51_alive(&self) -> Result<Option<Vec<usize>>> {
        let Some(v) = ext(self).filter(|_| self.cfg.arch.v51.prune > 0.0) else { return Ok(None) };
        let mut out = Vec::new();
        for b in &v.blocks {
            let (alive, _) = b.prune.as_ref().expect("--prune");
            out.push(alive.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?.round() as usize);
        }
        Ok(Some(out))
    }
}

/// The share cut after `s` of `total` steps (cubic, Zhu & Gupta 2017).
pub fn prune_share(target: f64, s: usize, total: usize) -> f64 {
    let (s0, s1) = ((PRUNE_FROM * total as f64).round(), (PRUNE_TO * total as f64).round());
    let s = s as f64;
    if s < s0 || s1 <= s0 {
        return if s >= s1 && s1 > 0.0 { target } else { 0.0 };
    }
    let f = ((s - s0) / (s1 - s0)).min(1.0);
    target * (1.0 - (1.0 - f).powi(3))
}

/// After an optimizer step (Model::homeostasis_step calls it): Pr's use averages, and a cut when one is due.
pub(crate) fn after_step(m: &Model) -> Result<()> {
    let Some(v) = ext(m).filter(|_| m.cfg.arch.v51.prune > 0.0) else { return Ok(()) };
    let total = v.prune.total.get();
    if total == 0 {
        return Ok(());
    }
    let s = v.prune.step.get() + 1;
    v.prune.step.set(s);
    let mut stats = v.prune.stats.borrow_mut();
    for (bx, st) in v.blocks.iter().zip(stats.iter_mut()) {
        if let (Some((_, use_)), Some(x)) = (&bx.prune, st.take()) {
            let new = use_.affine(USE_DECAY, 0.0)?.add(&x.affine(1.0 - USE_DECAY, 0.0)?)?;
            Var::from_tensor(use_)?.set(&new)?;
        }
    }
    drop(stats);
    let s0 = (PRUNE_FROM * total as f64).round() as usize;
    let s1 = (PRUNE_TO * total as f64).round() as usize;
    if s < s0 || ((s - s0) % PRUNE_EVERY != 0 && s != s1) {
        return Ok(());
    }
    let h = m.cfg.hidden();
    let p = prune_share(m.cfg.arch.v51.prune as f64, s, total);
    let want = h - (p * h as f64).round() as usize;
    for (i, (bx, blk)) in v.blocks.iter().zip(&m.blocks).enumerate() {
        let Some((alive, use_)) = &bx.prune else { continue };
        let al: Vec<f32> = alive.to_dtype(DType::F32)?.to_vec1()?;
        let live: Vec<usize> = (0..h).filter(|&j| al[j] > 0.5).collect();
        if live.len() > want {
            let n1: Vec<f32> = blk.mlp.w1.sqr()?.sum(0)?.sqrt()?.to_vec1()?;
            let n2: Vec<f32> = blk.mlp.w2.sqr()?.sum(1)?.sqrt()?.to_vec1()?;
            let us: Vec<f32> = use_.to_vec1()?;
            let mut sc: Vec<(f32, usize)> = live.iter().map(|&j| (n1[j] * n2[j] * (us[j] + 1e-8), j)).collect();
            sc.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let mut nal = al.clone();
            for &(_, j) in sc.iter().take(live.len() - want) {
                nal[j] = 0.0;
            }
            Var::from_tensor(alive)?.set(&Tensor::from_vec(nal, h, alive.device())?)?;
        }
        v.prune.alive_n.borrow_mut()[i] = Some(want.min(live.len()));
    }
    Ok(())
}

/// Pr: the pruned model with its cut units removed for real: every block keeps its live units (each block must have
/// the same number, a multiple of d), W1's columns and W2's rows of those units, and the config gets --mlp-mult
/// live/d and no --prune. The same logits (kWTA keeps a quarter of the live units either way); half the MLP's work at
/// 50%. Returns the new config and its weights (names -> tensors, on the CPU).
pub fn compact_pruned(cfg: &Config, weights: &HashMap<String, Tensor>) -> std::result::Result<(Config, HashMap<String, Tensor>), String> {
    let o = cfg.arch.v51;
    if o.prune <= 0.0 {
        return Err("compact_pruned: the model has no --prune".into());
    }
    if o.dendrite || o.columns > 0 {
        return Err("compact_pruned: --dendrite and --columns have unit-shaped weights of their own (not handled)".into());
    }
    let e = |x: candle_core::Error| x.to_string();
    let mut live: Vec<Vec<u32>> = Vec::new();
    for i in 0..cfg.layers {
        let al: Vec<f32> = weights.get(&format!("b{i}.v51.alive")).ok_or("no alive mask")?.to_dtype(DType::F32).map_err(e)?.to_vec1().map_err(e)?;
        live.push(al.iter().enumerate().filter(|(_, &a)| a > 0.5).map(|(j, _)| j as u32).collect());
    }
    let n = live[0].len();
    if live.iter().any(|l| l.len() != n) || n % cfg.d != 0 || n % cfg.arch.kwta_div != 0 {
        return Err(format!("compact_pruned: live units per block {:?}: must be equal and a multiple of d {}", live.iter().map(|l| l.len()).collect::<Vec<_>>(), cfg.d));
    }
    let mut ar = cfg.arch;
    ar.v51.prune = 0.0;
    ar.mem.mlp_mult = n / cfg.d;
    let c2 = cfg.with_arch(ar)?;
    let mut out = HashMap::new();
    for (k, t) in weights {
        if k.ends_with(".v51.alive") || k.ends_with(".v51.use") {
            continue;
        }
        out.insert(k.clone(), t.clone());
    }
    for (i, l) in live.iter().enumerate() {
        let idx = Tensor::from_vec(l.clone(), n, &Device::Cpu).map_err(e)?;
        let (w1, w2) = (format!("b{i}.mlp.w1"), format!("b{i}.mlp.w2"));
        let a = weights[&w1].to_device(&Device::Cpu).map_err(e)?.index_select(&idx, 1).map_err(e)?;
        let b = weights[&w2].to_device(&Device::Cpu).map_err(e)?.index_select(&idx, 0).map_err(e)?;
        out.insert(w1, a);
        out.insert(w2, b);
    }
    Ok((c2, out))
}

// ------------------------------------------------------------------------------------------- the carried state

/// What a State carries for the v51 parts, per row.
#[derive(Clone, Default)]
pub struct V51State {
    /// DA/M: block 0's last normed stream [B, d], whether there is one [B] (0 at the start of a stream), the last surprise [B]
    pub prev: Option<Tensor>,
    pub pvalid: Option<Tensor>,
    pub last_s: Option<Tensor>,
    /// M: [B, k]
    pub mood: Option<Tensor>,
    /// Y: [B, 2] the time since each rhythm's last reset (theta, gamma)
    pub rh: Option<Tensor>,
}

const NAMES: [&str; 5] = ["prev", "pvalid", "last_s", "mood", "rh"];

impl V51State {
    fn slots(&self) -> [&Option<Tensor>; 5] {
        [&self.prev, &self.pvalid, &self.last_s, &self.mood, &self.rh]
    }
    fn slots_mut(&mut self) -> [&mut Option<Tensor>; 5] {
        [&mut self.prev, &mut self.pvalid, &mut self.last_s, &mut self.mood, &mut self.rh]
    }

    /// The state at the start of a stream for `rows` rows.
    pub(crate) fn new(m: &Model, rows: usize, dev: &Device) -> Result<V51State> {
        let o = m.cfg.arch.v51;
        let mut s = V51State::default();
        if ext(m).is_none() {
            return Ok(s);
        }
        if o.surprise() {
            s.prev = Some(Tensor::zeros((rows, m.cfg.d), DType::F32, dev)?);
            s.pvalid = Some(Tensor::zeros(rows, DType::F32, dev)?);
            s.last_s = Some(Tensor::zeros(rows, DType::F32, dev)?);
        }
        if o.mood > 0 {
            s.mood = Some(Tensor::zeros((rows, o.mood), DType::F32, dev)?);
        }
        if o.rhythm {
            s.rh = Some(Tensor::zeros((rows, 2), DType::F32, dev)?);
        }
        Ok(s)
    }

    /// Rows `idx` (u32 [m]).
    pub(crate) fn select(&self, idx: &Tensor) -> Result<V51State> {
        let mut s = self.clone();
        for x in s.slots_mut() {
            if let Some(t) = x.as_ref() {
                *x = Some(t.index_select(idx, 0)?);
            }
        }
        Ok(s)
    }

    /// The states' rows one after the other.
    pub(crate) fn cat(parts: &[&V51State]) -> Result<V51State> {
        let mut s = V51State::default();
        for (k, x) in s.slots_mut().into_iter().enumerate() {
            let ts: Vec<&Tensor> = parts.iter().filter_map(|p| p.slots()[k].as_ref()).collect();
            if !ts.is_empty() {
                *x = Some(Tensor::cat(&ts, 0)?);
            }
        }
        Ok(s)
    }

    /// Every tensor under "v51.NAME" (State::save).
    pub(crate) fn save(&self, put: &mut dyn FnMut(String, &Tensor) -> Result<()>) -> Result<()> {
        for (k, x) in NAMES.iter().zip(self.slots()) {
            if let Some(t) = x {
                put(format!("v51.{k}"), t)?;
            }
        }
        Ok(())
    }

    /// The tensors `save` wrote, into a fresh state of the same model (State::load).
    pub(crate) fn load(&mut self, get: &dyn Fn(&str) -> Option<Tensor>) -> std::result::Result<(), String> {
        for (k, x) in NAMES.iter().zip(self.slots_mut()) {
            if x.is_some() {
                *x = Some(get(&format!("v51.{k}")).ok_or_else(|| format!("the saved state has no v51.{k}"))?);
            }
        }
        Ok(())
    }
}

/// What the chat app shows of the v51 parts for one token (`StepTrace::v51`).
#[derive(Clone, Debug, Default)]
pub struct V51Trace {
    /// M: the "mood signal" (the slow units, [k]); DA: the last surprise (nats); Y: the rhythms' phases (radians)
    pub mood: Vec<f32>,
    pub surprise: Option<f32>,
    pub rhythm: Vec<f32>,
}

/// The v51 signals a one-row State holds after a step (None without v51 signals).
pub(crate) fn trace(m: &Model, st: &V51State) -> Result<Option<V51Trace>> {
    if ext(m).is_none() || !(m.cfg.arch.v51.surprise() || m.cfg.arch.v51.rhythm) {
        return Ok(None);
    }
    let row = |t: &Option<Tensor>| -> Result<Vec<f32>> {
        match t {
            Some(x) => x.narrow(0, 0, 1)?.flatten_all()?.to_vec1::<f32>(),
            None => Ok(Vec::new()),
        }
    };
    let a = row(&st.rh)?;
    let rhythm = match ext(m).and_then(|v| v.rhythm.as_ref()) {
        Some(rp) => {
            let w: Vec<f32> = rp.lw.exp()?.to_dtype(DType::F32)?.to_vec1()?;
            a.iter().zip(&w).map(|(a, w)| (a * w).rem_euclid(std::f32::consts::TAU)).collect()
        }
        None => Vec::new(),
    };
    Ok(Some(V51Trace { mood: row(&st.mood)?, surprise: row(&st.last_s)?.first().copied(), rhythm }))
}

#[cfg(test)]
mod golden;
#[cfg(test)]
pub(crate) mod tests;
