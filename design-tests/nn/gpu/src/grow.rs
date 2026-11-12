// TESTING / DESIGN ONLY — ava_gpu::grow: A15, exact growth (grow_exact). A trained model is grown deeper and its MLPs
// wider so that the grown model computes the same function: same logits, same talking state, up to float rounding.
//
// The reference maths is design-tests/nn/parts/src/growth.rs (`widen`; Net2Net, Chen, Goodfellow & Shlens, ICLR 2016):
// new units copy incoming weights and the outgoing rows are divided by the copy count; a new residual block whose
// output weights are 0 is already exact.
//
// --mlp-times C   every MLP hidden unit is copied C times: w1's columns tiled C times (copy j of unit u is unit j·H + u),
//                 w2's rows shared among the copies (Ava's MLP has no bias). --mlp-split even is Net2Net's rule (every
//                 copy's row divided by C); the default --mlp-split first keeps the whole row on copy 0 and starts the
//                 others at 0. Both are exact in real arithmetic, but even changes how the sum is grouped (about 1e-7
//                 relative per MLP), and a spiking model can amplify that (a spike flips): the noise-filled test model of
//                 Ava 0.4 at d64 ended 1.6e-4 apart with even (as far as it moves when its w2 moves by one part in 2^23),
//                 bit for bit equal with first. The trained full-all-a16 (d128) is calm: 9.9e-7 with even, 0 with first
//                 (Split has the details). When the check fails anyway, grow_exact writes nothing and reports the model's
//                 own sensitivity beside the gap. The width is recorded as the block option
//                 --mlp-mult (src/memory.rs), so loading the grown model builds the wide MLP. kWTA (model::kwta_mask, and
//                 the fused CUDA kernel, which uses the same rule) keeps every value >= the k-th largest of its row. Let a
//                 row's old values be u (H units, k kept, k-th largest value t). The widened row holds every u_i C times,
//                 so its C·k-th largest value is again t (the C copies of the k largest come first), and the new mask
//                 keeps exactly the copies of the units the old mask kept, ties included (a value equal to t is kept in
//                 both). Each kept copy gives act(u_i)·w2_i / C, and the C copies add up to act(u_i)·w2_i, for relu and
//                 relu² alike. So it is exact if and only if the new k is exactly C·k: k = hidden / kwta_div with the
//                 hidden width divisible by kwta_div (Config::validate), so C·hidden / kwta_div = C·k always; `layout`
//                 checks it anyway and refuses otherwise. The copies' values are bit-identical (the same column of
//                 weights against the same input; checked by the tests on the CPU), so no tie is ever split. A7
//                 (--halting) ponders the same MLP step N times (wave5::mlp_step, the grown k), each pass exact, and its
//                 stop signal reads the pass's output, which is unchanged: exact too. H12 (--topo) is a training loss on
//                 the units' sheet: the wider sheet changes that loss, not the model.
// --deeper N --at L  N new blocks go after block L (default: the last), each computing the identity at first: every
//                 path from a new block into the stream starts at 0 (READOUTS: the spiking layer's wout, the memory's
//                 or attention's wo, the MLP's w2, the engram's, working memory's, workspace's, structure read's,
//                 binding memory's and oscillators' wo, the cerebellum's read-out, mood's vector, the product-key
//                 values and the dense mix). Everything else in a new block takes a fresh init (seeded by --seed), so it
//                 can learn; the tests fill those with noise to show that the list is complete. Index-dependent parts:
//                 * --residual dense: block i mixes the streams x_0..x_{i+1} with weights w_i [i + 2]. A new block's
//                   weights are 0 (its output is its input), and every later block's weights are re-indexed: the
//                   weight of old stream j goes to its new place, the new blocks' streams get 0. Each block then reads
//                   exactly the input it read before.
//                 * --value-residual: v0 comes from block 0, and new blocks go after block L >= 0, so block 0 stays first.
//                 * --workspace: wave5::ws_block = layers/2 moves when the layer count does, so the grown model records
//                   the workspace's block with --ws-layer when it differs from the new middle.
//                 * --pk-mem: the product-key memory's block is recorded with --pk-layer the same way.
//                 * --engram L: the engram's block index moves with its block (ckpt::detect reads it from the names).
//                 * --cerebellum: each block's granule wiring is drawn from a seed of its block number (not saved), so
//                   only blocks appended at the top keep every old block's wiring: anything else is refused.
//                 * model A: attention sits in every 4th block counting back from the last; a growth that moves an old
//                   block across that pattern is refused (N a multiple of 4 always keeps it).
//                 * --feedback: the top block's state is unchanged by identity blocks above it.
//                 * --halting in a new block: its passes all return the input (w2 = 0), mixed by the stop weights. With
//                   the part's start (stop bias 4) they sum to 1 only up to rounding (2e-7), and --modulate's surprise
//                   carries that into every block's spikes (4.7e-5 measured), so a new block's stop bias starts at 0
//                   (PINNED): weights 1/2, 1/4, 1/4, powers of two, whose mix is the input bit for bit. That holds for 2
//                   or 3 passes only; --halting 4 or more is refused for --deeper.
//                 * --topo divides its weight by the layer count: a training loss only.
// --wider-d is not offered: the cerebellum's fixed wiring is drawn for the width d (not saved), the fixed gate offsets
// are tied to head numbers, and the tied embedding scales the logits with the width; see the A15 report.
//
// Not carried: the optimizer's moments and step count. Growth starts a new run (a fresh AdamW, its own warm-up).
// The tool checks itself: the grown model's logits on a fixed batch must equal the source's within TOL (relative to the
// largest logit), before anything is written and again after the written file is loaded back (ckpt::load_auto).

use crate::model::{Config, Kind, Model, MLP_MULT, POS_EMB};
use candle_core::{DType, Device, Shape, Tensor, Var};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder, VarMap};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// The bar: the largest logit difference over the largest logit.
pub const TOL: f32 = 1e-5;

/// The tensors (names after "b{i}.") through which a block writes into the stream: 0 in every new block.
pub const READOUTS: [&str; 14] = [
    "mem.wout",
    "hebb.wo",
    "attn.wo",
    "mlp.w2",
    "eng.wo",
    "parts.wm.wo",
    "parts.ws.wo",
    "parts.tem.wo",
    "parts.holo.wo",
    "parts.sync.wo",
    "parts.cb.readout",
    "parts.mood.w",
    "pk.values",
    "dense",
];

/// The tensors (names after "b{i}.") that a new block needs at fixed values, and those values: A7's stop weights at
/// w = 0, b = 0, so every pass stops with probability exactly 1/2 and the stop weights (1/2, 1/4, 1/4 for 3 passes) are
/// powers of two. A new block's passes all return the input, and only then does their mix return it bit for bit (with
/// the part's b = 4 the weights sum to 1 up to rounding, about 2e-7, and --modulate's surprise passes that on to every
/// block's spiking: 4.7e-5 measured on Ava 0.4 at d64).
pub const PINNED: [(&str, f32); 2] = [("parts.halt.w", 0.0), ("parts.halt.b", 0.0)];

/// A7: the most passes whose mix of identical passes is exact at b = 0 (checked by the tests for 2 to 8 passes: 2 and 3
/// give the input bit for bit, 4 and more do not).
pub const HALT_EXACT_MAX: usize = 3;

/// How a widened MLP shares each unit's outgoing row among its C copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// copy 0 keeps the whole row, the other copies start at 0. The extra terms of the sum are exact zeros, so the output
    /// is bit for bit the old one wherever the matmul keeps the old terms' grouping (the CPU gemm does up to 512 hidden
    /// units, not at 1536: it cuts the inner dimension into equal blocks whose size follows its length). The copies' rows
    /// differ, so training tells them apart (the zero rows get gradients at once, their w1 columns from the next step
    /// on). The default.
    First,
    /// Net2Net's rule: every copy gets the row divided by C. Exact in real arithmetic; the sum's grouping changes (about
    /// 1e-7), and the copies stay identical under training (equal values, equal gradients) until something breaks the tie.
    Even,
}

/// What to grow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// copies of every MLP hidden unit (1: the MLPs stay as they are)
    pub mlp_times: usize,
    /// how the copies share the outgoing rows
    pub split: Split,
    /// new blocks
    pub deeper: usize,
    /// they go after this block (None: after the last)
    pub at: Option<usize>,
    /// the seed of the new blocks' fresh init
    pub seed: u64,
}

impl Default for Plan {
    fn default() -> Plan {
        Plan { mlp_times: 1, split: Split::First, deeper: 0, at: None, seed: 1 }
    }
}

/// The grown model's shape: its config, the new index of every old block, the new blocks, and what was recorded.
#[derive(Clone, Debug)]
pub struct Layout {
    pub cfg: Config,
    pub map: Vec<usize>,
    pub fresh: Vec<usize>,
    pub notes: Vec<String>,
}

/// The grown model's shape for `plan`, or why it cannot compute exactly what `cfg` computes.
pub fn layout(cfg: &Config, plan: &Plan) -> Result<Layout, String> {
    let (c, n, l0) = (plan.mlp_times, plan.deeper, cfg.layers);
    if c == 0 {
        return Err("--mlp-times C: C >= 1 copies of each unit".into());
    }
    if n == 0 && plan.at.is_some() {
        return Err("--at L places new blocks: it needs --deeper N".into());
    }
    if n == 0 && c == 1 {
        return Err("nothing to grow: give --mlp-times C (C >= 2) and/or --deeper N".into());
    }
    let at = plan.at.unwrap_or(l0 - 1);
    if at >= l0 {
        return Err(format!("--at {at}: the new blocks go after block L, 0 <= L < {l0}"));
    }
    let layers = l0 + n;
    let map: Vec<usize> = (0..l0).map(|i| if i <= at { i } else { i + n }).collect();
    let fresh: Vec<usize> = (at + 1..=at + n).collect();
    let mut ar = cfg.arch;
    let mut notes = Vec::new();
    if c > 1 {
        if cfg.kind != Kind::D {
            return Err(format!("--mlp-times records the new width as the block option --mlp-mult, which is for model D (this is {})", cfg.kind.name()));
        }
        let m0 = if ar.mem.mlp_mult == 0 { MLP_MULT } else { ar.mem.mlp_mult };
        ar.mem.mlp_mult = m0 * c;
        notes.push(format!("MLP hidden width {}·d -> {}·d (--mlp-mult {})", m0, m0 * c, m0 * c));
    }
    if n > 0 {
        // v44 K1: a new block's zero output is still one of the sources the attention mixes (a softmax share, and n
        // grows), so the grown model would not compute the old one
        if ar.attn_res {
            return Err("--deeper with --residual attn: a new block changes every later mix, so the grown model is not the old one".into());
        }
        if ar.mem.cerebellum > 0 && at != l0 - 1 {
            return Err(format!(
                "--cerebellum: each block's granule wiring is drawn from a seed of its block number (wave5::CB_SEED + 7919·i, \
                 not saved), so new blocks after block {at} would rewire blocks {}..{}; grow at the top (--at {}), where no \
                 old block moves",
                at + 1,
                l0 - 1,
                l0 - 1
            ));
        }
        if ar.mem.halting > HALT_EXACT_MAX {
            return Err(format!(
                "--halting {}: a new block's passes all return its input, mixed by stop weights that must sum to exactly 1 in \
                 float32; with stop bias 0 they do for 2 or 3 passes (powers of two), for {} passes only a saturated stop \
                 (no gradient) would, and the rounding reaches every later block's spikes",
                ar.mem.halting, ar.mem.halting
            ));
        }
        if ar.mem.workspace > 0 {
            let old = crate::wave5::ws_block_of(&ar.mem, l0);
            let new = map[old];
            let mid = crate::wave5::ws_block(layers);
            ar.mem.ws_layer = if ar.mem.ws_layer.is_none() && new == mid { None } else { Some(new) };
            if let Some(w) = ar.mem.ws_layer {
                notes.push(format!("the workspace stays with its block: {old} -> {w} (--ws-layer {w}; the middle of {layers} blocks is {mid})"));
            }
        }
        if let Some(old) = ar.mem.pk_block(l0) {
            let new = map[old];
            ar.mem.pk_layer = if ar.mem.pk_layer.is_none() && new == layers - 1 { None } else { Some(new) };
            if let Some(p) = ar.mem.pk_layer {
                notes.push(format!("the product-key memory stays with its block: {old} -> {p} (--pk-layer {p})"));
            }
        }
        notes.push(format!("{n} new block(s) after block {at}: old blocks -> {map:?}, new {fresh:?}"));
    }
    let engram = cfg.engram.map(|l| map[l]);
    let g = Config { layers, engram, arch: ar, ..*cfg };
    g.validate().map_err(|e| format!("the grown model: {e}"))?;
    if cfg.kind == Kind::A {
        if let Some(i) = (0..l0).find(|&i| g.attn_layer(map[i]) != cfg.attn_layer(i)) {
            return Err(format!(
                "model A has attention in every 4th block counting back from the last: block {i} (now {}) would {} it; grow by a \
                 multiple of 4 blocks",
                map[i],
                if cfg.attn_layer(i) { "lose" } else { "gain" }
            ));
        }
    }
    if c > 1 {
        let (k0, k1) = (cfg.kwta_k(), g.kwta_k());
        if g.hidden() != c * cfg.hidden() {
            return Err(format!("the grown MLP is {} wide, not {c} x {}", g.hidden(), cfg.hidden()));
        }
        let all = k0 >= cfg.hidden() && k1 >= g.hidden();
        if !all && k1 != c * k0 {
            return Err(format!(
                "kWTA keeps {k0} of {} units, the widened MLP would keep {k1} of {}, not {c} x {k0}: the copies of the old winners \
                 would not be exactly the new winners",
                cfg.hidden(),
                g.hidden()
            ));
        }
    }
    Ok(Layout { cfg: g, map, fresh, notes })
}

/// A grown model.
pub struct Grown {
    pub layout: Layout,
    pub varmap: VarMap,
    pub model: Model,
}

/// "b{i}.rest" -> (i, rest).
pub fn block_of(name: &str) -> Option<(usize, &str)> {
    let (i, rest) = name.strip_prefix('b')?.split_once('.')?;
    Some((i.parse().ok()?, rest))
}

/// Grow the model `cfg` with the weights `src` by `plan` (on `dev`; the tool uses the CPU). Every tensor of the source
/// finds exactly one place in the grown model and every tensor of the grown model is filled, or nothing is built.
pub fn grow(cfg: &Config, src: &VarMap, plan: &Plan, dev: &Device) -> Result<Grown, String> {
    let lay = layout(cfg, plan)?;
    let g = lay.cfg;
    // the grown model's tensors: names, shapes and the new blocks' fresh init (seeded; the others are placeholders)
    let fresh_set: HashSet<usize> = lay.fresh.iter().copied().collect();
    let made = Arc::new(Mutex::new(BTreeMap::new()));
    let backend = SeededInit { seed: plan.seed, draw: fresh_set.clone(), made: made.clone() };
    Model::new(g, VarBuilder::from_backend(Box::new(backend), DType::F32, dev.clone())).map_err(|e| format!("build the grown model: {e}"))?;
    let made = std::mem::take(&mut *made.lock().unwrap());
    let src_t: HashMap<String, Tensor> = src.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.as_tensor().clone())).collect();
    let inv: HashMap<usize, usize> = lay.map.iter().enumerate().map(|(old, &new)| (new, old)).collect();
    let mut used: HashSet<String> = HashSet::new();
    let mut values: Vec<(String, Tensor)> = Vec::with_capacity(made.len());
    let e = |name: &str, x: candle_core::Error| format!("{name}: {x}");
    for (name, init) in made {
        let want = init.dims().to_vec();
        let val = match block_of(&name) {
            Some((j, rest)) if fresh_set.contains(&j) => {
                if READOUTS.contains(&rest) {
                    init.zeros_like().map_err(|x| e(&name, x))?
                } else if let Some((_, v)) = PINNED.iter().find(|(p, _)| *p == rest) {
                    init.ones_like().and_then(|o| o.affine(*v as f64, 0.0)).map_err(|x| e(&name, x))?
                } else {
                    init
                }
            }
            Some((j, rest)) => {
                let old = *inv.get(&j).ok_or_else(|| format!("{name}: block {j} is neither old nor new"))?;
                let sname = format!("b{old}.{rest}");
                let s = src_t.get(&sname).ok_or_else(|| format!("{name}: the source has no {sname}"))?;
                used.insert(sname.clone());
                carry(rest, s, old, &lay.map, plan.mlp_times, plan.split).map_err(|x| e(&sname, x))?
            }
            None => {
                let s = src_t.get(&name).ok_or_else(|| format!("the grown model has {name}, which the source lacks"))?;
                used.insert(name.clone());
                s.copy().map_err(|x| e(&name, x))?
            }
        };
        if val.dims() != want.as_slice() {
            return Err(format!("{name}: the grown model needs {want:?}, the source gives {:?}", val.dims()));
        }
        values.push((name, val.to_dtype(DType::F32).and_then(|v| v.to_device(dev)).map_err(|x| e("to device", x))?));
    }
    let mut left: Vec<&String> = src_t.keys().filter(|k| !used.contains(*k)).collect();
    if !left.is_empty() {
        left.sort();
        return Err(format!("the source's {left:?} have no place in the grown model"));
    }
    let varmap = VarMap::new();
    {
        let mut data = varmap.data().lock().unwrap();
        for (name, v) in &values {
            data.insert(name.clone(), Var::from_tensor(v).map_err(|x| e(name, x))?);
        }
    }
    let model = Model::new(g, VarBuilder::from_varmap(&varmap, DType::F32, dev)).map_err(|x| format!("build the grown model: {x}"))?;
    if varmap.data().lock().unwrap().len() != values.len() {
        return Err("the grown model made tensors the plan did not fill".into());
    }
    model.set_track_activity(false);
    Ok(Grown { layout: lay, varmap, model })
}

/// An old block's tensor `rest` [b{old}.rest] in the grown model (its new index is map[old]).
fn carry(rest: &str, s: &Tensor, old: usize, map: &[usize], c: usize, split: Split) -> candle_core::Result<Tensor> {
    match rest {
        "mlp.w1" if c > 1 => Tensor::cat(&vec![s; c], 1),
        "mlp.w2" if c > 1 => match split {
            Split::Even => {
                let w = s.affine(1.0 / c as f64, 0.0)?;
                Tensor::cat(&vec![&w; c], 0)
            }
            Split::First => {
                let z = s.zeros_like()?;
                let mut rows = vec![s.copy()?];
                rows.extend((1..c).map(|_| z.clone()));
                Tensor::cat(&rows, 0)
            }
        },
        "dense" => {
            // R1 dense: block `old` mixes x_0 (the embedding) and the outputs of blocks 0..=old (x_{j+1}); in the grown
            // model the output of old block j is stream map[j] + 1 and the new blocks' streams get 0
            let w: Vec<f32> = s.to_dtype(DType::F32)?.to_vec1()?;
            let mut v = vec![0f32; map[old] + 2];
            v[0] = w[0];
            for j in 0..=old {
                v[map[j] + 1] = w[j + 1];
            }
            Tensor::from_vec(v, map[old] + 2, s.device())
        }
        _ => s.copy(),
    }
}

/// The fresh init of the grown model through `Model::new`'s own init hints, from a seed instead of candle's CPU RNG (which
/// cannot be seeded): every tensor gets its own stream (the seed and its name), so the init does not depend on build
/// order. Only the new blocks' tensors are drawn; the others are zeros (their values come from the source).
struct SeededInit {
    seed: u64,
    draw: HashSet<usize>,
    made: Arc<Mutex<BTreeMap<String, Tensor>>>,
}

impl SimpleBackend for SeededInit {
    fn get(&self, s: Shape, name: &str, h: Init, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let n = s.elem_count();
        let drawn = block_of(name).is_some_and(|(i, _)| self.draw.contains(&i));
        let v: Vec<f32> = if !drawn {
            vec![0.0; n]
        } else {
            let mut r = Rng::new(self.seed ^ fnv(name));
            match h {
                Init::Const(c) => vec![c as f32; n],
                Init::Randn { mean, stdev } => (0..n).map(|_| (mean + stdev * r.normal()) as f32).collect(),
                Init::Uniform { lo, up } => (0..n).map(|_| (lo + (up - lo) * r.uniform()) as f32).collect(),
                _ => candle_core::bail!("grow: {name}: a Kaiming init has no seeded draw here"),
            }
        };
        // a variable, so Model::new's in-place sets (R3's log-spaced leaks) reach the tensor kept here
        let t = Var::from_tensor(&Tensor::from_vec(v, s, dev)?.to_dtype(dtype)?)?.as_tensor().clone();
        self.made.lock().unwrap().insert(name.to_string(), t.clone());
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, _: DType, _: &Device) -> candle_core::Result<Tensor> {
        candle_core::bail!("grow: {name}: no stored tensor")
    }

    fn contains_tensor(&self, _: &str) -> bool {
        false
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// [0, 1)
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

// ------------------------------------------------------------------------------------------- the check

/// Fixed token ids [b, t] below `vocab`; with the glance's vocabulary, numbers with their length marks among them.
pub fn check_ids(b: usize, t: usize, vocab: usize, seed: u64) -> Vec<u32> {
    let mut r = Rng::new(seed);
    let top = vocab.min(4096) as u64;
    let mut v: Vec<u32> = (0..b * t).map(|_| (r.next() % top) as u32).collect();
    if vocab == ava_data::numbers::VOCAB_GLANCE {
        let mark = ava_data::numbers::MARK_BASE + 2;
        for row in 0..b {
            for p in (2..t.saturating_sub(4)).step_by(9) {
                v[row * t + p] = mark;
                for k in 0..3 {
                    v[row * t + p + 1 + k] = 48 + ((p + k) % 10) as u32;
                }
            }
        }
    }
    v
}

/// (the largest |logit difference|, that over the largest |logit|) of two models on `ids` [b, t].
pub fn logit_gap(a: &Model, b: &Model, ids: &Tensor) -> Result<(f32, f32), String> {
    let la = a.forward(ids).map_err(|e| format!("source forward: {e}"))?;
    let lb = b.forward(ids).map_err(|e| format!("grown forward: {e}"))?;
    gap(&la, &lb).map_err(|e| e.to_string())
}

pub fn gap(want: &Tensor, got: &Tensor) -> candle_core::Result<(f32, f32)> {
    let max = |x: &Tensor| x.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>();
    let d = max(&want.sub(got)?)?;
    let top = max(want)?;
    Ok((d, if top > 0.0 { d / top } else { d }))
}

// ------------------------------------------------------------------------------------------- grow_exact

pub const USAGE: &str = "usage: grow_exact --init DIR_OR_FILE --out DIR [--mlp-times C [--mlp-split first|even]] [--deeper N [--at L]]\n\
    \x20                 [--seed 1] [--kind D] [--check-rows 2] [--check-tokens 96]";

/// What grow_exact did (for its log and the tests).
#[derive(Clone, Debug)]
pub struct Report {
    pub out: PathBuf,
    pub cfg: Config,
    pub notes: Vec<String>,
    /// (absolute, relative) logit gaps: in memory, and after loading the written file back
    pub gap_mem: (f32, f32),
    pub gap_file: (f32, f32),
    /// seconds: load, grow, check in memory, save, load back and check
    pub secs: [f64; 5],
    pub params: (usize, usize),
}

/// The model file a --init names: the file itself, or DIR/model.safetensors.
pub fn model_file(p: &Path) -> PathBuf {
    if p.is_dir() {
        p.join("model.safetensors")
    } else {
        p.to_path_buf()
    }
}

/// grow_exact's whole run for its arguments (without the program name).
pub fn run(args: &[String]) -> Result<Report, String> {
    let (mut init, mut out) = (None::<PathBuf>, None::<PathBuf>);
    let mut plan = Plan::default();
    let (mut kind, mut rows, mut toks) = (Kind::D, 2usize, 96usize);
    let mut i = 0;
    while i < args.len() {
        let k = args[i].as_str();
        if k == "--help" || k == "-h" {
            return Err(USAGE.into());
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{k} needs a value\n{USAGE}"))?.as_str();
        let int = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {k}"));
        match k {
            "--init" => init = Some(v.into()),
            "--out" => out = Some(v.into()),
            "--mlp-times" => plan.mlp_times = int(v)?,
            "--mlp-split" => {
                plan.split = match v {
                    "first" => Split::First,
                    "even" => Split::Even,
                    _ => return Err(format!("--mlp-split {v:?}: first or even")),
                }
            }
            "--deeper" => plan.deeper = int(v)?,
            "--at" => plan.at = Some(int(v)?),
            "--seed" => plan.seed = v.parse().map_err(|_| format!("bad seed {v:?}"))?,
            "--kind" | "--arch" => kind = Kind::parse(v)?,
            "--check-rows" => rows = int(v)?.max(1),
            "--check-tokens" => toks = int(v)?.max(2),
            "--wider-d" => return Err("--wider-d is not offered: it cannot be made exact for every part (see src/grow.rs)".into()),
            _ => return Err(format!("unknown flag {k}\n{USAGE}")),
        }
        i += 2;
    }
    let (Some(init), Some(out)) = (init, out) else { return Err(format!("--init and --out are required\n{USAGE}")) };
    let src_path = model_file(&init);
    let dev = Device::Cpu;
    for f in ["model.safetensors", "model.arch.txt", "arch.txt"] {
        if out.join(f).exists() {
            return Err(format!("{} already holds {f}: give grow_exact an empty folder", out.display()));
        }
    }
    let clock = std::time::Instant::now();
    let mut secs = [0f64; 5];
    let mut lap = |k: usize, t0: &mut f64| {
        let now = clock.elapsed().as_secs_f64();
        secs[k] = now - *t0;
        *t0 = now;
    };
    let mut t0 = 0f64;
    let ctx = {
        let t = crate::ckpt::read(&src_path)?;
        t.get(POS_EMB).map_or(Ok(256), |p| p.dim(0)).map_err(|e| e.to_string())?
    };
    let (cfg, src_vm, src) = crate::ckpt::load_auto(&src_path, kind, ctx, &dev)?;
    src.set_track_activity(false);
    lap(0, &mut t0);
    let gr = grow(&cfg, &src_vm, &plan, &dev)?;
    lap(1, &mut t0);
    let t = toks.min(cfg.ctx);
    let ids = Tensor::from_vec(check_ids(rows, t, cfg.vocab, plan.seed ^ 0x5eed), (rows, t), &dev).map_err(|e| e.to_string())?;
    let want = src.forward(&ids).map_err(|e| format!("source forward: {e}"))?;
    let got = gr.model.forward(&ids).map_err(|e| format!("grown forward: {e}"))?;
    let gap_mem = gap(&want, &got).map_err(|e| e.to_string())?;
    lap(2, &mut t0);
    if !(gap_mem.1 <= TOL) {
        let mut why = String::new();
        if plan.mlp_times > 1 {
            // how far the source itself moves when every MLP's w2 moves by one part in 2^23 (the size of a regrouped sum)
            let (_, vm2, m2) = crate::ckpt::load_auto(&src_path, kind, ctx, &dev)?;
            m2.set_track_activity(false);
            for (k, v) in vm2.data().lock().unwrap().iter() {
                if k.ends_with(".mlp.w2") {
                    v.set(&v.as_tensor().affine(1.0 + f32::EPSILON as f64, 0.0).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
                }
            }
            let own = gap(&want, &m2.forward(&ids).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            why = format!(
                ": the widened MLP's matmul sums its {} units in other groups than the old {} (exact in real arithmetic, about \
                 1e-7 per MLP in float), and this model amplifies rounding: the source itself moves by {:.3e} relative when \
                 every MLP's w2 moves by one part in 2^23",
                gr.layout.cfg.hidden(),
                cfg.hidden(),
                own.1
            );
        }
        return Err(format!("the grown model is not exact: logit gap {:.3e} (relative {:.3e} > {TOL:e}){why}; nothing written", gap_mem.0, gap_mem.1));
    }
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let file = out.join("model.safetensors");
    gr.varmap.save(&file).map_err(|e| format!("save {}: {e}", file.display()))?;
    crate::ckpt::write_arch_beside(&file, &gr.layout.cfg.arch)?;
    lap(3, &mut t0);
    let (_, _, back) = crate::ckpt::load_auto(&file, kind, ctx, &dev)?;
    back.set_track_activity(false);
    let got2 = back.forward(&ids).map_err(|e| format!("reloaded forward: {e}"))?;
    let gap_file = gap(&want, &got2).map_err(|e| e.to_string())?;
    lap(4, &mut t0);
    let rep = Report { out: out.clone(), cfg: gr.layout.cfg, notes: gr.layout.notes.clone(), gap_mem, gap_file, secs, params: (count(&src_vm), count(&gr.varmap)) };
    let line = |k: &str, v: String| format!("{k}\t{v}\n");
    let mut rec = String::new();
    rec += &line("source", src_path.display().to_string());
    rec += &line("plan", format!("mlp_times={} split={:?} deeper={} at={:?} seed={}", plan.mlp_times, plan.split, plan.deeper, plan.at, plan.seed));
    rec += &line("from", format!("{} d={} layers={} hidden={}{}", cfg.kind.name(), cfg.d, cfg.layers, cfg.hidden(), cfg.arch.signature()));
    rec += &line("to", format!("{} d={} layers={} hidden={}{}", rep.cfg.kind.name(), rep.cfg.d, rep.cfg.layers, rep.cfg.hidden(), rep.cfg.arch.signature()));
    for n in &rep.notes {
        rec += &line("note", n.clone());
    }
    rec += &line("params", format!("{} -> {}", rep.params.0, rep.params.1));
    rec += &line("logit_gap", format!("{:.3e} (relative {:.3e}) in memory, {:.3e} (relative {:.3e}) loaded back", gap_mem.0, gap_mem.1, gap_file.0, gap_file.1));
    rec += &line("check", format!("{rows} x {t} tokens"));
    rec += &line("seconds", format!("load {:.2} grow {:.2} check {:.2} save {:.2} reload+check {:.2}", secs[0], secs[1], secs[2], secs[3], secs[4]));
    rec += &line("optimizer", "not carried: start a new run (fresh AdamW and warm-up)".into());
    std::fs::write(out.join("grow.txt"), rec).map_err(|e| format!("grow.txt: {e}"))?;
    if !(gap_file.1 <= TOL) {
        return Err(format!("the written model does not load back exactly: logit gap {:.3e} (relative {:.3e})", gap_file.0, gap_file.1));
    }
    Ok(rep)
}

fn count(vm: &VarMap) -> usize {
    vm.data().lock().unwrap().values().map(|v| v.elem_count()).sum()
}

#[cfg(test)]
mod tests;
