// TESTING / DESIGN ONLY — ava_gpu::trainkit: v42 (Wave 2A) helpers for grow_gpu's three phases, kept here so they
// can be tested on the CPU. Each one sits behind a grow_gpu flag whose default is the old behaviour.
//
// - X2 replay cursor: a small file `night=N`; night N's replay windows come from their own seed (night 0 is the old
//   seed), and the file moves on by one after each night, so nights stop replaying the same windows.
// - J3 bad-batch guard: a step is skipped when a loss or the gradient norm is not finite, or the norm is more than
//   K times the running median of the last accepted norms.
// - Q3 late answers: the answer batches placed in the last fraction of the steps (0 = spread over all, the old rule).
// - Q4 EMA: a running average copy of the weights, saved as <name>.ema.safetensors.
// - U4 group baseline: K draws per question, the group's mean reward as the baseline (GRPO style), and "I don't
//   know" as an extra action worth 0 (right +1, wrong −1: TruthRL's three-way reward).
// - U2 KL(reference || current) over next-token distributions, the forgetting gate.

use crate::mc::{grade_one, Item};
use crate::model::Model;
use ava_data::rng::{derive_seed, SplitMix64};
use candle_core::backprop::GradStore;
use candle_core::{Device, Result as CResult, Tensor, Var, D};
use candle_nn::VarMap;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

// ------------------------------------------------------------------------------------------- X2 replay cursor

/// The replay stream's label ("replay"), as grow_gpu has always used it.
pub const REPLAY_LABEL: u64 = 0x7265_706c_6179;
/// The label that moves a replay stream on to another night ("night").
pub const NIGHT_LABEL: u64 = 0x6e69_6768_74;

/// The seed of replay stream `stream`. Without a cursor (None) and on night 0 it is the old seed exactly; night N > 0
/// gets a stream of its own.
pub fn replay_seed(seed: u64, stream: u64, night: Option<u64>) -> u64 {
    let base = derive_seed(seed, REPLAY_LABEL, stream);
    match night {
        None | Some(0) => base,
        Some(n) => derive_seed(base, NIGHT_LABEL, n),
    }
}

/// The night a cursor file holds (`night=N`); a missing file is night 0.
pub fn read_night(path: &Path) -> Result<u64, String> {
    match std::fs::read_to_string(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(format!("replay cursor {}: {e}", path.display())),
        Ok(text) => text
            .lines()
            .find_map(|l| l.trim().strip_prefix("night="))
            .ok_or_else(|| format!("replay cursor {}: no night= line", path.display()))?
            .trim()
            .parse()
            .map_err(|_| format!("replay cursor {}: bad night", path.display())),
    }
}

/// Write `night=N` (to a temporary file first, then renamed into place, so a crash never leaves half a file).
pub fn write_night(path: &Path, night: u64) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, format!("night={night}\n")).map_err(|e| format!("replay cursor {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("replay cursor {}: {e}", path.display()))
}

/// The starts of `rows` random windows of `ctx` + 1 tokens in a text of `len` tokens (grow_gpu's replay draw).
pub fn window_starts(rng: &mut SplitMix64, rows: usize, len: usize, ctx: usize) -> Vec<usize> {
    (0..rows).map(|_| rng.below((len - ctx - 1) as u64) as usize).collect()
}

// ------------------------------------------------------------------------------------------- J3 bad batches

/// Why a step was skipped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    Good,
    NonFinite,
    /// the gradient norm was more than K times this running median
    Spike(f64),
}

/// The running median of the last `window` accepted gradient norms, and the skip rule.
pub struct BadBatch {
    mult: f64,
    window: usize,
    min_hist: usize,
    hist: VecDeque<f64>,
    /// steps skipped so far
    pub skipped: usize,
    /// skipped steps in a row (0 after a good step)
    pub in_a_row: usize,
}

impl BadBatch {
    /// `mult`: K; `window`: how many recent norms the median is over. The spike rule waits for min(20, window) norms.
    pub fn new(mult: f64, window: usize) -> BadBatch {
        let window = window.max(1);
        BadBatch { mult, window, min_hist: window.min(20), hist: VecDeque::with_capacity(window), skipped: 0, in_a_row: 0 }
    }

    pub fn median(&self) -> Option<f64> {
        if self.hist.len() < self.min_hist {
            return None;
        }
        let mut v: Vec<f64> = self.hist.iter().copied().collect();
        v.sort_by(f64::total_cmp);
        let n = v.len();
        Some(if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) })
    }

    /// Judge one step from its losses and gradient norm, BEFORE the optimizer step; a good step's norm joins the
    /// history, a bad one is counted.
    pub fn judge(&mut self, losses: &[f32], gn: f64) -> Verdict {
        let v = if !gn.is_finite() || losses.iter().any(|l| !l.is_finite()) {
            Verdict::NonFinite
        } else {
            match self.median() {
                Some(m) if m > 0.0 && gn > self.mult * m => Verdict::Spike(m),
                _ => Verdict::Good,
            }
        };
        if v == Verdict::Good {
            if self.hist.len() == self.window {
                self.hist.pop_front();
            }
            self.hist.push_back(gn);
            self.in_a_row = 0;
        } else {
            self.skipped += 1;
            self.in_a_row += 1;
        }
        v
    }
}

// ------------------------------------------------------------------------------------------- Q3 late answers

/// Where the answer batches go: `total` of them spread evenly over the steps `start..steps`, at most one per step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QaPlan {
    pub start: usize,
    pub steps: usize,
    pub total: usize,
}

impl QaPlan {
    /// `wanted` answer batches (epochs · batches) over `steps` steps, in the last `late_frac` of them (0: all steps,
    /// grow_gpu's old even spread, the same numbers exactly).
    pub fn new(steps: usize, wanted: usize, late_frac: f64) -> QaPlan {
        let span = if late_frac > 0.0 && steps > 0 { ((steps as f64 * late_frac.min(1.0)).round() as usize).clamp(1, steps) } else { steps };
        let total = if steps == 0 { 0 } else { wanted.min(span) };
        QaPlan { start: steps - span, steps, total }
    }

    /// The answer batch taken at step `s` (its index in 0..total), if any.
    pub fn at(&self, s: usize) -> Option<usize> {
        if s < self.start {
            return None;
        }
        let (k, span) = (s - self.start, (self.steps - self.start).max(1));
        let (lo, hi) = (k * self.total / span, (k + 1) * self.total / span);
        (hi > lo).then_some(lo)
    }

    /// Steps per answer batch inside the span.
    pub fn every(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.steps - self.start) as f64 / self.total as f64
        }
    }
}

// ------------------------------------------------------------------------------------------- Q4 EMA

/// A running average copy of every weight: e ← decay·e + (1 − decay)·w after each optimizer step.
pub struct Ema {
    pub decay: f64,
    names: Vec<String>,
    vars: Vec<Var>,
    shadow: Vec<Var>,
}

impl Ema {
    /// Starts as a copy of the weights now.
    pub fn new(varmap: &VarMap, decay: f64) -> CResult<Ema> {
        let mut all: Vec<(String, Var)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        all.sort_by(|x, y| x.0.cmp(&y.0));
        let mut shadow = Vec::with_capacity(all.len());
        for (_, v) in &all {
            shadow.push(Var::from_tensor(&v.as_tensor().copy()?)?);
        }
        let (names, vars) = all.into_iter().unzip();
        Ok(Ema { decay, names, vars, shadow })
    }

    pub fn update(&self) -> CResult<()> {
        let (a, b) = (self.decay, 1.0 - self.decay);
        for (v, e) in self.vars.iter().zip(&self.shadow) {
            e.set(&e.as_tensor().affine(a, 0.0)?.add(&v.as_tensor().affine(b, 0.0)?)?)?;
        }
        Ok(())
    }

    /// Saved under the weights' own names, so the file loads as a model.
    pub fn save(&self, path: &Path) -> CResult<()> {
        let map: HashMap<String, Tensor> = self.names.iter().cloned().zip(self.shadow.iter().map(|e| e.as_tensor().clone())).collect();
        candle_core::safetensors::save(&map, path)
    }

    /// Restore a copy written by `save` (every weight must be there).
    pub fn load(&self, path: &Path, dev: &Device) -> CResult<()> {
        let map = candle_core::safetensors::load(path, dev)?;
        for (n, e) in self.names.iter().zip(&self.shadow) {
            let x = map.get(n).ok_or_else(|| candle_core::Error::Msg(format!("{} lacks {n}", path.display())))?;
            e.set(x)?;
        }
        Ok(())
    }

    /// The copy's value of the weight called `name`.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.names.iter().position(|n| n == name).map(|i| self.shadow[i].as_tensor())
    }
}

/// `dir/model.safetensors` → `dir/model.ema.safetensors`.
pub fn ema_path(model_path: &Path) -> PathBuf {
    let stem = model_path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "model".into());
    model_path.with_file_name(format!("{stem}.ema.safetensors"))
}

// ------------------------------------------------------------------------------------------- U4 reward

/// The text of the "I don't know" action (scored like a choice: with a leading space, then EOS).
pub const IDK_TEXT: &str = "I don't know";

/// The item with "I don't know" added as its last choice (the right answer stays where it was).
pub fn with_idk(it: &Item) -> Item {
    let mut x = it.clone();
    x.choices.push(IDK_TEXT.to_string());
    x
}

/// An answer's outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Right,
    Wrong,
    Abstain,
}

impl Outcome {
    /// TruthRL's three-way reward: right +1, wrong −1, "I don't know" 0.
    pub fn reward(self) -> f64 {
        match self {
            Outcome::Right => 1.0,
            Outcome::Wrong => -1.0,
            Outcome::Abstain => 0.0,
        }
    }
}

/// Her greedy answer from the scores (lower is better). `idk`: the index of the "I don't know" choice, if any (the
/// last). She abstains when her margin over the real choices is below `tau` or when "I don't know" scores best;
/// otherwise she is right only when the right choice is strictly lowest among the real ones (mc::grade_one).
pub fn judge(scores: &[f64], right: usize, tau: f64, idk: Option<usize>) -> Outcome {
    let real = idk.unwrap_or(scores.len());
    let g = grade_one(&scores[..real], right);
    if g.margin < tau {
        return Outcome::Abstain;
    }
    if let Some(k) = idk {
        if (0..real).all(|j| scores[k] < scores[j]) {
            return Outcome::Abstain;
        }
    }
    if g.right {
        Outcome::Right
    } else {
        Outcome::Wrong
    }
}

/// Each choice's weight in the REINFORCE step for one draw `c` with reward `r` and an outside baseline `b` (grow_gpu's
/// v41 rule, the same arithmetic): (r − b)(1[j = c] − π_j) · scale_j / (temp · answered).
pub fn weights_single(r: f64, b: f64, c: usize, pi: &[f64], scale: &[f64], temp: f64, answered: usize) -> Vec<f32> {
    pi.iter()
        .enumerate()
        .map(|(j, &pj)| {
            let hit = if j == c { 1.0 } else { 0.0 };
            ((r - b) * (hit - pj) * scale[j] / (temp * answered as f64)) as f32
        })
        .collect()
}

/// The group's advantages: each reward minus the group's mean.
pub fn group_advantages(rewards: &[f64]) -> Vec<f64> {
    let m = rewards.iter().sum::<f64>() / rewards.len().max(1) as f64;
    rewards.iter().map(|r| r - m).collect()
}

/// Each choice's weight for K draws `draws` with rewards `rewards` of one question, the group's mean as the baseline:
/// the exact gradient of −(1/K) Σ_k A_k log π(c_k), i.e. Σ_k A_k (1[j = c_k] − π_j) · scale_j / (K · temp · answered).
/// All zero when every draw earned the same reward.
pub fn weights_group(draws: &[usize], rewards: &[f64], pi: &[f64], scale: &[f64], temp: f64, answered: usize) -> Vec<f32> {
    let adv = group_advantages(rewards);
    let k = draws.len().max(1) as f64;
    pi.iter()
        .enumerate()
        .map(|(j, &pj)| {
            let s: f64 = draws.iter().zip(&adv).map(|(&c, &a)| a * (if c == j { 1.0 } else { 0.0 } - pj)).sum();
            (s * scale[j] / (k * temp * answered as f64)) as f32
        })
        .collect()
}

/// A tally of answers: right − wrong and the abstain rate (how the owner scores recall).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub right: usize,
    pub wrong: usize,
    pub abstain: usize,
}

impl Tally {
    pub fn add(&mut self, o: Outcome) {
        match o {
            Outcome::Right => self.right += 1,
            Outcome::Wrong => self.wrong += 1,
            Outcome::Abstain => self.abstain += 1,
        }
    }

    pub fn n(&self) -> usize {
        self.right + self.wrong + self.abstain
    }

    /// (right − wrong) / items.
    pub fn right_minus_wrong(&self) -> f64 {
        (self.right as f64 - self.wrong as f64) / self.n().max(1) as f64
    }

    pub fn abstain_rate(&self) -> f64 {
        self.abstain as f64 / self.n().max(1) as f64
    }

    /// The share right among the answers given (NaN when she gave none).
    pub fn share_right(&self) -> f64 {
        let given = self.right + self.wrong;
        if given == 0 {
            f64::NAN
        } else {
            self.right as f64 / given as f64
        }
    }
}

// ------------------------------------------------------------------------------------------- U2 KL to a reference

/// Mean over rows of KL(p_ref || p_cur) = Σ_v p_ref (log p_ref − log p_cur), from logits [N, V]. The reference side
/// is detached, so the gradient flows into `cur_logits` only: (softmax(cur) − p_ref) / N per row.
pub fn kl_ref_cur(ref_logits: &Tensor, cur_logits: &Tensor) -> CResult<Tensor> {
    let lp_ref = candle_nn::ops::log_softmax(&ref_logits.detach(), D::Minus1)?;
    let p_ref = lp_ref.exp()?;
    let lp_cur = candle_nn::ops::log_softmax(cur_logits, D::Minus1)?;
    p_ref.mul(&lp_ref.sub(&lp_cur)?)?.sum(D::Minus1)?.mean_all()
}

/// KL(reference || current) on `ids` [B, T] and its gradient for the current model (candle's autograd through
/// `Model::forward` in training mode). `ref_logits`: the frozen reference model's logits on the same ids.
pub fn kl_grads(model: &Model, ref_logits: &Tensor, ids: &Tensor) -> CResult<(f32, GradStore)> {
    let cur = model.forward(ids)?;
    let kl = kl_ref_cur(ref_logits, &cur)?;
    let mut grads = kl.backward()?;
    crate::arch::ckpt::merge(&mut grads)?;
    Ok((kl.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?, grads))
}

// ------------------------------------------------------------------------------------------- I2 context warm-up (v42 2B)

/// The settings are usable: `start` divides `full` and full / start is a power of two.
pub fn ctx_warmup_check(start: usize, full: usize) -> Result<(), String> {
    if start == 0 || start > full || full % start != 0 || !(full / start).is_power_of_two() {
        return Err(format!("--ctx-start {start}: must divide --ctx {full} by a power of two"));
    }
    Ok(())
}

/// I2 (--ctx-warmup TOKENS): the context length `seen` tokens into training. With n = log2(full / start) doublings,
/// the first `warm` tokens are cut into n equal stages of start, 2·start, ..., full / 2; from `warm` on it is `full`.
/// `warm` = 0 (off) or start >= full: always `full`.
pub fn ctx_at(seen: u64, warm: u64, start: usize, full: usize) -> usize {
    if warm == 0 || start >= full || seen >= warm {
        return full;
    }
    let n = (full / start).trailing_zeros() as u64;
    start << (seen * n / warm) as u32
}

/// Windows of `ctx` + 1 tokens cut into ctx / c windows of c + 1 tokens each: piece j is w[j·c .. j·c + c + 1] (one
/// token shared with the next piece), so the same tokens are scored as targets, in the same order, each with a
/// shorter past. The tokens per step stay the same; the batch grows by ctx / c.
pub fn split_windows(windows: &[Vec<u32>], c: usize) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    for w in windows {
        let ctx = w.len() - 1;
        if c >= ctx {
            out.push(w.clone());
            continue;
        }
        for j in 0..ctx / c {
            out.push(w[j * c..j * c + c + 1].to_vec());
        }
    }
    out
}

// ------------------------------------------------------------------------------------------- I4 Grokfast (v42 2B)

/// I4 (--grokfast-alpha A --grokfast-lambda L): Grokfast-EMA (Lee et al. 2024). Each gradient gets its own slow average
/// added before clipping and the optimizer step: e <- A·e + (1 - A)·g (e starts as the first gradient), g <- g + L·e.
/// It boosts the slow part of the gradient, which speeds up generalisation that comes late (grokking).
pub struct Grokfast {
    pub alpha: f64,
    pub lamb: f64,
    ema: HashMap<String, Var>,
}

impl Grokfast {
    pub fn new(alpha: f64, lamb: f64) -> Grokfast {
        Grokfast { alpha, lamb, ema: HashMap::new() }
    }

    /// Filter every parameter's gradient in `grads` in place.
    pub fn apply(&mut self, params: &[crate::optim::Param], grads: &mut GradStore) -> CResult<()> {
        for p in params {
            let key = p.var.as_tensor();
            let Some(g) = grads.get(key) else { continue };
            let g = g.detach();
            let e = match self.ema.get(&p.name) {
                None => {
                    let v = Var::from_tensor(&g.copy()?)?;
                    let e = v.as_tensor().clone();
                    self.ema.insert(p.name.clone(), v);
                    e
                }
                Some(v) => {
                    let e = v.as_tensor().affine(self.alpha, 0.0)?.add(&g.affine(1.0 - self.alpha, 0.0)?)?;
                    v.set(&e)?;
                    e
                }
            };
            grads.insert(key, g.add(&e.affine(self.lamb, 0.0)?)?);
        }
        Ok(())
    }

    /// The averages as gf.<param> tensors (saved with the optimizer's moments, so a resume continues them).
    pub fn state(&self) -> HashMap<String, Tensor> {
        self.ema.iter().map(|(k, v)| (format!("gf.{k}"), v.as_tensor().clone())).collect()
    }

    /// Restore the averages `state` wrote (a parameter without one starts again from its next gradient).
    pub fn load_state(&mut self, map: &HashMap<String, Tensor>, dev: &Device) -> CResult<usize> {
        let mut n = 0;
        for (k, t) in map {
            if let Some(name) = k.strip_prefix("gf.") {
                self.ema.insert(name.to_string(), Var::from_tensor(&t.to_device(dev)?.copy()?)?);
                n += 1;
            }
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Config, Kind};
    use candle_core::DType;
    use candle_nn::VarBuilder;

    fn tmp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ava_trainkit_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// X2: night 0 is the old seed; two nights with a cursor draw different window starts; a night is reproducible.
    #[test]
    fn replay_cursor_moves_the_windows_on() {
        let (seed, stream, len, ctx) = (1u64, 1u64, 1_000_000usize, 256usize);
        assert_eq!(replay_seed(seed, stream, None), derive_seed(seed, 0x7265_706c_6179, stream));
        assert_eq!(replay_seed(seed, stream, Some(0)), replay_seed(seed, stream, None));
        let dir = tmp_dir("cursor");
        let cur = dir.join("replay_cursor.txt");
        let _ = std::fs::remove_file(&cur);
        let mut starts = Vec::new();
        for _ in 0..2 {
            // one night: read the cursor, draw the night's windows, move the cursor on
            let night = read_night(&cur).unwrap();
            let mut rng = SplitMix64::new(replay_seed(seed, stream, Some(night)));
            starts.push(window_starts(&mut rng, 8, len, ctx));
            write_night(&cur, night + 1).unwrap();
        }
        assert_eq!(read_night(&cur).unwrap(), 2);
        assert_ne!(starts[0], starts[1], "two nights drew the same windows");
        assert!(starts.iter().flatten().all(|&s| s + ctx + 1 <= len));
        let mut rng = SplitMix64::new(replay_seed(seed, stream, Some(1)));
        assert_eq!(window_starts(&mut rng, 8, len, ctx), starts[1]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// J3: non-finite values and spikes over K × the running median are skipped (and counted); good norms join the
    /// median; the spike rule waits for enough history.
    #[test]
    fn bad_batch_guard() {
        let mut g = BadBatch::new(10.0, 50);
        assert_eq!(g.judge(&[f32::NAN], 1.0), Verdict::NonFinite);
        assert_eq!(g.judge(&[2.0], f64::INFINITY), Verdict::NonFinite);
        assert_eq!(g.judge(&[2.0], 1000.0), Verdict::Good, "no history yet: only the finite check");
        let mut g = BadBatch::new(10.0, 50);
        for i in 0..30 {
            assert_eq!(g.judge(&[2.0, 3.0], 1.0 + (i % 3) as f64 * 0.1), Verdict::Good);
        }
        assert_eq!(g.median(), Some(1.1));
        assert_eq!(g.judge(&[2.0], 50.0), Verdict::Spike(1.1));
        assert_eq!(g.judge(&[2.0], 10.5), Verdict::Good);
        assert_eq!(g.skipped, 1);
        assert_eq!(g.in_a_row, 0);
    }

    /// Q3: fraction 0 is the old even spread exactly; a late fraction puts every batch in the last part, one per step.
    #[test]
    fn late_answers_plan() {
        for (steps, wanted) in [(1000usize, 37usize), (500, 500), (300, 900), (7, 3), (0, 5)] {
            let p = QaPlan::new(steps, wanted, 0.0);
            let old_total = if steps == 0 { 0 } else { wanted.min(steps) };
            for s in 0..steps {
                let (lo, hi) = (s * old_total / steps.max(1), (s + 1) * old_total / steps.max(1));
                assert_eq!(p.at(s), (hi > lo).then_some(lo), "steps {steps} wanted {wanted} s {s}");
            }
            let late = QaPlan::new(steps, wanted, 0.25);
            let got: Vec<usize> = (0..steps).filter_map(|s| late.at(s)).collect();
            assert_eq!(got, (0..late.total).collect::<Vec<_>>(), "each batch once, in order");
            let first = (0..steps).find(|&s| late.at(s).is_some());
            if let Some(f) = first {
                assert!(f >= steps - (steps as f64 * 0.25).round() as usize);
            }
        }
        assert_eq!(QaPlan::new(1000, 37, 0.25).total, 37);
        assert_eq!(QaPlan::new(1000, 400, 0.25).total, 250);
    }

    /// Q4: the EMA follows e ← d·e + (1 − d)·w and saves under the weights' names.
    #[test]
    fn ema_update_save_load() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let w = vb.get_with_hints(2, "x.w1", candle_nn::Init::Const(1.0)).unwrap();
        let ema = Ema::new(&varmap, 0.9).unwrap();
        let var = varmap.data().lock().unwrap()["x.w1"].clone();
        var.set(&Tensor::from_vec(vec![3f32, -1.0], 2, &Device::Cpu).unwrap()).unwrap();
        ema.update().unwrap();
        let e = ema.get("x.w1").unwrap().to_vec1::<f32>().unwrap();
        assert!((e[0] - (0.9 + 0.3)).abs() < 1e-6 && (e[1] - (0.9 - 0.1)).abs() < 1e-6, "{e:?}");
        drop(w);
        let dir = tmp_dir("ema");
        let p = ema_path(&dir.join("model.safetensors"));
        assert_eq!(p.file_name().unwrap(), "model.ema.safetensors");
        ema.save(&p).unwrap();
        let other = Ema::new(&varmap, 0.9).unwrap();
        other.load(&p, &Device::Cpu).unwrap();
        assert_eq!(other.get("x.w1").unwrap().to_vec1::<f32>().unwrap(), e);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// U4: one draw with an outside baseline is the v41 formula; a group's weights cancel (Σ_j w_j / scale_j = 0),
    /// vanish when every draw earned the same, and push the better-rewarded choice up (a positive weight on its NLL,
    /// which the step lowers).
    #[test]
    fn reinforce_weights() {
        let pi = [0.5, 0.3, 0.2];
        let scale = [1.0, 0.5, 0.25];
        let w = weights_single(1.0, 0.2, 1, &pi, &scale, 2.0, 4);
        for j in 0..3 {
            let hit = if j == 1 { 1.0 } else { 0.0 };
            assert_eq!(w[j], ((1.0f64 - 0.2) * (hit - pi[j]) * scale[j] / (2.0 * 4.0)) as f32);
        }
        assert_eq!(group_advantages(&[1.0, -1.0, 0.0, 1.0]), vec![0.75, -1.25, -0.25, 0.75]);
        let g = weights_group(&[0, 1, 2, 0], &[1.0, -1.0, 0.0, 1.0], &pi, &[1.0; 3], 1.0, 1);
        assert!(g.iter().map(|&x| x as f64).sum::<f64>().abs() < 1e-6);
        // the weight multiplies the choice's NLL in the loss the step lowers: the rewarded choice 0 gets a positive
        // weight, the punished choice 1 a negative one
        assert!(g[0] > 0.0 && g[1] < 0.0, "{g:?}");
        assert_eq!(weights_group(&[0, 1, 2], &[-1.0; 3], &pi, &scale, 1.0, 3), vec![0.0; 3]);
    }

    /// U4/U5: the greedy judge with "I don't know" and tau; the tally's right − wrong and abstain rate.
    #[test]
    fn judge_and_tally() {
        assert_eq!(judge(&[1.0, 2.0, 3.0], 0, 0.0, None), Outcome::Right);
        assert_eq!(judge(&[1.0, 2.0, 3.0], 1, 0.0, None), Outcome::Wrong);
        assert_eq!(judge(&[1.0, 1.5, 3.0], 0, 1.0, None), Outcome::Abstain);
        assert_eq!(judge(&[1.0, 2.0, 0.5], 0, 0.0, Some(2)), Outcome::Abstain);
        assert_eq!(judge(&[1.0, 2.0, 1.5], 1, 0.0, Some(2)), Outcome::Wrong);
        let it = Item { id: "x".into(), kind: "k".into(), bin: "".into(), prompt: "p".into(), choices: vec!["a".into(), "b".into()], right: 1 };
        let x = with_idk(&it);
        assert_eq!((x.choices.len(), x.right, x.choices[2].as_str()), (3, 1, IDK_TEXT));
        let mut t = Tally::default();
        for o in [Outcome::Right, Outcome::Right, Outcome::Wrong, Outcome::Abstain] {
            t.add(o);
        }
        assert_eq!(t.n(), 4);
        assert!((t.right_minus_wrong() - 0.25).abs() < 1e-12);
        assert!((t.abstain_rate() - 0.25).abs() < 1e-12);
        assert!((t.share_right() - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(Outcome::Abstain.reward(), 0.0);
    }

    /// U2: KL is 0 against itself, matches the sum by hand, and its gradient is (softmax(cur) − p_ref) / N.
    #[test]
    fn kl_value_and_gradient() {
        let dev = Device::Cpu;
        let r = Tensor::from_vec(vec![0.1f32, 1.0, -0.5, 2.0, 0.0, 0.3], (2, 3), &dev).unwrap();
        let c = Var::from_vec(vec![0.4f32, 0.2, 0.1, 1.0, 1.0, -1.0], (2, 3), &dev).unwrap();
        assert!(kl_ref_cur(&r, &r).unwrap().to_scalar::<f32>().unwrap().abs() < 1e-6);
        let kl = kl_ref_cur(&r, c.as_tensor()).unwrap();
        let sm = |v: &[f32]| -> Vec<f64> {
            let m = v.iter().cloned().fold(f32::MIN, f32::max) as f64;
            let z: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
            v.iter().map(|&x| (x as f64 - m).exp() / z).collect()
        };
        let (rv, cv) = (r.to_vec2::<f32>().unwrap(), c.as_tensor().to_vec2::<f32>().unwrap());
        let mut want = 0.0;
        for i in 0..2 {
            let (p, q) = (sm(&rv[i]), sm(&cv[i]));
            want += p.iter().zip(&q).map(|(a, b)| a * (a / b).ln()).sum::<f64>() / 2.0;
        }
        let got = kl.to_scalar::<f32>().unwrap() as f64;
        assert!(got > 0.0 && (got - want).abs() < 1e-5, "{got} vs {want}");
        let g = kl.backward().unwrap();
        let gc = g.get(c.as_tensor()).unwrap().to_vec2::<f32>().unwrap();
        for i in 0..2 {
            let (p, q) = (sm(&rv[i]), sm(&cv[i]));
            for j in 0..3 {
                let want = (q[j] - p[j]) / 2.0;
                assert!((gc[i][j] as f64 - want).abs() < 1e-5, "row {i} col {j}: {} vs {want}", gc[i][j]);
            }
        }
    }

    /// U2 on a small D model on the CPU: against its own logits the KL is 0 and the gradient vanishes.
    #[test]
    fn kl_grads_on_a_model() {
        let dev = Device::Cpu;
        let cfg = Config::with_kind(Kind::D, 64, 64, 2, 16);
        let varmap = VarMap::new();
        let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &dev)).unwrap();
        model.set_track_activity(false);
        let ids = Tensor::from_vec((0..32u32).map(|i| (i * 7) % 64).collect::<Vec<_>>(), (2, 16), &dev).unwrap();
        let ref_logits = model.forward(&ids).unwrap().detach();
        let (kl, g) = kl_grads(&model, &ref_logits, &ids).unwrap();
        assert!(kl.abs() < 1e-5, "{kl}");
        let tok = varmap.data().lock().unwrap()[crate::model::TOK_EMB].clone();
        let gt = g.get(tok.as_tensor()).expect("a gradient for the embedding");
        let mx = gt.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(mx < 1e-5, "{mx}");
    }

    /// I2: off means always the full context; on, the context doubles in equal stages and is full from `warm` on.
    #[test]
    fn ctx_warmup_stages() {
        for s in [0u64, 5, 1 << 40] {
            assert_eq!(ctx_at(s, 0, 256, 2048), 2048, "off");
        }
        assert_eq!(ctx_at(0, 100, 2048, 2048), 2048);
        // 256 -> 2048: three doublings, so stages 256, 512, 1024 over the first 300 tokens
        let want = [(0u64, 256usize), (99, 256), (100, 512), (199, 512), (200, 1024), (299, 1024), (300, 2048), (10_000, 2048)];
        for (seen, c) in want {
            assert_eq!(ctx_at(seen, 300, 256, 2048), c, "{seen}");
        }
        assert!(ctx_warmup_check(256, 2048).is_ok());
        assert!(ctx_warmup_check(256, 1536).is_err());
        assert!(ctx_warmup_check(300, 2400).is_ok(), "only the ratio must be a power of two");
        assert!(ctx_warmup_check(4096, 2048).is_err());
    }

    /// I2: cut windows keep every target in order, each with the same next token, and the token count per step.
    #[test]
    fn split_windows_keep_the_targets() {
        let w: Vec<Vec<u32>> = vec![(0..9).collect(), (100..109).collect()];
        let cut = split_windows(&w, 2);
        assert_eq!(cut.len(), 8);
        assert!(cut.iter().all(|p| p.len() == 3));
        let pairs = |ws: &[Vec<u32>]| -> Vec<(u32, u32)> { ws.iter().flat_map(|p| p.windows(2).map(|x| (x[0], x[1])).collect::<Vec<_>>()).collect() };
        assert_eq!(pairs(&cut), pairs(&w));
        assert_eq!(split_windows(&w, 8), w);
        assert_eq!(split_windows(&w, 64), w);
    }

    /// I4: the filtered gradient is g + L·e, e = A·e + (1 - A)·g with e = g at the first step; the state saves and
    /// restores; lambda 0 changes nothing.
    #[test]
    fn grokfast_ema_filter() {
        let var = Var::from_vec(vec![1.0f32, -2.0], 2, &Device::Cpu).unwrap();
        let params = vec![crate::optim::Param { name: "w".into(), var: var.clone(), decay: false }];
        let grads = |c: [f32; 2]| var.as_tensor().mul(&Tensor::new(&c, &Device::Cpu).unwrap()).unwrap().sum_all().unwrap().backward().unwrap();
        let mut gf = Grokfast::new(0.9, 2.0);
        let mut g1 = grads([1.0, 2.0]);
        gf.apply(&params, &mut g1).unwrap();
        assert_eq!(g1.get(var.as_tensor()).unwrap().to_vec1::<f32>().unwrap(), vec![3.0, 6.0]);
        let mut g2 = grads([3.0, 0.0]);
        gf.apply(&params, &mut g2).unwrap();
        // e = 0.9·(1, 2) + 0.1·(3, 0) = (1.2, 1.8); g = (3, 0) + 2·e = (5.4, 3.6)
        let got = g2.get(var.as_tensor()).unwrap().to_vec1::<f32>().unwrap();
        assert!((got[0] - 5.4).abs() < 1e-5 && (got[1] - 3.6).abs() < 1e-5, "{got:?}");
        let st = gf.state();
        assert_eq!(st.keys().cloned().collect::<Vec<_>>(), vec!["gf.w".to_string()]);
        let mut back = Grokfast::new(0.9, 2.0);
        assert_eq!(back.load_state(&st, &Device::Cpu).unwrap(), 1);
        let (mut ga, mut gb) = (grads([1.0, 1.0]), grads([1.0, 1.0]));
        gf.apply(&params, &mut ga).unwrap();
        back.apply(&params, &mut gb).unwrap();
        assert_eq!(ga.get(var.as_tensor()).unwrap().to_vec1::<f32>().unwrap(), gb.get(var.as_tensor()).unwrap().to_vec1::<f32>().unwrap());
        let mut off = Grokfast::new(0.9, 0.0);
        let mut g3 = grads([0.5, -0.25]);
        off.apply(&params, &mut g3).unwrap();
        assert_eq!(g3.get(var.as_tensor()).unwrap().to_vec1::<f32>().unwrap(), vec![0.5, -0.25]);
    }
}
