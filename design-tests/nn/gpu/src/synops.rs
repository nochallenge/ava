// TESTING / DESIGN ONLY — synops (v42 C14): synaptic operations per token, the energy bill a brain would pay.
//
// In a brain a neuron that does not fire costs its targets nothing. The rule is the one of
// design-tests/nn/parts/src/event.rs: a non-zero input to a matrix [d_in, d_out] costs d_out synaptic operations (one
// row added), so the product costs events × fan-out instead of d_in × d_out multiply-adds. Two of Ava's matrices take
// sparse input: Wout [d, d] reads the LIF spikes, and W2 [4d, d] reads the MLP's hidden layer after kWTA and ReLU (k of
// 4d winners, fewer when a winner is not positive). Every other matrix reads a dense vector (normed inputs, the memory's
// projections, the unembedding) and stays dense, as flops.rs counts it. The events come from Model::forward_trace (exact
// non-zeros, so a graded spike is one event), over real positions only (PAD inputs are not counted).
//
// Per token: spikes and active hidden units per block, the synaptic ops of the two sparse matrices, what those two
// cost dense, and the whole forward pass in multiply-adds, dense (flops::per_token / 2) and event-driven (the dense
// total − the two matrices' dense cost + their synaptic ops).
//
// v49: models with block options (src/arch.rs, `cfg.arch.needs_path()`, e.g. Ava 0.4) are counted by the rules below;
// every other model is counted exactly as before (the same numbers and the same report, line for line). The events
// still come from forward_trace (now through forward_arch's own blocks, so they are what that forward computed):
//   spikes (--spikes ternary|int2)  an event is a non-zero spike, whatever its size: a ±2 counts ONCE (one event
//            carrying a signed 2-bit payload; the target adds its Wout row times ±1 or ±2, a sign and a shift, not a
//            multiply), so Wout costs (non-zero spikes) × d. The ±2s are reported on their own line.
//   ALIF (--alif), homeostasis, --lif-init, --lif-scale, --surrogate-slope  they only change when a neuron fires, so
//            they are counted through the spikes; the adaptation trace and threshold (a few multiply-adds per neuron)
//            are elementwise, left out like the LIF scan itself (flops.rs). The homeostasis nudge runs after optimizer
//            steps only, never in a forward.
//   kWTA (--kwta 1/4)  only the kept units can fire: W2's events are the non-zero units after the mask (k = hidden/4
//            at most per token, so up to twice today's 1/8); the selection itself is elementwise.
//   relu2 (--mlp-act relu2)  event-driven into W2, from the code: a = relu(u ⊙ mask)² is exactly 0 off the kWTA
//            winners (and wherever relu is 0), so W2 reads a vector with at most k non-zeros, like today's relu; each
//            one is graded (a multiply-add per output, as today's relu units were). W1 reads the dense normed stream:
//            dense, d × hidden. The square is elementwise on k numbers (left out).
//   value residual (--value-residual)  dense: v ← (1 + p0) v + p1 v0 in every memory block after the first, 2·d
//            multiply-adds per token per block.
//   dense residual (--residual dense)  dense: after block i the stream is x + sum_j w_ij x_j over i + 2 streams,
//            (i + 2)·d multiply-adds per token. (--residual attn: flops.rs's 2·d·(i + 2).)
//   channel gates (--gates channel)  dense: the per-channel forget and write gates h·Wa1·Wa2 and h·Wb1·Wb2 (4·d·32) and
//            the erase key's h·We (d²), on top of the per-head gates h·Wa, h·Wb (2·d·heads). KDA (--gates kda): h·Wa1·Wa2.
//   head size (--head-size 64)  heads = d / head size, in the gates and the memory core.
//   memory core  dense (the fast weights F and the keys are dense): per head and token, the rule mem_gen runs
//            (src/mem_gen.rs; delta_rec on the CPU): r = eᵀF and y = qᵀF (2·hs²), the write (b·k)vᵀ and the erase
//            (α·b·e)rᵀ (one rank-1 update, hs², when e = k as with per-head gates and KDA; two, 2·hs², with channel
//            gates' separate erase key), and with momentum (--mem-momentum) S ← M·S + U and F ← αF + S (2·hs² more).
//            Ava 0.4: 6·hs² per head. Today's rule (per-head gates, no momentum) keeps flops.rs's --mem forms, at the
//            model's head size. The CPU chunk form of the options (delta_chunk) does more work for parallelism; it is
//            not what is counted.
//   engram (--engram L)  dense: Wq, Wk, Wv, Wo (4·d²), the look-back 2·d multiply-adds per key read (q·k and the
//            weighted value), the keys being every position up to and including the token: the mean over the counted
//            positions (not cfg.ctx: flops.rs's old engram term averages over the model's context, 4096 when loaded
//            by load_auto). The causal look-up reads every earlier key, so none of it is event-driven. --eng-gate:
//            flops.rs's h·W_th.
//   glance (--glance)  the place embedding is a table look-up added to the input: free, like the token embedding.
//   mag (--mag)  the magnitude code [MAG_F = 14] times mag_w [MAG_F, d], computed for every token: dense MAG_F·d;
//            event-driven, only its non-zero features (number tokens only) cost d each.
//   --bonus, --token-shift, --gate sigmoid  flops.rs's terms. --feedback: one d² projection per 64-token chunk.
//   --resid-fix, --timescales  init only. The order head (--top-weight) is used in training only.
// The Wave 2C/5 parts (src/memory.rs, src/wave5.rs: dendrites, delays, stp, wm, the product-key memory, the engram's
// options, the modulators' shadow pass, ...) are not counted: `check` refuses them (their training-only losses pass).

use crate::arch::{EngGate, Gating, OutGate, FB_CHUNK, GATE_RANK};
use crate::flops::{per_token, MemForm};
use crate::model::{Config, Kind, Trace, HEAD_SIZE};
use candle_core::{DType, Tensor};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SynCount {
    pub tokens: u64,
    /// per block: LIF spikes, and active hidden units after kWTA and ReLU, summed over the counted positions
    pub spikes: Vec<u64>,
    pub active: Vec<u64>,
    pub d: usize,
    pub hidden: usize,
    /// v49: counted by the block options' rules (`cfg.arch.needs_path()`); the three counts below only then
    pub arch: bool,
    /// per block: the spikes of size 2 (int2), each also one event in `spikes`
    pub spikes2: Vec<u64>,
    /// the magnitude code's non-zero features (each feeds one row of mag_w [MAG_F, d])
    pub mag: u64,
    /// the keys the engram reads: position + 1 summed over the counted positions
    pub keys: u64,
}

/// Non-zero entries of `x` [N, k] in the rows where `keep` [N] is 1.
fn nonzero(x: &Tensor, keep: &Tensor) -> candle_core::Result<u64> {
    let per_row = x.ne(0f32)?.to_dtype(DType::F64)?.sum(1)?; // [N]
    Ok(per_row.mul(&keep.to_dtype(DType::F64)?)?.sum_all()?.to_scalar::<f64>()?.round() as u64)
}

/// The Wave 2C/5 options whose forward cost is not counted (the training-only ones are left out of the check).
pub fn check(cfg: &Config) -> Result<(), String> {
    if !cfg.arch.needs_path() {
        return Ok(());
    }
    let m = cfg.arch.mem;
    let counted = crate::memory::MemOpts {
        mem_reset: 0,
        recall_loss: 0.0,
        jepa: 0.0,
        diffusion: 0.0,
        critic: 0.0,
        confidence: 0.0,
        topo: 0.0,
        fok_qa: false,
        reflex: 0.0,
        fok_push: 0.0,
        mlp_mult: 0,
        ..m
    };
    if !counted.is_default() || cfg.kind != Kind::D {
        return Err(format!("synops: the parts of {:?} are not counted (src/synops.rs counts D with src/arch.rs's block options)", m));
    }
    Ok(())
}

impl SynCount {
    pub fn new(cfg: &Config) -> SynCount {
        let arch = cfg.arch.needs_path();
        SynCount {
            tokens: 0,
            spikes: vec![0; cfg.layers],
            active: vec![0; cfg.layers],
            d: cfg.d,
            hidden: cfg.hidden(),
            arch,
            spikes2: if arch { vec![0; cfg.layers] } else { Vec::new() },
            mag: 0,
            keys: 0,
        }
    }

    /// Counts one traced forward pass; `keep` [N] (f32, 1 = a real position) in the trace's row order.
    pub fn add(&mut self, tr: &Trace, keep: &Tensor) -> candle_core::Result<()> {
        if tr.layers.len() != self.spikes.len() {
            candle_core::bail!("synops: {} blocks traced, {} expected", tr.layers.len(), self.spikes.len());
        }
        for (l, lt) in tr.layers.iter().enumerate() {
            self.spikes[l] += nonzero(&lt.spikes, keep)?;
            self.active[l] += nonzero(&lt.hidden, keep)?;
        }
        self.tokens += keep.to_dtype(DType::F64)?.sum_all()?.to_scalar::<f64>()?.round() as u64;
        if self.arch {
            for (l, lt) in tr.layers.iter().enumerate() {
                self.spikes2[l] += nonzero(&lt.spikes.abs()?.ge(1.5f32)?.to_dtype(DType::F32)?, keep)?;
            }
            if let Some(m) = &tr.mag {
                self.mag += nonzero(m, keep)?;
            }
            // the engram at position p of its row reads p + 1 keys (rows are batch-major, T positions each)
            let t = tr.dims.1.max(1);
            let k: Vec<f32> = keep.to_dtype(DType::F32)?.to_vec1()?;
            self.keys += k.iter().enumerate().filter(|(_, &v)| v != 0.0).map(|(i, _)| (i % t + 1) as u64).sum::<u64>();
        }
        Ok(())
    }

    /// Synaptic ops per token of the two sparse matrices (both fan out to d).
    pub fn synops_per_token(&self) -> f64 {
        let ev: u64 = self.spikes.iter().sum::<u64>() + self.active.iter().sum::<u64>();
        (ev * self.d as u64) as f64 / self.tokens.max(1) as f64
    }

    /// What the two sparse matrices cost per token if computed dense (multiply-adds).
    pub fn sparse_dense_macs(&self) -> f64 {
        (self.spikes.len() * (self.d * self.d + self.hidden * self.d)) as f64
    }

    /// The report: one line per block, then the totals per token.
    pub fn report(&self, cfg: &Config, mem: MemForm) -> String {
        let n = self.tokens.max(1) as f64;
        let mut s = String::from("block\tspikes_per_token\tspike_rate\tactive_per_token\tactive_rate\n");
        for l in 0..self.spikes.len() {
            let (sp, ac) = (self.spikes[l] as f64 / n, self.active[l] as f64 / n);
            s += &format!("{l}\t{sp:.2}\t{:.4}\t{ac:.2}\t{:.4}\n", sp / self.d as f64, ac / self.hidden as f64);
        }
        if self.arch {
            return s + &self.report_arch(cfg, mem);
        }
        let dense = per_token(cfg, mem).forward() / 2.0;
        let syn = self.synops_per_token();
        let sparse_dense = self.sparse_dense_macs();
        let event = dense - sparse_dense + syn;
        s += &format!("tokens\t{}\n", self.tokens);
        s += &format!("synops_per_token\t{syn:.0}\t(spikes x {} into Wout, active units x {} into W2)\n", self.d, self.d);
        s += &format!("those_two_dense_macs\t{sparse_dense:.0}\t(event-driven: {:.1}% of it)\n", 100.0 * syn / sparse_dense.max(1.0));
        s += &format!("forward_macs_dense\t{dense:.0}\n");
        s += &format!("forward_macs_event\t{event:.0}\t({:.1}% of dense)\n", 100.0 * event / dense.max(1.0));
        s
    }

    /// v49: the mean keys the engram reads per counted token.
    pub fn keys_per_token(&self) -> f64 {
        self.keys as f64 / self.tokens.max(1) as f64
    }

    /// v49: the totals of a model with block options (the first five lines as today's report, then the details).
    fn report_arch(&self, cfg: &Config, mem: MemForm) -> String {
        let n = self.tokens.max(1) as f64;
        let parts = arch_parts(cfg, mem, self.keys_per_token());
        let dense: f64 = parts.iter().map(|p| p.1).sum();
        let syn = self.synops_per_token();
        let sparse_dense = self.sparse_dense_macs();
        let (mag_dense, mag_syn) = (if cfg.mag { (crate::mag::MAG_F * self.d) as f64 } else { 0.0 }, self.mag as f64 * self.d as f64 / n);
        let event = dense - sparse_dense - mag_dense + syn + mag_syn;
        let sp: u64 = self.spikes.iter().sum();
        let sp2: u64 = self.spikes2.iter().sum();
        let mut s = format!("tokens\t{}\n", self.tokens);
        s += &format!("synops_per_token\t{syn:.0}\t(spikes x {} into Wout, active units x {} into W2)\n", self.d, self.d);
        s += &format!("those_two_dense_macs\t{sparse_dense:.0}\t(event-driven: {:.1}% of it)\n", 100.0 * syn / sparse_dense.max(1.0));
        s += &format!("forward_macs_dense\t{dense:.0}\n");
        s += &format!("forward_macs_event\t{event:.0}\t({:.1}% of dense)\n", 100.0 * event / dense.max(1.0));
        s += &format!("rules\tblock options{} (src/synops.rs): every non-zero spike one event (a ±2 once)\n", cfg.arch.signature());
        s += &format!("spikes_of_size_2_per_token\t{:.2}\t({:.1}% of the spikes)\n", sp2 as f64 / n, 100.0 * sp2 as f64 / sp.max(1) as f64);
        if cfg.mag {
            s += &format!("mag_synops_per_token\t{mag_syn:.0}\t(non-zero magnitude features x {} into mag_w; dense {mag_dense:.0})\n", self.d);
        }
        if cfg.engram.is_some() {
            s += &format!("engram_keys_per_token\t{:.1}\t(the mean look-back of the counted positions)\n", self.keys_per_token());
        }
        for (name, v) in &parts {
            s += &format!("dense_part\t{name}\t{v:.0}\n");
        }
        s += &format!("flops_rs_forward_macs\t{:.0}\t(flops.rs: today's block rules, look-back over ctx {}; for comparison only)\n", per_token(cfg, mem).forward() / 2.0, cfg.ctx);
        s
    }
}

/// v49: a block-options model's forward multiply-adds per token, by part (dense; the sparse matrices at their dense
/// cost too), by the rules at the top of this file. `keys`: the mean keys the engram reads per token.
pub fn arch_parts(cfg: &Config, mem: MemForm, keys: f64) -> Vec<(&'static str, f64)> {
    let ar = cfg.arch;
    let (d, l, v, hid) = (cfg.d as f64, cfg.layers, cfg.vocab as f64, cfg.hidden() as f64);
    let (hs, nh) = (ar.head_size as f64, (cfg.d / ar.head_size) as f64);
    let dd = d * d;
    let r = GATE_RANK as f64;
    let lf = l as f64;
    let mut p: Vec<(&'static str, f64)> = Vec::new();
    p.push(("lif_win", lf * dd));
    p.push(("lif_wout_sparse", lf * dd));
    p.push(("mem_wq_wk_wv_wo", lf * 4.0 * dd));
    p.push((
        "mem_out_gate",
        lf * match ar.gate {
            OutGate::Silu => dd,
            OutGate::Sigmoid => 2.0 * d * ar.gate_rank as f64,
        },
    ));
    p.push(("mem_head_gates", lf * 2.0 * d * nh));
    match ar.gates {
        Gating::Head => {}
        Gating::Channel => p.push(("mem_channel_gates_and_erase_key", lf * (4.0 * d * r + dd))),
        Gating::Kda => p.push(("mem_kda_fade", lf * 2.0 * d * r)),
    }
    let momentum = ar.mem_momentum != 0.0;
    let core = if ar.gates == Gating::Head && !momentum {
        // today's rule at this head size: flops.rs's delta_core forms (FLOPs / 2) with hs for its 32
        nh * match mem {
            MemForm::PerToken => 6.0 * hs * hs,
            MemForm::Chunk(c) => 6.0 * hs * hs + (5.0 * c as f64 - 1.0) * hs,
        } / 2.0
    } else {
        let ups = if ar.gates == Gating::Channel { 2.0 } else { 1.0 };
        nh * hs * hs * (2.0 + ups + if momentum { 2.0 } else { 0.0 })
    };
    p.push(("mem_core", lf * core));
    if ar.value_residual {
        p.push(("value_residual", (l.saturating_sub(1)) as f64 * 2.0 * d));
    }
    if ar.dense {
        p.push(("dense_residual", (0..l).map(|i| (i + 2) as f64 * d).sum()));
    }
    if ar.attn_res {
        p.push(("attn_residual", (0..l).map(|i| 2.0 * (i + 2) as f64 * d).sum()));
    }
    if ar.bonus {
        p.push(("bonus", lf * 2.0 * d));
    }
    if ar.token_shift {
        p.push(("token_shift", lf * 4.0 * d));
    }
    p.push(("mlp_w1", lf * d * hid));
    p.push(("mlp_w2_sparse", lf * hid * d));
    if cfg.engram.is_some() {
        p.push(("engram_wq_wk_wv_wo", 4.0 * dd));
        p.push(("engram_look_back", 2.0 * d * keys));
        match ar.eng_gate {
            EngGate::Off => {}
            EngGate::Head => p.push(("engram_gate", d * (cfg.d / HEAD_SIZE) as f64)),
            EngGate::Elem => p.push(("engram_gate", dd)),
        }
    }
    if ar.feedback {
        p.push(("feedback", dd / FB_CHUNK as f64));
    }
    if cfg.mag {
        p.push(("mag_code", (crate::mag::MAG_F as f64) * d));
    }
    p.push(("unembed", d * v));
    p
}

#[cfg(test)]
mod arch_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Kind;
    use crate::testkit::tiny;
    use ava_data::bpe::PAD;
    use candle_core::Device;

    /// parts/src/event.rs's rule, copied: (events, synops) of s [n, d_in] · W [d_in, d_out].
    fn event_rule(s: &[f32], d_out: usize) -> (usize, usize) {
        let ev = s.iter().filter(|&&v| v != 0.0).count();
        (ev, ev * d_out)
    }

    #[test]
    fn counts_every_event_once_and_skips_padding() {
        let (cfg, _vm, model) = tiny(Kind::D, 31);
        let dev = Device::Cpu;
        let row: Vec<u32> = b"spikes cost only when they fire".iter().map(|&b| b as u32).collect();
        let t = row.len();
        // row 0 real, row 1 the first 10 tokens then PAD
        let mut ids = row.clone();
        ids.extend_from_slice(&row[..10]);
        ids.extend(std::iter::repeat(PAD).take(t - 10));
        let keep: Vec<f32> = (0..2 * t).map(|i| if i < t + 10 { 1.0 } else { 0.0 }).collect();
        let (_, tr) = model.forward_trace(&Tensor::from_vec(ids, (2, t), &dev).unwrap()).unwrap();
        let mut c = SynCount::new(&cfg);
        c.add(&tr, &Tensor::from_vec(keep, 2 * t, &dev).unwrap()).unwrap();
        assert_eq!(c.tokens, (t + 10) as u64);
        // the same events as the parts crate's rule on the host copies of the kept rows
        let mut syn = 0usize;
        for lt in &tr.layers {
            let s: Vec<f32> = lt.spikes.narrow(0, 0, t + 10).unwrap().flatten_all().unwrap().to_vec1().unwrap();
            let h: Vec<f32> = lt.hidden.narrow(0, 0, t + 10).unwrap().flatten_all().unwrap().to_vec1().unwrap();
            syn += event_rule(&s, cfg.d).1 + event_rule(&h, cfg.d).1;
        }
        assert!(syn > 0, "a tiny model still fires");
        assert!((c.synops_per_token() - syn as f64 / (t + 10) as f64).abs() < 1e-9);
        // at most k winners per position
        assert!(c.active.iter().all(|&a| a <= ((t + 10) * cfg.kwta_k()) as u64));
        let r = c.report(&cfg, crate::flops::DEFAULT_MEM);
        assert!(r.contains("forward_macs_event") && r.lines().count() == 1 + cfg.layers + 5);
        // the padded row's first 10 positions count what the same 10 tokens alone count (causal; the memory's chunk
        // length differs with the row length, so a unit sitting exactly on its threshold may round the other way)
        let (_, tr10) = model.forward_trace(&Tensor::from_vec(row[..10].to_vec(), (1, 10), &dev).unwrap()).unwrap();
        let mut c10 = SynCount::new(&cfg);
        c10.add(&tr10, &Tensor::ones(10, DType::F32, &dev).unwrap()).unwrap();
        let mut c_pad = SynCount::new(&cfg);
        let keep2: Vec<f32> = (0..2 * t).map(|i| if i >= t && i < t + 10 { 1.0 } else { 0.0 }).collect();
        c_pad.add(&tr, &Tensor::from_vec(keep2, 2 * t, &dev).unwrap()).unwrap();
        assert_eq!(c_pad.tokens, 10);
        for l in 0..cfg.layers {
            assert!(c10.spikes[l].abs_diff(c_pad.spikes[l]) <= 2, "block {l}: {:?} vs {:?}", c10.spikes, c_pad.spikes);
            assert!(c10.active[l].abs_diff(c_pad.active[l]) <= 2, "block {l}: {:?} vs {:?}", c10.active, c_pad.active);
        }    }
}
