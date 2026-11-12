// TESTING / DESIGN ONLY — flops (v42): real FLOPs per token for each model kind, so families are compared by the
// compute they spend and not only by their weight count.
//
// Counted from the shapes in model.rs (a multiply-add is 2 FLOPs; matrices are [d_in, d_out] applied as h·W):
//   unembed  2·d·V        the tied output projection onto the V-word vocabulary (the embedding lookup is free)
//   dense    2·(matrix entries used per token):
//              S block   Win, Wout (2d²) + MLP W1, W2 (8d²)
//              T block   Wq, Wk, Wv, Wo (4d²) + MLP (8d²)
//              F block   S + Wq, Wk, Wv, Wo;  G adds Wa, Wb (2·d·H);  R and D add Wg (d²) too
//              A block   D's, except every 4th from the last: S's spiking part + attention's Wq, Wk, Wv, Wo + MLP
//              engram    Wq, Wk, Wv, Wo (4d²)
//   mixer    the parts that look back over the window, per token, averaged over the ctx positions of a window:
//              causal attention (T, A's attention blocks, the engram) and the parallel fast-weight read of F, G and R
//              (fused::gla): scores q·k and the weighted sum of values over i+1 earlier tokens, 4·d·(i+1), whose mean
//              over i = 0..ctx−1 is 2·d·(ctx+1);
//              D's gated delta rule per head of size hs = 32 (H = d / 32 heads):
//                per-token recurrence (fused::delta, hebb_fwd): 3 hs² multiply-adds per token (k·F for the erase, the
//                rank-1 write, q·F for the read) = 6·hs² FLOPs;
//                chunk form of c tokens (fused::hebc_*, lif.cu hc_prep/hc_scan/hc_out, c = HEBB_CK = 16), per chunk:
//                A = K Kᵀ below the diagonal and the two forward substitutions for U and W (3·c(c−1)·hs), Delta = U − W F
//                and the state update (4·c·hs²), the read γ Fᵀ q (2·c·hs²) and the in-chunk read (2·c(c+1)·hs):
//                6·hs² + (5c − 1)·hs FLOPs per token.
// Elementwise work (norms, the LIF scan, kWTA selection, gates, softmax) is left out: it is O(d) per token and under
// 1% at these sizes. The sparse MLP is counted dense, as the GPU runs it.
// Training = 3 × forward (the backward pass of a matmul is two matmuls of the same size). Recomputation inside fused
// backward kernels is not counted (the usual "model FLOPs" convention).

use crate::model::{Config, Kind, HEAD_SIZE};

/// How D's fast-weight core is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemForm {
    /// one token after another (fused::delta, hebb_fwd)
    PerToken,
    /// the chunk form with chunks of c tokens (fused::hebc_*)
    Chunk(usize),
}

/// The chunk form with lif.cu's chunk length (HEBB_CK = 16).
pub const DEFAULT_MEM: MemForm = MemForm::Chunk(crate::fused::HEBB_CK);

impl MemForm {
    pub fn parse(s: &str) -> Result<MemForm, String> {
        match s {
            "token" | "per-token" => Ok(MemForm::PerToken),
            _ => match s.strip_prefix("chunk").map(|c| c.parse::<usize>()) {
                Some(Ok(c)) if c > 0 => Ok(MemForm::Chunk(c)),
                _ => Err(format!("unknown memory form {s:?} (token, chunk16, ...)")),
            },
        }
    }
}

/// Forward FLOPs per token, by part.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Flops {
    pub unembed: f64,
    pub dense: f64,
    pub mixer: f64,
}

impl Flops {
    pub fn forward(&self) -> f64 {
        self.unembed + self.dense + self.mixer
    }

    /// Forward and backward.
    pub fn train(&self) -> f64 {
        3.0 * self.forward()
    }
}

/// Causal look-back (attention or a parallel fast-weight read) averaged over a window of ctx positions.
fn causal_core(d: f64, ctx: f64) -> f64 {
    2.0 * d * (ctx + 1.0)
}

/// D's delta-rule core per token over all heads.
pub fn delta_core(d: usize, mem: MemForm) -> f64 {
    let (h, hs) = ((d / HEAD_SIZE) as f64, HEAD_SIZE as f64);
    match mem {
        MemForm::PerToken => h * 6.0 * hs * hs,
        MemForm::Chunk(c) => h * (6.0 * hs * hs + (5.0 * c as f64 - 1.0) * hs),
    }
}

/// Forward FLOPs per token of a model with this config, trained on windows of cfg.ctx tokens.
pub fn per_token(cfg: &Config, mem: MemForm) -> Flops {
    let (d, v, ctx) = (cfg.d as f64, cfg.vocab as f64, cfg.ctx as f64);
    let h = (cfg.d / HEAD_SIZE) as f64;
    let dd = d * d;
    let (spike, mlp, qkvo, gates) = (2.0 * dd, 8.0 * dd, 4.0 * dd, 2.0 * d * h);
    let (mut w, mut mixer) = (0.0f64, 0.0f64);
    for i in 0..cfg.layers {
        match cfg.kind {
            Kind::S => w += spike + mlp,
            Kind::T => {
                w += qkvo + mlp;
                mixer += causal_core(d, ctx);
            }
            Kind::F => {
                w += spike + qkvo + mlp;
                mixer += causal_core(d, ctx);
            }
            Kind::G => {
                w += spike + qkvo + gates + mlp;
                mixer += causal_core(d, ctx);
            }
            Kind::R => {
                w += spike + qkvo + dd + gates + mlp;
                mixer += causal_core(d, ctx);
            }
            Kind::D => {
                // v44 K2: the low-rank sigmoid gate uses 2·d·R entries instead of Wg's d·d
                let og = match cfg.arch.gate {
                    crate::arch::OutGate::Silu => dd,
                    crate::arch::OutGate::Sigmoid => 2.0 * d * cfg.arch.gate_rank as f64,
                };
                w += spike + qkvo + og + gates + mlp;
                if cfg.arch.attn_res {
                    // v44 K1: after block i, a score and a weighted add per source (i + 2 sources), d each
                    w += 2.0 * d * (i + 2) as f64;
                }
                // v45: W1's per-channel fade (h·Wa1·Wa2), W2's bonus (a product, a sum and a scaled add per number),
                // W3's four mixes (a multiply-add per number each)
                if cfg.arch.gates == crate::arch::Gating::Kda {
                    w += 2.0 * d * crate::arch::GATE_RANK as f64;
                }
                if cfg.arch.bonus {
                    w += 2.0 * d;
                }
                if cfg.arch.token_shift {
                    w += 4.0 * d;
                }
                mixer += delta_core(cfg.d, mem);
            }
            Kind::A if cfg.attn_layer(i) => {
                w += spike + qkvo + mlp;
                mixer += causal_core(d, ctx);
            }
            Kind::A => {
                w += spike + qkvo + dd + gates + mlp;
                mixer += delta_core(cfg.d, mem);
            }
        }
    }
    if cfg.engram.is_some() {
        w += qkvo;
        mixer += causal_core(d, ctx);
        // v45 W4: the engram's output gate h·W_th
        w += match cfg.arch.eng_gate {
            crate::arch::EngGate::Off => 0.0,
            crate::arch::EngGate::Head => d * h,
            crate::arch::EngGate::Elem => dd,
        };
    }
    Flops { unembed: 2.0 * d * v, dense: 2.0 * w, mixer }
}

// ------------------------------------------------------------------------------------------------ measured runs

/// One measured run: its family (a model kind), shape and validation BPB.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    pub family: String,
    pub d: usize,
    pub layers: usize,
    pub bpb: f64,
    pub seed: u64,
}

/// Runs from a TSV: `family d layers bpb seed`; blank lines, '#' comments and a header starting with "family" are
/// skipped.
pub fn read_runs(text: &str) -> Result<Vec<Run>, String> {
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("family") {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 5 {
            return Err(format!("line {}: {} fields, want family d layers bpb seed", ln + 1, f.len()));
        }
        let bad = |what: &str| format!("line {}: bad {what}", ln + 1);
        out.push(Run {
            family: f[0].to_string(),
            d: f[1].parse().map_err(|_| bad("d"))?,
            layers: f[2].parse().map_err(|_| bad("layers"))?,
            bpb: f[3].parse().map_err(|_| bad("bpb"))?,
            seed: f[4].parse().map_err(|_| bad("seed"))?,
        });
    }
    Ok(out)
}

/// The model config of a run (its family must be a model kind: S, T, F, G, R, D or A).
pub fn config_of(r: &Run, vocab: usize, ctx: usize) -> Result<Config, String> {
    let kind = Kind::parse(&r.family)?;
    let c = Config { kind, vocab, d: r.d, layers: r.layers, ctx, glance: false, mag: false, engram: None, arch: crate::arch::Arch::default() };
    c.validate()?;
    Ok(c)
}

/// What a run's size is measured in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// weights without the token and position embeddings (v34's N)
    NonEmbed,
    /// every weight
    Total,
    /// training FLOPs per token (forward + backward)
    Flops,
    /// training FLOPs per token × tokens trained (tokens = tokens_per_weight × every weight)
    Compute,
}

impl Axis {
    pub fn parse(s: &str) -> Result<Axis, String> {
        match s {
            "nonembed" => Ok(Axis::NonEmbed),
            "total" => Ok(Axis::Total),
            "flops" => Ok(Axis::Flops),
            "compute" => Ok(Axis::Compute),
            _ => Err(format!("unknown axis {s:?} (nonembed, total, flops, compute)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Axis::NonEmbed => "non-embedding weights",
            Axis::Total => "all weights",
            Axis::Flops => "training FLOPs per token",
            Axis::Compute => "training FLOPs in all",
        }
    }
}

/// A config's size on an axis.
pub fn size_on(cfg: &Config, axis: Axis, mem: MemForm, tokens_per_weight: f64) -> f64 {
    let (total, nonembed) = cfg.weight_counts();
    match axis {
        Axis::NonEmbed => nonembed as f64,
        Axis::Total => total as f64,
        Axis::Flops => per_token(cfg, mem).train(),
        Axis::Compute => per_token(cfg, mem).train() * tokens_per_weight * total as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(kind: Kind, d: usize, l: usize) -> Config {
        Config::with_kind(kind, 4096, d, l, 256)
    }

    /// Vector parameters (norm gains, LIF leaks, gate biases) per config: the weights that are not matrix entries.
    fn vectors(c: &Config) -> usize {
        let (d, h) = (c.d, c.heads());
        let per = |i: usize| match c.kind {
            Kind::S => 3 * d,
            Kind::T => 2 * d,
            Kind::F => 4 * d,
            Kind::G | Kind::R | Kind::D => 4 * d + 2 * h,
            Kind::A if c.attn_layer(i) => 4 * d,
            Kind::A => 4 * d + 2 * h,
        };
        (0..c.layers).map(per).sum::<usize>() + d
    }

    #[test]
    fn dense_flops_are_twice_the_matrix_weights_of_model_rs() {
        // ties the count above to Config::weight_counts (model.rs) for every kind at several shapes
        for kind in [Kind::S, Kind::T, Kind::F, Kind::G, Kind::R, Kind::D, Kind::A] {
            for (d, l) in [(64, 2), (128, 4), (192, 6), (384, 8)] {
                let c = cfg(kind, d, l);
                let (_, nonembed) = c.weight_counts();
                let f = per_token(&c, DEFAULT_MEM);
                assert_eq!(f.dense, 2.0 * (nonembed - vectors(&c)) as f64, "{kind:?} d{d} L{l}");
                assert_eq!(f.unembed, 2.0 * d as f64 * 4096.0);
            }
        }
    }

    #[test]
    fn a_hand_count_of_small_models() {
        // T d64 L2 ctx 256: unembed 2·64·4096; dense 2·2·(4 + 8)·64²; attention 2 layers · 2·64·257
        let f = per_token(&cfg(Kind::T, 64, 2), DEFAULT_MEM);
        assert_eq!((f.unembed, f.dense, f.mixer), (524_288.0, 196_608.0, 65_792.0));
        assert_eq!(f.train(), 3.0 * 786_688.0);
        // S has no look-back core
        assert_eq!(per_token(&cfg(Kind::S, 64, 2), DEFAULT_MEM).mixer, 0.0);
        // D d128 (4 heads): per-token 4·6·32² = 24,576 per layer; chunk 16 adds 4·79·32 = 10,112
        assert_eq!(delta_core(128, MemForm::PerToken), 24_576.0);
        assert_eq!(delta_core(128, MemForm::Chunk(16)), 34_688.0);
        let d = per_token(&cfg(Kind::D, 128, 4), MemForm::Chunk(16));
        assert_eq!(d.mixer, 4.0 * 34_688.0);
        // A d192 L6: blocks 1 and 5 have attention, the other four the delta rule
        let a = per_token(&cfg(Kind::A, 192, 6), DEFAULT_MEM);
        assert_eq!(a.mixer, 2.0 * 2.0 * 192.0 * 257.0 + 4.0 * delta_core(192, DEFAULT_MEM));
        assert_eq!(MemForm::parse("chunk16"), Ok(MemForm::Chunk(16)));
        assert_eq!(MemForm::parse("token"), Ok(MemForm::PerToken));
        assert!(MemForm::parse("chunk0").is_err());
    }

    #[test]
    fn runs_read_from_tsv() {
        let rs = read_runs("family\td\tlayers\tbpb\tseed\n# v34\nT 64 2 2.3053 1\nD\t384\t8\t1.487072\t1\n\n").unwrap();
        assert_eq!(rs.len(), 2);
        assert_eq!(rs[1], Run { family: "D".into(), d: 384, layers: 8, bpb: 1.487072, seed: 1 });
        let c = config_of(&rs[1], 4096, 256).unwrap();
        // Ava-B's 17,781,312 non-embedding weights (v36 notes)
        assert_eq!(size_on(&c, Axis::NonEmbed, DEFAULT_MEM, 20.0), 17_781_312.0);
        assert_eq!(size_on(&c, Axis::Total, DEFAULT_MEM, 20.0), 19_354_176.0);
        assert!(read_runs("T 64 2 x 1").is_err() && read_runs("T 64 2").is_err());
        assert!(config_of(&Run { family: "Q".into(), d: 64, layers: 2, bpb: 2.0, seed: 1 }, 4096, 256).is_err());
    }
}
