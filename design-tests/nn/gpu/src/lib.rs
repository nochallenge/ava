// TESTING / DESIGN ONLY — ava_gpu: the Ava-S spiking language model on candle (CUDA or CPU).
//
// `model`  : Ava-S exactly as design-tests/nn/brain/src/model.rs defines it (tied embedding, pre-norm
//            RMSNorm blocks, SpikeMem = Win → LIF scan → Wout, SparseMLP = W1 → kWTA → relu → W2),
//            with the LIF scan run by the fused op of `lif`.
// `lif`    : the fused LIF scan, a custom op with a hand-written backward (CUDA kernels in lif.cu,
//            compiled by build.rs; a CPU path in Rust).
// `optim`  : AdamW(0.9, 0.95, 1e-8) with decoupled weight decay on the matrices only, global-norm
//            gradient clipping and the warmup + cosine schedule of ava_grad::optim.
// `eval`   : validation bits per byte exactly as brain/src/eval.rs (EOS = 1 byte, PAD skipped).
// v42 Wave 4 (the gpu side of the always-online learner, design-tests/nn/online): `hotswap` (ava_live follows the
// learner's CURRENT), `probe` (gate_probe, think, episode_surprise); eval yardsticks `tokgap` (T4), `preq` (I1) and
// `synops` (C14).
//
// Default data paths are relative to this crate; the CUDA toolkit path comes from the environment.

pub mod eval;
pub mod fused;
pub mod fwin_ops;
pub mod lif;
pub mod lif_gen;
pub mod mag;
pub mod mem_gen;
pub mod anchor;
pub mod arch;
pub mod ckpt;
pub mod contam;
pub mod diary;
pub mod dist;
// M7 + H1: gradient-free evolution strategies with low-rank noise (es_tune)
pub mod es;
pub mod flops;
pub mod grow;
pub mod hotswap;
// E10 + E2: int8 weights and event-driven sums for talking on the CPU (Model::set_int8)
pub mod int8;
pub mod mc;
pub mod memory;
pub mod model;
pub mod optim;
pub mod plastic;
pub mod preq;
pub mod probe;
pub mod qa;
// A13: sparse autoencoders and linear probes over her states (sae_train, probe_states)
pub mod sae;
pub mod lookup;
pub mod introspect;
pub mod scaling;
pub mod scan_fast;
pub mod state;
pub mod thread_env;
pub mod stats;
pub mod synops;
pub mod textgen;
#[cfg(test)]
pub(crate) mod testkit;
pub mod tokgap;
pub mod trainkit;
pub mod upkeep;
// v50 (runs/v50/prereg_v50.md): --dale, --loop, --neuromod
pub mod v50;
// v51 Part B2 (runs/v51/prereg_v51.md): --thalamus, --cerebellum-head; Part B: e-prop and online learning sessions
pub mod v51b;
pub mod eprop;
// v51 Part B's one redesign (runs/v51/prereg_v51.md, Amendment B-4): the forgetting test's shared trunk
pub mod v51e;
// v51 fast4 (runs/v51/fast4.md): the fused talking step with an online session (e-prop, the cerebellum's table)
pub mod fast4;
pub mod wave5;
// v51 (runs/v51/prereg_v51.md): round 2 brain parts
pub mod v51;
// v51 fused N (runs/v51/fused_n_check.sh): --neuromod 3 on the hand-written WIN training step
pub mod fwin_nm;
// v51 fast1 (runs/v51/fast1.md): round 2's P, DA, M, Y, Pr and S on the hand-written WIN training step
pub mod fwin_r2;

pub const DEFAULT_BPE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../release/ava-m/bpe.txt");
pub const DEFAULT_TRAIN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/nn/shards-4096/train");
pub const DEFAULT_VAL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/nn/shards-4096/val");

/// Target marker for "do not score" (PAD targets, targets past the eval limit).
pub const IGNORE: u32 = u32::MAX;

/// Speed round 2 (src/fused.rs: fused rmsnorm, kWTA + relu, AdamW; rows-first grad norm) on CUDA. On by
/// default; AVA_FAST=0 restores the primitive candle chains (for side-by-side checks).
pub fn fast_path() -> bool {
    static FAST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FAST.get_or_init(|| std::env::var("AVA_FAST").map_or(true, |v| v != "0"))
}

/// Speed round 7: bf16 matmul operands in the hand-written S and T steps. On by default since its pre-registered
/// check passed (S 2.0102 vs 2.0131, T 1.9602 vs 1.9594); AVA_BF16=0 turns it off.
pub fn bf16_on() -> bool {
    static B: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *B.get_or_init(|| !std::env::var("AVA_BF16").is_ok_and(|v| v == "0"))
}

/// Speed round 9: causal attention on the tensor cores in the hand-written bf16 T step. On by default: its check
/// passed at the edge (+0.004995 BPB, bar 0.005) and a paired test over three seeds gave a mean of +0.0003.
/// AVA_ATTN_TC=0 turns it off.
pub fn attn_tc_on() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| !std::env::var("AVA_ATTN_TC").is_ok_and(|v| v == "0"))
}

/// Pick the device: CUDA 0 unless `cpu` is set (or CUDA is unavailable, with a warning).
pub fn device(cpu: bool) -> candle_core::Result<candle_core::Device> {
    if cpu {
        return Ok(candle_core::Device::Cpu);
    }
    match candle_core::Device::new_cuda(0) {
        Ok(d) => {
            // v47: the CUDA pool keeps freed memory between steps (release threshold max) unless AVA_POOL_KEEP=0. With
            // the default threshold 0 the pool hands memory back at every sync and the next step maps it again.
            if !std::env::var("AVA_POOL_KEEP").is_ok_and(|v| v == "0") {
                use cudarc::driver::sys::{cuDeviceGetDefaultMemPool, cuMemPoolSetAttribute, CUmemPool_attribute, CUresult};
                unsafe {
                    let mut pool = std::mem::MaybeUninit::uninit();
                    if cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), 0) == CUresult::CUDA_SUCCESS {
                        let mut v = u64::MAX;
                        let attr = CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
                        let _ = cuMemPoolSetAttribute(pool.assume_init(), attr, &mut v as *mut u64 as *mut _);
                    }
                }
            }
            // AVA_TF32=1: f32 matmuls on the tensor cores (TF32 inputs, f32 accumulate). Off by default.
            if std::env::var("AVA_TF32").is_ok_and(|v| v == "1") {
                candle_core::cuda::set_gemm_reduced_precision_f32(true);
                eprintln!("TF32 matmuls on");
            }
            if bf16_on() {
                eprintln!("bf16 matmuls on (hand-written S and T steps; AVA_BF16=0 turns them off)");
            }
            Ok(d)
        }
        Err(e) => {
            eprintln!("warning: CUDA device 0 unavailable ({e}); falling back to CPU");
            Ok(candle_core::Device::Cpu)
        }
    }
}

/// Speed round 10: R's (and G's) fast-weight read as one fused op. On by default since round 10b's check passed
/// (R 1.879059 vs 1.882663 with TF32 off; with TF32 on too it failed at -0.0061). AVA_GLA=0 turns it off.
pub fn gla_on() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| fast_path() && !std::env::var("AVA_GLA").is_ok_and(|v| v == "0"))
}

/// Speed round 11: D's gated delta rule as one fused recurrence op. On by default since its check passed (D d128 L4
/// lr 0.001: 1.896743 fused vs 1.899273 chunked; 3.2x faster per step). AVA_DELTA=0 turns it off.
pub fn delta_on() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| fast_path() && !std::env::var("AVA_DELTA").is_ok_and(|v| v == "0"))
}

/// Speed round 12: the hand-written training step for D and A. On by default since its check passed (A d128 L4 lr
/// 0.004: 1.883378 vs 1.882973 autograd; 2.5x faster at Ava-B's shape). AVA_MANUAL_DA=0 turns it off.
pub fn manual_da_on() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| !std::env::var("AVA_MANUAL_DA").is_ok_and(|v| v == "0"))
}
// v50 (runs/v50/prereg_v50.md): the option-off identity check (CPU)
#[cfg(test)]
mod v50_golden;
// merge check (runs/v51/merge_a.md): round 1's trained models give the same numbers before and after a merge (CPU)
#[cfg(test)]
mod merge_ident;
// v51 Part B2 (runs/v51/prereg_v51.md): the option-off identity check (CPU)
#[cfg(test)]
mod v51b_golden;
// merge check B (runs/v51/merge_b.md): v51b's options give the same numbers in this crate as in gpu-v51b (CPU)
#[cfg(test)]
mod merge_ident_b;
