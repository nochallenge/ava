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
//
// Default data paths are relative to this crate; the CUDA toolkit path comes from the environment.

pub mod eval;
pub mod lif;
pub mod model;
pub mod optim;

pub const DEFAULT_BPE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../release/ava-m/bpe.txt");
pub const DEFAULT_TRAIN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/nn/shards-4096/train");
pub const DEFAULT_VAL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/nn/shards-4096/val");

/// Target marker for "do not score" (PAD targets, targets past the eval limit).
pub const IGNORE: u32 = u32::MAX;

/// Pick the device: CUDA 0 unless `cpu` is set (or CUDA is unavailable, with a warning).
pub fn device(cpu: bool) -> candle_core::Result<candle_core::Device> {
    if cpu {
        return Ok(candle_core::Device::Cpu);
    }
    match candle_core::Device::new_cuda(0) {
        Ok(d) => Ok(d),
        Err(e) => {
            eprintln!("warning: CUDA device 0 unavailable ({e}); falling back to CPU");
            Ok(candle_core::Device::Cpu)
        }
    }
}
