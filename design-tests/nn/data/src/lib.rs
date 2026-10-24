//! TESTING / DESIGN ONLY — ava_data: the data side of Ava's scalable language model.
//!
//! What lives here:
//! - `rng`    : splitmix64, the only randomness used (shuffles, the train/val split).
//! - `hash`   : a small fast deterministic hasher for the internal hash maps (std only).
//! - `bpe`    : byte-level BPE (pre-tokenizer, trainer, encoder, decoder, save/load).
//! - `shards` : sharded token files on disk, a streaming shuffled window sampler with a
//!              resumable cursor, and a sequential reader for evaluation.
//! - `input`  : input specs (`path`, `path@tab:N`, `path@sep:STR:N`) and a streaming,
//!              line-by-line document reader shared by the binaries.
//!
//! Memory is bounded everywhere: corpora are streamed line by line, the BPE trainer works
//! on a chunk-frequency table from a byte-limited sample, the shard writer streams to disk,
//! and the sampler holds one shard at a time.

pub mod bpe;
pub mod hash;
pub mod input;
pub mod rng;
pub mod shards;
