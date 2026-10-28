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
//! - `numbers`: v39's glance: length marks before digit runs (ids 4096..=4111, vocab 4112),
//!              the per-digit place code the model reads off them, and number words.
//! - `num_items`: v39's generated number test (six kinds; a dev set and a sealed test set
//!              that never share an item).
//! - `num_practice`: child-style number practice text, kept clear of the test's items.
//! - `num_mix`: rewrites a shard directory with length marks and/or practice docs mixed in.
//! - `sha256` : SHA-256, to prove a regenerated file matches the one on disk.
//! - `books`  : v40: PG-19 books, each one marked document (normalised line endings).
//! - `v40_mix`: v40: the reading mix (books 50%, usual text 47%, practice v2 3%), doc-shuffled.
//! - `fact_items`: v40: the find-the-fact long-memory test (a made-up code, a gap, a cue).
//! - `sleep_facts`: v40: made-up people (name, year, town, job) and questions about them.
//! - `recall` : v41: recall practice rows (made-up notes planted in reading text, cued later) and dev items.
//! - `v41_mix`: v41: 100 held-out PG-19 train books; the mix (usual 70%, books 25%, practice v3 5%).
//! - `qa_stems`: v41: normalised question stems and the split hash of the held-out QA slices.
//!
//! Memory is bounded everywhere: corpora are streamed line by line, the BPE trainer works
//! on a chunk-frequency table from a byte-limited sample, the shard writer streams to disk,
//! and the sampler holds one shard at a time.

pub mod books;
pub mod bpe;
pub mod fact_items;
pub mod hash;
pub mod input;
pub mod num_items;
pub mod num_mix;
pub mod num_practice;
pub mod numbers;
pub mod qa_stems;
pub mod recall;
pub mod rng;
pub mod sha256;
pub mod shards;
pub mod sleep_facts;
pub mod v40_mix;
pub mod v41_mix;
