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
//! - `regmix` : v42: mixture fitting (RegMix): Dirichlet proposals, per-source loss regression, a bounded simplex search.
//! - `docs`   : v42: the docs file (multi-line documents ended by a U+001E line) and one reader for every input kind.
//! - `dialogue`: v42: whole conversations as documents (Cornell threads, DailyDialog rebuilt from triples).
//! - `arc_pack`: v42: ARC_Corpus sentences packed into topic documents, exact + MinHash near dedup, coherence score.
//! - `babylm` : v42: BabyLM 2024 strict files packed in order into ~1024-token documents (PG-19 books left out).
//! - `tags`   : v42: origin tags and a soft toxicity score (ROT13-stored lexicon), as metadata, never a ban.
//! - `v42_shards`: v42: whole sources into shards, a val slice per source, and a tag sidecar per split.
//! - `fact_ledger`: v42: 200 hand-written facts, a ledger of how often her text states them, restatements reconciled with the test keys.
//! - `bios`   : v42: synthetic biographies (known bits per person, E exposures) and the bits-stored formulas.
//! - `dyck`   : v42: nested-bracket (Dyck-k) sequences as content-free pre-pretraining text.
//! - `wikidata`: v42: the Wikidata country/capital/continent and element/symbol tables (CC0).
//! - `wording_guard`: v42: finds the test templates filled with any name, so training text never holds them.
//! - `lessons`: v42: the owner's early lessons (question, answer TSVs) as documents in her taught QA format.
//! - `maths`  : v42 (C11): worked word problems as documents, and our own arithmetic / numeric-answer checker.
//! - `code_docs`: v42 (C11): permissively licensed Rust crates from the local cargo registry as documents.
//! - `reason` : v42 Wave 5: the reasoning curricula hub (seven families, one doc per (seed, index)); the families:
//!   `reason_search` (Countdown and maze search traces), `reason_causal` (cause-and-effect worlds with an exact
//!   engine), `reason_bayes` (belief updates over turns in exact fractions), `reason_rules` (list-rule induction
//!   over rules disjoint from the test's), `reason_state` (state tracking, 2 to 64 steps), `reason_analogy`
//!   (string changes and word relations the test does not use), `reason_library` (a small list language,
//!   program search and library compression).
//! - `reason_guard`: v42 Wave 5: the reasoning, social and other v42 test frames (copied from the evalsets
//!   generator code), so reasoning text never holds them; the number-run limit.
//!
//! Memory is bounded everywhere: corpora are streamed line by line, the BPE trainer works
//! on a chunk-frequency table from a byte-limited sample, the shard writer streams to disk,
//! and the sampler holds one shard at a time.

/// The repo's data folder (`<repo>/data`), found from this crate's own folder, so no machine path is written
/// into the code.
pub const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data");

pub mod books;
pub mod arc_pack;
pub mod babylm;
pub mod bios;
pub mod bpe;
pub mod code_docs;
pub mod dialogue;
pub mod docs;
pub mod dyck;
pub mod fact_items;
pub mod fact_ledger;
pub mod hash;
pub mod input;
pub mod lessons;
pub mod lp_mix;
pub mod maths;
pub mod num_items;
pub mod num_mix;
pub mod num_practice;
pub mod numbers;
pub mod qa_stems;
pub mod reason;
pub mod reason_analogy;
pub mod reason_bayes;
pub mod reason_causal;
pub mod reason_guard;
pub mod reason_library;
pub mod reason_rules;
pub mod reason_search;
pub mod reason_state;
pub mod recall;
pub mod regmix;
pub mod rng;
pub mod sha256;
pub mod shards;
pub mod sleep_facts;
pub mod tags;
pub mod v40_mix;
pub mod v41_mix;
pub mod v42_shards;
pub mod wikidata;
pub mod wording_guard;
