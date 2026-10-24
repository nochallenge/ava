// TESTING / DESIGN ONLY — ava_brain: the model crate of the scalable, brain-like Ava language model.
// Three pre-registered models on top of ava_grad (autodiff, LIF scan, kWTA, AdamW) and ava_data
// (byte-level BPE, token shards):
//   T — yardstick transformer (never part of Ava),
//   S — Ava spiking (SpikeMem LIF memory + kWTA sparse MLP, no attention),
//   H — Ava hybrid (SpikeMem + causal attention + kWTA sparse MLP).
// Modules: model (definitions, init, forward), eval (bits per byte, activity), trainer (one step),
// sys (peak resident memory for reports), generate (checkpoint architecture inference, sampling,
// autoregressive generation, file scoring — inference only).
// Binaries: train (training + evaluation + checkpoint), profile (per-op forward/backward timing),
// generate (sample a continuation), talk (line REPL), score (bits per byte of a text file).

pub mod eval;
pub mod generate;
pub mod model;
pub mod sys;
pub mod trainer;

pub use eval::{byte_lengths, evaluate, row_nll, train_io, window_io, BpbAcc, EvalResult};
pub use model::{Config, Forward, Kind, Model, OpTimes, OP_CATS};
pub use trainer::{Hyper, StepStats, Trainer};
