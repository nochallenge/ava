# Ava: architecture

The technical companion to [README.md](README.md). It covers the three crates that make up the
trained brain, the file formats, the exact definitions of the three models, the gradient of the
two brain ops, the v32 numbers in full, and where the time goes. Paths are relative to the
repository root. "As of v32" means the state measured on 2026-10-16; where the release build has
moved on, the text says so.

## 1. The three crates

All three live under `design-tests/nn/`, each with its own `[workspace]` so none is pulled into
the older crate. They use the Rust standard library only: no crates.io dependencies. A fourth crate, `nn/gpu`, is the exception: the same spiking model on candle with CUDA (section 13).

| Crate | Package | Depends on | Job |
|---|---|---|---|
| `nn/data` | `ava_data` | nothing | byte-level BPE, token shards, streaming sampler with a resumable cursor, sequential reader for evaluation |
| `nn/grad` | `ava_grad` | nothing | f32 tensors, parameter store and `AVAWTS01` checkpoints, define-by-run reverse-mode autodiff, threaded GEMM, the LIF scan and kWTA kernels, AdamW, clipping, the cosine schedule |
| `nn/brain` | `ava_brain` | `ava_grad`, `ava_data` | the T / S / H models, one training step, bits per byte, inference, and the `train`, `profile`, `generate`, `talk` and `score` binaries |

### Files

| File | What it does |
|---|---|
| `nn/data/src/bpe.rs` | pre-tokenizer, BPE trainer, encoder with a bounded chunk cache, decoder, save/load of `ava-bpe v1` |
| `nn/data/src/shards.rs` | `AVATOK01` shard files, `ShardWriter`/`TrainValWriter`, the shuffled `Sampler` and its cursor, `iter_sequential` for evaluation |
| `nn/data/src/input.rs` | input specs (`path`, `path@tab:N`, `path@sep:STR:N`), a line-by-line document reader with a 1 MiB line cap, the byte-fair and proportional mixers |
| `nn/data/src/rng.rs` | splitmix64, the only randomness in the crate; `mix64` for the train/val split rule |
| `nn/data/src/hash.rs` | a fast deterministic hasher for the BPE trainer's maps |
| `nn/data/src/bin/bpe_train.rs` | `bpe_train --vocab N --out FILE [--max-bytes B] INPUT...` |
| `nn/data/src/bin/make_shards.rs` | `make_shards --bpe FILE --out DIR [--shard-tokens N] [--val-per-mille V] [--max-docs-per-input M] [--seed S] [--verify-every K] INPUT...` |
| `nn/data/src/bin/shard_stats.rs` | `shard_stats DIR [--bpe FILE] [--ctx N] [--seed S] [--count-docs]` |
| `nn/grad/src/tensor.rs` | `Tensor { shape, data: Vec<f32> }`, row-major, no views or strides: every op makes a fresh contiguous tensor |
| `nn/grad/src/params.rs` | named parameters, a gradient buffer per tensor, a weight-decay flag per tensor, `Grads`, the `AVAWTS01` checkpoint |
| `nn/grad/src/graph.rs` | the tape: `Graph::new(&params)`, ops recorded as they run, `backward(loss)` walks the tape once in reverse |
| `nn/grad/src/matmul.rs` | blocked GEMM (4×16 register tile, KC 256, MC 64, AVX2+FMA path chosen at run time), batched GEMM, `threads()`, the row- and chunk-parallel helpers |
| `nn/grad/src/brain.rs` | `lif_forward[_par]`, `lif_backward[_par]`, `kwta_forward[_par]`, `SpikeFn`, `LifOpts`, `fraction_active` |
| `nn/grad/src/optim.rs` | `AdamW`, `clip_grad_norm`, `cosine_lr` |
| `nn/grad/src/rng.rs` | splitmix64 + Box-Muller normals for initialisation |
| `nn/grad/tests/` | finite-difference checks of every op and of two composite nets, the hand-worked LIF case, GEMM against a naive reference, optimiser and checkpoint round trips |
| `nn/brain/src/model.rs` | `Kind {T, S, H}`, `Config`, `Model::new` (init), `forward`, `weight_counts`, per-op timing |
| `nn/brain/src/trainer.rs` | one optimisation step: schedule → forward → backward → accumulate → clip → AdamW → zero |
| `nn/brain/src/eval.rs` | bits per byte in f64, activity statistics, the evaluation loop over `iter_sequential` |
| `nn/brain/src/generate.rs` | inference only: architecture inference from a checkpoint, sampling, autoregressive generation, file scoring |
| `nn/brain/src/sys.rs` | peak resident memory for the result line |
| `nn/brain/src/bin/train.rs` | the training binary (section 7) |
| `nn/brain/src/bin/profile.rs` | times every op the models use at the real shapes, forward and backward, and compares the sum with a measured step |
| `nn/brain/src/bin/generate.rs`, `talk.rs`, `score.rs` | sample a continuation; a line REPL; bits per byte of a text file (section 11) |
| `nn/brain/tests/models.rs` | shapes, loss at init, every parameter gets a gradient, exact causality, BPB on a hand-computed case, determinism of training steps, checkpoint round trip |
| `nn/brain/tests/generate.rs` | architecture inference, the sampling helpers, seed determinism of generation, the `score` round trip |

## 2. Data path

### Tokenizer (`ava-bpe v1`)

Byte-level BPE. The vocabulary is laid out so nothing is ever unknown:

| Ids | Meaning |
|---|---|
| 0..=255 | the raw bytes |
| 256 | EOS, the document separator (`encode_doc` appends it) |
| 257 | PAD (never produced by the encoder; pads the last evaluation window) |
| 258.. | learned merges in the order learned: id = 258 + rank |

`vocab_size = 258 + merges`; the 4096 vocabulary used so far has 3,838 merges.

The pre-tokenizer uses generic character classes only (letter, digit, whitespace, other) and no
word lists. Text is cut into chunks and merges never cross a chunk: one ASCII space plus a run of
letters, one space plus a run of other characters, a run of letters, a run of other characters,
a run of whitespace, and **every digit alone**. Digits never merge with anything, so numbers are
spelt digit by digit and arithmetic stays learnable. Training keeps pair counts incrementally
with a lazy max-heap, breaks ties by count then by the smaller pair, and is fully deterministic.
Encoding merges the lowest-rank pair repeatedly, which equals applying every merge in learned
order; a bounded cache (1M chunks, cleared when full) makes repeated words cheap. Decoding
concatenates bytes with `from_utf8_lossy`; `decode(encode_doc(s)) == s`.

File: plain text, first line `ava-bpe v1 <vocab_size>`, then one merge per line, `left right`
as decimal ids, in rank order.

### Shards (`AVATOK01`)

Little-endian files named `shard_00000.tok`, `shard_00001.tok`, ...:

| Bytes | Field |
|---|---|
| 0..8 | magic `AVATOK01` |
| 8..12 | u32 version = 1 |
| 12..16 | u32 vocab_size |
| 16..24 | u64 n_tokens |
| 24 | u8 width: 2 (u16 tokens) when vocab_size ≤ 65536, else 4 |
| 25..32 | zero padding (32-byte header) |
| 32.. | n_tokens tokens of `width` bytes |

`ShardWriter` streams tokens straight to disk (a `.tmp` renamed into place, header patched with
the final count), cuts shards at exactly `--shard-tokens` (default 50M; a document that does not
fit continues in the next shard), and holds no shard in memory. `make_shards` routes document
`i` to `val/` when `mix64(seed ^ i) % 1000 < val_per_mille` (default 5, that is 0.5%), writes a
copy of the tokenizer as `DIR/bpe.txt` and the run's numbers as `DIR/manifest.txt`.

**Sampling.** `Sampler` yields windows of `ctx + 1` consecutive tokens (input plus shifted
target). A shard of `n` tokens has `(n − 1) / ctx` slots; consecutive slots share one token, so
every token after the first is a target exactly once per epoch and no window crosses a shard.
Epoch `e` visits the shards in a shuffled order seeded by `derive_seed(seed, e, u64::MAX)` and
each shard's slots in a permutation seeded by `derive_seed(seed, e, shard_index)`. One shard is
loaded at a time.

**Cursor.** `ava-cursor v1 seed=.. ctx_len=.. epoch=.. shard_pos=.. slot_pos=..` names the next
window; `Sampler::resume` continues with exactly the windows the original would have produced.

**Evaluation.** `iter_sequential(ctx)` reads the shards in file order as one stream, windows at
0, ctx, 2·ctx, ... (they may cross shards), pads the last partial window with PAD, and drops a
final single token.

## 3. The gradient engine

**Tape.** `Graph::new(&params)` borrows the store read-only. Every op runs immediately and pushes
one node (value, op, `needs_grad`). `Var` is a copyable index; a `Var` may be used any number of
times and the contributions add. `g.param(id)` is memoised and reads the value from the store
without copying. Nodes that depend on no parameter are skipped in backward. `backward(v)` seeds
`dL/dv = 1` for every element (a non-scalar behaves like its sum) and returns `Grads`, which the
caller accumulates into the store after the graph's last use.

**Ops.** `matmul` (flat `[.., m, k] × [k, n]` or batched), `matmul_t`, `add`, `sub`, `mul`,
`scale`, `add_bias`, `relu`, `gelu`, `sigmoid`, `tanh`, `exp`, `rmsnorm`, `layernorm`,
`embedding` (backward scatter-adds), `reshape`, `transpose`, `softmax_last` (optionally
causal), `cross_entropy` (fused stable log-softmax, `IGNORE` targets skipped), `sum`, `mean`,
`kwta`, `lif_scan` / `lif_scan_opts`.

**GEMM.** `C += op(A) · op(B)`. B is packed once per call into 16-wide column panels; the
micro-kernel keeps a 4×16 tile of C in registers; k is blocked at 256 and rows at 64; a copy
compiled for AVX2+FMA is chosen at run time on x86-64. Threads come from `std::thread::scope`,
one contiguous row range each, so every output element is produced by exactly one thread with
the same k order. `threads()` reads `AVA_THREADS` (default 4) and `Graph::set_threads` clamps
to 1..=32 (`MAX_THREADS`).

**Checkpoint (`AVAWTS01`).** Little-endian: magic `AVAWTS01` (8 bytes), u32 n_tensors, then
per tensor: u32 name_len, UTF-8 name, u32 ndim, u64 dims[ndim], f32 data[prod(dims)]. Only the
tensors are stored, written to `<path>.tmp` and renamed into place. The model kind and size are
recoverable from the names and shapes (section 11): `tok_emb` is `[V, d]`; the number of
`b{i}.*` groups is `L`; `b{i}.mem.*` present means S or H; `pos_emb` present means T or H and
its shape `[ctx, d]` gives the training context. S stores no `pos_emb`, so nothing in an S
checkpoint depends on the context length.

## 4. The brain ops

### Leaky integrate-and-fire scan

Input current `I [B, T, D]`, leak parameter `p [D]`. Forward, per sequence and channel, with
`v_0 = 0`, `s_0 = 0`:

```
β   = σ(p)                          in (0, 1); initialised so that β = 0.9
v_t = β·v_{t−1} + I_t − θ·s_{t−1}   soft reset by subtraction
s_t = f(v_t − θ)                    θ = 1
```

`f` is the Heaviside step (`SpikeFn::Surrogate`, used by the models) or `σ(slope·x)`
(`SpikeFn::Sigmoid`, smooth, so the whole BPTT path can be finite-difference checked). Output:
spikes `[B, T, D]`; the membrane `v` is saved for backward.

**Backward is full backpropagation through time inside one op.** With `g_t = dL/ds_t` and
`δv_{T+1} = 0`, going `t = T..1`:

```
δs_t = g_t − θ·δv_{t+1}         the −θ term is dropped when the reset is detached (default)
δv_t = δs_t·f'(v_t − θ) + β·δv_{t+1}
dI_t = δv_t
dβ   = Σ_t δv_t · v_{t−1}
dp   = dβ · β(1 − β)
```

The surrogate derivative is the fast sigmoid `f'(x) = 1 / (1 + slope·|x|)²`, slope 4 in the
models. `detach_reset = true` (as in Zenke & Vogels 2021 / SpyTorch) treats the reset spike as a
constant in the backward pass; `false` gives the exact gradient of the forward function. Both
are checked against a case worked out on paper in `nn/grad/tests/lif.rs` (one channel, three
steps, θ = 1, β = 0.5, currents 0.8, 0.9, 0.3, slope 2). `dβ` is accumulated in f64.

**Threads.** The scan is split into batch groups × channel blocks; every (batch, channel) pair's
time loop runs whole inside one task in the same order as on one thread, so the split never
changes a bit. (As of v32 the scan ran on one thread; the parallel kernels are in the release
build.)

### k-winners-take-all

Per row of the last axis keep the `k` largest entries and zero the rest. Ties are broken by the
lower index (`f32::total_cmp`, value descending then index ascending, via
`select_nth_unstable_by`). Backward passes the gradient through the kept entries only (the
boolean mask is saved on the tape). `k ≥ d` is the identity; `k = 0` is all zeros. Rows are
independent, so the row-parallel version is bit-identical for any thread count.

### Finite-difference checks

`nn/grad/tests/common/mod.rs` perturbs every scalar of every listed parameter by ±1e-3 (central
differences), with a random weighted-sum loss so every output element gets its own upstream
gradient. Each op is checked alone, then two composite nets: a pre-norm transformer block with
causal attention, and a spiking block with kWTA + LIF (sigmoid spike function, since a step has
no finite-difference derivative).

## 5. The three models

Shared: vocabulary `V = 4096`, token embedding `E [V, d]` tied to the output
(`logits = x · Eᵀ`), pre-norm residual blocks with `rmsnorm` (gain initialised to 1, eps 1e-5),
a final `rmsnorm`. Weight matrices are stored `[d_in, d_out]` and applied as `h · W`.

| | T (yardstick) | S (Ava spiking) | H (hybrid) |
|---|---|---|---|
| Position embedding `[ctx, d]` | yes | **no** | yes |
| Block | `x += Attn(norm x)`; `x += MLP(norm x)` | `x += SpikeMem(norm x)`; `x += SparseMLP(norm x)` | `x += SpikeMem(norm x)`; `x += Attn(norm x)`; `x += SparseMLP(norm x)` |
| Attn | causal multi-head, head size 32, `q/k/v/o` `d→d` without biases, scores · 1/√32, causal softmax | none | as T |
| SpikeMem | none | `cur = h·Win [d,d]`; `s = lif_scan(cur [B,T,d], β [d], θ 1, Surrogate{slope 4}, reset detached)`; `out = s·Wout [d,d]` | as S |
| MLP | `d→4d`, GELU, `4d→d` | `u = h·W1 (d→4d)`; `a = relu(kwta(u, k = 4d/8))`; `out = a·W2 (4d→d)` | as S |

**Init.** Every weight matrix N(0, 0.02) except the projections that write into the residual
stream (`wo`, `w2`, `wout`): N(0, 0.02/√(2L)). Embeddings N(0, 0.02). `Win` N(0, 1/√d). LIF leak
parameter `logit(0.9)` in every channel. Norm gains 1. Tensors are created, and the init RNG
consumed, in a fixed order (`tok_emb`, `pos_emb`, then per block `mem.{norm,win,beta,wout}`,
`attn.{norm,wq,wk,wv,wo}`, `mlp.{norm,w1,w2}`, then `final.norm`), so a seed fixes every
initial value. Weight decay applies to weight matrices only (not embeddings, gains or leaks).

**Shapes through one S block** (`n = B·T`):

```
x        [n, d]
h        = rmsnorm(x)            [n, d]
cur      = h · Win               [n, d]     → reshape [B, T, d]
s        = lif_scan(cur, β)      [B, T, d]  (0/1)   → reshape [n, d]
x       += s · Wout              [n, d]
h        = rmsnorm(x)            [n, d]
u        = h · W1                [n, 4d]
a        = relu(kwta(u, d/2))    [n, 4d]    (at most 1/8 non-zero)
x       += a · W2                [n, d]
...
logits   = rmsnorm(x) · Eᵀ       [n, V]
```

Attention (T, H): `q, k, v = h·W` reshaped `[B, T, nh, 32]` and transposed to `[B, nh, T, 32]`;
scores `q·kᵀ [B, nh, T, T]`, scaled, causal softmax, `·v`, transposed back and reshaped to
`[n, d]`, then `·Wo`.

**Weight counts** (`V = 4096`; the `+2d`, `+3d`, `+4d` are norm gains and leaks, the final `+d`
the last norm):

| Model | Formula | d64 L2 | d96 L3 | d128 L4 |
|---|---|---|---|---|
| T | `V·d + ctx·d + L·(12d² + 2d) + d` | 368,960 | 737,952 | 1,328,256 |
| S | `V·d + L·(10d² + 3d) + d` | 344,512 | 670,656 | 1,181,312 |
| H | `V·d + ctx·d + L·(14d² + 4d) + d` | 385,600 | 793,824 | 1,460,352 |

Non-embedding (without `tok_emb` and `pos_emb`): T 98,624 / 332,448 / 787,584; S 82,368 /
277,440 / 657,024; H 115,264 / 388,320 / 919,680. The release model Ava-S (d 192, L 6) has
3,001,920 weights, 2,215,488 non-embedding, by the same formula.

## 6. Training step and evaluation

**One step** (`trainer.rs`), step `s` from 0: `lr = cosine_lr(s, warmup, total_steps, lr_max,
lr_max · min_lr_frac)` (linear warm-up: step 0 gets `lr_max/warmup`, step `warmup−1` gets
`lr_max`; then cosine to `min_lr` at `total_steps`); forward → mean cross-entropy in nats
(`IGNORE` targets skipped) → backward → accumulate → `clip_grad_norm` (returns the norm before
clipping) → AdamW → zero grads.

**AdamW** per element, `t` the step count: `m = β₁m + (1−β₁)g`; `v = β₂v + (1−β₂)g²`; if the
tensor decays, `w −= lr·wd·w`; `w −= lr·(m/(1−β₁ᵗ)) / (√(v/(1−β₂ᵗ)) + ε)`. β₁ 0.9, β₂ 0.95,
ε 1e-8.

**Bits per byte** (`eval.rs`):

```
BPB = (Σ cross-entropy in nats over scored targets) / ln 2 / (Σ bytes of those target tokens)
```

A token's bytes are its BPE bytes; EOS counts as 1 byte (the newline between documents); PAD
targets are skipped. The per-row cross-entropy is computed in f64 from the logits. Evaluation
reads `iter_sequential(ctx)` windows of the val shards in order and stops exactly at
`--eval-tokens` scored targets; every batch is independent, so the result does not depend on the
batch size beyond float rounding. Activity: mean spike rate per SpikeMem and the kWTA active
fraction before and after the ReLU, averaged over scored positions.

## 7. The `train` binary

```
train --model T|S|H --d D --layers L --lr LR --out DIR [--ctx 128] [--batch 32] [--tokens 4M]
      [--warmup 50] [--min-lr-frac 0.1] [--wd 0.1] [--clip 1.0] [--seed 1] [--data-seed 1]
      [--threads 4] [--eval-tokens 500000] [--log-every 50] [--max-steps N] [--bench-steps N]
      [--train DIR] [--val DIR] [--bpe FILE]
```

- `steps = tokens / (batch · ctx)`, rounded down; the schedule spans `steps`. `--max-steps N`
  stops early without changing the schedule, so a truncated run is exactly the first N steps of
  the full one.
- Data: `Sampler::new(train, ctx, data_seed)`, `batch` windows per step. Defaults point to
  `data/nn/shards-4096/{train,val}` and `data/nn/bpe-4096.txt` under the repository root.
- `DIR/log.tsv` gets every step: step, loss, lr, grad_norm, tokens_per_sec, fwd_s, bwd_s,
  opt_s, data_s.
- At the end: validation, one line appended to `DIR/result.tsv`, checkpoint `DIR/model.awts`.
- `result.tsv` columns: model, d, layers, ctx, batch, tokens, steps, steps_run, lr, warmup,
  min_lr_frac, wd, clip, seed, data_seed, threads, eval_tokens, weights_total,
  weights_nonembed, train_loss_last50, val_loss_nats_per_token, val_bpb, val_tokens_scored,
  val_bytes, spike_rates, kwta_active, kwta_active_post_relu, train_seconds, eval_seconds,
  seconds, tokens_per_sec, peak_rss_mib.
- `--bench-steps N` runs N steps, prints throughput and the time split, writes nothing.
- Threads: `--threads` (default 4) for the step; `AVA_THREADS` for code that calls the kernels
  directly; both clamped to 1..=32.

## 8. v32 in full

Setup for every run: vocab 4096, ctx 128, batch 32, 4M tokens (976 steps of 4,096 tokens =
3,997,696), lr 3e-3, warm-up 50, min-lr-frac 0.1, wd 0.1, clip 1, init seed 1, 4 threads, val
500,000 targets = 1,521,140 bytes. The two extra T-small runs keep init seed 1 and change the
**data seed** (2 and 3). Run folders are git-ignored; the numbers below are the `result.tsv`
lines (s/step = train seconds / 976).

| Run | Weights | Non-emb | Val BPB | Val nats/token | Train loss (last 50) | s/step | Tokens/s | Train s | Eval s | Peak RSS MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| T d64 L2 | 368,960 | 98,624 | 2.381829 | 5.022676 | 5.014067 | 0.176 | 23,283 | 171.7 | 13.7 | 389 |
| T d64 L2, data seed 2 | 368,960 | 98,624 | 2.391502 | 5.043074 | 5.033074 | 0.178 | 23,057 | 173.4 | 13.5 | 389 |
| T d64 L2, data seed 3 | 368,960 | 98,624 | 2.395760 | 5.052053 | 5.042100 | 0.172 | 23,776 | 168.1 | 12.8 | 389 |
| T d96 L3 | 737,952 | 332,448 | 2.329692 | 4.912733 | 4.902766 | 0.324 | 12,657 | 315.8 | 21.0 | 492 |
| T d128 L4 | 1,328,256 | 787,584 | 2.282057 | 4.812282 | 4.801822 | 0.552 | 7,419 | 538.8 | 31.4 | 639 |
| S d64 L2 | 344,512 | 82,368 | 2.356707 | 4.969702 | 4.956031 | 0.140 | 29,211 | 136.9 | 12.6 | 353 |
| S d96 L3 | 670,656 | 277,440 | 2.315974 | 4.883804 | 4.872172 | 0.219 | 18,702 | 213.8 | 15.6 | 421 |
| S d128 L4 | 1,181,312 | 657,024 | 2.273935 | 4.795155 | 4.783949 | 0.325 | 12,585 | 317.7 | 20.8 | 515 |
| H d64 L2 | 385,600 | 115,264 | 2.359992 | 4.976628 | 4.966284 | 0.186 | 22,077 | 181.1 | 14.1 | 416 |
| H d96 L3 | 793,824 | 388,320 | 2.335049 | 4.924029 | 4.911840 | 0.316 | 12,959 | 308.5 | 20.1 | 552 |
| H d128 L4 | 1,460,352 | 919,680 | 2.297918 | 4.845730 | 4.832720 | 0.479 | 8,553 | 467.4 | 31.2 | 745 |

Seed spread of T d64 L2 over data seeds 1, 2, 3: 2.3818, 2.3915, 2.3958 BPB; sample standard
deviation about 0.007.

**Activity** (per layer, over scored positions). The kWTA active fraction before the ReLU is
0.125 everywhere by construction.

| Run | Spike rates | kWTA active after ReLU |
|---|---|---|
| S d64 L2 | 0.1719, 0.2423 | 0.1246, 0.1108 |
| S d96 L3 | 0.0971, 0.2291, 0.2202 | 0.1236, 0.0951, 0.1010 |
| S d128 L4 | 0.0709, 0.1951, 0.1841, 0.2000 | 0.1232, 0.0648, 0.0798, 0.0891 |
| H d64 L2 | 0.1102, 0.2847 | 0.1230, 0.1148 |
| H d96 L3 | 0.0467, 0.2156, 0.2282 | 0.1236, 0.1001, 0.0991 |
| H d128 L4 | 0.0386, 0.2434, 0.2165, 0.2042 | 0.1235, 0.0828, 0.0867, 0.0979 |

The first spiking layer fires least; deeper layers settle near a fifth of neurons per step.

## 9. Where the time goes (as of v32)

Mean seconds per step over the 976 steps of each `log.tsv`, 4 threads:

| Run | Forward | Backward | Optimiser | Total |
|---|---|---|---|---|
| S d64 L2 | 0.045 | 0.088 | 0.007 | 0.140 |
| S d96 L3 | 0.072 | 0.138 | 0.010 | 0.219 |
| S d128 L4 | 0.107 | 0.205 | 0.014 | 0.325 |
| T d64 L2 | 0.055 | 0.113 | 0.008 | 0.176 |
| T d96 L3 | 0.107 | 0.204 | 0.013 | 0.324 |
| T d128 L4 | 0.187 | 0.346 | 0.019 | 0.552 |
| H d128 L4 | 0.153 | 0.304 | 0.022 | 0.479 |

S is cheaper than T at every size because its blocks have no attention and its per-token memory
is a linear scan; the gap widens with size (T's attention core and GELU grow faster than S's
LIF scan and kWTA).

**Bottlenecks, as of v32.** Only `matmul`, `softmax_last`, `cross_entropy` and their backward
passes used threads. `relu`, `gelu`, `kwta`, `lif_scan`, `rmsnorm` and the elementwise ops ran
on one thread, so T's GELU and S's kWTA were the largest single items after the matmuls, both
growing with size. At the small sizes the tied-output `logits` matmul (`[n, d] × [d, 4096]`) is
the other large item.

**Since v32.** The release build threads these ops: unary ops through `par_chunks`, `rmsnorm`
and the softmax/cross-entropy rows through `par_rows`, kWTA through `kwta_forward_par`, and the
LIF scan through `lif_forward_par` / `lif_backward_par` (section 4). Each keeps the per-element
order of the single-threaded code, so results are unchanged. Throughput of the release build at
the release size (S, d 192, 6 layers, context 256, batch 32): about 7,600 tokens per second on 8 threads of an i9-14900F, 1.08 s per step of 8,192 tokens (forward 0.30 s, backward 0.72 s, optimiser 0.06 s), measured with `train --bench-steps 6` while another training run shared the machine. The forward split at that size: projection matmuls 45%, logits 15%, kWTA 14%, LIF 10%, RMSNorm 6%.

## 10. Determinism

- Same init seed → same initial tensors on every machine (splitmix64 + Box-Muller, fixed
  creation order).
- Same data seed → same window order (splitmix64 shuffles, one derived seed per epoch and
  shard).
- Kernels are bit-identical for any thread count (each output element is written by one thread
  in the same order); results can differ between CPUs with and without FMA.
- kWTA ties and sampling ties are broken by index.
- `tests/models.rs` checks that two training runs with the same seeds produce identical
  parameters, and that a checkpoint round-trips exactly; `tests/generate.rs` checks that a seed
  fixes a whole generated continuation.

## 11. Inference: `generate`, `talk`, `score`

All three share `nn/brain/src/generate.rs` and the flags
`[--model PATH.awts] [--bpe PATH.txt] [--arch S|T|H] [--d D] [--layers L] [--ctx C] [--threads N]`.

**Architecture inference.** The kind and size are read from the checkpoint's tensor names and
shapes: `d` from `tok_emb`, `L` from the number of `b{i}.` prefixes, S if `.mem.` tensors exist
and no `.attn.`, H if both, T if only `.attn.`; `ctx` from `pos_emb` when present, otherwise
(S) from `--ctx`, default 256. Flags may repeat these values but not contradict them.

**Sampling**, in this order: repetition penalty (a logit of a token present in the current
window is divided by the penalty when positive and multiplied when negative), temperature
(0 = greedy), top-k (0 = off), top-p (nucleus over the candidates top-k left; ≥ 1 = off), then
one draw from the splitmix `Rng`, so a seed fixes the whole continuation. Defaults:
temperature 0.8, top-k 40, top-p 0.95, repetition penalty 1.1, seed 1.

**Generation** runs the full forward over the last `ctx` tokens for every new token (the LIF op
carries no state between calls yet), and takes the logits of the last position. Generation stops
at EOS unless `--raw`. `talk` keeps a rolling buffer of the last `ctx` tokens, stops a reply at
EOS or the first token containing a newline, and advances the seed by one per turn so a
transcript replays identically.

**Scoring.** `score --text FILE` tokenises the file as one document (EOS appended, 1 byte), cuts
windows of `ctx + 1` with stride `ctx` exactly like `iter_sequential`, and reports the same bits
per byte that `train` reports for validation.

## 12. What is not here yet

Sections 1 to 13 describe the CPU engine and the GPU crate as they were at v0.1. Section 14 describes
v0.2: Ava's design with the fast-weight memory, the hand-written GPU training steps and the fused kernels.
bf16, stateful generation and resumable checkpoints now exist on the GPU. Still missing:

- data parallelism across GPUs;
- the new design on the CPU engine, which still has only T, S and H;
- the engram memory, sleep, reward learning and the self-model, which exist only in the first-life
  programs under `design-tests/examples/`; SCALING.md, stages 6 to 8, describes how they go on
  top of the trained brain.

## 13. The GPU crate (`nn/gpu`) at v0.1

This section is kept as it was written for v0.1; section 14 lists what changed.

`ava_gpu` is the Ava-S model written against [candle](https://github.com/huggingface/candle)
0.11 (`candle-core`, `candle-nn` with the `cuda` feature; `cudarc` only for `mem_probe` to read
the CUDA pool's counters) plus `ava_data` for the tokenizer and shards. It was written in one afternoon
to train Ava-M on one RTX 4070 (12 GB) and is the only crate in the repository with
dependencies. Files: `src/model.rs` (config, RMSNorm, the LIF scan, the kWTA mask, forward and
loss), `src/optim.rs` (AdamW, gradient norm, clipping, cosine schedule), `src/eval.rs` (bits per
byte with the same PAD-skip and EOS-is-one-byte rules as `nn/brain`), `src/bin/train_gpu.rs`,
`src/bin/generate_gpu.rs`, `src/bin/mem_probe.rs` (per-step live and reserved bytes of the
CUDA pool, for forward only, forward + backward, or full steps), `tests/smoke.rs`, and
`build.sh` with the build environment.

**Same maths, different realisation.**

- **Spikes.** The scan is a candle custom op (`src/lif.rs`) backed by two CUDA kernels
  (`src/lif.cu`, compiled to PTX by `build.rs` with `nvcc`). One thread per (batch row, channel)
  runs the whole time loop, so a layer costs one launch forward and one backward, and the loads
  are coalesced across channels. The backward recomputes the membrane, then walks back in time:
  `gv_t = gs_t / (1 + 4|v_t − θ|)² + β · gv_{t+1}`, `gI_t = gv_t`, and `gβ += gv_t · v_{t−1}`,
  summed over the batch outside the kernel so the reduction is deterministic; the reset is
  detached. The forward uses round-to-nearest intrinsics in the same order as the reference, so
  no FMA contraction can move a spike. `model::lif_scan_unrolled` keeps the first version, a
  straight-through estimator built from standard ops (`sg = u / (1 + 4|u|)`,
  `s = sg + (H(u) − sg).detach()`), and `tests/smoke.rs` checks that the fused op gives its spikes
  exactly and its gradients to 1e-4, on the CPU and on CUDA.
- **kWTA** keeps `u >= k-th largest` per row, so exact ties beyond `k` are all kept where the
  CPU op keeps the lowest index. Irrelevant for float activations in practice.
- **Initialisation** uses candle's device RNG with the same distributions (N(0, 0.02);
  `Win` N(0, 1/√d); residual projections 0.02/√(2L); norms 1; β = logit(0.9)), so a given seed
  does not give the same numbers as `ava_grad`. On the CPU backend candle 0.11 cannot be seeded;
  the trainer warns and continues.
- **Optimiser.** AdamW on candle tensors with the same decoupled decay, bias correction and
  decay set (`win`, `wout`, `w1`, `w2` only); clipping is applied as a scale `clip / (‖g‖ + 1e-6)`
  inside the step. The moments are `Var`s written in place every step, like the weights, and
  the gradient is detached before it enters them. Two bugs taught this. The first version kept
  the step graph alive through the moments. The second made fresh moment tensors every step;
  they outlived the step's transient allocations and pinned chunks of CUDA's stream-ordered
  pool, so the pool's reserved memory grew about 17 MB per step while its live memory stayed
  flat, until the card filled and steps slowed from 1.7 s to 10 s (the first Ava-M run, stopped
  at step 808). `mem_probe full` shows the pool flat at 160 MB with the fix.
- **Determinism.** Unlike the CPU engine, the GPU path is not bit-reproducible: CUDA reductions
  change the last bits from run to run, so the same seeds give losses that differ in the third
  or fourth decimal after a few steps.
- **Precision.** f32 throughout; the per-row validation NLL is accumulated in f64 on the device.
- **Checkpoints** are safetensors from candle's `VarMap` (`tok_emb`, `b{i}.mem.{norm,win,beta,wout}`,
  `b{i}.mlp.{norm,w1,w2}`, `final.norm`), written every 500 steps and at the end. `generate_gpu`
  samples from them (temperature, top-k, seed; it re-runs the window for every token, like
  `generate`).

**Where the time goes.** With the scan unrolled, a step at d 384, L 8, ctx 256, batch 32 took
1.36 to 1.45 s, whatever the batch between 8 and 32: about 75,000 small kernel launches per step,
the matmuls a small part, and about 9 GB of the card in use. The fused kernel takes the same step
in 0.43 s (Ava-M: 2,929 steps in 1,268 s, 18,946 tokens per second), 3.2 times faster. Nsight
Systems over 12 steps now puts the two LIF kernels below 1% of GPU time each. The time goes to
sum reductions (33%, mostly the backward of the broadcast multiplies in RMSNorm), the kWTA sort
(18%), elementwise ops (about 27%) and the matmuls (about 13%). A top-k kernel for kWTA that
does not sort the whole row, and a fused RMSNorm with its backward, are the next speed-ups.

**Memory.** `mem_probe` at d 384, L 8, ctx 256, batch 32: 153 MB live between steps (weights and
the two AdamW moments), a peak of 3.4 GB of live tensors in the forward pass and 6.0 GB in the
backward. Most of it is the sparse layer's `[B·T, 4d]` tensors (the widened input, the kWTA mask,
the masked and rectified copies), which the graph keeps for the backward, eight layers deep; the
fused LIF scan holds only its input and its spikes. Leave about 6.5 GB of the card free for this
size, counting the CUDA context. On Windows, running out of it does not fail fast: the driver
pages for minutes before the step returns `CUDA_ERROR_OUT_OF_MEMORY`, and the dying process can
keep its memory until it is terminated through WMI. A fused kWTA (mask, ReLU and their backward
in one op) or activation checkpointing is the way to a bigger batch on the same card.

**Tests.** `./build.sh test --release --test smoke` (set `AVA_GPU_TEST_CUDA=1` to run them on
the GPU as well): loss at init ≈ ln 4096, finite non-zero gradients for every parameter, the
decay set, 30 AdamW steps on one batch driving the loss from 8.3 to about 1.2, the LIF forward
and backward identities, the fused LIF op against the unrolled reference, the kWTA mask, the schedule arithmetic, and three AdamW steps against
the formula in f64.

## 14. v0.2: Ava's design on the GPU

### The models

`src/model.rs` holds every design that was tested, picked by `--model`:

| Kind | What it is | Status |
|---|---|---|
| `D` | **Ava**: spiking memory, then the gated delta-rule fast-weight memory, then the kWTA sparse MLP, in every block | the release design |
| `S` | spiking memory and the kWTA sparse MLP (Ava-S, Ava-M, Ava-M2) | v0.1's design; loses ground to T (v34) |
| `A` | `D`, with causal softmax attention instead of the fast weights in every fourth block counted back from the last | worse than D at both sizes tested (v36) |
| `T` | the yardstick transformer: attention, GELU MLP, a learned position table | the yardstick |
| `F`, `G`, `R` | the v35 steps toward `D`: Hebbian fast weights, then learned gates, then unit q/k with an output gate | kept for the record |

The fast-weight block of `D` (one per block, `H = d / 32` heads of width 32; the README gives the
equations) has these tensors: `b{i}.hebb.{norm, wq, wk, wv, wg, wo, wa, ba, wb, bb}`. The per-head
offsets `logit γ_h`, with `γ_h = 1 − 2^−(5+h)`, are constants, not weights. `wa`, `ba`, `wb` and `bb`
start at 0, so at initialisation each head forgets at exactly `γ_h` and writes with `b = 1`. The
spiking memory and the MLP have the same tensors as in v0.1.

### The training step

For `S`, `T`, `D` and `A`, `Model::loss_and_grads_*` computes the loss and every gradient by hand,
without candle's autograd. Each step saves what its backward needs and calls the kernels below. The
matmuls take bf16 operands with f32 accumulation (cuBLAS `gemm_ex`); everything else is f32.
`AVA_BF16=0` gives f32 matmuls and `AVA_MANUAL_DA=0` returns `D` and `A` to autograd.

The kernels, all in `src/lif.cu` and wrapped in `src/fused.rs`:

- **Spiking memory.** `lif_fwd` and `lif_bwd`, one thread per (row, channel) over the whole time loop,
  as in v0.1, with bf16-writing variants.
- **Fast-weight memory.**
  - `hebb_fwd` runs the whole block after the input projection, one warp per (row, head). The
    state `F` sits in registers, 32 floats per lane. The kernel does the unit q/k, the gates, the
    delta-rule update, the read, the head norm and the silu gate. It saves `F` every 16 tokens.
  - `hebb_bwd` walks the chunks from the last. It recomputes each chunk's states from its checkpoint
    into a per-warp scratch (the same arithmetic, so the same values), then walks back through the
    chunk. It prefetches the next token's inputs.
  - The block's six input projections are one matmul against `[Wq | Wk | Wv | Wg | Wa | Wb]`.
- **Sparse code.** `kwta_relu_rows` is a radix select of the k-th largest value per row, with the mask
  and the ReLU in the same pass; a warp walks the histogram. `relu_mask_bwd` is its backward.
- **RMSNorm.** `rms_fwd`, `rms_bwd_rows` and `rms_bwd_gain`.
- **Attention** (`T` and `A`). Tensor-core kernels `attn_fwd_tc`, `attn_bwd_q_tc` and `attn_bwd_kv_tc`
  (bf16 inputs, f32 softmax, a saved log-sum-exp), with scalar f32 versions kept for the tests.
- **Loss and embedding.** `ce_fwd` and `ce_bwd` (fused cross-entropy), and `embed_bwd` (one block per
  vocabulary row, so each row's sum runs in a fixed order).
- **Optimiser.** `adamw_inplace` and `adamw_inplace_s` update each weight and its two moments in place,
  in one launch per tensor. The `_s` form reads the clip scale from the device.

**A loop that does not wait.** The gradient norm and the clip factor are computed on the device in
f64, with the same IEEE operations the host used before. Losses and norms are copied back in one go at
each log step, so the CPU prepares the next batch while the GPU works. `AVA_SYNC_LOOP=1` restores the
old loop, which gives byte-identical weights.

**Checkpoints.** Every 500 steps (`--ckpt-every`), `ckpt/` gets the weights, both AdamW moments and
the step count, the data cursor and the settings line. `--resume` continues from it. Sixty straight
steps and thirty steps plus a resumed thirty give identical losses.

**Generation.** `Model::stream` and `Model::step` carry each block's membranes, last spikes and fast
weights from token to token (A also keeps a key/value cache for its attention blocks). A token costs
one position of work, and the state has a fixed size: `L · (2d + H · 1024)` numbers. Token-by-token
logits match the full forward within 1e-3 (`fast_weight_carried_state_matches_the_full_forward`).

### How the speed-ups were checked

Every faster path was used only after passing the rule below, which was pre-registered for each
round:

- **Either** a full training run that matches the old path within 0.005 BPB, with every logged loss
  (every 200th step) within 0.05;
- **or** byte-identical weights after 100 steps (`cmp` of the two `model.safetensors`).

The full-run checks:

| Path | Check run | Old | New |
|---|---|---|---|
| bf16 matmuls (S and T) | S d128 L4 lr .004 | 2.0131 | 2.0102 |
| bf16 matmuls (S and T) | T d128 L4 lr .004 | 1.9594 | 1.9602 |
| Hand-written D/A step | A d128 L4 lr .004 | 1.882973 | 1.883378 |

These were byte-identical:

- checkpointed fast-weight states;
- the prefetch;
- in-place AdamW;
- the loop without waits;
- the warp kWTA select.

Two attempts failed or gave nothing and were not used: TF32 with the GLA path (−0.0061 BPB), and an
earlier kWTA rewrite.

### Speed and memory at the release shape

The release shape is d 384, L 8, ctx 256 and batch 32, on an RTX 4070 with the card to itself
(`--bench-steps 40`, first step excluded):

| Model | Tokens per second | Seconds per step |
|---|---|---|
| D (Ava) | 132,400 | 0.062 |
| A | 142,600 | 0.057 |
| S | 268,000 | 0.031 |

Before the chunked checkpoints, Nsight Systems put the fast-weight backward at 32% of D's kernel time;
after them, it was 22%. Matmuls take about 35%.

### Tests

`./build.sh test --release --test smoke`, with `AVA_GPU_TEST_CUDA=1` to run them on the GPU too.
The 24 tests include:

- every hand-written step against autograd in f32 (`hand_written_s_backward_matches_autograd`,
  `hand_written_t_backward_matches_autograd`, `hand_written_da_step_matches_autograd`);
- each fused kernel against the chain of standard ops it replaces;
- the delta rule against its token-by-token recurrence, with repeated keys;
- tensor-core attention against the f32 kernels;
- carried-state generation against the full forward;
- bf16 steps tracked against the f32 gradients.
