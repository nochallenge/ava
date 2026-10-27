# Model card: Ava

Ava is the release model of [Ava](../../README.md) v0.2, trained as run Ava-B. Each of her blocks
has three parts:

- spiking neurons (leaky integrate-and-fire, trained by surrogate gradients);
- fast synapses that follow the gated delta rule, an error-correcting memory rewritten at every token;
- a k-winners-take-all sparse layer.

She has no attention and no position embedding. She was trained on one consumer graphics card with the
`design-tests/nn/gpu` crate (candle and hand-written CUDA kernels). She is a tiny research model.

| | |
|---|---|
| Model | Ava (run Ava-B; kind `D` in `train_gpu`) |
| Release date | 2026-10-26 |
| Author | nochallenge |
| Licence | PolyForm Noncommercial 1.0.0 (source-available) |
| Files | `model.safetensors` (candle VarMap, f32; a download on the v0.2.0 release page as `ava.safetensors`), `bpe.txt` (`ava-bpe v1`, 4096 tokens), `config.txt`, `result.tsv`, `train_curve.tsv`, this card, `SHA256SUMS` |
| Checkpoint sha256 | d32901b530b62e3325548b2896f246157385d12f9127eb2a47ec3f8a58592ad1 |
| Tokenizer sha256 | b1ccf0b918a88f0f421d7f3bd91b56f911a4d9e47c246e58ddb685041850f49b (the same file as Ava-S and Ava-M) |
| Source commit | the commit tagged `v0.2.0` |

## Architecture

| | |
|---|---|
| Width `d` | 384 |
| Layers `L` | 8 |
| Memory heads per block | 12, each 32 wide |
| Training context `ctx` | 256 |
| Vocabulary | 4096 byte-level BPE tokens (256 bytes, EOS, PAD, 3,838 merges) |
| Weights, total | 19,354,176 |
| Weights, non-embedding | 17,781,312 |
| Per block | (1) spiking memory: RMSNorm, `win [d,d]`, LIF, `wout [d,d]`; (2) fast-weight memory: RMSNorm, `[wq, wk, wv, wg] [d,d]` and gates `wa, wb [d,12]` with biases, 12 heads of the gated delta rule, head norm, silu output gate, `wo [d,d]`; (3) sparse MLP: RMSNorm, `d → 4d`, kWTA keeps `4d/8` (192 of 1,536 units), ReLU, `4d → d` |
| Neurons | leaky integrate-and-fire, threshold 1, learnable leak per channel (initialised 0.9), soft reset by subtraction, hard 0/1 spikes, fast-sigmoid surrogate `1/(1 + 4|v − θ|)²`, reset detached |
| Fast synapses | per head `F [32, 32]`: `F ← α F + b k̂ (v − α Fᵀ k̂)ᵀ`, read `y = Fᵀ q̂`. `α = sigmoid(x·wa + ba + logit γ_h)` with `γ_h = 1 − 2^−(5+h)`, and `b = 2 sigmoid(x·wb + bb)` |
| Norm and output | RMSNorm (eps 1e-5) pre-norm; tied output `logits = x · Eᵀ` |
| Carried state | 104,448 numbers (408 KB in f32) for any length of text |

Tensor names: `tok_emb`; `b{i}.mem.{norm, win, beta, wout}`; `b{i}.hebb.{norm, wq, wk, wv, wg, wo,
wa, ba, wb, bb}`; `b{i}.mlp.{norm, w1, w2}`, for `i` in 0..8; and `final.norm`.

## Training

| | |
|---|---|
| Tokens trained | 387,083,520 (20 per weight; about three quarters of one pass over her 519M-token training shards) |
| Steps | 47,251 (batch 32 × ctx 256 = 8,192 tokens per step) |
| Peak learning rate | 0.001 (linear warm-up of 200 steps, cosine decay to 0.1 of the peak) |
| Optimiser | AdamW, β₁ 0.9, β₂ 0.95, ε 1e-8; weight decay 0.1 on `win, wout, wq, wk, wv, wo, w1, w2` only; gradient clip 1 |
| Seeds | init 1 (CUDA RNG), data 1 |
| Precision | bf16 matmul operands with f32 accumulation; everything else f32 |
| Hardware | one NVIDIA RTX 4070 (12 GB) |
| Training time | 44 min 22 s (2,662 s), plus 7 s of evaluation |
| Throughput | 145,397 tokens per second over the whole run, checkpoint saves included (0.056 s per step) |
| Software | `design-tests/nn/gpu` on candle 0.11 (CUDA 13.2); binary `train_gpu_b` (sha256 prefix eeb4187bf04863ab) |
| Command | `train_gpu --model D --d 384 --layers 8 --ctx 256 --batch 32 --tokens 387083520 --lr 0.001 --warmup 200 --min-lr-frac 0.1 --wd 0.1 --clip 1.0 --seed 1 --data-seed 1 --eval-tokens 1000000 --log-every 500 --out runs/v36/ava-b-D` |

**Pre-registered.** The design, size, budget, learning rate, binary and yardsticks were written down
before the launch, and the run was made once.

- **The design.** D was chosen by the author after a pre-registered test against model A (the same
  design with attention in every fourth block). D was better at both sizes tried (d 128: 1.875 against
  1.883; d 192: 1.707 against 1.719).
- **The learning rate.** 0.001 was D's best at d 192, the largest size tried; 0.002 gave a tie.

**Disclosed.**

- The training binary is the result of thirteen speed rounds. Each one was checked before use: by a
  full run within 0.005 BPB, or by byte-identical weights.
- The `tokens_per_sec` field of `result.tsv` counts the checkpoint saves as training time. The log
  lines on every 500th step also showed a low speed for that step, because the save fell inside the
  timer. This affects the printed speed only; the training is unchanged. The fix (the timer restarts
  after the save) is in the source but not in this run's binary.

### Training data

The same shards as Ava-S and Ava-M: English only, 0.5% of documents routed to validation by the
deterministic split rule.

- Tatoeba English sentences;
- the Cornell Movie-Dialogs Corpus;
- dialogue replies derived from the DailyDialog training split;
- the AI2 ARC science corpus.

No benchmark test set is in the training data. The data files are not distributed; see README.md,
Quick start.

## Evaluation

| | |
|---|---|
| Validation bits per byte | **1.487** (1.487072) |
| Validation loss | 3.145600 nats per token |
| Training loss, last 50 steps | 3.143049 |
| Scored | 1,000,000 target tokens (3,051,730 bytes) of the validation shards, read in order |
| Spike rate per layer (last step) | 0.0026, 0.0099, 0.0070, 0.0088, 0.0059, 0.0221, 0.0198, 0.0400 |
| BLiMP minimal pairs | **76.1%** (50,977 of 67,000; morphology 90.7%, syntax 70.8%, semantics 69.0%; scored in 14 s on the GPU. This is a practice look: this set was used in earlier versions, so it is spent and was not used to choose anything here) |

| For comparison (same 1M validation tokens) | Bits per byte |
|---|---|
| Ava-M2, plain spiking design, 13.4M weights, 268M tokens | 1.642 |
| The transformer's fitted line from v34 at Ava's non-embedding size (an extrapolation: the largest transformer trained had 6.3M non-embedding weights) | 1.449 |

Bits per byte is the total loss in bits divided by the bytes of the text predicted (EOS counts as one
byte). Lower is better, and it does not depend on the tokenizer.

**Against Ava-M2.** Ava is 0.155 BPB better than Ava-M2, the plain spiking model of the same width and
depth, which had fewer weights and read less text.

**Against the transformer.** She is 0.038 BPB behind the transformer's fitted line at her size. Her
gap to that line was:

| Non-embedding weights | Gap |
|---|---|
| 0.99M (d 128) | −0.049 |
| 3.3M (d 192) | −0.018 |
| 17.8M (d 384) | +0.038 |

So her design loses ground with size, and a straight-line fit of the gaps crosses zero at about 5M
weights. The pre-registered two-point guess for her was about 1.476; she scored 1.487.

Caveats:

- No transformer was trained at her size; the line is extrapolated from five transformers of up to
  6.3M non-embedding weights.
- Her learning rate was taken from d 192, not swept at d 384.

## Samples

Unselected, with seed 1. `generate_gpu` writes EOS as a line break and keeps going.

```
Once upon a time, it must be taken off-departure, and we have to leave with them all.
Ziri was trying to figure out what.
[22] In the same way, Gould's ideas may be explained on the basis of the theory of evolution [34].
THE STARTEB AND MAGNOM
```

```
The sun is at 50° latitude and 50° latitude.
Land and Animals of the World
This can be obtained by adding the two groups together, for example, in an overall pattern where all the genes are linked together in a complex matching selection process.
The first part of the human species has been a partial symb
```

```
My name is Ava and is my God.
He died on 24 September 1962 .
The first time I saw him was my mother there, the girl I had a friend in his car and my mother told me she was a caricature.
What is the role of AGF?
P. p. papillae frozen.
If your dog
```

```
Scientists discovered that a male asteroid (discovered as a match between the two groups of Taurus) was a small, bright-cotton star.
In this sense, I can find out what's left of the game; I'm doing a lot of work around in the game, and it's not even a good thing but a big part of me that'
```

On the CPU (`--cpu`) she generated about 205 new tokens per second with her state carried from token
to token.

Sampled with `generate_gpu --arch D --d 384 --layers 8 --cpu --tokens 80 --temperature 0.8 --seed 1`
(top-k 40 by default), with her state carried from token to token.

## Intended use and limits

Use her for research on brain-like, spiking and recurrent-memory language models and their scaling,
for education, and for any other non-commercial use the licence permits. Do not use her in
production, for decisions, for facts, or commercially without a licence from the author.

- She completes text only: no chat format and no safety tuning ([Ava-QA](../ava-qa/MODEL_CARD.md) is her
  version taught to answer questions). She will produce nonsense, repeat
  herself and invent facts.
- She was trained at context 256. Her memory runs on longer prompts, but that is untested.
- The checkpoint is candle safetensors. Run it with `generate_gpu --arch D` (CUDA, or `--cpu`). The
  CPU-only `generate` of `nn/brain` cannot load this design.
- One run, one seed. The GPU path is not bit-reproducible: the same seeds give losses that differ in
  the last digits from run to run.

## Licence

Ava, her weights and her tokenizer are released under the
[PolyForm Noncommercial License 1.0.0](../../LICENSE.md): free for any non-commercial purpose.
Commercial use needs a separate licence from the author (open an issue on the repository). Keep the
`Required Notice: Copyright (c) 2026 nochallenge` line with any copy you share.
