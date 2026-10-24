# Model card: Ava-S

Ava-S is the first released trained brain of [Ava](../../README.md): a spiking language model
in pure Rust, with leaky integrate-and-fire neurons trained by surrogate gradients and
k-winners-take-all sparse codes, no attention and no position embedding. It is a tiny research
model. Every number below is copied from the training run's `result.tsv`.

| | |
|---|---|
| Model | Ava-S (kind `S` in `design-tests/nn/brain`) |
| Release date | 2026-10-24 |
| Author | nochallenge |
| Licence | PolyForm Noncommercial 1.0.0 (source-available) |
| Files | `model.awts` (AVAWTS01 checkpoint), `bpe.txt` (`ava-bpe v1`, 4096 tokens), this card |
| Checkpoint sha256 | 910dfd3317b3dde9554ed86e5eef84d5950c5d378576821af7cd818de56463d3 |
| Tokenizer sha256 | b1ccf0b918a88f0f421d7f3bd91b56f911a4d9e47c246e58ddb685041850f49b |
| Source commit | the commit tagged `v0.1.0` |

## Architecture

| | |
|---|---|
| Width `d` | 192 |
| Layers `L` | 6 |
| Training context `ctx` | 256 (the weights do not depend on it; see Limits) |
| Vocabulary | 4096 byte-level BPE tokens (256 bytes, EOS, PAD, 3,838 merges) |
| Weights, total | 3,001,920 |
| Weights, non-embedding | 2,215,488 |
| Per block | spiking memory `Win [d,d]` → LIF → `Wout [d,d]`; sparse MLP `d → 4d`, kWTA keeps `4d/8` (96 of 768 units), ReLU, `4d → d` |
| Neurons | leaky integrate-and-fire, threshold 1, learnable leak per channel (initialised 0.9), soft reset by subtraction, hard 0/1 spikes |
| Surrogate gradient | fast sigmoid `1/(1 + 4|v − θ|)²`, reset detached |
| Norm and output | RMSNorm (eps 1e-5) pre-norm; tied output `logits = x · Eᵀ` |

Weights: `4096·d + L·(10d² + 3d) + d` = 786,432 + 6 · 369,216 + 192 = 3,001,920.

## Training

| | |
|---|---|
| Tokens trained | 48M |
| Steps | 5859 (batch 32 × ctx 256 tokens per step) |
| Peak learning rate | 0.002 (linear warm-up of 200 steps, cosine decay to 0.1 of the peak) |
| Optimiser | AdamW, β₁ 0.9, β₂ 0.95, ε 1e-8, weight decay 0.1 on weight matrices only, gradient clip 1 |
| Seeds | init 1, data 1 |
| Hardware | a home PC, 8 CPU threads, f32 |
| Training time | 2 h 06 min (7566 s), plus 111 s of evaluation |
| Throughput | 6343.3 tokens per second |
| Peak resident memory | 1433 MiB |
| Precision | f32 throughout |
| Software | `design-tests/nn/{data,grad,brain}`, Rust standard library only, no external crates |
| Command | `train --model S --d 192 --layers 6 --ctx 256 --batch 32 --tokens 48M --lr 0.002 --warmup 200 --min-lr-frac 0.1 --wd 0.1 --clip 1.0 --seed 1 --data-seed 1 --threads 8 --eval-tokens 1000000 --out runs/v33/ava-s` |

### Training data

English only. The corpus is the text Ava's first life read, tokenised once into AVATOK01 shards
with the 4096-token BPE:

- Tatoeba English sentences;
- the Cornell Movie-Dialogs Corpus;
- dialogue replies derived from the DailyDialog training split;
- the AI2 ARC science corpus (`ARC_Corpus.txt`).

About 521M BPE tokens are available in the shards; Ava-S read 48M of them, in the seeded
shuffled order of the streaming sampler. 0.5% of documents were routed to validation by the
deterministic split rule (`mix64(seed ^ i) % 1000 < 5`). **No test set of any benchmark is in
the training data**, and the test sets listed as fresh in SCALING.md were not opened. The data
files are not distributed with the model; each has its own licence (see README.md, Quick
start).

## Evaluation

| | |
|---|---|
| Validation bits per byte | 1.872216 |
| Validation loss | 3.960295 nats per token |
| Training loss, last 50 steps | 3.939180 |
| Spike rate per layer | 0.0571,0.1698,0.1379,0.1409,0.1362,0.1588 |
| kWTA active after ReLU, per layer | 0.0808,0.0296,0.0276,0.0377,0.0523,0.0789 |
| Scored | 1000000 target tokens of the validation shards, read in order |

**What bits per byte means.** The model assigns each next token a probability; the loss is the
negative log of that probability. Bits per byte divides the total loss in bits by the number of
bytes of the text predicted (EOS counts as one byte), so it does not depend on the tokenizer and
can be compared across models. Lower is better. For scale, in the pre-registered v32 comparison
on the same data, with 4M training tokens and context 128, the spiking model reached 2.357,
2.316 and 2.274 BPB at d64 L2, d96 L3 and d128 L4, against a plain transformer's 2.382, 2.330
and 2.282 (README.md, Results). Ava-S is larger and has read twelve times more text than those
runs, so its number is not directly comparable to them; it is reported for the record, and the
`score` binary gives the same quantity for any text file.

## Sample

The first two prompts tried, unselected; each continuation stops where the model emitted EOS
(her corpus is mostly one-line dialogue turns, so she tends to end after a sentence).

```
Once upon a time, they get to be very short.
The sun is toxin, the northeast, and in which it is not always so disturbed.
```

Sampled with `generate --temperature 0.8 --top-k 40 --seed 1`; the continuation is
deterministic for that seed.

## Intended use

- Research on spiking and sparse language models, surrogate-gradient training, and scaling
  brain-like designs.
- Education: a complete, dependency-free, readable training and inference stack in Rust.
- Any other **non-commercial** use permitted by the licence.

Not intended for: production systems, decision-making of any kind, factual question answering,
or any commercial use without a licence from the author.

## Limits

- Tiny. Three million weights trained on 48M tokens of a small, dialogue-heavy English corpus.
  It knows very little of the world and no other language.
- Completes text only. No instruction tuning, no chat format, no safety tuning.
- Will often produce nonsense, repeat itself and invent facts. Do not rely on anything it says.
- Trained at context 256. The spiking memory is recurrent and the weights do not depend on the
  context length, so longer prompts run, but the model was not trained on them and quality past
  256 tokens is untested. `generate` re-runs the window for every new token.
- CPU-trained at f32; outputs can differ in the last bits between CPUs with and without FMA.
- One run, one seed. The seed noise of this size has not been measured.
- The engram memory, sleep replay, reward learning and the abstaining self-model of Ava's first
  life are not part of this model yet.

## Licence

Ava-S, its weights and tokenizer are released under the
[PolyForm Noncommercial License 1.0.0](../../LICENSE.md): free for any non-commercial purpose;
commercial use needs a separate licence from the author (open an issue on the repository). Keep
the `Required Notice: Copyright (c) 2026 nochallenge` line with any copy you share.

## Citation

```bibtex
@software{ava2026,
  author = {nochallenge},
  title  = {Ava: a brain-like language model in pure Rust},
  year   = {2026},
  url    = {https://github.com/nochallenge/ava}
}
```
