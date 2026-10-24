# Model card: Ava-M

Ava-M is the larger of the two brains in [Ava](../../README.md)'s first release and the first
trained on a GPU: the same spiking design as [Ava-S](../ava-s/MODEL_CARD.md) (leaky
integrate-and-fire neurons trained by surrogate gradients, k-winners-take-all sparse codes, no
attention, no position embedding), four and a half times the weights, trained on one consumer
graphics card through the `design-tests/nn/gpu` crate (candle, CUDA). It is a tiny research
model.

| | |
|---|---|
| Model | Ava-M (the `nn/gpu` trainer; kind `S` in `nn/brain` terms) |
| Release date | 2026-10-24 |
| Author | nochallenge |
| Licence | PolyForm Noncommercial 1.0.0 (source-available) |
| Files | `model.safetensors` (candle VarMap), `bpe.txt` (`ava-bpe v1`, 4096 tokens), `config.txt`, `train_curve.tsv`, this card |
| Checkpoint sha256 | a7918292f2cdf55ceaa3c5eb4b40d9d4610c963605258e8b7122bd31d53d24e4 |
| Tokenizer sha256 | b1ccf0b918a88f0f421d7f3bd91b56f911a4d9e47c246e58ddb685041850f49b |
| Source commit | the commit tagged `v0.1.0` |

## Architecture

| | |
|---|---|
| Width `d` | 384 |
| Layers `L` | 8 |
| Training context `ctx` | 256 |
| Vocabulary | 4096 byte-level BPE tokens (256 bytes, EOS, PAD, 3,838 merges), the same file as Ava-S |
| Weights, total | 13,378,944 |
| Weights, non-embedding | 11,806,080 |
| Per block | spiking memory `win [d,d]` → LIF → `wout [d,d]`; sparse MLP `d → 4d`, kWTA keeps `4d/8` (192 of 1,536 units), ReLU, `4d → d` |
| Neurons | leaky integrate-and-fire, threshold 1, learnable leak per channel (initialised 0.9), soft reset by subtraction, hard 0/1 spikes (straight-through estimator with the fast-sigmoid surrogate `1/(1 + 4|v − θ|)²`, reset detached) |
| Norm and output | RMSNorm (eps 1e-5) pre-norm; tied output `logits = x · Eᵀ` |

Tensor names in the checkpoint: `tok_emb`, `b{i}.mem.{norm,win,beta,wout}`, `b{i}.mlp.{norm,w1,w2}`
for `i` in 0..8, `final.norm`. ARCHITECTURE.md, section 13, lists how this crate differs from
the from-scratch CPU engine (device RNG for the init, top-k mask ties, f32 only).

## Training

| | |
|---|---|
| Tokens trained | 24M (the schedule; one pass over the seeded shuffled shards, 4.6% of the available text) |
| Steps | 2929 (batch 32 × ctx 256 = 8,192 tokens per step) |
| Peak learning rate | 0.001 (linear warm-up of 100 steps, cosine decay to 0.1 of the peak) |
| Optimiser | AdamW, β₁ 0.9, β₂ 0.95, ε 1e-8, weight decay 0.1 on `win`, `wout`, `w1`, `w2` only, gradient clip 1 |
| Seeds | init 1 (CUDA RNG), data 1 |
| Hardware | one NVIDIA RTX 4070 (12 GB); peak 6.0 GB of live tensors per step, f32 |
| Training time | 21 min 07 s (1268 s), plus 15 s of evaluation |
| Throughput | 18,946 tokens per second, 0.43 s per step, with the fused LIF kernel |
| Software | `design-tests/nn/gpu` on candle 0.11 (CUDA 13.2, driver 610), `ava_data` for the shards |
| Command | `train_gpu --d 384 --layers 8 --ctx 256 --batch 32 --tokens 24M --lr 1e-3 --warmup 100 --min-lr-frac 0.1 --wd 0.1 --clip 1.0 --seed 1 --data-seed 1 --eval-tokens 500000 --log-every 50 --out runs/v34/ava-m` |

Pre-registered (binary hash, command and rule written down before the launch). Tokens per
weight: 1.8, far below the 20 or so a compute-optimal run would use; this is a short run by
design, and the model is under-trained.

**Every attempt, disclosed.** The same command was launched four times; the hyperparameters never
changed, and this is the only run that finished. (1) 2026-10-23: stopped at step 808 when the
step time rose from 1.7 s to 10 s; the cause was AdamW allocating fresh moment tensors every step,
which fragmented the CUDA memory pool (fixed; see ARCHITECTURE.md, section 13). (2) 2026-10-24,
with that fix: memory flat, stopped by hand at step 435 to make way for the fused LIF kernel.
(3) With the fused kernel: out of GPU memory at step 57, when other programs on the desktop took
the card's headroom. (4) This run. No result of a stopped run was used to choose anything.

### Training data

The same shards as Ava-S: Tatoeba English sentences, the Cornell Movie-Dialogs Corpus, dialogue
replies derived from the DailyDialog training split, and the AI2 ARC science corpus; English
only; 0.5% of documents routed to validation by the deterministic split rule. No benchmark test
set is in the training data. The data files are not distributed; see README.md, Quick start.

## Evaluation

| | |
|---|---|
| Validation bits per byte | **1.888** (1.888163) |
| Validation loss | 3.981660 nats per token |
| Training loss, last 50 steps | 3.960366 |
| Spike rate per layer (last step) | 0.1061, 0.0997, 0.0742, 0.0769, 0.0659, 0.0646, 0.0812, 0.0917 |
| Scored | 500,000 target tokens (1,521,140 bytes) of the validation shards, read in order |
| Ava-S, for comparison | 1.872 bits per byte (3.0M weights, 48M tokens, CPU; scored on the first 1,000,000 target tokens of the same shards) |

Bits per byte is the total loss in bits divided by the bytes of the text predicted (EOS counts
as one byte); lower is better; it does not depend on the tokenizer. Ava-M has four and a half
times Ava-S's weights and scores slightly worse, which is what the budgets predict: Ava-S read
twice the text at a fifth of the size (16 tokens per weight against 1.8). The comparison says
something about two short runs, not about scaling. The fair scaling test is the first item on the
roadmap.

## Sample

The same two prompts as Ava-S, unselected. `generate_gpu` writes EOS as a line break and keeps
going, so each sample runs on for several lines.

```
Once upon a time, we believe that it would seem to be able to know what the same aspects of our own.
The maximum therapolation of the gametes is produced from the foliage to another.
You could have a good for you.
The name was also a special form of the 18th century that there are many different parts that they are
```

```
The sun is to move off the water temperature for the soil surface.
College: Neta, 1988.
The spore of the Great Horn-mithromatus and Vesiculanata.
A good thing, that is called.
The trend of the Solidanians do not seem to be made as a
```

Sampled with `generate_gpu --tokens 80 --temperature 0.8 --top-k 40 --seed 1`.

## Intended use and limits

As for Ava-S: research on spiking and sparse language models and their scaling; education; any
other non-commercial use permitted by the licence. Not for production, decisions, facts, or any
commercial use without a licence from the author.

- Under-trained: 1.8 tokens per weight. Expect a worse model than its size suggests.
- Completes text only; no instruction tuning, no chat format, no safety tuning. It will produce
  nonsense, repeat itself and invent facts.
- Trained at context 256; the recurrent memory runs on longer prompts untested.
- `generate_gpu` re-runs the window for every token (no stateful LIF yet) and renders EOS as a
  newline.
- The checkpoint is candle safetensors, not the `AVAWTS01` format of the CPU crate; the CPU
  `generate` cannot load it yet. Run it with `generate_gpu` (CUDA, or `--cpu`).
- One run, one seed. The GPU path is not bit-reproducible: the same seeds give losses that differ
  in the last digits from run to run.

## Licence

Ava-M, its weights and tokenizer are released under the
[PolyForm Noncommercial License 1.0.0](../../LICENSE.md): free for any non-commercial purpose;
commercial use needs a separate licence from the author (open an issue on the repository). Keep
the `Required Notice: Copyright (c) 2026 nochallenge` line with any copy you share.
