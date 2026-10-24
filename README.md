# Ava

Ava is a brain-like language model written in pure Rust by one person on one home PC. Her
neurons are leaky integrate-and-fire spiking neurons, trained end to end by gradient descent
through a surrogate gradient. Her hidden codes are sparse: in the wide layer of every block only
one unit in eight is allowed to fire. She has no attention and no position embedding; her memory
of the sequence is the membrane state of her neurons, so the cost of a token does not grow with
the length of the context. Everything from the byte-level tokenizer to the autodiff engine and
the optimiser is written from scratch: no PyTorch, no Python, no external crates in the CPU
training stack, only the Rust standard library. An optional GPU trainer runs the same model on
CUDA through the candle crate.

She is a **tiny research model**. She does not rival any modern assistant and will often produce
nonsense. What she offers is a design that is brain-like from the first line of code, built so
that every part of it can be scaled, and a method: every number below comes from a test that was
written down before it ran.

**What is in this release**

- **Ava-S**, the first released trained brain: a spiking model with d = 192, 6 layers,
  3,001,920 weights (2,215,488 outside the embedding), trained on 48M tokens of her reading on an
  8-thread CPU: 1.872 bits per byte on held-out text. Weights, tokenizer and a
  [model card](release/ava-s/MODEL_CARD.md) are in `release/ava-s/`.
- **Ava-M**, the first trained on a GPU: the same design with d = 384, 8 layers, 13,378,944
  weights (11,806,080 outside the embedding), trained on 24M tokens on one RTX 4070 through the
  candle crate: 1.888 bits per byte. Weights (safetensors), tokenizer and a
  [model card](release/ava-m/MODEL_CARD.md) are in `release/ava-m/`.
- The three dependency-free crates that train her, under `design-tests/nn/`: the tokenizer and
  token shards (`data`), the gradient engine with the two brain ops (`grad`), and the three models
  with the `train`, `generate`, `talk`, `score` and `profile` binaries (`brain`); plus `gpu`, the
  same spiking model on candle/CUDA with `train_gpu` and `generate_gpu`.
- Her first life (September 2026): a spiking network with no weight matrix and no gradient
  descent that learned word forms, arithmetic as habits, solved ARC puzzles and knew when to say
  "I don't know". It stays in the repository as a set of tools and as the yardstick.
- The pre-registered v32 comparison of her spiking design against a plain transformer at three
  sizes, with the raw result lines.
- [ARCHITECTURE.md](ARCHITECTURE.md), the technical deep-dive, and [SCALING.md](SCALING.md),
  the plan for taking her further.

## Why she is different

| | Ava-S | A standard transformer |
|---|---|---|
| Neurons | Leaky integrate-and-fire spiking neurons; hard 0/1 spikes; each channel has its own learnable leak | Dense real-valued activations (GELU) |
| Codes | k-winners-take-all: in the wide layer only one unit in eight stays active, then ReLU | Dense: every unit is active |
| Sequence memory | Recurrent membrane state carried from token to token; no position embedding; cost linear in the context | Attention over every earlier token; position embedding; cost quadratic in the context |
| Knowledge | Distributed over the trained weights, plus (first life) an engram memory of whole episodes replayed in sleep | Distributed over the trained weights |
| Honesty | Abstention is part of the design: her first-life faculties say "I don't know" and are scored right minus wrong; the same is planned for the trained brain | Not part of the architecture |
| Language | Rust; the CPU training stack uses the standard library only, zero dependencies; the optional CUDA trainer uses candle | Usually Python plus a framework |

## How she thinks

Ava-S is a stack of `L` identical blocks over a residual stream of width `d`. In words:

1. **Tokens to vectors.** A byte-level BPE tokenizer (4096 tokens: the 256 bytes, EOS, PAD and
   3,838 learned merges) turns the text into ids. Each id picks a row of the embedding table
   `E [4096, d]`. There is no position embedding.
2. **Per block, first half: spiking memory.** RMSNorm the stream, multiply by `Win [d, d]` to
   get an input current for each of `d` neurons, and run the neurons through time, one step per
   token. Each neuron integrates its current, leaks, and when its membrane crosses the threshold
   it emits a spike (a 1) and is reset. The spike train is multiplied by `Wout [d, d]` and added
   back to the stream. This is the only place where information moves between positions, and it
   moves forward in time only, through the membrane.
3. **Per block, second half: sparse code.** RMSNorm the stream, widen it to `4d` units with
   `W1`, keep only the `4d / 8` largest units (k-winners-take-all), zero the rest, apply ReLU,
   project back with `W2 [4d, d]` and add to the stream.
4. **Output.** A final RMSNorm, then logits over the 4096 tokens through the same table `E`
   (tied output: `logits = x · Eᵀ`).

The neurons are leaky integrate-and-fire units, simulated in discrete time with one step per
token. For one channel at step `t`, with `v_0 = 0` and `s_0 = 0`:

```
v_t = β · v_{t−1} + I_t − θ · s_{t−1}        membrane: leak, integrate, soft reset by subtraction
s_t = H(v_t − θ)                             spike: 1 if the membrane reached the threshold
β   = sigmoid(p),  p learnable per channel   leak, initialised so that β = 0.9
θ   = 1                                      threshold
```

`H` is the Heaviside step, so the spikes are exactly 0 or 1. The membrane `v` is what she
carries from token to token: a neuron with β near 1 remembers for a long time, one with β near 0
forgets at once, and training decides each channel's β.

## How she learns

A spike is a step function, and a step has zero gradient almost everywhere, so plain
backpropagation cannot train it. Ava uses a **surrogate gradient**: the forward pass keeps the
hard step, the backward pass pretends the step was a smooth curve. Her stand-in is the fast
sigmoid derivative `f'(x) = 1 / (1 + 4|x|)²` with `x = v − θ`, which is largest when the membrane
is close to the threshold and fades away from it. The gradient flows back through time inside
the LIF op (full backpropagation through time) and into the leak parameter of every channel. The
reset term is treated as a constant in the backward pass (the detached reset of Zenke & Vogels
2021 / SpyTorch), which is the standard choice.

The rest of her learning is deliberately plain:

- **AdamW** (β₁ 0.9, β₂ 0.95, ε 1e-8) with decoupled weight decay on the weight matrices only,
  never on embeddings, norm gains or leaks; global gradient-norm clipping.
- **Cosine learning rate** with a linear warm-up, decaying to a fraction of the peak (one tenth
  in every run so far).
- **Autodiff checked against finite differences.** Every differentiable op, and two composite
  nets (a tiny transformer block and a spiking block with kWTA + LIF), are checked by perturbing
  every parameter by ±1e-3 and comparing with the analytic gradient. The LIF backward is also
  worked out by hand for three time steps, with the reset detached and not detached.
- **Deterministic.** The kernels give bit-identical results for any thread count, the random
  source is splitmix64, and the data order is fixed by a seed, so a run is reproducible from its
  seeds and its build.

## How she stores memories

**Two levels, as in a brain.**

**(a) The trained weights** are her cortex: knowledge smeared across all of them. Nothing is
stored in one place; "cat" and "dog" end up with similar codes because they appear in similar
places, and a new sentence changes every weight a little. This is the level that the gradient
trains, and the level that scales with data and compute.

**(b) The engram memory** is her hippocampus, built in her first life
(`design-tests/examples/compose.rs`, with the design note in `design-tests/hippo/design.md`) and
kept as the design for the memory layer that goes on top of the trained brain next:

- **Episodes are kept whole.** Every lesson she is taught is stored as one engram, from the moment
  she hears it, beside whatever her cortex slowly learns from it.
- **Recall is by overlap.** A question reactivates each engram by the content they share, with
  rare words weighing more and no stop list; engrams compete, as neighbouring neurons do, so a
  weak or ambiguous cue reaches her answer with little weight.
- **Sleep replays.** The day's lessons wait in a log; sleep replays them into long-term memory
  with the same learning she uses awake, and a ledger makes each log replay exactly once.
- **Familiarity is a reading.** "Have I heard this kind of question before?" is measured and
  shown, never spoken as a canned phrase.

With a trained brain, the plan (SCALING.md, stages 6 and 7) is a fast memory with learned keys,
so recall works by meaning rather than shared words, and sleep replay that trains the slow
weights.

## How she scales up

Ava-S has four knobs: the width `d`, the depth `L`, the number of winners `k` in the sparse layer
(`k = 4d / 8` in every run so far) and the context `ctx` used in training. Her weights are:

| Part | Weights |
|---|---|
| Embedding (tied with the output) | `4096 · d` |
| Spiking memory, per block | `2d²` (`Win`, `Wout`) plus `d` leaks and `d` norm gains |
| Sparse MLP, per block | `8d²` (`W1`, `W2`) plus `d` norm gains |
| Final norm | `d` |
| Total | `4096·d + L·(10d² + 3d) + d` |

For example d = 128, L = 4 gives 1,181,312 weights, 657,024 of them outside the embedding (the
exact counts of the v32 run), and the release model's d = 192, L = 6 gives 3,001,920 and
2,215,488. Nothing in her weights depends on the context length: the spiking memory is
recurrent, so a longer context costs her time in proportion, not in square, and a checkpoint
trained at one context runs at another.

**Adding neurons.** Widening `d` adds neurons to every spiking layer and to every sparse layer at
once; deepening `L` adds whole blocks. Both are one command-line flag to `train`. The sparse
layer's `k` is fixed at one eighth of `4d`, so the share of active units stays the same as she
grows.

**Any amount of text.** The data crate tokenises text once into binary shards (`AVATOK01`), 50M
tokens each by default, and streams them: one shard in memory at a time, a seeded shuffle over
shards and within each shard, and a one-line cursor (seed, context, epoch, shard, slot) that
resumes on exactly the next window. Memory is bounded everywhere, and token counts are `u64`.
The corpus she reads today is small; the code places no limit on it.

**GPU.** The from-scratch engine is CPU f32 with a hand-written blocked matrix multiply on up to
32 threads. `design-tests/nn/gpu` is the same model on [candle](https://github.com/huggingface/candle)
with CUDA: the LIF scan is a fused CUDA kernel with a hand-written backward (one launch per
layer and direction, one thread per neuron running the whole time loop; the same forward and
the same surrogate gradient as the CPU op, checked against a reference built from standard
ops), kWTA is a top-k mask, and the optimiser is the same AdamW. It trained Ava-M on one
consumer card. The fused kernel made a step 3.2 times faster than the first version, which
unrolled the scan into about 75,000 small kernel launches per step; bf16 and data parallelism
across GPUs are next.

**The scaling test** is the method that matters most: train her design and the yardstick
transformer at three or more sizes on the same data, and compare the slopes. A gain at one size
is not a scaling result. The first run of this test is below; the fair version (tokens in
proportion to weights, a learning-rate sweep per size) is next. See [SCALING.md](SCALING.md).

## How she models the human brain

These are analogies, and some are loose. The table says which.

| Ava-S | Brain | How close |
|---|---|---|
| Leaky integrate-and-fire neurons with a threshold, a spike and a soft reset | Cortical neurons: a membrane that integrates input, leaks, fires an action potential and resets | Close in form. The real neuron has many more time scales, dendrites and a refractory period; hers has one state variable per channel and one step per token |
| A learnable leak `β` per channel | Membrane time constants, which differ by cell type | Loose. Hers is learned by gradient descent, not set by biophysics |
| Spikes of exactly 0 or 1 | All-or-none action potentials | Close. The surrogate gradient has no biological counterpart; it is a training device |
| k-winners-take-all in the wide layer | Lateral inhibition and the sparse codes of cortex, where few neurons are active at once | Fair. Real inhibition is itself done by neurons and is graded; hers is an exact top-k |
| The residual stream | No single counterpart; the nearest is a recurrent cortical column that every layer reads and writes | Loose |
| Weights trained by backpropagation | Synaptic plasticity | Loose. The brain does not backpropagate; the mechanism differs, even if the outcome (distributed knowledge) is alike |
| Engram memory and sleep replay (first life) | Hippocampal episodes and consolidation during sleep | Fair in function (episodes kept whole, recalled by partial cue, replayed), not in mechanism |
| Engram competition and pattern separation (first life) | Dentate gyrus and lateral inhibition in the hippocampus | Fair in function |
| Abstaining under right-minus-wrong scoring (first life) | Metacognition, the feeling of knowing | Fair in function |
| Attention | None claimed | Ava-S has no attention. The hybrid model H used it, and lost ground with size |

## Results

### v32: her spiking design against a transformer, at three sizes

Pre-registered, run once each. Three models on the same data, the same tokenizer (BPE 4096),
the same budget (4M tokens, context 128, batch 32, 976 steps), the same learning rate (3e-3,
cosine to 3e-4) and the same seed. The score is validation **bits per byte** (BPB) over 500,000
target tokens (1,521,140 bytes); lower is better.

- **T**: plain transformer, the yardstick (causal attention, GELU MLP, position embedding).
- **S**: Ava spiking (spiking memory + sparse MLP; no attention, no position embedding).
- **H**: hybrid (spiking memory + attention + sparse MLP).

| Size | Weights T / S / H | T | S | H | S lead over T |
|---|---|---|---|---|---|
| d 64, L 2 | 368,960 / 344,512 / 385,600 | 2.382 | **2.357** | 2.360 | 0.025 |
| d 96, L 3 | 737,952 / 670,656 / 793,824 | 2.330 | **2.316** | 2.335 | 0.014 |
| d 128, L 4 | 1,328,256 / 1,181,312 / 1,460,352 | 2.282 | **2.274** | 2.298 | 0.008 |

**Verdict.** S beat T at every size tested, with fewer weights and faster training (29,211,
18,702 and 12,585 tokens per second on 4 CPU threads against T's 23,283, 12,657 and 7,419). But
her lead shrank as size grew (0.025, then 0.014, then 0.008 BPB), and H went from better than T
to worse. By the rule written down in advance, both designs lose ground to the transformer with
size.

**Seed noise.** T at the smallest size was run with three data seeds: 2.382, 2.392 and 2.396
BPB, a standard deviation of about 0.007. S's lead at the largest size is about one such
deviation.

**Caveat.** Every size was trained on the same 4M tokens with one learning rate. Larger models
are under-trained and may want a different learning rate, so the slopes are not yet a fair
scaling result. The fair test (tokens proportional to weights, a learning-rate sweep) is the next
item on the roadmap. The full result lines, with throughput, memory and spike rates, are in
[ARCHITECTURE.md](ARCHITECTURE.md).

### Ava-S, the release model

| | |
|---|---|
| Model | S (spiking memory + sparse MLP; no attention, no position embedding) |
| Size | d 192, 6 layers, trained at context 256 |
| Weights | 3,001,920 total, 2,215,488 non-embedding |
| Training | 48M tokens of her reading, 8 CPU threads, f32 |
| Validation bits per byte | **1.872** (1,000,000 target tokens, 3,051,730 bytes) |
| Validation loss | 3.960 nats per token |
| Training time | 2 h 06 min at 6,343 tokens per second, peak memory 1.4 GB |
| Spike rate per layer | 0.057, 0.170, 0.138, 0.141, 0.136, 0.159 |
| Sample | "Once upon a time, they get to be very short." (seed 1) |

The full card, with the exact command, seeds, checkpoint hash and a sample, is
[release/ava-s/MODEL_CARD.md](release/ava-s/MODEL_CARD.md).

### Ava-M, the GPU-trained model

| | |
|---|---|
| Model | the same spiking design (S), trained by the candle/CUDA crate `design-tests/nn/gpu` |
| Size | d 384, 8 layers, trained at context 256 |
| Weights | 13,378,944 total, 11,806,080 non-embedding |
| Training | 24M tokens of her reading on one RTX 4070, f32: 21 min 07 s at 18,946 tokens per second |
| Validation bits per byte | 1.888 |
| Validation loss | 3.982 nats per token |
| Spike rate per layer | 0.106, 0.100, 0.074, 0.077, 0.066, 0.065, 0.081, 0.092 |
| Sample | "The sun is to move off the water temperature for the soil surface." (seed 1) |

Ava-M has four and a half times Ava-S's weights but read half the text (1.8 tokens per weight
against 16), so the two numbers compare two short budgets, not the designs; the fair
scaling test is the first item on the roadmap. The full card, with the exact command, seeds and
checkpoint hash, is [release/ava-m/MODEL_CARD.md](release/ava-m/MODEL_CARD.md).

### Her first life, measured

Count-and-search Ava, before she had a gradient. Each line is one pre-registered run; the
ARC-AGI-1 evaluation set and the BLiMP test have each been used three times over the versions,
so treat both as spent.

| Test | Ava | For reference |
|---|---|---|
| ARC-AGI-1 puzzles, evaluation set | 60 of 400 right (15%), 3 wrong, the rest "I don't know" | o3: 76-88% |
| ConceptARC puzzles (160) | 9 right, 21 wrong | Her self-check is overconfident when examples are few |
| BLiMP grammar (test pairs) | 56.2% | GPT-2: about 81%; chance: 50% |
| Zorro child-language grammar | 64.9% | chance: 50% |
| AI2 science questions (Test) | Easy 65.7%, Challenge 34.0% | Challenge was built so that word matching fails |
| English past tense, words never taught | 95.7% (irregular verbs 79%) | suffix analogy: 95.1% |
| Long addition and subtraction, learned as habits | 100% up to 12 digits | |

She may answer "I don't know". She is judged by right minus wrong, and by the share right among
the answers she gives. A confident wrong answer counts against her.

## Quick start

You need a recent stable Rust toolchain. The three brain crates have no dependencies and build
on their own; each is detached from any parent workspace.

**Build the brain**

```
cd design-tests/nn/brain
cargo build --release        # train, generate, talk, score, profile
```

**Run the released model** (from the repository root):

```
design-tests/nn/brain/target/release/generate --model release/ava-s/model.awts \
  --bpe release/ava-s/bpe.txt --prompt "Once upon a time" --tokens 200 \
  --temperature 0.8 --top-k 40
```

`generate` reads the architecture from the checkpoint (an S checkpoint has no position table, so
`--ctx` sets its window; default 256), streams the continuation to stdout and stops at EOS.
Other sampling flags: `--top-p 0.95`, `--repetition-penalty 1.1`, `--seed 1`, `--threads 4`;
the output is deterministic for a seed and any thread count.

**Talk to her**

```
design-tests/nn/brain/target/release/talk --model release/ava-s/model.awts --bpe release/ava-s/bpe.txt
```

A plain line REPL: each line you type is appended to a rolling buffer of the last `ctx` tokens
and she continues it (up to `--tokens 60` by default). `quit` or `exit` ends the session. She
completes text; there is no chat format, so expect her to ramble.

**Score a text file** in bits per byte, the same quantity the training run reports:

```
design-tests/nn/brain/target/release/score --text some.txt --model release/ava-s/model.awts --bpe release/ava-s/bpe.txt
```

**Run the GPU model, Ava-M.** The `gpu` crate needs the CUDA toolkit, a C++ compiler for its
kernels and a network connection for its crates (candle 0.11); `build.sh` sets the environment
that worked on the author's Windows machine (MSVC, CUDA 13.2, compute capability 8.9). Without
a GPU, `--cpu` runs the same code on the CPU through candle, slowly.

```
cd design-tests/nn/gpu
./build.sh build --release                       # or: cargo build --release with the env from build.sh
cd ../../..
design-tests/nn/gpu/target/release/generate_gpu --model release/ava-m/model.safetensors \
  --bpe release/ava-m/bpe.txt --d 384 --layers 8 --ctx 256 --prompt "Once upon a time" --tokens 100 \
  --temperature 0.8 --top-k 40 --seed 1
design-tests/nn/gpu/target/release/train_gpu --d 384 --layers 8 --ctx 256 --batch 32 --tokens 24M \
  --lr 1e-3 --warmup 100 --min-lr-frac 0.1 --wd 0.1 --clip 1.0 --seed 1 --data-seed 1 --out runs/my-ava-m
```

**Train your own.** `train` takes the model kind, size and budget on the command line. This is
the exact v32 command for the largest spiking run (4 threads; 318 s of training plus 21 s of
evaluation on the author's PC):

```
target/release/train --model S --d 128 --layers 4 --ctx 128 --batch 32 --tokens 4M \
  --lr 0.003 --warmup 50 --min-lr-frac 0.1 --wd 0.1 --clip 1.0 --seed 1 --data-seed 1 \
  --threads 4 --eval-tokens 500000 --out runs/my-S-large
```

It writes `log.tsv` (every step), `result.tsv` (one line: configuration, weights, validation
BPB, spike rates, timings) and `model.awts` (the checkpoint). `--model T` trains the yardstick
transformer and `--model H` the hybrid. By default it reads `data/nn/shards-4096/{train,val}`
and `data/nn/bpe-4096.txt` under the repository root; `--train`, `--val` and `--bpe` point it
elsewhere. `--max-steps N` stops early without changing the schedule; `--bench-steps N` times N
steps and writes nothing. `profile` reports where a step's time goes, op by op.

**Make shards from your own text**

```
cd design-tests/nn/data
cargo build --release        # bpe_train, make_shards, shard_stats

# 1. learn a tokenizer from a bounded sample of your text (one document per line)
target/release/bpe_train --vocab 4096 --out my-bpe.txt corpus_a.txt corpus_b.tsv@tab:2

# 2. stream the text through it into train/ and val/ shards (0.5% of documents to val by default)
target/release/make_shards --bpe my-bpe.txt --out my-shards corpus_a.txt corpus_b.tsv@tab:2

# 3. look at what you made
target/release/shard_stats my-shards --count-docs
```

An input is a file (`path`), a tab-separated column (`path@tab:N`) or a field split by any
string (`path@sep:STR:N`). Sources are interleaved so every shard holds the global mixture.
Digits are always single tokens, so arithmetic stays learnable.

**Her first life** builds separately: `cd design-tests && cargo build --release --examples`.
That older crate has one dependency (`rustyline`, for its line editor); the programs read
`../data/...` and are documented in their file headers.

**Data is not included.** The datasets have their own licences, and some have none. Fetch them
yourself and place them under `data/` (git-ignored):

| Folder or file | What it is | Source |
|---|---|---|
| `eng_sentences.tsv` | Tatoeba English sentences | tatoeba.org downloads |
| `cornell_lines.txt` | Cornell Movie-Dialogs Corpus | Cornell University |
| `triples.tsv`, `replies.tsv` | dialogue turns | believed to come from the DailyDialog train split; check before use |
| `ai2-arc/ARC-V1-Feb2018-2/` | AI2 Reasoning Challenge, with ARC_Corpus.txt | Allen Institute for AI |
| `unimorph-eng.tsv` | English word forms | UniMorph |
| `blimp/`, `zorro/`, `clams-en/`, `posh-bench/` | grammar test sets | see the file headers in `design-tests/examples/` |
| `arc-agi-1/`, `arc-agi-2/`, `conceptarc/`, `mini-arc/` | ARC puzzle sets | github.com/fchollet/ARC-AGI and related repositories |
| `openbookqa/` | OpenBookQA, V1 Sep 2018 | Allen Institute for AI |
| `nn/` | the BPE file and shards made by the commands above | built locally |

## Repository layout

```
design-tests/nn/data/     ava_data   byte-level BPE, AVATOK01 shards, streaming sampler, cursor
design-tests/nn/grad/     ava_grad   tensors, autodiff tape, LIF scan, kWTA, AdamW, AVAWTS01 checkpoints
design-tests/nn/brain/    ava_brain  the T / S / H models, trainer, bits per byte;
                                     train, generate, talk, score, profile
design-tests/nn/gpu/      ava_gpu    the spiking model on candle/CUDA: train_gpu, generate_gpu
                                     (the one crate with dependencies; see build.sh)
design-tests/src/         her first life: the event-driven spiking neuron, synapses, sparse codes,
                          allocators, readout, brain files (AVABRAIN)
design-tests/examples/    one program per faculty: compose.rs (talk, engrams, sleep), arc.rs (ARC
                          solver with a self-check), procedure.rs (arithmetic as habits, reward
                          learning), know_what.rs (the self-model that abstains), exam.rs (word
                          forms), blimp_open.rs and read_think.rs (count-based language models,
                          kept as yardsticks), paired_boot.rs (statistics)
design-tests/hippo/       the hippocampus design note
release/ava-s/            the released CPU-trained weights, tokenizer and model card
release/ava-m/            the released GPU-trained weights (safetensors), tokenizer and model card
ARCHITECTURE.md           the technical deep-dive
SCALING.md                the plan: what scales, what goes, the staged path, how to stay honest
CONTRIBUTING.md           issues welcome, forks welcome, no outside code for now
LICENSE.md                PolyForm Noncommercial 1.0.0
```

The `design-tests` name is historical: it was the scratch crate where the first ideas were
measured, and it grew into Ava herself. Code marked `TESTING / DESIGN ONLY` is experimental,
which is all of it. The top-level `src/` is a hello-world placeholder from the project's first
day.

## Honesty and method

- **Pre-registration.** Every test is written down before it runs: the build's hash, the
  switches or flags, the data and the pass rule. It runs once and the verdict is applied by the
  letter. The v32 comparison was pre-registered as three models at three sizes with a fixed
  budget.
- **Right minus wrong.** Where she may abstain, a confident wrong answer counts against her, and
  the share right among the answers she gives is reported beside the total.
- **Statistics.** A gain counts only when its paired bootstrap 95% interval lies above 0
  (20,000 resamples, seed 1). Seed noise is measured, not assumed.
- **Byte-identical when off.** Every new mechanism in her first life sits behind a switch that
  is off by default, and with the switch off the outputs match the saved reference files byte for
  byte. The CPU-trained brain is deterministic for a given seed, data seed and build; the GPU
  trainer is not bit-reproducible (CUDA reductions vary in the last bits).
- **Fresh test sets.** New ideas are developed on practice and dev data only. Each test set is
  used once, every earlier look is disclosed, and the sets that are still unopened are listed in
  SCALING.md.
- **No hand-written knowledge.** No word lists, canned phrases or per-puzzle code.

## Limits

- She is tiny: around a million weights in the v32 runs, three million in Ava-S, thirteen million
  in Ava-M. Modern language models have thousands of times more.
- She has read little: 4M tokens per v32 run, 48M for Ava-S and 24M for Ava-M, from a small,
  dialogue-heavy English corpus. She knows no other language and very little of the world.
- Ava-S was trained on a CPU at f32 with hand-written kernels; Ava-M on one consumer GPU at
  f32 through candle. There is no bf16 yet and no training across several GPUs.
- There is no instruction tuning, no chat format and no safety tuning. She completes text.
- She will produce nonsense often, repeat herself, and make things up. Do not rely on anything
  she says.
- The v32 scaling result is one run per size at a fixed budget; it is suggestive, not settled.
- The first-life faculties and the trained brain are not yet connected: her abstention, engram
  memory and arithmetic habits live in the older programs.

## Roadmap

1. **The fair scaling test.** Tokens in proportion to weights, a learning-rate sweep per size,
   three or more sizes, seeds measured. This decides whether the spiking design earns its place.
2. **GPU scale.** The LIF scan is fused; next come a top-k kWTA kernel, a fused RMSNorm, bf16,
   bigger batches, and data parallelism so she can train across several GPUs.
3. **Hippocampus.** The engram memory with learned keys on top of the trained brain, and sleep
   replay that writes episodes into the weights.
4. **Reward learning.** Learning from right and wrong alone, as her arithmetic habits do.
5. **The self-model.** A calibrated "I don't know" for the trained brain, scored right minus
   wrong.
6. **Speed and size.** A stateful LIF for generation (today every new token re-runs the window),
   then bigger models on more text.

## Licence

Ava is **source-available** under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): the
source is open to read, use, change and share for any non-commercial purpose; commercial use
needs a separate licence from the author.

- **Free for any non-commercial use**: research, study, teaching, hobby projects, and
  non-profit, educational and government organisations. You may change her, build on her and
  share your versions under the same terms, including scaling her up.
- **Commercial use needs a separate licence from the author.** To ask for one, open an issue on
  this repository.
- Keep the `Required Notice:` line from [LICENSE.md](LICENSE.md) with any copy you share.

Outside code contributions are not accepted for now; see [CONTRIBUTING.md](CONTRIBUTING.md).
Forks for non-commercial work are welcome.

## Citation

```bibtex
@software{ava2026,
  author = {nochallenge},
  title  = {Ava: a brain-like language model in pure Rust},
  year   = {2026},
  url    = {https://github.com/nochallenge/ava}
}
```
