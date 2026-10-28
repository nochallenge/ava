# Ava

Ava is a brain-like language model written in pure Rust by one person on one home PC. Each of her
blocks has three parts:

- **spiking neurons** (leaky integrate-and-fire) whose membranes carry the past from token to token;
- **fast synapses** that change while she reads: an error-correcting associative memory that writes
  only what it got wrong;
- **sparse codes**: in the wide layer, only one unit in eight may be active.

She has **no position embedding and almost no attention**: one attention read in one block, added in
v0.3, which her own tests show she does not use. A transformer remembers by looking back over every
earlier token. Ava remembers by changing her own state: the membranes and fast synapses. Her working
memory of a text is a fixed-size state (about 408 KB at the release size), so every new token costs
about the same however long the text is. And like a brain, she can **sleep**: she reads a day's new
facts, sleeps on them for half a minute, and knows them afterwards.

She is trained end to end by gradient descent through a surrogate gradient, on one consumer GPU, with
hand-written CUDA kernels. Everything from the byte-level tokenizer to the training step is written in
Rust: no PyTorch, no Python. The CPU engine has no dependencies at all. The GPU trainer uses the candle
crate for its tensors and its own kernels for everything brain-like.

She is a **tiny research model**. She does not rival any modern assistant and will often produce
nonsense. What she offers is a different design, built so that every part of it can be scaled, and a
method: every number below comes from a test that was written down before it ran.

**What is in this release (v0.3)**

- **Ava 0.3**, one model that reads, counts, reads whole books, answers questions and sleeps
  ([model card](release/ava-0.3/MODEL_CARD.md); weights on the
  [v0.3.0 release page](https://github.com/nochallenge/ava/releases/tag/v0.3.0)). Each test was
  pre-registered and run once on fresh data:
  - **numbers: +18 points** over the same recipe without this version's training (60.5% vs 42.4%);
    comparing numbers of different lengths 90.5% vs 46.5%. She sees how long a number is at a glance
    before reading its digits, as people do;
  - **sleep: 96.2% after vs 23.8% before** on questions about 150 new people she read about and then
    slept on (chance 25%);
  - **long books: her memory helps across 16,384 tokens** of unseen books (0.033 bits per byte better
    than a memory cut every 256 tokens);
  - **questions: 40.7% vs 33.0%** on 1,200 held-out science questions, 4 choices (SciQ 56.5%).

  **Synaptic consolidation** keeps sleep from costing her reading: the weights her reading leans on are
  held in place, so sleep cost +0.016 bits per byte instead of +0.046. Three tests failed and are
  reported as such: the new attention read is unused, reward learning added nothing measurable, and
  her reading (1.511) is slightly below v0.2's Ava (1.487).
- **The v0.2 models stay**: Ava (the reading model, 1.487 bits per byte) and Ava-QA (SciQ 56.4%), in
  `release/ava/` and `release/ava-qa/`.

**What came in v0.2**

- **Ava**, the v0.2 release model (trained as run Ava-B): d = 384, 8 layers, 19,354,176 weights
  (17,781,312 outside the embedding). She read 387M tokens (20 per weight) on one RTX 4070 in
  44 minutes, and scores **1.487 bits per byte** on held-out text. The two v0.1 models scored 1.872 and
  1.888. Her [model card](release/ava/MODEL_CARD.md), tokenizer and settings are in `release/ava/`;
  the weights are a download on the [v0.2.0 release page](https://github.com/nochallenge/ava/releases/tag/v0.2.0).
- **Ava-QA**, Ava taught to answer questions in 95 seconds. While she was taught she also re-read her
  ordinary text, as the brain replays old memories while it learns something new. On SciQ science
  questions (4 choices) she went from 41.1% to **56.4%** right, and to 66.2% with the passage shown.
  She may say "I don't know", and then 64% of the answers she gives are right. Replay cut her
  forgetting of plain reading by about three quarters (1.487 to 1.550 BPB, against 1.748 without
  it). [Model card](release/ava-qa/MODEL_CARD.md) in `release/ava-qa/`.
- **Three pre-registered tests** of her design against a transformer, all at equal tokens per weight:
  - the fair scaling test of the plain spiking design (v34);
  - the fast-weight memories (v35);
  - memory against memory-plus-attention (v36).
- **A much faster GPU trainer** (`design-tests/nn/gpu`). Plain spiking Ava at Ava-M's shape now trains
  at 268,000 tokens per second with the card to itself; v0.1's Ava-M run logged 18,946 (that run's load
  on the card was not recorded). The trainer now has:
  - a hand-written training step for each model, with no autograd in the hot path;
  - fused CUDA kernels for the spiking neurons, the fast-weight memory, kWTA, RMSNorm, cross-entropy
    and AdamW;
  - bf16 matrix multiplies;
  - a loop that never waits for the GPU;
  - a chunk form of her memory for long texts: 16 tokens at a time with small matrix products, 1.8×
    faster training at context 4,096.

  Every speed-up was checked to give the same numbers before it was used.
- Everything from v0.1: Ava-S and Ava-M (`release/ava-s`, `release/ava-m`), the dependency-free CPU
  crates under `design-tests/nn/`, and her first life (a spiking network with no gradient descent that
  learned word forms and arithmetic habits, solved ARC puzzles, and knew when to say "I don't know").
- [ARCHITECTURE.md](ARCHITECTURE.md), the technical deep-dive of the CPU engine, and
  [SCALING.md](SCALING.md), the plan for taking her further.

## A different way for a language model to think

| | Ava | A standard transformer |
|---|---|---|
| How she remembers the text so far | Her state changes as she reads: spiking membranes, plus fast synapses rewritten at every token | Keeps every earlier token and looks back over all of them (attention) |
| Memory per conversation | Fixed: about 408 KB at the release size, whatever the length | Grows with every token (the key/value cache) |
| Cost of the next token | The same at token 10 and at token 10,000 | Grows with the length of the text |
| Neurons | Leaky integrate-and-fire; hard 0/1 spikes; each channel learns its own leak | Dense real-valued activations |
| Codes | k-winners-take-all: one unit in eight active in the wide layer | Dense: every unit active |
| How memory is written | Error-correcting: each token writes only the difference between what her memory predicted and what she saw | Nothing is written; everything is kept |
| Time scales | Every memory head forgets at its own rate (at the release size, from about 22 to about 45,000 tokens) and learns when to forget | One window, cut at the context length |
| Word order | Carried by the dynamics; no position embedding | A position embedding or rotation |
| Language | Rust from the tokenizer to the kernels | Usually Python plus a framework |

**What is new and what is not.** The parts have a long history:

- fast weights: Hinton and Plaut 1987; Schmidhuber 1992; Ba et al. 2016;
- the delta rule: Widrow and Hoff 1960;
- delta-rule fast weights as a sequence layer: Schlag, Irie and Schmidhuber 2021; Yang, Kautz and
  Hatamizadeh's Gated DeltaNet, 2024;
- spiking networks trained by surrogate gradients: Zenke and Vogels 2021 and others.

A fast-weight read is mathematically a form of linear attention, and the layer is close to Gated
DeltaNet's.

What Ava adds is:

- **the combination**: spiking membranes, error-correcting fast synapses at many time scales and sparse
  kWTA codes, in one block, with no softmax attention anywhere;
- **the evidence**: each part was kept only after it beat a transformer of equal size, trained on equal
  tokens, in a pre-registered test.

## How she thinks

Ava is a stack of `L` identical blocks over a residual stream of width `d`. In words:

1. **Tokens to vectors.** A byte-level BPE tokenizer (4096 tokens: the 256 bytes, EOS, PAD and
   3,838 learned merges) turns the text into ids. Each id picks a row of the embedding table
   `E [4096, d]`. There is no position embedding.
2. **Spiking memory.** RMSNorm the stream and multiply by `Win [d, d]` to get an input current for
   each of `d` neurons. Run the neurons through time, one step per token: each integrates its current
   and leaks, and when its membrane crosses the threshold it spikes (emits a 1) and resets. The spike
   train is multiplied by `Wout [d, d]` and added back to the stream.
3. **Fast-weight memory.** RMSNorm the stream and project it to a query, key, value and gate for each
   of `H = d / 32` heads, plus two scalars per head that set how much to forget and how hard to write.
   Each head holds a 32 × 32 matrix of fast synapses `F`, which it updates at every token and reads with
   the query. The result is gated, projected by `Wo [d, d]` and added to the stream.
4. **Sparse code.** RMSNorm the stream and widen it to `4d` units with `W1`. Keep only the `4d / 8`
   largest units (k-winners-take-all), zero the rest and apply ReLU. Project back with `W2 [4d, d]` and
   add to the stream.
5. **Output.** A final RMSNorm, then logits over the 4096 tokens through the same table `E`
   (tied output: `logits = x · Eᵀ`).

The neurons are leaky integrate-and-fire units in discrete time, one step per token. For one channel
at step `t`, with `v_0 = 0` and `s_0 = 0`:

```
v_t = β · v_{t−1} + I_t − θ · s_{t−1}        membrane: leak, integrate, soft reset by subtraction
s_t = H(v_t − θ)                             spike: 1 if the membrane reached the threshold
β   = sigmoid(p),  p learnable per channel   leak, initialised so that β = 0.9
θ   = 1                                      threshold
```

The fast synapses of one head follow the gated delta rule. Here `x` is the normed stream at one token
and `h` is the head's number (0, 1, 2, …):

```
q, k, v, g = x·Wq, x·Wk, x·Wv, x·Wg            this head's 32 columns of each
q̂ = q / |q|,   k̂ = k / |k|                    unit-length query and key
α = sigmoid(x·wa + ba + logit γ_h)            how much to keep; γ_h = 1 − 2^−(5+h)
b = 2 · sigmoid(x·wb + bb)                    how hard to write
e = v − α · Fᵀ k̂                               what the memory got wrong about this key
F ← α · F + b · k̂ eᵀ                           forget a little, then write the correction
y = Fᵀ q̂                                      read with the query
out = y / rms(y) ⊙ silu(g)                    head norm and output gate (then all heads · Wo)
```

`γ_h` starts each head at its own memory length. Head 0 keeps half of an imprint for about 22 tokens,
and each later head keeps it twice as long, up to about 45,000 tokens for the twelfth head at the
release size. The learned gate `α` then moves that rate token by token. A token writes only `e`, the
part of its value that the memory could not already predict from its key, so a fact she already holds
is not written twice.

## How she learns

A spike is a step function, and a step has zero gradient almost everywhere, so plain
backpropagation cannot train it. Ava uses a **surrogate gradient**: the forward pass keeps the hard
step, and the backward pass pretends the step was a smooth curve. Her stand-in is the fast sigmoid
derivative `f'(x) = 1 / (1 + 4|x|)²` with `x = v − θ`. It is largest when the membrane is close to the
threshold and fades away from it. The gradient flows back through time inside the LIF op and into the
leak of every channel. The reset term is treated as a constant in the backward pass (the detached
reset of Zenke and Vogels 2021), which is the standard choice.

The fast synapses are differentiable, so the gradient also flows back through every write and read of
`F`. The slow weights learn how to use a memory that rewrites itself at every token.

The rest of her learning is deliberately plain:

- **AdamW** (β₁ 0.9, β₂ 0.95, ε 1e-8). Weight decay 0.1 is decoupled and applies to `Win`, `Wout`,
  `Wq`, `Wk`, `Wv`, `Wo`, `W1` and `W2` only. It never applies to the embedding, the output gate `Wg`,
  the forget and write gates, norm gains or leaks. The global gradient norm is clipped at 1.
- **Cosine learning rate** with a linear warm-up, decaying to one tenth of the peak.
- **Checked against autograd.** The hand-written training steps are tested against candle's autograd,
  run in f32 on the same weights and batch. For Ava's design:
  - the loss agrees within 1e-4;
  - every gradient agrees in direction (cosine above 0.9999) and in size (within 1%).

  Each fused kernel is checked against a chain of standard ops. The CPU engine's ops are checked against
  finite differences.
- **bf16 matrix multiplies, f32 everywhere else.** In bf16, rounding flips a few spikes. Each bf16
  path was therefore accepted only after a full training run matched the f32 path's bits per byte
  within 0.005.

## How she stores memories

**Three levels, as in a brain.**

**(a) The slow weights** are her cortex: knowledge spread across all of them and learned by gradient
descent over the whole of her reading. This is the level that scales with data and compute.

**(b) The fast synapses and membranes** are her working memory. They hold what she has read in this
text: names, the topic, the last few clauses. They are rewritten at every token, have many time scales
at once, and start empty for each new text. Nothing is stored token by token. The memory is a fixed set
of matrices that the text keeps rewriting, so it can hold the gist of a long passage, not a verbatim
copy.

**(c) Sleep** moves the day into the slow weights, as the hippocampus replays the day into the cortex
during sleep. Since v0.3 this is part of the trained brain (`grow_gpu --phase sleep`):

- she re-reads the day's episodes, the most surprising ones more often;
- she re-reads her ordinary text at the same time, as the brain replays old memories;
- **synaptic consolidation** holds the important synapses in place. Before sleep, each weight's
  importance to her reading is measured (its mean squared gradient over 16 batches of ordinary text,
  as in EWC). After every sleep step, each weight is pulled back towards its value before sleep, in
  proportion to that importance. One dial trades how much of the day she keeps against how much of
  her reading she protects.

After half a minute of sleep she answers 96% of questions about 150 new people (24% before, at chance).
With the facts in front of her and no sleep, she is still at chance: she cannot yet look up something
she has just read. v0.3 added an attention read (the engram, block 4) and recall practice for that,
and it went unused (see Results).

Her first life had a separate engram memory (`design-tests/examples/compose.rs`; design note
`design-tests/hippo/design.md`), in which every lesson is kept whole, recalled by overlap with the
cue, and replayed in sleep exactly once. It is not connected to the trained brain.

## How she scales up

Ava has four knobs:

- the width `d`, a multiple of 32, which sets the number of heads `H = d / 32`;
- the depth `L`;
- the number of winners in the sparse layer, which is always one eighth of `4d`;
- the training context `ctx`.

Her weights are:

| Part | Weights |
|---|---|
| Embedding (tied with the output) | `4096 · d` |
| Spiking memory, per block | `2d²` (`Win`, `Wout`) + `d` leaks + `d` norm gains |
| Fast-weight memory, per block | `5d²` (`Wq`, `Wk`, `Wv`, `Wg`, `Wo`) + `2(d·H + H)` (the two gates) + `d` norm gains |
| Sparse MLP, per block | `8d²` (`W1`, `W2`) + `d` norm gains |
| Final norm | `d` |
| Total | `4096·d + L·(15d² + 4d + 2(d·H + H)) + d` |

For example, d = 384 and L = 8 give 19,354,176 weights, 17,781,312 of them outside the embedding (the
release model). Nothing in her weights depends on the context length. Her carried state is `L · (2d +
H · 32 · 32)` numbers: 104,448 at the release size (408 KB in f32), for any length of text. A
transformer of the same width and depth keeps `2 · d · L` = 6,144 numbers per token, so it passes
Ava's whole state after 17 tokens and keeps growing.

**Any amount of text.** The data crate tokenises text once into binary shards (`AVATOK01`, 50M tokens
each by default) and streams them:

- one shard is in memory at a time;
- a seeded shuffle runs over the shards and within each shard;
- a one-line cursor resumes on exactly the next window.

The trainer saves a full checkpoint every 500 steps: weights, optimiser moments and data cursor.
`--resume` continues a stopped run exactly.

**On a GPU.** `design-tests/nn/gpu` trains every model on one CUDA card. The speeds below are at the
release shape (d 384, 8 layers, context 256, batch 32) on an RTX 4070 with the card to itself:

| Model | Tokens per second |
|---|---|
| Ava's design (spiking + fast weights + kWTA) | 132,000 |
| Plain spiking Ava (v0.1's design) | 268,000 |

A 20-tokens-per-weight run at the release size takes about 45 minutes. The fast-weight memory is one
projection matmul and one fused kernel each way. The forward saves the 32 × 32 state every 16 tokens,
and the backward recomputes each 16-token chunk, so memory stays small. The trainer runs on one GPU;
data parallelism across GPUs is the next step for anyone scaling her (see the roadmap).

**The scaling test** is the method that matters most. Train her design and the yardstick transformer at
several sizes on the same data, with tokens in proportion to weights and a learning-rate sweep, and
compare the slopes. A gain at one size is not a scaling result. The results so far are below. In
short, her design beats a transformer per weight at 1M and 3.3M weights, where both were trained, but
its curve is flatter. At the release size, the transformer's fitted line is ahead.

## How she models the human brain

These are analogies, and some are loose. The table says which.

| Ava | Brain | How close |
|---|---|---|
| Leaky integrate-and-fire neurons with a threshold, a spike and a soft reset | Cortical neurons: a membrane that integrates input, leaks, fires an action potential and resets | Close in form. The real neuron has many more time scales, dendrites and a refractory period; hers has one state variable per channel and one step per token |
| A learnable leak `β` per channel | Membrane time constants, which differ by cell type | Loose. Hers is learned by gradient descent, not set by biophysics |
| Spikes of exactly 0 or 1, sparse in time (0.3–4% of steps per layer in the release model) | All-or-none action potentials, sparse firing | Close. The surrogate gradient is a training device with no biological counterpart |
| Fast synapses rewritten at every token | Short-term synaptic plasticity, proposed as a basis of working memory (Mongillo, Barak and Tsodyks 2008) | Fair in function: a memory held in synapses, not in activity. The mechanism is a matrix update, not calcium dynamics |
| Writing only the prediction error (`e = v − α·Fᵀk̂`) | Error-driven learning (the Rescorla–Wagner rule; predictive coding) | Fair in function |
| Memory heads with time scales from tens to tens of thousands of tokens | A hierarchy of intrinsic time scales across cortex (Murray et al. 2014) | Loose. Hers are set per head and moved by a learned gate |
| Learned forget and write gates | Neuromodulators that set when to store and when to let go | Loose |
| k-winners-take-all in the wide layer | Lateral inhibition and the sparse codes of cortex | Fair. Real inhibition is done by neurons and is graded; hers is an exact top-k |
| No position embedding: order lives in the dynamics | Brains have no position index; order is carried by activity and synapses | Fair |
| Weights trained by backpropagation | Synaptic plasticity | Loose. The brain does not backpropagate, even if the outcome (distributed knowledge) is alike |
| Sleep: the day's episodes re-read with ordinary text, the most surprising more often (v0.3) | Hippocampal replay into the cortex during sleep (complementary learning systems, McClelland et al. 1995) | Fair in function. Her sleep is extra training steps, not a separate state of the network |
| The anchor: important weights pulled back to their value before sleep (v0.3) | Synaptic consolidation: synapses that matter become harder to change (Benna and Fusi 2016) | Fair in function. Importance is measured by gradients (as in EWC, Kirkpatrick et al. 2017) |
| The glance: one mark for a number's length before its digits, and a learned code for each digit's place (v0.3) | Seeing a number's length and shape before reading it; a place-value sense | Loose. It is an input code; what each place means is learned |
| Engram memory and sleep replay (first life) | Hippocampal episodes and consolidation during sleep | Fair in function, not in mechanism |
| Abstaining under right-minus-wrong scoring (first life) | Metacognition, the feeling of knowing | Fair in function |
| Softmax attention | Brains attend, but do not keep and search a copy of every past input | Tested in v36 (model A) and dropped: it did worse than the memory alone at both sizes tested. v0.3 has one attention read for looking things up; it went unused |

## Results

All bits-per-byte (BPB) scores are on held-out validation text from her own corpus; lower is better.
Bits per byte does not depend on the tokenizer. Every comparison below was pre-registered with its
pass rule, run once, and judged by the letter.

### Ava 0.3: numbers, long books, sleep (v39–v41)

One model, made in three steps on one RTX 4070:

1. **v39:** Ava's recipe from scratch (387M tokens, 45 minutes) with the glance and 2% child-style number
   practice in her text.
2. **v41, grow:** 108M more tokens (34 minutes, sharing the GPU), made up of:
   - her usual text 70%, 337 whole public-domain books (PG-19) 25%, number practice 5%;
   - question-answer batches;
   - recall practice;
   - the last quarter at 8,192 tokens per window.
3. **v41, sleep** on 150 made-up people with the anchor (25 seconds), then **reward** learning on
   training questions (51 seconds).

Each sealed test was pre-registered and opened once. Brackets are 95% bootstrap intervals.

| Test | Result | Verdict |
|---|---|---|
| Numbers (842 fresh items) | 60.5% vs v39's 42.4%: **+18.1 [+12.9, +23.0]**. Comparing numbers of different lengths: 90.5% vs 46.5% | pass |
| Long text (100 unseen books, 16,384 tokens each) | the full memory beats a memory cut every 256 tokens by **0.033 bits per byte [0.031, 0.035]** from token 4,096 on | pass |
| Find a planted fact in a book (attention on − off) | +1.3 [−0.8, +3.3]; 27% (chance 25%) | fail |
| Sleep (450 questions about 150 new people) | **96.2% after vs 23.8% before: +72.4 [+68.2, +76.7]** | pass |
| Questions (1,200 held-out SciQ, ARC, OBQA; 4 choices) | **40.7% vs v39's 33.0%: +7.7 [+4.9, +10.3]**; SciQ 56.5%, ARC 34.8%, OBQA 30.8% | pass |
| Right − wrong, after reward vs before | −0.013 [−0.036, +0.011] | fail |
| Reading (val bits per byte, rule ≤ 1.497) | 1.511 | fail (narrowly) |

BLiMP: 74.1%.

**What it means.**

- **The anchor works.** On dev, sleep without it cost her reading +0.046 BPB; with it, +0.016, for 5
  points less recall of the day.
- **Scoring answers per byte** instead of in total put OpenBookQA back above chance.
- **Numbers improved through practice, not the sense-of-size code.** Switching that code off changes
  0.3 points; switching the place code off costs 8.
- **Attention went unused twice** (v40 and v41). Her fast-weight memory already covers short gaps, and
  the read was added late with its output at zero. The next tries are to train lookups from the start,
  or to make the fast-weight memory itself last longer.

The 1,200 questions are held-out slices of the training splits, because the official test splits were
spent by earlier versions. Full card: [release/ava-0.3/MODEL_CARD.md](release/ava-0.3/MODEL_CARD.md).

### Ava (v0.2), the reading model

| | |
|---|---|
| Model | spiking memory + fast-weight memory (gated delta rule) + sparse MLP; no attention, no position embedding |
| Size | d 384, 8 layers, 12 memory heads, trained at context 256 |
| Weights | 19,354,176 total, 17,781,312 non-embedding |
| Training | 387,083,520 tokens (20 per weight), 47,251 steps of 8,192 tokens, on one RTX 4070 |
| Training time | 44 min 22 s at 145,397 tokens per second (checkpoint saves included) |
| Validation bits per byte | **1.487** (1.487072; 1,000,000 target tokens, 3,051,730 bytes) |
| Validation loss | 3.146 nats per token |
| BLiMP grammar pairs | 76.1% of 67,000 pairs (GPT-2: about 81%; Ava-M2: 67.5%; chance: 50%) (a practice look; this set was spent in earlier versions) |
| Sample | "Once upon a time, it must be taken off-departure, and we have to leave with them all." (seed 1, the first of four prompts, unselected) |

**What it means.** She is by far the best Ava.

- **Against plain spiking Ava.** She scores 0.155 BPB better than Ava-M2, the plain spiking model of
  the same width and depth, which had fewer weights and read less (13.4M weights, 268M tokens). She
  also scores 76.1% on BLiMP.
- **Against the transformer, she has lost her lead.** The pre-registered yardstick, the transformer's
  fitted line, puts a transformer of her size at 1.449, which is 0.038 better than her. Her gap to that
  line has grown with size:

  | Non-embedding weights | Her gap to the line |
  |---|---|
  | 1M | −0.049 |
  | 3.3M | −0.018 |
  | 17.8M | +0.038 |

  A straight-line fit of the three gaps crosses zero at about 5M weights.

Two caveats count against a firm verdict:

- no transformer was trained at her size, so the line is extrapolated from transformers of up to 6.3M
  weights;
- her learning rate was chosen at d 192, not swept at her size.

The direction is clear all the same. Her design learns more per weight at small sizes; a transformer
gains more from size.

The full card, with the exact command, seeds, checkpoint hash and unselected samples, is
[release/ava/MODEL_CARD.md](release/ava/MODEL_CARD.md).

### Ava-QA: taught to answer questions (v37, v38)

Ava was taught on 40,089 question-answer pairs (SciQ, Dolly 15k and ARC, training splits only),
scored on the answer alone. Each multiple-choice option is scored by how likely she finds it as an
answer, and she may say "I don't know" when her best option does not beat the second by a margin
fixed on the dev set.

| | Ava | taught (v37) | **taught + replay = Ava-QA (v38)** |
|---|---|---|---|
| SciQ test, closed book (chance 25%) | 41.1% | 56.8% | **56.4%** |
| SciQ test, passage shown | 45.9% | 66.4% | **66.2%** |
| with "I don't know": right − wrong; share right | +21; 62.1% | +169; 62.1% | **+187; 64.3%** |
| her reading (BPB, lower is better) | 1.487 | 1.748 | **1.550** |
| BLiMP grammar (never taught) | 76.1% | 70.4% | **73.1%** |
| teaching time | | 75 s | 95 s |

**Replay.** Teaching alone made her forget part of her reading. In v38 every teaching step also
re-read 16 windows of her ordinary text, as the brain replays old memories while it learns. She kept
the whole answering skill and lost about a quarter as much reading (+0.063 BPB instead of +0.261).

**Honesty notes.** The SciQ test was sealed on download and opened once, by v37's pre-registered test.
The Ava-QA column is a disclosed second look at that spent set, made after Ava-QA was chosen on dev.
The learning rate and the replay weight were each picked by a pre-registered rule on dev. ARC
Challenge is still beyond her: 19.1% on dev, under the 25% chance level.

The full card is [release/ava-qa/MODEL_CARD.md](release/ava-qa/MODEL_CARD.md).

### v36: does attention help? (memory against memory plus attention)

Model A is Ava's design with softmax attention in place of the fast-weight memory in every fourth block,
counted back from the last. Both models were trained on 20 tokens per weight with a learning-rate
sweep. The transformer column is the pre-registered fitted line from v34:

| BPB | d 128, 4 layers (~1M weights) | d 192, 6 layers (~3.3M) |
|---|---|---|
| **Ava's design (memory only)** | **1.875** | **1.707** |
| A (memory + attention) | 1.883 | 1.719 |
| Transformer (fitted line) | ~1.925 | ~1.725 |

**Verdict.** Both beat the transformer at both sizes, and the memory-only design beat the attention
hybrid at both sizes, so Ava has no attention. Both lose ground with size: the memory design's lead
over the transformer fell from 0.049 to 0.018 BPB.

### v35: a working memory for the spiking brain

All models were trained at d 128, 4 layers, on equal tokens. The transformer's best run scored 1.959.

| Model | What it adds | BPB |
|---|---|---|
| S | plain spiking Ava | 2.013 |
| F | Hebbian fast weights | 2.090 |
| G | + learned forget and write gates | 2.027 |
| R | + unit-length keys and an output gate | 1.883 |
| **D** | + error-correcting write (the delta rule): **Ava's design** | **1.875** |

R and D beat the transformer; F and G did not even beat plain S. R was also run at d 192: it still
won (1.717 against the fitted 1.724), but its lead shrank from 0.042 to 0.008.

### v34: the fair scaling test of the plain spiking design

S (v0.1's design) and the transformer T were each trained at five sizes. Both were fed 20 tokens per
weight, and each was swept over three learning rates, with the best one counted:

| Size | S | T |
|---|---|---|
| d 64, 2 layers | **2.278** | 2.305 |
| d 96, 3 layers | 2.124 | **2.109** |
| d 128, 4 layers | 2.013 | **1.959** |
| d 192, 6 layers | 1.851 | **1.753** |
| d 256, 8 layers | 1.740 | **1.628** |

**Verdict: S loses ground.** The fitted slopes are −0.130 per e-fold of weights for S and −0.165 for T,
a gap growth of +0.145 BPB over the size range against a noise bar of 0.0036. That result is why she
got a working memory. The transformer's fitted line is `BPB = 4.199618 − 0.164786 · ln N`, with `N`
the non-embedding weights. It was fixed before any memory model was trained and is the yardstick for
v35, v36 and Ava.

### Earlier models

| Model | Design | Weights | Tokens | BPB |
|---|---|---|---|---|
| Ava-S (v0.1) | plain spiking, CPU, f32 | 3.0M | 48M | 1.872 |
| Ava-M (v0.1) | plain spiking, GPU, f32 | 13.4M | 24M | 1.888 |
| Ava-M2 (v35, not released) | plain spiking, GPU | 13.4M | 268M | 1.642 |

The v32 test (spiking against a transformer at three sizes on a fixed 4M-token budget) is described
with its full result lines in [ARCHITECTURE.md](ARCHITECTURE.md). Its caveat, that a fixed budget is not
a fair scaling test, is what v34 fixed.

### Her first life, measured

Count-and-search Ava, before she had a gradient. Each line is one pre-registered run. The ARC-AGI-1
evaluation set and the BLiMP test have each been used several times over the versions, so treat both
as spent.

| Test | Ava | For reference |
|---|---|---|
| ARC-AGI-1 puzzles, evaluation set | 60 of 400 right (15%), 3 wrong, the rest "I don't know" | o3: 76-88% |
| ConceptARC puzzles (160) | 9 right, 21 wrong | Her self-check is overconfident when examples are few |
| BLiMP grammar (test pairs) | 56.2% | GPT-2: about 81%; chance: 50% |
| Zorro child-language grammar | 64.9% | chance: 50% |
| AI2 science questions (Test) | Easy 65.7%, Challenge 34.0% | Challenge was built so that word matching fails |
| English past tense, words never taught | 95.7% (irregular verbs 79%) | suffix analogy: 95.1% |
| Long addition and subtraction, learned as habits | 100% up to 12 digits | |

She may answer "I don't know". She is judged by right minus wrong, and by the share right among the
answers she gives. A confident wrong answer counts against her.

## Quick start

You need a recent stable Rust toolchain. For the GPU crate you also need the CUDA toolkit, a C++
compiler for its kernels and a network connection for its crates (candle 0.11). `build.sh` sets the
environment that worked on the author's Windows machine (MSVC, CUDA 13.2, compute capability 8.9);
on Linux it should need only `cargo build --release` with `CUDA_COMPUTE_CAP` set for your card
(untested).
Without a GPU, `--cpu` runs the same code on the CPU, slowly.

**Build**

```
cd design-tests/nn/gpu
./build.sh build --release        # train_gpu, generate_gpu, blimp_gpu, sft_gpu, qa_eval, ...
cd ../../..
```

**Get Ava 0.3** from the [v0.3.0 release page](https://github.com/nochallenge/ava/releases/tag/v0.3.0)
and run her (from the repository root):

```
curl -L -o release/ava-0.3/model.safetensors https://github.com/nochallenge/ava/releases/download/v0.3.0/ava-0.3.safetensors
(cd release/ava-0.3 && sha256sum -c SHA256SUMS)
design-tests/nn/gpu/target/release/generate_gpu --model release/ava-0.3/model.safetensors \
  --bpe release/ava-0.3/bpe.txt --arch D --d 384 --layers 8 --glance \
  --prompt "I have 3 apples and 4 more, so now I have" --tokens 40 --temperature 0.8 --seed 1
```

`--glance` adds the number-length marks she reads. To teach her new facts in her sleep, see
`grow_gpu --phase sleep` in her [model card](release/ava-0.3/MODEL_CARD.md).

**Get the v0.2 models** from the [v0.2.0 release page](https://github.com/nochallenge/ava/releases/tag/v0.2.0):

```
curl -L -o release/ava/model.safetensors https://github.com/nochallenge/ava/releases/download/v0.2.0/ava.safetensors
curl -L -o release/ava-qa/model.safetensors https://github.com/nochallenge/ava/releases/download/v0.2.0/ava-qa.safetensors
(cd release/ava && sha256sum -c SHA256SUMS) && (cd release/ava-qa && sha256sum -c SHA256SUMS)
```

**Run Ava** (from the repository root):

```
design-tests/nn/gpu/target/release/generate_gpu --model release/ava/model.safetensors \
  --bpe release/ava/bpe.txt --arch D --d 384 --layers 8 --prompt "Once upon a time" \
  --tokens 100 --temperature 0.8 --top-k 40 --seed 1
```

Generation carries her state from token to token, so each new token costs one step, whatever the
length. `--window` re-runs the last `ctx` tokens instead, and `--check` compares the two. EOS
renders as a newline. She completes text: there is no chat format, so expect her to ramble.

**Ask Ava-QA a question:**

```
design-tests/nn/gpu/target/release/generate_gpu --model release/ava-qa/model.safetensors \
  --bpe release/ava-qa/bpe.txt --arch D --d 384 --layers 8 --qa --temperature 0 \
  --prompt "What gas do plants take in from the air to make food?"
```

She answers "carbon dioxide". `--chat` asks question after question, each from a fresh memory. To
teach her yourself, `qa_prep` builds the examples from SciQ, Dolly 15k and ARC (downloaded into
`data/qa/`), `sft_gpu` teaches (the exact command is in the Ava-QA card), and `qa_eval` scores the
multiple-choice sets.

**Train your own**, with the exact command of the release run:

```
design-tests/nn/gpu/target/release/train_gpu --model D --d 384 --layers 8 --ctx 256 --batch 32 \
  --tokens 387083520 --lr 0.001 --warmup 200 --min-lr-frac 0.1 --wd 0.1 --clip 1.0 \
  --seed 1 --data-seed 1 --eval-tokens 1000000 --log-every 500 --out runs/my-ava
```

The run writes these files:

- `log.tsv`, one line per step;
- `result.tsv`, one line: configuration, weights, validation BPB, spike rates and timings;
- `model.safetensors`;
- `ckpt/`, for `--resume`.

Other flags and switches:

- `--model` picks the design: `D` is Ava, `S` plain spiking, `A` memory plus attention, `T` the
  yardstick transformer.
- `--max-steps N` stops early without changing the schedule.
- `--bench-steps N` times N steps and writes nothing.
- `AVA_BF16=0` runs the matrix multiplies in f32.
- `AVA_MANUAL_DA=0` uses autograd instead of the hand-written step.
- `AVA_SYNC_LOOP=1` restores the waiting loop.

By default it reads `data/nn/shards-4096/{train,val}` and `data/nn/bpe-4096.txt` under the repository
root.

**Scale her up.** Change `--d` (a multiple of 32) and `--layers`, and set `--tokens` to about 20 times
the weight count that the trainer prints. Sweep the learning rate: 0.001 and 0.002 were best at the
sizes tested, and larger models usually want lower. To judge a new size fairly, train T at the same
size on the same tokens, and compare the slope, not one point.

**Score grammar** on BLiMP-style minimal pairs, batched on the GPU:

```
design-tests/nn/gpu/target/release/blimp_gpu --model release/ava/model.safetensors --arch D --d 384 --layers 8
```

**The CPU engine** (no dependencies at all; plain spiking Ava-S and the v32 models):

```
cd design-tests/nn/brain
cargo build --release        # train, generate, talk, score, profile
target/release/generate --model ../../../release/ava-s/model.awts --bpe ../../../release/ava-s/bpe.txt \
  --prompt "Once upon a time" --tokens 200 --temperature 0.8 --top-k 40
```

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

Each input is one of:

- a file (`path`);
- a tab-separated column (`path@tab:N`);
- a field split by any string (`path@sep:STR:N`).

Sources are interleaved, so every shard holds the global mixture. Digits are always single tokens, so
arithmetic stays learnable.

**Her first life** builds separately: `cd design-tests && cargo build --release --examples`. That
older crate has one dependency (`rustyline`, for its line editor). The programs read `../data/...`
and are documented in their file headers.

**Data is not included.** The datasets have their own licences, and some have none. Fetch them
yourself and place them under `data/`, which git ignores:

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
| `qa/` | SciQ, Dolly 15k and ARC, for teaching answers (`qa_prep`) | Allen Institute for AI; Databricks |
| `pg19/` | PG-19 public-domain books (v0.3 read 337 from its training split) | DeepMind (Apache 2.0) |
| `nn/` | the BPE file and shards made by the commands above | built locally |

## Repository layout

```
design-tests/nn/gpu/      ava_gpu    Ava on candle/CUDA: the models (D = Ava, S, A, T and the v35
                                     variants), hand-written training steps, CUDA kernels (lif.cu),
                                     train_gpu, generate_gpu, blimp_gpu; qa_prep, sft_gpu, qa_eval
                                     for teaching and scoring answers; grow_gpu (grow, sleep with
                                     the anchor, reward); mc_eval, num_eval, long_eval (v0.3 tests)
design-tests/nn/data/     ava_data   byte-level BPE, AVATOK01 shards, streaming sampler, cursor;
                                     number marks and practice, books, recall and sleep-fact items
design-tests/nn/grad/     ava_grad   CPU tensors, autodiff tape, LIF scan, kWTA, AdamW, AVAWTS01 checkpoints
design-tests/nn/brain/    ava_brain  the CPU models (T / S / H), trainer, bits per byte;
                                     train, generate, talk, score, profile
design-tests/src/         her first life: the event-driven spiking neuron, synapses, sparse codes,
                          allocators, readout, brain files (AVABRAIN)
design-tests/examples/    one program per first-life faculty: compose.rs (talk, engrams, sleep), arc.rs,
                          procedure.rs (arithmetic as habits), know_what.rs (the self-model that
                          abstains), exam.rs, blimp_open.rs, read_think.rs, paired_boot.rs
design-tests/hippo/       the hippocampus design note
release/ava-0.3/          Ava 0.3: tokenizer, settings, model card (weights: release page)
release/ava/              v0.2's reading model: the same
release/ava-qa/           Ava taught to answer questions: the same, for Ava-QA
release/ava-s/, ava-m/    the v0.1 models
ARCHITECTURE.md           the technical deep-dive (CPU engine and v0.1 GPU crate)
SCALING.md                the plan: what scales, what goes, the staged path, how to stay honest
CONTRIBUTING.md           issues welcome, forks welcome, no outside code for now
LICENSE.md                PolyForm Noncommercial 1.0.0
```

The `design-tests` name is historical: it was the scratch crate where the first ideas were measured,
and it grew into Ava herself. Code marked `TESTING / DESIGN ONLY` is experimental, which is all of it.
The top-level `src/` is a hello-world placeholder from the project's first day.

## Honesty and method

- **Pre-registration.** Every test is written down before it runs: the build's hash, the flags, the
  data and the pass rule. It runs once, and the verdict is applied by the letter, including when it
  goes against her (v34: "S loses ground"; v36: attention dropped).
- **Same tokens per weight, same tuning.** Her design and the transformer get the same data, the same
  tokens per weight and the same learning-rate sweep.
- **Speed never changes the numbers.** A faster path is used only if a full run matches the old one
  within 0.005 BPB and its loss curve stays within 0.05, or if it gives byte-identical weights.
- **Seed noise is measured, not assumed** (three data seeds; σ about 0.002 BPB).
- **Right minus wrong.** Where she may abstain, a confident wrong answer counts against her.
- **Fresh test sets.** New ideas are developed on practice and dev data only. Each test set is used
  once, and every earlier look is disclosed. The sets still unopened are listed in SCALING.md.
- **No hand-written knowledge.** No word lists, canned phrases or per-puzzle code.
- **Everything disclosed.** Stopped runs, amendments, binary changes during a test and learning rates
  at the edge of a grid are written in the notes for each test and summarised in the model cards.

## Limits

- She is tiny: about 20 million weights. Modern language models have thousands of times more.
- She has read little: about 495M tokens (Ava 0.3) of a small, dialogue-heavy English corpus and 337
  old books. She knows no other language and very little of the world.
- Her curve is flatter than a transformer's. At her size, the transformer's fitted line is 0.038
  BPB ahead of her; the line crosses hers at about 5M weights. Steepening the curve is the main open
  problem (see the roadmap).
- Ava completes text. Ava-QA answers single questions, but her answers are short and often wrong, and
  ARC Challenge questions are still beyond her. There is no conversation format and no safety tuning.
- She will produce nonsense often, repeat herself, and make things up. Do not rely on anything she
  says.
- Her memory helps across 16,384 tokens of a book (v0.3), but she cannot yet look up a fact she has
  just read: with it in front of her she answers at chance, and the attention read added for this is
  unused. She learns new facts by sleeping on them.
- Reward learning (learning from right and wrong alone) added nothing measurable in v0.3.
- She trains slower than a transformer of her size in this code. On an RTX 4070 at d 384 and 8 layers, with
  8,192 tokens per step:
  - at context 256: 148,000 tokens per second, against the transformer's 268,000;
  - at context 1,024: 137,000, against 237,000;
  - at context 4,096: 113,000, against 161,000 (65,000 before the chunk kernel).

  When a batch holds few sequences, her fast-weight memory is computed in 16-token chunks (the chunk kernel:
  1.8× faster training at context 4,096). With many sequences it is computed one token at a time, which is
  then just as fast. Both forms run on ordinary f32 GPU cores, while attention runs on tensor cores. At
  context 256 the memory takes about 15 of her 55 ms per step, so most of the remaining gap is in the rest
  of the block. Her advantage is at generation time: a fixed 408 KB memory instead of a key/value cache that
  grows by about 24 KB with every token.
- The GPU trainer is not bit-reproducible (CUDA reductions vary in the last bits) and has been
  tested on one card (RTX 4070, compute capability 8.9) and on Windows only.
- Sleep is now part of the trained brain (v0.3). The other first-life faculties (the engram memory,
  arithmetic habits) are not yet connected to it; abstention is, through Ava-QA's "I don't know".

## Roadmap

1. **A steeper curve.** Find the part that makes her gain faster with size than a transformer, and
   prove it at three or more sizes. Candidates:
   - a less sparse MLP (kWTA keeps one in eight today);
   - wider memory heads;
   - measuring what the spiking memory adds beside the fast weights. In the first block of the memory
     models it falls almost silent.
2. **A faster block.** The chunk form of the delta-rule memory (as in DeltaNet and Gated DeltaNet) is
   in. What remains to match a transformer's speed:
   - its small matrix products on tensor cores;
   - fewer passes over memory;
   - a leaner rest of the block at short contexts.
3. **Many GPUs.** Data parallelism, so anyone with a cluster can run the scaling test far beyond this
   PC.
4. **Answering questions.** Ava-QA (v0.2) and Ava 0.3 answer multiple-choice science questions with
   a calibrated "I don't know". Next: harder questions (ARC Challenge) and longer answers.
5. **Hippocampus.** Replay (v0.2) and sleep with synaptic consolidation (v0.3) are in. Next: looking
   things up in what she has just read. Attention added late went unused twice. Next tries: train
   lookups from the start, or make the fast-weight memory itself last longer.
6. **A diary.** Keep every conversation, sleep on each day, learn from corrections, and measure her
   growth over time. A diary stays private: public weights are trained without one.
7. **Numbers.** The glance and practice are in (v0.3). A learned sense of size is in her weights but
   unused.
8. **Reward learning.** Learning from right and wrong alone. Tried in v0.3; no measurable gain yet.

## Licence

Ava is **source-available** under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): the
source is open to read, use, change and share for any non-commercial purpose. Commercial use needs a
separate licence from the author.

- **Free for any non-commercial use**: research, study, teaching, hobby projects, and non-profit,
  educational and government organisations. You may change her, build on her and share your versions
  under the same terms, including scaling her up.
- **Commercial use needs a separate licence from the author.** To ask for one, open an issue on this
  repository.
- Keep the `Required Notice:` line from [LICENSE.md](LICENSE.md) with any copy you share.

Outside code contributions are not accepted for now; see [CONTRIBUTING.md](CONTRIBUTING.md). Forks
for non-commercial work are welcome.

## Citation

```bibtex
@software{ava2026,
  author = {nochallenge},
  title  = {Ava: a brain-like language model in pure Rust},
  year   = {2026},
  url    = {https://github.com/nochallenge/ava}
}
```
