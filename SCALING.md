# Scaling Ava toward a large language model

This guide is for whoever works on Ava, her author included, on the way to growing her into a
large language model. She is free for non-commercial use under the PolyForm Noncommercial
License; see [LICENSE.md](LICENSE.md).

**The goal is to scale a brain-like model, not to swap her for a transformer.** Ava is meant to
work like a human brain: spiking neurons, sparse codes spread over many units, memories that
are replayed in sleep, learning from reward, and knowing when she doesn't know. Spiking language
models do exist; see section 6. But nobody has scaled this combination to LLM size, and whether
it scales is the open question this project bets on.

Her author's wishes:
- **Build only what survives scaling.** If a part would be thrown out at frontier scale, don't
  build it. Section 2 sorts what stays from what goes.
- **Keep her brain-like.** Make her brain mechanisms scalable, rather than replacing them.
- **Add weights and gradient descent**, so that knowledge spreads across her neurons and she can
  *feel* that "cat" is like "dog". Black-box knowledge is fine.
- **Give her the capability to read any amount of text**, up to trillions of words. On a home PC
  she will read far less, but nothing in the code should stop her.
- **Keep her pure Rust.** No Python anywhere, not even for tooling. Rust crates are fine.
- **Keep measuring honestly** (section 7).

## 1. Where she is today
Until October 2026, Ava learned by **counting and by search**, never by gradients. That first
life is still in the repository (`design-tests/src`, `design-tests/examples`), and the table
below describes it. Since v32 she also has a **gradient-trained brain** in `design-tests/nn`:
three crates for the tokenizer and shards, the autodiff engine with the spiking and k-winners
ops, and the T/S/H models (see [ARCHITECTURE.md](ARCHITECTURE.md)). The rest of this guide is
the plan that brain follows; the *Status* notes below mark what is done.

**As of v0.2 (late October 2026)** the trained brain runs on a GPU (`design-tests/nn/gpu`), and her
design has a working memory. Each block has three parts:

- spiking neurons;
- fast synapses that follow the gated delta rule;
- the k-winners-take-all sparse layer.

Three pre-registered tests at 20 tokens per weight shaped her:

- **v34:** plain spiking Ava loses ground to a transformer as both grow.
- **v35:** the fast-weight memory beats the transformer at every size tried, but its lead shrinks with
  size.
- **v36:** adding softmax attention made her worse, so she has none.

The open problem is the slope: her curve is still flatter than a transformer's. The README has the
numbers; the release model is in `release/ava/`.

| Faculty | What it stores | How it answers |
|---|---|---|
| Grammar (`blimp_open.rs`) | Counts of which word follows which 1-4 words, in hash maps | Witten-Bell backoff gives each sentence a probability |
| Talk (`compose.rs`) | Sparse codes of contexts (allocated units); each unit's counted votes for the next word | The active units *vote* for the next word |
| Memory (`compose.rs`, engrams) | Each taught lesson kept whole; replayed in "sleep" | A question reactivates the lessons it overlaps |
| Science (`read_think.rs`) | Streams 14.6M sentences twice (word counts, then recall); keeps only each choice's 10 strongest matching sentences | Scores each choice by word overlap, and abstains below a margin |
| Self-model (`know_what.rs`) | When her own past answers were right | Speaks only when she is more likely right than wrong |
| Arithmetic (`procedure.rs`) | Habits learned from worked examples or from right/wrong alone | Runs the habits column by column |
| ARC puzzles (`arc.rs`) | Nothing learned across puzzles | Searches grid operations that fit every example, then checks itself |

Her neurons (`src/lib.rs`) are leaky integrate-and-fire neurons. They have whole-number weights
in millivolts and run one event at a time on the CPU. Her codes (the allocators in `src/v6.rs`)
are sparse and distributed. They are allocated by hashing and a learned competition, not by
meaning. So by default, two words that are used alike do not get alike codes. The optional
counted word classes (`src/wordclass.rs`) and meaning codes (`AVA_MEANING`) only partly change
this.

The trained brain's neurons are different: `design-tests/nn/grad/src/brain.rs` simulates the
same leaky integrate-and-fire neuron in time steps as batched tensors, with float weights and a
learnable leak per channel, trained by surrogate gradients (stage 3 below). Its codes are
learned: a k-winners-take-all layer keeps one unit in eight.

## 2. The rule: build only what scales
Before building anything, ask four questions:
1. Is it **learned**, rather than written by hand?
2. Does it get **better with more data or compute**, with nobody editing it?
3. Can it run as **batched maths on a GPU**?
4. Does it keep her **brain-like**?

If any answer is no, don't build it.

**Her brain design stays, made scalable:**

| Brain idea | Ava today | Scalable version |
|---|---|---|
| Spiking neurons | Event-driven, integer weights, CPU | The same leaky integrate-and-fire neurons, simulated in time steps as batched tensors on a GPU, with float weights trained by **surrogate gradients**. Spikes are non-differentiable, so the backward pass uses a smooth stand-in |
| Sparse distributed codes | Hashed and allocated | **Learned** sparse codes: layers where only the k most active units fire (k-winners-take-all), so meaning spreads across units and similar words get similar codes |
| Synapses that learn | Counts (Hebbian) | Weights trained by gradient descent. Hebbian or other local rules can return as an option, if they pass the scaling test |
| Cortical areas | One program per faculty | **Sparse experts**: many sub-networks, with only a few active for each input (mixture of experts) |
| Hippocampus and sleep | Engrams and sleep replay (`compose.rs`) | A fast memory with **learned** keys, so recall is by meaning rather than shared words. Sleep replays memories, and the replay trains the slow weights |
| Dopamine | Thompson sampling from right/wrong (`procedure.rs`) | Learning from a verifiable reward, which is how frontier models now learn to reason |
| Feeling of knowing | `know_what.rs` | A learned confidence output, calibrated on dev data, that decides between speaking and "I don't know" |
| Thinking before answering | ARC search with a self-check | Search guided by the network, which ranks what to try, and the self-check kept |

**These are hand-made, and scaling would throw them out. Stop building them:**
- n-gram and count tables used as her language model (`blimp_open.rs`, `blimp_class.rs`, the
  count votes in `compose.rs`);
- hand-written operation libraries (`arc.rs` `ops()`) and hand-given scaffolds (the column
  layout in `procedure.rs`);
- word-overlap cues for recall and memory;
- word-class tables and word-ending tables.

Keep the existing code anyway, for two reasons. It is the yardstick a new model must beat. And
with every switch off, her outputs must stay byte-identical.

## 3. Ceilings to remove first
These limits in today's code would stop her reading trillions of words:

| # | Ceiling | Where | Why it breaks | Fix |
|---|---|---|---|---|
| 1 | Her whole reading is loaded into RAM | `blimp_open.rs` `reading()` (line 304) loads three files whole, then shuffles them; `compose.rs` loads its corpus (line 1373) and sleep files (1680) whole, in file order | RAM sets the most she can read | A streaming, sharded reader (stage 2). `read_think.rs` already streams its 14.6M-sentence corpus (`stream()`, line 355), which is a model to follow |
| 2 | Contexts packed as word ids in 16-bit slots | `blimp_open.rs` `ctx_keys` (355); it dies at 65,535 words (about line 420) | The slots limit the vocabulary of the sentences being scored to 65,535 words. A model that predicts over all of her reading would hit the limit at once | A subword vocabulary (stage 1) |
| 3 | Count tables grow with the text | in-RAM hash maps in `blimp_open.rs` and `compose.rs` | Distinct n-grams grow almost linearly with the text | Goes away: a network's size is fixed however much it reads |
| 4 | Word classes have 16-bit ids | `compose.rs` lines 577, 1507, 1853, 2015; `src/reply.rs` lines 449, 469 | At most 65,535 classes | Goes away with the class tables (section 2) |
| 5 | The brain file is "AVABRAIN" version 1 | `src/store.rs` | It has no place for weight tensors | A tensor checkpoint: a small header, then raw little-endian f32 or bf16 arrays |
| 6 | Few threads | `read_think.rs` `threads()` (571) clamps to 1-4; `arc.rs` only defaults `AVA_THREADS` to 4 | Deliberate, to spare the author's PC | Raise the cap; train across GPUs with data parallelism |
| 7 | Whole-word tokens | `src/seq.rs` `tokenize` (40) | Unknown words have no representation | Byte-level BPE (stage 1) |
| 8 | CPU only, one event at a time | `src/lib.rs` | A trillion tokens needs GPUs | Time-stepped tensors (stage 3) |

*Status (2026-10-23).* Ceilings 1, 2, 5, 7 and 8 are addressed in `design-tests/nn`: a
streaming, sharded reader with a resumable cursor (`nn/data`); a byte-level BPE with a 4096
vocabulary; the `AVAWTS01` tensor checkpoint; time-stepped LIF tensors with f32 weights
(`nn/grad`). Ceilings 3 and 4 go away with the count tables, which stay only as yardsticks.
Ceiling 6: the new code takes up to 32 threads (`--threads`, `AVA_THREADS`, clamped to 1..=32),
and every op the models use is threaded with bit-identical results for any thread count; the
candle crate (`nn/gpu`) runs the same model on a GPU. The old programs keep their limits, as intended.

## 4. Where the network plugs in
Each faculty already has a scoring function, so a network can take over without the rest of
the program changing.

- **Grammar.** In `blimp_open.rs`, `Her::score` (line 470) returns the log probabilities of a
  sentence. The plain reading is `Her` with no openers. (`Naive::score` (624) is a slow
  reference that stores every n-gram, used by `check`.) A trained network gives
  log p(sentence) directly.
  - Define a trait such as `trait Scorer { fn logprob(&self, words: &[String]) -> f64; }`.
  - Implement it for the network. Keep the count model behind it only as the yardstick.
- **Talk.** In `compose.rs`, her next-word vote comes from `Stage::ask` (line 1153, decoded by
  the readout) and `native_k` (1117).
  - The `step` closure (around line 2957) combines them with the delayed-synapse readout and
    the engram. The network's next-word distribution replaces this.
  - (`Convo::vote` (1282) is an optional conversation-trace add-on, off by default.)
  - The reply exam in `compose.rs` measures whole replies, not just next words.
- **Memory.** Engram recall in `compose.rs` matches the words of a lesson. Learned keys make
  recall work by meaning.
  - Earlier "meaning from neighbours" attempts failed because *related* is not *the same*
    ("hot" sits close to "cold").
  - So train the keys to tell apart the lessons that are confused with each other, not just to
    group similar words.
- **Science.**
  - `read_think.rs` `recall` (450) finds sentences by word overlap; `scores` (519) scores each
    choice; `pick` (526) chooses and gives a margin.
  - `answers` (538) abstains below a margin chosen on Dev (`choose_theta`, 560).
  - Replace `recall` and `scores` with networks, and keep that abstention.
- **ARC puzzles.**
  - `arc.rs` tries the operations in `ops()` and the learned steps in `learners()` inside
    `solve`. A `Learner` is a boxed function from the example grids to proposed outputs
    (`type Learner`, near the top of the file).
  - Add a network as one more `Learner`, and as a prior that decides which steps to try first.
  - Over time, let the network propose the operations themselves, so the hand-written list can
    go.
  - Keep the self-check: it is why she gives only 3 wrong answers in 400.

## 5. A staged path
Put each stage behind a switch that is off by default, as all her experiments are.

**Stage 0: a home for the network.** Add a new crate, for example `nn/`, next to
`design-tests/`. Her programs call it only when a switch such as `AVA_NN=1` is set.
*Status: done.* It became three crates under `design-tests/nn/` (`data`, `grad`, `brain`), each
detached from the old crate, so the old programs are untouched and need no switch.

**Stage 1: tokenizer.**
- Write a byte-level BPE in Rust and train it on her reading.
- A vocabulary of 16k-32k fits in 16 bits, and the tokenizer is deterministic.
*Status: done* (`nn/data/src/bpe.rs`, `bpe_train`). The v32 runs used a 4096 vocabulary; digits
are always single tokens.

**Stage 2: a streaming, sharded reader.** This is what makes "trillions of words" a question
of disk and time rather than RAM.
- Tokenize the text once into binary shards of token ids.
- Read through memory mapping or buffered reads, never whole.
- Shuffle the order of shards by seed, with a buffer inside each shard.
- Save a cursor (shard, offset, seed) with each checkpoint, so training can resume exactly.
- Count with `u64` everywhere.
*Status: done* (`nn/data/src/shards.rs`, `make_shards`): `AVATOK01` shards of 50M tokens, one
shard in memory at a time, a seeded shuffle over shards and slots, and a one-line cursor that
resumes on exactly the next window.

**Stage 3: tensors, gradients, an optimiser.**
- **From scratch:** f32 tensors, reverse-mode autodiff, AdamW, and a blocked matrix multiply on
  threads. Check every gradient against finite differences. Fine up to roughly 10M weights on a
  CPU.
- **A Rust crate:** [candle](https://github.com/huggingface/candle) or
  [burn](https://github.com/tracel-ai/burn). Both are Rust and run on GPUs. Use this route to
  go big.
- Add the spiking pieces on top:
  - leaky integrate-and-fire layers unrolled over time steps;
  - a surrogate gradient for the spike;
  - k-winners-take-all layers.
*Status: the from-scratch route is done* (`nn/grad`): f32 tensors, a define-by-run tape, AdamW,
a blocked threaded GEMM, every gradient checked against finite differences, and the two brain
ops (`lif_scan` with backprop through time and the fast-sigmoid surrogate, `kwta`). The crate
route is now taken too: `nn/gpu` is the spiking model on candle 0.11 with CUDA (a straight-through spike with the same surrogate, a top-k mask for kWTA, the same AdamW), which trained Ava-M (d 384, 8 layers, 13.4M weights) on one consumer GPU. It is launch-bound until the LIF scan gets a fused kernel.

**Stage 4: her brain network, and a yardstick.** Train two models of the same size on the same
reading (her roughly 20M words):
- **Ava's network:** spiking layers, learned sparse codes and sparse experts.
- **A plain transformer** of the same size, also in Rust. It is the yardstick, never part of her.

Training cost is about 6 × weights × tokens FLOPs. For 30M weights and 300M tokens that is about
5×10¹⁶: under an hour on one consumer GPU, or days on a CPU. For targets, look at the
[BabyLM challenge](https://babylm.github.io/), which trains on 10M or 100M words. Her count
model scores 56% on BLiMP.
*Status: done in v32* (`nn/brain`), with three pre-registered models rather than two: T, the
yardstick transformer; S, Ava spiking (spiking memory + k-winners MLP, no attention, no position
embedding); and H, a hybrid with attention. Sparse experts are not built yet. The first released
checkpoint, Ava-S (S, d 192, 6 layers, 48M tokens), is in `release/ava-s/` with its model card;
`generate`, `talk` and `score` run it.

**Stage 5: the scaling test.** This is the question that matters most. Train both models at three
sizes, for example 1M, 10M and 100M weights, and plot the score against size.
- If her network's gap to the transformer holds or shrinks as size grows, her design scales.
  That would be a new result.
- If the gap widens, find which brain mechanism costs the most, and test it alone.
- Apply the same test to each mechanism: a mechanism earns its place only if its gain holds or
  grows with size.
*Status: run once, in v32*, at d64 L2, d96 L3 and d128 L4 with a fixed budget of 4M tokens and
one learning rate. S beat T at every size (2.357 / 2.316 / 2.274 against 2.382 / 2.330 / 2.282
bits per byte) with fewer weights, but its lead shrank with size, and H went from better than T
to worse; by the rule above, both designs lose ground. Three data seeds of T at the smallest
size spread about 0.007 BPB. Because every size saw the same 4M tokens, this is not yet a fair
scaling result: the next run gives each size tokens in proportion to its weights and sweeps the
learning rate. Numbers and caveats are in [README.md](README.md), the full lines in
[ARCHITECTURE.md](ARCHITECTURE.md).

**Stage 6: memory, sleep, reward and self-knowledge.** Wire in her other faculties:
- the engram memory with learned keys;
- sleep, which replays memories into the weights;
- learning from right/wrong for arithmetic and puzzles;
- the calibrated "I don't know".

Each one is a switch, and each passes its own test before it is kept.

**Stage 7: scale.**
- bf16 mixed precision, gradient checkpointing, several GPUs, and resuming from stage 2's cursor.
- Size the model to the data. A common rule of thumb is about 20 training tokens per weight, so
  a trillion tokens suits a model of about 50B weights. That is a cluster job, but by this stage
  the code no longer stops it.

**Stage 8: talking.** Fine-tune on dialogue (the corpora `compose.rs` reads). Keep her reply
checks: proper endings, how familiar the question feels, and her confidence reading (shown in
brackets, never spoken as a canned phrase).

## 6. Prior art
Spiking language models exist. Read these before starting:
- **SpikeGPT** (Zhu et al.; TMLR 2024, ICLR 2025): a spiking language model inspired by RWKV,
  with 46M and 216M weights, trained with surrogate gradients.
  [arXiv 2302.13939](https://arxiv.org/abs/2302.13939)
- **SpikingBrain** (Pan, Li et al., Chinese Academy of Sciences, 2025): spiking models at 7B,
  and 76B with experts. They were **converted from a trained transformer** (Qwen2.5-7B) with
  about 150B tokens of further training, not trained as brains from the start.
  [arXiv 2509.05276](https://arxiv.org/abs/2509.05276)

What neither does, and Ava aims to, is combine spiking neurons with:
- an engram memory and sleep replay;
- a self-model that abstains;
- learning from reward;
- search with a self-check;
- a design that is brain-like from the start.

## 7. Keep it honest
These rules are why her numbers can be trusted. Keep them as she grows.

- **Byte-identical when off.** Every new mechanism has a switch. With the switch off, outputs
  match saved reference files byte for byte.
- **Pre-register.** Before a test, write down the build's sha256, the switches, the data and
  the pass rule. Run it once and apply the rule by its letter. Disclose every earlier look at a
  test set.
- **Right − wrong.** She may abstain, and a confident wrong answer counts against her. Report
  right − wrong and the share right among the answers given, not only accuracy.
- **Statistics.** Use a paired bootstrap with 20,000 resamples and seed 1 (`paired_boot.rs`).
  A gain counts only if its 95% interval lies above 0.
- **Scaling curves.** Report each result at three or more sizes, next to the yardstick
  transformer. A gain at one size is not a scaling result.
- **Contamination.** Once she reads web-scale text, test sets will be in it.
  - Before training, remove every document that shares a long n-gram (for example 13 words) with
    any test set, and write down what was removed.
  - Her dialogue exam already shows the danger: about a quarter of its pairs were in her
    training text.
- **Fresh test sets** (unopened during development, as of 2026-10-27):
  - ARC-AGI-2 evaluation, ConceptARC MinimalTasks, the OpenBookQA test, CLAMS (English),
    PoSH-BENCH and Mini-ARC.
  - **Spent** (already used): ARC-AGI-1 evaluation, the BLiMP test, the ConceptARC corpus, Zorro,
    the ARC test and the SciQ test (opened once in v37; Ava-QA's SciQ number is a disclosed second look).
- **No hand-written knowledge.** No word lists, canned phrases or per-puzzle code. What she
  does should come from what she read or practised.
