# Model card: Ava 0.4

Ava 0.4 is a small language model built like a brain, and the first Ava you can talk to. She has spiking neurons,
a fast memory that she writes and reads as she goes, and a library she looks things up in before she answers. She
comes with a chat app for Windows. Behind the chat, a 3D brain lights up from her real activity, word by word.

She keeps learning while you talk. A small part that works like the cerebellum (CB2) adjusts her next-word guesses
from what she reads and says. Each chat keeps its own changes, and they slowly fade. Her saved weights never change.

The brain in this release is **the v52 brain, answer round 3**.

| | |
|---|---|
| Model | the v52 brain, answer round 3 (model D; block options in `arch.txt`) |
| Release date | 2026-11-11 |
| Author | nochallenge |
| Licence | PolyForm Noncommercial 1.0.0 (source-available) |
| Files | `model.safetensors` (candle VarMap, f32; a download on the v0.4.0 release page as `ava-0.4.safetensors`), `arch.txt`, `bpe.txt`, `config.txt`, `result.tsv`, this card, `SHA256SUMS` |
| Checkpoint sha256 | 269da7da2175c95fdd08f1c7c061a4fb2d89509c342ba221766e03a28947276d |
| Tokenizer sha256 | b1ccf0b918a88f0f421d7f3bd91b56f911a4d9e47c246e58ddb685041850f49b (the same file as Ava 0.3's) |
| Size | d = 384, 8 layers, 6 memory heads of 64; 26,840,893 weights |
| Source commit | the commit tagged `v0.4.0` |

## What is new in her design

Each of her 8 blocks has spiking neurons whose membranes carry the past, a fast-weight memory that she writes and reads at every word, and a wide layer where only some units may be active. Block 4 also has one attention read (the engram), and she keeps Ava 0.3's glance and magnitude code for numbers. New since Ava 0.3 are these block options, written in `arch.txt`:

- `--head-size 64`: her fast memory has 6 heads of 64 numbers each (Ava 0.3: 32).
- `--timescales fixed`: each memory head forgets at its own fixed rate, spread evenly on a log scale from fast to slow.
- `--gates channel`: each memory channel has its own forget rate and write strength, with a separate key for erasing (as in Kimi Linear and RWKV-7).
- `--value-residual`: every memory block can reuse the values of the first one.
- `--kwta 1/4`: in the wide layer a quarter of the units may be active at once (k-winners-take-all; Ava 0.3: one in eight).
- `--mlp-act relu2`: the winners' activity is squared (ReLU squared).
- `--mem-momentum 0.5`: memory writes have momentum: each write carries half of the one before it.
- `--resid-fix`: the weights that write into the main stream start smaller, counting every write a block makes.
- `--residual dense`: each block can read the outputs of all earlier blocks (DenseFormer).
- `--lif-init logspaced`: her neurons' leaks start spread from 2 to 512 tokens.
- `--lif-scale`: a neuron's input is scaled by how fast it leaks.
- `--spikes int2`: her neurons send small signed spikes, from -2 to +2, instead of only 0 or 1.
- `--alif`: firing thresholds adapt: a slower trace of recent spikes raises them.
- `--homeostasis 0.1`: each neuron has its own firing threshold, with a rule that can nudge it toward a 10% firing rate (switched on while she grew and was taught).

**New in v52.** On top of Ava 0.4's 14 block options, she has the brain parts v52 kept (each tested alone first, and kept by a rule written down before the runs): short-term plasticity: each spiking neuron's output tires and recovers with use (depression and facilitation); neuromodulators: a small nucleus reads the first block and sends three signals to the later blocks, like noradrenaline (input gain), acetylcholine (memory write strength) and dopamine (output scale); dendrites: each unit of the wide layer is scaled by a gain from its block's memory read, like the top dendrites of cortical neurons; letters: every word piece also carries a small vector made from its own letters and their places, so she can see spelling; dopamine: from the second block on, the wide layer's output is scaled by how surprised she was by the last word; mood: four slow units follow how surprised she has been lately and scale the input of every later block's neurons; rhythms: every block's memory writes and reads are scaled, per head, by slow and fast rhythms, like theta and gamma waves; a thalamus: a small relay of 16 units reads every block's output and sets graded gates on what each block passes on; a cerebellum: 12,288 granule units make a sparse code of her state and the current word, and a learned read-out corrects her next-word guess; it is the part that keeps learning in the chat app; the homeostasis nudge: the rule that moves each neuron's threshold toward its target firing rate stays on while she grows and is taught. Her training also changed: the spike reset now passes its gradient back (`--lif-reset-grad`), the fix for the training spikes found in v52.

## How she was made

She started from the v42 brain s5-v42, which had read 387 million tokens of the new v42 mix. She then read 108 million more tokens of the v41 mix with v52's changes, with 42,140 question-answer pairs mixed in (13,200 steps, the last quarter with 8,192 tokens of context), and was taught to answer in 3 short rounds; rounds 2 and 3 were made from her own practice answers.

The full settings are in `config.txt`, and each training phase's result is in `result.tsv`.

## Evaluation

Reading scores are bits per byte on held-out text from her own corpus (v41-mix); lower is better. Every test was
written down before it ran, and each was run once.

| | Ava 0.3 | Ava 0.4 |
|---|---|---|
| Reading, bits per byte (lower is better) | 1.511 | 1.513 |
| Grammar, BLiMP | 74.1% | 75.9% |
| Science multiple choice (dev set) | 46.8% | 41.7% |
| Simple questions: right / "I don't know" / wrong, of 100 | – | 54 / 31 / 15 |
| Fresh sealed test, run once | – | not run |
| Talking speed in the app (CPU, int8) | – | 425 tokens/s |

**Disclosures.** v52's first bake-off of brain parts was stopped and set aside when a fault in the training step was found (the spike reset passed no gradient back); it was run again with the fix, which this brain has. 13 amendments (F-1, S1-A, S1-B, S1-C, S1-D, S1-E, S3-A, S1-F, S1-G, S3-B, S4, S1-H, S1-I) changed v52's rules while it ran, each written down before the results it governs. 11 parts were tried; kept: letters, mood, neuromodulators, short-term plasticity, a cerebellum, dendrites, dopamine, rhythms, a thalamus, the homeostasis nudge. This brain replaced Ava 0.4b because it passed all 7 checks written down before the final run. The shipped answer round is the best of three on the 100 dev questions (right minus wrong: 11, 36, 39). Her grammar and science scores were measured on this brain after it was picked, with the same tools and dev sets as the other dev scores. One rule written down for the release fails: on the science multiple-choice dev questions she is more than 2 points behind Ava 0.3 (see the table), as Ava 0.4b was. Because of that the fresh sealed test was not opened, and releasing her is the owner's decision, not the rule's. Test sets used up so far: SciQ's test (v37) and Ava 0.3's fresh sets (v41).

## The chat app

The app is `ava_live` from the GPU crate, built for the CPU, and ships as `Ava-0.4-windows.zip` on the release
page, with her brain inside as `model.safetensors`. Her brain alone is also a download, `ava-0.4.safetensors`.

- She runs with 8-bit weights (int8), so she is quick on an ordinary processor. No graphics card and no internet
  are needed. She uses about 380 MB of memory.
- She learns live with CB2 only: the cerebellum's online table (12,288 granule units) learns beside the int8
  weights, and its changes fade back with a half-life of 256 tokens. E-prop in her last two blocks (E2) is
  available with `--learn e2,cb2`, but then she talks in f32 and more slowly.
- Each chat and its changes are saved in `chats\` beside the app. Delete a chat and she forgets what she learned in it.
- The app only answers this computer (127.0.0.1). Nothing you type leaves it.

## How to use her

**The app.**

1. Download `Ava-0.4-windows.zip` from the v0.4.0 release page. It holds the app and her brain.
2. Unzip the folder.
3. Double-click `Ava.exe`.

**From the repository.** Download `ava-0.4.safetensors` and save it as `release/ava-0.4/model.safetensors`. Check
it with `sha256sum -c SHA256SUMS`. The tools read `arch.txt` beside the model by themselves. Then, from the
repository root:

```
design-tests/nn/gpu/target/release/generate_gpu --model release/ava-0.4/model.safetensors \
  --bpe release/ava-0.4/bpe.txt --arch D --d 384 --layers 8 --glance \
  --prompt "The cat sat on the" --tokens 40 --temperature 0.8 --seed 1
```

To run the chat app from source, build `ava_live` and give it the same files and her library:

```
design-tests/nn/gpu/target/release/ava_live --model release/ava-0.4/model.safetensors \
  --bpe release/ava-0.4/bpe.txt --library library.tsv --int8 --learn cb2 --open
```

`library.tsv` is in the app download. `ava_live --help` lists every flag.

## Intended use and limits

Use her for research on brain-like language models, on learning while she talks, and on small models that look
things up. You may also use her for any other non-commercial use the licence permits. Do not use her for facts,
decisions or anything in production.

- She is a tiny research model. Her answers are short and often wrong or odd.
- She speaks English only. The app is for Windows 10 and 11 (64-bit).
- What she learns in a chat stays in that chat and fades. It never changes her saved brain.
- There is no safety tuning.
- The GPU path is not bit-reproducible.

## Licence and credits

Ava 0.4, her weights and her tokenizer are released under the
[PolyForm Noncommercial License 1.0.0](../../LICENSE.md). Keep the
`Required Notice: Copyright (c) 2026 nochallenge` line with any copy you share.

Two parts of the app download keep their own licences:

- **The 3D brain in the app**: BodyParts3D, (c) The Database Center for Life Science (DBCLS), licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  ([source](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/)). Only the brain's parts are used, turned into
  the hologram picture.
- **Her library** (`library.tsv`): almost all of it is text from the Simple English Wikipedia (the copy of
  1 November 2023), written by Wikipedia's contributors and licensed
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). It was changed: each article's opening text
  was cut into short passages at sentence ends. Because of that licence, `library.tsv` is shared under CC BY-SA
  4.0 as well. The other passages (about 1,400) are short facts written for her lessons.

The training data keep their own licences: the v42 reading mix (FineWeb-Edu, ODC-By; the Simple English Wikipedia, CC BY-SA 4.0; the ARC corpus; Tatoeba; BabyLM's CHILDES, BNC spoken, Switchboard, OpenSubtitles and Gutenberg parts; PG-19 books, Apache 2.0; the Cornell Movie-Dialogs Corpus and DailyDialog; reasoning practice, facts and lessons written for her), the v41 mix (the same kind of text, 337 PG-19 training books and number practice) and the questions she was taught with (SciQ, CC BY-NC 3.0; Dolly 15k, CC BY-SA 3.0; ARC, CC BY-SA 4.0; OpenBookQA; SQuAD 2.0), each under its own licence. The data files are not distributed.
