# Model card: Ava 0.3

Ava 0.3 is one model that reads, counts, reads whole books, answers questions, and **sleeps**: she can
read a day's new facts, sleep on them, and know them afterwards. While she sleeps, the synapses her
reading relies on are held in place (synaptic consolidation), so learning the new costs little of the
old.

She is the v0.2 design (spiking neurons, an error-correcting fast-weight memory, sparse k-winner codes)
with three additions: a glance that sees how long a number is before reading its digits, a sense of
size for numbers, and one attention read in her middle block (the engram). The tests below show that
she uses the glance and does **not** use the engram or the sense of size.

| | |
|---|---|
| Model | Ava 0.3 (kind `D` with glance marks; `grow_gpu`: grow, sleep, reward) |
| Release date | 2026-10-28 |
| Author | nochallenge |
| Licence | PolyForm Noncommercial 1.0.0 (source-available) |
| Files | `model.safetensors` (candle VarMap, f32; a download on the v0.3.0 release page as `ava-0.3.safetensors`), `bpe.txt`, `config.txt`, `result.tsv`, this card, `SHA256SUMS` |
| Checkpoint sha256 | 393a1714ff2121edab9cf4ffd097e27c8fa5efa8f31d13d3459a156af3174329 |
| Tokenizer sha256 | b1ccf0b918a88f0f421d7f3bd91b56f911a4d9e47c246e58ddb685041850f49b (Ava's; 16 length marks are added in code, vocab 4,112) |
| Size | d 384, 8 layers, 12 memory heads; 19,962,048 weights |
| Source commit | the commit tagged `v0.3.0` |

## What is new in her design

- **The glance.** Before every run of digits she sees one mark for its length (1 to 16+ digits), and
  each digit gets a learned code for its place (ones, tens, hundreds...). People see that 1,200 is
  longer than 950 before they read it; she now does too. The marks carry no text (0 bytes), and she must
  predict them like any other token, so her bits-per-byte stays a fair score. Switching the place code
  off costs her 8 points on the number dev set, so she uses it.
- **A sense of size** (`mag_w`): a learned code for each number's rough magnitude. Switching it off
  changes 0.3 points: she does not use it yet.
- **The engram**: one attention read in block 4, added so she could look things up in long texts. It
  starts switched off (read-out at 0) and was trained with recall practice. Switching it off changes
  her long-text score by 0.001 bits per byte or less after the first 256 tokens, and the find-a-fact
  test failed: she does not use it. Apart from this one read, she has no attention and no position
  embedding.
- **Sleep with an anchor.** A sleep phase re-reads the day's episodes, the most surprising ones more
  often, while also re-reading her ordinary text. Before sleep, the trainer measures how much her
  reading leans on each weight (the mean squared gradient over 16 batches of ordinary text, like EWC).
  After every sleep step, each weight is pulled back towards its value before sleep, in proportion to that
  importance. Weights her reading needs hardly move.

## How she was made

1. **v39 (numbers):** Ava's recipe from scratch, with the glance and 2% child-style number practice
   in her text: 387,083,520 tokens, 45 minutes on one RTX 4070 at 143,873 tokens per second. Val
   1.4877 BPB. Checkpoint d7c061dd….
2. **Grow (v41):** 108,134,400 more tokens (13,200 steps of 8,192) of a new mix:
   - her usual text 70% (Tatoeba, dialogue, Cornell movie lines, the ARC corpus);
   - 337 whole public-domain books from the PG-19 training split 25%;
   - number practice 5%;
   - answer batches from 42,140 question-answer pairs (SciQ, Dolly 15k, ARC and OpenBookQA training
     splits, with every dev and test question removed);
   - recall practice: in every second step, made-up facts are planted in a text and asked about later.

   The first three quarters used windows of 2,048 tokens; the last quarter used 8,192 tokens of new text.
   34 minutes at 58,004 tokens per second, sharing the GPU with a video player.
3. **Sleep:** 300 steps on 150 made-up people (each with a birth year, home town and job), anchor
   strength 100: 25 seconds.
4. **Reward:** 300 steps of learning from right and wrong on 18,176 multiple-choice training questions,
   with the same anchor: 51 seconds.

The anchor strength (100) and the scoring rule (per byte) were picked on dev data by rules written
before the run. One made-up first name in the recall-practice generator was changed after the run, so
regenerated practice text differs from what she read only where that name was drawn. Exact settings are in `config.txt`; the three phases' result lines are in `result.tsv`.

## Evaluation

Every test was written down before the run, opened once, and judged by its pre-registered rule.

| Test | Result | Verdict |
|---|---|---|
| Numbers (842 new items: comparisons, counting) | 60.5% vs 42.4% for the same recipe without the v41 phases (v39): **+18.1 points [+12.9, +23.0]**. Comparing numbers of different lengths: 90.5% (v39 46.5%) | pass |
| Long text (100 unseen books, 16,384 tokens each) | her memory over the whole book beats a memory cut every 256 tokens by **0.033 bits per byte [0.031, 0.035]** at positions 4,096–16,384 | pass |
| Find a planted fact (400 items) | engram on − off: +1.3 points [−0.8, +3.3]; 27% (chance 25%) | **fail** |
| Sleep (450 questions about the 150 new people) | **96.2% after sleep vs 23.8% before** (chance 25%) | pass |
| Questions (1,200 held-out SciQ, ARC and OBQA questions, 4 choices) | **40.7% vs v39's 33.0%: +7.7 [+4.9, +10.3]**. SciQ 56.5%, ARC 34.8%, OBQA 30.8% | pass |
| Right − wrong after reward, vs before it | −0.013 [−0.036, +0.011] | **fail** |
| Reading (val bits per byte; rule ≤ 1.497) | **1.511** | **fail** (narrowly) |

Brackets are 95% bootstrap intervals. Also measured, with no pass rule:

- BLiMP grammar pairs: **74.1%** (v0.2 Ava 76.1%; chance 50%).
- The anchor on dev: sleeping without it cost her reading +0.046 BPB; with it, +0.016, for 5 points less
  recall of the day (100% → 94.9%).
- Comparisons show the human distance effect: far pairs 98.9%, near pairs 43.6%.
- The 1,200 questions come from held-out slices of the same training splits, not the official test
  splits (those were spent by earlier versions). They test new questions of a familiar kind.
- With the new people's facts in view (no sleep), she answers 23.8%, at chance: she cannot yet look up
  a fact she has just read. She learns it by sleeping on it.

## Samples

Seed 1, temperature 0.8, the four pre-registered prompts, unselected:

```
I have 3 apples and 4 more, so now I have 7.
Lawrence, E. J. (1964).

Which is more, 1,200 or 950? The answer is 500. From big to small: 8806, 1493, 95, 32.

The old house at the end of the lane was covered
with lakes, and on the side of the lake I came across two of the lake
towers, surrounded by a mound of plastic or sand of bamboo-float-glass

Question: Why do leaves change colour in autumn?
Answer: A shimmering amount of sunlight.
```

The first is right. The second answers wrongly, then drifts into practice-like text. The third is
fluent but loose. The fourth has the right form and no real knowledge.

## How to use her

Download `ava-0.3.safetensors` from the v0.3.0 release page, save it as
`release/ava-0.3/model.safetensors`, and check it with `sha256sum -c SHA256SUMS`. Then, from the
repository root:

```
design-tests/nn/gpu/target/release/generate_gpu --model release/ava-0.3/model.safetensors \
  --bpe release/ava-0.3/bpe.txt --arch D --d 384 --layers 8 --glance \
  --prompt "I have 3 apples and 4 more, so now I have" --tokens 40 --temperature 0.8 --seed 1
```

`--glance` is required: it adds the length marks her tokenizer does not write. The magnitude code and
the engram are found in the file. `--qa` asks a question in her taught format.

To teach her new facts yourself, write them one per line and let her sleep on them:

```
design-tests/nn/gpu/target/release/grow_gpu --phase sleep --init release/ava-0.3/model.safetensors \
  --facts my_facts.txt --train my-shards/train --val my-shards/val --arch D --bpe release/ava-0.3/bpe.txt \
  --sleep-steps 300 --sleep-rows 16 --cycles 4 --surprise-alpha 1.0 --replay-rows 8 --replay-ctx 256 \
  --replay-weight 1.0 --anchor 100 --fisher-batches 16 --lr 3e-4 --warmup 30 --out runs/my-sleep
```

`--train` and `--val` are shards of ordinary text for her to re-read while she sleeps: make them with
`make_shards` (see the README), then add her length marks with
`num_shards --src my-plain-shards --out my-shards --bpe release/ava-0.3/bpe.txt --marks`. It takes about half a minute on an RTX 4070. Every flag is listed
at the top of `grow_gpu.rs`.

## Intended use and limits

Use her for research on brain-like language models, on learning new facts in sleep without
forgetting, and on number sense in small models, and for any other non-commercial use the licence
permits. Do not use her for facts, decisions or anything in production.

- She is a tiny research model (20M weights). Her answers are short and often wrong; the
  multiple-choice scores are her real measure.
- She slept on 150 made-up people (the sealed sleep test's facts, read by design). They are fiction.
- The engram and the sense of size are in her weights but unused. Reward learning added nothing
  measurable.
- Her reading is slightly worse than v0.2's Ava (1.511 vs 1.487 BPB; BLiMP 74.1% vs 76.1%).
- One run, no twin: the effect of each v41 addition cannot be separated. There is no safety tuning and
  no conversation format.
- The GPU path is not bit-reproducible.

## Licence

Ava 0.3, her weights and her tokenizer are released under the
[PolyForm Noncommercial License 1.0.0](../../LICENSE.md). Keep the
`Required Notice: Copyright (c) 2026 nochallenge` line with any copy you share. The training data keep
their own licences (among them SciQ CC BY-NC 3.0, Dolly 15k CC BY-SA 3.0, ARC CC BY-SA 4.0, PG-19
Apache 2.0); the data files are not distributed.
