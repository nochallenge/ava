# Model card: Ava-QA

Ava-QA is [Ava](../ava/MODEL_CARD.md), the v0.2 release model, taught to answer questions. The teaching
took 95 seconds on one RTX 4070. While she was taught, she also re-read her ordinary text, as the
brain replays old memories while it learns something new. The replay kept the answering skill and
cut her forgetting of plain reading by about three quarters.

Her design, size and tokenizer are exactly Ava's. Only the weights differ.

| | |
|---|---|
| Model | Ava-QA (Ava taught with `sft_gpu`; kind `D`) |
| Release date | 2026-10-27 |
| Author | nochallenge |
| Licence | PolyForm Noncommercial 1.0.0 (source-available) |
| Files | `model.safetensors` (candle VarMap, f32; a download on the v0.2.0 release page as `ava-qa.safetensors`), `bpe.txt`, `config.txt`, `result.tsv`, `train_log.tsv`, `dev_scores.txt`, this card, `SHA256SUMS` |
| Checkpoint sha256 | 41b907a28365e7dbae7e8b8810b160867664b3eb84051ef278ef777bdbb2681d |
| Tokenizer sha256 | b1ccf0b918a88f0f421d7f3bd91b56f911a4d9e47c246e58ddb685041850f49b (the same file as Ava's) |
| Starting point | Ava, checkpoint sha256 d32901b530b62e3325548b2896f246157385d12f9127eb2a47ec3f8a58592ad1 |
| Source commit | the commit tagged `v0.2.0` |

## How she was taught

**The examples.** 40,089 question-answer pairs (5,216,142 tokens), made by `qa_prep` from the training
splits only:

- **SciQ** train (Allen AI, CC BY-NC 3.0): 11,679 science questions; most come with a support passage;
- **Dolly 15k** (Databricks, CC BY-SA 3.0): human-written instructions and answers;
- **ARC** Easy and Challenge train (Allen AI, CC BY-SA 4.0): the text of the right choice is the answer.

Each example is `[EOS] prompt answer [EOS]`:

```
Question: {question}
Answer: {answer}
```

When an item has a passage, `Context: {passage}` comes first, on its own line. Only the answer and
the closing EOS are scored, so she learns to answer, not to ask. Examples over 512 tokens had their
passage shortened (1,704 passages), and 452 Dolly items whose answers alone were too long were
dropped.

**Replay.** Every teaching step also re-reads 16 random windows of 256 tokens of her ordinary
reading (train shard 0). The step's gradient is the answer gradient plus 1 × the reading gradient.

| | |
|---|---|
| Steps | 640 (2 epochs; batches of at most 16,384 tokens, sorted by length, padded with PAD, a new seeded order each epoch) |
| Learning rate | 0.001 peak, warm-up 50 steps, cosine decay to 0.1 of the peak |
| Optimiser | fresh AdamW, β₁ 0.9, β₂ 0.95; weight decay 0.1 on the matrices only; gradient clip 1 |
| Seed | 1 |
| Precision | bf16 matmul operands with f32 accumulation; everything else f32 |
| Time | 95.3 s at 136,670 tokens per second (padding included), plus 7 s of evaluation |
| Command | `sft_gpu --init release/ava/model.safetensors --arch D --d 384 --layers 8 --data data/qa/sft/train.sft --epochs 2 --lr 1e-3 --replay-rows 16 --replay-ctx 256 --replay-weight 1 --warmup 50 --min-lr-frac 0.1 --wd 0.1 --clip 1.0 --seed 1 --out DIR` |

**Pre-registered.** Both choices were made by rules written down before the runs:

- **The learning rate (v37).** Of 1e-4, 3e-4 and 1e-3, the highest SciQ dev accuracy, closed book:
  1e-3.
- **The replay weight (v38).** Of 0.5, 1 and 2, the lowest validation BPB among runs that kept SciQ dev
  closed book at 55.5% or more: 1. Weight 2 fell under the bar (53.7%).

## Evaluation

Every choice of a multiple-choice question is scored as an answer, `[EOS] prompt " choice" [EOS]`, by
its total log-probability; she picks the most likely, and a tie counts as wrong. She may also say "I
don't know": she answers only when her best choice beats the second by at least a margin `tau`, fixed
on the dev set. She is judged by right minus wrong, and by the share right among the answers she gives.

| | Ava (before) | Ava-QA |
|---|---|---|
| SciQ test, closed book (4 choices; chance 25%) | 41.1% | **56.4%** |
| SciQ test, passage shown | 45.9% | **66.2%** |
| closed book with "I don't know": right / wrong / abstained | 54 / 33 / 913 | 420 / 233 / 347 |
| closed book: right − wrong; share right among answers | +21; 62.1% | **+187; 64.3%** |
| her reading: validation bits per byte (lower is better) | 1.487 | 1.550 |
| BLiMP grammar pairs (never taught) | 76.1% | 73.1% |

| Dev sets (`dev_scores.txt`) | Ava-QA |
|---|---|
| SciQ dev, closed book | 57.3% (tau 1.3887) |
| SciQ dev, passage shown | 65.4% (tau 1.0419) |
| ARC Easy dev | 43.9% |
| ARC Challenge dev | 19.1% (below the 25% chance level; with "I don't know" she abstains on every question) |

**The SciQ test.** The test file was sealed when it was downloaded and opened once, in v37, by the
pre-registered test of the first taught model (56.8% closed book, 66.4% with the passage). The Ava-QA
numbers above are a disclosed second look at that spent set, made once, after Ava-QA was chosen on
dev. Against the v37 model the difference is −0.4 points closed book (95% interval −2.2 to +1.4) and
−0.2 with the passage: no clear change in answering, and much less forgetting.

**Forgetting.** Teaching without replay (v37) cost her +0.261 BPB of reading and 5.7 points of BLiMP.
With replay the cost is +0.063 BPB and 3.0 points.

## Samples

The first three questions asked, greedy (`--temperature 0`):

```
Question: What gas do plants take in from the air to make food?
Answer: carbon dioxide
Question: At what temperature does water boil at sea level?
Answer: 100 degrees
Question: What is the closest star to Earth?
Answer: the sun
```

Free-form answers are often wrong. In a v38 check of 12 questions she also answered "Belgium",
"Ronald Reagan" and "2" legs to questions with other answers, and repeated herself on questions
about names and health.

## How to use her

Download `ava-qa.safetensors` from the v0.2.0 release page, save it as
`release/ava-qa/model.safetensors`, and check it with `sha256sum -c SHA256SUMS`. Then, from the
repository root:

```
design-tests/nn/gpu/target/release/generate_gpu --model release/ava-qa/model.safetensors \
  --bpe release/ava-qa/bpe.txt --arch D --d 384 --layers 8 --qa --temperature 0 \
  --prompt "What gas do plants take in from the air to make food?"
```

`--qa` asks the prompt in her taught format and stops at her EOS. `--chat` asks question after
question, each from a fresh memory. `qa_eval` scores the multiple-choice sets above; its header
lists the flags.

## Intended use and limits

Use her for research on brain-like language models, on teaching a small model new skills without
forgetting, and on calibrated "I don't know", and for any other non-commercial use the licence
permits. Do not use her for facts, decisions or anything in production.

- She is a tiny research model. Her answers are short and often wrong; the multiple-choice scores are
  her real measure.
- She was taught to answer single questions, not to hold a conversation. There is no safety tuning.
- One teaching run, one seed. The GPU path is not bit-reproducible: the same seeds give results that
  differ slightly from run to run.

## Licence

Ava-QA, her weights and her tokenizer are released under the
[PolyForm Noncommercial License 1.0.0](../../LICENSE.md). Keep the
`Required Notice: Copyright (c) 2026 nochallenge` line with any copy you share. The teaching data keep
their own licences (SciQ CC BY-NC 3.0, Dolly 15k CC BY-SA 3.0, ARC CC BY-SA 4.0); the data files are
not distributed.
