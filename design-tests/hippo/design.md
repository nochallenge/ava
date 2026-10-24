HIPPOCAMPUS DESIGN for compose.rs (text only; no files edited, nothing compiled or run except light read-only Python)

## 0. Evidence gathered before designing (v18's lesson: measure the ceiling first)

I ran three light Python checks on data/triples.tsv. They read the data only and used no model. The scripts were in the session scratchpad (hippo_ceiling.py, fam_dev.py, hvote4.py). The tokenizer emulation reproduces the exam's 11,999 positions exactly.

**A. Familiarity works, and it has to be ORDERED BIGRAMS.**
- AUC of normal vs strange prompts, on the exam's 1,200 prompts:
  - Ordered prompt bigrams: mixed-up 0.933, reversed 0.944.
  - Unordered content-word pairs: mixed-up 0.701, reversed 0.500 (blind to word order by construction).
  - Bigrams and pairs averaged together: 0.854 / 0.806. The pairs dilute the bigrams, so pairs are dropped.
- I chose the read on a DEV block, never the exam. The dev block is filtered rows 31,200–32,399, which are outside both the exam and training. The best read was the mean of ln(1+c) over bigrams with sentence boundaries (`<s> w0` … `wn </s>`): dev mixed 0.969 / reversed 0.993. The exam values, printed alongside, were 0.965 / 0.996.
- Counting prompts only, each dialogue turn is counted exactly once, because triples.tsv is a sliding window and every turn appears once as some line's prompt. This also makes leave-one-out exact.
- Percentile φ of a prompt's familiarity, measured against her own training prompts (leave-one-out): normal prompts have median 0.50; mixed-up prompts have median 0.02, and 94% of them fall under 0.25.
- Size: 68,528 distinct boundary bigrams at 30k.

**B. The episodic copy route is predicted to be small on this exam.**
- 768 of the 1,200 exam lines continue the previous line's conversation. The mean number of earlier turns available is 2.76 (max 14).
- 5.00% of positions have a gold content word that appears in an earlier turn but not in the prompt. Only 1.74% of positions are rare golds.
- The designed episodic vote H (section 1), scored on its own, got top-1 12.09%, gold in its top-3 15.58%, gold in its top-10 22.54%.
- Rare golds in H's top-3 are only **2.00%** of all positions. This counts the end-of-reply token as common; leaving it out of the common set inflated the number to 10.6%, which was an artefact that I caught. So the rare κ is at most 2.00% before even requiring the switch to be wrong, which is under the v16/v18 abandon line of 3%.
- The 531 referring positions use v9's 16-word list (remember.rs `is_referring`). H alone gets top-1 4.52% and top-10 21.47% there.
- **Prediction, stated in advance:** the episodic vote will not move next-word accuracy on this exam. Its real value is in talk (names and topics the user referred back to). The reliability weight makes this safe either way: β learned as 0 means no effect.

## 1. Episodic memory of the current conversation

**Data structure** (a new struct beside Cont; it touches no feature, code or readout):
```rust
#[derive(Clone, Default)]
struct Hippo { turns: Vec<Vec<String>> }   // oldest first; the newest turn has age 0
const HIP_DECAY: f32 = 0.5;  // per turn, fixed before any run
const HIP_CAP: usize = 12;   // a memory bound only (0.5^12 = 0.0002), not a behaviour rule
fn hear(&mut self, t: &[String])   // push; drop the oldest beyond the cap
fn clear(&mut self)
fn vote(&self, ctx: &[&str], spec: &dyn Fn(&str) -> f32, k: usize) -> Vec<(String, f32)>
```

**Written once per turn:**
- Exam: before the eval loop (line 791), build `ep_of[ei]` (a snapshot of the earlier turns only) from a new `prevs` vector.
  - If `prevs[ei]` equals test[ei−1].q, the chain continues.
  - Otherwise `clear()`, then `hear(prevs[ei])`. A chain break counts as an event boundary.
  - At read time: snapshot, then `hear(q)`. The prompt is age 0.
  - This cannot leak: r_i is q_{i+1}, and it is heard only on the next exchange.
- Talk: one session episode. `hear(user line)`, then reply, then `hear(her reply)`.
- Decay: a turn's weight is 0.5^age. Forgetting is graded; there is no cutoff.

**Read: exactly the candidates it can propose:**
- (P) **Priming:** every distinct token of every held turn, prompt included. Weight is 0.5^age × spec(w), where spec(w) = max(0, ln(n_learned / (1 + r0[w]))). r0 and `ro.n_learned` already exist.
  - Frequency grades content-ness, so no stop-list is needed.
  - A word the cortex has never said (for example a new name) gets maximum spec. This is the only route by which she can say a word she never learned.
- (I) **Induction / pattern completion:** wherever ctx's last word occurs in a held turn at position j, propose t[j+1], or EOS at the end of a turn. Weight is 0.5^age × (2 if ctx's second-to-last word also matches t[j−1], else 1).
  - BOS never matches, so position 0 gets priming only.
- Never proposed: BOS, her own reply-so-far (the cortex has it already; v18 measured it at 0.03%), and anything from other conversations (that is the cortex's job).
- H = 0.5·P/ΣP + 0.5·I/ΣI, giving full weight to one side if the other is empty. Take the top 10 and renormalise to sum 1.

**Voting without diluting the main vote.** H never enters feats() or codes. That is the v9 rule satisfied by construction, the same way the B3 table sits beside the code.
- M(w) is the switch's distribution per position: the cross-fitted cell's choice of `av` or `nv` from `mixv`, already normalised over each source's top 10.
- final(w) = (1−β)·M(w) + β·H(w), with β in {0, 0.1, …, 1}.
- β is chosen per (rung × margin) cell (the existing 40 cells) on the OTHER exchange-parity half. Ties go to β = 0, which is exactly the switch.
- This mirrors the v24 softmix block (lines 1106–1173) line for line.
- The generator uses β pooled over the whole exam per cell, as `both` is pooled now.

**Hooks:**
- Reader (lines 559–579): keep field0 tokenized in `prevs`, parallel to `pairs`, and split it like pairs (line 586).
- Before the eval loop: build `ep_of`.
- Inside `probe & 16` (lines 870–878): push `hipv` (H top-10), the referring flag, and φ per position.
- New block after softmix, `probe & 128`: κ, the β fit, the bootstrap, the referring rows, and the TSV.
- `step()` (line 1189) / `finish()` (line 1225) get `hip: Option<&Hippo>`. None is byte-identical to today.
  - With Some: ask for top-10 (`k.max(10)`), mix in H at β_cell, and report confidence as that cell's mixed hit rate.
- The reply exam (line 1287) adds a 6th row, "switch + hippocampus", using `ep_of[ei]` + q.
  - Strange prompts keep the same `ep_of[ei]`, so only the question differs.

## 2. Novelty / familiarity of the question

**Data structure** (inside Cont, beside B3):
```rust
bi: HashMap<u64, u32>   // fnv1a("x\x1Fy") of the boundary bigrams of every PROMPT heard
loo: Vec<f32>           // sorted leave-one-out familiarity of every training prompt
```
- **Count:** in `Stage::write()`, when pos == 0. That is once per exchange, in the write pass or in live mode. Gate it with a Stage flag so nothing prints differently.
- **F(q)** = mean of ln(1 + c) over q's boundary bigrams, using cortex counts only.
  - Familiarity is judged BEFORE the turn is stored. This is the comparator idea: compare the input with memory before encoding it.
- **φ(q)** = percentile of F(q) in `loo`, computed once after training.
- **No threshold anywhere.** φ is continuous.

**How it enters the self-model** (both reads pre-registered; primary marked):
- **(primary) Feeling of knowing:** her reply confidence becomes ĉ = (mean step confidence) × u(φ), where u(φ) = 1 − exp(−φ/0.1). φ0 = 0.1 is fixed before any run and not tuned.
  - The curve is smooth and saturating. At φ ≥ 0.25 (most normal prompts) u ≥ 0.92, so normal confidence is barely touched. At φ = 0.02 (the median mixed-up prompt) u = 0.18.
  - The human grounding is cue familiarity (Reder & Ritter 1992): people's sense of knowing comes from how familiar the question's terms are.
- **(secondary, report-only)** A rung × margin × φ-quintile table, cross-fitted, shrunk to its 2D parent cell with 4 pseudo-counts. This avoids today's (h+1)/(n+2) = 0.5 for empty cells.
  - It is added as a 4th row in the probe-4 SELF-MODEL block (lines 1010–1063).
  - I predict it undershoots, because a table calibrated on normal prompts extrapolates flat below its lowest bin.

> **WITHDRAWN 2026-10-14 (user):** "teaching ava english is like teaching a person english,
> you dont teach a kid made up rules". The hedge phrases below are words WE would put in her
> mouth, so they are dropped even though they are graded. The hard `sure < line` prefix is
> already deleted from compose.rs, and so is talk's `repair`. Familiarity ĉ is kept only as a READING
> in talk's bracket ("the question feels Y% familiar") and as an exam measurement (F1–F3).
> If she ever says "i'm not sure", it must be because she learned to.

**(withdrawn) Talk: graded uncertainty, replacing the hard `if sure < line` at lines 1443 and 1568:**
- d = ECDF position of ĉ among the confidences ĉ of 300 normal calibration replies.
- Three hedge levels, each a sigmoid "neuron": p_k = σ((d − c_k)/0.05), with c_k = 0.75, 0.88, 0.97.
  - 0.75 is the centre of the old cutoff, so average behaviour stays similar.
- One seeded xorshift draw per input (seed = fnv1a of the line, so runs are reproducible). The highest level whose p_k exceeds the draw speaks.
- Wording:
  - Level 1: "i think …"
  - Level 2: "i'm not sure, but …"
  - Level 3: "i don't really follow. my best guess: …" when u(φ) is the lower factor (the strange part is the question), otherwise "i don't know. my best guess: …".
- She ALWAYS gives her answer, and strong confidence makes p_k ≈ 0. The status line prints both signals: "~X% sure of the words; the question feels Y% familiar".

## 3. Sleep: replay into long-term memory

**What is saved now** (no persistent brain yet): the session log is the hippocampus's pending store.
- Talk appends one line per exchange to ../data/sleep/talk-<unix secs>.tsv, which must be gitignored like data/school. Each line is flushed at once, and logs stay local and are never published.
- Format: `@sleep \t q \t r \t prev \t who`. Tokens are joined by U+001F, so code string literals that contain spaces survive. The line is tokenizer-independent and exact.
- L1 (q = user turn, r = her reply, who = ava) is written after she replies.
- L2 (q = her turn, r = the user's next turn, who = you) is written when the user answers.

**Sleep = `AVA_SLEEP=<file>[;<file>…]`:**
- The reader appends these lines after the corpus, so they are always on the train side and never in the test block. The 4th field supplies prev (tagged lines carry no field0).
- who = you lines are learned through the normal settle/write path, in live mode or lab mode.
- who = ava lines go to Fam counting only (heard, not learned as speech targets), so every utterance is counted once.
- Why sleep after the day is equivalent to learning during it: the readout sums counts, and v17 measured an order effect of 0.00 on morphology. This is a real difference from the brain: Ava's cortex does not need interleaving to avoid interference, so for her sleep is purely transfer and persistence.
- The hippocampus is cleared after sleep. Tomorrow the day's phrases are familiar through the cortex counts.

**Must be saved once brain persistence lands** (store.rs today saves only synapses, allocator and readout):
- Cont: c0, c1, c2, b3, r0, plus bi and the loo ECDF. The ECDF cannot be recomputed without the prompts, so save it and insert slept prompts at each sleep.
- Self-model tables: margin edges, both/ncell, β per cell, and the calibration ECDF for the hedge.
- A tokenizer version tag.
- A **consolidation ledger** of (file, length, FNV-64 of the bytes). Replay must happen exactly once, because counts add and a double replay double-counts.
- The episode itself is not saved.

## 4. RAM at 30k (brain ~3.6 GB)

| Component | Size |
|---|---|
| Bigram table (68.5k entries, u64 keys) | ~2 MB (≤4 MB with String keys) |
| loo ECDF | 0.12 MB |
| Live episode | <10 KB |
| Exam `ep_of` snapshots | ~1 MB |
| Per-position H/φ buffers (11,999 positions) | ~5 MB |
| β and κ tables | <50 KB |
| Sleep log | disk only |

- Total is about 10 MB, roughly 0.3% of the brain.
- The school corpus, with its code bigrams, should stay under 40 MB.
- Time: about +15 s on the reply exam. F and H are negligible per position.

## 5. Pre-registered tests

Run: `AVA_LIVE=1 compose 30000 0 0 0 5 2 5 0 128 244`. Probe 244 = 4 + 16 + 32 + 64 (familiarity) + 128 (episodic).

| Test | Pass condition | Prediction / note |
|---|---|---|
| **R0** regression | Every pre-existing TSV line (depth1, life, native, native3, selfmodel, softmix, reply, adapt) is byte-identical to the probe-52 run. | Required before reading anything else. |
| **H1** ceiling, decided first | κ_H = gold in H's top-3 AND switch top-1 wrong, raw and rare (rare = outside top-200 ∪ {EOS}). Verdict on rare: <3.0% abandon the vote for this exam; 3–5% build at 15k; ≥5% take it to 30k. | Predicted: abandon (≤2.00%). |
| **H2** next word | Better only if switch+hippocampus > switch (52.75) AND the exchange-bootstrap 95% CI of the difference is above 0 (1,000 resamples, the existing seed). | |
| **H3** referring words (v9's 16-word list, 531 positions) | switch+hippocampus ≥ switch on the point estimate, so v9's failure (3.37 → 0.56) must not recur. Report ava, switch, hippocampus and the CI. | v9's numbers come from a different exam and are not comparable. |
| **H4** replies | Better only if word-F1 > switch greedy (39.60) AND proper ≥ 96.6%. Also report against ava greedy (40.52). | |
| **H5** talk probe (secondary) | 10 scripted pairs: "my name is <name never seen in training> ." then "what is my name ?". Pass: the name appears in ≥6 of 10 with the hippocampus and 0 of 10 without. | Likely to fail if exam-learned β ≈ 0. That would mean the reliability has to be learned in talk, not hand-set. |
| **F1** strange questions (primary) | AUC normal vs mixed-up of ĉ, mean over the reply, on the SAME switch-greedy words as v23: ≥ 0.85 (from 0.659). | Likely pass: raw φ is 0.965 on the exam. |
| **F2** guard | ĉ's AUC for "word-F1 ≥ 0.5" on normal prompts ≥ 0.94 (from 0.952). | Uncertain: u touches the ~10% least familiar normal prompts. |
| **F3** report | Raw φ AUC (the ceiling), reversed-prompt AUC, mixed-up share under the line (from 33.2%), and the 3D-table row. | |
| **S1** fidelity | 2,000 dialogue rows from filtered rows 32,400–34,399 (past the dev block). Run A: AVA_SLEEP in @sleep format. Run B: the same rows appended as plain text, n = 32,000. Both with **pool 5 explicitly** (the default doubles to 10M neurons at n > 30k, the old confound). Core TSV lines identical. Plus a `#[cfg(test)]` round trip from logger to reader. | |
| **S2** order invariance (v17 on dialogue) | Run C interleaves the same @sleep rows at seeded random positions after the test block. Run A vs run C: \|Δ\| ≤ 0.20 points on top-1, switch, reply F1, and day recall. | |
| **S3** no harm, and consolidation | Run A vs the no-sleep baseline: Δ ≥ −0.10 on all three metrics. Day recall (teacher-forced top-1 on the day's positions after sleep) ≥ 90%, printed beside the live guess rate on the same positions ("before sleep"). | |

## 6. Hard rules found and their graded replacements

1. Talk's `sure < line` cutoff → sigmoid hedge levels with a seeded draw. She always answers.
2. v24 §7's "read the hippocampus first" → a reliability-weighted mix per cell. Strong cortical evidence wins.
3. No "strange question → refuse or ask" rule. Doubt is graded and expressed, never blocking.
4. A priming stop-list → frequency-graded specificity.
5. Familiarity thresholds → a continuous φ and a smooth u(φ).
6. The episode cap of 12 and the reset at a conversation break are memory bounds and event structure, not behaviour. Forgetting is the graded 0.5^age.
7. Not replaying her own replies is a choice of teaching signal (it avoids self-reinforcement), not a behavioural ban. It is opt-in via `AVA_SLEEP_SELF=1`.
8. The exactly-once replay ledger is bookkeeping.
9. Out of scope, still a hard rule: v23's "repair" (rerank only if the reply is not proper).

All constants are fixed before any run: λ = 0.5, the 50/50 mix of P and I, top-10, the β grid with ties to 0, φ0 = 0.1, hedge centres 0.75/0.88/0.97 with width 0.05, the 2× bonus for a two-word match, and cap 12. The familiarity read was chosen on the dev block.

## 7. Line budget, in implementation order

1. Familiarity (bit 64): about 65 lines — Cont fields, counting in write(), LOO ECDF, F1–F3 rows, hedge.
2. Episodic vote (bit 128): about 110 lines — reader `prevs`, Hippo, `ep_of`, the mix/κ/bootstrap/referring block, the hip parameter in step and finish.
3. Sleep: about 60 lines — reader @sleep / AVA_SLEEP / who handling, talk logger, day-recall exam, round-trip test.

The total is about 235. The first cut, if needed, is the report-only 3D table (about 15 lines).

When this is built, run one job at a time with free RAM ≥ 8 GB. Free RAM was 9.6 GB during these checks.