# Contributing

Thank you for your interest in Ava.

## Code contributions

**Ava does not accept outside code contributions for now.** The author keeps full rights to the
code so that the licensing (source-available, with commercial licences granted by the author)
stays simple. Pull requests will be closed with thanks. This may change later.

## What is welcome

- **Forks for non-commercial use.** Fork her, change her, scale her up, publish your versions
  under the same licence. Keep the `Required Notice:` line from `LICENSE.md`.
- **Issues.** Bug reports, questions about the design or the method, results you got with her,
  and pointers to prior art. Please include the commit sha and your platform (OS, CPU, Rust
  version, thread count).
- **Commercial licences.** Open an issue to ask.

## Reporting a result honestly

If you train or test her and want to report a number, please follow the method she uses
herself (SCALING.md, section 7):

1. **Pre-register.** Before the run, write down the commit sha (or the sha256 of the build),
   the exact command and flags, the data, the seeds and the pass rule. Run once. Apply the rule
   by its letter.
2. **Give the seeds.** `--seed`, `--data-seed` and the thread count; a run is reproducible from
   them. If you can, run two or three seeds and report the spread.
3. **Give the sha.** The sha256 of the checkpoint and of the tokenizer file.
4. **Say what the test set has been used for before.** A set that was looked at during
   development is spent; say so.
5. **Right minus wrong.** Where abstention is allowed, report right minus wrong and the share
   right among the answers given, not only accuracy.
6. **Scaling needs three sizes.** A gain at one size is not a scaling result; report at least
   three sizes beside the yardstick transformer, with the same tokens-per-weight ratio.
7. **Statistics.** For paired comparisons, a paired bootstrap (20,000 resamples) and the 95%
   interval; a gain counts when the interval lies above zero.

Attach the `result.tsv` line (it holds the configuration, the weights, the score and the
timings) and, where relevant, the `log.tsv`.
