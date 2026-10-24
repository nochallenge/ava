> **Note (2026-10):** the text below is from the project's first days. Since then this crate
> has grown into Ava herself; there is no separate implementation. Start with the top-level
> [README](../README.md), and see [SCALING.md](../SCALING.md) to take her further.

# design-tests — TESTING AND DESIGN ONLY

**This is not ava. Nothing in here is the implementation.**

This crate exists so the ideas in ava's design notes could be run and measured
instead of assumed. It is scratch: written to be broken,
poked at, and thrown away.

The real neuron goes in `../src/nueron1.rs`, written by hand.

## What's in here

| file | what it is |
|---|---|
| `src/lib.rs` | LIF neuron, event queue, unit, the text→sparse-code splitter, 6 unit tests |
| `examples/sweep.rs` | experiment 1 — sweep the recurrent weight. finds the dead/runaway problem |
| `examples/fix.rs` | experiment 2 — one spike per unit. fixes runaway, does not fix interference |
| `examples/window.rs` | experiment 3 — derives and tests the analytic weight window |
| `examples/separation.rs` | experiment 4 — pattern separation. this is the one that works |
| `examples/tiebreak.rs` | experiment 5 — finds the index-bias bug in `top_k` |
| `examples/fixed_topk.rs` | validates the corrected `top_k_hashed` |

## Run it

```
cargo test
cargo run --example sweep
cargo run --example separation
```

`[workspace]` in `Cargo.toml` deliberately detaches this from the parent `ava` package
so it never gets pulled into a build of the real thing.

## Headline results

- pure all-to-all excitation has **no** working weight: too low and nothing recalls,
  too high and everything fires forever
- a usable weight needs `V_th/cue <= w < V_th/max_overlap`, which only exists when
  `max_overlap < cue`
- removing stopwords took max pairwise overlap from **9 to 1** and made recall clean
  at every cue size — separation beat every weight tweak combined
- a k-of-n code carries about `k / FANOUT` distinct features before it silently
  overflows (k=40, fanout=8 → 5 features)
- `top_k` tie-breaking by index biased every code toward low unit numbers
