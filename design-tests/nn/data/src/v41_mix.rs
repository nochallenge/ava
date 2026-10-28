//! TESTING / DESIGN ONLY — v41_mix: v41's reading data (100 PG-19 train books held out; usual 70 / books 25 / practice v3 5).
//!
//! HELD-OUT BOOKS. Of PG-19's 600 train books (name order), those with at least `HELDOUT_MIN_TOKENS` (16,385)
//! marked tokens are eligible; a seeded shuffle of the eligible list (`SplitMix64`, `derive_seed(seed, HOLD_LABEL,
//! 0)`) takes the first `HELDOUT_BOOKS` (100). They become v41's SEALED long-text test (one document per book, as
//! v40's books-test) and the books of v41's sealed find-the-fact test. v41 starts from v39, which never read PG-19.
//!
//! TRAINING BOOKS. The other 500, in a seeded order (`derive_seed(seed, ORDER_LABEL, 0)`), taken whole until their
//! tokens reach `BOOK_TARGET` (45M = 25% of about 180M).
//!
//! THE MIX. `v40_mix::build_mix_with` with shares [books 0.25, usual 0.70, practice 0.05]: every training book, usual
//! text read in order from v39's train shards until 70/25 of the book tokens, practice v3 docs in order until
//! 5/25 of them; shuffled by document (seed).

use crate::rng::{derive_seed, SplitMix64};

/// [books, usual, practice] shares (v40_mix's source order).
pub const SHARES: [f64; 3] = [0.25, 0.70, 0.05];
pub const DEFAULT_SEED: u64 = 41;
pub const HELDOUT_BOOKS: usize = 100;
pub const HELDOUT_MIN_TOKENS: usize = 16_385;
pub const BOOK_TARGET: u64 = 45_000_000;
const HOLD_LABEL: u64 = 0x7634_315f_686f_6c64;
const ORDER_LABEL: u64 = 0x7634_315f_6f72_6472;

/// Indices of the held-out books (ascending), given every book's marked token count in name order.
pub fn heldout(book_tokens: &[usize], n: usize, min_tokens: usize, seed: u64) -> Result<Vec<usize>, String> {
    let mut eligible: Vec<usize> = (0..book_tokens.len()).filter(|&i| book_tokens[i] >= min_tokens).collect();
    if eligible.len() < n {
        return Err(format!("only {} books have {min_tokens}+ tokens, {n} wanted", eligible.len()));
    }
    SplitMix64::new(derive_seed(seed, HOLD_LABEL, 0)).shuffle(&mut eligible);
    let mut out = eligible[..n].to_vec();
    out.sort_unstable();
    Ok(out)
}

/// Indices of the training books: the books not held out, in a seeded order, whole, until `target` tokens.
pub fn train_books(book_tokens: &[usize], held: &[usize], target: u64, seed: u64) -> Vec<usize> {
    let mut rest: Vec<usize> = (0..book_tokens.len()).filter(|i| held.binary_search(i).is_err()).collect();
    SplitMix64::new(derive_seed(seed, ORDER_LABEL, 0)).shuffle(&mut rest);
    let mut out = Vec::new();
    let mut sum = 0u64;
    for i in rest {
        if sum >= target {
            break;
        }
        sum += book_tokens[i] as u64;
        out.push(i);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heldout_draw_is_seeded_eligible_and_disjoint_from_training() {
        let mut r = SplitMix64::new(3);
        let toks: Vec<usize> = (0..600).map(|_| 500 + r.below(300_000) as usize).collect();
        let h = heldout(&toks, 100, 16_385, 41).unwrap();
        assert_eq!(h, heldout(&toks, 100, 16_385, 41).unwrap());
        assert_ne!(h, heldout(&toks, 100, 16_385, 42).unwrap());
        assert_eq!(h.len(), 100);
        assert!(h.windows(2).all(|w| w[0] < w[1]));
        assert!(h.iter().all(|&i| toks[i] >= 16_385));
        let t = train_books(&toks, &h, 20_000_000, 41);
        assert!(t.iter().all(|i| h.binary_search(i).is_err()));
        let sum: u64 = t.iter().map(|&i| toks[i] as u64).sum();
        assert!(sum >= 20_000_000 && sum - (toks[*t.last().unwrap()] as u64) < 20_000_000);
        let mut u = t.clone();
        u.sort();
        u.dedup();
        assert_eq!(u.len(), t.len());
        // all of them when the target is out of reach
        assert_eq!(train_books(&toks, &h, u64::MAX, 41).len(), 500);
        assert!(heldout(&toks[..50], 100, 16_385, 41).is_err());
    }

    #[test]
    fn shares_and_targets() {
        assert!((SHARES.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let t = crate::v40_mix::targets_with(BOOK_TARGET, SHARES);
        assert_eq!(t, [45_000_000, 126_000_000, 9_000_000]);
        // v40's own targets are unchanged
        assert_eq!(crate::v40_mix::targets(60_000_000), [60_000_000, 56_400_000, 3_600_000]);
    }
}
