//! TESTING / DESIGN ONLY — splitmix64, the one random number generator of ava_data.
//!
//! Every shuffle and every split in this crate goes through `SplitMix64`, so a run is fully
//! reproducible from its seed. `mix64(x)` is the one-shot form: the first output of a
//! generator seeded with `x` (used as a hash, e.g. for the train/val split rule).

/// splitmix64 (Steele, Lea, Flood 2014; the reference constants of `SplittableRandom`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitMix64(pub u64);

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

#[inline]
fn finalize(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl SplitMix64 {
    pub fn new(seed: u64) -> SplitMix64 {
        SplitMix64(seed)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GAMMA);
        finalize(self.0)
    }

    /// Uniform in `0..n` without modulo bias (rejects the lowest `2^64 mod n` outputs).
    /// Panics if `n == 0`.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "SplitMix64::below(0)");
        let threshold = n.wrapping_neg() % n; // == 2^64 mod n
        loop {
            let r = self.next_u64();
            if r >= threshold {
                return r % n;
            }
        }
    }

    /// Fisher–Yates shuffle (from the back; `below(i + 1)` picks the swap partner).
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }
}

/// One-shot splitmix64: the first output of `SplitMix64::new(x)`.
#[inline]
pub fn mix64(x: u64) -> u64 {
    finalize(x.wrapping_add(GAMMA))
}

/// Derive an independent stream seed from a base seed and two labels:
/// `mix64(mix64(seed ^ mix64(a)) ^ b)`.
pub fn derive_seed(seed: u64, a: u64, b: u64) -> u64 {
    mix64(mix64(seed ^ mix64(a)) ^ b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_vector_seed_zero() {
        // splitmix64 seeded with 0: the published reference outputs.
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(r.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn mix64_is_first_output() {
        for s in [0u64, 1, 42, u64::MAX] {
            assert_eq!(mix64(s), SplitMix64::new(s).next_u64());
        }
    }

    #[test]
    fn deterministic() {
        let a: Vec<u64> = {
            let mut r = SplitMix64::new(7);
            (0..10).map(|_| r.next_u64()).collect()
        };
        let b: Vec<u64> = {
            let mut r = SplitMix64::new(7);
            (0..10).map(|_| r.next_u64()).collect()
        };
        assert_eq!(a, b);
        let mut c = SplitMix64::new(8);
        assert_ne!(a[0], c.next_u64());
    }

    #[test]
    fn below_in_range_and_covers() {
        let mut r = SplitMix64::new(3);
        assert_eq!(r.below(1), 0);
        let mut seen = [0usize; 7];
        for _ in 0..7000 {
            let x = r.below(7) as usize;
            assert!(x < 7);
            seen[x] += 1;
        }
        for &s in &seen {
            assert!(s > 800 && s < 1200, "below(7) badly skewed: {seen:?}");
        }
        // huge n near u64::MAX still in range
        for _ in 0..100 {
            assert!(r.below(u64::MAX - 3) < u64::MAX - 3);
        }
    }

    #[test]
    #[should_panic]
    fn below_zero_panics() {
        SplitMix64::new(1).below(0);
    }

    #[test]
    fn shuffle_is_a_permutation_and_deterministic() {
        let mut v: Vec<u32> = (0..1000).collect();
        SplitMix64::new(11).shuffle(&mut v);
        let mut w: Vec<u32> = (0..1000).collect();
        SplitMix64::new(11).shuffle(&mut w);
        assert_eq!(v, w);
        assert_ne!(v, (0..1000).collect::<Vec<u32>>());
        let mut s = v.clone();
        s.sort();
        assert_eq!(s, (0..1000).collect::<Vec<u32>>());
        // empty and single are fine
        let mut e: Vec<u8> = vec![];
        SplitMix64::new(1).shuffle(&mut e);
        let mut one = vec![5];
        SplitMix64::new(1).shuffle(&mut one);
        assert_eq!(one, vec![5]);
    }

    #[test]
    fn derive_seed_separates_labels() {
        let s = 1234;
        assert_ne!(derive_seed(s, 0, 0), derive_seed(s, 0, 1));
        assert_ne!(derive_seed(s, 0, 1), derive_seed(s, 1, 0));
        assert_eq!(derive_seed(s, 5, 9), derive_seed(s, 5, 9));
    }
}
