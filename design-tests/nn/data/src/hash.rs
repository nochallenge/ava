//! TESTING / DESIGN ONLY — a small fast deterministic hasher for ava_data's hash maps.
//!
//! std's default SipHash is slow for the millions of short keys the BPE trainer and the
//! encoder cache use, and it is randomly seeded (so map iteration order changes per run).
//! This hasher folds input words with the Fx multiply-rotate step and finishes with the
//! splitmix64 finaliser, so every output bit depends on every input bit (the plain Fx hash
//! leaves the low bits depending only on the low input bits, which is fatal for keys like
//! `(left << 32) | right` because hash maps index buckets by the low bits).

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

const K: u64 = 0x517c_c1b7_2722_0a95;

#[derive(Default, Clone, Copy)]
pub struct FastHasher(u64);

impl FastHasher {
    #[inline]
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(K);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            // tag the tail with its length so "a\0" and "a" differ even without a length prefix
            self.add(u64::from_le_bytes(buf) ^ ((rest.len() as u64) << 59));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        crate::rng::mix64(self.0)
    }
}

pub type FastBuild = BuildHasherDefault<FastHasher>;
pub type FastMap<K, V> = HashMap<K, V, FastBuild>;
pub type FastSet<K> = HashSet<K, FastBuild>;

pub fn new_map<K, V>() -> FastMap<K, V> {
    HashMap::with_hasher(FastBuild::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, Hash};

    fn h<T: Hash + ?Sized>(x: &T) -> u64 {
        FastBuild::default().hash_one(x)
    }

    #[test]
    fn deterministic_and_distinguishes() {
        assert_eq!(h(b"hello".as_slice()), h(b"hello".as_slice()));
        assert_ne!(h(b"hello".as_slice()), h(b"hellp".as_slice()));
        assert_ne!(h(b"a".as_slice()), h(b"a\0".as_slice()));
        assert_ne!(h(&1u64), h(&2u64));
    }

    #[test]
    fn pair_keys_spread_over_low_bits() {
        // keys sharing the low 32 bits must not pile into the same low-bit bucket
        let mut buckets = vec![0u32; 1024];
        for left in 0..10_000u64 {
            let key = (left << 32) | 97;
            buckets[(h(&key) & 1023) as usize] += 1;
        }
        let max = *buckets.iter().max().unwrap();
        assert!(max < 40, "poor low-bit spread, max bucket {max}");
    }

    #[test]
    fn map_works_with_borrowed_slice_lookup() {
        let mut m: FastMap<Box<[u8]>, u64> = new_map();
        m.insert(b"abc".to_vec().into_boxed_slice(), 3);
        assert_eq!(m.get(b"abc".as_slice()), Some(&3));
        assert_eq!(m.get(b"abd".as_slice()), None);
    }
}
