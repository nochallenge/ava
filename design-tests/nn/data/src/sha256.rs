//! TESTING / DESIGN ONLY — sha256: SHA-256 (FIPS 180-4), std only, for naming data files by content.
//!
//! `Sha256::new()`, `update(bytes)` any number of times, then `hex()`. `sha256_hex(bytes)` and
//! `sha256_file(path)` are the one-shot forms. Used to prove that a regenerated file matches the
//! one on disk and to report what was written.

use std::io::{self, Read};
use std::path::Path;

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

const H0: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];

/// Streaming SHA-256.
#[derive(Clone)]
pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    n: usize,
    len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Sha256 {
        Sha256 { h: H0, buf: [0; 64], n: 0, len: 0 }
    }

    fn block(h: &mut [u32; 8], b: &[u8]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b_, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b_) ^ (a & c) ^ (b_ & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b_;
            b_ = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b_, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);
        if self.n > 0 {
            let take = (64 - self.n).min(data.len());
            self.buf[self.n..self.n + take].copy_from_slice(&data[..take]);
            self.n += take;
            data = &data[take..];
            if self.n == 64 {
                let buf = self.buf;
                Self::block(&mut self.h, &buf);
                self.n = 0;
            }
        }
        while data.len() >= 64 {
            Self::block(&mut self.h, &data[..64]);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.n = data.len();
        }
    }

    pub fn finish(mut self) -> [u8; 32] {
        let bits = self.len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.n != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        debug_assert_eq!(self.n, 0);
        let mut out = [0u8; 32];
        for (i, x) in self.h.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes());
        }
        out
    }

    /// The digest as 64 lowercase hex digits.
    pub fn hex(self) -> String {
        self.finish().iter().map(|b| format!("{b:02x}")).collect()
    }
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    h.hex()
}

/// The sha256 of a file, streamed in 1 MiB reads.
pub fn sha256_file(path: impl AsRef<Path>) -> io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
    }
    Ok(h.hex())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_vectors() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        let mut h = Sha256::new();
        let chunk = [b'a'; 1000];
        for _ in 0..1000 {
            h.update(&chunk);
        }
        assert_eq!(h.hex(), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    #[test]
    fn split_updates_match_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 + 3) as u8).collect();
        let want = sha256_hex(&data);
        for cut in [0usize, 1, 55, 56, 63, 64, 65, 127, 500, 999, 1000] {
            let mut h = Sha256::new();
            h.update(&data[..cut]);
            h.update(&data[cut..]);
            assert_eq!(h.hex(), want, "cut {cut}");
        }
    }
}
